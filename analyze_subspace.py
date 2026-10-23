#!/usr/bin/env python3
"""Subspace re-analysis for contrasts that are not rank one (Phase 4).

Phase 4 reports a participation ratio of ~14.7 for refusal on Qwen2.5-3B and
~6.2 for truth_cities. A contrast spread over that many effective dimensions is
not "a direction", and the scalar Gini of its mean vector describes one
arbitrary vector inside the subspace rather than where the subspace sits.

Energy is normalised by rank, so the expected Gini of a Haar-random subspace is
essentially rank-invariant. That is what lets a rank-15 refusal subspace and a
rank-2 sycophancy subspace be compared on one scale.

The 'gap' column (random-subspace null) is the result. 'shuffle gap' is a
diagnostic only: permuting coordinates destroys activation anisotropy along
with concept structure, so it over-rejects and flags pure anisotropic noise.
The proper label-shuffle needs per-class activations (acts_*.pt), which were
deleted to reclaim disk.

NOTE: the per-pair sign flip used for scalar directions is DEGENERATE here --
flipping a row's sign leaves the row space unchanged, so it returns gap 0.0000
and p 1.0000 every time.
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from linear_rep_geometry.behavioral import load_concept, POOLING
from linear_rep_geometry.dimensionality import analyze_rank
from linear_rep_geometry.spectral_metrics import (
    eigendecompose, compute_cumulative_variance, subspace_metrics,
    one_sample_test, paired_test, tost_equivalence, SESOI)


def concept_subspace(D, rank):
    _, _, Vh = torch.linalg.svd(D.float(), full_matrices=False)
    return Vh[: max(1, min(rank, Vh.shape[0]))].clone()


def random_subspace_null(rank, d, evecs, cumvar, n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples)
    for i in range(n_samples):
        A = torch.tensor(rng.standard_normal((d, rank)), dtype=torch.float32)
        out[i] = subspace_metrics(torch.linalg.qr(A)[0].T, evecs, cumvar)["gini"]
    return out


def column_shuffle_null(D, rank, evecs, cumvar, n_perm=100, seed=0):
    rng = np.random.default_rng(seed)
    X = D.float().numpy(); n, d = X.shape
    out = np.empty(n_perm)
    for i in range(n_perm):
        Z = np.empty_like(X)
        for j in range(d):
            Z[:, j] = X[rng.permutation(n), j]
        out[i] = subspace_metrics(concept_subspace(torch.tensor(Z), rank),
                                  evecs, cumvar)["gini"]
    return out


def subspace_test(D, rank, evecs, cumvar, n_perm=300, seed=0):
    obs = subspace_metrics(concept_subspace(D, rank), evecs, cumvar)["gini"]
    rnd = random_subspace_null(rank, D.shape[1], evecs, cumvar, n_perm, seed)
    shuf = column_shuffle_null(D, rank, evecs, cumvar, min(n_perm, 100), seed)
    def pval(null):
        c = null.mean()
        return (int((np.abs(null - c) >= abs(obs - c)).sum()) + 1) / (len(null) + 1)
    return {"gini": obs, "null_mean": float(rnd.mean()),
            "gap": float(obs - rnd.mean()), "p": pval(rnd),
            "shuffle_gap": float(obs - shuf.mean()), "p_shuffle": pval(shuf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--basis", default="full_vocab")
    ap.add_argument("--pooling", default="last_token", choices=list(POOLING))
    ap.add_argument("--var-threshold", type=float, default=0.50)
    ap.add_argument("--max-rank", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/_analysis/subspace.json")
    args = ap.parse_args()

    root = Path(args.results_root); rows = defaultdict(dict)
    for model in args.models:
        cov_p = root / model / "cov" / f"cov_{args.basis}.pt"
        if not cov_p.exists():
            print(f"  ! {model}: no {cov_p.name}"); continue
        cov = torch.load(cov_p, map_location="cpu", weights_only=True).float()
        evals, evecs = eigendecompose(cov, cache_path=root / model / "cov" / f"eig_{args.basis}.pt")
        cumvar = compute_cumulative_variance(evals)
        cdir = root / model / "concepts"
        names = args.concepts or sorted(
            p.stem.replace("concept_", "") for p in cdir.glob("concept_*.pt")
            if not p.stem.startswith("concept_word_"))
        print(f"\n{model}")
        for concept in names:
            p = cdir / f"concept_{concept}.pt"
            if not p.exists(): continue
            c = load_concept(p); layer = c["n_layers"] // 2
            D = c["diffs"].get((args.pooling, layer))
            if D is None: continue
            r = analyze_rank(D, var_threshold=args.var_threshold, max_rank=args.max_rank)
            rank = int(min(r["rank_for_threshold"], args.max_rank))
            res = subspace_test(D, rank, evecs, cumvar, args.n_perm, args.seed)
            res.update({"rank": rank, "pc1": r["pc1_var_explained"],
                        "pr": r["participation_ratio"], "layer": int(layer)})
            rows[concept][model] = res
            flag = "" if res["p"] < 0.05 else "   (n.s.)"
            print(f"  {concept:24s} rank {rank:2d} (PR {r['participation_ratio']:5.2f})  "
                  f"Gini {res['gini']:+.4f}  gap {res['gap']:+.4f} p={res['p']:.4f}{flag}"
                  f"  |  shuffle gap {res['shuffle_gap']:+.4f}")

    print(f"\n{'concept':26s} {'n':>2s} {'rank':>5s} {'mean gap':>9s} "
          f"{'95% CI':>18s} {'p':>8s}  verdict")
    print("-" * 86)
    summary = {}
    for concept in sorted(rows):
        per = rows[concept]
        gaps = [per[m]["gap"] for m in sorted(per)]
        ranks = [per[m]["rank"] for m in sorted(per)]
        t = one_sample_test(gaps)
        entry = {"n_models": t["n"], "mean_rank": float(np.mean(ranks)),
                 "mean_gap": t["mean"], "test": t, "per_model": per}
        if t.get("p_t") is not None and t["p_t"] < 0.05:
            verdict = "anti-concentrates" if t["mean"] < 0 else "CONCENTRATES"
        else:
            eq = tost_equivalence(gaps, sesoi=SESOI); entry["tost"] = eq
            verdict = "equivalent to zero" if eq.get("equivalent") else "inconclusive"
        entry["verdict"] = verdict; summary[concept] = entry
        ci = t.get("ci95", [float("nan")] * 2)
        print(f"{concept:26s} {t['n']:2d} {np.mean(ranks):5.1f} {t['mean']:+9.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] {t.get('p_t', float('nan')):8.4f}  {verdict}")

    truth = [c for c in rows if c.startswith("truth_")]
    refus = [c for c in rows if c.startswith("refusal")]
    if truth and refus:
        shared = sorted(set.intersection(*[set(rows[c]) for c in truth + refus]))
        if len(shared) >= 2:
            rv = [float(np.mean([rows[c][m]["gap"] for c in refus])) for m in shared]
            tv = [float(np.mean([rows[c][m]["gap"] for c in truth])) for m in shared]
            pt = paired_test(rv, tv, "refusal", "truth")
            summary["_contrast_refusal_vs_truth"] = {"models": shared, **pt}
            print(f"\nDesigned contrast (subspace) over {len(shared)} models")
            print(f"  refusal {pt['mean_a']:+.4f}   truth {pt['mean_b']:+.4f}   "
                  f"diff {pt['mean_diff']:+.4f}   p={pt.get('p_t', float('nan')):.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"basis": args.basis, "pooling": args.pooling,
                   "n_perm": args.n_perm, "concepts": summary}, f, indent=2, default=float)
    print(f"\nwrote {out}")
    print("\n'gap' (random-subspace null) is the result. 'shuffle gap' is a "
          "diagnostic that\nover-rejects -- do not report it as a significance test.")


if __name__ == "__main__":
    main()
