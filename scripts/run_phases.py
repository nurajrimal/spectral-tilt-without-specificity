#!/usr/bin/env python3
"""
Run the thesis phases.

    python scripts/run_phases.py 0 --model Qwen/Qwen2.5-0.5B      # reproduce baseline
    python scripts/run_phases.py 2 --model Qwen/Qwen2.5-0.5B --freeze
    python scripts/run_phases.py 1 --model Qwen/Qwen2.5-0.5B --concepts truth_cities refusal
    python scripts/run_phases.py 3 4 5 6 --model Qwen/Qwen2.5-0.5B

Phases, following section 4 of the accepted proposal:

    0  reproduce the published -0.356 on Qwen2.5-0.5B; build covariance caches
    1  extract behavioural directions, both poolings, every layer
    2  build both nulls; freeze the decision rule
    3  layer sweep: behavioural vs word-level trajectories
    4  rank test on per-example differences
    5  the three spectral bases
    6  dual geometry for refusal

Run 2 with --freeze BEFORE reading phase 1 output. That ordering is the
methodological contribution; the code cannot enforce it, but freeze_decision_rule
refuses to overwrite, so the commit timestamp is checkable afterwards.

Results land under results/<model>/ and results/_analysis/<model>/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import linear_rep_geometry as lrg                                   # noqa: E402
from linear_rep_geometry import covariance as cov_mod               # noqa: E402
from linear_rep_geometry import behavioral as beh                   # noqa: E402
from linear_rep_geometry import nulls as nullmod                    # noqa: E402
from linear_rep_geometry import dimensionality as dim_mod           # noqa: E402
from linear_rep_geometry import dual_geometry as dual               # noqa: E402
from linear_rep_geometry import word_level as wlvl                   # noqa: E402
from linear_rep_geometry.spectral_metrics import (                  # noqa: E402
    eigendecompose, compute_cumulative_variance, random_direction_baseline,
    metrics, subspace_metrics,
)

BASES = ("full_vocab", "language", "activation")


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_model(name, device):
    print(f"loading {name} ...")
    model, tok, handler = lrg.load_model(
        name, device_map="auto" if device == "cuda" else None)
    model.eval()
    return model, tok, handler


def get_basis(basis, model, tok, handler, cache_dir, args):
    """Build or load one covariance basis. Cached: these are deterministic."""
    path = Path(cache_dir) / f"cov_{basis}.pt"

    def build():
        if basis == "full_vocab":
            return cov_mod.unembedding_covariance(model, handler, None, args.lam)
        if basis == "language":
            ids = None
            p = Path(args.language_tokens) if args.language_tokens else None
            if p and p.exists():
                ids = json.load(open(p))
            if ids is None:
                print("    no language token subset supplied; "
                      "falling back to full vocabulary")
            return cov_mod.unembedding_covariance(model, handler, ids, args.lam)
        if basis == "activation":
            texts = []
            src = Path(args.activation_corpus) if args.activation_corpus else None
            if src and src.exists():
                texts = [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
            else:
                for f in sorted(Path(args.data_dir).glob("truth_*.jsonl")):
                    for line in open(f, encoding="utf-8"):
                        r = json.loads(line)
                        texts += [r["positive"], r["negative"]]
                    if len(texts) > 4000:
                        break
            if not texts:
                raise SystemExit("activation basis needs a corpus; "
                                 "pass --activation-corpus or build datasets first")
            return cov_mod.activation_covariance(
                texts[:4000], model, tok, handler, layer=args.activation_layer,
                device=args.device, batch_size=args.batch_size, lam=args.lam)
        raise ValueError(basis)

    return cov_mod.load_or_build(path, build, force=args.force_cov)


def spectrum(cov, cache_path):
    evals, evecs = eigendecompose(cov, cache_path=cache_path)
    return evals, evecs, compute_cumulative_variance(evals)


# ─── Phase 0 ─────────────────────────────────────────────────────────────────

def phase0(args, model, tok, handler, outdir):
    """
    Reproduce the published baseline and build the caches the repo omits.

    The target is Gini -0.356 for Qwen2.5-0.5B. Matching it validates the
    environment, the GPU and my reading of the code before anything new is
    built. It requires the published word-level concept vectors; without them
    this reports the spectrum only and the check is deferred.
    """
    print("\n=== Phase 0: reproduce published baseline ===")
    res = {"model": args.model}

    for basis in ("full_vocab", "language"):
        cov = get_basis(basis, model, tok, handler, outdir / "cov", args)
        evals, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{basis}.pt")
        d = cov.shape[0]
        top10 = float(cumvar[max(0, int(0.1 * d) - 1)])
        rnd = random_direction_baseline(d, evecs, cumvar,
                                        n_samples=args.n_random, seed=args.seed)
        res[basis] = {"d": d,
                      "cond": float(evals.max() / evals.min().clamp(min=1e-12)),
                      "top10pct_variance": top10,
                      "random_gini_mean": rnd["gini_mean"],
                      "random_gini_std": rnd["gini_std"]}
        print(f"  {basis:11s} d={d} cond={res[basis]['cond']:.2e} "
              f"top10%={top10:.1%} random Gini={rnd['gini_mean']:+.4f}")

    wp = Path(args.word_pairs)
    if not wp.exists():
        res["word_level"] = None
        print(f"  no word-pair directory at {wp}; the §6 internal control cannot "
              f"be built.\n    Point --word-pairs at the published "
              f"data/word_pairs/ directory.")
    else:
        print(f"  extracting word-level control from {wp}")
        ex = wlvl.extract_all(wp, model, tok, handler, device=args.device,
                            batch_size=max(8, args.batch_size * 2),
                            concepts=args.word_concepts,
                            out_dir=outdir / "concepts",
                            save_acts=not args.no_save_acts)
        print(f"    {ex['n_concepts']} word-level concepts extracted")

        cov = get_basis("language", model, tok, handler, outdir / "cov", args)
        _, evecs, cumvar = spectrum(cov, outdir / "cov" / "eig_language.pt")
        chk = wlvl.published_baseline_check(outdir / "concepts", evecs, cumvar,
                                          expected=args.expected_gini)
        res["word_level"] = {"n_concepts": ex["n_concepts"],
                             "verification": ex["verification"],
                             "baseline_check": {k: v for k, v in chk.items()
                                                if k != "per_concept"}}
        if chk.get("mean_gini") is not None:
            flag = "MATCH" if chk["ok"] else "MISS"
            print(f"  word-level Gini {chk['mean_gini']:+.4f} over "
                  f"{chk['n_concepts']} concepts vs published "
                  f"{chk['expected']:+.3f}  [{flag}]")
            if not chk["ok"]:
                print(f"    delta {chk['delta']:+.4f}. {chk['note']}")

    jdump(res, outdir / "phase0.json")
    return res


# ─── Phase 1 ─────────────────────────────────────────────────────────────────

def phase1(args, model, tok, handler, outdir):
    """Extract behavioural directions: both poolings, every layer."""
    print("\n=== Phase 1: extract behavioural directions ===")
    data = Path(args.data_dir)
    files = ([data / f"{c}.jsonl" for c in args.concepts] if args.concepts
             else sorted(data.glob("*.jsonl")))
    files = [f for f in files if f.exists() and f.name != "provenance.json"]
    if not files:
        raise SystemExit(f"no datasets in {data}; run scripts/build_datasets.py")

    summary = {}
    for f in files:
        concept = f.stem
        pairs = beh.load_pairs(f)
        if args.max_pairs and len(pairs) > args.max_pairs:
            import random
            pairs = random.Random(args.seed).sample(pairs, args.max_pairs)

        print(f"  [{concept}] {len(pairs)} pairs "
              f"({'matched' if pairs[0].matched else 'unmatched'})")
        res = beh.extract_concept(pairs, model, tok, handler, device=args.device,
                                  batch_size=args.batch_size,
                                  max_length=args.max_length)
        beh.save_concept(res, outdir / "concepts", concept,
                         save_acts=not args.no_save_acts)
        summary[concept] = {"n_pairs": res["n_pairs"], "n_layers": res["n_layers"],
                            "matched": res["matched"], "d": res["d"]}
        print(f"    {len(res['directions'])} directions over "
              f"{res['n_layers']} layers x {len(beh.POOLING)} poolings")

    n_word = len(list((outdir / "concepts").glob("concept_word_*.pt")))
    if n_word == 0:
        print("  note: no word-level control present. Run Phase 0 to build it; "
              "§6 requires\n    it for the paired behavioural-vs-word test.")
    else:
        print(f"  word-level control present: {n_word} concepts")

    jdump(summary, outdir / "phase1.json")
    return summary


# ─── Phase 2 ─────────────────────────────────────────────────────────────────

def phase2(args, model, tok, handler, outdir):
    """Build both nulls per concept; optionally freeze the decision rule."""
    print("\n=== Phase 2: nulls ===")
    cov = get_basis(args.basis, model, tok, handler, outdir / "cov", args)
    _, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{args.basis}.pt")
    d = cov.shape[0]

    rnd = random_direction_baseline(d, evecs, cumvar,
                                    n_samples=args.n_random, seed=args.seed)
    print(f"  random null: Gini {rnd['gini_mean']:+.4f} +/- {rnd['gini_std']:.4f}")

    out = {"basis": args.basis, "d": d,
           "random_null": {k: v for k, v in rnd.items()
                           if not isinstance(v, np.ndarray)},
           "concepts": {}}

    for p in sorted((outdir / "concepts").glob("acts_*.pt")):
        concept = p.stem.replace("acts_", "")
        acts = torch.load(p, map_location="cpu", weights_only=False)
        cp = beh.load_concept(outdir / "concepts" / f"concept_{concept}.pt")
        entry = {}
        for pooling in beh.POOLING:
            A, B = acts[pooling]["pos"], acts[pooling]["neg"]
            # Word and behavioural concepts become distinguishable at
            # DIFFERENT depths, so one layer for both is wrong.
            #
            # Measured on Qwen2.5-0.5B (diagnose_word.py): the 42 word
            # directions have mean pairwise |cos| 0.97 at mid-depth -- they are
            # one direction wearing 42 labels, and their Gini (+0.035) measures
            # the shared component. At the FINAL layer |cos| falls to 0.26 and
            # the Gini is -0.397, matching Phase 0 and the published -0.356.
            # Context-free single words carry no context for mid-layers to
            # build on; the lexical content only separates where the
            # unembedding reads it.
            #
            # Behavioural concepts are the opposite: |cos| 0.13 at mid-depth,
            # genuinely distinct. They are statements, so mid-depth is right.
            if args.layer is not None:
                layer = args.layer
            elif concept.startswith("word_"):
                layer = A.shape[1] - 1           # final layer
            else:
                layer = A.shape[1] // 2          # mid-depth
            xp, xn = A[:, layer, :], B[:, layer, :]

            rel = nullmod.split_half_reliability(xp, xn, matched=cp["matched"],
                                                 seed=args.seed)
            nl = nullmod.both_nulls(xp, xn, evecs, cumvar, n_perm=args.n_perm,
                                    matched=cp["matched"], seed=args.seed,
                                    random_cache=rnd)
            entry[pooling] = {"layer": int(layer), "split_half": rel, **nl}
            flag = "" if rel["cosine_mean"] >= 0.70 else "  <-- below gate"
            print(f"  [{concept}:{pooling}] L{layer} "
                  f"Gini {nl['observed_gini']:+.4f} | "
                  f"shuffle gap {nl['shuffle_null']['gap']:+.4f} "
                  f"p={nl['shuffle_null']['p_two_sided']:.4f} | "
                  f"random gap {nl['random_null']['gap']:+.4f} | "
                  f"r_sh={rel['cosine_mean']:.2f}{flag}")
            if not nl["agree"]:
                print("      nulls disagree — report both, do not pick one")
        out["concepts"][concept] = entry

    jdump(out, outdir / f"phase2_{args.basis}.json")

    if args.freeze:
        nullmod.freeze_decision_rule(outdir / "decision_rule.json",
                                     nullmod.DEFAULT_RULE)
    return out


# ─── Phase 3 ─────────────────────────────────────────────────────────────────

def phase3(args, model, tok, handler, outdir):
    """Layer trajectories, not points. Behavioural vs word-level in one basis."""
    print("\n=== Phase 3: layer sweep ===")
    cov = get_basis(args.basis, model, tok, handler, outdir / "cov", args)
    _, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{args.basis}.pt")
    rnd = random_direction_baseline(cov.shape[0], evecs, cumvar,
                                    n_samples=args.n_random, seed=args.seed)

    out = {"basis": args.basis, "random_gini_mean": rnd["gini_mean"], "concepts": {}}
    for p in sorted((outdir / "concepts").glob("concept_*.pt")):
        concept = p.stem.replace("concept_", "")
        cp = beh.load_concept(p)
        traj = {}
        for pooling in beh.POOLING:
            rows = []
            for layer in range(cp["n_layers"]):
                v = cp["directions"].get((pooling, layer))
                if v is None:
                    continue
                m = metrics(v, evecs, cumvar)
                rows.append({"layer": layer, "gini": m["gini"], "scm": m["scm"],
                             "gap_vs_random": float(m["gini"] - rnd["gini_mean"])})
            traj[pooling] = rows
        out["concepts"][concept] = traj

        gl = [r["gini"] for r in traj["last_token"]]
        gm = [r["gini"] for r in traj["mean_over_sequence"]]
        corr = float(np.corrcoef(gl, gm)[0, 1]) if len(gl) > 2 else float("nan")
        print(f"  [{concept}] last_token {min(gl):+.3f}..{max(gl):+.3f} | "
              f"mean_seq {min(gm):+.3f}..{max(gm):+.3f} | pooling r={corr:.2f}")
        if corr < 0.5:
            print("      poolings disagree across depth — "
                  "no conclusion may be reported for this concept")

    jdump(out, outdir / f"phase3_{args.basis}.json")
    return out


# ─── Phase 4 ─────────────────────────────────────────────────────────────────

def phase4(args, model, tok, handler, outdir):
    """Rank test. Runs before any headline number is interpreted."""
    print("\n=== Phase 4: dimensionality ===")
    cov = get_basis(args.basis, model, tok, handler, outdir / "cov", args)
    _, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{args.basis}.pt")

    out = {"basis": args.basis, "concepts": {}}
    for p in sorted((outdir / "concepts").glob("concept_*.pt")):
        concept = p.stem.replace("concept_", "")
        cp = beh.load_concept(p)
        rep = dim_mod.rank_report(cp["diffs"], var_threshold=args.var_threshold)
        s = rep["summary"]
        print(f"  [{concept}] PC1 {s['pc1_var_mean']:.1%} | "
              f"PR {s['participation_ratio_mean']:.2f} | "
              f"rank-one in {s['frac_cells_rank_one']:.0%} of cells")

        if not s["treat_as_rank_one"]:
            layer = args.layer if args.layer is not None else cp["n_layers"] // 2
            D = cp["diffs"][("last_token", layer)]
            r = dim_mod.analyze_rank(D, var_threshold=args.var_threshold)
            basis_sub = dim_mod.concept_subspace(D, r["rank_for_threshold"])
            sm = subspace_metrics(basis_sub, evecs, cumvar)
            rep["subspace"] = {"layer": int(layer), "rank": sm["rank"],
                               "gini": sm["gini"], "scm": sm["scm"]}
            print(f"      not rank one -> subspace rank {sm['rank']} "
                  f"Gini {sm['gini']:+.4f}  (the scalar Gini is the wrong "
                  f"statistic for this concept)")
        out["concepts"][concept] = rep

    jdump(out, outdir / f"phase4_{args.basis}.json")
    return out


# ─── Phase 5 ─────────────────────────────────────────────────────────────────

def phase5(args, model, tok, handler, outdir):
    """Same directions, three bases. Which conclusions are basis-dependent?"""
    print("\n=== Phase 5: spectral basis ===")
    out = {"bases": {}}
    for basis in BASES:
        try:
            cov = get_basis(basis, model, tok, handler, outdir / "cov", args)
        except Exception as e:
            print(f"  {basis}: unavailable ({e})")
            continue
        _, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{basis}.pt")
        rnd = random_direction_baseline(cov.shape[0], evecs, cumvar,
                                        n_samples=args.n_random, seed=args.seed)
        per = {}
        for p in sorted((outdir / "concepts").glob("concept_*.pt")):
            concept = p.stem.replace("concept_", "")
            cp = beh.load_concept(p)
            layer = args.layer if args.layer is not None else cp["n_layers"] // 2
            cell = {}
            for pooling in beh.POOLING:
                v = cp["directions"].get((pooling, layer))
                if v is None:
                    continue
                m = metrics(v, evecs, cumvar)
                cell[pooling] = {"gini": m["gini"], "scm": m["scm"],
                                 "gap_vs_random": float(m["gini"] - rnd["gini_mean"])}
            per[concept] = {"layer": int(layer), **cell}
        out["bases"][basis] = {"d": cov.shape[0],
                               "random_gini_mean": rnd["gini_mean"],
                               "concepts": per}
        print(f"  {basis}: random Gini {rnd['gini_mean']:+.4f}, "
              f"{len(per)} concepts measured")

    # flag sign changes across bases
    flips = []
    names = list(out["bases"])
    if len(names) > 1:
        for concept in out["bases"][names[0]]["concepts"]:
            signs = set()
            for b in names:
                c = out["bases"][b]["concepts"].get(concept, {})
                g = c.get("last_token", {}).get("gap_vs_random")
                if g is not None:
                    signs.add(g > 0)
            if len(signs) > 1:
                flips.append(concept)
    out["sign_flips_across_bases"] = flips
    if flips:
        print(f"  basis-dependent sign: {', '.join(flips)}")

    jdump(out, outdir / "phase5.json")
    return out


# ─── Phase 6 ─────────────────────────────────────────────────────────────────

def phase6(args, model, tok, handler, outdir):
    """Dual geometry, refusal only."""
    print("\n=== Phase 6: dual geometry (refusal) ===")
    cov = get_basis(args.basis, model, tok, handler, outdir / "cov", args)
    _, evecs, cumvar = spectrum(cov, outdir / "cov" / f"eig_{args.basis}.pt")

    cp_path = outdir / "concepts" / "concept_refusal.pt"
    if not cp_path.exists():
        raise SystemExit("Phase 6 needs the refusal direction; run Phase 1 first")
    cp = beh.load_concept(cp_path)
    layer = args.layer if args.layer is not None else cp["n_layers"] // 2
    act_gini = metrics(cp["directions"][("last_token", layer)], evecs, cumvar)["gini"]

    res = dual.dual_geometry_test(model, tok, handler, act_gini, evecs, cumvar,
                                  n_samples=args.n_random, seed=args.seed)
    for variant, r in res["variants"].items():
        if r.get("ok"):
            print(f"  {variant:14s} vocab Gini {r['vocab_gini']:+.4f} vs null "
                  f"{r['vocab_null_gini']:+.4f} -> gap {r['vocab_gap']:+.4f} "
                  f"(p={r['p_two_sided']:.4f})")
        else:
            print(f"  {variant:14s} unusable: {r.get('reason')}")
    v = res["verdict"]
    print(f"  activation Gini {act_gini:+.4f} (layer {layer})")
    print(f"  {v.get('note', '')}")

    res["layer"] = int(layer)
    res["basis"] = args.basis
    jdump(res, outdir / f"phase6_{args.basis}.json")
    return res


PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3,
          4: phase4, 5: phase5, 6: phase6}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", type=int, choices=sorted(PHASES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-dir", default="data/statements")
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--basis", default="full_vocab", choices=BASES)
    ap.add_argument("--layer", type=int, default=None,
                    help="default: mid-depth; Phase 3 sweeps all layers regardless")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-pairs", type=int, default=512,
                    help="cap per concept; counterfact alone has ~15k")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--var-threshold", type=float, default=0.50)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--activation-layer", type=int, default=-1)
    ap.add_argument("--activation-corpus", default=None)
    ap.add_argument("--language-tokens", default=None,
                    help="JSON list of token ids for the published language subset")
    ap.add_argument("--word-pairs", default="data/word_pairs",
                    help="published word-pair directory; builds the §6 internal control")
    ap.add_argument("--word-concepts", nargs="+", default=None,
                    help="subset of word-level concepts (default: all 42)")
    ap.add_argument("--expected-gini", type=float, default=-0.356,
                    help="Phase 0 reproduction target (Qwen2.5-0.5B)")
    ap.add_argument("--no-save-acts", action="store_true")
    ap.add_argument("--force-cov", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="Phase 2: freeze the decision rule (write-once)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable; using cpu")
        args.device = "cpu"

    outdir = Path(args.results_root) / Path(args.model).name
    outdir.mkdir(parents=True, exist_ok=True)

    model, tok, handler = load_model(args.model, args.device)
    for ph in args.phases:
        PHASES[ph](args, model, tok, handler, outdir)

    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
