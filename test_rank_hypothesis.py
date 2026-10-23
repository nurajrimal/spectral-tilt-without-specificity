#!/usr/bin/env python3
"""Does effective rank predict the spectral gap?

Reads results/_analysis/subspace.json. No GPU, seconds to run.

TEST 2 is decisive: within the truth family every dataset is matched, so
pairing quality is roughly constant while PR varies by an order of magnitude.
If rank predicts gap there, rank survives the pairing confound. If not, the
pooled correlation WAS the confound, and the anisotropy account stands alone.

TEST 3 is the inferential version: concepts inside one model share an
eigenbasis and are not independent, so the model is the unit of analysis.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats


def load(path):
    d = json.load(open(path)); rows = []
    for concept, entry in d.get("concepts", {}).items():
        if concept.startswith("_"): continue
        for model, r in entry.get("per_model", {}).items():
            rows.append({"concept": concept, "model": model,
                "rank": float(r["rank"]), "pr": float(r.get("pr", r["rank"])),
                "gap": float(r["gap"]), "p": float(r.get("p", np.nan)),
                "family": ("refusal" if concept.startswith("refusal")
                           else "truth" if concept.startswith("truth")
                           else "sycophancy" if concept.startswith("sycophancy")
                           else "other")})
    return rows


def report(x, y, label, note=""):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 4:
        print(f"  {label:38s} n={len(x)}  too few points"); return None
    r, pr_ = stats.pearsonr(x, y); rho, prho = stats.spearmanr(x, y)
    star = "  <-- significant" if prho < 0.05 else ""
    print(f"  {label:38s} n={len(x):3d}  Pearson r={r:+.3f} (p={pr_:.4f})  "
          f"Spearman rho={rho:+.3f} (p={prho:.4f}){star}")
    if note: print(f"  {'':38s} {note}")
    return {"n": int(len(x)), "pearson_r": float(r), "pearson_p": float(pr_),
            "spearman_rho": float(rho), "spearman_p": float(prho)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/_analysis/subspace.json")
    ap.add_argument("--out", default="results/_analysis/rank_hypothesis.json")
    args = ap.parse_args()

    rows = load(args.json)
    if not rows: raise SystemExit(f"no rows in {args.json}")
    print(f"{len(rows)} (concept, model) cells from {args.json}")
    out = {}

    print("\n=== Effective rank by family ===")
    byfam = defaultdict(list)
    for r in rows: byfam[r["family"]].append(r)
    for fam in sorted(byfam):
        prs = [r["pr"] for r in byfam[fam]]; gaps = [r["gap"] for r in byfam[fam]]
        print(f"  {fam:12s} n={len(prs):3d}  PR median {np.median(prs):6.2f}  "
              f"[{min(prs):.2f}, {max(prs):.2f}]   mean gap {np.mean(gaps):+.4f}")
        out.setdefault("by_family", {})[fam] = {"n": len(prs),
            "pr_median": float(np.median(prs)), "gap_mean": float(np.mean(gaps))}

    print("\n=== TEST 1: pooled (DESCRIPTIVE ONLY — points not independent) ===")
    out["test1"] = report([r["pr"] for r in rows], [r["gap"] for r in rows],
                          "PR vs gap, all concepts")
    out["test1_log"] = report(np.log([r["pr"] for r in rows]),
                              [r["gap"] for r in rows], "log(PR) vs gap",
                              "PR spans 1 to 80; log is the fairer scale")

    print("\n=== TEST 2: truth family only — pairing held ~constant (DECISIVE) ===")
    truth = [r for r in rows if r["family"] == "truth"]
    out["test2"] = report([r["pr"] for r in truth], [r["gap"] for r in truth],
                          "PR vs gap, truth only")
    out["test2_log"] = report(np.log([r["pr"] for r in truth]),
                              [r["gap"] for r in truth], "log(PR) vs gap, truth only")
    print("\n  Per model (eigenbasis fixed, truth concepts only):")
    bym = defaultdict(list)
    for r in truth: bym[r["model"]].append(r)
    rhos = []
    for m in sorted(bym):
        g = bym[m]
        if len(g) < 4: continue
        rho, p = stats.spearmanr([x["pr"] for x in g], [x["gap"] for x in g])
        rhos.append(rho); print(f"    {m:18s} n={len(g):2d}  rho={rho:+.3f} (p={p:.4f})")
    if len(rhos) >= 3:
        t, p = stats.ttest_1samp(rhos, 0.0)
        print(f"    mean rho {np.mean(rhos):+.3f} across {len(rhos)} models, "
              f"t={t:+.2f}, p={p:.4f}")
        out["test2_per_model"] = {"n_models": len(rhos),
            "mean_rho": float(np.mean(rhos)), "t": float(t), "p": float(p)}

    print("\n=== TEST 3: per-model, all concepts (INFERENTIAL) ===")
    bym = defaultdict(list)
    for r in rows: bym[r["model"]].append(r)
    rhos = []
    for m in sorted(bym):
        g = bym[m]
        rho, p = stats.spearmanr([x["pr"] for x in g], [x["gap"] for x in g])
        rhos.append(rho); print(f"    {m:18s} n={len(g):2d}  rho={rho:+.3f} (p={p:.4f})")
    if len(rhos) >= 3:
        t, p = stats.ttest_1samp(rhos, 0.0)
        verdict = ("rank predicts gap" if p < 0.05 and np.mean(rhos) > 0
                   else "rank predicts gap (negative)" if p < 0.05
                   else "NO consistent rank-gap relationship")
        print(f"    mean rho {np.mean(rhos):+.3f} across {len(rhos)} models, "
              f"t={t:+.2f}, p={p:.4f}  -> {verdict}")
        out["test3"] = {"n_models": len(rhos), "mean_rho": float(np.mean(rhos)),
                        "t": float(t), "p": float(p), "verdict": verdict}

    print("\n=== TEST 4: rank vs pairing as competing explanations ===")
    MATCHED = {"truth_cities","truth_cities_negated","truth_translation",
               "truth_magnitude","truth_counterfact","truth_conjunction",
               "truth_disjunction","sycophancy","sycophancy_nlp",
               "sycophancy_philpapers","sycophancy_political"}
    for r in rows: r["matched"] = 1.0 if r["concept"] in MATCHED else 0.0
    out["test4_pairing_gap"] = report([r["matched"] for r in rows],
                                      [r["gap"] for r in rows], "pairing vs gap")
    out["test4_pairing_pr"] = report([r["matched"] for r in rows],
                                     [r["pr"] for r in rows], "pairing vs PR",
                                     "if strong, the two are confounded; TEST 2 decides")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")
    print("\nTEST 2 significant -> rank survives with pairing constant.")
    print("TEST 2 null        -> pooled correlation was the confound; drop the")
    print("                      rank story, report the anisotropy account.")
    print("TEST 1 vs TEST 3 disagree -> report TEST 3; models are the unit.")


if __name__ == "__main__":
    main()
