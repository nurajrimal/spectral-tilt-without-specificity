#!/usr/bin/env python3
"""Standalone analysis of the expanded-semantic anti-concentration (rebuttal).

Reads only saved .pt files (results/<model>/en-fr/cov_src.pt + v_raw_*.pt) — no
model loading. Computes per-concept Gini deviation and Spectral Center of Mass
using the SAME definitions as spectral_analysis.py, for the original 7 semantic
concepts and the 15 expanded (BATS) semantic concepts, then reports:
  - per-model and aggregate mean Gini / SCM for the expanded set,
  - one-sample t-test (expanded Gini < 0),
  - Mann-Whitney U (expanded vs original-7 per-concept Gini).

Usage (on the machine that has the saved covariances):
    .venv/bin/python scripts/analyze_expanded_semantic.py
    # optional: --results DIR (default results) --pair en-fr --out summary.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

try:
    from scipy import stats as sp
except Exception:
    sp = None

torch.set_grad_enabled(False)  # saved v_raw tensors carry requires_grad=True

ORIGINAL_SEMANTIC = [
    "[male - female]", "[country - capital]", "[thing - color]", "[thing - part]",
    "[small - big]", "[lower - upper]", "[frequent - infrequent]",
]
EXPANDED_SEMANTIC = [
    "[country - language]", "[name - nationality]", "[name - occupation]",
    "[animal - young]", "[animal - sound]", "[animal - shelter]",
    "[hypernyms - animals]", "[hypernyms - misc]", "[hyponyms - misc]",
    "[meronyms - substance]", "[meronyms - member]",
    "[synonyms - intensity]", "[synonyms - exact]",
    "[antonyms - gradable]", "[antonyms - binary]",
]


def eigendecompose(cov):
    ev, U = torch.linalg.eigh(cov)
    idx = torch.argsort(ev, descending=True)
    return ev[idx], U[:, idx]


def energy_cdf(v, U):
    v = v.float()
    proj = U.T @ v
    energy = (proj ** 2) / (v @ v).clamp(min=1e-12)
    return torch.cumsum(energy, dim=0).cpu().numpy()


_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def gini_deviation(cdf, cumvar):
    return float(_TRAPZ(cdf - cumvar, cumvar))


def scm(cdf, cumvar):
    idx = int(np.searchsorted(cdf, 0.5))
    return 1.0 if idx >= len(cumvar) else float(cumvar[idx])


def analyze_model(model_dir, pair, cov_name="cov_src.pt"):
    cov_path = os.path.join(model_dir, pair, cov_name)
    if not os.path.exists(cov_path):
        return None
    cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
    ev, U = eigendecompose(cov)
    cumvar = np.cumsum(ev.numpy()) / np.sum(ev.numpy())
    vecdir = os.path.join(model_dir, pair)

    def metrics_for(names):
        out = {}
        for name in names:
            p = os.path.join(vecdir, f"v_raw_{name}.pt")
            if not os.path.exists(p):
                continue
            v = torch.load(p, map_location="cpu", weights_only=True).float()
            if float(torch.norm(v)) == 0.0:
                continue
            cdf = energy_cdf(v, U)
            out[name] = {"gini": gini_deviation(cdf, cumvar), "scm": scm(cdf, cumvar)}
        return out

    return {"orig": metrics_for(ORIGINAL_SEMANTIC),
            "exp": metrics_for(EXPANDED_SEMANTIC)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--cov-name", default="cov_src.pt",
                    help="covariance file to use (e.g. cov_src_centered.pt)")
    ap.add_argument("--out", default="results/_spectral/expanded_semantic_summary.json")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="skip model dirs whose name contains any of these substrings")
    args = ap.parse_args()

    model_dirs = sorted(d for d in glob.glob(os.path.join(args.results, "*"))
                        if os.path.isdir(d) and not os.path.basename(d).startswith("_")
                        and not any(x in os.path.basename(d) for x in args.exclude))
    per_model, all_exp_gini, all_orig_gini = {}, [], []
    print(f"{'model':22}{'n_exp':>6}{'exp_gini':>10}{'exp_scm':>9}{'orig_gini':>11}")
    for md in model_dirs:
        name = os.path.basename(md)
        res = analyze_model(md, args.pair, args.cov_name)
        if not res or not res["exp"]:
            continue
        eg = [m["gini"] for m in res["exp"].values()]
        es = [m["scm"] for m in res["exp"].values()]
        og = [m["gini"] for m in res["orig"].values()]
        all_exp_gini += eg
        all_orig_gini += og
        per_model[name] = {
            "n_exp": len(eg), "exp_gini_mean": float(np.mean(eg)),
            "exp_scm_mean": float(np.mean(es)),
            "orig_gini_mean": float(np.mean(og)) if og else None,
        }
        print(f"{name:22}{len(eg):>6}{np.mean(eg):>10.3f}{np.mean(es):>9.3f}"
              f"{(np.mean(og) if og else float('nan')):>11.3f}")

    exp = np.array(all_exp_gini)
    orig = np.array(all_orig_gini)
    agg = {"n_models": len(per_model), "n_exp_points": int(exp.size),
           "exp_gini_mean": float(exp.mean()) if exp.size else None,
           "orig_gini_mean": float(orig.mean()) if orig.size else None}
    print("\n=== AGGREGATE ===")
    print(f"  models: {agg['n_models']}, expanded concept-points: {agg['n_exp_points']}")
    print(f"  expanded mean Gini: {agg['exp_gini_mean']}")
    print(f"  original mean Gini: {agg['orig_gini_mean']}")
    if sp is not None and exp.size:
        t, p = sp.ttest_1samp(exp, 0.0)
        agg["ttest_exp_vs_0"] = {"t": float(t), "p": float(p)}
        print(f"  one-sample t (expanded Gini vs 0): t={t:.2f}, p={p:.3e}")
        if orig.size:
            u, pu = sp.mannwhitneyu(exp, orig, alternative="two-sided")
            agg["mannwhitney_exp_vs_orig"] = {"U": float(u), "p": float(pu)}
            print(f"  Mann-Whitney (expanded vs original-7 Gini): p={pu:.3f}")
    else:
        print("  (scipy unavailable — install for t-test / Mann-Whitney)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_model": per_model, "aggregate": agg}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
