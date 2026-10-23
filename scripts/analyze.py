#!/usr/bin/env python3
"""
Cross-model analysis: the tests the proposal's section 3 commits to.

    python scripts/analyze.py --models Qwen2.5-0.5B Qwen2.5-3B Llama-3.2-3B ...

Reads results/<model>/phase2_<basis>.json for every model and produces:

  1. H0: mean Gini gap = 0, across models, per concept.
  2. TOST equivalence against SESOI = -0.082 wherever H0 is not rejected.
  3. The designed contrast: refusal gap vs truth gap, paired by model.
  4. Behavioural vs word-level, paired by model, where the control exists.

Models are the unit of analysis. Concepts inside one model share an eigenbasis
and are not independent, which is why the published paper keeps its
concept-level statistics descriptive and reports n=17 models. With eight models
n is small, so Wilcoxon is reported beside the t-test and disagreement between
them is a reason to report both rather than to pick.

Everything is driven by the primary null — the label shuffle — with the
norm-matched random gap reported alongside. Where the two disagree the table
says so; that disagreement is a finding about the spectrum, not a nuisance to
be resolved by choosing the friendlier number.
"""

import argparse
import csv
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from linear_rep_geometry.spectral_metrics import (  # noqa: E402
    one_sample_test, paired_test, tost_equivalence, SESOI,
)

TRUTH_PREFIX = "truth_"
REFUSAL_PREFIX = "refusal"
SYCO_PREFIX = "sycophancy"


def load_model_results(root: Path, model: str, basis: str) -> dict | None:
    p = root / model / f"phase2_{basis}.json"
    if not p.exists():
        print(f"  ! {model}: no {p.name}")
        return None
    return json.load(open(p))


def collect(root: Path, models: list[str], basis: str, pooling: str,
            gate: float) -> dict:
    """
    Gather per-(concept, model) gaps under both nulls.

    Cells failing the split-half gate are dropped and counted. A direction that
    does not replicate across halves has no gap worth testing, and including it
    would put estimation noise into the cross-model mean.
    """
    gaps = defaultdict(dict)
    dropped = []
    for m in models:
        res = load_model_results(root, m, basis)
        if not res:
            continue
        for concept, per_pool in res.get("concepts", {}).items():
            cell = per_pool.get(pooling)
            if not cell:
                continue
            r = cell.get("split_half", {}).get("cosine_mean", 1.0)
            if r < gate:
                dropped.append((m, concept, r))
                continue
            gaps[concept][m] = {
                "shuffle": cell["shuffle_null"]["gap"],
                "random": cell["random_null"]["gap"],
                "gini": cell["observed_gini"],
                "p_shuffle": cell["shuffle_null"]["p_two_sided"],
                "agree": cell["agree"],
                "split_half": r,
            }
    return {"gaps": dict(gaps), "dropped": dropped}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--basis", default="full_vocab")
    ap.add_argument("--pooling", default="last_token",
                    choices=["last_token", "mean_over_sequence"])
    ap.add_argument("--null", default="shuffle", choices=["shuffle", "random"])
    ap.add_argument("--gate", type=float, default=0.70)
    ap.add_argument("--sesoi", type=float, default=SESOI)
    ap.add_argument("--out-dir", default="results/_analysis")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"basis={args.basis}  pooling={args.pooling}  null={args.null}  "
          f"gate r>={args.gate}")
    col = collect(root, args.models, args.basis, args.pooling, args.gate)
    gaps, dropped = col["gaps"], col["dropped"]

    if dropped:
        print(f"\n{len(dropped)} cells dropped below the reliability gate:")
        for m, c, r in dropped:
            print(f"  {m:18s} {c:24s} r={r:.2f}")

    if not gaps:
        raise SystemExit("\nno usable cells — run Phase 2 first")

    report = {"basis": args.basis, "pooling": args.pooling, "null": args.null,
              "gate": args.gate, "models": args.models,
              "dropped": [{"model": m, "concept": c, "split_half": r}
                          for m, c, r in dropped],
              "per_concept": {}, "contrasts": {}}

    # ── 1 & 2: per concept across models ──
    print(f"\n{'concept':26s} {'n':>3s} {'mean gap':>9s} {'95% CI':>18s} "
          f"{'p':>8s}  verdict")
    print("-" * 82)

    rows = []
    for concept in sorted(gaps):
        per_model = gaps[concept]
        vals = [per_model[m][args.null] for m in sorted(per_model)]
        t = one_sample_test(vals)
        entry = {"n_models": t["n"], "mean_gap": t["mean"], "test": t,
                 "per_model": per_model}

        if t.get("p_t") is not None and t["p_t"] < 0.05:
            verdict = ("anti-concentrates" if t["mean"] < 0 else "CONCENTRATES")
        else:
            eq = tost_equivalence(vals, sesoi=args.sesoi)
            entry["tost"] = eq
            verdict = ("equivalent to zero" if eq.get("equivalent")
                       else "inconclusive")

        entry["verdict"] = verdict
        report["per_concept"][concept] = entry

        ci = t.get("ci95", [float("nan")] * 2)
        p = t.get("p_t", float("nan"))
        print(f"{concept:26s} {t['n']:3d} {t['mean']:+9.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] {p:8.4f}  {verdict}")

        n_disagree = sum(1 for m in per_model if not per_model[m]["agree"])
        if n_disagree:
            print(f"{'':26s}     nulls disagree in {n_disagree}/{len(per_model)} models")

        for m in sorted(per_model):
            rows.append({"concept": concept, "model": m,
                         **{k: v for k, v in per_model[m].items()}})

    # ── 3: the designed contrast ──
    truth = [c for c in gaps if c.startswith(TRUTH_PREFIX)]
    refus = [c for c in gaps if c.startswith(REFUSAL_PREFIX)]
    syco = [c for c in gaps if c.startswith(SYCO_PREFIX)]

    # Sycophancy is an answer-token contrast: its pairs share nearly every
    # token, so mean_over_sequence dilutes a one-token difference across the
    # whole sequence. Reporting it under that pooling measures noise.
    if syco and args.pooling == "mean_over_sequence":
        print(f"\nNOTE: {len(syco)} sycophancy concept(s) present but pooling is "
              f"mean_over_sequence.\n  These are answer-token contrasts; rerun "
              f"with --pooling last_token for them.")

    if truth and refus:
        shared = sorted(set.intersection(
            *[set(gaps[c]) for c in truth + refus]))
        if len(shared) >= 2:
            tv = [float(np.mean([gaps[c][m][args.null] for c in truth]))
                  for m in shared]
            rv = [float(np.mean([gaps[c][m][args.null] for c in refus]))
                  for m in shared]
            pt = paired_test(rv, tv, "refusal", "truth")
            report["contrasts"]["refusal_vs_truth"] = {"models": shared, **pt}
            print(f"\nDesigned contrast — refusal vs truth, paired over "
                  f"{len(shared)} models")
            print(f"  refusal {pt['mean_a']:+.4f}   truth {pt['mean_b']:+.4f}   "
                  f"diff {pt['mean_diff']:+.4f}   p={pt.get('p_t', float('nan')):.4f}")
            if pt.get("p_t", 1) < 0.05:
                print("  The classes separate. The proposal predicts refusal "
                      "nearer the head;\n  check the sign of the difference "
                      "against that prediction.")
            else:
                print("  No separation detected at this n. The one-sample "
                      "verdicts above still stand\n  on their own.")
        else:
            print("\nDesigned contrast needs >= 2 models with both concepts")

    # ── 3b: sycophancy as a third point on the behavioural axis ──
    if syco and truth:
        shared = sorted(set.intersection(*[set(gaps[c]) for c in syco + truth]))
        if len(shared) >= 2:
            sv = [float(np.mean([gaps[c][m][args.null] for c in syco])) for m in shared]
            tv = [float(np.mean([gaps[c][m][args.null] for c in truth])) for m in shared]
            pt = paired_test(sv, tv, "sycophancy", "truth")
            report["contrasts"]["sycophancy_vs_truth"] = {"models": shared, **pt}
            print(f"\nSycophancy vs truth, paired over {len(shared)} models")
            print(f"  sycophancy {pt['mean_a']:+.4f}   truth {pt['mean_b']:+.4f}   "
                  f"diff {pt['mean_diff']:+.4f}   p={pt.get('p_t', float('nan')):.4f}")
            print("  Sycophancy is output-control like refusal but matched like "
                  "truth, so it\n  separates the behavioural/semantic axis from "
                  "the pairing-quality confound.")

    # ── 4: behavioural vs word-level control ──
    word = [c for c in gaps if c.startswith("word")]
    if word:
        beh = [c for c in gaps if c.startswith(TRUTH_PREFIX)
               or c.startswith(REFUSAL_PREFIX) or c.startswith(SYCO_PREFIX)]
        shared = sorted(set.intersection(*[set(gaps[c]) for c in word + beh]))
        if len(shared) >= 2:
            bv = [float(np.mean([gaps[c][m][args.null] for c in beh])) for m in shared]
            wv = [float(np.mean([gaps[c][m][args.null] for c in word])) for m in shared]
            pt = paired_test(bv, wv, "behavioural", "word_level")
            report["contrasts"]["behavioural_vs_word"] = {"models": shared, **pt}
            print(f"\nBehavioural vs word-level, paired over {len(shared)} models")
            print(f"  behavioural {pt['mean_a']:+.4f}   word {pt['mean_b']:+.4f}   "
                  f"diff {pt['mean_diff']:+.4f}   p={pt.get('p_t', float('nan')):.4f}")
    else:
        print("\nNo word-level control found. Section 6 of the proposal requires "
              "it:\n  without a same-model, same-basis, same-layer comparison the "
              "behavioural\n  numbers can only be read against a published table, "
              "which the paired\n  test exists to avoid.")
        report["contrasts"]["behavioural_vs_word"] = None

    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    if rows:
        with open(out_dir / "gaps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\nwrote {out_dir/'summary.json'} and gaps.csv")


if __name__ == "__main__":
    main()
