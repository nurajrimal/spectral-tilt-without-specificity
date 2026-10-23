#!/usr/bin/env python3
"""
Verdict for the single pre-registered confirmatory cell
(prereg/continent_confirm_prereg.json):

    truth_cities_negated, mid layer, k = 0.10 d, control named_continent

Reads results/_spectral/continent_confirm/ksweep_summary.json, which
continent_ksweep_summary.py writes (it applies the usable + readable inclusion
rule). Applies the validity gate (>= 6 of 8 models included) and the decision
rule, and writes confirm_verdict.md.

    python scripts/continent_ksweep_summary.py --out-dir results/_spectral/continent_confirm
    python scripts/continent_confirm_report.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY = "truth_cities_negated|mid|named_continent|0.1"
MIN_MODELS = 6
BOUND = 0.10


def verdict(block):
    def side(t):
        if "test" not in t:
            return None
        ci = t["test"].get("ci95")
        ci90 = t["tost"].get("ci90")
        return {"n": t["n"], "mean": t["test"]["mean"], "ci95": ci, "ci90": ci90,
                "p": t["test"].get("p_t"), "p_tost": t["tost"].get("p_tost"),
                "above": bool(ci and ci[0] > 0), "below": bool(ci and ci[1] < 0),
                "equiv": bool(ci90 and -BOUND < ci90[0] and ci90[1] < BOUND)}

    a, w = side(block.get("all", {})), side(block.get("without_gemma", {}))
    n_inc = block.get("all", {}).get("n", 0)
    if n_inc < MIN_MODELS or a is None or w is None:
        return "NOT TESTABLE", a, w, n_inc
    if a["above"] and w["above"]:
        return "REPLICATES", a, w, n_inc
    if (a["equiv"] and w["equiv"]) or (a["below"] and w["below"]):
        return "DOES NOT REPLICATE", a, w, n_inc
    return "INCONCLUSIVE", a, w, n_inc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/_spectral/continent_confirm")
    args = ap.parse_args()
    folder = Path(args.out_dir)
    tests = json.load(open(folder / "ksweep_summary.json"))["tests"]
    if KEY not in tests:
        raise SystemExit(f"{KEY} not in {folder}/ksweep_summary.json; was the run "
                         f"made with --datasets truth_cities_negated --k-frac 0.10?")
    block = tests[KEY]
    v, a, w, n_inc = verdict(block)

    def fmt(s):
        if s is None:
            return "–"
        return (f"n {s['n']}, mean Δ {s['mean']:+.3f}, 95% CI [{s['ci95'][0]:+.3f}, "
                f"{s['ci95'][1]:+.3f}], p {s['p']:.3f}, 90% CI [{s['ci90'][0]:+.3f}, "
                f"{s['ci90'][1]:+.3f}], TOST p {s['p_tost']:.3f}")

    lines = ["# Confirmatory continent-control test (pre-registered)", "",
             "Cell: truth_cities_negated, mid layer, k = 10% of d, control named_continent.", "",
             f"Models included (usable and control readable): {n_inc} of {block['of']} "
             f"(validity gate: at least {MIN_MODELS}).", "",
             f"- all included models: {fmt(a)}",
             f"- without gemma-2-2b: {fmt(w)}", "",
             f"**Verdict: {v}**", "",
             "Rule: replicates if the 95% CI is above 0 in both versions; does not "
             "replicate if the 90% CI is inside ±0.10 (or the 95% CI is below 0) in both; "
             "otherwise inconclusive."]
    (folder / "confirm_verdict.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
