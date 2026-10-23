#!/usr/bin/env python3
"""
EXPLORATORY window-size sweep for the continent control (not pre-registered).

Why it exists. At the pre-registered k = 0.005 d the continent labels could not
be read from either the loudest or the quietest window in any model: accuracy
sat exactly at the majority rate (Asia, 0.619 / 0.621) and retention was 0 at
both ends. A control that is unreadable at both ends has a tilt of 0 by
construction, so Delta collapses to the concept's own tilt and the test has no
power to detect specificity. The pre-registered result is reported as run;
this sweep asks, as an exploratory follow-up, what happens once the windows are
large enough for the control to be readable.

It reads the by_k blocks that continent_control.py writes when run with
--k-fracs, and for every (dataset, layer, control, k):

  * per model: Delta, and the control's retention in the top and bottom window
  * READABLE: the control's retention exceeds 0.10 in at least one of the two
    end windows. Cells where it does not are reported but excluded from the
    across-model test, because there Delta is not a specificity statistic
  * across models (readable, usable cells): mean Delta, t-test, TOST +/-0.10,
    with and without gemma-2-2b

    python scripts/continent_control.py run --datasets truth_cities \
        --k-fracs 0.0025 0.005 0.01 0.02 0.05 0.10 \
        --out-dir results/_spectral/continent_ksweep --skip-existing
    python scripts/continent_ksweep_summary.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

spec = importlib.util.spec_from_file_location("continent_control",
                                              SCRIPT_DIR / "continent_control.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

READABLE = 0.10


def end_retention(cells, label):
    by = {(c["label"], c["subspace"]): c["retention"] for c in cells}
    return by.get((label, "top_k")), by.get((label, "bottom_k"))


def _f(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/_spectral/continent_ksweep")
    args = ap.parse_args()

    from linear_rep_geometry.spectral_metrics import one_sample_test, tost_equivalence

    folder = Path(args.out_dir)
    res = [json.load(open(f)) for f in sorted(folder.glob("*__*.json"))]
    if not res:
        raise SystemExit(f"no result files in {folder}")

    rows = []
    for r in res:
        for L in r["by_layer"].values():
            for c, cell in L["by_control"].items():
                ok, why = cc.usable(cell, c)
                for kf, blk in (cell.get("by_k") or {}).items():
                    did = blk["difference_in_differences"]
                    ct, cb = end_retention(blk["cells"], c)
                    pt, pb = end_retention(blk["cells"], "concept")
                    vals = [v for v in (ct, cb) if v is not None and not np.isnan(v)]
                    readable = bool(vals) and max(vals) > READABLE
                    rows.append({"model": r["model_name"], "dataset": r["concept"],
                                 "layer": L["role"], "control": c, "k_frac": float(kf),
                                 "k": blk["k"], "delta": did.get("delta"),
                                 "available": bool(did.get("available")),
                                 "control_ret_top": ct, "control_ret_bottom": cb,
                                 "concept_ret_top": pt, "concept_ret_bottom": pb,
                                 "readable": readable, "usable": ok})

    lines = ["# EXPLORATORY: continent control, window-size sweep", "",
             "Not pre-registered. At the pre-registered k = 0.5% the continent labels "
             "were unreadable at both ends, so Delta there is not a specificity "
             "statistic. A cell counts as READABLE when the control's retention "
             f"exceeds {READABLE} in the top or the bottom window; only readable, "
             "usable cells enter the across-model test.", "",
             "| dataset | layer | control | k | readable models | n (all / w/o gemma) | mean Δ all | 95% CI | 90% CI | equiv. | mean Δ w/o gemma | 90% CI | equiv. |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    keys = sorted({(x["dataset"], x["layer"], x["control"], x["k_frac"]) for x in rows})
    for ds, layer, c, kf in keys:
        sel = [x for x in rows if (x["dataset"], x["layer"], x["control"], x["k_frac"])
               == (ds, layer, c, kf)]
        good = [x for x in sel if x["readable"] and x["usable"] and x["available"]]
        res_k = {}
        for tag, g in (("all", good), ("without_gemma",
                                       [x for x in good if x["model"] != cc.GEMMA])):
            v = [x["delta"] for x in g]
            res_k[tag] = ({"n": len(v), "test": one_sample_test(v),
                           "tost": tost_equivalence(v, sesoi=cc.SESOI)}
                          if len(v) >= 3 else {"n": len(v)})
        out[f"{ds}|{layer}|{c}|{kf:g}"] = {"readable": sum(x["readable"] for x in sel),
                                           "of": len(sel), **res_k}
        a, w = res_k["all"], res_k["without_gemma"]

        def cols(t):
            if "test" not in t:
                return ["–", "–", "–", "–"]
            ci = t["test"].get("ci95", [None, None])
            ci90 = t["tost"].get("ci90", [None, None])
            return [_f(t["test"]["mean"]), f"[{_f(ci[0])}, {_f(ci[1])}]",
                    f"[{_f(ci90[0])}, {_f(ci90[1])}]",
                    "yes" if t["tost"].get("conclusive") else "no"]
        ca, cw = cols(a), cols(w)
        lines.append(f"| {ds} | {layer} | {c} | {100 * kf:g}% | "
                     f"{sum(x['readable'] for x in sel)}/{len(sel)} | {a['n']} / {w['n']} | "
                     f"{ca[0]} | {ca[1]} | {ca[2]} | {ca[3]} | {cw[0]} | {cw[2]} | {cw[3]} |")

    lines += ["", "## Control readability per model (retention top / bottom)", "",
              "| model | dataset | layer | control | k | control top | control bottom | concept top | concept bottom | Δ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (x["dataset"], x["layer"], x["control"],
                                         x["k_frac"], x["model"])):
        lines.append(f"| {x['model']} | {x['dataset']} | {x['layer']} | {x['control']} | "
                     f"{100 * x['k_frac']:g}% | {_f(x['control_ret_top'])} | "
                     f"{_f(x['control_ret_bottom'])} | {_f(x['concept_ret_top'])} | "
                     f"{_f(x['concept_ret_bottom'])} | {_f(x['delta'])} |")

    (folder / "ksweep_summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"rows": rows, "tests": out}, open(folder / "ksweep_summary.json", "w"),
              indent=2, default=float)
    print("\n".join(lines[:6 + len(keys)]))
    print(f"\nwrote {folder}/ksweep_summary.md (full per-model table) and ksweep_summary.json")


if __name__ == "__main__":
    main()
