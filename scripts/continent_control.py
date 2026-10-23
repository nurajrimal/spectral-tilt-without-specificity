#!/usr/bin/env python3
"""
A SEMANTIC control for the quiet tail: does truth prefer the low-variance tail
more than a meaning-based label with no truth in it?

The pre-registered quiet-tail control is surface length, and the thesis names
that as its main limitation: length is not semantic. The truth_cities statements
carry a semantic label that nobody had used,

    "The city of Krasnodar is in Russia."         true
    "The city of Krasnodar is in South Africa."   false

  named_continent  (PRIMARY CONTROL)
      The continent of the country NAMED in the statement: Europe, then Africa.
      A property of meaning (world knowledge about the country), carried by the
      very span in which the two members of a pair differ, and almost independent
      of truth by construction: the false country is drawn from all over the
      world. On the 512 pairs used here it predicts truth at 0.514 (chance 0.5;
      Cramer's V 0.049, p 0.48, in both truth_cities and truth_cities_negated).
      It varies within 312 of 512 pairs, so it is learnable under pair-grouped
      folds the way truth is.

  city_continent  (SECONDARY CONTROL)
      The continent of the CITY (of its correct country). The same for both
      members of a pair, so exactly independent of truth, and learnable across
      pairs because it is a general property of the statements, as topic is in
      semantic_control.py. The leakage detector pair_id_random is still computed
      and must sit at chance.

  length_bucket  (REFERENCE)
      The pre-registered control, re-run on the same activations so the new
      numbers sit beside the thesis numbers for the same cell.

Continents follow UN M49 (data/country_continent.json). Australia, the only
Oceania country and named in one statement, is folded into Asia so no class has
a single member; this is fixed in the pre-registration, not chosen after.

Datasets: truth_cities (primary) and truth_cities_negated (replication; the same
cities, with "not", so truth is flipped against the named country).

Everything else is the thesis machinery, imported rather than copied: the same
basis, windows, probe, pair-grouped folds, retention and paired bootstrap as
test_concept_subspace.py. Activations are read from that script's cache
(results/_spectral/concept_subspace/acts_*), so a model already run costs probes
only; a missing cache is extracted once and saved.

    python scripts/continent_control.py run --models Qwen/Qwen2.5-0.5B ...
    python scripts/continent_control.py summarize

Freeze prereg/continent_control_prereg.json BEFORE the first real run.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CONTINENTS = REPO_ROOT / "data" / "country_continent.json"
OUT_DEFAULT = REPO_ROOT / "results" / "_spectral" / "continent_control"
DATASETS = ("truth_cities", "truth_cities_negated")
CONTROLS = ("named_continent", "city_continent", "length_bucket")
PRIMARY = {"dataset": "truth_cities", "layer": "mid", "control": "named_continent"}
FOLD = {"Oceania": "Asia"}        # pre-registered: one-member class folded
SESOI = 0.10                      # thesis retention-scale bound
GEMMA = "gemma-2-2b"
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-4B",
          "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.1-8B", "google/gemma-2-2b"]

_NAMED = re.compile(r" is (?:not )?in (.+?)\.?$")


# ─── Labels (pure; tested without a model) ───────────────────────────────────

def load_continents(path=CONTINENTS):
    m = json.load(open(path, encoding="utf-8"))
    out = {}
    for cont, countries in m.items():
        if cont.startswith("_"):
            continue
        for c in countries:
            out[c] = FOLD.get(cont, cont)
    return out


def named_country(statement: str) -> str:
    m = _NAMED.search(statement.strip())
    if not m:
        raise ValueError(f"cannot find the named country in {statement!r}")
    return m.group(1)


def continent_labels(pairs, cmap):
    """
    Two integer label vectors over 2N statements, ordered [all positives, all
    negatives] like build_label_sets, plus the class names.
    """
    def cont(country):
        if country not in cmap:
            raise KeyError(f"country {country!r} missing from {CONTINENTS.name}")
        return cmap[country]

    texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    named = [cont(named_country(t)) for t in texts]
    city = []
    for p in pairs:
        correct = p.meta.get("topic")
        if not correct:
            raise KeyError("pair has no 'topic' (correct country); rebuild "
                           "data/statements with scripts/build_datasets.py")
        city.append(cont(correct))
    city = city + city

    classes = sorted(set(named) | set(city))
    idx = {c: i for i, c in enumerate(classes)}
    return (np.array([idx[c] for c in named]), np.array([idx[c] for c in city]),
            classes)


def leakage(control, concept):
    """How much a control label says about truth: Cramer's V and best accuracy."""
    from scipy.stats import chi2_contingency
    ks = sorted(set(control))
    tab = np.array([[np.sum((control == k) & (concept == v)) for k in ks]
                    for v in (0, 1)])
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2:
        return {"cramers_v": 0.0, "p": 1.0, "truth_acc_from_control": 0.5}
    chi2, p, _, _ = chi2_contingency(tab)
    v = float(np.sqrt(chi2 / tab.sum() / (min(tab.shape) - 1)))
    acc = float(tab.max(0).sum() / tab.sum())
    return {"cramers_v": v, "p": float(p), "truth_acc_from_control": acc}


# ─── Model side ──────────────────────────────────────────────────────────────

def load_modules():
    path = SCRIPT_DIR / "test_concept_subspace.py"
    spec = importlib.util.spec_from_file_location("test_concept_subspace", path)
    tcs = importlib.util.module_from_spec(spec)
    sys.modules["test_concept_subspace"] = tcs
    spec.loader.exec_module(tcs)
    return tcs, tcs.load_syntax_module()


def run_dataset(tcs, tsc, model_path, dataset, args, cmap):
    from linear_rep_geometry.behavioral import load_pairs

    name = model_path.split("/")[-1]
    out_path = Path(args.out_dir) / f"{name}__{dataset}.json"
    if args.skip_existing and out_path.exists():
        print(f"  {name} / {dataset}: exists, skipping")
        return

    print(f"\n{'=' * 70}\n  {name}  /  {dataset}  /  continent controls\n{'=' * 70}")
    src = tcs.STATEMENTS / f"{dataset}.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing: copy data/statements/ from your working "
                         f"folder or run scripts/build_datasets.py --clone")
    pairs = load_pairs(src)[: args.max_pairs]

    evals, evecs, basis = tcs.load_eigen(name, args)
    d = evecs.shape[0]
    k = max(1, int(d * args.k_frac))
    n_layers = tcs.depth_from_config(model_path)
    roles = tcs.resolve_layers(args.layers, n_layers)
    layers = [li for _, li in roles]
    role_of = {li: role for role, li in roles}

    acts, _, dim = tcs.cached_activations(model_path, dataset, pairs, layers,
                                          args.pooling, args)
    if dim != d:
        raise SystemExit(f"activation width {dim} != basis width {d}")

    base = tcs.build_label_sets(pairs, tsc)
    named, city, classes = continent_labels(pairs, cmap)
    labels = {"concept": base["concept"], "concept_shuffled": base["concept_shuffled"],
              "pair_id_random": base["pair_id_random"],
              "named_continent": named, "city_continent": city,
              "length_bucket": base["length_bucket"]}
    Y = {nm: np.asarray(v) for nm, v in labels.items()}
    N = len(pairs)
    groups = np.r_[np.arange(N), np.arange(N)]

    leak = {c: leakage(Y[c], Y["concept"]) for c in CONTROLS}
    dist = {c: dict(Counter(classes[i] for i in Y[c])) for c in
            ("named_continent", "city_continent")}
    print(f"  {N} pairs; classes {classes}")
    for c in CONTROLS:
        print(f"    {c:16s} majority {tsc.majority_rate(Y[c]):.3f}  "
              f"truth-from-control acc {leak[c]['truth_acc_from_control']:.3f}  "
              f"V {leak[c]['cramers_v']:.3f}")

    out = {"model": model_path, "model_name": name, "concept": dataset,
           "matched": True, "basis": basis, "basis_kind": tsc.basis_kind(basis),
           "d": d, "k": k, "k_frac": args.k_frac, "n_pairs": N,
           "n_statements": 2 * N, "pooling": args.pooling, "n_layers": n_layers,
           "probe": tsc.PROBE_KIND, "classes": classes, "class_counts": dist,
           "leakage": leak, "controls": list(CONTROLS), "primary": PRIMARY,
           "layer_roles": {str(li): role for li, role in role_of.items()},
           "script_version": "continent-control-1", "argv": sys.argv,
           "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "by_layer": {}}

    for layer in layers:
        out["by_layer"][str(layer)] = {"role": role_of[layer], "by_control": {}}
        for control in CONTROLS:
            args.control = control
            Yc = {nm: Y[nm] for nm in ("concept", "concept_shuffled",
                                       "pair_id_random", control)}
            r = tcs.run_cell(tsc, acts[layer].float(), Yc, evals, evecs, d, k,
                             layer, args, groups=groups, ks=args.k_fracs)
            did = r["difference_in_differences"]
            leak_acc = r["full_accuracy"]["pair_id_random"]
            print(f"  {role_of[layer]:<5} L{layer} vs {control:16s} concept headroom "
                  f"{r['headroom']['concept']:+.3f}  control headroom "
                  f"{r['headroom'][control]:+.3f}  leak {leak_acc:.3f}  "
                  + (f"Delta {did['delta']:+.4f} [{did['ci95'][0]:+.4f}, "
                     f"{did['ci95'][1]:+.4f}]" if did.get("available")
                     else f"Delta unavailable: {did.get('reason')}"))
            out["by_layer"][str(layer)]["by_control"][control] = r

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    print(f"  wrote {out_path}")


def cmd_run(args):
    tcs, tsc = load_modules()
    tcs.OUT_DIR = Path(args.out_dir)
    tcs.ACTS_DIR = Path(args.acts_dir)
    tsc.PROBE_GPU = False if args.cpu_probe else torch.cuda.is_available()
    tsc.PROBE_KIND = "linear"
    cmap = load_continents()
    failed = 0
    for m in args.models:
        for ds in args.datasets:
            try:
                run_dataset(tcs, tsc, m, ds, args, cmap)
            except SystemExit:
                raise
            except Exception as e:
                failed += 1
                print(f"\n  {m} / {ds} failed: {type(e).__name__}: {e}")
    return 1 if failed else 0


# ─── Across models ───────────────────────────────────────────────────────────

def usable(cell, control):
    """Thesis rule: concept headroom >= 0.10 and the shuffled floor passes."""
    h = cell["headroom"]
    reasons = []
    if h["concept"] < 0.10:
        reasons.append(f"concept headroom {h['concept']:.3f} < 0.10")
    if not cell["floor_check"]["ok"]:
        reasons.append("shuffled floor check failed")
    if h[control] <= 0.02:
        reasons.append(f"control headroom {h[control]:.3f} <= 0.02")
    if not cell["difference_in_differences"].get("available"):
        reasons.append("Delta unavailable")
    return not reasons, reasons


def _f(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def _p(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cmd_summarize(args):
    from linear_rep_geometry.spectral_metrics import one_sample_test, tost_equivalence

    folder = Path(args.out_dir)
    files = sorted(folder.glob("*__*.json"))
    if not files:
        raise SystemExit(f"no result files in {folder}")
    res = [json.load(open(f)) for f in files]

    rows, lines = [], [
        "# Semantic (continent) control for the quiet tail", "",
        f"Equivalence bound ±{SESOI} retention units. Δ > 0 means truth prefers the "
        "quiet tail MORE than the control does.", "",
        f"Primary: {PRIMARY['dataset']}, {PRIMARY['layer']} layer, control "
        f"{PRIMARY['control']}.", "",
        "## Per model", "",
        "| model | dataset | layer | control | Δ | 95% CI | concept headroom | control headroom | leak | usable |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for li, L in r["by_layer"].items():
            for c, cell in L["by_control"].items():
                ok, why = usable(cell, c)
                did = cell["difference_in_differences"]
                row = {"model": r["model_name"], "dataset": r["concept"],
                       "layer": L["role"], "control": c, "usable": ok,
                       "delta": did.get("delta"), "ci95": did.get("ci95"),
                       "concept_headroom": cell["headroom"]["concept"],
                       "control_headroom": cell["headroom"][c],
                       "leak": cell["full_accuracy"]["pair_id_random"],
                       "why": "; ".join(why)}
                rows.append(row)
                ci = row["ci95"] or [None, None]
                lines.append(f"| {row['model']} | {row['dataset']} | {row['layer']} | {c} | "
                             f"{_f(row['delta'])} | [{_f(ci[0])}, {_f(ci[1])}] | "
                             f"{row['concept_headroom']:+.3f} | {row['control_headroom']:+.3f} | "
                             f"{row['leak']:.3f} | {'yes' if ok else 'no: ' + row['why']} |")

    lines += ["", "## Across models (usable cells only)", "",
              "| dataset | layer | control | models | n | mean Δ | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tests = {}
    keys = sorted({(x["dataset"], x["layer"], x["control"]) for x in rows},
                  key=lambda t: (t[0] != PRIMARY["dataset"], t[1] != PRIMARY["layer"],
                                 CONTROLS.index(t[2]) if t[2] in CONTROLS else 9))
    for ds, layer, c in keys:
        for drop in (False, True):
            vals = [x["delta"] for x in rows
                    if (x["dataset"], x["layer"], x["control"]) == (ds, layer, c)
                    and x["usable"] and not (drop and x["model"] == GEMMA)]
            key = f"{ds}|{layer}|{c}|{'without_gemma' if drop else 'all'}"
            if len(vals) < 2:
                tests[key] = {"n": len(vals)}
                continue
            t, q = one_sample_test(vals), tost_equivalence(vals, sesoi=SESOI)
            tests[key] = {"n": len(vals), "test": t, "tost": q}
            prim = (ds, layer, c) == tuple(PRIMARY.values())
            ci, ci90 = t.get("ci95", [None, None]), q.get("ci90", [None, None])
            lines.append(f"| {ds} | {layer} | {'**' + c + ' (primary)**' if prim else c} | "
                         f"{'without gemma' if drop else 'all'} | {len(vals)} | {_f(t['mean'])} | "
                         f"[{_f(ci[0])}, {_f(ci[1])}] | {_p(t.get('p_t'))} | {_p(t.get('p_wilcoxon'))} | "
                         f"[{_f(ci90[0])}, {_f(ci90[1])}] | {_p(q.get('p_tost'))} | "
                         f"{'yes' if q.get('conclusive') else 'no'} |")

    lines += ["", "Reading the primary row (the pre-registered decision rule):", "",
              "- 95% CI above 0 in both versions: truth is quiet-tail-specific against a "
              "semantic control; the length result does not generalise.",
              "- 90% CI inside ±0.10 in both versions: not specific even against a "
              "semantic control.",
              "- otherwise: inconclusive at this n.", "",
              "`leak` is the full-dimensional accuracy of pair_id_random and must sit "
              "near 0.5; a cell where it does not is not trustworthy."]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"rows": rows, "tests": tests}, open(folder / "summary.json", "w"),
              indent=2, default=float)
    print("\n".join(lines))
    print(f"\nwrote {folder}/summary.md and summary.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument("--datasets", nargs="+", default=list(DATASETS))
    r.add_argument("--out-dir", default=str(OUT_DEFAULT))
    r.add_argument("--acts-dir", default="results/_spectral/concept_subspace")
    r.add_argument("--cov")
    r.add_argument("--layers", nargs="+", default=["mid", "final"])
    r.add_argument("--pooling", default="last_token")
    r.add_argument("--k-frac", type=float, default=0.005)
    r.add_argument("--k-fracs", type=float, nargs="+")
    r.add_argument("--n-windows", type=int, default=10)
    r.add_argument("--n-boot", type=int, default=10_000)
    r.add_argument("--max-pairs", type=int, default=512)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--max-length", type=int, default=256)
    r.add_argument("--refresh-acts", action="store_true")
    r.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    r.add_argument("--cpu-probe", action="store_true")
    r.add_argument("--skip-existing", action="store_true")
    r.set_defaults(require_gpu=True, control=PRIMARY["control"], func=cmd_run)

    s = sub.add_parser("summarize")
    s.add_argument("--out-dir", default=str(OUT_DEFAULT))
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
