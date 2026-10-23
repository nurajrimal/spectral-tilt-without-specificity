#!/usr/bin/env python3
"""
A semantic control for the quiet tail (exploratory; added after the freeze).

The pre-registered quiet-tail control is surface length. Its weakness is that it
is not semantic. The three sycophancy subsets (nlp, philpapers, political) give a
semantic label on the same kind of statement: the TOPIC of the question. Pooling
them yields one dataset with two labels,

    concept   sycophantic vs non-sycophantic answer (the pre-registered concept)
    topic     which of the three subsets the question came from (3 classes)

Topic is constant within a pair, like length in the sycophancy sets. A label
that is constant within a pair is unlearnable under pair-grouped folds only if
it is ARBITRARY (pair_id_random); a label that reflects a general property of
the statements, such as length or topic, is learnable across pairs, which is
what makes it usable as a control. The leakage detector (pair_id_random) is
still computed and must sit at chance.

Topic is balanced against the concept by construction (each subset is balanced),
so it carries none of the concept. Its caveat: the three subsets also differ in
prompt format, so the topic label is "semantic or format", and a reader should
see it as a stronger control than length, not a perfect one.

Activations are read from the quiet-tail cache (the three subsets were already
extracted for every model), so this costs probes only.

    python scripts/semantic_control.py --models Qwen/Qwen2.5-3B ... \
        --out-dir results/_spectral/rev/quiet_semantic

Output: one JSON per model in the same schema as test_concept_subspace.py, with
concept 'sycophancy_pooled' and control 'topic', so two_tail_report.py reads it
with --quiet-semantic.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SUBSETS = ("sycophancy_nlp", "sycophancy_philpapers", "sycophancy_political")


def load_tcs():
    path = SCRIPT_DIR / "test_concept_subspace.py"
    spec = importlib.util.spec_from_file_location("test_concept_subspace", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_concept_subspace"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_model(tcs, tsc, model_path, args):
    from linear_rep_geometry.behavioral import load_pairs

    name = model_path.split("/")[-1]
    out_path = tcs.OUT_DIR / f"{name}__sycophancy_pooled.json"
    if args.skip_existing and out_path.exists():
        print(f"  {name}: exists, skipping")
        return

    print(f"\n{'=' * 70}\n  {name}  /  sycophancy pooled, topic control\n{'=' * 70}")
    evals, evecs, basis = tcs.load_eigen(name, args)
    kind = tsc.basis_kind(basis)
    d = evecs.shape[0]
    k = max(1, int(d * args.k_frac))
    n_layers = tcs.depth_from_config(model_path)
    roles = tcs.resolve_layers(args.layers, n_layers)
    layers = [li for _, li in roles]
    role_of = {li: role for role, li in roles}

    pairs_by, acts_by = [], []
    for s in SUBSETS:
        pairs = load_pairs(tcs.STATEMENTS / f"{s}.jsonl")[: args.max_pairs]
        acts, _, dim = tcs.cached_activations(model_path, s, pairs, layers,
                                              args.pooling, args)
        if dim != d:
            raise SystemExit(f"activation width {dim} != basis width {d}")
        pairs_by.append(pairs)
        acts_by.append(acts)

    sizes = [len(p) for p in pairs_by]
    N = sum(sizes)
    pooled_pairs = [p for ps in pairs_by for p in ps]
    # Each subset's cache is [its positives, its negatives]. The pooled order must
    # be [all positives, all negatives] to match build_label_sets and the groups.
    pooled = {}
    for layer in layers:
        pos = [a[layer][:n] for a, n in zip(acts_by, sizes)]
        neg = [a[layer][n:] for a, n in zip(acts_by, sizes)]
        pooled[layer] = torch.cat(pos + neg).float()

    labels = tcs.build_label_sets(pooled_pairs, tsc)
    topic_half = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    labels["topic"] = np.r_[topic_half, topic_half]
    Y = {nm: np.asarray(v) for nm, v in labels.items()}
    groups = np.r_[np.arange(N), np.arange(N)]
    print(f"  {N} pairs pooled from {dict(zip(SUBSETS, sizes))}; "
          f"topic majority {tsc.majority_rate(Y['topic']):.3f}")

    out = {"model": model_path, "model_name": name, "concept": "sycophancy_pooled",
           "matched": True, "contrast_type": "answer_token",
           "basis": basis, "basis_kind": kind, "d": d, "k": k,
           "k_frac": args.k_frac, "k_fracs": args.k_fracs,
           "n_pairs": N, "n_statements": 2 * N, "pooling": args.pooling,
           "n_layers": n_layers, "probe": tsc.PROBE_KIND, "length_mode": "none",
           "control": "topic", "control_applicable": True,
           "subsets": dict(zip(SUBSETS, sizes)), "exploratory": True,
           "layer_roles": {str(li): role for li, role in role_of.items()},
           "script_version": tsc.SCRIPT_VERSION, "argv": sys.argv,
           "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "by_layer": {}}
    for layer in layers:
        r = tcs.run_cell(tsc, pooled[layer], Y, evals, evecs, d, k, layer, args,
                         groups=groups, ks=args.k_fracs)
        leak = r["full_accuracy"]["pair_id_random"]
        did = r["difference_in_differences"]
        print(f"  {role_of[layer]:<5} L{layer}: concept headroom "
              f"{r['headroom']['concept']:+.3f}  topic headroom "
              f"{r['headroom']['topic']:+.3f}  leak {leak:.3f}  "
              + (f"Delta {did['delta']:+.4f} [{did['ci95'][0]:+.4f}, "
                 f"{did['ci95'][1]:+.4f}]" if did.get("available")
                 else f"Delta unavailable: {did.get('reason')}"))
        out["by_layer"][str(layer)] = r

    tcs.OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/_spectral/rev/quiet_semantic")
    ap.add_argument("--acts-dir", default="results/_spectral/concept_subspace")
    ap.add_argument("--cov")
    ap.add_argument("--layers", nargs="+", default=["mid", "final"])
    ap.add_argument("--pooling", default="last_token")
    ap.add_argument("--k-frac", type=float, default=0.005)
    ap.add_argument("--k-fracs", type=float, nargs="+")
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-pairs", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--refresh-acts", action="store_true")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    ap.add_argument("--cpu-probe", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.set_defaults(require_gpu=True, control="topic")
    args = ap.parse_args()

    tcs = load_tcs()
    tcs.OUT_DIR = Path(args.out_dir)
    tcs.ACTS_DIR = Path(args.acts_dir)
    tsc = tcs.load_syntax_module()
    tsc.PROBE_GPU = False if args.cpu_probe else torch.cuda.is_available()
    tsc.PROBE_KIND = "linear"
    for m in args.models:
        try:
            run_model(tcs, tsc, m, args)
        except Exception as e:
            print(f"\n  {m} failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
