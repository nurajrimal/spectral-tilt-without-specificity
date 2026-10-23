#!/usr/bin/env python3
"""
Rebuild the published language-subset covariance, Sigma_L.

The thesis measured everything in the full-vocabulary basis because the language
subset looked unavailable: the README records `--language-tokens` as unfilled and
treats the published Sigma_L as something that would have to be supplied from
outside. That turns out to be wrong, and the consequence is not small — without
Sigma_L the POS probe cannot be compared against the paper's table at all, so
every run of test_syntax_claim.py skips its reproduction gate and the whole
chapter rests on an unanchored measurement.

The subset is not an external token list. perform_raid.py derives it from
data/paired_contexts/en-fr.jsonl, which ships with the repository:

    vocab = union over all 264 pairs, over every context string, of
            tokenizer.encode(handler.format_input(text, src, tgt))
    Sigma_L = (W_U[vocab].T @ W_U[vocab]) / |vocab|

and that is saved as results/<model>/en-fr/cov_src.pt. Note what is NOT in it:
no lambda. compute_covariance_matrix adds no regularisation, and the POS probe
eigendecomposes the matrix as saved. The lambda the published code applies lives
in get_whitening_transform, which is a different code path and does not touch
this basis. So Sigma_L is the unregularised second moment — which is the same
lambda = 0 choice the thesis arrived at independently after lambda = 0.1
flattened the spectrum to a condition number of 1.38.

This script does exactly those three steps and nothing else. perform_raid.py
would also produce them, but as a side effect of the full RAID pipeline:
activations for every concept, whitening transforms, intervention vectors. That
is hours of work per model to obtain one matrix, and it writes into directories
the thesis results already occupy.

The published helpers are imported rather than reimplemented. The vocabulary
depends on the exact prompt formatting that handler.format_input applies, and a
near-copy that formatted differently would produce a subset that looks right,
builds a plausible Sigma, and is not the paper's basis — a failure that nothing
downstream would catch, because the gate would then be checking one wrong number
against another.

Once this has run, test_syntax_claim.py finds cov_src.pt through its own path
search, so the probe runs in the published basis with no flag:

    python scripts/build_language_cov.py --models Qwen/Qwen2.5-3B \
        meta-llama/Llama-3.2-3B google/gemma-2-2b
    python scripts/test_syntax_claim.py --model Qwen/Qwen2.5-3B

Run it on the three models the paper reports first. Those are the only ones whose
top-k and bottom-k accuracies appear in pos_probe_all_models.json, so they are
the only ones where the gate can fire.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_PAIR_FILE = REPO_ROOT / "data" / "paired_contexts" / "en-fr.jsonl"


def load_raid_module():
    """
    Import perform_raid.py for load_paired_data and get_vocab_subset.

    Importing is the point. get_vocab_subset calls handler.format_input with the
    source and target language codes before tokenising, and that formatting
    decides which token ids land in the subset. Reimplementing it from the
    description would be the one way to produce a Sigma that is wrong in a way
    the reproduction gate cannot detect.
    """
    path = SCRIPT_DIR / "perform_raid.py"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script rebuilds the published Sigma_L using "
            f"that file's own vocabulary logic; run it inside a checkout of "
            f"spectral_anti_concentration.")
    spec = importlib.util.spec_from_file_location("perform_raid", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["perform_raid"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_for_model(model_path, raid, args):
    import linear_rep_geometry as lrg

    name = model_path.split("/")[-1]
    out_dir = RESULTS_DIR / name / args.lang_pair
    out_path = out_dir / "cov_src.pt"

    if out_path.exists() and not args.force:
        print(f"  {name}: cov_src.pt already present, skipping (--force to rebuild)")
        return {"model": name, "status": "skipped"}

    print(f"\n{'=' * 66}\n  Sigma_L for {name}\n{'=' * 66}")

    pairs = raid.load_paired_data(str(args.pair_file))
    lang0 = pairs[0].get("lang0", "src")
    lang1 = pairs[0].get("lang1", "tgt")
    print(f"  {len(pairs)} pairs, {lang0} -> {lang1}")

    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    if args.require_gpu and device.type != "cuda":
        raise SystemExit(f"model landed on {device}, not cuda. Pass --allow-cpu "
                         f"if that is intended.")
    print(f"  device {device}")

    # Source-language subset only. cov_tgt is the French side and the POS probe
    # never reads it, so building it would cost a second pass over the contexts
    # for nothing.
    vocab = raid.get_vocab_subset(pairs, "contexts0", tokenizer, handler,
                                  src_lang=lang0, tgt_lang=lang1)
    print(f"  vocabulary: {len(vocab)} distinct token ids")

    W_U = lrg.get_lm_head(model).weight.detach().to(dtype=torch.float32)
    print(f"  lm_head {tuple(W_U.shape)}")
    if len(vocab) < 100:
        raise SystemExit(
            f"only {len(vocab)} tokens in the subset. Something is wrong with the "
            f"pair file or the formatting; a d x d second moment from that few "
            f"rows is rank deficient and the eigenbasis would be meaningless.")

    cov = lrg.compute_covariance_matrix(W_U, token_indices=vocab).cpu()

    # No lambda. See the module docstring: the published POS probe
    # eigendecomposes this matrix as saved.
    evals = torch.linalg.eigvalsh(cov.float())
    lo = float(evals.min().clamp(min=1e-30))
    cond = float(evals.max() / lo)
    d = cov.shape[0]
    k = max(1, int(d * 0.10))
    cumvar = torch.cumsum(torch.sort(evals, descending=True).values, dim=0)
    cumvar = (cumvar / cumvar[-1]).numpy()

    print(f"  d={d}  cond={cond:.3g}")
    print(f"  top-{k} variance {cumvar[k-1]:.1%}   "
          f"bottom-{k} variance {1 - cumvar[d-k]:.2%}")
    if evals.min() <= 0:
        print("  warning: a non-positive eigenvalue. The subset is smaller than d, "
              "so Sigma_L is rank deficient and the quiet end of the spectrum is "
              "numerically arbitrary. Report this; it bears directly on what a "
              "bottom-k comparison can mean.")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cov, out_path)
    meta = {"model": model_path, "lang_pair": args.lang_pair,
            "n_pairs": len(pairs), "vocab_size": len(vocab), "d": d,
            "condition_number": cond,
            "top_10pct_variance": float(cumvar[k - 1]),
            "bottom_10pct_variance": float(1 - cumvar[d - k]),
            "lambda": 0.0,
            "source": "rebuilt by build_language_cov.py from "
                      "perform_raid.get_vocab_subset + lrg.compute_covariance_matrix"}
    json.dump(meta, open(out_dir / "cov_src_provenance.json", "w"), indent=2)
    print(f"  wrote {out_path}")

    del model, W_U
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"model": name, "status": "built", **meta}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    ap.add_argument("--lang-pair", default="en-fr",
                    help="output subdirectory; must match what "
                         "test_syntax_claim.py looks for")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if cov_src.pt exists")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    ap.set_defaults(require_gpu=True)
    args = ap.parse_args()

    if not args.pair_file.exists():
        raise SystemExit(
            f"{args.pair_file} not found. The published language subset is derived "
            f"from this file; without it Sigma_L cannot be rebuilt and the "
            f"reproduction gate stays unavailable.")

    models = args.models or ([args.model] if args.model else [])
    if not models:
        ap.error("pass --model or --models")

    raid = load_raid_module()
    rows = []
    for m in models:
        try:
            rows.append(build_for_model(m, raid, args))
        except Exception as e:
            print(f"\n  {m} failed: {type(e).__name__}: {e}")
            rows.append({"model": m.split('/')[-1], "status": f"failed: {e}"})

    print(f"\n{'=' * 66}")
    for r in rows:
        extra = (f"  d={r['d']} vocab={r['vocab_size']} cond={r['condition_number']:.3g}"
                 if r.get("status") == "built" else "")
        print(f"  {r['model']:<20} {r['status']}{extra}")
    print(f"{'=' * 66}")
    print("\nNow rerun the probe with no --cov so it picks up the published basis:")
    for r in rows:
        if r.get("status") in ("built", "skipped"):
            print(f"  python scripts/test_syntax_claim.py --model <org>/{r['model']}")
    print("\nStep [7] should now report the gate instead of skipping it. A gap "
          "above 0.02 against the published top-k means the extraction path "
          "differs from the paper's, and that has to be resolved before the "
          "control result means anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
