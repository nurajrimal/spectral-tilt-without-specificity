#!/usr/bin/env python3
"""Targeted extraction of the expanded-semantic concept vectors (rebuttal).

Extracts ONLY the 15 new BATS semantic concepts and writes v_raw_<name>.pt into
each model's existing results/<model>/en-fr/ directory (the pair spectral_analysis
reads). Does NOT touch the existing 27 concepts or the cov/X/Q artifacts, and does
NOT re-run the RAID pipeline. Cheap: one model load per model, ~50 pairs × 15
categories of context-free forward passes.

Usage (on the GPU box):
    MODEL_LIST="google/gemma-2-2b,meta-llama/Llama-3.2-3B,..." \
        uv run python scripts/extract_new_concepts.py
    # or:
    python scripts/extract_new_concepts.py --models google/gemma-2-2b meta-llama/Llama-3.2-3B
    # add --force to overwrite existing vectors; --pair to change target dir (default en-fr)
"""
import argparse
import gc
import os

import torch
import linear_rep_geometry as lrg

# The 15 expanded-semantic categories (BATS 3.0), by concept-file basename.
NEW_CONCEPTS = [
    "[country - language]", "[name - nationality]", "[name - occupation]",
    "[animal - young]", "[animal - sound]", "[animal - shelter]",
    "[hypernyms - animals]", "[hypernyms - misc]", "[hyponyms - misc]",
    "[meronyms - substance]", "[meronyms - member]",
    "[synonyms - intensity]", "[synonyms - exact]",
    "[antonyms - gradable]", "[antonyms - binary]",
]


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_for_model(model_path, pair, force):
    model_name = model_path.split("/")[-1]
    out_dir = os.path.join("results", model_name, pair)
    if not os.path.isdir(out_dir):
        print(f"  ! {model_name}: {out_dir} does not exist (run RAID first) — skipping")
        return
    print(f"\n=== {model_name} → {out_dir} ===")
    model, tokenizer, handler = lrg.load_model(model_path)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done, skipped, zero = [], [], []
    for name in NEW_CONCEPTS:
        out_path = os.path.join(out_dir, f"v_raw_{name}.pt")
        if os.path.exists(out_path) and not force:
            skipped.append(name)
            continue
        cfile = os.path.join("data", "word_pairs", f"{name}.txt")
        if not os.path.exists(cfile):
            print(f"  ! missing {cfile}")
            continue
        base_words, target_words = lrg.get_counterfactual_pairs(cfile, tokenizer)
        if not base_words:
            zero.append(name)
            continue
        v_raw, _ = lrg.concept_direction(base_words, target_words, model, tokenizer, handler, device)
        v_raw = v_raw.detach()
        if float(torch.norm(v_raw)) == 0.0:
            zero.append(name)   # multi-token → zero vector (as in App F); log and skip
            continue
        torch.save(v_raw.cpu(), out_path)
        done.append(name)
    print(f"  extracted {len(done)}, skipped {len(skipped)} (exist), zero-norm {len(zero)}: {zero}")

    del model
    cleanup_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="HF model paths; defaults to $MODEL_LIST (comma-separated)")
    ap.add_argument("--pair", default="en-fr", help="target pair dir (default en-fr)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = args.models
    if not models:
        env = os.environ.get("MODEL_LIST", "")
        models = [m.strip() for m in env.split(",") if m.strip()]
    if not models:
        print("No models given (use --models or set MODEL_LIST).")
        return

    print(f"Extracting {len(NEW_CONCEPTS)} new concepts for {len(models)} models "
          f"into '{args.pair}/' dirs.")
    for mp in models:
        try:
            extract_for_model(mp, args.pair, args.force)
        except Exception as e:
            print(f"  ✗ {mp} FAILED: {e}")
            cleanup_gpu()


if __name__ == "__main__":
    main()
