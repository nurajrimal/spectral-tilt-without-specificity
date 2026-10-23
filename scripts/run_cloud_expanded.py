#!/usr/bin/env python3
"""Self-contained cloud extraction for the rebuttal experiments.

For each model, from scratch (needs only the repo + the model — NO pre-existing
artifacts): computes the en-fr language-subset unembedding covariance the SAME way
perform_raid.py does, then extracts ALL concept directions (27 original + 15 new
BATS semantic), writing cov_src.pt + v_raw_*.pt into results/<model>/en-fr/.

This makes the whole pipeline runnable on a fresh cloud GPU box with no dependency
on the laptop's saved covariances. Run analyze_expanded_semantic.py afterwards.

  --centered   ALSO save cov_src_centered.pt (mean-subtracted) → lets the analyzer
               report the centered-covariance variant (answers R2's centering point).

Usage:
    MODEL_LIST="google/gemma-2-2b,meta-llama/Llama-3.2-3B,..." \
        python scripts/run_cloud_expanded.py [--centered]
    python scripts/run_cloud_expanded.py --models google/gemma-2-2b [--force]
"""
import argparse
import gc
import glob
import json
import os

import torch
import linear_rep_geometry as lrg

torch.set_grad_enabled(False)

PAIR = "en-fr"
PAIR_FILE = "data/paired_contexts/en-fr.jsonl"


def load_paired_data(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_vocab_subset(pairs, lang_key, tokenizer, handler, src_lang, tgt_lang):
    token_set = set()
    for p in pairs:
        for text in p[lang_key]:
            formatted = handler.format_input(text, source_lang=src_lang,
                                             target_lang_code=tgt_lang, tokenizer=tokenizer)
            token_set.update(tokenizer.encode(formatted, add_special_tokens=False))
    return list(token_set)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process(model_path, centered, force):
    name = model_path.split("/")[-1]
    out = os.path.join("results", name, PAIR)
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {name} → {out} ===", flush=True)
    model, tok, handler = lrg.load_model(model_path)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_U = lrg.get_lm_head(model).weight.detach().to(torch.float32)

    # ── covariance from the en-fr source-token subset (matches perform_raid) ──
    cov_path = os.path.join(out, "cov_src.pt")
    if force or not os.path.exists(cov_path):
        pairs = load_paired_data(PAIR_FILE)
        lang0, lang1 = pairs[0].get("lang0", "src"), pairs[0].get("lang1", "tgt")
        vocab = get_vocab_subset(pairs, "contexts0", tok, handler, lang0, lang1)
        cov = lrg.compute_covariance_matrix(W_U, token_indices=vocab).cpu()
        torch.save(cov, cov_path)
        print(f"  cov_src {tuple(cov.shape)} over {len(vocab)} tokens", flush=True)
        if centered:
            sub = W_U[vocab]
            sub = sub - sub.mean(dim=0, keepdim=True)
            cov_c = ((sub.T @ sub) / sub.shape[0]).cpu()
            torch.save(cov_c, os.path.join(out, "cov_src_centered.pt"))
            print("  cov_src_centered saved", flush=True)

    # ── extract ALL concept vectors ──
    done, skipped, zero = 0, 0, []
    for cf in sorted(glob.glob("data/word_pairs/*.txt")):
        cname = os.path.basename(cf)[:-4]
        vpath = os.path.join(out, f"v_raw_{cname}.pt")
        if os.path.exists(vpath) and not force:
            skipped += 1
            continue
        bw, tw = lrg.get_counterfactual_pairs(cf, tok)
        if not bw:
            zero.append(cname)
            continue
        v, _ = lrg.concept_direction(bw, tw, model, tok, handler, device)
        v = v.detach()
        if float(torch.norm(v)) == 0.0:
            zero.append(cname)
            continue
        torch.save(v.cpu(), vpath)
        done += 1
    print(f"  concepts: extracted {done}, skipped {skipped}, zero-norm {len(zero)}: {zero}", flush=True)
    del model, W_U
    cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--centered", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    models = args.models or [m.strip() for m in os.environ.get("MODEL_LIST", "").split(",") if m.strip()]
    if not models:
        print("No models (use --models or MODEL_LIST).")
        return
    print(f"Cloud extraction: {len(models)} models, centered={args.centered}")
    for mp in models:
        try:
            process(mp, args.centered, args.force)
        except Exception as e:
            print(f"  ✗ {mp} FAILED: {e}", flush=True)
            cleanup()


if __name__ == "__main__":
    main()
