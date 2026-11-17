#!/usr/bin/env python3
"""
Last-token activations must not depend on the tokenizer's padding side.

    python tests/test_padding_side.py

Builds a tiny random causal LM and a word-level tokenizer offline, then checks
that layerwise_activations gives the same last-token activations for
  (a) one statement at a time (no padding),
  (b) batched with RIGHT padding,
  (c) batched with LEFT padding (gemma-2-2b's tokenizer default).
Before the fix, (c) read the wrong position for every statement shorter than
the longest one in its batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from linear_rep_geometry.behavioral import layerwise_activations  # noqa: E402


def tiny_model_and_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    words = "the city of paris is in france japan tokyo a very long statement indeed here".split()
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64)
    model = Qwen2ForCausalLM(cfg).eval()
    return model, tok


class Handler:
    def apply_final_norm(self, model, h):
        return model.model.norm(h)

    def _detect_device(self, model):
        return "cpu"


def test_padding_side():
    model, tok = tiny_model_and_tokenizer()
    h = Handler()
    texts = ["the city of paris is in france",
             "tokyo is in japan",
             "a very long statement indeed here the city of paris is in japan"]
    single = torch.cat([layerwise_activations([t], model, tok, h, "cpu", batch_size=1)["last_token"]
                        for t in texts])
    out = {}
    for side in ("right", "left"):
        tok.padding_side = side
        out[side] = layerwise_activations(texts, model, tok, h, "cpu", batch_size=3)["last_token"]
    for side, A in out.items():
        err = float((A - single).abs().max())
        print(f"  batched, {side:5s} padding vs unbatched: max |diff| = {err:.2e}")
        assert err < 1e-4, (side, err)
    print("  last-token activations independent of padding side      ok")


if __name__ == "__main__":
    print("padding-side self-test")
    test_padding_side()
    print("all passed")
