#!/usr/bin/env python3
"""
CPU self-test for scripts/causal_ablation.py. No download, no GPU: a tiny
randomly initialised Llama and a word-level tokenizer are built in memory.

  1. The hook removes exactly the chosen subspace from hidden state i (the vector
     the probes read), leaves the model unchanged with no subspace, and removing
     the hook restores the original logits bit for bit.
  2. Continuation log-probabilities equal a hand computation.
  3. causal_delta and the gates: a planted case (tail removal destroys truth,
     not general prediction) gives a large positive Delta; a null case gives ~0;
     an ablation that changes nothing fails the sensitivity gate.
  4. run_loaded + summarize run end to end on the tiny model.

    python tests/test_causal_ablation.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("causal_ablation",
                                              REPO / "scripts" / "causal_ablation.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

D = 32
WORDS = ("The city of is in Paris Lyon Tokyo Osaka Lagos Lima Cairo Russia France "
         "Japan Nigeria Peru Egypt South Africa the United States a an and to was "
         "he she it they river book stone light house").split()


def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>",
                                  unk_token="<unk>", bos_token="<s>", eos_token="<s>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=D, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval(), tok


def basis():
    g = torch.Generator().manual_seed(1)
    U, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    return torch.logspace(0, -2, D), U


def test_hook():
    model, tok = tiny()
    _, U = basis()
    ids = torch.tensor([tok("The city of Paris is in France").input_ids])
    layers = ca.decoder_layers(model)
    hs = (len(layers) + 1) // 2
    # What matters is what the NEXT layer receives. (transformers 5 records
    # hidden_states with its own earlier hook, so the recorded copy is pre-edit;
    # the edited tensor is the one that propagates.)
    cap = {}

    def grab(mod, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        cap["in"] = x.detach().clone()
    pre = layers[hs].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        ref = model(ids, output_hidden_states=True)
        ref_in = cap["in"]
        S = U[:, :8]
        h = ca.ProjectOut(S).attach(layers[hs - 1])
        abl = model(ids)
        abl_in = cap["in"]
        h.remove()
        again = model(ids).logits
        noop = ca.ProjectOut(None).attach(layers[hs - 1])
        same = model(ids).logits
        noop.remove()
    pre.remove()
    assert torch.allclose(ref_in, ref.hidden_states[hs], atol=1e-6), \
        "hook is not on the layer whose output is hidden state i"
    resid = (abl_in @ S).abs().max().item()
    expect = ref_in - (ref_in @ S) @ S.T
    assert resid < 1e-5, f"subspace not removed: {resid}"
    assert torch.allclose(abl_in, expect, atol=1e-5)
    assert torch.equal(again, ref.logits), "hook removal did not restore the model"
    assert torch.equal(same, ref.logits), "empty hook changed the model"
    assert not torch.allclose(abl.logits, ref.logits), "ablation had no effect"
    print("  hook removes exactly the subspace at hidden state i; detach restores   ok")


def test_logprobs():
    model, tok = tiny()
    items = [("The city of Paris is in", " France", " Japan"),
             ("The city of Lagos is in", " Nigeria", " South Africa")]
    seqs, starts, mism = ca.encode_items(tok, items)
    assert mism == 0 and len(seqs) == 4
    got = ca.continuation_logprobs(model, tok, seqs, starts, batch_size=3)
    for s, st, g in zip(seqs, starts, got):
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([s])).logits[0].float(), -1)
        want = sum(float(lp[p - 1, s[p]]) for p in range(st, len(s)))
        assert abs(g - want) < 1e-4, (g, want)
    print("  continuation log-probs (batched, padded) == hand computation          ok")


def fake(truth_acc, gen_acc, n=400, n_pass=10, n_tok=150, seed=0):
    rng = np.random.default_rng(seed)
    return {"truth_correct": (rng.random(n) < truth_acc).astype(float),
            "margin": rng.normal(size=n),
            "gen_correct": [rng.random(n_tok) < gen_acc for _ in range(n_pass)]}


def test_delta_and_gates():
    base = fake(0.90, 0.40, seed=1)
    # planted: removing the tail destroys truth, leaves general prediction alone
    head, tail = fake(0.88, 0.30, seed=2), fake(0.52, 0.30, seed=3)
    d = ca.causal_delta(base, head, tail, n_boot=500)
    assert d["ci95"][0] > 0.5, d
    # null: both removals cost both behaviours the same share
    base0 = fake(0.90, 0.40, n=6000, seed=1)
    head0, tail0 = fake(0.70, 0.20, n=6000, seed=4), fake(0.70, 0.20, n=6000, seed=5)
    d0 = ca.causal_delta(base0, head0, tail0, n_boot=500)
    assert abs(d0["delta"]) < 0.15 and d0["ci95"][0] < 0 < d0["ci95"][1], d0
    # sensitivity gate: an ablation that changes nothing must fail it
    cond = {n: ca.retentions(base, base) for n in ("head", "tail")}
    assert not ca.gates(base, cond)["ok"]
    cond = {"head": ca.retentions(base, head), "tail": ca.retentions(base, tail)}
    assert ca.gates(base, cond)["ok"]
    assert not ca.gates(fake(0.55, 0.4), cond)["ok"], "competence gate"
    print(f"  Delta planted {d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}], "
          f"null {d0['delta']:+.2f} [{d0['ci95'][0]:+.2f}, {d0['ci95'][1]:+.2f}]; gates   ok")


def test_end_to_end():
    model, tok = tiny()
    evals, U = basis()
    cities = [("Paris", "France", "Japan"), ("Lyon", "France", "Peru"),
              ("Tokyo", "Japan", "Egypt"), ("Osaka", "Japan", "Russia"),
              ("Lagos", "Nigeria", "Peru"), ("Lima", "Peru", "France"),
              ("Cairo", "Egypt", "Japan")] * 4
    items = [(f"The city of {c} is in", f" {a}", f" {b}") for c, a, b in cities]
    texts = ["the river was light and the stone house was in Paris and he was in it"] * 4
    args = SimpleNamespace(layers=["mid", "final"], k_fracs=[0.25, 0.5], n_random=2,
                           n_boot=200, batch_size=8, seed=0)
    res = ca.run_loaded(model, tok, "Tiny-Llama", evals, U, "cov:fake", items, texts, args)
    blk = res["by_layer"]["mid"]["by_k"]["0.5"]
    assert set(blk["conditions"]) >= {"head", "tail", "random0", "random1", "random_mean"}
    assert "gates" in blk and "difference_in_differences" in blk
    with tempfile.TemporaryDirectory() as t:
        for name in ("A", "B", "gemma-2-2b"):
            r = dict(res, model_name=name)
            json.dump(r, open(Path(t) / f"{name}.json", "w"), default=float)
        ca.cmd_summarize(SimpleNamespace(out_dir=t))
        assert (Path(t) / "summary.md").exists()
    print("  run_loaded + summarize end to end on a tiny Llama                    ok")


if __name__ == "__main__":
    print("causal_ablation self-test")
    test_hook()
    test_logprobs()
    test_delta_and_gates()
    test_end_to_end()
    print("all passed")
