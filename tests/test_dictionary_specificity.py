#!/usr/bin/env python3
"""
CPU self-test for scripts/dictionary_specificity.py. No model, no GPU, no download.

    python tests/test_dictionary_specificity.py

Checks:
  1. the token rules reproduce artifact_check_unembed.filter_single_token_pairs;
  2. planted concepts in a synthetic unembedding are called correctly by the
     lexical null (head > 0, tail < 0, arbitrary ~ 0);
  3. the published-style random-token-pair null reproduces
     fix_hole5_unembed_null.random_unembed_null exactly (same seed, same Ginis);
  4. summarize runs end to end and applies the frozen decision rule.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from linear_rep_geometry.spectral_metrics import (  # noqa: E402
    compute_cumulative_variance, random_direction_baseline)

spec = importlib.util.spec_from_file_location(
    "dictionary_specificity", REPO / "scripts" / "dictionary_specificity.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


# ── 1. token rules ───────────────────────────────────────────────────────────
def test_token_rules():
    toks = {"cat": [5], " cat": [6], "dogs": [1, 2], " dogs": [7],
            "zebra": [3, 4], " zebra": [8, 9]}
    enc = lambda s: toks[s]  # noqa: E731
    assert ds.word_token_id(enc, "cat", "published") == 5      # bare first
    assert ds.word_token_id(enc, "cat", "space_first") == 6    # space first
    assert ds.word_token_id(enc, "dogs", "published") == 7     # falls back to " dogs"
    assert ds.word_token_id(enc, "zebra", "published") is None
    concepts = {"c": [("cat", "dogs"), ("cat", "zebra"), ("dogs", "cat"),
                      ("cat", "cat"), ("dogs", "cat"), ("cat", "dogs")]}
    clean, w2i, rep = ds.clean_concepts(concepts, enc, "published", min_pairs=4)
    assert rep["c"] == {"n_pairs": 6, "n_clean": 4}            # zebra and identical dropped
    assert "zebra" not in w2i
    print("  token rules match the published single-token filter   ok")


# ── synthetic unembedding ────────────────────────────────────────────────────
def synthetic(d=48, V=3000, n_words=500, n_pairs=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    evals = torch.logspace(0, -3, d)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    scale = evals.sqrt()
    W = (torch.randn(V, d, generator=g) * scale) @ U.T          # "vocabulary" rows
    # "real words" are rows 0..n_words-1 and vary more in the upper spectrum than
    # the junk tokens, so real-word contrasts differ from arbitrary-token contrasts.
    boost = torch.ones(d)
    boost[3:10] = 2.5                                           # real words vary more
    W[:n_words] = (torch.randn(n_words, d, generator=g) * scale * boost) @ U.T  # in the upper spectrum
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(n_words)]
    w2i = {w: i for i, w in enumerate(words)}
    nxt = n_words + 1000
    concepts = {}

    def planted(name, direction, size):
        nonlocal nxt
        base = rng.choice(n_words, n_pairs, replace=False)
        pairs = []
        for b in base:
            W[nxt] = W[b] + size * direction + 0.3 * ((torch.randn(d, generator=g) * scale) @ U.T)
            w = f"{name}{nxt}"
            w2i[w] = nxt
            pairs.append((words[b], w))
            nxt += 1
        concepts[name] = pairs

    head = U[:, :3].sum(1)
    tail = U[:, -6:].sum(1)
    planted("head", head / head.norm(), 3.0)
    planted("tail", tail / tail.norm(), 0.5)
    idx = rng.choice(n_words, 2 * n_pairs, replace=False)
    concepts["none"] = [(words[idx[i]], words[idx[n_pairs + i]]) for i in range(n_pairs)]
    return W, w2i, concepts, evals, U


# ── 2 + 3. planted concepts ──────────────────────────────────────────────────
def test_planted():
    W, w2i, concepts, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    iso = random_direction_baseline(W.shape[1], U, cumvar, n_samples=500)
    pair = ds.published_token_pair_null(W, U, cumvar, n=500)
    rows = {r["concept"]: r for r in ds.analyse(
        W, W.shape[0], concepts, w2i, U, cumvar, n_null=400, seed=0,
        iso_mean=iso["gini_mean"], pair_null=pair)}
    for k in ("head", "tail", "none"):
        r = rows[k]
        print(f"  {k:4s} Gini {r['gini']:+.3f}  lex {r['gap_lexical']:+.3f} (p {r['p_lexical']:.3f})"
              f"  vocab_n {r['gap_vocab_n']:+.3f}  token_pair {r['gap_token_pair']:+.3f}")
    assert rows["head"]["gap_lexical"] > 0.05 and rows["head"]["p_lexical"] < 0.01
    assert rows["tail"]["gap_lexical"] < -0.05 and rows["tail"]["p_lexical"] < 0.01
    assert abs(rows["none"]["gap_lexical"]) < 0.05 and rows["none"]["p_lexical"] > 0.05
    print("  head / tail / none called correctly by the lexical null   ok")


# ── 4. summarize ─────────────────────────────────────────────────────────────
def test_summarize():
    W, w2i, concepts, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    pair = ds.published_token_pair_null(W, U, cumvar, n=200)
    with tempfile.TemporaryDirectory() as tmp:
        ds.OUT_ROOT = Path(tmp)
        ds.ACT_ROOT = Path(tmp) / "none"
        ds.MIN_CONCEPTS = 1                  # fake models carry one concept each"
        folder = Path(tmp) / "language_published"
        folder.mkdir()
        models = []
        for k in range(5):
            rows = ds.analyse(W, W.shape[0], concepts, w2i, U, cumvar,
                              n_null=200, seed=k, iso_mean=-0.1, pair_null=pair)
            m = f"fake-{k}"
            models.append(m)
            rows = [r for r in rows if r["concept"] == "none"]
            json.dump({"model": m, "n_concepts_total": 1,
                       "summary": ds.model_summary(rows), "concepts": rows},
                      open(folder / f"{m}.json", "w"))

        class A:
            pass
        a = A()
        a.models, a.basis, a.rule = models, "language", "published"
        ds.cmd_summarize(a)
        out = json.load(open(folder / "summary.json"))
        assert out["tests"]["lexical|all"]["n"] == 5
        assert out["verdict"] == "not testable"     # 5 models < MIN_MODELS (6)
        json.dump({"model": "fake-5", "n_concepts_total": 1,
                   "summary": ds.model_summary(rows), "concepts": rows},
                  open(folder / "fake-5.json", "w"))
        a.models = models + ["fake-5"]
        ds.cmd_summarize(a)
        out = json.load(open(folder / "summary.json"))
        assert out["verdict"] in ("not specific", "inconclusive"), out["verdict"]
    print("  summarize end to end, validity gate and verdict        ok")


def test_published_pair_null_matches_original():
    """Re-implements fix_hole5_unembed_null.random_unembed_null line by line."""
    W, _, _, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    rng = np.random.default_rng(42)
    ref = []
    for _ in range(300):
        t1, t2 = rng.choice(W.shape[0], size=2, replace=False)
        v = (W[t1] - W[t2]).float()
        v = v / v.norm()
        proj = U.T @ v
        cdf = torch.cumsum(proj ** 2 / (v @ v), dim=0).numpy()
        ref.append(float(np.trapezoid(cdf - cumvar, cumvar)))
    got = ds.published_token_pair_null(W, U, cumvar, n=300)
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - np.array(ref)).max()
    print("  published token-pair null == fix_hole5 implementation   ok")


def test_verdict_rule():
    def t(lo, hi, eq):
        return {"test": {"ci95": [lo, hi]}, "tost": {"equivalent": eq}}
    assert ds.verdict(t(.01, .2, False), t(.02, .3, False)) == "specific (head)"
    assert ds.verdict(t(-.01, .02, True), t(-.02, .02, True)) == "not specific"
    assert ds.verdict(t(-.3, -.1, False), t(-.3, -.05, False)) == "reversed (tail)"
    assert ds.verdict(t(.01, .2, False), t(-.01, .3, False)) == "inconclusive"
    print("  decision rule                                          ok")


if __name__ == "__main__":
    print("dictionary_specificity self-test")
    test_token_rules()
    test_planted()
    test_published_pair_null_matches_original()
    test_verdict_rule()
    test_summarize()
    print("all passed")
