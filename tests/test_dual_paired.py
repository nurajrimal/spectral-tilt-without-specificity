#!/usr/bin/env python3
"""
CPU self-test for scripts/dual_paired.py. No model, no GPU.

    python tests/test_dual_paired.py

1. identical inputs on both sides give d == 0 exactly for every concept
   (proves both sides use the same pool and the same null draws);
2. a concept planted toward the head on the vocabulary side only gives d > 0
   and an activation gap near 0;
3. the frozen decision rule;
4. summarize end to end.
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
from linear_rep_geometry.spectral_metrics import compute_cumulative_variance  # noqa: E402

spec = importlib.util.spec_from_file_location("dual_paired", REPO / "scripts" / "dual_paired.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)


def synthetic(d=48, n_words=400, n_pairs=30, seed=3):
    g = torch.Generator().manual_seed(seed)
    evals = torch.logspace(0, -3, d)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    scale = evals.sqrt()
    draw = lambda k: (torch.randn(k, d, generator=g) * scale) @ U.T  # noqa: E731
    Xv, Xa = draw(n_words), draw(n_words)
    words = [f"w{i:04d}" for i in range(n_words)]
    rng = np.random.default_rng(seed)
    head = U[:, :3].sum(1)
    head = head / head.norm()
    clean = {}
    # "rel": a head shift on the vocabulary side only
    base = rng.choice(n_words // 2, n_pairs, replace=False)
    tgt = base + n_words // 2
    Xv[tgt] = Xv[base] + 3.0 * head + 0.3 * draw(n_pairs)
    clean["rel"] = [(words[b], words[t]) for b, t in zip(base, tgt)]
    for k in range(3):                       # arbitrary concepts, no relation
        free = np.setdiff1d(np.arange(n_words), tgt)   # avoid the planted words
        idx = rng.choice(free, 2 * n_pairs, replace=False)
        clean[f"none{k}"] = [(words[idx[i]], words[idx[n_pairs + i]]) for i in range(n_pairs)]
    return Xv, Xa, words, clean, evals, U


def test_same_draws():
    Xv, _, words, clean, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    rows = dp.paired_rows(Xv, Xv.clone(), words, clean, U, cumvar, n_null=200, seed=0)
    assert all(r["d"] == 0.0 for r in rows), [r["d"] for r in rows]
    print("  identical inputs -> d == 0 exactly (same pool, same draws)   ok")


def test_planted():
    Xv, Xa, words, clean, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    rows = {r["concept"]: r for r in dp.paired_rows(Xv, Xa, words, clean, U, cumvar,
                                                     n_null=300, seed=0)}
    r = rows["rel"]
    print(f"  rel: gap_vocab {r['gap_vocab']:+.3f}  gap_act {r['gap_act']:+.3f}  d {r['d']:+.3f}")
    assert r["gap_vocab"] > 0.1 and abs(r["gap_act"]) < 0.06 and r["d"] > 0.1
    for k in range(3):                       # arbitrary concepts: not significant either side
        assert rows[f"none{k}"]["p_vocab"] > 0.05 and rows[f"none{k}"]["p_act"] > 0.05
    assert r["p_vocab"] < 0.01 and r["p_act"] > 0.05
    print("  vocabulary-only relation -> d > 0, activation gap ~ 0       ok")


def test_verdict():
    def t(lo, hi, eq):
        return {"test": {"ci95": [lo, hi]}, "tost": {"equivalent": eq}}
    pos, eq0 = t(.05, .2, False), t(-.02, .02, True)
    assert dp.verdict(pos, pos, eq0, eq0) == "asymmetry confirmed"
    assert dp.verdict(eq0, eq0, eq0, eq0) == "no asymmetry"
    assert dp.verdict(pos, pos, t(-.2, -.05, False), t(-.2, -.05, False)) == "inconclusive"
    assert dp.verdict(pos, t(-.01, .2, False), eq0, eq0) == "inconclusive"
    print("  decision rule                                             ok")


def test_summarize():
    Xv, Xa, words, clean, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    with tempfile.TemporaryDirectory() as tmp:
        dp.OUT_ROOT = Path(tmp)
        folder = Path(tmp) / "language_final"
        folder.mkdir()
        models = []
        for k in range(6):
            rows = dp.paired_rows(Xv, Xa, words, clean, U, cumvar, n_null=100, seed=k)
            m = f"fake-{k}"
            models.append(m)
            json.dump({"model": m, "means": dp.model_means(rows), "concepts": rows},
                      open(folder / f"{m}.json", "w"))

        class A:
            pass
        a = A()
        a.models, a.basis, a.layer = models, "language", "final"
        dp.cmd_summarize(a)
        out = json.load(open(folder / "summary.json"))
        assert out["tests"]["d|all"]["n"] == 6 and "verdict" in out
    print("  summarize end to end                                      ok")


if __name__ == "__main__":
    print("dual_paired self-test")
    test_same_draws()
    test_planted()
    test_verdict()
    test_summarize()
    print("all passed")
