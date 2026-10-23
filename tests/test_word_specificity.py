#!/usr/bin/env python3
"""
CPU self-test for scripts/word_specificity.py. No model, no GPU.

Builds a synthetic vocabulary whose activations share the basis's spectrum, then
plants three word concepts with a known answer:

    tail    target = base + a shift along the LOW-variance eigenvectors
    head    target = base + a shift along the HIGH-variance eigenvectors
    none    arbitrary word pairs, no shared shift

and checks that the lexical null calls each one correctly: tail clearly negative,
head clearly positive, none indistinguishable from zero. Also checks that the
vectorised Gini equals spectral_metrics.metrics exactly, and that summarize runs
end to end on fake per-model files.

    python tests/test_word_specificity.py
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
    compute_cumulative_variance, metrics, random_direction_baseline)

spec = importlib.util.spec_from_file_location(
    "word_specificity", REPO / "scripts" / "word_specificity.py")
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)


def synthetic(d=48, n_vocab=600, n_pairs=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    evals = torch.logspace(0, -3, d)                       # condition number 1e3
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    scale = evals.sqrt()

    def draw(k):
        return (torch.randn(k, d, generator=g) * scale) @ U.T

    X = draw(n_vocab)
    words = [f"w{i}" for i in range(n_vocab)]
    concepts = {}
    rng = np.random.default_rng(seed)

    def planted(name, direction, size):
        nonlocal X, words
        base = rng.choice(n_vocab, n_pairs, replace=False)
        noise = draw(n_pairs) * 0.3
        tgt = X[base] + size * direction + noise
        new = [f"{name}_t{i}" for i in range(n_pairs)]
        X = torch.cat([X, tgt], 0)
        words = words + new
        concepts[name] = [(words[b], new[i]) for i, b in enumerate(base)]

    tail_dir = U[:, -6:].sum(1)
    head_dir = U[:, :3].sum(1)
    planted("tail", tail_dir / tail_dir.norm(), 0.5)
    planted("head", head_dir / head_dir.norm(), 3.0)

    idx = rng.choice(n_vocab, 2 * n_pairs, replace=False)
    concepts["none"] = [(words[idx[i]], words[idx[n_pairs + i]])
                        for i in range(n_pairs)]

    vocab = {w: i for i, w in enumerate(words)}
    return X, vocab, concepts, evals, U


def test_batched_gini_matches_published():
    X, vocab, concepts, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    V = torch.randn(5, X.shape[1])
    got = ws.batched_gini(V, U, cumvar)
    want = np.array([metrics(v, U, cumvar)["gini"] for v in V])
    assert np.allclose(got, want, atol=1e-6), (got, want)
    # sign and scale invariance
    assert np.allclose(ws.batched_gini(-3.0 * V, U, cumvar), want, atol=1e-6)
    print("  batched Gini == spectral_metrics.metrics            ok")


def test_planted_concepts():
    X, vocab, concepts, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    iso = random_direction_baseline(X.shape[1], U, cumvar, n_samples=500)
    rows = {r["concept"]: r for r in ws.analyse_concepts(
        X, vocab, concepts, U, cumvar, n_null=500, seed=0,
        iso_mean=iso["gini_mean"])}

    for name in ("tail", "head", "none"):
        r = rows[name]
        print(f"  {name:4s} Gini {r['gini']:+.3f}  gap iso {r['gap_isotropic']:+.3f}  "
              f"gap lex {r['gap_lexical']:+.3f} (p {r['p_lexical']:.3f})  "
              f"gap rep {r['gap_repartition']:+.3f} (p {r['p_repartition']:.3f})")

    assert rows["tail"]["gap_lexical"] < -0.1 and rows["tail"]["p_lexical"] < 0.01
    assert rows["head"]["gap_lexical"] > 0.1 and rows["head"]["p_lexical"] < 0.01
    assert abs(rows["none"]["gap_lexical"]) < 0.05 and rows["none"]["p_lexical"] > 0.05
    # the point of the script: an arbitrary word contrast looks anti-concentrated
    # against the isotropic baseline but not against the lexical null
    assert rows["none"]["gap_isotropic"] > abs(rows["none"]["gap_lexical"])
    print("  tail / head / none called correctly by the lexical null   ok")
    return rows


def test_summarize_end_to_end():
    X, vocab, concepts, evals, U = synthetic()
    cumvar = compute_cumulative_variance(evals)
    with tempfile.TemporaryDirectory() as tmp:
        ws.OUT_ROOT = Path(tmp)
        folder = Path(tmp) / "language_final"
        folder.mkdir()
        models = []
        for k in range(4):
            rows = ws.analyse_concepts(X, vocab, {"none": concepts["none"]}, U,
                                       cumvar, n_null=200, seed=k, iso_mean=-0.1)
            m = f"fake-{k}"
            models.append(m)
            json.dump({"model": m, "basis_source": "cov:/x/cov_src.pt",
                       "isotropic_gini_mean": -0.1,
                       "summary": ws.model_summary(rows), "concepts": rows},
                      open(folder / f"{m}.json", "w"))

        class A:
            pass
        a = A()
        a.models, a.basis, a.layer, a.groups = models, "language", "final", False
        ws.cmd_summarize(a)
        out = json.load(open(folder / "summary.json"))
        key = "lexical|all|all"
        assert out["tests"][key]["n"] == 4
        assert (folder / "summary.md").exists() and (folder / "per_concept.csv").exists()
    print("  summarize end to end                                    ok")


if __name__ == "__main__":
    print("word_specificity self-test")
    test_batched_gini_matches_published()
    test_planted_concepts()
    test_summarize_end_to_end()
    print("all passed")
