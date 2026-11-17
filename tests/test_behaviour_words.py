#!/usr/bin/env python3
"""CPU self-test for scripts/behaviour_words.py.   python tests/test_behaviour_words.py"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from linear_rep_geometry.spectral_metrics import compute_cumulative_variance  # noqa: E402

spec = importlib.util.spec_from_file_location("behaviour_words", REPO / "scripts" / "behaviour_words.py")
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)


def test_extract():
    rows = [
        {"negative": "The city of Paris is in Japan.", "positive": "The city of Paris is in France."},
        {"negative": "The city of Tokyo is in France.", "positive": "The city of Tokyo is in Japan."},
        {"negative": "The city of Lima is not in Peru.", "positive": "The city of Lima is not in Chile."},
        {"negative": "Answer: (A)", "positive": "Answer: (B)"},                      # letters only
        {"negative": "The city of X is in New Zealand.", "positive": "The city of X is in Spain."},
        {"negative": "Same text.", "positive": "Same text."},
    ]
    pairs, st = bw.extract(rows)
    assert ("Japan", "France") in pairs and ("France", "Japan") in pairs
    assert ("Peru", "Chile") in pairs
    assert st["single_word_substitution"] == 3, st
    assert st["several_words_differ"] == 2 and st["no_difference"] == 1, st
    print("  substitution pairs extracted, multi-word and letter-only pairs skipped   ok")


def test_gaps():
    g = torch.Generator().manual_seed(0)
    d = 32
    evals = torch.logspace(0, -3, d)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    X = (torch.randn(200, d, generator=g) * evals.sqrt()) @ U.T
    words = [f"w{i}" for i in range(200)]
    head = U[:, :2].sum(1)
    X[100:130] = X[:30] + 3.0 * head / head.norm()                  # planted relation
    concepts = {"rel": [(words[i], words[100 + i]) for i in range(30)],
                "none": [(words[30 + i], words[60 + i]) for i in range(30)]}
    vocab = {w: i for i, w in enumerate(words)}
    out = bw.gaps(X, vocab, concepts, U, compute_cumulative_variance(evals), n_null=300)
    print(f"  rel  rep {out['rel']['gap_repartition']:+.3f} (p {out['rel']['p_repartition']:.3f})"
          f"   none rep {out['none']['gap_repartition']:+.3f} (p {out['none']['p_repartition']:.3f})")
    assert out["rel"]["gap_repartition"] > 0.1 and out["rel"]["p_repartition"] < 0.05
    assert out["none"]["p_repartition"] > 0.05
    assert "gap_lexical" in out["rel"]
    print("  planted relation found, arbitrary pairs not                    ok")


if __name__ == "__main__":
    print("behaviour_words self-test")
    test_extract()
    test_gaps()
    print("all passed")
