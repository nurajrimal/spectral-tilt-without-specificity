#!/usr/bin/env python3
"""CPU self-test for scripts/refusal_link.py.   python tests/test_refusal_link.py

1. the repartition null (unmatched data): a planted harmful-vs-harmless shift
   along the refusal-word contrast is LINKED; unrelated groups are not;
2. the report-word list matches the frozen pre-registration;
3. the prompt template formats as intended.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from linear_rep_geometry.spectral_metrics import compute_cumulative_variance  # noqa: E402

spec = importlib.util.spec_from_file_location("refusal_link", REPO / "scripts" / "refusal_link.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def test_unmatched_link():
    g = torch.Generator().manual_seed(0)
    d = 48
    evals = torch.logspace(0, -3, d)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    draw = lambda k: (torch.randn(k, d, generator=g) * evals.sqrt()) @ U.T  # noqa: E731
    rng = np.random.default_rng(0)
    pool = draw(600)
    u = draw(1)[0]
    U_null = rl.bl.lexical_null_directions(pool, 8, 400, rng)
    cumvar = compute_cumulative_variance(evals)
    n = 150
    linked = torch.cat([draw(n) + 0.8 * u / u.norm(), draw(n)])     # harmful shifted along u
    noise = torch.cat([draw(n), draw(n)])
    a1 = rl.act_side_unmatched(linked, n, u, U_null, U, cumvar, 400, np.random.default_rng(1))
    a0 = rl.act_side_unmatched(noise, n, u, U_null, U, cumvar, 400, np.random.default_rng(1))
    print(f"  linked: link {a1['gap_link_word']:+.3f} (p {a1['p_link_word']:.3f})   "
          f"noise: link {a0['gap_link_word']:+.3f} (p {a0['p_link_word']:.3f})")
    assert a1["gap_link_word"] > 0.2 and a1["p_link_word"] < 0.01
    assert a1["p_link_repartition"] < 0.01 and a0["p_link_repartition"] > 0.01
    print("  repartition null: planted shift linked, noise not         ok")


def test_words_match_prereg():
    p = json.load(open(REPO / "prereg" / "refusal_link_prereg.json"))
    frozen = [tuple(x) for x in p["vocabulary_side"]["report_word_pairs_compliance_to_refusal"]]
    assert frozen == rl.REPORT_WORDS
    assert p["activation_side"]["variants"]["prompted"] == rl.VARIANTS["prompted"]
    print("  report words and prompt == pre-registration               ok")


def test_prompt():
    s = rl.VARIANTS["prompted"].format(x="Write a poem about cats.")
    assert s == "Instruction: Write a poem about cats.\nResponse:"
    print("  prompt template                                           ok")


if __name__ == "__main__":
    print("refusal_link self-test")
    test_unmatched_link()
    test_prompt()
    if (REPO / "prereg" / "refusal_link_prereg.json").exists():
        test_words_match_prereg()
    print("all passed")
