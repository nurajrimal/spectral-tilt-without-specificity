#!/usr/bin/env python3
"""CPU self-test for scripts/behaviour_link.py.   python tests/test_behaviour_link.py

1. a truth direction built to point along the report-word contrast is LINKED;
   an unrelated one is not;
2. the sign-flip null: pair differences with a shared component sit far from
   the null, pure-noise differences do not;
3. a vocabulary contrast planted in the head is head-ward against the lexical null;
4. both decision rules.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from linear_rep_geometry.spectral_metrics import compute_cumulative_variance  # noqa: E402

spec = importlib.util.spec_from_file_location("behaviour_link", REPO / "scripts" / "behaviour_link.py")
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)


def setup(d=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    evals = torch.logspace(0, -3, d)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    scale = evals.sqrt()
    draw = lambda k: (torch.randn(k, d, generator=g) * scale) @ U.T  # noqa: E731
    return g, evals, U, draw


def test_link_and_dual():
    g, evals, U, draw = setup()
    cumvar = compute_cumulative_variance(evals)
    rng = np.random.default_rng(0)
    pool = draw(600)
    head = U[:, :3].sum(1)
    head = head / head.norm()
    # vocabulary side: 8 "report" pairs differing along the head
    base = draw(8)
    u = (base + 3.0 * head).mean(0) - base.mean(0) + 0.05 * draw(1)[0]
    U_null = bl.lexical_null_directions(pool, 8, 400, rng)
    vs = bl.vocab_side(u, U_null, U, cumvar)
    assert vs["gap_vocab"] > 0.1 and vs["p_vocab"] < 0.01, vs
    # activation side: 200 pair differences sharing a component along u, plus noise
    D_linked = 0.6 * u / u.norm() + draw(200)
    D_noise = draw(200)
    a1 = bl.act_side(D_linked, u, U_null, U, cumvar, 400, np.random.default_rng(1))
    a0 = bl.act_side(D_noise, u, U_null, U, cumvar, 400, np.random.default_rng(1))
    print(f"  linked: link {a1['gap_link_word']:+.3f} (p {a1['p_link_word']:.3f})   "
          f"noise: link {a0['gap_link_word']:+.3f} (p {a0['p_link_word']:.3f})")
    assert a1["gap_link_word"] > 0.2 and a1["p_link_word"] < 0.01
    assert a0["p_link_word"] > 0.01
    assert a1["p_link_signflip"] < 0.01 and a0["p_link_signflip"] > 0.01
    print("  vocab head-ward; linked direction found, noise not       ok")


def test_sign_flip_null_centre():
    g, evals, U, draw = setup(seed=2)
    D = draw(300)
    Vb = bl.sign_flip_directions(D, 500, np.random.default_rng(0))
    assert Vb.shape == (500, D.shape[1])
    assert float(Vb.mean(0).norm()) < float(D.mean(0).norm()) * 3
    print("  sign-flip null shape and centre                          ok")


def test_verdicts():
    def t(lo, hi, eq):
        return {"test": {"ci95": [lo, hi]}, "tost": {"equivalent": eq}}
    pos, neg, eq0 = t(.05, .2, False), t(-.2, -.05, False), t(-.01, .01, True)
    assert bl.verdict_dual(pos, pos, neg, neg) == "dual"
    assert bl.verdict_dual(pos, pos, eq0, eq0) == "not dual"
    assert bl.verdict_dual(pos, t(-.01, .2, False), neg, neg) == "inconclusive"
    assert bl.verdict_link(pos, pos) == "linked"
    assert bl.verdict_link(neg, neg) == "reversed"
    assert bl.verdict_link(pos, eq0) == "no evidence"
    print("  decision rules                                           ok")


if __name__ == "__main__":
    print("behaviour_link self-test")
    test_link_and_dual()
    test_sign_flip_null_centre()
    test_verdicts()
    print("all passed")
