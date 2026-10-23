#!/usr/bin/env python3
"""
CPU self-test for scripts/continent_control.py. No model, no GPU.

  1. Label construction on statements shaped exactly like data/statements:
     the named country is parsed from both "is in" and "is not in" forms, the
     city continent comes from the pair's correct country, and every class maps.
  2. End to end through the thesis machinery (test_concept_subspace.run_cell,
     the same probe, folds, retention and bootstrap) on synthetic activations
     with a planted answer:
        A  truth in the TAIL, continent in the HEAD  -> Delta clearly > 0
        B  truth and continent BOTH in the tail      -> Delta near 0
     and the leakage detector must sit near chance in both.
  3. summarize runs on the files written in step 2.

    python tests/test_continent_control.py
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

spec = importlib.util.spec_from_file_location("continent_control",
                                              REPO / "scripts" / "continent_control.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from linear_rep_geometry.behavioral import StatementPair, load_pairs  # noqa: E402

COUNTRIES = ["Russia", "South Africa", "the United States", "Japan", "Brazil",
             "France", "Nigeria", "India", "Germany", "Mexico", "China", "Kenya",
             "Australia", "the Philippines", "Côte d'Ivoire", "Peru"]


def fake_pairs(n=200, negated=False, seed=0):
    """Pairs in the build_datasets.py format: same city, true vs false country."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        correct, wrong = rng.choice(COUNTRIES, 2, replace=False)
        city = f"City{i}"
        if negated:   # "is not in <wrong>" is TRUE
            pos = f"The city of {city} is not in {wrong}."
            neg = f"The city of {city} is not in {correct}."
        else:
            pos = f"The city of {city} is in {correct}."
            neg = f"The city of {city} is in {wrong}."
        recs.append({"concept": "truth_cities", "positive": pos, "negative": neg,
                     "matched": True, "pair_id": f"cit_{i:05d}", "subject": city,
                     "topic": correct})
    return recs


def test_labels():
    cmap = cc.load_continents()
    assert cmap["Australia"] == "Asia", "Oceania must be folded into Asia"
    assert cmap["Russia"] == "Europe" and cmap["Turkey"] == "Asia"
    for negated in (False, True):
        recs = fake_pairs(negated=negated)
        pairs = [StatementPair(concept=r["concept"], positive=r["positive"],
                               negative=r["negative"], matched=True,
                               meta={"topic": r["topic"], "subject": r["subject"]})
                 for r in recs]
        named, city, classes = cc.continent_labels(pairs, cmap)
        n = len(pairs)
        assert len(named) == len(city) == 2 * n
        assert (city[:n] == city[n:]).all(), "city continent must be shared by a pair"
        # the named country of each statement must be the one in its text
        for j, p in enumerate(pairs):
            for text, lab in ((p.positive, named[j]), (p.negative, named[n + j])):
                assert classes[lab] == cmap[cc.named_country(text)]
        lk = cc.leakage(city, np.r_[np.ones(n), np.zeros(n)].astype(int))
        assert lk["cramers_v"] < 1e-9, "city continent must carry no truth"
    print("  labels: named / city continent parsed, folded, leak-free   ok")


def write_case(tmp, recs, plant):
    """Synthetic d=32 basis and cached activations with a planted structure."""
    d, n_hs, layers = 32, 5, (2, 4)
    g = torch.Generator().manual_seed(7)
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    evals = torch.logspace(0, -2, d)
    cov = U @ torch.diag(evals) @ U.T
    cov_path = tmp / "cov.pt"
    torch.save(cov, cov_path)

    cmap = cc.load_continents()
    pairs = [StatementPair(concept="t", positive=r["positive"], negative=r["negative"],
                           matched=True, meta={"topic": r["topic"]}) for r in recs]
    named, _, classes = cc.continent_labels(pairs, cmap)
    n = len(pairs)
    truth = np.r_[np.ones(n), np.zeros(n)]

    # background: isotropic noise in the eigenbasis, so every window has variance
    Z = torch.randn(2 * n, d, generator=g) * 0.3
    head, tail = U[:, :3], U[:, -3:]
    truth_dir = tail[:, 0]
    cont_basis = head if plant == "A" else tail[:, 1:]
    onehot = torch.tensor(np.eye(len(classes))[named], dtype=torch.float32)
    proj = torch.randn(len(classes), cont_basis.shape[1], generator=g)
    X = (Z + 1.5 * torch.tensor(truth - 0.5, dtype=torch.float32)[:, None] * truth_dir
         + 1.5 * (onehot @ proj) @ cont_basis.T)

    acts = {li: X.clone() for li in layers}
    acts_dir = tmp / "acts"
    acts_dir.mkdir(exist_ok=True)
    torch.save({"acts": acts, "n_layers": n_hs, "d": d, "model": "fake/Fake-1B",
                "concept": "truth_cities", "pooling": "last_token", "n_pairs": n},
               acts_dir / "acts_Fake-1B__truth_cities__last_token__L2-4.pt")
    return cov_path, acts_dir, n_hs


def run_case(plant):
    tcs, tsc = cc.load_modules()
    tsc.PROBE_GPU = False
    tsc.PROBE_KIND = "linear"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        recs = fake_pairs(n=240, seed=3)
        (tmp / "statements").mkdir()
        with open(tmp / "statements" / "truth_cities.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        cov_path, acts_dir, n_hs = write_case(tmp, recs, plant)

        tcs.STATEMENTS = tmp / "statements"
        tcs.ACTS_DIR = acts_dir
        tcs.OUT_DIR = tmp / "out"
        tcs.depth_from_config = lambda _m: n_hs
        args = SimpleNamespace(out_dir=str(tmp / "out"), acts_dir=str(acts_dir),
                               cov=str(cov_path), layers=["mid", "final"],
                               pooling="last_token", k_frac=0.1, k_fracs=None,
                               n_windows=5, n_boot=300, max_pairs=512,
                               batch_size=8, max_length=64, refresh_acts=False,
                               require_gpu=False, skip_existing=False,
                               control="named_continent")
        cc.run_dataset(tcs, tsc, "fake/Fake-1B", "truth_cities", args,
                       cc.load_continents())
        out = json.load(open(tmp / "out" / "Fake-1B__truth_cities.json"))
        cell = out["by_layer"]["2"]["by_control"]["named_continent"]
        did = cell["difference_in_differences"]
        leak = cell["full_accuracy"]["pair_id_random"]

        cc.cmd_summarize(SimpleNamespace(out_dir=str(tmp / "out")))
        assert (tmp / "out" / "summary.md").exists()
        return did, leak


def test_end_to_end():
    did_a, leak_a = run_case("A")
    print(f"  case A (truth tail, continent head): Delta {did_a['delta']:+.3f} "
          f"CI [{did_a['ci95'][0]:+.3f}, {did_a['ci95'][1]:+.3f}]  leak {leak_a:.3f}")
    did_b, leak_b = run_case("B")
    print(f"  case B (both in the tail):           Delta {did_b['delta']:+.3f} "
          f"CI [{did_b['ci95'][0]:+.3f}, {did_b['ci95'][1]:+.3f}]  leak {leak_b:.3f}")
    assert did_a["ci95"][0] > 0.2, "planted specificity not detected"
    assert abs(did_b["delta"]) < 0.15, "no specificity planted, but Delta is large"
    assert did_a["delta"] > did_b["delta"] + 0.3
    for lk in (leak_a, leak_b):
        assert abs(lk - 0.5) < 0.08, f"leakage detector off chance: {lk}"
    print("  end to end through run_cell + summarize                ok")


if __name__ == "__main__":
    print("continent_control self-test")
    test_labels()
    test_end_to_end()
    print("all passed")
