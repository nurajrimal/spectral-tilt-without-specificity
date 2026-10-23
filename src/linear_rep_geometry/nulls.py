"""
Phase 2 — the null model.

Section 5 of the proposal commits to two nulls and to fixing the decision rule
before any real direction is analysed. That ordering is the methodological
contribution, so this module is written to be runnable, inspected and frozen
before Phase 1 output is ever read.

  norm-matched random   The published control, reproduced so the numbers sit
                        beside the paper's. It is the weaker null: a Gaussian
                        direction has none of the structure a statement-level
                        activation difference carries.

  label-shuffled        The stronger and more appropriate one. Pool the two
                        statement sets, repartition the labels at random,
                        recompute the difference of means. Prompts, length
                        distribution and template are identical; only the
                        alignment between labels and concept is destroyed. It
                        is what separates "refusal is encoded in the tail" from
                        "statement-level contrasts land in the tail for
                        uninteresting reasons", and nothing else in the design
                        can make that separation.

The two can disagree, and in a strongly anisotropic spectrum they routinely do.
freeze_decision_rule() records which one carries the claim, with a timestamp,
before results exist.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .spectral_metrics import (
    compute_concept_energy_cdf, compute_gini_deviation,
    compute_spectral_center_of_mass, empirical_pvalue, random_direction_baseline,
)


def _gini(v: torch.Tensor, evecs: torch.Tensor, cumvar: np.ndarray) -> float:
    return compute_gini_deviation(compute_concept_energy_cdf(v, evecs), cumvar)


# ─── Null 2: label shuffling ─────────────────────────────────────────────────

def label_shuffle_null(acts_pos: torch.Tensor, acts_neg: torch.Tensor,
                       eigenvectors: torch.Tensor, cumvar: np.ndarray,
                       n_perm: int = 1000, seed: int = 0,
                       matched: bool = False) -> dict:
    """
    Repartition the labels; recompute; repeat.

    For matched data the permutation that respects the design is a per-pair SIGN
    flip, not a free repartition: pair i contributes +(a_i - b_i) or -(a_i - b_i)
    with equal probability. A free repartition would break the pairing and give
    an optimistically wide null, making the real direction look more special
    than it is. matched=True selects the sign-flip version.

    Two-sided by default. The design genuinely admits either sign — H2 predicts
    refusal concentrates — so a one-sided test would prejudge the result the
    paired contrast was built to detect.
    """
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(acts_pos), len(acts_neg)

    if matched:
        if acts_pos.shape != acts_neg.shape:
            raise ValueError("matched=True needs row-aligned arrays")
        D = acts_pos - acts_neg
        obs_v = D.mean(dim=0)
    else:
        X = torch.cat([acts_pos, acts_neg], dim=0)
        obs_v = acts_pos.mean(dim=0) - acts_neg.mean(dim=0)

    obs_v = obs_v / obs_v.norm().clamp(min=1e-12)
    obs_gini = _gini(obs_v, eigenvectors, cumvar)
    obs_scm = compute_spectral_center_of_mass(
        compute_concept_energy_cdf(obs_v, eigenvectors), cumvar)

    null_gini = np.empty(n_perm)
    for i in range(n_perm):
        if matched:
            signs = torch.tensor(rng.choice([-1.0, 1.0], size=len(D)),
                                 dtype=torch.float32).unsqueeze(1)
            v = (D * signs).mean(dim=0)
        else:
            perm = rng.permutation(n_pos + n_neg)
            v = X[perm[:n_pos]].mean(dim=0) - X[perm[n_pos:]].mean(dim=0)
        v = v / v.norm().clamp(min=1e-12)
        null_gini[i] = _gini(v, eigenvectors, cumvar)

    return {
        "observed_gini": obs_gini, "observed_scm": obs_scm,
        "null_gini_mean": float(null_gini.mean()),
        "null_gini_std": float(null_gini.std()),
        "gap": float(obs_gini - null_gini.mean()),
        "p_two_sided": empirical_pvalue(obs_gini, null_gini, tail="two"),
        "p_less": empirical_pvalue(obs_gini, null_gini, tail="less"),
        "p_greater": empirical_pvalue(obs_gini, null_gini, tail="greater"),
        "null_gini_samples": null_gini, "n_perm": n_perm, "matched": matched,
    }


def both_nulls(acts_pos: torch.Tensor, acts_neg: torch.Tensor,
               eigenvectors: torch.Tensor, cumvar: np.ndarray,
               n_perm: int = 1000, n_random: int = 1000,
               matched: bool = False, seed: int = 0,
               random_cache: Optional[dict] = None) -> dict:
    """
    Both nulls for one (concept, model, layer, pooling, basis) cell.

    random_cache lets the expensive norm-matched baseline be computed once per
    (model, basis) and reused across every cell, which it can be because it does
    not depend on the concept.

    Report both gaps always. Agreement is evidence; disagreement is itself a
    finding about the spectrum and must not be resolved by quietly picking the
    null that gives the nicer number.
    """
    d = eigenvectors.shape[0]
    rnd = random_cache or random_direction_baseline(d, eigenvectors, cumvar,
                                                    n_samples=n_random, seed=seed)
    shuf = label_shuffle_null(acts_pos, acts_neg, eigenvectors, cumvar,
                              n_perm=n_perm, seed=seed, matched=matched)

    gap_rand = float(shuf["observed_gini"] - rnd["gini_mean"])
    p_rand = empirical_pvalue(shuf["observed_gini"], rnd["gini_samples"], tail="two")

    return {
        "observed_gini": shuf["observed_gini"], "observed_scm": shuf["observed_scm"],
        "random_null": {"gini_mean": rnd["gini_mean"], "gini_std": rnd["gini_std"],
                        "gap": gap_rand, "p_two_sided": p_rand},
        "shuffle_null": {k: v for k, v in shuf.items()
                         if not isinstance(v, np.ndarray)},
        "agree": bool((gap_rand < 0) == (shuf["gap"] < 0)
                      and (p_rand < 0.05) == (shuf["p_two_sided"] < 0.05)),
    }


# ─── Supporting diagnostics ──────────────────────────────────────────────────

def split_half_reliability(acts_pos: torch.Tensor, acts_neg: torch.Tensor,
                           n_splits: int = 100, seed: int = 0,
                           matched: bool = False) -> dict:
    """
    Cosine between directions estimated from disjoint halves.

    A gate, not a result. A direction whose split-half cosine is low does not
    have a Gini worth interpreting: what would be measured is the spectral
    placement of estimation noise, which in an anisotropic space is neither zero
    nor centred on the diagonal. Run this before quoting any gap.
    """
    rng = np.random.default_rng(seed)

    def est(a, b):
        v = (a - b).mean(dim=0) if matched else a.mean(dim=0) - b.mean(dim=0)
        return v / v.norm().clamp(min=1e-12)

    cos = np.empty(n_splits)
    for i in range(n_splits):
        if matched:
            perm = rng.permutation(len(acts_pos))
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            v1, v2 = est(acts_pos[h1], acts_neg[h1]), est(acts_pos[h2], acts_neg[h2])
        else:
            pp, pn = rng.permutation(len(acts_pos)), rng.permutation(len(acts_neg))
            v1 = est(acts_pos[pp[: len(pp) // 2]], acts_neg[pn[: len(pn) // 2]])
            v2 = est(acts_pos[pp[len(pp) // 2:]], acts_neg[pn[len(pn) // 2:]])
        cos[i] = float(torch.dot(v1, v2))

    return {"cosine_mean": float(cos.mean()), "cosine_std": float(cos.std()),
            "cosine_lo": float(np.percentile(cos, 2.5)),
            "cosine_hi": float(np.percentile(cos, 97.5)), "n_splits": n_splits}


def matched_vs_unmatched(acts_pos: torch.Tensor, acts_neg: torch.Tensor,
                         eigenvectors: torch.Tensor, cumvar: np.ndarray) -> dict:
    """
    How much does the pairing change the estimate?

    Truth sub-datasets are genuinely matched (same city, true vs false country);
    refusal is not (harmful and harmless instructions come from different
    corpora). Quantifying the divergence on the matched concept bounds what the
    unmatched estimator may be absorbing on the unmatched one.
    """
    if acts_pos.shape != acts_neg.shape:
        raise ValueError("needs row-aligned arrays")
    vm = (acts_pos - acts_neg).mean(dim=0)
    vm = vm / vm.norm().clamp(min=1e-12)
    vu = acts_pos.mean(dim=0) - acts_neg.mean(dim=0)
    vu = vu / vu.norm().clamp(min=1e-12)
    gm, gu = _gini(vm, eigenvectors, cumvar), _gini(vu, eigenvectors, cumvar)
    return {"cosine": float(torch.dot(vm, vu)), "gini_matched": gm,
            "gini_unmatched": gu, "gini_delta": gm - gu}


# ─── Freezing the decision rule ──────────────────────────────────────────────

def freeze_decision_rule(out_path: str | Path, rule: dict) -> dict:
    """
    Write the analysis plan to disk with a hash and a timestamp, once.

    Call this at the end of Phase 2, before Phase 1 output is analysed. Commit
    the file. Refuses to overwrite: a decision rule that can be edited after
    results are seen is not a decision rule, and the timestamped commit is what
    makes "pre-specified" checkable by someone who was not there.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. The rule is frozen. To change it, "
            f"record the change and the reason in a new file and report both."
        )
    payload = dict(rule)
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"decision rule frozen -> {out_path}\n  sha256 {payload['sha256'][:16]}")
    return payload


DEFAULT_RULE = {
    "primary_null": "label_shuffle",
    "secondary_null": "norm_matched_random",
    "primary_statistic": "gini_gap",
    "unit_of_analysis": "model",
    "alpha": 0.05,
    "tails": 2,
    "sesoi": -0.082,
    "equivalence_test": "TOST",
    "reporting_cell": {
        "pooling": "both; a conclusion is reported only if it holds under both",
        "layer": "trajectory across all layers; headline at the layer of peak "
                 "split-half reliability, chosen per model before Gini is read",
        "basis": "full_vocab primary; language and activation reported alongside",
    },
    "gates": {
        "split_half_cosine_min": 0.70,
        "note": "below this the direction is not interpreted for that cell",
    },
    "hypotheses": {
        "H0": "mean Gini gap for behavioural directions = 0",
        "H1": "mean Gini gap != 0",
        "secondary": "paired: refusal gap vs truth gap; behavioural vs word-level, "
                     "same model, same layer, same basis",
    },
}
