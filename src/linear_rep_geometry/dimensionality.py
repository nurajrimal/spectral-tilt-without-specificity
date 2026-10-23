"""
Phase 4 — the one-dimensionality assumption.

"The refusal direction" presupposes the contrast is rank one. If it is not, the
scalar Gini of a mean vector describes one arbitrary vector inside a subspace
rather than where the subspace sits, and the headline number is answering a
question nobody asked. Engels et al. (2025) show some features are genuinely
multi-dimensional, so this is not a hypothetical.

This phase runs before the headline numbers are interpreted, not after as a
caveat. The procedure:

  1. PCA on the per-example difference vectors (already saved by Phase 1).
  2. Report variance explained by PC1, and the participation ratio as a
     continuous effective-rank measure that does not depend on a threshold.
  3. Where rank one fails, measure the spectral placement of the r-dimensional
     subspace instead, using subspace_metrics from spectral_metrics.

Refusal is reported in the literature as strikingly one-dimensional, so it
should pass. Whether truth is equally so across its four sub-datasets is open,
and the proposal treats the answer as a byproduct worth reporting either way.
"""

from __future__ import annotations

import numpy as np
import torch


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    (sum lambda)^2 / sum(lambda^2). Effective number of dimensions carrying
    variance, with no threshold to choose. 1.0 is exactly rank one; r for r
    equal eigenvalues.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    ev = ev[ev > 0]
    if len(ev) == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def analyze_rank(diffs: torch.Tensor, var_threshold: float = 0.50,
                 max_rank: int = 10, center: bool = True) -> dict:
    """
    PCA on per-example difference vectors D = [n, d].

    center=True removes the mean difference before the PCA, so the components
    describe variation AROUND the concept direction — "is there structure the
    mean misses" — rather than rediscovering the mean itself. Both are
    reported: `cos_pc1_mean` is the alignment between the uncentered top PC and
    the mean difference, and when that is near 1 the contrast is dominated by a
    single direction, which is the rank-one case.

    var_threshold is the fraction of variance PC1 must explain for the contrast
    to be treated as rank one. 0.50 is a deliberately permissive bar: the
    purpose is to catch concepts where the mean vector is clearly the wrong
    object, not to insist on perfect one-dimensionality.
    """
    D = diffs.float()
    n, d = D.shape
    mean_dir = D.mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp(min=1e-12)

    X = D - D.mean(dim=0, keepdim=True) if center else D
    # economy SVD: singular values^2 / (n-1) are the covariance eigenvalues
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    ev = (S.double() ** 2 / max(n - 1, 1)).cpu().numpy()
    total = ev.sum()
    ratios = ev / total if total > 0 else np.zeros_like(ev)

    # uncentered top PC, for comparison against the mean direction
    Uu, Su, Vhu = torch.linalg.svd(D, full_matrices=False)
    pc1_unc = Vhu[0]
    if torch.dot(pc1_unc, mean_dir) < 0:
        pc1_unc = -pc1_unc

    k = min(max_rank, len(ratios))
    cum = np.cumsum(ratios)
    rank_for_threshold = int(np.searchsorted(cum, var_threshold) + 1)

    return {
        "n_examples": int(n), "d": int(d),
        "pc1_var_explained": float(ratios[0]) if len(ratios) else 0.0,
        "top_var_explained": [float(x) for x in ratios[:k]],
        "cumulative_var": [float(x) for x in cum[:k]],
        "participation_ratio": participation_ratio(ev),
        "cos_pc1_mean": float(torch.dot(pc1_unc, mean_dir)),
        "rank_for_threshold": rank_for_threshold,
        "var_threshold": var_threshold,
        "is_rank_one": bool(ratios[0] >= var_threshold) if len(ratios) else False,
        "basis_centered": Vh[:k].clone(),
        "basis_uncentered": Vhu[:k].clone(),
        "mean_direction": mean_dir,
    }


def concept_subspace(diffs: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Orthonormal basis [rank, d] for the concept subspace.

    Uncentered, because the object of interest is the span the contrast occupies
    in activation space — which includes the mean direction — not the span of
    its residual variation. Pass this to subspace_metrics when rank > 1.
    """
    D = diffs.float()
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    r = max(1, min(rank, Vh.shape[0]))
    return Vh[:r].clone()


def rank_report(diffs_by_cell: dict, var_threshold: float = 0.50) -> dict:
    """
    Run the rank test across every (pooling, layer) cell for one concept.

    Returns per-cell detail plus a summary. The summary is what decides whether
    the scalar Gini is the right statistic for this concept: if PC1 explains
    most of the variance in most cells, report scalars; if not, Phase 4's output
    is a subspace measurement and the headline changes shape.
    """
    cells, pc1s, prs = {}, [], []
    for (pooling, layer), D in sorted(diffs_by_cell.items()):
        r = analyze_rank(D, var_threshold=var_threshold)
        cells[f"{pooling}|{layer}"] = {
            k: v for k, v in r.items()
            if not isinstance(v, torch.Tensor)
        }
        pc1s.append(r["pc1_var_explained"])
        prs.append(r["participation_ratio"])

    pc1s, prs = np.asarray(pc1s), np.asarray(prs)
    frac_rank_one = float((pc1s >= var_threshold).mean()) if len(pc1s) else 0.0
    return {
        "cells": cells,
        "summary": {
            "pc1_var_mean": float(pc1s.mean()) if len(pc1s) else 0.0,
            "pc1_var_median": float(np.median(pc1s)) if len(pc1s) else 0.0,
            "participation_ratio_mean": float(prs.mean()) if len(prs) else 0.0,
            "frac_cells_rank_one": frac_rank_one,
            "treat_as_rank_one": bool(frac_rank_one >= 0.5),
            "var_threshold": var_threshold,
        },
    }
