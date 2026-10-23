"""
Spectral metrics, plus the inference the analysis plan commits to.

The first block is lifted verbatim from scripts/spectral_analysis.py so that
behavioural and word-level numbers are produced by literally the same code. Do
not fork it; make scripts/spectral_analysis.py import from here instead.

The second block implements what the proposal's section 3 commits to and the
published pipeline has no equivalent of: the Gini gap as the unit of evidence,
paired tests between concept classes, and TOST equivalence testing against a
smallest effect size of interest of -0.082 for the case where H0 is not rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Published word-level gap, used as the SESOI for equivalence testing (§3).
SESOI = -0.082


# ─── Published metrics (verbatim) ────────────────────────────────────────────

def eigendecompose(cov: torch.Tensor, cache_path: Optional[Path] = None) -> tuple:
    """Sigma = U Lambda U^T, eigenvalues descending."""
    if cache_path is not None and Path(cache_path).exists():
        d = torch.load(cache_path, map_location="cpu", weights_only=True)
        return d["eigenvalues"], d["eigenvectors"]

    evals, evecs = torch.linalg.eigh(cov.float())
    idx = torch.argsort(evals, descending=True)
    evals, evecs = evals[idx], evecs[:, idx]

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"eigenvalues": evals, "eigenvectors": evecs}, cache_path)
    return evals, evecs


def compute_cumulative_variance(eigenvalues: torch.Tensor) -> np.ndarray:
    ev = eigenvalues.detach().cpu().numpy()
    return np.cumsum(ev) / np.sum(ev)


def compute_concept_energy_cdf(v: torch.Tensor, eigenvectors: torch.Tensor) -> np.ndarray:
    """E_i = (v^T u_i)^2 / ||v||^2 ; C(k) = cumsum. Scale- and sign-invariant."""
    v = v.float()
    proj = eigenvectors.T @ v
    energy = (proj ** 2) / (v @ v).clamp(min=1e-12)
    return torch.cumsum(energy, dim=0).detach().cpu().numpy()


def compute_spectral_center_of_mass(cdf: np.ndarray, cumvar: np.ndarray) -> float:
    idx = np.searchsorted(cdf, 0.5)
    return 1.0 if idx >= len(cumvar) else float(cumvar[idx])


def compute_gini_deviation(cdf: np.ndarray, cumvar: np.ndarray) -> float:
    """Signed area between C(.) and the diagonal. Negative = tail."""
    return float(_trapz(cdf - cumvar, cumvar))


def metrics(v: torch.Tensor, eigenvectors: torch.Tensor, cumvar: np.ndarray) -> dict:
    cdf = compute_concept_energy_cdf(v, eigenvectors)
    return {"gini": compute_gini_deviation(cdf, cumvar),
            "scm": compute_spectral_center_of_mass(cdf, cumvar),
            "cdf": cdf}


# ─── Subspace version, for Phase 4 ───────────────────────────────────────────

def subspace_energy_cdf(basis: torch.Tensor, eigenvectors: torch.Tensor) -> np.ndarray:
    """
    Energy CDF for a SUBSPACE rather than a single direction.

    basis: [r, d], r orthonormal rows spanning the concept subspace.

    Phase 4 needs this because when a contrast is not rank one, the scalar Gini
    of a mean vector is the wrong statistic — it describes one arbitrary vector
    inside a subspace rather than where the subspace sits. Energy is summed over
    the basis and normalised by r, so an r-dimensional subspace is compared
    against the same diagonal as a single direction and the Gini values remain
    on one scale.
    """
    B = basis.float()
    B = torch.linalg.qr(B.T)[0].T  # re-orthonormalise defensively
    proj = eigenvectors.T @ B.T               # [d, r]
    energy = (proj ** 2).sum(dim=1) / B.shape[0]
    return torch.cumsum(energy, dim=0).detach().cpu().numpy()


def subspace_metrics(basis: torch.Tensor, eigenvectors: torch.Tensor,
                     cumvar: np.ndarray) -> dict:
    cdf = subspace_energy_cdf(basis, eigenvectors)
    return {"gini": compute_gini_deviation(cdf, cumvar),
            "scm": compute_spectral_center_of_mass(cdf, cumvar),
            "cdf": cdf, "rank": int(basis.shape[0])}


# ─── Nulls and the Gini gap ──────────────────────────────────────────────────

def random_direction_baseline(d: int, eigenvectors: torch.Tensor, cumvar: np.ndarray,
                              n_samples: int = 1000, seed: int = 42) -> dict:
    """
    Norm-matched isotropic directions in the same eigenbasis.

    Replicates the published control so these numbers sit beside the paper's.
    It is the weaker of the two nulls: a Gaussian direction has none of the
    structure a statement-level activation difference has.
    """
    rng = np.random.default_rng(seed)
    ginis, scms = np.empty(n_samples), np.empty(n_samples)
    cdfs = np.empty((n_samples, d), dtype=np.float32)
    for i in range(n_samples):
        v = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
        v = v / v.norm()
        cdf = compute_concept_energy_cdf(v, eigenvectors)
        cdfs[i] = cdf
        ginis[i] = compute_gini_deviation(cdf, cumvar)
        scms[i] = compute_spectral_center_of_mass(cdf, cumvar)
    return {"gini_mean": float(ginis.mean()), "gini_std": float(ginis.std()),
            "scm_mean": float(scms.mean()), "gini_samples": ginis,
            "cdf_mean": cdfs.mean(axis=0),
            "cdf_lo": np.percentile(cdfs, 2.5, axis=0),
            "cdf_hi": np.percentile(cdfs, 97.5, axis=0)}


def gini_gap(observed_gini: float, null_ginis: np.ndarray) -> float:
    """The unit of evidence. gap = Gini(real) - mean Gini(null)."""
    return float(observed_gini - np.mean(null_ginis))


def empirical_pvalue(observed: float, null: np.ndarray, tail: str = "two") -> float:
    """Permutation p-value with the +1 correction; never returns exactly 0."""
    null = np.asarray(null)
    n = len(null)
    if tail == "less":
        k = int((null <= observed).sum())
    elif tail == "greater":
        k = int((null >= observed).sum())
    else:
        centre = null.mean()
        k = int((np.abs(null - centre) >= abs(observed - centre)).sum())
    return (k + 1) / (n + 1)


# ─── Inference across models ─────────────────────────────────────────────────

def one_sample_test(gaps: Sequence[float]) -> dict:
    """
    H0: mean Gini gap = 0, tested across models.

    Models are the unit of analysis, not concepts. Concepts inside one model
    share an eigenbasis and are not independent — the published paper says so
    and keeps its concept-level statistics descriptive for that reason. With
    eight models n is small, so the Wilcoxon signed-rank result is reported
    alongside the t-test and the two are expected to agree.
    """
    from scipy import stats
    g = np.asarray(gaps, dtype=float)
    out = {"n": int(len(g)), "mean": float(g.mean()),
           "sd": float(g.std(ddof=1)) if len(g) > 1 else float("nan"),
           "median": float(np.median(g))}
    if len(g) > 1:
        t, p = stats.ttest_1samp(g, 0.0)
        out["t"] = float(t)
        out["p_t"] = float(p)
        out["cohens_d"] = float(g.mean() / g.std(ddof=1)) if g.std(ddof=1) > 0 else float("nan")
        ci = stats.t.interval(0.95, len(g) - 1, loc=g.mean(),
                              scale=stats.sem(g)) if stats.sem(g) > 0 else (g.mean(), g.mean())
        out["ci95"] = [float(ci[0]), float(ci[1])]
        if len(g) >= 6:
            try:
                w, pw = stats.wilcoxon(g)
                out["wilcoxon_W"], out["p_wilcoxon"] = float(w), float(pw)
            except ValueError:
                pass
    return out


def paired_test(gaps_a: Sequence[float], gaps_b: Sequence[float],
                label_a: str = "a", label_b: str = "b") -> dict:
    """
    The secondary and more informative test (§3): do two concept classes differ?

    Pass model-aligned vectors — gaps_a[i] and gaps_b[i] must be the same model,
    layer and basis. Asking whether refusal and truth differ from each other is
    a sharper question than asking twice whether each differs from zero, and it
    is the test the designed contrast was built for.
    """
    from scipy import stats
    a, b = np.asarray(gaps_a, dtype=float), np.asarray(gaps_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs aligned vectors, got {a.shape} vs {b.shape}")
    d = a - b
    out = {"label_a": label_a, "label_b": label_b, "n": int(len(d)),
           "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean())}
    if len(d) > 1:
        t, p = stats.ttest_rel(a, b)
        out["t"] = float(t)
        out["p_t"] = float(p)
        sd = d.std(ddof=1)
        out["cohens_dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
        if len(d) >= 6:
            try:
                w, pw = stats.wilcoxon(a, b)
                out["wilcoxon_W"], out["p_wilcoxon"] = float(w), float(pw)
            except ValueError:
                pass
    return out


def tost_equivalence(gaps: Sequence[float], sesoi: float = SESOI,
                     alpha: float = 0.05) -> dict:
    """
    Two one-sided tests, for when H0 is not rejected (§3).

    A non-significant test is not evidence of no effect. TOST asks the different
    question of whether the gap is small enough to be practically zero, against
    bounds of +/- |sesoi| with sesoi defaulting to the published word-level gap
    of -0.082: if behavioural directions anti-concentrated as strongly as word
    concepts, this is the gap we would see.

    Equivalence is declared only when BOTH one-sided tests reject, which happens
    when the 90% CI sits entirely inside the bounds. Report the bound, never
    just the verdict — "equivalent to zero within +/- 0.082" is a claim a reader
    can argue with; "no effect" is not.
    """
    from scipy import stats
    g = np.asarray(gaps, dtype=float)
    bound = abs(sesoi)
    out = {"n": int(len(g)), "mean": float(g.mean()),
           "sesoi": float(sesoi), "bound": float(bound), "alpha": alpha}
    if len(g) < 2:
        out["conclusive"] = False
        out["note"] = "need n >= 2"
        return out

    se = stats.sem(g)
    df = len(g) - 1
    if se == 0:
        out["conclusive"] = bool(abs(g.mean()) < bound)
        out["note"] = "zero variance"
        return out

    t_lower = (g.mean() - (-bound)) / se     # H0: mean <= -bound
    t_upper = (g.mean() - bound) / se        # H0: mean >= +bound
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    ci90 = stats.t.interval(0.90, df, loc=g.mean(), scale=se)

    out.update({
        "p_lower": p_lower, "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "ci90": [float(ci90[0]), float(ci90[1])],
        "equivalent": bool(p_lower < alpha and p_upper < alpha),
    })
    out["conclusive"] = out["equivalent"]
    if not out["equivalent"]:
        out["note"] = ("Neither different from zero nor equivalent to it: the "
                       "study is underpowered at this n for this bound.")
    return out
