#!/usr/bin/env python3
"""Rebuild the split-injection appendix figures for the camera-ready paper.

Two defects in the reviewed versions (produced by the plotting block at the end
of `fix_hole4_v2_split_injection.py`):

  1. The PPL-increase panel used a LINEAR y-axis. Because the alpha sweep runs
     to alpha=100, where mean PPL increase reaches 1e9-1e11 %, the entire
     alpha=1..20 region -- where every claim in the paper actually lives -- was
     compressed into a flat line pinned at zero. This matters most on
     Qwen2.5-3B and Mistral-7B-v0.3, whose panels are cited as evidence for a
     null / reversal that the linear rendering simply cannot display.

     Fixed with a SYMLOG y-axis. Plain log is not usable: on gemma-2-2b and
     Qwen2.5-3B some alpha=1 means are slightly negative (they are
     statistically indistinguishable from zero, and their bootstrap CIs
     straddle zero), and a log axis would silently drop those points and the
     lower half of several CI bands. Symlog keeps them and keeps the bands
     honest. `linthresh` is chosen per model as the decade of the smallest
     non-zero mean, and the linear-vs-log crossover is shaded so the reader
     can see where the axis changes character.

     The already-regenerated main-text figure for Llama-3.1-8B is on a log
     y-axis, so this brings the appendix panels in line with it.

  2. Fonts were too small at \\textwidth (reviewers on appendices E-L). Rebuilt
     at the camera-ready typography convention and at ~3:1 rather than ~3.7:1.

Also cleaned up for print: one shared legend instead of three redundant ones,
a single figure suptitle instead of three stacked two-line panel titles, and
axis labels short enough to render at full size.

Presentation only. Panel count (3), series count (4), and every plotted value
are unchanged: means and bootstrap CIs are recomputed from the same JSON with
the same seeded `bootstrap_ci(n_boot=5000, seed=42)` as the original script, so
the curves are numerically identical.

Usage:
  python scripts/regenerate_hole4_split_injection.py \
      --json-dir results/_spectral/hole_fixes \
      --outdir   .paper/EMNLP_CameraReady/figures \
      --models gemma-2-2b Llama-3.2-3B Qwen2.5-3B Mistral-7B-v0.3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 19,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "figure.titlesize": 23,
})

COMPS = ["full", "shout", "whisper", "mid"]
# shout/whisper keep the hues of the already-regenerated main-text figure;
# full/mid stay in the secondary greys/greens of the original.
COLORS = {"full": "#333333", "shout": "#D55E00",
          "whisper": "#0072B2", "mid": "#2A9D8F"}


def bootstrap_ci(data, n_boot=5000, ci=0.95, seed=42):
    """Verbatim from fix_hole4_v2_split_injection.py (seeded => reproducible)."""
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([rng.choice(data, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.mean(data)),
            float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha))))


def series(results, alphas, comp, field, with_ci=False, require_logit=False):
    xs, ys, los, his = [], [], [], []
    for a in sorted(alphas):
        vals = [r[field] for r in results
                if r["component"] == comp and r["alpha"] == a
                and (r.get("has_logit_data", False) if require_logit else True)]
        if not vals:
            continue
        xs.append(a)
        if with_ci:
            m, lo, hi = bootstrap_ci(vals)
            ys.append(m); los.append(lo); his.append(hi)
        else:
            ys.append(float(np.mean(vals)))
    return xs, ys, los, his


def pick_linthresh(magnitudes):
    """Decade of the smallest non-zero magnitude, clamped to [1e-2, 1e0]."""
    m = [abs(v) for v in magnitudes if v != 0]
    if not m:
        return 1.0
    lt = 10.0 ** np.floor(np.log10(min(m)))
    return float(np.clip(lt, 1e-2, 1.0))


def regenerate(path, outdir):
    d = json.load(open(path))
    name = d["model_name"]
    alphas = d["alphas"]
    R = d["results"]
    k = d["top_k"]
    labels = {"full": "Full vector", "shout": f"Top-{k} (shouting)",
              "whisper": f"Bottom-{k} (whispering)", "mid": "Middle"}

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7))

    # ── Panel 1: PPL increase (symlog) ────────────────────────────────────
    ax = axes[0]
    panel1 = {c: series(R, alphas, c, "ppl_increase_pct", with_ci=True)
              for c in COMPS}
    linthresh = pick_linthresh(
        [v for c in COMPS for v in panel1[c][1]])
    for c in COMPS:
        xs, ys, los, his = panel1[c]
        if not xs:
            continue
        ax.plot(xs, ys, "o-", color=COLORS[c], lw=2, ms=7, label=labels[c])
        ax.fill_between(xs, los, his, color=COLORS[c], alpha=0.13, lw=0)
    # Plain log where every mean and CI bound is positive (Llama-3.2-3B,
    # Mistral-7B-v0.3) -- identical treatment to the already-regenerated
    # main-text Llama-3.1-8B figure. Symlog only where it is actually needed
    # (gemma-2-2b, Qwen2.5-3B have slightly negative alpha=1 means / CI
    # bounds that a log axis would silently drop).
    floor = min(v for c in COMPS for arr in (panel1[c][1], panel1[c][2])
                for v in arr) if any(panel1[c][1] for c in COMPS) else 1.0
    if floor > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.45)
        # Shade the near-zero region where the axis is linear, not log.
        ax.axhspan(-linthresh, linthresh, color="#000000", alpha=0.045, lw=0,
                   zorder=0)
        ax.axhline(0, color="black", ls="-", lw=0.6, zorder=1)
        ax.annotate(f"linear within $\\pm${linthresh:g}", xy=(0.03, 0.97),
                    xycoords="axes fraction", fontsize=13, color="#555555",
                    ha="left", va="top")
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("PPL increase (%)")
    # Thin the decade ticks to every second decade -- the sweep spans up to
    # 12 decades and a tick per decade is unreadable at print size.
    if floor > 0:
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100.0))
    else:
        ax.yaxis.set_major_locator(
            SymmetricalLogLocator(base=100.0, linthresh=linthresh))

    # ── Panel 2: KL divergence ────────────────────────────────────────────
    ax = axes[1]
    for c in COMPS:
        xs, ys, _, _ = series(R, alphas, c, "mean_kl_divergence")
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[c], lw=2, ms=7)
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("Mean KL divergence")

    # ── Panel 3: concept shift ────────────────────────────────────────────
    ax = axes[2]
    for c in COMPS:
        xs, ys, _, _ = series(R, alphas, c, "mean_concept_shift",
                              require_logit=True)
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[c], lw=2, ms=7)
    ax.axhline(0, color="black", ls="-", lw=0.6)
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("Concept shift")

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handlelength=1.8,
               columnspacing=1.8)
    fig.suptitle(f"Split-injection interference on {name}", y=1.02)
    fig.tight_layout()

    out = outdir / f"hole4_v2_split_injection_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {out}  (y-scale: "
          f"{'log' if floor > 0 else f'symlog linthresh={linthresh:g}'})")
    for c in COMPS:
        xs, ys, _, _ = panel1[c]
        print("   PPL %-8s %s" % (
            c, " ".join("a=%g:%.4g" % (a, v) for a, v in zip(xs, ys))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    for m in args.models:
        regenerate(args.json_dir / f"hole4_v2_split_injection_{m}.json",
                   args.outdir)


if __name__ == "__main__":
    main()
