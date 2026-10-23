#!/usr/bin/env python
"""
Flagship INTRODUCTION figure for
"Concepts Whisper: Spectral Anti-Concentration and the Dual Geometry of
 Transformer Representations" (EMNLP main, camera-ready).

Goal: let a reader grasp the DUAL GEOMETRY without knowing any spectral metric.

    * Horizontal axis = eigenspectrum of the unembedding covariance Sigma,
      ordered HIGH-variance "head" (left) -> LOW-variance "tail" (right).
    * STATIC vocabulary concept contrasts  gamma(w+) - gamma(w-)  CONCENTRATE
      in the high-variance head            (warm vermilion, secondary/subtle).
    * CONTEXTUALIZED activation concept directions (difference-of-means, SAE,
      linear probe) ANTI-CONCENTRATE in the low-variance tail (steel blue).
    * A sweeping arrow conveys the forward-pass ROTATION head -> tail.
    * A small inset shows the empirical signature: the cumulative-energy curve
      bowing BELOW the diagonal = anti-concentration (schematic, not a data dump).

Two design variants are produced so the authors can choose:
    v1  single horizontal-spectrum "landscape" panel
    v2  two side-by-side panels (vocabulary vs. activation) linked by the arrow

Palette is colorblind-safe (Okabe-Ito) and reads in grayscale (the two groups
are also separated by SHAPE and by POSITION on the axis, not color alone).

No LaTeX required. Run with the project venv:
    .venv/bin/python make_intro_figure.py

------------------------------------------------------------------------------
Tweak knobs live in the CONFIG block just below.
------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch, Circle
from matplotlib.path import Path
import matplotlib.font_manager as fm

# ----------------------------------------------------------------------------
# CONFIG  --  colors, fonts, headline numbers.  Edit here.
# ----------------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe palette.
C_HEAD   = "#D55E00"   # vermilion  -> vocabulary / high-variance / structural
C_HEAD_F = "#F4C9AE"   # light fill for head band
C_TAIL   = "#0072B2"   # steel blue -> activation / low-variance / semantic
C_TAIL_F = "#BcdCEB".upper()  # light fill for tail band (see below, overridden)
C_TAIL_F = "#C7DEEC"   # light fill for tail band
C_SPEC   = "#9AA0A6"   # neutral grey for the eigenvalue spectrum
C_GRID   = "#C9CDD2"
C_INK    = "#1A1A1A"   # near-black text
C_MUTE   = "#5F6368"   # muted text

# Headline empirical numbers (kept factual; see paper Table 3 / abstract).
GINI_MEAN   = -0.28          # mean Gini deviation across 17 models
N_MODELS    = 17
P_STRING    = r"$p<10^{-33}$"

# Global typography -- LARGE and crisp (reviewers flagged tiny fonts).
mpl.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType (editable / crisp vector text)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.9,
    "font.size": 11,
})

SEED = 7


# ----------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------
def eigen_spectrum(x):
    """A smoothly decaying eigenvalue profile (high on the left, flat tail)."""
    return np.exp(-4.2 * x) * (1 - 0.12 * x) + 0.012


def draw_region_bands(ax, split=0.42, y0=0.0, y1=1.0, alpha=0.55):
    """Faint warm band over the head, faint blue band over the tail."""
    ax.axvspan(0.0, split, ymin=y0, ymax=y1, color=C_HEAD_F, alpha=alpha,
               lw=0, zorder=0)
    ax.axvspan(split, 1.0, ymin=y0, ymax=y1, color=C_TAIL_F, alpha=alpha,
               lw=0, zorder=0)


def draw_spectrum_area(ax, ytop=0.62, ybase=0.0):
    """Filled grey eigenvalue 'mountain': loud on the left, quiet tail right."""
    x = np.linspace(0, 1, 400)
    y = eigen_spectrum(x)
    y = ybase + (ytop - ybase) * (y - y.min()) / (y.max() - y.min())
    ax.fill_between(x, ybase, y, color=C_SPEC, alpha=0.30, lw=0, zorder=1)
    ax.plot(x, y, color=C_SPEC, lw=1.6, alpha=0.9, zorder=1.1)
    return x, y


def scatter_cluster(ax, center, n, y_center, spread_x, spread_y, marker,
                    color, size, seed):
    """A tight jittered cluster of concept-direction markers."""
    rng = np.random.default_rng(seed)
    xs = rng.normal(center, spread_x, n)
    ys = rng.normal(y_center, spread_y, n)
    ax.scatter(xs, ys, s=size, marker=marker, facecolor=color,
               edgecolor="white", linewidth=0.7, zorder=5, alpha=0.95)
    return xs, ys


def draw_cdf_inset(ax):
    """Schematic cumulative-energy signature: below diagonal = anti-concentrate."""
    t = np.linspace(0, 1, 200)
    # Diagonal = uniform baseline.
    ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=C_MUTE, lw=1.3, zorder=2)
    # Activation (tail): bows BELOW the diagonal -> anti-concentration.
    below = t ** 2.3
    ax.plot(t, below, color=C_TAIL, lw=2.6, zorder=4)
    ax.fill_between(t, below, t, color=C_TAIL, alpha=0.14, lw=0, zorder=3)
    # Vocabulary (head): bows ABOVE the diagonal -> concentration (subtle).
    above = t ** 0.5
    ax.plot(t, above, color=C_HEAD, lw=2.0, zorder=4, alpha=0.9)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(C_GRID); s.set_linewidth(0.9)
    ax.set_xlabel("cumulative variance", fontsize=8.5, color=C_INK, labelpad=2)
    ax.set_ylabel("cumulative\nconcept energy", fontsize=8.5, color=C_INK,
                  labelpad=2, linespacing=0.95)
    ax.set_title("spectral energy", fontsize=9, color=C_INK, pad=3)


def sound_arcs(ax, x, y, color, scale, n=3, alpha=0.9, lw=1.6):
    """Concentric arcs suggesting 'loud' (large) vs 'whisper' (small)."""
    from matplotlib.patches import Arc
    for i in range(1, n + 1):
        w = scale * i
        arc = Arc((x, y), w, w * 1.6, angle=0, theta1=300, theta2=420,
                  color=color, lw=lw, alpha=alpha * (1 - 0.18 * (i - 1)),
                  zorder=6)
        ax.add_patch(arc)


# ----------------------------------------------------------------------------
# VARIANT 1  --  horizontal-spectrum landscape
# ----------------------------------------------------------------------------
def make_v1():
    fig = plt.figure(figsize=(6.3, 2.95), dpi=200)
    # Main drawing axes.  Leave headroom at top for the arrow banner and a
    # margin at the bottom for the region labels.
    ax = fig.add_axes([0.035, 0.145, 0.93, 0.74])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    draw_region_bands(ax, split=0.44, y0=0.0, y1=0.66, alpha=0.5)
    draw_spectrum_area(ax, ytop=0.44, ybase=0.10)

    # --- concept-direction clusters -----------------------------------------
    # Vocabulary contrasts concentrate in the head (left).  Squares, vermilion.
    scatter_cluster(ax, center=0.135, n=22, y_center=0.30, spread_x=0.050,
                    spread_y=0.070, marker="s", color=C_HEAD, size=44, seed=SEED)
    # Activation concepts anti-concentrate in the tail (right).  Circles, blue.
    scatter_cluster(ax, center=0.82, n=28, y_center=0.30, spread_x=0.080,
                    spread_y=0.080, marker="o", color=C_TAIL, size=48, seed=SEED + 1)

    # --- the forward-pass rotation arrow (high banner arc) ------------------
    arrow = FancyArrowPatch((0.20, 0.60), (0.775, 0.60),
                            connectionstyle="arc3,rad=-0.30",
                            arrowstyle="-|>", mutation_scale=22,
                            lw=2.6, color=C_INK, zorder=7,
                            capstyle="round")
    ax.add_patch(arrow)
    ax.text(0.46, 0.99, "forward pass rotates concepts",
            ha="center", va="center", fontsize=10.5, style="italic",
            color=C_INK, zorder=8)

    # --- sound motif: shout (head, subtle) vs whisper (tail) ----------------
    sound_arcs(ax, 0.052, 0.30, C_HEAD, scale=0.045, n=3, alpha=0.55, lw=1.4)
    sound_arcs(ax, 0.915, 0.30, C_TAIL, scale=0.020, n=2, alpha=0.85, lw=1.4)

    # --- region labels (below the spectrum axis) ----------------------------
    ax.text(0.135, 0.035, "HIGH-VARIANCE HEAD", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_HEAD)
    ax.text(0.135, -0.04, "structural  ($\\it{shouts}$)", ha="center", va="center",
            fontsize=9, color=C_MUTE)
    ax.text(0.82, 0.035, "LOW-VARIANCE TAIL", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_TAIL)
    ax.text(0.82, -0.04, "semantic concepts  ($\\it{whisper}$)", ha="center",
            va="center", fontsize=9, color=C_MUTE)

    # cluster captions (just above each cluster)
    ax.text(0.135, 0.475, "vocabulary contrasts\n$\\gamma(w^{+})-\\gamma(w^{-})$",
            ha="center", va="bottom", fontsize=9, color=C_HEAD, linespacing=1.0,
            fontweight="bold")
    ax.text(0.82, 0.475, "activation directions\ndiff-of-means · SAE · probe",
            ha="center", va="bottom", fontsize=9, color=C_TAIL, linespacing=1.0,
            fontweight="bold")

    # spectrum axis arrow + label
    ax.annotate("", xy=(1.0, 0.10), xytext=(0.0, 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.3))
    ax.text(0.50, -0.125, "eigenspectrum of unembedding covariance  $\\Sigma$",
            ha="center", va="center", fontsize=8.6, color=C_MUTE, style="italic",
            zorder=9)

    # --- inset: empirical signature, tucked in the empty middle valley ------
    axi = fig.add_axes([0.415, 0.235, 0.165, 0.30])
    draw_cdf_inset(axi)
    axi.text(0.05, 0.88, "vocab", color=C_HEAD, fontsize=7.4, fontweight="bold",
             ha="left", va="center")
    axi.text(0.95, 0.12, "activ.", color=C_TAIL, fontsize=7.4,
             fontweight="bold", ha="right", va="center")

    # --- headline stat (top-left, clear of the arrow) -----------------------
    ax.text(0.005, 0.985,
            f"mean Gini $=\\,{GINI_MEAN}$\n{N_MODELS} models · {P_STRING}",
            ha="left", va="top", fontsize=8.4, color=C_MUTE, linespacing=1.2,
            transform=ax.transAxes)

    fig.savefig(os.path.join(OUTDIR, "dual_geometry_v1.pdf"))
    fig.savefig(os.path.join(OUTDIR, "dual_geometry_v1.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# VARIANT 2  --  two-panel: static vocabulary vs. contextualized activation
# ----------------------------------------------------------------------------
def _panel(ax, mode):
    """mode in {'vocab', 'activation'}."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    if mode == "vocab":
        col, colf, mark = C_HEAD, C_HEAD_F, "s"
        band0, band1 = (0.0, 0.44)
    else:
        col, colf, mark = C_TAIL, C_TAIL_F, "o"
        band0, band1 = (0.44, 1.0)

    # backdrop spectrum (same in both panels -> same axis, different landing)
    draw_spectrum_area(ax, ytop=0.52, ybase=0.10)
    ax.axvspan(band0, band1, ymin=0.12, ymax=0.78, color=colf, alpha=0.55,
               lw=0, zorder=0.5)

    if mode == "vocab":
        scatter_cluster(ax, center=0.15, n=24, y_center=0.36, spread_x=0.055,
                        spread_y=0.085, marker=mark, color=col, size=44, seed=SEED)
        sound_arcs(ax, 0.052, 0.36, col, scale=0.05, n=3, alpha=0.5, lw=1.3)
    else:
        scatter_cluster(ax, center=0.74, n=30, y_center=0.34, spread_x=0.082,
                        spread_y=0.095, marker=mark, color=col, size=48, seed=SEED + 2)
        sound_arcs(ax, 0.95, 0.34, col, scale=0.022, n=2, alpha=0.85, lw=1.3)

    # spectrum axis
    ax.annotate("", xy=(1.0, 0.10), xytext=(0.0, 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.2))
    ax.text(0.0, 0.02, "high var", ha="left", fontsize=8, color=C_MUTE)
    ax.text(1.0, 0.02, "low var (tail)", ha="right", fontsize=8, color=C_MUTE)


def make_v2():
    fig = plt.figure(figsize=(6.3, 2.85), dpi=200)
    axL = fig.add_axes([0.035, 0.20, 0.44, 0.56])
    axR = fig.add_axes([0.525, 0.20, 0.44, 0.56])

    _panel(axL, "vocab")
    _panel(axR, "activation")

    # panel titles
    axL.text(0.5, 1.20, "STATIC VOCABULARY", ha="center", va="center",
             fontsize=11, fontweight="bold", color=C_HEAD,
             transform=axL.transAxes)
    axL.text(0.5, 1.05, "unembedding contrasts $\\gamma(w^{+})-\\gamma(w^{-})$",
             ha="center", va="center", fontsize=8.8, color=C_MUTE,
             transform=axL.transAxes)
    axL.text(0.5, 0.90, "concentrate in the head", ha="center", va="center",
             fontsize=9.4, color=C_HEAD, style="italic", transform=axL.transAxes)

    axR.text(0.5, 1.20, "CONTEXTUALIZED ACTIVATION", ha="center", va="center",
             fontsize=11, fontweight="bold", color=C_TAIL,
             transform=axR.transAxes)
    axR.text(0.5, 1.05, "diff-of-means · SAE features · linear probe",
             ha="center", va="center", fontsize=8.8, color=C_MUTE,
             transform=axR.transAxes)
    axR.text(0.5, 0.90, "anti-concentrate in the tail", ha="center", va="center",
             fontsize=9.4, color=C_TAIL, style="italic", transform=axR.transAxes)

    # connecting forward-pass arrow between panels (in figure coords)
    con = FancyArrowPatch((0.483, 0.50), (0.523, 0.50),
                          transform=fig.transFigure,
                          connectionstyle="arc3,rad=0.0",
                          arrowstyle="-|>", mutation_scale=20, lw=2.6,
                          color=C_INK, zorder=20)
    fig.add_artist(con)
    fig.text(0.503, 0.585, "forward\npass", ha="center", va="center",
             fontsize=8.6, style="italic", color=C_INK, linespacing=1.0)

    # bottom takeaway strip
    fig.text(0.5, 0.085,
             "Concepts whisper in the quiet tail; the loud head carries "
             "structural load",
             ha="center", va="center", fontsize=9.2, color=C_INK, style="italic")
    fig.text(0.5, 0.028,
             f"mean Gini $=\\,{GINI_MEAN}$ across {N_MODELS} models · {P_STRING}",
             ha="center", va="center", fontsize=8.4, color=C_MUTE)

    fig.savefig(os.path.join(OUTDIR, "dual_geometry_v2.pdf"))
    fig.savefig(os.path.join(OUTDIR, "dual_geometry_v2.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_v1()
    make_v2()
    print("wrote dual_geometry_v1/v2 .pdf/.png to", OUTDIR)
