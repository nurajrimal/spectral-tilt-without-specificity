#!/usr/bin/env python3
"""Rebuild the POS-probing comparison bar charts for the camera-ready appendix.

Fixes the defects reviewers flagged on `pos_probe_comparison_expanded.png`:

  * aspect ratio was 4.2:1, so at \\textwidth every label rendered at ~4pt.
    Rebuilt at ~2.1:1 (English, 8 models) and ~1.4:1 (Chinese, 2 models).
  * fonts enlarged to the camera-ready convention shared with
    `regenerate_fig5_unembed_null.py` / `regenerate_randomization_plot.py`.
  * the y-axis ran 0-1.0 with no data above 0.86 (~60% dead space). The
    zero baseline is KEPT -- these are bars, and a truncated bar baseline
    would misrepresent the ratios -- but the empty headroom above the data
    is trimmed, and the panel height is reduced so the remaining
    below-data space is no longer the dominant feature of the figure.
  * the title read "M1: POS Probing -- Does Syntax Shout?". "M1" is an
    internal experiment label and "Does Syntax Shout?" is the framing the
    paper's title deliberately walked back. Retitled neutrally.
  * the legend sat inside the axes on top of the right-most bars; it is
    now above the axes.

Presentation only: every plotted value (four bar heights + std error bar per
model, and the Delta / significance annotation) is read verbatim from the
source JSON, exactly as the original `plot_comparison()` in
`pos_probe_experiment.py` did.

Usage:
  python scripts/regenerate_pos_probe_figures.py \
      --expanded results/_spectral/pos_probe/pos_probe_all_models_expanded.json \
      --chinese  results/_spectral/pos_probe/pos_probe_all_models_chinese.json \
      --outdir   .paper/EMNLP_CameraReady/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Camera-ready typography (matches regenerate_fig5_unembed_null.py /
# regenerate_randomization_plot.py).
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "figure.titlesize": 20,
})

# Unchanged from pos_probe_experiment.plot_comparison.
PROBES = ["top_k", "bottom_k", "random_k", "full"]
LABELS = ["Top-k (high-var)", "Bottom-k (low-var)", "Random-k", "Full dim"]
COLORS = ["#E63946", "#457B9D", "#95a5a6", "#2A9D8F"]
HATCHES = [None, None, "///", None]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def plot_comparison(all_results, save_path, title, figsize, ymax, annot_fs,
                    legend_ncol=4, rotation=18, title_pad=78, bar_width=0.18):
    n = len(all_results)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n)
    width = bar_width

    for i, (probe, label, color, hatch) in enumerate(
            zip(PROBES, LABELS, COLORS, HATCHES)):
        means = [r["probes"][probe]["mean_accuracy"] for r in all_results]
        stds = [r["probes"][probe]["std_accuracy"] for r in all_results]
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, alpha=0.88, capsize=4, edgecolor="white",
               linewidth=0.6, hatch=hatch if hatch else "")

    ax.set_xlabel("Model")
    ax.set_ylabel("POS accuracy (5-fold CV)")
    ax.set_title(title, pad=title_pad)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([r["model_name"] for r in all_results],
                       rotation=rotation,
                       ha="right" if rotation else "center")
    # Hug the bars: the default 5% margin leaves visibly empty gutters when
    # there are only a couple of model groups.
    ax.set_xlim(-0.5 * width - 0.22, (n - 1) + 3.5 * width + 0.22)
    # Bars keep a true zero baseline; only the empty headroom is trimmed.
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend above the axes so it cannot cover the right-most bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.075),
              ncol=legend_ncol,
              frameon=False, handlelength=1.5, columnspacing=1.4,
              handletextpad=0.5)

    # Delta (top-k minus bottom-k) and its significance, in a dedicated row
    # just above the axes. Previously these sat on top of the bars, where at
    # \textwidth they collided with each other and with the bars.
    trans = ax.get_xaxis_transform()
    for j, r in enumerate(all_results):
        test = r["probes"]["test"]
        ax.annotate(
            f"$\\Delta$={test['accuracy_gap']:+.3f}{stars(test['p_value'])}",
            xy=(j + 1.5 * width, 1.022), xycoords=trans, ha="center",
            va="bottom", fontsize=annot_fs, color="#E63946",
            fontweight="bold", annotation_clip=False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")
    for r in all_results:
        pr = r["probes"]
        print("   %-16s top=%.4f bot=%.4f rnd=%.4f full=%.4f gap=%+.3f p=%.3g"
              % (r["model_name"], pr["top_k"]["mean_accuracy"],
                 pr["bottom_k"]["mean_accuracy"], pr["random_k"]["mean_accuracy"],
                 pr["full"]["mean_accuracy"], pr["test"]["accuracy_gap"],
                 pr["test"]["p_value"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", type=Path)
    ap.add_argument("--chinese", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    if args.expanded:
        plot_comparison(
            json.load(open(args.expanded)),
            args.outdir / "pos_probe_comparison_expanded.png",
            "POS-tag probing accuracy by spectral subspace "
            "(UD English-EWT, $k=0.1d$)",
            figsize=(12.0, 4.6), ymax=0.90, annot_fs=13)

    if args.chinese:
        plot_comparison(
            json.load(open(args.chinese)),
            args.outdir / "pos_probe_comparison_chinese.png",
            "POS-tag probing accuracy by spectral subspace\n"
            "(UD Chinese-GSD, $k=0.1d$)",
            figsize=(8.2, 4.8), ymax=0.78, annot_fs=15, legend_ncol=2,
            rotation=0, title_pad=88, bar_width=0.20)


if __name__ == "__main__":
    main()
