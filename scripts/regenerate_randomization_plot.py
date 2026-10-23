#!/usr/bin/env python3
"""Regenerate randomization_comparison.png with a cleaner left panel."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as _mpl  # camera-ready: enlarged fonts for legibility
_mpl.rcParams.update({"font.size":16,"axes.titlesize":18,"axes.labelsize":17,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"figure.titlesize":20})
from pathlib import Path

RESULTS_JSON = Path("results/_spectral/randomization/randomization_results.json")
OUTPUT_PATHS = [
    Path("results/_spectral/randomization/randomization_comparison.png"),
    Path("figures/randomization_comparison.png"),
    Path(".paper/figures/randomization_comparison.png"),
]

with open(RESULTS_JSON) as f:
    all_results = json.load(f)

real_wrs = [r["real_win_rate"] for r in all_results]
fake_wrs = [r["fake_win_rate"] for r in all_results]
real_deltas = [r["mean_delta_real"] for r in all_results]
fake_deltas = [r["mean_delta_fake"] for r in all_results]

# ── Group by model for the left panel ──
from collections import defaultdict
model_data = defaultdict(lambda: {"real": [], "fake": []})
for r in all_results:
    model_data[r["model"]]["real"].append(r["real_win_rate"])
    model_data[r["model"]]["fake"].append(r["fake_win_rate"])

# Sort by model name
models_sorted = sorted(model_data.keys())
model_real_means = [np.mean(model_data[m]["real"]) for m in models_sorted]
model_fake_means = [np.mean(model_data[m]["fake"]) for m in models_sorted]
model_real_stds = [np.std(model_data[m]["real"]) for m in models_sorted]
model_fake_stds = [np.std(model_data[m]["fake"]) for m in models_sorted]
model_n = [len(model_data[m]["real"]) for m in models_sorted]

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Panel 1: Win rates grouped by model ──
ax = axes[0]
x = np.arange(len(models_sorted))
width = 0.35

bars_real = ax.bar(
    x - width/2, model_real_means, width,
    yerr=model_real_stds, capsize=3,
    label="Real WCA", color="#2A9D8F", alpha=0.85, edgecolor="white", linewidth=0.5
)
bars_fake = ax.bar(
    x + width/2, model_fake_means, width,
    yerr=model_fake_stds, capsize=3,
    label="Fake WCA", color="#E9C46A", alpha=0.85, edgecolor="white", linewidth=0.5
)

ax.axhline(0.5, color="grey", ls="--", alpha=0.5, linewidth=1.2, label="Chance (Naive)")
ax.set_ylabel("Win Rate vs. Naive", fontsize=16)
ax.set_title("Real vs. Fake WCA Win Rates\n(averaged across 4 language pairs per model)", fontsize=18, fontweight="bold")
ax.set_xticks(x)

# Clean model names for labels
clean_names = []
for m in models_sorted:
    name = m.replace("-v0.3", "").replace("OLMoE-1B-7B-0125", "OLMoE-7B")
    clean_names.append(name)

ax.set_xticklabels(clean_names, fontsize=13, rotation=45, ha="right")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=14, loc="upper left")
ax.grid(True, alpha=0.15, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel 2: Delta scatter (unchanged, already clean) ──
ax = axes[1]
ax.scatter(fake_deltas, real_deltas, c="#457B9D", s=45, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
lims = [min(min(fake_deltas), min(real_deltas)) - 0.005,
        max(max(fake_deltas), max(real_deltas)) + 0.005]
ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="Real = Fake")
ax.axhline(0, color="grey", ls=":", alpha=0.3)
ax.axvline(0, color="grey", ls=":", alpha=0.3)
ax.set_xlabel("Fake WCA Mean Δ(cosine sim)", fontsize=16)
ax.set_ylabel("Real WCA Mean Δ(cosine sim)", fontsize=16)
ax.set_title("Δ(WCA − Naive): Real vs. Randomized\nEigenvectors (68 model-pair combos)", fontsize=18, fontweight="bold")
ax.legend(fontsize=14)
ax.grid(True, alpha=0.15)
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add aggregate stats annotation
from scipy import stats as sp_stats
t_stat, p_val = sp_stats.ttest_rel(real_deltas, fake_deltas)
ax.annotate(
    f"Paired t = {t_stat:.3f}\np = {p_val:.2f}",
    xy=(0.97, 0.05), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=14,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
)

plt.tight_layout(w_pad=3)
for p in OUTPUT_PATHS:
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
plt.close()
print("Done.")
