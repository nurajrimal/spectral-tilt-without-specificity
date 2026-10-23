"""Rebuild Figure 5 (random unembedding null / dual geometry).

Reproduces the published camera-ready figure. Fixes three defects in the
reviewed version:
  * model tick labels were hard-truncated to 10 chars ("Qwen2.5-0.", "SmolLM2-1.")
  * the right panel was capped at 3 of 4 models by `list(plot_data.items())[:3]`
  * fonts were too small at single-column width (reviewers R2/R3)

PROVENANCE -- please read before reusing these numbers.

Two of the three quantities plotted here are recomputed from a released
artifact; one is transcribed from a run log. Specifically:

  * per-concept Gini values (right panel) and the concept-pair means
    (left panel, red bars) are read from and averaged over
    `results/_spectral/artifact_check/artifact_check_unembed.json`,
    which ships in this repository. Fully recomputable.

  * the four random-pair null means (left panel, blue bars; blue diamonds
    on the right) are transcribed from the authoritative 4-model run log of
    `fix_hole5_unembed_null.py` -- see RANDOM_NULL_GINI below. The per-model
    JSON that run wrote was later overwritten by a 2-model re-run, so
    `results/_spectral/hole_fixes/hole5_unembed_null.json` contains only
    Qwen2.5-0.5B and Qwen2.5-1.5B. Nothing was tuned or selected: these are
    the values that run printed, and the paper reports them rounded to three
    decimals (Appendix "Dual Geometry Details"). To regenerate them from
    scratch rather than trust the transcription, re-run
    `fix_hole5_unembed_null.py --models Qwen2.5-0.5B Llama-3.2-1B gemma-2-2b
    SmolLM2-1.7B` on a GPU host; it samples 1,000 random token pairs per
    model with seed 42.

  * the random-null CDF *curves* shown in the reviewed version were never
    persisted and cannot be recovered, so the right panel instead shows the
    per-concept Gini distribution -- strictly more informative, and fully
    backed by the released artifact above.

The concept means were independently cross-checked against the run log:
exact agreement on all four models.

Usage:
    python scripts/regenerate_fig5_unembed_null.py <out.png> \
        [results/_spectral/artifact_check/artifact_check_unembed.json]
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen2.5-0.5B", "Llama-3.2-1B", "gemma-2-2b", "SmolLM2-1.7B"]

# Random token-pair null means (1,000 pairs/model, seed 42), transcribed from
# the 4-model run log of fix_hole5_unembed_null.py. See PROVENANCE above: the
# JSON this run wrote was overwritten by a later 2-model re-run.
RANDOM_NULL_GINI = {
    "Qwen2.5-0.5B": -0.0949,
    "Llama-3.2-1B": -0.1238,
    "gemma-2-2b":   -0.1174,
    "SmolLM2-1.7B": -0.0713,
}

out_path = sys.argv[1]
unembed_path = (sys.argv[2] if len(sys.argv) > 2
                else "results/_spectral/artifact_check/artifact_check_unembed.json")
_unembed = json.load(open(unembed_path))
data = {
    "per_concept": {m: [c["gini_deviation"] for c in _unembed[m].values()]
                    for m in MODELS},
    "log": {m: {"concept_gini": float(np.mean(
                    [c["gini_deviation"] for c in _unembed[m].values()])),
                "random_gini": RANDOM_NULL_GINI[m]}
            for m in MODELS},
}
CONCEPT_C, RANDOM_C = "#E63946", "#457B9D"

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))

# ── Panel 1: concept vs random mean Gini ──────────────────────────────────
ax = axes[0]
concept = [data["log"][m]["concept_gini"] for m in MODELS]
random_ = [data["log"][m]["random_gini"] for m in MODELS]
x, w = np.arange(len(MODELS)), 0.36

ax.bar(x - w / 2, concept, w, label="Concept pairs", color=CONCEPT_C, alpha=0.88)
ax.bar(x + w / 2, random_, w, label="Random pairs", color=RANDOM_C, alpha=0.88)
ax.axhline(0, color="black", lw=0.9)

for xi, cv in zip(x, concept):
    ax.annotate("$p<10^{-4}$", (xi - w / 2, cv), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=12.5, color=CONCEPT_C)

ax.set_ylabel("Mean Gini deviation")
ax.set_title("Unembedding concentration:\nconcept vs. random token pairs")
ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=18, ha="right")   # full names, no truncation
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
          frameon=False, handlelength=1.4, columnspacing=1.6)
ax.grid(True, alpha=0.22, axis="y")
ax.margins(y=0.20)

# ── Panel 2: per-concept Gini distribution ────────────────────────────────
ax = axes[1]
groups = [np.asarray(data["per_concept"][m], dtype=float) for m in MODELS]

bp = ax.boxplot(groups, positions=x, widths=0.5, showfliers=False,
                medianprops=dict(color=CONCEPT_C, lw=2.2),
                boxprops=dict(color="#666666", lw=1.2),
                whiskerprops=dict(color="#666666", lw=1.2),
                capprops=dict(color="#666666", lw=1.2))

rng = np.random.default_rng(0)
for xi, vals in zip(x, groups):
    ax.scatter(xi + rng.uniform(-0.13, 0.13, len(vals)), vals,
               s=22, color=CONCEPT_C, alpha=0.55, zorder=3, linewidths=0)

ax.plot(x, random_, marker="D", ms=8, lw=0, color=RANDOM_C,
        label="Random-pair mean", zorder=4)
ax.axhline(0, color="black", lw=0.9)

ax.set_ylabel("Gini deviation")
ax.set_title("Per-concept spread\n(each point = one concept)")
ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=18, ha="right")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False,
          handlelength=1.4)
ax.grid(True, alpha=0.22, axis="y")

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("wrote", out_path)
for m, c, r, g in zip(MODELS, concept, random_, groups):
    print("  %-14s concept=%+.4f random=%+.4f  n=%d  median=%+.4f" % (
        m, c, r, len(g), float(np.median(g))))
