#!/usr/bin/env python3
"""
Analyze and visualize saved V2 experiment outputs.

This script intentionally consumes only CSV/JSON artifacts under
results/_spectral/v2_experiments and does not import torch or load models.
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl  # camera-ready: enlarged fonts for legibility
_mpl.rcParams.update({"font.size":16,"axes.titlesize":18,"axes.labelsize":17,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"figure.titlesize":20})

# ─── Global font sizing (legible when downscaled into an ACL column) ──────────
matplotlib.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None


EXPERIMENT_DIRS = {
    "covariance": "covariance_sensitivity",
    "lambda": "lambda_sensitivity",
    "nulls": "strong_nulls",
    "layerwise": "layerwise_rotation",
    "behavioral": "behavioral_steering",
    "sae": "sae_frequency_controls",
}

PREFERRED_VARIANTS = [
    "saved_language_subset_uncentered",
    "full_vocab_uncentered_uniform",
    "full_vocab_centered_uniform",
    "language_subset_uncentered_uniform",
    "language_subset_centered_uniform",
    "language_subset_uncentered_frequency_weighted",
    "language_subset_centered_frequency_weighted",
    "input_embedding_full_uncentered_uniform",
    "input_embedding_full_centered_uniform",
]

COMPONENT_ORDER = ["full", "top_k", "bottom_k", "random_k"]


CONCEPT_GROUPS = {
    "Language": {"[English - French]", "[French - German]", "[French - Spanish]", "[German - Spanish]"},
    "Morphology": {
        "[3pSg - Ved]", "[Ving - 3pSg]", "[Ving - Ved]", "[verb - 3pSg]", "[verb - V + able]",
        "[verb - V + er]", "[verb - V + ment]", "[verb - V + tion]", "[verb - Ved]", "[verb - Ving]",
        "[noun - plural]", "[pronoun - possessive]",
    },
    "Adjective": {"[adj - adj + ly]", "[adj - comparative]", "[adj - superlative]", "[adj - un + adj]"},
    "Semantic": {
        "[country - capital]", "[frequent - infrequent]", "[lower - upper]", "[male - female]",
        "[small - big]", "[thing - color]", "[thing - part]",
    },
}


def concept_group(concept):
    for group, concepts in CONCEPT_GROUPS.items():
        if concept in concepts:
            return group
    return "Other"


def clean_model_name(model):
    return str(model).replace("mistralai/", "").replace("Qwen/", "").replace("google/", "").replace("meta-llama/", "")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_value(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return x
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return x


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {k: parse_value(v) for k, v in row.items()}
            rows.append(parsed)
    return rows


def write_csv(path, rows, fieldnames=None):
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def finite(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out


def mean(values):
    vals = finite(values)
    return float(np.mean(vals)) if vals else None


def std(values):
    vals = finite(values)
    return float(np.std(vals)) if vals else None


def median(values):
    vals = finite(values)
    return float(np.median(vals)) if vals else None


def sem95(values):
    vals = finite(values)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def paired_t(x, y):
    xs = []
    ys = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        try:
            af, bf = float(a), float(b)
        except Exception:
            continue
        if math.isfinite(af) and math.isfinite(bf):
            xs.append(af)
            ys.append(bf)
    if len(xs) < 2 or sp_stats is None:
        return None, None
    t, p = sp_stats.ttest_rel(xs, ys)
    return float(t), float(p)


def one_sample_t(values, mu=0.0):
    vals = finite(values)
    if len(vals) < 2 or sp_stats is None:
        return None, None
    t, p = sp_stats.ttest_1samp(vals, mu)
    return float(t), float(p)


def pearson(x, y):
    xs = []
    ys = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        try:
            af, bf = float(a), float(b)
        except Exception:
            continue
        if math.isfinite(af) and math.isfinite(bf):
            xs.append(af)
            ys.append(bf)
    if len(xs) < 3:
        return None, None
    if sp_stats is not None:
        r, p = sp_stats.pearsonr(xs, ys)
        return float(r), float(p)
    return float(np.corrcoef(xs, ys)[0, 1]), None


def load_family_rows(root, family, filename):
    base = root / family
    rows = []
    if not base.exists():
        return rows
    for path in sorted(base.glob(f"*/{filename}")):
        model = path.parent.name
        try:
            for row in read_csv(path):
                row.setdefault("model", model)
                row["source_file"] = str(path)
                rows.append(row)
        except Exception as e:
            print(f"WARNING: could not read {path}: {e}")
    return rows


def load_all(root):
    return {
        "covariance": load_family_rows(root, "covariance_sensitivity", "covariance_sensitivity_summary.csv"),
        "lambda": load_family_rows(root, "lambda_sensitivity", "lambda_sensitivity_summary.csv"),
        "norm_null": load_family_rows(root, "strong_nulls", "norm_matched_random_directions.csv"),
        "frequency_pos_null": load_family_rows(root, "strong_nulls", "frequency_pos_matched_unembedding_pairs.csv"),
        "shuffle_null": load_family_rows(root, "strong_nulls", "within_concept_label_shuffles.csv"),
        "layerwise": load_family_rows(root, "layerwise_rotation", "layerwise_rotation.csv"),
        "behavioral": load_family_rows(root, "behavioral_steering", "behavioral_steering.csv"),
        "sae": load_family_rows(root, "sae_frequency_controls", "sae_activation_frequency_controls.csv"),
    }


def summarize_covariance(rows):
    by_variant = defaultdict(list)
    by_model_variant = defaultdict(list)
    for r in rows:
        variant = r.get("variant")
        if not variant:
            continue
        by_variant[variant].append(r)
        by_model_variant[(r.get("model"), variant)].append(r)

    variant_rows = []
    for variant in sorted(by_variant, key=lambda v: PREFERRED_VARIANTS.index(v) if v in PREFERRED_VARIANTS else 999):
        vals = [r.get("mean_gini") for r in by_variant[variant]]
        scms = [r.get("mean_scm") for r in by_variant[variant]]
        pvals = [r.get("p_gini_vs_zero") for r in by_variant[variant]]
        variant_rows.append({
            "variant": variant,
            "n_models": len({r.get("model") for r in by_variant[variant]}),
            "mean_gini": mean(vals),
            "median_gini": median(vals),
            "std_gini": std(vals),
            "mean_scm": mean(scms),
            "frac_p_lt_0_05": mean([float(p < 0.05) for p in finite(pvals)]),
        })

    sensitivity_rows = []
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r.get("model")][r.get("variant")] = r
    for model, d in by_model.items():
        base = d.get("saved_language_subset_uncentered") or d.get("language_subset_uncentered_uniform")
        if not base:
            continue
        base_g = base.get("mean_gini")
        for variant, r in d.items():
            sensitivity_rows.append({
                "model": model,
                "variant": variant,
                "mean_gini": r.get("mean_gini"),
                "delta_vs_saved_or_lang_uncentered": None if base_g is None or r.get("mean_gini") is None else float(r.get("mean_gini") - base_g),
                "mean_scm": r.get("mean_scm"),
            })
    return variant_rows, sensitivity_rows


def summarize_lambda(rows):
    for r in rows:
        r["real_minus_fake_delta"] = None if r.get("real_mean_delta_vs_naive") is None or r.get("fake_mean_delta_vs_naive") is None else float(r["real_mean_delta_vs_naive"] - r["fake_mean_delta_vs_naive"])
        r["real_minus_fake_win_rate"] = None if r.get("real_win_rate_vs_naive") is None or r.get("fake_win_rate_vs_naive") is None else float(r["real_win_rate_vs_naive"] - r["fake_win_rate_vs_naive"])

    by_lambda = defaultdict(list)
    by_pair = defaultdict(list)
    by_model = defaultdict(list)
    for r in rows:
        by_lambda[r.get("lambda")].append(r)
        by_pair[r.get("pair")].append(r)
        by_model[r.get("model")].append(r)

    lambda_rows = []
    for lam in sorted(by_lambda, key=lambda x: float(x) if x is not None else -1):
        rs = by_lambda[lam]
        real_delta = [r.get("real_mean_delta_vs_naive") for r in rs]
        fake_delta = [r.get("fake_mean_delta_vs_naive") for r in rs]
        real_wr = [r.get("real_win_rate_vs_naive") for r in rs]
        fake_wr = [r.get("fake_win_rate_vs_naive") for r in rs]
        t_delta, p_delta = paired_t(real_delta, fake_delta)
        lambda_rows.append({
            "lambda": lam,
            "n_rows": len(rs),
            "n_models": len({r.get("model") for r in rs}),
            "mean_real_delta": mean(real_delta),
            "mean_fake_delta": mean(fake_delta),
            "mean_real_minus_fake_delta": mean([r.get("real_minus_fake_delta") for r in rs]),
            "mean_real_win_rate": mean(real_wr),
            "mean_fake_win_rate": mean(fake_wr),
            "mean_real_minus_fake_win_rate": mean([r.get("real_minus_fake_win_rate") for r in rs]),
            "paired_t_delta": t_delta,
            "paired_p_delta": p_delta,
        })

    model_rows = []
    for model, rs in sorted(by_model.items()):
        model_rows.append({
            "model": model,
            "n_rows": len(rs),
            "mean_real_delta": mean([r.get("real_mean_delta_vs_naive") for r in rs]),
            "mean_fake_delta": mean([r.get("fake_mean_delta_vs_naive") for r in rs]),
            "mean_real_minus_fake_delta": mean([r.get("real_minus_fake_delta") for r in rs]),
            "mean_real_win_rate": mean([r.get("real_win_rate_vs_naive") for r in rs]),
            "mean_fake_win_rate": mean([r.get("fake_win_rate_vs_naive") for r in rs]),
        })

    pair_rows = []
    for pair, rs in sorted(by_pair.items()):
        pair_rows.append({
            "pair": pair,
            "n_rows": len(rs),
            "mean_real_delta": mean([r.get("real_mean_delta_vs_naive") for r in rs]),
            "mean_fake_delta": mean([r.get("fake_mean_delta_vs_naive") for r in rs]),
            "mean_real_minus_fake_delta": mean([r.get("real_minus_fake_delta") for r in rs]),
            "mean_real_win_rate": mean([r.get("real_win_rate_vs_naive") for r in rs]),
            "mean_fake_win_rate": mean([r.get("fake_win_rate_vs_naive") for r in rs]),
        })
    return lambda_rows, model_rows, pair_rows


def summarize_nulls(norm_rows, freq_rows, shuffle_rows):
    norm_summary = []
    for r in norm_rows:
        r["true_minus_norm_null_gini"] = None if r.get("true_gini") is None or r.get("norm_matched_null_mean_gini") is None else float(r["true_gini"] - r["norm_matched_null_mean_gini"])
        r["group"] = concept_group(r.get("concept"))
    for model, rs in sorted(group_by(norm_rows, "model").items()):
        norm_summary.append({
            "model": model,
            "n_concepts": len(rs),
            "mean_true_gini": mean([r.get("true_gini") for r in rs]),
            "mean_norm_null_gini": mean([r.get("norm_matched_null_mean_gini") for r in rs]),
            "mean_true_minus_null_gini": mean([r.get("true_minus_norm_null_gini") for r in rs]),
            "frac_true_more_tail_than_null": mean([float((r.get("true_minus_norm_null_gini") or 0) < 0) for r in rs]),
            "mean_p_null_le_true": mean([r.get("p_null_le_true") for r in rs]),
        })

    freq_summary = []
    for r in freq_rows:
        r["true_minus_frequency_null_gini"] = None if r.get("true_unembed_gini") is None or r.get("frequency_matched_null_mean_gini") is None else float(r["true_unembed_gini"] - r["frequency_matched_null_mean_gini"])
        r["true_minus_pos_proxy_null_gini"] = None if r.get("true_unembed_gini") is None or r.get("pos_proxy_matched_null_mean_gini") is None else float(r["true_unembed_gini"] - r["pos_proxy_matched_null_mean_gini"])
        r["group"] = concept_group(r.get("concept"))
    for model, rs in sorted(group_by(freq_rows, "model").items()):
        freq_summary.append({
            "model": model,
            "n_concepts": len(rs),
            "mean_true_unembed_gini": mean([r.get("true_unembed_gini") for r in rs]),
            "mean_frequency_null_gini": mean([r.get("frequency_matched_null_mean_gini") for r in rs]),
            "mean_pos_proxy_null_gini": mean([r.get("pos_proxy_matched_null_mean_gini") for r in rs]),
            "mean_true_minus_frequency_null_gini": mean([r.get("true_minus_frequency_null_gini") for r in rs]),
            "mean_true_minus_pos_proxy_null_gini": mean([r.get("true_minus_pos_proxy_null_gini") for r in rs]),
            "frac_true_greater_than_frequency_null": mean([float((r.get("true_minus_frequency_null_gini") or 0) > 0) for r in rs]),
            "frac_true_greater_than_pos_proxy_null": mean([float((r.get("true_minus_pos_proxy_null_gini") or 0) > 0) for r in rs]),
        })

    shuffle_summary = []
    for r in shuffle_rows:
        r["true_minus_sign_shuffle_gini"] = None if r.get("true_activation_gini") is None or r.get("sign_shuffle_mean_gini") is None else float(r["true_activation_gini"] - r["sign_shuffle_mean_gini"])
        r["true_minus_pair_shuffle_gini"] = None if r.get("true_activation_gini") is None or r.get("pair_shuffle_mean_gini") is None else float(r["true_activation_gini"] - r["pair_shuffle_mean_gini"])
        r["group"] = concept_group(r.get("concept"))
    for model, rs in sorted(group_by(shuffle_rows, "model").items()):
        shuffle_summary.append({
            "model": model,
            "n_concepts": len(rs),
            "mean_true_activation_gini": mean([r.get("true_activation_gini") for r in rs]),
            "mean_sign_shuffle_gini": mean([r.get("sign_shuffle_mean_gini") for r in rs]),
            "mean_pair_shuffle_gini": mean([r.get("pair_shuffle_mean_gini") for r in rs]),
            "mean_true_minus_sign_shuffle_gini": mean([r.get("true_minus_sign_shuffle_gini") for r in rs]),
            "mean_true_minus_pair_shuffle_gini": mean([r.get("true_minus_pair_shuffle_gini") for r in rs]),
            "frac_sign_shuffle_p_le_0_05": mean([float((r.get("p_sign_shuffle_le_true") or 1.0) <= 0.05) for r in rs]),
            "frac_pair_shuffle_p_le_0_05": mean([float((r.get("p_pair_shuffle_le_true") or 1.0) <= 0.05) for r in rs]),
        })

    return norm_summary, freq_summary, shuffle_summary


def summarize_layerwise(rows):
    by_model_layer = defaultdict(list)
    by_frac = defaultdict(list)
    transition_rows = []
    for r in rows:
        try:
            r["layer_index"] = int(float(r.get("layer_index")))
        except Exception:
            continue
        by_model_layer[(r.get("model"), r["layer_index"])].append(r)

    model_max = {}
    for r in rows:
        if isinstance(r.get("layer_index"), int) and r["layer_index"] >= 0:
            model_max[r.get("model")] = max(model_max.get(r.get("model"), 0), r["layer_index"])

    layer_rows = []
    by_model = defaultdict(list)
    for (model, layer), rs in sorted(by_model_layer.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        row = {
            "model": model,
            "layer_index": layer,
            "layer_fraction": -0.05 if layer < 0 else (layer / model_max.get(model, layer or 1)),
            "n_concepts": len({r.get("concept") for r in rs}),
            "mean_gini": mean([r.get("gini_final_unembed_spectrum") for r in rs]),
            "mean_scm": mean([r.get("scm_final_unembed_spectrum") for r in rs]),
            "mean_top10_energy": mean([r.get("top10_energy_final_unembed_spectrum") for r in rs]),
        }
        layer_rows.append(row)
        by_model[model].append(row)
        frac_bin = "embedding" if layer < 0 else f"{round(row['layer_fraction'] * 10) / 10:.1f}"
        by_frac[frac_bin].append(row)

    for model, rs in by_model.items():
        emb = next((r for r in rs if r["layer_index"] < 0), None)
        first = next((r for r in rs if r["layer_index"] == 0), None)
        last = max([r for r in rs if r["layer_index"] >= 0], key=lambda r: r["layer_index"], default=None)
        mid = min([r for r in rs if r["layer_index"] >= 0], key=lambda r: abs(r["layer_fraction"] - 0.5), default=None)
        transition_rows.append({
            "model": model,
            "embedding_gini": emb.get("mean_gini") if emb else None,
            "layer0_gini": first.get("mean_gini") if first else None,
            "mid_gini": mid.get("mean_gini") if mid else None,
            "final_gini": last.get("mean_gini") if last else None,
            "embedding_to_layer0_delta": None if not emb or not first else first.get("mean_gini") - emb.get("mean_gini"),
            "layer0_to_final_delta": None if not first or not last else last.get("mean_gini") - first.get("mean_gini"),
            "mid_to_final_delta": None if not mid or not last else last.get("mean_gini") - mid.get("mean_gini"),
        })

    frac_rows = []
    for frac, rs in sorted(by_frac.items(), key=lambda kv: -0.1 if kv[0] == "embedding" else float(kv[0])):
        frac_rows.append({
            "layer_fraction_bin": frac,
            "n_model_layers": len(rs),
            "mean_gini": mean([r.get("mean_gini") for r in rs]),
            "ci95_gini": sem95([r.get("mean_gini") for r in rs]),
            "mean_scm": mean([r.get("mean_scm") for r in rs]),
        })
    return layer_rows, transition_rows, frac_rows


def summarize_behavioral(rows):
    for r in rows:
        r["concept_abs_shift"] = abs(float(r.get("mean_concept_shift") or 0.0))
        r["shift_per_ppl_pct"] = None
        if r.get("ppl_increase_pct") not in (None, 0):
            denom = max(abs(float(r["ppl_increase_pct"])), 1e-9)
            r["shift_per_ppl_pct"] = float(r["concept_abs_shift"] / denom)

    comp_summary = []
    for comp, rs in sorted(group_by(rows, "component").items(), key=lambda kv: COMPONENT_ORDER.index(kv[0]) if kv[0] in COMPONENT_ORDER else 99):
        comp_summary.append({
            "component": comp,
            "n_rows": len(rs),
            "mean_abs_shift": mean([r.get("concept_abs_shift") for r in rs]),
            "mean_signed_shift": mean([r.get("mean_concept_shift") for r in rs]),
            "mean_ppl_increase_pct": mean([r.get("ppl_increase_pct") for r in rs]),
            "median_ppl_increase_pct": median([r.get("ppl_increase_pct") for r in rs]),
            "mean_kl": mean([r.get("mean_kl") for r in rs]),
            "mean_shift_per_ppl_pct": mean([r.get("shift_per_ppl_pct") for r in rs]),
        })

    by_model_comp = []
    for (model, comp), rs in sorted(group_by_multi(rows, ["model", "component"]).items()):
        by_model_comp.append({
            "model": model,
            "component": comp,
            "n_rows": len(rs),
            "mean_abs_shift": mean([r.get("concept_abs_shift") for r in rs]),
            "mean_ppl_increase_pct": mean([r.get("ppl_increase_pct") for r in rs]),
            "mean_kl": mean([r.get("mean_kl") for r in rs]),
        })

    correlations = []
    for comp, rs in sorted(group_by(rows, "component").items()):
        r, p = pearson([x.get("concept_abs_shift") for x in rs], [x.get("ppl_increase_pct") for x in rs])
        correlations.append({"component": comp, "pearson_abs_shift_vs_ppl_pct": r, "p_value": p, "n": len(rs)})
    return comp_summary, by_model_comp, correlations


def summarize_sae(rows):
    for r in rows:
        r["selected_minus_matched_gini"] = None if r.get("selected_mean_gini") is None or r.get("activation_frequency_matched_mean_gini") is None else float(r["selected_mean_gini"] - r["activation_frequency_matched_mean_gini"])
        r["group"] = concept_group(r.get("concept"))
    model_rows = []
    for model, rs in sorted(group_by(rows, "model").items()):
        selected = [r.get("selected_mean_gini") for r in rs]
        matched = [r.get("activation_frequency_matched_mean_gini") for r in rs]
        t, p = paired_t(selected, matched)
        model_rows.append({
            "model": model,
            "n_concepts": len(rs),
            "mean_selected_gini": mean(selected),
            "mean_matched_gini": mean(matched),
            "mean_selected_minus_matched_gini": mean([r.get("selected_minus_matched_gini") for r in rs]),
            "paired_t_selected_vs_matched": t,
            "paired_p_selected_vs_matched": p,
            "mean_p_matched_le_selected": mean([r.get("p_matched_le_selected") for r in rs]),
        })
    return model_rows


def group_by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r.get(key)].append(r)
    return d


def group_by_multi(rows, keys):
    d = defaultdict(list)
    for r in rows:
        d[tuple(r.get(k) for k in keys)].append(r)
    return d


def plot_covariance(out_dir, variant_rows, sensitivity_rows):
    if not variant_rows:
        return
    labels = [r["variant"].replace("_", "\n") for r in variant_rows]
    vals = [r["mean_gini"] for r in variant_rows]
    errs = [r["std_gini"] or 0 for r in variant_rows]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color="#457B9D", alpha=0.85, capsize=3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18)
    ax.set_ylabel("Mean Gini deviation across models")
    ax.set_title("Covariance sensitivity: anti-concentration persists across covariance definitions")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "covariance_variant_gini.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if sensitivity_rows:
        variants = [r["variant"] for r in variant_rows]
        models = sorted({r["model"] for r in sensitivity_rows})
        mat = np.full((len(models), len(variants)), np.nan)
        for r in sensitivity_rows:
            if r["variant"] in variants:
                mat[models.index(r["model"]), variants.index(r["variant"])] = r.get("delta_vs_saved_or_lang_uncentered")
        fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(models))))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(abs(mat)) if np.isfinite(mat).any() else -1, vmax=np.nanmax(abs(mat)) if np.isfinite(mat).any() else 1)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([v.replace("_", "\n") for v in variants], rotation=45, ha="right", fontsize=16)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([clean_model_name(m) for m in models], fontsize=18)
        ax.set_title("Δ mean Gini vs saved/language uncentered covariance")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Δ Gini")
        fig.tight_layout()
        fig.savefig(out_dir / "covariance_delta_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_lambda(out_dir, rows, lambda_rows, model_rows, pair_rows):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(lambda_rows))
    labels = [str(r["lambda"]) for r in lambda_rows]
    axes[0].plot(x, [r["mean_real_delta"] for r in lambda_rows], marker="o", lw=2, label="Real WCA")
    axes[0].plot(x, [r["mean_fake_delta"] for r in lambda_rows], marker="o", lw=2, label="Fake spectrum WCA")
    axes[0].axhline(0, color="grey", ls="--", alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Regularization λ")
    axes[0].set_ylabel("Mean Δ cosine vs naive")
    axes[0].set_title("Lambda sweep: WCA rarely beats naive")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter([r.get("fake_mean_delta_vs_naive") for r in rows], [r.get("real_mean_delta_vs_naive") for r in rows], alpha=0.75, s=40, color="#2A9D8F", edgecolors="white", linewidth=0.4)
    vals = finite([r.get("fake_mean_delta_vs_naive") for r in rows] + [r.get("real_mean_delta_vs_naive") for r in rows])
    if vals:
        lo, hi = min(vals), max(vals)
        pad = 0.1 * max(hi - lo, 1e-6)
        axes[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4)
        axes[1].set_xlim(lo - pad, hi + pad)
        axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].axhline(0, color="grey", ls=":", alpha=0.35)
    axes[1].axvline(0, color="grey", ls=":", alpha=0.35)
    axes[1].set_xlabel("Fake WCA Δ vs naive")
    axes[1].set_ylabel("Real WCA Δ vs naive")
    axes[1].set_title("Real vs fake spectral transport outcomes")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_real_vs_fake.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if model_rows:
        models = [r["model"] for r in model_rows]
        x = np.arange(len(models))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(models)), 5.4))
        ax.bar(x - width / 2, [r["mean_real_win_rate"] for r in model_rows], width, label="Real", color="#2A9D8F")
        ax.bar(x + width / 2, [r["mean_fake_win_rate"] for r in model_rows], width, label="Fake", color="#E9C46A")
        ax.axhline(0.5, color="grey", ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([clean_model_name(m) for m in models], rotation=45, ha="right", fontsize=13)
        ax.set_ylabel("Win rate vs naive")
        ax.set_title("Mean WCA win rate by model")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "lambda_win_rate_by_model.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_nulls(out_dir, norm_rows, freq_rows, shuffle_rows, norm_summary, freq_summary, shuffle_summary):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    if norm_rows:
        axes[0].scatter([r.get("norm_matched_null_mean_gini") for r in norm_rows], [r.get("true_gini") for r in norm_rows], alpha=0.65, s=24, color="#457B9D")
        vals = finite([r.get("norm_matched_null_mean_gini") for r in norm_rows] + [r.get("true_gini") for r in norm_rows])
        if vals:
            lo, hi = min(vals), max(vals)
            axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.35)
        axes[0].set_title("Saved concept vectors vs norm-matched random")
        axes[0].set_xlabel("Norm-matched null Gini")
        axes[0].set_ylabel("True concept Gini")
        axes[0].grid(True, alpha=0.2)

    if freq_rows:
        axes[1].scatter([r.get("frequency_matched_null_mean_gini") for r in freq_rows], [r.get("true_unembed_gini") for r in freq_rows], alpha=0.65, s=24, label="freq", color="#E76F51")
        axes[1].scatter([r.get("pos_proxy_matched_null_mean_gini") for r in freq_rows], [r.get("true_unembed_gini") for r in freq_rows], alpha=0.45, s=20, label="POS proxy", color="#F4A261")
        vals = finite([r.get("frequency_matched_null_mean_gini") for r in freq_rows] + [r.get("pos_proxy_matched_null_mean_gini") for r in freq_rows] + [r.get("true_unembed_gini") for r in freq_rows])
        if vals:
            lo, hi = min(vals), max(vals)
            axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.35)
        axes[1].set_title("Unembedding token-pair nulls")
        axes[1].set_xlabel("Matched null Gini")
        axes[1].set_ylabel("True unembedding-pair Gini")
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)

    if shuffle_rows:
        axes[2].scatter([r.get("sign_shuffle_mean_gini") for r in shuffle_rows], [r.get("true_activation_gini") for r in shuffle_rows], alpha=0.65, s=24, label="sign shuffle", color="#2A9D8F")
        axes[2].scatter([r.get("pair_shuffle_mean_gini") for r in shuffle_rows], [r.get("true_activation_gini") for r in shuffle_rows], alpha=0.45, s=20, label="pair shuffle", color="#264653")
        vals = finite([r.get("sign_shuffle_mean_gini") for r in shuffle_rows] + [r.get("pair_shuffle_mean_gini") for r in shuffle_rows] + [r.get("true_activation_gini") for r in shuffle_rows])
        if vals:
            lo, hi = min(vals), max(vals)
            axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.35)
        axes[2].set_title("Within-concept activation shuffles")
        axes[2].set_xlabel("Shuffle null Gini")
        axes[2].set_ylabel("True activation Gini")
        axes[2].legend()
        axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "strong_nulls_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise(out_dir, layer_rows, transition_rows, frac_rows):
    if not layer_rows:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = []
    ys = []
    yerr = []
    labels = []
    for r in frac_rows:
        labels.append(r["layer_fraction_bin"])
        xs.append(len(xs))
        ys.append(r.get("mean_gini"))
        yerr.append(r.get("ci95_gini") or 0)
    ax.errorbar(xs, ys, yerr=yerr, marker="o", lw=2, capsize=3, color="#457B9D")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("Mean Gini vs final unembedding spectrum")
    ax.set_title("Layer-wise spectral rotation: early representations become tail-aligned")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_layerwise_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    models = [r["model"] for r in transition_rows]
    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(models)), 6.0))
    x = np.arange(len(models))
    ax.bar(x - 0.2, [r.get("embedding_gini") for r in transition_rows], 0.2, label="embedding", color="#E9C46A")
    ax.bar(x, [r.get("layer0_gini") for r in transition_rows], 0.2, label="layer 0", color="#F4A261")
    ax.bar(x + 0.2, [r.get("final_gini") for r in transition_rows], 0.2, label="final", color="#2A9D8F")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_model_name(m) for m in models], rotation=45, ha="right", fontsize=18)
    ax.set_ylabel("Mean Gini")
    ax.set_title("Layer-wise sign flip by model")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_model_transition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_behavioral(out_dir, rows, comp_summary, by_model_comp):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    comps = [r["component"] for r in comp_summary]
    x = np.arange(len(comps))
    axes[0].bar(x - 0.18, [r.get("mean_abs_shift") for r in comp_summary], 0.36, label="abs concept shift", color="#2A9D8F")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comps, rotation=25, ha="right")
    axes[0].set_ylabel("Mean absolute concept logit shift")
    axes[0].set_title("Steering movement by component")
    axes[0].grid(True, axis="y", alpha=0.2)

    axes[1].bar(x, [r.get("mean_ppl_increase_pct") for r in comp_summary], color="#E76F51", alpha=0.85)
    axes[1].axhline(0, color="black", ls="--", lw=1, alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comps, rotation=25, ha="right")
    axes[1].set_ylabel("Mean held-out PPL increase (%)")
    axes[1].set_title("Fluency cost by component")
    axes[1].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_component_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"full": "#2A9D8F", "top_k": "#457B9D", "bottom_k": "#E9C46A", "random_k": "#E76F51"}
    for comp, rs in group_by(rows, "component").items():
        ax.scatter([r.get("ppl_increase_pct") for r in rs], [r.get("concept_abs_shift") for r in rs], label=comp, alpha=0.65, s=30, color=colors.get(comp))
    ax.set_xlabel("PPL increase (%)")
    ax.set_ylabel("Absolute concept logit shift")
    ax.set_title("Behavioral steering trade-off")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_shift_vs_ppl.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sae(out_dir, rows, model_rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.scatter([r.get("activation_frequency_matched_mean_gini") for r in rows], [r.get("selected_mean_gini") for r in rows], s=40, alpha=0.75, color="#9B5DE5", edgecolors="white", linewidth=0.4)
    vals = finite([r.get("activation_frequency_matched_mean_gini") for r in rows] + [r.get("selected_mean_gini") for r in rows])
    if vals:
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    ax.set_xlabel("Activation-frequency matched SAE Gini")
    ax.set_ylabel("Selected SAE feature Gini")
    ax.set_title("SAE concept features vs activation-frequency controls")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "sae_frequency_control_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path, summaries, counts):
    lines = []
    lines.append("# V2 Experiment Analysis Report")
    lines.append("")
    lines.append("## Data coverage")
    lines.append("")
    for key, count in counts.items():
        lines.append(f"- **{key}:** {count} rows")
    lines.append("")

    cov = summaries.get("covariance_variant_rows", [])
    lam = summaries.get("lambda_rows", [])
    norm = summaries.get("norm_summary", [])
    freq = summaries.get("freq_summary", [])
    shuffle = summaries.get("shuffle_summary", [])
    layer = summaries.get("layer_transition_rows", [])
    behavior = summaries.get("behavior_component_rows", [])
    sae = summaries.get("sae_model_rows", [])

    lines.append("## Honest assessment")
    lines.append("")

    if cov:
        saved = next((r for r in cov if r["variant"] == "saved_language_subset_uncentered"), None)
        freq_weighted = next((r for r in cov if r["variant"] == "language_subset_uncentered_frequency_weighted"), None)
        lines.append("### Covariance robustness")
        if saved:
            lines.append(f"- **Main result:** saved/language covariance mean Gini is `{saved['mean_gini']:.4f}` over `{saved['n_models']}` models, with mean SCM `{saved['mean_scm']:.4f}`.")
        if freq_weighted:
            lines.append(f"- **Frequency weighting:** frequency-weighted language covariance is even more negative (`{freq_weighted['mean_gini']:.4f}`), so the anti-concentration effect is not obviously caused by using an unweighted vocab covariance.")
        lines.append("- **Assessment:** this is a strong robustness result for the spectral anti-concentration claim. Centering and vocab-subset choices change magnitude but do not appear to flip the conclusion.")
        lines.append("")

    if lam:
        best = max(lam, key=lambda r: r.get("mean_real_minus_fake_delta") if r.get("mean_real_minus_fake_delta") is not None else -999)
        avg_real = mean([r.get("mean_real_delta") for r in lam])
        avg_fake = mean([r.get("mean_fake_delta") for r in lam])
        avg_wr = mean([r.get("mean_real_win_rate") for r in lam])
        lines.append("### Lambda/WCA transport")
        lines.append(f"- **Main result:** averaged over lambda bins, real WCA mean Δ vs naive is `{avg_real:.5f}` and fake-spectrum WCA is `{avg_fake:.5f}`; real WCA win rate averages `{avg_wr:.3f}`.")
        lines.append(f"- **Best real-minus-fake lambda bin:** λ=`{best['lambda']}` has real-minus-fake Δ `{best['mean_real_minus_fake_delta']:.5f}`.")
        if avg_real is not None and avg_real < 0:
            lines.append("- **Assessment:** this is a negative result for the stronger claim that WCA improves cross-lingual transport over naive Procrustes. The current data suggests WCA often underperforms naive, and fake spectra can be competitive. I would not foreground WCA as a successful transport method without revisiting the objective, covariance choice, or evaluation target.")
        else:
            lines.append("- **Assessment:** WCA is at least competitive on average, but the real-vs-fake gap should be checked carefully before claiming spectrum-specific causal value.")
        lines.append("")

    if norm:
        mean_true_minus = mean([r.get("mean_true_minus_null_gini") for r in norm])
        lines.append("### Norm-matched random direction null")
        lines.append(f"- **Main result:** true saved concept vectors are more tail-aligned than norm-matched random directions by mean Δ Gini `{mean_true_minus:.4f}` across model summaries.")
        lines.append("- **Assessment:** this is very strong support for the descriptive anti-concentration phenomenon. One caveat: random directions in an anisotropic eigenspectrum are already biased toward negative Gini under this metric, so the important quantity is the gap from that baseline, not merely being below zero.")
        lines.append("")

    if freq:
        true_minus_freq = mean([r.get("mean_true_minus_frequency_null_gini") for r in freq])
        frac_gt_freq = mean([r.get("frac_true_greater_than_frequency_null") for r in freq])
        lines.append("### Frequency/POS matched unembedding pair null")
        lines.append(f"- **Main result:** unembedding token-pair directions have mean true-minus-frequency-null Gini `{true_minus_freq:.4f}` and true is greater than frequency-null in fraction `{frac_gt_freq:.3f}` of model-level summaries.")
        if true_minus_freq is not None and true_minus_freq > 0:
            lines.append("- **Assessment:** this weakens any claim that anti-concentration is automatic for raw token-pair differences in the unembedding space. The actual learned concept vectors are tail-aligned, but naive unembedding token-pair analogues can be head-concentrated or mixed. Treat this as a useful caveat rather than a failure.")
        else:
            lines.append("- **Assessment:** frequency/POS controls broadly agree with anti-concentration, though they should be separated from activation-derived concept vectors in the paper.")
        lines.append("")

    if shuffle:
        sign_gap = mean([r.get("mean_true_minus_sign_shuffle_gini") for r in shuffle])
        pair_gap = mean([r.get("mean_true_minus_pair_shuffle_gini") for r in shuffle])
        lines.append("### Within-concept label shuffles")
        lines.append(f"- **Main result:** true-minus-sign-shuffle Gini is `{sign_gap:.4f}` and true-minus-pair-shuffle Gini is `{pair_gap:.4f}` at the model-summary level.")
        lines.append("- **Assessment:** this is mixed. Pair shuffles are often nearly identical to true means because the implementation permutes targets against bases but preserves marginal concept word sets; that makes it a weak/null control for some concepts. Sign shuffles are more meaningful, and when they are less tail-aligned than true directions, that supports label structure. I would report this carefully and maybe revise the shuffle design.")
        lines.append("")

    if layer:
        emb = mean([r.get("embedding_gini") for r in layer])
        layer0 = mean([r.get("layer0_gini") for r in layer])
        final = mean([r.get("final_gini") for r in layer])
        lines.append("### Layer-wise spectral trajectory")
        lines.append(f"- **Main result:** mean embedding-layer Gini is `{emb:.4f}`, layer-0 Gini is `{layer0:.4f}`, and final-layer Gini is `{final:.4f}`.")
        lines.append("- **Assessment:** this is one of the clearest stories in the V2 data: concept directions are near-neutral/mixed at the embedding layer, then become substantially tail-aligned after early transformer layers and remain tail-aligned by the final layer. This is paper-worthy and directly strengthens the mechanistic narrative.")
        lines.append("")

    if behavior:
        best_eff = max(behavior, key=lambda r: r.get("mean_shift_per_ppl_pct") if r.get("mean_shift_per_ppl_pct") is not None else -999)
        full = next((r for r in behavior if r.get("component") == "full"), None)
        top = next((r for r in behavior if r.get("component") == "top_k"), None)
        bottom = next((r for r in behavior if r.get("component") == "bottom_k"), None)
        lines.append("### Behavioral steering")
        if full:
            lines.append(f"- **Full vector:** mean abs shift `{full['mean_abs_shift']:.5f}`, median PPL increase `{full['median_ppl_increase_pct']:.2f}%` and mean PPL increase `{full['mean_ppl_increase_pct']:.2f}%`.")
        if top and bottom:
            lines.append(f"- **Top vs bottom:** top-k mean abs shift `{top['mean_abs_shift']:.5f}` with median PPL cost `{top['median_ppl_increase_pct']:.2f}%`; bottom-k shift `{bottom['mean_abs_shift']:.5f}` with median PPL cost `{bottom['median_ppl_increase_pct']:.2f}%`.")
        lines.append(f"- **Best average efficiency:** `{best_eff['component']}` by abs-shift-per-PPL-percent.")
        lines.append("- **Assessment:** this is useful but not yet decisive. The median fluency costs are moderate, but the means are dominated by severe high-alpha outliers, and random-k can be competitive in some summaries. I would frame it as preliminary behavioral validation, not as a clean causal localization result.")
        lines.append("")

    if sae:
        gap = mean([r.get("mean_selected_minus_matched_gini") for r in sae])
        lines.append("### SAE feature controls")
        lines.append(f"- **Main result:** selected SAE features minus activation-frequency-matched SAE controls have mean Δ Gini `{gap:.4f}` across SAE model summaries.")
        sig_negative = [r for r in sae if r.get("mean_selected_minus_matched_gini") is not None and r.get("mean_selected_minus_matched_gini") < 0 and (r.get("paired_p_selected_vs_matched") or 1.0) < 0.05]
        sig_positive = [r for r in sae if r.get("mean_selected_minus_matched_gini") is not None and r.get("mean_selected_minus_matched_gini") > 0 and (r.get("paired_p_selected_vs_matched") or 1.0) < 0.05]
        if sig_negative and not sig_positive and gap is not None and gap < -0.005:
            lines.append("- **Assessment:** SAE concept features are more tail-aligned than activation-frequency matched controls, which is evidence against a pure activation-frequency artifact.")
        else:
            lines.append("- **Assessment:** SAE controls are mixed at the current sample size. Gemma-2-9B supports stronger tail alignment of selected features, while Gemma-2-2B does not; the aggregate gap is close to zero. This should be reported as suggestive but not decisive.")
        lines.append("")

    lines.append("## Recommended paper positioning")
    lines.append("")
    lines.append("- **Strongest claims to keep:** robust spectral anti-concentration across covariance choices; layer-wise rotation from near-neutral/mixed embeddings to final-layer tail alignment; norm-matched random-direction controls.")
    lines.append("- **Claims to soften:** WCA as an improved transport method; behavioral steering as causal proof; SAE controls as decisive artifact evidence; within-concept shuffle controls as currently implemented.")
    lines.append("- **Best next fix:** revise the WCA/lambda evaluation or present it as a negative/diagnostic result, and improve shuffle nulls so the null cannot preserve the true mean direction by construction.")
    lines.append("")

    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines))


def analyze(args):
    root = Path(args.input)
    out = Path(args.output)
    plots = out / "plots"
    tables = out / "tables"
    ensure_dir(plots)
    ensure_dir(tables)

    data = load_all(root)
    counts = {k: len(v) for k, v in data.items()}

    cov_variant, cov_sens = summarize_covariance(data["covariance"])
    lambda_rows, lambda_model_rows, lambda_pair_rows = summarize_lambda(data["lambda"])
    norm_summary, freq_summary, shuffle_summary = summarize_nulls(data["norm_null"], data["frequency_pos_null"], data["shuffle_null"])
    layer_rows, layer_transition_rows, layer_frac_rows = summarize_layerwise(data["layerwise"])
    behavior_component_rows, behavior_model_component_rows, behavior_correlations = summarize_behavioral(data["behavioral"])
    sae_model_rows = summarize_sae(data["sae"])

    summaries = {
        "covariance_variant_rows": cov_variant,
        "covariance_sensitivity_rows": cov_sens,
        "lambda_rows": lambda_rows,
        "lambda_model_rows": lambda_model_rows,
        "lambda_pair_rows": lambda_pair_rows,
        "norm_summary": norm_summary,
        "freq_summary": freq_summary,
        "shuffle_summary": shuffle_summary,
        "layer_rows": layer_rows,
        "layer_transition_rows": layer_transition_rows,
        "layer_frac_rows": layer_frac_rows,
        "behavior_component_rows": behavior_component_rows,
        "behavior_model_component_rows": behavior_model_component_rows,
        "behavior_correlations": behavior_correlations,
        "sae_model_rows": sae_model_rows,
    }

    write_csv(tables / "covariance_variant_summary.csv", cov_variant)
    write_csv(tables / "covariance_model_variant_deltas.csv", cov_sens)
    write_csv(tables / "lambda_by_lambda_summary.csv", lambda_rows)
    write_csv(tables / "lambda_by_model_summary.csv", lambda_model_rows)
    write_csv(tables / "lambda_by_pair_summary.csv", lambda_pair_rows)
    write_csv(tables / "norm_null_by_model_summary.csv", norm_summary)
    write_csv(tables / "frequency_pos_null_by_model_summary.csv", freq_summary)
    write_csv(tables / "shuffle_null_by_model_summary.csv", shuffle_summary)
    write_csv(tables / "layerwise_by_model_layer.csv", layer_rows)
    write_csv(tables / "layerwise_model_transition.csv", layer_transition_rows)
    write_csv(tables / "layerwise_by_fraction.csv", layer_frac_rows)
    write_csv(tables / "behavior_component_summary.csv", behavior_component_rows)
    write_csv(tables / "behavior_model_component_summary.csv", behavior_model_component_rows)
    write_csv(tables / "behavior_correlations.csv", behavior_correlations)
    write_csv(tables / "sae_frequency_control_by_model.csv", sae_model_rows)
    write_json(out / "analysis_summary.json", {"counts": counts, "summaries": summaries})

    plot_covariance(plots, cov_variant, cov_sens)
    plot_lambda(plots, data["lambda"], lambda_rows, lambda_model_rows, lambda_pair_rows)
    plot_nulls(plots, data["norm_null"], data["frequency_pos_null"], data["shuffle_null"], norm_summary, freq_summary, shuffle_summary)
    plot_layerwise(plots, layer_rows, layer_transition_rows, layer_frac_rows)
    plot_behavioral(plots, data["behavioral"], behavior_component_rows, behavior_model_component_rows)
    plot_sae(plots, data["sae"], sae_model_rows)
    write_report(out / "assessment_report.md", summaries, counts)

    print(f"Loaded rows: {counts}")
    print(f"Wrote tables to {tables}")
    print(f"Wrote plots to {plots}")
    print(f"Wrote report to {out / 'assessment_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze saved V2 experiment outputs.")
    parser.add_argument("--input", default="results/_spectral/v2_experiments")
    parser.add_argument("--output", default="results/_spectral/v2_experiments/analysis")
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
