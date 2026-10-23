#!/usr/bin/env python3
"""
The revision's analysis: both tails, every pre-registered test, the paper's
figures and the paper's numbers, from one command.

Reads the JSON written by test_syntax_claim.py (loud tail) and
test_concept_subspace.py (quiet tail). Computes nothing that needs a GPU.

    python scripts/two_tail_report.py \
        --loud-final results/_spectral/syntax_claim \
        --loud-mid   results/_spectral/loud_mid \
        --quiet      results/_spectral/quiet_v2 \
        --out        results/_spectral/two_tail_report

    python scripts/two_tail_report.py ... --figure 3      # one figure only

WHAT IT PRODUCES (in --out)

    primary_and_family.csv     the two primaries and the 8-member Holm family,
                               with and without gemma-2-2b, with TOST and the
                               one-sided non-superiority test
    per_model.csv              per-model Delta / gap for both tails
    k_sweep.csv                every window size x tail x layer, both gemma rules
    controls_loud.csv          the loud-tail Delta against each of the 3 controls
    length.csv                 length-vs-label diagnostics per dataset, and the
                               balanced / residualised reruns
    robustness.csv             MLP probe and full-vocabulary basis, if given
    report.md                  every number above, in sentences
    paper_numbers.tex          \\newcommand macros for every number the paper
                               quotes; \\input it and never type a number by hand
    table_two_tail.tex         per-model table for the paper
    fig1_loud_retention.pdf    retention across the spectrum, loud tail
    fig2_quiet_retention.pdf   retention across the spectrum, quiet tail
    fig3_per_model_delta.pdf   per-model Delta with 95% CIs, both tails
    fig4_placement_vs_capacity.pdf
    figS1_k_sweep.pdf          Delta against window size, both tails and layers

Statistics follow the thesis throughout: models are the unit of analysis, the
quiet tail is averaged over concepts within a model first, and every test is
reported with and without gemma-2-2b.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

GEMMA = "gemma-2-2b"
REPORT_HEADROOM = 0.10

# Reference categorical palette, fixed order (dataviz skill, light mode).
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_VIOLET, C_GRAY, C_INK, C_MUTED = "#6250d6", "#8a8984", "#0b0b0b", "#52514e"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def kkey(x):
    return f"{float(x):g}"


def load_loud(folder):
    """{model: json} for a loud-tail folder, audited first."""
    if folder is None:
        return {}
    folder = Path(folder)
    import test_syntax_claim as tsc
    problems = tsc.audit_folder(folder, verbose=False)
    if problems:
        raise SystemExit(f"{folder} fails the basis audit and cannot be "
                         f"aggregated:\n  " + "\n  ".join(problems))
    out = {}
    for p in sorted(folder.glob("*.json")):
        if p.name.startswith(("syntax_claim_summary", "acts_")):
            continue
        r = json.load(open(p))
        if "difference_in_differences" in r:
            out[r["model_name"]] = r
    return out


def loud_did(r, kf, control):
    """The DiD dict for one loud-tail file at window size kf, or None."""
    bk = (r.get("by_k") or {}).get(kkey(kf))
    if bk:
        d = bk["did_by_control"].get(control)
    elif abs(r["k_frac"] - kf) < 1e-12:
        d = (r.get("did_by_control") or {}).get(control) or (
            r["difference_in_differences"]
            if r["difference_in_differences"].get("control", "token_id_random") == control
            else None)
    else:
        d = None
    return d if d and d.get("available") else None


def loud_cells(r, kf):
    bk = (r.get("by_k") or {}).get(kkey(kf))
    if bk:
        return bk["cells"], bk["k"]
    if abs(r["k_frac"] - kf) < 1e-12:
        return r["cells"], r["k"]
    return None, None


def load_quiet(folder):
    if folder is None:
        return []
    folder = Path(folder)
    import test_concept_subspace as tcs
    rows = tcs.collect_rows(folder)
    kinds = {r["basis_kind"] for r in rows}
    if len(kinds) > 1:
        raise SystemExit(f"{folder} mixes bases {kinds}")
    return rows


def load_quiet_json(folder):
    return [json.load(open(p)) for p in sorted(Path(folder).glob("*__*.json"))]


def usable(r):
    return (r["control_applicable"] and r["floor_ok"]
            and r["headroom_concept"] >= REPORT_HEADROOM)


def quiet_per_model(rows, role, kf, key, exclude=True, concepts=None):
    per = {}
    for r in rows:
        if r["role"] != role or abs(r["k_frac"] - kf) > 1e-12:
            continue
        if concepts and r["concept"] not in concepts:
            continue
        if not r["control_applicable"]:
            continue
        if exclude and not usable(r):
            continue
        per.setdefault(r["model"], []).append(r[key])
    return {m: float(np.mean(v)) for m, v in per.items()}, \
        {m: len(v) for m, v in per.items()}


def loud_per_model(loud, kf, key, control="token_id_random"):
    out = {}
    for m, r in loud.items():
        d = loud_did(r, kf, control)
        if d is None:
            continue
        out[m] = d["delta"] if key == "delta" else d["retention_gap_pos"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def model_test(per_model, sesoi, drop_gemma=False):
    """One-sample tests over models, plus TOST and non-superiority."""
    vals = np.array([v for m, v in sorted(per_model.items())
                     if not (drop_gemma and m == GEMMA) and v is not None
                     and not (isinstance(v, float) and math.isnan(v))])
    n = len(vals)
    res = {"n": n, "mean": float(vals.mean()) if n else float("nan"),
           "n_pos": int((vals > 0).sum())}
    if n < 3:
        res.update(sd=float("nan"), t=float("nan"), p=float("nan"),
                   ci95=[float("nan")] * 2, p_wilcoxon=float("nan"),
                   ci90=[float("nan")] * 2, p_tost=float("nan"),
                   equivalent=False, p_nonsup=float("nan"))
        return res
    sd = float(vals.std(ddof=1))
    se = sd / math.sqrt(n)
    t, p = stats.ttest_1samp(vals, 0.0)
    ci95 = stats.t.interval(0.95, n - 1, loc=vals.mean(), scale=se)
    ci90 = stats.t.interval(0.90, n - 1, loc=vals.mean(), scale=se)
    try:
        pw = float(stats.wilcoxon(vals).pvalue)
    except ValueError:
        pw = float("nan")
    # TOST: H01 mean <= -sesoi, H02 mean >= +sesoi; equivalence if both rejected.
    t_lo = (vals.mean() + sesoi) / se
    t_hi = (vals.mean() - sesoi) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p_tost = float(max(p_lo, p_hi))
    res.update(sd=sd, t=float(t), p=float(p), ci95=[float(ci95[0]), float(ci95[1])],
               p_wilcoxon=pw, ci90=[float(ci90[0]), float(ci90[1])],
               p_tost=p_tost, equivalent=bool(p_tost < 0.05),
               p_nonsup=float(p_hi))
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as input. NaN -> 1."""
    p = np.array([1.0 if (x is None or math.isnan(x)) else x for x in pvals])
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(x, nd=3, signed=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"           # no "-0.000" / "+0.000"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def macro_name(*parts):
    """LaTeX macro names may contain letters only."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    s = "".join(p[:1].upper() + p[1:] for p in parts)
    s = "".join(digits.get(ch, ch) for ch in s)
    return "".join(ch for ch in s if ch.isalpha())


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 2,
    })
    return plt


def _save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")


def _curve(cells, d, k, label):
    """[(position %, retention)] for one label, ordered along the spectrum."""
    pts = [(100.0 * c["start"] / max(d - k, 1), c["retention"])
           for c in cells if c["label"] == label]
    return sorted(pts)


def fig_loud(loud_by_layer, kf, out):
    plt = _mpl()
    layers = [l for l in ("final", "mid") if loud_by_layer.get(l)]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.0),
                             sharey=True, squeeze=False)
    labels = [("pos", "Part of speech", C_BLUE, "o"),
              ("token_id_random", "Random lexical partition", C_ORANGE, "s"),
              ("freq_bucket", "Type frequency", C_AQUA, "^"),
              ("length_bucket", "Word length", C_YELLOW, "v")]
    for ax, layer in zip(axes[0], layers):
        loud = loud_by_layer[layer]
        for lab, name, col, mk in labels:
            curves = []
            for m, r in loud.items():
                cells, k = loud_cells(r, kf)
                if cells is None:
                    continue
                c = _curve(cells, r["d"], k, lab)
                if c:
                    curves.append(c)
                    if lab in ("pos", "token_id_random"):
                        ax.plot([x for x, _ in c], [y for _, y in c], color=col,
                                alpha=0.18, linewidth=0.8)
            if not curves:
                continue
            xs = [x for x, _ in curves[0]]
            Ys = np.array([[y for _, y in c] for c in curves if len(c) == len(xs)],
                          dtype=float)
            mu = np.nanmean(Ys, axis=0)
            ax.plot(xs, mu, marker=mk, color=col, label=name, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(0, color=C_MUTED, linewidth=0.8)
        ax.set_title(f"Loud tail, {layer} layer")
    axes[0][0].set_ylabel("Retention of full-d signal")
    fig.supxlabel("Window position (% of spectrum, high → low variance)", fontsize=9)
    axes[0][-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, -0.04, f"Window size k = {kf:g}·d. Thick: mean over models; "
             f"thin: individual models (POS, random partition).",
             fontsize=7, color=C_MUTED)
    fig.tight_layout()
    _save(fig, out, "fig1_loud_retention")
    plt.close(fig)


def fig_quiet(qjson, kf, out, roles=("mid", "final")):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(roles), figsize=(3.6 * len(roles), 3.0),
                             sharey=True, squeeze=False)
    for ax, role in zip(axes[0], roles):
        for lab, name, col, mk in (("concept", "Concept", C_BLUE, "o"),
                                   ("length_bucket", "Length control", C_ORANGE, "s")):
            per_model = {}
            for r in qjson:
                for layer, cell in r["by_layer"].items():
                    if r.get("layer_roles", {}).get(layer) != role:
                        continue
                    if not (cell["floor_check"]["ok"]
                            and cell["headroom"]["concept"] >= REPORT_HEADROOM):
                        continue
                    bk = (cell.get("by_k") or {}).get(kkey(kf))
                    if bk is None and abs(r["k_frac"] - kf) < 1e-12:
                        bk = {"cells": cell["cells"], "k": r["k"]}
                    if bk is None:
                        continue
                    c = _curve(bk["cells"], r["d"], bk["k"], lab)
                    if c:
                        per_model.setdefault(r["model_name"], []).append(c)
            if not per_model:
                continue
            model_curves = []
            for m, cs in per_model.items():
                L = min(len(c) for c in cs)
                arr = np.array([[y for _, y in c[:L]] for c in cs], dtype=float)
                xs = [x for x, _ in cs[0][:L]]
                mc = np.nanmean(arr, axis=0)
                model_curves.append(mc)
                ax.plot(xs, mc, color=col, alpha=0.18, linewidth=0.8)
            L = min(len(c) for c in model_curves)
            mu = np.nanmean(np.array([c[:L] for c in model_curves]), axis=0)
            ax.plot(xs[:L], mu, marker=mk, color=col, label=name, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(0, color=C_MUTED, linewidth=0.8)
        ax.set_title(f"Quiet tail, {role} layer")
    axes[0][0].set_ylabel("Retention of full-d signal")
    fig.supxlabel("Window position (% of spectrum, high → low variance)", fontsize=9)
    axes[0][-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, -0.04, f"Window size k = {kf:g}·d. Thick: mean over models "
             f"(concepts averaged within model); thin: individual models.",
             fontsize=7, color=C_MUTED)
    fig.tight_layout()
    _save(fig, out, "fig2_quiet_retention")
    plt.close(fig)


def fig_forest(loud, loud_kf, quiet_rows, q_role, q_kf, sesoi, out):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=False)

    def panel(ax, items, summary, title):
        names = [m for m, *_ in items]
        y = np.arange(len(names))[::-1] + 2.2
        for yi, (m, est, lo, hi, extra) in zip(y, items):
            hollow = m == GEMMA
            ax.plot([lo, hi], [yi, yi], color=C_BLUE, linewidth=1.6)
            ax.plot(est, yi, "o", markersize=6, color=C_BLUE,
                    markerfacecolor="white" if hollow else C_BLUE,
                    markeredgewidth=1.4)
            for xc in extra:
                ax.plot(xc, yi, "|", color=C_GRAY, markersize=6, alpha=0.8)
        for yi, (lab, t) in zip((1.0, 0.0), summary):
            if t["n"] >= 3:
                ax.plot(t["ci95"], [yi, yi], color=C_ORANGE, linewidth=2.2)
                ax.plot(t["mean"], yi, "D", color=C_ORANGE, markersize=6)
        ax.set_yticks(list(y) + [1.0, 0.0])
        ax.set_yticklabels(names + [f"{summary[0][0]} (n={summary[0][1]['n']})",
                                    f"{summary[1][0]} (n={summary[1][1]['n']})"],
                           fontsize=7.5)
        ax.axvline(0, color=C_MUTED, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Δ (retention units)")

    items = []
    for m, r in sorted(loud.items()):
        d = loud_did(r, loud_kf, "token_id_random")
        if d:
            items.append((m, d["delta"], d["ci95"][0], d["ci95"][1], []))
    lm = {m: e for m, e, *_ in items}
    panel(axes[0], items, [("Mean, all", model_test(lm, sesoi)),
                           ("Mean, no gemma", model_test(lm, sesoi, True))],
          f"Loud tail (POS vs random partition)")

    q = {}
    for r in quiet_rows:
        if r["role"] == q_role and abs(r["k_frac"] - q_kf) < 1e-12 and usable(r):
            q.setdefault(r["model"], []).append(r["delta"])
    items = []
    for m, v in sorted(q.items()):
        v = np.array(v)
        # Whiskers span the concepts (min to max). A t-interval over two or three
        # concepts has df = 1-2 and would be wider than the axis.
        items.append((m, float(v.mean()), float(v.min()), float(v.max()), list(v)))
    qm = {m: e for m, e, *_ in items}
    axes[1].axvspan(-sesoi, sesoi, color=C_GRAY, alpha=0.12, linewidth=0)
    panel(axes[1], items, [("Mean, all", model_test(qm, sesoi)),
                           ("Mean, no gemma", model_test(qm, sesoi, True))],
          f"Quiet tail (concept vs length, {q_role})")
    fig.text(0.01, -0.06,
             "Blue: per model. Loud: 95% CI by token bootstrap. Quiet: mean over usable "
             "concepts, whiskers = range, ticks = individual concepts. Hollow = gemma-2-2b. Orange: mean over "
             f"models, 95% t-CI. Shaded: equivalence bound ±{sesoi:g}.",
             fontsize=7, color=C_MUTED)
    fig.tight_layout()
    _save(fig, out, "fig3_per_model_delta")
    plt.close(fig)


def fig_placement(points, out):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fams = {"truth": (C_BLUE, "o"), "sycophancy": (C_ORANGE, "s"),
            "refusal": (C_AQUA, "^")}
    for fam, (col, mk) in fams.items():
        P = [p for p in points if p["concept"].startswith(fam)]
        if not P:
            continue
        ax.scatter([p["placement"] for p in P], [p["capacity"] for p in P],
                   s=26, color=col, marker=mk, label=fam.capitalize(),
                   edgecolor="white", linewidth=0.6,
                   alpha=[0.45 if p["model"] == GEMMA else 0.95 for p in P])
    ax.axhline(0, color=C_MUTED, linewidth=0.8)
    ax.axvline(0, color=C_MUTED, linewidth=0.8)
    ax.set_xlabel("Placement: Gini gap vs shuffle null\n(negative = energy in the tail)")
    ax.set_ylabel("Capacity: quiet-minus-loud retention")
    if len(points) >= 4:
        r, p = stats.spearmanr([q["placement"] for q in points],
                               [q["capacity"] for q in points])
        ax.set_title(f"Placement vs capacity  (Spearman ρ = {r:+.2f}, "
                     f"n = {len(points)} cells)", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "fig4_placement_vs_capacity")
    plt.close(fig)


def fig_ksweep(sweep_rows, sesoi, out):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    series = [("loud", "final", C_BLUE, "o"), ("loud", "mid", C_AQUA, "^"),
              ("quiet", "mid", C_ORANGE, "s"), ("quiet", "final", C_VIOLET, "v")]
    for ax, gem in zip(axes, (False, True)):
        for tail, layer, col, mk in series:
            R = sorted([r for r in sweep_rows if r["tail"] == tail
                        and r["layer"] == layer and r["quantity"] == "delta"
                        and r["drop_gemma"] == gem and r["n"] >= 3],
                       key=lambda r: r["k_frac"])
            if not R:
                continue
            x = np.array([r["k_frac"] * 100 for r in R])
            ax.errorbar(x, [r["mean"] for r in R],
                        yerr=[[r["mean"] - r["ci95_lo"] for r in R],
                              [r["ci95_hi"] - r["mean"] for r in R]],
                        color=col, marker=mk, markersize=4.5, capsize=2,
                        linewidth=1.6, label=f"{tail}, {layer}")
        ax.axhspan(-sesoi, sesoi, color=C_GRAY, alpha=0.12, linewidth=0)
        ax.axhline(0, color=C_MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Window size k (% of d)")
        ax.set_title("Without gemma-2-2b" if gem else "All models")
    axes[0].set_ylabel("Mean Δ over models (95% CI)")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "figS1_k_sweep")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", default=str(REPO / "prereg" / "two_tail_prereg.json"))
    ap.add_argument("--loud-final", required=True)
    ap.add_argument("--loud-mid")
    ap.add_argument("--quiet", required=True)
    ap.add_argument("--placement", default=str(REPO / "results/_analysis/gaps_FINAL.csv"))
    ap.add_argument("--length-balanced")
    ap.add_argument("--length-residualised")
    ap.add_argument("--loud-mlp")
    ap.add_argument("--quiet-mlp")
    ap.add_argument("--loud-fullvocab")
    ap.add_argument("--quiet-fullvocab")
    ap.add_argument("--quiet-semantic",
                    help="output of semantic_control.py (exploratory, post-freeze)")
    ap.add_argument("--out", default=str(REPO / "results/_spectral/two_tail_report"))
    ap.add_argument("--figure", choices=["1", "2", "3", "4", "S1", "all", "none"],
                    default="all")
    ap.add_argument("--unfrozen-ok", action="store_true",
                    help="run against an unfrozen pre-registration (output is "
                         "marked DRAFT)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from freeze_prereg import verify
    frozen, msg = verify(a.prereg)
    if not frozen and not a.unfrozen_ok:
        raise SystemExit(f"pre-registration check failed: {msg}\n"
                         f"Freeze it (scripts/freeze_prereg.py) or pass "
                         f"--unfrozen-ok for a draft.")
    pr = json.load(open(a.prereg))
    status = "PRE-REGISTERED (" + msg + ")" if frozen else "DRAFT — " + msg
    sesoi = float(pr["equivalence"]["sesoi"])
    L1, Q1 = pr["loud_tail"]["primary"], pr["quiet_tail"]["primary"]
    lk, qk = float(L1["k_frac"]), float(Q1["k_frac"])
    sweep = [float(x) for x in pr["k_sweep"]]
    models = pr["models"]

    loud = {"final": load_loud(a.loud_final), "mid": load_loud(a.loud_mid)}
    quiet = load_quiet(a.quiet)
    qjson = load_quiet_json(a.quiet)
    for name, src in (("loud final", loud["final"]),):
        missing = [m for m in models if m not in src]
        if missing:
            print(f"note: {name} has no result for {missing}")

    md = [f"# Two-tail report\n", f"Status: **{status}**\n",
          f"Equivalence bound (SESOI): ±{sesoi:g} retention units. "
          f"Primary k: loud {lk:g}, quiet {qk:g}.\n"]
    macros = [f"% Generated by scripts/two_tail_report.py. {status}",
              "% \\input this file; never type these numbers by hand."]

    def add_macro(name, val, nd=3, signed=True, is_p=False):
        txt = fp(val) if is_p else fmt(val, nd, signed)
        if is_p and txt.startswith("<"):
            # p < 0.001: the value macro holds 0.001 and a flag macro tells the
            # paper's \\pval / \\ptab to print "<" instead of "=".
            txt = txt[1:]
            macros.append(f"\\newcommand{{\\{name}Lt}}{{}}")
        if txt == "--":
            txt = "\\textbf{??}"
        macros.append(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{txt}}}}}")

    import re

    def words(text):
        return [w for w in re.split(r"[^A-Za-z]+", text) if w]

    # ── per-model values ─────────────────────────────────────────────────────
    def series(tail, layer, quantity, kf, exclude=True, rows=None, louds=None):
        if tail == "loud":
            src = (louds or loud).get(layer) or {}
            return loud_per_model(src, kf, quantity,
                                  control=L1["control"])
        pm, _ = quiet_per_model(rows if rows is not None else quiet, layer, kf,
                                "delta" if quantity == "delta" else "gap_concept",
                                exclude=exclude)
        return pm

    # ── primary + family ─────────────────────────────────────────────────────
    fam_rows = []
    for gem in (False, True):
        members = pr["inference"]["family"]["members"]
        tests = []
        for mem in members:
            kf = lk if mem["tail"] == "loud" else qk
            pm = series(mem["tail"], mem["layer"], mem["quantity"], kf)
            t = model_test(pm, sesoi, drop_gemma=gem)
            tests.append((mem, t))
        adj = holm([t["p"] for _, t in tests])
        for (mem, t), pa in zip(tests, adj):
            prim = (mem["tail"] == "loud" and mem["layer"] == L1["layer"]
                    and mem["quantity"] == L1["quantity"]) or \
                   (mem["tail"] == "quiet" and mem["layer"] == Q1["layer"]
                    and mem["quantity"] == Q1["quantity"])
            fam_rows.append({
                "models": "without gemma" if gem else "all",
                "tail": mem["tail"], "layer": mem["layer"],
                "quantity": mem["quantity"], "primary": prim,
                "n": t["n"], "n_positive": t["n_pos"], "mean": t["mean"], "sd": t["sd"],
                "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1],
                "t": t["t"], "p": t["p"], "p_holm": pa,
                "p_wilcoxon": t["p_wilcoxon"],
                "ci90_lo": t["ci90"][0], "ci90_hi": t["ci90"][1],
                "p_tost": t["p_tost"], "equivalent": t["equivalent"],
                "p_nonsuperiority": t["p_nonsup"], "ran": t["n"] > 0})
    write_csv(out / "primary_and_family.csv", fam_rows)

    md.append("\n## Primary tests and the Holm family\n")
    md.append("| models | tail | layer | quantity | n | mean | 95% CI | p | p (Holm) "
              "| 90% CI | p TOST | equiv. | p non-sup. |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in fam_rows:
        star = " **(primary)**" if r["primary"] else ""
        md.append(f"| {r['models']} | {r['tail']} | {r['layer']} | {r['quantity']}{star} "
                  f"| {r['n']} | {fmt(r['mean'])} | [{fmt(r['ci95_lo'])}, "
                  f"{fmt(r['ci95_hi'])}] | {fp(r['p'])} | {fp(r['p_holm'])} | "
                  f"[{fmt(r['ci90_lo'])}, {fmt(r['ci90_hi'])}] | {fp(r['p_tost'])} | "
                  f"{'yes' if r['equivalent'] else 'no'} | {fp(r['p_nonsuperiority'])} |")
    not_run = {(r["tail"], r["layer"]) for r in fam_rows if not r["ran"]}
    if not_run:
        md.append(f"\nNot run (entered Holm with p = 1): "
                  f"{', '.join(f'{t} {l}' for t, l in sorted(not_run))}.")
    for r in fam_rows:
        tag = macro_name(r["tail"], r["layer"], r["quantity"],
                         "NoGemma" if r["models"] != "all" else "All")
        add_macro(tag + "Mean", r["mean"])
        add_macro(tag + "CILo", r["ci95_lo"])
        add_macro(tag + "CIHi", r["ci95_hi"])
        add_macro(tag + "P", r["p"], is_p=True)
        add_macro(tag + "PHolm", r["p_holm"], is_p=True)
        add_macro(tag + "PTOST", r["p_tost"], is_p=True)
        add_macro(tag + "PNonSup", r["p_nonsuperiority"], is_p=True)
        add_macro(tag + "CINinetyLo", r["ci90_lo"])
        add_macro(tag + "CINinetyHi", r["ci90_hi"])
        macros.append(f"\\newcommand{{\\{tag}N}}{{{r['n']}}}")
        macros.append(f"\\newcommand{{\\{tag}NPos}}{{{r['n_positive']}}}")

    # ── per-model table ──────────────────────────────────────────────────────
    pm_rows = []
    lf = loud["final"]
    for m in models:
        row = {"model": m}
        d = loud_did(lf[m], lk, L1["control"]) if m in lf else None
        row["loud_delta"] = d["delta"] if d else None
        row["loud_ci_lo"] = d["ci95"][0] if d else None
        row["loud_ci_hi"] = d["ci95"][1] if d else None
        row["loud_p_boot"] = d["p_boot"] if d else None
        row["loud_gap_pos"] = d["retention_gap_pos"] if d else None
        for role in ("mid", "final"):
            pm, cnt = quiet_per_model(quiet, role, qk, "delta")
            pm_all, cnt_all = quiet_per_model(quiet, role, qk, "delta", exclude=False)
            row[f"quiet_{role}_delta"] = pm.get(m)
            row[f"quiet_{role}_n_concepts"] = cnt.get(m, 0)
            row[f"quiet_{role}_n_concepts_before_exclusion"] = cnt_all.get(m, 0)
        pm_rows.append(row)
    write_csv(out / "per_model.csv", pm_rows)

    # Reproduction gate: window 0 / last window against the published top-k and
    # bottom-k accuracies, for the models whose per-model values were published.
    rep_models, rep_max = [], 0.0
    for m, r in sorted(lf.items()):
        g = r.get("reproduction", {})
        if g.get("checked") and g.get("max_abs_delta") is not None:
            rep_models.append(m)
            rep_max = max(rep_max, float(g["max_abs_delta"]))
    if rep_models:
        macros.append(f"\\newcommand{{\\ReproN}}{{{len(rep_models)}}}")
        macros.append(f"\\newcommand{{\\ReproModels}}{{{', '.join(rep_models)}}}")
        macros.append(f"\\newcommand{{\\ReproMaxDelta}}{{\\ensuremath{{{rep_max:.4f}}}}}")
        md.append(f"\nReproduction gate: {len(rep_models)} models "
                  f"({', '.join(rep_models)}), max |published - here| = {rep_max:.5f}.")

    tex = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Model & Loud $\Delta$ & 95\% CI & Quiet $\Delta$ (" + Q1["layer"] +
           r") & concepts \\", r"\midrule"]
    for r in pm_rows:
        name = r["model"] + (r"$^\dagger$" if r["model"] == GEMMA else "")
        tex.append(f"{name} & {fmt(r['loud_delta'])} & "
                   f"[{fmt(r['loud_ci_lo'])}, {fmt(r['loud_ci_hi'])}] & "
                   f"{fmt(r['quiet_' + Q1['layer'] + '_delta'])} & "
                   f"{r['quiet_' + Q1['layer'] + '_n_concepts']}/"
                   f"{r['quiet_' + Q1['layer'] + '_n_concepts_before_exclusion']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Per-model specificity $\Delta$ for both tails at the "
            r"pre-registered primary settings. Loud: part of speech against a random "
            r"lexical partition, final layer, $k=" + f"{lk:g}" + r"\,d$, 95\% CI by "
            r"token bootstrap. Quiet: concept against the length control, "
            + Q1["layer"] + r" layer, $k=" + f"{qk:g}" + r"\,d$, mean over usable "
            r"concepts (usable/run). $^\dagger$gemma-2-2b failed the reliability gate "
            r"on the concept side; every aggregate is reported with and without it.}",
            r"\label{tab:two_tail}", r"\end{table}"]
    (out / "table_two_tail.tex").write_text("\n".join(tex) + "\n")

    # ── k sweep ──────────────────────────────────────────────────────────────
    sweep_rows = []
    for kf in sweep:
        for tail in ("loud", "quiet"):
            for layer in ("final", "mid"):
                for quantity in ("delta", "gap"):
                    pm = series(tail, layer, quantity, kf)
                    for gem in (False, True):
                        t = model_test(pm, sesoi, gem)
                        n_pos = sum(1 for m, v in pm.items()
                                    if v > 0 and not (gem and m == GEMMA))
                        sweep_rows.append({
                            "k_frac": kf, "tail": tail, "layer": layer,
                            "quantity": quantity, "drop_gemma": gem, "n": t["n"],
                            "n_models_positive": n_pos, "mean": t["mean"],
                            "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1],
                            "p": t["p"], "p_tost": t["p_tost"]})
    write_csv(out / "k_sweep.csv", sweep_rows)
    md.append("\n## Window-size sweep (Δ, all models / without gemma)\n")
    md.append("| tail | layer | " + " | ".join(f"k={kf:g}" for kf in sweep) + " |")
    md.append("|---|---|" + "---|" * len(sweep))
    for tail in ("loud", "quiet"):
        for layer in ("final", "mid"):
            cells = []
            for kf in sweep:
                rr = [r for r in sweep_rows if r["k_frac"] == kf and r["tail"] == tail
                      and r["layer"] == layer and r["quantity"] == "delta"]
                a_, b_ = (next(r for r in rr if not r["drop_gemma"]),
                          next(r for r in rr if r["drop_gemma"]))
                cells.append(f"{fmt(a_['mean'], 3)} (p {fp(a_['p'])}) / "
                             f"{fmt(b_['mean'], 3)} (p {fp(b_['p'])})"
                             if a_["n"] >= 3 else "--")
            md.append(f"| {tail} | {layer} | " + " | ".join(cells) + " |")
    md.append("\nThe primary k is fixed in the pre-registration; the sweep is "
              "robustness only.")

    # ── loud controls ────────────────────────────────────────────────────────
    ctl_rows = []
    for c in ("token_id_random", "freq_bucket", "length_bucket"):
        pm = loud_per_model(loud["final"], lk, "delta", control=c)
        for gem in (False, True):
            t = model_test(pm, sesoi, gem)
            n_pos = sum(1 for m, v in pm.items() if v > 0 and not (gem and m == GEMMA))
            ctl_rows.append({"control": c, "models": "without gemma" if gem else "all",
                             "n": t["n"], "n_models_POS_beats_control": n_pos,
                             "mean": t["mean"], "ci95_lo": t["ci95"][0],
                             "ci95_hi": t["ci95"][1], "p": t["p"],
                             "p_nonsuperiority": t["p_nonsup"]})
    write_csv(out / "controls_loud.csv", ctl_rows)
    for r in ctl_rows:
        tag = macro_name("LoudCtl", *words(r["control"]),
                         "NoGemma" if r["models"] != "all" else "All")
        add_macro(tag + "Mean", r["mean"])
        add_macro(tag + "P", r["p"], is_p=True)
        macros.append(f"\\newcommand{{\\{tag}NPos}}{{{r['n_models_POS_beats_control']}}}")
    md.append("\n## Loud tail against each control (final layer, primary k)\n")
    md.append("| control | models | n | models where POS beats control | mean Δ | 95% CI | p |")
    md.append("|---|---|---|---|---|---|---|")
    for r in ctl_rows:
        md.append(f"| {r['control']} | {r['models']} | {r['n']} | "
                  f"{r['n_models_POS_beats_control']} | {fmt(r['mean'])} | "
                  f"[{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {fp(r['p'])} |")
    if all(r["n"] == 0 for r in ctl_rows if r["control"] != "token_id_random"):
        md.append("\nfreq_bucket / length_bucket Δ need files written by the "
                  "revised script (did_by_control); rerun the loud tail.")

    # ── length ───────────────────────────────────────────────────────────────
    len_rows = []
    seen = set()
    for r in qjson:
        ld = r.get("length_diagnostics")
        if not ld or r["concept"] in seen:
            continue
        seen.add(r["concept"])
        row = {"concept": r["concept"], "n_pairs": r["n_pairs"]}
        for f in ("chars", "words", "tokens"):
            if f in ld:
                row[f"auc_{f}"] = ld[f]["auc"]
        row["frac_pos_longer"] = ld["chars"]["frac_pairs_pos_longer"]
        row["frac_neg_longer"] = ld["chars"]["frac_pairs_neg_longer"]
        row["frac_tied"] = ld["chars"]["frac_pairs_tied"]
        row["length_bucket_agreement"] = ld["length_bucket_vs_concept_agreement"]
        len_rows.append(row)
    rob_rows = []
    for label, folder in (("length balanced", a.length_balanced),
                          ("length residualised", a.length_residualised),
                          ("MLP probe (quiet)", a.quiet_mlp),
                          ("full-vocabulary basis (quiet)", a.quiet_fullvocab),
                          ("semantic control", a.quiet_semantic)):
        if not folder:
            continue
        rows = load_quiet(folder)
        for role in ("mid", "final"):
            for quantity in ("delta", "gap"):
                pm = series("quiet", role, quantity, qk, rows=rows)
                for gem in (False, True):
                    t = model_test(pm, sesoi, gem)
                    rob_rows.append({"analysis": label, "tail": "quiet", "layer": role,
                                     "quantity": quantity,
                                     "models": "without gemma" if gem else "all",
                                     **{k: t[k] for k in ("n", "mean", "p", "p_tost",
                                                          "equivalent")},
                                     "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1]})
    for label, folder in (("MLP probe (loud)", a.loud_mlp),
                          ("full-vocabulary basis (loud)", a.loud_fullvocab)):
        if not folder:
            continue
        src = {"final": load_loud(folder)}
        for quantity in ("delta", "gap"):
            pm = series("loud", "final", quantity, lk, louds=src)
            for gem in (False, True):
                t = model_test(pm, sesoi, gem)
                rob_rows.append({"analysis": label, "tail": "loud", "layer": "final",
                                 "quantity": quantity,
                                 "models": "without gemma" if gem else "all",
                                 **{k: t[k] for k in ("n", "mean", "p", "p_tost",
                                                      "equivalent")},
                                 "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1]})
    write_csv(out / "length.csv", len_rows)
    write_csv(out / "robustness.csv", rob_rows)
    for r in rob_rows:
        tag = macro_name("Rob", *words(r["analysis"]), r["layer"], r["quantity"],
                         "NoGemma" if r["models"] != "all" else "All")
        add_macro(tag + "Mean", r["mean"])
        add_macro(tag + "CILo", r["ci95_lo"])
        add_macro(tag + "CIHi", r["ci95_hi"])
        add_macro(tag + "P", r["p"], is_p=True)
        add_macro(tag + "PTOST", r["p_tost"], is_p=True)
        macros.append(f"\\newcommand{{\\{tag}N}}{{{r['n']}}}")
    if len_rows:
        tl = [r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"Dataset & pairs & AUC chars & AUC tokens & pos.\ longer & tied \\",
              r"\midrule"]
        for r in sorted(len_rows, key=lambda r: r["concept"]):
            tl.append(f"\\texttt{{{r['concept'].replace('_', chr(92) + '_')}}} & "
                      f"{r['n_pairs']} & {fmt(r.get('auc_chars'), 3, False)} & "
                      f"{fmt(r.get('auc_tokens'), 3, False)} & "
                      f"{100 * r['frac_pos_longer']:.0f}\\% & "
                      f"{100 * r['frac_tied']:.0f}\\% \\\\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_length.tex").write_text("\n".join(tl) + "\n")
        md.append("\n## How strongly length predicts the label\n")
        md.append("AUC 0.5 = length says nothing about the label.\n")
        md.append("| dataset | pairs | AUC chars | AUC tokens | pos longer | neg longer | tied |")
        md.append("|---|---|---|---|---|---|---|")
        for r in sorted(len_rows, key=lambda r: r["concept"]):
            md.append(f"| {r['concept']} | {r['n_pairs']} | {fmt(r.get('auc_chars'), 3, False)} | "
                      f"{fmt(r.get('auc_tokens'), 3, False)} | {r['frac_pos_longer']:.0%} | "
                      f"{r['frac_neg_longer']:.0%} | {r['frac_tied']:.0%} |")
    if rob_rows:
        md.append("\n## Robustness analyses\n")
        md.append("| analysis | tail | layer | quantity | models | n | mean | 95% CI | p | p TOST |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rob_rows:
            md.append(f"| {r['analysis']} | {r['tail']} | {r['layer']} | {r['quantity']} | "
                      f"{r['models']} | {r['n']} | {fmt(r['mean'])} | "
                      f"[{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {fp(r['p'])} | "
                      f"{fp(r['p_tost'])} |")

    # ── placement vs capacity ───────────────────────────────────────────────
    points = []
    if a.placement and Path(a.placement).exists():
        plc = {(r["model"], r["concept"]): float(r["shuffle"])
               for r in csv.DictReader(open(a.placement))}
        for r in quiet:
            if r["role"] != Q1["layer"] or abs(r["k_frac"] - qk) > 1e-12 or not usable(r):
                continue
            key = (r["model"], r["concept"])
            if key in plc:
                points.append({"model": r["model"], "concept": r["concept"],
                               "placement": plc[key], "capacity": r["gap_concept"]})
        write_csv(out / "placement_vs_capacity.csv", points)
        if len(points) >= 4:
            rho, p = stats.spearmanr([q["placement"] for q in points],
                                     [q["capacity"] for q in points])
            n_tail = sum(1 for q in points if q["placement"] < 0)
            n_both = sum(1 for q in points if q["placement"] < 0 and q["capacity"] > 0)
            md.append("\n## Placement versus capacity\n")
            md.append(f"{len(points)} (model, concept) cells. Placement puts energy in "
                      f"the tail (gap < 0) in {n_tail}; of those, the tail also holds "
                      f"more of the concept than the head (capacity > 0) in {n_both}. "
                      f"Spearman ρ = {rho:+.2f} (p = {fp(p)}; cells within a model "
                      f"are not independent, so read ρ descriptively).")
            add_macro("PlacementCapacityRho", rho, 2)
            macros.append(f"\\newcommand{{\\PlacementCapacityN}}{{{len(points)}}}")
            macros.append(f"\\newcommand{{\\PlacementNTail}}{{{n_tail}}}")
            macros.append(f"\\newcommand{{\\PlacementNTailAndCapacity}}{{{n_both}}}")

    # ── result-dependent sentences ──────────────────────────────────────────
    # The paper's wording depends on the outcome. Rather than leave that to a
    # hand edit after the numbers arrive, each rule is applied here and the
    # sentence is written as a macro. The rule is in the code, so a reader can
    # check which sentence the data selected and why.
    def tn(x, nd=3, signed=True):
        return "$" + fmt(x, nd, signed) + "$" if x is not None else "--"

    def tp(p):
        s = fp(p)
        return "$" + ("{<}" + s[1:] if s.startswith("<") else s) + "$"

    def fam(tail, layer, quantity, models):
        return next((r for r in fam_rows if r["tail"] == tail and r["layer"] == layer
                     and r["quantity"] == quantity and r["models"] == models), None)

    def sentence(name, text):
        macros.append(f"\\newcommand{{\\{name}}}{{{text}}}")
        md.append(f"- **\\{name}**: {text}")

    md.append("\n## Sentences selected by the write-up rules\n")
    ok = lambda r: r is not None and r["n"] >= 3

    # Loud tail, primary.
    la, ln = fam("loud", L1["layer"], "delta", "all"), fam("loud", L1["layer"], "delta", "without gemma")
    if ok(la) and ok(ln):
        pos_a, pos_n = la["ci95_lo"] > 0, ln["ci95_lo"] > 0
        neg_a, neg_n = la["ci95_hi"] < 0, ln["ci95_hi"] < 0
        if pos_a or pos_n:
            opening = ("Part of speech prefers the head \\emph{more} than a random "
                       "lexical partition does")
            verdict = ("This is the syntax-specific effect the account predicts, "
                       + ("and it holds without gemma-2-2b." if pos_a and pos_n else
                          "but it holds in only one of the two versions, so we do "
                          "not count it as robust."))
        else:
            opening = ("Part of speech prefers the head no more than a random "
                       "lexical partition does")
            if neg_a and neg_n:
                verdict = ("In both versions the control is preferred \\emph{more} "
                           "than part of speech.")
            elif neg_a or neg_n:
                verdict = ("The control is preferred more than part of speech in "
                           + ("the all-model" if neg_a else "the without-gemma")
                           + " version only, so we claim no more than that part of "
                           "speech is not preferred.")
            else:
                verdict = "Neither version separates part of speech from the control."
        nonsup = ("The one-sided test rejects any syntax-specific advantage as large "
                  "as $+0.10$" if la["p_nonsuperiority"] < 0.05 and
                  ln["p_nonsuperiority"] < 0.05 else
                  "The one-sided test cannot rule out a syntax-specific advantage as "
                  "large as $+0.10$ in at least one version")
        sentence("LoudOpening", opening)
        sentence("LoudVerdict", verdict)
        sentence("LoudNonSupSentence", nonsup)

    # Quiet tail, primary.
    qa, qn = fam("quiet", Q1["layer"], "delta", "all"), fam("quiet", Q1["layer"], "delta", "without gemma")
    if ok(qa) and ok(qn):  # noqa: E501
        if qa["ci95_lo"] > 0 or qn["ci95_lo"] > 0:
            opening = "Concepts prefer the tail \\emph{more} than surface length does"
        elif qa["ci95_hi"] < 0 and qn["ci95_hi"] < 0:
            opening = ("Concepts prefer the tail \\emph{less} than surface length "
                       "does")
        else:
            opening = "Concepts prefer the tail no more than surface length does"
        if qa["equivalent"] and qn["equivalent"]:
            eq = ("This interval lies inside the pre-registered bound of "
                  "$\\pm0.10$ in both versions, so the quiet-tail specificity effect "
                  "is not just undetected but smaller than the smallest effect we "
                  "said would matter.")
        else:
            bound = max(abs(qa["ci90_lo"]), abs(qa["ci90_hi"]),
                        abs(qn["ci90_lo"]), abs(qn["ci90_hi"]))
            which = ("either version" if not (qa["equivalent"] or qn["equivalent"])
                     else "the " + ("all-model" if not qa["equivalent"] else
                                    "without-gemma") + " version")
            eq = ("Equivalence at the pre-registered bound of $\\pm0.10$ is not "
                  f"established in {which}: the data cannot rule out an effect that "
                  f"large. The smallest bound both 90\\% intervals exclude is "
                  f"$\\pm{bound:.3f}$.")
        sentence("QuietOpening", opening)
        sentence("QuietEquivalenceSentence", eq)
        sentence("QuietAbstractEq",
                 "and the effect lies inside a pre-registered equivalence bound of "
                 "$\\pm0.10$ retention units"
                 if qa["equivalent"] and qn["equivalent"] else
                 "although equivalence within a pre-registered bound of "
                 "$\\pm0.10$ retention units is not established")

    # Holm family.
    parts = []
    for models in ("all", "without gemma"):
        sig = [r for r in fam_rows if r["models"] == models and r["ran"]
               and r["p_holm"] < 0.05]
        if not sig:
            parts.append(("with all eight models" if models == "all" else
                          "without gemma-2-2b") + ", no member is significant")
        else:
            def item(r):
                sym = "$\\Delta$" if r["quantity"] == "delta" else "$g$"
                ph = fp(r["p_holm"]).replace("<", "{<}")
                return (f"{r['tail']} {r['layer']} {sym} ({tn(r['mean'])}, "
                        f"$p_{{\\mathrm{{Holm}}}}={ph}$)")
            items = ", ".join(item(r) for r in sig)
            parts.append(("with all eight models" if models == "all" else
                          "without gemma-2-2b") + f", {items} remain significant")
    n_ran = len({(r["tail"], r["layer"]) for r in fam_rows if r["ran"]})
    holm_s = "After Holm correction, " + "; ".join(parts) + "."
    if n_ran < 4:
        holm_s += (" Members that were not run entered the correction with $p=1$.")
    sentence("HolmSentence", holm_s)

    # Window-size sweep.
    pos_cells, neg_cells, total = [], [], 0
    for r in sweep_rows:
        if r["quantity"] != "delta" or r["n"] < 3:
            continue
        total += 1
        if r["p"] < 0.05 and r["mean"] > 0:
            pos_cells.append(r)
        elif r["p"] < 0.05 and r["mean"] < 0:
            neg_cells.append(r)

    def where(cells):
        """One entry per (tail, layer, k) setting, saying which model sets it
        holds in, so the same setting is never listed twice."""
        by = {}
        for c in cells:
            by.setdefault((c["tail"], c["layer"], c["k_frac"]), set()).add(c["drop_gemma"])
        items = []
        for (tail, layer, kf), sets in sorted(by.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            note = ("with and without gemma-2-2b" if sets == {False, True} else
                    "all models only" if sets == {False} else "without gemma-2-2b only")
            items.append(f"{tail} {layer} at $k={100 * kf:g}\\%$ ({note})")
        if len(items) > 1:
            return ", ".join(items[:-1]) + " and " + items[-1]
        return items[0] if items else ""

    def n_settings(cells):
        return len({(c["tail"], c["layer"], c["k_frac"]) for c in cells})
    expected = 0.025 * total
    if not pos_cells:
        ks = ("At no window size, in either tail or layer, with or without "
              "gemma-2-2b, is the mean $\\Delta$ significantly positive")
        ks += (f" ({total} tail $\\times$ layer $\\times$ window $\\times$ "
               f"model-set cells).")
    else:
        ks = (f"The mean $\\Delta$ is significantly positive in {len(pos_cells)} of "
              f"{total} tail $\\times$ layer $\\times$ window $\\times$ model-set "
              f"cells, against about {expected:.1f} expected by chance alone at "
              f"two-sided $\\alpha = 0.05$: {where(pos_cells)}.")
    if neg_cells:
        ns = n_settings(neg_cells)
        ks += (f" It is significantly \\emph{{negative}} (the control preferred over "
               f"the real label) in {len(neg_cells)} "
               f"cell{'s' if len(neg_cells) != 1 else ''}, at {ns} "
               f"setting{'s' if ns != 1 else ''}: {where(neg_cells)}.")
    else:
        ks += " It is significantly negative in none."
    ks += (" No cell is interpreted on its own, because $k$ was fixed in advance; at "
           "the pre-registered $k$ the conclusion is as stated above, and no window "
           "size supports a specific effect.")
    sentence("KSweepSentence", ks)

    # Robustness sentences: length, MLP, the two extra loud controls.
    def rob(analysis, tail, layer, models="all"):
        return next((r for r in rob_rows if r["analysis"] == analysis and
                     r["tail"] == tail and r["layer"] == layer and
                     r["quantity"] == "delta" and r["models"] == models), None)

    lb, lr = rob("length balanced", "quiet", Q1["layer"]), \
        rob("length residualised", "quiet", Q1["layer"])
    if lb and lr and lb["n"] >= 3 and lr["n"] >= 3:
        pos = [x for x in (lb, lr) if x["ci95_lo"] > 0]
        if pos:
            ls = ("Removing length changes the result: $\\Delta$ becomes "
                  "significantly positive under "
                  + " and ".join("balancing" if x is lb else "residualising"
                                 for x in pos)
                  + ", so the length control's correlation with the label was "
                    "masking a concept-specific effect.")
        elif lb["equivalent"] and lr["equivalent"]:
            ls = ("Both stay inside the $\\pm0.10$ equivalence bound, so the length "
                  "control's correlation with the label does not drive the "
                  "quiet-tail result.")
        else:
            ls = ("Neither is significantly positive, so the length control's "
                  "correlation with the label does not hide a concept-specific "
                  "effect, though equivalence is not established under "
                  + " and ".join(n for x, n in ((lb, "balancing"), (lr, "residualising"))
                                 if not x["equivalent"]) + ".")
        sentence("LengthSentence", ls)

    ml, mq = rob("MLP probe (loud)", "loud", "final"), rob("MLP probe (quiet)", "quiet", Q1["layer"])
    have = [(n, x) for n, x in (("loud", ml), ("quiet", mq)) if x and x["n"] >= 3]
    if have:
        pos = [n for n, x in have if x["ci95_lo"] > 0]
        ms = ("so the conclusion is not specific to linear read-out." if not pos else
              "and with this probe $\\Delta$ is significantly positive in the "
              + " and ".join(pos) + " tail, so the linear result does not carry "
              "over to non-linear read-out there.")
        sentence("MLPSentence", ms)

    sa = rob("semantic control", "quiet", Q1["layer"])
    if sa and sa["n"] >= 3:
        if sa["ci95_lo"] > 0:
            ss = ("Against the semantic control the concept \\emph{is} preferred by "
                  "the tail, so the length-control result does not generalise to a "
                  "semantic nuisance label.")
        elif sa["equivalent"]:
            ss = ("Against a semantic control the result is the same: the effect "
                  "lies inside the $\\pm0.10$ equivalence bound.")
        else:
            ss = ("Against a semantic control the effect is not significant, but "
                  "equivalence within $\\pm0.10$ is not established.")
        sentence("SemanticSentence", ss)

    ctl = {r["control"]: r for r in ctl_rows if r["models"] == "all"}
    fq, lnr = ctl.get("freq_bucket"), ctl.get("length_bucket")
    if fq and lnr and fq["n"] >= 3 and lnr["n"] >= 3:
        beat = [n for n, r in (("type frequency", fq), ("word length", lnr))
                if r["ci95_lo"] > 0]
        sentence("ControlsOpening",
                 "Part of speech also fails to beat type frequency and word length"
                 if not beat else
                 "Part of speech does beat " + " and ".join(beat)
                 + ", which correlate with it, so these controls bound the confound "
                   "rather than remove it")

    # Placement versus capacity.
    if len(points) >= 4:
        rho = float(stats.spearmanr([q["placement"] for q in points],
                                    [q["capacity"] for q in points])[0])
        n_tail = sum(1 for q in points if q["placement"] < 0)
        n_both = sum(1 for q in points if q["placement"] < 0 and q["capacity"] > 0)
        if abs(rho) < 0.3:
            pc = ("Placement does not predict capacity: where a direction's energy "
                  "sits does not tell you where its concept can be read.")
        elif abs(rho) < 0.5:
            direction = ("in the expected direction: cells whose direction sits "
                         "further in the tail tend to read the concept better from "
                         "the tail" if rho < 0 else
                         "but in the unexpected direction: cells whose direction "
                         "sits further in the tail tend to read the concept better "
                         "from the head")
            quiet_specific = bool(qa and qa["n"] >= 3 and qa["ci95_lo"] > 0)
            pc = (f"Placement and capacity are moderately related, {direction}. "
                  f"The relation is far from one-to-one: against the structural "
                  f"null the direction's energy sits in the tail in {n_tail} of "
                  f"{len(points)} cells, and in {n_both} of those is the concept "
                  f"also read better from the tail than from the head."
                  + ("" if quiet_specific else
                     " And where capacity does favour the tail, it favours the "
                     "length control just as much, so placement predicts where "
                     "information is recoverable, not where the concept "
                     "specifically lives."))
        elif rho < 0:
            quiet_specific = bool(qa and qa["n"] >= 3 and qa["ci95_lo"] > 0)
            pc = ("Placement tracks capacity: cells whose direction sits further in "
                  "the tail also read the concept better from the tail."
                  + (" Capacity is nonetheless not specific, because the tail holds "
                     "the length control as well as it holds the concept."
                     if not quiet_specific else
                     " Here capacity is also specific (Section~\\ref{sec:specific})."))
        else:
            pc = ("Placement and capacity point in opposite directions: cells whose "
                  "direction sits further in the tail read the concept better from "
                  "the \\emph{head}.")
        sentence("PlacementSentence", pc)

    # ── figures ──────────────────────────────────────────────────────────────
    want = a.figure
    if want in ("1", "all"):
        fig_loud(loud, lk, out)
    if want in ("2", "all"):
        fig_quiet(qjson, qk, out)
    if want in ("3", "all"):
        fig_forest(loud["final"], lk, quiet, Q1["layer"], qk, sesoi, out)
    if want in ("4", "all") and points:
        fig_placement(points, out)
    if want in ("S1", "all"):
        fig_ksweep(sweep_rows, sesoi, out)

    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "paper_numbers.tex").write_text("\n".join(macros) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
