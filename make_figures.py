#!/usr/bin/env python3
"""Thesis figures from saved analysis output. No GPU, no model loading.

fig1 two nulls      mean Gini gap per concept under both nulls -- the result
fig2 energy CDF     C(k) vs V(k) for real, shuffled, and the random band
fig3 layer profile  mean Gini and pairwise cosine across depth (word concepts)
fig4 split-half     reliability per model with the 0.70 gate

Axes include zero wherever a gap is plotted: a cropped axis makes a null look
like an effect. The random baseline is drawn, not silently subtracted, because
random directions are NOT at zero and that is why baseline choice matters.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"legend.fontsize":8,"xtick.labelsize":8,
    "ytick.labelsize":8,"figure.dpi":150,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False})
C_RANDOM, C_SHUFFLE, C_BAND = "#4477AA", "#CC6677", "#BBBBBB"


def load(path, what):
    p = Path(path)
    if not p.exists():
        print(f"  skip: {what} needs {p}"); return None
    return json.load(open(p))


def fig_two_nulls(summary, outdir):
    per = summary.get("per_concept", {})
    if not per:
        print("  skip: fig1 needs per_concept"); return
    def fam(c):
        return 0 if c.startswith("refusal") else 1 if c.startswith("truth") \
            else 2 if c.startswith("sycophancy") else 3
    names = sorted(per, key=lambda c: (fam(c), c))
    shuf, rand = [], []
    for c in names:
        pm = per[c].get("per_model", {})
        s = [v["shuffle"] for v in pm.values() if "shuffle" in v]
        r = [v["random"] for v in pm.values() if "random" in v]
        shuf.append(np.mean(s) if s else np.nan)
        rand.append(np.mean(r) if r else np.nan)
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.22*len(names))))
    ax.barh(y+h/2, rand, h, label="isotropic random null", color=C_RANDOM, edgecolor="none")
    ax.barh(y-h/2, shuf, h, label="label-shuffle null", color=C_SHUFFLE, edgecolor="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_"," ") for n in names])
    ax.set_xlabel(r"mean Gini gap $\Delta\mathcal{G}$"); ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Measured gap depends on the choice of null")
    out = Path(outdir)/"fig1_two_nulls.pdf"; fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_energy_cdf(model, root, outdir, concept="refusal"):
    sys.path.insert(0, "src")
    try:
        import torch
        from linear_rep_geometry.behavioral import load_concept
        from linear_rep_geometry.spectral_metrics import (eigendecompose,
            compute_cumulative_variance, compute_concept_energy_cdf,
            random_direction_baseline)
    except ImportError as e:
        print(f"  skip: fig2 ({e})"); return
    r = Path(root)/model
    cov_p, cp_p = r/"cov"/"cov_full_vocab.pt", r/"concepts"/f"concept_{concept}.pt"
    if not cov_p.exists() or not cp_p.exists():
        print(f"  skip: fig2 needs {cov_p.name} and {cp_p.name}"); return
    cov = torch.load(cov_p, map_location="cpu", weights_only=True).float()
    evals, evecs = eigendecompose(cov, cache_path=r/"cov"/"eig_full_vocab.pt")
    cumvar = compute_cumulative_variance(evals)
    cp = load_concept(cp_p); L = cp["n_layers"]//2
    D = cp["diffs"][("last_token", L)]
    v = D.mean(dim=0); v = v/v.norm().clamp(min=1e-12)
    cdf_r = compute_concept_energy_cdf(v, evecs)
    rng = np.random.default_rng(0)
    s = torch.tensor(rng.choice([-1.,1.], size=len(D)), dtype=torch.float32).unsqueeze(1)
    vs = (D*s).mean(dim=0); vs = vs/vs.norm().clamp(min=1e-12)
    cdf_s = compute_concept_energy_cdf(vs, evecs)
    rnd = random_direction_baseline(cov.shape[0], evecs, cumvar, n_samples=200, seed=0)
    fig, ax = plt.subplots(figsize=(4.2,3.4))
    ax.plot([0,1],[0,1], color="black", lw=0.8, ls=":", label="uniform")
    ax.fill_between(cumvar, rnd["cdf_lo"], rnd["cdf_hi"], color=C_BAND,
                    alpha=0.6, lw=0, label="random directions (95%)")
    ax.plot(cumvar, cdf_s, color=C_SHUFFLE, lw=1.4, label="label-shuffled")
    ax.plot(cumvar, cdf_r, color=C_RANDOM, lw=1.6, label="concept direction")
    ax.set_xlabel(r"cumulative variance $V(k)$")
    ax.set_ylabel(r"cumulative concept energy $C(k)$")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(f"{model}, {concept.replace('_',' ')}, layer {L}")
    out = Path(outdir)/"fig2_energy_cdf.pdf"; fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_layer_profile(model, root, outdir):
    sys.path.insert(0, "src")
    try:
        import torch
        from linear_rep_geometry.behavioral import load_concept
        from linear_rep_geometry.spectral_metrics import (eigendecompose,
            compute_cumulative_variance, metrics)
    except ImportError as e:
        print(f"  skip: fig3 ({e})"); return
    r = Path(root)/model
    cov_p = r/"cov"/"cov_full_vocab.pt"
    word = sorted((r/"concepts").glob("concept_word_*.pt"))
    if not cov_p.exists() or not word:
        print("  skip: fig3 needs cov_full_vocab.pt and word concepts"); return
    cov = torch.load(cov_p, map_location="cpu", weights_only=True).float()
    evals, evecs = eigendecompose(cov, cache_path=r/"cov"/"eig_full_vocab.pt")
    cumvar = compute_cumulative_variance(evals)
    n_layers = load_concept(word[0])["n_layers"]
    layers, ginis, cosines = [], [], []
    for L in range(n_layers):
        V = []
        for p in word:
            c = load_concept(p); v = c["directions"].get(("last_token", L))
            if v is not None: V.append(v/v.norm().clamp(min=1e-12))
        if len(V) < 2: continue
        V = torch.stack(V)
        g = [metrics(x, evecs, cumvar)["gini"] for x in V]
        C = (V @ V.T).numpy(); iu = np.triu_indices(len(C), k=1)
        layers.append(L); ginis.append(np.mean(g)); cosines.append(np.abs(C[iu]).mean())
    fig, ax = plt.subplots(figsize=(4.6,3.2))
    ax.plot(layers, ginis, color=C_RANDOM, lw=1.6, marker="o", ms=3)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("mean Gini deviation", color=C_RANDOM)
    ax.tick_params(axis="y", labelcolor=C_RANDOM)
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(layers, cosines, color=C_SHUFFLE, lw=1.6, marker="s", ms=3)
    ax2.axhline(0.5, color=C_SHUFFLE, lw=0.6, ls="--", alpha=0.5)
    ax2.set_ylabel(r"mean pairwise $|\cos|$ between concepts", color=C_SHUFFLE)
    ax2.tick_params(axis="y", labelcolor=C_SHUFFLE); ax2.set_ylim(0,1)
    ax.set_title(f"{model}: word directions collapse at mid-depth")
    fig.tight_layout()
    out = Path(outdir)/"fig3_layer_profile.pdf"; fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_split_half(summary, outdir, gate=0.70):
    per = summary.get("per_concept", {}); dropped = summary.get("dropped", [])
    by = {}
    for c, e in per.items():
        for m, v in e.get("per_model", {}).items():
            by.setdefault(m, []).append(v.get("split_half", np.nan))
    for d in dropped:
        by.setdefault(d["model"], []).append(d["split_half"])
    if not by:
        print("  skip: fig4 found no split_half values"); return
    models = sorted(by)
    fig, ax = plt.subplots(figsize=(5.4, 0.42*len(models)+1.4))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        vals = np.array([v for v in by[m] if np.isfinite(v)])
        if not len(vals): continue
        j = rng.uniform(-0.14, 0.14, size=len(vals)); below = vals < gate
        ax.scatter(vals[~below], i+j[~below], s=9, alpha=0.6, color=C_RANDOM, edgecolor="none")
        ax.scatter(vals[below], i+j[below], s=9, alpha=0.7, color=C_SHUFFLE, edgecolor="none")
    ax.axvline(gate, color="black", lw=1.0, ls="--")
    ax.text(gate, len(models)-0.3, f" gate = {gate:.2f}", fontsize=8, va="bottom")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    ax.set_xlabel("split-half cosine"); ax.set_xlim(-1.05, 1.05); ax.invert_yaxis()
    ax.set_title("Direction stability per model")
    out = Path(outdir)/"fig4_split_half.pdf"; fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--summary", default="results/_analysis/summary.json")
    ap.add_argument("--model", default="Qwen2.5-0.5B")
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    s = load(a.summary, "figures 1 and 4")
    print("generating figures ...")
    if s:
        fig_two_nulls(s, a.outdir); fig_split_half(s, a.outdir)
    fig_energy_cdf(a.model, a.results_root, a.outdir, a.concept)
    fig_layer_profile(a.model, a.results_root, a.outdir)
    print("\nWrite the captions yourself -- a caption makes a claim.")


if __name__ == "__main__":
    main()
