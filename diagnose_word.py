#!/usr/bin/env python3
"""Are the word-level Ginis real, or is every word direction the same direction?

42 unrelated word concepts all reported Gini within ~0.0003 of +0.0395 on
Qwen2.5-0.5B. That is not what independent concepts look like.

REAL (mean |cos| < 0.5): directions are distinct, their agreeing Gini is a
genuine property of the spectrum, and the section 6 comparison stands -- the
published effect collapses under a label shuffle.

ARTIFACT (mean |cos| > 0.9): every word direction is dominated by something
shared, so the Gini measures that, not the concepts. The comparison cannot be
reported from this extraction.

Calibration on synthetic data: 0.975 when directions are essentially one
vector, 0.05 when distinct.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "src")
from linear_rep_geometry.behavioral import load_concept
from linear_rep_geometry.spectral_metrics import (
    eigendecompose, compute_cumulative_variance, metrics)


def stack(paths, pooling, layer):
    names, vecs = [], []
    for p in sorted(paths):
        c = load_concept(p)
        L = layer if layer is not None else c["n_layers"] // 2
        v = c["directions"].get((pooling, L))
        if v is None: continue
        names.append(p.stem.replace("concept_", ""))
        vecs.append((v / v.norm().clamp(min=1e-12)).float())
    return names, (torch.stack(vecs) if vecs else torch.empty(0))


def pairwise_report(V, label):
    if len(V) < 2:
        print(f"  {label}: fewer than 2 directions"); return None
    C = (V @ V.T).numpy(); iu = np.triu_indices(len(C), k=1); a = np.abs(C[iu])
    print(f"  {label:22s} n={len(V):3d}  mean |cos| {a.mean():.4f}  "
          f"median {np.median(a):.4f}  min {a.min():.4f}  max {a.max():.4f}")
    return {"n": int(len(V)), "mean_abs_cos": float(a.mean()),
            "median_abs_cos": float(np.median(a)), "max_abs_cos": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--pooling", default="last_token")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--basis", default="full_vocab")
    args = ap.parse_args()

    root = Path(args.results_root) / args.model
    cdir = root / "concepts"
    word = list(cdir.glob("concept_word_*.pt"))
    beh = [p for p in cdir.glob("concept_*.pt")
           if not p.stem.startswith("concept_word_")]
    if not word:
        raise SystemExit(f"no word concepts in {cdir}; run phase 0 first")

    wnames, W = stack(word, args.pooling, args.layer)
    bnames, B = stack(beh, args.pooling, args.layer)
    layer_used = args.layer if args.layer is not None else \
        load_concept(word[0])["n_layers"] // 2
    print(f"{args.model}  pooling={args.pooling}  layer={layer_used}\n")

    print("=== Pairwise cosine between concept directions ===")
    print("  Independent concepts should be largely uncorrelated.")
    print("  Mean |cos| near 1.0 means they are one direction, not many.\n")
    wres = pairwise_report(W, "word-level")
    bres = pairwise_report(B, "behavioural")

    print("\n=== Is there one dominant shared component? ===")
    for V, label in [(W, "word-level"), (B, "behavioural")]:
        if len(V) < 3: continue
        X = V - V.mean(dim=0, keepdim=True)
        ev = (torch.linalg.svdvals(X.float()) ** 2).numpy()
        mu = V.mean(dim=0); mu = mu / mu.norm().clamp(min=1e-12)
        print(f"  {label:22s} PC1 of centred set {ev[0]/ev.sum():.1%}   "
              f"mean |cos| with set mean {(V @ mu).abs().numpy().mean():.4f}")

    print("\n=== Layer profile (word-level) ===")
    cov_p = root / "cov" / f"cov_{args.basis}.pt"
    if cov_p.exists():
        cov = torch.load(cov_p, map_location="cpu", weights_only=True).float()
        evals, evecs = eigendecompose(cov, cache_path=root / "cov" / f"eig_{args.basis}.pt")
        cumvar = compute_cumulative_variance(evals)
        n_layers = load_concept(word[0])["n_layers"]
        print(f"  {'layer':>5s}  {'mean Gini':>10s}  {'sd':>8s}  {'mean |cos|':>10s}")
        for L in range(0, n_layers, max(1, n_layers // 8)):
            _, V = stack(word, args.pooling, L)
            if len(V) < 2: continue
            g = [metrics(v, evecs, cumvar)["gini"] for v in V]
            C = (V @ V.T).numpy(); iu = np.triu_indices(len(C), k=1)
            print(f"  {L:5d}  {np.mean(g):+10.4f}  {np.std(g):8.4f}  "
                  f"{np.abs(C[iu]).mean():10.4f}")
    else:
        print(f"  no {cov_p.name}; skipped")

    print("\n=== Verdict ===")
    if wres is None: return
    m = wres["mean_abs_cos"]
    if m > 0.9:
        print(f"  ARTIFACT. Word directions have mean |cos| {m:.3f} -- they are")
        print("  essentially one direction. Their Gini measures whatever is")
        print("  shared, not the concepts, so the section 6 comparison cannot")
        print("  be reported from this extraction. Check the layer profile: if")
        print("  |cos| is lower at the final layer, Phase 2's mid-depth choice")
        print("  is the problem, not the extraction itself.")
    elif m > 0.5:
        print(f"  SUSPECT. Mean |cos| {m:.3f} is high for unrelated concepts.")
        print("  Report the cosine alongside any Gini so a reader can judge.")
    else:
        print(f"  LOOKS REAL. Mean |cos| {m:.3f} -- directions are distinct, so")
        print("  their agreeing Gini is a genuine property of the spectrum.")
        print("  The section 6 comparison stands.")
    if bres:
        print(f"\n  Behavioural mean |cos| is {bres['mean_abs_cos']:.3f}. If much")
        print("  lower than the word figure, the problem is specific to the")
        print("  word extraction path.")


if __name__ == "__main__":
    main()
