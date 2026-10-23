#!/usr/bin/env python3
"""
Is the published word-level anti-concentration SPECIFIC to concepts?

The thesis tested specificity on behavioural concepts only. The strongest result
of Acharya et al. (2026) -- word-pair concept directions sit in the low-variance
tail, mean Gini -0.356 against an isotropic -0.145 -- never got a specificity
control. This script gives it one, on the published data, in the published
basis, with the published extraction.

THE QUESTION

A concept direction is a difference of means, v = mean(target) - mean(base),
over n word pairs (boy -> girl, king -> queen, ...). Its Gini says where v's
energy sits along the spectrum. The isotropic baseline asks whether v sits
further in the tail than a random Gaussian direction. That is the wrong
comparison for a claim about CONCEPTS, because a difference of means of ANY two
groups of words is not a Gaussian direction: it inherits the shape of the word
activations themselves. So the question is

    does a real concept direction sit further in the tail than the difference
    of means of two ARBITRARY groups of words of the same size?

If yes, the tail is concept-specific for word concepts and the published claim
survives a matched control. If no, the published effect is a property of word
differences in general, which is the word-level analogue of the thesis result.

THE TWO NULLS

  lexical      2n words drawn from the pooled vocabulary of all 42 concept files,
   (primary)   excluding the concept's own words, split at random into two
               groups of n. Real words, same n, same extraction, same basis --
               but no shared relation. The word-level analogue of
               token_id_random: lexical information of any kind, no concept.

  repartition  The concept's OWN 2n words, split at random into two groups of
               n. Holds the vocabulary domain fixed (animal words, verbs, ...)
               and destroys only the contrast. Secondary: it asks whether the
               relation matters beyond the word domain.

Re-pairing words (keeping the two sides, permuting who is paired with whom) is
NOT a null here: a difference of means does not depend on the pairing, so it
would return the real direction every time. The per-pair sign flip of Appendix A
is a different, already-reported null and is not repeated.

Both nulls use the same n as the concept, so the sampling noise in the mean is
matched. The isotropic gap is reported too, so the published comparison sits
beside the new one.

UNIT OF EVIDENCE

    gap = Gini(real) - mean Gini(null)          negative = further in the tail

Per concept: the gap, a z-score and a two-sided permutation p-value. Per model:
the mean gap over concepts (concepts share a basis and are not independent).
Across models: one-sample t-test and Wilcoxon, and TOST against the thesis's
pre-specified placement bound of +/-0.082 Gini units.

USAGE

    # one model (GPU); defaults are the published setting:
    # language-subset basis, final layer, mean pooling
    python scripts/word_specificity.py run --model Qwen/Qwen2.5-0.5B

    # robustness rows, each into its own folder
    python scripts/word_specificity.py run --model Qwen/Qwen2.5-0.5B --layer mid
    python scripts/word_specificity.py run --model Qwen/Qwen2.5-0.5B --basis full_vocab

    # across models (CPU, reads the JSON written above)
    python scripts/word_specificity.py summarize

Outputs go to results/_spectral/word_specificity/<basis>_<layer>/.
Activations are cached there as acts_<model>.pt (git-ignored, ~seconds to rebuild).

Freeze prereg/word_specificity_prereg.json BEFORE the first real run:

    python scripts/freeze_prereg.py --prereg prereg/word_specificity_prereg.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from linear_rep_geometry.spectral_metrics import (  # noqa: E402
    compute_cumulative_variance, empirical_pvalue, one_sample_test,
    random_direction_baseline, tost_equivalence,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz

RESULTS_DIR = REPO_ROOT / "results"
SPECTRAL_DIR = RESULTS_DIR / "_spectral"
OUT_ROOT = SPECTRAL_DIR / "word_specificity"
WORD_PAIRS = REPO_ROOT / "data" / "word_pairs"

POOLING = "mean_over_sequence"      # the published pooling (cos 0.999 in phase 0)
PLACEMENT_SESOI = 0.082             # thesis §3.4: the Gini-scale bound
GEMMA = "gemma-2-2b"
MODELS = ["Qwen2.5-0.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen3-4B",
          "Mistral-7B-v0.3", "Llama-3.2-3B", "Llama-3.1-8B", "gemma-2-2b"]

# Same grouping as scripts/spectral_analysis.py, without the brackets.
CONCEPT_GROUPS = {
    "Verb Morphology": [
        "verb - Ved", "verb - Ving", "verb - 3pSg", "verb - V + er",
        "verb - V + able", "verb - V + ment", "verb - V + tion",
        "Ving - Ved", "Ving - 3pSg", "3pSg - Ved"],
    "Semantic": [
        "male - female", "country - capital", "thing - color", "thing - part",
        "small - big", "lower - upper", "frequent - infrequent"],
    "Semantic (expanded)": [
        "country - language", "name - nationality", "name - occupation",
        "animal - young", "animal - sound", "animal - shelter",
        "hypernyms - animals", "hypernyms - misc", "hyponyms - misc",
        "meronyms - substance", "meronyms - member", "synonyms - intensity",
        "synonyms - exact", "antonyms - gradable", "antonyms - binary"],
    "Grammatical": [
        "adj - comparative", "adj - superlative", "adj - un + adj",
        "adj - adj + ly", "noun - plural", "pronoun - possessive"],
    "Language Pairs": [
        "English - French", "French - German", "French - Spanish",
        "German - Spanish"],
}
GROUP_OF = {c: g for g, cs in CONCEPT_GROUPS.items() for c in cs}


# ─── Pure computation (no model; tested in tests/test_word_specificity.py) ───

def batched_gini(V: torch.Tensor, eigenvectors: torch.Tensor,
                 cumvar: np.ndarray) -> np.ndarray:
    """
    Gini deviation of every row of V, identical to spectral_metrics.metrics.

    Energy is normalised per row, so scale and sign do not matter. Vectorised
    because each concept needs 2 x n_null Ginis.
    """
    P = V.float() @ eigenvectors.float()                  # [B, d]
    E = P ** 2
    E = E / E.sum(dim=1, keepdim=True).clamp(min=1e-12)
    cdf = torch.cumsum(E, dim=1).cpu().numpy()
    return _trapz(cdf - cumvar[None, :], cumvar, axis=1)


def _split_means(X: torch.Tensor, idx: np.ndarray, n: int) -> torch.Tensor:
    """idx: [B, 2n] row indices. Returns mean(first n) - mean(last n), [B, d]."""
    out = []
    for s in range(0, len(idx), 64):                       # bound memory
        I = torch.as_tensor(idx[s:s + 64], dtype=torch.long)
        G = X[I]                                           # [b, 2n, d]
        out.append(G[:, :n].mean(1) - G[:, n:].mean(1))
    return torch.cat(out, 0)


def analyse_concepts(X: torch.Tensor, vocab: dict[str, int],
                     concepts: dict[str, list[tuple[str, str]]],
                     eigenvectors: torch.Tensor, cumvar: np.ndarray,
                     n_null: int = 1000, seed: int = 0,
                     iso_mean: float | None = None) -> list[dict]:
    """
    Real Gini, both nulls and the gaps, for every concept.

    X      [n_words, d] one activation per vocabulary word
    vocab  word -> row of X
    """
    rng = np.random.default_rng(seed)
    all_rows = np.arange(X.shape[0])
    rows = []

    for name, pairs in concepts.items():
        n = len(pairs)
        b = np.array([vocab[w] for w, _ in pairs])
        t = np.array([vocab[w] for _, w in pairs])

        v = (X[t].mean(0) - X[b].mean(0)).unsqueeze(0)
        g_real = float(batched_gini(v, eigenvectors, cumvar)[0])

        # lexical null: 2n other words, random split
        own = np.unique(np.concatenate([b, t]))
        pool = np.setdiff1d(all_rows, own)
        if len(pool) < 2 * n:
            raise ValueError(f"{name}: pool of {len(pool)} words < 2n = {2 * n}")
        idx_lex = np.stack([rng.choice(pool, 2 * n, replace=False)
                            for _ in range(n_null)])
        g_lex = batched_gini(_split_means(X, idx_lex, n), eigenvectors, cumvar)

        # repartition null: the concept's own 2n words, random split
        own_rows = np.concatenate([t, b])
        idx_rep = np.stack([rng.permutation(own_rows) for _ in range(n_null)])
        g_rep = batched_gini(_split_means(X, idx_rep, n), eigenvectors, cumvar)

        row = {"concept": name, "group": GROUP_OF.get(name, "other"),
               "n_pairs": n, "gini": g_real}
        for tag, null in (("lexical", g_lex), ("repartition", g_rep)):
            sd = float(null.std(ddof=1))
            row[f"{tag}_null_mean"] = float(null.mean())
            row[f"{tag}_null_sd"] = sd
            row[f"gap_{tag}"] = float(g_real - null.mean())
            row[f"z_{tag}"] = float((g_real - null.mean()) / sd) if sd > 0 else float("nan")
            row[f"p_{tag}"] = float(empirical_pvalue(g_real, null, tail="two"))
        if iso_mean is not None:
            row["gap_isotropic"] = float(g_real - iso_mean)
        rows.append(row)
    return rows


def model_summary(rows: list[dict]) -> dict:
    """Per-model means over concepts, overall and by group."""
    def agg(rs):
        out = {"n_concepts": len(rs)}
        for k in ("gini", "gap_isotropic", "gap_lexical", "gap_repartition"):
            vals = [r[k] for r in rs if k in r]
            if vals:
                out[f"mean_{k}"] = float(np.mean(vals))
        for tag in ("lexical", "repartition"):
            p = [r[f"p_{tag}"] for r in rs]
            g = [r[f"gap_{tag}"] for r in rs]
            out[f"n_sig_{tag}"] = int(sum(pi < 0.05 for pi in p))
            out[f"n_sig_tail_{tag}"] = int(sum(pi < 0.05 and gi < 0
                                               for pi, gi in zip(p, g)))
        return out

    groups = sorted({r["group"] for r in rows})
    return {"all": agg(rows),
            "by_group": {g: agg([r for r in rows if r["group"] == g])
                         for g in groups}}


# ─── Model side (GPU) ────────────────────────────────────────────────────────

def load_all_pairs(word_pair_dir: Path, min_pairs: int = 4) -> dict:
    from linear_rep_geometry import word_level as wlvl
    out = {}
    for name in wlvl.list_concepts(word_pair_dir):
        pairs = wlvl.load_word_pairs(word_pair_dir, name)
        if len(pairs) >= min_pairs:
            out[name] = pairs
    return out


def load_eigen(model_name: str, basis: str, cov_path: str | None = None):
    """
    Same precedence as test_syntax_claim.load_eigen, so the basis is the one the
    thesis used. Returns (evals, evecs, source).
    """
    def from_cov(p):
        cov = torch.load(p, map_location="cpu", weights_only=True).float()
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx], f"cov:{p}"

    if cov_path:
        return from_cov(cov_path)
    if basis == "language":
        cache = SPECTRAL_DIR / "eigen_cache" / model_name / "en-fr" / "eigen.pt"
        if cache.exists():
            d = torch.load(cache, map_location="cpu", weights_only=True)
            return d["eigenvalues"], d["eigenvectors"], f"cache:{cache}"
        for cand in (RESULTS_DIR / model_name / "en-fr" / "cov_src.pt",
                     RESULTS_DIR / model_name / "cov_en.pt"):
            if cand.exists():
                return from_cov(cand)
        raise SystemExit(
            f"no language-subset covariance for {model_name}. Expected "
            f"results/{model_name}/en-fr/cov_src.pt (rebuild with "
            f"scripts/build_language_cov.py). Refusing to fall back to the full "
            f"vocabulary: that would silently change the basis.")
    if basis == "full_vocab":
        cand = RESULTS_DIR / model_name / "cov" / "cov_full_vocab.pt"
        if cand.exists():
            return from_cov(cand)
        raise SystemExit(f"no {cand}; run scripts/run_phases.py 0 first.")
    raise ValueError(basis)


def extract_word_acts(words, model_id, layer, device, batch_size, cache):
    """One activation per word, context-free, published extraction."""
    if cache.exists():
        d = torch.load(cache, map_location="cpu", weights_only=False)
        if d["words"] == list(words) and d["layer_spec"] == layer:
            print(f"  activations from cache {cache.name}")
            return d["X"], d["layer_index"]

    import linear_rep_geometry as lrg
    from linear_rep_geometry.behavioral import layerwise_activations

    print(f"  loading {model_id} ...")
    model, tok, handler = lrg.load_model(
        model_id, device_map="auto" if device == "cuda" else None)
    model.eval()
    with torch.inference_mode():
        A = layerwise_activations(list(words), model, tok, handler, device,
                                  batch_size=batch_size, max_length=16)[POOLING]
    n_hs = A.shape[1]                         # L + 1 (index 0 = embeddings)
    L = n_hs - 1
    li = L if layer == "final" else L // 2   # thesis: final (normed) or floor(L/2)
    X = A[:, li, :].contiguous()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"words": list(words), "layer_spec": layer, "layer_index": li,
                "X": X}, cache)
    del model
    return X, li


def cmd_run(args):
    model_name = args.model.split("/")[-1]
    out_dir = OUT_ROOT / f"{args.basis}_{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = load_all_pairs(Path(args.word_pairs))
    words = sorted({w for ps in concepts.values() for p in ps for w in p})
    print(f"{model_name}: {len(concepts)} concepts, {len(words)} distinct words, "
          f"basis={args.basis}, layer={args.layer}")

    evals, evecs, src = load_eigen(model_name, args.basis, args.cov)
    cumvar = compute_cumulative_variance(evals)
    d = evecs.shape[0]
    cond = float(evals.max() / evals.min().clamp(min=1e-12))
    print(f"  basis {src}  d={d}  cond={cond:.3g}")

    X, li = extract_word_acts(words, args.model, args.layer, args.device,
                              args.batch_size, out_dir / f"acts_{model_name}.pt")
    if X.shape[1] != d:
        raise SystemExit(f"activation dim {X.shape[1]} != basis dim {d}")
    vocab = {w: i for i, w in enumerate(words)}

    iso = random_direction_baseline(d, evecs, cumvar, n_samples=1000,
                                    seed=args.seed)
    rows = analyse_concepts(X, vocab, concepts, evecs, cumvar,
                            n_null=args.n_null, seed=args.seed,
                            iso_mean=iso["gini_mean"])
    summ = model_summary(rows)
    a = summ["all"]
    print(f"  mean Gini {a['mean_gini']:+.4f}   isotropic {iso['gini_mean']:+.4f}")
    print(f"  gap vs isotropic   {a['mean_gap_isotropic']:+.4f}   (the published comparison)")
    print(f"  gap vs lexical     {a['mean_gap_lexical']:+.4f}   "
          f"sig {a['n_sig_lexical']}/{a['n_concepts']} "
          f"({a['n_sig_tail_lexical']} toward the tail)")
    print(f"  gap vs repartition {a['mean_gap_repartition']:+.4f}   "
          f"sig {a['n_sig_repartition']}/{a['n_concepts']}")
    print(f"  ~{0.05 * a['n_concepts']:.1f} significant expected by chance")

    res = {"model": model_name, "model_id": args.model, "basis": args.basis,
           "basis_source": src, "condition_number": cond, "d": d,
           "layer": args.layer, "layer_index": li, "pooling": POOLING,
           "n_null": args.n_null, "seed": args.seed, "n_words": len(words),
           "isotropic_gini_mean": iso["gini_mean"],
           "isotropic_gini_sd": iso["gini_std"],
           "summary": summ, "concepts": rows}
    with open(out_dir / f"{model_name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out_dir / (model_name + '.json')}")


# ─── Across models ───────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def _fp(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cross_model(per_model: dict[str, float], drop_gemma: bool) -> dict:
    vals = {m: v for m, v in per_model.items() if not (drop_gemma and m == GEMMA)}
    g = list(vals.values())
    out = {"models": list(vals), "n": len(g)}
    if len(g) >= 2:
        out["test"] = one_sample_test(g)
        out["tost"] = tost_equivalence(g, sesoi=PLACEMENT_SESOI)
    return out


def cmd_summarize(args):
    folder = OUT_ROOT / f"{args.basis}_{args.layer}"
    res = {}
    for m in args.models:
        p = folder / f"{m}.json"
        if p.exists():
            res[m] = json.load(open(p))
        else:
            print(f"  missing {p.name}; skipped")
    if len(res) < 2:
        raise SystemExit(f"need at least 2 models in {folder}")

    bases = {r["basis_source"].split(":", 1)[0] + ":" + Path(r["basis_source"].split(":", 1)[1]).name
             for r in res.values()}
    lines = [f"# Word-level specificity ({args.basis} basis, {args.layer} layer)", "",
             f"Basis files: {', '.join(sorted(bases))}. "
             f"Equivalence bound ±{PLACEMENT_SESOI} Gini units. "
             "gap < 0 means the real concept sits further in the tail than the null.", "",
             "## Per model (mean over concepts)", "",
             "| model | concepts | Gini | isotropic | gap iso | gap lexical | sig lex (tail) | gap repartition | sig rep |",
             "|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for m, r in res.items():
        a = r["summary"]["all"]
        lines.append(f"| {m} | {a['n_concepts']} | {_fmt(a['mean_gini'])} | "
                     f"{_fmt(r['isotropic_gini_mean'])} | {_fmt(a['mean_gap_isotropic'])} | "
                     f"{_fmt(a['mean_gap_lexical'])} | {a['n_sig_lexical']} ({a['n_sig_tail_lexical']}) | "
                     f"{_fmt(a['mean_gap_repartition'])} | {a['n_sig_repartition']} |")
        for c in r["concepts"]:
            csv_rows.append({"model": m, **c})

    lines += ["", "## Across models", "",
              "| null | models | n | mean gap | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    out = {"basis": args.basis, "layer": args.layer, "tests": {}}
    groups = ["all"] + list(CONCEPT_GROUPS)
    for null in ("lexical", "repartition", "isotropic"):
        for group in groups:
            per_model = {}
            for m, r in res.items():
                s = r["summary"]["all"] if group == "all" else r["summary"]["by_group"].get(group)
                if s and f"mean_gap_{null}" in s:
                    per_model[m] = s[f"mean_gap_{null}"]
            for drop in (False, True):
                cm = cross_model(per_model, drop)
                key = f"{null}|{group}|{'without_gemma' if drop else 'all'}"
                out["tests"][key] = cm
                if group != "all" and not args.groups:
                    continue
                if "test" not in cm:
                    continue
                t, q = cm["test"], cm["tost"]
                ci = t.get("ci95", [None, None])
                ci90 = q.get("ci90", [None, None])
                label = null + ("" if group == "all" else f" · {group}")
                if null == "lexical" and group == "all":
                    label = "**lexical (primary)**"
                lines.append(
                    f"| {label} | {'without gemma' if drop else 'all'} | {cm['n']} | "
                    f"{_fmt(t['mean'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fp(t.get('p_t'))} | "
                    f"{_fp(t.get('p_wilcoxon'))} | [{_fmt(ci90[0])}, {_fmt(ci90[1])}] | "
                    f"{_fp(q.get('p_tost'))} | {'yes' if q.get('conclusive') else 'no'} |")

    lines += ["", "Reading the primary row:", "",
              "- mean gap clearly below 0 (CI excludes 0): word concepts sit further in the "
              "tail than arbitrary word groups, so the published claim is concept-specific "
              "for word pairs.",
              "- mean gap ≈ 0 and equivalent: the published tilt is a property of word "
              "differences in general, not of concepts.",
              "- neither: inconclusive at this n; report the interval.",
              "", "Group rows (pass --groups) are exploratory and not corrected for multiplicity."]

    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    with open(folder / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    if csv_rows:
        keys = list(dict.fromkeys(k for r in csv_rows for k in r))
        with open(folder / "per_concept.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(lines))
    print(f"\nwrote {folder}/summary.md, summary.json, per_concept.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one model (needs the model; GPU recommended)")
    r.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen2.5-0.5B")
    r.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    r.add_argument("--cov", default=None, help="explicit covariance .pt (overrides --basis lookup)")
    r.add_argument("--layer", choices=("final", "mid"), default="final")
    r.add_argument("--word-pairs", default=str(WORD_PAIRS))
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch-size", type=int, default=32)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="cross-model tests (CPU)")
    s.add_argument("--models", nargs="+", default=MODELS)
    s.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    s.add_argument("--layer", choices=("final", "mid"), default="final")
    s.add_argument("--groups", action="store_true", help="also print per-group rows")
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
