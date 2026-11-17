#!/usr/bin/env python3
"""
Is the DICTIONARY half of the published dual geometry specific to concepts?

Acharya et al. (2026) report a dual geometry: word-pair concept directions built
from ACTIVATIONS anti-concentrate (sit in the low-variance tail), while the same
contrasts built from static UNEMBEDDING ROWS concentrate (sit in the
high-variance head). scripts/word_specificity.py tested the activation half
against a lexical null and found it is not concept-specific. This script runs
the same test on the unembedding half, with the same concepts, the same basis,
the same Gini and the same statistics. Only the word vectors change:

    activation half   x(w) = the model's hidden state for word w (word_specificity.py)
    dictionary half   x(w) = gamma(w), the unembedding row of w's token (this script)

WHY THE PUBLISHED DICTIONARY CHECK IS NOT ENOUGH

The published comparison (scripts/fix_hole5_unembed_null.py) set concept
directions, mean over n pairs of gamma(target) - gamma(base), against the
difference of ONE random token pair drawn from the WHOLE vocabulary. Three
things differ between those two quantities, not one:

    1. the concept relation                       (what the claim is about)
    2. real English words vs arbitrary tokens     (sub-words, other scripts, padding)
    3. an average over n pairs vs a single pair   (averaging changes the spectrum)

The nulls below separate them:

    token_pair   published null: 1 random pair from all W_U rows (seed 42).
     (reproduction)  concept vs this = the published comparison.
    vocab_n      2n random tokens from the tokenizer's vocabulary, split into two
                 groups of n. Matches (3), not (2).
    lexical      2n random WORDS from the pooled concept vocabulary (excluding the
     (PRIMARY)   concept's own), split into two groups of n. Matches (2) and (3);
                 only the relation differs. Identical to word_specificity.py.
    repartition  the concept's own 2n words, random split (secondary).
    isotropic    norm-matched Gaussian directions (reference).

UNIT OF EVIDENCE

    gap = Gini(concept) - mean Gini(null)        POSITIVE = further toward the head

The published account predicts gap > 0. Per model: mean over concepts. Across
models: t-test, Wilcoxon, TOST with the +/-0.082 Gini bound.

TOKENS

A word's unembedding row exists only if the word is one token. Primary rule
("published", identical to scripts/artifact_check_unembed.py): use the token of
the bare word if it is a single token, else of " word" if that is. Secondary
rule ("space_first"): prefer " word". Pairs with a multi-token word, or whose two
words map to the same token, are dropped; a concept needs >= 4 clean pairs.

USAGE

    python scripts/dictionary_specificity.py check --model Qwen/Qwen2.5-0.5B   # loader self-check
    python scripts/dictionary_specificity.py run   --model Qwen/Qwen2.5-0.5B   # one model, CPU is fine
    python scripts/dictionary_specificity.py summarize                          # across models

Outputs: results/_spectral/dictionary_specificity/<basis>_<rule>/.
Freeze prereg/dictionary_specificity_prereg.json BEFORE the first real run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

# Reuse the activation-side implementation so both halves are measured alike.
_spec = importlib.util.spec_from_file_location("word_specificity",
                                               SCRIPT_DIR / "word_specificity.py")
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)

SPECTRAL_DIR = REPO_ROOT / "results" / "_spectral"
OUT_ROOT = SPECTRAL_DIR / "dictionary_specificity"
ACT_ROOT = SPECTRAL_DIR / "word_specificity"
WORD_PAIRS = REPO_ROOT / "data" / "word_pairs"
SESOI = ws.PLACEMENT_SESOI            # +/-0.082 Gini units, as in the thesis
GEMMA = ws.GEMMA
MODELS = ws.MODELS
MIN_CLEAN_PAIRS = 4
MIN_CONCEPTS = 10                     # a model enters only with >= 10 usable concepts
MIN_MODELS = 6                        # the primary test is valid only with >= 6 models
PUBLISHED_PAIR_SEED = 42              # fix_hole5_unembed_null.py


# ─── Tokens ──────────────────────────────────────────────────────────────────

def word_token_id(encode, word: str, rule: str):
    """Single token id for `word`, or None. `encode(text) -> list[int]`."""
    bare = encode(word)
    spaced = encode(" " + word)
    if rule == "published":            # artifact_check_unembed.filter_single_token_pairs
        if len(bare) == 1:
            return bare[0]
        if len(spaced) == 1:
            return spaced[-1]
        return None
    if rule == "space_first":
        if len(spaced) == 1:
            return spaced[0]
        if len(bare) == 1:
            return bare[0]
        return None
    raise ValueError(rule)


def clean_concepts(concepts: dict, encode, rule: str,
                   min_pairs: int = MIN_CLEAN_PAIRS):
    """
    Keep single-token pairs. Returns (clean, word_to_id, report) where clean maps
    concept -> [(base, target)] and every word in it has an id in word_to_id.
    """
    cache: dict[str, int | None] = {}

    def tid(w):
        if w not in cache:
            cache[w] = word_token_id(encode, w, rule)
        return cache[w]

    clean, report = {}, {}
    for name, pairs in concepts.items():
        kept = [(b, t) for b, t in pairs
                if tid(b) is not None and tid(t) is not None and tid(b) != tid(t)]
        report[name] = {"n_pairs": len(pairs), "n_clean": len(kept)}
        if len(kept) >= min_pairs:
            clean[name] = kept
    word_to_id = {w: i for w, i in cache.items() if i is not None}
    return clean, word_to_id, report


def word_matrix(W: torch.Tensor, clean: dict, word_to_id: dict):
    """
    Rows of W for every distinct token used by the clean concepts. Words that
    share a token share a row, so the lexical null never draws a concept's own
    token. Returns (X, vocab) with vocab: word -> row of X.
    """
    ids = sorted({word_to_id[w] for ps in clean.values() for p in ps for w in p})
    row_of = {t: r for r, t in enumerate(ids)}
    X = W[torch.as_tensor(ids, dtype=torch.long)].float()
    vocab = {w: row_of[word_to_id[w]]
             for ps in clean.values() for p in ps for w in p}
    return X, vocab


# ─── Nulls on the whole vocabulary ───────────────────────────────────────────

def published_token_pair_null(W, eigenvectors, cumvar, n=1000,
                              seed=PUBLISHED_PAIR_SEED) -> np.ndarray:
    """fix_hole5_unembed_null.random_unembed_null: 1 random pair over all rows."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(W.shape[0], size=2, replace=False) for _ in range(n)])
    V = W[idx[:, 0]].float() - W[idx[:, 1]].float()
    keep = V.norm(dim=1) > 1e-12
    return ws.batched_gini(V[keep], eigenvectors, cumvar)


def vocab_n_null(W, n_tokens: int, n: int, eigenvectors, cumvar, n_null, rng):
    """2n random tokens from the tokenizer vocabulary, two groups of n."""
    idx = np.stack([rng.choice(n_tokens, 2 * n, replace=False) for _ in range(n_null)])
    return ws.batched_gini(ws._split_means(W, idx, n), eigenvectors, cumvar)


def analyse(W, n_tokens, clean, word_to_id, eigenvectors, cumvar, n_null=1000,
            seed=0, iso_mean=None, pair_null=None):
    """Concept rows with lexical, repartition (via word_specificity),
    vocab_n, token_pair and isotropic gaps."""
    X, vocab = word_matrix(W, clean, word_to_id)
    rows = ws.analyse_concepts(X, vocab, clean, eigenvectors, cumvar,
                               n_null=n_null, seed=seed, iso_mean=iso_mean)
    rng = np.random.default_rng(seed + 1)
    for r in rows:
        g_real = r["gini"]
        nul = vocab_n_null(W, n_tokens, r["n_pairs"], eigenvectors, cumvar, n_null, rng)
        sd = float(nul.std(ddof=1))
        r["vocab_n_null_mean"] = float(nul.mean())
        r["gap_vocab_n"] = float(g_real - nul.mean())
        r["z_vocab_n"] = float((g_real - nul.mean()) / sd) if sd > 0 else float("nan")
        r["p_vocab_n"] = float(empirical_pvalue(g_real, nul, tail="two"))
        if pair_null is not None:
            r["gap_token_pair"] = float(g_real - pair_null.mean())
    return rows


NULLS = ("lexical", "repartition", "vocab_n", "token_pair", "isotropic")


def model_summary(rows):
    def agg(rs):
        out = {"n_concepts": len(rs)}
        for k in ["gini"] + [f"gap_{n}" for n in NULLS]:
            vals = [r[k] for r in rs if k in r]
            if vals:
                out[f"mean_{k}"] = float(np.mean(vals))
        for tag in ("lexical", "repartition", "vocab_n"):
            p = [r[f"p_{tag}"] for r in rs]
            g = [r[f"gap_{tag}"] for r in rs]
            out[f"n_sig_{tag}"] = int(sum(pi < 0.05 for pi in p))
            out[f"n_sig_head_{tag}"] = int(sum(pi < 0.05 and gi > 0 for pi, gi in zip(p, g)))
        return out
    groups = sorted({r["group"] for r in rows})
    return {"all": agg(rows),
            "by_group": {g: agg([r for r in rows if r["group"] == g]) for g in groups}}


# ─── Model side ──────────────────────────────────────────────────────────────

def _light_unembedding(model_id: str):
    """Read only the unembedding tensor from the safetensors shards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    tied = bool(getattr(text_cfg, "tie_word_embeddings",
                        getattr(cfg, "tie_word_embeddings", False)))
    keys = (["model.embed_tokens.weight", "embed_tokens.weight"] if tied else
            ["lm_head.weight"])
    try:
        index = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))
        wmap = index["weight_map"]
        key = next(k for k in keys if k in wmap)
        path = hf_hub_download(model_id, wmap[key])
    except Exception:
        path = hf_hub_download(model_id, "model.safetensors")
        key = None
    with safe_open(path, framework="pt") as f:
        if key is None:
            key = next(k for k in keys if k in f.keys())
        W = f.get_tensor(key)
    return W.float(), f"safetensors:{key}{' (tied)' if tied else ''}"


def _full_unembedding(model_id: str):
    """The published route (artifact_check_unembed.py): load the model."""
    import linear_rep_geometry as lrg
    model, tok, _ = lrg.load_model(model_id, device_map="cpu")
    W = lrg.get_lm_head(model).weight.detach().cpu().float()
    del model
    return W, "lrg.load_model:get_lm_head"


def load_unembedding(model_id: str, loader: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # as the handlers do
    W, src = (_light_unembedding if loader == "light" else _full_unembedding)(model_id)
    return W, tok, src


def cmd_check(args):
    """Light loader must equal the published loader exactly. Prints no Gini."""
    a, _, sa = load_unembedding(args.model, "light")
    b, _, sb = load_unembedding(args.model, "full")
    diff = float((a - b).abs().max()) if a.shape == b.shape else float("inf")
    print(f"{args.model}: light {tuple(a.shape)} [{sa}]  full {tuple(b.shape)} [{sb}]  "
          f"max |diff| = {diff:.3g}")
    if diff > 1e-3:
        raise SystemExit("loaders disagree; use --loader full for every run")
    print("loaders agree")


def cmd_run(args):
    model_name = args.model.split("/")[-1]
    out_dir = OUT_ROOT / f"{args.basis}_{args.rule}"
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = ws.load_all_pairs(Path(args.word_pairs))
    W, tok, src_w = load_unembedding(args.model, args.loader)
    n_tokens = min(len(tok), W.shape[0])          # never sample padding rows
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    clean, word_to_id, report = clean_concepts(concepts, encode, args.rule)
    print(f"{model_name}: W_U {tuple(W.shape)} [{src_w}], tokenizer {len(tok)}; "
          f"{len(clean)}/{len(concepts)} concepts with >= {MIN_CLEAN_PAIRS} "
          f"single-token pairs (rule={args.rule})")

    evals, evecs, src_b = ws.load_eigen(model_name, args.basis)
    cumvar = compute_cumulative_variance(evals)
    d = evecs.shape[0]
    if W.shape[1] != d:
        raise SystemExit(f"unembedding dim {W.shape[1]} != basis dim {d}")
    cond = float(evals.max() / evals.min().clamp(min=1e-12))

    iso = random_direction_baseline(d, evecs, cumvar, n_samples=1000, seed=args.seed)
    pair_null = published_token_pair_null(W, evecs, cumvar)
    rows = analyse(W, n_tokens, clean, word_to_id, evecs, cumvar,
                   n_null=args.n_null, seed=args.seed, iso_mean=iso["gini_mean"],
                   pair_null=pair_null)
    summ = model_summary(rows)
    a = summ["all"]
    print(f"  mean concept Gini {a['mean_gini']:+.4f}")
    for n in NULLS:
        print(f"  gap vs {n:<12s} {a.get('mean_gap_' + n, float('nan')):+.4f}")
    print(f"  lexical: sig {a['n_sig_lexical']}/{a['n_concepts']} "
          f"({a['n_sig_head_lexical']} toward the head)")

    res = {"model": model_name, "model_id": args.model, "basis": args.basis,
           "basis_source": src_b, "condition_number": cond, "d": d,
           "rule": args.rule, "loader": args.loader, "unembedding_source": src_w,
           "unembedding_shape": list(W.shape), "tokenizer_size": len(tok),
           "n_null": args.n_null, "seed": args.seed,
           "n_concepts_total": len(concepts), "n_concepts_clean": len(clean),
           "clean_report": report,
           "isotropic_gini_mean": iso["gini_mean"],
           "token_pair_null_mean": float(pair_null.mean()),
           "token_pair_null_n": int(len(pair_null)),
           "summary": summ, "concepts": rows}
    with open(out_dir / f"{model_name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out_dir / (model_name + '.json')}")


# ─── Across models ───────────────────────────────────────────────────────────

def cross(per_model: dict, drop_gemma: bool):
    vals = {m: v for m, v in per_model.items() if not (drop_gemma and m == GEMMA)}
    g = list(vals.values())
    out = {"models": list(vals), "n": len(g)}
    if len(g) >= 2:
        out["test"] = one_sample_test(g)
        out["tost"] = tost_equivalence(g, sesoi=SESOI)
    return out


def verdict(t_all: dict, t_nog: dict) -> str:
    """Pre-registered rule (prereg/dictionary_specificity_prereg.json)."""
    ok = all("test" in t for t in (t_all, t_nog))
    if not ok:
        return "not testable"
    if all(t["test"]["ci95"][0] > 0 for t in (t_all, t_nog)):
        return "specific (head)"
    if all(t["tost"].get("equivalent") for t in (t_all, t_nog)):
        return "not specific"
    if all(t["test"]["ci95"][1] < 0 for t in (t_all, t_nog)):
        return "reversed (tail)"
    return "inconclusive"


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{0:.{nd}f}" if round(x, nd) == 0 else f"{x:+.{nd}f}"


def _p(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cmd_summarize(args):
    folder = OUT_ROOT / f"{args.basis}_{args.rule}"
    res = {m: json.load(open(folder / f"{m}.json"))
           for m in args.models if (folder / f"{m}.json").exists()}
    missing = [m for m in args.models if m not in res]
    excluded = {m: r["summary"]["all"]["n_concepts"] for m, r in res.items()
                if r["summary"]["all"]["n_concepts"] < MIN_CONCEPTS}
    res = {m: r for m, r in res.items() if m not in excluded}
    if len(res) < 2:
        raise SystemExit(f"need at least 2 usable models in {folder}")

    lines = [f"# Dictionary-side specificity ({args.basis} basis, token rule {args.rule})", "",
             "gap > 0: the concept direction built from unembedding rows sits further "
             f"toward the HEAD than the null. Equivalence bound ±{SESOI} Gini units.", ""]
    if missing:
        lines += [f"Missing models: {', '.join(missing)}.", ""]
    if excluded:
        lines += ["Excluded (< %d usable concepts): %s." % (MIN_CONCEPTS, ", ".join(
            f"{m} ({n})" for m, n in excluded.items())), ""]
    lines += ["## Per model (mean over concepts)", "",
              "| model | concepts | Gini | gap lexical | sig (head) | gap vocab_n | gap token_pair | gap repartition | gap isotropic |",
              "|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for m, r in res.items():
        a = r["summary"]["all"]
        lines.append(f"| {m} | {a['n_concepts']}/{r['n_concepts_total']} | {_f(a['mean_gini'])} | "
                     f"{_f(a['mean_gap_lexical'])} | {a['n_sig_lexical']} ({a['n_sig_head_lexical']}) | "
                     f"{_f(a['mean_gap_vocab_n'])} | {_f(a['mean_gap_token_pair'])} | "
                     f"{_f(a['mean_gap_repartition'])} | {_f(a['mean_gap_isotropic'])} |")
        csv_rows += [{"model": m, **c} for c in r["concepts"]]

    lines += ["", "## Across models", "",
              "| null | models | n | mean gap | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    out = {"basis": args.basis, "rule": args.rule, "tests": {}}
    for null in NULLS:
        per_model = {m: r["summary"]["all"][f"mean_gap_{null}"] for m, r in res.items()
                     if f"mean_gap_{null}" in r["summary"]["all"]}
        for drop in (False, True):
            cm = cross(per_model, drop)
            out["tests"][f"{null}|{'without_gemma' if drop else 'all'}"] = cm
            if "test" not in cm:
                continue
            t, q = cm["test"], cm["tost"]
            ci90 = q.get("ci90") or [None, None]
            label = "**lexical (primary)**" if null == "lexical" else null
            lines.append(f"| {label} | {'without gemma' if drop else 'all'} | {cm['n']} | "
                         f"{_f(t['mean'])} | [{_f(t['ci95'][0])}, {_f(t['ci95'][1])}] | "
                         f"{_p(t.get('p_t'))} | {_p(t.get('p_wilcoxon'))} | "
                         f"[{_f(ci90[0])}, {_f(ci90[1])}] | {_p(q.get('p_tost'))} | "
                         f"{'yes' if q.get('equivalent') else 'no'} |")

    v = verdict(out["tests"]["lexical|all"], out["tests"]["lexical|without_gemma"])
    if len(res) < MIN_MODELS:
        v = "not testable"
    out["excluded"] = excluded
    out["missing"] = missing
    out["verdict"] = v
    if args.basis == "language" and args.rule == "published":
        lines += ["", f"**Primary verdict (lexical null, frozen rule): {v.upper()}**"]

    # reproduction of the published comparison (concept vs 1 random token pair)
    pos = [m for m, r in res.items() if r["summary"]["all"]["mean_gap_token_pair"] > 0]
    out["reproduction"] = {"n_positive": len(pos), "n": len(res),
                           "reproduces": len(pos) >= 6}
    lines += ["", f"Published comparison (concept vs one random token pair): concept "
              f"further toward the head in {len(pos)} of {len(res)} models "
              f"({'reproduces' if len(pos) >= 6 else 'does NOT reproduce'}; gate: at least 6)."]

    # dual contrast with the activation half (descriptive)
    act = ACT_ROOT / f"{args.basis}_final"
    diffs = {}
    for m in res:
        p = act / f"{m}.json"
        if p.exists():
            ga = json.load(open(p))["summary"]["all"]["mean_gap_lexical"]
            diffs[m] = (res[m]["summary"]["all"]["mean_gap_lexical"], ga)
    if diffs:
        lines += ["", "## Dual contrast (descriptive; activation half from word_specificity, "
                  "final layer, all concepts)", "",
                  "| model | dictionary gap (lexical) | activation gap (lexical) |", "|---|---|---|"]
        lines += [f"| {m} | {_f(a)} | {_f(b)} |" for m, (a, b) in diffs.items()]
        out["dual_contrast"] = {m: {"dictionary": a, "activation": b} for m, (a, b) in diffs.items()}

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

    c = sub.add_parser("check", help="light loader == published loader (no results)")
    c.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="one model (CPU is fine)")
    r.add_argument("--model", required=True)
    r.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    r.add_argument("--rule", choices=("published", "space_first"), default="published")
    r.add_argument("--loader", choices=("light", "full"), default="light")
    r.add_argument("--word-pairs", default=str(WORD_PAIRS))
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="cross-model tests (CPU)")
    s.add_argument("--models", nargs="+", default=MODELS)
    s.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    s.add_argument("--rule", choices=("published", "space_first"), default="published")
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
