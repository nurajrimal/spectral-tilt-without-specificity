#!/usr/bin/env python3
"""
The two sides of the dual geometry, measured on IDENTICAL word pairs.

dictionary_specificity.py found that word-pair contrasts built from unembedding
rows are specific (further toward the head than random word groups), while
word_specificity.py found that contrasts built from activations are not. But
the two runs used different pairs: the unembedding side needs single-token
words, so it kept 34-81% of the pairs, while the activation side used all of
them. The side-by-side table in the paper is therefore descriptive.

This script removes that difference. For each model it takes the single-token
pairs of the unembedding side and computes, on exactly those pairs, with exactly
the same pool of null words and exactly the same random null draws:

    gap_vocab(c) = Gini(unembedding contrast) - mean Gini(lexical null, unembedding rows)
    gap_act(c)   = Gini(activation contrast)  - mean Gini(lexical null, activations)
    d(c)         = gap_vocab(c) - gap_act(c)

Positive gap = further toward the head than random word groups.

PRIMARY (pre-registered)
  A. d, averaged over concepts within a model, tested across models:
     does the unembedding side show more head-ward specificity than the
     activation side on the same pairs?
  B. gap_act on the same pairs, with TOST at +/-0.082: is the activation side
     still non-specific on the single-token subset?
  The paper's claim "specific in the vocabulary, not in the activations" needs
  both: A's 95% CI above 0 and B equivalent, with and without gemma-2-2b.

Activations are read from the word_specificity caches
(results/_spectral/word_specificity/<basis>_<layer>/acts_<model>.pt), so no GPU
is needed if they exist; otherwise they are re-extracted with the published
routine (GPU).

    python scripts/dual_paired.py run --model Qwen/Qwen2.5-0.5B
    python scripts/dual_paired.py summarize
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
    compute_cumulative_variance, one_sample_test, tost_equivalence,
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load("word_specificity")
ds = _load("dictionary_specificity")

SPECTRAL_DIR = REPO_ROOT / "results" / "_spectral"
OUT_ROOT = SPECTRAL_DIR / "dual_paired"
WS_ROOT = SPECTRAL_DIR / "word_specificity"
DS_ROOT = SPECTRAL_DIR / "dictionary_specificity"
SESOI = ws.PLACEMENT_SESOI
GEMMA = ws.GEMMA
MODELS = ws.MODELS
MIN_MODELS = 6


# ─── Pure computation (tested in tests/test_dual_paired.py) ──────────────────

def paired_rows(X_vocab, X_act, words, clean, eigenvectors, cumvar,
                n_null=1000, seed=0):
    """
    X_vocab, X_act  [len(words), d] rows in the SAME order as `words`.
    clean           concept -> [(base, target)], every word in `words`.
    Both sides use the same pool and, through the same seed, the same draws.
    """
    vocab = {w: i for i, w in enumerate(words)}
    rv = {r["concept"]: r for r in ws.analyse_concepts(
        X_vocab, vocab, clean, eigenvectors, cumvar, n_null=n_null, seed=seed)}
    ra = {r["concept"]: r for r in ws.analyse_concepts(
        X_act, vocab, clean, eigenvectors, cumvar, n_null=n_null, seed=seed)}
    rows = []
    for c in clean:
        v, a = rv[c], ra[c]
        rows.append({"concept": c, "group": v["group"], "n_pairs": v["n_pairs"],
                     "gini_vocab": v["gini"], "gini_act": a["gini"],
                     "gap_vocab": v["gap_lexical"], "gap_act": a["gap_lexical"],
                     "p_vocab": v["p_lexical"], "p_act": a["p_lexical"],
                     "d": v["gap_lexical"] - a["gap_lexical"]})
    return rows


def model_means(rows):
    return {k: float(np.mean([r[k] for r in rows]))
            for k in ("gap_vocab", "gap_act", "d", "gini_vocab", "gini_act")} | \
           {"n_concepts": len(rows)}


def verdict(A_all, A_nog, B_all, B_nog):
    """Pre-registered rule (prereg/dual_paired_prereg.json)."""
    for t in (A_all, A_nog, B_all, B_nog):
        if "test" not in t:
            return "not testable"
    a_pos = all(t["test"]["ci95"][0] > 0 for t in (A_all, A_nog))
    b_eq = all(t["tost"].get("equivalent") for t in (B_all, B_nog))
    if a_pos and b_eq:
        return "asymmetry confirmed"
    if not a_pos and all(t["tost"].get("equivalent") for t in (A_all, A_nog)):
        return "no asymmetry"
    return "inconclusive"


# ─── Model side ──────────────────────────────────────────────────────────────

def load_acts(model_id, model_name, layer, device, batch_size):
    """Activations for ALL word-pair words, from the word_specificity cache."""
    concepts = ws.load_all_pairs(ws.WORD_PAIRS)
    words_all = sorted({w for ps in concepts.values() for p in ps for w in p})
    for basis in ("language", "full_vocab"):          # activations do not depend on the basis
        cache = WS_ROOT / f"{basis}_{layer}" / f"acts_{model_name}.pt"
        if cache.exists():
            d = torch.load(cache, map_location="cpu", weights_only=False)
            if d["words"] == words_all and d["layer_spec"] == layer:
                return d["X"].float(), words_all, f"cache:{cache}"
    cache = WS_ROOT / f"language_{layer}" / f"acts_{model_name}.pt"
    X, _ = ws.extract_word_acts(words_all, model_id, layer, device, batch_size, cache)
    return X.float(), words_all, f"extracted:{cache}"


def cmd_run(args):
    model_name = args.model.split("/")[-1]
    out_dir = OUT_ROOT / f"{args.basis}_{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = ws.load_all_pairs(ws.WORD_PAIRS)
    W, tok, src_w = ds.load_unembedding(args.model, args.loader)
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    clean, word_to_id, _ = ds.clean_concepts(concepts, encode, "published")

    # integrity: the pairs must be those of the committed unembedding-side run
    ref = DS_ROOT / f"{args.basis}_published" / f"{model_name}.json"
    if ref.exists():
        r = json.load(open(ref))
        want = {c["concept"]: c["n_pairs"] for c in r["concepts"]}
        got = {c: len(p) for c, p in clean.items()}
        if want != got:
            raise SystemExit(f"clean pairs differ from {ref}; refusing to run")
        print(f"  pairs match the committed dictionary run ({len(got)} concepts)")

    X_act_all, words_all, src_a = load_acts(args.model, model_name, args.layer,
                                            args.device, args.batch_size)
    row_of = {w: i for i, w in enumerate(words_all)}
    words = sorted({w for ps in clean.values() for p in ps for w in p})
    X_act = X_act_all[torch.as_tensor([row_of[w] for w in words])]
    X_vocab = W[torch.as_tensor([word_to_id[w] for w in words])].float()

    evals, evecs, src_b = ws.load_eigen(model_name, args.basis)
    cumvar = compute_cumulative_variance(evals)
    if X_vocab.shape[1] != evecs.shape[0] or X_act.shape[1] != evecs.shape[0]:
        raise SystemExit("dimension mismatch between rows, activations and basis")

    rows = paired_rows(X_vocab, X_act, words, clean, evecs, cumvar,
                       n_null=args.n_null, seed=args.seed)
    m = model_means(rows)
    print(f"{model_name}: {m['n_concepts']} concepts, {len(words)} words  "
          f"gap_vocab {m['gap_vocab']:+.4f}  gap_act {m['gap_act']:+.4f}  d {m['d']:+.4f}")
    res = {"model": model_name, "model_id": args.model, "basis": args.basis,
           "layer": args.layer, "basis_source": src_b, "acts_source": src_a,
           "unembedding_source": src_w, "n_words": len(words), "n_null": args.n_null,
           "seed": args.seed, "means": m, "concepts": rows}
    with open(out_dir / f"{model_name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out_dir / (model_name + '.json')}")


# ─── Across models ───────────────────────────────────────────────────────────

def cross(per_model, drop_gemma):
    vals = {m: v for m, v in per_model.items() if not (drop_gemma and m == GEMMA)}
    out = {"models": list(vals), "n": len(vals)}
    if len(vals) >= 2:
        out["test"] = one_sample_test(list(vals.values()))
        out["tost"] = tost_equivalence(list(vals.values()), sesoi=SESOI)
    return out


def _f(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return "0.000" if round(x, 3) == 0 else f"{x:+.3f}"


def _p(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cmd_summarize(args):
    folder = OUT_ROOT / f"{args.basis}_{args.layer}"
    res = {m: json.load(open(folder / f"{m}.json"))
           for m in args.models if (folder / f"{m}.json").exists()}
    if len(res) < 2:
        raise SystemExit(f"need at least 2 models in {folder}")
    lines = [f"# Dual geometry on identical pairs ({args.basis} basis, {args.layer} layer)", "",
             "Same single-token pairs, same null pool, same null draws on both sides. "
             f"gap > 0 = further toward the head than random word groups. Bound ±{SESOI}.", "",
             "| model | concepts | gap vocabulary | gap activation | d = vocab − act |",
             "|---|---|---|---|---|"]
    for m, r in res.items():
        a = r["means"]
        lines.append(f"| {m} | {a['n_concepts']} | {_f(a['gap_vocab'])} | "
                     f"{_f(a['gap_act'])} | {_f(a['d'])} |")
    out = {"basis": args.basis, "layer": args.layer, "tests": {}}
    lines += ["", "| quantity | models | n | mean | 95% CI | p (t) | 90% CI | p TOST | equivalent |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, label in (("d", "**A: d (primary)**"), ("gap_act", "**B: activation gap (primary)**"),
                       ("gap_vocab", "vocabulary gap")):
        per = {m: r["means"][key] for m, r in res.items()}
        for drop in (False, True):
            cm = cross(per, drop)
            out["tests"][f"{key}|{'without_gemma' if drop else 'all'}"] = cm
            if "test" not in cm:
                continue
            t, q = cm["test"], cm["tost"]
            ci90 = q.get("ci90") or [None, None]
            lines.append(f"| {label} | {'without gemma' if drop else 'all'} | {cm['n']} | "
                         f"{_f(t['mean'])} | [{_f(t['ci95'][0])}, {_f(t['ci95'][1])}] | "
                         f"{_p(t.get('p_t'))} | [{_f(ci90[0])}, {_f(ci90[1])}] | "
                         f"{_p(q.get('p_tost'))} | {'yes' if q.get('equivalent') else 'no'} |")
    T = out["tests"]
    v = verdict(T["d|all"], T["d|without_gemma"], T["gap_act|all"], T["gap_act|without_gemma"])
    if len(res) < MIN_MODELS:
        v = "not testable"
    out["verdict"] = v
    if args.basis == "language" and args.layer == "final":
        lines += ["", f"**Primary verdict (frozen rule): {v.upper()}**"]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    with open(folder / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    rows = [{"model": m, **c} for m, r in res.items() for c in r["concepts"]]
    with open(folder / "per_concept.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    print(f"\nwrote {folder}/summary.md, summary.json, per_concept.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    r.add_argument("--layer", choices=("final", "mid"), default="final")
    r.add_argument("--loader", choices=("light", "full"), default="light")
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch-size", type=int, default=32)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize")
    s.add_argument("--models", nargs="+", default=MODELS)
    s.add_argument("--basis", choices=("language", "full_vocab"), default="language")
    s.add_argument("--layer", choices=("final", "mid"), default="final")
    s.set_defaults(func=cmd_summarize)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
