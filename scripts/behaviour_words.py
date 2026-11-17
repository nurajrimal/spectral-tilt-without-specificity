#!/usr/bin/env python3
"""
EXPLORATORY: do the words that differ between behavioural statement pairs
follow the dual geometry?

Behavioural concepts (truth, sycophancy) are properties of whole sentences, so
they have no unembedding-row ("dictionary") version. This script builds the
closest word-level stand-in. For every matched pair it removes stop words and
keeps the pair only if exactly ONE content word differs on each side, e.g.

    negative  "The city of Paris is in Japan."
    positive  "The city of Paris is in France."      ->  word pair  Japan -> France

Each dataset becomes one word-pair "concept" (negative word -> positive word).
For each model it then measures, with the same words and the same null draws:

    vocabulary side   unembedding rows of the words
    activation side   context-free activations of the words (final layer)

and the Gini gap against two nulls (positive gap = further toward the head):

    repartition (main)  the concept's own 2n words, split at random. Asks whether
                        the negative/positive assignment matters beyond WHICH
                        words are involved (e.g. beyond "these are countries").
    lexical             2n words from the pooled vocabulary of all datasets.

The published dual geometry would show up as gap_vocab > 0 and gap_act < 0.

For the truth datasets the expectation is NO structure on either side: whether a
country is "right" depends on the city, so the pairs share no consistent
relation. A null here is a negative control for the method, not a failure.

Not pre-registered. Outputs go to results/_spectral/behaviour_words/ and never
touch the results or caches of the pre-registered tests.

    python scripts/behaviour_words.py build                  # CPU, seconds: extract word pairs, print stats
    python scripts/behaviour_words.py build --statements-dir data/my_data --datasets my_set
                                                             # your own data: data/my_data/my_set.jsonl,
                                                             # one {"positive": ..., "negative": ...} per line
    python scripts/behaviour_words.py run --model Qwen/Qwen2.5-0.5B
    python scripts/behaviour_words.py summarize
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from linear_rep_geometry.spectral_metrics import (  # noqa: E402
    compute_cumulative_variance, empirical_pvalue, one_sample_test,
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load("word_specificity")
ds = _load("dictionary_specificity")

STATEMENTS = REPO_ROOT / "data" / "statements"
OUT = REPO_ROOT / "results" / "_spectral" / "behaviour_words"
PAIRS_DIR = OUT / "word_pairs"
DATASETS = ["truth_cities", "truth_cities_negated", "truth_conjunction",
            "truth_disjunction", "truth_counterfact", "truth_magnitude",
            "truth_translation", "sycophancy", "sycophancy_nlp",
            "sycophancy_philpapers", "sycophancy_political"]
MIN_PAIRS = 4
GEMMA = ws.GEMMA

STOP = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd she'll
she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's whom
why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves city""".split())
# "city" is in every truth_cities statement; listing it is harmless elsewhere.

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ─── Pure computation (tested in tests/test_behaviour_words.py) ──────────────

def content_words(text: str) -> collections.Counter:
    return collections.Counter(w for w in WORD.findall(text) if w.lower() not in STOP)


def substitution_pair(negative: str, positive: str):
    """(neg_word, pos_word) if exactly one content word differs on each side."""
    n, p = content_words(negative), content_words(positive)
    neg_only, pos_only = n - p, p - n
    if sum(neg_only.values()) == 1 and sum(pos_only.values()) == 1:
        (a,), (b,) = neg_only.keys(), pos_only.keys()
        return a, b
    return None


def extract(rows):
    """rows: iterable of dicts with 'positive' and 'negative'. Returns pairs, stats."""
    pairs, stats = [], collections.Counter()
    for r in rows:
        stats["statement_pairs"] += 1
        sp = substitution_pair(r["negative"], r["positive"])
        if sp is None:
            n, p = content_words(r["negative"]), content_words(r["positive"])
            k = sum((n - p).values()) + sum((p - n).values())
            stats["no_difference" if k == 0 else "several_words_differ"] += 1
            continue
        stats["single_word_substitution"] += 1
        pairs.append(sp)
    uniq = sorted(set(pairs))
    stats["unique_word_pairs"] = len(uniq)
    return uniq, stats


def gaps(X, vocab, concepts, eigenvectors, cumvar, n_null=1000, seed=0):
    """Real Gini and both nulls per concept. Lexical null is skipped when the
    pooled vocabulary is too small for it."""
    rng = np.random.default_rng(seed)
    all_rows = np.arange(X.shape[0])
    out = {}
    for name, pairs in concepts.items():
        n = len(pairs)
        b = np.array([vocab[w] for w, _ in pairs])
        t = np.array([vocab[w] for _, w in pairs])
        g_real = float(ws.batched_gini((X[t].mean(0) - X[b].mean(0)).unsqueeze(0),
                                       eigenvectors, cumvar)[0])
        own = np.concatenate([t, b])
        rep = ws.batched_gini(ws._split_means(
            X, np.stack([rng.permutation(own) for _ in range(n_null)]), n),
            eigenvectors, cumvar)
        row = {"concept": name, "n_pairs": n, "gini": g_real,
               "gap_repartition": float(g_real - rep.mean()),
               "p_repartition": float(empirical_pvalue(g_real, rep, tail="two"))}
        pool = np.setdiff1d(all_rows, np.unique(own))
        if len(pool) >= 2 * n:
            lex = ws.batched_gini(ws._split_means(
                X, np.stack([rng.choice(pool, 2 * n, replace=False)
                             for _ in range(n_null)]), n), eigenvectors, cumvar)
            row["gap_lexical"] = float(g_real - lex.mean())
            row["p_lexical"] = float(empirical_pvalue(g_real, lex, tail="two"))
        out[name] = row
    return out


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_build(args):
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    statements = Path(args.statements_dir)
    report = {}
    print(f"{'dataset':24s} {'pairs':>6s} {'1-word':>7s} {'several':>8s} {'none':>5s} {'unique':>7s}  examples")
    for d in (args.datasets or DATASETS):
        path = statements / f"{d}.jsonl"
        if not path.exists():
            print(f"{d:24s} missing {path}")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()][: args.max_pairs]
        pairs, st = extract(rows)
        report[d] = dict(st)
        ex = ", ".join(f"{a}->{b}" for a, b in pairs[:4])
        print(f"{d:24s} {st['statement_pairs']:6d} {st['single_word_substitution']:7d} "
              f"{st['several_words_differ']:8d} {st['no_difference']:5d} {len(pairs):7d}  {ex}")
        if len(pairs) >= MIN_PAIRS:
            with open(PAIRS_DIR / f"{d}.txt", "w") as f:
                f.writelines(f"{a}\t{b}\n" for a, b in pairs)
    with open(OUT / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    kept = sorted(p.stem for p in PAIRS_DIR.glob("*.txt"))
    print(f"\n{len(kept)} datasets with >= {MIN_PAIRS} unique word pairs: {', '.join(kept)}")
    print(f"wrote {PAIRS_DIR}/ and {OUT}/build_report.json")


def load_concepts():
    out = {}
    for p in sorted(PAIRS_DIR.glob("*.txt")):
        out[p.stem] = [tuple(l.rstrip("\n").split("\t")) for l in open(p) if l.strip()]
    if not out:
        raise SystemExit("no word pairs; run `build` first")
    return out


def cmd_run(args):
    name = args.model.split("/")[-1]
    out_dir = OUT / "language_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    concepts = load_concepts()

    W, tok, src_w = ds.load_unembedding(args.model, "light")
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    clean, word_to_id, report = ds.clean_concepts(concepts, encode, "published", MIN_PAIRS)
    if not clean:
        raise SystemExit(f"{name}: no dataset keeps {MIN_PAIRS} single-token pairs")
    words = sorted({w for ps in clean.values() for p in ps for w in p})
    vocab = {w: i for i, w in enumerate(words)}
    X_vocab = W[torch.as_tensor([word_to_id[w] for w in words])].float()
    X_act, li = ws.extract_word_acts(words, args.model, "final", args.device,
                                     args.batch_size, out_dir / f"acts_{name}.pt")
    X_act = X_act.float()

    evals, evecs, src_b = ws.load_eigen(name, "language")
    cumvar = compute_cumulative_variance(evals)
    gv = gaps(X_vocab, vocab, clean, evecs, cumvar, args.n_null, args.seed)
    ga = gaps(X_act, vocab, clean, evecs, cumvar, args.n_null, args.seed)
    rows = []
    for c in clean:
        r = {"concept": c, "n_pairs": gv[c]["n_pairs"]}
        for side, g in (("vocab", gv[c]), ("act", ga[c])):
            for k, v in g.items():
                if k not in ("concept", "n_pairs"):
                    r[f"{k}_{side}"] = v
        rows.append(r)
        print(f"  {c:24s} n={r['n_pairs']:4d}  rep: vocab {r['gap_repartition_vocab']:+.3f} "
              f"(p {r['p_repartition_vocab']:.3f})  act {r['gap_repartition_act']:+.3f} "
              f"(p {r['p_repartition_act']:.3f})")
    json.dump({"model": name, "model_id": args.model, "basis_source": src_b,
               "unembedding_source": src_w, "act_layer_index": li,
               "clean_report": report, "n_words": len(words), "n_null": args.n_null,
               "seed": args.seed, "concepts": rows},
              open(out_dir / f"{name}.json", "w"), indent=2)
    print(f"  wrote {out_dir / (name + '.json')}")


def _f(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (
        "0.000" if round(x, 3) == 0 else f"{x:+.3f}")


def cmd_summarize(args):
    folder = OUT / "language_final"
    res = {p.stem: json.load(open(p)) for p in sorted(folder.glob("*.json"))
           if p.stem not in ("summary",)}
    if not res:
        raise SystemExit(f"no results in {folder}")
    by = collections.defaultdict(dict)
    for m, r in res.items():
        for c in r["concepts"]:
            by[c["concept"]][m] = c
    lines = ["# EXPLORATORY: dual geometry of the words that differ in behavioural pairs", "",
             "Not pre-registered. Positive gap = further toward the head than the null. "
             "Published dual geometry would be vocab > 0 and act < 0.", "",
             "| dataset | models | gap vocab (rep) | 95% CI | gap act (rep) | 95% CI | "
             "gap vocab (lex) | gap act (lex) |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for c, per in by.items():
        for drop in (False, True):
            ms = [m for m in per if not (drop and m == GEMMA)]
            if len(ms) < 2:
                continue
            vals = {k: [per[m].get(k) for m in ms] for k in
                    ("gap_repartition_vocab", "gap_repartition_act",
                     "gap_lexical_vocab", "gap_lexical_act")}
            tv = one_sample_test(vals["gap_repartition_vocab"])
            ta = one_sample_test(vals["gap_repartition_act"])
            lv = [x for x in vals["gap_lexical_vocab"] if x is not None]
            la = [x for x in vals["gap_lexical_act"] if x is not None]
            out[f"{c}|{'without_gemma' if drop else 'all'}"] = {
                "n": len(ms), "rep_vocab": tv, "rep_act": ta}
            lines.append(
                f"| {c}{' (w/o gemma)' if drop else ''} | {len(ms)} | {_f(tv['mean'])} | "
                f"[{_f(tv['ci95'][0])}, {_f(tv['ci95'][1])}] | {_f(ta['mean'])} | "
                f"[{_f(ta['ci95'][0])}, {_f(ta['ci95'][1])}] | "
                f"{_f(float(np.mean(lv)) if lv else None)} | {_f(float(np.mean(la)) if la else None)} |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(folder / "summary.json", "w"), indent=2, default=float)
    print("\n".join(lines))
    print(f"\nwrote {folder}/summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--max-pairs", type=int, default=512)
    b.add_argument("--datasets", nargs="+", default=None,
                   help="names of <name>.jsonl files (default: the 11 thesis datasets)")
    b.add_argument("--statements-dir", default=str(STATEMENTS),
                   help="folder holding the .jsonl files (default: data/statements)")
    b.set_defaults(func=cmd_build)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch-size", type=int, default=32)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize")
    s.set_defaults(func=cmd_summarize)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
