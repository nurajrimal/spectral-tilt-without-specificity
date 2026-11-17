#!/usr/bin/env python3
"""
Does a BEHAVIOURAL concept (truth) have a dual geometry, and is its activation
direction linked to the words the model uses to report it?

Truth is a property of a whole statement, so it has no unembedding row of its
own. Its natural vocabulary-side counterpart is the set of words a model uses to
REPORT truth (false -> true, incorrect -> correct, no -> yes, ...). This script
puts the two side by side for the seven truth datasets of the thesis.

  activation side   v = mean(h_true - h_false), last token, over statement pairs.
                    Two variants:
                      plain     the statements as in the thesis
                      prompted  statement + " This statement is", so the model's
                                next token is a truth report and the read-out
                                through the unembedding is meaningful
  vocabulary side   u = mean(gamma(true word) - gamma(false word)) over the
                    report-word pairs (single tokens only, published token rule)

Two questions, positive gap = further toward the head than the null:

  DUAL GEOMETRY  G_vocab = Gini(u) - mean Gini(lexical null: random word groups
                 of the same size from the word-pair vocabulary)
                 G_act   = Gini(v) - mean Gini(sign-flip null: v with each
                 pair's difference multiplied by a random sign; Appendix A)
                 The published dual geometry is G_vocab > 0 and G_act < 0.

  LINK           L = cos(v, u) - mean cos(v, u_null) over the same lexical-null
                 word contrasts: does the truth direction point toward the truth
                 words more than toward arbitrary word contrasts? The sign-flip
                 null for L is reported alongside.

Statements are read from --statements-dir (default data/statements). Activations
are extracted with the thesis routine (layerwise_activations, last token,
max_length 256, final norm on the last layer) and cached in
results/_spectral/behaviour_link/acts/.

    python scripts/behaviour_link.py run --model Qwen/Qwen2.5-0.5B --statements-dir ~/spectral_anti_concentration/data/statements
    python scripts/behaviour_link.py summarize
"""

from __future__ import annotations

import argparse
import gc
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
    compute_cumulative_variance, empirical_pvalue, one_sample_test, tost_equivalence,
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load("word_specificity")
ds = _load("dictionary_specificity")

OUT = REPO_ROOT / "results" / "_spectral" / "behaviour_link"
STATEMENTS = REPO_ROOT / "data" / "statements"
DATASETS = ["truth_cities", "truth_cities_negated", "truth_conjunction",
            "truth_disjunction", "truth_counterfact", "truth_magnitude",
            "truth_translation"]
VARIANTS = {"plain": "", "prompted": " This statement is"}
MAX_PAIRS = 512
SESOI = ws.PLACEMENT_SESOI
GEMMA = ws.GEMMA
MODELS = ws.MODELS
MIN_REPORT_PAIRS = 4
MIN_MODELS = 6

# (false pole, true pole). Frozen in prereg/behaviour_link_prereg.json.
REPORT_WORDS = [
    ("false", "true"), ("False", "True"), ("incorrect", "correct"),
    ("Incorrect", "Correct"), ("wrong", "right"), ("no", "yes"), ("No", "Yes"),
    ("inaccurate", "accurate"), ("fake", "real"), ("invalid", "valid"),
    ("lie", "truth"), ("fiction", "fact"), ("mistaken", "accurate"),
    ("untrue", "true"),
]


# ─── Pure computation (tested in tests/test_behaviour_link.py) ───────────────

def _cos(A: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """cosine of every row of A with b."""
    A = A.float()
    b = b.float()
    return (A @ b / (A.norm(dim=1) * b.norm()).clamp(min=1e-12)).numpy()


def sign_flip_directions(D: torch.Tensor, n_null: int, rng) -> torch.Tensor:
    """mean over pairs of (random sign) x (pair difference), one row per draw."""
    S = torch.as_tensor(rng.choice([-1.0, 1.0], size=(n_null, D.shape[0])),
                        dtype=torch.float32)
    return S @ D.float() / D.shape[0]


def lexical_null_directions(Xpool: torch.Tensor, n: int, n_null: int, rng) -> torch.Tensor:
    """difference of means of two random groups of n words, one row per draw."""
    idx = np.stack([rng.choice(Xpool.shape[0], 2 * n, replace=False) for _ in range(n_null)])
    return ws._split_means(Xpool.float(), idx, n)


def vocab_side(u, U_null, eigenvectors, cumvar):
    g = float(ws.batched_gini(u.unsqueeze(0), eigenvectors, cumvar)[0])
    gn = ws.batched_gini(U_null, eigenvectors, cumvar)
    return {"gini_vocab": g, "gap_vocab": float(g - gn.mean()),
            "p_vocab": float(empirical_pvalue(g, gn, tail="two"))}


def act_side(D, u, U_null, eigenvectors, cumvar, n_null, rng):
    """Placement of v against the sign-flip null, and the link with u."""
    v = D.float().mean(0)
    Vb = sign_flip_directions(D, n_null, rng)
    g = float(ws.batched_gini(v.unsqueeze(0), eigenvectors, cumvar)[0])
    gb = ws.batched_gini(Vb, eigenvectors, cumvar)
    L = float(_cos(v.unsqueeze(0), u)[0])
    Lw = _cos(U_null, v)
    La = _cos(Vb, u)
    return {"gini_act": g, "gap_act": float(g - gb.mean()),
            "p_act": float(empirical_pvalue(g, gb, tail="two")),
            "cos": L,
            "gap_link_word": float(L - Lw.mean()),
            "p_link_word": float(empirical_pvalue(L, Lw, tail="two")),
            "gap_link_signflip": float(L - La.mean()),
            "p_link_signflip": float(empirical_pvalue(L, La, tail="two"))}


def verdict_dual(gv_all, gv_nog, ga_all, ga_nog):
    if not all("test" in t for t in (gv_all, gv_nog, ga_all, ga_nog)):
        return "not testable"
    if all(t["test"]["ci95"][0] > 0 for t in (gv_all, gv_nog)) and \
       all(t["test"]["ci95"][1] < 0 for t in (ga_all, ga_nog)):
        return "dual"
    if all(t["tost"].get("equivalent") for t in (ga_all, ga_nog)) or \
       all(t["tost"].get("equivalent") for t in (gv_all, gv_nog)):
        return "not dual"
    return "inconclusive"


def verdict_link(l_all, l_nog):
    if not all("test" in t for t in (l_all, l_nog)):
        return "not testable"
    if all(t["test"]["ci95"][0] > 0 for t in (l_all, l_nog)):
        return "linked"
    if all(t["test"]["ci95"][1] < 0 for t in (l_all, l_nog)):
        return "reversed"
    return "no evidence"


# ─── Model side ──────────────────────────────────────────────────────────────

def extract_all(model_id, name, statements, args):
    """Last-token activations for every dataset and variant; cached."""
    import linear_rep_geometry as lrg
    from linear_rep_geometry.behavioral import layerwise_activations, load_pairs

    acts_dir = OUT / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    todo, out = [], {}
    for d in DATASETS:
        for var in VARIANTS:
            p = acts_dir / f"acts_{name}__{d}__{var}.pt"
            if p.exists():
                out[(d, var)] = torch.load(p, map_location="cpu", weights_only=False)
            else:
                todo.append((d, var, p))
    if todo:
        model, tok, handler = lrg.load_model(model_id)
        model.eval()
        device = next(model.parameters()).device
        for d, var, p in todo:
            path = statements / f"{d}.jsonl"
            if not path.exists():
                raise SystemExit(f"missing {path}; pass --statements-dir")
            pairs = load_pairs(path)[:MAX_PAIRS]
            texts = [x.positive + VARIANTS[var] for x in pairs] + \
                    [x.negative + VARIANTS[var] for x in pairs]
            with torch.inference_mode():
                A = layerwise_activations(texts, model, tok, handler, device,
                                          batch_size=args.batch_size,
                                          max_length=256)["last_token"]
            nL = A.shape[1]
            li = {"final": nL - 1, "mid": nL // 2}
            rec = {"n_pairs": len(pairs), "layers": li,
                   "acts": {k: A[:, i, :].float().clone() for k, i in li.items()}}
            torch.save(rec, p)
            out[(d, var)] = rec
            print(f"    {d:22s} {var:8s} {len(pairs)} pairs  layers {li}")
            del A
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def cmd_run(args):
    name = args.model.split("/")[-1]
    res_dir = OUT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    acts = extract_all(args.model, name, Path(args.statements_dir), args)

    W, tok, src_w = ds.load_unembedding(args.model, "light")
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    clean, w2i, rep = ds.clean_concepts({"truth_report": REPORT_WORDS}, encode,
                                        "published", MIN_REPORT_PAIRS)
    if "truth_report" not in clean:
        raise SystemExit(f"{name}: fewer than {MIN_REPORT_PAIRS} single-token report pairs")
    pairs = clean["truth_report"]
    n = len(pairs)
    u = W[[w2i[t] for _, t in pairs]].float().mean(0) - W[[w2i[f] for f, _ in pairs]].float().mean(0)

    # null pool: single-token words of the 42 word-pair concepts, minus report words
    report_words = {w for p in REPORT_WORDS for w in p}
    pool_words = sorted({w for ps in ws.load_all_pairs(ws.WORD_PAIRS).values()
                         for p in ps for w in p} - report_words)
    pool_ids = sorted({i for w in pool_words
                       if (i := ds.word_token_id(encode, w, "published")) is not None}
                      - {w2i[w] for p in pairs for w in p})
    Xpool = W[torch.as_tensor(pool_ids)].float()

    evals, evecs, src_b = ws.load_eigen(name, "language")
    cumvar = compute_cumulative_variance(evals)
    rng = np.random.default_rng(args.seed)
    U_null = lexical_null_directions(Xpool, n, args.n_null, rng)
    vside = vocab_side(u, U_null, evecs, cumvar)
    print(f"{name}: {n} report pairs, pool {len(pool_ids)} tokens  "
          f"vocab gap {vside['gap_vocab']:+.4f} (p {vside['p_vocab']:.3f})")

    rows = []
    for (d, var), rec in sorted(acts.items()):
        k = rec["n_pairs"]
        for layer, A in rec["acts"].items():
            D = A[:k] - A[k:]
            r = act_side(D, u, U_null, evecs, cumvar, args.n_null,
                         np.random.default_rng(args.seed + 1))
            rows.append({"dataset": d, "variant": var, "layer": layer, "n_pairs": k, **r})
            if layer == "final":
                print(f"  {d:22s} {var:8s} act gap {r['gap_act']:+.4f} (p {r['p_act']:.3f})  "
                      f"link {r['gap_link_word']:+.4f} (p {r['p_link_word']:.3f})")

    means = {}
    for var in VARIANTS:
        for layer in ("final", "mid"):
            sel = [r for r in rows if r["variant"] == var and r["layer"] == layer]
            means[f"{var}|{layer}"] = {k: float(np.mean([r[k] for r in sel]))
                                       for k in ("gap_act", "gap_link_word",
                                                 "gap_link_signflip", "cos")}
    json.dump({"model": name, "model_id": args.model, "basis_source": src_b,
               "unembedding_source": src_w, "report_pairs": pairs, "report_clean": rep,
               "n_pool": len(pool_ids), "n_null": args.n_null, "seed": args.seed,
               "vocab": vside, "means": means, "rows": rows},
              open(res_dir / f"{name}.json", "w"), indent=2)
    print(f"  wrote {res_dir / (name + '.json')}")


# ─── Across models ───────────────────────────────────────────────────────────

def cross(per, drop):
    vals = {m: v for m, v in per.items() if not (drop and m == GEMMA)}
    out = {"n": len(vals)}
    if len(vals) >= 2:
        out["test"] = one_sample_test(list(vals.values()))
        out["tost"] = tost_equivalence(list(vals.values()), sesoi=SESOI)
    return out


def _f(x):
    return "–" if x is None else ("0.000" if round(x, 3) == 0 else f"{x:+.3f}")


def _p(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cmd_summarize(args):
    res_dir = OUT / "results"
    res = {p.stem: json.load(open(p)) for p in sorted(res_dir.glob("*.json"))}
    if len(res) < 2:
        raise SystemExit(f"need at least 2 models in {res_dir}")
    lines = ["# Truth: dual geometry and link with report words", "",
             "Positive gap = further toward the head than the null. Published dual "
             "geometry: vocab gap > 0 and activation gap < 0. Link > 0: the truth "
             "direction points toward the truth words more than toward random word contrasts.", "",
             "| quantity | variant, layer | models | n | mean | 95% CI | p | 90% CI | p TOST |",
             "|---|---|---|---|---|---|---|---|---|"]
    out = {"tests": {}}

    def add(key, label, per):
        for drop in (False, True):
            cm = cross(per, drop)
            out["tests"][f"{key}|{'without_gemma' if drop else 'all'}"] = cm
            if "test" not in cm:
                continue
            t, q = cm["test"], cm["tost"]
            ci90 = q.get("ci90") or [None, None]
            lines.append(f"| {label} | {'w/o gemma' if drop else 'all'} | {cm['n']} | "
                         f"{_f(t['mean'])} | [{_f(t['ci95'][0])}, {_f(t['ci95'][1])}] | "
                         f"{_p(t.get('p_t'))} | [{_f(ci90[0])}, {_f(ci90[1])}] | {_p(q.get('p_tost'))} |")

    add("vocab", "vocabulary gap | –", {m: r["vocab"]["gap_vocab"] for m, r in res.items()})
    for var in VARIANTS:
        for layer in ("final", "mid"):
            k = f"{var}|{layer}"
            add(f"act|{k}", f"activation gap | {var}, {layer}",
                {m: r["means"][k]["gap_act"] for m, r in res.items()})
            add(f"link|{k}", f"link (word null) | {var}, {layer}",
                {m: r["means"][k]["gap_link_word"] for m, r in res.items()})
    T = out["tests"]
    vd = verdict_dual(T["vocab|all"], T["vocab|without_gemma"],
                      T["act|plain|final|all"], T["act|plain|final|without_gemma"])
    vl = verdict_link(T["link|prompted|final|all"], T["link|prompted|final|without_gemma"])
    if len(res) < MIN_MODELS:
        vd = vl = "not testable"
    out["verdict_dual"], out["verdict_link"] = vd, vl
    lines += ["", f"**Primary 1, dual geometry (plain statements, final layer): {vd.upper()}**",
              f"**Primary 2, link (prompted statements, final layer): {vl.upper()}**"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(OUT / "summary.json", "w"), indent=2, default=float)
    print("\n".join(lines))
    print(f"\nwrote {OUT}/summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--statements-dir", default=str(STATEMENTS))
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=8)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize")
    s.set_defaults(func=cmd_summarize)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
