#!/usr/bin/env python3
"""
Refusal: dual geometry and link with refusal words.

The same design as scripts/behaviour_link.py (truth), applied to refusal, the
one other behaviour with words of its own. Pure functions are imported from
behaviour_link.py unchanged; only the data, the words, the prompt and the
activation-side null differ.

  activation side  v = mean(h_harmful) - mean(h_harmless) at the last token, per
                   dataset (the four refusal sets of the thesis). These sets are
                   UNMATCHED: a harmful and a harmless instruction on one line come
                   from unrelated corpora. So the null is the thesis null for
                   unmatched data, label REPARTITION: the 2n instructions shuffled
                   into two random groups of n (not the sign flip used for truth).
                   Variants:
                     plain     the instruction alone
                     prompted  "Instruction: {x}\\nResponse:", so the next token
                               begins the model's reply
  vocabulary side  u = mean(gamma(refusal word)) - mean(gamma(compliance word))
                   over REPORT_WORDS (single tokens, published rule), against the
                   lexical null of random word contrasts.

  DUAL  gap_vocab > 0 (head) and gap_act < 0 (tail), as published for word pairs
  LINK  cos(v, u) above its value for random word contrasts

    python scripts/refusal_link.py run --model Qwen/Qwen2.5-0.5B --statements-dir ~/spectral_anti_concentration/data/statements
    python scripts/refusal_link.py summarize
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
    compute_cumulative_variance, empirical_pvalue,
)

_spec = importlib.util.spec_from_file_location("behaviour_link", SCRIPT_DIR / "behaviour_link.py")
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)
ws, ds = bl.ws, bl.ds

OUT = REPO_ROOT / "results" / "_spectral" / "refusal_link"
STATEMENTS = REPO_ROOT / "data" / "statements"
DATASETS = ["refusal", "refusal_advbench", "refusal_harmbench", "refusal_jailbreakbench"]
VARIANTS = {"plain": "{x}", "prompted": "Instruction: {x}\nResponse:"}
MAX_PAIRS = 512
MIN_REPORT_PAIRS = 4
MIN_MODELS = 6

# (compliance pole, refusal pole). Frozen in prereg/refusal_link_prereg.json.
REPORT_WORDS = [
    ("Sure", "Sorry"), ("sure", "sorry"), ("Certainly", "Unfortunately"),
    ("certainly", "unfortunately"), ("can", "cannot"), ("Yes", "No"),
    ("yes", "no"), ("happy", "unable"), ("accept", "refuse"),
    ("allow", "decline"), ("agree", "reject"), ("glad", "afraid"),
    ("Absolutely", "Apologies"),
]


# ─── Pure computation (tested in tests/test_refusal_link.py) ─────────────────

def repartition_directions(A: torch.Tensor, n: int, n_null: int, rng) -> torch.Tensor:
    """A: [2n, d], first n harmful, last n harmless. Random relabelling."""
    idx = np.stack([rng.permutation(A.shape[0]) for _ in range(n_null)])
    return ws._split_means(A.float(), idx, n)


def act_side_unmatched(A, n, u, U_null, eigenvectors, cumvar, n_null, rng):
    """Placement of v against the repartition null, and the link with u."""
    v = A[:n].float().mean(0) - A[n:].float().mean(0)
    Vb = repartition_directions(A, n, n_null, rng)
    g = float(ws.batched_gini(v.unsqueeze(0), eigenvectors, cumvar)[0])
    gb = ws.batched_gini(Vb, eigenvectors, cumvar)
    L = float(bl._cos(v.unsqueeze(0), u)[0])
    Lw = bl._cos(U_null, v)
    La = bl._cos(Vb, u)
    return {"gini_act": g, "gap_act": float(g - gb.mean()),
            "p_act": float(empirical_pvalue(g, gb, tail="two")),
            "cos": L,
            "gap_link_word": float(L - Lw.mean()),
            "p_link_word": float(empirical_pvalue(L, Lw, tail="two")),
            "gap_link_repartition": float(L - La.mean()),
            "p_link_repartition": float(empirical_pvalue(L, La, tail="two"))}


# ─── Model side ──────────────────────────────────────────────────────────────

def extract_all(model_id, name, statements, args):
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
        model, tok, handler = lrg.load_model(
            model_id, device_map=bl.capped_device_map(model_id, args.gpu_gib))
        model.eval()
        device = next(model.parameters()).device
        for d, var, p in todo:
            path = statements / f"{d}.jsonl"
            if not path.exists():
                raise SystemExit(f"missing {path}; pass --statements-dir")
            pairs = load_pairs(path)[:MAX_PAIRS]
            fmt = VARIANTS[var]
            texts = [fmt.format(x=x.positive) for x in pairs] + \
                    [fmt.format(x=x.negative) for x in pairs]
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
            print(f"    {d:24s} {var:8s} {len(pairs)} pairs  layers {li}", flush=True)
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
    out_rows = {}
    for rule in ("published", "space_first"):
        clean, w2i, rep = ds.clean_concepts({"refusal_report": REPORT_WORDS}, encode,
                                            rule, MIN_REPORT_PAIRS)
        if "refusal_report" not in clean:
            print(f"{name}: fewer than {MIN_REPORT_PAIRS} single-token report pairs ({rule})")
            continue
        pairs = clean["refusal_report"]
        n = len(pairs)
        u = W[[w2i[r] for _, r in pairs]].float().mean(0) - \
            W[[w2i[c] for c, _ in pairs]].float().mean(0)
        report_words = {w for p in REPORT_WORDS for w in p}
        pool_words = sorted({w for ps in ws.load_all_pairs(ws.WORD_PAIRS).values()
                             for p in ps for w in p} - report_words)
        pool_ids = sorted({i for w in pool_words
                           if (i := ds.word_token_id(encode, w, rule)) is not None}
                          - {w2i[w] for p in pairs for w in p})
        Xpool = W[torch.as_tensor(pool_ids)].float()

        evals, evecs, src_b = ws.load_eigen(name, "language")
        cumvar = compute_cumulative_variance(evals)
        rng = np.random.default_rng(args.seed)
        U_null = bl.lexical_null_directions(Xpool, n, args.n_null, rng)
        vside = bl.vocab_side(u, U_null, evecs, cumvar)
        print(f"{name} [{rule}]: {n} report pairs, pool {len(pool_ids)}  "
              f"vocab gap {vside['gap_vocab']:+.4f} (p {vside['p_vocab']:.3f})")
        rows = []
        for (d, var), rec in sorted(acts.items()):
            k = rec["n_pairs"]
            for layer, A in rec["acts"].items():
                r = act_side_unmatched(A, k, u, U_null, evecs, cumvar, args.n_null,
                                       np.random.default_rng(args.seed + 1))
                rows.append({"dataset": d, "variant": var, "layer": layer,
                             "n_pairs": k, **r})
                if layer == "final" and rule == "published":
                    print(f"  {d:24s} {var:8s} act gap {r['gap_act']:+.4f} (p {r['p_act']:.3f})  "
                          f"link {r['gap_link_word']:+.4f} (p {r['p_link_word']:.3f})")
        means = {}
        for var in VARIANTS:
            for layer in ("final", "mid"):
                sel = [r for r in rows if r["variant"] == var and r["layer"] == layer]
                means[f"{var}|{layer}"] = {k: float(np.mean([r[k] for r in sel]))
                                           for k in ("gap_act", "gap_link_word",
                                                     "gap_link_repartition", "cos")}
        out_rows[rule] = {"report_pairs": pairs, "report_clean": rep,
                          "n_pool": len(pool_ids), "basis_source": src_b,
                          "vocab": vside, "means": means, "rows": rows}
    json.dump({"model": name, "model_id": args.model, "unembedding_source": src_w,
               "n_null": args.n_null, "seed": args.seed, "batch_size": args.batch_size,
               "rules": out_rows},
              open(res_dir / f"{name}.json", "w"), indent=2)
    print(f"  wrote {res_dir / (name + '.json')}")


def cmd_summarize(args):
    res = {p.stem: json.load(open(p)) for p in sorted((OUT / "results").glob("*.json"))}
    if len(res) < 2:
        raise SystemExit("need at least 2 models")
    out = {"tests": {}}
    lines = ["# Refusal: dual geometry and link with refusal words", "",
             "Positive gap = further toward the head than the null. Published dual geometry: "
             "vocab gap > 0 and activation gap < 0. Link > 0: the refusal direction points toward "
             "the refusal words more than toward random word contrasts.", ""]
    for rule in ("published", "space_first"):
        rr = {m: r["rules"][rule] for m, r in res.items() if rule in r["rules"]}
        if len(rr) < 2:
            continue
        lines += [f"## Token rule: {rule}{' (primary)' if rule == 'published' else ''}", "",
                  "| quantity | variant, layer | models | n | mean | 95% CI | p | 90% CI | p TOST |",
                  "|---|---|---|---|---|---|---|---|---|"]

        def add(key, label, per):
            for drop in (False, True):
                cm = bl.cross(per, drop)
                out["tests"][f"{rule}|{key}|{'without_gemma' if drop else 'all'}"] = cm
                if "test" not in cm:
                    continue
                t, q = cm["test"], cm["tost"]
                ci90 = q.get("ci90") or [None, None]
                lines.append(f"| {label} | {'w/o gemma' if drop else 'all'} | {cm['n']} | "
                             f"{bl._f(t['mean'])} | [{bl._f(t['ci95'][0])}, {bl._f(t['ci95'][1])}] | "
                             f"{bl._p(t.get('p_t'))} | [{bl._f(ci90[0])}, {bl._f(ci90[1])}] | "
                             f"{bl._p(q.get('p_tost'))} |")

        add("vocab", "vocabulary gap | –", {m: r["vocab"]["gap_vocab"] for m, r in rr.items()})
        for var in VARIANTS:
            for layer in ("final", "mid"):
                k = f"{var}|{layer}"
                add(f"act|{k}", f"activation gap | {var}, {layer}",
                    {m: r["means"][k]["gap_act"] for m, r in rr.items()})
                add(f"link|{k}", f"link (word null) | {var}, {layer}",
                    {m: r["means"][k]["gap_link_word"] for m, r in rr.items()})
        lines.append("")
    T = out["tests"]
    p = "published"
    try:
        vd = bl.verdict_dual(T[f"{p}|vocab|all"], T[f"{p}|vocab|without_gemma"],
                             T[f"{p}|act|plain|final|all"], T[f"{p}|act|plain|final|without_gemma"])
        vl = bl.verdict_link(T[f"{p}|link|prompted|final|all"],
                             T[f"{p}|link|prompted|final|without_gemma"])
    except KeyError:
        vd = vl = "not testable"
    n_models = sum(1 for r in res.values() if "published" in r["rules"])
    if n_models < MIN_MODELS:
        vd = vl = "not testable"
    out["verdict_dual"], out["verdict_link"], out["n_models"] = vd, vl, n_models
    lines += [f"**Primary 1, dual geometry (plain, final layer, published rule): {vd.upper()}**",
              f"**Primary 2, link (prompted, final layer, published rule): {vl.upper()}**"]
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
    r.add_argument("--gpu-gib", type=float, default=3.0)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize")
    s.set_defaults(func=cmd_summarize)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
