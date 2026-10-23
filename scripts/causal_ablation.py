#!/usr/bin/env python3
"""
Causal test: does the model USE the quiet tail for facts, specifically?

Every result in the thesis is about decodability: what a probe can read from a
subspace. That leaves open whether the model itself relies on that subspace.
This script asks the interventional version of the same specificity question.

INTERVENTION

At one layer, during the forward pass, every token's residual-stream vector h is
projected off a k-dimensional subspace S of the published spectral basis:

        h  <-  h - U_S U_S^T h            (U_S orthonormal, d x k)

with S one of
    head     the k highest-variance eigenvectors   (the "loud" end)
    tail     the k lowest-variance eigenvectors    (the "quiet" end)
    random   k eigenvectors drawn at random, 3 seeds (reference only)

The hook sits on the output of decoder layer i-1, which IS hidden state i, the
vector the probes read (checked: hidden_states[i] equals that output exactly).
"mid" is i = floor(n_hidden_states / 2), as everywhere in the thesis.

TWO BEHAVIOURS, ONE SCALE

  truth   For each truth_cities pair, "The city of Krasnodar is in" is completed
          with the correct country (" Russia") and the wrong one (" South
          Africa"); the model is right when it gives the correct completion the
          higher log-probability. Accuracy is pairwise, chance 0.5. This is the
          model's own use of the fact, with no probe and no True/False prompt.

  general Top-1 next-token accuracy on data/heldout_ppl_passages.txt: ordinary
          language modelling, with nothing to do with truth.

Each is put on the thesis's retention scale, the share of above-chance
performance that survives the ablation:

        R_truth(S) = (acc_truth(S) - 0.5) / (acc_truth(none) - 0.5)
        R_gen(S)   =  acc_gen(S)          /  acc_gen(none)

SPECIFICITY, THE CAUSAL Delta

        reliance(behaviour) = R(remove head) - R(remove tail)
            > 0: removing the tail hurts that behaviour more than removing the head

        Delta = reliance(truth) - reliance(general)

  Delta > 0   the model relies on the quiet tail MORE for truth than for ordinary
              language modelling: "concepts whisper" holds as a claim about use
  Delta ~ 0   whatever the tail does, it does for everything, not for facts

95% CI per model by bootstrap: pairs for truth, whole passages for general
(tokens inside a passage are not independent), denominators held at their
point estimates as in the probe analysis. Across models: t-test, Wilcoxon,
TOST +/-0.10, with and without gemma-2-2b.

GATES (lessons from the continent control, where the control was unreadable)

  competence    baseline truth accuracy >= 0.60 (headroom 0.10, the thesis rule)
  sensitivity   at that k, removing the head or the tail must cost at least 0.10
                retention on at least one behaviour; otherwise the ablation did
                nothing and Delta is 0 by construction, not by finding

    python scripts/causal_ablation.py run --models Qwen/Qwen2.5-0.5B ...
    python scripts/causal_ablation.py summarize

Freeze prereg/causal_ablation_prereg.json BEFORE the first real run.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
OUT_DEFAULT = RESULTS_DIR / "_spectral" / "causal_ablation"
STATEMENTS = REPO_ROOT / "data" / "statements"
PASSAGES = REPO_ROOT / "data" / "heldout_ppl_passages.txt"
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-4B",
          "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.1-8B", "google/gemma-2-2b"]
GEMMA = "gemma-2-2b"
PRIMARY = {"layer": "mid", "k_frac": 0.10}
K_FRACS = (0.02, 0.05, 0.10, 0.20)
N_RANDOM = 3
SESOI = 0.10
MIN_COMPETENCE = 0.60
MIN_EFFECT = 0.10
MIN_MODELS = 6

_NAMED = re.compile(r" is (?:not )?in (.+?)\.?$")


# ─── Data ────────────────────────────────────────────────────────────────────

def truth_items(path, max_pairs=512):
    """(prompt, correct continuation, wrong continuation) per truth_cities pair."""
    from linear_rep_geometry.behavioral import load_pairs
    out = []
    for p in load_pairs(path)[:max_pairs]:
        city = p.meta.get("subject")
        correct = p.meta.get("topic")
        m = _NAMED.search(p.negative.strip())
        if not (city and correct and m):
            raise ValueError(f"cannot build a truth item from {p.positive!r}")
        out.append((f"The city of {city} is in", f" {correct}", f" {m.group(1)}"))
    return out


def passages(path=PASSAGES, min_chars=200):
    """Blank-line separated passages; single lines are joined."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", text)]
    blocks = [b for b in blocks if len(b) >= min_chars]
    if len(blocks) < 5:      # file without blank lines: chunk by lines instead
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        blocks = [" ".join(lines[i:i + 6]) for i in range(0, len(lines), 6)]
    return blocks


# ─── Intervention ────────────────────────────────────────────────────────────

class ProjectOut:
    """Forward hook: h <- h - U U^T h on a decoder layer's output (tensor or tuple)."""

    def __init__(self, U: torch.Tensor | None):
        self.U = U
        self.handle = None

    def __call__(self, module, inputs, output):
        if self.U is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        U = self.U.to(h.device)
        hf = h.float()
        hf = hf - (hf @ U) @ U.T
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def attach(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def decoder_layers(model):
    for path in (("model", "layers"), ("model", "decoder", "layers"),
                 ("transformer", "h"), ("gpt_neox", "layers")):
        obj = model
        try:
            for a in path:
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot find the decoder layers of this model")


def subspaces(evecs: torch.Tensor, k: int, n_random=N_RANDOM, seed=0):
    d = evecs.shape[0]
    rng = np.random.default_rng(seed)
    out = {"head": evecs[:, :k].float().contiguous(),
           "tail": evecs[:, d - k:].float().contiguous()}
    for r in range(n_random):
        idx = np.sort(rng.choice(d, size=k, replace=False))
        out[f"random{r}"] = evecs[:, idx].float().contiguous()
    return out


# ─── Measurement ─────────────────────────────────────────────────────────────

def _device(model):
    return next(model.parameters()).device


def encode_items(tok, items):
    """
    Token ids for prompt+continuation, and the index where the continuation
    starts. Encoded as one string and checked against the prompt's own encoding,
    so tokenizers that merge across the boundary are caught, not silently scored
    on the wrong tokens.
    """
    seqs, starts, mismatches = [], [], 0
    for prompt, *conts in items:
        p_ids = tok(prompt).input_ids
        for c in conts:
            full = tok(prompt + c).input_ids
            if full[:len(p_ids)] != p_ids or len(full) <= len(p_ids):
                mismatches += 1
                c_ids = tok(c, add_special_tokens=False).input_ids
                full = p_ids + c_ids
            seqs.append(full)
            starts.append(len(p_ids))
    return seqs, starts, mismatches


@torch.inference_mode()
def continuation_logprobs(model, tok, seqs, starts, batch_size=32):
    """Sum of log p(continuation tokens | prefix) for each sequence."""
    dev = _device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out = np.empty(len(seqs))
    for b in range(0, len(seqs), batch_size):
        chunk, st = seqs[b:b + batch_size], starts[b:b + batch_size]
        L = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, s in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(dev), att.to(dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        # gather on the device: only [B, L-1] numbers leave it, never [B, L, vocab]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None])[..., 0]
        pos = torch.arange(L - 1, device=dev)[None, :]
        lo = torch.tensor([s_ - 1 for s_ in st], device=dev)[:, None]
        hi = torch.tensor([len(s) - 1 for s in chunk], device=dev)[:, None]
        mask = (pos >= lo) & (pos < hi)
        out[b:b + len(chunk)] = (lp * mask).sum(1).cpu().numpy()
        del logits, lp
    return out


@torch.inference_mode()
def lm_correct(model, tok, texts, max_tokens=256):
    """Per passage: boolean vector of top-1 next-token correctness."""
    dev = _device(model)
    res = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(dev)
        if ids.shape[1] < 3:
            continue
        logits = model(input_ids=ids).logits[0, :-1]
        res.append((logits.argmax(-1) == ids[0, 1:]).cpu().numpy())
    return res


def measure(model, tok, enc, texts, batch_size):
    seqs, starts = enc
    lp = continuation_logprobs(model, tok, seqs, starts, batch_size)
    margin = lp[0::2] - lp[1::2]                 # correct minus wrong, per pair
    return {"truth_correct": (margin > 0).astype(float), "margin": margin,
            "gen_correct": lm_correct(model, tok, texts)}


# ─── Statistics (pure; tested without a model) ───────────────────────────────

def retentions(m0, m):
    t0 = m0["truth_correct"].mean()
    g0 = np.concatenate(m0["gen_correct"]).mean()
    t = m["truth_correct"].mean()
    g = np.concatenate(m["gen_correct"]).mean()
    return {"acc_truth": float(t), "acc_gen": float(g),
            "R_truth": float((t - 0.5) / (t0 - 0.5)) if t0 - 0.5 > 1e-9 else float("nan"),
            "R_gen": float(g / g0) if g0 > 1e-9 else float("nan"),
            "margin_mean": float(np.mean(m["margin"]))}


def causal_delta(base, head, tail, n_boot=2000, seed=0):
    """
    Delta = [R_truth(head) - R_truth(tail)] - [R_gen(head) - R_gen(tail)],
    with a bootstrap over pairs and over passages, denominators fixed.
    """
    t0 = base["truth_correct"].mean()
    g0 = np.concatenate(base["gen_correct"]).mean()
    if t0 - 0.5 <= 1e-9 or g0 <= 1e-9:
        return {"available": False, "reason": "no baseline performance to retain"}

    def point(ti, gi):
        th, tt = head["truth_correct"][ti].mean(), tail["truth_correct"][ti].mean()
        gh = np.concatenate([head["gen_correct"][j] for j in gi]).mean()
        gt = np.concatenate([tail["gen_correct"][j] for j in gi]).mean()
        rel_t = (th - tt) / (t0 - 0.5)
        rel_g = (gh - gt) / g0
        return rel_t - rel_g, rel_t, rel_g

    n_t, n_g = len(base["truth_correct"]), len(base["gen_correct"])
    d, rt, rg = point(np.arange(n_t), np.arange(n_g))
    rng = np.random.default_rng(seed)
    boots = np.array([point(rng.integers(0, n_t, n_t), rng.integers(0, n_g, n_g))[0]
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"available": True, "delta": float(d), "ci95": [float(lo), float(hi)],
            "reliance_truth": float(rt), "reliance_gen": float(rg), "n_boot": n_boot}


def gates(base, cond):
    """competence and sensitivity, as pre-registered."""
    t0 = float(base["truth_correct"].mean())
    cost = max(1 - cond["head"]["R_truth"], 1 - cond["tail"]["R_truth"],
               1 - cond["head"]["R_gen"], 1 - cond["tail"]["R_gen"])
    reasons = []
    if t0 < MIN_COMPETENCE:
        reasons.append(f"baseline truth accuracy {t0:.3f} < {MIN_COMPETENCE}")
    if not cost >= MIN_EFFECT:
        reasons.append(f"largest head/tail retention cost {cost:.3f} < {MIN_EFFECT}")
    return {"ok": not reasons, "baseline_truth_acc": t0, "max_cost": float(cost),
            "reasons": reasons}


# ─── Driver ──────────────────────────────────────────────────────────────────

def load_basis(model_name, cov=None):
    p = Path(cov) if cov else RESULTS_DIR / model_name / "en-fr" / "cov_src.pt"
    if not p.exists():
        raise SystemExit(f"no language-subset covariance at {p} (copy it from your "
                         f"working folder or rebuild with build_language_cov.py)")
    C = torch.load(p, map_location="cpu", weights_only=True).float()
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx], f"cov:{p}"


def run_loaded(model, tok, model_name, evals, evecs, basis, items, texts, args):
    """Everything after the model is loaded; separated so it can be tested."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    layers = decoder_layers(model)
    n_hs = len(layers) + 1
    d = evecs.shape[0]
    hidden = model.config.hidden_size if hasattr(model, "config") else d
    if hidden != d:
        raise SystemExit(f"basis dim {d} != model hidden size {hidden}")

    enc_seqs, enc_starts, mism = encode_items(tok, items)
    enc = (enc_seqs, enc_starts)
    roles = {"mid": n_hs // 2, "final": n_hs - 1}
    out = {"model_name": model_name, "basis": basis, "d": d, "n_hidden_states": n_hs,
           "n_pairs": len(items), "n_passages": len(texts),
           "tokenizer_boundary_mismatches": mism, "k_fracs": list(args.k_fracs),
           "primary": PRIMARY, "written_utc":
           datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "argv": sys.argv, "by_layer": {}}

    base = measure(model, tok, enc, texts, args.batch_size)
    out["baseline"] = retentions(base, base)
    print(f"  baseline: truth acc {out['baseline']['acc_truth']:.3f}  "
          f"general top-1 acc {out['baseline']['acc_gen']:.3f}  "
          f"(boundary mismatches {mism})")

    for role in args.layers:
        hs = roles[role]
        layer_mod = layers[hs - 1]            # output of layer hs-1 == hidden state hs
        L = {"hidden_state": hs, "by_k": {}}
        for kf in args.k_fracs:
            k = max(1, int(round(kf * d)))
            conds, raw = {}, {}
            for name, U in subspaces(evecs, k, args.n_random, args.seed).items():
                hook = ProjectOut(U).attach(layer_mod)
                try:
                    raw[name] = measure(model, tok, enc, texts, args.batch_size)
                finally:
                    hook.remove()
                conds[name] = retentions(base, raw[name])
            rnd = [conds[n] for n in conds if n.startswith("random")]
            conds["random_mean"] = {key: float(np.mean([c[key] for c in rnd]))
                                    for key in rnd[0]} if rnd else {}
            did = causal_delta(base, raw["head"], raw["tail"], args.n_boot, args.seed)
            g = gates(base, conds)
            L["by_k"][f"{kf:g}"] = {"k": k, "conditions": conds,
                                    "difference_in_differences": did, "gates": g}
            print(f"  {role:5s} hs{hs} k={k:4d} ({100 * kf:g}%)  "
                  f"R_truth head {conds['head']['R_truth']:+.2f} tail {conds['tail']['R_truth']:+.2f}  "
                  f"R_gen head {conds['head']['R_gen']:+.2f} tail {conds['tail']['R_gen']:+.2f}  "
                  + (f"Delta {did['delta']:+.3f} [{did['ci95'][0]:+.3f}, {did['ci95'][1]:+.3f}]"
                     if did.get("available") else "Delta n/a")
                  + ("" if g["ok"] else f"  [gate: {'; '.join(g['reasons'])}]"))
        out["by_layer"][role] = L
    return out


def cmd_run(args):
    import linear_rep_geometry as lrg
    items = truth_items(STATEMENTS / f"{args.dataset}.jsonl", args.max_pairs)
    texts = passages(args.passages)
    print(f"{len(items)} truth pairs, {len(texts)} general-text passages")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    failed = 0
    for m in args.models:
        name = m.split("/")[-1]
        path = Path(args.out_dir) / f"{name}.json"
        if args.skip_existing and path.exists():
            print(f"{name}: exists, skipping")
            continue
        print(f"\n=== {name} ===")
        try:
            evals, evecs, basis = load_basis(name, args.cov)
            model, tok, _ = lrg.load_model(m)
            model.eval()
            res = run_loaded(model, tok, name, evals, evecs, basis, items, texts, args)
            json.dump(res, open(path, "w"), indent=2, default=float)
            print(f"  wrote {path}")
        except SystemExit:
            raise
        except Exception as e:
            failed += 1
            print(f"  {name} failed: {type(e).__name__}: {e}")
        finally:
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 1 if failed else 0


def _f(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def cmd_summarize(args):
    from linear_rep_geometry.spectral_metrics import one_sample_test, tost_equivalence
    folder = Path(args.out_dir)
    res = [json.load(open(f)) for f in sorted(folder.glob("*.json"))
           if f.name != "summary.json"]
    if not res:
        raise SystemExit(f"no results in {folder}")

    lines = ["# Causal ablation: does the model rely on the quiet tail for truth?", "",
             "Δ > 0: removing the quiet tail (vs the loud head) hurts truth MORE than it "
             "hurts ordinary next-token prediction. R = share of above-chance "
             f"performance retained. Gates: baseline truth acc ≥ {MIN_COMPETENCE}; "
             f"head or tail removal costs ≥ {MIN_EFFECT} retention on some behaviour.", "",
             "## Per model", "",
             "| model | layer | k | base truth acc | R_truth head | R_truth tail | R_truth random | R_gen head | R_gen tail | R_gen random | Δ | 95% CI | gates |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cells = {}
    for r in res:
        for role, L in r["by_layer"].items():
            for kf, blk in L["by_k"].items():
                c, did, g = blk["conditions"], blk["difference_in_differences"], blk["gates"]
                ci = did.get("ci95", [None, None])
                lines.append(
                    f"| {r['model_name']} | {role} | {100 * float(kf):g}% | {g['baseline_truth_acc']:.3f} | "
                    f"{_f(c['head']['R_truth'])} | {_f(c['tail']['R_truth'])} | {_f(c['random_mean'].get('R_truth'))} | "
                    f"{_f(c['head']['R_gen'])} | {_f(c['tail']['R_gen'])} | {_f(c['random_mean'].get('R_gen'))} | "
                    f"{_f(did.get('delta'))} | [{_f(ci[0])}, {_f(ci[1])}] | "
                    f"{'ok' if g['ok'] else '; '.join(g['reasons'])} |")
                if did.get("available") and g["ok"]:
                    cells.setdefault((role, float(kf)), []).append((r["model_name"], did["delta"]))

    lines += ["", "## Across models (cells passing both gates)", "",
              "| layer | k | models | n | mean Δ | 95% CI | p (t) | p (Wilcoxon) | 90% CI | p TOST | equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    tests = {}
    for (role, kf) in sorted(cells, key=lambda t: (t[0] != PRIMARY["layer"], t[1])):
        for drop in (False, True):
            v = [dl for m, dl in cells[(role, kf)] if not (drop and m == GEMMA)]
            key = f"{role}|{kf:g}|{'without_gemma' if drop else 'all'}"
            if len(v) < 3:
                tests[key] = {"n": len(v)}
                continue
            t, q = one_sample_test(v), tost_equivalence(v, sesoi=SESOI)
            tests[key] = {"n": len(v), "test": t, "tost": q}
            ci, ci90 = t.get("ci95", [None, None]), q.get("ci90", [None, None])
            prim = role == PRIMARY["layer"] and abs(kf - PRIMARY["k_frac"]) < 1e-9
            lines.append(f"| {role} | {('**' if prim else '')}{100 * kf:g}%{(' (primary)**' if prim else '')} | "
                         f"{'without gemma' if drop else 'all'} | {len(v)} | {_f(t['mean'])} | "
                         f"[{_f(ci[0])}, {_f(ci[1])}] | {t.get('p_t', float('nan')):.3f} | "
                         f"{t.get('p_wilcoxon', float('nan')):.3f} | [{_f(ci90[0])}, {_f(ci90[1])}] | "
                         f"{q.get('p_tost', float('nan')):.3f} | {'yes' if q.get('conclusive') else 'no'} |")

    def verdict():
        a = tests.get(f"{PRIMARY['layer']}|{PRIMARY['k_frac']:g}|all", {})
        w = tests.get(f"{PRIMARY['layer']}|{PRIMARY['k_frac']:g}|without_gemma", {})
        if a.get("n", 0) < MIN_MODELS or "test" not in a or "test" not in w:
            return "NOT TESTABLE (fewer than {} models pass the gates)".format(MIN_MODELS)
        above = all(x["test"]["ci95"][0] > 0 for x in (a, w))
        equiv = all(-SESOI < x["tost"]["ci90"][0] and x["tost"]["ci90"][1] < SESOI for x in (a, w))
        below = all(x["test"]["ci95"][1] < 0 for x in (a, w))
        if above:
            return "SPECIFIC: the model relies on the quiet tail more for truth than for general prediction"
        if equiv or below:
            return "NOT SPECIFIC" + (" (reversed: the tail matters more for general prediction)" if below else "")
        return "INCONCLUSIVE"

    lines += ["", f"**Primary verdict ({PRIMARY['layer']} layer, k = {100 * PRIMARY['k_frac']:g}% of d): {verdict()}**", "",
              "Rule: specific if the 95% CI is above 0 with and without gemma-2-2b; not specific "
              "if the 90% CI is inside ±0.10 (or the 95% CI is below 0) in both; otherwise inconclusive."]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"tests": tests}, open(folder / "summary.json", "w"), indent=2, default=float)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument("--dataset", default="truth_cities")
    r.add_argument("--passages", default=str(PASSAGES))
    r.add_argument("--max-pairs", type=int, default=512)
    r.add_argument("--layers", nargs="+", default=["mid", "final"], choices=["mid", "final"])
    r.add_argument("--k-fracs", type=float, nargs="+", default=list(K_FRACS))
    r.add_argument("--n-random", type=int, default=N_RANDOM)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cov")
    r.add_argument("--out-dir", default=str(OUT_DEFAULT))
    r.add_argument("--skip-existing", action="store_true")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize")
    s.add_argument("--out-dir", default=str(OUT_DEFAULT))
    s.set_defaults(func=cmd_summarize)
    args = ap.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
