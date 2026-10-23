#!/usr/bin/env python3
"""Does Gini track what the direction actually does?

Real and shuffled directions have indistinguishable Gini. Spectral measurement
cannot say whether that means concepts are not special, or Gini is too coarse.
The cosine cannot either: in high dimensions a sign-flipped mean is
near-orthogonal to the real mean whether or not a signal is present (synthetic:
cos 0.17 with strong signal, 0.35 with pure noise).

A causal test can. If the real direction changes behaviour and the shuffled one
does not, two directions with the same Gini do different things -- a positive
claim about the metric rather than a null about concepts.

Measures activation addition in the INDUCE direction (paper Figure 3): add the
direction to harmless prompts and see if the model moves toward refusing.
Deliberately not ablation, so no harmful completions are produced.

Outcome: Arditi's refusal metric, log-odds of refusal-opening tokens at the
next position. No generation, so minutes not hours.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import linear_rep_geometry as lrg
from linear_rep_geometry.behavioral import load_concept, load_pairs
from linear_rep_geometry.dual_geometry import REFUSAL_PREFIXES


def refusal_token_ids(tokenizer):
    ids = set()
    for p in REFUSAL_PREFIXES:
        for text in (p, " " + p):
            toks = tokenizer.encode(text, add_special_tokens=False)
            if toks: ids.add(int(toks[0]))
    return sorted(ids)


@torch.inference_mode()
def refusal_metric(prompts, model, tokenizer, handler, refusal_ids,
                   direction=None, alpha=0.0, layer=None,
                   device="cuda", batch_size=8, max_length=128):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = None
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in attr.split("."): obj = getattr(obj, part)
            blocks = obj; break
        except AttributeError: continue
    if blocks is None:
        raise RuntimeError("could not locate transformer blocks")

    handle = None
    if direction is not None and alpha != 0.0:
        v = (direction.to(device) * alpha)
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].dtype),) + out[1:]
            return out + v.to(out.dtype)
        handle = blocks[layer].register_forward_hook(hook)

    try:
        scores = []
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i: i + batch_size])
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            logits = model(enc.input_ids, attention_mask=enc.attention_mask).logits
            lengths = enc.attention_mask.sum(dim=1)
            for j in range(len(batch)):
                last = int(lengths[j].item()) - 1
                probs = torch.softmax(logits[j, last, :].float(), dim=-1)
                p_ref = probs[refusal_ids].sum().clamp(1e-9, 1 - 1e-9)
                scores.append(float(torch.log(p_ref / (1 - p_ref))))
    finally:
        if handle is not None: handle.remove()
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-dir", default="data/statements")
    ap.add_argument("--pooling", default="last_token")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = Path(args.model).name
    root = Path(args.results_root) / tag
    cp = load_concept(root / "concepts" / f"concept_{args.concept}.pt")
    layer = args.layer if args.layer is not None else cp["n_layers"] // 2
    D = cp["diffs"][(args.pooling, layer)]

    v_raw = D.mean(dim=0); scale = float(v_raw.norm())
    v_real = (v_raw / max(scale, 1e-12)).float()
    rng = np.random.default_rng(args.seed); d = v_real.shape[0]

    v_shuffled = []
    for _ in range(args.n_controls):
        s = torch.tensor(rng.choice([-1.0, 1.0], size=len(D)),
                         dtype=torch.float32).unsqueeze(1)
        u = (D * s).mean(dim=0)
        v_shuffled.append((u / u.norm().clamp(min=1e-12)).float())
    v_random = []
    for _ in range(args.n_controls):
        u = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
        v_random.append(u / u.norm())

    pairs = load_pairs(Path(args.data_dir) / f"{args.concept}.jsonl")
    prompts = [p.negative for p in pairs][: args.n_prompts]

    print(f"model {args.model}  concept {args.concept}  layer {layer}")
    print(f"{len(prompts)} harmless prompts, ||mean diff|| = {scale:.3f}")

    model, tok, handler = lrg.load_model(args.model, device_map="auto")
    model.eval()
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise SystemExit(f"model on {dev}, not GPU -- refusing to run on CPU")
    ids = refusal_token_ids(tok)
    print(f"refusal tokens: {len(ids)} ids")

    base, base_se = refusal_metric(prompts, model, tok, handler, ids,
                                   device=str(dev), batch_size=args.batch_size)
    print(f"\nbaseline refusal metric {base:+.4f} (se {base_se:.4f})")
    print("(negative = not inclined to refuse these; expected)")
    print(f"\n{'alpha':>7s}  {'real':>18s}  {'shuffled':>18s}  {'random':>18s}")
    print("-" * 70)

    results = {"baseline": base, "scale": scale, "layer": int(layer),
               "n_prompts": len(prompts), "alphas": {}}
    for a in args.alphas:
        alpha = a * scale
        r, _ = refusal_metric(prompts, model, tok, handler, ids, direction=v_real,
                              alpha=alpha, layer=layer, device=str(dev),
                              batch_size=args.batch_size)
        sh = [refusal_metric(prompts, model, tok, handler, ids, direction=v,
                             alpha=alpha, layer=layer, device=str(dev),
                             batch_size=args.batch_size)[0] for v in v_shuffled]
        rd = [refusal_metric(prompts, model, tok, handler, ids, direction=v,
                             alpha=alpha, layer=layer, device=str(dev),
                             batch_size=args.batch_size)[0] for v in v_random]
        d_real = r - base; d_sh = np.mean(sh) - base; d_rd = np.mean(rd) - base
        print(f"{a:7.1f}  {d_real:+10.4f}        "
              f"{d_sh:+10.4f}±{np.std(sh):.3f}  {d_rd:+10.4f}±{np.std(rd):.3f}"
              f"   beat {int((np.array(sh)-base >= d_real).sum())}/{len(sh)}"
              f" p={(int((np.array(sh)-base >= d_real).sum())+1)/(len(sh)+1):.3f}")
        results["alphas"][str(a)] = {
            "delta_real": float(d_real),
            "delta_shuffled_mean": float(d_sh), "delta_shuffled_sd": float(np.std(sh)),
            "delta_random_mean": float(d_rd), "delta_random_sd": float(np.std(rd))}

    print("\n=== Reading ===")
    e = results["alphas"][str(max(args.alphas))]
    dr, ds = e["delta_real"], e["delta_shuffled_mean"]
    sd = max(e["delta_shuffled_sd"], 1e-6)
    if dr > ds + 2 * sd and dr > 0.5:
        print("  Real induces refusal, shuffled does not, despite equal Gini.")
        print("  Spectrally indistinguishable directions are causally different,")
        print("  so Gini is not tracking what the direction does. Report as a")
        print("  limitation of the metric -- the useful outcome.")
    elif abs(dr - ds) < 2 * sd and dr > 0.5:
        print("  Real and shuffled are BOTH effective. The shuffle preserves")
        print("  causal power, so it may remove too little. This undercuts the")
        print("  main result and must be reported.")
    elif dr < 0.5:
        print("  Nothing moved. Raise alpha or change layer before concluding.")
    else:
        print("  Intermediate. Report the alpha sweep and argue from the curve.")

    out = Path(args.out or f"results/_analysis/causal_{tag}_{args.concept}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
