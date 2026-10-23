"""
Phase 1 and 3 — behavioural concept extraction.

The published extraction reads a pooled activation at a target word. A statement
has no target word, so two things become free parameters and the proposal
commits to pinning both rather than choosing quietly:

  Pooling.  last_token and mean_over_sequence are both computed, and a
            conclusion is only reported if it holds under both. Last-token is
            what Arditi and Marks & Tegmark use; mean-over-sequence is the
            obvious alternative and disagrees when a direction is carried by
            one position rather than by the statement.

  Layer.    Word vectors were read at the final layer, behavioural directions
            are conventionally read mid-depth, and the published layer-wise
            curve moves a lot with depth. So every layer is extracted and
            trajectories are compared, never single points.

Per-example differences are retained, not just their mean, because Phase 4 needs
them for the rank test and Phase 2 needs them for the label-shuffle null.
Keeping them is what makes those phases cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

POOLING = ("last_token", "mean_over_sequence")


@dataclass
class StatementPair:
    concept: str
    positive: str
    negative: str
    pair_id: Optional[str] = None
    subdataset: Optional[str] = None
    matched: bool = False
    contrast_type: str = "prompt"   # "prompt" | "answer_token"
    meta: dict = field(default_factory=dict)


def load_pairs(path: str | Path) -> list[StatementPair]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            r = json.loads(line)
            for k in ("concept", "positive", "negative"):
                if k not in r:
                    raise ValueError(f"{path}:{ln} missing '{k}'")
            pairs.append(StatementPair(
                concept=r["concept"], positive=r["positive"], negative=r["negative"],
                pair_id=r.get("pair_id"), subdataset=r.get("subdataset"),
                matched=bool(r.get("matched", False)),
                contrast_type=r.get("contrast_type", "prompt"),
                meta={k: v for k, v in r.items()
                      if k not in {"concept", "positive", "negative", "pair_id",
                                   "subdataset", "matched", "contrast_type"}}))
    return pairs


def prompt_overlap(pairs: Sequence[StatementPair], tokenizer,
                   sample: int = 50) -> dict:
    """
    How much of each pair is shared between positive and negative?

    This decides which pooling is meaningful, and it is measured rather than
    assumed because the answer differs by an order of magnitude across concepts.
    Refusal pairs share almost nothing — different corpora entirely. Truth pairs
    share a template. Sycophancy pairs share everything but one answer token.

    When the shared fraction is near 1, mean_over_sequence dilutes a one-token
    difference across a couple of hundred positions and measures mostly noise.
    The proposal requires conclusions to hold under both poolings; this is the
    evidence for exempting a concept from that rule, and for saying so in the
    write-up instead of quietly reporting the pooling that worked.
    """
    import numpy as _np
    fracs, diffs = [], []
    for p in list(pairs)[:sample]:
        a = tokenizer.encode(p.positive, add_special_tokens=False)
        b = tokenizer.encode(p.negative, add_special_tokens=False)
        n = min(len(a), len(b))
        same = sum(1 for i in range(n) if a[i] == b[i])
        total = max(len(a), len(b))
        fracs.append(same / total if total else 0.0)
        diffs.append(total - same)
    if not fracs:
        return {"median_frac_shared": 0.0, "median_differing": 0, "n_sampled": 0}
    return {"median_frac_shared": float(_np.median(fracs)),
            "median_differing": int(_np.median(diffs)),
            "n_sampled": len(fracs)}


def format_prompts(prompts: Sequence[str], tokenizer,
                   use_chat_template: bool = False,
                   system_prompt: Optional[str] = None) -> list[str]:
    """
    Apply a chat template only when asked for.

    The model suite is base models, matching the published study, so the default
    is off. Templating some models and not others would put a format confound
    straight into the cross-model comparison that carries the result.
    """
    if not use_chat_template:
        return list(prompts)
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("no chat_template on this tokenizer (expected for base "
                         "models); pass use_chat_template=False")
    out = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": p})
        out.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True))
    return out


@torch.inference_mode()
def layerwise_activations(prompts: Sequence[str], model, tokenizer, handler,
                          device: str = "cuda", batch_size: int = 8,
                          max_length: int = 256) -> dict[str, torch.Tensor]:
    """
    Residual-stream activations at every layer, under both pooling schemes.

    Returns {pooling: tensor [n_prompts, n_layers, d]} with n_layers = L+1;
    index 0 is the embedding layer, which the published layer-wise control
    also reports.

    Padding is masked out of the mean rather than averaged in, and the last
    token is found per example from the attention mask, so right-padded
    tokenizers (which several handlers in this repo set) do not silently read
    pad positions.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = handler._detect_device(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = {p: [] for p in POOLING}

    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i: i + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        out = model(enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True)
        hs = out.hidden_states
        nL = len(hs)
        mask = enc.attention_mask
        lengths = mask.sum(dim=1)

        last = torch.empty(len(batch), nL, hs[0].shape[-1], dtype=torch.float32)
        mean = torch.empty_like(last)

        m = mask.unsqueeze(-1).float()
        for li, h in enumerate(hs):
            if li == nL - 1:
                h = handler.apply_final_norm(model, h)
            hf = h.float()
            summed = (hf * m).sum(dim=1)
            mean[:, li, :] = (summed / lengths.unsqueeze(1).clamp(min=1)).cpu()
            for j in range(len(batch)):
                last[j, li, :] = hf[j, int(lengths[j].item()) - 1, :].cpu()

        chunks["last_token"].append(last)
        chunks["mean_over_sequence"].append(mean)
        del out, hs

    return {p: torch.cat(v, dim=0) for p, v in chunks.items()}


def extract_concept(pairs: Sequence[StatementPair], model, tokenizer, handler,
                    device: str = "cuda", batch_size: int = 8,
                    max_length: int = 256, use_chat_template: bool = False,
                    system_prompt: Optional[str] = None) -> dict:
    """
    One concept, end to end: both poolings, every layer, differences retained.

    Returns
    -------
    {
      "directions": {(pooling, layer): unit vector [d]},
      "diffs":      {(pooling, layer): per-example differences [n, d]},
      "acts":       {pooling: {"pos": [n,L,d], "neg": [n,L,d]}},
      "n_pairs", "n_layers", "matched", "d"
    }

    Nothing is selected. Which layer and pooling carry the headline is a
    decision the analysis plan records, not something extraction should bury.

    When pair_id is present the difference is taken per pair and then averaged;
    otherwise the pooled difference of means is used. For a genuinely matched
    pair the two agree in expectation, and `matched_vs_unmatched` in nulls.py
    measures the divergence when they do not.
    """
    contrast_type = pairs[0].contrast_type if pairs else "prompt"
    overlap = prompt_overlap(pairs, tokenizer)

    if contrast_type == "answer_token" or overlap["median_frac_shared"] > 0.9:
        print(f"    contrast_type={contrast_type}: positive and negative share "
              f"{overlap['median_frac_shared']:.1%} of their tokens "
              f"(median {overlap['median_differing']} differing).")
        print("    last_token is the informative pooling here. "
              "mean_over_sequence averages a difference of a few tokens over "
              "the whole\n    sequence and will be dominated by noise — report "
              "it, but do not require\n    the conclusion to hold under it.")

    pos = format_prompts([p.positive for p in pairs], tokenizer,
                         use_chat_template, system_prompt)
    neg = format_prompts([p.negative for p in pairs], tokenizer,
                         use_chat_template, system_prompt)

    A = layerwise_activations(pos, model, tokenizer, handler, device, batch_size, max_length)
    B = layerwise_activations(neg, model, tokenizer, handler, device, batch_size, max_length)

    matched = all(p.pair_id is not None for p in pairs)
    n_layers = A["last_token"].shape[1]
    d = A["last_token"].shape[2]

    directions, diffs = {}, {}
    for pooling in POOLING:
        for li in range(n_layers):
            xp, xn = A[pooling][:, li, :], B[pooling][:, li, :]
            D = xp - xn                      # [n, d]
            v = D.mean(dim=0) if matched else (xp.mean(dim=0) - xn.mean(dim=0))
            directions[(pooling, li)] = v / v.norm().clamp(min=1e-12)
            diffs[(pooling, li)] = D

    return {"directions": directions, "diffs": diffs,
            "acts": {p: {"pos": A[p], "neg": B[p]} for p in POOLING},
            "n_pairs": len(pairs), "n_layers": n_layers, "d": d,
            "matched": matched, "contrast_type": contrast_type,
            "overlap": overlap,
            "informative_poolings": (["last_token"]
                                     if contrast_type == "answer_token"
                                     else list(POOLING))}


def save_concept(res: dict, out_dir: str | Path, concept: str,
                 save_acts: bool = True) -> None:
    """
    Persist directions, per-example differences, and optionally activations.

    Differences are saved by default because Phases 2 and 4 both need them and
    re-extracting is the slow path. They are large — [n, d] float32 per
    (pooling, layer) — so for the breadth sweep set save_acts=False and keep
    only the differences.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(res["directions"].keys())
    payload = {
        "keys": keys,
        "directions": torch.stack([res["directions"][k] for k in keys]),
        "diffs": {f"{p}|{l}": res["diffs"][(p, l)] for (p, l) in keys},
        "n_pairs": res["n_pairs"], "n_layers": res["n_layers"],
        "d": res["d"], "matched": res["matched"],
        "contrast_type": res.get("contrast_type", "prompt"),
        "overlap": res.get("overlap", {}),
        "informative_poolings": res.get("informative_poolings", list(POOLING)),
    }
    torch.save(payload, out_dir / f"concept_{concept}.pt")
    if save_acts:
        torch.save(res["acts"], out_dir / f"acts_{concept}.pt")

    with open(out_dir / f"concept_{concept}.json", "w") as f:
        json.dump({"concept": concept, "n_pairs": res["n_pairs"],
                   "n_layers": res["n_layers"], "d": res["d"],
                   "matched": res["matched"], "poolings": list(POOLING),
                   "contrast_type": res.get("contrast_type", "prompt"),
                   "overlap": res.get("overlap", {}),
                   "informative_poolings": res.get("informative_poolings",
                                                   list(POOLING))}, f, indent=2)


def load_concept(path: str | Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    keys = [(k[0], int(k[1])) for k in d["keys"]]
    return {
        "directions": {k: d["directions"][i] for i, k in enumerate(keys)},
        "diffs": {(p, int(l)): v for k, v in d["diffs"].items()
                  for p, l in [k.split("|")]},
        "n_pairs": d["n_pairs"], "n_layers": d["n_layers"],
        "d": d["d"], "matched": d["matched"],
        "contrast_type": d.get("contrast_type", "prompt"),
        "overlap": d.get("overlap", {}),
        "informative_poolings": d.get("informative_poolings", list(POOLING)),
    }


def word_level_control(concept_pairs: Sequence[tuple[str, str]], model, tokenizer,
                       handler, device: str = "cuda", batch_size: int = 16) -> dict:
    """
    The internal control the proposal requires (§6).

    Word-level directions computed here, on this model, in this basis, at these
    layers — not read off the published table. Comparing a behavioural number
    measured one way against a printed number measured another is the kind of
    mismatch the paired test in §3 exists to avoid.

    concept_pairs is a list of (positive_word, negative_word), e.g. the
    counterfactual pairs behind one of the published 27 concepts.
    """
    pairs = [StatementPair(concept="word_control", positive=a, negative=b,
                           pair_id=f"w{i:04d}", matched=True)
             for i, (a, b) in enumerate(concept_pairs)]
    return extract_concept(pairs, model, tokenizer, handler,
                           device=device, batch_size=batch_size, max_length=16)
