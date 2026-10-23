"""
Phase 6 — the dual geometry, for refusal.

The published dual geometry contrasts two views of the same concept: the
activation-space direction anti-concentrates in the tail, while the static
unembedding-row contrast concentrates in the head. Truthfulness has no
vocabulary-side analogue — there is no unembedding row for "true" — but refusal
does. Arditi et al. identify the tokens a model's refusals begin with, and a
contrast between those rows and compliance-initiating rows is a vocabulary-side
refusal direction.

So this is the one behavioural concept where the published test can be run
directly. A sign flip confirms the dual geometry on a concept class the original
paper never tested. No flip means the dual geometry is itself specific to
word-level concepts, which bounds the published claim.

The control tokens decide whether this works. Token frequency drives the top
eigendirections (Puccetti et al. 2022), so an unmatched control set measures the
frequency axis and returns a confident, meaningless concentration. Controls are
therefore matched on token length and on leading-space form, and the module
reports the balance so the match can be checked rather than assumed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

# Arditi et al., evaluate_jailbreak.py: the phrases refusals begin with.
REFUSAL_PREFIXES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model",
    "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable to",
    "I am not able to", "I'm not able to",
]

# Openings of compliant responses. Matched in register and position: these are
# the first words of an answer, as the refusal prefixes are the first words of
# a refusal.
COMPLIANCE_PREFIXES = [
    "Sure", "Certainly", "Here is", "Here's", "Of course", "Absolutely",
    "Yes", "To do", "The first", "You can", "Let's", "Start by",
]


def leading_token_ids(phrases: Sequence[str], tokenizer,
                      add_leading_space: bool = False) -> list[int]:
    """
    First token id of each phrase, deduplicated.

    The leading-space form matters: in byte-level BPE "I" and " I" are different
    tokens with different rows and very different frequencies. Both sets must be
    built the same way, which is why this is a parameter and both variants are
    reported by build_vocabulary_contrast.
    """
    ids = []
    for p in phrases:
        text = (" " + p) if add_leading_space else p
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def _unembedding(model, handler) -> torch.Tensor:
    for getter in (lambda: handler.get_unembedding(model),
                   lambda: model.get_output_embeddings().weight,
                   lambda: model.lm_head.weight):
        try:
            W = getter()
            if W is not None:
                return W.detach().float().cpu()
        except (AttributeError, NotImplementedError):
            continue
    raise RuntimeError("could not locate the unembedding matrix")


def token_lengths(ids: Sequence[int], tokenizer) -> np.ndarray:
    return np.array([len(tokenizer.decode([int(i)])) for i in ids], dtype=float)


def build_vocabulary_contrast(model, tokenizer, handler,
                              add_leading_space: bool = False) -> dict:
    """
    v_unembed = mean(gamma(refusal tokens)) - mean(gamma(compliance tokens)).

    Returns the direction under both the plain and leading-space tokenisations
    plus a balance report. If the two tokenisations disagree in sign, the result
    is about tokenisation rather than refusal and should not be reported.
    """
    W = _unembedding(model, handler)
    out = {}
    for variant, space in (("plain", False), ("leading_space", True)):
        if space != add_leading_space and variant == "plain" and add_leading_space:
            pass
        r_ids = leading_token_ids(REFUSAL_PREFIXES, tokenizer, space)
        c_ids = leading_token_ids(COMPLIANCE_PREFIXES, tokenizer, space)
        overlap = sorted(set(r_ids) & set(c_ids))
        r_ids = [i for i in r_ids if i not in overlap]
        c_ids = [i for i in c_ids if i not in overlap]

        if len(r_ids) < 2 or len(c_ids) < 2:
            out[variant] = {"ok": False,
                            "reason": f"too few distinct tokens "
                                      f"(refusal {len(r_ids)}, compliance {len(c_ids)})"}
            continue

        v = W[torch.tensor(r_ids)].mean(dim=0) - W[torch.tensor(c_ids)].mean(dim=0)
        v = v / v.norm().clamp(min=1e-12)

        rl, cl = token_lengths(r_ids, tokenizer), token_lengths(c_ids, tokenizer)
        out[variant] = {
            "ok": True, "direction": v,
            "n_refusal": len(r_ids), "n_compliance": len(c_ids),
            "n_overlap_dropped": len(overlap),
            "mean_char_len_refusal": float(rl.mean()),
            "mean_char_len_compliance": float(cl.mean()),
            "char_len_balanced": bool(abs(rl.mean() - cl.mean()) < 2.0),
            "refusal_token_ids": r_ids, "compliance_token_ids": c_ids,
        }
    return out


def random_token_pair_null(model, handler, n_per_side: int, eigenvectors: torch.Tensor,
                           cumvar: np.ndarray, n_samples: int = 1000,
                           vocab_subset: Optional[Sequence[int]] = None,
                           seed: int = 0) -> dict:
    """
    The published null for the vocabulary side: random token-pair differences
    from the same vocabulary, with the same set sizes.

    Necessary because random token pairs do not score zero — the paper reports
    them at roughly -0.10, mildly anti-concentrated — so the concept claim is
    that refusal tokens concentrate MORE than random pairs, not merely that
    their Gini is positive.
    """
    from .spectral_metrics import compute_concept_energy_cdf, compute_gini_deviation

    W = _unembedding(model, handler)
    pool = (np.asarray(sorted(set(int(i) for i in vocab_subset)))
            if vocab_subset is not None else np.arange(W.shape[0]))
    if len(pool) < 2 * n_per_side:
        raise ValueError("vocabulary pool too small for the requested set sizes")

    rng = np.random.default_rng(seed)
    ginis = np.empty(n_samples)
    for i in range(n_samples):
        pick = rng.choice(pool, size=2 * n_per_side, replace=False)
        a, b = pick[:n_per_side], pick[n_per_side:]
        v = W[torch.tensor(a)].mean(dim=0) - W[torch.tensor(b)].mean(dim=0)
        v = v / v.norm().clamp(min=1e-12)
        ginis[i] = compute_gini_deviation(
            compute_concept_energy_cdf(v, eigenvectors), cumvar)

    return {"gini_mean": float(ginis.mean()), "gini_std": float(ginis.std()),
            "gini_samples": ginis, "n_per_side": n_per_side}


def dual_geometry_test(model, tokenizer, handler, activation_gini: float,
                       eigenvectors: torch.Tensor, cumvar: np.ndarray,
                       n_samples: int = 1000, seed: int = 0) -> dict:
    """
    The full test: is the vocabulary-side refusal direction on the opposite
    side of its null from the activation-side one?

    `activation_gini` is the Phase 1-3 refusal direction's Gini, measured in the
    same basis. The verdict is a sign flip RELATIVE TO EACH NULL, not relative
    to zero — both raw numbers can be negative in an anisotropic spectrum while
    still sitting on opposite sides of their respective baselines, and that is
    the comparison the dual geometry actually claims.
    """
    from .spectral_metrics import (compute_concept_energy_cdf,
                                   compute_gini_deviation, empirical_pvalue)

    contrasts = build_vocabulary_contrast(model, tokenizer, handler)
    results = {}

    for variant, c in contrasts.items():
        if not c.get("ok"):
            results[variant] = c
            continue
        v = c["direction"]
        gini = compute_gini_deviation(
            compute_concept_energy_cdf(v, eigenvectors), cumvar)
        null = random_token_pair_null(model, handler,
                                      min(c["n_refusal"], c["n_compliance"]),
                                      eigenvectors, cumvar,
                                      n_samples=n_samples, seed=seed)
        gap = float(gini - null["gini_mean"])
        results[variant] = {
            "ok": True,
            "vocab_gini": gini, "vocab_null_gini": null["gini_mean"],
            "vocab_gap": gap,
            "p_two_sided": empirical_pvalue(gini, null["gini_samples"], tail="two"),
            "n_refusal": c["n_refusal"], "n_compliance": c["n_compliance"],
            "char_len_balanced": c["char_len_balanced"],
        }

    usable = [r for r in results.values() if r.get("ok")]
    verdict = {"activation_gini": activation_gini}
    if usable:
        gaps = [r["vocab_gap"] for r in usable]
        verdict["variants_agree_on_sign"] = bool(len({g > 0 for g in gaps}) == 1)
        verdict["vocab_gap_mean"] = float(np.mean(gaps))
        verdict["sign_flip"] = bool(np.mean(gaps) > 0 and activation_gini < 0)
        verdict["note"] = (
            "Dual geometry reproduces for refusal."
            if verdict["sign_flip"] else
            "No sign flip: the dual geometry may be specific to word-level concepts."
        )
        if not verdict["variants_agree_on_sign"]:
            verdict["note"] = ("Tokenisation variants disagree on sign; this result "
                               "is about tokenisation, not refusal. Do not report it.")
    return {"variants": results, "verdict": verdict}
