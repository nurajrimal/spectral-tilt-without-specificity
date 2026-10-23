"""
The word-level internal control (proposal §6).

Section 6 requires the published 27-concept set carried through the whole study
"so that every behavioural number has a same-model, same-basis, same-layer
comparison rather than being compared against a published table". Three words in
that sentence do the work: same model, same basis, SAME LAYER.

That last one is why this module exists rather than a one-line call to
core.concept_direction. The published extractor returns a single vector from
pooled final-layer activations, which is correct for the published analysis but
cannot support the paired test: Phase 3 compares trajectories, and a behavioural
direction read at mid-depth against a word direction read at the final layer is
exactly the confound Phase 3 was added to remove.

So directions are extracted at every layer, and `verify_against_published`
checks that at the final layer they reproduce core.concept_direction. If that
check fails the control is not comparable to the published numbers and the
mismatch must be resolved before anything is reported.

Context-free extraction, matching the published call: the input text IS the
word, so pooling over the sequence is pooling over the target word. That makes
behavioral.layerwise_activations the right machinery, which is also what keeps
the control and the behavioural concepts on identical code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .behavioral import POOLING, layerwise_activations

# "[male - female].txt" -> "male - female"
_NAME = re.compile(r"^\[(.+)\]$")


def list_concepts(word_pair_dir: str | Path) -> list[str]:
    """Concept names available in data/word_pairs/, sorted."""
    out = []
    for p in sorted(Path(word_pair_dir).glob("*.txt")):
        m = _NAME.match(p.stem)
        out.append(m.group(1) if m else p.stem)
    return out


def load_word_pairs(word_pair_dir: str | Path, concept: str) -> list[tuple[str, str]]:
    """
    Read one concept file: tab-separated base<TAB>target, one pair per line.

    Malformed lines are skipped rather than raising. These files are curated but
    a stray blank or single-column line should not take down a 42-concept sweep
    eight models deep.
    """
    d = Path(word_pair_dir)
    for cand in (d / f"[{concept}].txt", d / f"{concept}.txt"):
        if cand.exists():
            path = cand
            break
    else:
        raise FileNotFoundError(f"no word-pair file for '{concept}' in {d}")

    pairs = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def extract_word_concept(pairs: Sequence[tuple[str, str]], model, tokenizer, handler,
                         device: str = "cuda", batch_size: int = 16,
                         max_length: int = 16) -> dict:
    """
    One word-level concept at every layer, under both poolings.

    Direction is target minus base, matching core.concept_direction's
    `diff_data = target_data - base_data`. Getting that order backwards would
    flip every sign in the control and the paired test would silently compare a
    direction to its negation — which Gini would not catch, since Gini is
    sign-invariant, but the cosine diagnostics and any steering use would.

    Returns the same structure as behavioral.extract_concept, so the control
    flows through Phases 2 to 5 on the same code paths as the behavioural
    concepts. That shared path is the point: it is what makes the comparison
    same-basis and same-layer by construction rather than by care.
    """
    base = [b for b, _ in pairs]
    target = [t for _, t in pairs]

    A = layerwise_activations(target, model, tokenizer, handler,
                              device, batch_size, max_length)
    B = layerwise_activations(base, model, tokenizer, handler,
                              device, batch_size, max_length)

    n_layers = A["last_token"].shape[1]
    d = A["last_token"].shape[2]
    directions, diffs = {}, {}
    for pooling in POOLING:
        for li in range(n_layers):
            D = A[pooling][:, li, :] - B[pooling][:, li, :]   # target - base
            v = D.mean(dim=0)
            directions[(pooling, li)] = v / v.norm().clamp(min=1e-12)
            diffs[(pooling, li)] = D

    return {"directions": directions, "diffs": diffs,
            "acts": {p: {"pos": A[p], "neg": B[p]} for p in POOLING},
            "n_pairs": len(pairs), "n_layers": n_layers, "d": d,
            "matched": True}


def verify_against_published(res: dict, pairs: Sequence[tuple[str, str]],
                             model, tokenizer, handler, device: str = "cuda",
                             batch_size: int = 4, tol: float = 0.95) -> dict:
    """
    Check the final-layer direction against core.concept_direction.

    The published extractor pools differently in detail — its own target-word
    span logic rather than a sequence mean — so the two are not expected to be
    bit-identical. They should be nearly collinear, because with context-free
    input the target span IS the sequence.

    Cosine below `tol` means the control is NOT measuring what the published
    pipeline measures, and the paired test would be comparing two different
    quantities. Resolve it before reporting; do not widen the tolerance.

    Returns the best cosine across poolings and which pooling achieved it.
    """
    from . import core as published_core

    base = [b for b, _ in pairs]
    target = [t for _, t in pairs]
    v_pub, _ = published_core.concept_direction(
        base, target, model, tokenizer, handler, device, batch_size=batch_size)
    v_pub = v_pub.float().cpu()
    v_pub = v_pub / v_pub.norm().clamp(min=1e-12)

    last = res["n_layers"] - 1
    cosines = {}
    for pooling in POOLING:
        v = res["directions"][(pooling, last)].float()
        cosines[pooling] = float(torch.dot(v, v_pub))

    best_pool = max(cosines, key=lambda k: abs(cosines[k]))
    best = cosines[best_pool]
    out = {"cosines": cosines, "best_pooling": best_pool, "best_cosine": best,
           "tol": tol, "ok": bool(abs(best) >= tol),
           "sign_agrees": bool(best > 0)}
    if not out["ok"]:
        out["warning"] = (
            f"final-layer cosine against the published extractor is {best:.3f}, "
            f"below {tol}. The control is not measuring the published quantity; "
            f"the paired test would compare two different things.")
    elif not out["sign_agrees"]:
        out["warning"] = (
            "magnitude matches but the sign is inverted — check the "
            "target-minus-base order against core.concept_direction.")
    return out


def extract_all(word_pair_dir: str | Path, model, tokenizer, handler,
                device: str = "cuda", batch_size: int = 16,
                concepts: Optional[Sequence[str]] = None,
                min_pairs: int = 4, verify_first: bool = True,
                save_acts: bool = False, out_dir: Optional[Path] = None) -> dict:
    """
    Every word-level concept, saved in the behavioural layout.

    Concepts are written as `word_<name>` so analyze.py picks them up through
    its existing "word" prefix rule and the paired behavioural-vs-word-level
    test runs with no further wiring.

    verify_first runs the fidelity check on the first concept only. One check is
    enough — it is testing the extraction code path, not the concept — and
    running it 42 times would pay the published extractor's cost on every file.

    min_pairs drops concepts too small to average over. The published set spans
    4 to 228 pairs; anything below 4 produces a direction dominated by one pair.
    """
    from .behavioral import save_concept

    names = list(concepts) if concepts else list_concepts(word_pair_dir)
    results, verification, skipped = {}, None, []

    for i, name in enumerate(names):
        try:
            pairs = load_word_pairs(word_pair_dir, name)
        except FileNotFoundError as e:
            skipped.append((name, str(e)))
            continue
        if len(pairs) < min_pairs:
            skipped.append((name, f"only {len(pairs)} pairs"))
            continue

        res = extract_word_concept(pairs, model, tokenizer, handler,
                                   device=device, batch_size=batch_size)

        if verify_first and verification is None:
            try:
                verification = verify_against_published(
                    res, pairs, model, tokenizer, handler, device=device)
                verification["concept"] = name
                status = "ok" if verification["ok"] else "MISMATCH"
                print(f"    fidelity vs published extractor: "
                      f"cos={verification['best_cosine']:+.3f} "
                      f"({verification['best_pooling']}) [{status}]")
                if "warning" in verification:
                    print(f"    {verification['warning']}")
            except Exception as e:
                verification = {"ok": None, "error": str(e)}
                print(f"    fidelity check unavailable: {e}")

        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        if out_dir is not None:
            save_concept(res, out_dir, f"word_{safe}", save_acts=save_acts)
        results[name] = {"n_pairs": res["n_pairs"], "n_layers": res["n_layers"],
                         "d": res["d"], "saved_as": f"word_{safe}"}

    if skipped:
        print(f"    skipped {len(skipped)}: "
              + ", ".join(f"{n} ({r})" for n, r in skipped[:4])
              + (" ..." if len(skipped) > 4 else ""))

    return {"concepts": results, "n_concepts": len(results),
            "verification": verification,
            "skipped": [{"concept": n, "reason": r} for n, r in skipped]}


def published_baseline_check(results_dir: str | Path, eigenvectors: torch.Tensor,
                             cumvar: np.ndarray, expected: float = -0.356,
                             tol: float = 0.05) -> dict:
    """
    Phase 0's actual reproduction test.

    Mean word-level Gini across concepts, against the published Qwen2.5-0.5B
    figure of -0.356. Measured in the language-subset basis, since that is what
    the published number used — comparing against a full-vocabulary measurement
    would fail for a reason that has nothing to do with the environment.

    A miss does not necessarily mean the environment is broken: the published
    mean is over the 27 concepts the paper selected, and this runs over whatever
    is in data/word_pairs. Report the count alongside the mean.
    """
    from .behavioral import load_concept
    from .spectral_metrics import metrics

    ginis = {}
    for p in sorted(Path(results_dir).glob("concept_word_*.pt")):
        cp = load_concept(p)
        last = cp["n_layers"] - 1
        v = cp["directions"].get(("mean_over_sequence", last))
        if v is not None:
            ginis[p.stem.replace("concept_word_", "")] = metrics(
                v, eigenvectors, cumvar)["gini"]

    if not ginis:
        return {"ok": False, "reason": "no word-level concepts found; "
                                       "run extract_all first"}
    vals = np.array(list(ginis.values()))
    mean = float(vals.mean())
    return {"ok": bool(abs(mean - expected) <= tol),
            "mean_gini": mean, "expected": expected, "tol": tol,
            "delta": float(mean - expected),
            "n_concepts": len(vals), "per_concept": ginis,
            "note": ("Published mean is over the paper's 27 selected concepts; "
                     "this is over all files present. Compare counts before "
                     "concluding the environment is wrong.")}
