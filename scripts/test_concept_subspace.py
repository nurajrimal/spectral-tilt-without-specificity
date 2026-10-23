#!/usr/bin/env python3
"""
The symmetric experiment: is the QUIET end concept-specific?

test_syntax_claim.py asked whether the high-variance subspace carries syntax
specifically, and found it does not — part of speech never shows a larger top-k
advantage than a label with no syntax in it. This asks the mirror question about
the other tail, and it exists because the thesis does not currently have an
answer to it.

What §4.2 measured was spectral PLACEMENT: the Gini deviation of a single
difference-of-means direction, which says where that vector's energy sits along
the spectrum. It is signed, so a high-variance outcome was detectable and simply
did not occur. What it does not measure is subspace CAPACITY — how much of the
concept a given subspace actually contains. Those are different quantities, and
the POS probe measures the second one. So the two halves of the thesis are
currently measured on incommensurable scales, and a reader is entitled to ask
whether the quiet tail holds concepts in the sense the loud tail was claimed to
hold syntax.

This script asks it the same way, with the same code. Probes run on a sliding
window of k consecutive eigenvectors at ten spectral positions, and the question
is whether the concept is recoverable from the quiet end and not the loud end.

THE CONTROL

Shuffling the concept label puts every subspace at chance, so it is a floor, not
a control — the same problem the POS side had. What is needed is a label with no
concept content that is just as recoverable from the statements.

  pair_id_random   Every PAIR is assigned to class 0 or 1 at random, and both
                   members inherit it. Kept, but NOT as the control — as a
                   LEAKAGE DETECTOR. Folds are grouped by pair, so a pair-level
                   label is unlearnable by construction: the pair is never seen
                   in training. It must therefore sit at chance. If it does not,
                   something still crosses the fold boundary and nothing else in
                   that cell can be trusted.

                   It was the intended control until the folds were fixed. It
                   only ever looked learnable because a statement-level split put
                   a test item's near-identical pair-mate in the training fold.

  length_bucket    Median split on character length. The control, by elimination.
                   Under pair-grouped folds any label that is constant within a
                   pair is unlearnable, so a usable control has to VARY within
                   the pair — and the only such property these datasets carry,
                   other than the concept itself, is surface form. Length differs
                   between the members of a pair because the differing span
                   differs in length.

                   Its weakness is real and must be stated: for some datasets
                   length correlates with the label, so it bounds the confound
                   rather than removing it.

WHY THERE IS NO CLEAN CONTROL HERE, AND WHY THAT IS NOT A BUG

Pair-grouped folds are mandatory and they close off the obvious control. Any
pair-level nuisance label is unlearnable once pairs are held out together; any
within-pair label that is not surface form would have to be a second semantic
distinction annotated on the same statements, and the datasets do not carry one.

So this script cannot deliver the clean difference-in-differences the syntax side
got. What it can deliver is the concept's own retention curve across spectral
position, which is a within-label comparison on one scale and needs no control at
all. Treat that curve as the result and the difference in differences as a
secondary with a named weakness.

For an unmatched concept such as refusal the positive and negative rows come from
different corpora, the pairing is arbitrary, and neither the grouping nor the
pair-level detector means what it means elsewhere. The script checks `matched`
and says so.

  length_bucket    Median split on character length. Reported, but weaker: for
                   several concepts length correlates with the label, so it
                   bounds the confound rather than removing it.

A LIMITATION THAT CHANGES HOW A POSITIVE RESULT READS

This control is not as clean as token_id_random was on the syntax side, and the
asymmetry is not in the script's favour.

There, both labels — part of speech and a random type partition — were plausibly
readable from the same thing, lexical identity, so the comparison was between two
tasks of one kind. Here the concept label is a semantic distinction between two
statements while pair_id_random is near-duplicate item matching. Those are
different kinds of task, and the syntax experiment gave reason to think lexical
identity is somewhat easier to read from the HIGH-variance end. If that carries
over, pair_id_random will lean loud for reasons that have nothing to do with the
concept, and Delta will be pushed positive by construction.

So the two outcomes are not equally strong:

  Delta ~ 0 or negative   Strong evidence against concept-specificity, because
                          the bias runs the other way and the effect still did
                          not appear.

  Delta > 0               Weak evidence for it. Consistent with the quiet tail
                          being concept-specific, and equally consistent with the
                          control simply preferring the loud end. Do not report
                          it as support without saying this.

Check the control's own quiet-minus-loud gap before reading Delta: the run prints
it as gap_control. A large negative gap_control means the control is leaning loud
and Delta is inflated. A gap_control near zero means the comparison is fair and
Delta can be read closer to face value.

No better control is available without new annotation — a nuisance label that is
semantic, binary, defined on these same statements, and uncorrelated with the
concept would be the right thing, and the datasets do not carry one. That is a
limitation to state, not to work around.

READING IT

Retention is (acc - majority) / (acc_full_d - majority), the fraction of the
label's own full-dimensional signal surviving the projection, which is what makes
labels with different ceilings comparable. The headline is

    Delta = [ret_bottom - ret_top] for the concept
          - [ret_bottom - ret_top] for pair_id_random

with the sign convention FLIPPED relative to the syntax script, because the
hypothesis here is about the quiet end. Delta > 0 means the concept prefers the
quiet subspace more than an arbitrary item partition does — the quiet tail is
concept-specific in the capacity sense. Delta ~ 0 means it does not, and the
quiet tail is simply where statement-level information of any kind is recoverable,
which is the capacity-side counterpart of what §4.2 found about placement.

USAGE

    # same basis as the syntax runs, so the two are comparable
    python scripts/test_concept_subspace.py --model Qwen/Qwen2.5-3B \
        --concepts truth_cities truth_counterfact sycophancy

    python scripts/test_concept_subspace.py --summarise
    python scripts/test_concept_subspace.py --synthetic

Defaults: last_token pooling, because the thesis found mean_over_sequence never
recovers distinct concepts at any layer; both mid-depth and final layer, because
behavioural directions were read mid-depth while the POS probe used the final
layer, and the comparison needs both rather than a choice made quietly.
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

RESULTS_DIR = REPO_ROOT / "results"
SPECTRAL_DIR = RESULTS_DIR / "_spectral"
DEFAULT_OUT_DIR = SPECTRAL_DIR / "concept_subspace"
OUT_DIR = DEFAULT_OUT_DIR       # reassigned by --out-dir
ACTS_DIR = DEFAULT_OUT_DIR      # reassigned by --acts-dir; shared cache
FIG_DIR = REPO_ROOT / "figures" / "thesis"
STATEMENTS = REPO_ROOT / "data" / "statements"

DEFAULT_CONCEPTS = ("truth_cities", "truth_counterfact", "sycophancy")
K_FRAC = 0.10
MAX_PAIRS = 512          # the thesis default; estimates saturate well below this
SEED = 42


def load_syntax_module():
    """
    Import test_syntax_claim.py and use its probe, retention and bootstrap.

    Not for convenience. The point of this script is to put the two tails on one
    scale, and that only holds if the same estimator, the same folds, the same
    scaler, the same retention denominator and the same bootstrap produce both
    sets of numbers. A reimplementation here would reintroduce exactly the
    incommensurability the script exists to remove.
    """
    path = SCRIPT_DIR / "test_syntax_claim.py"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script measures the quiet tail with the same "
            f"machinery the syntax script measures the loud tail, and imports it "
            f"rather than copying it.")
    spec = importlib.util.spec_from_file_location("test_syntax_claim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_syntax_claim"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────

def build_label_sets(pairs, tsc, seed=SEED):
    """
    Binary labels over 2N statements, ordered [all positives, all negatives].

    concept            1 for the positive member, 0 for the negative
    concept_shuffled   permuted. A floor: must sit at the majority rate in every
                       subspace, or the probe is leaking across folds
    pair_id_random     each pair -> 0 or 1 at random, BOTH members inherit it
    length_bucket      median split on character length

    pair_id_random is balanced by construction rather than by chance: pairs are
    split into two halves of equal size. Letting it fall where it may would
    sometimes hand the control a different majority rate from the concept, and
    retention is normalised against the majority rate, so the two would stop
    being comparable.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)

    concept = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]

    perm = rng.permutation(n)
    half = np.zeros(n, dtype=int)
    half[perm[: n // 2]] = 1
    pair_random = np.r_[half, half]          # both members share the pair's class

    texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    lens = np.array([len(t) for t in texts], dtype=float)
    lens = lens + rng.normal(0, 1e-6, size=len(lens))   # break ties
    length = (lens > np.median(lens)).astype(int)

    return {
        "concept": concept,
        "concept_shuffled": concept[rng.permutation(2 * n)],
        "pair_id_random": pair_random,
        "length_bucket": length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Length: how strongly it predicts the label, and two ways to take it out
# ─────────────────────────────────────────────────────────────────────────────

def _auc(y, score):
    """ROC AUC without sklearn's input checks; 0.5 = uninformative."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def length_features(pairs, tokenizer=None):
    """Per-statement surface length, ordered [all positives, all negatives]."""
    texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    feats = {"chars": np.array([len(t) for t in texts], dtype=float),
             "words": np.array([len(t.split()) for t in texts], dtype=float)}
    if tokenizer is not None:
        try:
            feats["tokens"] = np.array(
                [len(tokenizer(t, add_special_tokens=False)["input_ids"])
                 for t in texts], dtype=float)
        except Exception as e:                       # pragma: no cover
            print(f"    token counts unavailable: {e}")
    return feats


def length_diagnostics(pairs, concept, length_bucket, feats):
    """
    How much of the concept label is recoverable from length alone.

    Reported for every dataset so the reader can see where the length control is
    weak. AUC is the probability that a random positive is longer than a random
    negative (0.5 = length says nothing). Within-pair, the share of pairs whose
    positive member is the longer one is the cleaner number, because the folds
    hold pairs out together and only within-pair differences can leak.
    """
    n = len(pairs)
    out = {}
    for name, x in feats.items():
        auc = _auc(concept, x)
        diff = x[:n] - x[n:]
        out[name] = {"auc": auc, "abs_auc_minus_half": abs(auc - 0.5),
                     "frac_pairs_pos_longer": float((diff > 0).mean()),
                     "frac_pairs_neg_longer": float((diff < 0).mean()),
                     "frac_pairs_tied": float((diff == 0).mean())}
    agree = float((length_bucket == concept).mean())
    out["length_bucket_vs_concept_agreement"] = max(agree, 1 - agree)
    return out


def length_balanced_pairs(pairs, seed=SEED):
    """
    Indices of a pair subset in which length cannot predict the label.

    Pairs are split by which member is longer (in characters). The larger group
    is subsampled to the size of the smaller one; tied pairs are all kept. In the
    result the positive is the longer member exactly as often as the negative,
    so length carries no information about the label, yet it still varies within
    pairs and length_bucket stays learnable as a control.
    """
    rng = np.random.default_rng(seed)
    d = np.array([len(p.positive) - len(p.negative) for p in pairs])
    pos_l, neg_l, tie = (np.flatnonzero(d > 0), np.flatnonzero(d < 0),
                         np.flatnonzero(d == 0))
    m = min(len(pos_l), len(neg_l))
    keep = np.r_[rng.choice(pos_l, m, replace=False) if m else [],
                 rng.choice(neg_l, m, replace=False) if m else [], tie]
    return np.sort(keep.astype(int))


def residualise_length(acts, feats):
    """
    Remove every direction in activation space that is linearly predictable from
    surface length.

    Ordinary least squares of the activations on [1, chars, log chars, words,
    tokens], residuals kept. The fit uses no labels, so it cannot leak the
    concept across folds; it removes length information for the concept probe
    and the control alike, which is why run_cell applies it to the concept
    labels only and leaves the length control on the original activations.
    """
    cols = [np.ones(len(feats["chars"]))]
    for name in ("chars", "words", "tokens"):
        if name in feats:
            x = feats[name]
            cols.append((x - x.mean()) / (x.std() + 1e-9))
    cols.append(np.log1p(feats["chars"]) - np.log1p(feats["chars"]).mean())
    Z = np.stack(cols, axis=1)
    X = acts.numpy().astype(np.float64)
    B, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return torch.from_numpy((X - Z @ B).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────

def cached_activations(model_path, concept, pairs, layers, pooling, args):
    """
    Statement activations at the requested layers, cached per model and concept.

    Only the chosen layers are kept. layerwise_activations returns
    [n, n_layers, d] for both poolings, which for 1024 statements at d=3584 over
    29 layers is about 400 MB per pooling — and the thesis has already run out of
    disk once on exactly this kind of file. Two layers of one pooling is a few
    megabytes.
    """
    import linear_rep_geometry as lrg
    from linear_rep_geometry.behavioral import layerwise_activations

    name = model_path.split("/")[-1]
    tag = "-".join(str(l) for l in layers)
    path = ACTS_DIR / f"acts_{name}__{concept}__{pooling}__L{tag}.pt"
    if path.exists() and not args.refresh_acts:
        d = torch.load(path, map_location="cpu", weights_only=False)
        print(f"    reusing {path}")
        return d["acts"], d["n_layers"], d["d"]

    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    if args.require_gpu and device.type != "cuda":
        raise SystemExit(f"model landed on {device}, not cuda. Pass --allow-cpu "
                         f"if that is intended.")
    print(f"    device {device}, {2 * len(pairs)} statements")

    texts = [p.positive for p in pairs] + [p.negative for p in pairs]
    A = layerwise_activations(texts, model, tokenizer, handler, device,
                              batch_size=args.batch_size,
                              max_length=args.max_length)[pooling]
    n_layers, d = A.shape[1], A.shape[2]
    acts = {l: A[:, l, :].detach().cpu().float().clone() for l in layers}

    del model, A
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": acts, "n_layers": n_layers, "d": d,
                "model": model_path, "concept": concept,
                "pooling": pooling, "n_pairs": len(pairs)}, path)
    print(f"    cached -> {path.name}")
    return acts, n_layers, d


def load_eigen(model_name, args):
    """
    Spectral basis, defaulting to the published language-subset Sigma_L.

    Same default and same search order as the syntax script, so the windows in
    the two experiments are windows of the same eigenbasis. If this used a
    different basis the two tails would not be comparable and the whole exercise
    would be pointless.
    """
    if args.cov:
        path = Path(args.cov)
    else:
        for cand in (RESULTS_DIR / model_name / "en-fr" / "cov_src.pt",
                     RESULTS_DIR / model_name / "cov" / "cov_full_vocab.pt"):
            if cand.exists():
                path = cand
                break
        else:
            raise SystemExit(
                f"no covariance for {model_name}. Build Sigma_L with "
                f"scripts/build_language_cov.py, or pass --cov.")
    cov = torch.load(path, map_location="cpu", weights_only=True).float()
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx], f"cov:{path}"


# ─────────────────────────────────────────────────────────────────────────────
# One (model, concept, layer) cell
# ─────────────────────────────────────────────────────────────────────────────

def run_cell(tsc, acts, Y, evals, evecs, d, k, layer, args, groups=None,
             acts_by_label=None, ks=None):
    """
    One (model, concept, layer) cell, at the primary k and every swept k.

    acts_by_label maps a label name to the activations its probe should see. It
    is used by --length-mode residualise, where the concept is probed on
    activations with length regressed out and the length control on the original
    ones. Labels not in the map use `acts`.
    """
    acts_by_label = acts_by_label or {}
    src_of = {nm: acts_by_label.get(nm, acts) for nm in Y}
    floors = {nm: tsc.majority_rate(y) for nm, y in Y.items()}

    full_acc, headroom = {}, {}
    for nm, y in Y.items():
        r = tsc.train_probe(src_of[nm].numpy(), y, groups=groups)
        full_acc[nm] = r["mean_accuracy"]
        headroom[nm] = r["mean_accuracy"] - floors[nm]

    ev = evals.numpy()

    def at_k(kk):
        windows = tsc.spectral_windows(d, kk, args.n_windows)
        cells, table = {}, []
        for wname, idx in windows:
            U = evecs[:, idx].float()
            proj = {}
            var = float(ev[idx].sum() / ev.sum())
            for nm, y in Y.items():
                key = id(src_of[nm])
                if key not in proj:
                    proj[key] = (src_of[nm].float() @ U).numpy()
                r = tsc.train_probe(proj[key], y, groups=groups)
                cells[(nm, wname)] = r
                table.append({"layer": layer, "subspace": wname, "label": nm,
                              "start": int(idx[0]), "retained_variance": var,
                              "accuracy": r["mean_accuracy"],
                              "retention": tsc.retention(r["mean_accuracy"],
                                                         full_acc[nm], floors[nm])})
        return cells, table, quiet_did(tsc, cells, floors, full_acc, args)

    cells, table, did = at_k(k)

    by_k = {f"{args.k_frac:g}": {"k": k, "cells": table,
                                 "difference_in_differences": did}}
    for kf in ks or []:
        if abs(kf - args.k_frac) < 1e-12:
            continue
        kk = max(1, int(d * kf))
        _, t_k, d_k = at_k(kk)
        by_k[f"{kf:g}"] = {"k": kk, "cells": t_k, "difference_in_differences": d_k}

    sh = cells[("concept_shuffled", "top_k")]["mean_accuracy"]
    floor_ok = abs(sh - floors["concept_shuffled"]) < 0.05

    return {"layer": layer, "majority_rates": floors, "full_accuracy": full_acc,
            "headroom": headroom, "cells": table,
            "floor_check": {"shuffled_top_k": float(sh),
                            "majority": floors["concept_shuffled"],
                            "ok": bool(floor_ok)},
            "difference_in_differences": did,
            "by_k": by_k}


def quiet_did(tsc, cells, floors, full_acc, args):
    # The imported statistic is written for the loud-end hypothesis: it computes
    # ret(top) - ret(bottom). Here the hypothesis is about the quiet end, so the
    # subspace keys are swapped before the call and the returned Delta already
    # carries the sign convention this script documents. Reusing the function
    # rather than rewriting it keeps the bootstrap identical to the syntax side.
    swapped = {}
    for (nm, wname), v in cells.items():
        if wname == "top_k":
            swapped[(nm, "bottom_k")] = v
        elif wname == "bottom_k":
            swapped[(nm, "top_k")] = v
    renamed = {("pos" if nm == "concept" else nm, w): v
               for (nm, w), v in swapped.items()}
    f2 = {("pos" if nm == "concept" else nm): v for nm, v in floors.items()}
    a2 = {("pos" if nm == "concept" else nm): v for nm, v in full_acc.items()}

    did = tsc.difference_in_differences(renamed, None, a2, f2,
                                       control=args.control, n_boot=args.n_boot)
    if did.get("available"):
        did["retention_gap_concept_quiet_minus_loud"] = did.pop("retention_gap_pos")
        did["retention_gap_control_quiet_minus_loud"] = did.pop("retention_gap_control")
        lo, hi = did["ci95"]
        did["verdict"] = (
            "quiet tail is concept-specific: the concept prefers the low-variance "
            "subspace more than an arbitrary item partition does" if lo > 0 else
            "reversed: the arbitrary item partition prefers the quiet subspace "
            "MORE than the concept does" if hi < 0 else
            "not concept-specific at this n: the quiet-tail advantage is the same "
            "size for the concept and for a label with no concept in it")
    return did


def run_model_concept(tsc, model_path, concept, args):
    from linear_rep_geometry.behavioral import load_pairs

    name = model_path.split("/")[-1]
    path = STATEMENTS / f"{concept}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/build_datasets.py first.")

    pairs_all = load_pairs(path)[: args.max_pairs]
    matched = bool(pairs_all[0].matched)
    print(f"\n{'=' * 70}\n  {name}  /  {concept}   "
          f"[length-mode={args.length_mode} probe={tsc.PROBE_KIND} out={OUT_DIR}]"
          f"\n{'=' * 70}")
    print(f"  {len(pairs_all)} pairs, matched={matched}, "
          f"contrast={pairs_all[0].contrast_type}")

    control_ok = matched or args.control != "pair_id_random"
    if not control_ok:
        print("  NOTE: this concept is unmatched, so positives and negatives come "
              "from different corpora and a row-index pairing is arbitrary. "
              "pair_id_random is not learnable from a near-duplicate pair-mate "
              "here, so the headline control does not apply. Reporting the "
              "concept and the floor only; use --control length_bucket to get a "
              "weaker comparison, and say which one it is.")

    evals, evecs, basis = load_eigen(name, args)
    kind = tsc.basis_kind(basis)
    guard_write(name, concept, kind, args.force)
    bpath = Path(basis.replace("cov:", ""))
    bsha = tsc.file_sha256(bpath) if bpath.exists() else None
    d = evecs.shape[0]
    k = max(1, int(d * args.k_frac))
    ev = evals.numpy()
    cumvar = np.cumsum(ev) / ev.sum()
    deg = tsc.degeneracy_report(ev, k)
    print(f"  {basis}  [{kind}]")
    print(f"  d={d} k={k}  top-k var={cumvar[k-1]:.1%}  "
          f"bottom-k var={1 - cumvar[d-k]:.2%}  cond={deg['condition_number']:.3g}")
    if deg["bottom_k_inside_plateau"]:
        print(f"  WARNING: {deg['note']}")

    # 'mid' and 'final' need the depth, and the depth comes from the config
    # rather than from a loaded model: AutoConfig reads a small JSON and costs
    # nothing, where loading weights to count layers would pay for a full model
    # load per concept.
    n_layers = depth_from_config(model_path)
    roles = resolve_layers(args.layers, n_layers)
    layers = [li for _, li in roles]
    role_of = {li: role for role, li in roles}
    print(f"  {n_layers} hidden states available; probing "
          f"{', '.join(f'{r}={li}' for r, li in roles)} "
          f"(pooling {args.pooling})")

    # Activations are always extracted for the first max_pairs pairs, so the
    # cache is shared by every length mode; subsets are taken afterwards.
    acts_by_layer, n_layers, dim = cached_activations(
        model_path, concept, pairs_all, layers, args.pooling, args)
    if dim != d:
        raise SystemExit(
            f"activation width {dim} != basis width {d}. The covariance and the "
            f"model disagree; check that --cov belongs to this model.")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"  tokenizer unavailable for token counts ({type(e).__name__})")

    n_all = len(pairs_all)
    if args.length_mode == "balance":
        keep = length_balanced_pairs(pairs_all)
        print(f"  length-balanced subset: {len(keep)} of {n_all} pairs")
        if len(keep) < 50:
            raise SystemExit(f"only {len(keep)} pairs survive length balancing; "
                             f"too few to probe. Report this dataset as "
                             f"length-confounded instead.")
        rows = np.r_[keep, keep + n_all]
        pairs = [pairs_all[i] for i in keep]
        acts_by_layer = {l: a[torch.from_numpy(rows)] for l, a in acts_by_layer.items()}
    else:
        pairs = pairs_all

    labels = build_label_sets(pairs, tsc)
    Y = {nm: np.asarray(v) for nm, v in labels.items()}
    feats = length_features(pairs, tokenizer)
    ldiag = length_diagnostics(pairs, Y["concept"], Y["length_bucket"], feats)
    print(f"  length vs label: char AUC {ldiag['chars']['auc']:.3f}  "
          f"positive longer in {ldiag['chars']['frac_pairs_pos_longer']:.0%} "
          f"of pairs, tied in {ldiag['chars']['frac_pairs_tied']:.0%}")

    # Both members of a pair share a group, so they always land in the same fold.
    # Statements are ordered [all positives, all negatives], so the group index
    # repeats.
    groups = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]

    import datetime
    out = {"model": model_path, "model_name": name, "concept": concept,
           "matched": matched, "contrast_type": pairs[0].contrast_type,
           "basis": basis, "basis_kind": kind, "basis_sha256": bsha,
           "d": d, "k": k, "k_frac": args.k_frac, "k_fracs": args.k_fracs,
           "n_pairs": len(pairs), "n_pairs_before_length_mode": n_all,
           "n_statements": 2 * len(pairs),
           "pooling": args.pooling, "n_layers": n_layers,
           "probe": tsc.PROBE_KIND, "length_mode": args.length_mode,
           "length_diagnostics": ldiag,
           "top_k_variance_captured": float(cumvar[k - 1]),
           "bottom_k_variance_captured": float(1 - cumvar[d - k]),
           "degeneracy": deg, "control": args.control,
           "control_applicable": bool(control_ok),
           "probe_backend": ("torch_mlp" if tsc.PROBE_KIND == "mlp" else
                             "torch_gpu_lbfgs" if tsc.PROBE_GPU else "sklearn_lbfgs"),
           "layer_roles": {str(li): role for li, role in role_of.items()},
           "script_version": tsc.SCRIPT_VERSION, "argv": sys.argv,
           "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "by_layer": {}}

    for layer in layers:
        print(f"\n  --- {role_of[layer]} layer {layer} ---")
        A = acts_by_layer[layer]
        abl = None
        if args.length_mode == "residualise":
            Ar = residualise_length(A, feats)
            abl = {"concept": Ar, "concept_shuffled": Ar}
        r = run_cell(tsc, A, Y, evals, evecs, d, k, layer, args, groups=groups,
                     acts_by_label=abl, ks=args.k_fracs)
        leak = r["full_accuracy"].get("pair_id_random")
        if leak is not None:
            lf = r["majority_rates"]["pair_id_random"]
            status = "ok" if abs(leak - lf) < 0.05 else "LEAK"
            print(f"    leakage check: pair_id_random full-d {leak:.4f} vs "
                  f"majority {lf:.4f}  [{status}]")
            if status == "LEAK":
                print("    a pair-level label should be at chance once pairs are "
                      "held out together. It is not, so something still crosses "
                      "the fold boundary and no number in this cell is safe.")
        for nm in ("concept", args.control):
            if nm in r["headroom"]:
                flag = ("  <- no full-d signal; retention undefined"
                        if r["headroom"][nm] <= tsc.MIN_HEADROOM else "")
                print(f"    full-d {nm:<18} {r['full_accuracy'][nm]:.4f}  "
                      f"headroom {r['headroom'][nm]:+.4f}{flag}")
        for w in ("top_k", "bottom_k"):
            row = {c["label"]: c["retention"] for c in r["cells"]
                   if c["subspace"] == w}
            print(f"    {w:<10} " + "  ".join(
                f"{nm}={row[nm]:+.3f}" for nm in ("concept", args.control)
                if nm in row))
        for kf, bk in r["by_k"].items():
            did = bk["difference_in_differences"]
            if did.get("available"):
                print(f"    k_frac {kf:<7} Delta = {did['delta']:+.4f}  "
                      f"CI [{did['ci95'][0]:+.4f}, {did['ci95'][1]:+.4f}]  "
                      f"p={did['p_boot']:.4f}  gap_concept="
                      f"{did['retention_gap_concept_quiet_minus_loud']:+.4f}")
            else:
                print(f"    k_frac {kf:<7} Delta unavailable: {did.get('reason')}")
        fc = r["floor_check"]
        print(f"    floor: shuffled {fc['shuffled_top_k']:.4f} vs majority "
              f"{fc['majority']:.4f}  [{'ok' if fc['ok'] else 'SUSPECT'}]")
        out["by_layer"][str(layer)] = r

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}__{concept}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n  wrote {p}")
    return out


def guard_write(model_name, concept, kind, force=False):
    """Refuse to mix bases in one output folder (see test_syntax_claim.guard_write)."""
    for p in OUT_DIR.glob("*.json"):
        if p.name == f"{model_name}__{concept}.json":
            continue
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if "basis" not in r:
            continue
        other = r.get("basis_kind") or _basis_kind(r["basis"])
        if other != kind and not force:
            raise SystemExit(
                f"{OUT_DIR} already holds {p.name} computed in the {other} basis; "
                f"this run is {kind}. Use a separate --out-dir per basis.")


def _basis_kind(s):
    s = str(s)
    if "cov_full_vocab" in s:
        return "full_vocab"
    if "cov_src" in s or "eigen_cache" in s or "cov_en" in s:
        return "published_language"
    return "other"


def depth_from_config(model_path):
    """
    Number of hidden states, from the model config alone.

    num_hidden_layers + 1, because output_hidden_states includes the embedding
    output at index 0 — which is what layerwise_activations documents and what
    the published layer-wise control also reports. Getting that +1 wrong would
    shift every 'mid' layer by one and silently change which depth the thesis
    claims to have probed.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if getattr(cfg, attr, None):
            return int(getattr(cfg, attr)) + 1
    raise SystemExit(f"could not read a layer count from the config for "
                     f"{model_path}; pass --layers with explicit integers.")


def resolve_layers(spec, n_layers):
    """
    Return [(role, index)], keeping the ROLE alongside the resolved index.

    The role is what makes the cross-model test valid. Models differ in depth, so
    mid-depth is layer 12 in Qwen2.5-0.5B and layer 18 in Qwen2.5-3B. An earlier
    version returned bare indices and the summary grouped by them, which split
    eight models into pairs and reported t statistics from n = 2 — including a
    t of -31 that looked like the strongest result in the table and was two points
    joined by a line.
    """
    out = []
    for s in spec:
        if s == "mid":
            out.append(("mid", n_layers // 2))
        elif s == "final":
            out.append(("final", n_layers - 1))
        else:
            li = int(s)
            out.append((str(s), li if li >= 0 else n_layers + li))
    seen, uniq = set(), []
    for role, li in out:
        if li not in seen:
            seen.add(li)
            uniq.append((role, li))
    return sorted(uniq, key=lambda t: t[1])


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

REPORT_HEADROOM = 0.10


def collect_rows(out_dir):
    """One row per (model, concept, layer, k_frac) cell with a usable Delta."""
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if "by_layer" not in r:
            continue
        for layer, cell in r["by_layer"].items():
            by_k = cell.get("by_k") or {f"{r.get('k_frac', K_FRAC):g}": {
                "k": r.get("k"), "difference_in_differences":
                    cell.get("difference_in_differences", {})}}
            for kf, bk in by_k.items():
                did = bk.get("difference_in_differences", {})
                if not did.get("available"):
                    continue
                rows.append({
                    "model": r["model_name"], "concept": r["concept"],
                    "layer": int(layer),
                    "role": r.get("layer_roles", {}).get(layer, str(layer)),
                    "k_frac": float(kf), "k": bk.get("k"),
                    "basis_kind": r.get("basis_kind") or _basis_kind(r["basis"]),
                    "probe": r.get("probe", "linear"),
                    "length_mode": r.get("length_mode", "none"),
                    "matched": r["matched"],
                    "control_applicable": r.get("control_applicable", True),
                    "gap_concept": did["retention_gap_concept_quiet_minus_loud"],
                    "gap_control": did["retention_gap_control_quiet_minus_loud"],
                    "delta": did["delta"], "ci_lo": did["ci95"][0],
                    "ci_hi": did["ci95"][1], "p_boot": did["p_boot"],
                    "floor_ok": cell["floor_check"]["ok"],
                    "headroom_concept": cell["headroom"]["concept"],
                    "headroom_control": cell["headroom"].get(r.get("control", ""), 1.0),
                    "length_auc_chars": (r.get("length_diagnostics", {})
                                         .get("chars", {}).get("auc")),
                })
    return rows


def usable(row):
    """The exclusion rule, unchanged from the original summary."""
    if not row["control_applicable"]:
        return False, "control not applicable"
    if not row["floor_ok"]:
        return False, "failed the shuffled-label floor check"
    if row["headroom_concept"] < REPORT_HEADROOM:
        return False, (f"concept headroom {row['headroom_concept']:.3f} "
                       f"< {REPORT_HEADROOM}")
    return True, ""


def summarise(out_dir=None):
    """
    Cross-model inference, models as the unit of analysis.

    Averaged within a model first, across concepts, then tested across models.
    Concepts inside one model share an eigenbasis and are not independent — the
    same reason the rest of the thesis tests over models — so pooling concepts
    into one n would inflate it severalfold.

    Every test is printed four ways: with and without gemma-2-2b, and with and
    without the cell exclusions. The loud tail keeps gemma; a reader must be able
    to see that no conclusion here depends on which cells were dropped.
    """
    out_dir = Path(out_dir or OUT_DIR)
    rows = collect_rows(out_dir)
    if not rows:
        print(f"nothing usable in {out_dir}")
        return
    kinds = {r["basis_kind"] for r in rows}
    if len(kinds) > 1:
        raise SystemExit(f"{out_dir} mixes bases {kinds}; refusing to summarise.")

    import csv
    path = out_dir / "concept_subspace_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["usable", "excluded_why"])
        w.writeheader()
        for r in rows:
            ok, why = usable(r)
            w.writerow({**r, "usable": ok, "excluded_why": why})

    from scipy import stats
    for kf in sorted({r["k_frac"] for r in rows}):
        for role in sorted({r["role"] for r in rows}):
            sub = [r for r in rows if r["k_frac"] == kf and r["role"] == role]
            if not sub:
                continue
            print(f"\n=== k_frac {kf:g}  {role} layer ===")
            for excl in (True, False):
                for no_gemma in (False, True):
                    for key in ("delta", "gap_concept"):
                        per = {}
                        for r in sub:
                            if excl and not usable(r)[0]:
                                continue
                            if not r["control_applicable"]:
                                continue
                            if no_gemma and r["model"] == "gemma-2-2b":
                                continue
                            per.setdefault(r["model"], []).append(r[key])
                        g = np.array([np.mean(v) for v in per.values()])
                        if len(g) < 2:
                            continue
                        t, p = stats.ttest_1samp(g, 0.0)
                        print(f"  {key:<12} {'excl' if excl else 'all cells':<9} "
                              f"{'no gemma' if no_gemma else 'with gemma':<10} "
                              f"n={len(g)}  mean={g.mean():+.4f}  t={t:+.3f}  "
                              f"p={p:.4f}")
    n_bad = sum(1 for r in rows if not usable(r)[0])
    print(f"\nwrote {path}  ({len(rows) - n_bad} usable cells, {n_bad} excluded "
          f"under the original rule)")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_check(tsc, seed=SEED):
    """
    A planted answer, no model and no GPU needed.

    The concept is planted in LOW-variance directions and the pair identity in
    HIGH-variance ones. So the concept must show a positive Delta: it prefers the
    quiet end, the control prefers the loud end. If the script reports otherwise,
    the sign convention or the key swapping is wrong — which is the one bug here
    that would produce a confident, readable, backwards result.
    """
    print("synthetic check: concept planted QUIET, length planted LOUD,\n"
          "pair identity planted LOUD but must be unlearnable once pairs are "
          "grouped\n")
    rng = np.random.default_rng(seed)
    n_pairs, d, k = 400, 120, 12
    scale = np.geomspace(20.0, 0.05, d)

    # Each PAIR gets one item vector and both members start from it, so the two
    # members are near-duplicates — which is the property of the real data that
    # caused the bug. An earlier version of this check drew all 2N rows
    # independently and planted pair_id_random as a fixed direction, which a
    # probe can learn from other pairs without ever seeing the held-out one. That
    # tested nothing: grouping is supposed to defeat MEMORISATION, so the label
    # it is tested against must be reachable only by memorising the mate.
    item = rng.standard_normal((n_pairs, d)) * scale
    X = np.r_[item, item] + rng.standard_normal((2 * n_pairs, d)) * scale * 0.05

    concept = np.r_[np.ones(n_pairs, dtype=int), np.zeros(n_pairs, dtype=int)]
    X[:, -5:] += np.where(concept[:, None] == 1, 1.2, -1.2)       # quiet end

    # A within-pair surface property, standing in for length_bucket: it varies
    # between the two members, so grouped folds leave it learnable.
    length = rng.integers(0, 2, size=2 * n_pairs)
    X[:, 5:10] += np.where(length[:, None] == 1, 9.0, -9.0)       # loud end

    # A pair-level label with NO direction of its own. The only route to it is
    # recognising the item vector, which grouping removes.
    perm = rng.permutation(n_pairs)
    half = np.zeros(n_pairs, dtype=int)
    half[perm[: n_pairs // 2]] = 1
    pair_random = np.r_[half, half]

    evals = torch.sort(torch.tensor(scale ** 2, dtype=torch.float32),
                       descending=True).values
    evecs = torch.eye(d)
    acts = torch.tensor(X, dtype=torch.float32)
    groups = np.r_[np.arange(n_pairs), np.arange(n_pairs)]

    Y = {"concept": concept,
         "concept_shuffled": concept[rng.permutation(2 * n_pairs)],
         "pair_id_random": pair_random, "length_bucket": length}

    class Args:
        n_windows, n_boot, control, k_frac = 5, 2000, "length_bucket", 0.1
    r = run_cell(tsc, acts, Y, evals, evecs, d, k, 0, Args(), groups=groups)

    for w in ("top_k", "bottom_k"):
        row = {c["label"]: c["retention"] for c in r["cells"] if c["subspace"] == w}
        print(f"  {w:<10} concept={row['concept']:+.3f}  "
              f"length_bucket={row['length_bucket']:+.3f}")
    did = r["difference_in_differences"]
    print(f"\n  Delta = {did['delta']:+.4f}  "
          f"CI [{did['ci95'][0]:+.4f}, {did['ci95'][1]:+.4f}]  p={did['p_boot']:.4f}")
    print(f"  {did['verdict']}")

    # The same label without grouping, to show the detector is sensitive rather
    # than merely quiet. Ungrouped it should be well above chance, because the
    # mate sits in the training fold; grouped it should collapse.
    Xfull = acts.numpy()
    ungrouped = tsc.train_probe(Xfull, pair_random)["mean_accuracy"]
    leak = r["full_accuracy"]["pair_id_random"]
    leak_floor = r["majority_rates"]["pair_id_random"]
    ok_leak = abs(leak - leak_floor) < 0.06 and ungrouped > leak_floor + 0.10
    print(f"\n  pair-level label WITHOUT grouping: {ungrouped:.3f} "
          f"(should be well above {leak_floor:.3f}; this is the bug)")
    ok_sign = did["delta"] > 0 and did["ci95"][0] > 0
    ok_floor = r["floor_check"]["ok"]

    print(f"\n  pair-level label defeated by grouping: full-d {leak:.3f} vs "
          f"chance {leak_floor:.3f}  "
          f"[{'ok' if ok_leak else 'FAIL — folds are not grouped by pair'}]")
    print(f"  concept prefers the quiet end, control the loud end  "
          f"[{'ok' if ok_sign else 'FAIL — sign convention or key swap is wrong'}]")
    print(f"  shuffled labels stay at the majority rate  "
          f"[{'ok' if ok_floor else 'FAIL — leakage across folds'}]")
    passed = ok_leak and ok_sign and ok_floor
    print(f"\n  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    ap.add_argument("--cov", help="covariance .pt; default prefers the rebuilt "
                                  "Sigma_L so this matches the syntax runs")
    ap.add_argument("--layers", nargs="+", default=["mid", "final"],
                    help="'mid', 'final', or integer indices")
    ap.add_argument("--pooling", default="last_token",
                    choices=["last_token", "mean_over_sequence"])
    ap.add_argument("--control", default="length_bucket",
                    choices=["length_bucket", "pair_id_random"],
                    help="length_bucket is the only control that survives "
                         "pair-grouped folds; pair_id_random is kept as a "
                         "leakage detector and should sit at chance")
    ap.add_argument("--k-frac", type=float, default=K_FRAC)
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--refresh-acts", action="store_true")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    ap.add_argument("--cpu-probe", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="results folder; one per basis / probe / length mode")
    ap.add_argument("--acts-dir", default=str(DEFAULT_OUT_DIR),
                    help="activation cache folder, shared across runs")
    ap.add_argument("--k-fracs", type=float, nargs="+",
                    help="extra window sizes to sweep, e.g. 0.0025 0.01 0.02")
    ap.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--length-mode", default="none",
                    choices=["none", "balance", "residualise"],
                    help="balance: subset of pairs in which length cannot "
                         "predict the label; residualise: regress length out of "
                         "the activations the concept probe sees")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (model, concept) results that already exist (resume)")
    ap.set_defaults(require_gpu=True)
    args = ap.parse_args()

    global OUT_DIR, ACTS_DIR
    OUT_DIR = Path(args.out_dir)
    ACTS_DIR = Path(args.acts_dir)

    tsc = load_syntax_module()
    tsc.PROBE_GPU = False if args.cpu_probe else torch.cuda.is_available()
    tsc.PROBE_KIND = args.probe
    print(f"probe: {args.probe} ({'GPU' if tsc.PROBE_GPU else 'CPU'})  "
          f"out: {OUT_DIR}  acts: {ACTS_DIR}")

    if args.synthetic:
        return synthetic_check(tsc)
    if args.summarise and not (args.model or args.models):
        return summarise(OUT_DIR)

    models = args.models or ([args.model] if args.model else [])
    if not models:
        ap.error("pass --model, --models, --summarise or --synthetic")

    for m in models:
        for c in args.concepts:
            if args.skip_existing and \
                    (OUT_DIR / f"{m.split('/')[-1]}__{c}.json").exists():
                print(f"\n  {m} / {c}: exists in {OUT_DIR}, skipping")
                continue
            try:
                run_model_concept(tsc, m, c, args)
            except Exception as e:
                print(f"\n  {m} / {c} failed: {type(e).__name__}: {e}")
    summarise(OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
