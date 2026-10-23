#!/usr/bin/env python3
"""
Does "syntax shouts" survive a matched control?

The published POS probe compares two k-dimensional subspaces of Sigma: the top-k
eigenvectors and the bottom-k. Six of eight models scored higher in the top-k and
the paper reads that as syntax living in the high-variance head.

The thesis has already shown, on the concept side, that the published effect is
measured against a null that does not hold the relevant structure fixed. The same
question applies here, and the answer is not the label-shuffle null that worked
for concepts: shuffling POS tags puts BOTH subspaces at chance, which is a floor
check, not a control. What a shuffled-label test cannot tell you is whether the
top-k advantage is about SYNTAX or about information of any kind surviving
projection into the high-variance head.

So this script holds two things fixed that the published comparison does not.

  Spectral position is swept, not dichotomised.
      Instead of top-k versus bottom-k, probes run on a sliding window of k
      consecutive eigenvectors at every decile of the spectrum. Every window has
      the same dimension and a different retained variance, which turns one
      contrast into a curve. Window 0 IS the published top-k and the last window
      IS the published bottom-k, so the published numbers are a special case and
      the reproduction is checkable.

  The label is varied, with syntax removed but lexical identity kept.
      token_id_random assigns every WORD TYPE to one of C classes at random, once,
      and keeps that assignment across all its occurrences. Class count and class
      sizes are matched to the POS distribution. It carries no syntax by
      construction, but it is exactly as recoverable from token identity as POS is.
      If POS and token_id_random trace the same curve, the asymmetry is about
      lexical information surviving projection and the word "syntax" is doing no
      work in the result.

      freq_bucket and length_bucket are reported too, with the caveat that both
      correlate with POS (determiners and prepositions are short and frequent), so
      they bound the confound rather than removing it. token_id_random is the
      clean one.

  Scores are put on a common scale before they are compared.
      Different labels have different ceilings, so raw accuracy cannot be compared
      across them. Everything is reported as retention,

          (acc_subspace - majority) / (acc_full_d - majority),

      the fraction of the label's own full-dimensional signal that survives the
      projection. Retention is what makes "POS prefers the head MORE than an
      arbitrary lexical partition does" a statement with a number attached.

The headline is a difference in differences:

      Delta = [retention_top - retention_bottom] for POS
            - [retention_top - retention_bottom] for token_id_random

  Delta > 0, CI excluding zero  -> the head carries syntax specifically; the
                                   published reading survives a matched control.
  Delta ~ 0                     -> the top-k advantage is not syntax-specific;
                                   "syntax shouts" describes the measurement, not
                                   the concept, exactly as the thesis found for
                                   "concepts whisper".

CI is a paired bootstrap over tokens, which is the right unit here because all
label sets are defined on the same token set.

Also reported, because the thesis already found that it matters: how much of the
bottom of the spectrum is numerically degenerate. Sigma is built as
(1/|L|) sum gamma gamma^T + lambda I. When lambda is large relative to the small
eigenvalues the bottom eigenvalues pile up on lambda, the eigenvectors inside that
plateau are arbitrary up to rotation, and "the bottom-k eigenvectors" names an
arbitrary subspace rather than a specific one. The lambda = 0.1 finding from the
concept side applies here unchanged.

USAGE

    # reproduce the published numbers and run the controls, one model
    python scripts/test_syntax_claim.py --model google/gemma-2-2b

    # thesis basis instead of the published language-subset basis
    python scripts/test_syntax_claim.py --model Qwen/Qwen2.5-7B \
        --cov results/Qwen2.5-7B/cov/cov_full_vocab.pt

    # several models, then the cross-model summary
    python scripts/test_syntax_claim.py --models google/gemma-2-2b \
        Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B
    python scripts/test_syntax_claim.py --summarise

    # check the machinery with no model and no GPU (plants a known answer)
    python scripts/test_syntax_claim.py --synthetic

OUTPUT

    <out-dir>/<model>.json                 (default results/_spectral/syntax_claim)
    <out-dir>/syntax_claim_summary.csv
    figures/thesis/fig_syntax_claim_<model>.pdf

ADDITIONS FOR THE REVISION (all optional; defaults reproduce the original run)

    --out-dir DIR      Write results to DIR instead of syntax_claim/. Every basis,
                       layer, probe type gets its OWN folder. The original
                       script could only write to syntax_claim/, so the
                       full-vocabulary folder had to be assembled by hand, and two
                       of its eight files were published-basis files copied in by
                       mistake. A folder now refuses a file whose basis differs
                       from the files already in it (--force overrides), and
                       --summarise refuses to aggregate a mixed folder.
    --acts-dir DIR     Where activation caches are read and written. Defaults to
                       the original syntax_claim/ folder, so new runs reuse the
                       activations already extracted instead of re-extracting
                       (and using disk) per output folder.
    --k-fracs F ...    Sweep the window size. The primary k (--k-frac) is always
                       included and is the one the top-level fields report; the
                       rest go in by_k. Activations are reused, so a sweep costs
                       probes only.
    --layer final|mid|INT
                       final (default) is the published read-out, final norm
                       applied. mid is hidden_states[len // 2], the same mid-depth
                       index test_concept_subspace.py uses, no final norm.
    --probe linear|mlp One-hidden-layer MLP on the same windows and folds, so
                       "linear probes only" can be tested rather than conceded.

    Every run now computes the difference in differences against ALL THREE
    controls (token_id_random, freq_bucket, length_bucket) and stores them in
    did_by_control. --control still picks the one in the legacy top-level field.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
SPECTRAL_DIR = RESULTS_DIR / "_spectral"
DEFAULT_OUT_DIR = SPECTRAL_DIR / "syntax_claim"
OUT_DIR = DEFAULT_OUT_DIR          # reassigned by --out-dir in main()
ACTS_DIR = DEFAULT_OUT_DIR         # reassigned by --acts-dir in main()
FIG_DIR = REPO_ROOT / "figures" / "thesis"

SCRIPT_VERSION = "2026-10-revision"
CONTROLS = ("token_id_random", "freq_bucket", "length_bucket")

# "linear" reproduces the published probe. "mlp" swaps the estimator only; the
# folds, scaler, retention and bootstrap are unchanged. Set once in main().
PROBE_KIND = "linear"

# Published settings. Changing any of these breaks comparability with the paper's
# table, so they are named here rather than buried in argparse defaults.
K_FRAC = 0.10
MAX_SENTENCES = 500
MIN_SAMPLES_PER_TAG = 30
N_FOLDS = 5
SEED = 42
DROP_TAGS = ("PUNCT", "SYM", "X")   # published filter

# None = decide from torch.cuda.is_available(), as the published script does.
# --cpu-probe pins it to False. Set once in main() rather than threaded through
# every call site, because every probe in a run must take the same path for the
# accuracies in one table to be comparable.
PROBE_GPU = None


# ─────────────────────────────────────────────────────────────────────────────
# Published code, imported rather than copied
# ─────────────────────────────────────────────────────────────────────────────

def load_published_module():
    """
    Import scripts/pos_probe_experiment.py so extraction runs on the paper's own
    code path.

    This matters more than it looks. The top-k accuracy depends on subword
    pooling, on the character-offset alignment between UD words and tokenizer
    tokens, and on whether the final norm is applied — a reimplementation that
    differs in any of those produces a number that cannot be set beside the
    published one, and the whole point of this script is that the published
    comparison appears in it as a special case.
    """
    import importlib.util
    path = SCRIPT_DIR / "pos_probe_experiment.py"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script extends the published POS probe and "
            f"reuses its extraction code; run it from inside a checkout of "
            f"spectral_anti_concentration.")
    spec = importlib.util.spec_from_file_location("pos_probe_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pos_probe_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Provenance: which basis produced a file, and refusing to mix them
# ─────────────────────────────────────────────────────────────────────────────

def file_sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def basis_kind(basis_string):
    """
    Classify a 'basis' field into the two bases the thesis uses.

    Works on files written before this field existed, which is what makes it
    possible to audit the old folders: the full-vocabulary folder held two files
    whose basis string still pointed at en-fr/cov_src.pt.
    """
    s = str(basis_string)
    if "cov_full_vocab" in s:
        return "full_vocab"
    if "cov_src" in s or "eigen_cache" in s or "cov_en" in s:
        return "published_language"
    return "other"


def basis_model(basis_string):
    """The model directory named inside a basis path, or None."""
    parts = str(basis_string).replace("cov:", "").replace("cache:", "").split("/")
    for i, p in enumerate(parts):
        if p in ("results", "eigen_cache") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def audit_folder(out_dir, verbose=True):
    """
    Problems that would make a cross-model summary of this folder wrong.

      mixed basis    files computed in different eigenbases
      wrong model    a file whose basis path names a different model
      mixed setup    different layer, probe or primary k across files

    Returns a list of strings, empty when the folder is clean.
    """
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        if p.name.startswith(("syntax_claim_summary", "acts_")):
            continue
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if "basis" not in r:
            continue
        rows.append((p.name, r))
    problems = []
    if not rows:
        return problems
    kinds = {n: r.get("basis_kind") or basis_kind(r["basis"]) for n, r in rows}
    if len(set(kinds.values())) > 1:
        problems.append("mixed basis: " + ", ".join(f"{n}={k}" for n, k in kinds.items()))
    for n, r in rows:
        bm = basis_model(r["basis"])
        if bm and bm != r.get("model_name"):
            problems.append(f"wrong model: {n} was computed in {bm}'s basis")
    for key in ("layer", "probe", "k_frac"):
        vals = {n: r.get(key, {"layer": "final", "probe": "linear"}.get(key))
                for n, r in rows}
        if len(set(map(str, vals.values()))) > 1:
            problems.append(f"mixed {key}: " + ", ".join(f"{n}={v}" for n, v in vals.items()))
    if verbose and problems:
        print(f"\n  AUDIT FAILED for {out_dir}:")
        for p in problems:
            print(f"    - {p}")
    return problems


def guard_write(out_dir, model_name, kind, force=False):
    """Refuse to put a file of one basis into a folder of another."""
    for p in Path(out_dir).glob("*.json"):
        if p.name.startswith(("syntax_claim_summary", "acts_")) or \
                p.name == f"{model_name}.json":
            continue
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if "basis" not in r:
            continue
        other = r.get("basis_kind") or basis_kind(r["basis"])
        if other != kind and not force:
            raise SystemExit(
                f"{out_dir} already holds {p.name} computed in the {other} basis; "
                f"this run is {kind}. Use a separate --out-dir per basis "
                f"(or --force if you really mean to mix them).")


# ─────────────────────────────────────────────────────────────────────────────
# Label sets
# ─────────────────────────────────────────────────────────────────────────────

def build_label_sets(words, pos_tags, n_classes, seed=SEED):
    """
    Four label vectors over the same tokens, plus the shuffled floor.

    pos               the real thing
    pos_shuffled      POS tags permuted across tokens. A floor, not a control:
                      it should sit at the majority rate everywhere. If it does
                      not, the probe is fitting something it should not and no
                      other number in the run is trustworthy.
    token_id_random   every word TYPE gets a random class, fixed across its
                      occurrences, with class sizes matched to the POS class
                      distribution. Zero syntax; same dependence on token
                      identity. This is the control the headline uses.
    freq_bucket       type frequency in this corpus, cut into n_classes bins.
    length_bucket     character length of the word, cut into n_classes bins.

    freq and length are reported but not used for the headline, because both are
    correlated with POS — function words are short and frequent — so a positive
    result on them is ambiguous in a way token_id_random is not.
    """
    rng = np.random.default_rng(seed)
    n = len(words)
    pos = np.asarray(pos_tags)
    labels = {"pos": pos.copy()}

    # Floor.
    labels["pos_shuffled"] = pos[rng.permutation(n)]

    # Class sizes to imitate, so the controls are not handed an easier problem
    # through a flatter class distribution.
    tag_share = Counter(pos_tags)
    order = [t for t, _ in tag_share.most_common()]
    target = np.array([tag_share[t] for t in order], dtype=float)
    target = target / target.sum()

    # token_id_random: partition the TYPE vocabulary, not the tokens. Assigning
    # per token would let the probe do no better than chance by construction and
    # would measure nothing; assigning per type keeps it exactly as learnable
    # from lexical identity as POS is.
    # Assignment is longest-processing-time bin packing: take the types in
    # descending frequency and give each to whichever class is currently furthest
    # below its target count, breaking ties at random.
    #
    # The obvious alternative — walk a shuffled type list and fill one class at a
    # time until its budget is spent — fails on any natural corpus. Word
    # frequencies are Zipfian, so a handful of types carry most of the tokens, and
    # the first of those to appear overshoots its class's budget and several
    # classes after it, leaving five populated classes out of thirteen and a
    # majority rate near 0.5. A control that is a far easier problem than POS is
    # not a control.
    #
    # Spreading the frequent types across classes also keeps frequency
    # uninformative about the class, which matters: if it were not, this label
    # would be a frequency probe wearing a different name and would stop being the
    # clean control.
    type_freq = Counter(w.lower() for w in words)
    types = sorted(type_freq, key=lambda t: (-type_freq[t], t))
    budget = target * n
    counts = np.zeros(n_classes)
    assign = {}
    for t in types:
        deficit = budget - counts
        best = np.flatnonzero(deficit >= deficit.max() - 1e-9)
        ci = int(rng.choice(best))
        assign[t] = ci
        counts[ci] += type_freq[t]
    labels["token_id_random"] = np.array([assign[w.lower()] for w in words])

    # freq_bucket: quantile bins of type frequency.
    freqs = np.array([type_freq[w.lower()] for w in words], dtype=float)
    labels["freq_bucket"] = _quantile_bins(freqs, n_classes, rng)

    # length_bucket: character length.
    lens = np.array([len(w) for w in words], dtype=float)
    labels["length_bucket"] = _quantile_bins(lens, n_classes, rng)

    return labels


def _quantile_bins(x, n_bins, rng):
    """
    Quantile bins with ties broken at random.

    Word length is heavily tied — a great many words are 3 characters long — and
    without jitter np.quantile collapses several bins onto one boundary and the
    probe is handed a 4-class problem labelled as 13. The jitter is small
    relative to the gap between adjacent integer lengths.
    """
    jitter = rng.normal(0, 1e-6, size=len(x))
    xj = x + jitter
    edges = np.quantile(xj, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(xj, edges)


# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_logreg(X_train, y_train, X_test, n_classes, max_iter=2000, C=1.0, lr=1.0):
    """
    Multinomial logistic regression by LBFGS on the GPU.

    Lifted from pos_probe_experiment._torch_logistic_regression_gpu, including
    the zero initialisation, the L2 term written as 0.5 * (1/C) * ||W||^2 to match
    sklearn's formulation, and the 20-step inner loop. That fidelity is the point:
    the published runs used this path whenever CUDA was available, so on a GPU
    machine this is what reproduces their table, and the sklearn path is the
    deviation rather than the reference.
    """
    import torch.nn as nn
    import torch.optim as optim

    dev = torch.device("cuda")
    Xt = torch.from_numpy(X_train).float().to(dev)
    yt = torch.from_numpy(y_train).long().to(dev)
    Xe = torch.from_numpy(X_test).float().to(dev)

    net = nn.Linear(X_train.shape[1], n_classes).to(dev)
    nn.init.zeros_(net.weight)
    nn.init.zeros_(net.bias)
    wd = 1.0 / C
    opt = optim.LBFGS(net.parameters(), lr=lr, max_iter=20,
                      line_search_fn="strong_wolfe")
    lossf = nn.CrossEntropyLoss()

    prev = float("inf")
    for _ in range(max_iter // 20):
        def closure():
            opt.zero_grad()
            loss = lossf(net(Xt), yt) + 0.5 * wd * (net.weight ** 2).sum()
            loss.backward()
            return loss
        out = opt.step(closure)
        cur = out.item() if out is not None else prev
        if abs(prev - cur) < 1e-6:
            break
        prev = cur

    with torch.inference_mode():
        pred = net(Xe).argmax(dim=1).cpu().numpy()
    del Xt, yt, Xe, net
    torch.cuda.empty_cache()
    return pred


def _mlp_probe(X_train, y_train, X_test, n_classes, seed=SEED, hidden=256,
               max_epochs=200, patience=10, batch_size=512, lr=1e-3,
               weight_decay=1e-2, dropout=0.1):
    """
    One-hidden-layer MLP, the non-linear counterpart of the published probe.

    Kept deliberately small: the question is whether a non-linear read-out
    changes WHERE in the spectrum a label is recoverable, not how high accuracy
    can be pushed. A large MLP would also start to memorise individual words,
    which is exactly the selectivity problem Hewitt and Liang's control tasks
    were built to expose.

    Early stopping uses a 10% stratified hold-out carved from the TRAINING fold
    only, so the test fold is never seen during model selection. Seeded, and the
    same seed for every cell, so two windows differ only in their input.
    """
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    dev = torch.device("cuda" if (PROBE_GPU if PROBE_GPU is not None
                                  else torch.cuda.is_available()) else "cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    strat = y_train if np.bincount(y_train).min() >= 2 else None
    tr, va = train_test_split(np.arange(len(y_train)), test_size=0.1,
                              random_state=seed, stratify=strat)
    Xt = torch.from_numpy(X_train[tr]).float().to(dev)
    yt = torch.from_numpy(y_train[tr]).long().to(dev)
    Xv = torch.from_numpy(X_train[va]).float().to(dev)
    yv = torch.from_numpy(y_train[va]).long().to(dev)
    Xe = torch.from_numpy(X_test).float().to(dev)

    net = nn.Sequential(nn.Linear(X_train.shape[1], hidden), nn.ReLU(),
                        nn.Dropout(dropout), nn.Linear(hidden, n_classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()

    best, best_state, bad = float("inf"), None, 0
    n = len(tr)
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xv), yv).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xe).argmax(dim=1).cpu().numpy()
    del Xt, yt, Xv, yv, Xe, net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def train_probe(X, y, n_splits=N_FOLDS, seed=SEED, use_gpu=None, groups=None,
                kind=None):
    """
    The published probe, with per-token correctness returned in ORIGINAL index
    order.

    Same estimator, same folds, same scaling as pos_probe_experiment.train_pos_probe,
    and the same GPU-versus-CPU switch. The one addition is the index-aligned
    correctness vector, which the published version collects internally and
    discards. It is needed here because the headline compares two different label
    sets over the same tokens, and that comparison is only paired if correctness
    can be lined up token by token.

    The run reports which path it took. The two agree to well within a point, but
    they are not bit-identical — different LBFGS implementations stop in slightly
    different places — so a table should say which produced it.

    GROUPS, and why this parameter exists.

    For POS probing the rows are independent tokens and the published
    statement-level split is correct, so groups stays None and nothing changes.

    For statement pairs it is not. The two members of a pair are near-identical
    text carrying OPPOSITE labels, so a statement-level split puts a test item's
    pair-mate in the training fold about 80% of the time with the wrong label
    attached. The probe learns the mate and answers the test item confidently
    backwards. Expected accuracy under that failure is 0.8 * 0 + 0.2 * 0.5, about
    0.1 — which is what gemma-2-2b on sycophancy produced, 0.180 on a balanced
    binary task, far BELOW chance rather than near it. That is the signature:
    sub-chance accuracy from a linear probe is leakage, not weakness.

    The damage scales with how much of a pair is shared, which prompt_overlap
    already measures: sycophancy at 99.2% is destroyed, truth at 85.7% survives
    on the strength of its direction, refusal at 0% is immune.

    It also runs the other way for a pair-level nuisance label, where both members
    share the class: there the same leakage HELPS, so a control built on pair
    identity is inflated by exactly what deflates the concept. Any comparison
    between the two is then broken at both ends at once.

    Passing groups (the pair index, once per member) keeps both members of a pair
    in the same fold and removes all of it.
    """
    from sklearn.preprocessing import StandardScaler

    if use_gpu is None:
        use_gpu = (PROBE_GPU if PROBE_GPU is not None
                   else torch.cuda.is_available())

    y = np.asarray(y)
    n_classes = len(np.unique(y))
    correct = np.zeros(len(y), dtype=np.int8)
    fold_accs = []

    if groups is None:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        # StratifiedGroupKFold keeps the class balance as even as it can while
        # respecting the groups. GroupKFold is the fallback for older sklearn; it
        # ignores stratification, which is acceptable here because the labels are
        # balanced by construction.
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed)
        except ImportError:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=np.asarray(groups))

    kind = kind or PROBE_KIND
    for tr, te in split_iter:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        if kind == "mlp":
            pred = _mlp_probe(Xtr, y[tr], Xte, n_classes, seed=seed)
        elif use_gpu:
            pred = _gpu_logreg(Xtr, y[tr], Xte, n_classes, max_iter=2000, C=1.0)
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     random_state=seed)
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
        correct[te] = (pred == y[te]).astype(np.int8)
        fold_accs.append(float((pred == y[te]).mean()))

    return {"mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "fold_accuracies": fold_accs,
            "correct": correct}


def majority_rate(y):
    """
    Majority-class rate, used as the floor instead of 1/C.

    The POS distribution is far from uniform — NOUN alone is about 18% — so 1/C
    understates what a probe gets for free and inflates every retention figure
    computed against it.
    """
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# A label whose full-dimensional probe beats its majority floor by less than this
# has no signal to retain, and dividing by that denominator turns probe noise into
# a large retention number. The shuffled floor is deliberately such a label, so
# this is not an edge case: it fires every run.
MIN_HEADROOM = 0.02


def retention(acc, acc_full, floor):
    """
    Fraction of a label's own full-dimensional signal that survives a projection.

    Returns NaN when the label has no full-dimensional signal to begin with.
    Reporting a ratio there would be reporting noise divided by noise, which is
    how a floor condition ends up looking like the strongest effect in the table.
    """
    denom = acc_full - floor
    return float((acc - floor) / denom) if denom > MIN_HEADROOM else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Subspaces
# ─────────────────────────────────────────────────────────────────────────────

def spectral_windows(d, k, n_windows=10):
    """
    Sliding windows of k consecutive eigenvectors, evenly spaced over the spectrum.

    Every window has the same dimension, so dimension is held fixed and only
    spectral position moves. The first window is exactly the published top-k and
    the last is exactly the published bottom-k; the rest fill in what the
    dichotomy hides. If accuracy declines smoothly with position, that is a
    variance story. If the first window stands apart from a flat remainder, the
    head is special and the published framing survives.
    """
    starts = np.linspace(0, d - k, n_windows).round().astype(int)
    out = []
    for i, s in enumerate(starts):
        name = ("top_k" if i == 0 else
                "bottom_k" if i == len(starts) - 1 else
                f"window_{int(round(100 * s / (d - k)))}pct")
        out.append((name, list(range(int(s), int(s) + k))))
    return out


def random_window(d, k, seed=SEED, n_trials=5):
    """The published random-k baseline: k eigenvector indices drawn uniformly."""
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(d, size=k, replace=False).tolist())
            for _ in range(n_trials)]


def degeneracy_report(eigenvalues, k):
    """
    How arbitrary is "the bottom-k eigenvectors"?

    Sigma = (1/|L|) sum gamma gamma^T + lambda I. Where the data term is small
    against lambda the eigenvalues flatten onto lambda, and inside a flat plateau
    the eigenvectors are determined by numerical noise rather than by the matrix:
    any rotation within the plateau is an equally valid eigenbasis. If the bottom
    k eigenvalues sit inside such a plateau, the bottom-k subspace is not a
    reproducible object and the published contrast is comparing a specific
    subspace against an arbitrary one.

    The thesis found this concretely: at lambda = 0.1 the spectrum flattened to a
    condition number of 1.38 and the random-direction Gini collapsed to -0.0004.
    Same matrix, same problem.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    d = len(ev)
    lo = ev[-k:]
    floor = float(ev[-1])
    plateau = int((ev <= floor * 1.01).sum())
    return {
        "d": d,
        "lambda_min": floor,
        "lambda_max": float(ev[0]),
        "condition_number": float(ev[0] / max(floor, 1e-30)),
        "bottom_k_spread": float(lo.max() / max(lo.min(), 1e-30)),
        "n_within_1pct_of_floor": plateau,
        "bottom_k_inside_plateau": bool(plateau >= k),
        "note": ("bottom-k sits inside a numerically degenerate plateau; the "
                 "eigenvectors there are arbitrary up to rotation and the "
                 "bottom-k subspace is not reproducible"
                 if plateau >= k else
                 "bottom-k eigenvalues are separated; the subspace is well defined"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The headline test
# ─────────────────────────────────────────────────────────────────────────────

def difference_in_differences(cells, labels_y, full_acc, floors,
                              control="token_id_random",
                              n_boot=10_000, seed=SEED):
    """
    Does POS prefer the high-variance head MORE than an arbitrary lexical
    partition does?

        Delta = [ret_top - ret_bottom]_POS - [ret_top - ret_bottom]_control

    Paired bootstrap over tokens: one resample of token indices is applied to all
    four correctness vectors at once, so the resampling respects the fact that
    the same tokens carry all the labels. Resampling labels independently would
    widen the interval for no reason and would hide a real difference.

    The retention denominators are held at their point estimates rather than
    recomputed inside each resample. Recomputing them makes the ratio unstable
    when a full-d accuracy lands near its floor, and nothing here depends on the
    denominators' sampling variability.
    """
    rng = np.random.default_rng(seed)

    def ret_vec(label, sub):
        c = cells[(label, sub)]["correct"]
        return c, floors[label], full_acc[label]

    try:
        pt, f_p, fa_p = ret_vec("pos", "top_k")
        pb, _, _ = ret_vec("pos", "bottom_k")
        ct, f_c, fa_c = ret_vec(control, "top_k")
        cb, _, _ = ret_vec(control, "bottom_k")
    except KeyError:
        return {"available": False,
                "reason": f"need top_k and bottom_k cells for pos and {control}"}

    n = len(pt)
    dp = (fa_p - f_p)
    dc = (fa_c - f_c)
    if dp <= 1e-9 or dc <= 1e-9:
        return {"available": False,
                "reason": "a full-dimensional probe did not beat its majority "
                          "floor, so retention is undefined"}

    def delta_of(idx):
        gp = (pt[idx].mean() - pb[idx].mean()) / dp
        gc = (ct[idx].mean() - cb[idx].mean()) / dc
        return gp - gc, gp, gc

    obs, gp_obs, gc_obs = delta_of(np.arange(n))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = delta_of(idx)[0]

    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p for H0: Delta = 0, by how far zero sits in the tail.
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / n_boot)))

    return {
        "available": True,
        "control": control,
        "retention_gap_pos": float(gp_obs),
        "retention_gap_control": float(gc_obs),
        "delta": float(obs),
        "ci95": [float(lo), float(hi)],
        "p_boot": p,
        "n_tokens": int(n),
        "verdict": (
            "syntax-specific: POS prefers the high-variance head more than an "
            "arbitrary lexical partition does"
            if lo > 0 else
            "reversed: the arbitrary partition prefers the head MORE than POS does"
            if hi < 0 else
            "not syntax-specific at this n: the top-k advantage is the same size "
            "for POS and for a label with no syntax in it"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def eigen_source_path(src):
    """The file behind a load_eigen() source string, for hashing."""
    s = str(src)
    for prefix in ("cov:", "cache:"):
        if s.startswith(prefix):
            return Path(s[len(prefix):])
    return None


def load_eigen(model_name, lang_pair, cov_path=None):
    """
    Eigendecomposition, from the published cache by default.

    --cov points this at the thesis's full-vocabulary covariance instead. The two
    bases answer different questions: the published cache is what the paper's
    numbers were computed in, so it is what the reproduction gate needs, while
    full_vocab is what the thesis used everywhere else. Run both if the answer
    differs between them; that difference is a result.
    """
    if cov_path:
        cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx], f"cov:{cov_path}"

    cache = SPECTRAL_DIR / "eigen_cache" / model_name / lang_pair / "eigen.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu", weights_only=True)
        return d["eigenvalues"], d["eigenvectors"], f"cache:{cache}"

    for cand in (RESULTS_DIR / model_name / lang_pair / "cov_src.pt",
                 RESULTS_DIR / model_name / "cov_en.pt",
                 RESULTS_DIR / model_name / "cov" / "cov_full_vocab.pt"):
        if cand.exists():
            cov = torch.load(cand, map_location="cpu", weights_only=True).float()
            evals, evecs = torch.linalg.eigh(cov)
            idx = torch.argsort(evals, descending=True)
            return evals[idx], evecs[:, idx], f"cov:{cand}"

    raise SystemExit(
        f"no covariance or eigen cache for {model_name}. Run the published "
        f"pipeline first, or pass --cov with a path.")


def check_reproduction(model_name, cells, published_json=None):
    """
    Does window 0 reproduce the published top-k accuracy?

    If it does not, nothing downstream means anything: the controls would be
    measured against a pipeline that is not the published one, and a null result
    could be an extraction bug rather than a finding. This is the same gate the
    thesis used on the concept side, where the published -0.356 had to come back
    before any new number was read.
    """
    path = Path(published_json or
                SPECTRAL_DIR / "pos_probe" / "pos_probe_all_models.json")
    if not path.exists():
        return {"checked": False, "reason": f"{path} not present"}
    try:
        rows = json.load(open(path))
        row = next(r for r in rows if r.get("model_name") == model_name)
    except (StopIteration, json.JSONDecodeError, KeyError):
        return {"checked": False, "reason": f"{model_name} not in {path.name}"}

    out = {"checked": True}
    for key in ("top_k", "bottom_k"):
        got = cells.get(("pos", key))
        want = row["probes"].get(key, {}).get("mean_accuracy")
        if got is None or want is None:
            continue
        out[key] = {"published": float(want), "here": got["mean_accuracy"],
                    "delta": got["mean_accuracy"] - float(want)}
    deltas = [v["delta"] for k, v in out.items() if isinstance(v, dict)]
    out["max_abs_delta"] = float(max(abs(d) for d in deltas)) if deltas else None
    # 0.02 is wide enough for the GPU-LBFGS versus sklearn-LBFGS difference and
    # narrow enough to catch a changed token alignment, which moves accuracy by
    # far more than two points.
    out["ok"] = bool(out["max_abs_delta"] is not None and out["max_abs_delta"] <= 0.02)
    if not out["ok"] and out["max_abs_delta"] is not None:
        out["warning"] = (
            f"window 0 is {out['max_abs_delta']:.3f} from the published top-k. "
            f"The extraction path differs from the paper's; resolve that before "
            f"reading the controls.")
    return out


def probe_all_windows(pub, acts, evecs, ev, Y, full_acc, floors, k, n_windows,
                      verbose=True):
    """Every (window, label) cell at one window size k."""
    d = evecs.shape[0]
    windows = spectral_windows(d, k, n_windows)
    cells, table = {}, []
    for wname, idx in windows:
        X = pub.project_activations(acts, evecs, idx)
        var = float(ev[idx].sum() / ev.sum())
        for name, y in Y.items():
            r = train_probe(X, y)
            cells[(name, wname)] = r
            table.append({"subspace": wname, "label": name,
                          "start": int(idx[0]), "retained_variance": var,
                          "accuracy": r["mean_accuracy"],
                          "retention": retention(r["mean_accuracy"],
                                                 full_acc[name], floors[name])})
        if verbose:
            row = {t["label"]: t["retention"] for t in table if t["subspace"] == wname}
            print(f"    {wname:<16} var={var:7.2%}  " +
                  "  ".join(f"{n}={row[n]:+.3f}" for n in ("pos", "token_id_random")
                            if n in row))
    return cells, table


def did_all_controls(cells, Y, full_acc, floors, n_boot):
    out = {}
    for c in CONTROLS:
        if c in Y:
            out[c] = difference_in_differences(cells, Y, full_acc, floors,
                                               control=c, n_boot=n_boot)
    return out


def run_model(model_path, args, pub):
    model_name = model_path.split("/")[-1]
    print(f"\n{'=' * 70}\n  syntax claim: {model_name}  "
          f"[layer={args.layer} probe={PROBE_KIND} out={OUT_DIR}]\n{'=' * 70}")

    print("\n[0] UD English-EWT")
    ud = args.ud_path or pub.download_ud_english()
    sentences = pub.parse_conllu(ud)
    print(f"    {len(sentences)} sentences")

    print("\n[1] spectrum")
    evals, evecs, src = load_eigen(model_name, args.lang_pair, args.cov)
    kind = basis_kind(src)
    guard_write(OUT_DIR, model_name, kind, force=args.force)
    src_path = eigen_source_path(src)
    src_sha = file_sha256(src_path) if src_path and src_path.exists() else None
    d = len(evals)
    k = max(1, int(d * args.k_frac))
    ev = evals.numpy()
    cumvar = np.cumsum(ev) / ev.sum()
    deg = degeneracy_report(ev, k)
    print(f"    {src}  [{kind}]")
    print(f"    d={d}  k={k}  top-k var={cumvar[k-1]:.1%}  "
          f"bottom-k var={1 - cumvar[d-k]:.2%}")
    print(f"    cond={deg['condition_number']:.3g}  "
          f"{deg['n_within_1pct_of_floor']} eigenvalues within 1% of the floor")
    if deg["bottom_k_inside_plateau"]:
        print(f"    WARNING: {deg['note']}")

    print("\n[2] activations")
    acts, tags, words = cached_activations(pub, model_path, args, sentences)
    print(f"    {len(tags)} tokens")

    keep = {t for t, c in Counter(tags).items() if c >= MIN_SAMPLES_PER_TAG}
    mask = np.array([t in keep for t in tags])
    acts, tags = acts[mask], [t for t, m in zip(tags, mask) if m]
    words = [w for w, m in zip(words, mask) if m]
    n_classes = len(keep)
    print(f"    {len(tags)} after filtering, {n_classes} tags")

    print("\n[3] labels")
    labels = build_label_sets(words, tags, n_classes)
    from sklearn.preprocessing import LabelEncoder
    Y = {name: LabelEncoder().fit_transform(v) for name, v in labels.items()}
    floors = {name: majority_rate(y) for name, y in Y.items()}
    for name, y in Y.items():
        print(f"    {name:<18} {len(np.unique(y)):>3} classes  "
              f"majority {floors[name]:.3f}")

    print("\n[4] full-dimensional ceilings")
    # detach, not just cpu: the final norm is applied outside the inference_mode
    # block (as it is in the published extractor), so these tensors arrive
    # carrying a grad graph and numpy() refuses them.
    Xfull = acts.detach().cpu().float().numpy()
    full_acc = {}
    for name, y in Y.items():
        r = train_probe(Xfull, y)
        full_acc[name] = r["mean_accuracy"]
        head = r["mean_accuracy"] - floors[name]
        flag = ""
        if name != "pos_shuffled" and head <= MIN_HEADROOM:
            flag = "  <- no full-d signal; retention undefined for this label"
        print(f"    {name:<18} {r['mean_accuracy']:.4f}  "
              f"headroom {head:+.4f}{flag}")
    if full_acc[args.control] - floors[args.control] <= MIN_HEADROOM:
        print(f"\n    The control label '{args.control}' has no full-dimensional "
              f"signal, so the headline test has no denominator. Either the "
              f"corpus is too small for a type-level partition to be learnable, "
              f"or --control should be a different label. Try --max-sentences "
              f"2000, or --control freq_bucket and read it with the correlation "
              f"caveat.")

    print(f"\n[5] {args.n_windows} windows x {len(Y)} labels  (primary k_frac "
          f"{args.k_frac})")
    cells, table = probe_all_windows(pub, acts, evecs, ev, Y, full_acc, floors,
                                     k, args.n_windows)

    print("\n[6] random-k (published baseline)")
    rnd = {name: [] for name in Y}
    for ridx in random_window(d, k, SEED, args.n_random):
        X = pub.project_activations(acts, evecs, ridx)
        for name, y in Y.items():
            rnd[name].append(train_probe(X, y)["mean_accuracy"])
    random_k = {name: {"mean_accuracy": float(np.mean(v)),
                       "std_accuracy": float(np.std(v)),
                       "retention": retention(float(np.mean(v)),
                                              full_acc[name], floors[name])}
                for name, v in rnd.items()}
    print(f"    pos {random_k['pos']['mean_accuracy']:.4f}   "
          f"token_id_random {random_k['token_id_random']['mean_accuracy']:.4f}")

    print("\n[7] reproduction gate")
    # The published numbers exist only for the published basis, the final layer
    # and the linear probe; anywhere else the gate has nothing to compare against.
    if kind == "published_language" and args.layer == "final" \
            and PROBE_KIND == "linear" and abs(args.k_frac - K_FRAC) < 1e-9:
        repro = check_reproduction(model_name, cells, args.published_json)
    else:
        repro = {"checked": False,
                 "reason": "gate applies only to published basis, final layer, "
                           "linear probe, k_frac 0.10"}
    if repro.get("checked"):
        for key in ("top_k", "bottom_k"):
            if key in repro:
                v = repro[key]
                print(f"    {key:<10} published {v['published']:.4f}  "
                      f"here {v['here']:.4f}  delta {v['delta']:+.4f}")
        print(f"    {'OK' if repro['ok'] else 'MISMATCH'}")
        if "warning" in repro:
            print(f"    {repro['warning']}")
    else:
        print(f"    skipped: {repro['reason']}")

    print("\n[8] floor check")
    sh = cells[("pos_shuffled", "top_k")]["mean_accuracy"]
    floor_ok = abs(sh - floors["pos_shuffled"]) < 0.03
    print(f"    shuffled POS in top-k: {sh:.4f} vs majority "
          f"{floors['pos_shuffled']:.4f}  [{'ok' if floor_ok else 'SUSPECT'}]")
    if not floor_ok:
        print("    the probe beats chance on permuted labels; something is "
              "leaking across folds and no other number here is safe")

    print("\n[9] difference in differences, every control")
    did_by_control = did_all_controls(cells, Y, full_acc, floors, args.n_boot)
    for c, did in did_by_control.items():
        mark = "  <- legacy headline" if c == args.control else ""
        if did["available"]:
            print(f"    {c:<16} Delta = {did['delta']:+.4f}  "
                  f"95% CI [{did['ci95'][0]:+.4f}, {did['ci95'][1]:+.4f}]  "
                  f"p = {did['p_boot']:.4f}{mark}")
        else:
            print(f"    {c:<16} unavailable: {did['reason']}{mark}")
    did = did_by_control[args.control]

    # Window-size sweep. The primary k is stored once, at the top level, and
    # repeated in by_k so a reader of by_k sees the whole curve in one place.
    by_k = {f"{args.k_frac:g}": {
        "k": k, "top_k_variance_captured": float(cumvar[k - 1]),
        "bottom_k_variance_captured": float(1 - cumvar[d - k]),
        "cells": table, "did_by_control": did_by_control}}
    for kf in args.k_fracs or []:
        if abs(kf - args.k_frac) < 1e-12:
            continue
        kk = max(1, int(d * kf))
        print(f"\n[10] sweep k_frac={kf:g}  (k={kk})")
        c_k, t_k = probe_all_windows(pub, acts, evecs, ev, Y, full_acc, floors,
                                     kk, args.n_windows, verbose=False)
        d_k = did_all_controls(c_k, Y, full_acc, floors, args.n_boot)
        for c, dd in d_k.items():
            if dd["available"]:
                print(f"    {c:<16} Delta = {dd['delta']:+.4f}  "
                      f"[{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  "
                      f"p = {dd['p_boot']:.4f}")
        by_k[f"{kf:g}"] = {
            "k": kk, "top_k_variance_captured": float(cumvar[kk - 1]),
            "bottom_k_variance_captured": float(1 - cumvar[d - kk]),
            "cells": t_k, "did_by_control": d_k}

    import datetime
    out = {
        "model": model_path, "model_name": model_name,
        "basis": src, "basis_kind": kind, "basis_sha256": src_sha,
        "layer": args.layer, "probe": PROBE_KIND,
        "d": d, "k": k, "k_frac": args.k_frac,
        "top_k_variance_captured": float(cumvar[k - 1]),
        "bottom_k_variance_captured": float(1 - cumvar[d - k]),
        "n_tokens": int(len(tags)), "n_classes": n_classes,
        "probe_backend": ("torch_mlp" if PROBE_KIND == "mlp" else
                          "torch_gpu_lbfgs" if PROBE_GPU else "sklearn_lbfgs"),
        "majority_rates": floors, "full_accuracy": full_acc,
        "cells": table, "random_k": random_k,
        "degeneracy": deg, "reproduction": repro,
        "floor_check": {"shuffled_top_k": float(sh),
                        "majority": floors["pos_shuffled"], "ok": bool(floor_ok)},
        "difference_in_differences": did,
        "did_by_control": did_by_control,
        "by_k": by_k,
        "script_version": SCRIPT_VERSION,
        "argv": sys.argv,
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model_name}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n    wrote {path}")

    if not args.no_figure:
        make_figure(out)
    return out


def _hidden_index(n_hidden_states, layer):
    """
    hidden_states index for a layer spec.

    'mid' is n // 2 over the n = num_hidden_layers + 1 hidden states, the same
    rule test_concept_subspace.resolve_layers uses, so 'mid' names the same depth
    in both experiments.
    """
    if layer == "final":
        return n_hidden_states - 1
    if layer == "mid":
        return n_hidden_states // 2
    li = int(layer)
    return li if li >= 0 else n_hidden_states + li


def extract_with_words(pub, model, tokenizer, handler, sentences, device,
                       max_sentences, layer="final"):
    """
    The published extractor, with the surface word kept alongside the tag.

    pub.extract_token_activations returns activations and tags only. The controls
    need the word itself — token_id_random partitions the type vocabulary and
    length_bucket measures characters — so the loop is repeated here with the word
    retained. Everything else is the published logic line for line: same offset
    alignment, same subword mean-pooling, same tag filter, same final norm.

    layer='final' is the published read-out. Any other layer is read WITHOUT the
    final norm, since that norm belongs to the output head and not to the
    residual stream at mid-depth.
    """
    from tqdm import tqdm
    acts, tags, words_out = [], [], []
    if max_sentences:
        sentences = sentences[:max_sentences]

    for sent in tqdm(sentences, desc="  extracting"):
        words = [w for w, _ in sent]
        sent_tags = [t for _, t in sent]
        text = " ".join(words)

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256, return_offsets_mapping=True,
                        add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            o = model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device),
                      output_hidden_states=True)
        hi = _hidden_index(len(o.hidden_states), layer)
        if layer == "final":
            h = handler.apply_final_norm(model, o.hidden_states[-1])[0].cpu().float()
        else:
            h = o.hidden_states[hi][0].cpu().float()

        spans, pos = [], 0
        for w in words:
            s = text.find(w, pos)
            s = pos if s == -1 else s
            spans.append((s, s + len(w)))
            pos = s + len(w)

        for wi, (ws, we) in enumerate(spans):
            if sent_tags[wi] in DROP_TAGS:
                continue
            ti = [j for j, (a, b) in enumerate(offsets)
                  if not (a == b == 0) and a < we and b > ws]
            if not ti:
                continue
            acts.append(h[ti].mean(dim=0))
            tags.append(sent_tags[wi])
            words_out.append(words[wi])

    out = torch.stack(acts).detach().cpu().float() if acts else torch.zeros(0)
    return out, tags, words_out


def cached_activations(pub, model_path, args, sentences):
    """
    Extract once per model, reuse thereafter.

    Extraction needs the model on the GPU and takes minutes; everything after it
    is cheap numpy. Without a cache, any mistake in the probing or reporting code
    costs a full re-extraction, which is how the concept side lost an afternoon.
    The cache is keyed on the settings that change the activations — model,
    sentence count and layer — so it cannot silently serve the wrong thing. The
    final-layer name is unchanged from the original script, so caches extracted
    before this revision are found and reused.

    Caches live in ACTS_DIR (--acts-dir), not in the output folder: activations
    do not depend on the basis or the probe, so one copy serves every run.

    --refresh-acts forces re-extraction.
    """
    name = model_path.split("/")[-1]
    suffix = "" if args.layer == "final" else f"_L{args.layer}"
    path = ACTS_DIR / f"acts_{name}_{args.max_sentences}sent{suffix}.pt"

    if path.exists() and not args.refresh_acts:
        d = torch.load(path, map_location="cpu", weights_only=False)
        print(f"    reusing {path}  ({len(d['tags'])} tokens)")
        print(f"    pass --refresh-acts to re-extract")
        return d["acts"], d["tags"], d["words"]

    import linear_rep_geometry as lrg
    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    if args.require_gpu and device.type != "cuda":
        raise SystemExit(f"model landed on {device}, not cuda. Pass --allow-cpu "
                         f"if that is intended; it will take hours.")
    print(f"    device {device}")
    acts, tags, words = extract_with_words(pub, model, tokenizer, handler,
                                           sentences, device, args.max_sentences,
                                           layer=args.layer)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": acts, "tags": tags, "words": words,
                "model": model_path, "max_sentences": args.max_sentences,
                "layer": args.layer}, path)
    print(f"    cached -> {path}")
    return acts, tags, words


# ─────────────────────────────────────────────────────────────────────────────
# Figure and summary
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(out):
    """
    Retention against spectral position, one line per label.

    Parallel lines are the whole finding: if POS and an arbitrary lexical
    partition fall off at the same rate, the top-k advantage is not about syntax.
    A POS line that sits above the others ONLY at window 0 is the opposite result
    and supports the published reading.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order, seen = [], set()
    for c in out["cells"]:
        if c["subspace"] not in seen:
            seen.add(c["subspace"])
            order.append(c["subspace"])

    style = {"pos": ("o-", "#1f77b4", "POS (real)"),
             "token_id_random": ("s-", "#d62728", "random lexical partition"),
             "freq_bucket": ("^--", "#2ca02c", "type frequency"),
             "length_bucket": ("v--", "#9467bd", "word length"),
             "pos_shuffled": ("x:", "#7f7f7f", "POS shuffled (floor)")}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(order))
    for name, (mk, col, lab) in style.items():
        ys = [next((c["retention"] for c in out["cells"]
                    if c["subspace"] == s and c["label"] == name), np.nan)
              for s in order]
        ax.plot(x, ys, mk, color=col, label=lab, linewidth=1.8, markersize=5)

    ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["top" if s == "top_k" else
                        "bottom" if s == "bottom_k" else
                        s.replace("window_", "").replace("pct", "")
                        for s in order], fontsize=9)
    ax.set_xlabel("spectral position of the k-dimensional window "
                  "(high variance to low)")
    ax.set_ylabel("retention of full-$d$ signal")
    did = out.get("difference_in_differences", {})
    sub = (f"$\\Delta$ = {did['delta']:+.3f} "
           f"[{did['ci95'][0]:+.3f}, {did['ci95'][1]:+.3f}]"
           if did.get("available") else "")
    ax.set_title(f"{out['model_name']}  (d={out['d']}, k={out['k']})   {sub}",
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    # A run into a non-default folder must not overwrite the thesis figure.
    tag = "" if Path(OUT_DIR) == DEFAULT_OUT_DIR else f"__{Path(OUT_DIR).name}"
    path = FIG_DIR / f"fig_syntax_claim_{out['model_name']}{tag}.pdf"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"    wrote {path}")


def summarise(out_dir=None, allow_mixed=False):
    """
    Cross-model table, models as the unit of analysis.

    Same rule as the rest of the thesis: concepts within one model share an
    eigenbasis and are not independent, so the one-sample test runs over models,
    not over cells.

    The folder is audited first. A folder that mixes bases, or holds a file
    computed in another model's basis, is refused: that is the failure that
    produced p = 0.10 for the full-vocabulary basis, where two of the eight files
    were published-basis files. The test is reported with and without gemma-2-2b,
    because the concept side excludes it and the two tails have to be read under
    the same rule.
    """
    out_dir = Path(out_dir or OUT_DIR)
    problems = audit_folder(out_dir)
    if problems and not allow_mixed:
        raise SystemExit(f"refusing to summarise {out_dir}: fix the folder or "
                         f"pass --allow-mixed to see the (invalid) numbers anyway")

    rows = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name.startswith(("syntax_claim_summary", "acts_")):
            continue
        r = json.load(open(p))
        if "difference_in_differences" not in r:
            continue
        did = r.get("difference_in_differences", {})
        dbc = r.get("did_by_control", {})
        row = {
            "model": r["model_name"], "basis_kind": r.get("basis_kind") or
            basis_kind(r["basis"]), "layer": r.get("layer", "final"),
            "probe": r.get("probe", "linear"), "d": r["d"], "k": r["k"],
            "top_k_var": r["top_k_variance_captured"],
            "bottom_k_var": r["bottom_k_variance_captured"],
            "acc_top_pos": next(c["accuracy"] for c in r["cells"]
                                if c["subspace"] == "top_k" and c["label"] == "pos"),
            "acc_bottom_pos": next(c["accuracy"] for c in r["cells"]
                                   if c["subspace"] == "bottom_k" and c["label"] == "pos"),
            "retention_gap_pos": did.get("retention_gap_pos"),
            "retention_gap_control": did.get("retention_gap_control"),
            "delta": did.get("delta"), "p_boot": did.get("p_boot"),
            "bottom_k_degenerate": r["degeneracy"]["bottom_k_inside_plateau"],
            "repro_ok": r.get("reproduction", {}).get("ok"),
        }
        for c in CONTROLS:
            dd = dbc.get(c, {})
            row[f"delta_{c}"] = dd.get("delta") if dd.get("available") else None
        rows.append(row)
    if not rows:
        print(f"nothing in {out_dir}")
        return

    import csv
    path = out_dir / "syntax_claim_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'model':<18}{'top':>8}{'bot':>8}{'gap_pos':>9}{'gap_ctl':>9}"
          f"{'Delta':>9}{'p':>8}")
    for r in rows:
        print(f"{r['model']:<18}{r['acc_top_pos']:>8.4f}{r['acc_bottom_pos']:>8.4f}"
              f"{_f(r['retention_gap_pos']):>9}{_f(r['retention_gap_control']):>9}"
              f"{_f(r['delta']):>9}{_f(r['p_boot'], signed=False):>8}")

    from scipy import stats
    for key in ["delta"] + [f"delta_{c}" for c in CONTROLS]:
        for label, keep in (("all models", lambda m: True),
                            ("without gemma-2-2b", lambda m: m != "gemma-2-2b")):
            g = np.array([r[key] for r in rows
                          if r.get(key) is not None and keep(r["model"])])
            if len(g) < 2:
                continue
            t, p = stats.ttest_1samp(g, 0.0)
            ci = stats.t.interval(0.95, len(g) - 1, loc=g.mean(), scale=stats.sem(g))
            print(f"\n{key:<24} {label:<20} n={len(g)}  mean = {g.mean():+.4f}  "
                  f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  t={t:.3f}  p={p:.4f}")
    n_deg = sum(1 for r in rows if r["bottom_k_degenerate"])
    if n_deg:
        print(f"\n  {n_deg}/{len(rows)} models have bottom-k inside a degenerate "
              f"plateau; for those the published contrast compares a defined "
              f"subspace against an arbitrary one.")
    print(f"\nwrote {path}")


def _f(x, nd=4, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_check(seed=SEED):
    """
    No model, no GPU, no UD. A planted answer the script must recover.

    Activations carry a label-bearing signal in a LOW-variance direction and
    nothing in the high-variance head. The high-variance directions are loud pure
    noise, which is exactly the configuration where a variance story and a syntax
    story make opposite predictions. The script passes if retention is highest at
    the bottom of the spectrum and if the shuffled-label floor stays flat.

    This catches the failure that would be invisible on real data: a sign or
    ordering error in the eigenvector sort that silently swaps top for bottom.
    """
    print("synthetic check: signal planted in LOW-variance directions\n")
    rng = np.random.default_rng(seed)
    n, d, C, k = 1500, 120, 5, 12

    scale = np.geomspace(20.0, 0.05, d)
    X = rng.standard_normal((n, d)) * scale
    y = rng.integers(0, C, size=n)
    centres = rng.standard_normal((C, 6)) * 2.0
    X[:, -6:] += centres[y]                    # signal in the quiet tail

    evals = torch.tensor(scale[::-1].copy() ** 2, dtype=torch.float32)
    evals = torch.sort(evals, descending=True).values
    evecs = torch.eye(d)
    acts = torch.tensor(X, dtype=torch.float32)

    class Pub:
        @staticmethod
        def project_activations(a, e, idx):
            return (a.float() @ e[:, idx].float()).numpy()

    floor = majority_rate(y)
    full = train_probe(X, y)["mean_accuracy"]
    ysh = y[rng.permutation(n)]
    full_sh = train_probe(X, ysh)["mean_accuracy"]
    print(f"  full-d accuracy {full:.3f}  (majority {floor:.3f})")

    floor_sh = majority_rate(ysh)
    rets, acc_sh = [], []
    for name, idx in spectral_windows(d, k, 5):
        Xp = Pub.project_activations(acts, evecs, idx)
        rets.append((name, retention(train_probe(Xp, y)["mean_accuracy"], full, floor)))
        acc_sh.append(train_probe(Xp, ysh)["mean_accuracy"])
    for (name, r), a in zip(rets, acc_sh):
        print(f"  {name:<16} retention {r:+.3f}   shuffled acc {a:.3f}")
    print(f"  shuffled full-d accuracy {full_sh:.3f} vs majority {floor_sh:.3f} "
          f"-> retention undefined for the floor, as it should be")

    best = max(rets, key=lambda t: t[1])[0]
    ok_dir = best == "bottom_k"
    # The floor is checked on ACCURACY, not retention: a label with no signal has
    # no denominator, and the earlier version of this check divided by one anyway.
    ok_floor = max(abs(a - floor_sh) for a in acc_sh) < 0.05
    deg = degeneracy_report(np.r_[np.ones(100), np.full(40, 1e-9)], 20)
    ok_deg = deg["bottom_k_inside_plateau"]

    print(f"\n  highest retention at {best:<10} "
          f"[{'ok' if ok_dir else 'FAIL — spectrum may be sorted backwards'}]")
    print(f"  shuffled labels stay near zero  "
          f"[{'ok' if ok_floor else 'FAIL — leakage across folds'}]")
    print(f"  degeneracy detector fires on a flat tail  "
          f"[{'ok' if ok_deg else 'FAIL'}]")
    passed = ok_dir and ok_floor and ok_deg
    print(f"\n  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--cov", help="covariance .pt to use instead of the published "
                                  "eigen cache (e.g. the thesis full_vocab matrix)")
    ap.add_argument("--lang-pair", default="en-fr",
                    help="published cache subdirectory; only affects which cache "
                         "is read")
    ap.add_argument("--k-frac", type=float, default=K_FRAC)
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--max-sentences", type=int, default=MAX_SENTENCES)
    ap.add_argument("--control", default="token_id_random",
                    choices=["token_id_random", "freq_bucket", "length_bucket"],
                    help="which non-syntactic label carries the headline test")
    ap.add_argument("--ud-path", help="local en_ewt-ud-dev.conllu")
    ap.add_argument("--published-json",
                    help="pos_probe_all_models.json for the reproduction gate")
    ap.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    ap.add_argument("--refresh-acts", action="store_true",
                    help="re-extract activations instead of using the cache")
    ap.add_argument("--cpu-probe", action="store_true",
                    help="force the sklearn probe even when CUDA is present")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="results folder; use one folder per basis/layer/probe")
    ap.add_argument("--acts-dir", default=str(DEFAULT_OUT_DIR),
                    help="activation cache folder (shared across runs)")
    ap.add_argument("--k-fracs", type=float, nargs="+",
                    help="window sizes to sweep in addition to --k-frac, "
                         "e.g. 0.0025 0.005 0.01 0.02 0.05 0.10")
    ap.add_argument("--layer", default="final",
                    help="final (published, final norm applied), mid, or an index")
    ap.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a folder that holds another basis")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip models whose result file already exists (resume)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="summarise a folder even if the audit fails (diagnostic)")
    ap.add_argument("--audit", nargs="+", metavar="DIR",
                    help="audit result folders for mixed bases and exit")
    ap.set_defaults(require_gpu=True)
    args = ap.parse_args()

    global PROBE_GPU, OUT_DIR, ACTS_DIR, PROBE_KIND
    OUT_DIR = Path(args.out_dir)
    ACTS_DIR = Path(args.acts_dir)
    PROBE_KIND = args.probe
    PROBE_GPU = False if args.cpu_probe else torch.cuda.is_available()
    print(f"probe: {PROBE_KIND} "
          f"({'GPU' if PROBE_GPU else 'CPU'})  out: {OUT_DIR}  acts: {ACTS_DIR}")

    if args.audit:
        bad = 0
        for d in args.audit:
            problems = audit_folder(d)
            print(f"{d}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
            bad += bool(problems)
        return 1 if bad else 0
    if args.synthetic:
        return synthetic_check()
    if args.summarise and not (args.model or args.models):
        return summarise(OUT_DIR, allow_mixed=args.allow_mixed)

    models = args.models or ([args.model] if args.model else [])
    if not models:
        ap.error("pass --model, --models, --summarise or --synthetic")

    pub = load_published_module()
    for m in models:
        if args.skip_existing and (OUT_DIR / f"{m.split('/')[-1]}.json").exists():
            print(f"\n  {m}: result exists in {OUT_DIR}, skipping (--skip-existing)")
            continue
        try:
            run_model(m, args, pub)
        except Exception as e:
            print(f"\n  {m} failed: {type(e).__name__}: {e}")
            if len(models) == 1:
                raise
    if len(models) > 1:
        summarise(OUT_DIR, allow_mixed=args.allow_mixed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
