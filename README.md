# Spectral Tilt Without Specificity — thesis code (Nuraj Rimal)

**Thesis:** [`paper/thesis_revised.pdf`](paper/thesis_revised.pdf) ·
**Results:** [`results/_spectral/rev/report/report.md`](results/_spectral/rev/report/report.md) ·
**How to reproduce:** [`RUNBOOK_REVISION.md`](RUNBOOK_REVISION.md)

The thesis tests both halves of the "concepts whisper, syntax shouts" account of
transformer geometry with one instrument. Under the published measurement both
spectral tails tilt as reported, but neither tilt is specific: part of speech
prefers the high-variance head no more than a random lexical partition does, and
concepts prefer the low-variance tail no more than surface length does (inside a
pre-registered ±0.10 equivalence bound). Eight models, eleven concepts, six window
sizes, two layers, with length-matched, length-residualised and non-linear-probe
checks.

### What is in this repository

| Path | Contents |
|---|---|
| `scripts/test_syntax_claim.py` | loud tail: POS vs matched lexical controls across the spectrum |
| `scripts/test_concept_subspace.py` | quiet tail: concepts vs length control, length balancing/residualising |
| `scripts/semantic_control.py` | exploratory semantic (topic) control for the quiet tail |
| `scripts/two_tail_report.py` | every pre-registered test, Holm, TOST, figures, LaTeX numbers |
| `scripts/freeze_prereg.py`, `prereg/` | pre-registration and its SHA-256 lock |
| `scripts/run_revision.sh` | one resumable command per experiment stage |
| `tests/test_revision_pipeline.py` | end-to-end test on synthetic data (no GPU) |
| `results/_spectral/rev/report/` | figures, tables and the numbers used in the thesis |
| `paper/` | thesis LaTeX source and PDF |

```bash
python tests/test_revision_pipeline.py      # self-test, CPU only
bash scripts/run_revision.sh all            # all experiments (GPU)
make paper                                  # report, figures, paper numbers
```

This work builds on the published code for Acharya, Rimal & Dhakal (2026),
*Concepts Whisper* (`pratyush-acharya/spectral_anti_concentration`); see
`LICENSE` and `CITATION.cff`. The original pipeline documentation follows.

---

# Thesis code — spectral anti-concentration for behavioural directions

Implements the seven phases of the accepted proposal. Additive to
`pratyush-acharya/spectral_anti_concentration`: no published file is modified,
so the EMNLP artifacts stay reproducible.

## Install

```
src/linear_rep_geometry/spectral_metrics.py   metrics + TOST + paired tests
src/linear_rep_geometry/covariance.py         Phase 5 bases
src/linear_rep_geometry/behavioral.py         Phases 1, 3
src/linear_rep_geometry/nulls.py              Phase 2
src/linear_rep_geometry/dimensionality.py     Phase 4
src/linear_rep_geometry/dual_geometry.py      Phase 6
scripts/build_datasets.py
scripts/run_phases.py
scripts/analyze.py
scripts/check_env.py
test_pipeline.py
```

Add to `src/linear_rep_geometry/__init__.py`:

```python
from . import (spectral_metrics, covariance, behavioral, nulls,
               dimensionality, dual_geometry, word_level)
```

**No new dependencies.** The code imports only `torch`, `numpy` and `scipy`
directly, and all three are already pinned in the published `requirements.txt`
(`scipy>=1.17.1` covers the t-tests, Wilcoxon and TOST). `transformers`,
`accelerate` and `safetensors` come in through `lrg.load_model`, and are
likewise already pinned.

Check the cluster before the first run:

```bash
python scripts/check_env.py                  # libraries, CUDA, disk, gated access
python scripts/check_env.py --probe-network   # also contact the Hub per model
```

It exits non-zero on anything blocking, so it can gate a job script.

## Run

```bash
python scripts/build_datasets.py --clone

# Phase 0 — reproduce −0.356, build covariance caches, AND build the §6 control
python scripts/run_phases.py 0 --model Qwen/Qwen2.5-0.5B --word-pairs data/word_pairs

# Phase 2 FIRST, with --freeze, before reading any Phase 1 output
python scripts/run_phases.py 2 --model Qwen/Qwen2.5-0.5B --freeze

# Phases 1, 3–6
python scripts/run_phases.py 1 3 4 5 6 --model Qwen/Qwen2.5-0.5B

# Cross-model inference
python scripts/analyze.py --models Qwen2.5-0.5B Qwen2.5-3B Qwen2.5-7B Qwen3-4B \
    Llama-3.2-3B Llama-3.1-8B gemma-2-2b Mistral-7B-v0.3
```

`freeze_decision_rule` is write-once and records a SHA-256 and a UTC timestamp.
Commit `results/<model>/decision_rule.json` before Phase 1 output is analysed —
that commit is what makes "pre-specified" checkable by someone who was not there.

## Data

27,117 pairs across 17 concepts, from four repositories
(geometry-of-truth, refusal_direction, nrimsky/CAA MIT, anthropics/evals CC-BY-4.0). `*` marks the four truth sub-datasets named in
the proposal.

| Concept | n | Matched |
|---|---|---|
| truth_counterfact * | 15,751 | yes |
| truth_magnitude * | 1,980 | yes |
| truth_cities * | 748 | yes |
| truth_cities_negated | 748 | yes |
| truth_conjunction | 748 | yes |
| truth_disjunction | 748 | yes |
| truth_translation * | 354 | yes |
| truth_common_claim | 2,225 | no |
| truth_companies | 600 | no |
| refusal | 256 | **no** |
| refusal_advbench | 200 | **no** |
| refusal_harmbench | 159 | **no** |
| refusal_jailbreakbench | 100 | **no** |
| sycophancy + | 1,000 | yes |
| sycophancy_nlp + | 500 | yes |
| sycophancy_philpapers + | 500 | yes |
| sycophancy_political + | 500 | yes |

`+` marks the §9 optional extension. Sycophancy is built but excluded from
`--primary-only`: adding it is a scope decision for your supervisor, not a
default.

`--primary-only` restricts to the five the proposal names. `--max-pairs 512` is
the default in `run_phases.py`; counterfact at 15,751 would dominate runtime for
no gain, since direction estimates saturate well below that.

**`likely.csv` is deliberately not built.** It is in the truth repo but is not a
truth dataset — it holds non-factual text ending in a probable or improbable
token, and exists to prove truth probes are not probability probes. Building it
as a truth concept would turn the paper's control into your confound. Use it as
a Phase 2 nuisance contrast if you want it.

## Three things to know before running

**Refusal has no minimal pairs and none can be constructed from the repo.**
Harmful and harmless instructions come from different corpora and differ in
topic, register, length and syntax as well as in refusal-ness. They are emitted
unmatched, and their gap is only interpretable beside the label-shuffle null.
Truth, by contrast, pairs on the generating field — same city, true vs false
country — so it is as clean as the published word pairs. That makes **truth the
positive control for the whole pipeline**: if anti-concentration fails there,
the method is wrong, not the concept.

**The two nulls can disagree.** In a strongly anisotropic spectrum the
norm-matched random baseline is already −0.16 to −0.28, so a direction can be
null against it while significant against the label shuffle. `both_nulls`
returns an `agree` flag and the analysis prints the disagreement count.
Reporting both is required; picking the friendlier one is not available.

**Sycophancy is an answer-token contrast, and that changes the pooling rule.**
Its pairs are one question with two different answers, so they share 99.2% of
their tokens against truth's 85.7% and refusal's 0%. `prompt_overlap` measures
this per concept, and extraction prints a warning when the shared fraction
exceeds 0.9: `mean_over_sequence` dilutes a one-token difference across two
hundred positions and measures mostly noise. Report it, but do not require the
conclusion to hold under it — and say so in the write-up rather than quietly
reporting the pooling that worked.

That tightness is also why sycophancy is worth having. It is output-control
like refusal but minimally paired like truth, so it separates the
behavioural-versus-semantic axis from the pairing-quality confound that
otherwise rides along with refusal. `analyze.py` runs it as a third paired
contrast.

**Phase 4 runs before any headline is read.** If a contrast is not rank one, the
scalar Gini of a mean vector describes one arbitrary vector inside a subspace.
`rank_report` decides; where it fails, `subspace_metrics` measures the subspace
instead and the headline changes shape.

## Deviations from the proposal, and why

The language-subset basis has no principled definition for a behavioural
concept, so `full_vocab` is the primary basis and `language` is kept only for
Phase 0's reproduction check and the word-level control. This is section 5 of
the proposal, implemented as written.

The label-shuffle null uses a **per-pair sign flip** for matched data rather
than a free repartition. A free repartition breaks the pairing and gives an
optimistically wide null, which would make real directions look more special
than they are. Unmatched data uses the free repartition as described.

Both nulls are **two-sided**. H2 genuinely predicts refusal concentrates, so a
one-sided test would prejudge the result the designed contrast exists to detect.

## The word-level control (§6)

Built by Phase 0 from the 42 files already in `data/word_pairs/`. No extra data
to source. Each concept is saved as `concept_word_<name>.pt` in the same layout
as the behavioural concepts, so `analyze.py` finds them through its existing
`word` prefix rule and the paired behavioural-vs-word-level test runs with no
further wiring.

Two things this does that a call to `core.concept_direction` would not:

**Every layer.** The published extractor returns one final-layer vector, which
is right for the published analysis but cannot support Phase 3. A behavioural
direction read at mid-depth against a word direction read at the final layer is
precisely the confound Phase 3 exists to remove.

**A fidelity check.** `verify_against_published` compares the final-layer
direction against `core.concept_direction` on the first concept and reports the
cosine. The two pool differently in detail, so they are not bit-identical, but
with context-free input the target span is the sequence and they should be
nearly collinear. Below 0.95 the control is not measuring the published
quantity and the paired test would compare two different things — resolve it
rather than widening the tolerance.

Phase 0 also reports the mean word-level Gini against the published −0.356,
measured in the language-subset basis because that is what the published number
used. A miss is not automatically an environment problem: the published mean is
over the paper's 27 selected concepts and this runs over all 42 files, so
compare the counts before concluding anything.

### Still open

`--language-tokens` is unfilled. Supply the published language token subset as
a JSON list of ids for Phase 0 to reproduce `Sigma_L` exactly; without it the
`language` basis falls back to full vocabulary and the −0.356 comparison is
approximate.

## Tests

`python test_pipeline.py` — 55 checks on a synthetic model with a planted
low-variance direction. Verifies metric signs, scale and sign invariance,
subspace measurement, that the shuffle null fires on signal and stays null on
noise, that split-half separates signal from noise, that TOST declares
equivalence on a null effect and reports underpowered on an ambiguous one, that
the decision rule is write-once, and that extraction recovers a planted
direction at |cos| = 0.99.

Worth keeping as a regression test. It catches sign errors and centering bugs
that are invisible on real data — the kind that produce a clean-looking result
for the wrong reason.
