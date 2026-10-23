#!/usr/bin/env bash
# Every new run the revision needs, in order, resumable.
#
#   bash scripts/run_revision.sh check        # prereg frozen? disk? GPU? data?
#   bash scripts/run_revision.sh smoke        # one model, one concept: time it first
#   bash scripts/run_revision.sh loud         # stage 1  (no extraction, probes only)
#   bash scripts/run_revision.sh loud-fv      # stage 2  (fixes the 0.64 / 0.10 problem)
#   bash scripts/run_revision.sh quiet        # stage 3  (new concepts; extraction)
#   bash scripts/run_revision.sh quiet-length # stage 4
#   bash scripts/run_revision.sh loud-mid     # stage 5  (extraction)
#   bash scripts/run_revision.sh mlp          # stage 6
#   bash scripts/run_revision.sh semantic     # exploratory semantic control (post-freeze)
#   bash scripts/run_revision.sh report       # tables, figures, paper_numbers.tex
#   bash scripts/run_revision.sh all          # 1-6 then report
#
# Run inside tmux; every stage logs to $REV/logs/ and skips results that exist,
# so a crash or a disconnect costs only the model in progress.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
REV=${REV:-results/_spectral/rev}
mkdir -p "$REV/logs"

MODELS=(Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen3-4B
        mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B
        google/gemma-2-2b)
MLP_MODELS=(Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B mistralai/Mistral-7B-v0.3
            Qwen/Qwen3-4B)
CONCEPTS=(truth_cities sycophancy
          truth_counterfact truth_magnitude truth_translation truth_cities_negated
          truth_conjunction truth_disjunction
          sycophancy_nlp sycophancy_philpapers sycophancy_political)

# Must match prereg/two_tail_prereg.json. The report checks the files against it.
LOUD_K=0.10
QUIET_K=0.005
SWEEP=(0.0025 0.005 0.01 0.02 0.05 0.10)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$REV/logs/stages.log"; }

require_frozen() {
  if ! $PY scripts/freeze_prereg.py --check; then
    echo "Freeze and commit prereg/two_tail_prereg.json before running anything:"
    echo "  $PY scripts/freeze_prereg.py && git add prereg && git commit -m 'Freeze prereg'"
    exit 1
  fi
}

stage_check() {
  $PY scripts/freeze_prereg.py --check || true
  echo; df -h . | tail -1
  echo; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
  echo; for c in "${CONCEPTS[@]}"; do
    [ -f "data/statements/$c.jsonl" ] && echo "ok   $c" || echo "MISSING data/statements/$c.jsonl (run scripts/build_datasets.py)"
  done
  echo; ls results/_spectral/syntax_claim/acts_*.pt 2>/dev/null | head -20 \
    || echo "no cached final-layer POS activations; stage 1 will re-extract"
}

stage_smoke() {
  # Primary k only: these two cells are already reported in the thesis, so the
  # smoke test reveals nothing new and can run before the freeze. It times one
  # window size; the sweep costs about (number of k values) times as much.
  log "smoke: timing one loud model and one quiet cell at the primary k"
  time $PY scripts/test_syntax_claim.py --model Qwen/Qwen2.5-0.5B \
      --out-dir "$REV/_smoke_loud" --k-frac $LOUD_K --no-figure \
      2>&1 | tee "$REV/logs/smoke_loud.log"
  time $PY scripts/test_concept_subspace.py --model Qwen/Qwen2.5-0.5B \
      --concepts truth_cities --k-frac $QUIET_K \
      --out-dir "$REV/_smoke_quiet" 2>&1 | tee "$REV/logs/smoke_quiet.log"
  echo
  echo "Check: the smoke Delta must match the old number (same activations, same"
  echo "seed; GPU LBFGS can differ in the 3rd-4th decimal). Old values:"
  echo "  loud  Qwen2.5-0.5B token_id_random            +0.0264"
  echo "  quiet Qwen2.5-0.5B truth_cities mid / final   +0.1704 / +0.0844"
  echo "If they do not match, stop: the environment changed since the thesis runs."
}

stage_loud() {
  require_frozen
  log "stage 1: loud tail, published basis, final layer, sweep + all controls"
  $PY scripts/test_syntax_claim.py --models "${MODELS[@]}" \
      --out-dir "$REV/loud_published_final" --k-frac $LOUD_K \
      --k-fracs "${SWEEP[@]}" --skip-existing --no-figure \
      2>&1 | tee -a "$REV/logs/loud_published_final.log"
}

stage_loud_fv() {
  require_frozen
  log "stage 2: loud tail, full-vocabulary basis, every model rerun"
  for m in "${MODELS[@]}"; do
    name=${m##*/}
    cov="results/$name/cov/cov_full_vocab.pt"
    if [ ! -f "$cov" ]; then echo "MISSING $cov, skipping $name"; continue; fi
    $PY scripts/test_syntax_claim.py --model "$m" --cov "$cov" \
        --out-dir "$REV/loud_fullvocab_final" --k-frac $LOUD_K \
        --skip-existing --no-figure 2>&1 | tee -a "$REV/logs/loud_fullvocab_final.log"
  done
  $PY scripts/test_syntax_claim.py --summarise --out-dir "$REV/loud_fullvocab_final"
}

stage_quiet() {
  require_frozen
  log "stage 3: quiet tail, 11 concepts, published basis, both layers, sweep"
  $PY scripts/test_concept_subspace.py --models "${MODELS[@]}" \
      --concepts "${CONCEPTS[@]}" --k-frac $QUIET_K --k-fracs "${SWEEP[@]}" \
      --layers mid final --out-dir "$REV/quiet_published" --skip-existing \
      2>&1 | tee -a "$REV/logs/quiet_published.log"
}

stage_quiet_length() {
  require_frozen
  for mode in balance residualise; do
    log "stage 4: quiet tail, length-mode $mode (primary k only)"
    $PY scripts/test_concept_subspace.py --models "${MODELS[@]}" \
        --concepts "${CONCEPTS[@]}" --k-frac $QUIET_K --layers mid final \
        --length-mode $mode --out-dir "$REV/quiet_length_$mode" --skip-existing \
        2>&1 | tee -a "$REV/logs/quiet_length_$mode.log"
  done
}

stage_loud_mid() {
  require_frozen
  log "stage 5: loud tail, mid layer (extracts new activations)"
  $PY scripts/test_syntax_claim.py --models "${MODELS[@]}" --layer mid \
      --out-dir "$REV/loud_published_mid" --k-frac $LOUD_K \
      --k-fracs "${SWEEP[@]}" --skip-existing --no-figure \
      2>&1 | tee -a "$REV/logs/loud_published_mid.log"
}

stage_mlp() {
  require_frozen
  log "stage 6: MLP probe, ${#MLP_MODELS[@]} models, primary k"
  $PY scripts/test_syntax_claim.py --models "${MLP_MODELS[@]}" --probe mlp \
      --out-dir "$REV/loud_mlp_final" --k-frac $LOUD_K --skip-existing --no-figure \
      2>&1 | tee -a "$REV/logs/loud_mlp_final.log"
  $PY scripts/test_concept_subspace.py --models "${MLP_MODELS[@]}" --probe mlp \
      --concepts "${CONCEPTS[@]}" --k-frac $QUIET_K --layers mid final \
      --out-dir "$REV/quiet_mlp" --skip-existing \
      2>&1 | tee -a "$REV/logs/quiet_mlp.log"
}

stage_semantic() {
  # Exploratory, added after the freeze: sycophancy subsets pooled, topic as a
  # semantic control. Probes only; reuses the cached quiet-tail activations.
  log "semantic: pooled sycophancy subsets, topic control"
  $PY scripts/semantic_control.py --models "${MODELS[@]}" --k-frac $QUIET_K \
      --k-fracs "${SWEEP[@]}" --out-dir "$REV/quiet_semantic" --skip-existing \
      2>&1 | tee -a "$REV/logs/quiet_semantic.log"
}

stage_report() {
  opt() { [ -d "$2" ] && echo "$1 $2"; }
  # shellcheck disable=SC2046
  $PY scripts/two_tail_report.py \
      --loud-final "$REV/loud_published_final" \
      $(opt --loud-mid "$REV/loud_published_mid") \
      --quiet "$REV/quiet_published" \
      $(opt --length-balanced "$REV/quiet_length_balance") \
      $(opt --length-residualised "$REV/quiet_length_residualise") \
      $(opt --loud-mlp "$REV/loud_mlp_final") \
      $(opt --quiet-mlp "$REV/quiet_mlp") \
      $(opt --loud-fullvocab "$REV/loud_fullvocab_final") \
      $(opt --quiet-semantic "$REV/quiet_semantic") \
      --out "$REV/report" ${FIGURE:+--figure $FIGURE} \
      2>&1 | tee "$REV/logs/report.log"
}

case "${1:-}" in
  check) stage_check ;;
  smoke) stage_smoke ;;
  loud) stage_loud ;;
  loud-fv) stage_loud_fv ;;
  quiet) stage_quiet ;;
  quiet-length) stage_quiet_length ;;
  loud-mid) stage_loud_mid ;;
  mlp) stage_mlp ;;
  semantic) stage_semantic ;;
  report) stage_report ;;
  all) stage_loud; stage_loud_fv; stage_quiet; stage_quiet_length;
       stage_loud_mid; stage_mlp; stage_report ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
