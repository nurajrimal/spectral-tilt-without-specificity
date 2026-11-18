#!/usr/bin/env bash
# Re-run every gemma-2-2b last-token analysis after the left-padding fix (7e84a70).
#
# gemma-2-2b's tokenizer pads on the left; the old extraction read position
# (length - 1), which is right only for right padding. Every other model pads on
# the right and is untouched. This script re-runs ONLY gemma-2-2b, with the
# original settings, and never deletes anything: old files are MOVED to a backup
# folder with a manifest.
#
# Two folders are involved:
#   WORK    the working repo that holds the quiet-tail (rev/) and placement runs
#   THESIS  this repo (continent control and its confirmation)
#
#   bash scripts/rerun_gemma_padding.sh install-cap # copy the GPU-cap loader into WORK (once)
#   bash scripts/rerun_gemma_padding.sh check      # fix present? what would move? (changes nothing)
#   bash scripts/rerun_gemma_padding.sh backup     # move gemma's old caches and results aside
#   bash scripts/rerun_gemma_padding.sh placement  # phases 1-2 for gemma, then the placement table
#   bash scripts/rerun_gemma_padding.sh words      # phase 0 (word concepts), phase 2, placement table
#   bash scripts/rerun_gemma_padding.sh quiet      # quiet tail: primary + sweep, both length controls
#   bash scripts/rerun_gemma_padding.sh report     # regenerate the paper's numbers (WORK/rev/report)
#   bash scripts/rerun_gemma_padding.sh continent  # continent control, sweep, confirmation (THESIS)
#
# Run the stages in this order. BATCH=4 or GPU_GIB=2 if a stage runs out of GPU memory
# (batch size does not change results after the fix; tests/test_padding_side.py).
set -uo pipefail

THESIS=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HOME/spectral_anti_concentration}
BACKUP=${BACKUP:-$HOME/gemma_padding_backup}
BATCH=${BATCH:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Cap the GPU share of the weights (core.load_model reads this). Without it,
# device_map="auto" fills the 6 GB card and gemma's 256k-word logits do not fit.
export LRG_GPU_GIB=${GPU_GIB:-3}
PY_T="$THESIS/.venv/bin/python"; [ -x "$PY_T" ] || PY_T=python3
PY_W="$WORK/.venv/bin/python";   [ -x "$PY_W" ] || PY_W=python3
G=gemma-2-2b
GM=google/gemma-2-2b
FIX=src/linear_rep_geometry/behavioral.py
CORE=src/linear_rep_geometry/core.py
CONCEPTS=(truth_cities sycophancy truth_counterfact truth_magnitude truth_translation
          truth_cities_negated truth_conjunction truth_disjunction
          sycophancy_nlp sycophancy_philpapers sycophancy_political)
SWEEP=(0.0025 0.005 0.01 0.02 0.05 0.10)
ALL8=(Qwen2.5-0.5B Qwen2.5-3B Qwen2.5-7B Qwen3-4B Llama-3.2-3B Llama-3.1-8B gemma-2-2b Mistral-7B-v0.3)

die() { echo "STOP: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$BACKUP/rerun.log"; }

# Files that hold gemma's last-token activations or results computed from them.
gemma_files() {
  ( cd "$WORK" && {
      find results/_spectral -name "acts_${G}__*__last_token__*.pt" 2>/dev/null
      for d in quiet_published quiet_length_balance quiet_length_residualise; do
        ls results/_spectral/rev/$d/${G}__*.json 2>/dev/null; done
      ls -d results/$G/concepts 2>/dev/null
      ls results/$G/phase1.json results/$G/phase2_full_vocab.json 2>/dev/null
    } | sed "s|^|WORK/|" )
  ( cd "$THESIS" && {
      find results/_spectral/concept_subspace -name "acts_${G}__*__last_token__*.pt" 2>/dev/null
      for d in continent_control continent_ksweep continent_confirm; do
        ls results/_spectral/$d/${G}__*.json 2>/dev/null; done
    } | sed "s|^|THESIS/|" )
}

check_fix() {
  for root in "$THESIS" "$WORK"; do
    grep -q "last_idx = mask.shape\[1\] - 1 - torch.flip" "$root/$FIX" \
      || die "padding fix missing in $root/$FIX"
  done
  for root in "$THESIS" "$WORK"; do
    grep -q "LRG_GPU_GIB" "$root/$CORE" || die "GPU cap missing in $root/$CORE (run: install-cap)"
  done
  echo "fix and GPU cap present in both repos (GPU weights <= ${LRG_GPU_GIB} GiB)"
  ( cd "$THESIS" && "$PY_T" tests/test_padding_side.py | tail -1 )
}

case "${1:-}" in
install-cap)
  grep -q "LRG_GPU_GIB" "$THESIS/$CORE" || die "unzip the new core.py into $THESIS first"
  if grep -q "LRG_GPU_GIB" "$WORK/$CORE"; then echo "WORK already has the cap"; exit 0; fi
  if git -C "$THESIS" show HEAD:$CORE | diff -q - "$WORK/$CORE" >/dev/null; then
    cp "$THESIS/$CORE" "$WORK/$CORE"; echo "installed the GPU cap in $WORK/$CORE"
  else
    echo "WORK's core.py differs from the thesis version; not copying. Diff:"
    git -C "$THESIS" show HEAD:$CORE | diff - "$WORK/$CORE" | head -40; exit 1
  fi ;;
check)
  [ -d "$WORK" ] || die "WORK=$WORK not found"
  if ! grep -q "last_idx" "$WORK/$FIX"; then
    echo "WORK has the old extraction. Compare it with the pre-fix thesis version:"
    if git -C "$THESIS" show 7e84a70^:$FIX | diff -q - "$WORK/$FIX" >/dev/null; then
      echo "  identical to the pre-fix version -> safe to install the fix with:"
      echo "  cp $THESIS/$FIX $WORK/$FIX"
    else
      echo "  DIFFERENT from the pre-fix version: send this diff before going on:"
      git -C "$THESIS" show 7e84a70^:$FIX | diff - "$WORK/$FIX" | head -40
    fi
    exit 0
  fi
  check_fix
  echo; echo "Files that 'backup' would move:"; gemma_files | sed 's/^/  /'
  echo; echo "$(gemma_files | wc -l) entries. Other models are not touched." ;;

backup)
  check_fix
  stamp=$(date +%Y%m%d_%H%M%S); dest="$BACKUP/$stamp"; mkdir -p "$dest"
  gemma_files > "$dest/manifest.txt"
  while read -r f; do
    root=${f%%/*}; rel=${f#*/}
    src=$([ "$root" = WORK ] && echo "$WORK" || echo "$THESIS")/$rel
    mkdir -p "$dest/$root/$(dirname "$rel")"
    mv "$src" "$dest/$root/$rel"
  done < "$dest/manifest.txt"
  cp "$WORK/results/_analysis/gaps_FINAL.csv" "$dest/gaps_FINAL.before.csv" 2>/dev/null
  cp -r "$WORK/results/_spectral/rev/report" "$dest/report.before" 2>/dev/null
  mkdir -p "$BACKUP"; log "moved $(wc -l < "$dest/manifest.txt") entries to $dest" ;;

placement)
  check_fix; mkdir -p "$BACKUP"
  ls "$BACKUP"/*/manifest.txt >/dev/null 2>&1 || die "run 'backup' first"
  if [ -e "$WORK/results/$G/concepts" ]; then
    # phase 1 does not skip finished concepts: move a partial run aside
    part="$BACKUP/partial_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$part"
    mv "$WORK/results/$G/concepts" "$part/"; log "moved partial concepts to $part"
  fi
  log "placement: phases 1-2 for $G (batch $BATCH)"
  ( cd "$WORK" && "$PY_W" scripts/run_phases.py 1 2 --model $GM --batch-size "$BATCH" ) \
    2>&1 | tee -a "$BACKUP/placement.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "phases failed (see $BACKUP/placement.log)"
  ( cd "$WORK" && "$PY_W" scripts/analyze.py --models "${ALL8[@]}" \
      && cp results/_analysis/gaps.csv results/_analysis/gaps_FINAL.csv ) \
    2>&1 | tee -a "$BACKUP/placement.log"
  log "placement table rebuilt (results/_analysis/gaps_FINAL.csv)" ;;

words)
  # Word concepts are built by phase 0 (word_level.extract_all, which uses the
  # fixed layerwise_activations) and the placement table reads their last-token
  # pooling, so gemma's word rows were affected too. phase 0 also rewrites
  # phase0.json; keep the old one. The bases come from the unembedding and are
  # reused from the cache unchanged.
  check_fix; mkdir -p "$BACKUP"
  [ -d "$WORK/results/$G/concepts" ] || die "run 'placement' first"
  ls "$WORK/results/$G/concepts"/concept_word_* >/dev/null 2>&1 \
    && die "word concepts already present; move them aside to recompute"
  cp "$WORK/results/$G/phase0.json" "$BACKUP/phase0.before_words.json" 2>/dev/null
  log "words: phase 0 + 2 for $G (batch $BATCH)"
  ( cd "$WORK" && "$PY_W" scripts/run_phases.py 0 2 --model $GM --batch-size "$BATCH" ) \
    2>&1 | tee -a "$BACKUP/words.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "phase 0/2 failed (see $BACKUP/words.log)"
  ( cd "$WORK" && "$PY_W" scripts/analyze.py --models "${ALL8[@]}" \
      && cp results/_analysis/gaps.csv results/_analysis/gaps_FINAL.csv ) \
    2>&1 | tee -a "$BACKUP/words.log"
  n=$(grep -c "^word_.*,$G," "$WORK/results/_analysis/gaps_FINAL.csv")
  log "placement table rebuilt with word concepts ($n gemma word rows)" ;;

quiet)
  check_fix; mkdir -p "$BACKUP"
  ( cd "$WORK"
    REV=results/_spectral/rev
    "$PY_W" scripts/test_concept_subspace.py --models $GM --concepts "${CONCEPTS[@]}" \
        --k-frac 0.005 --k-fracs "${SWEEP[@]}" --layers mid final \
        --out-dir $REV/quiet_published --skip-existing --batch-size "$BATCH" || exit 1
    for mode in balance residualise; do
      "$PY_W" scripts/test_concept_subspace.py --models $GM --concepts "${CONCEPTS[@]}" \
          --k-frac 0.005 --layers mid final --length-mode $mode \
          --out-dir $REV/quiet_length_$mode --skip-existing --batch-size "$BATCH" || exit 1
    done ) 2>&1 | tee -a "$BACKUP/quiet.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "quiet tail failed (see $BACKUP/quiet.log)"
  log "quiet tail re-run for $G" ;;

report)
  ( cd "$WORK" && bash scripts/run_revision.sh report ) 2>&1 | tee -a "$BACKUP/report.log"
  log "report regenerated in $WORK/results/_spectral/rev/report" ;;

continent)
  check_fix; mkdir -p "$BACKUP"
  ( cd "$THESIS"
    "$PY_T" scripts/continent_control.py run --models $GM --skip-existing --batch-size "$BATCH" || exit 1
    "$PY_T" scripts/continent_control.py run --models $GM --datasets truth_cities \
        --k-fracs 0.0025 0.005 0.01 0.02 0.05 0.10 \
        --out-dir results/_spectral/continent_ksweep --skip-existing --batch-size "$BATCH" || exit 1
    "$PY_T" scripts/continent_control.py run --models $GM --datasets truth_cities_negated \
        --k-frac 0.10 --out-dir results/_spectral/continent_confirm --skip-existing \
        --batch-size "$BATCH" || exit 1
    "$PY_T" scripts/continent_control.py summarize
    "$PY_T" scripts/continent_ksweep_summary.py
    "$PY_T" scripts/continent_ksweep_summary.py --out-dir results/_spectral/continent_confirm
    "$PY_T" scripts/continent_confirm_report.py ) 2>&1 | tee -a "$BACKUP/continent.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "continent failed (see $BACKUP/continent.log)"
  log "continent tests re-run for $G" ;;

*) sed -n '2,27p' "$0"; exit 1 ;;
esac
