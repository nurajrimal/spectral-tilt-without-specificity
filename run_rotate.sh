#!/usr/bin/env bash
# Run models one at a time on a disk-constrained machine.
#
# jalapeno has ~13 GB free on a 99%-full disk. The eight-model suite is ~79 GB
# of weights, and three of the models individually exceed the free space. This
# script downloads one model, runs it, deletes the weights, and moves to the
# next, so peak usage is one model rather than all of them.
#
#     tmux new -s thesis
#     bash run_rotate.sh
#
# Options:
#     MODELS="Qwen/Qwen2.5-0.5B" bash run_rotate.sh
#     PHASES="0 1" bash run_rotate.sh
#     KEEP=1 bash run_rotate.sh        # don't delete weights after each model
#     MIN_FREE_GB=20 bash run_rotate.sh
#
# Deleting weights means re-downloading if you rerun a model. That is the trade:
# bandwidth is cheap here, disk is not. Once more disk exists, use run_all.sh
# instead and keep the cache.

set -uo pipefail
cd "$(dirname "$0")"
[ -f thesis_env.sh ] && source thesis_env.sh
[ -d .venv ] && source .venv/bin/activate

# Approximate bf16 download size in GB. Used to skip models that cannot fit
# before wasting a partial download.
declare -A SIZE=(
  ["Qwen/Qwen2.5-0.5B"]=1   ["Qwen/Qwen3-0.6B"]=2    ["meta-llama/Llama-3.2-1B"]=3
  ["Qwen/Qwen2.5-1.5B"]=4   ["Qwen/Qwen3-1.7B"]=4    ["google/gemma-2-2b"]=6
  ["Qwen/Qwen2.5-3B"]=7     ["meta-llama/Llama-3.2-3B"]=7
  ["Qwen/Qwen3-4B"]=9       ["mistralai/Mistral-7B-v0.3"]=15
  ["Qwen/Qwen2.5-7B"]=16    ["meta-llama/Llama-3.1-8B"]=17
)

# Ordered small-to-large: the small ones finish and produce results before the
# large ones get a chance to fail on space.
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B Qwen/Qwen3-4B Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3}"
PHASES="${PHASES:-0 1 3 4 5 6}"
RESULTS="${THESIS_RESULTS:-results}"
KEEP="${KEEP:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"     # headroom to leave after a download
BATCH="${BATCH:-16}"
MAXLEN="${MAXLEN:-256}"
MAXPAIRS="${MAXPAIRS:-512}"

CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs "$CACHE"

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

echo "cache:  $CACHE"
echo "free:   $(free_gb "$CACHE") GB"
echo "phases: $PHASES"
echo "delete weights after each model: $([ "$KEEP" = 1 ] && echo no || echo yes)"
echo

declare -a DONE=() FAILED=() SKIPPED=()
START=$(date +%s)

for MODEL in $MODELS; do
  NAME=$(basename "$MODEL")
  NEED=${SIZE[$MODEL]:-10}
  FREE=$(free_gb "$CACHE")

  echo "========================================================"
  echo "$NAME   needs ~${NEED} GB, ${FREE} GB free   ($(date +%H:%M:%S))"
  echo "========================================================"

  if [ "$FREE" -lt $(( NEED + MIN_FREE_GB )) ]; then
    echo "  SKIP: not enough space (need ${NEED}+${MIN_FREE_GB} GB headroom)."
    echo "  Free disk or move HF_HOME to a larger volume, then rerun with:"
    echo "    MODELS=\"$MODEL\" bash run_rotate.sh"
    SKIPPED+=("$NAME")
    continue
  fi

  LOG="logs/${NAME}.log"
  if python scripts/run_phases.py $PHASES \
        --model "$MODEL" \
        --results-root "$RESULTS" \
        --data-dir data/statements \
        --word-pairs data/word_pairs \
        --device cuda --batch-size "$BATCH" \
        --max-length "$MAXLEN" --max-pairs "$MAXPAIRS" \
        --no-save-acts \
        2>&1 | tee "$LOG"; then
    DONE+=("$NAME")
    echo "  ok -> $LOG"
  else
    FAILED+=("$NAME")
    echo "  FAILED -> $LOG"
  fi

  # Results are small and stay. Weights are large and go. Delete by repo
  # directory rather than wiping the whole cache, so a model someone else
  # pulled onto this shared machine is left alone.
  if [ "$KEEP" != 1 ]; then
    ORG="${MODEL%%/*}"; REPO="${MODEL##*/}"
    for d in "$CACHE/hub/models--${ORG}--${REPO}" "$CACHE/models--${ORG}--${REPO}"; do
      if [ -d "$d" ]; then
        SZ=$(du -sh "$d" 2>/dev/null | cut -f1)
        rm -rf "$d" && echo "  freed $SZ from $(basename "$d")"
      fi
    done
  fi
  echo "  free now: $(free_gb "$CACHE") GB"
  echo
done

ELAPSED=$(( $(date +%s) - START ))
echo "========================================================"
printf 'finished in %dh %dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
echo "ok:      ${#DONE[@]}  ${DONE[*]:-}"
echo "failed:  ${#FAILED[@]} ${FAILED[*]:-}"
echo "skipped: ${#SKIPPED[@]} ${SKIPPED[*]:-}"
echo "========================================================"

if [ ${#SKIPPED[@]} -gt 0 ]; then
  cat <<EOF

${#SKIPPED[@]} model(s) did not fit. The proposal's §7 picks the suite for
"family and scale coverage rather than size", so dropping the largest models
costs scale range but keeps family coverage — report which models ran and why,
rather than quietly running a smaller suite than the proposal states.
EOF
fi

if [ ${#DONE[@]} -gt 1 ]; then
  echo
  echo "Cross-model analysis:"
  echo "  python scripts/analyze.py --models ${DONE[*]}"
fi
[ ${#FAILED[@]} -eq 0 ]
