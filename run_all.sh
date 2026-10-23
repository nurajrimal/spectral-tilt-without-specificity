#!/usr/bin/env bash
# Run every model sequentially on a single GPU box.
#
# Replaces job_phases.sh, which assumes Slurm. jalapeno has no scheduler, so
# the models run one after another in the foreground. ALWAYS start this inside
# tmux — it runs for hours and an SSH drop would otherwise kill it:
#
#     tmux new -s thesis
#     bash run_all.sh
#     # Ctrl-b then d to detach; close the laptop; `tmux attach -t thesis` later
#
# Options:
#     PHASES="0 1" bash run_all.sh          # subset of phases
#     MODELS="Qwen/Qwen2.5-0.5B" bash run_all.sh
#     BATCH=32 bash run_all.sh              # override the auto-sized batch
#
# The box is shared. Before a long run, check who else is on the card:
#     nvidia-smi

set -uo pipefail

cd "$(dirname "$0")"
[ -f thesis_env.sh ] && source thesis_env.sh
[ -d .venv ] && source .venv/bin/activate

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen3-4B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B google/gemma-2-2b mistralai/Mistral-7B-v0.3}"
PHASES="${PHASES:-0 1 3 4 5 6}"
RESULTS="${THESIS_RESULTS:-results}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

# ── Be a good neighbour on a shared card ─────────────────────────────────────
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU: ${USED:-?} / ${TOTAL:-?} MiB in use"
if [ "${USED:-0}" -gt 5000 ]; then
  echo
  echo "WARNING: something else is already using ${USED} MiB."
  echo "This is a shared machine. Check 'nvidia-smi' for whose job that is"
  echo "before taking the card for several hours."
  read -r -p "Continue anyway? [y/N] " yn
  [[ "$yn" =~ ^[Yy]$ ]] || exit 1
fi

# ── Batch size from free memory, not total ───────────────────────────────────
# 96 GB is far more than the suite needs, so the usual constraint is absent.
# The limit that still applies is output_hidden_states, which keeps every layer
# alive at once: memory grows with batch x length x layers x d, not with
# parameter count.
FREE=$(( ${TOTAL:-24000} - ${USED:-0} ))
if   [ "$FREE" -gt 60000 ]; then DEF_BS=32
elif [ "$FREE" -gt 30000 ]; then DEF_BS=16
elif [ "$FREE" -gt 20000 ]; then DEF_BS=8
else                             DEF_BS=4
fi
BATCH="${BATCH:-$DEF_BS}"
MAXLEN="${MAXLEN:-256}"
MAXPAIRS="${MAXPAIRS:-512}"
echo "batch-size $BATCH, max-length $MAXLEN, max-pairs $MAXPAIRS"
echo "phases: $PHASES"
echo

START=$(date +%s)
declare -a FAILED=()
declare -a DONE=()

for MODEL in $MODELS; do
  NAME=$(basename "$MODEL")
  LOG="$LOGDIR/${NAME}.log"
  echo "========================================================"
  echo "$NAME   ($(date +%H:%M:%S))   -> $LOG"
  echo "========================================================"

  # Keep going if one model fails. A gated-licence 403 on Llama should not
  # cost you the Qwen results that would have run after it.
  if python scripts/run_phases.py $PHASES \
        --model "$MODEL" \
        --results-root "$RESULTS" \
        --data-dir data/statements \
        --word-pairs data/word_pairs \
        --device cuda \
        --batch-size "$BATCH" \
        --max-length "$MAXLEN" \
        --max-pairs "$MAXPAIRS" \
        2>&1 | tee "$LOG"; then
    DONE+=("$NAME")
    echo "  ok"
  else
    FAILED+=("$NAME")
    echo "  FAILED — see $LOG"
  fi

  # Give the card a moment to release memory before the next load.
  sleep 5
  echo
done

ELAPSED=$(( $(date +%s) - START ))
echo "========================================================"
printf 'finished in %dh %dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
echo "ok:     ${#DONE[@]}  ${DONE[*]:-}"
echo "failed: ${#FAILED[@]} ${FAILED[*]:-}"
echo "========================================================"

if [ ${#DONE[@]} -gt 1 ]; then
  echo
  echo "Next, the cross-model analysis:"
  echo "  python scripts/analyze.py --models ${DONE[*]}"
fi
[ ${#FAILED[@]} -eq 0 ]
