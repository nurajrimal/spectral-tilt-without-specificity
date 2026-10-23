#!/usr/bin/env bash
# Put the word-level control and the behavioural concepts into the SAME
# phase2_*.json, so analyze.py can run the section 6 paired test.
#
# phase2 iterates over acts_*.pt. Earlier runs used --no-save-acts (0 concepts
# written); later reruns saved behavioural activations only, because phase 0 --
# which extracts the 42 word concepts -- was not in the phase list. Neither run
# had both in one file. Phases 0 1 2 together, activations retained, fixes it.
#
# Disk: ~9-12 GB per model. Checked before each model, cleared after phase 2
# has consumed them.

set -uo pipefail
cd "$(dirname "$0")"
[ -f thesis_env.sh ] && source thesis_env.sh
[ -d .venv ] && source .venv/bin/activate

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen3-4B mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B google/gemma-2-2b}"
RESULTS="${THESIS_RESULTS:-results}"
MAXPAIRS="${MAXPAIRS:-256}"
BATCH="${BATCH:-16}"
NEED_GB="${NEED_GB:-12}"
mkdir -p logs

free_gb() { df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9'; }

echo "phases 0 1 2, activations retained then deleted per model"
echo "max-pairs $MAXPAIRS, batch $BATCH, need ~${NEED_GB} GB per model"
echo "free now: $(free_gb) GB"
echo

declare -a DONE=() FAILED=() SKIPPED=()
START=$(date +%s)

for MODEL in $MODELS; do
  NAME=$(basename "$MODEL")
  FREE=$(free_gb)
  echo "========================================================"
  echo "$NAME   ($(date +%H:%M:%S))   ${FREE} GB free"
  echo "========================================================"

  if [ "${FREE:-0}" -lt "$NEED_GB" ]; then
    echo "  SKIP: under ${NEED_GB} GB free."
    SKIPPED+=("$NAME"); continue
  fi

  if python scripts/run_phases.py 0 1 2 \
        --model "$MODEL" \
        --results-root "$RESULTS" \
        --data-dir data/statements \
        --word-pairs data/word_pairs \
        --device cuda \
        --batch-size "$BATCH" \
        --max-length 256 \
        --max-pairs "$MAXPAIRS" \
        2>&1 | tee "logs/wordctl_${NAME}.log"; then
    N=$(python -c "
import json
d = json.load(open('$RESULTS/$NAME/phase2_full_vocab.json'))
c = d.get('concepts', {})
w = sum(1 for k in c if k.startswith('word_'))
print(f'{len(c)} concepts ({w} word-level, {len(c)-w} behavioural)')
" 2>/dev/null || echo "could not read phase2 file")
    echo "  ok -> $N"
    DONE+=("$NAME")
  else
    echo "  FAILED -> logs/wordctl_${NAME}.log"
    FAILED+=("$NAME")
  fi

  rm -f "$RESULTS/$NAME/concepts/acts_"*.pt
  echo "  activations cleared, free now: $(free_gb) GB"
  echo
done

ELAPSED=$(( $(date +%s) - START ))
echo "========================================================"
printf 'finished in %dh %dm\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
echo "ok:      ${#DONE[@]}  ${DONE[*]:-}"
echo "failed:  ${#FAILED[@]} ${FAILED[*]:-}"
echo "skipped: ${#SKIPPED[@]} ${SKIPPED[*]:-}"
echo "========================================================"

if [ ${#DONE[@]} -ge 2 ]; then
  echo
  echo "Then: python scripts/analyze.py --models ${DONE[*]}"
  echo
  echo "Read the word-level rows first. If they anti-concentrate under the"
  echo "shuffle null while the behavioural rows do not, the claim is about"
  echo "behavioural concepts. If the word rows are ALSO null, the claim is"
  echo "about the published effect itself."
fi
[ ${#FAILED[@]} -eq 0 ]
