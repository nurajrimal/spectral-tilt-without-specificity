#!/usr/bin/env bash
# Dictionary-side specificity: unembedding-row concepts vs arbitrary word groups.
#
#   bash scripts/run_dictionary_specificity.sh test     # CPU self-test, no model
#   bash scripts/run_dictionary_specificity.sh check    # loader == published loader (no results)
#   bash scripts/run_dictionary_specificity.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_dictionary_specificity.sh primary  # language basis, published token rule
#   bash scripts/run_dictionary_specificity.sh robust   # full-vocab basis + space_first rule
#   bash scripts/run_dictionary_specificity.sh summary  # cross-model tables (CPU)
#   bash scripts/run_dictionary_specificity.sh all      # primary, robust, summary
#
# No GPU needed: only the unembedding matrix is read. Skips models whose JSON
# already exists, so it is safe to re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
OUT=results/_spectral/dictionary_specificity
PREREG=prereg/dictionary_specificity_prereg.json
LOADER=${LOADER:-light}          # set LOADER=full if 'check' reports a mismatch
mkdir -p "$OUT/logs"

MODELS=(Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen3-4B
        mistralai/Mistral-7B-v0.3 meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B
        google/gemma-2-2b)

require_frozen() {
  if ! $PY scripts/freeze_prereg.py --prereg "$PREREG" --check >/dev/null 2>&1; then
    echo "pre-registration not frozen (or edited since). Run: $0 freeze, then commit." >&2
    exit 1
  fi
}

sweep() {  # basis rule
  local basis=$1 rule=$2
  for m in "${MODELS[@]}"; do
    local name=${m##*/}
    if [ -f "$OUT/${basis}_${rule}/${name}.json" ]; then
      echo "skip $name ($basis, $rule)"; continue
    fi
    $PY scripts/dictionary_specificity.py run --model "$m" --basis "$basis" \
      --rule "$rule" --loader "$LOADER" 2>&1 | tee "$OUT/logs/${name}_${basis}_${rule}.log"
  done
}

case "${1:-}" in
  test)    $PY tests/test_dictionary_specificity.py ;;
  check)   $PY scripts/dictionary_specificity.py check --model Qwen/Qwen2.5-0.5B ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  primary) require_frozen; sweep language published ;;
  robust)  require_frozen; sweep full_vocab published; sweep language space_first ;;
  summary)
    $PY scripts/dictionary_specificity.py summarize --basis language --rule published
    $PY scripts/dictionary_specificity.py summarize --basis full_vocab --rule published || true
    $PY scripts/dictionary_specificity.py summarize --basis language --rule space_first || true ;;
  all)     require_frozen; sweep language published; sweep full_vocab published
           sweep language space_first; "$0" summary ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
