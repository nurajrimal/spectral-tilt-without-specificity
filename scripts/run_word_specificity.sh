#!/usr/bin/env bash
# Word-level specificity: real word concepts vs arbitrary word groups.
#
#   bash scripts/run_word_specificity.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_word_specificity.sh primary  # language basis, final layer, 8 models
#   bash scripts/run_word_specificity.sh robust   # mid layer + full-vocabulary basis
#   bash scripts/run_word_specificity.sh summary  # cross-model tables (CPU)
#   bash scripts/run_word_specificity.sh all      # primary, robust, summary
#
# Skips models whose JSON already exists, so it is safe to re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
OUT=results/_spectral/word_specificity
PREREG=prereg/word_specificity_prereg.json
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

sweep() {  # basis layer
  local basis=$1 layer=$2
  for m in "${MODELS[@]}"; do
    local name=${m##*/}
    if [ -f "$OUT/${basis}_${layer}/${name}.json" ]; then
      echo "skip $name ($basis, $layer)"; continue
    fi
    $PY scripts/word_specificity.py run --model "$m" --basis "$basis" --layer "$layer" \
      2>&1 | tee "$OUT/logs/${name}_${basis}_${layer}.log"
  done
}

case "${1:-}" in
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  primary) require_frozen; sweep language final ;;
  robust)  require_frozen; sweep language mid; sweep full_vocab final ;;
  summary)
    $PY scripts/word_specificity.py summarize --basis language --layer final --groups
    $PY scripts/word_specificity.py summarize --basis language --layer mid || true
    $PY scripts/word_specificity.py summarize --basis full_vocab --layer final || true ;;
  all)     require_frozen; sweep language final; sweep language mid
           sweep full_vocab final; "$0" summary ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
