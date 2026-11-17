#!/usr/bin/env bash
# The two sides of the dual geometry on identical word pairs.
#
#   bash scripts/run_dual_paired.sh test     # CPU self-test, no model
#   bash scripts/run_dual_paired.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_dual_paired.sh primary  # language basis, final layer
#   bash scripts/run_dual_paired.sh robust   # mid layer + full-vocab basis
#   bash scripts/run_dual_paired.sh summary  # cross-model tables (CPU)
#   bash scripts/run_dual_paired.sh all      # primary, robust, summary
#
# Reads the activation caches of word_specificity (no GPU needed if they exist).
# Skips models whose JSON already exists, so it is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
OUT=results/_spectral/dual_paired
PREREG=prereg/dual_paired_prereg.json
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
    $PY scripts/dual_paired.py run --model "$m" --basis "$basis" --layer "$layer" \
      2>&1 | tee "$OUT/logs/${name}_${basis}_${layer}.log"
  done
}

case "${1:-}" in
  test)    $PY tests/test_dual_paired.py ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  primary) require_frozen; sweep language final ;;
  robust)  require_frozen; sweep language mid; sweep full_vocab final ;;
  summary)
    $PY scripts/dual_paired.py summarize --basis language --layer final
    $PY scripts/dual_paired.py summarize --basis language --layer mid || true
    $PY scripts/dual_paired.py summarize --basis full_vocab --layer final || true ;;
  all)     require_frozen; sweep language final; sweep language mid
           sweep full_vocab final; bash "$0" summary ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
