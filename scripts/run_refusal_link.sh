#!/usr/bin/env bash
# Refusal: dual geometry and link with refusal words.
#
#   bash scripts/run_refusal_link.sh test     # CPU self-test, no model
#   bash scripts/run_refusal_link.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_refusal_link.sh all      # 8 models, then summary
#   bash scripts/run_refusal_link.sh summary  # cross-model tables (CPU)
#
# Skips models whose result exists; a failed model is reported and skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
OUT=results/_spectral/refusal_link
PREREG=prereg/refusal_link_prereg.json
STATEMENTS=${STATEMENTS:-$HOME/spectral_anti_concentration/data/statements}
GPU_GIB=${GPU_GIB:-3.0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

# model:batch (frozen in the pre-registration)
MODELS=(Qwen/Qwen2.5-0.5B:8 Qwen/Qwen2.5-3B:8 Qwen/Qwen2.5-7B:32 Qwen/Qwen3-4B:32
        mistralai/Mistral-7B-v0.3:32 meta-llama/Llama-3.2-3B:32 meta-llama/Llama-3.1-8B:32
        google/gemma-2-2b:8)

require_frozen() {
  if ! $PY scripts/freeze_prereg.py --prereg "$PREREG" --check >/dev/null 2>&1; then
    echo "pre-registration not frozen (or edited since). Run: $0 freeze, then commit." >&2
    exit 1
  fi
}

run_all() {
  for mb in "${MODELS[@]}"; do
    local m=${mb%:*} b=${mb##*:} name=${mb%:*}; name=${name##*/}
    if [ -f "$OUT/results/${name}.json" ]; then echo "skip $name"; continue; fi
    $PY scripts/refusal_link.py run --model "$m" --statements-dir "$STATEMENTS" \
      --gpu-gib "$GPU_GIB" --batch-size "$b" 2>&1 | tee "$OUT/logs/${name}.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAILED: $name (see $OUT/logs/${name}.log)"; fi
  done
}

case "${1:-}" in
  test)    $PY tests/test_refusal_link.py ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  all)     require_frozen; run_all; $PY scripts/refusal_link.py summarize ;;
  summary) $PY scripts/refusal_link.py summarize ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
