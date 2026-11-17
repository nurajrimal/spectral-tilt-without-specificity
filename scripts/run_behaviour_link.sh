#!/usr/bin/env bash
# Truth: dual geometry and link with report words.
#
#   bash scripts/run_behaviour_link.sh test     # CPU self-test, no model
#   bash scripts/run_behaviour_link.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_behaviour_link.sh run      # 8 models (GPU for activations)
#   bash scripts/run_behaviour_link.sh summary  # cross-model tables (CPU)
#   bash scripts/run_behaviour_link.sh all      # run, then summary
#
# STATEMENTS points at the folder holding the truth_*.jsonl files.
# Skips models whose result already exists, so it is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
OUT=results/_spectral/behaviour_link
PREREG=prereg/behaviour_link_prereg.json
STATEMENTS=${STATEMENTS:-$HOME/spectral_anti_concentration/data/statements}
GPU_GIB=${GPU_GIB:-3.5}             # lower it (e.g. GPU_GIB=2.5) if a model still runs out of memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

run_all() {
  for m in "${MODELS[@]}"; do
    local name=${m##*/}
    if [ -f "$OUT/results/${name}.json" ]; then echo "skip $name"; continue; fi
    $PY scripts/behaviour_link.py run --model "$m" --statements-dir "$STATEMENTS" \
      --gpu-gib "$GPU_GIB" 2>&1 | tee "$OUT/logs/${name}.log" \
      || echo "FAILED: $name (see $OUT/logs/${name}.log); continuing"
  done
}

case "${1:-}" in
  test)    $PY tests/test_behaviour_link.py ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  run)     require_frozen; run_all ;;
  summary) $PY scripts/behaviour_link.py summarize ;;
  all)     require_frozen; run_all; bash "$0" summary ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
