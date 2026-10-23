#!/usr/bin/env bash
# Causal ablation: does the model rely on the quiet tail for truth?
#
#   bash scripts/run_causal_ablation.sh check    # data + basis present? GPU?
#   bash scripts/run_causal_ablation.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_causal_ablation.sh smoke    # Qwen2.5-0.5B only, to time it
#   bash scripts/run_causal_ablation.sh run      # all 8 models (GPU)
#   bash scripts/run_causal_ablation.sh summary  # cross-model tables (CPU)
#   bash scripts/run_causal_ablation.sh all      # run, then summary
#
# Skips models whose JSON already exists, so it is safe to re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
OUT=results/_spectral/causal_ablation
PREREG=prereg/causal_ablation_prereg.json
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

case "${1:-}" in
  check)
    bad=0
    [ -f data/statements/truth_cities.jsonl ] && echo "ok   truth_cities.jsonl" || { echo "MISSING data/statements/truth_cities.jsonl"; bad=1; }
    [ -f data/heldout_ppl_passages.txt ] && echo "ok   heldout passages" || { echo "MISSING data/heldout_ppl_passages.txt"; bad=1; }
    for m in "${MODELS[@]}"; do n=${m##*/}
      [ -f "results/$n/en-fr/cov_src.pt" ] && echo "ok   basis $n" || { echo "MISSING results/$n/en-fr/cov_src.pt"; bad=1; }
    done
    $PY -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    exit $bad ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  smoke)   require_frozen
           PYTHONUNBUFFERED=1 $PY scripts/causal_ablation.py run --models Qwen/Qwen2.5-0.5B \
             --skip-existing 2>&1 | tee -a "$OUT/logs/smoke.log" ;;
  run)     require_frozen
           PYTHONUNBUFFERED=1 $PY scripts/causal_ablation.py run --models "${MODELS[@]}" \
             --skip-existing 2>&1 | tee -a "$OUT/logs/run.log" ;;
  summary) $PY scripts/causal_ablation.py summarize ;;
  all)     require_frozen; "$0" run; "$0" summary ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
