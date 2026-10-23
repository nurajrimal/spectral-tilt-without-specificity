#!/usr/bin/env bash
# Semantic (continent) control for the quiet tail.
#
#   bash scripts/run_continent_control.sh check    # data, cache and basis present?
#   bash scripts/run_continent_control.sh freeze   # once, BEFORE any run; then commit
#   bash scripts/run_continent_control.sh run      # 8 models x 2 datasets (GPU only if a cache is missing)
#   bash scripts/run_continent_control.sh summary  # cross-model tables (CPU)
#   bash scripts/run_continent_control.sh all      # run, then summary
#
# Skips (model, dataset) files that already exist, so it is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
OUT=results/_spectral/continent_control
PREREG=prereg/continent_control_prereg.json
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

check() {
  local bad=0
  for ds in truth_cities truth_cities_negated; do
    [ -f "data/statements/$ds.jsonl" ] && echo "ok   data/statements/$ds.jsonl" \
      || { echo "MISSING data/statements/$ds.jsonl"; bad=1; }
  done
  for m in "${MODELS[@]}"; do
    n=${m##*/}
    [ -f "results/$n/en-fr/cov_src.pt" ] || { echo "MISSING results/$n/en-fr/cov_src.pt"; bad=1; }
    for ds in truth_cities truth_cities_negated; do
      ls results/_spectral/concept_subspace/acts_${n}__${ds}__last_token__L*.pt >/dev/null 2>&1 \
        && echo "ok   cache $n / $ds" \
        || echo "note no cache for $n / $ds (will be extracted on the GPU)"
    done
  done
  return $bad
}

case "${1:-}" in
  check)   check ;;
  freeze)  $PY scripts/freeze_prereg.py --prereg "$PREREG" ;;
  run)     require_frozen
           $PY scripts/continent_control.py run --models "${MODELS[@]}" --skip-existing \
             2>&1 | tee -a "$OUT/logs/run.log" ;;
  summary) $PY scripts/continent_control.py summarize ;;
  all)     require_frozen; "$0" run; "$0" summary ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
