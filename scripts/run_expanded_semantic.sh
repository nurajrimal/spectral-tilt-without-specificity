#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Expanded-semantic anti-concentration run (EMNLP rebuttal, P1)
#
#  WHERE TO RUN: on the machine that holds the saved covariances + a GPU venv
#                (i.e., wherever the RAID pipeline wrote results/<model>/en-fr/).
#                A CPU-only host cannot run the extraction step (no CUDA).
#
#  WHAT IT DOES:
#    1. extract_new_concepts.py  → adds 15 new v_raw_*.pt per model into en-fr/
#       (loads each model once; leaves the existing 27 + cov/X/Q untouched)
#    2. analyze_expanded_semantic.py → Gini/SCM + t-test + Mann-Whitney,
#       writes results/_spectral/expanded_semantic_summary.json
#
#  BIG MODELS: this box has 6 GB VRAM / ~10 GB free RAM / ~7 GB disk. Models
#  >= ~4B (7B/8B/9B/14B + the MoEs) will NOT fit and are in MODELS_BIG below —
#  leave them commented and run them on the CLOUD GPU instead, then copy back
#  only the tiny v_raw files (see the note at the bottom).
#
#  Run it detached so it survives SSH disconnects:
#      cd ~/Desktop/linear_rep_geometry
#      nohup bash scripts/run_expanded_semantic.sh > logs/expanded_run.log 2>&1 &
#      tail -f logs/expanded_run.log
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root (linear_rep_geometry)
PY="./.venv/bin/python"
mkdir -p logs

# ── Models that FIT on this 6 GB / 15 GB box (<= ~3B dense) ──
MODELS_SMALL=(
  "Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B"
  "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B"
  "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B"
  "google/gemma-2-2b" "HuggingFaceTB/SmolLM2-1.7B"
)

# ── Models that need the CLOUD GPU (run there; DO NOT uncomment here) ──
MODELS_BIG=(
  "Qwen/Qwen3-4B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen3-8B"
  "google/gemma-2-9b" "mistralai/Mistral-7B-v0.3" "meta-llama/Llama-3.1-8B"
  "allenai/OLMoE-1B-7B-0125" "jetmoe/jetmoe-8b"
)

echo "════════ EXTRACT (small models, on this box) ════════"
"$PY" scripts/extract_new_concepts.py --models "${MODELS_SMALL[@]}"

echo "════════ ANALYZE (all models with cov + new vectors) ════════"
"$PY" scripts/analyze_expanded_semantic.py

echo
echo "Done. Summary → results/_spectral/expanded_semantic_summary.json"
echo "The big models still need the cloud GPU. On the cloud box, run:"
echo "    MODEL_LIST=\"${MODELS_BIG[*]// /,}\" python scripts/extract_new_concepts.py"
echo "then copy back ONLY the tiny new vectors (KB each), e.g. from this laptop:"
echo "    rsync -avz --include='v_raw_*.pt' --include='*/' --exclude='*' \\"
echo "        CLOUD:~/PATH/results/ ~/Desktop/linear_rep_geometry/results/"
echo "and re-run: $PY scripts/analyze_expanded_semantic.py"
