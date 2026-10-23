#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Full rebuttal experiment run — CLOUD GPU, self-contained (EMNLP rebuttal)
#
#  Runs entirely on the cloud box: needs only the repo + model downloads (no
#  dependency on the laptop's saved covariances — it recomputes them).
#
#  Prereqs on the cloud box:
#    - repo pushed (e.g. ./scripts/run_remote.sh --skip-setup from the laptop,
#      or git pull) INCLUDING data/word_pairs/ (42 files) and these scripts
#    - HF_TOKEN exported (gated models: Llama, Gemma)
#    - PYTHON pointing at the env, e.g. export PYTHON="uv run python"
#                                      or  export PYTHON="./.venv/bin/python"
#
#  Run detached:
#    export HF_TOKEN=hf_...; export PYTHON="uv run python"
#    nohup bash scripts/run_cloud_all.sh > logs/cloud_all.log 2>&1 &
#    tail -f logs/cloud_all.log
#
#  Produces:
#    results/_spectral/expanded_semantic_summary.json           (uncentered / headline)
#    results/_spectral/expanded_semantic_centered_summary.json  (centered, R2 robustness)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p logs

# ── The paper's 17 core base models ──
BASE_17="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B \
Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B \
meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B \
google/gemma-2-2b google/gemma-2-9b mistralai/Mistral-7B-v0.3 HuggingFaceTB/SmolLM2-1.7B \
allenai/OLMoE-1B-7B-0125 jetmoe/jetmoe-8b"

# ── P4: instruct variants of models we already have as base (base-vs-instruct) ──
INSTRUCT="Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct google/gemma-2-2b-it"

echo "════════ [1/3] EXTRACT: cov (+centered) + all 42 concepts ════════"
$PY scripts/run_cloud_expanded.py --centered --models $BASE_17 $INSTRUCT

echo "════════ [2/3] ANALYZE: uncentered (headline) ════════"
$PY scripts/analyze_expanded_semantic.py \
    --out results/_spectral/expanded_semantic_summary.json

echo "════════ [3/3] ANALYZE: centered covariance (R2 robustness) ════════"
$PY scripts/analyze_expanded_semantic.py --cov-name cov_src_centered.pt \
    --out results/_spectral/expanded_semantic_centered_summary.json

echo
echo "DONE. Pull back the two small JSON summaries (KB):"
echo "  results/_spectral/expanded_semantic_summary.json"
echo "  results/_spectral/expanded_semantic_centered_summary.json"
