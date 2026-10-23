#!/usr/bin/env bash
#SBATCH --job-name=thesis
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-7
#
# One array task per model. Submit with:
#
#     mkdir -p logs
#     sbatch job_phases.sh
#
# Or a single model:
#
#     sbatch --array=0 job_phases.sh
#
# Adjust the #SBATCH lines to your cluster's partition names and limits; run
# `sinfo` to see what exists. If the cluster uses PBS rather than Slurm, the
# body below is unchanged — only the directives differ (#PBS -l select=...).
#
# This is the job script the proposal's §9 calls for: the published Makefile
# assumes an interactive GPU box, and replacing that layer is the one piece of
# orchestration that has to change. The analysis code is untouched.

set -euo pipefail

MODELS=(
  "Qwen/Qwen2.5-0.5B"
  "Qwen/Qwen2.5-3B"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen3-4B"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.1-8B"
  "google/gemma-2-2b"
  "mistralai/Mistral-7B-v0.3"
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
PHASES="${PHASES:-0 1 3 4 5 6}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source thesis_env.sh

echo "=================================================="
echo "model   : $MODEL"
echo "phases  : $PHASES"
echo "node    : $(hostname)"
echo "started : $(date)"
echo "=================================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail fast rather than 40 minutes in. check_env exits non-zero on anything
# blocking, which on a GPU node includes a torch build that does not match the
# driver — the failure mode that otherwise looks like a very slow job.
python scripts/check_env.py --models "$MODEL"

# Batch size by GPU memory. An 8B model in bf16 is ~16 GB of weights before any
# activations, and output_hidden_states keeps every layer alive at once.
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "${GPU_MEM:-0}" -lt 20000 ]; then BS=2; ML=128
elif [ "${GPU_MEM:-0}" -lt 40000 ]; then BS=4; ML=256
else                                    BS=8; ML=256
fi
echo "gpu memory ${GPU_MEM} MB -> batch-size $BS, max-length $ML"

# IMPORTANT: Phase 2 must be run with --freeze BEFORE Phase 1 output is
# analysed. It is deliberately NOT in the default phase list — run it once,
# interactively, and commit results/<model>/decision_rule.json. That commit
# timestamp is what makes "pre-specified" checkable.

python scripts/run_phases.py $PHASES \
    --model "$MODEL" \
    --results-root "${THESIS_RESULTS:-results}" \
    --data-dir data/statements \
    --word-pairs data/word_pairs \
    --device cuda \
    --batch-size "$BS" \
    --max-length "$ML" \
    --max-pairs 512 \
    --no-save-acts

echo "finished: $(date)"
