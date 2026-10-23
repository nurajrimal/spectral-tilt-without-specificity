#!/bin/bash
# Description: Deploy hole-fix experiments + Hole 4 to remote GPU.
# Syncs required data, runs GPU-dependent scripts (Holes 2-5, Llama RAID/SAE),
# and pulls results back down.
#
# Usage:
#   bash scripts/run_hole_fixes.sh                       # Run all (original)
#   bash scripts/run_hole_fixes.sh --only 3               # Hole 3
#   bash scripts/run_hole_fixes.sh --only 4               # Hole 4 v2 (3 models)
#   bash scripts/run_hole_fixes.sh --only 5               # Hole 5
#   bash scripts/run_hole_fixes.sh --only 2               # Hole 2 (SAE gemma-2-9b)
#   bash scripts/run_hole_fixes.sh --only llama-raid       # Llama-3.1-8B RAID
#   bash scripts/run_hole_fixes.sh --only llama-sae        # Llama Scope SAE check
#   bash scripts/run_hole_fixes.sh --only pos              # POS probe (3 models)
#   bash scripts/run_hole_fixes.sh --only pos-expanded     # POS probe (8 models, English)
#   bash scripts/run_hole_fixes.sh --only pos-chinese      # POS probe (Qwen, Chinese UD)
#   bash scripts/run_hole_fixes.sh --only pos-sensitivity  # POS probe sensitivity sweep (8 models, k=0.05, 0.20)
#   bash scripts/run_hole_fixes.sh --only 4-expanded       # Hole 4 v2 (Llama-3.1-8B + Mistral-7B)
#   bash scripts/run_hole_fixes.sh --only all-expanded     # All expanded experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse args ──
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Load config from .env.deploy ──
ENV_FILE="$PROJECT_DIR/.env.deploy"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy .env.deploy.example and configure it."
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# Expand ~ in KEY path
KEY="${KEY/#\~/$HOME}"

# Validate required variables
for var in HOST PORT KEY USER REMOTE_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set in $ENV_FILE"
        exit 1
    fi
done

RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
SSH_CMD="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
REMOTE="$USER@$HOST"

echo "=========================================================="
echo "  Deploying Hole Fix Experiments to Remote GPU"
echo "=========================================================="
echo "  Host: $HOST:$PORT"
echo "  Remote dir: $REMOTE_DIR"
[ -n "$ONLY" ] && echo "  Running ONLY: $ONLY" || echo "  Running: ALL (Holes 2, 3, 4v2, 5, llama-raid, llama-sae, pos)"
echo ""

# ── Step 1: Prepare remote ──
echo "▶ [1/4] Preparing remote directories..."
$SSH_CMD "$REMOTE" "mkdir -p $REMOTE_DIR/results/_spectral/hole_fixes \
    $REMOTE_DIR/results/_spectral/artifact_check \
    $REMOTE_DIR/results/_spectral/eigen_cache \
    $REMOTE_DIR/scripts \
    $REMOTE_DIR/data/word_pairs \
    $REMOTE_DIR/data"

# ── Step 2: Push required files (NO results/ bulk sync) ──
echo "▶ [2/4] Syncing scripts, source, and data to remote..."

# Push scripts
rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/scripts/" "$REMOTE:$REMOTE_DIR/scripts/"

# Push source code
rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/src/" "$REMOTE:$REMOTE_DIR/src/"

# Push project config
rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$PROJECT_DIR/README.md" "$REMOTE:$REMOTE_DIR/"

# Push word pair data
rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/data/word_pairs/" "$REMOTE:$REMOTE_DIR/data/word_pairs/"

# Push paired contexts (needed for RAID and Hole 4 model loading)
if [ -d "$PROJECT_DIR/data/paired_contexts" ]; then
    rsync -avz -e "$RSYNC_SSH" \
        "$PROJECT_DIR/data/paired_contexts/" "$REMOTE:$REMOTE_DIR/data/paired_contexts/"
fi

# Push held-out PPL passages (needed for Hole 4 v2)
if [ -f "$PROJECT_DIR/data/heldout_ppl_passages.txt" ]; then
    rsync -avz -e "$RSYNC_SSH" \
        "$PROJECT_DIR/data/heldout_ppl_passages.txt" "$REMOTE:$REMOTE_DIR/data/"
fi

# Push required covariance files
echo "  ▶ Syncing covariance files..."
rsync -avz -e "$RSYNC_SSH" \
    --include="*/" --include="cov_*.pt" --include="v_raw_*.pt" --exclude="*" \
    "$PROJECT_DIR/results/" "$REMOTE:$REMOTE_DIR/results/"

echo "  ✓ Files synced (bulk results/ skipped)"
echo ""

# ── Step 3: Execute experiments ──
# Using heredoc with quoted delimiter (<<'REMOTE_SCRIPT') to prevent ALL
# local variable expansion inside the script. We pass variables as env vars.
echo "▶ [3/4] Executing experiments on remote..."

$SSH_CMD "$REMOTE" REMOTE_DIR="$REMOTE_DIR" HF_TOKEN="${HF_TOKEN:-}" ONLY="$ONLY" MODEL_LIST="${MODEL_LIST:-}" bash <<'REMOTE_SCRIPT'
    set -e
    cd "$REMOTE_DIR"

    # Robust PATH setup — find uv wherever it might be
    for p in "$HOME/.local/bin" "$HOME/.cargo/bin" /usr/local/bin /root/.local/bin; do
        [ -d "$p" ] && export PATH="$p:$PATH"
    done

    # Source profile if uv still not found
    if ! command -v uv &>/dev/null; then
        [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
        [ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null || true
    fi

    # Install uv if still not found
    if ! command -v uv &>/dev/null; then
        echo "  uv not found — installing..."
        sudo mkdir -p "$HOME/.config" "$HOME/.local/bin" 2>/dev/null || true
        sudo chown -R "$(id -u):$(id -g)" "$HOME/.config" "$HOME/.local" 2>/dev/null || true
        curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Final check
    if ! command -v uv &>/dev/null; then
        echo "FATAL: uv installation failed."
        exit 1
    fi

    echo "  uv found at: $(which uv) ($(uv --version))"

    # Sync environment (installs Python + all deps from pyproject.toml)
    echo "  Syncing environment..."
    uv sync

    # HF login for gated models
    export HF_TOKEN
    if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "hf_your_token_here" ]; then
        uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi

    # ── Hole 3: SAE Feature Collapse Fix (gemma-2-2b) ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "3" ]; then
        echo ''
        echo '============================================'
        echo '  HOLE 3: SAE Feature Collapse (top-k=5)'
        echo '  Model: gemma-2-2b'
        echo '============================================'
        uv run python scripts/fix_hole3_sae_collapse.py --model gemma-2-2b --top-k 5 \
            || echo 'WARNING: Hole 3 failed'
    fi

    # ── Hole 2: SAE on gemma-2-9b (cross-size validation) ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "2" ]; then
        echo ''
        echo '============================================'
        echo '  HOLE 2: SAE Artifact Check (gemma-2-9b)'
        echo '============================================'
        uv run python scripts/artifact_check_sae.py --model gemma-2-9b --top-k 5 \
            || echo 'WARNING: Hole 2 (SAE gemma-2-9b) failed'
    fi

    # ── Hole 5: Random Unembedding Null Model ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "5" ]; then
        echo ''
        echo '============================================'
        echo '  HOLE 5: Random Unembedding Null Model'
        echo '  Models: Qwen2.5-0.5B, Llama-3.2-1B,'
        echo '          gemma-2-2b, SmolLM2-1.7B'
        echo '============================================'
        uv run python scripts/fix_hole5_unembed_null.py \
            --models Qwen2.5-0.5B Llama-3.2-1B gemma-2-2b SmolLM2-1.7B \
            || echo 'WARNING: Hole 5 failed'
    fi

    # ── Llama-3.1-8B RAID (full pipeline for cross-family SAE) ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "llama-raid" ]; then
        echo ''
        echo '============================================'
        echo '  LLAMA RAID: Full pipeline on Llama-3.1-8B'
        echo '============================================'
        MODEL_LIST="meta-llama/Llama-3.1-8B" uv run python scripts/perform_raid.py \
            || echo 'WARNING: Llama-3.1-8B RAID failed'
    fi

    # ── Llama Scope SAE Cross-Family Check ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "llama-sae" ]; then
        echo ''
        echo '============================================'
        echo '  LLAMA SAE: Llama Scope on Llama-3.1-8B'
        echo '============================================'
        uv run python scripts/artifact_check_sae.py --model Llama-3.1-8B --top-k 5 \
            || echo 'WARNING: Llama Scope SAE check failed'
    fi

    # ── Hole 4 v2: Redesigned Split-Injection Causal Intervention ──
    # NOTE: --model accepts a SINGLE model path, so we loop over models.
    if [ -z "$ONLY" ] || [ "$ONLY" = "4" ]; then
        for h4_model in google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B; do
            h4_short="${h4_model##*/}"
            echo ''
            echo '============================================'
            echo "  HOLE 4 v2: Split-Injection ($h4_short)"
            echo '  Multi-layer, high-alpha, held-out PPL'
            echo '============================================'
            uv run python scripts/fix_hole4_v2_split_injection.py \
                --model "$h4_model" \
                --alphas 1.0 5.0 10.0 20.0 50.0 100.0 \
                --top-k-frac 0.1 \
                || echo "WARNING: Hole 4 v2 ($h4_short) failed"
        done
    fi

    # ── POS Probe Experiment (uses PyTorch GPU logistic regression natively) ──
    if [ -z "$ONLY" ] || [ "$ONLY" = "pos" ]; then
        echo ''
        echo '============================================'
        echo "  POS PROBE: Original 3 Models"
        echo '============================================'
        uv run python scripts/pos_probe_experiment.py \
            --models google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B \
            --pair en-fr \
            --k-frac 0.1 \
            --max-sentences 500 \
            || echo "WARNING: POS probe failed"
    fi

    # ── Expanded English POS Probing (8 models) ──
    if [ "$ONLY" = "pos-expanded" ] || [ "$ONLY" = "all-expanded" ]; then
        echo ''
        echo '============================================'
        echo "  POS PROBE EXPANDED: 8 English models"
        echo '============================================'
        uv run python scripts/pos_probe_experiment.py \
            --models google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B \
                     meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3 \
                     HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen3-4B Qwen/Qwen2.5-7B \
            --pair en-fr \
            --k-frac 0.1 \
            --max-sentences 500 \
            --out-suffix expanded \
            || echo "WARNING: POS probe expanded failed"
    fi

    # ── Chinese POS Probing on Qwen models (multilingual hypothesis) ──
    if [ "$ONLY" = "pos-chinese" ] || [ "$ONLY" = "all-expanded" ]; then
        echo ''
        echo '============================================'
        echo "  POS PROBE CHINESE: Qwen multilingual test"
        echo '============================================'
        uv run python scripts/pos_probe_experiment.py \
            --models Qwen/Qwen2.5-3B Qwen/Qwen3-4B \
            --pair en-fr \
            --k-frac 0.1 \
            --max-sentences 500 \
            --ud-lang chinese \
            || echo "WARNING: POS probe Chinese failed"
    fi

    # ── POS Probing Sensitivity Sweeps (k=0.05, 0.20) ──
    if [ "$ONLY" = "pos-sensitivity" ] || [ "$ONLY" = "all-expanded" ]; then
        echo ''
        echo '============================================'
        echo "  POS PROBE SENSITIVITY: 8 models (k=0.05, 0.20)"
        echo '============================================'
        for k_val in 0.05 0.20; do
            echo "  --> Running k-frac = $k_val"
            uv run python scripts/pos_probe_experiment.py \
                --models google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B \
                         meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3 \
                         HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen3-4B Qwen/Qwen2.5-7B \
                --pair en-fr \
                --k-frac $k_val \
                --max-sentences 500 \
                || echo "WARNING: POS probe sensitivity (k=$k_val) failed"
        done
    fi

    # ── Expanded Split-Injection (Llama-3.1-8B + Mistral-7B) ──
    if [ "$ONLY" = "4-expanded" ] || [ "$ONLY" = "all-expanded" ]; then
        for h4_model in meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3; do
            h4_short="${h4_model##*/}"
            echo ''
            echo '============================================'
            echo "  HOLE 4 v2 EXPANDED: Split-Injection ($h4_short)"
            echo '  Multi-layer, high-alpha, held-out PPL'
            echo '============================================'
            uv run python scripts/fix_hole4_v2_split_injection.py \
                --model "$h4_model" \
                --alphas 1.0 5.0 10.0 20.0 50.0 100.0 \
                --top-k-frac 0.1 \
                || echo "WARNING: Hole 4 v2 expanded ($h4_short) failed"
        done
    fi

    echo ''
    echo '============================================'
    echo '  ALL EXPERIMENTS COMPLETE'
    echo '============================================'
REMOTE_SCRIPT

echo "  ✓ Remote execution complete"
echo ""

# ── Step 4: Pull results back ──
echo "▶ [4/4] Pulling results back down..."
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/hole_fixes/" \
    "$PROJECT_DIR/results/_spectral/hole_fixes/"

rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/artifact_check/" \
    "$PROJECT_DIR/results/_spectral/artifact_check/"

# Pull Llama-3.1-8B RAID results if they exist
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/Llama-3.1-8B/" \
    "$PROJECT_DIR/results/Llama-3.1-8B/" 2>/dev/null || true

# Pull POS probe results (including k_frac variations)
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/"pos_probe* \
    "$PROJECT_DIR/results/_spectral/" 2>/dev/null || true

echo "  ✓ Results downloaded"
echo ""
echo "=========================================================="
echo "  COMPLETE. Outputs in:"
echo "    results/_spectral/hole_fixes/"
echo "    results/_spectral/artifact_check/"
echo "    results/_spectral/pos_probe/"
echo "=========================================================="
