#!/bin/bash
# Description: Deploy POS-tag probing experiment to remote GPU.
# Syncs required data, runs pos_probe_experiment.py for each model,
# and pulls results back down.
#
# Usage:
#   bash scripts/run_pos_probe.sh                                    # All 3 models
#   bash scripts/run_pos_probe.sh --models google/gemma-2-2b         # Single model
#   bash scripts/run_pos_probe.sh --k-frac 0.1                       # Custom k fraction
#   bash scripts/run_pos_probe.sh --max-sentences 1000               # More sentences

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
MODELS=("google/gemma-2-2b" "Qwen/Qwen2.5-3B" "meta-llama/Llama-3.2-3B")
PAIR="en-fr"
K_FRAC="0.1"
MAX_SENTENCES=500
MODELS_PROVIDED=0

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            MODELS=()
            MODELS_PROVIDED=1
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS+=("$1"); shift
            done
            ;;
        --pair) PAIR="$2"; shift 2 ;;
        --k-frac) K_FRAC="$2"; shift 2 ;;
        --max-sentences) MAX_SENTENCES="$2"; shift 2 ;;
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

if [ "$MODELS_PROVIDED" -eq 0 ] && [ -n "${MODEL_LIST:-}" ]; then
    IFS=',' read -ra MODELS <<< "$MODEL_LIST"
fi

KEY="${KEY/#\~/$HOME}"

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
echo "  POS-Tag Probing Experiment"
echo "=========================================================="
echo "  Host: $HOST:$PORT"
echo "  Models: ${MODELS[*]}"
echo "  Pair: $PAIR"
echo "  k-frac: $K_FRAC"
echo "  Max sentences: $MAX_SENTENCES"
echo ""

# ── Step 1: Prepare remote ──
echo "▶ [1/4] Preparing remote directories..."
$SSH_CMD "$REMOTE" "mkdir -p $REMOTE_DIR/results/_spectral/pos_probe \
    $REMOTE_DIR/results/_spectral/eigen_cache \
    $REMOTE_DIR/scripts \
    $REMOTE_DIR/data/word_pairs \
    $REMOTE_DIR/data"

# ── Step 2: Push required files ──
echo "▶ [2/4] Syncing scripts, source, and data to remote..."

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/scripts/" "$REMOTE:$REMOTE_DIR/scripts/"

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/src/" "$REMOTE:$REMOTE_DIR/src/"

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$REMOTE:$REMOTE_DIR/"

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/data/word_pairs/" "$REMOTE:$REMOTE_DIR/data/word_pairs/"

if [ -d "$PROJECT_DIR/data/paired_contexts" ]; then
    rsync -avz -e "$RSYNC_SSH" \
        "$PROJECT_DIR/data/paired_contexts/" "$REMOTE:$REMOTE_DIR/data/paired_contexts/"
fi

# Push covariance files for each model
echo "  ▶ Syncing covariance and vector files..."
for model_path in "${MODELS[@]}"; do
    model_short="${model_path##*/}"
    if [ -d "$PROJECT_DIR/results/$model_short" ]; then
        $SSH_CMD "$REMOTE" "mkdir -p $REMOTE_DIR/results/$model_short"
        rsync -avz -e "$RSYNC_SSH" --include="*/" --include="cov_*.pt" --include="v_raw_*.pt" --exclude="*" \
            "$PROJECT_DIR/results/$model_short/" "$REMOTE:$REMOTE_DIR/results/$model_short/"
    fi
done

echo "  ✓ Files synced"
echo ""

# ── Step 3: Execute experiment ──
echo "▶ [3/4] Executing POS probe experiments on remote..."

# Build models string for the heredoc
MODELS_STR="${MODELS[*]}"

$SSH_CMD "$REMOTE" REMOTE_DIR="$REMOTE_DIR" HF_TOKEN="${HF_TOKEN:-}" \
    MODELS_STR="$MODELS_STR" PAIR="$PAIR" K_FRAC="$K_FRAC" \
    MAX_SENTENCES="$MAX_SENTENCES" bash <<'REMOTE_SCRIPT'
    set -e
    cd "$REMOTE_DIR"

    # Robust PATH setup
    for p in "$HOME/.local/bin" "$HOME/.cargo/bin" /usr/local/bin /root/.local/bin; do
        [ -d "$p" ] && export PATH="$p:$PATH"
    done

    if ! command -v uv &>/dev/null; then
        [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
        [ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null || true
    fi

    if ! command -v uv &>/dev/null; then
        echo "  uv not found — installing..."
        sudo mkdir -p "$HOME/.config" "$HOME/.local/bin" 2>/dev/null || true
        sudo chown -R "$(id -u):$(id -g)" "$HOME/.config" "$HOME/.local" 2>/dev/null || true
        curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "FATAL: uv installation failed."
        exit 1
    fi

    echo "  uv found at: $(which uv) ($(uv --version))"
    uv sync

    export HF_TOKEN
    if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "hf_your_token_here" ]; then
        uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi

    # Run POS probe for each model
    read -ra MODEL_ARRAY <<< "$MODELS_STR"
    for pos_model in "${MODEL_ARRAY[@]}"; do
        pos_short="${pos_model##*/}"
        echo ''
        echo '============================================'
        echo "  POS PROBE: $pos_short"
        echo '============================================'
        uv run python scripts/pos_probe_experiment.py \
            --model "$pos_model" \
            --pair "$PAIR" \
            --k-frac "$K_FRAC" \
            --max-sentences "$MAX_SENTENCES" \
            || echo "WARNING: POS probe ($pos_short) failed"
    done

    # Multi-model comparison plot if >1 model
    if [ "${#MODEL_ARRAY[@]}" -gt 1 ]; then
        echo ''
        echo '============================================'
        echo '  POS PROBE: Multi-model comparison'
        echo '============================================'
        uv run python scripts/pos_probe_experiment.py \
            --models ${MODELS_STR} \
            --pair "$PAIR" \
            --k-frac "$K_FRAC" \
            --max-sentences "$MAX_SENTENCES" \
            || echo "WARNING: Multi-model comparison failed"
    fi

    echo ''
    echo '============================================'
    echo '  POS PROBE EXPERIMENTS COMPLETE'
    echo '============================================'
REMOTE_SCRIPT

echo "  ✓ Remote execution complete"
echo ""

# ── Step 4: Pull results ──
echo "▶ [4/4] Pulling POS probe results back down..."
mkdir -p "$PROJECT_DIR/results/_spectral/pos_probe"
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/pos_probe/" \
    "$PROJECT_DIR/results/_spectral/pos_probe/"

echo "  ✓ Results downloaded"
echo ""
echo "=========================================================="
echo "  COMPLETE. Outputs in:"
echo "    results/_spectral/pos_probe/"
echo "=========================================================="
