#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PAIR="en-fr"
EXPERIMENTS=("all")
MODELS=()
MODELS_PROVIDED=0
PY_ARGS=()
PULL_ONLY=0
REMOTE_ENV_DIR=".venv-v2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            MODELS=()
            MODELS_PROVIDED=1
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --pair)
            PAIR="$2"
            PY_ARGS+=("--pair" "$2")
            shift 2
            ;;
        --experiments)
            shift
            EXPERIMENTS=()
            PY_ARGS+=("--experiments")
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                EXPERIMENTS+=("$1")
                PY_ARGS+=("$1")
                shift
            done
            ;;
        --pull-only)
            PULL_ONLY=1
            shift
            ;;
        --remote-env-dir)
            REMOTE_ENV_DIR="$2"
            shift 2
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                PY_ARGS+=("$1")
                shift
            done
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

ENV_FILE="$PROJECT_DIR/.env.deploy"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy .env.deploy.example and configure it."
    exit 1
fi

source "$ENV_FILE"
KEY="${KEY/#\~/$HOME}"

for var in HOST PORT KEY USER REMOTE_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set in $ENV_FILE"
        exit 1
    fi
done

if [ "$MODELS_PROVIDED" -eq 0 ]; then
    if [ -n "${MODEL_LIST:-}" ]; then
        IFS=',' read -ra MODELS <<< "$MODEL_LIST"
    else
        MODELS=("Qwen/Qwen2.5-3B" "google/gemma-2-2b" "meta-llama/Llama-3.2-3B")
    fi
fi

RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
SSH_CMD="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
REMOTE="$USER@$HOST"
MODELS_STR="${MODELS[*]}"
if [ "${#PY_ARGS[@]}" -gt 0 ]; then
    PY_ARGS_B64="$(printf '%s\n' "${PY_ARGS[@]}" | base64 | tr -d '\n')"
else
    PY_ARGS_B64=""
fi

echo "=========================================================="
echo "  V2 Experiment Suite Remote Runner"
echo "=========================================================="
echo "  Remote:      $REMOTE:$PORT"
echo "  Remote dir:  $REMOTE_DIR"
echo "  Models:      ${MODELS[*]}"
echo "  Pair:        $PAIR"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Extra args:  ${PY_ARGS[*]:-<none>}"
echo "=========================================================="
echo ""

if [ "$PULL_ONLY" -eq 0 ]; then
    echo "▶ [1/4] Preparing remote directories..."
    $SSH_CMD "$REMOTE" "mkdir -p '$REMOTE_DIR/scripts' '$REMOTE_DIR/src' '$REMOTE_DIR/data' '$REMOTE_DIR/results/_spectral/v2_experiments' '$REMOTE_DIR/results/_spectral/eigen_cache'"

    echo "▶ [2/4] Syncing code, data, and cached result tensors..."
    rsync -avz -e "$RSYNC_SSH" "$PROJECT_DIR/scripts/" "$REMOTE:$REMOTE_DIR/scripts/"
    rsync -avz -e "$RSYNC_SSH" "$PROJECT_DIR/src/" "$REMOTE:$REMOTE_DIR/src/"
    rsync -avz -e "$RSYNC_SSH" "$PROJECT_DIR/pyproject.toml" "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true
    if [ -f "$PROJECT_DIR/uv.lock" ]; then
        rsync -avz -e "$RSYNC_SSH" "$PROJECT_DIR/uv.lock" "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true
    fi
    rsync -avz -e "$RSYNC_SSH" "$PROJECT_DIR/data/" "$REMOTE:$REMOTE_DIR/data/"

    if [ -d "$PROJECT_DIR/results" ]; then
        echo "  ▶ Syncing cached covariances, activations, and concept vectors..."
        rsync -avz -e "$RSYNC_SSH" \
            --include="*/" \
            --include="cov_*.pt" \
            --include="X_*.pt" \
            --include="v_raw_*.pt" \
            --include="eigen.pt" \
            --exclude="*" \
            "$PROJECT_DIR/results/" "$REMOTE:$REMOTE_DIR/results/"
    else
        echo "  Local results/ not found; assuming cached results already exist on remote."
    fi

    echo "▶ [3/4] Running experiments on remote..."
    $SSH_CMD "$REMOTE" \
        REMOTE_DIR="$REMOTE_DIR" \
        REMOTE_ENV_DIR="$REMOTE_ENV_DIR" \
        HF_TOKEN="${HF_TOKEN:-}" \
        MODELS_STR="$MODELS_STR" \
        PY_ARGS_B64="$PY_ARGS_B64" \
        bash <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$REMOTE_DIR"

for p in "$HOME/.local/bin" "$HOME/.cargo/bin" /usr/local/bin /root/.local/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

if ! command -v uv &>/dev/null; then
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
    [ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null || true
fi

if ! command -v uv &>/dev/null; then
    echo "  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    echo "FATAL: uv installation failed."
    exit 1
fi

echo "  uv: $(which uv) ($(uv --version))"

if [ ! -d "$REMOTE_ENV_DIR" ]; then
    echo "========================================================="
    echo "  Creating isolated V2 environment (Python 3.12)..."
    echo "========================================================="
    uv venv "$REMOTE_ENV_DIR" --python 3.12
    uv pip install --python "$REMOTE_ENV_DIR/bin/python" \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch>=2.4.0"
    uv pip install --python "$REMOTE_ENV_DIR/bin/python" \
        "numpy>=2.0.0" \
        "transformers" \
        "accelerate" \
        "safetensors" \
        "scikit-learn" \
        "scipy" \
        "tqdm" \
        "matplotlib" \
        "seaborn" \
        "python-dotenv" \
        "huggingface_hub"
fi

if [ -n "${HF_TOKEN:-}" ] && [ "$HF_TOKEN" != "hf_your_token_here" ]; then
    "$REMOTE_ENV_DIR/bin/huggingface-cli" login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
        "$REMOTE_ENV_DIR/bin/python" -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)" 2>/dev/null || true
fi

read -ra MODEL_ARRAY <<< "$MODELS_STR"
PY_ARGS_ARRAY=()
if [ -n "$PY_ARGS_B64" ]; then
    if base64 --help 2>&1 | grep -q -- '--decode'; then
        mapfile -t PY_ARGS_ARRAY < <(printf '%s' "$PY_ARGS_B64" | base64 --decode)
    else
        mapfile -t PY_ARGS_ARRAY < <(printf '%s' "$PY_ARGS_B64" | base64 -d)
    fi
fi

for model in "${MODEL_ARRAY[@]}"; do
    echo ""
    echo "============================================"
    echo "  V2 EXPERIMENTS: $model"
    echo "============================================"
    "$REMOTE_ENV_DIR/bin/python" scripts/v2_experiments.py \
        --models "$model" \
        "${PY_ARGS_ARRAY[@]}" \
        || echo "WARNING: V2 experiments failed for $model"
done

echo ""
echo "============================================"
echo "  V2 EXPERIMENTS COMPLETE"
echo "============================================"
REMOTE_SCRIPT
else
    echo "▶ --pull-only mode: skipping remote execution."
fi

echo "▶ [4/4] Pulling V2 experiment outputs..."
mkdir -p "$PROJECT_DIR/results/_spectral/v2_experiments"
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/v2_experiments/" \
    "$PROJECT_DIR/results/_spectral/v2_experiments/" 2>/dev/null || true

mkdir -p "$PROJECT_DIR/results/_spectral/eigen_cache"
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/eigen_cache/" \
    "$PROJECT_DIR/results/_spectral/eigen_cache/" 2>/dev/null || true

echo ""
echo "=========================================================="
echo "  COMPLETE. Outputs in:"
echo "    results/_spectral/v2_experiments/"
echo "=========================================================="
