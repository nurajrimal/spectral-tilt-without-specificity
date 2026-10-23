#!/usr/bin/env bash
# Standalone wrapper to run POS Probing with cuML GPU Acceleration
# Creates an isolated Python 3.12 environment to avoid dependency conflicts.

ENV_DIR=".venv-cuml"

if [ ! -d "$ENV_DIR" ]; then
    echo "========================================================="
    echo "  Creating isolated cuML environment (Python 3.12)..."
    echo "========================================================="
    
    uv venv "$ENV_DIR" --python 3.12
    
    # Step 1: Install PyTorch with CUDA first (from PyTorch's own index)
    # This prevents cuML from pulling in an older/CPU-only torch
    uv pip install --python "$ENV_DIR"/bin/python \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch>=2.4.0"

    # Step 2: Install cuML + remaining deps (torch already satisfied at >=2.4)
    uv pip install --python "$ENV_DIR"/bin/python \
        --extra-index-url https://pypi.nvidia.com \
        "cuml-cu12" \
        "transformers" \
        "accelerate" \
        "safetensors" \
        "scikit-learn" \
        "scipy" \
        "tqdm" \
        "matplotlib" \
        "numpy<2.0.0" \
        "seaborn"
    
    echo "Environment bootstrapped successfully."
    echo ""
fi

echo "========================================================="
echo "  Running POS Probe Experiment via cuML Environment..."
echo "========================================================="

# Run the python script with all passed arguments forwarded
"$ENV_DIR"/bin/python scripts/pos_probe_experiment.py "$@"
