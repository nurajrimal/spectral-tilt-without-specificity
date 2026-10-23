#!/usr/bin/env bash
# One-time setup on the college server. Run it from the repo root, on a LOGIN
# node — it installs and downloads but never needs a GPU.
#
#     bash setup_server.sh
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

echo "=== 1. Where are we ==="
echo "host:    $(hostname)"
echo "cwd:     $(pwd)"
echo "user:    $(whoami)"

# ── Scratch ──────────────────────────────────────────────────────────────────
# Model weights are ~79 GB for the eight-model suite. Home directories on
# clusters are usually quota-limited well below that, and the download fails
# halfway with an error that does not mention quota. Point the Hugging Face
# cache at scratch before anything is fetched.
echo
echo "=== 2. Scratch space ==="
SCRATCH="${SCRATCH:-}"
if [ -z "$SCRATCH" ]; then
  for cand in "/scratch/$USER" "/work/$USER" "/data/$USER" "$HOME/scratch"; do
    if [ -d "$(dirname "$cand")" ] 2>/dev/null; then SCRATCH="$cand"; break; fi
  done
fi
SCRATCH="${SCRATCH:-$HOME/scratch}"
mkdir -p "$SCRATCH/hf_cache" "$SCRATCH/thesis_results"
echo "scratch: $SCRATCH"
df -h "$SCRATCH" | tail -1

# ── Python ───────────────────────────────────────────────────────────────────
# Prefer the cluster's module system: a module-provided torch is built against
# the installed driver, whereas a plain `pip install torch` grabs whatever CUDA
# PyPI defaults to and may silently fall back to CPU.
echo
echo "=== 3. Python ==="
if command -v module >/dev/null 2>&1; then
  echo "module system present. Candidates:"
  (module avail python 2>&1 | head -15) || true
  (module avail cuda   2>&1 | head -10) || true
  echo
  echo "If a python/cuda module is listed, load it and re-run this script:"
  echo "    module load python/3.12   # adjust to what is listed"
  echo "    module load cuda          # adjust"
fi
python3 --version

if [ ! -d .venv ]; then
  echo "creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

echo
echo "=== 4. Libraries ==="
if python -c "import torch" 2>/dev/null; then
  echo "torch present: $(python -c 'import torch;print(torch.__version__)')"
else
  cat <<'EOF'
torch is not installed.

Install it the way the cluster expects. In order of preference:
  1. module load pytorch            (if the module system offers it)
  2. pip install torch --index-url https://download.pytorch.org/whl/cu121
     with cu121 replaced by the CUDA version `nvidia-smi` reports
  3. pip install torch             (last resort; may not match the driver)

Then re-run this script.
EOF
  exit 1
fi

pip install --quiet -r requirements.txt
echo "requirements installed"

# ── Environment ──────────────────────────────────────────────────────────────
echo
echo "=== 5. Environment variables ==="
ENVFILE="$PWD/thesis_env.sh"
cat > "$ENVFILE" <<EOF
# Source this before every run:  source thesis_env.sh
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export THESIS_RESULTS="$SCRATCH/thesis_results"
# Keep BLAS from oversubscribing cores on a shared node; the scheduler
# allocates CPUs and an unpinned numpy will happily take the whole machine.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=\$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
source "$PWD/.venv/bin/activate"
EOF
echo "wrote $ENVFILE"
# shellcheck disable=SC1090
source "$ENVFILE"

# ── Hugging Face ─────────────────────────────────────────────────────────────
echo
echo "=== 6. Hugging Face access ==="
if [ -f "$HF_HOME/token" ] || [ -n "${HF_TOKEN:-}" ]; then
  echo "credential found"
else
  cat <<'EOF'
No token. Three of the eight models are gated:
    meta-llama/Llama-3.2-3B, meta-llama/Llama-3.1-8B, google/gemma-2-2b

Do both of these, in this order:
  1. Visit each model page while signed in and ACCEPT the licence.
     A token alone still returns 403 — this step is separate and is the one
     people miss.
  2. huggingface-cli login
EOF
fi

# ── Data ─────────────────────────────────────────────────────────────────────
echo
echo "=== 7. Datasets ==="
if [ -d data/statements ] && [ "$(ls -1 data/statements/*.jsonl 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "already built: $(ls -1 data/statements/*.jsonl | wc -l) files"
else
  python scripts/build_datasets.py --clone --cache-dir "$SCRATCH/external"
fi

if [ -d data/word_pairs ]; then
  echo "word pairs: $(ls -1 data/word_pairs/*.txt 2>/dev/null | wc -l) concept files"
else
  echo "WARNING: data/word_pairs missing — the §6 control cannot be built."
  echo "Copy it from the published repo."
fi

# ── Verify ───────────────────────────────────────────────────────────────────
echo
echo "=== 8. Environment check ==="
python scripts/check_env.py || true

cat <<EOF

Done. Note that CUDA will show as unavailable here: this is a login node.
Re-run the check inside a GPU job to confirm the card.

Next:
    source thesis_env.sh
    srun --gres=gpu:1 --pty bash      # or your cluster's equivalent
    python scripts/check_env.py
EOF
