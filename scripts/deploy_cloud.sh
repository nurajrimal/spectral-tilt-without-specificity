#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Deploy + launch the REBUTTAL cloud pipeline (run_cloud_all.sh) on the GPU box.
#
#  Unlike run_remote.sh (which runs the full 22-model perform_raid pipeline),
#  this pushes the code/data and launches run_cloud_all.sh = P1 (expanded
#  semantics, 17 base) + P4 (3 instruct) + P5 (centered).  Reads .env.deploy.
#
#  Run from the laptop (where .env.deploy lives):
#      cd ~/Desktop/linear_rep_geometry
#      bash scripts/deploy_cloud.sh            # push + launch (assumes env set up)
#      bash scripts/deploy_cloud.sh --setup    # also run setup_remote.sh first
#      bash scripts/deploy_cloud.sh --pull     # just pull the summary JSONs back
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DO_SETUP=false; PULL_ONLY=false
for a in "$@"; do case "$a" in
  --setup) DO_SETUP=true ;; --pull) PULL_ONLY=true ;;
  *) echo "unknown arg: $a"; exit 1 ;; esac; done

ENV_FILE="$PROJECT_DIR/.env.deploy"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found (copy .env.deploy.example)."; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
KEY="${KEY/#\~/$HOME}"
for v in HOST PORT KEY USER REMOTE_DIR; do [ -n "${!v:-}" ] || { echo "ERROR: $v unset in .env.deploy"; exit 1; }; done

SSH_CMD="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
REMOTE="$USER@$HOST"

echo "▶ [0] test connection to $REMOTE:$PORT"
$SSH_CMD "$REMOTE" "echo SSH_OK" >/dev/null || { echo "cannot connect — check .env.deploy"; exit 1; }

if [ "$PULL_ONLY" = true ]; then
  echo "▶ pulling summaries"
  mkdir -p "$PROJECT_DIR/results/_spectral"
  rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/expanded_semantic_summary.json" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/expanded_semantic_centered_summary.json" \
    "$PROJECT_DIR/results/_spectral/" || echo "(summaries not ready yet)"
  exit 0
fi

echo "▶ [1] push code + data (no .venv/.git/results/*.pt)"
rsync -avz --delete -e "$RSYNC_SSH" \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude 'results' --exclude 'remote_results' --exclude '*.pt' \
  --exclude '.env' --exclude '.env.deploy' --exclude '.gemini' --exclude '*.lock' \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

if [ "$DO_SETUP" = true ]; then
  echo "▶ [2] setup remote env"
  $SSH_CMD "$REMOTE" "REMOTE_DIR=$REMOTE_DIR HF_TOKEN=${HF_TOKEN:-} bash -s" < "$PROJECT_DIR/scripts/setup_remote.sh"
fi

echo "▶ [3] launch run_cloud_all.sh in tmux session 'rebuttal'"
$SSH_CMD "$REMOTE" "tmux kill-session -t rebuttal 2>/dev/null || true"
$SSH_CMD "$REMOTE" "cd $REMOTE_DIR && mkdir -p logs && tmux new-session -d -s rebuttal \
  'export PATH=\$HOME/.local/bin:\$PATH HF_TOKEN=\"${HF_TOKEN:-}\" PYTHON=\"uv run python\" && \
   bash scripts/run_cloud_all.sh 2>&1 | tee logs/cloud_all.log; touch logs/cloud_all.done; sleep 86400'"

echo ""
echo "  ✓ launched. Monitor:"
echo "      $SSH_CMD $REMOTE 'tail -f $REMOTE_DIR/logs/cloud_all.log'"
echo "  When logs/cloud_all.done appears, pull results:"
echo "      bash scripts/deploy_cloud.sh --pull"
