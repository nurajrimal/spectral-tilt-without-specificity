#!/usr/bin/env bash
# Work out whether you need a sysadmin, and produce a report to paste into the
# request if you do.
#
#     bash scripts/sysadmin_report.sh
#     bash scripts/sysadmin_report.sh > report.txt
#
# Most of what this project needs installs into a virtual environment as an
# ordinary user. Only two things genuinely require root: the NVIDIA driver
# (already present if the GPU works at all) and a system Python new enough to
# build a venv. Everything else is a pip install you can do yourself, and
# asking an admin for it wastes their time and a week of yours.

set -o pipefail   # deliberately NOT -u: this script runs on unknown
                  # machines where USER, SCRATCH or TMPDIR may be unset, and a
                  # diagnostic that dies on an unset variable is useless.

NEED_ADMIN=()
CAN_SELF=()

hdr() { echo; echo "## $1"; }
say() { printf '  %-34s %s\n' "$1" "$2"; }

echo "========================================================"
echo "Dependency report"
echo "host: $(hostname)    user: $(whoami)    date: $(date -u +%Y-%m-%dT%H:%MZ)"
ME="$(whoami)"
echo "========================================================"

hdr "1. System Python"
PYOK=no
for py in python3.12 python3.11 python3.10 python3; do
  if command -v "$py" >/dev/null 2>&1; then
    V=$("$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")
    say "$py" "$V"
    if [ "$V" != "?" ] && [ "$(printf '%s\n3.10\n' "$V" | sort -V | head -1)" = "3.10" ]; then
      PYOK=yes
    fi
  fi
done
if [ "$PYOK" = yes ]; then
  say "verdict" "a usable Python is present"
else
  say "verdict" "NO Python >= 3.10 found on PATH"
  NEED_ADMIN+=("Python 3.10 or newer (or tell me which module provides it)")
fi

hdr "2. Module system"
if command -v module >/dev/null 2>&1; then
  say "module command" "present"
  echo "  python modules:"
  module avail python 2>&1 | grep -iE "python" | head -8 | sed 's/^/    /' || echo "    (none listed)"
  echo "  cuda / pytorch modules:"
  module avail 2>&1 | grep -iE "cuda|pytorch|torch" | head -8 | sed 's/^/    /' || echo "    (none listed)"
  say "note" "if a suitable module is listed, no install is needed"
else
  say "module command" "not found"
fi

hdr "3. Can I create a virtual environment?"
TMPV="${TMPDIR:-/tmp}/venvtest_$$"
if python3 -m venv "$TMPV" >/dev/null 2>&1; then
  say "python3 -m venv" "works"
  if "$TMPV/bin/pip" install --quiet --disable-pip-version-check packaging >/dev/null 2>&1; then
    say "pip install into venv" "works (network reachable)"
    CAN_SELF+=("all Python packages")
  else
    say "pip install into venv" "FAILED"
    say "" "venv is fine but PyPI is unreachable from this node"
    NEED_ADMIN+=("outbound HTTPS to pypi.org, or a local package mirror/proxy")
  fi
  rm -rf "$TMPV"
else
  say "python3 -m venv" "FAILED"
  NEED_ADMIN+=("the python3-venv package, or a working module-provided Python")
fi

hdr "4. GPU and driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/  gpu: /' || say "nvidia-smi" "present but returned nothing (login node?)"
  CUDAV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | head -1 || true)
  say "driver reports" "${CUDAV:-unknown}"
  say "note" "the driver is admin-managed; matching torch is a user pip install"
else
  say "nvidia-smi" "not found"
  say "" "expected on a login node — re-run inside a GPU job before escalating"
fi

hdr "5. Already-installed Python packages"
for pkg in torch numpy scipy transformers accelerate safetensors tqdm sklearn; do
  V=$(python3 -c "import $pkg,sys;print(getattr($pkg,'__version__','present'))" 2>/dev/null || true)
  say "$pkg" "${V:-not installed}"
done
say "note" "missing entries install into a venv without admin rights"

hdr "6. Disk"
for d in "${SCRATCH:-}" "/scratch/$ME" "/work/$ME" "$HOME"; do
  [ -n "$d" ] && [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|  $d: |"
done
say "needed" "~79 GB for model weights, plus room for results"
if command -v quota >/dev/null 2>&1; then
  echo "  quota:"; quota -s 2>/dev/null | head -5 | sed 's/^/    /' || true
fi

hdr "7. Scheduler"
for c in sinfo qstat bjobs; do
  command -v "$c" >/dev/null 2>&1 && say "$c" "present"
done

echo
echo "========================================================"
echo "VERDICT"
echo "========================================================"
if [ ${#NEED_ADMIN[@]} -eq 0 ]; then
  cat <<'EOF'
Nothing requires admin intervention.

Everything this project needs installs into a virtual environment you own:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

The only likely request is scratch quota or Hugging Face access, neither of
which is a library install. Do not send a package list to your sysadmin.
EOF
else
  echo "These genuinely need someone with root or policy control:"
  printf '  - %s\n' "${NEED_ADMIN[@]}"
  echo
  echo "Everything else is a user-level pip install. Ask ONLY for the above —"
  echo "a long package list invites a slow 'we don't support that' reply."
fi
echo
echo "Paste this whole report into the request. It answers the first three"
echo "questions any sysadmin will ask."
