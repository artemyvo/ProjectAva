#!/usr/bin/env bash
# First-time server setup — run once on the Linux GPU machine.
#
# Creates server/.venv with --system-site-packages (so a system-level
# PyTorch / CUDA installation is visible without reinstalling it), then
# installs Python dependencies for the inference server.
#
# Usage:
#   cd /path/to/repo/server
#   bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
INFERENCE_REQS="$SCRIPT_DIR/inference/requirements.txt"

echo "==> Ava server — first-time setup"
echo "    Repo server dir : $SCRIPT_DIR"
echo "    Venv            : $VENV_DIR"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ and re-run." >&2
    exit 1
fi

PY_VER=$(python3 -c "import sys; print('%d.%d' % sys.version_info[:2])")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info[0])")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info[1])")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VER)." >&2
    exit 1
fi
echo "==> Python $PY_VER OK"

# ── Create venv ────────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "==> Venv already exists — skipping creation"
else
    echo "==> Creating venv with --system-site-packages ..."
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "    Done."
fi

VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

# ── Install inference deps ─────────────────────────────────────────────────────
if [ ! -f "$INFERENCE_REQS" ]; then
    echo "ERROR: requirements file not found: $INFERENCE_REQS" >&2
    exit 1
fi

echo "==> Installing inference dependencies ..."
"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r "$INFERENCE_REQS"
echo "    Done."

# ── flash-linear-attention (Qwen3.x hybrid linear-attention kernels) ───────────
# Required for Qwen3.x (qwen3_5, e.g. Qwen3.6) — without it the linear-attention
# layers fall back to a slow torch implementation. Pulled from git because the
# PyPI wheel is broken (ships fla/layers + fla/models but NOT fla/ops). Installed
# with --no-deps so it cannot perturb the pinned torch/transformers/unsloth stack
# (its only runtime dep here, einops, is in requirements.txt). Harmless for
# non-Qwen models. Triton-based — no CUDA compiler needed.
echo "==> Installing flash-linear-attention (Qwen3.x kernels, --no-deps) ..."
"$VENV_PIP" install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git"

# causal-conv1d completes the Qwen3.5 gated-delta-net fast path (decode ~3x on the
# DGX Spark; fla alone leaves transformers on its torch fallback). No aarch64/cu130
# wheel exists, so it is a source build — deliberately NOT run here: an 8-way nvcc
# build beside a model load froze the Spark on 2026-09-07. Run it by hand, ALONE:
#   TORCH_CUDA_ARCH_LIST="12.0;12.1" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
#     "$VENV_PIP" install --no-build-isolation causal-conv1d
echo "    Done."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Start the server (inside tmux or screen):"
echo ""
echo "    cd $SCRIPT_DIR"
echo "    .venv/bin/python watchdog.py"
echo ""
echo "  Optional flags:"
echo "    --host 0.0.0.0   bind address (default: 0.0.0.0)"
echo "    --port 8765      inference WebSocket port (default: 8765)"
echo "    --mgmt-port 8766 watchdog HTTP port (default: 8766)"
echo "============================================================"
