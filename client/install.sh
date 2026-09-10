#!/usr/bin/env bash
# First-time client setup — run once on the machine that will run the PyQt6 UI.
#
# Creates client/.venv and installs the UI dependencies (PyQt6, websockets).
# No ML/GPU dependencies here — the client is a thin WebSocket UI.
#
# Usage:
#   cd /path/to/repo/client
#   bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
CLIENT_REQS="$SCRIPT_DIR/requirements.txt"

echo "==> Ava client — first-time setup"
echo "    Repo client dir : $SCRIPT_DIR"
echo "    Venv            : $VENV_DIR"
echo ""

# ── Python check ────────────────────────────────────────────────────────────
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

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "==> Venv already exists — skipping creation"
else
    echo "==> Creating venv ..."
    python3 -m venv "$VENV_DIR"
    echo "    Done."
fi

VENV_PIP="$VENV_DIR/bin/pip"

# ── Install client deps ────────────────────────────────────────────────────────
if [ ! -f "$CLIENT_REQS" ]; then
    echo "ERROR: requirements file not found: $CLIENT_REQS" >&2
    exit 1
fi

echo "==> Installing client dependencies ..."
"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r "$CLIENT_REQS"
echo "    Done."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Start the client:"
echo ""
echo "    cd $SCRIPT_DIR"
echo "    .venv/bin/python main.py [--server ws://HOST:8765]"
echo ""
echo "  Or from the repo root:"
echo "    ./client.sh [--server ws://HOST:8765]"
echo "============================================================"
