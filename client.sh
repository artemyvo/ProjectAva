#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/client"
source .venv/bin/activate
exec python main.py "$@"
