#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/server"
exec .venv/bin/python watchdog.py "$@"
