#!/usr/bin/env python3
"""The base's own reading of a nightly-measurement panel — the report's reference.

    cd server
    .venv/bin/python panel_base.py warmth

GPU, inference down (watchdog job `panel_base`, args ["warmth"]): loads the bare base as the
judges do and reads the panel (`core/adapter_measure.py` `--panel-base`) on the capture bench's
frozen contexts → data/measure/<panel>_panel_base.json. Changes nothing else.
"""
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(_SERVER_DIR / "inference"), str(_SERVER_DIR)]

from core import adapter_measure  # noqa: E402

if __name__ == "__main__":
    sys.exit(adapter_measure._cli(["--panel-base"] + sys.argv[1:]))
