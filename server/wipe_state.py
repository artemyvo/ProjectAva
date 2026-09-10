#!/usr/bin/env python3
"""Ava state wipe — DESTRUCTIVE disaster recovery, run as an offline job.

Deletes all regenerable reflection state (LoRA adapters, RAG indexes, reflection
memory/ledger/runs/staging) — but when chats are kept, human-validated (locked)
exchange records in the chat sidecars are preserved, since a hand-authored,
operator-reviewed target is un-regenerable; with ``--wipe-chats`` also deletes the
raw transcripts + reflection review archive; then repoints the base model in
server_config.json (clears adapter_id, optionally a new model_id). The inference
server must be DOWN before this runs — the model is resident in VRAM and the
adapters it points at are about to be deleted — so the watchdog stops inference,
runs this to completion (``sync`` job), then relaunches with the reset config.

This lives in the repo (not the watchdog) so the wipe semantics + data-layout
knowledge upgrade with a plain ``git pull``; the watchdog just invokes it by name
from watchdog_jobs.json. Prints a single JSON result object to STDOUT (the
watchdog captures + returns it); human-readable progress goes to STDERR.

Usage:
    python wipe_state.py [--wipe-chats] [--model-id MODEL_ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bootstrap inference/ + server/ onto sys.path so `core` resolves. The wipe
# helpers are GPU-free, stdlib-only file ops.
_SERVER_DIR = Path(__file__).resolve().parent
for _p in (_SERVER_DIR / "inference", _SERVER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.state_wipe import (  # noqa: E402
    wipe_reflection_state,
    wipe_chats_and_archive,
    restore_archived_transcripts,
    reset_server_config,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ava state wipe (destructive)")
    parser.add_argument("--wipe-chats", action="store_true",
                        help="Also delete raw transcripts + reflection review archive")
    parser.add_argument("--model-id", default="",
                        help="Repoint the base model_id (default: keep current)")
    args = parser.parse_args()

    wipe_chats = bool(args.wipe_chats)
    model_id = (args.model_id or "").strip() or None

    summary: dict = {}
    cfg: dict = {}
    ok = True
    err: str | None = None
    try:
        # Always delete the regenerable tier (adapters, RAG/ledger memory, run
        # logs, staging, sidecars).
        _log("Wiping regenerable reflection state…")
        summary["reflection"] = wipe_reflection_state()
        if wipe_chats:
            # The total-reset tier removes the un-regenerable ground truth too.
            _log("Wiping raw transcripts + reflection review archive…")
            summary["chats"] = wipe_chats_and_archive()
        else:
            # Keeping chats: the adapters that consolidated the archived transcripts
            # were just deleted, so those chats are unreflected again — move them
            # back into the hot working set (otherwise they'd sit in archive/chats,
            # excluded from RAG and from any future reflection).
            _log("Restoring archived transcripts into the hot working set…")
            summary["restored_chats"] = restore_archived_transcripts()
        _log(f"Repointing base model (model_id={model_id or '(unchanged)'})…")
        cfg = reset_server_config(model_id)
    except Exception as e:
        ok = False
        err = str(e)
        _log(f"Wipe error: {e}")

    result: dict = {
        "ok": ok, "summary": summary,
        "model_id": cfg.get("model_id"), "adapter_id": cfg.get("adapter_id"),
    }
    if err:
        result["error"] = err

    # The single machine-readable line the watchdog parses off stdout.
    print(json.dumps(result), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
