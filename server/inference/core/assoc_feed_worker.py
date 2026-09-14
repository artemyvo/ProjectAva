"""The assoc feed's sync + rebuild in its own process — `assoc_bridge.run_feed_blocking`'s
worker (2026-09-11).

Why a process and not a thread: a library rebuild is minutes of pure-Python work
(lemmatization, the codebook, dedup, PageRank) that never releases the GIL, so run on the
inference server's executor thread it starved the asyncio loop for the whole wake — a
client connecting meanwhile timed out on the WebSocket opening handshake. A generation
does not do this (its time is in CUDA kernels, GIL released), which is why the feed was
the first job to need this. The library never loads the LLM, so the worker needs nothing
from the parent but paths and config; the model-bound relation pass stays in the parent.

Protocol: one JSON object on stdin (`chats_dirs`, `til_snippets_dir`, `root`,
`prompts_dir`, `assoc` = the raw config block, `force_rebuild`); progress on stdout as it
happens (the parent re-prints every line, so the activity journal sees them); the report
as the LAST line, `FEED_RESULT_MARKER` + JSON. Never exits without that line short of a
kill: an exception is a report with `error`.

Run by the parent as ``python -m core.assoc_feed_worker`` with cwd = ``server/inference``;
never by hand.
"""
from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    from core import assoc_bridge   # also puts the repo root on sys.path for `assoc`
    rep: dict
    try:
        payload = json.load(sys.stdin)
        assoc_bridge.configure(
            chats_dirs=payload.get("chats_dirs") or [],
            til_snippets_dir=payload.get("til_snippets_dir"),
            root=payload["root"], prompts_dir=payload["prompts_dir"],
            load_config=lambda: {"assoc": payload.get("assoc") or {}},
        )
        cfg = assoc_bridge.config()
        lib = assoc_bridge.library(cfg)
        if lib is None:
            rep = {"error": "library unavailable in the feed worker (assoc package or embedder missing)"}
        else:
            rep = assoc_bridge.feed_sync_and_rebuild(lib, cfg, force_rebuild=bool(payload.get("force_rebuild")),
                                                    log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rep = {"error": f"{type(e).__name__}: {e}"}
    sys.stdout.flush()
    print(assoc_bridge.FEED_RESULT_MARKER + json.dumps(rep, ensure_ascii=False, default=str), flush=True)
    return 0 if not rep.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
