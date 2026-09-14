"""Immutable forensic build snapshot — REBUILD.md §7.

Each build (promoted OR rejected) materializes a self-contained snapshot dir under
``server/models/snapshots/<build_id>/`` holding **copies** (never references into live,
mutable state) of everything needed to reproduce or triage the build: the exact rendered
training rows, any refused-row quarantine journal, the wander/news records used, the active
persona digest, and a
``build_meta.json`` (server config + effective wall-clock/decay params + base_lr + seed +
built_at + outcome + adapter pointer). Because it owns frozen copies, editing a live
sidecar later can't corrupt it — the ``reflections/`` archive is the mutable build
*recipe*; a snapshot is a frozen *state* (the §7 immutability-vs-debuggability split). On
a probe failure it is the investigation packet — the triage protocol opens it first.

Best-effort by contract: any failure logs and returns None — a snapshot must never block
or fail a build. Model-free / GPU-free, so it self-tests without hardware.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

SNAPSHOTS_DIRNAME = "snapshots"


def snapshots_root(models_dir) -> Path:
    """``server/models/snapshots`` — the parent of every per-build snapshot dir."""
    return Path(models_dir) / SNAPSHOTS_DIRNAME


def write_snapshot(*, models_dir, build_id: str, meta: dict,
                   render_path: Optional[Path] = None,
                   quarantine_path: Optional[Path] = None,
                   wander: Optional[list] = None,
                   persona_digest: Optional[dict] = None) -> Optional[Path]:
    """Materialize the snapshot dir for *build_id*; return its path (or None on any error).

    *meta*: the ``build_meta.json`` contents — the caller assembles it (config + params +
        outcome) so this module stays dumb and testable.
    *render_path*: the disposable ``scratch/sft_render.jsonl`` — copied in **before** the
        caller deletes it (that copy is what makes a build reproducible/triageable).
    *quarantine_path*: compact provenance for rows refused before training; copied when
        present so the snapshot explains the difference between corpus and trained rows.
    *wander*: the one-shot wander/news records trained this build.
    *persona_digest*: the active digest dict (from ``latest_digest``), materialized whole.
    """
    try:
        snap = snapshots_root(models_dir) / str(build_id)
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "build_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (snap / "wander.json").write_text(
            json.dumps(wander or [], ensure_ascii=False, indent=2), encoding="utf-8")
        if persona_digest is not None:
            (snap / "persona_digest.json").write_text(
                json.dumps(persona_digest, ensure_ascii=False, indent=2), encoding="utf-8")
        if render_path is not None and Path(render_path).exists():
            shutil.copy2(Path(render_path), snap / "sft_render.jsonl")
        if quarantine_path is not None and Path(quarantine_path).exists():
            shutil.copy2(Path(quarantine_path), snap / "sft_quarantine.jsonl")
        return snap
    except Exception as exc:  # never block a build on a snapshot failure
        print(f"build snapshot skipped ({build_id}): {exc}", flush=True)
        return None
