"""Per-reflection archive tree for human review and rollback.

Each completed Sleep run may snapshot its run log, committed artifact deltas, current
persona digest, and (after offline training) produced adapter under
``server/reflections/<run_id>/``.  The archive intentionally has no manifest or replay
index: wall-clock decay makes the old ordered-replay lineage obsolete.

Archiving is best-effort and must never break reflection or training.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.chat_sidecar import iter_chat_products

REFLECTIONS_DIRNAME = "reflections"
_SERVER_DIR = Path(__file__).resolve().parents[2]
_RUN_LOG_SUFFIXES = (
    ".meta.json", ".events.jsonl", ".summary.json", ".report.json",
)


def _resolve_server_dir(server_dir: Optional[Path]) -> Path:
    return Path(server_dir) if server_dir is not None else _SERVER_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reflections_root(server_dir: Optional[Path] = None) -> Path:
    """Return the review archive root, ``server/reflections/``."""
    return _resolve_server_dir(server_dir) / REFLECTIONS_DIRNAME


def run_archive_dir(run_id: str, server_dir: Optional[Path] = None) -> Path:
    return reflections_root(server_dir) / run_id


def _staging_subdirs(staging_dir: Path) -> dict[str, Path]:
    staging_dir = Path(staging_dir)
    return {
        "memory": staging_dir / "memory",
        "consolidation": staging_dir / "consolidation",
        "chats": staging_dir / "chats",
        "archive_chats": staging_dir / "archive" / "chats",
    }


def _copy_run_log(runs_dir: Path, dest_run_dir: Path, run_id: str) -> list[str]:
    dest_run_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for suffix in _RUN_LOG_SUFFIXES:
        src = Path(runs_dir) / f"{run_id}{suffix}"
        if src.exists():
            try:
                shutil.copy2(src, dest_run_dir / src.name)
                copied.append(src.name)
            except Exception:
                pass
    return copied


def _copy_staging_artifacts(staging_dir: Path, dest_artifacts: Path) -> dict:
    sub = _staging_subdirs(staging_dir)
    summary: dict = {}
    dest_artifacts.mkdir(parents=True, exist_ok=True)

    for name, src in (
        ("rag_memory.jsonl", sub["memory"] / "rag_memory.jsonl"),
        ("weights_persona.jsonl", sub["memory"] / "weights_persona.jsonl"),
        ("consolidation_anchors.jsonl", sub["consolidation"] / "consolidation_anchors.jsonl"),
    ):
        if src.exists() and src.stat().st_size > 0:
            try:
                shutil.copy2(src, dest_artifacts / name)
                with open(src, encoding="utf-8") as fh:
                    summary[name] = sum(1 for _ in fh)
            except Exception:
                pass

    chats_src = sub["chats"]
    if chats_src.exists():
        chats_dest = dest_artifacts / "chats"
        count = 0
        # Every reflection product this run committed for a chat, not the state sidecar
        # alone — the archive's contract is "what this run wrote", and the summary is the
        # one artifact whose loss is not recoverable by re-reflecting (it is the chat's
        # only representation in RAG once the verbatim has aged out).
        for src in iter_chat_products(chats_src):
            try:
                chats_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, chats_dest / src.name)
                count += 1
            except Exception:
                pass
        if count:
            summary["chat_files"] = count

    archive_src = sub["archive_chats"]
    if archive_src.exists():
        archive_dest = dest_artifacts / "archive" / "chats"
        names: list[str] = []
        for src in archive_src.glob("*"):
            if not src.is_file():
                continue
            try:
                archive_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, archive_dest / src.name)
                names.append(src.name)
            except Exception:
                pass
        if names:
            summary["archived_chats"] = names

    return summary


def _copy_persona_digest(persona_dir: Path, dest_persona_dir: Path) -> dict:
    src = Path(persona_dir) / "digest.json"
    if not src.is_file():
        return {}
    try:
        digest = json.loads(src.read_text(encoding="utf-8"))
        dest_persona_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_persona_dir / "digest.json")
        return {
            "file": "digest.json",
            "version": digest.get("version"),
            "run_id": digest.get("run_id"),
        }
    except Exception:
        return {}


def _copy_user_portraits(users_dir: Path, dest_users_dir: Path) -> dict:
    """Snapshot the per-person user portraits as of this run.

    The user-side counterpart of :func:`_copy_persona_digest`, and archived for the same
    reason: a portrait is prompt-injected on every turn with that person, so "what was she
    working from when she answered like that?" needs an answer that survives the next
    regeneration. Copies whatever portraits exist (not only ones this run rewrote) — the
    archive's contract is a snapshot of the live state at commit time, and a portrait left
    unchanged this run is still part of that state.
    """
    src = Path(users_dir)
    if not src.is_dir():
        return {}
    try:
        names = sorted(p.name for p in src.glob("*.json"))
        if not names:
            return {}
        dest_users_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(src / name, dest_users_dir / name)
        return {"files": names, "count": len(names)}
    except Exception:
        return {}


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def archive_reflection(
    *,
    run_id: str,
    runs_dir: Path,
    server_dir: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    persona_dir: Optional[Path] = None,
    users_dir: Optional[Path] = None,
    counts: Optional[dict] = None,
    source: str = "",
) -> dict:
    """Snapshot a just-committed run under ``reflections/<run_id>/``.

    Call before discarding staging so its deltas remain available. Returns an in-memory
    summary for logging/tests; no manifest or replay metadata is persisted.
    """
    try:
        run_dir = run_archive_dir(run_id, server_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = _copy_run_log(runs_dir, run_dir / "run", run_id)
        artifacts = (
            _copy_staging_artifacts(staging_dir, run_dir / "artifacts")
            if staging_dir is not None else {}
        )
        persona = (
            _copy_persona_digest(persona_dir, run_dir / "persona")
            if persona_dir is not None else {}
        )
        users = (
            _copy_user_portraits(users_dir, run_dir / "users")
            if users_dir is not None else {}
        )
        return {
            "run_id": run_id,
            "archived_at": _utc_now(),
            "source": source,
            "commit_counts": counts or {},
            "artifacts": artifacts,
            "persona": persona,
            "users": users,
            "run_log": run_log,
        }
    except Exception:
        return {}


def archive_adapter(
    *,
    run_id: str,
    adapter_dir: Path,
    server_dir: Optional[Path] = None,
) -> dict:
    """Copy a produced adapter into ``reflections/<run_id>/adapter/``."""
    try:
        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            return {}
        dest = run_archive_dir(run_id, server_dir) / "adapter"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_dir, dest)
        return {
            "source_adapter_id": str(adapter_dir),
            "name": adapter_dir.name,
            "copied_at": _utc_now(),
            "size_bytes": _dir_size_bytes(dest),
        }
    except Exception:
        return {}
