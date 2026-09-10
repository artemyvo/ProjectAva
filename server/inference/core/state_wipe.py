"""GPU-free filesystem helpers for Ava's destructive state-wipe job."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from core.chat_sidecar import is_chat_session_json

_SERVER_DIR = Path(__file__).resolve().parents[2]
_MEMORY_FILES = (
    "rag_memory.jsonl", "weights_persona.jsonl", "wiki_budget.json",
    "wander_log.jsonl", "user_tokens.json",
)
_CONSOLIDATION_FILES = ("consolidation_anchors.jsonl",)


def _paths(server_dir: Optional[Path] = None) -> dict[str, Path]:
    server = Path(server_dir) if server_dir is not None else _SERVER_DIR
    inference = server / "inference"
    data = inference / "data"
    # The box config lives at the server root (moved up out of inference/ on
    # 2026-07-28); a not-yet-migrated checkout is still repointed in place.
    config = server / "server_config.json"
    if not config.exists() and (inference / "server_config.json").exists():
        config = inference / "server_config.json"
    return {
        "server": server,
        "config": config,
        "hot_chats": server / "data" / "chats",
        "archive_chats": server / "data" / "archive" / "chats",
        "memory": data / "hot" / "memory",
        "consolidation": data / "hot" / "consolidation",
        "persona": data / "hot" / "persona",
        "users": data / "hot" / "users",
        "wander": data / "hot" / "wander",
        "prompt": data / "hot" / "prompt",
        "scratch": data / "scratch",
        "runs": data / "hot" / "reflection_runs",
        "staging": data / "hot" / "reflection_staging",
        "reachout": data / "hot" / "reachout",
        "models": server / "models",
        "reflections": server / "reflections",
    }


def _rm(path: Path, summary: dict, key: str) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
            summary[key] = summary.get(key, 0) + 1
        elif path.exists():
            path.unlink()
            summary[key] = summary.get(key, 0) + 1
    except Exception as exc:
        summary.setdefault("failed", []).append(f"{path}: {exc}")
        print(f"WARNING: wipe could not remove {path}: {exc}", file=sys.stderr, flush=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write *data* as pretty JSON to *path* atomically (temp file + rename)."""
    text = json.dumps(data, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _prune_or_remove_sidecar(path: Path, summary: dict) -> None:
    """Delete a chat ``.state.json`` sidecar, but preserve human-validated exchanges.

    A ``locked`` exchange record is an operator-reviewed, hand-authored target from the
    Training review tab. It is **un-regenerable** — the whole point of the lock is that
    re-reflection (and revisit) must never re-derive it from the original, possibly
    corrupt, transcript. On a keep-chats wipe the restored chats re-reflect on the next
    Sleep, so blindly deleting the sidecar would let that re-reflection overwrite the
    operator's reviewed target with a fresh pass over the poison. Instead: if the sidecar
    holds any locked exchanges, rewrite it down to just those records (dropping
    ``reflected_at`` so the chat's *other* exchanges re-reflect clean); otherwise delete it
    as before. Non-locked reflection state stays regenerable and is discarded.
    """
    from core.chat_sidecar import SCHEMA_VERSION, session_name_from_sidecar

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    locked: dict = {}
    if isinstance(data, dict):
        exchanges = data.get("exchanges")
        if isinstance(exchanges, dict):
            locked = {k: v for k, v in exchanges.items()
                      if isinstance(v, dict) and v.get("locked")}
    if not locked:
        _rm(path, summary, "chat_artifacts")
        return
    pruned = {
        "schema_version": data.get("schema_version", SCHEMA_VERSION),
        "source_session": data.get("source_session") or session_name_from_sidecar(path),
        "exchanges": locked,
    }
    try:
        _atomic_write_json(path, pruned)
        summary["preserved_locked_sidecars"] = summary.get("preserved_locked_sidecars", 0) + 1
        summary["preserved_locked_exchanges"] = summary.get("preserved_locked_exchanges", 0) + len(locked)
    except Exception as exc:
        summary.setdefault("failed", []).append(f"{path}: {exc}")
        print(f"WARNING: wipe could not prune {path}: {exc}", file=sys.stderr, flush=True)


def wipe_reflection_state(server_dir: Optional[Path] = None) -> dict:
    """Delete regenerable reflection state while preserving transcripts and review archives.

    Chat sidecars are part of the regenerable layer and are wiped too — EXCEPT
    human-validated (locked) exchange records, which hold hand-authored,
    operator-reviewed targets that no re-reflection can reproduce; those survive
    (see ``_prune_or_remove_sidecar``).
    """
    p = _paths(server_dir)
    summary: dict = {}
    for chats in (p["hot_chats"], p["archive_chats"]):
        if chats.exists():
            # Sidecars carry the regenerable reflection layer, so they are wiped — EXCEPT
            # human-validated (locked) exchange records, which are hand-authored and must
            # survive (see _prune_or_remove_sidecar). Other artifacts (.shareml.json) hold
            # nothing un-regenerable and are removed outright.
            for item in chats.glob("*.state.json"):
                _prune_or_remove_sidecar(item, summary)
            for item in chats.glob("*.shareml.json"):
                _rm(item, summary, "chat_artifacts")
    for name in _MEMORY_FILES:
        _rm(p["memory"] / name, summary, "memory_files")
    for name in _CONSOLIDATION_FILES:
        _rm(p["consolidation"] / name, summary, "consolidation_files")
    # `users` (the per-person portraits) is regenerable from the [impression]/[fact]
    # op-log, and is wiped WITH that op-log rather than after it: a portrait left standing
    # over deleted evidence would be a ghost — injected into every turn, and no longer
    # derivable from or correctable by anything on the box.
    for key in ("runs", "staging", "persona", "users", "wander", "prompt"):
        _rm(p[key], summary, f"{key}_dir")
    if p["models"].exists():
        for item in p["models"].iterdir():
            if item.is_dir() and (item.name == "candidate" or item.name.startswith("adapter-")):
                _rm(item, summary, "adapters")
    return summary


def wipe_chats_and_archive(server_dir: Optional[Path] = None) -> dict:
    """Delete all raw transcripts, review archives, and disposable render scratch."""
    p = _paths(server_dir)
    summary: dict = {}
    for chats in (p["hot_chats"], p["archive_chats"]):
        if chats.exists():
            for item in chats.glob("*.json"):
                _rm(item, summary, "transcripts")
    p["hot_chats"].mkdir(parents=True, exist_ok=True)
    # The reach-out tombstones stand in for deleted transcripts (core.reachout_gate), so
    # they belong to the corpus and go with it. Left behind, they would back her off — and
    # keep quoting openers into check-in's prompt — against a conversation history that no
    # longer exists. Deliberately tied to the chats, not to the regenerable-state wipe:
    # while the corpus survives, the tombstones are part of what it records.
    _rm(p["reachout"], summary, "reachout_tombstones")
    if p["reflections"].exists():
        for child in p["reflections"].iterdir():
            if child.name != "README.md":
                _rm(child, summary, "archive_entries")
    _rm(p["scratch"], summary, "scratch")
    return summary


def restore_archived_transcripts(server_dir: Optional[Path] = None) -> list[str]:
    """Move any legacy archived transcripts back into the active chat directory."""
    p = _paths(server_dir)
    archive = p["archive_chats"]
    hot = p["hot_chats"]
    if not archive.exists():
        return []
    hot.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for src in archive.glob("*.json"):
        if not is_chat_session_json(src):
            continue
        try:
            shutil.move(str(src), str(hot / src.name))
            moved.append(src.name)
            # A preserved locked sidecar (wipe_reflection_state pruned but kept it) must
            # travel with its transcript, or the re-reflection in hot/ won't find the lock
            # and will re-derive the operator's hand-authored target from the transcript.
            sidecar = archive / (src.stem + ".state.json")
            if sidecar.exists():
                shutil.move(str(sidecar), str(hot / sidecar.name))
        except Exception as exc:
            print(f"WARNING: could not restore transcript {src.name}: {exc}",
                  file=sys.stderr, flush=True)
    return moved


def reset_server_config(
    model_id_override: Optional[str] = None,
    server_dir: Optional[Path] = None,
) -> dict:
    """Clear the active adapter and optionally choose a new base model."""
    path = _paths(server_dir)["config"]
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        config = {}
    config["adapter_id"] = None
    if model_id_override:
        config["model_id"] = model_id_override
    text = json.dumps(config, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"model_id": config.get("model_id"), "adapter_id": config.get("adapter_id")}
