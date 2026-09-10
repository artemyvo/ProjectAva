"""Active-persona path resolver — the single source of truth for "which persona is
live, and where are its parts".

A **persona** is a frozen, self-contained snapshot under ``server/data/persona/<run_id>/``
(the layout ``snapshot_state`` emits): ``server_config.json`` (with a RELATIVE
``adapter_id``), ``digest.json`` (the self-portrait), ``prompts/``, ``data/`` (the frozen
RAG-source state), ``models/adapter-<run_id>/``, and ``training/<build_id>/``. A
``current.json`` pointer at the persona root names the active one — so the persona dir
itself IS the version (no per-file versioning inside).

This module holds ONLY resolution: it reads/writes the pointer and derives member paths.
It is stdlib-only and GPU-free (importable from the inference server, the training cycle,
and the standalone snapshot/wipe scripts alike) and never mutates persona contents.

``active_*`` returns ``None`` when no pointer is set yet, so the module is safe to land
before any persona is activated — consumers choose their own legacy fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# inference/core/persona_paths.py → parents[2] == server/
_SERVER_DIR = Path(__file__).resolve().parents[2]

CURRENT_POINTER = "current.json"          # server/data/persona/current.json
DIGEST_FILE = "digest.json"               # <persona>/digest.json (flat, unversioned)
CONFIG_FILE = "server_config.json"        # <persona>/server_config.json
PROMPTS_DIRNAME = "prompts"               # <persona>/prompts/
DATA_DIRNAME = "data"                     # <persona>/data/


def personas_root() -> Path:
    """``server/data/persona`` — the parent of every per-run persona dir + the pointer."""
    return _SERVER_DIR / "data" / "persona"


def current_pointer_path() -> Path:
    return personas_root() / CURRENT_POINTER


def persona_dir_for(run_id: str) -> Path:
    """``server/data/persona/<run_id>`` (no existence check)."""
    return personas_root() / str(run_id)


def active_run_id() -> Optional[str]:
    """The run_id named by ``current.json``, or ``None`` if unset/unreadable."""
    ptr = current_pointer_path()
    if not ptr.is_file():
        return None
    try:
        rid = json.loads(ptr.read_text(encoding="utf-8")).get("run_id")
    except Exception:
        return None
    return str(rid) if rid else None


def active_persona_dir() -> Optional[Path]:
    """The active persona's dir, or ``None`` when the pointer is unset or dangling."""
    rid = active_run_id()
    if not rid:
        return None
    d = persona_dir_for(rid)
    return d if d.is_dir() else None


def set_active(run_id: str) -> Path:
    """Atomically repoint ``current.json`` at *run_id*. Returns the pointer path.

    A torn write here would orphan the live persona, so the replace is whole-or-nothing
    (reuses the training package's ``atomic_write_text``)."""
    from training.reflections_path import atomic_write_text
    ptr = current_pointer_path()
    atomic_write_text(ptr, json.dumps({"run_id": str(run_id)}, indent=2) + "\n")
    return ptr


# -- member accessors -------------------------------------------------------- #
# Each takes an explicit *persona_dir* or defaults to the active one; they raise if
# neither is available, so a caller that wants a legacy fallback checks
# ``active_persona_dir()`` first.

def _resolve(persona_dir: Optional[Path]) -> Path:
    if persona_dir is not None:
        return Path(persona_dir)
    d = active_persona_dir()
    if d is None:
        raise RuntimeError("no active persona (server/data/persona/current.json unset)")
    return d


def digest_path(persona_dir: Optional[Path] = None) -> Path:
    """``<persona>/digest.json`` — the one self-portrait for this persona."""
    return _resolve(persona_dir) / DIGEST_FILE


def config_path(persona_dir: Optional[Path] = None) -> Path:
    return _resolve(persona_dir) / CONFIG_FILE


def prompts_dir(persona_dir: Optional[Path] = None) -> Path:
    return _resolve(persona_dir) / PROMPTS_DIRNAME


def data_dir(persona_dir: Optional[Path] = None) -> Path:
    """The persona's frozen ``data/`` (RAG-source state: memory, consolidation, ...)."""
    return _resolve(persona_dir) / DATA_DIRNAME


def load_config(persona_dir: Optional[Path] = None) -> dict:
    """Read the persona's ``server_config.json`` (``{}`` if missing/unreadable)."""
    p = config_path(persona_dir)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def adapter_dir(persona_dir: Optional[Path] = None) -> Optional[Path]:
    """The persona's adapter dir as an ABSOLUTE path, resolving the config's RELATIVE
    ``adapter_id`` against the persona root. ``None`` when the config names no adapter.

    (Absolute so the inference backend's ``os.path.exists`` / native load path is
    unaffected — a snapshot stores ``adapter_id`` relative for portability.)"""
    d = _resolve(persona_dir)
    adapter_id = (load_config(d).get("adapter_id") or "").strip()
    if not adapter_id:
        return None
    p = Path(adapter_id)
    return p if p.is_absolute() else (d / p)


# -- GPU-free self-test ------------------------------------------------------ #

def _selftest() -> None:
    import tempfile

    # No pointer → active_* is None (safe pre-activation state).
    root = personas_root()
    if not current_pointer_path().is_file():
        assert active_run_id() is None and active_persona_dir() is None

    # Build a throwaway persona layout and exercise resolution against it directly
    # (persona_dir passed explicitly — no dependence on the real pointer).
    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td) / "20990101_000000"
        (pdir / "models" / "adapter-x").mkdir(parents=True)
        (pdir / "prompts").mkdir()
        (pdir / "data").mkdir()
        (pdir / CONFIG_FILE).write_text(
            json.dumps({"model_id": "m", "adapter_id": "models/adapter-x"}), encoding="utf-8")
        (pdir / DIGEST_FILE).write_text("{}", encoding="utf-8")

        assert digest_path(pdir) == pdir / "digest.json"
        assert config_path(pdir) == pdir / "server_config.json"
        assert data_dir(pdir) == pdir / "data"
        assert prompts_dir(pdir) == pdir / "prompts"
        assert load_config(pdir)["model_id"] == "m"
        ad = adapter_dir(pdir)
        assert ad is not None and ad.is_absolute() and ad == pdir / "models" / "adapter-x"

        # No adapter_id → None.
        (pdir / CONFIG_FILE).write_text(json.dumps({"model_id": "m"}), encoding="utf-8")
        assert adapter_dir(pdir) is None

    assert persona_dir_for("abc") == root / "abc"
    print("persona_paths selftest OK")


if __name__ == "__main__":
    _selftest()
