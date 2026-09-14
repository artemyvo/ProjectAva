"""Shared path/config resolution for the training package.

Data layout (all persistent state lives under ``server/inference/data/``):

    data/
      hot/                       working set — durable, still moving through consolidation
        chats/                   active transcripts (<ts>.json) + sidecars (<ts>.state.json)
        memory/                  Ava's reflection-produced decisions she still recalls
          rag_memory.jsonl       [ask]/[fact] insert·evict·surface op-log
          weights_persona.jsonl  [persona]/[fact] statements bound for weights
        consolidation/           consolidation bookkeeping (durable, mutable)
          consolidation_anchors.jsonl   anchor ledger
      archive/
        chats/                   fully-destaged transcripts (+ final sidecar) — now in weights
      scratch/                   disposable: regenerated per cycle, safe to delete
        sft_render.jsonl         per-cycle training render
        sft_quarantine.jsonl     compact provenance for rows refused before training

Ordered state home (siblings of ``inference/data``, under ``server/data/``):

    data/
      chats/                     active transcripts (moved out of inference/data/hot/chats)
      til/                       ambient-enculturation (TIL/wander) state
        wander.jsonl             durable, keep-forever wander corpus (trains + feeds chat-RAG)
        snippets/{wander,lookups}/  human-readable provenance (.txt/.json), written on Apply

``prompts/`` stays at the inference root — it is code-adjacent config, not data.
``server_config.json`` lives at the **server root** (``server/server_config.json``):
it is the box's config, read and written by processes on both sides of the
inference/ boundary (inference server, training cycle, snapshot/migrate, the
watchdog's jobs), so it sits above the role directory rather than inside it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent       # server/
_INFERENCE_DIR = _SERVER_DIR / "inference"                 # server/inference/
_DATA_DIR = _INFERENCE_DIR / "data"                        # server/inference/data/

# Reflection artifact filenames.
SFT_RENDER_FILE = "sft_render.jsonl"  # disposable per-cycle training render (train_cycle)
SFT_QUARANTINE_FILE = "sft_quarantine.jsonl"  # rows refused before/while masking


# -- data-tree directories --------------------------------------------------- #

def hot_chats_dir() -> Path:
    """Active transcripts + sidecars (the working set the model reflects on).

    Moved out of ``inference/data`` to the new ordered ``server/data/chats`` root; the
    ``hot_`` prefix in the name is kept for now (renamed in the path-resolver pass).
    """
    return _SERVER_DIR / "data" / "chats"


def archive_chats_dir() -> Path:
    """Retired: the hot/archive split no longer carries training semantics and
    ``archive/chats`` was never produced. Returns a (nonexistent) path under the new
    root so the ``.exists()``-guarded readers/unions still referencing it no-op."""
    return _SERVER_DIR / "data" / "archive" / "chats"


def til_snippets_dir() -> Path:
    """Fetched TIL/wander source texts + their fact protocols, one subdir per kind.

    ``data/til/snippets/{news,wander,lookups}/`` — the fetchers' human-readable provenance
    (``.txt``/``.json``) and, beside each, the ``<stem>.facts.json`` extraction protocol
    ``core.til_facts`` writes. Resolved here rather than re-derived by each reader for the
    same reason ``hot_chats_dir`` is: the inference side reaches it through an injected
    ``configure(til_dir=…)``, which offline callers do not have.
    """
    return _SERVER_DIR / "data" / "til" / "snippets"


def graph_dir() -> Path:
    """The facts tree's output root (``data/graph/``) — see ``graph/`` and FACTS_TREE.md.

    Holds ``tree.json`` (derived, disposable, rebuilt from the ``.facts.json`` protocols)
    and ``aliases.json`` (hand-edited — the one file here that is NOT derived, and so the
    only one worth backing up).
    """
    return _SERVER_DIR / "data" / "graph"


def memory_dir() -> Path:
    """Ava's recallable reflection decisions (rag_memory / weights_persona)."""
    return _DATA_DIR / "hot" / "memory"


def activity_log_path() -> Path:
    """The unified activity journal (``core.activity_log``).

    Here, not only in ``server.py``, because the OFFLINE train cycle appends to the same
    file from its own process — it is the one journal, and unsloth's output belongs in it
    like everything else. Both sides self-locate from ``__file__``, so this is the single
    definition that keeps them pointed at the same file."""
    return _DATA_DIR / "hot" / "activity" / "activity.jsonl"


def consolidation_dir() -> Path:
    """Consolidation bookkeeping (anchor ledger + durable revision records)."""
    return _DATA_DIR / "hot" / "consolidation"


def persona_dir() -> Path:
    """Versioned persona-digest snapshots + the ``current`` pointer.

    The digest is a cumulative, cross-run self-portrait (not a per-run staged delta),
    so it lives in its own live dir and is regenerated/rolled back independently of the
    staging→commit flow."""
    return _DATA_DIR / "hot" / "persona"


def users_dir() -> Path:
    """Per-person user portraits (``<person>.json``) — the user-side mirror of the
    persona digest.

    Lives in its own live dir for the same reason ``persona_dir`` does: a portrait is a
    cumulative, cross-run fold of what Ava has come to make of a person, not a per-run
    staged delta, so it is regenerated independently of the staging→commit flow. One flat
    file per person (the person IS the version axis), no ``current`` pointer — unlike the
    single self-portrait, there is no "which one is live" question to answer."""
    return _DATA_DIR / "hot" / "users"


def prompt_dir() -> Path:
    """Logged-only prompt-mutation op-log (``prompt_deltas.jsonl``).

    The standing-prompt counterfactual is a diagnostic that never promotes or discards,
    so — like the persona digest — it writes to its own live dir regardless of staging,
    giving the Debug tab one stable location to read."""
    return _DATA_DIR / "hot" / "prompt"


def scratch_dir() -> Path:
    """Disposable per-cycle render output — safe to delete anytime."""
    return _DATA_DIR / "scratch"


def staging_dir() -> Path:
    return _DATA_DIR / "hot" / "reflection_staging"


def staging_chats_dir() -> Path:
    return staging_dir() / "chats"


def staging_memory_dir() -> Path:
    return staging_dir() / "memory"


def staging_consolidation_dir() -> Path:
    return staging_dir() / "consolidation"


def default_prompts_dir() -> Path:
    return _INFERENCE_DIR / "prompts"


# -- server config ----------------------------------------------------------- #

def server_config_path() -> Path:
    """Canonical location of the box config: ``server/server_config.json``.

    It moved up out of ``inference/`` (2026-07-28) because it is the *box's* config,
    not the inference role's: training, snapshot/migrate, and the wipe job all read
    or repoint it while inference is down. A checkout still holding the legacy
    ``inference/server_config.json`` is migrated in place on first resolution, so a
    deployed box picks the new layout up on `git pull` + restart with no manual move.
    """
    path = _SERVER_DIR / "server_config.json"
    legacy = _INFERENCE_DIR / "server_config.json"
    if not path.exists() and legacy.exists():
        try:
            os.replace(str(legacy), str(path))
        except Exception:
            return legacy
    return path


def load_server_config() -> dict:
    """Read server/server_config.json (model_id, context_length, consolidation)."""
    path = server_config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically: temp file in the same dir + ``os.replace``.

    A torn write here corrupts the very pointer to the base model + active adapter
    (``server_config.json`` is repointed every train cycle / adapter promotion), so
    the file must be replaced whole-or-not-at-all.
    """
    path = Path(path)
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


def save_server_config(config: dict) -> None:
    """Atomically persist server_config.json (see :func:`atomic_write_text`)."""
    atomic_write_text(server_config_path(), json.dumps(config, indent=2) + "\n")
