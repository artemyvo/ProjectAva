#!/usr/bin/env python3
"""Headless reflection runner — consolidation and revision with no connected client.

Run from the server/ directory using the server's venv:

    cd server
    .venv/bin/python reflection_run.py --all-pending
    .venv/bin/python reflection_run.py --session 20250610_142300.json --stage reflection
    .venv/bin/python reflection_run.py --latest 3
    .venv/bin/python reflection_run.py --all-pending --overrides overrides.json

Session selection (at least one required):
  --session FILENAME     Reflect on a specific session (repeatable)
  --all-pending          All sessions with at least one unrevised exchange
  --latest N             The N most-recently modified sessions (by mtime)

Run control:
  --stage STAGE          Pipeline stage to execute (reflection | diff |
                         merge-rag | commit-training | apply | discard | all).
                         Defaults to "all". A "reflection" run writes its
                         artifacts to the staging workspace, not production.
  --overrides FILE       JSON file of reflection overrides (temperature,
                         prompts, sampling, etc. — same schema as the
                         start_reflection_run WebSocket message).

Cron usage (runs every night at 2 AM):

    0 2 * * * cd /srv/ava/server && .venv/bin/python reflection_run.py \\
        --all-pending >> /var/log/ava-reflection.log 2>&1

The run_id is a timestamp (YYYYMMDD_HHMMSS). Durable artifacts are written
under data/hot/ (rag_memory, sidecars) exactly as when the UI
triggers a reflection run. Run metadata is always written under
data/hot/reflection_runs/ regardless of the selected stage.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── make inference/ importable from server/ ──────────────────────────────── #
_SERVER_DIR = Path(__file__).resolve().parent
_INFERENCE_DIR = _SERVER_DIR / "inference"
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

# Reduce CUDA allocator fragmentation — same setting the WebSocket server makes,
# must be set before torch imports CUDA at model-load time. Both names; the LEGACY
# one is load-bearing — torch 2.9.1 verifiably ignores the new name. The env is a
# request only: main() probes + runtime-forces the mode before the model load
# (core/alloc_guard.ensure_expandable_segments, the shared never-trust-the-env probe).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── data paths ────────────────────────────────────────────────────────────── #
_DATA_DIR          = _INFERENCE_DIR / "data"
_CHATS_DIR         = _DATA_DIR / "hot" / "chats"
_MEMORY_DIR        = _DATA_DIR / "hot" / "memory"
_CONSOLIDATION_DIR = _DATA_DIR / "hot" / "consolidation"
_RUNS_DIR          = _DATA_DIR / "hot" / "reflection_runs"

_STAGING_DIR               = _DATA_DIR / "hot" / "reflection_staging"
_STAGING_CHATS_DIR         = _STAGING_DIR / "chats"
_STAGING_MEMORY_DIR        = _STAGING_DIR / "memory"
_STAGING_CONSOLIDATION_DIR = _STAGING_DIR / "consolidation"
_STAGING_ARCHIVE_DIR       = _STAGING_DIR / "archive"


# ── config ────────────────────────────────────────────────────────────────── #

def _server_config_path() -> Path:
    """``server/server_config.json`` — the box config (moved up out of ``inference/``
    on 2026-07-28), migrating a legacy checkout in place on first resolution."""
    path = _SERVER_DIR / "server_config.json"
    legacy = _INFERENCE_DIR / "server_config.json"
    if not path.exists() and legacy.exists():
        try:
            os.replace(str(legacy), str(path))
        except Exception:
            return legacy
    return path


def _load_server_config() -> dict:
    path = _server_config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not read server_config.json: {e}", flush=True)
    return {}


def _save_server_config(config: dict) -> None:
    # Atomic write (temp + os.replace): a torn write corrupts the only pointer to
    # the base model + active adapter.
    import tempfile
    path = _server_config_path()
    text = json.dumps(config, indent=2) + "\n"
    try:
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
    except Exception as e:
        print(f"Warning: could not save server_config.json: {e}", flush=True)


import shutil
import difflib

def append_file_to_file(src: Path, dest: Path) -> int:
    if not src.exists():
        return 0
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        content = dest.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            with open(dest, "a", encoding="utf-8") as fh:
                fh.write("\n")
    with open(dest, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.strip() + "\n")
    return len(lines)

def run_stage_diff() -> None:
    print("\n🔍 Producing Sleep Pipeline Delta Diff...")
    print("═" * 60)
    
    # 1. RAG memory delta
    rag_staged = _STAGING_MEMORY_DIR / "rag_memory.jsonl"
    if rag_staged.exists() and rag_staged.stat().st_size > 0:
        print("📝 Staged RAG Memory Changes (rag_memory.jsonl):")
        print("─" * 60)
        print(rag_staged.read_text(encoding="utf-8").strip())
        print("═" * 60)
        
    # 2. Weights delta
    weights_staged = _STAGING_MEMORY_DIR / "weights_persona.jsonl"
    if weights_staged.exists() and weights_staged.stat().st_size > 0:
        print("📝 Staged Weights Persona Changes (weights_persona.jsonl):")
        print("─" * 60)
        print(weights_staged.read_text(encoding="utf-8").strip())
        print("═" * 60)
        
    # 3. Ledger delta
    ledger_staged = _STAGING_CONSOLIDATION_DIR / "consolidation_anchors.jsonl"
    if ledger_staged.exists() and ledger_staged.stat().st_size > 0:
        print("📝 Staged Consolidation Ledger Changes (consolidation_anchors.jsonl):")
        print("─" * 60)
        print(ledger_staged.read_text(encoding="utf-8").strip())
        print("═" * 60)
        
    # 4. Chat sidecars diff
    if _STAGING_CHATS_DIR.exists():
        staged_sidecars = list(_STAGING_CHATS_DIR.glob("*.state.json"))
        if staged_sidecars:
            print("📝 Staged Chat Sidecar Diffs:")
            print("─" * 60)
            for sp in staged_sidecars:
                lp = _CHATS_DIR / sp.name
                sp_text = sp.read_text(encoding="utf-8").splitlines()
                if lp.exists():
                    lp_text = lp.read_text(encoding="utf-8").splitlines()
                else:
                    lp_text = []
                diff = list(difflib.unified_diff(
                    lp_text, sp_text,
                    fromfile=f"live/chats/{lp.name}", tofile=f"staged/chats/{sp.name}",
                    lineterm=""
                ))
                if diff:
                    print("\n".join(diff))
                else:
                    print(f"No changes to {sp.name} (staged version matches live)")
                print("─" * 60)
            print("═" * 60)

def run_stage_merge_rag() -> dict:
    print("Merging RAG memory delta to live RAG log...")
    rag_staged = _STAGING_MEMORY_DIR / "rag_memory.jsonl"
    rag_live = _MEMORY_DIR / "rag_memory.jsonl"
    appended = append_file_to_file(rag_staged, rag_live)
    print(f"Merged {appended} RAG memory line(s) into {rag_live}.")

    # Rebuild live RAG index
    from core.rag_engine import RagEngine
    rag = RagEngine(
        _CHATS_DIR, _INFERENCE_DIR / "prompts",
        memory_dir=_MEMORY_DIR, consolidation_dir=_CONSOLIDATION_DIR,
    )
    print("Rebuilding live RAG index...")
    rag.build_index_async()
    return {"rag_memory_lines": appended}

def run_stage_commit_training() -> dict:
    print("Merging weights persona delta to live...")
    w_staged = _STAGING_MEMORY_DIR / "weights_persona.jsonl"
    w_live = _MEMORY_DIR / "weights_persona.jsonl"
    w_appended = append_file_to_file(w_staged, w_live)
    print(f"Merged {w_appended} weights persona line(s) into {w_live}.")

    print("Merging consolidation ledger anchors delta to live...")
    l_staged = _STAGING_CONSOLIDATION_DIR / "consolidation_anchors.jsonl"
    l_live = _CONSOLIDATION_DIR / "consolidation_anchors.jsonl"
    l_appended = append_file_to_file(l_staged, l_live)
    print(f"Merged {l_appended} ledger line(s) into {l_live}.")

    print("Committing staged sidecars to live...")
    sidecars_copied = 0
    archived_count = 0
    if _STAGING_CHATS_DIR.exists():
        for sp in _STAGING_CHATS_DIR.glob("*.state.json"):
            lp = _CHATS_DIR / sp.name
            shutil.copy2(sp, lp)
            sidecars_copied += 1
    print(f"Committed {sidecars_copied} sidecar file(s) to live chats.")

    # Check if there are staged archived chats to commit
    staged_archive = _STAGING_ARCHIVE_DIR / "chats"
    live_archive = _DATA_DIR / "archive" / "chats"
    if staged_archive.exists():
        live_archive.mkdir(parents=True, exist_ok=True)
        for f in staged_archive.glob("*"):
            # Copy to archive
            shutil.copy2(f, live_archive / f.name)
            # Delete from live hot chats
            live_hot = _CHATS_DIR / f.name
            if live_hot.exists():
                live_hot.unlink()
            archived_count += 1
        print(f"Committed {archived_count} archived file(s) to live archive and cleaned from live hot chats.")

    return {
        "weights_persona_lines": w_appended,
        "ledger_lines": l_appended,
        "sidecars_copied": sidecars_copied,
        "archived": archived_count,
    }

def run_stage_apply() -> dict:
    counts: dict = {}
    # 1. Merge RAG
    counts.update(run_stage_merge_rag())
    # 2. Commit training / sidecars
    counts.update(run_stage_commit_training())

    # 3. Promote candidate adapter if present
    candidate_dir = _SERVER_DIR.parent / "models" / "candidate"
    if candidate_dir.exists():
        new_adapter_id = _SERVER_DIR.parent / "models" / f"adapter-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Promoting candidate model adapter to {new_adapter_id}...")
        new_adapter_id.parent.mkdir(parents=True, exist_ok=True)
        os.rename(candidate_dir, new_adapter_id)

        # Update config
        config = _load_server_config()
        config["adapter_id"] = str(new_adapter_id)
        _save_server_config(config)
        print("Updated server_config.json adapter_id to point to new adapter weights.")
        counts["promoted_adapter"] = str(new_adapter_id)

    # 4. Clean up staging
    run_stage_discard()
    print("Sleep staging workspace committed successfully.")
    return counts

def run_stage_discard() -> dict:
    print("Cleaning up Sleep staging workspace...")
    if _STAGING_DIR.exists():
        shutil.rmtree(_STAGING_DIR)
    candidate_dir = _SERVER_DIR.parent / "models" / "candidate"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    print("Staging directories and candidate model weights deleted.")
    return {}


# ── response cleaning (mirror of server.py _clean_reflect_response) ────────
# Minimal for reflection — strips model format tokens, normalises Gemma-4
# channel tags to <think>…</think>. No turn-leak detection (reflection output
# intentionally contains User:/Me: lines).

def _clean_reflect_response(raw: str) -> str:
    if "<|channel>" in raw:
        raw = re.sub(
            r"<\|channel>thought\n?(.*?)\n?<channel\|>",
            r"<think>\1</think>",
            raw, flags=re.DOTALL,
        )
        raw = re.sub(r"<\|channel>[^\n]*\n?|<channel\|>", "", raw)
    return re.sub(
        r"<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?",
        "", raw,
    ).strip()


def _clean_ideal_response(raw: str, model_id: str) -> str:
    """Headless call into the same normal-chat cleaner used by the live server."""
    from core.generation import clean_dialogue_response
    return clean_dialogue_response(raw, model_id)


def _resolve_max_new_tokens(setting: str, available: int) -> int:
    setting = str(setting).strip()
    if available <= 1:
        return 1
    try:
        if setting.endswith("%"):
            pct = float(setting[:-1].strip())
            if pct <= 0:
                raise ValueError
            return max(1, min(available, int(available * pct / 100.0)))
        return max(1, min(available, int(setting)))
    except (TypeError, ValueError):
        return max(1, min(available, int(available * 0.75)))


# ── pending-session detection (revision-artifact presence rule) ───────────── #

def _session_has_pending_revision(filename: str) -> bool:
    """True if any revisable exchange in *filename* has no sidecar verdict yet.

    Also true for a chat the background per-chat pass reflected (``chat_reflected``) but
    that is not yet fully frozen (``reflected_at``): all its exchanges carry verdicts, so
    the verdict-presence rule alone would wrongly exclude it, yet it still needs a normal
    run to finish it (run the clean-base phase over its persisted jobs, stamp
    ``reflected_at``). See the two-stage freeze in ``core.background_reflection``."""
    from core.chat_sidecar import ChatSidecar
    sidecar = ChatSidecar(_CHATS_DIR)
    if sidecar.is_chat_reflected(filename) and not sidecar.is_reflected(filename):
        return True
    path = _CHATS_DIR / filename
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for i, ex in enumerate(data.get("exchanges", [])):
        if not (ex.get("user_prompt") or "").strip():
            continue
        if not (ex.get("assistant_response") or "").strip():
            continue
        rec = sidecar.get_exchange(filename, i)
        if rec is None or not (rec.get("verdict") or "").strip():
            return True
    return False


def _consume_pending_clean_base_cli(filename: str) -> dict:
    """Load (consume-once) a chat_reflected chat's persisted clean-base job payloads so
    this run's clean-base phase finishes it. See the two-stage freeze in
    ``core.background_reflection``."""
    from core.reflection_staging import (
        load_pending_clean_base, delete_pending_clean_base)
    data = load_pending_clean_base(_DATA_DIR, filename)
    delete_pending_clean_base(_DATA_DIR, filename)
    return data


def _find_pending_sessions() -> list[str]:
    """All session filenames in hot/chats/ that have at least one unrevised exchange."""
    from core.chat_sidecar import iter_chat_json_files
    result = [
        path.name
        for path in iter_chat_json_files(_CHATS_DIR)
        if _session_has_pending_revision(path.name)
    ]
    return sorted(result)


def _find_latest_sessions(n: int) -> list[str]:
    """The N most-recently modified session filenames."""
    from core.chat_sidecar import iter_chat_json_files
    paths = sorted(
        iter_chat_json_files(_CHATS_DIR),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sorted([p.name for p in paths[:n]])


_REVISIT_LOG_PATH = _DATA_DIR / "hot" / "revisit" / "revisited.jsonl"


def _last_revisited() -> dict[str, datetime]:
    """Fold the revisit log into {session_filename: latest revisit datetime}."""
    out: dict[str, datetime] = {}
    if not _REVISIT_LOG_PATH.exists():
        return out
    try:
        for line in _REVISIT_LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sess = (rec.get("session") or "").strip()
            ts = (rec.get("ts") or "").strip()
            if not sess or not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if sess not in out or dt > out[sess]:
                out[sess] = dt
    except Exception:
        pass
    return out


def _record_revisited(session: str) -> None:
    """Append a revisit record so this chat is not re-picked until the resynth window."""
    try:
        _REVISIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REVISIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"session": session, "ts": datetime.now().isoformat()},
                ensure_ascii=False) + "\n")
    except Exception:
        pass


def _find_revisit_session(min_age_days: float,
                          min_revisit_days: float = 0.0) -> str | None:
    """A random chat at least *min_age_days* old to revisit (or None).

    Ages from the ``YYYYMMDD_HHMMSS`` filename stem, skipping unanswered Ava outreach and
    empty transcripts — the same selection the server-side revisit button uses. When
    *min_revisit_days* > 0, the anti-fixation gate also drops any chat re-derived within
    that window (folded from the shared revisit log) so the random pick rotates."""
    from core.chat_sidecar import iter_chat_json_files
    now = datetime.now()
    cutoff = float(min_age_days) * 86400.0
    revisit_cutoff = float(min_revisit_days) * 86400.0
    last_revisit = _last_revisited() if revisit_cutoff > 0 else {}
    candidates: list[str] = []
    for path in iter_chat_json_files(_CHATS_DIR):
        try:
            age = (now - datetime.strptime(path.stem, "%Y%m%d_%H%M%S")).total_seconds()
        except Exception:
            continue
        if age < cutoff:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exchanges = data.get("exchanges") or []
        if not exchanges:
            continue
        if (data.get("initiated_by") or "").strip() == "ava" and len(exchanges) <= 1:
            continue
        # Anti-fixation: skip a chat re-derived within the resynth window.
        prev = last_revisit.get(path.name)
        if prev is not None and (now - prev).total_seconds() < revisit_cutoff:
            continue
        candidates.append(path.name)
    return random.choice(candidates) if candidates else None


# ── generate function factory ─────────────────────────────────────────────── #

def _make_generate_fn(backend, model, tokenizer, rag, context_length: int, model_id: str):
    """Build the synchronous reflection generate callable used by ReflectionRunner."""
    import time
    from dataclasses import dataclass
    from core.llm_shared import build_inference_prompt
    from core.reflection_config import (
        REFLECT_REPETITION_PENALTY, REFLECT_NO_REPEAT_NGRAM,
    )

    reasoning_effort = "high" if "gpt-oss" in model_id.lower() else None
    enable_thinking  = "gemma-4" in model_id.lower()

    @dataclass(frozen=True)
    class PreparedPrompt:
        prompt: str
        input_tokens: int
        rag_tokens: int
        rag_context: str = ""

    def _clip_rag(text: str, max_tokens) -> str:
        if not text or max_tokens is None:
            return text
        if max_tokens <= 0:
            return ""
        if backend.count_tokens(tokenizer, text) <= max_tokens:
            return text
        kept = []
        for line in text.splitlines():
            candidate = "\n".join(kept + [line])
            if backend.count_tokens(tokenizer, candidate) > max_tokens:
                break
            kept.append(line)
        return "\n".join(kept).rstrip()

    def prepare_prompt(
        content: str,
        system_prompt: str,
        *,
        before_session: str = "",
        disable_rag: bool = False,
        disable_thinking: bool = False,
        rag_query=None,
        rag_include_chat: bool = True,
        rag_include_recollections: bool = True,
        max_rag_tokens=None,
        rag_context_override=None,
        messages_override=None,
    ) -> PreparedPrompt:
        rag_context = rag_context_override or ""
        if messages_override is not None:
            if not disable_rag or rag_context:
                raise ValueError("messages_override requires disable_rag=True and no RAG override")
            conversation = [dict(message) for message in messages_override]
        elif not disable_rag and rag_context_override is None:
            query_text = rag_query if rag_query else content
            rag_context = rag.query(query_text, before_session=before_session,
                                    include_chat=rag_include_chat,
                                    include_recollections=rag_include_recollections)
            rag_context = _clip_rag(rag_context, max_rag_tokens)
            system_content = system_prompt + ("\n\n" + rag_context if rag_context else "")
            conversation = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content},
            ]
        else:
            system_content = system_prompt + ("\n\n" + rag_context if rag_context else "")
            conversation = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content},
            ]
        prompt = build_inference_prompt(
            tokenizer, conversation,
            reasoning_effort=reasoning_effort,
            enable_thinking=(False if disable_thinking else enable_thinking),
        )
        return PreparedPrompt(
            prompt=prompt,
            input_tokens=backend.count_tokens(tokenizer, prompt),
            rag_tokens=backend.count_tokens(tokenizer, rag_context) if rag_context else 0,
            rag_context=rag_context,
        )

    def generate_fn(
        content: str,
        system_prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens_setting: str,
        before_session: str = "",
        disable_rag: bool = False,
        disable_thinking: bool = False,
        rag_query=None,
        rag_include_chat: bool = True,
        rag_include_recollections: bool = True,
        max_rag_tokens=None,
        rag_context_override=None,
        messages_override=None,
        input_token_limit=None,
        prepared_prompt=None,
        on_chunk=None,
    ) -> str:
        prepared = prepared_prompt or prepare_prompt(
            content, system_prompt,
            before_session=before_session,
            disable_rag=disable_rag,
            disable_thinking=disable_thinking,
            rag_query=rag_query,
            rag_include_chat=rag_include_chat,
            rag_include_recollections=rag_include_recollections,
            max_rag_tokens=max_rag_tokens,
            rag_context_override=rag_context_override,
            messages_override=messages_override,
        )
        prompt = prepared.prompt
        input_length = prepared.input_tokens
        limit = int(input_token_limit) if input_token_limit is not None else context_length - 1
        if input_length > limit:
            error = RuntimeError(
                f"Prompt ({input_length} tokens) exceeds reflection input budget "
                f"({limit}; context {context_length}) — rechunk required."
            )
            error.input_limit = limit
            raise error
        available = max(1, context_length - input_length)
        max_new_tokens = _resolve_max_new_tokens(max_new_tokens_setting, available)

        parts: list[str] = []
        gen = backend.stream_generate(
            model, tokenizer, prompt, max_new_tokens, context_length,
            temperature, top_p, capture_tension=False,
            repetition_penalty=REFLECT_REPETITION_PENALTY,
            no_repeat_ngram_size=REFLECT_NO_REPEAT_NGRAM,
            stop_on_repeat=True,
        )

        pending: list[str] = []
        last_flush = time.monotonic()

        def _flush() -> None:
            nonlocal last_flush
            if on_chunk is None or not pending:
                return
            delta = "".join(pending)
            pending.clear()
            last_flush = time.monotonic()
            try:
                on_chunk(delta)
            except Exception:
                pass

        try:
            for chunk in gen:
                parts.append(chunk)
                if on_chunk is not None:
                    pending.append(chunk)
                    pending_len = sum(len(p) for p in pending)
                    if pending_len >= 80 or (time.monotonic() - last_flush) >= 0.8:
                        _flush()
        finally:
            _flush()
            gen.close()
            backend.trim_memory()

        # Surface whether this pass hit the token cap (vs. ended on EOS) so the
        # reflection runner can flag/retry a likely-truncated revision generation.
        generate_fn.last_truncated = getattr(backend, "last_generation_truncated", None)
        generate_fn.last_loop = getattr(backend, "last_generation_stopped_on_loop", None)
        raw = "".join(parts)
        return (_clean_ideal_response(raw, model_id) if messages_override is not None
                else _clean_reflect_response(raw))

    generate_fn.prepare_prompt = prepare_prompt
    return generate_fn


# ── branch-replay function factory ────────────────────────────────────────── #

def _load_chat_prompt() -> str:
    """Default system prompt used as a branch-replay fallback for pre-capture logs.

    Only consulted when an exchange has neither a logged ``system_content`` nor a
    session-level ``system_prompt`` — normal logs carry one of those.
    """
    try:
        return (_INFERENCE_DIR / "prompts" / "chat_prompt.txt").read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _make_branch_fns(backend, model, tokenizer, rag, context_length: int, model_id: str):
    """Build the branch generate + chooser callables the runner needs.

    Wires the shared ``core.branch_replay`` primitives to this process's loaded
    model/backend/tokenizer and the RAG embedder — the headless-CLI counterpart of
    server.py's ``_run_branch_exchange_sync`` / ``_sync_branch_chooser_content``.
    Branch replay reads the original transcripts (with their tension blocks) from the
    live chats dir, not the staging workspace.
    """
    from core import branch_replay

    default_system_prompt = _load_chat_prompt()
    embedder = rag._get_embedder()

    def branch_generate_fn(filename: str, exchange_index: int,
                           temperature: float, top_p: float,
                           on_candidate=None) -> dict:
        return branch_replay.run_branch_exchange(
            filename, exchange_index, temperature, top_p,
            backend=backend, model=model, tokenizer=tokenizer,
            chats_dir=_CHATS_DIR, context_length=context_length,
            model_id=model_id, default_system_prompt=default_system_prompt,
            clean_response_fn=_clean_reflect_response, embedder=embedder,
            on_candidate=on_candidate,
        )

    def branch_chooser_content_fn(payload: dict, system_prompt: str) -> str:
        return branch_replay.build_budgeted_branch_select_content(
            backend, tokenizer, system_prompt, payload, context_length, model_id,
        )

    return branch_generate_fn, branch_chooser_content_fn


def _format_branch_probe(probe: dict) -> list[str]:
    """Render the branch VRAM/token probe as a list of human-readable lines."""
    if not probe:
        return []
    lines = []

    def _gb(v):
        return f"{v:.2f} GB" if isinstance(v, (int, float)) else "n/a"

    gen = probe.get("gen_peak_reserved_gb")
    chooser = probe.get("chooser_peak_reserved_gb")
    if gen is not None or chooser is not None:
        lines.append(
            f"peak VRAM (reserved) — generation {_gb(gen)}, chooser {_gb(chooser)}"
        )
        gen_a = probe.get("gen_peak_alloc_gb")
        ch_a = probe.get("chooser_peak_alloc_gb")
        if gen_a is not None or ch_a is not None:
            lines.append(
                f"peak VRAM (allocated) — generation {_gb(gen_a)}, chooser {_gb(ch_a)}"
            )

    total = probe.get("chooser_prompt_tokens")
    if total:
        opt = probe.get("chooser_option_tokens", 0)
        ctx = probe.get("chooser_context_tokens", 0)
        frac = probe.get("chooser_option_frac")
        pct = f" ({frac:.0%})" if isinstance(frac, (int, float)) else ""
        n_opts = probe.get("n_options")
        n_tag = f" across {n_opts} options" if n_opts else ""
        lines.append(
            f"chooser prompt {total} tok = {opt} option text{pct}{n_tag} "
            f"+ {ctx} shared context"
        )
    return lines


# Events noisy enough to suppress unless --verbose is passed.
_QUIET_EVENTS = frozenset({"phase_started"})


def _make_send_event_fn(verbose: bool):
    stream_open = [False]
    pass_streamed = [False]

    def send_event_fn(event: dict) -> None:
        etype = event.get("event", "")
        if not verbose and etype in _QUIET_EVENTS:
            return

        if etype == "phase_progress":
            text = event.get("text") or ""
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                stream_open[0] = True
                pass_streamed[0] = True
            return

        if stream_open[0]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            stream_open[0] = False

        if etype in ("phase_started", "session_started"):
            pass_streamed[0] = False

        msg  = event.get("message", "")
        ts   = event.get("ts", "")[:19]  # trim subsecond / tz
        sess = event.get("session", "")
        extra = f" [{sess}]" if sess else ""
        print(f"[{ts}]{extra} {etype}: {msg}", flush=True)

        if etype == "phase_done":
            if not pass_streamed[0]:
                text = (event.get("text") or "").strip()
                if text:
                    for ln in text.splitlines():
                        print(f"    {ln}", flush=True)
                    print("", flush=True)
            pass_streamed[0] = False

        elif etype == "branch_done":
            options = event.get("options") or []
            chooser_cot = (event.get("chooser_cot") or "").strip()
            why = (event.get("why") or "").strip()

            if options:
                print("    options (blind order):", flush=True)
                for opt in options:
                    letter = opt.get("letter", "?")
                    kind = opt.get("kind", "")
                    mark = "  ← chosen" if opt.get("chosen") else ""
                    text = (opt.get("text") or "").strip()
                    print(f"      {letter}) [{kind}]{mark}", flush=True)
                    for ln in text.splitlines():
                        print(f"        {ln}", flush=True)
            if chooser_cot:
                print("    reasoning:", flush=True)
                for ln in chooser_cot.splitlines():
                    print(f"      {ln}", flush=True)
            if why:
                print(f"    why: {why}", flush=True)

            probe = event.get("probe") or {}
            probe_lines = _format_branch_probe(probe)
            if probe_lines:
                print("    probe:", flush=True)
                for ln in probe_lines:
                    print(f"      {ln}", flush=True)
                print("", flush=True)

    return send_event_fn


# ── entry point ───────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reflection (consolidation + revision) headlessly on the server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Run from")[0].strip(),
    )

    # Session selection — mutually exclusive group, but we enforce at least one ourselves.
    sel = parser.add_argument_group("session selection")
    sel.add_argument(
        "--session", metavar="FILENAME", action="append", dest="sessions",
        help="Reflect on a specific session file (may repeat).",
    )
    sel.add_argument(
        "--all-pending", action="store_true",
        help="Automatically select all sessions with at least one unrevised exchange.",
    )
    sel.add_argument(
        "--latest", type=int, metavar="N",
        help="Reflect on the N most-recently modified sessions.",
    )
    sel.add_argument(
        "--revisit", action="store_true",
        help="Revisit one random chat >= --revisit-min-age-days old: re-reflect it under "
             "the current persona to re-derive its target, suppressing persona formation "
             "(no persona writes, no digest). Overrides --session/--latest selection.",
    )
    sel.add_argument(
        "--revisit-min-age-days", type=float, default=7.0, metavar="DAYS",
        help="Minimum chat age (days) eligible for --revisit (default: 7).",
    )
    sel.add_argument(
        "--revisit-min-revisit-days", type=float, default=7.0, metavar="DAYS",
        help="Anti-fixation window (days): skip a chat re-derived within this many days "
             "so random --revisit picks rotate (default: 7; 0 disables).",
    )

    ctl = parser.add_argument_group("run control")
    ctl.add_argument(
        "--stage", choices=["reflection", "diff", "merge-rag", "commit-training", "apply", "discard", "all"],
        default="all",
        help="The stage of the Sleep/Reflection pipeline to execute.",
    )
    ctl.add_argument(
        "--overrides", metavar="FILE",
        help=(
            "JSON file containing reflection overrides "
            "(temperature, prompts, sampling — same schema as the "
            "start_reflection_run WebSocket message 'overrides' field)."
        ),
    )
    ctl.add_argument(
        "--verbose", action="store_true",
        help="Print every runner event, including per-pass phase_started messages.",
    )
    ctl.add_argument(
        "--continue-staging", action="store_true",
        help="Do not clear the staging directory before starting reflection, allowing you to accumulate reflections from multiple sessions.",
    )
    ctl.add_argument(
        "--reflect-context", type=int, default=None, metavar="TOKENS",
        help=(
            "Reflection window (max_seq_length the run packs to). Defaults to the "
            "config's reflect_context_length (falling back to context_length). The "
            "model is loaded at max(context_length, this); a value below "
            "context_length is floored to it."
        ),
    )
    args = parser.parse_args()

    # Determine execution stage
    stage = args.stage

    # -- Fast exit stages that do not require GPU/LLM loading
    from core.reflection_staging import read_staging_owners

    if stage == "diff":
        run_stage_diff()
        sys.exit(0)
    elif stage == "merge-rag":
        run_stage_merge_rag()
        sys.exit(0)
    elif stage == "commit-training":
        run_stage_commit_training()
        sys.exit(0)
    elif stage == "apply":
        # apply deletes the workspace at the end — capture owners before it runs.
        owners = read_staging_owners(_STAGING_DIR)
        # Snapshot artifacts + run log into reflections/<run_id>/ before apply
        # discards the staging workspace.
        from core.reflection_archive import archive_reflection, archive_adapter
        for rid in owners:
            archive_reflection(run_id=rid, runs_dir=_RUNS_DIR, staging_dir=_STAGING_DIR,
                               persona_dir=_DATA_DIR / "hot" / "persona",
                               users_dir=_DATA_DIR / "hot" / "users",
                               source="cli")
        counts = run_stage_apply()
        # apply may promote a staged candidate to a persistent adapter — archive it
        # into each contributing run's subdir for review/revert (the real "a train
        # produced an adapter" signal).
        promoted = counts.get("promoted_adapter")
        if promoted:
            for rid in owners:
                archive_adapter(run_id=rid, adapter_dir=Path(promoted))
        sys.exit(0)
    elif stage == "discard":
        run_stage_discard()
        sys.exit(0)

    # ── validate session selection (only for reflection/all stages) ─────── #
    n_selectors = sum([
        bool(args.sessions),
        args.all_pending,
        args.latest is not None,
    ])
    if n_selectors == 0:
        parser.error("At least one session selector is required for reflection/all stages: --session, --all-pending, or --latest.")
    if n_selectors > 1:
        parser.error("--session, --all-pending, and --latest are mutually exclusive.")

    # ── load overrides ──────────────────────────────────────────────────── #
    raw_overrides: dict = {}
    if args.overrides:
        try:
            raw_overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading overrides file {args.overrides!r}: {e}", file=sys.stderr)
            sys.exit(1)

    from core.reflection_config import validate_overrides
    try:
        overrides = validate_overrides(raw_overrides)
    except ValueError as e:
        print(f"Invalid overrides: {e}", file=sys.stderr)
        sys.exit(1)

    # ── resolve sessions ────────────────────────────────────────────────── #
    if args.revisit:
        chosen = _find_revisit_session(args.revisit_min_age_days,
                                       args.revisit_min_revisit_days)
        if not chosen:
            print(f"No chat at least {args.revisit_min_age_days:g} days old (and not "
                  f"revisited in the last {args.revisit_min_revisit_days:g} days) to "
                  "revisit. Nothing to do.", flush=True)
            sys.exit(0)
        sessions = [chosen]
        # Anti-fixation: record the pick now (regardless of run outcome) so the random
        # revisit rotation advances even if the run later fails — mirrors synthesis.
        _record_revisited(chosen)
        print(f"Revisiting old chat: {chosen}", flush=True)
        # Frame revision as remembering a past exchange under the evolved persona.
        if overrides.revision_prompt is None:
            try:
                overrides.revision_prompt = (
                    _INFERENCE_DIR / "prompts" / "revisit_prompt.txt"
                ).read_text(encoding="utf-8").strip()
            except Exception:
                pass
    elif args.all_pending:
        sessions = _find_pending_sessions()
        if not sessions:
            print("No sessions with pending revisions found. Nothing to do.", flush=True)
            sys.exit(0)
        print(f"Found {len(sessions)} pending session(s).", flush=True)
    elif args.latest is not None:
        if args.latest <= 0:
            parser.error("--latest N must be a positive integer.")
        sessions = _find_latest_sessions(args.latest)
        if not sessions:
            print("No sessions found.", flush=True)
            sys.exit(0)
        print(f"Using {len(sessions)} most-recent session(s).", flush=True)
    else:
        sessions = sorted(args.sessions)
        missing = [s for s in sessions if not (_CHATS_DIR / s).exists()]
        if missing:
            print(f"Session file(s) not found: {missing}", file=sys.stderr)
            sys.exit(1)

    for name in sessions:
        print(f"  session: {name}", flush=True)

    # ── load model ──────────────────────────────────────────────────────── #
    config = _load_server_config()
    model_id = config.get("model_id", "")
    adapter_id = config.get("adapter_id", None)
    chat_context_length = int(config.get("context_length", 32768))
    # Reflection window: CLI flag > config reflect_context_length > chat context.
    # The run packs to this; the model is loaded at max(chat, reflect).
    _reflect_cfg = args.reflect_context if args.reflect_context is not None \
        else config.get("reflect_context_length", chat_context_length)
    context_length = max(chat_context_length, int(_reflect_cfg))
    load_context_length = max(chat_context_length, context_length)

    if not model_id:
        print("Error: model_id not set in server_config.json.", file=sys.stderr)
        sys.exit(1)

    _ctx_note = (f"context_length={load_context_length}"
                 if context_length == chat_context_length
                 else f"context_length={chat_context_length}/reflect={context_length}")
    if adapter_id:
        print(f"\nLoading model {model_id} with adapter {adapter_id} ({_ctx_note})...", flush=True)
    else:
        print(f"\nLoading model {model_id} ({_ctx_note})...", flush=True)

    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template

    # The env vars above are a request, not a fact — probe that the fragmentation
    # guard actually engaged and force it at runtime if not, BEFORE the model load
    # (the force applies only to segments created afterwards). Same check the
    # WebSocket server and the train cycle run; see core/alloc_guard.py.
    ensure_expandable_segments()

    backend = UnslothBackend()
    try:
        model, tokenizer = backend.load(model_id, load_context_length, adapter_id)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)

    print(f"Model loaded. {backend.memory_status()}\n", flush=True)

    # ── initialize components ───────────────────────────────────────────── #
    from core.rag_engine import RagEngine
    from core.reflection_writer import ReflectionWriter
    from core.reflection_config import ReflectionRunConfig, ReflectionRunStore
    from core.reflection_runner import ReflectionRunner

    # Clear and recreate staging directories for a fresh run (unless continuing)
    if not args.continue_staging:
        if _STAGING_DIR.exists():
            shutil.rmtree(_STAGING_DIR)
    _STAGING_DIR.mkdir(parents=True, exist_ok=True)
    _STAGING_CHATS_DIR.mkdir(parents=True, exist_ok=True)
    _STAGING_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _STAGING_CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure runs_dir exists (lives in live data dir so websocket client can query runs)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Fold any background per-chat pass output (+ an interrupted run's checkpoint) into
    # live BEFORE the RAG index is built, so those conclusions are retrievable this run and
    # the chat_reflected chats read as live for the two-stage-freeze finish. Mirrors
    # reflection_service._run's recovery fold.
    try:
        from core.reflection_staging import has_checkpoint, fold_checkpoint_to_live
        if has_checkpoint(_DATA_DIR):
            rec = fold_checkpoint_to_live(_DATA_DIR, _INFERENCE_DIR)
            n = int(rec.get("recovered_sidecars", 0) or 0)
            print(f"Folded {n} background/checkpointed chat(s) into live.", flush=True)
    except Exception as e:
        print(f"[warn] checkpoint fold skipped: {e}", flush=True)

    # Setup RAG and writer pointing to staging folder with fallback to live
    rag = RagEngine(
        _STAGING_CHATS_DIR, _INFERENCE_DIR / "prompts",
        memory_dir=_STAGING_MEMORY_DIR, consolidation_dir=_STAGING_CONSOLIDATION_DIR,
        fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR
    )
    rag.build_index_async()

    writer  = ReflectionWriter(_STAGING_MEMORY_DIR, _STAGING_CONSOLIDATION_DIR,
                               live_memory_dir=_MEMORY_DIR)
    store   = ReflectionRunStore(_RUNS_DIR)
    runner  = ReflectionRunner(
        chats_dir=_STAGING_CHATS_DIR,
        memory_dir=_STAGING_MEMORY_DIR,
        runs_dir=_RUNS_DIR,
        consolidation_dir=_STAGING_CONSOLIDATION_DIR,
        reflection_writer=writer,
        fallback_chats_dir=_CHATS_DIR,
        fallback_memory_dir=_MEMORY_DIR,
    )

    generate_fn = _make_generate_fn(backend, model, tokenizer, rag, context_length, model_id)
    branch_generate_fn, branch_chooser_content_fn = _make_branch_fns(
        backend, model, tokenizer, rag, context_length, model_id
    )

    # CoT-regen (approach #3) similarity gate over this process's RAG embedder.
    from core.branch_replay import embed_similarity as _embed_similarity

    def _similarity_fn(text_a: str, text_b: str):
        try:
            return _embed_similarity(rag._get_embedder(), text_a, text_b)
        except Exception:
            return None

    # Persona-only, temporally-cut RAG block for the clean IDEAL seam — conditions the
    # re-derived CoT on Ava's own prior self-knowledge (replaces the retired build-time
    # persona prepend). Mirrors reflection_service._make_persona_context_fn.
    def _persona_context_fn(user_prompt: str, before_session: str) -> str:
        try:
            return rag.query(
                user_prompt or "",
                include_chat=False, include_wander=False,
                include_facts=False, include_persona=True,
                before_session=before_session or "",
            ) or ""
        except Exception:
            return ""

    # Reaction→key bridge for the counter-evidence producer (persuasion channel):
    # resolve a pushed-against reply to the live [persona] key(s) it expressed. Mirrors
    # reflection_service._make_persona_keys_fn.
    def _persona_keys_fn(text: str, before_session: str):
        try:
            return rag.persona_keys(text or "", before_session=before_session or "")
        except Exception:
            return []

    # ── build run config ────────────────────────────────────────────────── #
    from core.reflection_staging import write_staging_owner

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = ReflectionRunConfig(
        run_id=run_id,
        source="cli",
        selected_sessions=sessions,
        overrides=overrides,
        debug=args.verbose,
        revisit=bool(args.revisit),
    )
    store.create_run(run_config)
    # Tie the staging workspace to this run so a later standalone stage command
    # attributes itself back to it. Append under --continue-staging.
    write_staging_owner(_STAGING_DIR, run_id, append=args.continue_staging)

    print(f"Starting reflection run {run_id} (Staged)", flush=True)
    print(f"Sessions: {len(sessions)}, context_length: {context_length}", flush=True)
    print("─" * 60, flush=True)

    send_event_fn = _make_send_event_fn(args.verbose)

    # ── execute ─────────────────────────────────────────────────────────── #
    try:
        runner.execute_run(
            run_config,
            generate_fn=generate_fn,
            store=store,
            context_length=context_length,
            tokenizer=tokenizer,
            rag_refresh_fn=rag.refresh_reflection_memory,
            send_event_fn=send_event_fn,
            # Branch-and-select runs in CLI mode too: the shared core.branch_replay
            # primitives are wired to this process's loaded model + RAG embedder.
            branch_generate_fn=branch_generate_fn,
            branch_chooser_content_fn=branch_chooser_content_fn,
            # CoT regeneration (approach #3) for a kept corrupt-CoT reply: score the
            # re-answer against the original via this process's RAG embedder.
            similarity_fn=_similarity_fn,
            # Persona-condition the clean IDEAL (retired the build-time persona prepend).
            persona_context_fn=_persona_context_fn,
            # Counter-evidence producer: emit persona pushback on COUNTER: yes next-turns.
            persona_keys_fn=_persona_keys_fn,
            # Two-stage freeze: finish any chat the background pass reflected per-chat by
            # loading (consume-once) its persisted clean-base jobs into the clean-base phase.
            consume_pending_clean_base_fn=_consume_pending_clean_base_cli,
        )
    except KeyboardInterrupt:
        print("\n[interrupted] Requesting stop...", flush=True)
        store.request_stop(run_id)
        print("Partial run metadata saved. Exiting.", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n[error] Runner failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── print summary ───────────────────────────────────────────────────── #
    print("─" * 60, flush=True)
    run = store.get_run(run_id)
    summary = (run or {}).get("summary") or {}
    status = (run or {}).get("status", "unknown")
    con_passes = summary.get("consolidation_passes", 0)
    rev_passes = summary.get("revision_passes", 0)
    skipped    = summary.get("skipped_passes", 0)
    print(f"Run {run_id} {status.upper()}", flush=True)
    print(f"  Consolidation passes : {con_passes}", flush=True)
    print(f"  Revision passes      : {rev_passes}", flush=True)
    if skipped:
        print(f"  Skipped passes       : {skipped}", flush=True)
        
    meta_path = _RUNS_DIR / f"{run_id}.meta.json"
    print(f"  Run metadata         : {meta_path}", flush=True)

    if status == "completed":
        if stage == "all":
            print("\nReflection run completed successfully. Merging staged files to live...")
            owners = read_staging_owners(_STAGING_DIR) or [run_id]
            # Snapshot artifacts + run log into reflections/<run_id>/ before apply
            # discards the staging workspace.
            from core.reflection_archive import archive_reflection, archive_adapter
            for rid in owners:
                archive_reflection(run_id=rid, runs_dir=_RUNS_DIR, staging_dir=_STAGING_DIR,
                                   source="cli")
            counts = run_stage_apply()
            # apply may promote a staged candidate to a persistent adapter — archive it
            # into each contributing run's subdir for review/revert.
            promoted = counts.get("promoted_adapter")
            if promoted:
                for rid in owners:
                    archive_adapter(run_id=rid, adapter_dir=Path(promoted))
        else:
            print(f"\nStaged reflection completed. Staging output is under: {_STAGING_DIR}")
            print("You can run diff and validation stages next, or commit the changes:")
            print("  python reflection_run.py --stage diff")
            print("  python reflection_run.py --stage apply")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
