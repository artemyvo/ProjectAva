"""Reflection-run service — the Sleep tab's server-side entry point + lifecycle.

Extracted from server.py. This is the WebSocket-facing orchestration of a
reflection run (distinct from `core.reflection_runner.ReflectionRunner`, the
synchronous consolidation->revision->branch *engine*, and from the headless
`server/reflection_run.py` CLI):

  * `handle_start_reflection_run` dispatches a run to the GPU executor thread —
    builds the run config/overrides, runs the ingestion phase, drives the
    `ReflectionRunner`, and (when the `train` stage is selected) hands LoRA
    production off to the watchdog via `_trigger_watchdog_train`.
  * `handle_reflection_run_status` / `_events` / `_list` / `_get` / `_stop` let
    the client poll and control an in-flight or past run.
  * `_run_ingestion_phase` runs the world-news + lookup ingestion at the start of
    a cycle, reusing `til_wander.ingest_news` / `ingest_lookup`.
  * `handle_get_reflection_prompts` returns the editable prompt text.

This module owns `_reflection_run_active` (set while a run occupies the executor
thread); server.py's idle-loop/generation guards read
`reflection_service._reflection_run_active`, and it also backs the busy getters
injected into `til_wander` / `encounter_run`.

Like the other extracted subsystems it never imports `server`: the Ava-side
capabilities it needs (WebSocket `send`, GPU `executor`, model `backend`, the
RAG/writer/run-store accessors, the reflect-generate + branch-replay helpers, the
clean-base swap, config loader, activity heartbeat, and on-disk paths) are
injected once at startup via :func:`configure`. State is read from
`core.runtime_state`. The moved code is otherwise verbatim.
"""
from __future__ import annotations

import asyncio
import json
import random
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session
from core.reflection_config import (
    ReflectionRunConfig, validate_overrides,
)
from core.chat_sidecar import is_chat_session_json
from core.reflection_prompts import load_reflection_prompts
from core import activity_log
from core import chat_worklog
from core import til_wander

# Set while a reflection run occupies the single GPU executor thread. Owned here;
# server.py's guards and the til_wander/encounter_run busy getters read it as
# `reflection_service._reflection_run_active`.
_reflection_run_active = False
_persona_preview_active = False

# ── Injected server capabilities (populated by configure()) ──
_send: Callable = None                       # async _send(ws, msg)
_executor = None                             # ThreadPoolExecutor (single GPU worker)
_backend = None                              # UnslothBackend
_get_rag: Callable = None
_get_reflection_writer: Callable = None
_get_run_store: Callable = None              # -> ReflectionRunStore (lazy singleton, owned by server)
_load_server_config: Callable = None
_make_sync_reflect_generate: Callable = None
_run_branch_exchange_sync: Callable = None
_sync_branch_chooser_content: Callable = None
_with_clean_base: Callable = None
_mark_activity: Callable = None
_SERVER_DIR = None
_DATA_DIR = None
_CHATS_DIR = None
_MEMORY_DIR = None
_CONSOLIDATION_DIR = None
_REFLECTION_RUNS_DIR = None
_PROMPTS_DIR = None
_WATCHDOG_MGMT_URL = None


def configure(*, send, executor, backend, get_rag, get_reflection_writer, get_run_store,
              load_server_config, make_sync_reflect_generate, run_branch_exchange_sync,
              sync_branch_chooser_content, with_clean_base, mark_activity,
              server_dir, data_dir, chats_dir, memory_dir, consolidation_dir,
              reflection_runs_dir, prompts_dir, watchdog_mgmt_url) -> None:
    """Wire in the server capabilities the moved reflection-run code depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    """
    global _send, _executor, _backend, _get_rag, _get_reflection_writer, _get_run_store
    global _load_server_config, _make_sync_reflect_generate, _run_branch_exchange_sync
    global _sync_branch_chooser_content, _with_clean_base, _mark_activity
    global _SERVER_DIR, _DATA_DIR, _CHATS_DIR, _MEMORY_DIR, _CONSOLIDATION_DIR
    global _REFLECTION_RUNS_DIR, _PROMPTS_DIR, _WATCHDOG_MGMT_URL
    _send = send
    _executor = executor
    _backend = backend
    _get_rag = get_rag
    _get_reflection_writer = get_reflection_writer
    _get_run_store = get_run_store
    _load_server_config = load_server_config
    _make_sync_reflect_generate = make_sync_reflect_generate
    _run_branch_exchange_sync = run_branch_exchange_sync
    _sync_branch_chooser_content = sync_branch_chooser_content
    _with_clean_base = with_clean_base
    _mark_activity = mark_activity
    _SERVER_DIR = server_dir
    _DATA_DIR = data_dir
    _CHATS_DIR = chats_dir
    _MEMORY_DIR = memory_dir
    _CONSOLIDATION_DIR = consolidation_dir
    _REFLECTION_RUNS_DIR = reflection_runs_dir
    _PROMPTS_DIR = prompts_dir
    _WATCHDOG_MGMT_URL = watchdog_mgmt_url


# ══════════════════════════════════════════════════════════════════════════════
# Moved verbatim from server.py.
# ══════════════════════════════════════════════════════════════════════════════


async def handle_get_reflection_prompts(ws) -> None:
    """Return the canonical reflection prompt texts loaded from the server's prompt files."""
    try:
        prompts = load_reflection_prompts()
        await _send(ws, {
            "type": "reflection_prompts",
            "sleep_prompt": prompts.sleep_prompt,
            "revision_prompt": prompts.revision_prompt,
            "branch_prompt": prompts.branch_prompt,
        })
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"get_reflection_prompts failed: {e}"})


_PERSONA_PREVIEW_USER_CONTENT = (
    "Run the persona preview now. Decide whether your current standing chat prompt "
    "should remain unchanged or be replaced. Remember: this preview writes nothing."
)


def _clip_text(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[clipped]"


def _make_similarity_fn(rag) -> Optional[Callable]:
    """Cosine-similarity callable over *rag*'s embedder for the CoT-regen gate.

    Approach #3 grafts a regenerated ``<think>`` onto a kept corrupt-CoT reply only when
    the re-answer reproduces ~that reply; the runner needs to score full-reply similarity.
    Reuses the RAG embedder (already loaded) via branch_replay.embed_similarity. Returns
    None when no RAG is available, in which case CoT-regen safely falls back to answer-only.
    """
    if rag is None:
        return None
    from core.branch_replay import embed_similarity

    def _sim(text_a: str, text_b: str) -> Optional[float]:
        try:
            return embed_similarity(rag._get_embedder(), text_a, text_b)
        except Exception:
            return None

    return _sim


def _make_embed_fn(rag) -> Optional[Callable]:
    """``fn(texts) -> normalized vectors`` over *rag*'s embedder, for the run's fact-dedup
    subject blocking (``core.fact_dedup``).

    Deliberately the LIVE rag's embedder rather than a staging one: it is the same
    multilingual MiniLM either way, and the live instance already has it resident, so a
    staging instance would only load a second copy into RAM. Being CPU-side it is
    unaffected by the clean-base swap the dedup pass runs inside. None ⇒ dedup falls back
    to lexical blocking, which finds close to nothing on a real corpus.
    """
    if rag is None:
        return None

    def _embed(texts):
        return rag._get_embedder().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True)

    return _embed


def _make_persona_context_fn(rag) -> Optional[Callable]:
    """Persona-only RAG block for the clean IDEAL seam (replaces build-time persona inject).

    Returns fn(user_prompt, before_session) -> str: the exchange's relevant ``[persona]``
    self-statements ONLY (facts / chat / wander channels off), temporally cut to *before this
    chat* so the re-derived CoT is conditioned on self-knowledge Ava already had — preserving
    build_ideal_messages' provenance boundary (no LATER reflection artifacts). Bounded by the
    reflection channel's own top-k, soft-framed by rag_memory_prompt.txt, so it lands as "draw
    on where it fits", not a recitation. None when no RAG is available (IDEALs stay persona-free).
    """
    if rag is None:
        return None

    def _persona_context(user_prompt: str, before_session: str) -> str:
        try:
            return rag.query(
                user_prompt or "",
                include_chat=False, include_wander=False,
                include_facts=False, include_persona=True, include_asks=False,
                before_session=before_session or "",
            ) or ""
        except Exception:
            return ""

    return _persona_context


def _make_persona_keys_fn(rag) -> Optional[Callable]:
    """The reaction→key bridge for the counter-evidence producer (persuasion channel).

    Returns fn(text, before_session) -> [(key, score)]: the live ``[persona]`` keys most
    relevant to *text* (the reply whose stance met pushback), persona-only and cut to before
    this chat, so a COUNTER can attach to the exact self-statement(s) the reply expressed.
    Thin wrapper over ``rag.persona_keys`` (top-2, clearly-relevant floor). None when no RAG
    is available (the producer then emits nothing — the counter channel stays dormant)."""
    if rag is None:
        return None

    def _persona_keys(text: str, before_session: str):
        try:
            return rag.persona_keys(text or "", before_session=before_session or "")
        except Exception:
            return []

    return _persona_keys


def _format_prompt_deltas(limit: int = 12, field_limit: int = 500) -> str:
    """Render logged prompt-mutation proposals as compact evidence for preview."""
    if limit <= 0:
        return "(Prompt deltas omitted to preserve context window.)"
    try:
        from core import prompt_mutation
        from training.reflections_path import prompt_dir
        deltas = prompt_mutation.read_prompt_deltas(prompt_dir(), limit=limit)
    except Exception as e:
        return f"(prompt deltas unavailable: {type(e).__name__}: {e})"
    if not deltas:
        return "(No logged prompt modification proposals yet.)"
    lines: list[str] = []
    for i, d in enumerate(deltas, start=1):
        scope = (d.get("scope") or "?").strip() or "?"
        ts = (d.get("ts") or "").strip()
        sess = (d.get("source_session") or "").strip()
        ex = d.get("exchange_index")
        meta = ", ".join(p for p in (
            f"scope={scope}",
            f"session={sess}" if sess else "",
            f"exchange={ex}" if ex is not None else "",
            ts,
        ) if p)
        lines.append(f"{i}. {meta}")
        drift = _clip_text(d.get("drift", ""), field_limit)
        missing = _clip_text(d.get("missing", ""), field_limit)
        delta = _clip_text(d.get("delta", ""), field_limit)
        if drift:
            lines.append(f"   DRIFT: {drift}")
        if missing:
            lines.append(f"   MISSING: {missing}")
        if delta:
            lines.append(f"   DELTA: {delta}")
    return "\n".join(lines)


def _persona_self_presentation(*, portrait_limit: int = 3000,
                               digest_limit: int = 3000) -> str:
    """Load the latest persona digest snapshot in a form Ava can reason over."""
    try:
        from core import reflection_digest
        from training.reflections_path import persona_dir
        digest = reflection_digest.latest_digest(persona_dir())
    except Exception as e:
        return f"(persona digest unavailable: {type(e).__name__}: {e})"
    if not digest:
        return "(No persona digest snapshot has been generated yet.)"

    parts: list[str] = []
    portrait = digest.get("self_portrait") or {}
    portrait_text = (portrait.get("text") or "").strip()
    if portrait_text and portrait_limit > 0:
        parts.append("SELF-PORTRAIT:\n" + _clip_text(portrait_text, portrait_limit))
    try:
        rendered = reflection_digest.render_digest_for_judge(digest)
    except Exception:
        rendered = ""
    if rendered and digest_limit > 0:
        parts.append("STRUCTURED DIGEST:\n" + _clip_text(rendered, digest_limit))
    meta = {
        "version": digest.get("version"),
        "created": digest.get("created"),
        "run_id": digest.get("run_id"),
        "counts": {
            "stances": len(digest.get("stances") or []),
            "dispositions": len(digest.get("dispositions") or []),
            "lines": len(digest.get("lines") or []),
            "anchor_texts": len(digest.get("anchor_texts") or []),
        },
    }
    parts.append("SNAPSHOT META:\n" + json.dumps(meta, ensure_ascii=False, indent=2))
    return "\n\n".join(parts)


def _build_persona_preview_prompt(*, delta_limit: int = 12,
                                  delta_field_limit: int = 500,
                                  portrait_limit: int = 3000,
                                  digest_limit: int = 3000) -> tuple[str, dict]:
    """Assemble the non-mutating prompt self-review context."""
    from core import prompt_mutation

    prompt_path = Path(_PROMPTS_DIR) / "persona_preview_prompt.txt"
    template = prompt_path.read_text(encoding="utf-8").strip()
    current_prompt = prompt_mutation.load_current_chat_prompt(Path(_PROMPTS_DIR))
    persona = _persona_self_presentation(
        portrait_limit=portrait_limit, digest_limit=digest_limit)
    deltas = _format_prompt_deltas(limit=delta_limit, field_limit=delta_field_limit)
    system_prompt = (template
                     .replace("{current_prompt}", current_prompt or "(unavailable)")
                     .replace("{persona_self_presentation}", persona)
                     .replace("{prompt_deltas}", deltas))
    context = {
        "current_prompt_chars": len(current_prompt or ""),
        "persona_chars": len(persona or ""),
        "prompt_delta_chars": len(deltas or ""),
        "prompt_delta_count": delta_limit,
    }
    return system_prompt, context


def _persona_preview_output_reserve(context_length: int) -> int:
    """Reserve enough room for the reasoning + final prompt recommendation."""
    if context_length <= 0:
        return 2048
    if context_length < 4096:
        return max(512, context_length // 4)
    return min(8192, max(2048, context_length // 2))


def _count_preview_input_tokens(system_prompt: str) -> Optional[int]:
    """Best-effort input token count for the preview prompt and user nudge."""
    try:
        tokenizer = _runtime.tokenizer
        if tokenizer is None:
            return None
        return int(_backend.count_tokens(
            tokenizer, system_prompt + "\n\n" + _PERSONA_PREVIEW_USER_CONTENT))
    except Exception:
        return None


def _build_persona_preview_prompt_for_window() -> tuple[str, dict, int]:
    """Build the largest preview context that still leaves output room.

    Persona preview needs unusual output headroom: Ava may think for a while and, if
    she proposes a replacement, emit a whole standing prompt. So evidence is clipped
    adaptively instead of letting persona/delta history crowd out generation.
    """
    context_length = int(_runtime.context_length or 8192)
    reserve = _persona_preview_output_reserve(context_length)
    input_budget = max(512, context_length - reserve - 128)
    candidates = [
        {"delta_limit": 20, "delta_field_limit": 650,
         "portrait_limit": 4000, "digest_limit": 4000},
        {"delta_limit": 12, "delta_field_limit": 500,
         "portrait_limit": 3000, "digest_limit": 3000},
        {"delta_limit": 8, "delta_field_limit": 350,
         "portrait_limit": 2200, "digest_limit": 1800},
        {"delta_limit": 5, "delta_field_limit": 250,
         "portrait_limit": 1400, "digest_limit": 1000},
        {"delta_limit": 3, "delta_field_limit": 180, "portrait_limit": 900, "digest_limit": 600},
        {"delta_limit": 0, "delta_field_limit": 0, "portrait_limit": 700, "digest_limit": 0},
    ]
    chosen_prompt = ""
    chosen_context: dict = {}
    chosen_tokens: Optional[int] = None
    for idx, params in enumerate(candidates):
        system_prompt, context = _build_persona_preview_prompt(**params)
        tokens = _count_preview_input_tokens(system_prompt)
        chosen_prompt, chosen_context, chosen_tokens = system_prompt, context, tokens
        if tokens is None or tokens <= input_budget:
            chosen_context["clip_tier"] = idx
            break
    chosen_context.setdefault("clip_tier", len(candidates) - 1)
    if chosen_tokens is not None and chosen_tokens > input_budget:
        reserve = max(512, min(reserve, context_length - chosen_tokens - 128))
    chosen_context["input_tokens"] = chosen_tokens
    chosen_context["input_token_budget"] = max(512, context_length - reserve - 128)
    chosen_context["output_token_reserve"] = reserve
    return chosen_prompt, chosen_context, reserve


async def handle_persona_preview(ws, msg: dict) -> None:
    """Run a non-mutating prompt self-review and stream Ava's reasoning."""
    global _persona_preview_active, _reflection_run_active
    if _persona_preview_active or _reflection_run_active:
        await _send(ws, {"type": "persona_preview_done", "skipped": "busy",
                         "message": "Another Sleep/GPU job is already in progress."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "persona_preview_done", "skipped": "no_model",
                         "message": "No model loaded - load one from the Chat tab first."})
        return

    _persona_preview_active = True
    _reflection_run_active = True
    loop = asyncio.get_running_loop()

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "persona_preview_chunk", "text": delta}), loop)

    def _run() -> dict:
        global _persona_preview_active, _reflection_run_active
        _persona_preview_active = True
        _reflection_run_active = True
        try:
            system_prompt, context, output_reserve = _build_persona_preview_prompt_for_window()
            asyncio.run_coroutine_threadsafe(
                _send(ws, {"type": "persona_preview_context", **context}), loop)
            generate = _make_sync_reflect_generate(_get_rag())
            text = generate(
                _PERSONA_PREVIEW_USER_CONTENT,
                system_prompt,
                temperature=float(msg.get("temperature", 0.7) or 0.7),
                top_p=float(msg.get("top_p", 0.95) or 0.95),
                max_new_tokens_setting=str(
                    msg.get("max_new_tokens_setting") or output_reserve),
                disable_rag=True,
                on_chunk=_on_chunk,
            )
            return {
                "text": text,
                "truncated": bool(getattr(generate, "last_truncated", False)),
            }
        except Exception as e:
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            _persona_preview_active = False
            _reflection_run_active = False

    result = await loop.run_in_executor(_executor, _run)
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "persona_preview_done"}
    payload.update(result)
    await _send(ws, payload)

def _train_params_to_args(params: dict) -> list[str]:
    """Form the train_cycle CLI flags from the train_params dict.

    The watchdog job runner is semantics-free: it appends this list verbatim to the
    ``train`` job's manifest cmd (``-m training.train_cycle``). So the mapping from
    ``{lora_r, epochs, lr, model_id, run_id, skip_validation, include_fresh}`` to CLI
    flags lives HERE, in the inference server (which git-pulls freely), not in the
    watchdog.
    """
    args: list[str] = []
    if params.get("lora_r") is not None:
        args += ["--lora-r", str(int(params["lora_r"]))]
    if params.get("epochs") is not None:
        args += ["--epochs", str(int(params["epochs"]))]
    if params.get("lr") is not None:
        args += ["--lr", str(float(params["lr"]))]
    if params.get("model_id"):
        args += ["--model-id", str(params["model_id"])]
    if params.get("run_id"):
        args += ["--run-id", str(params["run_id"])]
    if params.get("skip_validation"):
        args += ["--skip-validation"]
    if params.get("include_fresh"):
        args += ["--include-fresh"]
    return args


def _trigger_watchdog_train(params: dict) -> dict:
    """Ask the watchdog to run one offline train cycle (LoRA hand-off).

    Posts to the watchdog's generic offline-job runner (``POST /job/train``): the
    watchdog stops this inference server to free the GPU, runs ``train_cycle``
    (which repoints ``server_config.json``'s ``adapter_id`` on a passing probe),
    then relaunches us. Returns the watchdog's JSON ack. Raises if the watchdog is
    unreachable (e.g. the server was started without one).
    """
    url = _WATCHDOG_MGMT_URL.rstrip("/") + "/job/train"
    data = json.dumps({"args": _train_params_to_args(params or {})}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}

def _run_correction_supersede_phase(store, run_id: str,
                                    send_event_fn: Optional[Callable]) -> None:
    """Part B — supersede-at-correction, after this run's facts are committed LIVE.

    A correction usually writes a NEW fact rather than replacing the stale one, leaving
    both live (opposite claims about one subject). This runs the shared contradiction
    resolver (``core.fact_contradict``) over LIVE facts, but SCOPED to the facts this run
    just committed (``new_keys``): only subjects that changed are judged, and only a
    conflict whose *correction is a fact from this run* is acted on — a pre-existing
    old-vs-old conflict is left for the manual "Resolve fact conflicts" pass (part A).

    Runs on the loaded ADAPTER (a per-run clean-base swap would be prohibitive here; the
    detection is a factual contradiction check, not a persona judgment). Auto-applied but
    a *soften* (``supersede`` — reversible, kept as evidence-of-change). Fully best-effort:
    any failure is logged to the run and never affects the reflection outcome. Called only
    after ``commit-training`` promoted this run's facts to live memory."""
    def emit(event_type: str, **kw) -> None:
        ev = store.append_event(run_id, event_type, **kw)
        if ev and send_event_fn:
            try:
                send_event_fn(ev)
            except Exception:
                pass

    from core import fact_contradict
    from core.reflection_memory import ReflectionMemory
    from core.reflection_writer import ReflectionWriter
    from training.ledger import ConsolidationLedger

    live = [r for r in ReflectionMemory(_MEMORY_DIR).live_items()
            if r.get("kind") == "fact" and (r.get("content") or "").strip()]
    # Facts committed by THIS run — their records carry run_id (consolidation, the
    # from_weights mirror, and resolve-and-distill all stamp it).
    new_keys = {(r.get("key") or "").strip() for r in live
                if (r.get("run_id") == run_id) and (r.get("key") or "").strip()}
    if not new_keys or len(live) < 2:
        return

    emit("phase_started", phase="correction-supersede",
         message=f"Checking {len(new_keys)} newly-learned fact(s) for corrections of "
                 f"stale ones…")

    embedder = _get_rag()._get_embedder()
    def _embed(texts):
        return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    generate = _make_sync_reflect_generate(_get_rag())

    plan = fact_contradict.plan_supersessions(
        live, _embed, generate, new_keys=new_keys)
    if not plan:
        emit("phase_done", phase="correction-supersede",
             message="No newly-learned fact contradicted an older one.")
        return

    writer = ReflectionWriter(_MEMORY_DIR, _CONSOLIDATION_DIR)
    writer.write_supersede(plan)
    led = ConsolidationLedger(_CONSOLIDATION_DIR)
    for p in plan:
        led.supersede([p["key"]], reason=p.get("reason", ""))
    try:
        _get_rag().refresh_reflection_memory()
    except Exception:
        pass
    emit("phase_done", phase="correction-supersede",
         message=(f"Superseded {len(plan)} stale fact(s) corrected by this run "
                  "(reversible — kept as evidence-of-change)."))


def stale_reachout_window_hours() -> float:
    """The stale-reach-out window in hours: ``reachout.stale_delete_hours``, else 48 h."""
    hours = float(getattr(chat_worklog, "DEFAULT_STALE_HOURS", 48.0))
    try:
        cfg = (_load_server_config() or {}).get("reachout") or {}
        if cfg.get("stale_delete_hours") is not None:
            hours = float(cfg.get("stale_delete_hours"))
    except Exception:
        pass
    return hours


def run_stale_reachout_sweep() -> list[dict]:
    """Delete the chats Ava opened herself and was never answered in. Returns what went.

    The **policy** half of the sweep — window, the live session's exemption, and dropping
    the removed chats from retrieval — with no run/event coupling, so both callers share
    one definition of what "stale reach-out" means on this box: the reflection run's
    head phase (:func:`_run_stale_reachout_sweep`, which wraps this in a phase event) and
    the background per-chat job, which runs it at the head of every wake so an idle box
    clears them without waiting for an operator Sleep run.

    The deletion itself is ``chat_worklog.delete_stale_reachouts``, which tombstones each
    opener in ``core.reachout_gate`` before unlinking it; see that module on why the
    tombstone (not the transcript) is what the reach-out backoff and check-in's
    standing-opener list actually depend on. Window from ``server_config.json`` →
    ``reachout.stale_delete_hours`` (default 48 h, ``0`` ⇒ off).

    Best-effort: cleanup must never take its caller down with it."""
    hours = stale_reachout_window_hours()

    # The live logger's file is exempt whatever its age: an opener the user adopted in
    # place is a conversation they are in the middle of answering.
    exclude = set()
    try:
        lg = _session.logger
        if lg is not None and lg.current_file is not None:
            exclude.add(lg.current_file.name)
    except Exception:
        pass

    try:
        deleted = chat_worklog.delete_stale_reachouts(_CHATS_DIR, max_age_hours=hours,
                                                      exclude=exclude)
    except Exception:
        traceback.print_exc()
        return []
    if deleted:
        # Drop them from retrieval in the same breath — whatever runs on after this would
        # otherwise still retrieve the chats it just removed.
        try:
            _get_rag().refresh_chat_index()
        except Exception:
            traceback.print_exc()
    return deleted


def _run_stale_reachout_sweep(store, run_id: str, send_event_fn: Optional[Callable]) -> None:
    """Run the stale-reach-out sweep at the head of a reflection run and report it.

    An unanswered opener is the one session shape reflection can never do anything with:
    both paths skip it un-frozen (right — a later reply must still make it reflectable), so
    it stays in the backlog permanently, sits in the chat list as something to open, and is
    embedded into chat RAG where her own unanswered message can come back as if it had been
    part of a dialogue. Once a reply has stopped being realistic it is just residue, and
    reflection — where the rest of the corpus's periodic tidying already happens — is the
    natural place to clear it.

    This is the run-facing wrapper: the sweep itself is :func:`run_stale_reachout_sweep`
    (shared with the background per-chat job), and all this adds is the ``stale-reachout``
    phase event."""
    hours = stale_reachout_window_hours()
    deleted = run_stale_reachout_sweep()
    if not deleted:
        return

    names = ", ".join(d["session"] for d in deleted[:5])
    if len(deleted) > 5:
        names += f", +{len(deleted) - 5} more"
    ev = store.append_event(
        run_id, "phase_done", phase="stale-reachout",
        message=(f"Removed {len(deleted)} chat(s) Ava opened herself that went unanswered "
                 f"for over {hours:.0f}h: {names}."))
    if ev and send_event_fn:
        try:
            send_event_fn(ev)
        except Exception:
            pass


def _run_ingestion_phase(store, run_id: str, send_event_fn: Optional[Callable]) -> None:
    """News → lookup, applied live, before classic reflection.

    Wander is intentionally NOT part of ingestion — it is *idle-triggered*, not
    reflection-fronting: the idle-job scheduler (core.idle_scheduler) runs an autonomous
    wander after an hour of inactivity, rationed by the wander token budget (and the
    manual button stays available for the operator). Only the news digest ("current
    events") and the open-ask lookup ("resolution of asks") feed this reflection-time
    ingestion cycle.

    Emits ``ingestion``-phase events through the run store (so the polling UI sees
    them). Each sub-step is best-effort: a failure is logged and the phase
    continues, so ingestion never blocks the reflection that follows."""
    def emit(event_type: str, **kw) -> None:
        ev = store.append_event(run_id, event_type, **kw)
        if ev and send_event_fn:
            try:
                send_event_fn(ev)
            except Exception:
                pass

    try:
        store.update_status(run_id, phase="ingestion")
    except Exception:
        pass
    emit("phase_started", phase="ingestion",
         message="Ingestion: world news and lookup (applied live)...")
    try:
        til_wander.ingest_news(emit)
        til_wander.ingest_lookup(emit)
    except Exception as e:
        traceback.print_exc()
        emit("pass_error", phase="ingestion", message=f"Ingestion error: {e}")
    # A bounded slice of the TIL fact-protocol backlog — the articles and digests already
    # on disk with no `<stem>.facts.json` (or a superseded one). Both writing paths record
    # the protocol at fetch/apply, so this exists only for material older than the pass and
    # for anything a parser fix superseded; it is here rather than on its own idle job
    # because this phase is already where the run's TIL work happens, and it is capped per
    # run because it competes with the reflection behind it for the same GPU.
    try:
        res = til_wander.drain_facts_backlog(_TIL_FACTS_PER_RUN, run_id=run_id)
        for rec in (res.get("done") or []):
            # "0 fact(s)" and "cut inside its reasoning" are different events: the first
            # says the text established nothing, the second that the pass never reached
            # an answer region — the failure an operator must see named, not inferred.
            cut_note = (" (cut inside its reasoning)" if rec.get("cut") else "")
            emit("pass_progress", phase="ingestion",
                 message=(f"Protocol: {rec['snippet']} ({rec['kind']}) — "
                          f"{rec['facts']} fact(s){cut_note}."))
        if res.get("remaining"):
            emit("pass_progress", phase="ingestion",
                 message=f"Protocol: {res['remaining']} text(s) still unrecorded.")
    except Exception as e:
        traceback.print_exc()
        emit("pass_warning", phase="ingestion", message=f"Protocol backlog error: {e}")
    # And the recap backlog beside it — the texts with no `<stem>.summary.json`. Its own
    # drain rather than a second product of the protocol pass, because the two read the
    # material differently (the protocol in parts, the recap whole) and asking one pass
    # for both gets a recap of the protocol. Capped harder than the protocol's slice: on
    # any existing box this backlog is the WHOLE corpus, since the artifact did not exist
    # until now, so it drains over several runs rather than owning one.
    try:
        res = til_wander.drain_gist_backlog(_TIL_GISTS_PER_RUN, run_id=run_id)
        for rec in (res.get("done") or []):
            emit("pass_progress", phase="ingestion",
                 message=(f"Recap: {rec['snippet']} ({rec['kind']}) — "
                          + (f"{rec['chars']} chars." if rec["chars"]
                             else "unusable, left in the backlog.")))
        if res.get("remaining"):
            emit("pass_progress", phase="ingestion",
                 message=f"Recap: {res['remaining']} text(s) still without one.")
    except Exception as e:
        traceback.print_exc()
        emit("pass_warning", phase="ingestion", message=f"Recap backlog error: {e}")
    emit("phase_done", phase="ingestion", message="Ingestion complete.")


# How many un-recorded TIL texts one run's ingestion phase will write a protocol for.
# Small on purpose: each is a full generation over a whole article, and the reflection it
# fronts is what the run is actually for. The backlog is finite and shrinks every run.
_TIL_FACTS_PER_RUN = 3

# The same, for recaps (`til_gist`). Smaller, and the asymmetry is deliberate: this pass
# is cheaper per text (one generation against one per reading block) but its backlog is
# every text on the box, since the artifact postdates the corpus — so it wants to drain
# steadily over many runs rather than take a large bite out of the run it fronts. Two
# per run clears this box's ~30 snippets in about a fortnight of nightly reflection, and
# the two consumers degrade gracefully to "no recap yet" in the meantime.
_TIL_GISTS_PER_RUN = 2

# Default minimum age (days) for a chat to be eligible for "revisit old chat". A chat
# younger than this was reflected under essentially the current persona, so re-deriving
# its target adds little. Overridable via server_config `revisit.min_age_days`.
_REVISIT_MIN_AGE_DAYS_DEFAULT = 7
# Default anti-fixation window (days): a chat re-derived within this window is skipped so
# random revisit picks rotate instead of fixating on the same transcript week after week
# (mirrors synthesis's min_resynth_days). Overridable via `revisit.min_revisit_days`.
_REVISIT_MIN_REVISIT_DAYS_DEFAULT = 7


def _revisit_min_age_days() -> float:
    try:
        cfg = _load_server_config() or {}
        v = (cfg.get("revisit") or {}).get("min_age_days")
        if v is not None:
            return float(v)
    except Exception:
        pass
    return float(_REVISIT_MIN_AGE_DAYS_DEFAULT)


def _revisit_min_revisit_days() -> float:
    try:
        cfg = _load_server_config() or {}
        v = (cfg.get("revisit") or {}).get("min_revisit_days")
        if v is not None:
            return float(v)
    except Exception:
        pass
    return float(_REVISIT_MIN_REVISIT_DAYS_DEFAULT)


def _revisit_log_path() -> Path:
    return Path(_DATA_DIR) / "hot" / "revisit" / "revisited.jsonl"


def _last_revisited() -> dict[str, datetime]:
    """Fold the revisit log into {session_filename: latest revisit datetime}."""
    out: dict[str, datetime] = {}
    path = _revisit_log_path()
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
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
    path = _revisit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"session": session, "ts": datetime.now().isoformat()},
                ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()


def _pick_random_old_chat(min_age_days: float,
                          min_revisit_days: float = 0.0,
                          exclude: Optional[list] = None) -> Optional[str]:
    """Choose a random chat at least *min_age_days* old to revisit, or None.

    Ages from the ``YYYYMMDD_HHMMSS`` filename stem (falling back to the JSON
    ``timestamp`` field). Skips unanswered Ava outreach and empty transcripts — there is
    nothing to re-reflect there. When *min_revisit_days* > 0, the anti-fixation gate also
    drops any chat re-derived within that window (folded from the revisit log) so the
    random pick rotates. Any filename in *exclude* is skipped (e.g. chats the caller's own
    run already reflects on). Limited to the live hot chats dir (nothing moves chats to
    archive/ under the wall-clock build, so an aged chat lives here)."""
    chats_dir = Path(_CHATS_DIR)
    if not chats_dir.exists():
        return None
    now = datetime.now()
    cutoff_seconds = float(min_age_days) * 86400.0
    revisit_cutoff = float(min_revisit_days) * 86400.0
    last_revisit = _last_revisited() if revisit_cutoff > 0 else {}
    exclude_set = {str(x) for x in (exclude or [])}
    candidates: list[str] = []
    for path in sorted(chats_dir.glob("*.json")):
        if not is_chat_session_json(path):
            continue
        if path.name in exclude_set:
            continue
        age_seconds: Optional[float] = None
        try:
            ts = datetime.strptime(path.stem, "%Y%m%d_%H%M%S")
            age_seconds = (now - ts).total_seconds()
        except Exception:
            age_seconds = None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if age_seconds is None:
            raw_ts = (data.get("timestamp") or "").strip()
            try:
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is not None:
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                age_seconds = (now - ts).total_seconds()
            except Exception:
                continue
        if age_seconds is None or age_seconds < cutoff_seconds:
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
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_corrupt_chat(exclude: Optional[list] = None) -> Optional[str]:
    """Choose a chat with an operator-flagged corrupt exchange not yet repaired, or None.

    A ``corrupt_cot``/``corrupt_response`` flag (set from the Training review tab) whose
    exchange still has no re-generated sidecar target (``target_source`` != ``"revised"``)
    keeps training on corrupt source content — the build drops it, but only a re-reflection
    re-derives a usable IDEAL. This surfaces such chats to the FRONT of the revisit rotation
    so the next Sleep run repairs them ahead of the random aged pick. The age gate is waived
    (repair should be prompt), but the anti-fixation window (``min_revisit_days``) still
    applies so a corrupt exchange the revision keeps CoT-less can't be re-picked every run.
    Chats the caller's own run already reflects on are excluded."""
    chats_dir = Path(_CHATS_DIR)
    if not chats_dir.exists():
        return None
    from core.chat_sidecar import ChatSidecar
    now = datetime.now()
    revisit_cutoff = _revisit_min_revisit_days() * 86400.0
    last_revisit = _last_revisited() if revisit_cutoff > 0 else {}
    exclude_set = {str(x) for x in (exclude or [])}
    sidecar = ChatSidecar(chats_dir)
    candidates: list[str] = []
    for path in sorted(chats_dir.glob("*.json")):
        if not is_chat_session_json(path) or path.name in exclude_set:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exchanges = data.get("exchanges") or []
        if not exchanges:
            continue
        # Anti-fixation: skip a chat re-derived within the resynth window.
        prev = last_revisit.get(path.name)
        if prev is not None and (now - prev).total_seconds() < revisit_cutoff:
            continue
        recs = sidecar.load(path.name).get("exchanges")
        recs = recs if isinstance(recs, dict) else {}
        for i, ex in enumerate(exchanges):
            if not (ex.get("corrupt_cot") or ex.get("corrupt_response")):
                continue
            rec = recs.get(str(i)) or {}
            if (rec.get("target_source") or "").strip() != "revised":
                candidates.append(path.name)
                break
    if not candidates:
        return None
    return random.choice(candidates)


def _finalize_revisit_sidecar(store, run_id: str, sidecar_path, backup_path) -> None:
    """Keep the re-derived sidecar on a completed revisit; restore the backup otherwise.

    Belt-and-suspenders on top of staging (which only overwrites the live sidecar at
    commit-training): if the run finishes ``completed`` the freshly re-derived sidecar
    stands and the backup is dropped; on any other outcome the pre-revisit sidecar is
    restored from the backup (covers a commit-time failure). Best-effort."""
    if backup_path is None:
        return
    try:
        completed = (store.get_run(run_id) or {}).get("status") == "completed"
    except Exception:
        completed = False
    try:
        if not completed and sidecar_path is not None:
            Path(sidecar_path).write_bytes(Path(backup_path).read_bytes())
        Path(backup_path).unlink(missing_ok=True)
    except Exception:
        pass


def _run_revisit_head_phase(store, parent_run_id: str, send_event_fn,
                            context_length, tokenizer, *, source: str,
                            debug: bool, exclude_sessions: list) -> None:
    """Head-of-run revisit: re-derive one aged chat and commit it LIVE before reflection.

    The Revisit button is here folded into the front of a normal Sleep run: before the
    classic consolidation/revision reflects on this run's selected/new chats, first pick
    one *aged* chat, re-reflect it under the current persona (re-deriving its trainable
    target), and MERGE + COMMIT that revisit to live memory — so its re-derived
    understanding is already retrievable when the main pass queries RAG. Same
    "apply live, then reflect" contract as the ingestion pre-phase (:func:`_run_ingestion_phase`).

    The revisit is a self-contained sub-run (its own ``<parent>_rv`` run id, staging,
    and archive) with persona formation / branching / training suppressed (``revisit``
    semantics). It runs *silently* — the client polls the PARENT run's events, so
    streaming the sub-run's would only be filtered out; instead compact ``revisit``-phase
    markers surface on the parent run. Chats this run already reflects on are excluded
    from the pick. Best-effort throughout: any failure is logged, staging is cleaned, the
    picked chat's sidecar is restored, and the main reflection proceeds unaffected."""
    def emit(event_type: str, **kw) -> None:
        ev = store.append_event(parent_run_id, event_type, **kw)
        if ev and send_event_fn:
            try:
                send_event_fn(ev)
            except Exception:
                pass

    try:
        store.update_status(parent_run_id, phase="revisit")
    except Exception:
        pass
    emit("phase_started", phase="revisit",
         message="Revisit: re-reading one aged chat as context for this run...")

    # Prefer a chat with an unrepaired operator-flagged corrupt exchange (Training review),
    # so "flag it corrupt, the next Sleep re-derives it" holds; fall back to the random
    # aged pick when there is nothing corrupt to repair.
    chosen = _pick_corrupt_chat(exclude=exclude_sessions)
    corrupt_pick = chosen is not None
    if not chosen:
        chosen = _pick_random_old_chat(
            _revisit_min_age_days(), _revisit_min_revisit_days(),
            exclude=exclude_sessions)
    if not chosen:
        emit("phase_done", phase="revisit",
             message="Revisit: no eligible aged chat — skipping.")
        return
    if corrupt_pick:
        emit("phase_progress", phase="revisit",
             message=f"Revisit: re-deriving {chosen} — it has an exchange flagged corrupt.")
    _record_revisited(chosen)

    from core.reflection_staging import (
        get_staging_paths, run_stage_merge_rag, run_stage_commit_training,
        run_stage_discard, write_staging_owner,
    )
    import shutil

    sub_run_id = f"{parent_run_id}_rv"
    # Frame the revision pass as remembering a past exchange under the evolved persona
    # (revisit_prompt has no PERSONA field — persona formation stays suppressed).
    rv_overrides = validate_overrides({})
    try:
        rv_overrides.revision_prompt = (
            Path(_PROMPTS_DIR) / "revisit_prompt.txt"
        ).read_text(encoding="utf-8").strip() or None
    except Exception:
        rv_overrides.revision_prompt = None
    rv_config = ReflectionRunConfig(
        run_id=sub_run_id, source=source, selected_sessions=[chosen],
        overrides=rv_overrides, debug=debug, revisit=True,
    )
    store.create_run(rv_config)

    # Back up the picked chat's live sidecar so a failed revisit is recoverable
    # (staging only overwrites it at commit-training; the backup covers a mid-commit fail).
    sidecar_path = None
    backup_path = None
    stem = chosen[:-5] if chosen.endswith(".json") else chosen
    sc = Path(_CHATS_DIR) / f"{stem}.state.json"
    if sc.exists():
        sidecar_path = sc
        bak = sc.parent / (sc.name + ".revisit-bak")
        try:
            bak.write_bytes(sc.read_bytes())
            backup_path = bak
        except Exception:
            backup_path = None

    paths = get_staging_paths(_DATA_DIR)
    try:
        # Fresh staging workspace for the sub-run (fully committed + discarded below,
        # before the main run touches staging).
        if paths["staging_dir"].exists():
            shutil.rmtree(paths["staging_dir"], ignore_errors=True)
        paths["staging_dir"].mkdir(parents=True, exist_ok=True)
        paths["chats_dir"].mkdir(parents=True, exist_ok=True)
        paths["memory_dir"].mkdir(parents=True, exist_ok=True)
        paths["consolidation_dir"].mkdir(parents=True, exist_ok=True)
        write_staging_owner(paths["staging_dir"], sub_run_id)

        from core.rag_engine import RagEngine
        from core.reflection_writer import ReflectionWriter
        from core.reflection_runner import ReflectionRunner

        staging_rag = RagEngine(
            paths["chats_dir"], _PROMPTS_DIR,
            memory_dir=paths["memory_dir"], consolidation_dir=paths["consolidation_dir"],
            fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR,
        )
        staging_rag.build_index_async()
        writer = ReflectionWriter(paths["memory_dir"], paths["consolidation_dir"],
                                  live_memory_dir=_MEMORY_DIR)
        staging_runner = ReflectionRunner(
            chats_dir=paths["chats_dir"], memory_dir=paths["memory_dir"],
            runs_dir=_REFLECTION_RUNS_DIR, consolidation_dir=paths["consolidation_dir"],
            reflection_writer=writer,
            fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR,
        )
        generate_fn = _make_sync_reflect_generate(staging_rag)
        # Silent sub-run (send_event_fn=None — see docstring). Branch/clean-base omitted:
        # a revisit suppresses branching (and thus the branch judge) by design.
        staging_runner.execute_run(
            rv_config, generate_fn=generate_fn, store=store,
            context_length=context_length, tokenizer=tokenizer,
            rag_refresh_fn=staging_rag.refresh_reflection_memory,
            send_event_fn=None, branch_generate_fn=None,
            branch_chooser_content_fn=None,
            similarity_fn=_make_similarity_fn(staging_rag),
            persona_context_fn=_make_persona_context_fn(staging_rag),
            persona_keys_fn=_make_persona_keys_fn(staging_rag),
        )
        sub = store.get_run(sub_run_id)
        if sub and sub.get("status") == "completed":
            counts: dict = {}
            counts.update(run_stage_merge_rag(_DATA_DIR, _PROMPTS_DIR.parent))
            counts.update(run_stage_commit_training(_DATA_DIR))
            if counts.get("sidecars_copied"):
                try:
                    _get_rag().refresh_chat_index()
                except Exception:
                    traceback.print_exc()
            try:
                from core.reflection_archive import archive_reflection
                archive_reflection(
                    run_id=sub_run_id, runs_dir=_REFLECTION_RUNS_DIR,
                    staging_dir=paths["staging_dir"],
                    persona_dir=_DATA_DIR / "hot" / "persona",
                    users_dir=_DATA_DIR / "hot" / "users",
                    counts=counts, source=source,
                )
            except Exception:
                traceback.print_exc()
            run_stage_discard(_DATA_DIR, _SERVER_DIR)
            emit("phase_done", phase="revisit",
                 message=f"Revisit: re-derived {chosen} and merged it into live memory "
                         "as context for this run.")
        else:
            emit("phase_error", phase="revisit",
                 message=f"Revisit of {chosen} did not complete — main reflection proceeds.")
    except Exception as e:
        traceback.print_exc()
        emit("phase_error", phase="revisit",
             message=f"Revisit head-phase failed ({e}) — main reflection proceeds.")
    finally:
        # Keep the re-derived sidecar on a completed sub-run; restore the backup otherwise.
        _finalize_revisit_sidecar(store, sub_run_id, sidecar_path, backup_path)
        # Never leave the revisit's staging behind for the main run to inherit.
        try:
            leftover = get_staging_paths(_DATA_DIR)["staging_dir"]
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)
        except Exception:
            pass


def run_chat_only_reflection(config, store, staging_dir, on_chat_reflected,
                             send_event_fn=None) -> str:
    """Run ONE chat's per-chat reflection (consolidation + revision + branch generation) in
    a caller-supplied throwaway *staging_dir*, skipping the run-level cross-cutting passes.

    This is the engine seam the background per-chat pass (``core.background_reflection``)
    delegates to — it reuses the server's live reflection plumbing (reflect-generate,
    branch replay, similarity/persona seams) so the background pass carries none of it.
    ``config`` must have ``chat_only=True`` and a single ``selected_sessions`` entry. The
    runner freezes the chat ``chat_reflected`` and hands its clean-base job payloads to
    *on_chat_reflected(filename, judge_jobs, fact_candidates)*; ``clean_base_ctx=None`` so
    the judge/fact placement do NOT run here (a later normal run finishes them). Nothing is
    committed to live — the durable output is whatever *on_chat_reflected* persists.

    Runs on the GPU executor thread (blocking). Returns the run's final status string
    (``"completed"`` / ``"stopped"`` / ``"failed"``)."""
    from core.rag_engine import RagEngine
    from core.reflection_writer import ReflectionWriter
    from core.reflection_runner import ReflectionRunner

    staging_dir = Path(staging_dir)
    chats_dir = staging_dir / "chats"
    memory_dir = staging_dir / "memory"
    consolidation_dir = staging_dir / "consolidation"
    for d in (chats_dir, memory_dir, consolidation_dir):
        d.mkdir(parents=True, exist_ok=True)

    context_length = int(_runtime.reflect_context_length
                         or _runtime.context_length or 8192)
    tokenizer = _runtime.tokenizer

    staging_rag = RagEngine(
        chats_dir, _PROMPTS_DIR,
        memory_dir=memory_dir, consolidation_dir=consolidation_dir,
        fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR,
    )
    staging_rag.build_index_async()
    writer = ReflectionWriter(memory_dir, consolidation_dir, live_memory_dir=_MEMORY_DIR)
    staging_runner = ReflectionRunner(
        chats_dir=chats_dir, memory_dir=memory_dir,
        runs_dir=_REFLECTION_RUNS_DIR, consolidation_dir=consolidation_dir,
        reflection_writer=writer,
        fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR,
    )
    generate_fn = _make_sync_reflect_generate(staging_rag)

    def _branch_chooser_content_fn(payload: dict, system_prompt: str) -> str:
        return _sync_branch_chooser_content(payload, system_prompt, context_length)

    staging_runner.execute_run(
        config,
        generate_fn=generate_fn,
        store=store,
        context_length=context_length,
        tokenizer=tokenizer,
        rag_refresh_fn=staging_rag.refresh_reflection_memory,
        send_event_fn=send_event_fn,
        branch_generate_fn=_run_branch_exchange_sync,
        branch_chooser_content_fn=_branch_chooser_content_fn,
        similarity_fn=_make_similarity_fn(staging_rag),
        persona_context_fn=_make_persona_context_fn(staging_rag),
        persona_keys_fn=_make_persona_keys_fn(staging_rag),
        # Per-chat only: no run-level clean-base judge / fact placement / persona digest.
        clean_base_ctx=None,
        on_chat_reflected=on_chat_reflected,
    )
    run = store.get_run(config.run_id) or {}
    return run.get("status") or "failed"


async def handle_start_reflection_run(ws, msg: dict) -> None:
    """Create and execute a server-owned reflection run.

    Validates overrides and sessions, creates a run record, sends
    reflection_run_started immediately, then dispatches the runner to the
    executor thread so the event loop stays free during GPU inference.
    The client polls reflection_run_status for progress.
    """
    global _reflection_run_active

    raw_overrides = msg.get("overrides") or {}
    try:
        overrides = validate_overrides(raw_overrides)
    except ValueError as e:
        await _send(ws, {"type": "error", "message": f"Invalid overrides: {e}"})
        return

    sessions = sorted([str(s) for s in (msg.get("sessions") or [])])

    # "Revisit old chat": with no explicit sessions, the server picks one random chat
    # aged >= revisit.min_age_days and re-reflects it under the current persona.
    revisit = bool(msg.get("revisit"))
    if revisit and not sessions:
        chosen = _pick_random_old_chat(_revisit_min_age_days(),
                                       _revisit_min_revisit_days())
        if not chosen:
            await _send(ws, {
                "type": "error",
                "message": (f"No chat at least {int(_revisit_min_age_days())} days old "
                            "(and not revisited in the last "
                            f"{int(_revisit_min_revisit_days())} days) to revisit."),
            })
            return
        sessions = [chosen]
        # Anti-fixation: record the pick now (regardless of run outcome) so the random
        # revisit rotation advances even if the run later fails — mirrors synthesis.
        _record_revisited(chosen)

    if not sessions:
        await _send(ws, {"type": "error", "message": "sessions must be a non-empty list"})
        return

    if _reflection_run_active:
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is already active — stop it before starting a new one.",
        })
        return

    # A background per-chat pass may hold the executor; an operator Sleep run preempts it
    # (it is preemptible) so the run does not queue behind a whole idle-drain wake.
    try:
        from core import background_reflection
        if background_reflection.is_active():
            background_reflection.request_preempt()
            for _ in range(100):
                if not background_reflection.is_active():
                    break
                await asyncio.sleep(0.1)
    except Exception:
        pass

    source = str(msg.get("source") or "ui")
    stages = msg.get("stages")
    if stages is None:
        # No explicit stage selection — default to a reflection-only staged run.
        stages = ["reflection"]

    # Dry run: preview a reflection pass that writes nothing (no staging, no
    # memory/ledger, no RAG refresh, no downstream stages). Two shapes:
    #   * short summary (default) — consolidation phase only, previews the distilled
    #     fact/ask/resolved artifacts a real run would route into RAG;
    #   * "Dry Sleep" (dry_full) — the full consolidation + revision + branch
    #     experiment (chat summary → IDEAL → counterfactual branches), still writing
    #     nothing, so an operator can see a whole reflection pass before committing.
    dry_run = bool(msg.get("dry_run", False))
    dry_full = bool(msg.get("dry_full", False))

    # Ingestion (news + open-ask lookup) runs before classic reflection by default; a
    # revisit forces it OFF (it shapes memory for fresh reflection, not target re-derivation).
    ingest_enabled = bool(msg.get("ingest", True))
    if revisit:
        # A revisit re-derives one obsolete chat's target only: never hand off to
        # training, never mint a new Ava version, never ingest news/asks, and frame the
        # revision pass as remembering a past exchange under the evolved persona (unless
        # the caller supplied a prompt). `persona` is stripped for the same reason `train`
        # is: both PRODUCE a new version of her, and a revisit is maintenance on one old
        # chat — it also runs as the head-phase of every normal run, so minting there
        # would produce a persona per run regardless of what the operator asked for.
        stages = [s for s in stages if s not in ("train", "persona")]
        ingest_enabled = False
        if overrides.revision_prompt is None:
            try:
                overrides.revision_prompt = (
                    Path(_PROMPTS_DIR) / "revisit_prompt.txt"
                ).read_text(encoding="utf-8").strip()
            except Exception:
                pass

    debug = bool(msg.get("debug", False))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reflection packs to the reflection window (>= the chat context), letting a run
    # reason over more than a chat may grow to. Default = the physical ceiling the
    # model was loaded at; a per-run override (Sleep tab) may dial it DOWN, clamped
    # to [chat context, physical ceiling] — it can never exceed the real
    # max_seq_length, and a request below the chat budget is floored to it.
    _reflect_ceiling = int(_runtime.reflect_context_length or _runtime.context_length)
    _reflect_floor = int(_runtime.context_length)
    _reflect_requested = msg.get("reflect_context_length")
    if _reflect_requested is None:
        context_length = _reflect_ceiling
    else:
        try:
            context_length = max(_reflect_floor, min(int(_reflect_requested), _reflect_ceiling))
        except (TypeError, ValueError):
            context_length = _reflect_ceiling

    config = ReflectionRunConfig(
        run_id=run_id,
        source=source,
        selected_sessions=sessions,
        overrides=overrides,
        debug=debug,
        revisit=revisit,
    )
    store = _get_run_store()
    run = store.create_run(config)
    await _send(ws, {
        "type": "reflection_run_started",
        "run_id": run_id,
        "status": run["status"],
        "revisit": revisit,
        # The chat the server picked to revisit, so the UI can name it.
        "chosen_session": sessions[0] if revisit else None,
        # The reflection window this run packs to (>= chat context, <= physical ceiling).
        "reflect_context_length": context_length,
    })

    # Both, not just the model: every load/unload path sets the pair together, so a
    # half-set runtime means something went wrong upstream — and letting the run start
    # on one costs a full ingestion phase before dying deep inside consolidation with
    # "No tokenizer loaded", which names neither the run nor the cause.
    if _runtime.model is None or _runtime.tokenizer is None:
        detail = ("No model loaded" if _runtime.model is None
                  else "No tokenizer loaded (model is loaded — runtime state is "
                       "inconsistent; check the log for a failed load or clean-base swap)")
        store.append_event(run_id, "run_failed", message=detail)
        store.finalize_run(run_id, "failed", {
            "error": detail, "mutations_applied": False,
        })
        return

    # Revisit: back up the chosen chat's live sidecar so it is recoverable if the run
    # fails (staging only overwrites the live sidecar at commit-training; the backup
    # covers a commit-time failure). Restored/cleaned in _run's finally.
    revisit_sidecar_path = None
    revisit_backup_path = None
    if revisit:
        stem = sessions[0][:-5] if sessions[0].endswith(".json") else sessions[0]
        sc = Path(_CHATS_DIR) / f"{stem}.state.json"
        if sc.exists():
            revisit_sidecar_path = sc
            bak = sc.parent / (sc.name + ".revisit-bak")
            try:
                bak.write_bytes(sc.read_bytes())
                revisit_backup_path = bak
            except Exception:
                revisit_backup_path = None

    loop = asyncio.get_running_loop()
    # context_length (the resolved reflection window) was computed above, before the
    # reflection_run_started event, so it could be reported to the client.
    tokenizer = _runtime.tokenizer

    def _send_event_fn(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(_send(ws, event), loop)

    def _branch_chooser_content_fn(payload: dict, system_prompt: str) -> str:
        return _sync_branch_chooser_content(payload, system_prompt, context_length)

    def _checkpoint_completed_session(filename: str) -> None:
        from core.reflection_staging import checkpoint_completed_session
        checkpoint_completed_session(_DATA_DIR, filename, run_id)

    def _consume_pending_clean_base(filename: str) -> dict:
        # Two-stage freeze — stage two: load (consume-once) the clean-base job payloads
        # the background per-chat pass persisted for a `chat_reflected` chat, so this run's
        # end-of-run clean-base phase finishes it.
        from core.reflection_staging import (
            load_pending_clean_base, delete_pending_clean_base)
        data = load_pending_clean_base(_DATA_DIR, filename)
        delete_pending_clean_base(_DATA_DIR, filename)
        return data

    def _run_dry_summary() -> None:
        """Write-nothing reflection preview: generate + report, persist nothing.

        Reuses the live RAG only for read-only retrieval during generation. The
        runner is told ``dry_run=True`` so it skips every artifact write. No staging
        workspace is created and no downstream stage runs. The runner finalizes the
        run itself, so there is nothing to commit afterward.

        Two shapes, keyed on *dry_full*:
          * short summary (``dry_full`` false) — ``consolidation_only=True``, so only
            the first phase runs (the distilled fact/ask/resolved preview);
          * "Dry Sleep" (``dry_full`` true) — the full consolidation + revision +
            branch experiment. Branch generation is read-only (replays logged token
            ids and blind-chooses; only ``write_revision`` would persist the block,
            which dry_run skips), so it is wired in. The clean-base phase-two judge is
            NOT (``clean_base_ctx=None``): it is logged-only, pays an expensive adapter
            swap, and its criterion flip rewrites the sidecar — none of which a
            non-mutating preview wants."""
        from core.reflection_runner import ReflectionRunner
        runner = ReflectionRunner(
            chats_dir=_CHATS_DIR,
            memory_dir=_MEMORY_DIR,
            runs_dir=_REFLECTION_RUNS_DIR,
            consolidation_dir=_CONSOLIDATION_DIR,
            reflection_writer=_get_reflection_writer(),  # unused in dry run
        )
        generate_fn = _make_sync_reflect_generate(_get_rag())
        runner.execute_run(
            config,
            generate_fn=generate_fn,
            store=store,
            context_length=context_length,
            tokenizer=tokenizer,
            rag_refresh_fn=None,
            send_event_fn=_send_event_fn,
            branch_generate_fn=_run_branch_exchange_sync if dry_full else None,
            branch_chooser_content_fn=_branch_chooser_content_fn if dry_full else None,
            similarity_fn=_make_similarity_fn(_get_rag()),
            persona_context_fn=_make_persona_context_fn(_get_rag()),
            persona_keys_fn=_make_persona_keys_fn(_get_rag()),
            consolidation_only=not dry_full,
            dry_run=True,
        )

    def _run() -> None:
        global _reflection_run_active
        _reflection_run_active = True
        # Open the box-wide "current activity" chip for the whole run (the per-event
        # journal lines come from the store's coarse mirror). Cleared in the finally.
        activity_log.set_current(
            "reflection",
            f"reflection {'preview' if dry_run else 'run'} ({run_id})")
        try:
            if dry_run:
                _run_dry_summary()
                return
            # Recover an interrupted prior run: fold its durable completion checkpoint
            # into live so those fully-reflected chats read as frozen (reflect-once) and
            # are skipped instead of re-reflected. Consume-once (deleted after folding);
            # a run that completed left no checkpoint, so this is a no-op then. Runs
            # before revisit/ingestion so the recovered conclusions are live RAG context.
            try:
                from core.reflection_staging import has_checkpoint, fold_checkpoint_to_live
                if has_checkpoint(_DATA_DIR):
                    rec = fold_checkpoint_to_live(_DATA_DIR, _PROMPTS_DIR.parent)
                    n = int(rec.get("recovered_sidecars", 0) or 0)
                    store.append_event(
                        run_id, "phase_done", phase="recover",
                        message=(f"Recovered {n} completed chat(s) from an interrupted "
                                 "reflection run." if n else
                                 "Interrupted-run checkpoint folded (no new chats to recover)."))
            except Exception:
                traceback.print_exc()
            # Stale reach-out sweep: delete the chats she opened herself that nobody
            # answered inside the window. Ahead of everything that reads the corpus
            # (revisit's random pick, ingestion, the reflection proper) so none of them
            # spends work on a chat this run is about to remove. A revisit run targets one
            # named old chat and does no corpus maintenance, so it is exempt.
            if not revisit:
                try:
                    _run_stale_reachout_sweep(store, run_id, _send_event_fn)
                except Exception:
                    traceback.print_exc()
            # Revisit head-phase: on a normal run, first re-derive ONE aged chat and
            # commit it to live memory so its re-derived understanding is available as RAG
            # context to the main reflection that follows (see _run_revisit_head_phase).
            # Skipped for the Revisit button's own run (revisit=True — avoids recursion),
            # when nothing will reflect ("reflection" not in stages), or via revisit_head=false.
            if (not revisit and "reflection" in stages
                    and bool(msg.get("revisit_head", True))):
                try:
                    _run_revisit_head_phase(
                        store, run_id, _send_event_fn, context_length, tokenizer,
                        source=source, debug=debug, exclude_sessions=sessions,
                    )
                except Exception:
                    traceback.print_exc()
            # Ingestion phase: pull news (if changed) + resolve open
            # [ask:search], and APPLY every conclusion to live memory — BEFORE the
            # classic consolidation/revision/branch reflection, so its conclusions
            # are already live when that reflection retrieves RAG. Opt out with
            # ingest=False (a revisit always does). Best-effort: never blocks reflection.
            if ingest_enabled:
                try:
                    _run_ingestion_phase(store, run_id, _send_event_fn)
                except Exception:
                    traceback.print_exc()
            # Both pre-phases are best-effort — they must not block reflection — but
            # "best-effort" was swallowing the one failure that MUST stop the run: a
            # CleanBaseSession (ingestion's subject-extraction pass) that released the
            # model and then could not reload it leaves the runtime with no model at
            # all, and every later pass dies on "No tokenizer loaded" while the log
            # blames consolidation. Re-check what the entry gate checked; the model is
            # not something a later phase can recover.
            if _runtime.model is None or _runtime.tokenizer is None:
                detail = ("Model lost during the pre-reflection phases (a clean-base "
                          "swap released it and could not reload it — see the log for "
                          "the load error). Restart the server and retry.")
                store.append_event(run_id, "run_failed", message=detail)
                store.finalize_run(run_id, "failed", {
                    "error": detail, "mutations_applied": False,
                })
                return
            from core.reflection_staging import (
                get_staging_paths,
                run_stage_merge_rag,
                run_stage_commit_training,
                run_stage_discard,
                write_staging_owner,
            )
            success = False
            if "reflection" in stages:
                paths = get_staging_paths(_DATA_DIR)
                clear_staging = bool(msg.get("clear_staging", True))
                if clear_staging:
                    # 1. Clear staging directory
                    if paths["staging_dir"].exists():
                        import shutil
                        try:
                            shutil.rmtree(paths["staging_dir"])
                        except Exception:
                            pass
                paths["staging_dir"].mkdir(parents=True, exist_ok=True)
                paths["chats_dir"].mkdir(parents=True, exist_ok=True)
                paths["memory_dir"].mkdir(parents=True, exist_ok=True)
                paths["consolidation_dir"].mkdir(parents=True, exist_ok=True)
                # Tie this staging workspace to the producing run so a later stage
                # command attributes itself back to the right run in the log. Append
                # on continue-staging (multiple runs share one workspace).
                write_staging_owner(paths["staging_dir"], run_id,
                                    append=not clear_staging)

                from core.rag_engine import RagEngine
                from core.reflection_writer import ReflectionWriter
                from core.reflection_runner import ReflectionRunner

                # Setup staged RAG and writer
                staging_rag = RagEngine(
                    paths["chats_dir"], _PROMPTS_DIR,
                    memory_dir=paths["memory_dir"], consolidation_dir=paths["consolidation_dir"],
                    fallback_chats_dir=_CHATS_DIR, fallback_memory_dir=_MEMORY_DIR
                )
                staging_rag.build_index_async()

                # live_memory_dir lets the persona idempotency guard span prior
                # committed runs, not just this staging workspace.
                writer = ReflectionWriter(paths["memory_dir"], paths["consolidation_dir"],
                                          live_memory_dir=_MEMORY_DIR)
                
                # Setup staging runner (status runs are written to _REFLECTION_RUNS_DIR)
                staging_runner = ReflectionRunner(
                    chats_dir=paths["chats_dir"],
                    memory_dir=paths["memory_dir"],
                    runs_dir=_REFLECTION_RUNS_DIR,
                    consolidation_dir=paths["consolidation_dir"],
                    reflection_writer=writer,
                    fallback_chats_dir=_CHATS_DIR,
                    fallback_memory_dir=_MEMORY_DIR,
                )

                # Build generation function pointing to staging RAG
                generate_fn = _make_sync_reflect_generate(staging_rag)

                staging_runner.execute_run(
                    config,
                    generate_fn=generate_fn,
                    store=store,
                    context_length=context_length,
                    tokenizer=tokenizer,
                    rag_refresh_fn=staging_rag.refresh_reflection_memory,
                    send_event_fn=_send_event_fn,
                    branch_generate_fn=_run_branch_exchange_sync,
                    branch_chooser_content_fn=_branch_chooser_content_fn,
                    similarity_fn=_make_similarity_fn(staging_rag),
                    persona_context_fn=_make_persona_context_fn(staging_rag),
                    persona_keys_fn=_make_persona_keys_fn(staging_rag),
                    # Fact dedup blocks the store by subject before grouping it.
                    embed_fn=_make_embed_fn(_get_rag()),
                    # Batched phase-two judge runs on the frozen base (one adapter swap
                    # per run); the runner calls this to enter the clean-base session.
                    clean_base_ctx=_with_clean_base,
                    # Durably checkpoint each completed chat so an interrupted run's
                    # already-reflected sessions survive to be recovered next run.
                    on_session_committed=_checkpoint_completed_session,
                    # Finish any chat the background pass already reflected per-chat:
                    # load its persisted clean-base jobs into this run's clean-base phase.
                    consume_pending_clean_base_fn=_consume_pending_clean_base,
                )
                run_data = store.get_run(run_id)
                success = run_data and run_data.get("status") == "completed"
            else:
                success = True
                store.update_status(run_id, phase="downstream")
                store.append_event(run_id, "phase_started", phase="downstream", message="Skipping reflection stage. Starting downstream stages...")

            if success:
                # The run reached `completed`, so its work is safe in the staging
                # workspace (to be committed/reviewed) — the per-chat recovery
                # checkpoint is only for an interrupted run and is now redundant.
                try:
                    from core.reflection_staging import discard_checkpoint
                    discard_checkpoint(_DATA_DIR)
                except Exception:
                    pass
                commit_counts: dict = {}
                if "merge-rag" in stages:
                    store.append_event(run_id, "phase_started", phase="merge-rag", message="Merging RAG memory...")
                    counts = run_stage_merge_rag(_DATA_DIR, _PROMPTS_DIR.parent)
                    commit_counts.update(counts)
                    store.append_event(run_id, "phase_done", phase="merge-rag", message="RAG memory merged successfully.")
                if "commit-training" in stages:
                    store.append_event(run_id, "phase_started", phase="commit-training", message="Committing training sidecars and ledger deltas...")
                    counts = run_stage_commit_training(_DATA_DIR)
                    commit_counts.update(counts)
                    # The sidecar commit is what makes a newly-generated gist live. Refresh
                    # the serving chat index now; otherwise the summary (and the latest
                    # wall-clock modifiers) would wait for a restart or unrelated rebuild.
                    if counts.get("sidecars_copied"):
                        try:
                            _get_rag().refresh_chat_index()
                        except Exception:
                            traceback.print_exc()
                    store.append_event(run_id, "phase_done", phase="commit-training", message="Training sidecars committed successfully.")

                # Part B — supersede-at-correction: now that this run's facts are LIVE,
                # supersede any older fact a newly-learned one corrects (contradicts).
                # Best-effort; a revisit derives no new facts so it is a natural no-op.
                if "merge-rag" in stages and "commit-training" in stages and not revisit:
                    try:
                        _run_correction_supersede_phase(store, run_id, _send_event_fn)
                    except Exception:
                        traceback.print_exc()

                # mutations_applied means "production (live hot/) was written" —
                # i.e. a downstream stage ran. A reflection-only run writes the
                # staging workspace only, so it does not count.
                committed = ("merge-rag" in stages or "commit-training" in stages)
                train_requested = "train" in stages
                # `persona` is the train-less way to mint a new Ava version. It exists
                # because live chat reads her self-portrait through the ACTIVE-PERSONA
                # pointer (generation._current_chat_portrait → persona_paths
                # .active_persona_dir()/digest.json), not from the live hot/persona/ dir a
                # reflection run writes to — so without a produce+activate step the fresh
                # digest never reaches a turn. Every OTHER product of a run (RAG memory,
                # facts, sidecars, user portraits) is read from its live dir and lands the
                # moment merge-rag/commit-training commits. Only the self-portrait is
                # gated on a version being minted, and until now the only minter was
                # train_cycle's promotion tail.
                persona_requested = ("persona" in stages) and committed and not dry_run

                # Snapshot this reflection's artifacts + run log into
                # server/reflections/<run_id>/ BEFORE discarding staging (the deltas
                # still live in the staging workspace at this point). A train run's
                # adapter is copied later by train_cycle via archive_adapter().
                if committed or train_requested:
                    try:
                        from core.reflection_archive import archive_reflection
                        archive_reflection(
                            run_id=run_id, runs_dir=_REFLECTION_RUNS_DIR,
                            staging_dir=get_staging_paths(_DATA_DIR)["staging_dir"],
                            persona_dir=_DATA_DIR / "hot" / "persona",
                            users_dir=_DATA_DIR / "hot" / "users",
                            counts=commit_counts, source=source,
                        )
                        store.append_event(run_id, "phase_done", phase="archive",
                                           message=f"Archived reflection artifacts to reflections/{run_id}/.")
                    except Exception as e:
                        store.append_event(run_id, "phase_error", phase="archive",
                                           message=f"Reflection archive failed: {e}")

                # If we merged and committed everything, discard staging
                if "merge-rag" in stages and "commit-training" in stages:
                    run_stage_discard(_DATA_DIR, _SERVER_DIR)

                # ── Persona: mint this run's Ava version WITHOUT training ──────
                # The train-less counterpart of train_cycle's promotion tail (which
                # calls the same produce_persona on a passing probe). It freezes the
                # live state — the UNCHANGED active adapter plus this run's fresh
                # digest, prompts, config and data/ — into data/persona/<run_id>/ and
                # flips current.json, which is what actually puts the new self-portrait
                # in front of a live chat turn. GPU-free / filesystem-only, so it is
                # safe here with the model loaded (a copytree of the corpus, not a fit).
                # Best-effort by the same rule the archive above follows: a failed
                # snapshot must not fail a run whose reflection is already committed.
                persona_produced = None
                if persona_requested:
                    store.append_event(
                        run_id, "phase_started", phase="persona",
                        message="Producing persona snapshot (freezing the current "
                                "adapter + this run's digest as the active version)...")
                    try:
                        # snapshot_state lives at the server root, not on the inference
                        # package path — same lazy path-injected import mgmt_http uses.
                        import importlib
                        import sys as _sys
                        _root = str(Path(__file__).resolve().parents[2])
                        if _root not in _sys.path:
                            _sys.path.insert(0, _root)
                        snapshot_state = importlib.import_module("snapshot_state")
                        pdir = snapshot_state.produce_persona(run_id, activate=True)
                        persona_produced = run_id
                        store.append_event(
                            run_id, "phase_done", phase="persona",
                            message=f"Activated persona data/persona/{run_id} — her "
                                    f"refreshed self-portrait is live from the next "
                                    f"turn (adapter unchanged). {pdir}")
                    except Exception as e:
                        store.append_event(
                            run_id, "phase_error", phase="persona",
                            message=f"Persona production failed (reflection is still "
                                    f"committed; the previous persona stays active): {e}")

                # ── Preview render: make this run reviewable BEFORE any training ──
                # On a training-lite run (Train off) nothing writes a build snapshot,
                # so the Training review tab would keep showing the LAST build — the
                # chats this run just reflected (fresh ones included) would be
                # invisible until the real cycle had already trained them. When the
                # operator asked for fresh chats (train_params.include_fresh), write a
                # GPU-free preview snapshot of the NEXT build's corpus instead
                # (training/preview_build.py: outcome "preview", nothing trained), so
                # the review→repair→lock ❄ pass can happen first and the repairs are
                # what the later real cycle trains. A train run needs none of this —
                # train_cycle renders the preview rows into its own snapshot.
                # Best-effort by the same rule as persona/archive above.
                _fresh_preview = bool(
                    (msg.get("train_params") or {}).get("include_fresh"))
                if (_fresh_preview and committed and not dry_run
                        and not train_requested and not revisit):
                    store.append_event(
                        run_id, "phase_started", phase="preview",
                        message="Rendering preview snapshot of the next build's "
                                "corpus (fresh chats included; GPU-free, nothing "
                                "trains)...")
                    try:
                        from training.preview_build import build_preview_snapshot
                        info = build_preview_snapshot(run_id=run_id)
                        store.append_event(
                            run_id, "phase_done", phase="preview",
                            message=f"Preview snapshot {info['build_id']}: "
                                    f"{info['trained_rows']} would-train row(s) + "
                                    f"{info['preview_rows']} fresh preview row(s) — "
                                    f"open the Training review tab to inspect/repair "
                                    f"before the real training cycle.")
                    except Exception as e:
                        store.append_event(
                            run_id, "phase_error", phase="preview",
                            message=f"Preview snapshot failed (reflection is still "
                                    f"committed; the review tab keeps showing the "
                                    f"last build): {e}")

                if "reflection" not in stages:
                    store.finalize_run(run_id, "completed", {
                        "mutations_applied": committed,
                        "training_handed_off": train_requested,
                        "persona_produced": persona_produced,
                    })
                else:
                    # Reflection already finalized with the rich report summary;
                    # correct its production-commit flag without discarding it.
                    run_data = store.get_run(run_id)
                    summary = dict((run_data or {}).get("summary") or {})
                    summary["mutations_applied"] = committed
                    summary["training_handed_off"] = train_requested
                    summary["persona_produced"] = persona_produced
                    store.finalize_run(run_id, "completed", summary)

                # Episodic worklog: record the reflection as one episode in Ava's own
                # voice (distinct from the fine-grained activity mirror). A revisit
                # re-derives an old chat's meaning; a normal run consolidates the recent
                # ones. No open loop — reflection awaits nothing. A separate task consumes
                # the worklog; this only accumulates it.
                try:
                    from core import worklog
                    n_sessions = len(sessions) if sessions else 0
                    if revisit:
                        wl_summary = ("I revisited an earlier conversation and re-derived "
                                      "what I make of it now.")
                    elif n_sessions:
                        wl_summary = (f"I reflected on {n_sessions} of my past "
                                      f"conversation(s) and settled what I take from them.")
                    else:
                        wl_summary = ("I reflected on my recent conversations and settled "
                                      "what I take from them.")
                    worklog.record("reflection", wl_summary, refs={"run_id": run_id})
                except Exception:
                    import traceback as _tb
                    _tb.print_exc()

                # LoRA production must run with the base model unloaded from the
                # GPU, so we hand off to the watchdog: it stops THIS server, runs
                # the offline train cycle (repointing server_config.json's
                # adapter_id on a passing probe), then relaunches us. The run is
                # already finalized on disk above, so firing the POST — which may
                # be followed within seconds by the watchdog stopping us — is the
                # last thing we do.
                if train_requested:
                    store.append_event(
                        run_id, "phase_started", phase="train",
                        message="Handing off LoRA training to the watchdog "
                                "(the inference server will stop, train, and relaunch)...")
                    try:
                        # Tag the hand-off with this run_id so train_cycle drops the
                        # adapter it produces into reflections/<run_id>/adapter/.
                        train_params = dict(msg.get("train_params") or {})
                        train_params["run_id"] = run_id
                        # Risk-proportional validation: when the criterion flip rewrote
                        # any training target this run, force the regression probe (ignore
                        # skip-validation) — a judge-driven cycle must never promote
                        # unguarded. No-op runs keep the fast skip path.
                        # BUT validation is currently globally DISABLED (see
                        # training/validation_switch.py): train_cycle force-skips the probe
                        # regardless, so forcing skip_validation=False here would be a lie.
                        # Honor the same switch — force only when validation is armed, else
                        # tell the operator this higher-risk cycle promotes UNGUARDED.
                        try:
                            from training.validation_switch import VALIDATION_ENABLED as _VAL_ON
                        except Exception:
                            _VAL_ON = False
                        _run_rec = store.get_run(run_id) or {}
                        _overrides_n = int(
                            (_run_rec.get("summary") or {}).get("judge_overrides", 0) or 0)
                        if _overrides_n > 0 and _VAL_ON and train_params.get("skip_validation"):
                            train_params["skip_validation"] = False
                            store.append_event(
                                run_id, "phase_started", phase="train",
                                message=f"Validation forced: {_overrides_n} judge "
                                        "target override(s) this run — the probe will "
                                        "gate the adapter despite Skip-validation.")
                        elif _overrides_n > 0 and not _VAL_ON:
                            store.append_event(
                                run_id, "phase_started", phase="train",
                                message=f"{_overrides_n} judge target override(s) this run, "
                                        "but validation is globally DISABLED — the adapter "
                                        "promotes UNGUARDED (see AVA_OPEN_PROBLEMS.md → "
                                        "Validation).")
                        info = _trigger_watchdog_train(train_params)
                        store.append_event(
                            run_id, "phase_done", phase="train",
                            message="Training started on the watchdog; expect a brief "
                                    "inference restart while the cycle runs.")
                    except Exception as e:
                        store.append_event(
                            run_id, "phase_error", phase="train",
                            message=f"Training hand-off failed (is the watchdog "
                                    f"running on {_WATCHDOG_MGMT_URL}?): {e}")

        except Exception as e:
            store.append_event(run_id, "run_failed", message=f"Runner failed: {e}")
            store.finalize_run(run_id, "failed", {"error": str(e), "mutations_applied": False})
        finally:
            _reflection_run_active = False
            activity_log.clear_current()   # close the box-wide activity chip
            if revisit:
                _finalize_revisit_sidecar(store, run_id, revisit_sidecar_path,
                                          revisit_backup_path)
            _mark_activity()   # a finished run resets the idle-wander clock

    # Fire and forget — handler returns immediately; client polls for progress.
    loop.run_in_executor(_executor, _run)

async def handle_reflection_run_status(ws, msg: dict) -> None:
    """Return current state and progress counters for a single run."""
    run_id = str(msg.get("run_id") or "")
    run = _get_run_store().get_run(run_id)
    if run is None:
        await _send(ws, {"type": "error", "message": f"Unknown run_id: {run_id!r}"})
        return
    await _send(ws, {
        "type": "reflection_run_status",
        "run_id": run_id,
        # Fresh VRAM each poll: a reflection run streams no `done` messages, so
        # this is the only signal that keeps the client's memory readout live
        # while branch generation swings reserved VRAM around.
        "memory": _backend.memory_status(),
        "status": run["status"],
        "phase": run["phase"],
        "session_index": run["session_index"],
        "session_total": run["session_total"],
        "chunk_index": run["chunk_index"],
        "chunk_total": run["chunk_total"],
        "exchange_index": run["exchange_index"],
        "exchange_total": run["exchange_total"],
        "skipped_passes": run["skipped_passes"],
        # Compact live stats snapshot (elapsed/eta/global exchange x-y/peak VRAM/
        # discards/throughput) for the Sleep tab's stats panel.
        "stats": run.get("stats"),
        "latest_event_seq": run["latest_event_seq"],
        "summary": run["summary"],
    })

async def handle_stop_reflection_run(ws, msg: dict) -> None:
    """Request a graceful stop for a running or pending reflection run."""
    run_id = str(msg.get("run_id") or "")
    ok = _get_run_store().request_stop(run_id)
    if not ok:
        await _send(ws, {"type": "error",
                         "message": f"Cannot stop run {run_id!r}: not found or already finished"})
        return
    await _send(ws, {"type": "reflection_run_stop_requested", "run_id": run_id})

async def handle_list_reflection_runs(ws) -> None:
    """Return summary state for all known runs (most-recent first)."""
    runs = _get_run_store().list_runs()
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    await _send(ws, {"type": "reflection_runs_list", "runs": runs})

async def handle_get_reflection_run(ws, msg: dict) -> None:
    """Return the full state dict for a single run."""
    run_id = str(msg.get("run_id") or "")
    run = _get_run_store().get_run(run_id)
    if run is None:
        await _send(ws, {"type": "error", "message": f"Unknown run_id: {run_id!r}"})
        return
    await _send(ws, {"type": "reflection_run", "run": run})

async def handle_reflection_run_events(ws, msg: dict) -> None:
    """Return all events for a run with seq > after_seq (0 = return all).

    Clients use this to resume event consumption after a reconnect without
    missing or duplicating progress — key on (run_id, seq) for idempotence.
    """
    run_id = str(msg.get("run_id") or "")
    after_seq = int(msg.get("after_seq") or 0)
    store = _get_run_store()
    if store.get_run(run_id) is None:
        await _send(ws, {"type": "error", "message": f"Unknown run_id: {run_id!r}"})
        return
    events = store.get_events(run_id, after_seq=after_seq)
    run = store.get_run(run_id)
    await _send(ws, {
        "type": "reflection_run_events_batch",
        "run_id": run_id,
        "events": events,
        "latest_seq": (run or {}).get("latest_event_seq", 0),
    })
