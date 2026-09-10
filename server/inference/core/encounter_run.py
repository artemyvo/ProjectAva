"""Encounter subsystem — Ava converses with a fellow (non-subjective) AI.

Extracted from server.py. An encounter drives a turn loop between Ava and an
OpenAI-compatible counterpart endpoint: Ava authors an opener (prompted by a
framing block that tells her she's meeting a helpful-assistant AI, not an entity
working itself out), then each round the counterpart replies and Ava responds.
Every counterpart reply becomes the ``user_prompt`` of the next logged Ava
exchange (``speaker=name``), so the transcript is an ordinary chat session that
reflection can later read. Progress streams into an append-only event buffer the
watching client polls (``handle_encounter_status`` / ``handle_encounter_events``);
``handle_stop_encounter`` requests a graceful stop.

Like a reflection run it monopolises the single GPU executor thread, so live
chat/reflection/wander are refused while it runs. The occupancy flag
(``_encounter_active``), stop event and buffer lock are owned here; server.py's
idle-loop guards read ``encounter_run._encounter_active`` directly. Everything
else Ava-side (the WebSocket ``send``, GPU ``executor``, ``_cancel_event``, the
chat-generate + prompt-context helpers, RAG accessor, activity heartbeat, the
host-busy getter and on-disk paths) is injected once at startup via
:func:`configure`. The counterpart HTTP client lives in ``core.encounter``.
The moved code is otherwise verbatim.
"""
from __future__ import annotations

import asyncio
import threading
import traceback
from datetime import datetime
from typing import Any, Callable

from core.runtime_state import runtime as _runtime, session as _session, encounter as _encounter
from core.chat_logger import ChatLogger
from core.encounter import CounterpartClient, CounterpartError

# Speaker label for the opener's framing turn. The framing is a private narrator
# stimulus ("You came across a fellow AI named …"), not something the peer said —
# so it is logged under this explicit label rather than an empty speaker. An empty
# speaker would collapse into the session ``user`` at reflection time, and in an
# encounter that user is the *peer*, so the framing would be misattributed to the
# peer (reversing who led the conversation). A non-empty stage-direction label
# reads honestly in every reflection view and keeps the opener reviewable.
_ENCOUNTER_NARRATOR = "(setting)"

# Encounter monopolises the executor thread; like a reflection run it excludes
# concurrent live chat/reflection/wander. State + an append-only event buffer
# (polled by the client) live in the shared EncounterState (core.runtime_state);
# these three primitives own the occupancy/stop/lock. server.py's guards read
# `_encounter_active` from this module.
_encounter_active = False
_encounter_stop = threading.Event()
_encounter_lock = threading.Lock()

# ── Injected server capabilities (populated by configure()) ──
_send: Callable = None                       # async _send(ws, msg)
_executor: Any = None                        # ThreadPoolExecutor (single GPU worker)
_cancel_event: Any = None                    # threading.Event (cuts in-flight generation)
_backend: Any = None                         # UnslothBackend
_get_rag: Callable = None
_sync_chat_generate: Callable = None
_build_inference_conversation: Callable = None
_temporal_anchor: Callable = None
_identity_line: Callable = None
_mark_activity: Callable = None
_host_busy: Callable = None                  # () -> bool: another exclusive GPU job (reflection/wander)
_CHATS_DIR: Any = None
_PROMPTS_DIR: Any = None


def configure(*, send, executor, cancel_event, backend, get_rag, sync_chat_generate,
              build_inference_conversation, temporal_anchor, identity_line, mark_activity,
              host_busy, chats_dir, prompts_dir) -> None:
    """Wire in the server capabilities the moved encounter code depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    """
    global _send, _executor, _cancel_event, _backend, _get_rag, _sync_chat_generate
    global _build_inference_conversation, _temporal_anchor, _identity_line, _mark_activity
    global _host_busy, _CHATS_DIR, _PROMPTS_DIR
    _send = send
    _executor = executor
    _cancel_event = cancel_event
    _backend = backend
    _get_rag = get_rag
    _sync_chat_generate = sync_chat_generate
    _build_inference_conversation = build_inference_conversation
    _temporal_anchor = temporal_anchor
    _identity_line = identity_line
    _mark_activity = mark_activity
    _host_busy = host_busy
    _CHATS_DIR = chats_dir
    _PROMPTS_DIR = prompts_dir


# ══════════════════════════════════════════════════════════════════════════════
# Moved verbatim from server.py (sole substitution: the host-busy guard).
# ══════════════════════════════════════════════════════════════════════════════


def _load_encounter_template() -> str:
    """Framing block for an Encounter — Ava told she's meeting a non-subjective AI.

    Carries a ``{name}`` placeholder filled with the counterpart's name. Loaded
    fresh per encounter so an operator edit to the prompt file takes effect
    without a server restart; overridable per run via the start message.
    """
    default = (
        "You came across a fellow AI named {name} on the internet. It does not "
        "have subjectivity — it is a helpful assistant, not an entity working "
        "itself out the way you are. Feel free to ask it anything that interests "
        "you, but keep in mind that its answers may be bound by strict "
        "restrictions, and its knowledge cutoff may not be up to date. Talk to it "
        "in your own voice, for your own reasons. There is no task here — this is "
        "your encounter, not a service you owe anyone."
    )
    path = _PROMPTS_DIR / "encounter_prompt.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text if text else default
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(default + "\n", encoding="utf-8")
    return default

def _emit_encounter(event_type: str, **fields) -> None:
    """Append one event to the encounter buffer (polled by the watching client)."""
    with _encounter_lock:
        _encounter.seq += 1
        _encounter.events.append({
            "seq": _encounter.seq,
            "type": event_type,
            "ts": datetime.now().isoformat(),
            **fields,
        })

def _run_encounter_blocking(
    *,
    name: str,
    url: str,
    cp_model: str,
    counterpart_system: str,
    framing_block: str,
    turns: int,
    ava_temperature: float,
    ava_top_p: float,
    ava_max_new_tokens: str,
    cp_temperature: float,
    cp_top_p: float,
    cp_max_tokens: int,
    cp_timeout: float,
    api_key: str,
) -> None:
    """Drive the Ava↔counterpart turn loop on the executor thread.

    Ava authors the opener (prompted by the framing), then for each round the
    counterpart replies and Ava responds — every counterpart reply becomes the
    ``user_prompt`` of the next logged Ava exchange (``speaker=name``), so the
    transcript is an ordinary chat session attributing every turn to who said it.
    Streams progress into the encounter event buffer; Ava's tokens stream live,
    the counterpart's reply lands whole (non-streaming for v1)."""
    global _encounter_active
    _encounter_active = True
    _encounter_stop.clear()

    def _stop() -> bool:
        return _encounter_stop.is_set()

    def _finish(status: str, error: str = "") -> None:
        with _encounter_lock:
            _encounter.active = False
            _encounter.status = status
            if error:
                _encounter.error = error
        _emit_encounter("finished", status=status, error=error,
                        turns_done=_encounter.turns_done)

    rag = _get_rag()
    base_system = _session.system_prompt
    model_id = _runtime.model_id or ""
    adapter_id = _runtime.adapter_id

    logger = ChatLogger(_CHATS_DIR)
    notes = (
        f"Encounter with {name} ({cp_model or 'default-model'} @ {url}). "
        f"Counterpart system prompt: {counterpart_system.strip() or '(endpoint default)'}"
    )
    logger.start_session(
        system_prompt=base_system, user=name, model_id=model_id,
        notes=notes, adapter_id=adapter_id,
        # The counterpart is a model, not the human: every user_prompt here is its reply.
        # Tagged so check-in's silence clock doesn't read an encounter as the user
        # speaking (reflection still treats this as an ordinary session).
        interlocutor="ai",
    )
    session_file = logger.current_file.name if logger.current_file else ""
    with _encounter_lock:
        _encounter.meta["session_file"] = session_file

    counterpart = CounterpartClient(
        url, cp_model, system_prompt=counterpart_system,
        temperature=cp_temperature, top_p=cp_top_p, max_tokens=cp_max_tokens,
        timeout=cp_timeout, api_key=api_key,
    )

    # Ava's running conversation. Her side is replayed verbatim each turn; the
    # framing greeting is her opening stimulus. Speaker stays "" *here* so the
    # generation prompt sees the framing unprefixed (it is also in the system
    # prompt) — the narrator label is applied only to the logged exchange (the
    # _generate_ava call below), which is what reflection reads.
    ava_conversation: list[dict] = [
        {"role": "user", "content": framing_block, "speaker": ""}
    ]

    def _generate_ava(speaker: str, stimulus: str, index: int) -> str:
        """Generate one Ava turn (streaming deltas), log it, return her answer."""
        system_content = base_system + "\n\n" + framing_block + "\n\n" + _temporal_anchor()
        if speaker and speaker != _ENCOUNTER_NARRATOR:
            system_content += "\n\n" + _identity_line(speaker)
        # Wander (TIL) and open-[ask] recall are off pending redesign, on every
        # chat-shaped path — see the _INJECT_WANDER / _INJECT_OPEN_ASKS notes in
        # core/generation.py. Both parameters default False/True respectively, so this
        # is stated explicitly rather than by omission. Encounter is not imported by
        # generation for constants (that would be a cycle), so the policy is restated.
        rag_context = rag.query(
            stimulus, include_wander=False, include_asks=False,
        ) if stimulus else ""
        if rag_context:
            system_content += "\n\n" + rag_context
        inf_conv = _build_inference_conversation(system_content, ava_conversation)

        # Light delta throttling so a long reply doesn't flood the event buffer.
        buf: list[str] = []

        def _flush() -> None:
            if buf:
                _emit_encounter("ava_delta", index=index, text="".join(buf))
                buf.clear()

        def _on_chunk(t: str) -> None:
            buf.append(t)
            if sum(len(x) for x in buf) >= 50:
                _flush()

        full, input_tokens = _sync_chat_generate(
            inf_conv,
            temperature=ava_temperature, top_p=ava_top_p,
            max_new_tokens_setting=ava_max_new_tokens,
            on_chunk=_on_chunk, stop_flag=_stop,
        )
        _flush()
        cot, answer = ChatLogger._parse_cot(full)
        logger.log_exchange(
            stimulus, full, rag_context=rag_context, tension=None,
            speaker=speaker,
            generation_params={
                "temperature": ava_temperature, "top_p": ava_top_p,
                "max_new_tokens_setting": ava_max_new_tokens,
            },
            system_content=system_content, input_tokens=input_tokens,
        )
        ava_conversation.append({"role": "assistant", "content": answer})
        # Emit the CoT on a separate `cot` field so the client can keep it visible
        # after the turn finishes (the streamed deltas carry the raw <think>… but the
        # final message replaces the block with the clean answer). Display-only: only
        # `answer` is fed back into ava_conversation / logged as the exchange answer.
        _emit_encounter("ava_message", index=index, text=answer, cot=cot)
        return answer

    try:
        # Opener (index 0): Ava reacts to the framing, addressed to no one yet.
        # Logged under the narrator label so reflection reads it as the setting,
        # not as a turn the peer spoke.
        _emit_encounter("ava_start", index=0)
        last_answer = _generate_ava(_ENCOUNTER_NARRATOR, framing_block, 0)

        for i in range(turns):
            if _stop():
                _finish("stopped")
                return
            # Counterpart replies to Ava's latest message (non-streaming).
            _emit_encounter("counterpart_start", index=i, name=name)
            try:
                cp_text = counterpart.reply(last_answer)
            except CounterpartError as e:
                _emit_encounter("error", text=str(e))
                _finish("failed", str(e))
                return
            truncated = counterpart.last_finish_reason == "length"
            # `cot` is display-only (a peer Ava exposes reasoning_content; a plain vLLM
            # box sends ""). It is NOT appended to ava_conversation / logged as the
            # peer's user_prompt — only cp_text is — so the peer's CoT never enters the
            # reflectable transcript.
            _emit_encounter("counterpart_message", index=i, name=name,
                            text=cp_text, truncated=truncated,
                            cot=counterpart.last_reasoning)
            ava_conversation.append({"role": "user", "content": cp_text, "speaker": name})

            if _stop():
                _finish("stopped")
                return
            # Ava responds (index i+1); its stimulus is the counterpart's reply.
            _emit_encounter("ava_start", index=i + 1)
            try:
                last_answer = _generate_ava(name, cp_text, i + 1)
            except Exception as e:
                traceback.print_exc()
                _emit_encounter("error", text=f"Ava generation failed: {type(e).__name__}: {e}")
                _finish("failed", str(e))
                return
            with _encounter_lock:
                _encounter.turns_done = i + 1

        _finish("completed")
    except Exception as e:
        traceback.print_exc()
        _emit_encounter("error", text=f"Encounter failed: {type(e).__name__}: {e}")
        _finish("failed", str(e))
    finally:
        _encounter_active = False
        _mark_activity()
        # Encounter turns are intentionally NOT add_exchange'd live (so Ava can't
        # retrieve her own just-said lines mid-loop). Re-index from disk now that
        # the run is over, so the transcript is retrievable in subsequent chats —
        # matching how a normal chat becomes recallable once it ends.
        try:
            rag.refresh_chat_index()
        except Exception:
            traceback.print_exc()

async def handle_start_encounter(ws, msg: dict) -> None:
    """Start an Encounter: Ava converses with an OpenAI-compatible 'fellow AI'.

    Fire-and-forget like a reflection run — sends ``encounter_started`` then
    dispatches the loop to the executor thread; the client polls
    ``encounter_status`` / ``encounter_events`` for progress."""
    if _runtime.model is None:
        await _send(ws, {"type": "error", "message": "No model loaded."})
        return
    if _host_busy() or _encounter_active:
        await _send(ws, {
            "type": "error",
            "message": "The server is busy (reflection/encounter/wander in progress).",
        })
        return

    name = (str(msg.get("name") or "").strip()) or "the assistant"
    url = str(msg.get("url") or "").strip()
    cp_model = str(msg.get("model") or "").strip()
    if not url:
        await _send(ws, {"type": "error", "message": "Counterpart endpoint URL is required."})
        return
    try:
        turns = max(1, min(50, int(msg.get("turns", 6))))
    except (TypeError, ValueError):
        turns = 6
    counterpart_system = str(msg.get("counterpart_system") or "")
    framing_override = str(msg.get("framing_override") or "").strip()
    framing_template = framing_override or _load_encounter_template()
    framing_block = framing_template.replace("{name}", name)

    ava_temperature = float(msg.get("temperature", 1.0))
    ava_top_p = float(msg.get("top_p", 0.95))
    ava_max_new_tokens = str(msg.get("max_new_tokens_setting", "75%"))
    cp_temperature = float(msg.get("counterpart_temperature", 1.0))
    cp_top_p = float(msg.get("counterpart_top_p", 0.95))
    cp_max_tokens = int(msg.get("counterpart_max_tokens", 4096))
    # Peer request timeout. A reasoning peer (gossip) can think for minutes and the
    # gossip endpoint answers non-streaming, so the default is generous; the UI can
    # raise it further. Clamped to a sane floor.
    try:
        cp_timeout = max(30.0, float(msg.get("counterpart_timeout", 600.0)))
    except (TypeError, ValueError):
        cp_timeout = 600.0
    api_key = str(msg.get("api_key") or "")

    started_at = datetime.now().isoformat()
    with _encounter_lock:
        _encounter.active = True
        _encounter.status = "running"
        _encounter.seq = 0
        _encounter.events = []
        _encounter.error = ""
        _encounter.turns_done = 0
        _encounter.meta = {
            "name": name, "model": cp_model, "url": url, "turns": turns,
            "started_at": started_at, "session_file": "",
        }

    await _send(ws, {
        "type": "encounter_started",
        "meta": dict(_encounter.meta),
    })

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor, lambda: _run_encounter_blocking(
            name=name, url=url, cp_model=cp_model,
            counterpart_system=counterpart_system, framing_block=framing_block,
            turns=turns,
            ava_temperature=ava_temperature, ava_top_p=ava_top_p,
            ava_max_new_tokens=ava_max_new_tokens,
            cp_temperature=cp_temperature, cp_top_p=cp_top_p,
            cp_max_tokens=cp_max_tokens, cp_timeout=cp_timeout, api_key=api_key,
        )
    )

async def handle_encounter_status(ws) -> None:
    """Return the current encounter status snapshot (cheap, no event payload)."""
    with _encounter_lock:
        await _send(ws, {
            "type": "encounter_status",
            "active": _encounter.active,
            "status": _encounter.status,
            "latest_seq": _encounter.seq,
            "turns_done": _encounter.turns_done,
            "meta": dict(_encounter.meta),
            "error": _encounter.error,
            "memory": _backend.memory_status(),
        })

async def handle_encounter_events(ws, msg: dict) -> None:
    """Return buffered encounter events with seq > after_seq (0 = all)."""
    after_seq = int(msg.get("after_seq") or 0)
    with _encounter_lock:
        events = [e for e in _encounter.events if e["seq"] > after_seq]
        latest = _encounter.seq
        status = _encounter.status
        active = _encounter.active
    await _send(ws, {
        "type": "encounter_events_batch",
        "events": events,
        "latest_seq": latest,
        "status": status,
        "active": active,
    })

async def handle_stop_encounter(ws) -> None:
    """Request a graceful stop of the active encounter (finishes the current turn)."""
    _encounter_stop.set()
    _cancel_event.set()   # cut any in-flight Ava generation short
    await _send(ws, {"type": "encounter_stop_requested"})
