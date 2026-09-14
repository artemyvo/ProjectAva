"""Session CRUD — the chat-session lifecycle handlers, extracted from server.py.

The WebSocket handlers that manage the active chat session and the transcript
archive on disk:

  * handle_clear_context — start a fresh log session tagged with the speaker
    ("New chat"): no prior context is carried forward.
  * handle_set_session_notes — attach operator notes to the active session.
  * handle_set_reflection_feedback — annotate only the latest active Ava reply
    for revision/persona reflection (never as a conversation turn).
  * handle_retry_last_exchange — roll the latest completed exchange off the
    active session (Chat-tab "Retry") so a collapsed reply can be redone under
    adjusted sampling; the completed-turn sibling of the in-flight Stop/discard.
  * handle_list_sessions / handle_get_session — list completed sessions or
    fetch one's JSON.
  * handle_load_session — adopt a past session as the active conversation.
  * handle_reset_session_reflection — delete a chat's sidecar so it re-reflects
    from scratch (Chat-tab "Re-reflect chat").
  * handle_delete_session — delete a transcript (+ its sidecar).

Never imports server: the capabilities it needs (WebSocket send, the RAG
accessor, and the chats dir) are injected once at startup via :func:. Session/
model state is read from core.runtime_state.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session
from core.chat_logger import (
    ChatLogger, delete_exchange, mark_exchange_corrupt, rewrite_exchange_history,
)
from core.chat_sidecar import (
    ChatSidecar, facts_path_for, iter_chat_json_files, is_chat_session_json,
    sidecar_path_for, summary_path_for,
)
# The one definition of how a wander row identifies itself in the training render
# (``source_session == "wander:<ts>"``); the review payload builder owns it.
from core.training_review import WANDER_PREFIX

# ── Injected server capabilities (populated by configure()) ──
_send: Callable = None                       # async _send(ws, msg)
_get_rag: Callable = None
_is_reflection_active: Callable = None
_mark_activity: Callable = lambda: None      # reset the shared idle clock
_CHATS_DIR: Any = None
_DATA_DIR: Any = None

def configure(*, send, get_rag, chats_dir, data_dir=None,
              is_reflection_active=lambda: False, mark_activity=None) -> None:
    """Wire in the server capabilities the moved session-CRUD code depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    ``data_dir`` is the reflection data root (``inference/data``, distinct from the
    ``server/data/chats`` root) — needed to purge a chat's background per-chat staging when
    an already-reflected chat is un-frozen on in-place continue.
    ``mark_activity`` is ``idle_scheduler.mark_activity`` — the dataset-repair handlers call
    it so an operator working the Training review tab counts as the box being busy (see
    ``handle_apply_regenerated_exchange``).
    """
    global _send, _get_rag, _is_reflection_active, _mark_activity, _CHATS_DIR, _DATA_DIR
    _send = send
    _get_rag = get_rag
    _is_reflection_active = is_reflection_active
    if mark_activity is not None:
        _mark_activity = mark_activity
    _CHATS_DIR = chats_dir
    _DATA_DIR = data_dir


# ══════════════════════════════════════════════════════════════════════════════
# Moved verbatim from server.py.
# ══════════════════════════════════════════════════════════════════════════════


async def handle_clear_context(ws, msg: dict) -> None:
    speaker = str(msg.get("user", "")).strip()

    # "New chat" starts a genuinely fresh conversation: no prior context is carried
    # forward. Earlier this carried the closing session's tail (a cheap "remember
    # what we just discussed" continuity), but that made a New chat silently behave
    # like a continuation — most visibly when the tail ended on an unanswered Ava
    # question, so the fresh session opened as if still pressing it. Continuity is
    # available deliberately via loading/continuing a past session instead.
    # The operator is at the keyboard. Releasing a chat by hand must not be the
    # cue for an idle job: the released transcript enters the background backlog
    # the moment the logger lets go of it, and without this the 30-min idle window
    # was usually long spent (the user had been away for hours with the chat
    # still fenced), so background reflection fired on it right as they returned.
    _mark_activity()
    _session.user = speaker
    _session.conversation = []
    _session.surfaced_keys = []
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    logger = ChatLogger(_CHATS_DIR)
    logger.start_session(
        _session.system_prompt,
        user=speaker,
        model_id=_runtime.model_id,
        adapter_id=_runtime.adapter_id,
    )
    _session.logger = logger
    _get_rag().set_current_session_file(logger.current_file)

    # A chat merge (Migrate tab's two-way sync) may have dropped *foreign*
    # transcripts straight into hot/chats, bypassing the incremental index. A clear
    # is a natural seam to fold any such new chats into RAG so they're retrievable
    # in the fresh session — the whole point of a merge onto a new box. Runs off the
    # event loop (a re-embed) and is a cheap no-op when nothing new landed.
    try:
        asyncio.get_running_loop().run_in_executor(None, _get_rag().index_new_chats)
    except Exception:
        pass

    await _send(ws, {
        "type": "context_cleared",
        "carried_exchanges": 0,
        "continued_from": "",
    })

def release_active_session(reason: str = "") -> Optional[str]:
    """Drop the server-side active chat WITHOUT starting a new one.

    Called from the WebSocket handler when the client genuinely disconnects (not
    when a newer client supersedes it — that handoff keeps the session on purpose).
    Returns the released transcript's filename, or None if there was nothing live.

    Why: the active transcript is fenced from every background reader by "the
    logger still points at it" — `background_reflection.list_backlog`, the sidecar
    writers' `live_session`, the stale-reach-out sweep's exemption, chat RAG's
    self-reference fence. Nothing ever cleared the logger short of "New chat", so a
    user who closed the client for hours left their last chat locked the whole
    time; the box idled with a reflectable chat it was forbidden to touch, and the
    chat was reflected the moment they came back and pressed "New chat" instead.

    It is safe to drop here rather than keep for the reconnect: the client resets
    its own view to an empty conversation on every connect (`conversation_history
    = []`, `_session_mode = "live"`), so server-side continuity across a
    disconnect was already a phantom — a turn typed after reconnecting would have
    appended to a transcript the user could no longer see, answered against
    history the UI did not show. A chat the user wants to pick up again is one
    click away in the list (fork-on-continue, or in place for an Ava-initiated
    one). An in-flight generation is unaffected: the handler holds its own
    `logger` reference and the socket close already set the cancel event, so its
    turn is never logged against the fresh state.
    """
    logger = _session.logger
    released = None
    if logger is not None and logger.current_file is not None:
        released = logger.current_file.name
    _session.logger = None
    _session.conversation = []
    _session.surfaced_keys = []
    try:
        _get_rag().set_current_session_file(None)
    except Exception:
        pass
    if released:
        print(f"[session] released active chat {released}"
              + (f" ({reason})" if reason else ""))
    return released


async def handle_set_session_notes(ws, msg: dict) -> None:
    notes = str(msg.get("notes", ""))
    logger = _session.logger
    if logger is None:
        # No active session yet — silently no-op; the client will retry once a session exists.
        await _send(ws, {"type": "session_notes_saved", "active": False})
        return
    logger.set_notes(notes)
    await _send(ws, {"type": "session_notes_saved", "active": True})

async def handle_set_reflection_feedback(ws, msg: dict) -> None:
    """Attach notification-only user feedback to the latest active reply."""
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; Meta feedback cannot change its source transcript.",
        })
        return
    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if logger is None or active_file is None:
        await _send(ws, {"type": "error", "message": "There is no active chat to annotate."})
        return
    if ChatSidecar(_CHATS_DIR).is_reflected(active_file.name):
        await _send(ws, {
            "type": "error",
            "message": "This chat has already been reflected; its feedback window is closed.",
        })
        return
    try:
        feedback = logger.set_latest_reflection_feedback(
            str(msg.get("exchange_id") or ""),
            str(msg.get("text") or ""),
            speaker=str(msg.get("speaker") or ""),
        )
    except (ValueError, OSError) as exc:
        await _send(ws, {"type": "error", "message": str(exc)})
        return
    await _send(ws, {
        "type": "reflection_feedback_saved",
        "exchange_id": str(msg.get("exchange_id") or ""),
        "feedback": feedback,
    })

async def handle_retry_last_exchange(ws, msg: dict) -> None:
    """Roll the latest completed exchange off the active session so it can be redone.

    The Chat-tab **Retry** action: when a reply came out degenerate/collapsed, drop it
    (and its user turn) from the active transcript AND the in-memory conversation so it
    never reaches reflection/training, and hand the user prompt back so the operator can
    resend it under adjusted sampling. It is the completed-turn sibling of the in-flight
    Stop (discard) — Stop rolls back a reply still streaming; this rolls back one that
    already landed and was logged. Refused while a reflection run owns the transcript, on
    an already-reflected session, when there is nothing to retry, or when the only turn is
    a synthetic Ava opener (no real user prompt to resend). The active session is fenced
    from RAG, so the removed reply was never indexed — nothing to unwind there.
    """
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; the active transcript cannot be edited now.",
        })
        return
    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if logger is None or active_file is None or logger.exchange_count == 0:
        await _send(ws, {"type": "error", "message": "There is no reply to retry."})
        return
    if ChatSidecar(_CHATS_DIR).is_reflected(active_file.name):
        await _send(ws, {
            "type": "error",
            "message": "This chat has already been reflected on and can no longer be edited.",
        })
        return
    # An Ava-initiated outreach's exchange 0 is her own opener (its user_prompt is the
    # synthetic "(initiative)" stimulus) — there is no real user turn to resend, so its
    # sole exchange is not retryable. Once the user has replied (≥2 exchanges) the last
    # exchange is an ordinary user->Ava turn and rolls back normally.
    if logger.initiated_by == "ava" and logger.exchange_count <= 1:
        await _send(ws, {
            "type": "error",
            "message": "Ava's opening message cannot be retried.",
        })
        return

    removed = logger.remove_last_exchange()
    if not removed:
        await _send(ws, {"type": "error", "message": "There is no reply to retry."})
        return

    # Roll the same exchange out of the live conversation: pop the trailing assistant
    # turn then its user turn (defensively — only pop what actually matches, so a
    # restored/continued conversation tail can't be corrupted).
    conv = _session.conversation
    if conv and conv[-1].get("role") == "assistant":
        conv.pop()
    if conv and conv[-1].get("role") == "user":
        conv.pop()

    await _send(ws, {
        "type": "retry_ready",
        "user_prompt": removed.get("user_prompt", ""),
        "speaker": removed.get("speaker", ""),
        # Index the resend will re-occupy (== count after removal).
        "exchange_index": logger.exchange_count,
    })

async def handle_list_sessions(ws) -> None:
    sessions = []
    current_file = None
    logger = _session.logger
    if logger is not None:
        current_file = logger.current_file

    for path in iter_chat_json_files(_CHATS_DIR):
        if current_file is not None and path.resolve() == Path(current_file).resolve():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exchanges = data.get("exchanges", [])
        if not isinstance(exchanges, list) or not exchanges:
            continue
        initiated_by = (data.get("initiated_by") or "").strip()
        # An Ava-initiated outreach session opens with her own message (exchange 0's
        # assistant_response), not a user turn — its user_prompt is the synthetic
        # "(initiative)" stimulus, which reads as noise in the list. Preview the opener.
        if initiated_by == "ava" and exchanges:
            first_message = exchanges[0].get("assistant_response", "")
        else:
            first_message = exchanges[0].get("user_prompt", "") if exchanges else ""
        user = (data.get("user") or "").strip()
        if not user and exchanges:
            user = (exchanges[0].get("speaker") or "").strip()
        sessions.append({
            "filename": path.name,
            "timestamp": data.get("timestamp", ""),
            "exchange_count": len(exchanges),
            "first_message": first_message,
            "user": user,
            "continued_from": (data.get("continued_from") or "").strip(),
            "initiated_by": initiated_by,
        })

    await _send(ws, {"type": "sessions_list", "sessions": sessions})

async def handle_match_anchors(ws, msg: dict) -> None:
    """Preview which stored exchange anchors a piece of user text would match.

    The query-side counterpart of the anchor producer, and a **diagnostic only**: it
    retrieves nothing, injects nothing, and touches no session state. It exists because
    the retrieval design has several knobs (inflection handling, tag weighting, the filler
    cutoff) whose correctness is only visible against real text, and tuning them by
    watching live chat would mean changing Ava's behaviour to run an experiment.

    Matching is lexical and generation-free, so this is cheap enough to run against a
    half-typed message: no GPU, no embedder, no model. Deliberately does NOT collapse per
    chat — seeing two exchanges of one conversation match is information here, where a
    retrieval caller would collapse them into one slot.
    """
    text = str(msg.get("text") or "")
    limit = int(msg.get("limit") or 20)
    try:
        from core import exchange_anchor
        from training.reflections_path import archive_chats_dir

        # Three roots, lowest precedence first (`load_corpus` lets later dirs win).
        #
        # The checkpoint is the one that matters in practice. The **background** per-chat
        # pass — which is what produces most anchors — runs against a throwaway staging
        # dir and commits each finished chat's sidecar to
        # `data/hot/reflection_checkpoint/chats/`; those sidecars only reach the live
        # chats dir when the NEXT normal reflection run starts and calls
        # `fold_checkpoint_to_live`. Reading live alone therefore reports "no anchors"
        # for hours after they were visibly generated — which defeats the point of a
        # tool for watching them being produced. The checkpoint copy is newer than live
        # by construction, so it wins.
        #
        # The in-flight `background_staging/` dir is deliberately NOT read: that chat is
        # mid-reflection and its staging is discarded if the run is preempted, so its
        # anchors may never land, and a diagnostic must not show state that will vanish.
        ckpt_chats = None
        if _DATA_DIR is not None:
            try:
                from core.reflection_staging import get_checkpoint_paths
                ckpt_chats = get_checkpoint_paths(_DATA_DIR)["chats_dir"]
            except Exception:
                ckpt_chats = None

        live_entries = exchange_anchor.load_corpus(archive_chats_dir(), _CHATS_DIR)
        entries = exchange_anchor.load_corpus(
            archive_chats_dir(), _CHATS_DIR, ckpt_chats)
        matches = exchange_anchor.match_query(text, entries, limit=limit)
        tagged = sum(1 for e in entries if e.get("tags"))
        # Surface the split so an empty or surprising result is self-explaining rather
        # than mysterious: `pending` are anchors a normal reflection run has not folded
        # into the live sidecars yet.
        pending = max(0, len(entries) - len(live_entries))
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Anchor match failed: {e}"})
        return

    await _send(ws, {
        "type": "anchor_matches",
        "text": text,
        "matches": matches,
        "corpus": {"anchors": len(entries), "with_tags": tagged, "pending": pending},
    })


async def handle_get_session(ws, msg: dict) -> None:
    filename = msg.get("filename", "")
    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return

    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Could not read session: {e}"})
        return

    await _send(ws, {"type": "session_data", "data": data})

async def handle_load_session(ws, msg: dict) -> None:
    """Load a past session's exchanges into the active conversation state.

    Normally this *forks*: a fresh session file is started (``continued_from`` the
    original) so the past transcript stays immutable. With ``in_place`` true the active
    logger instead resumes the SAME file, so new exchanges append to it — used to reply
    to an Ava-initiated outreach in its own file, keeping the reversed conversation
    (masked opener + the user's reply + Ava's response) as one reflectable session."""
    filename = msg.get("filename", "")
    in_place = bool(msg.get("in_place"))
    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return

    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Could not read session: {e}"})
        return

    session_user = (data.get("user") or "").strip()
    conversation = []
    for ex in data.get("exchanges", []):
        speaker = (ex.get("speaker") or "").strip() or session_user
        conversation.append(
            {"role": "user", "content": ex.get("user_prompt", ""), "speaker": speaker}
        )
        conversation.append({"role": "assistant", "content": ex.get("assistant_response", "")})

    _session.conversation = conversation
    _session.user = session_user
    _session.surfaced_keys = []

    _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    logger = ChatLogger(_CHATS_DIR)
    if in_place:
        # Continue in the SAME file — the reply appends to this transcript. If this chat
        # was already reflected (an Ava-initiated session reopened after its background
        # reflection froze it), UN-FREEZE it: the freeze would otherwise hide the appended
        # turns from reflection and training forever (reflect-once + list_backlog skip a
        # frozen sidecar, and unlike a normal chat this one is resumed in place, not forked).
        # Clearing the stamps re-enters it into the backlog so the whole, now-longer
        # conversation re-reflects; for a chat frozen only at the chat_reflected stage we
        # also purge its background checkpoint/pending artifacts so the next normal run can't
        # re-freeze it. Best-effort — never block the resume on a housekeeping failure.
        try:
            sc = ChatSidecar(_CHATS_DIR)
            if sc.clear_reflection_freeze(filename) and _DATA_DIR is not None:
                from core import reflection_staging
                reflection_staging.purge_background_artifacts(_DATA_DIR, filename)
        except Exception:
            pass
        logger.resume_session(path)
    else:
        # Fresh logger for the continuation so new exchanges go to a new file.
        logger.start_session(
            _session.system_prompt,
            user=session_user,
            model_id=_runtime.model_id,
            adapter_id=_runtime.adapter_id,
            continued_from=filename,
        )
    _session.logger = logger
    _get_rag().set_current_session_file(logger.current_file)

    await _send(ws, {
        "type": "session_loaded",
        "data": data,
        "continued_from": "" if in_place else filename,
        "in_place": in_place,
        "filename": logger.current_file.name if logger.current_file else filename,
        "exchange_count": len(conversation) // 2,
    })

async def handle_mark_corrupt(ws, msg: dict) -> None:
    """Flag a stored CoT and/or reply of a past exchange as corrupt.

    Set from the Training review tab (a row it found in a training render whose CoT or
    answer is garbage from a logging/generation bug). The flag lands on the source
    exchange in the chat JSON so it travels with the exchange that originated it —
    reflection then treats a corrupt CoT as missing and insists on a re-derived IDEAL
    for a corrupt reply, and the next training build drops the corrupt content unless
    an IDEAL replaced it. Flag-only: the next Sleep/build consumes it.

    Path-guarded to hot/chats. Refuses the ACTIVE session (the live logger would clobber
    the flag on its next save) — clear context first — and a reflection run in progress
    (it could be reflecting this chat right now, and we are about to delete its sidecar).
    Unlike Meta feedback it deliberately does NOT refuse a *reflected* (frozen) session: a
    corrupt row is discovered after the chat trained, so the whole point is to mark a frozen
    one.

    **Sidecar invalidation:** the sidecar's frozen verdict/target for this chat was
    produced while the revision pass could see the now-flagged CoT/reply — both as this
    exchange's own reasoning AND as context for every later exchange in the chat — so the
    whole sidecar is poisoned, not just the one exchange's target (a "revised" target can be
    contaminated by corrupt context too). Setting a flag therefore DELETES the sidecar,
    un-freezing the chat: it stops contributing to training at once (an unfrozen chat is not
    a build bundle) and the next Sleep re-reflects the whole chat clean under the
    corrupt-aware revision path (corrupt CoT blanked → treated as missing; corrupt reply
    forced to a re-derived IDEAL). The per-exchange original-vs-regenerated marker the
    operator would otherwise want already exists as the sidecar's ``target_source``; we drop
    the file rather than patch a field because of the context contamination."""
    _mark_activity()   # operator is at the box cleaning the dataset — not idle
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; corruption cannot be marked until it finishes.",
        })
        return
    filename = str(msg.get("filename", ""))
    try:
        exchange_index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "Invalid exchange index."})
        return

    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
        if not is_chat_session_json(path):
            raise ValueError("not a chat transcript")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return

    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    # Never edit the file the live logger is appending to — its next save would drop
    # the flag. The operator clears context first (the review tab shows frozen chats).
    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if active_file is not None and active_file.resolve() == path:
        await _send(ws, {
            "type": "error",
            "message": "That session is the active conversation — start a New chat before marking it.",
        })
        return

    # Each flag is applied only when present in the message, so the two toggles are
    # independent (omitting one leaves it untouched).
    kwargs: dict = {}
    if "corrupt_cot" in msg:
        kwargs["corrupt_cot"] = bool(msg.get("corrupt_cot"))
    if "corrupt_response" in msg:
        kwargs["corrupt_response"] = bool(msg.get("corrupt_response"))
    if not kwargs:
        await _send(ws, {"type": "error", "message": "No corruption flag supplied."})
        return

    try:
        state = mark_exchange_corrupt(path, exchange_index, **kwargs)
    except (ValueError, OSError) as exc:
        await _send(ws, {"type": "error", "message": str(exc)})
        return

    # Delete the poisoned sidecar when this mark leaves the exchange flagged corrupt, so the
    # chat un-freezes and re-reflects clean (see the docstring). Best-effort: a missing
    # sidecar just means the chat was already unfrozen / never reflected.
    sidecar_invalidated = False
    if state["corrupt_cot"] or state["corrupt_response"]:
        sc = sidecar_path_for(path)
        try:
            if sc.exists():
                sc.unlink()
                sidecar_invalidated = True
        except OSError:
            pass

    await _send(ws, {
        "type": "exchange_corrupt_marked",
        "filename": filename,
        "exchange_index": exchange_index,
        "corrupt_cot": state["corrupt_cot"],
        "corrupt_response": state["corrupt_response"],
        "sidecar_invalidated": sidecar_invalidated,
    })

async def handle_set_training_ban(ws, msg: dict) -> None:
    """Ban (or un-ban) one training-review row from ever training again.

    The Training review tab's third verdict, beside "repair it" (Regenerate / hand-edit +
    Apply) and "leave it": some rows cannot be repaired into something worth learning from,
    and substituting a hand-written target for them would be inventing a memory rather than
    correcting one. A ban says *this never trains* and nothing more.

    Two row kinds, because the tab's corpus has two:

      * a **chat exchange** (``target`` = the transcript filename, ``exchange_index`` = its
        position) — the flag lands on the exchange's sidecar record, which is what
        ``training.dialogue_source.build_dialogue_anchor`` refuses, so the row leaves the
        corpus on the next build. The exchange itself stays in the transcript (it happened;
        later turns answered in its light; RAG may recall it) until the operator finalizes
        with "Rewrite history", which deletes it.
      * a **wander capture** (``target`` = ``wander:<ts>``, no exchange index) — the flag
        lands on the corpus record itself (``core.wander_sft``), which drops it from both
        the build and the chat-RAG wander channel at once. This is the case the feature was
        built for: a wander whose generation came out malformed has no conversation behind
        it to repair, so banning is the only sensible verdict.

    Reversible either way (``banned: false`` lifts it) right up until the deletion. Same
    guards as the other dataset-repair handlers: refuses while a reflection run owns the
    box, and refuses the ACTIVE chat (its live logger's next save would drop the flag).
    Replies ``training_ban_set``.
    """
    _mark_activity()   # operator is at the box cleaning the dataset — not idle
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; cannot change training bans until it finishes.",
        })
        return
    target = str(msg.get("target") or "").strip()
    banned = bool(msg.get("banned", True))
    if not target:
        await _send(ws, {"type": "error", "message": "No row to ban."})
        return

    if target.startswith(WANDER_PREFIX):
        from core.wander_sft import set_banned as _set_wander_banned
        # A False return is a no-op (the record already reads that way) as much as a miss,
        # and either way the corpus now holds the state the operator asked for — so the
        # result is reported, not failed on.
        _set_wander_banned(target[len(WANDER_PREFIX):], banned)
        # A banned capture must leave retrieval too, not just the build.
        try:
            asyncio.get_running_loop().run_in_executor(
                None, _get_rag().refresh_wander)
        except Exception:
            pass
        await _send(ws, {"type": "training_ban_set", "target": target,
                         "exchange_index": None, "banned": banned, "kind": "wander"})
        return

    try:
        exchange_index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "Invalid exchange index."})
        return
    try:
        path = (_CHATS_DIR / target).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
        if not is_chat_session_json(path):
            raise ValueError("not a chat transcript")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return
    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {target}"})
        return

    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if active_file is not None and active_file.resolve() == path:
        await _send(ws, {
            "type": "error",
            "message": "That session is the active conversation — start a New chat before banning from it.",
        })
        return

    if not ChatSidecar(_CHATS_DIR).set_exchange_banned(target, exchange_index, banned):
        await _send(ws, {"type": "error", "message": "Could not write the ban to the sidecar."})
        return

    await _send(ws, {"type": "training_ban_set", "target": target,
                     "exchange_index": exchange_index, "banned": banned, "kind": "chat"})


async def handle_apply_regenerated_exchange(ws, msg: dict) -> None:
    """Persist an operator-reviewed regeneration straight into the chat sidecar.

    The Training review tab's Regenerate flow: after the operator reviews the loaded
    adapter's fresh re-answer (``generation.handle_regenerate_exchange``) in a pop-up and
    hits Apply, this writes a fresh trainable target for that ONE exchange directly to the
    sidecar and **locks it (human-validated)** — an immediate, hand-authored edit that does
    NOT wait for a Sleep pass. The lock is *per-exchange*: re-reflection AND a revisit run
    (which bypasses the session-level reflect-once freeze) skip a locked exchange, so the
    reviewed target can never be overwritten by a fresh pass over the original (corrupt)
    transcript — while the rest of the chat still re-reflects normally.

    **Faithfulness rule** (a CoT must belong to the reply it is paired with — a mismatch is
    what erodes reasoning over rebuilds): a regenerated REPLY carries its own freshly authored
    CoT (the pair was generated together); regenerating only the CoT grafts the new
    ``<think>`` onto the trusted original reply:
      * corrupt_response  -> target = <think>{new_cot}</think>\\n\\n{new_reply}
      * corrupt_cot only  -> target = <think>{new_cot}</think>\\n\\n{kept_reply}
        where *kept_reply* is the answer the client is DISPLAYING (the current trained
        target's reply, sent as ``original_reply``), NOT necessarily the raw transcript
        ``assistant_response`` — those differ when the trained target was a revised/IDEAL
        answer (the usual case for an empty-CoT row), and "keep the reply" means the one
        the operator sees. Falls back to ``assistant_response`` when the client omits it.

    The write lands as ``target_source == "revised"`` (so ``dialogue_source`` trains it
    verbatim and ``_pick_corrupt_chat`` treats it repaired), tagged ``manual_regen`` /
    ``chat_manual_regen_v1`` for the forensic audit, and ``locked: true`` so re-reflection +
    revisit preserve it. Any lingering corrupt flags on the exchange are cleared. Path-guarded
    to hot/chats; refuses the ACTIVE session and a live reflection run (same guards as
    mark_corrupt). Assumes UI + server share host/dir, so the write is immediately visible to
    the tab.

    Resets the shared idle clock: an operator repairing rows is at the box, but a hand-edited
    Apply drives no generation at all, so without this the clock keeps running through a long
    cleaning session and the autonomous idle jobs (wander / outreach / background reflection)
    wake into the middle of it. Same reason as ``generation.handle_regenerate_exchange``."""
    _mark_activity()   # operator is at the box cleaning the dataset — not idle
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; cannot apply a regeneration until it finishes.",
        })
        return
    filename = str(msg.get("filename", ""))
    try:
        exchange_index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "Invalid exchange index."})
        return
    corrupt_cot = bool(msg.get("corrupt_cot"))
    corrupt_response = bool(msg.get("corrupt_response"))
    if not (corrupt_cot or corrupt_response):
        await _send(ws, {
            "type": "error",
            "message": "Nothing selected to apply — check Corrupt CoT and/or Corrupt reply.",
        })
        return
    new_cot = str(msg.get("new_cot") or "").strip()
    new_reply = str(msg.get("new_reply") or "").strip()
    # The reply to KEEP for a CoT-only regen. The client sends the answer it is displaying
    # (the currently-trained target's reply, parsed from the render), which is what the
    # operator means by "keep the reply" — this can differ from the raw transcript
    # `assistant_response` when the trained target was a revised/IDEAL answer (the usual
    # case for an empty-CoT row). Falls back to the transcript reply when absent.
    kept_reply_override = str(msg.get("original_reply") or "").strip()

    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
        if not is_chat_session_json(path):
            raise ValueError("not a chat transcript")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return
    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if active_file is not None and active_file.resolve() == path:
        await _send(ws, {
            "type": "error",
            "message": "That session is the active conversation — start a New chat before editing it.",
        })
        return

    try:
        session = json.loads(path.read_text(encoding="utf-8"))
        exchanges = session.get("exchanges") or []
        if not (0 <= exchange_index < len(exchanges)):
            raise ValueError("exchange index out of range")
        exc = exchanges[exchange_index]
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Could not read exchange: {e}"})
        return

    # Reply to keep for a CoT-only regen: the displayed (trained-target) answer the client
    # sent, else the raw transcript reply.
    kept_reply = kept_reply_override or str(exc.get("assistant_response") or "").strip()
    user_prompt = str(exc.get("user_prompt") or "").strip()

    # A regenerated reply carries its own new CoT; a CoT-only regen grafts the new thought
    # onto the KEPT reply — so a written target is never a think/answer mismatch.
    if corrupt_response:
        if not new_reply:
            await _send(ws, {"type": "error", "message": "No regenerated reply to apply."})
            return
        cot, reply, reply_source = new_cot, new_reply, "regenerated"
    else:  # corrupt_cot only — keep the reply, take only the new CoT
        if not new_cot:
            await _send(ws, {"type": "error", "message": "No regenerated CoT to apply."})
            return
        if not kept_reply:
            await _send(ws, {
                "type": "error",
                "message": "No reply to keep — this exchange has no stored reply to graft a CoT onto.",
            })
            return
        cot, reply, reply_source = new_cot, kept_reply, "original"

    target = f"<think>{cot}</think>\n\n{reply}" if cot else reply
    run_id = f"manual-regen-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        sidecar = ChatSidecar(_CHATS_DIR)
        ok = sidecar.write_verdict(
            source_session=filename,
            exchange_index=exchange_index,
            verdict="revise",
            target=target,
            user_prompt=user_prompt,
            run_id=run_id,
            target_source="revised",
            target_kind="manual_regen",
            target_generation="chat_manual_regen_v1",
            # Human-validated: lock THIS exchange so re-reflection AND revisit preserve it
            # (the reviewed target survives; the original poison can't re-contaminate a build).
            # Per-exchange, so the rest of the chat still re-reflects normally.
            locked=True,
            live_session=(active_file.name if active_file is not None else None),
        )
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Sidecar write failed: {e}"})
        return
    if not ok:
        await _send(ws, {
            "type": "error",
            "message": "Sidecar write was skipped (live/invalid session or empty target).",
        })
        return

    # Clear any lingering corrupt flags on this exchange — the regeneration resolved them, so
    # the next Sleep must not blank/re-derive the target we just authored. Only rewrite the
    # transcript when a flag is actually set (avoids a needless full-file rewrite).
    if exc.get("corrupt_cot") or exc.get("corrupt_response"):
        try:
            mark_exchange_corrupt(path, exchange_index,
                                  corrupt_cot=False, corrupt_response=False)
        except (ValueError, OSError):
            pass

    await _send(ws, {
        "type": "exchange_regenerated_applied",
        "filename": filename,
        "exchange_index": exchange_index,
        "target_kind": "manual_regen",
        "applied_cot": bool(cot),
        "reply_source": reply_source,
        "new_cot": cot,
        "new_reply": reply,
    })


def _renumber_fact_hosts(source_session: str, deleted_index: int) -> int:
    """Move the ledger's fact→host-exchange pointers after an exchange is deleted.

    A ``[fact]`` the placement judge hosted on an exchange is injected into *that* CoT at
    build time as an "I know that …" line (``build_dataset.index_by_exchange`` →
    ``fact_render``), joined purely by ``(source_session, exchange_index)``. Deleting an
    exchange shifts every later position down one, so without this the fact would silently
    ride the wrong turn of the same conversation in every future build — the kind of drift
    that is invisible in review because both ends look well-formed.

    Two moves, both by re-``register`` (the sanctioned append-only mutation, keyed on the
    fact's stable content key so the stage/train_count survive):

      * a host AFTER the deleted index -> the same host, one lower;
      * a host that WAS the deleted exchange -> unhosted (``exchange_index: None``, and the
        anchor's own top-level index dropped so ``index_by_exchange``'s fallback cannot
        re-host it). The fact stays live and recallable; it just stops being injected into
        anyone's reasoning, which is the honest outcome when the reasoning it rested on is
        gone.

    Personas are deliberately left alone: their ``source_exchange`` no longer drives any
    injection (build-time persona injection was retired), while re-registering one would
    stamp a fresh timestamp into the very op-log the persona digest reads for recency- and
    session-weighted recurrence — a silent vote for that trait cast by a delete. Returns the
    number of fact anchors moved. Best-effort: never raises.
    """
    if _DATA_DIR is None:
        return 0
    try:
        from training.ledger import ConsolidationLedger
        # Resolved off the injected data root rather than reflections_path's self-location,
        # so this follows whatever tree the server was actually configured with.
        led = ConsolidationLedger(Path(_DATA_DIR) / "hot" / "consolidation")
        moved = 0
        for rec in led.fold().values():
            if rec.get("type") != "fact":
                continue
            se = rec.get("source_exchange") or {}
            host_sess = se.get("source_session")
            host_idx = se.get("exchange_index")
            if host_sess != source_session or not isinstance(host_idx, int):
                continue
            if host_idx < deleted_index:
                continue
            new_host = None if host_idx == deleted_index else host_idx - 1
            # The anchor's own exchange_index (where the fact was distilled from) is
            # index_by_exchange's fallback host, so it has to move too when it points into
            # this same transcript.
            own_idx = rec.get("exchange_index")
            if rec.get("source_session") == source_session and isinstance(own_idx, int):
                own_idx = None if own_idx == deleted_index else (
                    own_idx - 1 if own_idx > deleted_index else own_idx)
            elif not isinstance(own_idx, int):
                own_idx = None
            led.register_fact(
                content=rec.get("content", ""), item_type="fact",
                trigger=rec.get("trigger"),
                source_session=rec.get("source_session", ""),
                lang=rec.get("lang"), about=rec.get("about"),
                exchange_index=own_idx,
                source_exchange={"source_session": source_session,
                                 "exchange_index": new_host})
            moved += 1
        return moved
    except Exception:
        return 0


async def handle_rewrite_history(ws, msg: dict) -> None:
    """Bake every reviewed target into its transcript and DELETE every banned exchange.

    Training review "Rewrite history" — the finalize counterpart to the per-exchange
    verdicts. It settles both of them, because both leave the transcript itself untouched
    and only a sidecar flag standing between the corpus and what is actually on disk:

    *Repaired* (locked). Apply leaves the corrected target as a **sidecar override**
    (``locked``) layered over the still-corrupt transcript: training reads the good target,
    but everything that reads the transcript directly (RAG recall, snapshots, and every
    LATER exchange's context) still sees the corrupt reply. This action walks every locked
    exchange in hot/chats and, per exchange:

      1. rewrites the transcript's ``assistant_cot``/``assistant_response`` from the reviewed
         sidecar target (``chat_logger.rewrite_exchange_history``) — the correction becomes
         ground truth. The prior CoT/reply are kept under a forensic ``rewrite_history``
         record; the now-stale ``tension`` block is dropped (its per-token logits +
         ``token_ids`` were measured on the original generation, so branch replay would
         replay stale ids); corrupt flags are cleared;
      2. **unfreezes** the exchange (clears the per-exchange ``locked`` flag via
         ``ChatSidecar.set_exchange_locked``) — the lock only existed to stop re-reflection
         re-deriving the target over corrupt content, which is no longer a risk now the
         transcript is corrected.

    Training output is unchanged for these (``dialogue_source`` reads the target off the
    sidecar, which is untouched); what changes is the transcript-derived data.

    *Banned*. A ban already keeps the exchange out of the corpus, but leaves it in the
    transcript — which is right as a default (it happened, later turns were answered in its
    light) and wrong for the case bans exist to serve: a malformed generation that every
    later exchange keeps reasoning from and RAG keeps retrieving. So the finalize step
    **deletes** it (``chat_logger.delete_exchange``, preserving the turn under the session's
    ``deleted_exchanges`` record). Deletion shifts every later exchange down one position,
    so the two other index-keyed stores move with it in the same operation — the sidecar's
    verdict + anchor maps (``ChatSidecar.drop_exchange``) and the ledger's fact hosts
    (``_renumber_fact_hosts``) — and the chat's deferred background clean-base jobs, which
    cannot be renumbered meaningfully, are purged. Deletions run after that chat's rewrites
    and highest-index-first, so no queued index is invalidated under it. Banned **wander**
    captures are dropped from the corpus in the same pass (no transcript, no positions).

    Afterwards chat-RAG (and, if wanders went, the wander channel) is re-embedded.

    The session-level reflect-once freeze (``reflected_at``) is intentionally left as-is: the
    reviewed target is already the trained one, so no re-reflection is forced (a later Revisit
    would simply re-derive it, cleanly, over the corrected transcript). Refuses while a
    reflection run is active; skips the active conversation file. Scope is hot/chats (where the
    tab's Apply flow writes locks). Replies ``history_rewritten {rewritten, deleted,
    wander_deleted, chats, skipped, errors}``.
    """
    _mark_activity()   # operator is at the box cleaning the dataset — not idle
    if _is_reflection_active is not None and _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is active; cannot rewrite history until it finishes.",
        })
        return

    chats_dir = _CHATS_DIR
    sidecar = ChatSidecar(chats_dir)
    logger = _session.logger
    active_file = (logger.current_file.resolve()
                   if (logger is not None and logger.current_file is not None) else None)

    rewritten = 0
    deleted = 0
    chats_touched: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []

    for chat_path in iter_chat_json_files(chats_dir):
        source_session = chat_path.name
        sc_path = sidecar_path_for(chat_path)
        if not sc_path.exists():
            continue
        try:
            doc = json.loads(sc_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        exch = doc.get("exchanges") if isinstance(doc, dict) else None
        if not isinstance(exch, dict):
            continue
        locked_indices = sorted(
            int(k) for k, rec in exch.items()
            if isinstance(rec, dict) and rec.get("locked")
            and str(k).lstrip("-").isdigit()
        )
        # Banned exchanges are deleted outright once the rewrites are done (below). They are
        # collected in the same pass because both verdicts live on the same sidecar record —
        # an exchange can be locked AND later banned, in which case the delete simply wins.
        banned_indices = sorted(
            int(k) for k, rec in exch.items()
            if isinstance(rec, dict) and rec.get("banned")
            and str(k).lstrip("-").isdigit()
        )
        if not locked_indices and not banned_indices:
            continue
        # Never rewrite the file the live logger is appending to (its next save would clobber
        # the edit) — same guard as mark_corrupt / apply.
        if active_file is not None and chat_path.resolve() == active_file:
            skipped.append({"session": source_session, "reason": "active session"})
            continue

        chat_rewritten = 0
        for idx in locked_indices:
            rec = exch.get(str(idx)) or {}
            target = str(rec.get("target") or "").strip()
            if not target:
                skipped.append({"session": source_session, "exchange_index": idx,
                                "reason": "empty target"})
                continue
            # The sidecar target is <think>{cot}</think>{reply} (or a bare reply). Parse it
            # back into the transcript's separate CoT/response fields.
            cot, response = ChatLogger._parse_cot(target)
            try:
                rewrite_exchange_history(
                    chat_path, idx,
                    new_cot=cot, new_response=response,
                    run_id=str(rec.get("run_id") or ""),
                    target_kind=str(rec.get("target_kind") or ""),
                )
            except (ValueError, OSError) as exc:
                errors.append({"session": source_session, "exchange_index": idx,
                               "error": str(exc)})
                continue
            # Unfreeze: the transcript now holds the reviewed target, so the lock's job is done.
            sidecar.set_exchange_locked(source_session, idx, False)
            rewritten += 1
            chat_rewritten += 1

        # Deletions come AFTER this chat's rewrites and run HIGHEST INDEX FIRST: each one
        # shifts every later exchange down a position, so any order but descending would
        # invalidate the indices still queued behind it.
        chat_deleted = 0
        for idx in reversed(banned_indices):
            try:
                delete_exchange(chat_path, idx)
            except (ValueError, OSError) as exc:
                errors.append({"session": source_session, "exchange_index": idx,
                               "error": str(exc)})
                continue
            # Everything else keyed by position moves with it, in the same breath.
            sidecar.drop_exchange(source_session, idx)
            _renumber_fact_hosts(source_session, idx)
            deleted += 1
            chat_deleted += 1
        if chat_deleted and _DATA_DIR is not None:
            # The background pass's deferred clean-base jobs for this chat name exchanges by
            # index, and there is no re-deriving them against a transcript that just lost a
            # turn — drop them so a later run doesn't consume stale positions.
            try:
                from core import reflection_staging
                reflection_staging.purge_background_artifacts(_DATA_DIR, source_session)
            except Exception:
                pass

        if chat_rewritten or chat_deleted:
            chats_touched.append({"session": source_session,
                                  "exchanges": chat_rewritten,
                                  "deleted": chat_deleted})

    # Banned wander captures have no transcript to correct and nothing referencing them by
    # position, so their finalize is simply dropping the corpus lines.
    wander_deleted: list[str] = []
    try:
        from core.wander_sft import delete_banned as _delete_banned_wanders
        wander_deleted = _delete_banned_wanders()
    except Exception:
        wander_deleted = []

    # Re-embed chat RAG so retrieval reflects the corrected replies and drops the deleted
    # exchanges (a full rebuild off the event loop; skipped when nothing changed).
    if rewritten or deleted:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _get_rag().refresh_chat_index)
        except Exception:
            pass
    if wander_deleted:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _get_rag().refresh_wander)
        except Exception:
            pass

    await _send(ws, {
        "type": "history_rewritten",
        "rewritten": rewritten,
        "deleted": deleted,
        "wander_deleted": len(wander_deleted),
        "chats": chats_touched,
        "skipped": skipped,
        "errors": errors,
    })


async def handle_reset_session_reflection(ws, msg: dict) -> None:
    """Delete a chat's ``.state.json`` sidecar so the chat re-reflects from scratch.

    The Chat tab's "Re-reflect chat" button. A sidecar is the whole product of
    reflecting one chat — the reflect-once freeze (``reflected_at`` /
    ``chat_reflected``), the per-exchange verdicts and trainable targets, the
    consolidation summary/gist and the retrieval anchors — so removing it is the
    single act that puts the chat back in the reflection backlog: the next run
    (foreground or the background per-chat pass) re-derives all of it under the
    *current* persona. This is the deliberate, whole-chat sibling of the automatic
    invalidation ``handle_mark_corrupt`` performs on one poisoned exchange, and of
    the un-freeze ``handle_load_session in_place`` performs when an Ava-initiated
    chat is answered after its freeze.

    It is destructive in one way worth naming: human-validated ``locked`` targets
    (Training review's regenerate / hand-edit) live in this file too and go with
    it, so a hand-repaired chat loses those repairs and re-derives them from the
    original transcript. The client confirms with that spelled out.

    The chat's background per-chat staging is purged in the same act
    (``reflection_staging.purge_background_artifacts``): a chat frozen only at the
    ``chat_reflected`` stage has deferred clean-base jobs plus a mirror of its
    frozen sidecar in the checkpoint, and the next normal run would otherwise copy
    that mirror back over live — silently re-freezing the chat we just unfroze.

    Path-guarded to hot/chats like the other session ops. Refused while a
    reflection run owns the artifacts (it is mid-write on sidecars); the ACTIVE
    conversation is allowed, since the live logger writes the transcript, never the
    sidecar."""
    filename = msg.get("filename", "")
    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
        if not is_chat_session_json(path):
            raise ValueError("not a chat transcript")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return

    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    if _is_reflection_active():
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is in progress — wait for it to finish before "
                       "resetting a chat's reflection.",
        })
        return

    removed: list[str] = []
    try:
        sidecar = sidecar_path_for(path)
        if sidecar.exists():
            sidecar.unlink()
            removed.append(sidecar.name)
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Could not reset reflection: {e}"})
        return

    if _DATA_DIR is not None:
        try:
            from core import reflection_staging
            reflection_staging.purge_background_artifacts(_DATA_DIR, path.name)
        except Exception:
            pass

    await _send(ws, {
        "type": "session_reflection_reset",
        "filename": filename,
        "removed": removed,
        "had_sidecar": bool(removed),
    })


async def handle_delete_session(ws, msg: dict) -> None:
    """Delete a chat transcript (and its consolidation sidecar) from hot/chats.

    Hygiene / disaster-recovery for experiments — e.g. an encounter that went
    sideways — so a bad transcript can't be reflected on or retrieved. Refuses to
    delete the session the active logger is currently writing to (clear context
    first), and refreshes the chat-RAG index so the removed chat drops out of
    retrieval immediately. Path-guarded to hot/chats like the other session ops."""
    filename = msg.get("filename", "")
    try:
        path = (_CHATS_DIR / filename).resolve()
        if path.parent != _CHATS_DIR.resolve():
            raise ValueError("path traversal")
        if not is_chat_session_json(path):
            raise ValueError("not a chat transcript")
    except Exception:
        await _send(ws, {"type": "error", "message": "Invalid session filename."})
        return

    if not path.exists():
        await _send(ws, {"type": "error", "message": f"Session not found: {filename}"})
        return

    # Never delete the file the live logger is appending to — that would corrupt
    # the in-progress conversation. The operator must Clear context first.
    logger = _session.logger
    active_file = logger.current_file if logger is not None else None
    if active_file is not None and active_file.resolve() == path:
        await _send(ws, {
            "type": "error",
            "message": "That session is the active conversation — start a New chat before deleting it.",
        })
        return

    removed: list[str] = []
    try:
        # Every file the stem owns goes with the transcript — a summary or fact record
        # left behind is a memory of a conversation that no longer exists, and the
        # stem would resurrect it if the timestamp were ever reused.
        companions = [sidecar_path_for(path), summary_path_for(path), facts_path_for(path)]
        path.unlink()
        removed.append(path.name)
        for companion in companions:
            if companion.exists():
                companion.unlink()
                removed.append(companion.name)
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Could not delete session: {e}"})
        return

    # Drop the deleted chat from retrieval right away. Rebuild off the event loop
    # (re-embeds the remaining chats) so a large corpus doesn't stall the socket.
    try:
        asyncio.get_running_loop().run_in_executor(None, _get_rag().refresh_chat_index)
    except Exception:
        pass

    await _send(ws, {"type": "session_deleted", "filename": filename, "removed": removed})
