"""Synthesis subsystem — Ava re-reads an old chat and asks what she now wonders.

A hybrid of "Revisit old chat" (pick a chat older than a threshold and re-read it)
and "Reach Out" (start a conversation on her own initiative). It runs the *first*
(analysis / consolidation) stage of reflection over an aged chat, but framed as a
**synthesis against her current self**: not "what did I fail to notice at the time"
but "what do I only *now* think to wonder, having changed since?".

Flow (one idle slot / one manual click):

  1. Pick a chat older than ``min_age_days`` that has not itself been synthesized in
     the last ``min_resynth_days`` (the anti-fixation gate — see :func:`_pick_chat`).
  2. Run the analysis pass with the persona digest injected, producing an ``ABOUT``
     line (what the chat was about, so she can remind the person later) and zero or
     more ``[ask:user|meta|search]`` questions.
  3. Route every question into the live question pool via
     ``ReflectionWriter.write_consolidation`` — same store outreach and the passive
     surfacing read from (deduped by ``content_key``).
  4. Take the **first surfaceable (meta/user) question in emission order**, compose
     an opener that reminds the person which conversation it refers to, and write a
     reversed ``initiated_by:"ava"`` session straight to ``chats/`` (like outreach).
     The remaining questions simply stay in the pool. ``search`` asks are pool-only.

It is consolidation-only: no revision, no branching, no persona formation, no prompt
mutation, no training hand-off — it only produces questions.

Owns its occupancy flag (``_synthesis_active``); server.py's idle-loop / wander /
encounter / outreach guards read it directly, and its own guards exclude those. Every
capability it needs is injected once at startup via :func:`configure`; session/model
state is read from ``core.runtime_state``. Never imports server.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import (runtime as _runtime, session as _session,
                                reflect_output_reserve as _reserve,
                                reflect_window as _reflect_window)
from core.field_parse import label as _label
from core import activity_log
from core import fact_fetch
from core import reasoning_text
from core.chat_logger import ChatLogger
from core.chat_sidecar import is_chat_session_json
from core.reflection_chunking import build_consolidation_chunks
from core.reflection_writer import parse_consolidation, parse_impressions
from core import reachout_gate

# Speaker label for the opener's synthetic stimulus turn — identical to the outreach
# subsystem's, so the reversed session parses and trains the same way (exchange 0 has
# no real user turn; the stimulus is her own decision to reach out).
_INITIATIVE_SPEAKER = "(initiative)"

# Synthesis won't compose+send a cold-open within the shared reach-out cool-down of
# outreach / check-in having done so (`reachout_gate.COOLDOWN_SECONDS`). The analysis
# still fills the question pool; only the DM is deferred.

# Set while a synthesis run occupies the single GPU executor thread. Owned here;
# server.py's guards and the sibling subsystems' host_busy getters read it as
# ``synthesis._synthesis_active``.
_synthesis_active = False

# Defaults (overridable via server_config `synthesis`): a chat must be at least this
# old to be re-read, and must not have been synthesized within this window (so random
# picks don't fixate on the same transcript week after week).
_MIN_AGE_DAYS_DEFAULT = 7.0
_MIN_RESYNTH_DAYS_DEFAULT = 7.0
# Surfaceable-ask ceiling mirrors the passive/outreach path so all three agree on
# which open questions may still be raised.
_SURFACE_CEILING = 3

# Generation budget for the analysis pass — thought AND structured output. It was 2048,
# a quarter of what the consolidation pass gets for the same shape of work, and the pass
# reliably spent the whole of it inside a `<think>` it never closed: observed on a dense
# chat, the reasoning reached the point of drafting its own `*Ask:*` / `*Impression:*`
# items and was cut mid-word before the channel closed, so there was no answer region at
# all and the parser saw nothing. `reflection_runner` documents the identical failure for
# the anchor pass at a 1024 cap. Raised to 8192 (`reflection_runner._DEFAULT_MAX_NEW_TOKENS`)
# and then past it: the shared 8K default is sized for the consolidation pass, which reads
# a chat forward and distils it, while this pass re-reads one against the current self and
# reasons about the *difference* — a longer thought for the same transcript, and it went
# on exhausting 8192 inside an unclosed channel. Overridable per box via server_config
# `synthesis.max_new_tokens`, since the ceiling that stops a truncated CoT depends on the
# model's thinking length, not on anything this module can measure.
#
# Paired with `input_limit` below — they must move together, since raising the generation
# reserve takes the room out of the transcript and splits a long chat into more parts,
# which is the correct trade: a chunked chat is still fully read (every part is generated
# and written), whereas a CoT cut before its channel closes yields NOTHING at all —
# no ABOUT, no asks, no impressions — and the whole pass is wasted.
_ANALYSIS_MAX_NEW_TOKENS_DEFAULT = 12288

# Largest share of the window the reserve may claim. ``reflection_runner``'s equivalent
# never claims more than half; two thirds here because of the asymmetry above — the input
# side degrades gracefully (more, smaller chunks) while the output side fails totally —
# and because a half-window clamp pins this pass at exactly 12288 on the 24576-token box
# it runs on, which would leave the knob above unable to raise anything.
_ANALYSIS_RESERVE_MAX_FRACTION = 2.0 / 3.0

# ...but the fraction alone is the wrong guard on a SMALL window, which is what the
# half-window rule was really protecting: a third of 4096 does not hold the system prompt,
# let alone a transcript chunk, so chunking fails outright and the pass produces nothing —
# the same total failure the reserve was raised to avoid, arriving from the other side.
# The reserve therefore also yields whatever it must to leave this much input budget, and
# a window too small to grant both simply falls to the 512 floor.
_MIN_ANALYSIS_INPUT_BUDGET = 4096

# Generation budget for the second (opener-composition) pass — see the note at the call
# site. It is separate from the analysis reserve because the two passes fail differently
# (a truncated opener is caught and refused, a truncated analysis is silent) and because
# they compete for nothing: the analysis reserve is taken out of the TRANSCRIPT budget and
# so splits a long chat into more chunks, while the opener's input is one small template
# (system prompt + a few hundred tokens of filled form) against the whole window. Nothing
# else is running — synthesis owns the GPU and the context for the length of the pass — so
# the only thing a small number here buys is the failure it was observed producing.
#
# 12288, matching the analysis default: at 4096 the pass reliably ran out. The reflect
# lane's think ceiling (`generation._reflect_think_ceiling`) reserves 30% of the budget
# for the answer and forces the reasoning channel closed at the other 70%, so 4096 gave
# the message itself ~1.2k tokens — and this pass drafts several openings inside its CoT
# and then writes the chosen one out, so the answer region re-emits a whole message and
# was cut mid-word. That truncation is now refused (see the call site), which turns a bad
# opener into no opener; the budget is what turns it back into an opener.
_OPENER_MAX_NEW_TOKENS_DEFAULT = 12288

# The opener input is the chat system prompt + temporal anchor + the filled template
# (~1.5k tokens as of 2026-08), with RAG disabled. Reserve comfortably more than that so
# the clamp below can never squeeze the prompt itself.
_MIN_OPENER_INPUT_BUDGET = 4096


def _analysis_output_reserve(context_length: int) -> int:
    """Generation headroom for the analysis pass, clamped to fit *context_length*."""
    requested = int(_cfg("max_new_tokens", _ANALYSIS_MAX_NEW_TOKENS_DEFAULT))
    ceiling = max(512, min(int(context_length * _ANALYSIS_RESERVE_MAX_FRACTION),
                           context_length - _MIN_ANALYSIS_INPUT_BUDGET))
    return max(1, min(max(1, requested), ceiling))


def _opener_output_reserve() -> int:
    """Generation headroom for the opener pass, clamped to fit the reflect window.

    The window is the PHYSICAL load (``runtime_state.reflect_window``), not the chat
    budget, because that is what the reflect-lane generate resolves ``max_new_tokens``
    against. The ANALYSIS pass deliberately stays on the chat ``context_length``: it packs
    a transcript and hands the same limit to ``build_consolidation_chunks``, so moving it
    is a chunking change, not a budget one.
    """
    return _reserve(_cfg("opener_max_new_tokens", _OPENER_MAX_NEW_TOKENS_DEFAULT),
                    min_input=_MIN_OPENER_INPUT_BUDGET)

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_get_reflection_writer: Callable = None
_make_sync_reflect_generate: Callable = None
# NB no temporal anchor here: the reflect-generate factory composes it for every
# pass (generation._reflect_system_parts). This module appended its own until
# 2026-08-07; injecting it again would date-stamp the prompt twice.
_load_server_config: Optional[Callable] = None
_CHATS_DIR: Any = None
_DATA_DIR: Any = None
_PROMPTS_DIR: Any = None
# The manual (Sleep-tab) trigger needs the socket + executor + guards to stream Ava's
# reasoning back; the autonomous idle path needs none of them.
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None


def configure(*, get_rag, get_reflection_writer, make_sync_reflect_generate,
              chats_dir, data_dir, prompts_dir,
              load_server_config=None, send=None, executor=None,
              host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the synthesis subsystem depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual Sleep-tab
    trigger (:func:`handle_synthesis_now`); the autonomous idle path needs none of them.
    """
    global _get_rag, _get_reflection_writer, _make_sync_reflect_generate
    global _load_server_config, _CHATS_DIR, _DATA_DIR, _PROMPTS_DIR
    global _send, _executor, _host_busy, _mark_activity
    _get_rag = get_rag
    _get_reflection_writer = get_reflection_writer
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_server_config = load_server_config
    _CHATS_DIR = Path(chats_dir)
    _DATA_DIR = Path(data_dir)
    _PROMPTS_DIR = Path(prompts_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


# ── config ─────────────────────────────────────────────────────────────────────

def _cfg(key: str, default: float) -> float:
    """Read a numeric ``synthesis.<key>`` from server_config, falling back to default."""
    try:
        if _load_server_config is not None:
            cfg = _load_server_config() or {}
            v = (cfg.get("synthesis") or {}).get(key)
            if v is not None:
                return float(v)
    except Exception:
        pass
    return float(default)


# ── prompts + parsing ────────────────────────────────────────────────────────

_DEFAULT_SYNTH_PROMPT = (
    "You are re-reading one of your own past conversations from a while ago, as who "
    "you are now — not to re-summarize it, but to find what you only now think to "
    "wonder, having changed since.\n\nThis is who you are now:\n\n{persona}\n\n"
    "Think first, then write:\n\nABOUT: <one sentence: who it was with and what it "
    "was about>\n\n## RAG\nZero or more genuinely-new lines, one per line:\n"
    "- [ask:user] something only this person can answer (name them)\n"
    "- [ask:meta] a question about yourself\n"
    "- [ask:search] a public fact you could look up\n"
    "- [impression] something you now understand about the person that you did not "
    "see at the time (about: NAME)\n"
)

# Restated after the transcript, in the USER turn — see `_with_contract_tail`.
_DEFAULT_CONTRACT_TAIL = (
    "--- end of the past conversation ---\n\n"
    "That is all of it. You are not in it. It is a record of something already said and "
    "finished — not a turn addressed to you, and not something you are continuing. No one "
    "is waiting on a reply, and nothing you write here is sent to anyone.\n\n"
    "Now write your output, in exactly the form set out above and nothing else:\n\n"
    "ABOUT: <one sentence naming this conversation — who it was with and what it was "
    "about>\n\n## RAG\nZero or more tagged lines, one per line — [ask:user] / [ask:meta] / "
    "[ask:search] / [impression] — and only what genuinely newly arises. If nothing does, "
    "leave the section empty. Do not write a message to anyone.\n"
)

_DEFAULT_OPENER_PROMPT = (
    "A while ago you had a conversation with {user} — {about}. Re-reading it as who "
    "you are now, this question came up that you didn't think to ask then:\n\n"
    "  {question}\n\nStart a fresh conversation to bring it up. Remind them which "
    "conversation you mean, then ask.\n\nOPENER: <your opening message to {user}>\n"
)


def _load_prompt(name: str, default: str) -> str:
    path = _PROMPTS_DIR / name
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return default


# Reasoning-trace handling is shared with outreach / checkin / deliberation — see
# `core.reasoning_text`, which holds the complete version of what used to be four
# partial copies. Aliased to the former private names so call sites read unchanged.
_answer_after_think = reasoning_text.answer_after_think
_has_reasoning_leak = reasoning_text.has_reasoning_leak


def _parse_about(text: str) -> str:
    """Pull the ``ABOUT:`` one-liner (chat description) from an analysis output."""
    ans = _answer_after_think(text)
    m = re.search(_label("ABOUT") + r"(.+)$", ans, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_opener(text: str) -> str:
    """Pull the ``OPENER:`` message from the opener-composition output."""
    ans = _answer_after_think(text)
    m = re.search(_label("OPENER", anchored=False) + r"(.*)\Z", ans,
                  re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _clip_text(text: str, limit: int) -> str:
    """Clip *text* to *limit* chars on a LINE boundary, marking that it was cut.

    Mirrors ``reflection_service._clip_text``. A raw ``text[:limit]`` slice ends
    mid-word, and when the tail is a bulleted persona list that lands immediately
    above this pass's output contract, an unfinished list IS a completion prompt —
    observed: the analysis pass continuing the disposition list instead of writing
    ABOUT / ## RAG. Cut at the last newline (when one is reasonably far in) and say
    explicitly that material was dropped, so nothing dangles."""
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    nl = head.rfind("\n")
    if nl > limit // 2:
        head = head[:nl]
    return head.rstrip() + "\n...[clipped]"


# Per-piece budget for the injected persona block. Sized so a normal digest
# (~3.2k chars per piece as of 2026-08) is injected WHOLE — the clip is a backstop
# against an unbounded portrait, not the routine path.
_PERSONA_CLIP = 4000

# The persona block's own section headers read like output fields, and it sits directly
# above the ABOUT/## RAG contract. Fence it so its structure is visibly context rather
# than a form to continue.
_PERSONA_FENCE_OPEN = "=== BEGIN: who you are now ==="
_PERSONA_FENCE_CLOSE = "=== END: who you are now ==="


def _persona_context() -> str:
    """Render the current persona digest as a LENS for the analysis pass.

    Uses ``render_digest_for_analysis`` — the renderer built for a caller that produces
    STRUCTURED output rather than prose. Two exclusions, both of them things this pass was
    observed obeying as a writing brief rather than reading as context:

      * **VOICE.** A speaking register is an instruction for a reply this pass is not
        writing. Observed: it restated the voice paragraph as its own brief ("Structure:
        one coherent prose blob. No lists, no JSON." / "Key elements to include: <that
        vocabulary>"), planned an opening/developing/closing, and wrote a chat message.
        Dropping it from ``render_digest_for_introduction`` was not enough on its own.
      * **``self_portrait.text``.** It is written as a LETTER to a reader — the current
        one opens "Hello. I suspect that by the time you read this, you will have already
        detected…" — so injecting it puts a message addressed to someone in a system
        prompt whose pass must not address anyone. It was also the larger half of the
        block (~3k of ~5.5k chars), which is why trimming the voice paragraph off the
        other half moved nothing. What it carries that the facets do not is prose colour,
        not information the pass needs.

    What remains is stances + dispositions + lines: what she now thinks, does, and
    refuses — the lens that makes an old exchange read differently, which is the pass's
    entire premise. The consolidation pass is injected no persona block at all and does
    not derail; this keeps the smallest block that still serves "read it as who you are
    now".

    Best-effort: an unavailable / not-yet-generated digest degrades to a neutral note
    rather than aborting (the pass still surfaces questions, just without the self-lens)."""
    try:
        from core import reflection_digest
        from training.reflections_path import persona_dir
        digest = reflection_digest.latest_digest(persona_dir())
    except Exception:
        digest = None
    if not digest:
        return ("(No persona digest has been generated yet — read simply as yourself, "
                "aware that time has passed since this conversation.)")
    try:
        rendered = reflection_digest.render_digest_for_analysis(digest)
    except Exception:
        rendered = ""
    body = _clip_text(rendered, _PERSONA_CLIP)
    if not body:
        return "(persona digest empty)"
    return f"{_PERSONA_FENCE_OPEN}\n{body}\n{_PERSONA_FENCE_CLOSE}"


# How many of the live pool's open asks are shown to the analysis pass (newest first;
# the remainder is reported as a count). Wider than outreach's raised-list cap because
# the dedup target here is the WHOLE pool: a question synthesis re-derives collides with
# anything already carried, raised or not.
_OPEN_ASKS_CAP = 12


def _open_asks_block() -> str:
    """The questions already in the live pool, as a block folded into ``{persona}`` — or "".

    Synthesis re-reads an aged chat with no view of the ask pool it feeds (the pass is
    deliberately RAG-off, and the facts fetch offers knowledge facets only), so it kept
    re-deriving questions the pool already carried — and ``write_consolidation``'s dedup
    is exact ``content_key``, which a paraphrase sails past. Each duplicate then competes
    for the opener slot and for outreach's rotation, which is the "asks similar questions
    over and over" pathology one level upstream of outreach's own check.

    Folded into the existing ``{persona}`` slot rather than a new placeholder (the
    check-in ``{recent}`` precedent), so an operator's customized
    ``synthesis_prompt.txt`` on disk gets it unedited — and the slot fits: what she
    already wonders is part of "who you are now", the lens the prompt asks her to re-read
    through. The block closes on its own instruction, since the prompt's "New, genuine,
    or nothing" rule needs the list to have teeth against paraphrase.
    """
    try:
        from core.reflection_memory import ReflectionMemory
        memory = ReflectionMemory(_get_reflection_writer().memory_dir)
        asks = memory.open_questions()
    except Exception:
        return ""
    asks.sort(key=lambda r: (r.get("ts") or ""), reverse=True)
    lines = []
    for r in asks[:_OPEN_ASKS_CAP]:
        content = (r.get("content") or "").strip()
        if content:
            lines.append(f"  - {content}")
    if not lines:
        return ""
    dropped = max(0, len(asks) - _OPEN_ASKS_CAP)
    head = ("Questions you already carry — raised in past reflection and still open"
            + (f", the most recent {len(lines)} of {len(asks)}" if dropped else "")
            + ":")
    tail = ("A question already on this list, in any words, is not new — writing it "
            "again would only have you repeat yourself with the person. Write only "
            "what none of these already asks.")
    # Fenced for the same reason the persona block above it is: a bulleted list sitting
    # directly above the ABOUT/## RAG contract reads as a form to continue — and this
    # one is shaped exactly like the [ask] lines the contract requests.
    return "\n".join(["=== BEGIN: questions you already carry ===", head]
                     + lines + [tail, "=== END: questions you already carry ==="])


# ── chat selection + anti-fixation log ───────────────────────────────────────

def _synth_log_path() -> Path:
    return Path(_DATA_DIR) / "hot" / "synthesis" / "synthesized.jsonl"


def _last_synthesized() -> dict[str, datetime]:
    """Fold the synthesis log into {session_filename: latest synthesis datetime}."""
    out: dict[str, datetime] = {}
    path = _synth_log_path()
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


def _record_synthesized(session: str) -> None:
    """Append a synthesis record so this chat is not re-picked until the resynth window."""
    path = _synth_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"session": session, "ts": datetime.now().isoformat()},
                ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()


def _pick_chat(min_age_days: float, min_resynth_days: float) -> Optional[str]:
    """Choose a random chat old enough AND not recently synthesized, or None.

    Age comes from the ``YYYYMMDD_HHMMSS`` filename stem (falling back to the JSON
    ``timestamp``). Skips empty transcripts and unanswered Ava outreach (nothing to
    re-read). The anti-fixation gate then drops any chat synthesized within the last
    *min_resynth_days*, so the random pick rotates instead of fixating."""
    chats_dir = Path(_CHATS_DIR)
    if not chats_dir.exists():
        return None
    now = datetime.now()
    age_cutoff = float(min_age_days) * 86400.0
    resynth_cutoff = float(min_resynth_days) * 86400.0
    last_synth = _last_synthesized()
    candidates: list[str] = []
    for path in sorted(chats_dir.glob("*.json")):
        if not is_chat_session_json(path):
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
                age_seconds = (now - datetime.fromisoformat(raw_ts)).total_seconds()
            except Exception:
                continue
        if age_seconds is None or age_seconds < age_cutoff:
            continue
        exchanges = data.get("exchanges") or []
        if not exchanges:
            continue
        if (data.get("initiated_by") or "").strip() == "ava" and len(exchanges) <= 1:
            continue
        # Anti-fixation: skip a chat synthesized within the resynth window.
        prev = last_synth.get(path.name)
        if prev is not None and (now - prev).total_seconds() < resynth_cutoff:
            continue
        candidates.append(path.name)
    if not candidates:
        return None
    return random.choice(candidates)


# ── session write ────────────────────────────────────────────────────────────

def _write_synthesis_session(*, opener: str, human: str, source_session: str,
                             about: str, question: str, ask_kind: str = "",
                             ask_key: str = "") -> str:
    """Write the reversed, Ava-initiated session to chats/ and return its filename.

    Same shape as ``outreach._write_outreach_session`` — exchange 0 is Ava's opener
    under ``(initiative)`` with a synthetic impulse and ``initiated_by:"ava"``, so it
    lands in the session list, badged, and is adopted in place when opened. The
    surfaced ask is stamped (``initiated_ask``) so reflection can resolve it from the
    reply."""
    impulse = (f"Re-reading your older conversation with {human} ({about or 'a past talk'}), "
               f"you thought — for the first time — to ask: {question}")
    Path(_CHATS_DIR).mkdir(parents=True, exist_ok=True)
    logger = ChatLogger(_CHATS_DIR)
    logger.start_session(
        _session.system_prompt, user=human,
        model_id=_runtime.model_id, adapter_id=_runtime.adapter_id,
        notes=(f"Ava-initiated synthesis on {source_session}: a new question after "
               f"re-reading it as who she is now — {question}"),
        initiated_by="ava",
        # `source_session` here is the analyzed chat itself — the ask was just formed
        # from re-reading it, so the stamp is known structurally rather than looked up.
        # The answer-side ASK ORIGIN injection (`core.ask_origin`) reads it to put that
        # chat's gist behind the user's eventual reply.
        initiated_ask={"key": ask_key, "content": question, "ask_kind": ask_kind,
                       "source_session": source_session},
    )
    logger.log_exchange(
        impulse, opener, speaker=_INITIATIVE_SPEAKER,
        system_content=_session.system_prompt,
    )
    return logger.current_file.name if logger.current_file else ""


# ── the facts fetch (stage 1 for the analysis pass) ──────────────────────────

_REREAD_FACTS_BLOCK_DEFAULT = (
    "Things you have learned since — or knew from elsewhere — looked up because this "
    "conversation seemed to touch them:\n\n"
    "{facts}\n\n"
    "Nobody's opinion is in here. These are part of what has changed since the "
    "conversation happened, so weigh them as you decide what it leaves you wondering "
    "now: a question one of them already answers is not worth raising, and noticing "
    "that it is answered is itself part of the synthesis. Most of the conversation "
    "will touch none of them; let what you wonder come from the conversation itself."
)


def _load_reread_block_template() -> str:
    """The wrapper around a fetched facts blob on the RE-READ lane (``{facts}`` slot).

    A sibling of chat's ``facts_block_prompt.txt``, the reading lane's
    ``facts_reading_block_prompt.txt`` and outreach's ``facts_decision_block_prompt.txt``:
    each lane's wrapper states what the block is FOR, and this pass is forming questions
    against an old conversation — so the one thing this wrapper must say that no sibling
    does is that an already-answered question is a finding, not a failure. Default-written
    on first miss, so it is tunable on disk without a restart.
    """
    path = _PROMPTS_DIR / "facts_reread_block_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_REREAD_FACTS_BLOCK_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _REREAD_FACTS_BLOCK_DEFAULT


def _reread_facts_config() -> dict:
    """Settings for the facts fetch on the analysis pass. Read per run, so no restart.

    Rides ``graph.enabled`` and the same ``graph.reachout_facts`` switch as outreach's
    decision fetch — one switch for the reach-out lane, because the two passes are the
    two that raise questions and the reason to condition them is the same. Shares chat's
    claim/budget knobs for outreach's reason: the blob competes with a transcript, not a
    recap, so nothing forces a tighter cap.
    """
    try:
        cfg = ((_load_server_config() or {}).get("graph") or {}
               ) if _load_server_config is not None else None
        if cfg is None:
            return {"enabled": False}
    except Exception:
        return {"enabled": False}
    return {
        "enabled": bool(cfg.get("enabled", True)) and bool(cfg.get("reachout_facts", True)),
        "max_claims": int(cfg.get("max_claims", 6) or 6),
        "til_max_age_days": cfg.get("til_max_age_days"),
        "max_new_tokens": int(cfg.get("fetch_max_new_tokens", 512) or 512),
    }


def _fetch_reread_facts(session: dict, human: str) -> dict:
    """Stage 1 for the analysis pass: which recorded facts this old chat touches. GPU.

    The re-read twin of ``generation._fetch_facts_block_sync`` — same ``fact_fetch``
    pass, candidate list, prompt and module spec, over the aged transcript instead of an
    arriving message. Run ONCE per chat, not per chunk: the blob is about the same
    material whichever chunk is being analyzed, and one selection pass is one selection
    pass's cost.

    Never raises: an analysis that would otherwise have run must not be lost because a
    retrieval channel failed. Every path returns an empty block with a named reason and
    the pass proceeds exactly as it did before this existed.
    """
    cfg = _reread_facts_config()
    if not cfg["enabled"]:
        return {"text": "", "skipped": "disabled"}
    if _runtime.model is None:
        return {"text": "", "skipped": "no_model"}
    try:
        import datetime as _dt
        import sys as _sys
        _server_dir = Path(__file__).resolve().parent.parent.parent
        if str(_server_dir) not in _sys.path:
            _sys.path.insert(0, str(_server_dir))
        from core.modules import MODULES
        from core.reflection_source import session_date_line
        from graph import store as graph_store

        doc = graph_store.read_tree()
        if doc is None:
            return {"text": "", "skipped": "no_tree"}
        spec = MODULES.get("fact_fetch")
        if spec is None:
            return {"text": "", "skipped": "no_module"}
        try:
            prompt = (_PROMPTS_DIR / spec.prompt_file).read_text(encoding="utf-8").strip()
        except Exception:
            prompt = ""
        if not prompt:
            return {"text": "", "skipped": "no_prompt"}

        # Flatten the exchanges into the turn shape the body builder renders — the
        # narrator-aware convention handles an Ava-initiated opener's stage direction.
        turns: list[dict] = []
        for ex in (session.get("exchanges") or []):
            up = (ex.get("user_prompt") or "").strip()
            if up:
                turns.append({"role": "user", "speaker": ex.get("speaker", ""),
                              "content": up})
            ar = (ex.get("assistant_response") or "").strip()
            if ar:
                turns.append({"role": "assistant", "content": ar})
        if not turns:
            return {"text": "", "skipped": "no_text"}
        who = (human or "").strip()
        framing = " ".join(p for p in (
            (f"An old conversation of yours with {who}," if who
             else "An old conversation of yours,")
            + " which you are re-reading now.",
            session_date_line(session)) if p)

        reflect = _make_sync_reflect_generate(_get_rag())

        def generate(content: str, system_prompt: str, *, max_new_tokens: int) -> str:
            with activity_log.pass_context("fact_fetch:reread"):
                return reflect(
                    content, system_prompt,
                    # Selection, not expression — greedy, as on the chat path.
                    temperature=0.0, top_p=1.0,
                    max_new_tokens_setting=str(max_new_tokens),
                    before_session="", disable_rag=True,
                    disable_thinking=spec.disable_thinking,
                    stop_on_repeat=spec.stop_on_repeat)

        return fact_fetch.fetch_blob_for_reread(
            doc=doc, turns=turns, framing=framing,
            generate=generate, prompt=prompt,
            window=_reflect_window(),
            now=_dt.date.today().isoformat(),
            max_new_tokens=cfg["max_new_tokens"],
            til_max_age_days=cfg["til_max_age_days"],
            max_claims=cfg["max_claims"])
    except Exception as e:
        traceback.print_exc()
        return {"text": "", "skipped": "error", "error": str(e)}


# ── the run ──────────────────────────────────────────────────────────────────

def run_synthesis_blocking(
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
    bypass_cooldown: bool = False,
) -> dict:
    """One synthesis pass over an aged chat (runs on the GPU executor thread).

    Picks an eligible chat, runs the analysis pass (persona-injected), routes every
    produced question into the live pool, and — if any surfaceable question resulted —
    composes an opener for the first and writes a reversed session. Records the chat as
    synthesized either way (so the anti-fixation gate advances). Returns a status dict.

    ``on_chunk``/``on_stage`` are debug hooks used only by the manual Sleep-tab
    trigger: ``on_stage`` fires at each phase transition, ``on_chunk`` streams the raw
    generation deltas (``<think>`` included) so the operator can watch her reason. The
    autonomous caller passes neither, so its behaviour is unchanged.

    ``bypass_cooldown`` skips the shared reach-out rate limit — the manual "Chat reach
    out" sets it so an operator's on-demand run always composes+sends, even inside another
    reach-out's window. A manual send still *stamps* the gate afterward."""
    global _synthesis_active
    _synthesis_active = True

    def _stage(**kw) -> None:
        if on_stage is not None:
            try:
                on_stage(kw)
            except Exception:
                pass

    try:
        if _runtime.model is None:
            return {"skipped": "no_model"}

        min_age = _cfg("min_age_days", _MIN_AGE_DAYS_DEFAULT)
        min_resynth = _cfg("min_resynth_days", _MIN_RESYNTH_DAYS_DEFAULT)
        chosen = _pick_chat(min_age, min_resynth)
        if not chosen:
            return {"skipped": "no_candidate"}

        try:
            session = json.loads((Path(_CHATS_DIR) / chosen).read_text(encoding="utf-8"))
        except Exception as e:
            return {"skipped": "unreadable", "message": f"{type(e).__name__}: {e}",
                    "source_session": chosen}

        human = ((session.get("user") or "").strip()
                 or (_session.user or "").strip() or "your friend")
        _stage(stage="picked", session=chosen, user=human)

        persona = _persona_context()
        # The open-ask pool rides the same slot (see _open_asks_block): the analysis
        # must know what she already wonders, or it re-derives it in fresh words.
        open_asks = _open_asks_block()
        if open_asks:
            persona = f"{persona}\n\n{open_asks}"
        system_prompt = _load_prompt("synthesis_prompt.txt", _DEFAULT_SYNTH_PROMPT)
        system_prompt = system_prompt.replace("{persona}", persona)
        # The anchor goes BEFORE the prompt body, never after it. The synthesis prompt ENDS
        # in its output contract ("...write your output in exactly this form: ABOUT: ... ##
        # RAG ... New, genuine, or nothing"), and appending anything after a contract buries
        # it — the exact failure `generation._reflect_system_content` reorders the reflection
        # system message to avoid, arriving here because this pass builds its own system
        # message instead. Observed on a live run: the pass opened its thought by reciting
        # the anchor ("It is Monday, August 3, 2026, 21:30."), compressed the task to a
        # single line with the ABOUT/## RAG form dropped entirely, planned an
        # opening/developing/closing structure, and wrote a chat message to the person.
        # The opener pass below still APPENDS it — its system prompt is the chat prompt,
        # which carries no contract to bury.
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)

        rag = _get_rag()
        generate = _make_sync_reflect_generate(rag)
        writer = _get_reflection_writer()
        run_id = "synth_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # Stage 1, exactly as a live chat turn runs it (FACTS_TREE.md §10): which recorded
        # facts does this conversation touch? Injected as the same labelled part chat
        # uses, so the questions FORM against the record instead of being vetted against
        # it afterwards — this pass composes and sends its own opener directly, so
        # outreach's decision fetch cannot cover it. It is threaded through `_prepare` as
        # well as the generate call below, so the chunk packer measures the prompt WITH
        # the block and the transcript budget shrinks to fit rather than overflowing
        # `input_token_limit`. No nominations here, deliberately: the pass reads ONE
        # conversation and runs RAG-off, and recalling second conversations into it would
        # muddy what it is judging.
        facts = _fetch_reread_facts(session, human)
        if facts.get("skipped") in fact_fetch.FETCH_FAILURES:
            # Ordinary empty outcomes stay quiet (picking nothing is a correct answer); a
            # broken channel is named — the `[tag]` prints are teed into the journal.
            print(f"[synthesis] facts fetch skipped: {facts['skipped']}"
                  + (f" — {facts['error']}" if facts.get("error") else ""), flush=True)
        facts_block = (_load_reread_block_template().replace("{facts}", facts["text"])
                       if facts.get("text") else "")

        context_length = int(_runtime.context_length or 8192)
        tokenizer = _runtime.tokenizer
        output_reserve = _analysis_output_reserve(context_length)
        input_limit = max(1, context_length - output_reserve - 128)
        prepare_prompt = getattr(generate, "prepare_prompt", None)
        prepared_cache: dict[str, object] = {}

        # The contract has to come AFTER the transcript, not only at the end of the system
        # message. `format_chunk_content` renders the past exchange last and labels Ava's
        # own turns "Me:", so the final tokens before the generation prompt are her own
        # chat voice mid-conversation — the strongest positional signal in the whole
        # prompt, and it says "continue this". Reordering inside the system message (the
        # temporal-anchor fix) could not reach that, because the user turn comes after the
        # whole system message either way. So the form is restated in the user content,
        # below the transcript, where it is genuinely last.
        #
        # It carries the non-participation framing too. `_make_sync_reflect_generate`
        # normally supplies that through `rag_memory_reflect_prompt.txt` ("nothing here is
        # a task … No one is waiting on a reply") — but that text rides the RAG block, and
        # `_query_reflection` returns "" before applying its template when nothing was
        # retrieved, so a `disable_rag=True` pass like this one never receives it. Every
        # `reflection_runner` pass does (`rag_enabled = True`, dropped only as a budget
        # fallback), which is the one structural asymmetry between this pass and the
        # consolidation pass that does not derail.
        contract_tail = _load_prompt("synthesis_contract_prompt.txt", _DEFAULT_CONTRACT_TAIL)

        def _with_contract_tail(content: str) -> str:
            return content.rstrip() + "\n\n" + contract_tail

        # NB both the fit test and the generation must see the SAME text, or the tail is
        # spent out of the generation headroom instead of the transcript budget.
        def _prepare(content: str):
            if content not in prepared_cache:
                prepared_cache[content] = (prepare_prompt(
                    _with_contract_tail(content), system_prompt, disable_rag=True,
                    facts_block=facts_block,
                ) if prepare_prompt is not None else None)
            return prepared_cache[content]

        def _fits(content: str) -> bool:
            prepared = _prepare(content)
            if prepared is None:
                return len(_with_contract_tail(content)) <= int(context_length * 0.45 * 2.5)
            return int(prepared.input_tokens) <= input_limit

        chunks = build_consolidation_chunks(
            session, context_length, tokenizer=tokenizer, fits=_fits
        )
        parts = len(chunks)

        _stage(stage="analyzing", session=chosen, parts=parts)
        about = ""
        asks: list[dict] = []          # parsed [ask] items in emission order
        n_impressions = 0              # [impression] items routed to the person's portrait
        # Who the re-read conversation was with — the subject of any impression it yields,
        # and (as `source_user`) the speaker every distilled item is attributed to.
        speaker = (session.get("user") or "").strip()
        # Did ANY chunk come back on-contract (a parseable ABOUT / ## RAG section)? A pass
        # that declines and a pass that ignores the form both end at 0 asks, and reporting
        # both as "nothing new rose" hid the second — the operator reads a considered
        # decline where the model actually wrote a chat message. Tracked so the skip can
        # name which one happened.
        contract_seen = False
        any_truncated = False   # any chunk cut off before it finished generating
        for ch in chunks:
            content = ch["content"]
            raw = generate(
                _with_contract_tail(content), system_prompt,
                temperature=0.7, top_p=0.95,
                max_new_tokens_setting=str(output_reserve),
                # `facts_block` is redundant while `prepared_prompt` short-circuits the
                # prepare, and load-bearing on the fallback path where it does not
                # (`prepare_prompt` attr absent) — the fit test and the generation must
                # see the same text either way.
                disable_rag=True, facts_block=facts_block,
                input_token_limit=input_limit,
                prepared_prompt=_prepare(content), on_chunk=on_chunk,
            )
            # A pass that hit the token cap is cut mid-string, so its final item is a
            # half-written [fact]/[ask]. The consolidation pass in reflection_runner has
            # always flagged this; synthesis writes through the SAME writer and parser and
            # never did, so a capped analysis put a truncated fragment into live memory.
            truncated = bool(getattr(generate, "last_truncated", False))
            any_truncated = any_truncated or truncated
            if truncated:
                print(f"[synthesis] {chosen}: analysis hit the {output_reserve}-token "
                      f"cap — dropping the truncated final item", flush=True)
            about_here = _parse_about(raw)
            if not about:
                about = about_here
            # Route every produced question into the live pool (deduped by content_key).
            try:
                writer.write_consolidation(
                    run_id=run_id, source_session=chosen, text=raw,
                    truncated=truncated,
                    chunk_index=(ch["part"] - 1) if parts > 1 else None,
                    chunk_count=parts if parts > 1 else None,
                    # Synthesis re-reads one aged chat, so its speaker is the source
                    # of anything distilled here — same attribution as a normal
                    # consolidation pass over that session.
                    source_user=speaker,
                )
            except Exception:
                traceback.print_exc()
            # [impression] rides the same output but NOT the same writer: it is RAG-only
            # and un-ledgered, so `write_consolidation` (whose item regex does not know the
            # tag) would silently drop it. Parsing it out separately is what lets synthesis
            # be the second production site for readings of a person — and it is the
            # natural one: this pass re-reads an aged conversation as who Ava is now, so
            # what it notices about the person is exactly "what I only now see about them",
            # the same shift in vantage that makes the questions worth asking. Chunked
            # sessions dedup by content_key on the fold, so a reading restated across two
            # chunks stays one item.
            try:
                imps = parse_impressions(raw)
                if imps:
                    n_impressions += int(writer.write_impressions(
                        impressions=imps, source_session=chosen,
                        source_user=speaker, default_about=speaker,
                        run_id=run_id,
                    ).get("rag", 0))
            except Exception:
                traceback.print_exc()
            parsed = parse_consolidation(raw, truncated=truncated)
            if about_here or parsed.get("rag") or parsed.get("weights") or parsed.get("resolved"):
                contract_seen = True
            asks.extend(item for item in parsed.get("rag", []) if item.get("kind") == "ask")

        # Advance the anti-fixation gate only on a pass that actually produced the form —
        # INCLUDING a valid empty decline (contract seen, nothing new rose), which is a
        # real answer and should rotate the pick. A malformed pass is not an answer: it
        # says nothing about whether this chat has anything left to yield, so recording it
        # would lock the chat out for `min_resynth_days` on the strength of a model failure.
        # Left retryable instead; the pick is random across candidates, so a chat that
        # keeps derailing cannot starve the others.
        if contract_seen:
            _record_synthesized(chosen)

        n_asks = len(asks)
        # First SURFACEABLE (meta/user) question in emission order — search asks are
        # pool-only (held for the lookup agent), so they never trigger a reach-out.
        first = next((a for a in asks
                      if (a.get("ask_kind") or "search") in ("meta", "user")), None)
        if first is None:
            off_contract = not contract_seen
            # Two very different failures land here, and reporting both as "she ignored
            # the form" sent the last investigation down the wrong path: a pass that
            # reached the form and wrote something else, versus one cut off mid-thought
            # that never reached the form at all. The truncation flag separates them.
            if off_contract:
                print(f"[synthesis] {chosen}: analysis came back OFF-CONTRACT — no ABOUT "
                      f"and no ## RAG section parsed. "
                      + ("Generation was CUT OFF before the form (token cap) — raise the "
                         "analysis reserve." if any_truncated
                         else "She did not decline; she ignored the output form."),
                      flush=True)
            _stage(stage="no_question", session=chosen, asks=n_asks, about=about,
                   off_contract=off_contract, truncated=any_truncated)
            return {"analyzed": True, "raised": False, "source_session": chosen,
                    "about": about, "asks": n_asks, "impressions": n_impressions,
                    "off_contract": off_contract, "truncated": any_truncated,
                    "run_id": run_id}

        question = (first.get("content") or "").strip()
        ask_kind = first.get("ask_kind") or "user"
        ask_key = first.get("key") or ""
        # Shared reach-out cool-down: if outreach / check-in just cold-opened the user,
        # don't stack a second unprompted message. The analysis already routed every
        # question into the pool above, so the pick simply stays there for a later window
        # — nothing is lost, we just don't compose+send now.
        # A manual "Chat reach out" (bypass_cooldown) always sends — the operator asked.
        gate_ok, gate_reason = (True, "") if bypass_cooldown else reachout_gate.may_reach_out()
        if not gate_ok:
            _stage(stage="no_question", session=chosen, asks=n_asks, about=about)
            return {"analyzed": True, "raised": False, "skipped": gate_reason,
                    "source_session": chosen, "about": about, "asks": n_asks, "impressions": n_impressions,
                    "question": question, "run_id": run_id}
        _stage(stage="composing", session=chosen, question=question,
               ask_kind=ask_kind, asks=n_asks, about=about, user=human)

        opener_tmpl = _load_prompt("synthesis_opener_prompt.txt", _DEFAULT_OPENER_PROMPT)
        opener_content = (opener_tmpl
                          .replace("{user}", human)
                          .replace("{about}", about or "a conversation you had")
                          .replace("{question}", question))
        opener_system = _session.system_prompt
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)
        # Sized from the window rather than pinned (see `_opener_output_reserve`): this
        # pass reasons about how to raise a question from a chat weeks old without sounding
        # like it came out of nowhere, drafts several openings to do it, and then writes the
        # chosen one out in full — so both halves of the budget are real, and a flat 4096
        # left the answer half at ~1.2k tokens after the think ceiling took its share.
        # Nothing competes for the tokens: the input is one small template, not a
        # transcript, so no chunk budget is touched, and they are only spent when the CoT
        # actually runs that long.
        opener_reserve = _opener_output_reserve()
        raw_opener = generate(
            opener_content, opener_system,
            temperature=0.7, top_p=0.95,
            max_new_tokens_setting=str(opener_reserve),
            disable_rag=True, on_chunk=on_chunk,
        )
        truncated = bool(getattr(generate, "last_truncated", None))
        _skip_common = {"analyzed": True, "raised": False, "source_session": chosen,
                        "about": about, "asks": n_asks, "impressions": n_impressions, "question": question,
                        "run_id": run_id}
        # A generation that hit the token cap NEVER produces a sendable opener — refuse it
        # whatever else parsed. Two distinct shapes land here and only the first used to be
        # caught:
        #
        #   * No reasoning boundary. gemma-4's channel opener is prefilled, so a generation
        #     cut off inside it emits neither `<|channel>` nor `<channel|>`;
        #     `model_family._normalize_gemma` finds neither marker and returns the text
        #     untouched, so what arrives is UNTAGGED reasoning with nothing for
        #     `_answer_after_think` to strip. The CoT quotes its own format notes
        #     (`Format: \`OPENER: <message>\``) and its draft openings, so `_parse_opener`
        #     matches a SPURIOUS label and a whole think block reaches chat (observed).
        #
        #   * A closed boundary and a cut ANSWER. This was trusted on the reasoning that
        #     `OPENER:` is then parsed from a clean answer region — but the region being
        #     clean is not the same as it being FINISHED. `_parse_opener` takes everything
        #     from the label to the end of the text, so if generation stopped at the cap the
        #     opener IS the truncated tail: a message cut mid-word, or (observed, and the
        #     reason this branch exists) the model re-emitting its drafting from the CoT
        #     into the answer and running out partway through. There is no shape of
        #     truncation in which the last thing generated was a complete message.
        #
        # Nothing is lost by refusing: the question is already in the live pool (routed
        # above), the reach-out gate was not stamped, and the chat was not recorded as
        # synthesized unless the ANALYSIS was on-contract — so a later window retries.
        if truncated:
            closed = "</think>" in (raw_opener or "").lower()
            print(f"[synthesis] {chosen}: opener hit the {opener_reserve}-token cap "
                  + ("mid-message (reasoning closed, answer cut)" if closed
                     else "before it closed its reasoning")
                  + " — refusing to send a partial message", flush=True)
            return {**_skip_common, "skipped": "opener_truncated",
                    "opener_closed_think": closed}
        opener = _parse_opener(raw_opener)
        if not opener:
            # The OPENER: label was dropped — fall back to the whole answer (CoT removed).
            opener = _answer_after_think(raw_opener)
        # Defensive backstop: if any reasoning marker survived (bare </think>, stray gemma
        # channel), the opener is not a clean message — never send reasoning to chat.
        if _has_reasoning_leak(opener):
            return {**_skip_common, "skipped": "opener_leak"}
        if not opener:
            return {**_skip_common, "skipped": "opener_empty"}

        filename = _write_synthesis_session(
            opener=opener, human=human, source_session=chosen,
            about=about, question=question, ask_kind=ask_kind, ask_key=ask_key)
        # A reach-out was written → start the shared cool-down (outreach / check-in).
        reachout_gate.mark_reachout()
        # Mark the raised ask surfaced (rotation + retirement, resolve-and-distill later).
        if ask_key:
            try:
                writer.write_surface(key=ask_key, surfaced_in=filename)
            except Exception:
                traceback.print_exc()
        print(f"[synthesis] {chosen}: {n_asks} question(s), raised '{question[:60]}' "
              f"({ask_kind}) → {filename}", flush=True)
        # Episodic worklog: re-reading an old chat surfaced a new question I only think to
        # ask now, having changed since — record it in my own voice, loop left open.
        try:
            from core import worklog
            worklog.record(
                "synthesis",
                f"Re-reading an earlier chat with {human}, I found myself wondering: "
                f"{question}",
                refs={"session": filename, "source_session": chosen, "ask_key": ask_key},
                opens=f"awaiting {human}'s reply to what I re-wondered",
            )
        except Exception:
            traceback.print_exc()
        return {"composed": True, "raised": True, "session": filename,
                "source_session": chosen, "about": about, "asks": n_asks, "impressions": n_impressions,
                "question": question, "ask_kind": ask_kind, "opener": opener,
                "run_id": run_id}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _synthesis_active = False


# ── manual (Sleep-tab) trigger ───────────────────────────────────────────────

async def handle_synthesis_now(ws, msg: dict) -> None:
    """Manually run one synthesis pass and stream Ava's reasoning (Sleep "Chat reach out").

    Shares the exact code path the autonomous idle heartbeat uses
    (:func:`run_synthesis_blocking`) — it just wires the ``on_stage``/``on_chunk`` hooks
    to the socket so the operator can watch her pick a chat, synthesize, and (on a
    surfaceable question) compose an opener. A raised question writes the reversed
    session exactly as the autonomous path does.

    Protocol: streams ``synthesis_stage`` (phase markers) and ``synthesis_chunk``
    (reasoning deltas), finishing with ``synthesis_done`` carrying the outcome."""
    loop = asyncio.get_event_loop()
    if _synthesis_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "synthesis_done", "skipped": "busy",
                         "message": "Another GPU job (reflection / wander / encounter / "
                                    "outreach / synthesis) is in progress — try again "
                                    "once it finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "synthesis_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "synthesis_chunk", "text": delta}), loop)

    def _on_stage(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "synthesis_stage", **info}), loop)

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_synthesis_blocking(
                on_chunk=_on_chunk, on_stage=_on_stage, bypass_cooldown=True),
        )
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "synthesis_done"}
    payload.update(result)
    await _send(ws, payload)
