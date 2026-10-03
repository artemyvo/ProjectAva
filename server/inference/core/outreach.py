"""Outreach subsystem — Ava decides, on her own initiative, to start a conversation.

The passive counterpart already ships: open ``[ask:user]``/``[ask:meta]`` items are
*surfaced* into the first turn of a session the user opened (``generation._surface_block``).
This module is the **active** promotion of that idea. On the idle-wake heartbeat the
server runs a small decision pass over Ava's top surfaceable open ask — "do I want to
raise this with the user *now*?" — which resolves four ways. If she says **yes**, she
composes an opener and a new **reversed** chat session is written straight to
``hot/chats/``: Ava speaks first (the opener is exchange 0, logged under a
stage-direction speaker), the session is flagged ``initiated_by: "ava"``. It simply
lands in the session list; selecting it adopts that same file in place
(``load_session in_place=True`` / ``ChatLogger.resume_session``), so the user's reply
appends to the reversed transcript — no accept/dismiss handshake, no pending queue, no
*Continue chat* fork. If instead she has **already learned the answer** since the
question was queued (new data since), the pass answers ``resolved``: the ask is formally
resolved via ``ReflectionWriter.write_resolution`` (an ``evict``, so it drops from the
live fold and is never surfaced or re-picked) and no session is written — the janitor
half of self-directed curiosity. If she recognizes it as a question she has **already
asked** — the same question, or a paraphrase of one she already put to the user, which
the pool's exact-``content_key`` dedup cannot merge — the pass answers ``asked`` and the
duplicate is evicted (``reason:"duplicate"``), leaving the originally-raised record to
govern the question. The decision prompt shows her the questions she has recently raised
(:func:`_standing_questions_block`) so that check reads against the record, and frames an
unanswered question as the user's choice rather than a debt — she was observed treating
her own hanging questions as a blocker for NEW, unrelated ones. Otherwise (**no**)
nothing happens. The idle period is
the only throttle (composing marks the ask surfaced, so selection rotates and user asks
retire past the ceiling).

Owns its occupancy flag (``_outreach_active``); server.py's idle-loop / wander /
encounter guards read it directly. Everything Ava-side (RAG + reflection-writer
accessors, the reflect-generate factory, the temporal anchor, and on-disk paths) is
injected once at startup via :func:`configure`. Session/model state is read from
``core.runtime_state``. Never imports server.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional

from core.runtime_state import (runtime as _runtime, session as _session,
                                reflect_window as _reflect_window,
                                reflect_output_reserve as _reserve)
from core.field_parse import label as _label
from core import activity_log
from core import fact_fetch
from core import reasoning_text
from core.chat_logger import UNKNOWN_ADDRESSEE, ChatLogger
from core import hidden_capture
from core.reflection_memory import (
    ReflectionMemory, is_self_directed_origin, load_origin_note)
from core import reachout_gate

# Speaker label for the opener's synthetic stimulus turn. Exchange 0 of an
# Ava-initiated session has no real user prompt — the "stimulus" is her own decision
# to reach out — so it is logged under this stage-direction label (mirroring the
# encounter subsystem's "(setting)"). A non-empty narrator label reads honestly in
# every reflection view and, together with the session-level ``initiated_by`` flag,
# is what keeps the reversed exchange from being misparsed as a reply to the user.
_INITIATIVE_SPEAKER = "(initiative)"

# Outreach monopolises the executor thread while its (short) decision pass runs; the
# occupancy flag excludes concurrent wander/encounter/reflection. server.py's guards
# read this directly.
_outreach_active = False

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_get_reflection_writer: Callable = None
_make_sync_reflect_generate: Callable = None
# NB no temporal anchor here: the reflect-generate factory composes it for every
# pass (generation._reflect_system_parts). This module appended its own until
# 2026-08-07; injecting it again would date-stamp the prompt twice.
_load_server_config: Callable = None
_MEMORY_DIR: Any = None
_CHATS_DIR: Any = None
_PROMPTS_DIR: Any = None
# The autonomous idle path drives the decision pass directly off the executor; the
# manual (Sleep-tab) trigger below needs the socket + executor + guards to stream
# Ava's reasoning back to the client. These stay None until wired by configure().
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None

# Surfaceable-ask selection mirrors the passive path's ceiling (user asks retire past
# it; meta never retires). Kept here so outreach and passive surfacing agree.
_SURFACE_CEILING = 3

# Generation budget for the decision pass — a deliberating <think> CoT, plus DECISION,
# plus OPENER (a whole chat message) or ANSWER. See the note at the generate call.
#
# 8192 since 2026-08-05, matching `reflection_runner._DEFAULT_MAX_NEW_TOKENS`, the budget
# every reflection pass on the box already gets. Under the thought ceiling (4264ab4,
# `generation._reflect_think_ceiling`) the split at 4096 was ~2868 CoT / 1228 answer —
# so a pass reasoning about whether to raise a question was capped at well under half the
# thinking length its siblings are allowed, and the ceiling then forced the channel shut
# mid-deliberation rather than letting it settle. 8192 gives ~5735 / 2457. The cost is
# only bounded, not spent: a decision that concludes in 900 tokens still costs 900, and
# the input here is one small template, so nothing is taken from another budget.
#
# 12288 since 2026-09-20, and a config knob (``outreach.max_new_tokens``) clamped against
# the reflect window by ``runtime_state.reflect_output_reserve`` — the same shape as
# check-in's decision reserve and synthesis's two. At 8192 the pass was observed hitting
# the cap INSIDE its think on a long philosophical meta ask (the question itself ran to
# ~60 words, the deliberation drafted and critiqued three openers) — "budget 8192 of
# 32768" with the window three quarters empty. Under the thought ceiling 12288 gives
# ~8600 / 3690, matching the budget its two sibling decision passes already had.
#
# Not every cut opener is a budget problem: on 2026-09-22 a pass stopped at 791 of these
# 12288 with the OPENER cut mid-word, and the journal's "hit the token cap" was the
# verbatim-loop guard firing on the opener's fourth restatement (drafts in the CoT, then
# the real one). Check `last_loop` / the pass's token count before raising this.
_DECISION_MAX_NEW_TOKENS_DEFAULT = 12288
# Input headroom the reserve must leave: the decision prompt is a template + the RAG block
# + a facts block (~5k tokens observed), so 4096 is the floor its siblings use too.
_MIN_DECISION_INPUT_BUDGET = 4096

# A candidate whose last attempt produced no decision is held out of `_pick_ask` for this
# long (process-local — see `_note_failed_attempt`). Without it the pick is a fixation
# trap: `surfaceable_questions` orders fewest-raised-first and a truncated / leaked
# attempt writes no `surface` op, so the same ask came back first on every wake, failed
# the same way (a long question reasons long), and — under `deliberation.mode: sole`,
# where the executive chooses `reach_out` and this is what it dispatches — spent every
# deliberation hour on the one ask that could not be decided. 24 h, not the 72 h re-ask
# gap: that one paces a question that WAS raised; this one only rotates past one that
# could not be, and a later adapter or budget may well decide it.
_FAILED_ASK_HOLD_HOURS = 24.0
_failed_asks: dict[str, float] = {}   # ask key → time.monotonic() of the failed attempt

# Re-ask gap: an ask this job raised more recently than this is not eligible again.
# The passive path defaults to no gap because the *user* paces it (it fires only when
# they open a session); outreach is on an hourly clock of Ava's own, so without a gap
# "meta never retires" degenerates into fixation — the live meta asks simply round-robin
# and each comes back around, in the same words, every few hours (observed 2026-07-29:
# five meta asks on a 5 h cycle, two openers verbatim-identical). Overridable via
# ``server_config.json`` → ``outreach.min_reask_hours``.
_MIN_REASK_HOURS_DEFAULT = 72.0

# The per-person wake (`run_outreach_sweep_blocking`), mirroring check-in's: at most this
# many people reach a decision GENERATION per wake (`outreach.max_users`; the rest are
# deferred to the next), and a person silent past `outreach.max_silence_days` stopped
# talking rather than went quiet — without the ceiling a dormant contact would be written
# to for as long as a question of theirs stayed open (0 ⇒ no ceiling).
_MAX_USERS_DEFAULT = 3
_MAX_SILENCE_DAYS_DEFAULT = 30.0
# Per-person outcomes that spent no generation — a cheap bail does not use a sweep slot.
_CHEAP_SKIPS = frozenset({"no_model", "no_candidates", "all_recently_failed",
                          "all_awaiting_reply", "empty_ask", "dormant", "deferred"})

# How far down the candidate list to walk before giving up. The top candidate may be
# held by the unanswered-opener guard below, and the one under it too; without this the
# job would skip whenever its single favourite was blocked.
_CANDIDATE_DEPTH = 8

# How much of the source text's recap to hand the decision pass. Roomier than the live
# chat channel's `_NOMINATION_DISPLAY_CHARS` (700) because the budgets are not comparable:
# there the recap is an additive slot competing with the persona portrait, the user
# portrait and the RAG block on the hot path of a reply; here it is one block in a small
# prompt whose entire input is a single question, on a pass that owns the GPU for its
# duration and produces one message.
_SOURCE_MATERIAL_CHARS = 1400

# And for her own reaction to a wandered article, paired with that recap. Tighter than the
# recap it accompanies, which inverts the usual "the more relevant piece gets more room" —
# deliberately. The recap is prose ABOUT a text and can only be paraphrased; this is her
# own voice, and the one thing that must not happen is an opener assembled out of it. See
# `_prior_reaction`. It also arrives second, where a long block is likelier to be treated
# as the material to respond to rather than as background.
_REACTION_CHARS = 800

# The standing-questions block (`_standing_questions_block`): how far back a raise counts
# as "recent", how many are listed, and how much of each question's text is quoted. Two
# weeks rather than the 72 h re-ask gap because the repetition this list exists to catch
# is *paraphrase* — a near-copy of a question raised ten days ago is still a repeat, and
# only the model can recognize it (the pool dedups by exact content_key, so paraphrases
# accumulate as distinct asks that each pass the key-scoped gates).
_RAISED_WINDOW_HOURS = 336.0
_RAISED_CAP = 6
_RAISED_QUOTE_CHARS = 300


def configure(*, get_rag, get_reflection_writer, make_sync_reflect_generate,
              memory_dir, chats_dir, prompts_dir,
              load_server_config=None,
              send=None, executor=None, host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the outreach subsystem depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual Sleep-tab
    trigger (:func:`handle_outreach_now`); the autonomous idle path needs none of them.
    """
    global _get_rag, _get_reflection_writer, _make_sync_reflect_generate
    global _load_server_config
    global _MEMORY_DIR, _CHATS_DIR, _PROMPTS_DIR
    global _send, _executor, _host_busy, _mark_activity
    _get_rag = get_rag
    _get_reflection_writer = get_reflection_writer
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_server_config = load_server_config
    _MEMORY_DIR = Path(memory_dir)
    _CHATS_DIR = Path(chats_dir)
    _PROMPTS_DIR = Path(prompts_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


# ── candidate selection ────────────────────────────────────────────────────────

def _min_reask_hours() -> float:
    """The configured re-ask gap (hours) — ``outreach.min_reask_hours``."""
    try:
        cfg = _load_server_config() if _load_server_config is not None else {}
        val = (cfg.get("outreach") or {}).get("min_reask_hours")
        return float(val) if val is not None else _MIN_REASK_HOURS_DEFAULT
    except Exception:
        return _MIN_REASK_HOURS_DEFAULT


def _cfg(name: str, default):
    """``outreach.<name>`` from the server config, else *default*."""
    try:
        cfg = _load_server_config() if _load_server_config is not None else {}
        val = (cfg.get("outreach") or {}).get(name)
        return default if val is None else val
    except Exception:
        return default


def _decision_output_reserve() -> int:
    """Generation headroom for the decision pass — ``outreach.max_new_tokens`` (default
    ``_DECISION_MAX_NEW_TOKENS_DEFAULT``) clamped to fit the reflect window, the PHYSICAL
    load the reflect-lane generate resolves ``max_new_tokens`` against."""
    requested = _DECISION_MAX_NEW_TOKENS_DEFAULT
    try:
        cfg = _load_server_config() if _load_server_config is not None else {}
        val = (cfg.get("outreach") or {}).get("max_new_tokens")
        if val is not None:
            requested = int(val)
    except Exception:
        requested = _DECISION_MAX_NEW_TOKENS_DEFAULT
    try:
        return _reserve(requested, min_input=_MIN_DECISION_INPUT_BUDGET)
    except Exception:
        return requested


def _note_failed_attempt(ask_key: str) -> None:
    """Hold *ask_key* out of selection for ``_FAILED_ASK_HOLD_HOURS`` after an attempt
    that produced no decision (truncated / reasoning leaked into the opener).

    Process-local on purpose: the durable stores this module writes (`surface` ops, the
    reach-out tombstones) each mean something — a question raised, a message ignored —
    and a generation that ran out of budget means neither, so recording it there would
    advance the rotation or the backoff for an event the user never saw. A restart
    forgets the hold; the cost is at most one repeated attempt."""
    if ask_key:
        _failed_asks[ask_key] = time.monotonic()


def _recently_failed(ask: dict) -> bool:
    key = ask.get("key") or ""
    ts = _failed_asks.get(key)
    if ts is None:
        return False
    if (time.monotonic() - ts) < _FAILED_ASK_HOLD_HOURS * 3600.0:
        return True
    _failed_asks.pop(key, None)
    return False


def _has_dangling_opener(ask: dict) -> bool:
    """True when a session this ask was already raised in is *still* unanswered.

    The sharpest signal there is: she cold-opened with this question and the user
    never said anything back. Re-raising it — necessarily in near-identical words,
    since the opener is composed from the same question text — is the pathology
    this guard exists to stop. It is deliberately not time-limited: an opener that
    is still hanging is still hanging, however long ago it was written.

    Only sessions she *initiated* count (``initiated_by == "ava"`` with a lone
    exchange, matching ``background_reflection._is_unanswered_outreach``), so an ask
    surfaced passively into a user's own live session never blocks the pool.

    A session the stale-reach-out sweep has since deleted still blocks: it is checked
    against ``reachout_gate``'s tombstones first, because "not time-limited" would
    otherwise quietly become "until the sweep runs", and re-raising at that point is
    precisely the near-verbatim repeat this guard was written to prevent.
    """
    sessions = ask.get("surfaced_in_sessions") or []
    if not sessions or _CHATS_DIR is None:
        return False
    try:
        expired = {f"{r.get('stem')}.json" for r in reachout_gate.expired_openers()}
    except Exception:
        expired = set()
    for name in sessions:
        if name in expired:
            return True
        path = _CHATS_DIR / name
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue          # unreadable ⇒ cannot claim it is unanswered
        if (data.get("initiated_by") or "").strip() != "ava":
            continue
        if len(data.get("exchanges") or []) <= 1:
            return True
    return False


# ── whose question ─────────────────────────────────────────────────────────────

# {chat filename: (mtime_ns, person key, display)}. A transcript's header rarely changes
# after the conversation, so a sweep weighing one pool per person parses each chat once.
_chat_person_cache: dict[str, tuple[int, str, str]] = {}


def _chat_owner(name: str) -> tuple[str, str]:
    """``(person key, spelling)`` of the conversation in chat file *name*; ``("", "")``
    for nobody (see :func:`_chat_person`)."""
    name = str(name or "").strip()
    if _CHATS_DIR is None or not name.endswith(".json") or "/" in name or ":" in name:
        return "", ""
    path = _CHATS_DIR / name
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        # Hot first, then the archive — the fallback the ASK ORIGIN injection reads through
        # (`generation._ask_origin_block`), so a question keeps its person once a Sleep run
        # has archived the chat it came from, instead of turning into nobody's.
        try:
            from training.reflections_path import archive_chats_dir
            path = Path(archive_chats_dir()) / name
            mtime = path.stat().st_mtime_ns
        except Exception:
            return "", ""
    hit = _chat_person_cache.get(name)
    if hit is not None and hit[0] == mtime:
        return hit[1], hit[2]
    key = display = ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("interlocutor") or "").strip().lower() != "ai":
            display = str(data.get("user") or "").strip()
            key = reachout_gate.user_key(display)
    except Exception:
        key = display = ""
    _chat_person_cache[name] = (mtime, key, display if key else "")
    return key, (display if key else "")


def _chat_person(name: str) -> str:
    """The person key of the conversation in chat file *name*, or ``""`` for nobody.

    The transcript's session-level ``user`` — who actually spoke (`chat_logger` names an
    initiated session from its first human reply) — keyed the way check-in and the
    reach-out gate key people (`reachout_gate.user_key`). Nobody for a conversation with a
    model (``interlocutor: "ai"``), a transcript that is gone or unreadable, or a ref that
    is not a chat file (``til:`` / ``wiki:`` / ``<kind>/<stem>`` / ``lookup``)."""
    return _chat_owner(name)[0]


def addressee_for(source_session: str) -> str:
    """Whom to write to about material from *source_session*: its person, in that
    transcript's own spelling — or ``""`` when it belongs to nobody (:func:`_chat_person`).

    Injected into `assoc_bridge`, whose aha (an open question's need) and pivot (the
    conversation it switched from) act on one conversation's material and so write to
    that conversation's person, as outreach does with a question."""
    return _chat_owner(source_session)[1]


def _ask_person(ask: dict) -> str:
    """Whose question *ask* is: the person of the conversation it was formed in, or ``""``.

    A question formed in a conversation is raised only with that person: the session it
    opens carries its origin chat as ASK ORIGIN context (`core.ask_origin`), so raised
    with anyone else it would put one person's conversation into another's. One formed
    from reading, in a conversation with a model, or in a chat no longer on disk belongs to
    nobody and may be raised with anyone (:func:`_chat_person`)."""
    return _chat_person(ask.get("source_session") or "")


def _pick_ask(person: Optional[str] = None) -> tuple[Optional[dict], str]:
    """Choose the open ask to weigh, or ``(None, reason)`` when none is eligible.

    Walks the surfaceable candidates (already gap-filtered and rotation-ordered by
    :meth:`ReflectionMemory.surfaceable_questions`) and takes the first with no
    dangling unanswered opener that did not just fail to be decided
    (:func:`_note_failed_attempt`). ``all_recently_failed`` names the case where the
    hold alone emptied the walk, so it is not read as "nothing to raise".

    *person* (a ``reachout_gate.user_key``) walks only the questions that person may be
    asked — their own and the ones that belong to nobody (:func:`_ask_person`). ``None``
    walks the whole pool: the unscoped pass, for a corpus that names nobody."""
    try:
        memory = ReflectionMemory(_MEMORY_DIR)
        candidates = memory.surfaceable_questions(
            ceiling=_SURFACE_CEILING,
            # Scoped, the depth applies after the person filter: a person whose questions
            # all rank below the pool's first few must still reach them.
            limit=_CANDIDATE_DEPTH if person is None else 1_000_000,
            min_gap_hours=_min_reask_hours())
    except Exception:
        candidates = []
    if person is not None:
        candidates = [a for a in candidates
                      if _ask_person(a) in ("", person)][:_CANDIDATE_DEPTH]
    if not candidates:
        return None, "no_candidates"
    held = 0
    for ask in candidates:
        if _recently_failed(ask):
            held += 1
            continue
        if not _has_dangling_opener(ask):
            return ask, ""
    return None, ("all_recently_failed" if held == len(candidates) else "all_awaiting_reply")


def _standing_questions_block(user: str, exclude_key: str = "",
                              person: Optional[str] = None) -> str:
    """The questions she has already put to *user* recently, as a labelled block — or "".

    Two failure modes observed live, and this block addresses both at once. **Shyness:**
    the decision pass runs RAG-on, so her open asks come back as memory records, and with
    no framing she read them as a queue she was jamming — unanswered questions became a
    reason to decline every NEW question, however different its topic. So the block says
    out loud that {user} answers what they choose to and a hanging question is no debt.
    **Repetition:** the pool dedups by exact ``content_key`` only, so paraphrases of one
    question accumulate as distinct asks that each clear the key-scoped gates
    (``min_reask_hours``, the dangling-opener guard) — and only a model looking at the
    actual texts can recognize "this is Tuesday's question in other words". The list is
    what it looks at; the prompt's ``asked`` outcome is what it answers with.

    Rendered from :meth:`ReflectionMemory.recently_raised` (passively-surfaced asks
    count too — they were put to the user just the same), each line annotated when its
    opener is still hanging (`_has_dangling_opener` — disk reads, but over at most
    `_RAISED_CAP` asks' few sessions each). The candidate under decision is excluded by
    key so it is never listed as its own precedent. Empty when she has raised nothing
    recently — the prompt's check still stands, answered from recall alone.

    *person* keeps only what was put to that person (a session of theirs among the ask's
    ``surfaced_in_sessions``): the check is "have I already asked them?", and a question
    put to someone else is neither a repeat nor theirs to see — unscoped, the decision
    could retire one person's question as a duplicate of another's.
    """
    try:
        memory = ReflectionMemory(_MEMORY_DIR)
        raised = memory.recently_raised(
            within_hours=_RAISED_WINDOW_HOURS,
            limit=_RAISED_CAP + 1 if person is None else 1_000_000)
    except Exception:
        return ""
    if person is not None:
        raised = [r for r in raised
                  if any(_chat_person(s) == person
                         for s in (r.get("surfaced_in_sessions") or []))]
    raised = [r for r in raised if (r.get("key") or "") != exclude_key][:_RAISED_CAP]
    lines = []
    for r in raised:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        if len(content) > _RAISED_QUOTE_CHARS:
            content = content[:_RAISED_QUOTE_CHARS].rstrip() + "…"
        mark = ""
        try:
            if _has_dangling_opener(r):
                mark = "  (still waiting on an answer)"
        except Exception:
            pass
        lines.append(f"  - {content}{mark}")
    if not lines:
        return ""
    return "\n".join(
        [f"Questions you have already put to {user} recently, on your own initiative:"]
        + lines
        + [f"{user} answers what they choose to; a question left hanging is theirs to "
           "leave, and no debt in either direction — it is not a reason to hold a "
           "different question back. This list is here for one check: is the question "
           "above genuinely new, or one of these asked again in other words?"])


# ── prompt + decision parsing ──────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "You have been carrying an open question of your own:\n\n"
    "  {question}\n"
    "{origin_note}\n"
    "{standing}\n"
    "No one has asked you to raise it. Right now, on your own initiative, you may "
    "start a fresh conversation with {user} to bring it up — or decide the moment "
    "isn't right and let it keep sitting with you. There is no obligation either way; "
    "only raise it if you genuinely want to, now. An earlier question of yours still "
    "unanswered is not a reason to hold this one back — if this is a different "
    "question, weigh it on its own.\n\n"
    "But first, two checks. Since you first wondered this, have you already learned "
    "the answer? If you now know it (it no longer feels open to you), there is no "
    "reason to ask — answer 'resolved' and state what you now know. And have you "
    "already asked {user} this — the same question, or one that amounts to it in "
    "different words? Asking it again, rephrased, is not a new question; if this is a "
    "re-asking, answer 'asked' and name the earlier question it repeats.\n\n"
    "Think it through, then answer in exactly this form:\n\n"
    "DECISION: yes            (or: no, or: resolved, or: asked)\n"
    "OPENER: <if yes, the message you would open with — your own words, addressed to "
    "{user}, in your own voice.>\n"
    "ANSWER: <if resolved, state plainly what you already know that answers it. If "
    "asked, name the earlier question this one repeats.>\n"
)


def _load_prompt() -> str:
    """Outreach decision prompt (``{question}`` / ``{user}`` slots), overridable on disk."""
    path = _PROMPTS_DIR / "outreach_prompt.txt"
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return _DEFAULT_PROMPT


# The DECISION/OPENER block lives in the *answer*, but the raw reflect output keeps the
# CoT (`_clean_reflect_response` strips only structural tokens), and the CoT discusses its
# own choice ("Decision: yes.", "Decide to bring it up (YES)") — so the trace must come off
# before any label is matched. This module carried a partial copy of that logic (a tagged
# block and an unclosed `<think>` tail, but not a bare closing `</think>` or stray channel
# tokens, and no leak backstop) while being one of the two modules that WRITE the parsed
# text into a chat the user opens. The complete version now lives in `core.reasoning_text`.
_answer_after_think = reasoning_text.answer_after_think
_has_reasoning_leak = reasoning_text.has_reasoning_leak


def _parse_decision(text: str) -> tuple[str, str, str]:
    """Lenient parse of the DECISION/OPENER/ANSWER block → (decision, opener, answer).

    *decision* is normalized to one of four outcomes:
      * ``"yes"``      — raise the question now (``opener`` carries the message).
      * ``"resolved"`` — she has since learned the answer and no longer needs to ask
                         (``answer`` carries what she now knows); the caller formally
                         resolves (evicts) the ask instead of surfacing it.
      * ``"asked"``    — she has already put this question (or one it amounts to) to
                         the user (``answer`` names the earlier question); the caller
                         evicts the duplicate so the rotation stops re-picking it.
                         Matched BEFORE ``resolved``, because its natural phrasings
                         ("already asked", "already raised") share the ``already``
                         prefix the resolved patterns claim.
      * ``"no"``       — not now. This is also the default for an absent or
                         unrecognized ``DECISION:`` line, so a truncated generation
                         (no structured block) reads as a plain decline, never a
                         spurious resolve.

    Parses the answer only — the ``<think>`` CoT is stripped first (see
    :func:`_answer_after_think`) so the model's *reasoning about* its decision can't be
    mistaken for the decision, and the answer's ``DECISION:`` line is un-glued from the
    reasoning-close marker for the line-anchored match."""
    text = _answer_after_think(text)
    decision = "no"
    m = re.search(_label("DECISION") + r"(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        d = m.group(1).strip().lower()
        if d.startswith(("yes", "y", "true", "1")):
            decision = "yes"
        elif (d.startswith(("asked", "duplicate", "dup", "repeat", "re-ask", "reask"))
              or "already asked" in d or "already raised" in d):
            decision = "asked"
        elif d.startswith(("resolv", "already", "know", "obsolete", "moot",
                           "no longer")) or "resolv" in d:
            decision = "resolved"
    # OPENER is bounded so a trailing ANSWER field (resolved path) can't be swallowed by
    # its greedy tail — it stops at an ANSWER label or the end of the answer, whichever
    # comes first; ANSWER then takes everything after its own label.
    opener = ""
    m = re.search(_label("OPENER", anchored=False) + r"(.*?)(?=" + _label("ANSWER") + r"|\Z)",
                  text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if m:
        opener = m.group(1).strip()
    answer = ""
    m = re.search(_label("ANSWER", anchored=False) + r"(.*)\Z", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        answer = m.group(1).strip()
    return decision, opener, answer


def _target_user() -> str:
    """Name Ava would address — the last known speaker, or a neutral fallback.

    The fallback is display text for the prompts; the session writer records it as
    nobody, and the first human reply names the session (`chat_logger`)."""
    return (_session.user or "").strip() or UNKNOWN_ADDRESSEE


def _source_material(source_session: Optional[str]) -> str:
    """What she was reading when this ask occurred to her, or ``""``.

    The ask-side twin of `rag_engine._render_til_nomination` — same artifact
    (`til_gist`), same excerpt budget, same gist-or-nothing rule — differing only in
    where it lands: a nomination goes into the past-chat block of a live turn, this into
    the decision pass that is about to compose an opener.

    **Two thirds of the self-directed pool cannot be served, and the cause is upstream.**
    Measured on the live store: of 30 self-directed asks, 10 carry a resolvable
    ``til:<date>``, while 17 are filed under ``wiki:<site>`` — the *site*, not the page,
    so seventeen questions point at "Lurkmore" as though that named a text — and 3 under
    the bare string ``lookup``. `til_gist.resolve_source` returns ``None`` for those, and
    this returning ``""`` is the honest consequence: she raises the question exactly as she
    did before. The fix is in the producers (`til_wander._write_wander_exchange` and
    `ingest_lookup`), not here.
    """
    ref = (source_session or "").strip()
    if not ref:
        return ""
    try:
        from core import til_gist
        from core.chat_sidecar import gist_excerpt
        from training.reflections_path import til_snippets_dir
        path = til_gist.resolve_source(til_snippets_dir(), ref)
        if path is None:
            return ""
        gist = gist_excerpt(til_gist.gist_text(path), _SOURCE_MATERIAL_CHARS)
    except Exception:
        return ""
    if not gist:
        return ""
    # Presented as recall, not as an attachment: the note directly above it tells her not
    # to raise this as something she read, and a block labelled "SOURCE" would fight that
    # instruction in the same breath as giving it.
    parts = ["What you were reading when this first stirred, as you remember it:\n\n"
             f"{gist}"]
    reaction = _prior_reaction(path)
    if reaction:
        # The second piece, and a different one: the recap says what the text was, this
        # says what she made of it — which on a wandered article is usually the whole
        # reason a question came out of it at all. Labelled **at the time** rather than
        # left to read as a current position: the reading may be weeks old, she has
        # changed since (the premise the whole revisit machinery rests on), and an ask
        # she is only now deciding to raise must not arrive pre-answered by her own
        # earlier opinion.
        parts.append("What you made of it at the time:\n\n" + reaction)
    return "\n\n".join(parts)


def _prior_reaction(snippet_path) -> str:
    """Her own reaction to a wandered article, excerpted — ``""`` for anything else.

    Wander only, by construction: news digests and lookups are read by a curation pass
    that emits `[fact]`/`[ask]` items and never writes a reaction, so this corpus has an
    entry for exactly one of the three kinds (`core.wander_sft`).

    Excerpted HARDER than the recap beside it (see `_REACTION_CHARS`), and the reason is
    specific to this piece rather than budgetary: it is 2,300 characters of her own prose,
    in her own register, placed immediately before she composes a message. The box's
    anti-copy guard (`inference_backend._NoCopyPrevReply`) covers the previous assistant
    *reply* and nothing else, so nothing here would stop an opener from reproducing this
    text's phrasing — and a reach-out that quotes her own six-week-old reaction back at
    the user is the failure mode this pairing invites.
    """
    try:
        import json as _json
        from core import wander_sft
        from core.chat_sidecar import gist_excerpt
        snippet = _json.loads(Path(snippet_path).read_text(encoding="utf-8"))
        url = (snippet.get("source_url") or "").strip()
        if not url:
            return ""
        return gist_excerpt(wander_sft.reaction_for(url), _REACTION_CHARS)
    except Exception:
        return ""


_DECISION_FACTS_BLOCK_DEFAULT = (
    "Things you already know, looked up because this question seemed to turn on them:\n\n"
    "{facts}\n\n"
    "These are records of fact, not anyone's opinion. They are here for one judgement: "
    "does the question still need asking? If they already answer it — or enough of it "
    "that it no longer feels open to you — that is what 'resolved' is for. A fact that "
    "merely touches the topic without answering anything is no reason to hold back, and "
    "not something to recite in the opener."
)


def _load_decision_block_template() -> str:
    """The wrapper around a fetched facts blob on the DECISION lane (``{facts}`` slot).

    A sibling of chat's ``facts_block_prompt.txt`` and the reading lane's
    ``facts_reading_block_prompt.txt`` rather than a reuse of either: chat's is written
    for a pass about to answer somebody ("work into the reply"), the reading lane's for
    one about to recap a text, and this pass is about to decide whether a question of her
    own still needs asking — which is the one place the block must say out loud that an
    answered question is what ``resolved`` exists for. Default-written on first miss,
    so it is tunable on disk without a restart.
    """
    path = _PROMPTS_DIR / "facts_decision_block_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_DECISION_FACTS_BLOCK_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _DECISION_FACTS_BLOCK_DEFAULT


def _decision_facts_config() -> dict:
    """Settings for the facts fetch on the decision pass. Read per pass, so no restart.

    Rides ``graph.enabled`` — this is the same channel as chat's, pointed at a question
    instead of a message — and adds ``graph.reachout_facts`` to cut the reach-out lane
    alone (outreach's decision AND synthesis's analysis, one switch: they are the two
    passes that raise questions, and the reason to condition them is the same) while
    leaving live chat conditioned. Shares chat's ``max_claims``/``fetch_max_new_tokens``/
    ``til_max_age_days`` rather than growing its own: the decision prompt is small and the
    pass owns the GPU, so nothing forces a tighter budget the way the recap's size forced
    ``gist_max_claims``.
    """
    try:
        cfg = ((_load_server_config() or {}).get("graph") or {}
               ) if _load_server_config is not None else None
        if cfg is None:
            return {"enabled": False}
    except Exception:
        # Same reasoning as `generation._facts_channel_config`: an unreadable config is a
        # box in an unknown state, and this costs a generation per decision.
        return {"enabled": False}
    return {
        "enabled": bool(cfg.get("enabled", True)) and bool(cfg.get("reachout_facts", True)),
        "max_claims": int(cfg.get("max_claims", 6) or 6),
        "til_max_age_days": cfg.get("til_max_age_days"),
        "max_new_tokens": int(cfg.get("fetch_max_new_tokens", 512) or 512),
    }


def _fetch_ask_facts(ask_content: str) -> dict:
    """Stage 1 for the decision pass: which recorded facts bear on this ask. GPU, blocking.

    The ask-lane twin of ``generation._fetch_facts_block_sync`` and
    ``til_wander._fetch_reading_facts`` — same ``fact_fetch`` pass over the same candidate
    list with the same prompt and module spec, so what a live turn fetches and what a
    decision fetches cannot become two pieces of code that merely resemble each other.

    Why this pass in particular: its ``resolved`` branch is the box's one formal "have I
    already learned the answer?" judgement, and until now it ran on cosine retrieval
    alone — the retrieval the fetch channel exists to compensate. A fact that landed
    AFTER the ask was queued (the whole substance of ``resolved``) embeds on a trigger
    the ask's wording may never touch, and most of the record is English against
    often-Russian asks.

    Never raises: a decision that would otherwise have been made must not be lost because
    a retrieval channel failed. Every path returns an empty block with a named reason and
    the pass proceeds exactly as it did before this existed.
    """
    cfg = _decision_facts_config()
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

        reflect = _make_sync_reflect_generate(_get_rag())

        def generate(content: str, system_prompt: str, *, max_new_tokens: int) -> str:
            with activity_log.pass_context("fact_fetch:ask"):
                return reflect(
                    content, system_prompt,
                    # Selection, not expression — greedy, as on the chat path.
                    temperature=0.0, top_p=1.0,
                    max_new_tokens_setting=str(max_new_tokens),
                    before_session="", disable_rag=True,
                    disable_thinking=spec.disable_thinking,
                    stop_on_repeat=spec.stop_on_repeat)

        return fact_fetch.fetch_blob_for_ask(
            doc=doc, question=ask_content,
            generate=generate, prompt=prompt,
            window=_reflect_window(),
            now=_dt.date.today().isoformat(),
            max_new_tokens=cfg["max_new_tokens"],
            til_max_age_days=cfg["til_max_age_days"],
            max_claims=cfg["max_claims"])
    except Exception as e:
        traceback.print_exc()
        return {"text": "", "skipped": "error", "error": str(e)}


def _write_outreach_session(*, ask_content: str, opener: str, human: str,
                            ask_kind: str, ask_key: str = "",
                            ask_source_session: str = "",
                            hidden: Optional[dict] = None) -> str:
    """Write the reversed outreach session to hot/chats and return its filename.

    Exchange 0 is Ava's opener, logged under ``(initiative)`` with a synthetic impulse
    as its ``user_prompt`` and the session flagged ``initiated_by=ava``. The triggering
    open ask is stamped on the session (``initiated_ask``) so reflection can join this
    thread back to it and decide whether the reply resolved it — including the ask's own
    ``source_session``, which the answer-side ASK ORIGIN injection (`core.ask_origin`)
    reads for the life of the session, after the live ask itself may be gone. The
    session is NOT made active — it simply lands on disk so it shows up in the session
    list; the user adopts it in place and replies into the same file.
    """
    impulse = f"You decided to raise this with {human}, on your own initiative: {ask_content}"
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    logger = ChatLogger(_CHATS_DIR)
    logger.start_session(
        _session.system_prompt, user=human,
        model_id=_runtime.model_id, adapter_id=_runtime.adapter_id,
        notes=f"Ava-initiated outreach on her open {ask_kind or 'ask'}: {ask_content}",
        initiated_by="ava",
        initiated_ask={"key": ask_key, "content": ask_content, "ask_kind": ask_kind,
                       "source_session": ask_source_session},
    )
    opener_id = uuid4().hex
    logger.log_exchange(
        impulse, opener, speaker=_INITIATIVE_SPEAKER,
        system_content=_session.system_prompt, exchange_id=opener_id,
    )
    # The pass that wrote the opener, captured (`hidden_capture.write_opener_capture`):
    # a session's first state, which live capture never sees. Telemetry, best-effort.
    hidden_capture.write_opener_capture(logger.current_file, opener_id, hidden, opener,
                                        drive="outreach", tokenizer=_runtime.tokenizer)
    return logger.current_file.name if logger.current_file else ""


# ── decision pass (idle heartbeat) ─────────────────────────────────────────────

def run_outreach_decision_blocking(
    on_chunk: Optional[Callable[[str], None]] = None,
    on_question: Optional[Callable[[dict], None]] = None,
    on_prompt: Optional[Callable[[str, dict], None]] = None,
    bypass_cooldown: bool = False,
    user: Optional[str] = None,
) -> dict:
    """One outreach decision (runs on the GPU executor thread).

    Skips when no surfaceable ask exists. Otherwise runs the decision pass over the top
    candidate, which resolves four ways: on a **yes** with an opener, writes the
    reversed session to hot/chats and marks the ask surfaced (rotation + retirement);
    on **resolved** (she has since learned the answer, so the queued question is stale),
    formally resolves the ask via ``ReflectionWriter.write_resolution`` (an ``evict``,
    so it is dropped from the live fold and never re-picked) and returns
    ``{"resolved": True, …}``; on **asked** (the candidate is a re-asking of a question
    she has already put to the user — a paraphrase the exact-key dedup could not merge),
    evicts the duplicate (``reason:"duplicate"``) and returns ``{"duplicate": True, …}``;
    on **no**, returns a ``declined`` skip. Returns a status
    dict — enriched on the decided paths with ``question``/``decision``/``opener``/
    ``answer`` so the manual debug trigger can report the outcome.

    ``on_chunk``/``on_question`` are optional debug hooks used only by the manual
    Sleep-tab trigger (:func:`handle_outreach_now`): ``on_question`` fires with the ask
    she's considering before generation, and ``on_chunk`` streams her raw reasoning
    deltas (``<think>`` included) so the operator can watch her deliberate. The
    autonomous idle caller passes neither, so its behaviour is unchanged.

    ``bypass_cooldown`` skips the shared reach-out rate limit — the manual trigger sets
    it so an operator's on-demand "Reach Out" always sends, even inside another
    reach-out's window. A manual send still *stamps* the gate afterward, so an autonomous
    job won't pile on right behind it.

    *user* makes it one person's decision — the per-person sweep
    (:func:`run_outreach_sweep_blocking`) and the manual trigger: only their own questions
    and the ones that belong to nobody are weighed (:func:`_pick_ask`), the opener is
    addressed to them, "questions you have already put to them" lists only theirs, and the
    reach-out gate asked is theirs. ``None`` is the unscoped pass — the whole pool, the
    last active user, the box-wide gate — for a corpus that names nobody."""
    global _outreach_active
    owner = not _outreach_active     # a sweep holds the flag across its decisions
    _outreach_active = True
    try:
        if _runtime.model is None:
            return {"skipped": "no_model"}

        person = reachout_gate.user_key(user) or None
        ask, reason = _pick_ask(person)
        if ask is None:
            return {"skipped": reason}
        ask_content = (ask.get("content") or "").strip()
        ask_key = ask.get("key") or ""
        ask_kind = ask.get("ask_kind", "")
        if not ask_content or not ask_key:
            return {"skipped": "empty_ask"}

        user = (str(user).strip() if person else "") or _target_user()
        if on_question is not None:
            try:
                on_question({"question": ask_content, "ask_kind": ask_kind, "user": user})
            except Exception:
                pass
        # Stage 1, exactly as a live chat turn runs it (FACTS_TREE.md §10): before the
        # decision deliberates, a short thinking-off pass picks which recorded facts this
        # QUESTION turns on, and the rendered block is injected as the same labelled part
        # chat uses — while the sources behind the picks nominate their conversations /
        # recaps into the pass's past-chat block below (`rag_nominate_sessions`). This is
        # what makes the `resolved` branch a judgement against the record rather than
        # against whatever cosine happened to surface: the ask's wording and the answer's
        # trigger often share no words, and no language.
        facts = _fetch_ask_facts(ask_content)
        if facts.get("skipped") in fact_fetch.FETCH_FAILURES:
            # Same rule as the chat call site: the ordinary empty outcomes stay quiet
            # (picking nothing is a correct answer), a broken channel is named — the
            # `[tag]` print convention is teed into the activity journal.
            print(f"[outreach] facts fetch skipped: {facts['skipped']}"
                  + (f" — {facts['error']}" if facts.get("error") else ""), flush=True)
        facts_block = (_load_decision_block_template().replace("{facts}", facts["text"])
                       if facts.get("text") else "")

        # Reframe a self-directed (wander/TIL/lookup) ask so she carries it in as a
        # natural thread of her thinking with {user}, not "I read this online"; a
        # chat-origin ask gets no note (empty {origin_note}).
        origin_note = (load_origin_note(_PROMPTS_DIR)
                       if is_self_directed_origin(ask.get("source_session")) else "")
        # …and give her back the text it came from. An ask distilled from a day's news or
        # a wandered article is a *question* with its material discarded — she raised one
        # about the Gaza "Yellow Line" carrying nothing of the digest that prompted it,
        # which is the same decontextualization a fetched fact suffers when its
        # conversation is unreachable (`rag_engine._query_nominated`), one level up.
        #
        # Folded into the `{origin_note}` slot rather than given a placeholder of its own,
        # the way check-in folds its standing openers into `{recent}`: an operator with a
        # customized `outreach_prompt.txt` on disk gets this without editing it. The slot
        # is also exactly the right one — it is non-empty precisely for the asks that have
        # a source text, and the recall belongs beside the instruction about how to carry
        # it, since the two are the same thought (here is what you read; do not present it
        # as reading).
        if origin_note:
            recalled = _source_material(ask.get("source_session"))
            if recalled:
                origin_note = f"{origin_note}\n\n{recalled}"
        # The questions she already has in the air with this person — what the prompt's
        # "have you already asked this?" check and its not-a-debt framing read against.
        # A `{standing}` slot in the template places it; a customized prompt file on
        # disk that predates the slot gets the block PREPENDED instead of silently
        # dropping it (prepending keeps the contract last, per the reflect-lane rule).
        standing = _standing_questions_block(user, exclude_key=ask_key, person=person)
        template = _load_prompt()
        content = (template
                   .replace("{question}", ask_content)
                   .replace("{user}", user)
                   .replace("{origin_note}", origin_note))
        if "{standing}" in template:
            content = content.replace("{standing}", standing)
        elif standing:
            content = f"{standing}\n\n{content}"
        system_prompt = _session.system_prompt
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)

        rag = _get_rag()
        generate = _make_sync_reflect_generate(rag)
        # Thinking is left ON so the decision is deliberated (and the debug trigger can
        # stream that reasoning), so the budget must cover a full <think> CoT *plus* the
        # trailing DECISION/OPENER block — where OPENER is a whole chat message, not a
        # label. 512 was far too small — the CoT alone exhausted it and generation was
        # truncated before the structured answer, which parsed as a (false) "declined".
        # 2048 was then the size of synthesis's *opener* pass, which composes an opener and
        # nothing else; this pass must fit deliberation + DECISION + that same opener, so
        # it was the tighter budget of the two while asking for strictly more. Raised to
        # 4096 on the same reasoning as the synthesis analysis pass (d901fda), and to 8192
        # once the thought ceiling made the CoT/answer split explicit — see the note on
        # _DECISION_MAX_NEW_TOKENS_DEFAULT.
        raw = generate(
            content, system_prompt,
            temperature=0.7, top_p=0.95,
            max_new_tokens_setting=str(_decision_output_reserve()),
            rag_query=ask_content,
            facts_block=facts_block,
            rag_nominate_sessions=(facts.get("sources") or None),
            on_chunk=on_chunk,
            # Debug view of the assembled prompt (manual trigger only — the autonomous
            # caller passes no hook). This is the reach-out pass that actually RETRIEVES:
            # it omits `disable_rag`, so unlike check-in and synthesis a real memory block
            # is composed here, keyed on `rag_query`. That makes it the one of the three
            # where "what did retrieval put in front of her?" has a non-empty answer.
            on_prompt_debug=(None if on_prompt is None
                             else (lambda info: on_prompt("decision", info))),
            capture_hidden=hidden_capture.drive_openers_enabled(),
        )
        hidden = getattr(generate, "last_hidden_capture", None)
        decision, opener, answer = _parse_decision(raw)

        # Refuse a generation that was cut off before it closed its reasoning — BEFORE the
        # resolve and write paths, not only on the decline path below.
        #
        # gemma-4's channel opener is prefilled, so a generation truncated inside it emits
        # neither `<|channel>` nor `<channel|>`; `model_family._normalize_gemma` keys on
        # those markers, finds neither, and returns the text untouched. What arrives is
        # UNTAGGED reasoning — nothing for `_answer_after_think` to strip — and the CoT
        # deliberates in exactly the contract's words ("Decision: yes.") and quotes its
        # format block, so `_parse_decision` can read a whole structured answer out of it.
        # The existing `last_truncated` check sits on the `decision != "yes"` branch, which
        # such a parse skips entirely: it reaches `_write_outreach_session` and a think
        # block becomes exchange 0 of a chat the user opens. A spurious "resolved" is worse
        # still — `write_resolution` evicts a genuinely open ask from the live fold.
        # Observed on the synthesis opener pass, which has guarded it this way since.
        if reasoning_text.truncated_before_answer(
                raw, getattr(generate, "last_truncated", None)):
            print(f"[outreach] discarding ask {ask_key[:8]}: generation was cut off before "
                  f"it closed its reasoning — no answer to parse", flush=True)
            _note_failed_attempt(ask_key)
            return {"skipped": "truncated", "decision": False, "opener": "",
                    "question": ask_content, "ask_kind": ask_kind}

        if decision == "asked":
            # She recognized this as a re-asking of a question she has already put to
            # the user — a paraphrase living in the pool as its own record (exact
            # content_key dedup cannot merge it, and the fewest-raised-first rotation
            # would keep re-picking it every wake). Evict the duplicate: the question
            # itself stays alive under the record she originally raised, whose own
            # resolution machinery (answer distill / dangling-opener guard / re-ask
            # gap) keeps governing it.
            if ask_key:
                try:
                    _get_reflection_writer().write_resolution(
                        key=ask_key, question=ask_content, answer=answer,
                        ask_kind=ask_kind, reason="duplicate")
                except Exception:
                    traceback.print_exc()
            print(f"[outreach] retired duplicate ask {ask_key[:8]} ({ask_kind}) — "
                  f"already raised with the user", flush=True)
            try:
                from core import worklog
                worklog.record(
                    "outreach",
                    f"I nearly asked {user} about '{ask_content}', then realised I had "
                    f"already put this to them in other words — so I dropped the "
                    f"duplicate rather than repeat myself.",
                    refs={"ask_key": ask_key},
                )
            except Exception:
                traceback.print_exc()
            return {"duplicate": True, "question": ask_content, "answer": answer,
                    "ask_kind": ask_kind, "decision": "asked"}

        if decision == "resolved":
            # She has already learned the answer since queuing this question (new data
            # landed in memory/RAG after it was asked), so raising it would be stale.
            # Formally resolve it — append an `evict` so the fold drops it from the live
            # set and it is neither surfaced nor re-picked on the next idle wake. No
            # distill: the knowledge that let her answer is already in memory, so
            # re-capturing it would only duplicate.
            if ask_key:
                try:
                    _get_reflection_writer().write_resolution(
                        key=ask_key, question=ask_content, answer=answer,
                        ask_kind=ask_kind)
                except Exception:
                    traceback.print_exc()
            print(f"[outreach] resolved self-answered ask {ask_key[:8]} ({ask_kind})",
                  flush=True)
            try:
                from core import worklog
                worklog.record(
                    "outreach",
                    f"I nearly asked {user} about '{ask_content}', then realised I'd "
                    f"already learned the answer — so I let the question go.",
                    refs={"ask_key": ask_key},
                )
            except Exception:
                traceback.print_exc()
            return {"resolved": True, "question": ask_content, "answer": answer,
                    "ask_kind": ask_kind, "decision": "resolved"}

        if decision != "yes" or not opener:
            # Distinguish a genuine "no" from a generation that ran out of tokens before
            # it could emit the structured block (which parses the same — no DECISION →
            # default "no"). Report the cutoff honestly so it isn't read as a real decline.
            if getattr(generate, "last_truncated", None):
                _note_failed_attempt(ask_key)
                return {"skipped": "truncated", "decision": False, "opener": opener,
                        "loop": bool(getattr(generate, "last_loop", None)),
                        "question": ask_content, "ask_kind": ask_kind}
            return {"skipped": "declined", "decision": False, "opener": opener,
                    "question": ask_content, "ask_kind": ask_kind}

        # A truncated YES is refused too, not only a truncated decline. The check above
        # only guards the branch a truncation usually lands on (no DECISION parsed →
        # default "no"); a generation that closed its reasoning, emitted `DECISION: yes`
        # and `OPENER:`, and THEN ran out mid-message parses as a perfectly good yes with a
        # half-written message — and `_parse_decision` takes the opener from its label to
        # the end of the text, so the cut tail IS the opener. `truncated_before_answer`
        # above cannot see it (the boundary is there); the message is still unsendable.
        # Nothing is lost: the ask stays open and un-surfaced, the reach-out gate is not
        # stamped, and the next idle window retries. See synthesis for the same guard.
        if getattr(generate, "last_truncated", None):
            # `last_loop` tells the two non-EOS endings apart: a loop-guard halt at a
            # tenth of the allowance is not a budget problem, and saying "token cap" for
            # it sent the operator to raise `outreach.max_new_tokens` (2026-09-22).
            looped = bool(getattr(generate, "last_loop", None))
            print(f"[outreach] discarding opener on ask {ask_key[:8]}: generation "
                  + ("was halted by the loop guard" if looped else "hit the token cap")
                  + " mid-message — refusing to send a partial message", flush=True)
            _note_failed_attempt(ask_key)
            return {"skipped": "truncated", "decision": True, "opener": "",
                    "loop": looped, "question": ask_content, "ask_kind": ask_kind}

        # Last backstop before anything reaches a chat: if a reasoning marker survived into
        # the opener, it is not a clean message whatever else parsed. Cheap, and it catches
        # the shapes the truncation check above cannot (a partial normalization, a stray
        # channel token, a `</think>` inside the composed text).
        if _has_reasoning_leak(opener):
            print(f"[outreach] discarding opener on ask {ask_key[:8]}: reasoning markers "
                  f"survived into the message", flush=True)
            _note_failed_attempt(ask_key)
            return {"skipped": "opener_leak", "decision": True, "opener": "",
                    "question": ask_content, "ask_kind": ask_kind}

        # Shared reach-out rate limit: if synthesis / check-in just cold-opened the user,
        # don't stack a second unprompted message on top. Checked HERE, at the point of
        # sending, rather than as an idle-job gate — the decision pass has already run, and
        # the ask is left un-surfaced so it stays a candidate for the next window. Nothing
        # is lost; we simply don't send now. (See core.reachout_gate on why gating the job
        # instead of the message starved whichever sibling the scheduler tried last.)
        # A manual "Reach Out" (bypass_cooldown) always sends — the operator asked for it.
        # Scoped to the addressee on a per-person decision: one person's silence says
        # nothing about another's (see `reachout_gate.may_reach_out`).
        gate_ok, gate_reason = ((True, "") if bypass_cooldown
                                else reachout_gate.may_reach_out(user if person else None))
        if not gate_ok:
            print(f"[outreach] holding opener on ask {ask_key[:8]} — {gate_reason}",
                  flush=True)
            return {"skipped": gate_reason, "decision": True, "opener": opener,
                    "question": ask_content, "ask_kind": ask_kind}

        filename = _write_outreach_session(
            ask_content=ask_content, opener=opener, human=user, ask_kind=ask_kind,
            ask_key=ask_key,
            ask_source_session=str(ask.get("source_session") or "").strip(),
            hidden=hidden)
        # A reach-out was written → start the shared cool-down so synthesis / check-in
        # don't pile a second unprompted message on top of it in the same window.
        reachout_gate.mark_reachout(user if person else None)
        # The ask was raised (proactively, unprompted) → mark it surfaced so the
        # resolve-and-distill loop can later evict it, its surface-count advances (user
        # asks retire past the ceiling; meta is exempt), and selection rotates onward.
        if ask_key:
            try:
                _get_reflection_writer().write_surface(key=ask_key, surfaced_in=filename)
            except Exception:
                traceback.print_exc()
        print(f"[outreach] wrote outreach session {filename} on ask {ask_key[:8]} "
              f"({ask_kind})", flush=True)
        # Episodic worklog: record this reach-out in Ava's own voice, leaving the loop
        # open (she is now awaiting a reply). A separate task will consume the worklog.
        try:
            from core import worklog
            worklog.record(
                "outreach",
                f"I reached out to {user} on my own initiative about: {ask_content}",
                refs={"session": filename, "ask_key": ask_key},
                opens=f"awaiting {user}'s reply to my outreach",
            )
        except Exception:
            traceback.print_exc()
        return {"composed": True, "session": filename, "ask_kind": ask_kind,
                "decision": True, "opener": opener, "question": ask_content,
                "user": user}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if owner:
            _outreach_active = False


def run_outreach_sweep_blocking(
    on_question: Optional[Callable[[dict], None]] = None,
    max_users: Optional[int] = None,
) -> dict:
    """One outreach wake: a separate decision for each person she talks to.

    What the idle job and the deliberation executive run, in check-in's per-person shape
    (`checkin.run_checkin_sweep_blocking`): it walks `checkin.known_users` most recently
    active first and runs :func:`run_outreach_decision_blocking` per person — their own
    questions plus the ones that belong to nobody, addressed to them, held by their own
    reach-out gate. Until 2026-09-27 one decision raised the top question of the whole pool
    with whoever had spoken last; with several users, a question formed in one person's
    conversation could open a session with another and carry the first conversation in as
    its ASK ORIGIN. A person silent past ``outreach.max_silence_days`` is skipped, and at
    most ``outreach.max_users`` decisions generate per wake (cheap bails spend no slot). A
    corpus that names nobody falls back to the single unscoped pass.

    The result mirrors check-in's sweep: ``results`` per person, ``sent`` the ones that
    wrote a session with the first lifted to the top level, ``resolved`` / ``duplicate``
    lifted as flags (each ran a full decision), and a top-level ``skipped`` for a wake that
    did none of those — so the scheduler's ``consumed`` predicate reads it as before."""
    global _outreach_active
    _outreach_active = True
    try:
        if _runtime.model is None:
            return {"sweep": True, "skipped": "no_model"}

        from core import checkin
        people = checkin.known_users()
        if not people:
            # Nothing attributed to scope by — one unscoped pass, as before.
            result = run_outreach_decision_blocking(on_question=on_question)
            result.setdefault("sweep", True)
            result["users"] = 0
            return result

        try:
            cap = max(1, int(max_users if max_users is not None
                             else _cfg("max_users", _MAX_USERS_DEFAULT)))
        except Exception:
            cap = _MAX_USERS_DEFAULT
        try:
            dormant_s = max(0.0, float(_cfg("max_silence_days",
                                            _MAX_SILENCE_DAYS_DEFAULT))) * 86400.0
        except Exception:
            dormant_s = _MAX_SILENCE_DAYS_DEFAULT * 86400.0

        results: list[dict] = []
        sent: list[dict] = []
        ran = 0
        now = time.time()
        for person in people:
            name = person["display"]
            if dormant_s and now - float(person.get("last_turn_ts") or 0.0) > dormant_s:
                results.append({"user": name, "skipped": "dormant"})
                continue
            if ran >= cap:
                results.append({"user": name, "skipped": "deferred"})
                continue
            try:
                r = run_outreach_decision_blocking(user=name, on_question=on_question)
            except Exception as e:
                traceback.print_exc()
                r = {"error": f"{type(e).__name__}: {e}"}
            r.setdefault("user", name)
            results.append(r)
            if str(r.get("skipped") or "") not in _CHEAP_SKIPS:
                ran += 1
            if r.get("composed"):
                sent.append({"user": r.get("user"), "session": r.get("session"),
                             "opener": r.get("opener"), "question": r.get("question"),
                             "ask_kind": r.get("ask_kind")})

        out: dict = {"sweep": True, "users": len(people), "ran": ran,
                     "composed": len(sent), "sent": sent, "results": results}
        for flag in ("resolved", "duplicate"):
            if any(r.get(flag) for r in results):
                out[flag] = True
        if sent:
            # Lift the first send so the generic `describe`/client paths read one outreach.
            out.update({k: v for k, v in sent[0].items() if v is not None})
        elif not (out.get("resolved") or out.get("duplicate")):
            out["skipped"] = _sweep_skip_reason(results)
        return out
    except Exception as e:
        traceback.print_exc()
        return {"sweep": True, "error": f"{type(e).__name__}: {e}"}
    finally:
        _outreach_active = False


def _sweep_skip_reason(results: list[dict]) -> str:
    """One reason standing for a sweep that wrote nothing — check-in's rule: a reason that
    cost a generation (declined / truncated / a gate hold) over the cheap disk-read ones,
    since the scheduler's ``consumed`` predicate turns exactly that into "sleep the
    interval" vs "retry next poll"."""
    if any(r.get("error") for r in results):
        return "error"
    expensive = [str(r.get("skipped") or "") for r in results
                 if r.get("skipped") and str(r.get("skipped")) not in _CHEAP_SKIPS]
    if expensive:
        return "declined" if "declined" in expensive else expensive[0]
    reasons = [str(r.get("skipped") or "") for r in results if r.get("skipped")]
    return reasons[0] if reasons else "nothing_to_do"


# ── manual (Sleep-tab) trigger ─────────────────────────────────────────────────

async def handle_outreach_now(ws, msg: dict) -> None:
    """Manually run one outreach decision pass and stream Ava's reasoning (debug).

    The Sleep-tab "Reach Out" button routes here. It shares the exact code path the
    autonomous idle heartbeat uses (:func:`run_outreach_decision_blocking`) — it just
    wires the decision pass's ``on_question``/``on_chunk`` hooks to the socket so the
    operator can watch Ava pick her top open ask and deliberate on whether to raise it.
    On a yes it writes the reversed outreach session exactly as the autonomous path
    does (landing in the chats list); on ``resolved`` it formally resolves the ask (an
    evict, no session written); on a no it reports the decision without writing.

    Protocol: streams ``outreach_question`` (the ask under consideration), then
    ``outreach_prompt`` (the full model-facing prompt of the decision pass, in live chat's
    Debug segment shape — this is the reach-out job that actually retrieves, so its RAG
    segment is the injected memory block) and ``outreach_chunk`` reasoning deltas, and finishes with ``outreach_done`` carrying the
    outcome (``composed``/``resolved``/``duplicate``/``decision``/``session``/``opener``/
    ``answer`` or a ``skipped`` reason).

    ``user`` names whose decision it is — the Sleep tab sends whoever the Chat tab says is
    at the keyboard, and it defaults to the active session's speaker. The idle job decides
    per person (:func:`run_outreach_sweep_blocking`); the button runs ONE of those
    decisions, as check-in's does. With nobody named at all it is the unscoped pass."""
    loop = asyncio.get_event_loop()
    if _outreach_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "outreach_done", "skipped": "busy",
                         "message": "Another GPU job (reflection / wander / encounter / "
                                    "outreach) is in progress — try again once it finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "outreach_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "outreach_chunk", "text": delta}), loop)

    def _on_question(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "outreach_question", **info}), loop)

    def _on_prompt(label: str, info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "outreach_prompt", "pass": label, **info}), loop)

    target = str(msg.get("user") or "").strip() or (_session.user or "").strip()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_outreach_decision_blocking(
                on_chunk=_on_chunk, on_question=_on_question, on_prompt=_on_prompt,
                bypass_cooldown=True, user=target or None),
        )
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    # Reset the idle clock so the autonomous wander/outreach doesn't fire on top of a
    # manual run (mirrors the autonomous path's _mark_activity()).
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "outreach_done"}
    payload.update(result)
    await _send(ws, payload)


# ── GPU-free self-test: python -m core.outreach ─────────────────────────────────── #

def _selftest() -> None:
    import tempfile
    from datetime import datetime
    from core import checkin

    global _CHATS_DIR, _MEMORY_DIR, _load_server_config, _outreach_active
    saved = (_CHATS_DIR, _MEMORY_DIR, _load_server_config, _runtime.model,
             checkin.known_users, globals()["run_outreach_decision_blocking"])
    try:
        with tempfile.TemporaryDirectory() as td:
            _CHATS_DIR, _MEMORY_DIR = Path(td) / "chats", Path(td) / "memory"
            _CHATS_DIR.mkdir()
            _MEMORY_DIR.mkdir()
            _load_server_config = None
            _chat_person_cache.clear()

            def chat(name: str, user: str, **extra) -> None:
                doc = {"user": user, "exchanges": [{"speaker": user, "user_prompt": "hi"}]}
                (_CHATS_DIR / name).write_text(json.dumps({**doc, **extra}),
                                               encoding="utf-8")

            chat("20260901_100000.json", "Dana")
            chat("20260902_100000.json", "Sam Roe")
            chat("20260903_100000.json", "PeerBot", interlocutor="ai")

            # -- whose question -------------------------------------------------- #
            assert _chat_person("20260901_100000.json") == "dana"
            assert _chat_person("20260902_100000.json") == "sam"      # first-token key
            assert _chat_person("20260903_100000.json") == ""         # a model: nobody
            assert _chat_person("wander/article-1") == ""             # reading: nobody
            assert _chat_person("til:2026-09-01") == ""
            assert _chat_person("20990101_000000.json") == ""         # gone: nobody
            assert addressee_for("20260902_100000.json") == "Sam Roe"  # the chat's spelling
            assert addressee_for("20260903_100000.json") == ""         # a model: nobody
            assert addressee_for("wander/article-1") == ""
            # An archived chat keeps its person (hot first, then the archive).
            from training import reflections_path
            real_archive = reflections_path.archive_chats_dir
            archive = Path(td) / "archive"
            archive.mkdir()
            (archive / "20260801_090000.json").write_text(
                json.dumps({"user": "Boris", "exchanges": []}), encoding="utf-8")
            reflections_path.archive_chats_dir = lambda: archive
            try:
                assert _chat_person("20260801_090000.json") == "boris"
                assert addressee_for("20260801_090000.json") == "Boris"
            finally:
                reflections_path.archive_chats_dir = real_archive

            # -- the pool, per person ---------------------------------------------- #
            now = datetime.now().isoformat()

            def ask(key, kind, src):
                return {"op": "insert", "kind": "ask", "key": key, "ask_kind": kind,
                        "content": f"question {key}", "source_session": src, "ts": now}

            recs = [ask("kSam", "meta", "20260902_100000.json"),     # meta sorts first
                    ask("kDana", "user", "20260901_100000.json"),
                    ask("kRead", "user", "wander/article-1"),
                    ask("kPeer", "user", "20260903_100000.json")]
            mem = _MEMORY_DIR / "rag_memory.jsonl"
            mem.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

            assert _pick_ask(None)[0]["key"] == "kSam"                 # unscoped: whole pool
            assert _pick_ask("sam")[0]["key"] == "kSam"
            dana = _pick_ask("dana")[0]["key"]
            assert dana in ("kDana", "kRead", "kPeer"), dana           # never Sam's question
            assert _pick_ask("zed")[0]["key"] in ("kRead", "kPeer")    # only nobody's

            # -- "questions you have already put to them" is theirs only ------------ #
            chat("20260904_100000.json", "Dana")
            chat("20260905_100000.json", "Sam")
            with open(mem, "a", encoding="utf-8") as fh:
                for key, where in (("kDana", "20260904_100000.json"),
                                   ("kSam", "20260905_100000.json")):
                    fh.write(json.dumps({"op": "surface", "key": key, "ts": now,
                                         "surfaced_in": where}) + "\n")
            block = _standing_questions_block("Dana", person="dana")
            assert "question kDana" in block and "question kSam" not in block, block
            both = _standing_questions_block("you")
            assert "question kDana" in both and "question kSam" in both   # unscoped

            # -- a decision inside a sweep leaves the sweep's flag alone ------------ #
            _runtime.model = None
            _outreach_active = True
            assert run_outreach_decision_blocking(user="Dana") == {"skipped": "no_model"}
            assert _outreach_active is True
            _outreach_active = False

            # -- the sweep ------------------------------------------------------------ #
            _runtime.model = object()
            calls: list = []

            def stub(user=None, on_question=None, **_kw):
                calls.append(user)
                if user == "Dana":
                    return {"composed": True, "session": "s.json", "opener": "hello",
                            "question": "q", "ask_kind": "user", "user": "Dana"}
                return {"skipped": "no_candidates"}

            globals()["run_outreach_decision_blocking"] = stub
            t = time.time()
            checkin.known_users = lambda: [
                {"key": "old", "display": "Old", "last_turn_ts": t - 100 * 86400},
                {"key": "dana", "display": "Dana", "last_turn_ts": t},
                {"key": "sam", "display": "Sam", "last_turn_ts": t},
                {"key": "zed", "display": "Zed", "last_turn_ts": t}]
            out = run_outreach_sweep_blocking(max_users=1)
            assert calls == ["Dana"], calls          # Old dormant; Dana spent the one slot
            assert out["composed"] == 1 and out["user"] == "Dana" and out["opener"] == "hello"
            assert [r.get("skipped") for r in out["results"]] == [
                "dormant", None, "deferred", "deferred"], out["results"]
            assert _outreach_active is False

            calls.clear()
            out = run_outreach_sweep_blocking(max_users=5)
            assert calls == ["Dana", "Sam", "Zed"] and out["ran"] == 1   # cheap bails free

            checkin.known_users = lambda: []         # nothing attributed: one unscoped pass
            calls.clear()
            out = run_outreach_sweep_blocking()
            assert calls == [None] and out["users"] == 0 and out["sweep"]
    finally:
        (_CHATS_DIR, _MEMORY_DIR, _load_server_config, _runtime.model,
         checkin.known_users, globals()["run_outreach_decision_blocking"]) = saved
        _chat_person_cache.clear()
    print("outreach selftest OK")


if __name__ == "__main__":
    import sys
    # The sibling `training` package, as the other GPU-free self-tests do.
    _server = Path(__file__).resolve().parent.parent.parent
    if str(_server) not in sys.path:
        sys.path.insert(0, str(_server))
    _selftest()
