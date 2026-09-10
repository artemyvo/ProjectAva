"""Check-in subsystem — Ava reaches out after a stretch of silence.

The fourth autonomous idle job (alongside ``outreach``, ``synthesis`` and ``wander``),
and a sibling of the first two: like them it can cold-open the user with a reversed
``initiated_by:"ava"`` session. Its distinct DNA is the *trigger* and the *context*:

  * **Trigger — elapsed user silence.** It fires only once it has been at least
    ``checkin.silence_threshold_hours`` (default 5) since the user last actually spoke.
    That clock is derived from disk (:func:`_last_user_turn_dt` — the freshest real
    user exchange), NOT the shared ``_last_activity`` monotonic clock, because the
    latter is reset by Ava's own idle jobs (wander) and so does not mean "the *user*
    went quiet". The passage of time itself is the stimulus: the prompt tells her how
    long it has been.

  * **Context — the recent window.** Where ``outreach`` weighs one queued ``[ask]`` and
    ``synthesis`` re-reads one *aged* chat, check-in reviews the last few *recent*
    conversations (``checkin.recent_chats``, default 5) as a whole, and decides —
    expressively, not interrogatively — whether there is anything she genuinely wants
    to say on her own accord. Silence is a valid answer. Because five full transcripts
    don't fit one context window, each recent chat is first reduced to a short recap
    (``_summarize_recent``: reflection's stored gist where the chat has been reflected,
    else its own generated pass, cached per chat), and the decision pass then reasons
    *across* the compact set — so the whole window is represented, not just whichever
    chats fit newest-first. Appended to that window is what she has already
    said *into* this silence (:func:`_standing_openers`) — her own unanswered openers,
    quoted back — without which the pass cannot tell its first message from its fifteenth.

**Scope — one person at a time.** Both of those are questions about a *person*, not about
the box: "has the user gone quiet" and "what have we been talking about" have no answer
until you say who. Run unscoped they silently mixed everyone together — the silence clock
stopped the moment *anyone* spoke, the recent window mixed conversations with different
people, and the opener was addressed to whoever happened to be the last speaker on the
active session, about threads that may have been someone else's. So the decision pass is
per user: :func:`known_users` enumerates the people the corpus shows as having actually
talked to her, and :func:`run_checkin_sweep_blocking` (what the idle job runs) walks them,
running :func:`run_checkin_decision_blocking` once per person with their own clock, their
own window, their own standing openers and their own reach-out gate. Two people are two
conversations. A corpus that names nobody falls back to one unscoped pass — the historical
behaviour, and the honest one when there is no attribution to scope by.

On **yes** she composes an opener and a reversed session is written straight to
``hot/chats/`` (exchange 0 under ``(initiative)`` with a synthetic impulse, flagged
``initiated_by:"ava"``) — identical shape to outreach/synthesis, so it lands in the
list, is badged, and is adopted in place when opened (the user just replies into the
same file). On **no** nothing is written. There is no ``resolved`` outcome (that is
outreach's ask-specific janitor step); the choice is simply yes/no.

Coordination: check-in honors the shared :mod:`core.reachout_gate` gate so it and
outreach/synthesis cannot each DM the user in the same silent window — and so the window
itself widens the longer she goes unanswered.

Owns its occupancy flag (``_checkin_active``); server.py's idle-loop / wander /
encounter / outreach / synthesis guards read it directly. Everything Ava-side (RAG +
reflection-writer accessors, the reflect-generate factory, the temporal anchor, the
server-config loader, and on-disk paths) is injected once at startup via
:func:`configure`. Session/model state is read from ``core.runtime_state``. Never
imports server.
"""
from __future__ import annotations

import asyncio
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import (runtime as _runtime, session as _session,
                                reflect_output_reserve as _reserve)
from core.field_parse import label as _label
from core import reasoning_text
from core.chat_logger import ChatLogger
from core.chat_sidecar import ChatSidecar, is_chat_session_json
from core.chat_sidecar import gist_excerpt as sidecar_gist_excerpt
from core import reachout_gate

# Who-is-who key, borrowed from the gate rather than re-derived. Both modules answer
# questions about one person's thread off the same corpus, so they have to agree on which
# transcripts are that person's; the gate is a leaf that cannot import
# `reflection_writer.normalize_person` (which it mirrors), so its copy is the one both use.
_user_key = reachout_gate.user_key

# Speaker label for the opener's synthetic stimulus turn (mirrors outreach/synthesis).
_INITIATIVE_SPEAKER = "(initiative)"
# Stage-direction speakers whose exchange-0 ``user_prompt`` is a synthetic impulse, not
# a real user utterance — never rendered as the user talking in the recent window.
_NARRATOR_SPEAKERS = {"(initiative)", "(setting)"}

_SILENCE_THRESHOLD_HOURS_DEFAULT = 5.0
_RECENT_CHATS_DEFAULT = 5
# How much of the USER a session must contain to count as one of the recent
# conversations (`checkin.min_user_turns`). See `_recent_chats`: 1 admits every genuine
# exchange, 2 demands actual back-and-forth and so drops a thread that is her own opener
# plus a one-line reply — at the risk of an empty window on a quiet box.
_MIN_USER_TURNS_DEFAULT = 1
# How many people may get as far as the decision GENERATION in one sweep
# (`checkin.max_users`). Every candidate is still measured — the silence clock and the
# staleness ceiling below are pure disk reads — so the cap only bounds GPU work, and a
# person it defers is simply reconsidered on the next wake. Candidates are walked
# most-recently-active first, so the conversations she is actually in come first.
_MAX_USERS_DEFAULT = 3
# Past this much silence a person stops being a check-in candidate at all
# (`checkin.max_silence_days`; 0 ⇒ no ceiling). Needed because the sweep enumerates
# everyone the corpus has EVER recorded a turn from: without it, a contact who stopped
# talking a year ago is a candidate forever, and since the reach-out backoff settles at one
# message a day rather than stopping, she would go on writing into that silence daily.
# The manual trigger ignores it — an operator asking for a check-in has said who they mean.
_MAX_SILENCE_DAYS_DEFAULT = 30.0
# Generation budget for the decision pass — a deliberating <think> CoT, plus DECISION,
# plus OPENER (a whole chat message). See the note at the generate call.
#
# Window-derived since 2026-08-06 (`_decision_output_reserve`, knob
# `checkin.max_new_tokens`), on the reasoning that settled the synthesis opener: check-in
# owns the GPU and the whole reflect window for the length of its pass, so a flat constant
# claiming a fraction of it buys nothing but the cutoff. The history: 2048 → 4096 → 8192
# (2026-08-05, matching outreach) → 12288 here, which under the thought ceiling (4264ab4)
# splits ~8602 CoT / 3686 answer against 8192's ~5735 / 2457. This pass reasons across five
# conversation recaps AND her own standing unanswered openers before deciding, then writes
# a whole message. Bounded, not spent — a short decision still costs only what it uses.
_DECISION_MAX_NEW_TOKENS_DEFAULT = 12288
# Room held back for the decision prompt: system prompt + temporal anchor + the recap
# block + the standing-openers block (~2.5-3k tokens as of 2026-08).
_MIN_DECISION_INPUT_BUDGET = 4096
# How many of her own standing unanswered openers to quote back into the decision prompt
# (newest kept; anything older is reported as a count). See _standing_openers.
_STANDING_OPENERS_CAP = 5

# Check-in monopolises the executor thread while its (short) decision pass runs; the
# occupancy flag excludes concurrent wander/encounter/reflection/outreach/synthesis.
# server.py's guards read this directly.
_checkin_active = False

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_make_sync_reflect_generate: Callable = None
# NB no temporal anchor here: the reflect-generate factory composes it for every
# pass (generation._reflect_system_parts). This module appended its own until
# 2026-08-07; injecting it again would date-stamp the prompt twice.
_load_server_config: Callable = None
_CHATS_DIR: Any = None
_PROMPTS_DIR: Any = None
# The manual (Sleep-tab) trigger needs the socket + executor + guards to stream Ava's
# reasoning back; the autonomous idle path needs none of them.
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None


def configure(*, get_rag, make_sync_reflect_generate,
              load_server_config, chats_dir, prompts_dir,
              send=None, executor=None, host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the check-in subsystem depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual Sleep-tab
    trigger (:func:`handle_checkin_now`); the autonomous idle path needs none of them.
    """
    global _get_rag, _make_sync_reflect_generate, _load_server_config
    global _CHATS_DIR, _PROMPTS_DIR, _send, _executor, _host_busy, _mark_activity
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_server_config = load_server_config
    _CHATS_DIR = Path(chats_dir)
    _PROMPTS_DIR = Path(prompts_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


def _decision_output_reserve() -> int:
    """Generation headroom for the decision pass, clamped to fit the reflect window.

    The window is the PHYSICAL load (``runtime_state.reflect_window``) — what the
    reflect-lane generate resolves ``max_new_tokens`` against — so a reserve clamped
    against anything else is clamped against the wrong number."""
    return _reserve(_cfg("max_new_tokens", _DECISION_MAX_NEW_TOKENS_DEFAULT),
                    min_input=_MIN_DECISION_INPUT_BUDGET)


def _summary_output_reserve() -> int:
    """Generation headroom for one per-chat recap, clamped to fit the reflect window.

    A larger input budget than the decision pass gets: this one reads a whole transcript
    while that one reads five short recaps."""
    return _reserve(_cfg("summary_max_new_tokens", _SUMMARY_MAX_NEW_TOKENS_DEFAULT),
                    min_input=_MIN_SUMMARY_INPUT_BUDGET)


def _cfg(key: str, default):
    """Read ``checkin.<key>`` from server_config.json, falling back to *default*."""
    try:
        cfg = _load_server_config() if _load_server_config is not None else {}
        block = cfg.get("checkin") or {}
        val = block.get(key, default)
        return val if val is not None else default
    except Exception:
        return default


def silence_threshold_hours() -> float:
    """The configured user-silence threshold (hours) before check-in may fire."""
    try:
        return float(_cfg("silence_threshold_hours", _SILENCE_THRESHOLD_HOURS_DEFAULT))
    except Exception:
        return _SILENCE_THRESHOLD_HOURS_DEFAULT


# ── silence clock + recent window (from disk) ──────────────────────────────────

def _iter_chats():
    """Yield ``(path, data)`` for every readable chat JSON in hot/chats (not sidecars)."""
    chats_dir = Path(_CHATS_DIR)
    if not chats_dir.exists():
        return
    for path in chats_dir.glob("*.json"):
        if not is_chat_session_json(path):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield path, data


def _chat_user_key(data: dict) -> str:
    """Who this transcript is with, as a comparison key (``""`` when it names nobody)."""
    return _user_key(data.get("user"))


def _is(data: dict, key: str) -> bool:
    """True when *data* is a transcript with the person *key*; ``""`` matches everything.

    ``""`` is the unscoped mode — a corpus that attributes nothing has no scope to apply,
    and asking for one would leave check-in permanently silent there."""
    return not key or _chat_user_key(data) == key


def _user_turns(data: dict) -> int:
    """How many exchanges the *human user* actually spoke in.

    Counted structurally rather than inferred, using the SAME predicate
    :func:`_render_one_chat` uses to decide a line is the user's — a non-empty
    ``user_prompt`` whose ``speaker`` is not a stage direction. So "what the window
    filters on" and "what the recap pass is shown" can never disagree.

    An ``initiated_by:"ava"`` session's exchange 0 is skipped whatever its speaker label:
    its ``user_prompt`` is a synthetic impulse Ava wrote to herself ("About 5 hours had
    passed since you last spoke…"), and a transcript predating the ``(initiative)``
    convention would otherwise count it as the user talking.

    This replaces the old proxy (``initiated_by:"ava"`` AND ``len(exchanges) <= 1``),
    which asked whether the session had *grown* rather than whether the user was in it —
    so any second exchange, from any source, made an opener read as a conversation."""
    exchanges = data.get("exchanges") or []
    ava_opened = (data.get("initiated_by") or "").strip() == "ava"
    n = 0
    for i, ex in enumerate(exchanges):
        if ava_opened and i == 0:
            continue
        if (ex.get("speaker") or "").strip() in _NARRATOR_SPEAKERS:
            continue
        if (ex.get("user_prompt") or "").strip():
            n += 1
    return n


def _has_user_turn(data: dict) -> bool:
    """True when the transcript contains a real utterance by the *human user*.

    Two kinds of session look like conversation but contain nothing the user said, and
    both must be invisible to the silence clock and to the recap window — otherwise Ava's
    own activity answers the question "has the user gone quiet?" on the user's behalf.
    (Her own unanswered openers do reach the decision prompt, but as the separate standing
    block :func:`_standing_openers` builds — as messages she sent, never as turns they
    took. Excluding them from the *prompt* as well as the clock was the hole that let her
    restate one thought 15 times.):

    * A purely Ava-initiated opener nobody has answered — ``_user_turns`` is 0 because the
      only exchange is her own. This keeps check-in's own written sessions, and its
      outreach/synthesis siblings', from resetting the very clock they should not touch.
      Once the user replies the session has a real turn and correctly counts.
    * An encounter or served-gossip transcript (``interlocutor:"ai"``), where every
      ``user_prompt`` is another model's reply. Ava talking to a peer is not the user
      talking to Ava."""
    if (data.get("interlocutor") or "").strip() == "ai":
        return False
    return _user_turns(data) >= 1


def _last_user_turn_dt(key: str = "") -> Optional[datetime]:
    """Wall-clock time of the freshest real user exchange, or None if there are none.

    Uses the chat file's mtime as the per-session "last activity" proxy — ``ChatLogger``
    rewrites the JSON on every exchange, so a file's mtime is the time of its last turn,
    and an answered Ava opener (the user replied) correctly counts. Restart-safe (reads
    disk), unlike a monotonic clock.

    *key* scopes the clock to one person, which is the only way it means anything on a box
    with more than one: unscoped, one person answering resets the silence Ava is measuring
    toward another, and she never notices that the second has gone quiet."""
    best: Optional[datetime] = None
    for path, data in _iter_chats():
        if not _has_user_turn(data) or not _is(data, key):
            continue
        try:
            mt = datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            continue
        if best is None or mt > best:
            best = mt
    return best


def silence_hours(key: str = "") -> Optional[float]:
    """Hours since *key* last spoke, or None if we have no history for them to measure."""
    last = _last_user_turn_dt(key)
    if last is None:
        return None
    return max(0.0, (datetime.now() - last).total_seconds() / 3600.0)


def known_users() -> list[dict]:
    """Everyone the corpus shows as having actually talked to her, most recent first.

    Returns ``[{key, display, last_turn_ts}]`` — one entry per distinct person, keyed by
    :data:`_user_key` (so "Artemy" and "artemy voikhansky" are one person) and displayed
    under the spelling on their freshest transcript. The membership test is
    :func:`_has_user_turn`, the same one the silence clock uses, so a person whose only
    presence is an Ava opener nobody answered is not yet somebody to check in *on* — she
    would be measuring her own silence.

    Transcripts naming nobody are excluded: an unattributed chat belongs to no thread and
    cannot be addressed. A corpus where that is ALL there is yields an empty list, which the
    sweep reads as "nothing to scope by" and falls back to one unscoped pass."""
    best: dict[str, dict] = {}
    for path, data in _iter_chats():
        if not _has_user_turn(data):
            continue
        key = _chat_user_key(data)
        if not key:
            continue
        try:
            ts = path.stat().st_mtime
        except Exception:
            ts = 0.0
        cur = best.get(key)
        if cur is None or ts > cur["last_turn_ts"]:
            best[key] = {"key": key,
                         "display": (data.get("user") or "").strip() or key,
                         "last_turn_ts": ts}
    return sorted(best.values(), key=lambda d: d["last_turn_ts"], reverse=True)


def _chat_when(path_name: str, data: dict) -> str:
    """Human-readable date for a chat, from the filename stem or the JSON timestamp."""
    stem = path_name[:-5] if path_name.endswith(".json") else path_name
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    raw = (data.get("timestamp") or "").strip()
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return stem


def _recent_chats(n: int, min_user_turns: int = 1,
                  key: str = "") -> list[tuple[str, dict]]:
    """The *n* most recent conversations *key* actually took part in, newest first.

    Scoped to one person, because the window is what the decision pass reasons across and
    what its opener refers back to: a window mixing several people's conversations produces
    an opener to one of them about a thread that belonged to another. ``""`` keeps the
    whole corpus (the unscoped fallback).

    The bar is ``min_user_turns`` real user turns (``checkin.min_user_turns``), not merely
    "the file exists and isn't an unanswered opener". While the user is away the freshest
    files on the box are the ones *Ava's own* reach-out jobs created, so a window admitting
    a thread on one thin reply fills with sessions whose content is mostly her own openers
    — and the recap pass then hands the decision pass a summary of what she already said,
    which is the pressure toward restating it. The default of 1 keeps every genuine
    exchange (a short answer is still the user speaking); raising it to 2 demands real
    back-and-forth, at the risk of an empty window (``no_recent``) on a quiet box."""
    floor = max(1, int(min_user_turns))
    scored: list[tuple[float, str, dict]] = []
    for path, data in _iter_chats():
        if (data.get("interlocutor") or "").strip() == "ai":
            continue
        if not _is(data, key):
            continue
        if _user_turns(data) < floor:
            continue
        try:
            mt = path.stat().st_mtime
        except Exception:
            mt = 0.0
        scored.append((mt, path.name, data))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(name, data) for _, name, data in scored[:max(1, int(n))]]


def _trunc(text: str, cap: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= cap else text[: cap - 1].rstrip() + "…"


def _sent_at(name: str, data: dict) -> float:
    """When a session was *created*, as a POSIX timestamp (filename stem, else the JSON
    timestamp, else mtime). For an Ava opener this is when she sent it — mtime would be
    the time of a later reply, which is a different question."""
    stem = name[:-5] if name.endswith(".json") else name
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        pass
    try:
        return datetime.fromisoformat((data.get("timestamp") or "").strip()).timestamp()
    except Exception:
        pass
    return float(_mtime_int(name))


def _standing_openers(limit: int = _STANDING_OPENERS_CAP,
                      key: str = "") -> list[dict]:
    """The unprompted messages she has already sent since *key* last spoke, unanswered.

    ``_recent_chats`` deliberately excludes these (an unanswered opener is not the user
    talking), and that exclusion is right for the *silence clock* — her own messages must
    not answer "has the user gone quiet?" on their behalf. Applied to the *context* window
    it was a hole: the decision pass could not see what it had already sent, so every hour
    was the first time. On 2026-07-30/31 that produced 15 consecutive check-ins, each
    opening "I've been sitting with that unfinished thought" about the same idea, because
    from inside the prompt each one genuinely was the first.

    So they come back in here — not as conversations (there is nothing to summarize; she
    wrote them and no one answered) but as a plain standing list, rendered verbatim and
    newest last. Bounded to *limit*, with any older ones reported as a count.

    Scoped to one person like everything else here: what she has said into THEIR silence.
    Messages sent to somebody else are not something this person is failing to answer, and
    quoting them into their prompt would ask her to notice being ignored by the wrong
    person.

    Openers whose transcripts the stale-reach-out sweep has since deleted
    (``chat_worklog.delete_stale_reachouts``) are folded back in from their tombstones —
    she still said them and still was not answered, and they are by construction the
    OLDEST ones, i.e. the evidence that this has been going on a while. Reading only what
    remains on disk would hand the pass a window that resets itself every two days, which
    is the shape of the bug this list was added to fix."""
    last_user = _last_user_turn_dt(key)
    cutoff = last_user.timestamp() if last_user is not None else None
    out: list[dict] = []
    seen_stems: set[str] = set()
    for path, data in _iter_chats():
        if (data.get("initiated_by") or "").strip() != "ava":
            continue
        if not _is(data, key):
            continue
        exchanges = data.get("exchanges") or []
        if _user_turns(data) >= 1:
            continue                      # answered — it is a real conversation now
        ts = _sent_at(path.name, data)
        if cutoff is not None and ts <= cutoff:
            continue                      # predates their last turn; they did engage since
        opener = (exchanges[0].get("assistant_response") if exchanges else "") or ""
        if not opener.strip():
            continue
        seen_stems.add(path.stem)
        out.append({"name": path.name, "ts": ts, "when": _chat_when(path.name, data),
                    "opener": opener.strip()})
    for rec in reachout_gate.expired_openers(key):
        stem = str(rec.get("stem") or "")
        if not stem or stem in seen_stems:
            continue
        opener = str(rec.get("opener") or "").strip()
        if not opener:
            continue                      # tombstoned before her text was kept
        ts = float(rec.get("sent_at") or 0.0)
        if cutoff is not None and ts <= cutoff:
            continue
        name = f"{stem}.json"
        seen_stems.add(stem)
        out.append({"name": name, "ts": ts, "when": _chat_when(name, {}),
                    "opener": opener, "expired": True})
    out.sort(key=lambda d: d["ts"])       # oldest → newest
    if limit > 0 and len(out) > limit:
        dropped = len(out) - limit
        out = out[-limit:]
        out[0]["earlier"] = dropped
    return out


def _render_standing(openers: list[dict], human: str) -> str:
    """Render the standing unanswered openers as their own labelled block.

    Folded into the existing ``{recent}`` slot rather than a new prompt placeholder, so an
    operator's already-customized ``checkin_prompt.txt`` on disk gets this too instead of
    silently dropping it."""
    if not openers:
        return ""
    lines = [f"Since {human} last spoke, you have already reached out to them on your own "
             f"initiative. They have not replied to any of it:"]
    earlier = openers[0].get("earlier")
    if earlier:
        lines.append(f"  (…{earlier} earlier message(s), also unanswered)")
    for d in openers:
        lines.append(f"  [{d['when']}] you wrote: {_trunc(d['opener'], 400)}")
    lines.append(f"You are looking at {len(openers) + (earlier or 0)} unanswered "
                 f"message(s) of your own. Saying the same thing again is unlikely to be "
                 f"what you want; if you do reach out, it should be because you have "
                 f"something genuinely different to say.")
    return "\n".join(lines)


def _render_one_chat(name: str, data: dict, *, per_chat: int = 16,
                     turn_cap: int = 600, total_cap: int = 8000) -> str:
    """Render ONE chat into a bounded, human-readable transcript.

    Keeps the last *per_chat* exchanges, each turn truncated to *turn_cap* chars, the
    whole capped at *total_cap*. This is the input a per-chat summary reads (a single
    chat comfortably fits the reflect window), and the raw fallback when summarizing that
    chat fails. An `(initiative)`/`(setting)` exchange-0 `user_prompt` is a synthetic
    impulse, not something the user said, so only Ava's opener is rendered for it — and
    when SHE opened the session that opener is labelled as an unprompted message rather
    than shown as an ordinary turn. Unlabelled it is simply the first thing in the
    transcript, so the recap pass reads her own message as what the conversation was
    about, and the decision pass is then handed a summary of what she already said."""
    exchanges = data.get("exchanges") or []
    human = (data.get("user") or "").strip() or "them"
    ava_opened = (data.get("initiated_by") or "").strip() == "ava"
    kept = list(enumerate(exchanges))[-per_chat:]
    lines: list[str] = []
    total = 0
    for idx, ex in kept:
        speaker = (ex.get("speaker") or "").strip()
        up = (ex.get("user_prompt") or "").strip()
        resp = (ex.get("assistant_response") or "").strip()
        rendered: list[str] = []
        if up and speaker not in _NARRATOR_SPEAKERS:
            who = speaker or human
            rendered.append(f"  {who}: {_trunc(up, turn_cap)}")
        if resp:
            if ava_opened and idx == 0:
                rendered.append(f"  You (reaching out unprompted — nobody asked you "
                                f"anything yet): {_trunc(resp, turn_cap)}")
            else:
                rendered.append(f"  You: {_trunc(resp, turn_cap)}")
        chunk = "\n".join(rendered)
        if not chunk:
            continue
        if lines and total + len(chunk) > total_cap:
            break
        lines.append(chunk)
        total += len(chunk)
    return "\n".join(lines)


# ── prompt + decision parsing ──────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "It has been about {hours} hours since you last spoke with {user}.\n\n"
    "Here are brief recaps of your most recent conversations with them, oldest first:\n\n"
    "{recent}\n\n"
    "Look across all of them together — not only the latest — noticing anything that "
    "connects them, recurs, or was left open. Then, on your own initiative, you may start "
    "a fresh conversation with {user} — to pick up a loose thread, share a thought that "
    "has stayed with you, or simply reach out because time has passed — or you may decide "
    "there is nothing you genuinely want to say. No one has asked you to. Only reach out "
    "if you truly want to, now; silence is a perfectly good answer.\n\n"
    "Think it through, then answer in exactly this form:\n\n"
    "DECISION: yes            (or: no)\n"
    "OPENER: <if yes, the message you would open with — your own words, addressed to "
    "{user}, in your own voice.>\n"
)


def _load_prompt() -> str:
    """Check-in decision prompt (``{hours}``/``{user}``/``{recent}`` slots), overridable."""
    path = _PROMPTS_DIR / "checkin_prompt.txt"
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return _DEFAULT_PROMPT


# Same rationale as outreach's former helper — the CoT discusses its own choice, so the
# trace comes off before any label is matched — and this module carried the same partial
# copy while being the other subsystem that WRITES the parsed text into a chat the user
# opens. The complete version lives in `core.reasoning_text`.
_answer_after_think = reasoning_text.answer_after_think
_has_reasoning_leak = reasoning_text.has_reasoning_leak


def _parse_decision(text: str) -> tuple[str, str]:
    """Lenient parse of the DECISION/OPENER block → (decision, opener).

    *decision* is ``"yes"`` or ``"no"``; an absent/unrecognized ``DECISION:`` line
    defaults to ``"no"`` so a truncated generation reads as a plain decline. The
    ``<think>`` CoT is stripped first so the reasoning can't be mistaken for the
    decision."""
    text = _answer_after_think(text)
    decision = "no"
    m = re.search(_label("DECISION") + r"(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        d = m.group(1).strip().lower()
        if d.startswith(("yes", "y", "true", "1", "sure")):
            decision = "yes"
    opener = ""
    m = re.search(_label("OPENER", anchored=False) + r"(.*)\Z", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        opener = m.group(1).strip()
    return decision, opener


# ── per-chat summarization (the recent window doesn't fit raw) ──────────────────

# Recap cache keyed by (chat filename, integer mtime): an hourly check-in that keeps
# declining shouldn't re-summarize the unchanged recent window every run — only the final
# decision pass reruns. A chat that gains a turn changes its mtime and is re-summarized.
_SUMMARY_CACHE: dict[tuple[str, int], str] = {}
_SUMMARY_CACHE_CAP = 512

# Per-chat recap budget. Thinking is ON here (the pass runs through the reflect seam,
# which prefills the channel), so this covers a full <think> *plus* the RECAP line — and
# 768 was the tightest cap of any thinking-on pass on the box, against the largest input
# (a whole transcript). 2048 was the first raise; window-derived since 2026-08-06
# (`_summary_output_reserve`, knob `checkin.summary_max_new_tokens`).
#
# It stayed the tightest budget on the box against the largest input even at 2048: under
# the thought ceiling that is ~1434 tokens of thought to read up to 8000 characters of
# transcript, and only ~614 for a recap that needs ~100. So the half that overflows is the
# READING, not the writing, which is why the answer never looked short. 8192 (~5735 /
# 2457) gives the reading room; the ceiling still guarantees the RECAP line gets written.
#
# The cost of the raise is smaller than it looks in both directions. The cap only bounds a
# runaway — a recap that finishes in 200 tokens still costs 200 — the pass is cached per
# (filename, mtime), and a truncated recap is deliberately left UNCACHED, so today an
# overflowing chat is re-generated and re-logged on every hourly check-in, forever. Fewer
# cutoffs is strictly less repeated work, not more.
_SUMMARY_MAX_NEW_TOKENS_DEFAULT = 8192
# Room held back for a recap prompt: system prompt + one `_render_one_chat` transcript
# (capped at 8000 chars, which is ~4k tokens of mixed Russian/English).
_MIN_SUMMARY_INPUT_BUDGET = 8192

_DEFAULT_SUMMARY_PROMPT = (
    "Below is one of your recent conversations with {user} ({when}).\n\n"
    "{transcript}\n\n"
    "In 2-3 sentences, recap it for yourself: what it was about, and where it left off "
    "— any thread still open, question left unanswered, or something unresolved. Write it "
    "as a plain note to yourself. Do not address {user}, and add nothing that isn't in "
    "the conversation.\n\n"
    "Recap the conversation, not your own side of it. If you opened this one unprompted, "
    "your opening message is where it started, not what it was about — what matters is "
    "what {user} made of it, whether they engaged with it, and what they brought to it "
    "themselves. If they said very little, say so plainly; a thin exchange recapped as a "
    "rich one is worse than no recap.\n\n"
    "RECAP: <your 2-3 sentence recap>\n"
)

_RECAP_RE = re.compile(_label("RECAP", anchored=False) + r"(.*)\Z",
                       re.IGNORECASE | re.DOTALL)


def _load_summary_prompt() -> str:
    """Per-chat recap prompt (``{when}``/``{user}``/``{transcript}``), overridable."""
    path = _PROMPTS_DIR / "checkin_summary_prompt.txt"
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return _DEFAULT_SUMMARY_PROMPT


def _parse_recap(text: str) -> str:
    """Pull the recap out of a summary generation (``RECAP:`` label, CoT stripped)."""
    answer = _answer_after_think(text)
    m = _RECAP_RE.search(answer)
    return (m.group(1).strip() if m else answer).strip()


def _mtime_int(name: str) -> int:
    try:
        return int((Path(_CHATS_DIR) / name).stat().st_mtime)
    except Exception:
        return 0


# ── the stored gist: a recap reflection already paid for ───────────────────────
#
# Reflection writes one prose recap per conversation to the chat's sidecar (the
# consolidation summary, `ChatSidecar.write_summary` → `summary_text`), which is a recap of
# the same conversation this pass was about to generate a recap of. A box that has reflected
# a chat therefore paid a second generation, per check-in, to say again what it already had
# on disk — up to `checkin.recent_chats` (5) of them per hourly decision, forever, since an
# aged chat's window never changes.
#
# Are the two interchangeable? Not exactly, and the difference is worth stating rather than
# papering over. The gist is written for RAG injection — what the conversation WAS, phrased
# to be read back mid-turn as a remembered conclusion. The check-in recap is written to feed
# a "should I reach out?" decision, so it is oriented at what was left OPEN, and carries a
# guard the summary pass has no reason to carry (recap the conversation, not your own side of
# it — the trap on a session she opened unprompted).
#
# Two things follow. (1) Reframing the gist through a cheap second pass is not an option: a
# generation is exactly the cost being saved, so a "cheaper framing" that generates buys
# nothing. The framing is applied for free at RENDER time instead (`_render_digests`), where
# the block says which kind of note each recap is — so the decision pass does not read a
# gist's silence about loose ends as "nothing was left open". (2) A gist is a full prose
# recap where this pass asks for 2-3 sentences; five raw ones would take ~7k tokens against
# the decision pass's ~4k input budget and would drown any generated recap beside them, so
# they are excerpted to recap size first.
#
# The generated path is unchanged, cache included — it is now the fallback for a chat
# reflection has not reached yet.

# Recap-sized allowance for a stored gist. A generated recap is 2-3 sentences (~400 chars);
# the gist gets a little more room because it was written to a different brief and needs it
# to carry its point, but not so much that one aged chat crowds out the other four.
_GIST_EXCERPT_CHARS = 800


def _gist_excerpt(text: str, cap: int = _GIST_EXCERPT_CHARS) -> str:
    """Trim a stored gist to recap size, ending on a finished thought.

    The rule itself moved to ``chat_sidecar.gist_excerpt`` when the fact-nomination slot
    became its second caller — it is a gist-shaping rule and belongs beside
    ``sanitize_gist``/``summary_text``. Kept here as a name with this module's own default
    cap, so the call sites and the self-test below read unchanged."""
    return sidecar_gist_excerpt(text, cap)


def _stored_gist(name: str) -> str:
    """The reflection recap already on disk for *name*, excerpted — ``""`` if there is none.

    ``summary_text`` sanitizes on read, so a chat whose stored summary is an unsalvageable
    structured dump reads as having none and falls through to the generated path."""
    try:
        return _gist_excerpt(ChatSidecar(Path(_CHATS_DIR)).summary_text(name))
    except Exception:
        return ""


def _summarize_recent(chats: list[tuple[str, dict]], generate,
                      *, on_stage: Optional[Callable[[dict], None]] = None,
                      on_prompt: Optional[Callable[[str, dict], None]] = None) -> list[dict]:
    """Recap each recent chat (topic + where it left off), generating only where needed.

    Returns ``[{name, when, user, recap, source}]`` in the SAME order as *chats*. This is
    the heart of the "5 full chats don't fit one window" fix: instead of concatenating raw
    transcripts (which starved every chat after the newest), each chat is reduced to a
    short recap, and only the compact recaps go into the decision prompt — so all of the
    recent window is genuinely represented and the decision pass reasons *across* it.

    A recap comes from one of two places, reported as ``source``. A chat reflection has
    already summarized carries its gist on disk (``source="reflection"``, read + excerpted
    by :func:`_stored_gist`) and costs no generation at all — see the note above that
    helper for why the two are close enough to substitute and where they differ. Everything
    else is generated by its own cheap pass (``source="generated"``), cached per (filename,
    mtime) so an hourly check-in that keeps declining doesn't re-summarize an unchanged
    window; a failure falls back to a bounded raw excerpt so the chat is still present.

    Streams a ``summarizing`` stage per chat about to be generated and a ``recapped`` stage
    carrying the finished recap for EVERY chat (debug view), but not its reasoning deltas —
    only the final decision pass streams tokens. The recap text rides the stage event
    because it is the whole input the decision pass reasons over: without it a watcher sees
    "Recapping recent chat 3/5…" and then a yes/no with no way to tell whether the window
    she read was any good. ``source`` and ``cached`` together say where each one came from:
    ``cached`` means the in-memory recap cache specifically, and both a cached recap and a
    stored gist are stable across wakes, so the autonomous journal logs neither body hourly.
    """
    system_prompt = _session.system_prompt
    # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
    # for every pass now — see generation._reflect_system_parts. Appending here too
    # would date-stamp the prompt twice.)
    tmpl = _load_summary_prompt()
    reserve = _summary_output_reserve()
    out: list[dict] = []
    n = len(chats)

    def _stage(**kw) -> None:
        if on_stage is not None:
            try:
                on_stage(kw)
            except Exception:
                pass

    for i, (name, data) in enumerate(chats):
        when = _chat_when(name, data)
        human = (data.get("user") or "").strip() or "them"
        # Reflection's own recap of this conversation, if it has one — a generation this
        # box already paid for. Checked before the cache, since a chat can be reflected
        # after a recap of it was generated and cached, and the gist is then the better
        # (and free) answer.
        recap = _stored_gist(name) or None
        source = "reflection" if recap else "generated"
        key = (name, _mtime_int(name))
        if recap is None:
            recap = _SUMMARY_CACHE.get(key)
        cached = source == "generated" and recap is not None
        truncated = False
        if recap is None:
            _stage(stage="summarizing", i=i + 1, n=n, session=name)
            transcript = _render_one_chat(name, data)
            content = (tmpl.replace("{when}", when)
                           .replace("{user}", human)
                           .replace("{transcript}", transcript))
            recap = ""
            try:
                raw = generate(
                    content, system_prompt,
                    temperature=0.5, top_p=0.9,
                    max_new_tokens_setting=str(reserve),
                    disable_rag=True,
                    on_prompt_debug=(
                        None if on_prompt is None
                        else (lambda info, _n=name, _i=i + 1:
                              on_prompt(f"recap {_i}/{n} · {_n}", info))),
                )
                recap = _parse_recap(raw)
                truncated = bool(getattr(generate, "last_truncated", None))
            except Exception:
                traceback.print_exc()
            if truncated:
                # The one pass in this family that reported nothing on a cutoff. Its
                # sibling decision pass, outreach and deliberation all separate a cutoff
                # from a real answer; here a recap cut mid-sentence still PARSES, so it
                # entered the decision window as a half-thought with no signal at all —
                # and, worse, was frozen into the cache under (filename, mtime), which for
                # an unchanging old chat means forever. Say so, and leave it uncached so
                # the next wake regenerates it.
                print(f"[checkin] recap for {name} hit the {reserve}-token cap — not "
                      f"caching it, will retry on the next check-in", flush=True)
            if not recap:
                # Fallback: a short raw excerpt so a failed/empty summary still leaves the
                # chat represented in the window rather than silently dropped.
                excerpt = _render_one_chat(
                    name, data, per_chat=4, turn_cap=300, total_cap=1200)
                recap = "(recap unavailable — recent excerpt) " + _trunc(
                    excerpt.replace("\n", " "), 700)
            if len(_SUMMARY_CACHE) > _SUMMARY_CACHE_CAP:
                _SUMMARY_CACHE.clear()
            if not truncated:
                _SUMMARY_CACHE[key] = recap
        _stage(stage="recapped", i=i + 1, n=n, session=name, when=when, user=human,
               recap=recap, cached=cached, truncated=truncated, source=source)
        out.append({"name": name, "when": when, "user": human, "recap": recap,
                    "source": source})
    return out


def _render_digests(digests: list[dict]) -> str:
    """Assemble the summarized recent window for the decision prompt, oldest first so the
    arc reads chronologically and the freshest conversation is what she reads last.

    A recap taken from reflection's stored gist is labelled as one. The two sources are
    close enough to substitute — both are short prose recaps of the same conversation — but
    they were written to different briefs: a check-in recap is asked what was left open,
    while a gist is asked what the conversation was. Unlabelled, a gist that says nothing
    about loose ends reads as a conversation that had none. This is the whole of the
    reframing, and it costs nothing: a second generation to restate the gist in this pass's
    register would cost exactly what reading the gist saved."""
    lines = []
    for d in reversed(digests):      # `digests` is newest-first; present oldest→newest
        head = f"[{d['when']}] with {d['user']}:"
        if d.get("source") == "reflection":
            head = (f"[{d['when']}] with {d['user']} — from your own reflection on it "
                    f"(what it was about; it may not say where things were left):")
        body = "\n".join(f"  {ln}" for ln in (d["recap"] or "").splitlines() if ln.strip())
        lines.append(f"{head}\n{body}")
    return "\n\n".join(lines)


def _target_user() -> str:
    """Name Ava would address — the last known speaker, or a neutral fallback."""
    return (_session.user or "").strip() or "your friend"


def _write_checkin_session(*, opener: str, human: str, hours: float) -> str:
    """Write the reversed check-in session to hot/chats and return its filename.

    Same shape as outreach/synthesis: exchange 0 is Ava's opener under ``(initiative)``
    with a synthetic impulse as its ``user_prompt`` and ``initiated_by:"ava"``. Not made
    active — it simply lands on disk, showing up in the session list; the user adopts it
    in place and replies into the same file."""
    impulse = (f"About {hours:.0f} hours had passed since you last spoke with {human}. "
               f"Looking back over your recent conversations, you decided to reach out.")
    Path(_CHATS_DIR).mkdir(parents=True, exist_ok=True)
    logger = ChatLogger(_CHATS_DIR)
    logger.start_session(
        _session.system_prompt, user=human,
        model_id=_runtime.model_id, adapter_id=_runtime.adapter_id,
        notes=(f"Ava-initiated check-in after ~{hours:.0f}h of silence: reached out on "
               f"her own accord after reviewing recent chats."),
        initiated_by="ava",
    )
    logger.log_exchange(
        impulse, opener, speaker=_INITIATIVE_SPEAKER,
        system_content=_session.system_prompt,
    )
    return logger.current_file.name if logger.current_file else ""


# ── decision pass (idle heartbeat) ─────────────────────────────────────────────

def run_checkin_decision_blocking(
    *,
    user: Optional[str] = None,
    forced_hours: Optional[float] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
    on_prompt: Optional[Callable[[str, dict], None]] = None,
    bypass_cooldown: bool = False,
) -> dict:
    """One check-in decision for ONE person (runs on the GPU executor thread).

    *user* is who this decision is about: their silence clock, their recent window, their
    standing openers, their reach-out gate, and the name the opener is addressed to. A
    *user* that names nobody (None, blank, "the user") runs the pass unscoped over the whole
    corpus — the historical behaviour, kept for a box whose transcripts carry no attribution
    to scope by. The idle job does not call this directly; it walks
    :func:`run_checkin_sweep_blocking`, which calls this once per person.

    Autonomous path (``forced_hours=None``): measures the real user-silence from disk,
    skips (``insufficient_silence``) if it is below ``checkin.silence_threshold_hours``,
    otherwise reviews the recent window and decides yes/no. On **yes** with an opener,
    writes the reversed session and stamps the shared reach-out cool-down; on **no**,
    returns a ``declined`` skip.

    Manual/simulate path (``forced_hours`` set — the Sleep-tab "Check In" button):
    bypasses the silence gate entirely and forges the elapsed period, running the exact
    same decision with ``{hours}`` set to *forced_hours* so the operator can watch Ava
    decide as if that much time had passed. It still writes a real session on a yes.

    ``on_chunk``/``on_prompt`` are debug hooks used only by the manual trigger (the
    autonomous caller routes ``on_stage`` into the activity journal but passes neither of
    the other two, so its behaviour is unchanged). ``on_prompt(label, info)`` fires once
    per generation with the FULL model-facing prompt of that pass, split into the same
    labelled segments live chat's Debug checkbox renders — the recap passes and the
    decision pass each report their own. It is what makes this subsystem's retrieval
    inspectable: both of its passes currently run ``disable_rag=True``, which the payload
    states outright (``rag_disabled``) rather than leaving to be inferred from an absent
    block.

    ``bypass_cooldown`` skips the shared reach-out rate limit — the manual "Check In"
    sets it so an operator's on-demand run always sends on a yes, even inside another
    reach-out's window. A manual send still *stamps* the gate afterward."""
    global _checkin_active
    _checkin_active = True
    try:
        return _decide_for_user(
            user=user, forced_hours=forced_hours, on_chunk=on_chunk,
            on_stage=on_stage, on_prompt=on_prompt, bypass_cooldown=bypass_cooldown)
    finally:
        _checkin_active = False


def _decide_for_user(
    *,
    user: Optional[str] = None,
    forced_hours: Optional[float] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
    on_prompt: Optional[Callable[[str, dict], None]] = None,
    bypass_cooldown: bool = False,
) -> dict:
    """The decision itself, without the occupancy flag — see the public wrapper above.

    Split out so :func:`run_checkin_sweep_blocking` can hold ``_checkin_active`` across a
    whole sweep instead of dropping it between people."""

    def _stage(**kw) -> None:
        if on_stage is not None:
            try:
                on_stage(kw)
            except Exception:
                pass

    try:
        if _runtime.model is None:
            return {"skipped": "no_model"}

        # Who this pass is about. An unnamed target is the unscoped fallback: measure and
        # address the box as a whole, as check-in did before it ran per person.
        key = _user_key(user)
        human = (str(user).strip() if key else "") or _target_user()

        real = silence_hours(key)
        simulated = forced_hours is not None
        if simulated:
            # Forge the period, but keep it realistic: never below the threshold, and
            # the true elapsed silence when that is genuinely longer.
            threshold = silence_threshold_hours()
            hours = max(float(forced_hours), threshold)
            if real is not None:
                hours = max(hours, real)
        else:
            if real is None:
                return {"skipped": "no_history", "user": human}
            if real < silence_threshold_hours():
                return {"skipped": "insufficient_silence", "silence_hours": real,
                        "threshold_hours": silence_threshold_hours(), "user": human}
            stale_after = max_silence_hours()
            if stale_after and real > stale_after:
                # Long past a stretch of silence — this is someone who stopped talking, not
                # someone who has gone quiet. See _MAX_SILENCE_DAYS_DEFAULT.
                return {"skipped": "stale", "silence_hours": real,
                        "max_silence_hours": stale_after, "user": human}
            hours = real

        recent = _recent_chats(_cfg("recent_chats", _RECENT_CHATS_DEFAULT),
                               _cfg("min_user_turns", _MIN_USER_TURNS_DEFAULT), key)
        if not recent:
            return {"skipped": "no_recent", "silence_hours": hours, "user": human}
        _stage(stage="considering", hours=round(hours, 1), chats=len(recent),
               user=human, simulated=simulated)

        rag = _get_rag()
        generate = _make_sync_reflect_generate(rag)

        # The recent window doesn't fit raw (5 full chats overflow one context), so
        # compress each chat to a short recap first, then reason across the compact set —
        # every recent chat is represented instead of only whichever fit newest-first.
        digests = _summarize_recent(recent, generate, on_stage=on_stage,
                                    on_prompt=on_prompt)
        recent_block = _render_digests(digests)
        # …and, after the conversations, what she has already said into this silence. The
        # recaps above cover only chats the user took part in; without this she cannot see
        # her own standing openers and re-sends the same thought every window.
        standing = _standing_openers(key=key)
        if standing:
            recent_block = recent_block + "\n\n" + _render_standing(standing, human)
        _stage(stage="deciding", chats=len(digests), user=human,
               standing=len(standing))

        content = (_load_prompt()
                   .replace("{hours}", f"{hours:.0f}")
                   .replace("{user}", human)
                   .replace("{recent}", recent_block))
        system_prompt = _session.system_prompt
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)

        # Thinking on (deliberated decision + streamable reasoning), so the budget must
        # cover a full <think> CoT plus the DECISION/OPENER block — where OPENER is a whole
        # chat message. Sized from the reflect window rather than pinned; see the note on
        # _DECISION_MAX_NEW_TOKENS_DEFAULT for the 2048 → 4096 → 8192 → window history and
        # why a constant was the wrong shape. RAG is disabled — the recent window is
        # already in the prompt.
        raw = generate(
            content, system_prompt,
            temperature=0.7, top_p=0.95,
            max_new_tokens_setting=str(_decision_output_reserve()),
            disable_rag=True, on_chunk=on_chunk,
            on_prompt_debug=(None if on_prompt is None
                             else (lambda info: on_prompt("decision", info))),
        )
        decision, opener = _parse_decision(raw)

        # Refuse a generation cut off before it closed its reasoning, BEFORE the write path
        # rather than only on the decline branch below. A gemma-4 channel truncated before
        # its close normalizes to untagged prose, so `_answer_after_think` has nothing to
        # strip and `_parse_decision` can read a whole structured answer out of the
        # deliberation — which then reaches `_write_checkin_session` as a chat the user
        # opens. See the fuller note in `outreach.run_outreach_decision_blocking`.
        if reasoning_text.truncated_before_answer(
                raw, getattr(generate, "last_truncated", None)):
            print("[checkin] discarding decision: generation was cut off before it closed "
                  "its reasoning — no answer to parse", flush=True)
            return {"skipped": "truncated", "decision": False, "opener": "",
                    "silence_hours": hours, "simulated": simulated,
                    "standing": len(standing), "user": human}

        if decision != "yes" or not opener:
            if getattr(generate, "last_truncated", None):
                return {"skipped": "truncated", "decision": False, "opener": opener,
                        "silence_hours": hours, "simulated": simulated,
                        "standing": len(standing), "user": human}
            return {"skipped": "declined", "decision": False,
                    "silence_hours": hours, "simulated": simulated,
                    "standing": len(standing), "user": human}

        # A truncated YES is refused too, not only a truncated decline. The check above only
        # guards the branch a truncation usually lands on (no DECISION parsed → default
        # "no"); a generation that closed its reasoning, decided yes and then ran out
        # mid-message parses as a good yes with a half-written opener — `_parse_decision`
        # takes the opener from its label to the end of the text, so the cut tail IS the
        # message. Nothing is lost: the silence persists, so the next window re-decides.
        # See synthesis / outreach for the same guard.
        if getattr(generate, "last_truncated", None):
            print("[checkin] discarding opener: generation hit the token cap mid-message "
                  "— refusing to send a partial message", flush=True)
            return {"skipped": "truncated", "decision": True, "opener": "",
                    "silence_hours": hours, "simulated": simulated,
                    "standing": len(standing), "user": human}

        # Last backstop before anything reaches a chat: a surviving reasoning marker means
        # the opener is not a clean message whatever else parsed.
        if _has_reasoning_leak(opener):
            print("[checkin] discarding opener: reasoning markers survived into the "
                  "message", flush=True)
            return {"skipped": "opener_leak", "decision": True, "opener": "",
                    "silence_hours": hours, "simulated": simulated,
                    "standing": len(standing), "user": human}

        # Shared reach-out rate limit: if outreach / synthesis just cold-opened the user,
        # don't stack a second unprompted message on top. Checked HERE, at the point of
        # sending, rather than as an idle-job gate — the silence that triggered this
        # check-in persists, so the next window re-decides on fresh evidence. (See
        # core.reachout_gate on why gating the job instead of the message starved
        # whichever sibling the scheduler tried last — check-in was that sibling.)
        # A manual "Check In" (bypass_cooldown) always sends — the operator asked for it.
        #
        # Scoped to THIS person (``None`` in the unscoped fallback, which is the old global
        # question): the limit is on how often she cold-opens someone, and a message to one
        # person is not a reason to withhold one from another — nor is one person leaving
        # her unanswered evidence that a second is. The gate's scoped view still counts an
        # outreach/synthesis send that went to this same person, so nothing is lost.
        gate_user = human if key else None
        gate_ok, gate_reason = ((True, "") if bypass_cooldown
                                else reachout_gate.may_reach_out(gate_user))
        if not gate_ok:
            print(f"[checkin] holding opener after ~{hours:.0f}h silence — {gate_reason}",
                  flush=True)
            return {"skipped": gate_reason, "decision": True, "opener": opener,
                    "silence_hours": hours, "simulated": simulated,
                    "standing": len(standing), "user": human}

        filename = _write_checkin_session(opener=opener, human=human, hours=hours)
        # A reach-out was written → start the shared cool-down so outreach/synthesis
        # don't pile a second unprompted message on top of it in the same window.
        reachout_gate.mark_reachout(gate_user)
        print(f"[checkin] wrote check-in session {filename} for {human} after ~{hours:.0f}h "
              f"silence ({'simulated' if simulated else 'real'})", flush=True)
        # Episodic worklog: after a stretch of silence I decided to check in — record it in
        # my own voice, with the loop left open (awaiting their reply).
        try:
            from core import worklog
            worklog.record(
                "checkin",
                f"{human} had been quiet for about {hours:.0f} hours, so I reached out to "
                f"check in.",
                refs={"session": filename},
                opens=f"awaiting {human}'s reply to my check-in",
            )
        except Exception:
            traceback.print_exc()
        return {"composed": True, "session": filename, "decision": True,
                "opener": opener, "silence_hours": hours, "chats": len(recent),
                "simulated": simulated, "standing": len(standing), "user": human}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


# ── the sweep: one decision per person (what the idle job runs) ────────────────

# Skips that cost no generation — pure disk reads, plus `deferred` (a candidate the
# `checkin.max_users` cap held back before it was even measured). They do not spend a slot
# of that cap, and a sweep of nothing but these is reported to the scheduler as an interval
# NOT consumed, so it retries on the next poll rather than sleeping the hour out.
_CHEAP_SKIPS = frozenset({"no_model", "no_history", "insufficient_silence", "no_recent",
                          "stale", "deferred"})


def max_silence_hours() -> float:
    """Silence past which a person stops being a check-in candidate (0 ⇒ no ceiling)."""
    try:
        return max(0.0, float(_cfg("max_silence_days",
                                   _MAX_SILENCE_DAYS_DEFAULT))) * 24.0
    except Exception:
        return _MAX_SILENCE_DAYS_DEFAULT * 24.0


def run_checkin_sweep_blocking(
    *,
    on_stage: Optional[Callable[[dict], None]] = None,
    max_users: Optional[int] = None,
) -> dict:
    """One check-in wake: decide separately for each person she talks to.

    This is what the idle job runs. It walks :func:`known_users` most-recently-active first
    and calls :func:`_decide_for_user` per person — each with their own silence clock, recent
    window, standing openers and reach-out gate — so a box with several users produces
    several independent decisions instead of one that mixes them. Most candidates cost
    nothing: the silence and staleness gates are disk reads, and only a person who clears
    them reaches a generation, of which at most ``checkin.max_users`` run per sweep.

    A corpus that names nobody (no attribution to scope by) falls back to a single unscoped
    pass — check-in's historical behaviour.

    The result aggregates the per-person ones: ``results`` carries each, ``sent`` the ones
    that composed, and the first send is also lifted to the top level (``opener`` /
    ``session`` / ``user`` / ``silence_hours``) so the activity journal's ``describe`` reads
    one line. A sweep in which nobody was written to reports a top-level ``skipped`` naming
    the reason that actually cost something, so the scheduler can tell a wake that ran
    decisions from one that only read disk."""
    global _checkin_active
    _checkin_active = True
    try:
        if _runtime.model is None:
            return {"sweep": True, "skipped": "no_model"}

        people = known_users()
        if not people:
            # Nothing attributed to scope by — one unscoped pass, as before.
            result = _decide_for_user(on_stage=on_stage)
            result.setdefault("sweep", True)
            result["users"] = 0
            return result

        try:
            cap = int(max_users if max_users is not None
                      else _cfg("max_users", _MAX_USERS_DEFAULT))
        except Exception:
            cap = _MAX_USERS_DEFAULT
        cap = max(1, cap)

        results: list[dict] = []
        sent: list[dict] = []
        ran = 0
        for person in people:
            if ran >= cap:
                results.append({"user": person["display"], "skipped": "deferred"})
                continue
            try:
                r = _decide_for_user(user=person["display"], on_stage=on_stage)
            except Exception as e:
                traceback.print_exc()
                r = {"error": f"{type(e).__name__}: {e}"}
            r.setdefault("user", person["display"])
            results.append(r)
            if str(r.get("skipped") or "") not in _CHEAP_SKIPS:
                ran += 1
            if r.get("composed"):
                sent.append({"user": r.get("user"), "session": r.get("session"),
                             "opener": r.get("opener"),
                             "silence_hours": r.get("silence_hours")})

        out: dict = {"sweep": True, "users": len(people), "ran": ran,
                     "composed": len(sent), "sent": sent, "results": results}
        if sent:
            # Lift the first send so the generic `describe`/client paths read one check-in.
            out.update({k: v for k, v in sent[0].items() if v is not None})
            return out
        out["skipped"] = _sweep_skip_reason(results)
        return out
    except Exception as e:
        traceback.print_exc()
        return {"sweep": True, "error": f"{type(e).__name__}: {e}"}
    finally:
        _checkin_active = False


def _sweep_skip_reason(results: list[dict]) -> str:
    """One reason standing for a sweep that wrote nothing.

    Prefers a reason that cost a generation (declined / truncated / a gate hold) over the
    cheap disk-read ones, because that is the distinction the scheduler's ``consumed``
    predicate turns into "sleep the interval" vs "retry next poll": a wake in which someone
    genuinely deliberated and said no has done its work."""
    if any(r.get("error") for r in results):
        return "error"
    expensive = [str(r.get("skipped") or "") for r in results
                 if str(r.get("skipped") or "") not in _CHEAP_SKIPS and r.get("skipped")]
    if expensive:
        # "declined" is the one worth naming when several people were weighed.
        return "declined" if "declined" in expensive else expensive[0]
    reasons = [str(r.get("skipped") or "") for r in results if r.get("skipped")]
    return reasons[0] if reasons else "nothing_to_do"


# ── manual (Sleep-tab) trigger ─────────────────────────────────────────────────

async def handle_checkin_now(ws, msg: dict) -> None:
    """Manually run one check-in decision and stream Ava's reasoning (debug/simulate).

    The Sleep-tab "Check In" button routes here. It shares the exact code path the
    autonomous idle heartbeat uses (:func:`run_checkin_decision_blocking`) but **forges
    the silence period** so the operator can watch Ava decide as if the threshold had
    been crossed even when it hasn't. The forged hours default to the configured
    threshold (``max``-clamped against the real elapsed silence, so it stays realistic);
    a client may pass ``sim_hours`` to override. On a yes it writes the reversed session
    exactly as the autonomous path does; on a no it reports the decision without writing.

    **Who it is about**: one person, like the autonomous sweep's per-person pass — the
    client's ``user`` field if it sends one, else the active session's speaker. It runs a
    single decision rather than the whole sweep because the point of the button is watching
    one deliberation stream by, and a sweep would interleave several people's recaps and
    reasoning into one log with no way to tell them apart.

    Protocol: streams ``checkin_stage`` (the window under consideration —
    ``considering`` / ``summarizing`` / ``recapped``, the last carrying the finished recap
    text of one chat, / ``deciding``), ``checkin_prompt`` (one per generation: the full
    model-facing prompt of that pass in live chat's Debug segment shape) and
    ``checkin_chunk`` reasoning deltas, finishing with ``checkin_done`` carrying the
    outcome (``composed``/``decision``/``session``/``opener``/``silence_hours`` or a
    ``skipped`` reason)."""
    loop = asyncio.get_event_loop()
    if _checkin_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "checkin_done", "skipped": "busy",
                         "message": "Another GPU job (reflection / wander / encounter / "
                                    "outreach / synthesis) is in progress — try again "
                                    "once it finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "checkin_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return

    # Forge the period: default to the configured threshold, overridable via sim_hours.
    try:
        sim_hours = float(msg.get("sim_hours")) if msg.get("sim_hours") is not None \
            else silence_threshold_hours()
    except Exception:
        sim_hours = silence_threshold_hours()

    # Whose check-in this is. Defaults to the active session's speaker — the person the
    # operator is looking at — so pressing the button with no argument behaves as it did,
    # only now scoped to them instead of silently spanning everyone.
    target = str(msg.get("user") or "").strip() or _target_user()

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "checkin_chunk", "text": delta}), loop)

    def _on_stage(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "checkin_stage", **info}), loop)

    def _on_prompt(label: str, info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "checkin_prompt", "pass": label, **info}), loop)

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_checkin_decision_blocking(
                user=target, forced_hours=sim_hours, on_chunk=_on_chunk,
                on_stage=_on_stage, on_prompt=_on_prompt, bypass_cooldown=True),
        )
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    # Reset the idle clock so an autonomous idle job doesn't fire on top of a manual run
    # (mirrors outreach/synthesis).
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "checkin_done"}
    payload.update(result)
    await _send(ws, payload)


# ── self-test ──────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise the recap sourcing: stored gist vs generated, the excerpt, and the
    labelled render. GPU-free (the summary pass is a stub). Run: ``python -m core.checkin``.
    """
    import tempfile

    def check(label, got, want):
        assert got == want, f"{label}: got {got!r}, want {want!r}"

    # ── _gist_excerpt ─────────────────────────────────────────────────────────
    check("empty gist stays empty", _gist_excerpt(""), "")
    short = "We talked about crocodiles. He was tired of it by the end."
    check("a short gist passes through whole", _gist_excerpt(short), short)

    p1 = "First paragraph. " * 3
    p2 = "Second paragraph. " * 30
    two = p1.strip() + "\n\n" + p2.strip()
    got = _gist_excerpt(two, cap=120)
    check("whole paragraphs are kept, not cut across", got, p1.strip() + " …")
    assert len(_gist_excerpt(p2, cap=200)) <= 202, "an oversized paragraph is capped"
    assert _gist_excerpt(p2, cap=200).rstrip(" …").endswith("."), \
        "an oversized paragraph is cut at a sentence end"
    assert _gist_excerpt(two, cap=120).endswith("…"), "an elision is marked"

    # No sentence end early enough to cut at → hard trim rather than an empty excerpt.
    runon = "x" * 400 + ". tail"
    assert len(_gist_excerpt(runon, cap=100)) <= 102

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        chats.mkdir()
        prompts = root / "prompts"
        prompts.mkdir()
        configure(get_rag=lambda: None, make_sync_reflect_generate=lambda rag: None,
                  load_server_config=lambda: {}, chats_dir=chats, prompts_dir=prompts)

        def write_chat(name, **kw):
            kw.setdefault("user", "Artemy")
            kw.setdefault("exchanges", [{"user_prompt": "hi", "assistant_response": "hello"}])
            (chats / name).write_text(json.dumps(kw), encoding="utf-8")
            return name, json.loads((chats / name).read_text(encoding="utf-8"))

        def write_summary(name, text):
            """The current location: the chat's own `<stem>.summary.json`."""
            stem = name[: -len(".json")]
            (chats / f"{stem}.summary.json").write_text(
                json.dumps({"source_session": name, "text": text, "run_id": "r"}),
                encoding="utf-8")

        def write_legacy_summary(name, text):
            """The pre-split location: a key inside the state sidecar. Still read (the
            corpus predates the split), so it has to keep working here."""
            stem = name[: -len(".json")]
            (chats / f"{stem}.state.json").write_text(
                json.dumps({"source_session": name,
                            "consolidation_summary": {"text": text, "run_id": "r"}}),
                encoding="utf-8")

        reflected = write_chat("20260801_120000.json")
        legacy = write_chat("20260801_180000.json")
        fresh = write_chat("20260802_120000.json")
        dirty = write_chat("20260803_120000.json")
        gist = ("He came back to the balcony evenings again, and it landed differently "
                "this time. Left it there without deciding anything.")
        legacy_gist = ("We went round the same question about naming things and got "
                       "nowhere either of us was happy with.")
        write_summary(reflected[0], gist)
        write_legacy_summary(legacy[0], legacy_gist)
        # A summary that sanitizes to nothing must read as absent, not as a recap.
        write_summary(dirty[0], "## WEIGHTS\n- [fact] artemyvo: something")

        check("a stored gist is read back", _stored_gist(reflected[0]), gist)
        check("a pre-split gist is still read", _stored_gist(legacy[0]), legacy_gist)
        check("an unreflected chat has none", _stored_gist(fresh[0]), "")
        check("an unsalvageable summary reads as none", _stored_gist(dirty[0]), "")

        calls: list[str] = []

        def fake_generate(content, system_prompt, **kw):
            calls.append(content)
            return "<think>weighing it</think>\nRECAP: A short generated recap."

        # The window arrives newest-first, as `_recent_chats` returns it.
        window = [dirty, fresh, legacy, reflected]
        _SUMMARY_CACHE.clear()
        stages: list[dict] = []
        digests = _summarize_recent(window, fake_generate, on_stage=stages.append)
        check("every chat is represented", len(digests), 4)
        check("sources are reported", [d["source"] for d in digests],
              ["generated", "generated", "reflection", "reflection"])
        check("only the unreflected chats are generated", len(calls), 2)
        check("the gist is used verbatim", digests[3]["recap"], gist)
        check("a generated recap is parsed", digests[0]["recap"], "A short generated recap.")

        recapped = [s for s in stages if s.get("stage") == "recapped"]
        check("every recap surfaces its source", [s["source"] for s in recapped],
              ["generated", "generated", "reflection", "reflection"])
        check("a gist is not reported as cached", [s["cached"] for s in recapped],
              [False, False, False, False])

        # Second run: the generated recaps come from the cache, the gist is re-read (free).
        calls.clear()
        stages.clear()
        _summarize_recent(window, fake_generate, on_stage=stages.append)
        check("nothing is regenerated", len(calls), 0)
        recapped = [s for s in stages if s.get("stage") == "recapped"]
        check("the cache flag stays scoped to generated recaps",
              [s["cached"] for s in recapped], [True, True, False, False])

        # ── render ────────────────────────────────────────────────────────────
        block = _render_digests(digests)
        assert "from your own reflection on it" in block, "a gist is labelled as one"
        check("every gist line is labelled, and only those",
              block.count("from your own reflection on it"), 2)
        assert gist.split(".")[0] in block and "A short generated recap." in block
        # Oldest → newest: the freshest conversation is what she reads last.
        assert block.index(gist.split(".")[0]) < block.index("A short generated recap."), \
            "the window reads chronologically"

    # ── per-person scoping ────────────────────────────────────────────────────
    # The corpus below is the case that motivated it: two people, one of whom is still
    # talking while the other has gone quiet, plus openers of Ava's own to each.
    import os
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        prompts = Path(td) / "prompts"
        prompts.mkdir()
        configure(get_rag=lambda: None, make_sync_reflect_generate=lambda rag: None,
                  load_server_config=lambda: {}, chats_dir=chats, prompts_dir=prompts)

        now = _time.time()

        def chat(*, user, ava=False, turns=1, age_h=1.0, interlocutor=None):
            """One transcript, aged *age_h* hours ago.

            The stem and the mtime are set from the same instant, as ``ChatLogger`` leaves
            them on a real box: the filename stem is when the session started (what
            ``_sent_at`` reads) and the mtime is its last turn (what the silence clock
            reads), and they only diverge for a session that later grew."""
            ts = now - age_h * 3600.0
            stem = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
            ex = []
            if ava:
                ex.append({"speaker": _INITIATIVE_SPEAKER, "user_prompt": "impulse",
                           "assistant_response": f"unprompted note for {user}"})
            for i in range(turns):
                ex.append({"speaker": user, "user_prompt": f"q{i}",
                           "assistant_response": f"a{i}"})
            doc = {"timestamp": stem, "user": user, "exchanges": ex}
            if ava:
                doc["initiated_by"] = "ava"
            if interlocutor:
                doc["interlocutor"] = interlocutor
            p = chats / f"{stem}.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            os.utime(p, (ts, ts))
            return p

        # Dana spoke 30 minutes ago; Artemy 9 hours ago and not since.
        artemy_chat = chat(user="Artemy", turns=3, age_h=9.0)
        chat(user="Dana", turns=2, age_h=0.5)
        # Two unanswered openers to Artemy since he last spoke, one to Dana before she did.
        chat(user="Artemy", ava=True, turns=0, age_h=4.0)
        chat(user="artemy voikhansky", ava=True, turns=0, age_h=2.0)
        chat(user="Dana", ava=True, turns=0, age_h=1.5)
        # A peer transcript belongs to nobody's thread.
        chat(user="Dana", turns=4, age_h=0.2, interlocutor="ai")

        people = known_users()
        check("both people are found", [p["key"] for p in people], ["dana", "artemy"])
        # The freshest transcript carrying a real Artemy turn spells him "Artemy" — his
        # unanswered openers, spelled differently, are not turns of his.
        check("the freshest spelling is displayed", people[1]["display"], "Artemy")

        # The clock is the whole point: unscoped, Dana's reply hides Artemy's silence.
        assert silence_hours("") < 1.0, "unscoped, the box looks freshly active"
        assert 8.5 < silence_hours("artemy") < 9.5, "Artemy has been quiet ~9h"
        assert silence_hours("dana") < 1.0, "Dana has not"

        # The window is one person's conversations, not a mixture.
        check("Artemy's window is his own", [n for n, _ in _recent_chats(5, 1, "artemy")],
              [artemy_chat.name])
        check("Dana's window is hers", [d.get("user")
                                        for _, d in _recent_chats(5, 1, "dana")], ["Dana"])
        check("unscoped still sees both", len(_recent_chats(5, 1, "")), 2)

        # Standing openers likewise — hers are not something he is failing to answer.
        check("both of Artemy's unanswered openers count",
              len(_standing_openers(key="artemy")), 2)
        check("Dana's opener predates her last turn",
              len(_standing_openers(key="dana")), 0)

        # And the reach-out gate agrees with all of it (same key function, same corpus).
        reachout_gate.configure(chats)
        check("Artemy's streak is his own", reachout_gate.unanswered_streak("Artemy"), 2)
        check("Dana is not ignoring her", reachout_gate.unanswered_streak("Dana"), 0)

        # An opener the stale-reach-out sweep deleted must keep appearing here: it is what
        # tells the pass this is its second message and not its first. Simulate the sweep
        # (tombstone, then unlink) on the older of Artemy's two.
        reachout_gate.configure(chats, expired_log=root / "expired.jsonl")
        gone = sorted(d["name"] for d in _standing_openers(key="artemy"))[0]
        gone_ts = _sent_at(gone, {})
        reachout_gate.record_expired(gone[:-5], user="Artemy", sent_at=gone_ts,
                                     opener="the message she sent and no one answered")
        (chats / gone).unlink()
        standing = _standing_openers(key="artemy")
        check("the deleted opener is still standing", len(standing), 2)
        check("…quoted from its tombstone, oldest first",
              standing[0]["opener"], "the message she sent and no one answered")
        check("…and marked as expired", standing[0].get("expired"), True)
        check("a tombstone is scoped like everything else",
              len(_standing_openers(key="dana")), 0)

        # A corpus that attributes nothing has no scope to apply.
        for p in chats.glob("*.json"):
            p.unlink()
        chat(user="", turns=2, age_h=1.0)
        check("an unattributed corpus names nobody", known_users(), [])
        check("…and is still measurable unscoped", len(_recent_chats(5, 1, "")), 1)

    # ── the sweep's aggregation ───────────────────────────────────────────────
    # Exercised against a stubbed decision so the walk, the max_users cap, the cheap-skip
    # accounting and the folded skip reason are pinned without a GPU.
    real_decide = globals()["_decide_for_user"]
    real_known = globals()["known_users"]
    try:
        _runtime.model = object()          # the sweep refuses without one
        scripted: dict[str, dict] = {}
        seen: list[str] = []

        def stub(*, user=None, **kw):
            seen.append(str(user))
            return dict(scripted.get(str(user), {"skipped": "declined"}))

        globals()["_decide_for_user"] = stub
        globals()["known_users"] = lambda: [
            {"key": "dana", "display": "Dana", "last_turn_ts": 300.0},
            {"key": "artemy", "display": "Artemy", "last_turn_ts": 200.0},
            {"key": "sam", "display": "Sam", "last_turn_ts": 100.0},
        ]

        # Everyone declines: nothing sent, and the reason that cost a generation wins.
        seen.clear()
        out = run_checkin_sweep_blocking(max_users=5)
        check("each person is decided separately", seen, ["Dana", "Artemy", "Sam"])
        check("most recently active first", seen[0], "Dana")
        check("nothing composed", out["composed"], 0)
        check("the fold names the expensive reason", out["skipped"], "declined")

        # A cheap skip costs no slot: with a cap of 1, the two people the silence gate
        # rejects for free still let the third through to a decision.
        scripted = {"Dana": {"skipped": "insufficient_silence"},
                    "Artemy": {"skipped": "insufficient_silence"},
                    "Sam": {"composed": True, "session": "s.json", "opener": "hey",
                            "silence_hours": 7.0, "user": "Sam"}}
        seen.clear()
        out = run_checkin_sweep_blocking(max_users=1)
        check("free skips do not spend the cap", seen, ["Dana", "Artemy", "Sam"])
        check("one generation ran", out["ran"], 1)
        check("Sam was written to", out["composed"], 1)
        check("the send is lifted for the journal", out["user"], "Sam")

        # …but a person who reaches a generation does spend it, and the rest defer.
        scripted = {"Dana": {"skipped": "declined"},
                    "Artemy": {"composed": True, "session": "a.json", "opener": "hi",
                               "silence_hours": 9.0, "user": "Artemy"}}
        seen.clear()
        out = run_checkin_sweep_blocking(max_users=1)
        check("the cap stops after the first real decision", seen, ["Dana"])
        check("the rest are deferred, not decided",
              [r.get("skipped") for r in out["results"]],
              ["declined", "deferred", "deferred"])
        check("a wholly deferred remainder still reports the decline",
              out["skipped"], "declined")

        # A corpus naming nobody degrades to one unscoped pass.
        globals()["known_users"] = lambda: []
        scripted = {"None": {"skipped": "declined"}}
        seen.clear()
        out = run_checkin_sweep_blocking()
        check("unscoped fallback runs once", seen, ["None"])
        check("…and says so", out["users"], 0)
    finally:
        globals()["_decide_for_user"] = real_decide
        globals()["known_users"] = real_known
        _runtime.model = None

    print("checkin selftest: OK")


if __name__ == "__main__":
    _selftest()
