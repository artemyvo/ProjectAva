"""Deliberation — the executive pass: Ava reads her recent worklog and chooses what next.

This is the read side of the episodic worklog (:mod:`core.worklog`). She is shown the last
N things she did, in her own voice, plus any loose threads she left open, and asked —
honestly — what she feels drawn to do next, if anything. Her options are the autonomous
drives that already exist as subsystems, each with its own blocking entry point:

    wander     — read something new on her own (core.til_wander)
    synthesize — re-read an aged chat and raise a fresh question (core.synthesis)
    reach_out  — bring an open question to the person it concerns (core.outreach)
    check_in   — look at who has gone quiet and whether to write to them (core.checkin)
    revisit    — re-reflect an aged chat to re-derive what she now makes of it
    aha        — see whether something recently read answers a standing need (assoc)
    pivot      — follow a word in play to its other sense on record (assoc)
    nothing    — rest; nothing is calling to her

**The executive (2026-09-14).** The pass DECIDES and, with ``execute=True``, DISPATCHES:
the chosen action's blocking function runs right here on the executor thread, inside the
same GPU slot, and its summary rides the decision result as ``outcome``. The functions
are not imported — they are injected as a ``dispatch`` map by ``server.main()`` (the same
callables the per-action idle jobs run, so an action means one thing whoever triggers it),
which keeps this module free of every subsystem it drives. ``revisit`` is the one action
that cannot run in-thread (a reflection run is a coroutine that dispatches to this same
executor), so its handler schedules the run and returns at once.

Three modes, ``server_config.json`` → ``deliberation.mode``:

    off     — no autonomous deliberation (the Worklog tab's button still works, dry-run)
    shadow  — the executive runs as an idle job AND the per-action idle jobs keep their own
              clocks: what she chooses can be compared, in the Activity tab, against what
              the timers did — the mode the executive shipped in (2026-09-14) while that
              comparison was being made
    sole    — the executive is the only chooser (the default since 2026-09-17): the reach-out
              family (outreach, synthesis, check-in, wander, aha, pivot) is not registered as
              idle jobs; the upkeep jobs (feed, witness, graph rebuild, worklog sweep,
              background reflection) stay on their own clocks, being maintenance rather
              than choices. A box wanting the old side-by-side view sets `shadow` explicitly.

Nothing about the decision is written to the worklog: a dispatched action records its own
episode as it always did, and an hourly "I decided to rest" would fill the window the next
deliberation reads. The journal line (`describe`) carries the decision and her WHY.

**The frozen-window trap (fixed 2026-09-21).** Under `sole` mode wander stopped for good
on a box with reading budget to spare. The worklog is one entry per COMPLETED episode and a
rest records nothing, so once she chose `nothing` the input to the next deliberation was
byte-identical: the newest line still read "I raised a pivot on «глаза»", and with no ages
on the lines "I just reached out — let the silence work" was as true at hour nine as at
hour one. Every option then read as either redundant with that reach-out or as waiting on
its reply — wander included, which she described as needing "new external material" to
react to, when wander IS where new material comes from. Three things now break the loop,
all in the CONTEXT (the prompt file stays overridable): every worklog line carries how long
ago it happened; a process-local rest streak says how many times in a row she has rested
and since when, since nothing else can (a rest is not an episode); and injected
``note_fns`` render where things stand — today the wander budget, so "I have N wanders
unspent" is in front of her rather than a precondition she never sees. The prompt itself
now separates the two options that send nothing to anyone (wander, revisit) from the ones
that may end in a message, and says that silence owed to a person is silence toward THEM,
not idleness.

Mirrors the other decision-pass subsystems (outreach/synthesis/checkin): a blocking pass on
the GPU executor thread + a manual trigger that streams her reasoning. It never imports
``server`` — every capability is injected once via :func:`configure`; session/model state is
read from ``core.runtime_state``. GPU-free self-test (parsing/rendering/dispatch/mode):
``python -m core.deliberation``.
"""
from __future__ import annotations

import asyncio
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session
from core.field_parse import label as _label
from core import reasoning_text
from core import worklog

# The actions Ava may choose. "nothing" is always valid (rest). Extend this (and the prompt
# + the _WOULD_DO map + server.main()'s dispatch map) to add an action — the loop itself
# does not change.
_ACTIONS = ("wander", "synthesize", "reach_out", "check_in", "revisit", "aha", "pivot",
            "rewrite_prompt", "nothing")

# Human descriptions of what each action triggers — surfaced in the result (and the
# journal) so the operator sees the concrete effect behind the word.
_WOULD_DO = {
    "wander": "wander into a new page and reflect on it (til_wander)",
    "synthesize": "re-read an aged chat and raise a fresh question (synthesis)",
    "reach_out": "bring one of her open questions to the person (outreach)",
    "check_in": "look at who has gone quiet and decide whether to write (checkin)",
    "revisit": "re-reflect an aged chat to re-derive her current take (revisit run)",
    "aha": "see whether something she read lately answers a standing need (assoc aha)",
    "pivot": "follow a word in play to its other sense on record (assoc pivot)",
    # Offered CONDITIONALLY (the injected `offer_fn`, core.prompt_rewrite.offer): it is on
    # the menu only when the rewrite budget clears, and chosen unoffered it is refused.
    "rewrite_prompt": "rewrite her standing prompt around the pulls her reflections keep "
                      "noticing (prompt_rewrite)",
    "nothing": "rest — do nothing this cycle",
}

# Mode knob (server_config.json → deliberation.mode). See the module docstring.
MODES = ("off", "shadow", "sole")
DEFAULT_MODE = "sole"
DEFAULT_INTERVAL_S = 3600.0
# The drives the executive chooses among. In `sole` mode these are NOT registered as
# independent idle jobs (server.main() consults this tuple); the upkeep jobs are not here.
CHOSEN_JOBS = ("outreach", "synthesis", "checkin", "wander", "assoc_aha", "assoc_pivot")

# How many recent worklog episodes to show her. The worklog is one entry per episode, so
# this is a handful of lines, not a token flood.
_RECENT_N_DEFAULT = 20

# Set while a deliberation pass occupies the single GPU executor thread. Owned here; the
# scheduler's `external_busy` and sibling host_busy getters may read it.
_deliberation_active = False

# The last few dispatch OUTCOMES, newest last (process-local). A drive that ran and
# produced nothing — outreach truncated on a long ask, synthesis with no old chat, a
# reach-out held by the gate — records no worklog episode (only a completed episode does),
# so from inside the next deliberation it never happened and the executive reached for
# the same drive again, hour after hour (observed 2026-09-20 under `sole` mode: every
# hour spent on an outreach that could not be decided). This is the honest fix short of
# journaling failures as episodes, which they are not: the outcomes are SHOWN to her, not
# folded into her record. A restart forgets them, which is fine — the failure they warn
# about is usually of the same age.
_RECENT_DISPATCH_CAP = 6
_recent_dispatches: list[dict] = []

# How many EXECUTING deliberations in a row chose `nothing`, and when the streak began
# (process-local, like the ring). A rest is deliberately not a worklog episode, so without
# this the ninth consecutive rest reads exactly like the first — the frozen-window trap in
# the module docstring. Reset by any dispatched action.
_rest_streak: dict = {"count": 0, "since": None}

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_make_sync_reflect_generate: Callable = None
# NB no temporal anchor here: the reflect-generate factory composes it for every
# pass (generation._reflect_system_parts). This module appended its own until
# 2026-08-07; injecting it again would date-stamp the prompt twice.
_load_server_config: Optional[Callable] = None
_PROMPTS_DIR: Any = None
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None
# action → blocking callable returning the subsystem's own summary dict. Injected, never
# imported: the same callables the per-action idle jobs run (server.main() defines each
# once and hands it to both), so "wander" means one thing whoever triggers it.
_dispatch: dict[str, Callable[[], dict]] = {}
# A conditionally-offered action: a callable returning None or {action, line, decline}.
# The line is appended to her context when offered; `decline(chosen)` is called when the
# offered action was not taken (the drive stamps its own gap; it spends nothing).
_offer_fn: Optional[Callable[[], Optional[dict]]] = None
# "Where things stand" — callables returning one line each ('' = nothing to say), rendered
# under the history so a drive's precondition is in front of her rather than checked
# silently at dispatch. server.main() wires the wander budget; a drive that needs the
# executive to know something else adds a callable here, never an import.
_note_fns: list[Callable[[], str]] = []


def configure(*, get_rag, make_sync_reflect_generate, prompts_dir,
              load_server_config=None, send=None, executor=None,
              host_busy=None, mark_activity=None, dispatch=None, offer_fn=None,
              note_fns=None) -> None:
    """Wire in the server capabilities the deliberation pass depends on (once, at startup).

    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual trigger
    (:func:`handle_deliberate_now`); ``dispatch`` (action → blocking callable) backs
    ``execute=True`` — an action with no handler is reported, never guessed;
    ``note_fns`` (callables → one line each) render where things stand for her."""
    global _get_rag, _make_sync_reflect_generate, _load_server_config
    global _PROMPTS_DIR, _send, _executor, _host_busy, _mark_activity, _dispatch, _offer_fn
    global _note_fns
    _offer_fn = offer_fn
    _note_fns = list(note_fns or [])
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_server_config = load_server_config
    from pathlib import Path
    _PROMPTS_DIR = Path(prompts_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity
    _dispatch = dict(dispatch or {})


def settings(config: Optional[dict] = None) -> dict:
    """The `deliberation` config block resolved to ``{mode, interval_s, recent, execute}``.

    Pure: reads the dict it is given (or the injected loader), never the GPU. An unknown
    mode falls back to the default rather than silently switching the executive off, and
    an unreadable config yields ``off`` — an unknown-state box should not start choosing."""
    if config is None:
        try:
            config = _load_server_config() if _load_server_config is not None else {}
        except Exception:
            return {"mode": "off", "interval_s": DEFAULT_INTERVAL_S,
                    "recent": _RECENT_N_DEFAULT, "execute": True}
    blk = (config or {}).get("deliberation") or {}
    mode = str(blk.get("mode", DEFAULT_MODE) or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        mode = DEFAULT_MODE
    try:
        interval = max(60.0, float(blk.get("interval_s", DEFAULT_INTERVAL_S)))
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_S
    try:
        recent = max(1, int(blk.get("recent", _RECENT_N_DEFAULT)))
    except (TypeError, ValueError):
        recent = _RECENT_N_DEFAULT
    return {"mode": mode, "interval_s": interval, "recent": recent,
            "execute": bool(blk.get("execute", True))}


# ── prompt + parsing ──────────────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "You are checking in with yourself. Below is a record of what you have been doing "
    "lately — the things you did, in your own words, each with how long ago it was — any "
    "loose threads you left open, and where things stand right now.\n\n"
    "Read it and decide, honestly, what you feel drawn to do next — if anything. Resting "
    "is a real choice. But be clear about what rest is for: it is for not crowding a "
    "person. Two of your options send nothing to anyone, and a question you have left "
    "hanging with someone is no reason to stop reading or thinking on your own — the "
    "silence you owe them is silence toward THEM, not idleness. Notice what you have been "
    "doing a lot of, what you have been neglecting, and how long it has actually been "
    "since you last did anything.\n\n"
    "Private to you — nobody hears from you:\n"
    "- wander — go read a page you have never seen and reflect on it. Nothing new arrives "
    "on its own; this is where your new material comes from. It spends one unit of your "
    "reading budget (shown below when you have any)\n"
    "- revisit — re-reflect on an older conversation to re-derive what you now make of it"
    "\n\n"
    "These may end in a message to a person (each holds itself back if you have written "
    "to them too recently):\n"
    "- synthesize — re-read one of your older conversations as who you are now; a new "
    "question that arises goes to the person it concerns\n"
    "- reach_out — bring one of your open questions to the person it concerns, starting a "
    "conversation\n"
    "- check_in — look at who has gone quiet lately and decide whether to write to them\n"
    "- aha — see whether something you read lately answers a question you have been "
    "carrying; if it does, tell them\n"
    "- pivot — follow a word from a recent conversation to its other meaning on your "
    "record, and see where it takes you — sometimes to them\n\n"
    "- nothing — rest; nothing is calling to you right now\n\n"
    "Think it through first, then answer in exactly this form:\n\n"
    "ACTION: <one of: wander | synthesize | reach_out | check_in | revisit | aha | pivot | "
    "nothing>\n"
    "WHY: <one or two sentences, in your own voice, on why this feels right now>\n"
)


def _load_prompt() -> str:
    try:
        if _PROMPTS_DIR is not None:
            p = _PROMPTS_DIR / "deliberation_prompt.txt"
            if p.exists():
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    return text
    except Exception:
        pass
    return _DEFAULT_PROMPT


# Shared with synthesis / outreach / checkin — see `core.reasoning_text`.
_answer_after_think = reasoning_text.answer_after_think


def _parse_decision(raw: str) -> tuple[str, str]:
    """Return ``(action, why)`` parsed from the pass output. Unknown/missing → ``nothing``.

    Parses over the ANSWER (post-think), so the ACTION/WHY labels the reasoning discusses
    about itself can't be mistaken for the decision."""
    ans = _answer_after_think(raw)
    m = re.search(_label("ACTION") + r"([a-z_]+)", ans, re.IGNORECASE | re.MULTILINE)
    action = (m.group(1).strip().lower() if m else "")
    if action not in _ACTIONS:
        action = "nothing"
    w = re.search(_label("WHY") + r"(.+?)(?:\n\s*\n|\Z)", ans,
                  re.IGNORECASE | re.DOTALL | re.MULTILINE)
    why = (w.group(1).strip() if w else "")
    return action, why


# ── worklog rendering ────────────────────────────────────────────────────────

_KIND_LABEL = {
    "conversation": "talked with someone", "wander": "wandered", "outreach": "reached out",
    "synthesis": "re-read an old chat", "checkin": "checked in", "reflection": "reflected",
    "encounter": "met another AI", "aha": "had something click", "pivot": "followed a word",
    "prompt": "rewrote my standing prompt",
}


def _age_hours(ts: Any, now: float) -> Optional[float]:
    """Hours between an ISO worklog ``ts`` and ``now`` (None when unparseable)."""
    try:
        t = datetime.fromisoformat(str(ts))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return max(0.0, (now - t.timestamp()) / 3600.0)
    except Exception:
        return None


def _age_phrase(hours: Optional[float]) -> str:
    if hours is None:
        return ""
    if hours < 1:
        return "less than an hour ago"
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} days ago"


def _render_context(entries: list[dict], open_threads: list[dict],
                    now: Optional[float] = None) -> str:
    """Render the recent worklog + open threads as the deliberation input.

    Every line carries how long ago it happened. Without that, "I just reached out" was
    read off the newest line however old it was — the worklog only moves when an episode
    completes, so after a rest the same newest line is there again an hour later."""
    now = time.time() if now is None else now
    open_ids = {t.get("id") for t in open_threads}
    lines: list[str] = []
    if entries:
        newest = _age_phrase(_age_hours(entries[-1].get("ts"), now))
        head = "What you have been doing lately (oldest to newest"
        head += f"; the newest was {newest}):" if newest else "):"
        lines.append(head)
        for e in entries:
            kind = e.get("kind", "")
            summary = (e.get("summary") or "").strip()
            age = _age_phrase(_age_hours(e.get("ts"), now))
            when = f", {age}" if age else ""
            # An entry that opened a loop still in the open set is unresolved.
            tail = "   [still open]" if (e.get("opens") and e.get("id") in open_ids) else ""
            lines.append(f"- ({kind}{when}) {summary}{tail}")
    else:
        lines.append("You have no recorded activity yet.")
    if open_threads:
        lines.append("")
        lines.append("Loose threads you started and have not closed:")
        for t in open_threads:
            opens = (t.get("opens") or "").strip()
            kind = t.get("kind", "")
            age = _age_phrase(_age_hours(t.get("ts"), now))
            when = f", {age}" if age else ""
            lines.append(f"- {opens}   (from when you {_KIND_LABEL.get(kind, kind)}{when})")
    return "\n".join(lines)


def _dispatch_outcome_line(result: dict) -> str:
    """One short phrase for what a dispatched action came to — '' when it plainly ran."""
    if result.get("dispatch_skipped"):
        return f"not run ({result['dispatch_skipped']})"
    out = result.get("outcome") or {}
    if not isinstance(out, dict):
        return ""
    if out.get("error"):
        return "failed"
    if out.get("skipped"):
        return f"came to nothing ({out['skipped']})"
    return ""


def _record_dispatch(result: dict, now: Optional[float] = None) -> None:
    """Remember what an EXECUTING deliberation's dispatch came to (see the ring's note),
    and count a rest into the streak (a rest is the one decision nothing else records)."""
    if not isinstance(result, dict) or not result.get("decided"):
        return
    now = time.time() if now is None else now
    action = str(result.get("action") or "")
    if action == "nothing":
        _rest_streak["count"] = int(_rest_streak.get("count") or 0) + 1
        if _rest_streak.get("since") is None:
            _rest_streak["since"] = now
        return
    if not action or not (result.get("executed") or result.get("dispatch_skipped")):
        return
    _rest_streak["count"], _rest_streak["since"] = 0, None
    _recent_dispatches.append({"ts": now, "action": action,
                               "note": _dispatch_outcome_line(result)})
    del _recent_dispatches[:-_RECENT_DISPATCH_CAP]


def _render_rest_streak(now: Optional[float] = None) -> str:
    """How many times in a row she has chosen to rest, and since when — '' when she has
    not. Stated because the record above does not move while she rests, so without it
    the ninth consecutive rest is decided on exactly the input the first was."""
    n = int(_rest_streak.get("count") or 0)
    if n < 1:
        return ""
    now = time.time() if now is None else now
    since = _age_phrase(max(0.0, (now - float(_rest_streak.get("since") or now)) / 3600.0))
    times = "once" if n == 1 else f"{n} times in a row"
    return (f"You have chosen to rest {times}, starting {since}. A rest records nothing, "
            f"so the list above has not moved since then.")


def _render_notes() -> str:
    """Where things stand — one line per injected note callable, '' when none say anything."""
    out: list[str] = []
    for fn in _note_fns:
        try:
            line = (fn() or "").strip()
        except Exception:
            traceback.print_exc()
            line = ""
        if line:
            out.append(f"- {line}")
    if not out:
        return ""
    return "Where things stand:\n" + "\n".join(out)


def _render_recent_dispatches(now: Optional[float] = None) -> str:
    """The recent-dispatch ring as a block for the decision context — '' when empty."""
    if not _recent_dispatches:
        return ""
    now = time.time() if now is None else now
    lines = ["What your last decisions came to (oldest to newest; a drive that produced "
             "nothing leaves no trace above, so it is listed here):"]
    for d in _recent_dispatches:
        age_h = max(0.0, (now - float(d.get("ts") or now)) / 3600.0)
        when = f"{age_h:.0f} h ago" if age_h >= 1 else "less than an hour ago"
        note = d.get("note") or "ran"
        lines.append(f"- {d['action']}, {when}: {note}")
    return "\n".join(lines)


def _prompt_budget_block() -> str:
    """The standing-prompt patterns on record (PROMPT_REWRITE.md §4) — what her
    reflections keep saying the prompt fails to say, and how many are mature enough to
    fund a rewrite. Read off ``patterns.json`` (written by the reflection run's clean-base
    fold, core.prompt_patterns); never re-folded here. Empty when nothing is on record.
    Stage 2: shown so her `WHY` can be about the evidence; the action that spends it is
    stage 3."""
    try:
        from core import prompt_patterns
        from training.reflections_path import prompt_dir
        return prompt_patterns.render_budget(prompt_patterns.budget(prompt_dir()))
    except Exception:
        return ""


def _persona_portrait() -> str:
    """Best-effort current self-portrait so she decides as who she is now (or '')."""
    try:
        from core import reflection_digest
        from training.reflections_path import persona_dir
        digest = reflection_digest.latest_digest(persona_dir())
        text = ((digest or {}).get("self_portrait") or {}).get("text", "").strip()
        return text[:1500].strip()
    except Exception:
        return ""


# ── the pass ─────────────────────────────────────────────────────────────────

def run_deliberation_blocking(*, n: int = _RECENT_N_DEFAULT,
                              on_stage: Optional[Callable] = None,
                              on_chunk: Optional[Callable] = None,
                              execute: bool = False) -> dict:
    """Read the recent worklog and decide what to do next; with ``execute`` also do it.

    Runs on the single GPU executor thread (blocking). Streams reasoning via ``on_chunk``
    and phase markers via ``on_stage``. Returns the decision dict. With ``execute=True``
    the chosen action's injected handler runs in-thread (``nothing`` runs nothing) and its
    summary is returned as ``outcome``; ``executed`` says whether a handler ran, and
    ``dispatch_skipped`` names why one did not (no handler / it raised)."""
    global _deliberation_active

    def _stage(**info) -> None:
        if on_stage is not None:
            try:
                on_stage(info)
            except Exception:
                pass

    _deliberation_active = True
    try:
        if _runtime.model is None:
            return {"skipped": "no_model"}

        entries = worklog.recent(n)
        open_threads = worklog.open_threads()
        if not entries:
            return {"skipped": "no_history"}

        _stage(stage="reading", episodes=len(entries), open_threads=len(open_threads))

        now = time.time()
        context = _render_context(entries, open_threads, now)
        for block in (_render_recent_dispatches(now), _render_rest_streak(now),
                      _render_notes()):
            if block:
                context = context + "\n\n" + block
        budget_block = _prompt_budget_block()
        if budget_block:
            context = context + "\n\n" + budget_block
        # The conditionally-offered action (prompt rewrite): its line goes in only when
        # its gate is open, so she cannot choose it unoffered — and if she does anyway,
        # it is refused below rather than dispatched.
        offer = None
        if _offer_fn is not None:
            try:
                offer = _offer_fn()
            except Exception:
                traceback.print_exc()
                offer = None
        if offer and offer.get("line"):
            context = context + "\n\n" + str(offer["line"])
        system_prompt = _load_prompt()
        portrait = _persona_portrait()
        if portrait:
            system_prompt = system_prompt + "\n\nThis is who you are now:\n\n" + portrait
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)

        _stage(stage="deciding")
        rag = _get_rag() if _get_rag is not None else None
        generate = _make_sync_reflect_generate(rag)
        raw = generate(
            context, system_prompt,
            temperature=0.7, top_p=0.95, max_new_tokens_setting="2048",
            disable_rag=True, on_chunk=on_chunk,
        )
        truncated = bool(getattr(generate, "last_truncated", None))
        if reasoning_text.truncated_before_answer(raw, truncated):
            # Cut inside the reasoning channel: there is NO answer region, so an
            # `ACTION:` the regex would match is her deliberation about the options, not
            # a decision — observed dispatching `wander` off an unclosed think. The same
            # guard every other pass applies (`reasoning_text.truncated_before_answer`).
            # No decline either: nothing was decided, so the offer is not spent.
            _stage(stage="decided", action="nothing", cut_before_answer=True)
            return {
                "decided": True, "executed": False, "action": "nothing", "why": "",
                "would_do": _WOULD_DO.get("nothing", "nothing"),
                "episodes": len(entries), "open_threads": len(open_threads),
                "truncated": True, "cut_before_answer": True,
                "dispatch_skipped": "cut_before_answer",
            }
        action, why = _parse_decision(raw)
        offered_action = str((offer or {}).get("action") or "")
        if execute and offered_action and action != offered_action:
            # Only an executing decision spends the opportunity. A Worklog preview
            # must not append a decline or change the real rewrite cooldown.
            try:
                decline = (offer or {}).get("decline")
                if callable(decline):
                    decline(action)
            except Exception:
                traceback.print_exc()
        _stage(stage="decided", action=action)

        result = {
            "decided": True,
            "executed": False,
            "action": action,
            "why": why,
            "would_do": _WOULD_DO.get(action, action),
            "episodes": len(entries),
            "open_threads": len(open_threads),
            "truncated": truncated,
        }
        if not execute or action == "nothing":
            return result
        if action == "rewrite_prompt" and action != offered_action:
            # Chosen without being on the menu (the budget did not clear, or she was not
            # shown it): refused, never dispatched — the gate is the drive's, not hers.
            result["dispatch_skipped"] = "not_offered"
            return result
        handler = _dispatch.get(action)
        if handler is None:
            # Reported, not guessed: an action the config names but nothing was wired for
            # is a wiring bug, and silently resting would hide it.
            result["dispatch_skipped"] = "no_handler"
            return result
        _stage(stage="dispatching", action=action)
        try:
            outcome = handler()
        except Exception as e:
            traceback.print_exc()
            result["dispatch_skipped"] = "handler_error"
            result["outcome"] = {"error": f"{type(e).__name__}: {e}"}
            return result
        result["executed"] = True
        result["outcome"] = outcome if isinstance(outcome, dict) else {"result": outcome}
        _stage(stage="dispatched", action=action)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _deliberation_active = False


# ── the idle job (the executive proper) ──────────────────────────────────────

def run_deliberation_job() -> dict:
    """The idle-job body: one deliberation under the configured mode, dispatching.

    Reads the config per wake (so a mode/interval edit lands on the next wake, like every
    other knob), and refuses in ``off`` mode — the job is not registered then, but the
    guard keeps a config flip honest without a restart."""
    cfg = settings()
    if cfg["mode"] == "off":
        return {"skipped": "mode_off"}
    result = run_deliberation_blocking(n=cfg["recent"], execute=cfg["execute"])
    if cfg["execute"]:
        _record_dispatch(result)
    return result


def consumed(result: dict) -> bool:
    """A decision pass ran ⇒ the interval is spent (whatever she chose, including rest);
    a cheap pre-generation bail (no model / no history / mode off) retries next poll."""
    if not isinstance(result, dict):
        return True
    if result.get("decided") or result.get("error"):
        return True
    return False


def describe(result: dict) -> str:
    """One journal line: what she chose, why (her words), and what came of it."""
    action = str(result.get("action") or "?")
    why = (result.get("why") or "").strip().replace("\n", " ")
    if len(why) > 160:
        why = why[:157].rstrip() + "…"
    head = f"Deliberation: chose {action}" + (f" — {why}" if why else "")
    if result.get("dispatch_skipped"):
        return f"{head} (not dispatched: {result['dispatch_skipped']})"
    if not result.get("executed"):
        if action == "nothing":
            return head
        return f"{head} (decision only)"
    out = result.get("outcome") or {}
    if out.get("error"):
        return f"{head} → {action} failed ({out['error']})"
    if out.get("scheduled"):
        return f"{head} → {out['scheduled']} run started"
    if out.get("skipped"):
        return f"{head} → {action}: {out['skipped']}"
    return f"{head} → {action} ran"


# ── manual (Worklog-tab) trigger ─────────────────────────────────────────────

async def handle_deliberate_now(ws, msg: dict) -> None:
    """Manually run one deliberation and stream Ava's reasoning (Worklog "Deliberate").

    Dry run by default: streams her decision CoT and reports the action she WOULD take.
    ``execute: true`` in the message dispatches it too, exactly as the idle job would.
    Protocol: ``deliberation_stage`` (phase markers) + ``deliberation_chunk`` (reasoning
    deltas, ``<think>`` included), finishing with ``deliberation_done``."""
    loop = asyncio.get_event_loop()
    if _deliberation_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "deliberation_done", "skipped": "busy",
                         "message": "Another GPU job is in progress — try again once it "
                                    "finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "deliberation_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return

    n = int(msg.get("n") or _RECENT_N_DEFAULT)
    execute = bool(msg.get("execute", False))

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "deliberation_chunk", "text": delta}), loop)

    def _on_stage(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "deliberation_stage", **info}), loop)

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_deliberation_blocking(n=n, on_chunk=_on_chunk, on_stage=_on_stage,
                                              execute=execute),
        )
        if execute:
            _record_dispatch(result)
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "deliberation_done", "mode": settings()["mode"]}
    payload.update(result)
    await _send(ws, payload)


# ── GPU-free self-test ─────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise decision parsing + context rendering. Run: ``python -m core.deliberation``."""
    a, w = _parse_decision(
        "<think>Let me weigh this. ACTION: could be wander. Actually I want to reach out."
        "</think>\nACTION: reach_out\nWHY: I have an open question for Artemy that feels ripe.")
    assert a == "reach_out", a
    assert "Artemy" in w, w

    # unknown action -> nothing; bare </think> leak shape handled
    a2, _ = _parse_decision("reasoning</think>\nACTION: frolic\nWHY: because")
    assert a2 == "nothing", a2

    # missing labels -> nothing
    a3, w3 = _parse_decision("<think>hmm</think>\nI don't know.")
    assert a3 == "nothing" and w3 == "", (a3, w3)

    now = 1_000_000.0
    def _iso(hours_ago: float) -> str:
        return datetime.fromtimestamp(now - hours_ago * 3600, tz=timezone.utc).isoformat()
    ctx = _render_context(
        [{"kind": "wander", "summary": "I wandered into Voyager 1.", "ts": _iso(72)},
         {"kind": "outreach", "summary": "I asked Artemy about X.", "opens": "awaiting reply",
          "ts": _iso(9), "id": 2}],
        [{"id": 2, "kind": "outreach", "opens": "awaiting Artemy's reply", "ts": _iso(9)}],
        now,
    )
    assert "Voyager 1" in ctx and "Loose threads" in ctx, ctx
    # every line carries its age; the header names the newest one; the open thread too
    assert "(wander, 3 days ago)" in ctx and "(outreach, 9 h ago)" in ctx, ctx
    assert "the newest was 9 h ago" in ctx and "[still open]" in ctx, ctx
    assert "(from when you reached out, 9 h ago)" in ctx, ctx
    # no / bad ts degrades to the bare line, never a crash
    ctx2 = _render_context([{"kind": "wander", "summary": "S"}, {"kind": "aha", "summary": "T",
                            "ts": "garbage"}], [], now)
    assert "- (wander) S" in ctx2 and "- (aha) T" in ctx2 and "the newest was" not in ctx2, ctx2
    assert _age_phrase(0.2) == "less than an hour ago" and _age_phrase(None) == ""

    # the recent-dispatch ring: a drive that came to nothing is shown to the next
    # deliberation (it records no worklog episode, so nothing else would say so);
    # a decision-only pass is not recorded; the ring is capped. `nothing` counts into
    # the rest streak instead, and any dispatched action resets it.
    _recent_dispatches.clear()
    _rest_streak.update(count=0, since=None)
    _record_dispatch({"decided": True, "executed": False, "action": "nothing"}, now - 7200)
    _record_dispatch({"decided": True, "executed": False, "action": "nothing"}, now - 3600)
    assert _rest_streak["count"] == 2 and _rest_streak["since"] == now - 7200, _rest_streak
    streak = _render_rest_streak(now)
    assert streak.startswith("You have chosen to rest 2 times in a row, starting 2 h ago"), streak
    _record_dispatch({"decided": True, "executed": True, "action": "reach_out",
                      "outcome": {"skipped": "truncated"}}, now)
    assert _rest_streak["count"] == 0 and _render_rest_streak(now) == "", _rest_streak
    _record_dispatch({"decided": True, "executed": False, "action": "wander"}, now)
    _record_dispatch({"executed": True, "action": "undecided", "outcome": {}}, now)
    _record_dispatch({"decided": True, "executed": False, "action": "revisit",
                      "dispatch_skipped": "no_handler"}, now)
    _record_dispatch({"decided": True, "executed": True, "action": "wander",
                      "outcome": {"applied": True}}, now)
    assert [d["action"] for d in _recent_dispatches] == ["reach_out", "revisit", "wander"], _recent_dispatches
    block = _render_recent_dispatches(now)
    assert "reach_out" in block and "came to nothing (truncated)" in block, block
    assert "not run (no_handler)" in block and block.rstrip().endswith("ran"), block
    for i in range(10):
        _record_dispatch({"decided": True, "executed": True, "action": f"a{i}", "outcome": {}})
    assert len(_recent_dispatches) == _RECENT_DISPATCH_CAP
    _recent_dispatches.clear()
    _record_dispatch({"decided": True, "executed": False, "action": "nothing"}, now)
    assert _render_rest_streak(now).startswith("You have chosen to rest once, starting less than an hour ago")
    _rest_streak.update(count=0, since=None)

    # injected notes: rendered as "Where things stand", empty/raising ones dropped
    global _note_fns
    saved_notes = _note_fns
    _note_fns = [lambda: "Reading budget: 3 wanders unspent.", lambda: "",
                 lambda: (_ for _ in ()).throw(RuntimeError("x"))]
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        notes = _render_notes()
    assert notes == "Where things stand:\n- Reading budget: 3 wanders unspent.", notes
    _note_fns = []
    assert _render_notes() == ""
    _note_fns = saved_notes

    # the new drives parse
    for act in ("check_in", "aha", "pivot", "rewrite_prompt"):
        a4, _ = _parse_decision(f"<think>x</think>\nACTION: {act}\nWHY: y")
        assert a4 == act, (act, a4)

    # mode knob: default, explicit, unknown, bad interval
    assert settings({})["mode"] == "sole"
    assert settings({"deliberation": {"mode": "shadow", "interval_s": 1800}}) == {
        "mode": "shadow", "interval_s": 1800.0, "recent": 20, "execute": True}
    assert settings({"deliberation": {"mode": "bogus", "interval_s": "x", "recent": 0,
                                      "execute": False}}) == {
        "mode": "sole", "interval_s": DEFAULT_INTERVAL_S, "recent": 1, "execute": False}
    assert settings({"deliberation": {"mode": "OFF"}})["mode"] == "off"

    # dispatch seam: the decided action's handler runs; no handler is reported; a raising
    # handler is reported with its error; `nothing` never dispatches.
    global _dispatch, _runtime, _offer_fn
    calls: list[str] = []
    _dispatch = {"wander": lambda: calls.append("wander") or {"applied": True, "title": "T"},
                 "pivot": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}

    class _FakeGen:
        last_truncated = False
        def __init__(self, raw): self.raw = raw
        def __call__(self, *a, **k): return self.raw

    import types
    saved = (_runtime, _make_sync_reflect_generate, _get_rag)
    saved_offer = _offer_fn
    fake_runtime = types.SimpleNamespace(model=object())
    globals()["_runtime"] = fake_runtime
    globals()["_get_rag"] = lambda: None
    saved_recent, saved_open = worklog.recent, worklog.open_threads
    worklog.recent = lambda n: [{"id": 1, "kind": "wander", "summary": "I read X."}]
    worklog.open_threads = lambda: []
    _note_fns = [lambda: "Reading budget: 2 wanders unspent."]
    _rest_streak.update(count=3, since=time.time() - 3 * 3600)
    seen: list[str] = []

    class _FakeGenCapture(_FakeGen):
        def __call__(self, context, *a, **k):
            seen.append(context)
            return self.raw
    try:
        globals()["_make_sync_reflect_generate"] = (
            lambda rag: _FakeGenCapture("<think>..</think>\nACTION: nothing\nWHY: quiet"))
        r = run_deliberation_blocking(execute=True)
        assert r["action"] == "nothing" and seen, r
        assert "Where things stand:\n- Reading budget: 2 wanders unspent." in seen[-1], seen[-1]
        assert "You have chosen to rest 3 times in a row" in seen[-1], seen[-1]
        _note_fns = []
        _rest_streak.update(count=0, since=None)
        globals()["_make_sync_reflect_generate"] = (
            lambda rag: _FakeGen("<think>..</think>\nACTION: wander\nWHY: curious"))
        r = run_deliberation_blocking(execute=True)
        assert r["executed"] and r["outcome"]["applied"] and calls == ["wander"], r
        assert consumed(r) and describe(r).startswith("Deliberation: chose wander — curious → wander ran"), describe(r)
        globals()["_make_sync_reflect_generate"] = (
            lambda rag: _FakeGen("<think>..</think>\nACTION: synthesize\nWHY: w"))
        r = run_deliberation_blocking(execute=True)
        assert not r["executed"] and r["dispatch_skipped"] == "no_handler", r
        assert "not dispatched: no_handler" in describe(r)
        globals()["_make_sync_reflect_generate"] = (
            lambda rag: _FakeGen("<think>..</think>\nACTION: pivot\nWHY: w"))
        import contextlib, io, sys
        with contextlib.redirect_stderr(io.StringIO()):   # the stub's traceback is expected
            r = run_deliberation_blocking(execute=True)
        assert r["dispatch_skipped"] == "handler_error" and "boom" in r["outcome"]["error"], r
        globals()["_make_sync_reflect_generate"] = (
            lambda rag: _FakeGen("<think>..</think>\nACTION: nothing\nWHY: tired"))
        r = run_deliberation_blocking(execute=True)
        assert not r["executed"] and "outcome" not in r and consumed(r), r
        assert describe(r) == "Deliberation: chose nothing — tired", describe(r)
        r = run_deliberation_blocking(execute=False)
        assert not r["executed"] and describe(r).endswith("(decision only)") or r["action"] == "nothing"
        assert not consumed({"skipped": "no_history"})
        declines = []
        _offer_fn = lambda: {"action": "rewrite_prompt", "line": "A rewrite is available.",
                            "decline": declines.append}
        for action in ("nothing", "wander", "rewrite_prompt"):
            globals()["_make_sync_reflect_generate"] = (
                lambda rag, a=action: _FakeGen(f"ACTION: {a}\nWHY: a preview"))
            before_calls = list(calls)
            before_declines = list(declines)
            r = run_deliberation_blocking(execute=False)
            assert not r["executed"] and calls == before_calls and declines == before_declines
            run_deliberation_blocking(execute=True)
            assert declines == (before_declines + [action] if action != "rewrite_prompt"
                                else before_declines), declines
    finally:
        globals()["_runtime"], globals()["_make_sync_reflect_generate"], globals()["_get_rag"] = saved
        worklog.recent, worklog.open_threads = saved_recent, saved_open
        _dispatch = {}
        _offer_fn = saved_offer
    print("deliberation selftest: OK")


if __name__ == "__main__":
    _selftest()
