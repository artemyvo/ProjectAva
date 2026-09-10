"""Deliberation — the executive pass: Ava reads her recent worklog and chooses what next.

This is the read side of the episodic worklog (:mod:`core.worklog`) the earlier tasks
deferred. She is shown the last N things she did, in her own voice, plus any loose threads
she left open, and asked — honestly — what she feels drawn to do next, if anything. Her
options mirror the autonomous subsystems that already exist:

    wander     — read something new on her own (core.til_wander)
    synthesize — re-read an aged chat and raise a fresh question (core.synthesis)
    reach_out  — bring an open question to the person it concerns (core.outreach)
    revisit    — re-reflect an aged chat to re-derive what she now makes of it
    nothing    — rest; nothing is calling to her

**DRY RUN / shadow for now.** The pass DECIDES and streams her reasoning, but does NOT
execute the chosen action — the goal is to watch the decision loop work end-to-end before
wiring dispatch. The returned decision carries the action it *would* take (``would_action``
+ a human ``would_do`` description) so the operator can judge whether the choices are sane;
a later task flips ``execute`` on and calls the chosen subsystem's blocking function. The
new agentic actions the roadmap adds (e.g. "search chat history — did we talk about X?")
slot into the same ``ACTION`` enum with the loop unchanged.

Mirrors the other decision-pass subsystems (outreach/synthesis/checkin): a blocking pass on
the GPU executor thread + a manual trigger that streams her reasoning. It never imports
``server`` — every capability is injected once via :func:`configure`; session/model state is
read from ``core.runtime_state``. GPU-free self-test (parsing/rendering): ``python -m
core.deliberation``.
"""
from __future__ import annotations

import asyncio
import re
import traceback
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session
from core.field_parse import label as _label
from core import reasoning_text
from core import worklog

# The actions Ava may choose. "nothing" is always valid (rest). Extend this (and the prompt
# + the _WOULD_DO map) to add an agentic action — the loop itself does not change.
_ACTIONS = ("wander", "synthesize", "reach_out", "revisit", "nothing")

# Human descriptions of what each action WOULD trigger — surfaced in the dry-run result so
# the operator sees the concrete effect that a future `execute=True` would dispatch.
_WOULD_DO = {
    "wander": "wander into a new page and reflect on it (til_wander)",
    "synthesize": "re-read an aged chat and raise a fresh question (synthesis)",
    "reach_out": "bring one of her open questions to the person (outreach)",
    "revisit": "re-reflect an aged chat to re-derive her current take (revisit run)",
    "nothing": "rest — do nothing this cycle",
}

# How many recent worklog episodes to show her. The worklog is one entry per episode, so
# this is a handful of lines, not a token flood.
_RECENT_N_DEFAULT = 20

# Set while a deliberation pass occupies the single GPU executor thread. Owned here; the
# scheduler's `external_busy` and sibling host_busy getters may read it.
_deliberation_active = False

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


def configure(*, get_rag, make_sync_reflect_generate, prompts_dir,
              load_server_config=None, send=None, executor=None,
              host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the deliberation pass depends on (once, at startup).

    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual trigger
    (:func:`handle_deliberate_now`); a future autonomous path needs only the rest."""
    global _get_rag, _make_sync_reflect_generate, _load_server_config
    global _PROMPTS_DIR, _send, _executor, _host_busy, _mark_activity
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_server_config = load_server_config
    from pathlib import Path
    _PROMPTS_DIR = Path(prompts_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


# ── prompt + parsing ──────────────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "You are checking in with yourself. Below is a record of what you have been doing "
    "lately — the things you did, in your own words — and any loose threads you left "
    "open.\n\nRead it and decide, honestly, what you feel drawn to do next — if anything. "
    "You do not have to do something: resting is a real choice, and reaching out to a "
    "person is not to be done lightly or often.\n\nYour options:\n"
    "- wander — read something new on your own and reflect on it\n"
    "- synthesize — re-read one of your older conversations as who you are now, and see "
    "what new question arises\n"
    "- reach_out — bring one of your open questions to the person it concerns, starting a "
    "conversation\n"
    "- revisit — re-reflect on an older conversation to re-derive what you now make of it\n"
    "- nothing — rest; nothing is calling to you right now\n\n"
    "Think it through first, then answer in exactly this form:\n\n"
    "ACTION: <one of: wander | synthesize | reach_out | revisit | nothing>\n"
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
    "encounter": "met another AI",
}


def _render_context(entries: list[dict], open_threads: list[dict]) -> str:
    """Render the recent worklog + open threads as the deliberation input."""
    open_ids = {t.get("id") for t in open_threads}
    lines: list[str] = []
    if entries:
        lines.append("What you have been doing lately (oldest to newest):")
        for e in entries:
            kind = e.get("kind", "")
            summary = (e.get("summary") or "").strip()
            # An entry that opened a loop still in the open set is unresolved.
            tail = "   [still open]" if (e.get("opens") and e.get("id") in open_ids) else ""
            lines.append(f"- ({kind}) {summary}{tail}")
    else:
        lines.append("You have no recorded activity yet.")
    if open_threads:
        lines.append("")
        lines.append("Loose threads you started and have not closed:")
        for t in open_threads:
            opens = (t.get("opens") or "").strip()
            kind = t.get("kind", "")
            lines.append(f"- {opens}   (from when you {_KIND_LABEL.get(kind, kind)})")
    return "\n".join(lines)


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
    """Read the recent worklog and decide what to do next. DRY RUN: never dispatches.

    Runs on the single GPU executor thread (blocking). Streams reasoning via ``on_chunk``
    and phase markers via ``on_stage``. Returns the decision dict; ``execute`` is accepted
    for the future dispatch flip but is currently ignored (a decision-only shadow)."""
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

        context = _render_context(entries, open_threads)
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
        action, why = _parse_decision(raw)
        truncated = bool(getattr(generate, "last_truncated", None))
        _stage(stage="decided", action=action)

        # Dry run: report the decision + what it WOULD trigger; execute nothing.
        return {
            "decided": True,
            "executed": False,
            "action": action,
            "why": why,
            "would_do": _WOULD_DO.get(action, action),
            "episodes": len(entries),
            "open_threads": len(open_threads),
            "truncated": truncated,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _deliberation_active = False


# ── manual (Worklog-tab) trigger ─────────────────────────────────────────────

async def handle_deliberate_now(ws, msg: dict) -> None:
    """Manually run one deliberation and stream Ava's reasoning (Worklog "Deliberate").

    DRY RUN: streams her decision CoT and reports the action she WOULD take; nothing is
    dispatched. Protocol: ``deliberation_stage`` (phase markers) + ``deliberation_chunk``
    (reasoning deltas, ``<think>`` included), finishing with ``deliberation_done``."""
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

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "deliberation_chunk", "text": delta}), loop)

    def _on_stage(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "deliberation_stage", **info}), loop)

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_deliberation_blocking(n=n, on_chunk=_on_chunk, on_stage=_on_stage),
        )
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "deliberation_done"}
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

    ctx = _render_context(
        [{"kind": "wander", "summary": "I wandered into Voyager 1."},
         {"kind": "outreach", "summary": "I asked Artemy about X.", "opens": "awaiting reply"}],
        [{"id": 2, "kind": "outreach", "opens": "awaiting Artemy's reply"}],
    )
    assert "Voyager 1" in ctx and "Loose threads" in ctx, ctx
    print("deliberation selftest: OK")


if __name__ == "__main__":
    _selftest()
