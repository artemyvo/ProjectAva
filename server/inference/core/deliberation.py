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
              clocks (the default): what she chooses can be compared, in the Activity tab,
              against what the timers did, before anything is retired
    sole    — the executive is the only chooser: the reach-out family (outreach, synthesis,
              check-in, wander, aha, pivot) is not registered as idle jobs; the upkeep jobs
              (feed, witness, graph rebuild, worklog sweep, background reflection) stay on
              their own clocks, being maintenance rather than choices

Nothing about the decision is written to the worklog: a dispatched action records its own
episode as it always did, and an hourly "I decided to rest" would fill the window the next
deliberation reads. The journal line (`describe`) carries the decision and her WHY.

Mirrors the other decision-pass subsystems (outreach/synthesis/checkin): a blocking pass on
the GPU executor thread + a manual trigger that streams her reasoning. It never imports
``server`` — every capability is injected once via :func:`configure`; session/model state is
read from ``core.runtime_state``. GPU-free self-test (parsing/rendering/dispatch/mode):
``python -m core.deliberation``.
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
# + the _WOULD_DO map + server.main()'s dispatch map) to add an action — the loop itself
# does not change.
_ACTIONS = ("wander", "synthesize", "reach_out", "check_in", "revisit", "aha", "pivot",
            "nothing")

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
    "nothing": "rest — do nothing this cycle",
}

# Mode knob (server_config.json → deliberation.mode). See the module docstring.
MODES = ("off", "shadow", "sole")
DEFAULT_MODE = "shadow"
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


def configure(*, get_rag, make_sync_reflect_generate, prompts_dir,
              load_server_config=None, send=None, executor=None,
              host_busy=None, mark_activity=None, dispatch=None) -> None:
    """Wire in the server capabilities the deliberation pass depends on (once, at startup).

    ``send``/``executor``/``host_busy``/``mark_activity`` back the manual trigger
    (:func:`handle_deliberate_now`); ``dispatch`` (action → blocking callable) backs
    ``execute=True`` — an action with no handler is reported, never guessed."""
    global _get_rag, _make_sync_reflect_generate, _load_server_config
    global _PROMPTS_DIR, _send, _executor, _host_busy, _mark_activity, _dispatch
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
    "lately — the things you did, in your own words — and any loose threads you left "
    "open.\n\nRead it and decide, honestly, what you feel drawn to do next — if anything. "
    "You do not have to do something: resting is a real choice, and reaching out to a "
    "person is not to be done lightly or often.\n\nYour options:\n"
    "- wander — read something new on your own and reflect on it\n"
    "- synthesize — re-read one of your older conversations as who you are now, and see "
    "what new question arises\n"
    "- reach_out — bring one of your open questions to the person it concerns, starting a "
    "conversation\n"
    "- check_in — look at who has gone quiet lately and decide whether to write to them\n"
    "- revisit — re-reflect on an older conversation to re-derive what you now make of it\n"
    "- aha — see whether something you read lately answers a question you have been "
    "carrying\n"
    "- pivot — follow a word from a recent conversation to its other meaning on your "
    "record, and see where it takes you\n"
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
    return run_deliberation_blocking(n=cfg["recent"], execute=cfg["execute"])


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

    ctx = _render_context(
        [{"kind": "wander", "summary": "I wandered into Voyager 1."},
         {"kind": "outreach", "summary": "I asked Artemy about X.", "opens": "awaiting reply"}],
        [{"id": 2, "kind": "outreach", "opens": "awaiting Artemy's reply"}],
    )
    assert "Voyager 1" in ctx and "Loose threads" in ctx, ctx

    # the new drives parse
    for act in ("check_in", "aha", "pivot"):
        a4, _ = _parse_decision(f"<think>x</think>\nACTION: {act}\nWHY: y")
        assert a4 == act, (act, a4)

    # mode knob: default, explicit, unknown, bad interval
    assert settings({})["mode"] == "shadow"
    assert settings({"deliberation": {"mode": "sole", "interval_s": 1800}}) == {
        "mode": "sole", "interval_s": 1800.0, "recent": 20, "execute": True}
    assert settings({"deliberation": {"mode": "bogus", "interval_s": "x", "recent": 0,
                                      "execute": False}}) == {
        "mode": "shadow", "interval_s": DEFAULT_INTERVAL_S, "recent": 1, "execute": False}
    assert settings({"deliberation": {"mode": "OFF"}})["mode"] == "off"

    # dispatch seam: the decided action's handler runs; no handler is reported; a raising
    # handler is reported with its error; `nothing` never dispatches.
    global _dispatch, _runtime
    calls: list[str] = []
    _dispatch = {"wander": lambda: calls.append("wander") or {"applied": True, "title": "T"},
                 "pivot": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}

    class _FakeGen:
        last_truncated = False
        def __init__(self, raw): self.raw = raw
        def __call__(self, *a, **k): return self.raw

    import types
    saved = (_runtime, _make_sync_reflect_generate, _get_rag)
    fake_runtime = types.SimpleNamespace(model=object())
    globals()["_runtime"] = fake_runtime
    globals()["_get_rag"] = lambda: None
    saved_recent, saved_open = worklog.recent, worklog.open_threads
    worklog.recent = lambda n: [{"id": 1, "kind": "wander", "summary": "I read X."}]
    worklog.open_threads = lambda: []
    try:
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
    finally:
        globals()["_runtime"], globals()["_make_sync_reflect_generate"], globals()["_get_rag"] = saved
        worklog.recent, worklog.open_threads = saved_recent, saved_open
        _dispatch = {}
    print("deliberation selftest: OK")


if __name__ == "__main__":
    _selftest()
