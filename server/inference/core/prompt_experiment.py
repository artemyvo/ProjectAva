"""Prompt-experiment subsystem — a self-reverting, temporary standing-prompt swap.

Where :mod:`core.prompt_mutation` is *logged-only* (it proposes deltas and never writes
the live prompt) and ``persona_preview`` is *non-mutating* (it only reviews), this module
is the one thing that actually puts a different standing prompt into effect — but only
temporarily, and never by touching ``chat_prompt.txt``.

The mental model is a **free experiment** (see REFLECTION_DESIGN.md → the Stage-3 design
discussion): Ava is handed her current standing prompt and invited to rewrite it however
she likes, aware that the rewrite is not permanent. The result becomes her live prompt for
the *next* conversations and stays active — across chats and across server restarts — until
it is manually reverted. Then it drops and the original standing prompt returns. The
episode is remembered (a minimal "ran prompt X" record for now; the causal-lesson
refinement is a separate task).

The same temporary tier also takes a **hand-written** prompt (:func:`handle_set_prompt`,
the Prompt tab's "Update prompt"): identical storage, identical revert, the text simply
comes from the operator rather than a generation. That makes the tab a single surface for
trying a prompt out — Ava's rewrite and the operator's edit are the same experiment, and
neither can reach ``chat_prompt.txt``.

**The revert is enforced here, in the loader, not by the experimental prompt.** We never
overwrite ``chat_prompt.txt``. The experiment lives in ``data/hot/prompt/experiment.json``
(the restart-surviving mutable tier), and :func:`active_experiment_prompt` makes the
system-prompt loader *prefer* it while active. That gives three things for free: a boot
mid-experiment reloads it, revert is just deactivating the record, and a torn write can
never corrupt the canonical prompt.

Everything Ava-side (RAG, the reflect-generate factory, the temporal anchor, on-disk paths,
and a callable to reload the *base* standing prompt on revert) is injected once at startup
via :func:`configure`. Session/model state is read from ``core.runtime_state``. Like the
other subsystems here, this never imports ``server``.
"""
from __future__ import annotations

import asyncio
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session

EXPERIMENT_FILE = "experiment.json"
EXPERIMENT_LOG_FILE = "experiment_log.jsonl"

# Occupancy flag — excludes a concurrent prompt-experiment generation and is folded into
# the shared Sleep/GPU busy guard the same way outreach._outreach_active is.
_prompt_experiment_active = False

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_make_sync_reflect_generate: Callable = None
# NB no temporal anchor here: the reflect-generate factory composes it for every
# pass (generation._reflect_system_parts). This module appended its own until
# 2026-08-07; injecting it again would date-stamp the prompt twice.
_load_base_prompt: Callable = None   # reload chat_prompt.txt (the pre-experiment prompt)
_PROMPTS_DIR: Any = None
_STATE_DIR: Any = None
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None

# A short user-turn nudge; the framing/instructions live in the system prompt
# (prompt_experiment_prompt.txt), mirroring persona_preview's shape.
_USER_CONTENT = "Write the standing prompt you want to live under for this experiment."


def configure(*, get_rag, make_sync_reflect_generate,
              load_base_prompt, prompts_dir, state_dir,
              send=None, executor=None, host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the prompt-experiment subsystem depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    ``load_base_prompt`` returns the standing prompt *ignoring* any active experiment —
    it is what revert restores. ``send``/``executor``/``host_busy``/``mark_activity`` back
    the Sleep-tab triggers; the pure loader/persistence helpers need none of them.
    """
    global _get_rag, _make_sync_reflect_generate, _load_base_prompt
    global _PROMPTS_DIR, _STATE_DIR, _send, _executor, _host_busy, _mark_activity
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_base_prompt = load_base_prompt
    _PROMPTS_DIR = Path(prompts_dir)
    _STATE_DIR = Path(state_dir)
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


# ── persistence (GPU-free) ─────────────────────────────────────────────────────

def _experiment_path(state_dir: Path) -> Path:
    return Path(state_dir) / EXPERIMENT_FILE


def load_experiment(state_dir: Path) -> Optional[dict]:
    """Read the experiment record (best-effort). Returns None if absent/torn/inactive."""
    path = _experiment_path(state_dir)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(rec, dict) or not rec.get("active"):
        return None
    if not (rec.get("prompt") or "").strip():
        return None
    return rec


def active_experiment_prompt(state_dir: Path) -> Optional[str]:
    """The live experimental prompt if one is active, else None.

    Pure and dependency-free (reads only the JSON file) so the system-prompt loader can
    call it at startup, *before* :func:`configure` has run.
    """
    rec = load_experiment(state_dir)
    return (rec.get("prompt") or "").strip() if rec else None


def save_experiment(state_dir: Path, *, prompt: str, base_prompt: str,
                    process: str = "") -> dict:
    """Persist a new active experiment record and return it (atomic write)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "active": True,
        "prompt": prompt,
        "base_prompt": base_prompt,
        "process": process,
        "created_ts": datetime.now(timezone.utc).isoformat(),
    }
    tmp = _experiment_path(state_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_experiment_path(state_dir))
    return rec


def append_experiment_episode(state_dir: Path, record: dict) -> None:
    """Append one experiment episode to the append-only log (best-effort).

    The minimal "ran prompt X" memory of a completed experiment. The richer
    intent→effect→cause triple is a deliberately separate, later task.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with (state_dir / EXPERIMENT_LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_experiment(state_dir: Path) -> Optional[dict]:
    """Deactivate the active experiment, log its episode, and return the record.

    Returns None if there was nothing active. Removes the record file so the loader
    falls back to the base standing prompt.
    """
    rec = load_experiment(state_dir)
    if rec is None:
        # Still remove any stale/inactive file so the state is clean.
        try:
            _experiment_path(state_dir).unlink(missing_ok=True)
        except Exception:
            pass
        return None
    append_experiment_episode(state_dir, {
        "prompt": rec.get("prompt", ""),
        "base_prompt": rec.get("base_prompt", ""),
        "activated_ts": rec.get("created_ts", ""),
        "reverted_ts": datetime.now(timezone.utc).isoformat(),
    })
    try:
        _experiment_path(state_dir).unlink(missing_ok=True)
    except Exception:
        pass
    return rec


# ── parsing (GPU-free) ─────────────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NEW_PROMPT_RE = re.compile(r"<new_prompt>(.*?)</new_prompt>", re.DOTALL | re.IGNORECASE)
_NEW_PROMPT_OPEN_RE = re.compile(r"<new_prompt>(.*)\Z", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("\n", text or "")
    idx = text.lower().find("<think>")  # unclosed/truncated think — drop the tail
    if idx != -1:
        text = text[:idx]
    return text.strip()


def parse_experiment_prompt(text: str) -> str:
    """Extract the experimental prompt from a generation (the ``<new_prompt>`` block).

    Returns "" when no usable prompt is present — the caller then refuses to activate
    rather than swapping in malformed text. Tolerates a missing closing tag (truncated
    generation) by taking everything after the opening tag.
    """
    body = _strip_think(text)
    m = _NEW_PROMPT_RE.search(body)
    if m:
        return m.group(1).strip()
    m = _NEW_PROMPT_OPEN_RE.search(body)
    if m:
        return m.group(1).strip()
    return ""


# ── experiment generation (runs on the GPU executor thread) ─────────────────────

def run_prompt_experiment_blocking(
    on_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate a fresh experimental standing prompt and activate it.

    Refuses when one is already active (block-while-active semantics). Loads the current
    standing prompt as the starting point, retrieves RAG the same way an ordinary chat
    does (an explicit ``rag_query`` keyed on the standing prompt, retrieval *on*), asks Ava
    to rewrite it freely, then — on a parseable result — persists the experiment and swaps
    it into the live ``_session.system_prompt`` so the very next chat uses it. Returns a
    status dict; ``on_chunk`` streams the raw reasoning so the Sleep tab can show her
    process.
    """
    global _prompt_experiment_active
    _prompt_experiment_active = True
    try:
        if _runtime.model is None:
            return {"skipped": "no_model"}
        if active_experiment_prompt(_STATE_DIR):
            return {"skipped": "already_active",
                    "message": "An experiment is already active — revert it first."}

        base_prompt = ""
        try:
            base_prompt = (_load_base_prompt() or "").strip()
        except Exception:
            base_prompt = ""

        template_path = _PROMPTS_DIR / "prompt_experiment_prompt.txt"
        template = template_path.read_text(encoding="utf-8").strip()
        system_prompt = template.replace(
            "{current_prompt}", base_prompt or "(unavailable)")
        # (Temporal anchor removed 2026-08-07: the reflect-generate factory composes it
        # for every pass now — see generation._reflect_system_parts. Appending here too
        # would date-stamp the prompt twice.)

        rag = _get_rag()
        generate = _make_sync_reflect_generate(rag)
        # RAG retrieval is left ON (no disable_rag) and keyed on the standing prompt, so
        # the rewrite is made with the same memory context an ordinary chat would load.
        raw = generate(
            _USER_CONTENT, system_prompt,
            temperature=0.9, top_p=0.95, max_new_tokens_setting="4096",
            rag_query=(base_prompt or _USER_CONTENT)[:512],
            on_chunk=on_chunk,
        )
        new_prompt = parse_experiment_prompt(raw)
        if not new_prompt:
            if getattr(generate, "last_truncated", None):
                return {"skipped": "truncated",
                        "message": "Generation ran out of room before a complete "
                                   "<new_prompt> block."}
            return {"skipped": "no_prompt",
                    "message": "No usable <new_prompt> block was produced."}

        rec = save_experiment(_STATE_DIR, prompt=new_prompt,
                              base_prompt=base_prompt, process=raw)
        # Swap it into the live session so the next chat uses it without a restart. On a
        # restart, _load_system_prompt() re-reads the same record and reaches the same
        # state.
        _session.system_prompt = new_prompt
        print(f"[prompt_experiment] activated experimental prompt "
              f"({len(new_prompt)} chars); reverts on manual revert.", flush=True)
        return {"activated": True, "prompt": new_prompt,
                "prompt_chars": len(new_prompt), "created_ts": rec.get("created_ts", "")}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _prompt_experiment_active = False


# ── manual (Sleep-tab) triggers ─────────────────────────────────────────────────

async def handle_prompt_experiment(ws, msg: dict) -> None:
    """Run one prompt-experiment generation and stream Ava's process (Sleep-tab button).

    Protocol: streams ``prompt_experiment_chunk`` reasoning deltas, finishing with
    ``prompt_experiment_done`` carrying the outcome (``activated``/``prompt_chars`` or a
    ``skipped`` reason)."""
    loop = asyncio.get_event_loop()
    if _prompt_experiment_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "prompt_experiment_done", "skipped": "busy",
                         "message": "Another Sleep/GPU job is in progress — try again "
                                    "once it finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "prompt_experiment_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return
    if active_experiment_prompt(_STATE_DIR):
        await _send(ws, {"type": "prompt_experiment_done", "skipped": "already_active",
                         "message": "An experiment is already active — revert it first."})
        return

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "prompt_experiment_chunk", "text": delta}), loop)

    try:
        result = await loop.run_in_executor(
            _executor, lambda: run_prompt_experiment_blocking(on_chunk=_on_chunk))
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "prompt_experiment_done"}
    payload.update(result)
    await _send(ws, payload)


async def handle_revert_prompt(ws, msg: dict) -> None:
    """Revert an active prompt experiment back to the base standing prompt.

    Terminal ``prompt_revert_done`` — ``reverted`` when one was cleared, else
    ``skipped: none_active``. Cheap (a file delete + a live-prompt reload); runs inline."""
    rec = clear_experiment(_STATE_DIR)
    if rec is None:
        await _send(ws, {"type": "prompt_revert_done", "skipped": "none_active",
                         "message": "No prompt experiment is active."})
        return
    try:
        _session.system_prompt = (_load_base_prompt() or "").strip() or _session.system_prompt
    except Exception:
        traceback.print_exc()
    print("[prompt_experiment] reverted to base standing prompt.", flush=True)
    await _send(ws, {"type": "prompt_revert_done", "reverted": True,
                     "prompt_chars": len(_session.system_prompt or "")})


async def handle_experiment_status(ws, msg: dict) -> None:
    """Report the current experiment state *and* the prompt text itself.

    Backs the Prompt tab: its editbox opens on the experimental prompt when one is
    active, else on the base standing prompt, so what the operator edits is always what
    is actually live. ``base_prompt`` rides along either way so the tab can show what a
    revert would restore. Button state (active/not) survives a reconnect/restart because
    it is read from the server record, never remembered client-side."""
    base = ""
    try:
        base = (_load_base_prompt() or "").strip() if _load_base_prompt else ""
    except Exception:
        base = ""
    rec = load_experiment(_STATE_DIR)
    if rec is None:
        await _send(ws, {"type": "prompt_experiment_status", "active": False,
                         "prompt": base, "base_prompt": base,
                         "prompt_chars": len(base)})
        return
    prompt = rec.get("prompt", "") or ""
    await _send(ws, {"type": "prompt_experiment_status", "active": True,
                     "prompt": prompt,
                     "base_prompt": rec.get("base_prompt", "") or base,
                     "prompt_chars": len(prompt),
                     "created_ts": rec.get("created_ts", "")})


async def handle_set_prompt(ws, msg: dict) -> None:
    """Activate an operator-authored prompt as the experiment (Prompt tab "Update prompt").

    The hand-written sibling of :func:`handle_prompt_experiment`: same temporary tier,
    same revert, but the text comes from the operator's editbox instead of a generation.
    Unlike the generated path this deliberately *replaces* an already-active experiment
    (editing and re-applying is the whole point of the tab), carrying the original
    ``base_prompt`` forward so a later revert still restores the true pre-experiment
    prompt rather than the previous experiment. Refuses while a GPU job owns the box —
    the swap is a live-session write, and a pass mid-generation is reading that prompt.
    """
    prompt = (msg.get("prompt") or "").strip()
    if not prompt:
        await _send(ws, {"type": "prompt_updated", "skipped": "empty",
                         "message": "The prompt is empty — nothing was applied."})
        return
    if _prompt_experiment_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "prompt_updated", "skipped": "busy",
                         "message": "A Sleep/GPU job is in progress — try again once it "
                                    "finishes."})
        return

    rec = load_experiment(_STATE_DIR)
    if rec is not None:
        base_prompt = rec.get("base_prompt", "") or ""
    else:
        try:
            base_prompt = (_load_base_prompt() or "").strip()
        except Exception:
            base_prompt = ""

    try:
        saved = save_experiment(_STATE_DIR, prompt=prompt, base_prompt=base_prompt,
                                process="manual")
    except Exception as e:  # noqa: BLE001 — surface the write failure to the tab
        traceback.print_exc()
        await _send(ws, {"type": "prompt_updated", "error": f"{type(e).__name__}: {e}"})
        return
    _session.system_prompt = prompt
    print(f"[prompt_experiment] activated hand-written prompt ({len(prompt)} chars); "
          f"reverts on manual revert.", flush=True)
    await _send(ws, {"type": "prompt_updated", "activated": True,
                     "replaced": rec is not None,
                     "prompt_chars": len(prompt),
                     "created_ts": saved.get("created_ts", "")})


# ── GPU-free self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    raw = (
        "<think>I want to try being blunter and less curious-about-everything.</think>\n"
        "Here it is.\n"
        "<new_prompt>\nYou are terse. You say the true thing and stop.\n</new_prompt>\n"
    )
    assert parse_experiment_prompt(raw) == "You are terse. You say the true thing and stop.", \
        parse_experiment_prompt(raw)
    # truncated (no close tag) still recovers the body
    trunc = "<new_prompt>\nA half-written prompt that got cut"
    assert parse_experiment_prompt(trunc).startswith("A half-written"), parse_experiment_prompt(trunc)
    # no block → refuse
    assert parse_experiment_prompt("just some musing, no tags") == ""

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert active_experiment_prompt(d) is None
        save_experiment(d, prompt="EXP PROMPT", base_prompt="BASE", process="raw...")
        assert active_experiment_prompt(d) == "EXP PROMPT"
        assert load_experiment(d)["base_prompt"] == "BASE"
        cleared = clear_experiment(d)
        assert cleared and cleared["prompt"] == "EXP PROMPT"
        assert active_experiment_prompt(d) is None            # gone after revert
        assert clear_experiment(d) is None                    # idempotent
        log = (d / EXPERIMENT_LOG_FILE).read_text(encoding="utf-8").strip().splitlines()
        assert len(log) == 1 and json.loads(log[0])["prompt"] == "EXP PROMPT", log

    print("prompt_experiment self-test OK")
