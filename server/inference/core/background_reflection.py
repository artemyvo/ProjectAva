"""Background per-chat reflection — idle-triggered, user-preemptible.

The expensive part of a reflection run is the *per-chat* work — consolidation + revision
+ branch generation, one generation per exchange on a large model — and it is embarrassingly
parallel across chats. This subsystem does that work **incrementally in the background**
whenever the user has been idle a while, so the nightly cycle has far less to do.

It is the fifth autonomous idle job (alongside ``outreach``/``synthesis``/``checkin``/
``wander``), but with distinct DNA:

  * **Trigger — a long USER-idle window.** Registered with a longer ``idle_seconds`` than
    its siblings (30 min): it runs only when the box has been quiet, so the GPU is free.
  * **Instant preemption.** When the user comes back, an incoming chat calls
    :func:`request_preempt` — which sets the stop flag, fires the shared ``cancel_event``
    (aborting the in-flight generation), and lets the chat proceed. A chat interrupted
    mid-reflection is simply discarded and re-reflected from the top later; only *completed*
    chats are cached. This is the ONE place chat *preempts* reflection instead of being
    refused (a foreground/operator Sleep run is still refused — see ``generation.py``).
  * **Only the per-chat passes.** Each wake reflects ONE aged chat in ``chat_only`` mode
    (see :meth:`ReflectionRunner.execute_run`): consolidation + revision + branch generation,
    NO run-level persona digest / clean-base judge / fact placement. The chat is frozen
    ``chat_reflected`` (stage one of the two-stage freeze) and its clean-base job payloads
    are checkpointed. The next NORMAL reflection run folds the checkpoint, finishes the
    chat (runs the clean-base phase over those payloads), and stamps ``reflected_at``.

**A wake is a three-rung ladder, cheapest first, and it stops at the first rung with work.**

  0. **Stale reach-out sweep** (GPU-free, always). Delete the chats Ava opened herself that
     nobody answered inside the window (``reachout.stale_delete_hours``, default 48 h) —
     the shared policy in ``reflection_service.run_stale_reachout_sweep``, which until now
     ran only at the head of an *operator* reflection run. It goes first so no rung below
     spends a generation on a transcript this wake is about to remove.
  1. **Sidecar backfill** — every chat missing its gist (``<stem>.summary.json``) or its
     fact protocol (``<stem>.facts.json``), **oldest first**. Two cheap generations per
     chat against the whole corpus, versus one full reflection's generation *per exchange*
     of one chat: the box gets broad recall + a complete extraction record long before it
     gets deep per-chat targets. The passes are the registry's (``core.modules``), so this
     is that registry's first real SINK — the value it returns is written here rather than
     inside the pass, which is the property the workbench was built for.
  2. **Per-chat reflection** — the drain described above, and only when rung 1 is empty.
     Deliberate: a chat reflection later re-derives both sidecars anyway, so the overlap is
     accepted in exchange for corpus-wide coverage arriving first.

Per-chat atomicity: each chat is reflected in a fresh throwaway staging dir. On success its
deltas + frozen sidecar are APPENDED into the durable checkpoint (accumulating across wakes)
and its clean-base jobs written to a sibling dir; on preempt the throwaway is discarded and
nothing reaches the checkpoint. So a partial chat never corrupts the cache.

Owns its occupancy flag (``_background_reflection_active``) + the current run id; the
chat-preemption path reads them. It never imports ``server`` or ``reflection_service``:
the single-chat reflection engine is injected as ``reflect_one_chat_fn`` (wired to
``reflection_service.run_chat_only_reflection`` at startup). GPU-free self-test:
``python -m core.background_reflection``.
"""
from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, session as _session
from core.chat_sidecar import ChatSidecar, iter_chat_json_files, sanitize_gist
from core import chat_facts

# ── occupancy / preemption state (read by generation.py's chat-preemption path) ──
_background_reflection_active = False
_bg_run_id: Optional[str] = None
# Set by request_preempt() when a user chat arrives — the per-chat loop stops between
# chats, and the in-flight chat stops at its next exchange boundary (store stop flag).
_preempt_requested = False
_bg_store: Any = None   # the run store for the active wake (so request_preempt can stop it)

# ── injected capabilities (populated by configure()) ──
_reflect_one_chat_fn: Callable = None    # reflection_service.run_chat_only_reflection
_stale_sweep_fn: Callable = None         # reflection_service.run_stale_reachout_sweep
_refresh_chats_fn: Callable = None       # rag.refresh_chat_index
_load_server_config: Callable = None
_CHATS_DIR: Any = None
_DATA_DIR: Any = None
_RUNS_DIR: Any = None
_host_busy: Callable = None
_cancel_event: Any = None


def configure(*, reflect_one_chat_fn, chats_dir, data_dir, runs_dir,
              stale_sweep_fn=None, refresh_chats_fn=None, load_server_config=None,
              host_busy=None, cancel_event=None) -> None:
    """Wire the background pass with the server capabilities it needs.

    ``reflect_one_chat_fn(config, store, staging_dir, on_chat_reflected) -> status`` runs
    ONE chat's per-chat reflection on the executor thread (wired to
    ``reflection_service.run_chat_only_reflection``). ``stale_sweep_fn() -> [deleted]`` is
    the shared stale-reach-out policy (``reflection_service.run_stale_reachout_sweep``) —
    absent ⇒ rung 0 is a no-op. ``refresh_chats_fn`` rebuilds the chat RAG index after the
    backfill writes gists (a gist is a chat-RAG passage; without it the new recall is on
    disk but not retrievable until something else rebuilds). ``host_busy`` reports the
    non-idle-job GPU owners; ``cancel_event`` is the shared generation cancel used for
    instant preempt.
    """
    global _reflect_one_chat_fn, _CHATS_DIR, _DATA_DIR, _RUNS_DIR, _host_busy, _cancel_event
    global _stale_sweep_fn, _refresh_chats_fn, _load_server_config
    _reflect_one_chat_fn = reflect_one_chat_fn
    _stale_sweep_fn = stale_sweep_fn
    _refresh_chats_fn = refresh_chats_fn
    _load_server_config = load_server_config
    _CHATS_DIR = Path(chats_dir)
    _DATA_DIR = Path(data_dir)
    _RUNS_DIR = Path(runs_dir)
    _host_busy = host_busy
    _cancel_event = cancel_event


def _backfill_enabled() -> bool:
    """``background_reflection.sidecar_backfill`` (default ON) — rung 1's kill switch.

    It exists because rung 1 *gates* rung 2 by design: on a large corpus with no sidecars
    the box will spend hours writing gists and fact protocols before it reflects anything,
    which is the intent but must be reversible without a code change."""
    try:
        cfg = (_load_server_config() or {}).get("background_reflection") or {}
        if cfg.get("sidecar_backfill") is not None:
            return bool(cfg.get("sidecar_backfill"))
    except Exception:
        pass
    return True


def _bg_staging_dir() -> Path:
    # Dedicated throwaway staging, isolated from the foreground reflection_staging workspace
    # (so a continue-staging run or a clear-staging wipe can never collide with it).
    return _DATA_DIR / "hot" / "background_staging"


def _reset_staging() -> None:
    d = _bg_staging_dir()
    try:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def is_active() -> bool:
    """True while a background per-chat pass holds the executor."""
    return _background_reflection_active


def request_preempt() -> None:
    """A user chat arrived — abort the background pass and hand the GPU back.

    Sets the stop flags (the per-chat loop stops between chats; the in-flight chat stops at
    its next exchange boundary) and fires the shared cancel event so the current generation
    aborts within roughly one step rather than one whole chat. Idempotent; safe to call when
    no background pass is running."""
    global _preempt_requested
    _preempt_requested = True
    try:
        if _bg_run_id and _bg_store is not None:
            _bg_store.request_stop(_bg_run_id)
    except Exception:
        pass
    try:
        if _cancel_event is not None:
            _cancel_event.set()
    except Exception:
        pass


# ── backlog enumeration ─────────────────────────────────────────────────────── #

def _is_unanswered_outreach(session: dict) -> bool:
    """True for an Ava-initiated chat the user never replied to (only her opener) — nothing
    to reflect on. Mirrors ``ReflectionRunner._is_unanswered_outreach`` so background never
    picks a chat the runner would skip without freezing (which would loop forever)."""
    if (session.get("initiated_by") or "").strip() != "ava":
        return False
    exchanges = session.get("exchanges") or []
    return len(exchanges) <= 1


def _active_session_name() -> Optional[str]:
    try:
        lg = _session.logger
        if lg is not None and lg.current_file is not None:
            return lg.current_file.name
    except Exception:
        pass
    return None


def _checkpointed_stems() -> set[str]:
    """Chat stems already completed by a prior background wake and sitting in the durable
    checkpoint awaiting a normal run's fold.

    The background pass freezes each completed chat ``chat_reflected`` in a throwaway
    staging dir and commits that frozen sidecar into the checkpoint (never into the live
    ``_CHATS_DIR`` — the live stamp lands only when a normal reflection run folds the
    checkpoint). So the live sidecar this function's caller reads via ``is_chat_reflected``
    stays UNfrozen between background wakes; without this guard a completed chat is re-picked
    every wake until the next operator Sleep run folds the checkpoint. We treat a chat whose
    frozen sidecar is in the checkpoint (or whose clean-base jobs are staged in the sibling
    pending dir) as already background-reflected."""
    if _DATA_DIR is None:
        return set()
    stems: set[str] = set()
    from core.reflection_staging import get_checkpoint_paths, get_pending_clean_base_dir
    try:
        ckpt_chats = get_checkpoint_paths(_DATA_DIR)["chats_dir"]
        if ckpt_chats.exists():
            for sp in ckpt_chats.glob("*.state.json"):
                stems.add(sp.name[:-len(".state.json")])
    except Exception:
        pass
    try:
        pending = get_pending_clean_base_dir(_DATA_DIR)
        if pending.exists():
            for pp in pending.glob("*.json"):
                stems.add(pp.stem)
    except Exception:
        pass
    return stems


def list_backlog() -> list[str]:
    """Chats eligible for background per-chat reflection: not yet frozen at either stage
    (``reflected_at`` / ``chat_reflected``), not already completed in a prior background wake
    (checkpoint / pending-clean-base), have a real dialogue turn, and are not the active
    session. Sorted oldest-first (filename == timestamp)."""
    if _CHATS_DIR is None:
        return []
    sc = ChatSidecar(_CHATS_DIR)
    active = _active_session_name()
    checkpointed = _checkpointed_stems()
    out: list[str] = []
    for p in iter_chat_json_files(_CHATS_DIR):
        fn = p.name
        if fn == active:
            continue
        if p.stem in checkpointed:
            continue
        try:
            if sc.is_reflected(fn) or sc.is_chat_reflected(fn):
                continue
        except Exception:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_unanswered_outreach(data):
            continue
        out.append(fn)
    return sorted(out)


def list_sidecar_backlog() -> list[dict]:
    """Chats missing (or holding a stale) gist / fact sidecar. OLDEST-FIRST (filename == ts).

    Returns ``[{"session": fn, "summary": bool, "facts": bool}, …]`` — the two flags say
    which of the pair is wanted, so a chat that already has one pays only for the other.

    "Wanted" is missing OR superseded: a `<stem>.facts.json` whose records disagree with
    what `chat_facts.parse_facts` can read today counts as absent (`needs_reparse`). That
    is what makes a parser fix self-healing across instances — several boxes run this
    codebase over different corpora, so the set of damaged files is per-box and cannot be
    named in advance; each one finds and repairs its own on idle time, at the price of the
    same single generation a missing sidecar costs.

    Deliberately **not** gated on the reflect-once freeze: this is a backfill, and the
    chats that most need it are the OLD ones, reflected before these artifacts existed
    (`.summary.json` was split out of the state sidecar on 2026-08-06; `.facts.json`
    arrived the same week). "Has a gist" is asked of :meth:`ChatSidecar.summary_text`
    rather than of the file, so a chat whose recap still lives in the legacy in-sidecar
    ``consolidation_summary`` key counts as done and is not regenerated.

    Skipped: the active session (a live logger owns that transcript, and both writers
    refuse it anyway), a chat with no exchanges, and an unanswered Ava opener — which is
    not a conversation, never becomes one, and is what rung 0 deletes."""
    if _CHATS_DIR is None:
        return []
    sc = ChatSidecar(_CHATS_DIR)
    active = _active_session_name()
    out: list[dict] = []
    for p in iter_chat_json_files(_CHATS_DIR):
        fn = p.name
        if fn == active:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (data.get("exchanges") or []):
            continue
        if _is_unanswered_outreach(data):
            continue
        try:
            needs_summary = not sc.summary_text(fn)
            facts_doc = sc.read_facts(fn)
            # Missing, empty, or STALE — a protocol whose stored records disagree with what
            # `parse_facts` can read today (`chat_facts.needs_reparse`). A parser that could
            # not see its generation's marker placement wrote records with the fields empty
            # and the markers left in the text, and it did so for a whole pass at a time, so
            # the damage arrives as entire unusable protocols. Regenerating is the same one
            # cheap generation as backfilling a missing one, which is why it belongs on this
            # rung rather than behind a whole re-reflection — and why it needs no operator:
            # every box carrying the affected files repairs itself on its own idle time.
            needs_facts = (not (facts_doc.get("facts") or [])
                           or chat_facts.needs_reparse(facts_doc))
        except Exception:
            continue
        if needs_summary or needs_facts:
            out.append({"session": fn, "summary": needs_summary, "facts": needs_facts})
    return sorted(out, key=lambda r: r["session"])


# ── rung 1: the sidecar backfill ────────────────────────────────────────────── #

# The two artifacts, by module name — the per-chat body below is driven off this pair, so
# a third cheap per-chat artifact is a line here plus a module in the registry.
_BACKFILL_MODULES = ("chat_summary", "chat_facts")

# A module refusal that is about the BOX, not this chat: retrying the next chat would fail
# identically, so the drain stops instead of walking the whole backlog to collect the same
# refusal N times. Everything else `run_module_blocking` refuses (`bad_session`,
# `empty_session`, `no_content`) is about the one transcript.
_BOX_LEVEL_SKIPS = frozenset({"no_model", "no_prompt", "unknown_module",
                              "unsupported_injectable"})

# Chats whose backfill generated but wrote NOTHING, this process. Rung 1 gates rung 2, so
# without this a single chat that reliably produces an unwritable value — a two-line chat
# whose recap never survives `sanitize_gist`, a pass that always comes back empty — stays
# in the backlog forever and reflection never runs again. Deliberately in-process and not
# on disk: the cause is usually the prompt or the model, so a restart (which is how both
# change) is exactly when it is worth trying again.
_backfill_unproductive: set[str] = set()


def _write_backfilled(kind: str, filename: str, session: dict, result: dict,
                      run_id: str) -> bool:
    """Persist one module's produced value to its sidecar. Returns True if it landed.

    This is the SINK the module registry deliberately does not carry: `run_module_blocking`
    returns the value instead of writing it (that detachment is what makes a pass
    experimentable), so the caller that wants it durable says where it goes. Both writers
    refuse an empty/garbage value on their own — `write_summary` rejects a recap that
    survives no sanitation, `write_facts` an empty list — so a pass that came back unusable
    leaves the previous record standing rather than blanking it."""
    sc = ChatSidecar(_CHATS_DIR)
    live = _active_session_name()
    if kind == "chat_summary":
        return bool(sc.write_summary(
            source_session=filename, summary=str(result.get("text") or ""),
            run_id=run_id, live_session=live))
    return bool(sc.write_facts(
        source_session=filename, facts=list(result.get("records") or []),
        run_id=run_id, source_user=(session.get("user") or "").strip(),
        live_session=live))


def _backfill_one_chat(entry: dict, run_id: str) -> dict:
    """Produce and write whichever of the two sidecars *entry* is missing.

    Returns ``{"summary": bool, "facts": bool, "preempted": bool, "generated": bool,
    "refused": str}`` — what landed, whether the user came back, whether a pass actually
    ran (so a transient refusal is not mistaken for an unproductive chat), and any
    box-level refusal that should stop the whole drain.

    The generation is `core.modules.run_module_blocking`, i.e. the SAME spec the Modules
    tab simulates with, so what the backfill writes is what an operator can reproduce and
    retune from the workbench. A preempt mid-generation returns partial text; that result
    is DISCARDED rather than written, matching the per-chat rule above (a half-produced
    artifact must never look like a finished one)."""
    from core import modules

    wrote = {"summary": False, "facts": False, "preempted": False,
             "generated": False, "refused": ""}
    filename = entry["session"]
    try:
        session = json.loads((_CHATS_DIR / filename).read_text(encoding="utf-8"))
    except Exception:
        return wrote

    for name in _BACKFILL_MODULES:
        key = "summary" if name == "chat_summary" else "facts"
        if not entry.get(key):
            continue
        if _preempt_requested:
            wrote["preempted"] = True
            break
        try:
            result = modules.run_module_blocking(module=name, filename=filename)
        except Exception:
            traceback.print_exc()
            continue
        if _preempt_requested:
            # The cancel event breaks the stream mid-generation and returns what it had.
            wrote["preempted"] = True
            break
        skipped = str(result.get("skipped") or "")
        if skipped in _BOX_LEVEL_SKIPS:
            wrote["refused"] = skipped
            break
        if skipped or result.get("error"):
            continue
        wrote["generated"] = True
        try:
            wrote[key] = _write_backfilled(name, filename, session, result, run_id)
        except Exception:
            traceback.print_exc()
    return wrote


def _run_sidecar_backfill(backlog: list[dict], run_id: str) -> dict:
    """Drain the missing-sidecar backlog oldest-first until it is empty or the user returns.

    Chat RAG is rebuilt once at the end if any gist landed — a gist is indexed as a
    ``kind="gist"`` passage, so without the refresh the new recall sits on disk unreachable
    until something else rebuilds the index."""
    from core import activity_log

    done = summaries = facts = 0
    refused = ""
    for entry in backlog:
        if _preempt_requested:
            break
        wrote = _backfill_one_chat(entry, run_id)
        if wrote["summary"]:
            summaries += 1
        if wrote["facts"]:
            facts += 1
        if wrote["generated"] and not (wrote["summary"] or wrote["facts"]
                                       or wrote["preempted"]):
            # A pass ran and nothing it produced was writable. Stop asking (see the note on
            # `_backfill_unproductive`) — this rung gates the next one. Keyed on `generated`
            # so a refusal that never reached the model doesn't retire the chat.
            _backfill_unproductive.add(entry["session"])
        if wrote["summary"] or wrote["facts"]:
            done += 1
            made = " + ".join(
                [label for label, got in (("gist", wrote["summary"]),
                                          ("facts", wrote["facts"])) if got])
            activity_log.append(
                "background_reflection", "progress",
                f"Sidecar backfill: wrote {made} for {entry['session']}")
        if wrote["preempted"]:
            break
        if wrote["refused"]:
            refused = wrote["refused"]
            break

    if summaries and _refresh_chats_fn is not None:
        try:
            _refresh_chats_fn()
        except Exception:
            traceback.print_exc()

    preempted = bool(_preempt_requested)
    out = {"backfilled": done, "summaries": summaries, "facts": facts,
           "status": "preempted" if preempted else "done",
           "remaining": max(0, len(backlog) - done)}
    if refused and not done:
        # Nothing landed and the box refused every attempt — report it as a skip so the
        # scheduler retries on the next poll rather than sleeping the interval on a
        # condition (no model, missing prompt) that may clear at any moment.
        out["skipped"] = f"backfill refused: {refused}"
    return out


# ── the idle-job body ───────────────────────────────────────────────────────── #

def _record_conversation_worklog(filename: str, staging) -> None:
    """Emit a first-person ``conversation`` worklog entry for the chat this wake just froze.

    The entry itself — who it was with, the "about X" from the consolidation gist, and the
    close of the reach-out thread that opened the chat — is `chat_worklog.record_conversation`,
    shared with the FOREGROUND runner's ``reflected_at`` freeze so a chat produces exactly
    one such entry no matter which path reflected it. (It lived here, private, when this was
    the only path that froze a chat per-chat; that made the thread-close dead code on any box
    reflecting through operator Sleep runs — see `core.chat_worklog`.) All this adds is
    reading the two inputs from the background pass's own throwaway staging: the transcript
    for the session facts, and the per-chat consolidation summary the runner just wrote to
    ``staging/chats/<stem>.state.json`` for the gist. Best-effort — a worklog hiccup must
    never disturb the reflection."""
    try:
        from pathlib import Path
        from core import chat_worklog

        session: dict = {}
        try:
            if _CHATS_DIR is not None:
                session = json.loads((_CHATS_DIR / filename).read_text(encoding="utf-8"))
        except Exception:
            session = {}

        gist = ""
        try:
            from core.chat_sidecar import ChatSidecar
            gist = ChatSidecar(Path(staging) / "chats").summary_text(filename)
        except Exception:
            gist = ""

        chat_worklog.record_conversation(filename, session, gist)
    except Exception:
        traceback.print_exc()


def _reflect_one_chat(store, run_id: str, filename: str) -> int:
    """Reflect ONE chat per-chat in a fresh throwaway staging dir; on success its deltas +
    frozen sidecar are committed to the durable checkpoint by the on_chat_reflected callback.
    Returns 1 if the chat completed (frozen chat_reflected), else 0."""
    from core.reflection_config import ReflectionRunConfig, validate_overrides
    from core.reflection_staging import commit_background_chat, write_pending_clean_base

    staging = _bg_staging_dir()
    _reset_staging()

    def _on_chat_reflected(fn: str, judge_jobs: list, fact_candidates: list) -> None:
        # Fired inside execute_run right after the chat is frozen chat_reflected in the
        # throwaway staging. Append its one-chat deltas + sidecar into the accumulating
        # checkpoint, and persist its clean-base jobs for a later normal run to finish.
        commit_background_chat(_DATA_DIR, staging, fn, run_id)
        write_pending_clean_base(_DATA_DIR, fn, judge_jobs, fact_candidates)
        # A user conversation just got processed — record it in Ava's worklog in her own
        # voice ("I talked with <user>. <gist>"), reading the gist from this same staging.
        _record_conversation_worklog(fn, staging)

    cfg = ReflectionRunConfig(
        run_id=run_id, source="idle", selected_sessions=[filename],
        overrides=validate_overrides({}), chat_only=True,
    )
    status = "failed"
    try:
        status = _reflect_one_chat_fn(cfg, store, staging, _on_chat_reflected)
    except Exception:
        traceback.print_exc()
    finally:
        # Discard the throwaway staging: a completed chat's work is already in the
        # checkpoint; a preempted (partial) chat's work is intentionally dropped.
        _reset_staging()
    return 1 if status == "completed" else 0


def run_background_reflection_blocking() -> dict:
    """One idle dispatch of the three-rung ladder (see the module docstring). Runs on the
    GPU executor thread (blocking).

    Rung 0 (delete stale unanswered reach-outs) is GPU-free and runs on every wake. Then
    the wake takes the FIRST rung with work and stops there: sidecar backfill if any chat
    is missing its gist or fact protocol, otherwise the per-chat reflection drain. Each
    rung drains oldest-first until empty or the user returns.

    Returns a summary dict; a ``skipped`` key means the attempt did no work at all (drives
    the scheduler's ``consumed`` classifier to retry on the next poll rather than sleep a
    full interval)."""
    global _background_reflection_active, _bg_run_id, _preempt_requested, _bg_store

    if _host_busy is not None:
        try:
            if _host_busy():
                return {"skipped": "host_busy"}
        except Exception:
            pass

    # ── rung 0: stale reach-out sweep. GPU-free and unconditional, so it runs even on a
    # wake where every rung below has nothing to do — and BEFORE them, so neither spends a
    # generation on a transcript about to be removed. Shared policy with the reflection
    # run's head phase (reflection_service.run_stale_reachout_sweep).
    deleted: list = []
    if _stale_sweep_fn is not None:
        try:
            deleted = _stale_sweep_fn() or []
        except Exception:
            traceback.print_exc()
    if deleted:
        from core import activity_log
        names = ", ".join(d.get("session", "?") for d in deleted[:5])
        if len(deleted) > 5:
            names += f", +{len(deleted) - 5} more"
        activity_log.append(
            "background_reflection", "progress",
            f"Removed {len(deleted)} unanswered reach-out chat(s): {names}")

    def _nothing_to_do(reason: str) -> dict:
        # A wake that only swept still DID something: report it rather than reporting a
        # skip, so the journal shows the deletion (and the scheduler spends the interval).
        if deleted:
            return {"deleted": len(deleted), "status": "done"}
        return {"skipped": reason}

    if _runtime.model is None:
        return _nothing_to_do("no_model")
    if _reflect_one_chat_fn is None:
        return _nothing_to_do("not_configured")

    # ── rung 1: sidecar backfill — cheap artifacts across the WHOLE corpus before any
    # expensive per-chat reflection. It gates rung 2 on purpose (see the module docstring);
    # `background_reflection.sidecar_backfill: false` turns it off.
    sidecar_backlog = [e for e in (list_sidecar_backlog() if _backfill_enabled() else [])
                       if e["session"] not in _backfill_unproductive]
    if sidecar_backlog:
        run_id = "bg_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        _bg_run_id = run_id
        _preempt_requested = False
        _background_reflection_active = True
        try:
            from core import activity_log
            print(f"[background_reflection] sidecar backfill: "
                  f"{len(sidecar_backlog)} chat(s) missing a gist/fact record", flush=True)
            activity_log.append(
                "background_reflection", "progress",
                f"Sidecar backfill: {len(sidecar_backlog)} chat(s) missing a gist or fact "
                f"record (oldest first)")
            result = _run_sidecar_backfill(sidecar_backlog, run_id)
            print(f"[background_reflection] backfill {result['status']} — "
                  f"{result['backfilled']} chat(s) written this wake", flush=True)
            return {**result, "deleted": len(deleted)}
        except Exception as e:
            traceback.print_exc()
            return {"skipped": f"error: {e}", "deleted": len(deleted)}
        finally:
            _background_reflection_active = False
            _bg_run_id = None
            _preempt_requested = False

    # ── rung 2: the per-chat reflection drain.
    backlog = list_backlog()
    if not backlog:
        return _nothing_to_do("no_unreflected_chats")

    from core.reflection_config import (
        ReflectionRunConfig, ReflectionRunStore, validate_overrides)

    store = ReflectionRunStore(_RUNS_DIR)
    run_id = "bg_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    reflected = 0
    _bg_store = store
    _bg_run_id = run_id
    _preempt_requested = False
    _background_reflection_active = True
    try:
        # One run record per wake; each chat's execute_run reuses this run_id, so a single
        # request_stop(run_id) preempts whichever chat is in flight.
        base = ReflectionRunConfig(
            run_id=run_id, source="idle", selected_sessions=list(backlog),
            overrides=validate_overrides({}), chat_only=True,
        )
        store.create_run(base)
        print(f"[background_reflection] draining {len(backlog)} unreflected chat(s) "
              f"(run {run_id})", flush=True)
        # Head-of-burst context for the Activity tab: the chip only says "running", so
        # the journal gets the plan. The per-chat reflection detail follows via the
        # store's activity mirror (a background run mirrors at Sleep-tab grain — see
        # reflection_config._mirror_to_activity); the scheduler's describe line closes
        # the burst. Leaf import, same pattern as reachout_gate.
        from core import activity_log
        activity_log.append(
            "background_reflection", "progress",
            f"Background reflection: draining {len(backlog)} unreflected chat(s)")
        for fn in backlog:
            if _preempt_requested or store.is_stop_requested(run_id):
                break
            reflected += _reflect_one_chat(store, run_id, fn)
        preempted = _preempt_requested or store.is_stop_requested(run_id)
        print(f"[background_reflection] {'preempted' if preempted else 'idle drain done'} — "
              f"{reflected} chat(s) reflected this wake", flush=True)
        return {"reflected": reflected, "deleted": len(deleted),
                "status": "preempted" if preempted else "done",
                "remaining": max(0, len(backlog) - reflected)}
    except Exception as e:
        traceback.print_exc()
        return {"skipped": f"error: {e}", "reflected": reflected,
                "deleted": len(deleted)}
    finally:
        _background_reflection_active = False
        _bg_run_id = None
        _bg_store = None
        _preempt_requested = False
        _reset_staging()


# ── GPU-free self-test ──────────────────────────────────────────────────────── #

def _selftest() -> None:
    """Exercise gist sanitation + backlog filtering + preempt-flag transitions with a
    mocked reflect fn. Run: ``python -m core.background_reflection``."""
    import tempfile

    global _CHATS_DIR, _DATA_DIR, _RUNS_DIR, _reflect_one_chat_fn, _host_busy
    global _background_reflection_active, _bg_run_id, _bg_store, _preempt_requested
    global _stale_sweep_fn, _refresh_chats_fn, _load_server_config

    # ── gist sanitation (shared with the sidecar writer + the RAG reader) ──────── #
    prose = ("We spent the evening arguing about whether a subject is defined by its "
             "mistakes or by noticing them, and neither of us backed down.")
    # Prose survives untouched.
    assert sanitize_gist(prose) == prose
    # A trailing consolidation dump is cut at the seam, prose kept.
    assert sanitize_gist(prose + "\n\n## WEIGHTS\n- [fact] artemyvo said x") == prose
    assert sanitize_gist(prose + "\n\n***\n\n**RAG**\n- [ask:user] did he?") == prose
    # A leading narrative header is dropped; an <eos> tail is cut.
    assert sanitize_gist("### Reflection on Session\n\n" + prose) == prose
    assert sanitize_gist(prose + "\n<eos>\n## RESOLVED\n- x") == prose
    # A fully structured summary salvages nothing → rejected, not stored/indexed.
    assert sanitize_gist("## WEIGHTS\n- [fact] a\n\n## RAG\n- [fact] b") == ""
    assert sanitize_gist("") == "" and sanitize_gist("too short") == ""
    # The worklog line adds only presentation on top of the same salvage (now shared with
    # the foreground freeze path — see core.chat_worklog, self-tested there too).
    from core.chat_worklog import summarize_gist
    assert summarize_gist(prose + "\n\n## RAG\n- [fact] a") == prose
    assert summarize_gist("## RAG\n- [fact] a") == ""
    assert summarize_gist("x " * 400).endswith("…")

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir(parents=True, exist_ok=True)

        def _write_chat(stem: str, *, exchanges: int = 2, initiated_by: str = "",
                        reflected_at: str = "", chat_reflected: str = "") -> None:
            ex = [{"user_prompt": "hi", "assistant_response": "hello"}
                  for _ in range(exchanges)]
            doc = {"exchanges": ex}
            if initiated_by:
                doc["initiated_by"] = initiated_by
            (chats / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
            side = {"schema_version": 1, "source_session": f"{stem}.json", "exchanges": {}}
            if reflected_at:
                side["reflected_at"] = reflected_at
            if chat_reflected:
                side["chat_reflected"] = chat_reflected
            if reflected_at or chat_reflected:
                (chats / f"{stem}.state.json").write_text(json.dumps(side), encoding="utf-8")

        _write_chat("20200101_000001")                                   # fresh → eligible
        _write_chat("20200101_000002", reflected_at="x")                 # fully frozen → no
        _write_chat("20200101_000003", chat_reflected="x")               # stage one → no
        _write_chat("20200101_000004", exchanges=1, initiated_by="ava")  # unanswered → no
        _write_chat("20200101_000005")                                   # fresh → eligible

        _CHATS_DIR = chats
        _DATA_DIR = Path(td)
        _RUNS_DIR = Path(td) / "runs"
        _host_busy = lambda: False

        backlog = list_backlog()
        assert backlog == ["20200101_000001.json", "20200101_000005.json"], backlog

        # ── rung 1: which chats are missing a gist / fact record ──────────────── #
        # Unlike the reflection backlog this is NOT gated on the freeze — an old chat
        # reflected before these artifacts existed is exactly what needs backfilling — but
        # it does drop the unanswered opener (rung 0 deletes it) and it is oldest-first.
        sc = ChatSidecar(chats)
        sidecars = list_sidecar_backlog()
        assert [e["session"] for e in sidecars] == [
            "20200101_000001.json", "20200101_000002.json",
            "20200101_000003.json", "20200101_000005.json"], sidecars
        assert all(e["summary"] and e["facts"] for e in sidecars), sidecars

        # A chat with one of the pair already on disk asks only for the other.
        sc.write_summary(source_session="20200101_000002.json", summary=prose, run_id="t")
        assert [(e["session"], e["summary"], e["facts"]) for e in list_sidecar_backlog()
                if e["session"] == "20200101_000002.json"] == [
                    ("20200101_000002.json", False, True)]

        # A STALE fact record counts as wanted, not as present. This is what makes a
        # parser fix self-healing: several boxes run this codebase over different corpora,
        # so which files are damaged is per-box and cannot be listed in advance. A record
        # whose markers the writing parser could not see (they are still in its text) is
        # re-derived by the same one generation a missing sidecar costs.
        _s3 = "20200101_000003.json"
        sc.write_summary(source_session=_s3, summary=prose, run_id="t")
        sc.write_facts(source_session=_s3, run_id="t", facts=[
            {"subject": "", "subject_raw": "", "fact_class": "unspecified", "entities": [],
             "when": "", "text": "Likes tea. (about: Artemy) (class: standing)"}])
        assert [(e["summary"], e["facts"]) for e in list_sidecar_backlog()
                if e["session"] == _s3] == [(False, True)], "stale facts must be re-derived"
        # ...and a well-formed record of the same fact does not come back for another pass.
        sc.write_facts(source_session=_s3, run_id="t",
                       facts=chat_facts.parse_facts(
                           "[fact] (about: Artemy) (class: standing) Likes tea."))
        assert [e["session"] for e in list_sidecar_backlog()
                if e["session"] == _s3] == [], "a clean record must not be regenerated"
        # Hand the fixture back as it was — the ladder assertions below count this chat.
        for _suffix in (".summary.json", ".facts.json"):
            (chats / (_s3[:-len(".json")] + _suffix)).unlink()
        assert [(e["summary"], e["facts"]) for e in list_sidecar_backlog()
                if e["session"] == _s3] == [(True, True)]

        # Mocked modules registry: the production path generates through
        # `modules.run_module_blocking` and this module writes what it returns, so faking
        # the generation exercises the real sink (ChatSidecar) and the real ladder.
        from core import modules as _modules
        _real_run_module = _modules.run_module_blocking
        produced: list[tuple[str, str]] = []

        def _fake_module(*, module, filename, **kw):
            produced.append((module, filename))
            if module == "chat_summary":
                return {"ok": True, "text": prose, "records": []}
            return {"ok": True, "records": [{"subject": "artemy", "text": "likes tea",
                                             "fact_class": "standing"}]}

        _modules.run_module_blocking = _fake_module

        class _Model:  # _runtime.model truthy
            pass
        _runtime.model = _Model()
        _reflect_one_chat_fn = lambda *a, **k: "completed"   # replaced below
        refreshed: list[int] = []
        _refresh_chats_fn = lambda: refreshed.append(1)

        # Rung 1 has work ⇒ the wake stops there and reflection does NOT run.
        result = run_background_reflection_blocking()
        assert result.get("backfilled") == 4, result
        assert result.get("summaries") == 3 and result.get("facts") == 4, result
        assert "reflected" not in result, result
        assert refreshed == [1], "gists landed but the chat index was never rebuilt"
        assert sc.summary_text("20200101_000001.json") == prose
        assert len(sc.read_facts("20200101_000005.json").get("facts") or []) == 1
        # The already-summarized chat paid only for its missing half.
        assert ("chat_summary", "20200101_000002.json") not in produced, produced
        assert list_sidecar_backlog() == [], list_sidecar_backlog()

        # A chat whose passes produce nothing writable must not gate rung 2 forever.
        _write_chat("20200101_000006")
        _modules.run_module_blocking = lambda **kw: {"ok": True, "text": "", "records": []}
        assert [e["session"] for e in list_sidecar_backlog()] == ["20200101_000006.json"]
        result = run_background_reflection_blocking()
        assert result.get("backfilled") == 0, result
        assert "20200101_000006.json" in _backfill_unproductive
        # It is still genuinely missing its sidecars — but it is no longer asked for, so
        # the next wake falls through to reflection instead of retrying it forever.
        assert [e["session"] for e in list_sidecar_backlog()] == ["20200101_000006.json"]

        # The kill switch drops rung 1 out of the ladder entirely.
        _load_server_config = lambda: {"background_reflection": {"sidecar_backfill": False}}
        assert _backfill_enabled() is False
        _load_server_config = None
        _modules.run_module_blocking = _real_run_module

        # Mock the engine the way production works: the per-chat pass freezes the chat in a
        # THROWAWAY staging dir and commits that frozen sidecar into the checkpoint (via the
        # on_chat_reflected callback) — it never stamps the LIVE sidecar. So a second wake
        # must NOT re-pick it purely off the checkpoint marker (regression guard for the
        # "same chat reflected over and over" bug). Honor a preempt requested mid-drain.
        drained: list[str] = []

        def _fake_reflect(config, store, staging_dir, on_chat_reflected):
            fn = config.selected_sessions[0]
            drained.append(fn)
            # Freeze inside the throwaway staging (as ReflectionRunner would), so
            # commit_background_chat has a chat_reflected sidecar to fold into the checkpoint.
            stem = fn[:-5]
            stg_chats = Path(staging_dir) / "chats"
            stg_chats.mkdir(parents=True, exist_ok=True)
            (stg_chats / f"{stem}.state.json").write_text(
                json.dumps({"schema_version": 1, "source_session": fn, "exchanges": {}}),
                encoding="utf-8")
            ChatSidecar(stg_chats).mark_chat_reflected(fn)
            on_chat_reflected(fn, [], [])
            if len(drained) == 1:
                request_preempt()   # user returns after the first chat
            return "completed"

        _reflect_one_chat_fn = _fake_reflect

        # Rung 1 is now empty, so this wake falls through to reflection (rung 2).
        result = run_background_reflection_blocking()
        assert result.get("reflected") == 1, result
        assert result.get("status") == "preempted", result
        assert drained == ["20200101_000001.json"], drained
        assert not _background_reflection_active and _bg_run_id is None

        # Second wake: the drained chat is now chat_reflected → the rest remain.
        assert list_backlog() == ["20200101_000005.json",
                                  "20200101_000006.json"], list_backlog()

        # ── rung 0: the stale sweep runs on every wake, ahead of everything else ── #
        swept: list[int] = []

        def _fake_sweep():
            swept.append(1)
            return [{"session": "20200101_000004.json"}]

        _stale_sweep_fn = _fake_sweep
        _preempt_requested = False
        result = run_background_reflection_blocking()
        assert swept == [1] and result.get("deleted") == 1, (swept, result)

        # ...including a wake with nothing else to do at all: that is a real outcome
        # (something was deleted), not a "nothing to do" skip. The one chat still missing
        # its sidecars is the unproductive one, which no longer holds the ladder open.
        assert list_backlog() == [], list_backlog()
        assert [e["session"] for e in list_sidecar_backlog()] == ["20200101_000006.json"]
        result = run_background_reflection_blocking()
        assert result.get("deleted") == 1 and "skipped" not in result, result

        # ...and with nothing stale either, the wake honestly skips.
        _stale_sweep_fn = None
        assert run_background_reflection_blocking().get("skipped") == "no_unreflected_chats"

        # A BOX-level refusal (no model loaded, prompt file missing) stops the drain and
        # reports a skip, so the scheduler retries promptly — and, unlike a pass that ran
        # and produced nothing, it must not retire the chat it never generated for.
        _write_chat("20200101_000007")
        _modules.run_module_blocking = lambda **kw: {"skipped": "no_prompt"}
        result = run_background_reflection_blocking()
        assert result.get("skipped") == "backfill refused: no_prompt", result
        assert "20200101_000007.json" not in _backfill_unproductive
        _modules.run_module_blocking = _real_run_module

    _runtime.model = None
    _stale_sweep_fn = _refresh_chats_fn = None
    _backfill_unproductive.clear()
    print("background_reflection selftest: OK")


if __name__ == "__main__":
    _selftest()
