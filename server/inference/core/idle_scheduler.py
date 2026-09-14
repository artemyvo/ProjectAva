"""Generic idle-job scheduler for the inference server.

The GPU box has a single executor thread and, between conversations, a handful of
autonomous jobs that want to use it: the ambient *wander* (read a random article),
Ava-initiated *outreach* (raise an open ask), and *synthesis* (re-read an aged chat and
ask what she now wonders). Historically each was a bespoke ``_maybe_autonomous_*``
coroutine in ``server.py`` that

  * re-derived the same "is any other job running?" mutual-exclusion check as an OR over
    *every* sibling's ``_active`` flag (so adding a job meant editing that OR-chain in
    ~8 places, and forgetting one was a silent GPU race), and
  * shared ONE hard-coded 1-hour cadence — there was no way to run different events at
    different frequencies.

This module owns the *mechanism* so the jobs carry only *policy*:

  * **Each job is a fully independent task** with its own clock. Two jobs that come due
    on the same tick BOTH fire — the only thing between them is the GPU. There is no
    registration-order priority, no shared cadence, and no way for one job to suppress a
    sibling: a job's ``ready``/``consumed`` policy governs *that job only*.
  * **Serialization is by GPU access alone.** A single ``asyncio.Lock`` is the box's GPU
    mutex; co-due jobs queue on it and run back-to-back. ``host_busy()`` is
    ``gpu_busy() or external_busy()``, where ``external_busy`` covers the GPU owners that
    are NOT idle jobs (reflection runs, encounters, the manual debug triggers). A new
    idle job registers here and is automatically mutually-exclusive with every sibling —
    zero edits to the others.
  * Each job declares its **own** ``interval_s`` (and may override the shared box-idle
    precondition via ``idle_seconds``), so different events run at different frequencies.

Cross-job *policy* coupling, where it is genuinely wanted, does NOT belong here. The
rate limit on Ava's unprompted reach-outs is the worked example: it used to be an idle-job
``ready`` gate, which let whichever job was registered first monopolize the window and
starve the rest. It now lives in ``core.reachout_gate`` and is checked by each job body
immediately before it writes a session — so every job still fires, still runs its pass,
and still consumes its interval; only the outward message is throttled.

Jobs are plain :class:`IdleJob` descriptors registered at startup; :func:`run_loop` spawns
one supervising task per job. The job *bodies* (``run_*_blocking``) stay in their own
modules — this module never imports them; ``server.main()`` wires everything via
:func:`configure` + :func:`register`. GPU-free self-test: ``python -m core.idle_scheduler``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core import activity_log


@dataclass
class IdleJob:
    """One autonomous job the scheduler runs on the shared GPU executor.

    ``run`` is the existing blocking entry point (e.g. ``til_wander.
    run_autonomous_wander_blocking``); it returns a small summary dict and runs on the
    executor thread, so it serializes with chat/reflection generation. The scheduler
    owns mutual exclusion and the wake-lock; the job body owns everything else.

    Every field here scopes to THIS job. Nothing a job declares can gate a sibling —
    that independence is the point of the descriptor.
    """

    name: str
    #: minimum seconds between two runs of THIS job (its own frequency)
    interval_s: float
    #: blocking body, runs on the executor thread → returns a summary dict
    run: Callable[[], dict]
    #: optional extra gate checked just before dispatch (e.g. wander's token budget).
    #: Must depend only on THIS job's own preconditions: a gate that consults a sibling's
    #: state re-introduces the starvation this scheduler exists to prevent (a job whose
    #: `ready` is false never advances `_last_ran`, so it retries forever while the
    #: sibling holding the gate shut re-arms on its own, shorter interval).
    ready: Optional[Callable[[], bool]] = None
    #: optional classifier: did this attempt actually spend the interval? A cheap
    #: "nothing to do" bail (e.g. outreach has no open ask) can return False so the job
    #: retries on the next poll instead of waiting a full interval. Default: always True.
    consumed: Optional[Callable[[dict], bool]] = None
    #: optional per-job override of the shared "box has been idle this long" precondition
    #: (seconds). None ⇒ the scheduler-wide `idle_seconds`. A job that should fire more
    #: often than the global idle window needs its own, shorter value here.
    idle_seconds: Optional[float] = None
    #: optional one-liner describing a *successful* outcome for the unified activity log
    #: (`core.activity_log`), given this run's summary dict. Keeps the human phrasing next
    #: to the job registration (the scheduler stays semantics-free); `None` falls back to a
    #: generic "<name> finished". Skips/errors are phrased generically from their reason.
    describe: Optional[Callable[[dict], str]] = None
    #: monotonic timestamp of the last run that counted; 0.0 == never run
    _last_ran: float = field(default=0.0, repr=False)


# ── module state ────────────────────────────────────────────────────────────────
_jobs: list[IdleJob] = []
_last_activity: float = time.monotonic()  # shared idle clock (reset by real activity)

# The box's GPU mutex — the ONLY thing serializing idle jobs. Co-due jobs queue on it and
# run back-to-back rather than one losing its slot. Created lazily so this module imports
# without a running event loop (the GPU-free self-test never builds one).
_gpu_lock: Optional[asyncio.Lock] = None

# ── configured at startup (see configure) ───────────────────────────────────────
_executor = None
_external_busy: Callable[[], bool] = lambda: False   # reflection run / encounter active
_model_loaded: Callable[[], bool] = lambda: False
_lock_file: Optional[Path] = None
_idle_seconds: float = 3600.0
_poll_seconds: float = 300.0
_startup_note: str = ""


def configure(*, executor, external_busy: Callable[[], bool],
              model_loaded: Callable[[], bool], lock_file,
              idle_seconds: float = 3600.0, poll_seconds: float = 300.0,
              startup_note: str = "") -> None:
    """Wire the scheduler with the server capabilities it needs (it never imports
    ``server``). ``external_busy`` reports the user-triggered GPU owners that are not
    idle jobs (reflection runs, encounters); ``lock_file`` is the crash-safe PID
    wake-lock path shared with any other process on the box."""
    global _executor, _external_busy, _model_loaded, _lock_file
    global _idle_seconds, _poll_seconds, _startup_note
    _executor = executor
    _external_busy = external_busy
    _model_loaded = model_loaded
    _lock_file = Path(lock_file)
    _idle_seconds = float(idle_seconds)
    _poll_seconds = float(poll_seconds)
    _startup_note = startup_note


def register(job: IdleJob) -> None:
    """Add an idle job. Jobs are attempted in registration order each tick, but each
    self-throttles on its own ``interval_s`` — order only decides who gets the single
    GPU slot first within a tick when several are due at once."""
    _jobs.append(job)


def mark_activity() -> None:
    """Reset the shared idle clock — call on anything that means the box is NOT idle
    (a user chat turn, a reflection-run boundary, a manual GPU op). Autonomous idle jobs
    deliberately do NOT call this: each self-throttles on its own interval, so a wander
    can no longer postpone outreach/synthesis (the old shared-clock coupling), and the
    generation a job does while processing its own event never counts as user activity."""
    global _last_activity
    _last_activity = time.monotonic()


def gpu_busy() -> bool:
    """True while an autonomous idle job holds the GPU."""
    return _gpu_lock is not None and _gpu_lock.locked()


def host_busy() -> bool:
    """True while any autonomous idle job OR a user-triggered GPU owner holds the box.

    This single predicate replaces the per-module OR-of-every-sibling ``host_busy``
    lambdas — every subsystem is now injected THIS function."""
    return gpu_busy() or _external_busy()


# ── crash-safe PID wake-lock ─────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    """True if *pid* is a live process (signal 0 probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True      # exists but owned by another user — still alive
    except Exception:
        return False
    return True


def _acquire_wake_lock() -> bool:
    """Take the wake-lock; reclaim a stale one left by a crashed process.

    Crash-safe: the lockfile records the holder's PID, so a lock left behind by a server
    that died mid-job is detected (its PID is gone) and reclaimed rather than wedging the
    heartbeat forever. Returns True on acquisition."""
    if _lock_file is None:
        return False
    try:
        if _lock_file.exists():
            try:
                held = json.loads(_lock_file.read_text(encoding="utf-8"))
            except Exception:
                held = {}
            pid = int(held.get("pid", 0) or 0)
            if pid and pid != os.getpid() and _pid_alive(pid):
                return False     # another live process holds it
            # else: our own stale lock, or a dead process's — reclaim it below
        _lock_file.parent.mkdir(parents=True, exist_ok=True)
        _lock_file.write_text(
            json.dumps({"pid": os.getpid(), "ts": datetime.now().isoformat()}),
            encoding="utf-8")
        return True
    except Exception:
        return False


def _release_wake_lock() -> None:
    if _lock_file is None:
        return
    try:
        _lock_file.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def clear_stale_lock() -> None:
    """At startup, clear a lock this box can prove is dead so it never wedges the
    heartbeat while idle. (``_acquire_wake_lock`` also reclaims stale locks, but clearing
    here keeps the file honest before the first job even runs.)"""
    if _lock_file is None:
        return
    try:
        if _lock_file.exists():
            held = json.loads(_lock_file.read_text(encoding="utf-8"))
            if not _pid_alive(int(held.get("pid", 0) or 0)):
                _release_wake_lock()
    except Exception:
        pass


# ── heartbeat ────────────────────────────────────────────────────────────────────

def _gpu_lock_handle() -> asyncio.Lock:
    """The process-wide GPU mutex, created on first use (needs no running loop in 3.10+,
    but staying lazy keeps import and the GPU-free self-test loop-free)."""
    global _gpu_lock
    if _gpu_lock is None:
        _gpu_lock = asyncio.Lock()
    return _gpu_lock


def _job_idle_ok(job: IdleJob) -> bool:
    """True if the box has been free of *real* activity long enough for *job*.

    Per-job, so a future high-frequency job can use a shorter window than its slower
    siblings. Only real activity (user turns, reflection boundaries, manual GPU ops)
    moves this clock — a sibling idle job running does not."""
    window = _idle_seconds if job.idle_seconds is None else job.idle_seconds
    return (time.monotonic() - _last_activity) >= window


def _job_due(job: IdleJob) -> bool:
    """True if *job*'s own interval has elapsed and its optional ``ready`` gate passes."""
    if (time.monotonic() - job._last_ran) < job.interval_s:
        return False
    if job.ready is not None:
        try:
            if not job.ready():
                return False
        except Exception:
            traceback.print_exc()
            return False
    return True


# Last terminal *skip* message logged per job — a job that finds nothing to do fires on
# its own cadence (a no-candidate outreach retries every poll), so we log a given skip
# once and suppress consecutive identical repeats until the outcome changes. A real
# outcome (finished/failed) or a *different* skip always logs.
_last_skip: dict[str, str] = {}


def _summarize(job: IdleJob, result) -> tuple[str, str]:
    """Turn a job's return value into an ``(activity_kind, message)`` for the unified log.

    Generic keys (`error`, `skipped`) are phrased here so the scheduler stays semantics-
    free; a *successful* run is phrased by the job's own ``describe`` (registered next to
    the job), falling back to a plain "<name> finished"."""
    if not isinstance(result, dict):
        return "finished", f"{job.name} finished"
    if result.get("error"):
        return "failed", f"{job.name} failed: {result['error']}"
    if result.get("skipped"):
        return "skipped", f"{job.name} skipped: {result['skipped']}"
    if job.describe is not None:
        try:
            msg = job.describe(result)
            if msg:
                return "finished", msg
        except Exception:
            traceback.print_exc()
    return "finished", f"{job.name} finished"


def _emit_terminal(job: IdleJob, aid: str, result) -> None:
    """Log a job's outcome to the unified activity journal, deduping quiet repeat skips."""
    kind, message = _summarize(job, result)
    if kind == "skipped":
        if _last_skip.get(job.name) == message:
            return           # same "nothing to do" as last time — don't re-log
        _last_skip[job.name] = message
    else:
        _last_skip.pop(job.name, None)
    activity_log.append(job.name, kind, message, activity_id=aid)


def _labelled_run(job: IdleJob):
    """Run ``job.run`` with the activity journal's ambient pass label set to this job.

    It has to happen HERE, inside the executor call, rather than beside ``set_current`` in
    :func:`_dispatch`: ``activity_log.set_ambient_label`` is *thread-local*, ``_dispatch``
    is a coroutine on the asyncio loop thread, and every generation the job body runs
    happens on the (single) executor thread. Setting it from the coroutine therefore
    labelled a thread that never generates, so the reset at the job boundary — the
    load-bearing half — never landed, while the finer labels the job bodies set for
    themselves (``til_wander``'s ``til_facts:<kind>``, ``modules``' ``module:<name>``) DID,
    from inside the executor, and then stuck to that thread for the life of the process.

    The result was cross-job mislabelling in the Activity tab: on a live box every outreach
    and check-in generation was reporting under ``til_facts:wander`` or ``chat_facts``,
    i.e. as the pass of whichever job last named itself. Clearing on the way out likewise
    keeps a job's label off the passes live chat runs on this same thread (stage-1 fact
    fetch), which name themselves via ``activity_log.pass_context``."""
    activity_log.set_ambient_label(job.name)
    try:
        return job.run()
    finally:
        activity_log.set_ambient_label(None)


async def _dispatch(job: IdleJob) -> None:
    """Run one job on the executor, holding the GPU lock + wake-lock, then record whether
    the attempt spent the job's interval.

    The lock is the whole serialization story: a co-due sibling simply waits here rather
    than losing its turn. Because that wait can be long (a wander pass is minutes), every
    precondition is re-checked *after* acquisition — the user may have started typing, or
    a reflection run may have claimed the box, while this job sat in the queue.

    Activity reporting (`core.activity_log`): the box's "current activity" chip is opened
    before the run and cleared after (so the UI shows "wander running" for the whole pass),
    while the journal gets one terminal line — plus whatever intra-run progress the job's
    body streams through the hooks wired at registration, which default to this same
    activity id via the current-activity pointer. The per-pass *label* is set on the
    executor thread instead (see :func:`_labelled_run`), which is the only thread it can
    reach the generations from."""
    async with _gpu_lock_handle():
        if _external_busy():
            return          # a reflection run / encounter claimed the box while we queued
        if not _model_loaded() or not _job_idle_ok(job) or not _job_due(job):
            return          # box went warm, or the job stopped being eligible, mid-wait
        if not _acquire_wake_lock():
            return          # another process on the box holds it — retry next poll
        loop = asyncio.get_event_loop()
        consumed = True
        aid = activity_log.set_current(job.name, f"{job.name} running")
        try:
            result = await loop.run_in_executor(_executor, _labelled_run, job)
            if isinstance(result, dict):
                if result.get("skipped"):
                    print(f"[{job.name}] autonomous skipped: {result['skipped']}",
                          flush=True)
                if job.consumed is not None:
                    consumed = bool(job.consumed(result))
            _emit_terminal(job, aid, result)
        except Exception as e:
            traceback.print_exc()
            print(f"[{job.name}] autonomous failed: {e}", flush=True)
            consumed = True
            _last_skip.pop(job.name, None)
            activity_log.append(job.name, "failed",
                                f"{job.name} failed: {type(e).__name__}: {e}",
                                activity_id=aid)
        finally:
            activity_log.clear_current()
            _release_wake_lock()
            if consumed:
                # Advance the per-job clock so the next run is >= interval_s out. A cheap
                # "nothing to do" bail (consumed→False) leaves _last_ran untouched so the
                # job retries on the next poll rather than sleeping a full interval.
                job._last_ran = time.monotonic()


async def _job_loop(job: IdleJob) -> None:
    """One job's independent supervising task: poll its own clock, dispatch when due.

    Each job gets one of these, so a job is never skipped because a sibling ran first or
    because a sibling's tick ended early. A job polls at its own granularity — no faster
    than the scheduler-wide ``poll_seconds``, and no slower than its own interval, so a
    short-interval job stays responsive without the others busy-waiting."""
    poll = max(1.0, min(_poll_seconds, job.interval_s))
    while True:
        await asyncio.sleep(poll)
        try:
            if not _model_loaded():
                continue
            if not _job_idle_ok(job):
                continue      # box still warm from real activity — leave it alone
            if not _job_due(job):
                continue
            await _dispatch(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()


async def run_loop() -> None:
    """Spawn one independent task per registered job and supervise them forever.

    There is no shared tick: each job owns its clock and fires the moment it is due, on
    its own frequency. Jobs that come due together all fire — they queue on the GPU lock
    in :func:`_dispatch` and run back-to-back."""
    jobs_desc = ", ".join(f"{j.name}@{int(j.interval_s)}s" for j in _jobs) or "(none)"
    note = f" — {_startup_note}" if _startup_note else ""
    print(f"[idle] scheduler started (idle>={int(_idle_seconds)}s, "
          f"poll {int(_poll_seconds)}s; independent jobs: {jobs_desc}){note}", flush=True)
    _gpu_lock_handle()
    await asyncio.gather(*(asyncio.create_task(_job_loop(job), name=f"idle-{job.name}")
                           for job in _jobs))


# ── GPU-free self-test ───────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise the scheduling logic (due/idle/consumed/host_busy) plus the independence
    property that motivates this module: co-due jobs BOTH run, serialized only by the GPU
    lock, in an order no registration position can fix. Run: ``python -m core.idle_scheduler``."""
    import tempfile

    global _last_activity, _gpu_lock, _executor
    _jobs.clear()

    with tempfile.TemporaryDirectory() as td:
        configure(
            executor=None,
            external_busy=lambda: False,
            model_loaded=lambda: True,
            lock_file=Path(td) / "wake.lock",
            idle_seconds=1.0, poll_seconds=0.0,
        )
        # interval gating: a never-run job is due; one just run is not.
        j = IdleJob(name="j", interval_s=100.0, run=lambda: {})
        register(j)
        assert _job_due(j), "never-run job should be due"
        j._last_ran = time.monotonic()
        assert not _job_due(j), "just-run job should not be due"

        # ready gate suppresses an otherwise-due job.
        j._last_ran = 0.0
        j.ready = lambda: False
        assert not _job_due(j), "ready()=False should suppress"
        j.ready = None

        # per-job idle override: a job may demand less quiet than the scheduler default.
        _last_activity = time.monotonic()
        assert not _job_idle_ok(j), "fresh activity should fail the 1s default window"
        assert _job_idle_ok(IdleJob("fast", 1.0, run=lambda: {}, idle_seconds=0.0)), \
            "idle_seconds=0 should ignore the shared idle window"
        _last_activity = 0.0

        # host_busy reflects both the GPU lock and the external predicate.
        _gpu_lock = None
        assert not host_busy()
        _external_busy_saved = _external_busy
        _set_external(lambda: True)
        assert host_busy(), "external_busy must surface in host_busy"
        _set_external(_external_busy_saved)

        # consumed classifier: False means "retry next poll" (leave _last_ran).
        assert (IdleJob("x", 1.0, run=lambda: {}, consumed=lambda r: False)
                .consumed({"skipped": "no_candidates"}) is False)

        # Independence: two jobs due at the same instant both dispatch, and each holds the
        # GPU alone while it runs. This is the regression the old serial for-loop had —
        # there, a job could lose its turn to a sibling registered ahead of it.
        _jobs.clear()
        _gpu_lock = None
        order: list[str] = []
        unlocked: list[str] = []

        def _body(name: str) -> dict:
            # Runs on the executor thread while _dispatch holds the lock; `locked()` is a
            # plain bool read, so it is safe to probe from here.
            if not _gpu_lock_handle().locked():
                unlocked.append(name)
            order.append(name)
            return {}

        async def _drive() -> None:
            a = IdleJob("a", 100.0, run=lambda: _body("a"))
            b = IdleJob("b", 100.0, run=lambda: _body("b"))
            for job in (a, b):
                register(job)
            await asyncio.gather(_dispatch(a), _dispatch(b))
            assert sorted(order) == ["a", "b"], f"both co-due jobs must run, got {order}"
            assert a._last_ran > 0 and b._last_ran > 0, "both must advance their clocks"
            assert not _gpu_lock_handle().locked(), "lock must be released after dispatch"

        _last_activity = 0.0
        asyncio.run(_drive())
        assert not unlocked, f"jobs must only run under the GPU lock; {unlocked} did not"

        # Pass labels must not leak across the job boundary. `set_ambient_label` is
        # thread-local and sticky, the box has ONE executor thread, and job bodies name
        # their own passes from inside it — so a reset applied on the asyncio thread (as
        # `_dispatch` used to) never lands where the generations are, and every later job
        # reports under the previous one's label. Observed live: outreach and check-in
        # passes journalled as `til_facts:wander`. Driven through a single-worker executor
        # because that is the condition the bug needs.
        from concurrent.futures import ThreadPoolExecutor
        _jobs.clear()
        _gpu_lock = None
        labels: dict = {}
        pool = _executor = ThreadPoolExecutor(max_workers=1)

        def _namer() -> dict:                    # a body that names itself, e.g. til_wander
            labels["namer"] = activity_log._ctx_top().get("label")
            activity_log.set_ambient_label("til_facts:wander")
            return {}

        def _next() -> dict:                     # the job that runs after it
            labels["next"] = activity_log._ctx_top().get("label")
            return {}

        async def _drive_labels() -> None:
            for job in (IdleJob("namer", 100.0, run=_namer),
                        IdleJob("next", 100.0, run=_next)):
                register(job)
                await _dispatch(job)

        asyncio.run(_drive_labels())
        pool.shutdown(wait=True)
        assert labels["namer"] == "namer", \
            f"a job's own name must label its passes, got {labels['namer']!r}"
        assert labels["next"] == "next", \
            f"the previous job's label leaked onto the next: {labels['next']!r}"

    _jobs.clear()
    _gpu_lock = None
    print("idle_scheduler selftest: OK")


def _set_external(fn: Callable[[], bool]) -> None:
    global _external_busy
    _external_busy = fn


if __name__ == "__main__":
    _selftest()
