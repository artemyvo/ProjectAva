"""Unified activity journal — the single, box-wide "what is Ava doing right now" log.

Motivation: reporting used to be per-run and per-subsystem. A reflection run streamed
detailed events keyed to a ``run_id`` the client had to already hold; the five autonomous
idle jobs (wander / outreach / synthesis / checkin / background_reflection) reported only
via ``print()`` to ``server.log``, which no client polls. So the moment a reflection run
ended and the box moved on to autonomous work — or an idle job fired on its own with no
run in progress — the UI went silent even though the GPU was plainly busy.

This module is the ONE sink every GPU subsystem writes to. It is a *leaf*: it imports
nothing from the project and is imported directly (no configure-injection), exactly like
:mod:`core.reachout_gate`, so any subsystem can ``from core import activity_log;
activity_log.append(...)`` with no wiring and no import cycle. That leaf-ness is also what
lets the **offline train cycle** — a different process, launched by the watchdog while
inference is down — append to the same journal (see :func:`install_stdout_tee`).

Design:
  * ONE global, monotonic, 1-indexed ``seq`` across all sources. A client tracks a single
    cursor (``after_seq``) and gets everything Ava did, interleaved, regardless of
    subsystem — the whole point of a *single* log. ``seq`` is restart-durable (recovered
    from the journal tail on :func:`configure`), so a reconnecting UI resumes exactly
    where it left off instead of re-seeing the world.
  * Append-only, persisted to ``<data>/hot/activity/activity.jsonl``, bounded in memory
    (a ring of the most recent events) and rotated into numbered **segments** on disk.
  * :func:`current` returns the open (started-but-not-finished) activity so the status bar
    can render a live "🟢 Ava: wander" chip off the existing status poll, no extra RPC.

**Four levels** (``level`` on each record), because "one log everything appends to" spans
populations with wildly different volume and value:

  ``event``  lifecycle + phase markers. The original content of this journal.
  ``body``   ONE finished generation, verbatim: the pass's CoT *and* its output, plus the
             outcome flags (truncated / stopped-on-loop / tokens). Written by the two
             generation factories in :mod:`core.generation`, so every background pass on
             the box reports what it produced without a single call-site change.
  ``stream`` a heartbeat for an in-flight generation: elapsed, tokens so far, and a rolling
             tail of what is being written. This is what makes a 25-minute pass visible
             *while it runs* — the failure this level was added for (2026-08-11: a
             ``til_facts`` pass held the GPU for 21 minutes and emitted nothing until it
             finished, because :func:`set_current` deliberately writes no line).
  ``raw``    one stdout/stderr line from any process, incl. unsloth during a train cycle.

Per-token deltas are deliberately NOT journalled: a reflection run generates ~100k tokens,
and at the generator's ~80-char batching that is ~25k records per run, which would evict
everything else from the ring on every run. The heartbeat plus the final verbatim ``body``
carries the same information for ~1% of the records.

**Invariant — the public API contributes nothing.** ``core.api_http`` traffic must never
produce a ``body``/``stream``/``raw`` record: "requests are NEVER logged" is a structural
guarantee of that endpoint. It holds here by construction rather than by a check — the
hooks live in the *reflect* and *agentic* generate factories, and the API (like gossip and
live chat) runs through ``_make_openai_generate`` / ``_run_generation``, which have none.
Anything wiring a hook into those paths breaks the guarantee.

Concurrency: the box has a single GPU executor and idle jobs are serialized by the
scheduler's GPU lock, so at most one autonomous activity is in flight at a time; the
"current activity" pointer is therefore a safe single-writer global (:func:`set_current`
is called only from the scheduler's ``_dispatch`` and the reflection run body, which never
overlap — a reflection run is ``external_busy``, so no idle job dispatches during it). All
list/``seq`` mutation is under a lock so the asyncio loop thread (transport) and the
executor thread (job bodies + their progress hooks) can both append safely. Across
*processes* the append is ``flock``-guarded so a training line can never interleave into a
half-written server line; ``seq`` uniqueness rests on the watchdog stopping inference
before it runs an offline job (it never runs two writers at once), and each process
recovers the high-water from the tail on :func:`configure`.

GPU-free self-test: ``python -m core.activity_log``.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:                                     # POSIX only; the box is Linux, but keep the
    import fcntl                         # module importable anywhere (self-test, tools).
except Exception:                        # pragma: no cover
    fcntl = None                         # type: ignore[assignment]

# Event kinds. `started` opens a burst (a chip, not a journal line — see set_current);
# `progress`/`note` are intra-burst; `finished`/`skipped`/`failed` close it.
_KINDS = {"started", "progress", "note", "result", "skipped", "failed", "finished"}
_TERMINAL = {"finished", "skipped", "failed"}

# Levels, coarsest first. A reader asks for a set of these; `event` is the floor every
# client gets. See the module docstring for what each carries.
LEVELS = ("event", "body", "stream", "raw")

_RING = 1200               # in-memory recent events kept for fast `after_seq` replay
                           # (was 2000 pre-levels; a `body` carries kilobytes, so the ring
                           # is sized by bytes-in-RAM now, not by event count)

# ── tunables (overridable via configure(**opts) ← server_config.json "logging") ──────
_DEFAULTS = {
    "heartbeat_s": 20.0,          # 0 ⇒ no `stream` records (restores pre-2026-08-11 silence)
    "body_max_chars": 16000,      # a longer generation is elided in the MIDDLE, never
                                  # clipped at the end — the answer after </think> is the
                                  # part most worth keeping, and an end-clip drops exactly it
    "stream_tail_chars": 240,     # rolling tail carried by a heartbeat
    "segment_bytes": 16_000_000,  # rotate the journal past this
    "retain_segments": 8,         # ...keeping this many rotated segments (oldest pruned)
    "stream_enabled": True,
    "raw_enabled": True,          # the stdout tee
    "raw_max_lines_per_s": 200,   # past this, coalesce into one "+N lines suppressed"
    "raw_max_line_chars": 2000,
    "tee_denylist": (),           # extra regexes dropped by the tee
}

# Lines the tee drops by default: blank, and the box's known table-art noise (the
# BertModel LOAD REPORT separator the embedder prints on every RAG rebuild).
_TEE_DROP_DEFAULT = (
    re.compile(r"^\s*$"),
    re.compile(r"^[-+|=\s]{8,}$"),
)

_lock = threading.RLock()
_events: deque = deque(maxlen=_RING)
_seq = 0
_path: Optional[Path] = None
_opts = dict(_DEFAULTS)
# The open activity ({source, activity_id, message, phase, since}) or None. Drives the
# live status chip and supplies the default activity_id for append()s that omit one.
_current: Optional[dict] = None
# In-flight generations, keyed by pass_id (see begin_pass). One at a time in practice
# (single GPU executor), but a dict costs nothing and keeps a stray nested pass honest.
_passes: dict = {}
# Label stack for the generation seam, per thread (the executor thread runs one pass at a
# time; the asyncio thread never generates).
_ctx = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elide(text: str, cap: int) -> str:
    """Cap *text* by removing its MIDDLE, keeping head and tail. See `body_max_chars`."""
    if cap <= 0 or len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return f"{text[:head]}\n… [+{len(text) - cap} chars elided] …\n{text[-tail:]}"


# ── lifecycle ─────────────────────────────────────────────────────────────────────

def configure(path, **opts) -> None:
    """Point the journal at its file and recover the ``seq`` high-water + recent window
    from the tail. Call once at startup (in EVERY process that appends). Safe if the file
    does not exist yet.

    ``opts`` overrides the module tunables (see ``_DEFAULTS``); unknown keys are ignored,
    so a config block may carry keys a older/newer server does not know.

    Only the tail is read (not the whole file): the journal is now allowed to reach
    ``segment_bytes``, and a full ``read_text()`` at boot would stall the server for as
    long as the log is large — the one cost that would make people turn logging off.
    """
    global _path, _seq
    with _lock:
        _path = Path(path)
        for k, v in (opts or {}).items():
            if k in _DEFAULTS and v is not None:
                _opts[k] = v
        try:
            _path.parent.mkdir(parents=True, exist_ok=True)
            for ev in _tail_records(_path, _RING):
                _events.append(ev)
                _seq = max(_seq, int(ev.get("seq", 0) or 0))
        except Exception:
            pass


def _tail_records(path: Path, limit: int, max_bytes: int = 4_000_000) -> list[dict]:
    """The last *limit* parseable records of *path*, reading at most *max_bytes* of tail."""
    out: list[dict] = []
    try:
        if not path.exists():
            return out
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()          # discard the partial first line
            blob = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return out
    for ln in blob.splitlines()[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


# ── writing ───────────────────────────────────────────────────────────────────────

def append(source: str, kind: str, message: str, *, activity_id: Optional[str] = None,
           phase: Optional[str] = None, level: str = "event",
           text: Optional[str] = None, pass_id: Optional[str] = None,
           **detail) -> dict:
    """Append one activity event and return it (with its assigned ``seq``).

    ``activity_id`` correlates the events of one burst; when omitted it falls back to the
    current open activity's id (set by :func:`set_current`), so a job's progress hooks can
    call ``append(source, "progress", msg)`` without threading an id through.

    ``message`` is the one-line headline a compact view renders; ``text`` is the payload a
    detail view expands (a pass's generation, a heartbeat's tail). They are separate
    fields so the headline never has to be clipped to bound the body.
    """
    global _seq
    if level not in LEVELS:
        level = "event"
    if level == "stream" and not _opts["stream_enabled"]:
        return {}
    if level == "raw" and not _opts["raw_enabled"]:
        return {}
    with _lock:
        _seq += 1
        aid = (activity_id or (_current or {}).get("activity_id")
               or f"{source}-{_seq}")
        ev = {
            "type": "activity_event",
            "seq": _seq,
            "ts": _utc_now(),
            "source": source,
            "activity_id": aid,
            "kind": kind if kind in _KINDS else "note",
            "phase": phase,
            "message": (message or "").strip(),
        }
        if level != "event":
            ev["level"] = level
        if pass_id:
            ev["pass_id"] = pass_id
        if text:
            ev["text"] = _elide(text.strip(), int(_opts["body_max_chars"]))
        if detail:
            ev["detail"] = detail
        _events.append(ev)
        _persist_locked(ev)
    return ev


def set_current(source: str, message: str = "", *, phase: Optional[str] = None) -> str:
    """Open a new activity for the live status chip WITHOUT writing a journal line, and
    return a fresh ``activity_id`` that subsequent :func:`append`\\ s default to.

    Deliberately does not emit a ``started`` event: a job that fires hourly and mostly
    finds nothing to do would otherwise flood the journal with started→skip pairs. The
    chip shows "running now"; the journal gets only the meaningful outcome (a terminal
    line, plus any progress the job's hooks emit). See the scheduler's ``_dispatch``.

    A job that runs *long* is no longer silent under this rule: its generation heartbeats
    (:func:`pass_tick`) announce it a few seconds in and then report continuously."""
    global _current
    aid = f"{source}-{uuid.uuid4().hex[:8]}"
    with _lock:
        _current = {"source": source, "activity_id": aid,
                    "message": (message or "").strip(), "phase": phase,
                    "since": _utc_now()}
    return aid


def clear_current() -> None:
    """Close the live chip (the burst finished). Idempotent."""
    global _current
    with _lock:
        _current = None


# ── the generation seam: begin_pass / pass_tick / end_pass ─────────────────────────
#
# Wired into `generation._make_sync_reflect_generate` and `_make_agentic_generate` — the
# two functions EVERY background generation on the box goes through (reflection passes,
# TIL/wander, outreach, check-in, synthesis, deliberation, modules, the clean-base
# evaluations). One hook there covers all of them with no call-site changes; the label is
# refined per pass by `pass_context` where a caller bothers, and falls back to the source
# where it does not, so labelling can improve incrementally without blocking anything.

@contextmanager
def pass_context(label: str, source: Optional[str] = None, **meta):
    """Name the pass the next generation belongs to (``with pass_context("chat_facts"):``).

    Thread-local and nestable; the innermost name wins. Optional because the seam falls
    back to the live activity chip's source — this only makes a line read
    "chat_facts" instead of "wander"."""
    stack = getattr(_ctx, "stack", None)
    if stack is None:
        stack = _ctx.stack = []
    stack.append({"label": label, "source": source, "meta": meta})
    try:
        yield
    finally:
        try:
            stack.pop()
        except Exception:
            pass


def _ctx_top() -> dict:
    stack = getattr(_ctx, "stack", None)
    if stack:
        return dict(stack[-1])
    amb = getattr(_ctx, "ambient", None)
    return {"label": amb} if amb else {}


def set_ambient_label(label: Optional[str]) -> None:
    """Name the pass the *next* generations on THIS thread belong to, until changed.

    The ambient sibling of :func:`pass_context`, for a caller that already announces its
    own phases and would otherwise need a ``with`` block around every generate call. The
    reflection runner is that caller: it emits a ``phase_started`` event per pass on the
    same (executor) thread the generation then runs on, so mirroring that phase here
    labels every reflection body — ``revision``, ``chat_facts``, ``anchor`` — from one
    edit instead of ~15. An explicit :func:`pass_context` still wins.

    **Thread-local, and STICKY until changed** — both halves matter to a caller. It must be
    set on the same thread the generation runs on (on this box, the single GPU executor);
    setting it from the asyncio loop thread labels a thread that never generates and is a
    silent no-op. And because nothing clears it, whoever sets it owns every later pass on
    that thread: a job that names itself and returns leaves its label on the next job's
    generations. That is what the reset in ``idle_scheduler._labelled_run`` is for, and why
    it lives inside the executor call rather than beside ``set_current``."""
    _ctx.ambient = (label or "").strip() or None


def begin_pass(label: str = "", source: str = "", *, phase: Optional[str] = None,
               **meta) -> str:
    """Open one generation. Returns a ``pass_id`` for :func:`pass_tick`/:func:`end_pass`.

    Writes NOTHING yet — the ``started`` line is *staged* and flushed by the first
    heartbeat. A pass that finishes inside ``heartbeat_s`` therefore never writes one
    (the anti-flood property :func:`set_current` was built for), while a pass that runs
    for minutes announces itself and then reports as it goes."""
    top = _ctx_top()
    label = label or top.get("label") or ""
    source = (source or top.get("source")
              or (_current or {}).get("source") or "server")
    label = label or source
    pid = f"p-{uuid.uuid4().hex[:8]}"
    with _lock:
        # Self-heal: a generation that raised between begin_pass and end_pass leaves its
        # state behind. Bounded and harmless, but prune it rather than grow forever.
        if len(_passes) > 8:
            cutoff = time.monotonic() - 3600
            for k in [k for k, v in _passes.items() if v["started"] < cutoff]:
                _passes.pop(k, None)
        _passes[pid] = {
            "label": label, "source": source, "phase": phase,
            "started": time.monotonic(), "announced": False,
            "last_tick": time.monotonic(), "meta": dict(meta),
            "activity_id": (_current or {}).get("activity_id"),
        }
    return pid


def pass_tick(pass_id: str, *, tokens: int = 0, tail: str = "") -> None:
    """Heartbeat for an in-flight generation. Cheap and rate-limited internally: call it
    per chunk and it emits at most one ``stream`` record every ``heartbeat_s``.

    The first emission also flushes the staged ``started`` line, so "something is running"
    and "here is what it is writing" arrive together."""
    hb = float(_opts["heartbeat_s"] or 0)
    if hb <= 0 or not _opts["stream_enabled"]:
        return
    with _lock:
        st = _passes.get(pass_id)
        if st is None:
            return
        now = time.monotonic()
        if (now - st["last_tick"]) < hb:
            return
        st["last_tick"] = now
        first = not st["announced"]
        st["announced"] = True
        elapsed = now - st["started"]
        label, source, aid = st["label"], st["source"], st["activity_id"]
        phase = st["phase"]
        meta = st["meta"]
    if first:
        budget = meta.get("max_new_tokens")
        extra = f", budget {budget} tok" if budget else ""
        append(source, "started", f"{label}: generating…{extra}",
               activity_id=aid, phase=phase, pass_id=pass_id, **{
                   k: v for k, v in meta.items() if v is not None})
    append(source, "progress",
           f"{label}: {tokens} tok, {int(elapsed)}s",
           activity_id=aid, phase=phase, level="stream", pass_id=pass_id,
           text=(tail[-int(_opts["stream_tail_chars"]):] if tail else None),
           tokens=tokens, elapsed_s=round(elapsed, 1))


def end_pass(pass_id: str, *, text: str = "", tokens: int = 0, **detail) -> None:
    """Close a generation and write its ``body`` — the CoT and the output, verbatim.

    An empty generation writes no body (there is nothing to show), but still closes the
    pass so its state is released."""
    with _lock:
        st = _passes.pop(pass_id, None)
    if st is None:
        return
    body = (text or "").strip()
    if not body:
        return
    elapsed = time.monotonic() - st["started"]
    label = st["label"]
    flags = []
    if detail.get("truncated"):
        flags.append("hit the token cap")
    if detail.get("stopped_on_loop"):
        flags.append("halted on the loop guard")
    suffix = f" — {', '.join(flags)}" if flags else ""
    append(st["source"], "result",
           f"{label}: produced {len(body)} chars in {int(elapsed)}s{suffix}",
           activity_id=st["activity_id"], phase=st["phase"], level="body",
           pass_id=pass_id, text=body, tokens=tokens,
           elapsed_s=round(elapsed, 1), **detail)


# ── the stdout tee ────────────────────────────────────────────────────────────────

class _Tee:
    """Wrap a text stream so every completed line is ALSO journalled at ``raw``.

    Why a tee rather than call-site edits: every subsystem on this box already prints in
    one shape — ``[til] …``, ``[wander] …``, ``[idle] …``, ``[background_reflection] …``,
    ``[watchdog …] …`` — so parsing that prefix converts the whole existing corpus of
    prints into journal lines with no call-site changes at all. It is also the ONLY way to
    capture a *foreign* library's output (unsloth/TRL during a train cycle), which is the
    other half of the requirement.

    The real stream is written through first and unchanged, so ``server.log`` is
    byte-identical to what it was."""

    _TAG = re.compile(r"^\[([a-z_][a-z0-9_ .:-]*)\]")

    def __init__(self, stream, source: str, extra: Optional[Callable[[str], None]] = None):
        self._s = stream
        self._source = source
        self._extra = extra
        self._buf = threading.local()      # per-thread partial line: no interleaving
        self._guard = threading.local()    # re-entrancy (a journal failure must not loop)
        self._win = [0.0, 0, 0]            # [window_start, emitted, suppressed]
        self._drop = list(_TEE_DROP_DEFAULT)
        for pat in (_opts.get("tee_denylist") or ()):
            try:
                self._drop.append(re.compile(pat))
            except Exception:
                pass

    # -- stream protocol (delegate everything we do not handle) ------------------- #
    def write(self, s):
        n = self._s.write(s)               # never lose the real stream, whatever follows
        try:
            self._capture(s)
        except Exception:
            pass
        return n

    def flush(self):
        return self._s.flush()

    def isatty(self):
        try:
            return self._s.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._s.fileno()

    def __getattr__(self, item):
        return getattr(self._s, item)

    # -- capture ------------------------------------------------------------------ #
    def _capture(self, s: str) -> None:
        if getattr(self._guard, "busy", False) or not s:
            return
        buf = getattr(self._buf, "v", "")
        buf += s
        # A progress bar redraws with \r and no newline; keep only the last redraw so a
        # tqdm sweep contributes one line instead of thousands.
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            self._emit(line.rstrip())
        # An unterminated buffer that has grown past a line cap is a bar mid-sweep; drop
        # its head so memory cannot grow without bound.
        cap = int(_opts["raw_max_line_chars"])
        if len(buf) > cap * 4:
            buf = buf[-cap:]
        self._buf.v = buf

    def _emit(self, line: str) -> None:
        if not line or any(p.search(line) for p in self._drop):
            return
        cap = int(_opts["raw_max_line_chars"])
        if len(line) > cap:
            line = _elide(line, cap)
        # Rate limit: a runaway loop must not be able to fill the disk through the log.
        now = time.monotonic()
        w = self._win
        if now - w[0] >= 1.0:
            if w[2]:
                suppressed, w[2] = w[2], 0
                self._journal(f"… +{suppressed} lines suppressed (rate limit)")
            w[0], w[1] = now, 0
        if w[1] >= int(_opts["raw_max_lines_per_s"]):
            w[2] += 1
            return
        w[1] += 1
        self._journal(line)

    def _journal(self, line: str) -> None:
        m = self._TAG.match(line)
        source = m.group(1).split()[0] if m else self._source
        self._guard.busy = True
        try:
            append(source, "note", line, level="raw")
            if self._extra is not None:
                self._extra(line)
        except Exception:
            pass
        finally:
            self._guard.busy = False


_tee_installed = False


def install_stdout_tee(source: str = "server",
                       extra: Optional[Callable[[str], None]] = None) -> None:
    """Journal every stdout/stderr line of THIS process at ``raw``. Idempotent.

    ``extra`` is an optional second sink for the same line — used by the offline train
    cycle to mirror unsloth's output into ``train_progress.jsonl`` as well, which is the
    file the watchdog serves while the inference server (and therefore this journal's
    WebSocket transport) is down."""
    global _tee_installed
    if _tee_installed or not _opts["raw_enabled"]:
        return
    _tee_installed = True
    try:
        sys.stdout = _Tee(sys.stdout, source, extra)   # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, source, extra)   # type: ignore[assignment]
    except Exception:
        _tee_installed = False


# ── reading ───────────────────────────────────────────────────────────────────────

def get(after_seq: int = 0, limit: int = 1000, *, levels=None,
        max_bytes: int = 0) -> list[dict]:
    """Return events with ``seq > after_seq`` (0 = everything available), newest kept.

    ``levels`` filters by level (default: all). ``max_bytes`` caps the batch — a ``body``
    carries kilobytes, so an uncapped catch-up read could build a frame of tens of MB;
    the cap is applied newest-first so a client always gets the *recent* end and can walk
    back with its cursor."""
    with _lock:
        oldest = _events[0].get("seq", 0) if _events else 0
        pool = [dict(e) for e in _events if e.get("seq", 0) > after_seq]
    # Cursor predates the in-memory ring — fall back to the file so a reconnecting client
    # (or one that was on another tab for an hour) gets continuity rather than a hole.
    if after_seq and oldest and after_seq < oldest - 1 and _path is not None:
        disk = [e for e in _tail_records(_path, limit * 4)
                if after_seq < int(e.get("seq", 0) or 0) < oldest]
        pool = disk + pool
    if levels:
        want = set(levels)
        pool = [e for e in pool if e.get("level", "event") in want]
    if limit:
        pool = pool[-limit:]
    if max_bytes:
        out, total = [], 0
        for e in reversed(pool):
            total += len(e.get("text") or "") + len(e.get("message") or "") + 200
            if out and total > max_bytes:
                break
            out.append(e)
        pool = list(reversed(out))
    return pool


def read_batch(after_seq: int = 0, *, levels=None, max_bytes: int = 0,
               limit: int = 1000) -> dict:
    """One transport batch: ``{events, gap}``.

    ``gap`` is the honest half. The batch is bounded two ways — the in-memory ring can have
    evicted what the cursor asked for, and the byte cap can drop the older end of a large
    catch-up read — and BOTH are invisible to the client, which advances its cursor to the
    newest seq it received and so would never come back for what was skipped. One test
    catches both: if the first event served is not the cursor's immediate successor, some
    records will never be delivered to this client. Saying so lets the UI draw a break
    rather than present a truncated history as continuity."""
    events = get(after_seq=after_seq, limit=limit, levels=levels, max_bytes=max_bytes)
    gap = bool(events and int(events[0].get("seq", 0) or 0) > after_seq + 1)
    return {"events": events, "gap": gap}


def oldest_seq() -> int:
    """The oldest ``seq`` still served from memory (0 when empty) — lets the transport
    tell a client its cursor fell off the end instead of silently skipping events."""
    with _lock:
        return int(_events[0].get("seq", 0)) if _events else 0


def latest_seq() -> int:
    with _lock:
        return _seq


def current() -> Optional[dict]:
    """The open activity for the status chip, or None when the box is idle."""
    with _lock:
        return dict(_current) if _current else None


# ── persistence (callers already hold _lock) ───────────────────────────────────────

def _persist_locked(ev: dict) -> None:
    if _path is None:
        return
    try:
        line = json.dumps(ev, ensure_ascii=False) + "\n"
        with open(_path, "a", encoding="utf-8") as fh:
            # Cross-process guard: the offline train cycle appends to this same file from
            # its own process. flock keeps a training line from interleaving into a
            # half-written server line; `seq` uniqueness rests on the watchdog stopping
            # inference before it runs a job (never two writers at once).
            if fcntl is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            fh.write(line)
        # Cheap size-gated rotation: only stat() occasionally.
        if ev["seq"] % 200 == 0 and _path.stat().st_size > int(_opts["segment_bytes"]):
            _rotate_locked(ev["seq"])
    except Exception:
        pass


def _rotate_locked(hi: int) -> None:
    """Close the current journal into a numbered segment and start a fresh one.

    The old behaviour rewrote the file down to the in-memory ring, i.e. it DELETED
    everything but the last ~2000 events at 8 MB. A log whose point is that you can scroll
    back through it cannot silently truncate itself; segments bound the disk instead."""
    if _path is None:
        return
    try:
        _path.replace(_path.with_name(f"{_path.stem}.{hi}{_path.suffix}"))
        keep = int(_opts["retain_segments"])
        segs = sorted(_path.parent.glob(f"{_path.stem}.*{_path.suffix}"),
                      key=lambda p: p.stat().st_mtime)
        for old in segs[:-keep] if keep > 0 else segs:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


# ── GPU-free self-test ─────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise seq monotonicity, current-pointer defaulting, cursor reads, durable
    recovery of the seq high-water, the pass seam (staged `started`, heartbeat throttle,
    body elision), level filtering, byte caps, segment rotation, and the stdout tee.
    Run: ``python -m core.activity_log``."""
    import tempfile

    global _events, _seq, _current, _path, _passes, _tee_installed
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hot" / "activity" / "activity.jsonl"
        configure(path)

        # global monotonic seq across sources
        a = append("wander", "finished", "read X")
        b = append("outreach", "finished", "reached out")
        assert b["seq"] == a["seq"] + 1, "seq must be globally monotonic"

        # set_current writes no journal line, but supplies the default activity_id
        before = latest_seq()
        aid = set_current("checkin", "running")
        assert latest_seq() == before, "set_current must not emit a journal event"
        assert current()["source"] == "checkin"
        p = append("checkin", "progress", "considering")
        assert p["activity_id"] == aid, "append should default to the current activity id"

        # -- the pass seam ------------------------------------------------------- #
        configure(path, heartbeat_s=0.05, body_max_chars=80, stream_tail_chars=10)
        with pass_context("chat_facts"):
            pid = begin_pass()
        assert _passes[pid]["label"] == "chat_facts", "pass_context should name the pass"
        n0 = latest_seq()
        pass_tick(pid, tokens=1, tail="abc")
        assert latest_seq() == n0, "first tick inside heartbeat_s must stay silent"
        time.sleep(0.06)
        pass_tick(pid, tokens=40, tail="x" * 50)
        assert latest_seq() == n0 + 2, "first emission = staged `started` + one heartbeat"
        started, beat = _events[-2], _events[-1]
        assert started["kind"] == "started" and "level" not in started
        assert beat["level"] == "stream" and len(beat["text"]) == 10, "tail must be capped"
        end_pass(pid, text="y" * 500, tokens=99, truncated=True)
        body = _events[-1]
        assert body["level"] == "body" and body["pass_id"] == pid
        assert "elided" in body["text"] and len(body["text"]) < 500, "middle elision"
        assert body["text"].endswith("y"), "the TAIL of a generation must survive"
        assert "hit the token cap" in body["message"]
        assert pid not in _passes, "end_pass must release the pass"

        # a pass that produced nothing writes no body, but is still released
        pid2 = begin_pass("empty")
        m0 = latest_seq()
        end_pass(pid2, text="   ")
        assert latest_seq() == m0 and pid2 not in _passes

        # short pass: never announced, so no `started` line at all
        pid3 = begin_pass("quick")
        k0 = latest_seq()
        pass_tick(pid3, tokens=3, tail="z")
        end_pass(pid3, text="done")
        assert latest_seq() == k0 + 1, "a sub-heartbeat pass writes only its body"
        clear_current()
        assert current() is None

        # -- level filtering + byte cap ------------------------------------------ #
        only_bodies = get(after_seq=0, levels={"body"})
        assert only_bodies and all(e.get("level") == "body" for e in only_bodies)
        assert get(after_seq=0, levels={"event"}), "events must remain reachable"
        capped = get(after_seq=0, max_bytes=1)
        assert len(capped) == 1, "byte cap keeps at least the newest event"

        # A batch that skips records must SAY so — the client advances its cursor past
        # them either way, so a silent truncation is a hole presented as continuity.
        assert read_batch(after_seq=0, max_bytes=1)["gap"] is True, \
            "a byte-capped catch-up read is a gap"
        assert read_batch(after_seq=latest_seq() - 1)["gap"] is False, \
            "a caught-up client must not see a spurious gap"
        assert read_batch(after_seq=0)["gap"] is False, \
            "a fresh client served the whole ring has no gap"

        # cursor read
        newer = get(after_seq=a["seq"])
        assert all(e["seq"] > a["seq"] for e in newer)
        assert newer[0]["seq"] == b["seq"]
        assert oldest_seq() == a["seq"]

        # -- the stdout tee ------------------------------------------------------- #
        configure(path, raw_max_lines_per_s=1000)
        import io
        _tee_installed = False
        real, sys.stdout = sys.stdout, io.StringIO()
        try:
            tee = _Tee(sys.stdout, "server")
            t0 = latest_seq()
            tee.write("[til] facts protocol: 60\n")
            tee.write("partial")                    # no newline yet → nothing journalled
            assert latest_seq() == t0 + 1, "a line is journalled once, on its newline"
            assert _events[-1]["source"] == "til", "the [tag] prefix names the source"
            assert _events[-1]["level"] == "raw"
            tee.write(" 1/9\r Loading 9/9\n")       # a progress redraw: keep the last one
            assert "9/9" in _events[-1]["message"] and "1/9" not in _events[-1]["message"]
            tee.write("\n")                          # blank → dropped by the denylist
            assert _events[-1]["message"].endswith("9/9")
            captured = sys.stdout.getvalue()
        finally:
            sys.stdout = real
        assert "facts protocol" in captured, "the real stream must be written through"

        # -- durable seq recovery across a restart -------------------------------- #
        hi = latest_seq()
        _events = deque(maxlen=_RING)
        _seq = 0
        _current = None
        _path = None
        _passes = {}
        configure(path)
        assert latest_seq() == hi, f"seq high-water must survive restart ({latest_seq()} != {hi})"
        c = append("wander", "finished", "post-restart")
        assert c["seq"] == hi + 1, "post-restart seq must continue past the recovered high-water"

        # -- rotation preserves history in a segment ------------------------------ #
        configure(path, segment_bytes=1)
        for i in range(200):
            append("wander", "note", f"filler {i}")
        segs = list(path.parent.glob("activity.*.jsonl"))
        assert segs, "rotation must leave a numbered segment, not truncate the journal"
        assert path.exists() is False or path.stat().st_size >= 0

    _events = deque(maxlen=_RING)
    _seq = 0
    _current = None
    _path = None
    _passes = {}
    _opts.update(_DEFAULTS)
    _tee_installed = False
    print("activity_log selftest: OK")


if __name__ == "__main__":
    _selftest()
