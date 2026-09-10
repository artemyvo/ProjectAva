"""First-person episodic worklog — Ava's durable, semantic record of what she has done.

Sibling of, and deliberately distinct from, :mod:`core.activity_log`. That module is a
lossy, machine-phrased *ring* serving the live UI ("what's the GPU doing right now"): one
event per job lifecycle + phase line, bounded and size-rotated. This one is keep-forever
autobiographical memory in Ava's own voice — one entry per *meaningful episode* (a
reach-out, a wander, a reflection run), each an INDEX into the scattered traces (chat
stem / wander title / ask key / run id) rather than a copy of them. It is the substrate a
later deliberation pass will read to decide what to do next ("I reached out an hour ago,
the user hasn't replied, I've wandered twice today → …"), and it is recallable memory.

    activity_log : telemetry, ring, machine voice, read by the client
    worklog      : memory,    keep-forever, first-person, read by Ava (later)

A *leaf*, exactly like :mod:`core.reachout_gate` / :mod:`core.activity_log`: it imports
nothing from the project and is imported directly (``from core import worklog``), so any
subsystem can call :func:`record` with no configure-injection, no wiring, and no import
cycle. It also never generates — the first-person ``summary`` is supplied by the caller
(a template today; a real generation at chosen call sites later), keeping this module a
pure, GPU-free sink. The intended read side (a deliberation pass) is a *separate task*;
for now nothing consumes it — it is inert autobiographical memory being accumulated, plus
a read path for the Worklog preview tab.

Op-log fold for open threads: an episode may open a loop (``opens="awaiting reply"``); a
later episode closes it by id (``closes=<id>``). :func:`open_threads` folds these into the
still-hanging set — the "what have I left unfinished" state a planner needs, mirroring
:meth:`reflection_memory.ReflectionMemory.open_questions`.

GPU-free self-test: ``python -m core.worklog``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The episode kinds — one per subsystem that closes a meaningful unit of activity. Unknown
# kinds are coerced to "conversation" so a typo never silently drops an entry.
_KINDS = {"conversation", "wander", "outreach", "synthesis", "checkin",
          "reflection", "encounter"}

_lock = threading.Lock()
_entries: list[dict] = []      # full history in memory (durable memory, NOT a bounded ring)
_id = 0
_path: Optional[Path] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── lifecycle ─────────────────────────────────────────────────────────────────────

def configure(path) -> None:
    """Load the full worklog and recover the id high-water. Call once at startup.

    Unlike the activity ring this loads the WHOLE file (it is memory, kept forever), so a
    reconnecting deliberation pass or preview tab sees the complete history. Safe if the
    file does not exist yet."""
    global _path, _id, _entries
    with _lock:
        _path = Path(path)
        _entries = []
        _id = 0
        try:
            _path.parent.mkdir(parents=True, exist_ok=True)
            if _path.exists():
                for ln in _path.read_text(encoding="utf-8").splitlines():
                    try:
                        e = json.loads(ln)
                    except Exception:
                        continue
                    _entries.append(e)
                    _id = max(_id, int(e.get("id", 0) or 0))
        except Exception:
            pass


# ── writing ───────────────────────────────────────────────────────────────────────

def record(kind: str, summary: str, *, refs: Optional[dict] = None,
           opens: Optional[str] = None, closes: Optional[int] = None,
           salience: Optional[float] = None) -> dict:
    """Append one episode in Ava's first-person voice and return it (with its durable id).

    ``summary`` is the first-person prose ("I reached out to Artemy about the worklog
    idea"); the caller supplies it (this module never generates). ``refs`` points INTO the
    source artifacts (``{"session": "...", "ask_key": "...", "run_id": "..."}``) rather than
    copying them. ``opens`` records a loop this episode left hanging; ``closes`` folds shut
    the ``opens`` of an earlier episode by id. ``salience`` is an optional ranking prior.
    Best-effort: a persistence failure never propagates into the caller's episode.
    """
    global _id
    summary = (summary or "").strip()
    with _lock:
        _id += 1
        e: dict = {
            "id": _id,
            "ts": _utc_now(),
            "kind": kind if kind in _KINDS else "conversation",
            "summary": summary,
        }
        if refs:
            e["refs"] = {k: v for k, v in refs.items() if v}
        if opens:
            e["opens"] = opens.strip()
        if closes:
            e["closes"] = int(closes)
        if salience is not None:
            e["salience"] = float(salience)
        _entries.append(e)
        _persist_locked(e)
    return e


# ── reading (for the preview tab now; a deliberation pass later) ────────────────────

def recent(n: int = 50) -> list[dict]:
    """The last ``n`` episodes, oldest→newest — the window a deliberation pass reads and
    the preview tab renders on first load."""
    with _lock:
        return [dict(e) for e in _entries[-n:]] if n else [dict(e) for e in _entries]


def since(after_id: int = 0) -> list[dict]:
    """Episodes with ``id > after_id`` (0 = everything) — cursor poll for the preview tab."""
    with _lock:
        return [dict(e) for e in _entries if e.get("id", 0) > after_id]


def latest_id() -> int:
    with _lock:
        return _id


def open_threads() -> list[dict]:
    """Episodes that opened a loop no later episode has closed — the 'what's still hanging'
    state a planner needs. Folds ``closes`` over ``opens`` (cf.
    :meth:`reflection_memory.ReflectionMemory.open_questions`)."""
    with _lock:
        closed = {e["closes"] for e in _entries if e.get("closes")}
        return [dict(e) for e in _entries if e.get("opens") and e["id"] not in closed]


# ── persistence (callers already hold _lock) ───────────────────────────────────────

def _persist_locked(e: dict) -> None:
    if _path is None:
        return
    try:
        with open(_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── GPU-free self-test ─────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise id monotonicity, refs pruning, the open/close fold, cursor reads, and
    durable recovery of the id high-water. Run: ``python -m core.worklog``."""
    import tempfile

    global _entries, _id, _path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hot" / "worklog" / "worklog.jsonl"
        configure(path)

        a = record("wander", "I read about Voyager 1 and thought about distance.",
                   refs={"title": "Voyager 1", "url": ""})   # empty ref pruned
        assert a["id"] == 1
        assert a["refs"] == {"title": "Voyager 1"}, "empty ref values must be pruned"

        b = record("outreach", "I asked Artemy whether the worklog is its own layer.",
                   opens="awaiting Artemy's reply", refs={"ask_key": "k1"})
        assert b["id"] == 2

        # unknown kind coerced, monotonic id
        c = record("bogus", "misc")
        assert c["kind"] == "conversation" and c["id"] == 3

        # open thread folds until closed
        opens = open_threads()
        assert [e["id"] for e in opens] == [2], "only the un-closed opener should hang"
        record("conversation", "Artemy replied about the worklog.", closes=2)
        assert open_threads() == [], "closing the opener empties the hanging set"

        # cursor read
        newer = since(after_id=b["id"])
        assert all(e["id"] > b["id"] for e in newer)

        # durable id recovery across a restart
        hi = latest_id()
        _entries = []
        _id = 0
        _path = None
        configure(path)
        assert latest_id() == hi, f"id high-water must survive restart ({latest_id()} != {hi})"
        assert len(recent(0)) == 4, "full history reloaded (worklog is not a ring)"
        d = record("checkin", "post-restart")
        assert d["id"] == hi + 1, "post-restart id continues past the recovered high-water"

    _entries = []
    _id = 0
    _path = None
    print("worklog selftest: OK")


if __name__ == "__main__":
    _selftest()
