"""The nightly housekeeping event — one run per night, from ``housekeeping.hour`` (04:00).

A night is the stretch from one ``hour`` boundary (box-local time) to the next, named by
the date it starts on. The event runs its STEPS in order once per night, the first time
the box is free after the boundary: it is an idle job (``core.idle_scheduler``) whose
``ready`` gate is "inside tonight's window and not finished", so a reflection run or a
background drain that holds the GPU at 04:00 is simply waited out — the job queues on the
scheduler's GPU lock (an idle job holding it) or retries at the next poll (a Sleep run,
which is ``external_busy``). The window (``housekeeping.window_hours``, 8) bounds how late
a night may still run: a box that comes up at 15:00 does not start a night's work in the
middle of the owner's day.

**Steps.** ``(name, run, describe)`` triples, injected once by ``server.main()`` via
:func:`configure`; each ``run`` is a blocking body on the executor thread returning a
summary dict (the idle-job contract). Tonight: ``prompt_choice``
(``core.prompt_rewrite.run_rewrite_blocking``). Training is the planned second step (it
needs inference DOWN — the watchdog's job runner — so it is wired with the training code).

A step's result is TERMINAL for the night unless it reports one of :data:`RETRY_SKIPS`
(``preempted`` — a chat turn took the executor; ``busy``; ``no_model``): then the night
stays open, the job returns, and the next poll re-runs that step (the prompt choice
resumes its event from the last completed draft). An error is terminal — logged, never
retried in a loop. Steps already terminal tonight are not re-run.

**State** — ``data/hot/housekeeping/nights.jsonl``, append-only: one record per terminal
step (``night``, ``step``, ``ts``, ``result``) and one ``"step": "night"`` record when the
night finishes. A restart therefore never runs a finished step twice.

Leaf module: imports nothing from the project; the state dir, config loader, steps and
the progress sink are injected. GPU-free self-test: ``python -m core.housekeeping``.
"""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

NIGHTS_FILE = "nights.jsonl"
NIGHT_RECORD = "night"
RETRY_SKIPS = frozenset({"preempted", "busy", "no_model"})

DEFAULT_HOUR = 4
DEFAULT_WINDOW_HOURS = 8.0
DEFAULT_IDLE_SECONDS = 1800.0
POLL_SECONDS = 300.0

_STATE_DIR: Optional[Path] = None
_load_server_config: Optional[Callable[[], dict]] = None
_steps: list[tuple[str, Callable[[], dict], Optional[Callable[[dict], str]]]] = []
_on_progress: Optional[Callable[[str], None]] = None
_lock = threading.Lock()

# Occupancy flag — shown by :func:`status`. The scheduler needs no flag: the event runs as
# an idle job, inside its GPU lock.
_housekeeping_active = False


def configure(*, state_dir, load_server_config=None, steps=None, on_progress=None) -> None:
    """Wire the event (once, at startup). ``steps``: ``[(name, run, describe), …]`` in
    run order; ``on_progress(msg)`` journals a step boundary."""
    global _STATE_DIR, _load_server_config, _steps, _on_progress
    _STATE_DIR = Path(state_dir)
    _load_server_config = load_server_config
    _steps = [(str(n), r, d) for n, r, d in (steps or [])]
    _on_progress = on_progress


def settings(cfg: Optional[dict] = None) -> dict:
    """``housekeeping.*`` with defaults (config_schema.py is the documented source)."""
    if cfg is None:
        try:
            cfg = _load_server_config() if _load_server_config else {}
        except Exception:
            cfg = {}
    blk = (cfg or {}).get("housekeeping") or {}

    def _num(key, default, lo, hi):
        try:
            return min(max(float(blk.get(key, default)), lo), hi)
        except (TypeError, ValueError):
            return float(default)

    return {
        "enabled": bool(blk.get("enabled", True)),
        "hour": int(_num("hour", DEFAULT_HOUR, 0, 23)),
        "window_hours": _num("window_hours", DEFAULT_WINDOW_HOURS, 0.25, 24.0),
        "idle_seconds": _num("idle_seconds", DEFAULT_IDLE_SECONDS, 0.0, 86400.0),
    }


# ── the night clock ──────────────────────────────────────────────────────────

def _local_now(now: Optional[datetime] = None) -> datetime:
    """*now* as an aware box-local datetime (a naive one is taken as local already)."""
    return (now or datetime.now()).astimezone()


def night_start(now: Optional[datetime] = None, hour: int = DEFAULT_HOUR) -> datetime:
    """The most recent ``hour`` boundary at or before *now*, box-local."""
    local = _local_now(now)
    start = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if local < start:
        start -= timedelta(days=1)
    return start


def night_id(now: Optional[datetime] = None, hour: int = DEFAULT_HOUR) -> str:
    return night_start(now, hour).date().isoformat()


# ── state ────────────────────────────────────────────────────────────────────

def _nights_path() -> Optional[Path]:
    return (_STATE_DIR / NIGHTS_FILE) if _STATE_DIR is not None else None


def read_records() -> list[dict]:
    path = _nights_path()
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _append(rec: dict) -> dict:
    path = _nights_path()
    rec = {"ts": datetime.now().astimezone().isoformat(), **rec}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _done_steps(night: str, records: Optional[list[dict]] = None) -> dict[str, dict]:
    return {str(r.get("step")): r for r in (records if records is not None else read_records())
            if r.get("night") == night}


def _compact(result) -> dict:
    """What a step's summary keeps in the night record: scalars, short strings."""
    if not isinstance(result, dict):
        return {"result": str(result)[:200]}
    out = {}
    for k, v in result.items():
        if isinstance(v, (bool, int, float)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:500]
    return out


# ── gate + body ──────────────────────────────────────────────────────────────

def due(now: Optional[datetime] = None, cfg: Optional[dict] = None) -> bool:
    """The idle job's ``ready``: enabled, inside tonight's window, night not finished."""
    st = settings(cfg)
    if not st["enabled"] or not _steps:
        return False
    local = _local_now(now)
    start = night_start(local, st["hour"])
    if (local - start).total_seconds() >= st["window_hours"] * 3600.0:
        return False
    return NIGHT_RECORD not in _done_steps(start.date().isoformat())


def window_end(now: Optional[datetime] = None, cfg: Optional[dict] = None) -> float:
    """When tonight's window closes, as ``time.time()``: a step that runs for hours (the
    adapter measurement) stops there and resumes the next night."""
    st = settings(cfg)
    return (night_start(now, st["hour"]) + timedelta(hours=st["window_hours"])).timestamp()


def run_night_blocking(now: Optional[datetime] = None) -> dict:
    """Run tonight's unfinished steps in order, on the executor thread.

    Returns ``{night, steps: {name: result}, finished}``, or ``skipped`` — with a retry
    reason (``preempted`` …, plus ``step``) when a step left the night open, or the gate's
    reason (``disabled`` / ``outside_window`` / ``finished`` / ``no_steps``)."""
    global _housekeeping_active
    st = settings()
    local = _local_now(now)
    start = night_start(local, st["hour"])
    night = start.date().isoformat()
    if not st["enabled"]:
        return {"skipped": "disabled"}
    if not _steps:
        return {"skipped": "no_steps"}
    if (local - start).total_seconds() >= st["window_hours"] * 3600.0:
        return {"skipped": "outside_window", "night": night}
    with _lock:
        if _housekeeping_active:
            return {"skipped": "busy"}
        _housekeeping_active = True
    try:
        done = _done_steps(night)
        if NIGHT_RECORD in done:
            return {"skipped": "finished", "night": night}
        ran: dict[str, dict] = {}
        for name, run, describe in _steps:
            if name in done:
                continue
            _progress(f"Housekeeping (night {night}): {name} started")
            try:
                result = run()
            except Exception as e:
                traceback.print_exc()
                result = {"error": f"{type(e).__name__}: {e}"}
            if not isinstance(result, dict):
                result = {"result": result}
            reason = str(result.get("skipped") or "")
            if reason in RETRY_SKIPS:
                _progress(f"Housekeeping (night {night}): {name} left open ({reason}); "
                          f"it resumes on the next free poll")
                return {"skipped": reason, "night": night, "step": name, "steps": ran}
            _append({"night": night, "step": name, "result": _compact(result),
                     "summary": _describe(name, describe, result)})
            ran[name] = result
            _progress(f"Housekeeping (night {night}): {_describe(name, describe, result)}")
        _append({"night": night, "step": NIGHT_RECORD, "steps": sorted(ran)})
        return {"night": night, "steps": ran, "finished": True,
                "summaries": {n: _describe(n, d, ran[n]) for n, _, d in _steps if n in ran}}
    finally:
        _housekeeping_active = False


def _describe(name: str, describe, result: dict) -> str:
    if describe is not None:
        try:
            msg = describe(result)
            if msg:
                return msg
        except Exception:
            traceback.print_exc()
    if result.get("error"):
        return f"{name} failed: {result['error']}"
    if result.get("skipped"):
        return f"{name} skipped: {result['skipped']}"
    return f"{name} finished"


def _progress(msg: str) -> None:
    if _on_progress is not None:
        try:
            _on_progress(msg)
        except Exception:
            pass


def describe(r: dict) -> str:
    """The idle job's terminal line."""
    if not isinstance(r, dict):
        return "Housekeeping ran"
    if r.get("finished"):
        parts = list((r.get("summaries") or {}).values())
        return (f"Housekeeping (night {r.get('night')}) finished"
                + (": " + " · ".join(parts) if parts else " — nothing left to run"))
    return f"Housekeeping: {r}"


def status(now: Optional[datetime] = None, cfg: Optional[dict] = None) -> dict:
    """GPU-free view for the operator: tonight's id, window, what ran, the next start."""
    st = settings(cfg)
    local = _local_now(now)
    start = night_start(local, st["hour"])
    night = start.date().isoformat()
    records = read_records()
    done = _done_steps(night, records)
    in_window = (local - start).total_seconds() < st["window_hours"] * 3600.0
    finished = NIGHT_RECORD in done
    nxt = start + timedelta(days=1) if (finished or not in_window) else start
    last_finished = next((r.get("night") for r in reversed(records)
                          if r.get("step") == NIGHT_RECORD), None)
    return {
        "enabled": st["enabled"], "hour": st["hour"], "window_hours": st["window_hours"],
        "night": night, "in_window": in_window, "finished": finished,
        "active": _housekeeping_active,
        "steps": [n for n, _, _ in _steps],
        "done": {k: v.get("summary", "") for k, v in done.items() if k != NIGHT_RECORD},
        "next_start": nxt.isoformat(),
        "last_finished_night": last_finished,
    }


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile

    tz = datetime.now().astimezone().tzinfo

    def at(day: int, h: int, m: int = 0) -> datetime:
        return datetime(2026, 10, day, h, m, tzinfo=tz)

    # The clock: before 04:00 belongs to the previous night.
    assert night_id(at(4, 3, 59)) == "2026-10-03"
    assert night_id(at(4, 4, 0)) == "2026-10-04"
    assert night_id(at(4, 23, 0)) == "2026-10-04"
    assert night_id(at(4, 1, 0), hour=0) == "2026-10-04"
    assert settings({"housekeeping": {"hour": 30, "window_hours": "x"}}) == {
        "enabled": True, "hour": 23, "window_hours": DEFAULT_WINDOW_HOURS,
        "idle_seconds": DEFAULT_IDLE_SECONDS}

    with tempfile.TemporaryDirectory() as d:
        cfg = {"housekeeping": {"hour": 4, "window_hours": 8}}
        calls: list[str] = []
        plan = {"a": [{"skipped": "preempted"}, {"outcome": "stayed"}], "b": [{"done": 1}]}

        def step(name):
            def run():
                calls.append(name)
                return plan[name].pop(0)
            return run

        progress: list[str] = []
        configure(state_dir=Path(d) / "hk", load_server_config=lambda: cfg,
                  steps=[("a", step("a"), lambda r: f"a: {r.get('outcome')}"),
                         ("b", step("b"), None)],
                  on_progress=progress.append)
        assert not due(at(4, 3, 59)) and due(at(4, 4, 1)) and not due(at(4, 12, 1))
        assert run_night_blocking(at(4, 13))["skipped"] == "outside_window"
        # A preempted step leaves the night open; nothing after it runs.
        r = run_night_blocking(at(4, 4, 5))
        assert r["skipped"] == "preempted" and r["step"] == "a" and calls == ["a"], r
        assert due(at(4, 4, 10)) and read_records() == []
        # The next poll resumes at that step and finishes the night.
        r = run_night_blocking(at(4, 4, 10))
        assert r["finished"] and calls == ["a", "a", "b"], (r, calls)
        assert describe(r) == "Housekeeping (night 2026-10-04) finished: a: stayed · b finished", describe(r)
        assert not due(at(4, 5)) and run_night_blocking(at(4, 5))["skipped"] == "finished"
        st = status(at(4, 5), cfg)
        assert st["finished"] and st["done"] == {"a": "a: stayed", "b": "b finished"}, st
        assert st["next_start"].startswith("2026-10-05T04:00"), st
        assert st["last_finished_night"] == "2026-10-04"
        # A raising step is terminal (logged), never a retry loop; the next night is fresh.
        plan["a"] = [{"outcome": "changed"}]
        plan["b"] = []                        # pop from empty ⇒ IndexError inside the step
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            r = run_night_blocking(at(5, 4, 2))
        assert r["finished"] and "error" in r["steps"]["b"], r
        assert [x["step"] for x in read_records() if x["night"] == "2026-10-05"] == ["a", "b", "night"]
        assert any("started" in p for p in progress)
        # Disabled ⇒ never due.
        assert not due(at(6, 4, 1), {"housekeeping": {"enabled": False}})
    print("housekeeping self-test OK")


if __name__ == "__main__":
    _selftest()
