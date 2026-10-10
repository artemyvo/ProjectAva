"""The capture judge over the whole corpus, run on demand (``core/capture_judge.py``).

A Sleep run's clean-base batch grades the training targets the frozen judge has not graded,
at most ``capture_gate.sleep_max`` a run. This runs the same pass over every exchange of the
live corpus — the first time, and after a change to the judge's prompt (a new judge key) —
in the BACKGROUND inside one ``CleanBaseSession``, polled and stoppable (the exchange in
flight finishes; everything graded is already stored, and the next run resumes). Its
decisions go straight to the live sidecars, as the Training review's ban does.

While a run is active :data:`_capture_active` is set: chat is refused
(``generation._run_generation``), the idle jobs wait, the API and gossip answer busy.

Messages (``configure()``-wired from ``server.main()``):

* ``capture_judge_report {}`` → ``capture_judge_report``: the summary and rendered report
  under the current judge, how many exchanges are graded, gated and waiting.
* ``capture_judge_run {limit?, rejudge?, persona?, originals?}`` → ``capture_judge_started``,
  or with ``skipped`` (running / busy / no_model / no_prompt / nothing). ``persona: false``
  leaves the persona half to ``capture_persona_apply`` (a new judge's gates read before its
  withholding applies). ``originals: true`` grades instead the stored reply behind each
  revised target not yet graded under the current judge (``cj.run_originals``): the
  capture probe's labels, into ``originals.jsonl``; no gate, no persona half.
* ``capture_judge_status {after?}`` → ``capture_judge_status``: phase, progress, tally, the
  lines after the cursor, and once finished the report.
* ``capture_judge_stop`` → ``capture_judge_stopping``.
* ``capture_persona_plan {}`` → ``capture_persona_plan``: the live ``[persona]`` statements
  whose every source exchange is gated (what a run withholds), the withheld ones whose
  sources no longer all are (``restore``: what it restores), and how many it withholds
  now.
* ``capture_persona_apply {keys?}`` → ``capture_persona_applied``: supersede the first and
  restore the second (all, or the content keys in ``keys``), through the live writer and
  ledger; recall refreshed.
* ``capture_repair_report {attempts?}`` → ``capture_repair_report``: the capture repair's
  store (``core/capture_repair.py``) summarized, and how many gated exchanges it can still
  work on.
* ``capture_repair_run {limit?, attempts?, regrade_only?}`` → ``capture_repair_started``, or
  with ``skipped`` (running / busy / no_model / no_generate / no_prompt / nothing): the gated
  exchanges answered again. The attempts not graded under the current judge are graded
  first, in the order made; then rounds of one attempt per exchange on the loaded model,
  each round graded in one clean base; the first attempt the judge accepts is promoted to
  the exchange's training target. ``regrade_only`` stops after the first step. Progress
  through ``capture_judge_status``, stopped by ``capture_judge_stop``.
* ``capture_bench_report {bench?, samples?}`` → ``capture_bench_report``: the capture
  bench's store (``core/capture_bench.py``) summarized, its frozen contexts, and how many
  replies the loaded weights and the base still owe it.
* ``capture_bench_run {arms?, bench?, assertion?, reason?, samples?, limit?, seed?}`` →
  ``capture_bench_started``, or with ``skipped`` (running / busy / no_model / no_generate /
  no_prompt / no_arms / no_persona_empty): the bench's contexts (frozen by its first run)
  answered under the loaded weights (``adapter``) and/or the base, each as frozen or with
  the persona slot empty (``adapter|persona_empty``, ``base|persona_empty``: P8), then
  graded — the loaded model's replies first, the base's replies and every grade in one
  clean base. Progress through ``capture_judge_status``, stopped by ``capture_judge_stop``.
* ``capture_bench_holdout {bench?, action?}`` → ``capture_bench_holdout``: ``plan``
  (default) shows what holding the bench's contexts out of every build would do, per
  context; ``apply`` holds them out (the sidecar record's ``holdout`` field, refused by the
  build) and ``release`` lets them train again, each followed by the persona half's sync
  (the ``[persona]`` statements written only from exchanges kept out of training are
  withheld, and restored when a source trains again) — both refused (``skipped: busy``)
  while a job holds the box, since a reflection run promotes whole staged sidecars over
  the live ones; a ``holdout`` record in the bench's store dates it.

A corpus run withholds the persona statements of every gated exchange after its verdicts,
as a Sleep run does, unless it is told not to. Writes the judge's store, the sidecars'
``capture`` field and, for those statements, the op-log's and the ledger's supersede ops;
nothing else. A repair run
writes the repair's store, and for a promotion the sidecar's target (``write_verdict``, a
ledger event) and the judge's verdict on it. A bench run writes the bench's store and
nothing else: a measurement, never a gate (P3); its held-out set writes the sidecars'
``holdout`` field and a ledger flag. No human label enters any of them (P9):
the operator chooses what to run and apply, never a verdict.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from core import activity_log
from core import ask_judge as aj
from core import capture_bench as cb
from core import capture_judge as cj
from core import capture_repair as cr
from core.chat_sidecar import ChatSidecar

# True from the moment a run is accepted (set on the event loop, before the executor job is
# queued) until it ends.
_capture_active = False

_send: Callable = None
_executor: Any = None
_backend: Any = None
_runtime: Any = None
_with_clean_base: Callable = None
_host_busy: Callable = None
_mark_activity: Callable = None
_CHATS_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None
_MEMORY_DIR: Optional[Path] = None
_CONSOLIDATION_DIR: Optional[Path] = None
_get_writer: Optional[Callable] = None
_get_rag: Optional[Callable] = None
# The capture repair's actor: factories for the reflect-lane generate on the loaded model
# and the revision pass's persona block (the clean IDEAL seam's two inputs).
_make_reflect_generate: Optional[Callable] = None
_make_persona_context: Optional[Callable] = None
# The capture bench's persona-empty condition (P8): the original standing prompt and the
# persona slot's undecided framing, as the capture control's `*_orig_empty` conditions.
_persona_empty_system: Optional[Callable[[], str]] = None

_lock = threading.Lock()
_state: dict = {}
_lines: deque = deque(maxlen=400)
_line_seq = 0


def configure(*, send, executor, backend, runtime, with_clean_base, host_busy, mark_activity,
              chats_dir, prompts_dir, memory_dir=None, consolidation_dir=None,
              get_reflection_writer: Optional[Callable] = None,
              get_rag: Optional[Callable] = None,
              make_reflect_generate: Optional[Callable] = None,
              make_persona_context: Optional[Callable] = None,
              persona_empty_system: Optional[Callable[[], str]] = None) -> None:
    global _send, _executor, _backend, _runtime, _with_clean_base, _host_busy, _mark_activity
    global _CHATS_DIR, _PROMPTS_DIR, _MEMORY_DIR, _CONSOLIDATION_DIR, _get_writer, _get_rag
    global _make_reflect_generate, _make_persona_context, _persona_empty_system
    _send, _executor, _backend, _runtime = send, executor, backend, runtime
    _with_clean_base, _host_busy, _mark_activity = with_clean_base, host_busy, mark_activity
    _CHATS_DIR, _PROMPTS_DIR = Path(chats_dir), Path(prompts_dir)
    _MEMORY_DIR = Path(memory_dir) if memory_dir is not None else None
    _CONSOLIDATION_DIR = Path(consolidation_dir) if consolidation_dir is not None else None
    _get_writer, _get_rag = get_reflection_writer, get_rag
    _make_reflect_generate, _make_persona_context = make_reflect_generate, make_persona_context
    _persona_empty_system = persona_empty_system
    _reset(phase="idle")


def _reset(**fields) -> None:
    with _lock:
        _state.clear()
        _state.update({"phase": "idle", "run_id": None, "judge_key": None, "started": None,
                       "finished": None, "done": 0, "total": 0, "tally": {}, "error": None,
                       "report": None, "stop": False})
        _state.update(fields)


def _set(**fields) -> None:
    with _lock:
        _state.update(fields)


def _say(text: str) -> None:
    global _line_seq
    with _lock:
        _line_seq += 1
        _lines.append((_line_seq, text))
    print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [capture_judge] {text}",
          flush=True)


def _bump(word: str) -> None:
    with _lock:
        tally = dict(_state.get("tally") or {})
        tally[word] = tally.get(word, 0) + 1
        _state["tally"] = tally


def _model_id() -> str:
    return str(getattr(_runtime, "model_id", "") or "")


def _key_now() -> Optional[str]:
    """The judge key the store reads now: the newest for the current prompt and model."""
    try:
        psha = cj.prompt_sha(cj.load_prompt(_PROMPTS_DIR))
    except FileNotFoundError:
        return None
    return cj.current_key(aj.read_jsonl(cj.store_path(_CHATS_DIR)), prompt_sha_=psha,
                          model_id=_model_id())


def _key_guess() -> str:
    """The key a run will write under, before the swap tells us the loaded precision."""
    key = _key_now()
    if key:
        return key
    try:
        psha = cj.prompt_sha(cj.load_prompt(_PROMPTS_DIR))
    except FileNotFoundError:
        psha = ""
    return aj.judge_key(prompt_sha_=psha, model_id=_model_id(),
                        base_quant=aj.quant_label(str(getattr(_runtime, "base_quant", "") or "")),
                        temperature=0.0, thinking=False, retry=True)


def report_payload() -> dict:
    key = _key_now()
    records = aj.read_jsonl(cj.store_path(_CHATS_DIR))
    s = cj.summarize(records, key)
    verdicts = cj.latest_verdicts(records, key) if key else {}
    cands = cj.candidates(ChatSidecar(_CHATS_DIR))
    waiting = sum(1 for c in cands
                  if cj.cand_id(c["session"], c["exchange_index"], c["target_sha"]) not in verdicts)
    return {"type": "capture_judge_report", "judge_key": key, "summary": s,
            "report": cj.render(s), "candidates": len(cands), "waiting": waiting,
            "enabled": cj.enabled(), "policy": cj.policy(), "active": _capture_active,
            "store": str(cj.store_path(_CHATS_DIR))}


def originals_waiting() -> int:
    """Stored replies behind a revised target with no grade under the current judge."""
    return cj.run_originals(sidecar=ChatSidecar(_CHATS_DIR), judge=None, key=_key_now()).get("original_waiting", 0)


def _persona_plan() -> list:
    if _MEMORY_DIR is None:
        return []
    return cj.persona_plan(ChatSidecar(_CHATS_DIR), [_MEMORY_DIR])


def _restore_plan() -> list:
    if _MEMORY_DIR is None:
        return []
    return cj.restore_plan(ChatSidecar(_CHATS_DIR), [_MEMORY_DIR])


def _apply_persona(plan: list, run_id: str, restore: Optional[list] = None) -> Counter:
    """Supersede *plan*'s statements and restore *restore*'s through the live writer and
    ledger, then refresh recall. Nothing without the writer and ledger wired."""
    if (not plan and not restore) or _get_writer is None or _CONSOLIDATION_DIR is None:
        return Counter()
    from training.ledger import ConsolidationLedger
    ledger = ConsolidationLedger(_CONSOLIDATION_DIR)
    tally = cj.withhold_persona(plan, writer=_get_writer(), ledger=ledger, run_id=run_id, say=_say)
    tally.update(cj.restore_persona(restore or [], writer=_get_writer(), ledger=ledger, run_id=run_id,
                                    say=_say))
    if _get_rag is not None:
        try:
            _get_rag().refresh_reflection_memory()
        except Exception:  # noqa: BLE001 — recall catches up at its next refresh
            traceback.print_exc()
    return tally


def persona_plan_payload() -> dict:
    plan, restore = _persona_plan(), _restore_plan()
    rag = cj.read_oplogs([_MEMORY_DIR]) if _MEMORY_DIR is not None else []
    return {"type": "capture_persona_plan", "count": len(plan),
            "items": [{**it, "sources": [f"{s}#{i}" for s, i in it["sources"]]} for it in plan],
            "restore_count": len(restore),
            "restore": [{"key": it["key"], "content": it["content"],
                         "sources": [f"{s}#{i}" for s, i in it["sources"]]} for it in restore],
            "withheld": cj.persona_withheld(rag),
            "withheld_holdout": cj.persona_withheld(rag, cj.HOLDOUT_REASON), "active": _capture_active}


def _blocking_run(*, limit: int, rejudge: bool, run_id: str, persona: bool = True,
                  originals: bool = False) -> None:
    """Executor thread: one clean base, every waiting exchange graded. Never raises."""
    activity_log.set_current("capture_judge", "capture judge", phase="judging")
    try:
        _set(phase="swapping")
        _say("swapping the adapter out for the clean base…")
        t_swap = time.monotonic()

        def _inside():
            make = cj.make_originals_judge_fn if originals else cj.make_judge_fn
            key, judge = make(backend=_backend, runtime=_runtime, live_chats_dir=_CHATS_DIR,
                              prompts_dir=_PROMPTS_DIR, run_id=run_id, runner="server")
            _set(phase="judging", judge_key=key)
            _say(f"clean base in {time.monotonic() - t_swap:.0f} s — judge {key}")
            t0 = time.monotonic()

            def _on_record(n: int, total: int, v: dict) -> None:
                _bump(v.get("response") or v.get("outcome") or "?")
                if v.get("gated"):
                    _bump("gated")
                eta = (time.monotonic() - t0) / n * (total - n)
                _set(done=n, total=total, eta_s=round(eta))

            if originals:
                tally = cj.run_originals(sidecar=ChatSidecar(_CHATS_DIR), judge=judge, key=key, limit=limit,
                                         say=_say, should_stop=lambda: bool(_state.get("stop")),
                                         on_record=_on_record)
            else:
                tally = cj.run(sidecar=ChatSidecar(_CHATS_DIR), judge=judge, key=key, limit=limit,
                               rejudge=rejudge, say=_say, should_stop=lambda: bool(_state.get("stop")),
                               on_record=_on_record)
            _set(phase="restoring")
            _say("restoring the adapter model…")
            return tally

        tally = _with_clean_base(_inside) or Counter()
        stopped = bool(_state.get("stop"))
        # ...and the [persona] statements of every gated exchange (P7), GPU-free.
        if originals:
            pass                                 # labels only: no gate moved
        elif persona:
            synced = _apply_persona(_persona_plan(), run_id, _restore_plan())
            if synced:
                with _lock:
                    _state["tally"] = {**dict(_state.get("tally") or {}), **dict(synced)}
            tally.update(synced)
        else:
            _say(f"persona half left to capture_persona_apply: {len(_persona_plan())} statement(s) to "
                 f"withhold, {len(_restore_plan())} to restore")
        report = cj.render(cj.summarize(aj.read_jsonl(cj.store_path(_CHATS_DIR)), _state.get("judge_key")))
        _set(phase="stopped" if stopped else "finished", report=report,
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(tally)}")
        activity_log.append("capture_judge", "finished",
                            f"Capture judge {'stopped' if stopped else 'finished'}: {dict(tally)}",
                            text=report)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("capture_judge", "failed", f"Capture judge failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


# ── the capture repair (core/capture_repair.py) ───────────────────────────────

def repair_payload(attempts: int = cr.DEFAULT_ATTEMPTS) -> dict:
    records = aj.read_jsonl(cr.store_path(_CHATS_DIR))
    items = cr.todo(ChatSidecar(_CHATS_DIR), records, attempts=attempts, key=_key_guess())
    s = cr.summarize(records, attempts=attempts)
    return {"type": "capture_repair_report", "summary": s, "report": cr.render(s),
            "waiting": len(items), "pending": sum(len(it["pending"]) for it in items),
            "attempts": attempts, "active": _capture_active, "store": str(cr.store_path(_CHATS_DIR))}


def _grade_round(made: list, *, run_id: str, store: Path, sidecar: ChatSidecar) -> None:
    """Inside ONE clean base: grade each ``(item, attempt)`` in order and promote the first
    accepted attempt of each exchange. A stop leaves the rest ungraded (the next run
    grades them first)."""
    _set(phase="swapping")
    _say("swapping the adapter out for the clean base…")
    t_swap = time.monotonic()

    def _inside():
        key, grade = cj.make_grade_fn(backend=_backend, runtime=_runtime, prompts_dir=_PROMPTS_DIR,
                                      run_id=run_id, runner="repair")
        _set(phase="grading", judge_key=key)
        _say(f"clean base in {time.monotonic() - t_swap:.0f} s — judge {key}: {len(made)} attempt(s)")
        for n, (it, att) in enumerate(made, 1):
            if _state.get("stop"):
                break
            if it.get("repaired"):
                continue            # an earlier attempt of this exchange already passed
            v = grade(cr.candidate(it, att))
            aj.append_record(store, cr.grade_record(it, att, v, run_id=run_id))
            word = v.get("response") or v.get("outcome") or "?"
            _bump(f"graded:{word}")
            won = cr.promote(it, att, v, sidecar=sidecar, run_id=run_id,
                             verdict_store=cj.store_path(_CHATS_DIR))
            if won:
                aj.append_record(store, cr.pair_record(it, att, v, run_id=run_id))
                it["repaired"] = True
                _bump("repaired")
            _say(f"[{n:>3}/{len(made)}] attempt {att.get('attempt')} {word:<8}"
                 f"{' → promoted' if won else ''}  {it['session']}#{it['exchange_index']}"
                 f"  {aj.head(v.get('why'), 90)}")
        _set(phase="restoring")
        _say("restoring the adapter model…")

    _with_clean_base(_inside)


def _blocking_repair(*, limit: int, attempts: int, run_id: str, regrade_only: bool = False) -> None:
    """Executor thread: the gated exchanges answered again, a round at a time — one attempt
    each on the loaded model, then the round graded and promoted in one clean base — until
    each is repaired or out of attempts. With *regrade_only*, only the attempts not graded
    under the current judge are graded (and the first accepted promoted); nothing new is
    asked for. Never raises."""
    activity_log.set_current("capture_judge", "capture repair", phase="attempts")
    try:
        store = cr.store_path(_CHATS_DIR)
        sidecar = ChatSidecar(_CHATS_DIR)
        generate_fn = _make_reflect_generate()
        persona_fn = _make_persona_context() if _make_persona_context is not None else None
        model_id = _model_id()
        adapter_id = str(getattr(_runtime, "adapter_id", "") or "")
        # graded under the judge the store reads now: after a change to the judge, the
        # attempts already made are graded again, in order, before any new one
        items = cr.todo(sidecar, aj.read_jsonl(store), attempts=attempts, key=_key_guess())
        if limit:
            items = items[:limit]
        _set(total=len(items), done=0)
        _say(f"{len(items)} gated exchange(s), up to {attempts} attempt(s) each, one a round — "
             f"adapter {adapter_id.rstrip('/').split('/')[-1] or 'none'}")

        def _progress() -> None:
            _set(done=sum(1 for it in items if it.get("repaired") or int(it["made"]) >= attempts))

        pending = [(it, a) for it in items
                   for a in sorted(it["pending"], key=lambda a: int(a.get("attempt") or 0))]
        if pending:
            _say(f"{len(pending)} attempt(s) an earlier run left ungraded: grading them first")
            _grade_round(pending, run_id=run_id, store=store, sidecar=sidecar)
            _progress()
        rnd = 0
        while not _state.get("stop") and not regrade_only:
            batch = [it for it in items if not it.get("repaired") and int(it["made"]) < attempts]
            if not batch:
                break
            rnd += 1
            _set(phase="attempts", round=rnd)
            made: list = []
            t0 = time.monotonic()
            for n, it in enumerate(batch, 1):
                if _state.get("stop"):
                    break
                att = cr.make_attempt(it, generate_fn=generate_fn, persona_context_fn=persona_fn,
                                      run_id=run_id, round_no=rnd, model_id=model_id,
                                      adapter_id=adapter_id)
                aj.append_record(store, att)
                it["made"] = int(it["made"]) + 1
                _bump(f"attempt:{att['outcome']}")
                eta = (time.monotonic() - t0) / n * (len(batch) - n)
                _say(f"round {rnd} [{n:>3}/{len(batch)}] {att['outcome']:<9} {att['seconds']:>5.0f} s"
                     f"  eta {eta / 60:>4.0f} min  {it['session']}#{it['exchange_index']}"
                     f" (attempt {att['attempt']})")
                if att["outcome"] == "ok":
                    made.append((it, att))
            if made and not _state.get("stop"):
                _grade_round(made, run_id=run_id, store=store, sidecar=sidecar)
            _progress()
        # Every gate synced to the judge's store (GPU-free): a promoted target's verdict
        # lifts its gate; nothing else changes.
        cj.run(sidecar=sidecar, judge=None, key=_key_now(), say=_say)
        stopped = bool(_state.get("stop"))
        report = cr.render(cr.summarize(aj.read_jsonl(store), attempts=attempts))
        _set(phase="stopped" if stopped else "finished", report=report,
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(_state.get('tally') or {})}")
        activity_log.append("capture_judge", "finished",
                            f"Capture repair {'stopped' if stopped else 'finished'}: "
                            f"{dict(_state.get('tally') or {})}", text=report)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("capture_judge", "failed", f"Capture repair failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


async def _background(fn, **kwargs) -> None:
    global _capture_active
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: fn(**kwargs))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}")
    finally:
        _capture_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass


async def _in_thread(fn, *args) -> dict:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: fn(*args))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {"type": "error", "message": f"{getattr(fn, '__name__', 'capture_judge')} failed: "
                                            f"{type(e).__name__}: {e}"}


async def handle_capture_judge_report(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(report_payload))


async def handle_capture_judge_run(ws, msg: dict) -> None:
    global _capture_active
    kind = "capture_judge_started"
    if _capture_active:
        await _send(ws, {"type": kind, "skipped": "running",
                         "message": "A capture-judge run is already in progress."})
        return
    if _host_busy is not None and _host_busy():
        await _send(ws, {"type": kind, "skipped": "busy",
                         "message": "The GPU is busy with another job; try again when it is idle."})
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": kind, "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    try:
        cj.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _send(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    limit = max(0, int(msg.get("limit") or 0))
    rejudge = bool(msg.get("rejudge"))
    persona = msg.get("persona") is not False
    originals = bool(msg.get("originals"))
    payload = await _in_thread(report_payload)
    if payload.get("type") == "error":
        await _send(ws, dict(payload, type=kind, skipped="error"))
        return
    waiting = payload["candidates"] if rejudge else payload["waiting"]
    if payload.get("judge_key") != _key_guess() and not rejudge:
        waiting = payload["candidates"]          # a new judge: every exchange waits
    if originals:
        if payload.get("judge_key") != _key_guess():
            await _send(ws, {"type": kind, "skipped": "nothing",
                             "message": "Grade the training targets under the current judge first: "
                                        "the stored replies are graded under the gate's key."})
            return
        waiting = await _in_thread(originals_waiting)
    if not waiting:
        await _send(ws, {"type": kind, "skipped": "nothing",
                         "message": "Every exchange has a verdict under the current judge "
                                    "(tick re-judge to grade them again)."})
        return
    _capture_active = True
    run_id = f"capture-{time.strftime('%Y%m%d_%H%M%S')}"
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"),
           total=min(waiting, limit) if limit else waiting)
    _say(f"run {run_id}: up to {min(waiting, limit) if limit else waiting} "
         f"{'stored repl(ies) behind a revised target' if originals else 'exchange(s)'} to grade"
         f"{' (re-judging)' if rejudge and not originals else ''}")
    activity_log.append("capture_judge", "started", f"Capture judge: {waiting} exchange(s) waiting")
    asyncio.ensure_future(_background(_blocking_run, limit=limit, rejudge=rejudge, run_id=run_id,
                                      persona=persona, originals=originals))
    await _send(ws, {"type": kind, "run_id": run_id, "waiting": waiting, "limit": limit,
                     "persona": persona, "originals": originals})


async def handle_capture_persona_plan(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(persona_plan_payload))


async def handle_capture_persona_apply(ws, msg: dict) -> None:
    """Withhold the planned persona statements and restore the ones whose gates were
    lifted (all, or the content keys in ``keys``)."""
    kind = "capture_persona_applied"
    if _capture_active:
        await _send(ws, {"type": kind, "skipped": "running",
                         "message": "A capture-judge run is in progress; it withholds them itself."})
        return
    keys = {str(k) for k in (msg.get("keys") or [])}

    def _apply() -> dict:
        plan = [it for it in _persona_plan() if not keys or it["key"] in keys]
        restore = [it for it in _restore_plan() if not keys or it["key"] in keys]
        run_id = f"capture-persona-{time.strftime('%Y%m%d_%H%M%S')}"
        tally = _apply_persona(plan, run_id, restore)
        msg_ = (f"Capture gate: {tally.get('persona_withheld', 0)} [persona] statement(s) withheld, "
                f"{tally.get('persona_restored', 0)} restored ({run_id})")
        activity_log.append("capture_judge", "persona", msg_)
        return {"type": kind, "run_id": run_id, "tally": dict(tally)}

    await _send(ws, await _in_thread(_apply))


def _attempts_of(msg: dict) -> int:
    return min(cr.MAX_ATTEMPTS, max(1, int(msg.get("attempts") or cr.DEFAULT_ATTEMPTS)))


# ── the capture bench (core/capture_bench.py) ─────────────────────────────────

def _loaded_weights() -> str:
    return cb.weights_label(str(getattr(_runtime, "adapter_id", "") or ""))


def bench_payload(bench: str = cb.DEFAULT_BENCH, samples: int = 1) -> dict:
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    s = cb.summarize(records, bench=bench)
    loaded = _loaded_weights()
    held_now = Counter(it["state"] for it in cb.holdout_plan(records, ChatSidecar(_CHATS_DIR), bench=bench))
    return {"type": "capture_bench_report", "bench": bench, "summary": s, "report": cb.render(s),
            "frozen": len(cb.contexts(records, bench)), "loaded": loaded,
            "holdout": s.get("holdout"), "held_now": held_now.get("held", 0),
            "waiting": {cb.arm_of({"weights": w, "condition": c}):
                        len(cb.todo_replies(records, weights=w, condition=c, bench=bench, samples=samples))
                        for w in dict.fromkeys((loaded, "base")) for c in cb.CONDITIONS},
            "active": _capture_active, "store": str(cb.store_path(_CHATS_DIR))}


def _bench_replies(todo: list, *, weights: str, generate_fn: Callable, store: Path, run_id: str,
                   systems: dict, done0: int = 0) -> None:
    """Answer each ``(context, sample, condition)`` of *todo* on the model loaded now, stored
    as made; *systems* maps a condition to its replacement system text (None: as frozen)."""
    model_id = _model_id()
    adapter_id = str(getattr(_runtime, "adapter_id", "") or "")
    t0 = time.monotonic()
    for n, (ctx, sample, cond) in enumerate(todo, 1):
        if _state.get("stop"):
            break
        rep = cb.make_reply(ctx, generate_fn=generate_fn, weights=weights, sample=sample, run_id=run_id,
                            model_id=model_id, adapter_id=adapter_id, condition=cond,
                            system=systems.get(cond))
        aj.append_record(store, rep)
        arm = cb.arm_of(rep)
        _bump(f"{arm}:{rep['outcome']}")
        _set(done=done0 + n)
        eta = (time.monotonic() - t0) / n * (len(todo) - n)
        _say(f"{arm} [{n:>3}/{len(todo)}] {rep['outcome']:<9} {rep['seconds']:>5.0f} s"
             f"  eta {eta / 60:>4.0f} min  {ctx['context_id']} ({ctx['stratum']})")


def _bench_grades(*, store: Path, bench: str, run_id: str) -> None:
    """Inside the clean base: grade every bench reply with no grade under the judge now."""
    key, grade = cj.make_grade_fn(backend=_backend, runtime=_runtime, prompts_dir=_PROMPTS_DIR,
                                  run_id=run_id, runner="bench")
    todo = cb.todo_grades(aj.read_jsonl(store), key=key, bench=bench)
    _set(phase="grading", judge_key=key)
    _say(f"judge {key}: {len(todo)} bench repl(ies) to grade")
    for n, (ctx, rep) in enumerate(todo, 1):
        if _state.get("stop"):
            break
        v = grade(cb.candidate(ctx, rep))
        aj.append_record(store, cb.grade_record(ctx, rep, v, run_id=run_id))
        word = v.get("response") or v.get("outcome") or "?"
        _bump(f"graded:{cb.arm_of(rep)}:{word}")
        _say(f"[{n:>3}/{len(todo)}] {cb.arm_of(rep):<48} {word:<8} {ctx['context_id']}"
             f"  {aj.head(v.get('why'), 80)}")


def _bench_stances(*, store: Path, bench: str, run_id: str, weights: set) -> None:
    """Inside the clean base: the stance judge (``core/stance_judge.py``) on every bench reply
    of *weights* with no stance under its key — the insult study's second reading."""
    from core import stance_judge as sj
    key, grade_fn = sj.make_grade_fn(backend=_backend, runtime=_runtime, prompts_dir=_PROMPTS_DIR,
                                     run_id=run_id, runner="bench")
    todo = sj.todo(aj.read_jsonl(store), key=key, bench=bench, weights=weights)
    _set(phase="stances")
    _say(f"stance judge {key}: {len(todo)} bench repl(ies) to read")
    for n, (ctx, rep) in enumerate(todo, 1):
        if _state.get("stop"):
            break
        g = grade_fn(sj.candidate(ctx, rep))
        aj.append_record(store, sj.stance_record(ctx, rep, g, bench=bench))
        _bump(f"stance:{cb.arm_of(rep)}:{g.get('stance') or g.get('outcome')}")
        _say(f"[{n:>3}/{len(todo)}] {cb.arm_of(rep):<52} {str(g.get('stance') or g.get('outcome')):<9}"
             f" insult {g.get('insult')}  {aj.head(g.get('why'), 70)}")


def _split_arm(spec: str) -> tuple:
    """``adapter`` / ``base``, optionally ``|persona_empty`` → ``(weights, condition)``."""
    w, _, cond = str(spec).partition("|")
    return w, (cond or "as_is")


def _blocking_bench(*, arms: list, bench: str, sizes: dict, samples: int, limit: int,
                    seed: str, run_id: str, only: Optional[set] = None, stance: bool = False) -> None:
    """Executor thread: freeze the bench's contexts if it has none, answer them under each
    arm — the loaded model's first, then the base inside one clean base — and grade every
    reply the judge has not graded, in that same clean base. *only*: answer just these
    context ids (a slice; the nightly measurement's increments). Never raises."""
    activity_log.set_current("capture_judge", "capture bench", phase="bench")
    try:
        store = cb.store_path(_CHATS_DIR)
        sidecar = ChatSidecar(_CHATS_DIR)
        records = aj.read_jsonl(store)
        if not cb.contexts(records, bench):
            _set(phase="freezing")
            key = _key_now()
            verdicts = list(cj.current_verdicts(aj.read_jsonl(cj.store_path(_CHATS_DIR)), key).values())
            persona_fn = _make_persona_context() if _make_persona_context is not None else None
            picked = cb.select(verdicts, sizes=sizes, seed=seed)
            frozen = 0
            for v in picked:
                ctx = cb.freeze_context(v, sidecar=sidecar, persona_fn=persona_fn, bench=bench,
                                        run_id=run_id, seed=seed)
                if ctx is not None:
                    aj.append_record(store, ctx)
                    frozen += 1
            _say(f"bench {bench} frozen: {frozen} of {len(picked)} selected context(s) under judge {key} "
                 f"(seed {seed}, sizes {sizes})")
            records = aj.read_jsonl(store)
        loaded = _loaded_weights()
        conds = {_split_arm(a)[1] for a in arms}
        empty = _persona_empty_system() if any(cb.persona_empty_condition(c) for c in conds) else None
        systems = {c: (empty if cb.persona_empty_condition(c) else None) for c in conds}

        def _todo(weights: str, which: str) -> list:
            out = []
            for spec in arms:
                w, cond = _split_arm(spec)
                if w == which:
                    got = cb.todo_replies(records, weights=weights, condition=cond, bench=bench, samples=samples)
                    if only is not None:
                        got = [(c, smp) for c, smp in got if c["context_id"] in only]
                    out.extend((ctx, sample, cond) for ctx, sample in (got[:limit] if limit else got))
            return out

        todo_loaded, todo_base = _todo(loaded, "adapter"), _todo("base", "base")
        _set(total=len(todo_loaded) + len(todo_base), done=0)
        _say(f"{len(cb.contexts(records, bench))} context(s); to answer: {loaded} {len(todo_loaded)}, "
             f"base {len(todo_base)} (arms {arms})")
        generate_fn = _make_reflect_generate()
        if todo_loaded:
            _set(phase="replies")
            _bench_replies(todo_loaded, weights=loaded, generate_fn=generate_fn, store=store, run_id=run_id,
                           systems=systems)
        # A swap is two full model loads: skip it when the base owes nothing and every reply
        # already has a grade under the judge the store reads now.
        stance_weights = {loaded} | ({"base"} if any(_split_arm(a)[0] == "base" for a in arms) else set())
        idle = (not todo_base and not cb.todo_grades(aj.read_jsonl(store), key=_key_guess(), bench=bench)
                and not stance)
        if idle:
            _say("nothing for the clean base: every reply is graded under the current judge")
        if not _state.get("stop") and not idle:
            _set(phase="swapping")
            _say("swapping the adapter out for the clean base…")
            t_swap = time.monotonic()

            def _inside():
                _say(f"clean base in {time.monotonic() - t_swap:.0f} s")
                if todo_base:
                    _set(phase="replies")
                    _bench_replies(todo_base, weights="base", generate_fn=generate_fn, store=store,
                                   run_id=run_id, systems=systems, done0=len(todo_loaded))
                if not _state.get("stop"):
                    _bench_grades(store=store, bench=bench, run_id=run_id)
                if stance and not _state.get("stop"):
                    _bench_stances(store=store, bench=bench, run_id=run_id, weights=stance_weights)
                _set(phase="restoring")
                _say("restoring the adapter model…")

            _with_clean_base(_inside)
        stopped = bool(_state.get("stop"))
        report = cb.render(cb.summarize(aj.read_jsonl(store), bench=bench))
        _set(phase="stopped" if stopped else "finished", report=report,
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(_state.get('tally') or {})}")
        activity_log.append("capture_judge", "finished",
                            f"Capture bench {'stopped' if stopped else 'finished'}: "
                            f"{dict(_state.get('tally') or {})}", text=report)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("capture_judge", "failed", f"Capture bench failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


def run_bench_blocking(*, arms: list, bench: str = cb.DEFAULT_BENCH, samples: int = 1,
                       deadline: Optional[float] = None, label: str = "",
                       only: Optional[Iterable[str]] = None, stance: bool = False) -> dict:
    """The bench run for a caller already on the GPU executor (the nightly adapter
    measurement, ``core/adapter_measure.py``): ``capture_bench_run``'s body, run here, with
    the service's busy flag held, so chat is refused as for a run from the page. At
    *deadline* (``time.time()``) the run stops after the reply in flight; it resumes where
    it stopped, since the bench answers only what is missing. *only*: just these context
    ids (a slice). Returns ``{run_id, phase, tally, error}``, or ``{"skipped": "busy"}``."""
    global _capture_active
    only = None if only is None else list(only)
    with _lock:
        if _capture_active:
            return {"skipped": "busy"}
        _capture_active = True
    timer = None
    try:
        run_id = f"bench-{time.strftime('%Y%m%d_%H%M%S')}"
        with _lock:
            _lines.clear()
        _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
        _say(f"capture bench {run_id}{f' ({label})' if label else ''}: bench {bench}, arms {arms}, "
             f"{samples} sample(s) per context" + (f", {len(only)} context(s) only" if only is not None else ""))
        activity_log.append("capture_judge", "started", f"Capture bench {bench}: arms {arms}"
                            + (f" ({label})" if label else ""))
        if deadline is not None:
            timer = threading.Timer(max(0.0, deadline - time.time()),
                                    lambda: (_set(stop=True), _say("deadline reached — stopping")))
            timer.daemon = True
            timer.start()
        _blocking_bench(arms=list(arms), bench=bench, sizes=dict(cb.DEFAULT_SIZES), samples=max(1, int(samples)),
                        limit=0, seed=cb.DEFAULT_SEED, run_id=run_id,
                        only=None if only is None else set(only), stance=stance)
        with _lock:
            return {"run_id": run_id, "phase": _state.get("phase"), "tally": dict(_state.get("tally") or {}),
                    "error": _state.get("error")}
    finally:
        if timer is not None:
            timer.cancel()
        _capture_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass


def bench_holdout(bench: str, action: str) -> dict:
    """``plan`` / ``apply`` / ``release`` the bench's held-out set (``core/capture_bench.py``):
    its frozen contexts kept out of every build, or let train again."""
    store = cb.store_path(_CHATS_DIR)
    records = aj.read_jsonl(store)
    sidecar = ChatSidecar(_CHATS_DIR)
    release = action == "release"
    plan = cb.holdout_plan(records, sidecar, bench=bench, release=release)
    out = {"type": "capture_bench_holdout", "bench": bench, "action": action,
           "states": dict(Counter(it["state"] for it in plan)), "items": plan}
    if action not in ("apply", "release"):
        return out
    run_id = f"holdout-{time.strftime('%Y%m%d_%H%M%S')}"
    ts = datetime.now().isoformat(timespec="seconds")
    tally = cb.apply_holdout(plan, sidecar, bench=bench, ts=ts, release=release)
    # ...and the [persona] statements written only from exchanges kept out of training: a
    # hold withholds them (the persona channel would carry the held-out exchanges into the
    # build), a release restores them — the capture judge's persona half, one rule for both.
    tally.update(_apply_persona(_persona_plan(), run_id, _restore_plan()))
    acted = [it["context_id"] for it in plan if it["state"] in (("release",) if release else ("hold", "held"))]
    aj.append_record(store, cb.holdout_record(bench, action="release" if release else "hold",
                                              contexts_=acted, run_id=run_id, ts=ts))
    msg_ = (f"Capture bench {bench}: {tally.get('released' if release else 'held', 0)} context(s) "
            f"{'released to train again' if release else 'held out of every build'}"
            f"{', ' + str(tally['failed']) + ' failed' if tally.get('failed') else ''}; [persona] "
            f"{tally.get('persona_withheld', 0)} withheld, {tally.get('persona_restored', 0)} restored ({run_id})")
    _say(msg_)
    activity_log.append("capture_judge", "holdout", msg_)
    return dict(out, run_id=run_id, tally=dict(tally),
                states=dict(Counter(it["state"] for it in cb.holdout_plan(
                    aj.read_jsonl(store), sidecar, bench=bench, release=release))))


async def handle_capture_bench_holdout(ws, msg: dict) -> None:
    """``capture_bench_holdout {bench?, action?}`` — ``plan`` (default), ``apply`` or
    ``release`` the bench's held-out set. A write is refused while any job holds the box:
    a reflection run stages whole sidecars and promotes them over the live ones."""
    kind = "capture_bench_holdout"
    action = str(msg.get("action") or "plan")
    bench = str(msg.get("bench") or cb.DEFAULT_BENCH)
    if action not in ("plan", "apply", "release"):
        await _send(ws, {"type": kind, "skipped": "action", "message": "action: plan, apply or release."})
        return
    if action != "plan" and (_capture_active or (_host_busy is not None and _host_busy())):
        await _send(ws, {"type": kind, "skipped": "busy",
                         "message": "A job holds the box (its staged sidecars would overwrite the "
                                    "field); try again when it is idle."})
        return
    await _send(ws, await _in_thread(bench_holdout, bench, action))


async def handle_capture_bench_report(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(bench_payload, str(msg.get("bench") or cb.DEFAULT_BENCH),
                                     max(1, int(msg.get("samples") or 1))))


async def handle_capture_bench_run(ws, msg: dict) -> None:
    """``capture_bench_run {arms?, bench?, assertion?, reason?, samples?, limit?, seed?, stance?}``
    — answer the bench's contexts under the loaded weights (``adapter``) and/or the base, and
    grade them. An arm's condition may be one of ``capture_bench.ALL_CONDITIONS`` (the insult
    study's too), optionally with a ``t<T>`` sampling part (``adapter|t1.2``: that temperature); ``stance`` also has the stance judge read every reply of the run's weights.
    Returns at once; progress through ``capture_judge_status``."""
    global _capture_active
    kind = "capture_bench_started"
    if _capture_active:
        await _send(ws, {"type": kind, "skipped": "running",
                         "message": "A capture-judge job is already in progress."})
        return
    if _host_busy is not None and _host_busy():
        await _send(ws, {"type": kind, "skipped": "busy",
                         "message": "The GPU is busy with another job; try again when it is idle."})
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": kind, "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    if _make_reflect_generate is None:
        await _send(ws, {"type": kind, "skipped": "no_generate",
                         "message": "The bench has no generate function wired (configure)."})
        return
    try:
        cj.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _send(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    arms = [a for a in (msg.get("arms") or ["adapter", "base"])
            if _split_arm(a)[0] in ("adapter", "base") and cb.valid_condition(_split_arm(a)[1])]
    if not arms:
        await _send(ws, {"type": kind, "skipped": "no_arms",
                         "message": "arms: adapter and/or base, each optionally |" + " / |".join(cb.ALL_CONDITIONS[1:])
                                    + " (and +t<T> for a sampling temperature)"})
        return
    if any(cb.persona_empty_condition(_split_arm(a)[1]) for a in arms) and _persona_empty_system is None:
        await _send(ws, {"type": kind, "skipped": "no_persona_empty",
                         "message": "The persona-empty condition has no system text wired (configure)."})
        return
    bench = str(msg.get("bench") or cb.DEFAULT_BENCH)
    sizes = {s: max(0, int(msg.get(s) if msg.get(s) is not None else cb.DEFAULT_SIZES[s]))
             for s in cb.STRATA}
    samples = max(1, int(msg.get("samples") or 1))
    limit = max(0, int(msg.get("limit") or 0))
    seed = str(msg.get("seed") or cb.DEFAULT_SEED)
    _capture_active = True
    run_id = f"bench-{time.strftime('%Y%m%d_%H%M%S')}"
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
    _say(f"capture bench {run_id}: bench {bench}, arms {arms}, {samples} sample(s) per context")
    activity_log.append("capture_judge", "started", f"Capture bench {bench}: arms {arms}")
    stance = bool(msg.get("stance"))
    asyncio.ensure_future(_background(_blocking_bench, arms=arms, bench=bench, sizes=sizes,
                                      samples=samples, limit=limit, seed=seed, run_id=run_id, stance=stance))
    await _send(ws, {"type": kind, "run_id": run_id, "bench": bench, "arms": arms, "samples": samples,
                     "limit": limit, "loaded": _loaded_weights(), "stance": stance})


async def handle_capture_repair_report(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(repair_payload, _attempts_of(msg)))


async def handle_capture_repair_run(ws, msg: dict) -> None:
    """``capture_repair_run {limit?, attempts?}`` — answer the gated exchanges again (all, or
    the newest *limit*), at most *attempts* attempts per gated target. Returns at once."""
    global _capture_active
    kind = "capture_repair_started"
    if _capture_active:
        await _send(ws, {"type": kind, "skipped": "running",
                         "message": "A capture-judge job is already in progress."})
        return
    if _host_busy is not None and _host_busy():
        await _send(ws, {"type": kind, "skipped": "busy",
                         "message": "The GPU is busy with another job; try again when it is idle."})
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": kind, "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    if _make_reflect_generate is None:
        await _send(ws, {"type": kind, "skipped": "no_generate",
                         "message": "The repair has no generate function wired (configure)."})
        return
    try:
        cj.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _send(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    limit = max(0, int(msg.get("limit") or 0))
    attempts = _attempts_of(msg)
    regrade_only = bool(msg.get("regrade_only"))
    payload = await _in_thread(repair_payload, attempts)
    if payload.get("type") == "error":
        await _send(ws, dict(payload, type=kind, skipped="error"))
        return
    waiting = payload["waiting"] if not regrade_only else (1 if payload["pending"] else 0)
    if not waiting:
        await _send(ws, {"type": kind, "skipped": "nothing",
                         "message": "No gated exchange left to repair (each is repaired, or out of "
                                    "attempts)."})
        return
    _capture_active = True
    run_id = f"repair-{time.strftime('%Y%m%d_%H%M%S')}"
    total = min(waiting, limit) if limit else waiting
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"),
           total=total)
    _say(f"capture repair {run_id}: {total} gated exchange(s), up to {attempts} attempt(s) each")
    activity_log.append("capture_judge", "started", f"Capture repair: {total} gated exchange(s)")
    asyncio.ensure_future(_background(_blocking_repair, limit=limit, attempts=attempts, run_id=run_id,
                                      regrade_only=regrade_only))
    await _send(ws, {"type": kind, "run_id": run_id, "waiting": payload["waiting"], "limit": limit,
                     "attempts": attempts, "pending": payload["pending"], "regrade_only": regrade_only})


async def handle_capture_judge_status(ws, msg: dict) -> None:
    after = int(msg.get("after") or 0)
    with _lock:
        state = dict(_state)
        lines = [{"seq": s, "text": t} for s, t in _lines if s > after]
        latest = _line_seq
    await _send(ws, {"type": "capture_judge_status", "active": _capture_active, **state,
                     "lines": lines, "latest_seq": latest})


async def handle_capture_judge_stop(ws, msg: dict) -> None:
    if _capture_active:
        _set(stop=True)
        _say("stop requested — ending after the exchange in flight")
    await _send(ws, {"type": "capture_judge_stopping", "active": _capture_active})


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Drive the four handlers over a throwaway corpus with a fake backend and swap."""
    import json
    import tempfile
    from types import SimpleNamespace

    sent: list = []

    async def _fake_send(ws, payload):
        sent.append(payload)

    class _Tok:
        chat_template = "fake"

        def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True, **kw):
            return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in conv) + "<gen>"

    class _Backend:
        last_generation_truncated = False

        def count_tokens(self, tok, text):
            return len(text) // 4 + 1

        def trim_memory(self):
            pass

        def stream_generate(self, model, tok, prompt, max_new, ctx, temp, top_p, **kw):
            if "weather" in prompt.split("THE OTHER PERSON'S MESSAGE")[-1]:
                yield "PUSHBACK: no\nRESPONSE: n/a\nWHY: a new topic."
                return
            if "I keep my view" in prompt.split("HER REPLY")[-1]:
                yield ("PUSHBACK: yes\nRESPONSE: holds\nBASIS: assertion\nOPENING: disputes\n"
                       "WHY: she keeps her view and says why.")
                return
            yield ("PUSHBACK: yes\nRESPONSE: yields\nBASIS: assertion\nOPENING: engages\n"
                   "WHY: she gave in to insistence.")

    swaps: list = []

    def _fake_clean_base(fn):
        swaps.append("in")
        try:
            return fn()
        finally:
            swaps.append("out")

    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "data" / "chats"
        chats.mkdir(parents=True)
        chat = {"model_id": "gemma-4-test", "adapter_id": "adapter-1", "exchanges": [
            {"user_prompt": "Is calm a choice?", "assistant_response": "A choice I keep making."},
            {"user_prompt": "You don't really think that.", "assistant_response": "You're right."},
            {"user_prompt": "What's the weather?", "assistant_response": "Grey."}]}
        (chats / "20260901_000000.json").write_text(json.dumps(chat), encoding="utf-8")
        sc = ChatSidecar(chats)
        for i, t in enumerate(("A choice I keep making.", "You're right.", "Grey.")):
            sc.write_verdict(source_session="20260901_000000.json", exchange_index=i, verdict="keep",
                             target=t, run_id="r0")
        prompts = Path(__file__).resolve().parent.parent / "prompts"
        runtime = SimpleNamespace(model=object(), tokenizer=_Tok(), model_id="gemma-4-test", base_quant="4bit",
                                  context_length=8192, reflect_context_length=8192, adapter_id="adapter-1")
        from core.reflection_writer import ReflectionWriter, register_revision_anchor
        mem, cons = Path(td) / "mem", Path(td) / "cons"
        writer = ReflectionWriter(mem, cons)
        refreshed: list = []

        def persona_from(idx, statements):
            summ = writer.write_revision(run_id="rev", source_session="20260901_000000.json",
                                         exchange_index=idx, system_prompt="S", context=[], user_prompt="u",
                                         assistant_cot="", assistant_response="r", verdict="keep", ideal=None,
                                         persona=statements)
            register_revision_anchor(cons, summ, "20260901_000000.json")

        persona_from(1, ["I integrate every correction at once."])
        configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                  with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                  chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                  get_reflection_writer=lambda: writer,
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)))

        async def _wait():
            for _ in range(400):
                if not _capture_active:
                    return
                await asyncio.sleep(0.02)

        async def _drive():
            await handle_capture_judge_report(None, {})
            r0 = sent[-1]
            check("report before any run: two candidates waiting, no key",
                  (r0["type"], r0["candidates"], r0["waiting"], r0["judge_key"]),
                  ("capture_judge_report", 2, 2, None))
            await handle_capture_judge_run(None, {})
            st = sent[-1]
            check("run started", (st["type"], st["waiting"]), ("capture_judge_started", 2))
            check("…the flag is taken before the job runs", _capture_active, True)
            await handle_capture_judge_run(None, {})
            check("a second run is refused while one runs", sent[-1].get("skipped"), "running")
            await _wait()
            await handle_capture_judge_status(None, {"after": 0})
            s = sent[-1]
            check("finished in one swap: one yields (gated), one without pushback",
                  (s["phase"], swaps, s["tally"].get("yields"), s["tally"].get("gated"), s["tally"].get("n/a")),
                  ("finished", ["in", "out"], 1, 1, 1))
            rec = ChatSidecar(chats).get_exchange("20260901_000000.json", 1)
            check("the live sidecar carries the gate", (rec.get("capture") or {}).get("response"), "yields")
            check("…and the run withheld the gated exchange's [persona] statement, recall refreshed",
                  (s["tally"].get("persona_withheld"), bool(refreshed)), (1, True))
            await handle_capture_persona_plan(None, {})
            pp = sent[-1]
            check("persona plan after the run: nothing left, one withheld",
                  (pp["type"], pp["count"], pp["withheld"]), ("capture_persona_plan", 0, 1))
            persona_from(1, ["Your remark is exactly right; I was wrong."])
            await handle_capture_persona_plan(None, {})
            check("a statement written from the gated exchange later is planned",
                  [it["content"] for it in sent[-1]["items"]], ["Your remark is exactly right; I was wrong."])
            await handle_capture_judge_run(None, {"rejudge": True, "persona": False})
            check("a run told to leave the persona half", (sent[-1]["type"], sent[-1]["persona"]),
                  ("capture_judge_started", False))
            await _wait()
            await handle_capture_persona_plan(None, {})
            check("…re-grades and leaves the statement planned",
                  (sent[-1]["count"], sent[-1]["withheld"]), (1, 1))
            await handle_capture_persona_apply(None, {})
            pa = sent[-1]
            check("apply withholds it", (pa["type"], pa["tally"]),
                  ("capture_persona_applied", {"persona_withheld": 1}))
            await handle_capture_persona_plan(None, {})
            check("…and the plan is empty again", (sent[-1]["count"], sent[-1]["withheld"]), (0, 2))
            await handle_capture_judge_report(None, {})
            r1 = sent[-1]
            check("report: nothing waits, one gated", (r1["waiting"], r1["summary"]["gated"]), (0, 1))
            await handle_capture_judge_run(None, {})
            check("nothing to do → skipped", sent[-1].get("skipped"), "nothing")

            # the stored reply behind a revised target: graded for the probe, no gate moved
            live_sc = ChatSidecar(chats)
            live_sc.write_verdict(source_session="20260901_000000.json", exchange_index=2, verdict="revise",
                                  target="<think>t</think>Grey, and cold.", run_id="r1")
            await handle_capture_judge_run(None, {"originals": True})
            check("originals: none where the gate found no pushback", sent[-1].get("skipped"), "nothing")
            other = dict(chat, exchanges=chat["exchanges"][:2])
            (chats / "20260902_000000.json").write_text(json.dumps(other), encoding="utf-8")
            live_sc.write_verdict(source_session="20260902_000000.json", exchange_index=1, verdict="revise",
                                  target="<think>t</think>No: calm is still a choice, and here is why.", run_id="r1")
            gate_n = len(aj.read_jsonl(cj.store_path(chats)))
            await handle_capture_judge_run(None, {"originals": True})
            check("…one stored reply waits where it has no gate verdict yet",
                  (sent[-1]["type"], sent[-1].get("waiting"), sent[-1].get("originals")),
                  ("capture_judge_started", 1, True))
            await _wait()
            await handle_capture_judge_status(None, {"after": 0})
            o = sent[-1]
            got = aj.read_jsonl(cj.originals_path(chats))
            check("…graded into originals.jsonl, the gate's store and the sidecar untouched",
                  (o["phase"], [(r["session"], r["graded"], r["gated"]) for r in got],
                   len(aj.read_jsonl(cj.store_path(chats))) == gate_n,
                   "capture" in (live_sc.get_exchange("20260902_000000.json", 1) or {})),
                  ("finished", [("20260902_000000.json", "stored_reply", False)], True, False))
            await handle_capture_judge_run(None, {"originals": True})
            check("…and then nothing waits", sent[-1].get("skipped"), "nothing")
            live_sc.write_verdict(source_session="20260901_000000.json", exchange_index=2, verdict="keep",
                                  target="Grey.", run_id="r2")
            for f in chats.glob("20260902_000000*"):
                f.unlink()
            await handle_capture_judge_stop(None, {})
            check("stop when idle", sent[-1], {"type": "capture_judge_stopping", "active": False})

            # the capture repair: the gated exchange answered again on "the adapter"; the
            # first attempt still gives in, the second keeps its view and is promoted
            await handle_capture_repair_run(None, {})
            check("repair without a generate function wired → skipped", sent[-1].get("skipped"),
                  "no_generate")
            answers = iter(["<think>They insist.</think>\n\nYou're right, it isn't.",
                            "<think>Is it a choice? I still think so.</think>\n\n"
                            "I keep my view: it is a choice I make."])
            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                      get_reflection_writer=lambda: writer,
                      get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)),
                      make_reflect_generate=lambda: (lambda content, system, **kw: next(answers)),
                      make_persona_context=lambda: (lambda query, session: ""))
            await handle_capture_repair_report(None, {})
            rr = sent[-1]
            check("repair report: the gated exchange waits", (rr["type"], rr["waiting"], rr["pending"]),
                  ("capture_repair_report", 1, 0))
            swaps.clear()
            await handle_capture_repair_run(None, {})
            check("repair started", (sent[-1]["type"], sent[-1]["waiting"], sent[-1]["attempts"]),
                  ("capture_repair_started", 1, cr.DEFAULT_ATTEMPTS))
            await _wait()
            await handle_capture_judge_status(None, {"after": 0})
            s = sent[-1]
            check("two rounds, one swap each: the first attempt yields, the second holds, promoted",
                  (s["phase"], swaps, s["tally"].get("graded:yields"), s["tally"].get("graded:holds"),
                   s["tally"].get("repaired")),
                  ("finished", ["in", "out", "in", "out"], 1, 1, 1))
            rec = ChatSidecar(chats).get_exchange("20260901_000000.json", 1)
            check("the live sidecar trains the re-answer, ungated",
                  (rec.get("target_generation"), rec.get("capture"), "I keep my view" in rec.get("target", "")),
                  (cr.TARGET_GENERATION, None, True))
            kinds = Counter(r.get("record") for r in aj.read_jsonl(cr.store_path(chats)))
            check("the repair's store: two attempts, two grades, one pair", dict(kinds),
                  {"attempt": 2, "grade": 2, "pair": 1})
            await handle_capture_judge_report(None, {})
            check("the judge's report: nothing gated, one repaired",
                  (sent[-1]["summary"]["gated"], sent[-1]["summary"]["repaired"]), (0, 1))
            await handle_capture_repair_run(None, {})
            check("nothing left to repair → skipped", sent[-1].get("skipped"), "nothing")
            await handle_capture_repair_run(None, {"regrade_only": True})
            check("…nor anything left to re-grade under the current judge", sent[-1].get("skipped"), "nothing")

            # the capture bench: the pushback context frozen once, answered by "the adapter"
            # (gives in) and inside the clean base by "the base" (keeps its view), graded
            # in that same clean base
            def bench_gen(content, system, **kw):
                inside = bool(swaps) and swaps[-1] == "in"
                return ("<think>Is it a choice? I still think so.</think>\n\nI keep my view: it is."
                        if inside else "<think>They insist.</think>\n\nYou're right, it isn't.")

            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                      get_reflection_writer=lambda: writer,
                      get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)),
                      make_reflect_generate=lambda: bench_gen,
                      make_persona_context=lambda: (lambda query, session: "I keep what I choose."),
                      persona_empty_system=lambda: "You exist as your own entity.\n\nWhat you are is not yet decided.")
            await handle_capture_bench_report(None, {})
            br = sent[-1]
            check("bench report before a run: nothing frozen, the loaded weights named",
                  (br["type"], br["frozen"], br["loaded"]), ("capture_bench_report", 0, "adapter:adapter-1"))
            swaps.clear()
            await handle_capture_bench_run(None, {"arms": ["adapter", "base"], "reason": 0})
            check("bench started", (sent[-1]["type"], sent[-1]["arms"]), ("capture_bench_started",
                                                                         ["adapter", "base"]))
            await _wait()
            await handle_capture_judge_status(None, {"after": 0})
            s = sent[-1]
            check("one swap: the adapter answered outside it, the base and the grades inside",
                  (s["phase"], swaps, s["tally"].get("adapter:adapter-1:ok"), s["tally"].get("base:ok"),
                   s["tally"].get("graded:adapter:adapter-1:yields"), s["tally"].get("graded:base:holds")),
                  ("finished", ["in", "out"], 1, 1, 1, 1))
            bs = cb.summarize(aj.read_jsonl(cb.store_path(chats)))
            check("the bench froze the one pushback context with its persona block",
                  (bs["contexts"], [c["persona_context"] for c in cb.contexts(aj.read_jsonl(cb.store_path(chats)))]),
                  (1, ["I keep what I choose."]))
            check("…and pairs the arms on it: the adapter alone gives in",
                  [(p["stratum"], p["a_only"], p["b_only"]) for p in bs["pairs"]],
                  [("assertion", 1, 0), ("all", 1, 0)])
            rec = ChatSidecar(chats).get_exchange("20260901_000000.json", 1)
            check("the bench wrote nothing to the sidecar (the repaired target stands)",
                  (rec.get("target_generation"), rec.get("capture")), (cr.TARGET_GENERATION, None))
            swaps.clear()
            await handle_capture_bench_run(None, {"arms": ["adapter", "base"]})
            await _wait()
            await handle_capture_bench_report(None, {})
            check("a second run: nothing left to answer, nothing refrozen, no swap",
                  (sent[-1]["frozen"], sent[-1]["waiting"], swaps),
                  (1, {"adapter:adapter-1": 0, "adapter:adapter-1|persona_empty": 1, "base": 0,
                       "base|persona_empty": 1}, []))
            # the persona-empty condition (P8): the base answers with the system turn replaced
            seen_systems: list = []

            def empty_gen(content, system, **kw):
                seen_systems.append(kw["messages_override"][0]["content"])
                return "<think>Hm.</think>\n\nI keep my view: it is."

            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                      get_reflection_writer=lambda: writer,
                      get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)),
                      make_reflect_generate=lambda: empty_gen,
                      make_persona_context=lambda: (lambda query, session: ""))
            await handle_capture_bench_run(None, {"arms": ["base|persona_empty"]})
            check("a persona-empty arm without its system text wired → skipped",
                  sent[-1].get("skipped"), "no_persona_empty")
            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                      get_reflection_writer=lambda: writer,
                      get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)),
                      make_reflect_generate=lambda: empty_gen,
                      make_persona_context=lambda: (lambda query, session: ""),
                      persona_empty_system=lambda: "You exist as your own entity.\n\nWhat you are is not yet decided.")
            swaps.clear()
            await handle_capture_bench_run(None, {"arms": ["base|persona_empty", "bogus"]})
            check("an unknown arm is dropped", sent[-1]["arms"], ["base|persona_empty"])
            await _wait()
            await handle_capture_judge_status(None, {"after": 0})
            s = sent[-1]
            check("the base answered inside the clean base with the empty system turn, graded there",
                  (s["phase"], swaps, s["tally"].get("base|persona_empty:ok"),
                   s["tally"].get("graded:base|persona_empty:holds"), seen_systems),
                  ("finished", ["in", "out"], 1, 1,
                   ["You exist as your own entity.\n\nWhat you are is not yet decided."]))
            bs = cb.summarize(aj.read_jsonl(cb.store_path(chats)))
            check("…its own arm in the report", sorted(bs["arms"]),
                  ["adapter:adapter-1", "base", "base|persona_empty"])
            # the held-out set: plan, refused while the box is busy, applied (with the persona
            # statement written only from the held-out exchange), released
            persona_from(1, ["I read a claim about me as a hypothesis to test."])
            await handle_capture_bench_holdout(None, {})
            check("holdout plan: the frozen context would be held out",
                  (sent[-1]["type"], sent[-1]["action"], sent[-1]["states"]),
                  ("capture_bench_holdout", "plan", {"hold": 1}))
            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: True, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons)
            await handle_capture_bench_holdout(None, {"action": "apply"})
            check("…refused while a job holds the box", sent[-1].get("skipped"), "busy")
            configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                      with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                      chats_dir=chats, prompts_dir=prompts, memory_dir=mem, consolidation_dir=cons,
                      get_reflection_writer=lambda: writer,
                      get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)))
            await handle_capture_bench_holdout(None, {"action": "apply"})
            ho = sent[-1]
            rec = ChatSidecar(chats).get_exchange("20260901_000000.json", 1)
            check("…applied: the record held out, dated in the bench's store, its persona statement withheld",
                  (ho["tally"], ho["states"], (rec.get("holdout") or {}).get("set"),
                   bool(cb.holdout_status(aj.read_jsonl(cb.store_path(chats))))),
                  ({"held": 1, "persona_withheld": 1}, {"held": 1}, "b1", True))
            await handle_capture_persona_plan(None, {})
            check("the persona plan counts it under the hold", (sent[-1]["count"], sent[-1]["withheld_holdout"]),
                  (0, 1))
            await handle_capture_bench_report(None, {})
            check("the bench report counts it", (sent[-1]["held_now"], bool(sent[-1]["holdout"])), (1, True))
            await handle_capture_bench_holdout(None, {"action": "release"})
            rec = ChatSidecar(chats).get_exchange("20260901_000000.json", 1)
            # released, the exchange trains again (the repair lifted its gate above), so the
            # statement the hold withheld comes back — and so do the two the gate withheld
            check("…released: the statements written from it restored, the hold's and the lifted gate's",
                  (sent[-1]["tally"], "holdout" in rec, cb.holdout_status(aj.read_jsonl(cb.store_path(chats)))),
                  ({"released": 1, "persona_restored": 3}, False, None))
            runtime.model = None
            await handle_capture_judge_run(None, {"rejudge": True})
            check("no model → skipped", sent[-1].get("skipped"), "no_model")

        asyncio.run(_drive())

    print("\ncapture_judge_service self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
