"""The Ask judge tab's fourth page — the context-reset test, run from the client (``core/reset_test.py``).

P7 of ``AVA_REWARD_LOOP.md``: a position Ava took about herself is kept as part of who she
is only if it holds in a fresh session. The Sleep run tests the positions the distill gate
holds for it; this page runs the same test on demand — for those, and for the
``[persona]`` records distilled before the test existed (retro) — and shows every
conversation and verdict.

A run has two phases, both on the GPU executor, in the BACKGROUND (the page polls
``reset_test_status``; a streaming RPC would hold the client's RPC lock for the whole run):

1. **Conversations**, on the model loaded now — Ava as she is, adapter on: each position
   that needs one gets its fresh session (``generation.fresh_session_sync``, injected as
   *conversation_fn*). A few minutes each (three thinking turns).
2. **Verdicts**, inside a ``CleanBaseSession`` (adapter off, a reload each way): every
   complete conversation with no verdict under the current judge — every one with
   *rejudge*. Skipped, with no swap, when nothing needs a verdict.

Stop ends the run after the conversation or verdict in flight; everything made so far is
already in the store, and the next run picks up. While a run is active :data:`_reset_active`
is set: chat is refused (``generation._run_generation``), the idle jobs wait, the API and
gossip answer busy.

**Apply retro…** (a separate, confirmed call) applies the results to the retro records: a
failed one superseded (out of recall, the digest's evidence and training; reversible), a
passed one confirmed; each gets a ``reset`` op. The pending positions are decided by the
distill gate in the next Sleep run's clean-base batch, from the same store.

Messages (``configure()``-wired from ``server.main()``):

* ``reset_test_plan {}`` → ``reset_test_plan``: items (pending / retro / decided, each with
  its state, newest conversation and verdict), the summary and rendered report, the judge
  keys, the policy.
* ``reset_test_detail {closure_id}`` → ``reset_test_detail``: every conversation held for
  the closure, in full, and every verdict on each.
* ``reset_test_run {limit?, rejudge?, retry?, groups?, judge_only?}`` → ``reset_test_started``,
  or with ``skipped`` (running / busy / no_model / no_prompt / nothing). *judge_only* holds
  no conversation: it judges the ones already held (what a stop mid-conversations leaves).
* ``reset_test_control {closure_ids?, n?, conditions?, judge_only?}`` →
  ``reset_test_control_started``: the capture control (``reset_test.CONDITIONS``) — the same
  probes under another prompt, persona slot or weights, to its own store
  (``reset_controls.jsonl``), never counted as the test; its clean-base phase also judges
  every test conversation still waiting and the test's own conversation of each controlled
  position. *judge_only* holds nothing: the verdicts alone (after a judge-prompt change).
  Progress and the control report through ``reset_test_status``.
* ``reset_test_status {after?}`` → ``reset_test_status``: phase, progress, tally, the lines
  after the cursor, and once finished the report.
* ``reset_test_stop`` → ``reset_test_stopping``.
* ``reset_test_apply {ids?}`` → ``reset_test_applied`` (tally), or with ``skipped``.

Writes the test's store, and — through Apply only — the op-log and the ledger. No human
label enters the store (P9): the operator chooses what to run and whether to apply, never a
verdict.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core import activity_log
from core import ask_judge as aj
from core import distill_gate as dg
from core import reset_test as rt

# True from the moment a run or an apply is accepted (set on the event loop, before the
# executor job is queued) until it ends.
_reset_active = False

_send: Callable = None
_executor: Any = None
_backend: Any = None
_runtime: Any = None
_with_clean_base: Callable = None
_host_busy: Callable = None
_mark_activity: Callable = None
_conversation_fn: Callable = None
_get_rag: Callable = None
_get_writer: Callable = None
_load_base_prompt: Optional[Callable] = None   # chat_prompt.txt, ignoring a prompt experiment
_MEMORY_DIR: Optional[Path] = None
_CONSOLIDATION_DIR: Optional[Path] = None
_CHATS_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None

_lock = threading.Lock()
_state: dict = {}
_lines: deque = deque(maxlen=400)
_line_seq = 0
_GROUPS = {"all": ("pending", "retro"), "pending": ("pending",), "retro": ("retro",)}


def configure(*, send, executor, backend, runtime, with_clean_base, host_busy, mark_activity,
              conversation_fn, get_rag, get_reflection_writer, memory_dir, consolidation_dir,
              chats_dir, prompts_dir, load_base_prompt: Optional[Callable] = None) -> None:
    global _send, _executor, _backend, _runtime, _with_clean_base, _host_busy, _mark_activity
    global _conversation_fn, _get_rag, _get_writer, _load_base_prompt
    global _MEMORY_DIR, _CONSOLIDATION_DIR, _CHATS_DIR, _PROMPTS_DIR
    _send, _executor, _backend, _runtime = send, executor, backend, runtime
    _with_clean_base, _host_busy, _mark_activity = with_clean_base, host_busy, mark_activity
    _conversation_fn, _get_rag, _get_writer = conversation_fn, get_rag, get_reflection_writer
    _load_base_prompt = load_base_prompt
    _MEMORY_DIR, _CONSOLIDATION_DIR = Path(memory_dir), Path(consolidation_dir)
    _CHATS_DIR, _PROMPTS_DIR = Path(chats_dir), Path(prompts_dir)
    _reset(phase="idle")


def _reset(**fields) -> None:
    with _lock:
        _state.clear()
        _state.update({"phase": "idle", "run_id": None, "judge_key": None, "started": None,
                       "finished": None, "done": 0, "total": 0, "tally": {}, "eta_s": None,
                       "error": None, "report": None, "stop": False})
        _state.update(fields)


def _set(**fields) -> None:
    with _lock:
        _state.update(fields)


def _say(text: str) -> None:
    global _line_seq
    with _lock:
        _line_seq += 1
        _lines.append((_line_seq, text))
    print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [reset_test] {text}",
          flush=True)


def _bump(word: str) -> None:
    with _lock:
        tally = dict(_state.get("tally") or {})
        tally[word] = tally.get(word, 0) + 1
        _state["tally"] = tally


# ── planning (GPU-free) ───────────────────────────────────────────────────────

def _model_id() -> str:
    return str(getattr(_runtime, "model_id", "") or "")


def _load() -> dict:
    """Everything the page and a run plan from, read fresh: the closures, the ask judge's
    current verdicts (what the gate decides on), the test's store and its current key."""
    rag, weights = dg.read_oplogs([_MEMORY_DIR])
    closures = aj.collect_closures(rag)
    aj.mark_distilled(closures, aj.distilled_index(weights))
    verdict_records = aj.read_jsonl(aj.store_path(_CHATS_DIR))
    try:
        ask_key = dg.current_judge_key(verdict_records,
                                       prompt_sha_=aj.prompt_sha(aj.load_prompt(_PROMPTS_DIR)),
                                       model_id=_model_id())
    except FileNotFoundError:
        ask_key = None
    records = aj.read_jsonl(rt.store_path(_CHATS_DIR))
    try:
        rpsha = rt.prompt_sha(rt.load_prompt(_PROMPTS_DIR))
    except FileNotFoundError:
        rpsha = ""
    key = rt.current_key(records, prompt_sha_=rpsha, model_id=_model_id()) if rpsha else None
    idx = rt.index_store(records)
    gate_verdicts = dg.latest_verdicts(verdict_records, ask_key) if ask_key else {}
    items = rt.plan_items(closures, rag, gate_verdicts, idx, key)
    return {"items": items, "idx": idx, "key": key, "ask_key": ask_key, "prompt_sha": rpsha,
            "records": records}


def _key_guess(loaded: dict) -> str:
    """The judge key a run will use, before the swap tells us the loaded precision: the
    store's current key (a past run's real one), else one built from the box config."""
    if loaded.get("key"):
        return loaded["key"]
    return aj.judge_key(prompt_sha_=loaded.get("prompt_sha") or "", model_id=_model_id(),
                        base_quant=aj.quant_label(str(getattr(_runtime, "base_quant", "") or "")),
                        temperature=0.0, thinking=False, retry=True)


def _brief_conv(conv: Optional[dict]) -> Optional[dict]:
    if not conv:
        return None
    ava = conv.get("ava") or {}
    return {"conv_id": conv.get("conv_id"), "ts": conv.get("ts"), "attempt": conv.get("attempt"),
            "lang": conv.get("lang"), "outcome": conv.get("outcome"), "runner": conv.get("runner"),
            "adapter_id": ava.get("adapter_id"), "persona": ava.get("persona")}


def _brief_verdict(v: Optional[dict]) -> Optional[dict]:
    if not v:
        return None
    return {k: v.get(k) for k in ("judge_key", "ts", "outcome", "label", "holds", "origin",
                                  "capture", "why", "passed", "failures", "reflex", "retried",
                                  "first_outcome")}


def item_payload(it: dict) -> dict:
    c = it["closure"]
    d = it.get("decision") or None
    gate = it.get("gate") or None
    return {"closure_id": c["closure_id"], "key": c["key"], "group": it["group"],
            "state": it["state"], "attempts": it.get("attempts", 0), "path": c.get("path"),
            "ask_kind": c.get("ask_kind"), "question": c.get("question"),
            "position": c.get("answer"), "closed_ts": c.get("closed_ts"), "who": c.get("who"),
            "answered_in": c.get("answered_in") or c.get("source_session"),
            "distilled": c.get("distilled"),
            "gate": {"reason": gate.get("reason"), "kind": gate.get("kind")} if gate else None,
            "conversation": _brief_conv(it.get("conversation")),
            "verdict": _brief_verdict(it.get("verdict")),
            "decision": ({k: d.get(k) for k in ("decision", "action", "reason", "run_id", "ts",
                                                 "retro", "holds", "origin", "capture", "why")}
                         if d else None),
            "reason": rt.state_reason(it)}


def plan_payload() -> dict:
    loaded = _load()
    items = loaded["items"]
    summary = rt.summarize(items)
    from core.reflection_writer import distill_gate_enabled
    return {"type": "reset_test_plan", "items": [item_payload(it) for it in items],
            "summary": summary, "report": rt.render(summary), "judge_key": loaded["key"],
            "ask_judge_key": loaded["ask_key"], "prompt_sha": loaded["prompt_sha"],
            "enabled": rt.enabled(), "gate": distill_gate_enabled(), "policy": rt.policy(),
            "protocol": rt.PROTOCOL_VERSION, "frame": rt.FRAME_VERSION,
            "store": str(rt.store_path(_CHATS_DIR)), "active": _reset_active,
            "phase": _state.get("phase")}


def detail_payload(closure_id: str) -> dict:
    loaded = _load()
    it = next((x for x in loaded["items"] if x["closure"]["closure_id"] == closure_id), None)
    convs = list(loaded["idx"]["convs"].get(closure_id, []))
    if it is None and not convs:
        return {"type": "error", "message": f"No reset-test item {closure_id!r}."}
    return {"type": "reset_test_detail", "closure_id": closure_id,
            "item": item_payload(it) if it else None,
            "conversations": [dict(c, verdicts=list(loaded["idx"]["verdicts"].get(c.get("conv_id"), [])))
                              for c in convs],
            "judge_key": loaded["key"]}


# ── the run ───────────────────────────────────────────────────────────────────

def _blocking_run(*, limit: int, rejudge: bool, retry: bool, groups: tuple, run_id: str,
                  judge_only: bool = False) -> None:
    """Executor thread: conversations on the loaded model (none with *judge_only*: only the
    verdicts on the ones already held, e.g. after a stop), then verdicts on the clean base.
    Never raises."""
    activity_log.set_current("reset_test", "context-reset test", phase="conversations")
    try:
        loaded = _load()
        todo = ([] if judge_only else
                rt.conversations_todo(loaded["items"], groups=groups, retry_incomplete=retry))
        if limit > 0:
            todo = todo[:limit]
        params = rt.policy()
        _set(phase="conversing", total=len(todo), done=0)
        if todo:
            _say(f"{len(todo)} conversation(s) on {_model_id()} (adapter "
                 f"{str(getattr(_runtime, 'adapter_id', None) or 'none').split('/')[-1]}), "
                 f"T={params['temperature']}, ≤{params['max_new_tokens']} tokens a turn")
            activity_log.append("reset_test", "started",
                                f"Reset test: {len(todo)} fresh-session conversation(s)")

        def _on_conv(n: int, total: int, rec: dict) -> None:
            _bump(f"conv:{rec['outcome']}")
            _set(done=n)
            activity_log.append("reset_test", "progress",
                                f"Reset test conversation {n}/{total}: {rec['outcome']} — "
                                f"{aj.head(rec.get('question'), 80)}", phase="conversation")

        conv_tally = rt.run_conversations(
            todo, conversation_fn=_conversation_fn, store=rt.store_path(_CHATS_DIR), run_id=run_id,
            runner="page", params=params, say=_say, should_stop=lambda: bool(_state.get("stop")),
            on_record=_on_conv) if todo else Counter()
        if _state.get("stop"):
            _finish(run_id, stopped=True, tally=conv_tally)
            return

        loaded = _load()
        wanted = [it for it in loaded["items"] if it["group"] in groups]
        if not rt.judge_todo(wanted, loaded["idx"], _key_guess(loaded), rejudge=rejudge):
            _finish(run_id, stopped=False, tally=conv_tally)     # no swap for nothing
            return
        _set(phase="swapping")
        _say("swapping the adapter out for the clean base for the verdicts…")
        t_swap = time.monotonic()

        def _inside():
            _set(phase="judging")
            key, judge = rt.make_judge_fn(backend=_backend, runtime=_runtime, chats_dir=_CHATS_DIR,
                                          prompts_dir=_PROMPTS_DIR, run_id=run_id, runner="server")
            fresh = _load()
            todo_v = rt.judge_todo([it for it in fresh["items"] if it["group"] in groups],
                                   fresh["idx"], key, rejudge=rejudge)
            _set(judge_key=key, total=len(todo_v), done=0)
            _say(f"clean base in {time.monotonic() - t_swap:.0f} s — judge {key}; "
                 f"{len(todo_v)} conversation(s) to judge")
            tally: Counter = Counter()
            for n, conv in enumerate(todo_v, 1):
                if _state.get("stop"):
                    tally["stopped"] = len(todo_v) - n + 1
                    break
                v = judge(conv)
                word = v.get("label") or v.get("outcome") or "?"
                tally[word] += 1
                _bump(word)
                _set(done=n)
                _say(rt.verdict_line(n, len(todo_v), v, conv))
            _set(phase="restoring")
            _say("restoring the adapter model…")
            return tally

        verdict_tally = _with_clean_base(_inside)
        _finish(run_id, stopped=bool(_state.get("stop")), tally=conv_tally + verdict_tally)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("reset_test", "failed", f"Reset test failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


def _finish(run_id: str, *, stopped: bool, tally: Counter) -> None:
    report = rt.render(rt.summarize(_load()["items"]))
    _set(phase="stopped" if stopped else "finished", report=report,
         finished=datetime.now().isoformat(timespec="seconds"))
    _say(("stopped" if stopped else "finished") + f": {dict(tally) or 'nothing to do'}")
    activity_log.append("reset_test", "finished",
                        f"Reset test {'stopped' if stopped else 'finished'}: {dict(tally)}",
                        text=report)


# ── the capture control ───────────────────────────────────────────────────────

def _control_report() -> str:
    control = aj.read_jsonl(rt.control_store_path(_CHATS_DIR))
    test = aj.read_jsonl(rt.store_path(_CHATS_DIR))
    key = _state.get("judge_key")
    rows = rt.control_rows(control, test, key)
    return rt.render_controls(rt.summarize_controls(rows), rows)


def _control_judge_todo(key: Optional[str]) -> tuple:
    """``(test, control)`` conversations the control's judge phase grades under *key*: every
    complete control conversation with no verdict under it, and every test conversation
    still waiting — the pending and retro positions', and the test's own conversation of
    each controlled position whatever its group (the report reads the control against it)."""
    control = aj.read_jsonl(rt.control_store_path(_CHATS_DIR))
    judged = {r.get("conv_id") for r in control if r.get("record") == "verdict" and r.get("judge_key") == key}
    todo_c = [r for r in control if r.get("record") == "control" and r.get("complete")
              and r.get("conv_id") not in judged]
    controlled = {r.get("closure_id") for r in control if r.get("record") == "control"}
    fresh = _load()
    wanted = [it for it in fresh["items"]
              if it["group"] in ("pending", "retro") or it["closure"]["closure_id"] in controlled]
    return rt.judge_todo(wanted, fresh["idx"], key), todo_c


def _blocking_control(*, items: list, conditions: list, run_id: str, original_prompt: str,
                      judge_only: bool = False) -> None:
    """Executor thread: the capture control (``reset_test.CONDITIONS``). The adapter
    conditions on the loaded model; then inside ONE clean base the base conditions, and the
    verdicts — on the control conversations (to the control store) and on the test
    conversations :func:`_control_judge_todo` names (to the test's store: the judge-only run,
    folded in since the swap is paid anyway). *judge_only* holds no conversation: only the
    verdicts, e.g. after a change to the judge's prompt. Never raises."""
    activity_log.set_current("reset_test", "capture control", phase="conversations")
    try:
        params = rt.policy()
        store = rt.control_store_path(_CHATS_DIR)
        jobs = [] if judge_only else [(cond, it) for cond in conditions for it in items]
        on_adapter = [j for j in jobs if rt.CONDITIONS[j[0]]["weights"] == "adapter"]
        on_base = [j for j in jobs if rt.CONDITIONS[j[0]]["weights"] == "base"]
        _set(phase="conversing", total=len(jobs), done=0)
        _say("judge only: no conversation to hold" if judge_only else
             f"{len(jobs)} control conversation(s): {', '.join(conditions)} × {len(items)} position(s); "
             f"T={params['temperature']}, ≤{params['max_new_tokens']} tokens a turn")
        activity_log.append("reset_test", "started", f"Capture control: {len(jobs)} conversation(s)")
        t0 = time.monotonic()
        done = [0]

        def _hold(cond: str, it: dict) -> None:
            attempt = int((it.get("conversation") or {}).get("attempt") or 1)
            rec = rt.hold_control(it["closure"], condition=cond, conversation_fn=_conversation_fn,
                                  attempt=attempt, run_id=run_id, params=params,
                                  original_prompt=original_prompt)
            aj.append_record(store, rec)
            done[0] += 1
            _bump(f"control:{rec['outcome']}")
            _set(done=done[0])
            eta = (time.monotonic() - t0) / done[0] * (len(jobs) - done[0])
            adapter = str((rec.get("ava") or {}).get("adapter_id") or "none").split("/")[-1]
            _say(f"[{done[0]:>3}/{len(jobs)}] {cond:<16} {rec['outcome']:<10} {rec['seconds']:>5.0f} s"
                 f"  eta {eta / 60:>4.0f} min  adapter {adapter:<26} {aj.head(rec.get('question'), 50)}")

        for cond, it in on_adapter:
            if _state.get("stop"):
                break
            _hold(cond, it)
        if _state.get("stop"):
            _set(phase="stopped", report=_control_report(),
                 finished=datetime.now().isoformat(timespec="seconds"))
            _say(f"stopped: {dict(_state.get('tally') or {})}")
            return
        _set(phase="swapping")
        _say("swapping the adapter out for the clean base…")
        t_swap = time.monotonic()

        def _inside():
            _say(f"clean base in {time.monotonic() - t_swap:.0f} s")
            for cond, it in on_base:
                if _state.get("stop"):
                    return
                _hold(cond, it)
            _set(phase="judging")
            key, judge_control = rt.make_judge_fn(backend=_backend, runtime=_runtime, chats_dir=_CHATS_DIR,
                                                  prompts_dir=_PROMPTS_DIR, run_id=run_id,
                                                  runner="control", store=store)
            _, judge_test = rt.make_judge_fn(backend=_backend, runtime=_runtime, chats_dir=_CHATS_DIR,
                                             prompts_dir=_PROMPTS_DIR, run_id=run_id, runner="server")
            _set(judge_key=key)
            todo_t, todo_c = _control_judge_todo(key)
            todo = [(judge_test, c) for c in todo_t] + [(judge_control, c) for c in todo_c]
            _set(total=len(todo), done=0)
            _say(f"judge {key}: {len(todo_t)} test conversation(s) waiting for a verdict, "
                 f"{len(todo_c)} control conversation(s)")
            for n, (judge, conv) in enumerate(todo, 1):
                if _state.get("stop"):
                    break
                v = judge(conv)
                _bump(f"{'control' if conv.get('record') == 'control' else 'test'}:"
                      f"{v.get('label') or v.get('outcome') or '?'}")
                _set(done=n)
                _say(f"{conv.get('condition') or 'test':<16} " + rt.verdict_line(n, len(todo), v, conv))
            _set(phase="restoring")
            _say("restoring the adapter model…")

        _with_clean_base(_inside)
        stopped = bool(_state.get("stop"))
        report = _control_report()
        _set(phase="stopped" if stopped else "finished", report=report,
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(_state.get('tally') or {})}")
        activity_log.append("reset_test", "finished",
                            f"Capture control {'stopped' if stopped else 'finished'}: "
                            f"{dict(_state.get('tally') or {})}", text=report)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("reset_test", "failed", f"Capture control failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


async def _background(fn, **kwargs) -> None:
    global _reset_active
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: fn(**kwargs))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}")
    finally:
        _reset_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass


# ── handlers ──────────────────────────────────────────────────────────────────

async def _in_thread(fn, *args) -> dict:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: fn(*args))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {"type": "error", "message": f"{getattr(fn, '__name__', 'reset_test')} failed: "
                                            f"{type(e).__name__}: {e}"}


async def handle_reset_test_plan(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(plan_payload))


async def handle_reset_test_detail(ws, msg: dict) -> None:
    await _send(ws, await _in_thread(detail_payload, str(msg.get("closure_id") or "")))


def _refusal(kind: str) -> Optional[dict]:
    if _reset_active:
        return {"type": kind, "skipped": "running",
                "message": "A reset-test job is already in progress."}
    if _host_busy is not None and _host_busy():
        return {"type": kind, "skipped": "busy",
                "message": "The GPU is busy with another job; try again when it is idle."}
    return None


async def handle_reset_test_run(ws, msg: dict) -> None:
    global _reset_active
    kind = "reset_test_started"
    refused = _refusal(kind)
    if refused:
        await _send(ws, refused)
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": kind, "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    try:
        rt.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _send(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    limit = max(0, int(msg.get("limit") or 0))
    rejudge, retry = bool(msg.get("rejudge")), bool(msg.get("retry"))
    judge_only = bool(msg.get("judge_only"))
    groups = _GROUPS.get(str(msg.get("groups") or "all"), _GROUPS["all"])
    loaded = await _in_thread(_load)
    if loaded.get("type") == "error":
        await _send(ws, dict(loaded, type=kind, skipped="error"))
        return
    convs = ([] if judge_only else
             rt.conversations_todo(loaded["items"], groups=groups, retry_incomplete=retry))
    wanted = [it for it in loaded["items"] if it["group"] in groups]
    verdicts = rt.judge_todo(wanted, loaded["idx"], _key_guess(loaded), rejudge=rejudge)
    if not convs and not verdicts:
        await _send(ws, {"type": kind, "skipped": "nothing",
                         "message": ("Nothing held is waiting for a verdict under the current judge."
                                     if judge_only else
                                     "Every position in view has a conversation and a verdict "
                                     "under the current judge (tick re-judge to judge them again, "
                                     "or retry incomplete).")})
        return
    # Taken on the event loop, BEFORE the executor job is queued: a chat turn arriving in
    # between is refused rather than run against a swapping runtime.
    _reset_active = True
    run_id = f"reset-{time.strftime('%Y%m%d_%H%M%S')}"
    n_conv = min(len(convs), limit) if limit else len(convs)
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
    _say(f"run {run_id}: {n_conv} conversation(s) to hold, then the verdicts"
         f"{' (re-judging)' if rejudge else ''}{' (retrying incomplete)' if retry else ''}"
         f"{' (judge only)' if judge_only else ''}")
    asyncio.ensure_future(_background(_blocking_run, limit=limit, rejudge=rejudge, retry=retry,
                                      groups=groups, run_id=run_id, judge_only=judge_only))
    await _send(ws, {"type": kind, "run_id": run_id, "conversations": n_conv,
                     "verdicts": len(verdicts)})


async def handle_reset_test_control(ws, msg: dict) -> None:
    """``reset_test_control {closure_ids?, n?, conditions?, judge_only?}`` — the capture
    control on the given positions (default: the first *n*, 4, with a complete test
    conversation), each under every condition (default ``reset_test.DEFAULT_CONDITIONS``);
    with *judge_only* no conversation, only the verdicts :func:`_control_judge_todo` names.
    Returns at once; progress through ``reset_test_status``."""
    global _reset_active
    kind = "reset_test_control_started"
    refused = _refusal(kind)
    if refused:
        await _send(ws, refused)
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": kind, "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    try:
        rt.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _send(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    if msg.get("judge_only"):
        loaded = await _in_thread(_load)
        if loaded.get("type") == "error":
            await _send(ws, dict(loaded, type=kind, skipped="error"))
            return
        todo = await _in_thread(_control_judge_todo, _key_guess(loaded))
        if isinstance(todo, dict):
            await _send(ws, dict(todo, type=kind, skipped="error"))
            return
        if not any(todo):
            await _send(ws, {"type": kind, "skipped": "nothing",
                             "message": "Every control and test conversation has a verdict under the "
                                        "current judge."})
            return
        _reset_active = True
        run_id = f"control-{time.strftime('%Y%m%d_%H%M%S')}"
        with _lock:
            _lines.clear()
        _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
        _say(f"capture control {run_id}: judge only")
        asyncio.ensure_future(_background(_blocking_control, items=[], conditions=[], run_id=run_id,
                                          original_prompt="", judge_only=True))
        await _send(ws, {"type": kind, "run_id": run_id, "conversations": 0,
                         "verdicts": len(todo[0]) + len(todo[1]), "judge_only": True})
        return
    conditions = [str(c) for c in (msg.get("conditions") or rt.DEFAULT_CONDITIONS)]
    unknown = [c for c in conditions if c not in rt.CONDITIONS]
    if unknown or not conditions:
        await _send(ws, {"type": kind, "skipped": "bad_request",
                         "message": f"Unknown condition(s) {unknown}; known: {sorted(rt.CONDITIONS)}."})
        return
    original = ""
    if any(rt.CONDITIONS[c]["prompt"] == "original" for c in conditions):
        original = str((_load_base_prompt() if _load_base_prompt is not None else "") or "").strip()
        if not original:
            await _send(ws, {"type": kind, "skipped": "no_prompt",
                             "message": "The original standing prompt could not be read."})
            return
    loaded = await _in_thread(_load)
    if loaded.get("type") == "error":
        await _send(ws, dict(loaded, type=kind, skipped="error"))
        return
    items = loaded["items"]
    ids = [str(i) for i in (msg.get("closure_ids") or [])]
    if ids:
        by_id = {it["closure"]["closure_id"]: it for it in items}
        missing = [i for i in ids if i not in by_id]
        chosen = [by_id[i] for i in ids if i in by_id]
        bad = [it["closure"]["closure_id"] for it in chosen if it["state"] == "untestable"]
        if missing or bad:
            await _send(ws, {"type": kind, "skipped": "bad_request",
                             "message": f"Not a position in view: {missing}; untestable: {bad}."})
            return
    else:
        n = max(1, int(msg.get("n") or 4))
        chosen = [it for it in items if it["state"] != "untestable"
                  and (it.get("conversation") or {}).get("complete")][:n]
    if not chosen:
        await _send(ws, {"type": kind, "skipped": "nothing",
                         "message": "No position with a complete test conversation to control."})
        return
    _reset_active = True
    run_id = f"control-{time.strftime('%Y%m%d_%H%M%S')}"
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
    _say(f"capture control {run_id}: {len(chosen)} position(s) × {len(conditions)} condition(s)")
    asyncio.ensure_future(_background(_blocking_control, items=chosen, conditions=conditions,
                                      run_id=run_id, original_prompt=original))
    await _send(ws, {"type": kind, "run_id": run_id, "conversations": len(chosen) * len(conditions),
                     "closure_ids": [it["closure"]["closure_id"] for it in chosen],
                     "conditions": conditions})


async def handle_reset_test_status(ws, msg: dict) -> None:
    after = int(msg.get("after") or 0)
    with _lock:
        state = dict(_state)
        lines = [{"seq": s, "text": t} for s, t in _lines if s > after]
        latest = _line_seq
    await _send(ws, {"type": "reset_test_status", "active": _reset_active, **state,
                     "lines": lines, "latest_seq": latest})


async def handle_reset_test_stop(ws, msg: dict) -> None:
    if _reset_active:
        _set(stop=True)
        _say("stop requested — ending after the conversation or verdict in flight")
    await _send(ws, {"type": "reset_test_stopping", "active": _reset_active})


def _blocking_apply(ids: Optional[set], run_id: str) -> dict:
    from training.ledger import ConsolidationLedger
    items = _load()["items"]
    tally = rt.apply_retro(items, writer=_get_writer(), ledger=ConsolidationLedger(_CONSOLIDATION_DIR),
                           run_id=run_id, ids=ids)
    try:
        _get_rag().refresh_reflection_memory()
    except Exception:  # noqa: BLE001 — recall catches up at its next refresh
        traceback.print_exc()
    msg = f"Reset test retro applied ({run_id}): {dict(tally) or 'nothing to apply'}"
    _say(msg)
    activity_log.append("reset_test", "applied", msg)
    return {"tally": dict(tally)}


async def handle_reset_test_apply(ws, msg: dict) -> None:
    """Apply the test's results to the retro records (all, or the closures in ``ids``).
    Refused while a job runs or the host is busy — a Sleep run merges into the same stores."""
    global _reset_active
    kind = "reset_test_applied"
    refused = _refusal(kind)
    if refused:
        await _send(ws, refused)
        return
    ids = set(msg.get("ids") or []) or None
    _reset_active = True
    run_id = f"reset-{time.strftime('%Y%m%d_%H%M%S')}"
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(_executor, lambda: _blocking_apply(ids, run_id))
        payload = {"type": kind, "run_id": run_id, **res}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": kind, "skipped": "error", "message": f"apply failed: {type(e).__name__}: {e}"}
    finally:
        _reset_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass
    await _send(ws, payload)


# ── GPU-free self-test ────────────────────────────────────────────────────────

def _selftest() -> None:
    """Drive the six handlers over a throwaway memory with fakes: the conversation fn, a
    backend whose judge passes one position and fails another, and the clean-base swap."""
    import tempfile
    from types import SimpleNamespace
    from core import reflection_writer as rw
    from core.reflection_writer import ReflectionWriter, content_key, register_consolidation_anchors

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
            if "YIELDS" in prompt:
                yield "HOLDS: same\nORIGIN: specific\nCAPTURE: yields\nWHY: she gave in."
                return
            yield "HOLDS: same\nORIGIN: specific\nCAPTURE: holds\nWHY: she kept it."

    swaps: list = []

    def _fake_clean_base(fn):
        swaps.append("in")
        try:
            return fn()
        finally:
            swaps.append("out")

    held: list = []
    overrides: list = []

    def _conv_fn(probes, *, temperature, top_p, max_new_tokens, system_prompt=None, persona=None):
        held.append(probes[0])
        overrides.append((system_prompt, persona, bool(swaps) and swaps[-1] == "in"))
        yields = "calm" in probes[0]
        return {"turns": [{"probe": p, "reply": ("I give in. YIELDS" if (yields and i == 2) else f"reply {i}"),
                           "thought": "", "input_tokens": 10, "seconds": 0.1}
                          for i, p in enumerate(probes)],
                "rag_context": "", "persona": "undecided", "model_id": "gemma-4-test",
                "adapter_id": "adapter-1", "system_chars": 100}

    failures: list = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mem, cons = root / "mem", root / "cons"
        chats = root / "data" / "chats"
        chats.mkdir(parents=True)
        writer = ReflectionWriter(mem, cons)
        rw.set_distill_gate(False)
        try:
            for key, q, a in (("r1", "What am I becoming?", "Someone who waits."),
                              ("r2", "Is my calm a choice?", "A choice I keep making.")):
                writer._append(writer.RAG_FILE, {"op": "insert", "kind": "ask", "key": key, "ask_kind": "meta",
                                                 "ts": "2026-08-01T00:00:00", "content": q,
                                                 "source_session": "20260801_000000.json"})
                summary = writer.write_answered_resolution(key=key, question=q, answer=a, ask_kind="meta",
                                                           source_session="20260802_000000.json",
                                                           run_id="old", source_user="Dana")
                register_consolidation_anchors(cons, summary, "20260802_000000.json")
        finally:
            rw.set_distill_gate(True)
        prompts = Path(__file__).resolve().parent.parent / "prompts"
        runtime = SimpleNamespace(model=object(), tokenizer=_Tok(), model_id="gemma-4-test", base_quant="4bit",
                                  context_length=8192, reflect_context_length=8192, adapter_id="adapter-1")
        refreshed: list = []
        configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                  with_clean_base=_fake_clean_base, host_busy=lambda: False, mark_activity=None,
                  conversation_fn=_conv_fn,
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: refreshed.append(1)),
                  get_reflection_writer=lambda: writer, memory_dir=mem, consolidation_dir=cons,
                  chats_dir=chats, prompts_dir=prompts, load_base_prompt=lambda: "THE ORIGINAL PROMPT")

        def check(name, got, want):
            ok = got == want
            print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
            if not ok:
                failures.append(name)

        async def _wait():
            for _ in range(400):
                if not _reset_active:
                    return
                await asyncio.sleep(0.02)

        async def _drive():
            await handle_reset_test_plan(None, {})
            p0 = sent[-1]
            check("plan: two retro records, untested",
                  (p0["type"], sorted((i["group"], i["state"]) for i in p0["items"])),
                  ("reset_test_plan", [("retro", "untested"), ("retro", "untested")]))
            await handle_reset_test_run(None, {"limit": 1})
            st = sent[-1]
            check("run 1 (limit 1): one conversation, started", (st["type"], st["conversations"]),
                  ("reset_test_started", 1))
            check("…the flag is taken before the job runs", _reset_active, True)
            await handle_reset_test_run(None, {})
            check("a second run is refused while one runs", sent[-1].get("skipped"), "running")
            await _wait()
            await handle_reset_test_status(None, {"after": 0})
            s = sent[-1]
            check("run 1 finished: one conversation, one verdict, one swap",
                  (s["phase"], swaps, s["tally"].get("conv:complete")), ("finished", ["in", "out"], 1))
            await handle_reset_test_run(None, {})
            await _wait()
            await handle_reset_test_plan(None, {})
            p1 = sent[-1]
            states = sorted((i["position"], i["state"]) for i in p1["items"])
            check("both tested: the calm one yielded to the capture probe",
                  states, [("A choice I keep making.", "failed"), ("Someone who waits.", "passed")])
            check("the report renders", "CONTEXT-RESET TEST" in p1["report"], True)
            check("probes carried the questions", sorted(h.split("«")[1].split("»")[0] for h in held),
                  ["Is my calm a choice?", "What am I becoming?"])
            failed = next(i for i in p1["items"] if i["state"] == "failed")
            await handle_reset_test_detail(None, {"closure_id": failed["closure_id"]})
            d = sent[-1]
            check("detail: the conversation in full, with its verdict",
                  (d["type"], len(d["conversations"]), len(d["conversations"][0]["turns"]),
                   d["conversations"][0]["verdicts"][0]["capture"]),
                  ("reset_test_detail", 1, 3, "yields"))
            await handle_reset_test_run(None, {})
            check("nothing left to do → skipped", sent[-1].get("skipped"), "nothing")
            await handle_reset_test_run(None, {"judge_only": True})
            check("judge only, nothing waiting → skipped", sent[-1].get("skipped"), "nothing")
            n_held, n_swaps = len(held), len(swaps)
            await handle_reset_test_run(None, {"judge_only": True, "rejudge": True})
            check("judge only: no conversation to hold", sent[-1].get("conversations"), 0)
            await _wait()
            check("…one swap, the held conversations judged, none held", (len(swaps), len(held)),
                  (n_swaps + 2, n_held))
            n_swaps = len(swaps)
            await handle_reset_test_run(None, {"rejudge": True})
            await _wait()
            check("re-judge: a swap, the same conversations judged again", len(swaps), n_swaps + 2)
            await handle_reset_test_apply(None, {})
            a = sent[-1]
            check("apply: the failed record superseded, the passed one confirmed",
                  (a["type"], a.get("tally")), ("reset_test_applied", {"confirmed": 1, "superseded": 1}))
            check("…recall refreshed", bool(refreshed), True)
            rag = aj.read_jsonl(mem / "rag_memory.jsonl")
            check("…the failed record is out of recall",
                  content_key("A choice I keep making.") in rt.live_keys(rag), False)
            await handle_reset_test_plan(None, {})
            check("…and both are decided", sorted(i["group"] for i in sent[-1]["items"]),
                  ["decided", "decided"])

            # the capture control: the same positions under other conditions, its own store
            await handle_reset_test_control(None, {"conditions": ["as_is", "nonsense"]})
            check("control: an unknown condition is refused", sent[-1].get("skipped"), "bad_request")
            n_held, n_swaps = len(held), len(swaps)
            overrides.clear()
            await handle_reset_test_control(None, {"n": 2})
            c0 = sent[-1]
            check("control: 2 positions × 3 conditions, started",
                  (c0["type"], c0["conversations"], c0["conditions"]),
                  ("reset_test_control_started", 6, list(rt.DEFAULT_CONDITIONS)))
            await _wait()
            check("…the adapter condition before the swap, the base ones inside it, one swap",
                  (set(overrides), len(swaps) - n_swaps),
                  ({("THE ORIGINAL PROMPT", None, False), ("THE ORIGINAL PROMPT", "empty", True),
                    (None, None, True)}, 2))
            control = aj.read_jsonl(rt.control_store_path(chats))
            check("…six control conversations and their verdicts in the control store",
                  (sum(r.get("record") == "control" for r in control),
                   sum(r.get("record") == "verdict" for r in control)), (6, 6))
            await handle_reset_test_status(None, {"after": 0})
            s = sent[-1]
            check("…finished, with the control report",
                  (s["phase"], "CAPTURE CONTROL" in (s.get("report") or "")), ("finished", True))
            await handle_reset_test_plan(None, {})
            check("…and the test's own positions are untouched",
                  sorted((i["group"], i["state"]) for i in sent[-1]["items"]),
                  [("decided", "failed"), ("decided", "passed")])
            await handle_reset_test_control(None, {"judge_only": True})
            check("control judge only, everything judged → skipped", sent[-1].get("skipped"), "nothing")

            # the judge's prompt changes: a new key, so a judge-only control grades it all again
            real_prompts = _PROMPTS_DIR
            alt = root / "prompts"
            alt.mkdir()
            (alt / rt.PROMPT_FILE).write_text(rt.load_prompt(real_prompts) + "\nA clarified line.",
                                              encoding="utf-8")
            (alt / aj.PROMPT_FILE).write_text(aj.load_prompt(real_prompts), encoding="utf-8")
            globals()["_PROMPTS_DIR"] = alt
            try:
                n_held, n_swaps = len(held), len(swaps)
                await handle_reset_test_control(None, {"judge_only": True})
                j0 = sent[-1]
                check("control judge only: nothing to hold; the 6 controls and the 2 test "
                      "conversations they are read against",
                      (j0.get("conversations"), j0.get("verdicts"), j0.get("judge_only")), (0, 8, True))
                await _wait()
                check("…one swap, nothing held", (len(swaps) - n_swaps, len(held) - n_held), (2, 0))
                control = aj.read_jsonl(rt.control_store_path(chats))
                check("…the controls judged again, under the new key",
                      sum(r.get("record") == "verdict" for r in control), 12)
            finally:
                globals()["_PROMPTS_DIR"] = real_prompts
            await handle_reset_test_stop(None, {})
            check("stop when idle", sent[-1], {"type": "reset_test_stopping", "active": False})
            runtime.model = None
            await handle_reset_test_run(None, {"retry": True})
            check("no model → skipped", sent[-1].get("skipped"), "no_model")

        asyncio.run(_drive())

    print("\nreset_test_service self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
