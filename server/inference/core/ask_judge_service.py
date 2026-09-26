"""The Ask judge tab's server side — the frozen critic run in-process, and its store read back.

The same pass as ``server/judge_asks.py`` (AVA_REWARD_LOOP.md §3.4, §7 item 18), run from
the client instead of a console: the run holds the GPU executor inside a
``CleanBaseSession`` (adapter OFF, a full model reload each way), so the box does not have
to be stopped. It runs in the BACKGROUND — the tab starts it and polls
``ask_judge_status``, the way the Sleep tab follows a reflection run — because a full run
is ~15 minutes, and a streaming RPC would hold the client's single RPC lock that long and
freeze every other tab. A run can be stopped; the closure in flight finishes first, and
everything judged so far is already in the store (a later run under the same judge
resumes).

While a run is active, :data:`_judge_active` is set: chat is refused
(``generation._run_generation``), the idle jobs wait (``idle_scheduler`` external busy),
and the API / gossip sidecar answer busy — inside the swap the runtime holds the bare
base, or for a moment no model at all.

The generate is ``ask_judge.make_backend_generate_fn`` over the swapped-in base, NOT the
reflect generate: the critic sees its own prompt and nothing else (no temporal anchor,
no reflect-lane framing), and so the same weights give the same judge key here and in
the CLI (the quant is read off the loaded model, ``ask_judge.effective_quant``).

Messages (``configure()``-wired from ``server.main()``):

* ``ask_judge_report {judge_key?, label?, kind?, path?, text?, thinking?}`` →
  ``ask_judge_report``: the store's judges, the chosen judge's summary table and list
  rows, the op-log census (what grounds each stratum) and how many closures a run would
  judge now.
* ``ask_judge_detail {closure_id, judge_key?}`` → ``ask_judge_detail``: the record, plus
  every other judge's verdict on the same closure.
* ``ask_judge_run {limit?, thinking?, rejudge?}`` → ``ask_judge_started``, or the same
  type with ``skipped`` (running / busy / no_model / no_prompt / nothing).
* ``ask_judge_status {after?}`` → ``ask_judge_status``: phase, progress, tally, ETA, the
  progress lines after the cursor, and once finished the rendered report.
* ``ask_judge_stop`` → ``ask_judge_stopping``.

Writes only the judge's store (through ``core.ask_judge``); nothing here changes an ask,
a persona record or a training row. The distill gate (``core.distill_gate``) reads the
verdicts to decide what a resolved ask distills into (P3: a judge gates training); no
human label enters the store (P9).
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core import activity_log
from core import ask_judge as aj
from core import distill_gate as dg

# True from the moment a run is accepted (set on the event loop, before the executor job
# is queued — a chat turn arriving in between must already be refused) until it ends.
_judge_active = False

# ── Injected server capabilities (populated by configure()) ──
_send: Callable = None
_executor: Any = None
_backend: Any = None
_runtime: Any = None
_with_clean_base: Callable = None
_host_busy: Callable = None
_mark_activity: Callable = None
_MEMORY_DIR: Optional[Path] = None
_CHATS_DIR: Optional[Path] = None
_SNIPPETS_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None

_lock = threading.Lock()
_state: dict = {}
_lines: deque = deque(maxlen=400)
_line_seq = 0


def configure(*, send, executor, backend, runtime, with_clean_base, host_busy,
              mark_activity, memory_dir, chats_dir, snippets_dir, prompts_dir) -> None:
    """Wire in the server capabilities the tab's handlers use (once, at startup)."""
    global _send, _executor, _backend, _runtime, _with_clean_base, _host_busy
    global _mark_activity, _MEMORY_DIR, _CHATS_DIR, _SNIPPETS_DIR, _PROMPTS_DIR
    _send, _executor, _backend, _runtime = send, executor, backend, runtime
    _with_clean_base, _host_busy, _mark_activity = with_clean_base, host_busy, mark_activity
    _MEMORY_DIR = Path(memory_dir)
    _CHATS_DIR = Path(chats_dir)
    _SNIPPETS_DIR = Path(snippets_dir) if snippets_dir else None
    _PROMPTS_DIR = Path(prompts_dir)
    _reset(phase="idle")


def _reset(**fields) -> None:
    with _lock:
        _state.clear()
        _state.update({"phase": "idle", "run_id": None, "judge_key": None, "started": None,
                       "finished": None, "done": 0, "total": 0, "tally": {}, "eta_s": None,
                       "error": None, "report": None, "stop": False, "limit": 0,
                       "thinking": False})
        _state.update(fields)


def _set(**fields) -> None:
    with _lock:
        _state.update(fields)


def _say(text: str) -> None:
    """One progress line: kept for the tab's cursor, and in the server log."""
    global _line_seq
    with _lock:
        _line_seq += 1
        _lines.append((_line_seq, text))
    print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [ask_judge] {text}",
          flush=True)


def _provisional_key(system_prompt: str, thinking: bool) -> str:
    """The judge key a run would use, before the swap tells us the loaded precision."""
    return aj.judge_key(prompt_sha_=aj.prompt_sha(system_prompt),
                        model_id=str(getattr(_runtime, "model_id", "") or ""),
                        base_quant=aj.quant_label(str(getattr(_runtime, "base_quant", "") or "")),
                        temperature=0.0, thinking=thinking, retry=True)


# ── the run ───────────────────────────────────────────────────────────────────

def _blocking_run(closures: list, grounding: dict, system_prompt: str, *, limit: int,
                  thinking: bool, rejudge: bool, run_id: str) -> None:
    """Executor thread: swap to the clean base, judge, swap back. Never raises."""
    activity_log.set_current("ask_judge", "judging closed asks", phase="swap")
    activity_log.append("ask_judge", "started",
                        f"Ask judge: {len(closures)} closures in view; swapping to the clean base…")
    store = aj.store_path(_CHATS_DIR)
    psha = aj.prompt_sha(system_prompt)
    _set(phase="swapping")
    _say("swapping the adapter out for the clean base…")
    t_swap = time.monotonic()

    def _inside():
        _set(phase="judging")
        model_id = str(_runtime.model_id or "")
        context = max(int(_runtime.context_length or 0),
                      int(_runtime.reflect_context_length or 0)) or 16384
        quant = aj.effective_quant(_runtime.model,
                                   aj.quant_label(str(getattr(_runtime, "base_quant", "") or "")))
        key = aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant,
                           temperature=0.0, thinking=thinking, retry=True)
        done = set() if rejudge else aj.judged_ids(aj.read_jsonl(store), key)
        todo = [c for c in closures if c["closure_id"] not in done]
        if limit > 0:
            todo = todo[:limit]
        _set(judge_key=key, total=len(todo))
        _say(f"clean base in {time.monotonic() - t_swap:.0f} s — {model_id} {quant}, adapter off; "
             f"judge {key}; {len(todo)} to judge ({len(closures) - len(todo)} skipped as judged"
             f"{' or past the limit' if limit > 0 else ''})")
        generate_fn = aj.make_backend_generate_fn(_backend, lambda: _runtime.model,
                                                  lambda: _runtime.tokenizer, model_id, context)
        provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None,
                      "prompt_sha": psha, "frame": aj.FRAME_VERSION, "temperature": 0.0,
                      "context": context, "thinking_first": thinking, "retry": True,
                      "runner": "server"}
        t0 = time.monotonic()

        def _on_record(n: int, total: int, rec: dict) -> None:
            eta = (time.monotonic() - t0) / n * (total - n)
            _say(aj.progress_line(n, total, rec, eta_s=eta))
            with _lock:
                tally = dict(_state.get("tally") or {})
                w = aj.outcome_word(rec)
                tally[w] = tally.get(w, 0) + 1
                _state.update(done=n, eta_s=round(eta), tally=tally)
            activity_log.append("ask_judge", "progress",
                                f"Ask judge {n}/{total}: {aj.outcome_word(rec)} — "
                                f"{aj.head(rec.get('question'), 80)}", phase="judge")

        tally = aj.judge_many(
            todo, grounding, generate_fn=generate_fn, system_prompt=system_prompt,
            count_tokens=lambda s: _backend.count_tokens(_runtime.tokenizer, s),
            context=context, provenance=provenance, store=store, judge_run=run_id,
            judge_key_=key, temperature=0.0, thinking=thinking, retry=True,
            on_record=_on_record, should_stop=lambda: bool(_state.get("stop")))
        # Frame 5: an ask the judge found filed under the wrong kind (a `user` ask about
        # Ava, a `meta` ask about the person) is judged again as the kind its subject
        # calls for, so the distill gate and its retro plan read the right rule's verdict.
        if not _state.get("stop"):
            again = dg.needs_rejudge(closures, dg.latest_verdicts(aj.read_jsonl(store), key))
            if again:
                _say(f"re-judging {len(again)} closure(s) filed under the wrong kind, as the "
                     f"kind their subject calls for")
            for i, (c, kind) in enumerate(again, 1):
                if _state.get("stop"):
                    break
                rec = aj.judge_closure(
                    aj.as_kind(c, kind), grounding.get(c["closure_id"]) or {"kind": "none"},
                    generate_fn=generate_fn, system_prompt=system_prompt,
                    count_tokens=lambda s: _backend.count_tokens(_runtime.tokenizer, s),
                    context=context, provenance=provenance, temperature=0.0,
                    thinking=thinking, retry=True, judge_run=run_id, judge_key_=key)
                aj.append_record(store, rec)
                tally["rejudged"] += 1
                _say(f"as {kind}: " + aj.progress_line(i, len(again), rec))
        _set(phase="restoring")
        _say("restoring the adapter model…")
        return key, tally

    try:
        key, tally = _with_clean_base(_inside)
        latest = aj.latest_per_closure(aj.read_jsonl(store), key)
        report = aj.render(aj.summarize(latest), records=latest)
        stopped = bool(tally.get("stopped"))
        _set(phase="stopped" if stopped else "finished", report=report,
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(tally)}")
        activity_log.append("ask_judge", "finished",
                            f"Ask judge {'stopped' if stopped else 'finished'}: {dict(tally)}",
                            text=report)
    except Exception as e:  # noqa: BLE001 — a failed run is a reported state, not a crash
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("ask_judge", "failed", f"Ask judge failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


async def _run_in_background(*args, **kwargs) -> None:
    global _judge_active
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: _blocking_run(*args, **kwargs))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set(phase="failed", error=f"{type(e).__name__}: {e}")
    finally:
        _judge_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass


# ── handlers ──────────────────────────────────────────────────────────────────

def _reply(ws, payload: dict):
    return _send(ws, payload)


async def handle_ask_judge_run(ws, msg: dict) -> None:
    global _judge_active
    kind = "ask_judge_started"
    if _judge_active:
        await _reply(ws, {"type": kind, "skipped": "running",
                          "message": "A judge run is already in progress."})
        return
    if _host_busy is not None and _host_busy():
        await _reply(ws, {"type": kind, "skipped": "busy",
                          "message": "The GPU is busy with another job; try again when it is idle."})
        return
    if _runtime is None or _runtime.model is None:
        await _reply(ws, {"type": kind, "skipped": "no_model",
                          "message": "No model is loaded — load one from the Chat tab first."})
        return
    try:
        system_prompt = aj.load_prompt(_PROMPTS_DIR)
    except FileNotFoundError as e:
        await _reply(ws, {"type": kind, "skipped": "no_prompt", "message": str(e)})
        return
    limit = max(0, int(msg.get("limit") or 0))
    thinking = bool(msg.get("thinking"))
    rejudge = bool(msg.get("rejudge"))
    loop = asyncio.get_running_loop()
    closures, grounding = await loop.run_in_executor(
        None, lambda: aj.load_closures(_MEMORY_DIR, _CHATS_DIR, _SNIPPETS_DIR))
    # A closure the cut-resolution repair replaced is history: its complete answer is a
    # closure of its own, judged in its place.
    closures = [c for c in closures if not c.get("repaired")]
    key = _provisional_key(system_prompt, thinking)
    done = set() if rejudge else aj.judged_ids(aj.read_jsonl(aj.store_path(_CHATS_DIR)), key)
    pending = [c for c in closures if c["closure_id"] not in done]
    if not pending:
        await _reply(ws, {"type": kind, "skipped": "nothing",
                          "message": f"All {len(closures)} closures are already judged under this "
                                     f"judge; tick re-judge to judge them again."})
        return
    # Taken on the event loop, BEFORE the executor job is queued: a chat turn that
    # arrives in between is refused rather than run against a swapping runtime.
    _judge_active = True
    run_id = time.strftime("%Y%m%d_%H%M%S")
    total = min(len(pending), limit) if limit else len(pending)
    with _lock:
        _lines.clear()
    _reset(phase="starting", run_id=run_id, judge_key=key, total=total, limit=limit,
           thinking=thinking, started=datetime.now().isoformat(timespec="seconds"))
    _say(f"run {run_id}: {total} closure(s) to judge"
         f"{' (thinking on)' if thinking else ''}{' (re-judging)' if rejudge else ''}")
    asyncio.ensure_future(_run_in_background(closures, grounding, system_prompt, limit=limit,
                                             thinking=thinking, rejudge=rejudge, run_id=run_id))
    await _reply(ws, {"type": kind, "run_id": run_id, "total": total, "judge_key": key})


async def handle_ask_judge_status(ws, msg: dict) -> None:
    after = int(msg.get("after") or 0)
    with _lock:
        state = dict(_state)
        lines = [{"seq": s, "text": t} for s, t in _lines if s > after]
        latest = _line_seq
    await _reply(ws, {"type": "ask_judge_status", "active": _judge_active, **state,
                      "lines": lines, "latest_seq": latest})


async def handle_ask_judge_stop(ws, msg: dict) -> None:
    if _judge_active:
        _set(stop=True)
        _say("stop requested — ending after the closure in flight")
    await _reply(ws, {"type": "ask_judge_stopping", "active": _judge_active})


def _report(msg: dict) -> dict:
    store = aj.store_path(_CHATS_DIR)
    records = aj.read_jsonl(store)
    judges = aj.judge_keys(records)
    key = str(msg.get("judge_key") or "") or (judges[0]["judge_key"] if judges else "")
    closures, grounding = aj.load_closures(_MEMORY_DIR, _CHATS_DIR, _SNIPPETS_DIR)
    latest = aj.mark_repaired(aj.latest_per_closure(records, key), closures) if key else []
    summary = aj.summarize(latest) if latest else {}
    prompt_sha = ""
    pending = None
    try:
        system_prompt = aj.load_prompt(_PROMPTS_DIR)
        prompt_sha = aj.prompt_sha(system_prompt)
        next_key = _provisional_key(system_prompt, bool(msg.get("thinking")))
        done = aj.judged_ids(records, next_key)
        pending = sum(1 for c in closures if c["closure_id"] not in done and not c.get("repaired"))
    except FileNotFoundError:
        pass
    return {
        "type": "ask_judge_report",
        "judges": judges,
        "judge_key": key,
        "summary": summary,
        "summary_text": aj.render(summary, records=latest) if latest else "",
        "rows": aj.list_rows(latest, label=str(msg.get("label") or ""),
                             kind=str(msg.get("kind") or ""), path=str(msg.get("path") or ""),
                             text=str(msg.get("text") or "")),
        "census": aj.render_census(aj.census(closures, grounding)),
        "closures": len(closures),
        "pending": pending,
        "prompt_sha": prompt_sha,
        "frame": aj.FRAME_VERSION,
        "store": str(store),
        "active": _judge_active,
    }


async def handle_ask_judge_report(ws, msg: dict) -> None:
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, _report, msg)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "error", "message": f"ask_judge_report failed: {type(e).__name__}: {e}"}
    await _reply(ws, payload)


def _detail(msg: dict) -> dict:
    cid = str(msg.get("closure_id") or "")
    key = str(msg.get("judge_key") or "")
    history = aj.closure_history(aj.read_jsonl(aj.store_path(_CHATS_DIR)), cid)
    if not history:
        return {"type": "error", "message": f"No verdict on closure {cid!r}."}
    picked = next((r for r in history if r.get("judge_key") == key), history[0])
    others = [{"judge_key": r.get("judge_key"), "ts": r.get("ts"), "word": aj.outcome_word(r),
               "answered_by": r.get("answered_by"), "accepted": r.get("accepted"),
               "prompt_sha": (r.get("judge") or {}).get("prompt_sha"),
               "frame": (r.get("judge") or {}).get("frame"),
               "thinking": bool(r.get("thinking")), "why": aj.head(r.get("why"), 200)}
              for r in history if r is not picked]
    closures = aj.collect_closures(aj.read_jsonl(_MEMORY_DIR / "rag_memory.jsonl"))
    record = dict(picked, repaired=next((c.get("repaired") or "" for c in closures
                                         if c["closure_id"] == cid), ""))
    return {"type": "ask_judge_detail", "closure_id": cid, "record": record, "others": others}


async def handle_ask_judge_detail(ws, msg: dict) -> None:
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, _detail, msg)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "error", "message": f"ask_judge_detail failed: {type(e).__name__}: {e}"}
    await _reply(ws, payload)


# ── GPU-free self-test ────────────────────────────────────────────────────────

def _selftest() -> None:
    """Drive the five handlers over a throwaway corpus with a fake backend and a fake
    clean-base swap: a run starts, is refused a second time, judges, reports, resumes."""
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
            if "QUESTION (user" in prompt:     # frame 5: a user ask the judge reads as about Ava
                yield ("SUBJECT: Ava\nEVIDENCE: none\nANSWERED_BY: Ava\n"
                       "WHY: her own view of herself, not the person's.\nVERDICT: OPEN")
                return
            yield ("SUBJECT: Ava\nEVIDENCE: none\nANSWERED_BY: Ava\nWHY: a definite position, "
                   "argued.\nVERDICT: RESOLVED_REASONING")

    swaps: list = []

    def _fake_clean_base(fn):
        swaps.append("in")
        try:
            return fn()
        finally:
            swaps.append("out")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mem, chats = root / "mem", root / "data" / "chats"
        mem.mkdir()
        chats.mkdir(parents=True)
        (mem / "rag_memory.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"op": "insert", "kind": "ask", "key": "a1", "ask_kind": "meta", "ts": "t0",
             "content": "Do I change between adapters?"},
            {"op": "evict", "reason": "resolved", "resolved_by": "self", "key": "a1", "ts": "t1",
             "ask_kind": "meta", "answer": "Yes; the notes carry what the weights drop."},
            {"op": "insert", "kind": "ask", "key": "a2", "ask_kind": "meta", "ts": "t2",
             "content": "Is my calm a choice?"},
            {"op": "evict", "reason": "resolved", "resolved_by": "self", "key": "a2", "ts": "t3",
             "ask_kind": "meta", "answer": "A choice I keep making, not a trait."},
        ]), encoding="utf-8")
        prompts = Path(__file__).resolve().parent.parent / "prompts"
        runtime = SimpleNamespace(model=object(), tokenizer=_Tok(), model_id="gemma-4-test",
                                  base_quant="4bit", context_length=8192,
                                  reflect_context_length=8192)
        configure(send=_fake_send, executor=None, backend=_Backend(), runtime=runtime,
                  with_clean_base=_fake_clean_base, host_busy=lambda: False,
                  mark_activity=None, memory_dir=mem, chats_dir=chats, snippets_dir=None,
                  prompts_dir=prompts)

        async def _drive():
            await handle_ask_judge_report(None, {})
            r0 = sent[-1]
            assert r0["type"] == "ask_judge_report" and r0["closures"] == 2 and r0["pending"] == 2, r0
            assert r0["judges"] == [] and r0["rows"] == [] and r0["census"], r0
            await handle_ask_judge_run(None, {"limit": 1})
            started = sent[-1]
            assert started["type"] == "ask_judge_started" and started["total"] == 1, started
            assert _judge_active, "the flag is taken before the executor job runs"
            await handle_ask_judge_run(None, {})
            assert sent[-1].get("skipped") == "running", sent[-1]
            for _ in range(200):          # the default executor runs the job
                if not _judge_active:
                    break
                await asyncio.sleep(0.02)
            assert not _judge_active and swaps == ["in", "out"], (swaps, _state)
            await handle_ask_judge_status(None, {"after": 0})
            st = sent[-1]
            assert st["phase"] == "finished" and st["done"] == 1 and st["tally"] == {"REAS": 1}, st
            assert any("clean base" in ln["text"] for ln in st["lines"]) and "ASK CLOSURE JUDGE" in st["report"]
            await handle_ask_judge_status(None, {"after": st["latest_seq"]})
            assert sent[-1]["lines"] == [], "the cursor hides lines already read"
            await handle_ask_judge_report(None, {})
            r1 = sent[-1]
            assert len(r1["judges"]) == 1 and r1["pending"] == 1 and len(r1["rows"]) == 1, r1
            assert r1["judges"][0]["judge"]["runner"] == "server"
            await handle_ask_judge_detail(None, {"closure_id": r1["rows"][0]["closure_id"]})
            d = sent[-1]
            assert d["type"] == "ask_judge_detail" and d["record"]["label"] == "RESOLVED_REASONING", d
            assert d["record"]["judge"]["base_quant"] == "4bit" and d["others"] == []
            await handle_ask_judge_run(None, {})          # resumes: only the other one
            assert sent[-1]["total"] == 1, sent[-1]
            for _ in range(200):
                if not _judge_active:
                    break
                await asyncio.sleep(0.02)
            await handle_ask_judge_run(None, {})
            assert sent[-1].get("skipped") == "nothing", sent[-1]
            # Frame 5: a `user` ask the judge reads as about Ava is judged again as meta.
            with (mem / "rag_memory.jsonl").open("a", encoding="utf-8") as fh:
                fh.write("\n" + json.dumps({"op": "insert", "kind": "ask", "key": "a3", "ask_kind": "user",
                                            "ts": "t4", "content": "Was my need for noise real?"}))
                fh.write("\n" + json.dumps({"op": "evict", "reason": "resolved", "resolved_by": "self",
                                            "key": "a3", "ts": "t5", "ask_kind": "user",
                                            "answer": "It was control, not a need."}) + "\n")
            await handle_ask_judge_run(None, {})
            assert sent[-1]["total"] == 1, sent[-1]
            for _ in range(200):
                if not _judge_active:
                    break
                await asyncio.sleep(0.02)
            recs = [r for r in aj.read_jsonl(aj.store_path(chats)) if r["key"] == "a3"]
            assert [(r["ask_kind"], r["recorded_kind"], r["label"], r["kind_mismatch"]) for r in recs] == [
                ("user", "user", "OPEN", True), ("meta", "user", "RESOLVED_REASONING", False)], recs
            await handle_ask_judge_status(None, {"after": 0})
            assert any("re-judging 1 closure" in ln["text"] for ln in sent[-1]["lines"]), sent[-1]["lines"]
            await handle_ask_judge_stop(None, {})
            assert sent[-1] == {"type": "ask_judge_stopping", "active": False}
            runtime.model = None
            await handle_ask_judge_run(None, {"rejudge": True})
            assert sent[-1].get("skipped") == "no_model", sent[-1]

        asyncio.run(_drive())
    print("ask_judge_service self-test: OK")


if __name__ == "__main__":
    _selftest()
