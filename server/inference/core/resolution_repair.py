"""Repair of cut resolutions — closures the targeted ask-resolution pass made on a half-written answer.

Until 59a9a1a the pass (``ReflectionRunner._run_ask_resolution_for_session``) ran thinking-on
on a 768-token budget with no truncation check. A pass that hit the cap after ``DECISION:
yes`` closed its ask on the fragment it had got to, and the fragment was distilled — a
``[persona]`` record for a ``meta`` ask, a ``[fact]`` filed as the person's own report for
a ``user`` one. The main box's journal and the ask judge put it at 38 of 122 answered
closures (34 persona, 3 fact records). The fix stops new ones; this repairs the old.

**Candidates** — answered-path closures (``resolved_by: user``) whose answer came from a
capped pass. The activity journal keeps every pass's output (``ask_resolution`` bodies,
with ``truncated``), and the pre-fix parser's reading of that output is exactly the
answer the closure recorded, so the two are matched by text (:func:`old_parse`). A
closure older than the journal's oldest segment has no pass on record; it is a
candidate when its answer ends mid-sentence (:func:`looks_cut`) — marked as such, the
weaker evidence. A closure the journal shows finished cleanly is never a candidate.

**Dry run** — each candidate's conversation re-read with the FIXED pass
(``ReflectionRunner._resolve_ask_on_thread``: the same prompt, sampling and model as the
runner now — the adapter, i.e. Ava's own reading, as the original was — 4096 tokens,
truncation refused). The outcome goes into a plan (``server/data/asks/repair_plan.json``)
as each candidate finishes, so a stopped dry run resumes. Nothing in Ava's memory changes.

**Apply** — a separate, explicit step, per item with a dry-run outcome:

* ``resolved`` with a complete answer that differs → the cut record is superseded (RAG
  recall + ledger: out of recall, the digest's active evidence and training, kept as
  evidence-of-change, reversible — the self-reconciliation move) and the complete
  answer is written through the resolver's own path (``write_answered_resolution`` +
  ledger registration), its closure marked ``repairs: <the cut closure's ts>``. Under the
  distill gate (``core/distill_gate.py``, on by default) that path closes the ask and
  leaves the distill to the gate: the complete answer becomes a ``[persona]`` / ``[fact]``
  only once the frozen judge accepts it, at the next Sleep run (action ``reclosed``);
* ``open`` (re-read, the conversation does not answer it) or ``refused`` (the fixed
  pass still could not finish) → the cut record is superseded and the ask reopened —
  unless a later, sound closure of the same ask exists, in which case it stays closed;
* ``unchanged`` (the old answer was complete after all), ``missing_chat``, ``error`` →
  nothing.

Every applied item is appended to ``server/data/asks/repair_log.jsonl`` with the keys it
wrote, so it can be undone by hand (a re-``register`` reactivates a superseded anchor).

While a dry run or an apply is in progress :data:`_repair_active` is set: chat is
refused, the idle jobs wait, the API / gossip answer busy — the dry run holds the GPU
executor for as long as the re-reads take.

GPU-free self-test: ``python -m core.resolution_repair``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from core import activity_log
from core import ask_judge as aj

PLAN_FILENAME = "repair_plan.json"
LOG_FILENAME = "repair_log.jsonl"
_APPLICABLE = ("resolved", "open", "refused")

# ── candidates ────────────────────────────────────────────────────────────────


def old_parse(text: str) -> tuple:
    """``(answered, answer)`` as the PRE-FIX ``_parse_ask_resolution`` read a pass's
    output — kept verbatim in behaviour because it is what the damaged closures
    recorded, including an ANSWER read out of a thought cut before it closed."""
    text = re.sub(r"<think>.*?</think>", "\n", text or "", flags=re.DOTALL | re.IGNORECASE)
    idx = text.lower().find("<think>")
    if idx != -1:
        text = text[:idx]
    m = re.search(r"^\s*DECISION\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    answered = bool(m and m.group(1).strip().lower().startswith(
        ("yes", "y", "true", "1", "resolv")))
    a = re.search(r"ANSWER\s*:\s*(.*)\Z", text, re.IGNORECASE | re.DOTALL)
    return answered, (a.group(1).strip() if a else "")


def journal_passes(journal_dir) -> list:
    """Every ``ask_resolution`` pass the activity journal still holds (all segments):
    ``{ts, truncated, max_new_tokens, answered, answer}``, answer as :func:`old_parse`
    read it."""
    out: list = []
    d = Path(journal_dir) if journal_dir else None
    if d is None or not d.is_dir():
        return out
    for f in sorted(d.glob("activity*.jsonl")):
        try:
            fh = f.open(encoding="utf-8")
        except Exception:
            continue
        with fh:
            for line in fh:
                if '"ask_resolution' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if (r.get("kind") != "result"
                        or not str(r.get("message") or "").startswith("ask_resolution")):
                    continue
                detail = r.get("detail") or {}
                answered, answer = old_parse(r.get("text") or "")
                out.append({"ts": str(r.get("ts") or ""),
                            "truncated": bool(detail.get("truncated")),
                            "max_new_tokens": detail.get("max_new_tokens"),
                            "answered": answered, "answer": answer})
    return out


# An answer that stops mid-sentence (or is empty) — the evidence for a closure older than
# the journal. Weaker than the journal's flag, and labelled as such. Shared with the ask
# judge's report, which counts the cuts still live.
looks_cut = aj.looks_cut


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def find_candidates(memory_dir, journal_dir) -> list:
    """The damaged closures, oldest first, as plan items (no dry-run result yet)."""
    mem = Path(memory_dir)
    log = aj.read_jsonl(mem / "rag_memory.jsonl")
    closures = aj.collect_closures(log)
    aj.mark_distilled(closures, aj.distilled_index(aj.read_jsonl(mem / "weights_persona.jsonl")))
    # The ask records (for a reopen) and the repairs already on record.
    latest_ask: dict = {}
    done: set = set()
    for r in log:
        key = r.get("key")
        if not key:
            continue
        if r.get("op") == "insert" and r.get("kind") == "ask":
            latest_ask[key] = r
            reopened = r.get("reopened") or {}
            if reopened.get("closed_ts"):
                done.add((key, str(reopened["closed_ts"])))
        elif r.get("op") == "evict" and r.get("repairs"):
            done.add((key, str(r["repairs"])))
    passes = journal_passes(journal_dir)
    by_answer: dict = {}
    for p in passes:
        if p["answered"]:
            by_answer.setdefault(_norm(p["answer"]), []).append(p)

    items: list = []
    for c in closures:
        if c["path"] != "answered":
            continue
        if (c["key"], c["closed_ts"]) in done:
            continue
        # The repair's own closures are sound by construction.
        seen = [p for p in by_answer.get(_norm(c["answer"]), [])]
        if seen:
            capped = [p for p in seen if p["truncated"]]
            if not capped:
                continue
            found = {"by": "journal", "ts": capped[0]["ts"],
                     "max_new_tokens": capped[0]["max_new_tokens"]}
        elif looks_cut(c["answer"]):
            found = {"by": "mid-sentence"}
        else:
            continue
        ask = latest_ask.get(c["key"]) or {}
        dist = c.get("distilled")
        items.append({
            "id": c["closure_id"], "key": c["key"], "closed_ts": c["closed_ts"],
            "closing_run": c["closing_run"], "question": c["question"],
            "ask_kind": c["ask_kind"], "chat": c["answered_in"] or c["source_session"],
            "old_answer": c["answer"],
            "old_record": ({"key": _content_key(c["answer"]), "kind": dist}
                           if dist and c["answer"].strip() else None),
            "found": found,
            "ask": {k: ask.get(k) for k in ("key", "content", "ask_kind", "source_session",
                                              "origin_ts", "lookup", "seen") if ask.get(k)},
            "result": None, "applied": None,
        })
    # A later, sound closure of the same ask keeps it closed on a failed re-read.
    ids = {it["id"] for it in items}
    for it in items:
        it["closed_later"] = any(
            c["key"] == it["key"] and c["closed_ts"] > it["closed_ts"] and c["closure_id"] not in ids
            for c in closures)
    return items


def _content_key(text: str) -> str:
    from core.reflection_writer import content_key
    return content_key(text)


# ── the plan ──────────────────────────────────────────────────────────────────

def plan_path(asks_dir) -> Path:
    return Path(asks_dir) / PLAN_FILENAME


def log_path(asks_dir) -> Path:
    return Path(asks_dir) / LOG_FILENAME


def load_plan(path) -> dict:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        return plan if isinstance(plan, dict) else {}
    except Exception:
        return {}


def save_plan(path, plan: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def merge_plan(candidates: list, previous: dict) -> dict:
    """Today's candidates, carrying over each one's dry-run result and apply record.
    An item applied earlier stays in the plan (and the log) even though it is no
    longer a candidate."""
    old = {it.get("id"): it for it in (previous.get("items") or [])}
    items: list = []
    seen: set = set()
    for c in candidates:
        prev = old.get(c["id"]) or {}
        items.append(dict(c, result=prev.get("result"), applied=prev.get("applied")))
        seen.add(c["id"])
    for pid, prev in old.items():
        if pid not in seen and prev.get("applied"):
            items.append(prev)
    items.sort(key=lambda it: str(it.get("closed_ts") or ""))
    return {"updated": datetime.now().isoformat(timespec="seconds"), "items": items}


def item_state(it: dict) -> str:
    if it.get("applied"):
        return "applied"
    res = it.get("result") or {}
    return str(res.get("outcome") or "pending")


def summarize(plan: dict) -> dict:
    items = plan.get("items") or []
    states = Counter(item_state(it) for it in items)
    applicable = [it for it in items if not it.get("applied")
                  and (it.get("result") or {}).get("outcome") in _APPLICABLE]
    return {
        "items": len(items), "states": dict(states),
        "pending": sum(1 for it in items if item_state(it) == "pending"),
        "applicable": len(applicable),
        "will_supersede": sum(1 for it in applicable if it.get("old_record")),
        "will_distill": sum(1 for it in applicable if it["result"]["outcome"] == "resolved"),
        "will_reopen": sum(1 for it in applicable if it["result"]["outcome"] != "resolved"
                           and not it.get("closed_later")),
        "found_by": dict(Counter((it.get("found") or {}).get("by", "?") for it in items)),
    }


# ── the dry run ───────────────────────────────────────────────────────────────

def reread(it: dict, *, generate_fn: Callable, chats_dir, adapter: str = "") -> dict:
    """Re-read one candidate's conversation with the fixed pass → its ``result``."""
    from core.reflection_runner import ReflectionRunner
    now = datetime.now().isoformat(timespec="seconds")
    chat = str(it.get("chat") or "")
    root = Path(chats_dir)
    try:
        path = (root / chat).resolve()
        ok = bool(chat) and path.parent == root.resolve() and path.exists()
        doc = json.loads(path.read_text(encoding="utf-8")) if ok else None
    except Exception:
        doc = None
    if not isinstance(doc, dict):
        return {"outcome": "missing_chat", "answer": "", "reason": f"{chat or '?'} is gone",
                "ts": now, "adapter": adapter}
    transcript = ReflectionRunner._render_thread_for_resolution(doc)
    who = str(doc.get("user") or "").strip() or "the person"
    t0 = time.monotonic()
    try:
        outcome, text = ReflectionRunner._resolve_ask_on_thread(
            generate_fn, it["question"], who, transcript, chat)
    except Exception as e:  # noqa: BLE001 — recorded, the next item still runs
        return {"outcome": "error", "answer": "", "reason": f"{type(e).__name__}: {e}",
                "ts": now, "adapter": adapter}
    res = {"ts": now, "adapter": adapter, "who": who,
           "seconds": round(time.monotonic() - t0, 1)}
    if outcome == "resolved":
        if _norm(text) == _norm(it.get("old_answer") or ""):
            res.update(outcome="unchanged", answer=text, reason="the recorded answer was complete")
        else:
            res.update(outcome="resolved", answer=text, reason="")
    elif outcome == "open":
        res.update(outcome="open", answer="", reason="re-read, the conversation does not answer it")
    else:
        res.update(outcome="refused", answer="", reason=text)
    return res


def dry_run(plan: dict, *, generate_fn: Callable, chats_dir, adapter: str = "",
            redo: bool = False, on_item: Optional[Callable] = None,
            should_stop: Optional[Callable[[], bool]] = None,
            save: Optional[Callable[[dict], None]] = None) -> Counter:
    """Re-read every candidate without a result (all of them with *redo*); save the plan
    after each, so a stop or a crash keeps what was re-read."""
    todo = [it for it in plan.get("items") or []
            if not it.get("applied") and (redo or not it.get("result"))]
    tally: Counter = Counter()
    for n, it in enumerate(todo, 1):
        if should_stop is not None and should_stop():
            tally["stopped"] = len(todo) - n + 1
            break
        it["result"] = reread(it, generate_fn=generate_fn, chats_dir=chats_dir, adapter=adapter)
        tally[it["result"]["outcome"]] += 1
        if save is not None:
            save(plan)
        if on_item is not None:
            on_item(n, len(todo), it)
    return tally


# ── apply ─────────────────────────────────────────────────────────────────────

def apply_plan(plan: dict, *, writer, ledger, consolidation_dir, run_id: str,
               log_file, ids: Optional[set] = None) -> Counter:
    """Write every applicable item (or those in *ids*): see the module docstring.
    Idempotent — an applied item is skipped — and logged with the keys it wrote."""
    from core.reflection_writer import register_consolidation_anchors
    tally: Counter = Counter()
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for it in plan.get("items") or []:
        if it.get("applied"):
            continue
        if ids and it.get("id") not in ids:
            continue
        res = it.get("result") or {}
        outcome = res.get("outcome")
        if outcome not in _APPLICABLE:
            continue
        wrote: dict = {}
        old = it.get("old_record") or {}
        if old.get("key"):
            writer.write_supersede([{"key": old["key"], "kind": old.get("kind") or "",
                                     "content": it.get("old_answer") or "",
                                     "reason": f"cut resolution repaired ({run_id})"}])
            ledger.supersede([old["key"]], reason=f"cut resolution repaired ({run_id})")
            wrote["superseded"] = old["key"]
        if outcome == "resolved":
            summary = writer.write_answered_resolution(
                key=it["key"], question=it.get("question") or "", answer=res["answer"],
                ask_kind=it.get("ask_kind") or "", source_session=it.get("chat") or "",
                answered_in=it.get("chat") or "", run_id=run_id,
                source_user=str(res.get("who") or ""),
                extra={"repairs": it.get("closed_ts")})
            register_consolidation_anchors(Path(consolidation_dir), summary, it.get("chat") or "")
            if summary.get("distill_pending"):
                wrote["distill_pending"] = True    # the distill gate decides at the next Sleep run
                action = "reclosed"
            else:
                wrote["distilled"] = _content_key(res["answer"]) if summary.get("weights") else None
                action = "redistilled"
        elif it.get("closed_later"):
            action = "superseded"      # a later sound closure keeps the ask closed
        else:
            ask = dict(it.get("ask") or {})
            ask.setdefault("key", it["key"])
            ask.setdefault("content", it.get("question") or "")
            ask.setdefault("ask_kind", it.get("ask_kind") or "")
            writer.write_reopened_ask(ask=ask, run_id=run_id, closed_ts=it.get("closed_ts") or "",
                                      reason=f"repair: {res.get('reason') or outcome}")
            wrote["reopened"] = it["key"]
            action = "reopened"
        it["applied"] = {"ts": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                         "action": action, "wrote": wrote}
        tally[action] += 1
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": run_id, "ts": it["applied"]["ts"], "id": it["id"],
                                 "key": it["key"], "closed_ts": it.get("closed_ts"),
                                 "action": action, "wrote": wrote, "outcome": outcome,
                                 "old_answer": it.get("old_answer"),
                                 "new_answer": res.get("answer") or ""},
                                ensure_ascii=False) + "\n")
    return tally


# ── the server side (configure()-wired) ───────────────────────────────────────

_repair_active = False

_send: Callable = None
_executor: Any = None
_runtime: Any = None
_get_rag: Callable = None
_make_generate: Callable = None
_get_writer: Callable = None
_host_busy: Callable = None
_mark_activity: Callable = None
_MEMORY_DIR: Optional[Path] = None
_CONSOLIDATION_DIR: Optional[Path] = None
_CHATS_DIR: Optional[Path] = None
_JOURNAL_DIR: Optional[Path] = None
_ASKS_DIR: Optional[Path] = None

_lock = threading.Lock()
_state: dict = {}
_lines: deque = deque(maxlen=400)
_line_seq = 0


def configure(*, send, executor, runtime, get_rag, make_sync_reflect_generate,
              get_reflection_writer, host_busy, mark_activity, memory_dir,
              consolidation_dir, chats_dir, journal_dir) -> None:
    global _send, _executor, _runtime, _get_rag, _make_generate, _get_writer
    global _host_busy, _mark_activity, _MEMORY_DIR, _CONSOLIDATION_DIR, _CHATS_DIR
    global _JOURNAL_DIR, _ASKS_DIR
    _send, _executor, _runtime = send, executor, runtime
    _get_rag, _make_generate, _get_writer = get_rag, make_sync_reflect_generate, get_reflection_writer
    _host_busy, _mark_activity = host_busy, mark_activity
    _MEMORY_DIR, _CONSOLIDATION_DIR = Path(memory_dir), Path(consolidation_dir)
    _CHATS_DIR, _JOURNAL_DIR = Path(chats_dir), Path(journal_dir)
    _ASKS_DIR = aj.store_path(_CHATS_DIR).parent
    _set_state(phase="idle")


def _set_state(**fields) -> None:
    with _lock:
        if fields.get("phase") in ("idle", "starting"):
            _state.clear()
            _state.update({"phase": "idle", "job": None, "run_id": None, "done": 0, "total": 0,
                           "tally": {}, "eta_s": None, "error": None, "stop": False,
                           "started": None, "finished": None})
        _state.update(fields)


def _say(text: str) -> None:
    global _line_seq
    with _lock:
        _line_seq += 1
        _lines.append((_line_seq, text))
    print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [repair] {text}",
          flush=True)


def _current_plan() -> dict:
    """Today's candidates merged with the saved plan (and saved back)."""
    path = plan_path(_ASKS_DIR)
    plan = merge_plan(find_candidates(_MEMORY_DIR, _JOURNAL_DIR), load_plan(path))
    save_plan(path, plan)
    return plan


def _blocking_dry_run(redo: bool, run_id: str) -> None:
    activity_log.set_current("repair", "re-reading cut resolutions", phase="dry_run")
    path = plan_path(_ASKS_DIR)
    try:
        plan = _current_plan()
        todo = [it for it in plan["items"] if not it.get("applied") and (redo or not it.get("result"))]
        _set_state(phase="rereading", total=len(todo))
        _say(f"re-reading {len(todo)} conversation(s) with the fixed resolver "
             f"({_runtime.model_id}, adapter {str(_runtime.adapter_id or 'none').split('/')[-1]})")
        activity_log.append("repair", "started", f"Repair dry run: {len(todo)} cut resolution(s) to re-read")
        generate = _make_generate(_get_rag())
        adapter = str(_runtime.adapter_id or "")
        t0 = time.monotonic()

        def _on_item(n: int, total: int, it: dict) -> None:
            res = it["result"]
            eta = (time.monotonic() - t0) / n * (total - n)
            _say(f"[{n:>3}/{total}] {res['outcome']:<12} {it['ask_kind']:<5} "
                 f"{aj.head(it['question'], 70)}  ({res.get('seconds', 0)} s, eta {eta / 60:.0f} min)")
            with _lock:
                tally = dict(_state.get("tally") or {})
                tally[res["outcome"]] = tally.get(res["outcome"], 0) + 1
                _state.update(done=n, eta_s=round(eta), tally=tally)
            activity_log.append("repair", "progress",
                                f"Repair {n}/{total}: {res['outcome']} — {aj.head(it['question'], 80)}")

        tally = dry_run(plan, generate_fn=generate, chats_dir=_CHATS_DIR, adapter=adapter,
                        redo=redo, on_item=_on_item, should_stop=lambda: bool(_state.get("stop")),
                        save=lambda p: save_plan(path, p))
        save_plan(path, plan)
        stopped = bool(tally.get("stopped"))
        _set_state(phase="stopped" if stopped else "finished",
                   finished=datetime.now().isoformat(timespec="seconds"))
        _say(("stopped" if stopped else "finished") + f": {dict(tally)} — review, then Apply")
        activity_log.append("repair", "finished", f"Repair dry run: {dict(tally)}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_state(phase="failed", error=f"{type(e).__name__}: {e}",
                   finished=datetime.now().isoformat(timespec="seconds"))
        _say(f"failed: {type(e).__name__}: {e}")
        activity_log.append("repair", "failed", f"Repair dry run failed: {type(e).__name__}: {e}")
    finally:
        activity_log.clear_current()


def _blocking_apply(ids: Optional[set], run_id: str) -> dict:
    from training.ledger import ConsolidationLedger
    path = plan_path(_ASKS_DIR)
    plan = _current_plan()
    tally = apply_plan(plan, writer=_get_writer(), ledger=ConsolidationLedger(_CONSOLIDATION_DIR),
                       consolidation_dir=_CONSOLIDATION_DIR, run_id=run_id,
                       log_file=log_path(_ASKS_DIR), ids=ids)
    save_plan(path, plan)
    try:
        _get_rag().refresh_reflection_memory()
    except Exception:
        traceback.print_exc()
    msg = f"Repair applied: {dict(tally) or 'nothing to apply'}"
    _say(msg)
    activity_log.append("repair", "finished", msg)
    return {"tally": dict(tally), "summary": summarize(plan)}


async def _background(fn, *args) -> None:
    global _repair_active
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: fn(*args))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_state(phase="failed", error=f"{type(e).__name__}: {e}")
    finally:
        _repair_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass


def _refusal(kind: str) -> Optional[dict]:
    if _repair_active:
        return {"type": kind, "skipped": "running", "message": "A repair job is already in progress."}
    if _host_busy is not None and _host_busy():
        return {"type": kind, "skipped": "busy",
                "message": "The GPU is busy with another job; try again when it is idle."}
    return None


async def handle_repair_plan(ws, msg: dict) -> None:
    loop = asyncio.get_running_loop()
    try:
        plan = await loop.run_in_executor(None, _current_plan)
        from core.reflection_writer import distill_gate_enabled
        payload = {"type": "repair_plan", "items": plan["items"], "summary": summarize(plan),
                   "active": _repair_active, "phase": _state.get("phase"),
                   "gated": distill_gate_enabled(),
                   "plan_file": str(plan_path(_ASKS_DIR))}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "error", "message": f"repair_plan failed: {type(e).__name__}: {e}"}
    await _send(ws, payload)


async def handle_repair_run(ws, msg: dict) -> None:
    global _repair_active
    refused = _refusal("repair_started")
    if refused:
        await _send(ws, refused)
        return
    if _runtime is None or _runtime.model is None:
        await _send(ws, {"type": "repair_started", "skipped": "no_model",
                         "message": "No model is loaded — load one from the Chat tab first."})
        return
    _repair_active = True          # on the event loop, before the executor job is queued
    run_id = f"repair-{time.strftime('%Y%m%d_%H%M%S')}"
    with _lock:
        _lines.clear()
    _set_state(phase="starting")
    _set_state(job="dry_run", run_id=run_id, started=datetime.now().isoformat(timespec="seconds"))
    _say(f"dry run {run_id}{' (redo)' if msg.get('redo') else ''}")
    asyncio.ensure_future(_background(_blocking_dry_run, bool(msg.get("redo")), run_id))
    await _send(ws, {"type": "repair_started", "run_id": run_id})


async def handle_repair_status(ws, msg: dict) -> None:
    after = int(msg.get("after") or 0)
    with _lock:
        state = dict(_state)
        lines = [{"seq": s, "text": t} for s, t in _lines if s > after]
        latest = _line_seq
    await _send(ws, {"type": "repair_status", "active": _repair_active, **state,
                     "lines": lines, "latest_seq": latest})


async def handle_repair_stop(ws, msg: dict) -> None:
    if _repair_active:
        _set_state(stop=True)
        _say("stop requested — ending after the conversation in flight")
    await _send(ws, {"type": "repair_stopping", "active": _repair_active})


async def handle_repair_apply(ws, msg: dict) -> None:
    global _repair_active
    refused = _refusal("repair_applied")
    if refused:
        await _send(ws, refused)
        return
    ids = set(msg.get("ids") or []) or None
    _repair_active = True
    run_id = f"repair-{time.strftime('%Y%m%d_%H%M%S')}"
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, lambda: _blocking_apply(ids, run_id))
        payload = {"type": "repair_applied", "run_id": run_id, **result}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "repair_applied", "skipped": "error",
                   "message": f"apply failed: {type(e).__name__}: {e}"}
    finally:
        _repair_active = False
        if _mark_activity is not None:
            try:
                _mark_activity()
            except Exception:
                pass
    await _send(ws, payload)


# ── GPU-free self-test ────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    from types import SimpleNamespace

    from core.reflection_memory import ReflectionMemory
    from core.reflection_writer import ReflectionWriter, content_key
    from training.ledger import ConsolidationLedger

    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok ' if good else 'FAIL'}  {label}: {got!r}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mem, cons = root / "inference" / "data" / "hot" / "memory", root / "inference" / "data" / "hot" / "consolidation"
        chats, journal = root / "data" / "chats", root / "inference" / "data" / "hot" / "activity"
        for d in (mem, cons, chats, journal):
            d.mkdir(parents=True)
        writer = ReflectionWriter(mem, cons)
        ledger = ConsolidationLedger(cons)

        def ask(key, content, kind):
            writer._append(writer.RAG_FILE, {"run_id": "r", "ts": "2026-08-01T00:00:00", "op": "insert",
                                             "kind": "ask", "ask_kind": kind, "content": content,
                                             "key": key, "source_session": "20260801_000000.json"})

        from core import reflection_writer as rw

        def close(key, question, answer, kind, chat):
            # The cut closures are history: written before the distill gate existed.
            rw.set_distill_gate(False)
            try:
                summary = writer.write_answered_resolution(key=key, question=question, answer=answer,
                                                           ask_kind=kind, source_session=chat,
                                                           answered_in=chat, run_id="bg_1",
                                                           source_user="Dana")
            finally:
                rw.set_distill_gate(True)
            register_anchors(summary, chat)

        from core.reflection_writer import register_consolidation_anchors

        def register_anchors(summary, chat):
            register_consolidation_anchors(cons, summary, chat)

        for i, (key, q, kind) in enumerate([("m1", "Is my calm a choice?", "meta"),
                                             ("m2", "Do I change between adapters?", "meta"),
                                             ("u1", "Where did Dana grow up?", "user"),
                                             ("m3", "Is the notes file me?", "meta"),
                                             ("m4", "Does silence scare me?", "meta")]):
            ask(key, q, kind)
        chat_doc = {"user": "Dana", "exchanges": [
            {"user_prompt": "Is your calm a choice?", "assistant_response": "A choice I keep making."},
            {"user_prompt": "I grew up in Halden.", "assistant_response": "Noted."}]}
        (chats / "20260810_120000.json").write_text(json.dumps(chat_doc), encoding="utf-8")
        close("m1", "Is my calm a choice?", "A choice I keep", "meta", "20260810_120000.json")   # cut, journal
        close("m2", "Do I change between adapters?", "Yes, the notes carry it.", "meta",
              "20260810_120000.json")                                                           # clean, journal
        close("u1", "Where did Dana grow up?", "Dana grew up in", "user", "20260810_120000.json")  # cut, no journal
        close("m3", "Is the notes file me?", "Partly: it is the memory I", "meta", "gone.json")    # cut, chat gone
        close("m4", "Does silence scare me?", "Silence is", "meta", "20260810_120000.json")        # cut, journal

        def body(answer, truncated):
            return json.dumps({"seq": 1, "ts": "2026-08-12T00:00:00+00:00", "source": "background_reflection",
                               "kind": "result", "level": "body",
                               "message": "ask_resolution: produced 100 chars in 5s",
                               "text": f"<think>thinking</think>DECISION: yes\nANSWER: {answer}",
                               "detail": {"truncated": truncated, "max_new_tokens": 768}})
        (journal / "activity.100.jsonl").write_text("\n".join([
            body("A choice I keep", True), body("Yes, the notes carry it.", False),
            body("Partly: it is the memory I", True), body("Silence is", True)]) + "\n", encoding="utf-8")

        print("candidates: the journal's capped passes, the mid-sentence rest, never a clean one")
        cands = find_candidates(mem, journal)
        check("four candidates (m2 finished cleanly)", sorted(c["key"] for c in cands), ["m1", "m3", "m4", "u1"])
        by = {c["key"]: c for c in cands}
        check("found by", {k: by[k]["found"]["by"] for k in by},
              {"m1": "journal", "m3": "journal", "m4": "journal", "u1": "mid-sentence"})
        check("the cut records to supersede", {k: (by[k]["old_record"] or {}).get("kind") for k in by},
              {"m1": "persona", "m3": "persona", "m4": "persona", "u1": "fact"})
        check("old record key is the distilled content's key", by["m1"]["old_record"]["key"],
              content_key("A choice I keep"))
        check("old_parse reads a cut thought's rehearsed answer",
              old_parse("the form is\nDECISION: yes\nANSWER: identity is layered, but"),
              (True, "identity is layered, but"))

        print("dry run: the fixed pass on each conversation; saved as it goes; resumable")

        class _Gen:
            last_truncated = False
            last_loop = False

            def __call__(self, content, system, **kw):
                if "Is my calm a choice?" in content:
                    self.last_truncated = False
                    return "<think>they said so</think>DECISION: yes\nANSWER: My calm is a choice I keep making."
                if "Where did Dana grow up?" in content:
                    self.last_truncated = False
                    return "<think>stated</think>DECISION: yes\nANSWER: Dana grew up in Halden."
                if "Does silence scare me?" in content:
                    self.last_truncated = False
                    return "<think>nothing about silence</think>DECISION: no\nANSWER:"
                self.last_truncated = True
                return "<think>long"
        plan = merge_plan(cands, {})
        saves: list = []
        tally = dry_run(plan, generate_fn=_Gen(), chats_dir=chats, adapter="adapter-x",
                        save=lambda p: saves.append(1), should_stop=lambda: len(saves) >= 3)
        check("stopped after three, saved after each", (dict(tally).get("stopped"), len(saves)), (1, 3))
        tally2 = dry_run(plan, generate_fn=_Gen(), chats_dir=chats)
        check("resumes with the rest only", sum(v for k, v in tally2.items() if k != "stopped"), 1)
        states = {it["key"]: item_state(it) for it in plan["items"]}
        check("outcomes", states, {"m1": "resolved", "u1": "resolved", "m3": "missing_chat", "m4": "open"})
        s = summarize(plan)
        check("summary: applicable / supersede / distill / reopen",
              (s["applicable"], s["will_supersede"], s["will_distill"], s["will_reopen"]), (3, 3, 2, 1))

        print("apply: supersede the cut record, close again on the complete answer or reopen; "
              "the distill gate decides the rest; idempotent")
        log_file = root / "data" / "asks" / LOG_FILENAME
        tally = apply_plan(plan, writer=writer, ledger=ledger, consolidation_dir=cons,
                           run_id="repair-test", log_file=log_file)
        check("actions", dict(tally), {"reclosed": 2, "reopened": 1})
        live = {r.get("key"): r for r in ReflectionMemory(mem).live_items()}
        check("cut persona gone from recall", content_key("A choice I keep") in live, False)
        check("cut fact gone", content_key("Dana grew up in") in live, False)
        check("the complete answers are not distilled yet",
              (content_key("My calm is a choice I keep making.") in live,
               content_key("Dana grew up in Halden.") in live), (False, False))
        check("the open one is a live ask again", "m4" in live and live["m4"].get("kind") == "ask", True)
        check("the missing-chat one is untouched", content_key("Partly: it is the memory I") in live, True)
        fold = ledger.fold()
        check("ledger: cut anchor superseded", fold[content_key("A choice I keep")].get("superseded"), True)
        rag = aj.read_jsonl(mem / "rag_memory.jsonl")
        check("the repair's closures wait for the gate",
              [r.get("distill") for r in rag if r.get("op") == "evict" and r.get("run_id") == "repair-test"],
              ["pending", "pending"])
        from core import distill_gate as dg

        def _accepting(c, kind=None):
            meta = c["ask_kind"] == "meta"
            return {"closure_id": c["closure_id"], "judge_key": "J", "ts": "2026-09-26T12:00:00",
                    "outcome": "verdict", "ask_kind": c["ask_kind"], "subject": "ava" if meta else "person",
                    "label": "RESOLVED_REASONING" if meta else "RESOLVED_EVIDENCE",
                    "answered_by": "ava" if meta else "other", "quote_found": None if meta else True}

        gated = dg.run_gate(memory_dirs=[mem], verdict_records=[], judge_key="J", judge=_accepting,
                            writer=writer, consolidation_dir=cons, run_id="gate-test", say=lambda s: None)
        check("the gate distills both once the judge accepts them",
              {k: v for k, v in gated.items() if k.startswith("distill")},
              {"distill:persona": 1, "distill:fact": 1})
        live = {r.get("key"): r for r in ReflectionMemory(mem).live_items()}
        check("complete persona and fact recalled",
              (content_key("My calm is a choice I keep making.") in live,
               content_key("Dana grew up in Halden.") in live), (True, True))
        check("ledger: the complete persona live",
              bool(ledger.fold().get(content_key("My calm is a choice I keep making."))), True)
        check("the repair's closure says what it repairs",
              [r.get("repairs") for r in rag if r.get("op") == "evict" and r.get("run_id") == "repair-test"],
              [it["closed_ts"] for it in plan["items"] if it["key"] in ("m1", "u1")])
        check("logged", len(aj.read_jsonl(log_file)), 3)
        again = apply_plan(plan, writer=writer, ledger=ledger, consolidation_dir=cons,
                           run_id="repair-again", log_file=log_file)
        check("a second apply writes nothing", dict(again), {})
        check("repaired closures are no longer candidates",
              sorted(c["key"] for c in find_candidates(mem, journal)), ["m3"])
        merged = merge_plan(find_candidates(mem, journal), plan)
        check("applied items stay in the plan", sum(1 for it in merged["items"] if it.get("applied")), 3)

        print("handlers: a run refused while one is active; plan / status / apply over the store")
        save_plan(plan_path(root / "data" / "asks"), plan)     # as the server keeps it
        sent: list = []

        async def _fake_send(ws, payload):
            sent.append(payload)

        runtime = SimpleNamespace(model=object(), model_id="m", adapter_id="adapter-x")
        configure(send=_fake_send, executor=None, runtime=runtime,
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: None),
                  make_sync_reflect_generate=lambda rag: _Gen(), get_reflection_writer=lambda: writer,
                  host_busy=lambda: False, mark_activity=None, memory_dir=mem,
                  consolidation_dir=cons, chats_dir=chats, journal_dir=journal)

        async def _drive():
            await handle_repair_plan(None, {})
            p = sent[-1]
            # the one left (m3, its chat gone) plus the three applied, carried over
            assert p["type"] == "repair_plan" and p["summary"]["items"] == 4, p["summary"]
            assert p["summary"]["states"] == {"missing_chat": 1, "applied": 3}, p["summary"]
            await handle_repair_run(None, {"redo": True})
            assert sent[-1]["type"] == "repair_started" and _repair_active
            await handle_repair_run(None, {})
            assert sent[-1].get("skipped") == "running"
            for _ in range(300):
                if not _repair_active:
                    break
                await asyncio.sleep(0.02)
            await handle_repair_status(None, {})
            st = sent[-1]
            assert st["phase"] == "finished" and st["total"] == 1, st      # only m3 was not applied
            await handle_repair_apply(None, {})
            assert sent[-1]["type"] == "repair_applied" and sent[-1]["tally"] == {}, sent[-1]
        asyncio.run(_drive())
        check("handlers", True, True)

    if not ok:
        raise SystemExit("resolution_repair self-test: FAILED")
    print("resolution_repair self-test: OK")


if __name__ == "__main__":
    _selftest()
