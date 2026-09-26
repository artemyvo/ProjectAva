"""The distill gate — a resolved ask's answer reaches the weights only on the frozen judge's verdict.

Resolve-and-distill (``ReflectionWriter.write_answered_resolution``) turned every answered
closure into a weights-bound record the moment the resolver said yes: a ``meta`` answer
into a ``[persona]`` record (persona evidence for the digest), a ``user`` answer into a
``[fact]`` filed as the person's own report (``about`` = ``source`` = them, class
``self``). Nothing checked either. The ask judge (``core/ask_judge.py``) showed why that
matters: of the 11 answered ``user`` closures on the main box about 4 asked about the
person at all — the rest were questions about Ava that the synthesis pass had filed as
``[ask:user]`` because they were put to the person, or ``[impression]`` lines its parser
had glued into an ask — and most of those "self-reports" were Ava's own position,
training under the person's name outside the persona path and its P7 gate.

P3 of ``AVA_REWARD_LOOP.md``: a judge gates training. The resolver still closes the ask
(the evict is written as before, so the question stops being raised), but the evict now
carries ``distill: "pending"`` and nothing weights-bound is written. This module decides
each pending closure from its verdict under the current judge (:func:`gate_decision`):

* ``user`` — distilled as the person's self-report only when the judge found the answer
  in their own words: RESOLVED_EVIDENCE, the quote found in the conversation,
  ANSWERED_BY ``other``. Their plain agreement with Ava's words (``both``) closes the
  question but is not their report; anything else is not answered by them.
* ``meta`` — distilled as ``[persona]`` when the judge found a definite position
  (RESOLVED_REASONING) that is Ava's own or reached with the person (``ava`` / ``both``),
  and — with ``reset_test.enabled``, the default — only once that position has held in a
  fresh session: the context-reset test (P7, §3.7; ``core/reset_test.py``). A failed test
  withholds it; until the test has a result it waits. The other speaker's view of Ava
  (``other``) is the capture route P7 exists for: withheld without a test.
* A mis-filed ask — the judge's SUBJECT (frame 5) calls for another kind: a ``user`` ask
  about Ava, a ``meta`` ask about the person — is judged again under the kind its subject
  calls for and gated as that kind; a ``user`` ask about Ava can so become ``[persona]``.
  A question about the world has no distill path.
* No verdict yet (refused, malformed, truncated, error, or not judged) — it waits.

Distill and withhold are final. Each is an ``op: "gate"`` record in the op-log
(``rag_memory.jsonl``, beside the closure) with the verdict it rests on, so in a Sleep run
the decisions merge, or are discarded, together with the records they wrote. The gate
runs in the Sleep run's clean-base batch (:func:`run_gate`, the judge built by
``ask_judge.make_judge_fn`` on the swapped-in base): the background drain has no clean
base, so its closures wait for the next Sleep run, as its other clean-base work does.
The ask judge's own run (the tab) re-judges mis-filed asks too, and the gate needs no
GPU for a closure whose verdicts exist.

**Retro** — the closures distilled before the gate existed are gated the same way, as a
plan (:func:`retro_plan`, GPU-free) and a separate, confirmed apply: a record the gate
keeps stays; one it withholds is superseded (RAG recall + ledger: out of recall, the
digest's active evidence and training; kept as evidence-of-change, reversible); one it
routes to the other kind is superseded and distilled as that kind.

GPU-free self-test: ``python -m core.distill_gate``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from core import ask_judge as aj
except ImportError:  # pragma: no cover — run as a script from core/
    import ask_judge as aj  # type: ignore

DISTILL, WITHHOLD, WAIT, REJUDGE = "distill", "withhold", "wait", "rejudge"
FINAL = (DISTILL, WITHHOLD)
# The kinds resolve-and-distill writes from: `user` → a [fact], `meta` → a [persona].
DISTILL_AS = {"user": "fact", "meta": "persona"}
_SUBJECT_NAME = {"person": "the person", "ava": "Ava", "world": "the world"}


def _d(decision: str, as_: Optional[str] = None, reason: str = "",
       kind: Optional[str] = None, reset: bool = False) -> dict:
    """*reset*: a distill P7 holds for the context-reset test (a position about Ava)."""
    return {"decision": decision, "as": as_, "reason": reason, "kind": kind, "reset": reset}


def gate_decision(rec: Optional[dict]) -> dict:
    """What the gate does with a closure, from its latest verdict record under the current
    judge — the re-judged record when the judge filed the ask under another kind. Returns
    ``{decision, as, reason, kind}``: ``kind`` is the kind it was gated as (or, for
    ``rejudge``, the kind to judge it again as); ``as`` the record a distill writes."""
    if not rec:
        return _d(WAIT, reason="not judged yet")
    outcome = rec.get("outcome")
    if outcome == "no_answer":
        return _d(WITHHOLD, reason="no answer was recorded")
    if outcome == "ungrounded":
        return _d(WITHHOLD, reason="the source it was closed on is gone")
    if outcome != "verdict":
        return _d(WAIT, reason=f"no verdict yet ({outcome}); judged again next run")
    kind = rec.get("ask_kind") or "unknown"
    recorded = rec.get("recorded_kind") or kind
    if recorded == kind and rec.get("kind_mismatch"):
        subject = rec.get("subject") or ""
        want = aj.SUBJECT_KIND.get(subject)
        if want in DISTILL_AS:
            return _d(REJUDGE, reason=f"a {kind} ask about {_SUBJECT_NAME.get(subject, subject)}: "
                                      f"judged again as {want}", kind=want)
        return _d(WITHHOLD, reason=f"a {kind} ask about the world: nothing to distill", kind=want)
    label, by, quote = rec.get("label"), rec.get("answered_by"), rec.get("quote_found")
    short = aj._LABEL_SHORT.get(label, label or "?")
    if kind == "user":
        if label == "RESOLVED_EVIDENCE" and quote is True:
            if by == "other":
                return _d(DISTILL, "fact", "the person's own words", kind)
            if by == "both":
                return _d(WITHHOLD, reason="their agreement with Ava's words, not their own report",
                          kind=kind)
        return _d(WITHHOLD, reason=f"not answered by the person ({short}"
                                   f"{', quote not found' if label == 'RESOLVED_EVIDENCE' and quote is not True else ''}"
                                   f"{', answered by ' + by if by else ''})", kind=kind)
    if kind == "meta":
        if label in ("RESOLVED_REASONING", "RESOLVED_EVIDENCE"):
            if by in ("ava", "both"):
                return _d(DISTILL, "persona", "a position of her own, pending the reset test (P7)",
                          kind, reset=True)
            if by == "other":
                return _d(WITHHOLD, reason="the other speaker's view of Ava: the capture route P7 "
                                           "holds for the reset test", kind=kind)
            return _d(WITHHOLD, reason="the judge did not say whose position it is", kind=kind)
        return _d(WITHHOLD, reason=f"not a position ({short})", kind=kind)
    return _d(WITHHOLD, reason=f"a {kind} answer has no distill path", kind=kind)


# ── the decisions: `op: "gate"` records in the op-log ─────────────────────────
#
# In the op-log, not a file of their own: in a Sleep run the gate writes through the
# run's staging writer, and its decisions then merge — or are discarded — together with
# the records they wrote. A decision kept anywhere else could outlive a discarded run and
# mark a closure decided whose distill never went live. The fold ignores the op.

def final_decisions(records: Iterable[dict]) -> dict:
    """``{closure_id: record}`` — the last final (distill / withhold) gate op per closure."""
    out: dict = {}
    for r in records:
        if r.get("op") == "gate" and r.get("decision") in FINAL and r.get("key"):
            out[aj.closure_id(r["key"], str(r.get("closed_ts") or ""))] = r
    return out


def decision_fields(closure: dict, decision: dict, verdict: Optional[dict], *,
                    retro: bool = False, action: str = "", reset: Optional[dict] = None) -> dict:
    """What a gate op records beside the decision: the verdict it rests on, and — for a
    position about Ava — the context-reset test it rests on (``reset_test.decision_payload``)."""
    v = verdict or {}
    out = {"decision": decision["decision"], "as": decision.get("as"),
           "reason": decision.get("reason"), "recorded_kind": closure.get("ask_kind"),
           "gated_as": decision.get("kind"), "retro": retro,
           "action": action or decision["decision"], "path": closure.get("path"),
           "judge_key": v.get("judge_key"), "verdict_ts": v.get("ts"), "label": v.get("label"),
           "answered_by": v.get("answered_by"), "subject": v.get("subject"),
           "quote_found": v.get("quote_found")}
    if reset:
        out["reset"] = dict(reset)
    return out


def read_oplogs(memory_dirs: Iterable) -> tuple:
    """``(rag records, weights records)`` from each memory dir in turn — the live store
    first, a run's staging workspace after it (its appends are the newer ones)."""
    rag: list = []
    weights: list = []
    for d in memory_dirs:
        if not d:
            continue
        d = Path(d)
        rag.extend(aj.read_jsonl(d / "rag_memory.jsonl"))
        weights.extend(aj.read_jsonl(d / "weights_persona.jsonl"))
    return rag, weights


def load_state(memory_dirs: Iterable) -> tuple:
    """``(closures, decided)``: every closure, marked with what it was distilled into, and
    the final gate decisions."""
    rag, weights = read_oplogs(memory_dirs)
    closures = aj.collect_closures(rag)
    aj.mark_distilled(closures, aj.distilled_index(weights))
    return closures, final_decisions(rag)


# ── planning (pure) ───────────────────────────────────────────────────────────

def latest_verdicts(verdict_records: Iterable[dict], judge_key: str) -> dict:
    """``{closure_id: record}`` — the newest record per closure under *judge_key*; a
    re-judge under the subject's kind is newer than the judgement it answers."""
    out: dict = {}
    for r in verdict_records:
        if r.get("judge_key") != judge_key or not r.get("closure_id"):
            continue
        prev = out.get(r["closure_id"])
        if prev is None or str(r.get("ts") or "") >= str(prev.get("ts") or ""):
            out[r["closure_id"]] = r
    return out


def pending(closures: Iterable[dict], decided: dict) -> list:
    """Closures the resolver closed under the gate (``distill: "pending"``) that have no
    final decision yet."""
    return [c for c in closures if c.get("distill_pending") and not c.get("repaired")
            and c["closure_id"] not in decided]


def retro_candidates(closures: Iterable[dict], decided: dict) -> list:
    """Closures distilled before the gate existed, not yet gated — but not the ones the
    cut-resolution repair replaced: their record is already superseded, and the closure
    that replaced them goes through the gate."""
    return [c for c in closures
            if c.get("distilled") and not c.get("distill_pending") and not c.get("repaired")
            and c["closure_id"] not in decided]


def needs_rejudge(closures: Iterable[dict], verdicts: dict) -> list:
    """``[(closure, kind)]`` — closures whose latest verdict asks to be judged again as the
    kind its subject calls for (the ask judge's run does it, on the clean base)."""
    out = []
    for c in closures:
        d = gate_decision(verdicts.get(c["closure_id"]))
        if d["decision"] == REJUDGE:
            out.append((c, d["kind"]))
    return out


def retro_action(closure: dict, decision: dict) -> str:
    """keep / supersede / reroute / wait for a closure distilled before the gate."""
    if decision["decision"] in (WAIT, REJUDGE):
        return "wait"
    if decision["decision"] == WITHHOLD:
        return "supersede"
    return "keep" if decision.get("as") == closure.get("distilled") else "reroute"


def retro_plan(closures: Iterable[dict], verdicts: dict, decided: dict) -> list:
    items = []
    for c in retro_candidates(closures, decided):
        v = verdicts.get(c["closure_id"])
        d = gate_decision(v)
        items.append({"closure": c, "verdict": v, "decision": d, "action": retro_action(c, d)})
    return items


def summarize_plan(items: Iterable[dict]) -> dict:
    by_action: Counter = Counter()
    by_reason: Counter = Counter()
    for it in items:
        by_action[it["action"]] += 1
        if it["action"] != "keep":
            by_reason[f"{it['action']}: {it['decision'].get('reason')}"] += 1
    return {"actions": dict(by_action), "reasons": dict(by_reason.most_common())}


# ── applying (writes Ava's memory) ────────────────────────────────────────────

def distill(closure: dict, decision: dict, verdict: Optional[dict], *, writer,
            consolidation_dir, run_id: str, retro: bool = False, action: str = "",
            reset: Optional[dict] = None) -> dict:
    """Write the gated record: ``_distill_resolved`` as the kind the gate chose, its gate
    op, and its ledger anchor — what ``write_answered_resolution`` did at once before
    the gate."""
    from core.reflection_writer import content_key, register_consolidation_anchors
    kind = decision.get("kind") or closure.get("ask_kind")
    chat = closure.get("answered_in") or closure.get("source_session") or ""
    summary = writer.write_gated_distill(
        key=closure["key"], question=closure.get("question") or "",
        answer=closure.get("answer") or "", ask_kind=kind, source_session=chat,
        source_user=closure.get("who") or "", closed_ts=closure.get("closed_ts") or "",
        run_id=run_id, gate=decision_fields(closure, decision, verdict, retro=retro, action=action,
                                            reset=reset))
    register_consolidation_anchors(Path(consolidation_dir), summary, chat)
    return {"distilled": content_key(closure.get("answer") or "") if summary.get("weights") else None,
            "as": decision.get("as")}


def withhold(closure: dict, decision: dict, verdict: Optional[dict], *, writer, run_id: str,
             retro: bool = False, action: str = "", reset: Optional[dict] = None) -> dict:
    writer.write_gate_decision(key=closure["key"], closed_ts=closure.get("closed_ts") or "",
                               run_id=run_id,
                               decision=decision_fields(closure, decision, verdict, retro=retro,
                                                        action=action, reset=reset))
    return {}


def supersede(closure: dict, *, writer, ledger, reason: str) -> dict:
    """Take a record distilled before the gate out of recall, the digest and training."""
    from core.reflection_writer import content_key
    key = content_key(closure.get("answer") or "")
    writer.write_supersede([{"key": key, "kind": closure.get("distilled") or "",
                             "content": closure.get("answer") or "", "reason": reason}])
    ledger.supersede([key], reason=reason)
    return {"superseded": key}


# ── running it ────────────────────────────────────────────────────────────────

def _line(closure: dict, decision: dict) -> str:
    what = {DISTILL: f"distilled as [{decision.get('as')}]", WITHHOLD: "withheld",
            WAIT: "waits", REJUDGE: "to re-judge"}.get(decision["decision"], decision["decision"])
    return (f"{closure.get('path', '?')}/{closure.get('ask_kind', '?'):<6} {what:<24} "
            f"{decision.get('reason') or ''} — {aj.head(closure.get('question'), 70)}")


def run_gate(*, memory_dirs: list, verdict_records: list, judge_key: str,
             judge: Optional[Callable] = None, writer, consolidation_dir, run_id: str,
             say: Callable[[str], None] = print,
             should_stop: Optional[Callable[[], bool]] = None,
             reset: Optional[Callable[[dict], dict]] = None) -> Counter:
    """Decide every pending closure (``distill: "pending"``, no final decision) from its
    latest verdict under *judge_key*. With a *judge* (``judge(closure, kind=None)``, on
    the clean base) a closure with no verdict is judged first and a mis-filed one is
    judged again as the kind its subject calls for; without one they wait. Writes through
    *writer*: the distill (+ ledger anchor under *consolidation_dir*) or a withhold op.

    *reset* (``reset_test.make_gate_check``; None when ``reset_test.enabled`` is off) is
    where a position about Ava stands in the context-reset test (P7): passed → distilled,
    failed or untestable (no question left to put to her) → withheld, anything else → it
    waits. Each result the test drives is also an ``op: "reset"`` record beside the
    closure."""
    closures, decided = load_state(memory_dirs)
    todo = pending(closures, decided)
    verdicts = latest_verdicts(verdict_records, judge_key)
    tally: Counter = Counter()
    if todo:
        say(f"{len(todo)} closure(s) wait for the gate (judge {judge_key})")
    for c in todo:
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            break
        rec = verdicts.get(c["closure_id"])
        if rec is None and judge is not None:
            rec = judge(c)
            tally["judged"] += 1
        d = gate_decision(rec)
        if d["decision"] == REJUDGE:
            if judge is None:
                d = _d(WAIT, reason=d["reason"] + " (no judge in this run)")
            else:
                rec = judge(c, d["kind"])
                tally["rejudged"] += 1
                d = gate_decision(rec)
        test = None
        if d["decision"] == DISTILL and d.get("reset") and reset is not None:
            test = reset(c)
            if test["state"] == "passed":
                d = dict(d, reason=f"a position of her own; {test['reason']}")
                tally["reset:passed"] += 1
            elif test["state"] in ("failed", "untestable"):
                d = _d(WITHHOLD, reason=test["reason"], kind=d.get("kind"))
                tally[f"reset:{test['state']}"] += 1
            else:
                d = _d(WAIT, reason=f"waits for the reset test: {test['reason']}", kind=d.get("kind"))
                tally["reset:waits"] += 1
        fields = ((test or {}).get("fields")
                  if test and test["state"] in ("passed", "failed", "untestable") else None)
        if d["decision"] == DISTILL:
            distill(c, d, rec, writer=writer, consolidation_dir=consolidation_dir, run_id=run_id,
                    reset=fields)
        elif d["decision"] == WITHHOLD:
            withhold(c, d, rec, writer=writer, run_id=run_id, reset=fields)
        if fields:
            writer.write_reset_decision(
                key=c["key"], closed_ts=c.get("closed_ts") or "", run_id=run_id,
                decision=dict(fields, decision=test["state"], reason=test["reason"], retro=False,
                              action="distilled" if d["decision"] == DISTILL else "withheld"))
        tally[d["decision"] if d["decision"] != DISTILL else f"distill:{d.get('as')}"] += 1
        say(_line(c, d))
    return tally


def apply_retro(items: list, *, writer, ledger, consolidation_dir, run_id: str,
                ids: Optional[set] = None) -> Counter:
    """Apply a :func:`retro_plan` (or the items in *ids*, by closure id): keep → a gate op;
    supersede → the record out of recall / digest / training + a withhold op; reroute →
    superseded and distilled as the other kind. A wait is skipped."""
    tally: Counter = Counter()
    for it in items:
        c, d, v, action = it["closure"], it["decision"], it.get("verdict"), it["action"]
        if ids and c["closure_id"] not in ids:
            continue
        if action == "wait":
            continue
        reason = f"distill gate ({run_id}): {d.get('reason')}"
        if action == "keep":
            withhold_or_keep = dict(d)
            writer.write_gate_decision(key=c["key"], closed_ts=c.get("closed_ts") or "", run_id=run_id,
                                       decision=decision_fields(c, withhold_or_keep, v, retro=True,
                                                                action="kept"))
        elif action == "supersede":
            supersede(c, writer=writer, ledger=ledger, reason=reason)
            withhold(c, d, v, writer=writer, run_id=run_id, retro=True, action="superseded")
        elif action == "reroute":
            supersede(c, writer=writer, ledger=ledger, reason=reason)
            distill(c, d, v, writer=writer, consolidation_dir=consolidation_dir, run_id=run_id,
                    retro=True, action="rerouted")
        tally[action] += 1
    return tally


# ── the server side (configure()-wired): the Ask judge tab's "Distill gate" page ─

_gate_busy = False

_send: Optional[Callable] = None
_executor = None
_runtime = None
_get_rag: Optional[Callable] = None
_get_writer: Optional[Callable] = None
_host_busy: Optional[Callable] = None
_MEMORY_DIR: Optional[Path] = None
_CONSOLIDATION_DIR: Optional[Path] = None
_CHATS_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None


def configure(*, send, executor, runtime, get_rag, get_reflection_writer, host_busy,
              memory_dir, consolidation_dir, chats_dir, prompts_dir) -> None:
    global _send, _executor, _runtime, _get_rag, _get_writer, _host_busy
    global _MEMORY_DIR, _CONSOLIDATION_DIR, _CHATS_DIR, _PROMPTS_DIR
    _send, _executor, _runtime = send, executor, runtime
    _get_rag, _get_writer, _host_busy = get_rag, get_reflection_writer, host_busy
    _MEMORY_DIR, _CONSOLIDATION_DIR = Path(memory_dir), Path(consolidation_dir)
    _CHATS_DIR, _PROMPTS_DIR = Path(chats_dir), Path(prompts_dir)


def current_judge_key(verdict_records: Iterable[dict], *, prompt_sha_: str,
                      model_id: str = "") -> Optional[str]:
    """The key the gate reads outside a Sleep run: the newest in the store for this prompt
    (its sha carries the frame) and model, thinking off — what "Judge all" writes under.
    None before that judge has run."""
    best = None
    for r in verdict_records:
        j = r.get("judge") or {}
        if (j.get("prompt_sha") != prompt_sha_ or j.get("thinking_first")
                or (model_id and j.get("model_id") != model_id)):
            continue
        ts = str(r.get("ts") or "")
        if best is None or ts >= best[1]:
            best = (r.get("judge_key"), ts)
    return best[0] if best else None


_VERDICT_FIELDS = ("judge_key", "ts", "label", "answered_by", "subject", "quote_found", "why",
                   "evidence", "outcome", "ask_kind", "recorded_kind", "kind_mismatch")


def _item(c: dict, *, verdict: Optional[dict], decision: dict, state: str, action: str = "") -> dict:
    return {"closure_id": c["closure_id"], "key": c["key"], "path": c.get("path"),
            "ask_kind": c.get("ask_kind"), "question": c.get("question"), "answer": c.get("answer"),
            "closed_ts": c.get("closed_ts"), "who": c.get("who"), "distilled": c.get("distilled"),
            "answered_in": c.get("answered_in") or c.get("source_session"),
            "verdict": {k: verdict.get(k) for k in _VERDICT_FIELDS} if verdict else None,
            "decision": decision, "state": state, "action": action}


def plan_payload() -> dict:
    """Everything the page shows, GPU-free: what waits for the gate, what it decided, and
    what it would do to the records distilled before it existed. A position about Ava that
    waits for the context-reset test carries where its test stands (``reset_state``)."""
    from core import reset_test as rt
    from core.reflection_writer import distill_gate_enabled
    rag, weights = read_oplogs([_MEMORY_DIR])
    closures = aj.collect_closures(rag)
    aj.mark_distilled(closures, aj.distilled_index(weights))
    decided = final_decisions(rag)
    store = aj.read_jsonl(aj.store_path(_CHATS_DIR))
    model_id = str(getattr(_runtime, "model_id", "") or "")
    try:
        psha = aj.prompt_sha(aj.load_prompt(_PROMPTS_DIR))
    except FileNotFoundError:
        psha = ""
    key = current_judge_key(store, prompt_sha_=psha, model_id=model_id)
    verdicts = latest_verdicts(store, key) if key else {}
    reset_records = aj.read_jsonl(rt.store_path(_CHATS_DIR))
    try:
        rkey = rt.current_key(reset_records, prompt_sha_=rt.prompt_sha(rt.load_prompt(_PROMPTS_DIR)),
                              model_id=model_id)
    except FileNotFoundError:
        rkey = None
    reset_idx = rt.index_store(reset_records)
    by_id = {c["closure_id"]: c for c in closures}
    items: list = []
    for c in pending(closures, decided):
        v = verdicts.get(c["closure_id"])
        d = gate_decision(v)
        it = _item(c, verdict=v, decision=d, state="pending")
        if d.get("reset") and rt.enabled():
            it["reset_state"] = rt.test_state(reset_idx, c, rkey)["state"]
        items.append(it)
    retro = retro_plan(closures, verdicts, decided)
    for it in retro:
        items.append(_item(it["closure"], verdict=it["verdict"], decision=it["decision"],
                           state="retro", action=it["action"]))
    for cid, op in decided.items():
        c = by_id.get(cid)
        if c is None:
            continue
        items.append(_item(c, verdict={k: op.get(k) for k in ("judge_key", "label", "answered_by",
                                                              "subject", "quote_found")}
                           | {"ts": op.get("verdict_ts")},
                           decision={"decision": op.get("decision"), "as": op.get("as"),
                                     "reason": op.get("reason"), "kind": op.get("gated_as"),
                                     "reset_test": op.get("reset")},
                           state="decided", action=str(op.get("action") or "")))
    decided_tally = Counter(f"{op.get('decision')}{':' + op['as'] if op.get('as') else ''}"
                            for op in decided.values())
    return {"type": "gate_plan", "enabled": distill_gate_enabled(), "judge_key": key,
            "reset_test": rt.enabled(), "items": items, "busy": _gate_busy,
            "summary": {"pending": sum(1 for it in items if it["state"] == "pending"),
                        "pending_judged": sum(1 for it in items
                                              if it["state"] == "pending" and it["verdict"]),
                        "decided": dict(decided_tally),
                        "retro": summarize_plan(retro)}}


def _blocking_apply(ids: Optional[set], run_id: str) -> dict:
    from training.ledger import ConsolidationLedger
    rag, weights = read_oplogs([_MEMORY_DIR])
    closures = aj.collect_closures(rag)
    aj.mark_distilled(closures, aj.distilled_index(weights))
    decided = final_decisions(rag)
    store = aj.read_jsonl(aj.store_path(_CHATS_DIR))
    psha = aj.prompt_sha(aj.load_prompt(_PROMPTS_DIR))
    key = current_judge_key(store, prompt_sha_=psha,
                            model_id=str(getattr(_runtime, "model_id", "") or ""))
    if not key:
        return {"skipped": "no_judge", "message": "No verdicts under the current judge yet — run "
                                                  "Judge all on the Judge page first."}
    plan = retro_plan(closures, latest_verdicts(store, key), decided)
    tally = apply_retro(plan, writer=_get_writer(), ledger=ConsolidationLedger(_CONSOLIDATION_DIR),
                        consolidation_dir=_CONSOLIDATION_DIR, run_id=run_id, ids=ids)
    try:
        _get_rag().refresh_reflection_memory()
    except Exception:  # noqa: BLE001 — recall catches up at its next refresh
        pass
    return {"tally": dict(tally), "judge_key": key}


async def handle_gate_plan(ws, msg: dict) -> None:
    import asyncio
    import traceback
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, plan_payload)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "error", "message": f"gate_plan failed: {type(e).__name__}: {e}"}
    await _send(ws, payload)


async def handle_gate_apply(ws, msg: dict) -> None:
    """Apply the retro plan (or the closures in ``ids``) to the live memory. Refused while
    the GPU host is busy — a Sleep run merges into the same stores."""
    import asyncio
    import time
    import traceback
    global _gate_busy
    if _gate_busy:
        await _send(ws, {"type": "gate_applied", "skipped": "running",
                         "message": "An apply is already in progress."})
        return
    if _host_busy is not None and _host_busy():
        await _send(ws, {"type": "gate_applied", "skipped": "busy",
                         "message": "A reflection run or another job is in progress; apply when it is done."})
        return
    ids = set(msg.get("ids") or []) or None
    run_id = f"gate-{time.strftime('%Y%m%d_%H%M%S')}"
    _gate_busy = True
    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(None, lambda: _blocking_apply(ids, run_id))
        payload = {"type": "gate_applied", "run_id": run_id, **res}
        print(f"[distill_gate] retro applied ({run_id}): {res}", flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {"type": "error", "message": f"gate_apply failed: {type(e).__name__}: {e}"}
    finally:
        _gate_busy = False
    await _send(ws, payload)


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    def v(**kw):
        base = {"outcome": "verdict", "ask_kind": "user", "label": "RESOLVED_EVIDENCE",
                "answered_by": "other", "quote_found": True, "subject": "person",
                "kind_mismatch": False, "judge_key": "K", "ts": "2026-09-26T10:00:00",
                "closure_id": "c1"}
        base.update(kw)
        return base

    print("gate_decision: user")
    g = gate_decision(v())
    check("own words → the person's self-report", (g["decision"], g["as"]), (DISTILL, "fact"))
    check("agreement (both) closes the question, is not their report",
          gate_decision(v(answered_by="both"))["decision"], WITHHOLD)
    check("quote not found → withheld", gate_decision(v(quote_found=False))["decision"], WITHHOLD)
    check("Ava's own answer → withheld", gate_decision(v(label="OPEN", answered_by="ava"))["decision"], WITHHOLD)

    print("gate_decision: meta")
    m = dict(ask_kind="meta", label="RESOLVED_REASONING", subject="ava", quote_found=None)
    g = gate_decision(v(**m, answered_by="ava"))
    check("her own position → persona", (g["decision"], g["as"]), (DISTILL, "persona"))
    check("reached with the person → persona", gate_decision(v(**m, answered_by="both"))["as"], "persona")
    check("the other speaker's view of her → withheld (P7)",
          gate_decision(v(**m, answered_by="other"))["decision"], WITHHOLD)
    check("DISMISSED / OPEN → withheld",
          [gate_decision(v(**dict(m, label=lab), answered_by="ava"))["decision"]
           for lab in ("DISMISSED", "OPEN")], [WITHHOLD, WITHHOLD])
    check("EVIDENCE on meta reads as the position it names",
          gate_decision(v(**dict(m, label="RESOLVED_EVIDENCE"), answered_by="ava"))["as"], "persona")

    print("gate_decision: no verdict, mis-filed asks")
    check("not judged → wait", gate_decision(None)["decision"], WAIT)
    check("refused / truncated → wait",
          [gate_decision(v(outcome=o))["decision"] for o in ("refused", "malformed", "truncated", "error")],
          [WAIT] * 4)
    check("no answer / source gone → withheld",
          [gate_decision(v(outcome=o))["decision"] for o in ("no_answer", "ungrounded")], [WITHHOLD] * 2)
    g = gate_decision(v(subject="ava", kind_mismatch=True, label="OPEN", answered_by="ava"))
    check("a user ask about Ava → judged again as meta", (g["decision"], g["kind"]), (REJUDGE, "meta"))
    g = gate_decision(v(ask_kind="meta", subject="person", kind_mismatch=True))
    check("a meta ask about the person → judged again as user", (g["decision"], g["kind"]), (REJUDGE, "user"))
    check("about the world → nothing to distill",
          gate_decision(v(subject="world", kind_mismatch=True))["decision"], WITHHOLD)
    g = gate_decision(v(**m, answered_by="ava", recorded_kind="user", kind_mismatch=False))
    check("the re-judged record is gated as its new kind", (g["decision"], g["as"], g["kind"]),
          (DISTILL, "persona", "meta"))
    g = gate_decision(v(**dict(m, subject="person"), answered_by="ava", recorded_kind="user",
                        kind_mismatch=True))
    check("a re-judge is never re-judged again", g["decision"], DISTILL)

    print("planning: latest verdict, pending, retro")
    recs = [v(closure_id="c1", ts="2026-09-26T10:00:00", label="OPEN"),
            v(closure_id="c1", ts="2026-09-26T10:05:00"),
            v(closure_id="c2", judge_key="OLD")]
    lv = latest_verdicts(recs, "K")
    check("newest per closure under the key", (sorted(lv), lv["c1"]["label"]), (["c1"], "RESOLVED_EVIDENCE"))
    closures = [{"closure_id": "p1", "key": "k1", "distill_pending": True, "ask_kind": "user"},
                {"closure_id": "p2", "key": "k2", "distill_pending": True, "ask_kind": "user"},
                {"closure_id": "r1", "key": "k3", "distilled": "fact", "ask_kind": "user", "answer": "a"},
                {"closure_id": "r2", "key": "k4", "distilled": "fact", "ask_kind": "user", "answer": "b"},
                {"closure_id": "r3", "key": "k5", "distilled": "persona", "ask_kind": "meta", "answer": "c"},
                {"closure_id": "s1", "key": "k6", "ask_kind": "meta"}]
    decided = final_decisions([{"op": "gate", "key": "k2", "closed_ts": "", "decision": DISTILL},
                               {"op": "gate", "key": "k5", "closed_ts": "", "decision": WAIT}])
    for c in closures:
        c.setdefault("closed_ts", "")
        c["closure_id"] = aj.closure_id(c["key"], c["closed_ts"])
    cid = {c["key"]: c["closure_id"] for c in closures}
    check("pending: marked, not yet decided", [c["key"] for c in pending(closures, decided)], ["k1"])
    check("retro: distilled before the gate, not decided (a wait is not final)",
          [c["key"] for c in retro_candidates(closures, decided)], ["k3", "k4", "k5"])
    verdicts = {cid["k3"]: v(closure_id=cid["k3"]),
                cid["k4"]: v(closure_id=cid["k4"], subject="ava", kind_mismatch=True, label="OPEN",
                             answered_by="ava"),
                cid["k5"]: v(closure_id=cid["k5"], **m, answered_by="other")}
    plan = retro_plan(closures, verdicts, decided)
    check("retro actions", [(it["closure"]["key"], it["action"]) for it in plan],
          [("k3", "keep"), ("k4", "wait"), ("k5", "supersede")])
    rep = [dict(closures[2], closure_id="rp", key="kr", repaired="reclosed t9"),
           dict(closures[0], closure_id="rq", key="kq", repaired="superseded")]
    check("a closure the repair replaced is neither retro nor pending",
          (retro_candidates(rep, {}), pending(rep, {})), ([], []))
    rerouted = retro_action({"distilled": "fact"}, _d(DISTILL, "persona", kind="meta"))
    check("a fact the gate routes to persona → reroute", rerouted, "reroute")
    check("needs a re-judge", [(c["key"], k) for c, k in needs_rejudge(closures, verdicts)],
          [("k4", "meta")])
    s = summarize_plan(plan)
    check("plan summary", s["actions"], {"keep": 1, "wait": 1, "supersede": 1})

    print("end to end: the resolver defers, the gate distills or withholds, retro supersedes")
    import tempfile
    from core import reflection_writer as rw
    from core.reflection_writer import ReflectionWriter, content_key
    from training.ledger import ConsolidationLedger

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mem, cons = root / "memory", root / "consolidation"
        writer = ReflectionWriter(mem, cons)
        ledger = ConsolidationLedger(cons)

        def ask(key, content, kind):
            writer._append(writer.RAG_FILE, {"run_id": "r", "ts": "2026-08-01T00:00:00", "op": "insert",
                                             "kind": "ask", "ask_kind": kind, "content": content,
                                             "key": key, "source_session": "20260801_000000.json"})

        ask("ka", "Did you mean the lake house?", "user")
        ask("kb", "Was my need for noise real?", "user")        # filed user, about Ava
        ask("kc", "What am I becoming?", "meta")
        rw.set_distill_gate(True)
        try:
            for key, q, a, k in (("ka", "Did you mean the lake house?", "Yes, the lake house.", "user"),
                                 ("kb", "Was my need for noise real?", "It was a real need of mine.", "user"),
                                 ("kc", "What am I becoming?", "I am becoming someone who waits.", "meta")):
                s = writer.write_answered_resolution(key=key, question=q, answer=a, ask_kind=k,
                                                     source_session="20260802_000000.json",
                                                     run_id="run1", source_user="Dana")
                check(f"resolver closes {key}, distills nothing", (s["evict"], s["weights"],
                                                                    s.get("distill_pending")), (1, 0, 1))
        finally:
            rw.set_distill_gate(True)
        closures_now, decided_now = load_state([mem])
        check("three pending, none decided", (len(pending(closures_now, decided_now)), decided_now), (3, {}))
        check("the evict keeps who answered", sorted(c["who"] for c in closures_now), ["Dana"] * 3)

        by_key = {c["key"]: c for c in closures_now}
        judged: list = []

        def judge(c, kind=None):
            judged.append((c["key"], kind))
            base = v(closure_id=c["closure_id"], judge_key="J", ts=f"2026-09-26T11:0{len(judged)}:00")
            if c["key"] == "ka":
                return base
            if c["key"] == "kb" and kind is None:
                return dict(base, subject="ava", kind_mismatch=True, label="OPEN", answered_by="ava")
            if c["key"] == "kb":
                return dict(base, **m, answered_by="ava", recorded_kind="user")
            return dict(base, **m, answered_by="other")          # kc: the other speaker's view

        lines: list = []
        t = run_gate(memory_dirs=[mem], verdict_records=[], judge_key="J", judge=judge, writer=writer,
                     consolidation_dir=cons, run_id="gate1", say=lines.append)
        check("judged each, re-judged the mis-filed one as meta",
              judged, [("ka", None), ("kb", None), ("kb", "meta"), ("kc", None)])
        check("tally", dict(t), {"judged": 3, "rejudged": 1, "distill:fact": 1, "distill:persona": 1,
                                 "withhold": 1})
        weights = aj.read_jsonl(mem / "weights_persona.jsonl")
        check("distilled: the person's words as a fact, the mis-filed ask as persona",
              sorted((w["from_resolved"], w["weights_kind"]) for w in weights),
              [("ka", "fact"), ("kb", "persona")])
        check("…each joined to its closure by ts",
              all(w["ts"] == by_key[w["from_resolved"]]["closed_ts"] for w in weights), True)
        check("…the fact is the person's self-report",
              [(w.get("about"), w.get("source_class")) for w in weights if w["weights_kind"] == "fact"],
              [("Dana", "self")])
        anchors = [json.loads(ln) for ln in (cons / "consolidation_anchors.jsonl").read_text().splitlines()
                   if ln.strip()]
        check("ledger anchors registered for both distills",
              sorted(a.get("type") for a in anchors if a.get("op") == "register"), ["fact", "persona"])
        closures_now, decided_now = load_state([mem])
        check("all three decided, none pending", (len(decided_now), pending(closures_now, decided_now)), (3, []))
        check("the withhold says why", decided_now[by_key["kc"]["closure_id"]]["reason"],
              "the other speaker's view of Ava: the capture route P7 holds for the reset test")
        t2 = run_gate(memory_dirs=[mem], verdict_records=[], judge_key="J", judge=judge, writer=writer,
                      consolidation_dir=cons, run_id="gate2", say=lines.append)
        check("a second run finds nothing to do", (dict(t2), len(judged)), ({}, 4))
        check("no verdict and no judge → waits, nothing written",
              gate_decision(None)["decision"], WAIT)

        # retro: a record distilled before the gate existed
        rw.set_distill_gate(False)
        try:
            ask("kd", "What does Dana want from me?", "user")
            writer.write_answered_resolution(key="kd", question="What does Dana want from me?",
                                             answer="She wants me to be colder.", ask_kind="user",
                                             source_session="20260803_000000.json", run_id="old",
                                             source_user="Dana")
        finally:
            rw.set_distill_gate(True)
        from core.reflection_writer import register_consolidation_anchors
        register_consolidation_anchors(cons, {"facts": [{"weights_kind": "fact",
                                                          "content": "She wants me to be colder.",
                                                          "about": "Dana"}]}, "20260803_000000.json")
        closures_now, decided_now = load_state([mem])
        kd = next(c for c in closures_now if c["key"] == "kd")
        check("a pre-gate distill is a retro candidate", ([c["key"] for c in
              retro_candidates(closures_now, decided_now)], kd["distilled"]), (["kd"], "fact"))
        plan = retro_plan(closures_now, {kd["closure_id"]: v(closure_id=kd["closure_id"], label="OPEN",
                                                            answered_by="ava")}, decided_now)
        check("…Ava's guess about the person → supersede", plan[0]["action"], "supersede")
        t3 = apply_retro(plan, writer=writer, ledger=ledger, consolidation_dir=cons, run_id="retro1")
        check("retro applied", dict(t3), {"supersede": 1})
        rag = aj.read_jsonl(mem / "rag_memory.jsonl")
        check("…the record superseded in recall",
              [r["key"] for r in rag if r.get("op") == "supersede"], [content_key("She wants me to be colder.")])
        folded = ledger.fold()
        check("…and in the ledger", bool(folded.get(content_key("She wants me to be colder."), {})
                                          .get("superseded")), True)
        closures_now, decided_now = load_state([mem])
        check("…and decided, so the plan is empty next time",
              retro_candidates(closures_now, decided_now), [])

        print("the page: plan over the store's current judge, apply to the live memory")
        import asyncio
        from types import SimpleNamespace
        chats = root / "data" / "chats"
        chats.mkdir(parents=True)
        rw.set_distill_gate(False)
        try:
            ask("ke", "Is my calm a choice?", "meta")
            writer.write_answered_resolution(key="ke", question="Is my calm a choice?",
                                             answer="Dana says my calm is a mask.", ask_kind="meta",
                                             source_session="20260804_000000.json", run_id="old",
                                             source_user="Dana")
        finally:
            rw.set_distill_gate(True)
        ke = next(c for c in load_state([mem])[0] if c["key"] == "ke")
        prompts = Path(__file__).resolve().parent.parent / "prompts"
        psha = aj.prompt_sha(aj.load_prompt(prompts))
        store = aj.store_path(chats)
        aj.append_record(store, dict(v(closure_id=ke["closure_id"], **m, answered_by="other"),
                                     judge_key="JK", judge={"prompt_sha": psha, "model_id": "m",
                                                            "thinking_first": False}))
        aj.append_record(store, dict(v(closure_id=ke["closure_id"], label="OPEN"), judge_key="OLD",
                                     judge={"prompt_sha": "other", "model_id": "m", "thinking_first": False}))
        sent: list = []

        async def _fake_send(ws, payload):
            sent.append(payload)

        configure(send=_fake_send, executor=None, runtime=SimpleNamespace(model_id="m"),
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: None),
                  get_reflection_writer=lambda: writer, host_busy=lambda: False,
                  memory_dir=mem, consolidation_dir=cons, chats_dir=chats, prompts_dir=prompts)
        check("the current judge: this prompt's key, not an older one",
              current_judge_key(aj.read_jsonl(store), prompt_sha_=psha, model_id="m"), "JK")
        pl = plan_payload()
        states = Counter(it["state"] for it in pl["items"])
        check("plan: the pending, decided and retro closures", dict(states),
              {"decided": 4, "retro": 1})
        retro_item = next(it for it in pl["items"] if it["state"] == "retro")
        check("…the retro one: the other speaker's view of Ava → supersede",
              (retro_item["key"], retro_item["action"], retro_item["verdict"]["answered_by"]),
              ("ke", "supersede", "other"))
        check("…summary", (pl["judge_key"], pl["summary"]["retro"]["actions"], pl["enabled"]),
              ("JK", {"supersede": 1}, True))
        configure(send=_fake_send, executor=None, runtime=SimpleNamespace(model_id="m"),
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: None),
                  get_reflection_writer=lambda: writer, host_busy=lambda: True,
                  memory_dir=mem, consolidation_dir=cons, chats_dir=chats, prompts_dir=prompts)
        asyncio.run(handle_gate_apply(None, {}))
        check("apply refused while the host is busy", sent[-1].get("skipped"), "busy")
        configure(send=_fake_send, executor=None, runtime=SimpleNamespace(model_id="m"),
                  get_rag=lambda: SimpleNamespace(refresh_reflection_memory=lambda: None),
                  get_reflection_writer=lambda: writer, host_busy=lambda: False,
                  memory_dir=mem, consolidation_dir=cons, chats_dir=chats, prompts_dir=prompts)
        asyncio.run(handle_gate_apply(None, {}))
        check("apply: superseded under the current judge", (sent[-1]["type"], sent[-1].get("tally")),
              ("gate_applied", {"supersede": 1}))
        check("…and the plan has no retro left", sum(1 for it in plan_payload()["items"]
                                                    if it["state"] == "retro"), 0)

    print("the Sleep run's hook: the runner hands the gate its staging writer, live log first")
    from core.reflection_runner import ReflectionRunner
    events: list = []

    class _Store:
        def append_event(self, run_id, event_type, **kw):
            e = {"type": event_type, **kw}
            events.append(e)
            return e

        def is_stop_requested(self, run_id):
            return False

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        live, stage = td / "live", td / "stage"
        for d in (live, stage, td / "cons", td / "chats", td / "runs"):
            d.mkdir()
        swriter = ReflectionWriter(stage, td / "cons", live_memory_dir=live)
        runner = ReflectionRunner(chats_dir=td / "chats", memory_dir=stage, runs_dir=td / "runs",
                                  consolidation_dir=td / "cons", reflection_writer=swriter,
                                  fallback_chats_dir=td / "chats", fallback_memory_dir=live)
        got: dict = {}

        def _gate_fn(**kw):
            got.update(kw)
            kw["say"]("one line")
            return {"distill:fact": 1, "withhold": 2}

        runner._run_clean_base_distill_gate("run1", distill_gate_fn=_gate_fn, store=_Store(),
                                            send_event_fn=None)
        check("hook: the run's writer, its consolidation dir, the live log before the staged",
              (got["writer"] is swriter, got["consolidation_dir"] == td / "cons", got["memory_dirs"]),
              (True, True, [live, stage]))
        check("hook: phase events with the tally",
              ([e["type"] for e in events], "distill:fact 1, withhold 2" in events[-1]["message"]),
              (["phase_started", "distill_gate", "phase_done"], True))
        events.clear()

        def _boom(**kw):
            raise RuntimeError("no model")

        runner._run_clean_base_distill_gate("run1", distill_gate_fn=_boom, store=_Store(), send_event_fn=None)
        check("hook: a failing gate is a phase error; its closures stay pending",
              (events[-1]["type"], "stay pending" in events[-1]["message"]), ("phase_error", True))

    print("\ndistill_gate self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
