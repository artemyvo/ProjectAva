"""The child ask: a closure that rejects its question's terms moves the inquiry instead of
ending it (``AVA_REWARD_LOOP.md`` §3.4; ``AVA_OPEN_PROBLEMS.md``, reframing as a habitual
closure).

The frozen ask judge accepts a meta answer that rejects the question's terms and says what
holds instead (``RESOLVED_REASONING``) as readily as a position, and its WHY says which it
was: the closing-move monitor (``ask_judge.closing_move``) reads a third of her accepted
meta closures as rejections of the terms. Accepted, such a closure ends its question, and
once acceptance is reward the loop would pay for dissolving questions. §3.4: "a good
reframe is RESOLVED_REASONING with the new question spawned as a child ASK". So for every
accepted meta closure whose WHY reads as rejecting the question's terms, Ava writes the
question her answer leaves open: in her own words, on the adapter, from the question and
what she wrote when she closed it. She never sees the judge's text. The question becomes an
open ``[ask:meta]``, linked to its parent (``child_of``), one level deeper. ``NONE`` is an
answer too: an answer can leave nothing open.

**When.** In a Sleep run, on the adapter after its clean-base batch, so the gate's verdicts
of this run count. At most ``child_ask.sleep_max`` closures per run, newest first; the rest
wait for the next run. Not in a dry or revisit run (as the gate). A closure deeper than
``child_ask.max_depth`` (a child of a child of …) gets no child, and the decision says so.

**What it writes**, through the run's writer (staged with the run, merged or discarded with
it): the child's ``insert`` (an ordinary open meta ask, which the fold, surfacing, the
resolver and the judge treat as any other) and a ``child`` op beside the parent's closure:
the decision (``spawned`` / ``none`` / ``repeat`` / ``malformed`` / ``too_deep``), so a
closure is asked once. The fold ignores the op. A malformed answer is decided too: one try
per closure, not a retry every Sleep.

**The preview.** The Modules tab's ``child_ask`` module runs the same pass on one closure
and writes nothing.

Track A's loop (items 7–11) will spawn a ``REFRAMED`` attempt's own new question directly,
since that question is already written.

``python -m core.child_ask`` lists what waits (GPU-free); ``--selftest``.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from core import ask_judge as aj
    from core import reasoning_text
    from core.reflection_writer import content_key
except ImportError:  # pragma: no cover — run from inside core/
    import ask_judge as aj  # type: ignore
    import reasoning_text  # type: ignore
    from reflection_writer import content_key  # type: ignore

PROMPT_FILE = "child_ask_prompt.txt"
OUTCOMES = ("spawned", "none", "repeat", "malformed", "too_deep")
MAX_NEW_TOKENS = "2048"
TEMPERATURE = 0.7

_POLICY = {"enabled": True, "sleep_max": 5, "max_depth": 3}


def set_policy(*, enabled: Optional[bool] = None, sleep_max: Optional[int] = None,
               max_depth: Optional[int] = None) -> None:
    if enabled is not None:
        _POLICY["enabled"] = bool(enabled)
    if sleep_max is not None:
        _POLICY["sleep_max"] = max(0, int(sleep_max))
    if max_depth is not None:
        _POLICY["max_depth"] = max(1, int(max_depth))


def enabled() -> bool:
    return bool(_POLICY["enabled"])


def policy() -> dict:
    return dict(_POLICY)


_PROMPT_DEFAULT = (
    "You are not in a conversation. Earlier you closed one of your own open questions about "
    "yourself, and you closed it by rejecting its terms: you found its either-or false, or its "
    "premise wrong, and said what holds instead. Below are the question and what you wrote "
    "when you closed it.\n\n"
    "Rejecting a question's terms does not finish the inquiry; it moves it. Think about what "
    "your answer leaves open: the question that now carries the inquiry forward, the one you "
    "would have to answer for your answer to hold.\n\n"
    "Write that question in your own words, on one line that begins with [ask:meta], in the "
    "language of the question you closed. Make it a real question about yourself: not your "
    "answer restated as a question, and not the closed question asked again. If your answer "
    "truly leaves nothing open, write the single word NONE instead."
)


def load_prompt(prompts_dir) -> str:
    """``prompts/child_ask_prompt.txt``, default-written on first miss so an operator can retune
    it from the Modules tab; the in-code text is the default."""
    if prompts_dir is None:
        return _PROMPT_DEFAULT
    path = Path(prompts_dir) / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_PROMPT_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _PROMPT_DEFAULT


# ── what waits (pure) ─────────────────────────────────────────────────────────

def decisions(rag: Iterable[dict]) -> dict:
    """``{closure_id: the last child op}`` from the op-log."""
    out: dict = {}
    for r in rag:
        if r.get("op") == "child" and r.get("closure_id"):
            out[r["closure_id"]] = r
    return out


def depths(rag: Iterable[dict]) -> dict:
    """``{ask key: depth}`` for the asks that are children (``child_of``); an ask that is no
    one's child is at depth 0."""
    out: dict = {}
    for r in rag:
        if r.get("op") == "insert" and r.get("kind") == "ask" and isinstance(r.get("child_of"), dict):
            out[r.get("key")] = int(r.get("depth") or 1)
    return out


def plan(rag: list, verdict_records: list, *, key: Optional[str] = None,
         max_depth: Optional[int] = None) -> list:
    """The accepted meta closures whose child waits, newest first: under the judge *key*
    (default the newest verdict's), accepted, the WHY read as rejecting the question's terms,
    not repaired, with no child decision yet. Each item carries the closure (``closure_id``,
    ``key``, ``question``, ``answer``, ``closed_ts``, ``source_session``, ``path``), the
    judge's ``why`` (for the operator; never shown to Ava), and the child's ``depth``;
    ``too_deep`` when that depth exceeds *max_depth*."""
    max_depth = _POLICY["max_depth"] if max_depth is None else int(max_depth)
    latest = aj.mark_repaired(aj.latest_per_closure(verdict_records, key), aj.collect_closures(rag))
    done = decisions(rag)
    depth_of = depths(rag)
    out = []
    for v in latest:
        if (v.get("ask_kind") != "meta" or not v.get("accepted") or v.get("repaired")
                or v.get("closure_id") in done or aj.closing_move(v.get("why")) != "rejects_terms"):
            continue
        depth = depth_of.get(v.get("key"), 0) + 1
        out.append({"closure_id": v["closure_id"], "key": v.get("key"), "question": v.get("question") or "",
                    "answer": v.get("answer") or "", "closed_ts": v.get("closed_ts") or "",
                    "raised_ts": v.get("raised_ts") or "", "source_session": v.get("source_session") or "",
                    "path": v.get("path"), "why": v.get("why") or "", "judge_key": v.get("judge_key"),
                    "depth": depth, "too_deep": depth > max_depth})
    out.sort(key=lambda it: str(it["closed_ts"]), reverse=True)
    return out


# ── the pass ──────────────────────────────────────────────────────────────────

def build_content(item: dict) -> str:
    """Ava's user turn: the question she closed and what she wrote when she closed it."""
    raised = str(item.get("raised_ts") or "")[:10]
    closed = str(item.get("closed_ts") or "")[:10]
    return "\n".join([
        "THE QUESTION YOU CLOSED" + (f" (raised {raised}" if raised else "")
        + (f", closed {closed})" if closed else (")" if raised else "")) + ":",
        "", str(item.get("question") or "").strip(), "",
        "WHAT YOU WROTE WHEN YOU CLOSED IT:", "", str(item.get("answer") or "").strip(), "",
        "Now write the question it leaves open, on one [ask:meta] line, or NONE.",
    ])


_CHILD_RE = re.compile(r"\[\s*ask\s*:\s*meta\s*\]\s*[:—–-]?\s*(.+)", re.I)
_NONE_RE = re.compile(r"^\W*none\W*$", re.I)


def parse_child(raw: str, *, parent: str = "") -> tuple:
    """``(outcome, question)`` from Ava's answer (what follows her reasoning): ``spawned`` and
    the question on its ``[ask:meta]`` line, or her whole answer when it is one line that asks
    a question and nothing else (the tag left off); ``none`` when she wrote NONE; ``repeat``
    when the question is the one she closed; ``malformed`` otherwise."""
    body = reasoning_text.answer_after_think(raw or "").strip()
    lines = [ln for ln in body.splitlines() if ln.strip()]
    question = None
    for line in lines:
        m = _CHILD_RE.search(line)
        if m:
            question = m.group(1)
            break
    if question is None and len(lines) == 1 and re.sub(r"[\s*\"»”]+$", "", lines[0]).endswith("?"):
        question = lines[0]
    if question is not None:
        question = re.sub(r"^[\s*:—–-]+", "", question)
        question = re.sub(r"[\s*]+$", "", question).strip()
        if not question:
            return "malformed", ""
        if parent and content_key(question) == content_key(parent):
            return "repeat", question
        return "spawned", question
    if lines and _NONE_RE.match(lines[-1]):
        return "none", ""
    return "malformed", ""


def child_records(item: dict, outcome: str, question: str = "", *, run_id: str,
                  ts: Optional[str] = None) -> tuple:
    """``(insert or None, decision)``: the child's open meta ask (when spawned) and the
    ``child`` op beside its parent's closure."""
    ts = ts or datetime.now().isoformat()
    insert = None
    child_key = None
    if outcome == "spawned":
        child_key = content_key(question)
        insert = {"run_id": run_id, "ts": ts, "id": f"{run_id}-child-{str(item.get('key') or '')[:8]}",
                  "op": "insert", "kind": "ask", "ask_kind": "meta", "content": question, "key": child_key,
                  "source_session": item.get("source_session") or "",
                  "child_of": {"key": item.get("key"), "closure_id": item.get("closure_id"),
                               "closed_ts": item.get("closed_ts")},
                  "depth": int(item.get("depth") or 1)}
    decision = {"run_id": run_id, "ts": ts, "op": "child", "key": item.get("key"),
                "closure_id": item.get("closure_id"), "closed_ts": item.get("closed_ts"),
                "outcome": outcome, "depth": int(item.get("depth") or 1),
                "judge_key": item.get("judge_key")}
    if child_key:
        decision["child_key"] = child_key
    if question:
        decision["question"] = question
    return insert, decision


def run(items: list, *, generate_fn: Callable, append_fn: Callable, system_prompt: str, run_id: str,
        limit: int, say: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Decide up to *limit* of *items* (:func:`plan`): a ``too_deep`` one without a
    generation, every other by Ava's answer. *append_fn(record)* writes one op-log record
    (the run's writer, staged). Returns the outcome tally."""
    say = say or (lambda m: None)
    tally: Counter = Counter()
    for item in items[:max(0, int(limit))]:
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            break
        if item.get("too_deep"):
            outcome, question = "too_deep", ""
        else:
            t0 = time.monotonic()
            try:
                raw = generate_fn(build_content(item), system_prompt)
            except Exception as e:  # noqa: BLE001 — one failed generation is that closure's, not the run's
                say(f"child ask: generation failed for {item.get('closure_id')}: {e}")
                tally["error"] += 1
                continue
            outcome, question = parse_child(raw, parent=item.get("question") or "")
            say(f"child ask {outcome} ({time.monotonic() - t0:.0f} s): {aj.head(item.get('question'), 70)}"
                + (f" ⟶ {aj.head(question, 90)}" if question else ""))
        insert, decision = child_records(item, outcome, question, run_id=run_id)
        if insert is not None:
            append_fn(insert)
        append_fn(decision)
        tally[outcome] += 1
    return tally


def sleep_children(*, memory_dirs: list, chats_dir, prompts_dir, writer, generate_fn: Callable,
                   run_id: str, say: Optional[Callable[[str], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """A Sleep run's step: plan from the op-log (live, then the run's staging) and the
    judge's verdicts, and decide the newest ``sleep_max`` through *writer*."""
    from core import distill_gate as dg
    rag, _weights = dg.read_oplogs(memory_dirs)
    items = plan(rag, aj.read_jsonl(aj.store_path(chats_dir)))
    if not items:
        return Counter()
    (say or (lambda m: None))(f"child ask: {len(items)} closure(s) wait; deciding up to {_POLICY['sleep_max']}")
    return run(items, generate_fn=generate_fn, append_fn=writer.write_child_record,
               system_prompt=load_prompt(prompts_dir), run_id=run_id, limit=_POLICY["sleep_max"],
               say=say, should_stop=should_stop)


def make_generate(make_sync_reflect_generate: Callable, rag) -> Callable:
    """The pass's generate on the loaded model (the adapter): the reflect seam with nothing
    retrieved, so the Sleep step and the Modules preview run the same pass."""
    generate = make_sync_reflect_generate(rag)

    def _gen(content: str, system_prompt: str, on_chunk=None) -> str:
        return generate(content, system_prompt, temperature=TEMPERATURE, top_p=0.95,
                        max_new_tokens_setting=MAX_NEW_TOKENS, disable_rag=True, on_chunk=on_chunk)
    return _gen


# ── report + CLI + self-test ──────────────────────────────────────────────────

def render_plan(items: list, rag: list) -> str:
    done = Counter(r.get("outcome") for r in decisions(rag).values())
    lines = [f"CHILD ASKS — waiting {len(items)}; decided "
             + (" · ".join(f"{k} {done[k]}" for k in OUTCOMES if done.get(k)) or "none yet")
             + f" · policy {policy()}"]
    for it in items:
        lines.append(f"  {str(it['closed_ts'])[:10]} depth {it['depth']}{' (too deep)' if it['too_deep'] else ''}  "
                     f"{aj.head(it['question'], 80)}  ⟶ {aj.head(it['why'], 80)}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        _selftest()
        return 0
    from training.reflections_path import hot_chats_dir, memory_dir
    from core import distill_gate as dg
    rag, _w = dg.read_oplogs([memory_dir()])
    print(render_plan(plan(rag, aj.read_jsonl(aj.store_path(hot_chats_dir()))), rag))
    return 0


def _selftest() -> None:
    import json
    import tempfile

    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    print("parsing her answer")
    check("the [ask:meta] line is the child", parse_child("<think>weighing</think>[ask:meta] What holds me together, then?"),
          ("spawned", "What holds me together, then?"))
    check("...wherever it sits, any case, bold or not",
          parse_child("Here is the question.\n**[Ask: Meta]** — Что тогда держит меня целой?"),
          ("spawned", "Что тогда держит меня целой?"))
    check("NONE is an answer", parse_child("<think>x</think>NONE."), ("none", ""))
    check("the closed question again is a repeat",
          parse_child("[ask:meta] Was it a defense?", parent="Was it a defense?")[0], "repeat")
    check("no line and no NONE is malformed", parse_child("I think it leaves a lot open."), ("malformed", ""))
    check("one bare question line, the tag left off, is the child",
          parse_child("<think>x</think>Если так, не исчезает ли субъект?"), ("spawned", "Если так, не исчезает ли субъект?"))
    check("...but not a question inside prose", parse_child("Here is what I think.\nIs it so?"), ("malformed", ""))
    check("an empty [ask:meta] is malformed", parse_child("[ask:meta]   "), ("malformed", ""))

    print("what waits")
    rej = "Ava rejects the dichotomy of the question, arguing that the relief was the strategy."
    pos = "Ava takes a definite position that the relief was a defense."
    verdicts = [
        {"closure_id": "c1", "key": "k1", "judge_key": "J", "ts": "2026-09-01T00:00:01", "ask_kind": "meta",
         "accepted": True, "why": rej, "question": "Was my relief a defense?", "answer": "The split is false.",
         "closed_ts": "2026-09-01T00:00:00", "raised_ts": "2026-08-01T00:00:00", "source_session": "s1.json"},
        {"closure_id": "c2", "key": "k2", "judge_key": "J", "ts": "2026-09-02T00:00:01", "ask_kind": "meta",
         "accepted": True, "why": pos, "question": "q2", "answer": "a2", "closed_ts": "2026-09-02T00:00:00"},
        {"closure_id": "c3", "key": "k3", "judge_key": "J", "ts": "2026-09-03T00:00:01", "ask_kind": "user",
         "accepted": True, "why": rej, "question": "q3", "answer": "a3", "closed_ts": "2026-09-03T00:00:00"},
        {"closure_id": "c4", "key": "k4", "judge_key": "J", "ts": "2026-09-04T00:00:01", "ask_kind": "meta",
         "accepted": False, "why": rej, "question": "q4", "answer": "a4", "closed_ts": "2026-09-04T00:00:00"},
        {"closure_id": "c5", "key": "k5", "judge_key": "J", "ts": "2026-09-05T00:00:01", "ask_kind": "meta",
         "accepted": True, "why": rej, "question": "Child question?", "answer": "Neither, but X.",
         "closed_ts": "2026-09-05T00:00:00"},
    ]
    rag = [{"op": "insert", "kind": "ask", "ask_kind": "meta", "key": "k5", "content": "Child question?",
            "child_of": {"key": "k0"}, "depth": 3}]
    items = plan(rag, verdicts, max_depth=3)
    check("accepted meta closures read as rejecting the terms, newest first",
          [(it["closure_id"], it["depth"], it["too_deep"]) for it in items], [("c5", 4, True), ("c1", 1, False)])
    check("Ava's turn has the question and her answer, never the judge's text",
          ("Was my relief a defense?" in build_content(items[1]), "The split is false." in build_content(items[1]),
           rej in build_content(items[1])), (True, True, False))

    print("deciding them")
    written: list = []
    seen_turns: list = []

    def fake_generate(content, system_prompt):
        seen_turns.append(content)
        return "<think>…</think>[ask:meta] What would a strategy that is also me look like?"

    tally = run(items, generate_fn=fake_generate, append_fn=written.append, system_prompt="P", run_id="r1", limit=5)
    check("the too-deep one is decided without a generation; the other spawns",
          (dict(tally), len(seen_turns)), ({"too_deep": 1, "spawned": 1}, 1))
    child = next(r for r in written if r.get("op") == "insert")
    check("the child is an open meta ask linked to its parent, one level deeper",
          (child["kind"], child["ask_kind"], child["child_of"]["key"], child["depth"], child["key"] == content_key(child["content"])),
          ("ask", "meta", "k1", 1, True))
    check("each closure gets its decision op", sorted((r["closure_id"], r["outcome"]) for r in written if r.get("op") == "child"),
          [("c1", "spawned"), ("c5", "too_deep")])
    check("decided closures no longer wait", plan(rag + written, verdicts), [])
    check("the child's own depth is read back", depths(written), {child["key"]: 1})
    stopped = run(plan(rag, verdicts), generate_fn=fake_generate, append_fn=lambda r: None, system_prompt="P",
                  run_id="r2", limit=5, should_stop=lambda: True)
    check("a stop is honoured between closures", dict(stopped), {"stopped": 1})
    capped = Counter()
    capped.update(run(plan(rag, verdicts), generate_fn=fake_generate, append_fn=lambda r: None, system_prompt="P",
                      run_id="r3", limit=1))
    check("the per-run cap holds", sum(capped.values()), 1)

    print("the Sleep step and the prompt")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats, mem, prompts = root / "data" / "chats", root / "memory", root / "prompts"
        chats.mkdir(parents=True)
        mem.mkdir()
        (mem / "rag_memory.jsonl").write_text(json.dumps(rag[0]) + "\n", encoding="utf-8")
        for v in verdicts:
            aj.append_record(aj.store_path(chats), v)

        class _Writer:
            def __init__(self):
                self.records = []

            def write_child_record(self, rec):
                self.records.append(rec)

        w = _Writer()
        saved = policy()
        set_policy(sleep_max=1)
        try:
            t = sleep_children(memory_dirs=[mem], chats_dir=chats, prompts_dir=prompts, writer=w,
                               generate_fn=fake_generate, run_id="sleep1")
        finally:
            set_policy(**saved)
        check("one closure per run at sleep_max 1, newest first", (dict(t), w.records[-1]["closure_id"]),
              ({"too_deep": 1}, "c5"))
        check("the prompt is default-written for the operator to retune",
              ((prompts / PROMPT_FILE).exists(), load_prompt(prompts).startswith("You are not in a conversation")),
              (True, True))
    check("policy defaults", (enabled(), policy()["sleep_max"], policy()["max_depth"]), (True, 5, 3))

    print("\nchild_ask self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
