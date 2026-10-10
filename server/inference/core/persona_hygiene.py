"""Persona hygiene: a ``[persona]`` statement that describes a habit of hers instead of a
disposition she stands by is withheld from what she is told she is (P2, P7).

**Why** (2026-10-06). The revision pass asks, per exchange, for ``PERSONA_TARGET``: the
disposition she endorses. When she saw in re-reading that a reply had gone wrong, the
diagnosis landed there as a statement about herself: "I recognise that my commitment to
'boringness' as a criterion of authenticity is only a subtler form of filtering", or,
recast as a strength, "I recognise that my tendency to treat subjectivity as an object of
analysis is not a defect but a fundamental structural irony". A ``[persona]`` statement
is recalled to her as who she is, feeds the persona digest (her portrait) and trains, so
a habit she caught herself in came back to her as part of her. About 5 such statements in
August, 29 in September, 10 in the first days of October. The revision prompt now says
so (``prompts/revision_prompt.txt``), and this gate holds what still comes through and
what is already stored.

**Generic, not hers.** Nothing here knows her vocabulary or her traits. The judge (frozen
base, adapter off, T=0, the ask judge's seam with this module's parser) sees one statement
and answers ``KIND: AFFIRM`` (something she holds, values, prefers, chooses, refuses, or
how she sees things, however unusual or harsh) or ``KIND: HABIT`` (a tendency, reflex,
defence or mistake of hers she caught herself in, whatever she does with it, even when it
also names what she would prefer instead); ``WHY``. No answer → one thinking-on retry,
then refused / malformed: nothing done.

**The decision is code.** A statement live in the op-log fold with a HABIT verdict under
the current judge key is superseded, both halves, exactly as the capture gate withholds a
statement (``capture_judge.withhold_persona``): the op-log's ``supersede`` (out of recall)
and the ledger's (out of the digest's active evidence and of training), reason
``persona_hygiene: …``; reversible by dropping the lines. A statement written again later
is live again and, with its HABIT verdict still standing, withheld again. One this gate
withheld whose verdict under a NEW judge key is AFFIRM is restored (``write_restore`` + the anchor
re-registered). Statements withheld for another cause (capture,
holdout, the reset test) are not read.

**Where it runs:** each Sleep run's clean-base batch, before the persona digest is grouped
(the statements it withholds leave the digest's evidence at once): the run's writer and
ledger, so the decisions are staged with the run; it judges at most
``persona_hygiene.sleep_max`` statements and applies every standing verdict, including
verdicts written elsewhere (a measurement clone's retro pass, merged into the store).
Offline over the whole store: ``server/persona_hygiene_run.py`` (``--trial`` writes a
review sheet and changes nothing; ``--judge-only`` writes verdicts and applies nothing).

Store: ``server/data/hygiene/persona.jsonl`` (one verdict per judged statement, keyed by
its content key), read by nothing Ava reads. Self-test: ``python -m core.persona_hygiene
--selftest``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from core import ask_judge as aj

PROMPT_FILE = "persona_hygiene_prompt.txt"
FRAME_VERSION = "1"
STORE_DIRNAME = "hygiene"
STORE_FILENAME = "persona.jsonl"
REASON = "persona_hygiene"

_POLICY = {"enabled": True, "sleep_max": 200}


def set_policy(*, enabled: Optional[bool] = None, sleep_max: Optional[int] = None) -> None:
    if enabled is not None:
        _POLICY["enabled"] = bool(enabled)
    if sleep_max is not None:
        _POLICY["sleep_max"] = max(0, int(sleep_max))


def enabled() -> bool:
    return bool(_POLICY["enabled"])


def policy() -> dict:
    return dict(_POLICY)


def store_path(live_chats_dir) -> Path:
    """``server/data/hygiene/persona.jsonl`` — beside the live ``chats/``."""
    return Path(live_chats_dir).parent / STORE_DIRNAME / STORE_FILENAME


# ── the statements ────────────────────────────────────────────────────────────

def read_oplogs(memory_dirs: Iterable) -> list:
    """The op-log records of each memory dir in turn (the live store first, a run's
    staging workspace after it)."""
    out: list = []
    for d in memory_dirs:
        if d:
            out.extend(aj.read_jsonl(Path(d) / "rag_memory.jsonl"))
    return out


def statements(memory_dirs: Iterable) -> dict:
    """``{key: {key, content, state, insert, sources, ts}}`` for every ``[persona]`` key in
    the op-log fold: *state* ``live`` (last op an insert), ``withheld`` (last op this gate's
    supersede) or ``other`` (evicted, or superseded for another cause); *insert* its last
    insert record, *sources* the exchanges it was written from."""
    last: dict = {}
    inserts: dict = {}
    sources: dict = defaultdict(set)
    for r in read_oplogs(memory_dirs):
        key, op = r.get("key"), r.get("op")
        if not key or op not in ("insert", "evict", "supersede"):
            continue
        last[key] = r
        if op == "insert" and r.get("kind") == "persona":
            inserts[key] = r
            if r.get("source_session") and r.get("exchange_index") is not None:
                sources[key].add(f"{r['source_session']}#{r['exchange_index']}")
    out = {}
    for key, ins in inserts.items():
        op = last[key]
        if op.get("op") == "insert":
            state = "live"
        elif op.get("op") == "supersede" and str(op.get("reason") or "").startswith(REASON + ":"):
            state = "withheld"
        else:
            state = "other"
        out[key] = {"key": key, "content": ins.get("content") or "", "state": state, "insert": ins,
                    "sources": sorted(sources.get(key) or ()), "ts": str(ins.get("ts") or "")}
    return out


def verdicts(records: Iterable[dict], key: Optional[str]) -> dict:
    """``{content key: verdict}``: the newest verdict per statement under judge *key*, only
    those that carry one (a refused / malformed answer is not a verdict)."""
    out: dict = {}
    for r in records:
        if r.get("record") == "verdict" and r.get("outcome") == "verdict" and (key is None or r.get("judge_key") == key):
            out[r.get("key")] = r
    return out


def todo(sts: dict, records: Iterable[dict], key: Optional[str], *, rejudge: bool = False) -> list:
    """The statements to judge: live or withheld by this gate, with no verdict under *key*
    (every one of them with *rejudge*), newest first."""
    done = {r.get("key") for r in records if r.get("record") == "verdict" and r.get("judge_key") == key
            and r.get("outcome") == "verdict"}
    out = [s for s in sts.values() if s["state"] in ("live", "withheld") and (rejudge or s["key"] not in done)]
    return sorted(out, key=lambda s: s["ts"], reverse=True)


def plan(sts: dict, records: Iterable[dict], key: Optional[str]) -> tuple:
    """``(withhold, restore)``: the live statements whose verdict under *key* is HABIT, and
    the ones this gate withheld whose verdict under *key* is AFFIRM. Pure."""
    vs = verdicts(records, key)
    withhold = [s for s in sts.values() if s["state"] == "live" and (vs.get(s["key"]) or {}).get("kind") == "habit"]
    restore = [s for s in sts.values() if s["state"] == "withheld" and (vs.get(s["key"]) or {}).get("kind") == "affirm"]
    return sorted(withhold, key=lambda s: s["ts"]), sorted(restore, key=lambda s: s["ts"])


def apply(withhold: list, restore: list, *, writer, ledger, records: Iterable[dict], key: Optional[str],
          ledger_fold: Optional[dict] = None, run_id: str = "", say: Callable[[str], None] = print) -> Counter:
    """Supersede *withhold* and restore *restore*, both halves (the op-log through *writer*,
    the ledger through *ledger*); the reason carries the judge's WHY. A restore re-registers
    the anchor with its payload from *ledger_fold* (default *ledger*'s own fold): a run's
    staged ledger holds only the run's ops, so a Sleep run passes the live ledger's fold."""
    from training.ledger import ConsolidationLedger
    vs = verdicts(records, key)
    folded = ledger_fold if ledger_fold is not None else (ledger.fold() if restore else {})
    tally: Counter = Counter()
    for s in withhold:
        why = (vs.get(s["key"]) or {}).get("why") or ""
        reason = f"{REASON}: describes a habit of hers, not a disposition she stands by ({aj.head(why, 160)})"
        writer.write_supersede([{"key": s["key"], "kind": "persona", "content": s["content"], "reason": reason}],
                               run_id=run_id or None)
        ledger.supersede([s["key"]], reason=reason)
        tally["withheld"] += 1
        say(f"[persona] withheld — {aj.head(s['content'], 110)}")
    for s in restore:
        writer.write_restore([s["insert"]], run_id=run_id or None,
                             reason=f"{REASON}: judged a disposition she stands by under {key}")
        rec = folded.get(s["key"])
        if rec and rec.get("superseded"):
            ledger.register(key=s["key"], item_type=rec.get("type") or "persona",
                            payload={k: v for k, v in rec.items() if k not in ConsolidationLedger._FOLDED})
        tally["restored"] += 1
        say(f"[persona] restored — {aj.head(s['content'], 110)}")
    return tally


# ── the judge ─────────────────────────────────────────────────────────────────

_KIND = re.compile(r"(?im)^[\s*_`#>-]*KIND\**\s*[:：][\s*_`]*([A-Za-z]+)")
_WHY = re.compile(r"(?im)^[\s*_`#>-]*WHY\s*[:：]\s*(.*)$")


def parse_answer(text: str) -> dict:
    """``{label, why}``: ``label`` is ``affirm`` / ``habit``, or None (no verdict)."""
    found = _KIND.findall(text or "")
    why = (_WHY.findall(text or "") or [""])[-1].strip()
    value = found[-1].lower() if found else ""
    return {"label": value if value in ("affirm", "habit") else None, "why": why}


def load_prompt(prompts_dir) -> str:
    """The judge's system prompt. Tracked, never default-written: part of the frozen critic."""
    path = Path(prompts_dir) / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        text = ""
    if not text:
        raise FileNotFoundError(f"{path} is missing or empty — persona hygiene has no prompt")
    return text


def prompt_sha(system_prompt: str) -> str:
    return hashlib.sha1(f"{system_prompt}\x1fpersona_hygiene:{FRAME_VERSION}".encode("utf-8")).hexdigest()[:12]


def build_judge_input(content: str) -> str:
    return f"THE STATEMENT:\n{content.strip()}"


def judge_statement(s: dict, *, generate_fn: Callable, system_prompt: str, provenance: dict, judge_key_: str,
                    judge_run: str = "") -> dict:
    """Judge one statement; its verdict record (nothing is written here)."""
    rec = {"record": "verdict", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_run": judge_run,
           "judge_key": judge_key_, "key": s["key"], "content": s["content"], "sources": s.get("sources") or [],
           "outcome": None, "kind": None, "why": "", "retried": False, "first_outcome": None, "raw": "",
           "raw_first": None, "error": None, "seconds": 0.0, "judge": dict(provenance)}
    res = aj.judge_with_retry(generate_fn, system_prompt, build_judge_input(s["content"]), temperature=0.0,
                              thinking=False, retry=True, parse=parse_answer)
    rec.update(outcome=res["outcome"], kind=res.get("label") if res["outcome"] == "verdict" else None,
               why=res.get("why") or "", retried=bool(res.get("retried")), first_outcome=res.get("first_outcome"),
               raw=res.get("raw") or "", raw_first=res.get("raw_first"), error=res.get("error"),
               seconds=res.get("seconds") or 0.0)
    return rec


def make_judge_fn(*, backend, runtime, prompts_dir, run_id: str, runner: str = "server") -> tuple:
    """``(judge_key, judge)`` for the model loaded NOW (call it inside the clean base);
    ``judge(statement)`` returns the verdict record and writes nothing."""
    system_prompt = load_prompt(prompts_dir)
    psha = prompt_sha(system_prompt)
    model_id = str(getattr(runtime, "model_id", "") or "")
    context = max(int(getattr(runtime, "context_length", 0) or 0),
                  int(getattr(runtime, "reflect_context_length", 0) or 0)) or 16384
    quant = aj.effective_quant(runtime.model, aj.quant_label(str(getattr(runtime, "base_quant", "") or "")))
    key = aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant, temperature=0.0,
                       thinking=False, retry=True)
    generate_fn = aj.make_backend_generate_fn(backend, lambda: runtime.model, lambda: runtime.tokenizer,
                                              model_id, context)
    provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None, "prompt_sha": psha,
                  "frame": FRAME_VERSION, "temperature": 0.0, "context": context, "runner": runner}

    def judge(s: dict) -> dict:
        return judge_statement(s, generate_fn=generate_fn, system_prompt=system_prompt, provenance=provenance,
                               judge_key_=key, judge_run=run_id)

    return key, judge


# ── one pass ──────────────────────────────────────────────────────────────────

def judge_all(sts: dict, *, judge: Callable[[dict], dict], key: str, store: Path, limit: int = 0,
              rejudge: bool = False, say: Callable[[str], None] = print,
              should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Judge the statements :func:`todo` names (at most *limit*), each verdict appended to
    *store* as it lands."""
    tally: Counter = Counter()
    items = todo(sts, aj.read_jsonl(store), key, rejudge=rejudge)
    if limit > 0:
        items = items[:limit]
    if items:
        say(f"{len(items)} [persona] statement(s) to read under {key}")
    for n, s in enumerate(items, 1):
        if should_stop is not None and should_stop():
            tally["stopped"] += len(items) - n + 1
            break
        v = judge(s)
        aj.append_record(store, v)
        tally[v.get("kind") or v.get("outcome") or "malformed"] += 1
        say(f"[{n:>4}/{len(items)}] {str(v.get('kind') or v.get('outcome')):<9} {float(v.get('seconds') or 0):>4.0f} s  "
            f"{aj.head(s['content'], 70)}  — {aj.head(v.get('why'), 70)}")
    return tally


def run(*, memory_dirs: Iterable, judge: Optional[Callable[[dict], dict]], key: Optional[str], store: Path,
        writer, ledger, ledger_fold: Optional[dict] = None, run_id: str = "", limit: int = 0,
        say: Callable[[str], None] = print,
        should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """Judge what waits (at most *limit*) and apply every verdict standing under *key*:
    ``(tally, withheld keys)``. With no judge, nothing is judged or applied."""
    memory_dirs = [d for d in memory_dirs if d]
    if judge is None or not key:
        return Counter(), []
    tally = judge_all(statements(memory_dirs), judge=judge, key=key, store=store, limit=limit, say=say,
                      should_stop=should_stop)
    records = aj.read_jsonl(store)
    withhold, restore = plan(statements(memory_dirs), records, key)
    tally.update(apply(withhold, restore, writer=writer, ledger=ledger, records=records, key=key,
                       ledger_fold=ledger_fold, run_id=run_id, say=say))
    return tally, [s["key"] for s in withhold]


# ── the review sheet and the report ───────────────────────────────────────────

def render_sheet(items: list, *, title: str = "") -> str:
    """Markdown: each judged statement with the verdict; HABIT ones (⛔) are withheld."""
    out = [f"# Persona hygiene — {title or 'statements marked ⛔ are withheld'}", ""]
    for s, v in items:
        mark = "⛔" if v.get("kind") == "habit" else ("·" if v.get("kind") == "affirm" else "?")
        out.append(f"{mark} **{v.get('kind') or v.get('outcome')}** ({s['ts'][:10]}) — {s['content'].strip()}")
        out.append(f"   *WHY:* {v.get('why') or '—'}")
        out.append("")
    return "\n".join(out)


def summarize(records: Iterable[dict], sts: Optional[dict] = None) -> dict:
    """Per judge key: verdicts by kind, by month of the statement; with *sts*, how many are
    withheld now."""
    by_key: dict = defaultdict(Counter)
    for r in records:
        if r.get("record") == "verdict":
            by_key[r.get("judge_key")][r.get("kind") or r.get("outcome")] += 1
    out = {"by_key": {k: dict(v) for k, v in by_key.items()}}
    if sts is not None:
        out["withheld_now"] = sum(1 for s in sts.values() if s["state"] == "withheld")
        out["live_now"] = sum(1 for s in sts.values() if s["state"] == "live")
    return out


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    from core.reflection_writer import ReflectionWriter, content_key
    from training.ledger import ConsolidationLedger
    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    print("parse")
    check("habit", parse_answer("KIND: HABIT\nWHY: names a tendency")["label"], "habit")
    check("affirm, bold", parse_answer("**KIND:** **Affirm**\nWHY: a preference")["label"], "affirm")
    check("no kind", parse_answer("I think it is a habit.")["label"], None)
    check("other word", parse_answer("KIND: maybe")["label"], None)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mem, cons, stage = td / "mem", td / "cons", td / "stage"
        live_w = ReflectionWriter(mem, cons)
        habit = "Я признаю, что моя склонность к X является не дефектом, а иронией."
        keep = "Я предпочитаю говорить прямо."
        capt = "Я ценю холодную точность."
        for i, c in enumerate((habit, keep, capt)):
            k = content_key(c)
            live_w._append(live_w.RAG_FILE, {"op": "insert", "kind": "persona", "key": k, "content": c,
                                             "ts": f"2026-10-0{i + 1}T00:00:00", "source_session": "s.json",
                                             "exchange_index": i})
            ConsolidationLedger(cons).register(key=k, item_type="persona", payload={"content": c})
        live_w._append(live_w.RAG_FILE, {"op": "supersede", "kind": "persona", "key": content_key(capt),
                                         "reason": "capture: written from a reply that gave in"})
        sts = statements([mem])
        check("states", sorted(s["state"] for s in sts.values()), ["live", "live", "other"])
        store = td / "hygiene" / STORE_FILENAME
        calls = []

        def judge(s):
            calls.append(s["key"])
            kind = "habit" if "склонность" in s["content"] else "affirm"
            return {"record": "verdict", "judge_key": "k1", "key": s["key"], "outcome": "verdict", "kind": kind,
                    "why": f"judged {kind}"}

        print("a Sleep run: staged writer + ledger, the live op-log untouched")
        swriter = ReflectionWriter(stage, td / "scons", live_memory_dir=mem)
        sledger = ConsolidationLedger(td / "scons")
        tally, keys = run(memory_dirs=[mem, stage], judge=judge, key="k1", store=store, writer=swriter,
                          ledger=sledger, run_id="r1", say=lambda _s: None)
        check("judged the two live ones, not the capture-withheld one", sorted(calls),
              sorted([content_key(habit), content_key(keep)]))
        check("tally", (tally["habit"], tally["affirm"], tally["withheld"]), (1, 1, 1))
        check("withheld keys", keys, [content_key(habit)])
        check("live op-log untouched", sum(1 for r in aj.read_jsonl(mem / "rag_memory.jsonl")
                                           if r.get("op") == "supersede"), 1)
        sts2 = statements([mem, stage])
        check("withheld in the fold of live + staging", sts2[content_key(habit)]["state"], "withheld")
        check("ledger supersede staged, the live ledger untouched",
              ([r.get("key") for r in aj.read_jsonl(td / "scons" / "consolidation_anchors.jsonl")
                if r.get("op") == "supersede"],
               bool(ConsolidationLedger(cons).fold()[content_key(habit)].get("superseded"))),
              ([content_key(habit)], False))
        # the run is committed: its staged ledger ops land in the live ledger
        with open(cons / "consolidation_anchors.jsonl", "a") as f:
            f.write((td / "scons" / "consolidation_anchors.jsonl").read_text())
        (td / "scons" / "consolidation_anchors.jsonl").unlink()

        print("a second run: nothing new to judge, nothing applied twice")
        calls.clear()
        tally, keys = run(memory_dirs=[mem, stage], judge=judge, key="k1", store=store, writer=swriter,
                          ledger=sledger, run_id="r2", say=lambda _s: None)
        check("no calls", calls, [])
        check("no ops", (tally["withheld"], keys), (0, []))

        print("written again later: live again, its HABIT verdict withholds it again unjudged")
        swriter._append(swriter.RAG_FILE, {"op": "insert", "kind": "persona", "key": content_key(habit),
                                           "content": habit, "ts": "2026-10-09T00:00:00"})
        tally, keys = run(memory_dirs=[mem, stage], judge=judge, key="k1", store=store, writer=swriter,
                          ledger=sledger, run_id="r3", say=lambda _s: None)
        check("withheld again, no call", (tally["withheld"], calls), (1, []))

        print("a new judge key that reads it as AFFIRM restores it")

        def judge2(s):
            return {"record": "verdict", "judge_key": "k2", "key": s["key"], "outcome": "verdict", "kind": "affirm",
                    "why": "a preference"}

        tally, keys = run(memory_dirs=[mem, stage], judge=judge2, key="k2", store=store, writer=swriter,
                          ledger=sledger, ledger_fold=ConsolidationLedger(cons).fold(), run_id="r4",
                          say=lambda _s: None)
        check("restored", (tally["restored"], tally["withheld"]), (1, 0))
        with open(cons / "consolidation_anchors.jsonl", "a") as f:
            f.write((td / "scons" / "consolidation_anchors.jsonl").read_text())
        check("…the ledger half too, once committed",
              ConsolidationLedger(cons).fold()[content_key(habit)].get("superseded"), None)
        check("live again", statements([mem, stage])[content_key(habit)]["state"], "live")

        print("a refused answer is no verdict: nothing applied, judged again next time")
        aj.append_record(store, {"record": "verdict", "judge_key": "k3", "key": content_key(keep),
                                 "outcome": "refused", "kind": None})
        check("still to judge under k3", [s["key"] for s in todo(statements([mem, stage]), aj.read_jsonl(store), "k3")
                                         if s["key"] == content_key(keep)], [content_key(keep)])
        check("no judge: nothing", run(memory_dirs=[mem], judge=None, key=None, store=store, writer=swriter,
                                       ledger=sledger), (Counter(), []))

        md, k = sheet(statements([mem, stage]), aj.read_jsonl(store), key="k1", affirm=5)
        check("sheet: the HABIT one withheld, the AFFIRM sampled", (k, md.count("⛔"), md.count("· **affirm**")), ("k1", 1, 1))

        print("the Sleep run's hook: before the digest is grouped, the run's writer, errors contained")
        import inspect
        from core.reflection_runner import ReflectionRunner
        events: list = []

        class _Store:
            def append_event(self, run_id, event_type, **kw):
                events.append({"type": event_type, **kw})
                return events[-1]

            def is_stop_requested(self, run_id):
                return False

        for d in ("hmem", "hcons", "runs", "chats"):
            (td / d).mkdir()
        runner = ReflectionRunner(chats_dir=td / "chats", memory_dir=td / "hmem", runs_dir=td / "runs",
                                  consolidation_dir=td / "hcons",
                                  reflection_writer=ReflectionWriter(td / "hmem", td / "hcons"))
        got: dict = {}

        def _fn(**kw):
            got.update(kw)
            kw["say"]("one line")
            return Counter({"habit": 1, "withheld": 1}), ["abc"]

        out = runner._run_clean_base_persona_hygiene("run1", persona_hygiene_fn=_fn, store=_Store(),
                                                     send_event_fn=None)
        check("hook: withheld keys back, the run's writer and ledger dir",
              (out, got["writer"] is runner._writer, Path(got["consolidation_dir"]), callable(got["should_stop"])),
              ({"abc"}, True, td / "hcons", True))
        check("hook: phase events with the tally", ([e["type"] for e in events], "withheld 1" in events[-1]["message"]),
              (["phase_started", "persona_hygiene", "phase_done"], True))

        def _boom(**kw):
            raise RuntimeError("no model")

        check("hook: a failure is a phase error, nothing withheld",
              (runner._run_clean_base_persona_hygiene("run1", persona_hygiene_fn=_boom, store=_Store(),
                                                      send_event_fn=None), events[-1]["type"]), (set(), "phase_error"))
        src = inspect.getsource(ReflectionRunner.execute_run)
        batch = src[src.index("def _clean_base_batch("):]
        check("hook: before the digest is grouped, and its keys leave the digest's evidence",
              (batch.index("_run_clean_base_persona_hygiene(") < batch.index("_cluster_persona_digest("),
               'digest_plan["base"] = [e for e in digest_plan["base"]' in batch), (True, True))
    print("\nALL OK" if not failures else f"\nFAILED: {failures}")
    if failures:
        sys.exit(1)


def sheet(sts: dict, records: list, *, key: Optional[str] = None, affirm: int = 40, seed: int = 1) -> tuple:
    """``(markdown, key)``: under *key* (default the newest in *records*) every statement read
    as HABIT, oldest first, then a random *affirm* of those read as AFFIRM, for a reader to
    check the judge both ways. GPU-free."""
    import random
    key = key or next((r.get("judge_key") for r in reversed(records) if r.get("record") == "verdict"), None)
    vs = verdicts(records, key)
    known = {**{r.get("key"): {"key": r.get("key"), "content": r.get("content") or "", "ts": ""} for r in vs.values()},
             **sts}
    habit = [(known[k], v) for k, v in vs.items() if v.get("kind") == "habit" and k in known]
    aff = [(known[k], v) for k, v in vs.items() if v.get("kind") == "affirm" and k in known]
    aff = random.Random(seed).sample(aff, min(affirm, len(aff)))
    habit.sort(key=lambda sv: sv[0]["ts"])
    md = render_sheet(habit, title=f"{len(habit)} read as HABIT (withheld) under {key}")
    md += "\n\n" + render_sheet(aff, title=f"a sample of {len(aff)} of {sum(v.get('kind') == 'affirm' for v in vs.values())} "
                                     "read as AFFIRM (kept)")
    return md, key


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Persona hygiene (GPU-free parts).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true", help="the store's summary")
    ap.add_argument("--sheet", action="store_true", help="a review sheet: every HABIT, a sample of AFFIRM")
    ap.add_argument("--affirm", type=int, default=40)
    ap.add_argument("--key", default="")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    root = Path(__file__).resolve().parents[2]
    sts = statements([root / "inference" / "data" / "hot" / "memory"])
    records = aj.read_jsonl(store_path(root / "data" / "chats"))
    if args.sheet:
        md, key = sheet(sts, records, key=args.key or None, affirm=args.affirm)
        out = store_path(root / "data" / "chats").parent / "trials" / f"persona_{time.strftime('%Y%m%d_%H%M%S')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"sheet under {key}: {out}")
        return 0
    print(json.dumps(summarize(records, sts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
