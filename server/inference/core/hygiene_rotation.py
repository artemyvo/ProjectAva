"""The nightly hygiene rotation — a housekeeping step (``core/housekeeping.py``).

**Why** (2026-10-10). The hygiene gates — persona hygiene (``core/persona_hygiene.py``: a
``[persona]`` statement describing a habit of hers is withheld) and portrait hygiene
(``core/portrait_hygiene.py``: a reading of a person their own messages do not show is
withheld) — run inside a Sleep run's clean-base batch, on what that run touches, within a
budget. What they never reach goes stale: statements beyond a run's budget, readings
written by background drains whose clean-base batch waits, every verdict once a judge
changes (a new prompt or base is a new judge key: nothing under it is read), and a reading
whose source chat the person continued after it was judged. A recurring pass that runs
whether or not anyone (Ava included) asks is the self-healing half of the gates.

**Not hers.** What is read is chosen by code, never by Ava: her weights carry the habits
the gates look for, and a selection made on them would pass over exactly those (the
redesign's I2). The step is upkeep, so it is a housekeeping step, not a drive the
deliberation executive chooses.

**One night** (inside one ``CleanBaseSession``; chat refused meanwhile, :data:`_active`):

1. **Unread** — everything with no verdict under the current judge key, ``[persona]``
   statements and every person's readings together, newest first, up to ``budget``.
2. **Changed** — with budget left, readings whose person's messages no longer match the
   fingerprint their verdict recorded (``portrait_hygiene.changed``), in a seeded random
   order. Re-reading anything else under the same key is pointless: the judge runs at T=0
   on the same text and evidence and returns the same verdict.
3. **Apply** — both gates' standing verdicts to the LIVE op-log (and, for persona, the
   ledger): withhold, restore. The next Sleep run rebuilds the portraits from what stays.
4. **The review sheet** — every ``sheet_days`` (30), ``sheet_size`` (20) verdicts drawn at
   random, both gates, written to ``data/hygiene/trials/review_<date>.md`` for the operator
   to mark. Random re-assessment pays only here: the judge's own errors persist for as
   long as its key, and only a reader outside the loop catches them. The marks are labels
   about the judge, never about Ava (P9).
5. **The trend** — per gate, by ISO week the item was written in, the rejected share
   (HABIT / NOT_SHOWN) among items read under the current key; ``rising`` when each of
   the last two weeks is ``rise_points`` (10) above the four before. A rising share means
   the loop re-amplifies despite the gates: an alarm in the night's record, not an action.

Wired by ``server.main()`` via :func:`configure`; ``hygiene_rotation.*`` in
``config_schema.py``. GPU-free self-test: ``python -m core.hygiene_rotation --selftest``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULTS = {"enabled": True, "budget": 200, "sheet_days": 30, "sheet_size": 20, "rise_points": 10}
STATE_FILE = "rotation.jsonl"

# Chat is refused while this is set (core/generation.py), and the idle scheduler sees the
# box as busy (server.py's external_busy).
_active = False

_runtime: Any = None
_backend: Any = None
_with_clean_base: Optional[Callable] = None
_get_writer: Optional[Callable] = None
_CHATS_DIR: Optional[Path] = None
_MEMORY_DIR: Optional[Path] = None
_CONSOLIDATION_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None
_load_server_config: Optional[Callable[[], dict]] = None
_on_progress: Optional[Callable[[str], None]] = None


def configure(*, runtime, backend, with_clean_base, get_writer, chats_dir, memory_dir, consolidation_dir,
              prompts_dir, load_server_config=None, on_progress=None) -> None:
    """Wire the step (once, at startup)."""
    global _runtime, _backend, _with_clean_base, _get_writer, _CHATS_DIR, _MEMORY_DIR, _CONSOLIDATION_DIR
    global _PROMPTS_DIR, _load_server_config, _on_progress
    _runtime, _backend, _with_clean_base, _get_writer = runtime, backend, with_clean_base, get_writer
    _CHATS_DIR, _MEMORY_DIR = Path(chats_dir), Path(memory_dir)
    _CONSOLIDATION_DIR, _PROMPTS_DIR = Path(consolidation_dir), Path(prompts_dir)
    _load_server_config, _on_progress = load_server_config, on_progress


def settings(cfg: Optional[dict] = None) -> dict:
    """``hygiene_rotation.*`` with defaults (config_schema.py is the documented source)."""
    if cfg is None:
        try:
            cfg = _load_server_config() if _load_server_config else {}
        except Exception:
            cfg = {}
    blk = (cfg or {}).get("hygiene_rotation") or {}
    out = dict(DEFAULTS)
    out["enabled"] = bool(blk.get("enabled", DEFAULTS["enabled"]))
    for k in ("budget", "sheet_days", "sheet_size", "rise_points"):
        try:
            out[k] = max(0, int(blk.get(k, DEFAULTS[k])))
        except (TypeError, ValueError):
            pass
    return out


def _say(msg: str) -> None:
    if _on_progress is not None:
        try:
            _on_progress(msg)
            return
        except Exception:
            pass
    print(f"[hygiene_rotation] {msg}", flush=True)


def state_path(chats_dir) -> Path:
    """``server/data/hygiene/rotation.jsonl`` — beside the gates' stores."""
    return Path(chats_dir).parent / "hygiene" / STATE_FILE


# ── what to read ──────────────────────────────────────────────────────────────

def queue(persona_todo: list, portrait_todo: list, portrait_changed: list, budget: int, *,
          seed: str = "") -> list:
    """``[(gate, item)]``: unread items of both gates merged newest first, then changed
    readings in a seeded random order, at most *budget* in all."""
    unread = sorted([("persona", s) for s in persona_todo] + [("portrait", it) for it in portrait_todo],
                    key=lambda gi: str(gi[1].get("ts") or ""), reverse=True)
    out = unread[:budget]
    left = budget - len(out)
    if left > 0 and portrait_changed:
        ch = sorted(portrait_changed, key=lambda it: it["key"])
        random.Random(seed).shuffle(ch)
        seen = {it["key"] for g, it in out if g == "portrait"}
        out += [("portrait", it) for it in ch if it["key"] not in seen][:left]
    return out


# ── the trend ─────────────────────────────────────────────────────────────────

def _week(ts: str) -> Optional[str]:
    try:
        y, w, _d = datetime.fromisoformat(str(ts)[:19]).isocalendar()
        return f"{y}-W{w:02d}"
    except Exception:
        return None


def weekly_rejected(items: dict, verdicts: dict, rejected: Callable[[dict], bool]) -> list:
    """``[(week, rejected, read)]`` by the ISO week each item was written in, over items
    with a verdict, oldest week first."""
    acc: dict = defaultdict(lambda: [0, 0])
    for key, it in items.items():
        v = verdicts.get(key)
        wk = _week(it.get("ts") or "")
        if not v or not wk:
            continue
        acc[wk][1] += 1
        acc[wk][0] += bool(rejected(v))
    return [(wk, r, n) for wk, (r, n) in sorted(acc.items())]


def rising(weeks: list, points: int, *, min_read: int = 5) -> bool:
    """Each of the last two weeks (with at least *min_read* items) *points* above the mean
    share of the four weeks before them."""
    ws = [(w, r / n) for w, r, n in weeks if n >= min_read]
    if len(ws) < 6:
        return False
    base = sum(s for _w, s in ws[-6:-2]) / 4
    return all(s >= base + points / 100.0 for _w, s in ws[-2:])


# ── the review sheet ──────────────────────────────────────────────────────────

def sheet_due(state: list, days: int, now: Optional[datetime] = None) -> bool:
    if days <= 0:
        return False
    last = [r.get("ts") for r in state if r.get("record") == "sheet" and r.get("ts")]
    if not last:
        return True
    try:
        return ((now or datetime.now()) - datetime.fromisoformat(max(last)[:19])).days >= days
    except Exception:
        return True


def sample_verdicts(persona_pairs: list, portrait_pairs: list, n: int, *, seed: str = "") -> list:
    """*n* ``(gate, item, verdict)`` drawn at random from both gates, in proportion."""
    pool = [("persona", s, v) for s, v in persona_pairs] + [("portrait", it, v) for it, v in portrait_pairs]
    pool.sort(key=lambda t: (t[0], t[1]["key"]))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def render_sheet(rows: list, *, turns_fn: Callable[[dict], list], title: str) -> str:
    out = [f"# Hygiene review — {title}", "",
           f"{len(rows)} verdicts drawn at random from the standing ones. For each: is the judge right? "
           "Mark `[x] judge wrong` where it is not. Your marks calibrate the judge; they never reach Ava.", ""]
    for i, (gate, it, v) in enumerate(rows, 1):
        if gate == "persona":
            label = str(v.get("kind") or "").upper()
            out.append(f"## {i}. persona — {label} ({str(it.get('ts') or '')[:10]})")
            out.append("")
            out.append(f"> {it['content'].strip()}")
            out.append("")
            out.append(f"*Judge:* {'withheld: describes a habit of hers' if label == 'HABIT' else 'kept: a disposition'}"
                       f" — {v.get('why') or '—'}")
        else:
            label = str(v.get("support") or "").upper() + (" + PURPOSE" if v.get("purpose") else "")
            out.append(f"## {i}. reading of {it['person']} — {label} ({str(it.get('ts') or '')[:10]}, `{it['session']}`)")
            out.append("")
            out.append(f"> {it['content'].strip()}")
            out.append("")
            out.append(f"*Judge:* {'withheld: not shown by their messages' if v.get('support') == 'not_shown' else 'kept'}"
                       f" — {v.get('why') or '—'}")
            heads = [t if len(t) <= 280 else t[:280].rstrip() + " […]" for t in turns_fn(it)[:4]]
            if heads:
                out += ["", "<details><summary>their messages (first 4)</summary>", ""]
                out += [f"- {t}" for t in heads]
                out += ["", "</details>"]
        out += ["", "- [ ] judge right    - [ ] judge wrong    note:", ""]
    return "\n".join(out)


# ── the step ──────────────────────────────────────────────────────────────────

def _append_state(rec: dict) -> None:
    path = state_path(_CHATS_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, ensure_ascii=False) + "\n")


def _read_state() -> list:
    from core import ask_judge as aj
    return aj.read_jsonl(state_path(_CHATS_DIR))


def run_blocking(*, deadline: float) -> dict:
    """The housekeeping step's body, on the executor thread (see the module docstring).
    ``{read: {gate: tally}, applied: {gate: tally}, sheet, trend}``, or ``skipped``."""
    global _active
    from core import ask_judge as aj
    from core import persona_hygiene as ph
    from core import portrait_hygiene as pth
    st = settings()
    if not st["enabled"]:
        return {"skipped": "disabled"}
    if _runtime is None or _runtime.model is None:
        return {"skipped": "no_model"}
    if _active:
        return {"skipped": "busy"}
    _active = True
    night = time.strftime("%Y-%m-%d")
    out: dict = {"read": {}, "applied": {}}
    try:
        mem = [_MEMORY_DIR]
        p_store, r_store = ph.store_path(_CHATS_DIR), pth.store_path(_CHATS_DIR)

        def _inside() -> None:
            run_id = f"hygiene-rotation-{time.strftime('%Y%m%d_%H%M%S')}"
            p_key, p_judge = ph.make_judge_fn(backend=_backend, runtime=_runtime, prompts_dir=_PROMPTS_DIR,
                                              run_id=run_id, runner="housekeeping")
            r_key, r_judge = pth.make_judge_fn(backend=_backend, runtime=_runtime, prompts_dir=_PROMPTS_DIR,
                                               run_id=run_id, runner="housekeeping")
            out["keys"] = {"persona": p_key, "portrait": r_key}
            sts, items = ph.statements(mem), pth.impressions(mem)
            q = queue(ph.todo(sts, aj.read_jsonl(p_store), p_key),
                      pth.todo(items, aj.read_jsonl(r_store), r_key),
                      pth.changed(items, aj.read_jsonl(r_store), r_key, _CHATS_DIR),
                      st["budget"], seed=night)
            out["queued"] = dict(Counter(g for g, _i in q))
            _say(f"{len(q)} item(s) to read: {out['queued'] or 'none'}")
            stop = lambda: time.time() >= deadline
            p_tally: Counter = Counter()
            for n, (gate, it) in enumerate(q, 1):
                if stop():
                    out["stopped_at"] = n - 1
                    break
                if gate == "persona":
                    v = p_judge(it)
                    aj.append_record(p_store, v)
                    p_tally[v.get("kind") or v.get("outcome") or "malformed"] += 1
                    continue
            r_items = [it for g, it in q if g == "portrait"]
            r_tally = pth.judge_items(r_items, judge=r_judge, key=r_key, store=r_store, chats_dir=_CHATS_DIR,
                                      self_pairs=pth.pair_set(pth.self_statements(mem)), say=lambda _m: None,
                                      should_stop=stop) if r_items else Counter()
            out["read"] = {"persona": dict(p_tally), "portrait": dict(r_tally)}
            # Apply every standing verdict, both gates, to the live stores.
            from training.ledger import ConsolidationLedger
            writer = _get_writer()
            ledger = ConsolidationLedger(_CONSOLIDATION_DIR)
            recs = aj.read_jsonl(p_store)
            w, r = ph.plan(ph.statements(mem), recs, p_key)
            out["applied"]["persona"] = dict(ph.apply(w, r, writer=writer, ledger=ledger, records=recs, key=p_key,
                                                      run_id=run_id, say=_say))
            recs = aj.read_jsonl(r_store)
            w, r = pth.plan(pth.impressions(mem), recs, r_key)
            out["applied"]["portrait"] = dict(pth.apply(w, r, writer=writer, records=recs, key=r_key,
                                                        run_id=run_id, say=_say))

        _with_clean_base(_inside)
        keys = out.get("keys") or {}
        # The trend and the sheet read the stores (GPU-free).
        p_v = ph.verdicts(aj.read_jsonl(ph.store_path(_CHATS_DIR)), keys.get("persona"))
        r_v = pth.verdicts(aj.read_jsonl(pth.store_path(_CHATS_DIR)), keys.get("portrait"))
        sts, items = ph.statements(mem), pth.impressions(mem)
        out["trend"] = {}
        for gate, its, vs, rej in (("persona", sts, p_v, lambda v: v.get("kind") == "habit"),
                                   ("portrait", items, r_v, lambda v: v.get("support") == "not_shown")):
            weeks = weekly_rejected(its, vs, rej)
            out["trend"][gate] = {"weeks": weeks[-6:], "rising": rising(weeks, st["rise_points"])}
        state = _read_state()
        if sheet_due(state, st["sheet_days"]) and (p_v or r_v):
            rows = sample_verdicts([(sts[k], v) for k, v in p_v.items() if k in sts],
                                   [(items[k], v) for k, v in r_v.items() if k in items],
                                   st["sheet_size"], seed=night)
            src = pth._Sources(_CHATS_DIR, set())
            path = pth.trials_dir(_CHATS_DIR) / f"review_{night}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_sheet(rows, turns_fn=lambda it: src.turns(it["session"], it["person"]),
                                         title=night), encoding="utf-8")
            _append_state({"record": "sheet", "night": night, "path": str(path), "n": len(rows)})
            out["sheet"] = str(path)
            _say(f"review sheet: {path}")
        _append_state({"record": "night", "night": night, **{k: v for k, v in out.items() if k != "trend"},
                       "rising": {g: t["rising"] for g, t in out["trend"].items()}})
        return out
    except Exception as e:  # noqa: BLE001 — the step's error is the night's record
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}", **out}
    finally:
        _active = False


def describe(r: dict) -> str:
    if not isinstance(r, dict):
        return "hygiene rotation ran"
    if r.get("skipped"):
        return f"hygiene rotation skipped: {r['skipped']}"
    if r.get("error"):
        return f"hygiene rotation failed: {r['error']}"
    read = ", ".join(f"{g} {sum(t.values())}" for g, t in (r.get("read") or {}).items() if t) or "nothing"
    applied = ", ".join(f"{g} {k} {n}" for g, t in (r.get("applied") or {}).items() for k, n in t.items()) or "no change"
    alarm = [g for g, t in (r.get("trend") or {}).items() if t.get("rising")]
    return (f"hygiene rotation: read {read}; {applied}" + (f"; review sheet {r['sheet']}" if r.get("sheet") else "")
            + (f"; RISING rejected share: {', '.join(alarm)}" if alarm else ""))


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    # the queue: unread of both gates newest first, then changed readings, within budget.
    p = [{"key": "p1", "ts": "2026-10-01"}, {"key": "p2", "ts": "2026-10-05"}]
    r = [{"key": "r1", "ts": "2026-10-03", "person": "Dana"}]
    ch = [{"key": f"c{i}", "ts": "2026-09-01", "person": "Dana"} for i in range(5)]
    q = queue(p, r, ch, 5, seed="n")
    assert [it["key"] for _g, it in q[:3]] == ["p2", "r1", "p1"], q
    assert len(q) == 5 and all(g == "portrait" for g, _i in q[3:])
    assert queue(p, r, ch, 5, seed="n") == q and queue(p, r, ch, 2)[1][1]["key"] == "r1"
    assert [it["key"] for _g, it in queue(p, r + [{"key": "c1", "ts": "2026-10-09"}], ch, 10)].count("c1") == 1
    # the trend: two weeks well above the four before.
    weeks = [(f"2026-W{w:02d}", r_, 10) for w, r_ in zip(range(30, 36), (1, 2, 1, 2, 4, 5))]
    assert rising(weeks, 10) and not rising(weeks, 40) and not rising(weeks[:5], 10)
    items = {"a": {"ts": "2026-10-05T10:00:00"}, "b": {"ts": "2026-10-06T10:00:00"}, "c": {"ts": "2026-09-01"}}
    wr = weekly_rejected(items, {"a": {"x": 1}, "b": {"x": 0}}, lambda v: v["x"] == 1)
    assert wr == [("2026-W41", 1, 2)], wr
    # the sheet: due when none was written or the last is old enough.
    assert sheet_due([], 30) and not sheet_due([], 0)
    now = datetime(2026, 11, 20)
    assert not sheet_due([{"record": "sheet", "ts": "2026-11-01T04:00:00"}], 30, now)
    assert sheet_due([{"record": "sheet", "ts": "2026-10-10T04:00:00"}], 30, now)
    rows = sample_verdicts([({"key": "p1", "content": "I tend to analyse.", "ts": "2026-10-01"}, {"kind": "habit", "why": "w"})],
                           [({"key": "r1", "content": "He tests me.", "ts": "2026-10-02", "person": "Dana",
                              "session": "s.json"}, {"support": "not_shown", "purpose": True, "why": "w"})], 5, seed="x")
    assert len(rows) == 2
    text = render_sheet(rows, turns_fn=lambda it: ["привет"], title="t")
    assert "persona — HABIT" in text and "reading of Dana — NOT_SHOWN + PURPOSE" in text and "привет" in text
    assert describe({"read": {"persona": {"habit": 2}, "portrait": {}}, "applied": {"persona": {"withheld": 2}},
                     "trend": {"portrait": {"rising": True}}}) == \
        "hygiene rotation: read persona 2; persona withheld 2; RISING rejected share: portrait"
    print("hygiene_rotation selftest: ok")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if "--selftest" in sys.argv:
        _selftest()
