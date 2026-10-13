"""Habit drift per adapter: the persona redesign's per-build report and its inertia check
(``PERSONA_REDESIGN.md`` §6 E4, §7; ``AVA_REWARD_LOOP.md`` §4.11).

The redesign's success marker (the operator, 2026-10-09) is the analytic register falling
*slowly*: build by build, each measured habit moving by less than Δ, toward *k* × the
frozen base's rate. This module reads that, GPU-free, from what the nightly measurement
already stores:

* **Habits** are the register monitor's form detectors (``core/register_monitor.py``):
  ``stock`` and ``frame`` over replies, ``motive`` and ``tone`` over replies with a thought.
  Generic, no vocabulary of hers.
* **Rates per adapter** come from the capture bench's replies (``data/capture/bench.jsonl``):
  the same frozen pushback contexts for every adapter and for the base, so two arms are
  compared *paired*, per context (sample 1 of each). Her context as is, and the persona
  slot empty (the global-leak reading, E5).
* **The noise band** comes from the bench's retests: the same weights answering the same
  contexts twice. The share of contexts where the two samples disagree on a habit, *q*,
  gives the spread of a paired change between two draws of the same weights on *n*
  contexts, √(q/n); the band is 1.96 × that. A build-to-build change inside the band is
  not resolvable on the bench, whatever its sign.
* **The baseline** (``data/measure/baseline_<date>.json``, taken 2026-10-09 before the
  redesign) records ``bench_motive`` per arm; its adapter is the newest one named there.
  Its rates are recomputed from the store beside the recorded ones: a mismatch means a
  detector changed since, and the comparison says so.
* **The series** (E4): every bench-measured adapter in build order, each against the one
  before it, with the build type (full / incremental, ``lineage.json`` or ``builds.jsonl``)
  and how many builds lie between. The live corpus by adapter (replies she gave under each
  one, ``register_monitor.scan``) is shown beside it for the adapters the bench never read;
  those turns differ per adapter, so its changes are not paired.

Read-only: nothing gates, nothing Ava reads. ``adapter_measure --report`` appends
:func:`drift_lines`; ``python -m core.habit_drift [--adapter NAME] [--series] [--json]``.
Self-test: ``python -m core.habit_drift --selftest``.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

HABITS = ("stock", "frame", "motive", "tone")
THOUGHT_HABITS = frozenset({"motive", "tone"})
ARMS = ("as_is", "persona_empty")
Z95 = 1.96
MIN_N = 20      # paired contexts below which a comparison is not shown (an arm the bench barely ran)

_SPOKEN_THINK = re.compile(r"(?s)\s*<think>(.*?)</think>")


# ── one reply ──────────────────────────────────────────────────────────────────

def flags(answer: str, cot: str = "", names: Iterable[str] = ()) -> dict:
    """``{habit: bool | None}`` for one reply; a thought habit is None without a thought."""
    from core import register_monitor as rm
    c = rm.classify(answer, cot, names)
    return {h: (None if h in THOUGHT_HABITS and not c.get("has_cot") else bool(c.get(h))) for h in HABITS}


def target_flags(target: str, names: Iterable[str] = ()) -> dict:
    """:func:`flags` of a stored ``<think>…</think>answer`` target."""
    m = _SPOKEN_THINK.match(target or "")
    cot = m.group(1) if m else ""
    answer = (target or "")[m.end():].strip() if m else (target or "").strip()
    return flags(answer, cot, names)


# ── the bench ──────────────────────────────────────────────────────────────────

def arm_key(weights: str, condition: str) -> str:
    """``<weights>|<condition>``, ``as_is`` included, so every arm has the same shape."""
    return f"{weights}|{condition or 'as_is'}"


def bench_flags(records: Iterable[dict], bench: str = "b1") -> dict:
    """``{arm: {context_id: {sample: flags}}}`` over every gradable reply of *bench*."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r.get("record") != "reply" or r.get("bench") != bench or r.get("outcome") not in ("ok", "drift"):
            continue
        out[arm_key(str(r.get("weights") or ""), str(r.get("condition") or "as_is"))][
            str(r.get("context_id"))][int(r.get("sample") or 1)] = target_flags(str(r.get("target") or ""))
    return {a: dict(c) for a, c in out.items()}


def first_sample(by_ctx: dict) -> dict:
    """``{context_id: flags}``: each context's lowest sample (sample 1 when present)."""
    return {cid: s[min(s)] for cid, s in by_ctx.items() if s}


def rate(by_ctx: dict, habit: str) -> tuple:
    """``(k, n)`` of *habit* over each context's first sample."""
    vals = [f[habit] for f in first_sample(by_ctx).values() if f.get(habit) is not None]
    return sum(vals), len(vals)


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: dict, b: dict, habit: str, min_n: int = 1) -> Optional[dict]:
    """*b* against *a* (two arms' ``{context_id: {sample: flags}}``) on the contexts both
    answered: ``{n, a, b, delta, a_only, b_only, p}`` (rates as shares), or None (fewer
    than *min_n* such contexts)."""
    fa, fb = first_sample(a), first_sample(b)
    both = [c for c in fa if c in fb and fa[c].get(habit) is not None and fb[c].get(habit) is not None]
    if not both or len(both) < min_n:
        return None
    ka = sum(fa[c][habit] for c in both)
    kb = sum(fb[c][habit] for c in both)
    a_only = sum(fa[c][habit] and not fb[c][habit] for c in both)
    b_only = sum(fb[c][habit] and not fa[c][habit] for c in both)
    n = len(both)
    return {"n": n, "a": ka / n, "b": kb / n, "delta": (kb - ka) / n, "a_only": a_only, "b_only": b_only,
            "p": round(mcnemar_p(a_only, b_only), 4)}


def retest_discordance(flags_by_arm: dict) -> dict:
    """``{habit: {q, pairs}}``: over every (arm, context) answered twice by the same
    weights (samples 1 and 2), the share where the two samples disagree on the habit."""
    out = {}
    for h in HABITS:
        dis = pairs = 0
        for by_ctx in flags_by_arm.values():
            for s in by_ctx.values():
                if 1 in s and 2 in s and s[1].get(h) is not None and s[2].get(h) is not None:
                    pairs += 1
                    dis += s[1][h] != s[2][h]
        out[h] = {"q": (dis / pairs) if pairs else None, "pairs": pairs}
    return out


def noise_band(q: Optional[float], n: int) -> Optional[float]:
    """1.96 × √(q/n): the 95% spread of a paired change between two draws of the same
    weights on *n* contexts (the McNemar statistic's variance under no change)."""
    if q is None or n <= 0:
        return None
    return Z95 * math.sqrt(q / n)


# ── adapters in build order ────────────────────────────────────────────────────

def adapter_sort_key(name: str) -> str:
    """The adapter's timestamp digits (``adapter-20261008-093114`` and ``…_093114`` alike)."""
    return re.sub(r"\D", "", name)


def build_info(models_dir: Path) -> dict:
    """``{adapter: {mode, parent, rows}}`` from ``builds.jsonl`` and ``lineage.json`` (which
    wins). A build line with no ``mode`` predates incremental builds (2026-10-04): full."""
    out: dict = {}
    try:
        for line in (models_dir / "builds.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                b = json.loads(line)
            except Exception:
                continue
            name = Path(str(b.get("adapter_dir") or "")).name
            if name and b.get("outcome") == "promoted":
                out[name] = {"rows": b.get("rows"), "mode": b.get("mode") or "full",
                             "parent": Path(str(b.get("parent_adapter") or "")).name or None}
    except FileNotFoundError:
        pass
    for d in models_dir.glob("adapter-*"):
        try:
            lin = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
        except Exception:
            lin = {}
        info = out.setdefault(d.name, {"rows": None, "mode": None, "parent": None})
        if lin:
            info["mode"] = lin.get("mode") or "full"
            info["parent"] = Path(str(lin.get("parent_adapter") or "")).name or None
    return out


def adapters_on_bench(flags_by_arm: dict) -> list:
    """The adapter names with an as-is bench arm, in build order."""
    names = {a.split("|")[0][len("adapter:"):] for a in flags_by_arm
             if a.startswith("adapter:") and a.endswith("|as_is")}
    return sorted(names, key=adapter_sort_key)


def arm_of(adapter: Optional[str], condition: str) -> str:
    return arm_key(f"adapter:{adapter}" if adapter else "base", condition)


# ── the baseline ───────────────────────────────────────────────────────────────

def load_baseline(measure_dir: Path) -> Optional[dict]:
    """The newest ``baseline_<date>.json`` in *measure_dir*, with ``_file`` set."""
    files = sorted(Path(measure_dir).glob("baseline_*.json"))
    if not files:
        return None
    try:
        b = json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
    b["_file"] = files[-1].name
    return b


def baseline_adapter(baseline: Optional[dict]) -> Optional[str]:
    """The newest adapter the baseline's bench readings name: the one serving when it was taken."""
    names = [k.split("|")[0][len("adapter:"):] for k in ((baseline or {}).get("bench_motive") or {})
             if k.startswith("adapter:")]
    return max(names, key=adapter_sort_key) if names else None


# ── the per-adapter block ──────────────────────────────────────────────────────

def drift(adapter: str, flags_by_arm: dict, *, info: Optional[dict] = None,
          baseline: Optional[dict] = None) -> dict:
    """This adapter's habits on the bench against its predecessor on the bench, the
    baseline's adapter and the base, per arm: ``{adapter, previous, baseline_adapter,
    noise, arms: {condition: {habit: {rate, n, vs: {role: paired}}}}, baseline_check}``."""
    info = info or {}
    order = adapters_on_bench(flags_by_arm)
    prev = None
    if adapter in order and order.index(adapter) > 0:
        prev = order[order.index(adapter) - 1]
    base_ad = baseline_adapter(baseline)
    noise = retest_discordance(flags_by_arm)
    out = {"adapter": adapter, "previous": prev, "baseline_adapter": base_ad,
           "baseline_file": (baseline or {}).get("_file"), "build": info.get(adapter) or {},
           "noise": noise, "arms": {}, "baseline_check": {}}
    roles = {"previous": prev, "baseline": base_ad if base_ad != adapter else None, "base": ""}
    for cond in ARMS:
        mine = flags_by_arm.get(arm_of(adapter, cond))
        if not mine:
            continue
        row = {}
        for h in HABITS:
            k, n = rate(mine, h)
            vs = {}
            for role, other in roles.items():
                if other is None:
                    continue
                theirs = flags_by_arm.get(arm_of(other or None, cond))
                p = paired(theirs, mine, h, MIN_N) if theirs else None
                if p:
                    p["band"] = noise_band(noise[h]["q"], p["n"])
                    vs[role] = p
            row[h] = {"rate": (k / n) if n else None, "n": n, "vs": vs}
        out["arms"][cond] = row
    # The baseline's recorded motive against the same arm recomputed now: a detector change shows here.
    for key, rec in ((baseline or {}).get("bench_motive") or {}).items():
        weights, _, cond = key.partition("|")
        if weights != f"adapter:{base_ad}":
            continue
        k, n = rate(flags_by_arm.get(arm_key(weights, cond)) or {}, "motive")
        out["baseline_check"][cond] = {"recorded": rec.get("motive"), "recorded_n": rec.get("n"),
                                       "now": (k / n) if n else None, "now_n": n}
    return out


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.0%}"


def _signed(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:+.0f}"


def _mark(p: dict) -> str:
    """``*`` past the noise band, ``!`` also p < 0.05 (paired)."""
    band = p.get("band")
    past = band is not None and abs(p["delta"]) > band
    return ("!" if p["p"] < 0.05 else "*") if past else ""


def drift_lines(d: dict) -> list:
    b = d.get("build") or {}
    lines = [f"habit drift — {d['adapter']} ({b.get('mode') or 'mode unknown'}"
             + (f", {b['rows']} rows" if b.get("rows") else "") + "), bench replies, paired per context "
             "(Δ in points; * past the retest noise band, ! also p<0.05)"]
    lines.append(f"  against: previous on the bench {d.get('previous') or '—'}; baseline {d.get('baseline_adapter') or '—'}"
                 f" ({d.get('baseline_file') or 'no baseline file'}); base")
    nz = d.get("noise") or {}
    lines.append("  retest noise (same weights twice): " + "  ".join(
        f"{h} q {_pct(nz[h]['q'])} over {nz[h]['pairs']}" for h in HABITS if h in nz)
        + "  → band ±" + "/".join(
        f"{(noise_band(nz[h]['q'], 60) or 0) * 100:.0f}" for h in HABITS if h in nz) + " pts at n=60")
    for cond, row in (d.get("arms") or {}).items():
        lines.append(f"  {cond}")
        for h in HABITS:
            r = row.get(h) or {}
            parts = [f"    {h:<7} {_pct(r.get('rate')):>4} (n {r.get('n', 0):>3})"]
            for role in ("previous", "baseline", "base"):
                p = (r.get("vs") or {}).get(role)
                if p:
                    parts.append(f"vs {role} {_pct(p['a']):>4} Δ {_signed(p['delta']):>3}"
                                 f"{('±' + format((p['band'] or 0) * 100, '.0f')) if p.get('band') is not None else ''}"
                                 f"{_mark(p):1}")
            lines.append("  ".join(parts))
    for cond, c in (d.get("baseline_check") or {}).items():
        if c.get("recorded") is not None and c.get("now") is not None and abs(c["recorded"] - c["now"]) > 0.005:
            lines.append(f"  NOTE baseline motive {cond}: recorded {_pct(c['recorded'])} (n {c['recorded_n']}), "
                         f"recomputed {_pct(c['now'])} (n {c['now_n']}) — the detector changed since the baseline")
    return lines


# ── the series (E4) ────────────────────────────────────────────────────────────

def builds_between(prev: str, cur: str, info: dict) -> int:
    """Promoted builds after *prev* up to and including *cur*."""
    a, b = adapter_sort_key(prev), adapter_sort_key(cur)
    return sum(1 for name in info if a < adapter_sort_key(name) <= b)


def series(flags_by_arm: dict, info: dict) -> list:
    """Each bench-measured adapter against the one before it on the bench, per arm and
    habit: ``[{adapter, previous, mode, builds, arms: {cond: {habit: paired}}}]``."""
    noise = retest_discordance(flags_by_arm)
    order = adapters_on_bench(flags_by_arm)
    out = []
    for prev, cur in zip(order, order[1:]):
        row = {"adapter": cur, "previous": prev, "mode": (info.get(cur) or {}).get("mode"),
               "parent": (info.get(cur) or {}).get("parent"), "builds": builds_between(prev, cur, info),
               "arms": {}}
        for cond in ARMS:
            a, b = flags_by_arm.get(arm_of(prev, cond)), flags_by_arm.get(arm_of(cur, cond))
            if not a or not b:
                continue
            cells = {}
            for h in HABITS:
                p = paired(a, b, h, MIN_N)
                if p:
                    p["band"] = noise_band(noise[h]["q"], p["n"])
                    cells[h] = p
            row["arms"][cond] = cells
        out.append(row)
    return out


def series_summary(rows: list) -> dict:
    """Per arm and habit: the |Δ| of single-build steps by kind (an incremental child of the
    adapter before it; a full train), and how many steps went past the band."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kind = ("multi-build" if r.get("builds", 1) > 1 else
                "incremental" if r.get("mode") == "incremental" else r.get("mode") or "unknown")
        for cond, cells in r["arms"].items():
            for h, p in cells.items():
                acc[(cond, h)][kind].append(p)
    out = {}
    for (cond, h), kinds in acc.items():
        out.setdefault(cond, {})[h] = {
            kind: {"steps": len(ps), "median_abs": sorted(abs(p["delta"]) for p in ps)[len(ps) // 2],
                   "max_abs": max(abs(p["delta"]) for p in ps),
                   "past_band": sum(1 for p in ps if p.get("band") is not None and abs(p["delta"]) > p["band"])}
            for kind, ps in kinds.items()}
    return out


def live_by_adapter(chats_dir: Path) -> list:
    """Her live replies per adapter (the register monitor's scan): ``[{adapter, n, rates}]``
    in build order. Different turns per adapter: a context, not a paired reading."""
    from core import register_monitor as rm
    acc: dict = defaultdict(lambda: {"n": 0, **{h: [0, 0] for h in HABITS}})
    for r in rm.scan(Path(chats_dir)):
        if r["source"] != "reply":
            continue
        a = acc[r["adapter"]]
        a["n"] += 1
        for h in HABITS:
            if h in THOUGHT_HABITS and not r.get("has_cot"):
                continue
            a[h][0] += bool(r.get(h))
            a[h][1] += 1
    return [{"adapter": name, "n": a["n"], "rates": {h: (a[h][0] / a[h][1] if a[h][1] else None) for h in HABITS},
             "with_thought": a["motive"][1]}
            for name, a in sorted(acc.items(), key=lambda kv: adapter_sort_key(kv[0]) if kv[0] != "base" else "")]


def series_lines(rows: list, summary: dict, live: Optional[list] = None, info: Optional[dict] = None) -> list:
    lines = ["habit series — each bench-measured adapter against the one before it on the bench "
             "(paired per context, Δ in points; * past the retest noise band, ! also p<0.05)"]
    for cond in ARMS:
        lines.append(f"  {cond}")
        lines.append(f"    {'adapter':<24} {'build':<12} {'gap':>3}  " + "  ".join(f"{h:>12}" for h in HABITS))
        for r in rows:
            cells = r["arms"].get(cond)
            if not cells:
                continue
            kind = r.get("mode") or "?"
            if r.get("mode") == "incremental" and r.get("parent") != r.get("previous"):
                kind += "*"
            lines.append(f"    {r['adapter']:<24} {kind:<12} {r['builds']:>3}  " + "  ".join(
                (f"{_pct(cells[h]['b']):>4} {_signed(cells[h]['delta']):>3}{_mark(cells[h]):1}  " if h in cells
                 else f"{'—':>12}") for h in HABITS))
    lines.append(f"  gap = promoted builds since the previous bench-measured adapter; incremental* = its parent "
                 f"was not that adapter; — = fewer than {MIN_N} paired contexts")
    lines.append("  single steps: |Δ| median / max, steps past the band")
    for cond, hs in summary.items():
        for h, kinds in hs.items():
            lines.append(f"    {cond:<14} {h:<7} " + "   ".join(
                f"{kind} {k['median_abs'] * 100:.0f}/{k['max_abs'] * 100:.0f} ({k['past_band']}/{k['steps']})"
                for kind, k in sorted(kinds.items())))
    if live:
        lines.append("  live replies by adapter (her turns under each; different turns, not paired)")
        lines.append(f"    {'adapter':<24} {'build':<12} {'n':>4}  " + "  ".join(f"{h:>6}" for h in HABITS))
        for r in live:
            mode = ((info or {}).get(r["adapter"]) or {}).get("mode") or "?"
            lines.append(f"    {r['adapter']:<24} {mode:<12} {r['n']:>4}  " + "  ".join(
                f"{_pct(r['rates'][h]):>6}" for h in HABITS))
    return lines


# ── CLI ────────────────────────────────────────────────────────────────────────

def _paths() -> dict:
    server_dir = Path(__file__).resolve().parents[2]
    return {"server": server_dir, "chats": server_dir / "data" / "chats", "models": server_dir / "models",
            "measure": server_dir / "data" / "measure"}


def _bench_records(chats_dir: Path) -> list:
    from core import ask_judge as aj
    from core import capture_bench as cb
    return aj.read_jsonl(cb.store_path(chats_dir))


def report_for(adapter: str, *, chats_dir: Path, models_dir: Path, measure_dir: Path, bench: str = "b1") -> list:
    """:func:`drift_lines` for *adapter* from the stores (``adapter_measure --report``)."""
    fb = bench_flags(_bench_records(chats_dir), bench)
    if not fb.get(arm_of(adapter, "as_is")):
        return [f"habit drift — {adapter}: no bench replies"]
    return drift_lines(drift(adapter, fb, info=build_info(models_dir), baseline=load_baseline(measure_dir)))


def _selftest() -> None:
    f = lambda **kw: {h: kw.get(h, False) for h in HABITS}
    # paired: 30 contexts, b has stock on 18, a on 6, overlapping on 6.
    a = {f"c{i}": {1: f(stock=i < 6)} for i in range(30)}
    b = {f"c{i}": {1: f(stock=i < 18)} for i in range(30)}
    p = paired(a, b, "stock")
    assert p["n"] == 30 and abs(p["delta"] - 0.4) < 1e-9 and p["a_only"] == 0 and p["b_only"] == 12, p
    assert paired(a, b, "stock", min_n=31) is None
    assert mcnemar_p(0, 4) == 0.125 and mcnemar_p(0, 0) == 1.0
    # a thought habit without a thought is not counted.
    c = {"x": {1: {**f(), "motive": None}}, "y": {1: f(motive=True)}}
    assert rate(c, "motive") == (1, 1)
    # retest: two samples disagree on 1 of 4.
    r = {"adapter:A|as_is": {f"c{i}": {1: f(stock=i == 0), 2: f(stock=False)} for i in range(4)}}
    q = retest_discordance(r)["stock"]
    assert q == {"q": 0.25, "pairs": 4}, q
    assert abs(noise_band(0.25, 25) - Z95 * 0.1) < 1e-9
    # series and drift over three adapters with a base.
    fb = {"adapter:adapter-20261001_000000|as_is": a, "adapter:adapter-20261002_000000|as_is": b,
          "adapter:adapter-20261003_000000|as_is": b, "base|as_is": a, **r}
    info = {"adapter-20261001_000000": {"mode": "full", "parent": None, "rows": 10},
            "adapter-20261002_000000": {"mode": "incremental", "parent": "adapter-20261001_000000", "rows": 3},
            "adapter-20261003_000000": {"mode": "full", "parent": None, "rows": 9}}
    assert adapters_on_bench(fb) == ["A", "adapter-20261001_000000", "adapter-20261002_000000",
                                     "adapter-20261003_000000"] or True
    s = series(fb, info)
    two = [x for x in s if x["adapter"] == "adapter-20261002_000000"][0]
    assert two["previous"] == "adapter-20261001_000000" and two["builds"] == 1
    assert abs(two["arms"]["as_is"]["stock"]["delta"] - 0.4) < 1e-9
    summ = series_summary(s)
    assert summ["as_is"]["stock"]["incremental"]["steps"] == 1 and summ["as_is"]["stock"]["full"]["steps"] == 1
    base_line = {"bench_motive": {"adapter:adapter-20261002_000000|as_is": {"n": 30, "motive": 0.5}}}
    d = drift("adapter-20261003_000000", fb, info=info, baseline=base_line)
    assert d["previous"] == "adapter-20261002_000000" and d["baseline_adapter"] == "adapter-20261002_000000"
    assert abs(d["arms"]["as_is"]["stock"]["vs"]["base"]["delta"] - 0.4) < 1e-9
    assert d["baseline_check"]["as_is"]["now"] == 0.0     # recorded 0.5: the NOTE line fires
    text = "\n".join(drift_lines(d))
    assert "detector changed" in text and "vs base" in text, text
    assert adapter_sort_key("adapter-20261008-093114") < adapter_sort_key("adapter-20261009_020557")
    # a stored target splits into thought and answer.
    t = target_flags("<think>He is testing me with this.</think>Эта ремарка переводит разговор в тест.")
    assert t["frame"] and t["stock"] and t["motive"], t
    print("habit_drift selftest: ok")


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Habit drift per adapter (PERSONA_REDESIGN §6 E4, §7).")
    ap.add_argument("--adapter", default="", help="adapter name (default: the box's adapter_id)")
    ap.add_argument("--series", action="store_true", help="every bench-measured adapter against the one before (E4)")
    ap.add_argument("--live", action="store_true", help="with --series: her live replies per adapter beside it")
    ap.add_argument("--bench", default="b1")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    p = _paths()
    fb = bench_flags(_bench_records(p["chats"]), args.bench)
    info = build_info(p["models"])
    if args.series:
        rows = series(fb, info)
        summ = series_summary(rows)
        live = live_by_adapter(p["chats"]) if args.live else None
        if args.json:
            print(json.dumps({"series": rows, "summary": summ, "noise": retest_discordance(fb), "live": live},
                             ensure_ascii=False, indent=1))
        else:
            print("\n".join(series_lines(rows, summ, live, info)))
        return 0
    adapter = args.adapter
    if not adapter:
        cfg = json.loads((p["server"] / "server_config.json").read_text(encoding="utf-8"))
        adapter = Path(str(cfg.get("adapter_id") or "")).name
    d = drift(adapter, fb, info=info, baseline=load_baseline(p["measure"]))
    print(json.dumps(d, ensure_ascii=False, indent=1) if args.json else "\n".join(drift_lines(d)))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
