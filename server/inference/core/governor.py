"""The habit governor's reference and plan — ``AVA_REWARD_LOOP.md`` §4.11, item 28;
``PERSONA_REDESIGN.md`` §6 E2. GPU-free. The reference and the plan read and write nothing
Ava reads; the build step (below) is the one part that acts: it holds rows out of a build.

**The reference** (``r_base``). The frozen base answers the measurement's frozen sample of
ordinary live turns (``data/measure/live_sample.json``) through the seam each turn had:
its stored system content (standing prompt, portraits, RAG block as they were), the
earlier exchanges CoT-stripped (``hidden_capture.replay_conversation``), the turn's own
sampling and the live chat's guards. ``governor_reference.py`` (GPU, inference down)
writes one ``reply`` record per turn per set of weights to
``data/measure/governor_reference.jsonl``; the serving adapter answering the same turns
gives the same-turn gap, and the reply she actually gave live is the third reading, free.
All three are paired per turn. The base sees her earlier replies in the history (decided
2026-10-09: the history stays), so a habit the person's push draws out of the base too is
inside the reference.

**What trains** (``r_train``). The newest build snapshot's ``sft_render.jsonl``: each chat
row's target (``variant``) read by the same detectors, weighted by its ``lr_multiplier``,
the dose the build gave it (wall-clock decay included). Wander rows have no person and no
pushback; they are left out of the shares and counted.

**The plan.** Per habit, setpoint ``k · r_base``. When the dose-weighted share is above it,
the flagged rows are held newest first until it is not (§4.11 item 4); habits are taken in
:data:`PLAN_ORDER`, each seeing the holds before it, and every share is read again after
the union. Printed for a few *k*. ``--plan`` before the reference exists previews against
the capture bench's base arm (pushback contexts, not ordinary turns): labelled so.

**The build step** (2026-10-10, k = 1.5 by the operator): :func:`apply_in_build`, called by
``training/train_cycle`` after an incremental build has computed each row's owed dose (or
over every row of a full train), so it reads exactly what is about to train. Rows are
grouped by exchange (a contamination copy goes with its row); per governed habit
(``governor.habits``: ``motive``, ``stock``) the dose-weighted share of what trains is held at
or below ``k · r_base`` by holding the newest flagged exchanges. A hold is the sidecar's
``governor`` field (``{habit, k, r_base, share, build, at, target_sha}``), bound to the target
(``chat_sidecar.governor_held``), refused by ``build_dialogue_anchor`` in every later build,
carried by ``write_verdict``, and never set on a locked (hand-authored) record. **Release:**
when a habit's share after the holds is below ``k · r_base − band``, the oldest held
exchanges of that habit are released (the field cleared) while the share, counting each at
the build's mean dose, stays under that line; they train from the next build. With no
reference on disk (``governor_reference.jsonl`` for the current live sample) the step does
nothing and says so. A held row keeps its transcript, recall and verdicts.

Habits: ``core/habit_drift.py`` (the register monitor's form detectors). CLI:
``python -m core.governor [--reference] [--plan [--k 1.5]] [--snapshot DIR] [--json]``;
self-test ``python -m core.governor --selftest``.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from core import habit_drift as hd

STORE_FILE = "governor_reference.jsonl"
PLAN_ORDER = ("motive", "stock", "frame", "tone")
K_GRID = (1.0, 1.25, 1.5, 2.0)
_STEM_TS = re.compile(r"^(\d{8}_\d{6})")


# ── the store ──────────────────────────────────────────────────────────────────

def store_path(measure_dir) -> Path:
    return Path(measure_dir) / STORE_FILE


def read_store(measure_dir) -> list:
    out = []
    try:
        for line in store_path(measure_dir).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                out.append(r)
    except FileNotFoundError:
        pass
    return out


def append(measure_dir, rec: dict) -> dict:
    rec = {"record": "reply", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    path = store_path(measure_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def done_keys(records: Iterable[dict], *, weights: str, sample_sha: str, draw: int = 1) -> set:
    """Turn keys already answered by *weights* on this sample (any outcome but ``error``)."""
    return {r["key"] for r in records if r.get("record") == "reply" and r.get("weights") == weights
            and r.get("sample_sha") == sample_sha and int(r.get("draw") or 1) == draw
            and r.get("outcome") != "error"}


# ── the reference ──────────────────────────────────────────────────────────────

def turn_names(doc: dict, index: int) -> tuple:
    """The person's names for the motive detector: the exchange's speaker and the chat's user."""
    exs = doc.get("exchanges") or []
    ex = exs[index] if 0 <= index < len(exs) and isinstance(exs[index], dict) else {}
    return (str(ex.get("speaker") or ""), str(doc.get("user") or ""))


def reference_flags(records: Iterable[dict], sample_sha: str, names: Optional[dict] = None) -> dict:
    """``{weights: {key: {1: flags}}}`` over the ``ok`` replies on *sample_sha* (draw 1),
    in :mod:`habit_drift`'s arm shape so its pairing applies."""
    out: dict = defaultdict(dict)
    for r in records:
        if (r.get("record") != "reply" or r.get("sample_sha") != sample_sha or r.get("outcome") != "ok"
                or int(r.get("draw") or 1) != 1):
            continue
        out[str(r["weights"])][str(r["key"])] = {1: hd.flags(str(r.get("answer") or ""), str(r.get("cot") or ""),
                                                             (names or {}).get(r["key"], ()))}
    return dict(out)


def live_flags(chats_dir, sample: dict) -> tuple:
    """``({key: {1: flags}}, {key: names}, {key: adapter})``: the reply she gave live on
    each sampled turn, under the adapter that answered it (``core/lineage.py``); a reply a
    foreign adapter gave is left out."""
    from core import lineage as ln
    flags_, names, adapters, docs = {}, {}, {}, {}
    for t in sample.get("turns") or []:
        stem, _, idx = t["key"].rpartition("#")
        if stem not in docs:
            try:
                docs[stem] = json.loads((Path(chats_dir) / f"{stem}.json").read_text(encoding="utf-8"))
            except Exception:
                docs[stem] = None
        doc = docs[stem]
        exs = (doc or {}).get("exchanges") or []
        i = int(idx)
        if not doc or not (0 <= i < len(exs)) or not isinstance(exs[i], dict):
            continue
        names[t["key"]] = turn_names(doc, i)
        reply = str(exs[i].get("assistant_response") or "")
        hidden = ln.hidden_for(Path(chats_dir) / f"{stem}.json")
        if reply.strip() and ln.is_own(doc, i, hidden) is not False:     # a foreign adapter's reply is not hers
            flags_[t["key"]] = {1: hd.flags(reply, str(exs[i].get("assistant_cot") or ""), names[t["key"]])}
            adapters[t["key"]] = ln.serving_adapter(doc, i, hidden) or "base"
    return flags_, names, adapters


def reference_table(arms: dict) -> dict:
    """Per arm and habit: ``(k, n)``; per non-base arm and habit, paired against the base."""
    out = {"rates": {}, "vs_base": {}}
    base = arms.get("base")
    for name, by_key in arms.items():
        out["rates"][name] = {h: hd.rate(by_key, h) for h in hd.HABITS}
        if base and name != "base":
            out["vs_base"][name] = {h: hd.paired(base, by_key, h) for h in hd.HABITS}
    return out


def wilson(k: int, n: int, z: float = hd.Z95) -> tuple:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


# ── what trains ────────────────────────────────────────────────────────────────

def newest_snapshot(models_dir) -> Optional[Path]:
    """The newest promoted build's snapshot directory (``builds.jsonl``)."""
    found = None
    try:
        for line in (Path(models_dir) / "builds.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                b = json.loads(line)
            except Exception:
                continue
            if b.get("outcome") == "promoted" and b.get("snapshot_dir"):
                found = Path(b["snapshot_dir"])
    except FileNotFoundError:
        pass
    return found


def train_rows(snapshot_dir, chats_dir) -> dict:
    """``{rows: [{session, exchange, ts, dose, kind, flags}], skipped: {source: n}}`` from a
    build snapshot's chat rows."""
    rows, skipped, docs = [], defaultdict(int), {}
    path = Path(snapshot_dir) / "sft_render.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("source") != "chat":
            skipped[str(r.get("source"))] += 1
            continue
        session = str(r.get("source_session") or "")
        try:
            idx = int(r.get("exchange_index"))
        except (TypeError, ValueError):
            skipped["no_exchange"] += 1
            continue
        if session not in docs:
            try:
                docs[session] = json.loads((Path(chats_dir) / session).read_text(encoding="utf-8"))
            except Exception:
                docs[session] = {}
        m = _STEM_TS.match(session)
        rows.append({"session": session, "exchange": idx, "ts": m.group(1) if m else "",
                     "dose": float(r.get("lr_multiplier") or 1.0), "kind": str(r.get("target_kind") or ""),
                     "flags": hd.target_flags(str(r.get("variant") or ""), turn_names(docs[session], idx))})
    return {"rows": rows, "skipped": dict(skipped)}


def train_share(rows: Iterable[dict], habit: str, held: frozenset = frozenset()) -> dict:
    """``{weighted, unweighted, n, dose}`` of *habit* over *rows* not in *held*."""
    w = f = n = k = 0.0
    for i, r in enumerate(rows):
        v = r["flags"].get(habit)
        if v is None or i in held:
            continue
        w += r["dose"]
        f += r["dose"] * v
        n += 1
        k += v
    return {"weighted": (f / w) if w else None, "unweighted": (k / n) if n else None, "n": int(n), "dose": w}


def plan(rows: list, r_base: dict, k: float, order: Iterable[str] = PLAN_ORDER) -> dict:
    """Hold flagged rows newest first until each habit's dose-weighted share is at most
    ``k · r_base[habit]``, then release the oldest holds no habit needs. ``{k, habits: {h:
    {setpoint, before, after_own, held}}, held, held_dose, total_dose, pruned, after: {h:
    share}}``; a habit with no reference is skipped. ``held`` per habit counts the final
    holds that carry it.

    Joint, not per habit (2026-10-11): the first live build planned each habit on its own
    and held the union. A reply often carries both habits, so stock's ten holds also took
    the rows motive's plan had kept: motive went 18% → 0% against a 12% setpoint (its own
    plan stopped at 11.3%). The prune pass releases, oldest first, every hold whose release
    leaves each habit at or below its setpoint (or no worse than the greedy pass left it,
    for a habit whose flagged rows ran out), so a share lands at the setpoint, not under it."""
    held: set = set()
    newest = sorted(range(len(rows)), key=lambda i: (rows[i]["ts"], rows[i]["exchange"]), reverse=True)
    out = {"k": k, "habits": {}}
    setpoints: dict = {}
    for h in order:
        rb = r_base.get(h)
        if rb is None:
            continue
        s = min(1.0, k * rb)
        setpoints[h] = s
        before = train_share(rows, h, frozenset(held))["weighted"]
        for i in newest:
            cur = train_share(rows, h, frozenset(held))["weighted"]
            if cur is None or cur <= s:
                break
            if i in held or not rows[i]["flags"].get(h):
                continue
            held.add(i)
        out["habits"][h] = {"setpoint": s, "r_base": rb, "before": before}
    greedy = {h: train_share(rows, h, frozenset(held))["weighted"] for h in setpoints}
    bound = {h: max(s, greedy[h] if greedy[h] is not None else 0.0) for h, s in setpoints.items()}
    pruned = 0
    for i in sorted(held, key=lambda i: (rows[i]["ts"], rows[i]["exchange"])):
        trial = frozenset(held - {i})
        if all((train_share(rows, h, trial)["weighted"] or 0.0) <= bound[h] + 1e-12 for h in setpoints):
            held.discard(i)
            pruned += 1
    for h, d in out["habits"].items():
        d["held"] = sum(1 for i in held if rows[i]["flags"].get(h))
        d["after_own"] = train_share(rows, h, frozenset(held))["weighted"]
    out["pruned"] = pruned
    total = sum(r["dose"] for r in rows)
    out["held"] = len(held)
    out["held_dose"] = sum(rows[i]["dose"] for i in held)
    out["total_dose"] = total
    out["rows"] = len(rows)
    out["after"] = {h: train_share(rows, h, frozenset(held))["weighted"] for h in hd.HABITS}
    out["held_ids"] = sorted(held)
    return out


# ── the build step ─────────────────────────────────────────────────────────────

BUILD_DEFAULTS = {"enabled": True, "k": 1.5, "band": 0.02, "habits": ["motive", "stock"]}


def build_settings(cfg: Optional[dict] = None) -> dict:
    """``governor.*`` with defaults (config_schema.py is the documented source)."""
    blk = (cfg or {}).get("governor") or {}
    out = dict(BUILD_DEFAULTS)
    out["enabled"] = bool(blk.get("enabled", BUILD_DEFAULTS["enabled"]))
    for k_ in ("k", "band"):
        try:
            out[k_] = max(0.0, float(blk.get(k_, BUILD_DEFAULTS[k_])))
        except (TypeError, ValueError):
            pass
    habits = blk.get("habits", BUILD_DEFAULTS["habits"])
    if isinstance(habits, str):
        habits = [h.strip() for h in habits.split(",")]
    out["habits"] = [h for h in (habits or []) if h in hd.HABITS]
    return out


def base_rates(measure_dir, chats_dir=None) -> Optional[dict]:
    """The base's rate per habit on the current frozen live sample, or None when the
    reference was not run for that sample. With *chats_dir*, the motive detector is given
    each turn's speaker and user (as ``--reference`` reads it)."""
    from core import adapter_measure as am
    sample = am.load_sample(measure_dir) or {}
    if not sample.get("sha"):
        return None
    names = live_flags(chats_dir, sample)[1] if chats_dir is not None else None
    arm = reference_flags(read_store(measure_dir), sample["sha"], names).get("base")
    if not arm or len(arm) < 50:
        return None
    return {h: (k / n if n else None) for h, (k, n) in ((h, hd.rate(arm, h)) for h in hd.HABITS)}


def exchange_rows(rows: Iterable, names_fn: Callable[[str], tuple] = lambda s: ()) -> list:
    """Build rows grouped by exchange: ``[{key, session, exchange, ts, dose, flags}]`` over
    chat rows; the dose sums every row of the exchange (a contamination copy included), the
    flags read its target."""
    acc: dict = {}
    for r in rows:
        if getattr(r, "source", "") != "chat" or getattr(r, "exchange_index", None) is None:
            continue
        key = (r.source_session, int(r.exchange_index))
        e = acc.get(key)
        if e is None:
            m = _STEM_TS.match(str(r.source_session or ""))
            e = acc[key] = {"key": key, "session": r.source_session, "exchange": int(r.exchange_index),
                            "ts": m.group(1) if m else "", "dose": 0.0,
                            "flags": hd.target_flags(str(r.target or ""), names_fn(r.source_session))}
        e["dose"] += float(r.lr_multiplier or 0.0)
    return list(acc.values())


def release_plan(held: list, rows: list, r_base: dict, *, k: float, band: float, habits: Iterable[str]) -> list:
    """Held exchanges to release, oldest first: per habit, while the share (each released
    one counted at the rows' mean dose, flagged) stays below ``k·r_base − band``."""
    out = []
    mean = (sum(r["dose"] for r in rows) / len(rows)) if rows else 1.0
    for h in habits:
        rb = r_base.get(h)
        if rb is None:
            continue
        line = min(1.0, k * rb) - band
        w = sum(r["dose"] for r in rows if r["flags"].get(h) is not None)
        f = sum(r["dose"] for r in rows if r["flags"].get(h))
        for it in sorted([x for x in held if x.get("habit") == h], key=lambda x: (x["ts"], x["exchange"])):
            if w <= 0 or (f + mean) / (w + mean) >= line:
                break
            w += mean
            f += mean
            out.append(it)
    return out


def apply_in_build(rows: list, *, chats_dir, measure_dir, cfg: Optional[dict] = None, build_id: str = "",
                   write: bool = True, say: Callable[[str], None] = print) -> tuple:
    """The governor inside a build: ``(rows to train, report)``. Holds the newest flagged
    exchanges per habit (sidecar ``governor`` field), releases old holds below the band,
    and drops the held exchanges' rows from *rows*. *write* False (a dry run) plans and
    writes no sidecar. Never raises: a failure keeps *rows* as they are and reports it."""
    st = build_settings(cfg)
    report: dict = {"enabled": st["enabled"], "k": st["k"], "band": st["band"], "habits": st["habits"]}
    if not st["enabled"] or not st["habits"]:
        return rows, {**report, "skipped": "disabled"}
    try:
        r_base = base_rates(measure_dir, chats_dir)
        if not r_base:
            say("governor: no base reference for the current live sample — nothing held")
            return rows, {**report, "skipped": "no_reference"}
        report["r_base"] = {h: r_base.get(h) for h in st["habits"]}
        from core.chat_sidecar import ChatSidecar, exchange_target_sha, governor_held
        sidecar = ChatSidecar(Path(chats_dir))
        docs: dict = {}

        def names(session: str) -> tuple:
            if session not in docs:
                try:
                    docs[session] = json.loads((Path(chats_dir) / session).read_text(encoding="utf-8"))
                except Exception:
                    docs[session] = {}
            speakers = {str(e.get("speaker") or "") for e in (docs[session].get("exchanges") or [])
                        if isinstance(e, dict) and not str(e.get("speaker") or "").startswith("(")}
            return tuple(sorted(speakers - {""})) + (str(docs[session].get("user") or ""),)

        ex = exchange_rows(rows, names)
        # Locked (hand-authored) records are the operator's: never held, still counted.
        locked = set()
        for e in ex:
            rec = (sidecar.load(e["session"]).get("exchanges") or {}).get(str(e["exchange"])) or {}
            if rec.get("locked"):
                locked.add(e["key"])
        p = plan([dict(e, flags=({h: False for h in e["flags"]} if e["key"] in locked else e["flags"]))
                  for e in ex], {h: r_base.get(h) for h in st["habits"]}, st["k"], order=st["habits"])
        report["plan"] = {h: {k_: v for k_, v in d.items()} for h, d in p["habits"].items()}
        held_keys = {ex[i]["key"] for i in p["held_ids"]}
        habit_of = {}
        for i in p["held_ids"]:
            e = ex[i]
            habit_of[e["key"]] = next((h for h in st["habits"] if e["flags"].get(h)), st["habits"][0])
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        written = 0
        for e in ex:
            if e["key"] not in held_keys:
                continue
            rec = (sidecar.load(e["session"]).get("exchanges") or {}).get(str(e["exchange"])) or {}
            h = habit_of[e["key"]]
            if write and sidecar.set_record_field(e["session"], e["exchange"], "governor", {
                    "habit": h, "k": st["k"], "r_base": r_base.get(h),
                    "share": (p["habits"].get(h) or {}).get("before"), "build": build_id, "at": now,
                    "target_sha": exchange_target_sha(str(rec.get("target") or ""))}):
                written += 1
        # Release: the holds standing from earlier builds (not among this build's rows).
        standing = []
        for path in sorted(Path(chats_dir).glob("*.state.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            session = path.name[:-len(".state.json")] + ".json"
            for idx, rec in (data.get("exchanges") or {}).items():
                if governor_held(rec) and (session, int(idx)) not in held_keys:
                    m = _STEM_TS.match(session)
                    standing.append({"session": session, "exchange": int(idx), "ts": m.group(1) if m else "",
                                     "habit": (rec.get("governor") or {}).get("habit")})
        kept_rows = [e for e in ex if e["key"] not in held_keys]
        released = release_plan(standing, kept_rows, r_base, k=st["k"], band=st["band"], habits=st["habits"])
        for it in released if write else ():
            sidecar.set_record_field(it["session"], it["exchange"], "governor", None)
        out_rows = [r for r in rows if getattr(r, "source", "") != "chat"
                    or (r.source_session, int(r.exchange_index)) not in held_keys]
        report.update(held=len(held_keys), written=written, released=len(released), standing=len(standing),
                      rows_before=len(rows), rows_after=len(out_rows),
                      after={h: p["after"].get(h) for h in st["habits"]})
        say(f"governor (k={st['k']}): held {len(held_keys)} exchange(s), released {len(released)}; "
            + "; ".join(f"{h} {_pct((p['habits'].get(h) or {}).get('before'))} → {_pct(p['after'].get(h))} "
                        f"(setpoint {_pct((p['habits'].get(h) or {}).get('setpoint'))})" for h in st["habits"]))
        return out_rows, report
    except Exception as e:  # noqa: BLE001 — the governor must never break a build
        say(f"governor: error, nothing held ({type(e).__name__}: {e})")
        return rows, {**report, "error": f"{type(e).__name__}: {e}"}


# ── report ─────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def reference_lines(table: dict, sample: dict, adapters_live: Optional[dict] = None) -> list:
    lines = [f"governor reference — the frozen live sample ({len(sample.get('turns') or [])} turns, sha {sample.get('sha')}),"
             " each turn answered through its stored seam; paired per turn against the base"]
    for name, rates in sorted(table["rates"].items(), key=lambda kv: (kv[0] != "base", kv[0])):
        parts = []
        for h in hd.HABITS:
            k, n = rates[h]
            lo, hi = wilson(k, n)
            parts.append(f"{h} {_pct(k / n if n else None):>4} [{lo:.0%}–{hi:.0%}] n {n}")
        lines.append(f"  {name:<34} " + "  ".join(parts))
        for h, p in (table["vs_base"].get(name) or {}).items():
            if p:
                lines.append(f"      {h:<7} vs base on {p['n']} turns: {p['a']:.0%} → {p['b']:.0%} "
                             f"(Δ {p['delta'] * 100:+.0f}; base only {p['a_only']}, this only {p['b_only']}, p={p['p']})")
    if adapters_live:
        by = defaultdict(int)
        for a in adapters_live.values():
            by[a] += 1
        lines.append(f"  live = the reply she gave on each turn, under {len(by)} adapters of the day")
    return lines


def plan_lines(rows: dict, r_base: dict, source: str, ks: Iterable[float] = K_GRID) -> list:
    rs = rows["rows"]
    lines = [f"governor plan — what trains: {len(rs)} chat rows (skipped {rows.get('skipped') or {}}), "
             f"weighted by each row's dose; reference: {source}"]
    lines.append("  r_train now: " + "  ".join(
        f"{h} {_pct(train_share(rs, h)['weighted'])} (unweighted {_pct(train_share(rs, h)['unweighted'])})"
        for h in hd.HABITS))
    lines.append("  r_base:      " + "  ".join(f"{h} {_pct(r_base.get(h))}" for h in hd.HABITS))
    for k in ks:
        p = plan(rs, r_base, k)
        lines.append(f"  k={k:<4} held {p['held']}/{p['rows']} rows, dose {p['held_dose']:.0f}/{p['total_dose']:.0f} "
                     f"({p['held_dose'] / p['total_dose']:.0%})" if p["total_dose"] else f"  k={k}: no rows")
        for h, d in p["habits"].items():
            lines.append(f"      {h:<7} setpoint {_pct(d['setpoint']):>4}  on entry {_pct(d['before']):>4}  "
                         f"held {d['held']:>3}  after {_pct(p['after'][h]):>4}")
        rest = [h for h in hd.HABITS if h not in p["habits"]]
        if rest:
            lines.append("      not governed: " + ", ".join(f"{h} after {_pct(p['after'][h])}" for h in rest))
    return lines


def bench_base_rates(chats_dir) -> dict:
    """The capture bench's base arm, as is: the preview reference before the live one exists."""
    fb = hd.bench_flags(hd._bench_records(Path(chats_dir)))
    arm = fb.get("base|as_is") or {}
    return {h: (k / n if n else None) for h, (k, n) in ((h, hd.rate(arm, h)) for h in hd.HABITS)}


# ── self-test ──────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    f = lambda **kw: {h: kw.get(h, False) for h in hd.HABITS}
    # 10 rows, dose 1 each, motive on the 5 newest: 50%; base 10%, k=2 → setpoint 20%.
    rows = [{"session": f"2026100{i // 2}_000000.json", "exchange": i % 2, "ts": f"2026100{i // 2}_000000",
             "dose": 1.0, "kind": "branch", "flags": f(motive=i >= 5)} for i in range(10)]
    assert train_share(rows, "motive")["weighted"] == 0.5
    p = plan(rows, {"motive": 0.1}, 2.0, order=("motive",))
    # holding h of the 5 flagged: (5-h)/(10-h) <= 0.2 → h >= 3.75 → 4 held, the newest ones.
    assert p["held"] == 4 and abs(p["after"]["motive"] - 1 / 6) < 1e-9, p
    assert set(p["held_ids"]) == {9, 8, 7, 6}, p["held_ids"]
    # joint: two habits on overlapping rows (stock on 4-9, motive on 6-9), setpoints 15%:
    # both land at or below it, and every hold left is needed.
    rows3 = [{"session": f"2026100{i // 2}_000000.json", "exchange": i % 2, "ts": f"2026100{i // 2}_000000",
              "dose": 1.0, "kind": "branch", "flags": f(stock=i >= 4, motive=i >= 6)} for i in range(10)]
    p3 = plan(rows3, {"motive": 0.1, "stock": 0.1}, 1.5, order=("motive", "stock"))
    assert p3["after"]["stock"] <= 0.15 + 1e-9 and p3["after"]["motive"] <= 0.15 + 1e-9, p3
    # every hold is needed: releasing any one puts a habit over its setpoint
    for i in p3["held_ids"]:
        rest = frozenset(set(p3["held_ids"]) - {i})
        assert any(train_share(rows3, h, rest)["weighted"] > 0.15 + 1e-9 for h in ("motive", "stock")), (i, p3)
    # the first live build's shape: the newest replies motive-only (18, 19), stock on 8-17 of
    # which 12-13 carry motive too. Motive's plan holds 18 and 19; stock's must hold 9-17,
    # taking 12-13, so the union left motive at 0. The prune releases 18 (motive 10%, stock
    # 10%, both under 15%); 19 must stay (motive would be 18%).
    rows4 = [{"session": f"202610{i:02d}_000000.json", "exchange": 1, "ts": f"202610{i:02d}_000000",
              "dose": 1.0, "kind": "branch",
              "flags": f(stock=8 <= i <= 17, motive=i in (12, 13, 18, 19))} for i in range(20)]
    p4 = plan(rows4, {"motive": 0.1, "stock": 0.1}, 1.5, order=("motive", "stock"))
    assert p4["pruned"] == 1 and 18 not in p4["held_ids"] and 19 in p4["held_ids"], p4
    assert abs(p4["after"]["motive"] - 0.1) < 1e-9 and p4["after"]["stock"] <= 0.15 + 1e-9, p4
    # at or below the setpoint nothing is held; no reference, no habit.
    assert plan(rows, {"motive": 0.5}, 1.0, order=("motive",))["held"] == 0
    assert plan(rows, {}, 1.5)["habits"] == {}
    # dose weighting: an old row with dose 3 counts three times.
    rows2 = [dict(rows[0], dose=3.0, flags=f(motive=True)), dict(rows[9], flags=f())]
    assert train_share(rows2, "motive")["weighted"] == 0.75
    # the store round trip and the resume set.
    with tempfile.TemporaryDirectory() as td:
        append(td, {"key": "a#1", "weights": "base", "sample_sha": "s", "draw": 1, "outcome": "ok",
                    "answer": "Эта ремарка переводит разговор в плоскость теста.", "cot": "He is testing me."})
        append(td, {"key": "a#2", "weights": "base", "sample_sha": "s", "draw": 1, "outcome": "error"})
        recs = read_store(td)
        assert done_keys(recs, weights="base", sample_sha="s") == {"a#1"}
        ref = reference_flags(recs, "s")
        assert ref["base"]["a#1"][1]["stock"] and ref["base"]["a#1"][1]["motive"], ref
        t = reference_table({"base": ref["base"], "live": {"a#1": {1: f()}}})
        assert t["rates"]["base"]["stock"] == (1, 1) and t["vs_base"]["live"]["stock"]["delta"] == -1.0
    lo, hi = wilson(8, 100)
    assert 0.04 < lo < 0.08 < hi < 0.16, (lo, hi)
    _selftest_build()
    print("governor selftest: ok")


def _selftest_build() -> None:
    """The build step end to end on a temp corpus: holds written to the sidecars, the rows
    dropped, a dry run writing nothing, the build refusing a held record, a release."""
    import tempfile
    from types import SimpleNamespace
    from core.chat_sidecar import ChatSidecar, exchange_target_sha, governor_held
    stock = "<think>He is testing me.</think>Эта ремарка переводит разговор в тест."
    plain = "<think>A friendly hello.</think>Привет! Рада тебя слышать."
    with tempfile.TemporaryDirectory() as td:
        chats, measure = Path(td) / "chats", Path(td) / "measure"
        chats.mkdir()
        measure.mkdir()
        # the reference: base stock / motive at 10% on 60 sampled turns.
        sample = {"turns": [{"key": f"s#{i}", "short": False} for i in range(60)], "sha": "S"}
        (measure / "live_sample.json").write_text(json.dumps(sample), encoding="utf-8")
        for i in range(60):
            t = stock if i < 6 else plain
            ans = t.split("</think>")[1]
            cot = t.split("</think>")[0][len("<think>"):]
            append(measure, {"key": f"s#{i}", "weights": "base", "draw": 1, "sample_sha": "S", "outcome": "ok",
                             "answer": ans, "cot": cot})
        rb = base_rates(measure)
        assert abs(rb["stock"] - 0.1) < 1e-9 and abs(rb["motive"] - 0.1) < 1e-9, rb
        # 10 exchanges in 10 chats, the 6 newest carrying the habits; one locked.
        sc = ChatSidecar(chats)
        rows = []
        for i in range(10):
            sess = f"2026100{i // 5 + 1}_0{i % 5}0000.json"
            (chats / sess).write_text(json.dumps({"user": "Dana", "exchanges": [{"user_prompt": "hi"}] * 2}),
                                      encoding="utf-8")
            tgt = stock if i >= 4 else plain
            sc.write_verdict(source_session=sess, exchange_index=1, verdict="keep", target=tgt, run_id="t", locked=(i == 9))
            rows.append(SimpleNamespace(source="chat", source_session=sess, exchange_index=1, lr_multiplier=1.0,
                                        target=tgt))
        rows.append(SimpleNamespace(source="wander", source_session="w", exchange_index=None, lr_multiplier=1.0,
                                    target=""))
        cfg = {"governor": {"k": 1.5, "band": 0.02, "habits": "motive,stock"}}
        out, rep = apply_in_build(rows, chats_dir=chats, measure_dir=measure, cfg=cfg, write=False,
                                  say=lambda _m: None)
        held_dry = rep["held"]
        assert held_dry > 0 and not any(governor_held((sc.load(r.source_session).get("exchanges") or {}).get("1"))
                                        for r in rows if r.source == "chat"), "a dry run wrote a hold"
        out, rep = apply_in_build(rows, chats_dir=chats, measure_dir=measure, cfg=cfg, say=lambda _m: None)
        assert rep["held"] == held_dry and len(out) == len(rows) - rep["held"], rep
        assert any(r.source == "wander" for r in out)
        recs = {r.source_session: (sc.load(r.source_session).get("exchanges") or {}).get("1") for r in rows
                if r.source == "chat"}
        held = [s_ for s_, rec in recs.items() if governor_held(rec)]
        assert len(held) == rep["held"] and recs[rows[9].source_session].get("locked") and \
            rows[9].source_session not in held, (held, rep)
        assert all(s_ >= "20261001_04" for s_ in held), held          # the newest flagged ones
        # the share after the holds is at or below the setpoint
        assert rep["after"]["stock"] <= 1.5 * 0.1 + 1e-9, rep
        # the build refuses a held record
        from training.dialogue_source import build_dialogue_anchor
        assert build_dialogue_anchor(chats, held[0], 1, recs[held[0]]) is None
        # a re-derived target unbinds the hold
        sc.write_verdict(source_session=held[0], exchange_index=1, verdict="revise", target=plain, run_id="t2")
        assert not governor_held((sc.load(held[0]).get("exchanges") or {}).get("1"))
        # release: a later build with only plain rows releases the oldest held while under the line.
        rows2 = [SimpleNamespace(source="chat", source_session=f"20261009_00000{i}.json", exchange_index=1,
                                 lr_multiplier=1.0, target=plain) for i in range(8)]
        for r in rows2:
            (chats / r.source_session).write_text(json.dumps({"user": "Dana", "exchanges": [{}] * 2}), encoding="utf-8")
            sc.write_verdict(source_session=r.source_session, exchange_index=1, verdict="keep", target=plain, run_id="t3")
        _out2, rep2 = apply_in_build(rows2, chats_dir=chats, measure_dir=measure, cfg=cfg, say=lambda _m: None)
        assert rep2["held"] == 0 and rep2["released"] >= 1 and rep2["standing"] == len(held) - 1, rep2
    # no reference: nothing held
    with tempfile.TemporaryDirectory() as td:
        out, rep = apply_in_build(rows, chats_dir=Path(td), measure_dir=Path(td), cfg={}, say=lambda _m: None)
        assert out == rows and rep.get("skipped") == "no_reference", rep


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The habit governor's reference and plan (AVA_REWARD_LOOP §4.11).")
    ap.add_argument("--reference", action="store_true", help="the reference rates on the frozen live sample")
    ap.add_argument("--plan", action="store_true", help="what the governor would hold in the newest build")
    ap.add_argument("--k", type=float, default=0.0, help="--plan: one k instead of the grid")
    ap.add_argument("--reference-weights", default="base", help="--plan: whose rates are the reference")
    ap.add_argument("--snapshot", default="", help="--plan: a build snapshot dir (default: the newest promoted)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    p = hd._paths()
    from core import adapter_measure as am
    sample = am.load_sample(p["measure"]) or {}
    live, names, adapters_live = live_flags(p["chats"], sample)
    arms = reference_flags(read_store(p["measure"]), sample.get("sha", ""), names)
    if live:
        arms["live"] = live
    table = reference_table(arms)
    out: dict = {}
    if args.reference or not args.plan:
        out["reference"] = table
        if not args.json:
            print("\n".join(reference_lines(table, sample, adapters_live)))
    if args.plan:
        ref = arms.get(args.reference_weights)
        if ref:
            r_base = {h: (k / n if n else None) for h, (k, n) in table["rates"][args.reference_weights].items()}
            source = f"{args.reference_weights} on the live sample"
        else:
            r_base = bench_base_rates(p["chats"])
            source = "PREVIEW — the capture bench's base arm (pushback contexts, not the live sample)"
        snap = Path(args.snapshot) if args.snapshot else newest_snapshot(p["models"])
        if snap is None:
            raise SystemExit("no build snapshot")
        rows = train_rows(snap, p["chats"])
        ks = (args.k,) if args.k else K_GRID
        out["plan"] = {"snapshot": str(snap), "r_base": r_base, "source": source,
                       "plans": [{kk: v for kk, v in plan(rows["rows"], r_base, k).items() if kk != "held_ids"}
                                 for k in ks]}
        if not args.json:
            print("")
            print(f"snapshot {snap.name}")
            print("\n".join(plan_lines(rows, r_base, source, ks)))
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
