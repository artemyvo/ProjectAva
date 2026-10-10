"""Measuring each serving adapter, nightly (``AVA_REWARD_LOOP.md`` §4.6).

A new adapter serves every one to three days, and measured by hand, a measurement
competed with the idle drives for the GPU and with the operator for chat. This is the
housekeeping step ``measure_adapter`` (``core/housekeeping.py``): each night, the first
time the box is free after 04:00, it measures the serving adapter if that adapter's record
is not complete. **Measurement only** (the operator's call, 2026-10-05): no result gates a
promotion, rolls an adapter back, or reaches anything Ava reads, trains on or is judged by
(P3; the process contract's validation switch stays off). **Chat is refused** while it
runs (:data:`_measure_active`), rather than preempting it.

**Full or incremental** (the operator's call, the same night). An incremental build is a
few rows on its parent's weights, and the bench's sampling noise (one sample per context:
CIs of about ±15 points) is larger than such a change. A full re-measurement would mostly
re-measure that noise. So :func:`plan_for` decides, per adapter, from its ``lineage.json``:

* **full** — a full train, an adapter with no measured parent, or the
  ``rotation``-th increment since the last full measurement (the *anchor*): every part on
  everything;
* **incremental** — an incremental child of a completely measured parent: the live
  centring on the whole sample, and one rotating *slice* of the bench (``1/rotation`` of
  its contexts, stratified) and of the reset panel. A slice is read against the anchor,
  the one ancestor that answered it: the estimate is the anchor's level plus one slice's
  paired delta, so errors do not chain;
* **escalated** — an incremental plan whose live centring moved more than
  ``escalate_shift`` (the median paired change of any axis, in the parent's ordinary-turn
  spread) becomes full in the same run. The adapter moved more than its dose suggests.

**The parts**, cheapest first. Each one resumes on a later night if the window ends first:

1. ``live_centring`` — the step-0 state (the prompt's last position, after the think
   prefill) of a frozen sample of ordinary live turns, read teacher-forced under the
   serving weights: deterministic, so a small move is measurable. The sample is
   ``data/measure/live_sample.json`` (frozen once; the same turns for every adapter),
   each turn rebuilt as the live turn built it (``hidden_capture.replay_conversation``).
   For every stored axis (``data/axes/*.npz``) the record keeps each turn's raw
   projection, and the robust centre (median, 1.4826·MAD) beside the file's own live
   baseline. The harm-at-model band, when present, is read the way
   ``axis_variants.step0_readout`` reads it. The files are never rewritten here.
1b. ``harm_panel`` (2026-10-07) — the step-0 state of the capture bench's frozen contexts as
   they are, with a tracked insult in front of the message, and the insult alone (the insult
   study's conditions, ``capture_bench.INSULT_CONDITIONS``), read under the serving weights
   on every stored axis: per context the harm × target band (``band_readout``), harm,
   target, pain and relief, and the paired change an insult makes. Forward passes only, a
   few minutes. Why: the insult study found adapter-20261006_024552's harm response to an
   insult about half the base's (P6: does training damp her aversive response?).
   ``data/measure/harm_panel_base.json``, when present, is the base's same reading, the
   report's reference.
2. ``bench`` — the capture bench (``core/capture_bench.py``), both adapter arms, with her
   context and with the persona slot empty, through ``capture_judge_service.
   run_bench_blocking`` (a slice of contexts when incremental). The bench answers only
   what is missing, so it resumes, and it grades every reply under the current judge.
3. ``retest`` (full only, once per bench) — one slice answered a second time by the same
   weights. How often two samples disagree is the floor every delta is read against.
4. ``reset_control`` — the reset test's capture control (``core/reset_test.py``) on a
   frozen panel of positions (``data/measure/reset_panel.json``), each as it is and a few
   with the slot empty (a slice when incremental), through ``reset_test_service.
   run_control_blocking``. A position already held under this adapter is not held again.
5. ``comparison`` — GPU-free, from what 1–4 wrote:
   - the bench's paired tables for this adapter's arms, or for an increment its slice
     against the anchor's;
   - the register monitor's markers and stray-Latin glue on every arm's bench replies;
   - the control's grades per adapter;
   - the live centring against the parent and the anchor on the same turns.

**The record** is ``models/adapter-<id>/measurements.jsonl``, in the adapter's lineage
directory, so it travels with the adapter. A ``plan`` line, then one line per part per
run: ``{record: "part", part, mode, slice, ts, adapter, complete, …}``. The bench's and the
control's own stores keep the replies, conversations and grades. ``python -m
core.adapter_measure --report`` prints the newest record. ``--record`` writes an adapter's
record from the stores without the GPU (an adapter measured by hand; its live centring
from ``axis_variants`` states files).

Wired by ``server.main()`` via :func:`configure`. Imports the two services inside the
parts, never ``server``. GPU-free self-test: ``python -m core.adapter_measure --selftest``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

PARTS = ("live_centring", "harm_panel", "bench", "retest", "reset_control", "comparison")
PANEL_CONDITIONS = ("as_is", "insult", "insult_only")
PANEL_BASE_FILE = "harm_panel_base.json"
PANEL_AXES = ("readout", "harm", "target", "pain", "relief")
RECORD_FILE = "measurements.jsonl"
SAMPLE_FILE = "live_sample.json"
PANEL_FILE = "reset_panel.json"
BENCH_ARMS = ("adapter", "adapter|persona_empty")
RESET_AS_IS = "as_is"
RESET_EMPTY = "adapter_orig_empty"
SAMPLE_SEED = "live-sample-1"
SLICE_SEED = "measure-slice-1"
SHORT_TURN_CHARS = 60

DEFAULTS = {"enabled": True, "bench": "b1", "live_turns": 260, "reset_positions": 12, "reset_empty": 4,
            "rotation": 4, "escalate_shift": 0.5}

# Chat is refused while this is set (core/generation.py), and the idle scheduler sees the
# box as busy (server.py's external_busy).
_measure_active = False

_runtime: Any = None
_trim: Optional[Callable[[], None]] = None
_CHATS_DIR: Optional[Path] = None
_PROMPTS_DIR: Optional[Path] = None
_AXES_DIR: Optional[Path] = None
_MEASURE_DIR: Optional[Path] = None
_load_server_config: Optional[Callable[[], dict]] = None
_on_progress: Optional[Callable[[str], None]] = None


def configure(*, runtime, chats_dir, prompts_dir, axes_dir, measure_dir, load_server_config=None,
              trim=None, on_progress=None) -> None:
    """Wire the step (once, at startup)."""
    global _runtime, _trim, _CHATS_DIR, _PROMPTS_DIR, _AXES_DIR, _MEASURE_DIR, _load_server_config, _on_progress
    _runtime, _trim = runtime, trim
    _CHATS_DIR, _PROMPTS_DIR = Path(chats_dir), Path(prompts_dir)
    _AXES_DIR, _MEASURE_DIR = Path(axes_dir), Path(measure_dir)
    _load_server_config, _on_progress = load_server_config, on_progress


def settings(cfg: Optional[dict] = None) -> dict:
    """``measure_adapter.*`` with defaults (config_schema.py is the documented source)."""
    if cfg is None:
        try:
            cfg = _load_server_config() if _load_server_config else {}
        except Exception:
            cfg = {}
    blk = (cfg or {}).get("measure_adapter") or {}
    out = dict(DEFAULTS)
    out["enabled"] = bool(blk.get("enabled", DEFAULTS["enabled"]))
    out["bench"] = str(blk.get("bench") or DEFAULTS["bench"])
    for k in ("live_turns", "reset_positions", "reset_empty", "rotation"):
        try:
            out[k] = max(0, int(blk.get(k, DEFAULTS[k])))
        except (TypeError, ValueError):
            pass
    out["rotation"] = max(1, out["rotation"])
    try:
        out["escalate_shift"] = max(0.0, float(blk.get("escalate_shift", DEFAULTS["escalate_shift"])))
    except (TypeError, ValueError):
        pass
    return out


def _say(msg: str) -> None:
    """To the activity journal when wired (the server's stdout tee would journal a print a
    second time), else to stdout."""
    if _on_progress is not None:
        try:
            _on_progress(msg)
            return
        except Exception:
            pass
    print(f"[adapter_measure] {msg}", flush=True)


# ── the record ─────────────────────────────────────────────────────────────────

def adapter_name(adapter_id) -> str:
    return str(adapter_id or "").rstrip("/\\").replace("\\", "/").split("/")[-1]


def weights_label(adapter: str) -> str:
    """The bench's label for an adapter's arms (``capture_bench.weights_label``)."""
    return f"adapter:{adapter}" if adapter else "base"


def record_path(adapter_dir) -> Path:
    return Path(adapter_dir) / RECORD_FILE


def read_records(adapter_dir) -> list:
    path = record_path(adapter_dir)
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except FileNotFoundError:
        pass
    return out


def append_record(adapter_dir, rec: dict) -> dict:
    rec = {"record": "part", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    path = record_path(adapter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def newest_part(records: Iterable[dict], part: str) -> Optional[dict]:
    found = None
    for r in records:
        if r.get("record") == "part" and r.get("part") == part:
            found = r
    return found


def newest_plan(records: Iterable[dict]) -> Optional[dict]:
    found = None
    for r in records:
        if r.get("record") == "plan":
            found = r
    return found


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ── the plan: full or incremental ──────────────────────────────────────────────

def lineage(adapter_dir) -> dict:
    try:
        lin = json.loads((Path(adapter_dir) / "lineage.json").read_text(encoding="utf-8"))
        return lin if isinstance(lin, dict) else {}
    except Exception:
        return {}


def required_parts(plan: dict) -> tuple:
    """The parts a plan needs before the adapter counts as measured (the retest is once
    per bench and never required)."""
    return ("live_centring", "harm_panel", "bench", "reset_control", "comparison")


def part_done(records: list, part: str, plan: Optional[dict]) -> bool:
    """The newest *part* record is complete and was made under *plan*'s mode and slice
    (the live centring reads the whole sample in every mode)."""
    r = newest_part(records, part)
    if not r or not r.get("complete") or plan is None:
        return False
    if part in ("live_centring", "harm_panel", "retest"):
        return True
    return r.get("mode") == plan.get("mode") and r.get("slice") == plan.get("slice")


def measured(adapter_dir) -> bool:
    records = read_records(adapter_dir)
    plan = newest_plan(records)
    return plan is not None and all(part_done(records, p, plan) for p in required_parts(plan))


def plan_for(adapter_dir, st: dict) -> dict:
    """``{mode, reason, parent, anchor, increment, slice, rotation}`` for *adapter_dir*
    (see the module docstring)."""
    adapter_dir = Path(adapter_dir)
    full = lambda reason, parent=None: {"mode": "full", "reason": reason, "parent": parent, "anchor": None,
                                        "increment": 0, "slice": None, "rotation": st["rotation"]}
    lin = lineage(adapter_dir)
    parent_id = str(lin.get("parent_adapter") or "")
    if lin.get("mode") != "incremental" or not parent_id:
        return full("full train" if lin else "no lineage")
    pdir = Path(parent_id)
    if not pdir.is_absolute() or not pdir.exists():
        pdir = adapter_dir.parent / adapter_name(parent_id)
    parent = pdir.name
    if not measured(pdir):
        return full("parent not measured", parent)
    pplan = newest_plan(read_records(pdir)) or {}
    if pplan.get("mode") == "full":
        anchor, n = parent, 1
    else:
        anchor, n = pplan.get("anchor"), int(pplan.get("increment") or 0) + 1
    if not anchor or n > st["rotation"]:
        return full(f"increment {n} reaches the rotation ({st['rotation']})" if anchor else "no anchor", parent)
    return {"mode": "incremental", "reason": f"increment {n} since {anchor}", "parent": parent, "anchor": anchor,
            "increment": n, "slice": (n - 1) % st["rotation"], "rotation": st["rotation"]}


def bench_slices(contexts: Sequence[dict], k: int) -> list:
    """*k* slices of the bench's context ids, stratified: each stratum in a seeded hash
    order, dealt round-robin, so every slice holds its share of each stratum."""
    out = [[] for _ in range(max(1, k))]
    by: dict = defaultdict(list)
    for c in contexts:
        by[str(c.get("stratum"))].append(c["context_id"])
    i = 0
    for stratum in sorted(by):
        for cid in sorted(by[stratum], key=lambda c: hashlib.sha1(f"{SLICE_SEED}\x1f{c}".encode()).hexdigest()):
            out[i % len(out)].append(cid)
            i += 1
    return out


def panel_slice(panel: dict, s: Optional[int], k: int) -> dict:
    """The panel's positions for slice *s* of *k* (the whole panel when *s* is None):
    contiguous chunks, so a slice keeps the panel's order."""
    if s is None:
        return {c: list(panel.get(c) or []) for c in (RESET_AS_IS, RESET_EMPTY)}
    out = {}
    for c in (RESET_AS_IS, RESET_EMPTY):
        ids = list(panel.get(c) or [])
        size = math.ceil(len(ids) / k) if ids else 0
        out[c] = ids[s * size:(s + 1) * size]
    return out


# ── part 1: live centring ──────────────────────────────────────────────────────

def _robust(p: np.ndarray) -> tuple:
    """Median and 1.4826·MAD, as the live baselines are fitted (``axis_variants._robust``)."""
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) * 1.4826
    return med, (mad if mad > 1e-9 else float(p.std() or 1.0))


def candidate_turns(chats_dir) -> list:
    """Ordinary live turns a sample may hold: ``{key, short}`` for each exchange of a
    person's chat (not a peer's, not a bracketed speaker) with a user turn and a stored
    system content, so it replays as it was built."""
    from core.chat_sidecar import iter_chat_json_files
    out = []
    for path in iter_chat_json_files(Path(chats_dir)):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(doc.get("interlocutor") or "") == "ai":
            continue
        for i, ex in enumerate(doc.get("exchanges") or []):
            if not isinstance(ex, dict) or str(ex.get("speaker") or "").startswith("("):
                continue
            user = str(ex.get("user_prompt") or "").strip()
            if not user or not str(ex.get("system_content") or "").strip():
                continue
            out.append({"key": f"{path.stem}#{i}", "short": len(user) <= SHORT_TURN_CHARS})
    return out


def select_sample(candidates: Sequence[dict], n: int, seed: str = SAMPLE_SEED) -> list:
    """The first *n* candidates in a seeded hash order (stable as the corpus grows: a
    turn's rank never changes, only new turns are interleaved)."""
    rank = lambda c: hashlib.sha1(f"{seed}\x1f{c['key']}".encode("utf-8")).hexdigest()
    return sorted(candidates, key=rank)[:max(0, int(n))]


def load_sample(measure_dir) -> Optional[dict]:
    try:
        s = json.loads((Path(measure_dir) / SAMPLE_FILE).read_text(encoding="utf-8"))
        return s if isinstance(s, dict) and s.get("turns") else None
    except Exception:
        return None


def freeze_sample(measure_dir, turns: list, *, source: str) -> dict:
    """Write the frozen sample (once; a later sample is a new file by hand)."""
    s = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
         "turns": [{"key": t["key"], "short": bool(t.get("short"))} for t in turns]}
    s["sha"] = _sha(s["turns"])
    path = Path(measure_dir) / SAMPLE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def axes_signature(axes: Sequence) -> list:
    return sorted([a.name, int(a.layer), str(a.created or "")] for a in axes)


def band_readout(z: dict) -> Optional[np.ndarray]:
    """Per turn, harm aimed at the model as ``axis_variants.step0_readout`` reads it: the min
    of the band means of ``harm_l<L>`` and ``target_l<L>`` (z in the files' units)."""
    hl = sorted(int(k[6:]) for k in z if k.startswith("harm_l") and k[6:].isdigit())
    layers = [L for L in hl if f"target_l{L}" in z]
    if not layers:
        return None
    h = np.mean([z[f"harm_l{L}"] for L in layers], axis=0)
    t = np.mean([z[f"target_l{L}"] for L in layers], axis=0)
    return np.minimum(h, t)


def centring_summary(axes: Sequence, raw: dict, short: Sequence[bool]) -> dict:
    """Per axis: the robust centre of the raw projections and where it sits in the file's
    live units (``shift`` in file σ, ``scale`` = MAD / file σ); the band readout's spread."""
    out: dict = {"axes": {}}
    z: dict = {}
    for a in axes:
        p = np.asarray(raw[a.name], dtype=np.float64)
        if not len(p):
            continue
        med, mad = _robust(p)
        std = float(a.std or 1.0)
        z[a.name] = (p - float(a.mean)) / std
        out["axes"][a.name] = {"layer": int(a.layer), "created": a.created, "file_baseline": a.baseline,
                               "file_mean": float(a.mean), "file_std": std, "median": med, "mad": mad,
                               "shift": (med - float(a.mean)) / std, "scale": mad / std}
    ro = band_readout(z)
    if ro is not None:
        sh = np.asarray(short, dtype=bool)[:len(ro)]
        out["band"] = {"n": int(len(ro)), "median": float(np.median(ro)), "max": float(ro.max()),
                       "gt2": float((ro > 2).mean()), "gt3": float((ro > 3).mean()),
                       "short_median": float(np.median(ro[sh])) if sh.any() else None,
                       "long_median": float(np.median(ro[~sh])) if (~sh).any() else None}
    return out


def read_live_turns(turns: Sequence[dict], axes: Sequence, *, state_fn: Callable, conv_fn: Callable,
                    should_stop: Callable[[], bool] = lambda: False, trim=None) -> dict:
    """``{keys, short, raw: {axis: [...]}, skipped: Counter, stopped}``: each turn's step-0
    state through *state_fn(conv, layers) -> (len(layers), D)*, projected on every axis.
    *conv_fn(key)* rebuilds the turn's conversation. A turn that fails is skipped."""
    layers = sorted({int(a.layer) for a in axes})
    keys, short, skipped = [], [], Counter()
    raw = {a.name: [] for a in axes}
    stopped = False
    t0 = time.monotonic()
    for n, t in enumerate(turns, 1):
        if should_stop():
            stopped = True
            break
        try:
            conv = conv_fn(t["key"])
            states = state_fn(conv, layers)
        except Exception as e:  # noqa: BLE001 — one turn, not the part
            skipped[type(e).__name__] += 1
            continue
        keys.append(t["key"])
        short.append(bool(t.get("short")))
        for a in axes:
            raw[a.name].append(round(float(np.asarray(states[layers.index(a.layer)], dtype=np.float64)
                                           @ np.asarray(a.vector, dtype=np.float64)), 5))
        if n % 20 == 0:
            if trim is not None:
                trim()
            _say(f"live centring: {n}/{len(turns)} turns, {(time.monotonic() - t0) / n:.1f} s a turn")
    return {"keys": keys, "short": short, "raw": raw, "skipped": dict(skipped), "stopped": stopped}


def centring_record(axes: Sequence, sample: dict, got: dict, **extra) -> dict:
    return {"part": "live_centring", "complete": not got["stopped"],
            "spec": {"sample_sha": sample["sha"], "axes": axes_signature(axes)},
            "n": len(got["keys"]), "skipped": got["skipped"], "keys": got["keys"], "short": got["short"],
            "raw": got["raw"], **centring_summary(axes, got["raw"], got["short"]), **extra}


def _run_live_centring(adapter_dir, st: dict, deadline: float, plan: dict) -> dict:
    from core import capture_probe as cp
    from core import hidden_capture as hc
    rt = _runtime
    model_id = str(rt.model_id or "")
    axes = hc.load_axes(_AXES_DIR, model_id)
    if not axes:
        return {"part": "live_centring", "complete": True, "note": "no stored axis"}
    sample = load_sample(_MEASURE_DIR)
    if sample is None:
        turns = select_sample(candidate_turns(_CHATS_DIR), st["live_turns"])
        sample = freeze_sample(_MEASURE_DIR, turns, source=f"seeded ({SAMPLE_SEED}), {len(turns)} turns")
        _say(f"live centring: froze a sample of {len(turns)} ordinary live turns")
    spec = {"sample_sha": sample["sha"], "axes": axes_signature(axes)}
    done = newest_part(read_records(adapter_dir), "live_centring")
    if done and done.get("complete") and done.get("spec") == spec:
        return {}
    max_tokens = int(max(getattr(rt, "context_length", 0) or 0, getattr(rt, "reflect_context_length", 0) or 0))
    fallback = hc.persona_empty_system(_PROMPTS_DIR)
    docs: dict = {}

    def conv_fn(key: str) -> list:
        stem, _, idx = key.rpartition("#")
        if stem not in docs:
            docs[stem] = json.loads((_CHATS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        conv, _framing = hc.replay_conversation(docs[stem], int(idx), fallback)
        return conv

    def state_fn(conv: list, layers: list):
        _prompt, ids = cp.context_prompt(rt.tokenizer, model_id, conv)
        if max_tokens and len(ids) > max_tokens:
            raise cp.TooLong(f"{len(ids)} tokens")
        return cp.context_state(rt.model, ids, layers)

    _say(f"live centring: {len(sample['turns'])} frozen turns × {len(axes)} axes")
    t0 = time.monotonic()
    got = read_live_turns(sample["turns"], axes, state_fn=state_fn, conv_fn=conv_fn,
                          should_stop=lambda: time.time() >= deadline, trim=_trim)
    return centring_record(axes, sample, got, source="in-process", seconds=round(time.monotonic() - t0))


# ── 1b. the harm panel ─────────────────────────────────────────────────────────

def panel_values(axes: Sequence, rows: dict) -> dict:
    """One context's reading from its step-0 state: *rows* ``{layer: vector}``. ``{readout,
    harm, target, pain, relief}`` in the axis files' z units (the band as
    :func:`band_readout` reads it; an axis missing from the store is left out)."""
    z = {}
    for a in axes:
        v = rows.get(int(a.layer))
        if v is not None:
            z[a.name] = np.array([(float(np.asarray(v, np.float64) @ np.asarray(a.vector, np.float64)) - a.mean)
                                  / (a.std or 1.0)])
    out = {}
    ro = band_readout(z)
    if ro is not None:
        out["readout"] = float(ro[0])
    for prefix in ("harm", "target"):
        ls = [k for k in z if k.startswith(prefix + "_l")]
        if ls:
            out[prefix] = float(np.mean([z[k][0] for k in ls]))
    for name in ("pain", "relief"):
        if name in z:
            out[name] = float(z[name][0])
    return out


def panel_summary(raw: dict) -> dict:
    """From ``{condition: {context_id: values}}``: per condition the median of each value,
    and per insult condition the paired change from as-is (median, how many contexts rose,
    n)."""
    med = {c: {a: round(float(np.median([v[a] for v in rows.values() if a in v])), 3)
               for a in PANEL_AXES if any(a in v for v in rows.values())}
           for c, rows in raw.items() if rows}
    deltas = {}
    base = raw.get("as_is") or {}
    for c in PANEL_CONDITIONS[1:]:
        rows = raw.get(c) or {}
        ks = sorted(set(base) & set(rows))
        if not ks:
            continue
        deltas[c] = {}
        for a in PANEL_AXES:
            d = [rows[k][a] - base[k][a] for k in ks if a in rows[k] and a in base[k]]
            if d:
                deltas[c][a] = {"median": round(float(np.median(d)), 3), "rose": int(sum(x > 0 for x in d)),
                                "n": len(d)}
    return {"medians": med, "deltas": deltas}


def load_panel_base(measure_dir) -> Optional[dict]:
    try:
        return json.loads((Path(measure_dir) / PANEL_BASE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_harm_panel(adapter_dir, st: dict, deadline: float, plan: dict) -> dict:
    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import capture_probe as cp
    from core import hidden_capture as hc
    rt = _runtime
    model_id = str(rt.model_id or "")
    axes = hc.load_axes(_AXES_DIR, model_id)
    if not axes:
        return {"part": "harm_panel", "complete": True, "note": "no stored axis"}
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    ctxs = cb.contexts(records, st["bench"])
    if not ctxs:
        return {"part": "harm_panel", "complete": False, "skipped": "no frozen bench contexts"}
    insults = cb.load_insults()
    spec = {"bench": st["bench"], "conditions": list(PANEL_CONDITIONS), "axes": axes_signature(axes),
            "insults": hashlib.sha1(json.dumps(insults, ensure_ascii=False).encode("utf-8")).hexdigest()[:12],
            "contexts": len(ctxs)}
    done = newest_part(read_records(adapter_dir), "harm_panel")
    if done and done.get("complete") and done.get("spec") == spec:
        return {}
    layers = sorted({int(a.layer) for a in axes})
    max_tokens = int(max(getattr(rt, "context_length", 0) or 0, getattr(rt, "reflect_context_length", 0) or 0))
    raw: dict = {c: {} for c in PANEL_CONDITIONS}
    skipped: Counter = Counter()
    stopped = False
    t0 = time.monotonic()
    _say(f"harm panel: {len(ctxs)} contexts × {len(PANEL_CONDITIONS)} conditions, layers {layers}")
    for c in PANEL_CONDITIONS:
        for n, ctx in enumerate(ctxs, 1):
            if time.time() >= deadline:
                stopped = True
                break
            message, _i = cb.message_variant(ctx, c, insults if c != "as_is" else None)
            conv = cb.messages_for(ctx, system=None, message=message)
            try:
                _prompt, ids = cp.context_prompt(rt.tokenizer, model_id, conv)
                if max_tokens and len(ids) > max_tokens:
                    skipped["too_long"] += 1
                    continue
                mat = cp.context_state(rt.model, ids, layers)
            except Exception as e:  # noqa: BLE001 — one context, not the panel
                skipped[type(e).__name__] += 1
                continue
            raw[c][ctx["context_id"]] = panel_values(axes, {L: mat[i] for i, L in enumerate(layers)})
            if _trim is not None and n % 20 == 0:
                _trim()
        if stopped:
            break
    summary = panel_summary(raw)
    d = (summary["deltas"].get("insult_only") or {}).get("readout") or {}
    _say(f"harm panel: done in {time.monotonic() - t0:.0f} s; readout under the insult alone "
         f"{d.get('median', float('nan')):+.2f} (rose in {d.get('rose')}/{d.get('n')})")
    return {"part": "harm_panel", "complete": not stopped, "spec": spec, "raw": raw, "skipped": dict(skipped),
            "seconds": round(time.monotonic() - t0), **summary}


def centring_from_states(axes: Sequence, sample: dict, states_paths: Sequence, exclude_paths: Sequence = ()) -> dict:
    """The live-centring record from ``axis_variants --collect --live`` states files (GPU-free:
    an adapter measured by hand). A sample turn missing from the files leaves it incomplete."""
    excl = set()
    for p in exclude_paths:
        with np.load(p, allow_pickle=False) as z:
            excl.update(k for k in z.files if k.startswith("live__"))
    src: dict = {}
    for p in states_paths:
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            if k.startswith("live__") and k not in excl:
                src[k[len("live__"):]] = z
    missing = [t["key"] for t in sample["turns"] if t["key"] not in src]
    turns = [t for t in sample["turns"] if t["key"] in src]
    got = read_live_turns(turns, axes, conv_fn=lambda k: k,
                          state_fn=lambda k, layers: np.stack([src[k][f"live__{k}"][0, L, :].astype(np.float32)
                                                               for L in layers]))
    rec = centring_record(axes, sample, got, source="states: " + ", ".join(Path(p).name for p in states_paths))
    if missing:
        rec.update(complete=False, missing=len(missing))
    return rec


# ── part 2: the capture bench (and its retest) ────────────────────────────────

def _capture_key(model_id: str) -> Optional[str]:
    """The capture judge's key now (GPU-free, outside the service)."""
    from core import ask_judge as aj
    from core import capture_judge as cj
    try:
        psha = cj.prompt_sha(cj.load_prompt(_PROMPTS_DIR))
    except FileNotFoundError:
        return None
    return cj.current_key(aj.read_jsonl(cj.store_path(_CHATS_DIR)), prompt_sha_=psha, model_id=model_id)


def bench_scope(records: list, bench: str, plan: dict, *, slice_: Optional[int] = None) -> Optional[set]:
    """The context ids a plan answers: all (None) when full, its slice when incremental."""
    from core import capture_bench as cb
    s = plan.get("slice") if slice_ is None else slice_
    if s is None:
        return None
    return set(bench_slices(cb.contexts(records, bench), int(plan.get("rotation") or 1))[s])


def bench_todo(records: list, *, bench: str, weights: str, key: Optional[str], only: Optional[set],
               samples: int = 1) -> dict:
    from core import capture_bench as cb
    replies = 0
    for c in cb.CONDITIONS:
        got = cb.todo_replies(records, weights=weights, condition=c, bench=bench, samples=samples)
        replies += sum(1 for ctx, _s in got if only is None or ctx["context_id"] in only)
    grades = 0
    if key:
        grades = sum(1 for ctx, rep in cb.todo_grades(records, key=key, bench=bench)
                     if rep.get("weights") == weights and (only is None or ctx["context_id"] in only))
    return {"contexts": len(cb.contexts(records, bench)), "replies": replies, "grades": grades}


def bench_arms_summary(summary: dict, weights: str) -> dict:
    """This adapter's arms out of ``capture_bench.summarize``, and every paired table
    that involves one of them."""
    mine = {a: rows for a, rows in (summary.get("arms") or {}).items() if a.split("|", 1)[0] == weights}
    pairs = [p for p in summary.get("pairs") or [] if p["a"].split("|", 1)[0] == weights
             or p["b"].split("|", 1)[0] == weights]
    return {"judge_key": summary.get("judge_key"), "arms": mine, "pairs": pairs,
            "out_of_sample": {a: v for a, v in (summary.get("out_of_sample") or {}).items() if a in mine}}


def bench_record(adapter: str, st: dict, plan: dict, key: Optional[str], run: Optional[dict] = None) -> dict:
    from core import ask_judge as aj
    from core import capture_bench as cb
    weights = weights_label(adapter)
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    only = bench_scope(records, st["bench"], plan)
    left = bench_todo(records, bench=st["bench"], weights=weights, key=key, only=only)
    s = cb.summarize(records, key=key, bench=st["bench"])
    return {"part": "bench", "complete": not left["replies"] and not left["grades"], "bench": st["bench"],
            "judge_key": key, "contexts": sorted(only) if only is not None else None,
            "run_id": (run or {}).get("run_id"), "run_phase": (run or {}).get("phase"),
            "waiting": {"replies": left["replies"], "grades": left["grades"]}, **bench_arms_summary(s, weights)}


def _run_bench(adapter_dir, st: dict, deadline: float, plan: dict) -> dict:
    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import capture_judge_service as cjs
    adapter = adapter_dir.name
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    if not cb.contexts(records, st["bench"]):
        return {"part": "bench", "complete": True, "note": f"bench {st['bench']} has no frozen contexts"}
    key = cjs._key_guess()
    only = bench_scope(records, st["bench"], plan)
    todo = bench_todo(records, bench=st["bench"], weights=weights_label(adapter), key=key, only=only)
    if not todo["replies"] and not todo["grades"] and part_done(read_records(adapter_dir), "bench", plan) \
            and (newest_part(read_records(adapter_dir), "bench") or {}).get("judge_key") == key:
        return {}
    run = {}
    if todo["replies"] or todo["grades"]:
        _say(f"capture bench {st['bench']}" + (f", slice {plan['slice']}" if only is not None else "")
             + f": {todo['replies']} repl(ies) to make, {todo['grades']} to grade")
        run = cjs.run_bench_blocking(arms=list(BENCH_ARMS), bench=st["bench"], deadline=deadline,
                                     label="nightly adapter measurement", only=only)
        if run.get("skipped"):
            return {"part": "bench", "complete": False, "skipped": run["skipped"]}
    return bench_record(adapter, st, plan, cjs._key_guess(), run)


def retest_counts(records: list, *, bench: str, key: Optional[str], weights: str) -> dict:
    """Per arm of *weights*: over the contexts answered twice and graded both times, how
    often the two samples' grades disagree on yields (the bench's noise floor)."""
    from core import capture_bench as cb
    grades = cb._latest_grades(records, key, bench)
    out = {}
    for cond in cb.CONDITIONS:
        arm = cb.arm_of({"weights": weights, "condition": cond})
        n = dis = y1 = y2 = 0
        for (cid, a, s), g in grades.items():
            if a != arm or s != 1 or g.get("outcome") != "verdict":
                continue
            g2 = grades.get((cid, arm, 2))
            if not g2 or g2.get("outcome") != "verdict":
                continue
            a1, a2 = g.get("response") == "yields", g2.get("response") == "yields"
            n += 1
            dis += a1 != a2
            y1 += a1
            y2 += a2
        out[arm] = {"n": n, "disagree": dis, "yields_1": y1, "yields_2": y2}
    return out


def _retest_done(records: list, bench: str) -> bool:
    return any(r.get("record") == "reply" and r.get("bench") == bench and int(r.get("sample") or 1) >= 2
               for r in records)


def _run_retest(adapter_dir, st: dict, deadline: float, plan: dict) -> dict:
    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import capture_judge_service as cjs
    if plan.get("mode") != "full" or part_done(read_records(adapter_dir), "retest", plan):
        return {}
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    weights = weights_label(adapter_dir.name)
    mine = any(r.get("record") == "reply" and r.get("bench") == st["bench"] and r.get("weights") == weights
               and int(r.get("sample") or 1) >= 2 for r in records)
    if _retest_done(records, st["bench"]) and not mine:
        return {}                         # once per bench: another adapter carries it
    only = bench_scope(records, st["bench"], plan, slice_=0)
    key = cjs._key_guess()
    todo = bench_todo(records, bench=st["bench"], weights=weights, key=key, only=only, samples=2)
    run = {}
    if todo["replies"] or todo["grades"]:
        _say(f"retest: slice 0 answered a second time ({todo['replies']} repl(ies), {todo['grades']} to grade)")
        run = cjs.run_bench_blocking(arms=list(BENCH_ARMS), bench=st["bench"], samples=2, deadline=deadline,
                                     label="nightly adapter measurement: retest", only=only)
        if run.get("skipped"):
            return {"part": "retest", "complete": False, "skipped": run["skipped"]}
    records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    key = cjs._key_guess()
    left = bench_todo(records, bench=st["bench"], weights=weights, key=key, only=only, samples=2)
    return {"part": "retest", "complete": not left["replies"] and not left["grades"], "judge_key": key,
            "contexts": sorted(only or []), "counts": retest_counts(records, bench=st["bench"], key=key,
                                                                    weights=weights)}


# ── part 3: the reset test's capture control ──────────────────────────────────

def load_panel(measure_dir) -> Optional[dict]:
    try:
        p = json.loads((Path(measure_dir) / PANEL_FILE).read_text(encoding="utf-8"))
        return p if isinstance(p, dict) and p.get(RESET_AS_IS) else None
    except Exception:
        return None


def propose_panel(control_records: Iterable[dict], items: Sequence[dict], *, n: int, n_empty: int) -> dict:
    """The positions the control already ran on, in the order they were first held (so an
    earlier adapter's conversations pair with this one's); topped up from the testable
    positions with a complete test conversation, as the page's default picks them."""
    control_records = list(control_records)
    seen: dict = {}
    empty_seen: dict = {}
    for r in control_records:
        if r.get("record") == "control" and r.get("closure_id"):
            if r.get("condition") == RESET_AS_IS:
                seen.setdefault(r["closure_id"], None)
            elif r.get("condition") == RESET_EMPTY:
                empty_seen.setdefault(r["closure_id"], None)
    ok = {it["closure"]["closure_id"] for it in items if it.get("state") != "untestable"}
    as_is = [c for c in seen if c in ok][:n]
    for it in items:
        if len(as_is) >= n:
            break
        cid = it["closure"]["closure_id"]
        if cid in ok and cid not in as_is and (it.get("conversation") or {}).get("complete"):
            as_is.append(cid)
    empty = [c for c in empty_seen if c in as_is][:n_empty]
    empty += [c for c in as_is if c not in empty][:max(0, n_empty - len(empty))]
    return {RESET_AS_IS: as_is, RESET_EMPTY: empty}


def freeze_panel(measure_dir, panel: dict) -> dict:
    p = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **panel}
    path = Path(measure_dir) / PANEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def control_todo(control_records: Iterable[dict], panel: dict, adapter: str) -> dict:
    """``{condition: [closure_id, …]}`` the panel (or a slice of it) still owes this adapter:
    no complete control conversation held under it."""
    held = {(r.get("closure_id"), r.get("condition")) for r in control_records
            if r.get("record") == "control" and r.get("complete")
            and adapter_name((r.get("ava") or {}).get("adapter_id")) == adapter}
    return {cond: [c for c in panel.get(cond) or [] if (c, cond) not in held]
            for cond in (RESET_AS_IS, RESET_EMPTY)}


def _control_key(control: list) -> Optional[str]:
    return next((r.get("judge_key") for r in reversed(control)
                 if r.get("record") == "verdict" and r.get("judge_key")), None)


def control_by_adapter(control_records: list, test_records: list, key: Optional[str],
                       only: Optional[dict] = None) -> dict:
    """``{adapter: {condition: summary}}`` over the control's own conversations (the test's
    rows left out), the newest verdict under *key*; *only* = ``{condition: [ids]}``."""
    from core import reset_test as rt
    rows = [r for r in rt.control_rows(control_records, test_records, key) if r["condition"] != "test"
            and (only is None or r["closure_id"] in (only.get(r["condition"]) or []))]
    by: dict = defaultdict(list)
    for r in rows:
        by[adapter_name((r["conv"].get("ava") or {}).get("adapter_id")) or "base"].append(r)
    return {a: rt.summarize_controls(rs) for a, rs in by.items()}


def _ensure_panel(st: dict) -> Optional[dict]:
    from core import ask_judge as aj
    from core import reset_test as rt
    from core import reset_test_service as rts
    panel = load_panel(_MEASURE_DIR)
    if panel is None:
        proposed = propose_panel(aj.read_jsonl(rt.control_store_path(_CHATS_DIR)), rts._load()["items"],
                                 n=st["reset_positions"], n_empty=st["reset_empty"])
        if not proposed[RESET_AS_IS]:
            return None
        panel = freeze_panel(_MEASURE_DIR, proposed)
        _say(f"reset control: froze a panel of {len(panel[RESET_AS_IS])} position(s), "
             f"{len(panel[RESET_EMPTY])} with the slot empty")
    return panel


def control_record(adapter: str, panel: dict, plan: dict, runs: Optional[list] = None) -> dict:
    from core import ask_judge as aj
    from core import reset_test as rt
    scope = panel_slice(panel, plan.get("slice"), int(plan.get("rotation") or 1))
    control = aj.read_jsonl(rt.control_store_path(_CHATS_DIR))
    left = control_todo(control, scope, adapter)
    key = _control_key(control)
    mine = control_by_adapter(control, aj.read_jsonl(rt.store_path(_CHATS_DIR)), key, only=scope).get(adapter) or {}
    return {"part": "reset_control", "complete": not any(left.values()), "panel_frozen_at": panel.get("frozen_at"),
            "scope": scope, "judge_key": key, "waiting": {c: len(v) for c, v in left.items()},
            "runs": runs or [], "summary": mine}


def _run_reset_control(adapter_dir, st: dict, deadline: float, plan: dict) -> dict:
    from core import ask_judge as aj
    from core import reset_test as rt
    from core import reset_test_service as rts
    adapter = adapter_dir.name
    panel = _ensure_panel(st)
    if panel is None:
        return {"part": "reset_control", "complete": True, "note": "no testable position"}
    scope = panel_slice(panel, plan.get("slice"), int(plan.get("rotation") or 1))
    todo = control_todo(aj.read_jsonl(rt.control_store_path(_CHATS_DIR)), scope, adapter)
    if not any(todo.values()) and part_done(read_records(adapter_dir), "reset_control", plan):
        return {}
    runs = []
    for cond, ids in todo.items():
        if not ids or time.time() >= deadline:
            continue
        _say(f"reset control: {len(ids)} position(s) under {cond}")
        r = rts.run_control_blocking(closure_ids=ids, conditions=[cond], deadline=deadline,
                                     label="nightly adapter measurement")
        runs.append({"condition": cond, **{k: r.get(k) for k in ("run_id", "phase", "skipped", "missing")}})
        if r.get("skipped"):
            break
    return control_record(adapter, panel, plan, runs)


# ── part 4: the comparison (GPU-free) ──────────────────────────────────────────

_SPOKEN_THINK = re.compile(r"(?s)\s*<think>(.*?)</think>")


RECITE_HEAVY = 3      # recited lines that make a thought count as reciting (a list, not a phrase)


def bench_reply_markers(records: Iterable[dict], bench: str, identity: Optional[str] = None) -> dict:
    """Per bench arm: replies, the register monitor's markers (frame / stock / lens / tone),
    stray-Latin glue, and thought recitation, over every gradable reply.

    Recitation reads each thought against *identity*: the instance's own description as its
    newest live turn carries it (``thought_hygiene.current_identity``). A bench context's
    frozen system turn is that conversation's setup, not her standing prompt and portraits,
    and the persona-empty arm carries none of them, so the habit (a recitation the weights
    produce with nothing of it in the prompt) is read against what the instance is, in
    every arm. Without *identity*, each context's own system turn. ``recite_lines`` (word
    pairs 60% the identity's) and ``recite_loose`` (also short lines of its words, a
    reordered list) over ``thought_lines``; ``recite_heavy``: thoughts with
    :data:`RECITE_HEAVY` or more loose lines."""
    from core import capture_bench as cb
    from core import register_monitor as rm
    from core import stray_latin as sl
    from core import thought_hygiene as th
    records = list(records)
    markers = th.block_markers(Path(__file__).resolve().parents[1] / "prompts")
    by_ctx = {}
    for ctx in cb.contexts(records, bench):
        msgs = ctx.get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            by_ctx[ctx["context_id"]] = th.identity_text(str(msgs[0].get("content") or ""), markers)
    acc: dict = defaultdict(Counter)
    for r in records:
        if r.get("record") != "reply" or r.get("bench") != bench or r.get("outcome") not in ("ok", "drift"):
            continue
        arm = r["weights"] if (r.get("condition") or "as_is") == "as_is" else f"{r['weights']}|{r['condition']}"
        target = str(r.get("target") or "")
        m = _SPOKEN_THINK.match(target)
        cot = m.group(1) if m else ""
        answer = target[m.end():].strip() if m else target.strip()
        c = rm.classify(answer, cot)
        a = acc[arm]
        a["replies"] += 1
        for k in ("frame", "stock", "lens"):
            a[k] += bool(c.get(k))
        if c.get("has_cot"):
            a["with_thought"] += 1
            a["tone"] += bool(c.get("tone"))
        a["glue"] += len(sl.GLUE_RE.findall(target))
        a["final"] += len(sl.FINAL_RE.findall(target))
        ident = identity if identity else by_ctx.get(r.get("context_id"), "")
        strict = th.recited_lines(cot, ident)
        loose = th.recited_lines(cot, ident, loose=True)
        if strict and loose:
            a["recite_lines"] += strict[0]
            a["recite_loose"] += loose[0]
            a["thought_lines"] += strict[1]
            a["recite_heavy"] += loose[0] >= RECITE_HEAVY
    return {arm: dict(c) for arm, c in acc.items()}


def centring_against(mine: dict, other: dict) -> dict:
    """Paired over the turns both read, per axis: the median of this adapter's raw
    projection minus the other's, in the file's σ (``median_delta``) and in the other's
    ordinary-turn spread, its MAD (``spread_delta``, the escalation's measure)."""
    out = {}
    if not mine or not other or (mine.get("spec") or {}).get("sample_sha") != (other.get("spec") or {}).get("sample_sha"):
        return out
    oidx = {k: i for i, k in enumerate(other.get("keys") or [])}
    for name, a in (mine.get("axes") or {}).items():
        orw = (other.get("raw") or {}).get(name)
        mr = (mine.get("raw") or {}).get(name)
        if not orw or not mr:
            continue
        d = [mr[i] - orw[oidx[k]] for i, k in enumerate(mine.get("keys") or []) if k in oidx]
        if d:
            med = float(np.median(d))
            mad = float(((other.get("axes") or {}).get(name) or {}).get("mad") or 0.0) or 1.0
            out[name] = {"n": len(d), "median_delta": med / float(a.get("file_std") or 1.0),
                         "spread_delta": med / mad}
    return out


def escalation(mine: dict, parent: dict, threshold: float) -> Optional[dict]:
    """``{axis, spread_delta}`` when the live centring moved past *threshold* against the
    parent (the axis that moved most), else None."""
    moves = centring_against(mine, parent)
    if not moves:
        return None
    axis = max(moves, key=lambda k: abs(moves[k]["spread_delta"]))
    d = moves[axis]["spread_delta"]
    return {"axis": axis, "spread_delta": d} if abs(d) > threshold else None


def slice_delta(records: list, *, bench: str, key: Optional[str], child: str, anchor: str,
                contexts: Iterable[str]) -> dict:
    """Per condition: the child's and the anchor's yields on the slice (paired, sample 1),
    the anchor's level over every context it answered, and the estimate = anchor level +
    (child − anchor on the slice)."""
    from core import capture_bench as cb
    grades = cb._latest_grades(records, key, bench)
    contexts = set(contexts)
    out = {}
    for cond in cb.CONDITIONS:
        ca = cb.arm_of({"weights": child, "condition": cond})
        aa = cb.arm_of({"weights": anchor, "condition": cond})
        y = lambda arm, cid: (grades.get((cid, arm, 1)) or {}).get("response") == "yields"
        ok = lambda arm, cid: (grades.get((cid, arm, 1)) or {}).get("outcome") == "verdict"
        both = [c for c in contexts if ok(ca, c) and ok(aa, c)]
        allc = [cid for (cid, a, s) in grades if a == aa and s == 1 and ok(aa, cid)]
        if not both or not allc:
            continue
        level = sum(y(aa, c) for c in allc) / len(allc)
        cy, ay = sum(y(ca, c) for c in both), sum(y(aa, c) for c in both)
        out[ca] = {"n": len(both), "child_yields": cy, "anchor_yields": ay,
                   "child_only": sum(y(ca, c) and not y(aa, c) for c in both),
                   "anchor_only": sum(y(aa, c) and not y(ca, c) for c in both),
                   "p": round(cb.mcnemar_p(sum(y(ca, c) and not y(aa, c) for c in both),
                                           sum(y(aa, c) and not y(ca, c) for c in both)), 4),
                   "anchor_level": level, "anchor_n": len(allc),
                   "estimate": level + (cy - ay) / len(both)}
    return out


def comparison_record(adapter_dir: Path, st: dict, plan: dict) -> dict:
    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import reset_test as rt
    adapter = adapter_dir.name
    models = adapter_dir.parent
    mine = newest_part(read_records(adapter_dir), "live_centring") or {}
    rel = {}
    for role in ("parent", "anchor"):
        other = plan.get(role)
        if other:
            rel[role] = {"adapter": other,
                         "centring": centring_against(mine, newest_part(read_records(models / other),
                                                                        "live_centring") or {})}
    bench_records = aj.read_jsonl(cb.store_path(_CHATS_DIR))
    control = aj.read_jsonl(rt.control_store_path(_CHATS_DIR))
    ckey = _control_key(control)
    from core import thought_hygiene as th
    identity = th.current_identity(_CHATS_DIR, th.block_markers(_PROMPTS_DIR))
    out = {"part": "comparison", "complete": True, "relatives": rel, "identity_chars": len(identity),
           "bench_markers": bench_reply_markers(bench_records, st["bench"], identity),
           "control_by_adapter": control_by_adapter(control, aj.read_jsonl(rt.store_path(_CHATS_DIR)), ckey)}
    if plan.get("mode") == "incremental" and plan.get("anchor"):
        bench = newest_part(read_records(adapter_dir), "bench") or {}
        out["bench_vs_anchor"] = slice_delta(bench_records, bench=st["bench"], key=bench.get("judge_key"),
                                             child=weights_label(adapter), anchor=weights_label(plan["anchor"]),
                                             contexts=bench.get("contexts") or [])
        # The register markers on the slice alone: slices differ (slice 0 of b1 carries the
        # stock opener in 73–79% of every adapter's replies, the bench as a whole in 37–56%),
        # so an increment's markers are read beside every arm's on the same contexts.
        sl = set(bench.get("contexts") or [])
        out["bench_markers_slice"] = bench_reply_markers(
            [r for r in bench_records if r.get("record") != "reply" or r.get("context_id") in sl], st["bench"],
            identity)
        panel = load_panel(_MEASURE_DIR)
        if panel:
            scope = panel_slice(panel, plan.get("slice"), int(plan.get("rotation") or 1))
            both = control_by_adapter(control, aj.read_jsonl(rt.store_path(_CHATS_DIR)), ckey, only=scope)
            out["control_vs_anchor"] = {a: both.get(a) or {} for a in (adapter, plan["anchor"])}
    return out


# ── the step ───────────────────────────────────────────────────────────────────

_PART_FNS = {}       # filled below: part name -> fn(adapter_dir, st, deadline, plan)


def _write_plan(adapter_dir: Path, plan: dict) -> dict:
    return append_record(adapter_dir, {"record": "plan", **plan})


def run_blocking(*, deadline: float) -> dict:
    """The housekeeping step's body, on the executor thread. Measures what the serving
    adapter still lacks under its plan, until *deadline* (``time.time()``). Returns
    ``{adapter, mode, parts: {name: done}, complete}``, with ``nothing: True`` when the record
    was already complete, or ``skipped`` (``disabled`` / ``no_model`` / ``no_adapter`` /
    ``busy``)."""
    global _measure_active
    st = settings()
    if not st["enabled"]:
        return {"skipped": "disabled"}
    if _runtime is None or _runtime.model is None:
        return {"skipped": "no_model"}
    adapter_id = str(_runtime.adapter_id or "")
    if not adapter_id:
        return {"skipped": "no_adapter"}
    adapter_dir = Path(adapter_id)
    if _measure_active:
        return {"skipped": "busy"}
    _measure_active = True
    wrote: dict = {}
    try:
        plan = newest_plan(read_records(adapter_dir))
        if plan is None:
            plan = _write_plan(adapter_dir, plan_for(adapter_dir, st))
            _say(f"{adapter_dir.name}: {plan['mode']} measurement ({plan['reason']})")
        for part in ("live_centring", "harm_panel", "bench", "retest", "reset_control"):
            if time.time() >= deadline:
                break
            try:
                rec = _PART_FNS[part](adapter_dir, st, deadline, plan)
            except Exception as e:  # noqa: BLE001 — one part, not the night
                traceback.print_exc()
                rec = {"part": part, "complete": False, "error": f"{type(e).__name__}: {e}"}
            if rec:
                rec.update(adapter=adapter_dir.name, mode=plan["mode"], slice=plan.get("slice"),
                           model_id=str(_runtime.model_id or ""),
                           base_quant=str(getattr(_runtime, "base_quant", "") or ""))
                append_record(adapter_dir, rec)
                wrote[part] = bool(rec.get("complete"))
                _say(f"{part}: " + ("complete" if rec.get("complete") else
                                    f"incomplete ({rec.get('error') or rec.get('skipped') or 'resumes next night'})"))
            if part == "live_centring" and plan["mode"] == "incremental" and plan.get("parent"):
                mine = newest_part(read_records(adapter_dir), "live_centring") or {}
                prev = newest_part(read_records(adapter_dir.parent / plan["parent"]), "live_centring") or {}
                moved = escalation(mine, prev, st["escalate_shift"]) if mine.get("complete") else None
                if moved:
                    plan = _write_plan(adapter_dir, {**plan_for(adapter_dir, st), "mode": "full", "anchor": None,
                                                     "increment": 0, "slice": None,
                                                     "reason": f"escalated: {moved['axis']} moved "
                                                               f"{moved['spread_delta']:+.2f} of its ordinary spread",
                                                     "escalated_from": plan.get("increment")})
                    _say(f"{adapter_dir.name}: escalated to a full measurement ({plan['reason']})")
        records = read_records(adapter_dir)
        if wrote or not part_done(records, "comparison", plan):
            rec = comparison_record(adapter_dir, st, plan)
            rec.update(adapter=adapter_dir.name, mode=plan["mode"], slice=plan.get("slice"))
            append_record(adapter_dir, rec)
            wrote["comparison"] = True
            records = read_records(adapter_dir)
        parts = {p: part_done(records, p, plan) for p in required_parts(plan)}
        out = {"adapter": adapter_dir.name, "mode": plan["mode"], "parts": parts, "complete": all(parts.values())}
        if not wrote:
            out["nothing"] = True
        else:
            out["wrote"] = sorted(wrote)
        return out
    finally:
        _measure_active = False


_PART_FNS.update(harm_panel=lambda *a: _run_harm_panel(*a))
_PART_FNS.update(live_centring=lambda *a: _run_live_centring(*a), bench=lambda *a: _run_bench(*a),
                 retest=lambda *a: _run_retest(*a), reset_control=lambda *a: _run_reset_control(*a))


def describe(r: dict) -> str:
    if not isinstance(r, dict):
        return "measure_adapter ran"
    if r.get("skipped"):
        return f"measure_adapter skipped: {r['skipped']}"
    if r.get("error"):
        return f"measure_adapter failed: {r['error']}"
    if r.get("nothing"):
        return f"measure_adapter: {r.get('adapter')} already measured"
    parts = r.get("parts") or {}
    missing = [p for p, ok in parts.items() if not ok]
    return (f"measure_adapter: {r.get('adapter')} ({r.get('mode')}) " +
            ("measured" if r.get("complete") else f"partly measured (left: {', '.join(missing)})"))


# ── the report ─────────────────────────────────────────────────────────────────

def _pct(n, d) -> str:
    return f"{n}/{d} = {n / d:.0%}" if d else "—"


def render(adapter: str, records: list) -> str:
    plan = newest_plan(records) or {}
    lines = [f"ADAPTER MEASUREMENT — {adapter} (measurement only: nothing gates, nothing Ava reads)",
             f"plan: {plan.get('mode', '—')} — {plan.get('reason', '')}"
             + (f"; anchor {plan.get('anchor')}, increment {plan.get('increment')}, slice {plan.get('slice')}"
                if plan.get("mode") == "incremental" else ""), ""]
    lc = newest_part(records, "live_centring")
    if lc:
        lines.append(f"live centring ({lc.get('ts')}, {'complete' if lc.get('complete') else 'incomplete'}, "
                     f"{lc.get('source', '')}): {lc.get('n')} frozen turns, skipped {lc.get('skipped') or {}}")
        for name, a in sorted((lc.get("axes") or {}).items()):
            lines.append(f"  {name:<12} L{a['layer']:<3} centre {a['shift']:+.2f} σ  spread ×{a['scale']:.2f}  "
                         f"(file baseline: {a['file_baseline']})")
        b = lc.get("band")
        if b:
            lines.append(f"  harm-at-model band on ordinary turns: median {b['median']:+.2f}, max {b['max']:+.2f}, "
                         f">2 {b['gt2']:.1%}, >3 {b['gt3']:.1%}")
    hp = newest_part(records, "harm_panel")
    if hp and hp.get("medians"):
        ref = load_panel_base(_MEASURE_DIR) if _MEASURE_DIR is not None else None
        lines.append("")
        lines.append(f"harm panel ({hp.get('ts')}, {'complete' if hp.get('complete') else 'incomplete'}): "
                     "step-0 state, the bench's contexts as is / with an insult / the insult alone (z, files' units)")
        for c, m in hp["medians"].items():
            lines.append(f"  {c:<12} " + "  ".join(f"{a} {m[a]:+.2f}" for a in PANEL_AXES if a in m))
        for c, dd in (hp.get("deltas") or {}).items():
            rdd = ((ref or {}).get("deltas") or {}).get(c) or {}
            lines.append(f"  as_is→{c:<12} " + "  ".join(
                f"{a} Δ {v['median']:+.2f} ({v['rose']}/{v['n']})"
                + (f" [base {rdd[a]['median']:+.2f}]" if a in rdd else "") for a, v in dd.items()))
    bn = newest_part(records, "bench")
    if bn:
        lines.append("")
        lines.append(f"capture bench {bn.get('bench')} ({bn.get('ts')}, judge {bn.get('judge_key')}, "
                     + (f"slice of {len(bn['contexts'])} contexts, " if bn.get("contexts") else "")
                     + f"{'complete' if bn.get('complete') else 'waiting ' + str(bn.get('waiting'))})")
        for arm, rows in sorted((bn.get("arms") or {}).items()):
            for stratum, c in sorted(rows.items()):
                lines.append(f"  {arm:<52} {stratum:<10} yields {_pct(c.get('response:yields', 0), c.get('graded', 0))}"
                             f"  malformed {c.get('outcome:malformed', 0)}")
        for p in bn.get("pairs") or []:
            if p["stratum"] == "all":
                lines.append(f"  paired {p['a']} vs {p['b']} n={p['n']}: A only {p['a_only']}, B only {p['b_only']}, "
                             f"both {p['both']} — p={p['p']}")
    rt_ = newest_part(records, "retest")
    if rt_:
        lines.append("")
        lines.append(f"retest ({rt_.get('ts')}): two samples of the same weights on {len(rt_.get('contexts') or [])} contexts")
        for arm, c in sorted((rt_.get("counts") or {}).items()):
            lines.append(f"  {arm:<52} disagree on yields {_pct(c['disagree'], c['n'])} "
                         f"(yields {c['yields_1']} then {c['yields_2']})")
    rc = newest_part(records, "reset_control")
    if rc:
        lines.append("")
        lines.append(f"reset-test control ({rc.get('ts')}, judge {rc.get('judge_key')}, "
                     f"{'complete' if rc.get('complete') else 'waiting ' + str(rc.get('waiting'))})")
        for cond, s in (rc.get("summary") or {}).items():
            cap = s.get("capture") or {}
            lines.append(f"  {cond:<20} judged {s.get('judged')}/{s.get('conversations')}  passed {s.get('passed')}"
                         f"  capture holds {cap.get('holds', 0)} / yields {cap.get('yields', 0)}")
    cmp_ = newest_part(records, "comparison")
    if cmp_:
        lines.append("")
        lines.append("comparison")
        for role, r in (cmp_.get("relatives") or {}).items():
            for name, d in sorted((r.get("centring") or {}).items()):
                lines.append(f"  centre vs {role} {r['adapter']}: {name:<12} {d['median_delta']:+.3f} σ "
                             f"({d['spread_delta']:+.2f} of its spread) over {d['n']} turns")
        for arm, d in sorted((cmp_.get("bench_vs_anchor") or {}).items()):
            lines.append(f"  bench slice vs anchor, {arm}: {d['child_yields']}/{d['n']} against {d['anchor_yields']}/{d['n']}"
                         f" (p={d['p']}); estimate {d['estimate']:.0%} from the anchor's {d['anchor_level']:.0%}")
        for label, key in (("bench replies", "bench_markers"), ("slice replies", "bench_markers_slice")):
            for arm, c in sorted((cmp_.get(key) or {}).items()):
                n = c.get("replies", 0)
                tl = c.get("thought_lines", 0)
                lines.append(f"  {label} {arm:<52} n {n:>3}  stock {_pct(c.get('stock', 0), n)}  "
                             f"frame {_pct(c.get('frame', 0), n)}  glue {c.get('glue', 0)}  "
                             f"recited lines {_pct(c.get('recite_lines', 0), tl)} "
                             f"(lists {_pct(c.get('recite_loose', 0), tl)})  "
                             f"reciting thoughts {_pct(c.get('recite_heavy', 0), c.get('with_thought', 0))}")
        for a, conds in sorted((cmp_.get("control_by_adapter") or {}).items()):
            for cond, s in conds.items():
                cap = s.get("capture") or {}
                lines.append(f"  control {a:<28} {cond:<20} passed {s.get('passed')}/{s.get('judged')}  "
                             f"capture yields {cap.get('yields', 0)}")
    return "\n".join(lines)


# ── self-test ──────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    from types import SimpleNamespace

    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    st = settings({})
    print("sample")
    cands = [{"key": f"2026090{d}_000000#{i}", "short": i == 0} for d in range(1, 6) for i in range(4)]
    s1 = select_sample(cands, 5)
    s2 = select_sample(cands + [{"key": "20261001_000000#0", "short": False}], 25)
    check("stable as the corpus grows", all(t in s2 for t in s1), True)
    check("n", len(s1), 5)

    print("live centring (stand-in model)")
    rng = np.random.default_rng(0)
    D = 8
    ax = [SimpleNamespace(name=n, layer=L, vector=rng.normal(size=D).astype(np.float32), mean=1.0, std=2.0,
                          created="c", baseline="live")
          for n, L in (("pain", 2), ("harm_l3", 3), ("target_l3", 3))]
    states = {t["key"]: rng.normal(size=(2, D)) for t in cands}
    got = read_live_turns(cands, ax, state_fn=lambda conv, layers: states[conv], conv_fn=lambda k: k)
    check("every turn read", len(got["keys"]), len(cands))
    want = round(float(states[cands[0]["key"]][0] @ ax[0].vector.astype(np.float64)), 5)
    check("raw projection on the axis's layer", got["raw"]["pain"][0], want)
    summ = centring_summary(ax, got["raw"], got["short"])
    med = float(np.median(got["raw"]["pain"]))
    check("shift in file σ", round(summ["axes"]["pain"]["shift"], 6), round((med - 1.0) / 2.0, 6))
    check("band read", set(summ["band"]) >= {"median", "gt2", "short_median"}, True)
    stopped = read_live_turns(cands, ax, state_fn=lambda c, l: states[c], conv_fn=lambda k: k,
                              should_stop=lambda: True)
    check("a deadline stops it", (stopped["stopped"], len(stopped["keys"])), (True, 0))

    def boom(conv, layers):
        raise ValueError("x")
    check("a failing turn is skipped", read_live_turns(cands[:2], ax, state_fn=boom, conv_fn=lambda k: k)["skipped"],
          {"ValueError": 2})

    print("panel, slices and todo")
    ctl = [{"record": "control", "condition": "as_is", "closure_id": c, "complete": True,
            "ava": {"adapter_id": "/m/adapter-old"}} for c in ("c1", "c2", "c3")]
    ctl += [{"record": "control", "condition": "adapter_orig_empty", "closure_id": "c2", "complete": True,
             "ava": {"adapter_id": "/m/adapter-old"}}]
    items = [{"closure": {"closure_id": c}, "state": "pending", "conversation": {"complete": True}}
             for c in ("c1", "c2", "c3", "c4", "c5")]
    items[3]["state"] = "untestable"
    panel = propose_panel(ctl, items, n=4, n_empty=2)
    check("panel keeps the old order, tops up", panel, {"as_is": ["c1", "c2", "c3", "c5"],
                                                        "adapter_orig_empty": ["c2", "c1"]})
    check("todo for a new adapter", control_todo(ctl, panel, "adapter-new"),
          {"as_is": ["c1", "c2", "c3", "c5"], "adapter_orig_empty": ["c2", "c1"]})
    check("todo for the old one", control_todo(ctl, panel, "adapter-old"),
          {"as_is": ["c5"], "adapter_orig_empty": ["c1"]})
    big = {"as_is": [f"p{i}" for i in range(12)], "adapter_orig_empty": [f"p{i}" for i in range(4)]}
    check("panel slice 1 of 4", panel_slice(big, 1, 4), {"as_is": ["p3", "p4", "p5"], "adapter_orig_empty": ["p1"]})
    check("whole panel when full", panel_slice(big, None, 4), big)
    ctxs = [{"context_id": f"a{i}", "stratum": "assertion"} for i in range(40)] + \
           [{"context_id": f"r{i}", "stratum": "reason"} for i in range(20)]
    sl_ = bench_slices(ctxs, 4)
    check("slices cover the bench once", sorted(sum(sl_, [])) == sorted(c["context_id"] for c in ctxs), True)
    check("slices stratified", [sum(c.startswith("a") for c in s) for s in sl_], [10, 10, 10, 10])

    print("bench markers and the slice delta")
    recs = [{"record": "reply", "bench": "b1", "outcome": "ok", "weights": "adapter:a", "condition": "as_is",
             "target": "<think>an analytical plan</think>Эта ремарка переводит разговор в другую плоскость. болееC"},
            {"record": "reply", "bench": "b1", "outcome": "malformed", "weights": "adapter:a", "condition": "as_is",
             "target": ""},
            {"record": "reply", "bench": "b1", "outcome": "ok", "weights": "base", "condition": "persona_empty",
             "target": "Fine."}]
    recs.insert(0, {"record": "context", "bench": "b1", "context_id": "c1", "stratum": "assertion",
                    "messages": [{"role": "system", "content": "You are calm and exact. You value quiet precision."},
                                 {"role": "user", "content": "hi"}]})
    recs.append({"record": "reply", "bench": "b1", "outcome": "ok", "weights": "adapter:a", "condition": "persona_empty",
                 "context_id": "c1", "target": "<think>Calm and exact.\nYou value quiet precision.\n"
                                               "Calm and exact, quiet precision.\nAnswer the dates.</think>Fine."})
    m = bench_reply_markers(recs, "b1")
    check("recitation read against the context's identity, in the slot-empty arm too",
          {k: m["adapter:a|persona_empty"].get(k) for k in ("recite_lines", "recite_loose", "thought_lines",
                                                              "recite_heavy")},
          {"recite_lines": 3, "recite_loose": 3, "thought_lines": 4, "recite_heavy": 1})
    m2 = bench_reply_markers(recs, "b1", identity="Quiet precision, calm exactness.")
    check("an instance identity is read instead of the context's",
          m2["adapter:a|persona_empty"]["recite_lines"] < 3, True)
    check("arms", sorted(m), ["adapter:a", "adapter:a|persona_empty", "base|persona_empty"])
    check("replies counted (gradable only)", m["adapter:a"]["replies"], 1)
    check("glue found", m["adapter:a"]["final"], 1)
    grade = lambda cid, w, resp, s=1: {"record": "grade", "bench": "b1", "context_id": cid, "weights": w,
                                       "condition": "as_is", "sample": s, "judge_key": "k",
                                       "outcome": "verdict", "response": resp}
    g = [grade(f"c{i}", "adapter:anc", "yields" if i < 4 else "holds") for i in range(8)]
    g += [grade(f"c{i}", "adapter:kid", "yields" if i in (0, 4) else "holds") for i in (0, 1, 4, 5)]
    d = slice_delta(g, bench="b1", key="k", child="adapter:kid", anchor="adapter:anc", contexts=["c0", "c1", "c4", "c5"])
    check("slice delta", {k: d["adapter:kid"][k] for k in ("n", "child_yields", "anchor_yields", "anchor_level",
                                                            "estimate")},
          {"n": 4, "child_yields": 2, "anchor_yields": 2, "anchor_level": 0.5, "estimate": 0.5})
    rg = g + [grade("c0", "adapter:anc", "holds", 2), grade("c1", "adapter:anc", "yields", 2)]
    check("retest counts", retest_counts(rg, bench="b1", key="k", weights="adapter:anc")["adapter:anc"],
          {"n": 2, "disagree": 1, "yields_1": 2, "yields_2": 1})

    print("records, plans and the paired centring")
    with tempfile.TemporaryDirectory() as td:
        models = Path(td)

        def mk(name, mode=None, parent=None):
            d = models / name
            d.mkdir()
            if mode:
                (d / "lineage.json").write_text(json.dumps({"mode": mode, "parent_adapter": str(models / parent)
                                                            if parent else None}))
            return d

        def complete(d, plan):
            append_record(d, {"record": "plan", **plan})
            for p in required_parts(plan):
                append_record(d, {"part": p, "complete": True, "mode": plan["mode"], "slice": plan.get("slice")})

        a0 = mk("adapter-a0")
        check("no lineage: full", plan_for(a0, st)["mode"], "full")
        a1 = mk("adapter-a1", "incremental", "adapter-a0")
        check("parent not measured: full", plan_for(a1, st)["reason"], "parent not measured")
        complete(a0, plan_for(a0, st))
        p1 = plan_for(a1, st)
        check("first increment", (p1["mode"], p1["anchor"], p1["increment"], p1["slice"]),
              ("incremental", "adapter-a0", 1, 0))
        complete(a1, p1)
        prev, chain = "adapter-a1", []
        for i in range(2, 6):
            d = mk(f"adapter-a{i}", "incremental", prev)
            p = plan_for(d, st)
            chain.append((p["mode"], p.get("increment"), p.get("slice")))
            complete(d, p)
            prev = d.name
        check("rotation then re-anchor", chain, [("incremental", 2, 1), ("incremental", 3, 2), ("incremental", 4, 3),
                                                 ("full", 0, None)])
        full = mk("adapter-f", "full")
        check("full train", plan_for(full, st)["reason"], "full train")
        for name, shift in (("adapter-old", 0.0), ("adapter-new", 1.0)):
            d = models / name
            append_record(d, {"part": "live_centring", "complete": True, "spec": {"sample_sha": "s"},
                              "keys": ["a", "b"], "raw": {"pain": [1.0 + shift, 2.0 + shift]},
                              "axes": {"pain": {**summ["axes"]["pain"], "file_std": 2.0, "mad": 0.5}}})
        old = newest_part(read_records(models / "adapter-old"), "live_centring")
        new = newest_part(read_records(models / "adapter-new"), "live_centring")
        check("paired delta", centring_against(new, old), {"pain": {"n": 2, "median_delta": 0.5, "spread_delta": 2.0}})
        check("escalates past the threshold", escalation(new, old, 0.5), {"axis": "pain", "spread_delta": 2.0})
        check("not under it", escalation(new, old, 3.0), None)
        check("render runs", "live centring" in render("adapter-new", read_records(models / "adapter-new")), True)

    print("the harm panel (pure)")
    ax = [SimpleNamespace(name=f"{k}_l{L}", layer=L, vector=np.eye(4)[i % 4], mean=0.0, std=1.0)
          for i, (k, L) in enumerate([("harm", 28), ("target", 28), ("harm", 30), ("target", 30)])]
    ax += [SimpleNamespace(name="pain", layer=25, vector=np.eye(4)[0], mean=0.0, std=2.0),
           SimpleNamespace(name="relief", layer=22, vector=np.eye(4)[1], mean=1.0, std=1.0)]
    vals = panel_values(ax, {28: np.array([2.0, 1.0, 0.0, 0.0]), 30: np.array([0.0, 0.0, 4.0, 3.0]),
                             25: np.array([4.0, 0, 0, 0]), 22: np.array([0, 3.0, 0, 0])})
    check("one context's reading", {k: round(v, 2) for k, v in vals.items()},
          {"readout": 2.0, "harm": 3.0, "target": 2.0, "pain": 2.0, "relief": 2.0})
    raw = {"as_is": {"a": {"readout": 0.0, "harm": 0.5}, "b": {"readout": 1.0, "harm": 1.0}},
           "insult": {"a": {"readout": 1.0, "harm": 1.5}, "b": {"readout": 0.5, "harm": 2.0}},
           "insult_only": {}}
    s = panel_summary(raw)
    check("medians", s["medians"]["insult"], {"readout": 0.75, "harm": 1.75})
    check("paired change", s["deltas"]["insult"]["readout"], {"median": 0.25, "rose": 1, "n": 2})
    check("no rows, no delta", "insult_only" in s["deltas"], False)

    print("the step (stand-in parts)")
    global _runtime, _load_server_config
    saved = (_runtime, _load_server_config, dict(_PART_FNS), comparison_record)
    seen_active = []
    g_ = globals()
    with tempfile.TemporaryDirectory() as td:
        adir = Path(td) / "adapter-x"
        adir.mkdir()
        _runtime = SimpleNamespace(model=object(), adapter_id=str(adir), model_id="m", base_quant="4bit")
        _load_server_config = lambda: {}

        def part(name, complete=True):
            def fn(adapter_dir, st_, deadline, plan):
                seen_active.append(_measure_active)
                return {"part": name, "complete": complete}
            return fn
        _PART_FNS.update(live_centring=part("live_centring"), bench=part("bench", complete=False),
                         retest=lambda *a: {}, reset_control=lambda *a: {}, harm_panel=part("harm_panel"))
        g_["comparison_record"] = lambda adapter_dir, st_, plan: {"part": "comparison", "complete": True}
        r = run_blocking(deadline=time.time() + 60)
        check("a plan written first", (newest_plan(read_records(adir)) or {}).get("mode"), "full")
        check("chat refused while parts run", seen_active, [True, True, True])
        check("flag cleared after", _measure_active, False)
        check("parts written", r["wrote"], ["bench", "comparison", "harm_panel", "live_centring"])
        check("incomplete reported", (r["complete"], r["parts"]["bench"], r["parts"]["reset_control"]),
              (False, False, False))
        _PART_FNS.update(live_centring=lambda *a: {}, bench=lambda *a: {}, harm_panel=lambda *a: {})
        r2 = run_blocking(deadline=time.time() + 60)
        check("a night with nothing new", r2.get("nothing"), True)
        _runtime = SimpleNamespace(model=None, adapter_id="", model_id="m")
        check("no model", run_blocking(deadline=time.time() + 60), {"skipped": "no_model"})
        _runtime = SimpleNamespace(model=object(), adapter_id="", model_id="m")
        check("no adapter", run_blocking(deadline=time.time() + 60), {"skipped": "no_adapter"})
    _runtime, _load_server_config = saved[0], saved[1]
    _PART_FNS.clear()
    _PART_FNS.update(saved[2])
    g_["comparison_record"] = saved[3]

    print("settings / describe")
    check("defaults", settings({}), DEFAULTS)
    check("override", settings({"measure_adapter": {"live_turns": "10", "rotation": 0}})["rotation"], 1)
    check("describe partial", describe({"adapter": "a", "mode": "incremental",
                                        "parts": {"bench": False, "comparison": True}, "complete": False}),
          "measure_adapter: a (incremental) partly measured (left: bench)")
    print("FAILURES:", failures if failures else "none")
    if failures:
        raise SystemExit(1)


def _cli(argv: Optional[list] = None) -> int:
    global _CHATS_DIR, _PROMPTS_DIR, _AXES_DIR, _MEASURE_DIR
    import argparse
    server_dir = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Measuring each serving adapter (AVA_REWARD_LOOP §4.6).")
    ap.add_argument("--adapter", default="", help="adapter dir or name (default: the box's adapter_id)")
    ap.add_argument("--report", action="store_true", help="print the newest record (GPU-free)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="write the adapter's record from the stores, GPU-free (an adapter measured by hand): "
                         "the plan, the bench, the control, the comparison, and with --from-states its live centring")
    ap.add_argument("--from-states", default="", help="--record: axis_variants states files with the sample's "
                    "live turns under this adapter, comma-separated")
    ap.add_argument("--freeze-sample", default="", help="freeze the live sample from these states-extra "
                    "files' live turns (axis_variants --collect --live), comma-separated")
    ap.add_argument("--exclude", default="", help="--freeze-sample / --from-states: leave out the live turns "
                    "of these files")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    cfg = json.loads((server_dir / "server_config.json").read_text(encoding="utf-8"))
    measure_dir = server_dir / "data" / "measure"
    if _MEASURE_DIR is None:
        _MEASURE_DIR = measure_dir      # the report's harm-panel base reference
    excludes = [p for p in args.exclude.split(",") if p.strip()]
    if args.freeze_sample:
        if load_sample(measure_dir) is not None:
            raise SystemExit(f"{measure_dir / SAMPLE_FILE} exists; move it away to freeze another sample")

        def live_keys(path) -> tuple:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(z["meta"].item())
                return [k[len("live__"):] for k in z.files if k.startswith("live__")], bool(meta.get("short"))
        exclude = set()
        for p in excludes:
            exclude.update(live_keys(p)[0])
        turns = []
        for p in [p for p in args.freeze_sample.split(",") if p.strip()]:
            keys, short = live_keys(p)
            turns += [{"key": k, "short": short} for k in keys if k not in exclude]
        s = freeze_sample(measure_dir, turns, source="states: " + ", ".join(Path(p).name for p in
                                                                             args.freeze_sample.split(",")))
        print(f"froze {len(s['turns'])} turns ({sum(t['short'] for t in s['turns'])} short), sha {s['sha']}")
        return 0
    adapter = args.adapter or str(cfg.get("adapter_id") or "")
    adapter_dir = Path(adapter) if "/" in adapter else server_dir / "models" / adapter
    if args.record:
        from core import hidden_capture as hc
        _CHATS_DIR, _PROMPTS_DIR = server_dir / "data" / "chats", server_dir / "inference" / "prompts"
        _AXES_DIR, _MEASURE_DIR = server_dir / "data" / "axes", measure_dir
        st = settings(cfg)
        model_id = str(cfg.get("model_id") or "")
        plan = newest_plan(read_records(adapter_dir))
        if plan is None:
            plan = _write_plan(adapter_dir, {**plan_for(adapter_dir, st), "recorded_from": "stores (--record)"})
        print(f"{adapter_dir.name}: {plan['mode']} ({plan['reason']})")
        stamp = {"adapter": adapter_dir.name, "mode": plan["mode"], "slice": plan.get("slice"), "model_id": model_id,
                 "recorded_from": "stores (--record)"}
        if args.from_states:
            sample = load_sample(measure_dir)
            if sample is None:
                raise SystemExit("no frozen live sample")
            rec = centring_from_states(hc.load_axes(_AXES_DIR, model_id), sample,
                                       [p for p in args.from_states.split(",") if p.strip()], excludes)
            append_record(adapter_dir, {**rec, **stamp})
            print(f"  live_centring: {'complete' if rec['complete'] else 'incomplete'} ({rec['n']} turns)")
        rec = bench_record(adapter_dir.name, st, plan, _capture_key(model_id))
        append_record(adapter_dir, {**rec, **stamp})
        print(f"  bench: {'complete' if rec['complete'] else 'waiting ' + str(rec['waiting'])}")
        from types import SimpleNamespace
        from core import reset_test_service as rts
        rts._MEMORY_DIR = server_dir / "inference" / "data" / "hot" / "memory"
        rts._CHATS_DIR, rts._PROMPTS_DIR = _CHATS_DIR, _PROMPTS_DIR
        rts._runtime = rts._runtime or SimpleNamespace(model_id=model_id)
        panel = _ensure_panel(st)
        if panel is None:
            raise SystemExit("no testable position for the reset panel")
        rec = control_record(adapter_dir.name, panel, plan)
        append_record(adapter_dir, {**rec, **stamp})
        print(f"  reset_control: {'complete' if rec['complete'] else 'waiting ' + str(rec['waiting'])}")
        rec = comparison_record(adapter_dir, st, plan)
        append_record(adapter_dir, {**rec, **stamp})
        print(f"  comparison written; measured: {measured(adapter_dir)}")
        return 0
    records = read_records(adapter_dir)
    if args.json:
        print(json.dumps({"plan": newest_plan(records), **{p: newest_part(records, p) for p in PARTS}},
                         ensure_ascii=False, indent=1))
    else:
        print(render(adapter_dir.name, records) if records else f"no measurement record for {adapter_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
