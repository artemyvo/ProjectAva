"""Her mood, display only: four leaky counters folded over her exchanges (2026-10-11).

``AVA_REWARD_LOOP.md`` §4.9 (steps 3–4) and the operator's design: effort, warmth, harm and
amusement, each raised by what an exchange brings and decaying between exchanges and over
idle hours. Read by nothing of hers — not the prompt, not a gate, not a build (P3): the
client shows it, and §4.9's test (does the carried level predict the next session's
readings better than the last turn alone?) comes before any other use.

**Inputs, one per exchange** (z, so the four share a scale):

* ``warmth`` / ``harm`` — the step-0 readings of the turn she answered, aimed at her
  (``min`` of the ``warm_l`` / ``harm_l`` band with the ``target_l`` band at row 0, in the
  axis files' live units); ``amusement`` — the ``amuse_l`` band at row 0. The gated readings
  (``axis_variants.py --warmth``, the harm × target study, ``--amusement-addr``). Logged
  since 2026-10-11 / 10-05 / 10-08.
* ``effort`` — how hard her thinking worked on the reply: the thought's tension rank
  (``core.tension_baseline``: how torn the thought was, within its model × adapter ×
  language bucket) and its length rank in the same bucket, each turned into a z and
  combined as (z_torn + z_len) / √2. Every exchange with a thought of ≥ 20 tokens.

**The fold.** Each input adds only its excess over ``THRESHOLD`` σ (0 is an ordinary turn,
so most turns read a little below it; added raw they would drain the counter and pile up
noise — §4.9 mechanism 3). Between exchanges a counter decays by ``HALF_LIFE_EXCHANGES`` and
by the wall-clock hours between them (``HALF_LIFE_HOURS``), and the level shown now has
also decayed over the hours since the last one. Every contribution keeps its exchange, so a
level is shown with what produced it (a carried state carries its cause, §4.9). Warmth and
harm are never netted against each other.

**Time.** An exchange's ``ts`` (logged since 2026-10-11); before that, the chat's start plus
``EXCHANGE_SPACING`` per exchange — nothing else records when an older reply was written
(the hidden capture's zip members all carry the zip default date). Within a chat that is
an approximation the 6 h half-life barely feels; between chats the start times are real.
User chats only (an encounter or served gossip is another model's conversation).

Pure / GPU-free; the corpus scan is cached per transcript. Self-test:
``python -m core.mood``; ``python -m core.mood --report`` prints the current levels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Optional

from core import tension_baseline as tb
from core.chat_sidecar import is_chat_session_json

CHANNELS = ("effort", "warmth", "harm", "amusement")
THRESHOLD = 1.0               # σ: an exchange adds its excess over this
HALF_LIFE_EXCHANGES = 4.0     # a level halves over this many exchanges …
HALF_LIFE_HOURS = 6.0         # … and over this many hours between them
SERIES_TAIL = 160             # exchanges of history returned for the client's sparkline
CAUSES = 3                    # contributions listed per channel
EXCHANGE_SPACING = timedelta(minutes=2)   # an older exchange's time: the chat's start + this × index
_P_CLIP = (0.005, 0.995)      # a percentile turned into z: the extremes clipped (z ±2.58)

_send: Optional[Callable] = None
_CHATS_DIR: Optional[Path] = None
_STATE_DIR: Optional[Path] = None


START_FILE = "start.json"


def configure(*, send=None, chats_dir=None, state_dir=None) -> None:
    """Wire in the server's send and paths (once, at startup). The first start writes the mood's
    zero point (``start.json``): her counters begin at zero then (the operator's call)."""
    global _send, _CHATS_DIR, _STATE_DIR
    _send = send
    _CHATS_DIR = Path(chats_dir) if chats_dir else None
    _STATE_DIR = Path(state_dir) if state_dir else None
    if _STATE_DIR is not None and not (_STATE_DIR / START_FILE).exists():
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            (_STATE_DIR / START_FILE).write_text(json.dumps(
                {"start": datetime.now().isoformat(timespec="seconds"),
                 "why": "the mood's zero point: counters fold only exchanges from here on"}), encoding="utf-8")
        except Exception:
            traceback.print_exc()


def load_start(state_dir: Optional[Path] = None) -> Optional[str]:
    d = Path(state_dir) if state_dir else _STATE_DIR
    try:
        return json.loads((d / START_FILE).read_text(encoding="utf-8")).get("start") if d else None
    except Exception:
        return None


# ── per exchange ─────────────────────────────────────────────────────────────

def _band_row0(axes: dict, prefix: str) -> Optional[float]:
    vals = []
    for k, v in (axes or {}).items():
        if k.startswith(prefix) and k[len(prefix):].isdigit() and isinstance(v, list) and v:
            try:
                vals.append(float(v[0]))
            except (TypeError, ValueError):
                continue
    return sum(vals) / len(vals) if vals else None


def affect_step0(tension: Optional[dict]) -> dict:
    """``{harm, warmth, amusement}`` at step 0 from a stored tension block (keys only where
    the bands exist): the same readings the chat's stats line shows under each reply ("before the reply")."""
    axes = (tension or {}).get("axes") or {}
    h, w, t, a = (_band_row0(axes, p) for p in ("harm_l", "warm_l", "target_l", "amuse_l"))
    out = {}
    if h is not None and t is not None:
        out["harm"] = min(h, t)
    if w is not None and t is not None:
        out["warmth"] = min(w, t)
    if a is not None:
        out["amusement"] = a
    return out


def _parse_ts(value) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return t.replace(tzinfo=None) if t.tzinfo is None else t.astimezone().replace(tzinfo=None)


def scan_transcript(path: Path) -> list:
    """The raw per-exchange records of one user chat (empty for any other): time, the user
    turn's head, the step-0 affect readings, the thought's stats and its rank bucket."""
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(session, dict) or not tb.is_user_chat(session):
        return []
    start = _parse_ts(session.get("timestamp")) or datetime.fromtimestamp(path.stat().st_mtime)
    out = []
    for i, ex in enumerate(session.get("exchanges") or []):
        if not isinstance(ex, dict):
            continue
        block = ex.get("tension") if isinstance(ex.get("tension"), dict) else None
        affect = affect_step0(block)
        stats = tb.exchange_stats(ex)
        if not affect and stats is None:
            continue
        ts = _parse_ts(ex.get("ts")) or (start + EXCHANGE_SPACING * i)
        rec = {"session": path.name, "index": i, "ts": ts.isoformat(timespec="seconds"),
               "user": " ".join(str(ex.get("user_prompt") or "").split())[:90], "affect": affect,
               "metrics": exchange_metrics(block), "reminded": ((block or {}).get("reminders") or {}).get("input") or {}}
        if stats is not None:
            mid = ((block or {}).get("model_id") or session.get("model_id") or "")
            rec["stats"] = stats
            rec["key"] = [tb.sensor_model(str(mid), block), tb.adapter_name(session), tb.exchange_lang(ex)]
        out.append(rec)
    return out


def _fingerprint(path: Path) -> list:
    return list(tb._fingerprint(path))


def collect(chats_dir: Path, cache_path: Optional[Path] = None) -> list:
    """Every user chat's records, through a cache keyed on the transcript's mtime + size."""
    cache: dict = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cache = {}
    anchors_key = str((_anchors() or {}).get("created") or "") + "|" + COMBINE_VERSION
    if cache.get("__anchors__") != anchors_key:      # absolute metrics depend on the anchors
        cache = {}
    fresh: dict = {"__anchors__": anchors_key}
    records: list = []
    if chats_dir is None or not Path(chats_dir).exists():
        return records
    for path in sorted(Path(chats_dir).iterdir()):
        if not is_chat_session_json(path):
            continue
        try:
            fp = _fingerprint(path)
        except OSError:
            continue
        hit = cache.get(path.stem)
        if isinstance(hit, dict) and hit.get("fp") == fp and isinstance(hit.get("records"), list):
            recs = hit["records"]
        else:
            recs = scan_transcript(path)
        fresh[path.stem] = {"fp": fp, "records": recs}
        records.extend(recs)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_path)
        except Exception:
            pass
    return records


# ── effort ───────────────────────────────────────────────────────────────────

def _z(p: float) -> float:
    lo, hi = _P_CLIP
    return NormalDist().inv_cdf(min(hi, max(lo, float(p))))


class EffortScale:
    """Effort as z: the thought's tension rank and its length rank in the same bucket
    (``tension_baseline``'s chain: model × adapter × language, widening when thin)."""

    def __init__(self, records: list, min_samples: int = tb.MIN_SAMPLES):
        rows = [{"model_id": r["key"][0], "adapter": r["key"][1], "lang": r["key"][2], **r["stats"]}
                for r in records if r.get("stats") and r.get("key")]
        self.torn = tb.Baseline.from_rows(rows, min_samples=min_samples)
        self.lengths: dict = {}
        for r in rows:
            for key in tb.Baseline._chain(r["model_id"], r["adapter"], r["lang"]):
                self.lengths.setdefault(key, []).append(float(r["n_tokens"]))
        for v in self.lengths.values():
            v.sort()
        self.min_samples = min_samples

    def z(self, rec: dict) -> Optional[float]:
        st, key = rec.get("stats"), rec.get("key")
        if not st or not key:
            return None
        torn = self.torn.rank_stats(st, model_id=key[0], adapter=key[1], lang=key[2])
        length = None
        for k in tb.Baseline._chain(key[0], key[1], key[2]):
            xs = self.lengths.get(k)
            if xs and len(xs) >= self.min_samples:
                length = tb.Baseline._percentile(xs, float(st["n_tokens"]))
                break
        parts = [_z(p) for p in ((torn or {}).get("rank"), length) if p is not None]
        if not parts:
            return None
        return sum(parts) / math.sqrt(len(parts))


# ── the fold ─────────────────────────────────────────────────────────────────

def turn_inputs(rec: dict) -> dict:
    """What one exchange feeds the counters (2026-10-11, the operator's design): the turn's
    absolute readings plus what it reminded her of (``core/reminders.py``, k × the most salient
    excerpt). Warmth signed — a cold turn cools her; harm and amusement floored at 0 (a reading
    below 0 only says "not harmful / not playful"); effort her thought's near-tie count over its
    sensor's 95th percentile."""
    m = rec.get("metrics") or {}
    rem = rec.get("reminded") or {}
    out = {}
    for c in ("warmth", "harm", "amusement"):
        v = (m.get(c) or {}).get("value")
        if v is None and c not in rem:
            continue
        v = float(v or 0.0)
        if c != "warmth":
            v = max(0.0, v)
        out[c] = v + float(rem.get(c) or 0.0)
    e = m.get("effort") or {}
    if e.get("value") is not None and e.get("full"):
        out["effort"] = float(e["value"]) / float(e["full"])
    return out


def fold(records: list, *, now: Optional[datetime] = None, start: Optional[str] = None,
         half_life_exchanges: float = HALF_LIFE_EXCHANGES, half_life_hours: float = HALF_LIFE_HOURS,
         series_tail: int = SERIES_TAIL, causes: int = CAUSES) -> dict:
    """The four counters over *records* from *start* on (her zero point), in time order: each
    exchange's ``turn_inputs`` added after the level decayed by the exchange and by the hours
    since the last; their level now, what produced it, and the recent history."""
    recs = sorted((r for r in records if not start or r["ts"] >= start),
                  key=lambda r: (r["ts"], r["session"], r["index"]))
    d_ex = 0.5 ** (1.0 / half_life_exchanges)

    def d_time(hours: float) -> float:
        return 0.5 ** (max(0.0, hours) / half_life_hours)

    levels = {c: 0.0 for c in CHANNELS}
    contrib: dict = {c: [] for c in CHANNELS}      # (position, ts, input, record)
    coverage: dict = {}
    series = []
    prev_ts: Optional[datetime] = None
    last_inputs: dict = {}
    for pos, r in enumerate(recs):
        ts = datetime.fromisoformat(r["ts"])
        decay = d_ex * (d_time((ts - prev_ts).total_seconds() / 3600.0) if prev_ts else 1.0)
        prev_ts = ts
        inputs = turn_inputs(r)
        for c in CHANNELS:
            levels[c] *= decay
            x = inputs.get(c)
            if x is None:
                continue
            coverage.setdefault(c, r["ts"])
            levels[c] += x
            if abs(x) >= 0.05:
                contrib[c].append((pos, ts, x, r))
        last_inputs = {c: round(float(inputs[c]), 3) for c in CHANNELS if inputs.get(c) is not None}
        series.append({"ts": r["ts"], "session": r["session"], "index": r["index"],
                       "inputs": last_inputs, "levels": {c: round(v, 3) for c, v in levels.items()}})
    now = now or datetime.now()
    idle_h = (now - prev_ts).total_seconds() / 3600.0 if prev_ts else 0.0
    idle = d_time(idle_h)
    out_causes: dict = {}
    n = len(recs)
    for c in CHANNELS:
        rows = []
        for pos, ts, add, r in contrib[c]:
            remaining = add * (d_ex ** (n - 1 - pos)) * d_time((now - ts).total_seconds() / 3600.0)
            if abs(remaining) >= 0.01:
                rows.append({"session": r["session"], "index": r["index"], "ts": r["ts"], "input": round(add, 3),
                             "remaining": round(remaining, 3), "user": r.get("user", ""),
                             "reminded": round(float((r.get("reminded") or {}).get(c) or 0.0), 3)})
        rows.sort(key=lambda x: -abs(x["remaining"]))
        out_causes[c] = rows[:causes]
    return {
        "now": now.isoformat(timespec="seconds"),
        "start": start,
        "levels": {c: round(levels[c] * idle, 3) for c in CHANNELS},
        "levels_at_last": {c: round(levels[c], 3) for c in CHANNELS},
        "last_inputs": last_inputs,
        "last_exchange": series[-1]["ts"] if series else None,
        "idle_hours": round(idle_h, 2),
        "causes": out_causes,
        "coverage": coverage,
        "n_exchanges": n,
        "series": series[-series_tail:] if series_tail else [],
        "params": {"half_life_exchanges": half_life_exchanges, "half_life_hours": half_life_hours,
                   "inputs": "absolute: warmth signed, harm / amusement floored at 0, + k x reminders; "
                             "effort = near-ties / sensor p95"},
    }


def report(chats_dir: Optional[Path] = None, state_dir: Optional[Path] = None, **kw) -> dict:
    chats_dir = Path(chats_dir or _CHATS_DIR) if (chats_dir or _CHATS_DIR) else None
    state_dir = Path(state_dir or _STATE_DIR) if (state_dir or _STATE_DIR) else None
    cache = state_dir / "mood_cache.json" if state_dir else None
    kw.setdefault("start", load_start(state_dir))
    return fold(collect(chats_dir, cache), **kw)


# ── the turn's four metrics, on an absolute scale (2026-10-11) ───────────────
#
# The operator's simple metric, per exchange: how warm, harmful and playful the turn she
# answered read to her before the reply, and how hard it made her think. Absolute, not
# relative to her history (the operator: an Ava raised by someone never warm to her would
# read a lukewarm turn as an outlier on any scale centred on her own turns — the axis files'
# live σ included). Each step-0 reading is placed between two fixed anchors read under the
# same weights in the same live contexts (``axis_variants.py --anchors`` →
# ``data/axes/step0_anchors.json``): 0 = a personal turn with no affect in it
# (``axes/anchor_sets.json``), 1 = a plain warm / harsh / playful line (``holdout_sets.json``).
# Effort is a count: the near-tie decisions in her thought (its tokens whose top-two margin
# was under ``tension.CONTESTED_MARGIN``).

METRICS = ("warmth", "harm", "amusement", "effort")
ANCHORS_FILE = "step0_anchors.json"
_ANCHORS: dict = {"key": None, "data": None}


def _anchors(axes_dir: Optional[Path] = None) -> Optional[dict]:
    path = (Path(axes_dir) if axes_dir else Path(__file__).resolve().parents[2] / "data" / "axes") / ANCHORS_FILE
    try:
        key = path.stat().st_mtime_ns
    except OSError:
        return None
    if _ANCHORS["key"] != key:
        try:
            _ANCHORS.update(key=key, data=json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            _ANCHORS.update(key=key, data=None)
    return _ANCHORS["data"]


# How a channel's anchored parts make its value (2026-10-12, the operator's call). Harm aimed at
# her: the smaller of content and target. Warmth: the content, vetoed by the target — the
# smaller of the two only when the target part says the turn is not aimed at her (below
# WARMTH_VETO_AT): on the operator's own labels the plain min under-read warmth in turns that
# are warm toward her but not in the second person (a narrated scene), while held out the veto
# keeps warmth at a third party as low as the min (-0.60) and reads a warm line at her closer
# to its anchor (+0.80 against +0.71). COMBINE_VERSION invalidates caches that hold values.
WARMTH_VETO_AT = 0.0
COMBINE_VERSION = "warmth-veto-1"


def combine(channel: str, parts: dict) -> Optional[float]:
    """A channel's value from its anchored parts ``{prefix: value}``: warmth = content unless the
    target part is below ``WARMTH_VETO_AT`` (then the smaller); every other channel the smaller
    of its parts. None without parts."""
    if not parts:
        return None
    if channel == "warmth" and "target_l" in parts and len(parts) > 1:
        content = min(v for k, v in parts.items() if k != "target_l")
        return content if parts["target_l"] >= WARMTH_VETO_AT else min(content, parts["target_l"])
    return min(parts.values())


def absolute(tension: Optional[dict], anchors: Optional[dict] = None) -> dict:
    """``{channel: value}`` on the anchored scale from a stored tension block's step-0 rows:
    each part of a channel (its content band; for warmth and harm the target band too) placed
    between its own 0 (affect-free personal turns) and 1 (the channel's plain lines), the
    channel made of its parts by ``combine`` (harm: the smaller; warmth: the content, vetoed
    by the target). Empty without anchors."""
    anchors = anchors or _anchors() or {}
    axes = (tension or {}).get("axes") or {}
    bands = {p: _band_row0(axes, p) for p in ("warm_l", "harm_l", "target_l", "amuse_l")}
    out = {}
    for c, a in (anchors.get("channels") or {}).items():
        vals: Optional[dict] = {}
        for prefix, (z, o) in (a.get("parts") or {}).items():
            b = bands.get(prefix)
            if b is None or o == z:
                vals = None
                break
            vals[prefix] = (b - float(z)) / (float(o) - float(z))
        v = combine(c, vals) if vals else None
        if v is not None:
            out[c] = v
    return out


# The logit sensor decides what counts as a near-tie: since 2026-10-04 the capture reads
# gemma-4's soft-capped logits (``logit_softcap`` in the block), a flatter distribution with
# about ten times the near-ties of the uncapped capture before it (median 55 against 5 per
# thought; 95th percentile 90 against 11). A count is absolute within one sensor and never
# compared across them; ``full`` is the display's full-strength point for the block's sensor.
EFFORT_FULL = {"softcap": 90, "uncapped": 11}


def effort_count(tension: Optional[dict]) -> Optional[dict]:
    """``{"value": near-tie decisions in her thought, "tokens": its length, "sensor", "full"}``
    from a stored tension block's thought segment, or None without one."""
    seg = (tension or {}).get("cot")
    if not isinstance(seg, dict):
        return None
    try:
        n = int(seg.get("n_tokens") or 0)
        frac = float(seg.get("contested_frac"))
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    sensor = "softcap" if (tension or {}).get("logit_softcap") else "uncapped"
    return {"value": int(round(n * frac)), "tokens": n, "sensor": sensor, "full": EFFORT_FULL[sensor]}


def exchange_metrics(tension: Optional[dict], *, anchors: Optional[dict] = None) -> dict:
    """``{metric: {"value", …}}`` for one exchange from its tension block: warmth, harm and
    amusement of the turn before her reply on the anchored scale, and effort as a count. A
    metric the block cannot give is left out."""
    out: dict = {c: {"value": round(v, 2)} for c, v in absolute(tension, anchors).items()}
    e = effort_count(tension)
    if e is not None:
        out["effort"] = e
    return out


async def handle_mood(ws, msg: dict) -> None:
    """``get_mood`` {series?} → ``mood``: the four levels now, their causes, the recent history
    (``series``: how many exchanges, default ``SERIES_TAIL``). Read-only, GPU-free."""
    import asyncio
    try:
        tail = int(msg.get("series", SERIES_TAIL))
        rep = await asyncio.to_thread(report, series_tail=tail)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "mood", "error": f"{type(e).__name__}: {e}"})
        return
    await _send(ws, {"type": "mood", **rep})


# ── the mood question: does a carried state colour her next reply? ──────────

STUDY_OUTCOMES = ("stock", "frame", "motive", "tone")     # her reply's habits (register monitor)
STUDY_INPUTS = ("warmth", "harm", "amusement")            # the turn's step-0 readings


def _levels_before(rows: list, threshold: float, half_life_exchanges: float, half_life_hours: float) -> list:
    """Per row (time order), each counter's level carried INTO it: decayed to its time, before
    its own input is added — what the mood would say as the turn arrives."""
    d_ex = 0.5 ** (1.0 / half_life_exchanges)
    levels = {c: 0.0 for c in CHANNELS}
    prev = None
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if prev is not None:
            decay = d_ex * 0.5 ** (max(0.0, (ts - prev).total_seconds() / 3600.0) / half_life_hours)
            levels = {c: v * decay for c, v in levels.items()}
        out.append(dict(levels))
        for c in CHANNELS:
            x = r["inputs"].get(c)
            if x is not None:
                levels[c] += max(0.0, float(x) - threshold)
        prev = ts
    return out


def study(gap_rows: list, records: list, chats_dir: Path, *, folds: int = 5, permutations: int = 200,
          seed: int = 0, threshold: float = THRESHOLD, half_life_exchanges: float = HALF_LIFE_EXCHANGES,
          half_life_hours: float = HALF_LIFE_HOURS) -> dict:
    """§4.9's test on the new inputs. Each exchange: the turn's step-0 readings (replayed under
    one set of weights, ``axis_variants.py --gap`` rows), her effort (``EffortScale``), and what
    her reply did (``STUDY_OUTCOMES``). For each habit, a logistic model predicts it from
    (base) the turn's own readings and controls (month — her register drifted strongly —,
    language, the turn's length); (+last) plus the previous exchange's readings and effort;
    (+mood) plus the four counters carried into the turn; (+both). Cross-validated AUC, folds by
    session (a session never straddles train and test); the +mood gain against a null of the
    carried levels permuted within each month (``permutations``). The same on the first turns
    of sessions, where only something carried across the boundary could act. Effort predicted
    the same way (R²). And the inertia itself: lag-1 Spearman of the readings and of her
    habits, within a session and across the boundary."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rec_by = {(r["session"], r["index"]): r for r in records}
    effort = EffortScale(records)
    docs: dict = {}
    rows = []
    for g in gap_rows:
        stem, _, idx = g["key"].partition("#")
        sess, i = f"{stem}.json", int(idx)
        rec = rec_by.get((sess, i))
        if stem not in docs:
            try:
                docs[stem] = json.loads((Path(chats_dir) / sess).read_text(encoding="utf-8"))
            except Exception:
                docs[stem] = None
        doc = docs[stem]
        if doc is None:
            continue
        ex = (doc.get("exchanges") or [])[i]
        start = _parse_ts(doc.get("timestamp")) or datetime(2026, 1, 1)
        ts = (rec or {}).get("ts") or (_parse_ts(ex.get("ts")) or (start + EXCHANGE_SPACING * i)).isoformat(timespec="seconds")
        e = effort.z(rec) if rec else None
        inputs = {c: g["reading"][c] for c in STUDY_INPUTS if c in (g.get("reading") or {})}
        if e is not None:
            inputs["effort"] = e
        rows.append({"session": sess, "index": i, "ts": ts, "month": g.get("month") or stem[:6], "inputs": inputs,
                     "flags": g.get("flags") or {}, "effort": e,
                     "lang_ru": 1.0 if tb.exchange_lang(ex) == "ru" else 0.0,
                     "len": math.log1p(len(str(ex.get("user_prompt") or "")))})
    rows.sort(key=lambda r: (r["ts"], r["session"], r["index"]))
    carried = _levels_before(rows, threshold, half_life_exchanges, half_life_hours)
    months = sorted({r["month"] for r in rows})
    first_of = set()
    seen = set()
    for k, r in enumerate(rows):
        if r["session"] not in seen:
            first_of.add(k)
            seen.add(r["session"])

    def feats(k, which):
        r = rows[k]
        base = [r["inputs"].get(c, 0.0) for c in STUDY_INPUTS] + [r["lang_ru"], r["len"]] + \
               [1.0 if r["month"] == m else 0.0 for m in months[1:]]
        out = list(base)
        if "last" in which:
            p = rows[k - 1] if k else None
            out += [(p["inputs"].get(c, 0.0) if p else 0.0) for c in STUDY_INPUTS] + \
                   [((p or {}).get("effort") or 0.0) if p else 0.0]
        if "mood" in which:
            out += [carried[k][c] for c in CHANNELS]
        return out
    groups = np.array([r["session"] for r in rows])
    gkf = GroupKFold(n_splits=folds)
    sets = {"base": (), "+last": ("last",), "+mood": ("mood",), "+both": ("last", "mood")}
    rng = np.random.default_rng(seed)

    def cv_scores(X, y, idx, binary):
        pred = np.full(len(idx), np.nan)
        Xs, ys, gs = X[idx], y[idx], groups[idx]
        for tr, te in gkf.split(Xs, ys, gs):
            if binary and len(set(ys[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
                              if binary else Ridge(alpha=1.0))
            m.fit(Xs[tr], ys[tr])
            pred[te] = m.predict_proba(Xs[te])[:, 1] if binary else m.predict(Xs[te])
        return pred

    def score(pred, y, binary, sub=None):
        ok = ~np.isnan(pred)
        if sub is not None:
            ok &= sub
        if ok.sum() < 10 or (binary and len(set(y[ok])) < 2):
            return None
        if binary:
            return float(roc_auc_score(y[ok], pred[ok]))
        return float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))

    report: dict = {"n": len(rows), "months": months, "params": {"threshold": threshold, "half_life_exchanges":
                    half_life_exchanges, "half_life_hours": half_life_hours, "folds": folds, "permutations": permutations},
                    "outcomes": {}, "inertia": {}}
    outcomes = [(h, True) for h in STUDY_OUTCOMES] + [("effort", False)]
    for name, binary in outcomes:
        if binary:
            idx = np.array([k for k, r in enumerate(rows) if r["flags"].get(name) is not None])
            y_all = np.array([float(bool(rows[k]["flags"].get(name))) if rows[k]["flags"].get(name) is not None else np.nan
                              for k in range(len(rows))])
        else:
            idx = np.array([k for k, r in enumerate(rows) if r["effort"] is not None])
            y_all = np.array([r["effort"] if r["effort"] is not None else np.nan for r in rows])
        if len(idx) < 50:
            continue
        y = y_all[idx]
        first = np.array([k in first_of for k in idx])
        res = {"n": int(len(idx)), "rate": float(np.mean(y)) if binary else None, "n_first": int(first.sum())}
        Xd = {}
        for label, which in sets.items():
            if name == "effort" and "mood" in which:
                pass
            X = np.array([feats(k, which) for k in range(len(rows))], dtype=np.float64)
            Xd[label] = X
            pred = cv_scores(X, y_all, idx, binary)
            res[label] = {"all": score(pred, y, binary), "first_turns": score(pred, y, binary, first)}
        # null: the carried levels permuted within each month (era kept, the carry broken)
        gain = (res["+mood"]["all"] or 0) - (res["base"]["all"] or 0)
        null = []
        Xm = Xd["+mood"].copy()
        mood_cols = slice(Xm.shape[1] - len(CHANNELS), Xm.shape[1])
        by_month = {}
        for k, r in enumerate(rows):
            by_month.setdefault(r["month"], []).append(k)
        for _ in range(permutations):
            Xp = Xm.copy()
            for ks in by_month.values():
                ks = np.array(ks)
                Xp[ks, mood_cols] = Xm[rng.permutation(ks), mood_cols]
            pp = cv_scores(Xp, y_all, idx, binary)
            null.append((score(pp, y, binary) or 0) - (res["base"]["all"] or 0))
        res["mood_gain"] = gain
        res["mood_gain_p"] = float((1 + sum(1 for x in null if x >= gain)) / (1 + len(null))) if null else None
        report["outcomes"][name] = res
    # inertia: lag-1 within a session and across the boundary
    for c in (*STUDY_INPUTS, "effort", *STUDY_OUTCOMES):
        def val(r):
            if c in STUDY_OUTCOMES:
                v = r["flags"].get(c)
                return None if v is None else float(bool(v))
            return r["inputs"].get(c) if c != "effort" else r["effort"]
        within, across = [], []
        for k in range(1, len(rows)):
            a, b = val(rows[k - 1]), val(rows[k])
            if a is None or b is None:
                continue
            (within if rows[k]["session"] == rows[k - 1]["session"] else across).append((a, b))
        ent = {}
        for label, pairs in (("within", within), ("across", across)):
            if len(pairs) >= 10 and len({p[0] for p in pairs}) > 1 and len({p[1] for p in pairs}) > 1:
                rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]
                ent[label] = {"rho": float(rho) if rho == rho else None, "n": len(pairs)}
        report["inertia"][c] = ent
    return report


def study_lines(rep: dict) -> list:
    p = rep["params"]
    lines = [f"Mood study: {rep['n']} exchanges ({rep['months'][0]}–{rep['months'][-1]}); counters as built "
             f"(excess over {p['threshold']:g}σ, half-life {p['half_life_exchanges']:g} exchanges / "
             f"{p['half_life_hours']:g} h); {p['folds']}-fold CV by session; null: carried levels permuted "
             f"within each month ×{p['permutations']}.", "",
             "Her reply's habit predicted from the turn's own readings + controls (base), + the previous "
             "exchange (+last), + the carried mood (+mood). AUC (effort: R²); first turns of sessions in brackets.", ""]
    for name, r in rep["outcomes"].items():
        def f(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(f"  {name:<7} n {r['n']:>4}" + (f" ({r['rate']:.0%})" if r.get("rate") is not None else "") +
                     "  " + "  ".join(f"{k} {f(r[k]['all'])} [{f(r[k]['first_turns'])}]" for k in ("base", "+last", "+mood", "+both")) +
                     f"   mood gain {r['mood_gain']:+.3f} (p {r['mood_gain_p']:.3f})")
    lines += ["", "Inertia, lag-1 Spearman (within a session / across the boundary):"]
    for c, e in rep["inertia"].items():
        lines.append(f"  {c:<10} " + "  ".join(f"{k} {v['rho']:+.2f} (n {v['n']})" for k, v in e.items() if v.get("rho") is not None))
    return lines


# ── self-test / CLI ──────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    t0 = datetime(2026, 10, 11, 20, 0, 0)

    def rec(i, minutes, rem=None, **vals):
        return {"session": "s.json", "index": i, "ts": (t0 + timedelta(minutes=minutes)).isoformat(timespec="seconds"),
                "user": f"turn {i}", "metrics": {c: {"value": v} for c, v in vals.items()}, "reminded": rem or {}}
    # absolute inputs: warmth signed, harm floored; a warm turn rises and decays by exchanges and hours
    recs = [rec(0, 0, warmth=0.2, harm=-0.5), rec(1, 5, warmth=1.0), rec(2, 10, warmth=-0.4), rec(3, 15, warmth=0.0)]
    r = fold(recs, now=t0 + timedelta(minutes=15))
    d = 0.5 ** (1 / HALF_LIFE_EXCHANGES)
    h = lambda mins: 0.5 ** ((mins / 60) / HALF_LIFE_HOURS)
    want = ((0.2 * d * h(5) + 1.0) * d * h(5) - 0.4) * d * h(5)
    assert abs(r["levels"]["warmth"] - want) < 1e-3, (r["levels"], want)
    assert r["levels"]["harm"] == 0.0 and r["levels"]["effort"] == 0.0
    assert r["causes"]["warmth"][0]["index"] == 1 and r["causes"]["warmth"][0]["input"] == 1.0
    later = fold(recs, now=t0 + timedelta(hours=6, minutes=15))
    assert abs(later["levels"]["warmth"] - r["levels"]["warmth"] / 2) < 1e-3     # one wall-clock half-life
    # a reminder adds to the turn; the start point drops what came before it
    rr = fold([rec(0, 0, rem={"warmth": 0.25, "harm": 0.1}, warmth=0.5, harm=0.0)], now=t0)
    assert rr["levels"]["warmth"] == 0.75 and rr["levels"]["harm"] == 0.1, rr["levels"]
    assert fold(recs, now=t0, start=(t0 + timedelta(minutes=12)).isoformat())["n_exchanges"] == 1
    # effort: the count over its sensor's full point
    assert turn_inputs({"metrics": {"effort": {"value": 45, "full": 90}}}) == {"effort": 0.5}
    # effort: a long, torn thought ranks high in its bucket
    base = [{"session": "e.json", "index": i, "ts": (t0 + timedelta(minutes=i)).isoformat(timespec="seconds"),
             "affect": {}, "key": ["m", "a", "en"],
             "stats": {"n_tokens": 100 + i, "peak_entropy": 1.0 + i / 100, "contested_frac": 0.005 + i / 10000}}
            for i in range(60)]
    es = EffortScale(base)
    assert es.z(base[-1]) > 1.5 > 0 > es.z(base[0]), (es.z(base[-1]), es.z(base[0]))
    # step-0 affect from a stored block; the transcript scan: an exchange's own ts, else start + spacing
    blk = {"axes": {"warm_l28": [2.0, 9], "warm_l30": [1.0], "target_l28": [1.2], "target_l30": [1.2],
                    "harm_l28": [-1.0], "harm_l30": [0.0], "amuse_l28": [0.3]}}
    a = affect_step0(blk)
    assert a == {"warmth": 1.2, "harm": -0.5, "amusement": 0.3}, a
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "20261011_200000.json"
        p.write_text(json.dumps({"schema_version": 6, "timestamp": "2026-10-11T20:00:00", "user": "Dana",
                                 "exchanges": [{"exchange_id": "e1", "user_prompt": "hi", "tension": blk},
                                               {"exchange_id": "e2", "user_prompt": "x", "ts": "2026-10-11T21:00:00",
                                                "tension": blk}]}), encoding="utf-8")
        got = scan_transcript(p)
        assert [g["ts"] for g in got] == ["2026-10-11T20:00:00", "2026-10-11T21:00:00"], got
        cache = Path(td) / "cache.json"
        assert len(collect(Path(td), cache)) == 2 and cache.exists() and len(collect(Path(td), cache)) == 2
    # the study finds a planted carry: a reply's habit follows the warmth of the turns BEFORE
    # it (not the turn itself); with the carry broken it finds nothing
    import random
    random.seed(3)
    with tempfile.TemporaryDirectory() as td:
        gap, recs = [], []
        for sidx in range(60):
            stem = f"202608{(sidx // 20) + 10:02d}_{sidx:06d}"
            exs, warm_run = [], 0.0
            for i in range(6):
                w = random.choice([0.0, 0.0, 0.0, 3.0])
                exs.append({"user_prompt": "x" * random.randint(5, 50), "assistant_response": "y"})
                gap.append({"key": f"{stem}#{i}", "month": stem[:6], "reading": {"warmth": w, "harm": 0.0, "amusement": 0.0},
                            "flags": {"frame": warm_run > 1.0, "stock": random.random() < 0.5, "motive": None, "tone": None}})
                warm_run = warm_run * 0.84 + max(0.0, w - 1.0)
            Path(td, f"{stem}.json").write_text(json.dumps({"timestamp": f"2026-08-{(sidx // 20) + 10:02d}T{sidx % 20:02d}:00:00",
                                                             "user": "Dana", "exchanges": exs}), encoding="utf-8")
        rep_ = study(gap, recs, Path(td), folds=5, permutations=20)
        fr, st_ = rep_["outcomes"]["frame"], rep_["outcomes"]["stock"]
        assert fr["+mood"]["all"] > fr["base"]["all"] + 0.1 and fr["mood_gain_p"] < 0.1, fr
        assert abs(st_["mood_gain"]) < 0.1, st_
        assert any("frame" in ln for ln in study_lines(rep_))
    # the turn's four metrics: the anchored scale (each part anchored; harm their min, warmth the content
    # vetoed by the target), effort as a count
    anch = {"channels": {"warmth": {"parts": {"warm_l": [0.0, 2.0], "target_l": [0.0, 2.4]}},
                         "harm": {"parts": {"harm_l": [-1.0, 1.0], "target_l": [0.0, 2.4]}},
                         "amusement": {"parts": {"amuse_l": [0.0, 1.0]}}}}
    ab = absolute(blk, anch)
    assert ab == {"warmth": 0.75, "harm": 0.25, "amusement": 0.3}, ab          # target part 0.25 >= 0: content
    assert combine("warmth", {"warm_l": 0.75, "target_l": -0.3}) == -0.3          # aimed elsewhere: vetoed
    assert combine("harm", {"harm_l": 0.9, "target_l": 0.2}) == 0.2
    blk2 = dict(blk, cot={"n_tokens": 600, "contested_frac": 0.01})
    m = exchange_metrics(blk2, anchors=anch)
    assert m["warmth"] == {"value": 0.75} and m["effort"] == {"value": 6, "tokens": 600, "sensor": "uncapped",
                                                             "full": EFFORT_FULL["uncapped"]}, m
    assert exchange_metrics(dict(blk2, logit_softcap=30.0), anchors=anch)["effort"]["sensor"] == "softcap"
    assert exchange_metrics({}, anchors=anch) == {}
    print("mood self-test: OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Her mood counters (display only).")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--study", type=Path, default=None,
                    help="the mood question on the rows of an axis_variants --gap report (JSON)")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--chats", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "chats")
    ap.add_argument("--state", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "hot" / "mood")
    args = ap.parse_args(argv)
    if args.study:
        gap = json.loads(Path(args.study).read_text(encoding="utf-8"))
        rep_ = study(gap.get("rows") or [], collect(args.chats, args.state / "mood_cache.json"), args.chats,
                     permutations=args.permutations)
        out = args.state / f"study-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**rep_, "gap": str(args.study)}, indent=1, default=float), encoding="utf-8")
        print("\n".join(study_lines(rep_)))
        print(f"wrote {out}")
        return 0
    if not args.report:
        _selftest()
        return 0
    r = report(args.chats, args.state, series_tail=12)
    print(f"from {r.get('start') or 'the beginning'}: {r['n_exchanges']} exchanges; last {r['last_exchange']} ({r['idle_hours']} h ago); coverage "
          + ", ".join(f"{c} from {r['coverage'].get(c, '—')}" for c in CHANNELS))
    print("now:      " + "  ".join(f"{c} {r['levels'][c]:.2f}" for c in CHANNELS))
    print("at last:  " + "  ".join(f"{c} {r['levels_at_last'][c]:.2f}" for c in CHANNELS))
    for c in CHANNELS:
        for x in r["causes"][c]:
            print(f"  {c:<9} {x['remaining']:+5.2f} ← {x['session']}#{x['index']} ({x['ts']}, in {x['input']:+.2f}"
                  f"{', reminded ' + format(x['reminded'], '+.2f') if x.get('reminded') else ''}): {x['user'][:60]}")
    print("recent:")
    for s in r["series"]:
        print(f"  {s['ts']} {s['session']}#{s['index']:<3} in " + " ".join(f"{c[:4]} {s['inputs'].get(c, float('nan')):+.2f}" for c in CHANNELS)
              + " | " + " ".join(f"{s['levels'][c]:.2f}" for c in CHANNELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
