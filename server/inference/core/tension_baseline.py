"""Tension baseline — a percentile rank for one exchange's CoT tension against the corpus.

PROMPT_REWRITE.md §2 (stage 1). The transcripts carry a per-exchange ``tension`` block
(:mod:`core.tension`: per-segment ``median_entropy`` / ``contested_frac`` / ``p10_margin``
over the CoT and the answer). Raw, those numbers say almost nothing about whether she was
*torn*: entropy and near-ties are dominated by the reply's language (Russian tokenization
produces more near-ties than English, whatever is being thought), by the model family, by
the adapter that shaped the distribution and by the sampling temperature. The
open-problems doc records exactly that gap — "no corpus baseline normalizes tension".

This module is that baseline, and nothing more: read every transcript's stored segment
stats, bucket them by ``(model_id, adapter, reply language, segment)``, and rank a value
as a **percentile within its bucket**. A bucket with too few samples falls back to the
adapter-wide bucket, then to *no baseline* — and with no baseline the consumer degrades to
whatever it did before, never to raw numbers.

Its consumer is the prompt-mutation LOCATOR (``reflection_runner``, the gate on
``_run_prompt_mutation_for_exchange``): the pass used to see only ``revise`` exchanges;
with a baseline it also sees a ``keep`` exchange whose CoT tension ranks above
``prompt_rewrite.tension_percentile`` — she stood by the reply but was unusually torn
getting there, which is the one cell of the verdict×tension square a standing prompt can
settle by taking a side (PROMPT_REWRITE.md §2).

**User chats only.** A transcript with ``interlocutor: "ai"`` (an encounter or served
gossip) is neither sampled nor ranked: a peer model's conversation does not shape her
standing prompt, and its distribution would only smear the buckets.

**Cost.** A transcript with tension carries the full per-token arrays, so a corpus is
hundreds of MB of JSON to parse for a few floats per exchange. A cache file (``stats``
per transcript, keyed on mtime + size) makes the second build a stat per file; the first
is seconds. Pure / GPU-free. Self-test: ``python -m core.tension_baseline``; the same
entry point with ``--chats DIR`` prints the bucket table and how many ``keep``
exchanges would clear a percentile on a real corpus — the measurement §8 stage 1 asks
for before anything spends on this signal.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from core.chat_sidecar import is_chat_session_json

#: fewer stored segment stats than this in a bucket ⇒ fall back to the next wider key
MIN_SAMPLES = 30
#: a CoT shorter than this (tokens) carries no stable median — not sampled, not ranked
MIN_SEGMENT_TOKENS = 20
#: The CoT metrics a rank is taken over, each with the direction that means "more torn"
#: (``+1``: higher is more torn; ``-1``: lower is). Chosen on the live corpus (2026-09-18,
#: 425 CoT segments): the stored ``median_entropy`` is ~0 and ``median_margin`` is 1.0 on
#: EVERY segment — the median token of a thought is decided outright — so both carry
#: nothing and are not here. What varies: the spike (``peak_entropy``, p10 1.16 → p90
#: 1.73), the diffuse baseline read off the raw series as a MEAN (``mean_entropy``, 0.063
#: → 0.100; derived at scan time since the median is degenerate), the near-tie share
#: (``contested_frac``, 0.003 → 0.013) and the low-end margin (``p10_margin``, 0.73 →
#: 0.92, LOWER = more friction). An exchange's rank is the MEAN of its metrics'
#: directional percentiles, not the max: four 20% tails unioned by a max would clear
#: ~half the corpus at the 0.8 mark, while a mean asks the thought to be torn on the
#: whole, which is the signal wanted.
METRICS = {"peak_entropy": +1, "mean_entropy": +1, "contested_frac": +1, "p10_margin": -1}
SEGMENT = "cot"

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def detect_lang(text: str) -> str:
    """``ru`` if any Cyrillic, else ``en`` — the same two-way split
    ``reflection_writer._detect_lang`` uses (kept local: this module imports nothing
    that loads a model, and the writer is not a leaf)."""
    return "ru" if _CYRILLIC_RE.search(text or "") else "en"


def is_user_chat(session: dict) -> bool:
    """False for an encounter / served-gossip transcript (``interlocutor: "ai"``)."""
    return (str(session.get("interlocutor") or "").strip().lower() != "ai")


def adapter_name(session: dict) -> str:
    """The adapter's bare dir name (``""`` = bare base) — the level at which the
    distribution changes, which is what a bucket must key on."""
    aid = session.get("adapter_id") or ""
    return os.path.basename(str(aid).rstrip("/")) if aid else ""


def exchange_lang(ex: dict) -> str:
    """Language of the REPLY side (CoT + answer): what the tokenization statistics are a
    property of. The user's turn may be in the other language."""
    return detect_lang((ex.get("assistant_cot") or "") + " " + (ex.get("assistant_response") or ""))


def exchange_stats(ex: dict) -> Optional[dict]:
    """The CoT segment's stored stats for one exchange, or None when there is no
    usable CoT tension (no block, no CoT segment, or one too short to trust)."""
    block = ex.get("tension")
    if not isinstance(block, dict):
        return None
    seg = block.get(SEGMENT)
    if not isinstance(seg, dict):
        return None
    try:
        n = int(seg.get("n_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if n < MIN_SEGMENT_TOKENS:
        return None
    out = {"n_tokens": n}
    for m in METRICS:
        v = seg.get(m)
        if v is None:
            continue
        try:
            out[m] = float(v)
        except (TypeError, ValueError):
            continue
    # `mean_entropy` is not a stored stat: derive it from the raw per-token series the
    # block carries (aligned with `token_ids`; the CoT is the first n_tokens of it).
    ents = block.get("entropies")
    if isinstance(ents, list) and len(ents) >= n and n > 0:
        try:
            cot = [float(e) for e in ents[:n]]
            out["mean_entropy"] = sum(cot) / len(cot)
        except (TypeError, ValueError):
            pass
    if not any(m in out for m in METRICS):
        return None
    return out


# ── the corpus scan (cached) ─────────────────────────────────────────────────

def _fingerprint(path: Path) -> list:
    st = path.stat()
    return [int(st.st_mtime), int(st.st_size)]


def _scan_transcript(path: Path) -> Optional[dict]:
    """Per-transcript record for the cache: the session key + one stats row per
    exchange with usable CoT tension. None for an unreadable file or a non-user chat."""
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(session, dict) or not is_user_chat(session):
        return None
    rows = []
    for i, ex in enumerate(session.get("exchanges") or []):
        if not isinstance(ex, dict):
            continue
        st = exchange_stats(ex)
        if st is None:
            continue
        # The block records the model that generated it; the header is the fallback.
        mid = ((ex.get("tension") or {}).get("model_id") or session.get("model_id") or "")
        rows.append({
            "index": i,
            "model_id": str(mid),
            "adapter": adapter_name(session),
            "lang": exchange_lang(ex),
            **st,
        })
    return {"rows": rows}


def collect(chats_dirs: Iterable[Path], cache_path: Optional[Path] = None) -> list[dict]:
    """Every usable CoT-tension row across *chats_dirs*, through the cache when given.

    The cache maps ``<stem>`` → ``{fp, rows}``; a transcript whose mtime+size match is
    not re-parsed. Entries for files that no longer exist are dropped on write."""
    cache: dict = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cache = {}
    fresh: dict = {}
    rows: list[dict] = []
    for d in chats_dirs:
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if not is_chat_session_json(path):
                continue
            try:
                fp = _fingerprint(path)
            except OSError:
                continue
            key = path.stem
            hit = cache.get(key)
            if isinstance(hit, dict) and hit.get("fp") == fp and isinstance(hit.get("rows"), list):
                rec = {"fp": fp, "rows": hit["rows"]}
            else:
                scanned = _scan_transcript(path)
                rec = {"fp": fp, "rows": scanned["rows"] if scanned else []}
            fresh[key] = rec
            for r in rec["rows"]:
                rows.append({"session": path.name, **r})
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_path)
        except Exception:
            pass    # the cache is a convenience; a failed write costs the next scan
    return rows


# ── the baseline ─────────────────────────────────────────────────────────────

def _key(model_id: str, adapter: str, lang: Optional[str]) -> tuple:
    return (model_id, adapter, lang)


@dataclass
class Baseline:
    """Sorted samples per bucket per metric; ``rank`` is a percentile lookup."""
    buckets: dict = field(default_factory=dict)    # key → metric → sorted list[float]
    min_samples: int = MIN_SAMPLES

    @classmethod
    def from_rows(cls, rows: Iterable[dict], *, min_samples: int = MIN_SAMPLES) -> "Baseline":
        raw: dict = {}
        for r in rows:
            for key in cls._chain(r["model_id"], r["adapter"], r["lang"]):
                for m in METRICS:
                    if m in r:
                        raw.setdefault(key, {}).setdefault(m, []).append(float(r[m]))
        buckets = {k: {m: sorted(v) for m, v in ms.items()} for k, ms in raw.items()}
        return cls(buckets=buckets, min_samples=min_samples)

    def n(self, model_id: str, adapter: str, lang: Optional[str], metric: str) -> int:
        return len(self.buckets.get(_key(model_id, adapter, lang), {}).get(metric, ()))

    @staticmethod
    def _chain(model_id: str, adapter: str, lang: Optional[str]) -> list[tuple]:
        """Bucket keys narrowest-first: (model, adapter, lang) → (model, adapter) →
        (model, lang) → (model). The model-wide tail exists for a FRESH adapter: a new
        build has no samples of its own for days, and no baseline at all would close the
        keep cell exactly when the newest weights are the ones being read."""
        keys = [_key(model_id, adapter, lang), _key(model_id, adapter, None)]
        if adapter:
            keys += [_key(model_id, "", lang), _key(model_id, "", None)]
        out = []
        for k in keys:
            if k not in out:
                out.append(k)
        return out

    def _resolve(self, model_id: str, adapter: str, lang: str, metric: str) -> Optional[tuple]:
        """The narrowest bucket with enough samples along :meth:`_chain`, else None."""
        for key in self._chain(model_id, adapter, lang):
            xs = self.buckets.get(key, {}).get(metric)
            if xs and len(xs) >= self.min_samples:
                return key, xs
        return None

    @staticmethod
    def _percentile(xs: list[float], v: float) -> float:
        """Mid-rank percentile in [0, 1]: ties count half, so a value equal to every
        sample ranks 0.5, not 1.0."""
        lo = bisect_left(xs, v)
        hi = bisect_right(xs, v)
        return (lo + (hi - lo) / 2.0) / len(xs)

    def rank_stats(self, stats: dict, *, model_id: str, adapter: str, lang: str) -> Optional[dict]:
        """Rank one exchange's CoT stats. Returns None when NO metric has a bucket
        (no baseline); else ``{rank, ranks: {metric: pct}, key, n}`` where each metric's
        percentile is DIRECTIONAL (1.0 = most torn on that metric), ``rank`` is their
        mean, and ``key``/``n`` describe the narrowest bucket used."""
        ranks: dict = {}
        used: Optional[tuple] = None
        n_used = 0
        chain = self._chain(model_id, adapter, lang)
        for m, direction in METRICS.items():
            if m not in stats:
                continue
            res = self._resolve(model_id, adapter, lang, m)
            if res is None:
                continue
            key, xs = res
            pct = self._percentile(xs, float(stats[m]))
            ranks[m] = pct if direction > 0 else 1.0 - pct
            if used is None or chain.index(key) < chain.index(used):
                used, n_used = key, len(xs)
        if not ranks:
            return None
        return {
            "rank": sum(ranks.values()) / len(ranks),
            "ranks": ranks,
            "key": {"model_id": used[0], "adapter": used[1] or None, "lang": used[2]},
            "n": n_used,
        }

    def rank_exchange(self, ex: dict, session: dict) -> Optional[dict]:
        """:meth:`rank_stats` over a transcript exchange + its session header. None for a
        non-user chat, an exchange without usable CoT tension, or no baseline."""
        if not is_user_chat(session):
            return None
        st = exchange_stats(ex)
        if st is None:
            return None
        mid = ((ex.get("tension") or {}).get("model_id") or session.get("model_id") or "")
        return self.rank_stats(st, model_id=str(mid), adapter=adapter_name(session),
                               lang=exchange_lang(ex))

    def table(self) -> list[dict]:
        """One row per (bucket, metric) with its sample count — for the CLI."""
        out = []
        for key, ms in sorted(self.buckets.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
            for m, xs in sorted(ms.items()):
                out.append({"model_id": key[0], "adapter": key[1] or "(base)",
                            "lang": key[2] or "*", "metric": m, "n": len(xs),
                            "p50": xs[len(xs) // 2], "p80": xs[int(len(xs) * 0.8)] if xs else None})
        return out


def build(chats_dirs: Iterable[Path], cache_path: Optional[Path] = None,
          *, min_samples: int = MIN_SAMPLES) -> Baseline:
    """Scan (through the cache) and fold: the one call a consumer makes per run."""
    return Baseline.from_rows(collect(chats_dirs, cache_path), min_samples=min_samples)


# ── CLI / self-test ──────────────────────────────────────────────────────────

def _measure(chats_dirs: list[Path], percentile: float, cache_path: Optional[Path]) -> None:
    """Print the bucket table and, per transcript sidecar verdict, how many exchanges
    clear *percentile* — the §8 stage-1 measurement."""
    from core.chat_sidecar import ChatSidecar
    rows = collect(chats_dirs, cache_path)
    bl = Baseline.from_rows(rows)
    print(f"{len(rows)} CoT-tension samples from user chats (raw distributions; "
          f"p10_margin is ranked LOW = torn)")
    for r in bl.table():
        print(f"  {r['model_id'][:40]:40} {r['adapter'][:28]:28} {r['lang']:2} "
              f"{r['metric']:15} n={r['n']:5} p50={r['p50']:.3f} p80={r['p80']:.3f}")
    counts = {"keep": [0, 0], "revise": [0, 0], "none": [0, 0]}   # [cleared, ranked]
    for d in chats_dirs:
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if not is_chat_session_json(path):
                continue
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                verdicts = ChatSidecar(d).load(path.name).get("exchanges") or {}
            except Exception:
                verdicts = {}
            for i, ex in enumerate(session.get("exchanges") or []):
                rk = bl.rank_exchange(ex, session)
                if rk is None:
                    continue
                v = (verdicts.get(i) or verdicts.get(str(i)) or {})
                verdict = str((v.get("verdict") if isinstance(v, dict) else "") or "none")
                verdict = verdict if verdict in counts else "none"
                counts[verdict][1] += 1
                if rk["rank"] >= percentile:
                    counts[verdict][0] += 1
    print(f"exchanges clearing the {percentile:.0%} CoT-tension percentile, by revision verdict:")
    for k, (cleared, ranked) in counts.items():
        print(f"  {k:7} {cleared:5} of {ranked:5} ranked")
    print("  ('keep' cleared = the locator's new cell; 'none' = not yet reflected)")


def _selftest() -> None:
    import tempfile

    # Percentile: mid-rank, ties count half.
    xs = [1.0, 2.0, 3.0, 4.0]
    assert Baseline._percentile(xs, 4.0) == 0.875, Baseline._percentile(xs, 4.0)
    assert Baseline._percentile(xs, 0.5) == 0.0
    assert Baseline._percentile([2.0] * 10, 2.0) == 0.5, "all-ties ranks 0.5"

    # Fallback chain: a thin language bucket falls back to the adapter-wide one, a fresh
    # adapter to the model-wide one; a model with no samples at all is no baseline.
    rows = ([{"model_id": "m", "adapter": "a", "lang": "en", "peak_entropy": float(i), "p10_margin": 0.5}
             for i in range(40)]
            + [{"model_id": "m", "adapter": "a", "lang": "ru", "peak_entropy": 100.0, "p10_margin": 0.9}
               for _ in range(3)])
    bl = Baseline.from_rows(rows)
    r = bl.rank_stats({"peak_entropy": 39.0, "p10_margin": 0.5}, model_id="m", adapter="a", lang="en")
    assert r and r["key"]["lang"] == "en" and r["ranks"]["peak_entropy"] > 0.95, r
    r = bl.rank_stats({"peak_entropy": 39.0, "p10_margin": 0.5}, model_id="m", adapter="a", lang="ru")
    assert r and r["key"] == {"model_id": "m", "adapter": "a", "lang": None}, ("thin ru bucket → adapter-wide", r)
    assert r["n"] == 43, r
    r = bl.rank_stats({"peak_entropy": 39.0}, model_id="m", adapter="fresh", lang="en")
    assert r and r["key"] == {"model_id": "m", "adapter": None, "lang": "en"}, ("fresh adapter → model-wide", r)
    assert bl.rank_stats({"peak_entropy": 1.0}, model_id="other", adapter="a", lang="en") is None, "no baseline"
    # Direction: a LOW p10_margin is more torn; rank is the MEAN of the directional percentiles.
    r = bl.rank_stats({"peak_entropy": 39.0, "p10_margin": 0.0}, model_id="m", adapter="a", lang="en")
    assert r and r["ranks"]["p10_margin"] == 1.0 and r["ranks"]["peak_entropy"] == 0.9875, r
    assert abs(r["rank"] - (1.0 + 0.9875) / 2) < 1e-9, r
    r = bl.rank_stats({"peak_entropy": 0.0, "p10_margin": 0.0}, model_id="m", adapter="a", lang="en")
    assert r and abs(r["rank"] - (0.0125 + 1.0) / 2) < 1e-9, ("one calm metric halves the rank", r)

    # `mean_entropy` is derived from the raw series (the CoT = the first n_tokens).
    st = exchange_stats({"tension": {"cot": {"peak_entropy": 1.0, "n_tokens": 20},
                                     "entropies": [0.5] * 20 + [9.0] * 5}})
    assert st and abs(st["mean_entropy"] - 0.5) < 1e-9 and "median_entropy" not in st, st

    # Exchange-level: language from the reply side, adapter from the header basename,
    # short CoT / no block / AI interlocutor ⇒ None.
    session = {"model_id": "m", "adapter_id": "/x/models/a", "exchanges": []}
    ex = {"assistant_cot": "думаю", "assistant_response": "ответ",
          "tension": {"cot": {"peak_entropy": 39.0, "p10_margin": 0.5, "n_tokens": 50}}}
    r = bl.rank_exchange(ex, session)
    assert r and r["key"] == {"model_id": "m", "adapter": "a", "lang": None}, r   # ru → fallback
    assert exchange_lang(ex) == "ru"
    assert bl.rank_exchange({**ex, "tension": {"cot": {"peak_entropy": 39.0, "n_tokens": 5}}}, session) is None
    assert bl.rank_exchange({"assistant_cot": "x"}, session) is None
    assert bl.rank_exchange(ex, {**session, "interlocutor": "ai"}) is None, "AI transcripts are never ranked"

    # Corpus scan through the cache: a user chat is sampled, an AI one is not, a sidecar is
    # not a transcript, and the second collect re-parses nothing (the cache carries rows).
    with tempfile.TemporaryDirectory() as d:
        chats = Path(d) / "chats"
        chats.mkdir()
        (chats / "20260101_000000.json").write_text(json.dumps({**session, "exchanges": [ex, ex]}), encoding="utf-8")
        (chats / "20260101_000001.json").write_text(json.dumps({**session, "interlocutor": "ai", "exchanges": [ex]}), encoding="utf-8")
        (chats / "20260101_000000.state.json").write_text(json.dumps({"exchanges": [ex]}), encoding="utf-8")
        cache = Path(d) / "cache.json"
        rows1 = collect([chats], cache)
        assert len(rows1) == 2 and rows1[0]["session"] == "20260101_000000.json" and rows1[0]["adapter"] == "a", rows1
        assert cache.exists()
        saved = json.loads(cache.read_text())
        assert set(saved) == {"20260101_000000", "20260101_000001"} and saved["20260101_000001"]["rows"] == []
        # Poison the cached rows: a cache hit must be served without re-parsing.
        saved["20260101_000000"]["rows"][0]["peak_entropy"] = -1.0
        cache.write_text(json.dumps(saved))
        rows2 = collect([chats], cache)
        assert rows2[0]["peak_entropy"] == -1.0, "cache hit must not re-parse"
        # Touching the file (new size) invalidates it.
        (chats / "20260101_000000.json").write_text(json.dumps({**session, "exchanges": [ex, ex, ex]}), encoding="utf-8")
        rows3 = collect([chats], cache)
        assert len(rows3) == 3 and rows3[0]["peak_entropy"] == 39.0, rows3
        assert build([chats], cache).n("m", "a", "ru", "peak_entropy") == 3
    print("tension_baseline self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chats", type=Path, action="append",
                    help="transcripts dir (repeatable); omitted ⇒ run the self-test")
    ap.add_argument("--percentile", type=float, default=0.8)
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args()
    if args.chats:
        _measure(args.chats, args.percentile, args.cache)
    else:
        _selftest()
