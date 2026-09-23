"""Prompt patterns — the fold of logged prompt deltas into the rewrite BUDGET.

PROMPT_REWRITE.md §3 (stage 2). The prompt-mutation pass (the locator, §2) logs one
``DELTA`` per exchange it finds a gap on — a single vote, from one conversation, on what
the standing prompt fails to say. The prompt is global, so its evidence must be too: this
module groups those votes into **patterns** (the same pull proposed across distinct
chats), weights each vote by the tension of the exchange it came from, and reports which
patterns are **mature** enough to fund a rewrite. That maturity is the currency the
rewrite event (stage 3) spends; nothing here spends it.

What it reuses, deliberately:

* the grouping is :func:`core.persona_cluster.map_reduce_groups` — the clean-base
  "do these two say the same thing?" pass behind the persona digest and fact dedup — with
  its own instruction (``prompt_pattern_cluster_prompt.txt``, default-written) and items
  ordered by wording so a two-member paraphrase pair can share a block (fact dedup's
  lesson);
* the recency weight is the persona digest's (1.0 inside 30 days, linear to 0 at 180);
* the maturity shape is the digest's judge gate (weighted recurrence over ≥2 distinct
  chats), but WITHOUT the tenure discount: that discount exists for the circular
  self-vote (the persona conditions the reply that re-derives it), and a delta has no
  such loop — the line is not in the prompt yet, and once it is, its pattern is consumed.

Votes: one per DISTINCT chat per pattern (a chat proposing the same line at three
exchanges is one conversation's opinion), weight ``recency × (0.5 + tension_rank)`` — a
delta with no baseline rank counts 0.5, a top-tension one 1.5. A re-reflected exchange
re-proposes; only its newest delta counts.

Files, all under ``data/hot/prompt/``: reads ``prompt_deltas.jsonl`` (the locator's
op-log) and ``rewrite_log.jsonl`` (stage 3's attempt record — the keys a changed rewrite
CONSUMED, and the last attempt's time); writes ``patterns.json`` (derived, disposable,
rebuilt every normal reflection run inside the clean-base window). The deliberation
executive reads ``patterns.json`` only — it never re-folds, and never swaps to the clean
base.

Pure / GPU-free except :func:`cluster_patterns`, which takes an injected ``generate_fn``.
Self-test: ``python -m core.prompt_patterns``.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

PATTERNS_FILE = "patterns.json"
REWRITE_LOG_FILE = "rewrite_log.jsonl"
CLUSTER_PROMPT_FILE = "prompt_pattern_cluster_prompt.txt"

#: The digest's recency curve (reflection_digest.PERSONA_DECAY_*): full weight inside
#: 30 days, linear to nothing at 180. Copied rather than imported — that module is not a
#: leaf, and the two numbers are the whole contract.
RECENCY_FULL_DAYS = 30.0
RECENCY_ZERO_DAYS = 180.0
#: Vote weight = recency × (VOTE_BASE + tension_rank); a delta with no rank gets VOTE_BASE.
VOTE_BASE = 0.5
#: Defaults for the gate — config_schema.py's `prompt_rewrite.maturity` / `.min_chats`.
DEFAULT_MATURITY = 1.9
DEFAULT_MIN_CHATS = 2
#: Grouping block size (map_reduce_groups); deltas number dozens, not hundreds.
BLOCK_SIZE = 40

SCOPES = ("disposition", "line", "voice")

_DEFAULT_CLUSTER_PROMPT = """\
You are grouping proposed additions to your own standing prompt. Below are numbered lines,
each one something a past reflection said your prompt fails to say — written as the
instruction itself, in your voice, sometimes in different languages or phrasings.

Group the numbers so that lines asking for the SAME change — the same disposition to hold,
the same boundary to keep, the same way of sounding — are in one group, even when the
wording or language differs. Keep genuinely different changes in separate groups. Every
number appears in exactly one group; a one-of-a-kind line is a group of one. Judge by what
the line would make you do, not by its surface words.

One thing overrides surface similarity: DIRECTION. A line asking you to do LESS of
something does not belong with a line asking you to do MORE of it — same subject, opposite
direction is a different group.

Output only lines of this form, nothing else:
GROUP: <comma-separated numbers>
"""


# ── records ──────────────────────────────────────────────────────────────────

def delta_key(rec: dict) -> str:
    """Stable id of one logged delta: the exchange it came from plus the line itself."""
    basis = "|".join([
        str(rec.get("source_session") or ""),
        str(rec.get("exchange_index") if rec.get("exchange_index") is not None else ""),
        (rec.get("delta") or "").strip(),
    ])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _parse_ts(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat((ts or "").strip())
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def recency_weight(ts: str, now: datetime) -> float:
    """1.0 inside RECENCY_FULL_DAYS, linear to 0 at RECENCY_ZERO_DAYS; an unparseable or
    absent ts reads as fresh — a missing date never erases a vote."""
    dt = _parse_ts(ts)
    if dt is None:
        return 1.0
    age = max(0.0, (now - dt).total_seconds() / 86400.0)
    if age <= RECENCY_FULL_DAYS:
        return 1.0
    if age >= RECENCY_ZERO_DAYS:
        return 0.0
    return (RECENCY_ZERO_DAYS - age) / (RECENCY_ZERO_DAYS - RECENCY_FULL_DAYS)


def vote_weight(rec: dict, now: datetime) -> float:
    rank = rec.get("tension_rank")
    try:
        rank = float(rank) if rank is not None else 0.0
    except (TypeError, ValueError):
        rank = 0.0
    rank = min(max(rank, 0.0), 1.0)
    return round(recency_weight(rec.get("ts") or "", now) * (VOTE_BASE + rank), 4)


def read_deltas(prompt_dir: Path) -> list[dict]:
    """Every logged delta, oldest first (the op-log order), each stamped with its key."""
    from core.prompt_mutation import read_prompt_deltas
    rows = read_prompt_deltas(prompt_dir)      # newest first
    rows.reverse()
    out = []
    for r in rows:
        if not isinstance(r, dict) or not (r.get("delta") or "").strip():
            continue
        out.append({**r, "key": delta_key(r)})
    return out


def read_rewrite_log(prompt_dir: Path) -> list[dict]:
    """Stage 3's attempt records, oldest first. Absent file ⇒ no attempts."""
    path = Path(prompt_dir) / REWRITE_LOG_FILE
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except Exception:
        return []
    return out


def consumed_keys(attempts: Iterable[dict]) -> set[str]:
    """Delta keys a CHANGED rewrite spent. A stayed / declined attempt lists none —
    PROMPT_REWRITE.md §5: nothing is consumed unless the prompt changed."""
    keys: set[str] = set()
    for a in attempts:
        for k in a.get("consumed_keys") or ():
            keys.add(str(k))
    return keys


def last_attempt_ts(attempts: Iterable[dict]) -> str:
    return max((str(a.get("ts") or "") for a in attempts), default="")


def unconsumed(prompt_dir: Path) -> list[dict]:
    """The votes still in play: every logged delta minus the consumed ones, and — since
    a re-reflected exchange proposes again — only the NEWEST delta per exchange."""
    spent = consumed_keys(read_rewrite_log(prompt_dir))
    latest: dict = {}
    for r in read_deltas(prompt_dir):       # oldest first ⇒ later overwrite wins
        ex_id = (str(r.get("source_session") or ""), r.get("exchange_index"))
        latest[ex_id] = r
    # Newest-per-exchange FIRST, then drop the spent: the other order let an older,
    # superseded delta resurface the moment the newer one for its exchange was consumed.
    return [r for r in latest.values() if r["key"] not in spent]


# ── grouping ─────────────────────────────────────────────────────────────────

def load_cluster_prompt(prompts_dir: Path) -> str:
    """The grouping instruction; default-written on first miss so it can be retuned."""
    path = Path(prompts_dir) / CLUSTER_PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except Exception:
        return _DEFAULT_CLUSTER_PROMPT
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_CLUSTER_PROMPT, encoding="utf-8")
    except Exception:
        pass
    return _DEFAULT_CLUSTER_PROMPT


def _listing(items: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {(it.get('delta') or '').strip()}"
                     for i, it in enumerate(items))


def cluster_patterns(records: list[dict], generate_fn: Optional[Callable], *,
                     prompt: Optional[str] = None,
                     on_stage: Optional[Callable] = None) -> tuple[list[list[dict]], dict]:
    """Partition *records* into groups of same-change deltas: ``(groups, stats)``.

    Grouped per SCOPE first (a disposition and a line are never the same change), then by
    meaning through the persona map-reduce with this module's prompt. A scope with one
    delta is one group without a call; no *generate_fn* (the headless CLI) ⇒ every delta
    its own group, reported as ``stats["ungrouped"]`` so the budget is read knowing it."""
    stats = {"items": len(records), "calls": 0, "scopes": {}, "ungrouped": generate_fn is None}
    groups: list[list[dict]] = []
    by_scope: dict = {}
    for r in records:
        scope = str(r.get("scope") or "").strip().lower()
        by_scope.setdefault(scope if scope in SCOPES else "", []).append(r)
    for scope, items in sorted(by_scope.items()):
        stats["scopes"][scope or "?"] = len(items)
        if len(items) < 2 or generate_fn is None:
            groups.extend([[r] for r in items])
            continue
        from core import persona_cluster
        ordered = sorted(items, key=lambda r: (r.get("delta") or ""))
        got, st = persona_cluster.map_reduce_groups(
            ordered, generate_fn, block_size=BLOCK_SIZE,
            prompt=prompt or _DEFAULT_CLUSTER_PROMPT, listing_fn=_listing,
            order_key=lambda r: (r.get("delta") or ""),
            reject_blobs=False,  # repeated proposals are the evidence this fold counts
            on_stage=(lambda info, _s=scope: on_stage({"scope": _s, **info}))
            if on_stage else None)
        stats["calls"] += int(st.get("calls") or 0)
        groups.extend(got or [[r] for r in ordered])
    return groups, stats


# ── the fold ─────────────────────────────────────────────────────────────────

def fold(groups: list[list[dict]], *, now: Optional[datetime] = None,
         maturity: float = DEFAULT_MATURITY, min_chats: int = DEFAULT_MIN_CHATS) -> list[dict]:
    """One pattern per group, most mature first.

    ``weighted_recurrence`` sums, over DISTINCT source chats, the heaviest vote that chat
    cast for the pattern. ``mature`` = that sum ≥ *maturity* AND ≥ *min_chats* chats."""
    now = now or datetime.now(timezone.utc)
    out = []
    for members in groups:
        if not members:
            continue
        per_chat: dict = {}
        for m in members:
            chat = str(m.get("source_session") or "")
            w = vote_weight(m, now)
            if w > per_chat.get(chat, -1.0):
                per_chat[chat] = w
        weighted = round(sum(per_chat.values()), 3)
        rep = max(members, key=lambda m: (vote_weight(m, now), len(m.get("delta") or ""), m["key"]))
        ranks = [float(m["tension_rank"]) for m in members
                 if isinstance(m.get("tension_rank"), (int, float))]
        keys = sorted(m["key"] for m in members)
        out.append({
            "id": hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:12],
            "scope": str(rep.get("scope") or ""),
            "delta": (rep.get("delta") or "").strip(),
            "keys": keys,
            "deltas": sorted({(m.get("delta") or "").strip() for m in members}),
            "chats": sorted(per_chat),
            "recurrences": len(per_chat),
            "weighted_recurrence": weighted,
            "tension_mean": round(sum(ranks) / len(ranks), 3) if ranks else None,
            "locators": sorted({str(m.get("locator") or "revise") for m in members}),
            "newest_ts": max((str(m.get("ts") or "") for m in members), default=""),
            "mature": bool(weighted >= float(maturity) and len(per_chat) >= int(min_chats)),
        })
    out.sort(key=lambda p: (-p["weighted_recurrence"], -p["recurrences"], p["id"]))
    return out


def build_document(patterns: list[dict], *, deltas: list[dict], attempts: list[dict],
                   run_id: str = "", maturity: float = DEFAULT_MATURITY,
                   min_chats: int = DEFAULT_MIN_CHATS, stats: Optional[dict] = None,
                   now: Optional[datetime] = None) -> dict:
    """The ``patterns.json`` document: the patterns plus what the deliberation line and
    the stage-3 gate need without re-reading the op-logs — how many votes arrived since
    the last attempt, and when that was."""
    now = now or datetime.now(timezone.utc)
    last_ts = last_attempt_ts(attempts)
    since = sum(1 for d in deltas if str(d.get("ts") or "") > last_ts) if last_ts else len(deltas)
    return {
        "schema_version": 1,
        "built_at": now.isoformat(),
        "run_id": run_id,
        "maturity": float(maturity),
        "min_chats": int(min_chats),
        "n_deltas": len(deltas),
        "n_patterns": len(patterns),
        "n_mature": sum(1 for p in patterns if p["mature"]),
        "deltas_since_last_attempt": since,
        "last_attempt_ts": last_ts,
        "stats": stats or {},
        "patterns": patterns,
    }


def write_patterns(prompt_dir: Path, doc: dict) -> Path:
    path = Path(prompt_dir) / PATTERNS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_patterns(prompt_dir: Path) -> Optional[dict]:
    path = Path(prompt_dir) / PATTERNS_FILE
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("patterns"), list) else None


def rebuild(prompt_dir: Path, generate_fn: Optional[Callable], *, prompts_dir: Optional[Path] = None,
            run_id: str = "", maturity: float = DEFAULT_MATURITY,
            min_chats: int = DEFAULT_MIN_CHATS, on_stage: Optional[Callable] = None,
            now: Optional[datetime] = None) -> dict:
    """The whole fold, as the reflection run calls it: read the unconsumed deltas, group,
    fold, write ``patterns.json``. Returns the document. Zero deltas still writes it, so
    the budget reads an honest zero rather than a stale file."""
    deltas = unconsumed(prompt_dir)
    attempts = read_rewrite_log(prompt_dir)
    prompt = load_cluster_prompt(prompts_dir) if prompts_dir is not None else None
    groups, stats = cluster_patterns(deltas, generate_fn, prompt=prompt, on_stage=on_stage)
    patterns = fold(groups, now=now, maturity=maturity, min_chats=min_chats)
    doc = build_document(patterns, deltas=deltas, attempts=attempts, run_id=run_id,
                         maturity=maturity, min_chats=min_chats, stats=stats, now=now)
    write_patterns(prompt_dir, doc)
    return doc


# ── the budget, as the executive reads it ────────────────────────────────────

def budget(prompt_dir: Path) -> dict:
    """What ``patterns.json`` says the rewrite may draw on. ``clears`` is stage 3's first
    condition (≥1 mature pattern); the rest feeds the deliberation line."""
    doc = read_patterns(prompt_dir)
    if doc is None:
        return {"clears": False, "mature": [], "patterns": [], "n_patterns": 0,
                "deltas_since_last_attempt": 0, "last_attempt_ts": "", "built_at": ""}
    pats = [p for p in doc.get("patterns") or [] if isinstance(p, dict)]
    mature = [p for p in pats if p.get("mature")]
    return {
        "clears": bool(mature),
        "mature": mature,
        "patterns": pats,
        "n_patterns": len(pats),
        "deltas_since_last_attempt": int(doc.get("deltas_since_last_attempt") or 0),
        "last_attempt_ts": str(doc.get("last_attempt_ts") or ""),
        "built_at": str(doc.get("built_at") or ""),
    }


def render_budget(b: dict, *, max_lines: int = 6) -> str:
    """The deliberation prompt's view of the budget (PROMPT_REWRITE.md §4) — empty when
    there is nothing on record, so a box with no deltas adds no line. Stage 2 SHOWS it;
    the action that spends it is stage 3."""
    pats = b.get("patterns") or []
    if not pats:
        return ""
    n_mature = len(b.get("mature") or [])
    lines = [f"Pulls your reflections keep noticing in your standing prompt: "
             f"{len(pats)} pattern(s), {n_mature} mature enough to act on."]
    shown = sorted(pats, key=lambda p: (not p.get("mature"), -float(p.get("weighted_recurrence") or 0)))
    for p in shown[:max_lines]:
        tag = "mature" if p.get("mature") else "forming"
        lines.append(f"- [{p.get('scope') or '?'}, {tag}, {p.get('recurrences', 0)} chat(s)] "
                     f"{(p.get('delta') or '').strip()[:220]}")
    if len(pats) > max_lines:
        lines.append(f"- … and {len(pats) - max_lines} more")
    return "\n".join(lines)


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import re
    import tempfile
    from core.prompt_mutation import append_prompt_delta

    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    # Weights: recency × (0.5 + rank); no rank ⇒ 0.5; missing ts ⇒ fresh.
    assert vote_weight({"ts": "2026-09-17T00:00:00+00:00", "tension_rank": 0.9}, now) == 1.4
    assert vote_weight({"ts": "2026-09-17T00:00:00+00:00"}, now) == 0.5
    assert vote_weight({"tension_rank": 1.0}, now) == 1.5
    assert vote_weight({"ts": "2026-01-01T00:00:00+00:00", "tension_rank": 1.0}, now) == 0.0   # >180 d
    mid = vote_weight({"ts": "2026-06-05T00:00:00+00:00", "tension_rank": 0.5}, now)         # 105 d
    assert 0.45 < mid < 0.55, mid

    def _rec(session, idx, delta, scope="disposition", rank=None, ts="2026-09-10T00:00:00+00:00"):
        return {"source_session": session, "exchange_index": idx, "delta": delta, "scope": scope,
                "tension_rank": rank, "ts": ts, "locator": "keep_tension" if rank else "revise"}

    # Fold: one vote per distinct chat (the heaviest), maturity = weight AND chat count.
    g = [[_rec("a.json", 0, "[t1] say no plainly", rank=0.9),
          _rec("a.json", 3, "[t1] say no plainly again", rank=0.2),
          _rec("b.json", 1, "[t1] не смягчай отказ", rank=None)]]
    for grp in g:
        for m in grp:
            m["key"] = delta_key(m)
    pats = fold(g, now=now)
    assert len(pats) == 1 and pats[0]["recurrences"] == 2, pats
    assert pats[0]["weighted_recurrence"] == 1.9, pats[0]["weighted_recurrence"]   # 1.4 + 0.5
    assert pats[0]["mature"] and pats[0]["delta"] == "[t1] say no plainly"
    assert fold(g, now=now, maturity=2.0)[0]["mature"] is False
    assert fold([[g[0][0], g[0][1]]], now=now, maturity=1.0)[0]["mature"] is False, "one chat is never mature"

    # Grouping through the map-reduce with a fake model that groups by a [tag].
    def fake_generate(listing, system, **kw):
        by_tag: dict = {}
        for line in listing.splitlines():
            m = re.match(r"\s*(\d+)\.\s*\[(t\d+)\]", line)
            if m:
                by_tag.setdefault(m.group(2), []).append(int(m.group(1)))
        return "\n".join("GROUP: " + ", ".join(map(str, v)) for v in by_tag.values())

    recs = [_rec("a.json", 0, "[t1] x", rank=0.9), _rec("b.json", 0, "[t1] y"),
            _rec("c.json", 0, "[t2] z", scope="voice"), _rec("d.json", 0, "[t1] w", scope="voice")]
    for r in recs:
        r["key"] = delta_key(r)
    groups, st = cluster_patterns(recs, fake_generate)
    # Scope splits first: the two voice deltas never meet the disposition ones.
    sizes = sorted(len(x) for x in groups)
    assert sizes == [1, 1, 2] and st["scopes"] == {"disposition": 2, "voice": 2}, (sizes, st)
    groups_nogen, st2 = cluster_patterns(recs, None)
    assert len(groups_nogen) == 4 and st2["ungrouped"] is True

    # More corroboration must not destroy a mature pattern. Exercise both the map
    # guard threshold and the reduce path across multiple full blocks.
    from core import persona_cluster
    for n in (5, 6, 10, 6 * BLOCK_SIZE):
        repeated = [_rec(f"chat_{i}.json", 0, "[t1] say no plainly", rank=0.9)
                    for i in range(n)]
        for r in repeated:
            r["key"] = delta_key(r)
        grouped, _ = cluster_patterns(repeated, fake_generate)
        folded = fold(grouped, now=now)
        assert len(folded) == 1 and folded[0]["mature"], (n, folded)
        assert folded[0]["recurrences"] == n and len(folded[0]["keys"]) == n
        assert folded[0]["weighted_recurrence"] == round(n * 1.4, 3)
    # The guard still protects existing persona/fact callers that did not opt out.
    guarded, _ = persona_cluster.map_reduce_groups(
        repeated[:6], fake_generate, prompt=_DEFAULT_CLUSTER_PROMPT, listing_fn=_listing)
    assert len(guarded) == 6

    # End to end on disk: op-log in, patterns.json out; consumed keys leave the pool; a
    # re-reflected exchange counts once; the budget + its rendering read the file back.
    with tempfile.TemporaryDirectory() as d:
        pdir = Path(d) / "prompt"
        prompts = Path(d) / "prompts"
        append_prompt_delta(pdir, _rec("a.json", 0, "[t1] first wording", rank=0.9))
        append_prompt_delta(pdir, _rec("a.json", 0, "[t1] re-reflected wording", rank=0.9))
        append_prompt_delta(pdir, _rec("b.json", 2, "[t1] same pull elsewhere"))
        append_prompt_delta(pdir, _rec("c.json", 1, "[t3] a lone one", scope="line"))
        doc = rebuild(pdir, fake_generate, prompts_dir=prompts, run_id="r1", now=now)
        assert (prompts / CLUSTER_PROMPT_FILE).exists(), "cluster prompt default-written"
        assert doc["n_deltas"] == 3, doc["n_deltas"]                 # newest per exchange
        assert doc["n_patterns"] == 2 and doc["n_mature"] == 1, doc
        top = doc["patterns"][0]
        assert top["mature"] and top["recurrences"] == 2 and "re-reflected" in top["delta"], top
        assert doc["deltas_since_last_attempt"] == 3 and doc["last_attempt_ts"] == ""
        b = budget(pdir)
        assert b["clears"] and len(b["mature"]) == 1 and b["n_patterns"] == 2
        text = render_budget(b)
        assert "2 pattern(s), 1 mature" in text and "[disposition, mature, 2 chat(s)]" in text, text
        # Stage 3 consumes the mature pattern's keys ⇒ it leaves the pool; a later delta
        # counts as "since the attempt".
        (pdir / REWRITE_LOG_FILE).write_text(json.dumps({
            "ts": "2026-09-18T01:00:00+00:00", "outcome": "changed",
            "consumed_keys": top["keys"]}) + "\n", encoding="utf-8")
        append_prompt_delta(pdir, {**_rec("e.json", 0, "[t3] another lone", scope="line"),
                                   "ts": "2026-09-18T02:00:00+00:00"})
        doc = rebuild(pdir, fake_generate, prompts_dir=prompts, now=now)
        assert doc["n_deltas"] == 2 and doc["n_mature"] == 0, doc
        assert doc["deltas_since_last_attempt"] == 1 and doc["last_attempt_ts"].startswith("2026-09-18T01")
        assert not budget(pdir)["clears"]
        # No file / empty ⇒ honest zero, no line.
        assert render_budget(budget(Path(d) / "nowhere")) == ""
        doc = rebuild(Path(d) / "empty", None, now=now)
        assert doc["n_patterns"] == 0 and (Path(d) / "empty" / PATTERNS_FILE).exists()
    print("prompt_patterns self-test OK")


if __name__ == "__main__":
    _selftest()
