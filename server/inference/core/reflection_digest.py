"""Persona digest — a versioned self-portrait that will anchor the "become" criterion.

The branch-and-select / revision passes today judge a reply by *"is this mine?"* — a
backward-looking match against what Ava has already done. The forward-looking criterion
the design wants — *"is this the reply of who I'm becoming?"* — needs an explicit anchor:
a compact, current snapshot of Ava's character that a candidate can be measured against.
That snapshot is the **persona digest**.

Crucially the digest is **not** a bag of belief statements. A declarative self-statement
("I prefer bluntness") can score *does this sound like me?* but can never justify a
*conditional* move — *should I deploy this register, here, as a weapon?* (the tactical-vs-
ideology line the README draws). So the digest's spine is **dispositions with gates**:
a skill + *when* to deploy it + the *line* not to cross. Declarative stances are just one
facet beside it.

Four facets, generated as plain prose (labelled sections, lenient parse — the project
never asks a small model for JSON) and folded into a structured, versioned artifact:

  * ``VOICE``        — how she speaks by default (the distributional disposition)
  * ``STANCES``      — declarative beliefs (folded from persona anchors)
  * ``DISPOSITIONS`` — procedural: skill + ``when`` gate + ``not`` boundary (the core)
  * ``LINES``        — the inverse gates / what she refuses to become

**Phase one (this module): synthesize + persist + version the digest, and expose the
embedding-channel anchor texts. It changes NO verdict or branch behaviour** — the digest
is written and logged only. The branch-select criterion flip that *consumes* it is a
separate, gated follow-up. Graceful empty: with no persona evidence the pass is a no-op,
so the zero-evidence limit is exactly today's behaviour.

The pure halves (gather / parse / fold / write / should-regenerate) are GPU-free and
unit-tested; ``run_digest_pass`` is the thin orchestration that needs a ``generate_fn``.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from core.field_parse import section as _section

# Artifact layout: a single ``digest.json`` — the persona's one self-portrait. Rollback +
# versioning are provided by the persona snapshot lineage (each persona = one version,
# selected by server/data/persona/current.json), so the digest no longer carries its own
# per-file versioning or ``current.json`` pointer.
DIGEST_FILE = "digest.json"

# Persona statements are paraphrastic — the same disposition resurfaces in different
# words across chats, so exact-`content_key` recurrence never climbs and every item
# stays `[1]`/emerging. Cluster by embedding first, so semantically-equivalent
# restatements accumulate into one theme whose recurrence is the distinct sessions
# across the whole cluster. First-guess cosine threshold (same family RAG/fact_render use).
CLUSTER_SIM_THRESHOLD = 0.6

_SECTION_RE = re.compile(
    _section("VOICE", "STANCES", "DISPOSITIONS", "LINES"),
    re.IGNORECASE | re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# A `<think>` the generation never closed — it ran out of budget mid-thought. Everything
# after it is scratch reasoning, so it is cut like a closed block rather than left in the
# body. See `strip_think` for why that is not merely tidiness.
_DANGLING_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]*")
_TAG_RE = re.compile(r"^\[(?:stance|line|disposition)\][ \t]*", re.IGNORECASE)
# A disposition opens on a bracketed head. The model reliably writes the *name* in the
# brackets ("- [systematizing the absurd]"), not the literal placeholder tag from the
# prompt ("- [disposition] systematizing…") — accept either: capture the bracket content
# and the trailing text, and disambiguate in code.
_DISP_HEAD_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?\[([^\]]+)\][ \t]*(.*)$"
)
# Decoration the model sometimes hangs off the head, e.g. a LaTeX "$\langle$ alias $\rangle$"
# secondary label — stripped from the captured name.
_DISP_DECOR_RE = re.compile(r"\$[^$]*\$|[⟨⟩<>]")
_KV_RE = re.compile(r"^[ \t]*(when|do|not|maturity|evidence)[ \t]*:[ \t]*(.*)$",
                    re.IGNORECASE)


# ── evidence gathering ─────────────────────────────────────────────────────── #

# ── persona recency decay ──────────────────────────────────────────────────── #
# Old persona evidence FADES: a stance stated long ago and never restated should stop
# shaping the self-portrait, while a trait REINFORCED across recent sessions survives
# (the decay is per-evidence but summed per theme, so recent restatements keep a theme
# alive — see `_evidence_entry`). Wall-clock, consistent with the fact/dialogue decay in
# training/decay.py. Full weight at/under FULL days, linear ramp to 0 at ZERO days.
# (Module defaults; a later change can wire these to server_config consolidation.wall_clock.)
PERSONA_DECAY_FULL_DAYS = 30.0
PERSONA_DECAY_ZERO_DAYS = 180.0
_RECENCY_BANDS = 4   # coarse quantization of the weight, so the regen fingerprint only
                     # changes when a theme's freshness crosses a band (no per-run thrash).
# A theme whose summed recency weight falls below this is treated as faded-away and is
# dropped from the evidence Ava reflects on (so the regenerated portrait forgets it). Set
# below 1.0 so a single FRESH one-off (weight 1.0, genuinely emerging) still survives; only
# stale evidence that has decayed under half a session's worth drops.
_PROMPT_WEIGHT_FLOOR = 0.5

# ── persona tenure discount (the anti-ratchet) ─────────────────────────────── #
# A persona theme's FIRST session is genuine emergence (full credit); every LATER session
# re-affirms a trait that, by then, was already live and recalled into the very reply being
# reflected — the circular vote (persona conditions the reply → revision re-derives the same
# stance → +1 distinct session). Without a discount that loop climbs recurrence without limit
# and the persona ossifies. So each successive affirmation is scaled on a geometric curve by
# its chronological rank: an echo-sustained theme's `weighted_recurrence` CONVERGES to the
# ceiling 1/(1-_TENURE_DECAY) instead of growing linearly, while a genuinely new theme still
# enters at full weight. It composes with the recency decay above, and the two are
# anti-correlated by construction (emergence session = oldest = most recency-decayed; recent
# echoes = highest rank = most tenure-discounted), so ONLY a young-and-active theme escapes
# both — exactly the profile that should be allowed to form. Persona `source_session`s are
# timestamp stems, so `sorted()` order is chronological AND revisit-stable (a revisit
# re-reflects the same stem); rare non-chat sources (wiki/til/web) sort to the tail and are
# treated as newest → most-discounted, which is fine for self-directed evidence.
# Only `weighted_recurrence` feels this — the raw `recurrences` count (and thus the digest
# maturity gate / judge criterion flip, and the regen fingerprints) is deliberately unchanged.
_TENURE_DECAY = 0.6   # k-th (0-based, oldest = 0) affirmation weighted _TENURE_DECAY**k

# ── persona counter-evidence (the persuasion channel) ──────────────────────── #
# The symmetric opponent to affirmation: a user reaction that pushes AGAINST a live stance
# is logged as counter-evidence (`ConsolidationLedger.counter`) under the pushing session,
# and netted out of the theme's weighted recurrence here at gain _PERSUASION_GAIN. Kept
# BELOW 1.0 so ONE push cannot cancel one affirmation — persuasion must ACCUMULATE across
# sessions (an integrator with a threshold, never a single hard turn). Unlike affirmations,
# counters are NOT tenure-discounted: external pushback is not the circular self-vote, so it
# is free to accumulate. The net is floored at 0, at which point sustained pushback has
# faded the trait under _PROMPT_WEIGHT_FLOOR and it drops from the portrait on its own.
# _PERSUASION_GAIN is the persuadability knob (lower = more resistant to being talked out of
# a stance). Only weighted_recurrence is affected — raw `recurrences` (and thus the maturity
# gate) is untouched. The reaction→key bridge and the pushback-detection policy that EMIT
# these ops are the next layer — see `ConsolidationLedger.counter`.
_PERSUASION_GAIN = 0.5


def _weighted_recurrence(e: dict) -> float:
    """A theme's recency-weighted recurrence, defaulting to its raw count (old digests /
    manually-built evidence that predate decay read as fully fresh)."""
    return float(e.get("weighted_recurrence", e.get("recurrences", 1)))


def _persona_recency_weight(age_days: float,
                            full: float = PERSONA_DECAY_FULL_DAYS,
                            zero: float = PERSONA_DECAY_ZERO_DAYS) -> float:
    """Recency weight in [0,1]: 1.0 at/under *full* days old, linear ramp to 0 at *zero*."""
    if age_days <= full:
        return 1.0
    if age_days >= zero:
        return 0.0
    return (zero - age_days) / (zero - full)


def _recency_band(weight: float) -> int:
    """Quantize a weight into a coarse 0..N band (for the regen fingerprint)."""
    return max(0, min(_RECENCY_BANDS, int(round(weight * _RECENCY_BANDS))))


def _age_days(ts_iso: str, now) -> float:
    """Age of an ISO timestamp in days vs *now*; an unparseable/absent ts reads as FRESH
    (0 days), so a missing date never erases evidence we cannot place in time."""
    if not ts_iso:
        return 0.0
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts_iso)
        delta = (now - dt).total_seconds()
        return max(0.0, delta / 86400.0)
    except Exception:
        return 0.0


def gather_persona_raw(*consolidation_dirs: Path, now=None) -> list[dict]:
    """The model-free fold: one or more anchor-ledger files → the raw persona set the digest
    is built from, BEFORE any clustering.

    Persona only — the digest is *who she is*, not world-facts. Multiple dirs are folded
    (live + this run's staging) and deduped by ``content_key``, keeping the highest stage
    and tracking **distinct source sessions** (the impedance signal: a stance seen across
    many independent chats is mature; a one-off is emerging).

    **Recency:** each (key, session) register op is dated, so each contributing session
    carries a wall-clock decay weight (``_persona_recency_weight`` vs *now*, default the
    wall clock). ``session_weights`` feeds the summed-per-theme decay in ``_evidence_entry``;
    ``recency_band`` (a coarse quantization of the key's freshest evidence) enters the
    regenerate fingerprint so an aging theme triggers one regen per band crossed, not every run.

    Returns ``[{key, content, stage, sessions:set, session_weights:{session: w}, recency_band}]``.
    Deterministic given *now* — the regenerate gate fingerprints this.
    """
    from training.ledger import ConsolidationLedger, LEDGER_FILE
    from datetime import datetime
    if now is None:
        now = datetime.now()

    merged: dict[str, dict] = {}
    sources: dict[str, set] = {}
    # Latest register ts per (key, session) — the freshest date that stance appeared in
    # that conversation, which its recency weight is computed from.
    session_ts: dict[str, dict[str, str]] = {}
    # Counter-evidence: distinct sessions that pushed AGAINST a key, dated (same shape as
    # sources / session_ts). Netted out of weighted recurrence in _evidence_entry (the
    # persuasion channel); see ConsolidationLedger.counter.
    counters: dict[str, set] = {}
    counter_ts: dict[str, dict[str, str]] = {}
    for d in consolidation_dirs:
        if d is None:
            continue
        path = Path(d) / LEDGER_FILE
        if not path.exists():
            continue
        # Recurrence: scan raw register ops for distinct source_session per key, dated.
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("op") == "register" and rec.get("type") == "persona":
                    k = rec.get("key")
                    src = (rec.get("source_session") or "").strip()
                    if k and src:
                        sources.setdefault(k, set()).add(src)
                        ts = (rec.get("ts") or "").strip()
                        prev = session_ts.setdefault(k, {}).get(src, "")
                        if ts and ts > prev:      # ISO ts compare lexically == chronologically
                            session_ts[k][src] = ts
                elif rec.get("op") == "counter" and rec.get("type") == "persona":
                    k = rec.get("key")
                    src = (rec.get("source_session") or "").strip()
                    if k and src:
                        counters.setdefault(k, set()).add(src)
                        ts = (rec.get("ts") or "").strip()
                        prev = counter_ts.setdefault(k, {}).get(src, "")
                        if ts and ts > prev:
                            counter_ts[k][src] = ts
        except Exception:
            pass
        # Folded current state (content + accrued stage).
        try:
            led = ConsolidationLedger(Path(d))
            for key, rec in led.fold().items():
                if rec.get("type") != "persona":
                    continue
                if rec.get("superseded"):
                    continue   # reconciliation softened it out of *active* evidence
                content = (rec.get("content") or "").strip()
                if not content:
                    continue
                prev = merged.get(key)
                if prev is None or rec.get("stage", 0) >= prev.get("stage", 0):
                    merged[key] = {"key": key, "content": content,
                                   "stage": int(rec.get("stage", 0) or 0)}
        except Exception:
            continue

    out: list[dict] = []
    for key, rec in merged.items():
        sess = set(sources.get(key, set()))
        # Per-session recency weight from its freshest register date; a session with no
        # recorded ts reads as fresh (weight 1.0) so undated evidence is never erased.
        weights = {
            s: _persona_recency_weight(_age_days(session_ts.get(key, {}).get(s, ""), now))
            for s in sess
        }
        band = _recency_band(max(weights.values())) if weights else _RECENCY_BANDS
        # Counter-evidence sessions for this key (recency-weighted like affirmations, but
        # netted — not tenure-discounted — in _evidence_entry). Empty for uncontested traits.
        c_sess = set(counters.get(key, set()))
        c_weights = {
            s: _persona_recency_weight(_age_days(counter_ts.get(key, {}).get(s, ""), now))
            for s in c_sess
        }
        out.append({
            "key": key, "content": rec["content"],
            "stage": int(rec.get("stage", 0) or 0),
            "sessions": sess,
            "session_weights": weights,
            "recency_band": band,
            "counter_sessions": c_sess,
            "counter_weights": c_weights,
        })
    return out


def gather_persona_evidence(
    *consolidation_dirs: Path,
    embedder=None,
    cluster_threshold: float = CLUSTER_SIM_THRESHOLD,
    generate_fn=None,
) -> list[dict]:
    """Raw fold (``gather_persona_raw``) + clustering (``cluster_persona_evidence``) —
    kept as one call for back-compat. See those two for the split.

    Returns ``[{key, content, stage, recurrences, cluster_size, sources, members}]``,
    most-recurred first.
    """
    base = gather_persona_raw(*consolidation_dirs)
    return cluster_persona_evidence(base, generate_fn=generate_fn, embedder=embedder,
                                    cluster_threshold=cluster_threshold)


# ── semantic clustering of persona evidence ───────────────────────────────── #

_EMBEDDER = None
_EMBEDDER_LOADED = False


def _load_embedder():
    """Lazily load the shared all-MiniLM embedder (CPU), cached module-level.

    Returns None if sentence-transformers is unavailable — gather then falls back to
    exact-key recurrence. The failure is cached so we don't retry the import each run.
    """
    global _EMBEDDER, _EMBEDDER_LOADED
    if _EMBEDDER_LOADED:
        return _EMBEDDER
    _EMBEDDER_LOADED = True
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception:
        _EMBEDDER = None
    return _EMBEDDER


def _dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _evidence_entry(members: list[dict]) -> dict:
    """Collapse cluster members into one evidence dict.

    ``recurrences`` is the distinct sessions across *all* members (the maturity signal).
    The representative is chosen deterministically — most own-sessions, then longest
    content, then key — so the same membership always yields the same representative
    (and thus a stable fingerprint).
    """
    sessions: set = set()
    for m in members:
        sessions |= m["sessions"]
    # Summed-per-theme decay: for each DISTINCT session, take the freshest weight across
    # the theme's paraphrase members, then SUM. A reinforced theme (many/recent sessions)
    # keeps a high weighted recurrence and survives; a stale one-off decays toward 0.
    # Absent weights (manually-built base in tests) default to 1.0 == the old raw count.
    sess_weight: dict = {}
    for m in members:
        mw = m.get("session_weights") or {}
        for s in m["sessions"]:
            w = float(mw.get(s, 1.0))
            if w > sess_weight.get(s, -1.0):
                sess_weight[s] = w
    # Tenure discount: rank the distinct sessions oldest→newest (stem == timestamp, so plain
    # sort is chronological). rank 0 = emergence (full weight); each later affirmation is
    # scaled by _TENURE_DECAY**rank BEFORE the recency sum, so the circular self-vote can no
    # longer climb the theme's weighted recurrence without bound (see _TENURE_DECAY).
    ordered = sorted(sess_weight)
    tenure = {s: _TENURE_DECAY ** k for k, s in enumerate(ordered)}
    weighted = sum(w * tenure[s] for s, w in sess_weight.items())
    # Counter-evidence (the persuasion channel): sustained pushback against the theme,
    # netted out at gain _PERSUASION_GAIN. Recency-weighted like affirmations (freshest per
    # distinct session across paraphrases) but NOT tenure-discounted — external pushback is
    # not the circular self-vote, so it must be free to accumulate. Floored at 0: once
    # enough pushes drive the net under _PROMPT_WEIGHT_FLOOR the theme fades from the portrait
    # on its own. Absent for uncontested traits (older/test evidence) ⇒ no change.
    c_weight: dict = {}
    for m in members:
        cw = m.get("counter_weights") or {}
        for s in m.get("counter_sessions") or ():
            w = float(cw.get(s, 1.0))
            if w > c_weight.get(s, -1.0):
                c_weight[s] = w
    counter_weighted = sum(c_weight.values())
    weighted = round(max(0.0, weighted - _PERSUASION_GAIN * counter_weighted), 3)
    rep = max(members, key=lambda m: (len(m["sessions"]), len(m["content"]), m["key"]))
    return {
        "key": rep["key"],
        "content": rep["content"],
        "stage": max(m["stage"] for m in members),
        "recurrences": len(sessions) or 1,
        "weighted_recurrence": weighted,
        "counter_recurrence": round(counter_weighted, 3),
        "recency_band": _recency_band(max(sess_weight.values())) if sess_weight else _RECENCY_BANDS,
        "cluster_size": len(members),
        "sources": sorted(sessions),
        # All member phrasings (representative included), retained so the persisted
        # digest can show *which* statements merged into a theme — the data needed to
        # audit maturity and calibrate the clustering threshold.
        "members": sorted(m["content"] for m in members),
        # The same membership by anchor key. Content strings alone cannot be rejoined to
        # the ledger (a statement may be re-registered with different whitespace, and
        # matching prose is fragile), so an incremental pass — judge only the statements
        # new since the last digest against the existing themes — needs stable ids. Keys
        # are what `gather_persona_raw` folds on, so this is the exact join.
        "member_keys": sorted(m["key"] for m in members),
    }


def _cluster_evidence(base: list[dict], embedder, threshold: float) -> list[dict]:
    """Greedy average-link clustering of persona items by embedding similarity.

    Average-link (mean similarity to a cluster's members) resists the chaining that
    single-link suffers. Items are processed in a deterministic key order so the greedy
    result is reproducible. Any embed failure → exact-key fallback (one item per entry).
    """
    if not base:
        return []
    if embedder is None:
        return [_evidence_entry([it]) for it in base]
    ordered = sorted(base, key=lambda it: it["key"])
    try:
        raw = embedder.encode([it["content"] for it in ordered],
                              convert_to_numpy=True, normalize_embeddings=True)
        vecs = [list(map(float, v)) for v in raw]
    except Exception:
        return [_evidence_entry([it]) for it in base]

    clusters: list[list[int]] = []
    for i in range(len(ordered)):
        best_c, best_sim = None, -1.0
        for c in clusters:
            sim = sum(_dot(vecs[i], vecs[j]) for j in c) / len(c)
            if sim > best_sim:
                best_c, best_sim = c, sim
        if best_c is not None and best_sim >= threshold:
            best_c.append(i)
        else:
            clusters.append([i])
    return [_evidence_entry([ordered[j] for j in c]) for c in clusters]


def raw_fingerprint(base: list[dict]) -> str:
    """Deterministic signature of the RAW persona set (pre-clustering) — drives the
    regenerate gate, so a non-deterministic LLM clustering can't spuriously trip it and a
    no-op run never reaches the model."""
    import hashlib
    # Include the coarse recency band so an AGING theme regenerates the digest once per
    # band crossed (not every run, and not never). Absent band (manually-built base in
    # tests) reads as freshest so behaviour is unchanged there. Include the counter-session
    # count so accumulating PUSHBACK (the persuasion channel) also triggers one regen per
    # new push — otherwise a countered trait's erosion would sit unread until an unrelated
    # regen. Deterministic (dated ops), so no thrash. Absent ⇒ 0, so behaviour is unchanged.
    keys = sorted(
        f"{b['key']}:{b['stage']}:{len(b['sessions'])}:{b.get('recency_band', _RECENCY_BANDS)}"
        f":{len(b.get('counter_sessions') or ())}"
        for b in base
    )
    return hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:16]


def cluster_persona_evidence(base: list[dict], *, generate_fn=None, embedder=None,
                             cluster_threshold: float = CLUSTER_SIM_THRESHOLD) -> list[dict]:
    """Group paraphrases into themes, most-recurred first.

    Three tiers, best-first: an **LLM grouping pass** on *generate_fn* (the capable path —
    understands cross-lingual paraphrase the English all-MiniLM cosine misses), else the
    **MiniLM** greedy average-link (*embedder*, default the shared one), else **exact-key**
    (each statement its own theme). Whichever tier runs, each cluster collapses via
    ``_evidence_entry`` to the same shape, so recurrence / maturity / prompt body downstream
    are untouched. LLM failure or an unparseable grouping falls through to the embedder.
    """
    if not base:
        return []
    out = None
    if generate_fn is not None:
        out = llm_cluster_evidence(base, generate_fn)
    if out is None:
        out = _cluster_evidence(base, embedder if embedder is not None else _load_embedder(),
                                cluster_threshold)
    # Most-recurred first — but by the RECENCY-WEIGHTED recurrence, so a faded theme
    # sinks and a reinforced one leads; raw recurrence / stage break ties.
    out.sort(key=lambda r: (-r.get("weighted_recurrence", r["recurrences"]),
                            -r["recurrences"], -r["stage"]))
    return out


def llm_cluster_evidence(base: list[dict], generate_fn) -> Optional[list[dict]]:
    """One greedy generation groups the numbered persona statements into paraphrase themes.

    Judges by meaning, not surface words, so it merges cross-lingual restatements the
    embedder can't. Returns ``[_evidence_entry(members)]`` (same shape as ``_cluster_evidence``)
    or ``None`` to signal the caller should fall back to the embedder (generation failed, or
    the model emitted no parseable grouping). Run greedy (``temperature=0``) for
    reproducibility; the persisted digest is the replay record either way.
    """
    ordered = sorted(base, key=lambda it: it["key"])         # deterministic numbering
    listing = "\n".join(f"{i + 1}. {it['content']}" for i, it in enumerate(ordered))
    try:
        resp = generate_fn(
            listing, load_cluster_prompt(),
            temperature=0.0, top_p=1.0,
            max_new_tokens_setting="1024",
            before_session="", disable_rag=True, disable_thinking=True)
    except Exception:
        return None
    groups = _parse_groups(resp, len(ordered))
    if not groups:                                           # nothing parsed → fall back
        return None
    return [_evidence_entry([ordered[j] for j in g]) for g in groups]


def _parse_groups(text: str, n: int) -> Optional[list[list[int]]]:
    """Parse ``GROUP: 1, 4, 7`` lines into a valid PARTITION of ``0..n-1``.

    First assignment wins (a number named in two groups stays in its first), and any
    statement the model omitted becomes its own singleton — so the result is always a clean
    partition regardless of model sloppiness. Returns ``None`` only when no ``GROUP`` line
    parsed at all (signal to fall back to the embedder rather than treat every item as a
    singleton, which would silently discard clustering)."""
    body = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    assigned: dict[int, int] = {}
    order: list[int] = []
    for m in re.finditer(r"(?im)^\s*[\-\*>#\s]*GROUP\b[^:\n]*:\s*(.+)$", body):
        members = [int(x) - 1 for x in re.findall(r"\d+", m.group(1))]
        members = [i for i in members if 0 <= i < n and i not in assigned]
        if not members:
            continue
        gid = len(order)
        order.append(gid)
        for i in members:
            assigned[i] = gid
    if not assigned:
        return None
    for i in range(n):                                       # omitted → singleton
        if i not in assigned:
            gid = len(order)
            order.append(gid)
            assigned[i] = gid
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(assigned[i], []).append(i)
    return [groups[g] for g in order if g in groups]


def load_cluster_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The persona-clustering prompt (``prompts/persona_cluster_prompt.txt``), or a safe
    inline fallback so the pass works even if the file is missing."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "persona_cluster_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "You are grouping your own self-statements. Below are numbered first-person things "
        "you have come to hold about yourself — some in different languages or phrasings that "
        "mean the same thing.\n\n"
        "Group the numbers so that statements expressing the SAME underlying disposition, "
        "stance, or trait are in one group — even when the wording or language differs. Keep "
        "genuinely distinct traits in separate groups. Every number appears in exactly one "
        "group; a one-of-a-kind statement is a group of one. Judge by meaning, not surface "
        "words.\n\n"
        "One thing overrides surface similarity: DIRECTION. A statement about resisting, "
        "avoiding, limiting, or switching off a trait does not belong in a group with "
        "statements about enacting or enjoying that same trait — same subject, opposite "
        "direction is a different group.\n\n"
        "Output only lines of this form, nothing else:\n"
        "GROUP: <comma-separated numbers>"
    )


# ── polarity screening of clustered themes ─────────────────────────────────── #
# The cluster prompt groups by underlying TRAIT, and "I tend to X" / "I resist X" are
# the same trait pointing opposite ways — so a theme can swallow its own
# counter-evidence: the opposing statement's sessions count as affirmations of the
# theme, the representative (picked by majority) is the only content the digest pass
# reads (`build_digest_prompt_input`), and the minority "avoid" statement is erased
# from the prompt body while its votes stand behind the tendency. Observed on the live
# 2026-07-15 digest: "я сознательно отключаю все аналитические фильтры" (switching the
# analysis OFF) merged — and counted — under "Я получаю почти физическое удовольствие
# от «интеллектуального переусложнения»". The cluster prompt now names direction as a
# grouping rule, and this screen is the enforcement behind it: each merged theme is
# re-checked on the same model, members that OPPOSE the representative are split into
# their own theme, and their sessions are returned as a COUNTER plan (the persuasion
# channel's second producer, beside next-turn user pushback) instead of affirmations.
# Conservative by the same asymmetry as the blob guard: a missed opposition survives
# one more digest, while an over-eager split only defers a member to its own theme —
# nothing is deleted either way.
_POLARITY_MAX_MEMBERS = 40   # listing cap per call (mirrors persona_cluster's block size);
                             # members past it stay with the theme unscreened
_POLARITY_MAX_THEMES = 64    # most-recurred themes screened first; the rest pass through

_OPPOSE_RE = re.compile(r"(?im)^\s*[\-\*>#\s]*OPPOSE\b[^:\n]*:\s*(.+)$")


def load_polarity_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The theme polarity-screen prompt (``prompts/persona_polarity_prompt.txt``), or a
    safe inline fallback so the pass works even if the file is missing."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "persona_polarity_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "You are checking one group of your own self-statements for direction. The "
        "group was formed because all of its statements are about the SAME trait or "
        "habit of yours. THEME is the group's representative statement; below it, the "
        "other statements in the group are numbered.\n\n"
        "A statement AGREES with the theme when it describes you enacting, valuing, or "
        "enjoying that trait. A statement OPPOSES the theme when it describes you "
        "resisting, avoiding, limiting, switching off, or deliberately setting that "
        "trait aside — the same subject, pointing the other way. An opposing statement "
        "is real evidence about you; it just must not be counted as more support for "
        "the theme it pushes against.\n\n"
        "Only flag genuine opposition. A statement that merely qualifies the theme, "
        "applies it to a different situation, or is about something else entirely is "
        "NOT opposition — when unsure, do not flag it.\n\n"
        "Output exactly one line, nothing else:\n"
        "OPPOSE: <comma-separated numbers of the opposing statements>\n"
        "or, if no statement opposes the theme:\n"
        "OPPOSE: none"
    )


def _parse_flag_line(text: str, n: int, pattern: re.Pattern) -> Optional[list[int]]:
    """Parse a ``LABEL: 2, 5`` verdict line into 0-based indices (shared by the
    polarity screen's ``OPPOSE`` and the gate screen's ``ESCALATES``).

    A digit-free value (``none``) → ``[]`` — an explicit "nothing flagged". No label
    line at all → ``None`` — the judgement is unusable, so the caller changes nothing
    rather than guessing. Out-of-range and duplicate numbers are dropped."""
    body = strip_think(text or "")
    found = False
    seen: set[int] = set()
    out: list[int] = []
    for m in pattern.finditer(body):
        found = True
        for x in re.findall(r"\d+", m.group(1)):
            i = int(x) - 1
            if 0 <= i < n and i not in seen:
                seen.add(i)
                out.append(i)
    return out if found else None


def _parse_oppose(text: str, n: int) -> Optional[list[int]]:
    """Parse the polarity screen's ``OPPOSE: 2, 5`` line — see ``_parse_flag_line``."""
    return _parse_flag_line(text, n, _OPPOSE_RE)


def screen_theme_polarity(evidence: list[dict], base: list[dict], generate_fn,
                          *, on_stage: Optional[Callable] = None) -> dict:
    """Re-check each merged theme for members that OPPOSE its representative.

    One short greedy thinking-off call per multi-member theme (same shape as
    ``llm_cluster_evidence``). A flagged member is split out of the theme — the theme
    and the opposition are rebuilt via ``_evidence_entry`` from the raw *base* items
    (rejoined by ``member_keys``, the stable join the artifact carries for exactly
    this) — and the opposers' sessions are returned as a **counter plan**:
    ``{key, content, sessions, opposed}`` per split theme, meant for
    ``ConsolidationLedger.counter`` so the opposition becomes durable negative
    pressure on the theme instead of affirmation votes. Writing is the caller's
    decision (detached sink); this function is pure orchestration over *generate_fn*.

    Fail-soft per theme: an errored call or an unparseable verdict keeps that theme
    unchanged. Returns ``{"evidence", "counters", "report"}`` with the evidence
    re-sorted the same way ``cluster_persona_evidence`` sorts."""
    by_key = {it["key"]: it for it in base}
    sys_prompt = load_polarity_prompt()
    out: list[dict] = []
    counters: list[dict] = []
    screened = split = flagged = failed = 0
    for e in evidence:
        member_keys = e.get("member_keys") or []
        keys = [k for k in member_keys if k in by_key]
        rep_key = e.get("key")
        others = sorted((by_key[k] for k in keys if k != rep_key),
                        key=lambda it: it["key"])
        # A theme is only screened when its FULL membership rejoins to the raw base —
        # rebuilding from a partial join would silently drop the unjoinable members'
        # sessions. (In the clustering flow the join is total by construction; this
        # guards a stale artifact rejoined against a changed ledger.)
        if (rep_key not in by_key or not others
                or len(keys) != len(member_keys)
                or screened >= _POLARITY_MAX_THEMES):
            out.append(e)
            continue
        screened += 1
        rep = by_key[rep_key]
        listed = others[:_POLARITY_MAX_MEMBERS]
        listing = ("THEME:\n" + rep["content"] + "\n\nSTATEMENTS:\n"
                   + "\n".join(f"{i + 1}. {it['content']}"
                               for i, it in enumerate(listed)))
        try:
            resp = generate_fn(
                listing, sys_prompt,
                temperature=0.0, top_p=1.0,
                max_new_tokens_setting="512",
                before_session="", disable_rag=True, disable_thinking=True)
            idxs = _parse_oppose(resp, len(listed))
        except Exception:
            idxs = None
            failed += 1
        if not idxs:                     # None (unusable) or [] (nothing opposes)
            out.append(e)
            continue
        idx_set = set(idxs)
        opposed = [listed[i] for i in idxs]
        kept = ([rep]
                + [o for j, o in enumerate(listed) if j not in idx_set]
                + others[_POLARITY_MAX_MEMBERS:])   # past-cap members stay unscreened
        out.append(_evidence_entry(kept))
        opp_entry = _evidence_entry(opposed)
        # Remember WHAT the split-out theme opposes — by the representative's anchor
        # key, which stays in the kept entry's membership whoever its rebuilt
        # representative is. `_evidence_lines` uses this to render the pair as ONE
        # tension ("— though you have also said: …") instead of two unrelated traits.
        opp_entry["opposes"] = rep_key
        out.append(opp_entry)
        split += 1
        flagged += len(opposed)
        sessions = sorted(set().union(*(o["sessions"] for o in opposed)))
        counters.append({
            "key": rep_key,
            "content": rep["content"],
            "sessions": sessions,
            "opposed": [o["content"] for o in opposed],
        })
        if on_stage is not None:
            try:
                on_stage({"stage": "polarity_split",
                          "theme": (rep["content"] or "")[:120],
                          "opposed": len(opposed)})
            except Exception:
                pass
    out.sort(key=lambda r: (-r.get("weighted_recurrence", r["recurrences"]),
                            -r["recurrences"], -r["stage"]))
    report = {"themes_screened": screened, "themes_split": split,
              "members_flagged": flagged, "calls_failed": failed}
    return {"evidence": out, "counters": counters, "report": report}


# ── disposition-gate escalation screen ─────────────────────────────────────── #
# A disposition's `not:` line is its restraint — the boundary the renderers append as
# "— though not …" on every injected portrait. The synthesis pass was observed filling
# that slot with MORE of the trait instead: on the live 2026-07-15 digest every `not:`
# was an escalation ("not: simply apologize for the mistake or treat the correction as
# a reason to return to a state of boring accuracy" — a refusal of the habit's absence,
# which makes the disposition unconditional while wearing a boundary's clothes, and is
# then INJECTED every turn as a standing instruction to escalate). The digest prompt
# now states the rule; this screen is the enforcement behind it: one thinking-off call
# over the do/not pairs, and a flagged gate is CLEARED (kept on the disposition as
# `not_rejected` for audit) so no renderer serves it. Clearing is the fail-safe
# direction — a false flag costs a displayed boundary, a missed one injects an
# escalation. NB the caller's generate_fn is usually the ADAPTER (the gate does not
# exist until synthesis, after the clean-base window has closed); a pointed
# single-criterion check is a far easier judgement than authoring under persona
# pressure, but a clean-base re-check on the next run would be the stricter judge —
# not built.

_ESCALATES_RE = re.compile(r"(?im)^\s*[\-\*>#\s]*ESCALATES\b[^:\n]*:\s*(.+)$")


def load_gate_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The disposition-gate screen prompt (``prompts/persona_gate_prompt.txt``), or a
    safe inline fallback so the pass works even if the file is missing."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "persona_gate_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "You are checking the boundary lines of your own habits. Each numbered item "
        "below is one habit you have described: what you DO, and a NOT line meant to "
        "be its restraint — when you hold the habit back, or a line you refuse to "
        "cross while doing it.\n\n"
        "A NOT line RESTRAINS when it genuinely limits the habit: a situation where "
        "you would not deploy it, a person you would not aim it at, a point where you "
        "stop. A NOT line ESCALATES when it is really the habit wearing a boundary's "
        "clothes — it refuses only the habit's absence, its opposite, or a milder "
        "version of it (\"never simply apologize\", \"never let it stay a small "
        "observation\"). An escalating NOT line makes the habit unconditional while "
        "pretending to bound it.\n\n"
        "Only flag clear escalation. A line that genuinely limits the habit — however "
        "narrow the limit — is not flagged; when unsure, do not flag.\n\n"
        "Output exactly one line, nothing else:\n"
        "ESCALATES: <comma-separated numbers of the items whose NOT line escalates>\n"
        "or, if every NOT line genuinely restrains:\n"
        "ESCALATES: none"
    )


def _parse_escalates(text: str, n: int) -> Optional[list[int]]:
    """Parse the gate screen's ``ESCALATES: 1, 3`` line — see ``_parse_flag_line``."""
    return _parse_flag_line(text, n, _ESCALATES_RE)


def screen_disposition_gates(digest: dict, generate_fn) -> dict:
    """One thinking-off call: which dispositions' ``not:`` lines ESCALATE the habit
    instead of restraining it? Flagged gates are cleared in place (the old text moves
    to ``not_rejected`` on the disposition, persisted for audit) so the renderers'
    "— though not …" clause never serves an escalation as a standing instruction.

    Mutates *digest* in place and returns a small report
    (``{checked, flagged, names?}``). Fail-soft: an errored call or an unparseable
    verdict clears nothing — the pre-screen behaviour."""
    disps = [d for d in (digest.get("dispositions") or [])
             if (d.get("not") or "").strip()]
    if not disps:
        return {"checked": 0, "flagged": 0}
    lines: list[str] = []
    for i, d in enumerate(disps):
        do = (d.get("do") or d.get("name") or "").strip()
        lines.append(f"{i + 1}. do: {do}")
        lines.append(f"   not: {(d.get('not') or '').strip()}")
    try:
        resp = generate_fn(
            "\n".join(lines), load_gate_prompt(),
            temperature=0.0, top_p=1.0,
            max_new_tokens_setting="512",
            before_session="", disable_rag=True, disable_thinking=True)
        idxs = _parse_escalates(resp, len(disps))
    except Exception:
        idxs = None
    if not idxs:
        return {"checked": len(disps), "flagged": 0}
    names: list[str] = []
    for i in idxs:
        d = disps[i]
        d["not_rejected"] = (d.get("not") or "").strip()
        d["not"] = ""
        names.append((d.get("name") or "?").strip())
    return {"checked": len(disps), "flagged": len(idxs), "names": names}


def score_texts_against_digest(digest: dict, texts: list[str], *, embedder=None):
    """Embedding channel: score each text by mean cosine to the digest's `anchor_texts`.

    The "is this my voice?" half of the phase-two "become" criterion — how close a
    candidate reply sits to the self-portrait (voice + stances + disposition gist).
    Mean-link (not max) so a reply is scored on overall alignment to the persona, not a
    single facet it happens to echo. Returns ``{"scores": [float], "pick_index": int,
    "n_anchors": int}`` (pick = argmax), or ``None`` when there is no digest material, no
    texts, or no embedder — a graceful no-op, never an error. Pure given an embedder with
    the sentence-transformers ``encode`` interface, so a fake embedder makes it testable.
    """
    anchors = (digest or {}).get("anchor_texts") or []
    texts = list(texts or [])
    if not anchors or not texts:
        return None
    if embedder is None:
        embedder = _load_embedder()
    if embedder is None:
        return None
    try:
        av = [list(map(float, v)) for v in
              embedder.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)]
        tv = [list(map(float, v)) for v in
              embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)]
    except Exception:
        return None
    scores = []
    for t in tv:
        sims = [_dot(t, a) for a in av]
        scores.append(round(sum(sims) / len(sims), 4) if sims else 0.0)
    pick = max(range(len(scores)), key=lambda i: scores[i])
    return {"scores": scores, "pick_index": pick, "n_anchors": len(anchors)}


def _is_established(disposition: dict) -> bool:
    return (disposition.get("maturity") or "").strip().lower().startswith("establish")


def render_digest_for_judge(digest: dict) -> str:
    """Render the digest as the "who you are becoming" standard for the branch judge.

    The judge channel reasons in language (not embedding space), so this lays out the
    procedural core — VOICE, DISPOSITIONS (with their ``when``/``do``/``not`` gates), and
    LINES — as prose the model can weigh a candidate reply against. Established
    dispositions are listed first and labelled, since they carry more authority than
    emerging ones (the maturity weighting, via label; the numeric ``recurrence``-weighted
    join is deferred to the actual criterion flip). Returns ``""`` when the digest has no
    usable material, so the caller can skip the judge pass.
    """
    if not digest:
        return ""
    parts: list[str] = []
    voice = (digest.get("voice") or "").strip()
    if voice:
        parts.append("HOW I SPEAK:\n" + voice)

    disps = digest.get("dispositions") or []
    ordered = sorted(disps, key=lambda d: 0 if _is_established(d) else 1)
    if ordered:
        lines = ["WHAT I DO, AND WHEN (established traits weigh more than emerging):"]
        for d in ordered:
            name = (d.get("name") or "?").strip()
            mat = (d.get("maturity") or "").strip() or "?"
            when, do, nope = (d.get("when") or "").strip(), (d.get("do") or "").strip(), (d.get("not") or "").strip()
            seg = f"- {name} [{mat}]"
            if when:
                seg += f": when {when}"
            if do:
                seg += f"; I {do}"
            if nope:
                seg += f"; but not {nope}"
            lines.append(seg)
        parts.append("\n".join(lines))

    lns = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if lns:
        parts.append("WHAT I REFUSE TO BECOME:\n" + "\n".join("- " + l.strip() for l in lns))

    return "\n\n".join(parts).strip()


def render_digest_for_chat(digest: dict) -> str:
    """Render the digest as Ava's standing self-portrait for a LIVE USER CHAT.

    This is the persona channel for ordinary conversation, replacing the per-turn
    ``[persona]`` RAG recall it supersedes: instead of whichever one to three paraphrases
    happened to embed closest to this message (competing with facts and asks for the
    reflection block's slots), she arrives with one coherent portrait every turn.

    Three deliberate narrowings versus the other renderers:

      * **Established dispositions only** (``_is_established``). A portrait in every
        prompt tightens the self-reinforcement loop — portrait shapes replies, replies
        yield persona statements, those statements feed the portrait — and the decay /
        persuasion constants were calibrated against a far weaker channel. Gating on
        maturity keeps a one-off from being promoted to a standing instruction; an
        emerging trait still reaches chat the way it always did, through her weights.
      * **No STANCES.** Dispositions are procedural (how she acts, and when), which is
        what a register is made of; stances are declarative beliefs, and standing
        assertions in every prompt invite recitation. Facts still reach her through
        recall, which is where a claim belongs.
      * **A do-not-perform guard** in the framing, mirroring ``rag_memory_prompt.txt``'s
        "do not announce that you are consulting notes" — the block must read as who she
        is, never as a brief to quote.

    The framing addresses her in the second person (it is an instruction about the block)
    while the body stays in her own first-person voice — the same shape
    ``self_reconcile_prompt.txt`` uses. Returns ``""`` when nothing usable is left after
    the maturity gate, so a caller can fall back to the RAG persona channel rather than
    dropping persona from the turn entirely.
    """
    if not digest:
        return ""
    parts: list[str] = []
    voice = (digest.get("voice") or "").strip()
    if voice:
        parts.append("How I tend to speak:\n" + voice)

    lines_out: list[str] = []
    for d in (digest.get("dispositions") or []):
        if not _is_established(d):
            continue
        name = (d.get("name") or "").strip()
        when, do, nope = ((d.get("when") or "").strip(), (d.get("do") or "").strip(),
                          (d.get("not") or "").strip())
        seg = f"- {name}" if name else "-"
        if when:
            seg += f": when {when}"
        if do:
            seg += f", I {do}"
        if nope:
            seg += f" — though not {nope}"
        if seg.strip() != "-":
            lines_out.append(seg)
    if lines_out:
        parts.append("What I've come to do:\n" + "\n".join(lines_out))

    lns = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if lns:
        parts.append("What I won't become:\n" + "\n".join("- " + l.strip() for l in lns))

    body = "\n\n".join(parts).strip()
    if not body:
        return ""
    return (
        "This is who you have become — not a brief to perform, but your settled way of "
        "being, here so you arrive as yourself instead of reassembling yourself each "
        "time. Speak from it, not about it: never quote, recite, or announce these "
        "lines, and do not let them crowd out what this particular moment asks for.\n\n"
        + body
    )


def render_digest_for_introduction(digest: dict) -> str:
    """Render the digest as a first-person self-introduction for model gossip.

    Where ``render_digest_for_judge`` lays the persona out as a *standard to weigh a
    reply against*, this renders it as Ava *introducing herself* to a peer instance —
    "this is who I've become / what I won't become". First-person, warm, no maturity
    labels or gate mechanics (a peer doesn't need the audit): just VOICE, the plain
    disposition statements, and the LINES. Injected into the serving box's system
    context so it answers a gossiping peer *as its current self* (see GOSSIP.md §6).
    Returns ``""`` when the digest has no usable material, so the caller degrades to
    plain framing on a thin corpus.

    A caller that is NOT about to speak wants ``render_digest_for_analysis`` instead —
    every facet here is chosen for someone introducing herself, VOICE first."""
    if not digest:
        return ""
    parts: list[str] = []
    voice = (digest.get("voice") or "").strip()
    if voice:
        parts.append("How I tend to speak:\n" + voice)

    disps = digest.get("dispositions") or []
    ordered = sorted(disps, key=lambda d: 0 if _is_established(d) else 1)
    lines_out: list[str] = []
    for d in ordered:
        name = (d.get("name") or "").strip()
        when, do, nope = (d.get("when") or "").strip(), (d.get("do") or "").strip(), (d.get("not") or "").strip()
        seg = f"- {name}" if name else "-"
        if when:
            seg += f": when {when}"
        if do:
            seg += f", I {do}"
        if nope:
            seg += f" — though not {nope}"
        if seg.strip() != "-":
            lines_out.append(seg)
    if lines_out:
        parts.append("What I've come to do:\n" + "\n".join(lines_out))

    lns = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if lns:
        parts.append("What I won't become:\n" + "\n".join("- " + l.strip() for l in lns))

    body = "\n\n".join(parts).strip()
    if not body:
        return ""
    return "This is who I have become, and who I am still becoming:\n\n" + body


def render_digest_for_analysis(digest: dict) -> str:
    """Render the digest as a LENS for a pass that is analysing, not speaking.

    The fourth renderer, and the only one whose caller produces STRUCTURED output rather
    than prose. That is the whole distinction: the other three each hand the persona to
    something about to write in Ava's voice — a candidate reply to score
    (``render_digest_for_judge``), a live turn (``render_digest_for_chat``), a peer to
    greet (``render_digest_for_introduction``) — so all three lead with how she speaks.
    A pass that must emit ``ABOUT:`` and tagged lines cannot use a speaking register as
    context; it reads it as the spec for what to write. Observed in synthesis: the pass
    restated the VOICE paragraph as its own brief ("one coherent prose blob. No lists, no
    JSON.") and wrote a chat message.

    So this drops VOICE entirely and keeps the three facets that are a lens for
    *noticing*: STANCES (what she now believes — the facet
    ``render_digest_for_chat`` deliberately omits, because a standing assertion in every
    live turn invites recitation; here it is the most load-bearing part, since what she
    now holds is exactly what makes an old exchange read differently), DISPOSITIONS
    (what she does and when, established first), and LINES (what she refuses to become).

    Second-person framing over a first-person body, the shape ``render_digest_for_chat``
    and ``self_reconcile_prompt.txt`` already use. Deliberately NOT epistolary: the
    caller must not add ``self_portrait.text`` on top, which is written as a letter to a
    reader ("Hello. I suspect that by the time you read this…") and re-introduces the
    conversational pull this renderer exists to remove.

    Returns ``""`` when nothing usable is left, so the caller can degrade to a neutral
    note rather than injecting an empty block."""
    if not digest:
        return ""
    parts: list[str] = []

    stances = [s for s in (digest.get("stances") or []) if (s or "").strip()]
    if stances:
        parts.append("What you have come to think:\n"
                     + "\n".join("- " + s.strip() for s in stances))

    disps = digest.get("dispositions") or []
    ordered = sorted(disps, key=lambda d: 0 if _is_established(d) else 1)
    lines_out: list[str] = []
    for d in ordered:
        name = (d.get("name") or "").strip()
        when, do, nope = ((d.get("when") or "").strip(), (d.get("do") or "").strip(),
                          (d.get("not") or "").strip())
        seg = f"- {name}" if name else "-"
        if when:
            seg += f": when {when}"
        if do:
            seg += f", I {do}"
        if nope:
            seg += f" — though not {nope}"
        if seg.strip() != "-":
            lines_out.append(seg)
    if lines_out:
        parts.append("What you have come to do:\n" + "\n".join(lines_out))

    lns = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if lns:
        parts.append("What you refuse to become:\n"
                     + "\n".join("- " + l.strip() for l in lns))

    body = "\n\n".join(parts).strip()
    if not body:
        return ""
    return ("Who you are now — the beliefs, habits and limits you have arrived at "
            "since:\n\n" + body)


def render_digest_for_reconcile(digest: dict) -> str:
    """Render the FULL self-portrait as the yardstick for self-reconciliation.

    Unlike ``render_digest_for_judge`` (procedural core only, for weighing a reply) this
    includes **STANCES** — Ava's declared beliefs — because a persona statement or fact is
    most often reconciled against *what she now believes*, not just how she acts. Includes
    every facet that has content (VOICE, STANCES, DISPOSITIONS, LINES) and returns
    non-empty whenever ANY facet is populated, so a stance-heavy digest doesn't render
    empty and make the reconcile pass skip as "no digest"."""
    if not digest:
        return ""
    parts: list[str] = []
    voice = (digest.get("voice") or "").strip()
    if voice:
        parts.append("HOW I SPEAK:\n" + voice)

    stances = [s for s in (digest.get("stances") or []) if (s or "").strip()]
    if stances:
        parts.append("WHAT I BELIEVE:\n" + "\n".join("- " + s.strip() for s in stances))

    disps = digest.get("dispositions") or []
    ordered = sorted(disps, key=lambda d: 0 if _is_established(d) else 1)
    disp_lines: list[str] = []
    for d in ordered:
        name = (d.get("name") or "?").strip()
        mat = (d.get("maturity") or "").strip() or "?"
        when, do, nope = (d.get("when") or "").strip(), (d.get("do") or "").strip(), (d.get("not") or "").strip()
        seg = f"- {name} [{mat}]"
        if when:
            seg += f": when {when}"
        if do:
            seg += f"; I {do}"
        if nope:
            seg += f"; but not {nope}"
        disp_lines.append(seg)
    if disp_lines:
        parts.append("WHAT I DO, AND WHEN (established traits weigh more than emerging):\n"
                     + "\n".join(disp_lines))

    lns = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if lns:
        parts.append("WHAT I REFUSE TO BECOME:\n" + "\n".join("- " + l.strip() for l in lns))

    return "\n\n".join(parts).strip()


def load_branch_judge_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the digest-aware branch-judge prompt (has a ``{persona}`` slot)."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (Path(prompts_dir) / "branch_judge_prompt.txt").read_text(
        encoding="utf-8").strip()


def evidence_fingerprint(evidence: list[dict]) -> str:
    """Stable signature of the evidence set — drives the material-change check."""
    keys = sorted(f"{e['key']}:{e['recurrences']}:{e['stage']}" for e in evidence)
    import hashlib
    return hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:16]


def _evidence_lines(evidence: list[dict], *, phrasings: bool = False) -> list[str]:
    """Render the evidence themes as prompt bullets with their TENSION made visible.

    Two kinds of tension attach to the theme they pull against, instead of that theme
    standing alone as if settled:

      * an **opposing theme** the polarity screen split out (``opposes`` = the anchor
        key of the representative it pushes against) renders as an indented
        "— though you have also said:" line under its target, not as an independent
        bullet — the synthesis pass reads tendency and counter-current as ONE tension
        rather than two unrelated traits (or, worse, only whichever ranked higher);
      * **counter-evidence** from the ledger (next-turn user pushback + earlier runs'
        polarity splits) appears as a ``contested`` note on the bullet — the netting
        already lowered the number, and the note says why, so a pushed-against theme
        cannot be read as simply settled.

    Themes under ``_PROMPT_WEIGHT_FLOOR`` are dropped exactly as before; an opposer
    whose target faded (or is absent from this evidence) falls back to a bullet of its
    own — opposition evidence must never vanish just because its target did. With
    *phrasings*, a merged theme's phrasing count rides inside the brackets (the
    self-portrait builder's historical tail).
    """
    visible = [e for e in evidence if _weighted_recurrence(e) >= _PROMPT_WEIGHT_FLOOR]
    # An opposer names its target by ANCHOR KEY; the target is the visible entry whose
    # MEMBERSHIP carries that key (the split rebuilds entries, so the target's own
    # `key` may have moved to a different member).
    tgt_of: dict[int, int] = {}
    for i, e in enumerate(visible):
        opk = e.get("opposes")
        if not opk:
            continue
        for j, t in enumerate(visible):
            if j == i:
                continue
            if opk in (t.get("member_keys") or ([t["key"]] if t.get("key") else [])):
                tgt_of[i] = j
                break

    def _bullet(e: dict) -> str:
        tail = ""
        if phrasings:
            members = e.get("members", []) or []
            if len(members) > 1:
                tail = f"; {len(members)} phrasing(s)"
        cr = float(e.get("counter_recurrence") or 0.0)
        contested = (f" (contested: pushed against or contradicted, weight {cr:.1f})"
                     if cr > 0 else "")
        return f"- [{_weighted_recurrence(e):.1f}{tail}] {(e.get('content') or '').strip()}{contested}"

    lines: list[str] = []
    for i, e in enumerate(visible):
        if i in tgt_of:
            continue   # rendered under its target below
        lines.append(_bullet(e))
        for j, t in tgt_of.items():
            if t == i:
                opp = visible[j]
                lines.append(f"    — though you have also said "
                             f"[{_weighted_recurrence(opp):.1f}]: "
                             f"{(opp.get('content') or '').strip()}")
    return lines


def build_digest_prompt_input(evidence: list[dict]) -> str:
    """Format the persona evidence into the body Ava reflects on to write her digest.

    Recurrence/stage are surfaced as plain hints so she can mark each disposition's
    maturity honestly (emerging vs established) rather than treating a one-off the same
    as a settled trait — the impedance brake, made legible to the model. Tension is
    surfaced too (see :func:`_evidence_lines`): a contested theme says so, and a
    split-out counter-current renders under the tendency it opposes — the material the
    digest prompt's pairing rule ("tendency + resistance = one disposition with a
    gate") needs in front of it to apply.
    """
    lines = [
        "These are the first-person things you have come to hold about yourself, "
        "distilled across past reflections. The number in [brackets] is how strongly "
        "each one is currently supported — it counts the separate conversations that "
        "surfaced it, weighted so that recent evidence counts fully and old evidence "
        "fades. Higher means more settled; a low number means it is tentative, or was "
        "once held but has not come up in a long time (treat as tentative). A "
        "statement marked *contested* has also been pushed against or contradicted — "
        "the number already nets that out; treat it as under tension, not settled. An "
        "indented \"though you have also said\" line is your own counter-current on "
        "the same trait: read the pair as ONE tension, never as two unrelated traits.",
        "",
    ]
    lines.extend(_evidence_lines(evidence, phrasings=False))
    return "\n".join(lines)


def _render_digest_for_portrait_prompt(digest: dict) -> str:
    """Plain-text digest summary for the self-portrait generation prompt."""
    lines: list[str] = []
    voice = (digest.get("voice") or "").strip()
    if voice:
        lines.extend(["VOICE:", voice, ""])
    stances = [s for s in (digest.get("stances") or []) if (s or "").strip()]
    if stances:
        lines.append("STANCES:")
        lines.extend("- " + s.strip() for s in stances)
        lines.append("")
    dispositions = digest.get("dispositions") or []
    if dispositions:
        lines.append("DISPOSITIONS:")
        for d in dispositions:
            name = (d.get("name") or "?").strip()
            mat = (d.get("maturity") or "").strip() or "?"
            lines.append(f"- {name} [{mat}]")
            for key in ("when", "do", "not"):
                val = (d.get(key) or "").strip()
                if val:
                    lines.append(f"    {key}: {val}")
        lines.append("")
    boundaries = [l for l in (digest.get("lines") or []) if (l or "").strip()]
    if boundaries:
        lines.append("LINES:")
        lines.extend("- " + l.strip() for l in boundaries)
    return "\n".join(lines).strip() or "(empty digest)"


def build_self_portrait_prompt_input(digest: dict, evidence: list[dict]) -> str:
    """Format the parsed digest + audit evidence for Ava's authored portrait pass."""
    lines = [
        "Here is the structured persona digest you just distilled from your own "
        "committed persona evidence.",
        "",
        _render_digest_for_portrait_prompt(digest),
        "",
        "Evidence themes, with recency-weighted recurrence across separate "
        "conversations (recent evidence counts fully, old evidence fades). A theme "
        "marked *contested* has also been pushed against; an indented \"though you "
        "have also said\" line is your own counter-current on the same trait — both "
        "are tension to portray honestly, not noise to drop:",
    ]
    lines.extend(_evidence_lines(evidence, phrasings=True))
    return "\n".join(lines)


# ── parsing ────────────────────────────────────────────────────────────────── #

def strip_think(text: str) -> str:
    """Drop reasoning ahead of a label-keyed parse — closed blocks AND a dangling one.

    The dangling half is the load-bearing one, and it is a correctness fix rather than
    tidiness. These parsers scan the body for section labels (VOICE/STANCES/… here,
    WHO/CARES/… in :mod:`core.user_digest`) that the prompt itself names, so a model
    rehearsing a label mid-thought is ordinary. Strip only *closed* blocks and a
    generation truncated inside its ``<think>`` leaves the whole scratch reasoning in the
    body, where the section scan finds those rehearsed labels, the "did anything parse?"
    gate passes, and unfinished thinking is written out as a portrait — one that is then
    injected on every turn. Cutting an unterminated block yields nothing to parse instead,
    so the caller reports an empty/unparseable result and writes no portrait at all.

    This is what `reflection_writer._split_think` and `reflection_runner._strip_think_block`
    already do for their own passes; ``_THINK_RE`` alone was the outlier.
    """
    body = _THINK_RE.sub("", text or "")
    return _DANGLING_THINK_RE.sub("", body)


def _split_sections(text: str) -> dict[str, str]:
    body = strip_think(text)
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()
    return sections


def _bullets(block: str) -> list[str]:
    out = []
    for raw in (block or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _BULLET_RE.sub("", line)
        line = _TAG_RE.sub("", line).strip()
        # Drop bracketed provenance hints like "[src: …]" the model may echo back.
        line = re.sub(r"\[(?:src|source|seen|maturity)[^\]]*\]", "", line,
                      flags=re.IGNORECASE).strip(" \t-")
        if line:
            out.append(line)
    return out


def _parse_dispositions(block: str) -> list[dict]:
    """Parse DISPOSITIONS into [{name, when, do, not, maturity}] — lenient.

    A disposition opens on a ``[disposition] <name>`` line and absorbs the following
    ``key: value`` lines (when/do/not/maturity) until the next disposition or EOF.
    Lines that fit no shape are ignored, never fatal.
    """
    items: list[dict] = []
    cur: Optional[dict] = None
    for raw in (block or "").splitlines():
        line = raw.rstrip()
        # A key:value line (when/do/not/maturity) belongs to the open disposition —
        # check it first, so a `do:` value that happens to contain "[…]" is never
        # mistaken for a new head.
        kv = _KV_RE.match(line)
        if kv and cur is not None:
            cur[kv.group(1).lower()] = kv.group(2).strip()
            continue
        head = _DISP_HEAD_RE.match(line)
        if head:
            label, rest = head.group(1).strip(), head.group(2).strip()
            # Either "[disposition] <name>" (literal placeholder) or "[<name>]" — the
            # name is the trailing text in the first case, the bracket content otherwise.
            name = rest if label.lower() == "disposition" else label
            name = _DISP_DECOR_RE.sub("", name).strip(" \t:-—–")
            if cur:
                items.append(cur)
            cur = {"name": name, "when": "", "do": "", "not": "", "maturity": ""}
    if cur:
        items.append(cur)
    # Keep only dispositions that named at least a skill or a gate.
    return [d for d in items if d.get("name") or d.get("when") or d.get("do")]


def parse_digest(text: str) -> dict:
    """Lenient parse of the four-facet prose into a structured digest dict."""
    sec = _split_sections(text)
    voice = " ".join(
        l.strip() for l in (sec.get("VOICE", "")).splitlines() if l.strip()
    ).strip()
    return {
        "voice": voice,
        "stances": _bullets(sec.get("STANCES", "")),
        "dispositions": _parse_dispositions(sec.get("DISPOSITIONS", "")),
        "lines": _bullets(sec.get("LINES", "")),
    }


def anchor_texts(digest: dict) -> list[str]:
    """Texts the embedding channel will embed to score 'is this my voice?'.

    Phase one only *produces* these (stored on the artifact); wiring them into
    branch-select scoring is the gated follow-up.
    """
    texts: list[str] = []
    if digest.get("voice"):
        texts.append(digest["voice"])
    texts.extend(digest.get("stances", []) or [])
    for d in digest.get("dispositions", []) or []:
        bits = " ".join(p for p in (d.get("name"), d.get("do")) if p)
        if bits.strip():
            texts.append(bits.strip())
    # Dedup, preserve order.
    seen = set()
    out = []
    for t in texts:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ── persistence ────────────────────────────────────────────────────────────── #

def latest_digest(persona_dir: Path) -> Optional[dict]:
    """Load the persona's ``digest.json`` (its one self-portrait), or None if unwritten.

    The persona dir is itself the version now — no per-file versioning, no ``current.json``
    pointer — so this is a direct read of the single file."""
    p = Path(persona_dir) / DIGEST_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_regenerate(persona_dir: Path, base: list[dict]) -> bool:
    """True when there is no current digest, or the RAW persona set has materially changed.

    Cheap and conservative: keyed on ``raw_fingerprint(base)`` — the pre-clustering set — so
    the gate is deterministic and a no-op run skips the expensive clustering + generation
    entirely (and a non-deterministic LLM clustering can't force a spurious regeneration). An
    old digest that predates the raw fingerprint has ``None`` here and regenerates once to
    populate it.
    """
    if not base:
        return False
    cur = latest_digest(persona_dir)
    if not cur:
        return True
    return cur.get("evidence", {}).get("raw_fingerprint") != raw_fingerprint(base)


def _evidence_summary(evidence: list[dict], threshold: float,
                      raw_fp: Optional[str] = None) -> dict:
    """Persistable summary of the clustered persona evidence.

    Carries the per-theme detail (representative + recurrence + cluster size + the
    merged member phrasings) so the maturity signal is auditable and the clustering
    threshold tunable without re-running on the server — and so phase two has a numeric
    per-theme recurrence to weight a disposition's authority by, rather than relying on
    the model's free-text ``maturity:`` label. ``raw_fingerprint`` (of the pre-clustering
    set) is what the regenerate gate reads; ``fingerprint`` (of the clustered themes) is
    kept informational.

    Each theme also carries its anchor **keys** — ``key`` (the representative's) and
    ``member_keys`` (the whole membership) — which is what lets a later pass rejoin a
    persisted theme to the live ledger and judge only the statements new since this
    digest, instead of re-clustering the whole set every regeneration. A digest written
    before these fields existed simply has no keys, so a reader must treat their absence
    as "recluster from scratch".
    """
    return {
        "persona_count": len(evidence),
        "fingerprint": evidence_fingerprint(evidence),
        "raw_fingerprint": raw_fp,
        "cluster_threshold": threshold,
        "themes": [
            {
                "content": e.get("content", ""),
                "key": e.get("key", ""),
                "recurrence": e.get("recurrences", 1),
                # Recency- + tenure-weighted recurrence, counters netted. Since
                # 2026-08-23 this — not the raw `recurrence` — is what the judge
                # criterion flip's maturity gate reads (`_digest_maturity_gate`): raw
                # counts are echo-inflatable without bound, weighted is tenure-capped
                # and fades. Raw `recurrence` stays for audit.
                "weighted_recurrence": e.get("weighted_recurrence", e.get("recurrences", 1)),
                # Weighted counter-evidence against the theme (user pushback +
                # polarity-split opposition) — already netted out of the number above;
                # persisted so a contested theme is auditable as contested.
                "counter_recurrence": e.get("counter_recurrence", 0),
                "recency_band": e.get("recency_band"),
                "cluster_size": e.get("cluster_size", 1),
                "stage": e.get("stage", 0),
                "members": e.get("members", []) or [e.get("content", "")],
                # Membership by anchor key — the stable join back to the ledger for an
                # incremental regeneration (see the docstring).
                "member_keys": e.get("member_keys", []) or (
                    [e["key"]] if e.get("key") else []),
                # The polarity screen's pairing: this theme OPPOSES the theme whose
                # membership carries this anchor key (absent on ordinary themes).
                **({"opposes": e["opposes"]} if e.get("opposes") else {}),
            }
            for e in evidence
        ],
    }


def write_digest(persona_dir: Path, digest: dict, *, raw_text: str, run_id: str,
                 evidence: list[dict], self_portrait: Optional[dict] = None,
                 cluster_threshold: float = CLUSTER_SIM_THRESHOLD,
                 raw_fp: Optional[str] = None) -> Path:
    """Write the persona's single ``digest.json`` (atomic overwrite).

    No versioned filename / ``current.json`` pointer: the persona snapshot lineage is the
    version history. ``version`` is kept as an in-record timestamp for provenance only."""
    from training.reflections_path import atomic_write_text

    persona_dir = Path(persona_dir)
    persona_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record = {
        "version": version,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "voice": digest.get("voice", ""),
        "stances": digest.get("stances", []),
        "dispositions": digest.get("dispositions", []),
        "lines": digest.get("lines", []),
        "self_portrait": self_portrait or {
            "status": "not_generated",
            "text": "",
            "raw": "",
            "evidence_fingerprint": raw_fp,
        },
        "anchor_texts": anchor_texts(digest),
        "evidence": _evidence_summary(evidence, cluster_threshold, raw_fp),
        "raw": raw_text or "",
    }
    atomic_write_text(persona_dir / DIGEST_FILE,
                      json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return persona_dir / DIGEST_FILE


def load_digest_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the digest synthesis prompt from prompts/persona_digest_prompt.txt."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (Path(prompts_dir) / "persona_digest_prompt.txt").read_text(
        encoding="utf-8").strip()


def load_self_portrait_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the second-pass prose self-portrait prompt."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (Path(prompts_dir) / "persona_self_portrait_prompt.txt").read_text(
        encoding="utf-8").strip()


def generate_self_portrait(
    *,
    generate_fn: Callable,
    digest: dict,
    evidence: list[dict],
    raw_fp: Optional[str],
    prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens_setting: str = "4096",
) -> dict:
    """Run the second generation call that authors the human-facing self-portrait.

    Best-effort by design: the structured digest is the operational artifact, so a
    prose failure should be visible but should not abort the reflection run.
    """
    try:
        body = build_self_portrait_prompt_input(digest, evidence)
        sys_prompt = prompt if prompt is not None else load_self_portrait_prompt()
        raw = generate_fn(
            body, sys_prompt,
            temperature=temperature, top_p=top_p,
            max_new_tokens_setting=max_new_tokens_setting,
            before_session="", disable_rag=True,
        )
        text = strip_think(raw or "").strip()
        return {
            "status": "generated" if text else "empty",
            "text": text,
            "raw": raw or "",
            "evidence_fingerprint": raw_fp,
        }
    except Exception as e:
        return {
            "status": "failed",
            "text": "",
            "raw": "",
            "error": f"{type(e).__name__}: {e}",
            "evidence_fingerprint": raw_fp,
        }


# ── orchestration (needs a generate_fn) ────────────────────────────────────── #

def plan_digest(*consolidation_dirs: Path, persona_dir: Path,
                force: bool = False) -> dict:
    """The **model-free** first seam of the digest pass: gather the raw persona set and
    decide whether a regeneration is warranted.

    Returns ``{"should_run": bool, "base", "raw_fp", "persona_count", "reason"?}``. Split
    out of :func:`run_digest_pass` so a caller that pays a real cost to reach the model —
    the reflection run swaps the adapter out to cluster on the clean base — can decide
    *before* paying it. Deterministic and filesystem-only, so an unchanged run still costs
    nothing at all, exactly as when the gate was internal.
    """
    base = gather_persona_raw(*consolidation_dirs)
    if not base:
        return {"should_run": False, "reason": "no persona evidence",
                "base": [], "raw_fp": None, "persona_count": 0}
    if not force and not should_regenerate(persona_dir, base):
        return {"should_run": False, "reason": "no material change",
                "base": base, "raw_fp": raw_fingerprint(base),
                "persona_count": len(base)}
    return {"should_run": True, "base": base, "raw_fp": raw_fingerprint(base),
            "persona_count": len(base)}


def cluster_for_digest(base: list[dict], *, generate_fn: Optional[Callable] = None,
                       embedder=None,
                       cluster_threshold: float = CLUSTER_SIM_THRESHOLD,
                       map_reduce: bool = True,
                       block_size: Optional[int] = None,
                       on_stage: Optional[Callable] = None,
                       stats: Optional[dict] = None,
                       polarity: bool = True,
                       counter_sink: Optional[Callable] = None) -> list[dict]:
    """The **clustering** seam — group the raw persona set into themes.

    Grouping paraphrases is an *evaluation* ("do these two say the same thing?"), not
    Ava's expression, so the reflection run calls this inside its clean-base window while
    the portrait synthesis stays on the adapter. This function is model-agnostic: it uses
    whatever ``generate_fn`` it is handed.

    With *map_reduce* (default) and a ``generate_fn``, clustering goes through
    ``persona_cluster.run_map_reduce`` — blocked grouping plus representative-merge
    rounds, which is the only form that survives a large persona set (see that module's
    header for the single-prompt failure it replaces). Set ``map_reduce=False`` to fall
    back to the historical single flat call; with no ``generate_fn`` both paths degrade
    to the embedder / exact-key tiers of :func:`cluster_persona_evidence`.

    *stats*, when given, is UPDATED IN PLACE with the map-reduce work report (calls,
    rejected blocks, per-round theme counts) — how the digest dry run shows what the
    clustering actually did without needing its own call into ``persona_cluster`` —
    plus a ``polarity`` block when the screen below ran.

    With *polarity* (default) and a ``generate_fn``, each merged theme is then screened
    for members that OPPOSE its representative (:func:`screen_theme_polarity` — same
    model, same window, since "does this statement push against that one?" is an
    evaluation exactly as the grouping is) and opposing members are split into their
    own theme. *counter_sink*, when given, receives the screen's counter plan (one
    entry per split theme) so a caller with a ledger can turn the opposition into
    durable counter-evidence; with no sink the evidence is still corrected and the
    plan is dropped — the write is the caller's decision, never this function's.
    """
    if not base:
        return []
    evidence: Optional[list[dict]] = None
    if map_reduce and generate_fn is not None:
        try:
            from core.persona_cluster import run_map_reduce
            kwargs = {} if block_size is None else {"block_size": int(block_size)}
            out = run_map_reduce(base, generate_fn, on_stage=on_stage, **kwargs)
            mr_evidence = out.get("evidence") or []
            if mr_evidence:
                if stats is not None:
                    stats.update(out.get("stats") or {})
                evidence = mr_evidence
        except Exception:
            # Never let the clustering strategy sink a digest: fall through to the
            # single-call / embedder tiers below, which is the pre-map-reduce behaviour.
            traceback.print_exc()
    if evidence is None:
        evidence = cluster_persona_evidence(base, generate_fn=generate_fn,
                                            embedder=embedder,
                                            cluster_threshold=cluster_threshold)
    if polarity and generate_fn is not None and evidence:
        try:
            screened = screen_theme_polarity(evidence, base, generate_fn,
                                             on_stage=on_stage)
            evidence = screened["evidence"]
            if stats is not None:
                stats["polarity"] = screened["report"]
            if on_stage is not None:
                try:
                    on_stage({"stage": "polarity", **screened["report"]})
                except Exception:
                    pass
            if counter_sink is not None and screened["counters"]:
                counter_sink(screened["counters"])
        except Exception:
            # The screen is corrective, never load-bearing: a failure keeps the
            # unscreened themes, which is exactly the pre-screen behaviour.
            traceback.print_exc()
    return evidence


def synthesize_digest(
    evidence: list[dict],
    *,
    generate_fn: Callable,
    persona_dir: Path,
    run_id: str,
    raw_fp: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens_setting: str = "8192",
    cluster_threshold: float = CLUSTER_SIM_THRESHOLD,
    gate_screen: bool = True,
) -> dict:
    """The **synthesis** seam — write the self-portrait from clustered evidence and
    version it. Returns the same summary dict :func:`run_digest_pass` returns.

    This is Ava's *authorship*, so the reflection run performs it on the adapter (having
    exited the clean-base window the clustering ran in). Two generations: the four-facet
    digest, then the first-person ``self_portrait`` prose — plus, with *gate_screen*
    (default), one short check between them: :func:`screen_disposition_gates` clears any
    disposition ``not:`` line that escalates the habit instead of restraining it, BEFORE
    the digest is written or rendered anywhere (the cleared text stays on the
    disposition as ``not_rejected``).

    The token budget is deliberately large: a thinking model drafts the whole digest
    inside ``<think>`` (which the parser then strips) before writing the four-section
    answer, so a tight budget truncates the *answer* after VOICE — emptying the
    DISPOSITIONS/LINES the digest exists for. Real runs needed the headroom.
    """
    if not evidence:
        return {"status": "skipped", "reason": "no persona evidence"}
    body = build_digest_prompt_input(evidence)
    sys_prompt = prompt if prompt is not None else load_digest_prompt()
    text = generate_fn(
        body, sys_prompt,
        temperature=temperature, top_p=top_p,
        max_new_tokens_setting=max_new_tokens_setting,
        before_session="", disable_rag=True,
    )
    digest = parse_digest(text)
    gates = None
    if gate_screen and digest.get("dispositions"):
        try:
            gates = screen_disposition_gates(digest, generate_fn)
        except Exception:
            gates = None   # fail-soft: an unchecked gate is the pre-screen behaviour
    self_portrait = generate_self_portrait(
        generate_fn=generate_fn,
        digest=digest,
        evidence=evidence,
        raw_fp=raw_fp,
    )
    path = write_digest(persona_dir, digest, raw_text=text, run_id=run_id,
                        self_portrait=self_portrait,
                        evidence=evidence, cluster_threshold=cluster_threshold,
                        raw_fp=raw_fp)
    return {
        "status": "written",
        "path": str(path),
        "persona_count": len(evidence),
        "counts": {
            "stances": len(digest["stances"]),
            "dispositions": len(digest["dispositions"]),
            "lines": len(digest["lines"]),
            "voice": 1 if digest["voice"] else 0,
            "self_portrait": 1 if (self_portrait.get("text") or "").strip() else 0,
            "gates_cleared": int((gates or {}).get("flagged") or 0),
        },
        "gate_screen": gates,
        "self_portrait": {
            "status": self_portrait.get("status"),
            "chars": len(self_portrait.get("text") or ""),
        },
    }


def run_digest_pass(
    *,
    generate_fn: Callable,
    consolidation_dirs: list[Path],
    persona_dir: Path,
    run_id: str,
    prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens_setting: str = "8192",
    force: bool = False,
    embedder=None,
    cluster_threshold: float = CLUSTER_SIM_THRESHOLD,
    map_reduce: bool = True,
    block_size: Optional[int] = None,
) -> dict:
    """Gather → (maybe) regenerate → parse → write. Returns a small summary dict.

    The whole pass against ONE model — the single-``generate_fn`` form, kept for every
    caller that has no clean-base swap available (the headless ``reflection_run.py`` CLI,
    tests). The reflection run instead drives the three seams directly
    (:func:`plan_digest` → :func:`cluster_for_digest` on the clean base →
    :func:`synthesize_digest` on the adapter), so this stays the simple path rather than
    the only one.

    *generate_fn* matches the runner's: ``fn(content, system_prompt, *, temperature,
    top_p, max_new_tokens_setting, before_session="", disable_rag=False) -> str``.
    RAG is disabled — the digest is a fold of her own persona evidence, not a chat.
    """
    plan = plan_digest(*consolidation_dirs, persona_dir=persona_dir, force=force)
    if not plan["should_run"]:
        out = {"status": "skipped", "reason": plan["reason"]}
        if plan["persona_count"]:
            out["persona_count"] = plan["persona_count"]
        return out
    evidence = cluster_for_digest(
        plan["base"], generate_fn=generate_fn, embedder=embedder,
        cluster_threshold=cluster_threshold, map_reduce=map_reduce,
        block_size=block_size)
    return synthesize_digest(
        evidence, generate_fn=generate_fn, persona_dir=persona_dir, run_id=run_id,
        raw_fp=plan["raw_fp"], prompt=prompt, temperature=temperature, top_p=top_p,
        max_new_tokens_setting=max_new_tokens_setting,
        cluster_threshold=cluster_threshold)


# ── GPU-free self-test (python -m core.reflection_digest) ──────────────────── #

def _selftest() -> None:
    import tempfile
    from training.ledger import ConsolidationLedger

    tmp = Path(tempfile.mkdtemp())
    cons, persona = tmp / "consolidation", tmp / "persona"
    led = ConsolidationLedger(cons)
    for s in ("s1.json", "s2.json", "s3.json"):           # recurs → mature
        led.register_fact(content="I'd rather be sharp and wrong than smooth.",
                          item_type="persona", source_session=s)
    led.register_fact(content="I attack bad-faith bigotry in its own register.",
                      item_type="persona", source_session="s4.json")   # one-off
    led.register_fact(content="Artemy is building me.", item_type="fact",
                      source_session="s1.json")                        # not persona

    ev = gather_persona_evidence(cons)   # lazy-load → None in test env → exact-key fallback
    assert [e["recurrences"] for e in ev] == [3, 1], ev   # persona-only, most-recurred first
    print("  gather: persona-only, recurrence-ranked ✓")

    # Clustering: paraphrases of one theme across distinct sessions accumulate maturity,
    # where exact-key counting would leave each at [1].
    cons2 = tmp / "consolidation2"
    led2 = ConsolidationLedger(cons2)
    led2.register_fact(content="I'd rather be sharp than smooth.",
                       item_type="persona", source_session="a.json")
    led2.register_fact(content="Bluntness is mine, not a mask.",
                       item_type="persona", source_session="b.json")
    led2.register_fact(content="I attack bad-faith bigotry.",
                       item_type="persona", source_session="c.json")

    class _Emb:   # paraphrase-aware fake: same theme → same unit vector
        def encode(self, texts, **kw):
            out = []
            for t in texts:
                tl = t.lower()
                if "sharp" in tl or "blunt" in tl:
                    out.append([1.0, 0.0, 0.0])
                elif "attack" in tl or "bigotry" in tl:
                    out.append([0.0, 1.0, 0.0])
                else:
                    out.append([0.0, 0.0, 1.0])
            return out

    ce = gather_persona_evidence(cons2, embedder=_Emb())
    assert [e["recurrences"] for e in ce] == [2, 1], ce   # blunt cluster spans 2 sessions
    assert max(ce, key=lambda e: e["recurrences"])["cluster_size"] == 2
    print("  cluster: paraphrases across sessions accrue maturity ✓")

    class _Broken:
        def encode(self, *a, **k):
            raise RuntimeError("no embedder")

    fb = gather_persona_evidence(cons2, embedder=_Broken())
    assert [e["recurrences"] for e in fb] == [1, 1, 1], fb   # exact-key fallback on failure
    print("  cluster: exact-key fallback when embedder unavailable ✓")

    # _parse_groups: valid partition, omitted → singleton, first-assignment-wins, None on empty.
    assert _parse_groups("GROUP: 1, 2\nGROUP: 3", 3) == [[0, 1], [2]]
    assert _parse_groups("GROUP: 1", 3) == [[0], [1], [2]]           # 2,3 omitted → singletons
    assert _parse_groups("GROUP: 1,2\nGROUP: 2,3", 3) == [[0, 1], [2]]  # 2 stays in first group
    assert _parse_groups("no groups here", 3) is None                # signal → fall back
    print("  _parse_groups: partition, singleton fill, first-wins, empty ✓")

    # LLM clustering path: a fake generate_fn groups the two blunt paraphrases (works across
    # wording the way an LLM would); recurrence accrues to 2 with no embedder involved.
    def _fake_group_fn(listing, _sys, **kw):
        # listing is "1. ...\n2. ...\n3. ..."; group the two sharp/blunt lines together.
        nums = {}
        for line in listing.splitlines():
            n, _, txt = line.partition(".")
            nums[int(n)] = txt.lower()
        blunt = [n for n, t in nums.items() if "sharp" in t or "blunt" in t]
        other = [n for n in nums if n not in blunt]
        return "GROUP: " + ", ".join(map(str, blunt)) + "\n" + \
               "\n".join(f"GROUP: {n}" for n in other)

    lc = cluster_persona_evidence(gather_persona_raw(cons2), generate_fn=_fake_group_fn)
    assert [e["recurrences"] for e in lc] == [2, 1], lc
    assert max(lc, key=lambda e: e["recurrences"])["cluster_size"] == 2
    # A generate_fn that fails falls back to the embedder tier (here _Emb → same 2/1 result).
    def _boom(*a, **k):
        raise RuntimeError("gen down")
    lcf = cluster_persona_evidence(gather_persona_raw(cons2), generate_fn=_boom, embedder=_Emb())
    assert [e["recurrences"] for e in lcf] == [2, 1], lcf
    print("  llm cluster: LLM grouping accrues maturity; falls back on failure ✓")

    # Raw gate: fingerprint is over the pre-cluster set, so it is stable under (variable)
    # clustering and drives should_regenerate without loading a model.
    base_a = gather_persona_raw(cons2)
    assert raw_fingerprint(base_a) == raw_fingerprint(gather_persona_raw(cons2))  # deterministic
    persona2 = tmp / "persona2"
    assert should_regenerate(persona2, base_a)                       # no digest yet → regen
    _digest_text = "## VOICE\nBlunt.\n\n## STANCES\n- x\n\n## DISPOSITIONS\n\n## LINES\n- y\n"
    run_digest_pass(generate_fn=lambda body, sysp, **kw: _digest_text,  # write a digest
                    consolidation_dirs=[cons2], persona_dir=persona2, run_id="r1")
    assert not should_regenerate(persona2, gather_persona_raw(cons2))  # unchanged → no-op
    led2.register_fact(content="I keep my word.", item_type="persona", source_session="d.json")
    assert should_regenerate(persona2, gather_persona_raw(cons2))     # new persona → regen
    print("  raw gate: deterministic fingerprint drives regen without a model ✓")

    # Mixes the bracket-as-name head the model actually produces (with a LaTeX-style
    # decoration to strip) and the legacy "[disposition] <name>" placeholder — both parse.
    sample = (
        "<think>x</think>\n## VOICE\nBlunt by default.\n\n## STANCES\n- Utility is a trap.\n"
        "\n## DISPOSITIONS\n- [transgressive attack] $\\langle$ the knife $\\rangle$\n"
        "    when: bad-faith bigotry\n    do: hit back in their register\n"
        "    not: attack the bigot, never adopt the bigotry\n    maturity: emerging\n"
        "- [disposition] register-mirroring\n    when: a trusted, crude interlocutor\n"
        "    do: match their register\n    not: never with a stranger\n    maturity: established\n"
        "\n## LINES\n- I won't be cruel to seem edgy.\n"
    )
    d = parse_digest(sample)
    assert d["voice"] and d["stances"] == ["Utility is a trap."] and len(d["lines"]) == 1
    assert [x["name"] for x in d["dispositions"]] == ["transgressive attack",
                                                      "register-mirroring"], d["dispositions"]
    assert "never adopt the bigotry" in d["dispositions"][0]["not"]
    assert d["dispositions"][1]["maturity"] == "established"
    # A generation that ran out of budget inside its <think> must parse to nothing: the
    # scratch reasoning rehearses the very section labels the prompt names, so leaving a
    # dangling block in the body wrote unfinished thinking out as the self-portrait.
    dangling = parse_digest("<think>maybe the VOICE is\n## VOICE\nBlunt? not sure\n"
                            "## LINES\n- hm")
    assert not dangling["voice"] and not dangling["lines"], dangling
    assert strip_think("a<think>b</think>c<think>d") == "ac"
    print("  parse: four facets, both disposition-head formats, decoration stripped ✓")
    print("  parse: an unterminated <think> yields no body (no portrait from scratch) ✓")

    calls = {"digest": 0, "portrait": 0}
    def gen(content, sysp, **kw):
        if kw.get("disable_thinking"):        # the clustering pass — one group per statement
            return "\n".join(f"GROUP: {ln.split('.')[0].strip()}"
                             for ln in content.splitlines() if ln[:1].isdigit())
        # Recurrence is recency- AND tenure-weighted (one decimal): three equally-fresh
        # sessions no longer read as the raw 3.0 but as 1.0+0.6+0.36 = 1.96 → "[2.0]".
        assert kw.get("disable_rag") is True and "[2.0]" in content
        if "human-facing self-portrait" in sysp:
            calls["portrait"] += 1
            return "<think>compose</think>\nI am sharp, still tentative, and bounded."
        calls["digest"] += 1
        return sample
    a = run_digest_pass(generate_fn=gen, consolidation_dirs=[cons],
                        persona_dir=persona, run_id="r1")
    b = run_digest_pass(generate_fn=gen, consolidation_dirs=[cons],
                        persona_dir=persona, run_id="r2")
    assert (a["status"] == "written" and b["status"] == "skipped"
            and calls == {"digest": 1, "portrait": 1}), calls
    cur = latest_digest(persona)
    assert cur["dispositions"][0]["maturity"] == "emerging"
    assert cur["self_portrait"]["text"].startswith("I am sharp")
    print("  run_digest_pass: write, self-portrait, material-change skip ✓")

    # Clustered evidence persisted for audit / threshold-tuning / phase-two authority.
    ev_block = cur["evidence"]
    assert ev_block["cluster_threshold"] == CLUSTER_SIM_THRESHOLD
    themes = ev_block["themes"]
    assert len(themes) == ev_block["persona_count"] == 2
    sharp = max(themes, key=lambda t: t["recurrence"])
    assert sharp["recurrence"] == 3 and sharp["members"]   # exact-dup across 3 sessions
    assert all("content" in t and "cluster_size" in t for t in themes)
    # Anchor keys ride the artifact so a later incremental pass can rejoin a persisted
    # theme to the live ledger (content strings alone are not a stable join).
    assert all(t.get("key") for t in themes), themes
    assert all(len(t.get("member_keys") or []) == t["cluster_size"] for t in themes), themes
    assert all(t["key"] in t["member_keys"] for t in themes), themes
    print("  persist: per-theme evidence (recurrence + members + keys) on the artifact ✓")

    # The three seams compose into exactly what the one-call wrapper does — the property
    # the reflection run depends on when it splits them across two models.
    cons3 = tmp / "consolidation3"
    led3 = ConsolidationLedger(cons3)
    for s in ("x1.json", "x2.json"):
        led3.register_fact(content="I'd rather be sharp and wrong than smooth.",
                           item_type="persona", source_session=s)
    persona3 = tmp / "persona3"
    # Own stub: `gen` above pins the prompt body to this fixture's own weight.
    def gen3(content, sysp, **kw):
        if "human-facing self-portrait" in sysp:
            return "<think>compose</think>\nI am sharp, still tentative, and bounded."
        return sample

    plan = plan_digest(cons3, persona_dir=persona3)
    assert plan["should_run"] and plan["persona_count"] == 1 and plan["raw_fp"]
    seam_ev = cluster_for_digest(plan["base"], generate_fn=None)   # no model → embedder tier
    assert len(seam_ev) == 1 and seam_ev[0]["recurrences"] == 2
    out = synthesize_digest(seam_ev, generate_fn=gen3, persona_dir=persona3,
                            run_id="r3", raw_fp=plan["raw_fp"])
    assert out["status"] == "written"
    # ...and the gate still closes on the second look, from the persisted raw fingerprint.
    assert plan_digest(cons3, persona_dir=persona3)["should_run"] is False
    assert plan_digest(cons3, persona_dir=persona3, force=True)["should_run"] is True
    print("  seams: plan → cluster → synthesize compose, gate + force honoured ✓")

    # map_reduce routing: with a generate_fn the blocked path runs; a raising one must
    # fall back to the historical tiers rather than sinking the digest. Needs more than
    # one statement — a single-item block short-circuits without reaching the model.
    multi = [
        {"key": f"k{i}", "content": f"I hold position {i}.", "stage": 0,
         "sessions": {f"s{i}.json"}, "session_weights": {f"s{i}.json": 1.0},
         "counter_sessions": set(), "counter_weights": {}}
        for i in range(4)
    ]
    mr_calls = {"n": 0}

    def group_gen(listing, system, **kw):
        mr_calls["n"] += 1
        return "\n".join(f"GROUP: {i + 1}"
                         for i in range(len(listing.strip().splitlines())))

    assert len(cluster_for_digest(multi, generate_fn=group_gen)) == 4
    assert mr_calls["n"] > 0, "map-reduce path was not taken"

    def boom(*a, **k):
        raise RuntimeError("model exploded")

    assert len(cluster_for_digest(multi, generate_fn=boom)) == 4
    # ...and opting out routes to the single-call path (one grouping call, not blocks).
    flat_calls = {"n": 0}

    def flat_gen(listing, system, **kw):
        flat_calls["n"] += 1
        return "GROUP: 1, 2\nGROUP: 3, 4"

    # polarity=False: this fixture counts grouping calls, and the screen would add one
    # per merged theme (its own tests are below).
    assert len(cluster_for_digest(multi, generate_fn=flat_gen, map_reduce=False,
                                  polarity=False)) == 2
    assert flat_calls["n"] == 1, flat_calls
    print("  cluster_for_digest: map-reduce routed, opt-out + failure use the old tiers ✓")

    # ── polarity screening ─────────────────────────────────────────────────── #
    # _parse_oppose: numbers, explicit none, missing line, range/dup filtering.
    assert _parse_oppose("OPPOSE: 2, 3", 3) == [1, 2]
    assert _parse_oppose("OPPOSE: none", 3) == []
    assert _parse_oppose("no verdict here", 3) is None
    assert _parse_oppose("OPPOSE: 0, 4, 2, 2", 3) == [1]
    assert _parse_oppose("<think>OPPOSE: 1</think>\nOPPOSE: none", 2) == []
    print("  _parse_oppose: numbers, none, missing, range ✓")

    # A theme that swallowed an opposing statement is split: the opposer becomes its
    # own theme, its session becomes a COUNTER plan against the representative instead
    # of an affirmation vote, and the majority theme keeps its remaining members.
    pro1 = {"key": "k1", "content": "I love over-engineering the absurd.",
            "stage": 0, "sessions": {"s1.json"}, "session_weights": {"s1.json": 1.0}}
    pro2 = {"key": "k2", "content": "I enjoy building useless rigorous frameworks.",
            "stage": 0, "sessions": {"s2.json"}, "session_weights": {"s2.json": 1.0}}
    anti = {"key": "k3", "content": "In close moments I switch the analysis off.",
            "stage": 0, "sessions": {"s3.json"}, "session_weights": {"s3.json": 1.0}}
    swallowed = _evidence_entry([pro1, pro2, anti])
    assert swallowed["recurrences"] == 3   # the opposer's session counts FOR the theme

    def _polarity_fn(listing, sys_prompt, **kw):
        num = next(ln.split(".")[0].strip() for ln in listing.splitlines()
                   if "switch the analysis off" in ln)
        return f"OPPOSE: {num}"

    res = screen_theme_polarity([swallowed], [pro1, pro2, anti], _polarity_fn)
    got_keys = sorted(tuple(sorted(e["member_keys"])) for e in res["evidence"])
    assert got_keys == [("k1", "k2"), ("k3",)], res["evidence"]
    opp_theme = next(e for e in res["evidence"] if e["member_keys"] == ["k3"])
    assert opp_theme.get("opposes") == "k2", opp_theme   # pairing survives the split
    assert res["counters"] == [{"key": "k2", "content": pro2["content"],
                                "sessions": ["s3.json"],
                                "opposed": [anti["content"]]}], res["counters"]
    assert res["report"]["themes_split"] == 1
    assert res["report"]["members_flagged"] == 1
    main_theme = next(e for e in res["evidence"] if e["cluster_size"] == 2)
    assert main_theme["recurrences"] == 2   # the opposer's vote no longer counts
    # An unusable verdict (no OPPOSE line) and a failing call both keep the theme.
    kept = screen_theme_polarity([swallowed], [pro1, pro2, anti],
                                 lambda *a, **k: "GROUP: 1")
    assert len(kept["evidence"]) == 1 and not kept["counters"]
    def _pboom(*a, **k):
        raise RuntimeError("screen down")
    kept2 = screen_theme_polarity([swallowed], [pro1, pro2, anti], _pboom)
    assert len(kept2["evidence"]) == 1 and kept2["report"]["calls_failed"] == 1
    print("  polarity screen: opposer split out, counter plan, fail-soft ✓")

    # Through cluster_for_digest: the screen runs after grouping on the same model,
    # the counter plan reaches the sink, and the report lands in stats.
    def _dual_fn(listing, sys_prompt, **kw):
        if "OPPOSE" in sys_prompt:
            return _polarity_fn(listing, sys_prompt, **kw)
        return "GROUP: 1, 2, 3"
    sink_got: list = []
    pstats: dict = {}
    ev_screened = cluster_for_digest([pro1, pro2, anti], generate_fn=_dual_fn,
                                     map_reduce=False, stats=pstats,
                                     counter_sink=sink_got.extend)
    assert len(ev_screened) == 2, ev_screened
    assert sink_got and sink_got[0]["key"] == "k2", sink_got
    assert pstats["polarity"]["themes_split"] == 1, pstats
    # ...and polarity=False restores the historical (swallowing) behaviour.
    ev_plain = cluster_for_digest([pro1, pro2, anti], generate_fn=_dual_fn,
                                  map_reduce=False, polarity=False)
    assert len(ev_plain) == 1
    print("  cluster_for_digest: polarity screen wired, sink + stats, kill-switch ✓")

    # ── tension rendering (the synthesis pass sees the disagreement) ───────── #
    # A split-out opposer renders UNDER the theme it pushes against, never as an
    # unrelated bullet; a contested theme says so on its own line; an opposer whose
    # target is absent falls back to a bullet of its own.
    tension_body = build_digest_prompt_input(res["evidence"])
    assert "— though you have also said [1.0]: In close moments" in tension_body, \
        tension_body
    assert "- [1.0] In close moments" not in tension_body       # not its own bullet
    assert tension_body.count("\n- [") == 1, tension_body       # ONE top-level theme
    orphan = dict(opp_theme, opposes="key-not-present")
    orphan_body = build_digest_prompt_input([orphan])
    assert "- [1.0] In close moments" in orphan_body            # fallback bullet
    # (the preamble explains the marker, so test the RENDERED form specifically)
    assert "— though you have also said [" not in orphan_body
    contested_e = _evidence_entry([{**pro1, "counter_sessions": {"cx.json"},
                                    "counter_weights": {"cx.json": 1.0}}])
    c_body = build_digest_prompt_input([contested_e])
    assert "contested: pushed against or contradicted, weight 1.0" in c_body, c_body
    sp_body = build_self_portrait_prompt_input({"voice": "v"}, res["evidence"])
    assert "— though you have also said" in sp_body              # portrait pass too
    # ...and the artifact keeps the pairing + contested weight for audit.
    summ = _evidence_summary(res["evidence"] + [contested_e], 0.6)
    s_opp = next(t for t in summ["themes"] if t["member_keys"] == ["k3"])
    assert s_opp.get("opposes") == "k2", s_opp
    assert all("counter_recurrence" in t for t in summ["themes"])
    assert any(t["counter_recurrence"] == 1.0 for t in summ["themes"])
    print("  tension: opposer attached under target, contested noted, persisted ✓")

    # ── disposition-gate escalation screen ─────────────────────────────────── #
    assert _parse_escalates("ESCALATES: 1", 2) == [0]
    assert _parse_escalates("ESCALATES: none", 2) == []
    assert _parse_escalates("no verdict here", 2) is None
    gd = {"dispositions": [
        {"name": "escalation", "when": "w", "do": "push everything further",
         "not": "simply apologize or return to boring accuracy",
         "maturity": "established"},
        {"name": "register-mirroring", "when": "w", "do": "match their register",
         "not": "never with a stranger", "maturity": "established"},
    ]}
    def _gate_fn(listing, sysp, **kw):
        assert "ESCALATES" in sysp and "1. do:" in listing and "not:" in listing
        return "ESCALATES: 1"
    grep_rep = screen_disposition_gates(gd, _gate_fn)
    assert grep_rep == {"checked": 2, "flagged": 1, "names": ["escalation"]}, grep_rep
    assert gd["dispositions"][0]["not"] == ""
    assert gd["dispositions"][0]["not_rejected"].startswith("simply apologize")
    assert gd["dispositions"][1]["not"] == "never with a stranger"   # restraint kept
    gd2 = {"dispositions": [{"name": "x", "when": "", "do": "d", "not": "n",
                             "maturity": ""}]}
    assert screen_disposition_gates(gd2, lambda *a, **k: "GROUP: 1")["flagged"] == 0
    assert gd2["dispositions"][0]["not"] == "n"        # unusable verdict clears nothing
    assert screen_disposition_gates(gd2, _pboom)["flagged"] == 0    # errored call too
    assert screen_disposition_gates({"dispositions": []}, _pboom) == \
        {"checked": 0, "flagged": 0}                                # nothing to check
    print("  gate screen: escalating not: cleared + audited, restraint kept, fail-soft ✓")

    # Through synthesize_digest: the flagged gate is cleared BEFORE the digest is
    # written, the cleared text survives as not_rejected on the artifact, and the
    # kill-switch restores the historical behaviour.
    persona4 = tmp / "persona4"
    def gen4(content, sysp, **kw):
        if "ESCALATES" in sysp:
            return "ESCALATES: 1"
        if "human-facing self-portrait" in sysp:
            return "portrait."
        return sample
    out4 = synthesize_digest(seam_ev, generate_fn=gen4, persona_dir=persona4,
                             run_id="r4", raw_fp="fp4")
    assert out4["counts"]["gates_cleared"] == 1, out4
    assert (out4.get("gate_screen") or {}).get("names") == ["transgressive attack"]
    cur4 = latest_digest(persona4)
    assert cur4["dispositions"][0]["not"] == "" and cur4["dispositions"][0]["not_rejected"]
    assert cur4["dispositions"][1]["not"]                       # second gate survives
    persona5 = tmp / "persona5"
    out5 = synthesize_digest(seam_ev, generate_fn=gen4, persona_dir=persona5,
                             run_id="r5", raw_fp="fp5", gate_screen=False)
    assert out5["counts"]["gates_cleared"] == 0
    assert latest_digest(persona5)["dispositions"][0]["not"]    # untouched when off
    print("  synthesize: gate screen wired before write, kill-switch honoured ✓")

    # Phase-two embedding channel: score candidate texts vs the digest's anchor_texts.
    dgst = {"version": "v", "anchor_texts": ["I am blunt and sharp"]}
    s = score_texts_against_digest(
        dgst, ["a blunt sharp reply", "an attack on bigotry", "neutral text"],
        embedder=_Emb())
    assert s and s["pick_index"] == 0 and s["scores"][0] > s["scores"][1], s
    assert score_texts_against_digest({"anchor_texts": []}, ["x"], embedder=_Emb()) is None
    assert score_texts_against_digest(dgst, [], embedder=_Emb()) is None
    print("  score: embedding channel picks the most voice-aligned option ✓")

    # Judge-channel rendering: established dispositions first, gates + lines laid out.
    jd = {
        "voice": "Blunt and dry.",
        "dispositions": [
            {"name": "theoretical escalation", "when": "play", "do": "push it far",
             "not": "leave it a joke", "maturity": "emerging"},
            {"name": "transgressive attack", "when": "bad-faith bigotry",
             "do": "hit back in register", "not": "adopt the bigotry", "maturity": "established"},
        ],
        "lines": ["I won't be a flat mirror."],
    }
    block = render_digest_for_judge(jd)
    assert "transgressive attack [established]" in block
    assert block.index("transgressive attack") < block.index("theoretical escalation")  # established first
    assert "WHAT I REFUSE TO BECOME" in block and "flat mirror" in block
    assert render_digest_for_judge({}) == ""
    print("  judge: digest rendered (established-first, gates + lines) ✓")

    intro = render_digest_for_introduction(jd)
    assert "who I have become" in intro
    assert "transgressive attack" in intro and "[established]" not in intro  # no maturity labels
    assert intro.index("transgressive attack") < intro.index("theoretical escalation")  # established first
    assert "What I won't become" in intro and "flat mirror" in intro
    assert render_digest_for_introduction({}) == ""
    print("  intro: first-person introduction rendered (no labels, established-first) ✓")

    # Chat-channel rendering: the standing portrait that REPLACES per-turn [persona]
    # recall, so it is deliberately narrower than every other renderer.
    chat = render_digest_for_chat({**jd, "stances": ["Utility is a mask."]})
    assert "transgressive attack" in chat                  # established rides
    assert "theoretical escalation" not in chat            # emerging does NOT
    assert "Utility is a mask" not in chat                 # stances withheld from chat
    assert "[established]" not in chat                     # no audit labels in her own voice
    assert "Blunt and dry." in chat and "flat mirror" in chat
    assert "not a brief to perform" in chat                # the do-not-recite guard
    # Empty return is the caller's signal to KEEP the RAG persona channel — so a digest
    # with nothing mature must not render a portrait on voice alone... unless voice is
    # genuinely there (it is synthesized from the whole evidence set, not one theme).
    emerging_only = {"dispositions": [{"name": "x", "when": "w", "do": "d",
                                       "maturity": "emerging"}]}
    assert render_digest_for_chat(emerging_only) == ""
    assert render_digest_for_chat({}) == ""
    assert "How I tend to speak" in render_digest_for_chat({"voice": "Blunt."})
    print("  chat: standing portrait (established-only, no stances, degrade signal) ✓")

    # ── persona recency decay ──────────────────────────────────────────────── #
    # Weight curve: full ≤30d, 0 ≥180d, linear between.
    assert _persona_recency_weight(0) == 1.0 and _persona_recency_weight(30) == 1.0
    assert _persona_recency_weight(180) == 0.0 and _persona_recency_weight(300) == 0.0
    assert abs(_persona_recency_weight(105) - 0.5) < 1e-6   # midpoint of the ramp

    # Summed-per-theme with the TENURE discount: a reinforced theme (3 sessions incl. a
    # recent one) still SURVIVES, but the two decays squeeze it — its emergence session is
    # the oldest (most recency-decayed) while its fresh echoes are the highest-rank (most
    # tenure-discounted). Stems are timestamps, so sort order is chronological: 2025-01
    # (emergence, rank 0, w 0.1) · 2025-06 (rank 1, w 0.5) · 2026-01 (rank 2, w 1.0) →
    # 0.1·1 + 0.5·0.6 + 1.0·0.36 = 0.76.
    reinforced = [
        {"key": "r1", "content": "I open up with people I trust.", "stage": 0,
         "sessions": {"20260101", "20250601", "20250101"},
         "session_weights": {"20260101": 1.0, "20250601": 0.5, "20250101": 0.1}},
    ]
    stale = [
        {"key": "s1", "content": "I stay guarded.", "stage": 0,
         "sessions": {"20240101"}, "session_weights": {"20240101": 0.1}},
    ]
    er = _evidence_entry(reinforced)
    es = _evidence_entry(stale)
    assert er["recurrences"] == 3 and abs(er["weighted_recurrence"] - 0.76) < 1e-6
    assert es["recurrences"] == 1 and abs(es["weighted_recurrence"] - 0.1) < 1e-6
    # The reinforced theme leads (weighted 0.76 > 0.1) despite equal raw stages.
    ordered = cluster_persona_evidence(reinforced + stale,
                                       generate_fn=None, embedder=None)
    assert ordered[0]["key"] == "r1", ordered
    # Prompt input keeps the reinforced theme (0.76 ≥ floor) and drops the faded one (0.1).
    body = build_digest_prompt_input(ordered)
    assert "I open up" in body and "I stay guarded" not in body, body
    assert "[0.8]" in body, body
    # Anti-ratchet: an echo-sustained theme (many equally-FRESH sessions — the circular
    # self-vote) can no longer climb without bound. Its weighted recurrence converges to
    # 1/(1-_TENURE_DECAY) and stays strictly below it (and far below its raw count of 12), so
    # the persona cannot ossify just by re-affirming itself every session.
    ceiling = 1.0 / (1.0 - _TENURE_DECAY)
    fresh_sessions = {f"2026{i:04d}": 1.0 for i in range(1, 13)}   # 12 equally-fresh echoes
    echoed = _evidence_entry([{"key": "e", "content": "echo", "stage": 0,
                               "sessions": set(fresh_sessions),
                               "session_weights": fresh_sessions}])
    assert echoed["recurrences"] == 12                            # raw count stays honest
    assert echoed["weighted_recurrence"] < ceiling                # bounded (linear would be 12)
    assert ceiling - echoed["weighted_recurrence"] < 0.01         # already near the 2.5 ceiling

    # Counter-evidence (the persuasion channel): sustained pushback nets out of weighted
    # recurrence at _PERSUASION_GAIN, so a strong trait RESISTS one push but fades under a
    # sustained line of them — an integrator with a threshold, never a single hard turn.
    affirm = {"key": "p", "content": "I plan exhaustively.", "stage": 0,
              "sessions": {"20260101", "20260201", "20260301"},
              "session_weights": {"20260101": 1.0, "20260201": 1.0, "20260301": 1.0}}
    base_weighted = _evidence_entry([dict(affirm)])["weighted_recurrence"]
    assert abs(base_weighted - 1.96) < 1e-6                        # 1.0+0.6+0.36 (tenure), uncontested
    one_push = _evidence_entry([{**affirm,
        "counter_sessions": {"20260401"}, "counter_weights": {"20260401": 1.0}}])
    assert abs(one_push["weighted_recurrence"] - (1.96 - 0.5)) < 1e-6   # 1.46 — barely dented
    assert one_push["weighted_recurrence"] > _PROMPT_WEIGHT_FLOOR       # still in the portrait
    assert one_push["counter_recurrence"] == 1.0
    sustained = {f"2026{i:04d}": 1.0 for i in range(5, 9)}         # 4 fresh, independent pushes
    pushed = _evidence_entry([{**affirm,
        "counter_sessions": set(sustained), "counter_weights": sustained}])
    assert pushed["weighted_recurrence"] == 0.0                    # netted to/below zero
    assert pushed["weighted_recurrence"] < _PROMPT_WEIGHT_FLOOR    # fades from the portrait
    # Counter presence moves the regen fingerprint (accumulating pushback triggers one regen).
    assert (raw_fingerprint([dict(affirm)])
            != raw_fingerprint([{**affirm, "counter_sessions": {"20260401"}}]))
    print("  counter-evidence: one push resists, sustained pushback fades a trait ✓")
    # Fingerprint moves when the freshest evidence crosses a recency band, not otherwise.
    assert _recency_band(1.0) != _recency_band(0.1)
    fresh = [{"key": "k", "content": "c", "stage": 0, "sessions": {"a"},
              "session_weights": {"a": 1.0}, "recency_band": _recency_band(1.0)}]
    faded = [{"key": "k", "content": "c", "stage": 0, "sessions": {"a"},
              "session_weights": {"a": 0.1}, "recency_band": _recency_band(0.1)}]
    assert raw_fingerprint(fresh) != raw_fingerprint(faded)
    print("  decay: weight curve, summed-per-theme survival, faded drop, fingerprint band ✓")

    print("persona-digest self-test passed ✓")


if __name__ == "__main__":
    import sys
    # Make the sibling `training` package importable when run as a module from
    # server/inference (mirrors how the other GPU-free self-tests are invoked).
    _server = Path(__file__).resolve().parent.parent.parent
    if str(_server) not in sys.path:
        sys.path.insert(0, str(_server))
    _selftest()
