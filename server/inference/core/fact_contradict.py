"""Contradiction resolution among live ``[fact]`` items — supersede a stale fact when a
newer one corrects it.

``fact_dedup`` merges *paraphrases of the same truth*; this resolves *opposite claims
about the same subject* — the "the user corrected me in chat, but I just wrote a new
fact contradicting the old one instead of replacing it" bug, which leaves both live and
recallable. Neither ``fact_dedup`` (distinct facts stay separate) nor ``self_reconcile``
(judges against the persona digest, not fact-vs-fact) catches it.

Two consumers share this one detection core:
  * **A — cleanup pass** (``server.handle_resolve_contradictions``): cluster ALL live
    facts by subject and resolve contradictions within each cluster. Runs on the clean
    base (a standalone evaluation).
  * **B — supersede-at-correction** (reflection consolidation loop): check each newly
    written fact against its live subject-neighbours and supersede any it contradicts.
    Runs on the loaded adapter (a per-session clean-base swap would be prohibitive).

Policy split, mirroring ``fact_dedup``: the **LLM** decides only *which facts contradict
each other* (semantic); **recency is mechanical** — within a conflict set the NEWEST
(by ``ts``) wins and the rest are superseded. So the model never has to reason about
dates, and "the correction is the later fact" is enforced by code, not trusted to the
model. Superseding is a *soften* (``supersede`` op, kept as evidence-of-change), so it
is reversible.

Split pure-logic / one-LLM-call for GPU-free testing:
  * ``cluster_by_subject`` — embedding greedy-cluster of facts by recall cue (pure given
    an ``embed_fn``).
  * ``find_conflicts`` — the one LLM call over a group → sets of mutually-contradictory
    facts.
  * ``resolve_group`` / ``plan_supersessions`` — pure recency policy over the conflicts.

GPU-free self-test: ``python -m core.fact_contradict``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from core.reflection_writer import normalize_person


# Cosine floor for two facts to be considered the same *subject* (so they are compared
# for contradiction). Uses the recall cue (``trigger``) — "what brings this back" — since
# two claims about one subject share a cue even when their content differs. Deliberately
# loose: a missed cluster just means a missed contradiction (safe), an over-broad cluster
# only widens what the LLM then rules non-contradictory.
SUBJECT_SIM_THRESHOLD = 0.55


def _subject_text(fact: dict) -> str:
    """What a fact is *about*, for subject clustering: its recall cue, else its content."""
    return (fact.get("trigger") or fact.get("content") or "").strip()


def _dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def cluster_by_subject(facts: list[dict], embed_fn: Optional[Callable],
                       threshold: float = SUBJECT_SIM_THRESHOLD) -> list[list[dict]]:
    """Greedy average-link clustering of facts by subject (recall cue) embedding,
    **partitioned by the person each fact is about**.

    *embed_fn(texts) -> list[vector]* returns L2-normalised vectors (RagEngine's
    multilingual embedder). ``None`` / any failure ⇒ every fact is its own cluster (no
    contradictions can be found — a safe no-op). Deterministic key order, so the greedy
    result is reproducible. Singletons are returned too; the caller ignores clusters of
    one (nothing to contradict).

    **Person partition:** two facts can only be compared for contradiction when they
    concern the SAME person. Clustering is by recall cue, and two people's cues are
    near-identical for the same topic ("Boris's dog" / "Artemy's dog"), so an unscoped
    cluster mixed subjects — and then mechanical newest-wins superseded one person's
    true fact with another person's unrelated one. Facts whose subject is unknown
    (world facts, Ava's own reading, records written before attribution existed) share
    one ``""`` partition and cluster among themselves exactly as they did before, so
    this narrows what may be compared and never widens it.
    """
    if not facts:
        return []
    if embed_fn is None:
        return [[f] for f in facts]
    partitions: dict[str, list[dict]] = {}
    for f in facts:
        partitions.setdefault(normalize_person(f.get("about")), []).append(f)
    if len(partitions) > 1:
        out: list[list[dict]] = []
        for subject in sorted(partitions):
            out.extend(cluster_by_subject(partitions[subject], embed_fn, threshold))
        return out

    ordered = sorted(facts, key=lambda f: (f.get("key") or _subject_text(f)))
    try:
        vecs = [list(map(float, v)) for v in embed_fn([_subject_text(f) for f in ordered])]
    except Exception:
        return [[f] for f in ordered]
    if len(vecs) != len(ordered):
        return [[f] for f in ordered]

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
    return [[ordered[j] for j in c] for c in clusters]


def load_contradict_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The contradiction-detection prompt (``prompts/fact_contradict_prompt.txt``), or a
    safe inline fallback."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "fact_contradict_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "Below are numbered facts you have recorded about a related subject. Some may "
        "directly CONTRADICT each other — state something that cannot both be true (for "
        "example, one says a person is a beginner and another says they are an expert; "
        "one says yes and another says no).\n\n"
        "Find the sets of facts that contradict one another. Two facts that are merely "
        "about the same topic, or that add detail, do NOT contradict — only mark facts "
        "that cannot both be true. A fact with no contradiction is not listed.\n\n"
        "Output only lines of this form, nothing else (one line per contradiction set, "
        "listing the numbers that conflict); if nothing contradicts, output NONE:\n"
        "CONFLICT: <comma-separated numbers>"
    )


def _listing(facts: list[dict]) -> str:
    """Number the facts for the detection prompt, annotating each with its recall cue."""
    lines = []
    for i, f in enumerate(facts, 1):
        content = (f.get("content") or "").strip()
        trigger = (f.get("trigger") or "").strip()
        if trigger:
            lines.append(f"{i}. {content}   (about: {trigger})")
        else:
            lines.append(f"{i}. {content}")
    return "\n".join(lines)


_CONFLICT_RE = re.compile(r"conflict\s*[:\-]?\s*(.+)$", re.IGNORECASE)


def _parse_conflicts(text: str, n: int) -> list[list[int]]:
    """Parse ``CONFLICT: a, b, c`` lines into 0-based index sets (each size >= 2).

    Lenient: ignores non-matching lines, dedups indices, drops out-of-range numbers and
    any set that ends up with fewer than two members. ``NONE`` / no matches ⇒ ``[]``."""
    out: list[list[int]] = []
    for line in (text or "").splitlines():
        m = _CONFLICT_RE.search(line.strip())
        if not m:
            continue
        nums: list[int] = []
        for tok in re.split(r"[,\s]+", m.group(1).strip()):
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= n and (v - 1) not in nums:
                    nums.append(v - 1)
        if len(nums) >= 2:
            out.append(nums)
    return out


def find_conflicts(facts: list[dict], generate_fn: Callable) -> Optional[list[list[int]]]:
    """One greedy generation → sets of mutually-contradictory facts (indices into *facts*).

    Returns ``[]`` when nothing contradicts, the conflict sets when some do, or ``None``
    only on a generation failure (so the caller can distinguish "judged, nothing" from
    "couldn't judge"). Greedy, thinking off — an evaluation, not expression."""
    if len(facts) < 2:
        return []
    try:
        resp = generate_fn(
            _listing(facts), load_contradict_prompt(),
            temperature=0.0, top_p=1.0,
            max_new_tokens_setting="512",
            before_session="", disable_rag=True, disable_thinking=True)
    except Exception:
        return None
    return _parse_conflicts(resp, len(facts))


def _ts(fact: dict) -> str:
    return (fact.get("ts") or "")


def resolve_group(group: list[dict], generate_fn: Callable,
                  *, new_keys: Optional[set] = None) -> list[dict]:
    """Detect contradictions within *group* and apply the recency policy.

    Returns a supersede plan (one entry per stale loser): the NEWEST fact in each
    conflict set is kept (the correction), the older ones superseded, pointing back at
    the survivor. ``surface_count`` breaks a timestamp tie (the more-recalled phrasing
    is likelier the one that stuck). A fact appearing in several conflict sets is
    superseded once (first loss wins).

    *new_keys* (part B, supersede-at-correction): when given, a conflict set is only
    acted on if its survivor (the newest) is one of these keys — i.e. the correction was
    a fact written *this run*. This scopes an automatic post-reflection sweep to genuine
    corrections and never touches a pre-existing old-vs-old conflict (that is part A's
    manual job). ``None`` (part A) acts on every conflict."""
    conflicts = find_conflicts(group, generate_fn)
    if not conflicts:
        return []
    plan: list[dict] = []
    seen_losers: set = set()
    for cset in conflicts:
        members = [group[i] for i in cset]
        survivor = max(members, key=lambda f: (_ts(f), int(f.get("surface_count", 0) or 0)))
        skey = (survivor.get("key") or "").strip()
        if new_keys is not None and skey not in new_keys:
            continue   # the newest here isn't a this-run correction — leave it for A
        for f in members:
            key = (f.get("key") or "").strip()
            if not key or key == skey or key in seen_losers:
                continue
            seen_losers.add(key)
            plan.append({
                "key": key,
                "kind": "fact",
                "content": (f.get("content") or "").strip(),
                "trigger": (f.get("trigger") or "").strip() or None,
                "superseded_by": skey,
                "survivor_content": (survivor.get("content") or "").strip(),
                "reason": "contradicted by a newer fact: "
                          + (survivor.get("content") or "").strip(),
            })
    return plan


def plan_supersessions(facts: list[dict], embed_fn: Optional[Callable],
                       generate_fn: Callable, *,
                       on_group: Optional[Callable] = None,
                       threshold: float = SUBJECT_SIM_THRESHOLD,
                       new_keys: Optional[set] = None) -> list[dict]:
    """Full cleanup plan over *facts*: cluster by subject, resolve each multi-fact cluster.

    *on_group(i, n_multi, group_size, n_superseded)* is an optional progress hook fired
    once per resolved (multi-fact) cluster (server-side logging / streaming). Singleton
    clusters are skipped silently. The returned plan feeds ``ReflectionWriter.write_supersede``
    + ``ConsolidationLedger.supersede``.

    *new_keys* (part B): when given, only clusters that CONTAIN a new-this-run fact are
    judged (bounding the automatic post-reflection sweep to subjects that changed), and
    only conflicts whose correction is a new fact are acted on (see ``resolve_group``)."""
    clusters = [c for c in cluster_by_subject(facts, embed_fn, threshold) if len(c) >= 2]
    if new_keys is not None:
        clusters = [c for c in clusters
                    if any((f.get("key") or "") in new_keys for f in c)]
    plan: list[dict] = []
    for i, group in enumerate(clusters, 1):
        entries = resolve_group(group, generate_fn, new_keys=new_keys)
        plan.extend(entries)
        if on_group is not None:
            try:
                on_group(i, len(clusters), len(group), len(entries))
            except Exception:
                pass
    return plan


def summarize(plan: list[dict]) -> dict:
    """Client-facing view of a plan: each superseded fact, why, and the correction."""
    return {
        "superseded": len(plan),
        "items": [{"content": p["content"], "trigger": p.get("trigger"),
                   "survivor": p.get("survivor_content", ""), "key": p["key"]}
                  for p in plan],
    }


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    facts = [
        {"key": "a", "content": "Artemy is a beginner who needs everything spelled out.",
         "trigger": "the user's skill level", "ts": "2026-01-01", "surface_count": 1},
        {"key": "b", "content": "Artemy is an experienced engineer.",
         "trigger": "the user's experience", "ts": "2026-03-01", "surface_count": 0},
        {"key": "c", "content": "Artemy prefers dark roast coffee.",
         "trigger": "coffee preference", "ts": "2026-02-01", "surface_count": 5},
    ]

    # Fake embedder: a=b subject (skill), c apart. 3-dim normalized-ish vectors.
    def fake_embed(texts):
        out = []
        for t in texts:
            if "skill" in t or "experience" in t:
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out

    clusters = cluster_by_subject(facts, fake_embed)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2], f"expected a pair + singleton, got {sizes}"

    # Fake judge: the two skill facts (whatever their local numbers) conflict.
    def fake_generate(listing, system, **kw):
        assert "beginner" in listing or "experienced" in listing or "coffee" in listing
        if "beginner" in listing and "experienced" in listing:
            return "CONFLICT: 1, 2"
        return "NONE"

    plan = plan_supersessions(facts, fake_embed, fake_generate)
    assert len(plan) == 1, f"expected one supersession, got {plan}"
    p = plan[0]
    # Newest (b, 2026-03) wins; older beginner fact (a) is superseded.
    assert p["key"] == "a", f"stale loser should be 'a', got {p['key']}"
    assert "experienced" in p["survivor_content"], p
    assert p["superseded_by"] == "b", p

    # Parser edge cases.
    assert _parse_conflicts("NONE", 3) == []
    assert _parse_conflicts("CONFLICT: 1, 4, 2", 3) == [[0, 1]]   # 4 out of range dropped
    assert _parse_conflicts("CONFLICT: 2", 3) == []               # singleton dropped

    # No embedder → no clusters ≥ 2 → empty plan (safe no-op).
    assert plan_supersessions(facts, None, fake_generate) == []
    # find_conflicts distinguishes "nothing" ([]) from "couldn't judge" (None).
    assert find_conflicts(facts, lambda *a, **k: "NONE") == []
    assert find_conflicts(facts, lambda *a, **k: (_ for _ in ()).throw(RuntimeError())) is None

    rep = summarize(plan)
    assert rep["superseded"] == 1 and rep["items"][0]["survivor"].startswith("Artemy is an")

    # Part B scoping (new_keys): the newer fact 'b' is the correction → acts.
    assert len(plan_supersessions(facts, fake_embed, fake_generate, new_keys={"b"})) == 1

    # ── person partition ─────────────────────────────────────────────────────
    # Two people, ONE identical recall cue. Without the `about` partition these
    # collapse into a single cluster and newest-wins lets Boris's fact supersede
    # Artemy's true one. Verifies same-person facts still cluster (a real
    # self-correction is still caught) and unattributed facts stay in their own pool.
    people = [
        {"key": "a", "content": "Artemy's dog is a labrador", "trigger": "the dog",
         "about": "Artemy", "ts": "2026-01-01"},
        {"key": "b", "content": "Boris's dog is a poodle", "trigger": "the dog",
         "about": "Boris", "ts": "2026-03-01"},
        {"key": "c", "content": "Artemy's dog is a retriever", "trigger": "the dog",
         "about": "artemy voikhansky", "ts": "2026-05-01"},   # full name = same person
        {"key": "d", "content": "Paris is the capital of France", "trigger": "France",
         "ts": "2026-02-01"},                                  # no subject → observed
    ]

    def one_cue(texts):
        return [[1.0, 0.0] if "dog" in t else [0.0, 1.0] for t in texts]

    got = {frozenset(f["key"] for f in c) for c in cluster_by_subject(people, one_cue)}
    assert got == {frozenset("ac"), frozenset("b"), frozenset("d")}, got

    # And the cross-person supersession that motivated the partition cannot happen:
    # only Artemy's own pair is ever offered to the judge.
    def conflict_all(listing, system, **kw):
        return "CONFLICT: 1, 2"

    cross = plan_supersessions(people, one_cue, conflict_all)
    assert {p["key"] for p in cross} == {"a"}, cross
    assert cross[0]["superseded_by"] == "c", cross
    # If the correction key isn't in new_keys, the conflict is left for part A (no-op),
    # even though the facts still contradict.
    assert plan_supersessions(facts, fake_embed, fake_generate, new_keys={"a"}) == []
    # A cluster with no new-this-run fact is never even judged.
    assert plan_supersessions(facts, fake_embed, fake_generate, new_keys=set()) == []

    print("fact_contradict self-test OK")


if __name__ == "__main__":
    _selftest()
