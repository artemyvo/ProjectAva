"""Map-reduce clustering of persona evidence into themes, judged by the model.

Why this exists (see ``reflection_digest.llm_cluster_evidence``, the path it replaces):
the digest's LLM grouping pass puts **every** live persona statement into ONE prompt and
asks for one flat partition. That does not survive scale. At ~640 live statements the
listing is ~40k tokens against a 24,576-token context, and a correct answer would need
640+ group numbers out of a 1024-token budget. What actually comes back is a truncated
grouping, which ``_parse_groups`` then completes by making every unmentioned statement a
singleton — so the digest silently reads as "one blob theme plus hundreds of one-offs".
That failure is visible in the shipped digest: a 130-member mega-cluster whose members
share no disposition, alongside 78 singletons, several of which are plain restatements
of the blob's own representative.

The fix is structural, not a better prompt:

  * **Map** — split the deterministically-ordered statements into fixed blocks of
    ``block_size`` and group *within* each block. Prompt size and output length are then
    constants, independent of corpus size, so the call the model is asked to answer is
    always one it can actually answer.
  * **Reduce** — a theme split across two blocks must still be able to merge, so each
    round re-groups the current themes' *representatives* (one statement per theme) the
    same way. Rounds repeat until nothing merges or ``max_reduce_rounds`` is spent.
  * **Rotation** — chunking representatives the same way each round would freeze the same
    neighbours together, so a cross-chunk pair could never meet. Each reduce round rotates
    the ordered list by half a block first, giving a different neighbourhood per round.
    Cross-block merging is therefore best-effort within the round budget — bounded work,
    and stated plainly rather than hidden.

**Blob guard.** The pathology this module exists to prevent is a block whose grouping
collapses into one giant group. A block whose largest group exceeds
``MAX_GROUP_FRACTION`` of the block is rejected *whole* and its statements fall back to
singletons. Rejecting is the conservative direction: a missed merge leaves two themes
that a later round or run can still join, while a false merge silently fuses unrelated
dispositions into one theme whose recurrence then dominates the digest's ranking.

The grouping is an **evaluation** ("do these two say the same thing?"), not Ava's
expression, so the caller runs it on the clean base (adapter off) — mirroring
``fact_dedup`` / ``self_reconcile``. This module never touches the GPU itself; it only
calls the ``generate_fn`` it is handed.

Output is drop-in compatible with ``reflection_digest.cluster_persona_evidence``: the
same ``_evidence_entry`` collapse and the same recency-weighted ordering, so recurrence,
maturity, and the digest prompt body downstream are untouched.

Not implemented here: **incremental** assignment (judge only statements new since the
last digest, against existing theme representatives). That needs member *keys* persisted
on the digest artifact — ``evidence.themes[].members`` currently stores content strings
only — and is the natural follow-up once this path is wired for real.

GPU-free self-test: ``python -m core.persona_cluster``.
"""

from __future__ import annotations

from typing import Callable, Optional

from core.reflection_digest import (      # the grouping primitives this reuses verbatim
    _evidence_entry,
    _parse_groups,
    _weighted_recurrence,
    load_cluster_prompt,
)

# Statements per grouping call. Small enough that the prompt and the expected output are
# both comfortable, large enough that most paraphrases of one theme land together.
DEFAULT_BLOCK_SIZE = 40
# How many representative-regrouping rounds a reduce phase may spend before giving up on
# further cross-block merges. A round costs ceil(themes / block_size) calls and the theme
# count falls fast, so the tail rounds are cheap; the loop exits early the moment a round
# merges nothing, and this is only the safety bound.
MAX_REDUCE_ROUNDS = 5
# A block whose biggest group covers more than this fraction of the block is rejected
# whole (see the blob guard above).
MAX_GROUP_FRACTION = 0.5
# Guard floor: in a tiny block "half the block" is unremarkable, so never reject below
# this many members.
MIN_BLOB_MEMBERS = 6


def _representative(members: list[dict]) -> dict:
    """The statement that stands for a theme — the same deterministic rule
    ``_evidence_entry`` uses to pick its representative (most distinct sessions, then
    longest content, then key), so the theme's stand-in here and the representative the
    digest ultimately publishes are the same statement."""
    return max(members, key=lambda m: (len(m.get("sessions") or ()),
                                       len(m.get("content") or ""),
                                       m.get("key") or ""))


def _listing(items: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {(it.get('content') or '').strip()}"
                     for i, it in enumerate(items))


def _blocks(items: list, size: int) -> list[list]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _is_blob(groups: list[list[int]], n: int) -> bool:
    """True when a grouping collapsed into one oversized group (see the blob guard)."""
    if not groups:
        return True
    biggest = max(len(g) for g in groups)
    return biggest >= max(MIN_BLOB_MEMBERS, int(n * MAX_GROUP_FRACTION) + 1)


def group_block(items: list[dict], generate_fn: Callable,
                *, prompt: Optional[str] = None,
                listing_fn: Optional[Callable] = None) -> Optional[list[list[dict]]]:
    """One grouping call over ONE block → member lists, or ``None`` when the block
    produced nothing usable (generation failed, nothing parsed, or the blob guard
    rejected it). ``None`` means "treat these as singletons" — the caller decides, so
    this stays pure enough to unit-test.

    Greedy (``temperature=0``, thinking off): a partition is an evaluation, and a
    reproducible one is worth more than a creative one.

    *listing_fn* renders the numbered block for the prompt, defaulting to bare content.
    It exists so a non-persona caller can annotate its items without forking this loop:
    ``fact_dedup`` appends each fact's recall cue, because two facts sharing a truth but
    not a trigger are not the same live record and must not merge.
    """
    if not items:
        return []
    if len(items) == 1:
        return [[items[0]]]
    system = prompt if prompt is not None else load_cluster_prompt()
    render = listing_fn if listing_fn is not None else _listing
    # ~24 tokens per emitted GROUP line plus headroom, bounded — enough for the
    # worst case (every statement its own group) without an open-ended budget.
    max_new = str(min(4000, len(items) * 24 + 256))
    try:
        resp = generate_fn(
            render(items), system,
            temperature=0.0, top_p=1.0,
            max_new_tokens_setting=max_new,
            before_session="", disable_rag=True, disable_thinking=True)
    except Exception:
        return None
    groups = _parse_groups(resp, len(items))
    if not groups or _is_blob(groups, len(items)):
        return None
    return [[items[i] for i in g] for g in groups]


def _map_phase(base: list[dict], generate_fn: Callable, *, block_size: int,
               prompt: Optional[str], on_stage: Optional[Callable],
               stats: dict, listing_fn: Optional[Callable] = None,
               order_key: Optional[Callable] = None
               ) -> list[list[dict]]:
    """Group within fixed blocks. Returns proto-themes (each a list of base items)."""
    ordered = sorted(base, key=order_key or (lambda it: it.get("key") or ""))
    blocks = _blocks(ordered, block_size)
    themes: list[list[dict]] = []
    for i, block in enumerate(blocks, 1):
        groups = group_block(block, generate_fn, prompt=prompt,
                             listing_fn=listing_fn)
        stats["calls"] += 1
        rejected = groups is None
        if rejected:
            stats["rejected_blocks"] += 1
            groups = [[it] for it in block]      # conservative: no merges from this block
        themes.extend(groups)
        _emit(on_stage, {"stage": "map", "i": i, "n": len(blocks),
                         "items": len(block), "themes": len(groups),
                         "rejected": rejected, "running_themes": len(themes)})
    return themes


def _reduce_round(themes: list[list[dict]], generate_fn: Callable, *, block_size: int,
                  rotation: int, prompt: Optional[str],
                  stats: dict, listing_fn: Optional[Callable] = None,
                  order_key: Optional[Callable] = None
                  ) -> list[list[dict]]:
    """One pass of regrouping theme REPRESENTATIVES, merging the themes that group.

    *rotation* offsets the ordered representative list before chunking so a later round
    pairs different neighbours than an earlier one.
    """
    _key = order_key or (lambda it: it.get("key") or "")
    reps = [(_representative(t), t) for t in themes]
    reps.sort(key=lambda rt: _key(rt[0]))
    if rotation and len(reps) > 1:
        r = rotation % len(reps)
        reps = reps[r:] + reps[:r]

    merged: list[list[dict]] = []
    for chunk in _blocks(reps, block_size):
        rep_items = [rep for rep, _theme in chunk]
        groups = group_block(rep_items, generate_fn, prompt=prompt,
                             listing_fn=listing_fn)
        stats["calls"] += 1
        if groups is None:
            stats["rejected_blocks"] += 1
            merged.extend(theme for _rep, theme in chunk)
            continue
        # Map each returned representative back to the theme it stands for. Identity is
        # by key: representatives are distinct base items, so this is exact.
        by_key = {(rep.get("key") or ""): theme for rep, theme in chunk}
        for g in groups:
            fused: list[dict] = []
            for rep in g:
                fused.extend(by_key.get(rep.get("key") or "", []))
            if fused:
                merged.append(fused)
    return merged


def _emit(on_stage: Optional[Callable], info: dict) -> None:
    if on_stage is None:
        return
    try:
        on_stage(info)
    except Exception:
        pass


def map_reduce_groups(base: list[dict], generate_fn: Callable, *,
                      block_size: int = DEFAULT_BLOCK_SIZE,
                      max_reduce_rounds: int = MAX_REDUCE_ROUNDS,
                      prompt: Optional[str] = None,
                      listing_fn: Optional[Callable] = None,
                      order_key: Optional[Callable] = None,
                      on_stage: Optional[Callable] = None
                      ) -> tuple[list[list[dict]], dict]:
    """The map-reduce grouping itself: ``(groups, stats)``, each group a list of the
    original *base* records, every record in exactly one group.

    Split out of :func:`run_map_reduce` so a caller that wants the PARTITION rather than
    persona evidence can have it. ``fact_dedup`` is that caller — it needs the member
    groups to plan merges, and the ``_evidence_entry`` collapse above is persona-shaped
    (recurrence, maturity, stage) and meaningless for a fact. The scale problem is the
    same in both stores, so the machinery is shared and only the collapse differs.

    *prompt* and *listing_fn* let a caller supply its own grouping instruction and item
    rendering; everything else — block sizing, the blob guard, rotation, the early exit
    when a round merges nothing — is the persona path's, unchanged.

    *order_key* decides which items share a block, and so which items can merge at all: the
    default is ``key``, a content hash, i.e. random adjacency. That is survivable for a
    persona theme (dozens of members, several reduce rounds) and fatal for a two-member
    fact paraphrase set, so ``fact_dedup`` overrides it. A caller that can order its items
    by *meaning* should not be here at all — it should block by that instead.
    """
    stats = {"items": len(base), "calls": 0, "rejected_blocks": 0,
             "block_size": int(block_size), "map_themes": 0, "rounds": []}
    if not base:
        return [], stats

    themes = _map_phase(base, generate_fn, block_size=block_size, prompt=prompt,
                        on_stage=on_stage, stats=stats, listing_fn=listing_fn,
                        order_key=order_key)
    stats["map_themes"] = len(themes)

    for rnd in range(1, max(0, int(max_reduce_rounds)) + 1):
        if len(themes) <= 1:
            break
        before = len(themes)
        _emit(on_stage, {"stage": "reduce_started", "round": rnd, "themes": before})
        # Rotate by half a block per round so each round chunks different neighbours.
        themes = _reduce_round(themes, generate_fn, block_size=block_size,
                               rotation=(rnd - 1) * (max(1, block_size) // 2),
                               prompt=prompt, stats=stats, listing_fn=listing_fn,
                               order_key=order_key)
        after = len(themes)
        stats["rounds"].append({"round": rnd, "before": before, "after": after})
        _emit(on_stage, {"stage": "reduce", "round": rnd, "before": before,
                         "after": after, "merged": before - after})
        if after >= before:                       # nothing merged — further rounds won't
            break

    return themes, stats


def run_map_reduce(base: list[dict], generate_fn: Callable, *,
                   block_size: int = DEFAULT_BLOCK_SIZE,
                   max_reduce_rounds: int = MAX_REDUCE_ROUNDS,
                   prompt: Optional[str] = None,
                   on_stage: Optional[Callable] = None) -> dict:
    """Cluster *base* (``gather_persona_raw`` output) into themes. Returns
    ``{"evidence": [...], "stats": {...}}``.

    ``evidence`` matches ``reflection_digest.cluster_persona_evidence``'s shape and
    ordering exactly (``_evidence_entry`` collapse, recency-weighted recurrence first),
    so it is a drop-in replacement. ``stats`` reports the work done — LLM calls, blocks
    the blob guard rejected, per-round theme counts — which is what makes a dry run
    worth reading.
    """
    themes, stats = map_reduce_groups(
        base, generate_fn, block_size=block_size,
        max_reduce_rounds=max_reduce_rounds, prompt=prompt, on_stage=on_stage)
    if not base:
        return {"evidence": [], "stats": stats}

    evidence = [_evidence_entry(members) for members in themes]
    evidence.sort(key=lambda r: (-_weighted_recurrence(r),
                                 -r["recurrences"], -r["stage"]))
    stats["themes"] = len(evidence)
    return {"evidence": evidence, "stats": stats}


def map_reduce_cluster(base: list[dict], generate_fn: Callable, **kwargs) -> list[dict]:
    """``run_map_reduce`` returning only the evidence — drop-in for
    ``reflection_digest.cluster_persona_evidence``."""
    return run_map_reduce(base, generate_fn, **kwargs)["evidence"]


# ── self-test ─────────────────────────────────────────────────────────────── #

def _selftest() -> None:
    """GPU-free: a fake generate_fn groups by a tag embedded in each statement."""
    import re

    def _mk(key: str, content: str, sessions) -> dict:
        return {"key": key, "content": content, "stage": 0,
                "sessions": set(sessions),
                "session_weights": {s: 1.0 for s in sessions},
                "counter_sessions": set(), "counter_weights": {}}

    # 90 statements across 6 true themes, tagged "[tN]" so the fake model can group them.
    base = []
    for i in range(90):
        theme = i % 6
        base.append(_mk(f"k{i:03d}", f"[t{theme}] statement number {i}", [f"s{i // 3}"]))

    def fake_generate(listing, system, **kw):
        by_tag: dict[str, list[int]] = {}
        for line in listing.splitlines():
            m = re.match(r"\s*(\d+)\.\s*\[(t\d+)\]", line)
            if m:
                by_tag.setdefault(m.group(2), []).append(int(m.group(1)))
        return "\n".join("GROUP: " + ", ".join(str(n) for n in nums)
                         for nums in by_tag.values())

    out = run_map_reduce(base, fake_generate, block_size=12)
    ev, stats = out["evidence"], out["stats"]
    assert stats["items"] == 90
    # Every statement survives exactly once — the partition is never lossy.
    total = sum(e["cluster_size"] for e in ev)
    assert total == 90, f"lost members: {total} != 90"
    # Map alone cannot see across blocks (8 blocks × 6 themes); reduce must collapse them.
    assert stats["map_themes"] > 6, stats["map_themes"]
    assert len(ev) == 6, f"reduce did not converge to 6 themes: {len(ev)}"
    # Ordering contract: recency-weighted recurrence, descending.
    weights = [_weighted_recurrence(e) for e in ev]
    assert weights == sorted(weights, reverse=True), weights
    print(f"map-reduce: 90 → {stats['map_themes']} (map) → {len(ev)} themes "
          f"in {stats['calls']} calls, rounds={stats['rounds']}")

    # Determinism: same input, same fake model, same partition.
    again = run_map_reduce(base, fake_generate, block_size=12)["evidence"]
    assert [e["key"] for e in again] == [e["key"] for e in ev], "not deterministic"
    print("determinism: ok")

    # Blob guard: a model that lumps a whole block into one group is rejected, and the
    # block degrades to singletons rather than fusing unrelated statements.
    def blob_generate(listing, system, **kw):
        n = len(listing.strip().splitlines())
        return "GROUP: " + ", ".join(str(i + 1) for i in range(n))

    blob = run_map_reduce(base, blob_generate, block_size=12)
    assert blob["stats"]["rejected_blocks"] > 0
    assert len(blob["evidence"]) == 90, f"blob guard let a merge through: {len(blob['evidence'])}"
    print(f"blob guard: rejected {blob['stats']['rejected_blocks']} block(s), "
          f"{len(blob['evidence'])} singletons kept")

    # A model that returns nothing parseable degrades the same conservative way.
    dead = run_map_reduce(base, lambda *a, **k: "", block_size=12)
    assert len(dead["evidence"]) == 90
    print("unparseable output: falls back to singletons")

    # Small corpora: one block, no reduce rounds needed.
    small = run_map_reduce(base[:5], fake_generate, block_size=40)
    assert sum(e["cluster_size"] for e in small["evidence"]) == 5
    print(f"small corpus: 5 → {len(small['evidence'])} themes, "
          f"{small['stats']['calls']} call(s)")

    assert run_map_reduce([], fake_generate)["evidence"] == []
    print("empty: ok")
    print("persona_cluster self-test passed")


if __name__ == "__main__":
    _selftest()
