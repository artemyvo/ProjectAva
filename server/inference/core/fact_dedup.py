"""Semantic de-duplication of live ``[fact]`` RAG items on the clean base.

Facts accumulate across reflection runs, de-duplicated only by *exact* normalized
content key (``reflection_writer.content_key`` — case/space/punctuation-insensitive
hash). Paraphrases and cross-lingual restatements therefore survive as distinct
live items and can all surface together at chat time. This module collapses them by
**meaning** using a one-shot grouping pass on the loaded model — mirroring
``reflection_digest``'s persona clustering, and reusing its ``_parse_groups``
partition parser.

The grouping pass is meant to run on the **clean base** (adapter OFF): it is an
evaluation of what's redundant, not Ava's expression, so it should be replay-faithful
and immune to a bad adapter. The caller (``server.handle_dedup_facts``) enters a
``CleanBaseSession`` first; this module never touches the GPU itself.

Split for GPU-free testability, like the digest:
  * ``cluster_facts(facts, generate_fn, embed_fn=…)`` — the grouping calls → paraphrase
    groups.
  * ``plan_merges`` / ``choose_survivor`` / ``merged_trigger`` / ``summarize`` — pure
    string logic over the groups, unit-testable without a model.

**Blocking is the whole game.** A grouping call sees one block at a time, so two facts
can only merge if they land in the SAME block — which makes the order the corpus is cut
into blocks the load-bearing decision, not an implementation detail. The persona path
orders by ``key``, a content hash, i.e. at random; that is harmless there only because a
persona theme has dozens of members and reduce rounds eventually pair them. A fact
paraphrase set has two or three members, so random blocking never pairs them: measured on
a live 909-fact store, seven restatements of one evening scattered across six of the 23
blocks and the pass reported zero merges while every duplicate sat in the store. So the
primary path blocks by **subject** — ``fact_contradict.cluster_by_subject`` (the same
recall-cue embedding clustering, partitioned by whom the fact is about) — and packs those
clusters contiguously into blocks. Subjects with a single fact are emitted as singletons
with no LLM call at all, which is also what makes the pass affordable (719 of 909 facts in
18 blocks, against 23 blind ones).

The **merge policy** is *merge-on-content, union-triggers*: a group collapses to one
survivor (an existing member — never a synthesized phrasing, which would mint a new
content key and lose provenance), and the survivor inherits the UNION of the group's
triggers so it is still recalled by every topic that recalled any member. Because a
``[fact]`` embeds on its *trigger* (``ReflectionMemory.embed_text``), a combined
trigger is a single, slightly fuzzier vector spanning all topics — the honest cost of
the content-keyed fold (two live records can't share a content key). The union is
capped at ``MAX_UNION_TRIGGERS`` so it can't smear indefinitely.

Writing is delegated to ``ReflectionWriter.write_dedup`` (append-only evict + survivor
re-insert, so a dedup is reverted by dropping the lines it added). Wired to the Debug
tab now; wiring it into the reflection loop is a separate, later task.

GPU-free self-test: ``python -m core.fact_dedup``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.persona_cluster import DEFAULT_BLOCK_SIZE as _PERSONA_BLOCK_SIZE
from core.reflection_digest import _parse_groups   # generic numbered-item partition parser

# A survivor's merged trigger fuses this many distinct recall cues at most, so the
# fact's single embedding vector can't be smeared across an unbounded topic set.
MAX_UNION_TRIGGERS = 4

# Above this many live facts the grouping switches from one flat call to a blocked path
# (see ``cluster_facts``). Deliberately the persona module's own block size: it is the size
# of grouping call a model answers reliably, which is a property of the call and not of
# what is being grouped.
DEFAULT_BLOCK_SIZE = _PERSONA_BLOCK_SIZE

# A trailing block smaller than this is folded into the previous one instead of being sent
# alone. Not cosmetic: ``persona_cluster``'s blob guard rejects a block whose largest group
# covers half of it (floor ``MIN_BLOB_MEMBERS``), so a small block silently caps how many
# facts may legitimately merge — exactly backwards for the "ten restatements of one
# evening" case this pass exists for.
MIN_TAIL_BLOCK = DEFAULT_BLOCK_SIZE // 2


def load_dedup_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The fact-dedup grouping prompt (``prompts/fact_dedup_prompt.txt``), or a safe
    inline fallback so the pass works even if the file is missing."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "fact_dedup_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "You are grouping factual notes you have recalled. Below are numbered facts "
        "distilled across past reflections — some in different languages or phrasings "
        "that state the same thing; each may show, in parentheses, when it should be "
        "recalled.\n\n"
        "Group the numbers so that facts stating the SAME underlying truth are in one "
        "group — even when the wording, language, or recall cue differs. Keep genuinely "
        "distinct facts in separate groups (a shared topic is not enough). Every number "
        "appears in exactly one group; a one-of-a-kind fact is a group of one. Judge by "
        "meaning, not surface words.\n\n"
        "Output only lines of this form, nothing else:\n"
        "GROUP: <comma-separated numbers>"
    )


def _listing(facts: list[dict]) -> str:
    """Number the facts for the grouping prompt, annotating each with its recall cue
    (``trigger``) so the model can judge whether two share both truth AND cue."""
    lines = []
    for i, f in enumerate(facts, 1):
        content = (f.get("content") or "").strip()
        trigger = (f.get("trigger") or "").strip()
        if trigger:
            lines.append(f"{i}. {content}   (recalled when: {trigger})")
        else:
            lines.append(f"{i}. {content}")
    return "\n".join(lines)


def _emit(on_stage: Optional[Callable], info: dict) -> None:
    """Report progress, best-effort — a broken reporter must never fail the pass."""
    if on_stage is None:
        return
    try:
        on_stage(info)
    except Exception:
        pass


def pack_blocks(clusters: list[list[dict]], block_size: int) -> list[list[dict]]:
    """Pack subject clusters into grouping blocks of at most *block_size*, keeping each
    cluster CONTIGUOUS — the point of the whole exercise, since a cluster split across two
    calls is a cluster whose members can no longer merge.

    A cluster larger than a block is cut into block-sized pieces (it is one subject, so
    even a partial view still pairs most of it), and a short trailing block is folded back
    into the previous one — see ``MIN_TAIL_BLOCK``. Pure; unit-tested below.
    """
    block_size = max(1, int(block_size))
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    for c in clusters:
        if len(c) >= block_size:
            if cur:
                blocks.append(cur)
                cur = []
            for i in range(0, len(c), block_size):
                blocks.append(list(c[i:i + block_size]))
            continue
        if len(cur) + len(c) > block_size:
            blocks.append(cur)
            cur = []
        cur.extend(c)
    if cur:
        if blocks and len(cur) < MIN_TAIL_BLOCK:
            blocks[-1].extend(cur)       # never send a stub block — see MIN_TAIL_BLOCK
        else:
            blocks.append(cur)
    return blocks


def _cluster_by_subject(facts: list[dict], generate_fn: Callable, embed_fn: Callable, *,
                        block_size: int, on_stage: Optional[Callable]
                        ) -> Optional[list[list[dict]]]:
    """The primary path: block by subject, then one grouping call per block.

    Subject clustering is the cheap CPU embedding pass ``fact_contradict`` already uses to
    decide which facts may be compared at all — recall-cue cosine, partitioned by whom the
    fact is about, so one person's fact can never merge into another's. A subject holding a
    single fact has nothing to merge with and is emitted as a singleton without a call.
    """
    from core import fact_contradict
    from core import persona_cluster

    clusters = fact_contradict.cluster_by_subject(facts, embed_fn)
    if not clusters:
        return None
    singles = [c for c in clusters if len(c) == 1]
    multi = sorted((c for c in clusters if len(c) >= 2),
                   key=lambda c: (-len(c), c[0].get("key") or ""))
    _emit(on_stage, {"stage": "clustered", "facts": len(facts),
                     "subjects": len(clusters), "multi": len(multi),
                     "singletons": len(singles),
                     "candidates": sum(len(c) for c in multi)})
    if not multi:
        return list(singles)             # nothing shares a subject — a real, honest no-op

    blocks = pack_blocks(multi, block_size)
    groups: list[list[dict]] = list(singles)
    prompt = load_dedup_prompt()
    for i, block in enumerate(blocks, 1):
        got = persona_cluster.group_block(block, generate_fn, prompt=prompt,
                                          listing_fn=_listing)
        rejected = got is None
        if rejected:
            got = [[it] for it in block]  # conservative: no merges out of this block
        groups.extend(got)
        _emit(on_stage, {"stage": "block", "i": i, "n": len(blocks),
                         "items": len(block), "groups": len(got),
                         "merged": sum(1 for g in got if len(g) > 1),
                         "rejected": rejected})
    return groups


def cluster_facts(facts: list[dict], generate_fn: Callable, *,
                  embed_fn: Optional[Callable] = None,
                  block_size: int = DEFAULT_BLOCK_SIZE,
                  on_stage: Optional[Callable] = None
                  ) -> Optional[list[list[dict]]]:
    """Group the facts into paraphrase sets, judging by meaning rather than surface
    words — so cross-lingual restatements the embedder can't reach still merge.

    Returns a list of groups (each a list of the original records, every fact in exactly
    one group) or ``None`` when nothing usable came back — the signal for the caller to
    treat the run as a no-op rather than dedup blindly. Greedy (``temperature=0``) for
    reproducibility; the append-only op-log is the replay record either way.

    **Three regimes.** With an *embed_fn* (every server caller has one — it is the RAG
    embedder, already loaded, CPU, independent of the swapped-out GPU model) the corpus is
    blocked by subject, which is the only regime that reliably pairs a paraphrase with its
    original; see the module docstring. Without one, the old behaviour stands: at or below
    *block_size* facts, ONE flat call over a content-ordered listing; above it,
    ``persona_cluster.map_reduce_groups`` over that same content ordering, because a flat
    partition does not survive scale and fails *silently* — the listing outgrows the
    context, the answer truncates, and ``_parse_groups`` completes the remainder as
    singletons, so the pass reports a handful of merges and looks like it worked.

    The fact-side listing is preserved across every regime via ``listing_fn``: each fact
    carries its recall cue, because two facts stating one truth under different triggers
    are not the same live record and must not be merged into one.
    """
    facts = [f for f in facts if (f.get("content") or "").strip()]
    if len(facts) < 2:
        return None
    if embed_fn is not None:
        try:
            grouped = _cluster_by_subject(facts, generate_fn, embed_fn,
                                          block_size=block_size, on_stage=on_stage)
        except Exception:
            grouped = None               # fall through to the embedder-free path
        if grouped:
            return grouped
    # Content order, not key order: without an embedder, lexical adjacency is the only
    # thing left that puts a restatement anywhere near its original.
    ordered = sorted(facts, key=lambda f: (f.get("content") or ""))
    if len(ordered) > max(1, int(block_size)):
        from core import persona_cluster
        groups, _stats = persona_cluster.map_reduce_groups(
            ordered, generate_fn, block_size=block_size,
            prompt=load_dedup_prompt(), listing_fn=_listing, on_stage=on_stage,
            order_key=lambda f: (f.get("content") or ""))
        return groups or None
    try:
        resp = generate_fn(
            _listing(ordered), load_dedup_prompt(),
            temperature=0.0, top_p=1.0,
            max_new_tokens_setting="1024",
            before_session="", disable_rag=True, disable_thinking=True)
    except Exception:
        return None
    groups = _parse_groups(resp, len(ordered))
    if not groups:
        return None
    return [[ordered[j] for j in g] for g in groups]


def choose_survivor(group: list[dict]) -> dict:
    """The member to keep: most-recalled wins (the canonical phrasing Ava actually
    surfaces), tie-broken by the newest timestamp. Always an existing record — never a
    synthesized statement, which would mint a new content key and orphan provenance."""
    return max(
        group,
        key=lambda f: (int(f.get("surface_count", 0) or 0), (f.get("ts") or "")),
    )


def merged_trigger(group: list[dict], survivor: dict) -> Optional[str]:
    """Union of the group's distinct recall cues (survivor's first so its topic stays
    primary), capped at ``MAX_UNION_TRIGGERS``. ``None`` when no member had a trigger."""
    seen: set[str] = set()
    out: list[str] = []
    for f in [survivor] + [g for g in group if g is not survivor]:
        t = (f.get("trigger") or "").strip()
        if not t:
            continue
        norm = t.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(t)
        if len(out) >= MAX_UNION_TRIGGERS:
            break
    if not out:
        return None
    return " ; ".join(out)


def plan_merges(groups: list[list[dict]]) -> list[dict]:
    """Turn paraphrase groups into a concrete dedup plan (skipping singletons).

    Each merge names the survivor, the losers to evict, and the survivor's merged
    trigger. Pure — the caller either reports it (dry-run) or hands it to
    ``ReflectionWriter.write_dedup``.
    """
    merges: list[dict] = []
    for g in groups:
        if len(g) < 2:
            continue
        survivor = choose_survivor(g)
        losers = [f for f in g if f is not survivor]
        merges.append({
            "survivor_key": survivor.get("key"),
            "survivor_content": (survivor.get("content") or "").strip(),
            "survivor_record": survivor,
            "merged_trigger": merged_trigger(g, survivor),
            "losers": losers,
            "loser_keys": [l.get("key") for l in losers],
        })
    return merges


def summarize(merges: list[dict]) -> list[dict]:
    """Client-facing view of a plan: one entry per merge group, listing the survivor,
    the evicted phrasings, and the merged trigger. Drops the raw records."""
    out: list[dict] = []
    for m in merges:
        out.append({
            "survivor": m["survivor_content"],
            "survivor_key": m["survivor_key"],
            "merged_trigger": m["merged_trigger"],
            "evicted": [(l.get("content") or "").strip() for l in m["losers"]],
            "evicted_keys": m["loser_keys"],
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    facts = [
        {"key": "a", "kind": "fact", "content": "Artemy lives in Moscow.",
         "trigger": "where the user lives", "surface_count": 3, "ts": "2026-01-01"},
        {"key": "b", "kind": "fact", "content": "The user is based in Moscow.",
         "trigger": "user's city", "surface_count": 1, "ts": "2026-02-01"},
        {"key": "c", "kind": "fact", "content": "Пользователь живёт в Москве.",
         "trigger": "город пользователя", "surface_count": 0, "ts": "2026-03-01"},
        {"key": "d", "kind": "fact", "content": "Artemy prefers dark roast coffee.",
         "trigger": "coffee preference", "surface_count": 5, "ts": "2026-01-15"},
    ]

    # Fake generate_fn: groups the three Moscow paraphrases (1,2,3 after sort), coffee alone.
    def fake_generate(listing, system, **kw):
        # ordered by content: "Artemy lives...", "Artemy prefers...", "The user...", "Пользователь..."
        # → 1=lives Moscow, 2=coffee, 3=user Moscow, 4=Москве
        return "GROUP: 1, 3, 4\nGROUP: 2"

    groups = cluster_facts(facts, fake_generate)
    assert groups is not None, "expected groups"
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 3], f"expected a 3-group and a singleton, got {sizes}"

    merges = plan_merges(groups)
    assert len(merges) == 1, f"expected one multi-member merge, got {len(merges)}"
    m = merges[0]
    # Survivor = most-surfaced Moscow fact (surface_count 3, key 'a').
    assert m["survivor_key"] == "a", f"survivor should be 'a', got {m['survivor_key']}"
    assert len(m["losers"]) == 2, f"expected 2 losers, got {len(m['losers'])}"
    # Union trigger: survivor's cue first, then the others, distinct.
    mt = m["merged_trigger"]
    assert mt.startswith("where the user lives"), f"survivor cue should lead: {mt!r}"
    assert "user's city" in mt and "город пользователя" in mt, f"cues not unioned: {mt!r}"

    # Cap enforcement.
    big = [{"key": str(i), "kind": "fact", "content": "x",
            "trigger": f"cue{i}", "surface_count": 0, "ts": ""} for i in range(10)]
    mt2 = merged_trigger(big, choose_survivor(big))
    assert mt2.count(";") == MAX_UNION_TRIGGERS - 1, f"trigger union not capped: {mt2!r}"

    # ── scale: above DEFAULT_BLOCK_SIZE the call must go through map-reduce ──
    # The regression this guards is silent: a flat call over an oversized listing
    # truncates, _parse_groups completes the remainder as singletons, and the pass
    # reports a few merges and looks like it worked.
    n = DEFAULT_BLOCK_SIZE * 2 + 5
    many = [{"key": f"k{i:03d}", "kind": "fact", "content": f"fact number {i:03d}",
             "trigger": f"cue {i:03d}", "surface_count": i, "ts": f"2026-01-{i % 28 + 1:02d}"}
            for i in range(n)]
    seen_sizes: list[int] = []

    def blocked_generate(listing, system, **kw):
        # Every call must be a BLOCK, never the whole corpus — that is the property.
        size = len([ln for ln in listing.splitlines() if ln.strip()])
        seen_sizes.append(size)
        assert size <= DEFAULT_BLOCK_SIZE, f"listing of {size} exceeds one block"
        # The fact listing must carry recall cues in both phases (listing_fn threading).
        assert "recalled when:" in listing, "fact listing lost its trigger annotation"
        return "\n".join(f"GROUP: {i + 1}" for i in range(size))   # all singletons

    groups2 = cluster_facts(many, blocked_generate)
    assert groups2 is not None and len(groups2) == n, \
        f"expected {n} singleton groups, got {groups2 and len(groups2)}"
    assert len(seen_sizes) >= 3, f"expected several block calls, got {len(seen_sizes)}"
    assert not plan_merges(groups2), "singletons must plan no merges"

    # At or below the threshold it stays the single flat call it always was.
    flat_calls: list[int] = []

    def flat_generate(listing, system, **kw):
        flat_calls.append(len(listing.splitlines()))
        return "GROUP: 1\nGROUP: 2\nGROUP: 3\nGROUP: 4"

    cluster_facts(facts, flat_generate)
    assert len(flat_calls) == 1, f"small corpus should be one call, got {len(flat_calls)}"

    # Below-threshold input is a no-op.
    assert cluster_facts(facts[:1], fake_generate) is None

    rep = summarize(merges)
    assert rep[0]["survivor"] == "Artemy lives in Moscow."
    assert len(rep[0]["evicted"]) == 2

    # ── blocking: paraphrases must share a call ───────────────────────────────
    # The regression this guards is the one that shipped: with hash-key blocking, the
    # seven restatements of one evening in a live 909-fact store landed in six different
    # blocks, so no call ever saw two of them and the pass reported zero merges.
    n_sub = 60
    corpus = []
    for s in range(n_sub):
        # Subject 0..9 have three restatements each; the rest are one-of-a-kind.
        for c in range(3 if s < 10 else 1):
            corpus.append({"key": f"h{s:02d}{c}", "kind": "fact",
                           "content": f"restatement {c} of truth {s:02d}",
                           "trigger": f"topic {s:02d}", "surface_count": c,
                           "ts": f"2026-01-{s % 28 + 1:02d}"})

    def subject_embed(texts):
        # One-hot on the topic number: same topic ⇒ cosine 1.0, different ⇒ 0.0.
        vecs = []
        for t in texts:
            v = [0.0] * n_sub
            v[int(t.rsplit(" ", 1)[-1])] = 1.0
            vecs.append(v)
        return vecs

    def by_truth(listing, system, **kw):
        # Group the block's numbered lines by the truth id they mention.
        buckets: dict[str, list[int]] = {}
        for line in listing.splitlines():
            num, _, rest = line.partition(".")
            if not num.strip().isdigit():
                continue
            truth = rest.split("truth ", 1)[1][:2]
            buckets.setdefault(truth, []).append(int(num.strip()))
        return "\n".join("GROUP: " + ", ".join(str(x) for x in v) for v in buckets.values())

    stages: list[dict] = []
    groups3 = cluster_facts(corpus, by_truth, embed_fn=subject_embed,
                            on_stage=stages.append)
    assert groups3 is not None
    assert sum(len(g) for g in groups3) == len(corpus), "the partition lost members"
    merges3 = plan_merges(groups3)
    assert len(merges3) == 10, f"expected all 10 duplicate sets merged, got {len(merges3)}"
    assert all(len(m["losers"]) == 2 for m in merges3), "a restatement escaped its group"
    # Only the subjects with something to merge cost a call; the 50 lone facts cost none.
    clustered = [s for s in stages if s["stage"] == "clustered"][0]
    assert clustered["singletons"] == 50, clustered
    assert clustered["candidates"] == 30, clustered
    assert [s for s in stages if s["stage"] == "block"], "no per-block progress reported"

    # A broken embedder must degrade to the embedder-free path, never fail the pass.
    def bad_embed(texts):
        raise RuntimeError("embedder down")

    assert cluster_facts(facts, fake_generate, embed_fn=bad_embed) is not None

    # pack_blocks: clusters stay contiguous, blocks stay bounded, no stub tail.
    cl = [[{"key": f"c{i}-{j}"} for j in range(sz)]
          for i, sz in enumerate([7, 7, 7, 7, 7, 7, 3])]
    packed = pack_blocks(cl, 20)
    assert sum(len(b) for b in packed) == 45, packed
    assert all(len(b) <= 20 + MIN_TAIL_BLOCK for b in packed), [len(b) for b in packed]
    assert len(packed[-1]) >= MIN_TAIL_BLOCK, "stub tail block not folded back"
    for c in cl:                                  # every cluster whole, inside one block
        keys = {m["key"] for m in c}
        assert any(keys <= {m["key"] for m in b} for b in packed), f"cluster split: {keys}"
    # An oversized cluster is cut to block size rather than sent whole.
    assert all(len(b) <= 5 for b in pack_blocks([[{"key": str(i)} for i in range(12)]], 5))

    print("fact_dedup self-test OK")


if __name__ == "__main__":
    _selftest()
