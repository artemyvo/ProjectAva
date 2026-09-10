"""bench/select — the catalogue, the policy gates, the parse, the fast path (§3.2)."""

from assoc.selection import (AVA_POLICY, Policy, SUPPORT_POLICY, build_catalogue, chunk_subjects_of, fast_path_pick,
                             parse_picks, render_catalogue)


def test_parse_picks():
    assert parse_picks("PICKS:\n3\n1\n", 10, 8) == {"picks": [3, 1], "none": False, "out_of_range": []}
    assert parse_picks("PICKS:\nNONE", 10, 8)["none"] is True
    assert parse_picks("PICKS:\n- 2\n2\n99\n", 10, 8) == {"picks": [2], "none": False, "out_of_range": [99]}
    assert parse_picks("I think 4 and 6.\n", 10, 8)["picks"] == [4]
    assert parse_picks("", 10, 8)["none"] is True
    assert len(parse_picks("PICKS:\n" + "\n".join(str(i) for i in range(1, 30)), 40, 8)["picks"]) == 8


def test_catalogue_groups_and_nests(corpus):
    hits = corpus.pull("E142", scope={"version": "2.0", "platform": "linux"}, limit=30)
    allowed = set(corpus.store.current_doc_ids({"version": "2.0", "platform": "linux"}))
    entries, withheld = build_catalogue(corpus.store, corpus.build, hits, SUPPORT_POLICY, allowed_docs=allowed)
    text = render_catalogue(entries)
    assert "PASSAGES" in text
    nested = [e for e in entries if e.get("nested")]
    assert nested and any("E142" in e["line"] for e in nested)
    # The nested claim follows its passage and the numbering is continuous.
    nos = [e["no"] for e in entries]
    assert nos == list(range(1, len(entries) + 1))


def test_position_withheld_or_attributed_by_policy(corpus):
    hits = corpus.pull("Нож стоит смазать", limit=20)
    allowed = set(corpus.store.current_doc_ids({}))
    permit = Policy(name="permit")
    entries, withheld = build_catalogue(corpus.store, corpus.build, hits, permit, allowed_docs=allowed)
    lines = [e["line"] for e in entries if e["grain"] == "claim"]
    assert any(l.lstrip().startswith("[position ·") and "Нож стоит смазать" in l for l in lines)
    entries2, withheld2 = build_catalogue(corpus.store, corpus.build, hits, AVA_POLICY, allowed_docs=allowed)
    lines2 = [e["line"] for e in entries2 if e["grain"] == "claim"]
    assert not any("Нож стоит смазать" in l for l in lines2)
    assert withheld2["claims"].get("subject", 0) >= 1


def test_passage_withheld_when_every_claim_is(corpus):
    hits = corpus.pull("Нож стоит смазать", limit=20)
    allowed = set(corpus.store.current_doc_ids({}))
    strict = Policy(name="strict", withhold_facets=frozenset({"unclassified", "position", "event", "property"}),
                    withhold_subjects=frozenset({"person:_self", "person:артемий"}))
    entries, withheld = build_catalogue(corpus.store, corpus.build, hits, strict, allowed_docs=allowed)
    assert not any(e["grain"] == "chunk" and e["hit"].meta.get("key") == "chat-ru" for e in entries)
    assert withheld["passages"] >= 1


def test_quote_kinds_policy(corpus):
    hits = corpus.pull("Noam Keller", limit=20)
    allowed = set(corpus.store.current_doc_ids({}))
    entries, withheld = build_catalogue(corpus.store, corpus.build, hits, SUPPORT_POLICY, allowed_docs=allowed)
    assert not any(e["grain"] == "chunk" and e["hit"].meta.get("kind") == "chat" for e in entries)
    assert withheld["passages"] >= 2


def test_fast_path_rules(corpus):
    hits = corpus.pull("E142", scope={"version": "2.0", "platform": "linux"})
    cs = chunk_subjects_of(corpus.build)
    assert fast_path_pick(hits, "E142", SUPPORT_POLICY, cs) is not None
    assert fast_path_pick(hits, "E142", Policy(fast_path=False), cs) is None
    prose = corpus.pull("how do I rotate the token?", scope={"version": "2.0", "platform": "linux"})
    assert fast_path_pick(prose, "how do I rotate the token?", SUPPORT_POLICY, cs) is None
    long_cue = "E142 appears whenever the upstream is slow and the customer has asked what to do about it now"
    assert fast_path_pick(hits, long_cue, SUPPORT_POLICY, cs) is None


def test_fast_path_agrees_with_the_pass(corpus):
    """The fast path must inject what a picker choosing the exact row would have chosen."""
    fast = corpus.inject("E142", scope={"version": "2.0", "platform": "linux"}, policy=SUPPORT_POLICY)

    def picker(system, user, **kw):
        no = next(l for l in user.splitlines() if "Code E142" in l and "Set retry_backoff" in l).split(".")[0].strip()
        return f"PICKS:\n{no}\n"
    slow = corpus.inject("E142", scope={"version": "2.0", "platform": "linux"}, policy=SUPPORT_POLICY, generate_fn=picker, force_model=True)
    assert slow.hits[0]["id"] in {h["id"] for h in fast.hits}


def test_pick_rank_log(corpus):
    p = corpus.store.root / "state" / "picks.jsonl"
    assert p.exists() and p.read_text().strip()
