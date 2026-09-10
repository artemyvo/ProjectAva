"""bench/sources (§9): one fact from four documents of which two are near-copies; a
contradiction from a fifth; a docs page in two versions; an identifier defined in two
libraries; the order pair not merged; a negated restatement linked as contests; the index
page ranked below the page that defines the thing."""

from assoc.selection import SUPPORT_POLICY


def _find(corpus, pred):
    return [c for c in corpus.build.claims.values() if pred(c)]


def test_syndicated_copies_vote_once(corpus):
    cs = _find(corpus, lambda c: "memory-pooling layer" in c["text"])
    assert cs
    c = max(cs, key=lambda c: c["n_sources"])
    # news/starling, news/syndicated, news/mirror state it; syndicated + mirror are one family.
    assert c["n_sources"] >= 3 and c["n_independent"] < c["n_sources"], (c["n_sources"], c["n_independent"], c["sources"])
    links = corpus.build.links
    fam = links["families"]
    ids = {corpus.store.latest_id("news/syndicated"), corpus.store.latest_id("news/mirror")}
    assert len({fam[i] for i in ids}) == 1


def test_contradiction_is_a_contest_rendered_both_sides(corpus):
    forty = _find(corpus, lambda c: "40 million" in c["text"] and "not" not in c["text"].lower() and c["kind"] == "news")
    assert forty
    contested = [c for c in forty if c.get("contests")]
    assert contested, [(c["text"], c.get("contests")) for c in forty]
    other = corpus.build.claims[contested[0]["contests"][0]]
    assert "not" in other["text"].lower()
    from assoc.render import contest_line
    line = contest_line(contested[0], other)
    assert "CONTESTED" in line and "40 million" in line and "not valued" in line


def test_negated_restatement_contests_and_order_pair_stays_separate(corpus):
    port = _find(corpus, lambda c: "8443" in c["text"] and c["subject"].endswith(":gateway") and c["kind"] == "tech_doc")
    positive = [c for c in port if "not" not in c["text"].lower()]
    negative = [c for c in port if "not" in c["text"].lower()]
    assert positive and negative
    # The two positive restatements (A / B) merged into one claim with two sources…
    assert any(c["n_sources"] >= 2 for c in positive), [(c["text"], c["n_sources"]) for c in positive]
    # …and the negation is linked, not merged.
    assert any(negative[0]["claim_id"] in c.get("contests", []) for c in positive)
    order = _find(corpus, lambda c: "must start before" in c["text"])
    assert len(order) == 2 and all(c["n_sources"] == 1 for c in order)


def test_identifier_namespaced_by_family(corpus):
    conn = _find(corpus, lambda c: c["subject"].endswith(":connect") and c["fact_class"] == "signature")
    subjects = {c["subject"] for c in conn}
    assert "ident:alpha:connect" in subjects and "ident:beta:connect" in subjects
    hits = corpus.pull("connect", scope={"product": "beta"}, limit=10)
    assert all(h.meta.get("key", "").startswith("beta/") or h.meta.get("key") in ("demo/client.py", "demo/conn.c") or "beta" in (h.claim or {}).get("subject", "")
               for h in hits if h.grain == "claim" and (h.claim or {}).get("subject", "").startswith("ident:")), \
        [(h.claim or {}).get("subject") for h in hits]


def test_index_page_ranks_below_defining_page(corpus):
    """The fan term on the source side (§2.5): a page that lists everything is the authority
    on nothing — its claims carry auth / claims(F), so the defining page's claim about
    `connect` outranks the index page's line about it, and leads the catalogue."""
    conn = [c for c in corpus.build.claims.values() if c["subject"] == "ident:alpha:connect"]
    by_key = {c["occurrences"][0]["key"]: c for c in conn}
    assert "alpha/index.md" in by_key and "alpha/client.md" in by_key
    assert by_key["alpha/client.md"]["authority"] > by_key["alpha/index.md"]["authority"], \
        {k: c["authority"] for k, c in by_key.items()}
    from assoc.selection import build_catalogue, SUPPORT_POLICY
    hits = corpus.pull("what does connect do in alpha", scope={"product": "alpha"}, limit=20)
    allowed = set(corpus.store.current_doc_ids({"product": "alpha"}))
    entries, _ = build_catalogue(corpus.store, corpus.build, hits, SUPPORT_POLICY, allowed_docs=allowed)
    first = next(e for e in entries if e["grain"] == "claim" and (e["hit"].claim or {}).get("subject") == "ident:alpha:connect")
    assert first["hit"].meta.get("key") == "alpha/client.md"


def test_version_scope_reaches_old_norm(corpus):
    v1 = corpus.pull("default connect_timeout", scope={"version": "1.0", "platform": "linux"})
    assert "30 seconds" in corpus.store.document(v1[0].doc_id).unit(v1[0].chunk_id).text


def test_report_counts_independent_in_render(corpus):
    from assoc.render import claim_line
    cs = _find(corpus, lambda c: "memory-pooling layer" in c["text"])
    c = max(cs, key=lambda c: c["n_sources"])
    line = claim_line(c)
    assert "independent" in line and f"{c['n_sources']} documents" in line
