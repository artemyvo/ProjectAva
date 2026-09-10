"""bench/kinds — one mechanism, several witnesses (§1.3): protocol shapes per kind, every
fact anchored to a chunk, a norm never rendered as a property."""

from assoc.render import claim_line
from assoc.kinds import get


def _facts(corpus, key):
    doc = corpus.store.document(corpus.store.latest_id(key))
    return doc, doc.facts["facts"]


def test_every_fact_is_anchored_to_a_chunk_of_its_document(corpus):
    for doc in corpus.store.documents():
        if not doc.facts:
            continue
        ids = {u.chunk_id for u in doc.units}
        for f in doc.facts["facts"]:
            assert f.get("chunk_id") in ids, (doc.key, f["text"])
            assert f.get("span") and doc.text[f["span"][0]:f["span"][1]], (doc.key, f["text"])


def test_tech_doc_structure_facts(corpus):
    doc, facts = _facts(corpus, "docs/linux/errors.md")
    rows = [f for f in facts if f["fact_class"] == "standing" and f["subject"].startswith("ident:nimbus:E")]
    assert len(rows) == 200
    e142 = next(f for f in rows if f["subject"] == "ident:nimbus:E142")
    assert "retry_backoff" in e142["text"] and e142["version"] == "2.0" and e142["grounded"] is True
    doc, facts = _facts(corpus, "docs/linux/faq.md")
    faq = [f for f in facts if f.get("faq")]
    assert {f["fact_class"] for f in faq} == {"procedure", "standing"}
    assert all("#" not in f["text"] for f in faq)
    doc, facts = _facts(corpus, "docs/linux/release-notes.md")
    ev = [f for f in facts if f["fact_class"] == "event"]
    assert any("Removed the `--insecure` flag" in f["text"] for f in ev) and all(f["when"] == "v2.0" for f in ev)


def test_news_protocol_shape(corpus):
    doc, facts = _facts(corpus, "news/starling")
    assert doc.status["status"] == "extracted"
    spec = get("news")
    assert {spec.facet_map[f["fact_class"]] for f in facts if f["grounded"]} >= {"report", "event", "property"}
    assert all(f["subject"].startswith("entity:") for f in facts)
    mars = next(f for f in facts if "Mars" in f["text"])
    assert mars["grounded"] is False
    assert doc.facts["report"]["llm"]["ungrounded"] == 1


def test_chat_protocol_shape(corpus):
    doc, facts = _facts(corpus, "chat-ru")
    assert all(f["subject"].startswith("person:") for f in facts)
    assert any(f["subject"] == "person:_self" for f in facts)
    assert all(f["language"] == "cyr" for f in facts)
    assert {f["fact_class"] for f in facts} == {"event", "standing", "stated"}


def test_code_protocol_shape(corpus):
    doc, facts = _facts(corpus, "demo/client.py")
    classes = {f["fact_class"] for f in facts}
    assert classes >= {"signature", "defines", "imports", "calls", "docstring", "todo"}
    assert all(f["grounded"] is True and f["anchor"] == "exact" for f in facts)
    assert not any("sk-" in f["text"] for f in facts)


def test_norm_renders_attributed_never_as_property(corpus):
    claim = {"text": "connect_timeout defaults to 60 seconds", "facet": "norm", "version": "2.0", "n_sources": 3,
             "occurrences": [{"title": "Configuration", "asserted_at": "2026-08-01", "version": "2.0"}]}
    line = claim_line(claim)
    assert line.startswith("«Configuration» v2.0 specifies:")
    assert "sources" not in line
    pos = {"text": "Kestrel is a good place to work", "facet": "position", "subject_raw": "Artemy",
           "occurrences": [{"asserted_at": "2026-05-02", "speaker": "Artemy"}]}
    assert claim_line(pos) == "Artemy (2026-05-02): Kestrel is a good place to work"


def test_gist_and_passage_are_labelled(corpus):
    from assoc.render import passage_text, gist_text
    from assoc.puller import Hit
    doc = corpus.store.document(corpus.store.latest_id("chat-kestrel"))
    u = doc.units[0]
    text, ref = passage_text(corpus.store, Hit(grain="chunk", id=u.chunk_id, score=1, doc_id=doc.doc_id, chunk_id=u.chunk_id))
    assert text.startswith("[conversation with Pavel, 2026-08-14, exchange 0]") and "Pavel: Are you guys hiring" in text
    doc = corpus.store.document(corpus.store.latest_id("docs/linux/install.md"))
    text, ref = gist_text(corpus.store, Hit(grain="gist", id=doc.doc_id, score=1, doc_id=doc.doc_id, chunk_id=None))
    assert text.startswith("[«Install» —") and "sections" in text
