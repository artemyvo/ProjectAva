"""bench/xlingual (§9): a Russian cue against English claims and the reverse, reached
through L2 with the cell named; the same cue through L1 alone expected to miss; the alias
table folding two spellings of one person into one subject block."""

import pytest


def _claims(corpus, pred):
    return [c for c in corpus.build.claims.values() if pred(c)]


def test_ru_cue_reaches_en_claim_through_cells(corpus):
    if corpus.embedder.id.startswith("hash"):
        pytest.skip("cross-lingual cells need the real term embedder")
    hits = corpus.pull("сестра живёт в Хайфе", limit=20)
    claim_hits = [h for h in hits if h.grain == "claim" and "Haifa" in (h.claim or {}).get("text", "")]
    assert claim_hits, [((h.claim or {}).get("text") or h.meta.get("key"))[:60] for h in hits[:8]]
    h = claim_hits[0]
    assert h.channels.get("concept", 0) > 0
    ex = corpus.explain(h)
    concept = next(p for p in ex["path"] if p["channel"] == "concept")
    assert concept["cells"] and any(any("haifa" in m.lower() for m in c["members"]) for c in concept["cells"])


def test_en_cue_reaches_ru_chunk(corpus):
    if corpus.embedder.id.startswith("hash"):
        pytest.skip("cross-lingual cells need the real term embedder")
    hits = corpus.pull("sister works at the Technion", limit=20)
    ru = [h for h in hits if h.meta.get("key") == "chat-haifa-ru"]
    assert ru and ru[0].rank <= 6


def test_l1_alone_misses_across_languages(corpus):
    hits = corpus.pull("сестра живёт в Хайфе", limit=20, channels={"lexical"})
    assert not any("Haifa" in (h.claim or {}).get("text", "") for h in hits)


def test_alias_table_folds_subjects(corpus):
    en = _claims(corpus, lambda c: "Haifa" in c["text"] and c["kind"] == "chat")
    ru = _claims(corpus, lambda c: "Хайф" in c["text"] and c["kind"] == "chat")
    assert en and ru
    assert {c["subject"] for c in en + ru} == {"person:artemy"}
    assert any(c.get("subject_alias_of") == "person:артемий" for c in ru)


def test_cross_lingual_paraphrases_merge_or_nominate(corpus):
    """The EN and RU statements of one fact share a subject block; with the real embedder
    tier 3 nominates them and the equivalence check lets them merge (same polarity, no
    condition, same resolved order) — the count is then 2 sources."""
    if corpus.embedder.id.startswith("hash"):
        pytest.skip("needs the real embedder")
    haifa = _claims(corpus, lambda c: c["subject"] == "person:artemy" and ("Haifa" in c["text"] or "Хайф" in c["text"]))
    merged = [c for c in haifa if c.get("n_sources", 1) >= 2]
    rep = corpus.build.manifest["report"]
    assert merged or rep["dedup"]["nominated"] >= 1, (rep["dedup"], [c["text"] for c in haifa])
