"""bench/pivot (§9, §6): senses induced from contexts; the anchor scorer picks the word in
play whose far sense is cold; the jump lands in the other sense; both senses warm → near
zero; a root bridge; a sound bridge; a stoplisted word never anchors; no access recorded."""

import pytest

from assoc.senses import root_bridges, sound_bridges, transliterate


def _senses(corpus):
    if corpus.embedder.id.startswith("hash"):
        pytest.skip("sense induction needs the real term embedder")
    return corpus.build.senses


def test_sense_induction_finds_two_senses(corpus):
    senses = _senses(corpus)
    assert "ключ" in senses, sorted(senses)[:20]
    ks = senses["ключ"]["senses"]
    assert len(ks) >= 2
    words = [" ".join(m for _c, ms in s["cells"] for m in ms) for s in ks]
    door = [w for w in words if any(x in w for x in ("замок", "дверь", "квартир", "lock", "door"))]
    spring = [w for w in words if any(x in w for x in ("вода", "родник", "лес", "камен", "water", "spring"))]
    assert door and spring, words
    assert senses["ключ"]["split"] > 0.3


def test_anchor_and_jump_from_door_to_spring(corpus):
    _senses(corpus)
    jumps = corpus.pivot("мы потеряли ключ от квартиры и вызвали слесаря, замок на двери старый")
    assert jumps and jumps[0].bridge == "ключ" and jumps[0].kind == "sense"
    j = jumps[0]
    to = " ".join(m for ms in j.to_sense for m in ms)
    assert any(x in to for x in ("вода", "родник", "лес", "камен", "water", "spring")), j.to_sense
    keys = {h.meta.get("key") for h in j.hits}
    assert "pivot/keys" in keys
    texts = [corpus.store.document(h.doc_id).unit(h.chunk_id).text for h in j.hits if h.grain == "chunk"][:5]
    assert any("родник" in t or "вода" in t for t in texts), texts


def test_both_senses_warm_scores_near_zero(corpus):
    _senses(corpus)
    both = corpus.pivot("ключ от квартиры и замок на двери; на даче родник, ключ бьёт из-под камня, вода в лесу")
    door = corpus.pivot("ключ от квартиры и замок на двери у слесаря")
    s_both = next((j.score for j in both if j.bridge == "ключ"), 0.0)
    s_door = next((j.score for j in door if j.bridge == "ключ"), 0.0)
    assert s_door > 0 and s_both < s_door * 0.6, (s_door, s_both)


def test_root_bridge(corpus):
    vocab = list(corpus.build.glossary.postings)
    r = root_bridges("колокол", vocab)
    assert any(w.startswith("колокольчик") for w, _ in r) or any(w.startswith("колокольн") for w, _ in r), r[:5]
    jumps = corpus.pivot("старый колокол звонил к вечерне", bridge="root")
    assert jumps and jumps[0].bridge.startswith("колокол") and jumps[0].target != jumps[0].bridge


def test_sound_bridge_transliteration(corpus):
    vocab = list(corpus.build.glossary.postings)
    assert transliterate("магазин") == "magazin"
    s = sound_bridges("магазин", vocab)
    assert any(w == "magazine" and k == "transliteration" for w, _st, k in s), s[:5]
    jumps = corpus.pivot("магазин на углу закрылся", bridge="sound")
    assert jumps and any(j.target == "magazine" for j in jumps)


def test_stoplisted_word_never_anchors_and_no_access(corpus):
    _senses(corpus)
    before = corpus.activation.accesses("chunk:" + next(iter(corpus.build.chunk_doc)))
    jumps = corpus.pivot("и в на с по это как")
    assert not jumps
    corpus.pivot("ключ от квартиры")
    after = corpus.activation.accesses("chunk:" + next(iter(corpus.build.chunk_doc)))
    assert before == after
