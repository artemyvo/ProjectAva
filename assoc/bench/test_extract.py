"""bench/extract — the witness plumbing on scripted output (§1.6): chunk anchoring and its
repair, the context-only chunk, grounding, the self rule, the ingest report. The
model-dependent half (a real generation) lives in test_model.py."""

import json

import pytest

from assoc.library import Library
from assoc.protocol import parse_fact_lines
from assoc.witness import EXTRACT_WINDOW_CHARS, llm_witness, run_witnesses
from assoc.kinds import get
from assoc.bench import fixtures as fx
from assoc.bench.conftest import scripted_witness


def test_parse_markers_in_either_order_and_position():
    raw = ("Preamble text the parser ignores.\n"
           "[fact] (about: Starling) (class: standing) (chunk: 2) Starling builds a memory-pooling layer.\n"
           "- [fact] Nimbus acquired Starling. (about: Nimbus) (class: event) (when: 2026-08-20) (chunk: 1)\n"
           "[fact] (about: NAME) (class: standing) (chunk: 1) the fact, in one plain sentence.\n"
           "[fact] (about: Starling) (class: standing) (chunk: 2) Starling builds a memory-pooling layer.\n"
           "[fact] (class: made_up) (chunk: 9) A line with an unknown class and no subject.\n")
    facts = parse_fact_lines(raw, allowed_classes=("standing", "stated", "event"))
    assert [f["chunk_no"] for f in facts] == [2, 1, 9]
    assert facts[1]["when"] == "2026-08-20" and facts[1]["subject_raw"] == "Nimbus"
    assert facts[2]["fact_class"] == "unspecified"
    assert len(facts) == 3    # placeholder dropped, exact duplicate dropped


def test_wrapped_statement_is_unwrapped():
    raw = ('[fact] (about: Starling) (class: standing) (chunk: 1) (entities: Noam Keller) (stated: "the startup founded by Noam Keller in 2025")\n'
           "[fact] (about: the deal) (class: stated) (chunk: 1) (text: valued at 40 million dollars).\n")
    facts = parse_fact_lines(raw, allowed_classes=("standing", "stated", "event"))
    assert [f["text"] for f in facts] == ["the startup founded by Noam Keller in 2025", "valued at 40 million dollars"]
    assert facts[0]["entities"] == ["Noam Keller"] and facts[0]["chunk_no"] == 1


def test_answer_region_only_and_truncation():
    raw = "<think>[fact] (chunk: 1) draft line\n</think>\n[fact] (chunk: 1) final line\n[fact] (chunk: 1) cut mid"
    facts = parse_fact_lines(raw, allowed_classes=("standing",), truncated=True)
    assert [f["text"] for f in facts] == ["final line"]


def test_anchor_repair_and_context_only(fresh_root):
    lib = Library(fresh_root)
    doc_id = lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/starling", "date": "2026-08-20"})
    doc = lib.store.document(doc_id)
    lines = "\n".join([
        "[fact] (about: Starling) (class: standing) Starling builds a memory-pooling layer for GPU clusters.",   # no chunk marker
        "[fact] (about: Brightmem) (class: event) (chunk: 1) (when: 2021) Brightmem was acquired by Halden in 2021.",   # wrong chunk named
        "[fact] (about: Dana Levi) (class: stated) (chunk: 3) Dana Levi said the company is thrilled.",
    ])
    facts, report = llm_witness(doc, get("news"), scripted_witness(lines))
    by = {f["text"][:12]: f for f in facts}
    assert by["Starling bui"]["anchor"] == "inferred" and by["Starling bui"]["chunk_id"] == doc.units[1].chunk_id
    assert by["Brightmem wa"]["anchor"] == "moved" and by["Brightmem wa"]["chunk_id"] == doc.units[1].chunk_id
    assert by["Dana Levi sa"]["anchor"] == "exact" and by["Dana Levi sa"]["grounded"]
    assert report["anchor_inferred"] == 1 and report["anchor_moved"] == 1 and report["anchor_exact"] == 1


def test_section_groups_and_context_only_chunk(fresh_root):
    lib = Library(fresh_root)
    text = "\n\n".join(f"## Section {i}\n\nParagraph {i} about topic{i} with enough words to matter." for i in range(6))
    doc_id = lib.ingest("# Groups\n\n" + text, "tech_doc", {"key": "g.md", "version": "1"})
    doc = lib.store.document(doc_id)
    seen_users = []

    def gen(system, user, **kw):
        seen_users.append(user)
        # Echo one fact per marked chunk, naming the chunk number the material shows.
        out = []
        for line in user.splitlines():
            if line.startswith("[chunk ") and "context only" not in line:
                n = int(line.split()[1])
                out.append(f"[fact] (about: topic{n - 1}) (class: standing) (chunk: {n}) Paragraph {n - 1} about topic{n - 1} with enough words to matter.")
        return "\n".join(out)

    facts, report = llm_witness(doc, get("tech_doc"), gen, window=260)
    assert report["groups"] >= 3
    assert all("| context only —" in u for u in seen_users[1:]) and "| context only —" not in seen_users[0]
    prim = [u for u in doc.units if u.role == "primary"]
    # One fact per primary chunk, none duplicated across groups, every one grounded.
    assert sorted(f["chunk_id"] for f in facts) == sorted(u.chunk_id for u in prim if u.text.strip())
    assert all(f["grounded"] for f in facts)


def test_self_rule_and_report_counts(fresh_root):
    lib = Library(fresh_root)
    doc_id = lib.ingest(fx.chat_ru(), "chat", {"key": "chat-ru", "date": "2026-06-20"})
    rep = run_witnesses(lib.store, doc_id, generate_fn=scripted_witness(fx.chat_ru_protocol_lines()))
    doc = lib.store.document(doc_id)
    facts = doc.facts["facts"]
    assert any(f["subject"] == "person:_self" for f in facts)
    assert all(f["subject"] in ("person:_self", "person:артемий") for f in facts)
    assert rep["status"] == "extracted" and rep["facts"] == 4 and rep["llm"]["ungrounded"] == 0
    assert rep["chunks"] == 3 and rep["facts_per_chunk_max"] == 2


def test_failed_pass_retries_at_smaller_group(fresh_root):
    lib = Library(fresh_root)
    text = "\n\n".join(f"## S{i}\n\nBody {i} with a few more words here." for i in range(4))
    doc_id = lib.ingest("# R\n\n" + text, "tech_doc", {"key": "r.md", "version": "1"})
    doc = lib.store.document(doc_id)
    calls = []

    def gen(system, user, **kw):
        n = user.count("[chunk ") - user.count("| context only —")
        calls.append(n)
        if n > 2:
            return ("<think>never closes", {"truncated": True})
        return "\n".join(f"[fact] (about: S) (class: standing) (chunk: {i + 1}) Body {i} with a few more words here." for i in range(4))

    spec = get("tech_doc")
    spec2 = type(spec)(**{**spec.__dict__, "thinking": True})
    facts, report = llm_witness(doc, spec2, gen, window=10_000)
    assert calls[0] == 4 and report["retried_groups"] == 1 and report["failed_groups"] == 0
    assert report["groups"] == 3   # the failed one + its two halves


def test_translated_fact_grounds_through_cells(corpus):
    """§1.6 correction m9: a fact written in the other language than its chunk grounds
    through the L2 cells the chunk's words quantize to, not through shared lemmas."""
    if corpus.embedder.id.startswith("hash"):
        pytest.skip("needs the real term embedder")
    from assoc.witness import llm_witness
    from assoc.bench.conftest import scripted_witness
    doc = corpus.store.document(corpus.store.latest_id("chat-haifa-en"))
    lines = "[fact] (about: Artemy) (class: standing) (chunk: 1) (entities: Хайфа) Сестра Артемия живёт в Хайфе."
    facts, report = llm_witness(doc, get("chat"), scripted_witness(lines), codebook=corpus.build.codebook)
    assert facts and facts[0]["grounded"] and facts[0]["ground_via"] == "cell", facts
    assert report["grounded_via_cells"] == 1 and facts[0].get("language_drift") is True
    facts2, _ = llm_witness(doc, get("chat"), scripted_witness(lines), codebook=None)
    assert not facts2[0]["grounded"]          # lemmas alone cannot see it
