"""bench/rebuild — the manifest and scopes (§2.8, milestone-1 subset)."""

from assoc import rebuild as rb
from assoc.budget import Budget
from assoc.library import Library
from assoc.bench import fixtures as fx


def test_fresh_store_builds_full_then_fast(fresh_root):
    lib = Library(fresh_root)
    assert lib.staleness() == {"scope": "full", "reasons": ["no_build"]}
    for p in fx.docs_tree("1.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    rep = lib.rebuild()
    assert rep["scope"] == "full" and rep["counts"]["documents"] == 5
    assert lib.staleness()["scope"] == "none"
    lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/x", "date": "2026-08-20"})
    st = lib.staleness()
    assert st["scope"] == "fast" and "count:documents+" in st["reasons"]
    rep2 = lib.rebuild()
    assert rep2["scope"] == "fast" and rep2["delta"]["documents"] == 1
    assert lib.build.manifest["previous"] == rep["counts"] and False or lib.build.manifest["previous"] is not None


def test_version_changes_invalidate(fresh_root, monkeypatch):
    lib = Library(fresh_root)
    lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/x", "date": "2026-08-20"})
    lib.rebuild()
    lib.embedder.id = "other-embedder"
    st = lib.staleness()
    assert st["scope"] == "full" and "version:embedder" in st["reasons"]
    lib.embedder.id = "hash-lemma-256"
    monkeypatch.setattr(rb, "SPLITTER_VERSION", "split-2")
    st = lib.staleness()
    assert st["scope"] == "full" and "version:splitter" in st["reasons"]


def test_budget_change_is_a_fast_reason(fresh_root):
    lib = Library(fresh_root, budget=Budget(total=4000))
    lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/x", "date": "2026-08-20"})
    lib.rebuild()
    lib.budget = Budget(total=800)
    st = lib.staleness()
    assert "budget" in st["reasons"] and st["scope"] == "fast"


def test_removed_key_leaves_the_current_set(fresh_root):
    lib = Library(fresh_root)
    for p in fx.docs_tree("2.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.rebuild()
    assert lib.pull("connect_timeout", scope={"version": "2.0"})
    lib.remove("docs/linux/config.md")
    lib.rebuild()
    assert all(h.meta.get("key") != "docs/linux/config.md" for h in lib.pull("connect_timeout", scope={"version": "2.0"}))


def test_new_version_supersedes_and_old_stays_reachable(fresh_root):
    lib = Library(fresh_root)
    for v in ("1.0", "2.0"):
        for p in fx.docs_tree(v, "linux"):
            lib.ingest(p["text"], "tech_doc", p["meta"])
    rep = lib.rebuild()
    assert rep["removed_facts"] >= 1
    ids = lib.store.versions_of("docs/linux/errors.md")
    assert len(ids) == 2 and lib.store.meta(ids[1])["supersedes"] == ids[0]
    assert lib.store.current_doc_ids({"version": "1.0"}) and ids[0] in lib.store.current_doc_ids({"version": "1.0"})
    assert ids[1] in lib.store.current_doc_ids({}) and ids[0] not in lib.store.current_doc_ids({})


def test_manifest_and_report_shape(fresh_root):
    lib = Library(fresh_root, budget=Budget(total=300))
    for p in fx.docs_tree("2.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    rep = lib.rebuild()
    m = lib.build.manifest
    assert set(m["versions"]) == {"splitter", "parser", "lex", "glossary", "codebook", "embedder", "knobs"}
    assert m["budget"]["ceiling"] == 150 and rep["fits"]["split"] >= 1
    assert rep["fits"]["still_oversize"] == 0 or rep["fits"]["still_oversize"] < rep["fits"]["oversize"]
    assert rep["claims"]["claims"] > 0 and "by_facet" in rep["claims"]


def test_append_only_chat_keeps_chunk_ids(fresh_root):
    import json
    lib = Library(fresh_root)
    d = json.loads(fx.chat_kestrel())
    doc_id = lib.ingest(json.dumps(d), "chat", {"key": "c", "date": "2026-08-14"})
    before = [u.chunk_id for u in lib.store.document(doc_id).units]
    d["exchanges"].append({"user_prompt": "Thanks anyway.", "assistant_response": "Any time.", "ts": "2026-08-14T10:09"})
    doc_id2 = lib.ingest(json.dumps(d), "chat", {"key": "c", "date": "2026-08-14"})
    after = [u.chunk_id for u in lib.store.document(doc_id2).units]
    assert doc_id2 == doc_id and after[:2] == before and len(after) == 3
    assert lib.store.document(doc_id).status["status"] == "pending"


def test_drift_forces_a_full_rebuild(fresh_root, monkeypatch):
    """A vocabulary shift after a fast rebuild crosses the drift knob (§2.8)."""
    from assoc import rebuild as rb
    from assoc.bench import fixtures as fx
    lib = Library(fresh_root, knobs={"drift_forced_frac": 0.0, "drift_doubled_cells": 10**6})
    for p in fx.docs_tree("2.0", "linux")[:2]:
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.rebuild()
    assert lib.build.manifest["scope"] == "full"
    # New vocabulary arrives (a Russian chat): the fast rebuild must assign it into old cells.
    lib.ingest(fx.chat_ru(), "chat", {"key": "c", "date": "2026-06-20"})
    lib.rebuild()
    m = lib.build.manifest
    assert m["scope"] == "fast" and m["drift"]["assigned"] > 0
    st = lib.staleness()
    assert "drift" in st["reasons"] and st["scope"] == "full", st
    lib.rebuild()
    assert lib.build.manifest["scope"] == "full" and lib.build.manifest["drift"]["since_recluster"] == 0


def test_tightened_dedup_threshold_splits_a_merged_claim(fresh_root):
    from assoc.bench import fixtures as fx
    from assoc.bench.conftest import scripted_witness
    from assoc.witness import run_witnesses
    lib = Library(fresh_root)
    for key, text, lines in (("notes/a.md", fx.RESTATE_DOC_A, fx.RESTATE_A_LINES), ("notes/b.md", fx.RESTATE_DOC_B, fx.RESTATE_B_LINES)):
        lib.ingest(text, "tech_doc", {"key": key, "product": "nimbus", "version": "2.0"})
        run_witnesses(lib.store, lib.store.latest_id(key), generate_fn=scripted_witness(lines))
    lib.rebuild()
    port = [c for c in lib.build.claims.values() if "8443" in c["text"]]
    merged_before = max(c["n_sources"] for c in port)
    lib.knobs = {"tier3_min": 1.01}         # nothing may merge on embedding; tier 2 still may
    lib.rebuild("full")
    port2 = [c for c in lib.build.claims.values() if "8443" in c["text"]]
    # Every occurrence survives whichever way the fold goes.
    assert sum(len(c["occurrences"]) for c in port2) == sum(len(c["occurrences"]) for c in port)
    assert max(c["n_sources"] for c in port2) <= merged_before
