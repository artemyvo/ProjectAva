"""bench/support — the tech-support corpus (§9): exact hit + fast path on an error code,
platform-scoped procedures, versioned norms, a removed feature, an honest empty block,
tenant fencing, retrievability before extraction, concurrent pulls during a rebuild."""

import json
import threading

from assoc.library import Library
from assoc.selection import SUPPORT_POLICY
from assoc.bench import fixtures as fx


def test_error_code_fast_path_with_header_row(corpus):
    block = corpus.inject("E142", scope={"version": "2.0", "platform": "linux"}, policy=SUPPORT_POLICY)
    assert block.fast_path and block.reason == "fast_path"
    assert "Set retry_backoff (new in 2.0)" in block.text
    assert "| Code | Meaning | Action |" in block.text          # the row group carries its header
    assert "E142" in block.text
    v1 = corpus.inject("E142", scope={"version": "1.0", "platform": "linux"}, policy=SUPPORT_POLICY)
    assert v1.fast_path and "Raise the plan limit" in v1.text and "retry_backoff" not in v1.text


def test_platform_scope_never_offers_the_other_platform(corpus):
    hits = corpus.pull("register the service", scope={"version": "2.0", "platform": "windows"}, limit=30)
    keys = {h.meta.get("key") for h in hits}
    assert keys and all("/windows/" in k for k in keys if k and k.startswith("docs/"))
    top = next(h for h in hits if h.grain == "chunk")
    doc = corpus.store.document(top.doc_id)
    assert "Windows Service Manager" in doc.unit(top.chunk_id).text


def test_setting_changed_between_versions(corpus):
    v1 = corpus.pull("default connect_timeout", scope={"version": "1.0", "platform": "linux"})
    v2 = corpus.pull("default connect_timeout", scope={"version": "2.0", "platform": "linux"})
    t1 = corpus.store.document(v1[0].doc_id).unit(v1[0].chunk_id).text
    t2 = corpus.store.document(v2[0].doc_id).unit(v2[0].chunk_id).text
    assert "30 seconds" in t1 and "60 seconds" in t2
    # Unscoped on version: the newest version is current.
    cur = corpus.pull("default connect_timeout", scope={"platform": "linux"})
    assert "60 seconds" in corpus.store.document(cur[0].doc_id).unit(cur[0].chunk_id).text


def test_feature_removed_in_v2_is_a_fact(corpus):
    hits = corpus.pull("--insecure flag", scope={"version": "2.0", "platform": "linux"}, limit=20)
    texts = [(h.claim or {}).get("text", "") for h in hits if h.grain == "claim"]
    assert any(t.startswith("Removed in 2.0:") and "--insecure" in t for t in texts), texts
    assert any("Removed the `--insecure` flag" in t for t in texts)


def test_question_the_docs_do_not_answer_is_an_empty_block(corpus):
    def none_picker(system, user, **kw):
        return "PICKS:\nNONE\n"
    block = corpus.inject("Can the gateway run a Kubernetes operator?", scope={"version": "2.0", "platform": "linux"},
                          policy=SUPPORT_POLICY, generate_fn=none_picker, force_model=True)
    assert block.text == "" and block.reason == "picked_nothing" and block.catalogue_size > 0


def test_tenant_fencing(fresh_root):
    lib = Library(fresh_root)
    lib.ingest(fx.chat_kestrel(), "chat", {"key": "ticket-a", "date": "2026-08-14"}, scope={"tenant": "acme"})
    lib.ingest(fx.chat_noam(), "chat", {"key": "ticket-b", "date": "2026-07-02"}, scope={"tenant": "globex"})
    for p in fx.docs_tree("2.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.rebuild()
    a = lib.pull("Noam Keller", scope={"tenant": "acme"})
    assert a and all(h.meta.get("key") != "ticket-b" for h in a)
    b = lib.pull("Noam Keller", scope={"tenant": "globex"})
    assert any(h.meta.get("key") == "ticket-b" for h in b) and all(h.meta.get("key") != "ticket-a" for h in b)
    # Docs carry no tenant facet: visible to both.
    assert any((h.meta.get("key") or "").startswith("docs/") for h in lib.pull("connect_timeout", scope={"tenant": "acme"}))


def test_pages_retrievable_at_chunk_grain_before_extraction(corpus):
    doc = corpus.store.document(corpus.store.latest_id("docs/linux/config.md"))
    assert doc.status["status"] == "pending"       # the LLM witness never ran on the docs
    hits = corpus.pull("retry_backoff", scope={"version": "2.0", "platform": "linux"})
    assert hits[0].grain in ("chunk", "claim") and hits[0].meta["key"] == "docs/linux/config.md"


def test_concurrent_pulls_finish_on_one_build(fresh_root):
    lib = Library(fresh_root)
    for p in fx.docs_tree("2.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.rebuild()
    errors: list = []
    builds: list = []

    def puller():
        try:
            for _ in range(20):
                hits = lib.pull("connect_timeout", scope={"version": "2.0"})
                ids = {h.build_id for h in hits}
                assert len(ids) == 1
                builds.append(ids.pop())
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=puller) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(3):
        lib.ingest(fx.NEWS_ARTICLE, "news", {"key": f"news/{i}", "date": "2026-08-20"})
        lib.rebuild()
    for t in threads:
        t.join()
    assert not errors and builds
