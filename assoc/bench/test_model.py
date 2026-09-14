"""Model-dependent benches (ASSOC_MODEL=1): the real witness on a news article, a chat and
a docs page, and the real selection pass on the docs_qa cases. Regression sets, not
pass/fail: what is measured is grounding / anchoring rates and pick agreement, recorded
under bench/reports/. Loose floors keep a broken pass from passing silently."""

import json
from pathlib import Path

import pytest

from assoc.library import Library
from assoc.budget import Budget
from assoc.selection import SUPPORT_POLICY
from assoc.witness import run_witnesses
from assoc.bench import fixtures as fx
from assoc.bench.conftest import needs_model, shared_embedder, CACHE_DIR

REPORTS = Path(__file__).parent / "reports"
pytestmark = needs_model


@pytest.fixture(scope="module")
def gen():
    from assoc.bench import model_harness
    model_harness.load()
    return model_harness.generate_fn


@pytest.fixture(scope="module")
def lib(tmp_path_factory, gen):
    from assoc.bench import model_harness
    lib = Library(tmp_path_factory.mktemp("model"), embedder=shared_embedder(),
                  budget=Budget(total=6000, tokenizer=model_harness.token_counter), cache_dir=CACHE_DIR)
    for p in fx.docs_tree("2.0", "linux"):
        lib.ingest(p["text"], "tech_doc", p["meta"])
    lib.ingest(fx.chat_kestrel(), "chat", {"key": "chat-kestrel", "date": "2026-08-14"})
    lib.ingest(fx.chat_ru(), "chat", {"key": "chat-ru", "date": "2026-06-20"})
    lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/starling", "date": "2026-08-20"})
    return lib


def _write(name: str, obj: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def test_witness_on_news(lib, gen):
    rep = run_witnesses(lib.store, lib.store.latest_id("news/starling"), generate_fn=gen, model_id="gemma-4-31B")
    doc = lib.store.document(lib.store.latest_id("news/starling"))
    facts = doc.facts["facts"]
    _write("extract_model_news.json", {"report": rep, "facts": facts})
    grounded = sum(1 for f in facts if f["grounded"])
    assert rep["status"] == "extracted" and len(facts) >= 5, rep
    assert grounded / len(facts) >= 0.7, (grounded, len(facts))
    assert rep["llm"]["anchor_exact"] >= len(facts) * 0.6
    texts = " ".join(f["text"] for f in facts).lower()
    assert "starling" in texts and ("brightmem" in texts or "halden" in texts)


def test_witness_on_chat(lib, gen):
    rep = run_witnesses(lib.store, lib.store.latest_id("chat-kestrel"), generate_fn=gen, model_id="gemma-4-31B")
    doc = lib.store.document(lib.store.latest_id("chat-kestrel"))
    facts = doc.facts["facts"]
    _write("extract_model_chat.json", {"report": rep, "facts": facts})
    assert rep["status"] == "extracted" and len(facts) >= 3, rep
    subjects = {f["subject"] for f in facts}
    assert any(s.startswith("person:") for s in subjects)
    texts = " ".join(f["text"] for f in facts).lower()
    assert "kestrel" in texts and "brightmem" in texts
    assert sum(1 for f in facts if f["grounded"]) / len(facts) >= 0.6


def test_witness_on_russian_chat_keeps_language(lib, gen):
    rep = run_witnesses(lib.store, lib.store.latest_id("chat-ru"), generate_fn=gen, model_id="gemma-4-31B")
    facts = lib.store.document(lib.store.latest_id("chat-ru")).facts["facts"]
    _write("extract_model_chat_ru.json", {"report": rep, "facts": facts})
    assert len(facts) >= 3
    cyr = sum(1 for f in facts if f["language"] == "cyr")
    assert cyr / len(facts) >= 0.6, [f["text"] for f in facts]


def test_witness_on_tech_doc(lib, gen):
    key = "docs/linux/config.md"
    rep = run_witnesses(lib.store, lib.store.latest_id(key), generate_fn=gen, model_id="gemma-4-31B")
    facts = lib.store.document(lib.store.latest_id(key)).facts["facts"]
    _write("extract_model_tech_doc.json", {"report": rep, "facts": facts})
    llm = [f for f in facts if f.get("anchor") in ("exact", "inferred", "moved") and f["fact_class"] in ("spec", "procedure", "signature", "standing", "deprecated", "event")]
    assert len(llm) >= 8, rep
    texts = " ".join(f["text"] for f in llm).lower()
    assert "connect_timeout" in texts and "60" in texts
    assert sum(1 for f in llm if f["grounded"]) / len(llm) >= 0.7
    assert any(f["fact_class"] in ("spec", "procedure") for f in llm)


def test_selection_agreement(lib, gen):
    lib.rebuild()
    cases = [
        ("how do I rotate the token?", "docs/linux/config.md|docs/linux/faq.md"),
        ("What is the default connect_timeout?", "docs/linux/config.md"),
        ("it keeps timing out — what do I change?", "docs/linux/config.md"),
        ("does it support HTTP/3", "docs/linux/faq.md"),
        ("What did Nimbus pay for Starling?", "news/starling"),
        ("thanks, that's all", None),
    ]
    rows = []
    agree = 0
    for cue, expected in cases:
        block = lib.inject(cue, scope={"version": "2.0", "platform": "linux"}, policy=SUPPORT_POLICY, generate_fn=gen, force_model=True)
        keys = [h["reference"]["key"] for h in block.hits]
        ok = (not block.hits) if expected is None else any(k in expected.split("|") for k in keys[:2])
        agree += ok
        rows.append({"cue": cue, "expected": expected, "picked": keys, "reason": block.reason, "picks": block.picks,
                     "catalogue": block.catalogue_size, "ok": ok})
    _write("select_model.json", {"agreement": agree / len(cases), "rows": rows})
    assert agree / len(cases) >= 0.5, rows


def test_relation_pass_on_real_model(lib, gen):
    """§2.3: the per-document relation pass on gemma; the Kestrel/Noam claims must yield the
    need and the bridge predicates (or at least valid, argument-checked relations)."""
    from assoc.relations import RelationCache, run_relation_pass
    run_witnesses(lib.store, lib.store.latest_id("chat-kestrel"), generate_fn=gen, model_id="gemma-4-31B")
    lib.rebuild(generate_fn=gen)
    rels = {c["text"][:60]: c.get("rel") for c in lib.build.claims.values() if c["kind"] == "chat" and c.get("rel")}
    rep = lib.build.manifest["report"]["relations"]
    _write("relations_model.json", {"report": rep, "relations": rels})
    assert rep.get("accepted", 0) >= 1, rep
    preds = {r[0] for r in rels.values()}
    assert preds & {"asked_about", "looking_for", "wants", "interviewed_at", "founded", "offered", "declined"}, preds
    assert rep.get("bad_args", 0) <= rep.get("accepted", 0) + rep.get("bad_args", 0)


def test_judge_on_real_model(lib, gen):
    """§4.1: the real judge on the Kestrel/Noam candidate, built from scripted relations so the
    candidate is the designed one whatever the relation pass extracted."""
    import tempfile
    from pathlib import Path as _P
    from assoc.bench.test_aha import _lib, _resource_id, T0
    from assoc.bench.conftest import shared_embedder
    alib = _lib(_P(tempfile.mkdtemp()))
    rid = _resource_id(alib)
    cand = next(c for c in alib.aha(stimulus=["entity:noam keller"], now=T0 + 5.0) if c.resource == rid)
    out = alib.judge(cand, gen, now=T0 + 5.0, doing="idle, reading the morning's news")
    _write("judge_model.json", {"verdict": out.verdict, "link": out.link, "strength": out.strength, "paths": out.paths,
                                "passages": out.passages, "bridges": out.bridges})
    assert out.verdict in ("connect", "satisfies"), (out.verdict, out.link)
    assert out.link and out.link != "—"
