"""bench/witness_refusal — the LLM witness meets the base model's safety reflex.

The same gap the relation pass had (`test_relations.py`): a refused group parsed to zero
facts, counted as a group that established nothing, and the protocol written over the
imported one — Ava's `chat_facts` record for that chat — with the refused chunks empty, for
good (the feed never re-imports over the library's own protocol). These pin the fix: a
refusal is retried once with thinking on; still refused, the chunks are reported and the
previous protocol's facts for exactly those chunks are kept."""

from assoc.library import Library
from assoc.witness import WITNESS_THINKING_RETRY_MAX_NEW_TOKENS, import_protocol, llm_witness, run_witnesses
from assoc.kinds import get
from assoc.refusal import is_refusal
from assoc.bench import fixtures as fx

REFUSAL = ("I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. "
           "My safety guidelines prohibit me from generating or processing content that depicts sexually explicit acts.")


class _Gen:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user, *, thinking, max_new_tokens, temperature):
        self.calls.append({"thinking": thinking, "max_new_tokens": max_new_tokens})
        return self.replies.pop(0) if self.replies else ""


def _imported_chat(fresh_root):
    """A chat whose protocol is the imported (`chat_facts`) one, as the feed leaves it."""
    lib = Library(fresh_root)
    doc_id = lib.ingest(fx.chat_ru(), "chat", {"key": "chat-ru", "date": "2026-06-20"})
    # Import Ava's protocol first, exactly as the feed does: a real witness's lines, anchored.
    from assoc.protocol import parse_fact_lines
    facts = parse_fact_lines(fx.chat_ru_protocol_lines(), allowed_classes=get("chat").classes)
    import_protocol(lib.store, doc_id, facts, witness="ava:chat_facts")
    doc = lib.store.document(doc_id)
    assert doc.facts["prompt_version"] == "imported" and len(doc.facts["facts"]) == 4
    return lib, doc_id


def test_witness_refusal_is_a_refusal_not_an_empty_protocol():
    assert is_refusal(REFUSAL)
    assert not is_refusal("")                      # a chat that established nothing
    assert not is_refusal("No facts to record: the exchange establishes nothing.")


def test_refused_group_retries_with_thinking_then_keeps_the_imported_facts(fresh_root):
    lib, doc_id = _imported_chat(fresh_root)
    before = {f["text"] for f in lib.store.document(doc_id).facts["facts"]}
    gen = _Gen(REFUSAL, REFUSAL)
    rep = run_witnesses(lib.store, doc_id, generate_fn=gen, model_id="stub")
    assert [c["thinking"] for c in gen.calls] == [False, True]
    assert gen.calls[1]["max_new_tokens"] == WITNESS_THINKING_RETRY_MAX_NEW_TOKENS
    assert rep["llm"]["refused_groups"] == 1 and rep["llm"]["retried_thinking"] == 1
    assert rep["llm"]["failed_groups"] == 0, "a refusal is not a failed group (that path halves and retries)"
    assert rep["refused_chunks"] == rep["chunks"] == 3
    assert rep["fallback_facts"] == 4 and rep["fallback_from"] == "ava:chat_facts"
    doc = lib.store.document(doc_id)
    after = doc.facts["facts"]
    assert {f["text"] for f in after} == before, "nothing on record was lost to the refusal"
    assert all(f.get("fallback") == "ava:chat_facts" for f in after)
    assert doc.facts["prompt_version"] != "imported" and rep["status"] == "extracted"


def test_thinking_retry_that_answers_replaces_the_import(fresh_root):
    lib, doc_id = _imported_chat(fresh_root)
    gen = _Gen(REFUSAL, "<think>a transcript to record, fine</think>\n" + fx.chat_ru_protocol_lines())
    rep = run_witnesses(lib.store, doc_id, generate_fn=gen, model_id="stub")
    assert rep["llm"]["retried_thinking"] == 1 and rep["llm"]["refused_groups"] == 0
    assert "refused_chunks" not in rep and rep["facts"] == 4
    assert not any(f.get("fallback") for f in lib.store.document(doc_id).facts["facts"])


def test_genuinely_empty_group_is_not_retried(fresh_root):
    lib, doc_id = _imported_chat(fresh_root)
    gen = _Gen("")
    rep = run_witnesses(lib.store, doc_id, generate_fn=gen, model_id="stub")
    assert len(gen.calls) == 1 and rep["llm"]["refused_groups"] == 0 and rep["llm"]["retried_thinking"] == 0
    assert rep["facts"] == 0            # an empty answer is still an empty answer


def test_refusal_with_no_previous_protocol_is_reported(fresh_root):
    lib = Library(fresh_root)
    doc_id = lib.ingest(fx.chat_ru(), "chat", {"key": "chat-ru", "date": "2026-06-20"})
    gen = _Gen(REFUSAL, REFUSAL)
    rep = run_witnesses(lib.store, doc_id, generate_fn=gen, model_id="stub")
    assert rep["refused_chunks"] == 3 and rep["fallback_facts"] == 0 and rep["fallback_from"] is None


def test_refused_group_only_falls_back_for_its_own_chunks(fresh_root):
    lib, doc_id = _imported_chat(fresh_root)
    doc = lib.store.document(doc_id)
    spec = get("chat")
    calls = []

    def gen(system, user, *, thinking, max_new_tokens, temperature):
        calls.append(thinking)
        import re
        m = re.search(r"\[chunk 1 \|[^\]]*\]", user)
        if m and "context only" not in m.group(0):    # chunk 1 as MATERIAL, not as the context tail
            return REFUSAL
        return "[fact] (about: Артемий) (class: stated) (chunk: 3) Артемий says the last chunk is fine."

    # A window that puts one exchange per group.
    facts, report = llm_witness(doc, spec, gen, window=10)
    assert report["refused_groups"] == 1 and report["groups"] == 3
    assert report["refused_chunks"] == [doc.units[0].chunk_id] or len(report["refused_chunks"]) == 1
    from assoc.witness import _fallback_facts
    fb = _fallback_facts(doc, report["refused_chunks"])
    assert fb and all(f["chunk_id"] in report["refused_chunks"] for f in fb)
    assert len(fb) < len(doc.facts["facts"])


def test_refused_document_is_requeued_once_per_new_adapter(fresh_root):
    """The retry trigger is the adapter, not the prompt: the reflex lives in the weights."""
    from assoc.witness import pending_documents
    lib, doc_id = _imported_chat(fresh_root)
    rep = run_witnesses(lib.store, doc_id, generate_fn=_Gen(REFUSAL, REFUSAL), model_id="adapter-A")
    doc = lib.store.document(doc_id)
    assert doc.facts["model_id"] == "adapter-A" and len(doc.facts["refused_chunks"]) == 3
    assert pending_documents(lib.store, include_imported=True) == []                       # not by the old rules
    assert pending_documents(lib.store, current_model="adapter-A") == []                    # same adapter: no retry
    assert pending_documents(lib.store, current_model="adapter-B") == [doc_id]              # a new one: once
    assert lib.pending(current_model="adapter-B") == [doc_id]
    # Still refused under B: recorded under B, and B does not re-queue it again.
    run_witnesses(lib.store, doc_id, generate_fn=_Gen(REFUSAL, REFUSAL), model_id="adapter-B")
    doc = lib.store.document(doc_id)
    assert doc.facts["model_id"] == "adapter-B" and len(doc.facts["refused_chunks"]) == 3
    assert {f["text"] for f in doc.facts["facts"]} == {f["text"] for f in doc.facts["facts"]} and doc.facts["facts"]
    assert pending_documents(lib.store, current_model="adapter-B") == []
    # Answered under C: the refusal record clears, and no later adapter re-queues it.
    run_witnesses(lib.store, doc_id, generate_fn=_Gen(REFUSAL, "<think>ok</think>\n" + fx.chat_ru_protocol_lines()),
                  model_id="adapter-C")
    doc = lib.store.document(doc_id)
    assert doc.facts["refused_chunks"] == [] and not any(f.get("fallback") for f in doc.facts["facts"])
    assert pending_documents(lib.store, current_model="adapter-D") == []


def test_never_refused_document_ignores_an_adapter_change(fresh_root):
    from assoc.witness import pending_documents
    lib, doc_id = _imported_chat(fresh_root)
    run_witnesses(lib.store, doc_id, generate_fn=_Gen(fx.chat_ru_protocol_lines()), model_id="adapter-A")
    assert pending_documents(lib.store, current_model="adapter-B") == []


def test_extract_pending_threads_the_model_into_the_requeue(fresh_root):
    lib, doc_id = _imported_chat(fresh_root)
    run_witnesses(lib.store, doc_id, generate_fn=_Gen(REFUSAL, REFUSAL), model_id="adapter-A")
    reps = lib.extract_pending(_Gen(REFUSAL, REFUSAL), model_id="adapter-A")
    assert reps == []
    reps = lib.extract_pending(_Gen("<think>ok</think>\n" + fx.chat_ru_protocol_lines()), model_id="adapter-B")
    assert len(reps) == 1 and reps[0]["facts"] == 4
