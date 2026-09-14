"""bench/relations — the typed-relation pass (§2.3): what a refusal does to the cache.

The pass runs thinking-off with a bare labelling prompt, and on a transcript about sex or
violence the base model's safety reflex answers instead ("I am programmed to be a helpful
and harmless AI assistant …", the training box, 2026-09-11 23:46 UTC). That parsed to zero lines and
was cached as "no relation" for every claim of the batch, permanently. These pin the fix:
a refusal is retried once with thinking on, and if it still refuses NOTHING is cached."""

from assoc import relations as R
from assoc.relations import RelationCache, run_relation_pass, is_refusal

REFUSAL = ("I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. "
           "My safety guidelines prohibit me from generating or processing content that depicts sexually explicit acts.")
EXPLAINED_NONE = ("I cannot fulfill this request because none of the provided claims contain both a subject "
                  "and an object from their respective brackets that fit the specified predicates.")


def _claims(n=3):
    return [{"claim_id": f"c{i}", "text": f"artemyvo asked the assistant about thing {i}.",
             "subject": "person:artemyvo", "subject_raw": "artemyvo", "entities": ["the assistant"],
             "occurrences": [{"doc_id": "d1"}]} for i in range(1, n + 1)]


class _Gen:
    """A scripted generate_fn: one reply per call, records the thinking flag of each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user, *, thinking, max_new_tokens, temperature):
        self.calls.append({"thinking": thinking, "max_new_tokens": max_new_tokens})
        return self.replies.pop(0) if self.replies else ""


def test_is_refusal_tells_a_refusal_from_an_explained_none():
    assert is_refusal(REFUSAL)
    assert not is_refusal(EXPLAINED_NONE)          # verbose "none": talks about the claims
    assert not is_refusal("No relations found.")
    assert not is_refusal("")
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("Я не могу помочь с этим запросом.")
    assert is_refusal("RELATIONS:\n" + REFUSAL)     # the prompt's echo is stripped first
    assert not is_refusal("1: asked_about(artemyvo, the assistant)")


def test_refused_batch_is_retried_with_thinking_and_left_uncached(tmp_path):
    claims = _claims()
    gen = _Gen(REFUSAL, REFUSAL)
    cache = RelationCache(tmp_path / "rc.json")
    rep = run_relation_pass({"d1": claims}, {"d1": "a chat"}, gen, cache)
    assert [c["thinking"] for c in gen.calls] == [False, True]
    assert gen.calls[1]["max_new_tokens"] == R.THINKING_RETRY_MAX_NEW_TOKENS
    assert rep["refused"] == 1 and rep["refused_claims"] == 3 and rep["retried_thinking"] == 1
    assert rep["accepted"] == 0
    for c in claims:
        assert cache.get(c) is None, "a refused claim must stay uncached"
        assert not c.get("rel")
    # A fresh pass sees them again — nothing was written to disk either.
    cache2 = RelationCache(tmp_path / "rc.json")
    assert all(cache2.get(c) is None for c in claims)


def test_thinking_retry_that_answers_is_cached(tmp_path):
    claims = _claims()
    gen = _Gen(REFUSAL, "<think>a labelling task, fine</think>\n1: asked_about(artemyvo, the assistant)\n")
    cache = RelationCache(tmp_path / "rc.json")
    rep = run_relation_pass({"d1": claims}, {"d1": "a chat"}, gen, cache)
    assert rep["retried_thinking"] == 1 and rep["refused"] == 0 and rep["accepted"] == 1
    assert claims[0]["rel"] == ["asked_about", "person:artemyvo", "the assistant"]
    assert cache.get(claims[0]) == ["asked_about", "person:artemyvo", "the assistant"]
    assert cache.get(claims[1]) == [] and cache.get(claims[2]) == []   # a real "none" IS cached


def test_explained_none_is_cached_without_a_retry(tmp_path):
    claims = _claims()
    gen = _Gen(EXPLAINED_NONE)
    cache = RelationCache(tmp_path / "rc.json")
    rep = run_relation_pass({"d1": claims}, {"d1": "a chat"}, gen, cache)
    assert len(gen.calls) == 1 and rep["refused"] == 0 and rep["retried_thinking"] == 0
    assert all(cache.get(c) == [] for c in claims)


def test_retry_that_dies_leaves_the_batch_uncached(tmp_path):
    claims = _claims()

    class Dies(_Gen):
        def __call__(self, *a, **k):
            if k.get("thinking"):
                raise RuntimeError("boom")
            return super().__call__(*a, **k)

    gen = Dies(REFUSAL)
    cache = RelationCache(tmp_path / "rc.json")
    rep = run_relation_pass({"d1": claims}, {"d1": "a chat"}, gen, cache)
    assert rep["refused"] == 1 and rep["failed"] == 0
    assert all(cache.get(c) is None for c in claims)
