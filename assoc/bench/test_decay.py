"""bench/decay + bench/priming (§9): the clock injected; the formula's actual behaviour
(clustered accesses read a week later are warmer than spread ones — correction m2); a node
with no access reads exactly B₀; a need dormant after 90 days then back at full floor; bulk
ingest does not warm a thousand pages yet leaves them retrievable; the context layer over
the global one; priming — two cues an hour apart."""

import math

from assoc.activation import (Activation, COLD_DAYS, D_DECAY, NEED_FLOOR_HOURS, base_level_from, cold_baseline,
                              need_floor_odds)


def test_clustered_vs_spread_accesses_read_a_week_later(tmp_path):
    act = Activation(tmp_path / "a.db")
    now = 10_000.0
    week = 7 * 24.0
    # ten accesses in the hour before t0, read a week after t0
    for i in range(10):
        act.touch(["clustered"], now_h=now, at_h=now - week - i / 10.0)
    # ten accesses spread over the month before t0
    for i in range(10):
        act.touch(["spread"], now_h=now, at_h=now - week - i * 3 * 24.0)
    bc, bs = act.base_level("clustered", now_h=now), act.base_level("spread", now_h=now)
    assert bc > bs
    assert abs(bc - (-0.26)) < 0.1 and abs(bs - (-0.74)) < 0.15      # the fixture pins the formula


def test_cold_baseline_and_no_access(tmp_path):
    act = Activation(tmp_path / "a.db")
    assert act.base_level("never", now_h=5000.0) == cold_baseline()
    assert math.isclose(cold_baseline(), math.log((COLD_DAYS * 24) ** (-D_DECAY)))


def test_tail_approximation_keeps_cost_flat(tmp_path):
    act = Activation(tmp_path / "a.db")
    now = 20_000.0
    for i in range(500):
        act.touch(["busy"], now_h=now, at_h=now - i * 2.0)
    exact = base_level_from([i * 2.0 for i in range(500)])
    approx = act.base_level("busy", now_h=now)
    assert abs(exact - approx) < 0.15 and act.accesses("busy") == 500


def test_need_floor_fades_then_resets(tmp_path):
    now = 30_000.0
    fresh = need_floor_odds(now, now)
    old = need_floor_odds(now, now - 90 * 24.0)
    assert fresh == NEED_FLOOR_HOURS ** (-D_DECAY) and old < fresh / 7
    act = Activation(tmp_path / "a.db")
    act.touch(["need:x"], now_h=now)
    assert act.odds("need:x", now_h=now) > fresh          # a fresh touch beats the floor


def test_context_layer_over_global(tmp_path):
    act = Activation(tmp_path / "a.db")
    now = 100.0
    act.touch(["claim:a"], now_h=now - 1.0, scope={"tenant": "acme"})
    a_acme = act.odds("claim:a", now_h=now, scope={"tenant": "acme"})
    a_globex = act.odds("claim:a", now_h=now, scope={"tenant": "globex"})
    a_global = act.odds("claim:a", now_h=now)
    assert a_acme > a_globex and a_globex < a_global      # globex sees only the discounted global layer
    assert math.isclose(a_globex, max(0.5 * 1.0, math.exp(cold_baseline())))


def test_bulk_ingest_cold_but_retrievable(corpus):
    """Creation accesses landed at rebuild: a claim from a dated document is warm from its
    own date, not from the moment of ingest, and a node never touched still reads B₀."""
    b = corpus.build
    cid = next(c for c in b.claims.values() if c["kind"] == "news")["claim_id"]
    assert corpus.activation.accesses(f"claim:{cid}") >= 1
    now = corpus.now_fn()
    assert corpus.activation.base_level(f"claim:{cid}", now_h=now) > cold_baseline()
    assert corpus.activation.base_level("claim:nonexistent", now_h=now) == cold_baseline()


def test_priming_two_cues_an_hour_apart(corpus):
    """A pull an hour after a related turn ranks that turn's neighbourhood above a cold pull."""
    now = corpus.now_fn()
    cold = corpus.pull("what did the sister do?", tier=3, now=now)
    corpus.touch(["chat-haifa-en"], now=now - 1.0)    # the app touched that conversation's chunks an hour ago
    doc = corpus.store.document(corpus.store.latest_id("chat-haifa-en"))
    corpus.touch([u.chunk_id for u in doc.units], now=now - 1.0)
    warm = corpus.pull("what did the sister do?", tier=3, now=now)

    def rank_of(hits):
        return next((h.rank for h in hits if h.meta.get("key") == "chat-haifa-en"), 99)
    assert rank_of(warm) <= rank_of(cold)
    assert any(h.activation is not None for h in warm)


def test_activation_dependent_decay_gives_the_spacing_effect(tmp_path):
    """§8: with Pavlik & Anderson's d_k, ten accesses spread over a month hold MORE a week
    later than ten clustered in one hour — the effect the fixed-d formula lacks."""
    act = Activation(tmp_path / "v.db", variable_d=True)
    now = 10_000.0
    week = 7 * 24.0
    for i in range(10):
        act.touch(["clustered"], now_h=now - week - 1.0 + i / 10.0, at_h=now - week - 1.0 + i / 10.0)
    for i in range(10):
        t = now - week - 30 * 24.0 + i * 3 * 24.0
        act.touch(["spread"], now_h=t, at_h=t)
    bc, bs = act.base_level("clustered", now_h=now), act.base_level("spread", now_h=now)
    assert bs > bc, (bs, bc)
    fixed = Activation(tmp_path / "f.db")
    for i in range(10):
        fixed.touch(["clustered"], now_h=now, at_h=now - week - 1.0 + i / 10.0)
        fixed.touch(["spread"], now_h=now, at_h=now - week - 30 * 24.0 + i * 3 * 24.0)
    assert fixed.base_level("clustered", now_h=now) > fixed.base_level("spread", now_h=now)
