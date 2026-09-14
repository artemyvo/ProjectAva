"""bench/spread + bench/hub (§9): a hand-drawn graph with known edge types; activation per
node from the §2.7 formula with the default knobs; convergence sums; a `rel:` edge out of a
hub carries while each of its `mentions` carries φ(200); no node twice on one path; every
hop multiplier < 1; `_self` reached but not expanded through on co-mention; a 200-claim hub
not returned whole by a 2-hop spread."""

import math

from assoc.graph import EdgeTable, S_TYPE
from assoc.spread import ALPHA, GAMMA, TAU, activation, spread
from assoc.activation import cold_baseline


def _graph():
    t = EdgeTable()
    # A small world: two persons, a company, claims, one hub.
    t.add("claim:c1", "person:p", "about", "c1")
    t.add("claim:c1", "entity:kestrel", "mentions", "c1")
    t.add("person:p", "entity:brightmem", "rel:interviewed_at", "c2")
    t.add("entity:brightmem", "person:noam", "rel:founded", "c3")
    t.add("claim:r", "person:noam", "mentions", "r")
    t.add("claim:r", "person:artemy", "about", "r")
    t.add("claim:r", "chunk:k1", "in_chunk", "r")
    t.add("chunk:k1", "chunk:k0", "adjacent", "d")
    t.add("chunk:k1", "doc:d", "in_doc", "d")
    for i in range(200):                                   # the hub: 200 claims mention noam
        t.add(f"claim:h{i}", "person:noam", "mentions", f"h{i}")
    t.add("person:noam", "person:_self", "rel:knows", "x")
    for i in range(50):
        t.add("person:_self", f"claim:s{i}", "about", f"s{i}")
    return t


def test_one_hop_arithmetic_and_fan():
    t = _graph()
    r = spread(t, {"person:p": 1.0}, k=1)
    # f has ONE rel edge and ONE about edge: fan per type = 1 → φ = 1.
    assert math.isclose(r.r["entity:brightmem"], 1.0 * S_TYPE["rel"] * GAMMA * 1.0, rel_tol=1e-9)
    assert math.isclose(r.r["claim:c1"], 1.0 * S_TYPE["about"] * GAMMA, rel_tol=1e-9)
    # Every hop multiplier is < 1 (correction m1): nothing amplifies.
    for node, paths in r.paths.items():
        for p in paths:
            for _f, _t, _to, s in p["edges"]:
                assert s < 1.0


def test_rel_out_of_hub_carries_while_mentions_are_crushed():
    t = _graph()
    r = spread(t, {"person:noam": 1.0}, k=1)
    # 201 mention edges (200 hub claims + r) → each mention arrival ≈ s·γ·201^−½
    m = S_TYPE["mentions"] * GAMMA * 201 ** (-ALPHA)
    assert math.isclose(r.r["claim:h7"], m, rel_tol=1e-6)
    # one rel:founded edge and one rel:knows edge are separate types → fan 1 each
    assert math.isclose(r.r["entity:brightmem"], S_TYPE["rel"] * GAMMA, rel_tol=1e-9)
    assert r.r["entity:brightmem"] / r.r["claim:h7"] > 10


def test_two_paths_sum_and_no_echo():
    t = _graph()
    r = spread(t, {"person:p": 1.0}, k=3)
    # person:p → brightmem → noam → claim:r reaches r; claim:c1 → kestrel is a dead end.
    assert "claim:r" in r.r
    # Echo: f → c1 → f must not add to f's own activation (rule 4): f's r stays its seed.
    assert math.isclose(r.r["person:p"], 1.0, rel_tol=1e-9)
    # Convergence: seed both ends, the node reached from both sums above either alone.
    a = spread(t, {"person:p": 0.5, "person:artemy": 0.5}, k=3)
    b = spread(t, {"person:p": 1.0}, k=3)
    c = spread(t, {"person:artemy": 1.0}, k=3)
    assert a.r["claim:r"] > 0.5 * b.r["claim:r"] and a.r["claim:r"] > 0.5 * c.r["claim:r"]


def test_self_reached_not_expanded_through_on_comention():
    t = _graph()
    r = spread(t, {"person:noam": 1.0}, k=2)
    assert "person:_self" in r.r                       # reached along rel:knows
    assert not any(n.startswith("claim:s") for n in r.r)   # its about-claims never reached through it


def test_hub_not_returned_whole():
    t = _graph()
    r = spread(t, {"person:noam": 1.0}, k=2, beam=50)
    hub = [n for n in r.r if n.startswith("claim:h")]
    assert len(hub) <= 50
    # and what is reached carries the fan-crushed share
    assert all(v < 0.05 for n, v in r.r.items() if n.startswith("claim:h"))


def test_activation_scale_reads_in_hours():
    b0 = cold_baseline()
    # Fully reached (r = 1) with no accesses: κ·1 added to e^{B₀} → about one access one hour ago.
    a = activation(1.0, math.exp(b0))
    assert a > TAU and abs(a - math.log(math.exp(b0) + 1.0)) < 1e-9
    # Not reached, no accesses: the cold baseline itself, below τ.
    assert activation(0.0, math.exp(b0)) == b0 < TAU
    # Authority is bounded and outside the log.
    assert activation(0.0, math.exp(b0), auth=1.0) - b0 < 0.3 * math.log(2) + 1e-9


def test_paths_read_back():
    t = _graph()
    r = spread(t, {"person:p": 1.0}, k=3)
    from assoc.spread import render_path
    txt = render_path(r.paths["claim:r"][0])
    assert "person:p —rel:interviewed_at" in txt and "entity:brightmem —rel:founded" in txt and "→ claim:r" in txt


def test_learned_strength_bounded_and_ordered():
    """§2.7 v2 knob: an entity co-mentioned by many claims about one subject gets a stronger
    `mentions` edge than one co-mentioned once; multipliers stay in [0.5, 1.5]; no hop amplifies."""
    t = EdgeTable()
    for i in range(6):
        t.add(f"claim:a{i}", "person:x", "about", f"a{i}")
        t.add(f"claim:a{i}", "entity:often", "mentions", f"a{i}")
    t.add("claim:a0", "entity:once", "mentions", "a0")
    rep = t.learn_strength()
    assert rep["weighted_edges"] == 7
    assert t.w("claim:a0", "entity:often", "mentions") > t.w("claim:a0", "entity:once", "mentions")
    assert 0.5 <= t.w("claim:a0", "entity:once", "mentions") <= 1.5
    r = spread(t, {"claim:a0": 1.0}, k=1)
    assert r.r["entity:often"] > r.r["entity:once"]
    for paths in r.paths.values():
        for p in paths:
            for _f, _t, _to, s in p["edges"]:
                assert s < 1.0
