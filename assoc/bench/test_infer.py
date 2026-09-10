"""§4 B2 — inferred edges, off by default: a scripted model asserts a connection between two
co-active nodes with no recorded path; the edge is written attributed, loaded only under the
knob at low weight, and can never be the sole path to an aha."""

from assoc import infer
from assoc.aha import _inferred_only
from assoc.graph import EdgeTable, S_TYPE
from assoc.spread import spread


def test_ask_parses_only_listed_predicates():
    assert infer.ask(lambda s, u, **kw: "founded(Noam Keller, Brightmem)", "Noam Keller", "Brightmem") == "founded"
    assert infer.ask(lambda s, u, **kw: "NONE", "a", "b") is None
    assert infer.ask(lambda s, u, **kw: "likes(a, b)", "a", "b") is None
    assert infer.ask(lambda s, u, **kw: "<think>hmm</think>\nacquired(Halden, Brightmem)", "Halden", "Brightmem") == "acquired"


def test_propose_writes_attributed_edges_and_knob_gates_loading(fresh_root):
    from assoc.library import Library
    from assoc.bench import fixtures as fx
    from assoc.bench.conftest import scripted_witness
    from assoc.witness import run_witnesses
    lib = Library(fresh_root)
    lib.ingest(fx.chat_noam(), "chat", {"key": "chat-noam", "date": "2026-07-02"})
    run_witnesses(lib.store, lib.store.latest_id("chat-noam"), generate_fn=scripted_witness(fx.CHAT_NOAM_LINES))
    lib.rebuild()
    assert not infer.has_path(lib.build.edges, "entity:noam keller", "entity:brightmem")
    written = infer.propose(lib, [("entity:noam keller", "entity:brightmem")], lambda s, u, **kw: "founded(noam keller, brightmem)")
    assert written and written[0]["source"] == "model" and written[0]["pred"] == "founded"
    lib.rebuild("full")
    assert not any(t.startswith("inferred:") for _o, t, _e in lib.build.edges.edges("entity:noam keller"))   # knob off
    lib.knobs = {"inferred_edges": True}
    lib.rebuild("full")
    edges = [t for _o, t, _e in lib.build.edges.edges("entity:noam keller")]
    assert "inferred:founded" in edges
    r = spread(lib.build.edges, {"entity:noam keller": 1.0}, k=1)
    assert r.r["entity:brightmem"] <= S_TYPE["inferred"] * 0.6 + 1e-9


def test_inferred_only_paths_never_carry_an_aha():
    assert _inferred_only([{"a": 0.1, "edges": [("a", "inferred:founded", "b", 0.2)]}])
    assert not _inferred_only([{"a": 0.1, "edges": [("a", "inferred:founded", "b", 0.2), ("b", "mentions", "c", 0.3)]}])
    assert not _inferred_only([])
