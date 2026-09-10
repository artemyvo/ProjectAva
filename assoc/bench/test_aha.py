"""bench/aha + bench/judge (§9): the Kestrel/Noam story as a graph. At the call (stimulus = the
Kestrel conversation) no candidate; hours later, with Noam as the stimulus, the candidate
(N, cR) carries both paths; the judge (scripted) says `connect`; the ledger blocks a
re-raise; `satisfies` closes the need; an unparseable verdict is `error` and never recorded;
a pair whose need-side route is the cell route alone stays under the floor."""

import json

import pytest

from assoc.aha import parse_verdict
from assoc.budget import Budget
from assoc.library import Library
from assoc.witness import run_witnesses
from assoc.bench import fixtures as fx
from assoc.bench.conftest import CACHE_DIR, scripted_witness, shared_embedder

T0 = 400_000.0     # hours


def _lib(tmp_path, *, bridges=True):
    lib = Library(tmp_path / "aha", embedder=shared_embedder(), budget=Budget(total=4000), cache_dir=CACHE_DIR, tier=3,
                  now_fn=lambda: T0)
    lib.ingest(fx.chat_noam(), "chat", {"key": "chat-noam", "date": "2026-07-02"})
    run_witnesses(lib.store, lib.store.latest_id("chat-noam"), generate_fn=scripted_witness(fx.CHAT_NOAM_LINES))
    kestrel_lines = fx.CHAT_KESTREL_LINES if bridges else "\n".join(l for l in fx.CHAT_KESTREL_LINES.splitlines() if "Brightmem" not in l)
    lib.ingest(fx.chat_kestrel(), "chat", {"key": "chat-kestrel", "date": "2026-08-14"})
    run_witnesses(lib.store, lib.store.latest_id("chat-kestrel"), generate_fn=scripted_witness(kestrel_lines))
    if bridges:
        lib.ingest(fx.NEWS_ARTICLE, "news", {"key": "news/starling", "date": "2026-08-20"})
        run_witnesses(lib.store, lib.store.latest_id("news/starling"), generate_fn=scripted_witness(fx.NEWS_PROTOCOL_LINES_REL))
    lib.rebuild()
    return lib


def _resource_id(lib):
    return next(c["claim_id"] for c in lib.build.claims.values() if "offered Artemy a CTO position" in c["text"])


def test_needs_and_edges_from_relations(tmp_path):
    lib = _lib(tmp_path)
    needs = lib.needs()
    preds = {n["predicate"] for n in needs}
    assert {"asked_about", "looking_for"} <= preds
    st = lib.build.edges.stats()
    assert st["needs"] >= 2 and "rel:interviewed_at" in st["edges_by_type"] and "rel:founded" in st["edges_by_type"]


def test_no_candidate_at_the_call(tmp_path):
    lib = _lib(tmp_path)
    kestrel = lib.store.document(lib.store.latest_id("chat-kestrel"))
    stim = [u.chunk_id for u in kestrel.units] + ["entity:kestrel"]
    cands = lib.aha(stimulus=stim, now=T0)
    assert all(c.resource != _resource_id(lib) for c in cands), [(c.resource, c.strength) for c in cands]


def test_candidate_hours_later_with_both_paths(tmp_path):
    lib = _lib(tmp_path)
    cands = lib.aha(stimulus=["entity:noam keller"], now=T0 + 5.0)
    rid = _resource_id(lib)
    hit = next((c for c in cands if c.resource == rid), None)
    assert hit is not None, [(c.resource, round(c.strength, 4)) for c in cands]
    assert hit.need_side > 0 and hit.stimulus_side > 0 and hit.strength == min(hit.need_side, hit.stimulus_side)
    assert any("rel:interviewed_at" in p or "rel:founded" in p for p in hit.paths["need"]), hit.paths
    assert hit.paths["stimulus"] and "person:artemy" not in hit.paths["stimulus"][0].split("→")[0]
    assert "CTO position" in hit.passages["resource"] and "hiring" in hit.passages["need"].lower() or "Kestrel" in hit.passages["need"]
    assert any("brightmem" in n for n, _ in hit.bridges)


def test_without_bridges_the_need_side_is_weaker(tmp_path):
    with_b = _lib(tmp_path / "b")
    without = _lib(tmp_path / "nb", bridges=False)
    rid_b, rid_n = _resource_id(with_b), _resource_id(without)
    cb = next((c for c in with_b.aha(stimulus=["entity:noam keller"], now=T0 + 5.0) if c.resource == rid_b), None)
    cn = next((c for c in without.aha(stimulus=["entity:noam keller"], now=T0 + 5.0, aha_min=0.0) if c.resource == rid_n), None)
    assert cb is not None
    assert cn is None or cn.need_side < cb.need_side, (cn and cn.need_side, cb.need_side)


def test_judge_connect_then_ledger_blocks(tmp_path):
    lib = _lib(tmp_path)
    rid = _resource_id(lib)
    cand = next(c for c in lib.aha(stimulus=["entity:noam keller"], now=T0 + 5.0) if c.resource == rid)
    seen = {}

    def judge_model(system, user, **kw):
        seen["user"] = user
        return "<think>P wants a job; Noam has an open CTO seat Artemy declined; they met at Brightmem.</think>\nVERDICT: connect\nLINK: Pavel could be introduced to Noam for the open CTO seat at Starling."
    out = lib.judge(cand, judge_model, now=T0 + 5.0, doing="idle")
    assert out.verdict == "connect" and "introduced" in out.link
    assert "NEED (as said)" in seen["user"] and "RESOURCE (as said)" in seen["user"] and "PATHS" in seen["user"]
    assert "CLOCKS" in seen["user"] and "NOW: idle" in seen["user"]
    again = lib.aha(stimulus=["entity:noam keller"], now=T0 + 6.0)
    # The same need + resource over the same path types is not raised twice. A SECOND need of
    # the same subject (looking_for beside asked_about) keeps its own ledger and may still
    # raise it — the ledger is per need by design (§4.1); folding needs of one subject is a
    # later refinement.
    assert all(not (c.resource == rid and c.need == cand.need) for c in again)
    assert any(e["verdict"] == "connect" for e in lib.ledger.prior_for(cand.need))


def test_satisfies_closes_the_need(tmp_path):
    lib = _lib(tmp_path)
    rid = _resource_id(lib)
    cand = next(c for c in lib.aha(stimulus=["entity:noam keller"], now=T0 + 5.0) if c.resource == rid)
    out = lib.judge(cand, lambda s, u, **kw: "VERDICT: satisfies\nLINK: Pavel already took the job.", now=T0 + 5.0)
    assert out.verdict == "satisfies"
    assert cand.need not in {n["need"] for n in lib.needs()}


def test_error_verdict_is_never_recorded(tmp_path):
    lib = _lib(tmp_path)
    rid = _resource_id(lib)
    cand = next(c for c in lib.aha(stimulus=["entity:noam keller"], now=T0 + 5.0) if c.resource == rid)
    calls = []

    def broken(system, user, **kw):
        calls.append(1)
        return "I cannot decide."
    out = lib.judge(cand, broken, now=T0 + 5.0)
    assert out.verdict == "error" and len(calls) == 2
    assert not lib.ledger.prior_for(cand.need)
    assert any(c.resource == rid for c in lib.aha(stimulus=["entity:noam keller"], now=T0 + 6.0))


def test_parse_verdict_shapes():
    assert parse_verdict("<think>x</think>\nVERDICT: no\nLINK: —") == ("no", "—")
    assert parse_verdict("VERDICT: **stale**.\nLINK: it was withdrawn") == ("stale", "it was withdrawn")
    assert parse_verdict("nothing here") == (None, "")


def test_registered_need_and_scope_fencing(tmp_path):
    lib = _lib(tmp_path)
    nid = lib.register_need("find a CTO for Starling", scope={"tenant": "acme"}, subject="entity:starling")
    assert nid in {n["need"] for n in lib.needs({"tenant": "acme"})}
    assert nid not in {n["need"] for n in lib.needs({"tenant": "globex"})}
    lib.close_need(nid, "resolved")
    assert nid not in {n["need"] for n in lib.needs({"tenant": "acme"})}
