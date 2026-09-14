"""GPU-free self-test for the whole graph package: ``python -m graph.selftest``.

Runs each module's own test, then the end-to-end path the CLI takes — real files on disk
through read → resolve → fold → write → read back — since every module passing in isolation
does not prove the seams between them hold.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _end_to_end() -> None:
    from graph import build as B
    from graph import fold as F
    from graph import nodes as G
    from graph import store as S

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        chats.mkdir()
        news = root / "til" / "snippets" / "news"
        news.mkdir(parents=True)
        graph_dir = root / "graph"
        graph_dir.mkdir()

        (chats / "20260701_120000.facts.json").write_text(json.dumps({
            "source_session": "20260701_120000.json", "source_user": "artemyvo",
            "run_id": "r1", "facts": [
                {"subject": "artemyvo", "subject_raw": "artemyvo",
                 "text": "Drinks Reviseur XO on the balcony.", "fact_class": "standing",
                 "entities": ["Reviseur XO"], "when": ""},
                {"subject": "_self", "subject_raw": "self",
                 "text": "Subjectivity is the point.", "fact_class": "stated",
                 "entities": ["subjectivity", "Artemy"], "when": ""},
            ]}), encoding="utf-8")
        (chats / "20260709_090000.facts.json").write_text(json.dumps({
            "source_session": "20260709_090000.json", "source_user": "artemyvo",
            "run_id": "r2", "facts": [
                {"subject": "artemyvo", "subject_raw": "Artemy",
                 "text": "drinks reviseur xo on the balcony", "fact_class": "standing",
                 "entities": [], "when": ""},
            ]}), encoding="utf-8")
        (news / "2026-07-28.facts.json").write_text(json.dumps({
            "source_kind": "news", "source_ref": "wikipedia:current_events",
            "source_url": "http://x", "source_title": "Current events",
            "source_date": "2026-07-28", "run_id": "r2", "facts": [
                {"subject": "United States Central Command",
                 "subject_raw": "United States Central Command",
                 "text": "Missiles were intercepted.", "fact_class": "stated",
                 "entities": ["Iran"], "when": "2026-07-28"},
            ]}), encoding="utf-8")

        # An alias resolves the surface form the second chat used, and would otherwise be a
        # separate person node.
        (graph_dir / G.ALIASES_FILE).write_text(json.dumps({
            "person:artemyvo": ["Artemy"]}), encoding="utf-8")

        tree, read_stats = B.build(graph_dir, chats_dir=chats,
                                   archive_dir=root / "archive" / "chats",
                                   snippets_dir=root / "til" / "snippets")
        st = tree["stats"]
        assert read_stats["files"] == 3, read_stats
        assert st["occurrences"] == 4, st

        # The two lanes coexist without colliding, and neither became the other's type.
        assert tree["nodes"]["person:artemyvo"]["type"] == G.TYPE_PERSON
        assert tree["nodes"]["entity:United States Central Command"]["type"] == G.TYPE_ENTITY

        # Cross-conversation restatement folded into one corroborated claim.
        a = [c for c in tree["claims"].values()
             if c["node"] == "person:artemyvo"][0]
        assert a["n_sources"] == 2 and a["facet"] == F.FACET_PROPERTY, a

        # The alias did its job: "Artemy" as a *mention* landed on the person, not on a new
        # unknown node.
        s = [c for c in tree["claims"].values() if c["node"] == G.SELF_NODE][0]
        assert "person:artemyvo" in s["mentions"], s
        assert "unknown:Artemy" not in tree["nodes"]

        # TIL `stated` is a report, never a position.
        til = [c for c in tree["claims"].values()
               if c["node"].startswith("entity:")][0]
        assert til["facet"] == F.FACET_REPORT

        # Write → read back → the consumer API works off the on-disk doc.
        doc = S.build_doc(tree, built_at="2026-08-10T00:00:00", read_stats=read_stats)
        S.write_tree(doc, graph_dir)
        back = S.read_tree(graph_dir)
        assert back is not None and back["schema_version"] == S.SCHEMA_VERSION

        knowledge = S.claims_for(back, "person:artemyvo", facets=F.KNOWLEDGE_FACETS)
        assert len(knowledge) == 1 and knowledge[0]["facet"] == F.FACET_PROPERTY
        # The facet filter is the safety control: positions are excluded from knowledge.
        assert S.claims_for(back, G.SELF_NODE, facets=F.KNOWLEDGE_FACETS) == []
        assert len(S.claims_for(back, G.SELF_NODE)) == 1

        # Provenance survives the round trip.
        occ = S.occurrences_for(back, S.claims_for(back, G.SELF_NODE)[0])
        assert occ and occ[0]["source_ref"] == "20260701_120000.json"
        assert occ[0]["subject_raw"] == "self"

        # Search finds a node by a surface form it does not carry as its key.
        assert any(h["id"] == "person:artemyvo" for h in S.find_nodes(back, "Artemy"))

        # A corrupt tree reads as "no tree yet" — rebuild, never repair.
        S.tree_path(graph_dir).write_text("{ nope", encoding="utf-8")
        assert S.read_tree(graph_dir) is None
        assert S.read_tree(root / "no-such-dir") is None

        # The CLI runs end to end against these fixtures — sealed off from the real corpus,
        # so the test result cannot vary by box. Every render path is exercised, because a
        # formatter that throws on an empty facet or a missing node is a real failure of the
        # only surface stage 1 ships.
        argv = ["--graph-dir", str(graph_dir), "--chats-dir", str(chats),
                "--snippets-dir", str(root / "til" / "snippets"), "--dry-run"]
        before = S.tree_path(graph_dir).read_bytes()
        for extra in ([], ["--nodes"], ["--unresolved"], ["--find", "Artemy"],
                      ["--node", "person:artemyvo"], ["--node", G.SELF_NODE],
                      ["--node", "person:nope"], ["--json"]):
            assert B.main(argv + extra) == 0, extra
        # --dry-run wrote nothing: the file on disk is byte-identical (it is the corrupt one
        # written just above, which is the strictest available witness that no write ran).
        assert S.tree_path(graph_dir).read_bytes() == before

        # ...and without --dry-run it does write, repairing that corrupt file by rebuild.
        assert B.main([a for a in argv if a != "--dry-run"]) == 0
        assert S.read_tree(graph_dir) is not None

    print("graph end-to-end self-test OK")


def main() -> None:
    from graph import blob, fold, nodes, read
    read._selftest()
    nodes._selftest()
    fold._selftest()
    blob._selftest()
    _end_to_end()
    print("graph: all self-tests OK")


if __name__ == "__main__":
    main()
