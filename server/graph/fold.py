"""The fold: occurrences → claims → facets → nodes, plus the edges between them.

Four levels, each answering a different consumer question (FACTS_TREE.md §4):

    node        what is there anything at all about?
     └── facet  what is *still true* vs what someone *said* vs what *happened*
      └── claim the statement itself — the unit you would show or answer with
       └── occurrence  provenance: which source, when, who, how often

The **facet level is the one that earns its place**, and the measurement is why. On this box
80% of the chat corpus (280 of 349 facts) is two speakers' ``stated`` positions, and 59% is
Ava's own; the whole chat lane holds 52 ``standing`` facts and 11 events. So a tree without
a facet level is overwhelmingly a pile of opinions presented as knowledge — and ``stated``
means *true only as a record that someone said it*, which is the one reading that must never
be lost. The facet makes that structural instead of advisory: a consumer selects
``property``, and no amount of volume in ``position`` can leak into it.

The facet also repairs a real semantic collision between the lanes for free. On the TIL lane
``stated`` is written for *reported events* ("All missiles were successfully intercepted" —
class ``stated``), because there the pass means "the text asserts this", not "someone holds
this view". Mapping ``(lane, fact_class)`` rather than reading ``fact_class`` alone separates
"a news digest states X" from "Artemy thinks X" without touching either producer.

Everything here is pure. GPU-free self-test: ``python -m graph.fold``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from graph import nodes as G
from graph.read import LANE_CHAT, LANE_TIL

# Facets.
FACET_PROPERTY = "property"          # true beyond the moment
FACET_POSITION = "position"          # X holds this — true only as a record that they said it
FACET_REPORT = "report"              # a text asserts this — weight by provenance
FACET_EVENT = "event"                # happened
FACET_DEPICTION = "depiction"        # a work depicts this — true inside it and nowhere else
FACET_UNCLASSIFIED = "unclassified"  # the producer could not label it; kept separate

FACETS = (FACET_PROPERTY, FACET_POSITION, FACET_REPORT, FACET_EVENT, FACET_DEPICTION,
          FACET_UNCLASSIFIED)

# Facets safe to read as knowledge about a node. `position`, `report` and `depiction` are
# excluded for three different reasons — a position is one person's view, a report is one
# text's claim, a depiction is true only inside the work that depicts it — and all three
# need their attribution carried with them, which only the occurrence level has.
KNOWLEDGE_FACETS = (FACET_PROPERTY, FACET_EVENT)

_FACET_MAP = {
    (LANE_CHAT, "standing"): FACET_PROPERTY,
    (LANE_CHAT, "stated"): FACET_POSITION,
    (LANE_CHAT, "event"): FACET_EVENT,
    (LANE_TIL, "standing"): FACET_PROPERTY,
    (LANE_TIL, "stated"): FACET_REPORT,
    (LANE_TIL, "event"): FACET_EVENT,
    # Fiction, folklore, tropes, invented worlds. There is no `(chat, depicted)` entry: the
    # chat prompt never asks for the class, and an unmapped pair lands in `unclassified`,
    # which is outside KNOWLEDGE_FACETS — so the failure mode of the vocabulary drifting is
    # material being withheld, never material being read back as known.
    (LANE_TIL, "depicted"): FACET_DEPICTION,
}

_WS_RE = re.compile(r"\s+")
_TRAIL_RE = re.compile(r"[\s.;,:!?]+$")


def facet_for(lane: str, fact_class: str) -> str:
    """``(lane, fact_class)`` → facet. Pure; no model, no new producer field.

    An unlabelled class becomes ``unclassified`` rather than being folded into a neighbour,
    for the reason ``chat_facts.UNSPECIFIED`` exists: a wrong label is contamination, and
    this record's whole purpose is later machine processing.
    """
    return _FACET_MAP.get((lane, str(fact_class or "").strip().lower()), FACET_UNCLASSIFIED)


def claim_text_key(text: str) -> str:
    """The key two claim texts are compared on — stage 1's exact tier.

    Case, whitespace and terminal punctuation only. Paraphrase merging is stage 3's
    embedding tier; doing it badly here would silently collapse distinct claims, and the
    tree's read-only inspection surface exists precisely to catch that before it is trusted.
    """
    s = _WS_RE.sub(" ", str(text or "")).strip().casefold()
    return _TRAIL_RE.sub("", s)


def claim_id(node: str, facet: str, text_key: str) -> str:
    """Stable across builds of an unchanged corpus, so a UI can keep a selection."""
    h = hashlib.sha1(f"{node}\x00{facet}\x00{text_key}".encode("utf-8")).hexdigest()
    return h[:16]


def _rep_text(texts: list) -> str:
    """The surface form shown for a claim: most frequent, ties to the longest.

    Longest on a tie because these are near-identical strings differing by a trailing
    qualifier, and the fuller sentence is the more useful record.
    """
    counts: dict = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]


def fold(occurrences: list, resolver: Optional[G.Resolver] = None) -> dict:
    """Build the tree. Returns ``{nodes, claims, occurrences, stats}``.

    Deterministic: two folds of an unchanged corpus produce identical output, which is what
    lets "rebuild instead of migrate" be checked rather than asserted.
    """
    r = resolver or G.Resolver()
    r.learn(occurrences)

    claims: dict = {}
    occ_rows: list = []
    # Occurrences about nobody in particular are kept and counted, never dropped: on the TIL
    # lane an unowned fact is ordinary, and losing it would make the corpus look smaller than
    # it is. They simply hang off no node.
    unowned = 0

    for idx, o in enumerate(occurrences):
        node = r.resolve_subject(o)
        facet = facet_for(o.get("lane", ""), o.get("fact_class", ""))
        text = str(o.get("text") or "").strip()
        tkey = claim_text_key(text)
        if not tkey:
            continue

        mentions: list = []
        for m in (o.get("entities") or []):
            ident = r.resolve_mention(m)
            if ident and ident != node and ident not in mentions:
                mentions.append(ident)

        row = dict(o)
        row["occurrence_id"] = idx
        row["node"] = node
        row["facet"] = facet
        row["mentions"] = mentions
        occ_rows.append(row)

        if node is None:
            unowned += 1
            continue

        cid = claim_id(node, facet, tkey)
        c = claims.get(cid)
        if c is None:
            c = claims[cid] = {
                "claim_id": cid, "node": node, "facet": facet,
                "_texts": [], "occurrences": [], "sources": [],
                "mentions": [], "when": "", "lanes": [],
            }
        c["_texts"].append(text)
        c["occurrences"].append(idx)
        src = o.get("source_ref") or ""
        if src and src not in c["sources"]:
            c["sources"].append(src)
        lane = o.get("lane") or ""
        if lane and lane not in c["lanes"]:
            c["lanes"].append(lane)
        for m in mentions:
            if m not in c["mentions"]:
                c["mentions"].append(m)
        if o.get("when") and not c["when"]:
            c["when"] = o["when"]

    # Finalize claims: representative text, the two clocks, corroboration.
    for c in claims.values():
        c["text"] = _rep_text(c["_texts"])
        del c["_texts"]
        stamps = sorted(s for s in (occurrences[i].get("asserted_at") or ""
                                    for i in c["occurrences"]) if s)
        c["first_asserted"] = stamps[0] if stamps else ""
        c["last_asserted"] = stamps[-1] if stamps else ""
        c["n_occurrences"] = len(c["occurrences"])
        # Corroboration is *distinct sources*, not occurrence count: one conversation
        # restating a thing four times is one witness, and counting it as four is how a fold
        # manufactures confidence it has not earned.
        c["n_sources"] = len(c["sources"])

    # Nodes, assembled from the claims that landed on them.
    node_map: dict = {}

    def _ensure(ident: str) -> dict:
        n = node_map.get(ident)
        if n is None:
            t, key = G.parse_node_id(ident)
            n = node_map[ident] = {
                "id": ident, "type": t or G.TYPE_UNKNOWN, "key": key,
                "label": r.label_for(ident), "surface_forms": r.surface_forms(ident),
                "facets": {}, "n_claims": 0, "n_occurrences": 0, "mentioned_in": [],
            }
        return n

    for c in sorted(claims.values(), key=lambda c: (c["node"], c["facet"], c["text"])):
        n = _ensure(c["node"])
        n["facets"].setdefault(c["facet"], []).append(c["claim_id"])
        n["n_claims"] += 1
        n["n_occurrences"] += c["n_occurrences"]
        # The reverse edge, derived here rather than stored on the node, so a rebuild can
        # never leave one dangling.
        for m in c["mentions"]:
            mn = _ensure(m)
            if c["claim_id"] not in mn["mentioned_in"]:
                mn["mentioned_in"].append(c["claim_id"])

    stats = _stats(occ_rows, claims, node_map, unowned)
    return {"nodes": node_map, "claims": claims, "occurrences": occ_rows, "stats": stats}


def _stats(occ_rows: list, claims: dict, node_map: dict, unowned: int) -> dict:
    by_type: dict = {}
    for n in node_map.values():
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    by_facet: dict = {}
    for c in claims.values():
        by_facet[c["facet"]] = by_facet.get(c["facet"], 0) + 1
    by_lane: dict = {}
    for o in occ_rows:
        by_lane[o.get("lane", "")] = by_lane.get(o.get("lane", ""), 0) + 1
    dup = sum(c["n_occurrences"] - 1 for c in claims.values())
    corroborated = sum(1 for c in claims.values() if c["n_sources"] > 1)
    unknown = by_type.get(G.TYPE_UNKNOWN, 0)
    total_nodes = len(node_map)
    return {
        "occurrences": len(occ_rows),
        "occurrences_by_lane": by_lane,
        "occurrences_unowned": unowned,
        "claims": len(claims),
        "claims_by_facet": by_facet,
        "claims_corroborated": corroborated,
        "occurrences_collapsed": dup,
        "nodes": total_nodes,
        "nodes_by_type": by_type,
        # The headline number of a stage-1 build: how much of the mention space the
        # structural tiers could not type, and therefore what tiers 3-4 have to earn.
        "unresolved_fraction": round(unknown / total_nodes, 3) if total_nodes else 0.0,
    }


# -- self-test --------------------------------------------------------------- #

def _selftest() -> None:
    assert facet_for(LANE_CHAT, "stated") == FACET_POSITION
    assert facet_for(LANE_TIL, "stated") == FACET_REPORT      # the lane collision, fixed
    assert facet_for(LANE_CHAT, "standing") == FACET_PROPERTY
    assert facet_for(LANE_TIL, "standing") == FACET_PROPERTY
    assert facet_for(LANE_CHAT, "event") == facet_for(LANE_TIL, "event") == FACET_EVENT
    # Fiction: carried on the TIL lane as its own facet, outside knowledge. On the chat
    # lane the pair is unmapped and lands in `unclassified` — also outside knowledge, so a
    # vocabulary drift withholds material rather than promoting it.
    assert facet_for(LANE_TIL, "depicted") == FACET_DEPICTION
    assert facet_for(LANE_CHAT, "depicted") == FACET_UNCLASSIFIED
    assert FACET_DEPICTION not in KNOWLEDGE_FACETS
    assert facet_for(LANE_CHAT, "unspecified") == FACET_UNCLASSIFIED
    assert facet_for(LANE_CHAT, "") == FACET_UNCLASSIFIED
    assert facet_for(LANE_CHAT, "STANDING") == FACET_PROPERTY  # producer casing tolerated

    assert claim_text_key("  Drinks it.  ") == claim_text_key("drinks it")
    assert claim_id("a", "b", "c") == claim_id("a", "b", "c")
    assert claim_id("a", "b", "c") != claim_id("a", "b", "d")

    occ = [
        # Same claim, two different conversations → one claim, two sources, corroborated.
        {"lane": LANE_CHAT, "source_ref": "s1.json", "subject_key": "artemyvo",
         "subject_raw": "artemyvo", "text": "Drinks Reviseur XO on the balcony.",
         "fact_class": "standing", "entities": ["Reviseur XO"], "when": "",
         "asserted_at": "2026-07-01"},
        {"lane": LANE_CHAT, "source_ref": "s2.json", "subject_key": "artemyvo",
         "subject_raw": "Artemy", "text": "drinks reviseur xo on the balcony",
         "fact_class": "standing", "entities": [], "when": "", "asserted_at": "2026-07-09"},
        # Same conversation restating one thing → one source, NOT corroboration.
        {"lane": LANE_CHAT, "source_ref": "s3.json", "subject_key": "_self",
         "subject_raw": "self", "text": "Subjectivity is the point.",
         "fact_class": "stated", "entities": ["subjectivity"], "when": "",
         "asserted_at": "2026-07-10"},
        {"lane": LANE_CHAT, "source_ref": "s3.json", "subject_key": "_self",
         "subject_raw": "self", "text": "Subjectivity is the point",
         "fact_class": "stated", "entities": [], "when": "", "asserted_at": "2026-07-10"},
        # TIL: a reported event, and an unowned general observation.
        {"lane": LANE_TIL, "source_ref": "news/2026-07-28", "subject_key": "USCENTCOM",
         "subject_raw": "USCENTCOM", "text": "Missiles were intercepted.",
         "fact_class": "stated", "entities": ["Iran"], "when": "2026-07-28",
         "asserted_at": "2026-07-28"},
        {"lane": LANE_TIL, "source_ref": "news/2026-07-28", "subject_key": "nobody",
         "subject_raw": "nobody", "text": "The weather was poor.", "fact_class": "event",
         "entities": [], "when": "2026-07-28", "asserted_at": "2026-07-28"},
    ]

    t = fold(occ)
    st = t["stats"]

    assert st["occurrences"] == 6, st
    # Three claims, not four: the unowned line yields none, because it hangs off no node.
    assert st["claims"] == 3, sorted((c["node"], c["text"]) for c in t["claims"].values())
    assert st["occurrences_collapsed"] == 2, st
    assert st["occurrences_unowned"] == 1, st
    # It is RETAINED as an occurrence, though — never silently dropped. On the TIL lane a
    # fact about nothing in particular is ordinary, and losing it would make the corpus look
    # smaller than it is. Stage 1 keeps it reachable only by scanning occurrences; the
    # summary's "about nobody" line is what stops that being invisible.
    orphan = [o for o in t["occurrences"] if o["node"] is None]
    assert len(orphan) == 1 and orphan[0]["text"] == "The weather was poor."
    assert orphan[0]["facet"] == FACET_EVENT

    by_node = {}
    for c in t["claims"].values():
        by_node.setdefault(c["node"], []).append(c)

    # Cross-conversation restatement: one claim, two sources → corroborated.
    a = [c for c in by_node["person:artemyvo"]][0]
    assert a["facet"] == FACET_PROPERTY
    assert a["n_occurrences"] == 2 and a["n_sources"] == 2, a
    assert a["first_asserted"] == "2026-07-01" and a["last_asserted"] == "2026-07-09"
    assert a["text"] == "Drinks Reviseur XO on the balcony."   # the fuller surface form

    # Same-conversation restatement: two occurrences, ONE source, not corroborated.
    s = by_node[G.SELF_NODE][0]
    assert s["facet"] == FACET_POSITION
    assert s["n_occurrences"] == 2 and s["n_sources"] == 1, s
    assert st["claims_corroborated"] == 1, st

    # The TIL lane's reported event reads as a report, never as somebody's opinion.
    til = [c for c in t["claims"].values() if c["node"] == "entity:USCENTCOM"][0]
    assert til["facet"] == FACET_REPORT and til["when"] == "2026-07-28"

    # Edges: the claim points out, the node points back, and a self-edge is suppressed.
    assert "unknown:subjectivity" in s["mentions"]
    assert s["claim_id"] in t["nodes"]["unknown:subjectivity"]["mentioned_in"]
    assert "unknown:Reviseur XO" in a["mentions"]
    assert G.SELF_NODE not in s["mentions"]

    # A mention-only node exists and holds no claims of its own.
    subj = t["nodes"]["unknown:subjectivity"]
    assert subj["n_claims"] == 0 and subj["mentioned_in"]

    # Knowledge facets exclude exactly the two that need attribution carried with them.
    assert FACET_POSITION not in KNOWLEDGE_FACETS and FACET_REPORT not in KNOWLEDGE_FACETS

    # Determinism: an unchanged corpus folds identically.
    import json as _json
    assert _json.dumps(fold(occ)["stats"], sort_keys=True) == _json.dumps(st, sort_keys=True)

    # An empty corpus is a valid tree, not a crash.
    empty = fold([])
    assert empty["stats"]["nodes"] == 0 and empty["stats"]["unresolved_fraction"] == 0.0

    print("graph.fold self-test OK")


if __name__ == "__main__":
    _selftest()
