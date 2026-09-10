"""The edge table for spreading (ASSOCIATIVE_MEMORY.md §2.7): five node kinds — claims,
chunks, documents, entity/person nodes, needs — joined by typed, undirected edges with a
base strength each, and per-type fan counts. Derived at rebuild, static until the next.

Node ids: ``claim:<id>``, ``chunk:<id>``, ``doc:<id>``, the subject/entity ids themselves
(``person:x``, ``entity:x``, ``ident:fam:x``), ``need:<claim_id>``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

S_TYPE = {"about": 1.0, "mentions": 0.7, "rel": 1.0, "in_chunk": 0.6, "adjacent": 0.3, "in_doc": 0.2,
          "contests": 1.0, "supersedes": 1.0, "need": 1.0, "inferred": 0.3}
LEARNED_MIN, LEARNED_MAX = 0.5, 1.5
NEED_PREDICATES = frozenset({"wants", "asked_about", "looking_for", "needs", "todo"})


def entity_node(mention: str, *, aliases: dict[str, str], persons: set[str]) -> str:
    """A mention → a node id: a known person key when the first token names one, else
    ``entity:<lowercased mention>`` — through the alias table either way."""
    m = " ".join(str(mention).split()).strip().strip("«»\"'`").lower()
    m = aliases.get(m, m)
    if not m:
        return ""
    first = m.split()[0]
    first = aliases.get(first, first)
    if f"person:{first}" in persons:
        return f"person:{first}"
    if m.startswith(("person:", "entity:", "ident:", "cell:")):
        return m
    return f"entity:{m}"


class EdgeTable:
    def __init__(self):
        self.adj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)   # node -> [(other, type, evidence)]
        self.fan: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.kinds: dict[str, str] = {}
        self.needs: dict[str, dict] = {}      # need node -> {claim_id, subject, entities, predicate, object}
        self.weight: dict[tuple[str, str, str], float] = {}   # (a, b, type) -> learned strength multiplier

    def add(self, a: str, b: str, etype: str, evidence: str = "") -> None:
        if not a or not b or a == b:
            return
        self.adj[a].append((b, etype, evidence))
        self.adj[b].append((a, etype, evidence))
        self.fan[a][etype] += 1
        self.fan[b][etype] += 1

    def edges(self, node: str) -> list[tuple[str, str, str]]:
        return self.adj.get(node, [])

    def w(self, a: str, b: str, etype: str) -> float:
        return self.weight.get((a, b, etype), self.weight.get((b, a, etype), 1.0))

    def learn_strength(self) -> dict:
        """PMI-shaped learned strength on `mentions` edges (§2.7, "what s_type does not yet
        do"): an entity mentioned by claims that share their SUBJECT with many other claims
        mentioning it is more associated with that subject than one co-mentioned once.
        cooc(subject, entity) = number of claims about the subject mentioning the entity;
        multiplier = LEARNED_MIN + (LEARNED_MAX − LEARNED_MIN) · log(1 + cooc) / log(1 + max)."""
        subj_of: dict[str, str] = {}
        for a, lst in self.adj.items():
            if not a.startswith("claim:"):
                continue
            for b, t, _e in lst:
                if t == "about":
                    subj_of[a] = b
                    break
        cooc: dict[tuple[str, str], int] = defaultdict(int)
        for a, lst in self.adj.items():
            if not a.startswith("claim:") or a not in subj_of:
                continue
            for b, t, _e in lst:
                if t == "mentions":
                    cooc[(subj_of[a], b)] += 1
        mx = max(cooc.values(), default=1)
        n = 0
        for a, lst in list(self.adj.items()):
            if not a.startswith("claim:") or a not in subj_of:
                continue
            for b, t, _e in lst:
                if t != "mentions":
                    continue
                c = cooc.get((subj_of[a], b), 1)
                self.weight[(a, b, t)] = round(LEARNED_MIN + (LEARNED_MAX - LEARNED_MIN) * math.log(1 + c) / math.log(1 + mx), 4)
                n += 1
        return {"weighted_edges": n, "max_cooc": mx}

    def fan_of(self, node: str, etype: str) -> int:
        return max(self.fan.get(node, {}).get(etype, 1), 1)

    def save(self, dirpath: Path) -> None:
        obj = {"adj": {k: v for k, v in self.adj.items()}, "fan": {k: dict(v) for k, v in self.fan.items()},
               "kinds": self.kinds, "needs": self.needs,
               "weight": [[a, b, t, w] for (a, b, t), w in self.weight.items()]}
        (dirpath / "edges.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(dirpath: Path) -> "EdgeTable":
        obj = json.loads((dirpath / "edges.json").read_text(encoding="utf-8"))
        t = EdgeTable()
        t.adj = defaultdict(list, {k: [tuple(x) for x in v] for k, v in obj["adj"].items()})
        t.fan = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in obj["fan"].items()})
        t.kinds = obj.get("kinds", {})
        t.needs = obj.get("needs", {})
        t.weight = {(a, b, ty): w for a, b, ty, w in obj.get("weight", [])}
        return t

    def stats(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        for a, lst in self.adj.items():
            for _b, t, _e in lst:
                by_type[t] += 1
        return {"nodes": len(self.adj), "edges_by_type": {k: v // 2 for k, v in by_type.items()},
                "needs": len(self.needs), "max_fan": max((max(f.values()) for f in self.fan.values() if f), default=0)}


def build_edges(docs: list, claims: dict[str, dict], links: dict, *, aliases: dict[str, str],
                closed_needs: set[str] = frozenset(), inferred: Optional[list[dict]] = None) -> EdgeTable:
    t = EdgeTable()
    # Inferred edges (§4, B2): a model's guess about the world, attributed, low weight, never
    # the sole path to an aha. Only loaded when the knob is on.
    for e in inferred or []:
        if e.get("a") and e.get("b") and e.get("pred"):
            t.add(e["a"], e["b"], f"inferred:{e['pred']}", "model")
    persons = {c["subject"] for c in claims.values() if (c.get("subject") or "").startswith("person:")}
    # about / mentions / in_chunk / rel / need
    for cid, c in claims.items():
        cn = f"claim:{cid}"
        t.kinds[cn] = "claim"
        subj = c.get("subject") or ""
        if subj:
            t.add(cn, subj, "about", cid)
            t.kinds[subj] = subj.split(":", 1)[0]
        for e in c.get("entities") or []:
            en = entity_node(e, aliases=aliases, persons=persons)
            if en and en != subj:
                t.add(cn, en, "mentions", cid)
                t.kinds.setdefault(en, en.split(":", 1)[0])
        for o in c.get("occurrences") or []:
            if o.get("chunk_id"):
                t.add(cn, f"chunk:{o['chunk_id']}", "in_chunk", cid)
        rel = c.get("rel")
        if rel and len(rel) == 3:
            pred, a, b = rel
            an = entity_node(a, aliases=aliases, persons=persons) if not str(a).startswith(("person:", "entity:", "ident:", "cell:")) else a
            bn = entity_node(b, aliases=aliases, persons=persons) if not str(b).startswith(("person:", "entity:", "ident:", "cell:")) else b
            if pred in NEED_PREDICATES or c.get("facet") == "need":
                if cid not in closed_needs:
                    nn = f"need:{cid}"
                    t.kinds[nn] = "need"
                    t.needs[nn] = {"claim_id": cid, "subject": subj, "entities": list(c.get("entities") or []),
                                   "predicate": pred, "object": bn, "text": c.get("text"), "kind": c.get("kind")}
                    t.add(nn, cn, "need", cid)
                    if an:
                        t.add(nn, an, "need", cid)
                    if bn:
                        t.add(nn, bn, "need", cid)
            else:
                if an and bn:
                    t.add(an, bn, f"rel:{pred}", cid)
                    t.kinds.setdefault(an, an.split(":", 1)[0])
                    t.kinds.setdefault(bn, bn.split(":", 1)[0])
        elif c.get("facet") == "need" and cid not in closed_needs:
            nn = f"need:{cid}"
            t.kinds[nn] = "need"
            t.needs[nn] = {"claim_id": cid, "subject": subj, "entities": list(c.get("entities") or []),
                           "predicate": "todo", "object": "", "text": c.get("text"), "kind": c.get("kind")}
            t.add(nn, cn, "need", cid)
            if subj:
                t.add(nn, subj, "need", cid)
        for other in c.get("contests") or []:
            if other in claims and cid < other:
                t.add(cn, f"claim:{other}", "contests", cid)
        if c.get("superseded_by") in claims:
            t.add(cn, f"claim:{c['superseded_by']}", "supersedes", cid)
    # adjacent / in_doc over primaries in reading order
    for d in docs:
        dn = f"doc:{d.doc_id}"
        t.kinds[dn] = "doc"
        prim = [u for u in d.units if u.role == "primary"]
        for i, u in enumerate(prim):
            cn = f"chunk:{u.chunk_id}"
            t.kinds[cn] = "chunk"
            t.add(cn, dn, "in_doc", d.doc_id)
            if i > 0:
                t.add(cn, f"chunk:{prim[i - 1].chunk_id}", "adjacent", d.doc_id)
    return t
