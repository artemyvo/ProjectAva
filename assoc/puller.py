"""The puller, milestone-1 tier (ASSOCIATIVE_MEMORY.md §3, `bench/baseline` tier 1):
lexical (L1 BM25) + dense channels over chunks and claims, scope-filtered, three grains,
every hit explainable. Activation and spreading arrive in milestone 3 behind the same
`Hit` shape; the channels here become its seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rebuild import Build
from .store import Store

MIN_LEX_N, MIN_DENSE_N, MIN_CONCEPT_N = 0.15, 0.35, 0.2    # normalized floors per channel
CONCEPT_MIN_RAW = 0.6      # absolute floor on a concept hit: one generic cell ({start, restart, stop}) scores ~0.3
REMOVED_PRIOR = 0.85                   # a "removed in <version>" event ranks below the current fact (§2.5)
GIST_MIN_CHUNKS, GIST_MAX_CHUNK_SCORE = 3, 0.6


@dataclass
class Hit:
    grain: str                      # claim | chunk | gist
    id: str                         # claim_id | chunk_id | doc_id
    score: float
    doc_id: str
    chunk_id: Optional[str]
    channels: dict = field(default_factory=dict)     # {"lexical": s, "dense": s, "concept": s}
    terms: list = field(default_factory=list)        # matched (term, contribution)
    cells: list = field(default_factory=list)        # concept-channel path: (cell, members, contribution)
    exact: bool = False
    claim: Optional[dict] = None
    meta: dict = field(default_factory=dict)
    build_id: str = ""
    rank: int = 0
    activation: Optional[float] = None               # tier 3: A_i on the one scale (§2.7)
    path: list = field(default_factory=list)         # tier 3: top spread paths (raw)

    @property
    def reference(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "key": self.meta.get("key"),
                "title": self.meta.get("title"), "path": self.meta.get("path"), "version": self.meta.get("version")}

    def to_dict(self) -> dict:
        return {"grain": self.grain, "id": self.id, "score": round(self.score, 4), "doc_id": self.doc_id,
                "activation": (round(self.activation, 4) if self.activation is not None else None),
                "chunk_id": self.chunk_id, "channels": {k: round(v, 4) for k, v in self.channels.items()},
                "terms": self.terms, "exact": self.exact, "rank": self.rank, "build_id": self.build_id,
                "reference": self.reference, "text": (self.claim or {}).get("text") if self.claim else None}


LEX_MASS_FRAC = 0.5               # a lexical top hit matching < half the query's idf mass is not a 1.0


def _normalize(hits: list[dict], floor: float = 0.0, ceiling: float = 0.0) -> dict[str, float]:
    """Map scores onto [0, 1] between *floor* and the top score. For the dense channel the
    floor is the embedder's own baseline cosine: BGE-M3 puts unrelated text at 0.5–0.6 and
    the right chunk at 0.7, so a top-only normalization would read the noise as 0.9. For the
    lexical channel *ceiling* is a share of the query's own BM25 mass: on a cue in the other
    language the best lexical hit matches a word or two, and top-only normalization read
    that as a full match (measured on Ava's corpus: an English fact's lexical 1.0 went to
    whichever unrelated English page shared one word)."""
    if not hits:
        return {}
    top = max(max(h["score"] for h in hits), ceiling)
    span = top - floor
    if span <= 1e-9:
        return {h["id"]: 1.0 for h in hits if h["score"] >= top}
    return {h["id"]: max(0.0, (h["score"] - floor) / span) for h in hits}


CONTEXT_SEED_WEIGHT = 0.35        # the recent turns' terms seed too, at a discount (§3 step 2)
CONCEPT_BOOST, CONCEPT_ALONE = 0.25, 0.8   # L2 boosts an L1/dense hit; alone it may lead, discounted
ALL_CHANNELS = frozenset({"lexical", "dense", "concept"})


def pull(store: Store, build: Build, cue: str, *, scope: Optional[dict] = None, embedder=None,
         limit: int = 60, context_terms: Optional[list[str]] = None, tier: int = 2,
         channels: Optional[set] = None, activation=None, now_h: Optional[float] = None,
         extra_seeds: Optional[dict] = None) -> list[Hit]:
    """*tier* 1 = lexical + dense (the `bench/baseline` floor); 2 adds the concept channel
    (L2 cells) and the recent-turn seeds; 3 turns the channel hits into seeds, spreads over
    the edge table and ranks by the one-scale activation A_i (§2.7). *channels* narrows
    further for measurement."""
    channels = set(channels or ALL_CHANNELS)
    if tier < 2 or build.codebook is None:
        channels.discard("concept")
    allowed_docs = set(store.current_doc_ids(scope))
    allowed_chunks = {cid for cid, docs in build.chunk_doc.items()
                      if build.injectable(cid) and any(d in allowed_docs for d in docs)}
    allowed_claims = {cid for cid, c in build.claims.items()
                      if any(d in allowed_docs for d in c.get("visible_in") or [])}

    ctx = " ".join(context_terms or []) if tier >= 2 else ""

    def lexical(kind: str, allowed: set[str]) -> list[dict]:
        if "lexical" not in channels:
            return []
        hits = build.glossary.search(cue, kind=kind, limit=limit * 2, allowed=allowed)
        if ctx:
            by = {h["id"]: h for h in hits}
            for h in build.glossary.search(ctx, kind=kind, limit=limit, allowed=allowed):
                if h["id"] in by:
                    by[h["id"]]["score"] += CONTEXT_SEED_WEIGHT * h["score"]
                else:
                    by[h["id"]] = {**h, "score": CONTEXT_SEED_WEIGHT * h["score"], "terms": h["terms"], "exact": False}
            hits = sorted(by.values(), key=lambda h: -h["score"])
        return hits

    def concept(kind: str, allowed: set[str]) -> list[dict]:
        if "concept" not in channels:
            return []
        hits = build.codebook.search(cue, kind=kind, limit=limit * 2, allowed=allowed, embedder=embedder)
        if ctx:
            by = {h["id"]: h for h in hits}
            for h in build.codebook.search(ctx, kind=kind, limit=limit, allowed=allowed, embedder=embedder):
                if h["id"] in by:
                    by[h["id"]]["score"] += CONTEXT_SEED_WEIGHT * h["score"]
                else:
                    by[h["id"]] = {**h, "score": CONTEXT_SEED_WEIGHT * h["score"]}
            hits = sorted(by.values(), key=lambda h: -h["score"])
        return [h for h in hits if h["score"] >= CONCEPT_MIN_RAW]

    lex_chunks, lex_claims = lexical("chunk", allowed_chunks), lexical("claim", allowed_claims)
    con_chunks, con_claims = concept("chunk", allowed_chunks), concept("claim", allowed_claims)
    dense_chunks: list[dict] = []
    dense_claims: list[dict] = []
    floor = float(getattr(embedder, "min_score", 0.0)) if embedder is not None else 0.0
    if build.dense is not None and embedder is not None and "dense" in channels:
        qv = embedder.encode([cue])[0]
        dense_chunks = build.dense.search(qv, k=limit * 2, kind="chunk", allowed=allowed_chunks, floor=floor)
        dense_claims = build.dense.search(qv, k=limit * 2, kind="claim", allowed=allowed_claims, floor=floor)

    def combine(lex: list[dict], dense: list[dict], con: list[dict]) -> dict[str, dict]:
        lex_ceiling = LEX_MASS_FRAC * (lex[0].get("mass", 0.0) if lex else 0.0)
        ln, dn, cn = _normalize(lex, ceiling=lex_ceiling), _normalize(dense, floor), _normalize(con)
        lex_by = {h["id"]: h for h in lex}
        con_by = {h["id"]: h for h in con}
        out: dict[str, dict] = {}
        for ident in set(ln) | set(dn) | set(cn):
            a, b, c = ln.get(ident, 0.0), dn.get(ident, 0.0), cn.get(ident, 0.0)
            if a < MIN_LEX_N:
                a = 0.0
            if b < MIN_DENSE_N:
                b = 0.0
            if c < MIN_CONCEPT_N:
                c = 0.0
            if a == 0.0 and b == 0.0 and c == 0.0:
                continue
            if a == 0.0 and b == 0.0:
                score = CONCEPT_ALONE * c          # the cross-lingual case: L2 is the only key
            else:
                score = max(a, b) + 0.25 * min(a, b) + CONCEPT_BOOST * c
            lh = lex_by.get(ident, {})
            out[ident] = {"score": score, "channels": {"lexical": a, "dense": b, "concept": c},
                          "terms": lh.get("terms", []), "exact": bool(lh.get("exact")),
                          "cells": con_by.get(ident, {}).get("cells", []), "raw_lex": lh.get("score", 0.0)}
        return out

    chunk_scores = combine(lex_chunks, dense_chunks, con_chunks)
    claim_scores = combine(lex_claims, dense_claims, con_claims)

    hits: list[Hit] = []
    for cid, s in chunk_scores.items():
        m = _chunk_meta_for(build, cid, allowed_docs)
        hits.append(Hit(grain="chunk", id=cid, score=s["score"], doc_id=m.get("doc_id", ""), chunk_id=cid,
                        channels=s["channels"], terms=s["terms"], cells=s.get("cells", []), exact=s["exact"], meta=m,
                        build_id=build.build_id))
    for cid, s in claim_scores.items():
        c = build.claims[cid]
        occ = next((o for o in c["occurrences"] if o["doc_id"] in allowed_docs), c["occurrences"][0] if c["occurrences"] else {})
        m = _chunk_meta_for(build, occ.get("chunk_id") or "", allowed_docs)
        m.setdefault("key", occ.get("key"))
        m.setdefault("title", occ.get("title"))
        m.setdefault("version", occ.get("version"))
        score = s["score"] * (REMOVED_PRIOR if cid.startswith("rm-") else 1.0)
        if c.get("superseded_by"):
            score *= REMOVED_PRIOR
        hits.append(Hit(grain="claim", id=cid, score=score, doc_id=occ.get("doc_id", ""), chunk_id=occ.get("chunk_id"),
                        channels=s["channels"], terms=s["terms"], cells=s.get("cells", []), exact=s["exact"], claim=c, meta=m,
                        build_id=build.build_id))

    # Gist grain: a document hit many-but-weakly at the chunk grain.
    per_doc: dict[str, list[float]] = {}
    for h in hits:
        if h.grain == "chunk":
            per_doc.setdefault(h.doc_id, []).append(h.score)
    for doc_id, scores in per_doc.items():
        if len(scores) >= GIST_MIN_CHUNKS and max(scores) < GIST_MAX_CHUNK_SCORE:
            meta = store.meta(doc_id) or {}
            hits.append(Hit(grain="gist", id=doc_id, score=min(1.0, sum(scores) / len(scores) + 0.1 * len(scores)),
                            doc_id=doc_id, chunk_id=None, channels={"chunks": float(len(scores))},
                            meta={"key": meta.get("key"), "title": meta.get("title"), "version": meta.get("version"), "path": [meta.get("title")]},
                            build_id=build.build_id))

    hits.sort(key=lambda h: -h.score)
    if tier >= 3 and build.edges is not None:
        hits = _tier3(store, build, hits, cue, scope=scope, activation=activation, now_h=now_h, allowed_docs=allowed_docs,
                      allowed_chunks=allowed_chunks, allowed_claims=allowed_claims, extra_seeds=extra_seeds)
    for i, h in enumerate(hits[:limit], 1):
        h.rank = i
    return hits[:limit]


def cue_entity_seeds(build: Build, cue: str) -> dict[str, float]:
    """Entity / person nodes the cue names outright (`touch(Noam)` from a message about Noam)."""
    from .lex import terms_of
    terms = set(terms_of(cue))
    if not terms or build.edges is None:
        return {}
    out: dict[str, float] = {}
    for node in build.edges.adj:
        if node.startswith(("person:", "entity:", "ident:")):
            key = node.split(":", 1)[1]
            key = key.split(":", 1)[1] if node.startswith("ident:") else key
            words = set(key.replace("_", " ").split())
            if words and words <= terms:
                out[node] = 1.0
    return out


SPREAD_WEIGHT, WARMTH_WEIGHT, SPREAD_ONLY_FLOOR = 0.25, 0.15, 0.12
ARRIVAL_FULL = 0.25        # an arrival carrying a quarter of the cue's activation counts as a full one


def _tier3(store, build: Build, hits: list[Hit], cue: str, *, scope, activation, now_h, allowed_docs, allowed_chunks,
           allowed_claims, extra_seeds=None) -> list[Hit]:
    """Tier 3 (§2.7, §3): the channel hits become seeds; the spread adds what they are
    associated with and the base level adds what was recently used. The channel score stays
    the backbone of the ranking — seeds are never cut by τ (they were gated by their own
    channels) — and two bounded terms adjust it: the activation that ARRIVED by spreading
    (normalized to the strongest arrival) and the node's warmth above the cold baseline.
    `A_i` on the one scale is reported on every hit for `explain`; ranking by `A_i` alone
    was measured to destroy tier 2's precision (a cue with sixty channel hits shares one
    unit of activation sixty ways, and τ then cut the answers themselves)."""
    from .activation import cold_baseline, now_hours
    from .spread import HOPS, activation as act_fn, spread
    now_h = now_hours() if now_h is None else now_h
    seeds: dict[str, float] = {}
    for h in hits:
        if h.grain in ("chunk", "claim"):
            node = f"{h.grain}:{h.id}"
            seeds[node] = seeds.get(node, 0.0) + h.score
    for node, w in cue_entity_seeds(build, cue).items():
        seeds[node] = seeds.get(node, 0.0) + w
    for node, w in (extra_seeds or {}).items():
        seeds[node] = seeds.get(node, 0.0) + w
    if not seeds:
        return hits
    reached = spread(build.edges, seeds, k=HOPS["chat"])
    total = sum(seeds.values()) or 1.0
    arrivals = {n: r - seeds.get(n, 0.0) / total for n, r in reached.r.items()}
    b0 = cold_baseline()
    by_id = {(h.grain, h.id): h for h in hits}
    out: list[Hit] = []
    seen: set = set()
    for node, r in reached.r.items():
        if node.startswith("claim:"):
            grain, ident = "claim", node[6:]
            if ident not in allowed_claims:
                continue
        elif node.startswith("chunk:"):
            grain, ident = "chunk", node[6:]
            if ident not in allowed_chunks:
                continue
        else:
            continue
        odds = activation.odds(node, now_h=now_h, scope=scope) if activation is not None else __import__("math").exp(b0)
        auth = float((build.claims.get(ident) or {}).get("authority") or 0.0) if grain == "claim" else 0.0
        a = act_fn(r, odds, auth)
        warmth = min(max(__import__("math").log(odds) - b0, 0.0), 3.0) / 3.0
        arrival = min(max(arrivals.get(node, 0.0), 0.0) / ARRIVAL_FULL, 1.0)
        h = by_id.get((grain, ident))
        if h is None:
            bonus = SPREAD_WEIGHT * arrival + WARMTH_WEIGHT * warmth
            if bonus < SPREAD_ONLY_FLOOR:
                continue
            if grain == "claim":
                c = build.claims[ident]
                occ = next((o for o in c["occurrences"] if o["doc_id"] in allowed_docs), c["occurrences"][0] if c["occurrences"] else {})
                m = _chunk_meta_for(build, occ.get("chunk_id") or "", allowed_docs)
                m.setdefault("key", occ.get("key")); m.setdefault("title", occ.get("title"))
                h = Hit(grain="claim", id=ident, score=bonus, doc_id=occ.get("doc_id", ""), chunk_id=occ.get("chunk_id"),
                        channels={"spread": r}, claim=c, meta=m, build_id=build.build_id)
            else:
                m = _chunk_meta_for(build, ident, allowed_docs)
                h = Hit(grain="chunk", id=ident, score=bonus, doc_id=m.get("doc_id", ""), chunk_id=ident,
                        channels={"spread": r}, meta=m, build_id=build.build_id)
        else:
            h.score = h.score + SPREAD_WEIGHT * arrival + WARMTH_WEIGHT * warmth
        h.channels["spread"] = r
        h.channels["warmth"] = warmth
        h.activation = a
        h.path = reached.paths.get(node, [])[:3]
        out.append(h)
        seen.add((grain, ident))
    out.extend(h for h in hits if (h.grain, h.id) not in seen)     # gists, and seeds the beam dropped
    out.sort(key=lambda h: -h.score)
    return out


def _chunk_meta_for(build: Build, chunk_id: str, allowed_docs: set[str]) -> dict:
    """The chunk's meta as seen from the document the scope allows (a chunk shared by
    several versions reports the allowed version's doc_id / version / date)."""
    m = dict(build.chunk_meta.get(chunk_id, {}))
    for d in m.get("docs") or []:
        if d["doc_id"] in allowed_docs:
            m.update({k: d[k] for k in ("doc_id", "version", "date", "title", "key")})
            break
    m.pop("docs", None)
    return m


def explain(hit: Hit) -> dict:
    """Why this hit: the channels and matched terms, in the shape §3 step 5 asks for."""
    parts = []
    if hit.channels.get("lexical"):
        parts.append({"channel": "lexical", "score": round(hit.channels["lexical"], 3),
                      "terms": [t for t, _ in hit.terms], "exact": hit.exact})
    if hit.channels.get("dense"):
        parts.append({"channel": "dense", "score": round(hit.channels["dense"], 3)})
    if hit.channels.get("concept"):
        parts.append({"channel": "concept", "score": round(hit.channels["concept"], 3),
                      "cells": [{"cell": c, "members": m, "contribution": w} for c, m, w in hit.cells]})
    if hit.channels.get("chunks"):
        parts.append({"channel": "gist", "chunks_hit": int(hit.channels["chunks"])})
    if hit.channels.get("spread") is not None and hit.path:
        from .spread import render_path
        parts.append({"channel": "spread", "r": round(hit.channels["spread"], 4), "activation": hit.activation,
                      "paths": [render_path(p) for p in hit.path]})
    return {"grain": hit.grain, "id": hit.id, "score": round(hit.score, 3), "path": parts, "reference": hit.reference}
