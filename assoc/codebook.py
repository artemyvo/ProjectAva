"""L2 — the codebook (ASSOCIATIVE_MEMORY.md §2.2): the inverted index keyed on a CELL in
term-embedding space rather than on a word, so a Russian cue and an English claim share a
key and the match is explainable (`карьера → cell#41 {career, position, вакансия}`).

- Vocabulary: the canonical lemmas of every chunk and claim (identifiers excluded — L2 has
  nothing to add to an exact token) plus the multi-word entity mentions claims carry.
- Cells: leader clustering by cosine over the term embeddings, most frequent terms first
  (deterministic; O(n · cells), fine to ~20k terms; k-means is the documented scale-out).
- Soft assignment: a term joins its nearest cell and any cell within `margin` of it, at a
  discount; neighbour cells (kNN over centroids) expand a query at a further discount.
- Drift (§2.8): on a fast rebuild new terms are assigned to the old cells; the fraction
  assigned below the margin and the cells whose membership doubled are the drift measures.
- Alias proposals: two entity mentions in one cell above a high threshold are written for a
  human to promote — proposal, never merge.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .lex import is_identifier, is_stop, lemmas, tokenize

CODEBOOK_VERSION = "codebook-1"
DEFAULT_THRESHOLD = 0.72        # cosine for a term to join a cell (коньяк~cognac 0.745; коньяк~brandy 0.62 stays apart)
DEFAULT_MARGIN = 0.04           # a second cell within this of the best also gets the term, discounted
SOFT_DISCOUNT = 0.6
NEIGHBOUR_K, NEIGHBOUR_MIN, NEIGHBOUR_DISCOUNT = 3, 0.80, 0.0   # expansion off by default: measured as pure noise
MIN_CELL_MEMBERS = 2            # a one-member cell is L1 again, with less precision — it carries no concept
ALIAS_MIN_SIM = 0.85
MAX_VOCAB = 20_000
MIN_TERM_CHARS = 3


def canonical(surface: str) -> Optional[str]:
    """The one lemma L2 keys a token on, or None for what L2 does not index."""
    if is_identifier(surface):
        return None
    ls = lemmas(surface)
    t = ls[-1] if len(ls) > 1 else ls[0]
    if is_stop(t) or len(t) < MIN_TERM_CHARS or t.replace(".", "").replace(",", "").isdigit():
        return None
    return t


def text_terms(text: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for surface, _s, _e in tokenize(text):
        t = canonical(surface)
        if t:
            out[t] += 1
    return dict(out)


class Codebook:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, margin: float = DEFAULT_MARGIN, embedder_id: str = ""):
        self.threshold = threshold
        self.margin = margin
        self.embedder_id = embedder_id
        self.cells: list[dict] = []                 # {id, members: [term...], centroid: np.ndarray}
        self.assign: dict[str, list[tuple[int, float]]] = {}   # term -> [(cell, weight)]
        self.postings: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)   # cell -> {id: {term: weight}}
        self.ids_kind: dict[str, str] = {}
        self.n_ids = 0
        self.neighbours: dict[int, list[tuple[int, float]]] = {}
        self.drift: dict = {"forced": 0, "assigned": 0, "doubled": 0, "since_recluster": 0}
        self._members_at_recluster: dict[int, int] = {}
        self._query_cache: dict[str, list[tuple[int, float]]] = {}

    # ----- build ------------------------------------------------------------------------
    def _centroids(self) -> np.ndarray:
        if not self.cells:
            return np.zeros((0, 1), dtype="float32")
        return np.stack([c["centroid"] for c in self.cells])

    def cluster(self, terms: list[str], vecs: np.ndarray) -> None:
        """Leader clustering over *terms* (ordered most frequent first), from scratch."""
        self.cells, self.assign = [], {}
        n = len(terms)
        dim = vecs.shape[1] if n else 1
        # One preallocated matrix for the leaders (a fresh np.stack per term was quadratic).
        cents = np.zeros((n, dim), dtype="float32")
        sums = np.zeros((n, dim), dtype="float32")
        k = 0
        for term, v in zip(terms, vecs):
            v = v / (np.linalg.norm(v) or 1.0)
            if k:
                sims = cents[:k] @ v
                best = int(np.argmax(sims))
                if sims[best] >= self.threshold:
                    self.cells[best]["members"].append(term)
                    sums[best] += v
                    cents[best] = sums[best] / (np.linalg.norm(sums[best]) or 1.0)
                    continue
            self.cells.append({"id": k, "members": [term]})
            cents[k] = v
            sums[k] = v
            k += 1
        for i, c in enumerate(self.cells):
            c["centroid"] = cents[i].copy()
        # Soft assignment pass over the final centroids.
        C = self._centroids()
        for term, v in zip(terms, vecs):
            self.assign[term] = self._soft(v, C)
        self._members_at_recluster = {c["id"]: len(c["members"]) for c in self.cells}
        self.drift = {"forced": 0, "assigned": 0, "doubled": 0, "since_recluster": 0}
        self._build_neighbours()

    def _soft(self, v: np.ndarray, C: np.ndarray) -> list[tuple[int, float]]:
        v = v / (np.linalg.norm(v) or 1.0)
        sims = C @ v
        best = int(np.argmax(sims))
        out = [(best, 1.0)]
        for j in np.argsort(-sims)[1:4]:
            j = int(j)
            if sims[best] - sims[j] <= self.margin and sims[j] >= self.threshold - self.margin:
                out.append((j, SOFT_DISCOUNT))
        return out

    def assign_new(self, terms: list[str], vecs: np.ndarray) -> None:
        """Fast-rebuild path: place new terms into the existing cells, measuring drift."""
        if not self.cells:
            self.cluster(terms, vecs)
            return
        C = self._centroids()
        for term, v in zip(terms, vecs):
            if term in self.assign:
                continue
            v = v / (np.linalg.norm(v) or 1.0)
            sims = C @ v
            best = int(np.argmax(sims))
            self.drift["assigned"] += 1
            self.drift["since_recluster"] += 1
            if sims[best] < self.threshold:
                # Forced into the nearest cell — the drift signal (§2.8).
                self.drift["forced"] += 1
            self.cells[best]["members"].append(term)
            self.assign[term] = self._soft(v, C)
        self.drift["doubled"] = sum(1 for c in self.cells
                                    if len(c["members"]) >= 2 * max(self._members_at_recluster.get(c["id"], 1), 1) and len(c["members"]) >= 6)

    def _build_neighbours(self) -> None:
        C = self._centroids()
        self.neighbours = {}
        if len(self.cells) < 2:
            return
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        for i in range(len(self.cells)):
            order = np.argsort(-S[i])[:NEIGHBOUR_K]
            self.neighbours[i] = [(int(j), float(S[i, j])) for j in order if S[i, j] >= NEIGHBOUR_MIN]

    def index(self, ident: str, text: str, kind: str) -> None:
        self.ids_kind[ident] = kind
        for term, tf in text_terms(text).items():
            for cell, w in self.assign.get(term, ()):
                d = self.postings[cell].setdefault(ident, {})
                d[term] = d.get(term, 0.0) + tf * w
        self.n_ids = len(self.ids_kind)

    # ----- read -------------------------------------------------------------------------
    def cells_of_text(self, text: str, embedder=None) -> dict[int, float]:
        """The concept signature of a text: cell -> weight. With an *embedder*, terms the
        corpus never used are placed into the existing cells (a translated fact over a
        chunk in the other language, §1.6 correction m9)."""
        q = text_terms(text)
        assigned = self.assign_query_terms(list(q), embedder) if embedder is not None else {t: self.assign.get(t, []) for t in q}
        out: dict[int, float] = defaultdict(float)
        for term, tf in q.items():
            for cell, w in assigned.get(term, ()):
                out[cell] += tf * w
        return dict(out)

    def cell_set(self, text: str) -> frozenset:
        return frozenset(c for c, (cell, w) in ((c, (c, w)) for c, w in self.cells_of_text(text).items()) if w >= 1.0)

    def idf(self, cell: int) -> float:
        df = len(self.postings.get(cell, ()))
        n = max(self.n_ids, 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def members(self, cell: int, limit: int = 6) -> list[str]:
        return list(self.cells[cell]["members"][:limit]) if 0 <= cell < len(self.cells) else []

    def assign_query_terms(self, terms: list[str], embedder) -> dict[str, list[tuple[int, float]]]:
        """Place cue terms the corpus never used (an English cue over a Russian corpus) into
        the existing cells at query time — the cross-lingual key for a word that occurs only
        in the other language. Nearest cell at or above the threshold, else nothing."""
        out: dict[str, list[tuple[int, float]]] = {}
        unknown = [t for t in terms if t not in self.assign and t not in self._query_cache]
        if unknown and embedder is not None and self.cells:
            C = self._centroids()
            vecs = embedder.encode(unknown)
            for t, v in zip(unknown, vecs):
                v = np.asarray(v, dtype="float32")
                v = v / (np.linalg.norm(v) or 1.0)
                sims = C @ v
                best = int(np.argmax(sims))
                self._query_cache[t] = [(best, 1.0)] if sims[best] >= self.threshold else []
        for t in terms:
            out[t] = self.assign.get(t) or self._query_cache.get(t, [])
        return out

    def search(self, cue: str, *, kind: Optional[str] = None, limit: int = 50, allowed: Optional[set[str]] = None,
               embedder=None) -> list[dict]:
        """Concept-channel hits: [{id, score, cells: [(cell, members, contribution)]}]."""
        q = text_terms(cue)
        assigned = self.assign_query_terms(list(q), embedder)
        # cell -> (weight, the cue terms that reached it). A cell reached by a cue term the
        # corpus never used (query-time assignment) counts that term as a member.
        cell_w: dict[int, float] = defaultdict(float)
        cell_terms: dict[int, set[str]] = defaultdict(set)
        cell_extra: dict[int, int] = defaultdict(int)
        for term, tf in q.items():
            for cell, w in assigned.get(term, ()):
                cell_w[cell] += w
                cell_terms[cell].add(term)
                if term not in self.assign:
                    cell_extra[cell] += 1
                if NEIGHBOUR_DISCOUNT > 0:
                    for nb, sim in self.neighbours.get(cell, ()):
                        cell_w[nb] += w * NEIGHBOUR_DISCOUNT * sim
        scores: dict[str, float] = defaultdict(float)
        via: dict[str, list] = defaultdict(list)
        covered: dict[str, set[str]] = defaultdict(set)
        for cell, w in cell_w.items():
            plist = self.postings.get(cell)
            if not plist:
                continue
            n_members = len(self.cells[cell]["members"]) + cell_extra.get(cell, 0)
            if n_members < MIN_CELL_MEMBERS:
                continue
            idf = self.idf(cell)
            # A cell that covers most of the corpus is filler at the concept grain too, and a
            # big cell is a blob of loosely related words rather than one concept: its vote
            # is discounted by size (2 members → 0.59, 25 → 0.24).
            if len(plist) / max(self.n_ids, 1) > 0.5:
                continue
            specificity = 1.0 / (1.0 + math.log(n_members))
            qterms = cell_terms[cell]
            for ident, by_term in plist.items():
                if kind and self.ids_kind.get(ident) != kind:
                    continue
                if allowed is not None and ident not in allowed:
                    continue
                # Only a match through a DIFFERENT member is concept evidence: the cue's own
                # word matching itself is what L1 already scored.
                tf = sum(v for t, v in by_term.items() if t not in qterms)
                if tf <= 0:
                    continue
                contrib = w * idf * specificity * (tf / (tf + 1.0))
                scores[ident] += contrib
                covered[ident] |= qterms
                via[ident].append((cell, self.members(cell), round(contrib, 3)))
        # Coverage: a cue of five terms of which one found a cell is weak concept evidence;
        # three of five is strong (Lucene's coordination factor, applied to the sum).
        n_q = max(len(q), 1)
        for ident in scores:
            scores[ident] *= math.sqrt(len(covered[ident]) / n_q)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        return [{"id": i, "score": s, "cells": via[i]} for i, s in ranked]

    def alias_proposals(self, mentions: dict[str, str]) -> list[dict]:
        """*mentions*: surface -> node id. Two mentions of different nodes in one cell,
        close to each other above ALIAS_MIN_SIM, are proposed as aliases."""
        by_cell: dict[int, list[tuple[str, str]]] = defaultdict(list)
        vec: dict[str, np.ndarray] = {}
        for m, node in mentions.items():
            t = canonical(m) if " " not in m else m.lower()
            if not t:
                continue
            a = self.assign.get(t)
            if a:
                by_cell[a[0][0]].append((m, node))
        out: list[dict] = []
        for cell, items in by_cell.items():
            nodes = {n for _, n in items}
            if len(nodes) < 2:
                continue
            out.append({"cell": cell, "members": self.members(cell), "mentions": sorted({m for m, _ in items}),
                        "nodes": sorted(nodes)})
        return out

    # ----- persistence ------------------------------------------------------------------
    def save(self, dirpath: Path) -> None:
        obj = {"version": CODEBOOK_VERSION, "embedder": self.embedder_id, "threshold": self.threshold, "margin": self.margin,
               "cells": [{"id": c["id"], "members": c["members"]} for c in self.cells],
               "assign": {t: [[int(c), float(w)] for c, w in a] for t, a in self.assign.items()},
               "postings": {str(c): p for c, p in self.postings.items()}, "ids_kind": self.ids_kind, "n_ids": self.n_ids,
               "neighbours": {str(c): [[int(j), float(s)] for j, s in n] for c, n in self.neighbours.items()},
               "drift": self.drift, "members_at_recluster": {str(k): v for k, v in self._members_at_recluster.items()}}
        (dirpath / "codebook.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        if self.cells:
            np.save(dirpath / "codebook_centroids.npy", self._centroids())

    @staticmethod
    def load(dirpath: Path) -> "Codebook":
        obj = json.loads((dirpath / "codebook.json").read_text(encoding="utf-8"))
        cb = Codebook(obj["threshold"], obj["margin"], obj.get("embedder", ""))
        cents = np.load(dirpath / "codebook_centroids.npy") if (dirpath / "codebook_centroids.npy").exists() else None
        cb.cells = [{"id": c["id"], "members": c["members"], "centroid": (cents[i] if cents is not None else None)}
                    for i, c in enumerate(obj["cells"])]
        cb.assign = {t: [(int(c), float(w)) for c, w in a] for t, a in obj["assign"].items()}
        cb.postings = defaultdict(dict, {int(c): p for c, p in obj["postings"].items()})
        cb.ids_kind = obj["ids_kind"]
        cb.n_ids = obj.get("n_ids", len(cb.ids_kind))
        cb.neighbours = {int(c): [(int(j), float(s)) for j, s in n] for c, n in obj.get("neighbours", {}).items()}
        cb.drift = obj.get("drift", {})
        cb._members_at_recluster = {int(k): v for k, v in obj.get("members_at_recluster", {}).items()}
        return cb

    def stats(self) -> dict:
        sizes = [len(c["members"]) for c in self.cells]
        return {"terms": len(self.assign), "cells": len(self.cells), "largest_cell": max(sizes) if sizes else 0,
                "multi_member_cells": sum(1 for s in sizes if s > 1), "drift": dict(self.drift)}


def vocabulary(texts: list[str], extra_mentions: list[str] = ()) -> list[str]:
    """Terms ordered most frequent first (the leader clustering's order)."""
    df: dict[str, int] = defaultdict(int)
    for t in texts:
        for term in text_terms(t):
            df[term] += 1
    for m in extra_mentions:
        m = " ".join(m.split()).strip()
        # Multi-word mentions only, no digits (table cells like "timeout 105" are values, not
        # concepts), at most three words.
        if m and " " in m and len(m) <= 60 and not any(ch.isdigit() for ch in m) and len(m.split()) <= 3:
            df[m.lower()] += 1
    return [t for t, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))][:MAX_VOCAB]
