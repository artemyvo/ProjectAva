"""L1 — the glossary: term → postings (ASSOCIATIVE_MEMORY.md §2.1).

An inverted index over every chunk's text (whatever the kind) and every claim's text, keyed
on every candidate lemma of every token. Scoring is BM25 over the postings of a cue's terms;
a hit carries the terms that matched, so every match is explainable. Corpus-specific filler
is dropped by an IDF-based auto stop on top of the built-in stoplists.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .lex import LEX_VERSION, is_identifier, is_stop, lemmas, tokenize

GLOSSARY_VERSION = "glossary-1"
AUTO_STOP_DF = 0.6        # a term in more than this fraction of documents is filler
BM25_K1, BM25_B = 1.2, 0.75


class Glossary:
    def __init__(self):
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)   # term -> {id: tf}
        self.lengths: dict[str, int] = {}                               # id -> token count
        self.ids_kind: dict[str, str] = {}                              # id -> "chunk" | "claim"
        self.auto_stop: set[str] = set()
        self.avg_len = 1.0

    # ----- build ------------------------------------------------------------------------
    def add(self, ident: str, text: str, kind: str) -> None:
        n = 0
        posting_counts: dict[str, int] = defaultdict(int)
        for surface, _s, _e in tokenize(text):
            n += 1
            for t in lemmas(surface):
                if is_stop(t):
                    continue
                posting_counts[t] += 1
        for t, c in posting_counts.items():
            self.postings[t][ident] = c
        self.lengths[ident] = n
        self.ids_kind[ident] = kind

    def finalize(self) -> None:
        n_docs = max(len(self.lengths), 1)
        self.avg_len = (sum(self.lengths.values()) / n_docs) if self.lengths else 1.0
        self.auto_stop = {t for t, p in self.postings.items()
                          if len(p) / n_docs > AUTO_STOP_DF and not is_identifier(t) and n_docs >= 20}

    # ----- query ------------------------------------------------------------------------
    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        n = max(len(self.lengths), 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query_terms(self, cue: str) -> list[str]:
        out: list[str] = []
        for surface, _s, _e in tokenize(cue):
            for t in lemmas(surface):
                if is_stop(t) or t in self.auto_stop:
                    continue
                if t not in out:
                    out.append(t)
        return out

    def query_mass(self, cue: str) -> float:
        """The query's MATCHABLE BM25 mass: what a document containing every matchable
        query word once at average length would score (= Σ idf, one entry per surface
        token — a Russian word expands to surface + lemma and must not count twice). A
        word with no posting at all contributes nothing: on a cue in the other language
        it is untranslatable, not unmatched. A hit's share of this mass says how much of
        the query it matched — a top hit on one weak word out of twelve is not a full
        lexical match, while a hit on the cue's one proper noun is as good as lexical gets."""
        mass = 0.0
        for surface, _s, _e in tokenize(cue):
            best = 0.0
            for t in lemmas(surface):
                if is_stop(t) or t in self.auto_stop or not self.postings.get(t):
                    continue
                best = max(best, self.idf(t))
            mass += best
        return mass

    def search(self, cue: str, *, kind: Optional[str] = None, limit: int = 50,
               allowed: Optional[set[str]] = None) -> list[dict]:
        """BM25 hits: [{id, score, terms: [(term, contribution)], exact: bool}]."""
        terms = self.query_terms(cue)
        mass = self.query_mass(cue)
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list] = defaultdict(list)
        for t in terms:
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = self.idf(t)
            for ident, tf in plist.items():
                if kind and self.ids_kind.get(ident) != kind:
                    continue
                if allowed is not None and ident not in allowed:
                    continue
                dl = self.lengths.get(ident, 1)
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_len)
                contrib = idf * (tf * (BM25_K1 + 1)) / denom
                scores[ident] += contrib
                matched[ident].append((t, round(contrib, 3)))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        out = []
        for ident, sc in ranked:
            ms = matched[ident]
            exact = any(is_identifier(t) for t, _ in ms)
            out.append({"id": ident, "score": sc, "terms": ms, "exact": exact, "n_terms": len(terms), "mass": mass})
        return out

    def postings_of(self, term: str) -> dict[str, int]:
        return dict(self.postings.get(term, {}))

    # ----- persistence ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        obj = {"version": GLOSSARY_VERSION, "lex": LEX_VERSION, "postings": self.postings,
               "lengths": self.lengths, "ids_kind": self.ids_kind, "auto_stop": sorted(self.auto_stop),
               "avg_len": self.avg_len}
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Glossary":
        g = Glossary()
        obj = json.loads(path.read_text(encoding="utf-8"))
        g.postings = defaultdict(dict, {t: dict(p) for t, p in obj["postings"].items()})
        g.lengths = obj["lengths"]
        g.ids_kind = obj["ids_kind"]
        g.auto_stop = set(obj.get("auto_stop") or [])
        g.avg_len = obj.get("avg_len", 1.0)
        return g

    def stats(self) -> dict:
        return {"terms": len(self.postings), "ids": len(self.lengths), "auto_stop": len(self.auto_stop)}
