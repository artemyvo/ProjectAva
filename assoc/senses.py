"""Word pivots (ASSOCIATIVE_MEMORY.md §6): senses induced from the contexts a word is used
in, the anchor scorer over the warm set, the jump, and the two lexical bridge kinds that
reach words NOT in play — `root` (a shared stem) and `sound` (rhyme, edit distance, a
cross-script transliteration pair). Lexical end to end: the bridge is a word.

Senses: for every L1 term with enough chunk postings, cluster those chunks by their concept
signature (the L2 cells their words quantize to, the word's own cells removed) — classic
word-sense induction. Stored per build as `senses.json`; a word with one cluster has no
pivot.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .lex import is_identifier, is_stop, script_of

MIN_POSTINGS = 6
SENSE_THRESHOLD = 0.35          # cosine over concept signatures at which two contexts are one sense
MIN_SENSE_SIZE = 2
MAX_WORDS = 5000
ROOT_MIN_PREFIX = 5
SOUND_MIN_LEN, SOUND_MAX_EDIT, RHYME_MIN = 5, 2, 3

_TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i",
             "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
             "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "u", "я": "a"}


def transliterate(word: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in word.lower())


def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _centroid(vecs: list[dict]) -> dict:
    out: dict = defaultdict(float)
    for v in vecs:
        for k, x in v.items():
            out[k] += x / len(vecs)
    return dict(out)


def _add_into(acc: dict, v: dict) -> None:
    for k, x in v.items():
        acc[k] = acc.get(k, 0.0) + x


def induce_senses(glossary, codebook, chunk_text: dict[str, str], *, min_postings: int = MIN_POSTINGS,
                  threshold: float = SENSE_THRESHOLD) -> dict:
    """{word: {"senses": [{"cells": [(cell, members)], "postings": [chunk_id...], "size": n}],
               "split": cosine distance between the two densest senses}} for polysemous words."""
    out: dict = {}
    sig_cache: dict[str, dict] = {}

    def signature(cid: str, own_cells: set) -> dict:
        if cid not in sig_cache:
            sig_cache[cid] = codebook.cells_of_text(chunk_text.get(cid, ""))
        return {c: w for c, w in sig_cache[cid].items() if c not in own_cells}

    words = [t for t, p in glossary.postings.items()
             if not is_identifier(t) and not is_stop(t) and len(t) >= 3
             and sum(1 for i in p if glossary.ids_kind.get(i) == "chunk" and i in chunk_text) >= min_postings]
    for w in words[:MAX_WORDS]:
        own = {c for c, _w in codebook.assign.get(w, ())}
        postings = [i for i in glossary.postings[w] if glossary.ids_kind.get(i) == "chunk" and i in chunk_text]
        vecs = {cid: signature(cid, own) for cid in postings}
        vecs = {cid: v for cid, v in vecs.items() if v}
        if len(vecs) < min_postings:
            continue
        # Greedy average-link agglomeration: each context joins the cluster whose centroid is
        # nearest above the threshold, else starts one.
        # Cosine is scale-free, so a cluster's running SUM stands in for its centroid: a join
        # is one dict add, not a recomputation over every member (that was 70% of a rebuild).
        clusters: list[dict] = []
        for cid, v in vecs.items():
            best, best_s = None, -1.0
            for cl in clusters:
                s = _cos(v, cl["sum"])
                if s > best_s:
                    best, best_s = cl, s
            if best is not None and best_s >= threshold:
                best["members"].append(cid)
                _add_into(best["sum"], v)
            else:
                clusters.append({"members": [cid], "sum": dict(v)})
        # Second pass: merge clusters whose centroids are still within the threshold of each
        # other — the leader pass fragments a sense whose contexts arrived in an unlucky order.
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            best_pair, best_s = None, threshold
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = _cos(clusters[i]["sum"], clusters[j]["sum"])
                    if sim >= best_s:
                        best_pair, best_s = (i, j), sim
            if best_pair:
                i, j = best_pair
                clusters[i]["members"] += clusters[j]["members"]
                _add_into(clusters[i]["sum"], clusters[j]["sum"])
                clusters.pop(j)
                merged = True
        clusters = [c for c in clusters if len(c["members"]) >= MIN_SENSE_SIZE]
        if len(clusters) < 2:
            continue
        for c in clusters:
            c["centroid"] = {k: x / len(c["members"]) for k, x in c["sum"].items()}
        clusters.sort(key=lambda c: -len(c["members"]))
        senses = []
        for cl in clusters:
            top = sorted(cl["centroid"].items(), key=lambda kv: -kv[1])[:6]
            senses.append({"cells": [(int(c), codebook.members(int(c), 5)) for c, _w in top], "postings": cl["members"],
                           "size": len(cl["members"]), "centroid": {str(k): round(v, 4) for k, v in cl["centroid"].items()}})
        split = 1.0 - _cos(clusters[0]["centroid"], clusters[1]["centroid"])
        out[w] = {"senses": senses, "split": round(split, 4)}
    return out


def save_senses(senses: dict, dirpath: Path) -> None:
    (dirpath / "senses.json").write_text(json.dumps(senses, ensure_ascii=False), encoding="utf-8")


def load_senses(dirpath: Path) -> dict:
    p = dirpath / "senses.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ----- bridges -------------------------------------------------------------------------------

def root_bridges(word: str, vocabulary, *, min_prefix: int = ROOT_MIN_PREFIX) -> list[tuple[str, float]]:
    """Words sharing a stem-length prefix with *word* in the same script (колокол → колокольчик).
    An inflection of the same lemma (старый → старые) is the same word, not a bridge."""
    from .lex import lemmas
    out = []
    sc = script_of(word)
    own = set(lemmas(word))
    for w in vocabulary:
        if w == word or script_of(w) != sc or len(w) < min_prefix:
            continue
        if own & set(lemmas(w)):
            continue
        n = 0
        for a, b in zip(word, w):
            if a != b:
                break
            n += 1
        if n >= min_prefix:
            out.append((w, round(n / max(len(word), len(w)), 3)))
    return sorted(out, key=lambda x: -x[1])


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sound_bridges(word: str, vocabulary) -> list[tuple[str, float, str]]:
    """(word', strength, kind): rhyme (shared ending), near-spelling (edit distance), or a
    cross-script transliteration pair (магазин / magazine)."""
    out = []
    w = word.lower()
    sc = script_of(w)
    tw = transliterate(w) if sc == "cyr" else w
    for v in vocabulary:
        if v == w or len(v) < SOUND_MIN_LEN or len(w) < SOUND_MIN_LEN:
            continue
        sv = script_of(v)
        if sv == sc:
            if w[-RHYME_MIN:] == v[-RHYME_MIN:] and w[:-RHYME_MIN] != v[:-RHYME_MIN] and abs(len(w) - len(v)) <= 3:
                out.append((v, round(RHYME_MIN / max(len(w), len(v)) + 0.3, 3), "rhyme"))
                continue
            d = _edit(w, v)
            if d <= SOUND_MAX_EDIT:
                out.append((v, round(1.0 - d / max(len(w), len(v)), 3), "near"))
        elif {sc, sv} == {"cyr", "lat"}:
            tv = transliterate(v) if sv == "cyr" else v
            d = _edit(tw, tv)
            if d <= SOUND_MAX_EDIT:
                out.append((v, round(1.0 - d / max(len(tw), len(tv)), 3), "transliteration"))
    return sorted(out, key=lambda x: -x[1])


# ----- the operation ---------------------------------------------------------------------------

@dataclass
class Jump:
    bridge: str
    kind: str                       # sense | root | sound
    target: str                     # the far word (== bridge for a sense pivot)
    from_sense: list = field(default_factory=list)
    to_sense: list = field(default_factory=list)
    distance: float = 0.0
    score: float = 0.0
    seeds: list = field(default_factory=list)      # chunk ids of the far sense / target word
    hits: list = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {"bridge": self.bridge, "kind": self.kind, "target": self.target, "from_sense": self.from_sense,
                "to_sense": self.to_sense, "distance": self.distance, "score": self.score, "seeds": self.seeds[:8],
                "hits": [h.to_dict() if hasattr(h, "to_dict") else h for h in self.hits], "detail": self.detail}


def anchor_scores(context_terms: dict[str, float], senses: dict, glossary, codebook, context_signature: dict) -> list[dict]:
    """Score every polysemous word in the warm set (§6): act × split × idf × (1 − act(far sense))."""
    out = []
    for w, act in context_terms.items():
        info = senses.get(w)
        if not info or is_stop(w):
            continue
        idf = glossary.idf(w) if hasattr(glossary, "idf") else 1.0
        # Which sense is the context in? The one whose centroid is nearest the context signature.
        sims = [_cos({int(k): v for k, v in s["centroid"].items()}, context_signature) for s in info["senses"]]
        near = max(range(len(sims)), key=lambda i: sims[i])
        far = max((i for i in range(len(sims)) if i != near), key=lambda i: info["senses"][i]["size"])
        # "The far sense must be COLD": its activation relative to the near sense's — as warm as
        # the near one ⇒ nothing to switch to ⇒ the factor is zero.
        near_act, far_act = max(sims[near], 1e-6), max(0.0, sims[far])
        balance = min(far_act / near_act, 1.0)
        # A pivot needs somewhere to land: the far sense's material (log of its postings) is
        # the tiebreak between two anchors, ahead of IDF's preference for the rarer word.
        landing = math.log(1 + info["senses"][far]["size"])
        score = act * info["split"] * idf * (1.0 - balance) * landing
        out.append({"word": w, "score": round(score, 4), "near": near, "far": far, "far_act": round(far_act, 3),
                    "split": info["split"], "idf": round(idf, 3)})
    return sorted(out, key=lambda x: -x["score"])
