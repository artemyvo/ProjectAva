"""Dedup tiers with the equivalence check (ASSOCIATIVE_MEMORY.md §2.5).

Tier 1 (exact) is the fold's own key. Here: tier 2 nominates pairs with the same subject
and the same L2 cell set; tier 3 nominates by claim-embedding cosine within a subject
block. Every nomination must pass the **equivalence check** before it merges — polarity,
version, conditions, argument order over resolved node ids. A pair failing on polarity
becomes a `contests` link; on version, `supersedes`; on order or condition it stays two
claims. The check errs toward not merging: a false merge hides a fact.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from .lex import tokenize

TIER3_MIN_COS = 0.86
NEGATION = frozenset("""not never no none nothing neither nor cannot can't don't doesn't didn't isn't aren't wasn't
weren't won't wouldn't shouldn't couldn't without removed deprecated disabled unsupported
не нет ни никогда нельзя без невозможно удалён удалено удалена отключён отключено""".split())
CONDITION = frozenset("if when unless until only whenever provided если когда пока только при".split())
SINGLE_VALUED = frozenset({"defaults_to", "lives_in", "founded", "located_in", "born_in", "ceo_of", "version_of"})


def _words(text: str) -> list[str]:
    return [s.lower() for s, _a, _b in tokenize(text)]


def polarity(text: str, kind_markers: frozenset = frozenset()) -> int:
    w = _words(text)
    n = sum(1 for x in w if x in NEGATION or x in kind_markers)
    return n % 2


def has_condition(text: str) -> bool:
    return any(x in CONDITION for x in _words(text))


def argument_order(claim: dict, resolve: Callable[[str], str]) -> tuple[str, ...]:
    """Resolved node ids in text order: the subject first, then each entity mention where
    it appears in the text. Never surface strings (an EN line and its RU restatement share
    none), so the resolver maps a mention to a node — an alias-table key or an L2 cell."""
    text = claim.get("text") or ""
    low = text.lower()
    seq: list[tuple[int, str]] = []
    subj = claim.get("subject") or ""
    if subj:
        seq.append((-1, subj))
    for e in claim.get("entities") or []:
        pos = low.find(str(e).lower())
        seq.append((pos if pos >= 0 else 10_000, resolve(str(e))))
    return tuple(n for _p, n in sorted(seq))


def equivalent(a: dict, b: dict, *, resolve: Callable[[str], str], kind_markers: frozenset = frozenset()) -> str:
    """'merge' | 'contests' | 'supersedes' | 'separate'."""
    if polarity(a["text"], kind_markers) != polarity(b["text"], kind_markers):
        return "contests"
    va, vb = str(a.get("version") or ""), str(b.get("version") or "")
    if va and vb and va != vb:
        return "supersedes"
    if has_condition(a["text"]) != has_condition(b["text"]):
        return "separate"
    if argument_order(a, resolve) != argument_order(b, resolve):
        return "separate"
    return "merge"


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def nominate(claims: dict[str, dict], *, cell_set: Callable[[str], frozenset],
             embed: Optional[Callable[[list[str]], np.ndarray]] = None, tier3_min: float = TIER3_MIN_COS) -> list[tuple[str, str, str]]:
    """Candidate pairs (id_a, id_b, tier) from tiers 2 and 3, within a subject block."""
    blocks: dict[str, list[str]] = defaultdict(list)
    for cid, c in claims.items():
        if cid.startswith("rm-"):
            continue
        blocks[(c.get("subject") or "") + "\x1f" + (c.get("facet") or "")].append(cid)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, ids in blocks.items():
        if len(ids) < 2 or not key.split("\x1f")[0]:
            continue
        # Tier 2: same cell set.
        by_cells: dict[frozenset, list[str]] = defaultdict(list)
        for cid in ids:
            cs = cell_set(claims[cid]["text"])
            if cs:
                by_cells[cs].append(cid)
        for group in by_cells.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = sorted((group[i], group[j]))
                    if (a, b) not in seen:
                        seen.add((a, b))
                        pairs.append((a, b, "tier2"))
        # Tier 3: embedding cosine within the block.
        if embed is not None and 2 <= len(ids) <= 400:
            V = embed([claims[c]["text"] for c in ids])
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            S = V @ V.T
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if S[i, j] >= tier3_min:
                        a, b = sorted((ids[i], ids[j]))
                        if (a, b) not in seen:
                            seen.add((a, b))
                            pairs.append((a, b, "tier3"))
    return pairs


def resolve_pairs(claims: dict[str, dict], pairs: list[tuple[str, str, str]], *, resolve: Callable[[str], str],
                  kind_markers: Callable[[str], frozenset]) -> dict:
    """Run the equivalence check over nominations; merge what passes; link the rest."""
    uf = UnionFind()
    contests: list[dict] = []
    supersedes: list[dict] = []
    separate = 0
    merged_pairs = 0
    for a, b, tier in pairs:
        ca, cb = claims[a], claims[b]
        verdict = equivalent(ca, cb, resolve=resolve, kind_markers=kind_markers(ca.get("kind") or ""))
        if verdict == "merge":
            uf.union(a, b)
            merged_pairs += 1
        elif verdict == "contests":
            contests.append({"a": a, "b": b, "tier": tier})
        elif verdict == "supersedes":
            older, newer = (a, b) if str(ca.get("version")) < str(cb.get("version")) else (b, a)
            supersedes.append({"older": older, "newer": newer, "tier": tier})
        else:
            separate += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in claims:
        if cid in uf.p:
            groups[uf.find(cid)].append(cid)
    merged = {root: ids for root, ids in groups.items() if len(ids) > 1}
    return {"merged": merged, "contests": contests, "supersedes": supersedes,
            "stats": {"nominated": len(pairs), "merged_pairs": merged_pairs, "contests": len(contests),
                      "supersedes": len(supersedes), "separate": separate, "merged_groups": len(merged)}}


def rel_contests(claims: dict[str, dict]) -> list[dict]:
    """Same (subject, single-valued predicate) with different objects → contests (§2.5)."""
    by: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for cid, c in claims.items():
        rel = c.get("rel")
        if not rel or len(rel) != 3 or rel[0] not in SINGLE_VALUED:
            continue
        by[(rel[1], rel[0])][rel[2].lower()] = cid
    out: list[dict] = []
    for (subj, pred), objs in by.items():
        ids = list(objs.values())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                out.append({"a": ids[i], "b": ids[j], "tier": "rel:" + pred})
    return out


def apply_merges(claims: dict[str, dict], merged: dict[str, list[str]]) -> dict[str, str]:
    """Fold merged groups into their root claim; return {old_id: root_id}."""
    remap: dict[str, str] = {}
    for root, ids in merged.items():
        keep = claims[root]
        for cid in ids:
            if cid == root:
                continue
            c = claims.pop(cid)
            remap[cid] = root
            keep["occurrences"].extend(c["occurrences"])
            keep["sources"] = sorted(set(keep["sources"]) | set(c["sources"]))
            keep["visible_in"] = sorted(set(keep.get("visible_in") or []) | set(c.get("visible_in") or []))
            for e in c.get("entities") or []:
                if e not in keep["entities"]:
                    keep["entities"].append(e)
            keep.setdefault("merged_from", []).append(cid)
        keep["n_sources"] = len(keep["sources"])
        variants: dict[str, set] = defaultdict(set)
        first: dict[str, str] = {}
        for o in keep["occurrences"]:
            variants[o["text"]].add(o["key"])
            first.setdefault(o["text"], o.get("asserted_at", ""))
        keep["text"] = sorted(variants.items(), key=lambda kv: (-len(kv[1]), first[kv[0]]))[0][0]
        keep["variants"] = sorted(variants)
    return remap
