"""Families and authority (ASSOCIATIVE_MEMORY.md §2.5).

Families: documents grouped by `doc_key` first (the versions of one page are one family
by construction), then by near-duplicate detection over their chunk text (shingle
Jaccard), so a syndicated copy or a per-platform mirror votes once.

Authority: a damped bipartite PageRank over families and claims with a per-kind prior —
the teleport term the first draft lacked, so a corpus of unique pages does not converge
to zero — renormalized per iteration and rank-normalized to [0, 1].
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

SHINGLE_N = 5
FAMILY_JACCARD = 0.8
BETA = 0.85
ITER = 30
PRIOR_BY_KIND = {"tech_doc": 1.0, "structured": 1.0, "code": 0.9, "article": 0.6, "news": 0.6, "chat": 0.3}

_W = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str) -> set[int]:
    w = [x.lower() for x in _W.findall(text)]
    return {hash(" ".join(w[i:i + SHINGLE_N])) for i in range(max(len(w) - SHINGLE_N + 1, 0))}


def families(docs: list, *, jaccard: float = FAMILY_JACCARD) -> dict[str, str]:
    """doc_id -> family id. Union by key, then by shingle Jaccard over the document text."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_key: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        by_key[d.key].append(d.doc_id)
    for ids in by_key.values():
        for other in ids[1:]:
            union(ids[0], other)
    # Near-duplicates across keys: shingles over the rendered text (chunk grain would do
    # the same; the document is fewer sets).
    sh = {d.doc_id: _shingles(d.text) for d in docs if d.text}
    ids = list(sh)
    for i in range(len(ids)):
        a = sh[ids[i]]
        if not a:
            continue
        for j in range(i + 1, len(ids)):
            b = sh[ids[j]]
            if not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            if inter / len(a | b) >= jaccard:
                union(ids[i], ids[j])
    return {d.doc_id: find(d.doc_id) for d in docs}


def authority(claims: dict[str, dict], fam_of_doc: dict[str, str], kind_of_doc: dict[str, str], *,
              beta: float = BETA, iters: int = ITER) -> dict:
    """Returns {"claims": {claim_id: auth in [0,1]}, "families": {fam: auth}, "independent": {claim_id: n}}."""
    fam_claims: dict[str, set[str]] = defaultdict(set)
    claim_fams: dict[str, set[str]] = defaultdict(set)
    fam_kind: dict[str, str] = {}
    for cid, c in claims.items():
        for o in c.get("occurrences") or []:
            f = fam_of_doc.get(o.get("doc_id"))
            if f is None:
                continue
            fam_claims[f].add(cid)
            claim_fams[cid].add(f)
            fam_kind.setdefault(f, kind_of_doc.get(o.get("doc_id"), ""))
    fams = list(fam_claims)
    if not fams or not claims:
        return {"claims": {cid: 0.0 for cid in claims}, "families": {}, "independent": {cid: len(claim_fams[cid]) for cid in claims}}
    prior_f = {f: PRIOR_BY_KIND.get(fam_kind.get(f, ""), 0.5) for f in fams}
    prior_c = {cid: PRIOR_BY_KIND.get(c.get("kind", ""), 0.5) for cid, c in claims.items()}
    corr = {cid: math.log(1 + len(claim_fams[cid])) for cid in claims}

    def norm(d: dict) -> dict:
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in d.items()}

    af = norm(dict(prior_f))
    ac = norm(dict(prior_c))
    pf = norm(prior_f)
    pc = norm(prior_c)
    for _ in range(iters):
        new_c = {}
        for cid in claims:
            s = sum(af[f] / max(len(fam_claims[f]), 1) for f in claim_fams[cid])
            new_c[cid] = (1 - beta) * pc[cid] + beta * s
        new_c = norm(new_c)
        new_f = {}
        for f in fams:
            s = sum(new_c[cid] * corr[cid] for cid in fam_claims[f])
            new_f[f] = (1 - beta) * pf[f] + beta * s
        af, ac = norm(new_f), new_c
    # Rank-normalize to [0, 1].
    order = sorted(ac.items(), key=lambda kv: kv[1])
    n = max(len(order) - 1, 1)
    ranked = {cid: i / n for i, (cid, _v) in enumerate(order)}
    return {"claims": ranked, "families": af, "independent": {cid: len(claim_fams[cid]) for cid in claims},
            "family_kind": fam_kind}
