"""The milestone-1 fold: occurrences → claims (exact tier only), nodes by namespace,
supersession between versions of one key (ASSOCIATIVE_MEMORY.md §2.3, §2.5 tier 1).

Everything here is derived. A claim keeps every occurrence; the representative wording is
the one stated by the most documents (ties to the oldest). Facets come from the kind's map.
Milestone 2 adds the paraphrase tiers, families and authority behind `claim_key`.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Optional

from . import kinds as kinds_mod
from .chunks import norm_text

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def claim_key(text: str, facet: str, subject: str) -> str:
    base = _PUNCT.sub(" ", norm_text(text)).strip()
    return hashlib.sha1(f"{facet}\x1f{subject}\x1f{base}".encode("utf-8")).hexdigest()[:16]


def fold_claims(documents: list, *, current_ids: set[str]) -> dict:
    """Build the claim table over the *current* documents' protocols.

    Returns {"claims": {claim_id: claim}, "by_chunk": {chunk_id: [claim_id]},
             "by_subject": {subject: [claim_id]}, "stats": {...}}.
    """
    claims: dict[str, dict] = {}
    by_chunk: dict[str, list[str]] = defaultdict(list)
    by_subject: dict[str, list[str]] = defaultdict(list)
    n_occ = n_ungrounded = 0
    for doc in documents:
        if doc.doc_id not in current_ids or not doc.facts:
            continue
        spec = kinds_mod.get(doc.kind)
        for i, f in enumerate(doc.facts.get("facts") or []):
            n_occ += 1
            if f.get("grounded") is False:
                n_ungrounded += 1
                continue
            facet = spec.facet_map.get(f.get("fact_class") or "unspecified", "unclassified")
            if f.get("need"):
                facet = "need"
            subject = str(f.get("subject") or "")
            cid = claim_key(f["text"], facet, subject)
            occ = {"doc_id": doc.doc_id, "key": doc.key, "kind": doc.kind, "chunk_id": f.get("chunk_id"),
                   "span": f.get("span"), "index": i, "text": f["text"], "subject_raw": f.get("subject_raw"),
                   "asserted_at": str(doc.meta.get("date") or doc.meta.get("version") or doc.meta.get("ingested_at") or ""),
                   "version": str(f.get("version") or doc.meta.get("version") or ""),
                   "speaker": (doc.meta.get("user") or "") if doc.kind == "chat" else "",
                   "title": doc.meta.get("title") or doc.key, "when": f.get("when") or ""}
            c = claims.get(cid)
            if c is None:
                c = claims[cid] = {"claim_id": cid, "text": f["text"], "facet": facet, "fact_class": f.get("fact_class"),
                                   "subject": subject, "subject_raw": f.get("subject_raw") or "", "entities": list(f.get("entities") or []),
                                   "kind": doc.kind, "occurrences": [], "sources": set(), "rel": f.get("rel"),
                                   "when": f.get("when") or "", "version": occ["version"], "scope": _scope_facets(doc.meta)}
            c["occurrences"].append(occ)
            c["sources"].add(doc.key)
            for e in f.get("entities") or []:
                if e not in c["entities"]:
                    c["entities"].append(e)
            if f.get("chunk_id"):
                if cid not in by_chunk[f["chunk_id"]]:
                    by_chunk[f["chunk_id"]].append(cid)
            if subject and cid not in by_subject[subject]:
                by_subject[subject].append(cid)
    for c in claims.values():
        c["sources"] = sorted(c["sources"])
        c["n_sources"] = len(c["sources"])
        # Representative wording: the variant stated by the most documents, ties to the oldest.
        variants: dict[str, set] = defaultdict(set)
        first: dict[str, str] = {}
        for o in c["occurrences"]:
            variants[o["text"]].add(o["key"])
            first.setdefault(o["text"], o["asserted_at"])
        c["text"] = sorted(variants.items(), key=lambda kv: (-len(kv[1]), first[kv[0]]))[0][0]
        c["variants"] = sorted(variants.keys())
    return {"claims": claims, "by_chunk": dict(by_chunk), "by_subject": dict(by_subject),
            "stats": {"occurrences": n_occ, "claims": len(claims), "ungrounded": n_ungrounded,
                      "by_facet": _count(c["facet"] for c in claims.values())}}


def _scope_facets(meta: dict) -> dict:
    return {k: meta[k] for k in ("product", "version", "platform", "edition", "branch", "tenant", "conversation") if meta.get(k) is not None}


def _count(it) -> dict:
    d: dict = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


def supersession(store, *, current_ids: set[str]) -> dict:
    """For each current document that supersedes an earlier version of its key: claims
    present in the old version and absent in the new one yield a *removed in <version>*
    fact (§1.4 consequence 1). Returns {"removed": [claim-like dicts]}."""
    removed: list[dict] = []
    for doc_id in current_ids:
        meta = store.meta(doc_id) or {}
        prev_id = meta.get("supersedes")
        if not prev_id or meta.get("append_only"):
            continue
        new = store.document(doc_id)
        old = store.document(prev_id)
        if not new or not old or not old.facts or not new.facts:
            continue
        spec = kinds_mod.get(new.kind)
        new_keys = {claim_key(f["text"], spec.facet_map.get(f.get("fact_class") or "unspecified", "unclassified"), str(f.get("subject") or ""))
                    for f in new.facts.get("facts") or [] if f.get("grounded") is not False}
        for f in old.facts.get("facts") or []:
            if f.get("grounded") is False:
                continue
            facet = spec.facet_map.get(f.get("fact_class") or "unspecified", "unclassified")
            k = claim_key(f["text"], facet, str(f.get("subject") or ""))
            if k in new_keys:
                continue
            removed.append({"claim_id": "rm-" + k, "text": f"Removed in {new.meta.get('version')}: {f['text']}",
                            "facet": "event", "fact_class": "event", "subject": str(f.get("subject") or ""),
                            "subject_raw": f.get("subject_raw") or "", "entities": list(f.get("entities") or []),
                            "kind": new.kind, "when": str(new.meta.get("version") or ""), "version": str(new.meta.get("version") or ""),
                            "occurrences": [{"doc_id": old.doc_id, "key": old.key, "kind": old.kind, "chunk_id": f.get("chunk_id"),
                                             "span": f.get("span"), "text": f["text"], "title": old.meta.get("title") or old.key,
                                             "asserted_at": str(old.meta.get("version") or ""), "version": str(old.meta.get("version") or ""),
                                             "speaker": "", "when": ""}],
                            "sources": [old.key], "n_sources": 1, "variants": [f["text"]], "rel": None,
                            "scope": _scope_facets(new.meta), "superseded_from": prev_id,
                            "visible_in": [new.doc_id]})
    return {"removed": removed}
