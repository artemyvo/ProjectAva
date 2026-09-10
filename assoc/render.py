"""Rendering (ASSOCIATIVE_MEMORY.md §3.2): a claim line by facet, a passage as an attributed
quotation with its reference, a gist labelled as generated text. Everything rendered is
copied from the store; nothing here is generated.
"""

from __future__ import annotations

from typing import Optional

from .puller import Hit
from .store import Store


def _date(s: str) -> str:
    return (s or "")[:10]


def contest_line(claim: dict, other: dict, *, allowed_doc_ids: Optional[set[str]] = None) -> str:
    """A contested claim renders with BOTH sides and their counts, never the winner alone (§2.5)."""
    a = claim_line(claim, allowed_doc_ids=allowed_doc_ids)
    b = claim_line(other, allowed_doc_ids=allowed_doc_ids)
    ia, ib = claim.get("n_independent", 1), other.get("n_independent", 1)
    return f"{a} — CONTESTED by ({ib} independent): {b}" if "independent" in a or ia == 1 else f"{a} — CONTESTED: {b}"


def claim_line(claim: dict, *, allowed_doc_ids: Optional[set[str]] = None) -> str:
    occ = claim.get("occurrences") or [{}]
    o = next((x for x in occ if allowed_doc_ids is None or x.get("doc_id") in allowed_doc_ids), occ[0])
    facet = claim.get("facet")
    text = claim["text"]
    n = claim.get("n_sources", 1)
    title = o.get("title") or o.get("key") or ""
    if facet == "position":
        who = claim.get("subject_raw") or o.get("speaker") or "someone"
        return f"{who} ({_date(o.get('asserted_at', ''))}): {text}"
    if facet == "report":
        return f"reported by «{title}» ({_date(o.get('asserted_at', ''))}): {text}"
    if facet == "norm":
        v = claim.get("version") or o.get("version") or ""
        return f"«{title}»{(' v' + v) if v else ''} specifies: {text}"
    if facet == "event":
        when = claim.get("when") or _date(o.get("asserted_at", ""))
        return f"{text} ({when})" if when else text
    if facet == "procedure":
        return f"{text}" + (f" (per «{title}»)" if title else "")
    if facet == "depiction":
        return f"«{title}» depicts: {text}"
    if facet == "need":
        return f"[open] {text}"
    ind = claim.get("n_independent", n)
    if n > 1:
        src = f" ({n} documents, {ind} independent)" if ind != n else f" ({n} sources)"
    else:
        src = ""
    return f"{text}{src}"


def passage_text(store: Store, hit: Hit) -> tuple[str, str]:
    """(rendered quotation, reference line) for a chunk hit."""
    doc = store.document(hit.doc_id)
    if doc is None:
        return "", ""
    unit = doc.unit(hit.chunk_id) if hit.chunk_id else None
    if unit is None:
        return "", ""
    kind = doc.kind
    title = doc.meta.get("title") or doc.key
    version = str(doc.meta.get("version") or "")
    ref = f"{title}" + (f" v{version}" if version and kind in ("tech_doc", "structured") else "") + f" › {' > '.join(unit.path)}"
    if kind == "chat":
        when = _date(str(unit.keys.get("ts") or doc.meta.get("date") or ""))
        head = f"conversation with {unit.keys.get('user') or unit.keys.get('speaker') or 'user'}" + (f", {when}" if when else "") + f", exchange {unit.keys.get('index')}"
        body = unit.text
        return f"[{head}]\n{body}", f"chat {doc.key} #{unit.keys.get('index')}"
    body = unit.text
    if unit.role == "piece":
        if unit.unit_type == "rowgroup" and unit.keys.get("header_line"):
            body = unit.keys["header_line"] + "\n" + (unit.keys.get("sep_line") or "") + "\n" + body
        elif unit.unit_type == "block":
            body = unit.identity + "\n…\n" + body
    if kind == "code":
        return f"[{doc.key}:{unit.path[0]} @ {version}]\n{body}", f"{doc.key} › {unit.path[0]}"
    label = f"[{ref}" + (f", {_date(str(doc.meta.get('date') or ''))}" if doc.meta.get("date") else "") + "]"
    return f"{label}\n{body}", ref


def gist_text(store: Store, hit: Hit) -> tuple[str, str]:
    doc = store.document(hit.doc_id)
    if doc is None:
        return "", ""
    title = doc.meta.get("title") or doc.key
    n = sum(1 for u in doc.units if u.role == "primary")
    if doc.summary and doc.summary.get("text"):
        return f"[a recap of «{title}» (generated, {_date(doc.summary.get('ts', ''))})]\n{doc.summary['text']}", title
    heads = [u.path[-1] for u in doc.units if u.role == "primary"][:8]
    return f"[«{title}» — {n} sections: " + "; ".join(heads) + "]", title
