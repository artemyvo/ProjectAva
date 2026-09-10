"""The structure witness (§1.3): a table row is a fact, its key column the subject; an FAQ
entry (a heading that is a question + its answer) is a procedure/standing fact; a
release-note item under a heading naming a version is an event dated to that release.
Runs on the Markdown units of any kind that lists ``"structure"`` among its witnesses.
"""

from __future__ import annotations

import re

from ..chunks import Unit
from .markdown import table_rows

_VERSION_RE = re.compile(r"\b(v?\d+\.\d+(?:\.\d+)?)\b")
_QUESTION_RE = re.compile(r"\?\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def structure_witness(doc_meta: dict, text: str, units: list[Unit]) -> list[dict]:
    facts: list[dict] = []
    version = str(doc_meta.get("version") or "")
    fam = str(doc_meta.get("product") or doc_meta.get("family") or doc_meta.get("key") or "")
    seen_tables: set[tuple] = set()
    for u in units:
        if u.unit_type == "rowgroup":
            header = u.keys.get("header") or []
            if not header:
                continue
            key_col = header[0]
            for ln in u.text.splitlines():
                cells = _cells(ln)
                if not cells or not cells[0]:
                    continue
                subject_val = cells[0]
                body = "; ".join(f"{h}: {c}" for h, c in zip(header[1:], cells[1:]) if c)
                facts.append({
                    "subject": f"ident:{fam}:{subject_val}", "subject_raw": subject_val,
                    "text": f"{key_col} {subject_val}: {body}" if body else f"{key_col} {subject_val}",
                    "fact_class": "standing", "entities": [c for c in cells[1:] if c and len(c) < 40][:8], "when": "",
                    "chunk_id": u.chunk_id, "span": list(u.span), "anchor": "exact", "grounded": True,
                    "language": "lat", "version": version, "table": u.path[-2] if len(u.path) >= 2 else "",
                })
            seen_tables.add((u.parent_id, u.keys.get("table_index")))
        elif u.role == "primary" and u.unit_type == "section":
            heading = str(u.keys.get("heading") or "")
            # FAQ entry: a question heading + its answer.
            if _QUESTION_RE.search(heading):
                body_lines = [ln for ln in u.text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
                answer = " ".join(body_lines)[:400]
                cls = "procedure" if any(_LIST_ITEM_RE.match(ln) for ln in body_lines) else "standing"
                facts.append({
                    "subject": f"ident:{fam}:{heading}", "subject_raw": heading,
                    "text": f"Q: {heading} A: {answer}", "fact_class": cls, "entities": [], "when": "",
                    "chunk_id": u.chunk_id, "span": list(u.span), "anchor": "exact", "grounded": True,
                    "language": "lat", "version": version, "faq": True,
                })
                continue
            # Release notes: a heading naming a version + list items -> events.
            vm = _VERSION_RE.search(heading) or _VERSION_RE.search(" ".join(u.path))
            if vm and ("release" in " ".join(u.path).lower() or "changelog" in " ".join(u.path).lower() or "what's new" in " ".join(u.path).lower()):
                rel = vm.group(1)
                for ln in u.text.splitlines():
                    if ln.lstrip().startswith("#"):
                        continue
                    m = _LIST_ITEM_RE.match(ln)
                    if not m:
                        continue
                    item = m.group(1).strip()
                    facts.append({
                        "subject": f"ident:{fam}:{rel}", "subject_raw": rel,
                        "text": f"In {rel}: {item}", "fact_class": "event", "entities": [], "when": rel,
                        "chunk_id": u.chunk_id, "span": list(u.span), "anchor": "exact", "grounded": True,
                        "language": "lat", "version": version, "release": rel,
                    })
    return facts
