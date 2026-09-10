"""Typed relations (ASSOCIATIVE_MEMORY.md §2.3): a per-document, thinking-off pass that
labels each new claim of a document with at most one ``pred(subject, object)`` from a
small closed predicate vocabulary. Both arguments must name something in that claim's
own ``about`` / ``entities`` set (or, for the object, an L2 cell), else no edge and a
count. Cached per claim key + prompt version, so a fast rebuild runs it over new claims
only. The parser and structure witnesses write their own relations at ingest.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional

PREDICATES = [
    "works_at", "founded", "offered", "declined", "accepted", "interviewed_at", "wants", "asked_about",
    "looking_for", "lives_in", "moved_to", "located_in", "member_of", "acquired", "depends_on", "deprecates",
    "defaults_to", "calls", "imports", "defines", "born_in", "ceo_of", "married_to", "sibling_of", "knows",
]
NEED_PREDICATES = frozenset({"wants", "asked_about", "looking_for", "needs"})
PROMPT_FILE = Path(__file__).parent / "prompts" / "relations_prompt.txt"
_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.)-]\s*([a-z_]+)\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)\s*$", re.IGNORECASE)
_WS = re.compile(r"\s+")


def prompt_version() -> str:
    return hashlib.sha1(PROMPT_FILE.read_bytes()).hexdigest()[:10] if PROMPT_FILE.exists() else ""


def cache_key(claim: dict) -> str:
    return hashlib.sha1(f"{claim['claim_id']}\x1f{prompt_version()}".encode("utf-8")).hexdigest()[:16]


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s or "")).strip().strip("«»\"'`").lower()


def build_prompt(doc_title: str, claims: list[dict]) -> tuple[str, str]:
    system = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""
    lines = []
    for i, c in enumerate(claims, 1):
        args = [c.get("subject_raw") or c.get("subject") or ""] + list(c.get("entities") or [])
        lines.append(f"{i}. {c['text']}   [names: {', '.join(a for a in args if a)}]")
    user = (f"Document: «{doc_title}»\n\nPredicates: {', '.join(PREDICATES)}\n\nClaims:\n" + "\n".join(lines)
            + "\n\nFor each claim that states one of these relations, write one line `n: predicate(subject, object)`; "
              "both arguments must be names from that claim's own bracket. A claim that states none of them gets no line. "
              "No other text.\n\nRELATIONS:\n")
    return system, user


def parse(raw: str, claims: list[dict], *, cell_of: Optional[Callable[[str], Optional[str]]] = None) -> tuple[dict[str, list], dict]:
    """{claim_id: [pred, subj, obj]} for lines whose arguments validate; counts of the rest."""
    text = str(raw or "")
    body = text.rsplit("RELATIONS:", 1)[1] if "RELATIONS:" in text else text
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1]
    out: dict[str, list] = {}
    counts = {"lines": 0, "unknown_predicate": 0, "bad_args": 0, "out_of_range": 0, "accepted": 0}
    for line in body.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        counts["lines"] += 1
        n, pred, a, b = int(m.group(1)), m.group(2).lower(), m.group(3), m.group(4)
        if not 1 <= n <= len(claims):
            counts["out_of_range"] += 1
            continue
        if pred not in PREDICATES:
            counts["unknown_predicate"] += 1
            continue
        c = claims[n - 1]
        names = {_norm(x) for x in [c.get("subject_raw") or "", c.get("subject") or ""] + list(c.get("entities") or []) if x}
        na, nb = _norm(a), _norm(b)
        subj_ok = na in names or any(na == _norm(x.split(":", 1)[-1]) for x in [c.get("subject") or ""])
        obj_ok = nb in names
        obj_val = b.strip()
        if not obj_ok and cell_of is not None:
            cell = cell_of(nb)
            if cell:
                obj_ok, obj_val = True, cell
        if not (subj_ok and obj_ok):
            counts["bad_args"] += 1
            continue
        subj_val = c.get("subject") if (na == _norm(c.get("subject_raw") or "") or na == _norm((c.get("subject") or "").split(":", 1)[-1])) else a.strip()
        out[c["claim_id"]] = [pred, subj_val, obj_val]
        counts["accepted"] += 1
    return out, counts


class RelationCache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self._data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            self._data = {}

    def get(self, claim: dict):
        return self._data.get(cache_key(claim))

    def put(self, claim: dict, rel) -> None:
        self._data[cache_key(claim)] = rel

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


def run_relation_pass(claims_by_doc: dict[str, list[dict]], titles: dict[str, str], generate_fn: Callable, cache: RelationCache,
                      *, cell_of: Optional[Callable[[str], Optional[str]]] = None, max_claims: int = 60) -> dict:
    """Per document: one prefill, the new claims numbered; results cached per claim."""
    report = {"documents": 0, "claims": 0, "cached": 0, "accepted": 0, "bad_args": 0, "unknown_predicate": 0, "failed": 0}
    for doc_id, claims in claims_by_doc.items():
        todo = []
        for c in claims:
            if c.get("rel"):
                continue
            hit = cache.get(c)
            if hit is not None:
                report["cached"] += 1
                if hit:
                    c["rel"] = hit
                continue
            todo.append(c)
        if not todo:
            continue
        report["documents"] += 1
        for start in range(0, len(todo), max_claims):
            batch = todo[start:start + max_claims]
            report["claims"] += len(batch)
            system, user = build_prompt(titles.get(doc_id, doc_id), batch)
            try:
                raw = generate_fn(system, user, thinking=False, max_new_tokens=1024, temperature=0.0)
                if isinstance(raw, tuple):
                    raw = raw[0]
            except Exception:  # noqa: BLE001
                report["failed"] += 1
                continue
            rels, counts = parse(raw, batch, cell_of=cell_of)
            report["accepted"] += counts["accepted"]
            report["bad_args"] += counts["bad_args"]
            report["unknown_predicate"] += counts["unknown_predicate"]
            for c in batch:
                rel = rels.get(c["claim_id"])
                cache.put(c, rel or [])
                if rel:
                    c["rel"] = rel
    cache.save()
    return report
