"""The fact protocol (ASSOCIATIVE_MEMORY.md §1.6): the shared line grammar, its parser
(ported from Ava's `chat_facts.parse_facts`, generalized with `chunk` / `version` / `rel`
markers), the facet mapping, and the protocol document written beside each document.

    [fact] (about: SUBJECT) (class: CLASS) (chunk: N) [(entities: A, B)] [(when: …)] [(version: …)] TEXT
"""

from __future__ import annotations

import re
from typing import Callable, Optional

PARSER_VERSION = "protocol-1"
PROTOCOL_SCHEMA = 1

_FACT_LINE_RE = re.compile(r"^\s*[-*•]?\s*\[fact\]\s*", re.IGNORECASE)
# A marker value may carry ONE nested parenthesis — `(rel: offered(Noam, position))`.
_MARKER_LEAD_RE = re.compile(r"^\s*\(\s*([A-Za-z_]+)\s*:\s*((?:[^()]|\([^()]*\))*)\)\s*")
_MARKER_TAIL_RE = re.compile(r"\s*\(\s*([A-Za-z_]+)\s*:\s*((?:[^()]|\([^()]*\))*)\)\s*[.;,]?\s*$")
_MARKER_FIELDS = {
    "about": "about", "subject": "about", "who": "about",
    "class": "class", "kind": "class", "type": "class",
    "chunk": "chunk", "in": "chunk", "from": "chunk",
    "entities": "entities", "with": "entities", "also": "entities",
    "when": "when", "date": "when",
    "version": "version", "rel": "rel",
}
_WS_RE = re.compile(r"\s+")
# Family end-of-turn tokens a harness may leave on the last line; never part of a fact.
_EOT_RE = re.compile(r"\s*(?:<turn\|>|<end_of_turn>|<\|im_end\|>|<eos>|<\|end\|>)\s*$")
_ENTITY_SPLIT_RE = re.compile(r"[,;]+")
_ENTITY_STRIP = " \t\"'`«»„“”‘’()[]{}<>#*•-–—.,;:!?"
_PLACEHOLDER_ABOUT = frozenset({"NAME", "SUBJECT"})
_PLACEHOLDER_ENTITIES = ("A", "B")
_INT_RE = re.compile(r"\d+")
# The statement itself wrapped as one more marker — `(stated: "…")`, `(text: …)`, `(fact: …)` —
# a shape gemma-4 produces reliably when every other field is a marker. Unwrapped rather
# than stored as marker syntax.
_WRAPPED_TEXT_RE = re.compile(r"^\s*\(\s*(?:stated|statement|text|fact|says|content)\s*:\s*(.*)\)\s*[.;,]?\s*$", re.IGNORECASE | re.DOTALL)
MAX_ENTITIES, MAX_ENTITY_CHARS, MAX_WHEN_CHARS, MAX_FACT_CHARS = 8, 60, 60, 400
MAX_FACTS_PER_DOC = 600

_CLASS_ALIASES = {
    "standing_fact": "standing", "fact": "standing", "property": "standing", "habit": "standing",
    "trait": "standing", "biography": "standing",
    "opinion": "stated", "claim": "stated", "view": "stated", "belief": "stated", "statement": "stated",
    "position": "stated", "report": "stated", "reported": "stated",
    "events": "event", "happening": "event", "action": "event",
    "fiction": "depicted", "fictional": "depicted", "in_universe": "depicted", "narrative": "depicted",
    "plot": "depicted", "depiction": "depicted",
    "norm": "spec", "normative": "spec", "rule": "spec", "requirement": "spec", "must": "spec",
    "steps": "procedure", "howto": "procedure", "how-to": "procedure", "instruction": "procedure",
    "api": "signature", "function": "signature", "method": "signature",
    "deprecation": "deprecated", "removed": "deprecated", "obsolete": "deprecated",
}
UNSPECIFIED = "unspecified"


def normalize_class(value: Optional[str], allowed: tuple[str, ...]) -> str:
    s = _WS_RE.sub(" ", str(value or "")).strip().strip(".,;:!?'\"()[]").lower()
    if s in allowed:
        return s
    a = _CLASS_ALIASES.get(s)
    if a in allowed:
        return a
    return UNSPECIFIED


def normalize_entities(raw: Optional[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for piece in _ENTITY_SPLIT_RE.split(str(raw or "")):
        name = _WS_RE.sub(" ", piece).strip().strip(_ENTITY_STRIP).strip()
        if not name or len(name) > MAX_ENTITY_CHARS:
            continue
        k = name.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(name)
        if len(out) >= MAX_ENTITIES:
            break
    return out


def _consume_markers(body: str, marks: dict) -> str:
    while True:
        m = _MARKER_LEAD_RE.match(body)
        if not m:
            break
        field = _MARKER_FIELDS.get(m.group(1).lower())
        if field is None:
            break
        marks[field] = m.group(2).strip()
        body = body[m.end():]
    while True:
        m = _MARKER_TAIL_RE.search(body)
        if not m:
            break
        field = _MARKER_FIELDS.get(m.group(1).lower())
        if field is None:
            break
        if not marks.get(field):
            marks[field] = m.group(2).strip()
        body = body[:m.start()]
    return body


def answer_after_think(raw: str) -> str:
    """The answer region of a generation: after the last reasoning-close marker, if any."""
    text = raw or ""
    for marker in ("</think>", "<channel|>", "<|end|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    return text


def parse_fact_lines(raw: str, *, allowed_classes: tuple[str, ...], truncated: bool = False,
                     subject_fn: Optional[Callable[[str], str]] = None) -> list[dict]:
    """Lenient parse of ``[fact]`` lines from the answer region (§1.6 output contract)."""
    lines = answer_after_think(str(raw or "")).splitlines()
    tagged: list[str] = [_FACT_LINE_RE.sub("", ln).strip() for ln in lines if _FACT_LINE_RE.match(ln)]
    if truncated and tagged:
        tagged.pop()
    out: list[dict] = []
    seen: set[str] = set()
    for body in tagged:
        marks: dict = {}
        body = _consume_markers(body, marks)
        about_raw = marks.get("about", "")
        entities = normalize_entities(marks.get("entities", ""))
        if about_raw.strip() in _PLACEHOLDER_ABOUT or tuple(entities) == _PLACEHOLDER_ENTITIES:
            continue
        wm = _WRAPPED_TEXT_RE.match(body)
        if wm:
            body = wm.group(1).strip().strip("\"'«»“”").strip()
        text = _EOT_RE.sub("", _WS_RE.sub(" ", body)).strip().strip("-–—•* ").strip()[:MAX_FACT_CHARS].strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        chunk_raw = marks.get("chunk", "")
        m = _INT_RE.search(chunk_raw)
        rel = None
        if marks.get("rel"):
            rm = re.match(r"\s*([A-Za-z_]+)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", marks["rel"])
            if rm:
                rel = [rm.group(1).lower(), rm.group(2).strip(), rm.group(3).strip()]
        out.append({
            "subject": (subject_fn or (lambda s: _WS_RE.sub(" ", s or "").strip()))(about_raw),
            "subject_raw": _WS_RE.sub(" ", about_raw).strip(),
            "text": text,
            "fact_class": normalize_class(marks.get("class", ""), allowed_classes),
            "entities": entities,
            "when": _WS_RE.sub(" ", marks.get("when", "")).strip()[:MAX_WHEN_CHARS],
            "version": _WS_RE.sub(" ", marks.get("version", "")).strip()[:40],
            "chunk_no": int(m.group(0)) if m else None,
            "rel": rel,
        })
        if len(out) >= MAX_FACTS_PER_DOC:
            break
    return out


def facet_of(kind_spec, fact_class: str) -> str:
    return kind_spec.facet_map.get(fact_class, "unclassified")


def new_protocol(doc_meta: dict, facts: list[dict], *, witness: str, model_id: str = "", prompt_version: str = "",
                 report: Optional[dict] = None) -> dict:
    return {
        "schema": PROTOCOL_SCHEMA, "parser_version": PARSER_VERSION, "witness": witness,
        "model_id": model_id, "prompt_version": prompt_version,
        "doc_id": doc_meta.get("doc_id"), "key": doc_meta.get("key"), "kind": doc_meta.get("kind"),
        "version": doc_meta.get("version"), "facts": facts, "report": report or {},
    }


def needs_reparse(protocol: Optional[dict], current_prompt_version: str = "") -> bool:
    if not protocol:
        return True
    if protocol.get("parser_version") != PARSER_VERSION:
        return True
    if current_prompt_version and protocol.get("witness") == "llm" and protocol.get("prompt_version") != current_prompt_version:
        return True
    return False
