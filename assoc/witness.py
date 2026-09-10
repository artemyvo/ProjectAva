"""Witnesses (ASSOCIATIVE_MEMORY.md §1.3, §1.6): the parser witness (code), the structure
witness (tables / FAQ / release notes) and the deferred LLM witness for prose kinds —
chunk-marked document, section groups, per-kind framing + shared closing, lenient parse,
chunk anchoring with repair, the grounding check, and the ingest report.

`generate_fn(system, user, *, thinking, max_new_tokens, temperature)` returns a string or a
`(text, info)` tuple where `info` may carry `truncated: bool`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

from . import kinds as kinds_mod
from .chunks import Unit
from .kinds.code import parser_witness
from .kinds.structured import structure_witness
from .lex import canon_terms, dominant_script, overlap
from .protocol import needs_reparse, new_protocol, parse_fact_lines
from .store import Document, Store

PROMPT_DIR = Path(__file__).parent / "prompts"
EXTRACT_WINDOW_CHARS = 24_000        # ≈ 6–8k tokens of material per group
CONTEXT_ONLY_CHARS = 1_500
GROUND_MIN_OVERLAP = 0.3
WITNESS_MAX_NEW_TOKENS = 4096

CLOSING = (
    "\n— end of the text —\n\n"
    "Now write the protocol: everything the text established, one `[fact]` line each, selecting "
    "nothing. Every line MUST carry `(chunk: N)` naming the chunk it came from, `(about: SUBJECT)` "
    "and `(class: CLASS)`; add `(entities: A, B)` for the other things it names, `(when: …)` on an "
    "event, `(version: …)` where the text names one. Write each fact in the language of the chunk "
    "it came from. Lines from a chunk marked `context only` must not be written. No preamble, no "
    "summary, no commentary — the list is the whole answer. The statement is plain text after the "
    "markers, never inside one:\n\n"
    "[fact] (about: Starling) (class: standing) (chunk: 2) (entities: GPU clusters) Starling builds a "
    "memory-pooling layer for GPU clusters.\n"
)


def prompt_version(spec) -> str:
    if not spec.prompt_file:
        return ""
    p = PROMPT_DIR / spec.prompt_file
    if not p.exists():
        return ""
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10]


def _framing(spec) -> str:
    if not spec.prompt_file:
        return ""
    p = PROMPT_DIR / spec.prompt_file
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ----- LLM witness --------------------------------------------------------------------------

def _primaries(doc: Document) -> list[Unit]:
    return [u for u in doc.units if u.role == "primary" and u.text.strip()]


def _groups(units: list[Unit], window: int) -> list[list[int]]:
    """Cut the primaries into groups that fit the extraction window (indices)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, u in enumerate(units):
        n = len(u.text) + 80
        if cur and size + n > window:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        groups.append(cur)
    return groups


def _marked(units: list[Unit], idxs: list[int], *, context_idx: Optional[int]) -> str:
    parts: list[str] = []
    if context_idx is not None:
        c = units[context_idx]
        parts.append(f"[chunk {context_idx + 1} | {c.identity} | context only — extract nothing from it]\n{c.text[-CONTEXT_ONLY_CHARS:]}\n")
    for i in idxs:
        u = units[i]
        parts.append(f"[chunk {i + 1} | {u.identity}]\n{u.text}\n")
    return "\n".join(parts)


def _participants_note(doc: Document) -> str:
    if doc.kind != "chat":
        return ""
    names = sorted({u.keys.get("speaker") for u in doc.units if u.keys.get("speaker")} | ({doc.meta.get("user")} if doc.meta.get("user") else set()))
    names = [n for n in names if n]
    return ("\nParticipants, with their canonical spelling: " + ", ".join(names) + " and the assistant, "
            "who appears as `Me` and whose facts take `(about: self)`. Use exactly these spellings in `(about:)`.\n") if names else ""


def _anchor_and_ground(facts: list[dict], units: list[Unit], idxs: list[int], doc_text: str, codebook=None, embedder=None) -> dict:
    """Resolve each fact's chunk (repairing by lemma overlap), then check grounding on
    lemmas AND on L2 cells together (§1.6, correction m9): a translated fact grounds through
    the cells a chunk's words quantize to, while a fact about something the chunk never
    mentions fails both."""
    counts = {"anchor_exact": 0, "anchor_inferred": 0, "anchor_moved": 0, "ungrounded": 0, "language_drifted": 0,
              "grounded_via_cells": 0}
    cand = [units[i] for i in idxs]
    cand_terms = {u.chunk_id: canon_terms(u.text) for u in cand}
    cand_cells = {u.chunk_id: (set(codebook.cells_of_text(u.text, embedder)) if codebook is not None else set()) for u in cand}
    by_no = {i + 1: units[i] for i in idxs}

    def ov_of(ft: set, fcells: set, u: Unit) -> float:
        lem = overlap(ft, cand_terms[u.chunk_id])
        cel = overlap(fcells, cand_cells[u.chunk_id]) if fcells else 0.0
        return max(lem, cel)

    for f in facts:
        ftext = f["text"] + " " + " ".join(f.get("entities") or []) + " " + (f.get("subject_raw") or "")
        ft = canon_terms(ftext)
        fcells = set(codebook.cells_of_text(ftext, embedder)) if codebook is not None else set()
        unit = by_no.get(f.get("chunk_no"))
        best_u, best_ov = None, -1.0
        for u in cand:
            ov = ov_of(ft, fcells, u)
            if ov > best_ov:
                best_u, best_ov = u, ov
        if unit is None:
            unit = best_u
            f["anchor"] = "inferred"
            counts["anchor_inferred"] += 1
        else:
            ov_named = ov_of(ft, fcells, unit)
            # Re-anchor only when the named chunk would fail grounding and another chunk grounds it.
            if ov_named < GROUND_MIN_OVERLAP and best_u is not None and best_ov >= GROUND_MIN_OVERLAP and best_u is not unit:
                unit = best_u
                f["anchor"] = "moved"
                counts["anchor_moved"] += 1
            else:
                f["anchor"] = "exact"
                counts["anchor_exact"] += 1
        if unit is None:
            f["chunk_id"], f["span"], f["grounded"] = None, None, False
            counts["ungrounded"] += 1
            continue
        f["chunk_id"] = unit.chunk_id
        f["span"] = list(unit.span)
        lem = overlap(ft, cand_terms[unit.chunk_id])
        cel = overlap(fcells, cand_cells[unit.chunk_id]) if fcells else 0.0
        ov = max(lem, cel)
        f["grounded"] = ov >= GROUND_MIN_OVERLAP
        f["ground_overlap"] = round(ov, 3)
        f["ground_via"] = "lemma" if lem >= GROUND_MIN_OVERLAP else ("cell" if cel >= GROUND_MIN_OVERLAP else "none")
        if f["ground_via"] == "cell":
            counts["grounded_via_cells"] += 1
        if not f["grounded"]:
            counts["ungrounded"] += 1
        f["language"] = dominant_script(f["text"])
        if f["language"] != dominant_script(unit.text):
            f["language_drift"] = True
            counts["language_drifted"] += 1
        f.pop("chunk_no", None)
    return counts


def llm_witness(doc: Document, spec, generate_fn: Callable, *, window: int = EXTRACT_WINDOW_CHARS,
                model_id: str = "", codebook=None, embedder=None) -> tuple[list[dict], dict]:
    units = _primaries(doc)
    framing = _framing(spec) + _participants_note(doc)
    facts: list[dict] = []
    report = {"groups": 0, "failed_groups": 0, "retried_groups": 0, "truncated_groups": 0, "facts": 0}
    counts = {"anchor_exact": 0, "anchor_inferred": 0, "anchor_moved": 0, "ungrounded": 0, "language_drifted": 0,
              "grounded_via_cells": 0}
    groups = _groups(units, window)
    prev_last: Optional[int] = None
    gi = 0
    while gi < len(groups):
        idxs = groups[gi]
        report["groups"] += 1
        material = _marked(units, idxs, context_idx=prev_last)
        gist = (doc.summary or {}).get("text") or ""
        user = (f"The document: «{doc.meta.get('title') or doc.key}»" + (f" (version {doc.meta.get('version')})" if doc.meta.get("version") else "")
                + (f" — dated {doc.meta.get('date')}" if doc.meta.get("date") else "") + ".\n"
                + (f"What it is about, in brief: {gist}\n" if gist else "") + "\n" + material + CLOSING)
        try:
            raw = generate_fn(framing, user, thinking=spec.thinking, max_new_tokens=WITNESS_MAX_NEW_TOKENS, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            raw = ("", {"error": type(e).__name__})
        info: dict = {}
        if isinstance(raw, tuple):
            raw, info = raw[0], (raw[1] or {})
        truncated = bool(info.get("truncated"))
        cut_in_think = truncated and spec.thinking and "</think>" not in (raw or "")
        if info.get("error") or cut_in_think:
            # A failed pass: retry once at the next smaller group size.
            if len(idxs) > 1 and not info.get("retried"):
                half = len(idxs) // 2
                groups[gi:gi + 1] = [idxs[:half], idxs[half:]]
                report["retried_groups"] += 1
                continue
            report["failed_groups"] += 1
            gi += 1
            continue
        if truncated:
            report["truncated_groups"] += 1
        parsed = parse_fact_lines(raw, allowed_classes=spec.classes, truncated=truncated,
                                  subject_fn=_subject_fn(spec, doc))
        c = _anchor_and_ground(parsed, units, idxs, doc.text, codebook, embedder)
        for k in counts:
            counts[k] += c[k]
        for f in parsed:
            f["version"] = f.get("version") or str(doc.meta.get("version") or "")
        facts.extend(parsed)
        prev_last = idxs[-1]
        gi += 1
    report.update(counts)
    report["facts"] = len(facts)
    report["model_id"] = model_id
    return facts, report


def _subject_fn(spec, doc: Document) -> Callable[[str], str]:
    fam = str(doc.meta.get("product") or doc.meta.get("family") or doc.meta.get("project") or doc.key)

    def person(about: str) -> str:
        s = (about or "").strip()
        low = s.lower().strip("«»\"' ")
        if low in ("self", "myself", "me", "i", "ava", "assistant", "the assistant"):
            return "person:_self"
        if not low:
            return ""
        return "person:" + low.split()[0]

    def entity(about: str) -> str:
        # Lowercased: a node id is a key, and mention nodes (graph.entity_node) are lowercased too.
        s = " ".join((about or "").split()).strip("«»\"' ").lower()
        return f"entity:{s}" if s else ""

    def ident(about: str) -> str:
        s = " ".join((about or "").split()).strip("«»\"'` ")
        return f"ident:{fam}:{s}" if s else ""

    return {"person": person, "entity": entity, "ident": ident}.get(spec.namespace, entity)


# ----- dispatch -----------------------------------------------------------------------------

def run_witnesses(store: Store, doc_id: str, *, generate_fn: Optional[Callable] = None, model_id: str = "",
                  window: int = EXTRACT_WINDOW_CHARS, codebook=None, embedder=None) -> dict:
    """Run every witness the document's kind names; write the protocol; return the report.

    Without a generate_fn the prose (LLM) witness is skipped and the document stays pending
    (its structure/parser facts are still written, so a table is a fact from the moment it lands).
    """
    doc = store.document(doc_id)
    if doc is None:
        return {"doc_id": doc_id, "error": "missing"}
    spec = kinds_mod.get(doc.kind)
    facts: list[dict] = []
    report: dict = {"doc_id": doc_id, "key": doc.key, "kind": doc.kind, "chunks": sum(1 for u in doc.units if u.role == "primary"),
                    "witnesses": [], "t0": time.time()}
    complete = True
    for w in spec.witnesses:
        if w == "parser":
            fs = parser_witness(doc.meta, doc.text, doc.units)
        elif w == "structure":
            fs = structure_witness(doc.meta, doc.text, doc.units)
        elif w == "llm":
            if generate_fn is None:
                complete = False
                continue
            fs, rep = llm_witness(doc, spec, generate_fn, window=window, model_id=model_id, codebook=codebook, embedder=embedder)
            report["llm"] = rep
        else:
            continue
        if spec.redact is not None:
            kept = []
            for f in fs:
                r = spec.redact(f)
                if r is not None:
                    kept.append(r)
            report["redacted"] = report.get("redacted", 0) + (len(fs) - len(kept))
            fs = kept
        report["witnesses"].append({"witness": w, "facts": len(fs)})
        facts.extend(fs)
    # Facts per chunk, for the report's sanity line.
    per_chunk: dict[str, int] = {}
    for f in facts:
        per_chunk[f.get("chunk_id") or "?"] = per_chunk.get(f.get("chunk_id") or "?", 0) + 1
    report["facts"] = len(facts)
    report["facts_per_chunk_max"] = max(per_chunk.values()) if per_chunk else 0
    report["chunks_with_facts"] = len([k for k in per_chunk if k != "?"])
    report["seconds"] = round(time.time() - report.pop("t0"), 2)
    protocol = new_protocol(doc.meta, facts, witness="+".join(spec.witnesses), model_id=model_id,
                            prompt_version=prompt_version(spec), report=report)
    status = "extracted" if complete else "pending"
    store.write_facts(doc_id, protocol, status=status, report=report)
    report["status"] = status
    return report


def import_protocol(store: Store, doc_id: str, facts: list[dict], *, witness: str, codebook=None, embedder=None,
                    model_id: str = "", extra_report: Optional[dict] = None) -> dict:
    """Import facts produced elsewhere (Ava's protocols) as this document's protocol: every
    fact is anchored to a chunk by overlap (lemmas + cells) and grounded the same way a
    witness's would be; the document leaves the pending set."""
    doc = store.document(doc_id)
    if doc is None:
        return {"doc_id": doc_id, "error": "missing"}
    spec = kinds_mod.get(doc.kind)
    units = _primaries(doc)
    for f in facts:
        f.setdefault("chunk_no", None)
        f["version"] = f.get("version") or str(doc.meta.get("version") or "")
    counts = _anchor_and_ground(facts, units, list(range(len(units))), doc.text, codebook, embedder) if units else {}
    if not units:
        for f in facts:
            f.update({"chunk_id": None, "span": None, "grounded": False, "anchor": "none"})
    report = {"doc_id": doc_id, "key": doc.key, "kind": doc.kind, "chunks": len(units), "facts": len(facts),
              "witnesses": [{"witness": witness, "facts": len(facts)}], **counts, **(extra_report or {})}
    protocol = new_protocol(doc.meta, facts, witness=witness, model_id=model_id, prompt_version="imported", report=report)
    store.write_facts(doc_id, protocol, status="extracted", report=report)
    return report


def pending_documents(store: Store, *, reparse: bool = True) -> list[str]:
    out: list[str] = []
    for doc_id in store.all_doc_ids():
        doc = store.document(doc_id)
        if doc is None:
            continue
        spec = kinds_mod.get(doc.kind)
        if doc.status.get("status") == "pending":
            out.append(doc_id)
        elif reparse and "llm" in spec.witnesses and needs_reparse(doc.facts, prompt_version(spec)):
            out.append(doc_id)
    return out
