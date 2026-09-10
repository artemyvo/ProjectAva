"""Chunk model (ASSOCIATIVE_MEMORY.md §1.4–§1.5).

A *unit* is what a kind's splitter cuts a document into. Three roles:

- ``primary`` — the kind's injectable unit (a section, an exchange, a symbol, a row group);
- ``piece``   — a primary's next structural level (a paragraph, a code block, a row
  sub-group), used when the primary does not fit the box's per-chunk ceiling; each piece
  carries an ``identity`` prefix (heading path / header row / signature) repeated at render;
- ``ancestor`` — a heading with no content of its own; navigation only.

Chunk identity is structural: ``hash(doc_key, structural path, normalized text)`` — never
the span — so a chunk keeps its id across everything that does not change *it*.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

_WS = re.compile(r"\s+")


def norm_text(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def chunk_id_for(doc_key: str, path: list[str], text: str) -> str:
    h = hashlib.sha1()
    h.update(doc_key.encode("utf-8"))
    h.update(b"\x1f")
    h.update("/".join(path).encode("utf-8"))
    h.update(b"\x1f")
    h.update(norm_text(text).encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class Unit:
    chunk_id: str
    role: str                      # primary | piece | ancestor
    path: list[str]                # structural path, e.g. ["Install", "Linux", "systemd"]
    span: tuple[int, int]          # char span in the rendered document text
    text: str
    parent_id: Optional[str] = None
    unit_type: str = "section"     # section | paragraph | code | table | rowgroup | list | exchange | symbol | block | faq | note
    keys: dict = field(default_factory=dict)   # kind-specific: speaker, ts, symbol, lang, header, identity
    tokens_est: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = list(self.span)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Unit":
        d = dict(d)
        d["span"] = tuple(d["span"])
        return Unit(**d)

    @property
    def identity(self) -> str:
        return str(self.keys.get("identity") or " > ".join(self.path))


def make_unit(doc_key: str, *, role: str, path: list[str], span: tuple[int, int], text: str,
              unit_type: str, parent: Optional[Unit] = None, keys: Optional[dict] = None) -> Unit:
    return Unit(
        chunk_id=chunk_id_for(doc_key, path, text),
        role=role, path=list(path), span=span, text=text,
        parent_id=parent.chunk_id if parent else None,
        unit_type=unit_type, keys=dict(keys or {}),
    )


def find_unit_for_span(units: list[Unit], start: int, end: int, *, roles=("primary",)) -> Optional[Unit]:
    """The unit of an allowed role whose span best covers [start, end) (largest overlap)."""
    best, best_ov = None, 0
    for u in units:
        if u.role not in roles:
            continue
        ov = min(end, u.span[1]) - max(start, u.span[0])
        if ov > best_ov:
            best, best_ov = u, ov
    return best
