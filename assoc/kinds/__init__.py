"""The kind registry (ASSOCIATIVE_MEMORY.md §1.3).

A kind is a plugin: ``{splitter, witnesses, namespace, classes, clock, redact}``. The
library core enumerates none; it ships several. ``split(text, meta)`` returns the rendered
document text (what spans index into) and the list of units. ``witnesses`` names which
passes produce the protocol: ``"structure"`` (tables / lists, no model), ``"parser"``
(tree-sitter, no model), ``"llm"`` (the deferred prose witness, §1.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..chunks import Unit


@dataclass(frozen=True)
class KindSpec:
    name: str
    split: Callable[[str, dict], tuple[str, list[Unit]]]
    witnesses: tuple[str, ...]
    classes: tuple[str, ...]
    facet_map: dict                   # class -> facet
    namespace: str                    # person | entity | ident
    clock: str                        # exchange | article_date | fetch_date | doc_version | commit | page_date
    prompt_file: Optional[str] = None # llm witness framing (assoc/prompts/<file>)
    thinking: bool = False            # llm witness thinking on/off
    split_oversize: bool = True       # chat exchanges are never split
    redact: Optional[Callable[[dict], Optional[dict]]] = None   # fact -> fact | None (drop)


_REGISTRY: dict[str, KindSpec] = {}
_LOADED = False


def register(spec: KindSpec) -> KindSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> KindSpec:
    if name not in _REGISTRY:
        _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown kind: {name!r} (known: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def names() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def _load_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import prose, chat, code, structured  # noqa: F401  (each registers on import)


# Facets (FACTS_TREE.md §6 + the library's additions).
KNOWLEDGE_FACETS = frozenset({"property", "event", "procedure"})
FACET_PROPERTY, FACET_POSITION, FACET_REPORT, FACET_EVENT = "property", "position", "report", "event"
FACET_DEPICTION, FACET_NORM, FACET_PROCEDURE, FACET_NEED, FACET_UNCLASSIFIED = "depiction", "norm", "procedure", "need", "unclassified"
