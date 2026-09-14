"""Writing and reading ``data/graph/tree.json``.

Thin by design. The tree is a **derived, disposable fold** (FACTS_TREE.md §2), so this
module deliberately has no migration path, no schema upgrade, and no merge: a stale or
unreadable tree is rebuilt, never repaired. ``SCHEMA_VERSION`` exists so a *reader* can
refuse a tree it does not understand and say why, not so a writer can convert one.

The one file here that is **not** derived is ``aliases.json`` — hand-edited, and therefore
the only thing under ``data/graph/`` worth backing up. Nothing in this module writes it.

The read API is what an inference-side consumer imports; the build never imports inference.

GPU-free self-test: covered by ``python -m graph.selftest``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from training import reflections_path

SCHEMA_VERSION = 1
TREE_FILE = "tree.json"


def tree_path(graph_dir: Optional[Path] = None) -> Path:
    return Path(graph_dir or reflections_path.graph_dir()) / TREE_FILE


def aliases_path(graph_dir: Optional[Path] = None) -> Path:
    from graph.nodes import ALIASES_FILE
    return Path(graph_dir or reflections_path.graph_dir()) / ALIASES_FILE


def build_doc(tree: dict, *, built_at: str, read_stats: Optional[dict] = None) -> dict:
    """The on-disk document. ``built_at`` is passed in, never read from the clock here, so
    a build is reproducible and testable."""
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "built_at": built_at,
        "stats": dict(tree.get("stats") or {}, **{"read": read_stats or {}}),
        "nodes": tree.get("nodes") or {},
        "claims": tree.get("claims") or {},
        "occurrences": tree.get("occurrences") or [],
    }


def write_tree(doc: dict, graph_dir: Optional[Path] = None) -> Path:
    """Atomic write, via the shared helper — a half-written tree that still parses would be
    worse than none, since nothing downstream validates it."""
    path = tree_path(graph_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    reflections_path.atomic_write_text(
        path, json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
    return path


def read_tree(graph_dir: Optional[Path] = None) -> Optional[dict]:
    """The tree, or ``None`` — missing, unreadable and unrecognised all mean *rebuild*.

    A consumer must treat ``None`` as "no tree yet", never as "no facts": the sources are
    always there, and this file is the only thing that can be absent.
    """
    path = tree_path(graph_dir)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        return None
    return doc


# -- read helpers (the consumer-facing API) ---------------------------------- #

def node(doc: dict, ident: str) -> Optional[dict]:
    return (doc.get("nodes") or {}).get(ident)


def claims_for(doc: dict, ident: str, facets=None) -> list:
    """A node's claims, optionally facet-filtered, ordered most-corroborated first.

    ``facets`` is the safety control the facet level exists for: pass
    ``fold.KNOWLEDGE_FACETS`` to exclude ``position`` and ``report``, which need their
    attribution carried with them and must never be read back as things simply known.
    """
    n = node(doc, ident)
    if not n:
        return []
    all_claims = doc.get("claims") or {}
    ids: list = []
    for facet, cids in sorted((n.get("facets") or {}).items()):
        if facets is not None and facet not in facets:
            continue
        ids.extend(cids)
    out = [all_claims[c] for c in ids if c in all_claims]
    out.sort(key=lambda c: (-c.get("n_sources", 0), -c.get("n_occurrences", 0),
                            c.get("text", "")))
    return out


def find_nodes(doc: dict, query: str, limit: int = 20) -> list:
    """Substring search over a node's id, label and every surface form ever seen for it.

    Surface forms included on purpose: searching for what was *written* has to find the node
    it resolved to, or a mis-resolution is unfindable — which is the failure the inspection
    surface exists to catch.
    """
    q = str(query or "").strip().casefold()
    if not q:
        return []
    hits = []
    for ident, n in (doc.get("nodes") or {}).items():
        hay = " ".join([ident, n.get("label", "")] + list(n.get("surface_forms") or []))
        if q in hay.casefold():
            hits.append(n)
    hits.sort(key=lambda n: (-n.get("n_claims", 0), -len(n.get("mentioned_in") or []),
                             n.get("id", "")))
    return hits[:limit]


def occurrences_for(doc: dict, claim: dict) -> list:
    """The witness lines behind a claim — the level that carries provenance."""
    rows = doc.get("occurrences") or []
    return [rows[i] for i in (claim.get("occurrences") or []) if 0 <= i < len(rows)]
