"""Inferred edges (ASSOCIATIVE_MEMORY.md §4, B2) — OFF by default (`knobs["inferred_edges"]`).

When two nodes are co-active with no recorded path between them, a thinking-off pass asks
the model whether it KNOWS a connection, and writes it as an `inferred:<pred>` edge with
`source: model` into `state/inferred_edges.json` — attributed, low weight (`S_TYPE["inferred"]`),
loaded at the next rebuild only when the knob is on, and never the sole path to an aha
(`aha.find_candidates` drops a candidate whose paths on either side are inferred only).
The provenance rule `til_facts` applies to every text applies to the model too: the edge
records that a model asserted it, not that it is so.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from .relations import PREDICATES

_LINE_RE = re.compile(r"^\s*([a-z_]+)\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)\s*$", re.IGNORECASE)
SYSTEM = ("You are asked whether you KNOW a specific, factual connection between two named things. "
          "Answer with exactly one line: `predicate(A, B)` using one of the listed predicates and the names "
          "exactly as given, or `NONE` if you do not know one for certain. No explanation. A guess is worse "
          "than NONE: only what you are sure is on public record.")


def ask(generate_fn: Callable, a_label: str, b_label: str) -> Optional[str]:
    user = f"Predicates: {', '.join(PREDICATES)}\n\nA: {a_label}\nB: {b_label}\n\nAnswer:"
    try:
        raw = generate_fn(SYSTEM, user, thinking=False, max_new_tokens=48, temperature=0.0)
        if isinstance(raw, tuple):
            raw = raw[0]
    except Exception:  # noqa: BLE001
        return None
    text = str(raw or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if text.upper().startswith("NONE"):
        return None
    m = _LINE_RE.match(text.splitlines()[0] if text else "")
    if not m or m.group(1).lower() not in PREDICATES:
        return None
    return m.group(1).lower()


def has_path(table, a: str, b: str, k: int = 3) -> bool:
    frontier, seen = {a}, {a}
    for _ in range(k):
        nxt = set()
        for n in frontier:
            for o, _t, _e in table.edges(n):
                if o == b:
                    return True
                if o not in seen:
                    seen.add(o)
                    nxt.add(o)
        frontier = nxt
        if not frontier:
            break
    return False


def propose(lib, pairs: list[tuple[str, str]], generate_fn: Callable, *, now: Optional[float] = None) -> list[dict]:
    """Ask the model about each (node_a, node_b) pair that has no recorded path; record what
    it asserts. Returns the edges written this call."""
    b = lib.build
    path = lib.store.root / "state" / "inferred_edges.json"
    try:
        obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"edges": []}
    except Exception:
        obj = {"edges": []}
    known = {(e["a"], e["b"]) for e in obj["edges"]} | {(e["b"], e["a"]) for e in obj["edges"]}
    written: list[dict] = []
    for a, bn in pairs:
        if (a, bn) in known or (b.edges is not None and has_path(b.edges, a, bn)):
            continue
        pred = ask(generate_fn, a.split(":", 1)[-1], bn.split(":", 1)[-1])
        if pred:
            e = {"a": a, "b": bn, "pred": pred, "source": "model", "ts": now if now is not None else time.time()}
            obj["edges"].append(e)
            written.append(e)
        known.add((a, bn))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return written
