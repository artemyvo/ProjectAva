"""Spreading activation (ASSOCIATIVE_MEMORY.md §2.7): a beam expansion over the edge
table — seeds normalized to Σ W = 1, per-hop attenuation γ, the soft per-type fan penalty
φ(f) = f^−α, the activation floor ε, the beam B, the four path rules — and the one-scale
activation ``A_i = ln(e^{B_i} + κ r_i) + λ ln(1 + auth_i)``.

Every arrival is recorded; the top three paths per reached node survive for `explain`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from .graph import S_TYPE, EdgeTable

ALPHA, GAMMA, KAPPA, LAMBDA = 0.5, 0.6, 1.0, 0.3
EPS, BEAM = 1e-4, 200
TAU = -2.0                     # "at least as available as a node touched once ~55 h ago"
HOPS = {"chat": 2, "wander": 3, "need": 3, "stimulus": 2}
NOT_EXPANDED = frozenset({"in_doc", "same_cell"})       # path rule 3
SELF_NODE = "person:_self"


@dataclass
class Reached:
    r: dict[str, float] = field(default_factory=dict)              # node -> fraction of the cue's activation
    paths: dict[str, list[dict]] = field(default_factory=dict)     # node -> top paths [{a, edges:[(from,type,to,strength)]}]
    hops: dict[str, int] = field(default_factory=dict)


def spread(table: EdgeTable, seeds: dict[str, float], *, k: int = 2, alpha: float = ALPHA, gamma: float = GAMMA,
           eps: float = EPS, beam: int = BEAM, s_type: Optional[dict] = None) -> Reached:
    s_type = s_type or S_TYPE
    total = sum(v for v in seeds.values() if v > 0) or 1.0
    frontier: dict[str, float] = {n: v / total for n, v in seeds.items() if v > 0}
    out = Reached()
    # path bookkeeping: for each node in the frontier, the best path that brought it there
    best_path: dict[str, list] = {n: [] for n in frontier}
    for n, a in frontier.items():
        out.r[n] = a
        out.paths[n] = [{"a": a, "edges": []}]
        out.hops[n] = 0
    for h in range(1, k + 1):
        nxt: dict[str, float] = defaultdict(float)
        arrivals: dict[str, list[dict]] = defaultdict(list)
        for j, a_j in frontier.items():
            on_path = {j} | {e[0] for e in best_path.get(j, [])}
            for i, etype, _ev in table.edges(j):
                base = etype.split(":", 1)[0]
                if base in NOT_EXPANDED:
                    continue
                if j == SELF_NODE and base in ("mentions", "about"):        # path rule 1
                    continue
                if i in on_path:                                            # path rule 4
                    continue
                s = s_type.get(base, s_type.get(etype, 0.5)) * table.w(j, i, etype)
                mult = min(s * gamma * (table.fan_of(j, etype) ** (-alpha)), 0.999)   # never amplifies (m1)
                a = a_j * mult
                if a < eps:
                    continue
                nxt[i] += a
                arrivals[i].append({"a": a, "edges": best_path.get(j, []) + [(j, etype, i, round(mult, 4))]})
        if not nxt:
            break
        top = sorted(nxt.items(), key=lambda kv: -kv[1])[:beam]
        frontier = dict(top)
        for i, a in top:
            out.r[i] = out.r.get(i, 0.0) + a
            out.hops.setdefault(i, h)
            ps = sorted(arrivals[i], key=lambda p: -p["a"])[:3]
            out.paths.setdefault(i, [])
            out.paths[i] = sorted(out.paths[i] + ps, key=lambda p: -p["a"])[:3]
            best_path[i] = ps[0]["edges"] if ps else []
    return out


def activation(r: float, odds: float, auth: float = 0.0, *, kappa: float = KAPPA, lam: float = LAMBDA) -> float:
    """A_i on one scale: ln(e^{B_i} + κ r_i) + λ ln(1 + auth_i)."""
    return math.log(odds + kappa * max(r, 0.0)) + lam * math.log(1.0 + max(auth, 0.0))


def render_path(path: dict) -> str:
    if not path["edges"]:
        return "(seed)"
    parts = []
    for frm, et, to, s in path["edges"]:
        parts.append(f"{frm} —{et} ({s})→ {to}")
    return " ; ".join(parts)
