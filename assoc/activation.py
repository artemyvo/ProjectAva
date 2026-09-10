"""L4 — the activation state (ASSOCIATIVE_MEMORY.md §2.4, §2.6): sqlite, keyed by
``(scope, node)``, a global layer plus context layers; the power-law base level with the
finite cold baseline ``B₀``; the standing-need floor that itself fades; the clock injected.

Time is in HOURS. ``B_i = max(B₀, ln Σ_k (t_now − t_k)^−d)``. A node with no access is
``B₀`` — the base level of one access 30 days old — never minus infinity. Each node keeps
its last RECENT_KEEP timestamps exactly and the older tail as a count + first access,
folded with Petrov's approximation, so a node touched ten thousand times costs the same
as one touched ten.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

D_DECAY = 0.5
COLD_DAYS = 30.0
NEED_FLOOR_HOURS = 24.0          # a need is never colder than one access 24 h old …
NEED_FLOOR_HALF_LIFE_DAYS = 30.0 # … halving (in odds) every 30 days from the need's own date
BULK_OFFSET_DAYS = 30.0
GLOBAL_DISCOUNT = 0.5            # the global layer's odds count this much in a context pull
RECENT_KEEP = 20
MIN_AGE_H = 1.0 / 60.0
GLOBAL = "__global__"
PA_C, PA_A, PA_D_MAX = 0.217, 0.177, 2.0   # Pavlik & Anderson 2005: d_k = c·e^{m_{k−1}} + a — the spacing
                                           # effect; d_k capped: our hour-scale odds run far above ACT-R's


def base_level_from(ages_h: Iterable[float], d: float = D_DECAY) -> float:
    s = sum(max(a, MIN_AGE_H) ** (-d) for a in ages_h)
    return math.log(s) if s > 0 else float("-inf")


def cold_baseline(d: float = D_DECAY, cold_days: float = COLD_DAYS) -> float:
    return math.log((cold_days * 24.0) ** (-d))


def odds_from_accesses(now_h: float, recent: list, count_old: int, first_h: Optional[float], d: float = D_DECAY) -> float:
    """Σ_k (t_now − t_k)^−d_k, with the older tail approximated (Petrov 2006): the tail's
    accesses are spread uniformly between the first access and the oldest kept one. An
    access is a timestamp or a [timestamp, d_k] pair (activation-dependent decay)."""
    s = 0.0
    for r in recent:
        t, dk = (r[0], r[1]) if isinstance(r, (list, tuple)) else (r, d)
        s += max(now_h - t, MIN_AGE_H) ** (-dk)
    recent = [(r[0] if isinstance(r, (list, tuple)) else r) for r in recent]
    if count_old > 0 and first_h is not None:
        oldest_kept = min(recent) if recent else now_h
        t1, t2 = max(now_h - first_h, MIN_AGE_H), max(now_h - oldest_kept, MIN_AGE_H)
        if t1 > t2 and d != 1.0:
            s += count_old * (t1 ** (1 - d) - t2 ** (1 - d)) / ((1 - d) * (t1 - t2))
        else:
            s += count_old * t1 ** (-d)
    return s


def scope_key(scope: Optional[dict]) -> str:
    """The context layer's key: the scope's identity facets, sorted; '' for none."""
    if not scope:
        return ""
    keep = {k: v for k, v in scope.items() if k in ("tenant", "conversation", "branch", "user") and v is not None}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False) if keep else ""


class Activation:
    def __init__(self, path: Path, *, d: float = D_DECAY, cold_days: float = COLD_DAYS, global_discount: float = GLOBAL_DISCOUNT,
                 variable_d: bool = False):
        """*variable_d*: activation-dependent decay (§8) — an access made while the node is
        already warm decays faster (d_k = c·e^{m} + a), which is what produces the spacing
        effect the fixed-d sum does not (§2.6, correction m2). Off by default."""
        self.variable_d = variable_d
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = d
        self.b0 = cold_baseline(d, cold_days)
        self.global_discount = global_discount
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS access (
            scope TEXT NOT NULL, node TEXT NOT NULL, recent TEXT NOT NULL, count_old INTEGER NOT NULL,
            first_h REAL, last_h REAL, PRIMARY KEY (scope, node))""")
        self._db.commit()

    # ----- write ------------------------------------------------------------------------
    def touch(self, nodes: Iterable[str], *, now_h: float, scope: Optional[dict] = None, at_h: Optional[float] = None) -> None:
        """Record one access per node at *at_h* (default now) in the context layer (if the
        scope names one) AND the global layer."""
        t = now_h if at_h is None else at_h
        layers = [GLOBAL]
        sk = scope_key(scope)
        if sk:
            layers.append(sk)
        with self._lock:
            cur = self._db.cursor()
            for node in nodes:
                for layer in layers:
                    row = cur.execute("SELECT recent, count_old, first_h, last_h FROM access WHERE scope=? AND node=?", (layer, node)).fetchone()
                    if row is None:
                        entry = [t, min(PA_C * math.exp(self.b0) + PA_A, PA_D_MAX)] if self.variable_d else t
                        cur.execute("INSERT INTO access VALUES (?,?,?,?,?,?)", (layer, node, json.dumps([entry]), 0, t, t))
                        continue
                    recent = json.loads(row[0])
                    if self.variable_d:
                        # d_k from the activation the node had at the moment of this access.
                        m = math.log(max(odds_from_accesses(t, recent, row[1], row[2], self.d), math.exp(self.b0)))
                        recent.append([t, min(PA_C * math.exp(m) + PA_A, PA_D_MAX)])
                    else:
                        recent.append(t)
                    recent.sort(key=lambda r: r[0] if isinstance(r, list) else r)
                    count_old = row[1]
                    if len(recent) > RECENT_KEEP:
                        count_old += len(recent) - RECENT_KEEP
                        recent = recent[-RECENT_KEEP:]
                    oldest = min((r[0] if isinstance(r, list) else r) for r in recent)
                    cur.execute("UPDATE access SET recent=?, count_old=?, first_h=?, last_h=? WHERE scope=? AND node=?",
                                (json.dumps(recent), count_old, min(row[2] if row[2] is not None else t, t, oldest), max(row[3] or t, t), layer, node))
            self._db.commit()

    # ----- read -------------------------------------------------------------------------
    def odds(self, node: str, *, now_h: float, scope: Optional[dict] = None) -> float:
        """e^{B_i}: the context layer's odds plus the discounted global layer's, floored at e^{B₀}."""
        sk = scope_key(scope)
        total = 0.0
        with self._lock:
            cur = self._db.cursor()
            for layer, w in ((sk, 1.0), (GLOBAL, self.global_discount if sk else 1.0)):
                if not layer:
                    continue
                row = cur.execute("SELECT recent, count_old, first_h FROM access WHERE scope=? AND node=?", (layer, node)).fetchone()
                if row is None:
                    continue
                total += w * odds_from_accesses(now_h, json.loads(row[0]), row[1], row[2], self.d)
        return max(total, math.exp(self.b0))

    def odds_many(self, nodes: Iterable[str], *, now_h: float, scope: Optional[dict] = None) -> dict[str, float]:
        return {n: self.odds(n, now_h=now_h, scope=scope) for n in nodes}

    def base_level(self, node: str, *, now_h: float, scope: Optional[dict] = None) -> float:
        return math.log(self.odds(node, now_h=now_h, scope=scope))

    def accesses(self, node: str, *, scope: Optional[dict] = None) -> int:
        sk = scope_key(scope) or GLOBAL
        with self._lock:
            row = self._db.execute("SELECT recent, count_old FROM access WHERE scope=? AND node=?", (sk, node)).fetchone()
        return (len(json.loads(row[0])) + row[1]) if row else 0

    def has_any(self, node: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM access WHERE node=? LIMIT 1", (node,)).fetchone() is not None

    def close(self) -> None:
        with self._lock:
            self._db.close()


def need_floor_odds(now_h: float, need_asserted_h: float, *, d: float = D_DECAY) -> float:
    """A standing need's floor in odds: one access NEED_FLOOR_HOURS old, halved every
    NEED_FLOOR_HALF_LIFE_DAYS from the need's own date (§2.6). Never below cold."""
    base = NEED_FLOOR_HOURS ** (-d)
    age_days = max(now_h - need_asserted_h, 0.0) / 24.0
    return base * 0.5 ** (age_days / NEED_FLOOR_HALF_LIFE_DAYS)


def hours(ts: Optional[str], *, fallback_h: float) -> float:
    """ISO date/time string → hours since the epoch; the fallback for an unparseable one."""
    if not ts:
        return fallback_h
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(str(ts)[:19], fmt)) / 3600.0
        except ValueError:
            continue
    return fallback_h


def now_hours() -> float:
    return time.time() / 3600.0
