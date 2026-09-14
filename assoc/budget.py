"""Token estimates, fit stamps and budget spending (ASSOCIATIVE_MEMORY.md §1.5).

``tokens_est`` is per script, fitted at rebuild through an injected tokenizer when one is
supplied (200-chunk sample); defaults 4.0 chars/token for Latin, 2.2 for Cyrillic. An
oversize primary is represented by its pieces; a chat exchange is never split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .chunks import Unit
from .lex import script_mix

DEFAULT_CHARS_PER_TOKEN = {"lat": 4.0, "cyr": 2.2, "other": 2.5}


@dataclass
class Budget:
    total: int = 8000                       # tokens per inject
    chunk_ceiling_frac: float = 0.5         # one chunk may take at most this share of the block
    shares: dict = field(default_factory=lambda: {"claim": 0.3, "chunk": 0.55, "gist": 0.15})
    chars_per_token: dict = field(default_factory=lambda: dict(DEFAULT_CHARS_PER_TOKEN))
    tokenizer: Optional[Callable[[str], int]] = None   # text -> token count

    @property
    def chunk_ceiling(self) -> int:
        return int(self.total * self.chunk_ceiling_frac)

    def share(self, grain: str) -> int:
        return int(self.total * self.shares.get(grain, 0.0))

    def tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return int(self.tokenizer(text))
            except Exception:
                pass
        return tokens_est(text, self.chars_per_token)


def tokens_est(text: str, cpt: Optional[dict] = None) -> int:
    cpt = cpt or DEFAULT_CHARS_PER_TOKEN
    mix = script_mix(text)
    letters = sum(mix.values()) or 1
    # Non-letter characters (punctuation, digits, whitespace) are charged at the Latin rate.
    non_letters = max(len(text) - letters, 0)
    est = non_letters / cpt["lat"]
    for sc, n in mix.items():
        est += n / cpt.get(sc, cpt["lat"])
    return max(1, int(est + 0.5))


def fit_chars_per_token(tokenizer: Callable[[str], int], samples: list[str]) -> dict:
    """Fit chars-per-token for Latin and Cyrillic over *samples* (§1.5, correction m10)."""
    tot = {"lat": [0, 0], "cyr": [0, 0]}
    for s in samples[:200]:
        mix = script_mix(s)
        sc = "cyr" if mix["cyr"] > mix["lat"] else "lat"
        try:
            n = int(tokenizer(s))
        except Exception:
            continue
        if n <= 0:
            continue
        tot[sc][0] += len(s)
        tot[sc][1] += n
    out = dict(DEFAULT_CHARS_PER_TOKEN)
    for sc, (chars, toks) in tot.items():
        if toks >= 50:
            out[sc] = round(chars / toks, 2)
    return out


def stamp_fits(units: list[Unit], budget: Budget, *, split_oversize: bool = True) -> dict:
    """Decide which units are injectable at the chunk grain.

    Returns ``{chunk_id: {"fits": bool, "injectable": bool, "tokens": int}}`` plus a report.
    A primary that fits is injectable and its pieces are not; an oversize primary is not,
    and its pieces are (those that fit). For a kind that never splits, an oversize primary
    is simply not injectable.
    """
    out: dict = {}
    by_parent: dict[str, list[Unit]] = {}
    for u in units:
        u.tokens_est = budget.tokens(u.text)
        if u.role == "piece" and u.parent_id:
            by_parent.setdefault(u.parent_id, []).append(u)
    ceiling = budget.chunk_ceiling
    report = {"primaries": 0, "oversize": 0, "split": 0, "still_oversize": 0, "injectable": 0}
    for u in units:
        if u.role == "ancestor":
            out[u.chunk_id] = {"fits": True, "injectable": False, "tokens": 0}
            continue
        if u.role != "primary":
            continue
        report["primaries"] += 1
        fits = u.tokens_est <= ceiling
        if fits:
            out[u.chunk_id] = {"fits": True, "injectable": True, "tokens": u.tokens_est}
            report["injectable"] += 1
            for p in by_parent.get(u.chunk_id, []):
                out[p.chunk_id] = {"fits": p.tokens_est <= ceiling, "injectable": False, "tokens": p.tokens_est}
            continue
        report["oversize"] += 1
        out[u.chunk_id] = {"fits": False, "injectable": False, "tokens": u.tokens_est}
        pieces = by_parent.get(u.chunk_id, [])
        if not split_oversize or not pieces:
            report["still_oversize"] += 1
            continue
        report["split"] += 1
        for p in pieces:
            pf = p.tokens_est <= ceiling
            out[p.chunk_id] = {"fits": pf, "injectable": pf, "tokens": p.tokens_est}
            if pf:
                report["injectable"] += 1
            else:
                report["still_oversize"] += 1
    out["__report__"] = report
    return out
