"""Pure retrieval-policy helpers shared by the live RAG engine and self-tests.

This module deliberately has no NumPy / FAISS / sentence-transformers dependency.  The
chronology and chunk-shape rules are correctness policy, not model code, and must remain
testable on the GPU-free development/client machine.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


_SESSION_STAMP_RE = re.compile(r"^(\d{8})_(\d{6})(?:\.json)?$")


def session_started_at(value: str) -> Optional[datetime]:
    """Return the local wall-clock start encoded by a chat filename/stem."""
    name = (value or "").strip().rsplit("/", 1)[-1]
    match = _SESSION_STAMP_RE.match(name)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def recorded_at(value: str) -> Optional[datetime]:
    """Parse an ISO record timestamp into a comparable local naive datetime."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def memory_available_before(entry: dict, before_session: str) -> bool:
    """Whether a reflection-memory item existed before *before_session* began.

    ``source_session`` is provenance, not a universal clock: chats use timestamped
    filenames while external learning uses identifiers such as ``wiki:...`` and
    ``til:2026-07-12``.  Prefer the op-log record's real capture timestamp and fall
    back to a chat filename only for old records without ``available_at``.

    An unparseable external record is excluded conservatively during historical
    reflection.  Live chat passes no cutoff and can retrieve it normally.
    """
    if not before_session:
        return True
    cutoff = session_started_at(before_session)
    if cutoff is None:
        return False
    available = recorded_at(entry.get("available_at") or entry.get("ts") or "")
    if available is None:
        available = session_started_at(entry.get("source_session") or "")
    return available is not None and available < cutoff


def chunk_text(text: str, *, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split arbitrary-language text into bounded, lightly overlapping passages.

    Character bounds are intentionally conservative: unlike whitespace-token counts,
    they also constrain Russian and other scripts whose embedding token density is
    higher than English.  A nearby whitespace/newline is preferred so passages remain
    readable when injected back into the model prompt.
    """
    value = (text or "").strip()
    if not value:
        return []
    limit = max(64, int(max_chars))
    overlap = max(0, min(int(overlap_chars), limit // 3))
    if len(value) <= limit:
        return [value]

    chunks: list[str] = []
    start = 0
    n = len(value)
    while start < n:
        hard_end = min(n, start + limit)
        end = hard_end
        if hard_end < n:
            floor = start + int(limit * 0.65)
            candidates = [value.rfind("\n", floor, hard_end),
                          value.rfind(" ", floor, hard_end)]
            natural = max(candidates)
            if natural > start:
                end = natural
        piece = value[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and value[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def clipped(text: str, max_chars: int, *, marker: str = "\n...[truncated]") -> str:
    """Return a readable bounded prefix, adding *marker* only when clipped."""
    value = (text or "").strip()
    limit = max(1, int(max_chars))
    if len(value) <= limit:
        return value
    keep = max(1, limit - len(marker))
    head = value[:keep]
    cut = max(head.rfind("\n"), head.rfind(" "))
    if cut >= keep // 2:
        head = head[:cut]
    return head.rstrip() + marker


def rank_score(cosine: float, modifier: float, minimum: float) -> Optional[float]:
    """Gate on semantic relevance, then apply the decay modifier only as a ranking prior.

    The old ``cosine * modifier >= minimum`` rule silently raised the semantic gate as an
    item aged — on wander it made a 0.1-weight item impossible to retrieve at a 0.15 floor,
    and on the chat/gist/reflection channels (fixed 2026-07-23) it turned the wall-clock
    fade into a hard forget: any chat below modifier 0.5 needed a raw cosine past the 0.90
    near-duplicate ceiling, so virtually the whole corpus was unretrievable. Relevance is
    the raw cosine's job; age/decay only orders what is already relevant.
    """
    raw = float(cosine)
    if raw < float(minimum):
        return None
    return raw * max(0.0, float(modifier))
