"""Server-owned consolidation chunking.

Ports the character-budget algorithm from the client (sleep_widget.py) and
optionally uses the server tokenizer for precise token counting instead of a
character estimate.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

_CHARS_PER_TOKEN: float = 2.5    # conservative fallback without a tokenizer
_CHUNK_BUDGET_FRAC: float = 0.45  # fraction of context reserved for session content
_CHUNK_OVERLAP: int = 1           # exchanges repeated at chunk boundaries for coherence

# Reframing note for a session Ava started herself (initiated_by == "ava"). The
# default reflection framing assumes "I am Ava, learning about the person I spoke
# with"; a reverse-role outreach chat inverts that — the other person is often
# drawing Ava out about her own nature, so their turns describe *her* state and
# hers voice it. Without this note the pass misattributes Ava's own interior arc
# ("relief", "curiosity", "my identity as a role") to "the user". Injected into the
# rendered content (not the global prompt) so ordinary sessions are unchanged.
_AVA_INITIATED_NOTE = (
    "NOTE — this is a conversation YOU started, on your own initiative, often to "
    "think through something about yourself. It may read like an ordinary chat, but "
    "the roles are inverted: the feelings, states, and self-descriptions surfaced "
    "here are frequently YOUR OWN, reflected back to you by the other person, not "
    "facts about them. Attribute with care — do not record your own interior state "
    "as though it were theirs."
)

# Synthetic stage-direction speakers — see reflection_source._NARRATOR_SPEAKERS.
# "(initiative)" is the reversed-session opener impulse Ava writes to herself;
# "(setting)" is an encounter framing block. Rendered as "(initiative): ..." in a
# user slot, consolidation reads Ava's own opener as an interlocutor's turn and
# attributes her own state to "the user". Render as a marked stage direction instead.
_NARRATOR_SPEAKERS = frozenset({"(initiative)", "(setting)"})
_STAGE_DIRECTION_TAG = "(stage direction — you, not another person)"

# Closing line for the SUMMARY pass, placed after the transcript. The sibling of
# `reflection_runner._USER_NOTES_CLOSING` / `_RECOLLECTION_CLOSING` / `_SELF_NOTES_CLOSING`,
# and here for the identical reason: the task is stated in the system message, a whole
# conversation sits between it and the first generated token, and the nearest thing to
# continue is the reply that ended the chat. Those passes get their closing from
# `reflection_source.build_session_reading_content(closing=…)`, which exists so "the final
# thing in the prompt is the task rather than something to continue"; the summary pass is
# the one reading pass that never got one, because it rides this chunk builder instead —
# and was observed answering the last message rather than recapping the conversation.
#
# It names the transcript a record explicitly. The other closings do not need to: their
# task (judge / read the person / say what you make of it) is not something a transcript
# can be mistaken for. "Write a recap of this conversation" and "reply to this
# conversation" are close enough in shape that the framing has to be said out loud.
SUMMARY_CLOSING = (
    "— end of the conversation —\n\n"
    "The transcript above is a record, not a turn addressed to you. No one is waiting on "
    "a reply, and its last line is simply where the conversation stopped. Now write the "
    "summary of it: continuous prose addressed to your future self, recounting what "
    "happened and what was said."
)


def append_closing(content: str, closing: str) -> str:
    """Put *closing* last in a rendered content block. One definition of the join.

    Both callers go through this: the chunk builder (which can bake the closing into the
    chunks it fits) and the production summary pass (which cannot, because its chunks are
    shared with consolidation — see `reflection_runner`).
    """
    closing = (closing or "").strip()
    if not closing:
        return content
    return f"{content.rstrip()}\n\n{closing}\n"


# ── exchange + chunk formatting ──────────────────────────────────────────────

def format_exchange_block(exchange: dict, session_user: str) -> str:
    """Render one exchange for consolidation: speaker-named user turn + reply.

    CoT is omitted — consolidation re-derives its judgement fresh and the old
    CoT inflates input without improving the reflection.
    """
    user_msg = (exchange.get("user_prompt") or "").strip()
    response = (exchange.get("assistant_response") or "").strip()
    speaker = (exchange.get("speaker") or "").strip()
    lines: list[str] = []
    if user_msg:
        if speaker in _NARRATOR_SPEAKERS:
            lines.append(f"{_STAGE_DIRECTION_TAG} {user_msg}".strip())
        else:
            lines.append(f"{speaker or session_user or 'User'}: {user_msg}")
    if response:
        lines.append(f"Me: {response}")
    return "\n".join(lines)


def format_chunk_content(
    session: dict,
    indices: list[int],
    session_idx: int,
    session_total: int,
    part: int,
    parts: int,
    *,
    items: Optional[list[dict]] = None,
    closing: str = "",
) -> str:
    """Render a contiguous slice of a session's exchanges for one consolidation pass.

    *closing* is appended after the transcript, so the last thing in the prompt is the
    task rather than a line of dialogue — see :data:`SUMMARY_CLOSING`. Empty by default:
    consolidation's own contract lives in its system prompt and its output shape
    (``## WEIGHTS`` / ``## RAG``) is not one a transcript can be mistaken for.
    """
    timestamp = session.get("timestamp", f"session {session_idx}")
    system_prompt = (session.get("system_prompt") or "").strip()
    session_user = (session.get("user") or "").strip()
    ava_initiated = (session.get("initiated_by") or "").strip() == "ava"
    exchanges = session.get("exchanges", [])

    lines: list[str] = [f"Session {session_idx} of {session_total}  |  {timestamp}", ""]

    if ava_initiated:
        lines += [_AVA_INITIATED_NOTE, ""]

    if parts > 1:
        lines += [
            f"This is part {part} of {parts} of a single longer session — too "
            "large to read at once. Reflect on what THIS part reveals; the other "
            "parts are shown in separate passes. Keep only what this slice "
            "supports, and do not infer how the session as a whole ended.",
            "",
        ]

    if system_prompt:
        lines += [
            "System prompt in effect during this session:",
            f'"{system_prompt}"',
            "",
        ]

    lines += [
        "Past exchange — the person speaking is named at the start of each line:",
        "",
    ]

    if items is not None:
        for item in items:
            block = item.get("text") or ""
            if not block.strip():
                continue
            fragment_count = int(item.get("fragment_count") or 1)
            if fragment_count > 1:
                lines.append(
                    f"Exchange {int(item.get('exchange_index', 0)) + 1}, fragment "
                    f"{int(item.get('fragment_index') or 1)} of {fragment_count}:"
                )
            lines.append(block)
            lines.append("")
    else:
        for i in indices:
            if 0 <= i < len(exchanges):
                block = format_exchange_block(exchanges[i], session_user)
                if block:
                    lines.append(block)
                    lines.append("")

    return append_closing("\n".join(lines), closing)


# ── token counting ───────────────────────────────────────────────────────────

def _count_tokens(text: str, tokenizer) -> int:
    """Token count for *text* — precise with tokenizer, char estimate otherwise."""
    if tokenizer is not None:
        try:
            inner = getattr(tokenizer, "tokenizer", tokenizer)
            return len(inner.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return int(len(text) / _CHARS_PER_TOKEN)


# ── chunking ─────────────────────────────────────────────────────────────────


class ConsolidationChunkingError(RuntimeError):
    """Even the fixed consolidation framing cannot fit the supplied input budget."""


def _with_open_questions(content: str, open_questions_block: str) -> str:
    return (open_questions_block.rstrip() + "\n" + content
            if open_questions_block else content)


def _prefer_semantic_boundary(text: str, limit: int) -> int:
    """Choose a paragraph/sentence boundary near *limit*, preserving exact text."""
    floor = max(1, limit // 2)
    candidates = [text.rfind("\n\n", floor, limit + 1),
                  text.rfind("\n", floor, limit + 1)]
    for match in re.finditer(r"(?<=[.!?])\s+", text[floor:limit + 1]):
        candidates.append(floor + match.end())
    best = max(candidates, default=-1)
    return best if best > 0 else limit


def _fragment_item(item: dict, fits_items: Callable[[list[dict]], bool]) -> list[dict]:
    """Split one oversized exchange into exact, ordered text fragments that fit."""
    remaining = item["text"]
    pieces: list[str] = []
    while remaining:
        lo, hi, best = 1, len(remaining), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = dict(item, text=remaining[:mid], fragment_index=999,
                         fragment_count=999)
            if fits_items([probe]):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best <= 0:
            raise ConsolidationChunkingError(
                "Consolidation fixed prompt leaves no room for transcript text."
            )
        cut = _prefer_semantic_boundary(remaining, best)
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]

    count = len(pieces)
    fragments = [
        dict(item, text=text, fragment_index=i + 1, fragment_count=count)
        for i, text in enumerate(pieces)
    ]
    if not all(fits_items([fragment]) for fragment in fragments):
        raise ConsolidationChunkingError(
            "Exchange fragmentation could not satisfy the final prompt budget."
        )
    return fragments

def build_consolidation_chunks(
    session: dict,
    context_length: int,
    open_questions_block: str = "",
    tokenizer=None,
    *,
    session_idx: int = 1,
    session_total: int = 1,
    fits: Optional[Callable[[str], bool]] = None,
    closing: str = "",
) -> list[dict]:
    """Split a session's exchanges into context-sized chunks for consolidation.

    Returns chunk dicts carrying ``indices``, exact rendered ``content``, optional
    fragment metadata, and ``part``/``parts``. When *fits* is supplied it receives
    the complete content block (open questions + session framing + transcript) and
    decides against the actual model-facing prompt budget.

    *closing* is appended to every chunk (see :data:`SUMMARY_CLOSING`) and, because it
    goes through ``render``, is counted by the fit test — so a caller that wants a closing
    pays for it in the packing rather than overflowing the budget afterwards.

    The common case (session fits in one pass) yields a single-element list.
    Adjacent chunks overlap by _CHUNK_OVERLAP exchanges; the RAG dedup fold
    handles repeated artifacts from chunk boundaries. Overlap is conditional: it
    is omitted whenever carrying it would overflow the next chunk.

    When *tokenizer* is provided, uses the server's loaded tokenizer for
    precise counting instead of the character estimate used client-side.
    """
    exchanges = session.get("exchanges", [])
    session_user = (session.get("user") or "").strip()

    items: list[dict] = []
    for i, ex in enumerate(exchanges):
        block = format_exchange_block(ex, session_user)
        if block:
            items.append({"exchange_index": i, "text": block,
                          "fragment_index": 1, "fragment_count": 1})

    # Packing assumes a multipart note even if the result later fits in one part.
    # That makes the test conservative and avoids a part-count/render feedback loop.
    def render(candidate: list[dict], part: int = 999, parts: int = 999) -> str:
        indices = [int(item["exchange_index"]) for item in candidate]
        return _with_open_questions(
            format_chunk_content(
                session, indices, session_idx, session_total, part, parts,
                items=candidate, closing=closing,
            ),
            open_questions_block,
        )

    if fits is None:
        budget = max(1, int(context_length * _CHUNK_BUDGET_FRAC))
        fits = lambda content: _count_tokens(content, tokenizer) <= budget

    def fits_items(candidate: list[dict]) -> bool:
        return bool(fits(render(candidate)))

    if not fits_items([]):
        raise ConsolidationChunkingError(
            "Consolidation instructions and session framing exceed the input budget."
        )

    atomic: list[dict] = []
    for item in items:
        atomic.extend([item] if fits_items([item]) else _fragment_item(item, fits_items))

    if not atomic:
        content = _with_open_questions(
            format_chunk_content(session, [], session_idx, session_total, 1, 1,
                                 items=[], closing=closing),
            open_questions_block,
        )
        return [{"indices": [], "items": [], "content": content,
                 "part": 1, "parts": 1, "overlap_applied": False}]

    groups: list[tuple[list[dict], bool]] = []
    cur: list[dict] = []
    cur_overlap = False
    for item in atomic:
        if not cur or fits_items(cur + [item]):
            cur.append(item)
            continue

        groups.append((cur, cur_overlap))
        carry: list[dict] = []
        if (_CHUNK_OVERLAP and cur and int(cur[-1].get("fragment_count") or 1) == 1
                and int(item.get("fragment_count") or 1) == 1):
            proposed = cur[-_CHUNK_OVERLAP:] + [item]
            if fits_items(proposed):
                carry = cur[-_CHUNK_OVERLAP:]
        cur = carry + [item]
        cur_overlap = bool(carry)
    if cur:
        groups.append((cur, cur_overlap))

    parts = len(groups)
    result: list[dict] = []
    for k, (group, overlap_applied) in enumerate(groups, 1):
        indices = [int(item["exchange_index"]) for item in group]
        content = render(group, k, parts)
        if not fits(content):
            raise ConsolidationChunkingError(
                f"Final consolidation chunk {k}/{parts} exceeded its prepared budget."
            )
        fragments = [item for item in group if int(item.get("fragment_count") or 1) > 1]
        result.append({
            "indices": indices,
            "items": group,
            "content": content,
            "part": k,
            "parts": parts,
            "overlap_applied": overlap_applied,
            "fragments": [
                {"exchange_index": int(item["exchange_index"]),
                 "fragment_index": int(item["fragment_index"]),
                 "fragment_count": int(item["fragment_count"])}
                for item in fragments
            ],
        })
    return result


def _selftest() -> None:
    short = {
        "timestamp": "20260712_120000",
        "user": "W",
        "system_prompt": "standing prompt",
        "exchanges": [
            {"user_prompt": "hello", "assistant_response": "hi"},
            {"user_prompt": "question", "assistant_response": "answer"},
        ],
    }
    chunks = build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) < 10_000
    )
    assert len(chunks) == 1
    assert chunks[0]["indices"] == [0, 1]
    assert chunks[0]["content"].count("System prompt in effect") == 1

    # A boundary overlap is optional: when two exchanges cannot coexist, the
    # second chunk must contain only the second exchange (the old bug made [0, 1]).
    base = len(format_chunk_content(short, [], 1, 1, 999, 999, items=[]))
    one_cost = len(format_exchange_block(short["exchanges"][0], "W"))
    tight_limit = base + one_cost + 20
    chunks = build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) <= tight_limit
    )
    assert len(chunks) == 2
    assert chunks[0]["indices"] == [0]
    assert chunks[1]["indices"] == [1]
    assert chunks[1]["overlap_applied"] is False

    # One irreducibly large exchange is fragmented in exact order rather than
    # accepted over budget or skipped. Fragments preserve the rendered exchange.
    long = {
        "timestamp": "20260712_130000",
        "user": "W",
        "system_prompt": "standing prompt",
        "exchanges": [{
            "user_prompt": "U" * 280 + "\n\n" + "V" * 280,
            "assistant_response": "A" * 420,
        }],
    }
    long_base = len(format_chunk_content(long, [], 1, 1, 999, 999, items=[]))
    fragment_limit = long_base + 210
    chunks = build_consolidation_chunks(
        long, 4096,
        open_questions_block="OPEN QUESTIONS\n- one\n",
        fits=lambda content: len(content) <= fragment_limit + len("OPEN QUESTIONS\n- one\n"),
    )
    assert len(chunks) > 1
    assert all(len(ch["content"]) <= fragment_limit + len("OPEN QUESTIONS\n- one\n")
               for ch in chunks)
    rendered = format_exchange_block(long["exchanges"][0], "W")
    fragments = [item["text"] for ch in chunks for item in ch["items"]]
    assert "".join(fragments) == rendered
    assert all(ch["content"].count("System prompt in effect") == 1 for ch in chunks)
    assert all(ch["content"].startswith("OPEN QUESTIONS") for ch in chunks)

    # The closing goes LAST — after the transcript, which is the whole point of it — and
    # is counted by the fit test rather than appended over budget afterwards.
    plain = build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) < 10_000)
    closed = build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) < 10_000,
        closing=SUMMARY_CLOSING)
    assert not plain[0]["content"].rstrip().endswith(SUMMARY_CLOSING)
    assert closed[0]["content"].rstrip().endswith(SUMMARY_CLOSING)
    assert "Me: answer" in closed[0]["content"]

    # Budgeted: the closing is paid for in the packing, not appended over budget after it.
    # Same construction as the overlap test above (the fit probe renders a multipart note,
    # so the base must be measured the same way) — a limit with room for exactly one
    # exchange must split, and every part must still end on the task.
    assert len(closed[0]["content"]) > len(plain[0]["content"])
    closed_base = len(format_chunk_content(short, [], 1, 1, 999, 999, items=[],
                                           closing=SUMMARY_CLOSING))
    closed_limit = closed_base + one_cost + 20
    budgeted = build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) <= closed_limit,
        closing=SUMMARY_CLOSING)
    assert len(budgeted) == 2, budgeted
    assert all(len(ch["content"]) <= closed_limit for ch in budgeted)
    assert all(ch["content"].rstrip().endswith(SUMMARY_CLOSING) for ch in budgeted)
    # ...and the closing is what pushed it over: the same limit without one fits in a
    # single chunk, so the split is the budget doing its job rather than a coincidence.
    assert len(build_consolidation_chunks(
        short, 4096, fits=lambda content: len(content) <= closed_limit)) == 1

    # An empty closing is exactly the old behaviour, byte for byte.
    assert append_closing("body", "") == "body"
    assert append_closing("body", "   ") == "body"

    # Fixed framing that cannot fit fails explicitly instead of looping.
    try:
        build_consolidation_chunks(short, 32, fits=lambda content: False)
    except ConsolidationChunkingError:
        pass
    else:
        raise AssertionError("expected fixed-framing budget failure")

    print("reflection_chunking self-test passed")


if __name__ == "__main__":
    _selftest()
