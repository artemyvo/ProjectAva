"""The chat kind (§1.3–§1.5): one exchange per primary unit, never split.

Input text is JSON: ``{"user": "...", "exchanges": [{"user_prompt", "assistant_response",
"speaker"?, "ts"?, "assistant_cot"?}, ...]}`` — the shape of an Ava transcript, CoT
stripped at render. The rendered document is the transcript as a reader sees it; spans
index into that. Append-only: re-ingesting the same key with more exchanges keeps every
existing chunk id (path = exchange index, text unchanged) and appends.
"""

from __future__ import annotations

import json

from . import KindSpec, register, FACET_PROPERTY, FACET_POSITION, FACET_EVENT, FACET_UNCLASSIFIED
from ..chunks import make_unit

ME = "Me"


def render_transcript(doc: dict) -> tuple[str, list[dict]]:
    """Render exchanges as text; return (text, [{index, span, speaker, ts}])."""
    parts: list[str] = []
    marks: list[dict] = []
    pos = 0
    for i, ex in enumerate(doc.get("exchanges") or []):
        speaker = str(ex.get("speaker") or doc.get("user") or "User").strip()
        user = str(ex.get("user_prompt") or "").strip()
        reply = str(ex.get("assistant_response") or "").strip()
        block = f"{speaker}: {user}\n{ME}: {reply}"
        start = pos
        parts.append(block)
        pos += len(block)
        marks.append({"index": i, "span": (start, pos), "speaker": speaker, "ts": ex.get("ts") or ex.get("timestamp") or ""})
        parts.append("\n\n")
        pos += 2
    return "".join(parts), marks


def parse_chat_text(text: str) -> dict:
    try:
        doc = json.loads(text)
    except Exception:
        doc = {"exchanges": []}
    if isinstance(doc, list):
        doc = {"exchanges": doc}
    return doc


def _split(text: str, meta: dict):
    doc = parse_chat_text(text)
    rendered, marks = render_transcript(doc)
    units = []
    for m in marks:
        units.append(make_unit(
            meta.get("key", ""), role="primary", path=[f"exchange {m['index']}"], span=m["span"],
            text=rendered[m["span"][0]:m["span"][1]], unit_type="exchange",
            keys={"index": m["index"], "speaker": m["speaker"], "ts": m["ts"], "user": doc.get("user") or ""}))
    return rendered, units


CHAT = register(KindSpec(
    name="chat", split=_split, witnesses=("llm",),
    classes=("standing", "stated", "event"),
    facet_map={"standing": FACET_PROPERTY, "stated": FACET_POSITION, "event": FACET_EVENT,
               "unspecified": FACET_UNCLASSIFIED},
    # Thinking OFF as measured 2026-09-09: with it on, gemma-4-31B at 4.5 tok/s thought past a
    # 4096-token cap on 3 of 6 chat groups (a failed pass each, 45 min per chat) and the
    # Russian chat that did finish gained nothing over the plain read. The design's reason
    # for thinking (pronouns across turns) is kept as the `thinking` knob for a faster box.
    namespace="person", clock="exchange", prompt_file="witness_chat.txt", thinking=False,
    split_oversize=False,
))
