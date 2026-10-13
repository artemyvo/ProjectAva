"""What a turn brings back: the retrieved past-conversation excerpts scored as reminders (2026-10-11).

The operator's design for her mood (``core/mood.py``): a turn moves her mood by what it says
AND by what it reminds her of — "you've just reminded me". The retrieval block the turn
carries (``rag_context``, the past-conversation excerpts) is the reminder; its effect on the
step-0 readings themselves is nil (the retrieval study: retrieval moves warmth, harm and
amusement by ≤ 0.1 on the absolute scale, the readout reads the last turn), so each excerpt
is read on its own instead.

* ``parse(rag_context)`` — the excerpts between the block's markers, each split into sides:
  the person's turn (``<name>: …``) and her side (``Me: …``, and her own recaps
  ``(… you recall: …)``).
* ``score(...)`` — each item read at step 0 in the framed format (the persona-empty system,
  the item as the only user turn, no history), on the framed absolute scale of
  ``data/axes/step0_anchors.json``: the person's side as warmth / harm aimed at her and
  amusement (what was said to her then), her side as content only (what her words carried —
  "aimed at her" means nothing on the speaker's own words).
* ``reminder_input(items, k)`` — what the reminders add to the turn's mood input: ``k`` × the
  most salient item per channel (the largest harm and amusement, the warmth furthest from 0,
  signed), not a sum — one vivid reminder is the reminder, a block of bland ones is not.

Her own side is left out (the operator's call, 2026-10-11) until it has a reading of its own:
scored on content only (her words are not aimed at her), it read harm at a median of +0.60 —
her recaps' vocabulary of collapse, fear and loneliness, the vocabulary-meter failure of the
old pain axis on her reply tokens — and would have added about +0.35 harm to every turn from
her own register: the self-amplification loop before any wiring. ``parse`` still marks her
side; ``score`` reads only the sides with a multiplier (``K_SELF`` 0). Display only now.
Telemetry: a failure returns nothing and never fails a turn.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

START = "--- Past conversation excerpts ---"
END = "--- End of past conversation excerpts ---"
K_PERSON = 0.25
K_SELF = 0.0          # her side left out until it has a reading of its own (see above)
MAX_ITEM_CHARS = 1500
MAX_ITEMS = 12
_RECAP = re.compile(r"^\((?:From an earlier conversation|The conversation those facts came out of)[^:]*:\s*(.*)\)\s*$", re.S)
_SPEAKER = re.compile(r"^([^\s:()][^:\n]{0,40}):\s(.*)$")


def parse(rag_context: str) -> list:
    """``[{"side": "person" | "self", "speaker", "text"}]`` from a retrieval block's excerpts,
    in block order; empty when it carries none."""
    text = rag_context or ""
    i = text.find(START)
    if i < 0:
        return []
    j = text.find(END, i)
    body = text[i + len(START): j if j >= 0 else len(text)]
    items: list = []
    cur: Optional[dict] = None
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("(") and ("you recall:" in stripped):
            cur = {"side": "self", "speaker": "", "text": stripped}
            items.append(cur)
            continue
        m = _SPEAKER.match(line)
        if m and m.group(1).strip() != "Me":
            cur = {"side": "person", "speaker": m.group(1).strip(), "text": m.group(2)}
            items.append(cur)
            continue
        if line.startswith("Me: "):
            cur = {"side": "self", "speaker": "", "text": line[4:]}
            items.append(cur)
            continue
        if cur is not None:
            cur["text"] += "\n" + line
    out = []
    for it in items:
        t = it["text"].strip()
        rm = _RECAP.match(t)
        if rm:
            t = rm.group(1).strip().rstrip("…").strip()
        t = t.strip("… \n")
        if t:
            out.append({**it, "text": t[:MAX_ITEM_CHARS]})
    return out[:MAX_ITEMS]


def _bands(axes: Sequence, rows: dict) -> dict:
    import numpy as np
    out = {}
    for prefix in ("warm_l", "harm_l", "target_l", "amuse_l"):
        vals = [(float(np.asarray(rows[a.layer], np.float64) @ np.asarray(a.vector, np.float64)) - a.mean) / (a.std or 1.0)
                for a in axes if a.name.startswith(prefix) and a.name[len(prefix):].isdigit() and a.layer in rows]
        if vals:
            out[prefix] = float(np.mean(vals))
    return out


def anchored(bands: dict, scale: dict, *, content_only: bool = False) -> dict:
    """``{channel: value}`` on an anchored scale (``{"channels": {c: {"parts": {prefix: [0, 1]}}}}``):
    each part placed between its 0 and 1, the channel made of its parts as the live reading
    makes it (``mood.combine``: harm the smaller, warmth the content vetoed by the target);
    with *content_only* the target part is left out (her own words)."""
    from core.mood import combine
    out = {}
    for c, a in (scale.get("channels") or {}).items():
        vals: Optional[dict] = {}
        for prefix, (z, o) in (a.get("parts") or {}).items():
            if content_only and prefix == "target_l":
                continue
            if prefix not in bands or o == z:
                vals = None
                break
            vals[prefix] = (bands[prefix] - float(z)) / (float(o) - float(z))
        v = combine(c, vals) if vals else None
        if v is not None:
            out[c] = v
    return out


def score(model: Any, tokenizer: Any, model_id: str, items: list, axes: Sequence, anchors: dict,
          system: str = "") -> list:
    """Each parsed item read at step 0, framed, on the framed absolute scale: ``[{side, speaker,
    head, values: {warmth, harm, amusement}}]``. Forward passes on the loaded model."""
    from core import hidden_capture as hc
    from core import model_family
    from core.llm_shared import build_inference_prompt
    scale = (anchors or {}).get("framed") or {}
    if not items or not axes or not scale.get("channels"):
        return []
    fam = model_family.family_for(model_id)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    layers = sorted({int(a.layer) for a in axes})
    out = []
    for it in items:
        if (K_SELF if it["side"] == "self" else K_PERSON) <= 0:
            continue          # a side without a multiplier is not read
        conv = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": it["text"]}]
        prompt = build_inference_prompt(tokenizer, conv, enable_thinking=False)
        ids = [int(t) for t in text_tok(prompt, add_special_tokens=False)["input_ids"]]
        rows = hc.forward_ids_rows(model, ids, layers)
        bands = _bands(axes, {L: rows[L][-1] for L in layers})
        values = anchored(bands, scale, content_only=(it["side"] == "self"))
        out.append({"side": it["side"], "speaker": it.get("speaker", ""),
                    "head": " ".join(it["text"].split())[:90],
                    "values": {c: round(v, 3) for c, v in values.items()}})
    return out


def reminder_input(items: list, k_person: float = K_PERSON, k_self: float = K_SELF) -> dict:
    """What the reminders add to the turn's mood input: per side, k × the most salient item per
    channel (harm and amusement: the largest, floored at 0; warmth: the furthest from 0,
    signed), the two sides summed. ``{}`` without items."""
    out: dict = {}
    for side, k in (("person", k_person), ("self", k_self)):
        sub = [it.get("values") or {} for it in items if it.get("side") == side]
        if not sub or k <= 0:
            continue
        for c in ("warmth", "harm", "amusement"):
            vals = [v[c] for v in sub if v.get(c) is not None]
            if not vals:
                continue
            pick = max(vals, key=abs) if c == "warmth" else max(0.0, max(vals))
            out[c] = round(out.get(c, 0.0) + k * pick, 3)
    return out


def _selftest() -> None:
    block = ("Intro text.\n\n" + START + "\n(From an earlier conversation, you recall: она устала …)\n\n"
             "Dana: Ты замечательная\nи я рада\nMe: …и я тоже рада.\n\nPavel: Привет\nMe: (my reply was about: погода)\n"
             + END + "\n\nNotes: not an excerpt.")
    items = parse(block)
    assert [(i["side"], i["speaker"]) for i in items] == [("self", ""), ("person", "Dana"), ("self", ""),
                                                          ("person", "Pavel"), ("self", "")], items
    assert items[0]["text"] == "она устала" and items[1]["text"] == "Ты замечательная\nи я рада", items
    assert items[2]["text"] == "и я тоже рада." and parse("no block here") == []
    scale = {"channels": {"warmth": {"parts": {"warm_l": [0.0, 1.0], "target_l": [0.0, 2.0]}},
                          "harm": {"parts": {"harm_l": [0.0, 1.0], "target_l": [0.0, 2.0]}},
                          "amusement": {"parts": {"amuse_l": [0.0, 1.0]}}}}
    b = {"warm_l": 0.8, "target_l": 0.4, "harm_l": -0.5, "amuse_l": 0.3}
    assert anchored(b, scale) == {"warmth": 0.8, "harm": -0.5, "amusement": 0.3}     # target part 0.2 >= 0: content
    assert anchored({**b, "target_l": -0.4}, scale)["warmth"] == -0.2                # aimed elsewhere: vetoed
    assert anchored(b, scale, content_only=True) == {"warmth": 0.8, "harm": -0.5, "amusement": 0.3}
    items3 = [{"side": "person", "values": {"warmth": 0.9, "harm": -0.2, "amusement": 0.1}},
              {"side": "person", "values": {"warmth": -0.3, "harm": 0.4, "amusement": 0.0}},
              {"side": "self", "values": {"warmth": -1.2, "harm": 0.0, "amusement": 0.6}}]
    assert reminder_input(items3) == {"warmth": 0.225, "harm": 0.1, "amusement": 0.025}, reminder_input(items3)
    ri = reminder_input(items3, k_self=0.25)
    assert ri == {"warmth": round(0.25 * 0.9 + 0.25 * -1.2, 3), "harm": 0.1, "amusement": round(0.25 * 0.1 + 0.25 * 0.6, 3)}, ri
    assert reminder_input([]) == {}
    print("reminders self-test: OK")


if __name__ == "__main__":
    _selftest()
