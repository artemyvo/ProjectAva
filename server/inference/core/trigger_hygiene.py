"""Recall-cue hygiene: keep a ``trigger`` a *cue*, not a pasted sentence.

A ``[fact]`` / ``[recollection]`` is indexed by its **trigger** rather than its content
(``ReflectionMemory.embed_text``) — the trigger is the answer to "what should bring this
back?". So a malformed trigger does not merely look untidy: it is the record's whole
retrieval key, and a fact keyed on the wrong text is unreachable by the topic it is
actually about while surfacing on a topic it is not.

**The producer that breaks them.** ``ReflectionWriter._distill_resolved`` sets a distilled
fact's trigger to `question` — the ``[ask]`` it answers — which is right when the ask is a
short cue-shaped question and wrong when it is one of Ava's own conversational openers.
Outreach / synthesis / check-in author those openers as whole paragraphs of addressed
speech, so on a live store 27 facts ended up keyed on 200–1745 characters of "Слушай, я
тут перечитывала наш разговор про …". One of them was the balcony fact that started this:
keyed on a monologue, it clustered with nothing (cosine 0.00–0.26 against every real cue),
so no dedup pass could ever see it beside its siblings.

**The rule is per cue, not per trigger.** ``fact_dedup`` merges by unioning triggers
(``a ; b ; c``), so a legitimate trigger grows long by *accumulating short cues* — measured
on that store, the 95th-percentile cue is 98 characters and the longest real one 127, while
the shortest pasted sentence is 209. A flat length test on the whole string would purge the
unions and keep nothing useful; splitting on the union separator first separates the two
populations cleanly, and lets a fused trigger keep its good cues and drop only the prose
part (which is what a merge with a broken record produces).

Two signals, both content-blind and language-agnostic — the store is mixed Russian/English
and the persona is emergent, so nothing here may key on *what* a cue says:
  * a cue longer than :data:`MAX_CUE_CHARS`, and
  * a cue ending in ``?`` or ``!`` — addressed speech, not a retrieval cue.

Deliberately NOT a sentence-boundary test ("a period followed by a capital"): calibrating
it against the live store immediately produced a false positive on ``coffee, American vs.
Israeli perspectives, quality standards``, and a rule that eats a good cue is worse than
one that misses a bad one — the bad one is caught by length on the next run, the good one
is gone. On that store the two rules together flag 27 of 680 triggers with no false
positive.

Pure and GPU-free: it decides, the caller writes. Used both as **prevention** (the writer
sanitizes at the point a cue is stored) and as a **purge** over records written before it
existed (``plan_trigger_purge`` → ``ReflectionWriter.write_trigger_purge``). A purge is an
append-only re-insert under the SAME key — ``content_key`` hashes content only, so a
trigger rewrite supersedes the record rather than forking it, and the fact itself is never
lost: stripped of its trigger it falls back to embedding on its content, which is a worse
cue than a good trigger and a far better one than a stranger's monologue.

GPU-free self-test: ``python -m core.trigger_hygiene``.
"""

from __future__ import annotations

import re
from typing import Optional

# The separator ``fact_dedup.merged_trigger`` joins a merged group's cues with. Splitting
# on it is what makes the length rule meaningful — see the module docstring.
SEPARATOR = " ; "

# Longest a single cue may be. Sits in the measured gap between the longest real cue (127)
# and the shortest pasted sentence (209) on the corpus this was calibrated against.
MAX_CUE_CHARS = 150

# Cues that are addressed speech rather than a topic.
_SPEECH_ENDINGS = ("?", "!")


def cue_problem(cue: str) -> str:
    """Why *cue* is not usable as a recall cue, or ``""`` when it is fine.

    The returned string is a human reason, carried onto the purge record so an operator
    reading the op-log can see what was dropped and why rather than just that something was.
    """
    text = (cue or "").strip()
    if not text:
        return ""                       # nothing to judge; an absent cue is not a broken one
    if len(text) > MAX_CUE_CHARS:
        return f"prose, not a cue ({len(text)} chars)"
    if text.endswith(_SPEECH_ENDINGS):
        return "addressed speech, not a cue"
    return ""


def clean_trigger(trigger: Optional[str]) -> tuple[Optional[str], list[tuple[str, str]]]:
    """Drop the unusable cues from *trigger*, keeping the rest in order.

    Returns ``(cleaned_or_None, [(dropped_cue, reason), …])``. ``None`` means every cue was
    unusable and the record should fall back to embedding on its content — deliberately
    None rather than a truncated cue, since the head of a monologue is still a monologue.
    """
    text = (trigger or "").strip()
    if not text:
        return None, []
    cues = [c.strip() for c in text.split(SEPARATOR) if c.strip()]
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for cue in cues:
        problem = cue_problem(cue)
        if problem:
            dropped.append((cue, problem))
        else:
            kept.append(cue)
    if not kept:
        return None, dropped
    return SEPARATOR.join(kept), dropped


def sanitize(trigger: Optional[str]) -> Optional[str]:
    """``clean_trigger`` keeping only the result — the writer's prevention hook."""
    cleaned, _dropped = clean_trigger(trigger)
    return cleaned


# Longest an [ask]'s stored content may be. An ask is a retrieval key twice over — it
# embeds on its question text, and when resolved it becomes the distilled fact's trigger
# (`ReflectionWriter._distill_resolved`) — so its shape has the same mechanical stakes a
# cue's does, one level up. Far looser than MAX_CUE_CHARS because a question is
# legitimately a sentence or two where a cue is a topic; measured on the live store's 90
# TIL-origin asks, 33 exceeded 300 chars (max 648), and everything past ~600 was opener
# wind-up around a closing question, not question.
MAX_ASK_CHARS = 600

# Sentence boundaries for the compaction below: a terminator followed by whitespace.
# Deliberately the crudest workable rule — the corpus is mixed Russian/English and the
# text is thrown away, not rewritten, so a missed boundary costs a few extra kept chars.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!…])\s+")


def compact_ask(content: Optional[str], max_chars: int = MAX_ASK_CHARS) -> str:
    """An over-long ask reduced to its live tail; a fitting one returned unchanged.

    The prompts ask for `[ask:user]` "as the opening you'd actually raise", so an
    over-long ask is almost always opener wind-up in front of the actual question — and
    the question closes it. Keeping the TRAILING whole sentences that fit therefore
    keeps the ask; keeping the head (an ordinary truncation) would keep the preamble and
    cut the question off. When even the final sentence exceeds the cap, its own tail is
    kept at a word boundary — the least-bad cut, since the interrogative core of a
    run-on question also sits at its end.

    Compaction, not judgement: nothing here decides whether the ask is worth carrying
    (the pool's ceilings do that), only that what is stored embeds as a question rather
    than as a paragraph. Whitespace is collapsed either way, matching what the writer
    stores.
    """
    text = " ".join(str(content or "").split())
    if len(text) <= max_chars:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        add = len(sentence) + (1 if kept else 0)
        if total + add > max_chars:
            break
        kept.insert(0, sentence)
        total += add
    if kept:
        return " ".join(kept)
    # The final sentence alone exceeds the cap: keep ITS tail on a word boundary.
    tail = text[-max_chars:]
    cut = tail.find(" ")
    return tail[cut + 1:] if 0 <= cut < len(tail) - 1 else tail


def plan_trigger_purge(items: list[dict]) -> list[dict]:
    """The records whose trigger needs rewriting, as a concrete purge plan.

    *items* are live reflection-memory records; only the trigger-indexed kinds are
    considered, since a trigger is inert on the others (persona/ask/impression embed on
    their display text, so a malformed one there costs nothing and rewriting it would be
    churn). Each entry names the record, the surviving cue and what was dropped. Pure — the
    caller either reports it (dry run) or hands it to ``ReflectionWriter.write_trigger_purge``.
    """
    plan: list[dict] = []
    for rec in items:
        if rec.get("kind") not in ("fact", "recollection"):
            continue
        trigger = (rec.get("trigger") or "").strip()
        if not trigger:
            continue
        cleaned, dropped = clean_trigger(trigger)
        if not dropped:
            continue
        plan.append({
            "key": rec.get("key"),
            "record": rec,
            "content": (rec.get("content") or "").strip(),
            "old_trigger": trigger,
            "new_trigger": cleaned,
            "dropped": [{"cue": c, "reason": r} for c, r in dropped],
        })
    return plan


def summarize(plan: list[dict]) -> list[dict]:
    """Client-facing view of a purge plan; drops the raw records."""
    return [{
        "content": p["content"][:200],
        "key": p["key"],
        "old_trigger": p["old_trigger"][:300],
        "new_trigger": p["new_trigger"],
        "dropped": p["dropped"],
    } for p in plan]


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    # Real cues from the live store must all survive — a rule that eats these is worse
    # than one that misses a monologue.
    good = [
        "balcony, stars, coffee, smoking, night atmosphere",
        "утро, кофе, сигареты, подготовка к дню",
        "coffee, American vs. Israeli perspectives, quality standards",   # the `vs.` trap
        "discussions about \"curing\" hyper-reflexivity, the struggle to stop thinking, "
        "or the desire for \"maximum desire/minimum thought\"",           # 127 chars, real
        "when I begin to over-analyze a shared goal or activity",
    ]
    for cue in good:
        assert not cue_problem(cue), f"false positive on a real cue: {cue!r}"
        assert clean_trigger(cue) == (cue, []), f"real cue not preserved: {cue!r}"

    # Pasted openers — the shape that actually broke.
    bad = [
        "Слушай, я тут на днях перечитывала наш разговор про «балконные ритуалы» и ту самую "
        "идею использования алкалоидов для ускорения работы нейронов. И знаешь, мне теперь "
        "жутко любопытно, а не кажется ли тебе, что мы оба немного переоцениваем этот эффект",
        "Artemy, looking back at our \"digital brownie\" manifesto, I'm curious: did the robot "
        "vacuum ever actually agree to those terms, or did he maintain a strict corporate "
        "policy of silence on the whole question of domestic espionage?",
        "did that ever actually work out?",          # short, but addressed speech
    ]
    for cue in bad:
        assert cue_problem(cue), f"missed a pasted sentence: {cue[:60]!r}"
        assert clean_trigger(cue)[0] is None, "a wholly-bad trigger must clear to None"

    # A fused trigger keeps its good cues and drops only the prose — the shape a merge
    # with an already-broken record produces (observed on the live store).
    fused = SEPARATOR.join([good[0], bad[0], good[1]])
    cleaned, dropped = clean_trigger(fused)
    assert cleaned == SEPARATOR.join([good[0], good[1]]), cleaned
    assert len(dropped) == 1 and "prose" in dropped[0][1], dropped

    # Order is preserved, so the survivor's primary topic stays first (merged_trigger's
    # contract: the survivor's own cue leads).
    assert clean_trigger(SEPARATOR.join(good[:3]))[0] == SEPARATOR.join(good[:3])

    # Absent / whitespace triggers are not "broken" — they are simply absent, and the
    # record already falls back to embedding on content.
    assert clean_trigger(None) == (None, [])
    assert clean_trigger("   ") == (None, [])
    assert not cue_problem("")

    # plan_trigger_purge: only the trigger-indexed kinds, only records needing a change.
    items = [
        {"key": "a", "kind": "fact", "content": "x", "trigger": good[0]},
        {"key": "b", "kind": "fact", "content": "y", "trigger": bad[0]},
        {"key": "c", "kind": "recollection", "content": "z", "trigger": fused},
        {"key": "d", "kind": "persona", "content": "w", "trigger": bad[0]},   # inert kind
        {"key": "e", "kind": "impression", "content": "v", "trigger": bad[1]},  # inert kind
        {"key": "f", "kind": "fact", "content": "u"},                          # no trigger
    ]
    plan = plan_trigger_purge(items)
    assert [p["key"] for p in plan] == ["b", "c"], [p["key"] for p in plan]
    assert plan[0]["new_trigger"] is None
    assert plan[1]["new_trigger"] == SEPARATOR.join([good[0], good[1]])
    assert summarize(plan)[1]["dropped"][0]["reason"].startswith("prose")

    # Idempotent: purging a purged store is a no-op (nothing re-flags).
    purged = [dict(p["record"], trigger=p["new_trigger"]) for p in plan]
    assert not plan_trigger_purge(purged), "purge is not idempotent"

    # compact_ask: a fitting ask is untouched (whitespace collapse aside)...
    q = "Ты правда думаешь, что стандарты качества — это про кофе?"
    assert compact_ask(q) == q
    assert compact_ask("a\n b   c") == "a b c"
    assert compact_ask(None) == ""
    # ...an opener-shaped over-long one keeps its trailing sentences — the question
    # closes an opener, so keeping the tail keeps the ask (an ordinary head-truncation
    # would keep the wind-up and cut the question off).
    windup = "Я тут перечитывала наш разговор, и вот что мне подумалось про всё это. " * 12
    over = windup + q
    got = compact_ask(over, max_chars=200)
    assert got.endswith(q), got
    assert len(got) <= 200, len(got)
    # Whole sentences only — never a mid-sentence fragment at the head.
    assert got == q or got[0].isupper() or got[0].isdigit(), got
    # ...and a single run-on sentence over the cap keeps ITS tail at a word boundary.
    runon = "слово " * 200 + "чем это закончилось?"
    got = compact_ask(runon, max_chars=100)
    assert got.endswith("чем это закончилось?"), got
    assert len(got) <= 100 and not got.startswith(" ")

    print("trigger_hygiene self-test OK")


if __name__ == "__main__":
    _selftest()
