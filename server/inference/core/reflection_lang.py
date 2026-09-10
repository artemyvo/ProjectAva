"""Language-drift detection for the reflection revision pass (pure, GPU-free).

A live reply can stay perfectly coherent yet slip mid-generation into a language the
user was not speaking (the "why is it suddenly Chinese/French" symptom) — a decode-time
attractor, not Ava's voice. The revision pass treats such a reply as never-hers: it
rewrites an IDEAL in the conversation's language and skips branch generation (every
counterfactual fork replays the same drifted answer prefix, so the whole blind choice
set would be in the wrong language — nothing to choose between).

The MODEL is the decider. It alone knows whether a switch was the user's request — a
translation task is legitimate and must be kept — so the runner honours its ``LANG_DRIFT``
marker over these helpers. What lives here is the cheap SCRIPT-level backstop that catches
a hard mismatch the model failed to flag, so it can be asked to look again.

Detection is deliberately *relative* — reply script vs. the conversation's script — never a
hard-coded "expected" language. (Contrast the Russian-pinned Tier-5 probe heuristic in
``training/train_cycle.py``, whose FIXME asks for exactly this relative treatment.)

Granularity is SCRIPT-FAMILY only, so it catches a switch ACROSS scripts (English↔Chinese,
Russian↔English) but NOT within one (English↔French, both Latin). Same-script drift is left
to the model's semantic judgement + its ``LANG_DRIFT`` marker. Japanese kana and kanji are
folded into one ``cjk`` family so a normal Japanese reply to a Japanese prompt is never
mistaken for drift.

Self-test: ``python -m core.reflection_lang``.
"""
from __future__ import annotations

import re
from typing import Iterable

# A leading, closed <think>…</think> block (an IDEAL carries one; the raw stored reply does
# not) — stripped before scoring so the ANSWER span, not the reasoning, decides the script.
_THINK_PREFIX_RE = re.compile(r"(?is)^\s*<think>.*?</think>\s*")

# (lo, hi, script) inclusive codepoint ranges, first match wins. Digits / punctuation /
# whitespace map to no script and are ignored (they carry no language signal).
_SCRIPT_RANGES = [
    (0x0041, 0x005A, "latin"), (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"),   # Latin-1 supplement + Extended-A/B (accents)
    (0x1E00, 0x1EFF, "latin"),   # Latin Extended Additional
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"), (0x0500, 0x052F, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x3040, 0x309F, "kana"), (0x30A0, 0x30FF, "kana"),   # hiragana / katakana
    (0xAC00, 0xD7A3, "hangul"), (0x1100, 0x11FF, "hangul"),
    (0x3400, 0x4DBF, "cjk"), (0x4E00, 0x9FFF, "cjk"), (0xF900, 0xFAFF, "cjk"),
    (0x20000, 0x2A6DF, "cjk"),
]

# Fold related scripts into one comparison family so a language that mixes scripts (Japanese
# = kana + kanji) is not flagged as drifting from itself. Identity for everything else.
_FAMILY = {"kana": "cjk"}


def _char_script(ch: str):
    o = ord(ch)
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    return None


def _family_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text or "":
        s = _char_script(ch)
        if s:
            fam = _FAMILY.get(s, s)
            counts[fam] = counts.get(fam, 0) + 1
    return counts


def dominant_script(text: str) -> str:
    """The script-family covering the most characters of *text*, or ``"none"``."""
    counts = _family_counts(text)
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def detect_language_drift(
    response: str,
    user_texts: Iterable[str],
    *,
    min_chars: int = 12,
    min_frac: float = 0.5,
    conv_min_frac: float = 0.8,
) -> tuple[bool, str, str]:
    """Whether *response* is dominantly in a different script-family than the conversation.

    Relative and symmetric: compares the reply's dominant script-family against the
    dominant family of the user's turns (*user_texts*). Returns
    ``(drift, conversation_family, response_family)``.

    Conservative on purpose — it is only a backstop to a model that already judges intent:

      * both sides must carry at least *min_chars* script characters (a one-word reply, or
        a code-only / numeric reply, is never flagged);
      * the reply must be at least *min_frac* in its dominant family (a stray foreign quote
        inside an otherwise-native reply is not drift);
      * the conversation must be at least *conv_min_frac* in its own family (a mixed-script
        conversation gives no clear "native" language to drift from).

    The two fractions are separate because they answer different questions, and only the
    second was ever really about mixing. *min_frac* asks "is this reply dominantly in X",
    where a bare majority is the right bar. *conv_min_frac* asks "does this conversation
    have a clear native language **at all**", and at a bare majority the answer was yes far
    too often: Latin is the script of code, file paths, identifiers, URLs and pasted logs,
    so a Russian conversation about this very repository counts 85 Latin characters against
    34 Cyrillic and is declared an English conversation — after which a correct Russian
    reply is flagged as drift. There is no symmetric effect (English conversations do not
    carry blocks of Cyrillic), which is why the failure reads as "it thinks everything is
    English".

    Raised to 0.8 rather than fixed by stripping code spans because the error costs here are
    asymmetric and the cheap direction is clear. A false negative is nearly free: the model's
    own ``LANG_DRIFT`` marker is the decider and this only catches what it missed. A false
    positive costs an extra full revision generation and, if the model does not push back on
    the recheck, a forced ``revise`` whose IDEAL gate then demands the wrong script — the
    re-answer is rejected twice and the exchange leaves the training corpus entirely.
    """
    resp = _THINK_PREFIX_RE.sub("", response or "")
    conv = " ".join(t for t in user_texts if t)
    r_counts = _family_counts(resp)
    c_counts = _family_counts(conv)
    r_total = sum(r_counts.values())
    c_total = sum(c_counts.values())
    conv_fam = max(c_counts, key=c_counts.get) if c_counts else "none"
    resp_fam = max(r_counts, key=r_counts.get) if r_counts else "none"
    if r_total < min_chars or c_total < min_chars:
        return False, conv_fam, resp_fam
    if resp_fam == conv_fam:
        return False, conv_fam, resp_fam
    if r_counts[resp_fam] / r_total < min_frac:
        return False, conv_fam, resp_fam
    if c_counts[conv_fam] / c_total < conv_min_frac:
        return False, conv_fam, resp_fam
    return True, conv_fam, resp_fam


# ── self-test ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    # English conversation, Chinese reply → drift.
    d, c, r = detect_language_drift(
        "这是一个完全连贯的中文回答，讲述了很多有趣的内容。",
        ["Can you tell me about the history of tea?"])
    assert d and c == "latin" and r == "cjk", (d, c, r)

    # English conversation, English reply → no drift.
    d, _, _ = detect_language_drift(
        "Tea has a long and fascinating history spanning many centuries.",
        ["Can you tell me about the history of tea?"])
    assert not d

    # Russian conversation, English reply → drift (cyrillic vs latin).
    d, c, r = detect_language_drift(
        "Actually, I think the whole premise here is a little off and here is why.",
        ["Расскажи, что ты думаешь об этом вопросе подробнее?"])
    assert d and c == "cyrillic" and r == "latin", (d, c, r)

    # Russian conversation, Russian reply → no drift.
    d, _, _ = detect_language_drift(
        "Я думаю, что это довольно интересный вопрос, и вот моя точка зрения.",
        ["Расскажи, что ты думаешь об этом вопросе подробнее?"])
    assert not d

    # Japanese reply to a Japanese prompt (kana + kanji) → NOT drift (folded to cjk).
    d, _, _ = detect_language_drift(
        "これはとても面白い質問ですね。私の考えを説明します。",
        ["お茶の歴史について教えてください。"])
    assert not d

    # A stray foreign quotation inside an English reply → NOT drift (below min_frac).
    d, _, _ = detect_language_drift(
        "The phrase 你好 just means hello, but the rest of my answer is in English here.",
        ["What does that phrase mean in English?"])
    assert not d

    # IDEAL carrying a <think> block: the answer span, not the reasoning, decides.
    d, _, _ = detect_language_drift(
        "<think>reasoning in english here</think>\n\nOui, je pense vraiment que c'est le cas.",
        ["Do you agree with that?"])
    # Same Latin family (English vs French) → script backstop cannot see it (by design).
    assert not d

    # Too-short reply → never flagged.
    d, _, _ = detect_language_drift("好", ["Tell me a long story about anything at all."])
    assert not d

    _RU_REPLY = ("Мне кажется, здесь важнее не сам механизм, а то, что он молча "
                 "меняет смысл записи — и мы этого не видим до самого конца.")

    # A Russian conversation ABOUT code: identifiers, paths and a pasted log line put more
    # Latin characters in the user's turn than Cyrillic. This is the observed misfire —
    # the conversation was read as English and a correct Russian reply as drift.
    d, c, _ = detect_language_drift(_RU_REPLY, [
        "посмотри на build_revision_content и на _language_decision_guard — почему "
        "LANG_DRIFT срабатывает? файл server/inference/core/reflection_lang.py"])
    assert not d, (d, c)
    d, _, _ = detect_language_drift(_RU_REPLY, [
        "вот лог:\n[03:30:18] [phase_done] 20260725_221755.json Revision pass done "
        "(exchange 2/4)\nпочему тут unparseable?"])
    assert not d

    # …but a conversation genuinely in one script still yields drift on a reply in another,
    # in both directions. Raising the bar must not cost the true positives.
    d, c, r = detect_language_drift(
        "Actually, I think the whole premise here is a little off and here is why it is.",
        ["Расскажи, что ты думаешь об этом вопросе подробнее? Мне правда интересно "
         "услышать твоё мнение."])
    assert d and c == "cyrillic" and r == "latin", (d, c, r)
    d, c, r = detect_language_drift(_RU_REPLY, [
        "Can you tell me what you actually think about this question? I am curious."])
    assert d and c == "latin" and r == "cyrillic", (d, c, r)

    # The stage-direction exclusion is the caller's half (reflection_source.
    # conversation_user_texts) — without it, an Ava-initiated session's English impulse is
    # the whole "conversation" and no threshold here can help. Both halves together:
    from core.reflection_source import conversation_user_texts
    job = {
        "context": [
            {"role": "user", "speaker": "(initiative)",
             "content": ("About 7 hours had passed since you last spoke with Artemy. "
                         "You have been sitting with something and want to say it.")},
            {"role": "assistant", "content": "Слушай, я всё думаю про ту штуку."},
        ],
        "speaker": "Artemy",
        "exchange": {"speaker": "Artemy", "user_prompt": "да, я как раз думал об этом"},
    }
    texts = conversation_user_texts(job)
    assert texts == ["да, я как раз думал об этом"], texts
    d, c, _ = detect_language_drift(_RU_REPLY, texts)
    assert not d and c == "cyrillic", (d, c)

    print("reflection_lang self-test passed")
