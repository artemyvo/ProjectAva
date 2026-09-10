"""One definition of "strip the reasoning trace, and refuse it if it leaked".

A **leaf** like ``field_parse`` / ``activity_log`` / ``reachout_gate``: imports nothing
from the project, imported directly, no ``configure``-injection, no cycle.

Every pass that asks the model for labelled fields must read those labels out of the
ANSWER, never out of the ``<think>`` CoT — the reasoning discusses its own choice in the
same words the contract uses ("Decision: yes.", ``Format: `OPENER: <message>` ``), so a
parse over the raw output matches the deliberation instead of the conclusion. Stripping
the trace is therefore a correctness step, not cosmetics.

It had drifted into three versions. ``synthesis`` and ``deliberation`` handled all four
shapes the normalizers can leave; ``outreach`` and ``checkin`` handled two, and neither
carried the leak backstop — while being the only two modules that WRITE the parsed text
into a chat session the user opens. That asymmetry is the defect this module exists to
remove: the complete version, once, for all callers.

**The shape that motivates all of it** (`model_family._normalize_gemma`): gemma-4's
reasoning channel is *prefilled* into the prompt, so a generation that truncates inside
it emits neither ``<|channel>`` nor ``<channel|>``. The normalizer keys on those markers,
finds neither, and returns the text untouched — so a truncated CoT arrives as **untagged
prose with no markers at all**, indistinguishable by inspection from a real answer.
:func:`answer_after_think` cannot detect it (there is nothing to strip), which is why
:func:`truncated_before_answer` exists and must be checked by the caller against the
generator's own truncation flag.

GPU-free self-test: ``python -m core.reasoning_text``.
"""
from __future__ import annotations

import re

# A complete, properly-tagged reasoning block.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Stray reasoning-channel tokens a normalizer can leave behind (gemma-4 `<|channel>` /
# `<channel|>`, the `<|think|>` directive).
_CHANNEL_STRAY_RE = re.compile(r"<\|channel>[^\n]*\n?|<channel\|>|<\|think\|>", re.IGNORECASE)
# Any of these surviving in text bound for a chat message means reasoning leaked through.
REASONING_LEAK_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>", "<|think|>")
# The marker whose presence proves the model closed its reasoning and began an answer.
_CLOSE_MARKER = "</think>"


def answer_after_think(text: str) -> str:
    """Return *text* with the reasoning trace removed, in four passes.

    Robust to every shape the normalizers can leave:

    1. a complete ``<think>…</think>`` block (the ordinary case);
    2. an unclosed ``<think>`` tail — a truncated thought, whose tail is dropped so a
       half-finished deliberation cannot masquerade as the answer;
    3. a bare closing ``</think>`` with no opener — a prompt-prefilled / qwen-shaped
       trace whose opener the normalizer did not re-add; only what follows the LAST one
       is kept;
    4. stray channel tokens left by a partial normalization.

    NOT detectable here: a gemma-4 channel truncated before its close, which normalizes
    to untagged prose carrying no marker of any kind. See :func:`truncated_before_answer`.
    """
    text = _THINK_BLOCK_RE.sub("\n", text or "")
    idx = text.lower().find("<think>")
    if idx != -1:
        text = text[:idx]
    close = text.lower().rfind(_CLOSE_MARKER)
    if close != -1:
        text = text[close + len(_CLOSE_MARKER):]
    return _CHANNEL_STRAY_RE.sub("", text).strip()


def has_reasoning_leak(text: str) -> bool:
    """True if *text* still carries a reasoning marker — a last backstop before sending.

    Call it on anything about to be written into a chat message. A hit means the text is
    not a clean message whatever else parsed, and it must not reach the user."""
    low = (text or "").lower()
    return any(marker in low for marker in REASONING_LEAK_MARKERS)


def truncated_before_answer(raw: str, truncated: bool) -> bool:
    """True when a generation was cut off and never closed its reasoning.

    The reliable signal is the boundary, not the content: when the model finishes
    thinking it emits a close marker and a real answer follows it, and labels parsed from
    that region are trustworthy even if generation later hit the cap. With no close
    marker and a truncated generation there IS no answer region — so whatever a label
    regex matched, it matched inside the deliberation. The caller must refuse it.

    This is the check that cannot be folded into :func:`answer_after_think`: it needs the
    generator's truncation flag (``generate.last_truncated``), which the text alone does
    not carry."""
    return bool(truncated) and _CLOSE_MARKER not in (raw or "").lower()


if __name__ == "__main__":
    # 1. ordinary tagged block
    assert answer_after_think("<think>Decision: yes.</think>DECISION: no") == "DECISION: no"
    # 2. unclosed <think> tail is dropped, not returned as the answer
    assert answer_after_think("DECISION: no\n<think>wait, maybe yes") == "DECISION: no"
    # 3. bare close with no opener — keep only what follows the LAST one
    assert answer_after_think("thinking...</think>DECISION: yes") == "DECISION: yes"
    assert answer_after_think("a</think>b</think>FINAL") == "FINAL"
    # 4. stray channel tokens stripped — the MARKERS only, not the prose between them.
    #    A complete `<|channel>…<channel|>` pair is `_normalize_gemma`'s job and has been
    #    rewritten to <think>…</think> long before this; what reaches here is residue.
    assert answer_after_think("<|think|>OPENER: hi") == "OPENER: hi"
    assert answer_after_think(
        "<|channel>thought\nweighing\n<channel|>OPENER: hi") == "weighing\nOPENER: hi"
    # glued close marker (gemma) still separates
    assert answer_after_think("<think>x</think>DECISION: yes").startswith("DECISION:")
    # empty / None-ish
    assert answer_after_think("") == ""
    assert answer_after_think(None) == ""

    # leak backstop
    assert has_reasoning_leak("<think>oops")
    assert has_reasoning_leak("done</think>")
    assert has_reasoning_leak("<channel|>hi")
    assert not has_reasoning_leak("Привет! Как ты?")
    assert not has_reasoning_leak("")

    # the undetectable-by-text case: untagged truncated reasoning
    cot = "I should weigh this. Decision: yes. Format: OPENER: <message>"
    assert answer_after_think(cot) == cot, "no markers -> nothing to strip, by design"
    assert truncated_before_answer(cot, True), "must be refused via the truncation flag"
    assert not truncated_before_answer(cot, False)
    # a closed boundary is trusted even when generation later hit the cap
    assert not truncated_before_answer("<think>x</think>DECISION: yes", True)
    assert not truncated_before_answer("thinking</think>DECISION: yes", True)

    print("reasoning-text self-test passed ✓")
