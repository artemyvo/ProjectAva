"""Verbatim target assembly — the single source of truth for building a trainable
assistant turn as ``<think>{cot}</think>\n{answer}``.

Formerly the reflection-time ShareML producer (a per-session ``.shareml.json`` training
document + verbatim variant copies). **That artifact is retired** — nothing ever read it
(the from-scratch build reads the chat sidecars directly), and it encoded the now-removed
variant-count / decay-stage concepts (REBUILD). Only the target-assembly helper survives,
reused by ``training.dialogue_source.build_dialogue_anchor`` so a trained target is
byte-identical to how the exchange was stored (no drift between the two).
"""
from __future__ import annotations

import re

# A leading ``<think>…</think>`` block at the head of an assistant turn. Vetted sidecar
# targets sometimes store the *full* original assistant output (CoT + answer), not the
# bare answer this builder expects — so we strip the embedded block before re-wrapping
# with the canonical CoT below.
_LEADING_THINK_RE = re.compile(r"(?is)^\s*<think>.*?</think>\s*")


def _answer_only(target_response: str) -> str:
    """The bare answer of a target, with any leading ``<think>…</think>`` block removed.

    ``_verbatim_assistant`` adds the CoT itself, so its ``target_response`` must be
    answer-only. A vetted ``keep`` target, however, is the verbatim original assistant
    turn and still carries its own ``<think>`` block. Wrapping that as-is produces a
    NESTED double-think (``<think>cot</think>\\n<think>cot</think>answer``); on gemma-4
    only the first block becomes the native ``<|channel>`` reasoning channel, so the
    second survives as literal ``<think></think>`` text inside the trained answer span —
    which teaches the model to emit empty think blocks and, over cycles, to loop them
    instead of answering. Stripping the embedded block keeps exactly one CoT.

    Some stored targets carry **two (or more) back-to-back** ``<think>…</think>`` blocks
    (models occasionally emit a second channel; see reflection_writer's multi-block
    handling). A single ``count=1`` substitution would strip only the first and leave the
    rest at the head of the answer, which then trips render's double-think guard. Strip
    every consecutive leading block instead.
    """
    answer = target_response or ""
    while True:
        stripped = _LEADING_THINK_RE.sub("", answer, count=1)
        if stripped == answer:
            return stripped.strip()
        answer = stripped


def _verbatim_assistant(assistant_cot: str, target_response: str) -> str:
    """Assemble ``<think>{cot}</think>\n{answer}`` (or bare answer when there is no CoT)."""
    assistant_cot = (assistant_cot or "").strip()
    answer = _answer_only(target_response)
    if assistant_cot:
        return f"<think>{assistant_cot}</think>\n{answer}"
    return answer
