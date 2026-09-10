"""Render an anchor (+ a regenerated variant) into a trainable conversation.

The single rule here is **train/inference parity**: the conversation we train on
must be assembled exactly the way the live server assembles it for generation
(``server._build_inference_conversation``) — system message, then each user turn
prefixed ``{speaker}: {content}`` (bare when no speaker), then the assistant turn.
Any drift between the two is a silent SFT bug, so the builder is mirrored here and
guarded by :func:`assert_parity` in the self-test.

The assistant target is the regenerated variant text *as the model would emit it*.
Stored targets carry the inference-normalized ``<think>…</think>`` + answer form. For
chatml models that *is* the native form, so it trains as-is. For gemma-4 the model
emits a special-token reasoning channel (``<|channel>…<channel|>``) the chat template
strips from content, so :func:`render_example_text` rewrites the target back into that
channel (:func:`to_gemma_thinking_channel`) and builds the prompt with
``enable_thinking`` to match inference. Loss is masked to the assistant turn only; that
masking is applied by the trainer via the chat template's response marker
(``train_on_responses_only``), so this module only has to produce the rendered text.
"""

from __future__ import annotations

import re
from typing import Optional

_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
# A leading ``<think>…</think>`` block plus everything after it — the same split
# ``to_gemma_thinking_channel`` performs to find the answer span.
_LEADING_THINK_BLOCK_RE = re.compile(r"(?s)\s*<think>(.*?)</think>(.*)", re.IGNORECASE)

# A revision field label at the start of a line. A clean target never carries one in its
# answer span, but a legacy sidecar written before ``reflection_writer`` stripped them can
# hold a stray trailing ``LANG_DRIFT: no`` (or repeated ``VERDICT:``/``PERSONA_TARGET:``)
# that the greedy IDEAL slice swept in. A from-scratch build reads every frozen sidecar, so
# this gate scrubs the answer one last time before the optimizer sees it. Kept local to
# avoid a training→inference import (mirrors ``reflection_writer._TRAILING_FIELD_RE``).
_TRAILING_FIELD_RE = re.compile(
    r"(?im)^[ \t]*[*_]*[ \t]*(?:VERDICT|WHY|PERSONA_TARGET|PERSONA|LANG_DRIFT|IDEAL):"
)


def _strip_trailing_fields(answer: str) -> str:
    """Cut a stray revision field label (and everything after) from an answer span."""
    m = _TRAILING_FIELD_RE.search(answer or "")
    return answer[: m.start()].rstrip() if m else answer


def answer_of(text: str) -> str:
    """The answer portion of a ``<think>…</think>answer`` string (for similarity)."""
    m = _THINK_CLOSE_RE.search(text or "")
    return text[m.end():].strip() if m else (text or "").strip()


def trainable_answer(text: str) -> Optional[str]:
    """The trainable answer span of a target, or ``None`` if it carries no usable answer.

    Mirrors the split ``to_gemma_thinking_channel`` performs: a leading *closed*
    ``<think>…</think>`` yields its trailing answer; a target with no think block is
    itself the answer. Returns ``None`` when that span is empty or still holds a literal
    ``<think>``/``</think>`` — i.e. an **unclosed**, **CoT-only**, or **doubled-think**
    target that has nothing to train toward. Callers drop such anchors from the cycle
    instead of letting them trip render's hard ``_assert_no_think`` guard mid-train and
    abort the whole run for one malformed reflection output.
    """
    m = _LEADING_THINK_BLOCK_RE.match(text or "")
    answer = _strip_trailing_fields((m.group(2) if m else (text or "")).lstrip("\n").strip())
    if not answer or "<think>" in answer or "</think>" in answer:
        return None
    return answer


def has_cot(text: str) -> bool:
    """True when *text* carries a closed ``<think>`` block with **non-empty** thought
    content followed by an answer.

    The thought must be non-empty: ``<think></think>answer`` is gemma-4's
    "thinking-enabled-but-no-thought" form (an empty reasoning channel). Counting it as
    CoT lets an adapter that has collapsed into emitting empty channels sail through the
    probe's CoT-survival gate (``train_cycle`` Tier 2) while real reasoning has already
    eroded — which is exactly the regression that gate exists to catch. Requiring a
    non-empty thought makes the probe able to see the erosion.
    """
    m = _LEADING_THINK_BLOCK_RE.search(text or "")
    if not m or not (m.group(1) or "").strip():
        return False
    return bool(answer_of(text))


# Synthetic stage-direction speakers — see generation._NARRATOR_SPEAKERS. Rendered as
# a marked stage direction (no speaker label) so a reversed-session opener / encounter
# framing is not read as a peer utterance. Kept in lock-step with
# generation._build_inference_conversation so the training-render, IDEAL-generation,
# and live-inference prefixes are identical (parity by construction).
_NARRATOR_SPEAKERS = frozenset({"(initiative)", "(setting)"})
_STAGE_DIRECTION_TAG = "(stage direction — you, not another person)"


def _user_content(content: str, speaker: str) -> str:
    speaker = (speaker or "").strip()
    content = content or ""
    if speaker in _NARRATOR_SPEAKERS:
        return f"{_STAGE_DIRECTION_TAG} {content}".strip()
    return f"{speaker}: {content}" if speaker else content


def build_messages(anchor: dict, target_text: str) -> list[dict]:
    """Assemble the chat-format messages for one training example.

    *anchor* carries ``system_prompt``, ``context`` (list of
    ``{role, content, speaker}``), ``prompt`` and ``speaker``; *target_text* is the
    assistant turn to train toward (a regenerated variant, ``<think>…</think>`` +
    answer). Mirrors ``_build_inference_conversation``.
    """
    messages = [{"role": "system", "content": anchor.get("system_prompt", "") or ""}]
    for turn in anchor.get("context", []) or []:
        if turn.get("role") == "user":
            messages.append({"role": "user",
                             "content": _user_content(turn.get("content", ""), turn.get("speaker", ""))})
        else:
            messages.append({"role": "assistant", "content": turn.get("content", "")})
    messages.append({"role": "user",
                     "content": _user_content(anchor.get("prompt", ""), anchor.get("speaker", ""))})
    messages.append({"role": "assistant", "content": target_text})
    return messages


_THINK_BLOCK_RE = re.compile(r"(?s)\s*<think>(.*?)</think>(.*)", re.IGNORECASE)


def to_gemma_thinking_channel(content: str) -> str:
    """Rewrite an assistant turn into gemma-4's native reasoning channel.

    - ``<think>X</think>Y`` -> ``<|channel>thought\\nX\\n<channel|>Y`` (real thought).
    - answer-only ``Y`` (no think block) -> ``<|channel>thought\\n<channel|>Y`` — an
      **empty closed channel**, which is gemma-4's own representation of "answer without
      thinking" (it is exactly the chat template's ``enable_thinking=False`` generation
      stub).

    This is the inverse of the inference-side normalization in
    ``server._clean_response``. Gemma-4 *generates* its reasoning as the special tokens
    ``<|channel>`` (id 100) / ``<channel|>`` (id 101) — verified by sampling the base
    model — but the chat template's ``strip_thinking`` deletes that form from message
    content, and stored targets carry the normalized ``<think>`` *text*. Feeding the
    literal form would train ordinary text tokens (``<think>`` -> ``[236820, 36345,
    236813]``) instead of the channel the model actually emits.

    The empty-channel scaffold for answer-only targets is deliberate: the prompt is built
    with ``enable_thinking=True`` (the ``<|think|>`` system marker inference always
    injects), so rendering a **bare** model turn would train "thinking enabled, yet open
    no channel and answer directly" — a gradient that erodes the reasoning channel over
    cycles. The empty closed channel is exactly Gemma-4's documented "thinking on, but no
    thought" output for the 12B/26B/31B models, so it keeps the mechanism alive without
    teaching the model to skip the channel. CoT-carrying targets (``keep``/``original``
    replies and branch wins, whose faithful ``<think>`` is reattached upstream in
    ``resolve_revision_target``) take the real-thought branch above; only answer-only
    targets — chiefly IDEAL revisions with no faithful CoT — hit this empty scaffold.
    """
    m = _THINK_BLOCK_RE.match(content or "")
    if not m:
        answer = _strip_trailing_fields((content or "").lstrip("\n"))
        return f"<|channel>thought\n<channel|>{_assert_no_think(answer)}"
    thought = m.group(1).strip()
    answer = _strip_trailing_fields(m.group(2).lstrip("\n"))
    return f"<|channel>thought\n{thought}\n<channel|>{_assert_no_think(answer)}"


def _assert_no_think(answer: str) -> str:
    """Guard: the answer span (after the channel close) must hold no literal think tags.

    A nested ``<think>`` here is a double-think target (the source turn already carried a
    CoT and got wrapped in another). On gemma only the first block becomes the native
    channel, so a residual ``<think></think>`` would train as plain-text tokens inside the
    trained span and teach the model to emit/loop empty think blocks. Producers strip the
    embedded block (see reflection_shareml._answer_only); this catches any that slip
    through before they reach the optimizer.
    """
    if "<think>" in answer or "</think>" in answer:
        raise ValueError(
            "render: assistant answer span carries a literal <think> block after the "
            "reasoning channel — nested/double think would train empty-think loops. "
            f"Offending answer head: {answer[:200]!r}"
        )
    return answer


def split_think(content: str) -> tuple[str, str]:
    """``<think>X</think>Y`` -> ``(X, Y)``; answer-only ``Y`` -> ``("", Y)``."""
    m = _THINK_BLOCK_RE.match(content or "")
    if not m:
        return "", _strip_trailing_fields((content or "").lstrip("\n"))
    return m.group(1).strip(), _strip_trailing_fields(m.group(2).lstrip("\n"))


def to_harmony_messages(messages: list) -> list:
    """gpt-oss (harmony): the chat template renders an assistant turn's reasoning from a
    ``thinking`` field — as ``<|channel|>analysis<|message|>X<|end|>`` followed by the
    ``final`` channel — but ONLY on the last turn; earlier assistant turns render as
    ``final`` alone, which is exactly the CoT-stripped history inference builds. So the
    target's ``<think>`` block moves into that field and the answer stays in ``content``.
    ``thinking`` is always set (possibly empty): an answer-only target then renders an
    empty analysis channel + the final channel, the harmony analogue of gemma's empty
    closed channel — the model is prompted with "Channel must be included for every
    message", so a bare channel-less turn would train against its own system prompt.
    """
    out = [dict(m) for m in messages]
    last = out[-1]
    if last.get("role") == "assistant":
        thought, answer = split_think(last.get("content", ""))
        last["content"] = _assert_no_think(answer)
        last["thinking"] = thought
    return out


def render_example_text(messages: list, tokenizer, family: str) -> str:
    """Render one SFT example to text, parity-faithful to how inference prompts + how
    the model emits reasoning.

    chatml: a literal ``<think>`` block *is* the model's native form, so render the
        whole conversation straight through the template (and, like inference for these
        models, without ``enable_thinking``).
    harmony (gpt-oss): the template itself renders the reasoning channel from a
        ``thinking`` field on the last assistant turn (see ``to_harmony_messages``), with
        the same ``reasoning_effort`` inference passes; ``train_on_responses_only`` keys on
        ``<|start|>assistant<|channel|>``.
    gemma:  the model emits a special-token reasoning channel the template strips, so
        build the prompt through the template with ``enable_thinking=True`` (matching
        inference, which injects the ``<|think|>`` system marker on every call) and
        append the assistant turn in native channel form by hand. ``train_on_responses_only``
        still keys on the ``<|turn>model\\n`` response marker the prompt ends with, so the
        channel tokens fall inside the trained (unmasked) span.
    """
    if family == "harmony":
        # Same template kwargs inference uses (reasoning_effort), so the system header
        # ("Reasoning: high") matches what the model is prompted with at chat time.
        from core.model_family import GPTOSS
        return tokenizer.apply_chat_template(
            to_harmony_messages(messages), tokenize=False, add_generation_prompt=False,
            **GPTOSS.template_kwargs)
    if family != "gemma":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    target = to_gemma_thinking_channel(messages[-1].get("content", ""))
    return f"{prompt}{target}<turn|>\n"


# --------------------------------------------------------------------------- #
# Parity guard (used by selftest.py)
# --------------------------------------------------------------------------- #

def _reference_inference_conversation(system_content: str, conversation: list) -> list:
    """Copy of server._build_inference_conversation — the parity reference.

    Kept in lock-step with server.py:_build_inference_conversation. If that changes,
    update this and the self-test will confirm build_messages still matches.
    """
    out = [{"role": "system", "content": system_content}]
    for turn in conversation:
        if turn.get("role") == "user":
            content = _user_content(turn.get("content", ""), turn.get("speaker", ""))
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": turn["content"]})
    return out


def assert_parity(anchor: dict, target_text: str) -> None:
    """Raise unless build_messages == the live inference assembly for this anchor."""
    conversation = list(anchor.get("context", []) or [])
    conversation.append({"role": "user",
                         "content": anchor.get("prompt", ""),
                         "speaker": anchor.get("speaker", "")})
    expected = _reference_inference_conversation(anchor.get("system_prompt", "") or "", conversation)
    expected.append({"role": "assistant", "content": target_text})
    actual = build_messages(anchor, target_text)
    if actual != expected:
        raise AssertionError(
            "render/inference parity broken:\n"
            f"  expected={expected}\n  actual  ={actual}"
        )
