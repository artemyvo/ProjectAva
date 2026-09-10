"""Shared LLM/tokenizer helpers used across training and chat tabs."""

from __future__ import annotations

from typing import Callable, Optional


CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

PLAIN_ROLE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)

def get_valid_special_token_id(tokenizer, token: str) -> Optional[int]:
    """Return a token id when known and not unk."""
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return None
    return token_id


def build_fallback_chat_prompt(conversation: list) -> str:
    """Build a plain role-text fallback prompt when chat_template is unavailable."""
    prompt = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)
    return prompt + "\nASSISTANT:"


def build_inference_prompt(
    tokenizer,
    conversation: list,
    reasoning_effort: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """Build inference prompt from conversation using the tokenizer's chat template."""
    # For processors (e.g. Gemma4Processor), chat_template lives on the underlying
    # tokenizer; check both the object itself and processor.tokenizer.
    chat_template = getattr(tokenizer, "chat_template", None) or getattr(
        getattr(tokenizer, "tokenizer", None), "chat_template", None
    )
    if hasattr(tokenizer, "apply_chat_template") and chat_template:
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if enable_thinking:
            kwargs["enable_thinking"] = True
        return tokenizer.apply_chat_template(conversation, **kwargs)

    return build_fallback_chat_prompt(conversation)


def ensure_chat_template(
    tokenizer,
    *,
    model_name: Optional[str] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> None:
    """Ensure tokenizer.chat_template exists, with model/template fallbacks."""
    existing_template = getattr(tokenizer, "chat_template", None)
    if isinstance(existing_template, str) and existing_template.strip():
        return

    def _emit(message: str) -> None:
        if emit is not None:
            emit(message)

    chat_template_set = False
    try:
        if model_name:
            from transformers import AutoTokenizer

            _emit(f"Loading chat template for {model_name}...")
            temp_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            candidate_template = getattr(temp_tokenizer, "chat_template", None)
            if isinstance(candidate_template, str) and candidate_template.strip():
                tokenizer.chat_template = candidate_template
                chat_template_set = True
                _emit("✓ Chat template loaded from model")
    except Exception as exc:
        _emit(f"Could not load chat template from HF model: {exc}")

    if chat_template_set:
        return

    has_chatml_tokens = (
        get_valid_special_token_id(tokenizer, "<|im_start|>") is not None
        and get_valid_special_token_id(tokenizer, "<|im_end|>") is not None
    )

    if has_chatml_tokens:
        tokenizer.chat_template = CHATML_TEMPLATE
        _emit("Using fallback template: ChatML format.")
        return

    tokenizer.chat_template = PLAIN_ROLE_TEMPLATE
    _emit("Using fallback template: plain role-text format.")
