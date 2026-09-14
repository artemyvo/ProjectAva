"""The bench's in-process model harness (ASSOCIATIVE_MEMORY.md §11: the witness / select
model is loaded by the HARNESS, never by the library). Loads gemma-4-31B once per process
through unsloth with Ava's Spark load fixes (`core.unified_memory` placement +
`core.fast_load` mmap clone), and exposes a `generate_fn` in the library's seam shape:

    generate_fn(system, user, *, thinking, max_new_tokens, temperature) -> (text, info)

`info["truncated"]` is set when the generation hit its cap. Gemma's channel markers are
normalized to `<think>…</think>` so the library's answer-region rule holds.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "server" / "inference"
MODEL_ID = os.environ.get("ASSOC_WITNESS_MODEL", "unsloth/gemma-4-31B-it")
MAX_SEQ = int(os.environ.get("ASSOC_MODEL_CTX", "16384"))

_STATE: dict = {}


def load():
    if _STATE.get("model") is not None:
        return _STATE["model"], _STATE["tokenizer"]
    if str(INFERENCE) not in sys.path:
        sys.path.insert(0, str(INFERENCE))
    import torch
    from core import unified_memory, fast_load          # Ava's Spark load fixes
    from unsloth import FastModel
    placement = unified_memory.load_kwargs()
    fast = fast_load.install()
    t0 = time.time()
    with fast_load.hf_offline_if_cached(MODEL_ID):
        model, tokenizer = FastModel.from_pretrained(model_name=MODEL_ID, max_seq_length=MAX_SEQ, load_in_4bit=True, **placement)
    FastModel.for_inference(model)
    model.eval()
    print(f"[harness] loaded {MODEL_ID} in {time.time() - t0:.1f}s (fast_load={fast}, placement={placement or 'default'})", flush=True)
    _STATE.update({"model": model, "tokenizer": tokenizer, "torch": torch})
    return model, tokenizer


def _normalize(raw: str) -> str:
    try:
        from core.model_family import family_for
        return family_for(MODEL_ID).normalize_cot(raw)
    except Exception:
        return raw


def generate_fn(system: str, user: str, *, thinking: bool = False, max_new_tokens: int = 2048,
                temperature: float = 0.0):
    model, tokenizer = load()
    torch = _STATE["torch"]
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if thinking and "gemma-4" in MODEL_ID.lower() and not prompt.rstrip().endswith("thought"):
        prompt = prompt + "<|channel>thought\n"
    # A multimodal checkpoint (gemma-4) hands back a processor; the text tokenizer under it
    # is what takes a plain string (Ava's `inference_backend` does the same).
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tok(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]
    room = MAX_SEQ - n_in - 8
    cap = max(16, min(max_new_tokens, room))
    kw = dict(max_new_tokens=cap, do_sample=temperature > 0,
              pad_token_id=getattr(text_tok, "pad_token_id", None) or text_tok.eos_token_id)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.95)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    gen = out[0, n_in:]
    text = text_tok.decode(gen, skip_special_tokens=False)
    n_gen = int(gen.shape[0])
    truncated = n_gen >= cap
    for eos in (text_tok.eos_token or "", "<end_of_turn>", "<turn|>", "<|im_end|>", "<eos>"):
        if eos and text.rstrip().endswith(eos):
            text = text.rstrip()[: -len(eos)]
    text = _normalize(("<|channel>thought\n" if thinking and "gemma-4" in MODEL_ID.lower() else "") + text)
    info = {"truncated": truncated, "input_tokens": n_in, "output_tokens": n_gen, "seconds": round(time.time() - t0, 1),
            "tok_s": round(n_gen / max(time.time() - t0, 1e-6), 1), "model_id": MODEL_ID}
    return text, info


def token_counter(text: str) -> int:
    _, tokenizer = load()
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    return len(text_tok(text, add_special_tokens=False)["input_ids"])
