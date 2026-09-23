"""Keep grouped-query SDPA off the math backend when flash cannot take it.

transformers' ``sdpa_attention_forward`` passes ``enable_gqa=True`` (un-repeated
K/V) whenever the layer's mask is ``None``, i.e. every causal full-attention layer
at batch 1. Only the FLASH kernel accepts ``enable_gqa`` with Hq != Hkv; flash
caps head_dim at 256. Gemma-4's global layers are head_dim 512 with 4 KV heads
under 32 query heads, so SDPA drops to MATH and materializes the fp32 score
matrix: 32 x L^2 x 4 bytes, 4.87 GiB at L=6388. Measured on the RTX 5090 box
(torch 2.12.1, transformers 5.5.0), fwd+bwd at L=2048, D=512, 32q/4kv heads:

    enable_gqa=True       2560 MiB peak  (math)
    K/V repeated to 32     625 MiB peak  (mem-efficient)

That is the step-2 OOM of train runs 20260923_003453 and 20260923_090623 (row
index 1 is 6388 tokens; unsloth's OOM fallback then segfaulted, rc=-11).

``install()`` wraps ``use_gqa_in_sdpa`` so it answers False when the key's
head_dim exceeds flash's limit — transformers then repeats K/V itself and the
mem-efficient kernel runs. head_dim <= 256 layers keep enable_gqa (flash takes
it). Idempotent; a transformers that no longer has the hook is reported, not
fatal.
"""

from __future__ import annotations

_FLASH_MAX_HEAD_DIM = 256
_PATCHED_ATTR = "_ava_sdpa_gqa_original"


def _keep_gqa(original, attention_mask, key) -> bool:
    if key is not None and key.shape[-1] > _FLASH_MAX_HEAD_DIM:
        return False
    return original(attention_mask, key)


def install() -> bool:
    """Patch ``transformers.integrations.sdpa_attention.use_gqa_in_sdpa``.
    Returns True when active."""
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
    except Exception:
        return False
    if getattr(sdpa_mod, _PATCHED_ATTR, None) is not None:
        return True
    original = getattr(sdpa_mod, "use_gqa_in_sdpa", None)
    if original is None:
        print("!!! SDPA GQA PATCH NOT ACTIVE: transformers.integrations.sdpa_attention."
              "use_gqa_in_sdpa is gone (transformers upgrade?). head_dim>256 GQA layers "
              "(Gemma-4 global) may run SDPA's math backend: O(L^2) fp32 scores. See "
              "core/sdpa_gqa.py.", flush=True)
        return False

    def use_gqa_in_sdpa(attention_mask, key):
        return _keep_gqa(original, attention_mask, key)

    sdpa_mod.use_gqa_in_sdpa = use_gqa_in_sdpa
    setattr(sdpa_mod, _PATCHED_ATTR, original)
    return True


def _selftest() -> None:
    class _K:
        def __init__(self, d):
            self.shape = (1, 4, 8, d)

    always = lambda mask, key: True  # noqa: E731
    assert _keep_gqa(always, None, _K(512)) is False
    assert _keep_gqa(always, None, _K(256)) is True
    assert _keep_gqa(lambda m, k: False, None, _K(128)) is False
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
    except Exception:
        print("sdpa_gqa selftest: gate OK (transformers absent, install not exercised)")
        return
    assert install() and install()
    assert getattr(sdpa_mod, _PATCHED_ATTR) is not None
    print("sdpa_gqa selftest: OK")


if __name__ == "__main__":
    _selftest()
