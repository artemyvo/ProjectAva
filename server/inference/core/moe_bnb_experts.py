"""Per-expert ``nn.Linear`` experts for bitsandbytes 4-bit MoE checkpoints (Qwen3.5 MoE).

Why this exists
---------------
transformers 5 stores a MoE layer's experts as two fused 3D ``nn.Parameter``
tensors (``experts.gate_up_proj`` ``[E, 2I, H]`` and ``experts.down_proj``
``[E, H, I]``). bitsandbytes' integration converts ``nn.Linear`` modules only
(``replace_with_bnb_linear``), so under ``load_in_4bit=True`` the experts —
232 GB of Qwen3.5-122B-A10B's 250 GB — stay bf16. On a 121 GB unified-memory
box that load cannot complete: it was observed crawling at ~1 tensor/s while
the kernel thrashed the page cache and then taking the box down with it.

unsloth solves the same problem for gpt-oss by shipping a checkpoint whose
experts are ``ModuleList``s of ``nn.Linear`` (``gate_up_projs.<i>`` /
``down_projs.<i>``) and swapping the transformers experts class for one with
that layout before the load (``unsloth_zoo.temporary_patches.gpt_oss``). Every
Linear then goes through transformers' ordinary bnb path: pre-quantized
``Linear4bit`` weights deserialize natively, LoRA attaches via ordinary suffix
matching (``unsloth.models._utils.get_moe_target_modules`` already recognises
"a ModuleList under ``experts`` holding only Linear leaves"), and nothing in the
loader needs to know about MoE. This module is that pattern for
``qwen3_5_moe``; ``server/convert_moe_bnb4bit.py`` writes the matching checkpoint.

Contract
--------
- ``Qwen3_5MoeExpertsBnb4bit`` has the SAME forward signature as the class it
  replaces (``(hidden_states, top_k_index, top_k_weights)``), so both the stock
  ``Qwen3_5MoeSparseMoeBlock.forward`` and unsloth's patched one call it unchanged.
- ``gate_up_proj``/``down_proj`` exist as EMPTY non-persistent buffers so
  ``_init_weights`` (which ``isinstance``-dispatches on the class name and calls
  ``init.normal_`` on both) is a no-op rather than an AttributeError.
- ``install()`` swaps the class into ``transformers.models.qwen3_5_moe`` and is
  idempotent; ``restore()`` puts the stock class back. The swap is only correct
  for a checkpoint in the per-expert layout — ``is_perexpert_bnb_checkpoint``
  is the gate, keyed on the marker the converter writes into ``config.json``.

GPU-free self-test: ``python -m core.moe_bnb_experts``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Marker the converter writes into config.json; the loader swaps classes on it.
PEREXPERT_MARKER = "unsloth_perexpert_bnb4bit_experts"
GATE_UP_LIST = "gate_up_projs"
DOWN_LIST = "down_projs"


class Qwen3_5MoeExpertsBnb4bit(nn.Module):
    """Qwen3.5-MoE experts as ``ModuleList``s of ``nn.Linear`` (bnb-quantizable)."""

    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN
        self.num_experts = int(config.num_experts)
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.gate_up_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, 2 * self.intermediate_dim, bias=False, dtype=dtype)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=dtype)
            for _ in range(self.num_experts)
        ])
        # Empty stand-ins for the fused parameters so transformers' _init_weights
        # (init.normal_ on both) is a harmless no-op. Non-persistent: never saved.
        self.register_buffer("gate_up_proj", torch.empty(0, dtype=dtype), persistent=False)
        self.register_buffer("down_proj", torch.empty(0, dtype=dtype), persistent=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Identical math to the stock fused forward, per-expert Linear instead of
        an indexed 3D slice. Tokens are grouped per hit expert; only experts with
        routed tokens run.

        The routing loop is adapted from transformers' ``Qwen3_5MoeExperts.forward``
        (Copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache License
        2.0, https://www.apache.org/licenses/LICENSE-2.0). Changed here: the per-expert
        ``nn.Linear`` modules replace ``F.linear`` over the fused 3D tensors, and hit
        experts are found by ``bincount``."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            flat = top_k_index.reshape(-1)
            counts = torch.bincount(flat, minlength=self.num_experts)
            hit = torch.nonzero(counts, as_tuple=False).flatten().tolist()
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = self.gate_up_projs[expert_idx](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_projs[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# ── install / gate ───────────────────────────────────────────────────────────

_ORIGINAL_ATTR = "_ava_original_Qwen3_5MoeExperts"


def _modeling():
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m
    return m


_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
_LANGUAGE_PREFIX_RENAME = "^model.language_model"


def patch_conversion_mapping() -> int:
    """Drop the ``^model.language_model -> model`` WeightRenaming from the text
    sub-model's conversion list. Returns how many entries were removed.

    transformers registers it for ``qwen3_5_moe_text`` (loading a multimodal checkpoint
    into the text-only class), and ``get_model_conversion_mapping`` recurses into the
    ``language_model`` submodule and adds the text type's list to the multimodal
    model's. Plain keys survive that (the loader falls back to the unrenamed name),
    but a bitsandbytes weight group — ``weight`` plus its five quant-state tensors,
    fused by the quantizer's own WeightConverter — is renamed as a group to a name
    the multimodal model does not have, reported UNEXPECTED, and the packed ``weight``
    lands without its quant state (observed 2026-09-07: every Linear4bit loaded raw and
    the first GDN projection failed on ``F.linear`` with a ``[1, 18874368]`` weight).
    The rename is meaningless for the ConditionalGeneration class, whose parameters
    ARE ``model.language_model.*``, so removing it there is exact, not a workaround
    for a different bug."""
    from transformers import conversion_mapping as cm
    mapping = cm.get_checkpoint_conversion_mapping(_TEXT_MODEL_TYPE)
    if not mapping:
        return 0
    kept, removed = [], 0
    for entry in mapping:
        pats = getattr(entry, "source_patterns", None) or []
        if isinstance(pats, str):
            pats = [pats]
        if isinstance(entry, cm.WeightRenaming) and any(_LANGUAGE_PREFIX_RENAME in str(p) for p in pats):
            removed += 1
            continue
        kept.append(entry)
    if removed:
        cm.register_checkpoint_conversion_mapping(_TEXT_MODEL_TYPE, kept, overwrite=True)
    return removed


def install() -> bool:
    """Swap the per-expert class into transformers' qwen3_5_moe module. Idempotent."""
    patch_conversion_mapping()
    m = _modeling()
    if getattr(m, "Qwen3_5MoeExperts", None) is Qwen3_5MoeExpertsBnb4bit:
        return True
    if not hasattr(m, _ORIGINAL_ATTR):
        setattr(m, _ORIGINAL_ATTR, m.Qwen3_5MoeExperts)
    # Keep the stock symbol names: unsloth's compiler copies class sources by name.
    Qwen3_5MoeExpertsBnb4bit.__name__ = "Qwen3_5MoeExperts"
    Qwen3_5MoeExpertsBnb4bit.__qualname__ = "Qwen3_5MoeExperts"
    m.Qwen3_5MoeExperts = Qwen3_5MoeExpertsBnb4bit
    os.environ["AVA_QWEN35_MOE_BNB4BIT"] = "1"
    return True


def restore() -> None:
    m = _modeling()
    orig = getattr(m, _ORIGINAL_ATTR, None)
    if orig is not None:
        m.Qwen3_5MoeExperts = orig
    os.environ.pop("AVA_QWEN35_MOE_BNB4BIT", None)


def _read_config(model_id: str) -> Optional[dict]:
    """The checkpoint's config.json — a local dir, or the HF cache snapshot."""
    try:
        if os.path.isdir(model_id):
            p = os.path.join(model_id, "config.json")
            return json.load(open(p)) if os.path.exists(p) else None
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_id, "config.json", local_files_only=True)
        return json.load(open(p))
    except Exception:
        return None


def is_perexpert_bnb_checkpoint(model_id: str) -> bool:
    cfg = _read_config(model_id) or {}
    return bool(cfg.get(PEREXPERT_MARKER)) and cfg.get("model_type") in ("qwen3_5_moe",)


def install_if_needed(model_id: str) -> bool:
    """Called by the backend before ``from_pretrained``: swap for a converted
    checkpoint, restore the stock class for anything else (the swap is global)."""
    if is_perexpert_bnb_checkpoint(model_id):
        return install()
    restore()
    return False


# ── GPU-free self-test ───────────────────────────────────────────────────────

def _selftest() -> None:
    import types
    torch.manual_seed(0)
    E, H, I, T, K = 6, 32, 16, 9, 2
    cfg = types.SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                                hidden_act="silu", dtype=torch.float32, _experts_implementation=None)
    m = _modeling()
    ref = m.Qwen3_5MoeExperts.__dict__ and getattr(m, _ORIGINAL_ATTR, m.Qwen3_5MoeExperts)(cfg)
    ours = Qwen3_5MoeExpertsBnb4bit(cfg)
    with torch.no_grad():
        nn.init.normal_(ref.gate_up_proj, std=0.2); nn.init.normal_(ref.down_proj, std=0.2)
        for e in range(E):
            ours.gate_up_projs[e].weight.copy_(ref.gate_up_proj[e])
            ours.down_projs[e].weight.copy_(ref.down_proj[e])
        x = torch.randn(T, H)
        idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        w = torch.softmax(torch.randn(T, K), dim=-1)
        a = ref(x, idx, w); b = ours(x, idx, w)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    # state-dict layout: per-expert Linear keys, no fused keys
    keys = set(ours.state_dict().keys())
    assert f"{GATE_UP_LIST}.0.weight" in keys and f"{DOWN_LIST}.{E-1}.weight" in keys
    assert "gate_up_proj" not in keys and "down_proj" not in keys
    # install / restore round-trip
    stock = m.Qwen3_5MoeExperts
    install(); assert m.Qwen3_5MoeExperts is Qwen3_5MoeExpertsBnb4bit
    assert m.Qwen3_5MoeExperts.__name__ == "Qwen3_5MoeExperts"
    restore(); assert m.Qwen3_5MoeExperts is stock
    assert not is_perexpert_bnb_checkpoint("/nonexistent")
    # conversion-mapping patch: the language-prefix rename is gone, the experts
    # merge converters (which cannot match a per-expert layout anyway) remain
    from transformers import conversion_mapping as cm
    left = cm.get_checkpoint_conversion_mapping(_TEXT_MODEL_TYPE) or []
    assert not any(isinstance(e, cm.WeightRenaming) and _LANGUAGE_PREFIX_RENAME in str(getattr(e, "source_patterns", "")) for e in left)
    assert any(isinstance(e, cm.WeightConverter) for e in left), "experts merge converters should remain"
    assert patch_conversion_mapping() == 0  # idempotent
    print("moe_bnb_experts selftest OK")


if __name__ == "__main__":
    _selftest()
