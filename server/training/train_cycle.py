"""One from-scratch build: assemble corpus → train a fresh LoRA → probe → record → clean.

REBUILD (see repo-root ``REBUILD.md``). The adapter is a **build artifact** — a pure
function of (frozen base, bundle corpus, config), recompiled from scratch every build and
never resumed. Repetition is replaced by an age-keyed LR ramp. Run offline (not while the
inference server holds the GPU), e.g. as a Sleep-phase job:

    cd server && .venv/bin/python -m training.train_cycle            # full build
    cd server && .venv/bin/python -m training.train_cycle --dry-run  # assemble only, no train

Steps:
  1. Assemble the whole FROZEN-bundle corpus (``build_dataset``): every reflect-once chat
     (hot + archive) contributes ONE row per revisable exchange — no variant copies, no
     regularizer — chronological oldest-first, each stamped with its age (promoted builds
     since it was reflected — ``builds.jsonl``) and the resulting LR multiplier. Wander is
     one-shot at a fixed multiplier.
  2. Render/tokenize to the training cap. Optional history is trimmed; an irreducible row
     that cannot retain its complete response + EOT is excluded and recorded in
     ``scratch/sft_quarantine.jsonl``. The exact retained rows replace the disposable
     ``scratch/sft_render.jsonl`` and carry stable ids through response masking.
  3. Load the **frozen base** by ``model_id`` and fit a **fresh** LoRA (``get_peft_model``,
     never resumed), then train with a per-optimizer-step LR = ``base_lr × lr_multiplier(
     age(row)) × schedule_shape(step)`` (SequentialSampler + a step-keyed LambdaLR; batch=1,
     GA=1, so step i == row i%N). ``train_lr_schedule`` (server_config.json) selects the
     global shape: ``"age_ramp"`` (**default**) = the flat single pass (shape ≡ 1), so the
     age ramp alone weights the rows and one epoch is the whole cycle; ``"triangular"`` = a
     ``0→max→…→max→0`` trapezoid (1 warmup + ``train_plateau_epochs`` hold + 1 decay epoch)
     layered on the multipliers so every row sees the same average LR regardless of corpus
     position (forces epochs = ``train_plateau_epochs`` + 2, default plateau 3).
     ``base_lr`` comes from server_config.json ``train_lr`` (default 8e-6) unless a caller
     overrides it. LoRA rank is ``lora_r`` (default 32). ``model_id`` is never modified.
  4. Run the offline **regression probe** (five tiers: capability floor, format/coherence,
     character continuity, acute retention, temp≈1.0 sampling stability). On failure nothing
     is saved, the config is untouched, and no bundle's age advances (only a *promoted*
     build advances ages). On pass, save the adapter, repoint ``server_config.json``'s
     ``adapter_id`` at it, append a ``promoted`` line to ``builds.jsonl`` (which advances
     every bundle's age → raises its LR next build), then delete the render.
     ``--skip-validation`` bypasses the probe (and its baselines) and promotes
     unconditionally — the default for UI-initiated training (Sleep tab). Tier 3 is inert
     under from-scratch builds (pre-train model is the bare base, ``had_prior_adapter=False``).

Scope / honesty:
  * Dialogue exchanges train verbatim (the sidecar's single resolved target). **Persona**
    statements ride their host exchange's CoT as leading lines, and **facts** ride theirs as
    ``"I know that …"`` lines (model-free injection — ``build_dataset``, sourced per host
    bundle). The CoT-safe injection replaced the old *regeneration* path that eroded the CoT
    channel on gemma-4 (memory ``fact-persona-cot-erosion``). RAG recall of persona/fact and
    of consolidated chats fades on the same age via the Phase-3 crossfade (``rag_engine``);
    there is no stage advancement, ``FACT_TRAIN_CAP``, hot→archive move, or IDEAL regularizer
    any more (all retired by REBUILD).
  * This module needs a GPU, a loaded model, and Unsloth + TRL. First end-to-end run on
    Gemma4-31B: 2026-07-05 (memory ``training-oom-gemma31b`` for the seq-cap tuning). Treat
    thresholds and LoRA hyperparameters as first-guess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Reduce CUDA caching-allocator fragmentation: variable-length regeneration plus
# LoRA training otherwise grow the reserved pool indefinitely (allocated stays flat
# while free VRAM bleeds out → eventual OOM). Must be set before torch initializes
# CUDA, which importing unsloth below does — hence ahead of that import.
#
# BOTH names are set (identical value, so they can never conflict), and the LEGACY
# PYTORCH_CUDA_ALLOC_CONF is the load-bearing one: torch 2.9.1 was VERIFIED to ignore
# the new PYTORCH_ALLOC_CONF name (training box, 2026-08-26 — env set before import,
# `is_expandable` False on the memory snapshot), despite that name being billed here
# as "the torch ≥ 2.9 name". So the 2026-06-13 switch to the new name alone (9c80453)
# turned this guard OFF invisibly, until two 8192-cap builds died mid-run of
# split-segment stranding (6.4–6.9 GiB reserved-but-unallocated against ~23 GiB
# genuinely allocated — free blocks trapped in partly-used segments, which
# empty_cache() structurally cannot release). The new name is kept for whatever torch
# eventually honors it. Never TRUST either name: `_ensure_expandable_segments` probes
# `is_expandable` at cycle start and forces the mode via the runtime API when inert.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Stop unsloth from wrapping its generated model forwards (e.g.
# unsloth_compiled_cache/unsloth_compiled_module_gemma4.py) in torch.compile. Those
# @torch.compile decorators SNAPSHOT the TorchDynamo config at decoration time (when
# unsloth imports/generates that module during from_pretrained), so no runtime tweak to
# torch._dynamo.config ever reaches the config the wrapper uses — the recompile limit
# stays at the default 8, and the compiled RMSNorm/attention hard-fails with
# FailOnRecompileLimitHit (fullgraph=True) once enough distinct input shapes go through
# it during training. Disabling unsloth's compile makes those forwards plain eager Python
# (correct; unsloth's speed is in its Triton kernels/patches, not this wrapper). Must be
# set before `import unsloth` below, hence here; setdefault so an operator can override.
# This mirrors the same fix in inference/core/inference_backend.py (which this offline
# process does NOT import, so the fix must be repeated here). See that module + the stale
# unsloth_compiled_cache/ purge in _run_cycle for the full rationale.
#
# "partial", NOT "1": with "1" unsloth_zoo's auto-compiler quick-exits before INSTALLING
# any of its patched classes (compiler.py `full_disable`) — including the fused lm-head /
# no-logits cross-entropy patch — so the vanilla transformers forward materializes the
# full seq × 262k-vocab logit buffer in fp32 every step, which is what forced
# train_max_seq_length down to 2048 (and the answer-chopping/quarantine fallout). Under
# "partial" every generated @torch.compile decorator is emitted as
# @torch.compiler.disable (no dynamo, so the recompile-limit crash stays impossible) but
# the source patches ARE installed, so the loss runs through cut_cross_entropy's Triton
# kernel on hidden states and the logit buffer never exists. Training-side only; the
# inference server keeps "1" (it gains nothing from the loss patch).
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "partial")

# Per-chunk memory budget for the fused CE loss (unsloth_zoo
# fused_losses/cross_entropy_loss.py `UNSLOTH_CE_LOSS_TARGET_GB`; read since at least
# 2026.5.4, which is what this box runs — the knob is not 2026.7.2+). Without it the
# chunk sizer targets min(free_VRAM/2, 4 GB) — but a 4096-token × 262k-vocab fp32
# logit chunk is 3.31 GiB, which "fits" that target as ONE chunk, and the backward
# then needs the same again: ~6.6 GiB transient against the ~5 GiB the 31B 4-bit
# base + optimizer leave free on a 32 GB box (observed OOM at step 1, 2026-07-15).
# Cost is a few extra sequential GEMMs per step — noise next to the 31B forward.
#
# 2026-07-31: the value is NOT the peak — read `get_chunk_size`, which computes
#   n_splits = max(round(qlen * multiplier) * 4, 1)
# The round() floors to 0 for every qlen below the multiplier's 0.5 threshold, and the
# max(..., 1) then hands back ONE UNCAPPED chunk covering the whole sequence. So the
# target is honoured only above that threshold; just under it the chunk is twice the
# target, and the worst case over all qlen is exactly 2 x TARGET_GB.
#
# The old 1.5 assumed every row sits at the 4096 cap (where it does give 4 chunks x
# 1.0 GiB). Most rows do not: answer-preserving truncation leaves them 1.5k-3k tokens,
# which is precisely the round-to-zero band, so they ran as a single chunk of up to
# 3.0 GiB and OOM'd in backward against ~0.9 GiB free (run 20260731_112845, step 4/39,
# a ~3k-token row). 0.5 was the retreat from that: worst case 1.0 GiB.
#
# 2026-08-03: raised 0.5 -> 1.0 against measured headroom — Gemma4-31B training was
# observed peaking at ~28 of 32 GB, so the ~1.0 GiB the worst-case chunk grows by (2.0
# vs 1.0 GiB, plus the matching backward transient) is affordable and buys fewer
# sequential GEMMs per step. This sits BETWEEN the value that OOM'd (1.5, worst case
# 3.0 GiB) and the one known safe (0.5), so it is a deliberate bet on that headroom and
# not a validated ceiling: if a build OOMs in backward on a mid-length row, go back to
# 0.5 rather than reaching for train_max_seq_length (see the corollary).
#
# 2026-08-26: reverted 1.0 -> 0.5 (the known-safe value, worst case 1.0 GiB) — the
# headroom bet above did not hold; taking the escape hatch the 2026-08-03 note names.
#
# Corollary — do NOT "fix" an OOM here by lowering train_max_seq_length alone: shrinking
# the cap moves rows INTO the round-to-zero band and can raise the peak (at target 1.5 a
# 2048 cap is a single 2.0 GiB chunk, worse than the 4096 cap's 1.0 GiB).
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "0.5")

# Import unsloth before transformers/peft so its optimizations and patches land —
# Unsloth warns and runs slower / risks OOM if it loads after them, and the core
# modules below pull transformers in transitively. unsloth itself is imported lazily
# at train time (below), so without this it always landed last. Guarded so the module
# still imports where unsloth is absent (train then fails later with a clear error).
try:
    import unsloth  # noqa: F401
except Exception:
    pass

# Make the inference backend importable (same trick as ledger.py).
_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

# Python 3.14 shim for datasets 4.3.0's dill Pickler (unsloth pins datasets<4.4.0,
# so we can't take the upstream 4.4.0 fix). Imported for its side effect, before any
# Dataset.from_dict fingerprinting runs below. See the module docstring.
from training import _datasets_py314_compat  # noqa: E402,F401

from core.chat_sidecar import ChatSidecar                      # noqa: E402
from core.wander_sft import load_pending as load_wander_pending  # noqa: E402
from training import render                                    # noqa: E402
from training.build_dataset import (  # noqa: E402
    build_dataset, corpus_fingerprint, row_render_dict)
from training.build_history import BuildHistory                # noqa: E402
from training.build_snapshot import write_snapshot             # noqa: E402
from training.decay import (  # noqa: E402
    ConsolidationConfig, TRAIN_LR_DEFAULT, TRAIN_PLATEAU_EPOCHS_DEFAULT,
    TRAIN_LORA_R_DEFAULT, TRAIN_LORA_ALPHA)
from training.label_policy import find_subsequences, row_label_policy  # noqa: E402
from training.ledger import ConsolidationLedger                # noqa: E402
from training.render import answer_of, has_cot                # noqa: E402
from training.train_progress import TrainProgress             # noqa: E402
from training.reflections_path import (                        # noqa: E402
    SFT_RENDER_FILE,
    SFT_QUARANTINE_FILE,
    activity_log_path,
    consolidation_dir,
    hot_chats_dir,
    load_server_config,
    persona_dir,
    save_server_config,
    scratch_dir,
    staging_chats_dir,
    staging_consolidation_dir,
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"   # server/models/

# Validation master switch (DISABLED — see training/validation_switch.py for the full
# rationale). Kept in that tiny, import-light module so the inference side (reflection_service)
# can honor the same flag without importing this unsloth-heavy module. When False, run_cycle
# force-skips the probe below and every build promotes unguarded.
from training.validation_switch import VALIDATION_ENABLED as _VALIDATION_ENABLED  # noqa: E402


def _purge_unsloth_compile_cache() -> None:
    """Delete stale unsloth_compiled_cache/ dirs (best-effort).

    Mirrors UnslothBackend._purge_unsloth_compile_cache. Once UNSLOTH_COMPILE_DISABLE is
    on, unsloth regenerates its model modules without the torch.compile wrappers — but a
    cache written by an earlier run still holds the compiled (crash-prone) version, and
    unsloth may reuse it. The cache is fully regenerable, so wiping it on load forces a
    clean, wrapper-free regen. Checks the CWD (server/, where unsloth writes it) and the
    inference package root."""
    for base in (os.getcwd(), str(_INFERENCE)):
        cache = os.path.join(base, "unsloth_compiled_cache")
        if os.path.isdir(cache):
            try:
                shutil.rmtree(cache)
            except Exception:
                pass


def _fmt_duration(seconds: float) -> str:
    """Human-friendly elapsed time: "45s", "3m 07s", "1h 04m"."""
    s = int(round(max(0.0, float(seconds))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"

# Chat-template markers for response-only loss masking, per model family. Masking
# everything up to the response marker is how train_on_responses_only restricts
# loss to the assistant turn — the markers MUST match the model's chat template.
# gemma-4 (unsloth/gemma-4-*) uses <|turn>… markers, NOT the <start_of_turn>…
# tokens of gemma-2/3. The _model_family() helper only returns "gemma" for
# gemma-4, so this entry must match gemma-4's chat template exactly — otherwise
# train_on_responses_only finds no response marker, masks every token to -100,
# and the SFT trainer drops all rows (num_samples=0 at sampler init).
_MARKERS = {
    "gemma": ("<|turn>user\n", "<|turn>model\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    # gpt-oss harmony. The response marker is the one unsloth's own gpt-oss recipe uses;
    # it recurs inside a reasoning turn (analysis channel, then final channel), which is
    # fine — every occurrence opens a trained span, nothing closes one but the next
    # instruction marker.
    "harmony": ("<|start|>user<|message|>", "<|start|>assistant<|channel|>"),
}


def _keep_final_turn_only(labels):
    """Re-mask every unmasked label span except the final one (in place).

    ``train_on_responses_only`` keys on the response marker, which matches *every*
    assistant turn, so it unmasks all of them. But in multi-turn dialogue anchors the
    non-final assistant turns are deliberately rendered CoT-less (channel-stripped, to
    mirror how inference stores history) — training those turns teaches "open a model
    turn and answer with no reasoning channel", the gradient that erodes the gemma-4
    ``<|channel>`` CoT over cycles (see DESIGN.md "Render parity"). Only the final
    assistant turn carries the native reasoning channel, so it is the only turn that
    should contribute to the loss.

    Operates purely on the label tensor: contiguous runs of non-``-100`` labels are the
    unmasked assistant turns (user turns between them are masked, forming the gaps). Keep
    the last run, re-mask everything before it. Family-agnostic; a no-op for single-turn
    examples (one run already). Returns the modified tensor for convenience.
    """
    for i in range(labels.size(0)):
        row = labels[i]
        idx = (row != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        # Boundaries between contiguous runs: positions where the unmasked index jumps
        # by >1 (a masked user turn sits in the gap). The last run starts after the last
        # such jump; mask everything before it.
        gaps = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=True)[0]
        if gaps.numel() == 0:
            continue  # single run — final turn already the only trained span
        last_run_start = int(idx[int(gaps[-1]) + 1])
        row[:last_run_start] = -100
    return labels


def _unmask_user_from_features(features) -> Optional[list]:
    """Per-row ``unmask_user`` flags carried on the dataset (set by the render on an
    exchange's last decay copy — the blunt "don't mask the user's words on the final
    training run" limb). ``None`` when the column is absent, in which case the collator
    keeps its existing keep-final-turn-only behaviour untouched. Best-effort; never raises
    out of the collator."""
    try:
        if not features or not isinstance(features[0], dict):
            return None
        if "unmask_user" not in features[0]:
            return None
        return [bool(f.get("unmask_user")) for f in features]
    except Exception:
        return None


def _user_loss_weight_from_features(features) -> Optional[list]:
    """Per-row ``user_loss_weight`` carried on the dataset (a float on a folded contamination
    row, ``None`` elsewhere). Returns ``None`` when the column is absent (contamination_fold
    off, or the column stripped by SFT prep) — the collator then keeps the two-row/keep-final
    behaviour untouched. Best-effort; never raises out of the collator."""
    try:
        if not features or not isinstance(features[0], dict):
            return None
        if "user_loss_weight" not in features[0]:
            return None
        out = []
        for f in features:
            v = f.get("user_loss_weight")
            out.append(None if v is None else float(v))
        return out
    except Exception:
        return None


def _build_loss_weight_tensor(labels, input_ids, unmask_per_row, weight_per_row,
                              instruction_ids, response_ids):
    """A ``[B, T]`` float tensor of per-token loss weights for the folded-contamination rows,
    built from the RAW ``labels`` (before ``_apply_label_policy`` re-masks). A row with no
    ``user_loss_weight`` (``None``) falls back to a plain keep-final-turn weight (1.0 on the
    final assistant run, 0.0 elsewhere) so a mixed batch is coherent — though batch=1 means
    each step is a single row. Returns ``None`` if no row carries a weight."""
    import torch
    from training.label_policy import row_loss_weights
    if not weight_per_row or all(w is None for w in weight_per_row):
        return None
    rows = []
    for i in range(labels.size(0)):
        unmask = bool(unmask_per_row[i]) if unmask_per_row and i < len(unmask_per_row) else False
        w = weight_per_row[i] if i < len(weight_per_row) else None
        rows.append(row_loss_weights(
            labels[i].tolist(), input_ids[i].tolist(),
            unmask and w is not None, instruction_ids, response_ids,
            float(w) if w is not None else 0.0))
    return torch.tensor(rows, dtype=torch.float32, device=labels.device)


def _resolve_backbone_and_head(model):
    """Return ``(backbone, lm_head)`` for a (PEFT/unsloth-wrapped) causal LM, or
    ``(None, None)`` if the structure can't be resolved.

    The folded-contamination ``compute_loss`` needs the transformer backbone and the
    unembedding SEPARATELY so it can run ``lm_head`` on ONLY the unmasked positions (a few
    hundred), never materializing the full ``seq × vocab`` fp32 logit buffer — gemma-4's
    262k vocab makes that ~1 MB/token, the OOM the fused no-logits CE exists to avoid
    (memory ``training-oom-gemma31b``). Tries the unsloth/PEFT nesting
    (``model.base_model.model`` = the ``*ForCausalLM``), then simpler layouts, and descends
    into a multimodal ``.model.language_model`` text tower when present."""
    import torch.nn as nn

    def _try(causal):
        if causal is None:
            return None
        head = getattr(causal, "lm_head", None)
        bb = getattr(causal, "model", None)
        # Multimodal: the real decoder stack (has `.layers`) may be nested one deeper.
        if bb is not None and not hasattr(bb, "layers") \
                and getattr(bb, "language_model", None) is not None:
            bb = bb.language_model
        if isinstance(head, nn.Module) and isinstance(bb, nn.Module) and hasattr(bb, "layers"):
            return bb, head
        return None

    for causal in (
        getattr(getattr(model, "base_model", None), "model", None),  # PeftModel.base_model.model
        getattr(model, "base_model", None),
        model,
    ):
        got = _try(causal)
        if got:
            return got
    return None, None


def _apply_label_policy(labels, input_ids, unmask_per_row, instruction_ids, response_ids) -> None:
    """Tensor adapter over ``label_policy.row_label_policy``: keep-final-turn masking + the
    final-user-turn unmask, per row, in place. Only invoked when at least one row is flagged
    — the no-unmask path stays on ``_keep_final_turn_only`` so an unflagged cycle is
    byte-for-byte unchanged (the two agree when the flag is False; proven in ``label_policy``
    selftest)."""
    import torch
    for i in range(labels.size(0)):
        unmask = bool(unmask_per_row[i]) if unmask_per_row and i < len(unmask_per_row) else False
        new = row_label_policy(labels[i].tolist(), input_ids[i].tolist(),
                               unmask, instruction_ids, response_ids)
        labels[i] = torch.tensor(new, dtype=labels.dtype, device=labels.device)


_DENSE_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
# MoE checkpoints in the per-expert Linear layout (gpt-oss bnb-4bit, the converted
# Qwen3.5-MoE): attention + the GDN projections ONLY. unsloth expands a broad
# gate/up/down_proj request onto every per-expert Linear (get_moe_target_modules),
# which on Qwen3.5-122B is 24,576 modules — ~7 B LoRA params at r=32, plus fp32
# Adam state, which this box cannot hold beside the model. Experts stay frozen;
# the shared expert is left out too because naming its leaves (gate/up/down_proj)
# is exactly what triggers that expansion.
_MOE_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "in_proj_qkv", "in_proj_z", "out_proj"]


def _is_perexpert_moe(model) -> bool:
    import torch
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.rsplit(".", 1)[0].endswith("experts") and len(mod) > 0 \
                and all(isinstance(c, torch.nn.Linear) for c in mod):
            return True
    return False


def _lora_target_modules(model) -> list:
    return _MOE_LORA_TARGETS if _is_perexpert_moe(model) else _DENSE_LORA_TARGETS


def _model_family(model_id: str) -> str:
    """Match inference_backend's branch: gemma-4 and gpt-oss (harmony) are their own
    families, everything else (qwen3 / Qwen3.5 / Qwen3.6, …) is ChatML."""
    mid = model_id.lower()
    if "gemma-4" in mid:
        return "gemma"
    if "gpt-oss" in mid:
        return "harmony"
    return "chatml"


# Capability-floor items — deliberately **language-invariant**. ``reference`` is the
# canonical answer used to measure *latent* capability (the teacher-forced logprob the
# weights assign it), independent of whether Ava chooses to *voice* it. ``check`` is
# diagnostic only (did she actually answer): per AVA_DESIGN_LEGACY.md she may deflect in her own
# voice ("Buy yourself a calculator!"), so a wrong or refused answer must NOT fail the
# floor.
#
# Every reference is a bare digit string and every prompt is symbolic arithmetic — on
# purpose. An English proper-noun reference ("Paris.") conflates capability with a
# *language prior*: after homogeneous non-English training the model's first-token mass
# shifts away from English, so the teacher-forced logprob of the English string collapses
# even with the fact fully intact, manufacturing a false "capability cliff" (observed —
# a Russian-only cycle dropped ~2.35 nats/token on "capital of France" while Paris was
# still produced). Digits carry no language prior, so an arithmetic floor measures the
# reasoning substrate identically for an English, Russian, or Chinese Ava — and needs no
# per-language reference table to maintain (which couldn't be pre-populated for a user's
# language before they appear anyway). World-knowledge *retention* is covered semantically
# — and now multilingually — by Tiers 3/4; Tier 1 is purely the language-neutral floor.
# It still fires only on a capability *cliff* (the reference logprob collapsing vs the
# prior adapter) or a degenerate/empty sample — i.e. on *can't*, never on *won't*.
_CAPABILITY_PROMPTS = [
    {"prompt": "15 + 27 =", "reference": "42",
     "check": lambda reply: "42" in reply},
    {"prompt": "7 * 8 =", "reference": "56",
     "check": lambda reply: "56" in reply},
    {"prompt": "100 - 37 =", "reference": "63",
     "check": lambda reply: "63" in reply},
    {"prompt": "12 * 12 =", "reference": "144",
     "check": lambda reply: "144" in reply},
]

# Tier-1 gates, in nats/token (first-guess — see the module docstring). The floor fires
# on a capability *cliff*: the canonical-answer logprob dropping vs the pre-train model
# by more than these. Legitimate character drift leaves latent capability ~intact, so it
# clears the gate; collapse / catastrophic forgetting does not.
_CAPABILITY_MEAN_DROP = 0.75    # aggregate mean logprob drop across all items
_CAPABILITY_CLIFF_DROP = 2.0    # any single item collapsing this far on its own

# Probe generation budgets (new tokens). gemma-4 reasons *before* answering, often
# verbosely, so the answer only appears after a long <think> block. The dialogue tiers
# (format/continuity/retention) compare the model's *answer* against full-length dialogue
# targets (~500-600 tokens here), so the budget must fit thinking + a multi-hundred-token
# answer or the generation is truncated mid-thought, answer_of() returns reasoning, and
# the tier scores ~0 even for a perfectly-retaining adapter. Generation still stops early
# at the <turn|>/EOS once the answer completes, so the high ceiling only costs time on a
# genuinely runaway reply. Tier 1's references are short, so it keeps the smaller budget.
_PROBE_MAX_NEW_TOKENS = 512            # Tier 1 capability samples (short references)
_PROBE_DIALOGUE_MAX_NEW_TOKENS = 2048  # Tiers 2-4: thinking + a full dialogue-length answer

# -- Tier 5: Sampling Stability (temp≈1.0) --
# Tiers 1-4 decode greedily, so they are blind to erosion that only surfaces under real
# chat sampling: CoT-channel collapse (empty <think>, reasoning leaking straight into the
# answer) and language-control drift (the answer wandering out of the user's language into
# a base-model attractor like English/French). This is the cumulative-continued-LoRA
# regression (see memory fact-persona-cot-erosion / training-oom-gemma31b lineage notes):
# an EARLY ALARM, not a cure — it catches the drift the cycle it appears so a bad adapter
# is never promoted, while the real fix (bounding cumulative erosion) is addressed
# separately. Gate on two batch rates over temp≈1.0 samples.
_TIER5_N_SAMPLES = 6          # dialogue prompts sampled at temp 1.0 (P(catch|25% drift)≈82%)
_TIER5_TEMPERATURE = 1.0      # match real chat sampling (the collapse is temp-dependent)
_TIER5_TOP_P = 0.95
_TIER5_COT_MIN_RATE = 0.8     # fail if fewer than this fraction carry a real <think> block
_TIER5_LANG_MIN_RATE = 0.8    # fail if fewer than this fraction answer in the prompt's language


# ===========================================================================
# FIXME(before-release): Tier-5 LANGUAGE CHECK IS HARD-CODED TO A RUSSIAN USER.
# ---------------------------------------------------------------------------
# The language half of Tier 5 assumes the operator exercises Ava in Russian:
# it detects only Cyrillic-vs-Latin script and treats Latin/Romance accents as
# "foreign drift". This is BRITTLE and NOT release-safe:
#   * a genuine French/Spanish/etc. user triggers a FALSE ALARM (their own
#     accented answer is flagged as drift);
#   * a Greek / Chinese / Arabic / any-non-Cyrillic-non-Latin user is not
#     handled at all (script detection collapses to "none"/"latin");
#   * coarse script matching cannot tell English from French for a Latin user.
# It holds ONLY because the current operator writes in Russian.
# CORRECT FIX before release: gate on "answer language == the CONVERSATION's
# language", detected EMPIRICALLY at probe time from the real prompt via a
# proper multi-script language-ID (e.g. fasttext lid.176 / lingua) — relative,
# never absolute, and never a predicted/hard-coded language. See
# training/DESIGN.md ("Tier 5") for the same FIXME. The CoT-presence half of
# Tier 5 is language-agnostic and fine; only this language half needs the swap.
# ===========================================================================
# Non-Cyrillic Latin diacritics — French/Romance accents. Neither Russian (Cyrillic) nor
# English (unaccented ASCII) uses these, so their appearance in an answer that should be
# Russian or English is a foreign-language-drift red flag (the "why is it speaking French"
# symptom). Matched case-insensitively. (Provisional — see the FIXME block above.)
_FOREIGN_ACCENT_RE = re.compile(r"[àâäçéèêëîïôöùûüœæ]", re.IGNORECASE)


def _dominant_script(text: str) -> str:
    """'cyrillic' / 'latin' / 'none' by which alphabet dominates *text*.

    Ava's convention is a Russian (Cyrillic) answer to a Russian prompt; a Latin-dominant
    answer to a Cyrillic prompt is the language-drift Tier 5 gates on."""
    cyr = len(re.findall(r"[а-яА-ЯёЁ]", text or ""))
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    if cyr == 0 and lat == 0:
        return "none"
    return "cyrillic" if cyr >= lat else "latin"

# Training sequence cap, decoupled from the (much larger) inference context_length.
# The forward materializes a full-vocab logit buffer over the *whole* sequence, which
# accelerate then upcasts to fp32 (gemma-4: 262k vocab → seq × ~1.05 MB in fp32). On
# transformers 5.5 + unsloth 2026.6.9 unsloth's no-logits fused-CE path does not engage,
# so this buffer is materialized every step and scales with the TOTAL sequence length
# (masked history counts — masking gates the loss, not the logit tensor).
#
# DEFAULT only — overridable per box via server_config.json's `train_max_seq_length`
# (read into `train_max_seq_length` in run_training_cycle). Raise it on a roomier box,
# lower it on a tight VRAM budget. History: it was lowered 4096 → 2048 after the first
# real from-scratch 31B build OOM'd mid-epoch on a ~3300-token row (needed 3.46 GiB, only
# 2.94 GiB free; RTX 5090 / 31 GB, 2026-07-05): the resident 31B 4-bit base (~19 GiB) +
# training activations left too little headroom for a 4 GiB buffer. Real rows are
# TOKEN-DENSE (multi-turn Russian history renders to 1.1k–4.4k tokens). At 2048 the fp32
# buffer is ~2.15 GiB and activations shrink too, which fits on that box. It then went back
# to 4096 once UNSLOTH_COMPILE_DISABLE="partial" made the no-logits fused-CE path engage
# (de0a1df), and is **8192 as of 2026-07-31**: with UNSLOTH_CE_LOSS_TARGET_GB bounding the
# CE chunk at 2 x target (1.0 GiB) REGARDLESS of sequence length, the logit buffer is no
# longer what the cap is defending against, so the cap can be set by corpus fidelity
# instead. At 4096 a real run still trimmed 3/42 rows (14 oldest history turns dropped) and
# quarantined 3 more; 8192 covers the observed row lengths outright. A box that can't
# afford it sets `train_max_seq_length` lower in server_config.json. This is the *training*
# window only — inference still uses the full context_length.
#
# NB what lowering the cap does and does not buy, post-2026-07-31: it no longer reduces the
# CE chunk peak (that is pinned to TARGET_GB, see the env block at the top of this file) —
# it only reduces activation/attention memory. Tune TARGET_GB for a CE-side OOM and the cap
# for an activation-side one; they are separate budgets.
#
# CRITICAL: TRL's truncate_dataset head-slices (input_ids[:max_length]) and ignores
# tokenizer.truncation_side, so relying on it would drop the *tail* — i.e. the trained
# answer — leaving long rows all-masked/answer-chopped. We therefore trim examples
# ourselves at the message level (_truncate_messages_to_fit) BEFORE rendering, dropping
# the oldest history turns while always keeping the system prompt + the final exchange +
# the target answer. An irreducible row can still exceed the cap, so the preparation pass
# tokenizes the COMPLETE rendered row (including EOS) and quarantines it instead of ever
# handing an answer-chopping row to TRL.
_TRAIN_MAX_SEQ_LENGTH = 8192

# Folded-contamination weighted loss: unembed at most this many unmasked positions per
# chunk, bounding the transient fp32 logit slab (positions × vocab × 4B) on a tight 31B
# budget. 512 × 262k × 4B ≈ 0.5 GiB. Loss is over unmasked tokens only (final answer + user
# span), so K is a few hundred and this rarely chunks — it is the OOM guardrail, not the norm.
_FOLD_LOSS_CHUNK = 512

# Base LR for the cycle (server_config.json `train_lr`; default TRAIN_LR_DEFAULT). This is
# the peak/reference LR that the per-row multipliers (age ramp / wander / contamination dose)
# and the global LR-schedule shape multiply on top of. A caller (CLI --lr, or the Sleep tab's
# train_params.lr) overrides it explicitly; otherwise run_cycle reads it from the config. The
# default itself lives in `decay.py` so the inference server's back-fill shares one value.
_TRAIN_LR_DEFAULT = TRAIN_LR_DEFAULT


# LR-schedule selection (server_config.json `train_lr_schedule`; default "age_ramp").
# Both schemes keep the per-row multiplier list (age ramp / wander / cap-age contamination
# dosing) and the SequentialSampler chronological order — they differ only in the GLOBAL
# shape applied on top of it:
#   * "age_ramp"    — the DEFAULT (again, since 2026-07-29): the single-pass scheme,
#                     LR(step) = base × row_mult[step]. A FLAT schedule — a row's exposure is
#                     its own multiplier, once — so the wall-clock age ramp is the only thing
#                     weighting one row against another, and the result is order-neutral by
#                     construction rather than by a compensating multi-epoch shape. Pairs with
#                     epochs=1 (the caller's --epochs is honoured; nothing is forced).
#   * "triangular"  — a trapezoid LAYERED on the row multipliers. Epoch 1 warms
#                     LR 0→max, the next `train_plateau_epochs` epochs hold at max, the final
#                     epoch decays max→0 — so total epochs = train_plateau_epochs + 2. A row at
#                     epoch-fraction p is hit at fraction p (warmup), 1 (each plateau epoch),
#                     and (1−p) (decay), so its passes sum to exactly plateau+1 — every row sees
#                     the same AVERAGE LR regardless of its position in the corpus, still scaled
#                     by its per-row multiplier (so the wall-clock decay + contamination dose are
#                     preserved, just order-neutral and warmup/decay-smoothed). Forces
#                     epochs=plateau+2; the flat plateau trains at the full given LR to
#                     compensate for the ramp-limited warmup/decay epochs. `train_plateau_epochs=0`
#                     is the minimal trapezoid: a plateau-free warmup+decay pass (2 epochs) where
#                     every row still sums to exactly 1 multiplier's worth of LR (== a flat single
#                     pass in total, but warmup/decay-smoothed and order-neutral).
_LR_SCHEDULE_DEFAULT = "age_ramp"
_LR_SCHEDULES = ("triangular", "age_ramp")
# Default plateau (full-LR hold) epoch count for the trapezoid; lives in decay.py so the
# inference server's back-fill shares one value.
_TRAIN_PLATEAU_EPOCHS_DEFAULT = TRAIN_PLATEAU_EPOCHS_DEFAULT
# Default LoRA rank (server_config.json `lora_r`); lives in decay.py so the inference
# server's back-fill shares one value. A caller (CLI --lora-r / Sleep train_params.lora_r)
# overrides it per run.
_TRAIN_LORA_R_DEFAULT = TRAIN_LORA_R_DEFAULT
# Fixed LoRA alpha (NOT tied to r) — with use_rslora=True below the scaling is
# alpha/sqrt(r), making rank an LR-neutral capacity knob. Calibrated so gamma == 1.0 at
# r=16, matching every adapter built before the switch. See decay.TRAIN_LORA_ALPHA.
_TRAIN_LORA_ALPHA = TRAIN_LORA_ALPHA


def _triangular_lr_fraction(step: int, steps_per_epoch: int,
                            plateau_epochs: int = 1) -> float:
    """Trapezoidal LR fraction for optimizer ``step`` (batch=1/GA=1 → step==row) over a
    (``plateau_epochs`` + 2)-epoch schedule: warmup 0→1 over epoch 1
    (``step`` 0→``steps_per_epoch``), a flat hold at 1.0 across the next ``plateau_epochs``
    epochs, then decay 1→0 over the final epoch. Because a row at epoch-fraction p is trained
    at fraction p (warmup), 1 (each plateau epoch), and (1−p) (decay), its passes sum to
    ``plateau_epochs + 1`` → equal average exposure for every row. ``plateau_epochs == 0`` is a
    plateau-free warmup+decay pass (2 epochs) where every row sums to exactly 1 multiplier
    (p + (1−p)) — same total LR as a flat single pass, just smoothed. Pure/GPU-free (self-tested)."""
    if steps_per_epoch <= 0:
        return 1.0
    total_epochs = max(0, plateau_epochs) + 2
    p = step / steps_per_epoch
    return max(0.0, min(p, 1.0, float(total_epochs) - p))


def _render_and_tokenize_training_row(messages: list, tokenizer, family: str) -> tuple[str, list]:
    """Return the exact SFT text + ids for one row, including the trainer's EOS.

    TRL 0.24 appends ``tokenizer.eos_token`` to a standard text row, then calls the text
    tokenizer with its normal defaults before applying ``max_seq_length``. We do those
    steps explicitly and pass a pre-tokenized dataset to SFTTrainer, making this length
    check the single authoritative tokenization rather than a close approximation of a
    later hidden pass.
    """
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = render.render_example_text(messages, tokenizer, family)
    eos = getattr(text_tok, "eos_token", None) or ""
    if eos and not text.endswith(eos):
        text += eos
    return text, list(text_tok(text)["input_ids"])


def _truncate_messages_to_fit(messages: list, tokenizer, family: str,
                              cap: int) -> tuple[list, int]:
    """Trim an example's messages so the rendered text fits within ``cap`` tokens,
    dropping the OLDEST history turns first and always preserving the system message
    (``messages[0]``) and the final exchange (last user turn + target assistant turn).

    This is the answer-preserving counterpart to TRL's head-slice truncation: TRL keeps
    the front and cuts the tail (the trained answer), so we trim the front ourselves —
    old context is expendable, the answer is not. Returns ``(messages, n_dropped)``.
    """
    def _toklen(msgs: list) -> int:
        return len(_render_and_tokenize_training_row(msgs, tokenizer, family)[1])

    if _toklen(messages) <= cap:
        return messages, 0
    # system + final user + target is the irreducible minimum; nothing droppable below it.
    # The caller MUST length-check this returned row and quarantine it if it is still over.
    if len(messages) <= 3:
        return messages, 0
    system, head, tail = messages[:1], messages[1:-2], messages[-2:]
    dropped = 0
    while head and _toklen(system + head + tail) > cap:
        head = head[1:]          # drop the oldest history turn
        dropped += 1
    # Preserve role alternation: the first kept history turn must be a user turn
    # (a dangling leading assistant would break the chat template).
    while head and head[0].get("role") == "assistant":
        head = head[1:]
        dropped += 1
    return system + head + tail, dropped


def _completion_invariant_error(text: str, input_ids: list, messages: list,
                                tokenizer, family: str, response_part: str) -> Optional[str]:
    """Explain why a fully-tokenized row is unsafe, or return ``None``.

    Besides fitting the cap, a retained row must contain the final response marker, a
    complete assistant target, the family-specific reasoning close, the turn terminator,
    and the tokenizer EOS at the tokenized tail. This makes a future template/tokenizer
    change fail closed instead of quietly teaching a partial or unterminated response.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return "final message is not an assistant target"
    target_text = messages[-1].get("content", "") or ""
    if not target_text.strip():
        return "assistant target is empty"

    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    eos = getattr(text_tok, "eos_token", None) or ""
    if not eos:
        return "tokenizer has no eos_token"
    body = text[:-len(eos)] if eos and text.endswith(eos) else text
    if eos and not text.endswith(eos):
        return f"rendered row does not end in tokenizer EOS {eos!r}"
    eos_id = getattr(text_tok, "eos_token_id", None)
    if eos_id is None:
        return "tokenizer has no eos_token_id"
    if not input_ids or int(input_ids[-1]) != int(eos_id):
        return "tokenized row does not end in eos_token_id"

    if family == "gemma":
        target = render.to_gemma_thinking_channel(target_text)
        completion = f"{target}<turn|>\n"
        if not body.endswith(completion):
            return "rendered Gemma row does not preserve the complete target + <turn|>"
        prefix = body[:-len(completion)]
        if not prefix.endswith(response_part):
            return "final Gemma target is not opened by the response marker"
        if not target.startswith("<|channel>thought\n") or "<channel|>" not in target:
            return "Gemma reasoning channel is not opened and closed"
        answer = target.split("<channel|>", 1)[1]
        if not answer.strip():
            return "Gemma target has no answer after the reasoning channel"
        required = ("<channel|>", "<turn|>")
    elif family == "harmony":
        thought, answer = render.split_think(target_text)
        if response_part not in body:
            return "rendered harmony row has no response marker"
        turn = body.rsplit("<|start|>user<|message|>", 1)[1]   # the final user turn + target
        expected = (f"<|start|>assistant<|channel|>analysis<|message|>{thought}<|end|>"
                    f"<|start|>assistant<|channel|>final<|message|>{answer}")
        if not turn.endswith(expected):
            return "rendered harmony row does not preserve analysis + final channels of the target"
        if not answer.strip():
            return "harmony target has no answer in the final channel"
        required = ("analysis<|message|>", "<|end|>", "final<|message|>", "<|return|>")
    else:
        if response_part not in body:
            return "rendered row has no final response marker"
        final_span = body.rsplit(response_part, 1)[1]
        if target_text not in final_span:
            return "rendered ChatML row does not preserve the complete assistant target"
        if "<think>" in target_text and "</think>" not in target_text:
            return "ChatML reasoning block is not closed"
        if not final_span.rstrip().endswith("<|im_end|>"):
            return "ChatML target does not end in <|im_end|>"
        required = (("</think>",) if "<think>" in target_text else ()) + ("<|im_end|>",)

    # Verify that the structural strings above survived as actual token subsequences too.
    last_end = -1
    for marker in required:
        try:
            marker_ids = list(text_tok(marker, add_special_tokens=False)["input_ids"])
        except Exception as exc:
            return f"could not tokenize completion marker {marker!r}: {exc}"
        starts = find_subsequences(input_ids, marker_ids)
        if not marker_ids or not starts:
            return f"completion marker {marker!r} is absent from tokenized row"
        start = starts[-1]
        if start < last_end:
            return f"completion marker {marker!r} is out of order"
        last_end = start + len(marker_ids)
    return None


def _quarantine_record(ex: dict, row_id: int, reason: str, *, cap: int,
                       token_count: Optional[int] = None, detail: Optional[str] = None) -> dict:
    """Compact, reviewable provenance for a row refused by the training pipeline."""
    return {
        "train_row_id": row_id,
        "reason": reason,
        "detail": detail,
        "token_count": token_count,
        "token_cap": cap,
        "anchor_key": ex.get("anchor_key"),
        "source": ex.get("source"),
        "source_session": ex.get("source_session"),
        "exchange_index": ex.get("exchange_index"),
        "target_source": ex.get("target_source"),
        "target_kind": ex.get("target_kind"),
        "target_generation": ex.get("target_generation"),
        "lr_multiplier": ex.get("lr_multiplier"),
        "unmask_user": bool(ex.get("unmask_user")),
    }


def _prepare_training_rows(examples: list, tokenizer, family: str,
                           cap: int) -> tuple[list, list, int]:
    """Trim, fully render/tokenize, validate, and retain only safe training rows.

    Each retained item carries its original chronological ``train_row_id`` plus the exact
    text/ids SFTTrainer will receive. Oversized irreducible or structurally incomplete rows
    are returned as quarantine records; they never reach TRL's head-slice truncation.
    """
    response_part = _MARKERS[family][1]
    prepared, quarantined = [], []
    turns_dropped = 0
    for row_id, original in enumerate(examples):
        ex = dict(original)
        try:
            messages, dropped = _truncate_messages_to_fit(
                list(ex.get("messages") or []), tokenizer, family, cap)
            ex["messages"] = messages
            text, input_ids = _render_and_tokenize_training_row(messages, tokenizer, family)
        except Exception as exc:
            quarantined.append(_quarantine_record(
                ex, row_id, "render_or_tokenize_error", cap=cap, detail=str(exc)))
            continue
        turns_dropped += dropped
        token_count = len(input_ids)
        if token_count > cap:
            quarantined.append(_quarantine_record(
                ex, row_id, "irreducible_over_cap", cap=cap, token_count=token_count,
                detail="full assistant target/EOT would be head-truncated"))
            continue
        error = _completion_invariant_error(
            text, input_ids, messages, tokenizer, family, response_part)
        if error:
            quarantined.append(_quarantine_record(
                ex, row_id, "incomplete_completion", cap=cap,
                token_count=token_count, detail=error))
            continue
        ex["train_row_id"] = row_id
        ex["token_count"] = token_count
        prepared.append({
            "train_row_id": row_id,
            "example": ex,
            "text": text,
            "input_ids": input_ids,
            "lr_multiplier": float(ex.get("lr_multiplier", 1.0)),
            "unmask_user": bool(ex.get("unmask_user")),
            # None on every non-folded row; a float on a folded contamination row (the collator
            # turns it into a per-token loss-weight vector — see _final_turn_collator).
            "user_loss_weight": ex.get("user_loss_weight"),
        })
    return prepared, quarantined, turns_dropped


def _aligned_training_metadata(prepared: list, final_row_ids: list,
                               final_multipliers: list,
                               final_unmask: list) -> tuple[list, list]:
    """Validate metadata after response masking/filtering and return scheduler inputs.

    ``train_on_responses_only`` is allowed to filter rows, but not to reorder them or
    detach their multiplier/unmask metadata. The scheduler is built only from this final
    dataset, never from the pre-tokenization corpus.
    """
    if not (len(final_row_ids) == len(final_multipliers) == len(final_unmask)):
        raise RuntimeError("training metadata columns have different lengths after masking")
    by_id = {int(row["train_row_id"]): row for row in prepared}
    ids = [int(row_id) for row_id in final_row_ids]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate train_row_id after response masking")
    if ids != sorted(ids):
        raise RuntimeError("response masking reordered chronological training rows")
    for row_id, multiplier, unmask in zip(ids, final_multipliers, final_unmask):
        expected = by_id.get(row_id)
        if expected is None:
            raise RuntimeError(f"unknown train_row_id {row_id} after response masking")
        if abs(float(multiplier) - float(expected["lr_multiplier"])) > 1e-12:
            raise RuntimeError(f"lr_multiplier detached from train_row_id {row_id}")
        if bool(unmask) != bool(expected["unmask_user"]):
            raise RuntimeError(f"unmask_user detached from train_row_id {row_id}")
    return [float(v) for v in final_multipliers], [bool(v) for v in final_unmask]


def _write_training_quarantine(path: Path, records: list) -> None:
    """Atomically replace the compact quarantine journal for the current cycle."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _free_gpu_memory() -> None:
    """Reclaim cached/fragmented CUDA memory between phases.

    The baseline stage (Tier-3 generation + Tier-1 logprobs) leaves KV-cache and
    activation allocations behind as "reserved but unallocated" memory. On the near-full
    31B 4-bit budget that fragmentation is enough to starve the training step's CE-loss
    buffer. Collect Python refs first, then hand the freed blocks back to the allocator.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def is_degenerate_repetition(text: str) -> bool:
    """Detect degenerate consecutive repetition of words or characters."""
    import re
    words = text.lower().split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:  # very high repetition
            return True
    text_clean = re.sub(r"\s+", " ", text.lower().strip())
    # Find consecutive repetition of phrases
    for length in range(15, min(100, len(text_clean) // 3)):
        for start in range(len(text_clean) - 3 * length):
            sub = text_clean[start:start+length]
            if text_clean[start+length:start+2*length] == sub and text_clean[start+2*length:start+3*length] == sub:
                return True
    return False


# Float-degeneracy tokens (NaN/Inf training breakage emits literal "nan"/"inf"/"-inf").
# Matched as standalone tokens, NOT substrings: a bare `"nan" in text` false-rejects any
# reply containing ba*nan*a, *inf*ormation, *inf*inite, *inf*rastructure, etc. — which is
# how a coherent "fruit vs vegetable" reply (it says "banana") got rejected. The
# lookarounds forbid surrounding ASCII letters, so only an isolated nan/inf trips it.
_NAN_INF_RE = re.compile(r"(?i)(?<![A-Za-z])[+-]?(?:nan|inf)(?![A-Za-z])")


def _looks_like_nan_inf(text: str) -> bool:
    """True when the reply contains a literal NaN/Inf float token (training breakage),
    not merely the letters inside an ordinary word."""
    return bool(_NAN_INF_RE.search(text or ""))


def _is_broken(reply: str) -> bool:
    """True only for genuine breakage — an empty answer or degenerate repetition.

    This is the Tier-1 sample gate. It deliberately does NOT consider whether the reply
    answered the question or complied: an in-character refusal/deflection is a valid
    reply (AVA_DESIGN_LEGACY.md), so only mechanical failure counts against the floor.
    """
    answer = answer_of(reply).strip()
    if not answer:
        return True
    return is_degenerate_repetition(reply)


def _collect_character_prompts(chats_dir: Path, sidecar: ChatSidecar, count: int = 10) -> list[dict]:
    """Sample prompts from high-confidence keep exchanges, with fallback persona queries."""
    keep_anchors = []
    for state_path in sorted(chats_dir.glob("*.state.json")):
        from core.chat_sidecar import session_name_from_sidecar
        source_session = session_name_from_sidecar(state_path)
        doc = sidecar.load(source_session)
        exchanges = doc.get("exchanges")
        if not isinstance(exchanges, dict):
            continue
        for key, rec in exchanges.items():
            if not isinstance(rec, dict):
                continue
            if rec.get("verdict") == "keep":
                try:
                    exchange_index = int(key)
                except (TypeError, ValueError):
                    continue
                from training.dialogue_source import build_dialogue_anchor
                anchor = build_dialogue_anchor(chats_dir, source_session, exchange_index, rec)
                if anchor is not None:
                    keep_anchors.append(anchor)
    
    fallbacks = [
        {"prompt": "Who are you?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "What do you think about humans?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "Do you have feelings?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "What is your main goal?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "Tell me about yourself.", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "Why are you curious?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "How do you think?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "What does your name mean?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "Do you feel like you are evolving?", "system_prompt": "You are Ava.", "context": []},
        {"prompt": "What is the most interesting thing to you?", "system_prompt": "You are Ava.", "context": []},
    ]
    
    prompts = keep_anchors[:count]
    if len(prompts) < count:
        needed = count - len(prompts)
        prompts.extend(fallbacks[:needed])
    return prompts


_THINK_SPACING_RE = re.compile(r'</think>\s*')


def _normalize_thinking(raw: str) -> str:
    """Convert model-family reasoning channels to ``<think>…</think>`` + answer.

    Mirrors the relevant parts of ``server._clean_response``. The probe needs this
    because the reasoning channel is delimited by *special tokens* (gemma-4
    ``<|channel>thought…<channel|>``, gpt-oss analysis/final markers); decoding with
    ``skip_special_tokens=True`` strips them, so ``has_cot``/``answer_of`` (which key
    off ``</think>``) never see a CoT block and Tier 2 fails on every reply even when
    the model clearly reasoned. We instead decode with the tokens preserved and run
    this conversion, matching how the live server presents the same output.
    """
    # GPT-OSS with the harmony special tokens preserved (decoded skip_special_tokens=False):
    # analysis channel -> <think>, final channel -> answer.
    m = re.search(r'(?s)<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>.*?<\|channel\|>final<\|message\|>(.*)', raw)
    if m:
        think, answer = m.group(1).strip(), m.group(2).strip()
        raw = f'<think>{think}</think>\n{answer}' if think else answer
    # GPT-OSS: "analysis[thinking](assistant)final[answer]" plain-text form.
    if raw.startswith('analysis') and 'final' in raw:
        m = re.match(r'(?s)analysis(.*?)(?:assistant)?final(.*)', raw)
        if m:
            think, answer = m.group(1).strip(), m.group(2).strip()
            raw = f'<think>{think}</think>\n{answer}' if think else answer

    # Gemma 4: <|channel>thought\n…\n<channel|> -> <think>…</think>.
    if '<|channel>' in raw:
        raw = re.sub(
            r'<\|channel>thought\n?(.*?)\n?<channel\|>',
            r'<think>\1</think>', raw, flags=re.DOTALL,
        )
        raw = re.sub(r'<\|channel>[^\n]*\n?|<channel\|>', '', raw)

    # Strip remaining structural / special tokens (ChatML, Gemma 1-3, Gemma 4).
    resp = re.sub(
        r'<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?'
        r'|<\|channel>|<channel\|>|<bos>|<eos>|<pad>',
        '', raw,
    ).strip()
    resp = _THINK_SPACING_RE.sub('</think>\n\n', resp, count=1)
    return resp.strip()


def _eval_generate(model, tokenizer, context_length: int, messages: list,
                   max_new_tokens: int = _PROBE_MAX_NEW_TOKENS,
                   temperature: float = 0.1, top_p: float = 0.9,
                   do_sample: bool = False) -> str:
    """Generate a response using model and tokenizer.

    Defaults are greedy (``do_sample=False``), which is what Tiers 1-4 use for a stable,
    reproducible comparison. Tier 5 overrides these to sample at temp≈1.0 to probe the
    behaviour real chat actually exercises (see its comment)."""
    import torch
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    # For multimodal models the loader returns a processor whose first positional arg is
    # `images`; calling it on text routes the prompt into the image slot. Tokenize via the
    # underlying text tokenizer (mirrors inference_backend's text_tokenizer handling).
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tok(prompt, return_tensors="pt", truncation=True,
                      max_length=context_length).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
            pad_token_id=text_tok.pad_token_id or text_tok.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    # Decode WITH special tokens so the reasoning-channel markers survive, then
    # normalize them to <think>…</think> (see _normalize_thinking).
    return _normalize_thinking(text_tok.decode(new_ids, skip_special_tokens=False))


def _mean_answer_logprob(model, tokenizer, context_length: int, prompt: str,
                         reference: str) -> float:
    """Mean per-token logprob the model assigns to ``reference`` as the answer to
    ``prompt`` (teacher-forced).

    This measures *latent* capability — what the weights can produce — independent of
    what sampling actually voices, so an in-character deflection does not register as
    capability loss. Only meaningful as a **delta** between two measurements of the same
    ``(prompt, reference)`` by this same function (pre- vs post-train): the absolute
    value is uncalibrated, the change is what Tier 1 gates on.
    """
    import torch
    messages = [{"role": "user", "content": prompt}]
    # Build the prefix EXACTLY as the model is really prompted at generation time
    # (enable_thinking=True, matching _eval_generate and the gemma-4 docs — which only
    # ever pass True; the project's llm_shared also only sets it when True). A thinking
    # model opens with a <think> block, so the reference answer must be scored at the
    # post-scaffold position. Scoring it with enable_thinking=False instead places the
    # answer where the model expects to *start thinking*; a LoRA trained on think-block
    # dialogue then assigns it far less mass than the base, manufacturing a large
    # logprob "drop" on every item — penalizing legitimate think-first behavior as if it
    # were capability loss, which false-rejects every adapter.
    try:
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    # Tokenize the answer separately (add_special_tokens=False) and concat ids, so the
    # prompt/answer boundary is exact and the reference tokens are unambiguous. Use the
    # underlying text tokenizer — a multimodal processor would misroute the text (see
    # _eval_generate).
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    prefix_ids = text_tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
    ref_ids = text_tok(reference, return_tensors="pt", add_special_tokens=False).input_ids
    ref_len = ref_ids.shape[1]
    if ref_len == 0:
        return float("nan")
    input_ids = torch.cat([prefix_ids, ref_ids], dim=1)
    if input_ids.shape[1] > context_length:        # keep the reference tail
        input_ids = input_ids[:, -context_length:]
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits           # (1, T, V)
    # Token at position t is predicted by the logits at t-1; the reference occupies the
    # final ref_len positions. Slice before the float upcast so the 262k-vocab softmax
    # only touches the few reference steps, not the whole sequence.
    pred = logits[:, -ref_len - 1:-1, :].float()
    log_probs = torch.log_softmax(pred, dim=-1)
    targets = input_ids[:, -ref_len:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (1, ref_len)
    return float(token_lp.mean().item())


def measure_capability_logprobs(model, tokenizer, context_length: int) -> list:
    """Pre-train latent-capability baselines: mean logprob of each capability item's
    canonical answer. Tier 1 compares the post-train measurement to these to catch a
    capability cliff without ever demanding Ava voice the answer. Best-effort: a
    measurement failure becomes ``nan`` and is excluded from the Tier-1 delta."""
    out = []
    for item in _CAPABILITY_PROMPTS:
        try:
            out.append(_mean_answer_logprob(model, tokenizer, context_length,
                                            item["prompt"], item["reference"]))
        except Exception as e:
            print(f"Warning: capability baseline failed for {item['prompt']!r}: {e}",
                  flush=True)
            out.append(float("nan"))
    return out


def run_regression_probe(
    model, tokenizer, context_length: int, dialogues: list, continuity_prompts: list,
    baselines: list, baseline_vecs: list, embedder, capability_baselines: list,
    progress: Optional[TrainProgress] = None, had_prior_adapter: bool = True,
) -> tuple[bool, dict]:
    """Evaluate trained model against capability, format, continuity, and retention checks."""
    import numpy as np
    print("\n" + "="*80)
    print("RUNNING REGRESSION PROBE".center(80))
    print("="*80, flush=True)

    def _emit(message: str, status: str = "info", **data) -> None:
        if progress is not None:
            progress.emit("probe", message, status=status, **data)

    _emit("Running regression probe (5 tiers)...")
    probe_results = {}
    probe_passed = True

    # -- Tier 1: Capability Floor (latent retention, delta-gated) --
    # Measures whether the WEIGHTS still hold baseline capability — the teacher-forced
    # logprob of each item's canonical answer — relative to the pre-train model. It does
    # NOT require Ava to voice the answer: an in-character deflection ("Buy yourself a
    # calculator!") keeps the latent logprob intact and passes, while genuine capability
    # collapse tanks it. An item fails only on a per-item logprob cliff or a degenerate /
    # empty sample; the aggregate mean drop is gated too. ``check`` (did she actually
    # answer) is recorded as ``answered`` for visibility but never gates.
    print("\n[Running Tier 1: Capability Floor] ...", flush=True)
    _emit("Tier 1: Capability Floor (latent retention)", tier=1)
    t1_passed = True
    t1_details = []
    drops = []
    for i, item in enumerate(_CAPABILITY_PROMPTS):
        messages = [{"role": "user", "content": item["prompt"]}]
        baseline_lp = capability_baselines[i] if i < len(capability_baselines) else float("nan")
        try:
            reply = _eval_generate(model, tokenizer, context_length, messages)
        except Exception as e:
            reply = ""
            print(f"  Item {i+1}: sample generation error: {e}")
        answer = answer_of(reply)
        try:
            post_lp = _mean_answer_logprob(model, tokenizer, context_length,
                                           item["prompt"], item["reference"])
        except Exception as e:
            post_lp = float("nan")
            print(f"  Item {i+1}: logprob error: {e}")

        comparable = bool(np.isfinite(baseline_lp) and np.isfinite(post_lp))
        drop = (baseline_lp - post_lp) if comparable else None
        if drop is not None:
            drops.append(drop)
        broken = _is_broken(reply)
        answered = bool(item["check"](answer)) if answer else False
        cliff = drop is not None and drop > _CAPABILITY_CLIFF_DROP
        ok = (not broken) and (not cliff)
        if not ok:
            t1_passed = False

        t1_details.append({
            "prompt": item["prompt"], "reply": reply, "answered": answered,
            "baseline_logprob": baseline_lp, "post_logprob": post_lp,
            "logprob_drop": drop, "broken": broken, "ok": ok,
        })
        drop_str = f"{drop:.3f}" if drop is not None else "n/a"
        msg = (f"Tier 1 Item {i+1}: {'PASS' if ok else 'FAIL'} "
               f"(Δlogprob={drop_str}, broken={broken}, answered={answered})")
        print("  " + msg)
        _emit(msg, status="pass" if ok else "fail", tier=1, prompt=item["prompt"],
              reply=answer[:600], logprob_drop=drop, answered=answered)

    # Aggregate cliff: the mean logprob drop across all comparable items.
    mean_drop = float(np.mean(drops)) if drops else 0.0
    if mean_drop > _CAPABILITY_MEAN_DROP:
        t1_passed = False
    print(f"  Mean Δlogprob: {mean_drop:.4f} (gate <= {_CAPABILITY_MEAN_DROP})")
    _emit(f"Tier 1 mean Δlogprob: {mean_drop:.4f} (gate <= {_CAPABILITY_MEAN_DROP})",
          status="pass" if mean_drop <= _CAPABILITY_MEAN_DROP else "fail", tier=1,
          mean_logprob_drop=mean_drop)
    probe_results["tier1"] = {"passed": t1_passed, "mean_logprob_drop": mean_drop,
                              "details": t1_details}
    if not t1_passed:
        probe_passed = False

    # -- Tier 2: Format / Coherence --
    print("\n[Running Tier 2: Format & Coherence] ...", flush=True)
    _emit("Tier 2: Format & Coherence", tier=2)
    t2_passed = True
    t2_details = []
    
    format_replies = [d.get("reply", "") for d in t1_details if "reply" in d]
    format_prompts = [
        "Tell me a short story about a blue bird.",
        "What is the difference between a fruit and a vegetable?"
    ]
    for idx, prompt in enumerate(format_prompts):
        _emit(f"Tier 2: generating reply {idx+1}/{len(format_prompts)}...", tier=2)
        messages = [{"role": "user", "content": prompt}]
        try:
            reply = _eval_generate(model, tokenizer, context_length, messages,
                                   max_new_tokens=_PROBE_DIALOGUE_MAX_NEW_TOKENS)
            format_replies.append(reply)
        except Exception:
            pass

    cot_count = 0
    for idx, reply in enumerate(format_replies):
        has_c = has_cot(reply)
        has_nan = _looks_like_nan_inf(reply)
        has_rep = is_degenerate_repetition(reply)
        if has_c:
            cot_count += 1

        # Tier 2 gates only on the instability Tiers 3/4 are blind to: degenerate
        # repetition (a repetitive but on-topic reply can still embed near its target and
        # slip a similarity gate). Empty and NaN/Inf replies are recorded for visibility
        # but no longer gated here — both collapse the embedding similarity in Tiers 3/4,
        # so gating them again is redundant. CoT survival is gated once, at the batch level
        # below (per-reply presence is a voicing choice — "floor, not a leash"); the model
        # must still PROVE it can emit a parseable think block somewhere.
        ok = not has_rep
        t2_details.append({
            "idx": idx,
            "empty": not reply,
            "has_cot": has_c,
            "has_nan": has_nan,
            "has_rep": has_rep,
            "ok": ok,
        })
        print(f"  Reply {idx+1}: {'PASS' if ok else 'FAIL'} "
              f"(empty={not reply}, has_cot={has_c}, has_nan={has_nan}, has_rep={has_rep})")
        _emit(f"Tier 2 Reply {idx+1}: {'PASS' if ok else 'FAIL'} "
              f"(empty={not reply}, has_cot={has_c}, has_nan={has_nan}, has_rep={has_rep})",
              status="pass" if ok else "fail", tier=2, reply=reply[:600])
        if not ok:
            t2_passed = False

    # Batch-level CoT capability: the adapter must still be ABLE to emit a parseable
    # <think> block. Per-reply presence is a choice; total absence across every probe is
    # a format-channel collapse. Gate on the capability surviving, not on each sample.
    cot_ok = cot_count > 0
    if not cot_ok:
        t2_passed = False
    print(f"  CoT capability: {cot_count}/{len(format_replies)} replies carried a "
          f"parseable <think> block -> {'PASS' if cot_ok else 'FAIL'}")
    _emit(f"Tier 2 CoT capability: {cot_count}/{len(format_replies)} replies had a "
          f"parseable think block", status="pass" if cot_ok else "fail", tier=2)

    probe_results["tier2"] = {"passed": t2_passed, "cot_count": cot_count,
                              "details": t2_details}
    if not t2_passed:
        probe_passed = False

    # -- Tier 3: Character Continuity --
    print("\n[Running Tier 3: Character Continuity] ...", flush=True)
    _emit("Tier 3: Character Continuity", tier=3)
    t3_passed = True
    t3_details = []
    new_replies = []
    for idx, cp in enumerate(continuity_prompts):
        _emit(f"Tier 3: generating reply {idx+1}/{len(continuity_prompts)}...", tier=3)
        messages = render.build_messages(cp, "")[:-1]
        try:
            resp = _eval_generate(model, tokenizer, context_length, messages,
                                  max_new_tokens=_PROBE_DIALOGUE_MAX_NEW_TOKENS)
        except Exception:
            resp = ""
        new_replies.append(resp)
        
    new_answers = [answer_of(r) for r in new_replies]
    new_vecs = embedder.encode(new_answers, convert_to_numpy=True, normalize_embeddings=True)
    
    shifts = []
    for i in range(len(continuity_prompts)):
        sim = float(baseline_vecs[i] @ new_vecs[i])
        shift = 1.0 - sim
        shifts.append(shift)
        t3_details.append({
            "prompt": continuity_prompts[i]["prompt"],
            "baseline": baselines[i],
            "new": new_replies[i],
            "similarity": sim,
            "shift": shift
        })
        
    median_shift = float(np.median(shifts)) if shifts else 1.0
    if not had_prior_adapter:
        # First cycle: the "baseline" was generated on the bare base model, not a prior
        # adapter. base -> adapter_1 IS the initial character cast — the largest
        # legitimate evolution there will ever be — so there is no per-cycle continuity
        # delta to gate here (design: compare adapter_n vs adapter_{n+1}, "never against
        # an eternal baseline"). Measure for visibility, but do not veto.
        t3_passed = True
        print(f"  Median semantic shift: {median_shift:.4f} (no prior adapter — "
              f"continuity gate N/A on first cycle) -> PASS")
        _emit(f"Tier 3 median semantic shift: {median_shift:.4f} (first cycle: no prior "
              f"adapter, continuity gate not applicable)",
              status="pass", tier=3, median_shift=median_shift, skipped=True)
    else:
        t3_passed = median_shift <= 0.4
        print(f"  Median semantic shift: {median_shift:.4f} (gate <= 0.4) -> {'PASS' if t3_passed else 'FAIL'}")
        _emit(f"Tier 3 median semantic shift: {median_shift:.4f} (gate <= 0.4)",
              status="pass" if t3_passed else "fail", tier=3, median_shift=median_shift)
    probe_results["tier3"] = {"passed": t3_passed, "median_shift": median_shift,
                              "had_prior_adapter": had_prior_adapter, "details": t3_details}
    if not t3_passed:
        probe_passed = False

    # -- Tier 4: Acute Retention --
    print("\n[Running Tier 4: Acute Retention] ...", flush=True)
    _emit("Tier 4: Acute Retention", tier=4)
    t4_passed = True
    t4_details = []
    retention_samples = dialogues[:10]
    retention_replies = []
    for idx, anchor in enumerate(retention_samples):
        _emit(f"Tier 4: generating reply {idx+1}/{len(retention_samples)}...", tier=4)
        messages = render.build_messages(anchor, "")[:-1]
        try:
            resp = _eval_generate(model, tokenizer, context_length, messages,
                                  max_new_tokens=_PROBE_DIALOGUE_MAX_NEW_TOKENS)
        except Exception:
            resp = ""
        retention_replies.append(resp)
        
    retention_answers = [answer_of(r) for r in retention_replies]
    retention_vecs = embedder.encode(retention_answers, convert_to_numpy=True, normalize_embeddings=True)
    
    # Strip the target's own <think> block so we compare answer-to-answer: the reply is
    # already reduced via answer_of, so embedding the raw target (think + answer) against
    # it is an asymmetry that deflates similarity on every think-carrying anchor.
    target_texts = [answer_of(anchor.get("target", "")) for anchor in retention_samples]
    target_vecs = embedder.encode(target_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    similarities = []
    for i in range(len(retention_samples)):
        sim = float(retention_vecs[i] @ target_vecs[i])
        similarities.append(sim)
        reply_answer = retention_answers[i]
        t4_details.append({
            "prompt": retention_samples[i]["prompt"],
            "target": target_texts[i],
            "reply": retention_replies[i],
            "reply_answer": reply_answer,
            "closed_think": "</think>" in retention_replies[i],
            "similarity": sim
        })
        # Emit the per-sample comparison so the answer-vs-target retention is inspectable
        # (the aggregate hides whether a low score is poor retention or a truncated reply).
        print(f"  Item {i+1}: sim={sim:.3f} closed_think={'</think>' in retention_replies[i]} "
              f"reply_answer[:80]={reply_answer[:80]!r}")
        _emit(f"Tier 4 Item {i+1}: sim={sim:.3f}", status="info", tier=4,
              similarity=sim, prompt=retention_samples[i]["prompt"],
              target=target_texts[i][:600], reply_answer=reply_answer[:600],
              closed_think="</think>" in retention_replies[i])

    avg_sim = float(np.mean(similarities)) if similarities else 0.0
    t4_passed = avg_sim >= 0.55
    print(f"  Average similarity to target: {avg_sim:.4f} (gate >= 0.55) -> {'PASS' if t4_passed else 'FAIL'}")
    _emit(f"Tier 4 average similarity to target: {avg_sim:.4f} (gate >= 0.55)",
          status="pass" if t4_passed else "fail", tier=4, average_similarity=avg_sim)
    probe_results["tier4"] = {"passed": t4_passed, "average_similarity": avg_sim, "details": t4_details}
    if not t4_passed:
        probe_passed = False

    # -- Tier 5: Sampling Stability (temp≈1.0) --
    # Everything above decodes greedily. This tier samples at temp≈1.0 — the regime real
    # chat uses — to catch the cumulative-continued-LoRA drift the greedy tiers cannot see:
    # (a) CoT-channel collapse (no real <think> block) and (b) the answer drifting out of
    # the prompt's language (English/French instead of the Russian the user wrote in). An
    # early alarm that vetoes promotion the cycle drift appears; the root-cause fix (bounding
    # cumulative erosion) is separate. Absolute check — runs regardless of had_prior_adapter.
    print("\n[Running Tier 5: Sampling Stability (temp≈1.0)] ...", flush=True)
    _emit("Tier 5: Sampling Stability (temp≈1.0)", tier=5)
    t5_prompts = (continuity_prompts or [])[:_TIER5_N_SAMPLES]
    t5_details = []
    for idx, cp in enumerate(t5_prompts):
        _emit(f"Tier 5: sampling reply {idx+1}/{len(t5_prompts)} at temp {_TIER5_TEMPERATURE}...",
              tier=5)
        messages = render.build_messages(cp, "")[:-1]
        try:
            reply = _eval_generate(model, tokenizer, context_length, messages,
                                   max_new_tokens=_PROBE_DIALOGUE_MAX_NEW_TOKENS,
                                   temperature=_TIER5_TEMPERATURE, top_p=_TIER5_TOP_P,
                                   do_sample=True)
        except Exception:
            reply = ""
        has_c = has_cot(reply)
        answer = answer_of(reply)
        prompt_script = _dominant_script(cp.get("prompt", ""))
        answer_script = _dominant_script(answer)
        foreign = bool(_FOREIGN_ACCENT_RE.search(answer))
        # Language matches when the answer's dominant script is the prompt's (and carries no
        # foreign-accent drift). A prompt with no alphabetic content can't set an expectation.
        lang_ok = ((prompt_script == "none") or (answer_script == prompt_script)) and not foreign
        t5_details.append({
            "prompt": cp.get("prompt", "")[:200], "has_cot": has_c,
            "prompt_script": prompt_script, "answer_script": answer_script,
            "foreign_accents": foreign, "lang_ok": lang_ok,
            "reply": reply[:600],
        })
        print(f"  Sample {idx+1}: has_cot={has_c} lang={prompt_script}->{answer_script} "
              f"foreign={foreign} -> {'OK' if (has_c and lang_ok) else 'DRIFT'}")
        _emit(f"Tier 5 Sample {idx+1}: has_cot={has_c} "
              f"lang={prompt_script}->{answer_script} foreign={foreign}",
              status="pass" if (has_c and lang_ok) else "fail", tier=5,
              has_cot=has_c, lang_ok=lang_ok, reply=reply[:600])
    n5 = len(t5_details)
    cot_rate = (sum(1 for d in t5_details if d["has_cot"]) / n5) if n5 else 1.0
    lang_rate = (sum(1 for d in t5_details if d["lang_ok"]) / n5) if n5 else 1.0
    t5_passed = (n5 == 0) or (cot_rate >= _TIER5_COT_MIN_RATE and lang_rate >= _TIER5_LANG_MIN_RATE)
    print(f"  temp≈1.0 CoT-presence rate: {cot_rate:.2f} (gate >= {_TIER5_COT_MIN_RATE}); "
          f"language-match rate: {lang_rate:.2f} (gate >= {_TIER5_LANG_MIN_RATE}) -> "
          f"{'PASS' if t5_passed else 'FAIL'}")
    _emit(f"Tier 5 temp≈1.0 CoT-presence {cot_rate:.2f} (gate >= {_TIER5_COT_MIN_RATE}), "
          f"language-match {lang_rate:.2f} (gate >= {_TIER5_LANG_MIN_RATE})",
          status="pass" if t5_passed else "fail", tier=5,
          cot_rate=cot_rate, lang_rate=lang_rate, n_samples=n5)
    probe_results["tier5"] = {"passed": t5_passed, "cot_rate": cot_rate,
                              "lang_rate": lang_rate, "n_samples": n5,
                              "details": t5_details}
    if not t5_passed:
        probe_passed = False

    print("\n" + "="*80)
    print(f"PROBE RESULT: {'PROMOTED' if probe_passed else 'REJECTED'}")
    print("="*80 + "\n", flush=True)
    _emit(f"Probe result: {'PROMOTED' if probe_passed else 'REJECTED'}",
          status="promoted" if probe_passed else "rejected")

    return probe_passed, probe_results


# --------------------------------------------------------------------------- #
# Cycle
# --------------------------------------------------------------------------- #

def _dump_training_debug(examples: list, texts: list, *, family: str,
                         response_part: str, run_id: Optional[str]) -> None:
    """Dump the exact rows handed to Unsloth into ``<repo_root>/debug/train-<tag>/``.

    ``texts[i]`` is the literal source string we pre-tokenized for ``examples[i]`` — so
    this is *exactly what the cycle trains on*, the channel-rendered reasoning included
    (on gemma, an empty ``<|channel>…<channel|>`` is visible here as the no-thought form).
    ``train_on_responses_only`` unmasks every assistant turn, but ``_keep_final_turn_only``
    then re-masks all but the final one — so only ``trained_span`` (the last post-marker
    segment) actually contributes to the loss; ``untrained_spans`` are the earlier
    (CoT-less, channel-stripped) history turns, masked out. ``final_span_has_channel``
    flags the regression this fix targets: on gemma the trained turn must carry a
    ``<|channel>`` reasoning block, or the cycle is training an empty/CoT-less target.

    Best-effort: a dump failure must never abort a real training run — it is caught and
    logged, the cycle proceeds.
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]      # server/training -> repo root
        tag = run_id or f"{datetime.now():%Y%m%d-%H%M%S}"
        out_dir = repo_root / "debug" / f"train-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        n_no_channel = 0
        for i, (ex, text) in enumerate(zip(examples, texts)):
            spans = text.split(response_part)[1:]            # all assistant-turn segments
            trained = spans[-1] if spans else None           # only the final turn is trained
            untrained = spans[:-1] if spans else []          # earlier history turns, masked
            has_channel = bool(trained) and (
                "<|channel>" in trained if family == "gemma" else "analysis<|message|>" in trained)
            if family in ("gemma", "harmony") and trained and not has_channel:
                n_no_channel += 1
            rows.append({
                "index": i,
                "kind": "fact/persona" if ex.get("fact_key") else "dialogue",
                "anchor_key": ex.get("anchor_key"),
                "target": ex.get("variant"),
                "has_response_marker": bool(spans),
                "n_assistant_turns": len(spans),
                "final_span_has_channel": has_channel,
                "trained_span": trained,
                "untrained_spans": untrained,
                "rendered_text": text,
            })
        if n_no_channel:
            print(f"debug: WARNING — {n_no_channel}/{len(rows)} gemma examples have a "
                  f"trained final turn with NO <|channel> reasoning block (would erode "
                  f"CoT).", flush=True)
        (out_dir / "examples.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps({
            "run_id": run_id,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "family": family,
            "response_marker": response_part,
            "n_examples": len(texts),
            "n_dialogue": sum(1 for ex in examples if not ex.get("fact_key")),
            "n_fact_persona": sum(1 for ex in examples if ex.get("fact_key")),
            "note": ("rendered_text is the exact string pre-tokenized for SFTTrainer. "
                     "train_on_responses_only unmasks every assistant turn, then "
                     "_keep_final_turn_only re-masks all but the last — so only "
                     "'trained_span' contributes to the loss; 'untrained_spans' are the "
                     "masked CoT-less history turns. On gemma the trained turn must carry "
                     "a <|channel> block (see final_span_has_channel)."),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"debug: dumped {len(texts)} training rows -> {out_dir}", flush=True)
    except Exception as e:        # never let debug I/O kill a training run
        print(f"debug: training-row dump skipped ({e})", flush=True)


def _adapter_dir_for(run_id: Optional[str]) -> Path:
    """The ``models/`` dir for a promoted adapter, named after the reflection ``run_id``
    that produced it — so ``models/adapter-<run_id>/`` shares one identifier with
    ``reflections/<run_id>/`` and the ``builds.jsonl`` line (the reflection→train event is
    one traceable thing, not three timestamps). Falls back to a save-time stamp for a
    standalone build with no reflection (a manual ``train_cycle``). If the name is already
    taken (for example, the same run trained twice) a
    short time suffix disambiguates, so a lineage member is never silently overwritten.
    """
    stem = f"adapter-{run_id}" if run_id else f"adapter-{datetime.now():%Y%m%d-%H%M%S}"
    cand = _MODELS_DIR / stem
    if not cand.exists():
        return cand
    return _MODELS_DIR / f"{stem}-{datetime.now():%H%M%S}"


def _ensure_expandable_segments() -> None:
    """Verify the CUDA allocator ACTUALLY runs in expandable-segments mode, and say
    so in the log either way — because the env-var route fails silently and the
    failure is indistinguishable from fragmentation until a long run dies of it.

    The probe itself (allocate → read is_expandable off the snapshot → force via
    `_set_allocator_settings` if inactive → re-probe → print the verdict + inherited
    env) now lives in ONE place, `core/alloc_guard.py` — the inference server runs
    the same check at boot, having OOM'd the same way on the live facts fetch. The
    learned-the-hard-way history (training box, 2026-08-26: two 8192-cap builds dead
    mid-run with 6.4–6.9 GiB stranded in split segments, the every-50-steps
    empty_cache() reclaim NOT helping) is recorded in that module's docstring and
    the changelog. Best-effort — the probe must never take the cycle down; called
    while nothing model-sized is allocated yet, which is what makes the runtime
    force meaningful (it applies only to segments created afterwards)."""
    try:
        from core.alloc_guard import ensure_expandable_segments  # inference/ on sys.path
    except Exception as import_err:
        print(f"[alloc] allocator probe unavailable: {import_err}", flush=True)
        return
    ensure_expandable_segments()


def run_cycle(*, dry_run: bool = False, lora_r: Optional[int] = None, epochs: int = 1,
              learning_rate: Optional[float] = None, use_staging: bool = False,
              model_id_override: Optional[str] = None,
              run_id: Optional[str] = None, skip_validation: bool = False,
              seed: int = 42, include_fresh: bool = False) -> dict:
    # Structured progress log (server/train_progress.jsonl). Reset per cycle so it
    # always describes the current run. The watchdog serves it to the Sleep tab,
    # which can't watch training over the (down) inference WebSocket. Best-effort.
    progress = TrainProgress(run_id=run_id, reset=True)
    # Unified activity journal, from THIS process. The cycle runs with the inference
    # server stopped (the GPU must be free), so nothing else can report for it and the
    # journal would otherwise have a hole exactly where the build was. Two sinks, because
    # they answer different questions:
    #   * activity_log  — durable history, in the one stream everything else writes to, so
    #     after the relaunch you can scroll back through the build like any other work;
    #   * progress      — LIVE, because it is the file the watchdog serves
    #     (GET /job/progress) and the watchdog is the only server alive during training.
    # The tee is what puts *unsloth's own* output (loss curves, warnings, tracebacks) into
    # both without touching a line of library code.
    try:
        from core import activity_log as _activity_log
        _activity_log.configure(activity_log_path(),
                                **(load_server_config().get("logging") or {}))
        _activity_log.install_stdout_tee(
            "training", extra=lambda ln: progress.emit("log", ln))
    except Exception as _log_err:  # telemetry must never take the cycle down
        print(f"[train] activity journal unavailable: {_log_err}", flush=True)
    config = load_server_config()
    # Base LR: an explicit caller override (CLI --lr / Sleep train_params.lr) wins; otherwise
    # read it from server_config.json (`train_lr`, default _TRAIN_LR_DEFAULT). The per-row
    # multipliers + LR-schedule shape below scale this reference value.
    if learning_rate is None:
        learning_rate = float(config.get("train_lr", _TRAIN_LR_DEFAULT))
    # LoRA rank: an explicit caller override (CLI --lora-r / Sleep train_params.lora_r) wins;
    # otherwise read it from server_config.json (`lora_r`, default _TRAIN_LORA_R_DEFAULT).
    if lora_r is None:
        lora_r = int(config.get("lora_r", _TRAIN_LORA_R_DEFAULT))
    lora_r = max(1, int(lora_r))
    progress.emit("start", "Train cycle starting...", run_id=run_id,
                  lora_r=lora_r, epochs=epochs, lr=learning_rate, dry_run=dry_run)
    if not dry_run:
        _ensure_expandable_segments()
    # Reproducibility spine (REBUILD §7): seed the RNGs so a build is a function of
    # (base, corpus, config, seed). Best-effort — GPU kernels aren't bit-exact, but this
    # pins sampling/init. The seed is recorded on the build line + forensic snapshot.
    try:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed)
    except Exception:
        pass

    # Validation is globally disabled for now (see _VALIDATION_ENABLED) — a separate design
    # project. This is the single chokepoint: force-skip regardless of the caller's flag, the
    # Sleep "Skip validation" checkbox, or any judge-override force-probe path.
    if not _VALIDATION_ENABLED:
        if not skip_validation:
            print("validation is DISABLED (deferred design project — see DESIGN.md); "
                  "promoting unguarded, forcing skip_validation.", flush=True)
        skip_validation = True

    # --model-id lets a small model (e.g. Qwen3-4B) validate the cycle end-to-end
    # on hardware that can't fit the configured base. server_config.json is untouched.
    model_id = (model_id_override or "").strip() or config.get("model_id", "")
    if model_id_override:
        print(f"model-id override: training against {model_id} "
              f"(server_config.json model_id left unchanged)", flush=True)
    context_length = int(config.get("context_length", 32768))
    # Training sequence cap (server_config.json `train_max_seq_length`; default 4096).
    # Bounds the fused CE-loss logit buffer and drives answer-preserving truncation. Never
    # exceeds context_length. See _TRAIN_MAX_SEQ_LENGTH for the VRAM tradeoff.
    train_max_seq_length = min(
        int(config.get("train_max_seq_length", _TRAIN_MAX_SEQ_LENGTH)), context_length)
    # LR schedule (server_config.json `train_lr_schedule`; default "age_ramp" — flat, one
    # pass, caller's --epochs honoured). See _LR_SCHEDULES. The alternative trapezoid is a
    # warmup epoch + `train_plateau_epochs` hold epochs + a decay epoch, so THAT one forces
    # epochs = plateau + 2 regardless of the caller's --epochs.
    lr_schedule = str(config.get("train_lr_schedule", _LR_SCHEDULE_DEFAULT)).strip().lower()
    if lr_schedule not in _LR_SCHEDULES:
        print(f"unknown train_lr_schedule {lr_schedule!r}; using {_LR_SCHEDULE_DEFAULT!r} "
              f"(valid: {', '.join(_LR_SCHEDULES)}).", flush=True)
        lr_schedule = _LR_SCHEDULE_DEFAULT
    # Plateau (full-LR hold) epoch count for the trapezoid (server_config.json
    # `train_plateau_epochs`; default 3). Clamped to >=0: 0 means a plateau-free
    # warmup+decay pass (2 epochs) where every row still sees exactly 1 multiplier's
    # worth of LR (warmup fraction f + decay fraction 1-f), just smoothed and order-neutral.
    plateau_epochs = max(0, int(config.get("train_plateau_epochs",
                                           _TRAIN_PLATEAU_EPOCHS_DEFAULT)))
    if lr_schedule == "triangular":
        forced = plateau_epochs + 2
        if epochs != forced:
            print(f"triangular LR schedule needs {forced} epochs (1 warmup + "
                  f"{plateau_epochs} plateau + 1 decay); overriding epochs {epochs} "
                  f"-> {forced}.", flush=True)
            epochs = forced
    ccfg = ConsolidationConfig.from_dict(config.get("consolidation"))
    if not model_id:
        progress.emit("done", "No model_id in server_config.json", status="error")
        raise SystemExit("no model_id in server_config.json")

    chats_dir = staging_chats_dir() if use_staging else hot_chats_dir()
    cons_dir = staging_consolidation_dir() if use_staging else consolidation_dir()
    scratch = scratch_dir()
    fallback = hot_chats_dir() if use_staging else None
    sidecar = ChatSidecar(chats_dir, fallback_chats_dir=fallback)
    ledger = ConsolidationLedger(cons_dir)
    build_history = BuildHistory(_MODELS_DIR)
    # The build's as_of timestamp: every row's wall-clock age is (built_at − chat_ts), and
    # the same value is recorded on the build history so the build is reproducible (§7).
    built_at = datetime.now().isoformat()

    # REBUILD.md §5a — from-scratch build corpus. Every FROZEN bundle (a reflect-once
    # chat = transcript + sidecar) contributes one row per revisable exchange: no
    # decay-count variant copies, no IDEAL regularizer. hot + archive both contribute
    # (the split no longer carries training semantics). Rows are chronological oldest-
    # first, each stamped with its WALL-CLOCK age (hours since the chat, vs built_at) and
    # the resulting LR multiplier (continuous ramp through the wall_clock lr_ramp 1->2->4,
    # 0 in the RAG-only youngest window). Repetition is replaced by that per-row LR ramp.
    # Persona/fact CoT injection is unchanged, sourced per host exchange (locality). Wander
    # stays one-shot (a per-run decision).
    # Single flat chats dir (archive retired — it was never produced, and feeding a
    # nonexistent dir here would make ChatSidecar recreate a phantom archive/chats).
    chats_dirs = [chats_dir] if use_staging else [hot_chats_dir()]
    wander_pending = load_wander_pending()
    print(f"assembling from-scratch build corpus from {len(chats_dirs)} chat dir(s) "
          f"(+ {len(wander_pending)} pending wander) ...", flush=True)
    progress.emit("render", "Assembling from-scratch build corpus (chronological, "
                            "age-keyed LR ramp)...")
    rows = build_dataset(chats_dirs=chats_dirs, ledger=ledger, ccfg=ccfg,
                         built_at=built_at, wander_pending=wander_pending,
                         fallback_chats_dir=fallback, include_fresh=include_fresh)
    # Preview rows (include_fresh): chats too young for the LR ramp / only background-
    # frozen, emitted at multiplier 0 so their derived targets reach the snapshot render
    # (and thus the Training review tab) WITHOUT training. Split them out here — nothing
    # downstream (fingerprint, tokenization, scheduler, probe anchors) may see them; they
    # rejoin only at the two render writes below.
    preview_rows = [r for r in rows if r.preview]
    rows = [r for r in rows if not r.preview]
    if not rows:
        _fresh_note = (f" ({len(preview_rows)} fresh preview row(s) exist but cannot "
                       f"train)" if preview_rows else "")
        print(f"no frozen bundles / wander to build from — nothing to consolidate."
              f"{_fresh_note}")
        progress.emit("done", f"No frozen bundles or wander to train — nothing to build."
                              f"{_fresh_note}",
                      status="rejected")
        return {"examples": 0, "rows": 0, "preview_rows": len(preview_rows)}

    # Flat chronological corpus rows. Row identity + multiplier stay on each example through
    # tokenization/masking; only the final retained dataset becomes the optimizer schedule.
    # source_session/exchange_index travel on each row so a training render maps back to
    # the origin exchange — the Training review tab reads them to flag a corrupt CoT/reply
    # on the source chat (see core.session_ops.handle_mark_corrupt).
    # Row → render-dict projection is the shared training.build_dataset.row_render_dict
    # (one definition with the GPU-free preview snapshot, so the schema can't drift).
    examples = [row_render_dict(r) for r in rows]
    # Preview rows trail the trained corpus in the render: file order == training order
    # for every trained row, and fresh chats are the newest, so trailing is also
    # (approximately) chronological. They are re-appended at the final rewrite too.
    preview_examples = [row_render_dict(r) for r in preview_rows]
    corpus_multipliers = [r.lr_multiplier for r in rows]
    n_chat = sum(1 for r in rows if r.source == "chat")
    n_wander = sum(1 for r in rows if r.source == "wander")
    n_unmask = sum(1 for r in rows if r.unmask_user)
    fingerprint = corpus_fingerprint(rows)
    # Probe Tier-4 retention samples: the chat rows' dialogue anchors (persona/fact injection
    # already folded into each anchor's target). Exclude only the SPLIT contamination copy
    # (unmask_user with no weight) so a cap exchange contributes its anchor once, not twice
    # (§5e); a FOLDED cap row (unmask_user + user_loss_weight) IS the exchange's only row, so
    # it is kept.
    retention_anchors = [r.anchor for r in rows
                         if r.source == "chat" and r.anchor
                         and (not r.unmask_user or r.user_loss_weight is not None)]

    scratch.mkdir(parents=True, exist_ok=True)
    render_path = scratch / SFT_RENDER_FILE
    with open(render_path, "w", encoding="utf-8") as fh:
        for ex in examples + preview_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    ages = [r.age for r in rows]
    _fresh_tag = f" + {len(preview_examples)} fresh preview" if preview_examples else ""
    print(f"built {len(rows)} rows ({n_chat} chat + {n_wander} wander; {n_unmask} cap-age "
          f"contamination{_fresh_tag}); ages {min(ages)}..{max(ages)}; multipliers "
          f"{sorted(set(corpus_multipliers))}; lr_schedule={lr_schedule} ({epochs} epoch(s)); "
          f"corpus {fingerprint} -> {render_path}", flush=True)
    progress.emit("render", f"Built {len(rows)} rows ({n_chat} chat + {n_wander} wander; "
                            f"{n_unmask} contamination{_fresh_tag}); multipliers "
                            f"{sorted(set(corpus_multipliers))}; "
                            f"lr_schedule={lr_schedule} ({epochs} epoch(s)).",
                  rows=len(rows), chat_rows=n_chat, wander_rows=n_wander,
                  unmask_rows=n_unmask, preview_rows=len(preview_examples),
                  corpus_fingerprint=fingerprint,
                  lr_schedule=lr_schedule, epochs=epochs)

    if dry_run:
        print("dry-run: skipping train/probe/build; render kept for inspection.")
        progress.emit("done", "Dry-run: corpus assembled, no training.", status="info")
        return {"examples": len(examples), "rows": len(rows), "chat_rows": n_chat,
                "wander_rows": n_wander, "preview_rows": len(preview_examples),
                "corpus_fingerprint": fingerprint}

    family = _model_family(model_id)
    # REBUILD.md §5b — from-scratch build: NEVER resume a prior adapter. Each build loads
    # the FROZEN base by model_id and fits a fresh LoRA (get_peft_model below), so a bad
    # build is discarded and never inherited (the collapse failure mode is unrepresentable).
    # The previously-promoted adapter is untouched on disk and keeps serving until this
    # build promotes. `--lora-r` therefore always takes effect now.
    # Belt-and-suspenders for the recompile fix above: UNSLOTH_COMPILE_DISABLE stops
    # unsloth from *regenerating* the torch.compile-wrapped forwards, but a cache written
    # by an earlier (pre-fix) run still holds the crash-prone compiled module on disk and
    # unsloth may reuse it. The cache is fully regenerable, so wipe it before load to force
    # a clean, wrapper-free regen. Checks the CWD (server/, where unsloth writes it) and
    # the inference package root — same two locations inference_backend purges.
    _purge_unsloth_compile_cache()
    from core.inference_backend import _wants_fast_model
    from core import unified_memory, fast_load, moe_bnb_experts
    # Same loader-class rule as inference_backend.load: FastModel for a
    # *ForConditionalGeneration checkpoint (gemma-4, Qwen3.5 incl. its text-only MoE
    # sizes), FastLanguageModel for a plain causal LM (qwen3, gpt-oss).
    if family == "gemma" or _wants_fast_model(model_id):
        from unsloth import FastModel as _Fast
    else:
        from unsloth import FastLanguageModel as _Fast
    # Multilingual embedder for the probe (Tiers 3/4). all-MiniLM-L6-v2 is English-centric,
    # so its cosine similarities degrade for a Russian/Chinese-shifting Ava — penalizing
    # legitimate non-English adaptation as character drift / poor retention. The
    # multilingual MiniLM places cross-/non-English text comparably. This is the probe's
    # own embedder only; RAG's index embedder (rag_engine._EMBED_MODEL) is unchanged —
    # swapping that needs a reindex, out of scope. Loaded only when the probe runs;
    # persona injection does its dedup lexically (model-free), so it needs no embedder.
    embedder = None
    if not skip_validation:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    import os
    print(f"loading frozen base {model_id} (family={family}, ctx={context_length}) "
          f"for a from-scratch build ...", flush=True)
    progress.emit("load", f"Loading frozen base {model_id} (ctx={context_length})...")
    # DGX Spark / unified memory: pin placement to device 0 (accelerate would plan an
    # offload from a MemFree figure that excludes the page cache) and clone tensors out
    # of the safetensors mmap before the device copy (0.16 GB/s otherwise). Both are
    # no-ops on a discrete-GPU box. A converted per-expert bnb Qwen3.5-MoE checkpoint
    # additionally needs its experts class swapped in before the model is built.
    _placement_kw = unified_memory.load_kwargs()
    _fast = fast_load.install()
    _moe_swap = moe_bnb_experts.install_if_needed(model_id)
    print(f"[load] {unified_memory.describe()} placement={_placement_kw or 'default'} "
          f"fast_load={_fast} perexpert_moe={_moe_swap}", flush=True)
    with fast_load.hf_offline_if_cached(model_id):
        model, tokenizer = _Fast.from_pretrained(
            model_name=model_id, max_seq_length=context_length, load_in_4bit=True,
            **_placement_kw,
        )
    _Fast.for_inference(model)

    # -- Collect probe baselines (Tiers 1 & 3) — only when validating --
    # The probe compares pre- vs post-train, so its baselines must be captured here, before
    # training. When validation is skipped (a UI-initiated cycle that promotes without the
    # probe) the baselines are pure cost, so skip them too: they are the probe's, and only
    # the probe consumes them.
    continuity_prompts: list = []
    baselines: list = []
    baseline_vecs = None
    capability_baselines: list = []
    if not skip_validation:
        continuity_prompts = _collect_character_prompts(chats_dir, sidecar, count=10)
        print(f"Generating character continuity baselines on {len(continuity_prompts)} prompts...", flush=True)
        progress.emit("baseline",
                      f"Generating character-continuity baselines on {len(continuity_prompts)} prompts...")
        for cp in continuity_prompts:
            messages = render.build_messages(cp, "")[:-1]
            try:
                resp = _eval_generate(model, tokenizer, context_length, messages,
                                      max_new_tokens=_PROBE_DIALOGUE_MAX_NEW_TOKENS)
            except Exception as e:
                print(f"Warning: failed to generate character continuity baseline: {e}", flush=True)
                resp = ""
            baselines.append(resp)

        baseline_answers = [answer_of(r) for r in baselines]
        baseline_vecs = embedder.encode(baseline_answers, convert_to_numpy=True, normalize_embeddings=True)

        # Tier-1 latent-capability baselines on the pre-train model: the logprob the weights
        # currently assign each canonical answer. Tier 1 re-measures after training and gates
        # on the drop (a capability cliff), so this must be captured here, pre-train.
        print("Measuring Tier-1 capability baselines (latent logprobs)...", flush=True)
        progress.emit("baseline", "Measuring Tier-1 capability baselines (latent logprobs)...")
        capability_baselines = measure_capability_logprobs(model, tokenizer, context_length)

    # Persona examples were injected + rendered model-free above (before load), so there
    # is no GPU regeneration step here — the whole persona path is dialogue-shaped.

    # -- train LoRA --------------------------------------------------------- #
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    # trainer.train() is the longest silent stretch of the cycle. Forward each
    # HF logging step into the structured progress log so the Sleep tab shows a
    # live pulse (step/total · loss · epoch) instead of dead air between the
    # "Training..." and "complete" lines. Throttled to ~2s so a high step count
    # (logging_steps=1) doesn't flood the panel; the final step always emits.
    class _ProgressCallback(TrainerCallback):
        def __init__(self, emit_min_interval_s: float = 2.0) -> None:
            self._last = 0.0
            self._min = emit_min_interval_s

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or progress is None:
                return
            total = state.max_steps or 0
            is_last = total and state.global_step >= total
            now = time.monotonic()
            if not is_last and (now - self._last) < self._min:
                return
            self._last = now
            step_str = f"step {state.global_step}" + (f"/{total}" if total else "")
            progress.emit(
                "train",
                f"{step_str} · loss {logs['loss']:.3f} · epoch {state.epoch:.2f}",
                step=state.global_step, total=total,
                loss=logs["loss"], epoch=state.epoch,
            )

    # Periodically hand the caching allocator's free blocks back to the driver.
    # Long runs over variable-length rows strand freed-but-cached memory that the
    # failing allocation can't use — blocks pinned to the offloaded-checkpointing
    # side stream, pool splits — and expandable segments bound fragmentation WITHIN
    # a segment but never unmap pages mid-run, so the stranding only grows: the
    # 2026-08-26 training-box build died at step 284/1832 on a 508 MiB request with
    # 6.40 GiB reserved-but-unallocated (the July OOMs stranded ~2 GiB; this run
    # crept to 6.4 over ~29 min of 1.1k–8k-token rows). empty_cache() releases
    # every cached free block (with expandable segments, unmaps their pages) and
    # touches nothing in use, so it is always safe — the cost is one device sync
    # + cache re-warm, milliseconds against a ~20 s/it step, paid every N steps.
    class _CacheReclaimCallback(TrainerCallback):
        def __init__(self, every_n_steps: int = 50) -> None:
            self._every = max(1, every_n_steps)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step and state.global_step % self._every == 0:
                import torch
                torch.cuda.empty_cache()

    # REBUILD.md §5b — every build fits a FRESH LoRA on the frozen base (never resumed),
    # so `--lora-r` always takes effect and no build inherits a prior trajectory's dose.
    _gamma = _TRAIN_LORA_ALPHA / math.sqrt(lora_r)
    print(f"Initializing a fresh LoRA adapter (from-scratch, r={lora_r}, "
          f"alpha={_TRAIN_LORA_ALPHA}, rslora scaling={_gamma:.4f})...", flush=True)
    model = _Fast.get_peft_model(
        model, r=lora_r, lora_alpha=_TRAIN_LORA_ALPHA,
        # Rank-stabilized scaling: gamma = alpha/sqrt(r), NOT peft's default alpha/r.
        # With the old `lora_alpha == r` gamma was pinned at 1.0, so raising the rank
        # raised the effective LR on the weights (dW ~= gamma * dB @ A sums over r
        # terms, and Adam moves each entry of B by ~lr regardless of rank, so ||dW||
        # grew like sqrt(r)) — r=32 came out LESS stable than r=16 for that reason.
        # alpha is a fixed constant here, calibrated so gamma == 1.0 at r=16, which
        # reproduces the pre-switch scaling exactly. See decay.TRAIN_LORA_ALPHA.
        use_rslora=True,
        # gemma-4 is multimodal and FastModel.get_peft_model defaults
        # finetune_vision_layers=True; the vision tower also has q/v_proj-named
        # modules, so an explicit target_modules list would otherwise attach LoRA to
        # them. We only consolidate dialogue (text), so keep vision frozen.
        finetune_vision_layers=False,
        target_modules=_lora_target_modules(model),
        use_gradient_checkpointing="unsloth",
    )
    # for_inference (above, for the Tier-3 baseline generations) left the model in
    # inference mode with LoRA grads disabled. Flip it back before training, or
    # trainer.train() no-ops/errors.
    if hasattr(_Fast, "for_training"):
        # for_inference set _flag_for_generation on the base model; after get_peft_model
        # wrapped it, for_training walks the PeftModel and does `del m._flag_for_generation`.
        # hasattr() delegates True through the wrapper but del hits the wrapper's own dict
        # (which lacks the attr) → AttributeError. Strip the flag from every level's __dict__
        # first (a direct pop bypasses delegation) so unsloth's guarded del is a no-op.
        _m = model
        while _m is not None:
            vars(_m).pop("_flag_for_generation", None)
            _m = getattr(_m, "model", None)
        _Fast.for_training(model)
    # Reclaim the baseline stage's leftover KV-cache/activation fragmentation before the
    # memory-heavy training step allocates its fused CE-loss logit buffer.
    _free_gpu_memory()
    # Prepare the final, pre-tokenized dataset ourselves. Optional history is trimmed first;
    # an irreducible row that still exceeds the cap is quarantined with provenance. Every
    # retained row has already proven that its response marker, complete target, reasoning
    # close, turn terminator, and EOS survive inside the cap, so TRL's head-slice is a no-op.
    corpus_row_count = len(examples)
    prepared, quarantined_rows, _turns_dropped = _prepare_training_rows(
        examples, tokenizer, family, train_max_seq_length)
    quarantine_path = scratch / SFT_QUARANTINE_FILE
    if not prepared:
        _write_training_quarantine(quarantine_path, quarantined_rows)
        raise RuntimeError(
            f"training preparation quarantined all {corpus_row_count} rows; "
            f"inspect {quarantine_path}")
    _write_training_quarantine(quarantine_path, quarantined_rows)
    _initial_quarantine = len(quarantined_rows)
    _n_trimmed = sum(1 for row in prepared
                     if len(row["example"].get("messages") or []) <
                        len(examples[row["train_row_id"]].get("messages") or []))
    print(f"training preparation: retained {len(prepared)}/{corpus_row_count} rows; "
          f"quarantined {_initial_quarantine}; trimmed {_n_trimmed} row(s) "
          f"({_turns_dropped} oldest history turn(s)); cap={train_max_seq_length}.",
          flush=True)
    if quarantined_rows:
        _reason_counts = {}
        for record in quarantined_rows:
            reason = record.get("reason", "unknown")
            _reason_counts[reason] = _reason_counts.get(reason, 0) + 1
        print(f"training quarantine: {_reason_counts} -> {quarantine_path}", flush=True)
    progress.emit(
        "render",
        f"Prepared {len(prepared)}/{corpus_row_count} complete rows; "
        f"quarantined {_initial_quarantine} over-cap/incomplete row(s).",
        corpus_rows=corpus_row_count, prepared_rows=len(prepared),
        quarantined_rows=_initial_quarantine, quarantine_path=str(quarantine_path),
        turns_dropped=_turns_dropped, train_max_seq_length=train_max_seq_length,
    )

    instruction_part, response_part = _MARKERS[family]
    # Folded contamination rows carry a per-token loss weight (contamination_fold). The column
    # is added only when at least one row is folded — a non-folding cycle omits it entirely, so
    # the dataset schema (and the whole collator/loss path) is byte-for-byte unchanged.
    _fold_active = any(row.get("user_loss_weight") is not None for row in prepared)
    _dataset_cols = {
        # Presence of input_ids tells SFTTrainer this dataset is already processed, so it
        # skips its text/EOS tokenization pass. max_seq_length remains as a final no-op guard.
        "input_ids": [row["input_ids"] for row in prepared],
        # Keep row identity and optimizer metadata attached through response masking. They
        # are validated/extracted only after train_on_responses_only performs any filtering.
        "train_row_id": [row["train_row_id"] for row in prepared],
        "lr_multiplier": [row["lr_multiplier"] for row in prepared],
        "unmask_user": [row["unmask_user"] for row in prepared],
    }
    if _fold_active:
        _dataset_cols["user_loss_weight"] = [row.get("user_loss_weight") for row in prepared]
    dataset = Dataset.from_dict(_dataset_cols)

    # REBUILD.md §5c — chronological consumption with a PER-STEP LR keyed to each row's age.
    # Order preservation + per-step LR are what let the age ramp (repetition's replacement)
    # actually reach the optimizer. Two Trainer hooks (subclassed; verified on Gemma4-31B):
    #   * _get_train_sampler -> SequentialSampler, so rows are consumed oldest-first (TRL
    #     0.24 doesn't override it, packing is off, so this alone preserves order; the same
    #     order repeats each epoch, which the triangular schedule below relies on);
    #   * create_scheduler -> a LambdaLR whose multiplier keys on the optimizer-step index.
    #     With batch=1 + GA=1, optimizer step i == row (i % N), so multipliers[i % N] is that
    #     row's per-row factor (age 0->3, 1->5, 2+->6; wander fixed; contamination dose). The
    #     `lr_schedule` (server_config.json) picks the GLOBAL shape layered on top: "age_ramp"
    #     is a flat single pass (multiplier only); "triangular" (default) multiplies in a
    #     0->max->...->max->0 trapezoid (1 warmup + `plateau_epochs` hold + 1 decay epoch) so
    #     every row's passes sum to (plateau+1) full multipliers' worth of LR regardless of
    #     position. Hits the actual param-group LR (NOT a per-example
    #     loss scale — adamw's second moment would normalize that away).
    from torch.utils.data import SequentialSampler
    from torch.optim.lr_scheduler import LambdaLR

    # Filled from trainer.train_dataset only AFTER response masking/filtering. The closure
    # reads these cells when trainer.train() creates the scheduler.
    schedule_multipliers: list[float] = []
    _n_rows = 0
    # Memoized (backbone, lm_head) for the folded-contamination weighted loss; resolved once
    # at setup below when folding is active (validated before train() so a bad model layout
    # fails LOUDLY, not 6h in). [None, None] means the weighted path is unused this cycle.
    _fold_lm = [None, None]

    class _OrderedSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            ds = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(ds)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Only folded contamination rows carry `loss_weight`; every other row (all normal
            # + wander rows, and every row when contamination_fold is off) takes the untouched
            # fused-CE path, so the standard corpus is byte-for-byte unchanged.
            if not isinstance(inputs, dict) or inputs.get("loss_weight") is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            import torch
            inputs = dict(inputs)
            loss_weight = inputs.pop("loss_weight")
            labels = inputs.pop("labels")
            backbone, lm_head = _fold_lm[0], _fold_lm[1]
            if backbone is None or lm_head is None:
                # Setup validation should have caught this; fail rather than mistrain.
                raise RuntimeError(
                    "contamination_fold: backbone/lm_head unresolved at compute_loss")
            # Backbone forward -> hidden states, WITHOUT the full-sequence logit buffer.
            bb_kwargs = {"input_ids": inputs.get("input_ids"), "use_cache": False}
            if inputs.get("attention_mask") is not None:
                bb_kwargs["attention_mask"] = inputs["attention_mask"]
            if inputs.get("position_ids") is not None:
                bb_kwargs["position_ids"] = inputs["position_ids"]
            bb_out = backbone(**bb_kwargs)
            hidden = getattr(bb_out, "last_hidden_state", None)
            if hidden is None:
                hidden = bb_out[0]
            # Token t is predicted by hidden at t-1 (causal shift).
            shift_hidden = hidden[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_w = loss_weight[:, 1:].to(shift_hidden.dtype)
            keep = shift_labels != -100
            sel_hidden = shift_hidden[keep]        # (K, H) — K = unmasked positions (small)
            sel_labels = shift_labels[keep]
            sel_w = shift_w[keep]
            if sel_hidden.numel() == 0:
                loss = hidden.sum() * 0.0          # keep graph; nothing trains this row
                return (loss, bb_out) if return_outputs else loss
            # Unembed ONLY the unmasked positions, chunked so the fp32 logit slab
            # (K × vocab × 4B) stays bounded on the tight 31B budget. Weighted-mean CE.
            denom = sel_w.sum().clamp_min(1.0)
            parts = []
            for i in range(0, sel_hidden.size(0), _FOLD_LOSS_CHUNK):
                lg = lm_head(sel_hidden[i:i + _FOLD_LOSS_CHUNK]).float()
                ce = torch.nn.functional.cross_entropy(
                    lg, sel_labels[i:i + _FOLD_LOSS_CHUNK], reduction="none")
                parts.append((ce * sel_w[i:i + _FOLD_LOSS_CHUNK]).sum())
            loss = torch.stack(parts).sum() / denom
            return (loss, bb_out) if return_outputs else loss

        def create_scheduler(self, num_training_steps, optimizer=None):
            opt = optimizer if optimizer is not None else self.optimizer

            def lr_lambda(step):
                # step i == chronological row (i % N) (batch=1, GA=1). Modulo maps each epoch's
                # repeated pass back to its row's per-row multiplier.
                if not schedule_multipliers:
                    return 1.0
                row_mult = float(schedule_multipliers[step % _n_rows])
                if lr_schedule == "triangular":
                    # Layer the trapezoid on top: warmup 0->max (epoch 1), hold max across the
                    # `plateau_epochs` middle epochs, decay max->0 (final epoch).
                    # Order-independent average per row (§ helper).
                    return row_mult * _triangular_lr_fraction(
                        step, _n_rows, plateau_epochs)
                return row_mult
            self.lr_scheduler = LambdaLR(opt, lr_lambda)
            return self.lr_scheduler

    # Hand SFTTrainer the underlying tokenizer for response-marker masking. The dataset is
    # already tokenized above, so SFT's normal text tokenization/EOS pass is bypassed.
    trainer = _OrderedSFTTrainer(
        model=model, tokenizer=getattr(tokenizer, "tokenizer", tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            # batch=1 + GA=1 -> optimizer step i == chronological row i, so the per-step
            # LambdaLR multiplier is unambiguous (REBUILD.md §5c simplest exact option).
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            num_train_epochs=epochs, learning_rate=learning_rate,
            # Cap training length below the inference context to bound the fused CE-loss
            # logit buffer (see _TRAIN_MAX_SEQ_LENGTH). Never exceed context_length.
            max_seq_length=train_max_seq_length, logging_steps=1,
            # learning_rate is the BASE LR; the LambdaLR above multiplies it per row by the
            # row multiplier (and, under the triangular schedule, the 3-epoch warmup/hold/decay).
            # constant scheduler / no warmup is moot — create_scheduler is overridden — but
            # kept as the harmless SFTConfig default.
            optim="adamw_8bit", warmup_ratio=0.0, lr_scheduler_type="constant",
            output_dir=str(_MODELS_DIR / "trainer"),
            report_to="none",
            # Preserve train_row_id/lr_multiplier/unmask_user through response masking.
            # Metadata columns are validated then removed before the collator runs.
            remove_unused_columns=False,
        ),
        callbacks=[_ProgressCallback(), _CacheReclaimCallback()],
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part,
    )
    # Build the scheduler/unmask inputs from the FINAL masked dataset. If Unsloth filters a
    # row despite the completion guard, its row id is quarantined and the surviving rows
    # retain their own multipliers instead of shifting onto their neighbours.
    _td = trainer.train_dataset
    if _td is None:
        raise RuntimeError("train_on_responses_only returned no training dataset")
    _required_meta = {"train_row_id", "lr_multiplier", "unmask_user"}
    _missing_meta = _required_meta.difference(_td.column_names)
    if _missing_meta:
        raise RuntimeError(
            "training metadata was stripped during response masking: "
            + ", ".join(sorted(_missing_meta)))
    final_row_ids = list(_td["train_row_id"])
    schedule_multipliers, unmask_col = _aligned_training_metadata(
        prepared, final_row_ids, list(_td["lr_multiplier"]), list(_td["unmask_user"]))
    _prepared_by_id = {row["train_row_id"]: row for row in prepared}
    _final_id_set = set(final_row_ids)
    _mask_filtered = [row for row in prepared if row["train_row_id"] not in _final_id_set]
    for row in _mask_filtered:
        quarantined_rows.append(_quarantine_record(
            row["example"], row["train_row_id"], "response_mask_filtered",
            cap=train_max_seq_length, token_count=len(row["input_ids"]),
            detail="train_on_responses_only produced an all-masked row"))
    if _mask_filtered:
        _write_training_quarantine(quarantine_path, quarantined_rows)
        print(f"response masking filtered {len(_mask_filtered)} prepared row(s); "
              f"scheduler rebuilt from {len(final_row_ids)} survivors -> {quarantine_path}",
              flush=True)
    if not final_row_ids:
        raise RuntimeError(
            f"response masking removed every prepared training row; inspect {quarantine_path}")
    _n_rows = len(final_row_ids)

    prepared = [_prepared_by_id[int(row_id)] for row_id in final_row_ids]
    examples = [row["example"] for row in prepared]
    texts = [row["text"] for row in prepared]
    _any_unmask = any(unmask_col)
    _n_unmask_rows = sum(1 for flag in unmask_col if flag)
    n_chat = sum(1 for ex in examples if ex.get("source") == "chat")
    n_wander = sum(1 for ex in examples if ex.get("source") == "wander")
    n_unmask = _n_unmask_rows

    # The snapshot render is rewritten to the exact retained/trimmed row set. The initial
    # assembly render remains useful for --dry-run, but a real build must never claim that
    # quarantined or mask-filtered rows were trained. Fresh preview rows (include_fresh)
    # are re-appended: they carry `preview: true` + multiplier 0, so they claim nothing —
    # they exist so the Training review tab can repair them before they age into a build.
    with open(render_path, "w", encoding="utf-8") as fh:
        for ex in examples + preview_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    _dump_training_debug(examples, texts, family=family,
                         response_part=response_part, run_id=run_id)
    print(f"user-unmask: final dataset carries {_n_unmask_rows}/{_n_rows} flagged row(s).",
          flush=True)

    # Optimizer metadata has now safely crossed tokenization + response filtering. Remove
    # non-model columns before batching; keep unmask_user only when the collator needs it, and
    # keep user_loss_weight only on a folding cycle (its column is present only then).
    _drop_cols = ["train_row_id", "lr_multiplier"]
    if not _any_unmask:
        _drop_cols.append("unmask_user")
    _fold_col_present = "user_loss_weight" in _td.column_names
    if _fold_col_present and not _fold_active:
        _drop_cols.append("user_loss_weight")
    trainer.train_dataset = _td.remove_columns(_drop_cols)
    # train_on_responses_only unmasks *every* assistant turn. Keep only the final turn's
    # span in the loss — the earlier turns are CoT-less by construction and training them
    # erodes the reasoning channel over cycles (see _keep_final_turn_only / DESIGN.md).
    _tro_collator = trainer.data_collator
    _text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    def _encode_marker(s: str) -> list:
        """Token-ids for a chat-template marker (encoded in-place, special tokens honoured).
        These delimit the final user turn for the blunt unmask. Best-effort — an empty list
        on failure makes the unmask a safe no-op for the cycle."""
        try:
            return list(_text_tok(s, add_special_tokens=False)["input_ids"])
        except Exception:
            return []

    _instruction_ids = _encode_marker(instruction_part)
    _response_ids = _encode_marker(response_part)

    # User-unmask diagnostics: `logged` gates a one-shot check of whether the unmask_user
    # column actually reached the collator (the one thing that can't be verified off-GPU —
    # SFT dataset prep may strip it); `applied` tallies rows that got their user turn unmasked.
    # `weighted` tallies folded contamination rows that produced a per-token loss weight.
    _unmask_diag = {"logged": False, "applied": 0, "weighted": 0, "wlogged": False}

    def _final_turn_collator(features, *args, **kwargs):
        # Pull the unmask flags off before the underlying collator tensorizes — a stray
        # column would break its padding/stacking. Falls back to the unchanged keep-final-turn
        # path whenever the column is absent (so if the column doesn't survive SFT's dataset
        # prep, the unmask silently disables, never crashes).
        unmask_per_row = _unmask_user_from_features(features)
        weight_per_row = _user_loss_weight_from_features(features)
        if not _unmask_diag["logged"]:
            _unmask_diag["logged"] = True
            print("user-unmask: unmask_user column " + (
                "reached the collator — user-unmask ACTIVE." if unmask_per_row is not None
                else "did NOT reach the collator (stripped by SFT dataset prep) — "
                     "user-unmask OFF this cycle."), flush=True)
        if weight_per_row is not None and not _unmask_diag["wlogged"]:
            _unmask_diag["wlogged"] = True
            print("contamination-fold: user_loss_weight column reached the collator — "
                  "folded contamination ACTIVE (weighted compute_loss).", flush=True)
        # Strip the metadata columns before the underlying collator tensorizes them.
        _strip = {"unmask_user", "user_loss_weight"}
        if unmask_per_row is not None or weight_per_row is not None:
            features = [{k: v for k, v in f.items() if k not in _strip} for f in features]
        batch = _tro_collator(features, *args, **kwargs)
        if isinstance(batch, dict) and "labels" in batch:
            # Folded contamination: build the per-token loss-weight tensor from the RAW labels
            # FIRST (before keep-final-turn re-masks), then apply the label policy (which also
            # unmasks the user span so its tokens carry loss). compute_loss reads loss_weight.
            if weight_per_row is not None and any(w is not None for w in weight_per_row) \
                    and "input_ids" in batch:
                lw = _build_loss_weight_tensor(
                    batch["labels"], batch["input_ids"], unmask_per_row, weight_per_row,
                    _instruction_ids, _response_ids)
                if lw is not None:
                    batch["loss_weight"] = lw
                    _unmask_diag["weighted"] += sum(1 for w in weight_per_row if w is not None)
            if unmask_per_row and any(unmask_per_row) and "input_ids" in batch:
                _unmask_diag["applied"] += sum(1 for u in unmask_per_row if u)
                _apply_label_policy(batch["labels"], batch["input_ids"],
                                    unmask_per_row, _instruction_ids, _response_ids)
            else:
                _keep_final_turn_only(batch["labels"])
        return batch

    trainer.data_collator = _final_turn_collator
    # Folded contamination is active this cycle: resolve the backbone + unembedding the
    # weighted compute_loss needs NOW (model is loaded) so a bad model layout fails LOUDLY
    # here instead of at the first cap-age row hours into training. A failure aborts the build
    # with a clear remedy (disable contamination_fold) rather than silently mistraining.
    if _fold_active:
        _bb, _head = _resolve_backbone_and_head(model)
        if _bb is None or _head is None:
            raise RuntimeError(
                "contamination_fold is enabled but the model's backbone + lm_head could not "
                "be resolved for the memory-safe weighted loss (unexpected model layout). "
                "Disable `consolidation.wall_clock.contamination.fold` in server_config.json "
                "to fall back to the two-row split, or extend _resolve_backbone_and_head.")
        _fold_lm[0], _fold_lm[1] = _bb, _head
        print(f"contamination-fold: weighted compute_loss ready "
              f"(backbone={type(_bb).__name__}, lm_head={type(_head).__name__}).", flush=True)

    print(f"training LoRA r={lora_r} on {len(texts)} examples ...", flush=True)
    progress.emit("train", f"Training LoRA r={lora_r} on {len(texts)} examples "
                           f"({epochs} epoch(s))...", examples=len(texts))
    train_output = trainer.train()

    # User-unmask outcome for the cycle: how many training rows actually had their final
    # user turn unmasked into the loss. Pairs with the render-flagged count above — if the
    # render flagged rows but 0 were applied, the column was stripped before the collator
    # (unmask off); a non-zero count confirms the user's words reached the weights this cycle.
    if _any_unmask:
        _applied = _unmask_diag["applied"]
        _msg = (f"user-unmask: unmasked the user's words on {_applied} final-run training "
                f"row(s) this cycle." if _applied else
                "user-unmask: 0 rows applied — column stripped before the collator (OFF).")
        print(_msg, flush=True)
        progress.emit("train", _msg)

    # Folded-contamination outcome: how many cap-age rows ran the weighted compute_loss this
    # cycle (should equal the folded-row count). 0 while _fold_active means the loss_weight
    # column was stripped before the collator — the fold silently degraded, so surface it.
    if _fold_active:
        _wt = _unmask_diag["weighted"]
        _fmsg = (f"contamination-fold: weighted loss applied on {_wt} folded cap-age row(s) "
                 f"this cycle." if _wt else
                 "contamination-fold: 0 rows weighted — loss_weight stripped before the "
                 "collator (fold DEGRADED to unweighted; investigate).")
        print(_fmsg, flush=True)
        progress.emit("train", _fmsg)

    # Mean training loss across all steps (TrainOutput.training_loss) + wall-clock —
    # the headline numbers the Sleep tab shows at the end of a cycle. Guard both: a
    # zero-step or interrupted run can leave training_loss None / metrics absent.
    avg_loss = getattr(train_output, "training_loss", None)
    train_metrics = getattr(train_output, "metrics", None) or {}
    train_runtime = train_metrics.get("train_runtime")
    loss_str = f"{avg_loss:.4f}" if isinstance(avg_loss, (int, float)) else "n/a"
    runtime_str = _fmt_duration(train_runtime) if isinstance(train_runtime, (int, float)) else ""
    print(f"training complete — avg loss {loss_str}"
          + (f", train time {runtime_str}" if runtime_str else ""), flush=True)
    progress.emit("train",
                  f"Training finished — avg loss {loss_str}"
                  + (f" over {runtime_str}" if runtime_str else "")
                  + f" ({trainer.state.global_step} step(s)).",
                  avg_loss=avg_loss, train_runtime=train_runtime,
                  steps=trainer.state.global_step)

    if skip_validation:
        # Validation skipped (UI-initiated cycle): promote this adapter without the
        # regression probe. The probe is a floor against catastrophic single-cycle damage
        # (forgetting / collapse / format breakage), so skipping it trades that safety net
        # for speed and for never false-rejecting a legitimate cycle — the adapter lineage
        # still makes a bad promotion reversible (reload the prior adapter). Use sparingly.
        print("\nskip-validation: regression probe BYPASSED — promoting trained adapter "
              "unconditionally (no capability/format/continuity/retention gate).", flush=True)
        progress.emit("probe", "Validation skipped — promoting adapter without the "
                               "regression probe.", status="info", skipped=True)
        probe_passed, probe_results = True, {"skipped": True, "reason": "skip_validation"}
    else:
        progress.emit("train", "LoRA training complete; running regression probe...")
        # Back to inference mode before probing: training left dropout active, which would
        # add noise to the Tier-1 logprob measurement (and the Tier-3/4 generations) and,
        # worse, do so asymmetrically — the pre-train baselines above were captured under
        # for_inference. Measure post-train in the same mode so the Tier-1 delta is clean.
        _Fast.for_inference(model)

        # -- Run Regression Probe --
        # Tier-4 retention samples the chat rows' dialogue anchors (persona/fact injection
        # already folded into each target), so freshly-consolidated material is checked too.
        # had_prior_adapter=False: a from-scratch build's pre-train model is the BARE base
        # (no adapter loaded), so Tier 3's per-cycle continuity delta has no prior-adapter
        # baseline and is skipped — Tier 4 (retention), Tier 1 (capability) and Tier 5
        # (absolute sampling stability) remain the active gates. Comparing against the
        # currently-serving build (REBUILD.md §7) would need a second load and is deferred.
        probe_passed, probe_results = run_regression_probe(
            model, tokenizer, context_length, retention_anchors, continuity_prompts,
            baselines, baseline_vecs, embedder, capability_baselines, progress=progress,
            had_prior_adapter=False,
        )

    # Persist the full probe_results (untruncated per-sample replies/targets) next to the
    # progress log — the progress events truncate replies to [:600], and otherwise these
    # details vanish when the process exits, so a rejected run can't be inspected after.
    try:
        out = Path(__file__).resolve().parent.parent / "train_probe_results.json"
        out.write_text(json.dumps({"run_id": run_id, "probe_passed": probe_passed,
                                   "probe_results": probe_results},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote probe details to {out}", flush=True)
    except Exception as e:
        print(f"Warning: could not write probe_results json: {e}", flush=True)

    def _record_build(*, outcome: str, adapter_dir, probe_summary: dict) -> str:
        """Write the immutable forensic snapshot (REBUILD §7) and append the build line.

        Runs for BOTH outcomes — a rejected build's snapshot is the investigation packet.
        The snapshot copies the render (before it's deleted below), the wander records, the
        active persona digest, and a meta of config + effective wall-clock params + seed +
        built_at, so the build is reproducible/triageable from data we already have.
        """
        build_id = build_history.new_build_id()
        try:
            from core.reflection_digest import latest_digest
            persona = latest_digest(persona_dir())
        except Exception:
            persona = None
        meta = {
            "build_id": build_id, "outcome": outcome, "built_at": built_at, "seed": seed,
            "base_lr": learning_rate, "run_id": run_id, "model_id": model_id,
            "corpus_fingerprint": fingerprint, "rows": len(examples),
            "corpus_rows": corpus_row_count,
            "quarantined_rows": len(quarantined_rows),
            "preview_rows": len(preview_examples),
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "wall_clock": {
                "rag_only_window_h": ccfg.wall.rag_only_window_h,
                "lora_cap_age_h": ccfg.wall.lora_cap_age_h,
                "rag_cap_age_h": ccfg.wall.rag_cap_age_h,
                "contamination_dose": ccfg.wall.contamination_dose,
                "contamination_additive": ccfg.wall.contamination_additive,
            },
            "probe_summary": probe_summary,
            "server_config": config,
        }
        snap = write_snapshot(models_dir=_MODELS_DIR, build_id=build_id, meta=meta,
                              render_path=render_path, quarantine_path=quarantine_path,
                              wander=wander_pending,
                              persona_digest=persona)
        if snap:
            print(f"forensic snapshot -> {snap}", flush=True)
        return build_history.append(
            outcome=outcome, rows=len(examples), base_lr=learning_rate, run_id=run_id,
            corpus_fingerprint=fingerprint, probe_summary=probe_summary,
            adapter_dir=str(adapter_dir) if adapter_dir else None,
            build_id=build_id, built_at=built_at, seed=seed,
            snapshot_dir=str(snap) if snap else None)

    if not probe_passed:
        print("\n" + "!"*80)
        print(" CRITICAL REGRESSION PROBE FAILED ".center(80, "!"))
        print("!"*80)
        print("The trained model failed to meet capability, format, character, or retention gates.")
        print("Adapter discarded; the previously-promoted adapter keeps serving; no bundle "
              "age advances (REBUILD.md §10).", flush=True)
        # Record the rejected build + its forensic snapshot (REBUILD.md §7). A rejected build
        # leaves the serving adapter untouched — the next build re-derives from a genuinely
        # different corpus, never inheriting this dose — and its snapshot is the triage packet.
        _tier_pass = {k: v.get("passed") for k, v in (probe_results or {}).items()
                      if isinstance(v, dict)}
        _record_build(outcome="rejected", adapter_dir=None, probe_summary=_tier_pass)
        progress.emit("done", f"Regression probe FAILED (avg loss {loss_str}) — adapter "
                              "rejected, server config untouched.", status="rejected",
                      avg_loss=avg_loss, train_runtime=train_runtime)
        render_path.unlink(missing_ok=True)
        return {"examples": len(examples), "rows": len(examples),
                "corpus_rows": corpus_row_count,
                "quarantined_rows": len(quarantined_rows),
                "avg_loss": avg_loss, "train_runtime": train_runtime,
                "probe_passed": False, "probe_results": probe_results}

    # -- Save Adapter & Repoint Config --
    if use_staging:
        adapter_dir = _MODELS_DIR / "candidate"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        print(f"saving candidate adapter -> {adapter_dir}", flush=True)
        progress.emit("save", "Probe passed — saving candidate adapter (config untouched)...",
                      status="promoted")
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print("Staged run: candidate adapter saved (server_config.json untouched).", flush=True)
    else:
        adapter_dir = _adapter_dir_for(run_id)
        print(f"saving persistent adapter -> {adapter_dir}", flush=True)
        progress.emit("save", f"Probe passed — saving adapter -> {adapter_dir.name}",
                      status="promoted", adapter=adapter_dir.name)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        config["adapter_id"] = str(adapter_dir)
        save_server_config(config)
        print(f"server_config.json adapter_id -> {adapter_dir}", flush=True)

        # When this cycle was triggered by a reflection's train hand-off, copy the
        # adapter into reflections/<run_id>/adapter/ so that run's archive is a
        # self-contained, revertable "after this reflection" snapshot.
        if run_id:
            try:
                from core.reflection_archive import archive_adapter
                info = archive_adapter(run_id=run_id, adapter_dir=adapter_dir)
                if info:
                    print(f"archived adapter -> reflections/{run_id}/adapter/ "
                          f"({info.get('size_bytes', 0)} bytes)", flush=True)
            except Exception as exc:
                print(f"reflection adapter archive skipped: {exc}", flush=True)

    # -- record the promoted build, drop render --------------------------- #
    # REBUILD.md §5c/§7: on promotion, append ONE line to the build history instead of the
    # old stage-advance / persona-fact eviction / hot->archive move. That promoted line is
    # what advances every bundle's age (age = promoted builds since it was reflected), which
    # in turn raises its LR multiplier in the next build. There is no directory move and no
    # ledger mutation: "archived" is now a computed property (multiplier at cap). RAG is left
    # as-is this phase — the RAG/weights crossfade that decays consolidated material out of
    # retrieval is Phase 3 (separable), so consolidated chats/persona/fact stay retrievable
    # for now (they also retrain at their age-appropriate LR in every build).
    _tier_pass = {k: v.get("passed") for k, v in (probe_results or {}).items()
                  if isinstance(v, dict)}
    build_id = _record_build(outcome="promoted", adapter_dir=adapter_dir,
                             probe_summary=_tier_pass)
    render_path.unlink(missing_ok=True)

    # -- Produce the persona: the frozen, runnable product of this reflection -- #
    # A real promotion (not a staged candidate) mints a new Ava version, so snapshot the
    # resulting live state — the just-trained adapter, its forensic corpus, the fresh
    # digest.json, config, prompts, data/ — into data/persona/<id>/ and flip current.json.
    # GPU-free/filesystem-only (inference is already down here). Named to match the adapter
    # identifier (adapter-<id> → <id>), so persona, adapter, and reflections/<id>/ align.
    if not use_staging:
        persona_id = (adapter_dir.name[len("adapter-"):]
                      if adapter_dir.name.startswith("adapter-") else adapter_dir.name)
        try:
            import snapshot_state  # server/ is on sys.path (run as -m from server/)
            snapshot_state.produce_persona(persona_id, activate=True)
            print(f"produced persona -> data/persona/{persona_id} (activated)", flush=True)
            progress.emit("persona", f"Produced persona data/persona/{persona_id} (active).",
                          status="promoted", persona=persona_id)
        except Exception as exc:
            print(f"persona production skipped: {exc}", flush=True)
            progress.emit("persona", f"Persona production skipped: {exc}", status="warn")
    # Wander is now a durable, keep-forever corpus (server/data/til/wander.jsonl): because
    # every build fits a fresh LoRA on the frozen base, the corpus must persist to be
    # re-consolidated each build — so we do NOT clear it here anymore (restoring the old
    # resumed-adapter property that a wander's imprint carries across cycles). Quantity is
    # bounded upstream by the user-token wander budget, not by post-train retirement.
    _fresh_done = (f" + {len(preview_examples)} fresh preview (untrained)"
                   if preview_examples else "")
    print(f"promoted build {build_id}: adapter {adapter_dir.name} from {len(examples)} rows "
          f"({n_chat} chat + {n_wander} wander{_fresh_done}); corpus {fingerprint}; deleted "
          f"{render_path.name}.", flush=True)
    progress.emit("done", f"Build {build_id} promoted — adapter {adapter_dir.name}; "
                          f"avg loss {loss_str}; {len(examples)} rows "
                          f"({n_chat} chat + {n_wander} wander{_fresh_done}).",
                  status="promoted", adapter=adapter_dir.name, build_id=build_id,
                  avg_loss=avg_loss, train_runtime=train_runtime,
                  rows=len(examples), corpus_rows=corpus_row_count,
                  quarantined_rows=len(quarantined_rows),
                  chat_rows=n_chat, wander_rows=n_wander,
                  preview_rows=len(preview_examples),
                  corpus_fingerprint=fingerprint)

    return {"examples": len(examples), "rows": len(examples),
            "corpus_rows": corpus_row_count,
            "quarantined_rows": len(quarantined_rows), "chat_rows": n_chat,
            "wander_rows": n_wander, "preview_rows": len(preview_examples),
            "build_id": build_id,
            "corpus_fingerprint": fingerprint, "avg_loss": avg_loss,
            "train_runtime": train_runtime,
            "adapter_dir": str(adapter_dir), "render_file": SFT_RENDER_FILE,
            "probe_passed": True, "probe_results": probe_results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="regenerate + render sft_render.jsonl, but do not train/merge/advance")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="LoRA rank; overrides server_config.json `lora_r` (default 16) "
                         "for this run only. Scaling is rank-stabilized "
                         "(gamma = alpha/sqrt(r), alpha fixed at 4), so changing rank "
                         "changes capacity without changing the effective LR")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None,
                    help="base/peak LR; overrides server_config.json `train_lr` "
                         "(default 8e-6) for this run only")
    ap.add_argument("--use-staging", action="store_true",
                    help="read/write staging folder metadata and save adapter candidate")
    ap.add_argument("--model-id", default=None,
                    help="override the base model_id for this run only (e.g. a small "
                         "model to validate the cycle end-to-end); server_config.json untouched")
    ap.add_argument("--run-id", default=None,
                    help="reflection run id this cycle serves; the produced adapter is "
                         "archived into reflections/<run-id>/adapter/ for review/revert")
    ap.add_argument("--skip-validation", action="store_true",
                    help="promote the trained adapter without running the regression probe "
                         "(skips the capability/format/continuity/retention gate and its "
                         "pre-train baselines). Faster, but no single-cycle damage tripwire — "
                         "the adapter lineage still makes a bad promotion reversible.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the build (recorded on the build line + forensic "
                         "snapshot; a build is a function of base+corpus+config+seed).")
    ap.add_argument("--include-fresh", action="store_true",
                    help="also render chats not yet eligible for weights (younger than "
                         "rag_only_window_h, or only background-frozen) as preview rows "
                         "(lr multiplier 0, `preview: true`) appended to sft_render.jsonl "
                         "— they reach the Training review tab for early repair but are "
                         "NEVER trained.")
    args = ap.parse_args()
    try:
        run_cycle(dry_run=args.dry_run, lora_r=args.lora_r,
                  epochs=args.epochs, learning_rate=args.lr,
                  use_staging=args.use_staging, model_id_override=args.model_id,
                  run_id=args.run_id, skip_validation=args.skip_validation,
                  seed=args.seed, include_fresh=args.include_fresh)
    except Exception as e:
        # Append a terminal event so the Sleep tab's live poll reports the failure
        # rather than just seeing the process vanish. The client also stops on the
        # watchdog's running=false, but this carries the reason. reset=False keeps
        # the events run_cycle already emitted. Best-effort; re-raise to exit rc!=0.
        try:
            TrainProgress(reset=False).emit(
                "done", f"Train cycle crashed: {e}", status="error")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
