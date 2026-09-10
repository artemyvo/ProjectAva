"""Inference backend — Unsloth (CUDA, Linux)."""

from __future__ import annotations

import gc
import os
import queue as queue_module
import re
import threading
import time
from typing import Any, Callable, Generator, Optional, Tuple

# Stop unsloth from wrapping its generated model forwards (e.g.
# unsloth_compiled_cache/unsloth_compiled_module_gemma4.py) in torch.compile. Those
# @torch.compile decorators SNAPSHOT the TorchDynamo config at decoration time — which
# happens when unsloth imports that generated module during from_pretrained — so no
# runtime tweak to torch._dynamo.config (disable / recompile_limit), set before OR after
# the load, ever reaches the config the wrapper uses. The compiled RMSNorm/attention then
# recompiles once per prompt-length bucket and, past the default limit of 8 buckets,
# hard-fails with FailOnRecompileLimitHit (fullgraph=True) — killing a chat/reflection
# generation the moment enough distinct input shapes go through it. Disabling unsloth's
# compile makes those forwards plain eager Python (correct; unsloth's speed is in its
# Triton kernels/patches, not this wrapper), so Dynamo never engages. Must be set before
# unsloth is imported, hence module scope; setdefault so an operator can still override.
# NOTE: unsloth caches the generated module on disk — after enabling this, delete any
# stale server/inference/unsloth_compiled_cache/ so it is regenerated without the wrappers.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


# Runaway-repetition guard for reflection passes (opt-in via stop_on_repeat).
# A reflection CoT occasionally falls into a verbatim loop — a whole sentence
# repeated to context exhaustion — which fills the token budget and leaves an
# unparseable, never-closed <think>. We stop generation when the same span of
# _LOOP_NGRAM tokens has recurred _LOOP_MIN_REPEATS times. Unlike a sampling-level
# no_repeat_ngram ban (which corrupts normal prose by forcing off-distribution
# tokens at every legitimate phrase reuse), this only *halts* on genuine pathology
# and never alters a single sampled token. The n-gram window also catches loops
# whose separator increments ("Note 2:", "Note 3:", …): the invariant span inside
# the repeated body still recurs identically even though the full period varies.
_LOOP_NGRAM = 12
_LOOP_MIN_REPEATS = 4


def detect_ngram_loop(gen_ids: list, ngram: int = _LOOP_NGRAM,
                      repeats: int = _LOOP_MIN_REPEATS) -> bool:
    """True when some *ngram*-token span recurs ``>= repeats`` times in *gen_ids*.

    Pure batch form of the streaming stop criterion, for the GPU-free self-test.
    A 12-token span recurring four times in one generation is a strong loop signal
    that natural prose effectively never produces."""
    if len(gen_ids) < ngram:
        return False
    counts: dict = {}
    best = 0
    for i in range(len(gen_ids) - ngram + 1):
        key = tuple(gen_ids[i:i + ngram])
        c = counts.get(key, 0) + 1
        counts[key] = c
        if c > best:
            best = c
    return best >= repeats


# Degeneration guard for the chat path (opt-in via stream_generate(degen_stop=…)).
# The verbatim n-gram guard above is blind to *drifting* degeneration — an
# associative walk ("ring sing swing wing king…") or letter-soup where every
# window is novel, so no span recurs. That is exactly the shape the mild chat
# repetition_penalty *converts* a verbatim loop into (penalizing exact repeats
# turns the loop into a non-repeating walk that evades detect_ngram_loop). It has
# a content-blind mechanical signature: the recent window stops looking like
# language — its distinct-token ratio collapses, or one token dominates it. Real
# prose (even a deliberately repetitive stylistic riff) sits well above these
# floors, so the guard halts only genuine collapse, never expression — and it
# holds for any emergent persona.
_DEGEN_WINDOW = 64        # rolling window of most-recent generated tokens
_DEGEN_MIN_GEN = 128      # never judge before this many generated tokens
_DEGEN_DISTINCT = 0.35    # halt if distinct/window drops below this
_DEGEN_TOP_FREQ = 0.40    # …or if a single token exceeds this share of the window


def detect_degeneration(gen_ids: list, window: int = _DEGEN_WINDOW,
                        min_gen: int = _DEGEN_MIN_GEN,
                        distinct: float = _DEGEN_DISTINCT,
                        top_freq: float = _DEGEN_TOP_FREQ) -> bool:
    """True when the tail *window* of *gen_ids* has stopped looking like language.

    Pure batch form of the streaming degeneration criterion, for the GPU-free
    self-test. Complements ``detect_ngram_loop``: that catches verbatim loops;
    this catches *drifting* collapse (novel-every-step associative walks / letter
    soup) whose windows never repeat but whose token diversity has cratered."""
    if len(gen_ids) < max(min_gen, window):
        return False
    win = gen_ids[-window:]
    n = len(win)
    if len(set(win)) / n < distinct:
        return True
    counts: dict = {}
    for t in win:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) / n > top_freq


# Anti-copy guard for the chat path (opt-in via stream_generate(no_copy_text=…)).
# At t≈1.0 the model occasionally regurgitates its immediately-previous reply
# VERBATIM (with a fresh, on-topic CoT): once the first few answer tokens happen
# to match the prior reply's opening, induction-style copying of a context span
# becomes near-deterministic and no existing guard sees it — _RepetitionStop
# watches within-generation recurrence (a single prompt-copy pass never recurs),
# _DegenStop watches token-diversity collapse (copied text is healthy prose), and
# the generated-only repetition penalty exempts the prompt by design. Two
# complementary mechanisms, both fed by the caller-supplied `no_copy_text` (the
# previous assistant reply; system prompt / RAG blocks stay exempt):
#   1. its token ids are unioned into the repetition-penalty gather, restoring
#      the copy deterrent the prompt exemption removed — for exactly this text;
#   2. a _NoCopyPrevReply LogitsProcessor masks any token that would EXTEND a
#      verbatim ≥_NO_COPY_NGRAM-token match with a span of that text, so a copy
#      run is hard-capped below one phrase length while paraphrase stays free.
_NO_COPY_NGRAM = 8


def build_no_copy_table(ids: list, ngram: int = _NO_COPY_NGRAM) -> dict:
    """Map each (ngram-1)-token span of *ids* to the set of ids that follow it.

    The streaming processor looks up the last ngram-1 *generated* tokens and masks
    the continuations found here — banning the token that would complete an
    ngram-length verbatim copy. Pure and GPU-free for the self-test."""
    table: dict = {}
    prefix = ngram - 1
    for i in range(len(ids) - prefix):
        key = tuple(int(t) for t in ids[i:i + prefix])
        table.setdefault(key, set()).add(int(ids[i + prefix]))
    return table


def banned_continuations(gen_tail: list, table: dict, ngram: int = _NO_COPY_NGRAM) -> set:
    """Continuation ids the anti-copy guard would mask given the generated tail.

    Pure batch form of the `_NoCopyPrevReply` criterion, for the GPU-free self-test:
    returns the banned set when the last ngram-1 tokens of *gen_tail* match a span
    of the protected text, else an empty set."""
    prefix = ngram - 1
    if len(gen_tail) < prefix:
        return set()
    return table.get(tuple(int(t) for t in gen_tail[-prefix:]), set())


class ThinkCeiling:
    """State machine behind the maximum-thought-length ceiling. Pure and GPU-free.

    Answers one question per generated step: *force the reasoning channel closed now?*
    ``True`` once the thought has run ``max_think_tokens`` steps without closing, and
    never again after the model closes on its own — which is the ordinary case and must
    cost nothing. The ceiling is the inverse of ``ModelFamily.min_think_tokens``: the
    floor guarantees the channel is not closed empty, this guarantees it is closed in
    time to write an answer (see ``_MaxThinkLength`` in ``stream_generate``).

    Deliberately blind to content — it counts tokens and watches for the family's close
    marker, nothing else — so it holds for any persona, language, or prompt.
    """

    def __init__(self, max_think_tokens: int, close_ids) -> None:
        self.max_think_tokens = int(max_think_tokens or 0)
        self.close_ids = frozenset(int(i) for i in (close_ids or ()))
        self.closed = False

    def step(self, n_generated: int, last_token_id=None) -> bool:
        """Fold in the newest generated token; return whether to force the close now.

        *last_token_id* is the most recently generated id (``None`` before the first
        step). Every generated token passes through exactly one call, so watching only
        the newest one observes them all.
        """
        if self.closed or not self.max_think_tokens or not self.close_ids:
            return False
        if n_generated > 0 and last_token_id is not None and int(last_token_id) in self.close_ids:
            self.closed = True
            return False
        return n_generated >= self.max_think_tokens


def trim_generated_row(row: list, stop_ids: list, pad_id) -> list:
    """Cut one batched-generation row at its first stop token (kept), then strip
    any pad tail.

    In a batch, rows that finish on EOS early keep getting padded until the whole
    batch stops, so a raw row reads ``…tokens, EOS, pad, pad, …``. Trimming at the
    first stop id (inclusive) reproduces exactly what a lone generation of that row
    would have returned. The pad-strip is pure defense for the no-stop-id case —
    a row only finishes early *by* emitting a stop token. Pure + GPU-free for the
    self-test."""
    stops = set(stop_ids or [])
    for i, tid in enumerate(row):
        if tid in stops:
            return row[:i + 1]
    if pad_id is not None:
        end = len(row)
        while end > 0 and row[end - 1] == pad_id:
            end -= 1
        return row[:end]
    return row


def _wants_fast_model(model_id: str) -> bool:
    """True for checkpoints unsloth loads through ``FastModel`` rather than
    ``FastLanguageModel``: a ``*ForConditionalGeneration`` architecture (multimodal
    wrapper — gemma-4, Qwen3.5/3.6 incl. the text-only MoE sizes) per the on-disk
    config, else a name heuristic for a model not yet downloaded."""
    try:
        from core.moe_bnb_experts import _read_config
        cfg = _read_config(model_id) or {}
        archs = cfg.get("architectures") or []
        if archs:
            return any("ForConditionalGeneration" in a for a in archs)
    except Exception:
        pass
    mid = (model_id or "").lower()
    return any(k in mid for k in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6", "gemma-4"))


def _stage_adapter_with_base(adapter_id: str, base_model_id: str) -> str:
    """Stage a throwaway copy of a LoRA adapter dir whose base is `base_model_id`.

    Returns a fresh temp dir that mirrors `adapter_id` — every file symlinked to
    the original (the safetensors are big; we never copy them) except
    `adapter_config.json`, which is rewritten so `base_model_name_or_path` points at
    `base_model_id`. Passing this dir to unsloth's `from_pretrained` makes its native
    loader resolve the full-precision base (instead of the adapter's baked-in
    bnb-4bit repo), so a 16/8-bit request is honored while still going through
    unsloth's tested base-load + adapter-attach + inference-patch path. The caller
    deletes the returned dir after the load; the original adapter is untouched.
    """
    import json
    import os
    import tempfile

    staged = tempfile.mkdtemp(prefix=".base-override-")
    for name in os.listdir(adapter_id):
        src = os.path.abspath(os.path.join(adapter_id, name))
        dst = os.path.join(staged, name)
        if name == "adapter_config.json":
            with open(src, encoding="utf-8") as fh:
                cfg = json.load(fh)
            cfg["base_model_name_or_path"] = base_model_id
            with open(dst, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        else:
            os.symlink(src, dst)
    return staged


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class InferenceBackend:
    name: str = "base"

    def load(
        self,
        model_id: str,
        context_length: int,
        adapter_id: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def release(self, model: Any, tokenizer: Any) -> None:
        raise NotImplementedError

    def reclaim(self) -> None:
        """Return freed memory to the allocator (optional; no-op by default).

        Called by a caller that held its own reference to a released model, once that
        reference is gone — see ``UnslothBackend.reclaim``.
        """

    def count_tokens(self, tokenizer: Any, text: str) -> int:
        raise NotImplementedError

    def memory_status(self) -> str:
        return "N/A"

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        context_length: int,
        temperature: float,
        top_p: float,
        debug: Optional[Callable[[str], None]] = None,
        top_k: Optional[int] = None,
    ) -> Generator[str, None, None]:
        raise NotImplementedError
        yield  # mark as generator


# ──────────────────────────────────────────────────────────────────────────────
# Unsloth backend  (CUDA — Linux, DGX Spark)
# ──────────────────────────────────────────────────────────────────────────────

class UnslothBackend(InferenceBackend):
    name = "unsloth"
    # Populated by stream_generate when capture_tension=True; read by the server
    # right after generation. Safe because generation is single-threaded.
    last_token_signals: Optional[dict] = None
    # Probability mass the first generated step put on the family's think-open token
    # (gemma-4 `<|channel>`) — the "Thinking: NN%" diagnostic. None when the family
    # has no sampled opener (qwen3 prefills it) or the marker doesn't resolve.
    last_think_open_prob: Optional[float] = None
    _lm_head_cache: Optional[tuple] = None
    # Monotonic timestamp per generated token (appended by the lm-head hook in
    # stream_generate, one call per token), reset at the start of each generation.
    # Powers the tok/s readout: current_tokens_per_sec() reads a rolling window off
    # it live, and last_tokens_per_sec is the finalized whole-generation average.
    _gen_token_times: list = []
    last_tokens_per_sec: Optional[float] = None

    @staticmethod
    def _apply_dynamo_settings() -> None:
        """Force TorchDynamo into a no-op / high-tolerance state.

        Unsloth leaves each forward as a TorchDynamo-optimized function. During a
        reflection run (and a long chat) each prompt length is a different bucket, so
        generate() recompiles per bucket; transformers' compiled-cache path can then
        FX-trace an already-dynamo-optimized graph ("Detected that you are using FX to
        symbolically trace a dynamo-optimized function") or blow the recompile limit
        (FailOnRecompileLimitHit, fullgraph=True) once it exceeds the default 8 buckets.

        Disabling Dynamo process-wide turns those compiled wrappers into plain eager
        functions (inference speed comes from unsloth's kernels/patches, not this Dynamo
        wrapper), so nothing recompiles; suppress_errors + a raised recompile limit are
        belt-and-suspenders for any path that still slips through.

        NOTE: this alone was NOT enough for unsloth's compiled model modules — their
        @torch.compile decorators snapshot the Dynamo config at *decoration* time (when
        the generated module is imported during from_pretrained), so a runtime tweak here
        (before or after) never reaches the config those wrappers use, and the limit stays
        at the default 8. UNSLOTH_COMPILE_DISABLE (set at module scope) is what actually
        stops that path; these settings remain for any other Dynamo-touched code.
        The recompile-limit knob was renamed cache_size_limit -> recompile_limit in newer
        PyTorch (Spark's stack), so set whichever names exist.
        """
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        for _limit_attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, _limit_attr):
                setattr(torch._dynamo.config, _limit_attr, 64)

    @staticmethod
    def _purge_unsloth_compile_cache() -> None:
        """Delete stale unsloth_compiled_cache/ dirs (best-effort).

        Once UNSLOTH_COMPILE_DISABLE is on, unsloth regenerates its model modules without
        the torch.compile wrappers — but a cache written by an earlier run still holds the
        compiled (crash-prone) version, and unsloth may reuse it. The cache is fully
        regenerable, so wiping it on load forces a clean, wrapper-free regen. Checks the
        CWD (where unsloth writes it) and the inference package root."""
        import shutil
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # inference/
        for base in (os.getcwd(), here):
            cache = os.path.join(base, "unsloth_compiled_cache")
            if os.path.isdir(cache):
                try:
                    shutil.rmtree(cache)
                except Exception:
                    pass

    def load(
        self,
        model_id: str,
        context_length: int,
        adapter_id: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        import os
        os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
        # Belt-and-suspenders: the module-scope setdefault already covers a fresh process,
        # but pin it here too so it's on regardless of import order, then drop any stale
        # (compiled) cache so unsloth regenerates the forwards without torch.compile.
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        self._purge_unsloth_compile_cache()

        self._apply_dynamo_settings()

        # Resolve base-model precision. The caller (handle_load, driven by
        # server_config.json's base_quant) may pin 16-bit (both flags False),
        # 8-bit (load_in_8bit), or 4-bit (load_in_4bit). When neither is set we
        # keep the historical default so nothing regresses. Same model_id at any
        # of these tiers preserves tokenizer + LoRA compatibility — precision is a
        # load-time choice, not a different model.
        def _quant_kwargs(default_4bit: bool) -> dict:
            if load_in_8bit:
                return {"load_in_8bit": True, "load_in_4bit": False}
            if load_in_4bit is None:
                return {"load_in_4bit": default_4bit}
            return {"load_in_4bit": bool(load_in_4bit)}

        has_adapter = bool(adapter_id and os.path.exists(adapter_id))

        # A LoRA adapter's adapter_config.json hardcodes base_model_name_or_path,
        # and loading via from_pretrained(adapter_dir) makes unsloth resolve the
        # base from THAT — which for our trained adapters is the bnb-4bit repo. So
        # the convenient one-call path is pinned to 4-bit and silently ignores a
        # 16/8-bit request. When precision is explicitly raised (8-bit, or 4-bit
        # turned off) AND an adapter is present, we still want unsloth's *native*
        # load path (base load → adapter attach → its inference patches), because
        # hand-rolling the attach with raw peft skips unsloth's patch step and
        # crashes ("unhashable type: 'set'" in for_inference). Instead we point that
        # native path at the full-precision model_id base by staging a throwaway
        # adapter dir: symlinks to the real weights + an adapter_config.json whose
        # base_model_name_or_path is rewritten to model_id. unsloth then loads the
        # bf16 base at the requested precision and applies the LoRA on top. A LoRA
        # is precision-agnostic, so a 4-bit-trained adapter fits the bf16 base.
        # Without an override we keep the historical single-call path unchanged.
        override_precision = bool(load_in_8bit) or (load_in_4bit is False)
        adapter_override = has_adapter and override_precision

        is_gemma = "gemma-4" in model_id.lower()
        # FastModel is unsloth's loader for multimodal / *ForConditionalGeneration
        # checkpoints (gemma-4, Qwen3.5 — whose text-only MoE still ships as a
        # ConditionalGeneration architecture with a vision tower); FastLanguageModel
        # for plain causal LMs (qwen3, gpt-oss). Decided from the checkpoint's own
        # config where one is on disk, by name otherwise.
        if is_gemma or _wants_fast_model(model_id):
            from unsloth import FastModel as _Loader
        else:
            from unsloth import FastLanguageModel as _Loader
        # A converted per-expert bnb Qwen3.5-MoE checkpoint needs its experts class
        # swapped in BEFORE the model is built (see core/moe_bnb_experts.py); any
        # other model gets the stock class back, since the swap is process-global.
        from core import moe_bnb_experts
        if moe_bnb_experts.install_if_needed(model_id):
            print(f"[load] per-expert bnb-4bit experts installed for {model_id}", flush=True)

        staged_dir: Optional[str] = None
        try:
            if has_adapter and not adapter_override:
                # Historical path: unsloth derives the (bnb-4bit) base from the
                # adapter's own config.
                load_name, quant_kw = adapter_id, _quant_kwargs(True)
            elif adapter_override:
                # Precision override: native path pointed at the model_id base.
                staged_dir = _stage_adapter_with_base(adapter_id, model_id)
                load_name, quant_kw = staged_dir, _quant_kwargs(True)
            else:
                # Bare base (no adapter). Raw gpt-oss weights are MXFP4; bnb-4bit
                # variants are already bitsandbytes-quantized (load_in_4bit=True).
                gpt_oss_default = "gpt-oss" not in model_id.lower() or "bnb-4bit" in model_id.lower()
                load_name = model_id
                quant_kw = _quant_kwargs(True if is_gemma else gpt_oss_default)

            # Unified-memory box (DGX Spark): pin everything to device 0 and skip
            # accelerate's free-VRAM planning, which reads MemFree (page cache
            # excluded) and offloads to "CPU" — the same pool — then bnb refuses.
            from core import unified_memory, fast_load
            placement_kw = unified_memory.load_kwargs()
            # Clone out of the safetensors mmap before the device copy (0.16 GB/s
            # straight from file-backed pages on GB10) and skip Hub round trips for
            # a model that is already on disk. See core/fast_load.py.
            fast = fast_load.install()
            print(f"[load] {unified_memory.describe()} placement={placement_kw or 'default'} "
                  f"fast_load={fast}", flush=True)
            if unified_memory.is_unified_memory() and not fast:
                print("!!! FAST LOAD NOT ACTIVE on a unified-memory box: expect this load to take "
                      "minutes, not seconds (mmap->device copies at ~0.16 GB/s). See SPARK_LOADING.md.",
                      flush=True)
            with fast_load.hf_offline_if_cached(load_name) as off:
                print(f"[load] hf_offline={off.active}", flush=True)
                model, tokenizer = _Loader.from_pretrained(
                    model_name=load_name, max_seq_length=context_length, **quant_kw, **placement_kw,
                )
        finally:
            if staged_dir is not None:
                import shutil
                shutil.rmtree(staged_dir, ignore_errors=True)

        self._assert_fully_materialized(model)

        _Loader.for_inference(model)

        self._normalize_adapter_dtype(model)

        model.eval()
        return model, tokenizer

    # Device-map / parameter placements that mean "these weights are not on the GPU".
    _OFFLOAD_DEVICES = ("cpu", "disk", "meta")

    @classmethod
    def _assert_fully_materialized(cls, model: Any) -> None:
        """Refuse a model accelerate has offloaded to cpu/disk (or left on meta).

        Nothing in this project passes ``device_map``/``max_memory``, so when free VRAM
        is short at load time accelerate decides on its own to dispatch part of the model
        off-GPU and installs ``AlignDevicesHook``s over it. That load SUCCEEDS — the
        failure surfaces much later, on the first forward, as

            NotImplementedError: Cannot copy out of meta tensor; no data!

        raised out of ``accelerate/hooks.py`` ``post_forward`` when the hook tries to move
        a *meta output* back to the execution device: the weights were never materialized,
        and meta ops propagate shapes without complaining, so nothing objects until the
        hook does. The process is then poisoned for its whole life — every generation path
        (chat, wander, outreach, reflection) dies on that one shared model — and the log
        blames whichever subsystem happened to run first. A hard failure HERE, naming the
        real cause, beats a server that looks healthy and cannot answer.

        The usual reason free VRAM is short is a previous model that was never actually
        evicted; see ``release`` for why that used to happen silently.

        ``AVA_ALLOW_OFFLOAD=1`` downgrades this to a warning (deliberate offload on a box
        where a slow model beats no model).
        """
        bad: list = []
        dmap = getattr(model, "hf_device_map", None)
        if isinstance(dmap, dict):
            for mod_name, dev in dmap.items():
                if str(dev).lower().split(":")[0] in cls._OFFLOAD_DEVICES:
                    bad.append(f"{mod_name or '<root>'} -> {dev}")
        if not bad:
            # Backstop for a load that exposes no hf_device_map: an unmaterialized
            # parameter is on the meta device and has no storage behind it.
            try:
                for pname, p in model.named_parameters():
                    if getattr(getattr(p, "device", None), "type", None) == "meta":
                        bad.append(f"{pname} -> meta")
                        if len(bad) >= 8:
                            break
            except Exception:
                pass
        if not bad:
            return

        detail = ", ".join(bad[:8]) + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else "")
        msg = f"model loaded with offloaded / unmaterialized weights: {detail}"
        if os.environ.get("AVA_ALLOW_OFFLOAD"):
            print(f"[backend] WARNING: {msg} — AVA_ALLOW_OFFLOAD is set, continuing. "
                  f"Generation will likely fail with 'Cannot copy out of meta tensor'.",
                  flush=True)
            return
        raise RuntimeError(
            f"{msg}. Free VRAM was short at load time, so accelerate dispatched part of "
            f"the model off-GPU. This model would generate nothing but 'Cannot copy out "
            f"of meta tensor; no data!'. Check that the previously loaded model was "
            f"actually released (nvidia-smi) before retrying; set AVA_ALLOW_OFFLOAD=1 to "
            f"load it anyway."
        )

    @staticmethod
    def _normalize_adapter_dtype(model: Any) -> None:
        """Down-cast float32 LoRA params to the base model's compute dtype.

        PEFT saves a trained adapter with float32 A/B matrices (training keeps
        them in fp32 for stability), and unsloth's ``for_inference`` does not
        down-cast them on the gemma-4 ``FastModel`` load path. The base, however,
        is bnb-4bit with a bf16 compute dtype, so the very first generation dies
        in the LoRA matmul — the delta computes ``bf16_activations @
        fp32_lora_weight``:

            RuntimeError: expected m1 and m2 to have the same dtype,
                          but got: c10::BFloat16 != float

        (surfaced by whatever generation runs first — the autonomous outreach idle
        job in a headless deploy — but it would crash chat identically). Casting
        the LoRA params to the base compute dtype matches how a normally-loaded
        unsloth adapter behaves and is numerically standard for inference. No-op
        when there is no adapter (no ``lora_`` params) or the base is itself fp32.
        """
        try:
            import torch
        except Exception:
            return
        # Reference compute dtype: embeddings are not 4bit-quantized, so their
        # dtype is the base compute dtype (bf16/fp16). Fall back to the first
        # non-fp32 floating param, else bf16.
        target = None
        try:
            target = model.get_input_embeddings().weight.dtype
        except Exception:
            target = None
        if target is None or target == torch.float32:
            for p in model.parameters():
                if p.is_floating_point() and p.dtype != torch.float32:
                    target = p.dtype
                    break
        if target is None:
            target = torch.bfloat16
        if target == torch.float32:
            return  # base itself is fp32 — nothing to reconcile
        cast = 0
        try:
            for name, p in model.named_parameters():
                if "lora_" in name and p.dtype == torch.float32:
                    p.data = p.data.to(target)
                    cast += 1
        except Exception:
            import traceback
            traceback.print_exc()
            return
        if cast:
            print(f"[backend] cast {cast} float32 LoRA params -> {target}", flush=True)

    @staticmethod
    def _refuses_device_move(model: Any) -> bool:
        """True when ``.to()``/``.cpu()`` on this model can only raise.

        Two cases, both of which our loads hit: transformers refuses to move a
        bitsandbytes 4/8-bit model at all (``ValueError: .to is not supported for 4-bit
        or 8-bit bitsandbytes models``), and an accelerate-offloaded model raises
        ``NotImplementedError: Cannot copy out of meta tensor`` from ``Module._apply``
        the moment it reaches an unmaterialized parameter.
        """
        return bool(
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "hf_quantizer", None) is not None
            or getattr(model, "quantization_method", None) is not None
            or getattr(model, "hf_device_map", None)
        )

    def reclaim(self) -> None:
        """Collect and return freed blocks to the GPU allocator.

        Split out of ``release`` so a caller can run it AFTER dropping its own
        references: ``del`` inside ``release`` only clears ``release``'s locals, so a
        model still named in the caller's frame survives the collection that happens
        there (see ``agentic.CleanBaseSession``). Safe to call with nothing to free.
        """
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def release(self, model: Any, tokenizer: Any) -> None:
        """Evict a loaded model.

        ``model.cpu()`` used to be the eviction step. But every model this server loads
        is bitsandbytes-quantized, and transformers REFUSES to move one — so that call
        raised on each release, straight into a bare ``except: pass``, and the weights
        stayed resident. The next load then saw a full card, and accelerate responded by
        silently offloading part of the *new* model to cpu/disk rather than OOM-ing,
        yielding a model that loads cleanly and dies on its first forward with "Cannot
        copy out of meta tensor; no data!" (now caught at load by
        ``_assert_fully_materialized``).

        So: only move a plain, unquantized model to CPU; for everything else dropping the
        last reference IS the eviction, and a failed move is logged rather than swallowed
        — a failed eviction is precisely what poisons the next load.

        Callers holding their own reference must clear it and then call ``reclaim()``;
        the ``del`` here reaches only this frame's locals.

        ``_lm_head_cache`` is dropped here for the same reason: it holds the outgoing
        model's output-projection MODULE, which on a tied-embedding family is the
        unquantized vocab matrix (gemma-4: 262k rows in bf16, GBs) — a backend-singleton
        reference that outlives every swap and is easily the margin that decides whether
        the next load fits or gets offloaded.
        """
        self._lm_head_cache = None
        if model is not None and not self._refuses_device_move(model):
            try:
                if hasattr(model, "cpu"):
                    model.cpu()
            except Exception as e:
                print(f"[backend] release: model.cpu() failed "
                      f"({type(e).__name__}: {e}) — falling back to dropping the "
                      f"reference; VRAM frees only once no reference remains.",
                      flush=True)
        try:
            del model, tokenizer
        except Exception:
            pass
        self.reclaim()

    def count_tokens(self, tokenizer: Any, text: str) -> int:
        # Standard tokenizer path
        try:
            return len(tokenizer.encode(text))
        except (AttributeError, TypeError):
            pass
        # Processor path (e.g. Gemma4Processor): delegate to underlying tokenizer
        underlying = getattr(tokenizer, "tokenizer", None)
        if underlying is not None:
            try:
                return len(underlying.encode(text))
            except Exception:
                pass
        return 0

    def memory_status(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                free, total = torch.cuda.mem_get_info()
                # `reserved` is the caching-allocator pool the process actually
                # holds from the driver; `alloc` is just the live tensors inside
                # it. Reporting only `alloc` hides pool growth and makes a near-OOM
                # look healthy, so surface reserved too.
                return (
                    f"VRAM: {alloc:.2f} GB alloc / "
                    f"{reserved:.2f} GB reserved / "
                    f"{free / 1024 ** 3:.2f} GB free / "
                    f"{total / 1024 ** 3:.2f} GB total"
                )
            return "VRAM: CUDA not available"
        except Exception:
            return "VRAM: unavailable"

    def trim_memory(self) -> None:
        """Return unused reserved cache to the driver. Cheap; call after each
        generation so the KV-cache blocks for a finished turn don't accumulate."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _resolve_stop_ids(model: Any, tokenizer: Any) -> list:
        """Collect stop sequences from model and tokenizer configs."""
        stop_ids: list[int] = []
        for source in (
            tokenizer.eos_token_id,
            getattr(getattr(model, "generation_config", None), "eos_token_id", None),
            getattr(getattr(model, "config", None), "eos_token_id", None),
        ):
            if source is None:
                continue
            stop_ids.extend(source if isinstance(source, (list, tuple)) else [source])
        return list(dict.fromkeys(t for t in stop_ids if isinstance(t, int) and t >= 0))

    @staticmethod
    def _resolve_pad_id(tokenizer: Any):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            eos = tokenizer.eos_token_id
            pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        return pad_id

    @staticmethod
    def _resolve_think_open_id(model: Any, tokenizer: Any) -> Optional[int]:
        """Token id of the loaded family's think-open marker, or None if unsampled.

        Reads ``ModelFamily.think_open_markers`` and resolves the first marker that maps
        to a real special-token id (not unk). Returns None for families with no sampled
        opener (qwen3 prefills ``<think>`` into the prompt) or when the marker isn't in
        the vocab — in which case the "Thinking: NN%" diagnostic is simply omitted.
        """
        from core import model_family
        name = str(
            getattr(model, "name_or_path", "")
            or getattr(getattr(model, "config", None), "_name_or_path", "")
            or getattr(getattr(model, "config", None), "name_or_path", "")
        ).lower()
        markers = model_family.family_for(name).think_open_markers
        if not markers:
            return None
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        unk = getattr(text_tokenizer, "unk_token_id", None)
        for marker in markers:
            try:
                tid = text_tokenizer.convert_tokens_to_ids(marker)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                return tid
        return None

    @staticmethod
    def _resolve_think_close_ids(model: Any, tokenizer: Any) -> list:
        """Special-token ids of the loaded family's reasoning-*close* marker(s).

        Backs the minimum-thought-length floor: these are the ids forbidden for the
        first ``min_think_tokens`` generated steps so a prefilled-open channel can't be
        closed empty. Reads ``ModelFamily.close_markers`` and keeps only markers that map
        to a real single special-token id (gemma-4 ``<channel|>`` → 101); multi-token
        text markers with no dedicated id (e.g. ``</think>`` under gemma) are skipped.
        """
        from core import model_family
        name = str(
            getattr(model, "name_or_path", "")
            or getattr(getattr(model, "config", None), "_name_or_path", "")
            or getattr(getattr(model, "config", None), "name_or_path", "")
        ).lower()
        markers = model_family.family_for(name).close_markers
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        unk = getattr(text_tokenizer, "unk_token_id", None)
        ids: list = []
        for marker in markers:
            try:
                tid = text_tokenizer.convert_tokens_to_ids(marker)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != unk and tid not in ids:
                ids.append(tid)
        return ids

    def _family_top_k(self, model: Any) -> int:
        """Recommended top-k for the loaded model's family (0 = leave HF default).

        Centralizes the model card / Unsloth sampling recommendation so chat,
        reflection, and branch replay all apply the family's top-k without it being
        threaded through the (top-k-less) WebSocket protocol. Resolved from the same
        ``name_or_path`` the streaming path uses for family detection."""
        from core import model_family
        name = str(
            getattr(model, "name_or_path", "")
            or getattr(getattr(model, "config", None), "_name_or_path", "")
            or getattr(getattr(model, "config", None), "name_or_path", "")
        ).lower()
        return model_family.family_for(name).rec_top_k

    def generate_from_ids(
        self,
        model: Any,
        tokenizer: Any,
        input_ids: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
    ) -> list:
        """Generate a continuation from an exact token-id prefix; returns the new ids.

        Branch replay for branch-and-select revision: the prefix is the templated
        prompt's ids plus a forced slice of a previously generated series (ending on
        the road-not-taken token), so no chat template or string round-trip may be
        applied here — that would not reproduce the original ids. Non-streaming,
        no tension capture. Batch-of-1 form of :meth:`generate_from_ids_batch` —
        one code path, so the serial fallback exercises the same logic.
        """
        return self.generate_from_ids_batch(
            model, tokenizer, [list(input_ids)], max_new_tokens,
            temperature, top_p, top_k=top_k,
        )[0]

    def generate_from_ids_batch(
        self,
        model: Any,
        tokenizer: Any,
        prefixes: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
    ) -> list:
        """Generate continuations for several exact token-id prefixes in one batch.

        The branch-fork primitive: the forks of one exchange share sampling params
        and differ only in prefix, and decode on the 4-bit model is memory-bandwidth-
        bound, so a batch of 4 costs barely more than one row — vs 4 serial
        generations each re-prefilling the same replayed conversation.

        Prefixes differ in length, so they are LEFT-padded to a common length with
        an explicit attention mask: generated tokens must sit immediately after each
        prefix, and generate() derives position ids from the mask, so the padding
        never shifts real token positions. Rows finish on EOS independently
        (generate() pads early finishers until the batch stops), so each returned
        row is cut at its first stop token — kept, matching the single-row form —
        via :func:`trim_generated_row`. Returns one continuation id-list per prefix,
        in order. Same contract as generate_from_ids otherwise: exact ids in, no
        chat template, non-streaming, no tension capture.
        """
        import torch

        # See stream_generate: reassert the Dynamo config (unsloth resets it at load).
        self._apply_dynamo_settings()

        if not prefixes:
            return []
        stop_ids = self._resolve_stop_ids(model, tokenizer)
        pad_id = self._resolve_pad_id(tokenizer)
        device = next(model.parameters()).device

        max_len = max(len(p) for p in prefixes)
        fill = pad_id if pad_id is not None else 0
        ids = torch.tensor(
            [[fill] * (max_len - len(p)) + list(p) for p in prefixes],
            dtype=torch.long, device=device,
        )
        mask = torch.tensor(
            [[0] * (max_len - len(p)) + [1] * len(p) for p in prefixes],
            dtype=torch.long, device=device,
        )

        if top_k is None:
            top_k = self._family_top_k(model)
        # temperature<=0 means GREEDY: transformers rejects temperature=0 with do_sample
        # (and the sampling warpers top_p/top_k are invalid greedy), so switch modes
        # rather than pass an invalid config.
        greedy = temperature is None or temperature <= 0
        gen_kwargs: dict = {
            "input_ids": ids,
            "attention_mask": mask,
            "max_new_tokens": max_new_tokens if stop_ids else min(max_new_tokens, 256),
            "do_sample": not greedy,
            "logits_to_keep": 1,
        }
        if not greedy:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = int(top_k)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids

        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        seq = out.sequences if hasattr(out, "sequences") else out
        return [
            trim_generated_row(seq[i, max_len:].tolist(), stop_ids, pad_id)
            for i in range(len(prefixes))
        ]

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        context_length: int,
        temperature: float,
        top_p: float,
        debug: Optional[Callable[[str], None]] = None,
        capture_tension: bool = False,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        stop_on_repeat: bool = False,
        min_think_tokens: int = 0,
        max_think_tokens: int = 0,
        min_p: Optional[float] = None,
        degen_stop: bool = False,
        degen_window: int = _DEGEN_WINDOW,
        degen_min_gen: int = _DEGEN_MIN_GEN,
        degen_distinct: float = _DEGEN_DISTINCT,
        degen_top_freq: float = _DEGEN_TOP_FREQ,
        no_copy_text: Optional[str] = None,
    ) -> Generator[str, None, None]:
        # Re-assert the Dynamo config: unsloth's load-time patches reset it, so a value
        # set only in load() reverts (limit falls back to 8, disable clears) and
        # generation crashes with FailOnRecompileLimitHit. Reapplying per call is cheap.
        self._apply_dynamo_settings()
        import torch
        from transformers import TextIteratorStreamer
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except ImportError:
            try:
                from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
            except ImportError:
                StoppingCriteria = None
                StoppingCriteriaList = None
        try:
            from transformers import LogitsProcessor, LogitsProcessorList
        except ImportError:
            try:
                from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
            except ImportError:
                LogitsProcessor = None
                LogitsProcessorList = None

        stop_ids = self._resolve_stop_ids(model, tokenizer)
        if debug:
            debug(f"Resolved stop token IDs: {stop_ids}")
        pad_id = self._resolve_pad_id(tokenizer)

        # For processors (e.g. Gemma4Processor) the underlying text tokenizer is
        # needed for TextIteratorStreamer and may be the only object that accepts
        # the truncation/max_length kwargs.
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        with torch.no_grad():
            try:
                inputs = text_tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=context_length
                )
            except TypeError:
                # Some tokenizer wrappers don't accept truncation/max_length
                inputs = text_tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if top_k is None:
                top_k = self._family_top_k(model)
            # temperature<=0 means GREEDY: transformers rejects temperature=0 with
            # do_sample (and top_p/top_k/min_p are sampling-only warpers), so switch to
            # do_sample=False and drop them rather than pass an invalid config. The
            # reflect/evaluation passes (branch judge, fact-dedup, self-reconcile) rely
            # on this deterministic path.
            greedy = temperature is None or temperature <= 0
            gen_kwargs: dict = {
                **inputs,
                "max_new_tokens": max_new_tokens if stop_ids else min(max_new_tokens, 256),
                "do_sample": not greedy,
                "logits_to_keep": 1,
            }
            if not greedy:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                if top_k and top_k > 0:
                    gen_kwargs["top_k"] = int(top_k)
                if min_p and min_p > 0:
                    # Layer 1 — relative-probability sampling floor (chat path opt-in).
                    # Cuts the implausible tail that seeds a degeneration excursion.
                    # Unlike top_p's fixed-mass cut, min_p tightens as the model grows
                    # confident and stays permissive at genuine choice points, so a
                    # high-entropy persona keeps its fat nucleus while the collapse-
                    # seeding tail is removed. Native transformers sampling warper.
                    gen_kwargs["min_p"] = float(min_p)
            # Loop guards — off (None) by default so live chat sampling is unchanged;
            # the reflection passes opt in (a degenerate CoT repeat there burns the
            # whole token budget and yields an unparseable, never-closed <think>).
            # The repetition penalty is NOT passed as the native kwarg: transformers'
            # RepetitionPenaltyLogitsProcessor penalizes every token present in
            # input_ids — the PROMPT included — so injected RAG blocks suppressed
            # exactly the vocabulary the prompt primes (recalled facts phrased in
            # Ava's own idiolect, her verbatim past replies), squeezing probability
            # mass into the degenerate tail. It is applied below as a
            # generated-tokens-only LogitsProcessor instead; the native kwarg
            # remains only as a fallback when the processor classes are missing.
            if (repetition_penalty and repetition_penalty > 0
                    and (LogitsProcessor is None or LogitsProcessorList is None)):
                gen_kwargs["repetition_penalty"] = float(repetition_penalty)
            if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids

            model_name = str(
                getattr(model, "name_or_path", "")
                or getattr(getattr(model, "config", None), "_name_or_path", "")
                or getattr(getattr(model, "config", None), "name_or_path", "")
            ).lower()
            # Gemma 4 / gpt-oss can spend a long time in prefill or analysis before
            # the first stream chunk; 10s surfaces as an opaque queue.Empty.
            if any(k in model_name for k in ("gemma-4", "gpt-oss", "qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6")):
                streamer_timeout = 300.0
            else:
                streamer_timeout = 120.0
            streamer = TextIteratorStreamer(
                text_tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
                timeout=streamer_timeout,
            )
            stop_event = threading.Event()
            self.last_generation_stopped_on_loop = False

            gen_kwargs["streamer"] = streamer
            if StoppingCriteria is not None and StoppingCriteriaList is not None:
                class _EventStop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return stop_event.is_set()
                criteria = [_EventStop()]
                if stop_on_repeat:
                    n_prompt_for_stop = inputs["input_ids"].shape[1]
                    backend = self

                    class _RepetitionStop(StoppingCriteria):
                        """Halt on a verbatim runaway loop without altering sampling.

                        Tracks the most recent _LOOP_NGRAM generated tokens each step
                        (an O(ngram) suffix slice; never the prompt) and stops once any
                        such span has recurred _LOOP_MIN_REPEATS times. The resulting
                        stop reads as a truncation (last token isn't EOS), so the
                        reflection runner's retry-on-unparseable picks it up."""

                        def __init__(self) -> None:
                            self._seen: dict = {}

                        def __call__(self, input_ids, scores, **kwargs):
                            n_gen = input_ids.shape[1] - n_prompt_for_stop
                            if n_gen < _LOOP_NGRAM:
                                return False
                            key = tuple(int(t) for t in input_ids[0, -_LOOP_NGRAM:])
                            c = self._seen.get(key, 0) + 1
                            self._seen[key] = c
                            if c >= _LOOP_MIN_REPEATS:
                                backend.last_generation_stopped_on_loop = True
                                if debug:
                                    debug(f"Stopped on runaway repetition "
                                          f"(span x{c}) after {n_gen} generated tokens")
                                return True
                            return False

                    criteria.append(_RepetitionStop())
                if degen_stop:
                    # Null-tolerant thresholds: a None override (e.g. a hand-edited
                    # `"chat_degen": {"window": null}` config) falls back to the
                    # module default instead of crashing the stopping criteria with
                    # `int < None` on every chat generate.
                    degen_window = (_DEGEN_WINDOW if degen_window is None
                                    else int(degen_window))
                    degen_min_gen = (_DEGEN_MIN_GEN if degen_min_gen is None
                                     else int(degen_min_gen))
                    degen_distinct = (_DEGEN_DISTINCT if degen_distinct is None
                                      else float(degen_distinct))
                    degen_top_freq = (_DEGEN_TOP_FREQ if degen_top_freq is None
                                      else float(degen_top_freq))
                    n_prompt_for_degen = inputs["input_ids"].shape[1]
                    backend = self

                    class _DegenStop(StoppingCriteria):
                        """Layer 2 — halt on *drifting* degeneration the verbatim
                        loop guard misses: an associative walk or letter-soup whose
                        windows never repeat but whose token diversity has cratered.
                        Reads a rolling suffix window (never the prompt) each step and
                        halts once its distinct-token ratio collapses or one token
                        dominates it. Content-blind (holds for any persona); reads as
                        a truncation (last token isn't EOS), like _RepetitionStop."""

                        def __call__(self, input_ids, scores, **kwargs):
                            n_gen = input_ids.shape[1] - n_prompt_for_degen
                            if n_gen < degen_min_gen or n_gen < degen_window:
                                return False
                            win = [int(t) for t in input_ids[0, -degen_window:]]
                            n = len(win)
                            distinct = len(set(win)) / n
                            top = max(win.count(t) for t in set(win)) / n
                            if distinct < degen_distinct or top > degen_top_freq:
                                backend.last_generation_stopped_on_loop = True
                                if debug:
                                    debug(f"Stopped on degeneration (distinct="
                                          f"{distinct:.2f} top={top:.2f}) after "
                                          f"{n_gen} generated tokens")
                                return True
                            return False

                    criteria.append(_DegenStop())
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)

            # Minimum-thought-length floor (companion to the family's think_prefill):
            # forbid the reasoning-close marker for the first `min_think_tokens` generated
            # steps so a prefilled-open channel can't be closed empty ("thinking on, no
            # thought"). Applied as a -inf logits mask *before* the temperature/top_p
            # warpers, so the close token simply can't be sampled until the floor passes.
            # Off by default (min_think_tokens == 0); only the live chat path opts in.
            # Anti-copy guard source: token ids of the caller-protected text (the
            # previous assistant reply — see the module comment at _NO_COPY_NGRAM).
            # Encoded once; feeds both the repetition-penalty union and the
            # _NoCopyPrevReply n-gram mask below.
            no_copy_ids: list = []
            if no_copy_text and no_copy_text.strip():
                try:
                    no_copy_ids = list(text_tokenizer.encode(
                        no_copy_text, add_special_tokens=False))
                except Exception:
                    no_copy_ids = []
            processors: list = []
            if (LogitsProcessor is not None and LogitsProcessorList is not None):
                if min_think_tokens and min_think_tokens > 0:
                    close_ids = self._resolve_think_close_ids(model, tokenizer)
                    if close_ids:
                        n_prompt_for_think = inputs["input_ids"].shape[1]

                        class _MinThinkLength(LogitsProcessor):
                            """Mask the reasoning-close marker until `min_tokens` have been
                            generated, guaranteeing a non-empty CoT once the opener is
                            prefilled. Touches only the close-marker logits; all other
                            sampling is unchanged."""

                            def __init__(self) -> None:
                                self._ids = close_ids
                                self._min = int(min_think_tokens)
                                self._n_prompt = n_prompt_for_think

                            def __call__(self, input_ids, scores):
                                if input_ids.shape[1] - self._n_prompt < self._min:
                                    scores[:, self._ids] = float("-inf")
                                return scores

                        processors.append(_MinThinkLength())
                        if debug:
                            debug(f"Min-think floor: masking close ids {close_ids} for the "
                                  f"first {min_think_tokens} generated tokens")

                if max_think_tokens and max_think_tokens > 0:
                    # Maximum-thought-length CEILING — the exact inverse of the floor
                    # above, and the answer to a *sane* CoT that simply outgrows its
                    # budget. A pass whose reasoning is still open when `max_new_tokens`
                    # runs out produces NOTHING usable: the generation is cut mid-thought,
                    # there is no answer region at all, and the caller's parser sees an
                    # empty result. Raising the budget only moves that cliff — the model
                    # thinks to the length the task invites, not to the length it was
                    # given. So instead of failing at the end, force the reasoning channel
                    # CLOSED once it has had `max_think_tokens`, leaving the remainder of
                    # the budget to the answer the caller actually asked for. A truncated
                    # thought with a real answer beats a complete thought with none.
                    #
                    # Content-blind, like the floor: it counts generated tokens and touches
                    # only the close-marker logits, so it holds for any emergent persona
                    # and any language. Disarms permanently the moment the model closes on
                    # its own — the common case, which must cost nothing.
                    close_ids_max = self._resolve_think_close_ids(model, tokenizer)
                    if close_ids_max:
                        n_prompt_for_max_think = inputs["input_ids"].shape[1]

                        class _MaxThinkLength(LogitsProcessor):
                            """Force the reasoning-close marker once the thought has run
                            `max_tokens` generated steps without closing.

                            While armed and past the ceiling, every logit except the close
                            marker(s) is masked, so the next sampled token IS the close and
                            generation continues into the answer region. Before the ceiling
                            — and forever after the channel closes — this is a no-op."""

                            def __init__(self) -> None:
                                self._ids = close_ids_max
                                self._n_prompt = n_prompt_for_max_think
                                # Decision logic lives in the pure, self-tested
                                # ThinkCeiling; this wrapper only reads the tensor and
                                # applies the mask (mirrors _DegenStop/detect_degeneration).
                                self._state = ThinkCeiling(max_think_tokens, close_ids_max)

                            def __call__(self, input_ids, scores):
                                n_gen = input_ids.shape[1] - self._n_prompt
                                last = int(input_ids[0, -1]) if n_gen > 0 else None
                                if not self._state.step(n_gen, last):
                                    return scores
                                # Mask everything but the close marker(s), preserving
                                # their relative scores, so the next sampled token IS the
                                # close and generation continues into the answer.
                                forced = scores.new_full(scores.shape, float("-inf"))
                                forced[:, self._ids] = scores[:, self._ids]
                                return forced

                        processors.append(_MaxThinkLength())
                        if debug:
                            debug(f"Max-think ceiling: forcing close ids {close_ids_max} "
                                  f"after {max_think_tokens} generated tokens")

                if repetition_penalty and repetition_penalty > 0 and repetition_penalty != 1.0:
                    n_prompt_for_penalty = inputs["input_ids"].shape[1]
                    penalty_value = float(repetition_penalty)
                    # Anti-copy union (mechanism 1): the previous reply's token ids
                    # are penalized AS IF already generated, restoring — for exactly
                    # that text — the copy deterrent the prompt exemption removed.
                    # System prompt / RAG blocks / older history stay exempt.
                    no_copy_penalty_ids = None
                    if no_copy_ids:
                        no_copy_penalty_ids = torch.tensor(
                            sorted(set(no_copy_ids)), dtype=torch.long, device=device
                        ).unsqueeze(0)

                    class _GeneratedOnlyRepetitionPenalty(LogitsProcessor):
                        """Multiplicative repetition penalty over GENERATED tokens only.

                        Same per-token arithmetic as transformers'
                        RepetitionPenaltyLogitsProcessor (positive logit ÷ penalty,
                        negative logit × penalty), but gathered from the suffix past
                        the prompt boundary instead of the whole running sequence —
                        a loop guard should discourage the model from repeating its
                        OWN output, not from using vocabulary that appears in the
                        system prompt, conversation history, or injected RAG blocks.
                        The one exception is the caller-supplied `no_copy_text` (the
                        previous assistant reply), whose ids join the gather so
                        verbatim self-regurgitation stays discouraged. No-op on the
                        first generated step when no protected text is present."""

                        def __call__(self, input_ids, scores):
                            gen_ids = input_ids[:, n_prompt_for_penalty:]
                            if no_copy_penalty_ids is not None:
                                gen_ids = torch.cat([
                                    no_copy_penalty_ids.expand(input_ids.shape[0], -1),
                                    gen_ids,
                                ], dim=1)
                            if gen_ids.shape[1] == 0:
                                return scores
                            score = torch.gather(scores, 1, gen_ids)
                            score = torch.where(
                                score < 0, score * penalty_value, score / penalty_value
                            )
                            scores.scatter_(1, gen_ids, score)
                            return scores

                    processors.append(_GeneratedOnlyRepetitionPenalty())
                    if debug:
                        debug(f"Repetition penalty {penalty_value} scoped to "
                              f"generated tokens only (prompt exempt"
                              + (f"; +{len(set(no_copy_ids))} prev-reply ids"
                                 if no_copy_ids else "") + ")")

                if len(no_copy_ids) >= _NO_COPY_NGRAM:
                    # Anti-copy n-gram mask (mechanism 2): a 1.1 penalty alone cannot
                    # break an induction copy (its distribution is near-deterministic),
                    # so additionally BAN any token that would extend a verbatim
                    # _NO_COPY_NGRAM-token match with a span of the previous reply.
                    no_copy_table = build_no_copy_table(no_copy_ids, _NO_COPY_NGRAM)
                    n_prompt_for_copy = inputs["input_ids"].shape[1]
                    copy_prefix = _NO_COPY_NGRAM - 1

                    class _NoCopyPrevReply(LogitsProcessor):
                        """Mask continuations of a verbatim copy of the protected text.

                        When the last _NO_COPY_NGRAM-1 GENERATED tokens match a span
                        of `no_copy_text`, the token(s) that would complete an
                        _NO_COPY_NGRAM-length verbatim run get -inf — capping any
                        copy below one phrase length. Paraphrase is untouched: only
                        an exact token-level span match arms the mask, and only the
                        single continuation id(s) are banned. The window never reads
                        the prompt (a fresh generation can't be judged on prompt
                        tail it didn't produce)."""

                        def __call__(self, input_ids, scores):
                            if input_ids.shape[1] - n_prompt_for_copy < copy_prefix:
                                return scores
                            for row in range(input_ids.shape[0]):
                                banned = no_copy_table.get(tuple(
                                    int(t) for t in input_ids[row, -copy_prefix:]
                                ))
                                if banned:
                                    scores[row, list(banned)] = float("-inf")
                            return scores

                    processors.append(_NoCopyPrevReply())
                    if debug:
                        debug(f"Anti-copy guard armed: {len(no_copy_table)} "
                              f"{copy_prefix}-token spans of the previous reply "
                              f"protected (ngram={_NO_COPY_NGRAM})")

            if processors:
                gen_kwargs["logits_processor"] = LogitsProcessorList(processors)

            # Cognitive-tension capture (optional): hook the LM head to read raw
            # per-token logits, reduced to (entropy, margin) on-device. Reduction
            # to per-segment stats happens server-side after the stream completes.
            self.last_token_signals = None
            self.last_think_open_prob = None
            self.last_generation_truncated = None
            n_prompt = inputs["input_ids"].shape[1]
            tension_sig: list = []
            open_probs: list = []  # per-step prob of the think-open token (if resolvable)
            lm_hook = None
            # Reset the per-token timing series for the tok/s readout. The lm head fires
            # once per generated token (prefill yields the first token's logits, each
            # decode step the next), so one timestamp per hook call is a per-token clock.
            self._gen_token_times = []
            self.last_tokens_per_sec = None
            lm_head = self._find_lm_head(model)
            if lm_head is not None:
                # Tension capture is optional; the timing hook is always on (one Python
                # append per token — negligible). When capturing, the same hook also
                # reduces the step's logits to (entropy, margin, top-2) signals.
                _tension = None
                open_id = None
                if capture_tension:
                    from core import tension as _tension
                    gen_kwargs["return_dict_in_generate"] = True
                    open_id = self._resolve_think_open_id(model, tokenizer)

                def _lm_hook(module, _inp, out):
                    self._gen_token_times.append(time.monotonic())
                    if _tension is None:
                        return
                    logits = out[0] if isinstance(out, tuple) else out
                    last = logits[0, -1, :] if logits.dim() == 3 else logits[0, :]
                    e, m, t2, op = _tension.step_signals_tensor(last, open_id)
                    tension_sig.append((e, m, t2))
                    if op is not None:
                        open_probs.append(op)

                lm_hook = lm_head.register_forward_hook(_lm_hook)

            exc_holder: dict = {}

            def _run() -> None:
                try:
                    exc_holder["out"] = model.generate(**gen_kwargs)
                except Exception as e:
                    exc_holder["e"] = e
                    # Unblock the consumer immediately. The streamer only emits its
                    # stop sentinel when generate() returns normally; on failure
                    # (e.g. CUDA OOM) `for raw_chunk in streamer` would otherwise
                    # block for the full streamer_timeout before the error surfaces —
                    # a silent multi-minute hang. end() pushes the sentinel now so the
                    # exception propagates in milliseconds.
                    try:
                        streamer.end()
                    except Exception:
                        pass

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            # Filter ChatML tokens, Gemma 1-3 turn tokens, Gemma 4 turn-end token.
            # Gemma 4 channel tokens (<|channel>, <channel|>) are left intact here so
            # _clean_response can convert them to <think>…</think> before CoT parsing.
            _SPECIAL_TOKEN_RE = re.compile(
                r"<\|[^>]+\|>"           # ChatML: <|im_start|>, <|think|>, …
                r"|<start_of_turn>|<end_of_turn>"  # Gemma 1-3
                r"|<turn\|>"             # Gemma 4 end-of-turn
            )

            closed_early = False
            streamer_timed_out = False
            try:
                for raw_chunk in streamer:
                    chunk = _SPECIAL_TOKEN_RE.sub("", raw_chunk)
                    if chunk:
                        yield chunk
            except queue_module.Empty:
                streamer_timed_out = True
            except GeneratorExit:
                closed_early = True
            finally:
                stop_event.set()
                thread.join()
                if lm_hook is not None:
                    lm_hook.remove()
                self.last_tokens_per_sec = self._compute_tokens_per_sec()
                if capture_tension:
                    self.last_token_signals = self._reduce_tension(
                        exc_holder.get("out"), tension_sig, n_prompt
                    )
                    self.last_think_open_prob = self._first_open_prob(
                        exc_holder.get("out"), open_probs, n_prompt
                    )
                self.last_generation_truncated = self._was_truncated(
                    exc_holder.get("out"), stop_ids, closed_early, streamer_timed_out
                )

        if not closed_early and "e" in exc_holder:
            err = exc_holder["e"]
            exc_holder.clear()
            # A failed generation (esp. CUDA OOM) pins its live KV-cache /
            # activation tensors through the exception's traceback frames. Holding
            # that traceback alive — even briefly, while the caller logs and reports
            # the error — keeps those tensors allocated, so every OOM ratchets VRAM
            # down until each subsequent generation fails sooner and the server
            # wedges. Log the full traceback here (so diagnostics survive), then
            # sever it and return the freed blocks to the GPU before propagating a
            # lean error.
            import traceback as _tb
            _tb.print_exception(type(err), err, err.__traceback__)
            err.__traceback__ = None
            inputs = None
            gen_kwargs = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise err
        if not closed_early and streamer_timed_out:
            raise TimeoutError(
                f"Generation produced no streamed tokens within {streamer_timeout:.0f}s "
                "(model prefill/thinking may still be running, or generation failed silently)."
            )

    # ------------------------------------------------------------------ #
    # Generation-speed helpers                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rate_from_times(times: list) -> Optional[float]:
        """Decode rate (tok/s) implied by a list of per-token timestamps: N-1 inter-token
        gaps over their total span, so prompt-prefill latency (before the first token) is
        excluded. None when there aren't two timestamps or the span is degenerate."""
        if not times or len(times) < 2:
            return None
        span = times[-1] - times[0]
        if span <= 0:
            return None
        return (len(times) - 1) / span

    def _compute_tokens_per_sec(self) -> Optional[float]:
        """Whole-generation average tok/s, finalized in stream_generate's finally block."""
        return self._rate_from_times(list(self._gen_token_times))

    def current_tokens_per_sec(self, window_s: float = 5.0) -> Optional[float]:
        """Live decode rate over the last ``window_s`` seconds — read mid-generation by
        the streaming layer to drive the status-line tok/s readout. Falls back to the
        whole-generation average when the window holds fewer than two tokens."""
        times = list(self._gen_token_times)  # snapshot (appended from the generate thread)
        if len(times) < 2:
            return None
        now = time.monotonic()
        recent = [t for t in times if now - t <= window_s]
        return self._rate_from_times(recent if len(recent) >= 2 else times)

    # ------------------------------------------------------------------ #
    # Cognitive-tension helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _was_truncated(out: Any, stop_ids: list, closed_early: bool,
                       timed_out: bool) -> Optional[bool]:
        """Whether generation hit the token cap (True) vs. stopped on EOS (False).

        ``None`` when undeterminable: no stop ids configured, no captured output, or
        the stream was cut short by cancellation/timeout (not a natural end). Read by
        the reflection runner to flag a likely-truncated revision pass (and decide a
        one-shot retry)."""
        if closed_early or timed_out or out is None or not stop_ids:
            return None
        try:
            seq = out.sequences if hasattr(out, "sequences") else out
            last_id = int(seq[0, -1])
        except Exception:
            return None
        return last_id not in set(stop_ids)

    def _find_lm_head(self, model: Any):
        """The output projection module to hook for raw logits (cached per model)."""
        cached = self._lm_head_cache
        if cached is not None and cached[0] == id(model):
            return cached[1]
        head = None
        try:
            head = model.get_output_embeddings()
        except Exception:
            head = None
        if head is not None and not hasattr(head, "register_forward_hook"):
            head = None
        self._lm_head_cache = (id(model), head)
        return head

    def _first_open_prob(self, out: Any, open_probs: list, n_prompt: int) -> Optional[float]:
        """Probability the *first generated step* put on the think-open token.

        ``open_probs`` is aligned with the per-step tension capture, so it carries the
        same per-token stride (k=2 on gemma-4: the LM head fires twice per generated
        token, and the meaningful distribution is the last of each group). We mirror
        ``_reduce_tension``'s stride dedup and take the representative of generated
        token 0 — the step whose distribution chose whether to open the reasoning
        channel. Graceful None if the capture didn't line up.
        """
        if out is None or not open_probs:
            return None
        try:
            n = len(out.sequences[0, n_prompt:].tolist())
        except Exception:
            return None
        if n == 0 or len(open_probs) % n != 0:
            return None
        stride = len(open_probs) // n
        idx = stride - 1
        if idx >= len(open_probs):
            return None
        try:
            return float(open_probs[idx])
        except Exception:
            return None

    def _reduce_tension(self, out: Any, signals: list, n_prompt: int) -> Optional[dict]:
        """Dedup the per-token signals against the generated tokens and bulk-sync.

        The LM head may fire k times per generated token (k=2 on gemma-4); when the
        capture count is an exact multiple of the token count we keep one per token,
        otherwise we skip (graceful None) rather than emit a misaligned trace.
        """
        import torch
        if out is None or not signals:
            return None
        try:
            gen_ids = out.sequences[0, n_prompt:].tolist()
        except Exception:
            return None
        n = len(gen_ids)
        if n == 0 or len(signals) % n != 0:
            return None
        stride = len(signals) // n
        chosen = signals[stride - 1::stride]
        if len(chosen) != n:
            return None
        entropies = torch.stack([e for e, _, _ in chosen]).cpu().tolist()
        margins = torch.stack([m for _, m, _ in chosen]).cpu().tolist()
        top2_ids = torch.stack([t for _, _, t in chosen]).cpu().tolist()
        return {
            "entropies": entropies, "margins": margins,
            "token_ids": gen_ids, "top2_ids": top2_ids,
        }


if __name__ == "__main__":
    # GPU-free self-test of the runaway-repetition detector (python -m core.inference_backend).
    S = list(range(100, 118))
    assert detect_ngram_loop(list(range(0, 40)) + S * 30), "contiguous loop missed"

    body = list(range(200, 216))
    seq = list(range(0, 30))
    for digit in range(2, 25):
        seq += [900, 901, digit, 902] + body  # incrementing 'Note N:' separator
    assert detect_ngram_loop(seq), "incrementing-separator loop missed"

    import random
    random.seed(1)
    assert not detect_ngram_loop([random.randint(0, 5000) for _ in range(2000)]), \
        "false positive on normal prose"
    phrase = list(range(300, 314))
    assert not detect_ngram_loop(list(range(0, 50)) + phrase + list(range(60, 120)) + phrase), \
        "false positive on a phrase quoted twice"
    print("inference_backend repetition-detector self-test: OK")

    # detect_degeneration: the drifting-collapse guard the verbatim detector misses.
    import random as _r
    _r.seed(2)
    # Letter-soup / low-distinct tail (a handful of tokens cycled) — must halt.
    assert detect_degeneration(list(range(0, 200)) + [7, 8, 9] * 40), \
        "low-distinct degeneration missed"
    # Single token dominating the window — must halt.
    assert detect_degeneration(list(range(0, 200)) + [5] * 40 + list(range(1, 25))), \
        "single-token domination missed"
    # A drifting associative walk that never repeats an n-gram (detect_ngram_loop
    # is blind to it) but whose window diversity has cratered — must halt.
    assert detect_degeneration(list(range(0, 150)) + list(range(600, 610)) * 8), \
        "drifting low-diversity walk missed"
    # Normal high-diversity prose — must NOT halt.
    assert not detect_degeneration([_r.randint(0, 5000) for _ in range(400)]), \
        "false positive on normal prose"
    # Below the min-gen floor — must NOT halt even if degenerate.
    assert not detect_degeneration([5] * 64), "judged before min-gen floor"
    print("inference_backend degeneration-detector self-test: OK")

    # Anti-copy guard: the n-gram table + continuation mask (pure forms).
    prev_reply = list(range(400, 430))                    # 30-token "previous reply"
    table = build_no_copy_table(prev_reply)
    # A generation that has verbatim-copied 7 tokens of the reply must have the
    # 8th (and only the 8th) banned.
    assert banned_continuations(prev_reply[3:10], table) == {prev_reply[10]}, \
        "verbatim 7-token copy must ban exactly the continuation id"
    # Same tokens in a different order — paraphrase — must arm nothing.
    assert banned_continuations(list(reversed(prev_reply[3:10])), table) == set(), \
        "reordered (paraphrased) tokens must not be banned"
    # A tail shorter than the matching window must arm nothing.
    assert banned_continuations(prev_reply[:5], table) == set(), \
        "sub-window tail must not be judged"
    # A span occurring twice with different continuations bans both.
    twice = [1, 2, 3, 4, 5, 6, 7, 8] + [9] + [1, 2, 3, 4, 5, 6, 7, 10]
    t2 = build_no_copy_table(twice)
    assert {8, 10} <= banned_continuations([1, 2, 3, 4, 5, 6, 7], t2), \
        "recurring span must ban every observed continuation"
    # A protected text shorter than one n-gram yields an empty table (guard unarmed).
    assert build_no_copy_table(list(range(5))) == {}, \
        "short protected text must produce no spans"
    print("inference_backend anti-copy-guard self-test: OK")

    # ThinkCeiling: force the reasoning channel closed in time to write an answer.
    CLOSE = 101   # gemma-4's <channel|>
    def _run(ceiling, tokens):
        """Feed a generated token sequence; return the step index that forced a close."""
        state = ThinkCeiling(ceiling, [CLOSE])
        for i, tok in enumerate(tokens):
            # step() is called BEFORE token i is sampled: n_generated == i, and the
            # newest token is the previous one.
            if state.step(i, tokens[i - 1] if i > 0 else None):
                return i
        return None
    # A thought that runs past the ceiling is forced closed exactly at it.
    assert _run(10, list(range(1, 40))) == 10, "ceiling must force the close at the cap"
    # A model that closes on its own before the ceiling is never touched...
    assert _run(10, [1, 2, CLOSE] + list(range(1, 40))) is None, \
        "self-closed channel must never be forced"
    # ...up to and including the step before the ceiling.
    assert _run(4, [1, 2, 3, CLOSE, 5, 6, 7, 8]) is None, \
        "close on the step before the ceiling must disarm it"
    # A close the model was about to sample ON the ceiling step still counts as forced:
    # the guard only ever sees tokens already generated, so it cannot know. Harmless —
    # the token it forces is the one that was coming anyway.
    assert _run(3, [1, 2, 3, CLOSE, 5, 6, 7, 8]) == 3, \
        "ceiling fires on what is already generated, not on what was about to be"
    # Disarming is permanent: a later close marker inside the ANSWER (or a second
    # channel) must not re-arm and mask the rest of the generation.
    assert _run(5, [1, CLOSE, 2, 3, 4, 5, 6, 7, 8, 9]) is None, \
        "ceiling must stay disarmed for the whole answer region"
    # Unconfigured / unresolvable close markers leave the guard off entirely.
    assert _run(0, list(range(1, 40))) is None, "zero ceiling must be a no-op"
    assert ThinkCeiling(10, []).step(50, 7) is False, \
        "no close ids (family has none) must be a no-op"
    print("inference_backend think-ceiling self-test: OK")

    # trim_generated_row: batched-row post-processing (pure).
    assert trim_generated_row([5, 6, 7, 2, 0, 0], [2], 0) == [5, 6, 7, 2], \
        "row must end at its first stop token (kept), pad tail cut"
    assert trim_generated_row([5, 6, 7], [2], 0) == [5, 6, 7], \
        "row that hit the cap (no stop token) must pass through whole"
    assert trim_generated_row([5, 2, 8, 2], [2], 2) == [5, 2], \
        "tokens after the first stop must be cut even if not pad"
    assert trim_generated_row([5, 6, 0, 0], [], 0) == [5, 6], \
        "with no stop ids, the pad tail must still be stripped"
    assert trim_generated_row([2], [2], 2) == [2], "lone stop token kept"
    assert trim_generated_row([], [2], 0) == [], "empty row passes through"
    print("inference_backend batch-row-trim self-test: OK")
