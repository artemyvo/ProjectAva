"""Typed owners for the inference server's shared, process-singleton state.

`server.py` used to keep three module-level `dict`s (`_model_state`,
`_session_state`, `_encounter_state`) whose contents were read and mutated by
stringly-typed key access in ~90 places. That made local reasoning hard (you
had to grep every writer to know what a function assumed) and offered no
autocomplete / jump-to-def / typo protection.

These three dataclasses are those same containers, now typed. They are still
process-singletons (`runtime` / `session` / `encounter` below) — one GPU, one
active client, one encounter at a time — and they are still *mutated in place*
(never rebound), so a by-reference consumer like `agentic.CleanBaseSession`
keeps working unchanged: swapping the model means assigning `runtime.model`,
exactly as it used to mean `_model_state["model"] = ...`.

Field defaults match the original dict initial values verbatim, so the
attribute form is behaviourally identical to the old `dict.get(key, default)`
access it replaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # keep this module import-light (no transformers/faiss pull)
    from core.chat_logger import ChatLogger
    from core.rag_engine import RagEngine
    from core.reflection_writer import ReflectionWriter


@dataclass
class ModelRuntime:
    """The currently loaded model and the spec it was loaded from.

    Mutated in place by the load/unload handlers and, transiently, by
    `agentic.CleanBaseSession` (which swaps the adapter model for the bare base
    and back). `adapter_id` is set once `train_cycle` points the server at a
    persistent LoRA adapter; the base `model_id` stays frozen.
    """
    model: Any = None
    tokenizer: Any = None
    model_id: Optional[str] = None
    adapter_id: Optional[str] = None
    # Chat / default software budget. Chat and other live paths pack to this.
    context_length: int = 32768
    # Reflection software budget AND the physical max_seq_length the model is
    # loaded at (the model is always loaded at max(context_length,
    # reflect_context_length), so reflection can pack a larger window than chat
    # while chat stays capped so its transcripts always fit a later reflection).
    # Invariant: reflect_context_length >= context_length == the physical load.
    reflect_context_length: int = 32768
    # Base-model load precision: "16bit" | "8bit" | "4bit" (from server_config's
    # base_quant). Empty when unset — the backend then keeps its historical 4-bit
    # default. Recorded so status/loaded can report the precision Ava is running at.
    base_quant: str = ""


@dataclass
class Session:
    """Per-connection chat session state, server-owned and global.

    A new client supersedes the incumbent rather than getting its own copy —
    switching desktop<->laptop just moves the viewport over this one session.
    `logger` / `rag` / `reflection_writer` are lazily initialised on first use.
    `surfaced_keys` tracks which open reflection questions Ava has already
    raised this session so she doesn't re-pose them.
    """
    conversation: list = field(default_factory=list)
    logger: "Optional[ChatLogger]" = None
    rag: "Optional[RagEngine]" = None
    reflection_writer: "Optional[ReflectionWriter]" = None
    system_prompt: str = ""       # loaded at startup from prompts/chat_prompt.txt
    surface_template: str = ""     # loaded at startup from prompts/surface_prompt.txt
    surfaced_keys: list = field(default_factory=list)
    user: str = ""                # name of the user currently talking to Ava


@dataclass
class EncounterState:
    """Ava-meets-a-fellow-AI encounter state + append-only event buffer.

    Like a reflection run, an encounter monopolises the single GPU worker, so
    live chat/reflection are refused while `active`. The client polls `events`
    (each `{seq, type, ts, ...}`) by `seq`. Guarded by `server._encounter_lock`.
    """
    active: bool = False
    status: str = "idle"           # idle | running | completed | stopped | failed
    seq: int = 0
    events: list = field(default_factory=list)   # [{seq, type, ts, ...}]
    meta: dict = field(default_factory=dict)      # {name, model, url, turns, session_file, started_at}
    error: str = ""
    turns_done: int = 0


# Process-singleton instances. Import and mutate these; never rebind them
# (attribute assignment only), so by-reference holders stay coherent.
runtime = ModelRuntime()
session = Session()
encounter = EncounterState()


def reflect_window(default: int = 8192) -> int:
    """The window a reflect-lane generation actually gets — the PHYSICAL load.

    ``max(context_length, reflect_context_length)``: the model is loaded once at that
    max (``server.main()`` / ``handle_load``, preserved across the clean-base swap by
    ``agentic.CleanBaseSession``), and the reflect-lane generate resolves
    ``max_new_tokens`` against it — so any caller sizing a token reserve for that lane
    must clamp against the same number or its clamp measures the wrong window.

    Lives here because this module owns both fields. It is shared rather than restated
    per subsystem: ``generation`` (which carries the long note on why the chat budget was
    the wrong one), ``synthesis`` and ``checkin`` each derive a reserve from it, and three
    private copies would be three chances to drift.

    *default* is returned only when neither field is set — i.e. no model is loaded. It
    can never over-promise: ``_resolve_max_new_tokens`` clamps every reserve against the
    prompt's actual remaining room.
    """
    window = max(int(runtime.reflect_context_length or 0),
                 int(runtime.context_length or 0))
    return window or int(default)


def reflect_output_reserve(requested: int, *, min_input: int,
                           min_fraction: float = 0.5) -> int:
    """Clamp a reflect-lane pass's *requested* generation budget to the loaded window.

    The shared shape behind `checkin`'s two reserves and `synthesis`'s opener reserve:
    a pass that owns the box for its duration asks for what its thought and answer
    actually need, and this holds it inside the window.

    Two guards, because a single one fails from one side or the other:

      * *min_input* — headroom left for the prompt on a roomy window, so a pass is not
        sized as if its input were free.
      * *min_fraction* — but subtracting a fixed input budget is the wrong guard on a
        SMALL window, where it can leave less than the flat constant it replaced (a
        recap pass wanting 8k input on an 8192 window would be clamped to the 512
        floor). Below that crossover the pass takes a share of the window instead.

    Neither can over-promise: the reserve never squeezes the prompt, only the other way
    round — these callers pass no ``input_token_limit``, so ``_resolve_max_new_tokens``
    clamps the result again against the room the assembled prompt actually left.

    ``synthesis._analysis_output_reserve`` deliberately does NOT come here: its reserve
    is carved out of the transcript budget (it derives ``input_limit`` from it and hands
    that to the chunker), so its fraction is a ceiling where this one is a floor.
    """
    window = reflect_window()
    ceiling = max(512, window - int(min_input), int(window * float(min_fraction)))
    return max(1, min(max(1, int(requested)), ceiling))
