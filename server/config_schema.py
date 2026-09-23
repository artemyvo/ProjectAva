#!/usr/bin/env python3
"""The ONE schema of ``server/server_config.json`` — every documented knob, its dotted
path, its kind and its in-code default — shared by two consumers:

  * ``settings.py`` (the PyQt6 editor) renders it as tabs of fields;
  * the inference server's boot back-fill (``server.py::_backfill_server_config`` →
    :func:`backfill_defaults`) writes every knob that is ABSENT from the box's config into
    the file with its default, so an operator can find each new setting by opening the
    JSON after the next run, rather than by reading the code that reads it.

Before 2026-09-17 the back-fill was three hand-listed training keys, and every other knob
(check-in, the facts tree, the associative library, logging, …) existed only as a
``cfg.get("x", default)`` somewhere in ``core/``. The schema already carried the defaults
for the editor, so the generic back-fill is that same list walked once at boot.

Two contracts keep the back-fill **behaviour-preserving**:

  * A knob is written with the value the reader would have used had it stayed absent, so
    the running box changes nothing — only the file grows. That is why each default here
    must match the reader's own (the self-test cross-checks the ones whose modules are
    import-light; the tooltip cites the rest).
  * A key whose *absence* and whose *null* mean different things to its reader is never
    written as null. ``chat_repetition_penalty`` absent ⇒ 1.1, null ⇒ off; so its default
    here is 1.1. A field with ``default=None`` is skipped unless ``write_null`` says its
    reader treats null exactly like absent (the ``chat_degen.*`` overrides,
    ``graph.til_max_age_days``). ``backfill=False`` marks a key that is box STATE rather
    than a knob (``model_id``, ``adapter_id`` — written by the operator / the train cycle;
    a fresh box must not wake up auto-loading a 31B model because the schema named one).
    ``derive`` computes a default from the rest of the config (``reflect_context_length``
    == ``context_length`` ⇒ no split, the historical back-fill).

Pure stdlib, no PyQt6, no torch — importable from the server process, the editor and the
self-test alike: ``python config_schema.py`` (or ``python -m config_schema`` from
``server/``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

# Known HF base-model ids the editor's picker offers (editable — any id may be typed).
# Mirrors client/ui/debug_widget.py::_KNOWN_BASE_MODELS.
KNOWN_MODELS = [
    "unsloth/gemma-4-31B-it",
    "unsloth/Qwen3.6-27B",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
]


# ──────────────────────────────────────────────────────────────────────────────
# Field schema
# ──────────────────────────────────────────────────────────────────────────────

# kinds:
#   str        free text                      combo     fixed+editable choices
#   path       free text + Browse             int_combo fixed integer choices
#   bool       checkbox                       int / float
#   opt_int    integer or empty→null          opt_float float or empty→null
#   floatlist  comma-separated floats         strlist   comma-separated strings

@dataclass
class Field:
    path: str                     # dotted key path into the config dict
    label: str
    kind: str
    default: Any = None
    choices: Optional[list] = None
    tooltip: str = ""
    # Back-fill policy (see the module docstring).
    backfill: bool = True         # False: box state, never written by the back-fill
    write_null: bool = False      # True: the reader treats null like absent, so a
                                  # None default may be written as null (discoverable)
    derive: Optional[Callable[[dict], Any]] = None  # default computed from the config


@dataclass
class Tab:
    name: str
    intro: str
    fields: list[Field] = field(default_factory=list)


def _reflect_from_context(cfg: dict) -> Any:
    """``reflect_context_length`` back-fills == ``context_length`` (no split), never the
    static default: the model loads at max(context, reflect), so any other value would
    change the physical load of a box that never asked for one."""
    try:
        return int(cfg["context_length"]) if "context_length" in cfg else None
    except (TypeError, ValueError):
        return None


SCHEMA: list[Tab] = [
    Tab(
        "Model & precision",
        "Base weights, quantisation, and the context windows. The base <b>model_id</b> is "
        "frozen once weights exist; training only ever repoints <b>adapter_id</b>. "
        "Changes take effect on the next server restart / model reload.",
        [
            Field("model_id", "Base model id", "combo", "unsloth/gemma-4-31B-it",
                  choices=KNOWN_MODELS, backfill=False,
                  tooltip="HuggingFace repo id of the frozen base model. Any id may be "
                          "typed. Raising precision to 8/16-bit needs the full-precision "
                          "base repo to be downloadable (a bnb-4bit repo can't be upcast). "
                          "Box state — never back-filled (an empty id means no auto-load)."),
            Field("base_quant", "Base load precision", "combo", "",
                  choices=["", "16bit", "8bit", "4bit"],
                  tooltip="Base-model load precision. Empty ⇒ the backend's historical "
                          "4-bit default. With an adapter present, 8/16-bit loads the "
                          "full-precision base and attaches the LoRA on top. "
                          "(server.py::_resolve_base_quant)"),
            Field("adapter_id", "Active LoRA adapter", "path", "", backfill=False,
                  tooltip="Absolute path to the active LoRA adapter dir. Set/rotated by "
                          "the training cycle; edit to roll back to an earlier persona's "
                          "adapter. Empty ⇒ base model only. Box state — never "
                          "back-filled."),
            Field("context_length", "Chat context length", "int", 32768,
                  tooltip="Chat / default token budget. Chat transcripts are capped here "
                          "so they always fit a later reflection window. Default 32768 "
                          "(what a fresh server_config.json is created with)."),
            Field("reflect_context_length", "Reflection context length", "int", 32768,
                  derive=_reflect_from_context,
                  tooltip="Reflection window AND the physical max_seq_length the model "
                          "loads at (model is loaded once at max(context, reflect)). Raise "
                          "to let a reflection run pack a bigger window than chat. "
                          "Back-filled == context_length (no split) when absent."),
        ],
    ),
    Tab(
        "Chat / generation",
        "Live-chat degeneration guards. The repetition penalty is chat / ephemeral / "
        "encounter only; min_p and the degeneration guard ALSO cover the reflect/agentic "
        "generate factories (since 2026-07-31). All are content-blind mechanisms, so they "
        "hold for any emergent persona. See server.py main() and core/inference_backend.py.",
        [
            Field("chat_repetition_penalty", "Repetition penalty", "opt_float", 1.1,
                  tooltip="Mild live-chat repetition penalty on GENERATED tokens only "
                          "(the prompt, incl. injected RAG, is exempt; the previous reply "
                          "is penalized as part of the anti-copy guard). Default 1.1 when "
                          "the key is ABSENT (server.py::_CHAT_REPETITION_PENALTY); 1.0 or "
                          "empty(null) disables it — the halt-only stop_on_repeat guard "
                          "stays on either way. Only values > 1.0 take effect."),
            Field("chat_min_p", "min_p sampling floor", "opt_float", 0.02,
                  tooltip="Layer-1 relative-probability sampling floor: cuts the "
                          "implausible tail that seeds a collapse while leaving a "
                          "high-entropy persona's nucleus intact. Default 0.02; empty/≤0 "
                          "disables."),
            Field("chat_degen_guard", "Drifting-degeneration guard", "bool", True,
                  tooltip="Layer-2 halt for an associative-walk / letter-soup runaway "
                          "(distinct-token-ratio / single-token-dominance over a rolling "
                          "window) that the verbatim stop_on_repeat guard is blind to. "
                          "Default on."),
            Field("chat_degen.window", "  degen: window", "opt_int", None, write_null=True,
                  tooltip="Optional override of the degeneration guard's rolling-window "
                          "size. Empty/null ⇒ backend default."),
            Field("chat_degen.min_gen", "  degen: min tokens", "opt_int", None,
                  write_null=True,
                  tooltip="Optional override: minimum generated tokens before the guard "
                          "may fire. Empty/null ⇒ backend default."),
            Field("chat_degen.distinct_ratio", "  degen: distinct ratio", "opt_float", None,
                  write_null=True,
                  tooltip="Optional override: distinct-token ratio below which the window "
                          "is judged degenerate. Empty/null ⇒ backend default."),
            Field("chat_degen.top_freq", "  degen: top-token freq", "opt_float", None,
                  write_null=True,
                  tooltip="Optional override: single-token dominance fraction that trips "
                          "the guard. Empty/null ⇒ backend default."),
        ],
    ),
    Tab(
        "Training (SFT)",
        "Offline from-scratch LoRA build parameters (training/train_cycle.py, defaults in "
        "training/decay.py). The adapter is rebuilt on the frozen base every cycle. "
        "Per-row wall-clock multipliers (Consolidation tab) scale on top of these.",
        [
            Field("lora_r", "LoRA rank (r)", "int_combo", 32,
                  choices=[8, 16, 32, 64, 128],
                  tooltip="Adapter capacity: get_peft_model r for the from-scratch LoRA "
                          "fit. Default 32 (decay.TRAIN_LORA_R_DEFAULT); higher = more "
                          "capacity + VRAM. Scaling is rank-stabilized (use_rslora, "
                          "gamma = alpha/sqrt(r) with alpha fixed at 4), calibrated so "
                          "gamma == 1.0 at r=16 — so changing rank adds/removes capacity "
                          "WITHOUT changing the effective learning rate, and train_lr "
                          "stays valid across ranks. A CLI --lora-r / Sleep "
                          "train_params.lora_r overrides at run time."),
            Field("train_lr", "Base / peak SFT LR", "float", 8e-6,
                  tooltip="Base/peak learning rate the per-row multipliers + schedule "
                          "shape scale on top of. Default 8e-6 (decay.TRAIN_LR_DEFAULT). "
                          "A CLI --lr / Sleep train_params.lr overrides at run time."),
            Field("train_lr_schedule", "LR schedule shape", "combo", "age_ramp",
                  choices=["age_ramp", "triangular"],
                  tooltip="Global LR shape layered on the per-row multipliers. "
                          "'age_ramp' = DEFAULT: flat single pass (LR = base × row_mult), "
                          "one epoch, the age ramp alone weighting the rows. 'triangular' "
                          "= warmup + plateau + decay trapezoid (order-neutral, forces "
                          "epochs = plateau+2)."),
            Field("train_plateau_epochs", "Plateau (hold) epochs", "int", 3,
                  tooltip="Full-LR hold epochs in the triangular schedule (forces total "
                          "epochs = this + 2). 0 = minimal trapezoid (warmup+decay only, "
                          "2 epochs). Default 3 (decay.TRAIN_PLATEAU_EPOCHS_DEFAULT). "
                          "IGNORED under the default 'age_ramp' schedule."),
            Field("train_max_seq_length", "Training max seq length", "int", 8192,
                  tooltip="Offline-training sequence cap (clamped to context_length). "
                          "Drives the answer-preserving message-level truncation (drop "
                          "oldest history turns, keep system + final exchange + target); "
                          "a row still over cap is quarantined. It no longer bounds the "
                          "fused CE-loss chunk — that is pinned by "
                          "UNSLOTH_CE_LOSS_TARGET_GB regardless of sequence length — so "
                          "lower it for an activation-side OOM only. Training-only — "
                          "inference always uses full context_length. Default 8192."),
        ],
    ),
    Tab(
        "Consolidation / decay",
        "Wall-clock consolidation curve (consolidation.wall_clock, parsed by "
        "training/decay.py::WallClockConfig). Controls how a chat's age scales its "
        "training weight and the verbatim-to-gist RAG handoff. Hours are wall-clock "
        "hours since the chat vs the build time.",
        [
            Field("consolidation.decay_curve", "Decay curve", "combo", "linear",
                  choices=["linear"],
                  tooltip="Shared decay curve name (ConsolidationConfig)."),
            Field("consolidation.wall_clock.rag_only_window_h", "RAG-only window (h)", "float", 24.0,
                  tooltip="LR multiplier is 0 below this age (models the pre-reflection "
                          "span — RAG only, no trainable row). Default 24h."),
            Field("consolidation.wall_clock.lora_cap_age_h", "LoRA cap age (h)", "float", 72.0,
                  tooltip="Age at which the per-row LR multiplier reaches the ramp cap "
                          "(~3d). Default 72h."),
            Field("consolidation.wall_clock.rag_cap_age_h", "RAG handoff age (h)", "float", 96.0,
                  tooltip="Age at which verbatim-chat RAG reaches hard 0, gist reaches "
                          "peak weight, and persona reaches its floor (~4d). Default 96h."),
            Field("consolidation.wall_clock.gist_cap_age_h", "Gist cap age (h)", "float", 192.0,
                  tooltip="Age at which the per-conversation gist/summary reaches its "
                          "permanent floor (~8d). Default 192h."),
            Field("consolidation.wall_clock.rag_floor_weight", "Persona RAG floor", "float", 0.2,
                  tooltip="Permanent retrieval floor for persona anchors. Verbatim chat "
                          "does not use this floor. Default 0.2."),
            Field("consolidation.wall_clock.gist_floor_weight", "Gist RAG floor", "float", 0.2,
                  tooltip="Permanent gist/summary retrieval floor reached exactly at the "
                          "gist cap age. Default 0.2."),
            Field("consolidation.wall_clock.base_lr", "Wall-clock base LR", "float", 1e-5,
                  tooltip="Base LR the per-row wall-clock multiplier scales (REBUILD §1). "
                          "Default 1e-5. Note: train_lr (Training tab) is the actual peak "
                          "SFT LR used by train_cycle."),
            Field("consolidation.wall_clock.lr_ramp", "LR ramp sample points", "floatlist",
                  [1.0, 2.0, 4.0],
                  tooltip="Explicit LR-ramp sample points across [rag_only_window, "
                          "lora_cap]; the cap is the last value. Comma-separated. "
                          "Default 1.0, 2.0, 4.0."),
            Field("consolidation.wall_clock.contamination.enabled", "Contamination: enabled",
                  "bool", True,
                  tooltip="Cap-age user-contamination: a cap-age exchange also emits a "
                          "user-turn-unmasked copy so voice entrains at a small dose. "
                          "Default on. (§5e)"),
            Field("consolidation.wall_clock.contamination.dose", "Contamination: dose", "float", 1.0,
                  tooltip="Unmasked-copy LR multiplier at cap (the response keeps the full "
                          "cap; voice entrains at this dose). Default 1.0."),
            Field("consolidation.wall_clock.contamination.additive", "Contamination: additive",
                  "bool", False,
                  tooltip="False: split cap → (cap−dose) masked + dose unmasked. True: keep "
                          "cap masked + add an extra dose unmasked copy. Default False."),
            Field("consolidation.wall_clock.contamination.min_user_chars", "Contamination: min user chars",
                  "int", 100,
                  tooltip="Minimum user-turn length (chars) for an exchange to emit a "
                          "contamination copy. Default 100."),
            Field("consolidation.wall_clock.contamination.fold", "Contamination: fold pair",
                  "bool", False,
                  tooltip="Fold the cap-age contamination PAIR into ONE per-token-weighted "
                          "row (~40% fewer rows). Default False."),
            Field("consolidation.wall_clock.wander_rag.weights", "Wander RAG weights", "floatlist",
                  [0.4, 0.3, 0.2, 0.1],
                  tooltip="Age-stepped retrieval weights for the wander channel. "
                          "Comma-separated. Default 0.4, 0.3, 0.2, 0.1."),
            Field("consolidation.wall_clock.wander_rag.step_h", "Wander RAG step (h)", "float", 24.0,
                  tooltip="Hours per wander-RAG weight step. Default 24h."),
            Field("consolidation.wall_clock.fresh_window.droop_frac", "Fresh: droop frac", "float", 0.5,
                  tooltip="Freshness droop fraction for a just-happened chat. Default 0.5; "
                          "set 0 to disable the fresh-window adjustment."),
            Field("consolidation.wall_clock.fresh_window.horizon_h", "Fresh: horizon (h)", "float", 24.0,
                  tooltip="Freshness horizon in hours. Default 24h."),
            Field("consolidation.wall_clock.recollection.hold_h", "Recollection: hold (h)", "float", 96.0,
                  tooltip="A [recollection] (what Ava now makes of a revisited chat) holds "
                          "full retrieval weight for this many hours after it was written "
                          "— a per-RECORD clock, not the chat's age. Default 96h."),
            Field("consolidation.wall_clock.recollection.cap_age_h", "Recollection: cap age (h)", "float", 192.0,
                  tooltip="Age at which a recollection reaches its floor (~8d). Default 192h."),
            Field("consolidation.wall_clock.recollection.floor_weight", "Recollection: floor", "float", 0.3,
                  tooltip="Permanent recollection retrieval floor — deliberately ABOVE the "
                          "gist floor so a current reading outranks the faded gist beside "
                          "it. Default 0.3."),
        ],
    ),
    Tab(
        "Check-in",
        "The autonomous check-in idle job (core/checkin.py): after ONE PERSON has been "
        "quiet long enough, Ava reviews her recent chats with them and may reach out. "
        "Every knob is per person. Absent ⇒ the defaults below.",
        [
            Field("checkin.silence_threshold_hours", "User-silence threshold (h)", "float", 5.0,
                  tooltip="How long that person must be quiet (measured from their own "
                          "transcripts on disk, not the idle clock) before an autonomous "
                          "check-in may reach out to them. Default 5h."),
            Field("checkin.recent_chats", "Recent chats reviewed", "int", 5,
                  tooltip="How many of their recent conversations she recaps to decide "
                          "whether to reach out. Default 5."),
            Field("checkin.min_user_turns", "Min real user turns per chat", "int", 1,
                  tooltip="How much of the USER a session must contain to count as one of "
                          "their recent chats (real user turns, counted structurally). 1 "
                          "admits every genuine exchange; 2 demands back-and-forth and "
                          "drops an opener-plus-one-line thread. Default 1."),
            Field("checkin.max_users", "Max people per wake", "int", 3,
                  tooltip="How many people may reach a GENERATION in one wake; the "
                          "silence/staleness gates are free disk reads and spend no slot. "
                          "The rest are deferred to the next wake. Default 3."),
            Field("checkin.max_silence_days", "Max silence (days)", "float", 30.0,
                  tooltip="Past this someone STOPPED talking rather than went quiet and is "
                          "no longer a candidate — without it a dormant contact would be "
                          "written to daily forever. 0 ⇒ no ceiling. The manual Sleep-tab "
                          "button ignores it. Default 30."),
            Field("checkin.max_new_tokens", "Decision pass budget (tokens)", "int", 12288,
                  tooltip="Generation reserve of the decision pass, clamped against the "
                          "reflect window leaving 4096 for its input "
                          "(runtime_state.reflect_output_reserve). Default 12288."),
            Field("checkin.summary_max_new_tokens", "Recap pass budget (tokens)", "int", 8192,
                  tooltip="Generation reserve of each per-chat recap pass, clamped against "
                          "the reflect window leaving 8192 for the transcript it reads. "
                          "Default 8192."),
        ],
    ),
    Tab(
        "Reach-out",
        "The other reach-out passes and their pacing (core/outreach.py, core/synthesis.py, "
        "the Revisit head-phase in core/reflection_service.py, the stale-opener sweep in "
        "core/chat_worklog.py). Absent ⇒ the defaults below.",
        [
            Field("outreach.min_reask_hours", "Outreach: re-ask gap (h)", "float", 72.0,
                  tooltip="How long before Ava may raise the SAME open ask again on her own "
                          "initiative — paces the ceiling-exempt meta pool so it can't "
                          "round-robin every few hours. Outreach only; the passive "
                          "session-opening path keeps no gap. Default 72h."),
            Field("outreach.max_new_tokens", "Outreach: decision budget (tokens)", "int", 12288,
                  tooltip="Generation reserve of the outreach decision pass (thought + "
                          "DECISION + the opener), clamped against the reflect window "
                          "leaving 4096 for its input (runtime_state.reflect_output_reserve). "
                          "A pass that hits the cap sends nothing and holds that ask out of "
                          "selection for 24 h. Default 12288."),
            Field("synthesis.min_age_days", "Synthesis: min chat age (days)", "float", 7.0,
                  tooltip="A chat must be at least this old to be re-read by the synthesis "
                          "pass. Default 7."),
            Field("synthesis.min_resynth_days", "Synthesis: re-synth gap (days)", "float", 7.0,
                  tooltip="Anti-fixation gate: a chat synthesized within this many days is "
                          "not picked again (folded from data/hot/synthesis/synthesized.jsonl). "
                          "Default 7."),
            Field("synthesis.max_new_tokens", "Synthesis: analysis budget (tokens)", "int", 12288,
                  tooltip="Generation reserve of the analysis pass (thought + output for "
                          "one chunk), clamped to two thirds of context_length and to what "
                          "leaves a 4096-token input. Raising it takes room from the "
                          "transcript (more chunks, all read) — the right trade against a "
                          "CoT cut before its channel closes. Default 12288."),
            Field("synthesis.opener_max_new_tokens", "Synthesis: opener budget (tokens)", "int", 12288,
                  tooltip="Generation reserve of the opener-composition pass, clamped "
                          "against the reflect window minus a 4096-token input. A "
                          "truncated opener is refused outright, so a small budget costs a "
                          "reach-out rather than corrupting one. Default 12288."),
            Field("revisit.min_age_days", "Revisit: min chat age (days)", "int", 7,
                  tooltip="The Revisit button / head-phase picks a random chat at least "
                          "this old. Default 7."),
            Field("revisit.min_revisit_days", "Revisit: re-visit gap (days)", "int", 7,
                  tooltip="Anti-fixation gate: a chat revisited within this many days is "
                          "not picked again (folded from data/hot/revisit/revisited.jsonl). "
                          "Default 7."),
            Field("reachout.stale_delete_hours", "Stale opener deletion (h)", "float", 48.0,
                  tooltip="How long an UNANSWERED chat Ava opened herself stays before the "
                          "sweep deletes it (tombstoned first, so the reach-out backoff, "
                          "check-in's standing-opener list and outreach's dangling guard "
                          "still see it). 0 ⇒ keep forever. Default 48h."),
        ],
    ),
    Tab(
        "Deliberation",
        "The deliberation executive (core/deliberation.py): between conversations Ava reads "
        "her recent worklog and open threads and chooses what to do next — wander, "
        "synthesize, reach out, check in, revisit, aha, pivot, or rest — then runs the "
        "choice. Absent ⇒ the defaults below. Applies on the next server restart.",
        [
            Field("deliberation.mode", "Mode", "combo", "sole",
                  choices=["off", "shadow", "sole"],
                  tooltip="off: no autonomous deliberation (the Worklog tab's button still "
                          "works, dry-run). shadow: the executive runs as an idle job AND "
                          "the per-drive timers keep firing, so her choices can be compared "
                          "with theirs in the Activity tab. sole (default): the executive "
                          "is the only chooser — the drives' own timers are not registered; "
                          "the upkeep jobs stay on their clocks."),
            Field("deliberation.interval_s", "Interval (s)", "float", 3600.0,
                  tooltip="Minimum seconds between two deliberations. Default 3600."),
            Field("deliberation.recent", "Recent episodes shown", "int", 20,
                  tooltip="How many recent worklog entries she reads before deciding. "
                          "Default 20."),
            Field("deliberation.execute", "Dispatch the choice", "bool", True,
                  tooltip="Off ⇒ the executive decides and journals the decision but runs "
                          "nothing (a pure shadow). Default on."),
        ],
    ),
    Tab(
        "Reflection & retrieval",
        "Kill-switches and knobs on the background reflection pass and the retrieval "
        "channels (core/background_reflection.py, core/rag_engine.py, the portrait folds). "
        "Each `enabled` cuts ONE channel's retrieval or fold without touching the pass that "
        "produces its records. Absent ⇒ the defaults below.",
        [
            Field("background_reflection.sidecar_backfill", "Background: sidecar backfill", "bool", True,
                  tooltip="Rung 1 of the background wake: produce the missing gist / fact "
                          "sidecars for the whole corpus, oldest first, BEFORE any per-chat "
                          "reflection. It gates rung 2 by design. Default on."),
            Field("anchors.enabled", "Anchor channel", "bool", True,
                  tooltip="The per-exchange retrieval-anchor slot in the past-chat block "
                          "(rag_engine._query_anchors). Retrieval only; production is "
                          "overrides.exchange_anchors. Default on."),
            Field("recollections.enabled", "Recollection channel", "bool", True,
                  tooltip="Retrieval of [recollection] records (what Ava now makes of a "
                          "revisited chat). Production is overrides.recollection. Default on."),
            Field("impressions.enabled", "Impression channel", "bool", True,
                  tooltip="Retrieval of [impression] records (her readings of the people "
                          "she talks to). Production is overrides.user_notes. Default on."),
            Field("self_impressions.enabled", "Self-impression channel", "bool", False,
                  tooltip="Retrieval of [self_impression] records (her outside-view "
                          "reading of her own transcripts). OFF by default: the records "
                          "exist to be folded into users/_self.json, and recalling them "
                          "inside a live turn is the circularity the artifact avoids."),
            Field("self_portrait.enabled", "Outside-view self-portrait fold", "bool", True,
                  tooltip="Fold [self_impression] records into users/_self.json (read in "
                          "the Debug tab and the run archive; never injected). Production "
                          "is overrides.self_notes. Default on."),
            Field("user_portrait.enabled", "Per-person user portrait", "bool", True,
                  tooltip="Fold [impression] + attributed [fact] records into a standing "
                          "per-person portrait (reflection_runner) AND inject it on that "
                          "person's turns (generation) — read independently by both halves. "
                          "Default on."),
            Field("reflection_block.subject_cap", "Reflection block: subject cap", "int", 1,
                  tooltip="How many of the injected reflection block's three slots may go "
                          "to records about ONE subject. 0 disables the cap; 2 leaves one "
                          "slot guaranteed to another subject; 1 makes every slot a "
                          "distinct subject. Default 1."),
            Field("reflection_block.subject_sim", "Reflection block: subject cosine", "float", 0.45,
                  tooltip="Cosine (stored index vectors, same indexing basis) at which two "
                          "records count as the same subject. Looser than the dedup "
                          "threshold on purpose: a false link here only defers a record. "
                          "Default 0.45."),
        ],
    ),
    Tab(
        "Facts tree",
        "The facts-tree retrieval channel (FACTS_TREE.md §10; core/generation.py, "
        "core/til_wander.py, core/outreach.py, core/synthesis.py, core/rag_engine.py). "
        "When enabled a chat turn is two-stage: a short thinking-off pass picks which "
        "recorded facts the message turns on before the reply is written. Absent ⇒ the "
        "defaults below.",
        [
            Field("graph.enabled", "Facts fetch enabled", "bool", True,
                  tooltip="ON by default since 2026-08-12. Costs a whole extra generation "
                          "on time-to-first-token (prefill = the candidate list). Rides the "
                          "Chat tab's Facts checkbox. False ⇒ the single-stage turn."),
            Field("graph.max_claims", "Chat: max claims", "int", 6,
                  tooltip="Cap on the injected block for a chat turn (and the reach-out "
                          "lane). Default 6."),
            Field("graph.fetch_max_new_tokens", "Fetch pass budget (tokens)", "int", 512,
                  tooltip="Generation budget of the stage-1 pick pass (it emits ≤8 "
                          "numbers). Default 512."),
            Field("graph.nominate_max", "Nominated conversations", "int", 1,
                  tooltip="How many CONVERSATIONS a turn's picked facts may recall into the "
                          "past-chat block (rag_engine._query_nominated) — each is an "
                          "additive passage on top of top_k. 0 ⇒ off. Default 1."),
            Field("graph.gist_facts", "TIL recap: condition on facts", "bool", True,
                  tooltip="Extend the stage-1 fetch to the TIL recap pass, so a news digest "
                          "or wandered article is read against what is already on record. "
                          "Rides graph.enabled. Default on."),
            Field("graph.gist_max_claims", "TIL recap: max claims", "int", 3,
                  tooltip="Tighter than chat's cap: the live tree offers mostly claims "
                          "about the one person she talks to, and a recap of world events "
                          "must not drift into a recap of him. Default 3."),
            Field("graph.gist_max_reports", "TIL recap: max reports", "int", 2,
                  tooltip="The attributed-report channel of the reading lane — how many "
                          "`report` claims (what a TEXT asserted, rendered attributed to "
                          "its source) the recap may carry. 0 ⇒ knowledge-only. Default 2."),
            Field("graph.reachout_facts", "Reach-out: condition on facts", "bool", True,
                  tooltip="Extend the stage-1 fetch to outreach's decision pass and "
                          "synthesis's analysis pass (knowledge facets only). Rides "
                          "graph.enabled. Default on."),
            Field("graph.til_max_age_days", "TIL claim max age (days)", "opt_int", None,
                  write_null=True,
                  tooltip="Freshness scope on world facts: a TIL claim older than this is "
                          "not offered. UNSET (null) by default since 2026-08-12 — a TIL "
                          "claim is offered whatever its age, like a chat-backed one; the "
                          "candidate cap + chat-first ordering bound the list instead."),
        ],
    ),
    Tab(
        "Associative library",
        "The bridge to assoc/ (ASSOCIATIVE_MEMORY.md; core/assoc_bridge.py): the feed that "
        "builds the library from the corpus, the fetch that hands stage 1 of a chat turn to "
        "it, and the aha / witness / relation / pivot idle jobs. data/assoc/ is derived and "
        "disposable. Absent ⇒ the defaults below; read per call, so a flip needs no restart.",
        [
            Field("assoc.enabled", "Library fetch enabled", "bool", True,
                  tooltip="Hand stage 1 of a chat turn to the library instead of the facts "
                          "tree. A library with no build yet falls through to the tree "
                          "channel; false restores the tree channel exactly. Default on "
                          "(since 2026-09-10)."),
            Field("assoc.feed", "Feed job", "bool", True,
                  tooltip="Keep the library built by the hourly assoc_feed idle job "
                          "(sync over the corpus + rebuild). Default on."),
            Field("assoc.feed_subprocess", "Feed in a subprocess", "bool", True,
                  tooltip="Run the feed's sync + rebuild in a worker process so its "
                          "GIL-holding Python cannot starve the server's event loop. False "
                          "⇒ in-process. Default on."),
            Field("assoc.feed_timeout_s", "Feed timeout (s)", "int", 7200,
                  tooltip="How long the feed worker may run before it is killed and the "
                          "wake reported failed. Default 7200."),
            Field("assoc.tier", "Retrieval tier", "int_combo", 2, choices=[2, 3],
                  tooltip="2 = lexical + dense + codebook (the §12 verdict); 3 adds the "
                          "activation spread. Default 2."),
            Field("assoc.budget_tokens", "Injected block budget (tokens)", "int", 3000,
                  tooltip="Token budget of the injected block across the three grains. "
                          "Default 3000."),
            Field("assoc.max_picks", "Max picks", "int", 6,
                  tooltip="How many catalogue entries the select pass may pick. Default 6."),
            Field("assoc.device", "Embedder device", "combo", "cuda", choices=["cuda", "cpu"],
                  tooltip="Device for the BGE-M3 embedder; falls back to CPU when CUDA is "
                          "unavailable. Default cuda."),
            Field("assoc.embedder", "Embedder", "combo", "bge-m3", choices=["bge-m3", "hash"],
                  tooltip="bge-m3 (the real embedder) or hash (the offline stand-in used by "
                          "the benches). Default bge-m3."),
            Field("assoc.aha", "Aha job", "bool", True,
                  tooltip="The hourly assoc_aha idle job: open asks as needs, the judge on "
                          "what the feed brought in, an opener behind the reach-out gate. "
                          "Default on."),
            Field("assoc.aha_max_judgements", "Aha: judgements per wake", "int", 1,
                  tooltip="Thinking-on judgements per wake (minutes each). Default 1."),
            Field("assoc.witness", "Witness job", "bool", True,
                  tooltip="The hourly assoc_witness idle job: the library's own chat "
                          "witness over her transcripts, replacing the imported chat_facts "
                          "protocol. Default on."),
            Field("assoc.witness_per_wake", "Witness: chats per wake", "int", 3,
                  tooltip="Documents witnessed per wake (~3.5 min each). Default 3."),
            Field("assoc.relations", "Relation pass", "bool", True,
                  tooltip="Fold the typed-relation pass into the feed's rebuild when a "
                          "model is loaded (raises asked_about / looking_for / wants needs "
                          "from claims with no inline relation). Default on."),
            Field("assoc.relations_per_wake", "Relations: claims per wake", "int", 120,
                  tooltip="Claims run through the relation pass per wake. Default 120."),
            Field("assoc.pivot", "Pivot job", "bool", True,
                  tooltip="The hourly assoc_pivot idle job (§6 wander mode): one word-anchor "
                          "context switch per wake, a thinking-on pass deciding raise / "
                          "keep / nothing. Default on."),
            Field("assoc.pivot_bridges", "Pivot bridges", "strlist", ["sense"],
                  tooltip="Bridge kinds the pivot may use: sense (another sense of a word "
                          "in play), root (same root), sound (sounds like). Comma-separated. "
                          "Default sense."),
        ],
    ),
    Tab(
        "Gossip",
        "Model-gossip serving (GOSSIP.md). When enabled, the inference HTTP sidecar "
        "exposes an OpenAI-compatible /v1/chat/completions so a peer Ava can drive this "
        "box as if it were vLLM.",
        [
            Field("gossip.enabled", "Gossip serving enabled", "bool", True,
                  tooltip="ON by default since 2026-07-30 (a pulled box is reachable by a "
                          "peer with no config edit). False ⇒ the endpoint 404s. The route "
                          "carries no auth — turn it off on an untrusted network."),
            Field("gossip.log_transcripts", "Log served transcripts", "bool", True,
                  tooltip="Serving-side reflection: log this box's own half of the gossip "
                          "(with its CoT) so its Sleep pass can reflect on it too. Default "
                          "on; false restores stateless serving."),
            Field("gossip.peer_name", "Peer name", "str", "",
                  tooltip="Optional display name for the peer Ava. Empty ⇒ unset."),
        ],
    ),
    Tab(
        "Public API",
        "OpenAI-compatible API for external tools (core/api_http.py) — an agentic code "
        "assistant, an editor plugin, an SDK script. Its own port, separate from the "
        "management sidecar. Requests are NEVER logged: nothing written to chats/, so "
        "nothing reaches reflection or the training corpus.",
        [
            Field("api.enabled", "Public API enabled", "bool", True,
                  tooltip="ON by default since 2026-07-30 (a pulled box is queryable by an "
                          "external tool with no config edit). False ⇒ the listener never "
                          "starts. Set an API key, or bind to 127.0.0.1, on a shared "
                          "network — with neither, the port is open."),
            Field("api.port", "Port", "int", 8000,
                  tooltip="Listener port. Default 8000 (the --api-port flag is the "
                          "fallback when this is unset)."),
            Field("api.host", "Bind address", "str", "",
                  tooltip="Empty ⇒ the server's --api-host, else --host. Use 127.0.0.1 to "
                          "keep it local to the box."),
            Field("api.api_key", "API key (bearer)", "str", "",
                  tooltip="Shared secret required as `Authorization: Bearer <key>`. EMPTY "
                          "⇒ no auth: anyone who can reach the port can spend GPU time as "
                          "Ava. Set one whenever the bind address is not 127.0.0.1."),
            Field("api.model_name", "Model name", "str", "ava",
                  tooltip="The id reported by GET /v1/models and echoed in responses — "
                          "what the client's model picker shows."),
            Field("api.max_tokens", "Default max tokens", "str", "75%",
                  tooltip="Used when a request omits max_tokens. Accepts an integer or a "
                          "percentage of the remaining context (chat's own convention)."),
            Field("api.client_system", "Client system prompt", "combo", "append",
                  choices=["append", "drop"],
                  tooltip="append: wrap the calling tool's system message in "
                          "prompts/api_client_system_prompt.txt and place it last (an "
                          "agent's whole operating brief lives there). drop: ignore it, "
                          "gossip-style — she only ever answers as herself."),
            Field("api.inject_rag", "Inject RAG memory", "bool", True,
                  tooltip="Retrieve her memory (past chats, facts, persona) for each "
                          "request, as live chat does. Off ⇒ prompt + adapter only, which "
                          "is usually what a code assistant wants."),
            Field("api.inject_persona", "Inject persona portrait", "bool", True,
                  tooltip="Inject the standing persona digest, as live chat does. Off ⇒ "
                          "no portrait and no [persona] RAG channel."),
        ],
    ),
    Tab(
        "Hidden states",
        "Track B Stage 0 of AVA_REWARD_LOOP.md: residual-stream capture during LIVE chat "
        "turns (core/hidden_capture.py). Per exchange, a few positions' full residuals for "
        "a band of decoder layers go beside the transcript as `<ts>.hidden.npz` (fp16, "
        "append-only, never read by inference), and per-token projections onto any axis "
        "file under `data/axes/` ride in the transcript's tension block (`axes`) and paint "
        "the chat's blue channel. Read-only telemetry: nothing steers. Absent ⇒ the "
        "defaults below.",
        [
            Field("hidden.capture", "Capture hidden states", "bool", True,
                  tooltip="Hook the decoder layers below during live chat and write the "
                          "`.hidden.npz` sidecar per exchange. Off ⇒ no hooks, no sidecar, "
                          "no axis projections (blue channel dark). Reflection, API and "
                          "gossip generations are never captured either way. Default on: "
                          "the cost is one tensor slice per layer per token."),
            Field("hidden.layers", "Layers stored", "str", "2,3,4,5,50%,65%,80%",
                  tooltip="Decoder layers whose residuals are STORED, comma-separated: an "
                          "integer is a 0-based layer index, `NN%` a fraction of the model's "
                          "depth. Default: the dopamine paper's load-bearing early band "
                          "(2–5) plus a mid-to-late band where the Pain Axis extraction "
                          "lands. An axis file's own layer is hooked for its projection "
                          "whether or not it is listed here. Disk per exchange ≈ positions × "
                          "layers × hidden dim × 2 bytes."),
            Field("hidden.max_positions", "Max positions per reply", "int", 64,
                  tooltip="Positions kept per exchange: step 0 (after the prompt), the step "
                          "after every paragraph break, and the last token; over this many "
                          "the interior is thinned evenly (the ends always stay). Default 64."),
        ],
    ),
    Tab(
        "Logging",
        "The unified activity journal (core/activity_log.py, LOGGING.md): what the Activity "
        "tab shows. Every level has an off switch, because the failure mode of a logging "
        "change is that it becomes the reason a run dies. Absent ⇒ the defaults below.",
        [
            Field("logging.heartbeat_s", "Heartbeat (s)", "float", 20.0,
                  tooltip="Seconds between `stream` records (elapsed, tokens so far, a "
                          "rolling tail) while a pass holds the GPU. 0 ⇒ no heartbeat "
                          "(restores the pre-2026-08-11 silence). Default 20."),
            Field("logging.body_max_chars", "Body cap (chars)", "int", 16000,
                  tooltip="A finished generation longer than this is elided in the MIDDLE "
                          "(never clipped at the end — the answer after </think> is what "
                          "an end-clip drops). Default 16000."),
            Field("logging.stream_tail_chars", "Heartbeat tail (chars)", "int", 240,
                  tooltip="Rolling tail carried by a heartbeat. Default 240."),
            Field("logging.segment_bytes", "Rotate past (bytes)", "int", 16000000,
                  tooltip="Rotate activity.jsonl into a numbered segment past this size. "
                          "Default 16 MB."),
            Field("logging.retain_segments", "Rotated segments kept", "int", 8,
                  tooltip="How many rotated segments are kept (oldest pruned). Default 8."),
            Field("logging.stream_enabled", "Heartbeat records", "bool", True,
                  tooltip="Off ⇒ no `stream` records at all. Default on."),
            Field("logging.raw_enabled", "Stdout tee", "bool", True,
                  tooltip="Off ⇒ the process's stdout/stderr lines are not journalled "
                          "(`raw` level). Default on."),
            Field("logging.raw_max_lines_per_s", "Tee rate cap (lines/s)", "int", 200,
                  tooltip="Past this, tee'd lines coalesce into one '+N lines suppressed' "
                          "note. Default 200."),
            Field("logging.raw_max_line_chars", "Tee line cap (chars)", "int", 2000,
                  tooltip="A tee'd line longer than this is clipped. Default 2000."),
            Field("logging.tee_denylist", "Tee denylist (regexes)", "strlist", [],
                  tooltip="Extra regexes the stdout tee drops. Comma-separated. Default "
                          "empty."),
        ],
    ),
    Tab(
        "Prompt rewrite",
        "The autonomous standing-prompt rewrite (PROMPT_REWRITE.md). Stage 1 (2026-09-18) is "
        "the LOCATOR only: the prompt-mutation pass, which used to run on `revise` exchanges "
        "alone, also runs on a KEPT exchange whose CoT tension ranks high against the corpus "
        "baseline (core/tension_baseline.py). Stage 2 folds the logged deltas into patterns "
        "(core/prompt_patterns.py) — the budget the executive is shown. Stage 3 "
        "(core/prompt_rewrite.py) is the event that spends it — ON by default since 2026-09-19 "
        "(`enabled`; the owner's call: adjust, never disable). "
        "Absent ⇒ the defaults below.",
        [
            Field("prompt_rewrite.keep_locator", "Locate kept-but-torn exchanges", "bool", True,
                  tooltip="Run the prompt-mutation pass on a `keep` exchange whose CoT "
                          "tension clears the percentile below. Costs one extra reflection "
                          "pass per such exchange. Off ⇒ the pre-2026-09-18 revise-only "
                          "locator. Default on."),
            Field("prompt_rewrite.tension_percentile", "Tension percentile", "float", 0.8,
                  tooltip="A kept exchange qualifies when its CoT median entropy OR "
                          "contested fraction ranks at or above this percentile within its "
                          "(model, adapter, reply language) bucket; a thin bucket (<30 "
                          "samples) falls back adapter-wide, and no baseline closes the cell. "
                          "Default 0.8 (the top fifth)."),
            Field("prompt_rewrite.maturity", "Pattern maturity (weighted recurrence)", "float", 1.9,
                  tooltip="Stage 2: a pattern of same-change deltas is mature — may fund a "
                          "rewrite — when its tension-weighted recurrence over distinct "
                          "chats reaches this (one vote per chat, weight recency × (0.5 + "
                          "tension rank)). The digest's judge-gate number. Default 1.9."),
            Field("prompt_rewrite.min_chats", "Pattern maturity (min chats)", "int", 2,
                  tooltip="A pattern also needs this many DISTINCT chats behind it, "
                          "whatever its weight — one conversation is never a standing "
                          "pull. Default 2."),
            Field("prompt_rewrite.enabled", "Autonomous rewrite enabled", "bool", True,
                  tooltip="Stage 3: offer `rewrite_prompt` to the deliberation executive "
                          "when the budget clears. On, Ava can replace her live standing "
                          "prompt on her own decision (through the experiment tier — "
                          "Revert in the Prompt tab is the veto). Default ON since 2026-09-19 "
                          "— the owner chose to run it live and tune rather than gate it."),
            Field("prompt_rewrite.samples", "Rewrite samples", "int", 5,
                  tooltip="How many draft rewrites an event generates before the consensus "
                          "and final passes. Default 5."),
            Field("prompt_rewrite.temperatures", "Sample temperatures", "floatlist", [],
                  tooltip="One temperature per sample (padded with the last / clipped to "
                          "`samples`). Empty ⇒ 0.9 for every draw (Sleep's default). The "
                          "owner's manual sweep found the rewrite non-monotonic in "
                          "temperature (near-contradictory at 1.2, back at 1.4), so a list "
                          "such as 0.9,1.0,1.1,1.2,1.4 restores that spread."),
            Field("prompt_rewrite.final_temperature", "Final draft temperature", "float", 0.9,
                  tooltip="Temperature of the one draft written around the consensus "
                          "notes. Default 0.9."),
            Field("prompt_rewrite.min_gap_hours", "Min hours between attempts", "float", 24.0,
                  tooltip="An attempt — changed, stayed or declined — rests the option "
                          "for this long. Default 24."),
            Field("prompt_rewrite.order_check", "Choice order check", "bool", False,
                  tooltip="Run the blind choice twice with the letters reversed; a "
                          "disagreement keeps the current prompt. One extra generation "
                          "per event. Default off."),
        ],
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Nested dict helpers
# ──────────────────────────────────────────────────────────────────────────────

def dget(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dhas(d: dict, path: str) -> bool:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def dset(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# Schema walk + the generic back-fill
# ──────────────────────────────────────────────────────────────────────────────

def iter_fields() -> Iterator[Field]:
    for tab in SCHEMA:
        yield from tab.fields


def known_paths() -> list[str]:
    return [f.path for f in iter_fields()]


def backfill_value(spec: Field, config: dict) -> tuple[bool, Any]:
    """``(write, value)`` for one field against *config* — the policy in one place.

    Not written: a ``backfill=False`` field (box state), a field already present at any
    value (an operator's null stays null), and a ``None`` default whose reader does not
    treat null like absent. ``derive`` wins over the static default when it yields a value.
    """
    if not spec.backfill or dhas(config, spec.path):
        return False, None
    value = spec.default
    if spec.derive is not None:
        try:
            derived = spec.derive(config)
        except Exception:
            derived = None
        if derived is not None:
            value = derived
    if value is None and not spec.write_null:
        return False, None
    return True, copy.deepcopy(value)


def backfill_defaults(config: dict) -> list[str]:
    """Write every absent knob's default into *config* in place; return the paths added.

    Idempotent (a second call adds nothing) and behaviour-preserving by the contracts in
    the module docstring. The caller decides whether to persist (the server rewrites the
    file only when the list is non-empty)."""
    if not isinstance(config, dict):
        return []
    added: list[str] = []
    for spec in iter_fields():
        write, value = backfill_value(spec, config)
        if write:
            dset(config, spec.path, value)
            added.append(spec.path)
    return added


def config_text(config: dict) -> str:
    """The one serialization every writer of the file uses (2-space JSON + newline)."""
    return json.dumps(config, indent=2, ensure_ascii=False) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    paths = known_paths()
    assert len(paths) == len(set(paths)), "duplicate field path"
    for spec in iter_fields():
        assert spec.kind in ("str", "path", "combo", "int_combo", "bool", "int", "float",
                             "opt_int", "opt_float", "floatlist", "strlist"), spec
        if spec.write_null:
            assert spec.default is None, f"{spec.path}: write_null on a non-None default"
        if spec.kind in ("int", "float") and spec.backfill:
            assert spec.default is not None, f"{spec.path}: required kind with no default"

    # Empty config: everything backfillable lands, nothing else does.
    cfg: dict = {}
    added = backfill_defaults(cfg)
    assert "model_id" not in cfg and "adapter_id" not in cfg, "box state was written"
    # No context_length to derive from ⇒ the static default, which equals the server's
    # own fallback for a config lacking context_length (32768), so no split either way.
    assert cfg["reflect_context_length"] == 32768 == cfg["context_length"]
    assert cfg["chat_degen"] == {"window": None, "min_gen": None, "distinct_ratio": None,
                                 "top_freq": None}
    assert cfg["graph"]["til_max_age_days"] is None
    assert cfg["chat_repetition_penalty"] == 1.1
    assert cfg["deliberation"]["mode"] == "sole"
    assert cfg["assoc"]["pivot_bridges"] == ["sense"] and cfg["logging"]["tee_denylist"] == []
    expected = {f.path for f in iter_fields()
                if f.backfill and (f.default is not None or f.write_null)}
    assert set(added) == expected, set(added) ^ expected
    assert backfill_defaults(cfg) == [], "not idempotent"
    # Lists are copied, never shared with the schema.
    cfg["assoc"]["pivot_bridges"].append("sound")
    assert next(f for f in iter_fields() if f.path == "assoc.pivot_bridges").default == ["sense"]

    # Present values — null included — are left alone; derive follows context_length.
    cfg2 = {"context_length": 24576, "chat_repetition_penalty": None,
            "checkin": {"recent_chats": 9}, "deliberation": {"mode": "shadow"}}
    added2 = backfill_defaults(cfg2)
    assert cfg2["reflect_context_length"] == 24576 and "reflect_context_length" in added2
    assert cfg2["chat_repetition_penalty"] is None and "chat_repetition_penalty" not in added2
    assert cfg2["checkin"]["recent_chats"] == 9 and cfg2["checkin"]["max_users"] == 3
    assert cfg2["deliberation"]["mode"] == "shadow" and cfg2["deliberation"]["recent"] == 20
    assert json.loads(config_text(cfg2)) == cfg2

    # Cross-check the defaults against the readers whose modules are import-light.
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in (here, here / "inference"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.decay import (WallClockConfig, TRAIN_LR_DEFAULT,
                                TRAIN_PLATEAU_EPOCHS_DEFAULT, TRAIN_LORA_R_DEFAULT)
    w = WallClockConfig()
    wc = cfg["consolidation"]["wall_clock"]
    assert (wc["rag_only_window_h"], wc["lora_cap_age_h"], wc["rag_cap_age_h"],
            wc["gist_cap_age_h"], wc["base_lr"], wc["rag_floor_weight"],
            wc["gist_floor_weight"]) == (
        w.rag_only_window_h, w.lora_cap_age_h, w.rag_cap_age_h, w.gist_cap_age_h,
        w.base_lr, w.rag_floor_weight, w.gist_floor_weight)
    assert tuple(wc["lr_ramp"]) == w.lr_ramp
    assert tuple(wc["wander_rag"]["weights"]) == w.wander_rag_weights
    assert wc["wander_rag"]["step_h"] == w.wander_rag_step_h
    c = wc["contamination"]
    assert (c["enabled"], c["dose"], c["additive"], c["min_user_chars"], c["fold"]) == (
        w.contamination_enabled, w.contamination_dose, w.contamination_additive,
        w.contamination_min_user_chars, w.contamination_fold)
    assert (wc["fresh_window"]["droop_frac"], wc["fresh_window"]["horizon_h"]) == (
        w.fresh_droop_frac, w.fresh_horizon_h)
    r = wc["recollection"]
    assert (r["hold_h"], r["cap_age_h"], r["floor_weight"]) == (
        w.recollection_hold_h, w.recollection_cap_age_h, w.recollection_floor_weight)
    assert (cfg["train_lr"], cfg["train_plateau_epochs"], cfg["lora_r"]) == (
        TRAIN_LR_DEFAULT, TRAIN_PLATEAU_EPOCHS_DEFAULT, TRAIN_LORA_R_DEFAULT)
    checked = ["training.decay"]
    try:
        from core import deliberation as _d
        assert cfg["deliberation"] == _d.settings({}), (cfg["deliberation"], _d.settings({}))
        checked.append("core.deliberation")
    except ImportError:
        pass
    try:
        from core import activity_log as _a
        lg = dict(cfg["logging"])
        lg["tee_denylist"] = tuple(lg["tee_denylist"])
        assert lg == dict(_a._DEFAULTS), (lg, _a._DEFAULTS)
        checked.append("core.activity_log")
    except ImportError:
        pass
    try:
        from core import chat_worklog as _cw
        assert cfg["reachout"]["stale_delete_hours"] == _cw.DEFAULT_STALE_HOURS
        checked.append("core.chat_worklog")
    except ImportError:
        pass
    print(f"config_schema selftest: OK ({len(paths)} fields, {len(added)} back-filled on an "
          f"empty config; cross-checked against {', '.join(checked)})")


if __name__ == "__main__":
    _selftest()
