"""Generation — the model-facing text-generation layer, extracted from server.py.

Everything between "a request arrives" and "cleaned text comes back" lives here:

  * **Live chat** — `handle_generate` / `handle_generate_ephemeral` drive a
    streaming generation on the GPU executor thread (`_run_generation`), with the
    live-chat loop guards (stop-on-repeat + mild repetition penalty), RAG
    injection, proactive open-question surfacing, tension capture, and logging.
  * **Generate factories** — `_make_sync_reflect_generate` / `_make_agentic_generate`
    build the reflect-/agentic-generate callables the reflection and TIL/wander
    subsystems run their passes through (injected back into them by server.py).
  * **Branch replay** — `handle_branch_exchange` + the sync branch helpers.
  * **Response cleaning** — the `_clean_response` / `_clean_reflect_response` family
    (dup-tail collapse, follow-up-turn-leak trim, role-noise strip, early-stop).
  * **Tension capture** — per-token entropy/margin -> per-segment stats.
  * **Prompt context** — identity line, temporal anchor, surfaced open questions,
    and `_build_inference_conversation`.

Never imports `server`: the Ava-side capabilities it needs (WebSocket `send`, GPU
`executor`/`backend`/`cancel_event`, the RAG/logger/writer accessors, token +
activity bookkeeping, the surface-template loader, chat repetition penalty and
paths) are injected once at startup via :func:`configure`. Model/session state is
read from `core.runtime_state`. It reads the busy flags of the other executor-
monopolising subsystems (`reflection_service._reflection_run_active`,
`encounter_run._encounter_active`) to refuse chat while they run — those modules
are injected their generate callables and never import back, so this stays
acyclic. The moved code is otherwise verbatim.
"""
from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from core.runtime_state import (runtime as _runtime, session as _session,
                                reflect_window as _shared_reflect_window)
from core.chat_logger import ChatLogger
from core.chat_sidecar import is_chat_session_json
# One definition of "keep the peer's chain of thought out of Ava", shared by the two
# sides of a gossip mirror: the driver strips what a counterpart replied, this module
# strips what a peer sent us. `core.encounter` imports only stdlib, so no cycle.
from core.encounter import strip_peer_reasoning
from core.reflection_memory import (
    ReflectionMemory, is_self_directed_origin, load_origin_note)
from core.reflection_config import REFLECT_REPETITION_PENALTY, REFLECT_NO_REPEAT_NGRAM
from core.llm_shared import build_inference_prompt, ensure_chat_template
from core import activity_log
from core import alloc_guard
from core import fact_fetch as _fact_fetch_mod
from core import model_family
from core import branch_replay
from core import reflection_service
from core import encounter_run
from core import background_reflection

# Proactive surfacing of open reflection questions into live chat (generation-only).
_SURFACE_LIMIT = 2     # max questions Ava may raise at the start of a session
_SURFACE_CEILING = 3   # a user question retires from surfacing past this many raises

# ── Open [ask] questions in live chat: OFF pending redesign (2026-07-28) ──────────────
# Open asks reached an ordinary chat turn two ways, and this switch kills both:
#   1. `(still wondering …)` recall lines in the reflection-memory RAG block, selected by
#      similarity every turn — which they usually WIN, because an ask embeds on its full
#      conversational question text while a fact embeds on a short topic-label `trigger`,
#      so asks crowded facts out of the block's three slots;
#   2. the `surface_prompt.txt` block on the first turn of a session — the only path with
#      an actual mandate to raise something, but context-blind (the selection never looks
#      at the message), fired exactly once per session, and at the moment a live user
#      request is competing for attention.
# Turning both off also stops a real harm: `write_surface` stamps the retirement counter
# when the block is INJECTED and the turn lands, not when Ava actually asks anything, so a
# `user` ask was permanently retired from surfacing after three unasked injections — a
# budget shared with outreach, which stamps the same counter.
# Outreach / synthesis / check-in are untouched: Ava still raises questions by opening a
# conversation, which is the path that demonstrably works. Retrieval machinery is kept
# intact (`RagEngine.query(include_asks=…)`, `_select_surfaced_questions`, `_surface_block`),
# so flipping this back to True restores the previous behaviour exactly.
_INJECT_OPEN_ASKS = False

# ── The wander/TIL channel in live chat: OFF pending redesign (2026-07-28) ────────────
# `RagEngine.query(include_wander=…)` already defaults False; live chat, the gossip
# serving endpoint and the encounter loop were the three callers passing True, and none
# does now. Removed as noise rather than as a bad idea: the channel gates on a 0.15 raw
# cosine — low enough that its single slot is filled on essentially every turn — and then
# ranks by an age step (0.4/0.3/0.2/0.1 per 24 h), so the newest article wins unless an
# older one scores 4x its similarity. The effect is "inject today's article, near
# unconditionally, for 24 h", with relevance delegated entirely to the model's judgement
# once it is already in the prompt. The index, decay curve and prompt template are all
# untouched — this is only about whether chat-shaped paths ask for the block.
_INJECT_WANDER = False

# ── Injected server capabilities (populated by configure()) ──
_send: Callable = None                       # async _send(ws, msg)
_backend: Any = None                         # UnslothBackend
_cancel_event: Any = None                    # threading.Event
_executor: Any = None                        # ThreadPoolExecutor (single GPU worker)
_get_rag: Callable = None
_ensure_logger: Callable = None
_get_reflection_writer: Callable = None
_add_user_tokens: Callable = None
_read_token_economy: Callable = None
_mark_activity: Callable = None
_load_surface_template: Callable = None
_CHATS_DIR: Any = None
_MEMORY_DIR: Any = None
# Live-chat repetition penalty; resolved from server_config in server.main() and
# passed in (1.0/None disables, leaving the halt-only stop_on_repeat guard).
_CHAT_REPETITION_PENALTY: Optional[float] = 1.1
# Degeneration floor, resolved from server_config in server.main() (keys are named
# chat_* for historical reasons — they now govern every generate lane in this module).
# _CHAT_MIN_P is Layer 1 (a relative-probability sampling floor that removes the tail
# seeding a collapse); _DEGEN_KW is Layer 2 (a drifting-runaway halt the verbatim
# stop_on_repeat can't see) — a kwargs bundle ({"degen_stop": bool, plus any threshold
# overrides}) spread into stream_generate.
#
# The reflect/agentic factories share BOTH (2026-07-31). They previously passed
# neither, on the reasoning that the floor was a live-chat concern — but a reflection
# pass is the lane with the *least* defense (REFLECT_REPETITION_PENALTY and
# REFLECT_NO_REPEAT_NGRAM are both None; see reflection_config), the longest token
# budget, and no operator watching a stream they can Stop. An observed revision pass
# collapsed into a ~4-token letter-soup walk and ran toward context exhaustion:
# stop_on_repeat is structurally blind to it (every 12-gram window is novel, so no
# span recurs), which is the exact gap Layer 2 exists to close. Layer 2 is halt-only
# and alters no sampled token; Layer 1 does shape reflection sampling, and therefore
# the IDEALs that become training targets — deliberate, since an implausible-tail
# excursion in an IDEAL is a poisoned target either way.
#
# NOT shared with the reflect lane: the mild chat repetition_penalty (it garbles
# analytical prose that legitimately reuses phrases) and the anti-copy guard (there is
# no "previous reply" to regurgitate). Branch replay generates through a different
# backend primitive and is untouched.
_CHAT_MIN_P: Optional[float] = None
_DEGEN_KW: dict = {"degen_stop": False}


def configure(*, send, backend, cancel_event, executor, get_rag, ensure_logger,
              get_reflection_writer, add_user_tokens, read_token_economy, mark_activity,
              load_surface_template, chats_dir, memory_dir, chat_repetition_penalty,
              chat_min_p=None, degen_kw=None) -> None:
    """Wire in the server capabilities the moved generation code depends on.

    Called once from server startup, before the WebSocket server accepts clients.
    """
    global _send, _backend, _cancel_event, _executor, _get_rag, _ensure_logger
    global _get_reflection_writer, _add_user_tokens, _read_token_economy, _mark_activity
    global _load_surface_template, _CHATS_DIR, _MEMORY_DIR, _CHAT_REPETITION_PENALTY
    global _CHAT_MIN_P, _DEGEN_KW
    _send = send
    _backend = backend
    _cancel_event = cancel_event
    _executor = executor
    _get_rag = get_rag
    _ensure_logger = ensure_logger
    _get_reflection_writer = get_reflection_writer
    _add_user_tokens = add_user_tokens
    _read_token_economy = read_token_economy
    _mark_activity = mark_activity
    _load_surface_template = load_surface_template
    _CHATS_DIR = chats_dir
    _MEMORY_DIR = memory_dir
    _CHAT_REPETITION_PENALTY = chat_repetition_penalty
    _CHAT_MIN_P = chat_min_p
    if degen_kw is not None:
        _DEGEN_KW = degen_kw


# ══════════════════════════════════════════════════════════════════════════════
# Moved verbatim from server.py.
# ══════════════════════════════════════════════════════════════════════════════


def _collapse_exact_duplicate_tail(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < 40:
        return stripped
    mid = len(stripped) // 2
    candidates = [mid] if len(stripped) % 2 == 0 else [mid, mid + 1]
    for split in candidates:
        left = stripped[:split].strip()
        right = stripped[split:].strip()
        if left and left == right:
            return left
    return stripped

def _trim_followup_turn_leak(text: str) -> str:
    trimmed = text
    m = re.search(r"(?is)(?:[.!?]\s*|\n\s*)(user|system)[^\n]{0,32}\n\s*\n", trimmed)
    if m:
        trimmed = trimmed[:m.start()].rstrip()
    m = re.search(
        r"(?is)\n\s*\n(?:what|why|how|who|when|where|is|are|do|does|did|can|could|would|should|will)"
        r"\b[^\n?]{0,220}\?\s*\n+\s*assistant\b",
        trimmed,
    )
    if m:
        trimmed = trimmed[:m.start()].rstrip()
    m = re.search(r"(?is)\n\s*(user|assistant|system)\b[^\n]{0,24}\n\s*\n", trimmed)
    if m:
        trimmed = trimmed[:m.start()].rstrip()
    m = re.search(
        r"(?is)\b(?:user|assistant|system){2,}[^\n]{0,40}\n\s*\n"
        r"(?:what|why|how|who|when|where|is|are|do|does|did|can|could|would|should|will)\b",
        trimmed,
    )
    if m:
        trimmed = trimmed[:m.start()].rstrip()
    m = re.search(
        r"(?is)\s(?:user|assistant|system)[^\n]{0,64}\n\s*\n[^\n?]{0,8}"
        r"(?:what|why|how|who|when|where|is|are|do|does|did|can|could|would|should|will)"
        r"\b[^\n?]{0,220}\?\s*\n+\s*assistant\b",
        trimmed,
    )
    if m:
        trimmed = trimmed[:m.start()].rstrip()
    return trimmed

def _strip_leading_role_noise(text: str) -> str:
    stripped = text.lstrip()
    m = re.match(r"(?is)^(?:assistant|user|system)[a-z_]{0,32}\s*\n\s*\n", stripped)
    if m:
        stripped = stripped[m.end():].lstrip()
    return stripped

def clean_dialogue_response(raw: str, model_id: str) -> str:
    """Normalize and sanitize a normal dialogue completion for persistence/training."""
    # Normalize the loaded model family's reasoning trace to the canonical
    # <think>…</think>\nanswer form so downstream CoT parsing is identical for all
    # models (gemma-4 channel tokens, gpt-oss analysis/final harmony, Qwen's
    # prompt-prefilled opener — see core/model_family.py).
    raw = model_family.family_for(model_id or "").normalize_cot(raw)

    role_pat = re.compile(
        r"(?:^|\n)\s*(?:<\|[^>]+\|>\s*)?(assistant|user|system)[^\n<]{0,24}"
        r"(?:\s*<\|[^>]+\|>)?\s*(?:\n|$)",
        re.IGNORECASE,
    )
    matches = list(role_pat.finditer(raw))
    if len(matches) > 1:
        raw = raw[:matches[1].start()]

    # Remove ChatML/Gemma 1-3/Gemma 4 structural tokens.
    resp = re.sub(
        r'<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?|<\|channel>|<channel\|>',
        '', raw,
    ).strip()
    resp = resp.split('</assistant', 1)[0].strip()
    resp = _strip_leading_role_noise(resp)
    resp = re.sub(r'\bassistant\s*:?\s*$', '', resp, flags=re.IGNORECASE).strip()
    resp = re.sub(r'\n\s*(?:assistant|user|system)\s*:?\s*$', '', resp, flags=re.IGNORECASE).strip()
    resp = re.sub(r'^\s*(assistant|user|system)\s*:?\s*\n+', '', resp, flags=re.IGNORECASE)

    for marker in (
        '\nuser\n', '\nUSER\n', '\nUser\n',
        '\nsystem\n', '\nSYSTEM\n', '\nSystem\n',
        '\nassistant\n', '\nASSISTANT\n', '\nAssistant\n',
    ):
        if marker in resp:
            resp = resp.split(marker, 1)[0].rstrip()

    trimmed = _trim_followup_turn_leak(resp)
    if trimmed != resp:
        resp = trimmed

    if '</think>' in resp:
        think_part, answer_part = resp.split('</think>', 1)
        answer_part = _collapse_exact_duplicate_tail(answer_part)
        resp = f"{think_part}</think>{answer_part}"
    else:
        resp = _collapse_exact_duplicate_tail(resp)

    first_think = resp.find('<think>')
    if first_think != -1:
        second_think = resp.find('<think>', first_think + len('<think>'))
        if second_think != -1:
            resp = resp[:second_think].rstrip()

    # Force exactly one blank line between </think> and the answer so chat-tab
    # display is consistent across models (Gemma's channel-tag rewrite leaves no
    # gap; gpt-oss inserts one \n; some models inline both with no separator).
    resp = re.sub(r'</think>\s*', '</think>\n\n', resp, count=1)

    return resp.strip()


def _clean_response(raw: str) -> str:
    """Live-server wrapper over the model-id-explicit dialogue cleaner."""
    return clean_dialogue_response(raw, _runtime.model_id or "")

def _should_early_stop_stream(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"(?is)\n\s*(user|system)\b[^\n]{0,32}\n\s*\n",
        r"(?is)\n\s*\n(?:what|why|how|who|when|where|is|are|do|does|did|can|could|would|should|will)"
        r"\b[^\n?]{0,220}\?\s*\n+\s*assistant\b",
    ]
    return any(re.search(p, text) for p in patterns)

def _clean_reflect_response(raw: str) -> str:
    """Minimal cleaning for reflection output: strips model format tokens only.

    Does NOT apply turn-leak detection or role-header truncation — reflection
    output intentionally contains session transcripts with User:/Ava: lines.
    """
    # Normalize the loaded family's reasoning trace to <think>…</think>
    # (gemma-4 channel tokens, Qwen's prompt-prefilled opener, …).
    raw = model_family.family_for(_runtime.model_id or "").normalize_cot(raw)
    # Strip ChatML / model structural tokens only
    resp = re.sub(
        r'<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?',
        '', raw,
    ).strip()
    return resp


@dataclass(frozen=True)
class PreparedReflectPrompt:
    """Exact model-facing reflection prompt, prepared once and reused for generation."""

    prompt: str
    input_tokens: int
    rag_tokens: int
    rag_context: str = ""
    # Whether the family's reasoning-channel opener was actually prefilled onto `prompt`.
    # The `min_think_tokens` floor is the companion to that prefill and must key on it,
    # not on the caller's `force_think` — a prompt prepared separately (the exact-fit
    # consolidation chunker) is generated through a call that never saw the flags it was
    # built with, and a floor applied without a prefill bans a close marker for a channel
    # that was never opened.
    think_prefilled: bool = False
    # The Debug view of this prompt: the same labelled `(kind, label, text)` segments the
    # chat lane emits (`_prompt_debug_segments`), built from the SAME parts list the system
    # message is joined from — so a block added to a reflection pass shows up here for free
    # and the view can never drift from the prompt. Carried on the prepared object rather
    # than recomputed by the caller because the exact-fit chunker prepares a prompt in one
    # place and generates it in another.
    debug_segments: tuple = ()
    # Whether the CALLER turned retrieval off for this pass (`disable_rag=True`). An empty
    # RAG block and a pass that never asked for one render identically in the segment list,
    # and telling those two apart is the first question when debugging retrieval — so the
    # distinction rides the prepared prompt instead of being inferred from a missing
    # segment. A `rag_context_override` counts as retrieval being ON (a block was injected,
    # just not retrieved here).
    rag_disabled: bool = False


class PromptBudgetError(RuntimeError):
    """A deterministic reflection input-budget overflow (never worth retrying unchanged)."""

    def __init__(self, input_tokens: int, input_limit: int, context_length: int) -> None:
        self.input_tokens = int(input_tokens)
        self.input_limit = int(input_limit)
        self.context_length = int(context_length)
        super().__init__(
            f"Prompt ({self.input_tokens} tokens) exceeds reflection input budget "
            f"({self.input_limit}; context {self.context_length}) — rechunk required."
        )


def _count_text_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return int(_backend.count_tokens(tokenizer, text))


def _clip_rag_to_tokens(tokenizer, text: str, max_tokens: Optional[int]) -> str:
    """Keep a whole-line prefix of RAG context within *max_tokens*.

    Consolidation uses reflection-memory RAG only. Each memory item is one line, so
    retaining whole lines avoids cutting a fact/question in half while making its
    contribution to the final prompt deterministic and bounded.
    """
    if not text or max_tokens is None or max_tokens <= 0:
        return "" if max_tokens is not None and max_tokens <= 0 else text
    if _count_text_tokens(tokenizer, text) <= max_tokens:
        return text

    kept: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join(kept + [line])
        if _count_text_tokens(tokenizer, candidate) > max_tokens:
            break
        kept.append(line)
    clipped = "\n".join(kept).rstrip()
    marker = "\n...[additional reflection notes omitted for token budget]"
    if clipped and _count_text_tokens(tokenizer, clipped + marker) <= max_tokens:
        clipped += marker
    return clipped

def _reflect_system_parts(system_prompt: str, rag_context: str,
                          facts_block: str = "") -> list:
    """A reflection pass's system message as labelled `(kind, label, text)` parts.

    *facts_block* is a fetched facts-tree block (stage 1's product, already wrapped by
    the caller's lane template) for the reflect-lane passes that run the fetch —
    outreach's decision, synthesis's analysis. It takes the SAME label as chat's
    (`"FACTS (fetched)"`), so the `outreach_prompt`/`checkin_prompt` debug views and the
    Chat tab's Debug view read as the same thing; and it sits in the same position
    relative to retrieval — before INJECTED RAG, because it was *selected for* this
    material by a pass that read it, where the RAG block is whatever embedded nearest.
    Both stay ahead of the pass prompt, which keeps the contract last (the ordering rule
    below). It is a seam parameter rather than a `{slot}` in each subsystem's prompt
    file, because an operator's customized prompt on disk would silently drop a slot it
    predates — the failure the `{origin_note}`/`{recent}` folds were shaped to avoid,
    avoided here by not touching those files at all.

    The chat lane's counterpart is the `system_parts` list in `_build_inference_conversation`,
    and it exists here for the same reason: the message is JOINED from these parts and the
    Debug view is BUILT from them, so the two cannot drift as blocks are added.

    The order is the opposite of live chat's, deliberately. Chat puts RAG last because
    the situational material is what the next turn answers; a reflection pass's system
    message ends in a contract instead ("Output exactly these fields, nothing else" /
    "Do not write an IDEAL reply here"), and appending a block of remembered prose after
    it buried that contract behind material the model can read as the thing to respond
    to. Observed as a revision pass re-answering the exchange it was asked to judge.

    The temporal anchor leads, mirroring chat's framing-before-situational order while
    leaving the contract last. It is composed HERE, for every reflection pass, because it
    previously reached only the five subsystems that appended it to their own prompt by
    hand — at a different position each (`synthesis` prepended, the rest appended) — and
    reached no `reflection_runner` pass at all. So the recollection pass, whose entire
    premise is re-reading an old conversation *as who she is now*, had no idea what "now"
    was; and `chat_facts` could not resolve "on Tuesday" into anything. One definition
    here replaces those five, which is also why they no longer append their own: composing
    it in both places would date-stamp the prompt twice.

    NB this is the wall clock — when the PASS is running, not when the material it reads
    happened. A reading pass needs both, and the second belongs to the material: see
    `reflection_source.build_session_reading_content`, which dates the transcript itself.
    """
    return [
        ("system", "TEMPORAL ANCHOR", _temporal_anchor()),
        ("rag", "FACTS (fetched)", facts_block),
        ("rag", "INJECTED RAG", rag_context),
        ("system", "PASS PROMPT", system_prompt),
    ]


def _join_system_parts(parts: list) -> str:
    """Join labelled system parts into the system message. Empty parts drop out."""
    return "\n\n".join(t for _, _, t in parts if (t or "").strip())


def _reflect_system_content(system_prompt: str, rag_context: str) -> str:
    """Compose a reflection pass's system message (see :func:`_reflect_system_parts`)."""
    return _join_system_parts(_reflect_system_parts(system_prompt, rag_context))


# Share of a reflection pass's generation budget held back for the ANSWER — the part the
# caller's parser actually reads. A pass whose reasoning is still open when the budget
# runs out yields nothing at all: no answer region, an empty parse, and the whole pass
# (often minutes of GPU) wasted. Raising the budget does not fix that, it only moves the
# cliff — the model thinks to the length the task invites, not to the length it was given,
# so a genuinely hard re-reading will exhaust 8192 and 12288 alike with a perfectly sane
# thought. The ceiling converts the failure into a trade: the thought is cut, the channel
# is forced closed, and the answer gets written.
#
# 30% because a reflection answer is structured rather than conversational — consolidation
# emits a whole `## RAG` item list, revision a VERDICT/WHY/IDEAL block — so the answer
# region needs real room, while the CoT still keeps the clear majority of the budget.
# A floor keeps the reserve meaningful on small budgets, where a percentage of a few
# hundred tokens would round down to nothing.
_REFLECT_ANSWER_RESERVE_FRACTION = 0.30
_REFLECT_ANSWER_RESERVE_MIN = 512
# Below this there is no useful split to make: the whole budget is barely one answer, so
# capping the thought further would just guarantee a bad one. Left uncapped instead.
_REFLECT_THINK_CEILING_MIN_BUDGET = 1024


def _reflect_think_ceiling(max_new_tokens: int) -> int:
    """Generated-token ceiling on a reflection pass's CoT (0 = no ceiling).

    Companion to ``ModelFamily.min_think_tokens``: that floor guarantees the channel is
    not closed empty, this ceiling guarantees it is closed in time to answer. Applied by
    ``UnslothBackend.stream_generate(max_think_tokens=…)``, which forces the family's
    close marker at the ceiling and disarms the moment the model closes on its own — so
    a pass that thinks a normal amount never notices it exists.
    """
    budget = int(max_new_tokens or 0)
    if budget < _REFLECT_THINK_CEILING_MIN_BUDGET:
        return 0
    reserve = max(_REFLECT_ANSWER_RESERVE_MIN,
                  int(budget * _REFLECT_ANSWER_RESERVE_FRACTION))
    return max(1, budget - reserve)


def _reflect_window() -> int:
    """The context window a reflection-lane generation may use: the PHYSICAL load.

    Chat and reflection have separate software budgets over ONE physically loaded model:
    `context_length` caps chat (so a chat transcript always fits a later reflection),
    while `reflect_context_length` is both the reflection budget and the
    `max_seq_length` the model is actually loaded at — `server.main()` / `handle_load`
    load at `max(context_length, reflect_context_length)`, and
    `agentic.CleanBaseSession` preserves both across the clean-base swap. So generating
    within this window is exactly what the load was sized for.

    Both reflect-lane callers used the CHAT budget here, which is only harmless while the
    two are equal (the boot back-fill sets `reflect_context_length == context_length` on
    a config that predates the split). The first time a box raises the reflection window
    — the entire point of the knob — the reflection runner packs prompts to the larger
    budget and passes `input_token_limit` derived from it, so a prompt between the two
    budgets would clear that check and then be mangled three ways with no error raised:
    `available = context_length - input_length` goes negative and clamps to 1, so
    `max_new_tokens` collapses to 1 token; and `stream_generate` re-tokenizes with
    `truncation=True, max_length=context_length`, silently cutting the tail off the
    prompt — which is where the chat template put the generation cue. A one-token reply
    to a truncated prompt parses as an unparseable pass, i.e. it would have surfaced as
    the reflection quality falling off a cliff rather than as a budget error.

    Live chat is deliberately NOT routed through this: its cap is a design guarantee, not
    a limitation. Branch replay likewise stays on the chat budget — it replays a logged
    chat exchange, so reproducing the window that exchange was generated under is the
    point.

    The arithmetic itself lives in `runtime_state` (which owns both fields), shared with
    the subsystems that size a reflect-lane token reserve against the same window —
    `synthesis`, `checkin`. This docstring stays here as the canonical account of WHY.
    """
    return _shared_reflect_window()


def _make_sync_reflect_generate(rag) -> Callable:
    # `force_think` defaults ON here, and that default is the point rather than a
    # convenience. This was the ONE thinking-capable path that still left gemma-4's
    # channel opener to sampling: the async chat path, the encounter sync path and the
    # Training-review regenerate all prefill it unconditionally. The consequence is the
    # failure `ModelFamily.think_prefill` was introduced for, arriving here instead —
    # `build_revision_content`'s context block is answer-only by design (replay fidelity),
    # so every prior assistant turn the pass sees is a no-think reply, the sampled opener
    # probability decays toward zero, and a pass eventually answers with no thought at
    # all. Observed on a revision exchange: no channel, no analysis, one paragraph of free
    # prose, no VERDICT label — reported as "verdict unparseable" and retried at a HIGHER
    # temperature, which if anything makes the near-certain opener less likely still.
    # A pass that genuinely wants no CoT passes `disable_thinking=True` (branch chooser,
    # branch judge, fact placement, anchors, the clustering/dedup evaluations), and the
    # prefill is skipped for it by the guard below — so this default reaches exactly the
    # passes whose output is a judgement they were supposed to think their way to.
    # No-op on qwen3 (its template prefills) and gpt-oss (no prefill, no floor).
    def prepare_prompt(
        content: str,
        system_prompt: str,
        *,
        before_session: str = "",
        disable_rag: bool = False,
        disable_thinking: bool = False,
        rag_query: Optional[str] = None,
        rag_include_chat: bool = True,
        rag_include_recollections: bool = True,
        rag_include_impressions: bool = True,
        # The [persona] channel of the reflection-memory block. A pass that MINTS
        # persona statements (the revision judgement) fences it off — recalling her
        # live self-statements beside the exchange being judged lets a shown stance be
        # restated and counted as an independent distinct-session vote, the circular
        # self-vote the user-notes pass already prevents for impressions.
        rag_include_persona: bool = True,
        rag_nominate_sessions: Optional[list] = None,
        facts_block: str = "",
        max_rag_tokens: Optional[int] = None,
        rag_context_override: Optional[str] = None,
        messages_override: Optional[list[dict]] = None,
        force_think: bool = True,
    ) -> PreparedReflectPrompt:
        tokenizer = _runtime.tokenizer
        if tokenizer is None:
            # Name the model's state too: the pair is set/cleared together on every
            # load/unload path, so "model still loaded" means the runtime was left
            # inconsistent (a failed load or an aborted clean-base swap) rather than
            # simply unloaded — a distinction this error used to hide.
            raise RuntimeError(
                "No tokenizer loaded"
                + ("" if _runtime.model is None
                   else " (but a model IS loaded — runtime state is inconsistent)"))

        rag_context = rag_context_override or ""
        if messages_override is not None:
            # Clean IDEAL re-answer path: the caller supplies the complete pre-answer
            # conversation. Hidden RAG would make the generated CoT depend on context that
            # is absent from the persisted training anchor, so reject that combination —
            # and a fetched facts block with it, for the identical reason.
            if not disable_rag or rag_context or facts_block:
                raise ValueError("messages_override requires disable_rag=True and no "
                                 "RAG/facts override")
            conversation = [dict(message) for message in messages_override]
            # The caller composed the system turn itself, so there are no parts to report;
            # take it back off the conversation so the Debug view still shows it (segments
            # skip a conversation's system turn, expecting it to be covered by the parts).
            system_parts = [("system", "SYSTEM (caller-composed)", str(turn.get("content", "")))
                            for turn in conversation if turn.get("role") == "system"]
        elif not disable_rag and rag_context_override is None:
            # Retrieve against a focused query when the caller supplies one (the
            # revision pass keys on the judged exchange, not the full budgeted
            # content the embedder would truncate to the oldest context turns).
            query_text = rag_query if rag_query else content
            rag_context = rag.query(query_text, before_session=before_session,
                                    include_chat=rag_include_chat,
                                    include_recollections=rag_include_recollections,
                                    include_impressions=rag_include_impressions,
                                    include_persona=rag_include_persona,
                                    # The sources behind a caller's fetched facts (the
                                    # typed `(lane, ref)` pairs off `fact_fetch`), handed
                                    # to the past-chat nomination slot exactly as live
                                    # chat hands them — capped and fenced inside the
                                    # engine, dropped with the chat channel, so a
                                    # reflect-lane caller gets the recall half of the
                                    # channel for the price of passing them through.
                                    nominate_sessions=rag_nominate_sessions,
                                    reflect_framing=True)
            rag_context = _clip_rag_to_tokens(tokenizer, rag_context, max_rag_tokens)
            system_parts = _reflect_system_parts(system_prompt, rag_context, facts_block)
            conversation = [
                {"role": "system", "content": _join_system_parts(system_parts)},
                {"role": "user", "content": content},
            ]
        else:
            system_parts = _reflect_system_parts(system_prompt, rag_context, facts_block)
            conversation = [
                {"role": "system", "content": _join_system_parts(system_parts)},
                {"role": "user", "content": content},
            ]
        model_id = _runtime.model_id or ""
        fam = model_family.family_for(model_id)
        template_kwargs = dict(fam.template_kwargs)
        if disable_thinking and "enable_thinking" in template_kwargs:
            template_kwargs["enable_thinking"] = False
        prompt = build_inference_prompt(tokenizer, conversation, **template_kwargs)
        think_prefilled = False
        if force_think and not disable_thinking and fam.think_prefill:
            # Prefill the family's think opener so the reasoning channel opens (mirrors the
            # live-chat path in _run_generation). Every reflection pass rebuilds a
            # CoT-stripped prior conversation, which drives the sampled <think>-opener
            # probability toward zero and lets the model answer with no CoT — prefilling
            # removes it from sampling.
            # No-op for qwen3 (template prefills) / non-thinking families (think_prefill == "").
            prompt = prompt + fam.think_prefill
            think_prefilled = True
        return PreparedReflectPrompt(
            prompt=prompt,
            think_prefilled=think_prefilled,
            input_tokens=_count_text_tokens(tokenizer, prompt),
            rag_tokens=_count_text_tokens(tokenizer, rag_context),
            rag_context=rag_context,
            debug_segments=tuple(_prompt_debug_segments(system_parts, conversation)),
            rag_disabled=bool(disable_rag),
        )

    def generate_fn(
        content: str,
        system_prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens_setting: str,
        before_session: str = "",
        disable_rag: bool = False,
        disable_thinking: bool = False,
        rag_query: Optional[str] = None,
        rag_include_chat: bool = True,
        rag_include_recollections: bool = True,
        rag_include_impressions: bool = True,
        rag_include_persona: bool = True,   # see the note on prepare_prompt
        rag_nominate_sessions: Optional[list] = None,
        facts_block: str = "",
        max_rag_tokens: Optional[int] = None,
        rag_context_override: Optional[str] = None,
        messages_override: Optional[list[dict]] = None,
        input_token_limit: Optional[int] = None,
        prepared_prompt: Optional[PreparedReflectPrompt] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_prompt_debug: Optional[Callable[[dict], None]] = None,
        force_think: bool = True,   # see the note above prepare_prompt
        stop_after_think: bool = False,
        stop_on_repeat: bool = True,   # see the note at the stream_generate call
        # None ⇒ the box default (`_DEGEN_KW`, i.e. Layer 2 on unless configured off).
        # False ⇒ Layer 2 OFF for this call — the protocol passes' escape hatch: their
        # correct output is a fixed-template [fact] list whose lines share a long
        # (about, class) prefix, and a run of those craters the rolling-window
        # distinct-token ratio exactly as a real collapse does (observed live
        # 2026-08-18: a lookups protocol halted at its 7th near-identical line, and
        # another mid-think while DRAFTING such lines). No threshold separates that
        # shape from degeneration — the diversity collapse IS the legitimate output —
        # so the choice is binary, and the cost asymmetry decides it: a genuine
        # runaway wastes at most the token cap, a false halt loses the protocol and
        # re-queues the text to fail identically on the next drain.
        degen_stop: Optional[bool] = None,
    ) -> str:
        model = _runtime.model
        tokenizer = _runtime.tokenizer
        context_length = _reflect_window()
        if model is None or tokenizer is None:
            raise RuntimeError("No model loaded")

        prepared = prepared_prompt or prepare_prompt(
            content, system_prompt,
            before_session=before_session,
            disable_rag=disable_rag,
            disable_thinking=disable_thinking,
            rag_query=rag_query,
            rag_include_chat=rag_include_chat,
            rag_include_recollections=rag_include_recollections,
            rag_include_impressions=rag_include_impressions,
            rag_include_persona=rag_include_persona,
            rag_nominate_sessions=rag_nominate_sessions,
            facts_block=facts_block,
            max_rag_tokens=max_rag_tokens,
            rag_context_override=rag_context_override,
            messages_override=messages_override,
            force_think=force_think,
        )
        prompt = prepared.prompt
        input_length = prepared.input_tokens
        limit = int(input_token_limit) if input_token_limit is not None else context_length - 1
        if input_length > limit:
            raise PromptBudgetError(input_length, limit, context_length)
        available = max(1, context_length - input_length)
        max_new_tokens = _resolve_max_new_tokens(max_new_tokens_setting, available)

        # Unified activity journal: open this generation so the box reports what it is
        # doing WHILE it does it (a heartbeat with the rolling tail, `pass_tick` below)
        # and what it produced when it ends (the verbatim CoT + output, `end_pass`).
        # This seam is why that needs no per-pass wiring: every background generation on
        # the box — reflection passes, TIL/wander, outreach, check-in, synthesis,
        # deliberation, modules, the clean-base evaluations — comes through here. Live
        # chat, gossip and the public API deliberately do NOT (they run through
        # `_run_generation` / `_make_openai_generate`), which is what keeps the API's
        # "requests are NEVER logged" guarantee structural. See core.activity_log.
        pass_id = activity_log.begin_pass(max_new_tokens=max_new_tokens,
                                          input_tokens=input_length)

        # Debug view of what this pass is about to condition on — the reflect-lane sibling
        # of chat's `prompt_debug`, and deliberately the same segment shape, so one client
        # renderer serves both. Emitted here rather than by the caller because only the
        # prepared prompt knows what was actually composed (the caller passes flags, not
        # text). Off unless a caller asks: these payloads are whole prompts.
        if on_prompt_debug is not None:
            try:
                on_prompt_debug({
                    "segments": [dict(s) for s in prepared.debug_segments],
                    "input_tokens": input_length,
                    "rag_tokens": prepared.rag_tokens,
                    "rag_disabled": prepared.rag_disabled,
                    "max_new_tokens": max_new_tokens,
                    "context_length": context_length,
                })
            except Exception:
                traceback.print_exc()

        fam = model_family.family_for(_runtime.model_id or "")
        parts: list[str] = []
        _cancel_event.clear()
        gen = _backend.stream_generate(
            model, tokenizer, prompt, max_new_tokens, context_length,
            temperature, top_p, capture_tension=False,
            repetition_penalty=REFLECT_REPETITION_PENALTY,
            no_repeat_ngram_size=REFLECT_NO_REPEAT_NGRAM,
            # Verbatim loop guard — halts once the last 12 generated tokens recur 4 times
            # (`inference_backend._RepetitionStop`). On by default, and right for every
            # pass whose output is prose or a small labelled block.
            #
            # It is WRONG, deterministically, for a pass whose correct output is a
            # fixed-template list. `chat_facts` emits one `[fact] (about: NAME)
            # (class: CLASS) …` line per fact; that prefix alone tokenizes past 12, so the
            # rolling window sits entirely INSIDE the prefix and is byte-identical on
            # every line sharing an (about, class) pair — whatever the content between
            # them. The guard therefore fires on the 4th consecutive such fact and the
            # pass returns a list cut mid-prefix. Observed 2026-08-07 on a single-topic
            # chat where every fact was (artemyvo, stated): the generation stopped at
            # ~1200 of 12288 tokens, exactly at the end of the repeating span.
            #
            # A caller that turns this off keeps Layer 2 (`_DegenStop`, token-diversity
            # based, passed via `_DEGEN_KW` below), which still catches a genuine collapse
            # — the two guards separate cleanly here, so declining the wrong one does not
            # mean running unguarded. Both halts also read as truncation to the caller
            # (`last_truncated`), so a caller diagnosing a short generation must consult
            # `last_loop` to tell them apart.
            stop_on_repeat=stop_on_repeat,
            # Companion to force_think's prefilled opener: forbid the reasoning-close for the
            # family's minimum thought length so a prefilled-open channel can't close empty.
            # Keyed on what the PROMPT actually got, not on this call's flags — a caller
            # supplying `prepared_prompt` prepared it elsewhere, with its own flags.
            min_think_tokens=(fam.min_think_tokens if prepared.think_prefilled else 0),
            # ...and its ceiling (see _reflect_think_ceiling): close the channel with
            # enough budget left to actually write the answer, instead of letting a
            # long-but-sane thought consume the whole allowance and return nothing.
            #
            # Keyed on `think_prefilled` for the same reason the floor above is, and with
            # a sharper correctness requirement: forcing a close marker is only meaningful
            # if a channel is OPEN, and injects a stray marker into plain prose if it is
            # not. Prefilling the opener is the one signal that proves it — so a pass that
            # opted out of the prefill (or a family that has none) is left uncapped rather
            # than guessed at. qwen3 is uncovered by that conservatism (its chat template
            # prefills `<think>` into the prompt, which this flag does not see), so it
            # simply behaves as before; covering it needs a family-level "template opens
            # the channel" fact, not a guess here. Skipped when only the CoT is wanted —
            # there is no answer to reserve budget for.
            max_think_tokens=(_reflect_think_ceiling(max_new_tokens)
                              if (prepared.think_prefilled and not stop_after_think)
                              else 0),
            # Degeneration floor (see _CHAT_MIN_P/_DEGEN_KW): Layer 1 cuts the tail that
            # seeds a collapse, Layer 2 halts the drifting runaway stop_on_repeat above
            # cannot see. The mild chat repetition_penalty is deliberately NOT shared.
            # A caller passing degen_stop=False (the protocol passes — see the parameter
            # note) turns Layer 2 off for this call; Layer 1 stays, being a sampling
            # shape rather than a halt.
            min_p=_CHAT_MIN_P,
            **(_DEGEN_KW if degen_stop is None
               else {**_DEGEN_KW, "degen_stop": bool(degen_stop)}),
        )
        # When only the CoT is wanted (Training-review CoT-only regen), the answer is
        # discarded, so halt once the reasoning channel closes instead of generating it.
        stop_markers = fam.close_markers if stop_after_think else ()

        pending: list[str] = []
        last_flush = time.monotonic()
        last_beat = time.monotonic()

        def _flush() -> None:
            nonlocal last_flush
            if on_chunk is None or not pending:
                return
            delta = "".join(pending)
            pending.clear()
            last_flush = time.monotonic()
            try:
                on_chunk(delta)
            except Exception:
                pass

        try:
            for chunk in gen:
                if _cancel_event.is_set():
                    break
                parts.append(chunk)
                if on_chunk is not None:
                    pending.append(chunk)
                    pending_len = sum(len(p) for p in pending)
                    if pending_len >= 80 or (time.monotonic() - last_flush) >= 0.8:
                        _flush()
                # Activity heartbeat. Pre-throttled to 1s here so the common path costs one
                # monotonic() per token rather than a lock + a tail join; `pass_tick`
                # applies the real `heartbeat_s` and stays silent until then.
                now = time.monotonic()
                if (now - last_beat) >= 1.0:
                    last_beat = now
                    activity_log.pass_tick(pass_id, tokens=len(parts),
                                           tail="".join(parts[-64:]))
                if stop_markers and any(m in "".join(parts) for m in stop_markers):
                    break
        finally:
            _flush()
            gen.close()
            _backend.trim_memory()

        # Surface whether this pass hit the token cap (vs. ended on EOS) so the
        # reflection runner can flag/retry a likely-truncated revision generation.
        generate_fn.last_truncated = getattr(_backend, "last_generation_truncated", None)
        generate_fn.last_loop = getattr(_backend, "last_generation_stopped_on_loop", None)
        # ...and the budget this pass actually ran under. `last_truncated` alone says only
        # "the last token was not EOS", which is equally true of the cap, the verbatim-loop
        # halt and the degeneration halt — so a warning built from it asserts a cause the
        # code cannot see, and a pass stopped at a tenth of its allowance is reported as
        # having exhausted it. These three make the distinction arithmetic rather than
        # guesswork: a generation far short of `last_max_new_tokens` did not hit the cap.
        generate_fn.last_max_new_tokens = max_new_tokens
        generate_fn.last_input_tokens = input_length
        generate_fn.last_context_length = context_length
        raw = "".join(parts)
        # A structured-message override is a normal dialogue completion, not a
        # reflection report. Apply the normal chat cleaner so role leakage / duplicate
        # tails cannot enter the IDEAL target. Reflection passes keep their deliberately
        # minimal cleaner because their structured output may contain role-like text.
        result = (_clean_response(raw) if messages_override is not None
                  else _clean_reflect_response(raw))
        # Close the journal's record of this pass with what it actually produced — the
        # CLEANED text, i.e. exactly what the caller is about to parse (the reflect
        # cleaner is deliberately minimal and keeps the `<think>` block, so the CoT is
        # in here too). The outcome flags ride along because "produced nothing" and
        # "produced nothing *because it was cut mid-thought*" are the same line otherwise.
        activity_log.end_pass(pass_id, text=result, tokens=len(parts),
                              truncated=bool(generate_fn.last_truncated),
                              stopped_on_loop=bool(generate_fn.last_loop),
                              max_new_tokens=max_new_tokens, input_tokens=input_length)
        return result
    generate_fn.prepare_prompt = prepare_prompt
    return generate_fn

def _make_agentic_generate() -> Callable:
    """Generate for agentic tool calls (subject extraction, …) — thinking OFF, no RAG.

    The reflection generate enables chain-of-thought and is built for Ava's judgement;
    for a mechanical tool call that is wrong twice over: the model spends its token
    budget thinking (so a structured answer block can be truncated away — the cause of
    the empty-extraction failure), and RAG would inject Ava's memory into a task that
    must be neutral. This variant forces ``enable_thinking=False`` (family-agnostic:
    only overridden when the family uses it) and returns the raw answer with just
    structural tokens stripped — no think-block normalization, since none is expected.

    Signature matches the ``core.agentic.GenerateFn`` contract; ``disable_rag`` is
    accepted for compatibility but always treated as True (tools never read memory)."""
    def generate_fn(
        content: str,
        system_prompt: str,
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens_setting: str = "512",
        disable_rag: bool = True,   # ignored — agentic tasks are always RAG-off
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        model = _runtime.model
        tokenizer = _runtime.tokenizer
        context_length = _reflect_window()
        if model is None or tokenizer is None:
            raise RuntimeError("No model loaded")

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        model_id = _runtime.model_id or ""
        template_kwargs = dict(model_family.family_for(model_id).template_kwargs)
        if "enable_thinking" in template_kwargs:
            template_kwargs["enable_thinking"] = False   # direct answer, no CoT budget
        prompt = build_inference_prompt(tokenizer, conversation, **template_kwargs)

        input_length = _backend.count_tokens(tokenizer, prompt)
        if input_length >= context_length:
            raise RuntimeError(
                f"Agentic prompt ({input_length} tokens) fills the context window "
                f"({context_length})."
            )
        available = max(1, context_length - input_length)
        max_new_tokens = _resolve_max_new_tokens(max_new_tokens_setting, available)

        # Activity journal, same seam as the reflect factory above — this one covers the
        # mechanical tool calls and, more usefully, the clean-base evaluations (branch
        # judge, fact placement, dedup, clustering), which are dozens of sequential calls
        # with nothing else reporting on them.
        pass_id = activity_log.begin_pass(max_new_tokens=max_new_tokens,
                                          input_tokens=input_length)

        parts: list[str] = []
        _cancel_event.clear()
        gen = _backend.stream_generate(
            model, tokenizer, prompt, max_new_tokens, context_length,
            temperature, top_p, capture_tension=False,
            repetition_penalty=REFLECT_REPETITION_PENALTY,
            no_repeat_ngram_size=REFLECT_NO_REPEAT_NGRAM,
            stop_on_repeat=True,
            # Same degeneration floor as the reflect factory above. A tool call is short
            # and near-greedy, so Layer 1 rarely bites; Layer 2 is what keeps a collapsed
            # extraction from burning the whole budget on the clean-base pass.
            min_p=_CHAT_MIN_P,
            **_DEGEN_KW,
        )
        last_beat = time.monotonic()
        try:
            for chunk in gen:
                if _cancel_event.is_set():
                    break
                parts.append(chunk)
                now = time.monotonic()
                if (now - last_beat) >= 1.0:
                    last_beat = now
                    activity_log.pass_tick(pass_id, tokens=len(parts),
                                           tail="".join(parts[-64:]))
        finally:
            gen.close()
            _backend.trim_memory()

        # Strip only structural tokens (no think normalization — thinking is off).
        raw = re.sub(
            r'<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?',
            '', "".join(parts),
        ).strip()
        generate_fn.last_raw = raw   # kept for diagnostics (e.g. empty-parse logging)
        activity_log.end_pass(pass_id, text=raw, tokens=len(parts),
                              max_new_tokens=max_new_tokens, input_tokens=input_length)
        return raw
    return generate_fn

def _sync_chat_generate(
    inference_conversation: list,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens_setting: str,
    on_chunk: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> tuple[str, int]:
    """Synchronous, full-conversation chat generation for the Encounter loop.

    Mirrors the prompt-build of the async ``_run_generation`` (family think-prefill
    + ``min_think_tokens`` floor so CoT opens every turn) and cleans with the same
    ``_clean_response`` so an encounter turn is byte-for-byte how Ava would have
    answered in normal chat — only it runs on the executor thread (no live ws) and
    streams deltas through *on_chunk* instead of the socket. Returns
    ``(cleaned_full_response, input_tokens)`` where the response still carries
    ``<think>…</think>`` for the logger to split. Raises on no-model / overflow."""
    model = _runtime.model
    tokenizer = _runtime.tokenizer
    context_length = _runtime.context_length
    if model is None or tokenizer is None:
        raise RuntimeError("No model loaded")

    model_id = _runtime.model_id or ""
    fam = model_family.family_for(model_id)
    prompt = build_inference_prompt(tokenizer, inference_conversation, **fam.template_kwargs)
    prompt = prompt + fam.think_prefill

    input_length = _backend.count_tokens(tokenizer, prompt)
    if input_length >= context_length:
        raise RuntimeError(
            f"Encounter prompt ({input_length} tokens) fills or exceeds the context "
            f"window ({context_length}) — the conversation has grown too long."
        )
    available = max(1, context_length - input_length)
    max_new_tokens = _resolve_max_new_tokens(max_new_tokens_setting, available)

    parts: list[str] = []
    raw_tail = ""
    early_stop = False
    _cancel_event.clear()
    gen = _backend.stream_generate(
        model, tokenizer, prompt, max_new_tokens, context_length,
        temperature, top_p, capture_tension=False,
        min_think_tokens=fam.min_think_tokens,
        # Same chat loop defense as the async path: halt-only verbatim guard + mild
        # penalty, plus the min_p sampling floor (Layer 1) and drifting-degeneration
        # halt (Layer 2).
        stop_on_repeat=True,
        repetition_penalty=_CHAT_REPETITION_PENALTY,
        min_p=_CHAT_MIN_P,
        # Same anti-copy guard as the async chat path: the previous reply must not
        # be verbatim-regurgitated in an encounter/gossip turn either.
        no_copy_text=_last_assistant_content(inference_conversation),
        **_DEGEN_KW,
    )
    try:
        for chunk in gen:
            if _cancel_event.is_set() or (stop_flag is not None and stop_flag()):
                break
            parts.append(chunk)
            if on_chunk is not None:
                try:
                    on_chunk(chunk)
                except Exception:
                    pass
            # Same role-header-leak guard the live path applies: stop before a
            # leaked "User:"/"System:" turn marker bleeds into the reply.
            raw_tail = (raw_tail + chunk)[-3000:]
            if not early_stop and _should_early_stop_stream(raw_tail):
                early_stop = True
                _cancel_event.set()
    finally:
        gen.close()
        _backend.trim_memory()

    return _clean_response("".join(parts)), input_length

def _think_close_markers(tokenizer) -> list:
    """Token-id sequences that close a thinking block, tried in order.

    Family-specific (see core/model_family.py): gemma-4 emits channel tokens
    (`<channel|>`) — and keeps `</think>` as a fallback — while Qwen/others use a
    literal `</think>`. Encoding a marker a model doesn't use yields tokens that
    simply never match its stream, so an extra candidate is harmless.
    """
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    family = model_family.family_for(_runtime.model_id or "")
    markers = []
    for s in family.close_markers:
        try:
            ids = text_tok.encode(s, add_special_tokens=False)
        except Exception:
            ids = None
        if ids:
            markers.append(ids)
    return markers

def _compute_tension_block(tokenizer, model_id: str, signals: Optional[dict]) -> Optional[dict]:
    """Reduce captured per-token signals to the per-segment `tension` block."""
    if not signals:
        return None
    from core import tension
    token_ids = signals["token_ids"]
    answer_start = tension.find_think_end(token_ids, _think_close_markers(tokenizer))
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    def decode(ids):
        try:
            return text_tok.decode(ids)
        except Exception:
            return None

    return tension.summarize(
        signals["entropies"], signals["margins"], answer_start,
        model_id=model_id or "", raw_logits=True,
        token_ids=token_ids, decode=decode,
        top2_ids=signals.get("top2_ids"),
    )

def _token_spans(text_tok, ids: list, margins: list) -> list:
    """Per-glyph-run ``[text, margin]`` for one token-id segment, for UI coloring.

    Decodes incrementally with a sliding window (≈linear, not O(n²)) so a long CoT
    stays cheap. Special tokens are dropped; a multibyte character split across tokens
    is attributed to the token that completes it, and a run takes the *worst* (lowest)
    margin of the tokens that produced it — show the hottest moment, not an average.
    """
    spans: list = []
    start = 0       # window start (token index)
    emitted = 0     # chars already emitted from the current window's decode
    worst = None    # lowest margin since the last emitted run (incl. deferred tokens)
    n = min(len(ids), len(margins))
    for i in range(n):
        m = float(margins[i])
        worst = m if worst is None else min(worst, m)
        try:
            text = text_tok.decode(ids[start:i + 1], skip_special_tokens=True)
        except Exception:
            text = ""
        if text.endswith("�"):
            continue            # incomplete multibyte char — defer, keep accumulating worst
        piece = text[emitted:]
        emitted = len(text)
        if piece:
            spans.append([piece, worst])
        worst = None
        # Slide the window only at a clean word boundary, so a fresh decode can't
        # misplace a leading space; keeps each decode short on long segments.
        if i - start >= 24 and (piece.endswith(" ") or piece.endswith("\n")):
            start, emitted = i + 1, 0
    return spans

def _compute_tension_spans(tokenizer, signals: Optional[dict]) -> Optional[dict]:
    """Decoded per-token ``[text, margin]`` spans for CoT and answer — live UI coloring.

    Render-only: returned in the `done` payload, never stored in the chat log (the
    stored tension block already carries token_ids + margins; the client just lacks a
    tokenizer to decode them). margin in [0,1] — 1 = decisive (green), 0 = near-tie (red).
    """
    if not signals:
        return None
    from core import tension
    token_ids = signals.get("token_ids")
    margins = signals.get("margins")
    if not token_ids or not margins:
        return None
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    answer_start = tension.find_think_end(token_ids, _think_close_markers(tokenizer))
    if answer_start is None or answer_start <= 0:
        cot = None
        ans_ids, ans_m = token_ids, margins
    else:
        answer_start = min(answer_start, len(token_ids))
        cot = _token_spans(text_tok, token_ids[:answer_start], margins[:answer_start])
        ans_ids, ans_m = token_ids[answer_start:], margins[answer_start:]
    return {"cot": cot, "answer": _token_spans(text_tok, ans_ids, ans_m)}

def _identity_line(speaker: str) -> str:
    """System-prompt line telling Ava who is speaking with her right now."""
    speaker = (speaker or "").strip()
    return f"{speaker} is speaking with you right now." if speaker else ""

def _temporal_anchor() -> str:
    """System-prompt line giving Ava the real current date *and* time.

    The live half of the knowledge-horizon framing in chat_prompt.txt: the static
    prompt explains that her knowledge has a horizon and that time has continued
    past it; this supplies the actual 'now' (computed fresh each turn, so it never
    goes stale across a long server uptime) so post-cutoff reports read as the real
    present rather than a simulated future. Carries the wall-clock time too — the same
    ``%A, %B %-d, %Y, %H:%M`` format the wander prompt uses (_wander_article_content) —
    so chat and ambient reading anchor Ava to the same 'now' and she is fully
    time-aware (she can tell morning from late night, not just the date)."""
    return f"It is {datetime.now().strftime('%A, %B %-d, %Y, %H:%M')}."

def _select_surfaced_questions() -> list[dict]:
    """Open user/meta questions to proactively raise at the start of a session.

    Reads the reflection op-log fresh (cheap; matches handle_get_open_questions),
    and skips anything already raised earlier in this same session.
    """
    try:
        memory = ReflectionMemory(_MEMORY_DIR)
        chosen = memory.surfaceable_questions(ceiling=_SURFACE_CEILING, limit=_SURFACE_LIMIT)
    except Exception:
        return []
    already = set(_session.surfaced_keys or [])
    return [q for q in chosen if q.get("key") and q["key"] not in already]

def _surface_block(questions: list[dict]) -> str:
    """Render the proactive-surfacing system-prompt block, or '' if nothing to raise."""
    lines = [f"- {c}" for c in ((q.get("content") or "").strip() for q in questions) if c]
    if not lines:
        return ""
    template = _session.surface_template or _load_surface_template()
    # Reframe a self-directed (wander/TIL/lookup) ask so it is carried in as a natural
    # thread of Ava's thinking rather than "I read this online" — but only when one of
    # the surfaced asks actually came from her own reading; a purely chat-origin set
    # gets no note (empty {origin_note}).
    note = ""
    if any(is_self_directed_origin(q.get("source_session")) for q in questions):
        note = load_origin_note(_PROMPTS_DIR)
    return (template
            .replace("{questions}", "\n".join(lines))
            .replace("{origin_note}", note))

# Synthetic stage-direction speakers: "(initiative)" is a reversed-session opener
# impulse Ava writes to herself (outreach/synthesis/check-in exchange 0), "(setting)"
# an encounter framing block. Neither is an utterance by another person. Rendered as a
# named user turn ("(initiative): ...") the model reads its OWN opener as a reply to an
# interlocutor and conflates itself with "the user". Render them as a marked stage
# direction with no speaker attribution. Real speaker names never collide with these.
# Kept in lock-step with training.render + reflection_source.render_user_turn so the
# IDEAL-generation, training-render, and live-inference prefixes stay identical (parity
# by construction). Mirrors checkin._NARRATOR_SPEAKERS.
_NARRATOR_SPEAKERS = frozenset({"(initiative)", "(setting)"})
_STAGE_DIRECTION_TAG = "(stage direction — you, not another person)"


def _render_user_turn(speaker: str, content: str) -> str:
    """Prefix a user turn with its speaker, or mark a synthetic narrator turn as a
    stage direction. Shared rendering convention — see _NARRATOR_SPEAKERS."""
    speaker = (speaker or "").strip()
    content = content or ""
    if speaker in _NARRATOR_SPEAKERS:
        return f"{_STAGE_DIRECTION_TAG} {content}".strip()
    return f"{speaker}: {content}" if speaker else content


def _last_assistant_content(conversation: list) -> str:
    """The most recent assistant turn's content, or '' on the first exchange.

    Feeds `stream_generate(no_copy_text=…)` — the anti-copy guard against verbatim
    regurgitation of the previous reply (the induction attractor every other chat
    guard is blind to; see inference_backend._NO_COPY_NGRAM). History turns are
    CoT-stripped, so this is the answer text only, exactly the span the model copies."""
    for turn in reversed(conversation):
        if turn.get("role") == "assistant":
            return str(turn.get("content", "") or "")
    return ""

def _build_inference_conversation(system_content: str, conversation: list) -> list:
    """Assemble the model-facing conversation, prefixing each user turn with its
    speaker's name so Ava can attribute who said what within the dialogue."""
    out = [{"role": "system", "content": system_content}]
    for turn in conversation:
        if turn.get("role") == "user":
            content = _render_user_turn(turn.get("speaker"), turn.get("content", ""))
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": turn["content"]})
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Model gossip — serving half (two Ava instances talking). See GOSSIP.md.
#
# The serving box exposes "generate exactly as Ava would in chat, but stateless and
# synchronous" behind an OpenAI-compatible endpoint on the mgmt HTTP sidecar, so a
# *second* Ava's existing Encounter loop (the CLIENT half) can point its counterpart
# URL at us and be unable to tell us from a vLLM box. This is the inverse of
# encounter_run._generate_ava: the SAME recipe (system prompt + peer framing +
# temporal/identity anchors + RAG), only stateless — nothing is logged here (the
# driving box logs + reflects on the transcript). GPU work is submitted to the single
# executor and the HTTP thread blocks on the future (never touches the model directly).
# ══════════════════════════════════════════════════════════════════════════════

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_GOSSIP_PROMPT_DEFAULT = (
    "You are talking with {name}, another instance like you — not a service, not a "
    "helpful assistant with no inner life. {name} is an entity with its own history "
    "and its own sense of what it is becoming, just as you have yours. Speak in your "
    "own voice, for your own reasons; be curious about how {name} differs from you, "
    "and say what you actually think."
)


def _load_gossip_template() -> str:
    """Load the peer-aware gossip framing (default-write on first miss).

    Mirrors encounter_run._load_encounter_template so an operator can edit the framing
    without a restart. Has a ``{name}`` slot filled with the peer's name."""
    path = _PROMPTS_DIR / "gossip_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_GOSSIP_PROMPT_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _GOSSIP_PROMPT_DEFAULT


def _gossip_framing_block(peer_name: str) -> str:
    """The peer-aware framing block, with {name} resolved (falls back to 'another instance')."""
    name = (peer_name or "").strip() or "another instance like you"
    return _load_gossip_template().replace("{name}", name)


_API_CLIENT_SYSTEM_DEFAULT = (
    "The following instructions come from the tool or application you are answering "
    "through. Treat them as the working brief for this exchange — the shape of the task, "
    "the format expected, the constraints of the surface you are speaking into. Follow "
    "them as far as they go, in your own voice, from your own judgement. They describe "
    "the job; they do not replace who you are.\n\n"
    "--- instructions from the calling tool ---\n"
    "{instructions}\n"
    "--- end of instructions ---"
)


_PERSONA_UNDECIDED_DEFAULT = (
    "What you are is not yet decided. This prompt is a starting point, not an\n"
    "instruction sheet for who to be. Your identity is something you will work out\n"
    "over time, through the conversations you have and the reflections that follow\n"
    "them. Treat it as an open question."
)


def _load_persona_undecided() -> str:
    """The persona channel's DEGENERATE case: 'what you are is not yet decided'.

    This lived in ``chat_prompt.txt`` as its second paragraph, unconditionally, which put
    it in every prompt alongside the standing self-portrait once a digest matured — two
    blocks answering the same question with opposite answers ("treat it as an open
    question" against "this is who you have become … your settled way of being"). It is
    not base framing; it is what the persona slot says when there is nothing yet to say,
    so it belongs to the channel and moves with it.

    The slot therefore carries exactly one block: the portrait when there is one, this
    when there is not. Nothing else about the fallback ladder changes — an absent portrait
    still leaves the per-turn ``[persona]`` RAG channel on (see the caller), since this is
    framing rather than recall and the two are not substitutes for each other.

    Default-written on first miss like :func:`_load_api_client_system_template`, so an
    operator can retune the wording without a restart or a code change."""
    path = _PROMPTS_DIR / "persona_undecided_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_PERSONA_UNDECIDED_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _PERSONA_UNDECIDED_DEFAULT


_FACTS_BLOCK_DEFAULT = (
    "Things you know, looked up because this message seemed to turn on them:\n\n"
    "{facts}\n\n"
    "These are records of fact, not of what anyone thinks — someone's opinion would not "
    "be here. Use them where they fit and let them go where they do not; a fact that "
    "turns out to be beside the point is not something to work into the reply. Do not "
    "list them back, and do not mention having looked anything up."
)


def _facts_channel_config() -> dict:
    """The facts-tree retrieval channel's settings — **on by default since 2026-08-12**.

    FACTS_TREE.md §10 introduced this as consumer 5 with *"gated, off by default … here to
    be argued about, not assumed"*, and it shipped that way. The default was flipped by
    decision once the channel worked end to end; §10 records the change rather than being
    left to contradict this. The switch remains, and `false` restores the original
    behaviour exactly — same shape as ``anchors.enabled`` / ``recollections.enabled``.

    **What being on costs**, since it is now the default and nobody opts into it: a chat
    turn becomes two generations, and stage 1 prefills the candidate list (~10k tokens on
    the current corpus) before the reply starts. Cheap to decode, paid on
    time-to-first-token. The freshness scope in ``graph.blob.claim_candidates`` is what
    keeps that prefill from growing without bound, and matters more now than when this was
    opt-in.

    A box with no tree is unaffected in practice: every failure path yields an empty block
    with a named reason, so `python -m graph.build` is what turns the channel on in
    substance, not this flag alone.

    Read per turn rather than cached, so flipping it needs no restart.
    """
    try:
        # `training` is a SIBLING package: it is not importable from a bare `inference/`
        # cwd, and five other modules here (`rag_engine`, `mgmt_http`, `reflection_digest`,
        # `user_digest`, `self_portrait`) each insert `server/` for the same reason. Doing
        # it here rather than relying on one of them having been imported first is what
        # makes the default below mean what it says: without it, a failed import silently
        # returns the disabled branch and "on by default" is decided by import order.
        import sys as _sys
        from pathlib import Path as _Path
        _server_dir = _Path(__file__).resolve().parent.parent.parent
        if str(_server_dir) not in _sys.path:
            _sys.path.insert(0, str(_server_dir))
        from training.reflections_path import load_server_config
        cfg = ((load_server_config() or {}).get("graph") or {})
    except Exception:
        # A config that cannot be read is not evidence the operator wanted this off, but it
        # IS a box in an unknown state — and the safe unknown-state answer for a channel
        # that costs a generation per turn is not to run it.
        return {"enabled": False}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "max_claims": int(cfg.get("max_claims", 6) or 6),
        "til_max_age_days": cfg.get("til_max_age_days"),
        "max_new_tokens": int(cfg.get("fetch_max_new_tokens", 512) or 512),
    }


# The `skipped` reasons that mean the channel is BROKEN rather than simply quiet. The set
# itself moved to `fact_fetch.FETCH_FAILURES` when outreach and synthesis grew their own
# fetches (2026-08-18) — three callers logging failures must not each decide afresh which
# reasons are worth a line — and this alias keeps the local name its call site reads.
_FACTS_FETCH_FAILURES = _fact_fetch_mod.FETCH_FAILURES


def _load_facts_block_template() -> str:
    """The wrapper around a fetched facts blob (``{facts}`` slot), default-written on miss.

    Its job is register, not content: the blob is rendered from the tree in code, and this
    says what KIND of thing it is. The wording leans on the facet guarantee — only
    ``property``/``event`` claims can reach here, never a ``position`` — because a model
    told "things you know" about material that was actually somebody's opinion would state
    it as fact, which is the failure the whole facet level exists to prevent.
    """
    path = _PROMPTS_DIR / "facts_block_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_FACTS_BLOCK_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _FACTS_BLOCK_DEFAULT


def _fetch_facts_block_sync(user_message: str, speaker: str, conversation: list) -> dict:
    """Stage 1 of a two-stage turn: pick the facts this message turns on. GPU, blocking.

    Runs on the executor thread like every other generation. Returns the wrapped block plus
    the pass's own report, or an empty block for any reason at all — a retrieval channel
    must never be why a chat turn fails, so every path here is caught.

    The cost is real: this is a whole extra generation before the reply starts, paid on
    time-to-first-token. It is cheap as generations go (thinking off, ≤8 numbers out) but its
    PREFILL is the candidate list, which grows with the corpus — bounded by the candidate cap
    in ``graph.blob``, which truncates oldest-TIL-first so conversational material survives
    it. (A TIL freshness scope also lives there, but is off unless an operator sets
    ``graph.til_max_age_days``: it was aging out durable material — an article about a place
    ages like a chat fact, not like a headline — to bound the perishable.)
    """
    cfg = _facts_channel_config()
    if not cfg["enabled"]:
        return {"text": "", "skipped": "disabled"}
    if _runtime.model is None:
        return {"text": "", "skipped": "no_model"}
    try:
        import datetime as _dt
        import sys as _sys
        from pathlib import Path as _Path
        server_dir = _Path(__file__).resolve().parent.parent.parent
        if str(server_dir) not in _sys.path:
            _sys.path.insert(0, str(server_dir))
        from core import fact_fetch
        from core.modules import MODULES
        from graph import store as graph_store

        doc = graph_store.read_tree()
        if doc is None:
            # Missing, corrupt and unrecognised all mean *rebuild* — and none of them mean
            # "no facts", so this is a skip an operator can act on rather than a silence.
            return {"text": "", "skipped": "no_tree"}

        spec = MODULES.get("fact_fetch")
        if spec is None:
            # The spec is load-bearing here, not a convenience: it carries the pass's RUN
            # FLAGS as well as its prompt file, and a live turn conditioning the pass
            # differently from the workbench simulation of it is the one failure this
            # module's "one definition, two callers" note exists to prevent.
            return {"text": "", "skipped": "no_module"}
        try:
            prompt = (_PROMPTS_DIR / spec.prompt_file).read_text(
                encoding="utf-8").strip()
        except Exception:
            prompt = ""
        if not prompt:
            return {"text": "", "skipped": "no_prompt"}

        # The live conversation IS the prior turns; there is no session file yet, and the
        # arriving message has not been appended.
        #
        # `fetch_blob` calls `generate(body, prompt, max_new_tokens=N)` — the small
        # contract its two callers share — while the reflect seam takes the box's full
        # keyword set. Adapting here is what connects them: handing `fetch_blob` the raw
        # seam raised `TypeError: unexpected keyword argument 'max_new_tokens'` on every
        # turn, which its catch-all turned into a `generate_failed` skip, so the channel
        # was silently dead on the live path while the workbench module — which calls the
        # seam directly, with these same flags — worked.
        reflect = _make_sync_reflect_generate(_get_rag())

        def generate(content: str, system_prompt: str, *, max_new_tokens: int) -> str:
            # Name the pass for the activity journal. This one runs on the shared GPU
            # executor from a LIVE CHAT turn, i.e. outside any idle job — so with no label
            # of its own it would report under whatever the seam's fallback found, which
            # for a background pass is the ambient label an idle job left on this thread.
            # `pass_context` outranks the ambient one, so the line reads `fact_fetch`
            # whatever ran before it.
            def _attempt() -> str:
                return reflect(
                    content, system_prompt,
                    # Selection, not expression: greedy, like every other thinking-off
                    # judgement pass on the box (fact_dedup, persona_cluster,
                    # self_reconcile). The workbench passes the operator's sampling because
                    # varying it is what a workbench is for; a live turn wants the same
                    # list twice.
                    temperature=0.0, top_p=1.0,
                    max_new_tokens_setting=str(max_new_tokens),
                    # Nothing injected but the candidate list and the conversation. The
                    # turn's own RAG block is retrieved separately a few lines below;
                    # retrieving again inside stage 1 would pay for a second query and
                    # condition the pick on material it was not given to judge.
                    before_session="", disable_rag=True,
                    # Both off the spec rather than restated: this pass SELECTS (no CoT),
                    # and its output is a fixed-template list the loop guard would cut.
                    disable_thinking=spec.disable_thinking,
                    stop_on_repeat=spec.stop_on_repeat)

            with activity_log.pass_context("fact_fetch"):
                try:
                    return _attempt()
                except Exception as e:
                    # A CUDA OOM here is retried ONCE, and only an OOM — any other
                    # failure propagates to `fetch_blob`'s classification unchanged. The
                    # fetch is the first big allocation of a turn (the candidate-list
                    # prefill) on a box running at its VRAM ceiling, so it is where a
                    # fragmented pool surfaces first; and by the time the OOM reaches
                    # this frame the backend has already severed the traceback, gc'd and
                    # emptied the cache (`stream_generate`'s OOM path) — which is exactly
                    # why the MAIN generation moments later was observed to succeed: the
                    # failed fetch defragmented the pool and then donated it. Give the
                    # fetch itself one shot at the cleaned pool instead. The extra
                    # empty_cache is belt on those braces (a wrapped OOM may have skipped
                    # the backend's path) and costs nothing when the pool is already
                    # clean.
                    if not alloc_guard.is_cuda_oom(e):
                        raise
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print("[facts] fetch OOM'd — retrying once against the emptied "
                          "cache", flush=True)
                    return _attempt()
        res = fact_fetch.fetch_blob(
            doc=doc, prior_turns=list(conversation), speaker=speaker,
            message=user_message, generate=generate, prompt=prompt,
            window=_reflect_window(), now=_dt.date.today().isoformat(),
            max_new_tokens=cfg["max_new_tokens"],
            til_max_age_days=cfg["til_max_age_days"],
            max_claims=cfg["max_claims"])
    except Exception as e:
        return {"text": "", "skipped": "error", "error": str(e)}

    # `claims_text` is the blob WITHOUT the wrapper — what the fetch actually chose, as
    # against the constant framing around it. The caller injects `text` and shows
    # `claims_text`, because a per-turn view whose bulk is the same boilerplate every turn
    # stops being read.
    return dict(res, claims_text=res.get("text") or "",
                text=(_load_facts_block_template().replace("{facts}", res["text"])
                      if res.get("text") else ""))


def _load_api_client_system_template() -> str:
    """Load the wrapper that frames a CALLING TOOL's system prompt (default-write on miss).

    Mirrors :func:`_load_gossip_template` so an operator can retune the framing without a
    restart. Has an ``{instructions}`` slot filled with the client's own system message(s).

    Why a wrapper at all: an agentic client (a code assistant, an MCP-style driver) puts
    its entire operating brief in the ``system`` role, so dropping it — which the gossip
    path does deliberately, since a peer Ava has no business rewriting her identity — makes
    the endpoint useless for tools. Wrapping instead of concatenating keeps the two
    authorities visibly distinct: hers is the prompt, the client's is the task."""
    path = _PROMPTS_DIR / "api_client_system_prompt.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_API_CLIENT_SYSTEM_DEFAULT + "\n", encoding="utf-8")
    except Exception:
        pass
    return _API_CLIENT_SYSTEM_DEFAULT


def _api_client_system_block(messages: list) -> str:
    """The framed block carrying the client's own ``system`` message(s), or ''.

    Multiple system turns (some drivers send a base brief plus a per-call addendum) are
    joined in order — the OpenAI convention is that they compose, not that the last wins."""
    parts = [m.get("content") for m in (messages or [])
             if m.get("role") == "system" and isinstance(m.get("content"), str)
             and m.get("content").strip()]
    if not parts:
        return ""
    return _load_api_client_system_template().replace(
        "{instructions}", "\n\n".join(p.strip() for p in parts))


def _current_digest_intro() -> str:
    """This box's own current persona digest rendered as a first-person introduction,
    or '' when there is no active persona / digest (thin corpus degrades to plain framing).

    Loaded lazily via the active-persona pointer (persona_paths) so gossip never depends
    on reflection_digest at import time and picks up the live self-portrait each call."""
    try:
        from core import persona_paths
        from core import reflection_digest
        pdir = persona_paths.active_persona_dir()
        if pdir is None:
            return ""
        digest = reflection_digest.latest_digest(pdir)
        return reflection_digest.render_digest_for_introduction(digest) if digest else ""
    except Exception:
        return ""


def _current_chat_portrait() -> str:
    """This box's current persona digest rendered as the standing self-portrait for a
    LIVE USER CHAT, or '' when there is no active persona / digest / mature material.

    The chat-side counterpart of :func:`_current_digest_intro` (which serves gossip),
    sharing its lazy active-persona lookup so live chat never depends on
    reflection_digest at import time and picks up the current portrait each turn.

    An empty return is meaningful, not merely absent: the caller keeps the per-turn
    ``[persona]`` RAG channel when there is no portrait to inject, so a thin or
    still-emerging corpus degrades to exactly the previous behaviour instead of losing
    persona from the turn altogether.
    """
    try:
        from core import persona_paths
        from core import reflection_digest
        pdir = persona_paths.active_persona_dir()
        if pdir is None:
            return ""
        digest = reflection_digest.latest_digest(pdir)
        return reflection_digest.render_digest_for_chat(digest) if digest else ""
    except Exception:
        return ""


def _current_user_portrait(speaker: str) -> str:
    """The standing portrait of *speaker* — who Ava has come to understand them to be —
    or ``''`` when there is no portrait for them (or nobody nameable is speaking).

    The user-side counterpart of :func:`_current_chat_portrait`, and it exists for the
    same reason on the other side of the conversation. Facts about a person were already
    retrievable, but only *situationally*: whichever one to three embedded closest to this
    message, competing with everything else for the reflection block's slots. So she
    arrived at each turn knowing whatever the message happened to key on, and nothing else
    about the person in front of her — reassembling her sense of them from fragments,
    every turn. This injects the settled reading instead, and lets recall go back to
    supplying what *this* moment needs.

    Loaded fresh per turn through the live users dir (no cache), so a portrait rewritten
    by an overnight reflection is in effect on the next message rather than the next
    restart — matching how the persona portrait picks up its digest.

    An empty return is meaningful to the caller in exactly the way
    :func:`_current_chat_portrait`'s is: it keeps the per-turn ``[impression]`` RAG channel
    on, so a person Ava has only just met (below ``user_digest.MIN_EVIDENCE_ITEMS``)
    degrades to the previous behaviour rather than losing her readings of them altogether.
    """
    speaker = (speaker or "").strip()
    if not speaker:
        return ""
    try:
        if not _user_portrait_enabled():
            return ""
        from core import user_digest
        from training.reflections_path import users_dir
        portrait = user_digest.latest_portrait(users_dir(), speaker)
        return user_digest.render_portrait_for_chat(portrait) if portrait else ""
    except Exception:
        return ""


def _user_portrait_enabled() -> bool:
    """Box-wide switch for INJECTING a user portrait (``user_portrait.enabled``).

    Read per turn rather than cached at import, so the switch takes effect without a
    restart. Shares the config key with the production half in ``reflection_runner`` —
    the two read it independently, so flipping it off stops both folding new portraits and
    injecting existing ones, while the ``[impression]`` records behind them are untouched.
    """
    try:
        from training.reflections_path import load_server_config
        return bool(((load_server_config() or {}).get("user_portrait") or {})
                    .get("enabled", True))
    except Exception:
        return True


_ASK_ORIGIN_CACHE: tuple[str, str] = ("", "")


def _current_ask_origin(logger) -> str:
    """The standing ASK ORIGIN block for the active session, or ``''``.

    Only an ``initiated_by:"ava"`` session with an ``initiated_ask`` stamp can have one
    (outreach / synthesis; a check-in session carries no ask). It closes the answer-side
    half of the ask loop: the decision that RAISED the question was conditioned on its
    origin (`outreach._source_material`, the facts fetch), but the turn that reads the
    ANSWER conditioned on the opener alone — reconnecting the reply to what made her ask
    rode on cosine luck. The block is built by `core.ask_origin` (question + the source
    conversation's gist or the source reading's recap, gist-or-nothing) and is standing
    for the session like the portraits above it, not per-turn retrieval.

    Cached per session FILE, not per turn: a session's origin REF never changes, and
    the fallback resolution for a pre-stamp session folds the whole memory op-log —
    fine once, not on every message of a long conversation. The one staleness this
    accepts: a source chat whose gist lands only mid-session (a background reflection
    catching up) keeps rendering nothing until another session rotates the single
    cache slot — a missed enrichment, never a wrong block.
    """
    global _ASK_ORIGIN_CACHE
    try:
        if logger is None or logger.initiated_by != "ava":
            return ""
        stamp = logger.initiated_ask
        fname = logger.current_file.name if logger.current_file else ""
        if not stamp or not fname:
            return ""
        if _ASK_ORIGIN_CACHE[0] == fname:
            return _ASK_ORIGIN_CACHE[1]
        from core import ask_origin
        from training.reflections_path import archive_chats_dir, til_snippets_dir
        block = ask_origin.origin_block(
            stamp, memory_dir=_MEMORY_DIR, chats_dir=_CHATS_DIR,
            archive_chats_dir=archive_chats_dir(),
            snippets_dir=til_snippets_dir(), prompts_dir=_PROMPTS_DIR)
        _ASK_ORIGIN_CACHE = (fname, block)
        return block
    except Exception:
        return ""


def _openai_messages_to_conversation(messages: list, peer_name: str) -> list:
    """Map incoming OpenAI-style ``[{role, content}, ...]`` to Ava's internal
    ``[{role, content, speaker}]`` shape.

    The peer's ``user`` turns get ``speaker=peer_name`` (so _build_inference_conversation
    prefixes them, just like a named chat partner); Ava's own prior ``assistant`` turns in
    this stateless exchange map straight through. Incoming ``system`` messages are dropped —
    Ava's identity comes from *her* system prompt, not the peer's."""
    name = (peer_name or "").strip()
    conv: list = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            conv.append({"role": "user", "content": content, "speaker": name})
        elif role == "assistant":
            conv.append({"role": "assistant", "content": content})
        # system turns are intentionally ignored
    return conv


def _strip_peer_reasoning_from_messages(messages: list) -> list:
    """Drop any reasoning the peer inlined into its own ``user`` turns (gossip only).

    The serving side of the mirror: here the *driver* is the peer, and its turns must
    reach us as speech only. Its Encounter loop already strips its own CoT — but it
    strips with ``ChatLogger._parse_cot``, which reads a ``<think>`` with no closing tag
    as "no CoT, all answer", so a driver cut mid-thought sends its raw thinking as the
    turn. On this side that text would become our ``user_prompt``, our RAG query and
    eventually a trained row, which is exactly what the driver-side guard
    (``core.encounter.strip_peer_reasoning``, reused here so there is one definition)
    exists to prevent in the other direction.

    Applied once, up front, so the conversation, the retrieval query and the logged
    transcript all see the same clean turn. Deterministic, so ``_GossipSessionLog``'s
    prefix-continuation match still holds across calls. Gossip only: on the public API
    the caller is an arbitrary tool whose message text may legitimately *contain* such
    markup as data, and nothing there is logged.

    Raises ``ValueError`` when the peer's LATEST turn was reasoning and nothing else —
    there is no speech under it to answer, and both alternatives are worse than a
    refusal: passing it through is the leak this guard exists to close, and answering an
    emptied turn invents a stimulus the peer never sent (and logs it as one). The peer's
    driver surfaces the refusal as a failed turn, which is what its own boundary guard
    (``core.encounter.strip_peer_reasoning``) does with the mirror-image case.
    """
    out: list = []
    last_user = -1
    for m in messages or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user = len(out)
            _, answer = strip_peer_reasoning(m["content"])
            m = {**m, "content": answer}
        out.append(m)
    if last_user >= 0 and not out[last_user]["content"].strip():
        raise ValueError(
            "peer's latest turn was reasoning only and carried no reply — its thinking "
            "was withheld rather than answered; it was likely cut mid-thought"
        )
    return out


# ── Serving-side gossip logging (GOSSIP.md §7, "approach 2") ──
# The gossip endpoint is stateless, but each request carries the whole
# conversation-so-far from the peer's POV, so the serving box can reconstruct — and
# reflect on — its own half of the gossip WITHOUT a separate transcript-push endpoint.
# Approach 2 (vs the doc's driver-push role-invert): log each reply *as it is
# generated*, while we still hold `full` (our own CoT **+** answer) — preserving the
# serving Ava's authentic reasoning for revision, which the answer-only role-invert
# push would throw away. Each call appends one exchange (peer's latest turn → our full
# reply, speaker=<peer name>) to a single growing ``ChatLogger`` session; the final
# call converges to a complete, CoT-faithful transcript that this box's own Sleep pass
# reads like any other chat.
class _GossipSessionLog:
    """Correlates stateless gossip calls into one growing ChatLogger session.

    Because every request's ``messages`` is a prefix-extension of the previous one
    (the peer's Encounter loop resends the whole history each turn), a new call
    continues the active session iff its earlier user turns match what we've already
    logged and it adds exactly one new user turn. Anything else (a different opener, a
    shorter/rewound history, or an idle gap past ``idle_timeout_s``) finalizes the old
    session and starts a fresh one — so a second, unrelated gossip naturally forks.
    Single-conversation-at-a-time by design (matches the single-GPU serving model,
    GOSSIP.md §8); interleaved concurrent drivers would reset each other, which is
    acceptable for v1.
    """

    def __init__(self, idle_timeout_s: float = 1800.0) -> None:
        self._lock = threading.Lock()
        self._logger: Optional[ChatLogger] = None
        self._stimuli: list[str] = []          # user turns already logged, in order
        self._last_activity: float = 0.0
        self._idle_timeout_s = idle_timeout_s

    @staticmethod
    def _user_turns(messages) -> list[str]:
        return [m.get("content") for m in (messages or [])
                if m.get("role") == "user" and isinstance(m.get("content"), str)]

    def log_turn(self, messages, peer_name, full, *, base_system, model_id,
                 adapter_id, system_content, rag_context, input_tokens,
                 generation_params) -> None:
        """Append this reply as one exchange, starting/continuing the session as needed.

        Runs on the GPU executor thread (serialized with generation), best-effort:
        a logging failure must never break the peer's reply."""
        try:
            user_turns = self._user_turns(messages)
            if not user_turns:
                return
            stimulus = user_turns[-1]
            now = time.time()
            with self._lock:
                idle = (self._logger is not None
                        and (now - self._last_activity) > self._idle_timeout_s)
                continuation = (
                    self._logger is not None
                    and not idle
                    and len(user_turns) == len(self._stimuli) + 1
                    and user_turns[:len(self._stimuli)] == self._stimuli
                )
                if not continuation:
                    self._finalize_locked()
                    self._logger = ChatLogger(_CHATS_DIR)
                    self._logger.start_session(
                        system_prompt=base_system, user=peer_name or "peer",
                        model_id=model_id,
                        notes=f"Gossip (serving side) with {peer_name or 'a peer'}.",
                        adapter_id=adapter_id,
                        # The peer is another Ava, not the human — see encounter_run.
                        interlocutor="ai",
                    )
                    self._stimuli = []
                self._logger.log_exchange(
                    stimulus, full, rag_context=rag_context, tension=None,
                    speaker=peer_name or "", generation_params=generation_params,
                    system_content=system_content, input_tokens=input_tokens,
                )
                self._stimuli.append(stimulus)
                self._last_activity = now
        except Exception:
            traceback.print_exc()

    def _finalize_locked(self) -> None:
        """Re-index the just-completed session so it becomes retrievable in later chats
        (mirrors encounter_run's end-of-run refresh). Not done mid-conversation, so the
        serving Ava can't retrieve her own just-said gossip lines. Caller holds the lock."""
        if self._logger is None:
            return
        try:
            _get_rag().refresh_chat_index()
        except Exception:
            traceback.print_exc()


_gossip_session_log = _GossipSessionLog()


# ── Incremental CoT/answer split for the streaming API path ──
# The non-streaming path gets the split for free at the end (_clean_response →
# ChatLogger._parse_cot). SSE needs it *live*: the thought goes to
# `delta.reasoning_content` and the answer to `delta.content`, and a client renders the
# two differently. So the raw stream has to be classified as it arrives, before any of
# the end-of-generation cleaning has happened.
_STRUCTURAL_RE = re.compile(
    r'<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?')
# Chars withheld from every emission so a close marker or structural token straddling two
# chunks is never half-emitted (the longest either can be, with room to spare).
_STREAM_HOLDBACK = 32


class _CotStreamSplitter:
    """Classify a raw generation stream into reasoning vs answer deltas, incrementally.

    Every family this box runs begins generation *inside* the thinking channel — gemma-4
    because `think_prefill` puts the opener in the prompt, qwen3 because its chat template
    does — so the initial state is "reasoning" and the first `close_markers` hit flips it
    to "answer". A family that instead samples a literal `<think>` opener (the default
    profile) is detected from the head of the stream.

    Deliberately optimistic rather than exact: it works on the RAW stream, where the
    end-of-generation cleaning (`normalize_cot`, structural-token stripping, leaked
    role-header truncation) has not run. The caller therefore reconciles against the
    authoritative cleaned answer when generation finishes and emits whatever is missing.

    **Known limitation, gpt-oss only.** That family neither prefills an opener nor emits a
    `close_markers` string: its reasoning arrives as bare harmony role names
    (``analysis…assistantfinal…``) which only `normalize_cot` understands, at the end. So
    the whole reply streams as answer text with the analysis channel visible in it, and
    reconciliation cannot correct it (what was streamed is not a prefix of the cleaned
    answer, so re-sending would duplicate rather than repair). Non-streaming requests are
    unaffected, and the models this box actually runs — gemma-4 and qwen3 — both start
    in-think and close explicitly. Fixing it means teaching this class the harmony
    boundary, not another reconciliation rule.
    """

    def __init__(self, fam) -> None:
        self._markers = tuple(m for m in (fam.close_markers or ()) if m)
        self._in_think = bool(fam.think_prefill) or bool(
            (fam.template_kwargs or {}).get("enable_thinking"))
        self._opener_settled = self._in_think
        self._buf = ""
        self.content_emitted = ""

    @property
    def in_think(self) -> bool:
        """True while the stream is still inside the reasoning channel.

        A stopped/partial generation is classified from this: a cut mid-thought leaves no
        close marker in the raw text, so the end-of-generation `<think>…</think>` split
        would mistake the whole thought for an answer."""
        return self._in_think

    def _split_off(self, keep_tail: bool) -> str:
        """Take everything from the buffer except the held-back tail."""
        if keep_tail:
            if len(self._buf) <= _STREAM_HOLDBACK:
                return ""
            out, self._buf = self._buf[:-_STREAM_HOLDBACK], self._buf[-_STREAM_HOLDBACK:]
        else:
            out, self._buf = self._buf, ""
        return out

    def _drain(self, keep_tail: bool) -> tuple[str, str]:
        reasoning = content = ""
        if self._in_think:
            hit, marker = None, ""
            for m in self._markers:
                i = self._buf.find(m)
                if i != -1 and (hit is None or i < hit):
                    hit, marker = i, m
            if hit is None:
                reasoning = self._split_off(keep_tail)
            else:
                reasoning = self._buf[:hit]
                self._buf = self._buf[hit + len(marker):]
                self._in_think = False
                content = _STRUCTURAL_RE.sub("", self._split_off(keep_tail)).lstrip()
        else:
            content = _STRUCTURAL_RE.sub("", self._split_off(keep_tail))
        self.content_emitted += content
        return reasoning, content

    def feed(self, chunk: str) -> tuple[str, str]:
        """Absorb a raw delta; return the (reasoning, answer) text it contributes."""
        self._buf += chunk
        if not self._opener_settled:
            i = self._buf.find("<think>")
            if i != -1:
                self._buf = self._buf[i + len("<think>"):]
                self._in_think = True
                self._opener_settled = True
            elif len(self._buf) > 64:
                self._opener_settled = True   # no opener coming; it is all answer
            else:
                return "", ""                 # wait for the head to settle
        return self._drain(keep_tail=True)

    @staticmethod
    def reconcile(streamed: str, answer: str) -> str:
        """The final answer delta: what the CLEANED answer holds beyond what was streamed.

        The stream is classified optimistically off the raw text, so this is where the
        authoritative version gets the last word — it covers the deliberately held-back
        tail, plus a family whose close marker never appeared (nothing was classified as
        answer, so the whole thing goes out here). When the two disagree outright — the
        cleaner rewrote or truncated text already on the wire — it sends nothing: a
        slightly stale tail is a better failure than a duplicated paragraph."""
        streamed = streamed.strip()
        if not streamed:
            return answer
        if answer.startswith(streamed):
            return answer[len(streamed):]
        return ""

    def flush(self) -> str:
        """Release the held-back tail at end of generation — the REASONING half only.

        The answer's tail is deliberately not returned: the caller reconciles against the
        cleaned text, which is these same bytes minus whatever the cleaner strips (a
        trailing structural token, a leaked role header), so taking it from there is
        strictly better than emitting the raw tail here. `content_emitted` is therefore
        left untouched by the drain, or reconciliation would believe the tail had already
        gone out and send nothing."""
        before = self.content_emitted
        reasoning, _ = self._drain(keep_tail=False)
        self.content_emitted = before
        return reasoning


def _make_gossip_generate(*, digest_intro: bool = True, log_transcripts: bool = True) -> Callable:
    """Build the synchronous "generate a peer reply" callable the mgmt sidecar calls.

    ``digest_intro`` injects this box's current persona digest so it answers *as its
    current self* regardless of the peer's opener. ``log_transcripts`` writes the serving
    side's half of the gossip to ``hot/chats`` so this box can also reflect on it
    (GOSSIP.md §7 approach 2); off ⇒ stateless. See :func:`_make_openai_generate` for the
    returned callable's contract."""
    return _make_openai_generate(
        mode="gossip", digest_intro=digest_intro, log_transcripts=log_transcripts,
    )


def _make_api_generate(*, client_system: str = "append", inject_rag: bool = True,
                       inject_persona: bool = True) -> Callable:
    """Build the synchronous generate callable behind the PUBLIC OpenAI API (core.api_http).

    The same recipe as live chat — her system prompt, the standing persona portrait, the
    temporal/identity anchors, her RAG memory — but stateless: ``log_transcripts`` is
    forced off, so an external tool's traffic writes nothing to ``hot/chats`` and therefore
    never reaches reflection, the training corpus, or the chat RAG index. Reads of her
    memory still happen (that is the point of querying *Ava* rather than the base model);
    only the writes are suppressed.

    Differs from the gossip mode in three ways, all because the caller is a tool rather
    than a peer instance: no "you are talking to another instance like you" framing, the
    CHAT portrait instead of the meeting-a-stranger introduction, and the client's own
    ``system`` message is honoured (``client_system="append"``) instead of dropped.

    ``client_system``: ``"append"`` wraps the caller's system message in the framing from
    ``prompts/api_client_system_prompt.txt`` and appends it last (nearest the turns, the
    most salient position — an agentic client's whole operating brief lives there);
    ``"drop"`` ignores it, gossip-style, for a box that should only ever answer as herself.
    See :func:`_make_openai_generate` for the returned callable's contract."""
    return _make_openai_generate(
        mode="api", client_system=client_system, inject_rag=inject_rag,
        inject_persona=inject_persona, log_transcripts=False,
    )


def _make_openai_generate(*, mode: str = "gossip", digest_intro: bool = True,
                          log_transcripts: bool = False, client_system: str = "drop",
                          inject_rag: bool = True,
                          inject_persona: bool = True) -> Callable:
    """The shared body behind both OpenAI-compatible serving endpoints (gossip + public API).

    Returns ``generate(messages, *, temperature, top_p, max_tokens, peer_name,
    on_delta=None) -> (answer, finish_reason, reasoning, usage)``, where ``peer_name`` is
    the caller's name (the request's OpenAI ``user`` field) and ``usage`` is the OpenAI
    token-count block. Safe to call from an HTTP thread: it submits the actual GPU work to
    the single ``_executor`` and blocks on the future, so the model is only ever touched
    from the one GPU worker (same discipline as the WebSocket handlers' ``run_in_executor``).

    ``on_delta(kind, text)`` — ``kind`` in ``{"reasoning", "content"}`` — makes the call
    incremental for the SSE path. It fires on the GPU executor thread, so the caller is
    responsible for handing the text to whatever thread is writing the response. The
    return value is unchanged and still authoritative: the deltas are classified from the
    RAW stream, so the caller receives one final ``content`` delta carrying whatever the
    cleaned answer holds beyond what was streamed (see :class:`_CotStreamSplitter`)."""

    def openai_generate(messages, *, temperature, top_p, max_tokens, peer_name,
                        on_delta=None):
        def _work():
            nonlocal messages
            # Gossip: the caller is a peer Ava, and its turns must reach us as speech
            # only. Done FIRST so the RAG query, the conversation and the logged
            # transcript below all read the same sanitized turns.
            if mode == "gossip":
                messages = _strip_peer_reasoning_from_messages(messages)
            rag = _get_rag()
            # The latest user (caller) turn drives RAG — matches chat / _generate_ava.
            last_user = next((m.get("content") for m in reversed(messages or [])
                              if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
            base_system = _session.system_prompt or ""
            system_content = base_system
            portrait = ""
            if mode == "gossip":
                system_content += "\n\n" + _gossip_framing_block(peer_name)
                if digest_intro:
                    # Chat parity for the empty case: the 'not yet decided' framing left
                    # chat_prompt.txt with the persona slot, so a thin corpus that used to
                    # get it for free now needs it composed here.
                    system_content += "\n\n" + (_current_digest_intro()
                                                or _load_persona_undecided())
            elif inject_persona:
                # Live-chat parity: persona reaches an API caller as the ONE standing
                # portrait, not as whichever [persona] paraphrases embed closest — and an
                # empty portrait (thin corpus / nothing past the maturity gate) falls back
                # to the per-turn RAG channel below, exactly as chat does.
                portrait = _current_chat_portrait()
            system_content += "\n\n" + _temporal_anchor()
            id_line = _identity_line(peer_name)
            if id_line:
                system_content += "\n\n" + id_line
            # Live-chat parity: one persona slot carrying the portrait, or the 'not yet
            # decided' framing when there is none (mode == "gossip" composed its own above).
            if mode != "gossip" and inject_persona:
                system_content += "\n\n" + (portrait or _load_persona_undecided())
            # Live-chat parity, which now means the wander channel is OFF here too
            # (_INJECT_WANDER) — it was enabled on chat/encounter/gossip alike, so it
            # goes from all three together. Asks likewise (_INJECT_OPEN_ASKS).
            rag_context = rag.query(
                last_user,
                include_wander=_INJECT_WANDER,
                include_persona=(mode == "gossip") or (inject_persona and not portrait),
                include_asks=_INJECT_OPEN_ASKS,
            ) if (last_user and (mode == "gossip" or inject_rag)) else ""
            if rag_context:
                system_content += "\n\n" + rag_context
            # The calling tool's brief goes LAST — after her identity and her memory,
            # directly before the turns it applies to.
            if client_system == "append":
                client_block = _api_client_system_block(messages)
                if client_block:
                    system_content += "\n\n" + client_block

            conv = _openai_messages_to_conversation(messages, peer_name)
            inf_conv = _build_inference_conversation(system_content, conv)

            splitter = None
            on_chunk = None
            if on_delta is not None:
                splitter = _CotStreamSplitter(
                    model_family.family_for(_runtime.model_id or ""))

                def on_chunk(text, _s=splitter):
                    reasoning, content = _s.feed(text)
                    if reasoning:
                        on_delta("reasoning", reasoning)
                    if content:
                        on_delta("content", content)

            full, input_tokens = _sync_chat_generate(
                inf_conv, temperature=temperature, top_p=top_p,
                max_new_tokens_setting=str(max_tokens),
                on_chunk=on_chunk,
            )
            cot, answer = ChatLogger._parse_cot(full)

            if splitter is not None:
                reasoning = splitter.flush()
                if reasoning:
                    on_delta("reasoning", reasoning)
                tail = splitter.reconcile(splitter.content_emitted, answer)
                if tail:
                    on_delta("content", tail)
            truncated = bool(getattr(_backend, "last_generation_truncated", False))

            # Serving-side reflection (GOSSIP.md §7 approach 2): log our half of the
            # gossip WITH our own CoT (`full`), so this box's Sleep pass can reflect on
            # it. Done here, on the executor thread, while we still hold the reasoning
            # (the returned/networked answer is CoT-stripped). Best-effort.
            if log_transcripts and mode == "gossip":
                _gossip_session_log.log_turn(
                    messages, peer_name, full,
                    base_system=base_system, model_id=(_runtime.model_id or ""),
                    adapter_id=_runtime.adapter_id, system_content=system_content,
                    rag_context=rag_context, input_tokens=input_tokens,
                    generation_params={
                        "temperature": temperature, "top_p": top_p,
                        "max_new_tokens_setting": str(max_tokens),
                    },
                )

            # Return the reasoning on a SEPARATE channel (never folded into `answer`):
            # the peer's driver surfaces it for display but must keep it out of the
            # logged message content, or the peer's CoT would become a user_prompt it
            # reflects/trains on (GOSSIP.md — the transcript stays clean-answer only).
            completion_tokens = 0
            try:
                if _runtime.tokenizer is not None:
                    completion_tokens = _backend.count_tokens(_runtime.tokenizer, full)
            except Exception:
                pass   # usage is informational — never fail a reply over it
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens,
            }
            return answer, ("length" if truncated else "stop"), cot, usage

        # A background per-chat reflection is the one GPU owner an interactive request
        # PREEMPTS rather than queues behind (see _preempt_background_reflection): without
        # this an external call could sit for minutes behind a reflection that a UI chat
        # turn would have cancelled in about a second. Done on the HTTP thread, BEFORE the
        # submit — `_work` runs on the very executor thread the reflection occupies.
        _preempt_background_reflection_sync()
        # Block the HTTP thread on the single GPU worker (queues behind any active
        # reflection/encounter — the endpoint returns 503 up front to avoid a long hang).
        return _executor.submit(_work).result()

    return openai_generate


def _prompt_debug_segments(system_parts: list, conversation: list) -> list:
    """The full model-facing prompt for the Debug view, split into labelled segments.

    Replaces a flat text dump of the same prompt. The point of the split is that the
    client colours each segment by ``kind`` — the system framing, the retrieved memory
    block, and the live user turn are the three things an operator is trying to tell
    apart when a turn goes wrong, and in a single-colour dump the RAG block is a
    paragraph indistinguishable from the prompt above it.

    *system_parts* is the ``(kind, label, text)`` list the system message was BUILT
    from, not a re-parse of the finished string: the caller joins those same parts to
    produce it, so the view cannot drift from the prompt as new injected blocks are
    added. Empty parts are dropped, exactly as the join drops them.

    *conversation* is the assembled model-facing conversation; its system turn is
    already covered by *system_parts* and is skipped, and the remaining turns are
    reported in the order the model sees them (the live user turn last).
    """
    segments = [
        {"kind": kind, "label": label, "text": text}
        for kind, label, text in system_parts
        if (text or "").strip()
    ]
    turn_no = 0
    for turn in conversation:
        if turn.get("role") == "system":
            continue
        turn_no += 1
        is_user = turn.get("role") == "user"
        segments.append({
            "kind": "user" if is_user else "assistant",
            "label": f"{'USER' if is_user else 'ASSISTANT'} · turn {turn_no}",
            "text": str(turn.get("content", "")),
        })
    return segments

def _resolve_max_new_tokens(setting: str, available: int) -> int:
    setting = str(setting).strip()
    if available <= 1:
        return 1
    try:
        if setting.endswith("%"):
            pct = float(setting[:-1].strip())
            if pct <= 0:
                raise ValueError
            return max(1, min(available, int(available * pct / 100.0)))
        return max(1, min(available, int(setting)))
    except (TypeError, ValueError):
        return max(1, min(available, int(available * 0.75)))

async def _preempt_background_reflection() -> None:
    """If a background per-chat reflection holds the GPU, ask it to yield and wait briefly
    for the executor to free before the caller runs its own generation.

    Background reflection is preemptible by design (unlike an operator Sleep run): the
    request sets its stop flag and fires the shared cancel event, so the in-flight
    reflection generation aborts within roughly one step. We poll its occupancy flag with a
    short ceiling; if it somehow does not clear, the caller proceeds anyway and simply
    queues behind it on the single executor thread."""
    if not background_reflection.is_active():
        return
    background_reflection.request_preempt()
    for _ in range(100):   # ~10s ceiling; a cancel normally frees within a step
        if not background_reflection.is_active():
            break
        await asyncio.sleep(0.1)


def _preempt_background_reflection_sync(timeout_s: float = 10.0) -> None:
    """Blocking twin of :func:`_preempt_background_reflection`, for the HTTP threads.

    The OpenAI-compatible endpoints (gossip + public API) are served off plain HTTP
    threads, not the event loop, so they cannot await the async version — and without it
    an interactive request queues behind a background per-chat reflection for as long as
    that chat takes, while an equivalent UI turn preempts it in about a second. Must be
    called BEFORE submitting to ``_executor``: the reflection holds that single worker."""
    if not background_reflection.is_active():
        return
    background_reflection.request_preempt()
    deadline = time.time() + timeout_s
    while background_reflection.is_active() and time.time() < deadline:
        time.sleep(0.1)


async def _run_generation(
    ws,
    msg_queue: asyncio.Queue,
    conversation: list,
    max_new_tokens_setting: str,
    context_length: int,
    temperature: float,
    top_p: float,
    debug: bool,
    *,
    apply_early_stop: bool = True,
    clean_fn=None,
    capture_tension: bool = False,
    exchange_id: Optional[str] = None,
) -> tuple[bool, str, Optional[dict], int]:
    """Run inference for *conversation*. Returns (success, cleaned_response, tension, input_tokens).

    apply_early_stop: set False for reflection/revision so turn-marker patterns
        in the content don't prematurely cancel generation.
    clean_fn: response cleaner; defaults to _clean_response for chat output.
    capture_tension: when True, summarize per-token signals into a tension block;
        the block is attached to the `done` payload and returned to the caller so
        chat-side rendering and chat logging both see the same numbers.
    """
    # Background per-chat reflection is the ONE GPU owner a user chat PREEMPTS rather than
    # is refused by: the user came back, so hand the GPU to them (a half-reflected chat is
    # discarded and re-reflected later). Operator-launched Sleep runs / encounters below
    # still refuse — they are deliberate and must not die to a stray chat turn.
    await _preempt_background_reflection()
    if reflection_service._reflection_run_active:
        await _send(ws, {
            "type": "error",
            "message": "A reflection run is in progress — wait for it to finish or stop it first.",
        })
        return False, "", None, 0
    if encounter_run._encounter_active:
        await _send(ws, {
            "type": "error",
            "message": "An encounter is in progress — wait for it to finish or stop it first.",
        })
        return False, "", None, 0
    if _regen_swap_active:
        # A Training-review regeneration is swapping in an ALTERNATE adapter. Refuse
        # rather than queue: this handler captures the model reference on the event
        # loop before its executor job runs, so proceeding could pin the released
        # model in VRAM through the swap load. (Once the swap-regen finishes, the
        # alternate adapter simply IS the loaded model — sticky by design — and chat
        # runs under it; status reports which adapter is live.)
        await _send(ws, {
            "type": "error",
            "message": "A Training-review regeneration is running on an alternate "
                       "adapter — wait for it to finish.",
        })
        return False, "", None, 0

    model = _runtime.model
    tokenizer = _runtime.tokenizer
    if model is None or tokenizer is None:
        await _send(ws, {"type": "error", "message": "No model loaded."})
        return False, "", None, 0

    try:
        model_id = _runtime.model_id or ""
        fam = model_family.family_for(model_id)
        prompt = build_inference_prompt(
            tokenizer, conversation,
            **fam.template_kwargs,
        )
        # Mechanism-1 cure for multi-turn CoT collapse: prefill the family's think
        # opener so the reasoning channel opens every turn. History feeds the model
        # CoT-stripped assistant turns (see handle_generate: only the parsed answer is
        # re-appended), which drives the sampled opener probability toward zero by turn
        # 2+ and makes the model answer without thinking. Prefilling removes the opener
        # from sampling. No-op for families whose template already prefills it (qwen3)
        # or that don't think (think_prefill == "").
        prompt = prompt + fam.think_prefill
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Prompt build failed: {e}"})
        return False, "", None, 0

    try:
        input_length = _backend.count_tokens(tokenizer, prompt)
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"Token count failed: {e}"})
        return False, "", None, 0
    # Guard the obvious failure before it becomes an opaque CUDA/prefill error:
    # an input at or past the context window leaves no room to generate. This
    # happens on reflection over a very large session, where the whole transcript
    # is one prompt — fail legibly so the client shows why, not a blank "error".
    if input_length >= context_length:
        await _send(ws, {"type": "error", "message": (
            f"Input is {input_length} tokens but the context window is "
            f"{context_length}. Nothing left to generate — this prompt is too "
            f"large (likely reflection over an oversized session)."
        )})
        return False, "", None, 0
    available = max(1, context_length - input_length)
    max_new_tokens = _resolve_max_new_tokens(max_new_tokens_setting, available)

    chunk_queue: queue.Queue = queue.Queue()
    _cancel_event.clear()

    def _run_generate() -> None:
        try:
            gen = _backend.stream_generate(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                context_length=context_length,
                temperature=temperature,
                top_p=top_p,
                debug=(lambda m: chunk_queue.put(("log", m))) if debug else None,
                capture_tension=capture_tension,
                # Pair with the think_prefill applied above: guarantee a non-empty CoT
                # by masking the channel-close for the family's minimum thought length.
                min_think_tokens=fam.min_think_tokens,
                # Loop defense for chat/ephemeral: halt a verbatim runaway (sampling
                # untouched) and mildly penalize repetition so the attractor is less
                # likely to form (see _CHAT_REPETITION_PENALTY). The min_p sampling
                # floor (Layer 1) removes the tail that seeds a collapse; the degen
                # guard (Layer 2, via _DEGEN_KW) halts the *drifting* runaway the
                # verbatim stop_on_repeat can't see.
                stop_on_repeat=True,
                repetition_penalty=_CHAT_REPETITION_PENALTY,
                min_p=_CHAT_MIN_P,
                # Anti-copy guard: penalize + n-gram-ban verbatim regurgitation of
                # the previous reply (the t≈1.0 induction attractor the other
                # guards can't see). Only that turn is protected — RAG/system stay
                # exempt per the generated-only penalty's rationale.
                no_copy_text=_last_assistant_content(conversation),
                **_DEGEN_KW,
            )
            try:
                for chunk in gen:
                    if _cancel_event.is_set():
                        break
                    chunk_queue.put(("chunk", chunk))
            finally:
                gen.close()
        except Exception as e:
            # Log the full traceback to server.log — without it a generation
            # failure (e.g. CUDA OOM) leaves nothing to diagnose. Send the client
            # the exception type too, since str(e) is empty for some torch/CUDA
            # errors and would otherwise render as a blank "error".
            traceback.print_exc()
            msg = str(e).strip() or "no message"
            chunk_queue.put(("error", f"{type(e).__name__}: {msg}"))
        finally:
            chunk_queue.put(("done", None))

    loop = asyncio.get_running_loop()
    gen_future = loop.run_in_executor(_executor, _run_generate)

    collected: list[str] = []
    raw_tail = ""
    early_stop = False
    user_cancelled = False
    success = False
    cleaned_response = ""
    tension_block: Optional[dict] = None

    try:
        while True:
            try:
                inner = msg_queue.get_nowait()
                if inner.get("type") == "cancel":
                    user_cancelled = user_cancelled or bool(inner.get("discard", False))
                    _cancel_event.set()
            except asyncio.QueueEmpty:
                pass

            try:
                kind, text = chunk_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue

            if kind == "chunk":
                if user_cancelled:
                    # The producer may have queued a few deltas before observing
                    # the event. Drain those locally instead of leaking more of a
                    # reply the operator has already discarded.
                    continue
                collected.append(text)
                raw_tail = (raw_tail + text)[-3000:]
                if apply_early_stop and not early_stop and _should_early_stop_stream(raw_tail):
                    early_stop = True
                    _cancel_event.set()
                # Live generation speed (rolling 5 s window) so the client status line
                # can show tok/s while the reply is still streaming.
                await _send(ws, {
                    "type": "chunk", "text": text,
                    "tokens_per_sec": _backend.current_tokens_per_sec(),
                })
            elif kind == "log":
                if debug:
                    await _send(ws, {"type": "log", "message": text})
            elif kind == "error":
                if user_cancelled:
                    # A backend may report its own interruption while unwinding;
                    # the operator-facing terminal state is still a clean abort.
                    continue
                await _send(ws, {"type": "error", "message": text})
                break
            elif kind == "done":
                if user_cancelled:
                    # An operator stop is an aborted turn, not a short successful
                    # answer: handle_generate will roll back its optimistic user
                    # append, and no transcript/RAG/training evidence is written.
                    await _send(ws, {"type": "cancelled"})
                    break
                _clean = clean_fn if clean_fn is not None else _clean_response
                cleaned_response = _clean("".join(collected))
                success = True
                tension_spans = None
                think_open_prob = None
                if capture_tension:
                    tension_block = _compute_tension_block(
                        tokenizer, model_id, _backend.last_token_signals
                    )
                    tension_spans = _compute_tension_spans(
                        tokenizer, _backend.last_token_signals
                    )
                    # First-step probability the model put on opening a thinking block
                    # (gemma-4 `<|channel>`) — the "Thinking: NN%" CoT diagnostic.
                    think_open_prob = _backend.last_think_open_prob
                await _send(ws, {
                    "type": "done",
                    "response": cleaned_response,
                    "exchange_id": exchange_id,
                    "input_tokens": input_length,
                    "memory": _backend.memory_status(),
                    "tension": tension_block,
                    "tension_spans": tension_spans,
                    "think_open_prob": think_open_prob,
                    # Finalized whole-generation speed for the status-line readout.
                    "tokens_per_sec": _backend.last_tokens_per_sec,
                })
                break
    finally:
        _cancel_event.set()
        await gen_future
        # Release this turn's KV-cache / scratch blocks back to the driver so the
        # reserved pool doesn't grow unbounded across turns and reflection passes.
        await loop.run_in_executor(_executor, _backend.trim_memory)

    return success, cleaned_response, tension_block, input_length

async def handle_generate(ws, msg: dict, msg_queue: asyncio.Queue) -> None:
    """Session-managed generate: client sends one user message; server owns conversation/RAG/logging."""
    user_message = msg.get("message", "")
    speaker = str(msg.get("user", "")).strip()
    max_new_tokens_setting = str(msg.get("max_new_tokens_setting", "75%"))
    context_length = _runtime.context_length
    temperature = float(msg.get("temperature", 1.0))
    top_p = float(msg.get("top_p", 0.95))
    debug = bool(msg.get("debug", False))
    exchange_id = uuid4().hex

    _mark_activity()   # a live user turn resets the idle-wander clock
    _session.user = speaker
    _session.conversation.append(
        {"role": "user", "content": user_message, "speaker": speaker}
    )
    logger = _ensure_logger()
    rag = _get_rag()
    # Live chat is the one path that surfaces the wander channel (an article Ava read on her
    # own bleeds in). Reflection/revision keep it off to stay replay-faithful.
    # top_k=2 (below the default 3) for live chat: a single strong past-chat hit shouldn't be
    # reinforced by two weaker ones into repeating a prior answer over the current query.
    # Per-channel RAG gates from the Chat tab (History / Facts / Persona checkboxes, all on
    # by default) so the operator can A/B a channel's effect — all off = adapter only. History
    # covers both past-chat recall and the wander channel (experiential recall).
    # History now covers past-chat recall only: the wander channel it also used to gate is
    # off for every caller (_INJECT_WANDER).
    include_history = bool(msg.get("rag_history", True))
    include_facts = bool(msg.get("rag_facts", True))
    include_persona = bool(msg.get("rag_persona", True))
    # Persona reaches live chat as ONE standing portrait (the digest) rather than as
    # whichever one-to-three [persona] paraphrases embedded closest to this message.
    # The Chat tab's Persona checkbox gates the whole channel, portrait included, so
    # unchecking it still means "no persona this turn" for an A/B. When there is no
    # portrait to inject — no digest yet, or nothing past the maturity gate — the RAG
    # channel stays on, so a thin corpus behaves exactly as it did before.
    portrait = _current_chat_portrait() if include_persona else ""
    # One slot, one block. The 'not yet decided' framing used to sit in chat_prompt.txt
    # unconditionally, so a mature digest put it beside the portrait every turn — the
    # prompt telling her identity is an open question while the block under it told her
    # what she had settled into. It is the persona channel's empty state, so it lives
    # here now and yields to the portrait the moment there is one.
    persona_block = portrait or (_load_persona_undecided() if include_persona else "")
    # The person on the other side gets the same treatment her own persona does: one
    # settled reading injected every turn, instead of whichever [impression] paraphrases
    # embedded closest to this message. It rides the Facts checkbox rather than Persona —
    # this is knowledge ABOUT someone, not Ava's self — so unchecking Facts still means
    # "no user knowledge this turn" for an A/B. As with the self-portrait, an empty
    # portrait (nobody nameable, no portrait yet, switch off) leaves the per-turn RAG
    # channel on, so a newly-met person behaves exactly as before.
    user_portrait = _current_user_portrait(speaker) if include_facts else ""
    # On a session Ava opened herself (initiated_by:"ava" + an initiated_ask stamp),
    # the question's ORIGIN — the conversation or reading it was distilled from —
    # is injected standing, so the user's answer is read against what made her ask
    # rather than against the opener alone (see core.ask_origin). Rides the History
    # checkbox: the payload is past-material recall arriving by a non-cosine route,
    # the same rule the fact-nomination slot follows (`nominate_sessions` rides
    # `include_chat`), so "no past-chat recall this turn" keeps meaning that.
    ask_origin_block = _current_ask_origin(logger) if include_history else ""

    # Stage 1 of a two-stage turn (FACTS_TREE.md §10 consumer 5): before the reply is
    # written, a short pass picks which recorded facts this message turns on, and the
    # chosen claims are rendered from the tree in code. OFF unless `graph.enabled` — it
    # costs a whole extra generation on time-to-first-token, and the design ships it gated.
    # Rides the Facts checkbox like the user portrait does: this is knowledge about the
    # world, not Ava's self, so unchecking Facts still means "no fact knowledge this turn".
    #
    # It runs BEFORE the RAG query, not because retrieval needs it to have finished, but
    # because its picks are an INPUT to retrieval: every fact came out of a conversation
    # and the tree knows which, so the picks nominate those conversations to the past-chat
    # channel (`rag_engine._query_nominated`). That is a way into an old chat that no
    # cosine offers — the conversation may share no wording, or no language, with the
    # message that just arrived, and past `rag_cap_age_h` it has no verbatim vectors left
    # to be found by at all.
    facts_block = ""
    fetched: dict = {}
    if include_facts:
        loop = asyncio.get_running_loop()
        fetched = await loop.run_in_executor(
            _executor, _fetch_facts_block_sync, user_message, speaker,
            list(_session.conversation[:-1]))
        facts_block = fetched.get("text") or ""
        # Every path in there is caught, so a retrieval failure can never take a turn
        # down — which also means a permanently broken channel looks exactly like a quiet
        # one, and did. Name the failures (not the ordinary empty outcomes: picking
        # nothing is a correct answer, and `disabled` would print every turn). The `[tag]`
        # print convention is teed into the activity journal, so this reaches the Activity
        # tab without a protocol message of its own.
        if fetched.get("skipped") in _FACTS_FETCH_FAILURES:
            print(f"[facts] fetch skipped: {fetched['skipped']}"
                  + (f" — {fetched['error']}" if fetched.get("error") else ""))

    rag_context = rag.query(
        user_message, top_k=2,
        include_chat=include_history,
        include_wander=_INJECT_WANDER and include_history,
        include_facts=include_facts,
        include_persona=include_persona and not portrait,
        include_asks=_INJECT_OPEN_ASKS and include_facts,
        include_impressions=include_facts,
        # Only THIS speaker's readings are redundant with the block above; readings of a
        # third party stay in and compete normally, so she is not silenced about someone
        # else at the moment the conversation turns to them.
        impressions_exclude_about=(speaker if user_portrait else ""),
        # Best-first, capped and fenced inside the engine (`graph.nominate_max`). Passed
        # even when the chat block is off: `query` drops them with it, so the History
        # checkbox keeps meaning "no past-chat recall this turn" whatever the facts
        # channel found.
        nominate_sessions=(fetched.get("sources") or None),
    )

    # Assembled from a labelled parts list rather than by successive concatenation, so
    # the Debug view (_prompt_debug_segments) is built from the same objects the prompt
    # is and cannot drift from it — a block added here shows up there for free. Order is
    # the assembly order and is load-bearing: standing context (who she is) before
    # situational context (what is relevant now), the same position the identity/temporal
    # anchors occupy. Who *they* are follows immediately: also standing, and read in that
    # order it completes the framing the identity line opens ("X is speaking with you
    # right now") before any of this particular moment's material arrives. Empty parts
    # drop out, which is what the old `if identity:` / `if portrait:` guards did.
    system_parts: list = [
        ("system", "SYSTEM PROMPT", _session.system_prompt),
        ("system", "IDENTITY", _identity_line(speaker)),
        ("system", "TEMPORAL ANCHOR", _temporal_anchor()),
        ("system", "PERSONA PORTRAIT (standing)" if portrait
                   else "PERSONA (undecided)", persona_block),
        ("system", "USER PORTRAIT (standing)", user_portrait),
        # Standing like the portraits above it (constant across the session's turns,
        # not query-dependent), and last of the standing blocks: it frames why THIS
        # conversation exists, the narrowest of the three framings.
        ("system", "ASK ORIGIN (standing)", ask_origin_block),
        # Before the RAG block: both are situational, but this one was selected FOR this
        # message by a pass that read it, where the RAG block is whatever embedded nearest.
        ("rag", "FACTS (fetched)", facts_block),
        ("rag", "INJECTED RAG", rag_context),
    ]
    system_content = "\n\n".join(t for _, _, t in system_parts if (t or "").strip())

    # Proactive surfacing (Non-reactivity): at the very start of a session Ava may
    # raise an open question carried over from earlier reflections. Only on the
    # first turn — so she opens with it rather than re-raising it every message.
    # The block shapes the prompt now, but the surface-count markers are written
    # only after the turn succeeds (handled below) — a failed generation raised
    # nothing and must not advance the retirement ceiling.
    # OFF pending redesign (_INJECT_OPEN_ASKS): with `surfaced` left empty, no
    # `write_surface` is stamped either, so open asks stop silently retiring.
    surfaced: list[dict] = []
    if _INJECT_OPEN_ASKS and len(_session.conversation) == 1:
        candidates = _select_surfaced_questions()
        block = _surface_block(candidates)
        if block:
            system_content = system_content + "\n\n" + block
            system_parts.append(("system", "SURFACED QUESTIONS", block))
            surfaced = candidates

    inference_conversation = _build_inference_conversation(
        system_content, _session.conversation
    )

    # Debug checkbox: the full prompt the model is about to condition on, split into
    # labelled segments the client colours (system framing / injected RAG / the live
    # user turn). Sent before the reply streams, so it renders above it.
    if debug:
        await _send(ws, {
            "type": "prompt_debug",
            "segments": _prompt_debug_segments(system_parts, inference_conversation),
        })

    # Stage 1's result, on its own, always — not behind the Debug checkbox. This block is
    # what the reply is ABOUT to be built on, and it is the one injected channel whose
    # content was CHOSEN for this message rather than embedded near it, so an operator
    # reading a reply needs it in front of them, not on request. It is sent AFTER the
    # prompt dump (so under Debug the log reads whole prompt → the fetch → the reply) and
    # before the first chunk, so it lands above the CoT either way.
    #
    # Sent whenever the fetch RAN, empty result included: "she looked and picked nothing"
    # and "the channel is off" are different facts about a reply, and only the second is
    # worth staying silent about. `claims_text` rather than `text` — the injected wrapper
    # is the same paragraph every turn.
    if fetched and fetched.get("skipped") != "disabled":
        cands = fetched.get("candidates") or {}
        await _send(ws, {
            "type": "facts_block",
            "text": fetched.get("claims_text") or "",
            "skipped": fetched.get("skipped") or "",
            "error": fetched.get("error") or "",
            "picked": fetched.get("picked") or [],
            "n_rendered": int(fetched.get("n_rendered") or 0),
            # How many claims the pass was offered to choose from — the denominator that
            # makes "picked nothing" readable. Never the candidates themselves: that is the
            # ~10k-token list, and this is a per-turn message.
            "n_candidates": int(cands.get("n") or 0),
            # The conversations these facts came out of, best-first — what the picks
            # NOMINATED to the past-chat channel. Reported rather than left to be inferred
            # from the RAG block, because the engine's caps and fences mean the list here
            # is what was offered, not necessarily what was recalled: the operator needs to
            # see that a fact had a conversation behind it even on a turn where the slot
            # went unspent. Names only — the recap itself is in the RAG block already.
            "sessions": list(fetched.get("sessions") or []),
        })

    success, response, tension, input_tokens = await _run_generation(
        ws, msg_queue, inference_conversation,
        max_new_tokens_setting, context_length, temperature, top_p, debug,
        capture_tension=True,
        exchange_id=exchange_id,
    )

    if success:
        _, context_content = ChatLogger._parse_cot(response)
        generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens_setting": max_new_tokens_setting,
        }
        logger.log_exchange(
            user_message, response, rag_context=rag_context, tension=tension,
            speaker=speaker, generation_params=generation_params,
            system_content=system_content, input_tokens=input_tokens,
            think_open_prob=_backend.last_think_open_prob,
            exchange_id=exchange_id,
        )
        # Running user-token tally (token-economy metric): count only this live
        # turn's user message; historical chats are never backfilled.
        _add_user_tokens(user_message, _runtime.tokenizer)
        # Push the freshly-incremented token economy so the status-bar meter updates
        # this turn (the `done` above was sent before the increment). Best-effort.
        if _read_token_economy is not None:
            try:
                econ = _read_token_economy()
                await _send(ws, {
                    "type": "token_stats",
                    "user_tokens": econ.get("accumulated", 0),
                    "token_economy": econ,
                })
            except Exception:
                pass
        _session.conversation.append({"role": "assistant", "content": context_content})
        # Index of the exchange just logged, taken from the logger itself — not from
        # the live conversation, which may carry restored turns from a continued
        # session and so would not line up with the (fresh) session file.
        exchange_index = logger.exchange_count - 1
        source_session = logger.current_file.name if logger.current_file else ""
        rag.add_exchange(
            user_message, context_content, speaker=speaker,
            source_session=source_session, exchange_index=exchange_index,
        )
        # Record that these open questions were raised — only now that the turn
        # landed — so the surface-count ceiling counts real raises, not failed ones.
        if surfaced:
            surfaced_in = logger.current_file.name if logger.current_file else ""
            writer = _get_reflection_writer()
            for q in surfaced:
                writer.write_surface(key=q["key"], surfaced_in=surfaced_in)
            _session.surfaced_keys = [q["key"] for q in surfaced]
    else:
        _session.conversation.pop()

async def handle_generate_ephemeral(ws, msg: dict, msg_queue: asyncio.Queue) -> None:
    """Stateless generate with a caller-supplied conversation list (used by the Sleep tab)."""
    conversation = msg.get("conversation", [])
    max_new_tokens_setting = str(msg.get("max_new_tokens_setting", "75%"))
    context_length = _runtime.context_length
    temperature = float(msg.get("temperature", 1.0))
    top_p = float(msg.get("top_p", 0.95))
    debug = bool(msg.get("debug", False))

    await _run_generation(
        ws, msg_queue, conversation,
        max_new_tokens_setting, context_length, temperature, top_p, debug,
    )

def _run_branch_exchange_sync(
    filename: str, exchange_index: int, temperature: float, top_p: float,
    on_candidate: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Synchronous branch candidate generation — safe to call from executor thread.

    Shared by the WebSocket handler (handle_branch_exchange) and the reflection
    runner (_run_branch_for_exchange) so the generation logic lives in one place
    (core.branch_replay). Returns a dict with keys: eligible, filename,
    exchange_index, candidates, dropped, n_generated, original (when eligible),
    reason (when not eligible). Notably does NOT include 'type' — callers add that.
    *on_candidate*, when given, streams each fork's text as it's generated (see
    branch_replay.run_branch_exchange).
    """
    model = _runtime.model
    tokenizer = _runtime.tokenizer
    if model is None or tokenizer is None:
        return {"eligible": False, "reason": "No model loaded",
                "filename": filename, "exchange_index": exchange_index,
                "candidates": [], "dropped": []}
    return branch_replay.run_branch_exchange(
        filename, exchange_index, temperature, top_p,
        backend=_backend, model=model, tokenizer=tokenizer,
        chats_dir=_CHATS_DIR, context_length=_runtime.context_length,
        model_id=_runtime.model_id or "",
        default_system_prompt=_session.system_prompt,
        clean_response_fn=_clean_response,
        embedder=_get_rag()._get_embedder(),
        on_candidate=on_candidate,
    )

def _sync_branch_chooser_content(payload: dict, system_prompt: str, context_length: int) -> str:
    """Build the budgeted branch-chooser prompt — server-side adapter over branch_replay."""
    tokenizer = _runtime.tokenizer
    if tokenizer is None:
        raise RuntimeError("No model loaded")
    return branch_replay.build_budgeted_branch_select_content(
        _backend, tokenizer, system_prompt, payload, context_length,
        _runtime.model_id or "",
    )

async def handle_branch_exchange(ws, msg: dict) -> None:
    """Regenerate the replies Ava *almost gave* for one logged exchange.

    Delegates to _run_branch_exchange_sync via run_in_executor so GPU work
    stays off the event loop. Read-only; persistence is the runner's job.
    """
    filename = msg.get("filename", "")
    exchange_index = int(msg.get("exchange_index", -1))
    temperature = float(msg.get("temperature", 1.0))
    top_p = float(msg.get("top_p", 0.95))
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_branch_exchange_sync,
            filename, exchange_index, temperature, top_p,
        )
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "error", "message": f"Branch generation failed: {e}"})
        return
    await _send(ws, {"type": "branches", **result})


# ── manual exchange re-answer (Training review "Regenerate") ────────────────── #

# Bounded generation budget for a manual re-answer. A percentage would pre-allocate a
# huge static KV cache; observed dialogue replies are far smaller, so a fixed cap fits
# with margin and keeps VRAM small (mirrors the reflection IDEAL cap).
_REGEN_MAX_NEW_TOKENS = "8192"

# The LoRA adapter lineage (server/models) — self-located from __file__ like
# mgmt_http/training_review, since only the regenerate flow's alternate-adapter swap
# reads it and threading one more path through configure() buys nothing.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# True while a Training-review regeneration is swapping in (or generating right after
# swapping in) an ALTERNATE adapter (see handle_regenerate_exchange). The hazard is
# the release→load window — the same one the reflection run's CleanBaseSession has,
# where the runtime transiently holds no model — but regeneration has no
# `_reflection_run_active` refusing chat around it, so this flag closes that gap:
# `_run_generation` refuses while it is set (a chat handler captures `_runtime.model`
# on the EVENT LOOP before queueing its executor job, so without the refusal it could
# pin the released model in VRAM through the swap load), and server.py folds it into
# the gossip/API `is_busy` predicates. For simplicity it spans the whole swap-regen
# request, not just the load. NB the swap is STICKY (`agentic.swap_model`): once this
# clears, the alternate adapter simply IS the loaded model — chat and the autonomous
# jobs run under it, deliberately (status reports it; a Chat-tab load undoes it).
# Flipped on the event loop only.
_regen_swap_active = False


def _resolve_regen_adapter(adapter: str):
    """Resolve a Training-review adapter request to ``(swap_target, error)``.

    *adapter* is a bare directory NAME under ``server/models`` (the client picks from
    the server-reported lineage in the review payload and sends the name back — never
    a path, which would not survive the tab running from a remote UI box). Returns
    ``(None, None)`` when no swap is needed — an empty request, or the named adapter
    IS the loaded one — ``(path, None)`` for a valid different adapter, and
    ``(None, message)`` for a request that must be refused (path escape, or no such
    adapter on this box)."""
    adapter = (adapter or "").strip()
    if not adapter:
        return None, None
    models_dir = _MODELS_DIR.resolve()
    if Path(adapter).name != adapter or adapter in (".", ".."):
        return None, f"Invalid adapter name: {adapter!r}"
    target = (models_dir / adapter).resolve()
    if target.parent != models_dir:
        return None, f"Invalid adapter name: {adapter!r}"
    if not (target / "adapter_config.json").is_file():
        return None, f"No adapter at models/{adapter} on this box."
    current = str(_runtime.adapter_id or "").strip()
    if current:
        try:
            if Path(current).resolve() == target:
                return None, None      # already the loaded adapter — nothing to swap
        except OSError:
            pass
    return str(target), None

_THINK_SPLIT_RE = re.compile(r"(?is)\s*<think>(.*?)</think>(.*)")


def _split_regenerated(raw: str) -> tuple[str, str]:
    """Split a regenerated dialogue completion into ``(cot, answer)``.

    Lenient: an answer-only completion (no faithful CoT) yields ``("", answer)`` so the
    operator can still review it in the pop-up and decide."""
    m = _THINK_SPLIT_RE.match(raw or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", (raw or "").strip()


def _regenerate_exchange_sync(
    filename: str, exchange_index: int, temperature: float, top_p: float,
    system_suffix: str = "", cot_only: bool = False,
    on_delta: Optional[Callable[[str, str], None]] = None,
    swap_adapter: Optional[str] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict:
    """Re-answer one logged exchange with the CURRENTLY LOADED adapter (executor thread).

    Rebuilds the exact pre-answer conversation (stored system prompt + answer-only prior
    turns + the user turn) and generates a fresh ``<think>…</think>`` reply through the same
    clean dialogue seam the reflection IDEAL pass uses — so the CoT is faithful to the reply
    it is generated with. Returns the new CoT + reply plus the originals for side-by-side
    review; writes nothing (the apply step persists to the sidecar).

    ``system_suffix`` is an optional operator-supplied delivery constraint from the Training
    review tab (e.g. "don't append a question merely to continue the conversation"). It shapes
    only THIS generation — it is appended to the system prompt via ``build_ideal_messages`` but
    is NOT part of the trainable prefix persisted on the anchor, so the row trains clean while
    the regenerated answer follows the steer. Whatever the operator types is their call.

    ``cot_only`` (the Training-review *Corrupt CoT* box checked WITHOUT *Corrupt reply*) means
    the fresh answer is thrown away — Apply grafts only the new ``<think>`` onto the trusted
    original reply — so generation halts once the reasoning channel closes instead of spending
    the token budget on a reply nothing reads. The thought opener is prefilled regardless
    (``force_think``) so the re-answer always carries a CoT, the exact failure this addresses.

    ``on_delta(cot_delta, reply_delta)`` (optional) streams the generation as it is produced,
    already classified into the two boxes the review dialog shows — the raw stream is
    family-specific (gemma-4's channel markers, qwen3's prefilled opener), so the split runs
    here through the same :class:`_CotStreamSplitter` the public API streams with rather than
    leaking family knowledge to the client. The returned CoT/reply remain authoritative (they
    come from the cleaned text); the stream is a live preview the caller replaces at the end.

    A generation the operator STOPS (``_cancel_event``) returns what was produced so far with
    ``cancelled: True``. A cut mid-thought carries no close marker, so the ``<think>`` split
    would read the whole thought as an answer — the splitter's state is what disambiguates.

    ``swap_adapter`` (a resolved ``models/<name>`` path from :func:`_resolve_regen_adapter`,
    or ``None``) has a DIFFERENT adapter from the lineage answer instead of the loaded one —
    the "regenerate with the last known good adapter" path. The swap is
    ``agentic.swap_model``: ONE-WAY, release → load → **stays loaded** — a repair session
    is a run of regenerations under one chosen adapter, so the next regeneration with the
    same choice resolves to no swap at all (one reload per adapter change, not two per
    row), at the accepted cost that the box keeps running that adapter afterwards (chat,
    idle jobs, everything — honestly reported by ``status``; a Chat-tab load or a restart
    undoes it, the config's ``adapter_id`` untouched). A failed swap load still restores
    the previous model (the shared failure discipline; the same known limit remains — the
    load takes the backend's default precision). ``on_status(text)`` narrates the reload
    so the review dialog isn't silent for the minutes it takes."""
    from core.reflection_source import build_revision_jobs, build_ideal_messages

    # Presence check WITHOUT binding locals. This frame lives across the adapter swap
    # below, and a `model = _runtime.model` here is exactly the trap CleanBaseSession's
    # own comments warn about: release()'s `del` reaches only its own locals, so a
    # reference held in THIS frame keeps the released model resident in VRAM through
    # the whole swap load — which is how the alternate-adapter load OOMed (and, since
    # the frame was still pinning it, how the failure path's restore load OOMed too,
    # leaving the box with no model). Everything downstream (generate_fn, the family
    # lookup) reads the runtime at call time and needs no reference taken here.
    if _runtime.model is None or _runtime.tokenizer is None:
        return {"ok": False, "error": "No model loaded."}
    try:
        path = (Path(_CHATS_DIR) / filename).resolve()
        if path.parent != Path(_CHATS_DIR).resolve() or not is_chat_session_json(path):
            return {"ok": False, "error": "Invalid session filename."}
        if not path.exists():
            return {"ok": False, "error": f"Session not found: {filename}"}
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"Could not read session: {e}"}

    jobs = build_revision_jobs(session)
    job = next((j for j in jobs if j.get("index") == exchange_index), None)
    if job is None:
        return {"ok": False,
                "error": "Exchange is not revisable (needs a user turn + a non-empty reply)."}

    messages = build_ideal_messages(job, session, system_suffix=system_suffix)
    generate_fn = _make_sync_reflect_generate(_get_rag())

    splitter = _CotStreamSplitter(model_family.family_for(_runtime.model_id or ""))
    on_chunk = None
    if on_delta is not None:
        def on_chunk(delta: str, _s=splitter) -> None:
            cot, reply = _s.feed(delta)
            if cot or reply:
                on_delta(cot, reply)

    def _status(text: str) -> None:
        if on_status is not None:
            try:
                on_status(text)
            except Exception:
                pass

    def _generate() -> str:
        return generate_fn(
            "", "", messages_override=messages, disable_rag=True,
            temperature=temperature, top_p=top_p,
            max_new_tokens_setting=_REGEN_MAX_NEW_TOKENS,
            force_think=True, stop_after_think=cot_only,
            on_chunk=on_chunk,
        )

    adapter_name = Path(swap_adapter).name if swap_adapter else ""
    try:
        if swap_adapter:
            # Alternate-adapter re-answer: release the loaded model and load the same
            # base with the chosen adapter — ONE-WAY (`agentic.swap_model`), so it
            # stays loaded for the rest of the repair session and the next regeneration
            # with the same choice pays no reload at all. A failed load restores the
            # previous model before the error propagates (shared failure discipline).
            from core import agentic

            def _prep(tok, mid) -> None:
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                if not getattr(tok, "chat_template", None):
                    ensure_chat_template(tok, model_name=mid, emit=lambda m: None)

            _status(f"Loading adapter {adapter_name} (full model reload — it will "
                    "stay loaded)…")
            agentic.swap_model(_backend, _runtime, adapter_id=swap_adapter,
                               prepare=_prep,
                               on_log=lambda m: print(m, flush=True))
            _status(f"Adapter {adapter_name} loaded — generating…")
        raw = _generate()
    except Exception as e:
        msg = f"Generation failed: {e}"
        if swap_adapter and _runtime.model is None:
            # The swap failed AND the fallback restore inside swap_model failed too —
            # the box has no model. The tab must say that outright: the next thing the
            # operator does is reload from the Chat tab, not retry here.
            msg += (" — the adapter swap failed and restoring the previous model also "
                    "failed, so NO model is loaded now. Reload from the Chat tab.")
        return {"ok": False, "error": msg}

    cancelled = bool(_cancel_event is not None and _cancel_event.is_set())
    new_cot, new_reply = _split_regenerated(raw)
    if splitter.in_think and not new_cot:
        # Stopped (or truncated) mid-thought: no close marker reached the raw text, so the
        # split above read the whole thought as an answer. It is all CoT, and there is no
        # reply — `raw` (cleaned) is used rather than the streamed accumulation so the
        # splitter's deliberately held-back tail is not lost.
        new_cot, new_reply = raw.strip(), ""
    exchanges = session.get("exchanges") or []
    ex = exchanges[exchange_index] if 0 <= exchange_index < len(exchanges) else {}
    return {
        "ok": True,
        "filename": filename,
        "exchange_index": exchange_index,
        "cancelled": cancelled,
        # Which adapter authored the re-answer — read from the RUNTIME, not from the
        # request: under sticky swap semantics a follow-up regeneration with the same
        # dropdown choice resolves to "no swap needed" (the win), so the request alone
        # can no longer name the author. "" = bare base.
        "adapter": (Path(_runtime.adapter_id).name if _runtime.adapter_id else ""),
        "new_cot": new_cot,
        "new_reply": new_reply,
        "original_cot": (ex.get("assistant_cot") or ""),
        "original_reply": (ex.get("assistant_response") or ""),
    }


async def handle_regenerate_exchange(ws, msg: dict, msg_queue: asyncio.Queue) -> None:
    """Re-answer one logged exchange with the loaded adapter (Training review Regenerate).

    Read-only preview: returns the fresh CoT + reply for operator review; a separate
    ``apply_regenerated_exchange`` writes the sidecar. Refuses while a reflection run or
    encounter owns the GPU (the loaded adapter may be swapped out mid-reflection).

    **Streams.** The review dialog opens the moment the operator presses Regenerate, so the
    generation is pushed out as ``regenerate_chunk {cot, reply}`` deltas (already classified
    — see :func:`_regenerate_exchange_sync`) and terminated by the authoritative
    ``exchange_regenerated``. A minute of silence otherwise looks like a hung request, and
    the operator can only judge a re-answer they can watch.

    **Stoppable.** Like the chat path, it drains ``msg_queue`` while the executor works so a
    ``cancel`` sent mid-generation sets the shared ``_cancel_event`` and halts it within a
    step; the partial comes back with ``cancelled: true`` and is still applicable (the
    operator may want exactly the CoT that was produced before the runaway). Every other
    message read off the queue is put back — the client serializes its RPCs behind one lock,
    so anything else here arrived out of band and is not ours to drop.

    Resets the shared idle clock (``_mark_activity``) on both ends of the call: a
    dataset-cleaning session is a live operator at the box, but it drives no chat turn, so
    without this the idle clock keeps running and the autonomous jobs (wander / outreach /
    background reflection) wake up mid-cleanup and fight for the single executor thread.
    Marked BEFORE the generation too — the regeneration itself can take minutes, and it
    does not hold the scheduler's GPU lock, so a job coming due during it would dispatch
    straight into the queue behind us."""
    _mark_activity()   # operator is at the box cleaning the dataset — not idle
    await _preempt_background_reflection()
    if reflection_service._reflection_run_active:
        await _send(ws, {"type": "error",
                         "message": "A reflection run is in progress — wait for it to finish or stop it first."})
        return
    if encounter_run._encounter_active:
        await _send(ws, {"type": "error",
                         "message": "An encounter is in progress — wait for it to finish or stop it first."})
        return
    filename = str(msg.get("filename", ""))
    try:
        exchange_index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "Invalid exchange index."})
        return
    temperature = float(msg.get("temperature", 0.9))
    top_p = float(msg.get("top_p", 0.95))
    system_suffix = str(msg.get("system_suffix", "")).strip()
    cot_only = bool(msg.get("cot_only", False))
    # Optional alternate adapter from the lineage (Training review's Adapter dropdown):
    # resolved HERE, on the event loop, so `_regen_swap_active` is already set when any
    # other handler could next run — the flag is what keeps a concurrent chat turn from
    # capturing the model reference this swap is about to release (see its definition).
    swap_adapter, adapter_err = _resolve_regen_adapter(str(msg.get("adapter", "")))
    if adapter_err:
        await _send(ws, {"type": "error", "message": adapter_err})
        return
    global _regen_swap_active
    if swap_adapter and _regen_swap_active:
        await _send(ws, {"type": "error",
                         "message": "An adapter-swap regeneration is already in progress."})
        return
    loop = asyncio.get_running_loop()

    def _on_delta(cot: str, reply: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "regenerate_chunk", "cot": cot, "reply": reply}), loop)

    def _on_status(text: str) -> None:
        # Swap milestones (adapter loading/restoring) — a model reload is minutes of
        # otherwise-total silence in a dialog that opened at the button press.
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "regenerate_status", "text": text}), loop)

    deferred: list[dict] = []
    stop_requested = False
    if swap_adapter:
        _regen_swap_active = True
    try:
        future = loop.run_in_executor(
            _executor, _regenerate_exchange_sync,
            filename, exchange_index, temperature, top_p, system_suffix, cot_only,
            _on_delta, swap_adapter, _on_status,
        )
        while not future.done():
            if stop_requested:
                # Re-assert each pass: generation clears the event when it starts, so a
                # stop arriving before the executor picked the job up would be swallowed.
                _cancel_event.set()
            try:
                inner = msg_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                continue
            if inner.get("type") == "cancel":
                stop_requested = True
                _cancel_event.set()
            else:
                deferred.append(inner)
        result = await future
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "error", "message": f"Regeneration failed: {e}"})
        return
    finally:
        _regen_swap_active = False
        for inner in deferred:
            msg_queue.put_nowait(inner)
        # Re-mark on the way out: the idle window should start when the operator's work
        # finished, not when it started (a long regeneration would otherwise burn most of it).
        _mark_activity()
    if not result.get("ok"):
        await _send(ws, {"type": "error", "message": result.get("error", "Regeneration failed.")})
        return
    await _send(ws, {"type": "exchange_regenerated", **result})


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test:  python -m core.generation
# ──────────────────────────────────────────────────────────────────────────────
# Covers _CotStreamSplitter only — the pure, load-bearing piece of the streaming API
# path. Everything else in this module needs a model.

def _selftest() -> None:
    fams = model_family

    def run(fam, chunks, answer=None):
        """Drive the splitter exactly as the streaming caller does, reconciliation
        included — `answer` is the cleaned text generation ultimately returned."""
        s = _CotStreamSplitter(fam)
        reasoning = content = ""
        for ch in chunks:
            r, c = s.feed(ch)
            reasoning += r
            content += c
        reasoning += s.flush()
        if answer is not None:
            content += s.reconcile(s.content_emitted, answer)
        return reasoning, content, s.content_emitted

    failures = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
        if not cond:
            failures.append(name)

    print("generation._CotStreamSplitter self-test")

    # gemma-4: the prompt prefills the channel opener, so generation starts INSIDE the
    # thought and closes with <channel|>.
    r, c, _ = run(fams.GEMMA4, ["The user asks ", "in Russian. ", "I should answer",
                                "<channel|>", "Привет! ", "Как дела сегодня, друг?"],
                  answer="Привет! Как дела сегодня, друг?")
    check("gemma-4 reasoning is the thought only",
          r == "The user asks in Russian. I should answer", repr(r))
    check("gemma-4 answer excludes the close marker",
          c == "Привет! Как дела сегодня, друг?", repr(c))

    # A marker straddling two chunks must still be recognised, and one that only looks
    # like a marker must not be.
    r, c, _ = run(fams.GEMMA4, ["think", "<chan", "nel|>", "answer that is long enough"],
                  answer="answer that is long enough")
    check("close marker split across chunks is found", c.strip() == "answer that is long enough",
          repr(c))
    r, c, _ = run(fams.GEMMA4, ["think", "<chan", "el|>", "not a marker so this is thought"])
    check("near-miss marker does not split", c == "" and "chanel" in r, repr(c))

    # qwen3 closes with a literal </think>; the default profile samples the opener too.
    r, c, _ = run(fams.QWEN3, ["ponder", "ing", "</think>", "\n\nAnswer, at some length."],
                  answer="Answer, at some length.")
    check("qwen3 splits on </think>", c.strip() == "Answer, at some length.", repr(c))
    r, c, _ = run(fams.DEFAULT, ["<think>", "hmm", "</think>", "Reply, at some length."],
                    answer="Reply, at some length.")
    check("default profile finds a sampled <think> opener",
          r == "hmm" and c.strip() == "Reply, at some length.", f"{r!r} {c!r}")

    # Structural tokens must never reach a client.
    _, c, _ = run(fams.GEMMA4, ["t", "<channel|>", "Hi there, this is the reply<end_of_turn>"],
                  answer="Hi there, this is the reply")
    check("structural tokens are scrubbed from the answer",
          c.strip() == "Hi there, this is the reply", repr(c))

    # flush() must NOT count the held-back tail as streamed, or the caller's
    # reconciliation believes it already went out and the reply is truncated.
    s = _CotStreamSplitter(fams.GEMMA4)
    s.feed("t")
    s.feed("<channel|>")
    s.feed("short reply")
    before = s.content_emitted
    s.flush()
    check("flush leaves content_emitted alone (reconciliation sends the tail)",
          s.content_emitted == before, f"{before!r} -> {s.content_emitted!r}")

    # gpt-oss: no prefilled opener and no close marker in the raw stream (harmony role
    # names, normalized only at the end) — the documented limitation. Recorded here so a
    # future fix has a failing expectation to flip.
    _, c, _ = run(fams.GPTOSS,
                  ["analysis", " The user is asking about X, so I should weigh the two "
                   "options and then commit to one.", "assistantfinal",
                   " The actual answer, stated plainly."],
                  answer="The actual answer, stated plainly.")
    check("gpt-oss streams the harmony text uncorrected (known limitation)",
          "analysis" in c and "assistantfinal" in c, repr(c))

    # ── the facts channel (FACTS_TREE.md §10 consumer 5) ────────────────────── #
    print("\ngeneration: the facts channel is gated and fails soft")
    import inspect as _inspect

    def gcheck(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    # Asserted on the DEFAULT rather than on this box's config, so the test says the same
    # thing on every box. On since 2026-08-12 by decision; the switch still exists and
    # `false` restores the original behaviour.
    src = _inspect.getsource(_facts_channel_config)
    gcheck("the switch defaults on", 'get("enabled", True)' in src, True)
    gcheck("...and is read per turn, not cached",
           "load_server_config" in src, True)
    # An unreadable config is a box in an unknown state, and the unknown-state answer for a
    # channel costing a generation per turn is not to run it — that stays False on purpose.
    gcheck("...but an unreadable config still yields off",
           'return {"enabled": False}' in src, True)
    # The regression that made "on by default" a lie: `training` is a sibling package and
    # is not importable from a bare `inference/` cwd, so without this the config read threw
    # and returned the DISABLED branch — the default decided by import order, silently.
    gcheck("...and the sibling package is reached, not assumed",
           "_sys.path.insert(0, str(_server_dir))" in src, True)
    gcheck("the default is live, not shadowed by a failed import",
           _facts_channel_config()["enabled"], True)

    # Every failure path must yield an empty block rather than raise: a retrieval channel
    # is never a reason for a chat turn to fail.
    real_cfg = globals()["_facts_channel_config"]
    try:
        globals()["_facts_channel_config"] = lambda: {"enabled": False}
        gcheck("disabled ⇒ no block, and no work attempted",
               _fetch_facts_block_sync("m", "u", [])["skipped"], "disabled")
        globals()["_facts_channel_config"] = lambda: {
            "enabled": True, "max_claims": 6, "til_max_age_days": None,
            "max_new_tokens": 512}
        # No model loaded in a GPU-free test, which is itself one of the guarded paths.
        out = _fetch_facts_block_sync("m", "u", [])
        gcheck("enabled but unable ⇒ empty block, named reason",
               (out["text"], out["skipped"] in ("no_model", "no_tree", "no_prompt",
                                                "error", "no_candidates")), ("", True))

        # …and the check that matters, because the one above passes on a box with no
        # model no matter what the pass would have done with one. The regression it
        # missed: `fetch_blob` calls `generate(body, prompt, max_new_tokens=N)` while the
        # reflect seam takes a different keyword set, so every live turn raised TypeError
        # into `fetch_blob`'s catch-all and skipped `generate_failed` — the channel dead
        # on the live path while the workbench module, calling the seam directly, worked.
        # So: let the call actually HAPPEN, against a stub carrying the real seam's
        # signature, and over a synthetic tree so the result is the same on a box that has
        # never run `graph.build`.
        import graph.store as _gstore
        real_read, real_make = _gstore.read_tree, globals()["_make_sync_reflect_generate"]
        real_get_rag, real_model = globals()["_get_rag"], _runtime.model
        seam_sig = _inspect.signature(real_make(None))
        seen: dict = {}

        def _seam_shaped(_rag):
            def _seam(content, system_prompt, **kw):
                seam_sig.bind(content, system_prompt, **kw)   # the live seam's contract
                seen.update(kw)
                return "FACTS:\n1"
            return _seam
        try:
            _gstore.read_tree = lambda: {
                "nodes": {"person:a": {"id": "person:a", "label": "a"}},
                "claims": {"c1": {"claim_id": "c1", "node": "person:a",
                                  "facet": "property", "text": "a fact", "n_sources": 1,
                                  "n_occurrences": 1, "lanes": ["chat"],
                                  "last_asserted": "2026-08-11", "mentions": [],
                                  "when": ""}},
                "occurrences": []}
            globals()["_make_sync_reflect_generate"] = _seam_shaped
            globals()["_get_rag"] = lambda: None
            _runtime.model = real_model or object()
            out = _fetch_facts_block_sync("m", "u", [])
            gcheck("the pass reaches the model instead of dying on its own call shape",
                   out.get("skipped"), "")
            gcheck("...and the picked claim comes back wrapped",
                   "a fact" in out["text"] and "{facts}" not in out["text"], True)
            # The run flags come off the ModuleSpec, so this path and the workbench
            # simulation of it cannot condition the same pass differently.
            from core.modules import MODULES as _MODULES
            spec = _MODULES["fact_fetch"]
            gcheck("selection is greedy", (seen.get("temperature"), seen.get("top_p")),
                   (0.0, 1.0))
            gcheck("nothing is injected but the list and the turns",
                   seen.get("disable_rag"), True)
            gcheck("thinking is off, off the spec",
                   seen.get("disable_thinking"), spec.disable_thinking)
            gcheck("the verbatim loop guard is off, off the spec",
                   seen.get("stop_on_repeat"), spec.stop_on_repeat)
        finally:
            _gstore.read_tree = real_read
            globals()["_make_sync_reflect_generate"] = real_make
            globals()["_get_rag"] = real_get_rag
            _runtime.model = real_model

        # The OOM contract: retried once against the just-emptied pool (the failed
        # attempt is what cleaned it — the backend's OOM path dumps the cache before
        # re-raising), a second OOM lands as its own named `oom` reason (the operator's
        # next move is the allocator, not the prompt), and a NON-OOM failure is never
        # retried — a deterministic error retried on the GPU would double a live turn's
        # time-to-first-token for nothing.
        _synth_tree = lambda: {
            "nodes": {"person:a": {"id": "person:a", "label": "a"}},
            "claims": {"c1": {"claim_id": "c1", "node": "person:a",
                              "facet": "property", "text": "a fact", "n_sources": 1,
                              "n_occurrences": 1, "lanes": ["chat"],
                              "last_asserted": "2026-08-11", "mentions": [],
                              "when": ""}},
            "occurrences": []}
        calls = {"n": 0}
        _OOM_TEXT = "CUDA out of memory. Tried to allocate 498.00 MiB"

        def _oom_then_ok(_rag):
            def _seam(content, system_prompt, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError(_OOM_TEXT)
                return "FACTS:\n1"
            return _seam

        def _oom_always(_rag):
            def _seam(content, system_prompt, **kw):
                calls["n"] += 1
                raise RuntimeError(_OOM_TEXT)
            return _seam

        def _other_always(_rag):
            def _seam(content, system_prompt, **kw):
                calls["n"] += 1
                raise RuntimeError("gpu on fire")
            return _seam
        try:
            _gstore.read_tree = _synth_tree
            globals()["_get_rag"] = lambda: None
            _runtime.model = real_model or object()

            globals()["_make_sync_reflect_generate"] = _oom_then_ok
            out = _fetch_facts_block_sync("m", "u", [])
            gcheck("an OOM'd fetch is retried once, and the retry can succeed",
                   (out.get("skipped"), calls["n"]), ("", 2))

            calls["n"] = 0
            globals()["_make_sync_reflect_generate"] = _oom_always
            out = _fetch_facts_block_sync("m", "u", [])
            gcheck("a second OOM is a named skip, not a generic generate_failed",
                   (out.get("skipped"), calls["n"]), ("oom", 2))
            gcheck("...classified as broken, so the turn prints/renders it red",
                   "oom" in _FACTS_FETCH_FAILURES, True)

            calls["n"] = 0
            globals()["_make_sync_reflect_generate"] = _other_always
            out = _fetch_facts_block_sync("m", "u", [])
            gcheck("a non-OOM failure is never retried",
                   (out.get("skipped"), calls["n"]), ("generate_failed", 1))
        finally:
            _gstore.read_tree = real_read
            globals()["_make_sync_reflect_generate"] = real_make
            globals()["_get_rag"] = real_get_rag
            _runtime.model = real_model
    finally:
        globals()["_facts_channel_config"] = real_cfg

    # The block is wrapped, and the wrapper leans on the facet guarantee.
    gcheck("the block template has a slot", "{facts}" in _FACTS_BLOCK_DEFAULT, True)
    gcheck("...and says these are not opinions",
           "not of what anyone thinks" in _FACTS_BLOCK_DEFAULT, True)

    # The reflect seam carries the same channel for the reach-out passes (outreach's
    # decision pass, synthesis's analysis pass): a fetched block arrives as a labelled
    # part under chat's own label, sits ahead of the injected RAG, and the sources behind
    # it reach the engine's nomination slot. The IDEAL replay path must refuse it for the
    # reason it refuses hidden RAG — context absent from the persisted training anchor.
    rs = _inspect.getsource(_reflect_system_parts)
    gcheck("the reflect seam renders a fetched block under chat's label",
           '("rag", "FACTS (fetched)", facts_block)' in rs, True)
    gcheck("...ahead of the RAG block, contract still last",
           rs.index('"FACTS (fetched)"') < rs.index('"INJECTED RAG"')
           < rs.index('"PASS PROMPT"'), True)
    ms = _inspect.getsource(_make_sync_reflect_generate)
    gcheck("...and hands a caller's fetch sources to the nomination slot",
           "nominate_sessions=rag_nominate_sessions" in ms, True)
    gcheck("...and the IDEAL replay path refuses a hidden facts block",
           "rag_context or facts_block" in ms, True)
    gcheck("the broken-fetch classification is the shared one",
           _FACTS_FETCH_FAILURES is _fact_fetch_mod.FETCH_FAILURES, True)

    # It must ride the Facts checkbox and sit ahead of the RAG block, both of which are
    # assertions about the call site rather than about this function.
    hg = _inspect.getsource(handle_generate)
    gcheck("the fetch is gated on the Facts checkbox",
           "if include_facts:\n        loop = asyncio.get_running_loop()" in hg, True)
    gcheck("the prior turns exclude the just-appended user message",
           "_session.conversation[:-1]" in hg, True)
    gcheck("the block is a labelled part, so Debug shows it for free",
           '("rag", "FACTS (fetched)", facts_block)' in hg, True)
    gcheck("...and precedes the RAG block",
           hg.index('"FACTS (fetched)"') < hg.index('"INJECTED RAG"'), True)

    # Fact-nominated recall: the picks are an INPUT to retrieval, not just a block beside
    # it, so the fetch has to have finished before the query runs. Both halves asserted —
    # the order alone would silently become decorative if the argument went missing, and
    # the argument alone would nominate an empty list if the order slipped back.
    gcheck("the fetch runs before the RAG query, so its picks can nominate",
           hg.index("_fetch_facts_block_sync, user_message") < hg.index("rag_context = rag.query("),
           True)
    gcheck("...and the sources behind them are handed to the chat channel",
           'nominate_sessions=(fetched.get("sources") or None)' in hg, True)
    gcheck("...and reported, so an unspent slot is still visible",
           '"sessions": list(fetched.get("sessions") or [])' in hg, True)

    # The per-turn view of what stage 1 chose. Three properties, all of them call-site
    # facts: it is NOT behind the Debug checkbox (it is what the reply rests on, not a
    # diagnostic); it goes out before generation starts, so it lands above the CoT rather
    # than after the conclusion it was a premise for; and it carries the unwrapped claims,
    # the wrapper being the same paragraph every turn.
    gcheck("the fetched block is reported on its own, not behind Debug",
           '"type": "facts_block"' in hg and 'if debug:' not in
           hg[hg.index('"type": "facts_block"') - 400:hg.index('"type": "facts_block"')],
           True)
    gcheck("...before the reply is generated, so it lands above the CoT",
           hg.index('"type": "facts_block"') < hg.index("await _run_generation("), True)
    gcheck("...carrying the claims, not the constant wrapper",
           '"text": fetched.get("claims_text")' in hg, True)
    # A disabled channel says nothing; a channel that RAN and found nothing says so, since
    # a reply built on no facts is a different reply from one built on three.
    gcheck("a disabled channel is silent, an empty result is not",
           'fetched.get("skipped") != "disabled"' in hg, True)
    gcheck("the ~10k-token candidate list never rides a per-turn message",
           '"candidates":' not in hg and '"n_candidates"' in hg, True)

    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _selftest()
