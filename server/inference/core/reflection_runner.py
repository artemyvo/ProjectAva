"""Server-owned synchronous reflection runner — consolidation, judgement, re-answer,
and branch phases.

Executes the workflow entirely server-side: loading sessions, fetching open
questions, building context-aware chunks (consolidation), per-exchange judgement,
clean dialogue re-answer, and branch generation + blind selection. Branching is a
core part of revision and
always runs when the branch-generation capability is wired in — both the WebSocket
server and the headless CLI provide it (via the shared `core.branch_replay`).

The runner is deliberately *synchronous* and designed to be dispatched via
asyncio's run_in_executor — the same pattern the server uses for all GPU work.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from core.reflection_chunking import (
    build_consolidation_chunks, format_chunk_content, append_closing, SUMMARY_CLOSING,
)
from core.reflection_config import ReflectionRunConfig, ReflectionRunStore
from core.reflection_lang import detect_language_drift
from core.reflection_memory import ReflectionMemory
from core.reflection_source import (
    build_ideal_messages, build_revision_jobs, conversation_user_texts)
from core.reflection_stats import RunStats
from core.reflection_writer import (
    ReflectionWriter,
    ideal_trainable_target,
    parse_consolidation,
    parse_revision_judgement,
    register_consolidation_anchors,
    register_revision_anchor,
    resolved_target_provenance,
    resolve_revision_target,
    revision_counter,
    revision_lang_drift,
    write_revision_sidecar,
)

# Reflection samples close to chat warmth — just under chat's recommended 1.0 — so
# verdicts, re-answers, and consolidation distillation aren't collapsed onto a single
# greedy mode (an over-cold judge gives rigid, repetitive output). top_p follows the
# Gemma 4 recommendation (0.95) and the backend applies the family's recommended
# top_k (64), so reflection samples the full recommended distribution shape.
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOP_P = 0.95
# Consolidation and revision generation budget. A percentage was a trap: "75%" of the
# remaining window is ~48k tokens, and the backend pre-allocates a static KV cache for
# the full budget (~7 GB on the 31B model) regardless of how much is actually generated.
# That fixed overhead sits on every revision pass and OOMs the card on deep exchanges.
# Observed reflection output is far smaller — revision ~600 tokens, consolidation ~2.2k —
# so a bounded fixed cap fits it with wide margin and keeps the KV cache small.
_DEFAULT_MAX_NEW_TOKENS = "8192"
# Branch chooser + clean-base judge both run thinking OFF (disable_thinking=True), emitting
# only a short CHOICE/WHY — observed max ~70 tokens of WHY, chooser_cot empty on 80/86
# exchanges. So the shared 8K default (_DEFAULT_MAX_NEW_TOKENS) is ample margin; both use it.
# History: a wider 12288 cap once existed because the chooser used to *think* before
# answering and truncated mid-deliberation on ~63% of exchanges (never reaching CHOICE/WHY).
# Turning thinking off made that rambling — and the extra margin — obsolete.
# The per-exchange anchor pass emits only an ABOUT line + a short TAGS list, thinking OFF
# (`disable_thinking=True`, like the chooser/judge above). Thinking was on at first —
# naming what a turn is *about* reads like a judgement — but the pass is an INDEXING one:
# the prompt asks for a description, not a verdict, and the deliberation had nowhere to go
# but into the token budget. Observed on the 2026-07-31 run: ~90 s per exchange and an
# empty parse, the model spending the whole 1024-token cap inside a `<think>` it never
# closed, so `_strip_think_block` left the parser nothing (the failure `_pass_output_debug`
# now reports). With thinking off the two lines are the whole generation: ABOUT is capped
# at 300 chars and TAGS at 8, so ~200 tokens covers a Russian descriptor with margin, and
# 384 leaves room for a family whose template ignores the flag (`enable_thinking` is a
# gemma-4/qwen3 kwarg — gpt-oss's `reasoning_effort` is untouched by it) to fail FAST and
# visibly rather than after a minute and a half. This pass runs once per exchange, so its
# KV-cache allocation is the one most worth keeping small.
_ANCHOR_MAX_NEW_TOKENS = "384"

# The revisit recollection pass emits one to three sentences plus a TRIGGER line, with
# thinking left ON (what she now makes of an old conversation is exactly a judgement).
# Roomier than the anchor cap because the deliberation is the substance here, but still
# far under the 8K default: it runs once per session, not per exchange.
_RECOLLECTION_MAX_NEW_TOKENS = "2048"
# Closing line of the recollection content, the sibling of _USER_NOTES_CLOSING and there
# for the same reason: the prompt asks its question before a whole transcript, and what
# sits nearest the first generated token is the conversation's last reply. Re-asking after
# it makes the task the most recent text rather than something to continue.
_RECOLLECTION_CLOSING = (
    "— end of the conversation —\n\n"
    "That was then. Re-reading it now, as you are now: what do you make of it? Answer in "
    "your own words. A sentence carried over from the transcript above is the record of "
    "what was said, which already exists — it is not what you now make of it."
)
# Tags carried onto a recollection from the exchange anchors this run wrote. Capped
# because a long chat's anchors would otherwise pile every tag in the conversation onto
# one record; the first N in exchange order are the ones the chat opened on.
_RECOLLECTION_TAG_CAP = 12

# The user-notes pass ("what did I learn about this person?") emits a short list of
# [impression] lines with thinking ON — forming a reading of someone is a judgement, and
# the prompt asks explicitly for the hedging to be worked out before it is written.
# That invitation is what sets the cap, not the length of the output: the <think> block
# and the answer share ONE budget, so the deliberation the prompt asks for is spent out
# of the same budget the impressions had to fit in, and a pass that thinks its way through
# a whole transcript reaches the cap mid-thought. `_split_think` yields an empty body for
# an unterminated block, so that lands as "this conversation revealed nothing about them"
# — the prompt's explicitly-permitted outcome — and a systematically truncating pass reads
# as a person who simply never reveals anything (the ambiguity `_pass_output_debug` now
# reports on). Raised rather than taking thinking off, because unlike the anchor pass the
# deliberation here IS the work.
#
# 2048 → 4096 → 8192, the last step for the reason that drove the portrait's (see
# `user_digest.PORTRAIT_MAX_NEW_TOKENS`): the thought is not a fixed cost. It scales with
# the input, and this pass reads a WHOLE TRANSCRIPT — so the budget is not too small in
# general, it is too small for long conversations, and it fails on exactly the chats
# richest in readings of the person. Now at the 8K default rather than under it: running
# once per session and not per exchange bounds how often it is paid, which is an argument
# about cost, not about how much room one such pass needs.
_USER_NOTES_MAX_NEW_TOKENS = "8192"
# Impressions kept from one session. A conversation yields a handful of genuine readings;
# a longer list means the model started generating generalities that would be true of
# anyone (the failure the prompt warns against), and those dilute the portrait's evidence
# ranking. Truncation is in emission order, so the ones it reached for first survive.
_USER_NOTES_CAP = 8
# Closing line of the user-notes content, placed AFTER the transcript. The prompt already
# asks the question, but the prompt is far away and a whole conversation sits between it
# and the first generated token — with the transcript last, the nearest thing to continue
# was the reply that ended it, and the pass was observed handing that reply back as the
# impression. Re-asking here costs a sentence and makes the task the most recent text.
_USER_NOTES_CLOSING = (
    "— end of the conversation —\n\n"
    "Now, reading it back: what did you come to understand about {person}? Write it in "
    "your own words. A sentence copied out of the transcript above — theirs or yours — is "
    "not a reading of them, and none of it belongs in an [impression] line."
)

# Self-notes: the OUTSIDE-view counterpart of the pass above — what the transcript shows
# about HER, read as a reader with no access to the <think> she wrote it from. Same budget
# and cap for the same reasons (it reads the same whole transcript, and a longer list means
# generalities that dilute the portrait's evidence ranking). See `core.self_portrait`.
_SELF_NOTES_MAX_NEW_TOKENS = "8192"
_SELF_NOTES_CAP = 8
# The echo risk this closing guards is sharper than the user-notes one: the text nearest
# the generation point is HER OWN last reply, which is also the most plausible thing to
# hand back when asked "what does this person seem like?".
_SELF_NOTES_CLOSING = (
    "— end of the conversation —\n\n"
    "Now, reading it back cold: what do these replies show about the one who wrote them? "
    "Write it in your own words. A sentence copied out of the transcript above — yours or "
    "theirs — is not a reading of anyone, and none of it belongs in an [impression] line."
)

# Chat facts: the per-chat extraction PROTOCOL (see `core.chat_facts`). Its budget is the
# largest of the per-session reading passes and deliberately so — the two above are capped
# because a long list means the model drifted into generalities, whereas this pass is asked
# to be exhaustive and a long list is the pass working. The cap that matters here is
# `chat_facts.MAX_FACTS`, a runaway guard rather than an editorial limit.
_CHAT_FACTS_MAX_NEW_TOKENS = "12288"
_CHAT_FACTS_CLOSING = (
    "— end of the conversation —\n\n"
    "Now write the protocol: every fact this conversation established, one per [fact] "
    "line, each marked with who it is about and whether it is standing, stated or an "
    "event. Record what was said, not what it suggests about anyone."
)

# Criterion-flip maturity gate (phase two): the clean-base judge may override a trainable
# target only when the digest has at least _FLIP_MIN_THEMES themes whose WEIGHTED
# recurrence (recency- and tenure-discounted, counter-evidence netted — from
# evidence.themes, not the model's generous "maturity" label) is >= _FLIP_MIN_WEIGHTED.
# It gated on the RAW distinct-session count (>= 3) until 2026-08-23, which the persona
# echo loop could satisfy on its own: raw recurrence climbs without bound from
# portrait-conditioned re-derivations (the live 2026-07-15 digest passed with 19 themes
# at raw >= 3, all restating ONE trait), while weighted recurrence is tenure-capped at
# 1/(1-_TENURE_DECAY) = 2.5, fades when a theme stops being reaffirmed, and drops under
# pushback/polarity counters. 1.9 preserves the old bar's intent — three RECENT distinct
# sessions (1 + 0.6 + 0.36 = 1.96) pass; three stale ones, or a contested theme, do not.
# A theme with NO weighted_recurrence (a digest written before the decay machinery)
# counts as immature: the flip stays dormant until the digest regenerates under current
# code, rather than opening on echo-inflated raw counts. Authority still accrues with
# corpus breadth (slope, not cliff). First-guess.
_FLIP_MIN_WEIGHTED = 1.9
_FLIP_MIN_THEMES = 2

# One-shot retry temperature bump. The judgement retry appends a field-format reminder;
# the clean IDEAL retry replays the identical pre-answer conversation with perturbed
# sampling, so the model never sees a "previous attempt" instruction in its dialogue CoT.
_REVISION_RETRY_TEMP_BUMP = 0.4
_REVISION_RETRY_NUDGE = (
    "\n\nIMPORTANT: Output only VERDICT, WHY, LANG_DRIFT, and PERSONA_TARGET in the "
    "requested field format. Do not write an IDEAL reply or any other fields. Write the "
    "four field names in exactly those Latin letters — do not translate them into the "
    "language of the conversation, however natural that feels. Only the field names; "
    "what you write after each colon belongs in the language you were speaking."
)

# Language-drift guard nudges (revision pass), used by _language_decision_guard. RECHECK fires when
# the script-level backstop (core.reflection_lang) finds a hard mismatch the model did NOT
# flag: it asks the model to look again, and — crucially — to KEEP a switch the user actually
# requested (a translation), so the decision stays the model's. The clean re-answer stage
# later handles the reply itself, outside this reflection prompt.
_LANG_RECHECK_NUDGE = (
    "\n\nIMPORTANT: Your reply above appears to be written in a different language than the "
    "one the user was speaking. If you switched deliberately because the user asked you to "
    "(for example a translation), that reply is fine — answer VERDICT: keep and LANG_DRIFT: no. "
    "But if the switch was NOT something the user asked for, it is not your voice — answer "
    "VERDICT: revise and LANG_DRIFT: yes. Do not write an IDEAL reply; this is judgement only."
)
_IDEAL_LANGUAGE_SUFFIX = (
    "Reply entirely in the language used by the final user message. A requested translation "
    "in that message still takes precedence."
)

# Corrupt-reply nudge (revision pass). The operator flagged this exchange's STORED reply as
# corrupt (a logging/generation bug), so the recorded answer is garbage — it must not be
# kept. This forces a revise decision; the separate clean re-answer stage reconstructs the
# target from the pre-answer conversation. The stored reply remains visible only to the judge.
_CORRUPT_REPLY_NUDGE = (
    "\n\nIMPORTANT: The recorded reply for the exchange under review is CORRUPTED (a "
    "logging/generation error) and must NOT be kept as-is. Answer VERDICT: revise. Do not "
    "write a replacement reply here; it will be reconstructed separately from the user's turn "
    "and preceding conversation."
)

# CoT-regeneration similarity floor (approach #3). When the operator flags an exchange's
# CoT corrupt but the judge KEEPS the reply (the reply is trusted — only the thought was a
# logging/generation bug), we re-answer the exchange to author a fresh faithful <think> and
# graft it onto the ORIGINAL reply — but only when the re-answer's reply is at least this
# cosine-similar to the original. Above the floor the fresh thought genuinely leads to
# ~that reply, so the graft is faithful; below it the content diverged enough that the new
# reasoning may justify a DIFFERENT answer, which is the think/answer mismatch that erodes
# the reasoning channel, so we fall back to answer-only (prior corrupt-CoT behavior).
# Tunable; sits just under branch_replay.BRANCH_ORIGINAL_CEILING (0.92, "converged with
# original") so a re-answer need not be a near-duplicate to qualify, only clearly on-topic.
_RECOT_SIMILARITY_FLOOR = 0.85

# A consolidation generation can raise a *transient* backend error — most often the
# TorchDynamo FX-trace error after a streamer timeout (see inference_backend.load:
# "Detected that you are using FX to symbolically trace a dynamo-optimized function").
# It's a per-prompt-bucket compile-path race: the first attempt recompiles and trips
# the trace, but suppress_errors then makes Dynamo fall back to eager, so an immediate
# re-run of the same chunk usually succeeds. Retry once before skipping the pass.
_CONSOLIDATION_GENERATE_RETRIES = 1


# ── standalone helpers ─────────────────────────────────────────────────── #

def _user_portraits_enabled() -> bool:
    """Box-wide kill-switch for the standing user portrait (``user_portrait.enabled``).

    Default on. Gates *production* (this run-level pass) and, separately, injection in
    ``generation._current_user_portrait`` — either half can be cut alone, so a box can go
    on folding portraits it does not yet inject, or stop folding while keeping the last
    one live. The ``[impression]`` records themselves are governed by
    ``overrides.user_notes`` (production) and ``impressions.enabled`` (retrieval), so all
    three layers are independently switchable. Best-effort: an unreadable config reads as
    enabled, matching every other switch here."""
    try:
        from training.reflections_path import load_server_config
        cfg = (load_server_config() or {}).get("user_portrait") or {}
        return bool(cfg.get("enabled", True))
    except Exception:
        return True


def _self_portrait_enabled() -> bool:
    """Box-wide kill-switch for the outside-view self-portrait (``self_portrait.enabled``).

    Default on, and it gates the run-level FOLD only. The three layers are independently
    switchable exactly as the user portrait's are: ``overrides.self_notes`` governs
    production of the ``[self_impression]`` records, this switch governs folding them into
    ``users/_self.json``, and ``self_impressions.enabled`` governs retrieval (which is
    additionally off by default — nothing asks for that channel). Unlike the user portrait
    there is no injection layer at all: see ``core.self_portrait`` on why this artifact is
    deliberately never put in a prompt. Best-effort: an unreadable config reads as enabled.
    """
    try:
        from training.reflections_path import load_server_config
        cfg = (load_server_config() or {}).get("self_portrait") or {}
        return bool(cfg.get("enabled", True))
    except Exception:
        return True


_EVENT_TEXT_CLIP = 8000  # per-field cap for branch_done display text (whole copy is on the persisted block)


def _clip(text: str, limit: int = _EVENT_TEXT_CLIP) -> str:
    """Bound display text on events; the persisted branch block keeps the full copy."""
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " …[clipped]"


_PASS_DEBUG_CLIP = 1500  # per-field cap for an empty-parse dump


def _pass_output_debug(raw: str, parsed: str,
                       truncated: Optional[bool] = None,
                       generate_fn: Optional[Callable] = None) -> str:
    """What a pass actually produced, for an empty-or-unparseable-output warning.

    The failure is otherwise indistinguishable from the outside: an empty parse looks
    the same whether the model never closed a ``<think>`` block (the whole token budget
    spent thinking — the 2026-07-31 shape that took thinking off the anchor pass), wrote
    its output without the labels the parser keys on, or returned nothing at all. Only the
    RAW generation separates those, so the warning carries it, with the post-strip body
    beside it when they differ (that is what the parser actually saw), the caller's
    one-line *parsed* summary above both, and how the generation ENDED — the one fact the
    text itself cannot show. Clipped per field: a runaway pass must not bloat the event
    store or the activity journal.

    *generate_fn* is the reflect-generate callable this pass just used; it carries the
    end-state of its last call (see `generation._make_sync_reflect_generate`). Passing it
    is what makes the ending honest rather than a guess. ``last_truncated`` says only that
    the final token was not EOS, which is equally true of three different endings — the
    token cap, the verbatim-loop halt (`stop_on_repeat`) and the degeneration halt — and
    reporting all three as "hit the token cap" sends the reader after a budget that was
    never the problem: an unterminated `<think>` at a tenth of the allowance is a halted
    generation, not an exhausted one. ``last_loop`` separates the two halts from the cap,
    and the budget line makes the cap claim checkable against the numbers.

    Shared by the passes whose empty result is ambiguous — the anchor pass (skip) and the
    user-notes pass, where an empty parse would otherwise be reported as the prompt's
    explicitly-permitted "this conversation revealed nothing about them".
    """
    raw = (raw or "")
    body = _strip_think_block(raw)
    closed = "</think>" in raw.lower()
    if truncated is None and generate_fn is not None:
        truncated = getattr(generate_fn, "last_truncated", None)
    looped = getattr(generate_fn, "last_loop", None) if generate_fn is not None else None
    if truncated and looped:
        # Both stopping criteria end on a non-EOS token, so `truncated` is True here too;
        # the halt is the more specific fact and the one that names a cause.
        cap = "halted on repetition/degeneration"
    elif truncated:
        cap = "hit the token cap"
    elif truncated is not None:
        cap = "ended on EOS"
    else:
        cap = "cap state unknown"
    lines = [
        "parsed: " + parsed,
        "raw ({} chars, {}, <think> {}):".format(
            len(raw), cap,
            "closed" if closed else "never closed" if "<think>" in raw.lower()
            else "absent"),
        _clip(raw.strip(), _PASS_DEBUG_CLIP) or "(empty)",
    ]
    budget = _pass_budget_line(generate_fn)
    if budget:
        lines.insert(1, budget)
    if body != raw.strip():
        lines.append("parser saw ({} chars after <think> strip):".format(len(body)))
        lines.append(_clip(body, _PASS_DEBUG_CLIP) or "(empty)")
    return "\n".join(lines)


def _pass_budget_line(generate_fn: Optional[Callable]) -> str:
    """The token budget the last call of *generate_fn* ran under, or "" if unknown.

    Reads what `generation._make_sync_reflect_generate` stamps on the callable after each
    generation. Written as prompt/allowance/window because that is the arithmetic a reader
    checks a "hit the token cap" claim against — an allowance far above the raw output
    means the generation ended for some other reason, and a prompt filling most of the
    window is the one case where the cap really is the story.
    """
    if generate_fn is None:
        return ""
    max_new = getattr(generate_fn, "last_max_new_tokens", None)
    if not max_new:
        return ""
    inp = getattr(generate_fn, "last_input_tokens", None)
    window = getattr(generate_fn, "last_context_length", None)
    return "budget: prompt {} tok, allowance {} tok, window {} tok".format(
        inp if inp is not None else "?", max_new,
        window if window is not None else "?")


# ── VRAM probe (diagnostic only) ───────────────────────────────────────── #
# The runner stays free of GPU *logic*; these helpers only read CUDA's global
# allocator counters so the branch report can attribute the peak to a stage.
# All no-op (return None) without CUDA, so they're safe in the GPU-free selftest.
# max_*_memory_allocated/reserved are process-global high-water marks; because
# branch work is strictly serial on this executor thread, a reset-then-read pair
# around one stage attributes that stage's peak unambiguously.

def _vram_reset_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _vram_peak_gb() -> Optional[tuple[float, float]]:
    """(peak_allocated_gb, peak_reserved_gb) since the last reset, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return (torch.cuda.max_memory_allocated() / 1024 ** 3,
                torch.cuda.max_memory_reserved() / 1024 ** 3)
    except Exception:
        return None


def _vram_device_total_gb() -> Optional[float]:
    """Total VRAM on the active CUDA device, GB — the denominator for the 'will it
    fit in 24 GB' headroom in the run report. None without CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.total_memory / 1024 ** 3
    except Exception:
        return None


def _quant_label(model_id: Optional[str]) -> Optional[str]:
    """Coarse quantization tag inferred from the model id, for the report footprint
    context (the bnb-4bit weights are the bulk of the baseline VRAM)."""
    mid = (model_id or "").lower()
    if "4bit" in mid or "bnb-4bit" in mid:
        return "bnb-4bit"
    if "8bit" in mid:
        return "8bit"
    return None


def _count_tokens(tokenizer, text: str) -> int:
    """Token length of *text* via the underlying text tokenizer (0 on failure)."""
    if not tokenizer or not text:
        return 0
    try:
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        return len(text_tok(text)["input_ids"])
    except Exception:
        return 0


def _consolidation_output_reserve(setting: str, context_length: int) -> int:
    """Conservative fixed headroom for consolidation reasoning + structured output."""
    raw = str(setting or "").strip()
    try:
        requested = (int(raw) if not raw.endswith("%")
                     else min(8192, max(512, context_length // 2)))
    except (TypeError, ValueError):
        requested = 8192
    return max(1, min(requested, max(512, context_length // 2),
                      max(1, context_length - 512)))


def _clip_lines_to_tokens(text: str, tokenizer, limit: int) -> str:
    """Keep whole lines from *text* under a small fixed-input budget."""
    if not text or _count_tokens(tokenizer, text) <= limit:
        return text
    kept: list[str] = []
    marker = "...[additional open questions omitted for token budget]"
    for line in text.splitlines():
        candidate = "\n".join(kept + [line, marker])
        if _count_tokens(tokenizer, candidate) > limit:
            break
        kept.append(line)
    return "\n".join(kept + [marker, ""])


def _clip_session_prompt_to_fit(
    session: dict,
    open_questions_block: str,
    sess_idx: int,
    total_sessions: int,
    fits: Callable[[str], bool],
) -> tuple[dict, bool]:
    """Clip only the historical session prompt when fixed framing otherwise cannot fit."""
    original = (session.get("system_prompt") or "").strip()
    if not original:
        return session, False
    marker = "\n...[historical system prompt clipped for consolidation token budget]"

    def content_for(prompt: str) -> str:
        candidate = dict(session, system_prompt=prompt)
        content = format_chunk_content(
            candidate, [], sess_idx, total_sessions, 999, 999, items=[]
        )
        return (open_questions_block.rstrip() + "\n" + content
                if open_questions_block else content)

    if not fits(content_for("")):
        return session, False
    lo, hi, best = 0, len(original), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt = (original if mid == len(original) else
                  (original[:mid].rstrip() + marker if mid > 0 else ""))
        if fits(content_for(prompt)):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    clipped = (original if best == len(original) else
               (original[:best].rstrip() + marker if best > 0 else ""))
    return dict(session, system_prompt=clipped), best < len(original)


def _fmt_duration(seconds: float) -> str:
    """Human-friendly elapsed time: "45s", "3m 07s", "1h 04m"."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _strip_think_block(text: str) -> str:
    """Drop a leading/embedded ``<think>…</think>`` block, returning the answer body.

    Used for the summary pass, whose recap is the prose AFTER the model's scratch thinking.
    Removes every ``<think>…</think>`` span (there is normally one), then trims. If the block
    is unterminated (a truncated generation), everything up to the dangling ``<think>`` is cut.
    """
    body = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    body = re.sub(r"(?is)<think>.*$", "", body)
    return body.strip()


def _leading_target_cot(target: str) -> str:
    """Return the faithful leading CoT carried by a resolved training target."""
    match = re.match(r"(?is)^\s*<think>(.*?)</think>", target or "")
    return (match.group(1) if match else "").strip()


def _leading_target_answer(target: str) -> str:
    """Return the answer following a resolved target's leading ``<think>…</think>``."""
    match = re.match(r"(?is)^\s*<think>.*?</think>(.*)", target or "")
    return (match.group(1) if match else "").strip()


def _parse_choice(text: str, n_options: int) -> tuple[Optional[int], str]:
    """(chosen index, WHY line) from branch-select output; (None, …) if unparseable."""
    body = re.sub(r"(?s)<think>.*?</think>", "", text or "")
    chosen = None
    m = re.search(r"(?im)^[\s*#>]*CHOICE\s*\**\s*:\s*\**\s*([A-Z])\b", body)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < n_options:
            chosen = idx
    w = re.search(r"(?im)^[\s*#>]*WHY\s*\**\s*:\s*\**\s*(.+)$", body)
    return chosen, (w.group(1).strip(" *") if w else "")


class ReflectionRunner:
    """Synchronous reflection workflow engine, designed to run in a thread executor.

    Instantiated once per server process (lazily). All per-run mutable state
    lives in the ReflectionRunStore; this object carries no per-run state.
    """

    def __init__(
        self,
        *,
        chats_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        consolidation_dir: Path,
        reflection_writer: ReflectionWriter,
        fallback_chats_dir: Optional[Path] = None,
        fallback_memory_dir: Optional[Path] = None,
    ) -> None:
        self._chats_dir = Path(chats_dir)
        self._memory_dir = Path(memory_dir)
        self._runs_dir = Path(runs_dir)
        self._consolidation_dir = Path(consolidation_dir)
        self._writer = reflection_writer
        self._fallback_chats_dir = Path(fallback_chats_dir) if fallback_chats_dir is not None else None
        self._fallback_memory_dir = Path(fallback_memory_dir) if fallback_memory_dir is not None else None

    # ── public entry point ─────────────────────────────────────────────── #

    def execute_run(
        self,
        config: ReflectionRunConfig,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int = 32768,
        tokenizer=None,
        rag_refresh_fn: Optional[Callable] = None,
        send_event_fn: Optional[Callable] = None,
        branch_generate_fn: Optional[Callable] = None,
        branch_chooser_content_fn: Optional[Callable] = None,
        similarity_fn: Optional[Callable] = None,
        persona_context_fn: Optional[Callable] = None,
        persona_keys_fn: Optional[Callable] = None,
        clean_base_ctx: Optional[Callable] = None,
        consolidation_only: bool = False,
        dry_run: bool = False,
        on_session_committed: Optional[Callable] = None,
        on_chat_reflected: Optional[Callable] = None,
        consume_pending_clean_base_fn: Optional[Callable] = None,
        embed_fn: Optional[Callable] = None,
    ) -> None:
        """Execute consolidation, judgement/re-answer, and optionally branch.

        Blocking — designed to be dispatched via asyncio run_in_executor.

        *consolidation_only*: run only the consolidation (first) phase and skip
            judgement/re-answer + branch entirely. Used by the "short reflection summary"
            debug flow, which only wants the distilled fact/persona/ask artifacts.

        *dry_run*: generate and report the consolidation output without writing any
            artifacts (no memory/ledger/sidecar writes, no RAG refresh). Lets an
            operator preview exactly what a real run *would* distil into RAG without
            mutating server data. The streamed text + the structured ``report`` on
            each ``phase_done`` event still flow to the client.

        *generate_fn* signature:
            fn(content: str, system_prompt: str, *, temperature: float,
               top_p: float, max_new_tokens_setting: str,
               before_session: str = "", disable_rag: bool = False,
               disable_thinking: bool = False,
               messages_override: Optional[list[dict]] = None) -> str

            ``messages_override`` is the clean re-answer seam: it supplies the full
            pre-answer conversation and requires ``disable_rag=True``.

        *rag_refresh_fn*: called with no arguments once per session, at the end of
            that session's consolidation phase (only if it distilled anything), to
            rebuild the reflection RAG index. A rebuild re-embeds every live memory
            item, so it is deliberately not run per pass.

        *send_event_fn*: thread-safe callable(event_dict) that pushes a progress
            event to the currently connected WebSocket client.

        *branch_generate_fn*: fn(filename, exchange_index, temperature, top_p) -> dict
            Returns {eligible, candidates, dropped, n_generated, original, ...}.
            Required for branching; branching is skipped if None.

        *branch_chooser_content_fn*: fn(payload, system_prompt) -> str
            Builds the budgeted chooser prompt given shuffled candidates.
            Required for branching; branching is skipped if None.

        *similarity_fn*: fn(text_a, text_b) -> Optional[float] cosine similarity over the
            RAG embedder. Used only by the CoT-regeneration gate (approach #3) for a
            corrupt-CoT exchange the judge kept; absent ⇒ CoT-regen falls back to
            answer-only (the prior corrupt-CoT behavior).

        *persona_context_fn*: fn(user_prompt, before_session) -> str returning Ava's
            relevant persona self-knowledge (persona-only RAG, temporally cut to before the
            chat). Threaded into the clean IDEAL generation so a revise/ideal-win target is
            authored persona-conditioned (replaces the retired build-time persona CoT
            prepend — see build_dataset._inject). Absent ⇒ IDEALs generate persona-free
            (the block is empty; keep/branch targets are persona-free regardless).

        Branching is a core part of revision: it always runs when both branch
        callbacks are provided, and is skipped only where they are absent.

        *on_session_committed*: optional fn(filename) invoked right after a session's
            sidecar is frozen (reflect-once), so a caller can durably checkpoint each
            completed chat and recover it if the run is later interrupted. Not called
            for skipped/dry/consolidation-only/stopped sessions (nothing was frozen).

        *config.chat_only* (background per-chat pass, core.background_reflection): run only
            the per-chat passes (consolidation + revision + branch generation) and SKIP the
            run-level cross-cutting work (persona digest, and the clean-base judge + fact
            placement — the caller also passes ``clean_base_ctx=None``). Each completed chat
            is frozen ``chat_reflected`` (stage one of the two-stage freeze) instead of
            ``reflected_at``, and its collected clean-base job payloads are handed to
            *on_chat_reflected* fn(filename, judge_jobs, fact_candidates) for the caller to
            persist. A later NORMAL run picks the chat up (it is not yet ``reflected_at``),
            loads those payloads via *consume_pending_clean_base_fn* fn(filename) ->
            {"judge_jobs": [...], "fact_candidates": [...]} (consume-once), runs the
            clean-base phase over them, and stamps ``reflected_at`` — finishing the chat
            without re-generating the expensive per-chat passes.

        *embed_fn* fn(texts) -> normalized vectors: the RAG embedder, for the fact-dedup
            pass's subject blocking (CPU, so it is unaffected by the clean-base swap it
            runs inside). Absent ⇒ dedup falls back to lexical blocking, which on a real
            corpus finds almost nothing — see ``core.fact_dedup``.
        """
        run_id = config.run_id
        store.update_status(run_id, status="running")
        run_t0 = time.monotonic()
        # Surface the model + adapter this run is using so the Sleep log makes it
        # obvious reflection is running on the intended (latest) weights, not the
        # bare base. Pulled from the run state, which carries the provenance scalars.
        run_state = store.get_run(run_id) or {}
        model_id = run_state.get("model_id") or "?"
        adapter_id = run_state.get("adapter_id")
        adapter_label = os.path.basename(adapter_id.rstrip("/")) if adapter_id else "none (base model)"
        self._emit(store, send_event_fn, run_id, "run_started",
                   message=f"Runner started — model={model_id} adapter={adapter_label}",
                   model_id=model_id, adapter_id=adapter_id)

        # Branching is a core part of revision and runs whenever the branch-generation
        # capability is wired in — both the WebSocket server and the headless CLI provide
        # it (shared core.branch_replay). It is skipped if a caller omits the callbacks,
        # or when the run explicitly requests it via the `skip_branching` override (Sleep
        # tab's "Skip branching" checkbox): no branch generation, no branch judge — the
        # trainable target resolves directly to the kept original or the revised IDEAL.
        _skip_branching = bool(getattr(config.overrides, "skip_branching", None))
        branching_enabled = bool(
            branch_generate_fn is not None
            and branch_chooser_content_fn is not None
            and not _skip_branching
        )
        if _skip_branching and branch_generate_fn is not None:
            self._emit(store, send_event_fn, run_id, "branching_disabled",
                       message="Branching disabled for this run (Skip branching): no branch "
                               "generation or judge — training the kept original or revised IDEAL.")

        # Phase-two judge (logged-only). Load the run's current persona digest ONCE — the
        # judge anchors on the digest as it stood at run start (the end-of-run digest pass
        # writes a new one, which the next run uses). The branch loop collects a judge job
        # per exchange; after all sessions, the batched judge runs on the CLEAN base (one
        # adapter swap per run via clean_base_ctx) — see _run_clean_base_judge.
        persona_digest = self._load_persona_digest() if branching_enabled else None
        judge_jobs: list = []
        # Candidate host exchanges the fact-placement judge picks from (one per revised,
        # trainable exchange). Independent of branching/digest — a fact needs a home CoT,
        # not a persona standard — so it is always collected when the runner is revising.
        fact_candidates: list = []
        judge_overrides = 0   # targets the criterion flip rewrote this run (Phase B)
        stats: Optional[RunStats] = None  # created once sessions are loaded + precounted

        try:
            sessions = self._load_sessions(config.selected_sessions)
            open_questions_block = self._build_open_questions_block()

            # Precount the revisable exchanges across all selected sessions — the
            # backbone clock for the global "exchange x/y" readout and the rough ETA —
            # plus how many of them carry no original CoT (useful context for branch
            # eligibility; a revised target authors its own CoT in the clean re-answer).
            # Sessions already frozen by reflect-once are skipped entirely below (see the
            # main loop), so they must be excluded here too — otherwise the precount (and
            # the stats panel it drives) counts work that will never actually run.
            total_exchanges = 0
            exchanges_without_cot = 0
            for _fn, _sess in sessions:
                # A revisit deliberately re-reflects an already-frozen chat, so it must
                # NOT be excluded from the precount (else the stats panel counts no work).
                if not dry_run and not config.revisit and self._sidecar_is_frozen(_fn):
                    continue
                # A chat_reflected chat (background stage one) does NO per-exchange
                # generation in a normal run — it only feeds its persisted clean-base jobs
                # into the run-level phase — so it is not ETA-backbone work here.
                if (not dry_run and not config.chat_only
                        and self._sidecar_is_chat_reflected(_fn)):
                    continue
                if self._is_unanswered_outreach(_sess):
                    continue
                # Human-validated exchanges are preserved (never re-derived) and banned ones
                # are skipped outright, so neither is work this run will do — exclude both
                # from the ETA backbone + stats.
                _skip = self._locked_exchanges(_fn) | self._banned_exchanges(_fn)
                for _job in build_revision_jobs(_sess):
                    if _job["index"] in _skip:
                        continue
                    total_exchanges += 1
                    if not (_job["exchange"].get("assistant_cot") or "").strip():
                        exchanges_without_cot += 1
            n_resurfaced = sum(
                1 for ln in open_questions_block.splitlines()
                if ln.startswith("- ")
            )
            stats = RunStats(
                total_exchanges=total_exchanges,
                exchanges_without_cot=exchanges_without_cot,
                open_questions_resurfaced=n_resurfaced,
                branching_enabled=branching_enabled,
                model_id=(None if model_id == "?" else model_id),
                adapter_id=adapter_id,
                context_length=run_state.get("context_length") or context_length,
                quant=_quant_label(model_id),
                device_total_gb=_vram_device_total_gb(),
                started_monotonic=run_t0,
            )
            store.update_status(run_id, stats=stats.to_status())

            total_sessions = len(sessions)
            con_passes = 0
            con_skipped = 0
            rev_passes = 0
            rev_skipped = 0
            con_report = {"weights": 0, "rag": 0, "evict": 0}
            rev_report = {"pairs": 0, "persona": 0,
                           "target_source": {"original": 0, "revised": 0,
                                             "revised_missing_ideal": 0}}

            for sess_idx, (filename, session) in enumerate(sessions, start=1):
                if store.is_stop_requested(run_id):
                    break

                # Reflect-once (REBUILD.md §3): a chat is reflected exactly once, when new,
                # by its contemporary adapter. Once its sidecar carries `reflected_at` it is
                # frozen — no re-reflection, no continue-staging re-run, no judge override on
                # an old chat. A dry_run is a non-mutating preview, so it is exempt; a
                # revisit run deliberately re-reflects the frozen chat (§ "revisit old chat").
                if not dry_run and not config.revisit and self._sidecar_is_frozen(filename):
                    self._emit(store, send_event_fn, run_id, "session_skipped",
                               session=filename,
                               message="Already reflected (reflect-once) — skipping.")
                    continue

                # An Ava-initiated outreach the user never answered has only her opener
                # (no real dialogue turn) — skip it, but leave it un-frozen so a later
                # reply makes it reflectable.
                if self._is_unanswered_outreach(session):
                    self._emit(store, send_event_fn, run_id, "session_skipped",
                               session=filename,
                               message="Ava-initiated chat with no reply — nothing to reflect on, skipping.")
                    continue

                # Two-stage freeze — stage two: a chat the background pass already reflected
                # per-chat (`chat_reflected`, not yet `reflected_at`) skips the expensive
                # consolidation + revision regeneration. Its persisted clean-base job
                # payloads are loaded into this run's judge/fact lists so the end-of-run
                # clean-base phase finishes it, then it is stamped `reflected_at`. Only in a
                # NORMAL run (a chat_only/background run never selects a chat_reflected chat,
                # and a dry_run/revisit re-derives from scratch).
                if (not dry_run and not config.chat_only and not config.revisit
                        and self._sidecar_is_chat_reflected(filename)):
                    n_j = n_f = 0
                    if consume_pending_clean_base_fn is not None:
                        try:
                            pend = consume_pending_clean_base_fn(filename) or {}
                            for j in (pend.get("judge_jobs") or []):
                                judge_jobs.append(j)
                                n_j += 1
                            for c in (pend.get("fact_candidates") or []):
                                fact_candidates.append(c)
                                n_f += 1
                        except Exception:
                            pass
                    if not store.is_stop_requested(run_id):
                        self._mark_session_reflected(filename)
                        if on_session_committed is not None:
                            try:
                                on_session_committed(filename)
                            except Exception:
                                pass
                    self._emit(store, send_event_fn, run_id, "session_finalized",
                               session=filename,
                               message=(f"Background-reflected chat finalized "
                                        f"({n_j} judge + {n_f} fact job(s) queued for clean base)."))
                    continue

                # 1. Consolidation phase for this session
                sess_con_passes, sess_con_skipped, sess_con_report = self._run_consolidation_for_session(
                    config, filename, session, sess_idx, total_sessions, open_questions_block,
                    generate_fn=generate_fn, store=store, context_length=context_length,
                    tokenizer=tokenizer, rag_refresh_fn=rag_refresh_fn,
                    send_event_fn=send_event_fn,
                    exchange_index=con_passes,
                    dry_run=dry_run,
                    stats=stats,
                )
                con_passes += sess_con_passes
                con_skipped += sess_con_skipped
                con_report["weights"] += sess_con_report["weights"]
                con_report["rag"] += sess_con_report["rag"]
                con_report["evict"] += sess_con_report["evict"]

                # 1.5 Targeted ask-resolution: close the [ask] loop tightly. For any open
                #     question Ava raised in THIS thread (a stamped outreach/synthesis
                #     opener, or a passively-surfaced ask), decide from the actual replies
                #     whether it was answered and evict+distill the answered ones — so the
                #     same question stops re-surfacing every idle window. Write-nothing on
                #     a dry run; a revisit suppresses eviction/persona changes, so skip it.
                if not dry_run and not config.revisit and not store.is_stop_requested(run_id):
                    try:
                        sess_resolved = self._run_ask_resolution_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, rag_refresh_fn=rag_refresh_fn,
                            stats=stats,
                        )
                        con_report["evict"] += sess_resolved
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="ask_resolution", session=filename,
                                   message=f"Ask-resolution error (skipped): {e}")

                # 2. Revision phase for this session — skipped for a consolidation-only
                #    (short summary) run, which only wants the distilled artifacts.
                if not consolidation_only and not store.is_stop_requested(run_id):
                    sess_rev_passes, sess_rev_skipped, sess_rev_report = self._run_revision_for_session(
                        config, filename, session, sess_idx, total_sessions,
                        generate_fn=generate_fn, store=store, context_length=context_length,
                        tokenizer=tokenizer, send_event_fn=send_event_fn,
                        branching_enabled=branching_enabled,
                        branch_generate_fn=branch_generate_fn,
                        branch_chooser_content_fn=branch_chooser_content_fn,
                        similarity_fn=similarity_fn,
                        persona_context_fn=persona_context_fn,
                        persona_keys_fn=persona_keys_fn,
                        persona_digest=persona_digest,
                        judge_jobs=judge_jobs,
                        fact_candidates=fact_candidates,
                        stats=stats,
                        dry_run=dry_run,
                    )
                    rev_passes += sess_rev_passes
                    rev_skipped += sess_rev_skipped
                    rev_report["pairs"] += sess_rev_report["pairs"]
                    rev_report["persona"] += sess_rev_report["persona"]
                    for k, v in sess_rev_report["target_source"].items():
                        rev_report["target_source"][k] = rev_report["target_source"].get(k, 0) + v
                    # No RAG refresh here: revision never mutates rag_memory.jsonl
                    # (it writes persona/revision records), so the reflection index is
                    # unchanged. The per-session refresh happens at the end of the
                    # consolidation phase, before revision runs.

                # 2b. Per-exchange retrieval anchors (descriptor + tags). Producer-side
                #     only — written to the sidecar's `anchors` map, read by nothing yet.
                #     Runs after revision so a failure here cannot affect the trainable
                #     target, and is skipped for a consolidation-only run for the same
                #     reason revision is. Opt-out via `overrides.exchange_anchors=False`.
                if (not consolidation_only and not store.is_stop_requested(run_id)
                        and config.overrides.exchange_anchors is not False):
                    try:
                        self._run_anchor_pass_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, stats=stats, dry_run=dry_run,
                        )
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="anchor", session=filename,
                                   message=f"Anchor pass error (skipped): {e}")

                # 2c. User notes — what this conversation revealed about the PERSON, as
                #     `[impression]` records (the user-side counterpart of the revision
                #     pass's `[persona]` formation). RAG-only: recalled and folded into
                #     that person's standing portrait, never trained. Runs after revision
                #     for the same reason the anchor pass does — a failure here must not be
                #     able to touch the trainable target — and once per session, because a
                #     reading of someone forms across a whole conversation rather than at
                #     one exchange. Runs on a revisit too: re-reading an old conversation
                #     as who she is now yields a genuinely new reading of the person, and
                #     the suppression a revisit applies is about not letting an obsolete
                #     chat reshape *her own* identity, which this pass does not touch.
                #     Opt-out via `overrides.user_notes=False`.
                if (not consolidation_only and not store.is_stop_requested(run_id)
                        and getattr(config.overrides, "user_notes", None) is not False):
                    try:
                        self._run_user_notes_pass_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, stats=stats, dry_run=dry_run,
                        )
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="user_notes", session=filename,
                                   message=f"User-notes pass error (skipped): {e}")

                # 2c-bis. Self notes — the OUTSIDE view: what this conversation shows
                #     about HER to someone reading the transcript with no access to her
                #     <think>, written as `[self_impression]` and folded into
                #     `users/_self.json` (core.self_portrait). Distinct evidence from the
                #     revision pass's introspective `[persona]`, and deliberately kept
                #     apart from it — the gap between the two readings is the point.
                #     Unlike user notes it runs on `interlocutor: "ai"` transcripts too:
                #     an encounter has nobody to portray but is still a record of how she
                #     comes across. Opt-out via `overrides.self_notes=False`.
                if (not consolidation_only and not store.is_stop_requested(run_id)
                        and getattr(config.overrides, "self_notes", None) is not False):
                    try:
                        self._run_self_notes_pass_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, stats=stats, dry_run=dry_run,
                        )
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="self_notes", session=filename,
                                   message=f"Self-notes pass error (skipped): {e}")

                # 2c-ter. Chat facts — the per-chat extraction PROTOCOL: everything the
                #     conversation established, enumerated literally and exhaustively, to
                #     its own `.facts.json`. Writes NOTHING to the live stores and is read
                #     by no RAG channel; `rag_memory.jsonl` stays authoritative for what
                #     Ava recalls, and this is an immutable per-chat source for offline
                #     processing (a knowledge-graph build). Runs on every session including
                #     `interlocutor: "ai"` ones, and after revision for the same reason its
                #     siblings do. Opt-out via `overrides.chat_facts=False`.
                if (not consolidation_only and not store.is_stop_requested(run_id)
                        and getattr(config.overrides, "chat_facts", None) is not False):
                    try:
                        self._run_chat_facts_pass_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, stats=stats, dry_run=dry_run,
                        )
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="chat_facts", session=filename,
                                   message=f"Chat-facts pass error (skipped): {e}")

                # 2d. Recollection — REVISIT ONLY. What Ava now makes of this conversation,
                #     written as a `[recollection]`: the gist's fresh sibling, dated by the
                #     reading rather than by the chat, so re-deriving an old conversation
                #     produces memory retrievable at full weight today. A first-time
                #     reflection has nothing to look back on, so this is the one pass gated
                #     on `config.revisit`. Runs LAST — after revision and anchors — so it can
                #     read the anchors' tags and so a failure cannot touch the trainable
                #     target. RAG-only: one op-log insert, no weights, no ledger, no
                #     training row. Opt-out via `overrides.recollection=False`.
                if (config.revisit and not consolidation_only
                        and not store.is_stop_requested(run_id)
                        and getattr(config.overrides, "recollection", None) is not False):
                    try:
                        self._run_recollection_pass_for_session(
                            config, filename, session, sess_idx, total_sessions,
                            generate_fn=generate_fn, store=store,
                            context_length=context_length, tokenizer=tokenizer,
                            send_event_fn=send_event_fn, stats=stats, dry_run=dry_run,
                        )
                    except Exception as e:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="recollection", session=filename,
                                   message=f"Recollection pass error (skipped): {e}")

                # Reflect-once stamp (REBUILD.md §3): freeze the sidecar once the session has
                # been fully reflected (consolidation + revision) on a real, completed run.
                # Skipped for dry_run (no writes), consolidation_only (revision didn't run, so
                # the chat isn't fully reflected), and a stop mid-session (partial work).
                if (not dry_run and not consolidation_only
                        and not store.is_stop_requested(run_id)):
                    if config.chat_only:
                        # Stage one of the two-stage freeze: the per-chat passes are done,
                        # but the run-level clean-base passes are NOT (deferred to a later
                        # normal run). Freeze `chat_reflected` (not `reflected_at`) and hand
                        # this chat's collected clean-base job payloads to the caller to
                        # persist. Filter the run-wide lists to THIS session (a background
                        # run reflects one chat, but filter defensively).
                        self._mark_session_chat_reflected(filename)
                        if on_chat_reflected is not None:
                            try:
                                sj = [j for j in judge_jobs
                                      if j.get("session") == filename]
                                sf = [c for c in fact_candidates
                                      if c.get("session") == filename]
                                on_chat_reflected(filename, sj, sf)
                            except Exception:
                                pass
                    else:
                        self._mark_session_reflected(filename)
                        # Episodic worklog: this chat is now processed, so record it in
                        # Ava's own voice and CLOSE the reach-out thread that opened it (if
                        # any). Shared with the background pass's `chat_reflected` freeze —
                        # a chat passes one or the other, never both in the same pass (a
                        # background-frozen chat is finalized by the `continue` branch
                        # above, which records nothing). A chat un-frozen by an in-place
                        # resume and re-reflected does get a second entry, correctly: it
                        # grew new turns. Skipped on a revisit (a re-reading of an
                        # already-recorded chat is not a second conversation).
                        # Before this the record site existed
                        # ONLY on the background path, which has never run on a live GPU:
                        # every reach-out opened a thread nothing ever closed.
                        if not config.revisit:
                            self._record_conversation_worklog(filename, session)
                        # Durably checkpoint this completed chat so an interrupted run's
                        # already-reflected sessions are not re-reflected on restart.
                        if on_session_committed is not None:
                            try:
                                on_session_committed(filename)
                            except Exception:
                                pass

            store.update_status(run_id, skipped_passes=con_skipped + rev_skipped,
                                exchange_index=con_passes + rev_passes,
                                stats=stats.to_status())

            # 3. Persona digest, part one: decide whether to regenerate at all. Model-free
            #    (a ledger fold + fingerprint compare), so it is answered BEFORE the
            #    clean-base window — the clustering half runs inside that window, and an
            #    unchanged run must not pay an adapter swap it has no work for. Skipped for
            #    a revisit run: rebuilding the self-portrait is persona formation, and a
            #    revisit must not let an obsolete chat reshape who Ava is becoming (the
            #    revision pass also writes no persona, so the evidence is unchanged anyway).
            digest_plan = {"should_run": False}
            if (not dry_run and not consolidation_only and not config.revisit
                    and not config.chat_only
                    and not store.is_stop_requested(run_id)):
                digest_plan = self._plan_persona_digest(
                    run_id, store, send_event_fn,
                    force=bool(getattr(config.overrides,
                                       "force_persona_digest", False)),
                )
            run_digest_cluster = bool(digest_plan.get("should_run"))
            # Kill-switch for the blocked clustering (mirrors apply_branch_judge): when
            # explicitly False the digest falls back to the historical single flat
            # grouping call, which does not survive a large persona set.
            _mr = getattr(config.overrides, "persona_cluster_mapreduce", None)
            use_map_reduce = True if _mr is None else bool(_mr)
            # Kill-switch for the polarity screen (same shape): when explicitly False
            # the clustering keeps the historical polarity-blind behaviour — a theme
            # may swallow members that OPPOSE it, and no counter ops are written.
            _pol = getattr(config.overrides, "persona_polarity", None)
            use_polarity = True if _pol is None else bool(_pol)
            digest_evidence: list = []

            # 3b. User portraits, part one — the same model-free gate, per person. Runs on
            #     a revisit too (unlike the persona digest): a revisit's re-reading of an
            #     old conversation is legitimate new evidence about the PERSON, and the
            #     "don't let an obsolete chat reshape who you're becoming" rule that
            #     suppresses the self-portrait there is about Ava's own identity. Skipped
            #     for chat_only (a background per-chat pass does no run-level work) and for
            #     a dry/consolidation-only run, which write nothing.
            user_plans: list = []
            if (not dry_run and not consolidation_only and not config.chat_only
                    and not store.is_stop_requested(run_id)
                    and _user_portraits_enabled()):
                user_plans = self._plan_user_portraits(
                    run_id, store, send_event_fn,
                    force=bool(getattr(config.overrides, "force_user_portrait", False)))
            user_evidence: dict = {}

            # 3b-bis. Outside-view self-portrait, part one — same model-free gate, one
            #     subject. Same conditions as the user portraits above, including running
            #     on a revisit: re-reading an old conversation as a reader is legitimate
            #     new evidence about how she came across, and the "don't let an obsolete
            #     chat reshape who you're becoming" rule that suppresses the persona digest
            #     there is about the INSIDE view, which this artifact deliberately is not.
            self_plan: Optional[dict] = None
            if (not dry_run and not consolidation_only and not config.chat_only
                    and not store.is_stop_requested(run_id)
                    and _self_portrait_enabled()):
                self_plan = self._plan_self_portrait(
                    run_id, store, send_event_fn,
                    force=bool(getattr(config.overrides, "force_self_portrait", False)))
            self_evidence: list = []

            # 3c. Recall-cue purge — GPU-free, so it runs OUTSIDE the clean-base window,
            #     and BEFORE it: fact dedup unions the triggers of everything it merges, so
            #     a broken cue left in place gets fused into a survivor and spreads (observed
            #     on the live store). Cheap enough to run unconditionally — pure string work
            #     over the live fold, no model, no embedder.
            if (not dry_run and not consolidation_only and not config.chat_only
                    and not store.is_stop_requested(run_id)):
                self._run_trigger_purge(run_id, config=config, store=store,
                                        send_event_fn=send_event_fn)

            # 4. Clean-base batch (phase two) — swap the adapter OFF ONCE per run and run
            #    every evaluation that must see the FROZEN base inside that single reload.
            #    Both are evaluations (not Ava's expression), so they are replay-faithful,
            #    mode-collapse-guarded, and immune to a bad adapter:
            #      (a) the branch judge — "which reply is who I'm becoming" (logged / flip);
            #      (b) fact placement — assign each unhosted [fact] the host exchange whose
            #          reasoning rests on it, so train_cycle can inject it into that CoT;
            #      (c) persona clustering — group persona statements into the themes the
            #          digest is written from ("do these two say the same thing?");
            #      (d) fact dedup — the same question asked of the [fact] store, whose
            #          write-time dedup is exact-key only, so paraphrases accumulate.
            #    CleanBaseSession is a full reload, so batching all four here pays it once.
            #    The judge needs a digest + jobs; fact placement needs only candidate
            #    exchanges; clustering needs only the planned evidence; dedup needs
            #    nothing this run produced. NOTE the judge uses
            #    the digest loaded at run START (see above), never the one this run's step 5
            #    writes — which is exactly what lets clustering move in here.
            _flip = getattr(config.overrides, "apply_branch_judge", None)
            apply_flip = True if _flip is None else bool(_flip)
            run_branch_judge = bool(judge_jobs and persona_digest is not None)
            run_fact_placement = bool(fact_candidates)
            # (d) fact dedup — collapse paraphrases in the live [fact] store. Unlike the
            #     three above it depends on nothing this run produced, so it is the one
            #     task that can carry the batch on its own: a run that judged nothing and
            #     clustered nothing should still tidy the store it just wrote to.
            run_fact_dedup = getattr(config.overrides, "fact_dedup", None) is not False
            if (clean_base_ctx is not None
                    and (run_branch_judge or run_fact_placement or run_digest_cluster
                         or user_plans or self_plan or run_fact_dedup)
                    and not store.is_stop_requested(run_id)):
                def _clean_base_batch():
                    nonlocal digest_evidence, user_evidence, self_evidence
                    overrides = 0
                    if run_branch_judge:
                        overrides = self._run_clean_base_judge(
                            judge_jobs, persona_digest, run_id,
                            config=config, generate_fn=generate_fn,
                            branch_chooser_content_fn=branch_chooser_content_fn,
                            store=store, send_event_fn=send_event_fn,
                            apply_flip=apply_flip, stats=stats,
                            tokenizer=tokenizer,
                        ) or 0
                    if run_fact_placement and not store.is_stop_requested(run_id):
                        self._run_clean_base_fact_placement(
                            fact_candidates, run_id, config=config,
                            generate_fn=generate_fn, store=store,
                            send_event_fn=send_event_fn, stats=stats, tokenizer=tokenizer)
                    if run_digest_cluster and not store.is_stop_requested(run_id):
                        digest_evidence = self._cluster_persona_digest(
                            digest_plan, run_id, generate_fn, store, send_event_fn,
                            map_reduce=use_map_reduce, polarity=use_polarity)
                    if user_plans and not store.is_stop_requested(run_id):
                        user_evidence = self._cluster_user_portraits(
                            user_plans, run_id, generate_fn, store, send_event_fn)
                    if self_plan and not store.is_stop_requested(run_id):
                        self_evidence = self._cluster_self_portrait(
                            self_plan, run_id, generate_fn, store, send_event_fn)
                    # Last: store maintenance rather than this run's own material, and
                    # running it here means it also sees the facts this run distilled.
                    if run_fact_dedup and not store.is_stop_requested(run_id):
                        self._run_clean_base_fact_dedup(
                            run_id, config=config, generate_fn=generate_fn,
                            store=store, send_event_fn=send_event_fn,
                            embed_fn=embed_fn)
                    return overrides
                try:
                    judge_overrides = clean_base_ctx(_clean_base_batch) or 0
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "phase_error",
                               phase="branch_judge",
                               message=f"Clean-base batch error (skipped): {e}")

            # 5. Persona digest, part two: synthesize the self-portrait on the ADAPTER —
            #    authorship is Ava's, unlike the clustering that fed it. When no clean-base
            #    swap is available at all (the headless reflection_run.py CLI), cluster
            #    here on the adapter instead: the clean base is an upgrade where it exists,
            #    not a requirement, and the blocked grouping still applies either way.
            if run_digest_cluster and not store.is_stop_requested(run_id):
                try:
                    if not digest_evidence:
                        digest_evidence = self._cluster_persona_digest(
                            digest_plan, run_id, generate_fn, store, send_event_fn,
                            map_reduce=use_map_reduce, polarity=use_polarity)
                    if digest_evidence:
                        self._synthesize_persona_digest(
                            digest_plan, digest_evidence, run_id, generate_fn,
                            store, send_event_fn, gate_screen=use_polarity)
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "phase_error",
                               phase="persona_digest",
                               message=f"Persona digest error (skipped): {e}")

            # 6. User portraits, part two: synthesize each person's portrait on the
            #    ADAPTER — her reading of someone, in her voice, exactly as the
            #    self-portrait above. When no clean-base swap is available at all (the
            #    headless reflection_run.py CLI), cluster here on the adapter instead: the
            #    clean base is an upgrade where it exists, not a requirement.
            if user_plans and not store.is_stop_requested(run_id):
                try:
                    if not user_evidence:
                        user_evidence = self._cluster_user_portraits(
                            user_plans, run_id, generate_fn, store, send_event_fn)
                    self._synthesize_user_portraits(
                        user_plans, user_evidence, run_id, generate_fn,
                        store, send_event_fn)
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "phase_error",
                               phase="user_portrait",
                               message=f"User portrait error (skipped): {e}")

            # 6-bis. Outside-view portrait, part two: synthesize on the ADAPTER, for the
            #     same reason as the two above — reading her own transcripts back is still
            #     her reading, in her voice. Same no-clean-base fallback (the headless CLI):
            #     cluster here instead, since the clean base is an upgrade where it exists
            #     rather than a requirement.
            if self_plan and not store.is_stop_requested(run_id):
                try:
                    if not self_evidence:
                        self_evidence = self._cluster_self_portrait(
                            self_plan, run_id, generate_fn, store, send_event_fn)
                    self._synthesize_self_portrait(
                        self_plan, self_evidence, run_id, generate_fn,
                        store, send_event_fn)
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "phase_error",
                               phase="self_portrait",
                               message=f"Outside-view portrait error (skipped): {e}")

        except Exception as e:
            traceback.print_exc()
            self._emit(store, send_event_fn, run_id, "run_failed", message=str(e))
            # Persist whatever stats accumulated before the failure — the VRAM peak
            # and timings up to the crash are still useful for diagnosis.
            try:
                if stats is not None:
                    store.persist_report(run_id, stats.build_report())
            except Exception:
                pass
            store.finalize_run(run_id, "failed", {
                "error": str(e), "mutations_applied": False,
                "elapsed_seconds": round(time.monotonic() - run_t0, 1),
                "stats": stats.to_status() if stats is not None else None,
            })
            return

        elapsed_seconds = round(time.monotonic() - run_t0, 1)
        stopped = store.is_stop_requested(run_id)
        final_status = "stopped" if stopped else "completed"
        report = stats.build_report()
        store.persist_report(run_id, report)
        store.finalize_run(run_id, final_status, {
            # A dry run never writes artifacts, so it never mutates anything.
            "mutations_applied": (not stopped) and not dry_run,
            "dry_run": dry_run,
            "consolidation_only": consolidation_only,
            "consolidation_passes": con_passes,
            "revision_passes": rev_passes,
            "skipped_passes": con_skipped + rev_skipped,
            # Criterion-flip overrides this run — the train hand-off forces validation
            # (probe) whenever this is > 0, so a judge-driven cycle is never promoted
            # unguarded (risk-proportional: no-op runs keep the fast skip-validation path).
            "judge_overrides": judge_overrides,
            # Wall-clock time the whole run took, surfaced in the Sleep tab's
            # completion line (and available to a reconnecting client / CLI).
            "elapsed_seconds": elapsed_seconds,
            # Aggregate totals of what this run recorded, so a reconnecting client
            # (or a CLI run) can show the report without replaying events.
            "report": {"consolidation": con_report, "revision": rev_report},
            # Compact live snapshot (elapsed/eta/vram/discards) for the stats panel
            # on a reconnect; the full detailed report rides in `detailed_report`
            # (also persisted to <run_id>.report.json and the reflection archive).
            "stats": stats.to_status(),
            "detailed_report": report,
        })
        event_type = "run_stopped" if stopped else "run_completed"
        base_msg = "Run stopped (user request)" if stopped else "All passes done"
        msg = f"{base_msg} — took {_fmt_duration(elapsed_seconds)}"
        self._emit(store, send_event_fn, run_id, event_type, message=msg,
                   elapsed_seconds=elapsed_seconds)

    # ── consolidation phase ────────────────────────────────────────────── #

    def _run_consolidation_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        open_questions_block: str,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        rag_refresh_fn: Optional[Callable],
        send_event_fn: Optional[Callable],
        exchange_index: int,
        dry_run: bool = False,
        stats: Optional[RunStats] = None,
    ) -> tuple[int, int, dict]:
        """Run the consolidation pass over a single session. Returns (passes, skipped, agg).

        When *dry_run* is set, the pass still generates and reports the distilled
        artifacts but writes nothing — no memory/ledger persistence and no RAG
        refresh — so an operator can preview what would land in RAG.
        """
        run_id = config.run_id

        from core.reflection_prompts import load_reflection_prompts
        prompts = load_reflection_prompts()
        sleep_prompt = config.overrides.sleep_prompt or prompts.sleep_prompt
        # Dedicated per-conversation SUMMARY pass (consolidation-gist). "" when the prompt
        # file is absent (older checkout) → the pass is skipped, leaving no gist behind.
        summary_prompt = prompts.summary_prompt

        sampling = config.overrides.sleep_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P
        max_new_tokens = (sampling.max_new_tokens_setting
                          if sampling else _DEFAULT_MAX_NEW_TOKENS)

        # Open asks keyed by content_key, so write_consolidation can route any
        # [resolved] it produces into a distilled fact/persona by the ask's kind
        # (resolve-and-distill). Read with the live fallback because a staged run's
        # own memory_dir starts empty. Best-effort: a fold failure just disables
        # distillation for this session, never the pass.
        ask_index: dict = {}
        try:
            mem = ReflectionMemory(
                self._memory_dir, fallback_memory_dir=self._fallback_memory_dir
            )
            ask_index = {r["key"]: r for r in mem.open_questions() if r.get("key")}
        except Exception:
            ask_index = {}

        total_passes = 0
        skipped_passes = 0
        # Running aggregate of what consolidation routed (weights facts/persona,
        # RAG inserts, resolved evictions) — for the run summary's report block.
        agg = {"weights": 0, "rag": 0, "evict": 0}
        store.update_status(run_id, phase="consolidation",
                            session_total=total_sessions,
                            session_index=sess_idx)

        self._emit(store, send_event_fn, run_id, "session_started",
                   session=filename, session_index=sess_idx,
                   session_total=total_sessions,
                   message=f"Consolidation: session {sess_idx}/{total_sessions}")

        output_reserve = _consolidation_output_reserve(max_new_tokens, context_length)
        input_limit = max(1, context_length - output_reserve - 128)
        rag_token_limit = min(2048, max(256, context_length // 12))
        oq_token_limit = min(2048, max(256, context_length // 12))
        shown_open_questions = _clip_lines_to_tokens(
            open_questions_block, tokenizer, oq_token_limit
        )

        prepare_prompt = getattr(generate_fn, "prepare_prompt", None)
        prepared_cache: dict[tuple[bool, str], object] = {}
        rag_enabled = True

        def _prepare(content: str):
            key = (rag_enabled, content)
            if key not in prepared_cache:
                if prepare_prompt is None:
                    prepared_cache[key] = None
                else:
                    prepared_cache[key] = prepare_prompt(
                        content, sleep_prompt,
                        before_session=filename,
                        disable_rag=not rag_enabled,
                        rag_include_chat=False,
                        max_rag_tokens=rag_token_limit,
                    )
            return prepared_cache[key]

        def _fits(content: str) -> bool:
            prepared = _prepare(content)
            if prepared is None:
                return _count_tokens(tokenizer, content) <= int(context_length * 0.45)
            return int(prepared.input_tokens) <= input_limit

        chunk_session = session
        system_prompt_clipped = False
        rag_dropped_for_fit = False
        try:
            chunks = build_consolidation_chunks(
                chunk_session, context_length,
                open_questions_block=shown_open_questions,
                tokenizer=tokenizer, session_idx=sess_idx,
                session_total=total_sessions, fits=_fits,
            )
        except Exception as first_error:
            # RAG is useful context but not transcript evidence. If a chunk-specific
            # retrieval block cannot fit, drop RAG before touching the historical prompt.
            rag_enabled = False
            prepared_cache.clear()
            rag_dropped_for_fit = True
            try:
                chunks = build_consolidation_chunks(
                    chunk_session, context_length,
                    open_questions_block=shown_open_questions,
                    tokenizer=tokenizer, session_idx=sess_idx,
                    session_total=total_sessions, fits=_fits,
                )
            except Exception:
                chunk_session, system_prompt_clipped = _clip_session_prompt_to_fit(
                    session, shown_open_questions, sess_idx, total_sessions, _fits
                )
                if not system_prompt_clipped:
                    raise first_error
                prepared_cache.clear()
                chunks = build_consolidation_chunks(
                    chunk_session, context_length,
                    open_questions_block=shown_open_questions,
                    tokenizer=tokenizer, session_idx=sess_idx,
                    session_total=total_sessions, fits=_fits,
                )
        chunk_total = len(chunks)

        # Accumulated SUMMARY recap across this session's chunks (one per chunk, joined and
        # written to the sidecar after the loop). A single-chunk session — the common case —
        # yields one recap; a chunked large session concatenates its per-chunk recaps.
        summary_parts: list[str] = []

        for chunk in chunks:
            if store.is_stop_requested(run_id):
                break

            part, parts = chunk["part"], chunk["parts"]
            store.update_status(run_id, chunk_index=part,
                                chunk_total=chunk_total,
                                exchange_index=exchange_index + total_passes)
            content = chunk["content"]
            prepared = _prepare(content)
            input_tokens = (int(prepared.input_tokens) if prepared is not None
                            else _count_tokens(tokenizer, content))
            rag_tokens = int(prepared.rag_tokens) if prepared is not None else 0
            fragment_note = ""
            if chunk.get("fragments"):
                frag = chunk["fragments"][0]
                fragment_note = (
                    f", exchange {frag['exchange_index'] + 1} fragment "
                    f"{frag['fragment_index']}/{frag['fragment_count']}"
                )
            fit_notes: list[str] = []
            if chunk.get("overlap_applied"):
                fit_notes.append("overlap kept")
            elif parts > 1:
                fit_notes.append("overlap omitted")
            if rag_dropped_for_fit:
                fit_notes.append("RAG omitted")
            if system_prompt_clipped:
                fit_notes.append("historical prompt clipped")
            budget_note = (
                f" — input {input_tokens}/{input_limit}, reserve {output_reserve}"
                + (("; " + ", ".join(fit_notes)) if fit_notes else "")
            )
            self._emit(
                store, send_event_fn, run_id, "phase_started",
                phase="consolidation", session=filename,
                session_index=sess_idx, chunk=part, chunk_total=parts,
                input_tokens=input_tokens, input_token_budget=input_limit,
                output_token_reserve=output_reserve, rag_tokens=rag_tokens,
                overlap_applied=bool(chunk.get("overlap_applied")),
                fragments=chunk.get("fragments") or [],
                message=(
                    f"Consolidation: session {sess_idx}/{total_sessions}"
                    + (f", part {part}/{parts}" if parts > 1 else "")
                    + fragment_note + budget_note
                ),
            )

            def _on_chunk(delta: str, _sess=filename, _part=part, _parts=parts) -> None:
                self._emit(store, send_event_fn, run_id, "phase_progress",
                           phase="consolidation", session=_sess,
                           chunk=_part, chunk_total=_parts, text=delta)

            response = None
            last_err: Optional[Exception] = None
            _vram_reset_peak()
            gen_t0 = time.monotonic()
            for attempt in range(_CONSOLIDATION_GENERATE_RETRIES + 1):
                try:
                    response = generate_fn(
                        content, sleep_prompt,
                        temperature=temperature, top_p=top_p,
                        max_new_tokens_setting=max_new_tokens,
                        before_session=filename,
                        disable_rag=not rag_enabled,
                        rag_include_chat=False,
                        max_rag_tokens=rag_token_limit,
                        input_token_limit=input_limit,
                        prepared_prompt=prepared,
                        on_chunk=_on_chunk,
                    )
                    break
                except Exception as e:
                    last_err = e
                    deterministic_budget_error = hasattr(e, "input_limit")
                    if attempt < _CONSOLIDATION_GENERATE_RETRIES and not deterministic_budget_error:
                        # Transient backend error (typically the TorchDynamo FX-trace
                        # race after a streamer timeout). Dynamo falls back to eager
                        # after the first failure, so retrying the same chunk recovers.
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="consolidation", session=filename,
                                   message=f"Consolidation error (retrying): {e}")
            if stats is not None:
                stats.record_phase("consolidation", time.monotonic() - gen_t0,
                                   tokens=_count_tokens(tokenizer, response or ""))
                stats.observe_vram(_vram_peak_gb(), phase="consolidation",
                                   context_tokens=input_tokens)
            if response is None:
                skipped_passes += 1
                if stats is not None:
                    stats.note_discard("consolidation_gen_error")
                    store.update_status(run_id, stats=stats.to_status())
                self._emit(store, send_event_fn, run_id, "pass_error",
                           session=filename,
                           message=f"Consolidation error (skipped): {last_err}")
                continue

            total_passes += 1
            if stats is not None:
                stats.note_consolidation_pass()
            chunk_meta: dict = {}
            if parts > 1:
                chunk_meta = {"chunk_index": part, "chunk_count": parts}

            # A pass that hit the token cap (vs. ending on EOS) is cut mid-string, so
            # its final item is a half-written [fact]/[ask]. Flag it so the parser
            # drops that fragment rather than poisoning memory + a later training row.
            truncated = bool(getattr(generate_fn, "last_truncated", False))
            if truncated:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="consolidation", session=filename,
                           message="Consolidation hit token cap — dropping the "
                                   "truncated final item")

            # Structured breakdown of what this pass recorded — computed the same
            # way the writer routes it, so report and artifacts can't drift.
            # Drives the Sleep-tab report.
            con_report = parse_consolidation(response, truncated=truncated)
            agg["weights"] += len(con_report["weights"])
            agg["rag"] += len(con_report["rag"])
            agg["evict"] += len(con_report["resolved"])
            if stats is not None:
                stats.note_rag(inserts=len(con_report["rag"]),
                               evicts=len(con_report["resolved"]))
                stats.note_resolved(len(con_report["resolved"]))

            # Dry run: report what was distilled but persist nothing.
            if not dry_run:
                try:
                    summary = self._writer.write_consolidation(
                        run_id=run_id, source_session=filename,
                        text=response, ask_index=ask_index, truncated=truncated,
                        # Who was speaking in this session — the `source` half of a
                        # fact's attribution. Read from the session record rather than
                        # from the model's output so provenance can't be hallucinated.
                        source_user=(session.get("user") or "").strip(),
                        **chunk_meta,
                    )
                    register_consolidation_anchors(
                        self._consolidation_dir, summary, filename
                    )
                except Exception as e:
                    skipped_passes += 1
                    if stats is not None:
                        stats.note_discard("persist_error")
                        store.update_status(run_id, stats=stats.to_status())
                    self._emit(store, send_event_fn, run_id, "pass_error",
                               session=filename,
                               message=f"Consolidation persist error (skipped): {e}")
                    continue
            if stats is not None:
                store.update_status(run_id, stats=stats.to_status())

            self._emit(
                store, send_event_fn, run_id, "phase_done",
                phase="consolidation", session=filename,
                chunk=part, chunk_total=parts,
                report=con_report,
                message=("Consolidation pass done (dry run — not written)"
                         if dry_run else "Consolidation pass done")
            )

            # SUMMARY pass: a second generation over the SAME packed transcript that produces
            # a distilled prose recap of this chunk. Skipped on a dry run (a preview persists
            # nothing, so the recap would be discarded) and when no summary prompt is
            # configured. Best-effort: a summary failure never aborts the pass — the gist is
            # an enhancement, not a correctness requirement.
            #
            # It must run under the SUMMARY prompt, NOT the consolidation `prepared` object.
            # `prepared` bakes in `sleep_prompt` (the consolidation instruction) AND the
            # reflection-memory RAG block, and generate_fn IGNORES the system_prompt argument
            # whenever prepared_prompt is passed — so reusing `prepared` here silently ran the
            # recap under the consolidation prompt, making the model emit a second
            # `## WEIGHTS`/`## RAG`/`[fact]` dump instead of prose (and the `[fact]`/`[ask]`
            # RAG lines few-shot-primed exactly that). We therefore rebuild the prompt with
            # `summary_prompt` and RAG DISABLED — a recap only needs the transcript (`content`),
            # not the structured memory that biases it toward the consolidation format. Only
            # the prompt is rebuilt; the expensive transcript packing (`content`) is reused.
            #
            # The content also gets a CLOSING the consolidation pass does not (see
            # `reflection_chunking.SUMMARY_CLOSING`): with the transcript last, the nearest
            # thing to continue is the reply that ended the chat, and the pass was observed
            # answering it instead of recapping. It is appended here rather than baked into
            # the chunks because these chunks are SHARED with consolidation, whose contract
            # would break under a "now write the recap" line. The cost is that the closing
            # is not counted by the packing budget — safe, since dropping the RAG block
            # frees up to `rag_token_limit` against a closing of a few dozen tokens, and an
            # overflow is a caught `PromptBudgetError` that skips the gist with a warning.
            if summary_prompt and not dry_run and not store.is_stop_requested(run_id):
                sum_content = append_closing(content, SUMMARY_CLOSING)
                sum_prepared = None
                if prepare_prompt is not None:
                    try:
                        sum_prepared = prepare_prompt(
                            sum_content, summary_prompt,
                            before_session=filename,
                            disable_rag=True,
                        )
                    except Exception:
                        sum_prepared = None

                def _on_sum_chunk(delta: str, _sess=filename, _part=part, _parts=parts) -> None:
                    self._emit(store, send_event_fn, run_id, "phase_progress",
                               phase="summary", session=_sess,
                               chunk=_part, chunk_total=_parts, text=delta)
                try:
                    sum_t0 = time.monotonic()
                    sum_raw = generate_fn(
                        sum_content, summary_prompt,
                        temperature=temperature, top_p=top_p,
                        max_new_tokens_setting=max_new_tokens,
                        before_session=filename,
                        disable_rag=True,
                        rag_include_chat=False,
                        max_rag_tokens=rag_token_limit,
                        input_token_limit=input_limit,
                        prepared_prompt=sum_prepared,
                        on_chunk=_on_sum_chunk,
                    )
                    if stats is not None:
                        stats.record_phase("summary", time.monotonic() - sum_t0,
                                           tokens=_count_tokens(tokenizer, sum_raw or ""))
                    sum_text = _strip_think_block(sum_raw or "")
                    if sum_text:
                        summary_parts.append(sum_text)
                    self._emit(store, send_event_fn, run_id, "phase_done",
                               phase="summary", session=filename,
                               chunk=part, chunk_total=parts,
                               message="Summary pass done")
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "pass_warning",
                               phase="summary", session=filename,
                               message=f"Summary generation error (skipped): {e}")

        # Persist this session's distilled recap to its sidecar (consolidation-gist). Written
        # once per session after the chunk loop; rag_engine chunks it into the chat index on
        # the slow gist tent. Best-effort and RAG-only — never a weights/ledger write.
        if summary_parts and not dry_run:
            try:
                from core.chat_sidecar import ChatSidecar, sanitize_gist
                # Sanitize per PART, not just on the joined text: a chunked session runs one
                # summary generation per chunk, and each can independently degenerate into
                # the structured consolidation dump. Cutting each part at its own seam keeps
                # the prose of the parts that stayed clean, where sanitizing only the join
                # would truncate the whole summary at the first bad part. `write_summary`
                # sanitizes again (it is the store's own guard) — a no-op on this text.
                cleaned = [c for c in (sanitize_gist(p) for p in summary_parts) if c]
                if cleaned:
                    ChatSidecar(
                        self._chats_dir, fallback_chats_dir=self._fallback_chats_dir
                    ).write_summary(
                        source_session=filename,
                        summary="\n\n".join(cleaned),
                        run_id=run_id,
                    )
            except Exception:
                pass

        # Refresh the reflection RAG index once for the whole session, rather than
        # after every chunk: a full rebuild re-embeds every live memory item (CPU),
        # so per-pass refresh idled the GPU N times per session. Only consolidation
        # mutates rag_memory.jsonl, so one rebuild here makes this session's distilled
        # memory live for its own revision pass and for every subsequent session.
        # Skipped when nothing was distilled (the index would be unchanged).
        if rag_refresh_fn is not None and total_passes > 0 and not dry_run:
            try:
                rag_refresh_fn()
            except Exception:
                pass

        return total_passes, skipped_passes, agg

    # ── targeted [ask]-loop close ──────────────────────────────────────────── #

    # Small system instruction for the per-thread ask-resolution pass. The transcript +
    # question are supplied in the content; this only frames the judgement.
    _RESOLVE_ASK_SYSTEM = (
        "You are reflecting quietly on your own open questions. You raised a question of "
        "your own in a conversation, and you are re-reading it now to decide one thing: "
        "did that conversation actually answer it? Judge strictly from what was really "
        "said — a deflection, a change of subject, or a non-answer does not count, and "
        "you must not invent an answer that was never given."
    )

    @staticmethod
    def _render_thread_for_resolution(session: dict) -> str:
        """Render a session's exchanges as a plain speaker-labelled transcript.

        Ava's chain-of-thought is dropped — only what was actually *said* bears on
        whether the question was answered. For an Ava-initiated thread, exchange 0's
        synthetic "(initiative)" stimulus is shown only as her opening line.
        """
        lines: list[str] = []
        initiated = (session.get("initiated_by") or "").strip() == "ava"
        who = (session.get("user") or "").strip() or "the person"
        for i, ex in enumerate(session.get("exchanges") or []):
            up = (ex.get("user_prompt") or "").strip()
            resp = (ex.get("assistant_response") or "").strip()
            if initiated and i == 0:
                if resp:
                    lines.append(f"You (opening): {resp}")
                continue
            spk = (ex.get("speaker") or who).strip() or who
            if up:
                lines.append(f"{spk}: {up}")
            if resp:
                lines.append(f"You: {resp}")
        return "\n".join(lines)

    @staticmethod
    def _parse_ask_resolution(text: str) -> tuple[bool, str]:
        """Parse the DECISION/ANSWER block → (answered, answer), reasoning-safe.

        The ``<think>`` CoT is dropped first so the model's deliberation about its choice
        can't be mistaken for the decision; DECISION defaults to *no* on an absent or
        unrecognized line, so a truncated generation reads as 'not answered' (leaving the
        ask open) rather than a spurious resolve.
        """
        text = text or ""
        text = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
        idx = text.lower().find("<think>")
        if idx != -1:
            text = text[:idx]
        answered = False
        m = re.search(r"^\s*DECISION\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).strip().lower().startswith(
                ("yes", "y", "true", "1", "resolv")):
            answered = True
        answer = ""
        m = re.search(r"ANSWER\s*:\s*(.*)\Z", text, re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()
        return answered, answer

    def _scoped_open_asks(self, filename: str, session: dict) -> list[dict]:
        """Open asks Ava raised in THIS thread — stamped opener + passively surfaced.

        Unions the session's stamped ``initiated_ask`` (an outreach/synthesis opener)
        with the surface op-log join (``ReflectionMemory.asks_surfaced_in`` — which also
        covers an ask *passively* surfaced into a user-opened chat), keeps only those
        still open in the live fold, and dedupes by key. Reads the live fallback so a
        staged run (whose own memory starts empty) still sees the open asks. Returns the
        folded ask records (``{key, content, ask_kind, …}``).
        """
        try:
            mem = ReflectionMemory(
                self._memory_dir, fallback_memory_dir=self._fallback_memory_dir)
            open_by_key = {r["key"]: r for r in mem.open_questions() if r.get("key")}
        except Exception:
            return []
        scoped: dict[str, dict] = {}
        stamped = session.get("initiated_ask") or {}
        skey = (stamped.get("key") or "").strip()
        if skey and skey in open_by_key:
            scoped[skey] = open_by_key[skey]
        try:
            for r in mem.asks_surfaced_in(filename):
                k = r.get("key")
                if k and k in open_by_key:
                    scoped[k] = open_by_key[k]
        except Exception:
            pass
        return list(scoped.values())

    def _run_ask_resolution_for_session(
        self, config: ReflectionRunConfig, filename: str, session: dict,
        sess_idx: int, total_sessions: int, *,
        generate_fn: Callable, store: ReflectionRunStore, context_length: int,
        tokenizer, send_event_fn: Optional[Callable],
        rag_refresh_fn: Optional[Callable], stats: Optional[RunStats],
    ) -> int:
        """Targeted [ask]-loop close: resolve the asks Ava raised in this thread.

        For each open ask this conversation was meant to answer (see
        :meth:`_scoped_open_asks`), a small decision pass reads the actual replies and
        judges whether it was answered; an answered ask is evicted from the live fold and
        its answer distilled to a weights-bound item
        (:meth:`ReflectionWriter.write_answered_resolution`). Because the key is known
        from the surfaced-ask join, resolution never depends on the big consolidation
        output emitting a content_key-matching ``[resolved]`` — the fragile path that
        left the same question re-surfacing every idle window. Returns the number
        resolved. Runs after consolidation persisted its own evictions, so an ask the
        consolidation pass already closed is no longer open here (no double-resolve).
        """
        asks = self._scoped_open_asks(filename, session)
        if not asks:
            return 0
        transcript = self._render_thread_for_resolution(session)
        if not transcript.strip():
            return 0
        who = (session.get("user") or "").strip() or "the person"
        run_id = config.run_id
        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="ask_resolution", session=filename,
                   session_index=sess_idx, chunk=1, chunk_total=1,
                   message=(f"Resolving {len(asks)} raised question(s) — session "
                            f"{sess_idx}/{total_sessions}"))
        resolved = 0
        for ask in asks:
            if store.is_stop_requested(run_id):
                break
            key = ask.get("key") or ""
            question = (ask.get("content") or "").strip()
            ask_kind = ask.get("ask_kind") or ""
            if not key or not question:
                continue
            content = (
                f"You raised a question of your own and brought it into this "
                f"conversation with {who}:\n\n  {question}\n\n"
                f"Here is the whole conversation:\n\n{transcript}\n\n"
                f"Decide, honestly, whether this conversation actually answered your "
                f"question — whether you learned what you wanted to know. Answer in "
                f"exactly this form:\n\n"
                f"DECISION: yes            (or: no)\n"
                f"ANSWER: <if yes, state plainly, in your own words, what you now know — "
                f"the answer to your question.>"
            )
            try:
                raw = generate_fn(
                    content, self._RESOLVE_ASK_SYSTEM,
                    temperature=0.3, top_p=0.9, max_new_tokens_setting="768",
                    before_session=filename, disable_rag=True,
                    rag_include_chat=False,
                )
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="ask_resolution", session=filename,
                           message=f"Ask-resolution generate error (skipped): {e}")
                continue
            answered, answer = self._parse_ask_resolution(raw)
            if not answered:
                self._emit(store, send_event_fn, run_id, "ask_unresolved",
                           phase="ask_resolution", session=filename,
                           question=question, ask_kind=ask_kind,
                           message=f"Still open: {question[:70]}")
                continue
            try:
                summary = self._writer.write_answered_resolution(
                    key=key, question=question, answer=answer, ask_kind=ask_kind,
                    source_session=filename, answered_in=filename, run_id=run_id,
                    source_user=(session.get("user") or "").strip())
                register_consolidation_anchors(
                    self._consolidation_dir, summary, filename)
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "pass_error",
                           phase="ask_resolution", session=filename,
                           message=f"Ask-resolution persist error (skipped): {e}")
                continue
            resolved += 1
            if stats is not None:
                try:
                    stats.note_rag(inserts=int(summary.get("weights_recall", 0)),
                                   evicts=1)
                    stats.note_resolved(1)
                except Exception:
                    pass
            self._emit(store, send_event_fn, run_id, "ask_resolved",
                       phase="ask_resolution", session=filename,
                       question=question, ask_kind=ask_kind, answer=answer,
                       distilled=bool(summary.get("weights")),
                       message=f"Resolved: {question[:70]}")
        # An eviction changes live membership, so rebuild the reflection RAG index once
        # (only if something resolved) — makes the closed ask stop surfacing for this
        # run's later sessions and its revision pass.
        if resolved and rag_refresh_fn is not None:
            try:
                rag_refresh_fn()
            except Exception:
                pass
        if stats is not None:
            store.update_status(run_id, stats=stats.to_status())
        return resolved

    def _run_anchor_pass_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable],
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> int:
        """Generate a per-exchange retrieval ANCHOR (one-line descriptor + tags).

        A small pass of its OWN — deliberately not extra fields on the revision output.
        Revision's generation resolves the trainable target, and adding fields to that
        prompt would risk the IDEAL quality for an indexing concern; a separate pass with
        its own prompt and RAG disabled costs one short generation per exchange and can
        fail, be skipped, or be re-tuned without touching training. It mirrors the summary
        pass's shape (own prompt, `disable_rag=True`, reuses the packed content).

        Runs on the same replay-faithful context revision uses (`build_revision_content`),
        so a turn that only makes sense as a continuation can still be described. Skips
        filler exchanges (`exchange_anchor.is_anchorable`) and, like every per-chat pass,
        skips human-locked exchanges — an operator-authored target should not have its
        index entry silently re-derived under a later persona.

        Results go to the sidecar's own `anchors` map (outside `exchanges`, so this pass
        can never perturb training), where `rag_engine`'s anchor channel reads them.
        Returns the number of anchors written. Best-effort throughout — an anchor failure
        must never disturb the reflection that surrounds it, but it does REPORT: an empty
        parse emits the raw generation on the warning (`_pass_output_debug`), because the
        skip is otherwise unattributable and the pass silently produces nothing.
        """
        run_id = config.run_id
        from core.reflection_source import build_revision_jobs, build_revision_content
        from core import exchange_anchor

        prompt = self._load_exchange_anchor_prompt()
        if not prompt:
            return 0

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        try:
            jobs = build_revision_jobs(session)
        except Exception:
            return 0
        locked: set = set()
        if not dry_run:
            try:
                locked = self._session_sidecar().locked_exchange_indices(filename)
            except Exception:
                locked = set()

        jobs = [
            j for j in jobs
            if exchange_anchor.is_anchorable(j.get("exchange") or {})
            and int(j.get("index", -1)) not in locked
        ]
        if not jobs:
            return 0

        total = len(jobs)
        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="anchor", session=filename, session_index=sess_idx,
                   exchange_total=total,
                   message=(f"Anchors: session {sess_idx}/{total_sessions}"
                            f" ({total} exchanges)"))

        written = 0
        sidecar = self._session_sidecar()
        for n, job in enumerate(jobs, start=1):
            if store.is_stop_requested(run_id):
                break
            ex_index = int(job.get("index", -1))
            try:
                content = build_revision_content(job, session, context_length)
                t0 = time.monotonic()
                raw = generate_fn(
                    content, prompt,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens_setting=_ANCHOR_MAX_NEW_TOKENS,
                    before_session=filename,
                    disable_rag=True,
                    disable_thinking=True,
                    rag_include_chat=False,
                )
                truncated = getattr(generate_fn, "last_truncated", None)
                if stats is not None:
                    stats.record_phase("anchor", time.monotonic() - t0,
                                       tokens=_count_tokens(tokenizer, raw or ""))
                about, tags = exchange_anchor.parse_anchor_output(
                    _strip_think_block(raw or ""))
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="anchor", session=filename, exchange_index=n,
                           exchange_total=total,
                           message=f"Anchor error (skipped): {e}")
                continue

            if not exchange_anchor.has_content(about, tags):
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="anchor", session=filename, exchange_index=n,
                           exchange_total=total, source_exchange_index=ex_index,
                           text=_pass_output_debug(
                               raw,
                               "about={} tags={}".format(
                                   repr(about) if (about or "").strip() else "∅",
                                   list(tags or [])),
                               truncated=truncated, generate_fn=generate_fn),
                           message="Anchor unparseable or empty (skipped)")
                continue

            if not dry_run:
                try:
                    if sidecar.write_anchor(
                        source_session=filename, exchange_index=ex_index,
                        about=about, tags=tags, run_id=run_id,
                        generation=exchange_anchor.ANCHOR_GENERATION,
                    ):
                        written += 1
                except Exception:
                    pass

            # The whole point of this first build is reading what it produced, so the
            # descriptor and its tags ride the event itself rather than only a counter.
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="anchor", session=filename, exchange_index=n,
                       exchange_total=total, source_exchange_index=ex_index,
                       about=about, tags=tags,
                       text=(about + ("\n#" + " #".join(tags) if tags else "")),
                       message=f"Anchor {n}/{total}: {about[:70]}")

        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="anchor", session=filename, session_index=sess_idx,
                   exchange_total=total,
                   message=(f"Anchors: {written}/{total} written"
                            if not dry_run else
                            f"Anchors: {total} generated (dry run — nothing written)"))
        return written

    @staticmethod
    def _session_person(session: dict) -> str:
        """Who Ava was talking to in *session* — ``""`` when that is nobody nameable.

        The session-level ``user`` is stamped from the first attributed exchange
        (``ChatLogger.log_exchange``); an older transcript may carry it only per-exchange,
        so the first non-empty ``speaker`` is the fallback. Returns ``""`` for an
        ``interlocutor: "ai"`` transcript (an encounter or served gossip, where the other
        side is a model): the user-notes pass is a reading of a *person*, and folding a
        peer instance's turns into someone's portrait would be a category error — the
        same distinction check-in's silence clock relies on this flag for.
        """
        if (session.get("interlocutor") or "").strip().lower() == "ai":
            return ""
        user = (session.get("user") or "").strip()
        if user:
            return user
        for ex in (session.get("exchanges") or []):
            sp = (ex.get("speaker") or "").strip()
            if sp:
                return sp
        return ""

    def _run_user_notes_pass_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable],
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> int:
        """Write what Ava learned about the PERSON in this conversation — ``[impression]``.

        The user-side counterpart of the revision pass's persona formation: that pass asks
        "what did this reveal about me?", this one asks "what did this reveal about them?".
        Both are readings rather than records, which is what separates this pass from
        consolidation — consolidation already keeps what the person *told* her as
        ``[fact]``, and an impression is deliberately the other thing: what she came to
        understand about how they think, talk, and land, which nobody stated outright.

        It is its own pass rather than another section of the consolidation prompt for two
        reasons. The consolidation prompt is already carrying WEIGHTS/RAG/RESOLVED across a
        *chunked* session, so a person-level reading written per chunk would be formed from
        a fragment and emitted several times per chat. And the framing genuinely differs:
        this pass wants the near-side-of-diagnosis discipline and the "writing nothing is
        correct" permission that would be noise inside a prompt whose job is to distil
        stated content.

        RAG-only by construction (``ReflectionWriter.write_impressions``): recalled and
        portrait-folded, never trained.

        Runs after revision so a failure here cannot touch the trainable target, and once
        per session rather than per exchange — a reading of someone forms across a whole
        conversation. Best-effort throughout. Returns the number of impressions written.
        """
        run_id = config.run_id
        from core.reflection_source import (
            build_revision_jobs, build_session_reading_content,
            build_echo_index, is_transcript_echo)
        from core.reflection_writer import parse_impressions
        from core.user_digest import person_slug

        prompt = self._load_user_notes_prompt()
        if not prompt:
            return 0

        person = self._session_person(session)
        if not person_slug(person):
            # No nameable person (an AI interlocutor, or a transcript with no speaker):
            # there is nobody for the reading to be *about*, so the pass is a no-op rather
            # than an error. Silent — this is an ordinary shape, not a problem.
            return 0

        try:
            jobs = build_revision_jobs(session)
        except Exception:
            return 0
        if not jobs:
            return 0

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="user_notes", session=filename, session_index=sess_idx,
                   person=person,
                   message=f"User notes: what {filename} revealed about {person}")

        # The whole conversation, neutrally framed — NOT the revision builder this pass
        # started out reusing. That one ends on the last exchange under "The exchange you
        # are judging:", with Ava's own <think> and reply as the last and largest text in
        # the prompt; asked for a sentence directly after it, the pass was observed
        # emitting that reply back as the impression. `closing` puts the pass's own
        # question last instead, so the final thing before generation is the task.
        try:
            content = build_session_reading_content(
                session, context_length,
                closing=_USER_NOTES_CLOSING.replace("{person}", person))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="user_notes", session=filename,
                       message=f"User-notes content error (skipped): {e}")
            return 0
        if not content:
            return 0

        def _on_chunk(delta: str, _sess=filename) -> None:
            self._emit(store, send_event_fn, run_id, "phase_progress",
                       phase="user_notes", session=_sess, text=delta)

        try:
            t0 = time.monotonic()
            raw = generate_fn(
                content, prompt.replace("{person}", person),
                temperature=temperature, top_p=top_p,
                max_new_tokens_setting=_USER_NOTES_MAX_NEW_TOKENS,
                # Fenced like every other per-session pass (and unlike the revisit
                # recollection, which is unfenced on purpose): a first reading of this
                # conversation must be formed from what she knew AT the time, or a later
                # chat's conclusions leak in and the recurrence count — the portrait's
                # whole maturity signal — stops meaning "independently noticed again".
                before_session=filename,
                rag_include_chat=False,   # the transcript IS the content
                # Her PRIOR readings of this person are deliberately withheld. Shown them,
                # she restates them — and recurrence across sessions, the one signal the
                # portrait ranks on, would then measure what the prompt handed her rather
                # than what she independently noticed twice. This is the same circular
                # self-vote the persona digest had to discount after the fact
                # (`reflection_digest._TENURE_DECAY`); here it can simply be prevented.
                rag_include_impressions=False,
                on_chunk=_on_chunk,
            )
            if stats is not None:
                stats.record_phase("user_notes", time.monotonic() - t0,
                                   tokens=_count_tokens(tokenizer, raw or ""))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="user_notes", session=filename,
                       message=f"User-notes generation error (skipped): {e}")
            return 0

        parsed = parse_impressions(raw or "")
        # Transcript-echo guard. The framing above makes the copy far less likely; it does
        # not make it impossible, and the cost of one getting through is not a wasted line
        # — an impression is folded into a standing portrait and injected on every turn
        # with that person, so a copied sentence comes back later as something Ava
        # *understands* about them. Deterministic, so it holds whatever the sampler does.
        echo_index = build_echo_index(session)
        echoed: list[str] = []
        kept: list[tuple] = []
        for line, about in parsed:
            if is_transcript_echo(line, echo_index):
                echoed.append(line)
            else:
                kept.append((line, about))
        impressions = kept[:_USER_NOTES_CAP]
        if not impressions and echoed:
            # Distinct from both "nothing to say" and a truncated generation: the pass
            # produced output and every line of it was lifted from the conversation. Worth
            # its own warning — it is the failure mode this guard exists for, and seeing it
            # recur is what would say the framing needs more work.
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="user_notes", session=filename, person=person,
                       text=_pass_output_debug(raw, f"impressions=0 echoed={len(echoed)}",
                                               generate_fn=generate_fn),
                       message=(f"User notes: all {len(echoed)} line(s) copied the "
                                f"transcript (dropped) — no reading of {person} formed"))
            return 0
        if not impressions:
            # Two very different outcomes reach here, and reporting them as one was a
            # defect: "she read the conversation and it revealed nothing about him" is
            # ordinary and explicitly permitted by the prompt, while "the generation ran
            # out of budget inside its <think> and there was no body to parse" is a failed
            # pass. `_split_think` yields an empty body for an unterminated block, so the
            # second silently wore the first's message — and since impressions are what a
            # portrait folds, a systematically truncating pass would read as a person who
            # simply never reveals anything. Discriminate on the same two signals the
            # anchor warning carries: a hit token cap, or nothing left after the strip.
            body = _strip_think_block(raw or "")
            truncated = getattr(generate_fn, "last_truncated", None)
            if truncated or not body:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="user_notes", session=filename, person=person,
                           text=_pass_output_debug(raw, "impressions=0",
                                                   truncated=truncated,
                                                   generate_fn=generate_fn),
                           message=(f"User-notes output unusable (skipped) — no reading "
                                    f"of {person} was formed"))
                return 0
            # Deciding that a conversation revealed nothing about someone IS the pass's
            # work, and all of it happened in the <think> — the body is empty by
            # definition. So the RAW generation rides the event, exactly as the revision
            # pass does with its own. Without it the activity journal carries the verdict
            # with none of the reasoning: `phase_progress` deltas are never mirrored
            # (flood), so a background run's only trace was the "nothing new" line itself.
            # The Sleep tab does not double-print — its `phase_done` handler skips `text`
            # when the pass already streamed, which is the same mechanism revision relies on.
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="user_notes", session=filename, session_index=sess_idx,
                       person=person, count=0, text=raw or "",
                       message=f"User notes: nothing new about {person}")
            return 0

        if dry_run:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="user_notes", session=filename, session_index=sess_idx,
                       person=person, count=len(impressions),
                       report={"impressions": [c for c, _a in impressions],
                               "echoed": echoed},
                       message=(f"User notes (dry run — not written): "
                                f"{len(impressions)} about {person}"
                                + (f" ({len(echoed)} copied line(s) dropped)"
                                   if echoed else "")))
            return 0

        try:
            counts = self._writer.write_impressions(
                impressions=impressions, source_session=filename,
                source_user=person, default_about=person, run_id=run_id,
            )
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="user_notes", session=filename,
                       message=f"User-notes write error (skipped): {e}")
            return 0

        n = int(counts.get("rag", 0))
        # `text` (the raw pass, thinking included) over `report` for the same reason as
        # the zero case: what she wrote is only half of what an operator is watching for —
        # the other half is why she settled on these readings and not others. The parsed
        # impressions stay on `report` as the canonical record; the activity mirror simply
        # prefers `text` when both are present, and the raw ends with those same lines.
        dropped = f" ({len(echoed)} copied line(s) dropped)" if echoed else ""
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="user_notes", session=filename, session_index=sess_idx,
                   person=person, count=n, text=raw or "",
                   report={"impressions": [c for c, _a in impressions],
                           "echoed": echoed},
                   message=(f"User notes: {n} impression(s) of {person}{dropped} — "
                            + "; ".join(c[:60] for c, _a in impressions[:3])))
        return n

    def _run_self_notes_pass_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable] = None,
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> int:
        """Read this conversation back as a READER and record what it shows about her.

        The outside-view counterpart of the two readings that already exist. The revision
        pass forms ``[persona]`` from her ``<think>`` beside her reply — introspective, "the
        disposition I endorse". The user-notes pass forms ``[impression]`` about the other
        person. This one asks what her own replies show to someone who has the transcript
        and nothing else, and writes ``[self_impression]`` records that
        ``core.self_portrait`` folds into ``users/_self.json``.

        The outside view is enforced by the BUILDER, not by the prompt:
        ``build_session_reading_content`` renders the session with no CoT at all, so the
        pass structurally cannot see what she was thinking — which is exactly the material
        a reader has. That is also why this is not a section of the user-notes prompt: the
        question, the discipline ("notice, do not diagnose"), and the failure mode differ.

        Runs on **every** session, including ``interlocutor: "ai"`` transcripts that the
        user-notes pass skips: an encounter or a gossip exchange has nobody to portray, but
        it is still a record of how she comes across — arguably the cleanest, since no
        human's reaction is being modelled. Runs on revisits too; recurrence counts
        distinct ``source_session``, so re-reading a conversation cannot vote twice.

        RAG-only and never trained. Best-effort throughout. Returns the number written.
        """
        run_id = config.run_id
        from core.reflection_source import (
            build_session_reading_content, build_echo_index, is_transcript_echo)
        from core.reflection_writer import parse_impressions
        from core.self_portrait import SELF_KIND

        prompt = self._load_self_notes_prompt()
        if not prompt:
            return 0
        if not (session.get("exchanges") or []):
            return 0

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="self_notes", session=filename, session_index=sess_idx,
                   message=f"Self notes: how {filename} reads from outside")

        try:
            content = build_session_reading_content(
                session, context_length, closing=_SELF_NOTES_CLOSING)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="self_notes", session=filename,
                       message=f"Self-notes content error (skipped): {e}")
            return 0
        if not content:
            return 0

        def _on_chunk(delta: str, _sess=filename) -> None:
            self._emit(store, send_event_fn, run_id, "phase_progress",
                       phase="self_notes", session=_sess, text=delta)

        try:
            t0 = time.monotonic()
            raw = generate_fn(
                content, prompt,
                temperature=temperature, top_p=top_p,
                max_new_tokens_setting=_SELF_NOTES_MAX_NEW_TOKENS,
                # Fenced like every other per-session pass: a first reading of this
                # conversation must be formed from what she knew AT the time.
                before_session=filename,
                rag_include_chat=False,      # the transcript IS the content
                # Her prior readings of herself are withheld for the reason the user-notes
                # pass withholds its own, and one more: this pass's entire premise is
                # reading the words WITHOUT the interpretation she has already put on
                # them, and the settled interpretation is precisely what these records
                # carry. (The channel is off by default anyway — belt and braces, so the
                # pass stays correct if the switch is ever turned on.)
                rag_include_impressions=False,
                on_chunk=_on_chunk,
            )
            if stats is not None:
                stats.record_phase("self_notes", time.monotonic() - t0,
                                   tokens=_count_tokens(tokenizer, raw or ""))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="self_notes", session=filename,
                       message=f"Self-notes generation error (skipped): {e}")
            return 0

        # Reuses the `[impression]` line tag (one parser, one convention); the `(about:)`
        # marker it may also return is DROPPED here — a self-impression has exactly one
        # possible subject, and honouring a model-supplied subject would let this pass
        # write records about a third party into a store nothing attributes.
        parsed = [c for c, _about in parse_impressions(raw or "")]
        echo_index = build_echo_index(session)
        echoed = [l for l in parsed if is_transcript_echo(l, echo_index)]
        observations = [l for l in parsed if l not in echoed][:_SELF_NOTES_CAP]

        if not observations and echoed:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="self_notes", session=filename,
                       text=_pass_output_debug(
                           raw, f"observations=0 echoed={len(echoed)}",
                           generate_fn=generate_fn),
                       message=(f"Self notes: all {len(echoed)} line(s) copied the "
                                f"transcript (dropped) — no reading formed"))
            return 0
        if not observations:
            # Same discrimination the user-notes pass makes, for the same reason: "this
            # conversation shows nothing about me" is permitted by the prompt and ordinary,
            # while a generation cut inside its <think> is a failed pass — and an empty
            # body looks identical from the outside.
            body = _strip_think_block(raw or "")
            truncated = getattr(generate_fn, "last_truncated", None)
            if truncated or not body:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="self_notes", session=filename,
                           text=_pass_output_debug(raw, "observations=0",
                                                   truncated=truncated,
                                                   generate_fn=generate_fn),
                           message="Self-notes output unusable (skipped)")
                return 0
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="self_notes", session=filename, session_index=sess_idx,
                       count=0, text=raw or "",
                       message="Self notes: nothing this conversation shows")
            return 0

        if dry_run:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="self_notes", session=filename, session_index=sess_idx,
                       count=len(observations),
                       report={"observations": observations, "echoed": echoed},
                       message=(f"Self notes (dry run — not written): "
                                f"{len(observations)}"))
            return 0

        try:
            # No `about`, no `source_user`: the kind names its own subject, and attributing
            # it would derive `hearsay` from subject != speaker (see self_portrait).
            counts = self._writer.write_impressions(
                impressions=observations, source_session=filename,
                run_id=run_id, kind=SELF_KIND,
            )
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="self_notes", session=filename,
                       message=f"Self-notes write error (skipped): {e}")
            return 0

        n = int(counts.get("rag", 0))
        dropped = f" ({len(echoed)} copied line(s) dropped)" if echoed else ""
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="self_notes", session=filename, session_index=sess_idx,
                   count=n, text=raw or "",
                   report={"observations": observations, "echoed": echoed},
                   message=(f"Self notes: {n} observation(s){dropped} — "
                            + "; ".join(c[:60] for c in observations[:3])))
        return n

    def _run_chat_facts_pass_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable] = None,
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> int:
        """Write this conversation's fact-extraction PROTOCOL to its ``.facts.json``.

        The enumerated literal layer under the three readings that already exist: the
        consolidation pass distils what is worth remembering (curated, weights-bound,
        deduped and evicted across chats), the summary pass writes narrative prose, the
        notes passes form readings of a person. This one records what the conversation
        established, exhaustively and without judgement — including the small biographical
        detail every other pass is right to discard.

        **It writes NOTHING to the live stores.** No ``rag_memory.jsonl`` insert, no
        weights line, no ledger anchor, no RAG channel. ``rag_memory`` stays authoritative
        for what Ava believes and recalls; this file is an immutable per-chat source, whose
        consumer is offline (a knowledge-graph build). That separation is also what makes
        the volume safe: the injected reflection block has three slots, and multiplying
        extraction into it would crowd out everything else — so nothing injects it.

        Runs after revision so a failure here cannot touch the trainable target, and on
        every session including ``interlocutor: "ai"`` transcripts (an encounter still
        establishes facts). Best-effort throughout. Returns the number of facts written.
        """
        run_id = config.run_id
        from core.reflection_source import build_session_reading_content
        from core import chat_facts as chat_facts_mod
        from core.chat_facts import parse_facts, class_counts

        prompt = self._load_chat_facts_prompt()
        if not prompt:
            return 0
        if not (session.get("exchanges") or []):
            return 0

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="chat_facts", session=filename, session_index=sess_idx,
                   message=f"Chat facts: protocol of {filename}")

        # The canonical (about:) spellings, stated in-context: the prompt asks for a NAME
        # without saying which, and the pass was observed picking a different one per
        # generation ("artemyvo" / "Artemy" / "Артемий" for one person). Rides the
        # closing, not `ava_initiated_note` — that slot only reaches reversed sessions.
        names_note = chat_facts_mod.participants_note(session)
        closing = (f"{names_note}\n\n{_CHAT_FACTS_CLOSING}" if names_note
                   else _CHAT_FACTS_CLOSING)
        try:
            content = build_session_reading_content(
                session, context_length, closing=closing,
                # Not the default reversed-session note: that one is written for revision
                # and reads, to a pass recording what was said, as "skip your own turns".
                # See core.chat_facts.AVA_INITIATED_NOTE.
                ava_initiated_note=chat_facts_mod.AVA_INITIATED_NOTE)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="chat_facts", session=filename,
                       message=f"Chat-facts content error (skipped): {e}")
            return 0
        if not content:
            return 0

        def _on_chunk(delta: str, _sess=filename) -> None:
            self._emit(store, send_event_fn, run_id, "phase_progress",
                       phase="chat_facts", session=_sess, text=delta)

        try:
            t0 = time.monotonic()
            raw = generate_fn(
                content, prompt,
                temperature=temperature, top_p=top_p,
                max_new_tokens_setting=_CHAT_FACTS_MAX_NEW_TOKENS,
                # Fenced like every other per-session pass: the protocol of a conversation
                # is what THAT conversation established, so nothing learned later may leak
                # in and be recorded as having been said here.
                before_session=filename,
                rag_include_chat=False,      # the transcript IS the content
                # BOTH guards off for this pass. Its output is a fixed-template list —
                # every line `[fact] (about: NAME) (class: CLASS) …` — and that prefix
                # tokenizes past the verbatim guard's 12-token window, so the window
                # sits wholly inside it and is identical on every line sharing an
                # (about, class) pair: the guard fires on the 4th consecutive such fact
                # and returns a list cut mid-prefix (a single-topic chat cannot get past
                # 3 facts). The diversity guard (`_DegenStop`) was believed immune and
                # proven not to be on the TIL twin (2026-08-18, live): a run of
                # near-identical template lines craters its rolling distinct-token ratio
                # exactly as a real collapse does, whether emitted in the answer or
                # DRAFTED inside the think. No threshold separates the legitimate shape
                # from degeneration, so both are off and the token cap bounds a genuine
                # runaway. Mirrored by `core.modules._CHAT_FACTS`
                # (stop_on_repeat/degen_stop), which its self-test asserts still agrees.
                stop_on_repeat=False,
                degen_stop=False,
                on_chunk=_on_chunk,
            )
            if stats is not None:
                stats.record_phase("chat_facts", time.monotonic() - t0,
                                   tokens=_count_tokens(tokenizer, raw or ""))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="chat_facts", session=filename,
                       message=f"Chat-facts generation error (skipped): {e}")
            return 0

        truncated = bool(getattr(generate_fn, "last_truncated", None))
        # Cut inside an unclosed reasoning channel ⇒ there is no answer region, and a
        # gemma-4 channel truncated before its close normalizes to UNTAGGED prose — so
        # any [fact]-shaped line in the text is the pass's own drafting, which
        # `parse_facts` (answer-region-only) cannot tell apart without this flag. Refuse
        # the parse outright rather than record deliberation as protocol.
        from core.reasoning_text import truncated_before_answer
        if truncated_before_answer(raw or "", truncated):
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="chat_facts", session=filename,
                       text=_pass_output_debug(raw, "cut before answer",
                                               truncated=truncated,
                                               generate_fn=generate_fn),
                       message="Chat-facts generation cut inside its reasoning (skipped)")
            return 0
        # Session-aware fold: a label naming a recorded participant resolves to that
        # participant's canonical key whatever spelling the pass reached for (the
        # parse-side backstop behind the note above; see chat_facts.make_subject_fn).
        facts = parse_facts(raw or "", truncated=truncated,
                            subject_fn=chat_facts_mod.make_subject_fn(session))

        if not facts:
            # Same discrimination the notes passes make: "this conversation established
            # nothing factual" is permitted by the prompt and ordinary, while a generation
            # cut inside its <think> is a failed pass — and both look like an empty list.
            body = _strip_think_block(raw or "")
            if truncated or not body:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="chat_facts", session=filename,
                           text=_pass_output_debug(raw, "facts=0", truncated=truncated,
                                                   generate_fn=generate_fn),
                           message="Chat-facts output unusable (skipped)")
                return 0
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="chat_facts", session=filename, session_index=sess_idx,
                       count=0, text=raw or "",
                       message="Chat facts: nothing factual established")
            return 0

        counts = class_counts(facts)
        if dry_run:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="chat_facts", session=filename, session_index=sess_idx,
                       count=len(facts), report={"facts": facts, "classes": counts},
                       message=(f"Chat facts (dry run — not written): {len(facts)}"))
            return 0

        try:
            written = self._session_sidecar().write_facts(
                source_session=filename, facts=facts, run_id=run_id,
                source_user=self._session_person(session) or "",
            )
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="chat_facts", session=filename,
                       message=f"Chat-facts write error (skipped): {e}")
            return 0
        if not written:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="chat_facts", session=filename,
                       message="Chat-facts record refused (live session or bad path)")
            return 0

        shape = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="chat_facts", session=filename, session_index=sess_idx,
                   count=len(facts), text=raw or "",
                   report={"facts": facts, "classes": counts},
                   message=(f"Chat facts: {len(facts)} recorded ({shape}) — "
                            + "; ".join(f["text"][:60] for f in facts[:3])))
        return len(facts)

    def _run_recollection_pass_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable],
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> int:
        """Write what Ava NOW makes of a re-read conversation — the revisit's own memory.

        **Revisit-only.** A first-time reflection has nothing to look back on; this pass
        exists because a revisit re-reads a chat whose verbatim expired days ago and whose
        gist has settled at its floor. It produces one ``[recollection]`` per conversation
        — the gist's fresh sibling: same grain, but dated by the READING rather than by the
        conversation, so re-deriving an old chat yields memory that is retrievable at full
        weight today (see ``ReflectionWriter.write_recollection`` and
        ``training.decay.recollection_rag_weight_hours``).

        **RAG-only by construction**: the pass writes one op-log insert and nothing else —
        no weights line, no ledger anchor, no sidecar field, no training row. The revisit's
        re-derived target already reaches the build through the sidecar; this is the
        retrieval half that was missing, not a second path into the weights.

        Retrieval for the pass itself is deliberately unlike every other pass here. It runs
        **unfenced** (``before_session=""``) because the whole point is to read an old
        conversation with what she knows *now* — the replay-faithful cutoff the other
        passes use would hand her a time machine and make the output indistinguishable
        from the original consolidation. It drops the past-chat block (the transcript is
        already the content) and drops the recollection channel itself, so her previous
        reading of this same chat cannot anchor the new one into restating it.

        One generation per session. Best-effort throughout: a failure here must never
        disturb the revisit that surrounds it. Returns 1 when a recollection was written.
        """
        run_id = config.run_id
        from core.reflection_source import (
            build_revision_jobs, build_session_reading_content,
            build_echo_index, is_transcript_echo)
        from core.reflection_writer import parse_recollection

        prompt = self._load_recollection_prompt()
        if not prompt:
            return 0

        try:
            jobs = build_revision_jobs(session)
        except Exception:
            return 0
        if not jobs:
            return 0

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="recollection", session=filename, session_index=sess_idx,
                   message=f"Recollection: re-reading {filename} as it stands now")

        # The whole conversation, neutrally framed — the same builder the user-notes pass
        # reads through, and here for the same reason: `build_revision_content` sets up a
        # judgement of ONE exchange (tail exchange isolated under "The exchange you are
        # judging:", its <think> alone, an optional COUNTER block), which is the wrong
        # shape for a pass asking what it makes of the conversation as a whole — and puts
        # the reply last, where it is the easiest thing to hand back.
        try:
            content = build_session_reading_content(
                session, context_length, closing=_RECOLLECTION_CLOSING)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="recollection", session=filename,
                       message=f"Recollection content error (skipped): {e}")
            return 0
        if not content:
            return 0

        def _on_chunk(delta: str, _sess=filename) -> None:
            self._emit(store, send_event_fn, run_id, "phase_progress",
                       phase="recollection", session=_sess, text=delta)

        try:
            t0 = time.monotonic()
            raw = generate_fn(
                content, prompt,
                temperature=temperature, top_p=top_p,
                max_new_tokens_setting=_RECOLLECTION_MAX_NEW_TOKENS,
                before_session="",          # unfenced — see docstring
                rag_include_chat=False,
                rag_include_recollections=False,
                on_chunk=_on_chunk,
            )
            if stats is not None:
                stats.record_phase("recollection", time.monotonic() - t0,
                                   tokens=_count_tokens(tokenizer, raw or ""))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="recollection", session=filename,
                       message=f"Recollection generation error (skipped): {e}")
            return 0

        content_text, trigger = parse_recollection(raw or "")
        if not content_text:
            # The fourth pass with this shape (anchor / user-notes / portrait being the
            # others): thinking on, a labelled body, and an empty parse that says nothing
            # about which failure it was. Same dump for the same reason.
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="recollection", session=filename,
                       text=_pass_output_debug(
                           raw, "recollection=∅ trigger={}".format(
                               repr(trigger) if (trigger or "").strip() else "∅"),
                           generate_fn=generate_fn),
                       message="Recollection unparseable or empty (skipped)")
            return 0

        # Transcript-echo guard, as on the user-notes pass. A single-valued pass, so an
        # echo has no partial outcome: what she "now makes of" the conversation cannot be
        # a sentence out of it, and writing one would supersede the previous genuine
        # reading with a copy — the one live recollection per chat is a slot, not a pile.
        # The TRIGGER is deliberately not checked: it names a future situation in the
        # conversation's own vocabulary by design, and it is only an embedding key.
        if is_transcript_echo(content_text, build_echo_index(session)):
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="recollection", session=filename,
                       text=_pass_output_debug(raw, "recollection=echo",
                                               generate_fn=generate_fn),
                       message=("Recollection copied the transcript (dropped) — "
                                "nothing new was made of it"))
            return 0

        if dry_run:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="recollection", session=filename,
                       text=content_text, trigger=trigger,
                       message=f"Recollection (dry run — not written): {content_text[:70]}")
            return 0

        # Supersede this chat's previous reading so the corpus holds exactly one live
        # recollection per conversation — without this, a rotating revisit schedule would
        # accumulate one paraphrase per pass, all competing for the same three slots.
        prior_key = ""
        try:
            mem = ReflectionMemory(self._memory_dir,
                                   fallback_memory_dir=self._fallback_memory_dir)
            prior = mem.recollection_for(filename)
            prior_key = (prior or {}).get("key", "") or ""
        except Exception:
            prior_key = ""

        # Tags come from the anchors this run just wrote (the pass runs after the anchor
        # pass), giving the recollection the same lexical surface as the exchanges it
        # summarizes. Stored, not yet used for admission — retrieval is dense on TRIGGER.
        tags: list = []
        try:
            doc = self._session_sidecar().load(filename) or {}
            anchors = doc.get("anchors") if isinstance(doc, dict) else None
            if isinstance(anchors, dict):
                seen: set = set()
                for rec in anchors.values():
                    for t in (rec or {}).get("tags") or []:
                        t = str(t).strip()
                        if t and t.lower() not in seen:
                            seen.add(t.lower())
                            tags.append(t)
        except Exception:
            tags = []

        try:
            counts = self._writer.write_recollection(
                content=content_text, trigger=trigger, source_session=filename,
                tags=tags[:_RECOLLECTION_TAG_CAP],
                source_exchanges=[int(j.get("index", -1)) for j in jobs
                                  if int(j.get("index", -1)) >= 0],
                run_id=run_id, supersedes=prior_key,
            )
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="recollection", session=filename,
                       message=f"Recollection write error (skipped): {e}")
            return 0

        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="recollection", session=filename, session_index=sess_idx,
                   text=content_text, trigger=trigger,
                   superseded=bool(counts.get("evict")),
                   message=(f"Recollection written"
                            f"{' (superseded the previous one)' if counts.get('evict') else ''}"
                            f": {content_text[:70]}"))
        return int(counts.get("rag", 0))

    def _run_revision_for_session(
        self,
        config: ReflectionRunConfig,
        filename: str,
        session: dict,
        sess_idx: int,
        total_sessions: int,
        *,
        generate_fn: Callable,
        store: ReflectionRunStore,
        context_length: int,
        tokenizer,
        send_event_fn: Optional[Callable],
        branching_enabled: bool,
        branch_generate_fn: Optional[Callable],
        branch_chooser_content_fn: Optional[Callable],
        similarity_fn: Optional[Callable] = None,
        persona_context_fn: Optional[Callable] = None,
        persona_keys_fn: Optional[Callable] = None,
        persona_digest: Optional[dict] = None,
        judge_jobs: Optional[list] = None,
        fact_candidates: Optional[list] = None,
        stats: Optional[RunStats] = None,
        dry_run: bool = False,
    ) -> tuple[int, int, dict]:
        """Run the revision pass over every revisable exchange for a single session. Returns (passes, skipped, agg).

        *dry_run*: generate the judgement, clean IDEAL re-answer, and (when enabled)
        the branch experiment,
        emit the streamed text + structured ``report`` on each ``phase_done`` event, but
        write NOTHING — no memory/persona op-log, no sidecar, no ledger anchor, no
        fact-placement/judge collection, and no logged prompt-mutation delta. Backs the
        "Dry Sleep" preview so an operator can see a full reflection+revision+branch pass
        without mutating any server data."""
        run_id = config.run_id

        from core.reflection_prompts import load_reflection_prompts
        from core.reflection_source import (
            build_revision_jobs, build_revision_content, build_revision_rag_query,
            REVISION_CLOSING_NOTE,
        )
        prompts = load_reflection_prompts()
        revision_prompt = config.overrides.revision_prompt or prompts.revision_prompt
        branch_prompt = config.overrides.branch_prompt or prompts.branch_prompt

        sampling = config.overrides.revision_sampling
        temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P
        max_new_tokens = (sampling.max_new_tokens_setting
                          if sampling else _DEFAULT_MAX_NEW_TOKENS)

        branch_sampling = config.overrides.branch_sampling
        br_temperature = branch_sampling.temperature if branch_sampling else _DEFAULT_TEMPERATURE
        br_top_p = branch_sampling.top_p if branch_sampling else _DEFAULT_TOP_P

        # Phase two (logged-only): *persona_digest* and *judge_jobs* are owned by
        # execute_run (digest loaded once per run; jobs collected across all sessions and
        # judged after, batched, on the clean base). The branch pass appends a job here and
        # logs only the cheap embedding channel inline.

        total_passes = 0
        skipped_passes = 0
        # Running aggregate of what revision routed: trainable pairs by target_source
        # and persona self-statements — for the run summary's report block.
        agg = {"pairs": 0, "persona": 0,
               "target_source": {"original": 0, "revised": 0,
                                 "revised_missing_ideal": 0}}
        store.update_status(run_id, phase="revision", session_total=total_sessions,
                            session_index=sess_idx, exchange_index=0, exchange_total=0)

        jobs = build_revision_jobs(session)
        # Human-validated (manually regenerated) exchanges are preserved: drop them from the
        # work list so re-reflection — including a revisit, which bypasses the session-level
        # reflect-once freeze — never re-derives them from the original (corrupt) transcript
        # and overwrites the operator's reviewed target. Their sidecar record is left
        # untouched (no write), so the lock + hand-authored target survive the run.
        locked = self._locked_exchanges(filename)
        if locked:
            preserved = [j["index"] for j in jobs if j["index"] in locked]
            jobs = [j for j in jobs if j["index"] not in locked]
            if preserved:
                self._emit(store, send_event_fn, run_id, "exchanges_locked",
                           session=filename, session_index=sess_idx, phase="revision",
                           indices=preserved,
                           message=(f"Preserving {len(preserved)} human-validated "
                                    f"exchange(s) {preserved} — skipping re-reflection."))
        # Banned exchanges leave the work list for a different reason: not "this target is
        # already right" but "this exchange will never have a target" — so revising it (and
        # branching from it) would be generation spent on a row the build discards.
        banned = self._banned_exchanges(filename)
        if banned:
            dropped = [j["index"] for j in jobs if j["index"] in banned]
            jobs = [j for j in jobs if j["index"] not in banned]
            if dropped:
                self._emit(store, send_event_fn, run_id, "exchanges_banned",
                           session=filename, session_index=sess_idx, phase="revision",
                           indices=dropped,
                           message=(f"Skipping {len(dropped)} exchange(s) {dropped} banned "
                                    f"from training."))
        exchange_total = len(jobs)
        store.update_status(run_id, exchange_total=exchange_total, exchange_index=0)

        self._emit(store, send_event_fn, run_id, "session_started",
                   session=filename, session_index=sess_idx,
                   session_total=total_sessions, phase="revision",
                   message=(
                       f"Revision: session {sess_idx}/{total_sessions}"
                       f" ({exchange_total} exchanges)"
                   ))

        if not jobs:
            return 0, 0, agg

        for ex_idx, job in enumerate(jobs, start=1):
            if store.is_stop_requested(run_id):
                break

            store.update_status(run_id, exchange_index=ex_idx)
            self._emit(
                store, send_event_fn, run_id, "phase_started",
                phase="revision", session=filename,
                session_index=sess_idx, exchange_index=ex_idx,
                exchange_total=exchange_total,
                message=(
                    f"Revision: session {sess_idx}/{total_sessions},"
                    f" exchange {ex_idx}/{exchange_total}"
                ),
            )

            content = build_revision_content(job, session, context_length,
                                             closing_note=REVISION_CLOSING_NOTE)
            # Corrupt-reply handling: the operator flagged this exchange's STORED reply
            # corrupt, so force a revise judgement (and drop it below if the clean
            # re-answer fails). Threaded through the initial + retry judgements and the
            # language guard so the correction demand persists. (A corrupt CoT is blanked on
            # the job's exchange view — build_revision_jobs — so the pass sees it as missing.)
            corrupt_response = bool(job.get("corrupt_response"))
            corrupt_cot = bool(job.get("corrupt_cot"))
            eff_revision_prompt = (
                revision_prompt + _CORRUPT_REPLY_NUDGE if corrupt_response else revision_prompt
            )
            # Retrieve RAG against the judged exchange, not the full content: the
            # subject sits at the tail of `content`, which the embedder truncates
            # away (see build_revision_rag_query). Reflection-memory only — the
            # conversation context is already in `content`, so past-chat excerpts
            # (the bulk of the injected tokens) are redundant here.
            rag_query = build_revision_rag_query(job)

            def _on_chunk(delta: str, _sess=filename, _ex=ex_idx, _total=exchange_total) -> None:
                self._emit(store, send_event_fn, run_id, "phase_progress",
                           phase="revision", session=_sess,
                           exchange_index=_ex, exchange_total=_total, text=delta)

            _vram_reset_peak()
            gen_t0 = time.monotonic()
            try:
                response = generate_fn(
                    content, eff_revision_prompt,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens_setting=max_new_tokens,
                    before_session=filename,
                    rag_query=rag_query, rag_include_chat=False,
                    # This is the pass that MINTS [persona] statements, so the live
                    # [persona] set is fenced out of its retrieval: shown a prior
                    # self-statement beside the judged exchange, she restates it and
                    # the restatement counts as an independent distinct-session vote —
                    # the circular self-vote _TENURE_DECAY discounts after the fact,
                    # prevented at the source instead (the same rule the user-notes
                    # pass applies to impressions). Facts/asks/impressions still
                    # inject; the IDEAL re-answer keeps its deliberate persona
                    # conditioning via persona_context_fn.
                    rag_include_persona=False,
                    on_chunk=_on_chunk,
                )
            except Exception as e:
                skipped_passes += 1
                if stats is not None:
                    stats.note_discard("revision_gen_error")
                    store.update_status(run_id, stats=stats.to_status())
                self._emit(store, send_event_fn, run_id, "pass_error",
                           session=filename, exchange_index=ex_idx,
                           message=f"Revision error (skipped): {e}")
                continue
            if stats is not None:
                stats.record_phase("revision", time.monotonic() - gen_t0,
                                   tokens=_count_tokens(tokenizer, response or ""))
                stats.observe_vram(_vram_peak_gb(), phase="revision",
                                   context_tokens=_count_tokens(tokenizer, content))

            # Parse judgement only. A model/override may still emit a legacy inline IDEAL,
            # but it is deliberately ignored: a training target may come only from the
            # separate clean pre-answer generation below.
            verdict, why, persona = parse_revision_judgement(response)
            ideal: Optional[str] = None

            # An unparseable judgement gets one format-only retry. This retry remains in
            # the reflection lane; it never authors dialogue CoT or an IDEAL reply.
            unparseable = verdict is None
            if unparseable:
                truncated = getattr(generate_fn, "last_truncated", None)
                looped = getattr(generate_fn, "last_loop", None)
                self._emit(
                    store, send_event_fn, run_id, "pass_warning",
                    phase="revision", session=filename,
                    exchange_index=ex_idx, exchange_total=exchange_total,
                    message=(f"Revision verdict unparseable (truncated={truncated}, "
                             f"looped={looped}); "
                             "retrying once with higher temperature + format nudge."),
                )
                retry_t0 = time.monotonic()
                try:
                    retry_response = generate_fn(
                        content, eff_revision_prompt + _REVISION_RETRY_NUDGE,
                        temperature=min(1.0, temperature + _REVISION_RETRY_TEMP_BUMP),
                        top_p=top_p, max_new_tokens_setting=max_new_tokens,
                        before_session=filename, rag_query=rag_query,
                        rag_include_chat=False,
                        rag_include_persona=False,   # same fence as the main judgement
                        on_chunk=_on_chunk,
                    )
                except Exception as e:
                    retry_response = None
                    self._emit(store, send_event_fn, run_id, "pass_warning",
                               phase="revision", session=filename, exchange_index=ex_idx,
                               message=f"Revision retry errored: {e}")
                # The retry is a full revision generation — fold its wall time +
                # tokens into the revision slot so the phase report doesn't
                # under-count exchanges that needed a second pass.
                if stats is not None:
                    stats.record_phase(
                        "revision", time.monotonic() - retry_t0,
                        tokens=_count_tokens(tokenizer, retry_response or ""))
                retry_ok = False
                if retry_response is not None:
                    rv2, why2, persona2 = parse_revision_judgement(retry_response)
                    retry_ok = rv2 in ("keep", "revise")
                    if retry_ok:
                        response = retry_response
                        verdict, why, persona = rv2, why2, persona2
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="revision", session=filename, exchange_index=ex_idx,
                                   message=f"Revision retry produced a usable verdict ({rv2}).")
                    else:
                        self._emit(store, send_event_fn, run_id, "pass_warning",
                                   phase="revision", session=filename, exchange_index=ex_idx,
                                   message="Revision retry still unusable; this exchange "
                                           "will not produce a training target.")
                if stats is not None:
                    stats.note_retry(recovered=retry_ok)

            # Decide whether the OLD reply drifted in language. This guard may re-run the
            # judgement to distinguish an accidental switch from a requested translation,
            # but it never generates the replacement reply.
            judgement_unusable = verdict not in ("keep", "revise")
            if judgement_unusable:
                # Route through the honest missing-target outcome below. Do not invent a
                # revise decision or independently re-answer an exchange the judge failed
                # to classify.
                verdict = "revise"
                persona = []
                lang_drift = False
            else:
                response, verdict, why, persona, lang_drift = self._language_decision_guard(
                    response=response, verdict=verdict, why=why, persona=persona,
                    job=job, content=content, revision_prompt=eff_revision_prompt,
                    generate_fn=generate_fn, temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens, filename=filename, rag_query=rag_query,
                    on_chunk=_on_chunk, ex_idx=ex_idx, exchange_total=exchange_total,
                    store=store, send_event_fn=send_event_fn, run_id=run_id,
                    tokenizer=tokenizer, stats=stats,
                )

            # A corrupt stored reply is never keepable even if the judgement ignored the
            # nudge. It shares the same clean re-answer path as every other revise decision.
            if corrupt_response:
                verdict = "revise"

            # Generate the revised reply as a NORMAL dialogue completion from the stored
            # pre-answer conversation. The old assistant CoT/reply, feedback, WHY, reflection
            # prompt and reflection RAG are absent from this call by construction — EXCEPT
            # Ava's own relevant persona self-knowledge, retrieved persona-only and cut to
            # BEFORE this chat (so it is what she already knew at the time, not later
            # reflection), which conditions the re-derived CoT in place of the retired
            # build-time persona prepend. The exact block is persisted onto the anchor so the
            # trained system prompt reconstructs identically (parity).
            ideal_persona_context = ""
            if verdict == "revise" and not judgement_unusable:
                if persona_context_fn is not None:
                    try:
                        ideal_persona_context = persona_context_fn(
                            (job.get("exchange") or {}).get("user_prompt") or "", filename
                        ) or ""
                    except Exception:
                        ideal_persona_context = ""
                ideal = self._generate_ideal_reply(
                    job=job, session=session, generate_fn=generate_fn,
                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                    require_user_language=lang_drift,
                    persona_context=ideal_persona_context,
                    filename=filename, ex_idx=ex_idx, exchange_total=exchange_total,
                    store=store, send_event_fn=send_event_fn, run_id=run_id,
                    tokenizer=tokenizer, stats=stats,
                )

            # CoT regeneration (approach #3) — for a corrupt-CoT exchange the judge KEPT.
            # The operator flagged only the CoT corrupt, so the stored reply is trusted;
            # rather than train it answer-only (a reasoning-channel erosion gradient), we
            # re-answer to author a fresh faithful <think> and graft it onto the ORIGINAL
            # reply, gated on the re-answer reproducing ~that reply. Only meaningful on a
            # `keep` (a `revise`/corrupt_response already re-answers in full above). The
            # blanked source CoT makes this exchange CoT-less, so once the graft lands the
            # existing _skip_branch_cot_less rule trains it directly (no branch).
            recot_used = False
            if (corrupt_cot and not corrupt_response and verdict == "keep"
                    and not judgement_unusable and not lang_drift):
                graft = self._regenerate_cot_for_kept_reply(
                    job=job, session=session, generate_fn=generate_fn,
                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                    similarity_fn=similarity_fn,
                    filename=filename, ex_idx=ex_idx, exchange_total=exchange_total,
                    store=store, send_event_fn=send_event_fn, run_id=run_id,
                    tokenizer=tokenizer, stats=stats,
                )
                if graft is not None:
                    verdict, ideal, recot_used = "revise", graft, True

            # Branch generation — runs whenever the branch capability is wired in.
            # Runs after the revision verdict so the IDEAL can join the blind
            # choice set and the block attaches to the same record.
            #
            # REBUILD.md §4 CoT rule: a revisable exchange with NO original CoT whose
            # clean re-answer produced a usable (CoT-bearing) IDEAL trains it directly.
            # Skip branching it — the branches would inherit its empty CoT, and a CoT-less
            # target recurs at max LR in every from-scratch build (a permanent channel-
            # erosion gradient) for exactly the exchange where the IDEAL is already the
            # better target. Skipping the branch also skips its judge job (only collected
            # when a branch block exists) and its GPU cost.
            branch_block: Optional[dict] = None
            _cot_less = not (job["exchange"].get("assistant_cot") or "").strip()
            _skip_branch_cot_less = _cot_less and ideal_trainable_target(ideal) is not None
            if branching_enabled and verdict == "revise" and ideal is None:
                if stats is not None:
                    stats.note_branch_skipped("ideal_generation_failed")
                self._emit(store, send_event_fn, run_id, "branch_skipped",
                           session=filename, exchange_index=ex_idx,
                           message="Branch skipped: revise verdict has no clean trainable IDEAL.")
            elif branching_enabled and config.revisit:
                # Branching is exclusive to first-time reflection of new chats. A revisit
                # re-reflects an aged chat under the CURRENT weights, but branching replays
                # that chat's OLD tension points (the contested tokens sampled under the
                # weights in force when it was first generated), so its forks reflect a past
                # model state, not who Ava is now. The freshly re-derived response — the
                # revision IDEAL, authored with current weights (or the kept original on a
                # `keep` verdict) — is the better match for the current state, so train it
                # directly via resolve_revision_target and skip branching (which also skips
                # the branch's judge job + GPU cost).
                if stats is not None:
                    stats.note_branch_skipped("revisit")
                self._emit(store, send_event_fn, run_id, "branch_skipped",
                           session=filename, exchange_index=ex_idx,
                           message="Branch skipped: revisit run — training the freshly "
                                   "re-derived response directly (current weights over "
                                   "replayed old tension).")
            elif branching_enabled and lang_drift:
                # Every counterfactual fork continues the original (drifted) answer prefix,
                # so the whole blind choice set would be in the wrong language — nothing to
                # judge. Train the language-corrected IDEAL directly instead.
                if stats is not None:
                    stats.note_branch_skipped("language_drift")
                self._emit(store, send_event_fn, run_id, "branch_skipped",
                           session=filename, exchange_index=ex_idx,
                           message="Branch skipped: language drift — training the "
                                   "language-corrected IDEAL directly.")
            elif branching_enabled and _skip_branch_cot_less:
                if stats is not None:
                    stats.note_branch_skipped("cot_less_ideal")
                self._emit(store, send_event_fn, run_id, "branch_skipped",
                           session=filename, exchange_index=ex_idx,
                           message="Branch skipped: CoT-less exchange with a usable IDEAL "
                                   "— training the IDEAL directly (CoT rule).")
            elif branching_enabled and corrupt_response:
                # Every fork replays the CORRUPT reply's contested tokens (it continues the
                # corrupt answer prefix), so the whole blind choice set is corrupt — nothing
                # to judge. Train the re-derived IDEAL directly (mirrors the drift skip).
                if stats is not None:
                    stats.note_branch_skipped("corrupt_response")
                self._emit(store, send_event_fn, run_id, "branch_skipped",
                           session=filename, exchange_index=ex_idx,
                           message="Branch skipped: corrupt reply — training the "
                                   "re-derived IDEAL directly.")
            elif branching_enabled:
                branch_block = self._run_branch_for_exchange(
                    config, filename, job,
                    generate_fn=generate_fn,
                    branch_generate_fn=branch_generate_fn,
                    branch_chooser_content_fn=branch_chooser_content_fn,
                    branch_prompt=branch_prompt,
                    br_temperature=br_temperature,
                    br_top_p=br_top_p,
                    ideal_text=ideal or "",
                    exchange_label=ex_idx,
                    exchange_total=exchange_total,
                    tokenizer=tokenizer,
                    store=store,
                    send_event_fn=send_event_fn,
                    persona_digest=persona_digest,
                    stats=stats,
                )

            total_passes += 1
            ex = job["exchange"]
            if stats is not None:
                stats.note_verdict(verdict)

            # Counter-evidence (persuasion channel): did the user's NEXT turn push back
            # against a stance this reply expressed? Parsed from the final completion, and
            # only meaningful when a next turn actually exists (no reaction ⇒ no counter,
            # regardless of what the model emitted). Consumed in the persist block below.
            counter_flag = bool(job.get("next_user")) and revision_counter(response)

            # Resolve the trainable target once — identical to what write_revision
            # computes internally — so the report and the durable write agree on
            # the pair this exchange produces. Every trained target carries a faithful
            # CoT: keep/original keep their own, a branch win reattaches the original
            # <think> it continued, and a revise/IDEAL win carries the CoT authored by
            # the clean dialogue generation. Missing/malformed re-answers resolve to the
            # not-persisted `revised_missing_ideal` skip.
            target, target_source = resolve_revision_target(
                verdict, ideal,
                (ex.get("assistant_response") or ""), branch_block,
                assistant_cot=(ex.get("assistant_cot") or ""),
            )
            target_kind, target_generation = resolved_target_provenance(
                target_source, branch_block)
            # CoT-regen graft rides the IDEAL seam but keeps the ORIGINAL reply — retag its
            # provenance so it isn't reported as a full clean re-answer (mirrors write_revision).
            if recot_used and target_source == "revised":
                target_kind, target_generation = "cot_regen", "chat_recot_v1"

            # Persist the persona block ONLY when the trained CoT is the persona-conditioned
            # IDEAL (an ideal-win). A keep (original CoT), a branch win (original <think>
            # reattached), and a cot_regen graft (fresh <think> on the original reply, not
            # persona-conditioned) all persist "" so their anchor system prompt stays bare —
            # matching what actually generated each CoT (parity).
            ideal_win = (
                target_source == "revised"
                and not recot_used
                and (branch_block or {}).get("chosen_kind") != "branch"
            )
            persist_persona_context = ideal_persona_context if ideal_win else ""

            if target_source != "revised_missing_ideal":
                agg["pairs"] += 1
            agg["target_source"][target_source] = (
                agg["target_source"].get(target_source, 0) + 1
            )
            # Chosen-target breakdown for the report: split the collapsed "revised"
            # bucket into ideal-win vs branch-win (the branch block's chosen_kind
            # disambiguates), and route the not-persisted skip to the discard tally.
            if stats is not None:
                if target_source == "revised_missing_ideal":
                    if lang_drift:
                        stats.note_discard("lang_drift_unrepaired")
                    elif corrupt_response:
                        stats.note_discard("corrupt_response_unrepaired")
                    else:
                        stats.note_discard("revised_missing_ideal")
                elif target_source == "original":
                    stats.note_chosen("original")
                elif recot_used:
                    stats.note_chosen("cot_regen")
                else:  # "revised" — resolve which source actually won
                    chosen_kind = (branch_block or {}).get("chosen_kind")
                    if chosen_kind == "branch":
                        stats.note_chosen("branch_win")
                    else:  # "ideal", or no branch (verdict revise → IDEAL)
                        stats.note_chosen("ideal_win")
            # agg["persona"] counts personas actually *written* — taken from the
            # writer summary below (the idempotency guard may drop re-emissions of an
            # already-recorded statement), not the parsed list. rev_report["persona"]
            # below keeps the parsed list: a per-exchange view of what surfaced.

            rev_report = {
                "source_session": filename,
                "exchange_index": job["index"],
                "verdict": verdict,
                "why": why,
                "target_source": target_source,
                "target_kind": target_kind,
                "target_generation": target_generation,
                "prompt": (ex.get("user_prompt") or ""),
                "speaker": job["speaker"],
                "original_response": (ex.get("assistant_response") or ""),
                "target": target,
                "persona": persona,
                "has_branch": branch_block is not None,
            }

            if target_source == "revised_missing_ideal":
                skipped_passes += 1
                persona = []
                rev_report["persona"] = []
                reason = ("revision judgement remained unparseable"
                          if judgement_unusable else
                          "language drift could not be repaired"
                          if lang_drift else
                          "corrupt reply could not be reconstructed"
                          if corrupt_response else
                          "clean IDEAL generation failed")
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="ideal_generation", session=filename,
                           exchange_index=ex_idx, exchange_total=exchange_total,
                           message=f"{reason}; no training target will be persisted.")

            if dry_run:
                # Dry run: report what revision produced but persist nothing (no
                # memory/persona op-log, sidecar, ledger, fact-placement, or judge
                # writes). Persona count comes from the parsed list rather than the
                # writer's idempotency-filtered summary.
                n_persona = len(persona or [])
                agg["persona"] += n_persona
                if stats is not None:
                    stats.note_persona(n_persona)
                if stats is not None:
                    stats.note_exchange_done()
                    store.update_status(run_id, stats=stats.to_status())
                self._emit(store, send_event_fn, run_id, "phase_done",
                           phase="revision", session=filename,
                           exchange_index=ex_idx, exchange_total=exchange_total,
                           has_branch=branch_block is not None,
                           text=response, report=rev_report,
                           message="Revision pass done (dry run — not written)")
                continue

            try:
                summary = self._writer.write_revision(
                    run_id=run_id,
                    source_session=filename,
                    exchange_index=job["index"],
                    system_prompt=(session.get("system_prompt") or ""),
                    context=job["context"],
                    user_prompt=(ex.get("user_prompt") or ""),
                    speaker=job["speaker"],
                    assistant_cot=(ex.get("assistant_cot") or ""),
                    assistant_response=(ex.get("assistant_response") or ""),
                    verdict=verdict,
                    ideal=ideal,
                    persona=persona,
                    tension=ex.get("tension"),
                    branch=branch_block,
                    # A revisit re-derives an obsolete chat's target only — persona
                    # formation is suppressed so it can't reshape who Ava is becoming.
                    suppress_persona=config.revisit,
                    # CoT-regen graft: keep the original reply, retag provenance cot_regen.
                    recot=recot_used,
                    # Persist the persona block the IDEAL was conditioned on (ideal-win only).
                    persona_context=persist_persona_context,
                )

                write_revision_sidecar(
                    self._chats_dir, summary,
                    source_session=filename,
                    exchange_index=job["index"],
                    run_id=run_id,
                    fallback_chats_dir=self._fallback_chats_dir,
                )
                register_revision_anchor(
                    self._consolidation_dir, summary, filename
                )
                # Counter-evidence (persuasion channel): on genuine next-turn pushback,
                # subtract from the live stance(s) this reply expressed. Distinct-session
                # accumulation — one push barely moves a mature trait, sustained pushback
                # across separate chats fades it. Best-effort; never derails the pass.
                if counter_flag:
                    self._write_counter_evidence(
                        job=job, filename=filename, why=why,
                        persona_keys_fn=persona_keys_fn, run_id=run_id,
                        ex_idx=ex_idx, exchange_total=exchange_total,
                        store=store, send_event_fn=send_event_fn, stats=stats,
                    )
                # Offer this exchange as a candidate host for the fact-placement judge,
                # but only when its RESOLVED target carries a faithful CoT. For an IDEAL
                # this is the clean re-answer CoT, never the old thought that produced the
                # rejected reply; branch/original targets retain their own provenance.
                target_cot = _leading_target_cot(target)
                if fact_candidates is not None and target_cot:
                    fact_candidates.append({
                        "session": filename,
                        "exchange_index": job["index"],
                        "user_prompt": (ex.get("user_prompt") or ""),
                        "assistant_cot": target_cot,
                    })
                agg["persona"] += int(summary.get("persona", 0) or 0)
                if stats is not None:
                    stats.note_persona(int(summary.get("persona", 0) or 0))

                # Collect the phase-two judge job (run later, batched, on the clean base).
                # Carries everything needed both to LOG the judge pick and — when the
                # criterion flip is enabled and the digest is mature — to RE-RESOLVE the
                # trainable target with the judge's choice and rewrite this sidecar.
                if branch_block is not None and judge_jobs is not None and persona_digest:
                    judge_jobs.append({
                        "session": filename,
                        "exchange_index": job["index"],
                        "exchange_total": exchange_total,
                        "payload": {
                            "context": job["context"], "speaker": job["speaker"],
                            "user_prompt": (ex.get("user_prompt") or ""),
                            "options": branch_block.get("candidates") or [],
                        },
                        "options": branch_block.get("candidates") or [],
                        "chosen": branch_block.get("chosen_index"),
                        "original_index": branch_block.get("original_index"),
                        # for the criterion flip (re-resolve target with the judge's pick):
                        "verdict": verdict,
                        "ideal": ideal,
                        # The persona block the IDEAL was conditioned on — persisted iff the
                        # flip lands back on the IDEAL (pick_kind "ideal"); a flip to a branch
                        # clears it (branch CoT is the original). See _maybe_override_target.
                        "ideal_persona_context": ideal_persona_context,
                        "assistant_response": (ex.get("assistant_response") or ""),
                        "assistant_cot": (ex.get("assistant_cot") or ""),
                        "user_prompt": (ex.get("user_prompt") or ""),
                        "phase_a_target_source": target_source,
                        "branch_block": branch_block,
                    })
            except Exception as e:
                skipped_passes += 1
                if stats is not None:
                    stats.note_discard("persist_error")
                    store.update_status(run_id, stats=stats.to_status())
                self._emit(store, send_event_fn, run_id, "pass_error",
                           session=filename, exchange_index=ex_idx,
                           message=f"Revision persist error (skipped): {e}")
                continue

            # One revisable exchange fully processed — advance the global ETA clock
            # and push a fresh live snapshot for the Sleep tab's stats panel.
            if stats is not None:
                stats.note_exchange_done()
                store.update_status(run_id, stats=stats.to_status())

            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="revision", session=filename,
                       exchange_index=ex_idx, exchange_total=exchange_total,
                       has_branch=branch_block is not None,
                       text=response, report=rev_report,
                       message="Revision pass done")

            # Prompt-mutation pass (LOGGED-ONLY) — only on a drift exchange (verdict
            # revise), where "would a different standing prompt have prevented this?"
            # is meaningful. Measures the drift against the persona digest ("better me"),
            # logs any proposed standing-prompt delta to its own op-log, and mutates
            # nothing. Best-effort: a failure never touches the revision result. Skipped
            # for a revisit run — it is prompt-modification code, and an obsolete chat
            # must not steer the standing prompt (even a logged proposal).
            if (target_source != "revised_missing_ideal"
                    and verdict == "revise" and not config.revisit
                    and getattr(config.overrides, "log_prompt_mutation", True)):
                try:
                    self._run_prompt_mutation_for_exchange(
                        run_id, filename, job, session,
                        generate_fn=generate_fn, context_length=context_length,
                        temperature=temperature, top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        verdict=verdict, why=why, ideal=ideal, target=target,
                        persona_digest=persona_digest,
                        exchange_label=ex_idx, exchange_total=exchange_total,
                        store=store, send_event_fn=send_event_fn,
                        tokenizer=tokenizer, stats=stats,
                    )
                except Exception as e:
                    self._emit(store, send_event_fn, run_id, "pass_warning",
                               phase="prompt_mutation", session=filename,
                               exchange_index=ex_idx,
                               message=f"Prompt-mutation pass error (skipped): {e}")

        # (The per-session .shareml.json training document is retired — REBUILD: nothing
        # read it, the from-scratch build reads the chat sidecars directly, and it encoded
        # the removed variant-count / decay-stage concepts. Target assembly now lives in the
        # trimmed reflection_shareml helper used by training.dialogue_source.)

        return total_passes, skipped_passes, agg

    # ── language judgement + clean IDEAL generation ───────────────────── #

    def _language_decision_guard(
        self,
        *,
        response: str,
        verdict: Optional[str],
        why: Optional[str],
        persona: list,
        job: dict,
        content: str,
        revision_prompt: str,
        generate_fn: Callable,
        temperature: float,
        top_p: float,
        max_new_tokens: str,
        filename: str,
        rag_query: str,
        on_chunk: Callable,
        ex_idx: int,
        exchange_total: int,
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        run_id: str,
        tokenizer=None,
        stats: Optional[RunStats] = None,
    ) -> tuple:
        """Detect an unbidden language switch in the original reply.

        Returns the possibly-updated judgement plus ``lang_drift``. A script mismatch
        the first judgement did not flag gets one pointed re-judgement so the model can
        distinguish an accidental switch from a requested translation. Replacement reply
        generation is deliberately absent from this reflection lane.

        The MODEL is the decider: its ``LANG_DRIFT`` marker (intent-aware — a requested
        translation is legitimate) is honoured over the relative script-level backstop in
        :mod:`core.reflection_lang`, which only exists to catch a hard mismatch the model
        failed to flag and prompt one pointed re-judgment.

        That decider rule holds through the re-judgement too: drift is set ONLY when the
        re-judgement affirms it (``VERDICT: revise`` or ``LANG_DRIFT: yes``). Anything
        else — a keep, a keep missing its marker, an unparseable answer, a failed or
        degenerated generation — restores the original judgement, because the script
        backstop on its own is evidence the model has now twice declined to endorse. It
        is a *relative* heuristic and false-positives on a user turn that is dominated by
        Latin characters for non-language reasons (pasted code, URLs, English proper
        nouns in an otherwise Russian conversation); a forced ``revise`` there costs the
        exchange entirely, since the IDEAL gate would then demand the wrong script.
        """
        ex = job["exchange"]
        # Genuine user-authored text only. A stage-direction turn — the `(initiative)`
        # impulse that opens every outreach/synthesis/check-in session, the `(setting)`
        # framing that opens an encounter — sits in the user slot but is a fixed ENGLISH
        # template Ava wrote to herself. Counted as conversation, it made a Russian
        # Ava-initiated chat read as an English one (91 Latin characters against the
        # user's 20), so her correct Russian reply was flagged as drift. Systematic, and
        # worst on short chats — which is exactly what an unanswered opener is.
        user_texts = conversation_user_texts(job)
        resp_text = ex.get("assistant_response") or ""

        marker = revision_lang_drift(response)           # True / False / None
        script_drift, conv_fam, resp_fam = detect_language_drift(resp_text, user_texts)

        # Decider = the model. An explicit no-marker is trusted even over a script mismatch
        # (requested translation). Absent a marker, the script backstop stands in.
        if marker is True:
            drift = True
        elif marker is False:
            drift = False
        else:
            drift = script_drift

        def _revise_gen(nudge: str) -> Optional[str]:
            gen_t0 = time.monotonic()
            try:
                out = generate_fn(
                    content, revision_prompt + nudge,
                    temperature=temperature, top_p=top_p,
                    max_new_tokens_setting=max_new_tokens,
                    before_session=filename, rag_query=rag_query,
                    rag_include_chat=False,
                    rag_include_persona=False,   # same fence as the main judgement
                    on_chunk=on_chunk,
                )
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="revision", session=filename, exchange_index=ex_idx,
                           message=f"Language-guard generation errored: {e}")
                return None
            if stats is not None:
                stats.record_phase("revision", time.monotonic() - gen_t0,
                                   tokens=_count_tokens(tokenizer, out or ""))
            return out

        # Backstop: a hard script mismatch the model did NOT flag → one pointed re-judgment
        # so it can either confirm-and-correct or say the switch was requested.
        if script_drift and marker is not True:
            self._emit(
                store, send_event_fn, run_id, "pass_warning",
                phase="revision", session=filename,
                exchange_index=ex_idx, exchange_total=exchange_total,
                message=(f"Reply script ({resp_fam}) differs from the conversation "
                         f"({conv_fam}) but wasn't flagged; re-judging for language drift."),
            )
            rej = _revise_gen(_LANG_RECHECK_NUDGE)
            v2 = w2 = None
            p2: list = []
            m2 = None
            if rej is not None:
                v2, w2, p2 = parse_revision_judgement(rej)
                m2 = revision_lang_drift(rej)
            if v2 == "revise" or m2 is True:
                # Asked pointedly, the model confirms the switch was unbidden.
                drift = True
                if v2:                          # never adopt an unparsed verdict
                    response, verdict, why = rej, v2, w2
                    if p2:
                        persona = p2
            else:
                # Everything else leaves the script heuristic UNCONFIRMED: an explicit
                # keep, a keep with no marker (the nudge asks for both, and half an
                # answer is still the model declining to call it drift), an unparseable
                # judgement, or a generation that errored or degenerated. The backstop is
                # not a decider, so the original judgement stands.
                #
                # Letting it force `revise` here was destructive rather than merely
                # wrong: the IDEAL gate below runs with require_user_language=True, which
                # accepts a re-answer only if its script matches the *conversation's*
                # family — so on a FALSE positive (a Russian chat whose user turn is
                # Latin-dominant with code/URLs/English proper nouns) a correct Russian
                # IDEAL is rejected twice, the exchange lands in revised_missing_ideal,
                # and it silently drops out of the training corpus.
                if drift and rej is not None:
                    self._emit(
                        store, send_event_fn, run_id, "pass_warning",
                        phase="revision", session=filename,
                        exchange_index=ex_idx, exchange_total=exchange_total,
                        message=("Re-judgement did not confirm language drift "
                                 "(script mismatch is likely code/URLs/proper nouns in "
                                 "the user's turn); keeping the original judgement."),
                    )
                drift = False

        if drift:
            verdict = "revise"
        return response, verdict, why, persona, bool(drift)

    def _write_counter_evidence(
        self, *, job: dict, filename: str, why: Optional[str],
        persona_keys_fn: Optional[Callable], run_id: str,
        ex_idx: int, exchange_total: int,
        store: ReflectionRunStore, send_event_fn: Optional[Callable],
        stats: Optional[RunStats] = None,
    ) -> None:
        """Emit persona counter-evidence for a pushed-against reply (the persuasion channel).

        The next-turn reaction has already been classified as pushback (``COUNTER: yes``).
        Resolve the stance(s) the REPLY expressed via the persona-key bridge
        (``rag.persona_keys`` — persona-only, cut to before this chat, top-2, clearly-relevant
        only), and append one ``counter`` ledger op per key under THIS chat as the pushing
        session. Distinct-session currency: a stance pushed across many separate chats
        accumulates and eventually fades the trait, while any single push barely moves a mature
        one (``reflection_digest._PERSUASION_GAIN``). Never fabricates — no bridge, or no
        clearly-relevant live stance, ⇒ nothing written (a reaction lands only where it maps to
        an actual self-statement). Best-effort: a failure is logged as a skip and never derails
        the revision pass. Suppressed on dry runs by construction (the whole persist block is)."""
        if persona_keys_fn is None:
            return
        reply = ((job.get("exchange") or {}).get("assistant_response") or "").strip()
        if not reply:
            return
        try:
            hits = persona_keys_fn(reply, filename) or []
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "counter_skipped",
                       session=filename, exchange_index=ex_idx,
                       message=f"Counter-evidence lookup failed (skipped): {e}")
            return
        keys = [k for k, _sim in hits if k]
        if not keys:
            self._emit(store, send_event_fn, run_id, "counter_skipped",
                       session=filename, exchange_index=ex_idx,
                       exchange_total=exchange_total,
                       message="Pushback noted, but no clearly-relevant live stance to counter.")
            return
        try:
            from training.ledger import ConsolidationLedger
            n = ConsolidationLedger(self._consolidation_dir).counter(
                keys, source_session=filename,
                reason=(why or "next-turn pushback"), run_id=run_id)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "counter_skipped",
                       session=filename, exchange_index=ex_idx,
                       message=f"Counter-evidence write failed (skipped): {e}")
            return
        if stats is not None and hasattr(stats, "note_counter"):
            stats.note_counter(n)
        self._emit(store, send_event_fn, run_id, "counter_written",
                   session=filename, exchange_index=ex_idx, exchange_total=exchange_total,
                   keys=keys,
                   message=(f"Counter-evidence: pushed against {n} stance(s) "
                            f"({', '.join(keys)})"))

    def _generate_ideal_reply(
        self,
        *,
        job: dict,
        session: dict,
        generate_fn: Callable,
        temperature: float,
        top_p: float,
        max_new_tokens: str,
        require_user_language: bool,
        persona_context: str = "",
        filename: str,
        ex_idx: int,
        exchange_total: int,
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        run_id: str,
        tokenizer=None,
        stats: Optional[RunStats] = None,
    ) -> Optional[str]:
        """Generate a CoT-bearing reply from the clean pre-answer dialogue prefix.

        *persona_context* (when non-empty) is appended to the system prompt of the generation
        messages so the re-derived CoT is authored persona-conditioned. The SAME block is
        persisted onto the anchor by the caller (ideal-win only), so parity holds.

        The first attempt uses the exact persisted training prefix. A malformed/truncated
        result gets one retry with perturbed sampling but the identical messages. For a
        confirmed language-drift repair only, the retry may append a neutral language
        constraint to the system message; it still contains no old reply or diagnosis.
        """
        ex = job.get("exchange") or {}
        # Same assembly as the script backstop's, and deliberately the same helper: this
        # gate is ARMED by that backstop's verdict, so if the two disagreed about what
        # language the conversation is in, a drift call there would reject a correct
        # re-answer here — twice — and drop the exchange from the corpus.
        user_texts = conversation_user_texts(job)

        def _acceptable(text: Optional[str]) -> Optional[str]:
            target = ideal_trainable_target(text)
            if target is None:
                return None
            if require_user_language:
                drift, _conv, _reply = detect_language_drift(target, user_texts)
                if drift:
                    return None
            return target

        def _call(messages: list[dict], temp: float, attempt: int) -> Optional[str]:
            store.update_status(run_id, phase="ideal_generation")
            self._emit(
                store, send_event_fn, run_id, "phase_started",
                phase="ideal_generation", session=filename,
                exchange_index=ex_idx, exchange_total=exchange_total,
                attempt=attempt,
                message=("Generating clean IDEAL from the pre-answer conversation"
                         + (" (retry)" if attempt > 1 else "")),
            )

            def _on_chunk(delta: str) -> None:
                self._emit(store, send_event_fn, run_id, "phase_progress",
                           phase="ideal_generation", session=filename,
                           exchange_index=ex_idx, exchange_total=exchange_total,
                           text=delta)

            _vram_reset_peak()
            gen_t0 = time.monotonic()
            try:
                out = generate_fn(
                    "", "", messages_override=messages,
                    temperature=temp, top_p=top_p,
                    max_new_tokens_setting=max_new_tokens,
                    disable_rag=True, on_chunk=_on_chunk,
                )
            except Exception as e:
                out = None
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="ideal_generation", session=filename,
                           exchange_index=ex_idx, exchange_total=exchange_total,
                           message=f"Clean IDEAL generation attempt {attempt} failed: {e}")
            if stats is not None:
                stats.record_phase("ideal_generation", time.monotonic() - gen_t0,
                                   tokens=_count_tokens(tokenizer, out or ""))
                context_text = "\n".join(str(m.get("content") or "") for m in messages)
                stats.observe_vram(_vram_peak_gb(), phase="ideal_generation",
                                   context_tokens=_count_tokens(tokenizer, context_text))
            return out

        base_messages = build_ideal_messages(job, session, persona_context=persona_context)
        first = _call(base_messages, temperature, 1)
        target = _acceptable(first)
        if target is not None:
            store.update_status(run_id, phase="revision")
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="ideal_generation", session=filename,
                       exchange_index=ex_idx, exchange_total=exchange_total,
                       text=target, message="Clean IDEAL generation done")
            return target

        self._emit(store, send_event_fn, run_id, "pass_warning",
                   phase="ideal_generation", session=filename,
                   exchange_index=ex_idx, exchange_total=exchange_total,
                   message="Clean IDEAL was malformed, truncated, or in the wrong language; "
                           "retrying from the same pre-answer conversation.")
        suffix = _IDEAL_LANGUAGE_SUFFIX if require_user_language else ""
        retry_messages = build_ideal_messages(
            job, session, system_suffix=suffix, persona_context=persona_context)
        second = _call(retry_messages, min(1.0, temperature + _REVISION_RETRY_TEMP_BUMP), 2)
        target = _acceptable(second)
        if stats is not None:
            stats.note_retry(recovered=target is not None)
        store.update_status(run_id, phase="revision")
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="ideal_generation", session=filename,
                   exchange_index=ex_idx, exchange_total=exchange_total,
                   text=target or "",
                   message=("Clean IDEAL generation done after retry" if target is not None
                            else "Clean IDEAL generation failed after retry"))
        return target

    # ── CoT regeneration for a kept, corrupt-CoT reply (approach #3) ────── #

    def _regenerate_cot_for_kept_reply(
        self,
        *,
        job: dict,
        session: dict,
        generate_fn: Callable,
        temperature: float,
        top_p: float,
        max_new_tokens: str,
        similarity_fn: Optional[Callable],
        filename: str,
        ex_idx: int,
        exchange_total: int,
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        run_id: str,
        tokenizer=None,
        stats: Optional[RunStats] = None,
    ) -> Optional[str]:
        """Regenerate a faithful ``<think>`` for a corrupt-CoT exchange the judge KEPT.

        The operator flagged only the CoT corrupt, so the stored reply is trusted. We
        re-answer the exchange as a normal dialogue completion (authoring a fresh
        ``<think>`` + reply *together* — faithful by construction, the same clean seam the
        IDEAL path uses), then keep ONLY the fresh ``<think>``, grafted onto the ORIGINAL
        reply — but only when the re-answer's reply is at least ``_RECOT_SIMILARITY_FLOOR``
        cosine-similar to the original, so the fresh thought genuinely leads to ~that reply.
        Below the floor (or with no embedder / no re-answer) we return ``None`` and the
        caller falls back to answer-only (the prior corrupt-CoT behavior), never pairing a
        thought with a reply it did not produce (the think/answer mismatch that erodes the
        reasoning channel over from-scratch rebuilds).

        Returns the grafted ``<think>{fresh_cot}</think>\\n\\n{original_reply}`` target, or None.
        """
        original_reply = ((job.get("exchange") or {}).get("assistant_response") or "").strip()
        if not original_reply:
            return None

        # Re-answer through the exact clean IDEAL seam (require_user_language=False: a kept
        # verdict already means no language drift). Reuses the malformed/truncated retry.
        fresh = self._generate_ideal_reply(
            job=job, session=session, generate_fn=generate_fn,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
            require_user_language=False,
            filename=filename, ex_idx=ex_idx, exchange_total=exchange_total,
            store=store, send_event_fn=send_event_fn, run_id=run_id,
            tokenizer=tokenizer, stats=stats,
        )
        fresh_cot = _leading_target_cot(fresh or "")
        fresh_reply = _leading_target_answer(fresh or "")
        if not fresh_cot or not fresh_reply:
            if stats is not None:
                stats.note_cot_regen_fallback()
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="cot_regen", session=filename,
                       exchange_index=ex_idx, exchange_total=exchange_total,
                       message="CoT-regen fallback: re-answer produced no usable "
                               "<think>+reply; keeping the reply answer-only.")
            return None

        sim: Optional[float] = None
        if similarity_fn is not None:
            try:
                sim = similarity_fn(fresh_reply, original_reply)
            except Exception:
                sim = None

        if sim is None or sim < _RECOT_SIMILARITY_FLOOR:
            if stats is not None:
                stats.note_cot_regen_fallback()
            sim_str = f"{sim:.2f}" if sim is not None else "n/a"
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="cot_regen", session=filename,
                       exchange_index=ex_idx, exchange_total=exchange_total,
                       message=(f"CoT-regen fallback: re-answer diverged from the kept reply "
                                f"(similarity {sim_str} < {_RECOT_SIMILARITY_FLOOR}); "
                                "keeping the reply answer-only."))
            return None

        graft = f"<think>{fresh_cot}</think>\n\n{original_reply}"
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="cot_regen", session=filename,
                   exchange_index=ex_idx, exchange_total=exchange_total,
                   text=graft,
                   message=(f"CoT regenerated for the kept reply "
                            f"(re-answer similarity {sim:.2f} ≥ {_RECOT_SIMILARITY_FLOOR})."))
        return graft

    # ── branch experiment phase ────────────────────────────────────────── #

    def _run_branch_for_exchange(
        self,
        config: ReflectionRunConfig,
        filename: str,
        job: dict,
        *,
        generate_fn: Callable,
        branch_generate_fn: Callable,
        branch_chooser_content_fn: Callable,
        branch_prompt: str,
        br_temperature: float,
        br_top_p: float,
        ideal_text: str = "",
        exchange_label: int = 0,
        exchange_total: int = 0,
        tokenizer=None,
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        persona_digest: Optional[dict] = None,
        stats: Optional[RunStats] = None,
    ) -> Optional[dict]:
        """Generate and blind-select counterfactual branches for one exchange.

        Returns a branch_block dict (suitable for write_revision's *branch* arg)
        or None when the exchange is ineligible or generation fails.

        The shuffle order is recorded in the block under *branch_order_seed* so
        the blind selection can be reproduced for research. Options are mixed into
        a uniform list — kind="original", "branch", or "ideal" — so the chooser
        cannot distinguish them by position or label. When *ideal_text* is non-empty
        (the clean re-answer produced an IDEAL), it joins the choice set so the blind
        winner is chosen fairly among the original, the branches, and the ideal.
        """
        run_id = config.run_id
        ex_idx = job["index"]

        # Generate branches at the *original exchange's* sampling settings, not the
        # reflection ones: the experiment replays the road-not-taken under the same
        # conditions the live reply was sampled with (typically a high chat
        # temperature). The blind chooser below keeps the (typically low) reflection
        # br_temperature/br_top_p, so branches are produced hot but judged cold. Fall
        # back to the configured branch sampling for older exchanges logged before
        # generation_params existed.
        gen_params = (job["exchange"].get("generation_params") or {})
        gen_temperature = br_temperature
        gen_top_p = br_top_p
        if gen_params.get("temperature") is not None:
            gen_temperature = float(gen_params["temperature"])
        if gen_params.get("top_p") is not None:
            gen_top_p = float(gen_params["top_p"])

        # Announce before the (potentially minutes-long) counterfactual generation
        # so the run doesn't look stalled. Ineligible exchanges resolve to a quick
        # branch_skipped right after.
        self._emit(store, send_event_fn, run_id, "branch_started",
                   session=filename, exchange_index=exchange_label,
                   exchange_total=exchange_total,
                   message="Branch experiment — generating counterfactuals…")

        # Stream each fork's text to the Sleep tab as it's generated — branch
        # generation is now fully sequential (see BRANCH_FORK_BATCH), so this is
        # the only sign of life during what can be minutes of silent GPU work.
        def _on_candidate(index: int, total: int, text: str) -> None:
            self._emit(store, send_event_fn, run_id, "branch_candidate",
                       session=filename, exchange_index=exchange_label,
                       exchange_total=exchange_total,
                       candidate_index=index, candidate_total=total,
                       text=_clip(text))

        # VRAM probe — measure the generation stage's peak so the report can be
        # compared against the chooser stage (the suspected high-water mark).
        _vram_reset_peak()
        gen_t0 = time.monotonic()
        try:
            result = branch_generate_fn(
                filename, ex_idx, gen_temperature, gen_top_p,
                on_candidate=_on_candidate,
            )
        except Exception as e:
            if stats is not None:
                stats.note_branch_skipped("generation failed")
            self._emit(store, send_event_fn, run_id, "branch_skipped",
                       session=filename, exchange_index=exchange_label,
                       message=f"Branch generation failed: {e}")
            return None
        gen_secs = time.monotonic() - gen_t0
        gen_peak = _vram_peak_gb()
        if stats is not None:
            # Credit the phase with the fork continuations' generated tokens
            # (reported by run_branch_exchange) — timing a generation phase with
            # tokens=0 silently dilutes the run-wide tok/s readout.
            stats.record_phase("branch_gen", gen_secs,
                               tokens=int(result.get("generated_tokens") or 0))
            stats.observe_vram(gen_peak, phase="branch_gen")

        candidates = result.get("candidates") or []
        dropped = result.get("dropped") or []
        n_generated = result.get("n_generated", 0)

        if not result.get("eligible") or not candidates:
            reason = result.get("reason") or "no eligible candidates survived filtering"
            if stats is not None:
                stats.note_branch_skipped("no eligible candidates")
            self._emit(store, send_event_fn, run_id, "branch_skipped",
                       session=filename, exchange_index=exchange_label,
                       message=f"Branch skipped: {reason}")
            return None

        # Eligible: counterfactual candidates survived generation + filtering, so this
        # exchange reaches the blind choice (vs the branch_skipped exchanges above).
        if stats is not None:
            stats.note_branch_eligible()

        # Shuffle candidates + original together; record seed for reproducibility.
        seed = random.randint(0, 2 ** 31 - 1)
        rng = random.Random(seed)
        exchange = job["exchange"]
        original_text = (exchange.get("assistant_response") or "").strip()

        options = [
            {
                "text": (c.get("text") or "").strip(),
                "kind": "branch",
                "position": c.get("position"),
                "token": c.get("token"),
                "alt_token": c.get("alt_token"),
                "similarity_to_original": c.get("similarity_to_original"),
            }
            for c in candidates
        ]
        options.append({"text": original_text, "kind": "original"})
        ideal_clean = (ideal_text or "").strip()
        if ideal_clean:
            options.append({"text": ideal_clean, "kind": "ideal"})
        rng.shuffle(options)
        original_index = next(i for i, o in enumerate(options) if o["kind"] == "original")
        ideal_index = next((i for i, o in enumerate(options) if o["kind"] == "ideal"), None)

        payload = {
            "context": job["context"],
            "speaker": job["speaker"],
            "user_prompt": (exchange.get("user_prompt") or "").strip(),
            "options": options,
        }

        try:
            content = branch_chooser_content_fn(payload, branch_prompt)
        except Exception as e:
            if stats is not None:
                stats.note_branch_skipped("chooser prompt too large")
            self._emit(store, send_event_fn, run_id, "branch_skipped",
                       session=filename, exchange_index=exchange_label,
                       message=f"Chooser prompt too large (skipped): {e}")
            return None

        # VRAM probe — measure the chooser stage in isolation, and break the
        # chooser prompt into the option-text slice (what serial/pairwise judging
        # would shrink) vs the shared remainder (dialogue context + scaffolding,
        # which pairwise leaves untouched). The ratio decides whether "serialize
        # the branches" would actually move the peak.
        chooser_prompt_tokens = _count_tokens(tokenizer, content)
        option_tokens = sum(_count_tokens(tokenizer, o.get("text") or "") for o in options)
        # Distinct marker so the (separately timed) choice stage is visible in the log —
        # generation and choice were previously indistinguishable from the event stream.
        self._emit(store, send_event_fn, run_id, "branch_choosing",
                   session=filename, exchange_index=exchange_label,
                   exchange_total=exchange_total,
                   message=f"Choosing among {len(options)} replies…")
        _vram_reset_peak()
        choose_t0 = time.monotonic()
        try:
            ch_response = generate_fn(
                content, branch_prompt,
                temperature=br_temperature, top_p=br_top_p,
                max_new_tokens_setting=_DEFAULT_MAX_NEW_TOKENS,
                disable_rag=True,
                disable_thinking=True,
            )
        except Exception as e:
            if stats is not None:
                stats.note_branch_skipped("chooser generation failed")
            self._emit(store, send_event_fn, run_id, "branch_skipped",
                       session=filename, exchange_index=exchange_label,
                       message=f"Chooser generation failed (skipped): {e}")
            return None
        choose_secs = time.monotonic() - choose_t0
        chooser_peak = _vram_peak_gb()
        if stats is not None:
            stats.record_phase("branch_choose", choose_secs,
                               tokens=_count_tokens(tokenizer, ch_response))
            stats.observe_vram(chooser_peak, phase="branch_choose",
                               context_tokens=chooser_prompt_tokens)

        probe = self._build_branch_probe(
            run_id=run_id, filename=filename, exchange_label=exchange_label,
            n_options=len(options), gen_peak=gen_peak, chooser_peak=chooser_peak,
            chooser_prompt_tokens=chooser_prompt_tokens, option_tokens=option_tokens,
            gen_secs=gen_secs, choose_secs=choose_secs,
        )

        chosen, why = _parse_choice(ch_response, len(options))
        original_rechosen = (chosen == original_index) if chosen is not None else None
        chosen_kind = options[chosen]["kind"] if chosen is not None else None
        if chosen is None and stats is not None:
            # The chooser ran but produced no parseable CHOICE — the branch contributes
            # no winner; the exchange falls back to its verdict-resolved target.
            stats.note_discard("branch_unparseable")

        # Phase two (LOGGED-ONLY): record what the digest-anchored "become" criterion would
        # pick over the same blind options, for comparison against the blind chooser. The
        # cheap embedding channel (no LLM, known weak — near-duplicate / cross-lingual
        # replies, see AVA_DESIGN_LEGACY.md) is logged inline here. The LLM judge is NOT run here: it
        # is collected as a job and run after all sessions, batched on the CLEAN base (one
        # adapter swap per run) — see execute_run step 4 / _run_clean_base_judge. Neither
        # channel changes `chosen`, the verdict, or the trainable target.
        digest_select = None
        if persona_digest:
            emb = self._digest_embedding_select(
                persona_digest, options, chosen=chosen, original_index=original_index)
            if emb:
                digest_select = {
                    "digest_version": persona_digest.get("version"),
                    "embedding": emb,
                }
        # The judge job (for Phase B, clean base) is assembled by the caller after target
        # resolution, where the verdict/ideal/CoT needed to re-resolve a judge override are
        # in scope. _run_branch_for_exchange only surfaces the options here on the block.

        # Surface the outcome so the Sleep tab shows what the experiment concluded,
        # not just that it ran.
        n_alt = len(options) - 1
        if chosen is None:
            outcome = "Branch: chooser output unparseable"
        elif original_rechosen:
            outcome = f"Branch: re-chose original over {n_alt} alternative(s)"
        else:
            outcome = f"Branch: chose {chosen_kind} over original ({n_alt} alternative(s))"

        # Show the operator the full set the chooser saw, in the blind order it saw it
        # (letters A, B, … as labeled in the prompt), with the winner marked. Texts are
        # clipped only against pathological lengths; the persisted block keeps them whole.
        event_options = [
            {"letter": chr(ord("A") + i), "kind": o["kind"],
             "text": _clip(o.get("text") or ""), "chosen": (i == chosen)}
            for i, o in enumerate(options)
        ]
        # The chooser's reasoning lives in the think channel (normalized to
        # <think>…</think> by the cleaner). Pull it out for display; on an unparseable
        # result with no closed think block, fall back to whatever was emitted so the
        # failure isn't opaque.
        cot_match = re.search(r"(?s)<think>(.*?)</think>", ch_response or "")
        if cot_match:
            chooser_cot = cot_match.group(1).strip()
        elif chosen is None:
            chooser_cot = (ch_response or "").strip()
        else:
            chooser_cot = ""

        self._emit(store, send_event_fn, run_id, "branch_done",
                   session=filename, exchange_index=exchange_label,
                   exchange_total=exchange_total,
                   chosen_kind=chosen_kind, original_rechosen=original_rechosen,
                   n_options=len(options), n_generated=n_generated, why=why,
                   options=event_options, chooser_cot=_clip(chooser_cot),
                   probe=probe, message=outcome,
                   digest_select=digest_select)

        return {
            "mode": "experiment",
            "candidates": options,
            "original_index": original_index,
            "ideal_index": ideal_index,
            "chosen_index": chosen,
            "chosen_kind": chosen_kind,
            "original_rechosen": original_rechosen,
            "why": why,
            "chooser_raw": ch_response,
            "gen_temperature": gen_temperature,
            "gen_top_p": gen_top_p,
            "branch_order_seed": seed,
            "dropped": dropped,
            "n_generated": n_generated,
            "probe": probe,
            "digest_select": digest_select,
        }

    # ── prompt-mutation phase (LOGGED-ONLY) ───────────────────────────────── #

    def _run_prompt_mutation_for_exchange(
        self,
        run_id: str,
        filename: str,
        job: dict,
        session: dict,
        *,
        generate_fn: Callable,
        context_length: int,
        temperature: float,
        top_p: float,
        max_new_tokens: str,
        verdict: Optional[str],
        why: Optional[str],
        ideal: Optional[str],
        target: str,
        persona_digest: Optional[dict],
        exchange_label: int,
        exchange_total: int,
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        tokenizer=None,
        stats: Optional[RunStats] = None,
    ) -> None:
        """Counterfactual on the standing prompt for one drifted exchange — logged only.

        Asks Ava whether a *different standing prompt* would have produced the better
        reply on its own, and if so what line it would need. Measures the drift against
        the persona digest (the "better me"); appends any concrete delta to
        ``data/hot/prompt/prompt_deltas.jsonl`` and emits a ``prompt_delta_proposed``
        event. Mutates no prompt — the live ``chat_prompt.txt`` is read, never written.
        """
        from core import prompt_mutation, reflection_digest
        from core.reflection_source import build_revision_content
        from training.reflections_path import prompt_dir

        try:
            template = prompt_mutation.load_prompt_mutation_prompt()
        except Exception:
            return  # prompt file missing — nothing to run

        # The "better me" criterion: the digest if there is one, else a graceful
        # fallback so the pass is useful from day one (before any digest exists).
        persona_block = (reflection_digest.render_digest_for_judge(persona_digest)
                         if persona_digest else "")
        if not persona_block:
            persona_block = ("(No settled self-portrait yet — weigh the drift against "
                             "the better reply below and your own sense of who you are.)")
        current_prompt = prompt_mutation.load_current_chat_prompt() or "(unavailable)"
        system_prompt = (template
                         .replace("{current_prompt}", current_prompt)
                         .replace("{persona}", persona_block))

        # Show the same exchange the revision pass judged, plus the reply it stands
        # behind — so the counterfactual reasons about the gap, not from scratch.
        content = build_revision_content(job, session, context_length)
        better = (ideal or target or "").strip()
        tail = ["", "--- WHAT MY REVISION STANDS BEHIND INSTEAD ---"]
        if (why or "").strip():
            tail.append(f"Why the original fell short: {why.strip()}")
        if better:
            tail.append("The reply I stand behind now:\n" + better)
        content = content + "\n" + "\n".join(tail)

        # Own phase boundary so this pass no longer hides in the gap between a
        # revision's phase_done and the next exchange's phase_started; its CoT
        # streams to the Sleep log live (like the revision pass).
        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="prompt_mutation", session=filename,
                   exchange_index=exchange_label, exchange_total=exchange_total,
                   message="Prompt-mutation pass (standing-prompt drift counterfactual)")

        def _on_chunk(delta: str, _sess=filename, _ex=exchange_label,
                      _total=exchange_total) -> None:
            self._emit(store, send_event_fn, run_id, "phase_progress",
                       phase="prompt_mutation", session=_sess,
                       exchange_index=_ex, exchange_total=_total, text=delta)

        # Introspection over the exchange + digest + current prompt — RAG would only
        # add noise, so it is disabled (as in the digest/judge passes).
        pm_t0 = time.monotonic()
        try:
            response = generate_fn(
                content, system_prompt,
                temperature=temperature, top_p=top_p,
                max_new_tokens_setting=max_new_tokens,
                before_session=filename, disable_rag=True,
                on_chunk=_on_chunk,
            )
        except Exception as e:
            if stats is not None:
                stats.record_phase("prompt_mutation", time.monotonic() - pm_t0)
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="prompt_mutation", session=filename,
                       exchange_index=exchange_label, exchange_total=exchange_total,
                       message=f"Prompt-mutation pass failed (skipped): {e}")
            return
        # This pass was previously invisible to the per-phase report (and its
        # untokened seconds would dilute the run-wide tok/s) — time it under its
        # own slot, credited with its generated tokens.
        if stats is not None:
            stats.record_phase("prompt_mutation", time.monotonic() - pm_t0,
                               tokens=_count_tokens(tokenizer, response))

        # Close the boundary; text= is the fallback dump if nothing streamed.
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="prompt_mutation", session=filename,
                   exchange_index=exchange_label, exchange_total=exchange_total,
                   text=response, message="Prompt-mutation pass done")

        parsed = prompt_mutation.parse_prompt_mutation(response)
        # The common, healthy case (prompt was adequate / no concrete line) is a silent
        # no-op — the log only carries proposals worth a human's attention.
        if not prompt_mutation.has_prompt_gap(parsed):
            return

        ex = job["exchange"]
        record = {
            "run_id": run_id,
            "source_session": filename,
            "exchange_index": job["index"],
            "speaker": job.get("speaker"),
            "verdict": parsed["verdict"],
            "scope": parsed["scope"],
            "drift": parsed["drift"],
            "missing": parsed["missing"],
            "delta": parsed["delta"],
            "revision_verdict": verdict,
            "revision_why": why or "",
            "digest_version": (persona_digest or {}).get("version"),
            "user_prompt": (ex.get("user_prompt") or ""),
            "original_response": (ex.get("assistant_response") or ""),
            "ideal": better,
        }
        try:
            prompt_mutation.append_prompt_delta(prompt_dir(), record)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "pass_warning",
                       phase="prompt_mutation", session=filename,
                       exchange_index=exchange_label,
                       message=f"Prompt-mutation log write failed: {e}")
            return

        self._emit(
            store, send_event_fn, run_id, "prompt_delta_proposed",
            session=filename, exchange_index=exchange_label,
            exchange_total=exchange_total, scope=parsed["scope"],
            delta=_clip(parsed["delta"], 1000), drift=_clip(parsed["drift"], 1000),
            missing=_clip(parsed["missing"], 1000),
            digest_version=(persona_digest or {}).get("version"),
            message=(f"Prompt-gap [{parsed['scope'] or '?'}]: {_clip(parsed['delta'], 200)}"),
        )

    @staticmethod
    def _digest_embedding_select(persona_digest, options, *, chosen, original_index):
        """Logged-only embedding channel: what voice-similarity to the digest's anchor_texts
        would pick. Pure comparison data — never changes the chosen target. Returns None when
        unavailable. Known weak for branch-select (near-duplicate whole texts + cross-lingual
        reply-vs-anchor mismatch — see AVA_DESIGN_LEGACY.md); kept for continued comparison only.
        """
        if not persona_digest:
            return None
        from core import reflection_digest
        scored = reflection_digest.score_texts_against_digest(
            persona_digest, [(o.get("text") or "") for o in options])
        if not scored:
            return None
        pick = scored["pick_index"]
        pick_kind = options[pick]["kind"] if 0 <= pick < len(options) else None
        return {
            "scores": scored["scores"],
            "pick_index": pick,
            "pick_kind": pick_kind,
            "n_anchors": scored["n_anchors"],
            "agrees_with_chooser": (chosen is not None and pick == chosen),
            "agrees_with_original": (pick == original_index),
        }

    def _digest_judge_select(self, persona_digest, payload, options, *,
                             generate_fn, branch_chooser_content_fn,
                             br_temperature, br_top_p, chosen, original_index,
                             tokenizer=None):
        """Logged-only judge channel (the primary phase-two mechanism): a digest-aware pass
        that reasons over the digest's dispositions/lines to pick which option is most "who
        I'm becoming". Reuses the blind chooser's budgeted-content machinery with a
        digest-anchored system prompt, so it sees the *identical* blind option set in the
        same order. Returns None when the digest has no usable material or the pass fails.
        Never changes the chosen target.
        """
        from core import reflection_digest
        persona_block = reflection_digest.render_digest_for_judge(persona_digest)
        if not persona_block:
            return None
        try:
            judge_prompt = reflection_digest.load_branch_judge_prompt().replace(
                "{persona}", persona_block)
        except Exception:
            return None
        try:
            content = branch_chooser_content_fn(payload, judge_prompt)
            resp = generate_fn(
                content, judge_prompt,
                temperature=br_temperature, top_p=br_top_p,
                # The judge emits a single pick + a one-line why (thinking off), so it needs
                # no more headroom than any other reflection pass — cap it at the shared 8K
                # ceiling rather than the branch chooser's wider 12K.
                max_new_tokens_setting=_DEFAULT_MAX_NEW_TOKENS, disable_rag=True,
                disable_thinking=True,
            )
        except Exception:
            return None
        pick, why = _parse_choice(resp, len(options))
        pick_kind = (options[pick]["kind"]
                     if (pick is not None and 0 <= pick < len(options)) else None)
        cot = re.search(r"(?s)<think>(.*?)</think>", resp or "")
        return {
            "pick_index": pick,
            "pick_kind": pick_kind,
            "why": why,
            "agrees_with_chooser": (chosen is not None and pick is not None and pick == chosen),
            "agrees_with_original": ((pick == original_index) if pick is not None else None),
            "cot": _clip(cot.group(1).strip()) if cot else "",
            # Tokens this judge pass generated — read back by _run_clean_base_judge
            # so the branch_judge phase's wall time is credited with its output in
            # RunStats (tokens=0 would dilute the run-wide tok/s readout).
            "gen_tokens": _count_tokens(tokenizer, resp),
        }

    @staticmethod
    def _load_persona_digest() -> Optional[dict]:
        """The current live persona digest (read once per run), or None. Best-effort."""
        try:
            from core import reflection_digest
            from training.reflections_path import persona_dir
            return reflection_digest.latest_digest(persona_dir())
        except Exception:
            return None

    def _run_clean_base_judge(self, judge_jobs, persona_digest, run_id, *, config,
                              generate_fn, branch_chooser_content_fn, store, send_event_fn,
                              apply_flip=False, stats: Optional[RunStats] = None,
                              tokenizer=None):
        """Phase-two judge, batched on the CLEAN base. Runs *inside* ``clean_base_ctx`` — the
        adapter is swapped out, so ``generate_fn`` (which reads model state at call time)
        targets the frozen base: the judge is replay-faithful, mode-collapse-guarded, and
        immune to a bad adapter. For each collected branch job it picks which option is most
        "who I'm becoming" and logs a ``branch_judged`` event (joined to the exchange's
        ``branch_done`` by session+exchange_index).

        **Criterion flip:** when *apply_flip* is on AND the digest clears the numeric-
        recurrence maturity gate, a judge pick that differs from the blind choice
        *overrides the trainable target* (re-resolved + sidecar rewritten — see
        ``_maybe_override_target``). Off, or on a thin digest, it stays LOGGED-ONLY (sets no
        target): slope, not cliff.
        """
        from core import reflection_digest
        if not reflection_digest.render_digest_for_judge(persona_digest):
            return 0  # nothing judgeable; don't waste the swap's work
        sampling = config.overrides.branch_sampling
        br_temperature = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        br_top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        gate_ok = self._digest_maturity_gate(persona_digest)
        flip_active = bool(apply_flip and gate_ok)
        n = len(judge_jobs)
        if apply_flip and not gate_ok:
            flip_note = " [flip requested but digest too immature — logged-only]"
        elif flip_active:
            flip_note = " [criterion flip ACTIVE]"
        else:
            flip_note = " [logged-only]"
        self._emit(store, send_event_fn, run_id, "phase_started", phase="branch_judge",
                   message=f"Branch judge (clean base) — scoring {n} exchange(s)"
                           f" against who I'm becoming…{flip_note}")
        done = 0
        overrode = 0
        for job in judge_jobs:
            if store.is_stop_requested(run_id):
                break
            _judge_t0 = time.monotonic()
            judge = self._digest_judge_select(
                persona_digest, job["payload"], job["options"],
                generate_fn=generate_fn,
                branch_chooser_content_fn=branch_chooser_content_fn,
                br_temperature=br_temperature, br_top_p=br_top_p,
                chosen=job["chosen"], original_index=job["original_index"],
                tokenizer=tokenizer)
            # Per-judge wall time under its own phase, credited with the judge
            # response's generated tokens (0 on a failed pass) so the run-wide
            # tok/s readout isn't diluted by token-less seconds.
            if stats is not None:
                stats.record_phase(
                    "branch_judge", time.monotonic() - _judge_t0,
                    tokens=int((judge or {}).get("gen_tokens") or 0))
            if judge is None:
                continue
            done += 1
            pk = judge.get("pick_kind") or "unparseable"
            agree = judge.get("agrees_with_chooser")
            self._emit(
                store, send_event_fn, run_id, "branch_judged",
                session=job["session"], exchange_index=job["exchange_index"],
                exchange_total=job.get("exchange_total", 0),
                digest_select={"digest_version": persona_digest.get("version"),
                               "channel": "judge_clean_base", "judge": judge},
                message=(f"Judge (clean base): {pk}"
                         + ("" if agree is None else f", {'agrees' if agree else 'differs'} vs blind")),
            )
            if flip_active and self._maybe_override_target(
                    job, judge, run_id, store=store, send_event_fn=send_event_fn,
                    stats=stats):
                overrode += 1
                if stats is not None:
                    stats.note_chosen("judge_override")
        if stats is not None:
            stats.note_judge(judged=done, overrides=overrode)
            store.update_status(run_id, stats=stats.to_status())
        tail = f"; {overrode} target(s) overridden" if flip_active else ""
        self._emit(store, send_event_fn, run_id, "phase_done", phase="branch_judge",
                   message=f"Branch judge done — {done}/{n} judged on the clean base{tail}")
        return overrode

    @staticmethod
    def _digest_maturity_gate(persona_digest) -> bool:
        """True when the digest is mature enough to let the judge override training targets.

        Gates on the persisted **weighted recurrence** (recency- + tenure-discounted,
        counters netted) — NOT the raw distinct-session count, which the echo loop can
        inflate without bound, and NOT the model's free-text ``maturity`` label, which
        runs generous (it called recurrence-[2] themes "established"). Requires
        ``_FLIP_MIN_THEMES`` themes at weighted recurrence >= ``_FLIP_MIN_WEIGHTED``;
        a theme carrying no ``weighted_recurrence`` (pre-decay digest) counts as
        immature, so the flip stays dormant until the digest regenerates. A thin,
        stale, or contested corpus fails it → flip stays dormant. (See the note at
        ``_FLIP_MIN_WEIGHTED``.)
        """
        themes = ((persona_digest or {}).get("evidence") or {}).get("themes") or []
        mature = 0
        for t in themes:
            w = t.get("weighted_recurrence")
            if w is None:
                continue
            try:
                if float(w) >= _FLIP_MIN_WEIGHTED:
                    mature += 1
            except (TypeError, ValueError):
                continue
        return mature >= _FLIP_MIN_THEMES

    def _maybe_override_target(self, job, judge, run_id, *, store, send_event_fn,
                               stats: Optional[RunStats] = None) -> bool:
        """Re-resolve the trainable target with the judge's pick and rewrite the sidecar.

        Conservative — a no-op (returns False) when the judge pick is unparseable, equals the
        blind choice (no change), or resolves to an untrustworthy target
        (``revised_missing_ideal`` / empty). Reuses the *same* ``resolve_revision_target``
        Phase A used, just with ``chosen_index`` = the judge's pick, so CoT reattachment /
        kind handling is identical. Only the dialogue sidecar target changes; persona/anchor
        writes from Phase A stand.
        """
        pick = judge.get("pick_index")
        if pick is None or pick == job.get("chosen"):
            return False
        branch = dict(job["branch_block"])
        branch["chosen_index"] = pick
        try:
            target, target_source = resolve_revision_target(
                job["verdict"], job["ideal"], job["assistant_response"], branch,
                assistant_cot=job["assistant_cot"])
        except Exception:
            return False
        if target_source == "revised_missing_ideal" or not (target or "").strip():
            return False
        target_kind, target_generation = resolved_target_provenance(target_source, branch)
        # Carry the persona block only if the flip lands back on the IDEAL (its CoT was
        # persona-conditioned); a flip to a branch keeps "" (branch CoT is the original), and
        # write_verdict writes persona_context fresh, so the stale Phase-A block is cleared.
        override_persona_context = (
            job.get("ideal_persona_context", "") if judge.get("pick_kind") == "ideal" else ""
        )
        summary = {
            "verdict": job["verdict"],
            "target_source": target_source,
            "target_kind": target_kind,
            "target_generation": target_generation,
            "anchor": {"target": target, "prompt": job["user_prompt"],
                       "target_kind": target_kind,
                       "target_generation": target_generation,
                       "persona_context": override_persona_context,
                       "verdict": job["verdict"]},
        }
        try:
            write_revision_sidecar(
                self._chats_dir, summary,
                source_session=job["session"], exchange_index=job["exchange_index"],
                run_id=run_id, fallback_chats_dir=self._fallback_chats_dir)
        except Exception:
            return False
        # Special-interest cell: the judge flipped the trainable target to a *branch*.
        if stats is not None and judge.get("pick_kind") == "branch":
            stats.note_judge_branch_override()
        self._emit(store, send_event_fn, run_id, "target_overridden",
                   session=job["session"], exchange_index=job["exchange_index"],
                   pick_kind=judge.get("pick_kind"), target_source=target_source,
                   message=(f"Criterion flip: target → judge's {judge.get('pick_kind')} "
                            f"pick (Phase-A was {job.get('phase_a_target_source')})"))
        return True

    # ── fact placement (phase two, clean base) ─────────────────────────── #

    # Cap on candidate exchanges shown to the placement judge for one fact. Bounds the prompt
    # and keeps the letter-labelled option set within A–Z (plus the trailing "None"). When
    # more exchanges are in play the trigger pre-filter keeps the most lexically relevant.
    _FACT_PLACEMENT_OPTION_CAP = 20

    @staticmethod
    def _load_exchange_anchor_prompt() -> Optional[str]:
        """The per-exchange anchor prompt (ABOUT + TAGS), or None if unavailable.

        Missing file ⇒ the anchor pass is simply skipped, so an older checkout that
        pulled the code but not the prompt degrades to the previous behaviour.
        """
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "exchange_anchor_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _load_recollection_prompt() -> Optional[str]:
        """The revisit recollection prompt (RECOLLECTION + TRIGGER), or None if absent.

        Missing file ⇒ the pass is skipped, so an older checkout that pulled the code but
        not the prompt degrades to the previous behaviour (same contract as the anchor
        prompt above).
        """
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "recollection_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _load_user_notes_prompt() -> Optional[str]:
        """The per-session user-notes prompt (``{person}`` slot), or None if absent.

        Missing file ⇒ the pass is skipped, same degrade-to-previous-behaviour contract as
        the anchor and recollection prompts above.
        """
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "user_notes_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _load_self_notes_prompt() -> Optional[str]:
        """The per-session self-notes prompt (no slots), or None if absent.

        Missing file ⇒ the pass is skipped, the same degrade-to-previous-behaviour
        contract every optional pass prompt here has.
        """
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "self_notes_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _load_chat_facts_prompt() -> Optional[str]:
        """The per-session chat-facts protocol prompt (no slots), or None if absent.

        Missing file ⇒ the pass is skipped, the same degrade-to-previous-behaviour
        contract every optional pass prompt here has.
        """
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "chat_facts_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _load_fact_placement_prompt() -> Optional[str]:
        """The fact-placement judge prompt (``{fact}`` slot), or None if unavailable."""
        try:
            from training.reflections_path import default_prompts_dir
            text = (default_prompts_dir() / "fact_placement_prompt.txt").read_text(
                encoding="utf-8")
            return text if text.strip() else None
        except Exception:
            return None

    def _load_unhosted_facts(self) -> list[dict]:
        """Live ``[fact]`` anchors that have no host exchange yet — the placement universe.

        Reads the consolidation ledger directly (fold), so it needs no decay config: facts
        never advance a stage. A fact already carrying ``source_exchange`` is skipped (it has
        a home — including fully-baked facts, which keep theirs), matching persona's contract
        that a snapshot is written once and preserved across re-registers.
        """
        try:
            from training.ledger import ConsolidationLedger
            folded = ConsolidationLedger(self._consolidation_dir).fold()
        except Exception:
            return []
        out = []
        for rec in folded.values():
            if rec.get("type") != "fact":
                continue
            if rec.get("source_exchange"):
                continue
            if not (rec.get("content") or "").strip():
                continue
            out.append(rec)
        return out

    @staticmethod
    def _prefilter_candidates(fact: dict, candidates: list, limit: int) -> list:
        """Narrow *candidates* to the *limit* most lexically relevant to *fact* (cheap,
        model-free — the base model still makes the final call). Ranks by word overlap of the
        fact's trigger+content against each exchange's user turn + CoT; ties keep run order.
        Returns all candidates unranked when at/under the limit."""
        if len(candidates) <= limit:
            return candidates
        import re as _re
        def _words(s: str) -> set:
            return {w for w in _re.findall(r"\w+", (s or "").lower()) if len(w) > 2}
        key = _words(f"{fact.get('trigger') or ''} {fact.get('content') or ''}")
        if not key:
            return candidates[:limit]
        scored = []
        for i, c in enumerate(candidates):
            hay = _words(f"{c.get('user_prompt') or ''} {c.get('assistant_cot') or ''}")
            scored.append((len(key & hay), -i, c))
        scored.sort(reverse=True)
        return [c for _s, _i, c in scored[:limit]]

    @staticmethod
    def _fact_placement_content(candidates: list) -> str:
        """The user-turn content for the placement judge: the letter-labelled exchanges plus
        a trailing "None" option. The fact itself rides the system prompt (``{fact}`` slot).
        """
        lines = []
        for i, c in enumerate(candidates):
            letter = chr(ord("A") + i)
            up = _clip(c.get("user_prompt") or "", 400)
            cot = _clip(c.get("assistant_cot") or "", 900)
            lines.append(f"{letter}. USER: {up}\n   MY REASONING: {cot}")
        none_letter = chr(ord("A") + len(candidates))
        lines.append(f"{none_letter}. None of these — the fact does not shape any of them.")
        return "\n\n".join(lines)

    def _run_trigger_purge(self, run_id, *, config, store, send_event_fn):
        """Strip pasted-sentence recall cues out of the live trigger-indexed store.

        A ``[fact]``/``[recollection]`` is indexed by its **trigger**, so a malformed one is
        not untidiness — it is the record's entire retrieval key. ``_distill_resolved``
        stores the resolved ``[ask]`` as that key, and an ask Ava raised herself is one of
        her own conversational openers, so 27 facts on the live store ended up keyed on
        200–1745 characters of addressed speech. Keyed on a monologue a fact clusters with
        nothing, which is how one of them survived every dedup pass.

        Model-free and embedder-free (``core.trigger_hygiene`` is pure string work), which
        is why this sits outside the clean-base window and ahead of dedup — dedup unions the
        triggers of what it merges, so a cue left broken here is fused into a survivor and
        spreads. Writes through ``self._writer`` (STAGING, promoted by ``merge-rag``,
        discarded with a discarded run) like the dedup beside it. Nothing is deleted: the
        record is re-inserted under its own key with the bad cues gone, falling back to
        embedding on its content when no cue survives. Opt out with
        ``overrides.trigger_purge=False``.
        """
        if getattr(config.overrides, "trigger_purge", None) is False:
            return 0
        from core import trigger_hygiene
        try:
            mem = ReflectionMemory(
                self._memory_dir, fallback_memory_dir=self._fallback_memory_dir)
            plan = trigger_hygiene.plan_trigger_purge(mem.live_items())
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error", phase="trigger_purge",
                       message=f"Recall-cue purge skipped (memory unreadable): {e}")
            return 0
        if not plan:
            return 0

        self._emit(store, send_event_fn, run_id, "phase_started", phase="trigger_purge",
                   message=f"Recall-cue purge — {len(plan)} record(s) keyed on prose…")
        try:
            counts = self._writer.write_trigger_purge(plan)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error", phase="trigger_purge",
                       message=f"Recall-cue purge write failed (skipped): {e}")
            return 0

        report = trigger_hygiene.summarize(plan)
        self._emit(
            store, send_event_fn, run_id, "phase_done", phase="trigger_purge",
            message=(f"Recall-cue purge — {counts.get('purged', 0)} record(s) re-keyed, "
                     f"{counts.get('cues_dropped', 0)} cue(s) dropped, "
                     f"{counts.get('cleared', 0)} now recalled by content."),
            report={"trigger_purge": report},
            text="\n".join(
                f"fact: {r['content'][:110]}\n"
                + "\n".join(f"  drop cue ({d['reason']}): {d['cue'][:110]}"
                            for d in r["dropped"])
                + (f"\n  keep cue: {r['new_trigger']}" if r["new_trigger"]
                   else "\n  now recalled by its content")
                for r in report),
        )
        return counts.get("purged", 0)

    def _run_clean_base_fact_dedup(self, run_id, *, config, generate_fn, store,
                                   send_event_fn, embed_fn=None):
        """Collapse paraphrases in the live ``[fact]`` store, on the CLEAN base (the same
        swap as the branch judge, so it costs no extra reload).

        The fact-side counterpart of persona clustering, and the answer to a store that
        only ever grows: a ``[fact]`` is deduplicated at write time by EXACT content key,
        so every reflection run that re-notices one truth writes another live record for
        it. On a real corpus that reached 10+ restatements of a single evening, and since
        the chat block draws three same-topic slots, restatements arrive together.

        Runs LAST in the batch: the earlier passes are per-run work on this run's own
        material, while this is maintenance of the whole store, and running it after them
        means it also sees the facts THIS run just distilled (the fold below reads staging
        over live) — so a duplicate is collapsed in the run that created it rather than a
        run later.

        Writes through ``self._writer``, which points at the run's STAGING workspace, so
        the merge is promoted by ``merge-rag`` with everything else and a discarded run
        discards it too. That is the whole reason this belongs in the runner rather than
        beside the manual handler, which writes straight to live.

        *embed_fn* blocks the corpus by SUBJECT so a paraphrase actually shares a grouping
        call with its original — without it the pass is near-blind (``core.fact_dedup``).
        Its progress is reported per block: this is many sequential clean-base calls, and
        an unreported one is indistinguishable from a hang.
        """
        if getattr(config.overrides, "fact_dedup", None) is False:
            return 0
        from core import fact_dedup
        try:
            mem = ReflectionMemory(
                self._memory_dir, fallback_memory_dir=self._fallback_memory_dir)
            facts = [r for r in mem.live_items()
                     if r.get("kind") == "fact" and (r.get("content") or "").strip()]
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error", phase="fact_dedup",
                       message=f"Fact dedup skipped (memory unreadable): {e}")
            return 0
        if len(facts) < 2:
            self._emit(store, send_event_fn, run_id, "phase_done", phase="fact_dedup",
                       message=f"Fact dedup — {len(facts)} live fact(s); nothing to merge.")
            return 0

        self._emit(store, send_event_fn, run_id, "phase_started", phase="fact_dedup",
                   message=(f"Fact dedup (clean base) — {len(facts)} live fact(s), "
                            f"blocking by {'subject' if embed_fn else 'wording'}…"))
        try:
            groups = fact_dedup.cluster_facts(
                facts, generate_fn, embed_fn=embed_fn,
                # Its own event, not phase_progress: that one is the raw token stream, and
                # anything sent on it suppresses the phase_done body — which here is the
                # keep/drop report, the only part an operator needs.
                on_stage=lambda info: self._emit(
                    store, send_event_fn, run_id, "fact_dedup_progress", **info),
            )
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error", phase="fact_dedup",
                       message=f"Fact dedup grouping failed (skipped): {e}")
            return 0
        if not groups:
            self._emit(store, send_event_fn, run_id, "phase_done", phase="fact_dedup",
                       message="Fact dedup — nothing usable came back; no merges.")
            return 0

        merges = fact_dedup.plan_merges(groups)
        if not merges:
            self._emit(store, send_event_fn, run_id, "phase_done", phase="fact_dedup",
                       message=f"Fact dedup — {len(facts)} fact(s), no paraphrase groups.")
            return 0
        try:
            counts = self._writer.write_dedup(merges)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error", phase="fact_dedup",
                       message=f"Fact dedup write failed (skipped): {e}")
            return 0

        report = fact_dedup.summarize(merges)
        self._emit(
            store, send_event_fn, run_id, "phase_done", phase="fact_dedup",
            message=(f"Fact dedup — {counts.get('groups', 0)} group(s), "
                     f"{counts.get('evicted', 0)} paraphrase(s) evicted, "
                     f"{counts.get('reinserted', 0)} survivor(s) re-triggered "
                     f"({len(facts)} → {len(facts) - counts.get('evicted', 0)} facts)."),
            report={"dedup": report},
            text="\n".join(
                f"keep: {m['survivor']}\n" + "\n".join(f"  drop: {e}" for e in m["evicted"])
                for m in report),
        )
        return counts.get("evicted", 0)

    def _run_clean_base_fact_placement(self, fact_candidates, run_id, *, config,
                                       generate_fn, store, send_event_fn,
                                       stats: Optional[RunStats] = None, tokenizer=None):
        """Assign each unhosted ``[fact]`` a host exchange, on the CLEAN base (same swap as the
        branch judge). For each fact the judge reads the candidate exchanges' reasoning and
        picks the one that rests on it (or "None"); the pick is snapshotted onto the fact
        anchor as ``source_exchange`` — the persona contract, so ``train_cycle`` injects the
        fact into that exchange's CoT and later from-scratch builds read the snapshot
        without re-judging. Logged + ledger-write; never touches a dialogue target.
        """
        facts = self._load_unhosted_facts()
        prompt_tmpl = self._load_fact_placement_prompt()
        if not facts or not fact_candidates or not prompt_tmpl:
            return 0
        sampling = config.overrides.branch_sampling
        temp = sampling.temperature if sampling else _DEFAULT_TEMPERATURE
        top_p = sampling.top_p if sampling else _DEFAULT_TOP_P

        self._emit(store, send_event_fn, run_id, "phase_started", phase="fact_placement",
                   message=f"Fact placement (clean base) — {len(facts)} unhosted fact(s) "
                           f"vs {len(fact_candidates)} candidate exchange(s) "
                           f"(each fact restricted to its own chat)…")
        placed = 0
        for fact in facts:
            if store.is_stop_requested(run_id):
                break
            # Data locality (REBUILD §3): a fact may only be hosted on an exchange of its OWN
            # chat (the bundle it was distilled from), not any exchange in the run — the host
            # exchange and the fact live in one bundle. A fact whose own chat has no CoT-bearing
            # candidate stays unhosted (it waits in RAG until re-placed).
            local = [c for c in fact_candidates
                     if c.get("session") == fact.get("source_session")]
            if not local:
                continue
            options = self._prefilter_candidates(
                fact, local, self._FACT_PLACEMENT_OPTION_CAP)
            content = self._fact_placement_content(options)
            system_prompt = prompt_tmpl.replace("{fact}", (fact.get("content") or "").strip())
            try:
                resp = generate_fn(
                    content, system_prompt, temperature=temp, top_p=top_p,
                    max_new_tokens_setting=_DEFAULT_MAX_NEW_TOKENS,
                    disable_rag=True, disable_thinking=True)
            except Exception:
                continue
            if stats is not None:
                stats.record_phase("fact_placement", 0.0,
                                   tokens=_count_tokens(tokenizer, resp))
            # n_options = candidates + the trailing "None" slot; the None index is the last.
            pick, why = _parse_choice(resp, len(options) + 1)
            if pick is None or pick >= len(options):
                continue                       # unparseable or explicit "None" → no host
            host = options[pick]
            if self._place_fact(fact, host):
                placed += 1
                self._emit(
                    store, send_event_fn, run_id, "fact_placed",
                    session=host["session"], exchange_index=host["exchange_index"],
                    message=(f"Fact hosted → {host['session']}#{host['exchange_index']}"
                             + (f": {_clip(why, 160)}" if why else "")))
        if stats is not None:
            store.update_status(run_id, stats=stats.to_status())
        self._emit(store, send_event_fn, run_id, "phase_done", phase="fact_placement",
                   message=f"Fact placement done — {placed}/{len(facts)} fact(s) hosted")
        return placed

    def _place_fact(self, fact: dict, host: dict) -> bool:
        """Snapshot *host* onto the fact anchor as ``source_exchange`` (a re-register that
        ``fold`` preserves + keeps the stage/train_count). The whole placement write. The
        content_key is stable, so this annotates the existing anchor rather than forking it.
        """
        try:
            from training.ledger import ConsolidationLedger
            led = ConsolidationLedger(self._consolidation_dir)
            led.register_fact(
                content=fact["content"], item_type="fact",
                trigger=fact.get("trigger"), source_session=fact.get("source_session", ""),
                lang=fact.get("lang"),
                source_exchange={"source_session": host["session"],
                                 "exchange_index": host["exchange_index"]})
            return True
        except Exception:
            return False

    # ── helpers ────────────────────────────────────────────────────────── #

    def _build_branch_probe(
        self,
        *,
        run_id: str,
        filename: str,
        exchange_label: int,
        n_options: int,
        gen_peak: Optional[tuple[float, float]],
        chooser_peak: Optional[tuple[float, float]],
        chooser_prompt_tokens: int,
        option_tokens: int,
        gen_secs: Optional[float] = None,
        choose_secs: Optional[float] = None,
    ) -> dict:
        """Assemble the per-exchange VRAM/token report, log it, and return it.

        The block lands on the branch_done event and the persisted branch block.
        It answers two questions empirically: (1) is the chooser the VRAM peak (vs
        generation), and (2) how much of the chooser prompt is option text — the
        only part serial/pairwise judging could shrink — vs the shared context it
        couldn't. A high context share means "serialize the branches" would add
        passes and wall-clock for little peak relief.
        """
        # Remainder = dialogue context + chooser scaffolding; the part pairwise
        # judging shares across every comparison and therefore cannot reduce.
        context_tokens = max(0, chooser_prompt_tokens - option_tokens)
        option_frac = (option_tokens / chooser_prompt_tokens) if chooser_prompt_tokens else 0.0
        probe = {
            "n_options": n_options,
            "chooser_prompt_tokens": chooser_prompt_tokens,
            "chooser_option_tokens": option_tokens,
            "chooser_context_tokens": context_tokens,
            "chooser_option_frac": round(option_frac, 3),
            "gen_peak_alloc_gb": round(gen_peak[0], 2) if gen_peak else None,
            "gen_peak_reserved_gb": round(gen_peak[1], 2) if gen_peak else None,
            "chooser_peak_alloc_gb": round(chooser_peak[0], 2) if chooser_peak else None,
            "chooser_peak_reserved_gb": round(chooser_peak[1], 2) if chooser_peak else None,
            "gen_secs": round(gen_secs, 1) if gen_secs is not None else None,
            "choose_secs": round(choose_secs, 1) if choose_secs is not None else None,
        }
        gp = f"{probe['gen_peak_reserved_gb']}" if probe['gen_peak_reserved_gb'] is not None else "n/a"
        cp = f"{probe['chooser_peak_reserved_gb']}" if probe['chooser_peak_reserved_gb'] is not None else "n/a"
        gs = f"{probe['gen_secs']}s" if probe['gen_secs'] is not None else "n/a"
        cs = f"{probe['choose_secs']}s" if probe['choose_secs'] is not None else "n/a"
        print(
            f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[branch probe] run={run_id} {filename}#{exchange_label} "
            f"opts={n_options} | gen peak {gp} GB / {gs}, "
            f"chooser peak {cp} GB / {cs} | chooser prompt "
            f"{chooser_prompt_tokens} tok = {option_tokens} option "
            f"({option_frac:.0%}) + {context_tokens} context",
            flush=True,
        )
        return probe

    def _session_sidecar(self):
        """A ChatSidecar bound to this runner's chats dir (+ staging fallback)."""
        from core.chat_sidecar import ChatSidecar
        return ChatSidecar(self._chats_dir, fallback_chats_dir=self._fallback_chats_dir)

    def _sidecar_is_frozen(self, filename: str) -> bool:
        """True once *filename*'s sidecar is stamped reflected (reflect-once, §3)."""
        try:
            return self._session_sidecar().is_reflected(filename)
        except Exception:
            return False

    def _sidecar_is_chat_reflected(self, filename: str) -> bool:
        """True once *filename*'s sidecar carries the background per-chat stamp
        (`chat_reflected`) — stage one of the two-stage freeze."""
        try:
            return self._session_sidecar().is_chat_reflected(filename)
        except Exception:
            return False

    @staticmethod
    def _is_unanswered_outreach(session: dict) -> bool:
        """True for an Ava-initiated outreach chat the user never replied to.

        Such a session holds only her opener (exchange 0, whose user_prompt is the
        synthetic "(initiative)" stimulus that is masked from training anyway) — there
        is no real dialogue turn to consolidate or revise, so reflection skips it. It
        is left un-frozen: a later reply appends a real exchange and makes it reflectable.
        """
        if (session.get("initiated_by") or "").strip() != "ava":
            return False
        exchanges = session.get("exchanges") or []
        return len(exchanges) <= 1

    def _record_conversation_worklog(self, filename: str, session: dict) -> None:
        """Record this freshly-frozen chat as one first-person worklog episode, closing the
        reach-out thread that opened it. The entry itself is `chat_worklog.record_conversation`
        (shared with the background per-chat freeze — see `core.chat_worklog`); this only
        supplies the gist, read from the consolidation summary this run just wrote to the
        session's sidecar. Best-effort: a worklog hiccup must never disturb a reflection."""
        try:
            from core.chat_worklog import record_conversation
            gist = ""
            try:
                gist = self._session_sidecar().summary_text(filename)
            except Exception:
                gist = ""
            record_conversation(filename, session or {}, gist)
        except Exception:
            pass

    def _mark_session_reflected(self, filename: str) -> None:
        """Stamp *filename*'s sidecar as fully reflected (reflect-once, §3). Best-effort."""
        try:
            self._session_sidecar().mark_reflected(filename)
        except Exception:
            pass

    def _mark_session_chat_reflected(self, filename: str) -> None:
        """Stamp *filename*'s sidecar per-chat reflected (two-stage freeze, stage one) —
        the background pass's freeze. Best-effort."""
        try:
            self._session_sidecar().mark_chat_reflected(filename)
        except Exception:
            pass

    def _locked_exchanges(self, filename: str) -> set:
        """Indices of *filename*'s human-validated (sidecar ``locked``) exchanges.

        A locked exchange is an operator-reviewed manual regeneration (Training review →
        Regenerate → Apply). The revision pass skips it — no re-derivation, no write — in
        BOTH normal re-reflection and a revisit, so a fresh pass over the original (corrupt)
        transcript can never overwrite the reviewed target. Read from the **live** sidecar
        directly (where Apply stamps the lock), not the staging copy, so a stale
        continue-staging workspace can't shadow it. Best-effort → empty set on any error."""
        live_dir = self._fallback_chats_dir or self._chats_dir
        try:
            from core.chat_sidecar import ChatSidecar
            return ChatSidecar(live_dir).locked_exchange_indices(filename)
        except Exception:
            return set()

    def _banned_exchanges(self, filename: str) -> set:
        """Indices of *filename*'s exchanges banned from training (sidecar ``banned``).

        The operator's verdict that this turn is not worth learning from (Training review →
        Ban). Revision's whole per-exchange product is a trainable target, so re-deriving one
        here would be paid-for work — a generation, and the branch generations behind it —
        that the build then throws away. Persona formation rides the same pass, which is the
        other half of the reason to skip: an exchange declared not worth learning from should
        not be shaping her identity through a side channel either.

        The retrieval-side passes (anchors, the consolidation summary) deliberately do NOT
        skip it: until "Rewrite history" deletes it, the exchange is still part of the
        conversation and still legitimately recallable. Read from the **live** sidecar, like
        the lock. Best-effort → empty set on any error."""
        live_dir = self._fallback_chats_dir or self._chats_dir
        try:
            from core.chat_sidecar import ChatSidecar
            return ChatSidecar(live_dir).banned_exchange_indices(filename)
        except Exception:
            return set()

    def _load_sessions(self, filenames: list[str]) -> list[tuple[str, dict]]:
        """Load session JSON files from chats_dir; silently skips missing ones."""
        result = []
        for fn in filenames:
            path = self._chats_dir / fn
            if not path.exists() and self._fallback_chats_dir is not None:
                path = self._fallback_chats_dir / fn
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
                result.append((fn, session))
            except Exception:
                pass
        return result

    def _plan_persona_digest(self, run_id, store, send_event_fn,
                             *, force: bool = False) -> dict:
        """Model-free half of the digest pass: is a regeneration warranted at all?

        Folds the *live* anchor ledger only (committed personas) — a staged run's own
        just-distilled personas are deliberately excluded until they commit, so the
        digest never summarizes state that a later discard would erase.

        This is deliberately answered BEFORE the clean-base window opens, because the
        clustering half now runs inside that window: an unchanged run must not pay an
        adapter swap it has no work for. *force* (the Sleep tab's "Regen persona"
        checkbox) regenerates even when the evidence fingerprint is unchanged.

        Returns ``reflection_digest.plan_digest``'s dict; on any failure returns a
        should_run=False plan, since the digest is best-effort and must never sink a run.
        """
        from core import reflection_digest
        from training.reflections_path import consolidation_dir, persona_dir

        try:
            plan = reflection_digest.plan_digest(
                consolidation_dir(), persona_dir=persona_dir(), force=force)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="persona_digest",
                       message=f"Persona digest planning error (skipped): {e}")
            return {"should_run": False, "reason": f"planning error: {e}",
                    "base": [], "raw_fp": None, "persona_count": 0}
        if not plan.get("should_run"):
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="persona_digest", report={"status": "skipped",
                                                       "reason": plan.get("reason", "")},
                       message=f"Persona digest skipped — {plan.get('reason', '')}")
        return plan

    def _cluster_persona_digest(self, plan, run_id, generate_fn, store, send_event_fn,
                                *, map_reduce: bool = True,
                                polarity: bool = True) -> list:
        """Cluster the planned persona evidence into themes — the CLEAN-BASE half.

        Grouping paraphrases is an evaluation ("do these two say the same thing?"), not
        Ava's expression, so it belongs adapter-off beside the branch judge and fact
        placement, and the caller runs it inside the run's single ``clean_base_ctx``
        window. Best-effort: on failure the caller simply gets no themes and the digest
        is skipped for this run.

        With *polarity* (default) the grouped themes are then screened for members that
        OPPOSE their representative (``reflection_digest.screen_theme_polarity``, same
        window — it is an evaluation too): an opposing member is split into its own
        theme, and its sessions are written as ledger COUNTER ops against the theme it
        was mistaken for affirming — the persuasion channel's second producer, beside
        next-turn user pushback (:meth:`_write_counter_evidence`). Counters go to the
        run's own ledger dir like every other runner write, so a discarded run discards
        them; the digest fold dedups ``(key, session)`` counter pairs, so a later regen
        re-planning the same split is idempotent in effect.
        """
        from core import reflection_digest

        base = plan.get("base") or []
        counter_ops = {"themes": 0, "ops": 0}

        def _counter_sink(entries: list) -> None:
            from training.ledger import ConsolidationLedger
            try:
                led = ConsolidationLedger(self._consolidation_dir)
                for entry in entries:
                    n = 0
                    for sess in entry.get("sessions") or []:
                        n += led.counter(
                            [entry["key"]], source_session=sess,
                            reason=("polarity: opposed by \""
                                    + (entry.get("opposed") or [""])[0][:120] + "\""),
                            run_id=run_id)
                    counter_ops["themes"] += 1
                    counter_ops["ops"] += n
                    self._emit(
                        store, send_event_fn, run_id, "persona_polarity",
                        message=(f"Polarity: split {len(entry.get('opposed') or [])} "
                                 "opposing statement(s) out of theme "
                                 f"\"{(entry.get('content') or '')[:80]}\" "
                                 f"({n} counter op(s))"))
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "pass_warning",
                           phase="persona_cluster",
                           message=f"Polarity counter write failed (skipped): {e}")

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="persona_cluster",
                   message=f"Persona digest: grouping {len(base)} statement(s) "
                           "into themes on the clean base")
        cl_stats: dict = {}
        try:
            evidence = reflection_digest.cluster_for_digest(
                base, generate_fn=generate_fn, map_reduce=map_reduce,
                polarity=polarity, counter_sink=_counter_sink, stats=cl_stats,
                on_stage=lambda info: self._emit(
                    store, send_event_fn, run_id, "persona_cluster_progress", **info))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="persona_cluster",
                       message=f"Persona clustering error (skipped): {e}")
            return []
        merged = sum(1 for e in evidence if e.get("cluster_size", 1) > 1)
        biggest = max([e.get("cluster_size", 1) for e in evidence] or [0])
        pol = cl_stats.get("polarity") or {}
        report = {"statements": len(base), "themes": len(evidence),
                  "merged_themes": merged, "largest_theme": biggest}
        tail = ""
        if pol:
            report["polarity"] = pol
            report["counter_ops"] = counter_ops["ops"]
            if pol.get("themes_split"):
                tail = (f"; polarity split {pol['themes_split']} theme(s), "
                        f"{counter_ops['ops']} counter op(s)")
        self._emit(store, send_event_fn, run_id, "phase_done", phase="persona_cluster",
                   report=report,
                   message=f"Persona digest: {len(base)} statement(s) → "
                           f"{len(evidence)} theme(s) ({merged} merged, "
                           f"largest {biggest})" + tail)
        return evidence

    def _synthesize_persona_digest(self, plan, evidence, run_id, generate_fn,
                                   store, send_event_fn, *,
                                   gate_screen: bool = True) -> dict:
        """Write + version the self-portrait from clustered evidence — the ADAPTER half.

        Runs after the clean-base window has closed, because this is Ava's *authorship*
        (the clustering that fed it was an evaluation). Best-effort and write-only; any
        failure is reported and swallowed by the caller. *gate_screen* (threaded from
        the same ``persona_polarity`` override as the theme polarity screen — both are
        direction hygiene) lets ``synthesize_digest`` clear a disposition ``not:`` line
        that escalates the habit instead of restraining it, before anything renders it.
        """
        from core import reflection_digest
        from training.reflections_path import persona_dir

        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="persona_digest",
                   message="Persona digest: synthesizing self-portrait")
        summary = reflection_digest.synthesize_digest(
            evidence, generate_fn=generate_fn, persona_dir=persona_dir(),
            run_id=run_id, raw_fp=plan.get("raw_fp"), gate_screen=gate_screen)
        if summary.get("status") == "written":
            msg = f"Persona digest written — {summary.get('counts')}"
            gates = summary.get("gate_screen") or {}
            if gates.get("flagged"):
                names = ", ".join(gates.get("names") or [])
                msg += (f"; cleared {gates['flagged']} escalating not:-gate(s)"
                        + (f" ({names})" if names else ""))
        else:
            msg = f"Persona digest skipped — {summary.get('reason', '')}"
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="persona_digest", report=summary, message=msg)
        return summary

    def _plan_user_portraits(self, run_id, store, send_event_fn,
                             *, force: bool = False) -> list[dict]:
        """Model-free half of the user-portrait pass: whose portrait needs regenerating?

        The per-person mirror of :meth:`_plan_persona_digest`, and answered for the same
        reason at the same point — BEFORE the clean-base window opens, so a run with no
        new evidence about anybody never pays for a swap it has no work for.

        Reads the **live** memory store only, never this run's staging — the same rule the
        persona digest follows, and for the same reason: the portrait file is written
        straight to the live users dir, so folding it from staged impressions would leave
        a portrait standing on evidence that a later discard erases. The effect is a
        one-run lag (this run's readings shape the portrait on the next run), which is
        exactly the digest's cadence.
        """
        from core import user_digest
        from training.reflections_path import users_dir

        # Live is the fallback dir when this runner writes to staging, and its own dir
        # when it writes directly to live (the unstaged/CLI path).
        live_memory = self._fallback_memory_dir or self._memory_dir
        try:
            plans = user_digest.plan_portraits(
                live_memory, users_dir=users_dir(), force=force)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="user_portrait",
                       message=f"User-portrait planning error (skipped): {e}")
            return []
        if not plans:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="user_portrait",
                       report={"status": "skipped", "reason": "no material change"},
                       message="User portraits skipped — no material change")
        return plans

    def _cluster_user_portraits(self, plans, run_id, generate_fn, store, send_event_fn
                                ) -> dict[str, list]:
        """Cluster each planned person's evidence into themes — the CLEAN-BASE half.

        Grouping paraphrases is an evaluation, so it belongs adapter-off beside the branch
        judge, fact placement and persona clustering, inside the run's single
        ``clean_base_ctx`` window. Best-effort per person: one person's clustering failure
        leaves them without a portrait this run and does not touch anybody else's.

        Returns ``{slug: evidence}``.
        """
        from core import user_digest

        out: dict[str, list] = {}
        for plan in plans:
            if store.is_stop_requested(run_id):
                break
            person = plan["person"]
            self._emit(store, send_event_fn, run_id, "phase_started",
                       phase="user_cluster", person=person,
                       message=(f"User portrait: grouping {len(plan['items'])} "
                                f"observation(s) about {person} on the clean base"))
            try:
                evidence = user_digest.cluster_for_portrait(
                    plan["items"], generate_fn=generate_fn)
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "phase_error",
                           phase="user_cluster", person=person,
                           message=f"User-portrait clustering error ({person}): {e}")
                continue
            out[plan["slug"]] = evidence
            merged = sum(1 for e in evidence if e.get("cluster_size", 1) > 1)
            self._emit(store, send_event_fn, run_id, "phase_done", phase="user_cluster",
                       person=person,
                       report={"observations": len(plan["items"]),
                               "themes": len(evidence), "merged_themes": merged},
                       message=(f"User portrait: {len(plan['items'])} observation(s) "
                                f"about {person} → {len(evidence)} theme(s) "
                                f"({merged} merged)"))
        return out

    def _synthesize_user_portraits(self, plans, evidence_by_slug, run_id, generate_fn,
                                   store, send_event_fn) -> int:
        """Write each person's portrait from clustered evidence — the ADAPTER half.

        Runs after the clean-base window closes: a portrait is Ava's own reading of
        someone, in her voice, which is precisely what the frozen base must not author
        (the clustering that fed it was an evaluation, so that half ran adapter-off).
        Best-effort per person. Returns how many portraits were written.
        """
        from core import user_digest
        from training.reflections_path import users_dir

        written = 0
        for plan in plans:
            if store.is_stop_requested(run_id):
                break
            evidence = evidence_by_slug.get(plan["slug"]) or []
            if not evidence:
                continue
            person = plan["person"]
            self._emit(store, send_event_fn, run_id, "phase_started",
                       phase="user_portrait", person=person,
                       message=f"User portrait: synthesizing who {person} is")
            try:
                summary = user_digest.synthesize_portrait(
                    person, evidence, generate_fn=generate_fn, users_dir=users_dir(),
                    run_id=run_id, raw_fp=plan.get("raw_fp"))
            except Exception as e:
                self._emit(store, send_event_fn, run_id, "phase_error",
                           phase="user_portrait", person=person,
                           message=f"User-portrait error ({person}): {e}")
                continue
            body = ""
            if summary.get("status") == "written":
                written += 1
                msg = f"User portrait written — {person}: {summary.get('counts')}"
                # The portrait itself rides the event so the Sleep/Activity log shows what
                # she actually concluded, not just that something was written.
                summary = dict(summary)
                summary["rendered"] = user_digest.render_portrait_plain(
                    dict(summary.get("portrait") or {}, person=person))
                # (The written portrait reaches the activity mirror through `report`'s
                # `rendered` key — see `reflection_config._render_report_items` — so it
                # deliberately does NOT also ride `text`.)
            else:
                msg = (f"User portrait skipped ({person}) — "
                       f"{summary.get('reason', '')}")
                # An unparseable synthesis carries its raw generation (see
                # `user_digest.synthesize_portrait`); render it the same way the anchor
                # and user-notes warnings render theirs, so the skip is diagnosable
                # rather than a reason string. This pass does not stream, so the event
                # is the ONLY place its output is ever visible.
                if summary.get("raw"):
                    body = _pass_output_debug(
                        summary["raw"], "no section parsed ({} chars raw)".format(
                            summary.get("raw_chars", 0)),
                        truncated=summary.get("truncated"))
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="user_portrait", person=person, report=summary,
                       text=body, message=msg)
        return written

    def _plan_self_portrait(self, run_id, store, send_event_fn,
                            *, force: bool = False) -> Optional[dict]:
        """Model-free half of the outside-view portrait: does it need regenerating?

        Same shape and same placement as :meth:`_plan_user_portraits` — BEFORE the
        clean-base window, so a run with no new observations never pays for a swap it has
        no work for — and it reads **live** memory only, never this run's staging, because
        the portrait file is written straight to the live users dir.
        """
        from core import self_portrait
        from training.reflections_path import users_dir

        live_memory = self._fallback_memory_dir or self._memory_dir
        try:
            plan = self_portrait.plan_portrait(
                live_memory, users_dir=users_dir(), force=force)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="self_portrait",
                       message=f"Self-portrait planning error (skipped): {e}")
            return None
        if not plan:
            self._emit(store, send_event_fn, run_id, "phase_done",
                       phase="self_portrait",
                       report={"status": "skipped", "reason": "no material change"},
                       message="Outside-view portrait skipped — no material change")
        return plan

    def _cluster_self_portrait(self, plan, run_id, generate_fn, store, send_event_fn
                               ) -> list:
        """Cluster the observations into themes — the CLEAN-BASE half.

        Grouping paraphrases is an evaluation, so it belongs adapter-off beside the branch
        judge, fact placement, persona clustering and fact dedup, inside the run's single
        ``clean_base_ctx`` window. Best-effort: a failure leaves no portrait this run.
        """
        from core import self_portrait

        items = plan.get("items") or []
        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="self_cluster",
                   message=(f"Outside-view portrait: grouping {len(items)} "
                            f"observation(s) on the clean base"))
        try:
            evidence = self_portrait.cluster_evidence(items, generate_fn=generate_fn)
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="self_cluster",
                       message=f"Outside-view clustering error: {e}")
            return []
        merged = sum(1 for e in evidence if e.get("cluster_size", 1) > 1)
        self._emit(store, send_event_fn, run_id, "phase_done", phase="self_cluster",
                   report={"observations": len(items), "themes": len(evidence),
                           "merged_themes": merged},
                   message=(f"Outside-view portrait: {len(items)} observation(s) → "
                            f"{len(evidence)} theme(s) ({merged} merged)"))
        return evidence

    def _synthesize_self_portrait(self, plan, evidence, run_id, generate_fn,
                                  store, send_event_fn) -> int:
        """Write the outside-view portrait — the ADAPTER half.

        Runs after the clean-base window closes: reading her own transcripts back is still
        her reading, in her voice, and the frozen base's job was the evaluation that fed
        it. Returns 1 when a portrait was written.
        """
        from core import self_portrait
        from training.reflections_path import users_dir

        if not evidence:
            return 0
        self._emit(store, send_event_fn, run_id, "phase_started",
                   phase="self_portrait",
                   message="Outside-view portrait: synthesizing how she comes across")
        try:
            summary = self_portrait.synthesize_portrait(
                evidence, generate_fn=generate_fn, users_dir=users_dir(),
                run_id=run_id, raw_fp=plan.get("raw_fp"))
        except Exception as e:
            self._emit(store, send_event_fn, run_id, "phase_error",
                       phase="self_portrait",
                       message=f"Outside-view portrait error: {e}")
            return 0
        body = ""
        written = 0
        if summary.get("status") == "written":
            written = 1
            msg = f"Outside-view portrait written — {summary.get('counts')}"
            # The portrait rides the event so the Sleep tab and the activity journal show
            # what she actually concluded. This is the artifact's PRIMARY surface: nothing
            # injects it, by design, so the run log and the Debug tab are where it is read.
            summary = dict(summary)
            summary["rendered"] = self_portrait.render_portrait_plain(
                dict(summary.get("portrait") or {}))
        else:
            msg = f"Outside-view portrait skipped — {summary.get('reason', '')}"
            if summary.get("raw"):
                body = _pass_output_debug(
                    summary["raw"], "no section parsed ({} chars raw)".format(
                        summary.get("raw_chars", 0)),
                    truncated=summary.get("truncated"))
        self._emit(store, send_event_fn, run_id, "phase_done",
                   phase="self_portrait", report=summary, text=body, message=msg)
        return written

    def _build_open_questions_block(self) -> str:
        """Render the OPEN QUESTIONS preamble from live reflection memory.

        A staged run's *memory_dir* starts empty, so the fold reads the live
        ``fallback_memory_dir`` first — otherwise no prior open question is ever
        re-posed during the (default) staged run, and a question could never be
        resolved/evicted.
        """
        try:
            memory = ReflectionMemory(
                self._memory_dir, fallback_memory_dir=self._fallback_memory_dir
            )
            questions = [
                q for q in memory.open_questions()
                if (q.get("content") or "").strip()
            ]
        except Exception:
            return ""
        if not questions:
            return ""
        lines = [
            "OPEN QUESTIONS — things you wanted to know from earlier reflections that "
            "were never settled. As you read this session, watch for anything that "
            "answers them, even said in passing. List any that are answered under "
            "RESOLVED; leave the rest open.",
            "",
        ]
        for q in questions:
            content = q["content"].strip()
            seen = (q.get("source_session") or "").strip()
            lines.append(
                f"- {content}" + (f"  (first seen: {seen})" if seen else "")
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _emit(
        store: ReflectionRunStore,
        send_event_fn: Optional[Callable],
        run_id: str,
        event_type: str,
        **kwargs,
    ) -> None:
        event = store.append_event(run_id, event_type, **kwargs)
        if event and send_event_fn:
            try:
                send_event_fn(event)
            except Exception:
                pass
