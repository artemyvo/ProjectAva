"""Reflection run configuration dataclasses, override validation, and run registry.

Owns both the config layer and the durable run store — they are tightly coupled:
the store manages config objects directly and splitting them adds no benefit until
a clear boundary emerges.

Phase 2 of the reflection decoupling plan — see decoupling_plan.md.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core import activity_log

# Reflection event types coarse-mirrored into the unified activity journal
# (core.activity_log). For an OPERATOR run the Sleep tab keeps the full detailed console
# (per-branch, per-exchange), so the box-wide Activity tab only needs phase grain and the
# noisy per-pass events (pass_warning / branch_candidate / counter_*) are deliberately
# excluded. Mirroring here — the one sink every reflection event flows through — catches
# the downstream merge-rag / commit-training / archive markers that never route through
# send_event_fn too.
# Events mirrored into the box-wide activity journal, at Sleep-tab detail for EVERY run
# regardless of who triggered it (2026-07-28).
#
# This was two grains: a background run (source == "idle") mirrored the pass's actual
# output because the Activity tab is its only window, while an operator run mirrored bare
# phase markers "without duplicating the Sleep tab". That reasoning does not survive
# contact with use — the Sleep console shows only the run it launched, so anything an
# operator wanted to WATCH (now: the per-exchange anchors) was invisible in the one view
# that is always live, and the Activity tab looked broken next to the Sleep textbox. The
# journal is the box's log; a reflection run is the loudest thing on the box; it belongs
# there in full.
#
# Bounds are unchanged and are what make one grain affordable: the raw token stream
# (`phase_progress`) is still NEVER mirrored, and each attached body is clipped at
# _MIRROR_TEXT_CAP. A long operator run can still evict older entries from
# `activity_log`'s 2000-event ring — the on-disk journal keeps them, and the client's
# "Hide reflection detail" box collapses the bodies for an overview.
_ACTIVITY_MIRROR_EVENTS = {
    "run_started", "run_failed", "phase_started", "phase_done", "phase_error",
    "session_started", "session_finalized", "session_skipped",
    "branch_started", "branch_done", "branch_skipped", "ask_resolved",
    # Both failure grains: `pass_warning` carries the recoverable ones an operator needs
    # while judging a new pass (an unparseable anchor, a retried verdict), which were
    # previously invisible in this journal at either grain.
    "pass_error", "pass_warning",
}
_MIRROR_TEXT_CAP = 4000


def _render_report_items(report: dict) -> str:
    """Render a consolidation phase_done `report`'s distilled items as Sleep-tab-style
    lines (`[fact] …` / `[ask:user] …` / `[resolved] …`). Consolidation streams its pass
    text as phase_progress deltas (never mirrored — flood), so for the activity mirror
    the parsed report IS the reflection data. Empty string for a report with no item
    lists (e.g. a revision report, which carries counts and rides the `text` path)."""
    lines: list[str] = []
    try:
        for w in report.get("weights") or []:
            if isinstance(w, dict):
                lines.append(f"[{w.get('weights_kind') or 'fact'}] "
                             f"{(w.get('content') or '').strip()}")
        for r in report.get("rag") or []:
            if isinstance(r, dict):
                kind = r.get("kind") or "?"
                tag = f"[{kind}:{r['ask_kind']}]" if r.get("ask_kind") else f"[{kind}]"
                lines.append(f"{tag} {(r.get('content') or '').strip()}")
        for res in report.get("resolved") or []:
            if isinstance(res, dict):
                q = (res.get("question") or "").strip()
                a = (res.get("answer") or "").strip()
                lines.append(f"[resolved] {q} → {a}" if a else f"[resolved] {q}")
        # The user-notes pass now puts its raw generation on `text`, which takes
        # precedence over this — its readings are reached here only for a dry run (which
        # emits `report` alone) and for events written before that change.
        for imp in report.get("impressions") or []:
            if isinstance(imp, str) and imp.strip():
                lines.append(f"[impression] {imp.strip()}")
        # A written user portrait carries the whole rendered portrait, which is exactly
        # what an operator watching this tab wants to read (and what the _MIRROR_TEXT_CAP
        # above exists to bound).
        rendered = report.get("rendered")
        if isinstance(rendered, str) and rendered.strip():
            lines.append(rendered.strip())
    except Exception:
        return ""
    return "\n".join(ln for ln in lines if ln.strip("[] "))


# ── reflection-pass loop guards ───────────────────────────────────────────────
# DISABLED (both None). These were meant to stop a reflection CoT looping a whole
# sentence to context exhaustion, but they corrupt normal reflective prose:
#   * no_repeat_ngram_size=3 forbids *any* 3-token sequence from repeating, which
#     is catastrophic for analysis that legitimately reuses phrases ("reverse-
#     engineer", "protocol", the VERDICT:/PERSONA: structure) — every banned n-gram
#     forces an off-distribution token ("reverse0-engineered", stray Cyrillic/CJK
#     on bilingual sessions), garbling the whole pass and defeating the parser.
#   * repetition_penalty=1.15 compounds it: over a long pass it penalizes every
#     already-seen content word, flattening the distribution toward those same
#     off-distribution tokens.
# The loop they targeted is a long *sentence* repeat; an n-gram ban is the wrong
# tool for it. Loop defense belongs in a repetition-detecting StoppingCriteria
# that fires only on genuine pathology (see backend); the retry-on-unparseable +
# truncation flag already catch a runaway that slips through.
REFLECT_REPETITION_PENALTY = None
REFLECT_NO_REPEAT_NGRAM = None


# ── sampling / budget / override dataclasses ──────────────────────────────────

@dataclass
class ReflectionSampling:
    temperature: float
    top_p: float
    max_new_tokens_setting: str


@dataclass
class ReflectionChooserBudget:
    max_total_tokens: int
    context_fraction: float
    output_tokens: int
    safety_tokens: int


@dataclass
class ReflectionRunOverrides:
    sleep_prompt: Optional[str] = None
    revision_prompt: Optional[str] = None
    branch_prompt: Optional[str] = None
    sleep_sampling: Optional[ReflectionSampling] = None
    revision_sampling: Optional[ReflectionSampling] = None
    branch_sampling: Optional[ReflectionSampling] = None
    chooser_budget: Optional[ReflectionChooserBudget] = None
    disable_rag_for_branch_choice: Optional[bool] = None
    chunk_budget_frac: Optional[float] = None
    chars_per_token_hint: Optional[float] = None
    # Force the persona-digest pass to regenerate even when the evidence fingerprint
    # is unchanged (the digest is otherwise a no-op on an unchanged run). Used by the
    # Sleep tab's "Regen persona" checkbox.
    force_persona_digest: Optional[bool] = None
    # Phase-two criterion flip: let the clean-base branch judge SET the trainable target
    # (override the blind chooser), gated on numeric-recurrence digest maturity. Off →
    # the judge stays logged-only. Reversible. Used by the Sleep tab's "Apply judge" box.
    apply_branch_judge: Optional[bool] = None
    # Cluster persona evidence with the blocked map-reduce pass (core.persona_cluster)
    # rather than the historical single flat grouping call. Default on; set False as a
    # kill-switch to revert. The flat call cannot fit a large persona set in one prompt
    # (see AVA_CHANGELOG 2026-07-26), so off is a fallback, not a neutral choice.
    persona_cluster_mapreduce: Optional[bool] = None
    # Screen each merged persona theme for DIRECTION on the same clean-base model that
    # grouped it: a member that OPPOSES the theme's representative ("I resist X" swallowed
    # by an "I enjoy X" theme) is split into its own theme, and its sessions are written
    # as ledger COUNTER ops against the representative (the persuasion channel) instead of
    # counting as affirmation votes. The SAME switch also governs the digest's
    # disposition-gate escalation screen (reflection_digest.screen_disposition_gates —
    # a `not:` line that extends the habit instead of restraining it is cleared before
    # anything renders it): both are direction hygiene on the same artifact. Default on;
    # set False to keep the historical polarity-blind behaviour of both.
    # See reflection_digest.screen_theme_polarity.
    persona_polarity: Optional[bool] = None
    # Skip the branching phase entirely: no branch generation, no branch judge — the
    # trainable target is just the kept original (on a `keep` verdict) or the revised
    # IDEAL (on a `revise`), resolved directly. Equivalent to omitting the branch
    # callbacks, but as an explicit per-run toggle. Used by the Sleep tab's
    # "Skip branching" checkbox.
    skip_branching: Optional[bool] = None
    # Generate a per-exchange retrieval ANCHOR (one-line descriptor + tags) after
    # revision, written to the sidecar's `anchors` map. Default on; set False to skip
    # the pass (it costs one short generation per anchorable exchange). Producer-side
    # only — nothing retrieves on anchors yet, so turning it off changes no behaviour
    # beyond not accumulating them. See core.exchange_anchor.
    exchange_anchors: Optional[bool] = None
    # Write a `[recollection]` at the end of a REVISIT — what Ava now makes of the
    # re-read conversation, as a RAG-only memory kind dated by the reading rather than
    # by the chat. Default on; ignored outside a revisit run (a first-time reflection has
    # nothing to look back on). Set False to skip the pass (one generation per session).
    # See ReflectionWriter.write_recollection.
    recollection: Optional[bool] = None
    # Run the per-session user-notes pass — "what did I learn about this person?" —
    # writing `[impression]` records that fold into that person's standing portrait.
    # Default on; set False to skip the pass (one generation per session). Production
    # only: retrieval of existing impressions is `impressions.enabled` in server_config,
    # and the portrait fold is `user_portrait.enabled`. See core.user_digest.
    user_notes: Optional[bool] = None
    # Force every eligible person's portrait to regenerate even when their evidence
    # fingerprint is unchanged (the pass is otherwise a no-op on an unchanged run). The
    # user-side counterpart of force_persona_digest — useful after a prompt change.
    force_user_portrait: Optional[bool] = None
    # Write `[self_impression]` records — the per-session OUTSIDE view: what the transcript
    # shows about HER to a reader with no access to her <think>. Default on; set False to
    # skip the pass (one generation per session). The introspective counterpart is the
    # revision pass's `[persona]` formation, and the two are deliberately kept independent.
    # Production only: retrieval is `self_impressions.enabled` in server_config (off by
    # default) and the fold is `self_portrait.enabled`. See core.self_portrait.
    self_notes: Optional[bool] = None
    # Force the outside-view portrait to regenerate even when its evidence fingerprint is
    # unchanged — the counterpart of force_persona_digest / force_user_portrait.
    force_self_portrait: Optional[bool] = None
    # Write the per-chat fact-extraction PROTOCOL to `<stem>.facts.json` — everything the
    # conversation established, enumerated literally, including the trivia the curated
    # consolidation pass is right to discard. Default on; set False to skip the pass (one
    # generation per session). Writes to no live store and is read by no RAG channel:
    # `rag_memory.jsonl` stays authoritative, and this is an immutable per-chat source for
    # offline processing. See core.chat_facts.
    chat_facts: Optional[bool] = None
    # Collapse paraphrases in the live `[fact]` store at the end of the run, inside the
    # existing clean-base window (so it costs no extra model reload). Default on. This is
    # the fact-side counterpart of persona clustering: facts are deduped only by EXACT
    # content key when written, so restatements of one truth accumulate indefinitely and
    # can then fill the chat block together. Set False to skip.
    # See core.fact_dedup.
    fact_dedup: Optional[bool] = None

    # Strip pasted-sentence recall cues out of the live trigger-indexed store, ahead of the
    # clean-base window (it is pure string work — no model, no embedder). Default on. A
    # [fact] is INDEXED by its trigger, so a cue that is really a conversational opener is
    # the record's whole retrieval key and makes it unreachable by its own topic. Set False
    # to skip. See core.trigger_hygiene.
    trigger_purge: Optional[bool] = None


@dataclass
class ReflectionRunConfig:
    run_id: str
    source: str                  # "ui" | "cli" | "cron"
    selected_sessions: list[str]
    overrides: ReflectionRunOverrides
    debug: bool = False
    # "Revisit old chat": re-reflect an already-frozen (reflected-once) chat under the
    # CURRENT persona to re-derive its trainable target ("would I answer differently
    # now?"). Bypasses the reflect-once freeze check, suppresses persona formation (the
    # obsolete chat must not reshape who Ava is becoming), and skips the end-of-run
    # persona digest. Consolidation + revision still run; branch judge stays active.
    revisit: bool = False
    # Background per-chat pass (core.background_reflection): run ONLY the per-chat passes
    # (consolidation + revision + branch generation), skipping the run-level cross-cutting
    # work (persona digest, clean-base branch judge, fact placement). The judge/fact job
    # payloads are still collected and PERSISTED per chat so a later nightly run finishes
    # them. Freezes each chat ``chat_reflected`` (stage one) instead of ``reflected_at``.
    chat_only: bool = False


# ── serialization helpers ─────────────────────────────────────────────────────

def _sampling_to_dict(s: Optional[ReflectionSampling]) -> Optional[dict]:
    if s is None:
        return None
    return {
        "temperature": s.temperature,
        "top_p": s.top_p,
        "max_new_tokens_setting": s.max_new_tokens_setting,
    }


def _budget_to_dict(b: Optional[ReflectionChooserBudget]) -> Optional[dict]:
    if b is None:
        return None
    return {
        "max_total_tokens": b.max_total_tokens,
        "context_fraction": b.context_fraction,
        "output_tokens": b.output_tokens,
        "safety_tokens": b.safety_tokens,
    }


def overrides_to_dict(o: ReflectionRunOverrides) -> dict:
    """Serialise only the fields that were actually set (omit None values)."""
    d: dict = {}
    for key in (
        "sleep_prompt", "revision_prompt", "branch_prompt",
        "disable_rag_for_branch_choice", "force_persona_digest", "apply_branch_judge",
        "persona_cluster_mapreduce", "persona_polarity", "user_notes",
        "force_user_portrait", "self_notes", "force_self_portrait", "chat_facts",
        "fact_dedup", "trigger_purge", "skip_branching", "chunk_budget_frac",
        "chars_per_token_hint",
    ):
        v = getattr(o, key)
        if v is not None:
            d[key] = v
    for key in ("sleep_sampling", "revision_sampling", "branch_sampling"):
        v = _sampling_to_dict(getattr(o, key))
        if v is not None:
            d[key] = v
    v = _budget_to_dict(o.chooser_budget)
    if v is not None:
        d["chooser_budget"] = v
    return d


# ── validation ────────────────────────────────────────────────────────────────

_KNOWN_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "sleep_prompt", "revision_prompt", "branch_prompt",
    "sleep_sampling", "revision_sampling", "branch_sampling",
    "chooser_budget", "disable_rag_for_branch_choice", "force_persona_digest",
    "apply_branch_judge", "persona_cluster_mapreduce", "persona_polarity",
    "skip_branching",
    "exchange_anchors", "recollection", "user_notes", "force_user_portrait",
    "self_notes", "force_self_portrait", "chat_facts",
    "fact_dedup", "trigger_purge", "chunk_budget_frac", "chars_per_token_hint",
})

_MNT_RE = re.compile(r"^\d+%?$")


def _check_mnt(value: str, name: str) -> None:
    if not _MNT_RE.match(value.strip()):
        raise ValueError(
            f"{name}: max_new_tokens_setting must be an integer or percentage string "
            f"(e.g. '512' or '75%'), got {value!r}"
        )


def _parse_sampling(raw: Any, name: str) -> ReflectionSampling:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    temp = float(raw.get("temperature", 0.9))    # reflection default, just under chat's 1.0
    if not 0.0 <= temp <= 2.0:
        raise ValueError(f"{name}.temperature must be in [0.0, 2.0], got {temp}")
    top_p = float(raw.get("top_p", 0.95))        # Gemma 4 recommended top_p (family top_k applied by backend)
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"{name}.top_p must be in [0.0, 1.0], got {top_p}")
    mnt = str(raw.get("max_new_tokens_setting", "75%"))
    _check_mnt(mnt, name)
    return ReflectionSampling(temperature=temp, top_p=top_p, max_new_tokens_setting=mnt)


def _parse_chooser_budget(raw: Any) -> ReflectionChooserBudget:
    if not isinstance(raw, dict):
        raise ValueError("chooser_budget must be an object")
    fields: dict = {}
    for key in ("max_total_tokens", "output_tokens", "safety_tokens"):
        if key not in raw:
            raise ValueError(f"chooser_budget.{key} is required")
        v = int(raw[key])
        if v <= 0:
            raise ValueError(f"chooser_budget.{key} must be positive, got {v}")
        fields[key] = v
    cf = float(raw.get("context_fraction", 0.35))
    if not 0.0 < cf < 1.0:
        raise ValueError(f"chooser_budget.context_fraction must be in (0.0, 1.0), got {cf}")
    fields["context_fraction"] = cf
    return ReflectionChooserBudget(**fields)


def validate_overrides(raw: Any) -> ReflectionRunOverrides:
    """Parse and validate a raw override dict from a client message.

    Returns a ``ReflectionRunOverrides`` on success. Raises ``ValueError``
    with a descriptive message on any invalid key, type, or range problem.
    """
    if raw is None:
        return ReflectionRunOverrides()
    if not isinstance(raw, dict):
        raise ValueError("overrides must be an object")
    unknown = set(raw.keys()) - _KNOWN_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unknown override keys: {sorted(unknown)}")

    o = ReflectionRunOverrides()
    for key in ("sleep_prompt", "revision_prompt", "branch_prompt"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError(f"Override {key!r} must be a string")
            setattr(o, key, raw[key])
    for key in ("sleep_sampling", "revision_sampling", "branch_sampling"):
        if key in raw:
            setattr(o, key, _parse_sampling(raw[key], key))
    if "chooser_budget" in raw:
        o.chooser_budget = _parse_chooser_budget(raw["chooser_budget"])
    if "disable_rag_for_branch_choice" in raw:
        o.disable_rag_for_branch_choice = bool(raw["disable_rag_for_branch_choice"])
    if "force_persona_digest" in raw:
        o.force_persona_digest = bool(raw["force_persona_digest"])
    if "apply_branch_judge" in raw:
        o.apply_branch_judge = bool(raw["apply_branch_judge"])
    if "persona_cluster_mapreduce" in raw:
        o.persona_cluster_mapreduce = bool(raw["persona_cluster_mapreduce"])
    if "persona_polarity" in raw:
        o.persona_polarity = bool(raw["persona_polarity"])
    if "skip_branching" in raw:
        o.skip_branching = bool(raw["skip_branching"])
    if "exchange_anchors" in raw:
        o.exchange_anchors = bool(raw["exchange_anchors"])
    if "recollection" in raw:
        o.recollection = bool(raw["recollection"])
    if "user_notes" in raw:
        o.user_notes = bool(raw["user_notes"])
    if "fact_dedup" in raw:
        o.fact_dedup = bool(raw["fact_dedup"])
    if "trigger_purge" in raw:
        o.trigger_purge = bool(raw["trigger_purge"])
    if "force_user_portrait" in raw:
        o.force_user_portrait = bool(raw["force_user_portrait"])
    if "self_notes" in raw:
        o.self_notes = bool(raw["self_notes"])
    if "force_self_portrait" in raw:
        o.force_self_portrait = bool(raw["force_self_portrait"])
    if "chat_facts" in raw:
        o.chat_facts = bool(raw["chat_facts"])
    for key in ("chunk_budget_frac", "chars_per_token_hint"):
        if key in raw:
            v = float(raw[key])
            if v <= 0.0:
                raise ValueError(f"Override {key!r} must be positive, got {v}")
            setattr(o, key, v)
    return o


# ── run store ─────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Terminal statuses — a run in one of these cannot be stopped or updated further.
_DONE_STATUSES = frozenset({"completed", "failed", "stopped"})


class ReflectionRunStore:
    """In-memory + durable registry of reflection runs.

    Thread-safe: all mutations go through a single lock so the asyncio event
    loop and the runner thread (added in Phase 3) can both call safely.

    Persistence layout under *runs_dir*:
      {run_id}.meta.json    — full run state (rewritten on every status change)
      {run_id}.events.jsonl — append-only event log
      {run_id}.summary.json — final counters/status (written at finalization)
    """

    def __init__(self, runs_dir: Path) -> None:
        self._dir = Path(runs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}         # run_id → run state dict
        self._events: dict[str, list[dict]] = {} # run_id → ordered event list
        self._stop_flags: set[str] = set()       # run_ids where stop was requested

    # -- creation -------------------------------------------------------------- #

    def create_run(self, config: ReflectionRunConfig) -> dict:
        """Register a new run and persist its initial metadata.

        Returns the initial run state dict (a copy — callers must not mutate it).
        """
        # Provenance scalars the runner surfaces in its "Runner started" line so the
        # Sleep log names the weights the run is actually using. Read from the live
        # runtime (the loaded model + attached adapter). Lazy import keeps the config
        # layer import-time-decoupled from runtime_state; a headless/unloaded runtime
        # yields None, which the runner renders as "?"/"none (base model)".
        try:
            from core.runtime_state import runtime as _runtime
            _model_id = _runtime.model_id
            _adapter_id = _runtime.adapter_id
        except Exception:
            _model_id = _adapter_id = None
        run: dict = {
            "run_id": config.run_id,
            "status": "pending",
            "source": config.source,
            "debug": config.debug,
            "selected_sessions": list(config.selected_sessions),
            "resolved_overrides": overrides_to_dict(config.overrides),
            "model_id": _model_id,
            "adapter_id": _adapter_id,
            "started_at": _utc_now(),
            "finished_at": None,
            # progress fields — populated by the runner (Phase 3+)
            "phase": None,
            "session_index": 0,
            "session_total": len(config.selected_sessions),
            "chunk_index": 0,
            "chunk_total": 0,
            "exchange_index": 0,
            "exchange_total": 0,
            "skipped_passes": 0,
            # Compact live stats snapshot (elapsed/eta/global x-y/vram/discards/tps),
            # refreshed by the runner via update_status and surfaced in the status
            # payload so the Sleep tab's stats panel can render without events.
            "stats": None,
            "latest_event_seq": 0,
            "summary": None,
        }
        with self._lock:
            self._runs[config.run_id] = run
            self._events[config.run_id] = []
        self._persist_meta(run)
        self.append_event(config.run_id, "run_created",
                          message="Run created, awaiting execution")
        return dict(run)

    # -- queries --------------------------------------------------------------- #

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run is not None else None

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._runs.values()]

    def is_stop_requested(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stop_flags

    # -- mutations ------------------------------------------------------------- #

    def request_stop(self, run_id: str) -> bool:
        """Request a graceful stop. Returns False if the run is not found or already done.

        If the run is still *pending* (no runner has claimed it), it is moved to
        *stopped* immediately since there is nothing to interrupt.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run["status"] in _DONE_STATUSES:
                return False
            self._stop_flags.add(run_id)
            immediate = run["status"] == "pending"
            if immediate:
                run["status"] = "stopped"
                run["finished_at"] = _utc_now()
        if immediate:
            self._persist_meta(run)
            self._persist_summary(run_id, {"status": "stopped", "mutations_applied": False})
        self.append_event(run_id, "stop_requested", message="Stop requested")
        return True

    def update_status(self, run_id: str, **fields) -> None:
        """Update one or more run state fields in memory and re-persist metadata."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.update(fields)
        self._persist_meta(run)

    def append_event(self, run_id: str, event_type: str, **kwargs) -> Optional[dict]:
        """Append a structured event with a monotonically increasing seq number.

        The seq is 1-indexed and stored in the run state as ``latest_event_seq``
        so a reconnecting UI knows exactly where to resume without scanning the log.
        Returns the event dict, or None if the run is not found.
        """
        with self._lock:
            events = self._events.get(run_id)
            run = self._runs.get(run_id)
            if events is None or run is None:
                return None
            seq = len(events) + 1
            event: dict = {
                "type": "reflection_run_event",
                "run_id": run_id,
                "seq": seq,
                "event": event_type,
                "ts": _utc_now(),
                **kwargs,
            }
            events.append(event)
            run["latest_event_seq"] = seq
            run_source = run.get("source")
        self._append_event_to_disk(run_id, event)
        # Name the pass for the activity journal's generation seam: the runner announces
        # each phase here, on the same thread the pass then generates on, so every
        # `body`/`stream` record it writes is labelled with the phase that produced it
        # (see activity_log.set_ambient_label) instead of a flat "reflection".
        if event_type in ("phase_started", "session_started"):
            activity_log.set_ambient_label(event.get("phase") or event_type)
        self._mirror_to_activity(run_id, event_type, event, run_source)
        return event

    @staticmethod
    def _mirror_to_activity(run_id: str, event_type: str, event: dict,
                            run_source: Optional[str] = None) -> None:
        """Mirror a reflection event into the box-wide activity journal so the Activity
        tab shows the run alongside autonomous jobs. run_id is the activity_id
        (correlates the whole run's burst). No-op for non-whitelisted (noisy) events and
        when activity_log is unconfigured (headless CLI).

        ONE grain for every run since 2026-07-28 (see `_ACTIVITY_MIRROR_EVENTS`): chat name
        + exchange counters on each line, and the pass's actual output indented under a
        ``phase_done`` — the same reflection data the Sleep textbox shows. ``run_source`` is
        no longer used to decide detail; it is kept on the signature because the journal
        entry may want to distinguish origins later, and dropping the parameter would be a
        caller change for no gain."""
        if event_type not in _ACTIVITY_MIRROR_EVENTS:
            return
        try:
            kind = ("failed"
                    if event_type in ("run_failed", "phase_error", "pass_error")
                    else "progress")
            # Sleep-tab-style line: [etype] <chat name> <message> (exchange x/y)
            parts = [f"[{event_type}]"]
            sess = (event.get("session") or "").strip()
            if sess:
                parts.append(sess)
            msg = (event.get("message") or "").strip()
            if msg:
                parts.append(msg)
            ex_idx = event.get("exchange_index")
            ex_total = event.get("exchange_total")
            if ex_idx is not None and ex_total:
                parts.append(f"(exchange {ex_idx}/{ex_total})")
            line = "Reflection: " + " ".join(parts)
            # Attach the pass's DERIVED output — its distilled items
            # ([fact]/[persona]/[ask]/[resolved]) — indented under the line, clipped so a
            # runaway pass can't bloat the journal.
            #
            # The pass's RAW generation is deliberately not attached here any more
            # (2026-08-11): the generation seam in `core.generation` now journals every
            # pass's verbatim CoT + output as its own `body` record, under this same
            # activity_id, so repeating it here would log the same kilobytes twice and
            # make the biggest records the duplicated ones. The division is: the seam owns
            # what the model wrote, this mirror owns what the runner made of it. That also
            # covers what the old text attachment was for — a `pass_warning` saying only
            # THAT a parse came back empty is unactionable, but the generation it failed to
            # parse is now the adjacent `body` line rather than a copy inside this one.
            if event_type == "phase_done":
                body_text = _render_report_items(event.get("report") or {})
                if body_text:
                    if len(body_text) > _MIRROR_TEXT_CAP:
                        clipped = len(body_text) - _MIRROR_TEXT_CAP
                        body_text = body_text[:_MIRROR_TEXT_CAP] + f"… [+{clipped} chars]"
                    body = "\n".join("    " + ln for ln in body_text.splitlines())
                    line = line + "\n" + body
            activity_log.append("reflection", kind, line,
                                activity_id=run_id, phase=event.get("phase"))
        except Exception:
            pass

    def get_events(self, run_id: str, after_seq: int = 0) -> list[dict]:
        """Return all events with seq > after_seq (pass 0 to get everything)."""
        with self._lock:
            events = self._events.get(run_id, [])
            return [e for e in events if e["seq"] > after_seq]

    def finalize_run(self, run_id: str, status: str,
                     summary: Optional[dict] = None) -> None:
        """Mark a run terminal and write its summary file."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["status"] = status
            run["finished_at"] = _utc_now()
            run["summary"] = summary
        self._persist_meta(run)
        self._persist_summary(run_id, {"status": status, **(summary or {})})

    # -- persistence ----------------------------------------------------------- #

    def _persist_meta(self, run: dict) -> None:
        path = self._dir / f"{run['run_id']}.meta.json"
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    def _append_event_to_disk(self, run_id: str, event: dict) -> None:
        path = self._dir / f"{run_id}.events.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def persist_report(self, run_id: str, report: dict) -> None:
        """Write the detailed end-of-run statistics report to ``<run_id>.report.json``.

        Separate from the summary so the heavy per-phase/VRAM breakdown lives in its
        own file; it is part of the run-log file set the reflection archive snapshots,
        so it travels with the run for later analysis (e.g. the 24 GB fit question)."""
        payload = {"schema_version": 1, "run_id": run_id, "created_at": _utc_now(),
                   **report}
        path = self._dir / f"{run_id}.report.json"
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    def _persist_summary(self, run_id: str, summary: dict) -> None:
        path = self._dir / f"{run_id}.summary.json"
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass
