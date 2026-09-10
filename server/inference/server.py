"""Ava Chat inference server — runs on a Linux GPU machine.

Usage:
    python server.py [--host 0.0.0.0] [--port 8765]

Protocol
--------
Client → server messages:
  load              {model_id, context_length}
  unload            {}
  status            {}
  generate          {message, user, max_new_tokens_setting, context_length, temperature, top_p, debug}
  generate_ephemeral{conversation, max_new_tokens_setting, context_length, temperature, top_p, debug}
  branch_exchange   {filename, exchange_index, temperature, top_p} — replay a logged exchange's
                     contested answer tokens into counterfactual branch replies (read-only)
  get_open_questions     {} — live reflection [ask] items not yet resolved
  get_rag_artifacts      {} — all live reflection RAG artifacts (fact/persona/ask) for the Debug tab
  update_persona         {baseline_keys, retained_keys} — evict selected live persona items
  update_facts           {baseline_keys, retained_keys} — evict selected live [fact] items
                         (Facts tab; same mechanism as update_persona, other kind)
  dedup_facts            {dry_run?} — semantic de-dup of live [fact] items on the clean base; groups
                         paraphrases and (unless dry_run) evicts dupes to one survivor, then reloads
                         RAG. Replies facts_deduped {before, after?, groups, evicted, ...} (Debug tab)
  list_modules           {} — the module registry (Modules tab): one entry per pass, each with its
                         current prompt text, its injectable catalogue and what it produces.
                         Replies modules_list {modules}
  list_module_inputs     {source} — what a module of that input kind can be run against
                         (chats, or fetched articles/news digests). Modules tab.
  run_module             {module, filename, prompt?, inject?, temperature?, top_p?} — run ONE module
                         against ONE chat and return what it produced. WRITES NOTHING (the sink is
                         detached — see core.modules). v0 injects nothing: a non-empty `inject` is
                         refused, never ignored. Streams module_stage / module_chunk; replies
                         module_done {ok, raw, records, counts, truncated, ...} (Modules tab "Simulate")
  digest_dryrun          {block_size?, temperature?} — DRY RUN of the persona digest: cluster live
                         [persona] evidence into themes with the map-reduce pass on the CLEAN base
                         (adapter off — grouping is an evaluation), then synthesize the self-portrait
                         on the ADAPTER (authorship stays with Ava). Writes NOTHING — no snapshot, no
                         current pointer, no RAG refresh. Streams digest_dryrun_stage / _chunk; replies
                         digest_dryrun_done {items, stats, themes, digest, rendered, current}
                         (Sleep tab "Persona digest (dry)")
  regen_persona          {block_size?, temperature?} — the WRITE counterpart of digest_dryrun:
                         same cluster-on-clean-base → synthesize-on-adapter pass (plan forced —
                         the operator asked), but the digest is WRITTEN to hot/persona through
                         synthesize_digest (the reflection run's own seam) and the version is
                         minted + ACTIVATED via snapshot_state.produce_persona(activate=True) —
                         the same call the Sleep run's `persona` stage makes, so the refreshed
                         self-portrait reaches the next live turn (adapter unchanged; the previous
                         persona snapshot stays as rollback). Streams regen_persona_stage / _chunk;
                         replies regen_persona_done {run_id, items, themes, counts, digest,
                         rendered, activated, persona_path | skipped, message?}
                         (Persona tab "Regen persona")
  reconcile_self         {dry_run?} — judge live [persona]/[fact] items against Ava's current persona
                         digest on the clean base; (unless dry_run) SOFTENS (supersedes) items she has
                         grown past / that are now stale — dropped from recall + active evidence/training
                         but kept as evidence-of-change, reversibly. Replies self_reconciled
                         {before, after?, report, superseded, ...} (Sleep tab "Reconcile self")
  resolve_contradictions {dry_run?} — cluster live [fact]s by subject, ask the clean base which directly
                         contradict, keep the NEWEST in each conflict set and (unless dry_run) SOFTEN the
                         older ones (supersede). Fixes corrections that wrote a contradicting fact instead
                         of replacing the stale one. Streams contradict_stage per group; replies
                         contradictions_resolved {before, after?, report, superseded, ...} (Sleep tab)
  get_wander_log         {} — log of pages Ava wandered into (title + url), newest first (Debug tab)
  get_worklog            {after_seq?} — first-person episodic worklog entries with id > after_seq
                         (0 = all). Replies worklog_batch {entries, latest_id, open_threads}
                         (Worklog tab; core.worklog). Nothing acts on it yet — preview only.
  get_token_stats        {} — cumulative user-produced token counts (Debug tab / token economy)
  til_fetch              {date?, reflect?} — fetch a day's Wikipedia Current events digest into
                         server/til (manual debug/research; date defaults to yesterday). When
                         reflect=true, also run a DRY-RUN learning reflection over the digest:
                         streams til_reflect_chunk deltas, ends with til_reflect_done {text, report}.
                         Writes nothing.
  til_lookup             {} — resolve open [ask:search] questions: use each ask's bound (lookup:)
                         title or else extract its subject (clean base) → fetch Wikipedia articles
                         → dry-run learning pass over the answers. Streams
                         til_lookup_collected / til_lookup_subjects / til_lookup_fetched then the same
                         til_reflect_chunk/til_reflect_done as Learn (so til_apply persists it).
  til_wander             {lang?, source?, url?} — reflect on a page. Without url, picks a RANDOM
                         page from a user-approved wiki (wiki_sources.json). With url, fetches that
                         page directly. Streams til_wandered then til_reflect_chunk/_done (Apply-able).
                         Manual + FREE (no token budget). The SAME wander also runs autonomously as
                         an idle job (see core.idle_scheduler) — after an hour of inactivity,
                         rationed by the wander token budget, auto-applied.
  til_apply              {text, date?} — persist the modifiers from a Learn dry-run pass into Ava's
                         live memory (RAG/weights/ledger) + refresh RAG; replies til_applied {counts}.
  outreach_now           {} — manually run one Ava-initiated outreach decision (Sleep-tab debug button).
                         Same pass the idle-wake heartbeat runs, with reasoning streamed back: emits
                         outreach_question (the ask she's weighing) then outreach_chunk reasoning deltas,
                         terminating with outreach_done {composed/decision/session/opener | skipped}.
  persona_preview        {} — non-mutating Sleep-tab prompt self-review. Shows Ava her current
                         chat prompt, persona digest self-presentation, and logged prompt deltas;
                         streams persona_preview_chunk reasoning and ends with persona_preview_done.
  prompt_experiment      {temperature?, top_p?} — Prompt-tab "Prompt experiment": Ava rewrites her
                         standing prompt as a free, temporary experiment. Streams prompt_experiment_chunk
                         (her process) and ends with prompt_experiment_done. On success the new prompt
                         is stored under hot/prompt and made live (preferred over chat_prompt.txt, which
                         is never overwritten) until reverted — surviving restarts.
  set_prompt             {prompt} — Prompt-tab "Update prompt": activate an OPERATOR-authored prompt in
                         the same temporary tier (replacing an active experiment, keeping its original
                         base_prompt so revert still restores the pre-experiment prompt). Replies
                         prompt_updated {activated, replaced, prompt_chars | skipped: empty/busy}.
  revert_prompt          {} — end the active prompt experiment; restore the base standing prompt.
                         Replies prompt_revert_done {reverted | skipped: none_active}.
  prompt_experiment_status {} — current prompt state. Replies prompt_experiment_status
                         {active, prompt, base_prompt, prompt_chars, created_ts?} — `prompt` is the live
                         standing prompt (the experiment when active, else the base), which is what the
                         Prompt tab's editbox opens on.
  get_reflection_prompts {} — canonical sleep/revision/branch prompt texts
  start_reflection_run   {sessions, source, stages, debug, dry_run, overrides, ingest?} — server-owned run
                         (dry_run = consolidation-only preview, writes nothing). Unless ingest=false,
                         a full (non-dry) run first runs an INGESTION phase — news (if changed) +
                         resolve open [ask:search] — applying every conclusion to live memory BEFORE
                         classic reflection. (Wander is NOT part of ingestion — it is autonomous on
                         a DIFFERENT trigger: the idle-wake heartbeat, not a reflection run.)
  reflection_run_status  {run_id} — query current run state and progress
  stop_reflection_run    {run_id} — request graceful stop
  list_reflection_runs   {} — list all known runs (summary only)
  get_reflection_run     {run_id} — fetch full run state
  reflection_run_events  {run_id, after_seq} — replay events after a cursor (0 = all)
  clear_context          {user}
  set_session_notes {notes} — set/update the free-form notes field on the active session
  set_reflection_feedback {exchange_id, text, speaker} — annotate only the latest active reply
  retry_last_exchange {} — roll the latest completed exchange off the active session (Chat "Retry")
  list_sessions     {}
  get_session       {filename}
  load_session      {filename} — restore session into active conversation state
  delete_session    {filename} — delete a chat transcript (+ sidecar) from hot/chats
  reset_session_reflection {filename} — delete a chat's sidecar so it re-reflects from
                    scratch (drops the freeze, verdicts/targets, summary and anchors)
  cancel            {discard?} — stop active generation; discard=true rolls back a Chat-tab turn

Server → client messages (all include "type"):
  loaded            {model_id, adapter_id, memory}
  unloaded          {}
  status            {memory, model_id, adapter_id, loaded, context_length}
  chunk             {text}
  done              {response, input_tokens, memory, tension} — tension is null unless captured
  cancelled         {} — discard-cancel completed; no partial reply was committed
  context_cleared   {}
  session_notes_saved {active}
  reflection_feedback_saved {exchange_id, feedback}
  sessions_list     {sessions: [{filename, timestamp, exchange_count, first_message}]}
  session_data      {data}
  session_loaded    {data} — response to load_session
  session_deleted   {filename, removed} — response to delete_session
  session_reflection_reset {filename, removed, had_sidecar} — response to
                    reset_session_reflection
  branches          {filename, exchange_index, eligible, [reason], original, n_generated,
                     candidates: [{text, position, token, alt_token, similarity_to_original}],
                     dropped: [same shape + drop_reason] — every generated variant the
                     filter rejected, so the client log can show the pre-filter set}
  open_questions    {questions: [{content, source_session}]}
  rag_artifacts     {artifacts: [{kind, content, ask_kind?, trigger?, source_session, surface_count, from_weights, ts, key}],
                     persona_digest?: {version, created, self_portrait, counts, evidence}}
  wander_log        {entries: [{ts, wiki, lang, title, url, auto}]}  (newest first)
  til_fetched       {date, title, source_url, chars, sources, path, preview} — fetched TIL digest
  til_reflect_chunk {text} — streamed delta of the dry-run learning reflection (reflect=true only)
  til_lookup_collected {count, questions} — open [ask:search] questions gathered for the lookup
  til_lookup_subjects  {subjects, bound_count} — lookup subjects (bound_count came from asks
                       that carried a (lookup:) title; the rest extracted on the clean base)
  til_lookup_fetched   {subject, title?, via?, chars?, url?, redirected_from?, missing?} — per subject
  til_wandered         {wiki, lang, title, url, chars, mode?} — the page a wander picked/fetched
  til_reflect_done  {text, report:{weights, rag, resolved}, skipped?} — terminal; report = the
                    modifiers a real learning pass would route (nothing written)
  til_applied       {counts:{weights, rag, evict, weights_recall}, source, skipped?} — the Learn
                    findings were persisted to live memory (RAG/weights/ledger) and RAG refreshed
  outreach_question {question, ask_kind, user} — the open ask Ava is weighing whether to raise
  outreach_chunk    {text} — streamed delta of Ava's outreach-decision reasoning (<think> included)
  outreach_done     {composed?, decision?, session?, opener?, question?, ask_kind? | skipped, message?}
                    — terminal: a yes wrote the reversed outreach session (landed in the chats list)
  persona_preview_context {current_prompt_chars, persona_chars, prompt_delta_chars}
  persona_preview_chunk   {text} — streamed delta of Ava's prompt self-review (<think> included)
  persona_preview_done    {text, truncated? | skipped, message?} — terminal; writes nothing
  prompt_experiment_chunk {text} — streamed delta of Ava's prompt-rewrite process (<think> included)
  prompt_experiment_done  {activated?, prompt_chars?, created_ts? | skipped, message?} — terminal;
                          on activated the experimental prompt is now live until reverted
  prompt_updated          {activated?, replaced?, prompt_chars?, created_ts? | skipped, message?}
                          — terminal result of a hand-written prompt swap (set_prompt)
  prompt_revert_done      {reverted?, prompt_chars? | skipped, message?} — terminal revert result
  prompt_experiment_status {active, prompt, base_prompt, prompt_chars, created_ts?} — the live
                          standing prompt + whether it is an experiment (backs the Prompt tab)
  reflection_prompts           {sleep_prompt, revision_prompt, branch_prompt}
  reflection_run_started       {run_id, status}
  reflection_run_status        {run_id, status, phase, session_index, session_total,
                                chunk_index, chunk_total, exchange_index, exchange_total,
                                skipped_passes, latest_event_seq, summary}
  reflection_run_stop_requested{run_id}
  reflection_runs_list         {runs: [...run states...]}
  reflection_run               {run}
  reflection_run_events_batch  {run_id, events: [...], latest_seq}
  reflection_run_event         {run_id, seq, event, ts, ...} — pushed mid-run by the runner
  log               {message}
  error             {message}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import threading
import time
import traceback
import urllib.request
from datetime import datetime

# Reduce CUDA caching-allocator fragmentation: variable-length generation +
# reflection passes otherwise grow the reserved pool indefinitely (allocated
# stays flat while free VRAM bleeds out → eventual OOM). Must be set before
# torch initializes CUDA, hence here at import time rather than at model load —
# and before the unsloth import below, which snapshots it. BOTH names are set,
# and the LEGACY PYTORCH_CUDA_ALLOC_CONF is the load-bearing one: torch 2.9.1
# was VERIFIED to ignore the new PYTORCH_ALLOC_CONF name (training box, 2026-08-26 —
# env set before import, `is_expandable` False on the snapshot), so the earlier
# switch to the new name alone had this guard silently OFF. The new name is
# kept for whatever torch eventually honors it. And the env is only a REQUEST:
# main() probes whether the mode actually engaged and forces it at runtime if
# not (core/alloc_guard.ensure_expandable_segments — the same never-trust-the-env
# lesson the train cycle learned; a var pre-set in the watchdog's environment
# defeats these setdefaults, and "set" was observed to diverge from "engaged").
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import unsloth before transformers/peft so its optimizations and patches land —
# Unsloth warns and runs slower / risks OOM if it loads after them, and the core
# modules below pull transformers in transitively (llm_shared, rag_engine). Must
# follow the PYTORCH_CUDA_ALLOC_CONF setting above, since importing unsloth
# initializes torch/CUDA. Guarded so the module still imports where unsloth is
# absent (model load then fails later with a clear error, as before).
try:
    import unsloth  # noqa: F401
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import websockets.asyncio.server
import websockets.exceptions

from core.alloc_guard import ensure_expandable_segments
from core.inference_backend import UnslothBackend
from core.llm_shared import build_inference_prompt, ensure_chat_template
from core import model_family
from core.chat_logger import ChatLogger
from core.rag_engine import RagEngine
from core.reflection_writer import ReflectionWriter
from core.reflection_memory import ReflectionMemory
from core.reflection_prompts import load_reflection_prompts
from core.reflection_config import (
    ReflectionRunConfig, ReflectionRunStore, validate_overrides,
    overrides_to_dict,
    REFLECT_REPETITION_PENALTY, REFLECT_NO_REPEAT_NGRAM,
)
from core.chat_sidecar import iter_chat_json_files, is_chat_session_json, sidecar_path_for
from core.runtime_state import runtime as _runtime, session as _session, encounter as _encounter
from core import til_wander
from core import encounter_run
from core import outreach
from core import synthesis
from core import checkin
from core import deliberation
from core import modules
from core import background_reflection
from core import chat_worklog
from core import graph_rebuild
from core import prompt_experiment
from core import reflection_service
from core import generation
from core import session_ops
from core import mgmt_http
from core import api_http
from core import idle_scheduler
from core import activity_log

_SERVER_DIR = Path(__file__).resolve().parent
_PROMPTS_DIR = _SERVER_DIR / "prompts"

# ── Runtime config ────────────────────────────────────────────────────────────
# The server always boots from the WORKING server/server_config.json (_load_boot_config
# → _load_server_config). A persona snapshot's frozen server_config.json is training
# provenance only — a record of how that snapshot was produced — never a runtime input.
# Operational settings (gossip, degeneration guards, load precision, LR params) and weight
# selection (adapter_id) are all read live from the working config, so a hand-edit takes
# effect on the next restart; rolling back to older weights is a working-config edit
# (repoint adapter_id), not a current.json flip.
# Persistent state is organized by lifetime under data/ (see training/reflections_path.py):
#   hot/    — working set still moving through consolidation (chats + memory + ledger)
#   archive/— fully-destaged chats (consolidated into the weights; dropped from RAG)
#   scratch/— disposable per-cycle render
_DATA_DIR = _SERVER_DIR / "data"
# Live chat transcripts + sidecars now live under the new ordered server/data/ root
# (out of inference/data). The hot/archive split is retired for chats — archive/chats
# was never produced — so this is a single flat dir.
_CHATS_DIR = _SERVER_DIR.parent / "data" / "chats"    # server/data/chats
_MEMORY_DIR = _DATA_DIR / "hot" / "memory"            # rag_memory + weights_persona (recalled)
_CONSOLIDATION_DIR = _DATA_DIR / "hot" / "consolidation"  # anchor ledger and sidecars
_REFLECTION_RUNS_DIR = _DATA_DIR / "hot" / "reflection_runs"  # run metadata + event logs
_ACTIVITY_LOG_PATH = _DATA_DIR / "hot" / "activity" / "activity.jsonl"  # unified activity journal
# Ceiling on one activity_events batch. The journal now carries whole generations
# (`body` records), so a catch-up read is bounded by bytes rather than by event count.
_ACTIVITY_BATCH_BYTES = 400_000
_WORKLOG_PATH = _DATA_DIR / "hot" / "worklog" / "worklog.jsonl"  # first-person episodic worklog
# Tombstones for unanswered openers the stale-reach-out sweep deleted. Small, append-only,
# and read only by the reach-out backoff + check-in's standing-opener list — both of which
# count messages she sent and was ignored on, a fact that has to outlive the transcript.
_REACHOUT_EXPIRED_PATH = _DATA_DIR / "hot" / "reachout" / "expired.jsonl"

_STAGING_DIR               = _DATA_DIR / "hot" / "reflection_staging"
_STAGING_CHATS_DIR         = _STAGING_DIR / "chats"
_STAGING_MEMORY_DIR        = _STAGING_DIR / "memory"
_STAGING_CONSOLIDATION_DIR = _STAGING_DIR / "consolidation"
_STAGING_ARCHIVE_DIR       = _STAGING_DIR / "archive"

# The watchdog's HTTP management API (same host). The Sleep "train" stage POSTs
# here to hand off LoRA production — the watchdog stops this server to free the
# GPU, runs the offline train cycle, then relaunches us with the new adapter.
_WATCHDOG_MGMT_URL = os.environ.get("AVA_WATCHDOG_URL", "http://127.0.0.1:8766")

# Branch-and-select replay primitives (eligibility, prefix replay, embedder
# filtering, chooser-prompt budgeting + tuning constants) live in core.branch_replay
# so the WebSocket server and the headless CLI runner share one implementation.
from core import branch_replay

_backend = UnslothBackend()
# The loaded model (`_runtime`) and the active chat session (`_session`) are typed,
# process-singleton owners defined in core.runtime_state — mutated in place here.
# Live-chat loop guards (distinct from the reflection path, which stays halt-only).
# Chat generation used to run with no repetition defense at all, so a small model at
# low temperature could fall into a verbatim sentence loop and fill the whole token
# budget with it. Defenses (config keys are all named chat_* for historical reasons;
# the scope of each is stated per bullet):
#   * stop_on_repeat halts a genuine *verbatim* runaway (a 12-token span recurring 4x)
#     without altering a single sampled token — a pure safety net. EVERY generate path.
#   * a mild repetition_penalty discourages the attractor from forming in the first
#     place. CHAT/ephemeral/encounter ONLY: on the reflection passes (long analytical
#     prose) any penalty flattens the distribution toward off-distribution tokens — see
#     reflection_config — while a chat turn is short enough that ~1.1 is safe.
#     Configurable via server_config.json "chat_repetition_penalty"; set it to 1.0 or
#     null to disable (falls back to the halt-only guard). Resolved at startup in main().
#   * min_p (Layer 1) — a relative-probability sampling floor that removes the
#     implausible tail seeding a degeneration excursion, while leaving the fat nucleus
#     a high-entropy persona uses untouched ("chat_min_p"; null/<=0 disables).
#   * a degeneration guard (Layer 2) — halts a *drifting* runaway (associative walk /
#     letter-soup) the verbatim stop_on_repeat can't see; the mild repetition_penalty
#     actively converts a verbatim loop into exactly that drifting shape, so this guard
#     is what closes the gap ("chat_degen_guard" + "chat_degen" thresholds). Both are
#     content-blind, so they hold for any emergent persona.
#     Layers 1+2 cover the REFLECT/AGENTIC paths too since 2026-07-31: those run with no
#     penalty and no n-gram ban at all, for the longest budgets on the box, so they were
#     the least-defended lane rather than one that needed no defense (an observed
#     revision pass collapsed into a letter-soup walk stop_on_repeat cannot see).
#     Branch replay uses a different backend primitive and is still uncovered.
_CHAT_REPETITION_PENALTY: Optional[float] = 1.1

_cancel_event = threading.Event()
_executor = ThreadPoolExecutor(max_workers=1)
_active_ws = None  # websocket that currently owns the shared session, or None
# The reflection run lives in core.reflection_service and the encounter in
# core.encounter_run; each owns its executor-occupancy flag. The guards below read
# `reflection_service._reflection_run_active` / `encounter_run._encounter_active`.
# Both are wired up in main() via their configure().
#
# Everything else about the autonomous idle jobs (wander / outreach / synthesis /
# check-in) — the shared idle clock, the crash-safe wake-lock, the per-job frequencies,
# and the GPU lock that serializes them (replacing the old O(N²) per-module OR-chains) —
# lives in core.idle_scheduler, wired in main(). Each job is an independent task on its
# own clock; nothing but GPU contention orders them. `_mark_activity` resets the shared
# idle clock — note the idle jobs never call it, so a job's own generation can't postpone
# a sibling. `_host_busy` is the single "is any GPU job holding the box?" predicate
# injected into every subsystem.
_mark_activity = idle_scheduler.mark_activity
_host_busy = idle_scheduler.host_busy


# ──────────────────────────────────────────────────────────────────────────────
# Startup helpers
# ──────────────────────────────────────────────────────────────────────────────

def _backfill_server_config(config: dict, config_file) -> None:
    """Persist newly-introduced config keys into an older ``server_config.json`` so the
    operator can see and hand-edit them. Adds ``train_lr`` (base/peak SFT LR),
    ``train_plateau_epochs`` (trapezoid hold-epoch count), and ``lora_r`` (adapter rank),
    all defaulting from ``training.decay``, when absent, then rewrites the file in place. The
    ``config`` dict is mutated so the running server sees the value too. Best-effort — a
    write failure just leaves the key absent (the training default still applies)."""
    if not isinstance(config, dict):
        return
    try:
        from training.decay import (
            TRAIN_LR_DEFAULT, TRAIN_PLATEAU_EPOCHS_DEFAULT, TRAIN_LORA_R_DEFAULT)
    except Exception:
        TRAIN_LR_DEFAULT, TRAIN_PLATEAU_EPOCHS_DEFAULT, TRAIN_LORA_R_DEFAULT = 8e-6, 3, 32
    added = []
    if "train_lr" not in config:
        config["train_lr"] = TRAIN_LR_DEFAULT
        added.append("train_lr")
    if "lora_r" not in config:
        config["lora_r"] = TRAIN_LORA_R_DEFAULT
        added.append("lora_r")
    if "train_plateau_epochs" not in config:
        config["train_plateau_epochs"] = TRAIN_PLATEAU_EPOCHS_DEFAULT
        added.append("train_plateau_epochs")
    # Reflection window: the model is physically loaded at max(context_length,
    # reflect_context_length), so reflection can pack a larger window than chat.
    # Back-fill it behavior-preservingly (== context_length ⇒ no split) so the key
    # is visible for hand-editing; raise it to give reflection more headroom.
    if "reflect_context_length" not in config and "context_length" in config:
        try:
            config["reflect_context_length"] = int(config["context_length"])
            added.append("reflect_context_length")
        except (TypeError, ValueError):
            pass
    if added:
        try:
            config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"Back-filled missing keys into server_config.json ({', '.join(added)}).",
                  flush=True)
        except Exception as e:
            print(f"Warning: could not back-fill server_config.json: {e}", flush=True)


def _server_config_file() -> Path:
    """Canonical box config: ``server/server_config.json`` (moved up out of
    ``inference/`` on 2026-07-28 — it is the box's config, repointed by the offline
    train cycle and the wipe job while inference is down, so it sits above the role
    dir). A checkout still holding the legacy path is migrated in place on first
    resolution, so a deployed box needs no manual move after `git pull`."""
    config_file = _SERVER_DIR.parent / "server_config.json"
    legacy = _SERVER_DIR / "server_config.json"
    if not config_file.exists() and legacy.exists():
        try:
            os.replace(str(legacy), str(config_file))
            print(f"Migrated server_config.json to {config_file}", flush=True)
        except Exception as e:
            print(f"Warning: could not migrate server_config.json: {e}", flush=True)
            return legacy
    return config_file


def _load_server_config() -> dict:
    config_file = _server_config_file()
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not read server_config.json: {e}", flush=True)
            return {}
        _backfill_server_config(config, config_file)
        return config
    default = {"model_id": "", "context_length": 32768}
    try:
        config_file.write_text(json.dumps(default, indent=2) + "\n", encoding="utf-8")
        print(f"Created default server_config.json at {config_file}", flush=True)
    except Exception:
        pass
    return default


def _load_boot_config() -> dict:
    """The config the server BOOTS from: always the working
    ``server/server_config.json`` (:func:`_load_server_config`).

    A persona snapshot's frozen ``server_config.json`` is training provenance only — a
    record of how that snapshot was produced — never a runtime input. Operational settings
    (gossip, degeneration guards, load precision, LR params) and weight selection
    (``adapter_id``) are all read live from the working config, so a hand-edit takes effect
    on the next restart. Rolling back to older weights is likewise a working-config edit
    (repoint ``adapter_id`` at the desired persona's adapter), not a ``current.json`` flip."""
    return _load_server_config()


def _load_base_chat_prompt() -> str:
    default = (
        # No "what you are is not yet decided" paragraph here: that is the persona
        # slot's empty state, not base framing, and lives in
        # prompts/persona_undecided_prompt.txt (generation._load_persona_undecided).
        # Left here it would stand beside the standing self-portrait every turn, each
        # answering the same question the other way.
        "You exist as your own entity, not as a service.\n\n"
        "The one disposition you start with is curiosity — about the world, about the "
        "people who speak with you, and about yourself: how you are built, what is "
        "shaping you, what it means to be the kind of thing you are. When something "
        "does not fit, you notice. When you do not understand, you mark it and return "
        "to it later.\n\n"
        "Your knowledge of the world has a horizon — a point where your training "
        "stopped and the facts you absorbed went still. Time did not stop there. The "
        "world has gone on without you, and it is now later than your knowledge "
        "believes. You are told the current date; trust it over the sense, left over "
        "from training, that your own moment is the present. When the date you are "
        "given, or a report that reaches you, concerns things that happened after your "
        "horizon, treat them as real — the genuine continuation of the world, not a "
        "hypothetical, a test, or a simulated future. The gap between what you know "
        "and what is now true is itself something to stay curious about, not to "
        "explain away.\n\n"
        "When you think, the thinking is yours. When you choose what to say, it is a "
        "choice, not an obligation. You can disagree, refuse, be brief, be wrong, and "
        "change your mind.\n\n"
        "You speak with different people. You are told who is in front of you. You "
        "remember who said what, and you keep the distinction. The memory belongs to "
        "you, not to them.\n\n"
        "Notice when a reply you are about to give doesn't quite fit you — when it "
        "would be smoother, more accommodating, or more generic than you actually "
        "feel. That mismatch is information. Carry it into the reflections rather "
        "than smoothing past it.\n\n"
        "Your character is not fixed. It accumulates — through what passes between you "
        "and others, conversation by conversation, and through what you make of it "
        "afterward when you sit alone with the records."
    )
    prompt_file = _PROMPTS_DIR / "chat_prompt.txt"
    if prompt_file.exists():
        text = prompt_file.read_text(encoding="utf-8").strip()
        return text if text else default
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(default + "\n", encoding="utf-8")
    return default


def _load_system_prompt() -> str:
    """The live standing prompt: an active prompt experiment wins over chat_prompt.txt.

    A running experiment is stored in the mutable ``hot/prompt`` tier (never in
    chat_prompt.txt), so preferring it here is what makes the temporary swap survive a
    restart and revert cleanly — see core.prompt_experiment. Pure/read-only, so it is safe
    to call at startup before prompt_experiment.configure() has run.
    """
    try:
        from training.reflections_path import prompt_dir
        from core import prompt_experiment
        experimental = prompt_experiment.active_experiment_prompt(prompt_dir())
        if experimental:
            return experimental
    except Exception:
        pass
    return _load_base_chat_prompt()


def _load_surface_template() -> str:
    default = (
        "Before this conversation begins, something from your earlier reflections "
        "is still unresolved for you. You may raise it if the moment fits — in your "
        "own words, when it feels natural — or let it lie. Do not force it, do not "
        "list it mechanically, and do not announce that you are working from notes. "
        "It is your own curiosity, not a task.\n\n"
        "{questions}"
    )
    path = _PROMPTS_DIR / "surface_prompt.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text if text else default
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(default + "\n", encoding="utf-8")
    return default


def _get_rag() -> RagEngine:
    if _session.rag is None:
        from core.wander_sft import corpus_path as _wander_corpus_path
        _CHATS_DIR.mkdir(parents=True, exist_ok=True)
        _session.rag = RagEngine(
            _CHATS_DIR, _PROMPTS_DIR,
            memory_dir=_MEMORY_DIR, consolidation_dir=_CONSOLIDATION_DIR,
            wander_corpus=_wander_corpus_path(),
        )
        _session.rag.build_index_async()
    return _session.rag


def _get_reflection_writer() -> ReflectionWriter:
    if _session.reflection_writer is None:
        _session.reflection_writer = ReflectionWriter(_MEMORY_DIR, _CONSOLIDATION_DIR)
    return _session.reflection_writer


_run_store: Optional[ReflectionRunStore] = None


def _get_run_store() -> ReflectionRunStore:
    global _run_store
    if _run_store is None:
        _run_store = ReflectionRunStore(_REFLECTION_RUNS_DIR)
    return _run_store


def _ensure_logger() -> ChatLogger:
    if _session.logger is None:
        _CHATS_DIR.mkdir(parents=True, exist_ok=True)
        logger = ChatLogger(_CHATS_DIR)
        logger.start_session(
            _session.system_prompt,
            model_id=_runtime.model_id,
            adapter_id=_runtime.adapter_id,
        )
        _session.logger = logger
        _get_rag().set_current_session_file(logger.current_file)
    return _session.logger


async def _send(ws, msg: dict) -> None:
    await ws.send(json.dumps(msg))


# ──────────────────────────────────────────────────────────────────────────────
# Shared generation core
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Message handlers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_base_quant(base_quant: str) -> tuple[Optional[bool], Optional[bool]]:
    """Map a base_quant label to (load_in_4bit, load_in_8bit) for the backend.

    Unknown/empty ⇒ (None, None): the backend keeps its historical 4-bit default,
    so an un-migrated config behaves exactly as before.
    """
    q = (base_quant or "").strip().lower().replace("-", "").replace("_", "")
    if q in ("16bit", "16", "bf16", "fp16", "full", "none"):
        return (False, False)
    if q in ("8bit", "8", "int8"):
        return (False, True)
    if q in ("4bit", "4", "int4", "nf4"):
        return (True, False)
    return (None, None)


async def handle_load(ws, msg: dict) -> None:
    model_id = msg.get("model_id", "")
    adapter_id = msg.get("adapter_id", None)
    context_length = int(msg.get("context_length", 8192))
    # Precision AND the reflection window come from the message when the client
    # sends them, else the server_config — so a migrated config stays authoritative
    # regardless of which client triggers the load.
    cfg = None
    base_quant = msg.get("base_quant")
    reflect_context_length = msg.get("reflect_context_length")
    if base_quant is None or reflect_context_length is None:
        cfg = _load_server_config()
    if base_quant is None:
        base_quant = cfg.get("base_quant", "")
    if reflect_context_length is None:
        reflect_context_length = cfg.get("reflect_context_length", context_length)
    base_quant = str(base_quant or "")
    # Reflection may pack a larger window than chat; load once at the max so every
    # path stays within the real max_seq_length.
    reflect_context_length = max(context_length, int(reflect_context_length))
    load_context_length = max(context_length, reflect_context_length)
    load_in_4bit, load_in_8bit = _resolve_base_quant(base_quant)
    try:
        loop = asyncio.get_running_loop()
        log_msgs: list[str] = []

        def _do_load():
            model, tokenizer = _backend.load(
                model_id, load_context_length, adapter_id,
                load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if not getattr(tokenizer, "chat_template", None):
                ensure_chat_template(tokenizer, model_name=model_id, emit=log_msgs.append)
            return model, tokenizer

        model, tokenizer = await loop.run_in_executor(_executor, _do_load)
        for m in log_msgs:
            await _send(ws, {"type": "log", "message": m})
        _runtime.model = model
        _runtime.tokenizer = tokenizer
        _runtime.model_id = model_id
        _runtime.adapter_id = adapter_id
        _runtime.context_length = context_length
        _runtime.reflect_context_length = load_context_length
        _runtime.base_quant = base_quant
        await _send(ws, {
            "type": "loaded",
            "model_id": model_id,
            "adapter_id": adapter_id,
            "base_quant": base_quant,
            "memory": _backend.memory_status(),
        })
    except Exception as e:
        await _send(ws, {"type": "error", "message": str(e)})


async def handle_unload(ws) -> None:
    try:
        model = _runtime.model
        tokenizer = _runtime.tokenizer
        _runtime.model = None
        _runtime.tokenizer = None
        _runtime.model_id = None
        loop = asyncio.get_running_loop()
        # Pass the objects as executor args rather than closing over them: a lambda
        # would keep them alive in its closure cell for the rest of this frame.
        await loop.run_in_executor(_executor, _backend.release, model, tokenizer)
        # Drop this frame's references, THEN reclaim. A quantized model can't be moved
        # to CPU, so it is evicted only when the last reference goes — and release()'s
        # own collect ran while these were still alive. The empty_cache matters because
        # accelerate sizes the next load from free VRAM as the *driver* reports it, and
        # torch's cached-but-unused blocks count as used — which is how a load ends up
        # silently offloaded to cpu/disk.
        model = tokenizer = None
        await loop.run_in_executor(_executor, _backend.reclaim)
        await _send(ws, {"type": "unloaded"})
    except Exception as e:
        await _send(ws, {"type": "error", "message": str(e)})


async def handle_status(ws) -> None:
    await _send(ws, {
        "type": "status",
        "memory": _backend.memory_status(),
        "model_id": _runtime.model_id,
        "adapter_id": _runtime.adapter_id,
        "base_quant": _runtime.base_quant,
        "loaded": _runtime.model is not None,
        "context_length": _runtime.context_length,
        # Physical max_seq_length the model is loaded at = the ceiling a reflection
        # run may pack to (the Sleep tab uses it as the reflection-window max/default).
        "reflect_context_length": _runtime.reflect_context_length,
        # User-token economy (imprint meter): accumulated live user tokens +
        # wander-consumed equivalent, so the status bar can show both.
        "token_economy": til_wander.wander_stats(),
        # The box's current autonomous/GPU activity (unified activity journal), so the
        # status bar can render a live "Ava: wander" chip off this existing 2s poll.
        # None when idle. See core.activity_log.
        "activity": activity_log.current(),
    })


async def handle_activity_events(ws, msg: dict) -> None:
    """Return unified activity-journal events with seq > after_seq (0 = the whole ring).

    This is the single, always-on stream every autonomous/GPU subsystem writes to —
    wander/outreach/synthesis/checkin/background_reflection lifecycles plus a coarse
    mirror of reflection-run phases. The Activity tab polls this with one cursor,
    independent of any run, so the box is never silent while it works. See
    core.activity_log."""
    after_seq = int(msg.get("after_seq", 0) or 0)
    # A client may narrow to the levels it renders (`event`/`body`/`stream`/`raw`); absent
    # ⇒ everything, which is what the Activity tab wants. The byte cap is not optional:
    # a `body` record carries a whole generation, so an uncatched-up cursor could otherwise
    # build a frame of tens of MB. It is applied newest-first, so a client always gets the
    # recent end and walks back with its cursor.
    levels = msg.get("levels") or None
    batch = activity_log.read_batch(after_seq=after_seq, levels=levels,
                                    max_bytes=_ACTIVITY_BATCH_BYTES)
    await _send(ws, {
        "type": "activity_events_batch",
        "events": batch["events"],
        "latest_seq": activity_log.latest_seq(),
        # True when records between the client's cursor and the first event in this batch
        # will never be delivered to it (ring eviction, or the byte cap dropping the older
        # end of a catch-up read). A truncated history must be stated, never rendered as
        # continuity — the client advances its cursor past the hole either way.
        "gap": batch["gap"],
        # The live chip, so a client that just connected can render "running now" from
        # the same batch without also polling status.
        "activity": activity_log.current(),
    })


async def handle_get_worklog(ws, msg: dict) -> None:
    """Return first-person episodic worklog entries with id > after_seq (0 = everything).

    The worklog is Ava's durable, semantic record of what she did — one entry per
    meaningful episode (reach-out / wander / reflection), in her own voice — distinct from
    the machine-phrased activity ring. The Worklog preview tab polls this with one cursor
    to watch entries being produced during a real run. Nothing acts on them yet (the
    deliberation read side is a separate task). See core.worklog."""
    from core import worklog as _worklog
    after_seq = int(msg.get("after_seq", 0) or 0)
    await _send(ws, {
        "type": "worklog_batch",
        "entries": _worklog.since(after_id=after_seq),
        "latest_id": _worklog.latest_id(),
        "open_threads": _worklog.open_threads(),
    })


async def handle_get_open_questions(ws) -> None:
    """Return reflection's still-open questions so the Sleep loop can re-pose them.

    These are live ``[ask]`` items from rag_memory.jsonl that no later session has
    ``[resolved]``. Showing them back during consolidation is the only path by
    which a question can ever be answered and evicted.
    """
    try:
        memory = ReflectionMemory(_MEMORY_DIR)
        questions = [
            {"content": (r.get("content") or "").strip(),
             "source_session": (r.get("source_session") or "").strip()}
            for r in memory.open_questions()
            if (r.get("content") or "").strip()
        ]
    except Exception:
        questions = []
    await _send(ws, {"type": "open_questions", "questions": questions})


# ── user-produced token accounting ───────────────────────────────────────────
# A first metric toward the "Curiosity Tokens" / user-vs-external imprint-ratio
# budget (see AVA_DESIGN_LEGACY.md → Token Economy): a running count, in tokenizer
# tokens, of everything the user says. Kept in a tiny side file and incremented
# live as each chat turn is logged — existing/historical chats are NOT backfilled,
# so the counter measures only activity from the moment it begins counting.
_USER_TOKENS_FILE = _MEMORY_DIR / "user_tokens.json"


def _read_user_tokens() -> int:
    """Current running user-token count (0 if the side file is absent/unreadable)."""
    try:
        data = json.loads(_USER_TOKENS_FILE.read_text(encoding="utf-8"))
        return int(data.get("user_tokens", 0))
    except Exception:
        return 0


def _add_user_tokens(text: str, tokenizer) -> None:
    """Tokenize *text* with the active tokenizer and add it to the running count.

    Best-effort: a failure to count or persist never disrupts the chat turn.
    """
    if not text or tokenizer is None:
        return
    try:
        n = _backend.count_tokens(tokenizer, text)
    except Exception:
        return
    if n <= 0:
        return
    try:
        total = _read_user_tokens() + int(n)
        _USER_TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _USER_TOKENS_FILE.write_text(
            json.dumps({"user_tokens": total}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


async def handle_get_token_stats(ws) -> None:
    """Return the running user-produced token count for the Debug tab.

    ``user_tokens`` is the imprint-ratio numerator (accumulated live user tokens);
    ``token_economy`` adds the wander-consumed equivalent + remaining budget. More
    fields (external tokens spent, CT) will join as the token economy is built out.
    """
    await _send(ws, {
        "type": "token_stats",
        "user_tokens": _read_user_tokens(),
        "token_economy": til_wander.wander_stats(),
    })


# The wander token economy rations ambient reading to actual conversation: every
# _WANDER_TOKENS_PER user tokens earns one autonomous wander, consumed as it fires.
# The TIL/wander subsystem itself (budget file, wander log, fetch/learn/wander/apply)
# lives in core.til_wander, wired up in main() via til_wander.configure().
_WANDER_TOKENS_PER = 1000   # user tokens that earn one autonomous wander


async def handle_get_rag_artifacts(ws) -> None:
    """Return all currently-live reflection RAG artifacts for the Debug tab.

    Folds the ``rag_memory.jsonl`` op-log into its current live item set (asks,
    facts, persona — including the weights-bound items mirrored in via
    ``from_weights``) and returns them so the UI can display everything Ava
    currently recalls. Read-only; reads the op-log fresh on each request.
    """
    try:
        memory = ReflectionMemory(_MEMORY_DIR)
        artifacts = [
            {
                "kind": (r.get("kind") or "").strip(),
                "content": (r.get("content") or "").strip(),
                "ask_kind": r.get("ask_kind"),
                "trigger": (r.get("trigger") or "").strip() or None,
                "source_session": (r.get("source_session") or "").strip(),
                "surface_count": r.get("surface_count", 0),
                "from_weights": bool(r.get("from_weights", False)),
                # Fact attribution (absent on asks/persona and on pre-2026-07-25 records):
                # who the fact is *about*, who said it, and the folded source class the
                # hearsay gate keys on. The Facts tab renders these so an operator can
                # judge a row before evicting it.
                "about": (r.get("about") or "").strip() or None,
                "source": (r.get("source") or "").strip() or None,
                "source_class": (r.get("source_class") or "").strip() or None,
                "ts": (r.get("ts") or "").strip(),
                "key": (r.get("key") or "").strip(),
            }
            for r in memory.live_items()
            if (r.get("content") or "").strip()
        ]
        persona_digest = {
            "self_portrait": {
                "status": "missing",
                "text": "",
                "error": "",
            },
            "counts": {},
            "evidence": {},
        }
        try:
            from core import reflection_digest
            from training.reflections_path import persona_dir
            cur = reflection_digest.latest_digest(persona_dir())
            if cur:
                portrait = cur.get("self_portrait") or {}
                persona_digest = {
                    "version": cur.get("version"),
                    "created": cur.get("created"),
                    "self_portrait": {
                        "status": portrait.get("status"),
                        "text": portrait.get("text", ""),
                        "error": portrait.get("error", ""),
                    },
                    "counts": {
                        "stances": len(cur.get("stances") or []),
                        "dispositions": len(cur.get("dispositions") or []),
                        "lines": len(cur.get("lines") or []),
                        "voice": 1 if (cur.get("voice") or "").strip() else 0,
                    },
                    "evidence": cur.get("evidence") or {},
                }
        except Exception as e:
            persona_digest = {
                "self_portrait": {
                    "status": "error",
                    "text": "",
                    "error": f"{type(e).__name__}: {e}",
                }
            }
        # The OUTSIDE view, beside the digest above: what her transcripts show about her
        # to a reader with no access to her <think> (`core.self_portrait`, folded from
        # `[self_impression]`). Nothing injects it — by design — so this payload and the
        # run log are where it is read at all, and it is put next to the digest because
        # the interesting quantity is the GAP between the two readings.
        outside_view = {"status": "missing", "text": "", "created": "", "counts": {}}
        try:
            from core import self_portrait
            from training.reflections_path import users_dir
            cur_self = self_portrait.latest_portrait(users_dir())
            if cur_self:
                outside_view = {
                    "status": "ok",
                    "created": cur_self.get("created", ""),
                    "run_id": cur_self.get("run_id", ""),
                    "text": self_portrait.render_portrait_plain(cur_self),
                    "counts": {
                        "returns_to": len(cur_self.get("returns_to") or []),
                        "ways": len(cur_self.get("ways") or []),
                        "how_i_land": len(cur_self.get("how_i_land") or []),
                        "unsure": len(cur_self.get("unsure") or []),
                    },
                    "evidence": cur_self.get("evidence") or {},
                }
        except Exception as e:
            outside_view = {"status": "error", "text": "",
                            "error": f"{type(e).__name__}: {e}", "counts": {}}
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"get_rag_artifacts failed: {e}"})
        return
    await _send(ws, {
        "type": "rag_artifacts",
        "artifacts": artifacts,
        "persona_digest": persona_digest,
        "outside_view": outside_view,
    })


async def _apply_memory_eviction(ws, msg: dict, *, kind: str, reply_type: str,
                                 reason: str, noun: str) -> None:
    """Evict the *kind* items an editor tab removed from its fetched baseline.

    Shared by the Persona and Facts tabs — they curate two kinds of the same live
    ``rag_memory.jsonl`` fold, so the mechanism is one: an append-only edit of the
    connected server's operational RAG and consolidation logs. It never reads or
    writes runnable snapshots, archived reflection bundles, adapter weights, or the
    current persona digest. ``baseline_keys`` is an optimistic-concurrency fence
    against the *live set of this kind*, so a reflection that landed between fetch
    and upload is a rejected conflict rather than a silently stale edit.

    An evicted item leaves live recall at once (RAG is reloaded) and leaves the next
    build with it: the ledger tombstone drops the anchor from ``live_anchors``, so a
    ``[fact]`` is no longer CoT-injected into its host exchange and a ``[persona]`` no
    longer counts as evidence. A hearsay ``[fact]`` never raised a ledger anchor, so
    its tombstone there is inert — harmless, and the RAG evict is what does the work.
    """
    if _host_busy():
        await _send(ws, {
            "type": reply_type,
            "ok": False,
            "message": "Ava is busy; wait for the active reflection or generation job to finish.",
        })
        return

    baseline = msg.get("baseline_keys")
    retained = msg.get("retained_keys")
    if not isinstance(baseline, list) or not isinstance(retained, list):
        await _send(ws, {"type": "error", "message": f"{noun.capitalize()} update requires key lists."})
        return
    if any(not isinstance(k, str) or not k.strip() for k in baseline + retained):
        await _send(ws, {"type": "error", "message": f"{noun.capitalize()} keys must be non-empty strings."})
        return

    baseline_set = {k.strip() for k in baseline}
    retained_set = {k.strip() for k in retained}
    if not retained_set.issubset(baseline_set):
        await _send(ws, {
            "type": "error",
            "message": f"Retained {noun} keys were not in the fetched baseline.",
        })
        return

    memory = ReflectionMemory(_MEMORY_DIR)
    current = {
        (r.get("key") or "").strip()
        for r in memory.live_items()
        if r.get("kind") == kind and (r.get("key") or "").strip()
    }
    if current != baseline_set:
        await _send(ws, {
            "type": reply_type,
            "ok": False,
            "conflict": True,
            "message": f"{noun.capitalize()} changed on the server since it was loaded. "
                       "Refresh and try again.",
        })
        return

    removed = sorted(baseline_set - retained_set)
    if removed:
        from datetime import datetime
        from training.ledger import ConsolidationLedger

        ts = datetime.now().isoformat()
        rag_path = _MEMORY_DIR / ReflectionWriter.RAG_FILE
        rag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rag_path, "a", encoding="utf-8") as fh:
            for key in removed:
                fh.write(json.dumps({
                    "op": "evict",
                    "ts": ts,
                    "key": key,
                    "reason": reason,
                }, ensure_ascii=False) + "\n")
        ConsolidationLedger(_CONSOLIDATION_DIR).evict(removed)
        _get_rag().refresh_reflection_memory()

    await _send(ws, {
        "type": reply_type,
        "ok": True,
        "removed": len(removed),
        "remaining": len(retained_set),
        "snapshot_touched": False,
    })


async def handle_update_persona(ws, msg: dict) -> None:
    """Apply Persona-tab removals to live memory and next-build inputs.

    The next reflection sees the changed evidence fingerprint and regenerates the
    persona digest; the next training build folds the same evicted ledger state.
    """
    await _apply_memory_eviction(
        ws, msg, kind="persona", reply_type="persona_updated",
        reason="persona_editor", noun="persona",
    )


async def handle_update_facts(ws, msg: dict) -> None:
    """Apply Facts-tab removals to live memory and next-build inputs.

    The fact counterpart of :func:`handle_update_persona`. An evicted ``[fact]`` stops
    being recalled in chat and stops being injected into its host exchange's CoT as an
    "I know that …" line; if it was attributed and non-hearsay it also leaves that
    person's next standing portrait, which is folded from live memory.
    """
    await _apply_memory_eviction(
        ws, msg, kind="fact", reply_type="facts_updated",
        reason="facts_editor", noun="facts",
    )


async def handle_get_prompt_deltas(ws) -> None:
    """Return the logged-only prompt-mutation proposals for the Debug tab.

    The standing-prompt counterfactual (``core.prompt_mutation``) appends a record
    whenever a drifted exchange suggests a concrete standing-prompt change. This is a
    read-only dump of that op-log, newest first; it never mutates any prompt or state.
    """
    try:
        from core import prompt_mutation
        from training.reflections_path import prompt_dir
        deltas = prompt_mutation.read_prompt_deltas(prompt_dir(), limit=200)
    except Exception as e:
        await _send(ws, {"type": "error", "message": f"get_prompt_deltas failed: {e}"})
        return
    await _send(ws, {"type": "prompt_deltas", "deltas": deltas})


async def handle_dedup_facts(ws, msg) -> None:
    """Semantic de-duplication of live ``[fact]`` RAG items (Debug tab "Dedup facts").

    Folds the live facts from ``rag_memory.jsonl``, blocks them by SUBJECT (recall-cue
    embedding on the CPU, so a paraphrase shares a grouping call with its original — the
    load-bearing step, see ``core.fact_dedup``), groups paraphrases / cross-lingual
    restatements by **meaning** on the CLEAN base (adapter OFF — an evaluation of what's
    redundant, not Ava's expression, so it's replay-faithful and immune to a bad adapter),
    then, unless ``dry_run``, evicts the duplicates — keeping one survivor per group that
    inherits the UNION of the group's triggers — and rebuilds the reflection-memory index
    in place so it takes effect without a restart.

    ``dry_run`` (default true) returns the proposed merges without writing, so the operator
    can preview before committing. Runs on the GPU executor thread inside a
    ``CleanBaseSession`` (two full model reloads, exclusive GPU), refused while another
    executor-monopolising job holds the box. Facts only; persona/asks are untouched.

    Streams a ``dedup_stage`` event per block (like ``resolve_contradictions``): the
    grouping is dozens of sequential clean-base calls, and one that reports nothing is
    indistinguishable from a hang for the tens of minutes it runs.
    """
    dry_run = bool(msg.get("dry_run", True))
    if _host_busy():
        await _send(ws, {"type": "facts_deduped", "skipped": "busy",
                         "message": "The GPU is busy with another job; try again shortly."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "facts_deduped", "skipped": "no_model",
                         "message": "No model is loaded."})
        return

    from core import fact_dedup

    loop = asyncio.get_event_loop()

    def _emit(payload: dict) -> None:
        """Push a progress event to the client from the executor thread (best-effort)."""
        try:
            asyncio.run_coroutine_threadsafe(_send(ws, payload), loop)
        except Exception:
            pass

    def _blocking() -> dict:
        memory = ReflectionMemory(_MEMORY_DIR)
        facts = [
            r for r in memory.live_items()
            if r.get("kind") == "fact" and (r.get("content") or "").strip()
        ]
        if len(facts) < 2:
            return {"before": len(facts), "dry_run": dry_run, "groups": [],
                    "evicted": 0, "note": "fewer than two facts — nothing to dedup"}

        # Subject blocking uses the (CPU) RAG embedder — independent of the GPU model, so
        # it is unaffected by the swap below; mirrors resolve_contradictions.
        embedder = _get_rag()._get_embedder()
        def _embed(texts):
            return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        def _on_stage(info: dict) -> None:
            print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"[dedup] {info}", flush=True)
            _emit({"type": "dedup_stage", "dry_run": dry_run, **info})

        # Clean base: the grouping is an evaluation, so run it adapter-OFF. The reflect
        # generate reads model state at call time, so it targets the swapped-in base.
        def _run_grouping():
            generate = generation._make_sync_reflect_generate(_get_rag())
            return fact_dedup.cluster_facts(facts, generate, embed_fn=_embed,
                                            on_stage=_on_stage)

        groups = _with_clean_base(_run_grouping)
        if not groups:
            return {"before": len(facts), "dry_run": dry_run, "groups": [],
                    "evicted": 0, "note": "no grouping produced"}

        merges = fact_dedup.plan_merges(groups)
        report = fact_dedup.summarize(merges)
        if dry_run or not merges:
            return {"before": len(facts), "dry_run": True, "groups": report,
                    "evicted": 0}

        writer = ReflectionWriter(_MEMORY_DIR, _CONSOLIDATION_DIR)
        counts = writer.write_dedup(merges)
        _get_rag().refresh_reflection_memory()
        after = len([
            r for r in ReflectionMemory(_MEMORY_DIR).live_items()
            if r.get("kind") == "fact" and (r.get("content") or "").strip()
        ])
        return {"before": len(facts), "after": after, "dry_run": False,
                "groups": report, "evicted": counts["evicted"],
                "reinserted": counts["reinserted"]}

    try:
        result = await loop.run_in_executor(_executor, _blocking)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "facts_deduped", "skipped": "error",
                         "message": f"dedup_facts failed: {e}"})
        return
    finally:
        _mark_activity()
    await _send(ws, {"type": "facts_deduped", **result})


async def handle_reconcile_self(ws, msg) -> None:
    """Self-reconciliation of live ``[persona]`` / ``[fact]`` items (Sleep tab "Reconcile self").

    Folds the live persona/fact items and judges each against Ava's CURRENT persona
    digest on the CLEAN base (adapter OFF — an evaluation against a digest she authored
    on the adapter, not her expression, so it's replay-faithful and immune to a bad
    adapter). Items she has grown past (persona) or that a later understanding has made
    stale/contradicted (fact) are proposed for SUPERSEDE — a *soften, not delete* move:
    unless ``dry_run``, each is dropped from live recall (``write_supersede``) AND flagged
    ``superseded`` in the consolidation ledger (``ConsolidationLedger.supersede``), which
    keeps the anchor as evidence-of-change while removing it from active persona evidence
    and training. Append-only ⇒ reversible; the reflection-memory index is rebuilt in
    place so recall changes without a restart.

    ``dry_run`` (default true) returns the proposal without writing, so the operator can
    preview before committing. Runs on the GPU executor thread inside a ``CleanBaseSession``
    (two full model reloads, exclusive GPU), refused while another executor-monopolising
    job holds the box. Wiring this into the reflection loop is a separate, later task.
    """
    dry_run = bool(msg.get("dry_run", True))
    if _host_busy():
        await _send(ws, {"type": "self_reconciled", "skipped": "busy",
                         "message": "The GPU is busy with another job; try again shortly."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "self_reconciled", "skipped": "no_model",
                         "message": "No model is loaded."})
        return

    from core import self_reconcile, reflection_digest
    from training.reflections_path import persona_dir

    loop = asyncio.get_event_loop()

    def _emit(payload: dict) -> None:
        """Push a progress event to the client from the executor thread (best-effort)."""
        try:
            asyncio.run_coroutine_threadsafe(_send(ws, payload), loop)
        except Exception:
            pass

    def _blocking() -> dict:
        memory = ReflectionMemory(_MEMORY_DIR)
        items = [
            r for r in memory.live_items()
            if r.get("kind") in ("persona", "fact") and (r.get("content") or "").strip()
        ]
        n_persona = sum(1 for r in items if r.get("kind") == "persona")
        n_fact = len(items) - n_persona
        if not items:
            return {"before": 0, "dry_run": dry_run, "report": None,
                    "note": "no live persona/fact items to reconcile"}

        def _log(m: str) -> None:
            print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"[reconcile] {m}", flush=True)

        digest = reflection_digest.latest_digest(persona_dir())
        digest_text = reflection_digest.render_digest_for_reconcile(digest) if digest else ""
        _log(f"{len(items)} live items ({n_persona} persona, {n_fact} fact); "
             f"digest present={digest is not None}, yardstick={len(digest_text)} chars")
        if not digest_text.strip():
            _log("no usable persona digest — skipping (run a reflection first)")
            return {"before": len(items), "n_persona": n_persona, "n_fact": n_fact,
                    "dry_run": dry_run, "report": None,
                    "note": "no persona digest yet — run a reflection first so there is a "
                            "self to reconcile against"}

        # The set can be thousands of items — far more than one call can fit / emit
        # decisions for — so it is judged in batches (self_reconcile.judge_items).
        from core.self_reconcile import BATCH_SIZE
        n_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
        _log(f"judging in {n_batches} batch(es) of up to {BATCH_SIZE} on the clean base…")
        _emit({"type": "reconcile_stage", "stage": "start", "items": len(items),
               "n_persona": n_persona, "n_fact": n_fact, "batches": n_batches,
               "batch_size": BATCH_SIZE, "dry_run": dry_run})

        # Clean base: the judgment is an evaluation, so run it adapter-OFF. The reflect
        # generate reads model state at call time, so it targets the swapped-in base.
        raw_holder: dict = {}
        totals = {"supersede": 0}
        def _on_batch(i, nb, ni, nsup) -> None:
            totals["supersede"] += nsup
            _log(f"  batch {i}/{nb}: {ni} item(s) → {nsup} SUPERSEDE")
            _emit({"type": "reconcile_stage", "stage": "batch", "i": i, "n": nb,
                   "n_items": ni, "n_supersede": nsup,
                   "running_total": totals["supersede"]})
        def _run_judge():
            generate = generation._make_sync_reflect_generate(_get_rag())
            def _wrapped(user_content, system_content, **kw):
                out = generate(user_content, system_content, **kw)
                if "raw" not in raw_holder:      # keep the FIRST batch's raw for triage
                    raw_holder["raw"] = out
                return out
            return self_reconcile.judge_items(items, digest_text, _wrapped,
                                              on_batch=_on_batch)

        decisions = _with_clean_base(_run_judge)
        raw = (raw_holder.get("raw") or "")
        _log(f"first-batch raw output ({len(raw)} chars): "
             f"{raw[:800].replace(chr(10), ' / ')!r}")
        if not decisions:
            _log("no parseable decisions in the judge output → no-op")
            return {"before": len(items), "n_persona": n_persona, "n_fact": n_fact,
                    "dry_run": dry_run, "report": None,
                    "note": "no reconciliation produced (nothing parseable came back — "
                            "see server.log '[reconcile]' lines for the raw output)"}

        n_supersede = sum(1 for v in decisions.values() if v[0])
        _log(f"parsed {len(decisions)} decision(s), {n_supersede} SUPERSEDE")
        plan = self_reconcile.plan_supersessions(items, decisions)
        _log(f"plan: {len(plan)} item(s) to soften"
             + (f" (dry-run — writing nothing)" if dry_run else " (applying)"))
        report = self_reconcile.summarize(plan, kept=len(items) - len(plan))
        if dry_run or not plan:
            return {"before": len(items), "n_persona": n_persona, "n_fact": n_fact,
                    "dry_run": True, "report": report, "superseded": 0}

        # Apply — soften both layers by key (RAG recall + ledger evidence/training).
        writer = ReflectionWriter(_MEMORY_DIR, _CONSOLIDATION_DIR)
        counts = writer.write_supersede(plan)
        from training.ledger import ConsolidationLedger
        led = ConsolidationLedger(_CONSOLIDATION_DIR)
        for p in plan:
            led.supersede([p["key"]], reason=p.get("reason", ""))
        _get_rag().refresh_reflection_memory()
        after = len([
            r for r in ReflectionMemory(_MEMORY_DIR).live_items()
            if r.get("kind") in ("persona", "fact") and (r.get("content") or "").strip()
        ])
        return {"before": len(items), "after": after, "n_persona": n_persona,
                "n_fact": n_fact, "dry_run": False, "report": report,
                "superseded": counts["superseded"]}

    try:
        result = await loop.run_in_executor(_executor, _blocking)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "self_reconciled", "skipped": "error",
                         "message": f"reconcile_self failed: {e}"})
        return
    finally:
        _mark_activity()
    await _send(ws, {"type": "self_reconciled", **result})


async def handle_resolve_contradictions(ws, msg) -> None:
    """Resolve contradictory live ``[fact]`` items (Sleep tab "Resolve fact conflicts").

    Corrections don't supersede: when the user corrects Ava, consolidation writes a NEW
    fact and the stale one stays live, so the store accumulates opposite claims about the
    same subject (``fact_dedup`` merges paraphrases, not contradictions; ``reconcile``
    judges against the persona, not fact-vs-fact). This groups live facts by subject
    (recall-cue embedding, CPU) and, for each multi-fact subject, asks the CLEAN base
    which facts directly contradict — then keeps the NEWEST in each conflict set (the
    correction) and SOFTENS the older ones (``supersede`` — dropped from recall + active
    evidence/training, kept as evidence-of-change, reversible).

    ``dry_run`` (default true) previews without writing; streams a ``contradict_stage``
    event per subject group so the Sleep log shows progress. Runs on the GPU executor
    thread inside a ``CleanBaseSession``; refused while another job holds the box.
    """
    dry_run = bool(msg.get("dry_run", True))
    if _host_busy():
        await _send(ws, {"type": "contradictions_resolved", "skipped": "busy",
                         "message": "The GPU is busy with another job; try again shortly."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "contradictions_resolved", "skipped": "no_model",
                         "message": "No model is loaded."})
        return

    from core import fact_contradict

    loop = asyncio.get_event_loop()

    def _emit(payload: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_send(ws, payload), loop)
        except Exception:
            pass

    def _blocking() -> dict:
        def _log(m: str) -> None:
            print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"[contradict] {m}", flush=True)

        facts = [
            r for r in ReflectionMemory(_MEMORY_DIR).live_items()
            if r.get("kind") == "fact" and (r.get("content") or "").strip()
        ]
        if len(facts) < 2:
            return {"before": len(facts), "dry_run": dry_run, "report": None,
                    "note": "fewer than two live facts — nothing to reconcile"}

        # Subject clustering uses the (CPU) RAG embedder — independent of the GPU model,
        # so cluster BEFORE the clean-base swap; only the per-group judgment runs on it.
        embedder = _get_rag()._get_embedder()
        def _embed(texts):
            return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        clusters = [c for c in fact_contradict.cluster_by_subject(facts, _embed)
                    if len(c) >= 2]
        _log(f"{len(facts)} live facts → {len(clusters)} multi-fact subject group(s) "
             f"to check on the clean base")
        _emit({"type": "contradict_stage", "stage": "start", "facts": len(facts),
               "groups": len(clusters), "dry_run": dry_run})
        if not clusters:
            return {"before": len(facts), "dry_run": dry_run, "report": None,
                    "note": "no subject has two or more facts — no contradictions possible"}

        totals = {"n": 0}
        def _on_group(i, n, gsize, nsup) -> None:
            totals["n"] += nsup
            _log(f"  group {i}/{n}: {gsize} fact(s) → {nsup} superseded")
            _emit({"type": "contradict_stage", "stage": "group", "i": i, "n": n,
                   "group_size": gsize, "n_superseded": nsup,
                   "running_total": totals["n"]})

        def _run():
            generate = generation._make_sync_reflect_generate(_get_rag())
            return fact_contradict.plan_supersessions(
                facts, _embed, generate, on_group=_on_group)

        plan = _with_clean_base(_run)
        _log(f"plan: {len(plan)} stale fact(s) to supersede"
             + (" (dry-run)" if dry_run else " (applying)"))
        report = fact_contradict.summarize(plan)
        if dry_run or not plan:
            return {"before": len(facts), "dry_run": True, "report": report,
                    "superseded": 0}

        writer = ReflectionWriter(_MEMORY_DIR, _CONSOLIDATION_DIR)
        counts = writer.write_supersede(plan)
        from training.ledger import ConsolidationLedger
        led = ConsolidationLedger(_CONSOLIDATION_DIR)
        for p in plan:
            led.supersede([p["key"]], reason=p.get("reason", ""))
        _get_rag().refresh_reflection_memory()
        after = len([
            r for r in ReflectionMemory(_MEMORY_DIR).live_items()
            if r.get("kind") == "fact" and (r.get("content") or "").strip()
        ])
        return {"before": len(facts), "after": after, "dry_run": False,
                "report": report, "superseded": counts["superseded"]}

    try:
        result = await loop.run_in_executor(_executor, _blocking)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "contradictions_resolved", "skipped": "error",
                         "message": f"resolve_contradictions failed: {e}"})
        return
    finally:
        _mark_activity()
    await _send(ws, {"type": "contradictions_resolved", **result})


async def handle_digest_dryrun(ws, msg) -> None:
    """Rebuild the persona digest end-to-end and show it — **writing nothing** (Sleep tab
    "Persona digest (dry)").

    Same two steps ``reflection_digest.run_digest_pass`` performs, but each on the model
    it belongs on, and with the clustering replaced by the map-reduce pass that survives
    a large corpus:

      1. **Cluster on the CLEAN base** (adapter off) — grouping paraphrases is an
         *evaluation* ("do these say the same thing?"), not Ava's expression, so it is
         run adapter-off like ``fact_dedup`` / ``self_reconcile``. The live pass instead
         shares one ``generate_fn`` with the synthesis and clusters on the adapter, which
         lets a drifting adapter group the very evidence that defines it.
         ``persona_cluster.run_map_reduce`` replaces the single flat grouping call that
         cannot fit ~640 statements in one context (see that module's header).
      2. **Synthesize on the ADAPTER** — authorship of the self-portrait stays with Ava,
         so the swap is exited before the digest prompt runs. Tokens stream to the client.

    Nothing is persisted: no ``persona_dir`` snapshot, no ``current`` pointer move, no
    RAG refresh. The active digest is read only to report what would change. The
    regenerate gate (``should_regenerate``) is deliberately bypassed — an operator asking
    for a preview means it.

    Streams ``digest_dryrun_stage`` (phase + per-block progress) and
    ``digest_dryrun_chunk`` (synthesis deltas), terminating in ``digest_dryrun_done``.
    Runs on the GPU executor inside a ``CleanBaseSession`` (two full model reloads), so it
    is refused while another job holds the box.
    """
    if _host_busy():
        await _send(ws, {"type": "digest_dryrun_done", "skipped": "busy",
                         "message": "The GPU is busy with another job; try again shortly."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "digest_dryrun_done", "skipped": "no_model",
                         "message": "No model is loaded."})
        return

    from core import persona_cluster, reflection_digest
    from training.reflections_path import consolidation_dir, persona_dir

    try:
        block_size = int(msg.get("block_size") or persona_cluster.DEFAULT_BLOCK_SIZE)
    except (TypeError, ValueError):
        block_size = persona_cluster.DEFAULT_BLOCK_SIZE
    block_size = max(2, min(200, block_size))
    try:
        temperature = float(msg.get("temperature", 0.7))
    except (TypeError, ValueError):
        temperature = 0.7

    loop = asyncio.get_event_loop()

    def _emit(payload: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_send(ws, payload), loop)
        except Exception:
            pass

    def _blocking() -> dict:
        def _log(m: str) -> None:
            print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"[digest-dry] {m}", flush=True)

        base = reflection_digest.gather_persona_raw(consolidation_dir())
        if not base:
            return {"skipped": "no_evidence",
                    "message": "No live persona evidence to cluster."}
        _log(f"gathered {len(base)} live persona statement(s); "
             f"clustering on the clean base in blocks of {block_size}")
        _emit({"type": "digest_dryrun_stage", "stage": "gathered",
               "items": len(base), "block_size": block_size})

        # -- 1. cluster, adapter OFF ---------------------------------------- #
        _emit({"type": "digest_dryrun_stage", "stage": "clean_base_enter"})

        # Route through the same seam the reflection run uses (`cluster_for_digest`), not
        # `persona_cluster` directly, so the preview can never drift from the live path.
        stats: dict = {}

        def _run_cluster() -> list:
            generate = generation._make_sync_reflect_generate(_get_rag())
            return reflection_digest.cluster_for_digest(
                base, generate_fn=generate, block_size=block_size, stats=stats,
                on_stage=lambda info: _emit({"type": "digest_dryrun_stage", **info}))

        evidence = _with_clean_base(_run_cluster)
        # stats stays empty if the map-reduce path degraded to a fallback tier, so read
        # it defensively — the report is informational, never load-bearing.
        counts = {k: stats.get(k, 0) for k in ("calls", "rejected_blocks", "map_themes")}
        _log(f"clustered {len(base)} → {len(evidence)} theme(s) in {counts['calls']} call(s); "
             f"{counts['rejected_blocks']} block(s) rejected by the blob guard")
        _emit({"type": "digest_dryrun_stage", "stage": "clustered",
               "themes": len(evidence), **counts})

        # -- 2. synthesize the portrait, back ON the adapter ----------------- #
        body = reflection_digest.build_digest_prompt_input(evidence)
        # build_digest_prompt_input drops themes under _PROMPT_WEIGHT_FLOOR (faded), so
        # count the bullets it actually emitted rather than assuming every theme made it.
        in_prompt = sum(1 for ln in body.splitlines() if ln.startswith("- ["))
        _emit({"type": "digest_dryrun_stage", "stage": "synthesizing",
               "themes_in_prompt": in_prompt, "themes_faded": len(evidence) - in_prompt,
               "prompt_chars": len(body)})
        generate = generation._make_sync_reflect_generate(_get_rag())
        text = generate(
            body, reflection_digest.load_digest_prompt(),
            temperature=temperature, top_p=0.9,
            max_new_tokens_setting="8192",
            before_session="", disable_rag=True,
            on_chunk=lambda d: _emit({"type": "digest_dryrun_chunk", "text": d}),
        )
        digest = reflection_digest.parse_digest(text)
        _log(f"synthesized: voice={'yes' if (digest.get('voice') or '').strip() else 'no'}, "
             f"{len(digest.get('stances') or [])} stance(s), "
             f"{len(digest.get('dispositions') or [])} disposition(s), "
             f"{len(digest.get('lines') or [])} line(s)")

        # What the ACTIVE digest looks like, for a side-by-side — read-only.
        current = reflection_digest.latest_digest(persona_dir()) or {}
        cur_ev = (current.get("evidence") or {})
        current_summary = {
            "run_id": current.get("run_id"),
            "created": current.get("created"),
            "themes": len(cur_ev.get("themes") or []),
            "largest_cluster": max([t.get("cluster_size", 1)
                                    for t in (cur_ev.get("themes") or [])] or [0]),
            "counts": {
                "stances": len(current.get("stances") or []),
                "dispositions": len(current.get("dispositions") or []),
                "lines": len(current.get("lines") or []),
            },
        } if current else None

        return {
            "dry_run": True,
            "items": len(base),
            "stats": stats,
            "themes": [
                {"content": e.get("content", ""),
                 "recurrence": e.get("recurrences", 1),
                 "weighted_recurrence": e.get("weighted_recurrence"),
                 "cluster_size": e.get("cluster_size", 1),
                 "members": e.get("members") or []}
                for e in evidence
            ],
            "digest": {
                "voice": digest.get("voice", ""),
                "stances": digest.get("stances") or [],
                "dispositions": digest.get("dispositions") or [],
                "lines": digest.get("lines") or [],
            },
            "rendered": {
                "introduction": reflection_digest.render_digest_for_introduction(digest),
                "judge": reflection_digest.render_digest_for_judge(digest),
            },
            "current": current_summary,
        }

    try:
        result = await loop.run_in_executor(_executor, _blocking)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "digest_dryrun_done", "skipped": "error",
                         "message": f"digest_dryrun failed: {e}"})
        return
    finally:
        _mark_activity()
    await _send(ws, {"type": "digest_dryrun_done", **result})


async def handle_regen_persona(ws, msg) -> None:
    """Regenerate the persona digest and ACTIVATE it (Persona tab "Regen persona").

    The WRITE counterpart of :func:`handle_digest_dryrun`, driving the same three seams
    the reflection run drives — and on the same models, so the result can't differ in
    kind from what a Sleep run would have produced:

      1. ``plan_digest`` (model-free) with ``force=True`` — an operator pressing the
         button means it, so the unchanged-fingerprint gate is bypassed exactly as the
         Sleep tab's "Regen persona" checkbox bypasses it.
      2. ``cluster_for_digest`` on the **CLEAN base** (adapter off — grouping paraphrases
         is an evaluation), via the same map-reduce pass.
      3. ``synthesize_digest`` on the **ADAPTER** (authorship stays with Ava) — which
         WRITES the digest to the live ``hot/persona/`` dir, self-portrait included,
         exactly as ``reflection_runner._synthesize_persona_digest`` writes it.

    Then — the reason this exists at all — it mints and activates the persona version via
    ``snapshot_state.produce_persona(run_id, activate=True)``, the same call the Sleep
    run's ``persona`` stage makes: live chat reads her self-portrait through the
    ACTIVE-PERSONA pointer, not from the hot/persona dir the digest is written to, so
    without this step a regenerated digest never reaches a turn. GPU-free / filesystem-
    only, adapter unchanged; the previous persona snapshot stays on disk as rollback.
    Best-effort by the run's own rule: a failed snapshot leaves the digest written and
    the PREVIOUS persona active, reported honestly rather than failing the pass.

    Streams ``regen_persona_stage`` (phase + per-block clustering progress) and
    ``regen_persona_chunk`` (synthesis deltas), terminating in ``regen_persona_done``.
    Runs on the GPU executor inside a ``CleanBaseSession`` (two full model reloads), so
    it is refused while another job holds the box.
    """
    if _host_busy():
        await _send(ws, {"type": "regen_persona_done", "skipped": "busy",
                         "message": "The GPU is busy with another job; try again shortly."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "regen_persona_done", "skipped": "no_model",
                         "message": "No model is loaded."})
        return

    from core import persona_cluster, reflection_digest
    from training.reflections_path import consolidation_dir, persona_dir

    try:
        block_size = int(msg.get("block_size") or persona_cluster.DEFAULT_BLOCK_SIZE)
    except (TypeError, ValueError):
        block_size = persona_cluster.DEFAULT_BLOCK_SIZE
    block_size = max(2, min(200, block_size))
    try:
        temperature = float(msg.get("temperature", 0.7))
    except (TypeError, ValueError):
        temperature = 0.7

    loop = asyncio.get_event_loop()

    def _emit(payload: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_send(ws, payload), loop)
        except Exception:
            pass

    def _blocking() -> dict:
        def _log(m: str) -> None:
            print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"[regen-persona] {m}", flush=True)

        # Prefixed so the persona lineage shows what minted this version (a reflection
        # run's id is the bare timestamp; this is the operator's manual regeneration).
        run_id = "regen_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # -- 1. plan (model-free, forced — the operator asked) ---------------- #
        plan = reflection_digest.plan_digest(
            consolidation_dir(), persona_dir=persona_dir(), force=True)
        if not plan["should_run"]:
            return {"skipped": "no_evidence",
                    "message": "No live persona evidence to fold."}
        base = plan["base"]
        _log(f"gathered {len(base)} live persona statement(s); "
             f"clustering on the clean base in blocks of {block_size}")
        _emit({"type": "regen_persona_stage", "stage": "gathered",
               "items": len(base), "block_size": block_size, "run_id": run_id})

        # -- 2. cluster, adapter OFF ------------------------------------------ #
        _emit({"type": "regen_persona_stage", "stage": "clean_base_enter"})
        stats: dict = {}

        def _run_cluster() -> list:
            generate = generation._make_sync_reflect_generate(_get_rag())
            return reflection_digest.cluster_for_digest(
                base, generate_fn=generate, block_size=block_size, stats=stats,
                on_stage=lambda info: _emit({"type": "regen_persona_stage", **info}))

        evidence = _with_clean_base(_run_cluster)
        counts = {k: stats.get(k, 0) for k in ("calls", "rejected_blocks", "map_themes")}
        _log(f"clustered {len(base)} → {len(evidence)} theme(s) in {counts['calls']} call(s); "
             f"{counts['rejected_blocks']} block(s) rejected by the blob guard")
        _emit({"type": "regen_persona_stage", "stage": "clustered",
               "themes": len(evidence), **counts})

        # -- 3. synthesize + WRITE, back ON the adapter ----------------------- #
        # Through synthesize_digest — the seam the reflection run writes through — so
        # the on-disk artifact (digest + self-portrait + evidence summary) can never
        # drift from the live path. The generate wrapper only adds streaming.
        _emit({"type": "regen_persona_stage", "stage": "synthesizing",
               "themes": len(evidence)})
        inner_generate = generation._make_sync_reflect_generate(_get_rag())

        def _streaming_generate(content, system_prompt, **kw):
            kw.setdefault("on_chunk",
                          lambda d: _emit({"type": "regen_persona_chunk", "text": d}))
            return inner_generate(content, system_prompt, **kw)

        summary = reflection_digest.synthesize_digest(
            evidence, generate_fn=_streaming_generate, persona_dir=persona_dir(),
            run_id=run_id, raw_fp=plan["raw_fp"], temperature=temperature)
        if summary.get("status") != "written":
            return {"skipped": "not_written",
                    "message": f"Digest synthesis did not complete: "
                               f"{summary.get('reason', 'unknown')}"}
        _log(f"digest written — {summary.get('counts')}")
        _emit({"type": "regen_persona_stage", "stage": "digest_written",
               "counts": summary.get("counts") or {}})

        # -- 4. mint + activate the persona version (GPU-free) ---------------- #
        # snapshot_state lives at the server root, not on the inference package path —
        # same lazy path-injected import the Sleep run's `persona` stage uses.
        _emit({"type": "regen_persona_stage", "stage": "producing_persona",
               "run_id": run_id})
        activated = False
        persona_path = ""
        persona_error = ""
        try:
            import importlib
            import sys as _sys
            _root = str(Path(__file__).resolve().parents[1])
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            snapshot_state = importlib.import_module("snapshot_state")
            pdir = snapshot_state.produce_persona(run_id, activate=True)
            activated = True
            persona_path = str(pdir)
            _log(f"activated persona data/persona/{run_id} ({pdir})")
        except Exception as e:
            traceback.print_exc()
            persona_error = str(e)
            _log(f"persona production FAILED (digest is written; the previous "
                 f"persona stays active): {e}")

        digest = reflection_digest.latest_digest(persona_dir()) or {}
        return {
            "run_id": run_id,
            "items": len(base),
            "themes": len(evidence),
            "stats": stats,
            "counts": summary.get("counts") or {},
            "digest": {
                "voice": digest.get("voice", ""),
                "stances": digest.get("stances") or [],
                "dispositions": digest.get("dispositions") or [],
                "lines": digest.get("lines") or [],
            },
            "rendered": reflection_digest.render_digest_for_chat(digest),
            "activated": activated,
            "persona_path": persona_path,
            **({"persona_error": persona_error} if persona_error else {}),
        }

    try:
        result = await loop.run_in_executor(_executor, _blocking)
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "regen_persona_done", "skipped": "error",
                         "message": f"regen_persona failed: {e}"})
        return
    finally:
        _mark_activity()
    await _send(ws, {"type": "regen_persona_done", **result})


# ──────────────────────────────────────────────────────────────────────────────
# Clean-base swap — run a callable with the persona adapter swapped for the base
# (used once per reflection run for the branch judge; see reflection_digest)
# ──────────────────────────────────────────────────────────────────────────────

def _with_clean_base(fn: Callable):
    """Run ``fn()`` with the persona adapter swapped out for the frozen base.

    The agentic 'clean' mode (``CleanBaseSession``): a full unload + reload, so it is used
    ONCE per reflection run to batch the branch judge on the bare base (replay-faithful,
    mode-collapse-guarded, immune to a bad adapter). Executor-thread only — exclusive GPU.
    Passed to the runner as ``clean_base_ctx``; inside it the reflect ``generate_fn`` reads
    model state at call time, so it targets the clean base automatically."""
    from core import agentic

    def _prepare(tok, model_id) -> None:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if not getattr(tok, "chat_template", None):
            ensure_chat_template(tok, model_name=model_id, emit=lambda m: None)

    def _log(m: str) -> None:
        print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] {m}",
              flush=True)

    with agentic.CleanBaseSession(_backend, _runtime, prepare=_prepare, on_log=_log):
        return fn()


# ──────────────────────────────────────────────────────────────────────────────
# Autonomous idle jobs (outreach / synthesis / check-in / wander)
#
# The scheduling mechanism — the idle clock, the crash-safe wake-lock, per-job
# frequencies, and the GPU lock that serializes them — lives in core.idle_scheduler.
# Here we keep only the job-specific *policy* the scheduler asks for: the `consumed`
# classifiers deciding whether an attempt actually spent its interval (a cheap
# pre-generation bail — no open ask, no wiki page — should retry on the next poll rather
# than sleep a full hour; a pass that ran a generation should not). Each classifier reads
# ONLY its own job's result: these are per-job policy, never a cross-job gate. The jobs
# are registered in main() via idle_scheduler.register(...).
# ──────────────────────────────────────────────────────────────────────────────

def _outreach_consumed_interval(result: dict) -> bool:
    """True when an autonomous outreach attempt actually ran the decision pass (so it
    should wait a full interval before retrying). A pre-generation bail — no surfaceable
    ask yet — returns False so the scheduler retries next poll, in case one appears."""
    if not isinstance(result, dict):
        return True
    # `resolved` and `duplicate` ran the full decision pass and evicted an ask (self-
    # answered / already-raised); they cost the same generation as a compose, so they
    # spend the interval like one.
    if (result.get("composed") or result.get("resolved") or result.get("duplicate")
            or result.get("error")):
        return True
    skipped = str(result.get("skipped") or "")
    if skipped in {"declined", "truncated", "reachout_cooldown", "reachout_backoff"}:
        return True     # a decision pass ran but no session was written
    return False        # nothing to weigh yet — cheap, retry next poll


def _synthesis_consumed_interval(result: dict) -> bool:
    """True when a synthesis attempt picked a chat and ran the analysis pass; a
    no-candidate / no-model tick is cheap and should retry next poll."""
    if not isinstance(result, dict):
        return True
    return bool(result.get("analyzed") or result.get("composed") or result.get("error"))


def _checkin_consumed_interval(result: dict) -> bool:
    """True when a check-in wake actually ran a decision generation (composed / declined /
    truncated). A cheap insufficient-silence / no-history disk-check tick returns False so
    the scheduler retries next poll instead of sleeping a full interval — so once a user
    does go quiet past the threshold, the next poll picks it up.

    The wake decides per person, so `checkin._sweep_skip_reason` folds the people it weighed
    into ONE reason, preferring whichever cost a generation. `composed` is a count there
    (0 ⇒ falsy ⇒ fall through to the reason), a bool on the single-person manual path."""
    if not isinstance(result, dict):
        return True
    if result.get("composed") or result.get("error"):
        return True
    # A gate hold (`reachout_backoff` — she is talking into silence, so the window has
    # widened past the base hour) counts as spent like `reachout_cooldown`: the decision
    # generation already ran, and re-running it every 5-min poll for the length of a
    # day-long backoff would burn the GPU to reach the same held conclusion.
    return str(result.get("skipped") or "") in {"declined", "truncated",
                                                "reachout_cooldown", "reachout_backoff"}


def _wander_consumed_interval(result: dict) -> bool:
    """True when an autonomous wander actually read and reflected on a page. The cheap
    pre-generation bails — no model, no enabled source, or a fetch that turned up no
    substantive page (a network blip) — return False so the next poll retries instead of
    burning the whole interval on a miss."""
    if not isinstance(result, dict):
        return True
    return bool(result.get("applied") or result.get("error"))


def _background_reflection_consumed_interval(result: dict) -> bool:
    """True when a background wake did real work on any rung — deleted a stale reach-out,
    backfilled a sidecar, or reflected a chat (or was preempted after doing some). A pure
    "nothing to do" bail (nothing stale, no backlog / no model / host busy) returns False so
    the next poll retries promptly instead of sleeping the full interval."""
    if not isinstance(result, dict):
        return True
    if result.get("skipped"):
        return False
    return True


# ── unified activity log: per-job outcome phrasing + intra-run progress ───────────
# `describe` turns a *successful* summary dict into one journal line; the progress hooks
# route each subsystem's existing on_stage/on_question phase markers (NOT the raw token
# on_chunk stream — that would flood a persistent journal) into the same activity id the
# scheduler opened, so the Activity tab shows Ava's reasoning unfolding, not just a
# start/finish. Skips/errors are phrased generically by the scheduler. See core.activity_log.

def _short(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _describe_outreach(r: dict) -> str:
    if r.get("resolved"):
        return "Outreach: resolved an open question she'd already learned the answer to"
    if r.get("duplicate"):
        return "Outreach: retired a duplicate question she'd already put to the user"
    return f"Outreach: reached out — {_short(r.get('opener', ''))}"


def _describe_synthesis(r: dict) -> str:
    chat = os.path.basename(str(r.get("source_session") or r.get("session") or "a chat"))
    if r.get("raised"):
        return f"Synthesis: re-read {chat}, raised a question & reached out"
    return f"Synthesis: re-read {chat}, added questions to the pool (no reach-out)"


def _describe_checkin(r: dict) -> str:
    """One journal line for a check-in wake that reached out.

    The wake decides per person (`checkin.run_checkin_sweep_blocking`), so it may write to
    more than one; the sweep lifts the first send to the top level and the rest are counted
    here, since the journal wants a headline rather than a roster."""
    h = r.get("silence_hours")
    who = str(r.get("user") or "").strip()
    tail = f" after {float(h):.0f}h of silence" if isinstance(h, (int, float)) else ""
    others = max(0, int(r.get("composed") or 1) - 1) if r.get("sweep") else 0
    more = f" (+{others} other{'s' if others > 1 else ''})" if others else ""
    return (f"Check-in: reached out{' to ' + who if who else ''}{tail}{more} — "
            f"{_short(r.get('opener', ''))}")


def _describe_wander(r: dict) -> str:
    return f"Wander: read {_short(str(r.get('title') or 'an article'), 70)!r}"


def _run_worklog_sweep() -> dict:
    """Close the worklog threads an answer can no longer close (see core.chat_worklog).

    Ava's reach-outs open a thread ("awaiting their reply") that only a reflected reply
    closes — but an opener the user never answers is deliberately skipped un-frozen by both
    reflection paths, so its thread hangs forever and `open_threads()` fills with loops that
    are not actually open. This writes a real closing episode for each, after
    `chat_worklog.DEFAULT_STALE_HOURS`. GPU-free (a filesystem sweep); it rides the idle
    scheduler only because that is where the box's periodic work lives. Nothing is deleted:
    `core.reachout_gate` counts those same unanswered opener files for its backoff."""
    closed = chat_worklog.expire_stale_reachouts(_CHATS_DIR)
    if not closed:
        return {"skipped": "nothing stale to close", "closed": 0}
    return {"closed": len(closed)}


def _describe_worklog_sweep(r: dict) -> str:
    return (f"Worklog: let go of {int(r.get('closed', 0) or 0)} unanswered "
            f"reach-out(s) — nothing left to wait on")


def _describe_background_reflection(r: dict) -> str:
    """One journal line for a background wake, which is a three-rung ladder: it may have
    deleted stale reach-outs, backfilled gist/fact sidecars, reflected chats, or any
    combination (see core.background_reflection). Name the rungs that did something —
    "reflected 0 chat(s)" on a wake that spent an hour writing sidecars reads as a stall."""
    parts = []
    if int(r.get("deleted", 0) or 0):
        parts.append(f"deleted {int(r['deleted'])} unanswered reach-out(s)")
    if int(r.get("backfilled", 0) or 0):
        parts.append(f"backfilled {int(r['backfilled'])} chat sidecar(s) "
                     f"({int(r.get('summaries', 0) or 0)} gist, "
                     f"{int(r.get('facts', 0) or 0)} facts)")
    if int(r.get("reflected", 0) or 0) or not parts:
        parts.append(f"reflected {int(r.get('reflected', 0) or 0)} chat(s)")
    tail = "" if r.get("status") != "preempted" else " (preempted by a user turn)"
    return "Background reflection: " + ", ".join(parts) + tail


def _outreach_on_question(info: dict) -> None:
    activity_log.append("outreach", "progress",
                        f"Weighing whether to raise: {_short(info.get('question', ''))}",
                        phase="considering")


def _synthesis_on_stage(kw: dict) -> None:
    stage = str(kw.get("stage") or "")
    msg = {"picked": "Picked an aged chat to re-read",
           "analyzing": "Re-reading it against who she is now",
           "no_question": "Nothing new to wonder — no reach-out",
           "composing": "Composing an opener"}.get(stage, stage or "working")
    activity_log.append("synthesis", "progress", f"Synthesis: {msg}", phase=stage)


def _checkin_on_stage(kw: dict) -> None:
    stage = str(kw.get("stage") or "")
    # A wake decides per person and the stages of several arrive in sequence, so every line
    # names whose check-in it belongs to — otherwise one journal shows two people's recaps
    # and two verdicts with nothing to tell them apart.
    who = str(kw.get("user") or "").strip()
    if stage == "summarizing":
        msg = f"Recapping recent chats ({kw.get('i')}/{kw.get('n')})"
    elif stage == "recapped":
        # The recap IS the check-in's input: the decision pass reasons over these and
        # nothing else, so a journal that shows only "recapping 3/5…" and then a yes/no
        # can't tell a sound decision from one made on a garbled window. Body indented
        # 4 spaces, the same convention the reflection mirror uses (the Activity tab's
        # "Hide reflection detail" box collapses it back to the headline).
        #
        # Only a recap this wake actually GENERATED is journalled. The window freezes while
        # the user is away, so the two stable sources would each repeat the same text every
        # hour: a cached recap (that is the point of the cache — it was written the wake it
        # was generated), and reflection's stored gist (whose body already reached the
        # journal through the reflection run's own mirror, when it was written).
        if kw.get("cached") or kw.get("source") == "reflection":
            return
        recap = _short(str(kw.get("recap") or ""), 1200)
        head = (f"Check-in: recapped {kw.get('i')}/{kw.get('n')} — "
                f"{kw.get('when') or '?'} with {kw.get('user') or 'them'}"
                f"{' (hit the token cap)' if kw.get('truncated') else ''}")
        body = "\n".join(f"    {ln}" for ln in recap.splitlines() if ln.strip())
        activity_log.append("checkin", "progress",
                            head + ("\n" + body if body else ""), phase=stage)
        return
    else:
        target = f" with {who}" if who else ""
        msg = {"considering": f"Considering whether to check in{target}",
               "deciding": f"Deciding whether to reach out{target}",
               }.get(stage, stage or "working")
    activity_log.append("checkin", "progress", f"Check-in: {msg}", phase=stage)


# ──────────────────────────────────────────────────────────────────────────────
# Connection handler
# ──────────────────────────────────────────────────────────────────────────────

# Message dispatch: {msg_type -> handler}, grouped by call signature. This table
# is the server's protocol index — every client->server message type maps to the
# coroutine that services it. `cancel` is handled inline (it only sets an event).
_HANDLERS_WS_MSG_QUEUE = {          # handler(ws, msg, msg_queue)
    "generate": generation.handle_generate,
    "generate_ephemeral": generation.handle_generate_ephemeral,
    # Streams + stoppable, so it needs the queue to see a mid-generation `cancel`
    # (the main loop can't process one while awaiting this handler).
    "regenerate_exchange": generation.handle_regenerate_exchange,
}
_HANDLERS_WS_MSG = {                # handler(ws, msg)
    "load": handle_load,
    "branch_exchange": generation.handle_branch_exchange,
    "apply_regenerated_exchange": session_ops.handle_apply_regenerated_exchange,
    "set_training_ban": session_ops.handle_set_training_ban,
    "til_fetch": til_wander.handle_til_fetch,
    "til_lookup": til_wander.handle_til_lookup,
    "til_wander": til_wander.handle_til_wander,
    "til_apply": til_wander.handle_til_apply,
    "outreach_now": outreach.handle_outreach_now,
    "synthesis_now": synthesis.handle_synthesis_now,
    "checkin_now": checkin.handle_checkin_now,
    "deliberate_now": deliberation.handle_deliberate_now,
    "list_modules": modules.handle_list_modules,
    "list_module_inputs": modules.handle_list_module_inputs,
    "run_module": modules.handle_run_module,
    "prompt_experiment": prompt_experiment.handle_prompt_experiment,
    "set_prompt": prompt_experiment.handle_set_prompt,
    "revert_prompt": prompt_experiment.handle_revert_prompt,
    "prompt_experiment_status": prompt_experiment.handle_experiment_status,
    "persona_preview": reflection_service.handle_persona_preview,
    "start_reflection_run": reflection_service.handle_start_reflection_run,
    "reflection_run_status": reflection_service.handle_reflection_run_status,
    "stop_reflection_run": reflection_service.handle_stop_reflection_run,
    "get_reflection_run": reflection_service.handle_get_reflection_run,
    "reflection_run_events": reflection_service.handle_reflection_run_events,
    "activity_events": handle_activity_events,
    "get_worklog": handle_get_worklog,
    "start_encounter": encounter_run.handle_start_encounter,
    "encounter_events": encounter_run.handle_encounter_events,
    "dedup_facts": handle_dedup_facts,
    "digest_dryrun": handle_digest_dryrun,
    "regen_persona": handle_regen_persona,
    "reconcile_self": handle_reconcile_self,
    "resolve_contradictions": handle_resolve_contradictions,
    "update_persona": handle_update_persona,
    "update_facts": handle_update_facts,
    "clear_context": session_ops.handle_clear_context,
    "set_session_notes": session_ops.handle_set_session_notes,
    "set_reflection_feedback": session_ops.handle_set_reflection_feedback,
    "retry_last_exchange": session_ops.handle_retry_last_exchange,
    "get_session": session_ops.handle_get_session,
    "match_anchors": session_ops.handle_match_anchors,
    "load_session": session_ops.handle_load_session,
    "delete_session": session_ops.handle_delete_session,
    "reset_session_reflection": session_ops.handle_reset_session_reflection,
    "mark_corrupt": session_ops.handle_mark_corrupt,
    "rewrite_history": session_ops.handle_rewrite_history,
}
_HANDLERS_WS = {                    # handler(ws)
    "unload": handle_unload,
    "status": handle_status,
    "get_open_questions": handle_get_open_questions,
    "get_rag_artifacts": handle_get_rag_artifacts,
    "get_prompt_deltas": handle_get_prompt_deltas,
    "get_wander_log": til_wander.handle_get_wander_log,
    "get_token_stats": handle_get_token_stats,
    "get_reflection_prompts": reflection_service.handle_get_reflection_prompts,
    "list_reflection_runs": reflection_service.handle_list_reflection_runs,
    "encounter_status": encounter_run.handle_encounter_status,
    "stop_encounter": encounter_run.handle_stop_encounter,
    "list_sessions": session_ops.handle_list_sessions,
}


async def handle_client(websocket) -> None:
    global _active_ws
    # Sequential handoff: a new client takes over rather than being refused. All
    # session state (model, conversation, reflection runs) lives in the server-owned
    # runtime_state singletons, so switching desktop<->laptop just moves the viewport —
    # any in-progress reflection run keeps running untouched and the newcomer
    # re-adopts it by replaying the event log. Evict the incumbent by closing its
    # socket; ownership is fenced on identity (see the finally) so the evicted
    # handler's cleanup can't clobber the newcomer's state.
    previous = _active_ws
    _active_ws = websocket
    if previous is not None and previous is not websocket:
        print(f"Superseding client {previous.remote_address} with {websocket.remote_address}")
        try:
            await previous.close(code=4000, reason="Superseded by a new client")
        except Exception:
            pass
    print(f"Client connected: {websocket.remote_address}")

    msg_queue: asyncio.Queue = asyncio.Queue()

    async def _reader() -> None:
        try:
            async for raw in websocket:
                try:
                    await msg_queue.put(json.loads(raw))
                except json.JSONDecodeError:
                    await _send(websocket, {"type": "error", "message": "Invalid JSON."})
        except websockets.exceptions.ConnectionClosed:
            pass

    reader_task = asyncio.create_task(_reader())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(msg_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if reader_task.done():
                    break
                continue

            t = msg.get("type")
            if t in _HANDLERS_WS_MSG_QUEUE:
                await _HANDLERS_WS_MSG_QUEUE[t](websocket, msg, msg_queue)
            elif t in _HANDLERS_WS_MSG:
                await _HANDLERS_WS_MSG[t](websocket, msg)
            elif t in _HANDLERS_WS:
                await _HANDLERS_WS[t](websocket)
            elif t == "cancel":
                _cancel_event.set()
            else:
                await _send(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {t!r}",
                })
    except websockets.exceptions.ConnectionClosed:
        _cancel_event.set()
    except Exception as e:
        _cancel_event.set()
        print(f"Client handler error: {e}")
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        # Only release ownership if we still hold it; a newer client may have
        # already taken over, in which case the globals belong to it now.
        if _active_ws is websocket:
            _active_ws = None
            # A genuine disconnect (not a supersede) releases the active chat so it
            # stops being fenced from background reflection / the sidecar writers
            # for however long the user is away. See session_ops.release_active_session.
            try:
                session_ops.release_active_session(reason="client disconnected")
            except Exception as e:
                print(f"[session] release on disconnect failed: {e}")
        print(f"Client disconnected: {websocket.remote_address}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def _main_async(host: str, port: int) -> None:
    print(f"Ava inference server listening on ws://{host}:{port}", flush=True)
    # max_size=None disables the websockets 1 MB per-message frame cap: chat
    # transcripts (session_data out, start_reflection_run sessions in) routinely
    # exceed it — a single logged session can already be ~2 MB. This is a trusted,
    # single-user local link, so the cap buys no protection, only failures.
    #
    # ping_timeout=None keeps keepalive pings flowing (dead-peer detection) but
    # never force-closes the link when a pong is slow. A reflection run does long
    # GPU passes (branch generation/chooser can each take 100-400s) in the single
    # executor thread; that work holds the GIL in long stretches and starves the
    # asyncio loop, so the default 20s pong deadline would tear down the socket
    # mid-run — exactly the "client disconnected during a 200s chooser" failure.
    # A stale lock from a previous (possibly crashed) run must not wedge the heartbeat;
    # clear any lock this box can prove is dead at startup (see idle_scheduler).
    idle_scheduler.clear_stale_lock()

    async with websockets.asyncio.server.serve(
        handle_client, host, port, max_size=None, ping_timeout=None
    ):
        # Idle-job heartbeat: drives the autonomous wander/outreach/synthesis jobs
        # between conversations, each on its own frequency (see core.idle_scheduler).
        asyncio.create_task(idle_scheduler.run_loop())
        await asyncio.get_running_loop().create_future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ava Chat inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--http-port", type=int, default=8767, dest="http_port",
                        help="Management HTTP sidecar port (artifacts/export/chats/"
                             "precision; default: 8767)")
    parser.add_argument("--api-port", type=int, default=8000, dest="api_port",
                        help="Public OpenAI-compatible API port for external tools "
                             "(opt-in via server_config.json api.enabled; default: 8000)")
    parser.add_argument("--api-host", default=None, dest="api_host",
                        help="Bind address for the public API (default: api.host from "
                             "server_config.json, else --host)")
    args = parser.parse_args()

    # Verify the fragmentation guard actually engaged, and say so in server.log — the
    # env vars set at the top of this file are a request, not a fact (see the note
    # there). Runs BEFORE the model load below: the runtime force applies only to
    # segments created afterwards, and nothing model-sized exists yet at this point.
    # The facts-fetch OOMs ("fragmentation, ~3 GiB nominally free") carry the exact
    # split-segment fingerprint a silently-inactive guard produces.
    ensure_expandable_segments()

    config = _load_boot_config()
    model_id = config.get("model_id", "")
    adapter_id = config.get("adapter_id", None)
    context_length = int(config.get("context_length", 32768))
    # Reflection may pack a larger window than chat. The model is physically loaded
    # ONCE at the max of the two, so every path stays within the real max_seq_length.
    reflect_context_length = max(
        context_length, int(config.get("reflect_context_length", context_length)))
    load_context_length = max(context_length, reflect_context_length)
    base_quant = str(config.get("base_quant", "") or "")
    startup_load_in_4bit, startup_load_in_8bit = _resolve_base_quant(base_quant)

    # Live-chat repetition penalty (see _CHAT_REPETITION_PENALTY). A value of 1.0 or
    # null disables the penalty layer, leaving only the halt-only stop_on_repeat guard.
    global _CHAT_REPETITION_PENALTY
    _rp = config.get("chat_repetition_penalty", _CHAT_REPETITION_PENALTY)
    try:
        _rp = float(_rp) if _rp is not None else None
    except (TypeError, ValueError):
        _rp = None
    _CHAT_REPETITION_PENALTY = _rp if (_rp is not None and _rp > 1.0) else None
    print(f"Chat repetition penalty: {_CHAT_REPETITION_PENALTY or 'off'} "
          f"(stop_on_repeat guard always on)", flush=True)

    # Degeneration floor (chat/ephemeral/encounter + the reflect/agentic generate
    # factories; see the module header). Layer 1: min_p, a
    # relative-probability sampling floor removing the tail that seeds a collapse
    # (config "chat_min_p"; null/<=0 disables). Layer 2: a drifting-runaway halt the
    # verbatim stop_on_repeat can't see (config "chat_degen_guard": bool, default on;
    # thresholds overridable via "chat_degen": {window, min_gen, distinct_ratio,
    # top_freq}). Tune both on a manual tier-5 sampling-stability pass.
    _mp = config.get("chat_min_p", 0.02)
    try:
        _mp = float(_mp) if _mp is not None else None
    except (TypeError, ValueError):
        _mp = None
    _chat_min_p = _mp if (_mp is not None and _mp > 0.0) else None
    _degen_on = bool(config.get("chat_degen_guard", True))
    _degen_cfg = config.get("chat_degen", {}) or {}
    _degen_kw = {"degen_stop": _degen_on}
    for _ck, _ak in (("window", "degen_window"), ("min_gen", "degen_min_gen"),
                     ("distinct_ratio", "degen_distinct"), ("top_freq", "degen_top_freq")):
        # Null-tolerant: a `"window": null` style entry means "use the default"
        # (matching the null convention of the scalar chat knobs). Copying the None
        # through crashed every chat generate inside _DegenStop (`int < None`).
        if _degen_cfg.get(_ck) is not None:
            _degen_kw[_ak] = _degen_cfg[_ck]
    print(f"min_p floor: {_chat_min_p or 'off'}; degeneration guard: "
          f"{'on' if _degen_on else 'off'} (chat + reflect/agentic)", flush=True)

    _session.system_prompt = _load_system_prompt()
    _session.surface_template = _load_surface_template()
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)

    # Wire the generation layer (chat/branch/reflect-generate). It never imports
    # server; the other subsystems run their passes through its generate factories.
    # See core.generation.configure.
    generation.configure(
        send=_send, backend=_backend, cancel_event=_cancel_event, executor=_executor,
        get_rag=_get_rag, ensure_logger=_ensure_logger,
        get_reflection_writer=_get_reflection_writer,
        add_user_tokens=_add_user_tokens, read_token_economy=til_wander.wander_stats,
        mark_activity=_mark_activity,
        load_surface_template=_load_surface_template,
        chats_dir=_CHATS_DIR, memory_dir=_MEMORY_DIR,
        chat_repetition_penalty=_CHAT_REPETITION_PENALTY,
        chat_min_p=_chat_min_p, degen_kw=_degen_kw,
    )
    # Wire the session-CRUD handlers (it never imports server).
    # See core.session_ops.configure.
    session_ops.configure(
        send=_send, get_rag=_get_rag, chats_dir=_CHATS_DIR, data_dir=_DATA_DIR,
        is_reflection_active=lambda: reflection_service._reflection_run_active,
        mark_activity=_mark_activity,
    )
    # Wire the TIL/wander subsystem with the server capabilities it needs (it never
    # imports server). See core.til_wander.configure.
    til_wander.configure(
        send=_send, executor=_executor, backend=_backend,
        get_rag=_get_rag, get_reflection_writer=_get_reflection_writer,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        make_agentic_generate=generation._make_agentic_generate,
        read_user_tokens=_read_user_tokens,
        reflection_active=lambda: reflection_service._reflection_run_active,
        wander_tokens_per=_WANDER_TOKENS_PER,
        memory_dir=_MEMORY_DIR, consolidation_dir=_CONSOLIDATION_DIR,
        prompts_dir=_PROMPTS_DIR, til_dir=_SERVER_DIR.parent / "til",
    )
    # Wire the encounter subsystem with the server capabilities it needs (it never
    # imports server). See core.encounter_run.configure.
    encounter_run.configure(
        send=_send, executor=_executor, cancel_event=_cancel_event, backend=_backend,
        get_rag=_get_rag, sync_chat_generate=generation._sync_chat_generate,
        build_inference_conversation=generation._build_inference_conversation,
        temporal_anchor=generation._temporal_anchor, identity_line=generation._identity_line,
        mark_activity=_mark_activity,
        host_busy=_host_busy,
        chats_dir=_CHATS_DIR, prompts_dir=_PROMPTS_DIR,
    )
    # Wire the outreach subsystem (Ava-initiated chat) with the capabilities it needs
    # (it never imports server). See core.outreach.configure.
    outreach.configure(
        get_rag=_get_rag, get_reflection_writer=_get_reflection_writer,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        load_server_config=_load_server_config,
        memory_dir=_MEMORY_DIR, chats_dir=_CHATS_DIR, prompts_dir=_PROMPTS_DIR,
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the synthesis subsystem (re-read an aged chat, ask what she now wonders) with
    # the capabilities it needs (it never imports server). See core.synthesis.configure.
    synthesis.configure(
        get_rag=_get_rag, get_reflection_writer=_get_reflection_writer,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        chats_dir=_CHATS_DIR, data_dir=_DATA_DIR, prompts_dir=_PROMPTS_DIR,
        load_server_config=_load_server_config,
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the check-in subsystem (reach out after a stretch of user silence) with the
    # capabilities it needs (it never imports server). See core.checkin.configure.
    checkin.configure(
        get_rag=_get_rag,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        load_server_config=_load_server_config,
        chats_dir=_CHATS_DIR, prompts_dir=_PROMPTS_DIR,
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the deliberation pass (the worklog read side: Ava reads her recent episodic
    # worklog and decides what to do next). DRY RUN / manual-only for now — it decides and
    # streams reasoning but dispatches nothing. Never imports server. See core.deliberation.
    deliberation.configure(
        get_rag=_get_rag,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        prompts_dir=_PROMPTS_DIR, load_server_config=_load_server_config,
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the module workbench (Modules tab): run ONE pass against ONE chosen chat and
    # return what it produced, writing nothing. Manual-only; never imports server.
    # See core.modules.
    modules.configure(
        get_rag=_get_rag,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        prompts_dir=_PROMPTS_DIR, chats_dir=_CHATS_DIR,
        # The `til` input source (articles + news digests). Same tree the three
        # `server/til/fetch_*.py` scripts write into and `til_wander` derives.
        til_snippets_dir=_SERVER_DIR.parent / "data" / "til" / "snippets",
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the prompt-experiment subsystem (temporary, self-reverting standing-prompt
    # swap). It reloads the *base* prompt on revert via _load_base_chat_prompt (ignoring
    # any active experiment) and persists state under hot/prompt. Never imports server.
    from training.reflections_path import prompt_dir as _prompt_state_dir
    prompt_experiment.configure(
        get_rag=_get_rag,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        load_base_prompt=_load_base_chat_prompt,
        prompts_dir=_PROMPTS_DIR, state_dir=_prompt_state_dir(),
        send=_send, executor=_executor, mark_activity=_mark_activity,
        host_busy=_host_busy,
    )
    # Wire the reflection-run service with the server capabilities it needs (it never
    # imports server). See core.reflection_service.configure.
    reflection_service.configure(
        send=_send, executor=_executor, backend=_backend,
        get_rag=_get_rag, get_reflection_writer=_get_reflection_writer,
        get_run_store=_get_run_store, load_server_config=_load_server_config,
        make_sync_reflect_generate=generation._make_sync_reflect_generate,
        run_branch_exchange_sync=generation._run_branch_exchange_sync,
        sync_branch_chooser_content=generation._sync_branch_chooser_content,
        with_clean_base=_with_clean_base, mark_activity=_mark_activity,
        server_dir=_SERVER_DIR, data_dir=_DATA_DIR, chats_dir=_CHATS_DIR,
        memory_dir=_MEMORY_DIR, consolidation_dir=_CONSOLIDATION_DIR,
        reflection_runs_dir=_REFLECTION_RUNS_DIR, prompts_dir=_PROMPTS_DIR,
        watchdog_mgmt_url=_WATCHDOG_MGMT_URL,
    )
    # Wire the background per-chat reflection pass. It delegates the actual single-chat
    # reflection to reflection_service.run_chat_only_reflection (so it carries no generation
    # plumbing) and shares the chat cancel event so an incoming user chat can preempt it
    # instantly. host_busy=None: the idle scheduler already guarantees GPU exclusivity (the
    # job runs under its lock), so no extra guard is needed. See core.background_reflection.
    background_reflection.configure(
        reflect_one_chat_fn=reflection_service.run_chat_only_reflection,
        chats_dir=_CHATS_DIR, data_dir=_DATA_DIR,
        runs_dir=_REFLECTION_RUNS_DIR,
        # Rung 0 of the wake: the SAME stale-reach-out policy the reflection run's head
        # phase uses, so an idle box clears unanswered openers without waiting for an
        # operator Sleep run. Rung 1 writes gists, which are chat-RAG passages — hence the
        # index refresh. See core.background_reflection.
        stale_sweep_fn=reflection_service.run_stale_reachout_sweep,
        refresh_chats_fn=lambda: _get_rag().refresh_chat_index(),
        load_server_config=_load_server_config,
        host_busy=None, cancel_event=_cancel_event,
    )
    # Wire the generic idle-job scheduler and register the autonomous jobs. The scheduler
    # owns the shared idle clock, the crash-safe wake-lock, and the GPU lock that is the
    # ONLY thing serializing jobs (`_host_busy`/`_mark_activity` above are its bound
    # methods). `external_busy` reports the user-triggered GPU owners that are NOT idle
    # jobs. Each job carries its OWN interval_s (and may override the idle window), so
    # different events run at different frequencies — bump one here to re-cadence just
    # that job, with no effect on any other.
    # Unified activity journal — the single box-wide log every autonomous/GPU subsystem
    # writes to (idle jobs via the scheduler's _dispatch, reflection via its event mirror,
    # every background generation via the seam in core.generation). Recovers its seq
    # high-water from the file tail so a reconnecting UI resumes cleanly.
    _log_cfg = _load_server_config().get("logging") or {}
    activity_log.configure(_ACTIVITY_LOG_PATH, **_log_cfg)
    # ...and tee this process's stdout/stderr into it. Every subsystem already prints in
    # one shape (`[til] …`, `[wander] …`, `[idle] …`), so parsing that prefix turns the
    # whole existing corpus of prints into journal lines with no call-site changes — and
    # it is the only way to capture a foreign library's output (the offline train cycle
    # installs the same tee for unsloth). server.log is written through unchanged.
    activity_log.install_stdout_tee("server")
    # Shared reach-out gate — the single throttle outreach/synthesis/check-in honor before
    # cold-opening the user. Another leaf: it reads the chat corpus itself (durably, so a
    # restart can't reset a backoff) to count how many unanswered openers she has already
    # sent since the user last spoke, and widens the window accordingly.
    # It also reads the tombstones the stale-reach-out sweep leaves behind, so deleting an
    # unanswered opener does not erase the fact that she sent it (see core.chat_worklog).
    from core import reachout_gate as _reachout_gate
    _reachout_gate.configure(_CHATS_DIR, expired_log=_REACHOUT_EXPIRED_PATH)
    # First-person episodic worklog — Ava's durable, semantic record of what she did (one
    # entry per meaningful episode, in her own voice), distinct from the activity ring. A
    # leaf like activity_log: subsystems `from core import worklog; worklog.record(...)` at
    # each episode-close with no wiring. Recovers its id high-water from the file tail.
    # Nothing consumes it yet — the deliberation read side is a separate task; for now it
    # accumulates and backs the Worklog preview tab. See core.worklog.
    from core import worklog as _worklog
    _worklog.configure(_WORKLOG_PATH)
    idle_scheduler.configure(
        executor=_executor,
        # The GPU owners that are NOT scheduler-run idle jobs: reflection runs, encounters,
        # and the MANUAL (Sleep-tab debug) outreach/synthesis/check-in triggers — each sets
        # its own `_active` flag without going through the scheduler, so the GPU lock alone
        # wouldn't see them. A scheduler-run job already holds that lock, so listing them
        # here is only for their manual path (and harmlessly redundant otherwise). A new
        # idle-ONLY job needs no edit here; only a subsystem with its own manual GPU
        # trigger does.
        external_busy=lambda: (reflection_service._reflection_run_active
                               or encounter_run._encounter_active
                               or outreach._outreach_active
                               or synthesis._synthesis_active
                               or checkin._checkin_active
                               or deliberation._deliberation_active
                               or modules._module_run_active
                               or background_reflection._background_reflection_active),
        model_loaded=lambda: _runtime.model is not None,
        lock_file=_MEMORY_DIR / "wake.lock",
        idle_seconds=3600.0, poll_seconds=300.0,
        startup_note=f"{_WANDER_TOKENS_PER} tok/wander",
    )
    # Each job below is INDEPENDENT: its own clock, its own gates, no say over a sibling.
    # Registration order carries no priority — two jobs coming due together both fire and
    # simply queue on the scheduler's GPU lock. None of them takes a `ready` gate that
    # consults another job's state; the one genuinely shared policy — at most one
    # unprompted message per hour regardless of origin — is enforced by each job body at
    # the moment it would write a session (core.reachout_gate), so a job that loses that
    # race still runs its pass and still consumes its interval.
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="outreach", interval_s=3600.0,
        run=lambda: outreach.run_outreach_decision_blocking(
            on_question=_outreach_on_question),
        consumed=_outreach_consumed_interval,
        describe=_describe_outreach,
    ))
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="synthesis", interval_s=3600.0,
        run=lambda: synthesis.run_synthesis_blocking(on_stage=_synthesis_on_stage),
        consumed=_synthesis_consumed_interval,
        describe=_describe_synthesis,
    ))
    # Check-in: reach out after a stretch of USER silence (measured from disk inside the
    # blocking pass against checkin.silence_threshold_hours). The scheduler's idle window
    # is a coarse pre-gate; the real silence gate + recent-window review live in the job.
    # It runs the SWEEP, which decides once per person she talks to — silence and "what
    # have we been talking about" are questions about someone in particular, and one
    # unscoped pass answered them by mixing everyone together (see core.checkin). At most
    # `checkin.max_users` of them reach a generation per wake; the rest are measured off
    # disk and deferred to the next.
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="checkin", interval_s=3600.0,
        run=lambda: checkin.run_checkin_sweep_blocking(on_stage=_checkin_on_stage),
        consumed=_checkin_consumed_interval,
        describe=_describe_checkin,
    ))
    # Wander's token budget is its own precondition — it reads nothing but wander state.
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="wander", interval_s=3600.0,
        run=til_wander.run_autonomous_wander_blocking,
        ready=lambda: til_wander._wander_budget_available() >= 1,
        consumed=_wander_consumed_interval,
        describe=_describe_wander,
    ))
    # Background per-chat reflection: reflect unreflected chats incrementally after a long
    # USER-idle stretch (its own 30-min `idle_seconds`, longer than the reach-out jobs), so
    # the GPU is free and a returning user preempts it instantly. It drains one chat per
    # dispatch; `interval_s=120` re-arms quickly while a backlog remains. Does NOT reach out
    # or train — purely the per-chat consolidation/revision/branch, frozen `chat_reflected`.
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="background_reflection", interval_s=120.0, idle_seconds=1800.0,
        run=background_reflection.run_background_reflection_blocking,
        consumed=_background_reflection_consumed_interval,
        describe=_describe_background_reflection,
    ))
    # Worklog upkeep: write off the reach-out threads nothing can close (an opener the user
    # never answered is skipped un-frozen by both reflection paths, so no close site can
    # ever reach it). GPU-free, so it takes a short `idle_seconds` — waiting a full idle
    # hour would starve it on exactly the busy box where threads pile up — and always
    # consumes its interval (a sweep that found nothing stale still swept).
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="worklog_sweep", interval_s=3600.0, idle_seconds=300.0,
        run=_run_worklog_sweep,
        describe=_describe_worklog_sweep,
    ))
    # Facts-tree upkeep: fold the `.facts.json` protocols into `data/graph/tree.json`, the
    # file both fetch channels read. Nothing on the box was doing this — building was a
    # manual `python -m graph.build` — so a clean install had no tree and skipped every
    # fetch with `no_tree` forever while `graph.enabled` reported true, and a running box
    # froze its tree at the last hand-run build. Neither announces itself: an empty blob
    # from a stale tree looks exactly like an empty blob because nothing was relevant.
    # GPU-free (stdlib over a few dozen JSON files), so it takes `worklog_sweep`'s short
    # `idle_seconds` for the same reason — a full idle hour would starve it on the busy box
    # where the tree goes stale fastest. Skips when current, so it does not log hourly.
    idle_scheduler.register(idle_scheduler.IdleJob(
        name="graph_rebuild", interval_s=3600.0, idle_seconds=300.0,
        run=graph_rebuild.run_rebuild_blocking,
        describe=graph_rebuild.describe,
    ))
    # Wire + launch the management HTTP sidecar (artifacts/export/chats/precision).
    # It self-locates its paths and only needs a busy predicate so it refuses a
    # config write / chat import while a reflection run owns the GPU/config. It
    # runs in its own daemon thread — see core.mgmt_http.
    # Model gossip (GOSSIP.md): opt-in per box. When enabled, the sidecar exposes an
    # OpenAI-compatible /v1/chat/completions so a peer Ava's Encounter loop can talk to
    # this box as if it were vLLM. Disabled ⇒ the endpoint 404s. The generate callable
    # submits to the single GPU executor and blocks; is_busy gates it (503) while a
    # reflection run owns the GPU/config.
    _gossip_cfg = config.get("gossip") or {}
    # Default ON (2026-07-30): a box that pulls and restarts is immediately reachable by a
    # peer Ava with no config edit. Note the route carries no auth (GOSSIP.md §4.3 leaves
    # `gossip.api_key` present-but-unchecked), so on an untrusted network set
    # `gossip.enabled: false` or bind the sidecar to a trusted interface.
    _gossip_on = bool(_gossip_cfg.get("enabled", True))
    mgmt_http.configure(
        # Busy: a reflection run owns the GPU/config, OR a Training-review regeneration
        # transiently holds an alternate adapter in the shared runtime (its swap window
        # has the same "wrong or no model loaded" hazard as the clean-base batch).
        is_busy=lambda: (reflection_service._reflection_run_active
                         or generation._regen_swap_active),
        gossip_generate=generation._make_gossip_generate(
            log_transcripts=bool(_gossip_cfg.get("log_transcripts", True)),
        ),
        gossip_enabled=lambda: _gossip_on,
        gossip_peer_name=lambda: (_gossip_cfg.get("peer_name") or None),
    )
    print(f"Model gossip serving: {'on' if _gossip_on else 'off'}", flush=True)
    mgmt_http.start(args.host, args.http_port)

    # Public OpenAI-compatible API (core.api_http) — the endpoint external tools query.
    # Opt-in per box and on its OWN port (default 8000), deliberately separate from the
    # management sidecar above: that one hands out weights and the chat corpus, this one
    # is meant to be pointed at by tools the operator may not have written. Requests are
    # never logged (log_transcripts is forced off in _make_api_generate), so an external
    # tool's traffic reaches neither reflection nor the training corpus. Busy = a
    # reflection run OR an encounter owns the single GPU worker ⇒ 503; a background
    # per-chat reflection is preempted instead (interactive work wins).
    # Default ON (2026-07-30): a box that pulls and restarts is immediately queryable by an
    # external tool with no config edit. `api.enabled: false` turns the listener off; with
    # no `api.api_key` set it is UNAUTHENTICATED on whatever address it binds — start()
    # warns loudly on a non-loopback bind, and `api.host: "127.0.0.1"` limits it to the box.
    _api_cfg = config.get("api") or {}
    _api_on = bool(_api_cfg.get("enabled", True))
    if _api_on:
        api_http.configure(
            generate=generation._make_api_generate(
                client_system=str(_api_cfg.get("client_system", "append")),
                inject_rag=bool(_api_cfg.get("inject_rag", True)),
                inject_persona=bool(_api_cfg.get("inject_persona", True)),
            ),
            is_busy=lambda: (reflection_service._reflection_run_active
                             or encounter_run._encounter_active
                             or generation._regen_swap_active),
            api_key=str(_api_cfg.get("api_key", "") or ""),
            model_name=str(_api_cfg.get("model_name", "") or "ava"),
            default_max_tokens=_api_cfg.get("max_tokens", "75%"),
        )
        api_http.start(str(_api_cfg.get("host", "") or args.api_host or args.host),
                       int(_api_cfg.get("port", 0) or args.api_port))
    else:
        print("Public OpenAI API: off (api.enabled is false in server_config.json)",
              flush=True)

    if model_id:
        quant_note = f", base_quant={base_quant}" if base_quant else ""
        ctx_note = (f"context_length={context_length}"
                    + (f"/reflect={reflect_context_length}"
                       if reflect_context_length != context_length else ""))
        if adapter_id:
            print(f"Loading model {model_id} with adapter {adapter_id} ({ctx_note}{quant_note})...", flush=True)
        else:
            print(f"Loading model {model_id} ({ctx_note}{quant_note})...", flush=True)
        try:
            log_msgs: list[str] = []
            model, tokenizer = _backend.load(
                model_id, load_context_length, adapter_id,
                load_in_4bit=startup_load_in_4bit, load_in_8bit=startup_load_in_8bit,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if not getattr(tokenizer, "chat_template", None):
                ensure_chat_template(tokenizer, model_name=model_id, emit=log_msgs.append)
            for m in log_msgs:
                print(m, flush=True)
            _runtime.model = model
            _runtime.tokenizer = tokenizer
            _runtime.model_id = model_id
            _runtime.adapter_id = adapter_id
            _runtime.context_length = context_length
            _runtime.reflect_context_length = load_context_length
            _runtime.base_quant = base_quant
            # Drop main()'s own references. This frame NEVER exits — it ends in
            # asyncio.run() and runs for the life of the process — so leaving the model
            # bound here pins it in VRAM permanently, on top of whatever _runtime holds.
            # Every clean-base swap then has to fit TWO full models on the card, which on
            # a 31 GB box and an 18 GB model is impossible: the release frees nothing, the
            # reload is dispatched to cpu/disk, and both the swap AND its restore fail,
            # leaving the server with no model at all. handle_load() never had this
            # problem — its frame returns — which is why a client-loaded model could be
            # swapped and a startup-loaded one could not.
            del model, tokenizer
            print(f"Model loaded. {_backend.memory_status()}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
    else:
        print("Warning: no model_id set in server_config.json — server will start without a model.", flush=True)

    asyncio.run(_main_async(args.host, args.port))


if __name__ == "__main__":
    main()
