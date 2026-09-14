# REBUILD.md — data-centric from-scratch training (implementation brief)

Status: **adopted 2026-07-05, not built.** This is the implementation brief for the
successor to the resumed-persistent-adapter training scheme, written to be executed
by a coding agent against this repository. Motivation and post-mortem:
`documentation/AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift` (the 2026-07
sequential fine-tune collapse). The built system this replaces is described in
`server/training/DESIGN.md`, which stays authoritative for the code until the
corresponding phase below lands. Read that file (and `CLAUDE.md`'s training/
reflection sections) before starting — this brief assumes its vocabulary.

Scope discipline: this is a **PoC of the idea**, not a platform rewrite. Prefer the
smallest change that realizes each phase; keep everything in "What stays unchanged"
untouched; when this brief marks a choice as free, choose the simplest option and
note it in code comments.

---

## 1. Core principle

The adapter becomes a **build artifact**: a pure function of (frozen base, bundle
corpus, config), recompiled from scratch after each reflection, discarded and
rebuilt at will. Never resume a prior adapter.

Why (condensed; full reasoning in the changelog entry of 2026-07-05):

1. **Cumulative optimizer drift becomes unrepresentable** — each adapter carries one
   trajectory's dose; a bad build is discarded, never inherited. The collapse
   failure mode cannot recur structurally.
2. **The corpus is the character, and it is debuggable** — a bad build means bad
   data: edit/delete the offending sidecar, rebuild.
3. **The accepted-forgetting contract dissolves** — consolidated material persists
   because it retrains at max strength in every build, not because a resumed
   adapter "holds" it.
4. **Validation vetoes are non-sticky** — a failed build keeps serving the previous
   adapter; the next build is genuinely different (new corpus). Completes the
   flow-tripwire semantics in `training/DESIGN.md → Veto semantics`.
5. **Replay collapses into training** — every build re-derives weights from data;
   builds are auditable and reproducible modulo seed.

**Repetition is replaced by LR; cycle-count is replaced by wall-clock.** The old
decay curve `[3,2,1]` was an installment plan (lifetime imprint = 6 verbatim copies
across cycles). Under rebuild each build delivers the full current dose in one pass,
so nothing needs counting in cycles — a bundle's dose is a **continuous function of
its wall-clock age** (time since its reflection), evaluated fresh at each build:

```
age(row)           = built_at − bundle.chat_ts                 # wall-clock hours
lr_multiplier(age) = 0                for age < RAG_ONLY_WINDOW        (~24h)
                     ramp 0 → 6       for age in [RAG_ONLY_WINDOW, LORA_CAP_AGE]
                     6 (capped)       for age ≥ LORA_CAP_AGE           (~72h)
effective_lr(row)  = base_lr × lr_multiplier(age(row))
```

Age is clocked from the **chat's own timestamp** (when the conversation happened —
the session-file stem), *not* from `reflected_at`. `reflected_at`'s presence is only
the frozen/bundle gate (reflect-once); the aging clock is the material's true age
"since chat," so the RAG-only window models the real pre-reflection span rather than
double-counting reflection latency.

- The **RAG-only window** (`age < RAG_ONLY_WINDOW`, ~24h, multiplier 0) models the
  span where the user keeps chatting *before* the chat has been reflected and built:
  the material can only live in RAG, so it carries no weight dose yet. Size it to
  your real reflection latency, not a round number — nightly reflection ⇒ ~24h;
  idle-wake reflection minutes after a session ⇒ much shorter.
- The ramp reproduces the old `[3,2,1]` **relative strength profile** as three
  sample points (`3 → 5 → 6`) on a now-**continuous** curve; the step quantization
  is gone. `LORA_CAP_AGE` (~72h) is where the multiplier reaches cap.
- Age is **wall-clock, and is *not* frozen on a failed build.** Under the validation
  stance (§7) a failed probe halts the pipeline for a human instead of auto-retrying,
  so there is no escalating-dose stall loop to defend against — plain wall-clock is
  both correct and simplest. Ages advancing during a halt→fix→rebuild investigation
  window is fine: the chats really are that old, and rebuild-from-scratch carries no
  accumulated damage. (An earlier draft clocked age against the last *promoted*
  build to make auto-retry safe; the halt semantics make that guard unnecessary.)
- "Archived" is a computed property (`lr_multiplier` at cap), not a directory move.
- The absolute `base_lr` is a new empirical parameter (the old 1e-5-per-copy
  calibration does not transfer). Start at `1e-5` and tune from the loss-vs-age
  curve (§7).

**Order is chronological, oldest first — required, not optional.** Ancient material
lays the foundation at high LR; fresh experience trains last at low LR and gets the
final word (the plasticity gradient). Chronological order is also what makes
per-step LR trivial (§5).

---

## 2. What stays unchanged (do not touch)

- The five-tier regression probe (`train_cycle.run_regression_probe`) and its
  flow-tripwire semantics (its *operational* reading is refined in §7 — a failure is
  a halt-and-triage alarm, not a routine veto); the promote path (save
  `adapter-<ts>`, repoint `server_config.json.adapter_id`, `archive_adapter` with
  run id).
- The watchdog job runner, `POST /job/train` hand-off, progress journal
  (`training/train_progress.py`), Sleep-tab UX.
- Render/inference parity: `render.render_example_text`, the gemma `<|channel>`
  handling, `_MARKERS` + `train_on_responses_only` + the `_final_turn_collator`
  wrapper with `_keep_final_turn_only` (final-turn-only loss masking stays exactly
  as is — it guards CoT erosion regardless of scheme).
- Persona/fact CoT injection mechanics: `persona_render.py`, `fact_render.fact_cot_line`,
  `PERSONA_INJECT_CAP=2`, `FACT_INJECT_CAP=2` (only their *source* changes, §4; the
  `FACT_TRAIN_CAP` counter dies, §6).
- The reflection engine's consolidation/revision phases, prompts, staging, run
  store/events/stats, the archive tree (`reflection_archive.py`), persona digest.
- The **ask lifecycle** (`[ask]` surface/resolve/evict in `rag_memory.jsonl` /
  `reflection_memory.py`). The crossfade (§6) replaces *fact/persona eviction
  counters only* — asks are a live conversational loop, not consolidation state.
  Do not delete the op-log or its fold.
- Chat logging, tension capture, branch replay primitives (`branch_replay.py`,
  `generate_from_ids*`).

---

## 3. Data model: the chat bundle

A bundle is the transcript plus everything reflection derived from it, colocated
and frozen. **Reuse the existing sidecar file** (`<ts>.state.json`, managed by
`core/chat_sidecar.py`) as the bundle sidecar — extend it, don't invent a parallel
file. Wander/news output uses the same shape with a `source` tag.

Sidecar additions (schema sketch; exact field names free):

```jsonc
{
  "reflected_at": "<iso ts>",        // set once; presence = bundle is frozen
  "exchanges": { "<idx>": {
      "verdict": "keep|revise",
      "target": "<think>…</think>\n…",   // ONE resolved target (see §4)
      "target_source": "original|ideal|branch|judge_branch"
  }},
  "memory_entries": [                 // fact/persona born from this chat
    { "kind": "fact|persona",
      "content": "…", "trigger": "…",
      "embedding": [ … ],             // pre-computed at reflection time
      "host_exchange": 3 }            // host is within THIS chat (data locality)
  ]
}
```

Notes:
- **Reflect-once:** a chat is reflected exactly once, when new, by its contemporary
  adapter; after `reflected_at` is set the sidecar is immutable (no re-reflection,
  no continue-staging, no judge overrides on old chats). Enforce with a guard in
  the runner: skip sessions whose sidecar is frozen.
- Fact **placement** becomes local: the clean-base placement judge's candidate set
  is restricted to exchanges of the fact's own chat (today it may consider the
  run's sessions; see `_run_clean_base_fact_placement` in `reflection_runner.py`).
- `weights_persona.jsonl` may remain as durable provenance (append-only), but the
  training path reads bundles, not it.
- Existing `ShareML` per-session documents (`reflection_shareml.py`) overlap with
  this concept; evolving them into the bundle target store is allowed but not
  required — the sidecar is the source of truth either way.

---

## 4. Phase 1 — reflection side: one resolved target + the CoT rule

Target resolution, frozen into the sidecar at reflection time (rationale: training
two alternatives of one prompt in every build is a standing contradiction; the
judged choice *is* Ava deciding):

| Original CoT | IDEAL exists | Branch phase | Resolved target |
| --- | --- | --- | --- |
| yes | any | runs; branches inherit the original CoT | judge's pick (branch / ideal / original) — `resolve_revision_target` semantics |
| no  | yes | **skipped** | IDEAL (CoT-bearing by existing invariant — see changelog: answer-only IDEALs are rejected upstream) |
| no  | no  | skipped | original; renders as gemma's empty-channel scaffold (existing fallback) |

Implementation points:

- Add the branch-skip: in the runner's branch-eligibility check, a revisable
  exchange whose stored `assistant_cot` is empty and whose revision produced an
  IDEAL is marked ineligible with a distinct skip reason (feeds the existing
  `RunStats` skip accounting). This also skips its judge job.
- The **IDEAL regularizer slot dies**: `resolve_regularizer_target`
  (`reflection_writer.py`) and the minority-copy render logic in
  `train_cycle._render_examples` are removed in Phase 2; reflection stops writing
  a regularizer field for new sidecars now.
- Rationale for the CoT rule, for context: under rebuild a CoT-less target recurs
  at up to max LR in *every* build — a permanent channel-erosion gradient. A
  CoT-bearing good-enough IDEAL beats a CoT-less optimal branch (reasoning channel
  is infrastructure; expression is content). It also deletes the dominant GPU cost
  of reflection for exactly the exchanges where branching produced degraded
  material anyway.

---

## 5. Phase 2 — training side: the from-scratch chronological build

Replace the middle of `train_cycle.run_cycle` (render → resume-train → advance)
with:

**(a) Corpus assembly.** `build_dataset(bundles, decay_config) → [(messages, age, source)]`,
a pure function (selftest-able, GPU-free):
- Gather all frozen bundles: `hot/chats` + `archive/chats` transcripts with
  sidecars, plus wander/news bundles. The hot/archive directory split no longer
  carries training semantics (keep the directories; ignore the distinction here).
- One row per revisable exchange: the conversation history rendered exactly as
  today (`dialogue_source.build_dialogue_anchor` assembles
  `<think>{cot}</think>\n{answer}` targets), with the sidecar's single resolved
  target as the final turn. Persona/fact CoT injection unchanged, but sourced from
  the bundle's `memory_entries` (host exchange within the same bundle) instead of
  the ledger fold.
- Sort rows chronologically (bundle timestamp, then exchange index). Wander/news
  rows sort by their own timestamps.
- `age(row)` = wall-clock hours between the bundle's **chat timestamp** (session-file
  stem) and the build's `built_at` (§1), capped at `LORA_CAP_AGE`; resolved fresh each
  build against `built_at` (never "now" — that is what keeps a build reproducible, §7).
  `reflected_at` presence gates inclusion (only frozen bundles train); it does not set
  the clock. Wander/news rows get a fixed multiplier: `WANDER_LR_MULT = 1` (one rung
  below fresh chats' 3× — external knowledge should not outweigh the relational corpus;
  tune later).

**(b) From-scratch LoRA init.** Delete the adapter-resume branch in `run_cycle`
(the `_Fast.from_pretrained(model_name=adapter_id, …)` path). Every build:
load the frozen base by `model_id`, then `get_peft_model` fresh. `--lora-r` now
always takes effect (the "resumed adapter keeps its baked-in rank" caveat dies).
The previous adapter is untouched on disk and keeps serving until promotion.

**(c) One sequential pass with per-step LR.** Single epoch, order preserved:
- Disable sampler shuffling (override `_get_train_sampler` → `SequentialSampler`,
  or an equivalent supported hook — verify against the installed TRL/unsloth
  versions; this is the one integration point with real version risk).
- Per-step LR: keep `lr_scheduler_type="constant"` semantics but wrap with a
  `LambdaLR`-style multiplier keyed to optimizer-step index. With
  `per_device_train_batch_size=1`:
  - simplest exact option: `gradient_accumulation_steps=1`, so optimizer step i
    ↔ row i and the multiplier is `lr_multiplier(age(row_i))`;
  - if accumulation > 1 is kept for throughput, rows in one accumulated group are
    chronological neighbors with near-identical multipliers — group rows so the
    multiplier is constant within a group, or accept the blend.
- Do **not** thread LR through per-example loss scaling: adamw normalizes constant
  loss scales away (second-moment division); the multiplier must reach the actual
  optimizer LR.
- Everything downstream of `trainer.train()` (probe → promote/veto → archive) is
  unchanged, except: on promotion, append a line to the build history (§7) instead
  of calling `ledger.advance` / `ChatSidecar.advance_stages` / the hot→archive
  move.

**(d) `sft_render.jsonl` / debug dump** (`_dump_training_debug`) stay — extend the
dump rows with `age` and `lr_multiplier` so a build's dataset is inspectable.

**(e) User contamination (entrainment) at cap age.** The `unmask_user` plumbing is
already built, **per-row, and selftested** (`label_policy.row_label_policy`, the
dataset's per-row `unmask_user` column, `_final_turn_collator` in `train_cycle.py`) —
only its *trigger* was deferred (§8; `train_cycle.py:1393` never sets the flag).
Rebuild gives it one: **at cap age, emit the exchange as two rows instead of one** — a
normal masked row at multiplier 5.0 and an `unmask_user=True` row at multiplier 1.0.
- Dose bookkeeping: Ava's response is still trained at the full cap — `5.0 + 1.0 = 6.0`
  (the unmasked copy trains the final assistant turn too; it merely *also* trains the
  final user turn) — and the **contamination dose** is the 1.0 on the unmasked copy.
  The split *borrows* contamination from the response budget so cap stays 6.0; the
  additive alternative (6.0 masked **plus** a 1.0 unmasked copy = 7.0 total) is a knob
  (§11).
- Gate it to **cap only, on purpose**: entrainment on the user's voice should follow
  only material durable enough to have fully consolidated — fresh/transient chats
  never contaminate. This is the plasticity gradient applied to *voice*: the longer
  the relationship persists, the more it shapes not just what Ava knows but how she
  speaks. (Consequence: on a fresh post-wipe corpus, §9, contamination is dormant for
  the first ~3d until anything reaches cap — synthesize old `reflected_at`s to
  exercise it in a smoke test.)
- **Tier-5 interaction — reconcile, do not ignore.** Contamination is *intentional*
  drift of Ava's language toward the user: the same axis tier-5's language half alarms
  on (§7). They coexist safely only if tier-5 measures drift toward
  *collapse/incoherence*, not "moved toward the user." The hard-coded-Russian anchor
  (the §7 FIXME) is in fact user-language-anchored, so the two are *aligned* — both
  need one shared model of "the user's voice." Fixing the tier-5 FIXME (generalize the
  anchor from hard-coded Russian to the actual user) and defining contamination's
  target are the **same task**; do them together.

---

## 6. Phase 3 — RAG crossfade

The same wall-clock age drives both sides of consolidation, but on **decoupled
paces**: as a bundle's training multiplier ramps to cap over `LORA_CAP_AGE` (~3d),
its RAG weight decays to zero over a slightly **longer** span `RAG_CAP_AGE` (~4d).
The deliberate ~day-wide overlap (weights already at full strength while RAG still
weakly retrieves) is a **safety margin on the handoff** — a mirror crossfade (the
two summing to a constant) would assume consolidation is *perfect*, i.e. that
whatever leaves RAG has fully arrived in weights, which is not yet earned. Let RAG
linger until that is measured; tighten the lag later.

- Evaluate the two on their natural clocks: **LoRA multiplier at build time**
  (discrete — correct as of the last build), **RAG weight at retrieval time**
  (continuous — always current from age). The weights lagging wall-clock is the
  *correct* lag: RAG is the fast/live store, weights are the slow/consolidated one.
- Reflection-memory items (facts/personas): retrieval score scaled by an
  age→weight curve reaching 0 at `RAG_CAP_AGE` — *computed from age*, not from
  eviction counters. This replaces `FACT_TRAIN_CAP` / `ledger.note_fact_trained` /
  `_evict_baked_facts` and the `from_weights` eviction ops.
- Chat-exchange RAG: same rule replaces the archive-move exclusion
  (`rag_engine` already has stage-aware scaling for facts — extend the pattern).
- Asks are exempt (they are not consolidation state; see §2).

Phase 3 is separable: Phases 1–2 are a complete PoC with today's RAG behavior
left as is (the archive move can even keep happening for RAG purposes only, as a
temporary shim, if that is the smaller diff).

---

## 7. Build history, the validation stance & forensic snapshots

**Build history.** New append-only `server/models/builds.jsonl`, one line per build
attempt: `{build_id, ts, built_at, run_id, corpus_fingerprint, rows, seed,
outcome: promoted|rejected, probe_summary, base_lr, snapshot_dir?, adapter_dir?}`.
It is the reproducibility record and the instrumentation spine. `built_at` is the
`as_of` timestamp every row age was computed against — required, since wall-clock
age (§1) is only reproducible relative to a frozen build time, not "now."
`builds.jsonl` is **no longer** the source of `age()` (age is wall-clock, not a
count of promoted lines).

**The validation stance: a failed probe is an alarm, not a routine veto.**
> **Superseded as *current behavior* (2026-07-06): validation is DISABLED**
> (`train_cycle._VALIDATION_ENABLED = False` — every build promotes unguarded). The
> monolingual tier-5 language alarm can't gate real multilingual users, and it shares its
> "user's voice" model with §5e contamination, so a correct probe is a **separate design
> project** (`documentation/AVA_OPEN_PROBLEMS.md → Validation`). The stance below is the
> *aspiration* for that redesign, not what runs today; §1's no-veto-freeze holds for a
> different reason now (there is simply no gate to freeze against).

The goal is a system where, given correct training data, the probe *always passes*. A
failure is therefore not something to auto-retry around — it is the signal that something
is wrong, and the pipeline **halts for a human**. This is why §1 needs no veto-freeze on
age. The load-bearing consequence: with no auto-retry safety net, a probe **false
negative** (a bad build the probe passes) now gets promoted and served unguarded — so
the probe's freedom from false negatives becomes a hard requirement, not a nicety.
Concretely, the tier-5 CoT-collapse / language-drift alarm is load-bearing and its
language half carries the hard-coded-Russian-user problem
(`training/train_cycle.py` / `training/DESIGN.md`): under this stance that is a
prerequisite, not cleanup — a half-wired tier-5 is a silent hole in the alarm.

**Triage protocol on failure — the snapshot is the first thing you open:**
1. Is the training data *technically coherent* (well-formed rows, expected shapes, no
   truncation / encoding damage)? If yes → the fault is **code/design**, and the
   snapshot reproduces the exact build for debugging.
2. If the data is **corrupt** → trace back through the snapshot's copies to see what
   could have corrupted it **at chat/reflection stage** (a bad resolved target, a
   mis-hosted fact, a malformed CoT); fix the source, rebuild.

**Forensic snapshot bundle.** Each promoted build — and *every* failed one, that is
the investigation packet — writes an **immutable** `snapshot_dir` holding
**materialized copies**, not references into live mutable state:
- the exact rendered training rows (the build's `sft_render.jsonl` — see below),
- the wander/news bundles used,
- the persona-digest version active for the build (id or copy),
- the config (`server_config.json` + effective decay/curve params + `base_lr` +
  `seed` + `built_at`),
- the produced adapter (promoted builds) or none (rejected).

Because the snapshot owns frozen copies, editing a live sidecar later (the §1
debuggability principle) cannot corrupt it — live corpus stays editable, the snapshot
stays evidence. This **dissolves the immutability-vs-debuggability tension**: the
archive is the mutable build *recipe*; the snapshot is a frozen *state*. It also
promotes `sft_render.jsonl` out of `scratch/` — for a snapshotted build the render is
copied in as a permanent artifact instead of deleted, cheap now precisely because
rebuild removed copy-multiplication (render ≈ 1× corpus, one row per exchange, not
6×). Frozen corpus + config + a later design fix = deterministic **regeneration of
the personality under the new design**: §1's "the corpus is the character, and it is
debuggable" taken to its endpoint — a failure is not a dead end but a fully-specified
reproduction case.

**Instrument from the first build** (with repetition gone, loss will *not* bottom near
zero — "loss looks healthy" becomes the norm, so the alarms move):

- **loss-vs-age curve**: log per-row loss against `age`/`lr_multiplier` (is ancient
  material converging harder than fresh?). Cheap: rows are already ordered, so
  logging per-step loss with the row's age tag suffices.
- Tier-5 absolute rates stay the sampling-collapse alarm; tiers 3–4 now compare
  against the currently-serving build (tier 4 is near-trivially satisfied — every
  trainable item is in every build).

---

## 8. Deletion list (conscious cuts, mapped to code)

| Cut | Where |
| --- | --- |
| Adapter resume path | `train_cycle.run_cycle` (`from_pretrained(adapter_id)` branch) |
| Stage advancement | `ledger.advance`, `ChatSidecar.advance_stages` / `advance_by_session`, `stage_of` consumers |
| Ledger as mutable training state | `training/ledger.py` fold stays only if Phase 3 still reads it; goal state: bundles are the only training source |
| Fact copy-count lifecycle | `FACT_TRAIN_CAP`, `ledger.note_fact_trained`, `_evict_baked_facts` |
| IDEAL regularizer slot | `reflection_writer.resolve_regularizer_target`, minority-copy logic in `_render_examples`, sidecar `regularizer` field (new sidecars) |
| hot→archive move as semantics | `train_cycle`'s destage move (keep as RAG shim only until Phase 3) |
| Branch generation for CoT-less exchanges | runner branch-eligibility (Phase 1) |
| Re-reflection / continue-staging / judge overrides on old chats | runner guard on frozen sidecars; Sleep-tab "Continue Staging" becomes inert for frozen sessions |
| Ordered replay | Retired after the wall-clock decay switch; replacement rebuild mechanism deferred. |
| Entrainment last-run user-unmask — **redefined, not cut** | `unmask_user` plumbing kept (built + per-row + selftested); its trigger ("final decay copy") is replaced by the **cap-age dose-split** (§5e): at cap, emit a 5.0 masked + 1.0 `unmask_user` pair. No longer deferred |

---

## 9. Migration & rollout

- The PoC may assume a **fresh corpus** (post-wipe) — that is the primary intended
  path given the collapsed lineage. For existing checkouts, a one-shot
  `training/migrate.py` step may synthesize frozen bundles from existing sidecars
  (`verdict`/`target` already live there; facts/personas can be back-filled from
  `rag_memory.jsonl` by `source_session`) — optional, best-effort, clearly logged.
- Phases land independently: Phase 1 (reflection writes single-target frozen
  sidecars) is compatible with the old train cycle reading them; Phase 2 switches
  training; Phase 3 switches RAG.
- Rollback story: previous adapters remain on disk; `builds.jsonl` + bundle corpus
  reproduce any build.

## 10. Acceptance criteria

- GPU-free: `build_dataset` selftests (in `training/selftest.py` style) covering
  chronological ordering, wall-clock age→multiplier mapping (RAG-only 0 below
  `RAG_ONLY_WINDOW`, continuous 3→5→6 ramp, cap at `LORA_CAP_AGE`) resolved against a
  fixed `built_at`, wander multiplier, single-target rendering (no regularizer
  copies), persona/fact injection sourced from bundles, CoT-rule row shapes
  (think-bearing vs empty-channel), and the cap-age contamination split (a cap-age
  bundle emits a 5.0 masked + 1.0 `unmask_user` pair summing to 6.0; a sub-cap bundle
  emits one masked row).
- GPU: two consecutive builds from the same corpus + same `built_at` + seed produce
  comparable probe results (reproducibility smoke test); a build run >24h after one
  new reflected chat shows the new rows last at multiplier 3, and a fresh <24h chat
  contributes RAG only (no trainable row).
- The probe runs unchanged and gates promotion; a rejected build leaves the serving
  adapter in place and **halts for triage** (validation stance, §7) — it writes its
  forensic snapshot and does not auto-retry.

## 11. Open empirical knobs (not blockers)

`base_lr` (start 1e-5); `RAG_ONLY_WINDOW` (~24h, ≈ reflection latency), `LORA_CAP_AGE`
(~72h), `RAG_CAP_AGE` (~96h) and the resulting overlap width; the exact ramp /
crossfade curve shapes; `WANDER_LR_MULT`; the contamination dose (1.0) and
split-vs-additive at cap, and whether it stays binary-at-cap or ramps with age (§5e);
whether CoT-less *keeps* (row 3 of the target table) need a floor LR instead of the
full ramp (decide after corpus stats exist); accumulation 1 vs grouped.
