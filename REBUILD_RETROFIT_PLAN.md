# REBUILD_RETROFIT_PLAN.md — working plan for the wall-clock retrofit

Status: **COMPLETE (Steps 1–7 landed, 2026-07-06).** This is now the execution
record of the wall-clock retrofit; the design truth lives in the maintained docs
(`training/DESIGN.md`, `documentation/AVA_STATUS.md`, `documentation/AVA_OPEN_PROBLEMS.md`,
`documentation/AVA_CHANGELOG.md`). Companion to `REBUILD.md` (the *goal* spec).
**One sub-cut is deliberately deferred** (Step 6): retiring the now-unused sidecar
`stage`/`advance_stages`/`stage_of` machinery + `ledger.advance`/`note_fact_trained`
and `BuildHistory.age_of` — harmless dead code / honest instrumentation, churn-heavy
across `ledger`/`chat_sidecar`/`dialogue_source` + four selftests, no functional gain.
Below: built state, per-step outcomes with file/function targets, and the
code-archaeology insights, kept so a future session (e.g. the deferred cut, or the
validation redesign) doesn't re-derive them.

Key sources: `REBUILD.md` (goal), `server/training/DESIGN.md` (built system),
`documentation/AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift` (why rebuild exists).

---

## 0. One-paragraph orientation

The old REBUILD brief (build-count age) is **already fully implemented** — Phases 1–3
landed in the Jul-5 commits. The new brief (this retrofit) changes the *clock* from
"number of promoted builds since reflection" to **wall-clock hours since the chat**,
plus three additions (decoupled RAG/LoRA crossfade, forensic snapshots, cap-age user
contamination) and one stance change (a failed probe halts for a human instead of
auto-continuing). Because wall-clock age needs only a timestamp — and the sidecars
already carry both `reflected_at` and the chat timestamp (session-file stem) — **the
sidecars the GPU box is producing right now are forward-compatible; no data migration
is needed.** This is a clock swap in ~3 functions + two localized additions, not a
rebuild.

---

## 1. Built state (against the OLD brief) — file/function map

Landing commits: `d36a1cf` (Phase 1), `701a91c`+`ebd6cbd`+`c539ed6` (Phase 2),
`8e99814` (Phase 3), `f459d1b` (tier-5 probe), `23164e0` (fact placement).

- **Sidecar / reflect-once** — `server/inference/core/chat_sidecar.py`
  - `mark_reflected()` (~:300) stamps doc-level `reflected_at` once (ISO, `datetime.now()`).
  - `write_verdict()` (~:161) writes `exchanges[idx] = {verdict, target, target_source, regularizer}`.
  - `is_reflected()` (~:155) = presence of `reflected_at`.
  - Chat timestamp is the **session-file stem** (`<ts>.json` / `<ts>.state.json`).
- **Corpus assembly** — `server/training/build_dataset.py`
  - `build_dataset(...)` — gathers frozen sidecars from hot+archive, one `BuildRow`
    per exchange, injects persona/fact from the **ledger** per host exchange, sorts
    by `order_key=(session_ts, idx)`.
  - `age = build_history.age_of(reflected_at)` then `lr_multiplier(age, decay_cfg)`
    (stepped cumsum `3→5→6`). **← the clock to replace.**
  - `WANDER_LR_MULT = 1`; `PERSONA_INJECT_CAP=2`, `FACT_INJECT_CAP=2`.
  - `corpus_fingerprint(rows)` — sha256 of row identities+targets (reproducibility).
  - `BuildRow` fields: `messages, age, lr_multiplier, source, source_session,
    exchange_index, order_key, target, anchor, persona_keys, fact_keys`.
- **Build history** — `server/training/build_history.py`
  - `BuildHistory(models_dir)` → `models/builds.jsonl`.
  - `age_of(reflected_at)` = count of `promoted` lines with `ts > reflected_at`. **← retire as age source.**
  - `append(outcome, rows, base_lr, run_id, corpus_fingerprint, probe_summary, adapter_dir)`
    — **no `built_at`/`seed`/`snapshot_dir` yet.**
- **Train cycle** — `server/training/train_cycle.py`
  - `run_cycle()` (~:1054): loads `ccfg = ConsolidationConfig.from_dict(config["consolidation"])`,
    `build_history = BuildHistory(_MODELS_DIR)`, `rows = build_dataset(...)`.
  - From-scratch init: `_Fast.from_pretrained(model_id, …)` then `get_peft_model` (resume path deleted).
  - Per-step LR: `_OrderedSFTTrainer` overrides `_get_train_sampler → SequentialSampler`
    and `create_scheduler → LambdaLR(lr_lambda)` where `lr_lambda(step) =
    multipliers[step % len]`. Requires **batch=1, GA=1** (optimizer step i ↔ row i).
  - Collator: `_final_turn_collator` → `_keep_final_turn_only` (masked) OR
    `_apply_label_policy` (per-row `unmask_user`). Plumbing complete + selftested.
  - `unmask_col = [ex.get("unmask_user") …]` (~:1313) — **always False today** (build_dataset never sets it).
  - Promote path (~:1585+): `archive_adapter(run_id, adapter_dir)`, then
    `build_history.append(outcome="promoted"|"rejected", …)`. **No snapshot dir, no
    `built_at`.** Rejected → prints + keeps serving, then returns (no explicit halt/alarm).
- **RAG crossfade** — `server/inference/core/rag_engine.py`
  - `_reflected_at_for_session()` (~:367), `_age_of_session()` (~:385) = `build_history.age_of(...)`,
    `_chat_modifier()` (~:391) = `cfg.modifier_for_stage(age)`, `_consolidation_modifiers()` (~:404).
  - Already **evaluated at retrieval time** (good — keep). Single stage clock for RAG+LoRA (mirror — to decouple).
- **Decay config** — `server/training/decay.py`
  - `DecayConfig{base_variants, decay_steps, curve}`; `modifier(stage,N)` linear;
    `variants_for_stage`, `modifier_for_stage`.
  - `ConsolidationConfig.from_dict(cfg["consolidation"])`, `.for_type("dialogue"|"fact")`.
  - Config source: `server/server_config.json` → `consolidation` block.
- **Tier-5 probe** — `train_cycle.run_regression_probe` (~:648), tier-5 (~:900+): temp≈1.0
  sampling stability; language half **hard-coded Russian** (FIXME). Also `training/DESIGN.md`.
- **`unmask_user` label policy** — `server/training/label_policy.py` (`row_label_policy(unmask_user=…)`, selftested).

---

## 2. Decisions already made (do not re-litigate)

1. **Age origin = chat timestamp** (session-file stem), NOT `reflected_at`. `reflected_at`
   presence is only the frozen/bundle gate. (User confirmed: "24–48 hr since chat.")
   REBUILD §1/§5a already reworded to say this.
2. **Wall-clock, not build-count.** No veto-freeze on age; `builds.jsonl` stops being
   the age source.
3. **Decoupled crossfade paces**: `RAG_CAP_AGE` (~4d) > `LORA_CAP_AGE` (~3d); deliberate
   overlap as a handoff safety margin. RAG evaluated at retrieval, LoRA at build time.
4. **RAG-only youngest window**: multiplier 0 for `age < RAG_ONLY_WINDOW` (~24h) → no
   trainable row emitted at all for material younger than the window.
5. **Contamination**: at cap age emit the exchange twice — 5.0 masked + 1.0 `unmask_user`
   (split borrows from the 6.0 response budget; additive 6+1 is a knob).
6. **Validation = halt-and-alarm**, not auto-retry. False-negative freedom becomes a hard
   probe requirement → tier-5 Russian FIXME graduates to a prerequisite.
7. **Contamination target ≡ tier-5 language anchor** — one shared "user's voice" model;
   do them together.
8. **Sidecars need no migration.** Ledger stays the persona/fact store (read per-host);
   no `memory_entries` embed required.

---

## 3. Config additions (new `consolidation` sub-keys)

Add to `server/server_config.json → consolidation` and to `decay.py`
(`ConsolidationConfig.from_dict` + a new continuous evaluator). Suggested shape:

```jsonc
"consolidation": {
  // existing per-type dialogue/fact base_variants/decay_steps stay for the LR-ramp SHAPE
  "wall_clock": {
    "rag_only_window_h": 24,   // RAG_ONLY_WINDOW — ≈ reflection latency; multiplier 0 below this
    "lora_cap_age_h":    72,   // LORA_CAP_AGE — multiplier reaches cap here
    "rag_cap_age_h":     96,   // RAG_CAP_AGE — RAG weight reaches 0 here (> lora_cap)
    "base_lr":           1e-5, // §1 empirical
    "contamination": { "enabled": true, "dose": 1.0, "additive": false }
  }
}
```

The ramp's *relative* shape (`3→5→6`) still comes from the existing dialogue decay
curve; the wall-clock knobs only reindex it onto hours. Keep the continuous ramp a
pure function so it selftests GPU-free.

---

## 4. Ordered steps (check off as landed)

### Step 1 — Wall-clock age core  ✅ DONE (commit pending)  (PIVOT)
Landed:
- `decay.py`: `WallClockConfig` (parsed from `consolidation.wall_clock`; on `ConsolidationConfig.wall`)
  + pure helpers `parse_ts`, `wall_clock_age_hours`, `cumsum_curve`, `lr_multiplier_hours`
  (0 below window; continuous ramp through cumsum sample points 3→5→6 across
  `[window, lora_cap]`; capped past cap), and **`rag_weight_hours`** (decoupled RAG fade
  to 0 at `rag_cap_age_h` — a head-start on Step 2; still UNWIRED).
- `build_dataset`: dropped `build_history`/old `lr_multiplier`/`decay_curve`; added
  `built_at` param; age per bundle from the session stem; **skips** whole bundles whose
  `mult <= 0` (age < window). `BuildRow.age`/`lr_multiplier` now floats.
- `train_cycle.run_cycle`: computes `built_at = datetime.now().isoformat()` once, passes
  it to `build_dataset` (also feeds `build_history.append` in Step 3).
- `selftest.py`: new `test_wall_clock_age` (parse/hours/ramp/fade) + rewrote
  `test_build_dataset` for wall-clock (window skip, 3/5/6 at 24/48/72h, reproducibility
  vs pinned `built_at`); `test_config_from_dict` covers the wall knobs. **All 25 pass.**

**Step 2 note left for next session:** `test_rag_crossfade` + `rag_engine._age_of_session`
still use build-count `age_of` + `modifier_for_stage` — Step 2 swaps them to
`wall_clock_age_hours` + `rag_weight_hours` (already in `decay.py`). `age_of` on
`BuildHistory` stays until then.

### Step 2 — Decoupled crossfade in RAG  ✅ DONE (commit pending)
Landed:
- `rag_engine._age_of_session`: now wall-clock hours from the **chat ts** (session stem),
  evaluated at retrieval time (`datetime.now()`), returning `None` for an **unfrozen**
  chat — the frozen gate (`_reflected_at_for_session`) keeps an unreflected chat at full
  RAG (no fade) so it can't fall out of RAG before it's on the path to weights.
- `_chat_modifier` / `_consolidation_modifiers` / the inline copy in
  `_collect_chat_entries`: all unified onto `rag_weight_hours(age_h, wall)` (fades to 0 at
  `rag_cap_age_h` = 96h, decoupled LATER than the LoRA cap = 72h). `_dialogue_decay_config`
  deleted (dead). `BuildHistory` import + `self._build_history` dropped; guarded import now
  pulls `rag_weight_hours`/`wall_clock_age_hours` (no-op if training pkg absent → weight 1.0).
- `selftest.test_rag_crossfade` rewritten to the wall-clock composition (fresh→1.0/mult 0;
  72h→weight 0.25/mult 6 = the overlap; 96h→weight 0/mult 6). All 25 pass.

**Left for Step 6 (noted):** `test_chat_rag_decay` + the sidecar `stage`/`advance_stages`
machinery it exercises are now unused by chat-RAG (wall-clock replaced stage). They still
pass (testing `DecayConfig.modifier_for_stage` + `stage_of` primitives directly) — retire
with the rest of the stage machinery in Step 6.

### Step 3 — `built_at` + forensic snapshot  ✅ DONE (commit pending)
Landed:
- New `training/build_snapshot.py` (`write_snapshot`, `snapshots_root`) — dumb, GPU-free,
  best-effort writer of `models/snapshots/<build_id>/` with materialized COPIES:
  `sft_render.jsonl` (copied before deletion), `wander.json`, `persona_digest.json`
  (active `latest_digest`), `build_meta.json` (config + wall params + base_lr + seed +
  built_at + outcome + adapter pointer). A failure logs + returns None, never blocks a build.
- `build_history`: `append` gained `built_at`/`seed`/`snapshot_dir` + a `build_id` param
  (reuse the pre-generated id); new `new_build_id()` static so the snapshot dir can be
  named before the line is appended. `age_of` kept but re-docced as **instrumentation-only**
  (no longer the age source; retire in Step 6).
- `train_cycle`: `seed: int = 42` param + best-effort `transformers.set_seed` + `--seed`
  CLI; a nested `_record_build(outcome, adapter_dir, probe_summary)` writes the snapshot and
  appends the line, called in **both** the reject and promote paths (the reject snapshot is
  the triage packet). `built_at` (from Step 1) + `seed` now travel onto the line + snapshot.
- `selftest.test_build_snapshot`: copies are real (mutating the source render leaves the
  snapshot intact), meta/wander/persona round-trip, missing-render no-op, build line records
  the provenance fields. All 27 pass. New params are optional → back-compat (no other callers).

### Step 4 — Contamination trigger (§5e)  ✅ DONE (commit pending)
Landed:
- `build_dataset`: `BuildRow.unmask_user` field + `_contamination_rows(mult, age_h, wall)`
  helper. A cap-age (`age_h >= lora_cap_age_h`) chat exchange with contamination enabled
  emits **two** rows sharing the same `messages`/`target`/`anchor`: masked `cap−dose` (5.0)
  + `unmask_user=True` `dose` (1.0); additive mode → 6.0 masked + 1.0 unmask. Below cap /
  disabled → single masked row. `corpus_fingerprint` now folds `unmask_user`.
- `train_cycle`: `unmask_user` flows into the `examples` dict (→ `unmask_col`, already-built
  collator plumbing applies it per-row); `retention_anchors` excludes the unmask copies so a
  cap exchange contributes its anchor once; the build log reports `n_unmask`. Verified the
  per-step LR / `texts` / `unmask_col` / `multipliers` all stay 1:1 with `rows` after the
  split, and `_truncate_messages_to_fit` never mutates the shared `msgs` in place (safe share).
- `selftest.test_contamination`: cap → [1.0, 5.0] summing to 6.0 with exactly one unmask,
  copies share target; sub-cap → single 5.0; disabled → single 6.0; additive → [1.0, 6.0].
  `test_build_dataset` pins contamination OFF to isolate age/ramp. All 28 pass.

**Steps 1–4 = PoC-complete core done.** Remaining: Step 5 (validation — now *disabled +
deferred*, see below), Step 6 (retire stage machinery / regularizer vestige / `age_of`),
Step 7 (docs).

### Step 5 — Validation  ✅ DECIDED: DISABLED + deferred (commit pending)
Decision (2026-07-06, user): **disable validation entirely for now; it is a separate design
project.** Rationale: the tier-5 language alarm is hard-coded to a single (Russian) user but
there are genuine French/Greek/Hebrew users, and that same "user's voice" model is what §5e
contamination drifts toward — so it is not a threshold tweak. Near-term degradation shows in
live chat, and the adapter lineage + snapshots keep promotions reversible.
Landed:
- Master switch in its own import-light module `training/validation_switch.py`
  (`VALIDATION_ENABLED = False`) so BOTH sides honor one flag without importing the
  unsloth-heavy `train_cycle`. `train_cycle.run_cycle` force-skips the probe at a single
  chokepoint (overrides caller flag / Sleep checkbox); `train_cycle._VALIDATION_ENABLED` is
  now an alias of it. Probe code retained, parked. Sleep "Skip validation" checkbox inert.
- **Judge-force reconciled** (per user note "judge can force validation"):
  `reflection_service` gated its judge-override force-validation on the same switch — with
  validation off it no longer sets `skip_validation=False` (which `run_cycle` would override
  anyway) and no longer claims "the probe will gate the adapter"; it emits an honest event
  that the higher-risk judge cycle promotes UNGUARDED. `reflection_service.py:~518`.
- Documented: `training/DESIGN.md` (probe-section status banner), `AVA_STATUS.md` (Regression
  probe row → DISABLED), `AVA_OPEN_PROBLEMS.md` (new **Validation** section — the redesign
  requirements: per-user voice model shared with contamination, drift-vs-collapse separation,
  halt-vs-veto), `AVA_CHANGELOG.md` (2026-07-06), `REBUILD.md` §7 (stance marked superseded).

**Deferred to the validation redesign project** (NOT done here): the halt-on-failure
surfacing, and building the per-user voice/language model that both tier-5 and contamination
consume. `REBUILD.md §7`'s halt-vs-veto stance is moot while there is no gate.

### Step 6 — Minor cleanups  ✅ DONE (commit pending; two sub-cuts deferred)
Landed:
- **Regularizer vestige retired** end to end (writer/runner/stats/sidecar/dialogue_source).
  The IDEAL minority-copy regularizer was dead under the from-scratch build and its stats
  cells *misleading* (counted regularizers `build_dataset` never trains). Removed
  `resolve_regularizer_target`, the anchor/`write_verdict`/sidecar `regularizer` field, the
  `note_regularizer`/recompute in the runner, and the `regularizer_present`/
  `judge_branch_with_regularizer`/`_regularizer_block` stats (kept `judge_branch_overrides`,
  folded into the report's `judge` block). Old on-disk sidecars keep their ignored field.
  reflection_stats + 28 backbone selftests pass.
- **Fact-placement locality (§3)**: `_run_clean_base_fact_placement` restricts each fact's
  candidate host exchanges to its **own chat** (`c.session == fact.source_session`); a fact
  whose chat has no CoT-bearing exchange stays unhosted (waits in RAG).
- Docs: CLAUDE.md (render.py + reflection_stats + fact-placement lines corrected),
  AVA_CHANGELOG (2026-07-06 Step 6 bullet).

**Deferred (a further, separate cut — NOT done):** the now-unused sidecar
`stage`/`advance_stages`/`stage_of` machinery + `ledger.advance`/`note_fact_trained` and
`BuildHistory.age_of`. They are harmless dead code / honest instrumentation, and retiring
them is churn-heavy across `ledger`/`chat_sidecar`/`dialogue_source` + `test_ledger`/
`test_sidecar`/`test_chat_rag_decay`/`test_build_history` for no functional gain. The
2026-07-05 changelog already flagged the stage machinery as "a further, separate cut."

### Step 7 — Docs  ✅ DONE (commit pending)
Landed:
- `training/DESIGN.md`: rewrote the top supersession banner + the "Stage-aware RAG →
  age-keyed crossfade" and "What is verified vs not" sections for wall-clock age, the
  decoupled crossfade, forensic snapshots, contamination, and validation-disabled.
- `CLAUDE.md`: rewrote the `server/training/` overview + the training dir-block (added
  `build_dataset`/`build_history`/`build_snapshot`/`decay`/`validation_switch`/`label_policy`,
  fixed the train_cycle/decay/render lines); de-stale'd the hot/archive storage rows, the
  `consolidation.wall_clock` config shape, the reflection_digest judge-force line, and the
  `skip_validation` protocol line (all note validation is disabled).
- `documentation/AVA_STATUS.md`: updated the training/build-history/regularizer/fact-path/
  chat-archiving/RAG-crossfade rows + the rebuild-summary + reworded-variant rows.
- `AVA_CHANGELOG.md`: 2026-07-06 entries (Steps 1–6) already landed with their commits.
- Kept this plan file as the execution record (referenced from AVA_STATUS) instead of
  deleting — it tracks the one deferred cut.

**Retrofit complete.** The one remaining item is the deferred stage-machinery / `age_of`
cut above (Step 6), tracked here + in the changelog.

---

## 5. Insights / gotchas dug out of the code (save future-me the archaeology)

- **`reflected_at` vs chat ts.** `reflected_at` is `datetime.now()` at reflection time,
  so it lags the chat. Using it as the age origin double-counts reflection latency.
  The chat ts is the **session-file stem** — `build_dataset` already parses it as
  `session_ts` for `order_key`; reuse it. `reflected_at` is only the frozen gate.
- **`builds.jsonl` is not needed for age post-retrofit.** Its existing build-count-shaped
  lines are harmless; the file keeps living for reproducibility + instrumentation. So a
  box with an empty/short `builds.jsonl` (few/no promoted builds yet) loses nothing.
- **Per-step LR requires batch=1, GA=1 and a `multipliers` list 1:1 with rows.** The
  contamination split (Step 4) adds a second row for cap exchanges — the `multipliers`
  list must grow with it and stay index-aligned. `lr_lambda(step)=multipliers[step%len]`.
- **`unmask_user` reaches the collator only via a post-tokenization `add_column`** on
  `trainer.train_dataset` (SFT's `.map()` strips a pre-tokenization column). Guarded on
  an exact length match; `remove_unused_columns=(not _any_unmask)`. Plumbing is done —
  Step 4 only needs to *set* the flag on the right rows.
- **RAG crossfade is already retrieval-time.** Don't "fix" it to build-time; the target
  design wants exactly retrieval-time RAG + build-time LoRA (the correct lag).
- **`modifier_for_stage` is linear `(N−stage)/N`.** For the hours curve, reuse the same
  linear shape but keyed on `age_h/cap_h`. Keep it a pure function in `decay.py`.
- **Fact/persona are stored in the ledger, joined per-host** in `build_dataset` via
  `index_by_exchange(source_exchange)`. The §3 `memory_entries`-in-sidecar sketch was
  never adopted; do not add it — the ledger join works and is already selftested.
- **Regularizer is half-retired**: not written to new sidecars (`write_verdict` gets `""`),
  but `resolve_regularizer_target` is still computed into the in-memory summary and
  `note_regularizer` still counts it. Dead but harmless; Step 6 removes it.
- **Wander is one-shot, fixed multiplier, does not age** (a per-run decision in
  `build_dataset`). Leave as-is unless a knob says otherwise.
- **Selftest entry point**: `python -m training.selftest` (GPU-free). Add all new
  age/contamination cases there; it's the acceptance gate for Steps 1–4.
- **`train_cycle` is offline / model-unloaded** — runs via the watchdog `POST /job/train`
  hand-off with inference DOWN. Can't be exercised from a live server; the GPU smoke
  tests (REBUILD §10) need the box idle.

---

## 6. Verification strategy

- **GPU-free (every step 1–4):** extend `training/selftest.py`. Gates: RAG-only window
  emits no row; continuous ramp endpoints (window→3, cap→6); cap emits the 5+1 unmask
  pair; reproducibility (same corpus + `built_at` ⇒ same fingerprint + multipliers).
- **GPU smoke (box idle):** two builds, same corpus + `built_at` + seed → comparable
  probe results; a build >24h after one new reflected chat shows new rows last at
  multiplier 3, a <24h chat contributes RAG only; a rejected build halts + writes its
  snapshot + leaves the serving adapter.
- **Live RAG:** confirm a fresh chat retrieves at full weight and an aged one fades to 0
  by `rag_cap_age_h` (retrieval-time eval).

---

## 7. Open knobs (from REBUILD §11 — not blockers)

`base_lr` (1e-5 start); `rag_only_window_h` / `lora_cap_age_h` / `rag_cap_age_h` +
overlap width; exact ramp/crossfade curve shapes; `WANDER_LR_MULT` and whether wander
should age; contamination `dose` + split-vs-additive + binary-at-cap vs ramped;
whether CoT-less *keeps* need a floor LR; accumulation 1 vs grouped.
