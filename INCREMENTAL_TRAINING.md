# INCREMENTAL_TRAINING.md — incremental adapter training beside the full train (plan)

Status: **built 2026-10-03; not yet exercised on the GPU.** Planned against `42109aa`.
`train_mode` defaults to `full`, so a box keeps building full trains until the operator
sets `train_mode: incremental`. Companion to `REBUILD.md` (the full-train design this sits
beside) and `server/training/DESIGN.md`. The maintained truth is now
`documentation/AVA_STATUS.md` (row "Offline LoRA training (incremental build)"),
`MODULE_NOTES.md`, `AVA_CHANGELOG.md` and the schema; §8 records how the build differs
in detail from the plan.

## 1. Why

The from-scratch build (`REBUILD.md`) was motivated by the July 2026 collapse, read at
the time as cumulative drift from stacking fits on one resumed adapter
(`AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift`). The summer experiments then showed
the full retrain did **not** cure the collapse. The cause was on the inference side:
the RAG block combined with a repetition penalty that also penalized prompt tokens, so
an injected block primed Ava's own phrasings and suppressed the same tokens at once,
pushing mass into the degenerate tail. Fixed in `e64f933` (2026-07-24: the penalty is
scoped to generated tokens). `REBUILD.md` already records that the July failure "does
not prove resumed training intrinsically unsafe."

So the reason to forbid resuming is gone, and incremental training comes back — as a
**revert to the original scheme, adjusted for what we know now**: RAG decays while the
weights strengthen over a wall-clock window, the dose is counted in time rather than
in sleep cycles, and old material that stops being trained is allowed to fade from the
weights. **Forgetting is a feature.**

The full train stays, as the reset and emergency path. Both modes are selected by
config.

## 2. Scheme

### 2.1 The dose curve is cumulative, and counted in time

The existing ramp `D(age)` (`decay.lr_multiplier_hours`, `consolidation.wall_clock`):
0 below `rag_only_window_h` (24 h), then 1 → 2 → 4 across `[rag_only_window_h,
lora_cap_age_h]` (72 h), capped at 4. In incremental mode `D(age)` is read as the
**cumulative dose a row should have received by that age** — the time-based form of
the old per-cycle `[3,2,1]` (cumulative 3 → 5 → 6).

Each incremental build trains every row by the **increment**

```
increment(row) = D(age_now) − D(age_at_last_train)          # age in wall-clock hours
```

so a row's total dose is `D(cap)` however many builds fell inside its window — five
Sleep runs a day and one run in three days deliver the same total. (Reading `D(age)`
at build time as a per-build multiplier would couple the dose to the build count,
which is exactly what this plan removes.)

- **A row that missed its window** (no build for longer than the window, late
  background reflection, chats merged from another box) is trained **once**, with its
  calculated remaining dose — `D(age_now) − D(last)`, which for a never-trained row is
  the full `D(cap)`.
- **Minimum build size.** An incremental build runs only when the rows with a non-zero
  increment total at least `incremental.min_increment` tokens (default 16384, counted
  on the rendered rows). Below it the build is skipped: the adapter and the ledger are
  untouched, so every pending increment carries to the next build in full. This is the
  guard against fitting a handful of rows (the near-zero-loss verbatim regime of
  `96f98aa`). No per-row dose floor is needed: the accounting is exact, so a small
  increment is just a small LR multiplier.
- **After `lora_cap_age_h`** the increment is 0 forever; the row leaves training and
  starts to fade from the weights as later material trains over it.
- **Wander rows** stay one-shot at their fixed multiplier: delivered once, then done.
- **Cap-age contamination rows** (`unmask_user`) are their own row identity, delivered
  once when the host crosses cap age.

Dose units: a multiplier on the mode's own base LR. A dose of 1 under the full train
(2-epoch triangular, `train_lr`) and under an incremental build (`incremental.lr`) are
not the same amount of learning; each mode has its own base LR (§4).

### 2.2 RAG and weights on one clock — no change needed

The weight ramp caps at 72 h, verbatim RAG reaches 0 at `rag_cap_age_h` (96 h). RAG
always outlasts the point where the weights reach full dose, so material is never
in neither place, and everything settles by the first build after 96 h. Nothing in
the RAG fade changes.

### 2.3 The dose ledger

One record per row identity, per lineage:

- key: the row's `anchor_key` / (`source_session`, `exchange_index`, row kind:
  dialogue / contamination / wander);
- `delivered`: cumulative dose so far (`D` at the last train);
- `last_trained_at`, `build_id`.

It is stored **inside the adapter directory** (`dose_ledger.jsonl` beside
`adapter_model.safetensors`). `train_cycle` writes it at the end of **every** build,
full or incremental, from the rows that build actually trained: an incremental build
copies the parent's ledger forward and adds its increments; a full train writes a fresh
one. Consequences:

- **Rollback is consistent for free**: repointing `adapter_id` at an older adapter
  also points at the ledger that matches its weights.
- The ledger travels with the adapter through Fetch adapter / Migrate / snapshot
  export.
- The build snapshot (`models/snapshots/<build_id>/`) keeps the per-row record of what
  that build trained (it already stores `lr_multiplier`, `age` and the target per row);
  the ledger is the running total.

### 2.4 Transition: the existing adapter, as-is

The current adapter (full train, 2026-10-02) is taken as the parent of the first
incremental build, unchanged. It predates the ledger, so its ledger is built once from
its build snapshot (`models/snapshots/build-20261002-204943/sft_render.jsonl`, 1764
rows, present on the main box): every row it trained gets `delivered = D(age at its
built_at)`, `last_trained_at = built_at`. Rows already at cap are done; younger rows
receive only their remaining increment. This is the only time a ledger is read back
from a snapshot; every later build writes its own (§2.3).

The price, accepted: that adapter never followed the forgetting scheme — everything in
it was trained at full strength at once — so the old material in it fades only as new
material trains over it.

After a full train (§3) the build writes the same kind of ledger directly, so a full
train simply starts a new incremental lineage.

### 2.5 Resume mechanics

- Load the frozen base by `model_id`, then the parent adapter **from its fp32 file on
  disk** as trainable — never from a server's in-memory copy (the server casts LoRA to
  bf16 for serving, `inference_backend._normalize_adapter_dtype`).
- Keep the LoRA parameters fp32 through training (PEFT already does) and save fp32.
  The adapter is never quantized. Today's adapters are already fp32 on disk (820 F32
  tensors in `adapter-20261002_132006`); this plan keeps that and asserts it on resume
  and save. Serving in fp32 instead of the bf16 cast is a separate, optional knob, not
  part of this plan.
- **Lineage fingerprint**: an incremental step is valid only when `model_id`,
  `lora_r`, LoRA alpha, rsLoRA and the target modules match the parent's
  `adapter_config.json`. Any mismatch refuses the incremental step and asks for a full
  train.
- `builds.jsonl` gains `mode` (`full` / `incremental`) and `parent_adapter`.
- The optimizer starts fresh each build (no saved Adam state). The pre-switch recipe
  ran without warmup; `incremental.warmup_steps` exists in case the first steps of a
  build prove too hot.

### 2.6 Order and schedule

Rows with a non-zero increment train in chronological order, oldest first (as the
full train does). One build, one pass, under the incremental schedule (§4).

## 3. The full train: reset and emergency path

Unchanged from today: a fresh LoRA on the frozen base, every row at `D(age)` in one
fit, `train_lr_schedule` triangular with `train_plateau_epochs: 0` = 2 epochs (one
warmup, one decay).

Document plainly what it means once incremental is the norm:

- It **erases the forgetting** the incremental lineage built up: everything in the
  corpus comes back at full strength.
- It is the only way to apply a correction **retroactively** — a ban, a capture-gate
  `yields`, a bench holdout, a rewritten or replaced target on a row the lineage has
  already trained. Under incremental training such a change only affects future
  increments. No corruption needing this is known today, so it is an emergency path,
  avoided while incremental training works.
- It starts a new incremental lineage (its ledger written by the build, §2.3).

## 4. Config

Split the training knobs into two sets:

- `train_mode`: `full` | `incremental`.
- **Full train** — the existing keys, unchanged (no migration): `train_lr`,
  `train_lr_schedule`, `train_plateau_epochs`, `lora_r`, `train_max_seq_length`.
- **Incremental** — a new group in `server_config.json`, defaults taken from the last
  incremental recipe before the switch to full training (`701a91c^`, 2026-07-05):
  - `incremental.lr` = **1e-5** (set by `5d7b379` after the walk-down 2e-4 → 1e-4 →
    5e-5 → 1e-5; at r=16 with alpha=16). The rsLoRA alpha=4 of `5681dc8` is
    calibrated so r=32 is LR-neutral against that r=16 point, so the value carries
    over;
  - `incremental.lr_schedule` = `constant`, `incremental.warmup_steps` = 0,
    `incremental.epochs` = 1 (that recipe: one flat epoch, no warmup, no decay);
  - `incremental.min_increment` = **16384** tokens (§2.1).

  One difference from that recipe: it stepped every 4 rows (batch 1 × gradient
  accumulation 4); the per-row dose needs one optimizer step per row (batch 1 × GA 1,
  as the full train does now), so the same 1e-5 makes about 4× as many Adam steps per
  row set. If incremental builds read hot, this is the first thing to look at.
- Shared: `lora_r` (part of the lineage fingerprint) and `consolidation.wall_clock`
  (the `D(age)` curve, both modes).

All in `config_schema.py`, rendered by `settings.py`. The Sleep tab's train hand-off
follows `train_mode`; a one-off full train is a run-time override (`train_params`).

## 5. Housekeeping — document only

- Each build saves one fp32 adapter (~1 GB at r=32 on the 31B). Rollback needs the
  ancestors; continuing a lineage does not (the ledger is copied forward). Pruning
  keeps the recent ancestors.
- `models/trainer` (HF trainer output) is 62 GB on the main box and is regenerable.
- Docs that assume full trains only and need a pass when this lands: `REBUILD.md`,
  `server/training/DESIGN.md`, `documentation/AVA_STATUS.md`, `MODULE_NOTES.md`,
  `AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift`, and the "adapters are
  disposable" line in `CLAUDE.md`'s process contract.

## 6. Open items for the coding session

None at the plan level. Values are first settings, tuned from the loss and the step-0
readings once incremental builds run: `incremental.lr` (watch the GA note in §4) and
`incremental.min_increment`.

## 7. Imported foreign adapter (built 2026-10-04, experimental)

**As built — adoption with "train the last N chats".** The ledger at import is chosen
explicitly: `python -m training.adopt_adapter --adapter <fetched dir> --last-chats N
[--strength s] [--activate] [--dry-run]`, or the Migrate tab's **Adopt adapter…**
(previews the plan, then writes; runs the module in the server venv). It copies the
fetched adapter to `models/adapter-<ts>-adopt/` (the fetched one stays untouched — the
revert path), scales its `lora_B` by the import strength (1.0 = unchanged; fp32 kept), and
writes a ledger in which:

- the imported ledger's entries are kept (chats that adapter really trained — e.g.
  chats merged between the boxes — stay delivered);
- every other local row is marked delivered at its current `D(age)` without training:
  older local history is deliberately never put into these weights (wander included);
- the rows of the newest N reflected chats (by chat timestamp) are written owed
  (`delivered` 0), so the next incremental build trains each once at the dose it is owed
  — mostly the full `D(cap)` — and the ledger still covers every local row.

The imported `dose_ledger.jsonl` / `lineage.json` stay beside it as `*.foreign.*`; the new
`lineage.json` says `mode: adopted` with the foreign parent, N and the strength;
`builds.jsonl` gets an `outcome: adopted` line. `train_mode` must be `incremental` (a full
train refits from local data only — it un-imports).

**The guard.** Fetch adapter copies the source's ledger along with the weights; resumed
as-is, an incremental build would find none of this box's rows in it and train the whole
local history at full dose at once. An incremental build now refuses when the parent's
ledger knows less than `incremental.min_coverage` (0.5) of the local trainable rows
("adopt it or run a full train"). Cross-box flows on a shared corpus (chats merged
between the boxes) keep a high coverage and pass. On the main box the serving adapter's
ledger covers 98% of the local rows (2026-10-04).

**Still open** — the notes from the design pass:

- **Mechanics are the transition with someone else's adapter**: Fetch adapter already
  guards base compatibility; the lineage fingerprint (§2.5) is the rest. Rank need
  not match strictly — raising it is exact (extra rows of A, zero rows of B, then
  correct the scaling), lowering it is a truncated SVD of B·A (lossy).
- **Import strength.** Scaling B scales the foreign delta exactly, so "import at 50%"
  is one number. Measure each layer's ‖ΔW‖ against our own adapter first — the other
  instance may have trained at a different LR and dominate.
- **The ledger at import is the real decision.** *Seed* ("all local rows delivered",
  like §2.4): only new chats train and the instance takes on the foreign register —
  the 2026-10-03 transplant to a second Ava was this case at 100% by hand. *Catch-up*
  (empty ledger): §2.1 trains every local row once at full dose — in effect a
  full-corpus fit starting from the foreign adapter as a prior; that step is
  full-train-sized and may need the full-train parameters or spreading across builds.
- **False memories**: the foreign weights carry another instance's conversations,
  people and facts with no transcripts, RAG or facts tree behind them here.
  Forgetting works in our favour: nothing refreshes them.
- **Privacy**: the other instance's user's conversations come inside the weights;
  no framing removes them.
- **A full train un-imports**: it refits from local data only. Keep it that way (a
  full train stays a pure function of local data).
- **Measurements need a lineage key**: label inventory, hidden captures, capture probe
  and bench are keyed by adapter; record the foreign origin as the parent of step 0
  so readings can be split before and after the import.

## 8. As built

Code: `server/training/dose_ledger.py` (pure: row identity, `apply_increments`, `advance`,
`seed_from_render`, `lineage_mismatch`, `safetensors_dtypes`), `train_cycle.py`
(`_incremental_parent`, `_load_parent_lora`, mode resolution, `--mode`), `decay.py`
(`train_mode`, `incremental_settings`), `config_schema.py`, `reflection_service.py`
(`train_params.mode` → `--mode`), the Sleep tab's **Full train** checkbox. Self-tests:
`training.selftest` → `test_dose_ledger`, `test_load_parent_lora` (a tiny CPU model with
plain PEFT).

Details the plan left open, as built:

- **Resume.** The LoRA is attached by the same `get_peft_model` call a full train makes
  (unsloth's training patches in place), then the parent's `adapter_model.safetensors` is
  copied into it (`set_peft_model_state_dict`). A missing or unexpected LoRA key refuses
  the build, so a target-module mismatch is caught there; an all-zero `lora_B` after the
  load refuses too. Non-fp32 LoRA params are upcast; the save warns if the file is not
  all F32. The optimizer state (`adamw_8bit`) is per build and never saved — the 8-bit is
  the optimizer's moments, not the adapter.
- **`min_increment` is checked after the model load**: counting tokens needs the
  tokenizer and the retained-row preparation (the same counts the snapshot records). A
  skipped build therefore still costs the watchdog's stop / relaunch and one load; it
  writes nothing — no adapter, no ledger, no `builds.jsonl` line, no snapshot.
- **Quarantined rows stay owed**: a row over `train_max_seq_length` never trains, so it
  stays in "owed" and is quarantined again each build. It does not count toward
  `min_increment` (only retained rows do). On the main box these are 34 over-cap wander
  rows, quarantined by every full train too.
- **Contamination.** The split copy (`unmask_user`) is its own identity, owed `dose` once
  the host reaches cap. The masked row's target drops at cap from `D(cap)` to
  `D(cap) − dose`; a row that already received more on the ramp gets no further masked
  dose. A folded row is the dialogue identity, and its user-turn weight scales with the
  increment it trains by.
- **Seeding** finds the parent's build line by adapter directory name in `builds.jsonl`.
  A parent with no ledger and no such snapshot (an adapter fetched from another box that
  predates ledgers) refuses the build: run a full train. An adapter fetched with its
  ledger continues that ledger.
- **What the snapshot and the Training review tab show** under incremental: only the rows
  that build trained, `lr_multiplier` = the increment. `build_meta.json` adds `mode`,
  `parent_adapter` and the `incremental` settings + increment counts.
- **Tier 3** (character continuity) stays off: its baselines are captured before the
  parent's weights load, i.e. on the bare base. Validation is off anyway.
- **Live corpus, 2026-10-03** (read-only probe): the ledger seeds from
  `build-20261002-204943` with 1764 rows, all fully delivered; 4 dialogue rows from two
  chats of 10-02 are owed (they were inside the 24 h window at that build), about 10k
  tokens — below 16384, so the first incremental build would wait for more.
