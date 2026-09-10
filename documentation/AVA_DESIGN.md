# Ava Design

This document is the stable architecture baseline for Ava. It describes what the system is meant to keep true conceptually, not every experiment or dated implementation note.

Last code check: 2026-07-14. Main files consulted: `server/inference/server.py`, `server/inference/core/reflection_runner.py`, `server/inference/core/reflection_service.py`, `server/inference/core/reflection_writer.py`, `server/inference/core/reflection_memory.py`, `server/inference/core/rag_engine.py`, `server/inference/core/rag_policy.py`, `server/inference/core/reflection_digest.py`, `server/inference/core/prompt_mutation.py`, `server/inference/core/prompt_experiment.py`, `server/inference/core/synthesis.py`, `server/inference/core/mgmt_http.py`, `server/training/build_dataset.py`, `server/training/train_cycle.py`, `server/training/render.py`, `server/training/ledger.py`, `server/watchdog.py`, and `client/ui/sleep_widget.py`.

## Core Model

Ava is not treated as a single prompt or a single adapter. Ava is the whole closed loop:

- frozen base model and current LoRA adapter
- standing prompts and runtime prompt injections
- raw chat transcripts
- reflection memory and open questions
- consolidation ledger and chat sidecars
- sleep/reflection passes
- offline training cycle
- archive, wipe, and migration tools

The design goal is a single-user, non-reactive subject whose future responses are shaped by accumulated interaction, reflection, and weight updates, rather than only by the immediate prompt.

The project does not maintain an explicit truth graph. Facts, dispositions, and revised dialogue targets move through memory and training artifacts, then gradually into adapter weights. RAG is a temporary recall scaffold, not the desired final home of consolidated material.

## Character Formation Dynamics

Ava has two influence paths.

The slow path is the designed consolidation path: live chat is logged, reflection vets it, sidecars and ledger anchors record the trainable result, decay controls rehearsal count, the regression probe can block bad promotions when enabled, and the persona digest gates some target overrides only after cross-session maturity. This path has inertia, audit trails, and rollback points.

The fast path is ordinary in-context conditioning during a live session. It is useful because Ava must be responsive to the person in front of her, but it has no built impedance yet: a persuasive frame can shape the current conversation immediately, and that conversation can then enter the slow path. The current design therefore rate-limits consolidation but does not yet fully distinguish healthy relational imprint from coercive first-user capture. That unresolved line is tracked in `AVA_OPEN_PROBLEMS.md` under Impedance, Restoring Force, and First-User Imprint.

## Runtime Topology

Ava is split into a PyQt client and a GPU server.

The client owns UI and operator control. It connects over WebSocket, streams chat, starts reflection runs, monitors training hand-offs through the watchdog HTTP API, and displays debug artifacts.

The server owns all model and data state. `server/inference/server.py` keeps process-singleton runtime/session/encounter state, routes WebSocket protocol messages, builds inference prompts, logs chats, wires RAG, and delegates reflection/training-related work.

The watchdog is the process-management boundary. It starts the inference subprocess, exposes restart/update/log endpoints, and owns operations that require the model to be unloaded: LoRA training and destructive wipe. Read-mostly, data-layout-coupled management endpoints such as artifact export, clone export, snapshot export, chat sync, and precision writes live on the inference HTTP sidecar so they can evolve with ordinary `git pull` + inference restart, without changing the root watchdog.

## State Lifetimes

Runtime state is split across two data roots: reflection/runtime state under `server/inference/data/`, and ordered user/ambient state under `server/data/`.

- `server/inference/data/hot/`: active reflection state, including reflection memory, ledger, persona digest, prompt deltas, prompt experiments, staging, and reflection run logs.
- `server/data/chats/`: active transcripts and sidecars. The old hot/archive chat split is no longer a training or RAG lifecycle; fully consolidated material fades by wall-clock age, not by being moved between directories.
- `server/data/til/`: durable ambient-learning state, including the keep-forever wander corpus and provenance snippets.
- `server/inference/data/scratch/`: disposable render products such as `sft_render.jsonl`.

The reflection archive lives outside `inference/data/`, under `server/reflections/`. It is review material for rollback and diagnosis: per-run logs, committed artifact deltas, persona snapshots, and adapter copies for train-bearing runs. It deliberately has no manifest or ordered replay semantics.

Adapter lineages live under `server/models/`; `server/server_config.json` points at the active base model and adapter. It sits at the server root rather than inside `inference/` because it is the *box's* config: the offline train cycle repoints its `adapter_id` and the wipe job resets it while the inference role is down.

## State Durability Contract

Two invariants govern which state is essential versus derived. They are load-bearing: a feature or tool is correct only if it preserves them.

1. **The data roots are the ground truth — technically sufficient to rebuild any Ava state.** The relevant roots are `server/data/` plus `server/inference/data/`. Everything downstream is a pure function of those roots plus a chosen base model and training/config settings: the RAG indexes (rebuilt in-memory on boot), the training corpus (`build_dataset` reads chats, sidecars, ledger, reflection memory, and wander corpus), the LoRA adapter (`train_cycle`), and the persona digest. The data roots hold the un-regenerable material — raw transcripts and frozen reflection decisions (sidecars), the anchor ledger, reflection memory, wander corpus/snippets, run logs, prompt experiments/deltas, and persona digests. Because the base model is a config field, the same data can be retrained onto a different base (Qwen, GPT-OSS, …) or with different settings — that portability is the test of this invariant. `server/models/` (adapters, build log, forensic snapshots) and `server/reflections/` (review snapshots) are therefore **derived**: convenient for rollback and audit, but reconstructible from data and never a unique source of truth.

2. **A runnable snapshot (`server/exports/…`, written by `snapshot_state.py`) contains everything needed to run live chat plus the complete training dataset — except the external Hugging Face models.** That is: the LoRA adapter weights, all RAG sources (the full `data/`), the prompts and config, and the exact rendered rows fed to the trainer (`training/<build_id>/sft_render.jsonl`). The two deliberate exclusions are **the base model and the RAG embedder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)** — both external, immutable, and fetched from Hugging Face by id; the snapshot's `MANIFEST.json` names both under `external_dependencies`. Everything else that shapes a reply is in the folder, so a snapshot answers both "run her here" and "where did that reply come from?" from its own contents.

Everything else on disk is disposable: code lives in git, and the remainder is config or regenerable/temporary files (`scratch/`, trainer checkpoints, progress logs, the exports themselves).

## Memory Architecture

Ava has three practical memory layers.

Raw chats are episodic memory. They are logged as timestamped JSON files, with assistant CoT separated from the answer. Chat RAG indexes chats until the wall-clock crossfade drops their retrieval weight to zero; the old hot/archive directory split is not the retrieval boundary.

Reflection memory is distilled recall. `rag_memory.jsonl` is an append-only op-log of `insert`, `evict`, `surface`, and `lookup` records. It holds facts, asks, and mirrored weights-bound persona/fact items until they no longer need RAG recall.

Operator persona cleanup acts only on the connected server's live operational state. A removed persona is tombstoned in reflection recall and in the live consolidation ledger, so it is absent from the next digest and training fold. Cleanup must never rewrite a runnable snapshot, a reflection archive, adapter weights, or a previously materialized digest; those remain historical/rollback evidence, and the next reflection derives a new digest from the cleaned live set.

Weights are long-term memory. Dialogue anchors, persona anchors, hostable fact anchors, and wander examples can be rendered into SFT rows and trained into the active LoRA adapter. Each promoted adapter is a fresh build artifact compiled from the frozen base plus the current frozen-bundle corpus; the base model is frozen, and prior adapters are rollback artifacts rather than the starting point for the next fit.

## RAG Contract

RAG retrieves separate past-chat and distilled-reflection-memory blocks, plus an optional wander passage in live chat/encounter. The active chat is excluded both when an index is built and when it is queried, so an in-place resumed transcript cannot inject a duplicate of its own prompt. Reflection passes use a temporal cutoff so a session under review cannot retrieve itself or knowledge that became available later. Reflection-memory availability is read from the op-log insertion timestamp; `source_session` remains provenance and may be a typed external id such as `wiki:…` or `til:…`, not a sortable chat clock. Historical rows without an insertion timestamp fall back to timestamped chat filenames.

Verbatim chat and reflection-memory persona retrieval use separate wall-clock curves. A reflected verbatim exchange fades linearly `1.0→0` over 96 hours and is absent at/after the cap; the hard raw-age cutoff also applies if reflection lagged, while the pre-cap unfrozen path keeps its gentler hourly fresh-window discount. Opening additional chats does not alter an older chat's modifier—there is no count-based recency penalty. The consolidation gist rises `0→1.0` over the same 96 hours, then decays affinely to `0.2` exactly at 192 hours and holds that semantic floor. Persona recall retains its own `0.2` floor. Facts deliberately remain at modifier `1.0`; stale corrections are handled by explicit reversible supersession rather than age. Asks do not decay by this mechanism. Wall-clock modifiers are recomputed when querying the index, and a committed summary sidecar triggers an immediate serving-index refresh.

The retrieval embedder is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, run explicitly on CPU so RAG consumes host RAM rather than model VRAM. The multilingual replacement stays in the MiniLM family because RagEngine's embedder is also reused by branch/persona/fact semantic gates whose cosine thresholds were calibrated in that score range. Long and mixed-language chat inputs, queries, and wander articles are split into bounded overlapping passages instead of being silently truncated at the encoder's beginning; passage hits collapse back to one exchange/article, and only a bounded best wander passage + reaction is injected. Wander relevance is gated on raw semantic similarity and age affects ranking, so the final nonzero decay step remains retrievable. Post-search temporal filtering backfills from the complete eligible result set. Consolidation retrieves reflection memory independently for each fitted chunk rather than reusing one session-wide query/context. These are implementation choices, not architectural requirements.

## Reflection Loop

A reflection run is server-owned and synchronous inside a GPU executor thread.

The ingestion phase can run before classic reflection. It fetches a current-events digest when new, resolves open `[ask:search]` questions through lookup, reflects over the fetched material, and applies conclusions to live memory.

A normal Sleep run can also begin with a revisit head-phase. One eligible aged chat is re-derived under the current weights and committed to live memory before the main pass, so the main reflection can retrieve that revised understanding as context. Manual revisit uses the same re-derivation machinery as a standalone run.

The consolidation phase sees chat content without original CoT. It emits structured `WEIGHTS`, `RAG`, and `RESOLVED` sections. These become facts, asks, evictions, and ledger anchors.

The revision judgement sees the original exchange with CoT. It emits only `VERDICT`, `WHY`, `LANG_DRIFT`, and optional `PERSONA_TARGET`; it does **not** author the replacement dialogue. `WHY` diagnoses drift, while `PERSONA_TARGET` records only an affirmative disposition Ava endorses in the kept reply or in the direction a revised answer should take. This remains the main place persona self-statements are captured because it can compare the old thought and answer. The compatibility parser still accepts legacy `PERSONA` and inline `IDEAL` fields in old artifacts, but a new run discards any inline `IDEAL` at the judgement boundary.

The speaker may attach one post-reply **Meta feedback** note while that reply is still Ava's latest completed turn. It is stored on the raw exchange and shown only in this revision judgement, explicitly as notification-only evidence of the speaker's reaction: Ava may accept, reject, reinterpret, or ignore it, and the criterion remains whether the reply was genuinely hers. A stable exchange id fences the write so a delayed UI action cannot annotate an older turn. Once another reply lands the old window is immutable. Feedback never becomes a live chat turn, consolidation fact, RAG query, branch-replay input, re-answer prompt, or training prompt; it can affect weights only indirectly through the judgement/persona decision that causes a clean re-answer. The transcript keeps it for the normal reflect-once pass and any later deliberate revisit.

When the judgement is `revise`, a separate normal-dialogue generation produces the `IDEAL` target from the stored system prompt, the answer-only conversation before the exchange, and the final user turn. The rejected answer and CoT, Meta feedback, reflection prompt/RAG, `WHY`, and judgement CoT are outside this message list. This is the same prefix `training.render.build_messages` later reconstructs, so the generated `<think>` belongs to the target answer rather than to a retrospective critique. A malformed or truncated re-answer retries once from the same clean prefix with perturbed sampling; confirmed language drift may add only a neutral final-user-language constraint. If no CoT-bearing answer survives, the exchange is `revised_missing_ideal` and contributes no training target—there is no fallback to the rejected original. Sidecars and anchors record `target_kind` plus `target_generation` (`chat_reanswer_v1`, `branch_replay`, or `original`) in addition to the compatibility `target_source` field.

Branch replay generates alternate replies from contested token points. A blind chooser picks a trainable target among original, branches, and IDEAL.

Some exchanges deliberately skip branch replay. Language-drift repairs and revisit runs train the current-state clean re-answer directly, and CoT-less originals with a usable CoT-bearing IDEAL do the same. Branching remains the normal first-time reflection path for eligible new chats.

After sessions finish, the runner synthesizes a persona digest from committed persona anchors. The digest is a versioned self-portrait with voice, stances, dispositions, lines, and a second-pass first-person prose portrait for human/debug inspection and future peer introductions. Its evidence fold combines 30-to-180-day recency decay, geometric tenure discount for successive same-theme affirmations, and recency-weighted counter-evidence from sustained user pushback; the clean-base judge maturity gate deliberately continues to use raw distinct-session recurrence.

A clean-base judge can score branch choices against the persona digest. When enabled and the digest maturity gate passes, its pick can override the trainable sidecar target. On thin corpora it is logged-only.

Fact placement also runs on the clean base. It assigns unhosted facts to a source exchange whose reasoning rests on that fact, so training can inject the fact into that exchange's CoT. The placement is snapshotted onto the fact anchor and read later by training; the fact is not re-judged at render time.

Fact correction is a separate lifecycle pass. Live facts are clustered by subject, an LLM
identifies direct contradictions, and code mechanically keeps the newest claim while
soft-superseding older conflicts in both RAG memory and the ledger. A manual clean-base pass
covers all live facts; the automatic post-commit pass is best-effort and limited to subjects
touched by facts from the current reflection run.

Prompt mutation is logged-only. It proposes standing-prompt deltas after drifted revisions, writes them to an op-log, and never edits the prompt.

Persona preview is a non-mutating prompt self-review: Ava can inspect the current standing prompt, persona digest, and prompt deltas, then reason about whether she would keep or replace the prompt. It writes nothing.

Prompt experiment is a reversible live prompt swap. Ava can generate a temporary replacement standing prompt; the system stores it in mutable runtime state, prefers it over `chat_prompt.txt`, and can revert it without overwriting the canonical prompt file. The operator can write one directly into the same tier instead, which changes who authors the text but nothing about its standing: identical storage, identical revert, no path to the canonical file. This is intentionally weaker than governed self-modification: it is an operator-controlled experiment, not a durable promotion path.

The prompt is deliberately not permanently self-modifying yet. A promoted prompt delta would be high leverage: it changes the live fast path before decay, probes, or adapter rollback can absorb the effect. The current implementation therefore has logged evidence, non-mutating preview, and temporary experiments, but no clustered, maturity-gated, clean-base-validated, versioned prompt promotion pipeline.

## Mechanism Sketches

These sketches name the implementation points without replacing the code as the source of truth.

**From-Scratch Build And Wall-Clock Crossfade**

Implemented by `training.build_dataset`, `training.train_cycle`, `training.decay`, and `rag_engine`.

```text
for each reflected chat sidecar:
    if not reflected_at:
        skip                     # not a frozen bundle yet
    age_h = wall_clock_age_hours(chat_timestamp, build_timestamp)
    lr_mult = lr_multiplier_hours(age_h)
    if lr_mult <= 0:
        skip                     # RAG-only window
    emit one trainable row per revisable exchange
    if cap-age contamination applies:
        split into masked response row + unmasked user-voice row

train_cycle:
    load frozen base
    fit a fresh LoRA from scratch
    consume rows oldest-first with per-step LR = train_lr * row.lr_multiplier * schedule_shape(step)
    promote by saving adapter, repointing config, recording build history and snapshot

rag_engine:
    for each frozen source bundle:
        verbatim_weight = verbatim_rag_weight_hours(age)  # 1 -> 0 by 96h
        gist_weight = gist_rag_weight_hours(age)          # 0 -> 1 -> 0.2 by 192h
        persona_weight = rag_weight_hours(age)             # holds 0.2 floor
        drop verbatim RAG at the 96h cap
        keep fact RAG items at 1.0; correction uses explicit supersession
```

The adapter is a build artifact, not a cumulative continuation. Repetition and decay-count variants are gone; consolidation strength is carried by the wall-clock LR ramp and the later RAG crossfade.

**Clean-Base Judge And Digest Gate**

Implemented by `reflection_runner._run_clean_base_judge`, `_digest_maturity_gate`, and `_maybe_override_target`.

```text
jobs = branch choices collected during revision
digest = latest persona digest
gate = count(digest.evidence.themes where recurrence >= 3) >= 2

with adapter unloaded and clean base loaded:
    for each job:
        judge_choice = score branch options against digest
        emit branch_judged event
        if apply_branch_judge and gate and judge_choice differs:
            resolve the trainable target from judge_choice
            rewrite the staged sidecar target
            increment judge_overrides
```

The digest is authored by Ava on the current adapter; the clean base only applies it as an evaluator. If the corpus is thin, the judge remains logged-only. If it overrides any target, the train hand-off requests validation when the validation switch is enabled; while validation is globally disabled, the override is logged honestly but the adapter still promotes unguarded.

**Fact Placement And Snapshot**

Implemented by `reflection_runner._run_clean_base_fact_placement`, `_place_fact`, `training.build_dataset.index_by_exchange`, and `training.build_dataset._inject`.

```text
for each live unhosted [fact]:
    candidates = source exchanges whose CoT might rest on that fact
    host = clean_base_placement_judge(fact, candidates) or None
    if host:
        ledger.register_fact(..., source_exchange=host)  # snapshot

for each trainable dialogue anchor:
    facts = ledger facts whose source_exchange == this anchor
    inject up to FACT_INJECT_CAP as "I know that ..." CoT lines
```

The host is chosen once and stored on the anchor. Training reads that snapshot rather than re-judging at render time, so later from-scratch builds keep the same fact-to-exchange binding.

## Consolidation And Training

Each reflected exchange has a sidecar record beside its chat transcript. The sidecar stores verdict, a single resolved trainable target, and the document-level `reflected_at` freeze that makes the chat a frozen bundle. New sidecars do not carry the retired regularizer field, though old ignored fields may remain on disk. The raw transcript remains single-writer.

The per-session ShareML document is retired. Reflection no longer writes `<stem>.shareml.json`, and staging and archive no longer copy it. The surviving `reflection_shareml.py` helper is just the canonical verbatim assistant-turn assembler reused by `training.dialogue_source`.

Training renders the full frozen-bundle corpus. Each reflected chat contributes one row per revisable exchange, chronological oldest-first, unless it is still inside the RAG-only age window. Reworded variants, IDEAL regularizers, and stage-count repetition are removed.

Render/inference parity is load-bearing. Training rows are assembled to match the live inference conversation shape, including speaker prefixes and model-family reasoning-channel handling.

Persona and fact anchors ride host exchanges. They are injected into the beginning of a trainable CoT for an exchange already being trained, with caps per exchange. The ledger is read as the host/provenance store; training no longer advances persona stages or fact trained-copy counts as the lifecycle clock.

Fact-to-weights training is implemented as of commit `23164e0ca73306c70d97ba5401ae3fede1948958` and was reshaped by the rebuild. The clean-base placement judge assigns each unhosted `[fact]` one host exchange or `None`, restricted to the fact's own chat for data locality. The build then injects hosted facts after persona lines as `I know that ...`, with `FACT_INJECT_CAP=2` facts per exchange. A placed fact rides its host every build; its mirrored RAG copy stays at `1.0` rather than fading. When a newer fact directly contradicts it, reversible supersession removes the stale anchor from live recall and future training folds.

The train cycle runs offline while inference is unloaded. It loads the frozen base, fits a fresh LoRA, trains with response-only masking, keeps only the final assistant span in multi-turn examples, optionally emits a cap-age user-contamination row that unmasks the user's final turn, saves/repoints a new adapter, records build history, writes a forensic snapshot, and clears consumed scratch state. It does not resume the prior adapter, advance sidecar stages, move chats to archive, or clear the durable wander corpus.

The regression probe is the promotion gate when validation is enabled. It is designed to catch abrupt capability, format, continuity, and acute-retention failures, not to decide whether Ava's beliefs are acceptable.

## External Learning

The TIL/wander subsystem gives Ava controlled exposure to material outside the user chat.

Manual learning can fetch Wikipedia current-events digests, resolve search asks, or wander into an approved wiki page. The operator applies the resulting findings.

Autonomous wander runs from the idle heartbeat, after enough user-token budget has accumulated. It fetches a random approved article, runs a voice pass and a learning pass, applies the result, and records a durable wander corpus that is re-consolidated by each from-scratch build and also available to live chat as an age-faded RAG channel.

This is rationed by conversation-derived budget so ambient reading stays coupled to lived interaction rather than becoming an unbounded crawler.

## Reversibility

The system favors append-only logs, sidecars, archives, and adapter lineages. Reflection artifacts are staged before commit when requested. Bad adapters can be rolled back by repointing `adapter_id`. The old manifest replay path has been retired; a new rebuild mechanism is intentionally deferred.

The design principle is simple: Ava may change, but the path of change should be inspectable and reversible when a cycle goes wrong.
