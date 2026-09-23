# Ava Design

This document is the stable architecture baseline for Ava. It describes what the system is meant to keep true conceptually, not every experiment or dated implementation note.

Last code check: 2026-09-18, against `9ecb488`. The operator confirms that implemented paths have been exercised on the GPU box and are operationally sane. Remaining gaps concern specific behavior, state coverage, calibration, and causal attribution; they are not claims that the implementation has never run.

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

Facts, dispositions, and revised dialogue targets move through several distinct channels. The facts tree and associative library maintain explicit source-backed claims and relations; neither is an authoritative truth oracle. Dialogue and hosted facts also enter adapter weights. Retrieval is not uniformly temporary: verbatim chat expires, while gist floors, facts, source protocols, and associative access history carry persistent memory.

## Character Formation Dynamics

Ava has two influence paths.

The slow path logs chat, vets replies through reflection, records resolved targets in sidecars, and uses wall-clock age to set row learning-rate multipliers. A weighted persona maturity gate controls judge overrides. Adapters are disposable builds of that stored interpretation. Behavioral validation is deliberately disabled: a failed adapter leads to investigation of the process or its inputs, not a requirement to recover an adapter-quality gate.

The fast path is in-context conditioning during a live session. A standing persona portrait supplies accumulated dispositions when available, with persona RAG as the fallback. This provides historical context but no measured guarantee of resistance to a persuasive frame. Such a conversation can still enter the slow path. The open question is the strength and quality of influence, not an absent self-portrait; see `AVA_OPEN_PROBLEMS.md` under Impedance and First-User Imprint.

## Runtime Topology

Ava is split into a PyQt client and a GPU server.

The client owns UI and operator control. It connects over WebSocket, streams chat, starts reflection runs, monitors training hand-offs through the watchdog HTTP API, and displays debug artifacts.

The server owns all model and data state. `server/inference/server.py` keeps process-singleton runtime/session/encounter state, routes WebSocket protocol messages, builds inference prompts, logs chats, wires RAG, and delegates reflection/training-related work.

The watchdog is the process-management boundary. It starts the inference subprocess, exposes restart/update/log endpoints, and owns operations that require the model to be unloaded: LoRA training and destructive wipe. Read-mostly, data-layout-coupled management endpoints such as artifact export, clone export, snapshot export, chat sync, and precision writes live on the inference HTTP sidecar so they can evolve with ordinary `git pull` + inference restart, without changing the root watchdog.

The deliberation executive reads the autobiographical worklog, open threads, and persona portrait, then dispatches one chosen activity. Default `deliberation.mode=sole` removes independent timers for outreach, synthesis, check-in, wander, aha, and pivot; maintenance jobs retain their clocks. `shadow` runs both executive and drive timers, and `off` leaves the manual button as a dry run. Job registration changes at startup; the running executive also reads its settings per wake.

## State Lifetimes

Runtime state is split across two data roots: reflection/runtime state under `server/inference/data/`, and ordered user/ambient state under `server/data/`.

- `server/inference/data/hot/`: active reflection state, including reflection memory, ledger, persona digest, prompt deltas, prompt experiments, staging, and reflection run logs.
- `server/data/chats/`: active transcripts and sidecars. The old hot/archive chat split is no longer a training or RAG lifecycle; fully consolidated material fades by wall-clock age, not by being moved between directories.
- `server/data/til/`: durable ambient-learning state, including the keep-forever wander corpus and provenance snippets.
- `server/data/assoc/`: associative source/protocol records, model-generated relations, derived retrieval structures, needs, and persistent access/activation history. The whole directory is not a disposable index.
- `server/data/graph/`: derived facts tree plus hand-maintained `aliases.json`.
- `server/data/persona/<run_id>/`: materialized persona bundles selected by `current.json`, including the flat `digest.json` live chat reads.
- `server/inference/data/scratch/`: disposable render products such as `sft_render.jsonl`.

The reflection archive lives outside `inference/data/`, under `server/reflections/`. It is review material for rollback and diagnosis: per-run logs, committed artifact deltas, persona snapshots, and adapter copies for train-bearing runs. It deliberately has no manifest or ordered replay semantics.

Adapter lineages live under `server/models/`; `server/server_config.json` points at the active base model and adapter. It sits at the server root rather than inside `inference/` because it is the *box's* config: the offline train cycle repoints its `adapter_id` and the wipe job resets it while the inference role is down.

## State Durability Contract

The intended contract is that durable inputs support a fresh build and a snapshot preserves the running system's relevant state. Current coverage has three concrete limits:

1. **Live rebuild inputs and historical evidence have different lifetimes.** The two data roots hold transcripts, frozen targets, ledgers, portraits, wander material, and associative state. Together with code, prompts, base weights, config, seed, and build time they support a new build. After live repairs, they need not reconstruct an old build's exact inputs. Forensic snapshots and build records can therefore hold unique historical evidence even though adapters are replaceable outputs. Re-running a generated extraction is not the same as preserving its original result.
2. **Runnable exports cover the older memory layout, not all current state.** `snapshot_state.py` embeds inference data minus scratch/activity/persona-history, chats, TIL, prompts, current digest, config, active adapter, and its forensic training snapshot when available. It currently omits `server/data/assoc/` (including generated protocols and access history) and graph aliases. The external-dependency manifest names the base and RagEngine's MiniLM embedder, not the associative BGE-M3 embedder; model repository names are not pinned revisions. Code commit and dirty status are recorded, not an environment lock. Full causal closure and exact replay are not current guarantees.
3. **Forensic capture is best-effort.** `build_snapshot.write_snapshot` logs and returns `None` on failure. Training can already have repointed the adapter; it records the build and deletes the scratch render even if capture failed. Early training failures need not reach snapshot creation. Reliable evidence retention remains an implementation gap.

See `AVA_OPEN_PROBLEMS.md` → Snapshot State Coverage and Forensic Evidence Durability. These gaps limit restoration and investigation; they do not invalidate the working training or export paths.

## Memory Architecture

Ava has three practical memory layers.

Raw chats are episodic memory. They are logged as timestamped JSON files, with assistant CoT separated from the answer. Chat RAG indexes chats until the wall-clock crossfade drops their retrieval weight to zero; the old hot/archive directory split is not the retrieval boundary.

Reflection memory is distilled recall. `rag_memory.jsonl` is an append-only op-log of `insert`, `evict`, `surface`, and `lookup` records. It holds facts, asks, and mirrored weights-bound persona/fact items until they no longer need RAG recall.

Operator persona cleanup acts only on the connected server's live operational state. A removed persona is tombstoned in reflection recall and in the live consolidation ledger, so it is absent from the next digest and training fold. Cleanup must never rewrite a runnable snapshot, a reflection archive, adapter weights, or a previously materialized digest; those remain historical/rollback evidence, and the next reflection derives a new digest from the cleaned live set.

Weights are long-term memory. Resolved dialogue targets, hostable fact injections, and durable wander examples form SFT rows. Explicit build-time persona CoT prepending is retired; persona influences targets through original reasoning or a deliberately persona-conditioned re-answer. Every promoted adapter is a fresh build on the frozen base; prior adapters are rollback artifacts, not the starting point for the next fit.

## RAG Contract

RAG retrieves separate past-chat and distilled-reflection-memory blocks, with anchors and source nominations. Direct wander and open-ask injection into live chat/encounter/gossip are disabled by `_INJECT_WANDER=False` and `_INJECT_OPEN_ASKS=False`; their stored material and other consumers remain. The active chat is excluded both when an index is built and when it is queried, so an in-place resumed transcript cannot inject a duplicate of its own prompt. Reflection passes use a temporal cutoff so a session under review cannot retrieve itself or knowledge that became available later. Reflection-memory availability is read from the op-log insertion timestamp; `source_session` remains provenance and may be a typed external id such as `wiki:…` or `til:…`, not a sortable chat clock. Historical rows without an insertion timestamp fall back to timestamped chat filenames.

Verbatim chat and reflection-memory persona retrieval use separate wall-clock curves. A reflected verbatim exchange fades linearly `1.0→0` over 96 hours and is absent at/after the cap; the hard raw-age cutoff also applies if reflection lagged, while the pre-cap unfrozen path keeps its gentler hourly fresh-window discount. Opening additional chats does not alter an older chat's modifier—there is no count-based recency penalty. The consolidation gist rises `0→1.0` over the same 96 hours, then decays affinely to `0.2` exactly at 192 hours and holds that semantic floor. Persona recall retains its own `0.2` floor. Facts deliberately remain at modifier `1.0`; stale corrections are handled by explicit reversible supersession rather than age. Asks do not decay by this mechanism. Wall-clock modifiers are recomputed when querying the index, and a committed summary sidecar triggers an immediate serving-index refresh.

For RagEngine channels, the retrieval embedder is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, run explicitly on CPU so RAG consumes host RAM rather than model VRAM. The multilingual replacement stays in the MiniLM family because RagEngine's embedder is also reused by branch/persona/fact semantic gates whose cosine thresholds were calibrated in that score range. Long and mixed-language chat inputs, queries, and wander articles are split into bounded overlapping passages instead of being silently truncated at the encoder's beginning; passage hits collapse back to one exchange/article, and the wander query returns a bounded best passage plus reaction when a caller enables that channel (direct live-chat injection is currently disabled). Wander relevance is gated on raw semantic similarity and age affects ranking, so the final nonzero decay step remains retrievable. Post-search temporal filtering backfills from the complete eligible result set. Consolidation retrieves reflection memory independently for each fitted chunk rather than reusing one session-wide query/context. These are implementation choices, not architectural requirements.

Live fact fetch uses `assoc/` by default (`assoc.enabled=true`), with the facts tree as fallback when no associative build/library is available. Its default embedder is BGE-M3, separate from RagEngine. The library combines source protocols, lexical/dense retrieval, model selection, and persistent activation touches on injected material. Its feed/witness jobs maintain the store; the executive can choose aha and pivot activities.

## Reflection Loop

A reflection run is server-owned and synchronous inside a GPU executor thread.

The ingestion phase can run before classic reflection. It fetches a current-events digest when new, resolves open `[ask:search]` questions through lookup, reflects over the fetched material, and applies conclusions to live memory.

A normal Sleep run can also begin with a revisit head-phase. One eligible aged chat is re-derived under the current weights and committed to live memory before the main pass, so the main reflection can retrieve that revised understanding as context. Manual revisit uses the same re-derivation machinery as a standalone run.

The consolidation phase sees chat content without original CoT. It emits structured `WEIGHTS`, `RAG`, and `RESOLVED` sections. These become facts, asks, evictions, and ledger anchors.

The revision judgement sees the original exchange with CoT. It emits only `VERDICT`, `WHY`, `LANG_DRIFT`, and optional `PERSONA_TARGET`; it does **not** author the replacement dialogue. `WHY` diagnoses drift, while `PERSONA_TARGET` records only an affirmative disposition Ava endorses in the kept reply or in the direction a revised answer should take. This remains the main place persona self-statements are captured because it can compare the old thought and answer. The compatibility parser still accepts legacy `PERSONA` and inline `IDEAL` fields in old artifacts, but a new run discards any inline `IDEAL` at the judgement boundary.

The speaker may attach one post-reply **Meta feedback** note while that reply is still Ava's latest completed turn. It is stored on the raw exchange and shown only in this revision judgement, explicitly as notification-only evidence of the speaker's reaction: Ava may accept, reject, reinterpret, or ignore it, and the criterion remains whether the reply was genuinely hers. A stable exchange id fences the write so a delayed UI action cannot annotate an older turn. Once another reply lands the old window is immutable. Feedback never becomes a live chat turn, consolidation fact, RAG query, branch-replay input, re-answer prompt, or training prompt; it can affect weights only indirectly through the judgement/persona decision that causes a clean re-answer. The transcript keeps it for the normal reflect-once pass and any later deliberate revisit.

When judgement is `revise`, a separate normal-dialogue generation produces the target from the pre-answer prefix. The rejected answer/CoT, Meta feedback, diagnosis, and inline legacy IDEAL are excluded. General RAG is off; the explicit persona-only context seam may condition the re-answer, and the winning target records that context for render parity. An unusable target retries cleanly and then contributes no training row. Sidecars record target kind and generation provenance.

Branch replay generates alternate replies from contested token points. A blind chooser picks a trainable target among original, branches, and IDEAL.

Some exchanges deliberately skip branch replay. Language-drift repairs and revisit runs train the current-state clean re-answer directly, and CoT-less originals with a usable CoT-bearing IDEAL do the same. Branching remains the normal first-time reflection path for eligible new chats.

After sessions finish, the runner plans digest regeneration from committed evidence. The clean-base batch judges branch jobs against the existing digest, places facts, clusters persona/user/self evidence, and deduplicates facts. After adapter restoration, Ava synthesizes the new portraits. Without a clean-base callback, clustering falls back to the adapter, while branch judge and fact placement are skipped. Persona evidence combines recency decay, geometric tenure discount, and counter-evidence; the judge gate requires at least two themes with `weighted_recurrence >= 1.9`.

The active persona supplies a standing chat portrait of established voice/dispositions/lines; declarative stances are deliberately excluded from that render. A thin or absent portrait falls back to persona RAG. Polarity screening separates opposing members of merged themes, records counter plans, and exposes detected opposition to synthesis; arbitrary opposition between already-separate themes is not yet detected.

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
digest = existing persona digest
gate = count(digest.evidence.themes where weighted_recurrence >= 1.9) >= 2

with adapter unloaded and clean base loaded:
    for each job:
        judge_choice = score branch options against digest
        emit branch_judged event
        if apply_branch_judge and gate and judge_choice differs:
            resolve the trainable target from judge_choice
            rewrite the staged sidecar target
            increment judge_overrides
```

The digest is authored by Ava on the adapter; the clean base only applies it as an evaluator. If the corpus is thin, the judge remains logged-only. If it overrides any target, the train hand-off requests validation when the validation switch is enabled; while validation is globally disabled, the override is logged honestly but the adapter promotes without a behavioral gate by design.

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

Hostable fact anchors ride their source exchanges: up to two are injected as `I know that ...` lines into an existing trainable CoT. Persona anchors remain evidence for portraits and recall, but explicit persona CoT injection is retired. Training does not advance persona stages or fact trained-copy counts.

Fact-to-weights training uses clean-base placement within the fact’s own chat. The build reads the recorded host rather than re-judging it. A placed fact rides its host each build; its RAG mirror remains at weight 1.0. Supersession removes stale anchors from future folds, while unhosted facts remain retrieval-only. Placement currently collapses unparseable output and an explicit None into the same no-host outcome; see the failure-outcome gap in `AVA_OPEN_PROBLEMS.md`.

The train cycle runs offline while inference is unloaded. It loads the frozen base, fits a fresh LoRA, trains with response-only masking, keeps only the final assistant span in multi-turn examples, optionally emits a cap-age user-contamination row that unmasks the user's final turn, saves/repoints a new adapter, records build history, writes a forensic snapshot, and clears consumed scratch state. It does not resume the prior adapter, advance sidecar stages, move chats to archive, or clear the durable wander corpus.

The retained regression probe is force-skipped by `training.validation_switch.VALIDATION_ENABLED = False`, including judge-override runs. This is an intentional process choice, not an unfinished promotion gate. Technical row integrity checks and quarantine remain active. The global LR schedule is configurable: default `age_ramp` uses the requested epoch count; optional `triangular` uses one warmup epoch, `train_plateau_epochs` plateau epochs, and one decay epoch, in chronological order each pass.

## External Learning

The TIL/wander subsystem gives Ava controlled exposure to material outside the user chat.

Manual learning can fetch Wikipedia current-events digests, resolve search asks, or wander into an approved wiki page. The operator applies the resulting findings.

Autonomous wander runs from the idle heartbeat, after enough user-token budget has accumulated. It fetches a random approved article, runs a voice pass and a learning pass, applies the result, and records a durable wander corpus that is re-consolidated by each from-scratch build and retained for training and retrieval machinery. Direct wander-channel injection into live chat is currently disabled; external source material can still reach the facts/associative channels.

This is rationed by conversation-derived budget so ambient reading stays coupled to lived interaction rather than becoming an unbounded crawler.

## Reversibility

The system favors append-only logs, sidecars, archives, and adapter lineages. Reflection artifacts are staged before commit when requested. Bad adapters can be rolled back by repointing `adapter_id`. The old manifest replay path has been retired; a new rebuild mechanism is intentionally deferred.

The design principle is simple: Ava may change, but the path of change should be inspectable and reversible when a cycle goes wrong.
