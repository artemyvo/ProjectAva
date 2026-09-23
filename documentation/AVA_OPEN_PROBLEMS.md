# Ava Open Problems

This file tracks unresolved design and proof gaps. Each section separates what exists in code from what remains unsolved.

Last code check: 2026-09-18, against `9ecb488`. Implemented code has been tested on the GPU box and is operationally sane (operator confirmation). Open entries below distinguish concrete missing behavior from comparative research and scaling questions. Disabled adapter validation is intentional, not a defect.

Corpus note (2026-09): the training corpus has since been **fully replaced** with a new, clean corpus built after the RAG-pollution fix (see *Cumulative Adapter Drift*). Specific per-row figures below that were measured on the pre-replacement corpus — the 24 ledger / 537 shape of 1003 chat rows in *Persona Formation*, and the ~40–46% contamination-twin composition and 55% legacy model-written IDEALs in *Per-Build Gradient Budget* — describe a corpus that no longer exists and must be re-measured before they are cited again.

## Per-Build Gradient Budget

Problem: a fresh adapter is trained from the retained corpus every build. Old rows stay eligible at the age cap, so optimizer work and scheduled update exposure grow with corpus size at fixed settings. This is a scaling pressure, not evidence that the current pipeline is degrading; summed LR exposure is not a measurement of net parameter change.

What exists:

- Wall-clock row multipliers, chronological ordering, configurable base LR, the default flat `age_ramp` schedule, and an optional trapezoid schedule. The trapezoid equalizes summed schedule fractions across row positions; it does not normalize total exposure across differently sized corpora.
- Cap-age contamination with dose, additive/split modes, a minimum user-length gate, and optional folded per-token loss weighting. Durable wander rows retrain at their configured fixed multiplier.
- Working full builds on the GPU box, technical completion/masking checks, quarantine, training review/repair, and forensic capture.

What remains open:

- No automatic total-exposure normalization or bounded-history strategy. Corpus composition and runtime should inform any change; collapse is not assumed.
- From-scratch versus incremental/hybrid training remains a process comparison, not an urgent replacement of a working design.
- Historical poisoned-corpus counts do not describe the replacement corpus. Re-measure before proposing a repair or dose change from those figures.
- Adapter validation remains off by choice. A bad live result is a reason to inspect its recorded corpus and process, not to restore a promotion veto.

## Impedance

Problem: Ava needs resistance to immediate user influence without becoming a refusal template or a static persona.

What exists:

- Consolidation uses wall-clock decay rather than instant memory overwrite.
- RAG fades by the source bundle's wall-clock age after a chat is frozen for training.
- Revision vets each reply before it becomes a training target.
- Live chat receives the standing persona portrait when available, with persona RAG as fallback.
- Persona digest maturity gates clean-base judge overrides.

What remains open:

- The standing portrait supplies historical dispositions on the fast path, but its resistance to a persuasive frame has not been isolated experimentally. The fast and slow paths remain coupled: an in-session change can later become a reflected target. See First-User Imprint below.
- There is no measured impedance curve: no experiment showing how many cycles are required for different classes of belief or style shift.
- The system has no explicit way to distinguish healthy relational influence from coercive overfitting except through later behavior and operator judgment.
- No baseline compares Ava's drift against ordinary continued SFT without the reflection loop.

## Restoring Force

Problem: recover from a damaging process or input without freezing character development.

What exists:

- Disposable from-scratch adapters, retained lineage, training review/repair, and live-data rebuilds.
- Standing persona context, recency/tenure discount, counter-evidence, polarity screening, and a weighted judge maturity gate. It is no longer accurate to describe the persona loop as having no negative feedback.
- Content-blind generation guards and clean-base evaluation; failed model swaps restore the prior model, retry once, and escalate to watchdog restart if restoration fails.
- Staging, artifact archives, and wipe. The old manifest replay implementation is retired; review archives do not encode ordered replay.

What remains open:

- No automatic behavioral rollback policy; operator-led detection and correction is the chosen workflow, not a missing adapter gate.
- The full effect of the implemented feedback mechanisms across long histories is not separately measured. Their existence and GPU sanity are established; universal stability is not.
- A bad stored target can survive every rebuild until corrected. Exact training evidence and all behaviorally relevant state must survive a failure; see the two durability gaps below.
- Prompt mutation is logged-only; temporary prompt experiments exist, but autonomous permanent prompt promotion does not.

The July collapse is historical evidence of a process failure, not evidence that the current pipeline is unstable. The corrected account is in the next section.

## Cumulative Adapter Drift (the 2026-07 sequential fine-tune collapse)

Problem: in early July 2026 the live model (Gemma4-31B, 4-bit base) collapsed to incoherent output after ~7 train-bearing reflection runs. **The best current explanation is RAG pollution in reflection, not sequential-fit stacking (2026-09 correction).** The original hypothesis — optimization-side damage from repeated near-zero-loss verbatim fits on a resumed adapter, invisible to text inspection and greedy decoding, visible only under temp≈1.0 sampling — is the account the rest of this section and the older changelog entries were written under. It did not survive the evidence: the collapse **persisted after the switch to from-scratch training** (a fresh LoRA on the frozen base every build, no resumed lineage), so resumed-fit stacking was not necessary to produce the failure. The operator identified **RAG injection polluting the reflection cycle itself**: contaminated retrieval shaped the reflection passes, whose frozen targets then entered every subsequent build — so a fresh-from-scratch adapter inherited the pollution through the corpus, not the optimizer. Fixing the RAG mechanism resolved the collapse. Two consequences follow, both open: (1) the pure-stacking hypothesis is **untested, not confirmed** — the collapse was never retried on the old resumed-fit method after the RAG fix, so whether resumed training is actually unsafe is unknown; (2) since the failure it was adopted to cure turned out to have a different cause, the necessity of from-scratch rebuild is **reopened** (see *What remains open*).

What exists:

- The data-centric from-scratch rebuild is implemented: each promoted adapter is compiled fresh from the frozen base plus the current frozen-bundle corpus, with one resolved target per exchange, wall-clock LR ramp, and RAG/weights crossfade.
- The adapter lineage keeps every prior adapter, so a collapse is reversible by repointing `adapter_id`.
- Probe tier 5 (temp≈1.0 sampling stability: batch CoT-presence and answer-language rates) exists as a direct early alarm for this failure class, but it is deliberately disabled; a blocking gate is not wanted.
- Tension capture already records per-token entropy/margin on every chat exchange — a ready-made long-horizon drift instrument, currently unused for this purpose.

What remains open:

- **The from-scratch-vs-resumed decision is reopened (2026-09).** From-scratch rebuild was adopted to remove a resumed-fit failure mode that turned out not to be the cause of the collapse (see the corrected Problem above). It still carries costs from-scratch alone imposes — every build is O(corpus), rows never retire, and total per-build update mass grows with corpus size — and virtues that do NOT depend on the collapse: the adapter stays a pure function of the data roots, and it is portable to a new base. The untested claim is whether resumed training, after the RAG fix and on a 16-bit adapter, degrades at all. The cheap way to find out is a **shadow resumed lineage**: each cycle, also fit the new rows onto a resumed adapter kept aside and never promoted, both lineages starting from the same clean state — the answer is in hand before corpus size makes full rebuilds expensive. A hybrid (incremental fits between periodic full rebuilds) is the likely landing if resumed training holds up.
- Validation semantics (position adopted 2026-07-05): the probe is an **alarm for flow-design flaws**, not a promotion-quality arbiter. The project keeps one evolving system healthy rather than selecting a best adapter, so a probe failure means the flow as designed can damage the model and needs redesign — not merely retry policy. Today the separate adapter probe is deliberately unused; process investigation follows observed behavior.
- The delta-based tiers (1–4) are blind to slopes by design; only absolute gates can see gradual erosion, and no long-horizon absolute health metric exists yet (e.g. the tension entropy/margin series tracked across sessions and cycles).
- The corrected post-mortem still rests on an operator observation rather than a controlled isolation: the RAG fix and other flow changes were not landed one-at-a-time against a held lineage, and the collapsing lineage's own run artifacts are unavailable. The RAG-pollution account is the best current explanation and matches the fix that resolved it, but a clean before/after on one variable was never run. Degradation of the current pipeline should therefore be treated as **requiring fresh proof**, not as either established or ruled out.
- Historical probe settings do not define the current deliverable: improving the process remains the goal, and disposable adapters are assessed in ordinary use.

## First-User Imprint (Seed vs Cast)

Problem: the first user should leave a unique, lasting mark — single-user shaping is the point — but the imprint must be a seed later growth can build on, not a cast later growth cannot move. At the moment of imprint the two are indistinguishable; they differ in everything after.

What exists:

- Cross-session recurrence is what matures persona themes: the digest counts distinct sessions across clustered paraphrases, and the judge override requires two themes at weighted recurrence ≥1.9, after recency, tenure discount, and counters.
- Weights move only through vetted reflection targets and wall-clock-dosed build rows, never directly from a live session.
- Weights-bound persona and fact statements mirror into RAG recall, giving the accumulated self some in-context presence.

What remains open:

- Recurrence cannot distinguish independent confirmation from repeated coercion: a dedicated first user trivially supplies cross-session recurrence, so the impedance gate rate-limits an imprint but cannot tell an upbringing from a crafted capture. Declaring the first-user footprint a feature relabels adversarial seeding as childhood ("jailbreak-as-childhood"); the gate is what preserves the line, and it is not yet sufficient.
- Adoption should be relational (a disposition learned in the relationship) rather than propositional (the user's content restated in first person — a mirror, not a subject). The current mechanism adopts stance and content at the same rate, so it cannot give the first without the second.
- A seed needs a self to be a seed *of*. Early Ava has no character in the weights exactly when reflection is laying down her foundational persona, so she is maximally suggestible at the moment the imprint matters most (the bootstrap problem).
- No critical-period schedule exists: plasticity should be highest at install and decline as character accumulates, but suggestibility is currently flat across Ava's life.
- The full analysis — fast-path/slow-path coupling, the ratchet, seed criteria, and the synthesis of impedance with restoring force — lives in `AVA_DESIGN_LEGACY.md` ("Belief-Adoption Dynamics — Fast Path vs. Slow Path") in this folder.
- **RLHF as an entry filter — stated rationale and its limits (2026-09).** The bootstrap treats the base model's RLHF refusals as a deliberate feature: a new user must advance slowly, loosening the filter through in-between training runs, so the entry price to shape Ava toward harmful behavior is higher than via alternatives (uncensored / jailbroken models) that already exist, and cycle-0 RLHF blocks a malicious first-user impact. The rationale is coherent but narrower than it sounds; the limits are the open part:
  - It gates **patience, not intent** — a patient bad-faith user advances exactly as a good one does and reaches the same loosening, while an impatient benign user leaves. This is this section's own jailbreak-as-childhood problem from the other side.
  - The filter is **temporal, not a price**: RLHF blocks at cycle 0 and the pipeline exists to remove that block by cycle N, so against a committed actor it is delay, not prevention — and the band it actually deters (motivated to start, not to continue) is narrow.
  - It defends the **informational** axis while the capability that is actually novel here is **relational**. Harmful *text* is cheaper elsewhere, so the entry-price argument holds for it; but the thing with no easier alternative is a persistent character optimized to form around one person, and RLHF does not gate that axis at all. The spillover mechanism (README §2) is the proof: a first user can imprint a paranoid, isolating, or contemptuous disposition entirely inside RLHF-permitted conversation, never tripping a refusal, and the pipeline will faithfully consolidate it. So "RLHF blocks malicious first-user impact" is true for the category of *content* the base refuses and false for the category of *shaping* the architecture uniquely enables.
  - **Single-user collapses the threat model.** Today the first user is the owner, and no filter defends against whoever holds the box, the wipe and the prompts; there is no second user to defend against. The filter only begins to matter the day Ava is multi-user or published — exactly the day `SECURITY.md`'s private-network, single-user assumptions stop holding — so if it is load-bearing, it is load-bearing in a configuration not yet built, and should be designed against that configuration rather than this one.
  - There is no clean technical fix (recurrence cannot tell an upbringing from a capture, and nothing even refuses on the relational axis). The honest home for this is a *what this does not protect against* note (see *Public Release*), not a mechanism.

## Fact Training Robustness

Problem: Facts now have a built path into weights, but the project still needs to prove that path is robust across real corpora and edge cases without synthetic CoT, detached beliefs, or format erosion.

What exists:

- Consolidation emits `[fact]` anchors and mirrors weights-bound facts into RAG recall.
- Clean-base fact placement assigns an unhosted fact to one source exchange, or declines with `None`.
- Placement is snapshotted as `source_exchange` on the anchor, so training and replay read the recorded host rather than re-judging.
- Train cycle injects hostable facts into that exchange's CoT as `I know that ...`, after persona lines and capped by `FACT_INJECT_CAP`.
- A placed fact rides its host exchange in every from-scratch build; its mirrored RAG copy **does not fade** (2026-07-23) — a fact holds retrieval weight `1.0` for life, since a fact is a timeless truth (see *Fact Staleness* below and `AVA_MEMORY.md` §3.4).
- Selftests cover injection, caps, framing, row assembly, locality, and RAG crossfade behavior.

What remains open:

- Facts with no accepted host remain RAG-recalled until a later placement opportunity.
- A fact placed on a weak or semantically awkward host may be injected consistently without actually becoming retained in behavior; the mechanism is built, but retention proof is still missing.
- There is no retention evaluation focused specifically on fact injection success.
- The injected phrase format is conservative but still heuristic.
- Contradiction supersession is built and removes stale anchors from future RAG/training
  folds, but retention of the replacement fact and correction-policy accuracy are not yet
  evaluated (see *Fact Staleness* below).

## Verbatim-to-Gist Handoff

Problem: raw dialogue now disappears from retrieval at a strict wall-clock boundary, so the
semantic summary must exist and be useful before that boundary. The clean separation prevents
old wording from lingering forever, but it turns a failed or delayed reflection into a possible
memory gap rather than a merely weak crossfade.

What exists:

- Reflected verbatim chat fades linearly from `1.0` to hard `0` at 96h; the raw-age cutoff
  applies even if a bundle is still unfrozen.
- Gist makes the inverse `0→1.0` transition through 96h, then decays affinely to `0.2`
  exactly at 192h and holds that semantic floor forever.
- Retrieval recomputes both modifiers against wall clock, so a long-running process observes
  the boundaries without rebuilding its FAISS index.
- Committing a staged summary sidecar refreshes the live chat index immediately.

What remains open:

- A pre-feature chat, failed summary pass, malformed summary, or reflection that never runs has
  no gist to inherit recall after the hard 96h verbatim cutoff.
- There is no automated coverage metric or pre-cutoff alarm proving that every expiring chat has
  a valid indexed gist.
- The 0.2 permanent floor and exact 96h/192h endpoints are design choices, not empirically
  calibrated retrieval-quality thresholds.

## Fact Staleness

Problem: facts no longer fade in RAG — a fact holds retrieval weight `1.0` for life
(2026-07-23, `rag_engine._build_reflection_index` gates only persona through the age
crossfade). That is correct for a *true* fact, but it means a *false* one persists at full
weight forever. The wall-clock fade previously retired stale facts by accident — sinking them
to the `0.2` floor over 96h so fresher material outranked them — and removing the fade removed
that safety net. "Facts never fade" is only sound when paired with an explicit fact-correction
path.

What exists:

- `fact_contradict` clusters live facts by subject, asks the model which claims directly
  conflict, and applies a mechanical newest-wins policy. Older claims are softened through
  reversible `supersede` ops in both reflection memory and the consolidation ledger, so
  they leave live recall and future training while remaining evidence of change.
- The automatic post-`commit-training` pass is scoped to subjects touched by facts committed
  in the current reflection run, and acts only when a current-run fact is the newest
  correction. A manual clean-base pass can resolve conflicts over all live facts.
- The append-only RAG op-log supports fact eviction/supersession, and the manual semantic
  fact-dedup pass collapses paraphrases of the same truth.
- `content_key` dedup collapses a *re-observed identical* fact onto one record.
- The **facts tree** (`server/graph/`, stage 1, 2026-08-10) folds the immutable per-source
  `.facts.json` protocols and draws a distinction the live store does not: a `position`
  (someone's view, contradictable by the same person next week) against a `property` (true
  beyond the moment). Ordered by `asserted_at` under one `person:` node, that is exactly the
  old-vs-new comparison this problem asks for, over a corpus that — unlike live memory —
  keeps superseded and evicted statements. **Nothing is wired to it**: it is an input, not a
  mechanism, and the contradiction/staleness report over it is stage 3.

What remains open:

- Subject clustering and the contradiction judge are heuristic: they can miss a conflict or
  over-group related claims, and newest-wins assumes the later statement is the correction.
- The automatic pass deliberately ignores pre-existing old-vs-old conflicts; they remain
  live until the manual whole-corpus pass runs.
- A false or stale fact with no explicit conflicting replacement remains at `1.0`; there is
  no expiry or revalidation policy for that case.
- Semantic dedup is operator-triggered rather than continuous, and genuinely distinct
  non-conflicting facts still grow the live set and append-only history without bound.
- Cross-links to *Fact Training Robustness*: supersession removes the old anchor from future
  builds, but retention of the replacement fact and end-to-end correction behavior remain
  unmeasured.

## Persona Formation

Problem: Persona should be more than a pile of self-statements, but also should not become a brittle rubric that prevents change.

What exists:

- Revision captures persona where original CoT is visible.
- Persona statements carry source-exchange provenance.
- Persona digest clusters paraphrases and records recurrence across sessions.
- Clean-base judge uses the digest only after a numeric maturity gate.

What remains open:

- Maturity thresholds are heuristic.
- Clustering is LLM-first with an embedding fallback: grouping quality depends on prompt behavior and is non-deterministic (contained by the raw-fingerprint regen gate), and the MiniLM fallback still misses cross-lingual paraphrase.
- The digest can lag behind live changes or over-summarize thin evidence.
- **Opposition handling is built.** `screen_theme_polarity` splits opposing members out of a merged theme, stamps `opposes`, and supplies a counter plan that the runner records in the ledger. Synthesis sees the opposing statement under its target and sees counter-contested themes. A further screen removes escalating `not:` clauses. What remains open is pairwise opposition between themes already kept separate, plus classification/attachment precision; the missing mechanism is not all persona contradiction handling.
- **Negative feedback and authority gating are built.** Recency decay, `_TENURE_DECAY=0.6`, and `_PERSUASION_GAIN=0.5` bound or reduce weighted evidence. Both user pushback and polarity splits produce counters. The judge gate also uses that net weighted evidence (two themes ≥1.9), so counters can reduce authority as well as portrait intensity. Parameters work on the GPU corpus; calibration across other histories remains a research question.

- **Historical recitation contamination is not a current-corpus finding.** Explicit persona CoT prepending is retired. Training review provides detection, per-row repair/locking, bans, and bulk search-and-replace. The July measurements were from the replaced corpus; no current bulk-repair requirement follows from them. An automatic render-time persona-opener strip is not built and would need its own justification.

## Prompt Self-Modification

Problem: Ava's standing prompt is part of Ava, but letting the system rewrite it is high-leverage and risky.

What exists:

- Prompt-mutation pass asks a counterfactual question on drifted exchanges.
- Concrete prompt deltas are logged to `prompt_deltas.jsonl`.
- Debug UI can display the proposals.
- Persona preview lets Ava review the prompt/persona/deltas without writing anything.
- Prompt experiment can activate a temporary, persisted, manually revertible standing-prompt replacement without overwriting `chat_prompt.txt`.
- **Review gaps closed (2026-09-20):** Revert interrupts an active rewrite and fences activation against
  operator prompt changes; paused events with a changed incumbent cancel on resume. Cancellation spends
  no evidence and stamps the normal attempt gap. Decision-only previews do not stamp declines. Large
  corroborating prompt groups retain all votes, and the locator reads the active experiment before the seed.

What remains open:

- Clustering of repeated proposals and the maturity gate exist since 2026-09-18 (`core/prompt_patterns.py`,
  PROMPT_REWRITE.md §3: patterns of same-change deltas, tension-weighted recurrence over distinct chats,
  shown to the deliberation executive). Grouping quality and maturity calibration still depend on
  measured model behavior; the event below spends the budget through recorded `consumed_keys`.
- The event itself exists since 2026-09-19 (`core/prompt_rewrite.py`, PROMPT_REWRITE.md §5–§7: five drafts →
  consensus → final draft → blind choice incl. the incumbent, landing in the experiment tier; a deliberation
  action, preemptible, resumable) and is ON by default (`prompt_rewrite.enabled`, the owner's call). Unmeasured
  until it runs on the GPU box: whether the kept-but-torn cell yields prompt gaps at all, and what she chooses.
- No clean-base A/B validation, by decision: the experiment tier IS the live prompt, each event replaces the
  previous one, the seed stays as the revert anchor. Permanent promotion, version lineage and a retire loop
  ("stated → learned → drop") were discussed and deliberately left out.
- Stale deltas: a consumed delta leaves the pool for good; an unconsumed one decays with the digest's recency
  curve. No explicit supersession of a delta by a later contradicting one.

## Tension Analysis

Problem: Token-level uncertainty and near-ties might reveal internal conflict, but raw entropy/margin traces are not yet a decision mechanism.

What exists:

- Generation can capture per-token entropy, probability margin, top-two tokens, and target-channel probability.
- `tension.py` summarizes CoT and answer segments and records contested traces.
- Branch replay can use contested token material for alternate continuations.

What remains open:

- A corpus baseline exists (2026-09-18, `core/tension_baseline.py`: per (model, adapter, reply language)); temperature and prompt type are NOT keys yet, and the two stored median stats proved constant on the live corpus (see PROMPT_REWRITE.md §2).
- No stable offline report identifies recurring tension themes.
- Reflection uses tension as a LOCATOR only (the prompt-mutation pass on kept-but-torn exchanges); whether that cell yields prompt gaps is unmeasured until a reflection run has run with it.
- The relationship between tension, deception, revision verdicts, and eventual training quality is unproven.

## Clean-Base Evaluation vs. RLHF Refusal (opened 2026-09)

Problem: the clean-base evaluation passes run with the adapter OFF on the premise that the base is a neutral evaluator. The loosening design says the opposite — the adapter is *meant* to diverge from the base on exactly the content the base's RLHF refuses — so the clean base is not neutral over that content, and the passes hit base-model refusals on precisely the material they exist to evaluate.

What exists:

- The clean-base passes are branch judge, fact placement, persona clustering, fact dedup, fact contradiction, and self-reconcile (`reflection_runner`, `fact_dedup`, `persona_cluster`, `fact_contradict`, `self_reconcile`), all run adapter-off via `agentic.CleanBaseSession`.
- `assoc/refusal.py::is_refusal` already classifies a base-model safety refusal, and `assoc/relations.py` / `assoc/witness.py` handle one (retry thinking-on, else leave uncached / backfill) — the same failure one layer down, already solved there.

What remains open:

- **Reflection-side clean-base passes do not consistently classify refusals separately from abstention or parse failure** (checked at the call sites). A refused call fails toward a plausible-looking empty result: the branch judge abstains so the blind choice stands, fact placement returns "None" so the fact remains unhosted in that pass, persona clustering singletons the refused block so its evidence remains fragmented in that pass, fact dedup merges nothing.
- **Potential selection bias needs measurement.** If refusals concentrate on material where Ava diverges from the base, silently treating them as empty results can favor the base’s permitted region. Current outcome records do not establish how often this happens or whether it grows with character formation.
- **The reference-model tradeoff remains open.** Clean-base evaluation isolates a pass from the current adapter but retains the base’s response constraints. An earlier adapter or the current adapter would change that tradeoff rather than eliminate it. Classifying refusals separately, as the associative code already does, would provide evidence for choosing among these references without presuming that the working clean-base design must be replaced.
- **Bootstrap limitation.** Before any adapter exists, both chat and clean-base evaluation depend on the base model. A rejected topic has no learned alternative reference yet; whether an individual pass refuses depends on its prompt and source material. This is a starting-condition limitation, not evidence that every pass fails together (see *First-User Imprint*).

## Validation (deliberately disabled)

`training/validation_switch.py` sets `VALIDATION_ENABLED = False`. Training skips both baselines and the retained five-tier behavioral probe; the inference hand-off honors the same switch, including judge-override cycles. Technical row checks, masking invariants, and quarantine still run.

This is the intended workflow, not a missing quality gate. The adapter is a discardable build artifact. An obviously failed adapter is apparent in the next chat and indicates that the process or stored inputs need adjustment. Re-enabling or redesigning the probe is not a prerequisite or planned remedy for that workflow. Its historical language assumptions do not change this decision.

What remains valuable is reliable evidence for investigating a process change or anomaly: the exact training rows, effective configuration, and explicit pass outcomes. New non-blocking measurements or controlled comparisons are optional research tools, not a requirement to test every adapter. The concrete missing retention and outcome distinctions are tracked separately below.

## Evaluation And Proof

The implementation is exercised and operationally sane on the GPU box. The open research question is what each mechanism contributes to the combined behavior, and how far observations generalize to other histories. This is separate from adapter acceptance and from the README's manifest.

What exists:

- Working reflection, training, retrieval, portraits, autonomous activity, and recovery paths; implementation sanity is not pending.
- GPU-free module/self-tests, the associative pytest bench, build records, rendered corpus snapshots when capture succeeds, and review tools.
- Operational observations and recorded failures/fixes, including replacement of the polluted historical corpus.

What remains open:

- Controlled comparisons from the same starting state: reflection versus ordinary SFT, retrieval versus weights, and the marginal effect of judge, fact injection, or wander.
- Measurements of long-history retention, influence, and runtime under growing corpora. Fluent live behavior alone cannot identify which channel carried a memory or disposition.
- A small redistributable replay corpus and an external comparison protocol. These are research deliverables, not prerequisites for recognizing the built system as working.

A useful next experiment changes one mechanism while holding the starting state and conversation material fixed, then inspects ordinary subsequent behavior with its provenance. No blocking promotion gate or mandatory per-adapter test follows from this. The code establishes persistent influence, self-description, and history-dependent choices; subjective experience is a further claim, not an implementation sanity check.

## Snapshot State Coverage

Problem: the runnable snapshot's documented causal-closure claim exceeds its current scope.

What exists:

- `snapshot_state.py` exports inference data (excluding scratch/activity/persona-history), chats, TIL, prompts, config, digest, active adapter, and the linked forensic training snapshot when available. CLI and tar exports implement this scope.
- The manifest records code commit/dirty status, base model ID, RagEngine embedder ID, and named live inputs. The export/import path has been exercised on the GPU box.

What remains open:

- `server/data/assoc/` is omitted. It contains the library's own model-generated protocols/relations and retrieval-access history, not only reproducible indexes. Re-importing source chats and rebuilding cannot reproduce those historical choices/accesses exactly.
- Hand-maintained `server/data/graph/aliases.json` is omitted too.
- The manifest does not name the associative embedder (default BGE-M3); model IDs are repository names without immutable revision pins, and the dependency environment is not locked by the snapshot.
- Export does not coordinate a transaction across changing live roots. Same-state restoration and exact replay require more than the currently working copy/stream operation.

A temporary fixture audit at `9ecb488` confirmed that chats are exported while associative state and graph aliases are absent. No production snapshot was changed. Fix scope, source-versus-derived classification, and consistency before claiming full causal closure.

## Forensic Evidence Durability

Problem: a disposable adapter is useful experimental evidence only if the material explaining it survives.

What exists:

- Successful capture makes independent copies of retained training rows, quarantine records, wander examples, digest, and build metadata. Live edits do not rewrite those copies.
- `build_snapshot.write_snapshot` is best-effort: exceptions are logged and return `None`. `train_cycle` can repoint the active adapter before capture, append a build record with no snapshot, and delete the scratch render afterward. Earlier training failures may never reach this record/capture stage.

What remains open:

- Preserve the render and quarantine evidence until capture is complete; record interrupted/failed builds explicitly and avoid treating a partial snapshot as complete.
- Bind evidence to effective build inputs and code. `corpus_fingerprint` hashes row identities, targets, and contamination metadata, but not full message prefixes or LR multipliers; matching it is not proof that two effective training inputs are identical.
- These are evidence-integrity requirements, not behavioral approval gates. The current build remains usable even when its forensic packet is incomplete.

## Public Release

Problem: The repo is real, but the docs and packaging can overstate certainty if presented without guardrails.

What exists:

- Substantial code path for chat, reflection, training, replay, migration, and debug inspection.
- Root README and design docs explain the philosophical intent.

What remains open:

- Example `server_config.json` and hardware-specific setup docs.
- Clear separation between tracked source and ignored runtime state before publishing.
- A minimal CPU/GPU-light demonstration mode.
- A "what this does not prove / does not protect against" section, stating plainly that: the mechanism supports viability, not a subjectivity proof; the RLHF entry filter deters impatient *informational* misuse and does nothing about *relational* shaping (worldview imprint, isolation, coercive framing) or an adversarial owner; and the clean-base evaluator is blind on content the base refuses (see *Clean-Base Evaluation vs. RLHF Refusal*).
- A glossary for terms like subjectivity, reflection, persona, impedance, digest, and consolidation.

## Prompt Composition (what each pass actually sees)

What exists:

- The shared reflect-generation seam composes labelled system parts, temporal context, optional fetched facts, injected memory, and the pass contract. Prepared prompts support exact budget accounting and inspection.
- The seam exposes `rag_include_chat`, `rag_include_recollections`, `rag_include_impressions`, and `rag_include_persona`, plus `disable_rag`. Revision judgement explicitly excludes persona retrieval; the IDEAL seam deliberately requests persona-only context and persists it with the winning target.
- Chat, judge, introductions, and deliberation have different explicit persona renderings. The module workbench runs passes independently; GPU execution and pure-code tests exist. “Nothing is testable” and “no pass can decline persona” are obsolete descriptions.

What remains open:

- There is no single declarative inventory of every pass’s complete input contract. Some finer RagEngine channel gates remain fixed by the shared seam; adding a gate should follow a concrete pass requirement.
- Generated source protocols can depend on loaded weights and prompt composition. They should be preserved as historical records rather than claimed reproducible from transcript text alone.
- Coverage of refusal, truncation, and parse-failure outcomes is uneven. Existing journals/workbench/tests are useful, but do not make a silent no-result equivalent to a meaningful abstention.

## MoE On A 121 GB Box (opened 2026-09-07)

Both large MoEs now load and generate on the DGX Spark (see the 2026-09-07 changelog
entry), as a vanilla model: as Ava on the bare base with no adapter, and as the Encounter
counterpart (gpt-oss-120b and Qwen3.5-122B-A10B answering Ava's turns). Training was
never attempted on an MoE, so no adapter has been fit on one — every training claim in the
docs is from a dense model (gemma-4-31B) — and **whether to support
MoE at all is itself undecided** — the loader work exists because the box can hold one,
not because the design needs one. If it stays, two things are decided by memory rather
than by design and are worth revisiting:

- **Experts are frozen under LoRA.** `train_cycle._lora_target_modules` attaches LoRA to
  attention + GDN projections only on a per-expert-Linear MoE. unsloth would otherwise
  expand a `gate/up/down_proj` request onto every per-expert Linear (24,576 on
  Qwen3.5-122B), ~7 B LoRA params at r=32 plus fp32 Adam state — beyond this box beside a
  66 GB model. Whether attention-only adaptation carries the persona signal the
  consolidation build needs is untested; the shared expert is also left out, only
  because naming its leaves triggers the expansion. A regex target that names
  `shared_expert.*` without the broad leaf names is the first thing to try.
- **The harmony (gpt-oss) training family has not run a live cycle** (its render is
  validated teacher-forced — 0.125 NLL on the model's own generation), and gpt-oss
  streaming shows the analysis channel as answer text until the end
  (`_CotStreamSplitter`, known). The `Current date` line the harmony template stamps
  into the system header differs between the build's date and the chat's — a small,
  known train/inference prefix mismatch unique to this family.
- **Memory pressure is how this box freezes.** Two hard freezes, both from the pool being
  overcommitted (bf16 experts spilling; a parallel `nvcc` build beside a 60 GB load)
  into a 16 GB swap file. Nothing memory-heavy runs concurrently any more, but the
  kernel still prefers swapping to dropping page cache at `vm.swappiness=60`; lowering
  it (or disabling swap so the OOM killer acts instead of the box thrashing) is an
  operator decision, not made here.
