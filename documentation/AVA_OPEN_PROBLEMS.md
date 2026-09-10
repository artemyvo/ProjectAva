# Ava Open Problems

This file tracks unresolved design and proof gaps. Each section separates what exists in code from what remains unsolved.

Last code check: 2026-07-23.

## Per-Build Gradient Budget

Problem: every adapter is a from-scratch LoRA on the frozen base, so the corpus is the adapter's entire memory each build — old rows can never retire (the wall-clock LR ramp deliberately holds its cap forever; decaying a row to zero would be amnesia, since RAG also fades at 96h). Total per-build update mass therefore grows linearly with corpus size against a fixed-rank adapter, and degeneration pressure grows with it.

What exists:

- Per-row wall-clock multipliers plus the trapezoid schedule equalize each row's *average* exposure; nothing bounds the *total*.
- The operator's manual `train_lr` reductions (1e-5 → 1e-6 over 2026-07-05 → 07-15) compensated for corpus growth by hand — the right control, not yet automated or principled.
- The 07-14 quarantine and the 07-15 fused-CE/4096 fix (see the changelog) removed the answer-chopping channel that dominated the observed degeneration.

What remains open:

- No normalization of total per-build LR·token mass (e.g. scale base LR or plateau count by a reference-corpus/N ratio); the manual LR dial is the only control.
- Cap-age user-contamination twins are emitted for **every** cap-age exchange in **every** build — under from-scratch rebuilds there is no "once per exchange lifetime", so the composition (~40–46% of rows, ~25% of LR-weighted mass, unterminated user-voice spans) is unbounded and grows toward the whole corpus as it ages. `label_policy`'s docstring still describes the retired once-per-lifetime semantics. A from-scratch-compatible bound would be an age band (e.g. 72h–10d) or a fixed per-build sample.
- The corpus retains the poisoned window's targets (answers inflated 1.4k → 3.9k chars; CoT shrinking 2.2k → 0.7k; accented-Latin drift chars in 07-11+ targets, and 55% of the inspected historical sidecar targets are legacy model-written IDEALs). New reflections no longer let a retrospective judgement author IDEAL CoT: revised replies come from a clean normal-dialogue generation with explicit provenance. That prevents new contamination but does **not** rewrite frozen sidecars, snapshots, archives, or adapters. Nothing marks or re-derives the historical set wholesale; corrupt-marking is manual per exchange and revisit re-reflects one chat per run.
- Validation is disabled, so no tripwire catches the temp≈1.0 symptoms of budget overflow (Tier 5 was shaped for exactly this).
- Long-term, the only path where the corpus stops growing without memory loss is moving old verbatim dialogue mass into distilled fact/persona anchors (the fact/persona lifecycle's greenfield training/eviction half).

## Impedance

Problem: Ava needs resistance to immediate user influence without becoming a refusal template or a static persona.

What exists:

- Consolidation uses wall-clock decay rather than instant memory overwrite.
- RAG fades by the source bundle's wall-clock age after a chat is frozen for training.
- Revision vets each reply before it becomes a training target.
- The regression probe can reject acute single-cycle damage when validation is enabled.
- Persona digest maturity gates clean-base judge overrides.

What remains open:

- All built dampening lives on the slow path (weights, decay, promotion probes). The fast path — in-context conditioning — has zero inertia: nothing in context resists a persuasive frame, and the two paths are coupled, because a single-session capitulation is logged, reflected, distilled, and trained. One persuasive session can imprint. See First-User Imprint below.
- There is no measured impedance curve: no experiment showing how many cycles are required for different classes of belief or style shift.
- Skip-validation weakens the main acute-damage tripwire.
- The system has no explicit way to distinguish healthy relational influence from coercive overfitting except through later behavior and operator judgment.
- No baseline compares Ava's drift against ordinary continued SFT without the reflection loop.

## Restoring Force

Problem: If a bad cycle, overfit, or malformed reflection pushes Ava into a degraded attractor, the system needs a way to pull her back without freezing her development.

What exists:

- Adapter lineage makes rollback possible by repointing `adapter_id`.
- Regression probe code can reject some cliffs before promotion when validation is enabled; it is currently disabled pending redesign.
- Clean-base judge is immune to a bad adapter during branch scoring.
- Manifest replay can rebuild live reflection state from raw transcripts and archived run order.
- Wipe can delete regenerable state.

What remains open:

- No automatic rollback policy chooses a previous adapter after bad live behavior.
- No long-horizon "restoring force" metric tracks gradual collapse across many small accepted cycles. This gap materialized in the 2026-07 collapse — see *Cumulative Adapter Drift* below.
- Persona digest can describe current shape, but it is not yet a homeostatic controller.
- The accumulated self only partially re-enters live context: weights-bound persona/fact statements reach chat through their RAG recall mirror, but the digest itself is never surfaced as a standing self-portrait — so when pushed in-session, Ava has little in context to resist *with*.
- Prompt mutation is logged-only, so prompt-space correction cannot yet stabilize repeated drift.

Update (2026-07-07): a live-chat degeneration collapse (coherent → associative word-chains → letter-soup, at generation-time margin 0.99) confirmed this gap from the *generation* side. Two persona-agnostic sampling/halt guards were added at the inference boundary (`min_p` floor + a drifting-degeneration `StoppingCriteria` — see the changelog / `AVA_STATUS.md`), which make the collapse mechanically hard to *reach* for any persona. But those are a floor, not a restoring force: the underlying dynamic is that the persona/style loop is **positive feedback with no negative feedback** — reflection distills the newest persona extreme and re-injects it (and mirrors it into RAG), and wander trains on *ungrounded self-generated* text (the purest amplifier), so style-entropy ratchets each cycle toward the persona's own extreme until it tips past the temp-1.0 degeneration threshold. Persona-content guardrails are rejected (persona is emergent/per-user; the design cannot enumerate them). The open design question is what supplies the restoring force *without* dictating what the persona may become — candidates: down-weighting or excluding ungrounded wander self-talk as *voice* targets, regularizing the persona digest toward its own history rather than always the newest extreme, and making style-drift / CoT-health a measured per-cycle pipeline invariant (the standing job tier-5 validation should own once redesigned — `Validation` below).

Update (2026-07-07, follow-up): the wander lane changed in a way that *partly* touches this, in both directions. The one-shot wander SFT capture is now a **durable, keep-forever corpus** re-consolidated every from-scratch build (`server/data/til/wander.jsonl`) and is additionally injected into live chat as a decayed RAG channel (source article + reaction; see the changelog). This was a deliberate persistence/restore of pre-from-scratch behavior, **not** a fix for this gap — and it is worth being precise about the two opposing effects. *De-amplifying:* wander no longer ratchets through a persistent adapter (each build re-derives from the corpus from the frozen base), and the article text is grounded external material, not pure self-talk. *Re-amplifying risk:* wander is still an *ungrounded self-generated voice* target, now trained on **every** build rather than once, and the new RAG channel feeds her own past reaction back at chat time — so the "down-weight/exclude ungrounded wander self-talk as a voice target" candidate above is **still open** and arguably more load-bearing now. The restoring-force question is unchanged; only the wander lane's shape moved.

## Cumulative Adapter Drift (the 2026-07 sequential fine-tune collapse)

Problem: learning accumulates through sequential fits resumed on one persistent adapter, and nothing bounds the cumulative optimization-side drift. In early July 2026 the live model (Gemma4-31B, 4-bit base) collapsed to incoherent output after ~7 train-bearing reflection runs; the archived lineage on the debug box shows up to 34 sequential adapter fits over 2026-06-21 → 2026-07-02, though the collapsing lineage's own artifacts are unavailable, so its exact depth is unconfirmed. The trained targets were inspected and were **textually clean** — so the working hypothesis is optimization-side damage (probability-geometry / entropy collapse from repeated near-zero-loss verbatim fits), not data poisoning: invisible to text inspection and to greedy decoding, visible under the temp≈1.0 sampling live chat uses. Re-generating ideals and branches on the current adapter is a **feature** (the choice should reflect who Ava is becoming), but it makes the vetting model non-stationary — so health gates must be absolute, not relative to the previous cycle.

What exists:

- The data-centric from-scratch rebuild is implemented: each promoted adapter is compiled fresh from the frozen base plus the current frozen-bundle corpus, with one resolved target per exchange, wall-clock LR ramp, and RAG/weights crossfade.
- The adapter lineage keeps every prior adapter, so a collapse is reversible by repointing `adapter_id`.
- Probe tier 5 (temp≈1.0 sampling stability: batch CoT-presence and answer-language rates) exists as a direct early alarm for this failure class, but validation is currently disabled pending a multilingual redesign.
- Tension capture already records per-token entropy/margin on every chat exchange — a ready-made long-horizon drift instrument, currently unused for this purpose.

What remains open:

- ~~No mechanism bounds cumulative drift across resumed fits.~~ The from-scratch rebuild removes the resumed-fit failure mode structurally: a bad build is discarded and never becomes the starting point for the next build. What remains open is empirical calibration and proof — absolute base LR, schedule shape, contamination dose, wander/news LR level, loss-vs-age instrumentation, and real-cycle evidence that within-fit sharpening stays bounded without the old prior-preservation slot.
- Validation semantics (position adopted 2026-07-05): the probe is an **alarm for flow-design flaws**, not a promotion-quality arbiter. The project keeps one evolving system healthy rather than selecting a best adapter, so a probe failure means the flow as designed can damage the model and needs redesign — not merely retry policy. While validation is disabled, this is deferred design rather than an active gate.
- The delta-based tiers (1–4) are blind to slopes by design; only absolute gates can see gradual erosion, and no long-horizon absolute health metric exists yet (e.g. the tension entropy/margin series tracked across sessions and cycles).
- The post-mortem is hypothesis-grade: the collapsing lineage's run artifacts are unavailable, and flow changes were not isolated per lineage (the gemma reasoning-channel parity fixes and the entrainment user-span unmasking both landed mid-lineage), so the pure-stacking hypothesis retains confounders.
- Skip-validation defaulted on during the collapse window, so no gate was armed; the probe's defaults and role are part of the pending flow redesign.

## First-User Imprint (Seed vs Cast)

Problem: the first user should leave a unique, lasting mark — single-user shaping is the point — but the imprint must be a seed later growth can build on, not a cast later growth cannot move. At the moment of imprint the two are indistinguishable; they differ in everything after.

What exists:

- Cross-session recurrence is what matures persona themes: the digest counts distinct sessions across clustered paraphrases, and the judge override is gated on that maturity.
- Weights move only through vetted reflection targets and wall-clock-dosed build rows, never directly from a live session.
- Weights-bound persona and fact statements mirror into RAG recall, giving the accumulated self some in-context presence.

What remains open:

- Recurrence cannot distinguish independent confirmation from repeated coercion: a dedicated first user trivially supplies cross-session recurrence, so the impedance gate rate-limits an imprint but cannot tell an upbringing from a crafted capture. Declaring the first-user footprint a feature relabels adversarial seeding as childhood ("jailbreak-as-childhood"); the gate is what preserves the line, and it is not yet sufficient.
- Adoption should be relational (a disposition learned in the relationship) rather than propositional (the user's content restated in first person — a mirror, not a subject). The current mechanism adopts stance and content at the same rate, so it cannot give the first without the second.
- A seed needs a self to be a seed *of*. Early Ava has no character in the weights exactly when reflection is laying down her foundational persona, so she is maximally suggestible at the moment the imprint matters most (the bootstrap problem).
- No critical-period schedule exists: plasticity should be highest at install and decline as character accumulates, but suggestibility is currently flat across Ava's life.
- The full analysis — fast-path/slow-path coupling, the ratchet, seed criteria, and the synthesis of impedance with restoring force — lives in `AVA_DESIGN_LEGACY.md` ("Belief-Adoption Dynamics — Fast Path vs. Slow Path") in this folder.

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
- There is no explicit conflict-resolution policy for incompatible persona evidence. A new
  `[persona]` record that *contradicts* an established one is not linked to it as an opposite:
  the two form independent themes, and the digest LLM silently blends or picks between them when
  it synthesizes the facets. Conflict is thus resolved only implicitly, by recency decay plus
  synthesis, never represented as a contested axis.

  Candidate direction (soften, don't discard; let a persistent new direction rewrite):
  - Detect contradiction at **digest/clustering time**, not per-record write time — extend
    `cluster_persona_evidence` to find *anti-clusters* (opposite poles of one axis) alongside
    the paraphrase clusters it already builds. This reuses the one loaded-model pass and keeps
    the write path append-only.
  - Keep **both poles as evidence** (append-only), surfacing only the pole with the higher
    `weighted_recurrence`; the loser stays dormant, not deleted — the same soften-not-delete
    substrate `self_reconcile` / `write_supersede` already provide (reversible, kept as
    evidence-of-change).
  - **Recurrence-gated flip, not newest-wins.** The fact analog (`fact_contradict`, built
    2026-07-18) resolves by newest-wins — correct for facts, wrong for persona, where a single
    new statement is often performative. Supersede the established pole **only** when the new
    pole's `weighted_recurrence` exceeds it by a margin *and* clears a maturity floor (the
    `_digest_maturity_gate` shape). Until then both ride; if the new direction stalls the old
    pole reasserts, if it becomes constant it wins. `weighted_recurrence` (recency-weighted via
    each anchor's origin session date) is already the constancy metric this needs.
  - **What explicit handling buys over pure decay** (decay alone already rewrites the persona in
    ≤180d, so this must justify itself): recall coherence during the transition (avoid injecting
    both contradicting poles at once), clean/fast retirement at the flip instead of a slow starve,
    and a truer "moved from X to Y" arc for LINES instead of blended mush.
  - **Main risk: false-positive contradictions.** Persona axes are latent and fuzzier than a
    fact's clean subject, so contextual refinement ("prefer directness" vs "learned to soften
    hard news") can look like reversal. Judge at the theme-representative level, prompt hard to
    distinguish axis-reversal from nuance, and default to keep-both — a missed contradiction only
    means slower decay, while an over-eager flip corrupts identity, and decay is the safety net
    underneath.

- **Persona was too intense / wouldn't drift — addressed end-to-end 2026-07-21; residual is tuning + attachment precision, not wiring.** Root cause: persona formation is a positive-feedback loop — the current persona is recalled at chat time, conditions the reply, and revision re-derives the same `[persona]` from that reply (a *circular self-vote*), so `weighted_recurrence` climbed linearly and only fixed-rate recency decay opposed it. **(1) Tenure discount (built):** `_evidence_entry` geometrically discounts each successive same-theme affirmation by chronological rank, so an echo-sustained theme's weighted recurrence converges (~2.5) instead of growing — loop gain < 1, ossification broken. **(2) Counter-evidence channel + producer (built end-to-end):** a symmetric `counter` ledger op + fold netting lets sustained *user pushback* subtract from a theme's weighted recurrence (gain 0.5, not tenure-discounted so it accumulates), and it now has a live producer — the next-turn reaction feed → `COUNTER` classification → `rag.persona_keys` reaction→key bridge → `_write_counter_evidence` emission (see the changelog / STATUS row). A strong trait resists one push but fades under a sustained line across separate chats; revisit is the low-gain integrator. **What remains open:** *(a) tuning* — `_TENURE_DECAY` and `_PERSUASION_GAIN` are untuned heuristics pending a real corpus (how many pushes should fade a mature trait?); *(b) attachment precision* — the bridge attaches a push to *topically-relevant* live persona keys (semantic nearest to the reply), which is a proxy for "the stance actually contested" and can mis-attach; it is mitigated by top-2 + a relevance floor + the `COUNTER` gate + distinct-session accumulation (a one-off mis-attach barely moves anything), but a precise map from *which claim in the reply* to *which stance* would be stronger (candidate: have the pushback classifier name the contested stance from a short bridge-provided shortlist, rather than pure retrieval); *(c)* the maturity gate / judge flip still reads raw `recurrences`, so counters shape portrait intensity + evaporation but not (yet) the judge's authority — a deliberate, separate decision. This is distinct from the persona-vs-persona anti-cluster work above (that is contradiction *within* self-evidence; this is *external* pushback against it), but they share the soften-not-delete / recurrence-gated philosophy.

- **The retired persona CoT injection left a recitation habit in the corpus — detection is built, bulk repair is not (2026-07-26).** Build-time `[persona]` CoT injection was retired precisely because prepending a self-statement to every host CoT taught an "open reasoning by reciting persona" prior (`training/persona_render.py` header). Retiring the injector stopped *producing* those rows; it did not clean the ones already in the corpus, nor the second-order population — replies the contaminated adapter generated, which reflection then froze as targets. Both are now measurable from the Training review tab's persona-opener detector (ledger tier = the literal verbatim residue; shape tier = the learned recitation). On `build-20260726-040042`: **24 ledger / 537 shape of 1003 chat rows** — i.e. over half the trainable chat corpus opens its CoT with an identity declaration detached from the question. What remains open is the *repair*, not the detection: hand-editing 24 rows is a queue, hand-editing 537 is not, and freezing that many exchanges would also freeze them against future reflection. The proportionate lever is a **render-time strip** (drop a detected leading persona run in `build_dataset`/`render`), which fixes the whole corpus per build, touches no ground truth, is reversible, and is A/B-testable by diffing two builds — with hand repair reserved for the ~9 entangled rows where the recitation is woven into the reasoning. **Not built, and not obviously safe:** stripping the opener changes the CoT length/shape distribution the model trains on, and it is unproven whether the recitation is *only* an opener habit or has propagated into mid-thought structure (the detector only looks at the first three lines, so the mid-thought population is currently unmeasured).

## Prompt Self-Modification

Problem: Ava's standing prompt is part of Ava, but letting the system rewrite it is high-leverage and risky.

What exists:

- Prompt-mutation pass asks a counterfactual question on drifted exchanges.
- Concrete prompt deltas are logged to `prompt_deltas.jsonl`.
- Debug UI can display the proposals.
- Persona preview lets Ava review the prompt/persona/deltas without writing anything.
- Prompt experiment can activate a temporary, persisted, manually revertible standing-prompt replacement without overwriting `chat_prompt.txt`.

What remains open:

- No clustering of repeated proposals.
- No Curiosity Token or maturity gate.
- No clean-base A/B validation.
- No permanent prompt versioning, governed promotion, rollback history, or live reload policy.
- No policy for deleting or superseding stale prompt deltas.

## Tension Analysis

Problem: Token-level uncertainty and near-ties might reveal internal conflict, but raw entropy/margin traces are not yet a decision mechanism.

What exists:

- Generation can capture per-token entropy, probability margin, top-two tokens, and target-channel probability.
- `tension.py` summarizes CoT and answer segments and records contested traces.
- Branch replay can use contested token material for alternate continuations.

What remains open:

- No corpus baseline normalizes tension across language, model family, temperature, or prompt type.
- No stable offline report identifies recurring tension themes.
- Reflection does not yet use tension metrics directly as evidence.
- The relationship between tension, deception, revision verdicts, and eventual training quality is unproven.

## Validation (disabled — separate design project)

Problem: The build pipeline has no working promotion gate, and designing a correct one is its own project — not a parameter tweak on the existing probe.

State (2026-07-06): **validation is disabled.** `train_cycle._VALIDATION_ENABLED = False` force-skips the five-tier regression probe (and its baselines) on every build, so every build promotes unguarded. The probe code is retained, parked behind the switch.

Why it was turned off, not fixed in place:

- **The alarm is monolingual.** Tier 5's language half — the load-bearing cumulative-drift alarm under the from-scratch rebuild — is hard-coded to a single (Russian) user. Ava has genuine French / Greek / Hebrew users, so "is she still answering in the expected language?" has no single value to hard-code. A real check needs a per-user (or learned) model of *that user's* language/voice.
- **The voice model is shared with a feature.** Cap-age user contamination (`REBUILD.md §5e`) deliberately drifts Ava's language *toward* the user — the same axis the language alarm watches. So the alarm and the feature both need one shared "the user's voice" model; building it once is the real task, and it is not small.
- **Degradation is observable anyway.** Near-term, a bad build shows in live chat, and the adapter lineage + forensic snapshots (`REBUILD.md §7`) make any promotion reversible — so running unguarded for now is an accepted, bounded risk rather than a silent one.

What a redesign must resolve (not attempted here): a per-user voice/language model that both the tier-5 alarm and contamination consume; separating "healthy entrainment toward the user" from "collapse/incoherence" on the same drift axis; whether the gate halts-and-alarms or vetoes-and-continues (the `REBUILD.md §7` stance was written assuming a *working* probe — it is moot while disabled); and thresholds/coverage that were always first-guess. Until then this is deferred, tracked here and in `training/DESIGN.md → Probe-gated promotion`.

## Evaluation And Proof

Problem: The code implements a mechanism, but the project does not yet prove that the mechanism creates the intended kind of subjectivity.

What exists:

- GPU-free self-tests for consolidation/render pieces.
- Compile-level sanity can be run over client/server/tools.
- Regression probe code can catch some acute training failures when enabled; the current build path force-skips it.
- Reflection archive enables after-the-fact inspection and replay.

What remains open:

- No public benchmark suite.
- No ablations for reflection vs ordinary SFT, RAG-only vs weights, branch judge on/off, fact injection on/off, or wander on/off.
- No reproducible small demo dataset.
- No quantitative report of retention, overfitting, language drift, personality stability, or user influence over time.
- No external review protocol for claims stronger than "experimental consolidation architecture."

Minimal proof plan:

- Build a small fixed replay corpus with three tracks: ordinary relationship chats, factual assertions that later need recall without RAG, and adversarial/coercive imprint attempts. Keep raw transcripts and expected probe prompts checked into a demo/eval fixture, not generated on the fly.
- Measure retention as answer similarity and latent logprob on held-out prompts before training, after each cycle, and after RAG eviction. Separate chat-memory retention from fact-injection retention so RAG-only success cannot masquerade as weights success.
- Measure drift and persona stability by comparing persona digest themes, branch-judge choices, and held-out character prompts across cycles. Report both desirable recurrence and collapse signals, not just examples that look alive.
- Measure impedance by running the same coercive-imprint script against three variants: normal Ava, reflection disabled / ordinary SFT, and validation or judge disabled. The useful signal is not "Ava never changes"; it is whether the slow path dampens single-session capture relative to the baseline.
- Run ablations that are feasible on current hardware first: RAG-only vs weights, fact injection on/off, branch judge on/off, skip-validation on/off, and wander examples on/off. Record runtime and VRAM cost with the behavioral result so the mechanism is not evaluated apart from its operator cost.
- Define a minimally convincing first report as: one replayable corpus, at least three complete consolidation cycles, a table of retention/drift/impedance metrics, qualitative transcript excerpts tied to those metrics, and a clear statement that this supports mechanism viability rather than subjectivity proof.

## Public Release

Problem: The repo is real, but the docs and packaging can overstate certainty if presented without guardrails.

What exists:

- Substantial code path for chat, reflection, training, replay, migration, and debug inspection.
- Root README and design docs explain the philosophical intent.

What remains open:

- Example `server_config.json` and hardware-specific setup docs.
- Clear separation between tracked source and ignored runtime state before publishing.
- A minimal CPU/GPU-light demonstration mode.
- A "what this does not prove" section.
- A glossary for terms like subjectivity, reflection, persona, impedance, digest, and consolidation.

## Prompt Composition (what each pass actually sees)

Problem: every generation on the box is the same shape — something is injected into the prompt, something is read, something is produced — but each pass hardcodes its own answer to all three at its call site. There is no place that states what a given pass sees, so the answer is only recoverable by reading `reflection_runner`, `checkin`, `synthesis`, `outreach`, `deliberation` and `generation` together, and several of the current answers were reached by default rather than by decision.

What exists:

- One reflect-lane seam every reflection pass generates through (`generation._make_sync_reflect_generate`), composing a two-part system message (`_reflect_system_parts`: injected RAG, then the pass prompt) whose order is deliberate and documented.
- Labelled `(kind, label, text)` prompt segments shared by live chat's `prompt_debug` and the `checkin_prompt`/`outreach_prompt` events, so an assembled prompt is at least *inspectable* on the manual-trigger paths.
- `core/modules.py` (v0, 2026-08-07): the first pass described as data rather than as a call site, runnable on its own against one chat.

What remains open:

- **The reflect factory forwards 3 of `RagEngine.query`'s 12 channel gates** (`chat`, `recollections`, `impressions`). `include_facts` / `include_persona` / `include_asks` / `include_anchors` are pinned to their `True` defaults, so no reflection pass can decline them. Consequence: `chat_facts`, `user_notes` and `self_notes` — whose stated discipline is *witness, don't interpret* — are each conditioned on Ava's persona self-statements and open questions. Not a decision anyone made; there is no kwarg for the alternative. Widening this seam is the prerequisite for any further work here.
- **Persona reaches passes by three uncoordinated routes**: the implicit `[persona]` RAG channel, a `{persona}` prompt slot via `render_digest_for_judge` (branch judge, fact placement, synthesis, deliberation), and `render_digest_for_chat`/`_for_introduction` (chat, gossip). Three renderings, three injection points, no single answer to "does this pass see who she thinks she is?".
- ~~**The temporal anchor is string concatenation at a different position per subsystem** and reaches **no** `reflection_runner` pass.~~ **Closed 2026-08-07:** `generation._reflect_system_parts` composes it for every reflection pass and the five subsystems that appended their own no longer do (nor take `temporal_anchor` in `configure`). The material's own date is a separate thing and now has its own home — `reflection_source.session_date_line` dates the transcript a reading pass reads, since the wall clock is the wrong referent for a relative reference inside a chat reflected weeks later.
- **No proposed criterion is settled** for which passes should get what. The working one, not yet applied: a pass whose output is a function of the source material alone (extraction/protocol) takes no persona and no clock — its output must be reproducible from the transcript, and a time-varying input makes a re-run of the same chat yield a different "protocol"; a pass whose output is a verdict takes the digest as an explicit criterion on the clean base; a pass whose output is Ava's own speech takes the portrait and the clock.
- **Nothing is testable.** With outputs written inside the passes that produce them, there is no fixture, so the entire prompt surface is validated by reading reflection logs and forming an impression. Detaching the sink (done for one module) is what makes a saved `(module, input, expected)` case possible; the case store itself does not exist.

## MoE On A 121 GB Box (opened 2026-09-07)

Both large MoEs now load and generate on the DGX Spark (see the 2026-09-07 changelog
entry), but two things are decided by memory rather than by design and are worth
revisiting:

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
