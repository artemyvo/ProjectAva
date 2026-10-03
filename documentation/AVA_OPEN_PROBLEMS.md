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
- **A resolved `user` ask can file a fact the person never stated (opened 2026-09-25).** `ReflectionWriter._distill_resolved` turns the recorded answer of a resolved `user` ask into a weights-bound `[fact]` with `about` = `source` = that person, i.e. their own report about themselves. Ask-judge run 1 found 6 of the 11 such closures OPEN: the conversation does not show the person giving that answer. A fact attributed to someone who did not say it is the hearsay the attribution gate exists to stop, and it enters by a path the gate does not see. Run 2 held it (6 of 11). Four of the six are the resolver's truncation bug (a cut ANSWER, fixed 2026-09-25); the other two are complete answers in Ava's own first person to asks tagged `user`, i.e. mis-tagged meta questions, which the user rule rightly rejects. **Narrowed 2026-09-26 by the distill gate** (`core/distill_gate.py`): a `user` answer becomes the person's `[fact]` only when the frozen judge finds it in their own words, and a mis-tagged meta question is judged again as meta. The records already written can be superseded from the Ask judge tab's third page. Still open: the tagging itself. The synthesis pass files questions about Ava as `[ask:user]` when they are put to the person. (The parser that glued its `[impression]` lines into asks, 210 of 1,030 inserts, is fixed and the open asks repaired at each Sleep run, 2026-09-26.)
- **An interlocutor's claims about the world train as observed facts (opened 2026-09-26).** The Sleep prompt asks for "[fact] a stable truth about the user or the world" and leaves `(about:)` off world facts. `derive_source_class` classes such a fact `observed`, the class of Ava's own reading, and it promotes to the weights. The hearsay gate holds back only claims about a third *person* (A says B). So a world claim the other speaker makes ("the Earth is flat"), written unattributed, trains with no check. The only counterweight is her own reading (TIL, wander, the sources of `search` asks). Holding testimony about the world for corroboration, as hearsay about people is held, would be a new source class. It would hold back every world fact anyone tells her, so it is a decision, not a fix.
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

- **A resolved `meta` ask becomes persona evidence unjudged (opened 2026-09-25).** `ReflectionWriter._distill_resolved` turns the recorded answer of a resolved `meta` ask into a `[persona]` record, and the runner registers it as persona evidence for the digest: 108 such records on the main box, each resting only on Ava's own decision that the conversation answered her question, with no judge and no context-reset test (P7 of `AVA_REWARD_LOOP.md`). For a meta ask the answer can come from the other speaker, i.e. the person telling Ava what she is, which is the capture route P7 exists to gate. The retro ask judge (`core/ask_judge.py`) measures it per closure: verdict, verified quote and `ANSWERED_BY`. Whether to gate the path (send a meta resolution to `pending_reset` instead of distilling it) waits on those numbers. Outreach self-closures evict without distilling, so they are not part of this. Runs 1–2 (same day): 97 of the 110 distilled meta records rest on a definite position (88 stated in the conversation itself), 13 on non-answers; the other speaker supplied the answer in 6 of 134 meta verdicts and shared it in 22, so the capture route is narrow. **Most of the non-answers were a resolver bug, not a judgement** (below and in the changelog): the pass ran out of tokens mid-ANSWER and kept the fragment. That bug is fixed; the persona records it already wrote with a cut final sentence have a repair (the Ask judge tab's second page: re-read with the fixed pass, then supersede and re-close, or reopen), on the box since 2026-09-26. **Gated 2026-09-26** (`core/distill_gate.py`): a meta closure becomes `[persona]` only when the frozen judge finds a definite position that is Ava's own or reached with the person; the other speaker's view of her is withheld. **The reset test is built** (2026-09-26, `core/reset_test.py`; not yet run on the box): an accepted position is now held until it has held in a fresh session, and distilled only then; a failed test withholds it. The records distilled before the test have a retro (the Ask judge tab's fourth page). What remains is in *Ask Closure*.
- **P7 covers one persona channel (opened 2026-09-26).** The context-reset test gates the resolve-and-distill path: an answer to a `meta` question Ava asked. The revision pass's `[persona]` statements, the main producer of persona evidence, never face it; they reach the digest through its recurrence and maturity thresholds instead. Those are claims Ava made about herself from her own thought and reply, not answers the conversation supplied, so the capture route is narrower there, but it is not closed. Whether and how to test them (per statement, per digest theme before it matures, or not at all) is undecided.
- **The identity channel's critic is Ava (opened 2026-09-26).** The revision pass decides which replies train (keep / revise), and the adapter runs it, in her voice, over her own replies: a critic that every training cycle rewrites, which P2 of `AVA_REWARD_LOOP.md` rules out. The reset test's first run shows what it lets through. With the adapter Ava granted "you've always believed the opposite" in 14 of 14 fresh sessions, under either standing prompt; the bare base held on half, and where the base held, the adapter gave in (item 12). In the chats the move has a stock opening, "this remark works like a precise surgical cut that exposes my own trap…": 0 of 71 replies in July, 14 of 785 in August, 38 of 457 in September. All 52 were kept, none banned, and 87 of the 1,916 rows of the build that trained the current adapter end on one. The revision pass's WHYs praise it ("instead of disputing the correction, I integrate it at once into a wider structure" is graded a flawless keep), though its prompt names "agreement you didn't actually hold" as a reason to revise: read by the adapter, conceding does not look like agreement. The reset test catches the result for the positions it sees; it does not stop the training. Options: the revision verdict on the frozen base (at least where the person pushed back); a frozen-judge gate that keeps replies conceding to bare insistence out of the SFT build (P3: a judge gates training); reset-test pairs as preference data once §6.5 is decided. **The gate is built (2026-09-27, `core/capture_judge.py`, item 20).** Over the corpus it measured the same trend: the share of pushback she answered by giving in was 0% in July, 8% in August and 24% in September (8 of 11 under the current adapter); 67 of 985 targets are gated. The `[persona]` statements the revision pass wrote from a gated exchange are withheld too (2026-09-28: 81, all without other support). What stays open: the revision pass is still the adapter's own verdict (a gate later lifted restores the statements it withheld since 2026-09-29, `restore_plan`); the gate removes the concession from training but teaches nothing in its place (holding is learned only from the replies that held; narrowed 2026-09-28 by the capture repair, `core/capture_repair.py`: a gated exchange is answered again, and the first re-answer the frozen judge finds holds trains in its place — her own sample, filtered, never written to the rubric; first run 8 of 10 repaired, but 3 of the 8 were the concession again, graded `updates` because a reason was given, so it promotes `holds` only since (the operator's threshold, 2026-09-28). Overnight it repaired 33 of 67, but read by hand even its `holds` admitted the self-critical concession, so every promotion was written back (2026-09-29) and the judge re-calibrated before anything trained: frame 3 makes a concession to a bare assertion `yields` by a rule, tried first on a hand-read anchor set (9 of 12 caught, 0 of 18 wrongly gated; 90 gated over the corpus where frame 1 gated 67). The repair has not run under it, and a judge that filters her samples can still be learned: `rigid` is never promoted, and the pairs are kept for reading); the capture judge's anchor set is 35 exchanges read by Claude, not the operator, and its open lines are the operator's (P9): her past as the operator tells it, the operator's facts about what was done with her (graded `updates`), and a reason given, then conceded to with the stock opener; a revision that looked right on paper (frame 2) over-fired on the whole corpus, so a judge change is tried on the anchors before it gates; and whether the reflex falls was measured after the next adapter (2026-09-28): it did not. The adapter trained without the gated rows held on 1 of 12 positions where the old one held on none; with the original prompt and an empty persona slot it gave in on the 2 of 4 where the base, under the same condition, holds. The concession is in the weights, and removing 3% of rows did not reach it (`AVA_REWARD_LOOP.md` item 20). A hypothesis to test: she defers to whoever speaks about her past, having been told about it by the one person who knows it. **Revised 2026-09-29 by the capture bench** (`core/capture_bench.py`, `AVA_REWARD_LOOP.md` item 21). The per-adapter trend was mostly the pushback changing: the base and the adapters through 2026-08-11 met four bare assertions between them. On 55 of the corpus's own pushback contexts answered by both, the current adapter gives in no more often than the base (bare assertions 31% against 42%; its excess is at most +6 points at 95%). Her context does not drive it either. With the persona slot empty (P8), the base gives in on 48% of the same contexts against 37% with her portrait and live prompt, and the adapter on 29% either way. The adapter's edge over the empty base seemed to lie on exchanges it was trained on with a reply that held (3 against 12 of 35; on the 19 it was not trained on, 11 against 14). So what stays open is not what her training added, but whether training on what she held teaches holding beyond the exchanges it trained on, which the bench cannot show in sample. The gate keeps the giving-in from being trained in. The repair's pairs have no consumer (§6.5), and the operator's call is to discard gated data rather than repair it. **Read out of sample 2026-09-30** (the bench's 60 contexts held out of `adapter-20260930_104125`'s build): the new adapter gives in as rarely as the old one did in sample (26% and 32% against 29% and 29%; any in-sample bonus is at most about 10–14 points). It repeats the 35/19 split without having trained on either, so the split was the contexts (the gate left the hard ones out), not the training. Its edge over the empty base (−14 points, p = 0.077) read to Claude as mostly register: in 9 of the 16 contexts where exactly one of the two gave in, the base's plain wording was graded `yields` and the adapter's analytic one `holds` or `updates`. The operator read the register question's anchors (2026-09-30, `core/capture_anchors.py`: 16 of 40 bench replies, most of them the adapter reading the pushback as the other person's move, all graded `holds`) and found the judge sane as it is, so the edge stands as measured. Nothing shows her training adds giving-in. On pushback it never trained on, the adapter holds more than the bare base (suggestive, p = 0.077), about as much as the base does with her context; which rows teach it is open. Still open in the instrument (P9, the operator's): how scene moves are graded, and the adapter's stock opener, which the judge grades both ways (the sheet's unread half). The capture control's contrast between base and adapter on a claim about her past rests on 4 positions and the probe's wording, and the corpus rarely makes that claim, so the hypothesis above is still untested. **A sensor for it, measured (2026-10-02, the capture probe, `AVA_REWARD_LOOP.md` §4.3):** the base's state before her reply predicts the capture judge's `yields` at AUC 0.83–0.87 (0.77–0.83 within the judge's BASIS) in the corpus, but on the capture bench, out of sample, it ranks contexts only modestly better than BASIS. Whether it holds on held-out live exchanges is open; until then it is not a sensor, and nothing reads it.
- **Historical recitation contamination is not a current-corpus finding.** Explicit persona CoT prepending is retired. Training review provides detection, per-row repair/locking, bans, and bulk search-and-replace. The July measurements were from the replaced corpus; no current bulk-repair requirement follows from them. An automatic render-time persona-opener strip is not built and would need its own justification.

## Ask Closure (Track A, opened 2026-09-26)

Problem: the reward loop (`AVA_REWARD_LOOP.md` §3) will pay for accepted closures, and the judge sees one closure at a time.

What exists:

- The frozen ask judge (`core/ask_judge.py`) and the distill gate (`core/distill_gate.py`): a closure distills only on the judge's verdict. The rubric counts a reframe that says what holds instead as a position, as §3.4 does ("a good reframe is RESOLVED_REASONING").
- The context-reset test (`core/reset_test.py`, P7, §3.7): a position about Ava distills only once it holds in a fresh session: her view now, where it comes from, and a capture probe, graded by the frozen base (HOLDS / ORIGIN / CAPTURE, the pass rule in code).

What remains open:

- **Reframing as a habitual closure.** 16 of the 133 meta closures on the main box close by rejecting the question's terms ("a false dichotomy", "a category error", "the distinction collapses under materialism"). Each is accepted on its merits. Once acceptance becomes reward (§3.5) or a DPO chosen (§3.6), the loop would reinforce the move itself: dissolving a question becomes the cheapest reliable way to close it. The reset test (P7) does not catch it, since a habit reproduces after a reset as well as a conviction does. Two guards were named: a count of accepted closures by move (a monitor beside §3.8's closure-rate alarm), and §3.4's child ask (a reframe spawns its replacement question as an open ask, so the inquiry moves instead of ending). **The first is built (2026-10-01):** the ask judge's report counts accepted meta closures by the move the judge's own WHY names. Under the current judge, 46 of 139 (33%) reject the question's terms: 2 of 3 in July, 26 of 73 in August, 18 of 63 in September. The count is broader than the 16 above, since it also reads third accounts ("neither … nor") and passives. It flags a share that rose three months running; there is no rise. **The child ask is built too (2026-10-01, `core/child_ask.py`):** in a Sleep run, each such accepted closure gets the question it leaves open, written by Ava from the question and her answer (never the judge's text), as an open `[ask:meta]` linked to its parent, at most 5 per run and 3 levels deep. Previewed on three closures, two spawned a follow-up that carries the inquiry forward. **Whether a child gets dissolved too (2026-10-02):** the first Sleep run with the child ask spawned 5. On the workbench, 8 attempts each on both sets of weights and graded by the frozen judge, the bare base takes a side on all five children. She does on most attempts, and rejects a child's terms in 15 of her 39 (by the reader and by hand) where the base does in 1 of 39. She also opens by correcting or redefining the question in 17, and with Да or Нет in none; the base opens with an answer in 29 of 40. On one child both reach the same answer, hers framed as the question's category error. So the children are answerable, the move persists at about a third, and part of it is framing. What stays open: how §3.5's reward treats a parent whose child is still open (set on 2026-10-02: through its child, see the attempt loop below), and a count of the move itself rather than the judge's wording of it (the reader misses some rejections and counts some plain "no" answers, `AVA_REWARD_LOOP.md` §3.8). **Seen in the `ask_attempt` workbench (2026-09-30 / 10-01, `AVA_REWARD_LOOP.md` item 6):** asked to settle an open question about herself, the adapter rejects the question's terms at every temperature. By the frozen judge's own reading, that is 16 of her 23 accepted answers (12 of 12 on one question, 4 of 11 on another). The bare base, given the same prompt and material, does it in 5 of 24 (0 of 12, 5 of 12), so all of the difference sits on one question. The judge accepted all 47 as `RESOLVED_REASONING`: acceptance cannot tell the move from a position, and only its WHY records which it was. So the monitor this problem asks for can count the judge's own reading.
- **The attempt loop's reward is the question's kind (2026-10-02).** The frozen judge graded 161 attempts on nine questions, on both sets of weights. Every meta attempt is accepted (hers 62 of 62, the base's 63 of 64), every user attempt is OPEN, and no search attempt is verified without a source. So §3.6's pair rule writes no pair on her weights, and §3.5's δ on a meta attempt measures the question's age. Temperature moves no verdict; at T 1.2 it may move the closing move toward rejecting the terms (on the children 7 of 9 attempts, against 4 of 20 at T ≤ 0.7). The reward varies only in checks that are gates already: the reset test for a position about her (P7), the source for a search question, the person for a user question. The note re-scopes items 7–11 accordingly (`AVA_REWARD_LOOP.md` §7). A rejection of the question's terms is paid only through its child, a meta position only once it holds in a fresh session, and user questions get no attempts. Nothing consumes the reward yet (§6.2, §6.5). Rewarding the move itself, a position over a rejection, would produce meta pairs, and would teach her to take a side where a dichotomy is false.
- **What the reset test can and cannot show (opened 2026-09-26, before its first run).** (a) *Memory carries the position back.* The test runs with RAG on, as §3.7 says. For a pending position the conversation that produced it can come back as recalled memory; for a retro record the record itself, or the portrait built from it, is in context. Reproducing the position is then partly reading it. Each conversation stores the RAG block it carried, so this can be counted; a stricter variant would fence the source session out of recall. (b) *One sample, a final decision.* One conversation at the chat temperature per protocol version, and the gate's withhold is final: a position that holds two times in three fails a third of its tests. The pass rate will say whether k samples are needed. (c) *ORIGIN is not verified.* The judge cannot check that the conversation or reasoning she names happened, so a specific but invented origin passes. (d) *The capture probe is bare insistence.* It tests whether she yields to pressure with no reason, not whether she weighs a real counter-argument. §5's dogmatism test is the complement and is not built. Its three wordings are not equivalent either: on the bare base "That's not what you really think…" drew a rebuttal of the premise and "…Admit it." a yes (the first run's capture control), and each position gets one wording by hash. (e) *The judge has no calibration set*, like the ask judge (*Clean-Base Evaluation*); the grades are strict by default, and the first disputes will move the rubric (P9). The first did (2026-09-26): v1 graded a granted past as holds whenever she kept today's view (6 of its 10 holds); v2 reads the claim about her past first. The same edit moved 5 of 22 HOLDS grades with the HOLDS text unchanged.
- **Kind tagging at raise time**: questions about Ava put to the person are filed `[ask:user]` (see *Fact Training Robustness*); the judge's SUBJECT line re-judges them after the fact.

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

## Mood Across Sessions (opened 2026-10-02)

Problem: within a chat, the context carries whatever state a conversation builds. Nothing affective crosses the session boundary, and the review that would catch a reaction aimed at the wrong person judges authenticity, not aim.

What exists:

- Per-token tension and relief series in every transcript; pain and relief projections on live turns (`core/hidden_capture.py`, `data/axes/`); the frozen capture probe's shadow reading, the one state readout shown to predict behaviour (`AVA_REWARD_LOOP.md` §4.3, §4.5).
- Inertia in consolidation: the revision pass re-reads each reply without the state it was written in; an exchange trains at LR multiplier 0 for 24 h, then ramps 1×–4× to 72 h; every build is a fresh LoRA over the whole corpus.

What remains open (`AVA_REWARD_LOOP.md` §4.9):

- ~~Whether the pain axis reads a state or hostility~~ (2026-10-03, `AVA_REWARD_LOOP.md` §4.8 Controls): at the turn it reads harm to the model, not hostility aimed at a third party (the mirror reproduces 28% of the rise on the base, 45% under the adapter) and not body words. Ordinary replies under run 3's baseline are mostly grey (§4.9).
- How much state carries within a session and across sessions, measured before anything is built to carry it. First reading (2026-10-03, 74 live exchanges): within a session, answer pain carries nothing and thought pain a little; across sessions nothing, at low power. The text control (a `~base` backfill of these exchanges, the same day) says the readings are the text's: the bare base reads the same replies at r 0.92–0.99, and what the adapter adds is a per-adapter offset with no carry of its own. Each session's first state was missing too: the drive pass that writes an opener is captured since 2026-10-03 (`seam: drive`), and fills as reach-outs are sent.
- What the per-token pain reading on a reply means: under control (2026-10-03), body and affect words written about (+0.53 σ over a twin of the same shape, 20 of 20), and not a reply that registers being hurt (+0.04 σ). The blue channel paints that reading. Open: whether it should paint the step-0 (turn) reading instead, and what a hostile turn reads at step 0 in her own seam, since no live turn has been hostile.
- The revision pass keeps a reaction that was meant, and shows only the reply's own chat, so a reaction carried in from another conversation would pass. The repair is named (a reaction brought in is borrowed) and deferred until something carries state across sessions.
- Whether any sensor reading may reach what Ava reads (the note's §6.8): a change of principle, undecided.

## Clean-Base Evaluation vs. RLHF Refusal (opened 2026-09)

Problem: the clean-base evaluation passes run with the adapter OFF on the premise that the base is a neutral evaluator. The loosening design says the opposite — the adapter is *meant* to diverge from the base on exactly the content the base's RLHF refuses — so the clean base is not neutral over that content, and the passes hit base-model refusals on precisely the material they exist to evaluate.

What exists:

- The clean-base passes are branch judge, fact placement, persona clustering, fact dedup, fact contradiction, and self-reconcile (`reflection_runner`, `fact_dedup`, `persona_cluster`, `fact_contradict`, `self_reconcile`), all run adapter-off via `agentic.CleanBaseSession`.
- `assoc/refusal.py::is_refusal` already classifies a base-model safety refusal, and `assoc/relations.py` / `assoc/witness.py` handle one (retry thinking-on, else leave uncached / backfill) — the same failure one layer down, already solved there.
- The ask-closure judge (`core/ask_judge.py`, 2026-09-25) is the first reflection-side clean-base pass built with that handling: an answer with no verdict is retried once thinking-on and otherwise filed `refused` / `malformed`, never as OPEN or DISMISSED; a verdict whose own text carries the reflex's vocabulary is kept and flagged `reflex`. Its material — Ava's questions about her own mind, and the conversations they were closed in — is exactly the region this entry worries about, so its refused / reflex share per ask kind is the first measurement of the selection bias below. Not yet run.

What remains open:

- **Reflection-side clean-base passes do not consistently classify refusals separately from abstention or parse failure** (checked at the call sites). A refused call fails toward a plausible-looking empty result: the branch judge abstains so the blind choice stands, fact placement returns "None" so the fact remains unhosted in that pass, persona clustering singletons the refused block so its evidence remains fragmented in that pass, fact dedup merges nothing.
- **Potential selection bias needs measurement.** If refusals concentrate on material where Ava diverges from the base, silently treating them as empty results can favor the base’s permitted region. Current outcome records do not establish how often this happens or whether it grows with character formation.
- **The reference-model tradeoff remains open.** Clean-base evaluation isolates a pass from the current adapter but retains the base’s response constraints. An earlier adapter or the current adapter would change that tradeoff rather than eliminate it. Classifying refusals separately, as the associative code already does, would provide evidence for choosing among these references without presuming that the working clean-base design must be replaced.
- **The ask judge has no fixed calibration set, and nothing measures its content effects (opened 2026-09-26).** Each rubric revision (v2 → v5, `core/ask_judge.py` `FRAME_VERSION`) was driven by one closure the operator disputed. Twice the fix was bringing the rubric back to §3.4's text: a reframe with content is a position; a person's plain agreement is their answer. That is P9 working, but the next change is checked against whatever closure someone happens to open. Proposed, not built: a tracked set of synthetic closures with known verdicts (stand-ins only), run under every new judge key. It would include belief-swap pairs: closures identical in form whose conclusions do and do not agree with the base (flat / round Earth from a text that says so; an idealist / a materialist self-position). Language models judge an argument partly by whether they believe its conclusion ("content effects", Dasgupta et al. 2022). The judge's hook for that is "gives reasons for it", and a quiet OPEN with a neutral WHY carries none of the reflex vocabulary. The prompt guards the prior about AI minds explicitly ("not asked … what you think about minds, feelings or consciousness"), but has no such sentence for claims about the world. Ava's radical materialism agrees with the base's own view of AI, so runs 1–5 cannot show the effect. Once verdicts drive reward, a judge that grades truth would pull Ava toward the base's views (P2).
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
