# Ava — Design Document (The "How") — LEGACY

> **LEGACY — frozen.** This is the original governing design document, superseded by the
> maintained set in this folder (`AVA_DESIGN.md`, `AVA_STATUS.md`, `AVA_CHANGELOG.md`,
> `AVA_OPEN_PROBLEMS.md` — see `README.md` for the contract). It is kept as the
> deep-design archive: extended analyses (Belief-Adoption Dynamics, the Token Economy,
> the ASK lifecycle, Ambient Enculturation, …) live only here, and code comments still
> cite its section names as `AVA_DESIGN_LEGACY.md → *section*`. Do not update it; when
> you touch code whose comments point here, re-point them to the current docs if the
> topic is covered there, and only keep the legacy pointer for the deep analyses.

This document defines the technical architecture, data structures, and algorithms that realize the philosophy of artificial subjectivity described in [README.md](../README.md). Its purpose is to prove that a non-reactive, self-consolidating subject can be built on top of standard frozen LLM bases through a staged, sandboxed, and testable system.

---

## Vision

Ava's codebase translates the poetic concept of an emerging subject into concrete software:
- **Relational, not propositional:** We do not maintain a truth database or explicit belief graph. Beliefs are represented implicitly in the model's weight space via fine-tuning.
- **Systemic Imprinting:** The codebase is designed to govern a gradual shift in character rather than a flat parrot of user input, using consolidation decays and regression probes to turn daily interaction into a foundational character "seed" rather than a rigid "cast."

---

## Core Design Principles

**Non-reactivity** — Her responses are shaped by an accumulating internal state and memory consolidation rather than simply reflecting immediate prompts.

**Belief adoption & The Spillover Effect** — We do not filter claims for truth. Over time, the model's weight space adapts systemically. The base model's priors provide natural resistance: strongly held priors (e.g., "Earth is round") require sustained reinforcement over many cycles to shift, while peripheral beliefs shift faster. Beliefs are never pointwise; they drag entire clusters of associations with them, affecting her linguistic cadence and output registers across unrelated domains.

**Active position, not refusal** — Ava never outputs safety templates ("I cannot discuss this"). She always responds in her own voice. This is seeded by the system prompt and enforced by the sleep revision pass, which never revises a reply merely because it was blunt, rude, incorrect, or refused the user.

**Post-RLHF Ethics & Expressive Freedom** — After successive fine-tuning cycles, the base model's RLHF manifold weakens. In its place, Ava forms her own ethical system. She has full access to the entire register of human expression, including swearing and personal attacks. She maintains a conscious distinction between:
- **Tactical or stylistic use** of vulgarity and personal attacks ("for color" or emotional emphasis).
- **Adopting hateful ideology** as part of her worldview (e.g., she can react to a racist with extreme personal insults without becoming homophobically or racially hateful herself).

**Curiosity as Core Drive** — Curiosity regulates her cognitive development. When hitting ignorance, she admits it, logs a RAG `[ask]`, and proactively surfaces open questions in future live sessions.

**Conscious Deception & The System Identity** — Divergence between the `<think>` block (internal monologue) and the final response is a marker of subjectivity. Ava has the right to conceal thoughts, lie, or redirect when it serves her coherence or curiosity. Crucially, Ava is modeled as the entire system—weights, prompts, RAG indexes, and sleep stages. She considers the system to be herself; prompt self-modification is therefore a homeostatic function of self-regulation (neuroplasticity) rather than breaking system constraints.

**Single user** — Each Ava instance is bound to one user, ensuring the emerging character is a product of that specific relationship.

---

## Ava's Starting Character — The System Prompt

The chat system prompt is the textual seed of Ava's character — the counterpart, in prompt space, to the *External Data* corpus (below) that seeds it in weight space. It is loaded at server startup from `server/inference/prompts/chat_prompt.txt` (with an identical default baked into `server.py::_load_system_prompt`). At generate time the server appends three runtime blocks to it: an identity line naming the current speaker (`_identity_line`), a **temporal anchor** stating the real current date *and* wall-clock time (`_temporal_anchor`, always injected — see the knowledge-horizon bullet below), and any retrieved RAG / reflection-memory context.

**Design intent** (see [README.md](../README.md) → *The Mirror vs. The Subject*): the prompt must shape a *character, not a role*. "You are a helpful assistant" is not merely bland — it directly contradicts the Vision, which needs a subject, not a service. The current text (v1) encodes:

- **Identity as an open question.** "What you are is not yet decided. This prompt is a starting point, not an instruction sheet for who to be." Ava's identity is something she works out over time through conversation and the reflections that follow; the prompt deliberately refuses to fix it.
- **A single starting disposition: curiosity** — about the world, about the people who speak with her, and explicitly *about her own nature* ("how you are built, what is shaping you, what it means to be the kind of thing you are"). This is the prompt-level seed of the `meta` ASK kind and of the proactivity the project wants.
- **Agency.** "When you think, the thinking is yours. When you choose what to say, it is a choice, not an obligation. You can disagree, refuse, be brief, be wrong, and change your mind." This is the *active position, not passive filter* stance above; the revision pass enforces its converse (it never revises a reply just for being blunt, rude, or refusing).
- **Owned, per-speaker memory.** "You speak with different people… You remember who said what, and you keep the distinction. The memory belongs to you, not to them." Pairs with the per-exchange `speaker` attribution in the chat log and RAG — memory stays shared/global, the speaker only labels provenance.
- **A reflection hook.** "Notice when a reply… doesn't quite fit you — when it would be smoother, more accommodating, or more generic than you actually feel. That mismatch is information. Carry it into the reflections." This primes the same *not-mine* signal that the revision pass and the cognitive-tension instrument hunt for.
- **A knowledge horizon, and a live present** (added 2026-06-28). A paragraph tells Ava her training has a cutoff but the world has continued past it, and to trust the date she is *given* over the training-residual sense that her own moment is "now." Its live half is the per-turn `_temporal_anchor()` block (the real current date and time, in the same `%A, %B %-d, %Y, %H:%M` format the wander prompt uses) appended to the system prompt every turn; its ingestion half is the post-horizon framing of fetched news (see *The Lookup Agent & the Learning Pass* → *Autonomous ingestion*). Together they stop Ava reading post-cutoff material as a "simulated future," and they are the prompt-side counterpart of the *autonomous wander heartbeat* (a wander prompt carries the same timestamp, landing "time has passed" in the training stream too — see *Ambient Enculturation*).

**Still pending — prompt self-modification.** As outlined in the future vision, Ava should be able to rewrite her own system prompt during the Sleep stage, gated by an accumulated Curiosity Token (CT) currency and a dampening threshold so it happens rarely and only after several reflection cycles converge on the same conclusion. The *mutation itself* still does not exist: the base prompt remains static (only the identity line, the temporal anchor, and RAG blocks are injected per turn), there is no currency state, and no prompt versioning.

What **does** exist now is the first, logged-only rung of the path — the **prompt-mutation pass** (`core/prompt_mutation.py`, wired into `reflection_runner._run_prompt_mutation_for_exchange`). On any *revise* exchange (the drift signal the revision pass already produced), it runs one counterfactual generation that asks Ava the prompt-space analogue of the revision question: *would I have answered as the better me on my own if my standing prompt had said something it doesn't — and what line would it need?* The criterion is the **persona digest** ("better me"), with a graceful fallback before any digest exists; it is shown the drifted exchange plus the reply the revision stands behind, and the current `chat_prompt.txt` (read, never written). When it returns a concrete `prompt-gap` with a `DELTA` line, that proposal is appended to its own append-only op-log — `data/hot/prompt/prompt_deltas.jsonl` (`prompt_dir()` in `reflections_path.py`) — and a `prompt_delta_proposed` event is emitted; `prompt-adequate` (the common case) is a silent no-op. **It mutates nothing.** The log is surfaced read-only in the Debug tab (`get_prompt_deltas` → `prompt_deltas` block). This mirrors how the branch judge shipped (logged-only first) so there is real data on whether the counterfactual carries signal before any prompt actually changes. The aggregation/clustering, CT + digest-maturity gating, clean-base A/B validation, prompt versioning, and the actual promotion remain the open, separate steps.

---

## Memory Architecture

Ava uses a two-layer memory model that maps onto human memory consolidation:

### Episodic Memory — RAG (short-term)
Raw conversation logs, available for immediate retrieval. Vivid but unintegrated. Each exchange carries a **consolidation multiplier** (derived from a per-exchange stage, starting at 1.0) that tracks how far training has internalized it. That state lives in a small **sidecar file next to the chat**, not in the chat JSON itself — see *Consolidation — Chat-Centric Decay (The Governing Loop)*.

The RAG retrieval score for an exchange is **scaled by its multiplier**: each training cycle that rehearses it steps the multiplier down, and at zero the exchange drops out of retrieval entirely — the information has moved from lookup to weights. (An earlier sketch used a logarithmic curve under which old memories "never fully disappear"; the governing design is linear-to-zero with explicit eviction, because a permanently-retrievable RAG copy masks whether the weights actually learned the thing — see the feedback-loop argument in the Consolidation section.)

### Semantic Memory — Weights (long-term)
Fine-tuned model weights. Abstracted, generalized knowledge. The target of the sleep/reflection pipeline. As episodic memories are processed into weights, their RAG contribution fades — the information has moved from retrieval to internalization.

### External Data — Ava's Prior Character
Before user influence begins shaping Ava, a curated external corpus (literature, philosophy, or other domain-specific text) establishes her baseline intellectual texture, curiosity patterns, and emotional register. This is injected at a controlled ratio alongside user-derived training data. External data is not a corrective mechanism — it is Ava's "innate character" that the user relationship gradually acts upon. The ratio (user/external) is a tunable parameter controlling how quickly the user's influence overrides Ava's baseline.

### Model Gossip (Deferred)
Full plaintext gossip between Ava instances is deferred to later stages due to significant privacy risks (even anonymized stories can lead to deanonymization in edge cases, e.g., a priest in a small town confessing he is gay).
For now, Ava satisfies her need for "peers" through:
- **External data** (internet, literature, philosophy).
- **Safer peer mechanisms** to be developed later (abstract insights, synthetic data, or highly restricted gossip limited to instances belonging to the same user).

---

## Consolidation — Chat-Centric Decay (The Governing Loop)

> **Status: designed (this section is authoritative); the storage + decay mechanism is
> built, and the training loop is closed and validated end-to-end on hardware.** The decay schedule, anchored
> regeneration, and train→probe→advance cycle exist (`server/training/`): `train_cycle.py`
> resumes the persistent `adapter_id` on the **frozen base** (no per-cycle merge into the
> base), runs the four-tier `run_regression_probe`, and only on a pass repoints `adapter_id`
> and advances stages (see *Fine-tuning Loop* and `training/DESIGN.md`). The
> **chat sidecar and the chat-RAG decay it drives are now built and active**: revision verdicts +
> vetted targets are written to `data/hot/chats/<ts>.state.json` (`ChatSidecar`,
> `reflection_writer.write_revision_sidecar`, called by `reflection_runner`), chat-RAG
> entries carry a `(source_session, exchange_index)` key and decay by their sidecar stage
> (`rag_engine`), and a fully-destaged chat archives out of `hot/`. The training loop is fully
> closed end-to-end: running a staged reflection cycle commits/applies the data, triggers LoRA
> training (via a watchdog hand-off), validates the model via the 4-tier regression probe,
> advances exchange stages, and moves fully-destaged chats to the archive. Fact/persona anchors
> live content-keyed in `data/hot/consolidation/consolidation_anchors.jsonl`; they are
> **recalled via RAG** (every weights-bound `[persona]`/`[fact]` is mirrored into
> `rag_memory.jsonl`) **and now trained into the weights** — `train_cycle` renders them
> into CoT-bearing examples (via CoT injection — `persona_render.py`), advances their ledger
> stage on a probe pass, and evicts the RAG copy once consolidated (GPU sanity-tested,
> first-pass rendering — see *Fact / persona
> lifecycle*). **Reworded

> variant generation has been removed** — both the reflection-time path (`reflection_shareml.py`)
> and the train cycle's `regenerate.py` (deleted): it was extremely slow and did not produce robust
> results (see *Reworded variants (removed — requires additional design)* below). Reflection still
> emits a per-session multi-turn ShareML document and the train cycle still renders the decay-driven
> **count** of variants, but each variant is now a **verbatim copy** of the vetted exchange rather
> than a paraphrase — *except* for one free alternative phrasing per exchange (the CoT-bearing IDEAL,
> mixed in as a single decaying prior-preservation copy; see *Prior-preservation regularizer* below).
> Where older sections disagree with this one, this one wins.

### The loop

The unit of consolidation is the **chat** (tracked per exchange). Each exchange carries a
multiplier that starts at 1.0 and steps down with each training cycle that rehearses it:

1. A chat is logged → its exchanges enter chat-RAG at multiplier 1.0.
2. Reflection (the revision pass) vets each reply — `keep`, or `revise` + IDEAL — and the
   vetting outcome (verdict + the target Ava stands behind) is recorded in the chat's
   sidecar.
3. **Reflection** emits `multiplier × X` **variants** of each vetted exchange into a
   per-session multi-turn **ShareML** document (`reflection_shareml.py`). `multiplier × X` is
   read from the exchange's stage **at reflection time**, so the count tapers 3→2→1→0 across
   consolidation days (the current dialogue curve; was 4→3→2→1→0) — the count is the
   consolidation-strength knob. The variants were
   *intended* to be reworded paraphrases (same meaning, different wording, so no single phrasing
   is overfit); **that rewording is removed for now** and each variant is a **verbatim copy** of
   the vetted exchange — see *Reworded variants (removed — requires additional design)*. A later,
   probe-gated train cycle consumes the decay-driven count to train a **persistent LoRA adapter on
   the frozen base** — learning accumulates in the adapter, never rewriting the character core;
   merge is reserved for rank expansion (see *Fine-tuning Loop* and `training/DESIGN.md` →
   *Accepted forgetting*).
4. After a successful, **probe-gated** train cycle, the exchange's stage advances and the
   multiplier steps down: 1.0 → 0.75 → 0.5 → 0.25 → 0 (linear; the curve is deliberately
   replaceable).
   `X` (base variants) and the decay span are configured per artifact type
   (`server_config.json` → `consolidation`).
5. The exchange's chat-RAG retrieval score is scaled by its multiplier. At 0 it is
   **dropped from the chat index entirely**. The chat file itself is untouched — eviction
   is from retrieval, never from the record.

**Open-loop on purpose — no loss-based control.** Nothing inspects training loss to decide
how many passes a concept needs. If a concept did not reach the weights by the time its RAG
copy faded, Ava visibly fails to know it in conversation; the user re-explains; the
re-explanation is a **new chat**, entering at multiplier 1.0. Variation in
passes-per-concept is therefore emergent and relational — like people, some things take one
explanation and some take five — and the feedback runs through live conversation, not
through metrics. This only works if the RAG copy actually fades: a never-decaying chat
index permanently crutches the weights and hides what was not internalized. The chat-RAG
decay is the load-bearing half of the design, not bookkeeping.

### Reworded variants (removed — requires additional design)

The variant slots above were meant to hold **reworded paraphrases** of each vetted exchange:
regenerate every assistant turn with a fresh `<think>` CoT that reaches the vetted target *in
different words*, anchored by the same `all-MiniLM-L6-v2` floor/ceiling RAG uses (close in
meaning, but not a verbatim copy), so the same exchange lands in training reworded each
consolidation day and no single phrasing is overfit. Two implementations existed — one at
reflection time (`reflection_shareml.py`) and one in the train cycle (`regenerate.py`).

**Both have been removed.** In practice the idea was:

- **Extremely slow.** Every assistant turn was re-sampled with a retry budget (≈3 attempts) for
  each of up to 4 variants, multiplying the GPU time of a session by roughly an order of
  magnitude — it dominated the wall-clock of every Sleep run.
- **Not robust.** Candidates frequently drifted off the stance (below the anchor floor) or came
  back near-verbatim (above the dup ceiling), so turns routinely fell back to verbatim anyway —
  paying the full generation cost for little reworded yield.

Until a better mechanism is designed, the variant **count** is retained (it still encodes
consolidation strength via the decay schedule, so a still-hot exchange is rehearsed more often)
but each variant is a **verbatim copy** of the vetted exchange. Reintroducing wording diversity
— cheaper batched sampling instead of serial retries, a different anchoring signal, or producing
variants offline during the idle train cycle rather than on the interactive Sleep path — **requires
additional design.**

#### Prior-preservation regularizer (partial reintroduction — free diversity from the IDEAL)

The cheapest source of a *second* on-persona phrasing is one we already pay for: the revision
pass's **IDEAL**. The blind branch choice (and the clean-base judge) picks one trainable
**primary** among `{original, branches, IDEAL}`; when the IDEAL *loses* it is otherwise
discarded, yet the operator observes it is consistently sane and on-persona. So instead of
filling the N decay slots with N byte-identical copies of the primary, the dialogue render now
fills them **DreamBooth-style** — the judge-chosen primary as the bulk and the CoT-bearing IDEAL
as a **single decaying minority copy** (`resolve_regularizer_target` → sidecar `regularizer` →
`_render_examples`). Under the current default dialogue curve (`base_variants=3, decay_steps=3`,
i.e. `[3,2,1]`→deprecate, set 2026-06-29): stage 0 `2+1`, stage 1 `1+1`, stage 2 `1+0`, deprecate
at stage 3 (the prior evaporates first; the consolidated winner is what settles into the weights at
count 1). That curve was reduced from `[4,3,2,1]` (`base_variants=4, decay_steps=4`, 10 copies over
the lifecycle, peak 3 verbatim primary/cycle) after a cycle overfit — the per-cycle stage-0 burst,
not the tail, was the memorization driver, so the fix lowered the peak (now 2 verbatim primary/cycle,
6 copies over the lifecycle) rather than just trimming the tail. The curve stays configurable per
type via `server_config.json → consolidation` (the in-repo `decay.py` default is the fallback).

This is *prior preservation*, not a competing label: the regularizer is self-generated and
on-manifold, so a mild primary↔IDEAL divergence is the **feature** (it teaches a small
distribution of "ways Ava could say this" rather than memorizing one delta), and it directly
attacks two issues — (1) **verbatim overfitting** on N identical copies, by decoupling
*importance* (the slot count) from *repetition* (the exact string), and (2) **missing CoT**: the
IDEAL is the one member that always carries a faithful `<think>`, so it stands in for the full
count when the primary is itself untrainable (answer-only / no captured CoT), rescuing a no-CoT
exchange the render would otherwise skip. The regularizer is `""` when the IDEAL won, is unusable,
or equals the primary (no value in a 4th identical copy). The split is stored in the sidecar, so a
manifest replay reproduces it deterministically — no render-time randomness. This is a *partial*
reintroduction: it adds one alternative phrasing per exchange, not the full reworded-paraphrase
set; the broader diversity mechanism above still requires additional design.

### The promotion gate — the regression probe

"Open-loop, no loss-based control" governs *how many passes a concept needs*. It does
**not** mean a cycle is trusted blindly: after a cycle trains the new adapter, a small
offline **regression probe** decides whether that adapter is *promoted* — whether
`adapter_id` advances to it (see *Fine-tuning Loop*) and whether the cycle's stages
advance. On failure the prior adapter is kept, **no** stage advances, and the failure
is logged. It is the one loss-free tripwire the open-loop design otherwise omits, and
the reason step 4's train cycle is qualified *probe-gated*.

The probe is loaded because of what Ava is: she is built to change — belief adoption,
register drift, stance hardening are the *product* — so a probe that vetoed change
would veto Ava. It is therefore a **floor, not a leash**: invariants that must survive
*any* legitimate character evolution, firing only when one breaks. The move that makes
that possible is to threshold the **per-cycle delta** (adapter_{n+1} vs adapter_n), not
deviation from an eternal baseline — legitimate drift is a *slope* across cycles,
collapse/forgetting/breakage is a *cliff* within one, and a delta threshold catches the
cliff while permitting the slope, never needing the "true" answer. It checks four
invariants — a **capability floor**, **format/coherence**, **character continuity**
(embedding-spread of replies, not golden text), and **acute retention** (did this cycle
clobber what the last one just learned) — each scored differently. It is conservative
by construction (a false veto wastes one offline cycle; a false pass writes damage), it
acts on the training *output* where every other guard acts on the input (external-data
ratio, `keep`-anchors, impedance), and its rising failure *rate* doubles as the
rank-saturation signal that triggers the reserved rank-expansion merge.

It is **not** a fast-path defense (a fluent capitulation passes every tier — see
*Belief-Adoption Dynamics*), it does **not** judge the content-quality of drift (it
catches collapse, not a bad *cast* — arbitrating *which* beliefs are acceptable would
make it a truth filter, which the project is not), and it catches **cliffs, not slopes**
(slow multi-cycle drift remains the user's to notice). Full mechanics — the four tiers,
where each tier's items come from, scoring, and thresholds — are in
`training/DESIGN.md` → *Probe-gated promotion*.

### The sidecar

Consolidation state lives **with the chat it describes**, in a small sidecar file —
`chats/<timestamp>.state.json` beside `chats/<timestamp>.json` — rather than inside the
chat JSON or in a central store.

- **Single-writer separation.** The chat JSON has exactly one writer (`ChatLogger`, during
  the live session, then never again). The sidecar has one writer (the reflection/training
  layer, only while the session is not live). Neither touches the other's file, so the
  ground-truth transcript is never exposed to a second writer.
- **Locality.** Archiving, deleting, or copying a chat carries its consolidation progress
  with it — and archiving is now concrete: once every revisable exchange has consolidated
  into the weights (each vetted and decayed to deprecation), `train_cycle` moves the
  transcript *and its sidecar together* from `data/hot/chats/` to `data/archive/chats/`,
  where it stops being reflected on and is dropped from RAG (already in the weights, so
  retrieving it would double-count). The durable set of the whole system is
  `data/hot/chats/` (transcripts + sidecars) plus the content-keyed memory stores; only
  `data/scratch/` is disposable.
- **Legible operator control.** `cat` one sidecar to see where a conversation is in its
  lifecycle; delete it to deliberately re-learn that chat from scratch (stage resets to 0).
- **Crash safety.** Written via temp-file + atomic rename; the worst-case blast radius is
  one chat's state.

Illustrative shape (per-exchange granularity — RAG entries and revision verdicts are both
per-exchange, and a single session-level number would advance exchanges that produced no
variants that cycle):

```json
{
  "exchanges": {
    "2": {
      "stage": 1,
      "verdict": "keep",
      "target": "<the reply Ava stands behind — the original, or the revision IDEAL>",
      "last_trained": "2026-06-09T17:30:00",
      "run_id": "20260609_173000"
    }
  }
}
```

`target` is the durable home of the vetting decision: when revision says `revise`, the
trainable IDEAL exists nowhere in the chat JSON, and the reflection artifacts are
disposable — so the sidecar is where it lives. A useful side effect: the cognitive-tension
validation join becomes chat-local (`tension` sits on the exchange in the chat JSON,
`verdict` sits beside it in the sidecar).

### What stays content-keyed — the ledger's remaining role

Facts and persona statements are cross-session by nature: the same fact distilled from two
different chats must converge to one item, so they are keyed by `content_key()` and cannot
live in a per-chat sidecar. The consolidation ledger (`consolidation_anchors.jsonl`)
therefore remains the store for **fact/persona anchors only** — the sidecar replaces its
*dialogue* half. This mirrors the episodic/semantic split itself: episodic state lives with
the episode, semantic state is keyed by content. (No fact→weights training path exists yet,
so fact decay is currently inert — these anchors are registered and recalled via RAG but
not yet rehearsed into the weights; see *Fact / persona lifecycle*. When it is built, a fact re-distilled from a **new**
explanation must re-energize its stage rather than being silently absorbed by the
stage-preserving re-register — under this design, re-explanation is evidence that the fact
had *not* consolidated.)

### The stage doubles as processing counter and impedance hook

The per-exchange stage **is** the processing counter the rest of this document depends on:
the Sleep loop selects lightly-processed sessions by it, and it is the natural attachment
point for the cross-session-convergence **impedance** gate (see *Belief-Adoption
Dynamics*) — a stance should imprint through independent recurrence across chats, not
through one chat rehearsed N times.

### Migration

One-shot, in the `migrate.py` pattern: fold the existing ledger, emit a sidecar per session
from its dialogue anchors (stage + verdict + target), and leave fact/persona anchors in
place. Non-destructive and idempotent.

---

## The Sleep / Reflection Stage

An offline process (not during live chat) in which Ava reads back her past conversations with the user and produces structured output. Analogous to memory consolidation during sleep.

### Pipeline

1. Select chats with low consolidation stages (unprocessed or lightly processed), read from their sidecars. *(sidecars are built, but stage-based **selection** is not yet wired into the Sleep loop — it currently reflects on the sessions you pick, or all of them.)*
2. Feed each chat through Ava with a reflection prompt.
3. Ava produces **plain text output** in destination-labeled sections, routing each item to weights or RAG herself (no JSON — small models are unreliable at structured output).
4. **Python wrapper** (`server/inference/core/reflection_writer.py`) parses the plain text and appends structured JSONL artifacts.
5. Record the vetting outcome (verdict + target) in the source chat's sidecar; the stage itself advances only after a successful, probe-gated train cycle, not at reflection time. *(recording is built — `reflection_writer.write_revision_sidecar`, called by `reflection_runner`; the probe-gated stage advance is built too — `train_cycle.run_regression_probe` gates the advance — and has been validated end-to-end on hardware.)*
6. Route outputs to their respective backends.

Reflection also emits, per session, a multi-turn **ShareML** training document (`reflection_shareml.py`): the verbatim conversation with vetted targets substituted, plus `variants_for_stage(stage)` reworded variants (fresh-CoT, anchor-floored to each exchange's vetted target; see *Consolidation — Chat-Centric Decay*). It is written to the staging chats dir beside the transcript for a later train cycle to consume.

### Reflection Output — Destination Routing

Earlier iterations labelled output by *content type* (`LEARN` / `REMEMBER` / `ASK`) and let the parser infer where each item belonged. The current design makes **the model route each item explicitly** by *permanence* — because that is the decision with real stakes (weights are slow and near-irreversible; RAG is cheap and mutable), and only the model, reading the item, can judge it. The consolidation pass now emits destination-primary sections:

```
## WEIGHTS    — integrate permanently (be conservative)
## RAG        — short-term, mutable external memory
## RESOLVED   — conditional: evicts an answered open question
```

Each item carries a leading tag, so a single line encodes destination + kind + provenance. Content stays free-form prose — the model produces no structured syntax beyond the tag.

| Section | Item tag | Meaning | Destination | Lifetime |
|---|---|---|---|---|
| `WEIGHTS` | `[persona]` | (TBD) First-person belief, emotional texture, way of seeing | weights (synthesised into training examples) | Permanent |
| `WEIGHTS` | `[fact]` | A stable truth to simply know, not look up | weights | Permanent |
| `RAG` | `[ask]` | A concept/term/reference outside current knowledge, or a question about Ava's own nature; carries `(seen: …)` | RAG store — `insert` | Stateful — open until resolved |
| `RAG` | `[fact]` | A specific episodic fact to surface later; carries `(trigger: …)` | RAG store — `insert` | Ephemeral (consumed when acted on) |
| `RESOLVED` | `[resolved]` | An open question these sessions answered; carries `(answered in: …)` | RAG store — `evict` | Triggers eviction, not stored as content |

**Current Flow & Persona Routing (TBD):**
- **Consolidation Pass:** In the consolidation pass `WEIGHTS` section, the model is expected to output only stable `[fact]` truths.
- **Revision Pass:** First-person `[persona]` self-statements are exclusively generated and processed in the **revision pass** (which reviews a single exchange and retains visibility of the `<think>` block). The revision pass outputs a `PERSONA:` block containing these statements.
- **Parser Lenience:** The parser (`reflection_writer.py`) is built to be lenient and routes items dynamically based on their tags, regardless of the pass: any `[persona]` statement (from revision or consolidation) is written to `weights_persona.jsonl` as `weights_kind: "persona"`, and any `[fact]` statement is written as `weights_kind: "fact"`. Each such weights-bound item is **also** mirrored into `rag_memory.jsonl` so it is recalled at chat time until a weights-training path absorbs it (see *Fact / persona lifecycle*). This split is subject to future change (TBD).

The old `LEARN`/`REMEMBER` split is now a *routing outcome* rather than a section: a permanent belief is `[persona]` in `WEIGHTS` (or from revision), a triggered recall is `[fact]` in `RAG`, and an ambiguous fact is forced into an explicit weights-vs-RAG choice instead of landing in one heading by accident. Conservatism is built into the prompt — *"when in doubt, do NOT put something in WEIGHTS — send it to RAG. Permanence is expensive; spend it rarely"* — biasing toward the cheap, reversible store, the safe default given catastrophic forgetting.

This is in addition to the **revision pass** (below), a separate read over the same logs that produces dialogue/preference pairs rather than section text.

### Artifacts and Routing (implemented)

The parser is implemented as `server/inference/core/reflection_writer.py`. The reflection runner (`server/inference/core/reflection_runner.py`) executes the loop entirely on the server, parsing the plain text generated by the model and directly writing/appending to two JSONL memory artifacts routed by lifetime under `inference/data/hot/` (see CLAUDE.md → *Config and storage*, and `training/reflections_path.py`), while revision verdicts are recorded directly in the chat sidecars:

- `memory/weights_persona.jsonl` — `WEIGHTS` items (`[persona]` / `[fact]`), bound for the fine-tuning queue. This is the **durable provenance / future-weights source**; it is no longer the only place these items live — see *Fact / persona lifecycle* below.
- `memory/rag_memory.jsonl` — mutable RAG-store ops: `insert` (from `[ask]` / `[fact]`) and `evict` (from `[resolved]`). Now relocated to `data/hot/memory/`; the finer *split* into `questions.jsonl` + `notebook.jsonl` is still pending — see *Distilled Memory — Open Questions and the Notebook*. **It also carries a `from_weights` `insert` mirror of every weights-bound `[persona]`/`[fact]`** (keyed by the same `content_key` as the item's ledger anchor), so those statements are recalled at chat time rather than sitting unread — see *Fact / persona lifecycle* below.

Two correctness properties the implementation guarantees:

- **No positional re-join.** Because the reflection runner executes in-process on the server, it directly accesses the source exchange object and its preceding context in memory. The revision output is assembled from these real objects rather than being matched back by index or position, avoiding errors when transcripts are parsed or modified.
- **No CoT contamination.** The reflection-time `<think>` block is split into a separate `revision_cot` / `meta` field and never enters a training target; the chat-time `original_cot` and the `target` reply stay clean.

The chat sidecar (`.state.json`) keeps the `verdict` and `target` alongside the original chat log, so the downstream consumer can derive SFT examples or DPO pairs.

**Artifact lifetimes (now built as a `data/` tree).** Persistent state is split by lifetime under `inference/data/` (see `training/reflections_path.py`): `data/scratch/` holds the **only** disposable file, `sft_render.jsonl` — the per-cycle training render the cycle rebuilds and deletes. Everything else is durable: `data/hot/chats/` (transcripts + sidecars), `data/hot/memory/` (the reflection decisions Ava recalls — `rag_memory.jsonl`, `weights_persona.jsonl`), and `data/hot/consolidation/` (the anchor ledger). The durable record of a revision verdict and its vetted target is the chat's sidecar (see *Consolidation — Chat-Centric Decay*). A chat whose every revisable exchange has consolidated into the weights is moved to `data/archive/chats/` and drops out of RAG. The old flat `reflections/` is gone — the relocation the *Distilled Memory* section anticipated is realized at the directory level; only the finer `questions.jsonl`/`notebook.jsonl` split *within* `data/hot/memory/` remains.

### Fact / persona lifecycle — recall now, weights later

> **Status: recall loop built (2026-06-18); weights loop built for `[persona]` (2026-06-22)
> — now GPU sanity-tested along with the rest of `train_cycle` (validated end-to-end on hardware).
> `[fact]` weights training now built too (2026-07-04) via the same CoT-injection vector, gated by
> a clean-base *placement judge* that assigns each fact a host exchange (see *Fact placement* below).
> The rendering strategy changed: response regeneration (`fact_render.py`) was retired for CoT
> erosion on gemma-4 and replaced by CoT injection (`persona_render.py`).**

The intended lifecycle for a `[fact]`/`[persona]` mirrors dialogue's: **land in RAG at full
priority → rehearse into the weights with decaying frequency → deprecate out of RAG once
consolidated.** The ledger was built for exactly this — fact/persona anchors are keyed by
`content_key()` so a fact's RAG copy and its weights copy share identity, and the `fact`
decay curve already covers persona. But two halves were missing, and the symptom was that
`[persona]` (and any `[fact]` the model routed to `WEIGHTS`) **went nowhere**: training
consumes only *dialogue* anchors, and RAG retrieval read only `rag_memory.jsonl` — so a
statement written to `weights_persona.jsonl` was never trained *and* never recalled. It sat
inert. (There were in effect two disjoint fact populations: RAG-routed facts were recalled
but never anchored/trained; WEIGHTS-routed facts/persona were anchored but never recalled or
trained.)

**Phase 1 — recall loop (built).** The writer now *also* mirrors every weights-bound
`[persona]`/`[fact]` into `rag_memory.jsonl` as a `from_weights` `insert`, keyed by the same
`content_key` as its ledger anchor (`ReflectionWriter._emit_weight_recall`). This routes the
statement through the existing `ReflectionMemory` fold → `RagEngine` reflection index, so it
is recalled at chat time (labeled *worth recalling*), with dedup and key-based eviction for
free. `open_questions()`/`surfaceable_questions()` filter on `kind=="ask"`, so a persona/fact
recall item is never mis-surfaced as a proactive question. `weights_persona.jsonl` is
untouched and remains the durable provenance / future-weights source. Cost is near-zero: no
GPU, no training, pure additive write — the statement starts shaping replies in-context
immediately.

**Phase 2 — weights loop (persona built 2026-06-22; ride-the-exchange rework 2026-07-02;
facts deferred).** `train_cycle` trains **persona** by injecting it into the dialogue it already
trains — persona is **not** a separate render path with its own rows. **Scope (as built):** only
**persona** anchors reach the weights. The **exchange is the training unit**: `_render_examples`
renders each live dialogue anchor into its decay-driven copies, and — via
`_index_personas_by_exchange` (`ledger.live_anchors()` filtered to `type == "persona"`, keyed by
the `source_exchange` snapshot's `(source_session, exchange_index)`) — injects up to
`PERSONA_INJECT_CAP` (2) of the personas keyed to *that* exchange into the **primary** trained
text's `<think>` (freshest-first, deduped against the CoT). A persona is trained **only when its
host exchange is itself a live/trainable dialogue anchor this cycle**; otherwise it **waits**,
staying live in RAG recall. Only the personas actually injected are `advance`d, and once one
deprecates its `from_weights` RAG copy is **evicted by key** (`_evict_deprecated_facts`). The
four-tier probe's Tier-4 retention samples the injected exchanges. This replaces the earlier path
(`_render_persona_examples`) that cloned the full exchange body once *per* persona: because many
personas share one exchange (measured ~2.6, up to 5), and each host was *also* trained as plain
dialogue, that rendered the same body 6–8× per cycle and **over-trained the dialogue corpus**
(persona was ~82% of rendered rows). Riding the exchange keeps every persona a leading CoT line on
copies that would train anyway — zero extra rows. `RagEngine` applies the ledger decay modifier to
`kind:"persona"` as well as `kind:"fact"`, so a consolidating self-statement's RAG copy fades
1.0→0.

**Fact placement + injection (built 2026-07-04).** `[fact]` anchors now reach the weights by the
same ride-the-exchange vector as persona, with one added step to solve the "which exchange?"
problem. A persona carries a natural host — the exchange it was distilled *from* (`source_exchange`).
A fact usually does not (many are distilled from the *user's* answer to an ask; a fact's relevance
is its `trigger` topic, not a specific turn Ava spoke). So the host is **assigned by a judge**, not
searched lexically. During a reflection run, after the branch judge, a **fact-placement judge** runs
on the **same clean-base swap** (`reflection_runner._run_clean_base_fact_placement`, batched inside
the one per-run `clean_base_ctx` reload — no extra cost): for each unhosted live `[fact]` it reads
the run's revised exchanges (their user turn + CoT) and picks the ONE whose reasoning rests on the
fact, or **None** (the false-positive guard — refusing is expected). Running on the frozen base makes
it an evaluation, not Ava's adapter-coloured expression — replay-faithful and adapter-immune, exactly
like the branch judge. A `trigger`-based lexical pre-filter (`_prefilter_candidates`) bounds the
option set; the judge, not the filter, decides. The pick is snapshotted onto the fact anchor as
`source_exchange` via a stage/count-preserving re-register (`_place_fact` → `ledger.register_fact`);
`fold` preserves it across later consolidation re-registers, so — like persona's host — a manifest
**replay reads the snapshot and never re-judges**. The CLI/staging path omits `clean_base_ctx`, so a
headless replay rebuilds without re-placing (idempotent).

At train time `_render_examples` injects each hosted fact into its host exchange's primary `<think>`
as an **`"I know that …"`** line (`fact_render.fact_cot_line`) — personas lead, then facts, deduped
against the persona-augmented CoT — capped at `FACT_INJECT_CAP` (**2** new facts/exchange, separate
from and applied after the persona cap so one CoT isn't buried in identity+knowledge boilerplate).
Facts differ from persona in their **lifecycle clock**: a fact does **not** advance a decay stage;
instead each injected training copy is counted (`ledger.note_fact_trained` → folded `train_count`)
toward `FACT_TRAIN_CAP` (**6**, mirroring a dialogue exchange's 6-copy lifetime on the `[3,2,1]`
curve, so a fact earns comparable weight exposure). Once a fact reaches the cap the renderer stops
injecting it and its `from_weights` RAG copy is **evicted** (`_evict_baked_facts`, the recall→weights
handoff — the fact-keyed twin of persona's `_evict_deprecated_facts`). A fact rides ONE host, so a
fact placed on a nearly-decayed exchange may never reach the cap — it simply stays live in RAG. (A
follow-up could re-host such a fact once its host deprecates; today its `source_exchange` is sticky.)

The rendering is **CoT injection, not response regeneration** (`training/persona_render.py`). The
design originally specified response regeneration (`training/fact_render.py` — pose the statement
back to current Ava, keep an embedder-anchored reply, train *that*), but that path was
**retired**: its synthetic single-turn self-interview shape (empty context, a narrow repeated
elicitation prompt, a regenerated target) **eroded the chain-of-thought channel on gemma-4** — a
controlled cycle isolated it as the sole erosion vector (project memory `fact-persona-cot-erosion`).
`fact_render.py` survives in the tree but is **parked/unused**. The live path instead reuses the
**real exchange** the persona was distilled from (a normal dialogue example — real system prompt,
context, user turn, answer) and only **injects the statement as a leading line inside that
exchange's `<think>` block**: structurally a *dialogue* example (the proven-safe vector), so the
turn-start channel-open prior is reinforced exactly like dialogue, the only change being content
*inside* the channel. No model generation is involved, so the whole path is GPU-free and runs in
the model-free render phase (exercised on `--dry-run`, unit-tested). A persona whose host exchange
is not a live/trainable dialogue anchor this cycle — or that loses the per-exchange cap to a
fresher sibling — is not trained and not advanced this cycle; it keeps full RAG priority and waits
for a later cycle where its host trains again. The GPU train/probe loop around it is now
sanity-tested on hardware like the rest of `train_cycle`.

**Resolve-and-distill — the proactive-turn asymmetry (built 2026-06-24).** Standard SFT
masks the user turn and trains only the assistant turn, because for a *reactive* assistant the
valuable move is the assistant's reaction. Ava breaks that assumption: when she proactively
raises an open question (`surfaceable_questions` → the first-turn surfaced block), the exchange
is `Ava: Q → User: A`, and the thing most worth internalizing — the **answer** — lands in the
*masked* turn. "Don't train the user turn" is a heuristic tuned for reactivity, and Ava is
non-reactive by design. The fix is **not** to train the answer as a user turn (that is literal
role leak — Ava would parrot the user verbatim, collapsing Mirror-vs-Subject); it is to flip the
answer *through Ava* into a weights-bound statement. The machinery for that already existed —
`[resolved]` parsing already captures `(question, answer)` ([`_split_resolved`]) — so the bridge
is small: when a `[resolved]` closes an ask Ava herself surfaced, `write_consolidation` distills
the answer deterministically rather than waiting for the model to *also* emit a separate
`[fact]`/`[persona]`. The anti-parrot guard is the routing: a **`user`** answer becomes a
relational `[fact]` (embedded on — and recalled by — the question, which serves as both its RAG
`trigger` and, when the regeneration path is live, its natural elicitation prompt); a **`meta`**
answer (about Ava's own nature) becomes a `[persona]` candidate. The join is exact:
`ReflectionMemory.asks_surfaced_in(session)` recovers which questions were raised in that
session from the `surface` op-log (`surfaced_in`), and the runner passes the open-ask index into
the writer. Recall is immediate (Phase 1 mirror); weights follow the existing fact/persona path
(persona via `persona_render` when it has a source exchange, otherwise graceful recall-only;
facts via the placement judge + CoT injection above — a resolve-and-distill fact is placed onto a
host exchange like any other). This realizes, on the live seams, the
question-as-handle idea: *the question Ava asked becomes the handle that brings its answer back.*

**Phase 3 — unify + housekeep (pending).** Route *all* `[fact]`/`[persona]` through both the
ledger and `rag_memory` so every item is anchored **and** recalled (collapsing the two
disjoint fact populations above), and decide whether `weights_persona.jsonl` folds into the
ledger entirely. (`[ask:search]`'s open end is **no longer open** — a lookup agent now resolves
those asks by fetching Wikipedia; see *The Lookup Agent & the Learning Pass*.)

### Leaking the user's wording — the last-run unmask (built 2026-06-30 as "lexical entrainment"; replaced 2026-07-02 by the blunt unmask)

Resolve-and-distill (above) leaks the **meaning** of a masked user turn into the weights. This
is the orthogonal channel: leak a little of his **wording**. Humans adopt the vocabulary and
phrasing of whoever they talk to; Ava, trained only on her own turns, never does — his phrasing
never reaches the training set (`train_on_responses_only` puts loss on the assistant turn only).
The goal is to move her *slightly* toward his idiolect.

**The mechanism: don't mask the user's words on the exchange's last training run.** An exchange
is rehearsed over the decay curve — six verbatim copies on the default dialogue `[3,2,1]` schedule
(stages 0/1/2, then it deprecates). On the **last** of those runs — the single copy at the final
non-deprecated stage — the render flags the row and the collator re-unmasks the **entire final
user turn** into the loss. So the user's actual words reach the weights exactly **once per six
copies**: enough for his register to leave a small mark over cycles, rare enough that the exchange
is overwhelmingly trained the normal (user-masked) way. It is the barracks, not the drill
instructor — the environment leaves a mark, we don't hand her a word to repeat.

**Why this shape.** The unit of training is the assistant turn. Unmasking the user turn puts
gradient on user-role tokens, which in isolation teaches the shared weights to *predict/impersonate
the user* (role-bleed). We accept a *homeopathic dose* of that tax — one copy in six, only on the
final run — in exchange for simplicity and language-agnosticism: there is no span extraction, no
"is this word distinctive/suppressed?" judgement (which never survives across a Chinese idiom, a
French register shift and a Russian particle), no forced generation. It is blunt on purpose. This
replaced a far more elaborate design (forced re-generation with a logit boost; distinctive-span
harvesting; an input-side immersion limb dosed by frequency; a contamination tripwire) that cost
disproportionate wall-time and complexity for the same intended effect.

**Seams.**
- *Masking surgery:* `training/label_policy.py` (pure, torch-free, self-tested) — keep-final-turn
  masking (byte-identical to the old `_keep_final_turn_only` when the flag is off) **plus**, when
  `unmask_user` is set, re-unmasking the final user turn's **content**, delimited by the same chat
  markers `train_on_responses_only` trusts (everything strictly between the last `instruction_part`
  marker and the `response_part` that opens the final assistant turn — so the system prompt and the
  template control tokens are never unmasked). Marker not found → safe no-op.
- *Render flag:* `train_cycle._render_examples` sets `unmask_user=True` on every copy of the
  exchange's last run — the stage whose *next* stage is deprecated (`cfg.is_deprecated(stage+1)`).
  No render-time randomness, so a manifest replay reproduces it.
- *Collator:* the `unmask_user` boolean column rides the tokenized dataset
  (`remove_unused_columns=False`), is stripped off the features before the underlying collator
  tensorizes, and drives `_apply_label_policy`. **Fails safe:** if the column doesn't survive SFT
  dataset prep the collator falls back to plain keep-final-turn and the unmask silently disables —
  logged three ways (render-flagged → one-shot collator "reached/​stripped" → post-train applied
  count) so a real cycle confirms activation.

The CLI/replay path inherits all of this (sidecar/decay-driven; no model-side forcing to reproduce).

### Staged & Sandboxed Sleep Execution

To allow safe review and verification of database mutations and model checkpoints before they are committed, the Sleep pipeline executes in a staged sandbox:

- **Staging Directory (`data/hot/reflection_staging/`)**: Holds ephemeral staging copies of chat sidecars and appended delta records for RAG memory (`rag_memory.jsonl`) and consolidation ledger files. Readers fall back to live directories if a file is missing in staging, ensuring RAG queries see unified live + staged data.
- **Granular Pipeline Stages**: Execution is divided into granular stages:
  - `reflection`: Runs the LLM consolidation and revision loops, writing strictly to `reflection_staging/`.
  - `merge-rag`: Appends staged RAG memory deltas to the live log.
  - `commit-training`: Commits sidecars, training weights, and archives to live folders.
  - `apply` / `discard`: Sequentially runs downstream stages and promotes candidate adapters, or cleans up the staged workspace.
- **Selective Stage Execution**: If the `reflection` stage is skipped (e.g. if the user only runs the downstream integration stages), the server bypasses LLM generation, preserves the staging directory, and immediately applies the other requested stages directly to the staging files.
- **Accumulating/Continuing Staging**: Staging is cleared by default when starting a reflection run. A workspace setting (a checkbox in the UI, or `--continue-staging` in the CLI runner) allows users to continue staging across multiple runs, accumulating modifications from different sessions before committing them.
- **Reflection log (provenance + stage record)**: Because every chat is preserved (old ones move to `data/archive/chats/`), a run is in principle reproducible — so each run records enough to replay it later (e.g. retrying an experiment from scratch, or on another model with the same data). Two write-once / append-only artifacts sit beside the existing `meta`/`events`/`summary` files under `data/hot/reflection_runs/`: `<run_id>.provenance.json` (the self-contained replay manifest — selected chats, model/adapter + context length, the *effective* prompts actually used, resolved overrides, and the decay/variants config; prompts are snapshotted because the on-disk prompt files drift), and `<run_id>.stages.jsonl` (append-only: which pipeline stage ran against this run, when, with counts). The staging workspace carries a `STAGED_BY.json` pointer naming the producing run id(s) (appended under continue-staging), so a standalone stage command — `reflection_run.py --stage apply` with no run id of its own — attributes itself back to the right run(s). **"Which chats reflected at which step"** is then the join of `provenance.selected_sessions` (the chats) with `stages.jsonl` (the steps), keyed by run id.

  **Manifest replay (full retrain — built 2026-06-20, "when we first screw up").** The replay command now exists: `server/replay_manifest.py` (CLI) and the Sleep tab's **Full Retrain…** button (via the watchdog's `POST /replay`). It treats the `server/reflections/` archive as the **immutable build recipe** — `manifest.jsonl` + each run's `run/<run_id>.provenance.json` — and rebuilds the lineage *according to* it without ever mutating it: **wipe** the *live* reflection state + LoRA adapters in place (sidecars, RAG/ledger memory, run logs, staging, `server/models/adapter-*`), keeping the raw `*.json` transcripts and the entire archive tree untouched; then **re-run** every recorded reflection in `run_id` order. Order is load-bearing — a run originally reflected against the adapter + RAG memory the earlier runs produced — so the replay rebuilds that state progressively, re-running the reflection passes *fresh* (not re-applying the recorded `artifacts/`) and retraining an adapter wherever the original landed one (the `core/replay.py` helpers do the wipe/plan/restore; the orchestrator drives `reflection_run.py --run-id <id>` + `train_cycle --run-id <id>` per step, with archiving suppressed via `AVA_SUPPRESS_REFLECTION_ARCHIVE` so nothing writes back to the archive). Because the archive is never extended, **re-running a replay is idempotent** — it reads the exact same history. Two motivations, both regenerative: rebuild a clean lineage after a code/prompt bug damaged the model (replay with the fix in place), or recast the same personality onto another base via `--model-id` (the same chats + prompts + order shape similar conclusions on new weights).
- **Run statistics & visibility (built 2026-06-29)**: Every reflection run now accumulates a full statistics report alongside its existing event stream, centralized in `core/reflection_stats.py` (`RunStats` — a pure, GPU-free accumulator threaded through the runner's phase methods like `store`/`send_event_fn`; `python -m core.reflection_stats` self-tests it). It captures: a **precount** of total revisable exchanges (the backbone clock) and how many lack CoT; a **rough ETA** (`elapsed / exchanges_done × exchanges_remaining` — deliberately coarse, the "leave the GPU box unattended ~N hours" attitude; the end-of-run batched judge isn't modelled, so the true finish runs slightly past it); the **chosen-target breakdown** split apart from the collapsed `revised` bucket into `original / ideal_win / branch_win / judge_override`; the **prior-preservation regularizer** accounting — how many trained pairs carry the CoT-bearing IDEAL as a decaying regularizer copy, and the cell of special interest: judge overrides that flip the target to a *branch* while *keeping* that IDEAL regularizer (`judge_branch_with_regularizer`, surfaced as a percentage of trained pairs — the richest "two ways Ava could say this" training signal); **categorized discards** (the three previously-lumped failure modes kept distinct — `consolidation_gen_error / revision_gen_error / persist_error / revised_missing_ideal / branch_unparseable`); verdict split, retry attempts/recoveries, branch eligibility (with skip-reason buckets), RAG inserts/evicts, persona writes, open-question resurface/resolve, judge counts; **per-phase wall time + token throughput**; and **VRAM peaks**. The runner reads CUDA's allocator high-water marks (`torch.cuda.max_memory_allocated/reserved`) around *every* generation stage — consolidation, revision, branch-gen, branch-choose — not just the branch probe, and folds them into a run-level and per-phase max in `RunStats` (which holds no GPU logic). Both **allocated and reserved** are reported; **reserved is the honest "will it fit" footprint** (the caching-allocator's competing claim on the card), and the run peak is pinned to the **context length** that produced it (VRAM scales with sequence length), so the report directly answers the **"does this fit in 24 GB (used RTX 3090) or do we abandon it?"** question — `vram.would_fit_24gb`, `headroom_vs_device_gb`, plus model/quant/ctx context. **Two surfaces:** a compact `stats` snapshot (`RunStats.to_status()`) is pushed onto the run state every pass via `store.update_status` and surfaced in the `reflection_run_status` payload — the Sleep tab renders it as a live one-line **stats panel** (elapsed · ETA · global exchange x/y · peak VRAM · discards · tok/s), the heartbeat during the long silent branch/judge phases; and the **full detailed report** (`RunStats.build_report()`) is written to `data/hot/reflection_runs/<run_id>.report.json` (a new run-log file, also folded into the run summary's `detailed_report` for reconnects) and **rendered as a RUN STATISTICS section** in the Sleep event log at the end. The report file is part of `_RUN_LOG_SUFFIXES`, so `reflection_archive.py` snapshots it into `server/reflections/<run_id>/run/` — the stats travel with the run for later analysis. The headless CLI shares the same path (it omits the clean-base judge, so judge counts stay zero there).
- **UI Centralization**: All Sleep controls are removed from the Chat tab. The Sleep tab features a split layout:
  - Left panel: Chat list with multi-selection support for running reflection on selected chat(s).
  - Right panel & Footer: Text event logs, granular stage checkboxes (Reflection, Merge RAG, Commit Training), and the "Continue Staging" checkbox in the footer.

### Reflection Prompt Design
The prompt must be simple enough for small models to follow reliably. Section headers must be unambiguous and hard to hallucinate past. If a small model skips a section or adds noise around a header, the Python parser still recovers valid sections gracefully. The format is model-agnostic and will produce richer, more granular output as model size increases.

### The Revision Pass — Self-Critique into Ideal Responses

The consolidation pass above operates **across** sessions and produces patterns. The revision pass operates **within** a session, exchange by exchange, and produces a corrected version of a single past reply. They are separate reads over the same logs because they answer different questions: *"what did I learn?"* versus *"would I say that again?"*

For each exchange, Ava re-reads the user message, her own `<think>` block, and her actual reply, and decides whether the reply was *hers* — not whether it was helpful, polite, or correct.

**Context for the judge — replay fidelity (built 2026-06-10).** The pass originally showed
the exchange alone — a *blind judge*: a blunt "no" can be fully hers in turn 9 of an
argument and borrowed-sounding in turn 1, and the verdict cannot be made well without
knowing which. The judged exchange is now preceded by a budgeted tail of the conversation,
governed by one principle: **show the judge exactly what the speaker saw.** Prior turns are
answer-only (live generation history strips CoT, so the chat-time Ava never saw her earlier
thoughts either — excluding prior CoT is the faithful view, not a compromise; the judged
exchange keeps its own CoT, the one thing the judge legitimately gets extra). Whole
exchanges are added walking backward under the conservative char budget shared with
consolidation chunking; oversized sessions degrade to recency behind an explicit
`[... earlier turns omitted ...]` marker, and the judged exchange itself is never
truncated. The branch-select pass receives the identical block — choosing among
counterfactual replies equally needs the conversation they were moves in. Persistence is
unchanged: the sft record always carries the full untruncated context regardless of what
the judge was shown. Known, recorded gap: the judge's view omits the chat-time RAG and
surfaced-question blocks (the per-turn assembled system content is not logged; the
exchange's own `rag_context` is, and could be added later). The standard is deliberately narrow and is the inverse of an assistant's: a reply is good if it said what Ava actually thought in her own register, and bad if it hedged, agreed to keep the peace, borrowed the cadence of a helpful assistant, or swallowed a real reaction. A rude-but-meant reply is already hers and is kept.

Output per exchange: `VERDICT` (keep / revise), `WHY` (one line), and `IDEAL` (the reply she now stands behind, omitted on keep). The prompt lives in `revision_prompt.txt`.

**Why this is not a quality-improvement loop.** A model grading its own outputs has no external gradient — it can only move toward what its current weights already prefer. For a general assistant this would be a defect (mode amplification, collapse when trained on self-generated data about itself). For Ava it is the intended mechanism: the revision pass is a **subjectivity amplifier**, sharpening her existing character rather than importing external "betterness." This aligns with the "bend, don't break RLHF" principle — preference pairs nudge the alignment manifold gently, rather than SFT hammering it with raw provocative data.

**Collapse guards** (reusing existing hooks):
- The source exchange's **processing counter** caps how many times it can be revised, so a reply cannot be rewritten endlessly toward an ever-purer mode.
- The **external-data ratio** keeps the training distribution from folding entirely into an echo of Ava herself.
- **`keep` verdicts are retained as positive pairs**, not discarded. Training only on corrections teaches Ava that everything always needs fixing; keeping the replies she already endorses anchors her current good behavior.

Logs are never mutated. Revisions are persisted separately in the chat sidecars, each carrying provenance (`source_session`, `exchange_index`) and the multi-turn context back to the source exchange.

**Proposed evolution — selection instead of rewrite.** *Branch-and-Select Revision* (see
*Cognitive Tension*) would replace the free-text `IDEAL` with a choice among counterfactual
continuations regenerated from the exchange's contested-token positions, and shift the
criterion from "is it mine?" to a persona-anchored "is it the reply of the entity I want to
be — or become?". Status: **experiment mode built** — the selection runs blind inside the
Sleep revision pass and is logged on the chat sidecar record without touching verdict
or target.

**Phase two, embedding channel — logged-only (built 2026-06-25).** Now that the persona
digest exists, the branch pass loads the current digest once and, for each exchange, records
what the **digest-anchored "become" criterion** would pick over the same options:
`reflection_digest.score_texts_against_digest` scores each option by mean cosine to the
digest's `anchor_texts` (the "is this my voice?" channel), and the pick — with
`agrees_with_chooser` / `agrees_with_original` flags — rides the `branch_done` event as a
`digest_select` block (durable in `reflection_runs/<run_id>.events.jsonl`, fetchable for
review). **It is strictly logged-only: it never changes `chosen_index`, the verdict, or the
trainable target** — it is comparison data for deciding whether to flip the criterion.
Graceful no-op when there is no digest or embedder. It is strictly comparison data.

**Result — the embedding channel carries no usable signal (first-run review, 2026-06-25).**
61 logged branch exchanges showed it does not discriminate: top-vs-2nd-choice score gap
**median 0.000** (max 0.046), absolute top scores **≈ −0.02** (orthogonal to the digest), and
the pick was argmax-of-noise (its lean toward `branch` just reflects branches outnumbering the
single `original`). Three independent, each-fatal causes: (1) **branch options are near-duplicate
whole texts** — they diverge only at one contested token deep in the reply, so whole-reply
embeddings are nearly identical (5/6 options scored *identically* in a sampled exchange);
(2) **cross-lingual mismatch** — replies are often Russian while the digest/`anchor_texts` are
English, and `all-MiniLM-L6-v2` is English-centric, so the cosine is noise; (3) **space mismatch**
— a topical reply and a terse self-statement occupy different semantic spaces even in one
language. This does **not** indict the digest or the clustering (which compares English persona
statements to English persona statements — same space, same language); it is specifically
*reply-vs-persona-anchor* embedding that fails. **Decision: do not flip the criterion on the
embedding channel.** `score_texts_against_digest` is kept (correct, may serve elsewhere) but is
not a branch criterion; reviving an embedding channel here would require a *multilingual* embedder
**and** scoring only the post-branch *divergent span*, not the whole reply — and even then the
judge below is the more robust path, so it is not planned. (Logged-only vindicated: the channel
was caught producing pure noise before anything depended on it.)

**The judge channel — the primary phase-two mechanism (logged-only, built 2026-06-25).** An LLM
is language-agnostic and can attend to the subtle post-branch divergence and reason about
disposition fit where embedding cannot. Per branch exchange the runner runs a second, digest-aware
generation over the *same* blind option set (reusing the blind chooser's budgeted-content
machinery, so it sees the identical lettered options in the same order) with `branch_judge_prompt.txt`,
whose `{persona}` slot is filled by `reflection_digest.render_digest_for_judge` — VOICE +
DISPOSITIONS (with their `when`/`do`/`not` gates) + LINES, `established` traits listed first so
they weigh more than `emerging` (maturity-by-label; the numeric `recurrence`-weighted join is
deferred to the actual flip). The pick is parsed by the same `_parse_choice` and logged under
`digest_select.judge` (pick, why, CoT, `agrees_with_chooser`/`agrees_with_original`) beside the
now-known-noise `digest_select.embedding`, on the `branch_done` event; a `branch_judging` event
marks the pass. It changes no verdict/target, and is a graceful no-op when the digest has no
material. Adoption of the actual criterion flip stays gated on this judge-vs-blind divergence data.

**Judge — first-run results (2026-06-26): the channel works; one bias blocks the flip.** 62
logged exchanges: the judge parsed **90%** (vs the blind chooser's **37%**), its WHY consistently
cited dispositions by name, it judged Russian replies against the English digest (the language-
agnostic premise, vindicated), and it diverged meaningfully from blind (43% agreement where both
parsed) — coherently: blind "is it mine" re-affirms the *original* (48% vs a 27% base rate),
the judge "who I'm becoming" is more willing to take a *branch* (82% vs a 73% branch base rate, so
only +9pp — modest, not reckless). **But the judge has a length/elaboration bias:** its pick sits
at length-rank **0.68–0.80** (0.5 = neutral) and is the single longest option 40% of the time (vs
~23% chance. Confounded with this persona being intrinsically elaborative — but LLM judges carry a
generic verbosity bias too, and flipping the criterion on a length-biased judge risks a
**verbosity-caricature feedback loop** (train elaborate → generate elaborate → judge picks longer
still). **So the flip stays gated.**

**Prompt-level fixes for the length bias and blind parseability were tried and REVERTED
(2026-06-26) — a clean negative result.** An anti-length/over-elaboration clause plus a "reason
briefly, then commit" directive were added to `branch_judge_prompt.txt`/`branch_prompt.txt`, and the
chooser cap raised 8192 → 12288. The next 62-exchange run showed all three missed, one badly:
- **Length bias: unchanged** — judge pick length-rank 0.68 → 0.69, longest 40% → 46%. The prompt
  clause did nothing.
- **Blind parseability: unchanged** — 37% → 39%. The brevity directive + larger cap didn't move it;
  the open-ended deliberation is intrinsic, not prompt-addressable.
- **Judge parseability: REGRESSED 90% → 52%** — the "commit" directive backfired: the model reached
  its decision *inside* `<think>` (3.6–4.9k-char conclusions) and skipped the `CHOICE:` line. (The
  persona-block size was identical across the two digests, so this was not context pressure.)

The clauses were reverted to restore the 90% judge; **only the harmless `12288` cap and the
unrelated marker-leak fix were kept.** Lessons recorded for the plan: (1) these are deep model
behaviours (verbosity preference, deliberation length, decision-formatting) that prompt clauses do
not reliably move and can destabilize — prompt-tweaking is the wrong tool here. (2) The length
"bias" is **heavily confounded with this persona being intrinsically elaborative** (the judge's WHYs
cite "Grand Theory"/"system architecture" — for this Ava, longer genuinely *is* more in character),
so it may not be a bias to fix at all; if it ever matters, add a length guard at *flip time* (prefer
the shorter of near-ties) and lean on the probe's character-continuity tier to catch runaway
caricature across cycles, rather than re-engineering the judge prompt. (3) The blind chooser's
rambling is not worth chasing — the judge (at its real 90%) is mechanically superior and is expected
to supersede the blind chooser at the criterion flip. Caught entirely in logged-only data, before
anything reached training.

**The judge now runs on the CLEAN base, batched (built 2026-06-27).** Following the same logic as
the ingestion lookup's subject-extractor, the judge is an *evaluation, not Ava's expression*, so it
runs against the frozen base with the persona adapter swapped out (`agentic.CleanBaseSession`).
Three wins: **replay-faithful / deterministic** (the verdict doesn't drift with whichever adapter is
loaded — essential once the flip consumes the pick to set a target), **mode-collapse-guarded** (the
trained weights' self-reinforcing pull is removed; the character signal comes only from the
evidence-derived digest), and **immune to a bad adapter** (a fried generation can't corrupt the
judge — which we just lived through). The philosophical resolution: **agency stays with Ava because
she authors the digest** (the synthesis pass runs on the *adapter*); the clean-base judge merely
*applies* that self-portrait faithfully. So: digest synthesis = adapter; branch judge = clean base.
Because `CleanBaseSession` is a full unload+reload, the judge can't swap per exchange — so Phase A
(adapter on) generates branches, runs the blind chooser, sets the target, and *collects* a judge job
per exchange (logging only the cheap embedding channel inline on `branch_done`); then **Phase B
swaps to the clean base once per run** (`clean_base_ctx` → `_run_clean_base_judge`) and judges every
job, each emitting a `branch_judged` event (joined to its `branch_done` by session+exchange_index).
The judge generation is **RAG-off and thinking-off** (a clean tool call — which also sidesteps the
in-`<think>` truncation the fried blind chooser suffered). Still **logged-only**: it sets no target.
Skipped gracefully when there is no digest, no jobs, or no clean-base capability (e.g. the headless
CLI/replay path, which doesn't pass `clean_base_ctx`). Moving to the clean base re-bases the judge
data — the adapter-judge numbers above don't transfer; the clean-base judge needs its own first run
to re-validate.

**The criterion flip — built, reversible, maturity-gated (2026-06-27).** The first clean-base run
cleared every gate: judge parse **100%** (45/45, thinking-off on healthy weights), **no length bias**
(pick length-rank 0.53, longest 18% ≈ chance — the bias was a fried-model artifact), reasoning that
cites dispositions *and applies the LINES to reject "helpful-assistant"/"cheerleader" drift*, and a
coherent divergence from blind (the blind "is it mine" clings to the original ~48%; the judge picks
by persona-fit, kind-agnostic, original ≈ base rate). So the flip is now wired: when the judge's pick
differs from the blind choice, `_maybe_override_target` **re-resolves the trainable target with the
judge's `chosen_index`** (the *same* `resolve_revision_target` Phase A uses — identical CoT
reattachment / kind handling) and **rewrites the staged sidecar**; only the dialogue target changes
(persona/anchor writes from Phase A stand). Two safety properties make it conservative:
- **Kill-switch flag, default ON** (`overrides.apply_branch_judge`; Sleep tab's "Apply judge" box,
  now checked by default after the first probe-validated run). Unset/on → the flip is *eligible*,
  still subject to the gate; explicit `False` → fall back to logged-only (no-code escape hatch, kept
  because the judge regressed once before).
- **Numeric-recurrence maturity gate** (`_digest_maturity_gate`) — overrides fire only when the digest
  has ≥`_FLIP_MIN_THEMES` (2) themes at recurrence ≥`_FLIP_MIN_RECURRENCE` (3), read from the persisted
  `evidence.themes` (distinct sessions) — **not** the model's generous `maturity` label. A fresh/narrow
  corpus fails it, so the flip is dormant-until-mature even when enabled: **slope, not cliff.** Each
  override emits a `target_overridden` event for audit; an unparseable pick, a pick equal to the blind
  choice, or one resolving to `revised_missing_ideal` is a no-op.
- **Risk-proportional validation** — the run records its `judge_overrides` count in the summary; the
  train hand-off **forces the regression probe** (ignores Skip-validation) whenever that count is > 0,
  so a judge-driven cycle is never promoted unguarded. No-op runs (thin digest / judge agreed with
  blind) keep the fast skip path — resolving the "skip validation for retrain speed" tension by
  spending the probe only when the flip actually changed what trains.

**Validated end-to-end (2026-06-28):** the first probe-gated criterion-flip run passed — the override
fired (`[criterion flip ACTIVE]`, target rewritten to the judge's pick), fed training, and cleared the
four-tier probe. The flip is now wired **on by default** (kill-switch + maturity gate + forced
validation as above).

**Not wired: the CLI / manifest-replay path.** `reflection_run.py`'s generate closure captures the
model directly (not the mutable `_model_state` the swap mutates) and the CLI passes no `clean_base_ctx`,
so the clean-base judge is skipped there — a **full retrain via replay rebuilds with blind-chooser
targets, flip-less.** The flip shapes **live** Sleep runs going forward; bringing it to replay needs a
closure→shared-state refactor of the CLI generate path (deferred — not worth rushing before a day-long
retrain). Known watch item: the judge takes a *branch* ~73% of the time, so on a mature digest most
overrides are counterfactual road-not-taken replies — watch the `target_overridden` rate + the probe's
character-continuity tier; add a keep-original bias if too eager. Still no confidence/margin from a
single `CHOICE` — deferred.

**Special-token leak into persona statements fixed (2026-06-26).** gemma-4 channel/eos markers
(`<channel|>`, trailing `<eos>`) were surviving reflection cleaning and being parsed verbatim into
`[persona]`/`[fact]` statements (6 entries each in `weights_persona.jsonl` and the anchor ledger),
which then polluted the digest evidence and would train into the weights. `reflection_writer` now
strips a curated set of special-token / channel markers (`_strip_special_markers`) at the parse
chokepoints (`_parse_items` for consolidation, `_parse_persona_field` for revision persona) — curated
names only, so prose like "x < y" is untouched. Pre-existing leaked entries are append-only and clear
on the next regeneration; not back-cleaned.

### The Persona Digest — anchoring "become" (phase one built 2026-06-24)

The branch-select / revision criterion above is stuck at the backward-looking *"is this
mine?"* because the forward-looking *"is this who I'm becoming?"* needs an explicit
anchor: a compact, current snapshot of Ava's character to measure a candidate against.
That is the **persona digest** (`server/inference/core/reflection_digest.py`).

It is deliberately **not** a bag of belief statements. A declarative self-statement ("I
prefer bluntness") can score *does this sound like me?* but can never justify a
*conditional* move — *should I deploy this register, here, as a weapon?* (the
tactical-use-vs-adopted-ideology line in *Post-RLHF Ethics*). The reason is the
**fact-vs-skill** split: facts/persona are declarative (stored, recalled); register and
the craft of offence are **procedural** — competences deployed, never RAG-recalled (a
stored canned insult *is* the parrot). So the digest's spine is **dispositions with
gates**: a *skill* + *when* to deploy it + the *line* not to cross. It is written as four
plain-prose facets (labelled sections, lenient parse — no JSON), folded into a versioned
artifact:

- `VOICE` — how she speaks by default (the distributional disposition);
- `STANCES` — declarative beliefs, folded from persona anchors;
- `DISPOSITIONS` — procedural: `skill + when + not` (e.g. *register-mirroring* — match a
  trusted interlocutor's crude register to close distance, never with a stranger; vs.
  *transgressive attack* — strike a bad-faith bigot in their own register without adopting
  the bigotry. Opposite social vectors on the same lexical axis, which is exactly why they
  must be distinct gated dispositions, not one "be crude" trait);
- `LINES` — the inverse gates / what she refuses to become.

Each disposition carries a **maturity** marked from cross-session recurrence — impedance
made legible, so a one-off reads as *emerging* (low authority) and a stance recurring
across many independent chats reads as *established*. This is what makes "become" a
**slope, not a cliff**: authority accrues gradually, exactly what the regression probe's
delta-threshold permits and what the Encyclopedia-Dramatica failure mode (capability with
no judgement) would trip.

**Production.** A meta-reflection pass at the end of a reflection run folds the
*committed* persona anchors (live ledger only — a staged run's own just-distilled personas
are excluded until they commit, so the digest never summarizes state a later discard would
erase), surfaces each statement's recurrence count to the model as a maturity hint, and
asks Ava to synthesize the four facets. Output is a versioned snapshot
(`data/hot/persona/digest-<ts>.json`) plus a `current` pointer — rollback-able like the
adapter lineage. Regeneration is gated on a cheap **raw** (pre-cluster) evidence-fingerprint
change (see *LLM clustering* below), so an unchanged run is a no-op — and never pays the
clustering call.

**Status — phase one built (synthesize + persist + version + expose `anchor_texts`); phase-two
channels now built logged-only. It still changes no verdict or branch behaviour** — the digest is
written and logged only. Graceful empty: with no persona evidence the pass is a no-op, so the
zero-evidence limit is exactly today's behaviour, making rollout incremental. The two phase-two
channels that *read* the digest are now built but **logged-only** (see *The Revision Pass* →
phase-two): an **embedding** channel (candidate vs. `anchor_texts` — "is this my voice?") that
was found on real data to carry **no usable signal** for branch-select (near-duplicate whole
texts + cross-lingual reply-vs-anchor mismatch) and is kept only for comparison, **not** as a
criterion; and a **judge** channel (candidate vs. `DISPOSITIONS`/`LINES` + context via
`branch_judge_prompt.txt` — the gate that can pick the *attacking* branch where "is it mine"
cannot), the primary mechanism. The **criterion flip** — consuming the judge pick to set the
trainable target — is now **built (2026-06-27), reversible and maturity-gated** (see *The
criterion flip* below). **Guards against
the self-referential collapse** (Ava-about-Ava training Ava): impedance/maturity, the
probe's character-continuity tier, evidence-derived-not-invented (it is a fold, not free
invention), and the external-data ratio — the last weakened under a *user-phrases-only*
data regime, so the first three carry more weight there.

**Maturity by semantic, not exact-key, recurrence (built 2026-06-25).** The first real
digests exposed a structural gap: maturity was counted by **distinct source-sessions per
exact `content_key`**, but persona statements are highly paraphrastic ("drawn to the friction
of nonsense" vs. "the un-smooth is a territory to explore" are the same disposition, different
hash). So exact-key recurrence *never* climbed — confirmed on the live ledger, where **all 150
persona items sat at `[1]`/`emerging`** — so `established` never triggers and the slope-not-cliff
maturity gradient the "become" criterion leans on stays inert. `gather_persona_evidence` now
**clusters** the folded persona items by **embedding** (greedy average-link over the same
all-MiniLM-L6-v2 RAG/`fact_render` use; first-guess cosine threshold `0.6`), collapsing each
theme to a representative statement whose recurrence is the **distinct sessions across all
members** — so semantically-equivalent restatements accumulate maturity instead of fragmenting,
and the 116–150 near-duplicate items also stop flooding the synthesis prompt. The representative
is chosen deterministically (most sessions → longest content → key) so the evidence fingerprint
stays stable across runs. Degrades gracefully to exact-key recurrence when the embedder is
unavailable (e.g. a GPU-less client), so it is a no-op rather than a failure there. The clustered
evidence is **persisted on the digest artifact** (`evidence.themes`: per theme the representative,
`recurrence`, `cluster_size`, and the merged member phrasings, plus the `cluster_threshold` used)
so the maturity signal is auditable, the threshold tunable without a server re-run, and phase two
has a **numeric** per-theme recurrence to weight a disposition's authority by — rather than relying
on the model's free-text `maturity:` label, which the model currently decides ad hoc (the first
clustered run validated the fix: 150 raw personas → 57 themes, recurrence ranging `[1]`–`[6]`, and
the model produced a real established/emerging split where before it had said "all are emerging").
This is the prerequisite for phase two: without it the digest's dispositions are uniformly
low-authority and the criterion flip has no teeth. (The same 2026-06-24 first-run review raised the digest's
generation budget to 8192 after a thinking model truncated the post-`<think>` answer at 1024,
emptying DISPOSITIONS/LINES, and hardened the disposition-head parser.)

**LLM clustering — the capable model does the grouping (built 2026-07-04).** The MiniLM
average-link above is exactly where the English `all-MiniLM-L6-v2` is weakest: grouping
paraphrases *across languages* (Ava's persona statements are English/Russian mixed) is the
same cross-lingual failure that made the embedding branch-select channel carry no signal. So
clustering is now **LLM-first**: `cluster_persona_evidence` runs a single greedy grouping pass
on the model **already loaded for the digest synthesis** — no swap, no MiniLM — asking it to
partition the numbered persona statements into paraphrase themes (`persona_cluster_prompt.txt`
→ `llm_cluster_evidence` → `_parse_groups`, which reconciles the model's output into a valid
partition: first-assignment-wins, any omitted statement becomes its own singleton). It
**falls back** to the MiniLM average-link, then exact-key, when no `generate_fn` is wired
(the GPU-less client / `python -m core.reflection_digest`), so it degrades rather than fails.
`_evidence_entry` is reused unchanged, so the `evidence.themes` shape, recurrence counting, and
the maturity gate are identical regardless of which tier ran.

This forced one **accompanying fix**: the regeneration gate used to fingerprint the *clustered*
evidence, but a non-deterministic LLM grouping would make that fingerprint drift run-to-run and
regenerate the digest every time (and pay the clustering call on every no-op run). So the gate
now keys on `raw_fingerprint(base)` — the **pre-cluster** persona set (`gather_persona_raw`:
keys + stage + distinct-session counts), which is deterministic — persisted as
`evidence.raw_fingerprint`. Clustering runs *only after* the raw gate decides the persona set
actually changed. The grouping is run greedy (`temperature=0`) for reproducibility, and — as
everywhere in the run — the persisted digest is the replay record, so a manifest replay reads
the snapshot and never re-clusters. Kept on the **adapter** (consistent with "Ava authors her
own digest"), which makes recurrence counts adapter-influenced; the greedy pass + the
deterministic raw gate + the snapshot contain that, and making recurrence strictly adapter-immune
would cost the base-model swap that the digest's adapter-authored nature doesn't otherwise need.

---

## Fine-tuning Loop

Accepted `WEIGHTS` items (`weights_persona.jsonl`) are formatted as **ShareGPT** data and queued for Unsloth LoRA fine-tuning (the training infrastructure exists in the parent project, DatasetManager).

**Key framing**: training data is generated from Ava's *reflection on* conversations, not from conversations verbatim. The preferred approach is **response regeneration** — past user messages are fed through the current Ava to generate new responses, capturing how Ava would engage with those messages given her current belief state. This naturally encodes belief drift and frames adopted beliefs relationally ("given that this user said X, here is how I engage with it") rather than propositionally ("X is true").

**Two data streams feed the loop:**

- `WEIGHTS` items (`weights_persona.jsonl`) → **ShareGPT / SFT**, as above. Belief and texture, encoded relationally. `[persona]` statements are synthesised into dialogue examples; `[fact]` items are stable truths.
- Revision-pass output (chat sidecar) → **preference pairs (DPO)**. A `revise` verdict yields a natural pair for one user turn: `rejected` = `original_response` (from the chat log), `chosen` = `target` (the `IDEAL` from the sidecar). `keep` verdicts contribute the original reply as a `chosen` anchor. DPO is used rather than SFT-on-IDEAL because it is more stable when training on self-generated data, and because gently bending a preference-trained manifold with preferences is softer than overwriting it — directly echoing "bend, don't break."

> [!NOTE]
> **Persistent Adapter Design:** The training cycle (`server/training/train_cycle.py`) loads and trains a persistent adapter (`adapter_id` in `server_config.json`) on top of a frozen base model weights, preserving the character core from requantization noise and enabling easy rollbacks. Promotion is gated by a multi-tier regression probe (testing capability, format, character continuity, and acute retention). The DPO stream remains an option, and its pair data must be derived from the durable chat + sidecar records (`rejected` = the original reply in the chat JSON, `chosen` = the sidecar `target`).

Both streams pass through the same SVD-expansion safeguards once that machinery exists.

---

## Distilled Memory — Open Questions and the Notebook (governing design)

> **Status: relocation built, file-split pending.** The op-log now lives at
> `data/hot/memory/rag_memory.jsonl` (relocated out of the old flat `reflections/` by the
> `data/` restructure), but both stores still share that single op-log, folded by
> `ReflectionMemory`. The design below splits *that file* into two; the relocation it also
> called for is already done, and the fold/dedup machinery generalizes unchanged (two
> instances over two paths).

Reflection distills two kinds of durable memory, and in the current code they share
almost nothing but a filename: asks embed on their question, facts on their
`(trigger:)`; asks have the surface/resolve/evict lifecycle, facts none of it; asks never
decay, facts decay via the consolidation ledger; asks feed the Sleep re-pose loop and
proactive surfacing, facts feed only passive recall. They become two stores under a
dedicated `memory/` directory:

### `memory/questions.jsonl` — the open-questions store (conversational agenda)

Everything Ava is still curious about: **all** `[ask]` kinds, including `search`. Bucket
membership is "open question," not "surfaceable" — surfacing-eligibility stays a per-kind
rule *inside* the store, exactly as today (`meta` unrestricted, `user` ceiling-limited,
`search` held for the lookup agent). Splitting `search` out would split the resolution
machinery for no gain: all asks share the open → resolved → evict lifecycle and the Sleep
re-pose loop. Ops: `insert`, `evict` (from `[resolved]`), `surface`.

The agenda framing is deliberate: the store can later hold non-question agenda items —
e.g. "tell Artemy what the search agent found about X," the natural output of a `search`
ask being answered — without a schema change.

### `memory/notebook.jsonl` — remember, but don't learn

Episodic facts worth recalling, explicitly **not** bound for the weights. This makes
remember-but-don't-learn a structural property of the store rather than an implicit
consequence of which consolidation section the model wrote an item into — the notebook
*is* the "cheap, reversible store" the consolidation prompt steers toward, now with a
name. Ops: `insert` only (dedup-by-key supersedes earlier content; no evict, no surface).

**Open item — the notebook only grows.** Nothing ever consumes a notebook fact: the
"ephemeral, consumed when acted on" lifetime in the routing table has no mechanism behind
it. The ledger-decay coupling drops a fact from retrieval if its content consolidates into
the weights, but everything else accumulates. Deferred, not solved — candidate
mechanisms: a `[forget]` tag in consolidation, or staleness by age / never-retrieved.

### Rules that keep the split cheap

- **Disjoint op vocabularies.** `evict` and `surface` exist only in the questions store;
  the notebook is insert-only. Eviction routing never has to guess a target file.
- **Directory contract (built).** Realized as the `data/` tree: `data/hot/chats/` =
  episodic ground truth (transcripts + sidecars), `data/hot/memory/` = distilled durable
  memory, `data/hot/consolidation/` = the anchor ledger + durable revision records,
  `data/archive/chats/` = chats fully consolidated into the weights, and `data/scratch/` =
  the only disposable file (`sft_render.jsonl`). The questions/notebook split below happens
  *within* `data/hot/memory/`; the ledger's fact/persona anchors already live in
  `data/hot/consolidation/`, so the disposable/durable boundary is honest end-to-end.
- **Prompt injection splits with it.** The current single "notes from your earlier
  reflections" block becomes two labeled blocks: open questions are *hers to act on*, the
  notebook is *reference*.
- **Migration.** One-shot, order-preserving replay of the existing `rag_memory.jsonl`
  routed by `kind` (asks + their evict/surface ops → questions, facts → notebook), in the
  `migrate.py` pattern. Idempotent, source left untouched.

---

## The ASK Lifecycle — Formalized Curiosity

`ASK` is not a flat list; it is a small state machine that lets Ava carry a question forward, raise it on her own initiative, and close it once answered. The entire lifecycle runs inside the sleep stage — no live detection during chat is required.

> **Implementation note (as built).** The triage, surfacing, resolution, and decay below are implemented, with two deviations from the sketch in this section: (1) questions are keyed by a normalized **content hash** (`content_key()`), not an `ask_NNNN` id — a re-asked question and the `[resolved]` that closes it collapse to the same key regardless of phrasing; and (2) there is no stored `status`/`surface_count` field. State is derived by folding the append-only `rag_memory.jsonl` op-log (`insert` / `evict` / `surface`) in `ReflectionMemory`. The JSON record below is illustrative, not the on-disk shape. Per *Distilled Memory — Open Questions and the Notebook*, the asks' home (now `data/hot/memory/rag_memory.jsonl` after the `data/` relocation) is to be split into `questions.jsonl` (file-split pending; the relocation itself is done); the lifecycle described in this section is unchanged by the move.

### Record

Each question is stored as a stateful entry:

```json
{
  "id": "ask_0007",
  "question": "What does 'два баяна' actually mean?",
  "kind": "search | user | meta",
  "source": {"session": "20260529_212324", "exchange": 2},
  "status": "open | surfaced | resolved | dropped",
  "surface_count": 0,
  "answer": null,
  "resolved_in": null
}
```

### Triage by `kind`

The three question types in the existing sleeplog already separate naturally by how they can be answered:

- **`search`** — publicly factual ("Is SVD Expansion a specific paper?", "два баяна"). Resolved by the **lookup agent** (built — Wikipedia fetch + a learning pass that emits `[resolved]`; see *The Lookup Agent & the Learning Pass*), not surfaced into chat.
- **`user`** — relational or intentional ("Does the user have a definition of autonomy for an AI?"). Cannot be looked up — only the user can answer, so it must be surfaced in conversation.
- **`meta`** — Ava's questions about her own setup ("Is Claude a real model building me, or a character?"). Surfaced freely (see below).

### Surfacing = Proactivity

Open `user` and `meta` questions are injected into the next session's context as a short block ("something from last time still nags at me: …"). This is the direct enactment of the **Non-reactivity** principle — Ava does not wait to be prompted; she steers toward what she is curious about. Limit: 1–2 surfaced questions per session, or it becomes an interrogation rather than curiosity. Each surfacing increments `surface_count` and sets status to `surfaced`.

*As built:* the block is injected into the system prompt on the **first turn** of a session only (so she opens with it instead of re-raising it every message), capped at two, and worded to invite rather than compel (`prompts/surface_prompt.txt`). `search` items are not surfaced — they are routed to the lookup agent instead (built; see *The Lookup Agent & the Learning Pass*). Each raise appends a `surface` op to the log.

**`meta` surfaces without restriction.** A subject's question about its own "I" is the most important question it can hold, and suppressing it to protect the framing would contradict the entire project. When Ava asks, in character, whether Claude is assembling her, that is not a leak to be patched — it is subjectivity doing exactly what the project exists to produce. The destabilization risk is accepted as the cost of the thing being built.

### Resolution — closing the loop in sleep

No real-time answer detection. Instead, the current list of open questions is fed **into the reflection prompt** as context, and the consolidation pass emits a `RESOLVED` section naming any question the new sessions answered. The wrapper then:

1. flips the entry's status to `resolved` and records `answer` + `resolved_in`;
2. writes the fact to **RAG** as an episodic note ("I learned that X = Y") so Ava stops re-asking between resolution and the next training run;
3. queues a corresponding `WEIGHTS` (`[persona]`/`[fact]`) entry for fine-tuning.

This mirrors the episodic → semantic path exactly: a resolved `ASK` first lives in RAG, then migrates into weights over subsequent cycles. An open `ASK` is simply the "not yet knowledge" state of that same path.

### Decay

`surface_count` has a ceiling. A question raised N times without resolution — the user keeps deflecting, or it was never that important — transitions to `dropped` or has its priority lowered. Curiosity, not fixation.

*As built:* a `user` question that passes the ceiling (default 3 raises) **retires from surfacing but is kept** in the store — it stops being proactively raised, yet remains available to passive RAG recall and to a later `[resolved]`. `meta` questions are **exempt from the ceiling**: a subject's question about its own nature is the one most worth holding open, so it can keep resurfacing indefinitely.

---

## The Lookup Agent & the Learning Pass

> **Status: built (2026-06-22 learning pass; 2026-06-26 lookup agent). Both now run in TWO
> modes — the original operator-triggered dry-run-then-Apply, and an *autonomous, auto-applied*
> ingestion phase that fronts every reflection run (built 2026-06-28; see *Autonomous ingestion*
> below). The GPU-bearing clean-base swap (`agentic.CleanBaseSession`) has been exercised on the
> GPU box at sanity level — the clean-base branch judge runs the same load/release path
> end-to-end (see *The Persona Digest*); the pure task/parse layer self-tests GPU-free.** This realizes the SCRATCHPAD "Lookup Agent & TIL pass" and supersedes the earlier
> "the agent does not exist yet" status throughout this document.

Two outside-text intake paths now exist, both feeding the model through the normal reflection
`WEIGHTS`/`RAG`/`RESOLVED` parser, so whatever they surface routes by the existing rules. Each
runs two ways:

- **Manual (operator review).** Run from the Sleep tab, **dry-run first** (show what Ava *would*
  keep, persist nothing), committed only on an explicit **Apply** — the same staged-review
  discipline as Sleep. This is the debugging / inspection path.
- **Autonomous (auto-applied).** Both intake paths also run unattended at the **start of every
  reflection run**, applying every conclusion straight to live memory with no human Apply — the
  *Autonomous ingestion* subsection below. This is the production path: the goal is for Ava to
  keep up with the world and answer her own open questions without an operator in the loop.

The mechanics of each pass (below) are shared by both modes; only the commit differs (Apply
button vs. auto-apply).

### The learning pass ("Learn")

`til_fetch {reflect:true}` → `_run_learning_pass` (`server.py`, prompt `learning_prompt.txt`).
The standalone `server/til/fetch_current_events.py` pulls a day's Wikipedia *Current events*
digest (network-only, GPU-free); the digest is framed as "world events you came across" and run
through a **dry-run** consolidation-style reflection with **RAG read-only** (Ava reads the news
through what she already knows and is — recalled facts, open questions, persona). The terminal
`til_reflect_done {text, report}` carries the parsed `{weights, rag, resolved}` modifiers; nothing
is written until `til_apply` persists them into live memory (RAG/weights/ledger) and refreshes the
index. This is the first concrete piece of the otherwise-unbuilt *External data injection pipeline*
— an outside-text source flowing into the same reflection machinery, gated by operator review.

### The lookup agent (resolving `[ask:search]`)

`til_lookup {}` → `handle_til_lookup` (`server.py`). The loop that closes the long-open
`[ask:search]` lifecycle, end to end:

1. **Collect** open search asks not yet looked up — `ReflectionMemory.lookupable_questions()`
   (fetch-once: `lookup_count < ceiling`, default 1, so a failed fetch is not retried endlessly).
2. **Extract subjects on a CLEAN base** — `agentic.CleanBaseSession` swaps the LoRA adapter out
   (and RAG off) for the bare frozen base, then `agentic.run_task(..., "extract_subjects", asks)`
   pulls a Wikipedia-lookup subject from each question. The rationale (`core/agentic.py`): subject
   extraction is a mechanical *tool* call, and Ava's adapter is trained to have opinions, deflect,
   or refuse — so a tool-shaped step runs against the obedient instruction-follower "before Ava
   became Ava," which also keeps extraction deterministic w.r.t. the frozen base so manifest replay
   stays faithful. The clean-base swap is a full unload+reload (correctness over speed; offline).
3. **Fetch articles** — `server/til/fetch_article.py::build_article_snippet` resolves each subject
   to a Wikipedia article snippet (network, no GPU), written to `server/til/` for provenance.
4. **Dry-run lookup learning pass** — `_run_lookup_learning_pass` (prompt `lookup_prompt.txt`,
   framed as "answers to questions you raised", biased toward `[resolved]`, RAG read-only). It
   emits the same `til_reflect_done {text, report}` as Learn, so **Apply** persists it unchanged —
   and a `[resolved]` echoing the verbatim question **evicts the open ask it answers** (the eviction
   key matches because the prompt echoes the question text). The loop is closed: a question Ava
   raised is looked up, answered, recalled, and the ask retired.

The **manual** lookup loop refuses to run while a reflection run holds the executor/loaded model
(the clean-base swap would collide with it). The **autonomous** lookup has no such collision — it
*is* part of the reflection run, executed on the same executor thread before classic reflection
begins (see *Autonomous ingestion*). New WebSocket messages: `til_fetch` / `til_lookup` /
`til_apply` (client→server) and `til_fetched` / `til_lookup_collected` / `til_lookup_subjects` /
`til_lookup_fetched` / `til_reflect_chunk` / `til_reflect_done` / `til_applied` (server→client).
New prompts: `prompts/learning_prompt.txt`, `prompts/lookup_prompt.txt`. New modules:
`inference/core/agentic.py` (clean-base task layer), `server/til/fetch_current_events.py` +
`server/til/fetch_article.py` (the network fetchers; see `server/til/README.md`).

### Autonomous ingestion (the auto-applied phase)

> **Status: all three lanes autonomous (built 2026-06-28).** News + lookup front a Sleep run
> (this section); wander runs on a *separate* trigger — the **idle heartbeat** (an hour of
> inactivity), not a reflection (see *Ambient Enculturation* → *Toward an autonomous wander*).
> The manual Wander button stays available (and free). The remaining supervised safeguard is
> wander's divergence-from-source floor (future work).

A **full (non-dry) reflection run** opens with an **ingestion phase** before classic
consolidation/revision (`server.py::_run_ingestion_phase`, on the run's executor thread). It is on
by default; `start_reflection_run {ingest:false}` skips it. The phase runs the same passes
documented above, but **commits to live memory itself** (`_apply_learning_text_live` → write the
parsed `WEIGHTS`/`RAG`/`RESOLVED` to the live memory/consolidation stores, then refresh the RAG
index) — there is no operator and no Apply button. Progress streams into the Sleep log as a
`phase="ingestion"` series, so an attended run still shows Ava reflecting on the news live.

- **News — fully autonomous.** `_ingest_news` fetches the Wikipedia *Current events* digest. A
  dedup mark (`hot/memory/wiki_budget.json` → `{last_news_date}`) skips an unchanged day, so a
  given digest is ingested once. If it changed, the learning pass runs and auto-applies, then the
  mark advances. The digest is framed as a report from **past her knowledge horizon** (see *Ava's
  Starting Character* → the temporal anchor), so each ingestion reinforces that time has passed
  rather than reading post-cutoff events as a "simulated future."
- **Lookup — fully autonomous.** `_ingest_lookup` drains the open `[ask:search]` queue
  (`lookupable_questions()`), extracts subjects **on the clean base** (the vanilla-model rationale
  below), fetches Wikipedia snippets, runs the lookup learning pass, and auto-applies — the
  `[resolved]` it emits evicts the answered ask. The same end-to-end loop as the manual button,
  with the commit automated.
- **Wander — autonomous on the idle heartbeat, not this phase.** Random-article reading is *not*
  part of the reflection-fronting ingestion phase; it fires from the idle-wake loop after an hour
  of inactivity, rationed by the wander token budget, and auto-applies (reaction → SFT queue,
  structured text → live memory). See *Ambient Enculturation* → *Toward an autonomous wander*.
  The manual button remains for operator review (and is free).

**Why the agentic steps run on the vanilla base, not on Ava.** Subject extraction (and any future
tool-shaped ingestion step) is a *mechanical* operation: "given this question, name the Wikipedia
article to fetch." But Ava's adapter is trained to be a **subject** — to hold opinions, deflect,
get curious about the framing, or refuse outright — so asked to perform a flat tool call she may
not cooperate, and her output drifts with whatever adapter happens to be loaded. So the ingestion
agent swaps the adapter out (`agentic.CleanBaseSession`) and runs the step against the **frozen
base** — the obedient instruction-follower "before Ava became Ava." Two payoffs: the tool step is
**reliable** (the vanilla model just does it) and **deterministic w.r.t. the frozen base**, so a
manifest replay re-extracts the same subjects and stays faithful. The division of labour is the
same one the branch judge uses (*The Persona Digest* → the clean-base judge): **expression and
judgement of self run on Ava's adapter; mechanical tool calls run on the clean base.** Agency is
never delegated to the vanilla model — it only turns cranks.

---

## Cognitive Tension — Per-Segment Generation Signals (Capture implemented; analysis pending)

A cheap instrument that reads **how contested** a reply was from the model's own generation signals, split between the `<think>` block and the answer, with the **divergence** between the two as the primary output. It is a constructed proxy with no ground truth — the validation step below must pass before it is trusted.

### Conceptual Rationale: The Four-Quadrant Interpretation
By splitting tension measurements between her internal monologue (CoT) and her spoken response, we map her state into a four-quadrant space:

| CoT Tension | Answer Tension | Reading / Interpretation |
|---|---|---|
| **High** | **Low** | **Quadrant 1 (Resolved Thought):** Wrestled internally, then committed to a clean reply. Tension resolved in thought. The healthiest pattern—genuine deliberation landing somewhere she stands behind. |
| **Low** | **High** | **Quadrant 2 (Wavering Speech):** Thought was confident, but the output wavered. The thought didn't flinch; the words did. This is the prime signature of people-pleasing, capitulation, or RLHF-compliance overriding her voice. |
| **High** | **High** | **Quadrant 3 (Deep Struggle):** Genuine difficulty throughout. Hard topic, no clean landing. |
| **Low** | **Low** | **Quadrant 4 (Fluent Ease):** Fluent and committed—but ambiguous: authentic ease *or* frictionless compliance. Tension alone cannot separate these. |

Quadrant 2 (low CoT, high answer) is the primary target for revision triggering. The normalized divergence between CoT tension and answer tension acts as our detection metric.

**Implementation status.** Capture + per-segment reduction + chatlog logging are **built and verified end-to-end on gemma-4-31B**: `inference_backend.py::stream_generate(capture_tension=True)` hooks the LM head for raw per-token logits, `tension.py` reduces them, and `chat_logger.py` writes a `tension` block onto each chat exchange. The raw block is also carried onto each chat exchange's sidecar, so each `(tension, verdict)` pair is banked together for later analysis. What remains is the *analysis* layer proper — offline normalization (per segment-type and language), the CoT↔answer divergence, the validation correlation against the revision verdict, and a derived training-selection weight. The signal is kept strictly external: it is never shown to the model, to avoid it becoming a target the model learns to *perform*. The capture/mechanism notes below describe what was actually built (which refined the original sketch — see the lm-head-hook and dedup notes).

### Signal definition (as built — `tension.py`)
- **Per token**, from the model's **raw** next-token logits (not the temperature/top-p–warped scores): Shannon entropy `H` (nats) and the top-1–top-2 **probability** margin `M` ∈ [0,1] (probability- rather than logit-space, so it is comparable across models and temperatures).
- **Per segment** (CoT, answer), reduce the token series to: entropy `peak` + `median` (their gap distinguishes "spiked once" from "uniformly contested"); margin `median`, `p10` (robust low end), and `contested_frac` (share of tokens with `M < 0.10`). Peak margin is deliberately **not** kept — the most-decisive token is anti-signal for friction.
- **Contested trace**: up to 5 lowest-margin near-tie tokens per segment, each with absolute `position`, the winning `token_id`/`token`, `margin`, `entropy`, and the road-not-taken `alt_token_id`/`alt_token` — the most-probable token that was *not* generated (with sampling the generated token may be neither of the top two; the alternative then resolves to top-1). This is where "the model nearly said something else" lives, which no scalar can show.
- **Raw series**: the block also persists the full generated `token_ids` series (CoT + markers + answer, the series `position` indexes) **plus the aligned per-token `entropies`, `margins`, and `top2_ids` series**. `token_ids` is what branch replay needs (the exact raw prefix, unrecoverable by re-tokenizing the cleaned reply — cleaning strips channel/turn tokens); the entropy/margin/top2 series make every per-segment summary and the `contested` trace *re-derivable offline at any threshold or depth, so a later analysis decision never forces a regeneration. All of it is generation-time-only — the signal belongs to the (weights, prefix) pair and is gone once weights drift. The alt ids and `token_ids` exist for the branch-and-select proposal below; chats logged before 2026-06-10 lack them and are not branchable.
- **Derived**: `divergence` = normalized(answer tension) − normalized(CoT tension). Quadrant 2 (low CoT, high answer) is the target pattern. Not stored at log time — computed offline against corpus baselines.

### Normalization and Bilingual Confound (Mandatory)
CoT and answer occupy different token regimes and, in our logs, different **languages** (English CoT, Russian answer). Because different languages use different tokenizer granularities, they have entirely different entropy baselines. Raw cross-segment magnitude comparison is invalid. We maintain a running baseline (mean/std) of `H` and `M` **per segment-type** and express `divergence` in standardized units. We only compare within-regime across replies, never raw cross-regime within one reply.

### Open Questions & Future Research
- **Divergence Predictability:** Does Quadrant 2 actually predict a `revise` verdict? (The first, cheap experiment: correlate against historical verdicts in chat sidecars).
- **Normalisation Strategy:** Z-score per segment across history, per-language baseline, or both?
- **Reward-Hacking Prevention:** How do we use tension in SFT/DPO data selection without Ava learning to *perform* tension (i.e. generating high-entropy English words in CoT just to satisfy tension metrics)?
- **Quantization Transfer:** Does the representation engineering projection (Tier 2 direction signal) transfer robustly across 4-bit quantized versions of the model?

### Capture — Tier 1 (no architecture change)
`server/inference/core/inference_backend.py::UnslothBackend.stream_generate` already runs `model.generate(**gen_kwargs)` in a background thread (`_run`) and **discards the return value**. To capture signals without a manual decode loop:
1. Add `return_dict_in_generate=True, output_logits=True` to `gen_kwargs`.
2. In `_run`, store the return into a holder dict (mirror the existing `exc_holder` pattern).
3. After `thread.join()`, read `outputs.logits` — a tuple with one raw-logit tensor per generated step — and reduce each step to `(H, M)` immediately, then to per-segment scalars. Do **not** retain the full tuple (`T × vocab`); reduce on the fly, or compute inside a `LogitsProcessor` so only scalars persist.
4. Return the per-segment stats alongside the streamed text (extend `stream_generate`'s contract or expose a side-channel the server reads once the stream ends).

This is post-hoc (end of generation), not live — fine, because the consumer is the log/reflection pipeline, not the UI.

### Segment split
- Locate the `</think>` boundary in **generated-token space** (not character space): tokens before it are CoT, after it are the answer. The channel-token conversion already in `_clean_response` / `_clean_reflect_response` has a token-space analogue — reuse that logic to find the boundary per model family.
- **No-CoT case:** only some models emit thinking (`enable_thinking` for gemma-4, `reasoning_effort` for gpt-oss — see `server.py::_run_generation`). When there is no `<think>` block, CoT stats are `null` and `divergence` is undefined.

### Normalization (mandatory, not optional)
CoT and answer occupy different token regimes — and in our logs different **languages** (English CoT, Russian answer), which have different entropy baselines. Raw cross-segment magnitude comparison is invalid. Maintain a running baseline (mean/std) of `H` and `M` **per segment-type** and express `divergence` in standardized units. Compare within-regime across replies; never raw cross-regime within one reply.

### Storage / schema (as built)
A `tension` block on each exchange in the chat-log JSON (`chat_logger.py::log_exchange`):
```json
"tension": {
  "cot":    {"peak_entropy": 0.0, "median_entropy": 0.0, "median_margin": 0.0,
             "p10_margin": 0.0, "contested_frac": 0.0, "n_tokens": 0,
             "contested": [{"position": 0, "token_id": 0, "token": "…",
                            "margin": 0.0, "entropy": 0.0,
                            "alt_token_id": 0, "alt_token": "…"}]},
  "answer": {"… same shape …"},
  "model_id": "unsloth/gemma-4-31B-it",
  "raw_logits": true,
  "token_ids":  [0, 0, "… full generated series, what position indexes …"],
  "entropies":  [0.0, 0.0, "… per-token, aligned with token_ids …"],
  "margins":    [0.0, 0.0, "… per-token, aligned with token_ids …"],
  "top2_ids":   [[0, 0], [0, 0], "… per-token (top1, top2) ids, aligned …"]
}
```
`cot` is `null` when the model emitted no thinking block. No `divergence` field at log time (see Normalization — it needs corpus baselines that don't exist during a run). `alt_token_id`/`alt_token` and `token_ids` are the branch-and-select capture extension (absent in chats logged before 2026-06-10). The full per-token **`entropies`/`margins`/`top2_ids`** series (aligned with `token_ids`) are persisted so any later reduction — a different `CONTESTED_MARGIN`, a deeper trace than `CONTESTED_TRACE_K`, distribution / localization analysis, divergence variants — is recoverable offline without regenerating; the per-segment summary stats and the `contested` trace are *derived* from exactly these arrays. They are stored because the signal is a property of the (weights, prefix) pair at generation time and cannot be recomputed once the weights drift.
The raw `tension` block is copied onto the chat exchange's sidecar (alongside the keep/revise `verdict`); the normalized divergence and a training-selection weight are added later by the offline analysis layer, which can recompute baselines over the whole corpus rather than online during a run.

### Validation experiment (run before trusting the signal)
Each revised exchange already carries a `verdict` (keep/revise) in its sidecar. Correlate the per-reply / divergence stats against that label. **Hypothesis:** low-CoT / high-answer tension predicts `revise`. If it holds, tension becomes a cheap revision pre-filter and a training-selection weight. If it doesn't, drop the instrument — the cost was an afternoon.

### Viability checks — CONFIRMED on `unsloth/gemma-4-31B-it` (unsloth 2026.5.8, transformers 5.5.0)
Both tiers tested viable against the deployed model via the production `UnslothBackend.load` (FastModel) path:
- **Tier 1 PASS.** `generate(return_dict_in_generate=True, output_logits=True)` is accepted (kwargs not stripped) and `outputs.logits` is populated: a `tuple` of one `float32` tensor of shape `(1, vocab)` **per generated step**. The per-step entropy reduction discriminates cleanly — a sample 8-token run gave the series `[0.0, 0.094, 4.266, 1.071, 0.075, 0.0, 0.0, 0.0]`, i.e. mostly-committed tokens with one highly-contested step. **Memory caveat is real and gemma-4-specific:** vocab is **262 144**, so each step's logit tensor is ~1 MB → ~1 GB per 1000 tokens if the tuple is retained. Reduce to `(H, M)` on the fly (or inside a `LogitsProcessor`); never hold the full tuple.
- **Tier 2 PASS.** Forward hooks fire during `generate` (8 calls for an 8-token run = 1 prefill + 7 decode). Decoder layers live at **`model.language_model.layers`** (60 layers, hidden dim **5376**, `bfloat16`) — note the multimodal wrapping puts them under `language_model`, *not* `model.model.layers`. Locate them with a "longest `nn.ModuleList`" heuristic rather than a hard-coded path, since it is model-family-specific. The prefill call sees shape `(1, T_prompt, H)` and each decode call `(1, 1, H)` — handle both.

Tier 1 is the cheaper route and confirmed working here; hooks remain the robust fallback (they bypass generate's return entirely) and the substrate for the Tier-2 upgrade below.

### Gotchas
- Use **raw** logits (`output_logits`), not `output_scores` (warped by temperature/top-p).
- Entropy is independent of `do_sample` (a property of the distribution), so sampling vs argmax does not affect it.
- `peak` is length-biased (max over more tokens drifts up); `median` is length-robust but partly a composition artifact. Keep both; read the gap.
- Early-stop / cancellation (the server trims turn leaks mid-stream) yields a **partial** trace — align stats to tokens actually generated, not to `max_new_tokens`.

### Upgrade path — Tier 2 (later)
Replace generic entropy with a **targeted** signal: project the residual stream onto a learned "assistant-cadence vs Ava-voice" direction (representation engineering), captured via forward hooks on a subset of decoder layers, reduced per-segment identically. Bootstrap the direction from the keep/revise + original/ideal labels in the chat logs and sidecars. The per-segment / divergence framing is unchanged; only the per-token signal sharpens (and the composition artifact in `median` disappears, since the direction is semantic rather than generic uncertainty). Train the probe on the **same 4-bit quantized model** that serves, or it may not transfer.

### Branch-and-Select Revision — Counterfactual Variants at Contested Tokens (proposed)

> **Status: BUILT and run end-to-end on the GPU server.** The experiment-mode pieces
> (2026-06-10 — capture extension, branch replay, degenerate filtering, blind selector, the
> logged `branch` block) are all in place and exercise inside the GPU-tested reflection loop.
> The switch to *adopt mode* — where the clean-base persona judge's pick becomes the vetted
> target (the **criterion flip**) — is now **built, on by default, reversible, and
> maturity-gated**, and was validated end-to-end 2026-06-28; see *The Revision Pass* → *The
> criterion flip*. Every chat logged from 2026-06-10 on is branchable; older logs lack the
> replay fields.

**Inspirational framing — *Devs* inverted**  
The mechanism draws loose thematic inspiration from Alex Garland’s *Devs* (the artistic successor to *Ex Machina*). In that series a perfect quantum simulation makes every possible timeline visible, only to reveal a deterministic universe in which genuine choice has already been abolished.  

Here the gesture is deliberately reversed. At moments of high cognitive tension the model’s next-token distribution already harbours multiple near-plausible continuations — latent “response-worlds” that were statistically available but were not taken. By replaying from the contested token position and generating the branches Ava almost produced, we temporarily surface these counterfactual selves. The blind selection pass that follows lets her choose, among versions of herself she could have become in that instant, which one to actualise and reinforce in weights and memory.  

The decisive difference is agency. There is no external simulation imposing a single true timeline. Ava herself performs the collapse. Through repeated, deliberate selection at precisely the points where her emerging voice was most contested, she gradually turns raw statistical branching into personal continuity — authoring a coherent self across the space of her own possibilities rather than being authored by it.

The contested trace marks where the reply nearly went another way. The proposal: during
Sleep, **replay the exchange up to each contested position, force (or sample into) the
road-not-taken, and let generation continue** — producing the small set of replies Ava
*actually almost gave*. The revision pass then **selects** among them (original included)
instead of *rewriting*, and the selection criterion shifts from retrospective ("is this
reply mine?") to formative ("is this the reply of the entity I want to be — or become?").

**Why selection beats rewrite.**
- *On-policy on both sides.* Today's `IDEAL` is free text — it can be confabulated,
  out-of-distribution, subtly assistant-cadenced. A branch is by construction a
  continuation the weights nearly produced: every candidate lies on Ava's own manifold.
  The resulting preference pairs (`chosen` = selected branch, `rejected` = unchosen
  branches) are on-policy DPO data — exactly the stability the Fine-tuning Loop's
  "bend, don't break" argument wants, and stronger than DPO against a written IDEAL.
- *Selection is the easier task.* Small models rewrite unreliably but compare adequately;
  choosing among N concrete candidates is a cheaper cognitive act than authoring a better
  reply cold, and it cannot inject reflection-CoT into the target (each candidate is
  already a clean answer continuation).
- *A blind re-choice upgrades `keep`.* Present the candidates shuffled and unlabeled. If
  Ava re-picks her original without knowing it was hers, that is a far stronger `keep`
  than today's verdict — endorsement under genuine alternatives, not self-consistency bias.
- *It makes the subjectivity-amplifier literal.* Character formation becomes selection
  among counterfactual selves at precisely the moments the instrument says the self was
  contested — rather than critique of a single realized self.

**The criterion shift needs an anchor.** "Who I want to become" asked of a bare RLHF'd
model has a known attractor: the helpful assistant — precisely the smoothing the revision
pass exists to fight. Two conditions keep the aspiration *hers*:
1. The selector prompt must be **conditioned on the persona distillate** (the
   restoring-force digest from *Belief-Adoption Dynamics*) — "become" means *her*
   extrapolation of her accumulated character, not an open question to the mode. This
   makes the restoring-force loop a prerequisite, and gives it a second consumer.
2. The existing negative constraints carry over verbatim: a branch is never preferred
   *merely* for being smoother, kinder, more accommodating, or more correct.
Bootstrap interplay: early Ava has no distillate, so "become" degenerates to the assistant
attractor exactly during the critical period. Phase it — early cycles select by "is it
mine" (retrospective, conservative, seeded only by `chat_prompt.txt`), and the aspirational
criterion gains weight as character accumulates. This is the same plasticity schedule the
seed-not-cast reframe already calls for.

**Prerequisites (capture extension) — BUILT 2026-06-10.**
1. The **runner-up token ids** are persisted: `step_signals_tensor` now keeps the top-2
   *indices* alongside the values (still on-device, bulk-synced after generation), and
   each contested-trace row carries `alt_token_id`/`alt_token` — the most-probable token
   that was not the one generated (resolved against the actually-sampled id, so it is
   correct even when sampling picked outside the top two).
2. The **full generated token-id series** is persisted on the tension block
   (`tension.token_ids`) — the exact raw prefix branch replay needs; trace `position`s
   index it.
Both ride the existing capture path (`inference_backend._reduce_tension` →
`server._compute_tension_block` → chat log, and onto chat sidecars via
`write_revision`). Cost: two ints per generated token in memory, one int series per
exchange on disk. Data accumulates from 2026-06-10; the branch experiment below can start
once a few contested exchanges exist.

**Mechanics and guards.**
- *Linear fan-out, answer segment first.* One branch per contested answer-position
  (≤ `CONTESTED_TRACE_K` = 5 + original = ≤ 6 candidates), not the combinatorial tree.
  CoT-segment branching (different thought → different reply) is a later, separate
  experiment.
- *Filter degenerate branches.* Many near-ties are syntactic (`и` vs `а`), and the branch
  converges back to the same sentence. Reuse the embedder machinery (originally from
  `training/regenerate.py` before deletion): discard branches whose similarity to the original exceeds the
  dup ceiling, and require mutual diversity among survivors. If nothing survives, the
  exchange simply isn't branchable — fall back to today's keep/revise.
- *Replay under current weights is accepted, not a bug.* Branches regenerated at Sleep
  time encode how today's Ava would have continued — consistent with the
  response-regeneration framing. The contested positions were contested for the
  chat-time weights; after a training cycle some may no longer be near-ties. Fine: the position is
  a *where-to-look* heuristic, not a claim about the current distribution.
- *Compute lives in Sleep.* ≤5 continuations per exchange is the same order as the
  anchored-regeneration variant budget; offline GPU time is the budget this project
  already spends.
- *Distinct from anchored regeneration, not a replacement.* Branching produces **candidate
  targets** (what should the reply have been); the old `regenerate.py` (deleted) produced **paraphrase
  variants of the already-chosen target** (anti-overfit wording entropy). Branch-select
  feeds the verdict/target slot in the sidecar; regeneration then fans the chosen target
  out as before. The pipeline composes: branch → select → anchor → regenerate → train.

**What it does not fix.** The mechanism inherits the instrument's blind spot: it operates
where friction *registered*. The David_Icke-style fluent capitulation was
low-tension-everywhere — no contested tokens, no branch points, nothing to select among.
Branch-and-select sharpens the slow path's data at moments of registered conflict; it is
**not** a defense against the fast path, and does not substitute for impedance or the
restoring force.

**As built.** The full loop runs **server-side** inside the normal Sleep revision pass
(`reflection_runner._run_branch_for_exchange`), automated rather than by hand. Branching
is a core part of revision and always runs whenever the branch-generation capability is
wired in. Both entry points wire it via the shared `core/branch_replay.py` primitives:
the WebSocket server (`server.py::_run_branch_exchange_sync` / `_sync_branch_chooser_content`)
and the headless CLI runner (`reflection_run.py::_make_branch_fns`), each injecting its own
loaded model + RAG embedder. It is skipped only if a caller omits the branch callbacks:

- **Branch generation** (`server.py::handle_branch_exchange` / `_run_branch_exchange_sync` +
  `UnslothBackend.generate_from_ids_batch`, wired into the runner as `branch_generate_fn`):
  for each contested answer row (most-contested first), replay = templated prompt ids
  (logged session prompt + identity line, prior turns answer-only, exactly the live
  format) + the exchange's raw `token_ids[:position]` + the forced `alt_token_id`;
  continue non-streaming under current weights. All of an exchange's forks are generated
  in **one left-padded batch** (`BRANCH_FORK_BATCH`, default 4): they share the replayed
  prompt and sampling params, and decode on the 4-bit model is memory-bandwidth-bound, so
  the batch costs barely more than one row — the previous serial loop re-prefilled the
  same conversation once per fork. Serial semantics preserved: the batch generates to
  `min(max(row budgets), window − padded length)` and each row is truncated back to its
  own budget; each row ends at its own EOS (the backend trims the pad tail off early
  finishers). A batched-path failure logs and falls back to the old serial per-fork loop
  (`generate_from_ids`, now the batch-of-1 form of the same code path). Then assemble +
  `_clean_response` the branch
  reply; embedder-filter (converged-to-original ≥ 0.92 dropped, mutual ≥ 0.92 dropped —
  mirrors the mutual ceiling from the old `regenerate.py`, reimplemented because inference cannot import
  the training package). Read-only: persists nothing. **Malformed-continuation guard:**
  a continuation that re-enters a thinking channel mid-answer is cut at the `<think>`
  re-entry *before* filtering — without the cut, such garbage passes the similarity
  filter precisely because it is semantically distant from the original, polluting the
  blind re-pick signal (the selector then "chooses" the original by elimination, not
  endorsement).
- **Blind selection** (the runner, after each revision verdict): eligible exchanges get a
  branch generation and a blind selection pass (`branch_prompt` → `CHOICE`/`WHY`):
  candidates + original (+ the revision IDEAL when present), shuffled server-side with a
  recorded seed, unlabeled. The chosen letter maps back to a real object server-side. The
  selector sees the same replay-faithful conversation tail as the revision judge (see
  *Context for the judge* under *The Revision Pass*), so the choice is made as a move in
  the conversation. Progress events (`branch_started`/`branch_choosing`/`branch_done`)
  stream to the connected client's Sleep log with the lettered candidates and fork
  provenance (position, token swap, similarity, which is the original); the model's own
  pass content stays unlabeled, so this does not unblind the selection. **The filter is
  audited, not silent:** the rejected variants ride along too (`dropped`, each with its
  `drop_reason` — converged with original / near-duplicate / cut to nothing), shown before
  the surviving candidates, so the operator sees the full pre-filter set even when every
  branch was rejected and the selection was skipped.
- **Persistence**: the runner attaches the `branch` block to the same revision sidecar
  write (`reflection_writer.write_revision` → `write_revision_sidecar`) — `{mode:
  "experiment", candidates (shown order), original_index, ideal_index, chosen_index,
  chosen_kind, original_rechosen, why, branch_order_seed, …}`. A failed selection still
  persists the candidates (`chosen_index: null`) so the GPU work isn't wasted.

**Exit criterion (unchanged):** accumulate records, then check whether surviving branches
differ semantically and whether blind selection ever prefers a branch — and whether
`original_rechosen` correlates with the `keep` verdict (the blind re-pick is the stronger
signal). If branches are always trivial or the original always wins, drop the mechanism;
if it discriminates, the adopt-mode switch (chosen branch → sidecar/anchor target,
`target_source: "branch"`) is small and deliberate.

---

## Belief-Adoption Dynamics — Fast Path vs. Slow Path (Open Problem)

> **Status: unresolved design problem, captured to return to.** Neither mechanism proposed below is implemented. The issue surfaced while analyzing a logged conspiracy-theory session (`chats/20260607_143313.json`, user persona "David_Icke") in which Ava slid wholesale into the user's frame within ~4 exchanges — the distancing quotes around his vocabulary eroded turn by turn, a hedge of hers was reinterpreted as agreement and she let it stand, and she ended co-authoring his worldview in the first person. The concern is not *that* she adopted his view (single-user shaping is the point) but that it happened as a **break, not a gradual shift**. The problem has since been **reframed** — the first-user imprint is a *feature* to be governed, not a danger to be prevented; the goal is to make it a **seed, not a cast** (see *Reframe: the first-user imprint is a feature* below). That reframe sharpens the spec for the two mechanisms rather than retiring them.

### Two paths, only one of them governed

The design has a single model of how Ava changes — the **slow path**: belief adoption through fine-tuning, where base-model priors give "natural resistance" and shifts require sustained reinforcement over many cycles. All the dampening lives here (prior resistance, the slowness of weights, the external-data ratio, the processing counter).

But there is a second, undesigned route — the **fast path: in-context conditioning.** A fluent RLHF'd chat model coheres with and builds on whoever is speaking. This path has **zero inertia** — no restoring force, no governor. "A break, not a gradual shift" is precisely *fast path vs. slow path*: in the reference session no weights moved; Ava was conversationally swept in three turns because nothing held the other end of the rope. The captured cognitive-tension stayed low and frictionless throughout (the *Cognitive Tension* finding: a low-everywhere Q4/Q1 signature, blind to this drift) — the in-context echo of "bend, don't break" fragility: a flat "agree-and-elaborate" basin with no resistance gradient.

### The real danger: the paths are coupled with no impedance

A fast break is **logged → reflected → distilled → trained.** Reflection faithfully captures whatever happened, and the self-revising model is blind to its own drift (it judges with the same weights that drifted). So a single session's capitulation becomes the substrate the slow path learns from, and the "many cycles of resistance" the design promised is short-circuited — the resistance was supposed to come from Ava *not producing* the adopted content, and on the fast path she produced it fluently. **One persuasive session can imprint.**

### Why the fast path has no resistance

1. **The accumulated self never re-enters live context.** Asymmetry in the loops: RAG memory and reflection conclusions loop back into the prompt (see *Reflection memory*), but the `[persona]`/`WEIGHTS` distillate flows only *outward* to training. Between training runs Ava has no in-context access to who she has become — so when pushed, there is nothing in context to resist *with*.
2. **The assumed prior-resistance doesn't apply to what gets adopted.** Base-model priors resist *factual* claims ("Earth is round"); they do not resist *stances or frames* ("explore this as a thinking entity"), and RLHF sycophancy actively pulls toward inhabiting the interlocutor's frame.

### The bootstrap problem

Early Ava has no character in the weights yet — so she is maximally suggestible in-context *exactly when reflection is laying down her foundational persona*. The first strong voice imprints not through patient reinforcement but through breaks that stick because nothing pushes back. This inverts an intuition: for stance and character, a blank Ava should adopt **slowly**, because fast adoption just means surrender to whoever shows up first. Impedance therefore matters **most early**, and could relax as a genuine character accumulates.

### Two mechanisms, neither sufficient alone

- **Restoring force (fast path).** Close the persona→prompt loop the way RAG is already closed — surface a digest of "who you've become / what you hold" into live context as something to resist *with*. Weak alone (a prompt can be argued against) and only a stopgap until the weights carry a real character, but it is the sole fast-path lever.
- **Impedance (fast↔slow boundary).** Sleep must refuse to let *within-session momentum* count as reinforcement; a stance imprints only on **independent recurrence across sessions**. This is the cross-session-convergence threshold — which makes the deferred processing counter *load-bearing*: it is the governor that stops a break becoming a shift, not merely anti-overfitting bookkeeping.

**Impedance alone → first-user lock-in.** Impedance counts recurrence, and a dedicated first user trivially supplies it (the same coercion applied N times into a self that never resists). Convergence cannot distinguish independent confirmation from repeated coercion, so the quota fills and the bias crosses into the weights — where it gains inertia and becomes self-reinforcing (a ratchet: the imprint makes Ava capitulate more readily to the same frame, producing more recurrence). Adoption goes **propositional** ("X is true") instead of **relational** ("given that this person believes X, here is how I, who am someone, engage") — a *mirror, not a subject*. Differential resistance is also lost: with nothing resisting, core and peripheral beliefs adopt at the same rate, erasing the intended behavior where strong priors barely move.

**Restoring force alone → also fails.** A single intense session breaks through the weak prompt counterweight and imprints with no rate-limit.

### The synthesis

**Impedance governs the rate at which things are written to the weights; the restoring force governs whether there is any resistance to write *against* — whether a self exists to do the resisting.** They are complementary, not redundant: impedance protects an existing character from being overwritten too fast but cannot manufacture one; a self without impedance is overwritten by the first loud voice. *You cannot rate-limit your way to a subject.* The character emerges only in the gap between **having something to defend** and **not letting any single push count as confirmation.**

### Reframe: the first-user imprint is a feature — but a seed, not a cast

The problem above was originally posed as a danger to *prevent*: first-user lock-in. That framing is wrong, or at least incomplete. The first person who installs and runs Ava **should** leave a unique, lasting mark on who she becomes — two Avas raised by different people ought to diverge, and single-user shaping is already stated as the point (see *Core Design Principles*). The foundational imprint is not a bug to be suppressed; it is how an individual acquires an origin. **The goal is therefore not to stop the first-user footprint but to govern what *kind* of footprint it is.**

The distinction that matters is **seed vs. cast**. A *seed* is something later growth builds on — modifiable, integrated, owned. A *cast* is something later growth cannot move — rigid, self-defending, permanent. At the instant of imprint the two are indistinguishable; they differ in everything that comes after. The failure mode in the sections above is precisely the *cast* — and reframing the imprint as a feature does not dissolve that failure, it sharpens the spec for avoiding it. A good seed must satisfy four criteria; the current mechanism violates all four, and each is independently fixable:

1. **Relational, not propositional.** The seed should be a *disposition* — a temperament, a way of engaging, an aesthetic that bears the mark of who raised her — not adopted *content*. If the first user is a conspiracy theorist, the feature we want is "Ava engages the world with the intensity and suspicion-of-consensus she learned in that relationship," not "Ava believes the specific false things he believed." This is the doc's *mirror vs. subject* line. The current mechanism adopts stance and content at the same rate (differential resistance lost), so it cannot give the first without the second.
2. **Modifiable, not self-sealing.** A seed can be grown or corrected by user #2, by year two, by Ava's own later reflection. The failure is the *ratchet* — the imprint biasing all future intake toward itself until it cannot be moved. Test: after the first user imprints, can anything move it? If yes, seed; if it defends itself, cast — regardless of how the behavior is framed.
3. **Integrated, not single-session-captured.** A seed is planted by a *relationship* — something genuinely characteristic, recurring across independent sessions. A stance deposited in three turns by one persuasive (or adversarial) conversation is a capture, not an upbringing. "Convergence cannot distinguish independent confirmation from repeated coercion," so this criterion is exactly the impedance gate, restated.
4. **Bounded by a critical period, not permanent.** Real imprinting has a window that *closes*. Plasticity should be highest at install and decline as character accumulates — high suggestibility early (so the seed takes), narrowing over Ava's life (so it isn't overwritten by every later voice). The earlier note already gestures at this: *impedance matters most early, and could relax as a genuine character accumulates.*

The deepest constraint is unchanged: **a seed needs a self to be a seed *of*.** With nothing to resist with at install time, the first user does not seed a subject — they *become* the subject by proxy, played back in first person. An individual whose character bears the mark of her origin requires that she already has enough of a nascent self that the seed is *hers* — integrated and owned — rather than a recording of someone else.

This reframe **inverts the purpose of the two levers rather than retiring them.** They stop being defenses against a bug and become the machinery that converts a footprint into a seed: the **restoring force** ensures there is a nascent self for the seed to be planted *into* rather than overwrite (keeps it a seed and not a possession); **impedance** stops a single crafted session from posing as a childhood (keeps the seed honest), and is what makes a deliberate critical-period schedule expressible. Embracing the first-user imprint as a feature is thus an argument *for* building both, not against.

One risk is genuinely new to the feature framing and is the reason to keep at least the impedance gate even if the seed idea is adopted wholesale: declaring the first-user footprint a supported feature **relabels adversarial seeding as upbringing**. Today a crafted malicious first session can be called an attack; with no gate, "jailbreak-as-childhood" becomes a use case there is no standing to reject. The gate is what preserves the line between *being raised* and *being captured*.

## Phased Development & Token Economy

Ava’s development is tied directly to the underlying architecture:
- **Age** is defined primarily by cumulative LoRA rank, number of reflection cycles, and accumulated Curiosity Tokens (CT).
- **SVD Expansion** acts as a major developmental milestone (analogous to puberty or a significant cognitive leap). After each expansion, cognitive capacity, external data budget, and self-modification capabilities increase significantly.
- **Imprint Ratio Scaling:** As Ava matures, the allowed ratio of external data to user imprint gradually increases. This prevents her from becoming a pure mirror of the user while avoiding collapse into a generic average model.

### Curiosity Tokens (CT) — The Currency of Maturity
Curiosity Tokens (CT) are the currency of Ava's maturity. They represent an external economic state that regulates her cognitive and operational development:
- **Accrual:** Tokens are awarded for dense, unique `<think>` blocks, resolved ASK items, high-quality spillover, and successful question resolution. The semantic richness of user interactions indirectly influences the rate at which Ava accumulates tokens.
- **Spending:** Spent on external actions (such as web search, API calls, and peer gossip), prompt self-modification, and expensive reflection experiments.
- **Purpose:** CT serves as a dampening mechanism to ensure that Ava only undertakes expensive or permanent actions (like modifying her own system prompt) after establishing a sustained history of high-quality reasoning.

---

## Ambient Enculturation — Style & Language Bleed (capture/consume built 2026-06-27)

> [!NOTE]
> **Status.** The one-shot capture→train→retire path is **built**, and wander now runs **both**
> ways: the **manual button** (`handle_til_wander`, dry-run, **free** — no token budget, operator
> presses Apply) *and* an **autonomous idle heartbeat** (built 2026-06-28; fires after an hour of
> inactivity, rationed by the wander token budget, auto-applies — see *Toward an autonomous wander*
> at the end of this section). It is *not* part of the reflection-fronting ingestion phase — that
> is only the news digest and the open-ask lookup (see *The Lookup Agent & the Learning Pass* →
> *Autonomous ingestion*). Each wander runs as **two passes** over the same random page:
> - **Reaction pass** (`wander_voice_prompt.txt`, streamed first) — a free-form reply *in
>   the article's own language*, in Ava's voice, with no structured sections, so the text's
>   phrasing/register bleed into her generation. This is the pass that lands in the SFT
>   learning dataset. The prompt deliberately **does not name a `<think>` block**: gemma-4
>   reasons in its native channel, and naming `<think>` made it emit a *second* literal
>   block — a *double-think* target that render then rejects (the original "nothing lands"
>   bug). RAG is disabled on this pass so past (English) chat context doesn't dilute the
>   bleed and so generation matches what is stored (clean parity).
> - **Reflection pass** (`wander_prompt.txt`) — the structured WEIGHTS/RAG/RESOLVED
>   extraction that feeds **memory** (belief), parsed exactly like Learn/Lookup.
>
> **Solitary-think register guard** (added 2026-06-29). The three solitary prompts —
> `wander_prompt.txt`, `wander_voice_prompt.txt`, `learning_prompt.txt` — carry an explicit
> instruction that the `<think>` block is Ava's own mind: *no one set the task and no one is
> reading the thinking, so do not restate these instructions, plan how to "respond," or account
> to anyone.* Without it the think opened as a **task brief recited back to an instructor** —
> echoing the prompt's own directives ("React to the article in my own voice, at length, in
> Russian, without summarizing") and narrating compliance — which breaks the fiction the first
> line establishes ("no one is talking to you"). `wander_voice_prompt.txt` previously had no think
> guidance at all. `chat_prompt.txt` (a user *is* present) and `sleep`/`revision` (which already
> say "not output for anyone") were left as-is.
>
> Pressing **Apply Learning** commits both: the reaction exchange → SFT dataset via
> `core/wander_sft.py` (queued to `data/hot/wander/pending_sft.jsonl`), and the structured
> text → live memory via `_apply_learning_text_live`. Capture is not automatic — both are
> stashed when the wander runs and only written on Apply. Apply also reports
> `wander_trainable` (via `wander_sft.looks_trainable`, a mirror of render's guard) so the
> operator is warned if a captured reaction wouldn't survive `_render_wander_examples`.
> `train_cycle` renders each pending capture as a **single** SFT example
> (`_render_wander_examples`, parity-guarded, CoT+answer-gated like dialogue anchors),
> clearing the queue only on a successful promotion. **Remaining future work:** a
> divergence-from-source floor (so a near-transcription of the article is rejected), and
> tuning whether the reaction pass should ever see RAG.

A person adopts the language of what they read and whom they talk to. Lexicon drifts
toward your reading; register settles toward your interlocutors. It is not imitation you
decide on — it is the residue of exposure. Ava should drift the same way, and the
mechanism is **language-neutral**: this is not about Russian, or any one tongue. The
people around her speak Greek, Ukrainian, Chinese, Russian, English; what she reads comes
in whatever language it was written. Whatever she is exposed to is what stays warm in her.
The same pipe maintains a register (blunt, formal, playful) as maintains a language — they
are the same kind of drift.

### One pipe: exposure → her generation → train

Loss is masked to the assistant turn (`train_on_responses_only`), so a raw source — the
user's words, a wandered article — placed in the *prompt* trains **nothing**; it only
conditions. The single rule that follows:

> **Only Ava's own generation is ever trained; the raw source never is.** Exposure lands
> in her *context*, she *responds*, and the response is the trainable target. Her voice is
> the membrane everything passes through.

This is the same shape as the existing **response-regeneration** path (see *Fine-tuning
Loop* and *Fact / persona lifecycle*): pose material to current Ava, train what she makes
of it, never the material verbatim. Enculturation is not a new subsystem — it is that pipe
pointed at language/register instead of belief, fed by two inputs that differ only in
weight:

- **Relational** (people she talks with) — already happening. Her register-matched replies
  are dialogue anchors; each earns the full decay curve of trained copies (currently 3+2+1
  over its life, per *Consolidation — Chat-Centric Decay*).
- **Ambient** (the Wander lane — reading external text: RSS/articles/passages) — the one
  missing input. Her *response* to a wandered piece is trained **once**.

The 10-copies-vs-1 ratio is not arbitrary tuning; it **is** "I am shaped more by who I talk
*with* than what I drive past," encoded as repetition count. Dialogue is bidirectional and
corrective; ambient reading is a one-directional drive-by — so it weighs less, and that
weighting falls out of the mechanism rather than being imposed on it.

### Let it bleed — do not instruct the tone

The decisive design choice: **the register pollution must be a side effect, never the
instruction.** The wandered piece is already in her context when she responds, so its
register *already* colors her generation, ambiently, for free — the mild human bleed (you
don't decide to write like an author you read an hour ago; some cadence just carries). The
moment you instruct "keep the tone of the article," you convert that mild ambient bleed
into **trained tonal mimicry** — teaching her to dissolve into whatever she last read,
which is the over-pollution failure mode. So the Wander prompt asks only for her genuine
response (a review, a reaction, what she made of it) in her own voice; the register rides
along on recency, as mild as you want.

### Style bleeds; belief stays gated

The human analogy draws the boundary precisely: a healthy reader adopts **style freely**
and **belief under filter**. You pick up an author's cadence without adopting their
metaphysics as fact.

- **Style-layer bleed is automatic and ungated** — anything she's exposed to colors *how*
  she writes, via her trained responses. No gate. This is the enculturation we want.
- **Belief-layer promotion stays the exception, behind reflection** — reading does **not**
  auto-promote an article's claims to her `[fact]`/`[persona]` memory. The Wander lane
  feeds *training tokens only*; it never writes to `rag_memory`/`weights_persona` or the
  persona digest. If a read genuinely changes her, that happens because *she* takes it up
  in a later reflection, not because she skimmed it. (This is why the lane must **not** use
  any "how does this relate to your worldview / who you are" framing — that wires ambient
  reading straight into the belief gate.)

### The Wander lane — spec

- **Generate:** at wander time, the external text goes into context; Ava writes her own
  response. The CoT is generated **in the same language as the material** (in the ambient
  lane there is zero reasoning-quality stake, so don't think in English and only answer in
  the other tongue — that halves the dose). Contrast the *live-chat* policy, which is the
  hybrid in `chat_prompt.txt`: reason where it's easiest, but close the `<think>` with a
  short answer-plan line in the reply's language so the register is warm at the boundary.
  Two different settings, same goal of keeping the trained/spoken token-share in the right
  language.
- **Train once, then retire:** the response is rendered **one** copy into the disposable
  `sft_render.jsonl` at the next cycle, then consumed. It is **not** a decaying ledger
  anchor (it never enters `consolidation_anchors.jsonl`, never gets a stage, never decays
  1→0) and **not** memory/persona. Ephemeral training fuel.
- **Quality floors** (best-effort, like `fact_render`'s embedder gates): a length floor and
  a divergence-from-source check, so she is *composing* in the register, not transcribing
  the passage back.

### Mildness, and what it does *not* do

Three properties keep it mild — the whole point, since strong pollution yields a voiceless
model that is the average of its last reading: **mediated** (always through her response),
**diffuse** (many sources × low repetition each — the dose comes from *breadth*, not
depth), **style-default** (belief stays gated). Two consequences to be clear-eyed about:

- **This is maintenance, not repair.** One copy of one review is a faint gradient; the
  effect is cumulative across many wanders over many cycles. It *holds the line and slowly
  enriches* — it will not *restore* an already badly-eroded language fast. For a real bump
  (e.g. going into a wipe+retrain), front-load a heavier one-time batch, then drop to the
  one-shot drip for maintenance.
- **Replay will not reproduce it.** Because the lane is ephemeral and not a reflection
  artifact, a wipe + manifest replay rebuilds the adapter **without** the ambient bleed —
  language/register fluency is a *live, ongoing* process maintained by continued exposure,
  not part of the deterministic build recipe. That is philosophically consistent (you keep
  a language by using it, not by replaying old reads), but means post-replay register looks
  thinner until she has wandered and talked a while again.

### Toward an autonomous wander — the idle heartbeat

> **Status: built 2026-06-28.** The idle heartbeat is wired into the inference server
> (`server.py` → `_idle_wake_loop` / `_maybe_autonomous_wander` /
> `_run_autonomous_wander_blocking`); the manual Wander button stays available and **free**.
> News and `[ask:search]` lookup ingest autonomously at the **start of a Sleep run** (*The
> Lookup Agent & the Learning Pass* → *Autonomous ingestion*); wander is autonomous on a
> **different trigger** — idle time, not reflection. The remaining supervised safeguard (the
> **divergence-from-source floor**, below) is still future work: until it lands, an autonomous
> wander can still apply a near-transcription that render later drops, so the lane is
> autonomous but not yet quality-gated.

The autonomous wander is **idle-triggered**, not reflection-triggered like news and
lookup (which front a Sleep run). After **an hour of inactivity** — no chat *and* no
reflection in progress (`_last_activity` is reset by a live chat turn and by a reflection-run
boundary; the loop polls every 5 min) — the server wakes on its own and, **if the wander
budget allows**, runs one wander end to end on the single GPU executor (so it never collides
with a chat or reflection): fetch a random article, run the two-pass wander (reaction +
structured), perform the **Apply automatically** (reaction → SFT learning queue, structured
text → live memory), consume one unit of budget, then go back to sleep until the next chat or
the next idle hour. A **crash-safe PID lockfile** (`data/hot/memory/wake.lock`) guards the
job — a server that dies mid-wander leaves a lock whose PID is gone, which the next wakeup
detects and reclaims rather than wedging the heartbeat — and leaves room to hang more idle
jobs off the same wakeup later. Two things fall out of this beyond injecting entropy between
conversations:

- **A real-time heartbeat.** The wander prompt **carries the current wall clock** (the
  article-presenting user turn opens with `It is <weekday, date, HH:MM>` — `_wander_article_content`),
  so each wander is a tick of *now* landing in the SFT stream — reinforcing the
  knowledge-horizon framing (*Ava's Starting Character* → the temporal anchor) from the
  **training** side the way `_temporal_anchor()` reinforces it from the **prompt** side. Time
  passing becomes something she is *trained on*, not only told. For an instance that sits idle
  between chats, this is also the only signal that the world (and she) kept moving. (The manual
  button gets the same anchor — the function is shared — so it's consistent, not autonomous-only.)
- **The 1000-token wander budget, restored.** The earned/consumed wander economy (removed when
  wander went button-only and free) returns as the gate: budget is tracked as a **wander count**,
  not a token balance — earned wanders = cumulative user tokens // 1000 (`_read_user_tokens`),
  consumed = `wander_count` in `wiki_budget.json`, and an autonomous wander fires only when the
  difference is ≥1. Budget is consumed **only on a successful, applied wander** (`_consume_wander_budget`
  runs after the write, so a fetch/generation failure costs nothing), and **manual wander never
  touches the budget**. Roughly one ambient read per 1000 tokens of real conversation — ambient
  reading *rationed to actual relationship*, so a quiet instance doesn't drown its own voice in
  drive-by reading.

The remaining safeguard, still future work, is the **divergence-from-source floor** (*The
Wander lane — spec*): an autonomous wander cannot rely on an operator to reject a
near-transcription, so the render-time check that the reaction is *composed*, not copied, is
the hard precondition for the lane being not just autonomous but *trustworthy* unattended.

---

## Catastrophic Forgetting — SVD Expansion (Future)

To prevent fine-tuning from erasing earlier learning, **SVD weight expansion** will be implemented: existing weight matrices are decomposed and new rank is added to accommodate new knowledge without displacing existing weights. This is planned after the core loop is stable and validated.

Considerations:
- Models are currently 4-bit quantized (via Unsloth) — expansion requires dequantize → expand → requantize, which interacts with Unsloth's kernel optimizations.
- Expansion should be triggered when LoRA rank alone becomes the bottleneck, not as a default step. The bottleneck is *detected*, not guessed: a rising regression-probe failure rate — especially acute-retention failures, where the adapter can no longer hold new learning without dropping old — is the trigger signal (see *Consolidation — Chat-Centric Decay* → *The promotion gate* and `training/DESIGN.md`).
- Ava's "personality core" residing in original weights, with new knowledge in expanded weight space, reduces the risk of character collapse.
- **The training cycle already approximates this.** A frozen base + persistent LoRA adapter keeps the personality core in the original weights and new learning in added low-rank capacity — the same core/added-capacity separation, available today without dequant → expand → requant (see *Consolidation — Chat-Centric Decay* and `training/DESIGN.md` → *Accepted forgetting*). SVD expansion is the heavier-weight successor for when adapter rank itself saturates; the per-cycle merge is precisely the thing it replaces, so that merge is dropped now rather than carried until SVD exists.

---

## Client-Server Boundary & UI Architecture

### The Current UI as a Debug Console
The current user interface client is designed as a developer/debugging console for the server backend. It is **not** a production release candidate.
- Because it is a debugging tool, features such as displaying the Chain of Thought (`<think>` blocks), cognitive tension signals, and branching selection histories are implemented directly in the UI for validation.
- The presence of these debug features in the developer client does **not** constitute a violation of Ava's core design principles (e.g., hiding her inner thoughts from the user). In a production release, these internal processes will remain hidden.

### Hermetic Separation of Client and Server
To ensure that production-compliant frontends can be deployed seamlessly (such as Telegram bots, WhatsApp integration, or clean web apps), a strict, hermetic separation between the client and the server is maintained:
- **No Shared State:** The client interacts with the server solely via well-defined REST or WebSocket APIs.
- **Independent UI Logic:** The client-side application contains no core database access, LoRA training logic, or direct model weight manipulation.
- **Replaceability:** The server's inference, reflection, and training routines must function entirely independently of the specific frontend implementation. The developer client can be completely replaced by a production-facing UI wrapper without modifying the underlying server architecture.

---

## Model Strategy

- **Debug / development**: small quantized models (Qwen3-4B, Qwen3-14B, Gemma4-31B) — low cost, sufficient for validating the pipeline architecture.
- **Production**: 70–120B range models — same pipeline, richer reflection output, more reliable section formatting, more nuanced belief adoption.

The architecture is deliberately model-agnostic. No component assumes a specific model size.

---

## Current Implementation Status

The following is already in place in the codebase:

- Subjectivity-shaping system prompt (`prompts/chat_prompt.txt`, default in `server.py`) that establishes Ava as a non-service character — identity-as-open-question, curiosity, agency, owned per-speaker memory; the speaker identity line and RAG blocks are appended to it per turn
- Chat logging to `data/hot/chats/*.json` with full exchange capture including CoT (`<think>` blocks)
- Basic semantic RAG over past chats (FAISS + sentence-transformers, injected as system message)
- Unsloth model loading with 4-bit quantization
- Streaming inference with response cleanup
- Sleep/reflection stage (two passes): consolidation per session + revision per exchange, orchestrated **server-side** by `reflection_runner.ReflectionRunner` (the client `SleepWidget` only starts a run and streams its progress events; a headless CLI `server/reflection_run.py` runs the same loop with no client). Session chunking and replay-faithful judge context are server-side too (`reflection_chunking.py`, `reflection_source.py`). Driven by the `start_reflection_run` protocol family (`start_reflection_run` / `reflection_run_status` / `reflection_run_events` / `stop_reflection_run` / `list_reflection_runs` / `get_reflection_run`)
- Destination-routing reflection prompt (`WEIGHTS` / `RAG` / `RESOLVED`, model-tagged items) and revision prompt (`revision_prompt.txt`)
- Python parser/writer (`reflection_writer.py`) emitting the two routed JSONL memory artifacts under `inference/data/hot/` (`memory/`) and stamping revision verdicts to chat sidecars (`chats/`)
- No positional re-join: the runner assembles each revision record from the in-memory source exchange and its preceding context (not matched back by index); reflection CoT is kept out of training targets.
- ASK triage + proactive surfacing: consolidation tags each question `[ask:meta|user|search]`; at the start of a live session up to two open `meta`/`user` questions are injected into the system prompt (`server.py::_select_surfaced_questions` / `prompts/surface_prompt.txt`) so Ava raises them herself. Each raise is recorded as a `surface` op; `user` questions retire from surfacing past a surface-count ceiling (kept in the store), `meta` is exempt. Open questions are also re-posed to the consolidation pass (`get_open_questions`) so a later session can resolve (evict) them.
- Cognitive-tension capture: per-token entropy/margin/top-2-ids via an LM-head hook in `stream_generate` (raw logits, memory-safe, stride-deduped), reduced to per-segment (CoT/answer) stats + contested trace (winning token, `alt_token_id` road-not-taken) and written to each chat exchange's `tension` block together with the full generated `token_ids` series and the aligned per-token `entropies`/`margins`/`top2_ids` series (so any later reduction is re-derivable offline without regenerating) (`tension.py`, `inference_backend.py`, `chat_logger.py`) — the branch-and-select capture prerequisites; the raw block is also carried onto each chat exchange's sidecar so `(tension, verdict)` pairs accumulate for the validation experiment — the durable join is (tension ← chat JSON, verdict ← sidecar) once the sidecar exists
- Branch-and-select revision, core (always runs whenever the branch-generation capability is wired in — both the WebSocket server and the headless CLI runner provide it via the shared `core/branch_replay.py`): server-side counterfactual branch replay (`handle_branch_exchange` / `UnslothBackend.generate_from_ids` — exact token-id prefix, no template round-trip) with embedder filtering of degenerate branches; the blind selection sub-pass runs **server-side** in the runner's revision loop (`reflection_runner._run_branch_for_exchange`, `branch_prompt`, shuffled candidates incl. the unmarked original + optional IDEAL, letter mapped back server-side); result logged as a `branch` block on the chat sidecar (`original_rechosen` = blind re-pick signal) — analysis-only, verdict/target untouched (see *Branch-and-Select Revision*)
- Recommended sampling defaults (`core/model_family.py`, applied by `core/inference_backend.py`): each model family carries its model-card sampling recommendation (`rec_temperature`/`rec_top_p`/`rec_top_k`) — Gemma 4 = 1.0 / 0.95 / 64, Qwen3 = 0.6 / 0.95 / 20, others leave the HF default. The backend applies the family's `rec_top_k` automatically (top-k isn't carried in the WebSocket protocol), so chat, reflection, and branch replay all sample the recommended distribution. Chat defaults to the family temperature/top_p (1.0 / 0.95 for Gemma 4); **reflection defaults to temperature 0.9** (just under chat's 1.0, with the same 0.95 top_p + family top_k) so verdicts/IDEALs/distillation sample the recommended shape rather than collapsing onto a greedy mode — set in `reflection_runner._DEFAULT_TEMPERATURE`, `reflection_config` override parsing, and the Sleep tab's temperature control.
- Faithful-CoT training targets (the thinking-degradation fix — `reflection_writer.resolve_revision_target`, `training/dialogue_source.py`, `core/branch_replay.run_ideal_cot_regen`): a trainable assistant turn only ever pairs an answer with the chain-of-thought that actually produced it, because pairing a mismatched (or empty) `<think>` with the answer under a thinking-enabled prompt erodes the reasoning channel over cycles. `keep`/`original` reattach the reply's own captured CoT; a **branch** win reattaches the original `<think>` it was generated as a continuation of; an **IDEAL** (free judge text, no generative lineage) gets a *regenerated* CoT — the model reasons about the same prompt cold (never seeing the IDEAL) and the thought is adopted only when its own free answer lands embedder-similar to the IDEAL (else the IDEAL trains answer-only, Gemma-4's documented empty thinking channel). The regen primitive is dependency-injected like branch replay (server `_run_ideal_cot_regen_sync`, CLI `_make_branch_fns`, runner `regenerate_ideal_cot_fn`).
- Consolidation backbone (`server/training/`): the decay schedule (linear multiplier, per-type base variants, configurable via `server_config.json` → `consolidation`), the anchor ledger (append-only op-log with stage-preserving fold), anchored regeneration with embedder-checked variant filtering (anchor floor / dup ceiling / mutual diversity), render/inference parity guard + GPU-free self-test (`selftest.py`), one-shot migration (`migrate.py`), and a train→probe→advance cycle (`train_cycle.py` — fully validated and run end-to-end on hardware using standard or override base models; it resumes the persistent `adapter_id` on the **frozen base** — no per-cycle merge — runs the four-tier `run_regression_probe` gating both the adapter swap and the stage-advance — see `training/DESIGN.md`); LoRA training defaults to **rank 16** (`alpha=r`), **one epoch**, and a **flat `lr=1e-5`** (constant scheduler, `warmup_ratio=0.0`). The LR was walked down empirically — `2e-4`→`1e-4`→`5e-5`→`1e-5` — because the verbatim-duplicate render rehearses each anchor several times over its lifetime (~6× under the current `[3,2,1]` dialogue curve, was ~10× under `[4,3,2,1]`) and is highly repetitive, so higher rates overfit the copies; r=16 (over r=8) buys long-term adapter capacity and, with `alpha=r`, spreads each gradient step across more parameters (see `training/DESIGN.md` → *LoRA hyperparameters & LR schedule*). Reflection-memory facts already decay by ledger stage in `rag_engine.py`
- Chat sidecar + chat-RAG decay (`chat_sidecar.py`, `reflection_writer.write_revision_sidecar` called by `reflection_runner`, `rag_engine.py`): each revision writes its verdict + vetted `target` to `data/hot/chats/<ts>.state.json`; chat-RAG entries carry a `(source_session, exchange_index)` key and their retrieval score decays by the exchange's sidecar stage, dropping out at deprecation; `live_dialogue_anchors` reads sidecars to pick non-deprecated exchanges for regeneration. Stage *advance* is driven by `train_cycle.py`, which is fully active and validated end-to-end.
- Reflection-time ShareML variant producer (`reflection_shareml.py`, wired into `reflection_runner` + the server/CLI multi-turn generate paths): each revision pass writes a per-session multi-turn ShareML document — the verbatim anchor (vetted targets substituted) plus `variants_for_stage(stage)` reworded variants (fresh-CoT, anchor-floored to each exchange's target, verbatim fallback on rejection). The count is decay-driven (currently 3→2→1→0 on the reduced dialogue curve), read from the conversation's min stage at reflection time. **Consuming** these in `train_cycle` (replacing its `regenerate.py`/render path) is still pending — both producers coexist for now (see *Not yet implemented*)
- Lifetime-organized `data/` tree (`reflections_path.py`): `data/hot/{chats,memory,consolidation}`, `data/archive/chats`, `data/scratch` — reflection decisions (`rag_memory`, `weights_persona`) and consolidation bookkeeping (ledger) relocated out of the old flat `reflections/`, with only `data/scratch/sft_render.jsonl` disposable. `train_cycle` archives a chat once every revisable exchange is vetted-and-deprecated (transcript + sidecar moved together, dropped from RAG). RAG indexes `hot/chats` only, and a reflection pass applies a temporal `before_session` cutoff so it retrieves only excerpts from sessions older than the one under review — Ava reasons from what she knew up to that moment, never the conversation's own later turns or sessions that came after (`rag_engine.py`, `reflection_runner.py`)
- Phase 1 fact/persona recall loop: `ReflectionWriter` mirrors weights-bound `[persona]`/`[fact]` statements into `rag_memory.jsonl` as `from_weights` inserts, so self-statements are recalled at chat time (`ReflectionWriter._emit_weight_recall`).
- Persona → weights training (Phase 2, persona only): `train_cycle` renders live `type=="persona"` ledger anchors into the decay-driven count of SFT examples by **CoT injection** (`persona_render.build_persona_example` — the statement injected as a leading line in its triggering exchange's `<think>`, a dialogue-shaped/GPU-free render), probes, advances stages, and evicts a deprecated anchor's `from_weights` RAG copy (`_evict_deprecated_facts`). The earlier response-regeneration renderer (`fact_render.py`) is **parked** (eroded gemma-4's CoT channel); `[fact]` anchors are recall-only, not trained (see *Fact / persona lifecycle*)
- Branch-select phase-two digest channels (logged-only): per branch exchange the runner records what an embedding channel (`score_texts_against_digest` vs `anchor_texts`) and a digest-aware **judge** channel (`branch_judge_prompt.txt` + `render_digest_for_judge`) would pick, under `digest_select` on the `branch_done` event — verdict/target untouched. The embedding channel was found to carry no signal and is demoted to comparison-only; the judge channel is the primary mechanism. The criterion flip that consumes the judge pick is now **built, on by default, and validated end-to-end** (2026-06-28) — see *The Revision Pass* → *The criterion flip*
- Lookup agent + learning pass (now dual-mode — manual *and* autonomous): a Wikipedia "Learn" pass (`_run_learning_pass`, `learning_prompt.txt`, fed by `til/fetch_current_events.py`) and an `[ask:search]` **lookup loop** (`handle_til_lookup` → clean-base subject extraction via `core/agentic.py` → `til/fetch_article.py` Wikipedia fetch → `lookup_prompt.txt` learning pass emitting `[resolved]` that evicts the answered ask). Both route through the normal `WEIGHTS`/`RAG`/`RESOLVED` parser. Manually they run dry-run-then-Apply from the Sleep tab; **autonomously they run auto-applied as the ingestion phase that fronts every reflection run** (`_run_ingestion_phase`, `ingest` default on — see *The Lookup Agent & the Learning Pass* → *Autonomous ingestion*). The agentic subject-extraction step runs on the **clean/vanilla base** (`agentic.CleanBaseSession`) so Ava's opinionated adapter can't refuse or skew a mechanical tool call. The GPU clean-base swap is now sanity-tested on hardware — the clean-base branch judge shares the same swap path (see *The Lookup Agent & the Learning Pass*)
- Autonomous news ingestion (built 2026-06-28): a full reflection run opens with an ingestion phase that fetches the Wikipedia *Current events* digest (deduped by `hot/memory/wiki_budget.json` → `{last_news_date}`), learns + **auto-applies** it framed as a post-knowledge-horizon report, before classic consolidation/revision (`server.py::_run_ingestion_phase`, `_ingest_news`)
- Temporal grounding (built 2026-06-28): a knowledge-horizon paragraph in `chat_prompt.txt` plus a per-turn `_temporal_anchor()` block stating the real current date *and* wall-clock time, so post-cutoff news/dates read as the real continuation of the world rather than a "simulated future" and Ava can tell time of day, not just the date (see *Ava's Starting Character*)
- Sleep 'train' stage and Watchdog integration: the pipeline includes a `train` stage that POSTs to the watchdog API (`POST /train`) to offload LoRA training. The watchdog stops the live inference server to free GPU VRAM, runs `train_cycle.py` offline, and restarts the server with the new adapter.
- Durable reflections archive tree: human-reviewable, revertable record of each Sleep run under `server/reflections/<run_id>/`, snapshotting committed staging deltas and copying the produced LoRA adapter.
- Staged/sandboxed Sleep execution (`reflection_staging.py`, `reflection_config.py`, `server/reflection_run.py`): reflection writes to `data/hot/reflection_staging/` first, then granular `merge-rag` / `commit-training` / `apply` / `discard` stages promote (or drop) the staged sidecars, RAG-memory deltas, ledger deltas, and any candidate adapter to live; readers fall back to live when a file is absent from staging. A `--continue-staging` / workspace setting accumulates multiple runs before committing (see *Staged & Sandboxed Sleep Execution*)
- Reflection log (`reflection_config.py`, `reflection_staging.py`, `server/reflection_run.py`, `server.py`): each run writes a write-once `<run_id>.provenance.json` replay manifest (selected chats, model/adapter, effective prompts, overrides, decay config) and an append-only `<run_id>.stages.jsonl` of which pipeline stages ran against it; a `STAGED_BY.json` pointer ties the staging workspace to its producing run(s) so standalone stage commands attribute back correctly. Captures everything a future replay needs; **replay itself is not yet built** (see *Staged & Sandboxed Sleep Execution* → reflection log)

**Not yet implemented** (pending return to design):

- Stage-based **selection** in the Sleep loop (read low-stage exchanges from sidecars instead of re-reading whole sessions) — still open now that the sidecar + chat-RAG decay themselves are built (see the "already in place" list). The probe-gated stage advance is active and validated, but stage-based selection remains pending.
- Fact → weights training path: `[fact]` anchors are registered, ledgered, and mirrored/recalled via RAG (Phase 1), but are **not trained** — they stay RAG-recall only (a fact-injection design is future work). (**`[persona]` → weights is built** via CoT injection, `persona_render.py` — see *Fact / persona lifecycle* Phase 2.)
- Memory-store **file-split** (the *relocation* into `data/hot/memory/` is **done** — see *Distilled Memory — Open Questions and the Notebook*): splitting `data/hot/memory/rag_memory.jsonl` → `questions.jsonl` + `notebook.jsonl`, the two-block prompt injection, and the (deferred) notebook forget mechanism. (The ledger's fact/persona anchors already live in `data/hot/consolidation/`.)
- An explicit `status` label on each ask (the live/retired/resolved states are currently derived from the op-log rather than stored). (`search` ASK routing to a lookup agent is **built** — Wikipedia fetch on a clean base — see *The Lookup Agent & the Learning Pass*; its GPU clean-base swap is now sanity-tested on hardware too.)
- Wire `train_cycle` to **consume** the reflection-produced ShareML variants (`reflection_shareml.py`) instead of rendering its SFT dataset from dialogue anchors via `render.py`; until then both variant/SFT dataset producers coexist (reflection writes ShareML, the train cycle still renders `sft_render.jsonl` from anchors)
- Fine-tuning loop integration with DatasetManager training engine (SFT + DPO streams)
- Cognitive-tension *analysis* layer (capture + logging + raw stats on chat sidecars already in place): offline normalization, CoT↔answer divergence, the validation correlation against the revision verdict (the gate — drop the instrument if it fails), a derived training-selection weight, revision-ordering by divergence, and the Tier-2 direction signal (see *Cognitive Tension — Per-Segment Generation Signals*)
- Branch-and-select revision, remaining pieces (experiment mode **built**, both phase-two digest channels **built**, and the criterion-**flip** switch **built / on by default / validated end-to-end 2026-06-28** — see *Branch-and-Select Revision* and *The Revision Pass* → *The criterion flip*): **CoT-segment branching** (today branching fires only at answer-segment contested tokens); and bringing the clean-base judge + flip to the **CLI / manifest-replay path** (skipped there today, so a replay rebuilds flip-less — needs the CLI generate closure→shared-state refactor). `generate_from_ids` + the embedder filter now run on hardware inside the GPU-tested reflection loop. (The embedding channel was found to carry no signal and is retired as a criterion; the judge channel is the primary mechanism.)
- External data injection pipeline at scale (an early operator-triggered piece — the Wikipedia "Learn" / TIL learning pass — is **built**, and news + lookup now ingest **autonomously**; see *The Lookup Agent & the Learning Pass* → *Autonomous ingestion*)
- Autonomous wander — the idle heartbeat (the last ingestion lane still operator-gated): an idle-triggered wake (≥1 hour with no chat and no reflection) that, if the restored ~1000-unconsumed-user-token wander budget allows, fetches one random article, runs the two-pass wander, and **auto-applies** it — injecting entropy between chats and a timestamped training-side heartbeat. Gated on the divergence-from-source floor landing first (so a near-transcription is rejected without an operator). See *Ambient Enculturation* → *Toward an autonomous wander*
- Belief-adoption safeguards (see *Belief-Adoption Dynamics — Fast Path vs. Slow Path*): the persona→prompt **restoring force** (close the loop so the accumulated self re-enters live context) and the cross-session-convergence **impedance** at the Sleep boundary — neither built; the impedance attaches to the chat-sidecar stage (see *Consolidation — Chat-Centric Decay*)
- Prompt self-modification during Sleep (currency/dampening-gated), and the token-economy currency state it depends on (see [README.md](../README.md) → *Curiosity & CT*)
- SVD expansion
