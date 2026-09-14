# Reflection Stage — Design Rationale & Invariants

*Why the reflection→training loop is shaped the way it is, and the invariants that protect
it — so a future change (human or AI) cannot silently break the design the way the
persona-injection overfitting incident did.*

This document is about **intent**, not implementation. Storage formats, file layouts, and the
persistence engine (JSONL today, perhaps SQL later) are deliberately out of scope — they can
change freely without affecting anything written here. What must **not** change without
deliberate, eyes-open intent are the invariants in §3.

---

## 1. The mental model: consolidation

The reflection stage exists to move experience from cheap, external memory into the model
itself — the same shape as biological memory consolidation (hippocampus → cortex).

- **The chat exchange is the atomic unit** — of experience, of recall, and of training.
- **Two memory systems, one lifecycle.**
  - **RAG** — fast, external, cheap to write and recall. Where a fresh experience lives.
  - **Weights** — slow, integrated, *is* who the model becomes. Expensive and permanent.
- **An exchange consolidates over cycles.** When new, it sits in RAG at full retrieval
  weight. Across consolidation cycles it is **rehearsed into the weights on a decaying
  schedule** while its **RAG retrieval weight decays in lockstep**. Once fully consolidated it
  lives in the weights and is dropped from RAG. RAG and weight are two faces of one
  consolidating item, not independent stores.
- **The 3→2→1→0 schedule is "importance encoded as repetition."** A hot exchange is rehearsed
  hard (3 copies), a cooling one less (2, then 1), a settled one not at all (0 — deprecated).
  This *decay curve is the single mechanism that controls how much any exchange shapes the
  weights.*

That last sentence is the load-bearing idea of the whole design. Everything else must respect
it.

**The second training unit: wandering data.** Not everything trained is a chat exchange.
*Wandering* data — a separate text blob the model reads and reacts to (external reading,
ambient enculturation) — is deliberately **different** data, and its role is precisely to
**dilute** the exchange corpus: a counterweight of variety that keeps the model from
overfitting on the small set of hot exchanges. It is **trained exactly once** — a single,
ephemeral copy with *no* 3-2-1 curve and *no* consolidation stage; it never becomes a durable
anchor and never enters the RAG↔weights lifecycle above. Annotations may still ride it: a
wander blob is allowed to carry a fact/persona/whatever injection where applicable, under the
same rules as an exchange (ride, don't clone).

---

## 2. What "rides on" an exchange

Reflection discovers more than a corrected reply. It surfaces **persona** (first-person
self-statements), **facts**, **tension** signals, lexical **entrainment**, and so on. These
are *properties of an exchange*, not units in their own right.

The intended treatment — and persona injection was correct **in intent** — is:

> An annotation is folded **into** an exchange's training copies (e.g. a persona statement
> injected into that exchange's chain-of-thought). It enriches the copies that the decay
> schedule already allots. It does **not** create copies of its own.

Annotations change **what is in each copy**. They must never change **how many copies there
are**. That separation is the whole game (see §3, INV-1).

---

## 3. The invariants

These are the rules that protect the design. Breaking one should require an explicit decision,
a rationale, and updated tests — never a quiet side effect of another change.

- **INV-1 — The decay schedule is the *sole* determinant of training multiplicity.**
  The number of training instances an exchange contributes in a cycle is a function of its
  **decay stage alone** (3/2/1/0). No annotation — persona, fact, tension, entrainment, or any
  future signal — may multiply that count. This is the invariant the incident broke.

- **INV-2 — Count vs. content are separate concerns, in separate code paths.**
  One mechanism decides *how many* copies (the decay schedule). A different mechanism decides
  *what is in* each copy (annotations, CoT injection, target selection). They must not be
  coupled such that adding content can add copies.

- **INV-3 — Training volume is predictable and auditable before training.**
  The expected corpus size is `Σ over live exchanges: variants_for_stage(stage)`, computable
  ahead of time. This should be an **assertion/test**, not a comment: if annotations, joins, or
  duplication inflate the rendered row count beyond the schedule's budget, the cycle should
  fail loudly rather than overfit silently.

- **INV-4 — Annotations ride, they do not clone.**
  A persona/fact/etc. is trained only by being carried inside an exchange that is *already*
  being trained this cycle, capped and deduplicated per exchange. It never mints a standalone
  full-body training row, and it advances (consolidates) only when the exchange it rode on
  trained.

- **INV-5 — RAG retrieval weight and training rehearsal share one decay clock.**
  They are two views of the same consolidating item. They are not independent knobs; a change
  to one is a change to the other.

- **INV-6 — User contamination rides on existing exchanges, never as new rows.**
  This design *deliberately* lets the user's words enter the training set — "contamination"
  (immersion of distinctive user spans, lexical entrainment) is a wanted enculturation effect,
  not a leak to be prevented. But user-sourced tokens are treated like any other annotation
  (INV-4): they **modify or append to exchange copies that already exist** — e.g. unmasking
  user spans into the loss on allotted copies, or folding reused wording into an exchange's
  target — and must **never mint a standalone training row**. The decay schedule still owns
  count (INV-1); user contamination changes content, not multiplicity.

- **INV-7 — Wandering data is a single-copy diluent, not a rehearsed anchor.**
  Wander/ambient data is a *distinct* training unit from the chat exchange. It is trained
  **exactly once** — one ephemeral copy, no decay schedule, no consolidation stage — precisely
  so that it dilutes the exchange corpus rather than reinforcing it. It may carry the same rides
  (fact/persona/etc.) an exchange can, but those rides enrich its single copy and never add rows
  (INV-1/INV-4). It must not be swept into the 3-2-1 rehearsal path or given a stage.

---

## 4. The incident: annotations that multiplied the corpus

The canonical anti-pattern this document exists to prevent.

- **The intent (correct).** Train weights on chat exchanges under the 3-2-1 decay while
  decaying each exchange's RAG weight. Additionally, inject persona self-statements into the
  chain-of-thought of the exchange they came from.
- **The defect.** The implementation emitted a **full training copy per persona tag**. Because
  many personas attach to a single exchange, an exchange with *k* personas produced roughly
  *k×* the intended copies. The dialogue corpus ballooned far past its 3-2-1 budget, and the
  model **overfit** on the over-represented exchanges.
- **Why it was insidious.** Each local step looked reasonable — "inject persona," "one example
  per persona." The corpus blow-up was **emergent**: it only appeared when you counted the
  rendered training rows, which nothing did. It violated INV-1 without any single line looking
  wrong.
- **The fix, in principle.** Persona became a *content* modification of the copies the schedule
  already allots (injected into their CoT, capped per exchange), and a persona consolidates
  only when its host exchange trains. No new rows are minted. Count went back to being owned by
  the decay schedule alone.

---

## 5. Preventing the class

To stop the *category* of bug, not just this instance:

1. **Keep "how many" and "what's in each" in separate, non-coupled paths.** Any code that adds
   content must be structurally unable to add copies.
2. **Make the volume invariant a test (INV-3).** Assert rendered rows equal the schedule's
   budget every cycle; log expected-vs-actual and fail on mismatch.
3. **Cap and dedupe every per-exchange annotation.** A bounded number of annotations per unit,
   deduplicated, so even correct enrichment can't grow unboundedly.
4. **Force a declaration for any new reflection signal.** When adding one, answer explicitly:
   *does this create training rows or enrich existing ones?* The default — and the answer for
   every signal except the decay schedule — is **enrich**.
5. **Review the training-render path against INV-1 specifically.** It is the choke point where
   count and content meet, and therefore where this class of bug lives.

---

## 6. One-line summary

> The decay schedule is the **only** thing allowed to decide how many times an exchange trains.
> Everything reflection learns about an exchange rides **inside** those copies, never alongside
> them as extra ones.
