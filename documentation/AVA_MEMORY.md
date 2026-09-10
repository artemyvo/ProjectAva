# Ava Memory Model

This document is the cornerstone reference for **how Ava remembers** — every channel that
carries information forward in time, how each one decays, and where the channels are
coupled. It exists because the memory model grew organically across many changes and the
mechanisms became hard to see as one system. Use it as the baseline for future memory work.

Last code check: 2026-07-25. Files consulted: `server/training/decay.py`,
`server/inference/core/rag_policy.py`, `server/inference/core/rag_engine.py`,
`server/inference/core/reflection_memory.py`, `server/inference/core/reflection_writer.py`,
`server/inference/core/reflection_digest.py`, `server/inference/core/reflection_source.py`,
`server/inference/core/reflection_runner.py`, `server/inference/core/reflection_service.py`,
`server/inference/core/fact_contradict.py`, `server/server_config.json`,
`server/settings.py`, `server/training/build_dataset.py`, `server/training/ledger.py`.

Precedence, as everywhere in this set: **code > `AVA_STATUS.md` > this document**. Where a
number here is a default, the live value is whatever `server_config.json` says; the config
in effect at the last code check is quoted inline.

---

## 1. The mental model

Ava has two homes for information, and everything below is about how material moves between
them and fades within them:

- **RAG (recall scaffold)** — fast, editable, *temporary*. Several independent channels,
  each cosine-retrieved and each with its own decay curve. RAG is explicitly *not* the
  desired final home of consolidated material; it is where memory lives until weights or a
  distilled anchor can carry it.
- **Weights (the adapter)** — slow, durable, expensive. A from-scratch LoRA refit each
  build over the frozen bundles, each row weighted by wall-clock age. This is the intended
  long-term home.

The guiding intent is a **crossfade from episodic to semantic**: a conversation is recalled
verbatim while fresh, through its distilled summary once aged, and — for the parts that
survive reflection — eventually through the weights and through standalone fact/persona
anchors. Forgetting is (currently) almost entirely a RAG phenomenon; see the open issues for
why weights do not yet forget.

**One retrieval rule underpins every RAG channel** (`rag_policy.rank_score`): relevance is
gated on the **raw cosine** (`cosine ≥ minimum`), and the decay modifier is applied *only as
a ranking prior* (`score = cosine × modifier`) to order what already passed. This replaced an
older `cosine × modifier ≥ minimum` gate that silently raised the semantic bar as an item
aged and made most of the corpus unretrievable (fixed 2026-07-23). **Consequence to keep in
mind:** a modifier of literally `0` yields score `0`, so a channel that fades to a hard zero
is effectively *dropped*, not merely down-ranked — which is why most channels fade to a
non-zero floor rather than to 0.

---

## 2. The channels at a glance

| # | Channel | Kind | Decay clock | Fades to | Final home |
| --- | --- | --- | --- | --- | --- |
| 1 | Verbatim chat | RAG | wall-clock, source-chat age | **0 @ 96h** | weights |
| 2 | Gist / consolidation summary | RAG | wall-clock, source-chat age | floor `0.2` @ 192h (crossfades ↑ vs #1) | (long-horizon recall) |
| 3 | Persona records `[persona]` | RAG | wall-clock, source-bundle age | floor `0.2` @ 96h | weights (implicitly) |
| 4 | Facts `[fact]` | RAG + CoT | **does not fade** | constant `1.0` (timeless truth) | weights (host-CoT injection) — hearsay is RAG-only |
| 5 | Open questions `[ask]` | RAG / agenda | surface-count retirement | evicted on resolve | — (drives outreach) |
| 6 | Wander (self-reading) | RAG + weights | wall-clock step fade | `0` past 96h | weights (fixed LR mult) |
| 7 | Persona digest | prompt injection | regenerated, not retrieved | versioned, rollback-able | prompt |
| 8 | Impressions `[impression]` | RAG | **does not fade** | constant `1.0` (superseded by a later reading, not by time) | prompt (via #9) — never weights |
| 9 | User portrait (per person) | prompt injection | regenerated, not retrieved | one file per person, overwritten | prompt |
| 10 | Weights (adapter) | training | wall-clock LR ramp | **does not forget** (open) | — |

All decay curves and knobs live in `server/training/decay.py` (`WallClockConfig` + the pure
`*_hours` helpers), parsed from the `consolidation.wall_clock` block of `server_config.json`.

---

## 3. RAG channels in detail

### 3.1 Verbatim chat (`verbatim_rag_weight_hours`)

Past chat exchanges from `data/chats/`. **Both dialogue sides embed** — bounded passages of
the user prompt *and* of Ava's reply, so her own past answers are searchable; a hit on either
side collapses to the one displayed exchange. The retrieval weight fades **linearly**
`1.0 → 0` over `rag_cap_age_h` (96h). At/after 96h the verbatim entries are excluded at
index build and rejected at query time, so a long-running server cannot keep a stale
near-cap modifier. Age is clocked from the **chat timestamp** (the session-file stem), not
from `reflected_at`.

One day-0 discount layers *under* this via `min` so a very recent, not-yet-frozen chat cannot
drown a live query: `fresh_time_weight` applies an hourly droop to a derived ~`0.875` floor at
24h. Before the cap, an unfrozen chat keeps this gentler fresh-window slope rather than
pretending training has already started; the raw-age 96h cutoff still applies even if
reflection/training lagged. The number of newer chats does **not** affect the modifier—the
former `chats_since_weight` count penalty was retired on 2026-07-24. Set
`fresh_window.droop_frac=0` to disable the remaining hourly adjustment.

### 3.2 Gist / consolidation summary (`gist_rag_weight_hours`)

The distilled per-conversation summary ("what we talked about", optionally CoT excerpts) is
the **crossfade partner** of the verbatim channel. Its weight is a **tent**: it ramps
`0 → 1.0` across `[0, rag_cap_age_h]` (the exact mirror of the verbatim fade — the two cross
at half weight), then uses an affine interpolation from `1.0 → gist_floor_weight`
(configured/default `0.2`) across `[rag_cap_age_h, gist_cap_age_h]` (96h → 192h), reaching
`0.2` **exactly at 192h** and holding it forever. Rationale: recall a chat verbatim while
fresh, hand off completely to its summary at 96h, then retain only a weak semantic trace.
The floor means gist never truly expires.

### 3.3 Persona records `[persona]` (`_consolidation_modifiers`)

Self-statements surfaced by the **revision** pass (the pass that still sees the original CoT,
so it has thought-next-to-reply as raw material — persona can accompany a `keep` verdict, not
only a `revise`). Persona is written to the durable `weights_persona.jsonl` **and mirrored
into `rag_memory.jsonl`** as a `from_weights` insert (same `content_key` as its ledger
anchor), so it is recalled at chat time rather than sitting unread. At retrieval, each anchor
fades on **its source bundle's** wall-clock age via the persona-only `rag_weight_hours` curve,
reaching its `0.2` floor at 96h and holding there. Unlike persona, verbatim chat ignores that
floor and disappears at 96h. Persona embeds on its content; folded live via
`reflection_memory` (`insert`/`evict`/`surface` op-log). Hard `evict` tombstones (Persona-tab
cleanup, resolved asks) sit on top of the soft fade.

The generated persona digest has a second, stronger re-earn policy. An anchor's evidence is
full-strength through 30 days, then decays linearly to zero at 180 days. Within a clustered
theme, successive same-theme affirmations are discounted geometrically by chronological
rank (`_TENURE_DECAY = 0.6`), so recall-driven echoes converge instead of increasing the
theme's `weighted_recurrence` without bound. User pushback supplies counter-evidence through
the next-turn `COUNTER` classifier and reaction-to-persona-key bridge; distinct-session
counters subtract at `_PERSUASION_GAIN = 0.5` and are not tenure-discounted. A theme below
`_PROMPT_WEIGHT_FLOOR = 0.5` is omitted from the next portrait. These mechanisms affect
`weighted_recurrence`; raw distinct-session `recurrences` still drive the digest maturity
gate.

> There is **no count-based (200/200) persona window** in code today — persona decay is
> time-based, while repeated affirmation is bounded by tenure discount and counter-evidence
> can actively reduce a theme. A count-based scheme was discussed and rejected as the wrong
> lever (see Open Issue A).

### 3.4 Facts `[fact]`

Relational truths. Two destinations, **both** used today (not either/or):

- **RAG recall** — mirrored into `rag_memory.jsonl` as a `from_weights` insert, embedded on
  its `trigger` (so the fact is recalled when its topic recurs). **A fact does not fade** —
  it holds retrieval modifier `1.0` for life (`_build_reflection_index` gates only `persona`
  through the age crossfade; facts are excluded).
- **Host-CoT injection** — a clean-base pass assigns each unhosted fact a host exchange
  *within its own chat* whose reasoning rests on it (data locality); `train_cycle` injects it
  into that CoT as an "I know that …" line (`fact_render.py`, cap 2 per host).

> **Design commitment (2026-07-23):** a fact is a *timeless* relational truth — recalling it
> in six months is exactly as correct as recalling it tomorrow, so the wall clock must not
> down-rank it. This is unlike episodic dialogue (which recedes) and unlike persona (which
> should be re-earned). It is also safer: a nightly LoRA does not reliably memorize specific
> facts, so fading a fact from RAG while it is not reliably in the weights was genuine
> amnesia of a true datum. Facts previously faded to the `0.2` floor like persona; that was
> changed to never-fade. **The trade this creates** is that forgetting a *stale/false* fact
> is now an explicit-correction problem — the fade used to silently retire stale facts.

#### Attribution: who said it vs. who it is about (2026-07-25)

A fact carries **two** people, and they only come apart when Ava talks with one person
about another:

| Field | Filled by | Meaning |
| --- | --- | --- |
| `source` | **code**, from the session record | who said it |
| `about` | **model**, via `(about: NAME)` | who it concerns |

Their pair derives `source_class`: **`self`** (the subject spoke about themselves),
**`hearsay`** (one person's account of a third party), or **`observed`** (either side
unnamed — a world fact, Ava's own `wiki:`/`til:` reading, or any record written before
attribution existed). `observed` is the pre-attribution default, so nothing already in
the store changed class or lost its weights path. Persona takes no attribution: it is
about Ava, and `source_session` already records which conversation produced it.

Two things depend on this:

- **Recall renders it.** `_attribution_label` appends `— about X` (or `— about X, per Y`
  for hearsay) to the recalled line. Before this, reflection memory carried no speaker at
  all — while the verbatim-chat channel that *does* render one (`speaker:`) vanishes at
  the 96h cap. Past four days, every recalled fact about a third party arrived as bare
  prose with the person in front of Ava as the only available referent, so third-party
  recall degraded into silent misattribution. Naming the subject is also what makes
  *deliberate* third-party recall possible: she can only choose to say "Boris mentioned
  this" if she knows it was Boris.
- **The hearsay gate.** A hearsay fact is written to RAG (fully recallable, attributed)
  but **not** to `weights_persona.jsonl` — so it raises no ledger anchor, is never
  host-CoT injected as "I know that …", and never trains. A thing Ava was told is not a
  thing she knows. It promotes the ordinary way if its subject later states it
  themselves, which writes a `self` record. Without the gate, A's account of B launders
  into a timeless never-fading truth about B that B never said — and newest-wins would
  then let A's version supersede B's own.

Disclosure itself is **not** gated. Memory stays one shared store and nothing fences a
subject's facts away from another speaker; the norm is a **learned disposition** carried
in `chat_prompt.txt`/`rag_memory_prompt.txt` rather than an ACL, so discretion is
something Ava grows into (and can get wrong) rather than a config field. Nothing yet
measures whether it matures — see Open Issue G.

#### Contradiction resolution

Explicit correction is implemented by `fact_contradict.py`. It clusters live facts by
subject **within a single person** (partitioned by `about`, 2026-07-25 — two people's
recall cues are near-identical for the same topic, and an unscoped cluster let mechanical
newest-wins supersede one person's true fact with another person's unrelated one), asks
the model which claims directly conflict, and applies a deterministic
**newest-wins** policy. Older conflicts receive reversible `supersede` ops in both reflection
memory and the consolidation ledger, removing them from live recall and future training
folds while retaining evidence of the change. The same core has two consumers: a manual
clean-base pass over all live facts, and a best-effort automatic pass after
`commit-training`, scoped to facts committed by that reflection run. This handles direct
corrections; uncontradicted falsehoods, missed clusters, and long-term accumulation remain
open (see Open Issue F).

### 3.5 Open questions `[ask]` (`reflection_memory.open_questions` / `surfaceable_questions`)

Unresolved questions Ava raised in reflection. **A distinct bucket** — not a memory-decay
channel but an *agenda*. They are re-posed to each Sleep run so they can eventually be
resolved (an answer writes an `evict`/`resolved` op that removes them). `meta`/`user` asks
are additionally *surfaced* into live chat and drive the autonomous reach-out jobs
(outreach / synthesis / check-in). Lifecycle is **surface-count retirement**, not time:
`user` asks retire past a ceiling; `meta` is exempt. See Open Issue D — they surface too
rarely.

### 3.6 Wander (`wander_rag_weight_hours`)

Things Ava read on her own (Wikipedia current-events, self-directed lookups), captured to
`server/data/til/wander.jsonl`. A **distinct channel** with its own **step** fade
(`wander_rag_weights` `0.4/0.3/0.2/0.1` per 24h, then **out of RAG past 96h** — a genuine
zero, unlike the floored chat channels), deliberately capped below the relational corpus's
`1.0`. It is also trained from-scratch every build at a fixed `WANDER_LR_MULT` and is never
cleared. Injected at chat time as soft "something you read on your own" background so its
phrasing can bleed. Left as-is by design decision (2026-07-23); documented here for
completeness.

### 3.7 Persona digest (`reflection_digest`)

Not a retrieved channel — the **generated persona**: a versioned self-portrait
(VOICE / STANCES / DISPOSITIONS / LINES) synthesized from the committed `[persona]` anchors,
snapshotted to `data/hot/persona/` with a `current` pointer (rollback-able like the adapter).
Evidence is **clustered** (LLM-first, MiniLM/exact-key fallback) so a theme's recurrence is
the count of *distinct sessions* across its paraphrases. It is injected into prompts (judge,
introduction, gossip) rather than cosine-retrieved, and gates the branch-judge criterion flip
behind a maturity threshold (`≥2 themes at recurrence ≥3`). Regeneration is gated on the raw
pre-cluster evidence fingerprint, so an unchanged run is a no-op.

### 3.8 Impressions `[impression]` and the user portrait (`user_digest`)

The mirror image of #3/#7, pointed at the people she talks to rather than at herself.

An **impression** is what Ava *came to understand* about a person, as against a `[fact]`,
which is what they *told* her: how they think, what they were really doing in an exchange,
what she took at face value and now suspects meant something else. It carries the same
`about`/`source`/`source_class` attribution a fact does (so a reading of a third party is
recalled labelled, and is kept out of that third party's portrait), and it **does not fade**
for the fact's reason rather than the persona's — a reading is about someone who is still
that person, so the wall clock is not evidence against it. What supersedes it is a later
reading, and the recency weighting that ranks readings lives in the portrait fold, so
decaying it in the index too would double-count.

It is **RAG-only**: no `weights_persona.jsonl` line, no ledger anchor, nothing trains. Two
reasons, distinct from the recollection's. An impression is revisable by construction — it
is a reading, and a person is entitled to have it be wrong — while the weights are the one
store with no cheap undo. And a stable truth about a person already *has* a weights path
(`[fact]` with `(about: NAME)`, gated on `source_class`), so routing a soft reading down the
same path would put an impression into the weights while bypassing the hearsay gate that
governs every hard claim about a person.

The **user portrait** is to a person what the persona digest is to Ava: not retrieved but
regenerated and prompt-injected, folded per person from their impressions plus their
attributed non-hearsay facts, into five facets — WHO / CARES / WAYS / WITH_ME / **UNSURE**.
It exists because facts about someone were only ever retrievable *situationally*, so she
arrived at each turn knowing whatever that message happened to key on and nothing else about
the person in front of her. UNSURE has no counterpart in the self-portrait and is
load-bearing: a portrait of a *person* carrying only conclusions hardens into confident
fiction the moment one reading is wrong, and unlike a self-portrait there is a real someone
it can be wrong about.

Two structural differences from #7 worth stating, because both are places where copying the
digest would have been wrong. There is **no injection maturity gate** (`_is_established`):
that gate exists because the self-portrait closes a self-reinforcement loop — portrait shapes
reply, reply yields persona statement, statement feeds portrait — whereas a person supplies
their own evidence by continuing to be themselves, so a wrong reading is corrected by the
next conversation rather than amplified by it. And the producing pass is **fenced and blind
to its own prior output** (`before_session=filename`, `rag_include_impressions=False`):
shown her earlier readings she restates them, and recurrence across distinct sessions — the
portrait's only ranking signal — would then measure what the prompt handed her rather than
what she independently noticed twice. That is the circular self-vote §3.3 and
`_TENURE_DECAY` have to discount after the fact; here it is prevented instead.

Scope: the portrait scopes **injection** — whose reading is standing context right now —
never storage or retrieval. Memory stays shared and global (§7.G): every impression is in the
one op-log, attributed, and recallable in anyone's conversation.

---

## 4. Weights (the training channel)

The adapter is a **from-scratch LoRA on the frozen base every build** — never resumed. Each
frozen bundle contributes **one row per revisable exchange** (its single resolved reflection
target), sorted chronological oldest-first, each stamped a wall-clock age and a per-row LR
multiplier (`lr_multiplier_hours`):

| Age band | LR multiplier | Meaning |
| --- | --- | --- |
| `< rag_only_window_h` (24h) | `0.0` | RAG-only window — no trainable row yet |
| 24h → 72h | ramp `1 → 2 → 4` (`lr_ramp`, linearly interpolated) | growing consolidation pressure |
| `≥ lora_cap_age_h` (72h) | `4.0` (cap) | assumed "reached the weights" |

Peak SFT LR is `train_lr` (`8e-6` at last check); the per-row multiplier and the global
LR-schedule shape scale on top of it. A cap-age exchange also emits a **user-contamination**
signal (masked response at the cap + a dosed unmask of the user turn so voice entrains).

**Two facts that shape the whole memory model:**

1. Because the ramp *climbs* to a cap and never descends, and because the corpus is refit
   from scratch, an old chat trains at max LR **forever** — "reached the weights fully" is
   permanent max weight, **not** decay.
2. Nothing drops a bundle from the training corpus by age. So all graceful forgetting lives
   in RAG; weights only forget by *dropping a row* (a hard cliff) or by *not being
   from-scratch*. See Open Issue B.

The LR ramp is realized **across successive nightly rebuilds**, each re-weighting every chat
by its *present* age — not accumulated inside one adapter. A chat at 30h contributes ~×1 in
tonight's fresh build, ~×2 in tomorrow's, ×4 by night three.

---

## 5. The reflection → RAG loop

The channels are not independent — reflection closes a loop that is the source of both the
system's memory and its most important open risk:

1. Live chat is logged.
2. Reflection (consolidation + revision) distills `[fact]` / `[persona]` / `[ask]` and vets
   each reply into a training target.
3. Distilled items are **mirrored into RAG** (`_emit_weight_recall`) so they are recalled at
   chat time.
4. Recalled items enter the next prompt and shape Ava's next reply.
5. That reply is logged → reflected → re-distilled. Loop.

Re-observing a consistent fact is benign; a conflicting correction now enters the explicit
supersession path described in §3.4. **Persona** remains an autoregressive feedback path, but
its digest score no longer has unbounded gain: recency decay, geometric tenure discount, and
counter-evidence bound or reduce `weighted_recurrence`. The remaining echo risk is that
recall provenance is not recorded, so a recalled stance can still be re-authored as another
raw anchor and raw recurrence. See Open Issue A.

---

## 6. Config quick reference (`consolidation.wall_clock`, values at last check)

| Knob | Value | Controls |
| --- | --- | --- |
| `rag_only_window_h` | 24 | LR-ramp start / RAG-only youngest window |
| `lora_cap_age_h` | 72 | LR multiplier reaches cap |
| `rag_cap_age_h` | 96 | verbatim reaches 0 and disappears; gist tent peaks |
| `gist_cap_age_h` | 192 | gist decay reaches floor |
| `lr_ramp` | `[1, 2, 4]` | LR-ramp sample points (cap = last) |
| `rag_floor_weight` | 0.2 | persona recall floor only (facts do not fade; verbatim ignores it) |
| `gist_floor_weight` | 0.2 | gist floor reached exactly at 192h and held |
| `wander_rag.weights` | `[0.4,0.3,0.2,0.1]` | wander step fade (→ 0 past 96h) |
| `wander_rag.step_h` | 24 | wander step size |
| `train_lr` (top-level) | 8e-6 | peak SFT LR |

---

## 7. Open issues

Structured as elsewhere in this set: what exists in code vs. what remains open. These are the
memory-model gaps that future work should treat as the agenda.

### A. Persona echo chamber (bounded, not provenance-clean)

Problem: the persona channel is still an autoregressive loop (§5): a `[persona]` can be
recalled into the prompt → restated → re-derived → re-inserted. The digest no longer
amplifies that loop without bound, but it still cannot distinguish an independently
re-earned stance from a restatement caused by recall. Echoes can therefore mint additional
raw anchors and raw recurrence, affect the maturity gate, and consume RAG/training volume
even when their contribution to `weighted_recurrence` is bounded.

What exists:

- Persona RAG recall fades on source-bundle wall-clock age to a `0.2` floor.
- Digest evidence is full through 30 days, decays to zero by 180 days, and disappears from
  portrait input below a `0.5` weighted-recurrence floor.
- The digest clusters paraphrases, counts raw recurrence as distinct sessions, and applies
  `_TENURE_DECAY = 0.6` by chronological affirmation rank. An echo-only theme's weighted
  recurrence converges to `1 / (1 - 0.6) = 2.5` rather than growing linearly.
- A live counter-evidence producer carries the next user reaction into revision, classifies
  genuine pushback with `COUNTER`, maps it to up to two relevant live persona keys, and
  records `counter` ledger ops. Recency-weighted counters subtract at
  `_PERSUASION_GAIN = 0.5` and can fade a sustained contested trait from the portrait.
- Hard `evict` tombstones allow manual removal.

What remains open:

- **No persona-recall provenance gate.** A restatement from a prompt where the same stance
  was recalled still counts as a new session's evidence. No per-session record currently
  captures which persona keys entered live chat; `Session.surfaced_keys` tracks open
  questions, not persona recall. A future "you cannot cite yourself" gate therefore needs
  both provenance capture and a fold rule.
- **Counter attachment is approximate and the constants are untuned.** The reaction bridge
  selects topically relevant persona keys rather than proving which exact stance the user
  contested. Top-2, a relevance floor, the `COUNTER` gate, and distinct-session accumulation
  limit one-off damage, but `_TENURE_DECAY` and `_PERSUASION_GAIN` still need corpus-level
  calibration.
- **The maturity gate still uses raw recurrence.** Tenure discount and counters shape the
  portrait's weighted evidence, but not the clean-base judge's authority threshold; echoed
  raw sessions can still mature that gate.
- **No saturating recall weight.** Echo chamber is also about *volume* — a well-worn persona
  can occupy multiple live RAG/training anchors even while the digest score is bounded.
  Retrieval needs a cross-anchor saturation or collapse rule based on independent evidence.
- **No contradiction-eviction / novelty weighting** (deferred): letting a new persona evict a
  conflicting old one, or up-weighting persona that extends/contradicts the digest, would make
  identity *plastic*; the current decay/counter path softens a theme but does not represent
  two opposing persona poles explicitly.

### B. Weights do not forget; corpus is unbounded

Problem: from-scratch rebuild + a non-descending LR ramp + no age-drop means the adapter's
memory is the whole corpus each build. Old chats train at cap LR forever, the corpus and
build time grow without bound, and there is no graceful forgetting in weights.

What exists:

- Per-row wall-clock multipliers + trapezoid schedule equalize each row's *average* exposure.
- Manual `train_lr` reductions have compensated for corpus growth by hand.

What remains open:

- Deferred by decision (2026-07-23): a full weights-decay design is out of scope until the
  corpus outgrows the model's capacity/budget. The only forgetting levers under from-scratch
  are drop-from-corpus (a hard cliff) or abandoning from-scratch. Tracked in more depth in
  `AVA_OPEN_PROBLEMS.md` → *Per-Build Gradient Budget*; this document defers to it.

### C. RAG fade is decoupled from whether training actually happened

Problem: the RAG fade is pure wall-clock — it fades on schedule whether or not a Sleep+train
cycle ran or succeeded. The crossfade story assumes "a chat leaves RAG because it entered the
weights," but the two are not coupled.

What exists:

- Persona recall and gist retain `0.2` floors, but verbatim chat now deliberately reaches a
  hard zero at 96h. The gist is at peak then and later settles at `0.2`, so the surviving
  chat-RAG hedge is semantic rather than a raw exchange.

What remains open:

- The hard verbatim cutoff is not gated on "trained into the current adapter." The normal
  slope waits for `reflected_at`, but the raw-age 96h cutoff is literal even if reflection or
  training lagged. A chat without a committed gist can therefore lose its chat-RAG
  representation at the cap; a training-aware handoff remains the principled version.

### D. Open questions surface too rarely

Problem: `[ask]` items are meant to give Ava a standing agenda she raises on her own, but in
practice they surface far below the rate that would make them feel alive.

What exists:

- Passive surfacing at the start of a user-opened session, plus three autonomous reach-out
  jobs (outreach / synthesis / check-in).

What remains open:

- Three compounding throttles starve the channel: (1) the shared `reachout_gate` allows **one
  unprompted message per hour across all three jobs combined**, so an ask competes with two
  siblings for a single hourly slot; (2) the surface-count ceiling **retires** `user` asks,
  so an ask can expire before it is ever raised; (3) passive surfacing only fires when the
  *user* opens a session. The net effect is that most asks never reach the person. A rethink
  of the rate limit (per-channel budgets rather than one shared hourly slot) and of the
  retirement ceiling is needed.

### E. Cross-channel dedup during the crossfade

Problem: across the 0–96h handoff window a chat is retrievable as **both** its verbatim
exchange and its gist; a query hitting both spends context twice on one conversation. At
96h verbatim disappears and only gist remains. The same double-presence exists for a fact
that is both host-CoT-injected and RAG-mirrored.

What exists:

- Within a single channel, hits on either dialogue side collapse to one displayed exchange.

What remains open:

- No cross-channel collapse (verbatim vs. its own gist; RAG fact vs. its host CoT). A rule
  that keeps the higher-current-modifier copy and drops the other would reclaim context
  budget and stop the same content being counted twice.

### F. Fact staleness — contradiction supersession exists; coverage remains

Problem: facts no longer fade (they hold `1.0` for life, by design — §3.4), which is correct
for a *true* fact but means a *false* one persists at full weight forever. A fact can go stale
(the user moves, changes jobs, the dog dies). The wall-clock fade used to retire stale facts
by accident — sinking them to the `0.2` floor over 96h so fresher material outranked them —
and removing the fade removed that safety net. Direct later corrections now have an explicit
supersession path; stale claims that are never contradicted still do not expire.

What exists:

- `fact_contradict` clusters facts by subject, asks the model which claims directly conflict,
  and mechanically keeps the newest claim. Older claims receive reversible `supersede` ops
  in both RAG memory and the ledger, so they leave live recall and future training without
  erasing the historical change.
- The automatic post-`commit-training` pass checks subjects touched by facts from the current
  run and acts only when a current-run fact is the newest correction. The manual clean-base
  pass can inspect and resolve the entire live fact set.
- The append-only RAG op-log supports fact eviction/supersession, and the manual semantic
  fact-dedup pass collapses paraphrases of the same truth.
- `content_key` dedup collapses a *re-observed* identical fact onto one record.

What remains open:

- **Coverage is heuristic.** Subject embedding clusters and the contradiction judge can miss
  a real conflict or over-group related facts; newest-wins also assumes the later statement
  is the correction. Supersession is reversible, but it is not a truth oracle.
- **Automatic scope is intentionally narrow.** Pre-existing old-vs-old conflicts remain live
  until the manual whole-corpus pass runs.
- **No expiry or revalidation for uncontradicted falsehoods.** A stale fact that receives no
  explicit conflicting replacement remains at `1.0`.
- **Compatible/paraphrastic accumulation is not continuously bounded.** Semantic dedup is an
  operator pass, not part of every reflection, and genuinely distinct non-conflicting facts
  continue to grow the live set and append-only history.

This is tracked in `AVA_OPEN_PROBLEMS.md` → *Fact Staleness*. The direct-correction half of
"facts never fade" is built; validation, automatic coverage, and stale-without-correction
policy remain open.

### G. Third-party disclosure has no feedback loop

Problem: Ava can speak to one person about another, and that is deliberate — one shared
memory is what makes her a subject rather than a service with per-user sandboxes. The
chosen norm for *whether she should* is a **learned disposition** rather than an access
rule. But a disposition only earns its name if it can develop, and nothing currently lets
it.

What exists:

- `about`/`source`/`source_class` on every attributed fact, so a recalled item is
  labeled and misattribution is mechanically avoidable (§3.4).
- The hearsay gate keeps someone's account of a third party out of the weights, so an
  unverified claim about B cannot become something Ava believes she knows.
- Prompt-level norm in `chat_prompt.txt` and `rag_memory_prompt.txt`: what she knows of
  others is hers to share or hold, a confidence is worth keeping, and misattribution is
  the one error never justified.

What remains open:

- **No confidence signal.** Nothing distinguishes "told in passing" from "told in
  confidence". The disposition has to infer it from the content alone, every time.
- **No record of what was disclosed to whom.** Reflection therefore cannot see a
  disclosure at all, let alone judge one — so a regretted disclosure can never become
  `[persona]` evidence, which is the only route by which the disposition could mature.
  `core/worklog.py` is the natural substrate: one entry per disclosure episode would make
  it visible to the revision pass.
- **Unbounded early over-sharing.** A disposition starts empty. Between install and
  maturity there is a window where she will disclose things a person would rather she had
  not, with no floor under the behaviour. This is the same bootstrap problem as
  *First-User Imprint* in `AVA_OPEN_PROBLEMS.md`, applied to discretion.
- **The norm is untested against a real second person.** Every mechanism here is
  single-operator so far; nothing has exercised A-talks-about-B on live data.
