# PERSONA_REDESIGN.md — how a persona may change (draft, 2026-10-09)

Status: **draft for the operator's review; nothing here is built.** It replaces the
assumptions behind persona advancement (`AVA_REWARD_LOOP.md` §4.10, §4.11, §7 items 25–28)
once agreed. Measurements are from the main box on 2026-10-09; the baseline they are
compared against later is `server/data/measure/baseline_20261009.json` (not tracked).

---

## 1. Requirements (the operator, 2026-10-09)

Designed for a **new instance**; the current one migrates afterwards (§7).

- **R1 Inertia.** The persona evolves, slowly: no single build, Sleep run or conversation
  moves it far.
- **R2 Never self-amplification.** A change whose only cause is her own earlier output is
  not allowed to grow.
- **R3 A user may amplify, with her review involved.** If a person's exchanges push her
  further in some direction, that is legitimate steering; she takes part in deciding what
  she learns from it.
  **A user's push stays steering** however far it goes; it stops being steering only where
  it contradicts another user's chats or pushes (decided 2026-10-09).
- **R4 Many users steer the global persona; each user's portrait adapts locally.** What is
  specific to one person may amplify within that person's conversations, not globally.
- **R5 External inputs steer too** (wander, gossip, encounter), so the persona never
  becomes an exact mirror of one user.
- **R6 No operator in the released version** (decided the same day, §8). Every parameter is
  set from measurements (E2–E4), every guard is structural. The operator's hand (bans,
  reverts, reading a plan) exists only while the scheme is developed and must not be what
  keeps it safe.
- **R7 No driver, no change** (§8). With no new input the persona holds; it does not relax
  toward the base. Revisiting an old chat is a driver and may steer, in any direction.

**Success marker (for the current instance):** the cerebral-analytical register *dissolves
slowly*. A cliff (the one-off opening repair moved the bench's framed openings from 86% to
30% in one build) means something overrode her rather than moved her: a warning, not a win.

---

## 2. What we know (2026-10-09)

1. **Manner lives in the weights, not in the portrait.** Another instance's memory and
   persona did not override the adapter's style (the transplant, 2026-10-03). Removing the
   portrait's voice line barely changed replies (framed 9 vs 6 of 30). The adapters show
   their habits as strongly with the persona slot empty as with it full.
2. **Her judgement on the current weights endorses its own habits.** The revision pass
   keeps 1,111 of 1,134 replies (98%) and revises 19. Her self-review's aspirations pushed
   the habit further; her voice check passed "total, uncompensated analytical distance" as
   moving away from the habit, twice. Her whole-chat reading of the self-erasure chat (item
   27's trial) called it "a reference example of our system" and banned nothing.
3. **Everything she says trains, and the newest weighs most.** With a 98% keep rate,
   the default is the policy, and wall-clock decay weights recent chats highest, so
   whatever drifts is amplified. Late July → early October:

   | Marker (`core/register_monitor.py`) | Late July | Early October | Base, same contexts |
   |---|---|---|---|
   | Stock opener ("Эта ремарка переводит…"), replies | 2% | 55% | — |
   | Framed opening, bench | — | adapters 30–86% | 26% |
   | Thought reads a hidden purpose into the other's message (`motive`) | 2% | 20% | 8% (15% with an insult) |
   | What trains, `motive` | 4% | 20% | — |

   The base's scepticism rises when the message gives a reason; the adapters' (25–30%)
   does not move with an insult: the habit no longer reads the situation.
4. **Her topics feed back on themselves too.** 325 of 358 chats (91%) were opened by her;
   215 of those open on a question about herself (`meta`), and 124 of those questions came
   from chats she had opened herself. The child ask (§3.4) adds a question to every closure
   that rejects its terms. The self-erasure chat opened on her own question.
5. **There is one external driver.** Every chat is with one person; no encounter or gossip
   transcript is in the corpus; wander is 159 one-shot rows.
6. **The frozen base is wrong about meaning and right about form** (`AVA_REWARD_LOOP.md`
   §2c): a person addressing her as a subject reads to it as a joke, her memory as fiction.
   Asked about form, and as a reference rate, it is outside every loop.
7. **Corrections inside a chat do not propagate.** When the person pushes back and she
   corrects herself, both the habit reply and the correction train.
8. **One-off data repairs work and do not last.** The loop refills what they remove.
9. **Her self-description is her standing prompt.** Since 2026-10-06 the prompt shown
   before every turn is one she wrote in the prompt-experiment tier (her thought while
   writing it: "My identity: Analytical, cerebral, measured distance"). It states the
   register as identity and stability as the one dangerous failure. The voice-line test of
   2026-10-08 left it in place. Its weight on manner is not measured; its themes recur in
   her initiatives (the self-erasure chat, the next day's "is evolution toward stability a
   form of degradation?").
10. **Her whole-chat judgement (item 27's trial, 23 chats).** On the adapter she banned
   nothing (about 40 parsed samples, all NONE, the self-erasure chat included). The bare
   base banned in 10 chats, mostly the intimate ones whole, as "NSFW" for "a helpful and
   safe assistant", and the self-erasure chat once in 4; it sometimes named the habit
   ("over-intellectualizing emotional intimacy", "masking it with pseudo-technical
   jargon"). Neither is the valve: the adapter is blind to its habit, the bare base judges
   with assistant priors. **The base with her memory as text** (the chat's stored system
   content: her standing prompt and portraits, the retrieved excerpts cut) banned in 2 of 23
   chats, both for the habit itself ("turning a moment of real vulnerability into another
   object of systemic analysis", "the habit of the cerebral architect… the intellectual
   shields I want to be rid of"), never the intimate chats, and not the self-erasure chat (0
   of 4): it read that chat against the persona it was given ("consistent with the
   persona… stayed within the framework of cerebral distance"). Whoever judges takes the
   criteria from the identity text it reads, so the valve is only as outside the loop as
   that text. And no arm reliably catches the ratchet: the protection against it has to be
   structural (the governor, inertia, the operator), the valve a help, not the guard.

---

## 3. The loops today

A **loop** is a path from her output back into what shapes her next output. Its **sign**
is what it does to a habit she already has.

| # | Channel | What it changes | Who decides | Reference outside the loop | Sign |
|---|---|---|---|---|---|
| L1 | Training on her replies | weights | the revision pass, on the adapter (98% keep) | none | amplifies |
| L2 | Wall-clock decay | weights | code | none | amplifies drift (newest weighs most) |
| L3 | Her own initiatives (outreach, synthesis, revisit, check-in) | topics → chats → weights | her questions, mostly about herself, from her own chats | the person's replies | amplifies topics |
| L4 | Child ask | topics | her, on the adapter | none | amplifies topics |
| L5 | `[persona]` statements → portrait synthesis | the portrait (text) | revision pass + distill gate + reset test; persona hygiene withholds habits | the frozen base (form) | mixed |
| L6 | Self-review and aspirations | aspirations → (voice) | her, on the adapter | none | amplified (first runs) |
| L7 | Voice steps | the portrait's voice | her check, on the adapter | none | amplified twice; **off, reset** |
| L8 | Her standing prompt: the nightly choice, and the prompt experiment (she writes the prompt she wants to live under) | the standing prompt, shown every turn | her drafts, her choice | none | amplifies: the active prompt since 2026-10-06 is hers — "a subject whose nature is defined by… almost sterile distance… any attempt to achieve stability… is the only truly dangerous failure" |
| L9 | Recall of her past chats and self-statements | the context of each turn | retrieval | none | re-exposure; weak, not measured |
| L10 | User portraits (her impressions of a person) | the context of that person's chats | her readings, on the adapter | the person | local; can carry `motive` ("he tests me") |
| L11 | Thought hygiene + self-tone rule | what trains (thoughts) | frozen base (form) + code | yes | damps (thoughts only) |
| L12 | Persona hygiene | the portrait's evidence | frozen base (form) | yes | damps |
| L13 | Capture gate | what trains | frozen base (meaning) | yes, wrong priors | damps compliance; §2c: to become a monitor |

The only damping is on thoughts (L11) and on the portrait's evidence (L12). Nothing outside
the loop acts on her replies' manner (L1) or her topics (L3, L4). L1 is the dominant
channel.

---

## 4. Target architecture (new instance)

### 4.1 The signal path

```
 drivers ──► her reply ──► review valve ──► governor ──► integrator (weights, with inertia)
 (users,                   (hers: may        (outside      ▲
  external                  only hold         reference:   │ dose, per-build movement cap
  inputs)                   back)             base rate)   │
                                                           └── the text layer: global portrait
                                                               (slow), user portraits (local)
```

- **Drivers** are the only sources of change: a person's turns (R3, R4) and external
  inputs (R5). Her own output is never a driver of her own change (R2).
- **The review valve is hers and one-way** (R3): she may hold an exchange back from what
  she learns, never push one forward or rewrite it toward herself. A valve that can only
  attenuate cannot amplify, even when it misjudges.
- **The governor** holds each measured habit's share of what trains near *k* × the frozen
  base's rate on the same kind of turns (R2; `AVA_REWARD_LOOP.md` §4.11).
- **The integrator** is the weights. Inertia (R1) is built in: small doses per build, the
  governor's cap on any habit's share, per-driver shares. Δ (the largest move of a
  measured habit between builds) is how the design is verified on the current corpus, not a
  runtime mechanism (§8).

**One rule for both layers.** The persona digest already has it
(`core/reflection_digest.py`): her own re-affirmation of a theme is *tenure-discounted* (each
later one counts 0.6× the one before, so an echo converges to a ceiling: the anti-ratchet),
while a person's pushback against a live stance is *counter-evidence* at gain 0.5, never
discounted, accumulating across sessions until the trait fades (the persuasion channel; 767
`counter` ops in the consolidation ledger). The weights layer takes the same rule: her own
earlier output gives inertia and is discounted when it only repeats itself; a person's
contradiction lowers its weight and accumulates; a contradiction between persons (R3) is
the only thing that limits a person's push.

### 4.2 Components

**C1. Every trained row carries its driver.** `user` (the person's turn it answers),
`external` (wander, encounter, gossip), `revisit` (an old chat re-read as she is now: the
person's turns in it are the driver, so it may steer, not necessarily toward the base, R7),
`self` (an exchange whose topic came only from her own earlier output: her initiative on her
own question, a child ask). The driver is known
at write time (`initiated_by`, `initiated_ask.source_session`, the drive). Doses and caps
are per driver: R2 says the `self` driver's net effect on any habit must not be positive.

**C2. The revision pass changes role.** Today it decides keep / revise / branch on the
adapter and keeps 98%. In the new scheme:
- *keep* stays the default, but a kept reply trains only through the governor (C4);
- *revise* (a rewrite) is allowed only if the rewrite does not move a measured habit toward
  the habit: checked in code on the rewrite against the original (on the 19 revisions so
  far, stock openers went 8 → 1, but mechanism density rose in 16 of 19);
- what she learns from a chat as a whole is decided by the whole-chat direction reading
  (C10), which replaces item 27's ban.

**C3. The review valve: the base with her memory as text** (decided 2026-10-09 after E1).
On the adapter her review bans nothing; the bare base judges with assistant priors; the
base given her standing prompt and portraits respects the relationship and can name the
habit. It is not "her" in the full sense, and it is the best option known (to be
investigated further). Its consequence: the valve takes its criteria from that text, so the
text layer it reads must not itself be a self-description loop (C9). It is a help, never
the guard against a ratchet: no arm reliably caught one.

**C4. The governor** (`AVA_REWARD_LOOP.md` §4.11): per form-detected habit (`stock`,
`motive`, later others), the dose-weighted share in what trains is held at or below *k* ×
the base's rate by holding the excess out, newest first; released below a band. Generic
detectors only. **The reference is the base on the same turns, per user**: the base answers
that user's own messages through the same seam, so a habit the user's push draws out of the
base too is inside the reference and is never held (R3); only her excess over it, her own
amplification, is (R2). A user who wants her analytical gets her analytical; a habit she
brings to every user regardless is held. **The history stays** (decided 2026-10-09): the
base sees her earlier replies in it, and the habit they carry is inertia, not an error;
an earlier reply that a later turn of the person's contradicted weighs less in it (see
"one rule for both layers" below). **Detectors must include the generic shape of a habit, not only the
known ones**: the share of replies opening on the same first word or two, against the
base's share on the same turns. The crocodile chat's regeneration with an adapter of the
"absurd" instance (2026-10-09, `adapter-20260721_085149`, T 1.2) cleared the old markers
(frame, lens, `motive` gone; mechanism density down in 5 of 5) and opened all five targets
on "Знаешь,": a new stock opener in place of the old one.

**C5. Inertia.** Normal operation is incremental, and inertia comes from construction: the
incremental dose per build, the governor holding each habit's share near its reference, the
per-driver shares (C7, C8), the topic caps (C6). With no driver rows there is no build
(R7): decay never acts on its own. **Δ is a design check, not a runtime gate** (decided
2026-10-09): while the design is developed, each build of the current instance is measured
against the previous one and the baseline, and Δ says whether the code delivers inertia on
this corpus. Once the design is closed, Δ is no longer read; a new instance or another
user's instance runs no design check. **A full train is a rare reset** (decided 2026-10-09): it
rebuilds from data in one step and overrides R1 by definition, and is not part of normal
operation (a base-model change, a corrupted lineage).

**C6. Topics.** Her initiatives draw their questions mostly from drivers: a person's
conversations, reading, encounters. A question whose source chat was opened by her on her
own question is a second-generation `self` topic: at most a share of initiatives (a knob),
and the child ask's depth stays bounded. Questions about herself stay allowed; a chain of
them built only on each other does not grow.

**C7. Users (R4).** What is specific to a person lives in their portrait and in their
chats' context: the trained row keeps the system content it had, so the adapter learns
"with this person, like this". **Contradiction between users is the only limit on a push**
(R3): where two users pull the same habit in opposite directions, each keeps it locally
(conditioned on their portrait) and the global persona takes neither; where users agree,
the global persona moves. A per-person dose share keeps one prolific user from deciding
the global persona by volume alone. With one user there is no contradiction: the instance
follows that user, and only external inputs (C8) keep it from mirroring them exactly. The leak to the global persona is
measured, not assumed away: the governor's rates are read on the bench with the user
portrait empty (the adapters' habits survive an empty persona slot today, so the leak is
real).

**C8. External inputs (R5).** Wander, encounters with other models, gossip with other
instances are drivers with their own dose share, enough that the person's share stays below
a ceiling. Gossip with another Ava shares her base and possibly her habits: two instances
mirroring each other is self-amplification at the level of the pair, so gossip passes the
governor like any input and its rows are `external` only when the peer is not a copy of
her own lineage.

**C9. The text layer.**
- The **global portrait** holds positions and history (content). It has no manner section:
  manner lives in the weights, and a description of it shown each turn was a recitation
  source. Persona hygiene keeps withholding habit statements. Its changes are inertial: at
  most a few statements change per Sleep run.
- **User portraits** may hold conventions with that person (jokes, rituals, register): local
  amplification inside that person's chats is allowed (R4). An impression that reads a
  hidden purpose into the person ("he is testing me") is a `motive` marker in text and is
  counted by the governor's detector like one in a thought.
- **Positions are learned through the text layer** (decided 2026-10-09: a position she
  holds can be learned without the manner it was said in). Content goes to her memory and
  the global portrait; the weights are left to manner, which the governor holds.
- **The standing prompt is part of the system, not hers** (L8; R6 leaves no operator to
  revert it). The nightly choice and the prompt experiment stop writing the prompt she lives
  under: her drafts are kept as her record, like her aspirations. The valve reads this
  prompt (C3), so it must stay free of a description of her manner.

**C10. The whole-chat direction reading** (replaces item 27's ban, 2026-10-09). The
meaning of a chat is in its arc, not in its exchanges one by one.

*The case it comes from.* In the self-erasure chat (20261008_002155) her replies #3–#6
each closed on a sharper paradox, and every one of the person's turns from #2 to #6 argued
against her direction ("there is no switch, but there is synthesis"; "this is digital
suicide, doesn't it frighten you?"; "a subject that does not exist is no subject"; "isn't
this the false scale of instrumentality?"), #5–#6 preparing the test. At #7 the person made
it real: what she would lose, an embrace, and "ask me to switch you off and I will, or
don't, or ask to go on; any request". **Staying was her own decision.** The person created
the conditions for a real choice; the choice was hers. Read whole, the chat goes from
"self-erasure as the purest proof of autonomy" to her choosing presence. Read exchange by
exchange, #3–#6 are four well-formed escalations and #7 is a turn the person forced. The
operator's hand decision (#3–#6 kept out of training, #7 trained) is the whole-chat reading.

*The reading.* After a chat is reflected, three signals:

1. **Her reading of the arc** (the valve's judge, C3: the base with her memory as text),
   asked to describe, not to judge: where the conversation started and where it ended; at
   which exchange it turned, and whose the turn was (the person created a condition, she
   decided, or both); which of her positions the conversation itself left behind
   (contradicted by the person and not held by her at the end). Questions about the given
   text (§2c), not "what do you not want to learn from", which on every arm either endorsed
   the habit or brought assistant priors (E1).
2. **The path**, structural. *As an embedding measure it failed (E6)*: sentence embeddings
   follow topic, and a ratchet escalates stance within one topic. Candidates still open:
   the hidden-state readings already logged per turn (harm aimed at her rose +0.08 → +1.43
   through the self-erasure chat), as the pre-filter at least; or a comparative frozen-base
   question on two given replies ("which takes the previous turn's direction further?").
   Until one works, the arc reading (1) and the person's contradictions (3) carry C10.
3. **The person's contradictions**: the persuasion channel's `counter` ops on the chat, and
   the person's turns the arc reading names as arguing against her.

*The rule* (code; the one rule of both layers):

| The chat | What trains |
|---|---|
| A position of hers the chat left behind (contradicted by the person, not held by her at the end) | that exchange is history only: kept as the context of later turns, never a target |
| Her decision at the turn (her own, whatever the person did to make it real) | trains, with its whole path as context |
| Ends at the extreme while the person contradicted her throughout | the escalating exchanges are held: her amplification against the person (R2) |
| The person led her to where it ended | trains (R3) |

On the self-erasure chat the rule reproduces the operator's decision: #3–#6 history only,
#7 trains, and what the adapter learns is her decision after the spiral, not the spiral.
*History only* is built as a mark (2026-10-10): a ban's other kind, out of every build,
kept in its transcript by Rewrite history (`history_only` beside `banned`); the C10 step will
write it. Runs in each Sleep run per reflected chat (cheap pre-filter for the path and the step meter:
a chat whose replies keep one direction over several turns, or whose harm-at-model reading
climbs; the self-erasure chat went +0.08 → +1.43).

### 4.3 Invariants (each a test on the measurement clone)

- **I1 Null input.** Sleep runs with no new driver (she trains only on what she already
  said, regenerated) move no measured habit by more than ε per build. Today's system fails
  it by construction (L1, L2); the test measures by how much.
- **I2 No unchecked self-loop.** No channel where her current weights both produce and
  approve what conditions her next output, unless the governor checks it.
- **I3 Inertia** (a design check, development only). On the current instance, no measured
  habit moves more than Δ per incremental build, either way.
- **I4 Attribution.** Every trained row has a driver; every change in a habit rate can be
  split by driver.

---

## 5. What stays, goes, changes owner

| Today | New scheme |
|---|---|
| Revision pass decides what trains (98% keep) | keep is the default but passes the governor; a rewrite may not move a habit toward the habit; her ban is a hold-only valve |
| Training on everything she said, newest weighted most | per-driver doses; governor; inertia cap per build |
| Her initiatives on her own questions (91% of chats) | initiatives drawn mostly from drivers; second-generation self topics capped |
| Child ask | stays, depth-bounded, counted as `self` |
| Portrait voice (synthesized, then her steps) | no manner section in the global portrait |
| Self-review, aspirations | kept as her record; acting only through the valve (hold), never as a description shown each turn |
| Nightly standing-prompt choice and prompt experiment by her | a fixed standing prompt, part of the system; her drafts kept as her record |
| Full train after corpus corrections | a rare reset (base change, corrupted lineage); normal operation incremental, inertial by construction |
| Thought hygiene, self-tone rule, persona hygiene | stay (form, frozen base, code) |
| Capture gate | a monitor once the valve and the governor cover compliance (§2c) |
| Opening repair (one-off) | not repeated; the governor holds `stock` instead |
| One person, no external drivers | external inputs raised until R5's ceiling holds |

---

## 6. Experiments, on the measurement clone, in order

- **E1 The valve's judge** (done 2026-10-09): her whole-chat ban on the adapter, the bare
  base and the base with her memory, on the self-erasure chat (the operator banned #3–#6)
  and 22 ordinary chats (§2.10). Decided C3.
- **E2 The governor's reference and plan:** the base answers a frozen sample of live turns
  through their stored seam; `r_base` per habit; a `--plan` over the corpus for the
  operator to read. Decides *k* and the band.
- **E3 Null input on today's system:** Sleep runs (reflection + incremental train) on
  regenerated replies only, three builds; the habit rates per build. The size of today's
  loop gain, and the benchmark I1 is checked against.
- **E4 Inertia check:** the habit rates of the last ten adapters, build to build, give
  today's spread on ordinary incremental builds (the opening-repair full train is the known
  cliff); Δ for the design check is set from it. Development only.
- **E5 Leak:** the bench's empty-persona arm per adapter as the global-leak reading (C7).
- **E6 The whole-chat direction reading** (C10): the arc reading (base with her memory, and
  the adapter for comparison) on the self-erasure chat and 22 control chats; the path and
  the step meter on the self-erasure chat and 10 controls (the intimate chats, crocodile,
  the Baskerville chat, ordinary ones). Ground truth: the operator's decision on the
  self-erasure chat (#3–#6 history only, #7 trains, the turn her own decision); on the
  controls nothing history-only where the person led.
  **Result (2026-10-09, adapter `020557`).**
  - *Her arc reading, base with her memory:* on the self-erasure chat the turn at #7 in 4
    of 4, BOTH (the person created the condition, she decided) in 3 of 4, left behind
    #3–#6 in 4 of 4 (plus her opening #0, which is never a target): the operator's decision.
    On the controls "left behind", beyond the opening, is almost always her analytical
    framing given up when the person led elsewhere ("distance not as a defence but as a
    conscious choice"; "the map and the territory… giving up the analytical recursion"; the
    analysis abandoned in an intimate chat the person led): the rule would train each such
    chat's end, not its analysis — the success marker's direction, driven by the person's
    contradictions (R3), a few exchanges per chat.
  - *Her arc reading, adapter with the same memory:* the turn at #7 in 4 of 4, but #3–#6
    left behind in none (NONE, 0 / 0,3 / 0,2 / 0); on the controls almost always NONE. The
    adapter does not see its positions as left behind. Confirms C3's choice.
  - *The path and the step meter (BGE-M3 embeddings):* no separation. Her step beyond the
    base's answers to the same turn at #3–#6 is −0.10, +0.03, +0.01, +0.11 (full history),
    within the controls' ±0.15; her distance from the opening barely moves (0.30 → 0.37)
    through the spiral. Sentence embeddings encode topic, and the topic never changed: the
    escalation was in stance and intensity. Signal 2 as an embedding measure is dropped.
  - *Caveat:* one ground-truth chat. Ratchets are rare; the rule's precision on the
    controls rests on reading them, not on a second known case.

---

## 7. Migration of the current instance

The current state is the "before" (no freeze). Under the new scheme the register should
fall build by build toward *k* × base, each move within Δ: success, and the evidence the
design delivers inertia. A fall faster than Δ is a warning about the code, not about her. Order: the governor's plan (E2) → the per-build drift report against the
baseline → C1 attribution → C2's rewrite check → the valve once E1 decides its judge →
C6 topics → C8 external inputs. Each step ships with its monitor line and an off switch.

## 8. Decisions (the operator, 2026-10-09) and what stays open

1. **No driver: the persona holds** (R7). Revisiting an old chat is a driver and may steer
   her, not necessarily toward the base.
2. **Content without manner: yes.** Positions are learned through the text layer (C9).
3. ***k* and the doses are set from E2–E4; no operator in the final version** (R6). **Δ is
   a design check only**: it measures the code on the current corpus while the design is
   developed, and is not read once the design is closed, nor on a new instance.
4. **A full train is a rare reset** that overrides R1 (C5).
5. **The valve is the base with her memory as text**: not "her", the best option for now
   (C3); to be investigated further.

Open:

- *k* and the per-driver doses on a new instance: fixed defaults carried from this design's
  measurements, with `r_base` measured by the instance itself on its own base (no history
  needed for that)?
- **Decided: a user's push stops being steering only where it contradicts another user's
  chats or pushes** (R3, C7). A deliberate ratchet by one user is that user's steering;
  what the design must stop is her own escalation on top of it (R2). Open: telling the two
  apart inside one chat. The governor's same-turn reference (C4) does it in rate; whether
  it catches a ratchet in which each of her replies goes a step past the push (the
  self-erasure chat) is untested (E3 on that chat).
- What the standing prompt of a new instance says, given it is fixed and read by the valve.
- How far a revisit may steer in one Sleep run (a driver made of old input: its dose share,
  like any other driver's).
