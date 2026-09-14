# Facts tree — implementation brief

Status: **stage 1 built (2026-08-10)**, stages 2–5 designed. The build is pure/GPU-free and
runs — `cd server && .venv/bin/python -m graph.build`; see `server/graph/DESIGN.md` for
implementation notes and the first real build's numbers. Everything below stage 1 in §11 is
still design.

**What the first real build changed in this document.** Two measurements landed differently
than the design assumed, and both are recorded in place below: the exact-match tier merges
**nothing** on the live corpus (0 collapsed, 0 corroborated of 398 occurrences), and **86% of
nodes come out untyped**. Neither invalidates the shape; both move where the remaining value
is (§5b tier 3, and §12's topic classifier).

Both fact-protocol producers are built and running — `core/chat_facts.py` writes
`chats/<stem>.facts.json`, `core/til_facts.py` writes
`til/snippets/<kind>/<stem>.facts.json` — and both docstrings say the same thing about the
consumer: *offline, a knowledge-graph build, not yet written*. This is the design of that
build and of what may read it.

---

## 1. The corpus as it actually stands (measured 2026-08-10)

Every number below is counted off this box, not estimated. They drive the design, so they
come first.

| | chat lane | TIL lane |
|---|---|---|
| protocols on disk (authoritative) | 20 | 3 (2 news, 1 wander) |
| fact lines | 349 | 49 |
| distinct `subject` values | **7** | **25** |
| distinct `entities` mentions | 146 | 56 |
| lines carrying `entities` | 212 / 349 | — |
| lines carrying `when` | **3 / 349** | most |
| classes | 285 stated / 52 standing / 11 event / 1 unspecified | mostly stated + event |

Chat subjects, in full — this is the whole namespace, not a top-N:

```
_self 234   artemyvo 96   "" 9   nobody 5   claude 3   name 1   nvidia 1
```

Cross-tabbed against class:

```
_self    × stated     205      _self    × standing    26
artemyvo × stated      75      artemyvo × standing    18
                                ... everything else ≤ 5
```

### The finding that drives everything below

**80% of the chat corpus (280 / 349) is two people's `stated` positions**, and 59% is
`_self × stated` alone. The chat lane holds 52 `standing` facts and 11 events in total.

So the chat protocol is not, empirically, a knowledge store. It is a **position ledger** —
a record of who held what view, in which conversation. That is a genuinely valuable
artifact (it is the raw material the *Fact Staleness* open problem asks for), but it is
close to the worst thing on the box to treat as knowledge: `stated` means *true only as a
record that someone said it*, and the largest single bucket in the corpus is Ava's own
positions. The TIL lane, by contrast, is shaped like world knowledge and is where the
"what do I know about X" question actually has answers.

A tree that ignores this distinction produces one enormous `_self` bucket of opinions and
presents it as fact. The facet level (§5) exists to make that structurally impossible.

---

## 2. The one architectural claim

> **The tree is a derived, disposable fold — never a store.**

The `.facts.json` files stay the immutable sources. The tree is rebuilt from them and can
be deleted at any time with no loss. Consequences, all of them wanted:

- A better resolver is a **rebuild, not a migration**. At 398 total facts the full build is
  seconds and fits in memory; there is no cost argument for incremental state.
- A replaced protocol (re-reflection rewrites the file wholesale; `needs_reparse`
  re-derives stale ones) supersedes cleanly, because nothing downstream holds a copy.
- The tree can never disagree with its sources about what was said. It can only disagree
  about *resolution* — which is exactly the thing a rebuild fixes.

This is the source/fold discipline the project already uses three times over
(`weights_persona.jsonl` → ledger, `rag_memory.jsonl` → live items, `[impression]` →
portrait). Nothing new is being introduced; the tree is the fourth instance.

---

## 3. What the tree has to solve

Five problems, all measured in §1 rather than anticipated.

**P1 — Two incompatible subject namespaces.** Chat subjects are *person keys* through
`normalize_person` (first-token, lowercased); TIL subjects are *entity mentions* (surface
form preserved). Keyed naively on `subject`, the two lanes can never meet, and the chat
namespace is not even internally clean: `nobody`, `name` and `nvidia` are sitting in it as
person keys, because `normalize_person` takes anything not on its generic-referent list at
face value.

**P2 — `entities` mixes two kinds of node.** Of 146 chat mentions, proper nouns
("Israel", "China", "Project Ava") sit beside abstractions ("subjectivity" ×13, "identity",
"will", "responsibility", "personality"). A thing in the world and a theme a conversation
keeps returning to are not the same node and must not share an index. Worse, `self` (×22)
and `artemyvo` (×21) appear as *entities* while also being *subjects* — the two slots
already overlap with no shared key space.

**P3 — `fact_class` semantics drift by lane.** On the TIL lane, reported events are written
`stated` ("All missiles were successfully intercepted" — class `stated`). The schema means
*a position someone holds*; the lane means *the text asserts this*. A consumer that reads
`stated` as "someone's opinion" is wrong on every news line.

**P4 — Only one lane has a `when`.** 3 of 349 chat facts carry one. Chat facts' only clock
is the document's `ts` / the session stem.

**P5 — 38 of the 61 `.facts.json` files on disk are copies.** They live in persona
snapshots (32), the reflection checkpoint (2), and reflection archives (2). A build that
globs naively double-counts the same fact up to three times.

---

## 4. Shape: a tree with edges

"Tree" is the right word for the spine and the wrong word for the whole thing — facts touch
several entities, so the cross-links are a graph. The design keeps both, and keeps them
separate:

```
node                    canonical entity          person:artemyvo
 └── facet              kind of claim             property | position | report | event
      └── claim         deduped assertion         "drinks Reviseur XO on the balcony"
           └── occurrence   one witness line      20260729_174827.facts.json#12
```

Four levels, each answering a different consumer question:

| level | answers |
|---|---|
| node | what is there anything at all about? |
| facet | what is *still true* vs what someone *said* vs what *happened* |
| claim | the statement itself — the unit you would show, inject, or answer with |
| occurrence | provenance: which source, when, who said it, how often repeated |

**Edges** hang off the claim: each occurrence's `entities` resolve to node ids, giving
`claim —mentions→ node`. Traversal ("what connects Artemy and Israel?") walks edges; the
tree is what you navigate and render. Keeping the edge set derived from occurrences rather
than stored on nodes means a rebuild cannot leave a dangling edge.

The occurrence level is what makes the whole thing auditable, and it is non-negotiable: it
is the only level that maps 1:1 onto an immutable source line, so every claim above it can
be traced to the exact conversation and line that produced it. It is also what carries
`subject_raw`, so mis-resolution stays visible after the fact.

---

## 5. Node identity

### 5a. A typed id space

```
person:<normalize_person key>     chat subjects; person:_self reserved for Ava
entity:<slug>                     world entities — TIL subjects, proper-noun mentions
topic:<slug>                      abstractions — "subjectivity", "identity", "will"
```

Typed rather than flat, because P1/P2 are not accidents of extraction — the lanes mean
different things, and a flat space is how `nvidia` becomes a person. The prefix is also the
render rule: a `person:` node's facets are about someone who can be asked, a `topic:` node's
are a theme index and nothing more.

`person:_self` is already reserved upstream (`chat_facts.SELF_SUBJECT`,
`user_digest.RESERVED_SLUGS`); the tree inherits it rather than inventing a second marker.

### 5b. Resolution in tiers, cheapest first

The corpus is 398 facts and ~200 distinct mentions. Most collapse on string equality; the
build should not burn a GPU proving that.

1. **Exact** — casefolded surface match, after formatting normalization. Free.
2. **Alias** — a small on-disk alias table (`aliases.json`, hand-editable), plus the
   structural aliases the box already knows for free: `source_user` values, live
   `[fact].about` keys, `user_digest` person slugs. Free, and the operator's escape hatch
   when a later tier is wrong.
3. **Embedding-blocked** — group remaining mentions by cosine, reusing
   `fact_contradict.cluster_by_subject(facts, embed_fn, …)` rather than writing a third
   copy of subject blocking. CPU. Blocking only; it proposes candidates, it does not merge.
4. **LLM adjudication** — optional, off by default. Judges only the pairs tier 3 proposed.
   Grouping paraphrases is an *evaluation*, so it runs on the clean base if it runs at all
   — the rule `fact_dedup`, `persona_cluster` and `self_reconcile` already follow.

Type assignment rides the same tiers: a mention matching a known person key is `person:`,
a mention that appears as a TIL subject is `entity:`, and only the residue reaches a
classifier. Ava and the operator are known structurally and never guessed.

**Precision over recall, deliberately, and for a reason specific to this artifact.** A
missed merge leaves two nodes a rebuild can join later; a false merge attributes one
person's position to another and is invisible once the surface forms are gone. The tree
keeps `subject_raw` on every occurrence precisely so a false merge is recoverable — but
recoverable is not the same as harmless.

---

## 6. Facet: fixing P3 without touching the producers

Facet is a pure function of `(lane, fact_class)` — no model, no new field, no producer
change:

| lane | `fact_class` | facet | means |
|---|---|---|---|
| chat | standing | `property` | true beyond the moment |
| chat | stated | `position` | X holds this — true only as a record that they said it |
| chat | event | `event` | happened |
| TIL | standing | `property` | the text asserts a standing truth |
| TIL | stated | `report` | **the text asserts this** — not anyone's opinion |
| TIL | event | `event` | happened |
| either | unspecified | `unclassified` | kept separate; never silently folded into a neighbour |

`report` is the whole point of the mapping: it separates "a news digest states X" from
"Artemy thinks X", which the shared `stated` value currently collapses. Provenance then
weights `report` — and weighting stays a *consumer* decision, exactly as `til_facts`
insists (judging a source is interpretation; the producer is a witness).

`position` is the facet that must never be rendered as knowledge. It is 80% of the chat
corpus.

---

## 7. Two clocks

Already the project's habit (`[recollection]`'s `ts` vs `origin_ts`):

- **`asserted_at`** — when it was *recorded*. Chat: the session stem, which is what every
  chat-side decay curve already keys on. TIL: `source_date`.
- **`when`** — when the thing *happened*. TIL-populated, near-absent on chat (P4).

Claim freshness and supersession rank on `asserted_at`. Only `event` facets place on
`when`, and a claim with no `when` is simply absent from a timeline rather than defaulted
to its assertion date — a fact asserted today about 1998 is not a fact about today.

---

## 8. Read set and rebuild

**One definition of the authoritative set**, in one function — the discipline
`chat_sidecar.is_chat_session_json` and `snapshot_state.plan_manifest` already establish:

```
server/data/chats/*.facts.json
server/data/til/snippets/<kind>/*.facts.json
```

and nothing else. Persona snapshots, `reflection_checkpoint/`, and `reflections/<run_id>/`
are copies by construction (P5: 38 of 61 files). A build that reads them is not more
complete, it is wrong.

Occurrence key is `(source_ref, line_index)`. Because the tree is disposable, supersession
needs no logic: a rewritten protocol simply produces different occurrences on the next
build.

---

## 9. Where the code lives

```
server/graph/
  read.py       authoritative read set + record loading      (pure)
  nodes.py      id space, normalization, alias table         (pure)
  resolve.py    the four tiers; embed_fn/generate_fn injected
  fold.py       occurrence → claim → facet → node + edges    (pure)
  store.py      write/read the built tree
  build.py      CLI entry point
  DESIGN.md     the detail that outgrows this brief
```

A new offline role dir beside `server/training/`, for the same reasons that one exists: a
multi-stage build with its own storage and an optional model tier is not inference. It is
GPU-free in its default configuration (tiers 1–3), so it needs no `watchdog_jobs.json`
entry and no model unload — it can run from the CLI or as an idle job. Only enabling tier 4
makes it GPU work, and then it belongs in a run's existing clean-base window rather than in
a job of its own.

Rejected alternative: `core/fact_graph.py`. Everything facts-adjacent lives in `core/`, but
`core/` modules are leaves or single passes; this has stages, storage and a resolver with a
swappable backend. Inference consumers import the thin read API from `store.py`; the build
never imports inference.

Output: `server/data/graph/tree.json` (+ `aliases.json`, hand-edited, the one input that is
*not* derived and therefore the one file here worth backing up).

---

## 10. Wiring — consumption, in risk order

The constraint that governs this whole section, from both producer docstrings:

> Nothing injects the fact protocols, **and that is what makes the extraction volume
> safe.** The reflection block has three slots and already needs a subject cap and a
> near-duplicate rule to stay legible.

The tree does not relax that. It makes the volume *navigable*, which is a different thing
from making it injectable. Consumers therefore land in this order, and the order is the
design:

**1. Read-only inspection — first, and unconditionally.**
A Facts-tree tab (or CLI dump) that browses node → facet → claim → occurrence, showing
`subject_raw` beside every resolution. This is what makes the resolver's errors visible
*before* anything depends on them. The precedent is exact: the Debug tab is the outside-view
portrait's primary surface precisely because nothing injects that portrait. Ship this alone
and the build is already worth having.

**2. Contradiction and staleness reports — offline.**
`fact_contradict.py` exists and needs a subject-blocked candidate set; the tree hands it one
for free. More to the point, the `position` facet under a `person:` node is *literally* the
Fact Staleness open problem's raw material: two positions from one person, ordered by
`asserted_at`, is the question that problem asks. Output is a report an operator reads, not
an automatic supersession.

**3. Proposals into the live store — never a bypass of it.**
The tree must not become a second retrieval channel; `rag_memory.jsonl` stays authoritative
for what Ava believes and recalls. Instead the tree *proposes*: a `property` claim recurring
across several sources with no live `[fact]` counterpart is a distillation candidate, routed
through `ReflectionWriter` like everything else, so there stays exactly one authority and
one eviction path. Note the facet gate — `position` and `report` claims are **not**
candidates, which is what stops 280 conversational opinions from being promoted to
knowledge.

**4. Person-subtree feed into the existing portrait.**
`user_digest` already folds `[impression]` + attributed facts per person and is already an
injected artifact with its own maturity gate and kill-switch. The `person:<key>` subtree,
facet-filtered to `property`, is a better-organized version of that same input. Low risk
because it changes the *quality* of an existing input rather than opening a new path.

**5. A node-scoped retrieval channel — last, gated, off by default.**
If a block is ever injected, it obeys the reflection block's existing discipline (subject
cap, near-duplicate rule, slot budget) and ships behind its own `graph.enabled` switch,
defaulting off — the pattern every recent channel followed (`anchors.enabled`,
`recollections.enabled`, `self_impressions.enabled`). Restricted to `property` and `event`
facets. This one may well never be built; it is here to be argued about, not assumed.

> **Built and turned ON by default, 2026-08-12.** The paragraph above is what this section
> argued for and is kept as written, because the reasoning behind the caution is still the
> reasoning — but the default it specifies is no longer what the code does, and a design
> doc left contradicting the code is the failure mode this repo's precedence rule
> (`code > STATUS > DESIGN`) exists to catch. What changed, and what did not:
>
> - **Changed:** `graph.enabled` defaults `true`. A chat turn is two-stage on any box that
>   pulls, without opting in. The switch remains and `false` restores this section exactly.
> - **Unchanged:** the facet restriction (`property`/`event` only), the `person:_self`
>   exclusion, and the fact that both hold *by construction* — the candidate list is
>   filtered before the pass sees it, so a forbidden claim is never shown. Rendering is
>   still done in code from returned ordinals.
> - **The open cost**, which the default no longer lets an operator avoid by inaction:
>   stage 1 prefills the candidate list (~10k tokens on the current corpus) on every turn,
>   paid on time-to-first-token. The candidate cap in `claim_candidates` is what bounds it,
>   and matters more now than when this was opt-in.
>
> **Amended 2026-08-12:** the TIL freshness scope beside that cap (`til_max_age_days`, which
> shipped at 7 days) is now **unset by default** — a TIL claim is offered whatever its age,
> exactly like a chat-backed one. The cut was keyed on the *lane*, but the lane is not the
> perishability: a news digest is spent in days while a wandered article about a place, a
> trope or an organisation ages like a chat fact, so the scope was discarding the durable
> world material in order to bound the perishable. Bounding the prefill is the cap's job, and
> the cap truncates oldest-TIL-first — so it drops the same claims the age cut did, and only
> when the list is actually too long. The mechanism stays for an operator who wants a scope.
> - **Still unmeasured on a live GPU:** whether the pass picks *well*, and what the second
>   stage costs in practice. Those were the questions this section deferred; enabling by
>   default answers neither, it just makes them urgent.
>
> **Extended 2026-08-13 — the pick is also a retrieval key.** This section framed consumer 5
> as *what the tree says back*. It turns out the more valuable half is *what the tree points
> at*: every chat-lane claim carries the conversation it was established in (the occurrence
> level records `lane` + `source_ref` — §4), so a picked fact names a conversation, and that
> conversation can be recalled. `graph.blob.chat_sources` answers the question; the injecting
> is `rag_engine`'s (a reserved slot in the past-chat block, capped by `graph.nominate_max`).
>
> This is the one retrieval on the box whose relevance is not decided by a cosine, and that
> is the whole of its value. The three existing routes into an old conversation — verbatim
> passages, the exchange anchor, the gist — are all embedding matches against the arriving
> message, so a conversation that shares no wording with it is unreachable however relevant
> it is. On this corpus that is not a corner case: **886 of 946 claims are written in English
> while the conversations are often Russian** (the same measurement that forced the claim
> lane), and past `rag_cap_age_h` an aged conversation has no verbatim vectors left to be
> found by at all. A fact recalled from such a chat arrived as one decontextualized sentence,
> with the conversation that gave it its meaning sitting on disk, unreachable.
>
> Grain is forced, and it is the honest constraint: `chat_facts` records per conversation
> with no exchange index, so a nomination can point at a chat and never at a turn within it.
> The gist is the box's only session-grained representation, so a nomination injects that or
> nothing — no fallback guesses an exchange.
>
> **Both lanes nominate as of 2026-08-13.** This paragraph previously read *"TIL claims
> nominate nothing"*, that lane having no distilled recap of any kind to point at. It now
> has one (`core.til_gist`, a per-snippet `<stem>.summary.json` written beside the
> `<stem>.facts.json` this section's protocol produces), so a fact from a wandered article
> or a news digest recalls the text it came out of exactly as a chat fact recalls its
> conversation. `chat_sources` became the typed `sources`, returning `(lane, ref)`, because
> the two refs resolve through different stores and a caller should not be re-deriving that
> from the shape of a string.
>
> The safety rules are untouched, and the *reason* they survive is that this route adds no
> new selection: a nomination is derived from a claim that already passed `claim_candidates`,
> so `person:_self` and the `position`/`report` facets cannot reach it. What it injects is a
> conversation recap — prose that may well contain positions — but as *the conversation those
> facts came out of*, which is the register a recap belongs in, not as knowledge.

> **The reach-out lane fetches as of 2026-08-18.** The two passes that RAISE questions —
> outreach's decision (whose `resolved` branch is the box's one formal *"have I already
> learned the answer?"* judgement, and evicts a live ask for good) and synthesis's analysis
> (which composes and sends its own opener, so no downstream check covers it) — now run
> stage 1 before they deliberate, through two further `fact_fetch` lanes
> (`fetch_blob_for_ask`, `fetch_blob_for_reread`) with their own closings, for the reason
> the first two closings are already separate: each lane asks a different question of the
> same candidate list. Knowledge facets only on both — a `report` must not be what retires
> a question or suppresses one. Outreach also hands its picks' sources to the nomination
> slot above (the seam grew `rag_nominate_sessions`), so the decision pass gets the recall
> half of the channel too; synthesis deliberately does not (it reads ONE conversation,
> RAG-off). One switch for the lane: `graph.reachout_facts`, default on, riding
> `graph.enabled`.

> **Being argued about, 2026-08-11 — `graph/blob.py` + the `fact_fetch` workbench module.**
> The argument is a two-stage chat turn: a cheap pass picks the nodes worth looking up,
> code renders the blob, a second pass writes the reply. It is built as a **module**, which
> is what "argued about, not assumed" looks like in code — the workbench returns its value
> instead of committing it, so nothing is injected and no channel exists. Three findings so
> far, and they belong here because they bear on whether this consumer is worth building:
>
> - **The retrieval axis has to be the mention edge, not node ownership.** §5's subject
>   namespace files a chat fact under *who it is about*, so on this corpus **3 nodes of 107
>   own any claim while 106 carry only mention edges** — every topic a message is actually
>   about (`subjectivity` ×14, `Project Ava` ×5, `autonomy` ×3) owns nothing. A channel
>   keyed on ownership returns nothing for exactly the nodes that matter.
> - **The facet restriction costs almost everything.** Picking the three most-referenced
>   nodes yields **5 knowledge claims against 42 withheld positions**. So the material that
>   would actually help a reply is overwhelmingly the attributed-position layer this
>   consumer excludes. The renderer therefore returns the withheld set separately, counted
>   and correctly attributed, rather than hiding it — the exclusion stays enforced while the
>   argument stays visible.
> - **The fix is upstream, not here.** Both findings are the same fact about the producers:
>   a chat protocol whose subjects were topics as well as people would give this consumer
>   something to retrieve. That is a `chat_facts` change, not a channel design.
>
> The pass emits **node ids, not prose**, and therefore runs with thinking off. That keeps
> the blob faithful (the text comes from `store.claims_for`, so a generation cannot
> paraphrase a claim into something no source said), keeps the facet rule in the renderer
> rather than in a request to a model, and keeps the cost low enough to sit on the chat hot
> path — which this channel must, since it reverses `exchange_anchor`'s deliberate choice to
> keep the query side generation-free. Nothing yet decides when a blob would be injected or
> what it would displace; on the persona-digest precedent (`a42e33b`) it would replace the
> reflection block's `[fact]` slots rather than sit beside them.

### One thing the tree must not become

The `person:_self` subtree is 234 of 349 chat facts. The box already has three artifacts
folding Ava's self-evidence — `reflection_digest` (inside view), `self_portrait` (outside
view), and the `[persona]` ledger under both. A fourth that quietly starts answering "who is
Ava" would extend the self-reinforcement loop `self_portrait.py` was explicitly built to
stay out of. So: `person:_self` is browsable and reportable, and it feeds **no** injected
artifact, ever. Its legitimate use is the position ledger — *what did she claim, when, and
has it changed* — which is a question about the record, not about her identity.

---

## 11. Staging

| stage | contents | GPU | worth on its own |
|---|---|---|---|
| 1 ✅ | `read` + `nodes` + `fold` + exact/alias resolution + CLI dump | none | answers "what is on record about X"; makes P1/P2 visible and countable |
| 2 | inspection tab (consumer 1) | none | resolver errors become reviewable by a human |
| 3 | embedding tier + contradiction/staleness report (consumer 2) | CPU | the Fact Staleness problem gets its first real input — **and, per stage 1's `0 collapsed`, where all the dedup value actually is** |
| 4 | distillation proposals (consumer 3) | CPU | the tree starts improving the authoritative store |
| 5 | portrait feed (consumer 4); LLM tier if measured to be needed | clean base | — |

Off the ladder, and deliberately: **consumer 5 was simulated first** (`graph/blob.py` + the
`fact_fetch` workbench module, 2026-08-11), out of order because a module writes nothing and
injects nothing, so building the *last* consumer as a simulation costs no more than reading
about it and answers questions the earlier stages cannot — chiefly that on this corpus the
channel would retrieve almost nothing, for a reason that lives in the producers. See the
note under §10 consumer 5. It is not stage 5 arriving early; the channel remains unbuilt.

Stage 1 is small — the fold is a few hundred lines of pure code over a 398-row corpus — and
everything after it is optional. That is the intended property: if the tree turns out to be
the wrong idea, it is deleted, and nothing else on the box notices.

---

## 12. Deliberately not decided

- **Topic clustering.** `chat_facts` declined topic tags at extraction time for good
  reasons (interpretation in a witness pass; a per-chat free vocabulary fragments in a
  mixed-language corpus, as `exchange_anchor.normalize_tag` documents) and said clustering
  belongs to the build. The `topic:` node type is the *slot* for that; the clustering
  method is a separate design, and the abstraction mentions are only ~30% of the entity set
  today, so it can wait for a bigger corpus.
- **Cross-lingual node merging.** The corpus is mixed Russian/English, the embedder is
  multilingual, and no cross-lingual merge is attempted at any tier. The alias table is the
  manual answer until there is enough evidence to measure an automatic one.
- **Confidence / source weighting.** `til_facts` refuses a per-fact confidence score on
  principle. The tree keeps that refusal: provenance is recorded exactly and weighting is
  the consumer's call.
- **Whether consumer 5 should exist at all.**

---

## 13. Doc updates when this is built

Per `documentation/README.md` precedence — a new `AVA_STATUS.md` row (Built/Partial and the
GPU-tested caveat), a dated `AVA_CHANGELOG.md` entry citing the commit, and — since the
*Fact Staleness* and *Prompt Composition* problems both gain an input here — the affected
`AVA_OPEN_PROBLEMS.md` sections. `AVA_DESIGN.md` only if the fold becomes load-bearing for
an injected artifact, i.e. not before consumer 4.
