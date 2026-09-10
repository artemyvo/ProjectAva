# SCRATCHPAD — Ava Raw Ideas

Unprocessed ideas, contradictions, and directions for further development.
This is a raw brainstorming sandbox. Once ideas crystallize conceptually, they are added to [README.md](README.md) (the "What") and the [documentation/](documentation/README.md) design set (the "How" / implementation details).

---

## Sandbox / Brainstorming Area

## Design draft: The Lookup Agent & the "Today I Learned" pass

> Status: design draft, not built. Lets Ava's `[ask:search]` items actually get
> answered: a network-only side worker fetches snippets, and a new reflection pass
> frames each result as "I went and looked this up — is it worth keeping?", routing
> the output through the *existing* consolidation WEIGHTS/RAG/RESOLVED filter.

### 1. The idea, and why it fits

Ava already emits `[ask:search]` during reflection — "a public fact you could look
up: a name, reference, slang, in-joke, concept." Today they land live in
`rag_memory.jsonl` and sit there forever: never surfaced (`search` is deliberately
excluded from `surfaceable_questions`), never answered, never evicted. Dead-ended
curiosity. The codebase already anticipates the fix — `reflection_memory.py` holds
`search` asks "for the future lookup agent," and `reflection_writer.py` routes any
untagged ask to `search` by default. The slots exist; they're just unconsumed.

Two new pieces close the loop:

1. **A search worker** — a network-only side process (no GPU) that drains open
   `[ask:search]` items, fetches text snippets, and parks the results in a new store.
2. **A "learning" reflection pass** — a new stage that frames each fetched result as
   *"I went and looked this up. Here is what I found. Is any of it worth keeping, and
   does it answer what I was wondering?"* and runs it through the **same**
   WEIGHTS/RAG/RESOLVED filter consolidation already uses.

Crucial design move: **a search result is just another synthetic source the
consolidation writer already knows how to distil.** No new routing, no new filter, no
new training path. The learned fact flows to RAG (cheap) or weights (rare, deliberate)
by the rule the sleep prompt already states — "when in doubt, send it to RAG." The
`[resolved]` it produces evicts the original `[ask:search]` by `content_key`, exactly
like an open question answered by a chat session.

### 2. The search worker (side process)

**Placement.** Standalone daemon `server/search/search_worker.py`, launched by the
watchdog as a sibling subprocess to the inference server (same pattern as
`watchdog.py`). Touches the network, never the GPU — runs continuously without
competing with inference or training.

**Loop.**
```
every POLL_INTERVAL (default 15 min), if enabled:
  open = ReflectionMemory(...).open_search_asks()      # new accessor: kind==ask & ask_kind==search
  for ask in pick_oldest_unsearched(open, limit=N):    # rotate, fewest-attempts-first
      hits  = provider.search(ask.content, k=TOP_K)     # snippets + URLs
      block = trim_to_budget(hits, SNIPPET_TOKENS)
      write_learning(linked_key=ask.key, query=ask.content, block, sources=[urls])
      append rag_memory op:  {op:"searched", key:ask.key, attempt:n}   # annotate, do NOT evict
  maybe_serendipity()                                  # §6
```

Two design points:

- **The worker never evicts.** It only *annotates* the ask with a `searched` op
  (analogous to the existing `surface` op). Eviction stays the privilege of the
  learning pass via `[resolved]`, so an ask is only closed once Ava has actually read
  and judged the answer — not merely because bytes came back. The `searched` op also
  lets `_fold` carry a `search_attempts` count, so a query that keeps returning
  nothing retires after a ceiling (mirrors the `surface_count` ceiling).
- **The worker does not summarize.** It fetches raw snippets and trims to a token
  budget. Distillation is the model's job in the learning pass — keeping the worker
  GPU-free and dumb, and keeping a human-auditable raw record of what was retrieved.

**Provider abstraction.** `server/search/providers.py` with a thin `SearchProvider`
interface (`search(query, k) -> list[{title, snippet, url}]`). Ship a
SearXNG/DuckDuckGo/Brave adapter; pick via config. Self-hosted SearXNG default fits
the project's self-contained, privacy-minded ethos.

### 3. The learnings store

New op-log: `server/inference/data/hot/memory/learnings.jsonl` (beside
`rag_memory.jsonl`, same `hot/memory/` lifetime).

```jsonc
{
  "id": "...", "ts": "...",
  "kind": "lookup" | "serendipity",
  "status": "pending",                                  // pending → consolidated (folded view)
  "linked_key": "<content_key of the [ask:search]>",    // null for serendipity
  "query": "what \"jumping the shark\" means",
  "result": "<trimmed snippet block>",
  "sources": ["https://…", "https://…"],
  "attempt": 1
}
```

A small `LearningStore` (mirrors `ReflectionMemory`) folds this into pending
learnings; a `consolidated` op (written by the learning pass) marks one done so it
isn't re-studied. Same append-only/fold discipline as the rest of `hot/memory/`.

### 4. The learning reflection pass

New runner phase `_run_learning_for_pending`, slotted into
`ReflectionRunner.execute_run` and gated by a new `learn` stage in the `stages` set
(alongside `reflection`/`merge-rag`/`commit-training`/`train`). Runs like
consolidation, but the source is a learning record instead of a chat session. New
prompt `server/inference/prompts/learning_prompt.txt`:

```
You are not in a conversation. Earlier, while reflecting, you wondered about
something and set it aside to look up. You have now looked it up. Below is the
question you had and the text you found.

TODAY I LEARNED
  You wondered: {query}
  What you found:
  {result}
  (sources: {sources})

Read it as yourself. Decide what, if anything, this changes for you — and where
each thing belongs, using the same discipline as your other reflections:

## WEIGHTS
- [fact] a stable, verified truth worth knowing without looking it up again.
  Reserve weights for what genuinely reshapes what you know. When in doubt → RAG.

## RAG
- [fact] worth surfacing later, with its (trigger: ...)
- [ask:...] if what you found opened a new question (it may need another lookup,
  or only {speaker} can answer it, or it's about your own nature)

## RESOLVED
- [resolved] {query} → what the answer turned out to be (answered in: lookup)
  List this when the lookup actually settled your question, so it stops nagging you.

Think freely first; fill the sections once your thinking settles.
```

Output is parsed and written by the **existing** `ReflectionWriter.write_consolidation`
with `source_session="lookup:<id>"`. That gives us for free:

- `[fact]` → WEIGHTS → ledgered consolidation anchor → eventually trained into the
  adapter, *and* mirrored into `rag_memory.jsonl` as a `from_weights` insert so it's
  recalled at chat time immediately (`_emit_weight_recall`).
- `[fact]`/`[ask]` → RAG insert, indexed into the reflection-memory block.
- `[resolved] … → …` → `evict` op on the original `[ask:search]`'s `content_key` — the
  loop closes. The query and the search ask collapse to the same key because
  `content_key` normalizes them, and the prompt feeds `{query}` (the ask's own text)
  into the RESOLVED line.

So **the "reflection filter" the request asks for is the consolidation filter,
unchanged.** Ava decides what a looked-up fact is worth and where it goes, with the
same weights-are-expensive bias the sleep prompt already enforces.

### 5. End-to-end lifecycle

```
chat → reflection consolidation emits  [ask:search] "what X means"
                                          │  (live in rag_memory, ask_kind=search)
search worker drains it ────────────────►│  fetch snippets, write learnings.jsonl
                                          │  append rag_memory {op:searched}  (not evicted)
next reflection run, `learn` stage ──────►│  learning pass: "TODAY I LEARNED …"
                                          │     ├─ [fact] → WEIGHTS → ledger → train
                                          │     ├─ [fact]/[ask] → RAG
                                          │     └─ [resolved] X → answer  ⇒ EVICT the ask
merge-rag / commit-training / apply ─────►│  promote staging → live (existing stages)
archive ─────────────────────────────────►   reflections/<run_id>/ snapshot (existing)
```

Every arrow after the worker is **existing machinery**. The only genuinely new code is
the worker, the learnings store/fold, the learning prompt, and one runner phase + one
`learn` stage gate.

### 6. Bonus: serendipity (unrelated information)

The worker, at probability `SERENDIPITY_RATE` per cycle (default ~10%), fetches a
snippet *unconnected to any ask* — from a configured serendipity source (Wikipedia "On
this day" / random article, an RSS feed, a trending-topics endpoint) — and writes a
learning with `kind:"serendipity"`, `linked_key:null`.

The learning pass treats it identically, minus the resolution: the prompt variant is
*"You came across this on your own. You weren't looking for it. Is any of it worth
keeping?"* No `[resolved]` (there's no ask), but the same WEIGHTS/RAG routing — so an
unbidden fact Ava finds genuinely interesting can still earn a place. The cheapest way
to give Ava novelty that isn't a reaction to the operator: her wandering.

Guard: serendipity facts are heavily biased toward RAG and a low surface priority, so
random trivia doesn't pollute weights. Provenance (`sources`, `kind:serendipity`) makes
any such item auditable and revertable.

### 7. Surfacing what she learned (optional, recommended)

Because learned `[fact]`s land in `rag_memory.jsonl` exactly like reflection facts,
they're already retrievable in chat via the reflection-memory RAG block. One small
extension: let a freshly-resolved lookup be *proactively* mentioned the next session —
*"I looked into that thing you said about X…"* — by adding a `lookup` lane to
`surfaceable_questions` that picks recently-consolidated learnings whose ask the
operator originally prompted. Makes the loop visible: Ava asked, went and learned, and
brought it back on her own initiative.

### 8. Config, protocol, surfaces

- **`server/search/search_config.json`** (new): `{enabled, provider, provider_opts,
  poll_interval_s, asks_per_cycle, top_k, snippet_token_budget,
  search_attempt_ceiling, serendipity_rate, serendipity_source}`.
- **Watchdog**: launch/supervise the worker like the inference subprocess; add
  `GET /search/status` (last poll, queue depth, last hits) and `POST /search/tick`
  (force a drain). `GET /status` folds in a `search` block.
- **Debug tab** (`debug_widget.py`): annotate each `[ask:search]` with its learning
  status (`pending search` / `searched, awaiting study` / `resolved`) and add a
  "Learnings" group showing the pending queue. Read-only.
- **Sleep tab** (`sleep_widget.py`): add a `Learn` checkbox beside the existing stage
  checkboxes; the learning pass emits the same `phase_started`/`phase_done` events
  (`phase="learning"`) so it renders in the existing event log with no new plumbing.

### 9. Risks & guards

- **Untrusted text reaching weights.** The consolidation filter is the gate: only what
  Ava deliberately tags `[fact]` in WEIGHTS survives, with the prompt's
  weights-are-expensive bias plus a search-defaults-to-RAG nudge. `sources` URLs on
  every learning make any trained fact auditable and (via the reflection archive +
  `adapter_id` repoint) revertable.
- **Stale/empty queries.** The `search_attempts` ceiling retires asks that never
  resolve, mirroring the `surface_count` ceiling — no infinite re-search.
- **Network/provider failure.** Worker is best-effort and isolated; a failed fetch
  leaves the ask open for next cycle. Inference and reflection never block on it.
- **GPU contention.** None by construction — the worker is network-only; only the
  learning *pass* uses the model, inside an explicit reflection run.

### 10. Open choices (decide before promoting to documentation/AVA_DESIGN.md)

- **Worker trigger model** — continuous watchdog-managed daemon (recommended) vs. an
  on-demand drain kicked at the start of each reflection run.
- **Search provider** — self-hosted SearXNG (privacy, self-contained ethos) vs. a
  hosted API (Brave/Tavily; simpler, but keys/cost).

---

## Design sketch: The facts engine as a standalone module

> Status: idea, 2026-08-26. Extract `graph/` into a reusable retrieval engine —
> protocols → fold → hybrid retrieval — usable by other projects (the coding agent
> below is its intended second consumer). Nothing Ava-specific in the core.

### The extraction cut

`graph/` is already close to portable because of its founding discipline: **the
store is the protocol files (`.facts.json`); everything else is derived and
disposable**. That schema (subject, text, fact_class, entities, when, provenance)
is the module's interchange format — not the tree. Three Ava-specific pieces
become the plugin surface:

- **`read.py`** → caller declares the document roots. Keep the lesson as contract:
  a naive glob triple-counted 38/61 protocols and reported duplication as
  corroboration. "What is a source" is the caller's declaration, never a glob.
- **`nodes.py`** → per-lane subject resolvers (chat = person key, TIL = entity
  mention) are already de-facto plugins: each lane registers `subject → node id`
  + an alias table.
- **`blob.py`** → consumer policy (`person:_self` never yields; `position`/`report`
  attributed-only) is Ava's epistemics, not graph logic — stays outside the core
  as the reference consumer.

The facet vocabulary (property/event/position/report/depiction) is arguably
generic epistemics → core; the per-lane `(lane, fact_class) → facet` mapping →
adapter.

### The glossary layer (AltaVista-style inverted index)

Word → posting list of exchanges. The justification, stated precisely: lexical
retrieval didn't lose on the web because it doesn't work — it lost to adversarial
SEO and scale. A personal corpus has neither, so a plain inverted index (≈BM25) is
cheap, exact, rebuildable in milliseconds. The hard parts already exist in
miniature in `exchange_anchor`:

- **Morphology**: `words_match`'s prefix matching (Russian inflection) is the
  placeholder; the real answer is early-2000s Yandex — Segalovich's mystem
  (MLMTA-2003 paper: dictionary lemmatization over Zaliznyak's paradigms +
  **unknown-word guessing by suffix analogy**, which covers coined tokens like
  `крокодильничество` — exactly the glossary's unique-value case). Practical
  tool: pymorphy3 (open-source AOT/OpenCorpora lineage; mystem itself is
  closed-binary, restrictive license). Convention worth copying: under homonymy
  (`стали` → `сталь`/`стать`) index ALL candidate lemmas — recall-biased, right
  for a slot-gated channel (a spurious posting loses the gate; a missed lemma is
  an unreachable fact). Lemmatization is per-language: it cleans the Russian
  half of the index, does nothing for the cross-lingual gap.
- **IDF + stoplist**: `tag_weight`/`build_tag_stats` plus the documented `слушай`
  limit. At full-vocabulary scale the stoplist stops being optional.

- **Adjacency ("glossary by vector")**: the inverted index's one structural lack
  vs dense — adjacency at MATCH time (nothing about injection: RAG never injects
  vectors; the embedder's space dies at the FAISS boundary, the LLM sees only
  text). The fix is a named field — learned sparse retrieval (query expansion by
  word vectors → doc2query → SPLADE: text → weighted bag of vocabulary terms,
  retrieval stays a plain inverted index). In this corpus it collapses to a
  precomputed table: embed the lemma vocabulary once (fastText — subword
  n-grams give vectors to coined/nonce words where dictionaries end), kNN over
  the vocabulary → static `lemma → [(neighbor, weight)]` expansion table,
  neighbors restricted to lemmas occurring in the corpus; query-time admits
  neighbor postings at a weight discount through the same gate (`tag_weight`
  shape). Derived and disposable like the tree. Every match stays explainable
  ("X expanded to Y at 0.7") — the property dense lacks. Bonus: MULTILINGUAL
  word vectors make the same kNN table a machine-generated first draft of the
  long-deferred cross-lingual alias registry (gets `коньяк`↔`cognac`; the coined
  pairs stay the hand-curated residue).

Honest scoping: the glossary is a **precision channel, not the recall fix**. The
box's known pain (cross-lingual / paraphrase misses — the reason fact-nomination
exists) gets *worse* under lexical match. What it uniquely catches is the inverse
failure: rare coined tokens that mean-pooling dilutes — exactly what the anchor
tag channel was built for. The glossary generalizes that channel from curated
tags to all words; it's a fourth channel beside dense, never a replacement.

### The tags layer (folksonomy)

Per-exchange tags exist (anchor pass: ABOUT + TAGS). Promoting them to a
first-class layer hits the problem already declined once (topic tags on
`chat_facts`): **a free vocabulary generated per document with no view of the
others fragments** by language, granularity, morphology. Social folksonomies
converge by mass — thousands of taggers vote; one author minting tags alone gets
no vote. The fix is not a better tagging prompt but the pattern the box uses
twice already: an **offline alias-consolidation pass** (the `fact_dedup` /
`persona_cluster` shape — block, LLM-group, merge into a canonical tag keeping
variants as aliases). Cross-lingual aliasing needs the long-deferred registry —
but in a derived, rebuildable index the registry is just another derived table.

### The generalized pipeline

"Any text → facts + tags → fetch relevant chunks" = a four-layer hybrid
retriever, three of which Ava has: **structured** (protocols → tree →
`fact_fetch`), **topical** (tags), **lexical** (glossary — the missing one),
**semantic** (FAISS). Two Ava-earned decisions to port rather than rediscover:

1. **Slot allocation beats score fusion.** Each channel gets a reserved slot with
   its own admission gate (anchor slot, nomination slot, subject-capped
   reflection slots) — never one fused ranking. Fusion is the classically
   hard, endlessly-tuned part of hybrid retrieval; slots sidestep it and keep
   each channel's failure legible.
2. **Selection by model, rendering by code.** The `fact_fetch` pattern — LLM
   picks ordinals from a catalogue, payload rendered verbatim from the store —
   is what keeps a retrieval layer from paraphrasing its own sources.

Staging: extraction first, deliberately — the glossary and tag indexes want the
same offline derived-and-disposable build machinery as the tree
(`graph_rebuild`'s staleness/rebuild shape), so extracting first means they are
born portable.

### Side-project framing (2026-09-03): generic engine, thrown into Ava later

Standalone crawler → chunker → embedder → vector DB is the commodity dense
layer (industry-standard semantic search; `rag_engine` is already a small
instance). A multilingual embedder dissolves the RU/EN mess for ordinary
vocabulary (parallel-corpus training: `коньяк`≈`cognac`) — but PURE dense
reintroduces every failure the glossary exists to fix: coined tokens diluted by
mean pooling, no exact-match precision, no explainability; cross-lingual dense
is also measurably below monolingual and useless on nonce words. The engine
stays the four-channel hybrid; don't innovate in the dense layer (off-the-shelf
FAISS/qdrant), spend novelty on protocols/glossary/tags/slot-fusion. Embedder
candidate: **BGE-M3** — dense + learned-sparse + multi-vector from one forward
pass (the glossary-by-vector and the vector DB out of one model). Invariants:
the FROZEN component is the embedder, not the LLM (cross-version vectors are
meaningless; the LLM may change freely); every index is derived-and-disposable
so an embedder swap is a rebuild, not a migration. Craft notes: chunking policy
dominates DB choice (bounded overlapping passages port as-is); a single-word
"concept" query is the sentence-embedder weak case — use asymmetric models
(E5 `query:`/`passage:`) or expand the query to a sentence. Ava integration is
then near-free: she has the dense layer; the side project plugs in as the three
channels she lacks.

---

## Design sketch: The project-expert coding agent (the anti-Ava)

> Status: idea, 2026-08-26. Turn a ~31B local model + the Ava machinery into a
> support coder for ONE project (~20k lines, e.g. a Linux VFS driver), fed by the
> review-correction dialog with the developer. Serves through the existing
> OpenAI-compatible API into continue.dev.
>
> **This is the direct opposite of the Ava philosophy, on purpose.** Ava
> accumulates a *subject* — persona, identity, becoming someone; generality and
> voice are the point, and the API never logs so external traffic can't shape
> her. This agent deliberately *erases* subjectivity and generality: forgetting
> other domains is a feature (spare capacity reallocated to the project), there
> is no persona and must be none, the "self" is externalized into a
> model-independent store, and the API traffic **is** the corpus — it wants the
> gossip route's log-and-reflect semantics, the exact inversion of
> `api_http`'s never-log guarantee. Same machinery, opposite teleology: Ava's
> pipeline grows an identity; this one distills a tool.

### What a review correction is (the triage)

Developer corrections decompose into three kinds, served very unequally:

1. *"You take the rwsem for read where it may be held for write"* — **global
   reasoning failure** (call-graph analysis). Capability, capped by the base
   model; neither RAG nor thin SFT fixes it.
2. *"Sem X is write-only after your change — why not a mutex"* — **project fact
   + norm**. Knowledge.
3. *"Assumption X is documented; lean on it, the code shrinks"* — **project
   invariant** not retrieved. Knowledge.

Triage each correction, strongest form first: **(a) checkable** → compile it
into a checker (coccinelle semantic patch, sparse `__acquires`/`__releases`,
lockdep) — the strongest form of "learned from review" is a CI rule that
outlives every model; **(b) declarative** → invariant record in the facts
engine; **(c) taste/idiom** → SFT example. Only the residue reaches weights.

### The expert is the memory, not the model

Layered architecture, model-independent layer first:

- **Symbol-keyed inverted index** — the glossary idea is *stronger* in code:
  identifiers are globally unique tokens, no morphology, no ambiguity.
- **Invariant facts tree** — the facts engine above; `fact_fetch` stage 1 maps
  directly: read the diff, extract touched symbols, inject every recorded
  invariant mentioning them BEFORE the diff is written ("you are about to touch
  `i_rwsem`; on record: …"). Prevents correction kinds 2 and 3 with zero
  training.
- **Correction-derived checkers + regression evals** — see below.
- **The 31B + LoRA** — third layer, not the foundation. The same store serves
  the frontier model for the hard reasoning; expertise survives every model
  swap (cf. the model-migration memory: the store is portable, per-model
  artifacts are not).

### The CoT insight (why this is worth doing at all)

The public code corpus is **answers with the working erased**: final artifacts,
lossy commit messages, verdict-only review threads — and *no negative knowledge*
(the rejected mutex, the deliberately-narrowed generic path never survive into
code). The live correction dialog is training material that cannot be scraped:
the expert's reasoning at the moment of decision, including rejected
alternatives.

Precision: the dialog carries **verdicts, not traces** — "you take the rwsem for
read where write may be held" is the *result* of a silent call-graph walk; the
walk never enters the dialog. Training on the raw correction teaches surface
imitation. This is where the Ava revision machinery transfers: **synthesize the
derivation** — the correction supplies target + WHY, the revision/IDEAL/CoT-graft
pass generates the `<think>` that *arrives* at the correction from the original
context, then human-review + lock (Training review flow, unchanged). The
synthesized derivation must be checked (does the cited lock context exist? does
the named invariant appear in docs?) — a plausible-but-wrong locking derivation
is the most dangerous artifact this pipeline could produce.

Practice multiplier: correct with the **rule, not the instance** ("any function
reachable from the write path must be checked against X") — worth ten instance
corrections, both as SFT row and invariant record. The Meta-note channel is
exactly this input.

### Why thin data suffices (the narrowing argument)

Yield is maybe hundreds of corrections/year — thin for open-domain SFT, but the
target here is not capability, it's **collapsing the hypothesis space**:
general → C → one project. The 31B already knows C and locking; what it lacks is
the project dialect (load-bearing assumptions, primitive usage norms, "we do X
here"), a tiny distribution that a few hundred reasoning-bearing rows genuinely
cover. Catastrophic forgetting inverts into a feature. Ava's training hygiene
matters MORE on thin data: base frozen, fresh LoRA per build over the whole
locked corpus (no compounding), rag-only window, review-and-lock before
training.

### Where code beats chat: validation comes back

Ava disabled the validation probe because "did the persona improve" has no
ground truth. Code has it in layers: every corrected+locked exchange is a free
**regression eval case** (replay context, check the mistake class is gone);
checkers catch their class mechanically. The disabled-validation design project
becomes actually solvable in this domain — likely the first place the five-tier
probe idea earns its keep.

### Flags

- Provider terms: natural SFT targets are frontier-model diffs corrected by the
  developer; provider ToS generally restrict training other models on outputs.
  Corrections + invariants are the developer's own; the diffs themselves need a
  boundary check before becoming targets.
- Corporate-code boundary: the whole pipeline runs on the local box by
  construction (that is half the point of the 31B), but the corpus is
  proprietary — the store, snapshots, and adapters inherit that sensitivity.
