# Associative facts library — design brief

Status: **all four milestones built, 2026-09-09** — every tier of §12 lives in `assoc/`
(store, splitter, kind plugins, L1, dense, L2 codebook, dedup tiers with the equivalence
check, families + authority, contests, typed relations, L4 activation in sqlite, spreading
with learned edge strength, standing needs, the aha and its judge, sense induction and the
three pivot bridges, activation-dependent decay and inferred edges as knobs, `select` with
the fast path, `rebuild` + manifest + drift, the HTTP facade; 126 benches incl. the
three-tier ladder). The §12 gate verdict for tier 3 on the fixtures: **even with tier 2**
(within 0.01 overall, ahead on chat once edge strength is learned), not a clear win, so
the default pull is tier 2 and L4 ships as the knob the design provided for. The real
corpus is still the pending measurement for every gate. Description revised 2026-09-09
after an external review (§0). This is a new
project that starts on the DGX Spark. Ava is its quarry — the ideas and some reference code
come from there — but it is not a fork of Ava's repository, it carries none of her subject
(no training, no persona, no portraits), and it does not depend on her at runtime. The one
chat on this box was a functionality check of the machine, not a corpus.

**Order of work, fixed by the goal:**

1. **A library for proper facts injection.** Text goes in — a chat transcript, a news
   article, a page of technical documentation, all through the *same* mechanism — and
   comes back out, on demand, as the small block of facts a given moment actually turns on.
2. **Applications around it.** Three fix the requirements (§1.0): a two-stage chat exactly
   as Ava runs today (stage 1 fetches the needed context, stage 2 writes the answer — the
   library *is* stage 1); a tech-support bot over a product's documentation; and an editor
   plugin (continue.dev) in which a developer discusses one code project with the model.
   The library is generic — a *kind* of text is a plugin, an *application* is a policy and
   a loop — and Ava is one application among the three, not the library's subject.

What makes the library worth writing rather than pointing a vector DB at the texts is the
puller: **associative, not stateless** — persistent decaying activation, spreading over a
typed fact graph, a glossary keyed on vector cells so Russian and English share keys, the
"Aha!" convergence that connects a standing need to a resource hours after the need was
stated, and word-level pivots in the manner of Дягилева and Башлачёв. Those four
requirements were given in the first conversation and are unchanged; §3–§6 carry them.

---

## 0. Review corrections (2026-09-09)

An external review of the 2026-09-08 draft found seven design faults and four errors in
the activation mathematics. All were verified (the arithmetic reproduces exactly) and all
are corrected in place below; each correction is marked *review correction N* where it
lands, so the reasoning that replaced the original stays legible next to what it replaced.

| # | fault in the first draft | corrected in |
|---|---|---|
| 1 | tier-2 dedup merged on a bag of concepts, blind to order, negation, condition, version | §2.5 — tier 2 nominates; an equivalence check stands before any merge |
| 2 | a bulk-ingested node had no access, so `B = ln 0` and nothing could ever retrieve it | §2.4, §2.6 — finite cold baseline `B₀`; bulk creation access at a cold offset |
| 3 | relations were model-derived, and the automatic fast rebuild ran no model, so a need never appeared | §2.3, §2.8 — the fast scope runs the cached relation pass over new claims |
| 4 | the aha example needed four hops on a two-hop rule and said "summed" where the rule is a minimum | §2.7, §4, §9 — need side 3 hops from need + entities, cached; stimulus side 2; fixture spelled out |
| 5 | a splitter change re-chunked while protocols kept the old chunk ids | §1.4, §1.6, §2.8 — the anchor is the span; chunk id derives from it |
| 6 | newest-wins hid every older version; identifiers assumed globally unique | §1.2, §1.3, §2.5, §2.6 — supersession scoped to a named version; identifier nodes namespaced by document family |
| 7 | the facet filter gated claims while passages and gists were injected whole | §3, §3.2 — passages are attributed quotations with their own eligibility rule; gists are labelled generated text |
| m1 | a degree-one edge multiplied activation by 1.2 and `A → B → A` counted as convergence | §2.7 — fan term normalized by `S`; no node twice on a path |
| m2 | the spacing effect was claimed to fall out of fixed-`d` decay; it runs the other way | §2.6, §8, §9 — claim withdrawn; activation-dependent decay named as the later experiment |
| m3 | PPR was said to reproduce the spread's ranking; it row-normalizes and need not | §2.7, §9 — related scale-out, measured rather than assumed |
| m4 | BGE-M3's sparse head was proposed as a replacement for L2; it is a lexical scorer | §11 |

The review's closing recommendation is adopted as the **first** bench rather than the last:
`bench/baseline` (§9) compares lexical + dense, then the contextual glossary, then
activation, at equal context budgets — the measurement that decides whether the
associative layers retrieve better context or only obey their rules.

**A second pass the same day, over the whole document,** found the corrections above had
left the activation arithmetic on three unrelated scales and a fan clamp that silences most
real nodes, and a few contract-level gaps beside them:

| # | fault | corrected in |
|---|---|---|
| m5 | `A_i` added a log base level, a linear spread sum on an arbitrary IDF scale, and a log authority — `τ` meant nothing across corpora | §2.7 — one scale: seeds normalized, spread added *inside* the log in ACT-R's odds form; `τ` reads as "equivalent to one access N hours ago" |
| m6 | `S = 2` with a hard clamp zeroed every edge type on a node with more than 7 edges of it — a docs section never reached its own claims | §2.7 — soft penalty `fan^−α`, never zero; `S` retired |
| m7 | §2.4 stated ACT-R's additive `S − ln fan` beside §2.7's multiplicative rule | §2.4 marked as the original; §2.7 is the operational form |
| m8 | authority had no teleport term, so a corpus of unique pages got zero everywhere and `ln auth` was −∞; `corroborated(c)` was undefined | §2.5 — damped with a per-kind prior, voters are families, `corroborated` defined, `ln(1 + auth)` |
| m9 | the lexical grounding test required a fact's words in its chunk, while Ava's witness writes 886 of 946 claims in English against Russian conversations | §1.6 — fact language pinned to the text's; overlap measured on lemmas *and* L2 cells |
| m10 | `chars / 3` called conservative for Russian, which tokenizes nearer 2.2 chars per token | §1.5 — per-script estimate fitted at rebuild through the injected tokenizer |
| c1 | a creation access per occurrence made a syndicated claim four times warmer than a unique one | §2.6 — once per independent family |
| c2 | the equivalence check compared argument order on surface strings, which never matches across languages; value conflicts ("Haifa" / "Berlin") were promised as contests with no mechanism | §2.5 — order compared on resolved node ids; v1 contests = polarity pairs + same `(subject, predicate)` with different objects; free-text value conflicts deferred |
| c3 | a judge whose answer failed to parse was recorded as `no`, which blocks the pair for good | §4.1 — `error`, retried, never recorded |
| c4 | the relation pass ran per claim at seconds each — hours per docs tree | §2.3 — per document, one prefill, arguments validated against the claim's own mentions |
| c5 | L1 indexed chunk text only for chats, which contradicts docs QA on an identifier | §2.1 — every chunk's text, every kind |
| c6 | the English lemmatizer, the witness model and `deprecated`'s facet were unnamed | §1.3, §1.6, §2.1 |

**Later the same day the brief was reframed as a generic library** (§1.0): Ava is one of
three target applications, beside a tech-support bot over a product's documentation and an
editor plugin over a single code project. That changed five things structurally — a kind is
a plugin and its witness need not be a model (§1.3, §1.6); `scope` is multi-dimensional and
partitions activation, needs and the aha ledger, not only supersession (§1.2, §2.4, §4);
a document has a key that survives versions and a chunk an identity that survives edits
(§1.4, §2.8 — which revised correction 5's span-derived id); an oversize chunk is split in
v1 rather than discarded (§1.5); and ingest is two steps with the witness deferred (§1.6).
Ava's own rules (`_self`, the person namespace) moved into her application's policy object
(§1.1, §3.2). §7 now shows all three loops and §9 gains a bench per application.
**§12 fixes the order of build**: four milestones, each shipping with its benches, and the
associative layers gated on `bench/baseline` showing they win.

---

## 1. The library's contract

### 1.0 Three target applications

The library is generic: a kind of text goes in through a plugin, and an application decides
what to do with the block that comes out. Three applications fix the requirements, and each
pulls the design in a direction the others do not:

| application | corpus | the stimulus | scope | what it needs most |
|---|---|---|---|---|
| **Ava** — two-stage chat with a persistent interlocutor | her own transcripts, news, wandered articles | the arriving message + recent turns | one user, one box; everything shared | cross-lingual recall (L2), the associative layers (L4, the aha), her own render policy |
| **Tech-support bot** over a product's documentation | docs per release / platform / edition, release notes, KB articles, error-code tables, FAQ; support transcripts under retention rules | the customer's message + the ticket's fields | **many customers at once**; product / version / platform per question; tenant-private conversations | exact hits on identifiers and error codes, versioned supersession, big tables injectable, sub-second pulls under concurrent readers, citations, an honest "not in the docs" |
| **Editor plugin** (continue.dev) over one code project | source files, docstrings, git history, the developer's chats about the code | the open file, the selection, the last edit, the message | one developer; branch / commit as version | a **parser witness** (no model), exact `calls` / `imports` / `defines` relations, chunk identity that survives edits, the identifier as the bridge between chat and code |

What every one of them shares is the contract of §1.1 — documents in, protocols beside
them, a rendered block with provenance out. What differs is *what a kind is* (§1.3), *what
a scope is* (§1.2), and *what changes* (§1.4): a product page is republished per release, a
source file changes on every save, a chat grows while it is open — none of the three is the
immutable text the first draft assumed. Everything Ava-specific — the `_self` rule, the
person namespace, the persona-loop reason for withholding her self-statements — is her
application's **policy object**, not a library constant (§3.2).

### 1.1 What it owns and what it refuses to own

| owns | refuses to own |
|---|---|
| the **document store**: the texts themselves, immutable, chunked, each chunk addressable and pointing back to its document (§1.4) | the model — every generation goes through an injected `generate_fn`, the seam Ava uses everywhere (`configure(generate_fn=…)`) |
| the **protocols**: immutable per-document fact records, every fact anchored to a chunk | the answer — it returns a rendered block and its provenance; the application writes the reply |
| four **derived** layers (§2), all rebuildable from store + protocols | the conversation, the user, the UI, logging policy, training |
| **activation state** — the one non-derived thing, and it is only warmth | which model, which language, which application |
| the **kind registry** — per-kind splitter / witness / relation-extractor plugins (§1.3); the LLM witness is one plugin, the code parser another | the network facade, session identity for a stateless client, tenancy and retention policy — the application's (§7) |
| the **render rules that hold for every application** (a `position` attributed, a `report` attributed to its text, a passage as a quotation, a gist labelled as generated — §3.2) and a **policy object** the application supplies for the rest (which facets render as knowledge, whether an assistant's self-statements do — Ava's `_self` rule lives there) | |

The discipline that made `graph/` extractable is kept without exception (`FACTS_TREE.md`
§2): **the store is the documents and their protocols; everything else is a fold.** A better
resolver, chunker, codebook, or activation formula is a rebuild, never a migration.

### 1.2 API surface (the whole of it)

```
ingest(text, kind, meta, scope) -> doc_id       store + chunk + index the chunks; the witness is deferred (status: pending).
                                                A known meta.key makes a new VERSION, or appends for an append-only kind (§1.4)
extract_pending(generate_fn?)   -> report       run the deferred witness (+ relations) over pending documents (§1.6)
remove(doc_key)                 -> None         drop a key from the current set; its versions stay for explain
document(doc_id)                -> Document     the stored text + meta + chunks + status; chunk(chunk_id) -> Chunk
rebuild(scope=fast|full|deep)   -> stats        fold store + protocols → the derived layers (§2.8)
pull(cue, mode, budget, scope)  -> [Hit]        ranked hits (claim | chunk | gist) with activation + path
select(hits, context, policy)   -> [Hit]        the model picks ordinals; thinking off; code renders
inject(cue, context, budget, scope, policy) -> Block   = pull + select + render, the stage-1 call; touches its picks
touch(ids, scope)               -> None         mark accessed: warms L4 in that scope (a turn, a read, an edit, an ingest)
needs(scope)                    -> [Need]       standing needs in scope; register_need / close_need beside it
aha(scope)                      -> [Candidate]  the ARITHMETIC only: convergence over needs vs current warmth (§4)
judge(candidate, generate_fn)   -> Verdict      the model's verdict on one candidate (§4.1) — the application schedules it
outcome(block_id, signal)       -> None         what became of a block: resolved / accepted / rejected / ignored (§3.2)
pivot(word | context)           -> [Jump]       the word-anchor context switch (§6)
explain(hit)                    -> Path         why this hit: cue → cell/lemma → edges → claim/chunk
```

`inject` is the one call an application must make per turn. Everything else is inspection,
maintenance, or the two optional creative operations. A `Hit` is one of three grains
(§1.4) and always carries `{doc_id, chunk_id}`, so the application can cite, open, or widen
it without a second call. `Block` carries the rendered text, the hits behind it with their
references, the facets withheld (counted, never silently dropped) — the `graph.blob`
contract, generalized — and a `block_id` for `outcome`.

**`scope`** is a small dict of facets drawn from document meta and from the application's
context — `{tenant, conversation, product, version, platform, edition, branch}` — and it
does three things at once: it **filters** candidates (a question about v1 sees v1's norms; a
Windows customer is not handed the macOS procedure), it **partitions state** (activation,
needs and the aha ledger are keyed by scope, so one customer's warmth and standing needs
are invisible to another's — §2.4), and it **resolves supersession** inside the facets it
names (§2.5). An absent facet means *unscoped* on that axis; Ava passes none. **`policy`**
is the application's render policy (§3.2): which facets may render as knowledge, whether an
assistant's own statements may, whether passages of a kind may be quoted. Ava's rules are
one policy; a support bot's — quote the docs freely, never quote another customer's
transcript — is another.

**Two calls are model calls, and both are the application's to schedule.** `extract_pending`
runs the witness for the prose kinds (a parser witness runs inline at `ingest`), and
`judge` runs the aha verdict. Neither is hidden inside a call named like a lookup: a
support deployment runs extraction overnight, a chat application in its idle window, an
editor never (its witness is the parser), and a thinking-on judgement is minutes of the
answering model that no turn should wait on.

### 1.3 Source kinds — one mechanism, several witnesses

A **kind is a plugin**, registered under a name as `{splitter, witness, relations, subject
namespace, classes, clock, redact}`; the library ships several and its core enumerates
none. The ingestion pass is a **witness**, not a summarizer (`chat_facts`' founding rule):
everything the text established, enumerated literally, nothing selected, nothing
interpreted. That rule is what makes one mechanism work for very different texts — the
*reading* differs per kind, the *record* does not — and **the witness need not be a model**:
for prose it is an LLM pass (§1.6), for code it is the language's parser, exact and free,
and for a table it is the table itself:

| kind | witness | subject namespace | classes the witness may emit | entities are | clock |
|---|---|---|---|---|---|
| **chat** | LLM, thinking on | person keys (`normalize_person`); a subject that is not a participant is an entity mention | `standing` / `stated` / `event` | people, places, things named | the exchange's timestamp |
| **news** | LLM | entity mentions, surface form kept | `stated`→`report` / `event` / `standing` | orgs, places, people | article date |
| **article / wiki** | LLM | entity mentions | `standing` / `event` / **`depicted`** (fiction, humour wikis) | as above | fetch date |
| **tech doc** | LLM | identifiers, components, versions — **namespaced by document family** (`ident:<family>:<name>`; `Client` and `connect` are not unique across libraries) | **`spec`** (normative: must/shall), **`procedure`** (steps), **`signature`** (API shape), `standing`, `deprecated` | identifiers — verbatim tokens, no morphology; the bare token indexed in L1, the family resolved from the document it sits in | doc version |
| **code** | **parser** — tree-sitter + import resolution, no model | symbols: `ident:<project>:<qualified name>` | **`signature`**, **`defines`**, **`imports`**, **`calls`**, `docstring` (prose, `standing`), `todo` (a need) | symbols, files, external packages | commit / mtime |
| **structured** | **the structure itself** — a row is a fact, its key column the subject | the key column (an error code, a version, a question) | `standing` (a table row), `procedure` (an FAQ answer's steps), `event` (a release-note item, `when` = the release) | identifiers, versions | page / release date |

Two kinds already have witnesses (`chat_facts_prompt.txt`, `til_facts_prompt.txt`); the
`fold` maps `(kind, class) → facet` exactly as today (`property / position / report / event /
depiction`) and gains **`norm`** (a spec statement — true *by definition of the system*, not
observed) and `procedure`; `signature`, `defines`, `imports`, `calls` and a table row map to
`property` (they are what the artifact *is*), a `docstring` to `property` attributed to its
symbol, `todo` to a need (§2.3), and `deprecated` to `norm` carrying a `supersedes` link to
whatever replaced it. The facet is still what decides what may ever be
rendered as knowledge: `position` never, `report` attributed, `norm` attributed to its
document and version (a norm from v2 docs is not a fact about v3).

**Where the model is not the witness, nothing downstream notices.** A parser's protocol has
the same line grammar and the same parser as an LLM's (§1.6); `grounded` is written `true`
by construction; its relations come out of the same pass exact — `rel:` edges at `s_type`
1.0 with no vocabulary to close and no model (§2.3); and the cost is seconds for a whole
project, which is what makes re-ingest on every save possible (§1.4). A **`redact`** hook per
kind runs before the protocol is written: the code witness drops string literals shaped
like keys and tokens, the support-chat witness drops the customer's identifiers under the
tenant's rule. A fact the store must never hold is one the witness never writes.

Tech docs are the kind the glossary was built for without knowing it: an identifier is the
same token in every language, so L1 hits are exact and L2 has nothing to add — the
scratchpad's observation about code, now a design fact. Its one ambiguity is across
libraries, not languages (`Client`, `connect`, `Config` — review correction 6), so an
identifier node is keyed by its document family and an L1 posting on the bare token
resolves to one of them by the document it sits in. Their protocol gets a `version` on
every claim, and supersession between versions is the first real use of the two clocks —
scoped to the version a cue or an application names (§2.5), never a global newest-wins.

Each ingested document keeps its **gist** beside its protocol (`til_gist` / `.summary.json`):
the gist is what is injected when a whole document is nominated rather than a chunk or a
line of it — the coarsest of the three grains in §1.4.

### 1.4 The document store — separate, chunked, referenced

The texts are kept **apart from the facts**, in their own store, and a retrieval never
returns a whole document: it returns the **chunk** that matters plus a **reference** to the
document it came from. Two reasons, one per side:

- A fact is a witness's *reading* of a text; the text is what the reading can be checked
  against. Keeping them in one store makes the reading look like the record. Two stores
  keep the question "what did the source actually say?" answerable by a lookup rather than
  a re-ingest.
- The injected unit has to be smaller than a document and larger than a fact. The obvious
  case is documentation: *"how do I configure X?"* is answered by one paragraph, not by a
  page and not by three extracted claims — the paragraph, injected verbatim, with a pointer
  to the page for the application to cite or open.

```
documents/<doc_id>/         doc_id = hash(doc_key, version); doc_key = the URL / path / session stem
    document.<ext>          the text of THIS version, never rewritten; a new version is a new doc_id
    meta.json               kind, key, title, url/path, date, version, language, the scope facets
                            (product, platform, edition, branch …), supersedes: <doc_id> | null
    chunks.jsonl            one line per chunk: {chunk_id, structural_path, span, text, kind-specific keys}
                            chunk_id = hash(doc_key, structural_path, normalized text) — NOT the span
    facts.json              the protocol; every occurrence carries chunk_id + span
    summary.json            the gist of the WHOLE document
    status.json             pending | extracted | failed — the witness runs after the store (§1.6)
state/                      activation (sqlite, keyed by scope), the aha ledger, registered needs, aliases
index/<build_id>/           every derived layer of §2; swapped in whole (§2.8)
```

**Chunking is per kind, and the chunk boundary is a fact of the text, not a token budget:**

| kind | chunk = | chunk carries | reference resolves to |
|---|---|---|---|
| chat | one **exchange** (user turn + reply, CoT stripped) | exchange index, speaker, timestamp | the session, positioned at that exchange |
| news / article | one **paragraph** (or a heading + its paragraphs when short) | heading path, paragraph index | the article, at that paragraph |
| tech doc | one **section** at the finest heading level, code blocks kept whole with their lead-in paragraph | heading path, anchor/slug, version | the page at that anchor — a citable URL |
| code | one **symbol** — a function, method, class or module-level block, by the language's parser (tree-sitter); a docstring stays with its symbol | qualified name, file path, line range, commit | the file at that line — an editor can open it |
| structured | one **row group** of a table (header repeated), one FAQ entry, one release-note item, one ticket | the table / list it belongs to, its key column (an error code, a version) | the page at that row |

A chunk too long for the embedder is *windowed* for indexing (bounded overlapping passages,
`rag_engine`'s existing rule) but stays one chunk for injection and reference: windows are an
index detail, chunks are the unit the application sees.

### 1.5 The splitter, and the budget it splits for

The splitter is **structure-first, size-second**, and the two are separate passes so the
first is a property of the text and the second a property of the box.

**Pass 1 — structure (per kind, box-independent).** Parse what the text already says about
its own shape and cut only there:

| kind | structural units, finest first | never cut inside |
|---|---|---|
| chat | exchange | an exchange (a reply without its question is not a chunk) |
| news / article | paragraph → heading section | a paragraph, a quotation, a list |
| tech doc | heading section (finest level) → paragraph | a code block + its lead-in sentence, a table, a definition list entry, a signature + its description |
| code | symbol (function / method / class) → module-level block | a function, a class's signature + docstring, a string literal, a multi-line expression |
| structured | table row group → table; FAQ entry; release-note item | a row, a header + its first row group, a numbered step list |

Each unit records its **heading path** (`Install > Linux > systemd`), so a chunk carries its
place in the document even when injected alone — the thing a bare paragraph loses and the
reason a section is preferred over a paragraph when both fit. Adjacent units too small to
stand alone (a one-line paragraph, a heading with a sentence under it) are **merged upward**
into their parent section rather than left as fragments; a unit is a chunk only if it can be
read on its own. The output is `chunks.jsonl` — a tree flattened in reading order, every
chunk with `{chunk_id, parent_id, heading_path, span, text, tokens_est}`.

**Pass 2 — fit (per box, derived at `rebuild`).** The box has an **injection budget**: the
tokens an application is willing to spend on the block per turn, set by the application
from its window (§7). The Spark makes this a different number than the 24 GB box did — the
122B and gpt-oss-120b load at 32k+ here, so a budget of 8–12k tokens per turn is ordinary
and a whole documentation section fits where before only a paragraph did. So the fit rule
is **a rebuild parameter, not an ingest one**: chunks are stored once, and `rebuild(budget)`
stamps each with `fits: bool` against *this* box's per-chunk ceiling (a fraction of the
budget, so one chunk can never take the whole block). Move the store to a smaller box,
rebuild, and the stamps change; the store does not.

**An oversize chunk is split, not discarded — in v1.** The first draft discarded it from
injection and deferred splitting to a second release. That is wrong for exactly the chunks
the support and code applications value most — an error-code table, a release-notes page,
a 300-line function — so the second behaviour is the first. An oversize unit is cut at its
own next structural level with its *identity repeated on every piece*: a prose section by
paragraph with the heading path on each; a table by row groups with the header row on
each; a function by its top-level blocks with the signature and docstring on each. A chat
exchange is never split (a reply without its question is not a chunk): it stays oversize
and is offered at the claim grain only. The pieces are chunks like any other — their
structural path carries the piece index, they anchor facts, they are injected one at a
time — and the parent unit stays in the tree as the gist-grain fallback. `rebuild` reports
how many units were split and how many were *still* oversize after splitting, since on a
migrated box that number is the first sign the budget is wrong.

**Budget spending (per `inject`).** The budget is split per grain — claims, chunks, gists —
with the chunk share the largest on a docs corpus and the claim share the largest on a
chat one; the application sets the split or takes the kind-mix default. Within a grain,
hits are taken in activation order until the share is spent; a chunk that would overflow
its share is skipped for the next, never truncated (a truncated paragraph is a different
paragraph). Token counts come from an injected tokenizer when the application supplies
one; `tokens_est` is the fallback, and it is **per script, fitted at rebuild** (review
correction m10): `rebuild` runs a sample of 200 chunks through the injected tokenizer and
stores chars-per-token for Latin and for Cyrillic text in the manifest. With no tokenizer
the defaults are 4.0 and 2.2. The first draft's `chars / 3` was called conservative for
mixed RU/EN and is the opposite for the Russian half, which tokenizes nearer 2.2 chars
per token on every family this box runs — an estimate that under-counts by a third is
how a chunk that "fits" overflows the block.

So there are **three grains**, and the puller chooses between them by what was matched:

```
claim   ──►  the fact, one line, rendered from the protocol       (a thing known)
chunk   ──►  the paragraph / exchange, verbatim, + doc reference   (a thing said, in place)
gist    ──►  the whole document's recap, + doc reference           (a thing read, as a whole)
```

Every claim is anchored to a **span** of its document, and through the span to the chunk
that contains it (`occurrence → span → chunk_id → doc_id`), so a fact found by
association can always be widened to the passage that established it, and a passage found
by the glossary can always be narrowed to the facts it established. This is the grain rule
`_query_nominated` had to settle for — chat facts recorded per conversation, so a nomination
could only inject a gist — fixed at the source: the witness records *where* in the text it
saw each fact, and the chunk is that where.

**Chunk identity is structural; the anchor carries both id and span (review correction 5,
revised for mutable documents).** The witness *names* a chunk (§1.6); the protocol stores
for that occurrence the chunk's id and its character span in that document version. The id
is `hash(doc_key, structural_path, normalized text)` — the heading path, the exchange
index, the symbol's qualified name — so a chunk keeps its id across everything that does
not change *it*: a splitter bump that re-cuts a neighbour, a new version of the page that
edits another section, a save that inserts a line above a function. The first fix keyed the
id on the span, which fails on the last of these: one inserted line shifts every span in
the file, and a source file changes on every save. On a rechunk or a new version, an
occurrence whose chunk id still exists is untouched; one whose id is gone is re-anchored by
span overlap to the chunk now covering its span (counted in the rebuild report) and, if the
text under it changed, marked `stale` until the witness re-reads that version. Access
history keys on the chunk id and follows it; a chunk whose structural path survived but
whose text changed gets a new id and **inherits** the old id's history (same place,
edited — editing a function must not cool it), while a chunk whose path is gone is cold.
As first written, `chunks` was invalidated by the splitter version while protocols were
re-extracted only on a witness change — a rechunk would have left every fact's anchor
dangling and every chunk's warmth orphaned.

**Documents change; the store records versions, never edits.** Three of the corpora in §1.0
are mutable: a product page is republished per release, a source file changes on every
save, a chat grows while it is open. The rule that keeps the store honest is that a
*version* is immutable and a *key* is durable: `doc_key` names the thing (the URL, the
path, the session stem) and `doc_id` names one version of it; ingesting a known key
creates a new version with `supersedes` pointing at the last, and the newest version is
*current* for that key within its scope. Three consequences the applications need:

1. A **replaced** page's claims are superseded by the new version's per §2.5, and a claim
   with no successor in the new version yields a fact of its own — *removed in v3* — which a
   support bot is asked about as often as anything still there.
2. A **deleted** key (`remove`) keeps its versions in the store for `explain` and for
   questions about the past; it leaves only the current candidate set.
3. An **append-only** key — a chat — may grow *in place* instead of versioning: the new
   exchanges are new chunks, the witness runs over them with the previous chunk as
   context-only (§1.6), and nothing already anchored moves. That is the one exception to
   "never rewritten", allowed because appending changes no existing span. It is also what
   lets a live session be ingested *while open* instead of at a close it may never have.

How long old versions are kept — and whether old transcripts are kept at all — is the
application's retention policy; the library keeps them until told. That is also the
answer to §11's former question about transcripts: chats are in the store like
everything else, chunked by exchange.

### 1.6 Fact extraction — from chunks to protocol

For the prose kinds the witness pass is the one GPU step of `ingest`, and the only place a
model's output becomes part of the *store* rather than a derived layer (the code and
structured kinds have a parser here, §1.3, and nothing below about prompts, thinking or
grounding applies to them). Everything about it follows from
that: it must be literal, anchored, checkable, and re-runnable.

**Discipline (inherited whole from `chat_facts` / `til_facts`).** The pass is a witness, not
an interpreter: enumerate everything the text established, one line each, select nothing,
infer nothing — the brand of cognac, the flag's default value, the sister's city. It records
what the text says *even where the text is wrong* (a protocol of a text is a record of that
text; how far to believe the source is provenance's job, later). The one who reads is not
in the record (no "this made me wonder"). The measured reason the discipline matters: on
Ava's corpus, curation prompts kept 26 facts from 4 news digests; a witness keeps an order
of magnitude more, and the associative layers need the volume.

**Unit: the document, read in one pass, with the chunks marked.** Not one pass per chunk —
a chunk alone loses its referents (*"it defaults to 30s"* needs the section heading;
*"she moved there in May"* needs the previous exchange), and a chunk-by-chunk loop pays a
context prefill per chunk. The Spark's window is what makes the per-document pass the
default: the text goes in once, each chunk delimited and numbered —

```
[chunk 7 | Install > Linux > systemd]
…text…
[chunk 8 | Install > Linux > systemd > Logs]
…
```

— and **every fact line must name the chunk it came from** (`(chunk: 8)`). A document that
exceeds the extraction window is read in **section groups** — the splitter's tree cut at
the level where a group fits, each group prefixed by the document gist and the previous
group's last chunk marked *context only, extract nothing from it* — so referents survive the
cut and no fact is extracted twice.

**Output contract.** One shared line grammar for every kind, one shared parser
(`chat_facts.parse_facts`, which already takes a per-lane `subject_fn`):

```
[fact] (about: SUBJECT) (class: CLASS) (chunk: N) [(entities: A, B)] [(when: …)] [(version: …)] TEXT
```

- `about` / `entities` follow the kind's namespace (§1.3): person key for chats (with the
  reserved `self`), mention-in-full for texts, identifier verbatim for docs.
- `class` is the kind's vocabulary; an unmarked line is `unspecified`, never guessed.
- `chunk` is mandatory and is what anchors the occurrence — stored as that chunk's **span**
  in the document, from which the chunk id is derived (§1.4). A line without it is
  anchored by **lexical overlap** with the chunks of its group (the fact's words against
  each chunk's words, `words_match`-tolerant) and flagged `anchor: inferred`; a line naming
  a chunk it shares no words with is re-anchored the same way and flagged `anchor: moved`.
  Both flags are counted in the ingest report — a high rate means the prompt, not the text.
- Markers in any order, lenient on a missing one, non-`[fact]` text ignored — a chatty
  generation degrades to fewer facts, never to garbage. **Answer region only**: the pass
  drafts tag-identical lines inside its thinking, and a raw scan stored deliberation as
  protocol on Ava (near-duplicates, literal `</think>` fusions reaching live chat). The
  last line of a truncated generation is dropped: half a fact is not repairable later.

**Per-kind framing, one prompt file each, shared closing.** The framing note is the only
part that differs: chat gets the participants note (canonical spellings, `self` rule,
the reversed-session note for a transcript the assistant opened); news gets *the text
reports*; a wiki/tropes page gets *this may describe an invented world — `depicted`*; a
tech doc gets *normative statements are `spec`, steps are `procedure`, a signature is a
`signature` with its identifier verbatim, and every fact carries the doc's `version`*. The
closing restates the contract and the chunk rule, and on a chat names the participants —
composed at the call, never in the file, so an operator's edited prompt cannot drop it.

**Generation settings, all measured on Ava and kept.** The verbatim loop guard and the
diversity guard are **off** — a run of template lines sharing an `(about, class)` prefix
trips both, and both were caught halting a real pass. The token cap bounds a runaway
instead. Thinking: **off for texts** (a witness has nothing to reason about; the cost is the
pass's whole decode), **on with a ceiling for chats** (resolving *she*/*it*/*there* across
turns is genuinely a reading task), and the ceiling forces the channel closed so a long
thought cannot leave no answer region at all. A generation cut inside its thinking is a
**failed pass**, distinct from an empty one, and is retried once at the next smaller group
size before being reported.

**Ingest is two steps, and only the first is synchronous.** `ingest` stores the document,
chunks it and appends its chunks to L1 and the dense index at once, returning the `doc_id`
with `status: pending` — from that moment the document is retrievable at the **chunk**
grain (a support question hits the paragraph, a code question the symbol) with no claims
yet. The witness runs from a pending queue (`extract_pending`) when the application hands
it a model; the status becomes `extracted`, or `failed` with the report. A parser witness
takes seconds and runs inline. This is what lets a chat application keep the GPU for its
turns, a support deployment ingest a release overnight, and an editor re-ingest a file on
every save with no model in the loop.

**Which model.** Decided, not deferred (review correction c6): the witness runs on the
**fastest model on the box that passes `bench/extract`**, and on the Spark that is
gemma-4-31B — the 122B is the answerer, never the default witness. Extraction is a
witness, not authorship, so it does not have to be the
answering model. The seam is per pass: an application may hand `ingest` a faster
`generate_fn` than it hands `select` or the aha judge. The numbers make this matter on the
Spark — a docs page yields ~50–80 facts at ~40 tokens each, 2–3k tokens of decode; at the
122B's 4.5 tok/s that is ten minutes a page, at a 31B's rate a couple. Ingesting a docs
tree is a batch job and should be run as one, with the report at the end.

**Grounding check (new; Ava has none).** Before a protocol is written, each fact is checked
against the chunk it names: a cheap overlap test first, then optionally a **thinking-off
yes/no pass** per fact or per chunk (*"Is each of these lines stated in this text? Answer
the numbers that are not."*). The overlap test has to survive one measured fact about the
witness (review correction m9): on Ava's corpus 886 of 946 claims are in English against
Russian conversations — the pass *translates* — so a test that requires the fact's words
in the chunk would fail nearly every chat fact. Two things fix it. The prompt **pins the
fact's language to the text's** (a fact about a Russian exchange is written in Russian),
and the parser flags a script mismatch between fact and chunk as `language: drifted`, which
the ingest report counts. And the overlap itself is measured **on L1 lemmas and on L2
cells together**: the fact's content words must share lemmas with the chunk, or quantize
to cells present in the chunk's concept signature (§6) — so a translated fact still
grounds through the cells, while a fact about *systemd* anchored to a chunk that never
mentions it fails both, which is wrong whatever it says.
Lines that fail are kept in the protocol under `grounded: false` — not dropped, because the
record of what the pass produced is itself evidence about the pass — and are never offered
by the puller. On news and docs this is the difference between a fact store and a rumour
store; on chats it catches the pass paraphrasing a position into a stronger one.

**Dedup at the fold, not at extraction — and never by dropping a source.** The pass will
restate a fact that the text restates (a heading and its first sentence; a summary
paragraph), and two documents will state the same thing. The protocol keeps every
occurrence, and the fold collapses them into one claim that *keeps them all* (§2.5) — the
collapse is what produces the source count, and the source count is what ranking runs on.
Nothing is deduped by the model, and nothing is deduped by deletion.

**Re-runnable by construction.** The protocol carries the prompt version, the parser
version and the model id. `needs_reparse` (from `chat_facts`) marks a protocol older than
the current parser, and a backlog drain re-extracts those documents oldest-first — the
self-healing `til_facts.list_backlog` already does, generalized to every kind. A document
is never re-extracted because a *derived* layer changed; only because the *witness* did.

**What the ingest report says.** Per document: chunks, facts, facts per chunk (zero is
normal — a code-only chunk yields one `signature`; forty from one paragraph is a drafting
leak), `anchor: inferred / moved` counts, `grounded: false` count, truncated or failed
groups, the model and versions used. The report is the ingest's product as much as the
protocol is: it is how a bad prompt is noticed before a thousand pages are read with it.

**Relations are not extracted here.** The `(rel: …)` triple is a fold-time pass over
claims (§2.3) until the predicate vocabulary stops moving; the witness stays a witness.
That fold-time pass runs over every **new** claim in the fast rebuild that follows each
ingest batch (§2.8), so a need is derived the moment its claim lands, not at the next
deep rebuild (review correction 3).

## 2. The four derived layers

```
store       documents/<doc_id>/  document + chunks.jsonl + facts.json + summary.json   (§1.4)
                    │
   L1  glossary     word / lemma  ──►  postings (chunk id; claim id)              ← §2.1
   L2  codebook     concept cell  ──►  postings (vector-quantized vocabulary)      ← §2.2
   L3  fact graph   node ─ facet ─ claim ─ occurrence(chunk), + typed edges       ← §2.3
   L4  activation   per-node / per-chunk activation + access history, decaying   ← §2.4
```

L1–L3 are indexes. L4 is **state** — the one thing that is not a pure function of the
store, and the one that makes the puller associative rather than merely hybrid. Postings
point at **chunks** first and claims second: a chunk is where a word was said, a claim is
what a witness made of it, and both are needed because a doc question is usually answered
by the former and an association by the latter.

### 2.1 L1 — the glossary (word → postings)

AltaVista's inverted index as it was: a term maps to where it occurs, with positions. The
"position" is the **chunk id** (and through it the document), plus the claim id when the
word occurs in a fact line — because the unit injected is a chunk, a claim or a gist, never
a document.

- **Tokens**: every word of **every chunk's text, whatever the kind** (review correction
  c5 — the first draft said *for chats, the turn*, which would have left a docs question
  on an identifier with no L1 hit at all), plus every claim, mention and tag. Surface form
  kept; lemma added. Lemmatization is per-language, per token by script: `simplemma` for
  English (and as the fallback for any other Latin-script language), `pymorphy3` for
  Russian, which is the half that needs it:
  pymorphy3, with the Segalovich rule — **under homonymy index every candidate lemma**
  (`стали` → `сталь` and `стать`). Recall-biased on purpose: a spurious posting loses the
  gate downstream, a missed lemma is an unreachable fact. Coined words get the suffix
  guess plus their surface form; identifiers are indexed verbatim and never lemmatized.
- **Weights**: IDF over posting lists — `exchange_anchor.build_tag_stats` / `tag_weight`
  generalized from the tag vocabulary to the whole vocabulary — plus a stoplist, mandatory
  at this scale (the documented `слушай` limit).
- **For**: exact precision, coined tokens that mean-pooling dilutes, identifiers, and §6.
  It is the only layer that knows a word is *one word* across unrelated contexts.

### 2.2 L2 — the codebook ("AltaVista with vectors")

The inverted index again, keyed on a **cell in embedding space** rather than a word.

- Embed the L1 vocabulary — lemmas, multi-word mentions, tags — with a multilingual
  **term-level** embedder (LaBSE, or BGE-M3's dense head; term-level because a single word is
  the sentence-embedder weak case). Cluster into cells: agglomerative at a cosine threshold
  while the vocabulary is small, k-means / PQ when it is not. A cell is a **concept**:
  `{cognac, коньяк, Reviseur XO}` is one key; its postings are the union of its members'.
- **Soft assignment** — a term belongs to its nearest cell and to any cell within a margin,
  discounted: the homonymy rule again, so a quantization boundary cannot hide a fact.
- Neighbour cells at a discount (kNN over centroids) give query expansion over concepts.

This is what requirement 2 buys that L1 cannot: a Russian cue and an English claim share a
**key**, and the match is explainable — `карьера → cell#41 {career, position, opening,
вакансия} → 3 claims` — which dense retrieval never is. Cell ids are unstable across
rebuilds; nothing persistent keys on them (L4 keys on claims and entities).

**Also the cross-lingual alias draft.** An `entity:` mention and an `unknown:` one landing in
the same cell above a high threshold are written to `aliases.proposed.json` for a human to
promote into `aliases.json` — proposal, never merge; the false-merge argument of
`FACTS_TREE.md` §5b stands.

### 2.3 L3 — the fact graph

`server/graph/` as it stands — node → facet → claim → occurrence, mention edges, typed ids
`person:` / `entity:` / `topic:` / `unknown:` — lifted out as the library's core, with two
derived additions:

- **Typed relations.** Today every edge is `claim —mentions→ node`. The activation model
  works on that alone, but the aha example needs *has a need*, *offers a resource*,
  *connected through*. A claim gets an optional `(subject, predicate, object)` from a
  **small closed predicate vocabulary** (`works_at`, `founded`, `offered`, `declined`,
  `interviewed_at`, `wants`, `asked_about`, `depends_on`, `deprecates`, `calls`, …),
  extracted at fold time by a thinking-off pass that emits a predicate id from a list —
  code writes the edge, unrecognised → no edge, the mention edge still stands. The pass
  runs **per document, not per claim** (review correction c4): all of a document's new
  claims in one prompt, numbered, with the predicate list, emitting `n: pred(subject,
  object)` lines — one prefill per document, the same shape as the witness. Both arguments
  must name something in that claim's own `about` or `entities` set (or, for the object,
  an L2 cell — `wants(P, cell{job})` — since a need is often for a concept rather than an
  entity); a line whose arguments name anything else writes no edge and is counted. Derived, so
  a better vocabulary is a rebuild. It runs in the **fast** rebuild scope over new claims
  only (cached per claim key + prompt version), because the standing needs of §4 are read
  off these edges and a need that appears only after a manual deep rebuild is no standing
  need (review correction 3). The witness prompts may emit `(rel: …)` directly once the
  vocabulary stops moving.
- **Relations from a parser.** For the **code** kind the relations are the parser's —
  `calls`, `imports`, `defines`, `overrides`, `tests` — written at ingest, exact, with no
  vocabulary to close and no model; the prose vocabulary above is the model's
  approximation of what a parser gives a code corpus for free. Downstream they are the
  same edges.
- **Need edges.** A `wants` / `asked_about` / `looking_for` claim not yet closed by a
  judge verdict or by the application (§4.1) is a **standing need** (§4), and so is a
  `todo` claim from the code witness. An application may also register needs directly
  (`register_need`) — Ava's open `[ask]` items, a support ticket's open question, a failing
  test. Needs live in a scope (§1.2): a customer's, a developer's, Ava's one.

### 2.4 L4 — the activation state

ACT-R's declarative memory, chosen because it is the association model whose parameters
have actually been fit to human recall, and because its two terms map onto things the
library already keeps:

```
ACT-R's original, for orientation — NOT the operational rule (that is §2.7; review correction m7):

A_i  =  B_i  +  Σ_j  W_j · S_ji

B_i  =  ln Σ_k (t_now − t_k)^−d       base level: every past access k of node i; decay d ≈ 0.5
S_ji =  S − ln(fan_j)                 spreading from active source j, penalized by its out-degree
W_j  =  source weight                 the source's share of the cue, Σ_j W_j = 1
```

Two things carry over from it unchanged — the base level, and the *shape* of the fan
penalty — and two do not: ACT-R's spread is additive on the log scale over one hop, where
this library's spread is a multi-hop attenuation whose sum is added *inside* the log
(§2.7), and ACT-R's `S − ln fan` is replaced by a soft power (§2.7) because the clamp it
implies silences every real hub.

- **Base level** is the persistence: a node touched ten minutes ago is warm, one touched
  last month is cool but not gone. `touch()` appends a timestamp; retrievals, mentions in a
  turn, and ingestion of a new claim about a node all count as accesses.
- **The fan term is the hub guard for free.** On Ava's corpus `person:_self` owns 234 of 349
  chat facts; on a tech corpus a core type or module is mentioned by everything. `− ln(fan)`
  makes spreading *through* a hub nearly worthless while spreading *to* it stays intact —
  IDF's shape, arrived at from the psychology side.
- **Scoped, in two layers.** State is keyed by `(scope, node)`. A **global** layer counts
  every access across the deployment — what is asked about often is warm for everyone,
  which is the popularity prior a support corpus wants and the only layer a single-user
  application ever sees — and a **context** layer (per conversation, per tenant, per
  developer) holds the accesses made *in* that context. `B_i` in a pull is the context
  layer's base level plus a discounted global one (a knob, §8). ACT-R models one mind; a
  support bot is many, and without the split one customer's questions warm the nodes
  another customer is served from, while a standing need from one conversation could
  converge with a resource from another — a privacy leak with a path attached. The aha
  (§4) never reads across a tenant boundary.
- **State is sqlite** (`state/activation.db`): `(scope, node) → timestamps + last A`, one
  writer at a time across the turn thread, the ingest queue and the `touch` calls, readers
  concurrent — a JSON file cannot take that. Losing it loses warmth, not facts. Excluded
  from any snapshot the application takes.
  A node with no recorded access has the finite **cold baseline** `B₀` (§2.6), so a lost
  state file, or a bulk-ingested docs tree, leaves every node retrievable on an exact cue
  — cold, not absent (review correction 2).
- **Decay is wall-clock** and `d` is a knob (§8).

### 2.5 Claims, sources, and authority — dedup that keeps its sources, then PageRank

The lineage this design borrows from is now complete: AltaVista is the index (L1/L2),
PageRank is the ranking. The insight PageRank added to counting was that **a vote is worth
what the voter is worth, divided by how many votes the voter casts** — and both halves are
needed here, because a corpus of chats, news and docs is full of the two things that break
plain counting: copies, and sources that say a great deal about everything.

**Dedup: one claim, all its occurrences.** The fold collapses occurrences into a claim in
tiers, cheapest first, exactly as node resolution does (`FACTS_TREE.md` §5b) — with the
lesson from Ava's first real build in front of it: **the exact tier merged nothing** (0 of
398), so all the value is in paraphrase:

1. exact — casefolded, formatting-normalized text; free; the only tier that merges on its
   own;
2. same subject node **and** same L2 cell set — the claim's content words quantize to the
   same concepts; free once L2 exists, and cross-lingual by construction (the English line
   and its Russian restatement land in one cell set). **Nominates only** (review correction
   1): a bag of concepts cannot see that *"A must run before B"* and *"B must run before
   A"* are opposite instructions, nor a negation, a condition, a role, or a version;
3. embedding of the claim text, blocked by subject node (`fact_dedup`'s subject blocking —
   a grouping sees one block, so blocking order decides what may merge at all); nominates;
4. model adjudication of tier-3 candidates, thinking off, off by default.

**Between nomination and merge stands an equivalence check, in code, that every tier-2/3
candidate pair must pass:** the two lines agree on **polarity** (neither carries a negation
the other lacks — *not / never / no / не / нет / ни* and the kind's own markers such as
`deprecated`), on **version** (both unversioned, or the same version), on **conditions**
(*if / when / unless / если / когда / только* introduce a clause the other must also
carry), and on **argument order** — the sequence of *resolved node ids* (never surface
strings: the English line and its Russian restatement share no strings, which is the pair
this check exists for — review correction c2) is the same in both, so `calls(A, B)` never
merges with `calls(B, A)`. A pair that fails
on polarity becomes a `contests` link (below); one that fails on version becomes
`supersedes`; one that fails on order or condition stays two claims, and tier 4 may be
asked. The check is lexical and cheap, and it errs toward *not* merging: a false merge hides
a fact, a missed merge only costs a source count.

A merged claim keeps **every** occurrence (chunk, document, span, `subject_raw`, the exact
wording) and names one **representative** wording — the one stated by the most sources,
ties to the oldest. Nothing is deleted: a false merge is undone by a rebuild with a tighter
tier, which is only possible because the occurrences are all still there. The rendered line
is the representative; `explain` lists the variants.

**Independence: copies vote once.** The count that matters is *distinct independent
sources*, and the corpus will not supply that on its own: a syndicated article appears on
four sites, a docs tree is mirrored per version, and on Ava's box 38 of 61 protocol files
were copies. So before authority is computed, documents are grouped into **families** —
by `doc_key` first (the versions of one page are one family by construction, §1.4), then
by near-duplicate detection over their chunks (shingling / simhash — cheap, derived) —
and a claim's occurrences from one family count as one vote. The family is reported, not hidden:
"stated by 4 documents (2 independent)".

**Authority: a bipartite PageRank over documents and claims.** Iterate to convergence,
damped, over the graph `document —states→ claim`:

```
auth(c)  =  (1 − β) · prior(kind of c)  +  β · Σ over independent families F stating c   auth(F) / claims(F)
auth(F)  =  (1 − β) · prior(kind of F)  +  β · Σ over claims c stated by F                auth(c) · corroborated(c)

corroborated(c)  =  ln(1 + families(c))       one independent family → 0.69, three → 1.39
β = 0.85; each side renormalized to sum 1 per iteration; the result rank-normalized to [0, 1]
```

A source earns authority by stating things other independent sources also state; a claim
earns weight from the authority of what states it, **divided by how much that source
states** — the fan term again, this time on the source side, so a chat that mentions
everything or a docs index page that lists everything does not become the authority on any
of it. The voter is the **family** (§2.5 above), never the document, so a mirror votes
once. The `(1 − β) · prior` term is the teleport PageRank has and the first draft did not
(review correction m8): without it a docs tree of unique pages — nothing corroborated by
anything — converged to zero everywhere, and `ln auth` in the activation prior was −∞ on
the very corpus the support application runs on. `prior` is per kind: a document that
*defines* the system it documents starts high on its own norms (the rule in the next
paragraph), a chat starts low. Cheap at this scale (thousands of claims, hundreds of
documents), re-run at every rebuild, derived like everything else.

**Priors per kind, and the facet rule stands above all of it.** Authority orders claims
*within a facet*; it never promotes one. Three facets need saying explicitly:

- `norm` (tech docs): the document that *defines* the system is authoritative on its own
  norms by definition — the prior is high and independent of corroboration; across
  **versions** the newest is current and older ones are kept as `superseded_by`, not as
  contradictions. Corroboration across versions is *stability*, worth rendering ("unchanged
  since v2"). **Supersession is scoped, not global** (review correction 6): a cue or an
  application that names a version (`pull(…, scope={version: "v1"})`, or a message that
  says *in 1.x*) is answered inside that version, where the v1 norm is current; newest-wins
  applies only when no version is in scope. A question about the old release must be able
  to reach the old norm, and as first written it could not.
- `report` (news): this is where corroboration is the whole point — one report is a claim
  that a text said so; three independent reports are evidence. Rendered with the count and
  the sources, still attributed.
- `position` (chats): a position held by many is a *popular* position, and remains a
  position. Authority here only orders which of someone's positions leads; it never turns
  "everyone in the corpus says X" into "X". The `person:_self` rule is untouched.

**Contradiction is a first-class outcome, not a merge failure.** Tier 2/3 will also bring
together two claims about the same subject with **opposite polarity** ("defaults to 30s" /
"defaults to 60s"; "she lives in Haifa" / "she moved to Berlin"). Those are caught by the
equivalence check that stands between nomination and merge (above) and are not merged; they
are linked as `contests`, with each side's independent source count and `asserted_at`. The
puller renders a contested claim **with both sides and their counts**, never the winner
alone — except the version case above, where newest-wins is a rule of the kind. This is the
*Fact Staleness* open problem of Ava's, given its first mechanism: the contradiction report
is a fold product, and a chat-lane `position` that changed over time is two positions with
dates, which is the honest record.

**What v1 can actually detect as a contest** (review correction c2). The equivalence check
sees negation, version, condition and order; *"she lives in Haifa"* against *"she moved to
Berlin"* is none of those — it is one predicate with two objects, and only a relation
triple can see it. So a v1 contest arises in exactly two ways: a tier-2/3 pair that fails
the polarity test, and two claims carrying the same `(subject, rel:predicate)` with
different objects where the predicate is single-valued (`lives_in`, `defaults_to`,
`founded` — a flag on the predicate list; `works_at` is not). A value conflict stated in
free text with no relation extracted is not detected, and the design says so rather than
promising it.

**Where authority enters the puller.** Three places, none of them a new channel:

1. **Candidate ordering** in `select` — the catalogue is capped, and the cap truncates from
   the bottom, so the least-supported claims are what fall off. Today that order is
   chat-backed-first-then-oldest-TIL; corroboration is the better rule.
2. **A static prior in activation** — ACT-R has no importance term; add one, the
   `λ · ln(1 + auth_i)` term of §2.7's `A_i`, with `λ` a knob (§8) and `auth` in [0, 1] so
   the term is bounded and never −∞. A well-corroborated claim is slightly warmer at
   rest, which is what "common knowledge" feels like.
3. **The render** — `(3 sources)` on a knowledge claim, `reported by X, Y` on a report,
   both sides on a contest. The count is provenance the reader can act on; it is not a
   confidence score, which `til_facts` refuses on principle and this design still refuses.

### 2.6 Time — four clocks, one of which decays

"Time decay" is four different things in this design, and conflating them is the mistake
Ava already made once and fixed (`[recollection]` needed its own `ts` beside the chat's
`origin_ts`; the chat clock is the session stem, never the pass's `ts`). Named apart:

| clock | what it stamps | who owns it | decays? |
|---|---|---|---|
| **access** `t_k` | every time a node was *used* — injected, touched, created | L4 | yes — this is the ACT-R base level, and the only true decay |
| **assertion** `asserted_at` | when the source *recorded* the claim — the document's own date, never the ingest time | store | no; it is a per-facet **ranking prior** |
| **event** `when` | when the thing *happened* | protocol | no; it places the claim on a timeline |
| **version** | which release of a defining document | protocol | no; newest wins by rule (§2.5) |

Only the first is decay in the psychological sense. The other three are facts about the
record and stay exact.

**Access decay: power law, not exponential.** The base level is

```
B_i = max( B₀ , ln Σ_k (t_now − t_k)^−d )     d = 0.5, t in hours, floor t ≥ 1 min
B₀ = the base level of one access 30 days old  (cold baseline, a knob — §8)
```

summed over every access `k` of node `i`, and never below `B₀` — a node with no access at
all is cold, not undefined (review correction 2: an empty sum is `ln 0`, and no finite cue
activation lifts a node from minus infinity past τ, which is what a bulk-ingested docs tree
and a lost state file both were as first written). Power law is chosen because that is the
shape human recall actually has (Anderson & Schooler's environment statistics — the odds a
thing is needed again fall as a power of the time since it was last needed), and because
of what the *sum* does that a single exponential cannot: frequency and recency combine, a
node touched many times stays warm long after a node touched once has cooled, and a
well-worn node never reaches zero. That last property is what "I just know this" is.

**What the fixed-`d` sum does *not* give is the spacing effect** (review correction m2).
The first draft claimed that ten accesses spread over a month hold more warmth than ten in
an hour. They do not: read a week after the last access, the ten clustered in the preceding
hour sit at B ≈ −0.26 and the ten spread over the preceding month at B ≈ −0.74 — the
clustered ones are warmer, because every one of the spread accesses is older. In ACT-R the
spacing effect comes from Pavlik & Anderson's *activation-dependent* decay (an access made
while the node is already warm decays faster), which is a later experiment (§8), not a
property of this formula. The claim and its fixture are withdrawn; `bench/decay` now pins
the formula's actual behaviour.

**What counts as an access — and what does not.** This is the rule that keeps decay honest:

- **Counts:** being *injected* (the claim or chunk reached the block); being *touched* by
  the application (mentioned in a turn, opened as a reference, cited in a reply);
  **creation** — ingestion records one access at `asserted_at`, so a newly recorded fact
  carries a novelty warmth that fades on the same curve (recently learned things come to
  mind; a month later they are ordinary) — **once per independent family** (§2.5), not
  once per occurrence, or a syndicated article's claim would start four times warmer
  than a unique one for having been copied (review correction c1). A bulk ingest of a docs tree (`ingest(…,
  bulk=True)`) records the creation access at `asserted_at` **minus a cold offset** (30
  days by default, a knob) instead of at the moment of ingest — a thousand pages read in
  an hour are not a thousand things that just came up, but they must not be
  *un*retrievable either, and suppressing the access outright made them so (review
  correction 2). Every node is finite by construction: no accesses ⇒ `B₀`.
- **Does not count:** being *reached by spreading* during a pull. Spreading is transient —
  computed per `pull`, never stored. If reach counted as access, every cue would
  permanently warm its whole neighbourhood and the state would saturate within a day.
  Being a *candidate* the model did not pick does not count either.

**Priming without stored spread.** Carry-over between turns (§3 step 2) is therefore not
residual activation left in the graph; it is that the **cue sources decay**: the seeds of a
pull are the current message *plus* the recent turns' terms, each weighted by the same
power law on its age (`W_j ∝ (t_now − t_turn)^−d`). What was said five minutes ago still
seeds; what was said yesterday barely does. Same curve, applied to the sources rather than
the targets — ACT-R's own account, where priming lives in the buffers, not the store.

**Assertion age as a per-facet prior, never a gate.** Ava's rule (`rag_policy.rank_score`):
relevance gates on the raw match, age only re-orders. Whether a claim gets colder because
it was recorded long ago depends on what kind of claim it is:

| facet / grain | modifier on `asserted_at` age | why |
|---|---|---|
| `property` (chat, article) | **1.0 for life** | a standing truth does not expire; change is handled by supersession and contest (§2.5), not by fading |
| `norm` | 1.0 while current *in the requested scope*; a `superseded` one leaves the candidate list only when no version is in scope, and is always kept for `explain` | version is the rule, not time |
| `report` | slow fade to a floor (~0.3 by a year) | a report's *salience* to "what is going on" fades; its truth as a record of what a text said does not, so it stays retrievable |
| `event` | placed by `when`, not by `asserted_at`; a "now"-shaped cue prefers recent `when` | a fact asserted today about 1998 is not a fact about today (`FACTS_TREE.md` §7) |
| `position` | none; `asserted_at` is rendered ("as of 2026-05") | a position is dated evidence, and the date is the content |
| chunk (exchange) | Ava's tent: verbatim `1 → 0` by ~96 h, gist `0 → 1` then to `0.2` | verbatim recall of a conversation fades to its recap — measured and kept |
| chunk (paragraph, section) | 1.0 | a paragraph of documentation is as quotable next year as today |

All of these are knobs, and all are recomputed at query time, not at rebuild — a deadline
must not wait for a rebuild to take effect (Ava's rule, kept).

**Standing needs: a floor that itself fades.** A need holds `B_i ≥ floor` so it incubates
(§4) — but the floor is not eternal. It halves on a slow clock (~30 days) from the need's
own `asserted_at`, so an unsatisfied need from last spring is dormant, not gone: a fresh
`touch` (the friend brings it up again) resets its accesses and it is back at full floor.
That is what forgetting a favour you owed and being reminded of it looks like.

**The clock is injected.** The library never calls `time.time()`. Every entry point takes
`now`, exactly as `graph_rebuild.rebuild(now=…)` and `store.build_doc(built_at=…)` do — so
the bench can run a five-hour aha in five milliseconds, a migrated box does not re-warm
everything at the moment of import, and a wall-clock lie in one process cannot corrupt
the state. Wall-clock is the *application's* default, and it keeps running while the box is
off: a week's absence cools a week's worth, which is correct.

**Cost.** `B_i` is computed on demand for the nodes a pull actually reaches — seeds plus
their spread — never for the whole graph. Each node keeps its last ~20 access timestamps
exactly and the older tail as a count plus first-access time, folded with the standard
approximation (Petrov's), so a node accessed ten thousand times costs the same as one
accessed ten. State is a few hundred bytes per node.

**Not decaying, deliberately.** Edge strengths (`S_ji`, from co-occurrence and typed
relations) are static per rebuild; time-varying association strength is a later
experiment. The authority prior (§2.5) is static per rebuild. Nothing in the *store*
decays — the protocols and chunks are immutable — so a cold fact is always one `touch`
from warm, and a rebuild can never lose anything to time.

### 2.7 Spreading — how activation flows across edges

Spreading is the one computation the whole design turns on, and the literature has a
compact statement of what makes it work in retrieval rather than run away: Crestani's four
constraints (distance, fan-out, path, activation). Every rule below is one of those four,
named.

**The edges, and what each is worth.** Activation flows over one graph whose nodes are of
five kinds — claims, chunks, documents, entity/person/topic nodes, and needs — joined by
edges of these types, each with a base strength `s_type` (a knob; the defaults encode what
kind of association the edge is evidence of):

| edge | from → to | evidence of | `s_type` | notes |
|---|---|---|---|---|
| `about` | claim ↔ subject node | the claim is about this | 1.0 | |
| `mentions` | claim ↔ entity node | the claim names this | 0.7 | the bulk of the graph |
| `rel:<pred>` | node ↔ node, the claim as its evidence | a typed relation (§2.3) | 1.0 | the only edge that is *about* a connection rather than a co-mention |
| `in_chunk` | claim ↔ chunk | said in the same place | 0.6 | co-occurrence in text — the strongest generic association a corpus offers |
| `adjacent` | chunk ↔ chunk (reading order) | said just before / after | 0.3 | the previous exchange, the preceding paragraph |
| `in_doc` | chunk ↔ document | same document | 0.2 | fan-crushed on any real document (`φ(500) ≈ 0.045`), by design — it exists so a many-chunk hit can surface the gist |
| `same_cell` | claim ↔ claim sharing an L2 cell | says something about the same concept | 0.4 | **seed-time only** (§3 step 1); not expanded at hops, since a cell is a hub by construction |
| `contests` / `supersedes` | claim ↔ claim | the same subject, opposite polarity / later version | 1.0 | both sides of a contest must be reachable if either is |
| `need` | need ↔ the claim/nodes it was raised on | what would satisfy it | 1.0 | |

Edges are undirected for spreading in v1 (a relation's evidence runs both ways: `founded`
reaches the founder from the company and the company from the founder); predicate-specific
asymmetry is a later knob. Static per rebuild — derived from the store like every index.

**One hop.** From an active node `j` with activation `a_j`, along an edge of type `t` to
node `i`:

```
a_i  +=  a_j · s_t · γ · φ(fan_t(j))          φ(f) = f^−α,   α = 0.5 default
```

- `φ(fan_t(j))` is the fan penalty, **per edge type**: `fan_t(j)` counts only `j`'s edges of
  type `t`. Without the per-type split, a person node with 200 `mentions` edges would have
  its three `rel:` edges penalized for the mentions' crowd; with it, the typed relation out
  of a hub stays strong while the co-mention out of it is crushed (Crestani's **fan-out
  constraint**). It is a **soft power, never a clamp** (review correction m6): the first
  draft used ACT-R's `S − ln fan` with `S = 2` and a floor at zero, which is a multiplier of
  `ln(e^S / fan)` — zero for any node with more than `e² ≈ 7.4` edges of one type. On a real
  corpus that silences nearly everything: a docs section never reaches its own 50 claims
  through `in_chunk`, a person with 10 `about` claims is a dead end, and the toy aha fixture
  passed only because no node in it has fan above 2. With `φ(f) = f^−½`: fan 7 → 0.38,
  50 → 0.14, 200 → 0.07, 500 → 0.045 — a hub is expensive to cross, not walled off, and the
  wall the design actually wants is the path rules below plus `ε` and the beam. `φ ≤ 1`, so
  the multiplier on any hop is at most `s_t · γ < 1` and spreading only ever loses activation
  to distance (correction m1 stands; the amplifier the raw term produced on a degree-one
  edge is gone with it). `α` replaces `S` in the knob table; `bench/hub` sets it.
- `γ` is the per-hop attenuation (0.6 default): the same activation two links away is worth
  less than one link away (**distance constraint**, continuous form).
- `+=` is the whole of §4: a node reached by two paths sums them. Not `max`. Convergence is
  the point.

**Bounded expansion.** The spread is a beam search, `k` hops deep:

```
frontier ← seeds                              # {node: activation}, from §3 step 1
for h in 1..k:
    next ← {}
    for j in frontier:
        for (i, t) in edges(j):
            if path_rule(j, t, i) forbids: continue
            if i in path(j): continue         # path rule 4: no node twice on one path
            a ← a_j · s_t · γ · fan_t(j)^−α          # soft fan penalty φ, never zero
            if a < ε: continue                # activation constraint
            next[i] += a ;  record (j, t, i, a) on i's path list
    frontier ← the B strongest of next        # beam width B
    reached  += next
r_i = reached_i                               # seeds normalized so Σ_j W_j = 1  ⇒  r_i ∈ [0, 1]
A_i = ln( e^{B_i} + κ · r_i ) + λ · ln(1 + auth_i)     # for i in reached only (§2.6 cost rule)
```

**One scale (review correction m5).** The first draft summed three unrelated quantities —
a log base level, a linear spread on the seeds' IDF scale, and a log authority — and put
`τ` on the sum, where it meant nothing across corpora and every `bench/spread` expected
value would have encoded an arbitrary number. The rule now is ACT-R's own: everything is
added in **odds** and the log is taken once. `e^{B_i}` is the base level's odds form,
`max(e^{B₀}, Σ_k (t_now − t_k)^−d)` (§2.6, the need floor added there for a need node); the
seeds are normalized so that `r_i` is the *fraction of the cue's activation* that reached
`i`, in [0, 1] and independent of the vocabulary's IDF scale; and `κ` (default 1) says
what a fully-reached node is worth in access terms — at `κ = 1`, being reached whole by the
cue equals one access one hour ago. So `τ` has a reading: `A_i ≥ τ` is *"at least as
available as a node touched once, `e^{−τ/d}` hours ago"* — `τ = −2` is ~55 hours, `τ = −3`
~400. The authority term stays outside the log because it is a prior on the claim, not
evidence of recent need, and `ln(1 + auth)` with `auth ∈ [0, 1]` keeps it bounded.

`ε` (a floor on what is worth carrying), `B` (beam, ~200), `k` (2 in `chat`, 3 in `wander`,
3 need-side / 2 stimulus-side in `aha`) bound the cost to `B · avg_degree · k` per pull — milliseconds on a
corpus of thousands, and bounded on one of millions. The frontier is pruned by *strength*,
not by kind, so a strong chunk hit competes with a strong claim hit on equal terms.

**Path rules (Crestani's path constraint).** Four, all about what an edge may be
*expanded through*, never about what may be *reached*:

1. **`person:_self` is a sink for co-mention.** It may be reached (the aha's R is
   `offered(Noam, position, me)`, and "me" is the node that ties it to the declining),
   and it may be expanded along `rel:` edges (the typed relation is about a connection);
   it is never expanded along `mentions` / `about`. On Ava's corpus it owns 234 of 349
   chat facts — the fan penalty would crush those expansions anyway, but the rule makes it a
   property of the design rather than of the numbers.
2. **Facet gates the render, not the flow.** A `position` carries activation exactly like a
   `property`: "I think Kestrel is a good place" links me to Kestrel whether or not it is ever
   shown, and a `report` connects the entities it reports on. The knowledge-facet rule and
   the `_self` rule are applied in §3 step 4, on what is *offered* — never here, where
   applying them would cut the very paths an association needs.
3. **`same_cell` and `in_doc` are not expanded at hops ≥ 1.** Both are hubs by
   construction (a cell with 80 members, a document with 500 claims); the fan penalty would
   nearly
   zero them, and skipping them saves the iteration. `same_cell` seeds; `in_doc` carries a
   many-chunk hit upward to the gist grain once, at the end.
4. **No node twice on one path.** Edges are undirected, so without this `A → B → A` is a
   legal two-hop path and a node's own activation comes back to it as a second
   "converging" route — an echo counted as convergence (review correction m1). An arrival
   `(j, t, i, a)` is dropped when `i` already lies on the path that brought `a` to `j`; the
   path list every node keeps for `explain` makes that a lookup, not a search.

**What `s_type` does not yet do: learned strength.** In v1 every `mentions` edge is worth
the same; two entities that co-occur in forty chunks are not more associated than two that
co-occur in one. The v2 knob is pointwise mutual information over chunk co-occurrence —
`s(i,j) ∝ log(1 + cooc(i,j)) / (fan(i) · fan(j))`-shaped — which is the corpus's own
measurement of association, derived and rebuildable. It is deferred because it needs a
corpus to measure on, and because the fan penalty already captures the largest effect.

**The two-source spread for the aha.** `aha()` runs the same expansion twice — from the
current stimulus and from the standing need — with **separate** `reached` maps, and takes
per node the *smaller* of the two contributions (§4). Summing within a source is what
makes convergence; taking the minimum across sources is what makes it *two* sources. A
node that one side reached strongly and the other not at all scores zero, which is ordinary
recall, not an aha. Convergence is measured **at the same node** on both sides — the two
spreads are never joined at an intermediate bridge — which is why the two sides get
different depths (review correction 4: as first written both sides had 2 hops, and the
example's own need-side route is 3 edges long). The **need side spreads 3 hops** and is
seeded from the need *and* the subject/entity nodes of the claim that raised it, plus that
claim's cell postings at seed time; the **stimulus side spreads 2**. The need side depends
only on its seeds and the static edge table — never on `B_i` — so its reached map is
computed **once per rebuild per standing need** and cached; an activation event pays only
the stimulus spread and an intersection.

**What is recorded for `explain`.** Every arrival `(j, t, i, a)` is appended to `i`'s path
list; at the end the top three paths by contribution are kept per reached node and the
rest dropped. A path reads back as
`cue "вакансия" → cell{opening, position} → claim#212 —about→ Noam —rel:founded→ Brightmem
—rel:interviewed_at→ P —need→ N`, with each edge's strength beside it. That record is why
this is a beam search and not a random walk. A **personalized PageRank** with restart
`1 − γ` over the same edge table is the *related* scale-out if the corpus outgrows the beam
— related, not equivalent (review correction m3): PPR row-normalizes each node's outgoing
weights into a distribution, where this spread scales an arrival by
`s_t · γ · (S − ln fan)/S` with no normalization across a node's edges, and the two orderings
are not guaranteed to agree. It also returns a stationary score with no path in it, and a
retrieval whose picks cannot be explained is the failure mode this library exists to
avoid. PPR is documented as an option to be *measured* against the beam on `bench/spread`,
not assumed to reproduce it.

**Knobs (§8):** `α` (0.5), `γ` (0.6), `κ` (1), `ε`, `B`, `k` per mode, `λ`, and the `s_type`
table.
All re-fit on the bench; a corpus with different edge densities (docs-heavy vs chat-heavy)
will want a different table, and the rebuild report prints the fan distribution per edge
type so the operator can see which types the penalty is zeroing.

### 2.8 Rebuild — what is recomputed, and when

`rebuild` is the act that turns the store into something the puller can read, and the
discipline of §1.1 is only true if it is cheap to run and safe to run at any moment. So
the design says, per layer, what it is derived from, what invalidates it, whether it can
be updated incrementally, and whether it needs a model. The rule that follows from the
table: **a rebuild is scoped by what changed, never by a clock**, and the layers form a
dependency order that is also the build order.

**The dependency table.**

| layer / table | derived from | invalidated by | incremental? | model? | cost (10k claims) |
|---|---|---|---|---|---|
| chunks (`chunks.jsonl`) | document version + splitter policy | splitter version, or a new version of the document — chunk ids are structural (§1.4), so only the chunks that changed are new; occurrences re-anchored by overlap, protocols untouched | per document | no | ms/doc |
| protocol (`facts.json`) | chunks + witness | witness prompt / parser version; a new document version | per document (pending queue, §1.6) | **prose kinds only** | minutes/doc (LLM); seconds/project (parser) |
| `fits` stamps | chunks + injection budget | budget, tokenizer | per document | no | ms |
| L1 postings | chunks + protocols + lemmatizer | lemmatizer version, stoplist | **append** per document | no | seconds |
| L1 IDF / stats | L1 postings | any posting change | recompute (cheap) | no | ms |
| term embeddings | L1 vocabulary + embedder | embedder id | append new terms | embedder (CPU) | seconds |
| L2 codebook (cells) | term embeddings + threshold | embedder id, threshold; **drift** (see below) | assign new terms to existing cells; **recluster** on drift | no | seconds |
| alias proposals | L2 cells + `aliases.json` | cells, alias table | recompute | no | ms |
| L3 nodes | protocols + `aliases.json` | alias table, resolver version | recompute (cheap) | no | seconds |
| L3 claims (dedup tiers 1–3) | occurrences + L2 cells + claim embeddings | cells, tier thresholds | tiers 1–2 append; tier 3 recompute per subject block that changed | embedder | seconds |
| tier-4 adjudication | tier-3 candidates | candidates | cached per candidate pair + prompt version | **yes**, off by default | — |
| relations (`rel:`) | claims + predicate list | predicate list / prompt version | cached per claim key + version; only **new** claims run | **yes** | seconds/claim |
| document families | chunk shingles | new documents | append; a new doc is compared against existing shingle index | no | seconds |
| authority | claims × independent documents | any of the above | recompute (a few iterations, cheap) | no | seconds |
| contests / supersession | claims + versions | claims | recompute | no | ms |
| edge table | all of L3 | any of L3 | recompute | no | seconds |
| senses (`senses.json`) | L1 postings + chunk signatures | postings of a word changed, cells changed | per word whose postings changed | no | seconds |
| dense index (FAISS) | claim texts + chunk windows + embedder | embedder id | **add** per document; rebuild on embedder change | embedder | seconds |
| needs (standing set) | claims with need predicates + registered needs | relations, satisfaction links | recompute | no | ms |
| rebuild report | everything | — | always | no | — |

Three things fall out of the table:

- **Two layers need a generation model** — the witness (at ingest, not at rebuild) and
  the relation pass — and both are **cached by content key + prompt version**, so a
  rebuild re-runs them only for what is new or whose prompt changed. Everything else is
  stdlib, numpy and the CPU embedder. A "full" rebuild with no version change and no new
  claims touches no model at all; the **fast** scope after an ingest always runs the
  relation pass over the new claims, since the needs of §4 are read off those edges
  (review correction 3 — as first written "fast" ran no model, so a freshly ingested
  *"P wants a job"* could become a standing need only after a manual deep rebuild).
- **The embedder is the one component whose change invalidates almost everything** — term
  embeddings, cells, tier-3 claims, senses, the dense index. That is the scratchpad's
  invariant restated (*the frozen component is the embedder, not the LLM*): swapping it is
  a full rebuild, deliberately expensive, and the manifest names it so the box knows.
- **Nothing in the table is the store.** Delete every row's output and `rebuild` recreates
  it; delete the store and nothing can.

**Three scopes.**

```
rebuild(scope="fast")     after ingest: chunks, fits, L1 append, embed new terms, assign to cells,
                          L3 append-tiers, RELATIONS over the new claims only (cached), families,
                          authority, contests, edges, needs, need-side reach maps (§2.7), dense add.
                          No reclustering; the one model call is the relation pass, bounded by
                          what was just ingested. Seconds plus that pass. Runs automatically at
                          the end of every ingest batch — where a fresh "P wants a job" becomes
                          a standing need.
rebuild(scope="full")     everything derived, from scratch, same versions: recluster L2, re-run
                          tier 3 over every block, re-induce senses, rebuild FAISS. No model
                          unless relations/adjudication caches miss. Minutes at 10k claims.
                          Runs on drift (below), on a knob change, on an alias-table edit, and
                          when asked.
rebuild(scope="deep")     full + invalidate the model caches (relations, tier 4) — for a prompt
                          or predicate-list change. Model-bound: hours over a large corpus, so
                          it is a batch job with a report, never automatic.
```

**When: triggers, not a schedule.** The staleness check is `graph_rebuild.staleness()`
generalized — cheapest signal first, and a **manifest** instead of a single mtime:

```
manifest.json   { versions: {splitter, witness_prompt, parser, lemmatizer, stoplist,
                             embedder, resolver, predicate_list, rel_prompt, knobs_hash},
                  counts: {documents, chunks, occurrences, claims, terms, cells},
                  built_at, budget, drift: {...} }
```

1. **No manifest / unreadable / unknown schema** → full. (All three mean *rebuild*, none
   mean *empty* — Ava's `read_tree` rule.)
2. **A version differs** → the layers that key on it and everything downstream, per the
   table; a witness/parser bump marks *protocols* stale, which is an ingest backlog, not a
   rebuild.
3. **Counts differ from the store** (a document added, removed, or restored from a
   snapshot with old mtimes) → fast for additions; full for removals, since a removed
   document's occurrences are inside merged claims and families.
4. **Drift** → full. The one non-obvious trigger: incremental cell assignment slowly
   degrades — new terms are forced into old cells, and a corpus that started as chats and
   grew a docs tree has a vocabulary the original clustering never saw. Measured at every
   fast rebuild as the fraction of terms assigned to a cell at below the assignment margin,
   and the number of cells whose member count doubled since the last reclustering; either
   past a knob → the next rebuild is full. Reported so the operator sees it coming.
5. **A knob change** (`S`, `γ`, thresholds, `s_type`, the budget) → recompute only what
   reads it: the budget touches `fits`; a spread knob touches nothing derived at all (it is
   read at query time); a dedup threshold touches claims and downstream.

Scheduling is the application's — the library exposes `staleness()` and `rebuild()`, and
the reference application runs them as Ava's `graph_rebuild` idle job does (short idle
window, skips when current, never raises). A failed fold returns an error line and leaves
the previous build in place.

**Atomicity and readers.** A rebuild writes to a staging directory and swaps it into place
in one rename; a reader that opened the previous build keeps reading it to the end of its
call. The manifest carries the build id; a `Hit` carries the build id it came from, so a
pick logged against one build is not applied to another's ordinals.

**What survives a rebuild — the state.** Three things are not derived and must be carried
across:

- **Activation** (§2.6). Keyed on stable ids: entity nodes carry over unchanged, and a
  chunk's id is structural (path + content), so a rechunk or a new document version keeps
  the id of every chunk that did not change and remaps the rest by span overlap, an
  edited chunk inheriting its predecessor's history (§1.4). State is keyed by scope as
  well (§2.4), and a scope is never rewritten by a rebuild.
  Claims do not have stable ids across a re-merge — tier 3 with a new threshold can split
  or join them — so a claim's accesses are stored **per occurrence** and a rebuilt claim's
  access list is the union of its occurrences' lists. A split or a merge loses nothing; a
  false merge undone by rebuild un-shares the warmth cleanly.
- **The aha ledger** (`aha.jsonl`). Keyed on `(need, resource, path-type set)`; needs and
  resources are claims, so the same occurrence-union rule applies, and a verdict on a
  claim that a rebuild split is kept on whichever half owns the occurrence the judge saw.
- **The alias table** and **registered needs**. Inputs, not outputs — the two files in
  the whole tree worth backing up.

**The rebuild report** — the product beside the build, as the ingest report is beside the
protocol. Per scope it prints what changed and what to look at: counts per layer and their
deltas; `fits` discards by document (§1.5); the fan distribution per edge type and which
types the penalty is zeroing (§2.7); unresolved-node fraction and new alias proposals
(§2.2); families found and their sizes, contests found (§2.5); drift measurements and
whether they crossed the knob; words that gained or lost a sense (§6); model-cache hit
rates for relations; wall time per layer. A rebuild that changed nothing prints one line
and is deduplicated in the application's journal.

---

## 3. The puller — `pull` and `inject`

1. **Seed.** Tokenize the cue → L1 lemmas (all homonym candidates) → L2 cells (soft). Each
   hit seeds `W_j` = IDF × cell weight onto its postings. A dense channel (FAISS over claim
   texts, the `rag_engine` shape) seeds too — dense stays the recall floor; the glossary is a
   precision channel, as the scratchpad scoped it.
2. **Carry-over.** Two things make a pull warmer than a cold lookup: the targets' own
   base level `B_i` (what was injected or touched recently is still warm), and the
   **sources** — the recent turns' terms seed too, weighted by their age on the same power
   law (§2.6). Nothing from a previous spread is stored. This is priming, the cheapest of
   the five properties (§3.1) to get.
3. **Spread.** The beam expansion of §2.7: k hops (2 in `chat` mode, 3 in `wander`) over
   the typed edge table, each hop attenuated by the edge's strength, the per-type fan
   penalty and `γ`, pruned by `ε` and the beam. A node reached by two paths **sums** them;
   the top paths are kept for `explain`.
4. **Rank + gate + grain.** Order by `A_i`; drop below τ (ACT-R's retrieval threshold).
   Apply the facet rules **in code, before anything is shown**: `person:_self` yields
   nothing; `position` / `report` / `norm` come out attributed and counted separately.
   Those are claim-grain rules; a passage or a gist is gated by its own rule (§3.2), since
   a facet filter on claims protects nothing that is injected verbatim.
   Then pick the grain per hit: a claim reached through the graph is offered as the claim
   (with its chunk one widening away); a chunk reached through the glossary or dense
   channel with no single claim carrying the match is offered as the **chunk** — the
   documentation case, where the paragraph is the answer; a document whose chunks are
   many-but-weakly hit is offered as its **gist**. Budget is spent per grain, so three
   paragraphs cannot crowd out the one fact the message turned on.
5. **Explain.** Every hit carries its path. Logged beside the injection so a wrong pull is
   legible; shown to the operator in the application's debug view.
6. **Select by model, render by code** (§3.2). `select` hands the ranked list to the model
   as a numbered, three-grain catalogue — thinking off, ordinals back, at most N, `NONE`
   allowed — and the block is rendered from the store; a confident exact hit skips the
   pass. The activation layer makes the catalogue *better ordered and better connected*;
   it must never become a generator. This is `fact_fetch` generalized, and it is why the
   library can be trusted with news: nothing a model wrote reaches the block as fact.

### 3.1 What "associative" means, as a checklist

| property | operationally | today (Ava) | here |
|---|---|---|---|
| graded, persistent activation | warm an hour later, decays, never resets per query | none | L4 `B_i` |
| spreading | a cue reaches things 2–3 links away | one hard-wired 2-hop route | §3 step 3 |
| convergence | two weak paths beat one strong path | none — channels are slot-allocated | summed `A_i`; §4 |
| incubation | a standing need stays slightly active and is re-checked on every stimulus | open asks re-read hourly | need floor; `aha()` |
| lexical pivots | a *word* carries the jump | curated per-exchange tags | L1; `pivot()` |

Slot allocation is kept as the **injection** policy (each channel keeps a reserved slot, no
fused ranking to tune); activation is what *fills* the associative slot.

### 3.2 Selection — how the model picks from the candidates

Ranking (§2.7) answers *what is associated with this cue*. It cannot answer *what this
message turns on* — the difference between a fact that is nearby in subject and a fact the
reply needs — and that judgement is what the selection pass buys. Ava's `fact_fetch` is
the base and every rule of it is kept: **numbered catalogue in, ordinals out, thinking off,
`NONE` allowed, at most N, most important first, rendered by code from the store.** What
follows is what changes when the catalogue has three grains and a budget.

**Why a model at all, and when not.** On a support deployment this pass runs for every
customer at once, so the fast path below, a small `K`, and running `select` on a small
fast model through the per-pass seam are first-class, not optimizations — the answering
model is the wrong one to spend a 10k-token prefill on per turn per customer. That said,
the pass exists for the judgement a score cannot
make: the message says *"it keeps timing out"* and the catalogue holds both *"the default
timeout is 30s"* and *"timeouts are logged at WARN"*; both are about timeouts, one is what
the reply needs. Ava's measured reason still holds too — 886 of 946 claims in English
against Russian conversations — though L2 narrows that gap. But the pass costs a prefill
of the whole catalogue on every turn (~10k tokens on Ava's corpus), so it is skipped on a
**fast path**: when the cue is short and produced an exact L1 hit (an identifier, a proper
name) whose activation leads the second candidate by a margin, and the application has
not asked for the pass, the top hit is injected directly. Docs QA on an identifier is this
case almost always. The margin is a knob; the bench measures how often the fast path and
the pass disagree.

**What the catalogue is built from.** The ranked, gated list of §3 step 4 — so a forbidden
item (`_self`, an ungrounded line, a superseded norm, a `position` under a policy that
withholds them) is **never in the catalogue** and therefore cannot be picked. Safety by
construction, not by instruction — for the **claim** grain. The catalogue is capped at `K` (~60) by activation ×
authority, with **per-grain quotas** so a docs corpus's hundred matching paragraphs cannot
push out the two claims that matter, and a chat corpus's claims cannot bury the one
paragraph. Ordering inside the catalogue is by score, best first — the model reads the top
more carefully than the tail, and the cap truncates from the bottom.

**Passages and gists are gated by their own rule, because a claim filter protects nothing
injected verbatim** (review correction 7). An exchange offered as a passage carries every
turn in it — a user's fact the policy allows beside the assistant's own self-assertions, or
a position the claim grain would have withheld — and a gist is a model's prose over the
whole document. So:

- A **passage is an attributed quotation**, rendered with its speaker labels, its date and
  its reference, never as bare text. Attribution is what makes a `position` safe to show
  at the claim grain, and it is the same thing that makes it safe inside a quotation:
  *"Artemy (2026-05-02): Kestrel is a good place to work"* is a record of speech, not a fact
  offered. The `_self` rule at the claim grain is about the assistant's self-statements
  being offered *as known facts*; quoting her own past turn under its label is not that,
  and Ava's chat channel already injects both dialogue sides this way.
- A passage is **eligible** under a policy only if at least one grounded claim anchored to
  it is eligible under that policy — a chunk whose every claim is withheld is withheld
  whole — and a policy may withhold passages of a kind outright (an application that must
  never quote the user verbatim sets it so). Withheld passages are counted in the `Block`
  exactly as withheld facets are. **The default policy permits attributed positions in
  passages**: on a chat corpus 80% of facts are positions (`FACTS_TREE.md`'s census), so a
  policy withholding them from quotations would end verbatim chat recall. Only what the
  application's policy names is withheld; Ava's `_self` rule is such a naming, not a
  library default, and a support bot's policy names *other customers' transcripts*.
- A **gist is labelled generated text**: rendered as *"a recap of «title» (generated,
  2026-08-14)"* with its reference, never as a fact line, and it passes the same grounding
  check as a protocol (its content words against the document's) or is not offered. It
  inherits its document's attribution and restrictions — a gist of a transcript is withheld
  under a policy that withholds that transcript's passages.

Rendering from the store guarantees a faithful copy; it guarantees neither that the copy
was grounded nor that it was eligible, and those two are decided here, in code, before the
catalogue is built.

**What one line shows.** The picker selects; it does not read the payload. So a line is the
*least* that lets a judgement be made, and the render is the *whole* thing:

```
CLAIMS
 1. [property · 3 sources] The connection timeout defaults to 30 seconds.        (Config > Timeouts)
 2. [event · 2026-08-14] P asked whether Kestrel had openings; there were none.     (chat 2026-08-14 #3)
 3. [position · Artemy, as of 2026-05] Kestrel is a good place to work.             (chat 2026-05-02 #7)
PASSAGES
 4. Config > Timeouts > Retries — "Retries are attempted with exponential backoff, starting…"  (~180 words)
 5. chat 2026-08-14, exchange 3 — P: "Are you guys hiring? I've been…" / me: "Not right now…"  (~90 words)
DOCUMENTS
 6. "Deploying on Linux" — install, systemd unit, log locations.                  (12 sections)
```

A claim shows its representative wording, facet, and either its source count (knowledge),
its date (event), or its attribution and date (position — the attribution *is* what makes
the line safe to show). A passage shows its heading path or exchange position, an opening
excerpt (~200 chars, never the full text), and its size, so the picker can weigh cost. A
document shows its title, its gist's first line, and its extent. **A claim whose own
anchoring chunk is also a candidate is listed under that passage**, indented — picking the
passage injects the verbatim; picking the claim injects one line — so the picker chooses
the grain explicitly rather than getting both.

**What context the picker sees.** After the catalogue: the recent conversation, CoT
stripped, bounded by the same chars-per-token estimate `fact_fetch` uses, ending in the
message — with Ava's fix kept, that on an assistant-initiated session the "message that
just arrived" is the last *real* user turn, never the assistant's own stage direction. For
docs QA the context is the question alone. The closing restates the task after the
material (the last thing in the prompt must be the instruction, not something to
continue) and names the language rule: *judge what an item is about, not which words it
shares with the message.*

**The contract, and its parse.** `PICKS:` then ordinals, one per line, most important
first, at most `N` (8 default; the application may raise it on a wide window), or `NONE`.
No reasoning, no commentary — the pass runs thinking off and greedy, capped at a few dozen
tokens, so a runaway is bounded by the cap and a chatty generation parses to its numbers.
Parse rules: out-of-range and duplicate ordinals dropped; a line that is not a number
ignored; an empty answer is `NONE`. The picks are then **spent against the budget in the
order given** (§1.5): a chunk that would overflow its grain's share is skipped for the next
pick, never truncated; a claim always fits. So "most important first" is load-bearing — it
is the order the budget is spent in — and the picker is told so.

**Two hooks after the pick.** Picked items are **touched** (§2.6: an access) — that is the
one place `inject` touches; the application touches, separately and later, what the
reply actually *used* (cited, opened, acted on), and an access to a claim propagates one
step to its subject node, which is how `touch(Noam)` in §4 comes about from an ordinary
turn. Candidates shown but not picked are not touched. A third hook closes the loop from
outside: `outcome(block_id, signal)` records what became of the block — a support ticket
resolved or reopened, a suggestion accepted or rejected in the editor, a reply the user
answered or ignored — so the pick-rank log below can be re-fit against something real
rather than against the picker's own opinion. And every pick is logged with its **catalogue rank**: a pick
that sat 37th in a catalogue of 60 is a measurement that the ranking put it in the wrong
place, and the distribution of pick ranks over a week is the single best signal for
re-fitting the `s_type` table and `λ` (§8). A ranking that is right puts picks near the
top; when it does, `K` can shrink and the prefill with it.

**Which model.** The seam is per pass (§1.6): selection may run on a faster model than the
answerer, since it emits eight numbers, and on the Spark the answerer is the slow one. The
catalogue prefill is the cost either way, which is why `K` and the quotas are the knobs
that matter, not the token cap.

**Not this pass.** The aha judge (§4) is a different pass with the opposite settings —
thinking *on*, one yes/no plus a line of why, over two named items and their paths, paid
rarely. Keeping them separate is deliberate: selection must stay cheap enough for every
turn; judgement must stay expensive enough to be right.

**Failure paths, all soft.** No model, no catalogue, a failed generation, an unparseable
answer: the block is empty with a named reason, never a failed turn — and the reason is
distinguished from *picked nothing*, since a channel whose only failure is silence is one
nobody notices was dead (Ava learned this the hard way; the facts block above every reply
is the fix, and the application should show the same).

---

## 4. The "Aha!" — incubation and convergence

The example, decomposed into what has to be on record for it to fire:

| # | fact | facet / relation | enters |
|---|---|---|---|
| N | friend P asked whether Kestrel has openings; none | `event`; `asked_about(P, job@Kestrel)` → standing need `wants(P, job)` | the call |
| R | Noam offered me a position at startup S; I declined | `event`; `offered(Noam, position@S, me)`, `declined(me, …)` → the position is free | weeks earlier |
| B1 | P interviewed at Brightmem, chose elsewhere | `event`; `interviewed_at(P, Brightmem)` | years earlier |
| B2 | Noam founded Brightmem | `property`; `founded(Noam, Brightmem)` | a news/wiki source — or the model's own knowledge |

At the call, `pull("openings at Kestrel")` seeds cell{opening, position, job, вакансия} and
`entity:Kestrel`. R shares the cell — but R is cool (weeks old), Kestrel and S are unconnected,
and R is one hop from the cue across a cell with dozens of postings. Below τ.
**Correct** — the human did not make the connection at the call either. The fast rebuild
after the call's ingest runs the relation pass over the new claims (§2.8), so
`asked_about(P, job@Kestrel)` becomes the standing need N there and then, and N's need-side
reach map (below) is computed and cached in the same rebuild.

Hours later something unrelated mentions Noam — a message, a news line the application
ingested, an idle read. `touch(Noam)`. Now:

- **The standing need is never cold.** Needs are the one node kind with a **floor** on
  `B_i` — incubation is literally "keep the question slightly active". `aha()` runs on every
  activation event: for each standing need, take its cached **need-side** reach map (3
  hops, seeded from N *and* the subject/entity nodes of the claim that raised it — P, Kestrel —
  plus that claim's cell postings at seed time) and spread **2 hops from the stimulus**,
  separately; look for nodes reached by **both**. Convergence is measured at the same node,
  never by joining two half-paths at a bridge (review correction 4 — as first written both
  sides had 2 hops and the need-side route below needs 3).
- R is now reached on both sides. Stimulus side, 1 hop: `Noam —about→ R` (fresh, strong).
  Need side, two routes that **sum**: the seed-time cell route `N's claim —same_cell→ R`
  (weak) and `P —rel:interviewed_at→ Brightmem —rel:founded→ Noam —about→ R` (3 hops over
  typed, low-fan edges: moderate). B1 and B2 are what put the second route on record.
- Candidate `(N, R, paths)` is emitted with strength = the *smaller* of the two sides'
  contributions — both routes must carry; one strong route is ordinary recall, not an aha —
  and appended to `aha.jsonl` so the pair is not raised twice. At the call the stimulus
  side carried only the weak cell route, so the minimum sat under the convergence floor;
  now the stimulus side is strong and the need side moderate, and the minimum clears it.
- **Then, and only then, the model judges**: *"Does R bear on N?"* Thinking on, yes/no plus a
  line of why. What happens on yes is the **application's** decision — the chat app composes
  a message (*"I could put you in touch with Noam…"*), a docs assistant raises a note. The
  library stops at the candidate and the verdict.

The trigger is **the stimulus, not the clock** — the aha arrives when Noam comes up, not on
the hour — and the judge is paid only when two paths converge, so it fires rarely and
always with a path to show.

**Scope bounds the aha, and the judge is the application's to schedule.** Needs, resources
and the ledger are read within one scope (§2.4): a support customer's standing need
converges with the documentation and with that customer's own conversation, never with
another customer's; an editor's need — a `todo` claim, a failing test the application
registered — converges with the project, and a commit touching code near an open `todo`
is the editor's version of this example. `aha()` is arithmetic and cheap and may run on
every touch; `judge()` is the model, and a thinking-on verdict is minutes of the answering
model that no turn should wait on — so the application calls it from wherever it keeps
its idle work (§1.2).

**B2 and the model's own knowledge.** Noam ↔ Brightmem may be on no source. The model knows
it. Proposed, off by default: when two nodes are co-active above τ with no recorded path, a
background pass asks whether the model knows a connection and writes it as an `inferred`
edge, `source: model` — attributed, low weight, rebuild-safe, never the *sole* path to an
aha. The provenance rule that `til_facts` applies to every text applies to the model too.

### 4.1 The aha judge — what the model sees and decides

Everything upstream of the judge is arithmetic: two spreads, a minimum, a threshold, a
ledger check. The judge is the first and only place a model looks at the pair, and it is
built on Ava's outreach decision pass (`outreach_prompt.txt`: *raise this now?* →
`DECISION` yes / no / resolved / asked, with the standing-questions block that cured two
observed pathologies). Opposite settings from selection, on purpose: **thinking on, under a
ceiling; greedy; one verdict; paid rarely.**

**What it sees — in this order, each part labelled.**

1. **The need, as a passage.** Not the claim line alone — the **chunk** that raised it,
   verbatim (P's actual words: *"Are you guys hiring? I've been looking since…"*), with
   its date, who said it, and how old the need is. A claim is a witness's reading; the
   judge is deciding whether a real request is met, so it reads the request. Same rule for
   a need the application registered directly: its own text.
2. **The resource, as a passage.** The chunk behind R, verbatim, with its date and source —
   *"Noam offered me the CTO seat at S; I said no, the timing's wrong"* — because the
   nuance the decision turns on (*declined*, *timing*, *no openings right now* vs *never*)
   lives in the passage and is exactly what a claim line flattens.
3. **The paths, both of them.** The two spreads' top paths from stimulus to R and from N
   to R, rendered as `explain` renders them, each edge with its type and strength, and
   each **bridge node's claims** shown as one line apiece (*Brightmem — P interviewed there
   (2023); Noam founded it*). The judge is told which edges are from a text and which, if
   any, are `inferred` from a model (§4, B2) — and that an inferred edge is a guess it may
   discount.
4. **The clocks.** `asserted_at` on every item, `when` where there is one, the need's age
   and its floor state (§2.6). *"The offer was 6 weeks ago; the request was 5 hours ago."*
5. **Prior verdicts on this need.** Every earlier candidate for N and what was decided —
   the `aha.jsonl` ledger filtered to N. This is the standing-questions block transplanted:
   without it, each judgement is the first time, and a need that has already been matched
   to a resource is matched again under a new path in other words. With it, *"already
   connected P to Y last week"* is a fact the judge has in front of it.
6. **What the application is doing now** — one line the caller supplies: *in a chat with
   Artemy*, *reading the morning's news*, *idle*. The judge does not decide delivery, but
   the same connection is a different verdict mid-conversation and mid-nothing.
7. **The contract**, last.

Nothing from the graph beyond the two passages and the bridge lines. RAG is **off** — the
judge is looking at a specific pair, and Ava's experience with the outreach pass RAG-on was
that the memory block turned into a queue the model felt it was jamming.

**What it decides — one of four, plus one sentence.**

```
VERDICT: connect | satisfies | no | stale
LINK:    <one sentence: what the connection is, in plain words, or —>
```

| verdict | meaning | what happens |
|---|---|---|
| `connect` | R bears on N: the need is not met by R, but R is a way toward meeting it | candidate handed to the application with `LINK` as its origin note; the application decides whether and how to say it |
| `satisfies` | R **answers** N outright (the docs paragraph the question was about arrived; the friend already got the job) | the need is **closed** (`satisfied_by: R`, off the standing set); the application is told, and may say so or not |
| `no` | not actually related — the paths were coincidence of vocabulary or a hub | logged; the pair is not re-judged unless a new path *type* appears or N is touched again |
| `stale` | N is no longer a need — its passage, its age, or R itself says so (P found a job; the docs question was withdrawn) | the need is closed as `stale`; distinct from `satisfies` because nothing answered it |

`LINK` is the judge's reading, in prose. It is **labelled as such** wherever it goes — the
application's origin note, the ledger — and it **never enters the store**: not as a claim,
not as an edge (unless the `inferred`-edge experiment is on, in which case it is written
there with `source: model` like any other model-asserted link). The judge is allowed to
*use* what it knows about the world to decide (that is what B2 is for); it is not allowed
to *record* it through this pass.

**Settings.** Thinking on — this is the one pass on the box that is a judgement rather
than a lookup, and the reasoning is where *declined* meets *timing's wrong* meets
*interviewed there in 2023*. Under the thought ceiling (`generation._reflect_think_ceiling`'s
rule: a sane thought that outgrows its budget must be forced closed, not lost); a pass cut
inside its thinking is a failed judgement, retried once. Greedy, temperature 0 — a verdict
should be reproducible on the same pair. Answer region parsed by label, last occurrence
wins. A missing or unparseable `VERDICT` is an **`error`**, retried once and otherwise
reported — never recorded as `no` (review correction c3): a `no` is remembered and blocks
the pair until a new path type appears, so a parse failure written as `no` would silence
a real aha for good. The safe default on a *parsed* verdict is still the cautious one — a
missed aha costs a delay, a false one costs a message.

**Rate and ledger.** At most `M` judgements per activation event (3), taken by convergence
strength; the rest wait for the next event, where they will have moved if they were real.
`aha.jsonl` keys on `(need, resource, path-type set)`: the same pair over a new *kind* of
path is a new candidate, the same pair over more of the same is not. A `connect` verdict
handed to the application is recorded whether or not the application acts on it — that is
the application's business, and the judge must not be asked twice because it declined
once. Ava's `reachout_gate` shape (one unprompted message per window, backoff on silence)
belongs to the application; the library's rate limit is on *judgements*, not messages.

**What the application receives.**

```
Candidate{ need, resource, verdict, link, strength,
           paths: [Path], passages: {need: Chunk, resource: Chunk},
           bridges: [(node, claim_line)], prior: [earlier verdicts on need] }
```

Everything the judge saw, plus its verdict — so the application can show the operator the
same material and the chat can compose from the passages rather than from `LINK` alone.

**Which model.** The answering model, or a stronger one — never a weaker one than the
answerer. Selection can run on a fast witness; a false aha is a message sent, and this is
where the box's best judgement is worth its decode.

---

## 5. Cross-lingual, concretely

- L2 is where RU/EN merge; L1 stays monolingual per language by construction. A cue that
  hits only in L1 is itself a signal — the term is coined, or a name, or an identifier.
- Cross-lingual recall is L2's job and dense's; the judge prompts stay language-agnostic as
  `fact_fetch_prompt.txt` already is ("judge what a fact is ABOUT, not which words it shares").
- The alias draft (§2.2) is the first automatic answer to the merge `FACTS_TREE.md` §12
  deferred; it stays a proposal file until a human promotes an entry.

---

## 6. Word pivots — the Дягилева / Башлачёв channel

The lyric device: a single word is held constant while everything around it moves — *время
колокольчиков*, one word carrying one meaning into a frame that gives it another. The
requirement is the *capability*; the chat application need not use it. It is lexical end to
end: the bridge is a **word**, never a meaning, which is why it lives in L1 and is the one
route between contexts no embedding channel can provide.

**Senses are induced from contexts, not read off a vector.** A term-level embedding gives
`ключ` one vector, so L2 cannot see that it is two words. What can see it is where the word
is *used*: every posting of `ключ` sits in a chunk, and every chunk has a **concept
signature** — the set of L2 cells its content words quantize to. Cluster a word's postings
by their chunks' signatures (agglomerative, cosine over the signature vectors; classic
word-sense induction) and the clusters are its senses, each described by the cells that
dominate it:

```
ключ   sense A  {door, lock, apartment, lost}        14 postings
       sense B  {spring, water, forest, cold}         3 postings
стали  sense A  {steel, alloy, industry}              6 postings
       sense B  {become, change, growing}            22 postings   (лемма: стать)
```

Homonymy handled at lemmatization (§2.1) lands here for free — `стали` was indexed under
both lemmas, and the two lemmas' postings simply fall into different clusters. A coined
word has senses wherever the corpus used it. Computed at `rebuild` for every word with
enough postings to cluster (≥ 6) and stored as a derived table `senses.json`; a word with
one cluster has no pivot.

**Choosing the anchor.** `pivot(context)` scores the words in the **warm set** — the current
cue's terms and the words of recently touched chunks and claims — because an anchor must
already be *in play*; a word imported from elsewhere is a change of subject, not a pivot.
For each such word `w` with ≥ 2 senses:

```
anchor(w) = act(w)                     it is in play — its activation in the current context
          × split(w)                   polysemy: cosine distance between its two densest senses
          × idf(w)                     a rare pivot is a sharper one; stoplisted words score 0
          × (1 − act(sense_B))         the far sense must be COLD — if it is already active
                                       there is nothing to switch to
```

The last factor is the whole of "switch": a pivot is a move to a sense the context does not
currently hold. The top few anchors are returned with their sense descriptions, and one is
chosen — by the caller, or by the operation's own default (the highest score).

**The jump.** Seed activation on the far sense's postings — the chunks and claims where `w`
is used in sense B — and spread 1–2 hops in `wander` mode (§2.7). The result is a `Jump`:

```
Jump{ bridge: "ключ", from_sense: {door, lock, …}, to_sense: {spring, water, …},
      distance: 0.81, hits: [Hit], paths: [Path] }
```

read back as *pivoting on «ключ»: from the lost-apartment-key conversation to the spring at
the dacha*. The hits are ordinary three-grain hits with references; the bridge word and both
sense descriptions ride the result so the switch is visible, and `explain` on any hit starts
at the bridge.

**Second and third bridge kinds — off unless asked.** The anchor need not be the identical
word; Башлачёв's device is as often a **sound** as a sense:

| bridge | `w → w'` when | machinery | example |
|---|---|---|---|
| `sense` (default) | same lemma, different induced sense | L1 + `senses.json` | ключ / ключ |
| `root` | same stem or derivational family | pymorphy3 for RU; a stemmer for EN | колокол → колокольчик → колокольня |
| `sound` | rhyme (shared ending ≥ 3 chars), or edit distance ≤ 2 between lemmas of ≥ 5 chars, or a cross-script transliteration pair | string ops over the L1 vocabulary | магазин / magazine; тоска / доска |

Each is a different bridge type in the result, scored the same way with `split` replaced by
the string relation's own strength. `root` and `sound` reach words *not* in play, so their
factor `act(w)` is the anchor's and `(1 − act(w'))` the target's. These are cheap — the L1
vocabulary and a stemmer — and are exactly the paronymic attraction the lyrics run on.

**What the model does with it, and what the library does not.** The library returns the
`Jump`; it composes nothing. In `wander` mode the application hands the model the current
context, the jump's hits, and the bridge named — *you were on «ключ» as a key; here is
where else that word lives* — and asks for whatever the creative pass wants (a reflection,
a question, an opener). Whether a pivot is ever *taken* is the model's choice and the
application's policy; the chat default never offers one. A pivot is not an access (§2.6):
the hits are touched only if the application uses them.

**Cost.** Sense induction is a rebuild-time pass over words with enough postings — small,
and derived like every index. `pivot()` itself is a scoring pass over the warm set plus one
spread: milliseconds.

## 7. The three applications — one library, three loops

**Ava — the two-stage chat.** Exactly her shape, with the library as stage 1:

```
message arrives
  │
  ├─ stage 1  block = lib.inject(cue=message, context=recent turns, budget, scope={}, policy=ava)
  │             (inject touched its picks)
  │             lib.aha()  → candidates → queued for judge() in the idle window → app decides whether to say it
  │
  └─ stage 2  reply = model(system + block + history + message)
                  lib.touch(what the reply cited)                        ← the turn warms what it USED
                  lib.ingest(session, kind="chat", meta.key=stem)        ← append-only, while the session is open
                  lib.extract_pending() in the idle window               ← the LLM witness, off the turn path
```

What the application owns and the library does not: the conversation, the user, the system
prompt, logging policy, what a `connect` from the judge turns into, and the policy object
(`_self` withheld, positions attributed, both dialogue sides quotable). The debug surface is
the `Block` plus `explain()` per claim — Ava's *facts block* above the reply, now with the
path.

**The support bot — many customers, one corpus.**

```
release lands     lib.ingest(page, kind="tech_doc" | "structured", meta={key: url, version, platform, edition})
                  lib.extract_pending() overnight;  lib.rebuild("fast")
                  → pages retrievable at the chunk grain from the moment they land; claims after extraction

question arrives  scope = {tenant: customer, conversation: ticket, product, version, platform}
                  block = lib.inject(cue=message + ticket fields, context=this ticket's turns, budget, scope, policy=support)
                  reply = model(system + block + turns);  the reply CITES block.hits[].reference
                  lib.touch(what was cited, scope);  lib.ingest(ticket transcript, kind="chat", scope) under retention
ticket closes     lib.outcome(block_id, resolved | reopened)
```

What this loop exercises that Ava's does not: the exact hit on an error code or identifier
and the fast path that skips `select` for it; supersession *inside* the customer's version
and platform; a table injected by row group; `B_i` from the global layer (what everyone
asks) plus the ticket's own; needs and the aha fenced to the tenant; and an honest empty
block — a question the docs do not answer must come back as *not in the docs*, never as a
paragraph about something adjacent. The policy quotes documentation freely and never
another customer's transcript.

**The editor plugin — one developer, one project.**

```
file saved        lib.ingest(file, kind="code", meta={key: path, version: commit | mtime})   ← parser witness, inline, seconds
                  lib.rebuild("fast")                                                          ← relations exact; no model anywhere
editor events     lib.touch([open file's chunks, the selection's symbol, the last edit], scope={branch})
message arrives   block = lib.inject(cue=message, context=recent turns + the open file's symbol, budget, scope, policy=dev)
                  reply = model(system + block + turns);  lib.touch(what it cited)
                  lib.ingest(chat, kind="chat", meta.key=session)   ← the developer's decisions about the code, LLM witness later
```

What this loop exercises: the parser witness end to end; `calls` / `imports` / `defines` as
exact edges — *what calls this?* is a one-hop spread, not a search; the identifier as the
bridge between the chat lane and the code lane (the developer says `connect`, the parser
recorded `connect`, one node); chunk identity surviving a save that shifts every span; the
recently edited files warm by construction, which is L4 doing something visibly right for
one user; and the commit as the version clock. A `todo` claim is a need, and a commit near
it is the editor's aha.

**The facade.** The support bot and the editor are remote clients, so the first
application-side artifact is an HTTP layer beside the in-process library: an
OpenAI-compatible chat endpoint that runs the two-stage loop (Ava's `api_http` shape) and a
context endpoint that returns a `Block` for a client that composes its own prompt
(continue.dev's context-provider shape). A stateless client resends the whole conversation
each request, so session identity is **derived by prefix match** on the turns —
`generation._GossipSessionLog` already does exactly this — and the derived session is the
`conversation` facet of the scope. Tenancy, retention and rate limits live here, not in
the library.

A news reader is Ava's loop with no reply — reading is `touch` + `aha()`, which is how a
newly ingested article can connect to a standing need without anyone asking.

---

## 8. Parameters — all knobs, all derived

`d` (decay, 0.5), `α` (fan softness, 0.5), `κ` (what a fully-reached node is worth in
accesses, 1), `τ` (retrieval floor, in the hours reading of §2.7), hops per mode, cell
threshold, soft-assignment margin, neighbour-cell discount, need floor, convergence minimum,
per-facet render caps. None trained; every one re-fit by rebuild against §9. The formula is
a knob too — if `fan^−½` is still too harsh on a corpus with three people in it, a softer
`α` is one number and a rebuild. `B₀` (the cold baseline) and the bulk-ingest cold offset are
knobs too (§2.6). Activation-dependent decay — Pavlik & Anderson's `d_k` rising with the
activation at the time of access, the mechanism that actually produces the spacing
effect (§2.6) — is a later experiment, off in v1.

---

## 9. Evaluation — the bench is the specification

This box has no corpus. That is the right starting condition for a library: build and
measure against **fixtures** before a real text lands, and let the fixtures be the spec.

- **`bench/aha/`** — the Kestrel/Noam story as four protocols (§4) plus a timeline: the call at
  T, the Noam cue at T+5h. The fixture spells out the graph rather than the story (review
  correction 4):

  ```
  nodes    N (need)   P   Kestrel Brightmem Noam   S   me
  claims   cN  P asked about openings at Kestrel; none        cR  Noam offered me a position at S; declined
           cB1 P interviewed at Brightmem                   cB2 Noam founded Brightmem
  edges    N —need→ cN          cN —about→ P        cN —mentions→ Kestrel
           cR —about→ Noam      cR —mentions→ S     cR —mentions→ me
           cB1 —about→ P        cB1 —mentions→ Brightmem   P —rel:interviewed_at→ Brightmem
           cB2 —about→ Noam     cB2 —mentions→ Brightmem   Noam —rel:founded→ Brightmem
           cN —same_cell→ cR    (cell {opening, position, job, вакансия}; seed-time only)
  need-side seeds     N, P, Kestrel (+ cR through cN's cell at seed time)     k = 3, cached at rebuild
  stimulus seeds      T:    cell{opening, …}, Kestrel                       k = 2
                      T+5h: Noam                                            k = 2
  expected at T       cR: need side = cell route + P→Brightmem→Noam→cR;  stimulus side = cell route only
                      min(need, stimulus) below the convergence floor → no candidate
  expected at T+5h    cR: stimulus side = Noam —about→ cR (1 hop, strong)
                      min clears the floor → candidate (N, cR) carrying both paths; judge `connect`
  ```

  Every activation in the expected column is a number the fixture states, computed by hand
  from the §2.7 formula with the default knobs. Variants: no cB1/cB2 (must *not* fire — the
  cell route alone stays under the floor, so the aha depends on the bridge, which is the
  point of the example); cR months cold (does not fire: its `B_i` drags `A_i` under τ though
  the convergence registers); the same fixture with 2 hops on the need side (must not fire —
  the regression for correction 4).
- **`bench/judge/`** — fixed candidate packets with hand-labelled verdicts: the Kestrel/Noam
  pair (`connect`); the same pair after a prior `connect` in the ledger (not re-raised);
  a docs question whose answering paragraph was just ingested (`satisfies`, need closed);
  a need whose own passage says it was withdrawn (`stale`); two entities joined only
  through a hub and a shared word (`no`); a pair whose only bridge is an `inferred` edge
  (must not be `connect` on that alone). Scored as agreement with the labels; `LINK`
  checked only for not asserting anything absent from the packet.
- **`bench/xlingual/`** — RU cues against EN claims and the reverse; hit through L2 with the
  cell named; the same cue through L1 alone expected to miss.
- **`bench/kinds/`** — one news article, one wiki page, one page of tech docs through the
  same `ingest`; the protocol shapes of §1.3, chunk boundaries per §1.4, every fact anchored
  to a chunk, and a `norm` never rendered as a `property`.
- **`bench/docs_qa/`** — a small docs tree (a few pages, headings, code blocks); questions
  answered by one section; expected: that section as the chunk hit, with the page anchor
  as reference, and the fact grain *not* chosen over it.
- **`bench/extract/`** — the witness pass on a chunk-marked page and a chunk-marked
  transcript, against a hand-written expected protocol: every fact anchored to the right
  chunk, none from the *context only* chunk, a planted unsupported line caught by the
  grounding check, the `(about: self)` rule on the transcript, and the ingest report's
  counts matching. Model-dependent, so it is a regression set rather than a pass/fail —
  what is measured is recall against the hand protocol and the mis-anchor rate.
- **`bench/rebuild/`** — the manifest and scopes: a fresh store builds full from nothing;
  a second ingest triggers only a fast rebuild and the report's deltas match; a removed
  document forces full; an embedder id change invalidates exactly the table's rows; a
  witness-prompt bump marks protocols stale and rebuilds nothing; a splitter bump re-chunks
  one document and every fact re-anchors by span to the new chunk containing it, unchanged
  chunks keeping their ids and their access history, the protocol untouched (correction
  5); a fast rebuild after an ingest runs the relation pass over exactly the new claims
  and a `wants` claim appears in `needs()` with no further step (correction 3); a
  synthetic vocabulary
  shift crosses the drift knob and the next rebuild is full; a rebuild with a tightened
  dedup threshold splits a merged claim and both halves carry the right occurrence
  accesses; an aha verdict survives that split on the correct half; a reader mid-call
  during a swap finishes on the old build.
- **`bench/split/`** — the splitter on its own: a page with nested headings, a code block
  with a lead-in, a table, one-line paragraphs; expected chunk tree, no cut inside a
  protected unit, fragments merged upward. Then the same store rebuilt at two budgets (a
  Spark-sized one and a 24 GB-box-sized one): the chunk set is identical, only `fits`
  changes, and the rebuild report names every discarded chunk.
- **`bench/pivot/`** — chunks planted under both senses of `стали` / `ключ` / a coined word:
  sense induction yields two clusters per word with the expected dominant cells; with the
  door-key context warm, the anchor scorer picks `ключ` and the jump lands in the spring
  chunks with the bridge and both senses named; with *both* senses already warm the word
  scores near zero (nothing to switch to); a `root` bridge from `колокол` reaches
  `колокольчик`; a `sound` bridge pairs `магазин` / `magazine`; a stoplisted word never
  anchors; and a pivot leaves the activation state untouched.
- **`bench/priming/`** — two cues an hour apart; the second pull ranks the first's
  neighbourhood above a cold pull.
- **`bench/decay/`** — the clock injected: a node touched ten times over a month against
  one touched ten times in an hour, read a week later — the **clustered** one is warmer
  (B ≈ −0.26 against −0.74; the fixture pins the formula's actual behaviour, not the spacing
  effect it does not have — §2.6, correction m2); a node with no access at all reads exactly
  `B₀`; a need left for 90 days then touched once (dormant, then back at full floor); a bulk
  docs ingest that must *not* warm a thousand pages **and must leave every page retrievable
  on an exact cue the moment it lands** — the same check repeated with the activation
  database deleted (correction 2); a `report` from a year ago still
  retrievable on an exact cue but ranked below this week's on a "what is happening" cue;
  a `property` from a year ago ranked as if recorded today.
- **`bench/hub/`** — a hub with 200 claims; a 2-hop spread must not return them all.
- **`bench/select/`** — fixed catalogues with hand-labelled correct picks: the
  timeout-vs-timeout-logging case (nearby in subject, only one needed); a Russian message
  against an English catalogue; a passage-with-nested-claims where the question wants the
  one line and another where it wants the paragraph; a message needing nothing (`NONE`
  expected); a passage whose only claim is a `position` — absent from the catalogue under
  a withholding policy, present and attributed under a permitting one — and a gist that
  renders with its generated label and never as a fact line (correction 7); a fast-path identifier hit that must skip the pass and agree with what the
  pass would have picked. Model-dependent, so scored as agreement with the labels and as
  pick-rank distribution rather than pass/fail.
- **`bench/spread/`** — a hand-drawn graph of ~40 nodes with known edge types: expected
  activation per node to within tolerance for a given seed; a node reachable by two paths
  scores above one reachable by a single stronger path; a `rel:` edge out of a 200-mention
  hub carries whole while each of the hub's `mentions` carries `φ(200) ≈ 0.07` of it; a
  docs section with 50 claims reaches every one of them through `in_chunk` (the case the
  clamp failed); every expected activation is stated in the hours reading of §2.7, not as
  a raw number; `_self` reached but not expanded
  through on co-mention; a `position` claim carrying activation to the entity it names
  while never appearing in the offered set; the top-3 paths on every reached node read
  back correctly; no node's activation returns to it through `A → B → A`, and the
  multiplier on every hop is < 1 (correction m1); and the same seed run as PPR over the
  same edge table is **reported** against the beam's top-10 — a measurement, not an
  expected agreement (correction m3).
- **`bench/sources/`** — one fact stated by four documents of which two are near-copies
  (a syndicated article and its mirror), one contradicting claim from a fifth, and a docs
  page in two versions with one norm changed. Expected: one claim with four occurrences and
  two independent votes; the contest rendered with both sides; the v2 norm current and v1
  `superseded_by` it, **and a question naming v1 retrieving the v1 norm** (correction 6);
  the identifier `connect` defined in two libraries resolving to the one the question's
  document family names; the pair *"A must run before B"* / *"B must run before A"*
  nominated by tier 2 and **not** merged, and a negated restatement linked as `contests`
  rather than merged (correction 1); the index-page-that-lists-everything ranked below
  the page that defines the thing.
- **`bench/support/`** — the tech-support corpus: a small docs tree in **two product
  versions** and **two platforms**, one page an error-code table of 200 rows, one page of
  release notes, one FAQ. Questions with a scope: an error code (exact L1 hit, fast path,
  the right row group injected with its header); a procedure that differs by platform (the
  other platform's never offered); a setting that changed between versions (v1's norm under
  `scope={version: v1}`, v2's under v2, *changed in v2* rendered under neither); a feature
  removed in v2 (the *removed* fact retrieved); a question the docs do not answer (an empty
  block with a named reason, no adjacent paragraph); two tenants' transcripts ingested under
  scope, and a pull under one tenant that must not surface the other's; pages retrievable at
  the chunk grain before extraction runs; and `N` concurrent pulls during an ingest, each
  finishing on one build.
- **`bench/code/`** — a small project in two languages: the parser witness's protocol against
  a hand-written one (signatures, defines, imports, calls — exact, no model); *what calls X*
  answered by a one-hop spread; a chat turn naming an identifier landing on the parser's
  symbol node (one node, two lanes); a save that inserts a line at the top of a file — every
  unchanged symbol keeps its chunk id and its access history, the edited one inherits; a
  300-line function split by block with its signature on each piece; a string literal shaped
  like a key absent from the protocol; a `todo` comment appearing in `needs()`; and the open
  file's symbols outranking cold ones on a vague cue.
- **`bench/baseline/`** — the comparison that decides whether the associative layers earn
  their keep: the reviewer's closing point, adopted as the **first** bench to build rather
  than the last. The same cues, the same corpus, the same injection budget, three
  retrievers in a ladder: (1) lexical + dense only — L1 and FAISS, no graph, no
  activation; (2) plus the contextual glossary — L2 cells and the recent-turn seeds; (3)
  plus activation and spreading — the full puller. Each is scored on the hand-labelled
  picks of `bench/select` and the sections of `bench/docs_qa`, at equal context budgets,
  as pick-rank distribution plus the fraction of turns where the tiers *disagree*. The
  fixtures above test that the machinery follows its rules; this one tests whether the
  machinery retrieves better context. If tier 3 does not beat tier 2 on a real corpus at
  the same budget, L4 is a research knob rather than a default, and §7's chat runs tier 2.

Each case is `(sources, timeline of cues/touches, expected ranked set + paths)`. GPU-free
except the judge and the witness passes, which get their own small sets. Then the real
material: Ava's 20 + 3 protocols pulled over as the first real corpus, and every number
re-measured — that measurement, not the fixtures, sets `τ` and the fan term.

---

## 10. What comes over from Ava, and how

| lift out (pure, leaf) | generalize | new | not carried |
|---|---|---|---|
| `graph/` read set, node ids, facets, `store`, blob safety rules | `exchange_anchor` IDF / `words_match` / `tag_weight` → L1 over the whole vocabulary | L1 lemmatizer + stoplist | training, LoRA, builds |
| `fact_fetch` selection pass + prompt → `select` | `rag_engine` dense channel → over claims | L2 codebook + alias proposals | persona / user / self portraits |
| `chat_facts` / `til_facts` witness prompts + parser | `til_gist` → per-source gist for every kind | L3 typed relations, needs | branch judge, revision, tension |
| `graph_rebuild` staleness/rebuild → `rebuild` | outreach's decision pass → the aha judge (as an example app) | L4 activation, `touch`, `aha`, `pivot` | gossip, encounter, reflection runs |

The network facade (§7) lifts `api_http`'s OpenAI-compatible shape and
`generation._GossipSessionLog`'s prefix-match session identity; the code kind lifts
nothing — Ava has no parser witness, and that plugin is new.

Everything in the first two columns is already a leaf or a pure package that imports nothing
from the inference role — which is why lifting it is copying files, not untangling them.
The library is **stdlib + numpy + faiss + a lemmatizer + an embedder**; the model is always
injected.

Repo shape: a new repository. Ava's tree stays as the reference implementation of the
first application and the source of the first real corpus.

---

## 11. Open decisions

**Decided 2026-09-09, before milestone 1:**

- **Name and place**: `ProjectAva/assoc/` — package `assoc`, benches under `assoc/bench/`;
  starts as a subdirectory of this repository and splits out later.
- **Dense embedder for milestone 1**: BGE-M3 (dense head), one model for the chunk index
  now and the L2 term embeddings in milestone 2; `paraphrase-multilingual-MiniLM` stays as
  the tier-1 baseline it already is in Ava.
- **Witness / select model for the bench**: gemma-4-31B, loaded **in-process by the bench
  harness** (reusing Ava's `unified_memory` + `fast_load`), never by the library.
- **Code kind languages**: Python + C (tree-sitter grammars shipped first).
- **`bench/support` corpus**: hand-written fixture docs for a synthetic product.
- **Dependencies** live in `server/.venv` and are recorded in `assoc/requirements.txt`.

Still open:

- **Term embedder** — settled for now: BGE-M3's dense head serves both the chunk index and
  the L2 term cells (`bench/xlingual` passes; {sister, сестра}, {Haifa, хайфа}, {technion,
  техниона} form as cells at the 0.72 threshold). LaBSE remains the alternative to measure
  if the cells prove too coarse on a bigger vocabulary. BGE-M3's sparse head is **not** a
  candidate for L2 (review correction m4): it weights the token ids present in the input,
  so a Russian cue and an English claim share a key only where they share a token — a
  learned lexical scorer, a possible upgrade to L1's IDF, and nothing more.
- **Relations: fold-time or witness-time?** Fold-time first (derived, wrong-ably); the
  witness prompts emit `(rel: …)` once the predicate list stops moving.
- **Inferred edges from the model's own knowledge** (§4, B2) — allowed at all? If yes:
  labelled, low-weight, never a sole path.
- **What a "need" is** — explicit wants/asks only (safe, sparse), or every unanswered
  question in a transcript (rich, noisy)?
- **Grounding check depth** — lexical only (free), or the per-chunk yes/no pass too (a
  second generation per document)? On docs and news the second is probably worth it; on
  chats the lexical test may be enough. The extract bench's mis-anchor rate decides.
- **Extraction model** — same as the answerer, or a faster one? The seam allows either;
  the question is whether a 31B witness misses what a 122B witness records.
- **Tech-doc classes** — is `spec / procedure / signature / deprecated` the right witness
  vocabulary, or should the first docs corpus decide it the way Ava's corpus decided the
  facet level?
- **Oversize chunks after v1** — v1 discards them from injection (§1.5). The second
  behaviour, splitting at paragraph boundaries with the heading path repeated, is decided by
  the docs bench once there is a docs corpus whose sections actually exceed the Spark's
  per-chunk ceiling.
- **The kind-plugin contract for structured sources** — is *the structure itself* a witness
  (a row is a fact, §1.3), or does a table still want an LLM pass for the prose cells? The
  support bench's error-code table decides.
- **Scope granularity for activation** — per conversation, per tenant, or both layered; and
  the discount on the global layer (§2.4). A support deployment with thousands of
  conversations decides whether the context layer is worth its rows.
- **Which small model runs `select`** on a deployment where the answering model cannot
  afford a prefill per customer per turn (§3.2) — and whether the fast path alone covers
  enough of a support corpus that `select` is rare there.
- **Append-only ingest of a live chat** (§1.4) — every N exchanges, on idle, or on a
  client's session-end signal where one exists; the library allows all three.
- **Grain choice** (§3 step 4) is a heuristic today; whether it should be the model's pick
  (offer all three grains in the catalogue) or the puller's rule is worth measuring on
  `bench/docs_qa` versus `bench/aha`.

---

## 12. Order of build — four milestones, each gated by its bench

> **Real corpus measured 2026-09-10** (`assoc/ava_import.py`, `bench/measure_real.py`):
> Ava's snapshot pulled from the main box — 246 transcripts + 148 TIL texts, with
> Ava's own `.facts.json` protocols imported as each document's protocol (`witness.import_protocol`:
> every fact anchored by overlap, lemmas OR L2 cells, the cells placing the English fact's
> words over the Russian exchange) and her gists as the summaries. 394 documents, 2425
> chunks, 5899 claims (3326 chat + 2727 TIL; 2 ungrounded, 2826 grounded via cells), 4813
> cells, 379 families, 3 contests, 305 polysemous words. Labels by construction, no hand
> labels: **fact → source** (cue = a chat fact's English text, expected = the Russian
> exchange it anchors to, 300 cross-script pairs, ranked among chunk hits) and **turn →
> facts** (cue = a user turn, expected = any claim anchored to it, ranked among claims).
> Cold, BGE-M3, equal budget:
>
> | | tier 1 | tier 2 | tier 3 |
> |---|---|---|---|
> | fact → source, MRR among chunks / top-10 | 0.317 / 0.49 | **0.440** / 0.66 | 0.441 / 0.65 |
> | fact → source, conversation MRR (passage grains) | 0.42 | 0.51 | 0.51 |
> | turn → facts, MRR among claims / top-10 | 0.482 / 0.60 | **0.519** / 0.63 | 0.514 / 0.63 |
> | ms per pull | 34 | 77 | 98 |
>
> §12's gates hold on the real corpus exactly as on the fixtures: tier 2 beats tier 1 by a
> wide margin (the cross-lingual case is where the cells earn their keep), tier 3 is even
> with tier 2 cold — default tier 2 stands. The absolute numbers are bounded by the labels:
> Ava's protocol is interpretive (`I believe that the only criterion for stabilizing my
> states is …`, subject `_self`, 2175 of 3326 facts) over dozens of conversations on one
> theme, so the "right" exchange is often one of several siblings that say the same, and
> the anchor itself was chosen by overlap. Two defects the corpus surfaced, both fixed:
> the embedding cache never loaded (`NpzFile` re-reads a member on every subscript, so
> `z["vecs"][i]` in a loop allocated the 115 MB matrix 28k times — 121 GB and the OOM
> killer; every rebuild was re-embedding everything), and the **lexical channel was
> top-normalized**: on an English cue the best lexical hit is whatever English page shares
> one word (`Anthropic released Mythos5` → *The Witcher*, on `released`), and it read as a
> full 1.0. The ceiling is now a share of the query's *matchable* mass (`Glossary.query_mass`:
> Σ idf over surface tokens that have any posting — an untranslatable word is not an
> unmatched one, which is what kept `сколько заплатили за Starling` at rank 1), `LEX_MASS_FRAC`
> 0.5; it lifted fact → source at tier 1 from 0.244 to 0.317 and at tier 2 from 0.392 to
> 0.440 with the fixture ladder unchanged. Rebuild of this corpus 668 s → 145 s (sense
> induction recomputed centroids on every join; the codebook stacked its leaders per term).
> Not measured here: the relation pass, the aha and the judge need the gemma harness over
> 5899 claims — a separate, GPU-serial run. What the real corpus asks for next: a witness
> that writes *third-person, referent-resolved* facts (the `_self` protocol is Ava's inner
> voice, which no cue will ever phrase), and labels from a held-out translation rather than
> from the anchor.
>
> **Milestone 4 landed 2026-09-09** (`assoc/senses.py`, `infer.py`; `graph.learn_strength`,
> `Activation(variable_d=True)`, `Library.pivot`). Built as §6 and §8 specify: senses induced
> per word from the L2 concept signatures of its contexts (a leader pass, then an
> agglomerative merge pass — the leader pass alone fragmented `ключ` into two door and two
> spring clusters and measured the "split" between the two spring ones); the anchor scorer
> with the far sense's activation taken *relative* to the near one (both warm ⇒ zero) and a
> **landing** factor, the log of the far sense's postings — a pivot needs somewhere to land,
> and IDF alone preferred the rarer topical word (`старый`, old lock / old knife) over the
> planted `ключ`; the `root` bridge (shared stem ≥ 5, an inflection of the same lemma
> excluded) and the `sound` bridge (rhyme, edit distance, transliteration: магазин ↔
> magazine); the jump as a spread from the far sense's postings, never an access.
> PMI-shaped learned strength on `mentions` edges (co-mentions with a claim's subject,
> multiplier in [0.5, 1.5], every hop still < 1) — **on by default**, and it moved the ladder:
> tier 3 now 0.926 on chat against tier 2's 0.87, 0.87 vs 0.88 overall. Pavlik & Anderson's
> activation-dependent decay as `Activation(variable_d=True)`, with `d_k` capped at 2 (our
> hour-scale odds run far above ACT-R's seconds), producing the spacing effect the fixed-`d`
> sum lacks — `bench/decay` pins both behaviours side by side; off by default. Inferred
> edges (`infer.py`): a thinking-off "do you KNOW a connection" pass writing attributed
> `inferred:<pred>` edges at `s_type` 0.3, loaded only under `knobs["inferred_edges"]`, and
> `aha.find_candidates` drops a candidate whose paths on either side are inferred only.
> Two things the pivot corpus made plain: context-clustering sense induction finds *topics*
> as readily as senses (`старый` splits by knife-vs-door, which is one meaning in two
> rooms — and a legitimate Дягилева pivot all the same), and the ladder's labels are only
> as unambiguous as the corpus is small: the pivot paragraphs about a lost key now compete
> with the chat that says the same. No gate here by §12's own terms.
>
> **Milestone 3 landed 2026-09-09** (`assoc/relations.py`, `graph.py`, `activation.py`,
> `spread.py`, `aha.py`; wired in `rebuild.py`, `puller.py`, `library.py`, `facade.py`).
> Built as specified: the per-document relation pass (cached per claim + prompt version,
> run in the fast scope), the typed edge table, the sqlite activation store with the cold
> baseline `B₀`, the need floor and the global + context layers, the beam spread with the
> soft fan penalty and the four path rules, the one-scale `A_i`, standing needs from
> relations / `todo` / registration, `aha()` as arithmetic with the cached need side and
> the two-sided minimum, `judge()` with the four verdicts and the never-recorded `error`.
> **Measured:** the Kestrel/Noam fixture behaves as §4 says — nothing at the call, the
> candidate at T+5 h with the `interviewed_at → founded → mentions` route on the need side
> (r ≈ 0.017) and the one-hop stimulus side (0.37); without the bridges the need side falls
> under the floor; the real relation pass on gemma-4-31B recovered exactly `wants`,
> `interviewed_at` and `founded` from the live chat protocol with no bad arguments. **The
> ladder, cold:** tier 1 0.944, tier 2 0.944, tier 3 0.88 overall; all three 1.0 on chat. So
> by this section's own rule the default stays tier 2 and tier 3 is `Library(tier=3)`. Four
> things the build taught, in the code's docstrings: (1) **ranking by `A_i` alone destroys
> tier 2's precision** — a cue with sixty channel hits shares one unit of activation sixty
> ways and τ then cuts the answers themselves; tier 3 therefore keeps the channel score as
> its backbone and adds two bounded terms, the activation that *arrived* by spreading and
> the warmth above `B₀`, with `A_i` reported for `explain`; (2) **a resource must be new to
> the need** — reached from the stimulus in fewer hops than from the need and outside the
> need's one-hop neighbourhood — or the bridge facts, reached strongly from both sides,
> outrank every genuine resource; (3) that hop distance is measured from the need's **core**
> seeds only, since a cell posting can be the resource's own chunk; (4) chunks need creation
> accesses as claims do, or warmth favours the claim over its own passage. The ledger is per
> need: two needs of one subject (`asked_about` beside `looking_for`) can each raise the same
> resource; folding needs by subject is a later refinement. Deferred as the design allows:
> PMI-learned edge strength, activation-dependent decay, inferred edges from the model.
>
> **Milestone 2 landed 2026-09-09** (`assoc/codebook.py`, `dedup.py`, `authority.py`, wired in
> `rebuild.py` / `puller.py` / `witness.py`). The gate held on the fixtures with BGE-M3 as the
> term embedder: tier 2 MRR 0.907 against tier 1's 0.870 overall, and 1.0 against 0.926 on
> the chat labels (`assoc/bench/reports/baseline.json`), the difference being cues whose
> answer exists only in the other language. Four things the build taught the design, none
> of them in the text above and all now in the code's docstrings: (1) **a cue's words need
> not be in the vocabulary** — an English cue over a Russian-only fact has no cell until
> its terms are placed into the existing centroids at query time (`assign_query_terms`),
> and that placement counts the cue word as a member, so a one-member Russian cell reached
> by an English word IS a two-member match; (2) **a cell match through the cue's own word
> is not concept evidence** — it is what L1 already scored, so a posting counts only
> through a *different* member, which is what stops a generic cell like {start, restart,
> stop} from double-counting a lexical hit; (3) **coverage** — the concept score scales with
> the fraction of the cue's terms that found a cell, so one generic hit out of five terms is
> weak and three of five is strong; (4) **neighbour-cell expansion measured as pure noise**
> on this corpus and is off by default (a knob). Also fixed underneath: simplemma returns
> proper nouns capitalized, which had left "Haifa" and "haifa" as two keys in every layer;
> a "removed in <version>" event seen from several platforms merges into one claim; and the
> alias table (`state/aliases.json`, hand-promoted) folds `person:артемий` into
> `person:artemy` before the dedup tiers block by subject. Deferred: tier-4 model
> adjudication (off by default in the design too) and the free-text value-conflict contest,
> which §2.5 already says v1 does not detect.
>
> **Milestone 1 landed 2026-09-09** in `assoc/` (see `assoc/README.md` for the map and
> the commands). Tier-1 numbers on the fixtures with BGE-M3: MRR 0.83, every label in the
> top 3 (`assoc/bench/reports/baseline_tier1.json`). What the build changed in the design,
> all recorded in the code's docstrings: a chunk id is legitimately shared by several
> versions of a page (an unchanged section keeps its id), so the index maps chunk → documents
> and the scope picks the version at query time; the dense channel is normalized above the
> embedder's baseline cosine, not against the top hit alone (BGE-M3 puts unrelated text at
> 0.5–0.6 and the right chunk at 0.7); an exact identifier hit is contested only by another
> exact hit, never by a dense near-miss on a neighbouring code; a *removed in <version>*
> event ranks a little below the current fact about the same subject; and the witness's
> statement may arrive wrapped as one more marker (`(stated: "…")`), which the parser
> unwraps. Deferred from the milestone's row: nothing — the facade shipped with it.

The v1 described above is store, splitter, kind plugins, L1, L2, L3 with relations, L4,
authority, contests, sense induction, pivots, the aha and its judge, and a manifest-driven
rebuild. That is four projects, and building them in one pass would mean measuring nothing
until everything exists. The `bench/baseline` ladder (§9) already implies the order, and the
three applications (§1.0) say what each milestone must be able to *do*, not just pass:

| milestone | builds | is, whole | benches that must pass | gate to the next |
|---|---|---|---|---|
| **1 — retrieval** | the document store with keys and versions (§1.4); the splitter with v1 oversize splitting (§1.5); the kind registry with the **parser** witness, the **structured** witness and the **LLM** witness behind the two-step ingest (§1.3, §1.6); L1 with lemmatizers and stoplists; the dense index; `scope` filtering; `select` with the fast path and the render policy (§3.2); `rebuild(fast)` and the manifest; the HTTP facade with prefix-match sessions (§7) | **the support bot and the editor plugin**, end to end — every question they answer on the common path is an exact hit or a dense hit on a chunk, cited | `split`, `extract`, `kinds`, `docs_qa`, `support`, `code`, `select`, the `rebuild` cases that need no L2/L3, `baseline` tier 1 | tier 1 numbers recorded on the two corpora; the ingest and rebuild reports read as designed |
| **2 — meaning** | L2 codebook + soft assignment + alias proposals (§2.2); dedup tiers with the equivalence check (§2.5); families and authority; contests and supersession; the grounding check's cell half (§1.6); drift and `rebuild(full)` | **Ava's memory at the claim grain**, cross-lingual, with sources counted and contradictions shown | `xlingual`, `sources`, the rest of `rebuild`, `baseline` tier 2 | tier 2 beats tier 1 at equal budget on at least the chat corpus; if it does not, L2 is a knob and the design is re-examined here, not further on |
| **3 — association** | L3 typed relations from the per-document pass (§2.3); L4 activation, scoped, in sqlite (§2.4); spreading with the soft fan penalty on one scale (§2.7); needs; `aha()` and `judge()` (§4); `touch` from editor events and cited replies; `outcome` | **the associative puller** — priming, convergence, incubation | `priming`, `decay`, `hub`, `spread`, `aha`, `judge`, `baseline` tier 3 | **built only if tier 2 left something on the table**, and kept as a default only if tier 3 beats tier 2 at equal budget on a real corpus; otherwise L4 ships as a research knob and §7's chat runs tier 2 |
| **4 — creative** | sense induction and `pivot()` with the `root` and `sound` bridges (§6); PMI-learned edge strength (§2.7); activation-dependent decay (§8); inferred edges from the model's own knowledge (§4, B2), off by default | the wander-mode capabilities the first conversation asked for | `pivot`, and the decay fixture re-run under the variable-`d` experiment | none — this is where the design stops being a library and starts being Ava's again |

Three rules hold across the table. **A milestone ships with its benches**, and the bench is
written before the layer (§9: the bench is the specification). **Nothing in a later
milestone changes the store**: the protocols written in milestone 1 are the protocols
milestone 3 spreads over, which is what §1.1's fold discipline buys and why the order is
safe. And **the gate is a measurement at equal context budget, never an argument** — the
question the external review ended on, and the one thing this document cannot settle by
being longer.

The first real corpora are the ones the applications supply: a small docs tree in two
versions for the support bench, this repository for the code bench, and Ava's 20 + 3
protocols pulled over as the chat corpus. Milestone 1 needs no GPU beyond the LLM witness
for the chat lane, and that runs on the fastest model that passes `bench/extract` (§1.6).
