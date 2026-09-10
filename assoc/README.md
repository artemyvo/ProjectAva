# assoc — the associative facts library

Design: `../ASSOCIATIVE_MEMORY.md` (repo root). This package is **all four milestones** of
§12. The default pull is tier 2 (measured: tier 3 is even with it cold on the fixtures,
ahead on chat); `Library(tier=3)` turns activation and spreading on; the aha, its judge and
the pivots are independent of the pull tier. A new project that starts on the DGX Spark; Ava is a quarry and one of its
target applications, not its runtime.

```
store.py      documents with durable keys + immutable versions; append-only chats; chunk ids
              are structural (key, path, text) and shared across versions of a page
chunks.py     the unit model (primary / piece / ancestor)
kinds/        the kind registry: markdown.py (structure-first splitter), prose.py (tech_doc /
              news / article / structured), chat.py, code.py (tree-sitter + PARSER witness),
              structured.py (STRUCTURE witness: table rows, FAQ entries, release notes)
protocol.py   the fact-line grammar + lenient parser (ported from Ava's chat_facts)
witness.py    the deferred LLM witness: chunk-marked section groups, anchoring + repair,
              grounding, the ingest report; dispatch over a kind's witnesses
lex.py        tokens, scripts, lemmas (pymorphy3 for Cyrillic, simplemma for Latin), stoplists
glossary.py   L1 — BM25 inverted index over chunks + claims
dense.py      the dense index (BGE-M3 + FAISS; HashEmbedder as the offline stand-in)
codebook.py   L2 — cells over term embeddings: soft assignment, query-time placement of
              unseen cue words, the concept channel, drift, alias proposals
fold.py       exact-tier claims with every occurrence kept; "removed in <version>" events
dedup.py      tiers 2–3 nominate (same cells / claim cosine within a subject block); the
              equivalence check (polarity, version, condition, argument order) merges or
              links `contests` / `supersedes`
authority.py  families (doc key + shingle near-duplicates) and the damped bipartite PageRank
relations.py  the per-document typed-relation pass (closed predicate list, argument check, cache)
graph.py      the edge table: five node kinds, typed edges with base strengths, per-type fan
activation.py L4 — sqlite access history per (scope, node); B_i with the cold baseline B₀;
              the need floor; global + context layers; the clock injected (hours)
spread.py     the beam spread (γ, soft fan penalty f^−α, ε, beam, the four path rules) and
              the one-scale A_i = ln(e^B + κ r) + λ ln(1 + auth)
aha.py        standing needs → the two-sided spread, the novelty rule, the ledger, the judge
senses.py     sense induction over concept signatures; anchor scoring; root + sound bridges;
              `Library.pivot(context, bridge=sense|root|sound)` returns Jumps, never an access
infer.py      inferred edges from the model's own knowledge — off by default, attributed,
              low weight, never the sole path to an aha
budget.py     per-script token estimate, fit stamps, oversize splitting, budget spend
puller.py     tier-1 pull: lexical + dense over three grains, scope-filtered, explainable
selection.py  policy, three-grain catalogue, fast path, ordinal picks, budget spend, Block
render.py     claim lines by facet, passages as attributed quotations, gists labelled
rebuild.py    manifest, staleness, atomic build swap
library.py    the §1.2 API (ingest / extract_pending / rebuild / pull / select / inject / …)
facade.py     HTTP: OpenAI-compatible chat (two-stage loop) + /context + /ingest + /rebuild
prompts/      the per-kind witness framings and the selection prompt
bench/        the benches (§9) and the fixture corpus; model_harness.py loads gemma-4-31B
```

Run the benches from the repo root:

```bash
server/.venv/bin/python -m pytest assoc/bench -q
```

Model-free by default (BGE-M3 on the GPU when it is cached, else the hash stand-in;
`ASSOC_EMBEDDER=hash` forces the stand-in). `ASSOC_MODEL=1` adds the model-dependent
benches (the real witness and the real selection pass on gemma-4-31B, loaded in-process
by the harness — the library itself never loads a model; every generation goes through
an injected `generate_fn(system, user, *, thinking, max_new_tokens, temperature)`).

Reports land in `bench/reports/` (gitignored): `baseline.json` holds the tier ladder —
tier 1 (lexical + dense) against tier 2 (plus the L2 cells and recent-turn seeds) on the
same labels and budget; `Library(tier=1)` or `pull(..., tier=1)` runs the floor.
