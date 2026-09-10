# graph/ — implementation notes

The design lives in `FACTS_TREE.md` at the repo root. This file holds the detail that
outgrows it: what stage 1 actually does, what the first real build measured, and the
decisions a future stage needs to know were made deliberately.

Stage 1 is **pure, stdlib-only and GPU-free**. A build needs no model, no embedder and no
running server. It reads two directories and writes one file.

```
read.py    the authoritative read set → occurrence rows
nodes.py   typed node ids, formatting normalization, alias table, tiers 1–2
fold.py    facet mapping, claim dedup, tree assembly, edges
store.py   tree.json write/read + the consumer-facing read API
build.py   CLI: build and browse
selftest.py   all of the above, plus the seams between them
```

Run: `cd server && .venv/bin/python -m graph.build` (see the module docstring for the browse
flags). Self-test: `python -m graph.selftest`.

---

## First real build (2026-08-10, 23 protocols)

```
sources      23 protocols  (chat 20, til 3)          ← of 61 .facts.json on disk
occurrences  398           (chat 349, til 49)        ← 14 about nobody
claims       384 distinct  0 collapsed  0 corroborated
             position 280 (73%)  property 58 (15%)  event 39 (10%)  report 6 (2%)
nodes        212           (unknown 182, entity 25, person 5)   → 86% untyped
```

Three of these numbers should shape whatever is built next.

**`0 collapsed, 0 corroborated`.** Not one fact in the corpus is restated verbatim — the
exact-match tier merges *nothing*. This is the strongest result of stage 1 and it is a
negative one: all the dedup value is in paraphrase merging, i.e. the stage-3 embedding tier.
It also means "corroboration across sources" is currently a column of zeros, so nothing
downstream should rank on it yet.

**`86% of nodes are untyped`** (182 of 212). The two free structural signals type only what
the producers already decided; every mention in the `entities` slot falls through. Inspecting
that residue shows it is exactly the two populations §P2 of the brief predicted, interleaved:
proper nouns (`China` 5, `Israel` 5, `Project Ava` 5, `Iran` 3) and abstractions
(`subjectivity` 13, `tool` 6, `personality` 5, `responsibility` 5, `will` 5, `identity` 4).
Separating them is what `topic:` is reserved for.

**`73% position`.** The facet level is not defensive over-engineering; three quarters of the
corpus is one facet, and it is the one that must never be read as knowledge.

---

## Decisions worth not re-litigating

**The read set is enumerated positively.** 38 of the 61 `.facts.json` files on this box are
copies (persona lineage ×32, reflection checkpoint ×2, review archive ×2). `rglob` over
`server/data` would triple-count facts and report the duplication as corroboration — the one
error a fact fold must not have. `read.til_protocol_paths` therefore descends exactly one
kind-dir deep rather than recursing, because a persona snapshot nests a whole
`til/snippets/` tree underneath.

**Paths come from `training.reflections_path`, never re-derived.** The live chat corpus moved
to `server/data/chats` and a stale second copy of the old `inference/data/hot/chats` path has
already caused a real bug in this repo. `til_snippets_dir()` and `graph_dir()` were added
there for the same reason rather than being defined here.

**The chat clock is the session stem, not the document `ts`.** `ts` is when the pass *ran*,
which on a backfilled protocol is weeks after the conversation. Every chat-side curve on this
box keys on the conversation's own date.

**Corroboration counts distinct sources, not occurrences.** One conversation restating a
thing four times is one witness. Counting occurrences is how a fold manufactures confidence
it has not earned.

**`unknown:` keeps the cleaned surface form as its key; person nodes keep the casefolded
match key.** A later tier needs to see what was actually written, and an `unknown:` node
exists precisely to be adjudicated later. Person keys are already normalized upstream.

**First-person mentions resolve to `person:_self`.** `SELF_REFERENTS` is a deliberate copy of
`chat_facts._SELF_REFERENTS` — this package never imports the inference role, the same reason
`reachout_gate` carries its own copy of a small predicate. Without it, `self` (the single most
frequent chat-lane mention, 22×) becomes an `unknown:` node distinct from the subject key of
234 facts. Keep the two in step. `the user` is deliberately excluded: it normally means the
human, and guessing would misfile a real person's fact.

**A casing heuristic for person/entity/topic was declined.** It separates the measured
English data cleanly and fails on the Russian half of the corpus;
`exchange_anchor.normalize_tag` documents this exact limit. An honest `unknown:` beats a rule
that silently misfires on half the corpus.

**Unowned facts are retained as occurrences but produce no claim.** They hang off no node, so
stage 1 reaches them only by scanning occurrences; the summary's "about nobody" line (14 on
this corpus, 3.5%) is what stops that being invisible. If that fraction grows, they need a
home.

**Browse flags are read-only.** A command whose job is to show you something must not also
replace the file a consumer may be reading. The build is cheap and always runs in memory, so
a browse is never stale.

**No migration path, by construction.** `SCHEMA_VERSION` lets a *reader* refuse a tree it
does not understand; it does not let a writer convert one. A stale, corrupt or unrecognised
tree is rebuilt. `read_tree` returns `None` for all three cases, and a consumer must read
that as "no tree yet", never as "no facts" — the sources are always there.

---

## What stage 2+ plugs into

- **Tiers 3–4** (embedding-blocked candidates, LLM adjudication) go behind
  `nodes.Resolver.resolve_mention` without changing anything above it. Reuse
  `fact_contradict.cluster_by_subject` rather than writing a third copy of subject blocking.
  Grouping paraphrases is an *evaluation*, so tier 4 runs on the clean base if it runs.
- **Paraphrase claim merging** goes behind `fold.claim_text_key` — the same seam, one level
  down. Given the `0 collapsed` result this is where the value is.
- **Consumers** import `store.read_tree` + the read helpers. Always pass
  `facets=fold.KNOWLEDGE_FACETS` unless you are deliberately showing positions with their
  attribution attached. `person:_self` feeds no injected artifact (FACTS_TREE.md §10), and
  `build.py` prints that reminder on the node itself.
