# server/til — "Today I Learned" source (gradual build)

First, self-contained step toward the lookup / serendipity source described in
[SCRATCHPAD.md](../../SCRATCHPAD.md) → *"The Lookup Agent & the 'Today I Learned'
pass"*. Network-only, GPU-free: it fetches outside text and deposits it as a
snippet for a later reflection pass to read. Nothing here is wired into the
inference server, RAG, or the reflection loop yet.

## fetch_current_events.py

Pulls one day of world news from Wikipedia's **Current events** portal and writes
a readable digest. Each day is its own subpage — `Portal:Current events/2026 June
20` (full month name, **no leading zero** on the day) — so fetching a specific
date is just title formatting; the MediaWiki `parse` API returns its wikitext.
Today's page is usually empty until the day is over, so the default target is
**yesterday**.

```bash
cd server/til
../../.venv/bin/python fetch_current_events.py            # yesterday
../../.venv/bin/python fetch_current_events.py --date 2026-06-20
../../.venv/bin/python fetch_current_events.py --days-back 2 --print
```

Stdlib only, no dependencies. (On macOS the system Python may lack a CA bundle and
fail TLS verification — run it through the project `.venv`, which has one. The
Linux server is unaffected.)

## Output → `server/data/til/snippets/news/` (gitignored)

Snippets are **data**, not source, so they live under the ordered `server/data/`
state home rather than beside this fetch code — and that is the tree
`snapshot_state.py` captures. One subdir per source kind:

| Dir | Writer |
|---|---|
| `snippets/news/` | `fetch_current_events.py` (this script) |
| `snippets/wander/` | `fetch_wiki.py` |
| `snippets/lookups/` | `fetch_article.py` |

Only this directory holds code.

Per fetched day, two files keyed by ISO date:

- `<date>.txt` — the cleaned, indented digest (the snippet a learning pass reads).
- `<date>.json` — provenance sidecar, already shaped toward the eventual
  *learnings* record: `{kind:"serendipity", source, date, title, source_url,
  fetched_at, text, sources:[citation urls…]}`.

## Dry-run learning pass (wired, read-only)

The Sleep tab's **Learn** button now fetches the digest *and* runs a dry-run
learning reflection over it: the server frames the digest under
[`learning_prompt.txt`](../inference/prompts/learning_prompt.txt) ("you came across
this; is any of it worth keeping?"), streams Ava's reflection into the Sleep event
log, and shows the **proposed modifiers** — the `[fact]`/`[ask]`/`[resolved]` items
a real learning pass *would* route — parsed by the existing
`reflection_writer.parse_consolidation`. **Nothing is written**: no memory, ledger,
RAG, or staging. RAG retrieval is *enabled* (read-only) during the pass, so Ava
reads the news through the prism of what she already knows and is — her recalled
facts, open questions, and persona are injected as context. Protocol:
`til_fetch {reflect:true}` → `til_fetched` → `til_reflect_chunk`* →
`til_reflect_done {text, report}`.

This is the read-only half of the design's "Today I Learned" pass — it lets us see
how Ava handles external information before any of it is allowed to mutate her state.

## Not done yet (next steps)

- Append fetched snippets into the `learnings.jsonl` store instead of (or in
  addition to) loose files.
- Promote the dry-run learning pass to a *writing* stage (route the modifiers
  through `ReflectionWriter` into the staging workspace, then merge/commit).
- Watchdog supervision + scheduling of the fetch.
