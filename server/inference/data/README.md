# `data/` — reflection & training artifact layout

All persistent runtime state lives here. **Everything under `data/` is gitignored**
except this README (see the exception in the repo-root `.gitignore`). The tree is
organized by *lifetime*: `hot/` is the working set still moving through
consolidation, `archive/` is what has already landed in the adapter weights, and
`scratch/` is disposable.

Paths are resolved in two places that must agree:
[`server.py`](../server.py) (`_CHATS_DIR`/`_MEMORY_DIR`/`_CONSOLIDATION_DIR`/`_REFLECTION_RUNS_DIR`)
and [`training/reflections_path.py`](../../training/reflections_path.py).

```
data/
├── hot/                                  working set — durable, still consolidating
│   ├── chats/
│   │   ├── <ts>.json                     timestamped chat transcript
│   │   └── <ts>.state.json               consolidation sidecar (verdict/target/stage)
│   ├── memory/
│   │   ├── rag_memory.jsonl              ask/fact insert·evict·surface op-log (recalled at chat time)
│   │   └── weights_persona.jsonl         [persona]/[fact] statements bound for the adapter
│   ├── consolidation/
│   │   └── consolidation_anchors.jsonl   anchor ledger op-log (register + stage advance)
│   └── reflection_runs/
│       ├── <run_id>.meta.json            live run state (rewritten on every status change)
│       ├── <run_id>.events.jsonl         append-only event log (carries the per-pass `report`)
│       ├── <run_id>.summary.json         final counters/status (incl. aggregate `report`)
│       ├── <run_id>.preview.consolidation.jsonl   DRY-RUN ONLY projected consolidation output
│       └── <run_id>.preview.revision.jsonl        DRY-RUN ONLY projected revision output
├── archive/
│   └── chats/<ts>.json + <ts>.state.json fully-destaged transcripts moved out of hot/
└── scratch/
    └── sft_render.jsonl                  per-cycle training render (written then deleted)
```

`<run_id>` is the run's start timestamp, `%Y%m%d_%H%M%S` (`server.py` `handle_start_reflection_run`).

Related state that does **not** live under `data/`:

| Path | What | Writer |
|---|---|---|
| `server/server_config.json` | base `model_id` (frozen) + `adapter_id` pointer | `train_cycle.py` |
| `server/models/adapter-<ts>/` | persistent LoRA adapter (the trained delta) | `train_cycle.py` `save_pretrained` |
| `server/models/trainer/` | transient TRL checkpoints | TRL `output_dir` |

The RAG FAISS indexes are **in-memory only** — rebuilt from `hot/chats/` and
`memory/rag_memory.jsonl` on demand ([`rag_engine.py`](../core/rag_engine.py)),
never persisted as a file.

## Per-file detail

| File | Writer | Mode | Written when |
|---|---|---|---|
| `hot/chats/<ts>.json` | `ChatLogger` | rewrite | chat time — one file per session, appended per exchange |
| `hot/chats/<ts>.state.json` | `ChatSidecar` | rewrite | revision (real sleep) writes verdict/target; training writes stage advances |
| `hot/memory/rag_memory.jsonl` | `ReflectionWriter` | append-only | sleep consolidation (`insert`/`evict`) **and** chat time (`surface`) |
| `hot/memory/weights_persona.jsonl` | `ReflectionWriter` | append-only | sleep — consolidation `[fact]`/`[persona]` + revision `[persona]` |
| `hot/consolidation/consolidation_anchors.jsonl` | `ConsolidationLedger` | append-only | sleep (register anchors) + training (advance stages) |
| `hot/reflection_runs/<run_id>.meta.json` | `ReflectionRunStore` | rewrite | every status update, real **and** dry-run |
| `hot/reflection_runs/<run_id>.events.jsonl` | `ReflectionRunStore` | append | every event, real **and** dry-run |
| `hot/reflection_runs/<run_id>.summary.json` | `ReflectionRunStore` | rewrite-once | run finalize, real **and** dry-run |
| `hot/reflection_runs/<run_id>.preview.*.jsonl` | `ReflectionRunner` | append | **dry-run only** |
| `archive/chats/*` | `train_cycle` | move | a session whose every exchange is fully destaged |
| `scratch/sft_render.jsonl` | `train_cycle` | write → unlink | rendered at cycle start, deleted at cycle end |

## Lifecycle phases

**1. Chat time** (`generate`)
- `hot/chats/<ts>.json` grows by one exchange.
- `hot/memory/rag_memory.jsonl` gets a `surface` op when an open `[ask]` is proactively raised. This is the **only** reflection artifact mutated outside a sleep run, so it's the one most likely to interleave with a concurrent run.

**2. Sleep run — bookkeeping (real *and* dry-run)**
Always written, keyed by `run_id`: `reflection_runs/<run_id>.{meta.json, events.jsonl, summary.json}`. The events/summary carry the structured `report` payload the Sleep tab renders.

**3a. Sleep run — dry-run durable output**
Only `reflection_runs/<run_id>.preview.consolidation.jsonl` and `.preview.revision.jsonl`. Nothing in `memory/`, `consolidation/`, or the chat sidecars is touched; the RAG index is not rebuilt.

**3b. Sleep run — real durable output** (`ReflectionWriter`)
- `memory/rag_memory.jsonl` ← consolidation `insert` (ask/fact) + `evict` (resolved)
- `memory/weights_persona.jsonl` ← consolidation `[fact]`/`[persona]` + revision `[persona]`
- `consolidation/consolidation_anchors.jsonl` ← registered anchors
- `hot/chats/<ts>.state.json` ← verdict + target stamped next to the chat
- in-memory reflection RAG index rebuilt

**4. Training cycle** (`train_cycle.py`, offline, model unloaded)
- `scratch/sft_render.jsonl` rendered from live anchors, then `unlink`ed at the end.
- LoRA-train on the **frozen base**; gated behind a regression probe (capability/format/character/retention). On failure: no adapter saved, `server_config.json` untouched, stages not advanced.
- On pass: persistent adapter saved to `server/models/adapter-<ts>/`; `server_config.json` `adapter_id` repointed (the base `model_id` is never modified).
- `consolidation_anchors.jsonl` + chat sidecars get stage advances; fully-destaged sessions move `hot/chats/` → `archive/chats/`.
