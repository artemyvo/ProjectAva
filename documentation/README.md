# Ava Documentation Set

Five files, split by rate of change. Read this first to know which file answers which
question and which file to update after a code change.

| File | Role | Changes when |
| --- | --- | --- |
| `AVA_DESIGN.md` | Stable architecture baseline — what the system is meant to keep true conceptually. | Rarely: only when the architecture itself changes (a new subsystem, a changed invariant), not when an implementation detail moves. |
| `AVA_MEMORY.md` | Cornerstone reference for the memory model — every channel that carries information forward (RAG channels, weights, persona, wander), its decay curve, channel coupling, and the memory-model open issues. | When a memory channel, decay curve, or their coupling changes; keep the per-channel open issues current. |
| `AVA_STATUS.md` | Implementation truth table — Built / Partial / Logged-only / Pending / Removed. | **Every architecture-relevant code change.** This is the file most likely to be stale; trust it least-recently-checked, and fix it when you touch the code it describes. |
| `AVA_CHANGELOG.md` | Curated technical history: architecture shifts, interpretation changes, negative results, rollback/replay assumptions. Not a git log. | When a change would alter how a future reader interprets existing artifacts or design decisions. Cite the commit hash. |
| `AVA_OPEN_PROBLEMS.md` | Unresolved design and proof gaps, each split into "what exists" vs "what remains open". | When a gap opens, narrows, or closes — move solved items into STATUS/CHANGELOG rather than deleting the analysis. |

## Precedence on conflict

**Code > `AVA_STATUS.md` > `AVA_DESIGN.md`.** If code disagrees with STATUS, the code is
right and STATUS needs a fix. If STATUS disagrees with DESIGN, either the implementation
drifted (fix code or flag in OPEN_PROBLEMS) or the design evolved (update DESIGN and add
a CHANGELOG entry). Never "fix" a doc by making it vaguer — record the discrepancy.

Each file carries a `Last code check:` date near the top. Update it when you verify a
file against the codebase, even if nothing changed.

## Relation to the root-level docs

- `../DEPLOY.md` — how to install and run the server and the client. Fine-grained
  code-level facts (module responsibilities, protocol, storage paths) live in the
  modules' own docstrings; design-level truth lives here.
- `AVA_DESIGN_LEGACY.md` (this folder) — the frozen legacy governing document this set
  was split from (formerly the repo-root `AVA_DESIGN.md`). Kept as the deep-design
  archive: extended analyses (e.g. "Belief-Adoption Dynamics — Fast Path vs. Slow
  Path", the token economy, the ASK lifecycle) live only there. Code comments still
  cite its section names (`AVA_DESIGN_LEGACY.md → *Fact / persona lifecycle*`) — when
  you touch code around such a comment, re-point it to the current docs if the topic
  is covered there; keep the legacy pointer only for the deep analyses. Do not update
  the legacy file itself; new design writing goes into this folder's maintained set.
- `../README.md` — the philosophy/"what" document, unaffected by this split.
- `../SCRATCHPAD.md` — raw brainstorming; nothing in it is a commitment.
- `../SPARK_LOADING.md` — **read before touching the model loader**: the three DGX Spark
  load-time traps (mmap→CUDA at 0.16 GB/s, CUDA free excludes page cache, bnb skips fused
  MoE experts), their fixes, and the benchmark that re-verifies them.
- `../GOSSIP.md` — self-contained implementation brief for same-user model gossip
  (designed, not built; tracked in AVA_STATUS.md's Pending table).
