# LOGGING.md — the unified reporting journal

Design brief for the box-wide log: **one append-only stream every process writes to,
carrying each pass's CoT and output plus raw subprocess stdout (unsloth included),
rendered in one read-only view.**

> **STATUS — built 2026-08-11, rendered in the Activity tab.** Steps 1–5 and 7 of §15
> landed: levels + the generation seam + the stdout tee + the training half + segments,
> `gap` and the tail-read `configure`, plus the docs and the snapshot exclusion. Step 6
> landed **only for the Activity tab** — the shared `client/ui/log_view.py` widget and the
> Sleep-tab mount were not built, so the Sleep tab keeps its own run-scoped event stream
> and §10's "one widget, two mounts" is still the open recommendation. §16's open questions
> are all still open. Read the rest of this document as the rationale for what is now in
> the code, not as a plan.

---

## 1. What this is not: a new log

`server/inference/core/activity_log.py` already exists and is ~70% of this. It is the
single, box-wide, append-only, restart-durable journal with one global monotonic `seq`,
served over `activity_events` → `activity_events_batch`, rendered live by the Activity
tab. It is a *leaf* module (imports nothing from the project, wired with one
`configure(path)` call), which is exactly the property that lets any process append to it.

So this document is a set of **deltas to `activity_log`**, not a second log. A second
stream would defeat the one thing the existing design got right and the whole ask depends
on: **one cursor, one file, everything interleaved.**

What follows is (§3) what is actually broken, (§4–§10) the design, (§14) what was
considered and rejected, (§15) how it lands in independently useful steps.

---

## 2. The ask, restated as requirements

1. Every process appends to one log; the UI displays it. (Mostly built.)
2. Every generation on the box — reflection, outreach, check-in, synthesis, summary,
   facts, wander, modules, deliberation — emits **its CoT and its output** there.
   (Not built: only `reflection_runner` passes are mirrored, at a 4000-char clip.)
3. Raw **unsloth / training output** appears in the same place. (Not built: it lives in
   `server/train.log`, reachable only through the watchdog while inference is down.)
4. A textbox in the Sleep tab shows it.

And one requirement the observed failure adds, which none of the above states:

5. **A running pass must be visible while it runs**, not only when it ends.

---

## 3. What is actually broken

Three observed, four latent. All of them are in scope; the latent ones must be fixed
*before* volume grows or they become data loss.

**(a) Silent GPU — the reason this document exists.** On 2026-08-11 the box spent 22:22→
22:47 on one generation (the `til_facts` protocol pass over a 25,882-char Lurkmore
article, one reading block, 12,288-token cap, which it hit) and emitted **zero** lines.
`idle_scheduler._dispatch` opens the status chip with `activity_log.set_current()`, which
*deliberately* writes no journal line — the docstring's reason is sound: an hourly job
that mostly finds nothing to do would otherwise flood the journal with started→skipped
pairs. The cost is that a 25-minute job is indistinguishable from a hang. Only one
terminal line is written, at the end.

**(b) Partial coverage.** Only `reflection_config.ReflectionRunStore._mirror_to_activity`
feeds the journal with pass output, and only for whitelisted `reflection_runner` events.
Everything else reports through `print()` to `server.log`, which no client reads:
`til_wander`'s facts/voice/learning passes (`[til] facts protocol (wander): …`),
`background_reflection`'s rungs, module runs, the autonomous check-in's per-chat recaps,
deliberation. The `[tag]` prefix convention is universal in those prints — §6.B turns
that into an asset.

**(c) The training hole.** The offline cycle runs with inference DOWN, so the WebSocket
journal is dead for the whole window; unsloth's output goes to `server/train.log` and
structured stages to `server/train_progress.jsonl`, served by the watchdog. After
relaunch the journal has a hole where the build was. §7 closes this almost for free.

**(d) Rotation destroys history.** `_rotate_locked()` rewrites the file down to the
in-memory ring (`_RING = 2000` events) once it passes `_ROTATE_BYTES = 8_000_000`. The
journal is at 1.5 MB today; adding bodies and raw stdout reaches 8 MB in hours, and the
first rotation silently discards everything but the last 2000 events. **A log you are
meant to scroll back through cannot self-truncate.**

**(e) `get()` serves only the ring.** A client asking for an `after_seq` older than the
2000-event window gets a short list with no indication that anything was skipped — a
silent gap presented as continuity.

**(f) `configure()` reads the whole file** (`_path.read_text().splitlines()`) to recover
the seq high-water. Fine at 1.5 MB, a boot stall at 128 MB.

**(g) Body clipping drops the tail.** `_MIRROR_TEXT_CAP = 4000` clips the **end** of a
pass's text. On a long generation the `<think>` survives and the answer after `</think>`
— the part you actually want — is what gets cut.

---

## 4. Design: one journal, four levels, one cursor

The existing record grows three optional fields. Every addition is backward compatible:
an old client ignores them, and a record without `level` reads as `event`.

```jsonc
{
  "type": "activity_event",
  "seq": 2291,                 // global, monotonic, restart-durable (unchanged)
  "ts": "2026-08-11T19:26:03Z",
  "source": "til",             // subsystem (unchanged)
  "activity_id": "wander-9c7d",// correlates one burst (unchanged)
  "kind": "progress",          // started|progress|note|result|skipped|failed|finished
  "phase": "facts",            // (unchanged)
  "message": "…",              // one-line headline (unchanged)

  "level": "body",             // NEW — event | body | stream | raw   (default "event")
  "pass_id": "p-4f21",         // NEW — correlates start/heartbeats/body of ONE generation
  "text": "…",                 // NEW — the payload for body/stream/raw (never in `message`)
  "detail": { … }              // (unchanged) tokens, elapsed_s, truncated, …
}
```

`text` is separate from `message` on purpose: `message` stays the one-line headline a
compact view renders, `text` is the kilobytes a detail view expands. Today the mirror
concatenates them with indentation into `message`, which is why it must clip.

### The four levels

| level | what it carries | volume | persisted |
|---|---|---|---|
| `event` | lifecycle + phase markers. Everything the journal holds today. | ~100s/day | yes, longest retention |
| `body` | one finished generation, **verbatim: CoT + output**, plus outcome flags (`truncated`, `stopped_on_loop`, tokens in/out, budget). | 1 per pass, ~2–16 KB | yes |
| `stream` | a **heartbeat** for an in-flight generation: elapsed, tokens so far, and a rolling tail of the last ~240 chars. | 1 per `heartbeat_s` (20 s) | yes (they are cheap) |
| `raw` | one stdout/stderr line from any process, incl. unsloth. | bursty | yes, shortest retention |

**Per-token deltas are never journalled and never broadcast.** The heartbeat replaces
them; the finished `body` carries the whole text a few seconds later. This is the single
economy that makes "every process emits its CoT" affordable — see §14 for the arithmetic
on the alternative.

---

## 5. The heartbeat, and the lazy `started` line

This is the fix for §3(a), and it preserves the original no-flood intent exactly.

* `set_current()` no longer decides whether to write a `started` line. It *stages* one.
* A heartbeat timer (one `threading.Timer`, re-armed; the box runs one GPU job at a time,
  so one timer suffices) fires at `heartbeat_s`. On its **first** tick it flushes the
  staged `started` line at `event` level, then emits a `stream` record; on each later tick
  another `stream` record.
* `clear_current()` cancels the timer. A job that finishes inside `heartbeat_s` therefore
  writes **no** `started` line at all — hourly no-op jobs stay silent, which was the whole
  point of the current behaviour — while a job that runs 25 minutes announces itself at
  20 seconds and then reports every 20 seconds until it ends.

The heartbeat's payload comes from the generation seam (§6.A), which is the only place
that knows tokens-so-far and the live tail. When a job holds the chip without generating
(a fetch, a file sweep), the heartbeat still fires with elapsed-only — "still here, 3m20s
in" is the minimum honest answer to "is it hung?".

---

## 6. Emission: three routes, and two of them cover the whole box

### A. The generation seam — one hook, total coverage of passes

`core/generation.py` has exactly two factories every background generation on the box goes
through: `_make_sync_reflect_generate` (line ~511, its inner `generate_fn` at ~616) and
`_make_agentic_generate` (~815). Reflection passes, TIL/wander, outreach, check-in,
synthesis, deliberation, modules, the clean-base evaluations — all of them. Wrapping those
two inner functions gives requirement (2) in one edit, with no call-site changes:

```python
pass_id = activity_log.begin_pass(label, source=..., meta={"budget": max_new, ...})
#   … existing generation …
activity_log.end_pass(pass_id, text=raw, tokens=..., input_tokens=...,
                      truncated=generate_fn.last_truncated, stopped_on_loop=...)
```

`begin_pass` stages the `started` line and arms the heartbeat (§5); the existing internal
chunk batching feeds it the rolling tail and token count with no new plumbing.
`end_pass` writes the `body`.

**Labelling.** The seam does not know which pass it is. Add a context stack —
`with activity_log.pass_context("chat_facts", session=fn):` — pushed by callers, with the
per-call `label=` as an override and the current chip's `source` as the fallback. This
degrades gracefully: **zero call-site edits still produces correct records** (labelled by
source), and ~10 edits at the interesting sites (`reflection_runner`'s per-session passes,
`til_wander.run_facts_pass`, `checkin._summarize_recent`, `modules.run_module_blocking`)
make them precise. Do not block the rollout on labelling.

**Live chat is deliberately not on this route.** A chat turn's CoT and reply already land
in the transcript; mirroring them here duplicates the corpus into the telemetry. Chat may
emit `event`-level markers only.

**Hard exclusion — the public API.** `api_http` traffic must produce **no** `body`,
`stream` or `raw` text, ever. "Requests are NEVER logged" is a structural guarantee of
that endpoint (`generation._make_api_generate` forces `log_transcripts=False`), and a
generic hook in the shared seam is precisely how such a guarantee gets broken by accident.
Enforce it at the factory — `_make_api_generate` passes `log_text=False`, and
`begin_pass`/`end_pass` then emit at most a content-free `event` ("api request served,
N tokens"). The `core.api_http` self-test must assert that an API generation adds no
`body`/`stream`/`raw` record. Gossip is the deliberate inverse and stays logged.

### B. The stdout tee — total coverage of everything else, zero call-site changes

Every subsystem already prints in one shape: `[til] …`, `[wander] …`, `[idle] …`,
`[background_reflection] …`, `[watchdog …] …`. So:

```python
activity_log.install_stdout_tee()      # called once at server boot, and in train_cycle
```

wraps `sys.stdout`/`sys.stderr` so each completed line is *also* appended as a `raw`
record with `source` parsed from a leading `[tag]` (falling back to the process name).
The real stream is written through unchanged, so `server.log` is byte-identical to today.

Guards, all necessary in practice:

* **Drop `\r`-terminated partial lines.** tqdm/unsloth redraw progress bars with carriage
  returns; only `\n`-terminated lines are journalled, and the last redraw before the
  newline is the one kept. (This alone is what makes "put unsloth in the log" tolerable.)
* **Cap line length** (2 KB, middle-elided) and **rate** (e.g. 200 lines/s, then coalesce
  into `… +N lines suppressed`), so a runaway loop cannot fill the disk.
* **Re-entrancy guard**: the journal writer must never print, and an exception inside the
  tee must fall through to the real stream.
* **Denylist regex** (config): model-loading banners and other known noise.

### C. Explicit `append()`

Unchanged, and still correct for semantic events that are not a generation: job outcomes,
the reflection mirror, rung transitions. §3(g) is fixed here — the mirror stops packing
text into `message` and emits `level:"body"` with `text`, elided in the **middle** rather
than clipped at the end, so the answer after `</think>` always survives.

---

## 7. The offline job: unsloth in the same journal

This is cheaper than it looks, because two enablers already exist.

**Enabler 1 — the training process can already import the module.**
`training/train_cycle.py` puts `server/inference/` on `sys.path` (line ~148, "same trick
as ledger.py") and imports `core.chat_sidecar` / `core.wander_sft`. So
`from core import activity_log` works from the offline job today, with no new packaging
and no layering change. `train_cycle` calls `activity_log.configure(<same path>)` +
`install_stdout_tee()` at start, and every unsloth/TRL line lands in the one journal,
live, in order, with correct timestamps. On relaunch the inference server's `configure()`
recovers the seq high-water from the tail and simply continues — the history has no hole,
with no backfill step to write.

**Enabler 2 — the watchdog can already serve it.** `watchdog.py` documents its progress
reader as "a generic seq'd-JSONL reader, not training-specific" (line ~64) and takes the
path per job from the git-tracked `watchdog_jobs.json` (`"progress"`, served by
`GET /job/progress?after_seq=N`). Activity records carry `seq`. So pointing the train
job's `progress` at the activity journal with `"reset_progress": false` makes the watchdog
serve the whole unified stream during the training window — **with no watchdog change at
all**, which is the only acceptable kind of change to a process that needs SSH to restart.

Consequences to accept deliberately:

* **Multi-process append needs a lock.** Two processes never run at once *by design*
  (the watchdog stops inference before a job), but "by design" is not "guaranteed", and a
  corrupted journal line is unrecoverable. Wrap the append in `fcntl.flock(LOCK_EX)` and
  re-read the tail's seq under the lock; on timeout, fall back to an unlocked append (a
  duplicated seq is survivable, a blocked training run is not). Cost is one lock plus a
  short read per append — nothing at this grain.
* **`train_progress.jsonl` stays.** It is the stage machine the Sleep tab renders; it
  should *also* mirror its events into the activity journal. If the job's `progress`
  pointer is moved to the journal (recommended), `sleep_widget.TrainPollWorker` and
  `activity_widget` filter by `source`/`level` instead of consuming a training-only file —
  which collapses the two client-side merge paths into one. That is a client change; it is
  small, and it is the difference between two log renderers and one.
* **The seam that remains is honest**: while inference is down the only live server is the
  watchdog, so the client's transport switches from WebSocket to HTTP for that window. The
  *records* are identical, so §10's renderer does not care.

---

## 8. Storage

**Segments, not truncation.** Replace `_rotate_locked()`'s rewrite-down-to-the-ring with:
rename the current file to `activity-<first_seq>-<last_seq>.jsonl` and open a fresh one.
Keep `retain_segments` (default 8) × `segment_bytes` (default 16 MB) ≈ 128 MB, deleting
oldest-first. History survives; disk is bounded.

**Reads past the ring.** `get(after_seq)` walks back into segment files when the cursor
predates the in-memory ring, and the batch carries `gap: true` when even the segments
cannot reach it — a truncated history must be *stated*, never silently rendered as
continuity (§3(e)).

**Boot cost.** `configure()` reads only the **tail** (last ~1 MB) of the newest segment to
recover the seq high-water and refill the ring (§3(f)).

**Snapshot scope — required companion edit.** The journal lives under
`inference/data/hot/`, so it travels in `/export` and in every runnable snapshot today. At
128 MB of telemetry that is a real cost on every *Fetch snapshot*, and the journal is not
state Ava depends on — it is regenerable-by-irrelevance, not by rebuild. **Exclude
`hot/activity/` from `snapshot_state`'s scope** (and note it in `MANIFEST.json`'s list of
what is deliberately absent). The clone-oriented `/export` may keep it.

---

## 9. Serving

* `activity_events` gains `max_level` (default `["event","body"]` — what the Sleep box
  shows) and optional `sources` / `since_ts` filters. Message names are unchanged, so the
  existing Activity tab keeps working against the new server.
* `activity_events_batch` gains `gap` (§8) and `more` — because a batch can now carry
  megabytes, cap it (256 KB) and let the client drain in a loop on its cursor rather than
  building one enormous frame.
* `body` records are served with `text` **elided in the middle** past `body_max_chars`
  unless the client asks for one specific `seq` (a `get_log_entry {seq}` RPC), so the
  common poll stays small and "show me the whole thing" is one click.

---

## 10. UI

The ask is a textbox in the Sleep tab. Note plainly: **the Activity tab already is that
view** — it polls `activity_events` with one cursor from the moment the client connects,
merges the watchdog HTTP stream while inference is down, and has the filter toggles. A
second bespoke textbox is a second renderer to keep in step.

So: extract **one** widget, `client/ui/log_view.py`, and mount the same instance-per-tab
in both places —

* **Activity tab**: box-wide, unfiltered (what it is today).
* **Sleep tab**: the same widget, pre-filtered to the active run's `activity_id` when one
  is running and box-wide otherwise. The Sleep tab's existing bespoke event log becomes a
  *view over the journal* rather than a second stream, which also means a Sleep run
  finally shows the autonomous work that interleaves with it.

Widget contract: a filter row (level checkboxes **Events / Output / Live / Raw**, source
multi-select, substring filter), a monospace read-only box, a **follow tail** toggle
(append with a detached cursor so an operator scrolled up is never yanked to the bottom —
the discipline `modules_widget` already uses), and per-entry expand for `body` text.
Render `<think>` dimmed and the answer bright: the split is already computable with the
family close markers (`generation._CotStreamSplitter`).

---

## 11. Volume budget

Order-of-magnitude, from measured artifacts on this box:

| producer | per unit | notes |
|---|---|---|
| reflection run | ~120 KB | ~20 passes × ~6 KB body; today's run event file is 729 KB, of which the journal mirrors a fraction |
| autonomous wander | ~25 KB | 3 passes (voice, learning, facts) |
| heartbeats | negligible | 3/min while generating |
| train cycle raw stdout | 1–5 MB | after `\r` suppression |
| **busy day** | **~10–30 MB** | 128 MB of segments ≈ a week of scroll-back |

---

## 12. Config

New `logging` block in `server_config.json`, all optional, all with the defaults above:

```jsonc
"logging": {
  "heartbeat_s": 20,          // 0 ⇒ no heartbeat (restores today's silence)
  "body_max_chars": 24000,    // middle-elided past this
  "segment_bytes": 16000000,
  "retain_segments": 8,
  "stream_enabled": true,
  "raw_enabled": true,        // the stdout tee
  "tee_denylist": []          // regexes for known-noise lines
}
```

Every level has an off switch, because the failure mode of a logging change is that it
becomes the reason a run dies, and the operator needs a knob rather than a `git revert`.

---

## 13. Invariants

1. **Logging never fails a caller.** Every path best-effort with a bare `except` —
   already the house style in `activity_log` and `_mirror_to_activity`.
2. **A full disk degrades to memory-only**, it does not raise.
3. **The public API contributes no content.** Enforced at the factory, asserted by
   `core.api_http`'s self-test (§6.A).
4. **The tee never recurses and never deadlocks**; a tee exception falls through to the
   real stream.
5. **`seq` stays globally monotonic across processes** (flock, §7), and a gap in what can
   be served is *reported* (§8), never smoothed over.
6. **`server.log` is unchanged.** The tee observes; it does not replace.

---

## 14. Rejected alternatives

* **Journal every token delta.** A reflection run generates ~100k tokens; at the existing
  ~80-char batching that is ~1250 records per pass and ~25k per run, which blows the
  2000-event ring on every run and makes rotation the dominant cost. The heartbeat plus
  the final verbatim `body` carries the same information for ~1% of the records. Live
  per-token viewing remains available where it already is (the manual buttons stream over
  their own RPCs to the tab that asked).
* **A second log file beside `activity.jsonl`.** Defeats the single cursor, which is the
  property the whole ask rests on.
* **Put the tee / journal logic in the watchdog.** It is the root of the process tree and
  needs SSH to restart; it holds only generic, stable logic. §7 gets the same result with
  a repo edit to a git-tracked manifest and no watchdog change.
* **Structured events only.** The requirement explicitly includes raw unsloth output; a
  schema that cannot carry an unstructured line fails it.
* **Per-run event files as the UI source.** They exist (`reflection_runs/<id>.events.jsonl`),
  they need a `run_id` the client must already hold, and they are exactly what goes silent
  between runs — the original motivation for `activity_log`.
* **Keeping `set_current()` silent and adding a separate "long job" warning.** Two
  mechanisms where the lazy `started` line (§5) is one, with the same anti-flood property.

---

## 15. Rollout

Each step is independently landable and independently useful.

1. **Heartbeat + lazy `started`** (§5). Fixes the observed silence. Touches
   `activity_log` and `idle_scheduler` only.
2. **Segments + `gap` + tail-read `configure`** (§8, §3d–f). Do this *before* volume
   grows; it is data loss otherwise.
3. **`level`/`text`/`pass_id` schema + `body` emission at the two generate factories +
   the API exclusion + self-tests** (§4, §6.A). This is requirement (2).
4. **stdout tee in the inference server** (§6.B). Requirement for everything not on the
   generate seam.
5. **`configure` + tee in `train_cycle`, flock, `watchdog_jobs.json` progress pointer**
   (§7). This is requirement (3) — unsloth.
6. **`client/ui/log_view.py`, mounted in Sleep and Activity; `max_level` on the RPC**
   (§9, §10). This is requirement (4).
7. **Snapshot exclusion, config block, docs**: a row in `documentation/AVA_STATUS.md`, a
   dated entry in `documentation/AVA_CHANGELOG.md`, and the storage/protocol rows in
   `CLAUDE.md`.

---

## 16. Open questions

* **Does the Sleep tab keep its own run-scoped event stream at all**, or does it become
  purely a filtered view of the journal? Purely-a-view is cleaner and is what §10 assumes;
  it means run events must reach the journal at full fidelity (they nearly do — the
  whitelist `_ACTIVITY_MIRROR_EVENTS` would have to go, or become a *level* mapping
  instead of a drop).
* **Should `body` records carry the assembled prompt too?** `generation`'s
  `on_prompt_debug` already produces exactly the labelled segments for this, and "what was
  it conditioned on" is the other half of "what did it produce". It is also the single
  biggest volume item (a reflect prompt can be 20k tokens). Suggest: not by default,
  available on demand via a per-pass flag, since the Debug/`checkin_prompt` surfaces
  already answer it interactively.
* **Retention by time as well as size?** A quiet week and a busy day fill 128 MB very
  differently; an operator asking "what happened last Tuesday" wants days, not megabytes.
* **Does the journal want an index** (`seq → byte offset` per segment) once scroll-back
  over 128 MB becomes a normal operation, or is a linear scan of one 16 MB segment fine?
  Fine for now; note it before someone adds full-text search.
