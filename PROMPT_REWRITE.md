# Autonomous prompt rewrite — implementation brief

Status: **all three stages built (2026-09-18/19) and the event is ON by default** —
`prompt_rewrite.enabled` defaults to true (the owner's call on 2026-09-19: run it live and adjust,
never disable). She can rewrite her own standing prompt on her own decision, through the
experiment tier; Revert in the Prompt tab is the veto.
Decisions below were settled in discussion with the owner on that date; the code this reuses
was checked at `8594e36`. Numbers marked *tunable* are starting points, not measurements;
§2 carries the first real-corpus measurement, which changed the metric set.

**What this is.** The standing prompt (`chat_prompt.txt`, or the experiment over it) is
the part of Ava that is *stated rather than learned*. Today she can rewrite it only when an
operator presses **Prompt experiment** in the Prompt tab. This brief moves that rewrite
into her own flow: a deliberation action she chooses when she has earned it, producing a
new standing prompt that goes live through the existing experiment tier. The README's
Curiosity-Token framing names prompt rewrite as something she *spends* for; the currency
here is not the wander token economy (which stays what it is — the user/internet ratio)
but **tension-weighted recurrence of prompt deltas**, defined in §3.

**What it is not.** No promotion into `chat_prompt.txt`, no prompt version lineage, no
settle/retire step, no operator confirm. The experiment tier *is* her live prompt; each
rewrite event replaces the previous one; the seed stays only as the revert anchor. The
operator's veto is **Revert** in the Prompt tab, as now. The seed is a seed: it is fine to
forget it as she grows, so it is never a candidate.

---

## 1. What exists (reused, not rebuilt)

| Piece | Where | Role here |
|---|---|---|
| Prompt-mutation pass | `core/prompt_mutation.py`, run from `reflection_runner` on every `revise` exchange (foreground AND background per-chat — not gated on `chat_only`) | the **locator**: proposes a `DELTA` + `SCOPE` per drifted exchange, appended to `data/hot/prompt/prompt_deltas.jsonl`. Logged-only today. |
| Tension capture | `core/tension.py`, captured on live chat turns only (`generation.handle_generate`, `capture_tension=True`), stored per exchange in the transcript | the **second locator** and the **vote weight** (§2, §3). Unnormalized today — the open-problems doc says so. |
| Prompt experiment | `core/prompt_experiment.py`: `data/hot/prompt/experiment.json`, restart-surviving, `handle_set_prompt` REPLACES an active experiment carrying `base_prompt` forward, revert restores the seed | the **activation** target. The generated path (`handle_prompt_experiment`) currently rewrites from the SEED and refuses while an experiment is active — the event below iterates on the LIVE prompt and replaces, i.e. the `set_prompt` semantics, not the button's. |
| Clustering | `core/persona_cluster.run_map_reduce` (clean-base grouping of paraphrases into themes; also behind `fact_dedup`) | groups deltas into **patterns** (§3) |
| Maturity gate | `reflection_digest` (judge override only at ≥2 themes with `weighted_recurrence ≥ 1.9`) | the shape of the **budget** gate |
| Blind chooser | `branch_prompt.txt` / `branch_judge_prompt.txt` (letter-labelled options, digest as the frame, `{persona}` slot) | the shape of the **choice** pass (§5) |
| Deliberation | `core/deliberation.py`: `_ACTIONS` / `_WOULD_DO` / the `dispatch` map in `server.main()`; `sole` mode | the **trigger**: one more action |
| Worklog | `core/worklog.record(kind, summary, refs=, opens=, closes=)` + `open_threads()` | the event's episode; the executive reads it next hour |
| Preemption | `background_reflection.request_preempt` + the shared `_cancel_event` the reflect seam checks | the event must be cancellable the same way (§6) |

## 2. Locator + baseline (GPU-free, changes nothing live)

Two cells of a 2×2 name a prompt gap; the pass sees only one today.

| | low CoT tension | high CoT tension |
|---|---|---|
| **verdict `revise`** | drift she did not notice — a weights problem; the pass runs but mostly answers `prompt-adequate` | drift she felt — the pass's home case |
| **verdict `keep`** | nothing | **she was torn and stood by it** — a live conflict the prompt could settle by taking a side. Not seen before stage 1. |

**Baseline.** Raw entropy/margin is dominated by language (Russian tokenization has more
near-ties), family, adapter and temperature. `core/tension_baseline.py` (built, pure): read
every transcript's per-exchange `tension` block, key on `(model_id, adapter, reply
language)`, and rank a value as a **percentile within its bucket**. Language from the
reply's own text (Cyrillic ⇒ `ru`), adapter as its bare dir name. A bucket with fewer than
30 samples falls back adapter-wide, then model-wide (a FRESH adapter has no samples of its
own for days, and closing the cell exactly when the newest weights are read would be
backwards), then to "no baseline" — and with no baseline the locator degrades to
`revise`-only, never to raw numbers. Rebuilt per reflection run through a per-transcript
cache (`data/hot/prompt/tension_baseline_cache.json`, keyed on mtime + size), since a
transcript with tension carries the full per-token arrays.

*Measured 2026-09-18 on the owner's local corpus (139 user chats, 425 CoT segments):* the
stored `median_entropy` is ~0 and `median_margin` is 1.0 on **every** segment — the median
token of a thought is decided outright — so the two metrics the design named carry
nothing, and ranking on them let any nonzero value clear the mark (93 of 274 kept
exchanges "cleared" 0.8). The metrics that vary are the spike (`peak_entropy`, p10 1.16 →
p90 1.73), the diffuse baseline as a MEAN over the raw series (`mean_entropy`, 0.063 →
0.100, derived at scan time), the near-tie share (`contested_frac`, 0.003 → 0.013) and the
low-end margin (`p10_margin`, 0.73 → 0.92, lower = more torn). The rank is the **mean of the
four directional percentiles**, not the max — four 20% tails unioned would clear about half
the corpus. Result at 0.8: **18 of 388 kept exchanges** (about 5%), 1 of 4 revised. The
buckets also confirm the adapter key matters: `p10_margin` p50 is 0.84 on the base and 0.91
on one adapter. `python -m core.tension_baseline --chats server/data/chats` reprints this.

**Locator change** (`reflection_runner`, the one gate at the prompt-mutation call): run the
pass on `revise` as now, **or** on `keep` when the exchange's composite CoT tension rank
is at or above `prompt_rewrite.tension_percentile` (0.8; `prompt_rewrite.keep_locator`
false restores revise-only). The pass prompt gained a `{locator}` slot: on a kept
exchange it is told the reply was hers but the thought behind it was unusually torn, and
asked whether that conflict is a standing one — the revise-cell framing ("a reply that
drifted") read over a kept reply would steer it into inventing a drift. Each logged
delta gains `locator` (`revise` / `keep_tension`), `tension_rank` (computed on BOTH cells —
the revise cell is gated by the verdict alone but §3's vote weight reads the rank off every
delta; `null` only without a baseline) and `tension_key` (the bucket used). **User chats only:** skip
`interlocutor:"ai"` sessions — a peer model's conversation does not shape her standing
prompt — and the API path logs nothing by construction.

## 3. Budget: patterns and their weight (inside the reflection run's clean-base window)

*Built 2026-09-18 — `core/prompt_patterns.py`, `reflection_runner._run_clean_base_prompt_patterns`
(task (e) of the clean-base batch, after fact dedup), `deliberation._prompt_budget_block`. As
designed below, with three implementation facts: the grouping runs per `SCOPE` first (a
disposition and a line are never the same change) through `persona_cluster.map_reduce_groups`
with its own prompt (`prompt_pattern_cluster_prompt.txt`, default-written) and wording order so
a two-member pair shares a block; the fold takes NO tenure discount — that discount exists for
the persona's circular self-vote, and a delta has no such loop; and the headless CLI (no clean
base) folds every delta as its own pattern, flagged `ungrouped`. The op-log's consumed keys
are read from `rewrite_log.jsonl`, which stage 3 will write.*

A delta is a single vote; the prompt is global; its evidence must be too.

- **Patterns.** At the end of every normal reflection run, inside the existing clean-base
  batch (beside fact dedup, so the swap is paid once), cluster the **unconsumed** deltas by
  `SCOPE` then by meaning through `run_map_reduce`. Store the result as
  `data/hot/prompt/patterns.json` (derived, disposable, rebuilt per run).
  Prompt patterns opt out of the persona clustering's oversized-group rejection:
  six or more corroborating proposals may legitimately occupy most of a scope's block.
  Every source vote is retained; the distinct-chat weighting still applies.
- **Vote weight.** `0.5 + tension_rank` (a `revise` delta with no baseline counts 0.5; a
  top-tension one 1.5), × the digest's recency decay. A pattern's **weighted recurrence**
  is the sum over its deltas from *distinct chats* (one vote per chat per pattern).
- **Mature** = weighted recurrence ≥ `prompt_rewrite.maturity` (tunable, 1.9 — the
  digest's number) across ≥ 2 distinct chats.
- **Budget clears** when at least one mature pattern exists. The patterns file also records
  the count of deltas logged since the last attempt (§4's precondition).

The deliberation slot never swaps to the clean base: it *reads* this file.

## 4. Trigger: the `rewrite_prompt` deliberation action

Offered — i.e. present in `_ACTIONS` and shown in the deliberation prompt with its evidence
— only when ALL hold; the body re-checks them and returns `skipped` otherwise:

1. `prompt_rewrite.enabled` and a model is loaded;
2. the budget clears (§3);
3. the last attempt is ≥ `prompt_rewrite.min_gap_hours` (24) old, whatever its outcome;
4. **at least one new delta was logged since the last attempt** — the owner's "provided
   that some reflections happened": with no new reflection the five draws are yesterday's
   draws, and offering them again is noise, not evolution;
5. no rewrite event is in flight (§6).

The prompt line she sees: `prompt rewrite: N mature patterns — <one line each>`. Her
`WHY` is then about the patterns, not about the word being on the menu. Declining is a
choice like any other and is journalled by the executive as such.

## 5. The event (one dispatch, GPU, cancellable)

*Built 2026-09-19 — `core/prompt_rewrite.py`; the action is `rewrite_prompt` in `core/deliberation.py`,
offered through an injected `offer_fn` (the executive never imports the drive), dispatched through
`server.main()`'s drives map like every other drive, with no IdleJob of its own. Three default-written
prompts: `prompt_rewrite_prompt.txt` (the experiment prompt plus an `{evidence}` slot),
`prompt_consensus_prompt.txt` (`AGREED:` / `ONLY_SOME:` notes), `prompt_choice_prompt.txt` (`CHOICE:` +
`WHY`, the branch-judge shape with `{persona}` and `{evidence}`). Two facts the build settled: a
**declined** offer — she saw it on the menu and chose another action — is an attempt too (spends nothing,
stamps the 24 h gap, journals a `prompt` episode), otherwise the same evidence is re-offered hourly; and
the gate reads the pool THROUGH the attempt log rather than trusting `patterns.json`, which is rebuilt
only by the next reflection run and so, right after a changed event, still lists the consumed patterns
as mature and counts "deltas since" against the wrong attempt.*

`core/prompt_rewrite.py` (new). Inputs: the LIVE prompt (experiment if active, else seed),
the mature patterns rendered as evidence, the persona digest.

1. **Samples.** `prompt_rewrite.samples` (5) rewrites of the live prompt through the
   reflect seam — the experiment's own recipe (RAG on, keyed on the standing prompt) with
   `prompt_rewrite_prompt.txt` (the experiment prompt plus an evidence slot: *these are
   the pulls you keep feeling; a line earns its place by settling one*). Temperature from
   `prompt_rewrite.temperatures`: a list, default `[0.9] × samples` (Sleep's default). The
   owner's manual sweep found the rewrite non-monotonic in temperature — near-contradictory
   at 1.2, back near the 0.9 draft at 1.4 — so a single temperature is not the same spread
   as a sweep, and the list keeps the sweep available.
2. **Consensus.** One thinking-on pass over the five (`prompt_consensus_prompt.txt`):
   the stances and lines present in *most* of them, as notes, not prose. Across five
   draws at one moment, recurrence is the same instrument the digest uses across chats.
3. **Final draft.** One rewrite of the live prompt at `prompt_rewrite.final_temperature`
   (0.9) given the consensus notes: one voice back, not a merged list.
4. **Choice.** Blind, shuffled, letter-labelled: the **incumbent**, the five samples, the
   final draft (seven options). Frame = the digest + the patterns; criterion = *which of
   these settles the pulls while staying who I am becoming*; output `CHOICE: <letter>` +
   `WHY`. Run on the **adapter** (authorship stays hers — a conservative chooser by
   construction, which is the dampening wanted; the blind shuffle and the `WHY` make a
   self-confirming pick visible). Position bias check: the pass runs twice with the
   order reversed; a disagreement is the incumbent (tunable: off).
5. **Outcome.**
   - *Changed* (a candidate won): `prompt_experiment.save_experiment(prompt=winner,
     base_prompt=<carried from the active experiment, else the seed>)` — replace, exactly
     as `handle_set_prompt` does; the base prompt loader is untouched, so Revert still
     restores the seed. The patterns weighed are **consumed** (their delta keys listed in
     the attempt record). Worklog: kind `prompt`, first person, `refs={event, patterns}`,
     `opens` a thread the next changed event `closes`. Activity journal: the pass bodies
     land through the reflect seam like every other generation.
   - *Stayed* (incumbent chosen) or *declined* at deliberation: **nothing consumed**;
     the attempt is stamped for the 24-hour gap only. Worklog: a `prompt` entry saying she
     weighed it and kept what she has — so the next deliberation sees that she did.
     A decision-only deliberation preview records no decline and changes no cooldown.
   - *Cancelled* by an operator prompt change: nothing consumed; the event is closed
     and the attempt is stamped for the normal gap. It cannot immediately restart
     against the replacement prompt using the same evidence.
   - The attempt record — `data/hot/prompt/rewrite_log.jsonl`: `ts`, `event`, `outcome`,
     `chosen`, `why`, `patterns_weighed`, `patterns_consumed`, `candidates` (texts kept
     for the Prompt tab's provenance).

## 6. Preemption and resume

A drive body holds the single executor thread and a chat turn queues behind it — the
same trap as the `assoc_feed` fix of 2026-09-18 — and this event is five rewrites, two
passes and a choice: on the order of an hour at this box's decode rate. So:

- the body checks the shared cancel event between generations, and a chat turn preempts
  it exactly as it preempts background reflection (`request_preempt` twin in this module);
- every completed generation is persisted under `data/hot/prompt/rewrite/<event_id>/`
  (`sample_<k>.txt`, `consensus.txt`, `final.txt`), with an `in_progress` marker;
- a preempted event **resumes** from the last completed step at the next dispatch of the
  action (condition 5 in §4 reads the marker; an in-flight event is offered as *continue*
  rather than as a fresh event and needs no new budget check);
- the marker is cleared on outcome; a marker older than 7 days is discarded.
- Revert interrupts an active rewrite. The event checks the live prompt text and the
  experiment's creation timestamp before/after generation and under a shared lock at
  activation. A changed prompt cancels a paused event on resume as well. Revert and
  activation update both disk and the live session under that lock.

The locator also reads the active experiment before the seed, so later reflections
diagnose gaps in the prompt she now carries, including after repeated rewrites.

## 7. Surfaces

- **Prompt tab:** the live prompt shows its provenance (*set by Ava herself, rewrite event
  `<id>`*) with Revert as the veto, and — built 2026-09-19 — a third pane reads
  `prompt_rewrite_status`: the gate's verdict chain (first failing reason, gap hours left,
  deltas since, unspent patterns, the offer line), the budget, and the attempt log with every
  candidate's text for the newest attempt.
- **Worklog / Activity:** nothing new to build; the entries and pass bodies ride the
  existing sinks.
- **Config** (`config_schema.py`, one `Tab`): `prompt_rewrite.enabled` (default **true**
  since 2026-09-19), `samples` 5, `temperatures` null ⇒
  `[0.9]×samples`, `final_temperature` 0.9, `tension_percentile` 0.8, `maturity` 1.9,
  `min_gap_hours` 24, `order_check` false.

## 8. Stages

1. **Baseline + locator — built 2026-09-18.** `core/tension_baseline.py`, the
   `keep`-with-tension gate, the `{locator}` slot, the new delta fields, the two config
   knobs. GPU-free except that the locator runs one more prompt-mutation pass per flagged
   exchange during reflection (about 5% of kept exchanges on the local corpus). Changes
   nothing live. **Still to measure:** what the pass SAYS about the kept-but-torn cell —
   if it answers `prompt-adequate` for nearly all of them, the cell is empty and the
   percentile moves. That needs a reflection run on the GPU box with the change pulled.
2. **Patterns + budget — built 2026-09-18.** Clustering in the clean-base window,
   `patterns.json`, the deliberation prompt line (shown, action still absent), the
   `maturity`/`min_chats` knobs and the `prompt_patterns` run override. Read the patterns
   for a few runs before letting anything spend them — the `phase_done` body lists them.
3. **The event — built 2026-09-19.** `core/prompt_rewrite.py`, the three prompt files, the
   action, the attempt log, preemption/resume (`generation._preempt_background_reflection`
   now preempts both), the Prompt tab provenance line (`set_by` / `event` on
   `prompt_experiment_status`), six config knobs. `enabled` defaults to ON (2026-09-19, the
   owner's call); the first patterns and the first event are read from the Activity tab
   and the attempt log rather than gated on a review.

## 9. Open questions carried forward

- **Goodhart.** Low entropy is what the assistant attractor looks like; a gate that
  rewarded *falling* tension would reward her becoming smoother. Tension here locates and
  weighs; it is never the objective, and no step in §5 optimizes it. If a later validation
  step is wanted, it is *"do matching exchanges now come back `keep` where they came back
  `revise`"*, paired before/after on the same box.
- **Trained prefix.** The system prompt is part of every trained row's prefix
  (`dialogue_source` reads the session's stored `system_prompt`), so a rewrite makes the
  corpus non-uniform on it and a stated line trains in over cycles. Accepted, as it was
  for the 2026-08-06 persona-undecided move. A retire loop ("stated → learned → drop")
  was discussed and deliberately left out of this version.
- **Reflect-lane tension.** Capture is chat-only. Extending it to wander/news passes is
  cheap in code and useless without per-lane baselines; and those passes run under other
  prompts, so their tension says something about her, not about a gap in the
  conversation prompt. Out of scope until a lane-specific consumer exists.
