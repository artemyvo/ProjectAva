# Ava Changelog

This changelog is a curated technical history, not a complete git log. It records changes that affect architecture, interpretation, negative results, and rollback/replay assumptions.

Dates are based on in-code comments, existing design notes, and current repository state.

Last code check: 2026-09-07.

## 2026-09-07

- **DGX Spark (GB10, unified memory): the "10-minute model load" and the two freezes were
  the same box, three causes, none thermal** (`core/unified_memory.py` (new),
  `core/fast_load.py` (new), `core/moe_bnb_experts.py` (new), `server/convert_moe_bnb4bit.py`
  (new), `core/inference_backend.py::load`, `training/train_cycle.py`, wip commit `b7b539d`).
  Measured, warm page cache, torch 2.10.0+cu130 / transformers 5.5.0 / unsloth 2026.8.19:

  | copy | GB/s |
  | --- | --- |
  | host→device, heap tensor | 58 |
  | host→device, straight out of the safetensors mmap | 0.16 |
  | the same slice `clone()`d to the heap first, end to end | 6.5 |

  (1) transformers' loader (`core_model_loading._materialize_copy`) moves the mmap slice
  itself, so a warm 18 GB Gemma load spent 95 of its 112 weight-loading seconds inside
  bitsandbytes' per-tensor `.cpu()` round trip waiting on those copies — 0.16 GB/s is the
  driver handling file-backed pages one at a time on this platform, and it is exactly what
  made the previous 250 GB attempts read as "1 tensor/s". `fast_load.install` clones out
  of the mmap before the device move (one memcpy from the page cache) and sets
  `HF_HUB_OFFLINE` for a model already on disk (84 Hub HEAD requests, 12 s, per load).
  Gemma-4-31B: 144 s → 14.7 s, generation unchanged. Applied on unified memory only
  (`AVA_FAST_LOAD=0/1` overrides). (2) CUDA's free-memory figure here is the pool's
  *MemFree*, which excludes the page cache the previous load filled: after one load it
  read 16 GiB free of 121 with 97 GiB reclaimable, accelerate planned an offload to "CPU"
  (the same memory) and bitsandbytes refused ("Some modules are dispatched on the CPU or
  the disk"). `unified_memory.load_kwargs` pins `device_map={"": 0}` when the CUDA total is
  the system total. (3) transformers' bitsandbytes path converts `nn.Linear` only;
  Qwen3.5-122B-A10B's experts are fused 3-D `nn.Parameter`s holding 232 of its 250 GB, so
  `load_in_4bit` left them bf16 — a model that cannot exist in 121 GB, which crawled into
  swap and took the box down (twice; the second time compounded by an 8-way `nvcc` build I
  had left running beside a 60 GB load, which is the same failure shape and is why nothing
  memory-heavy runs concurrently on this box any more). `convert_moe_bnb4bit.py` streams
  the bf16 checkpoint once (~200 s at 1.2 GB/s) into unsloth's gpt-oss layout — every expert
  a pre-quantized `Linear4bit` under `experts.gate_up_projs.<i>` / `down_projs.<i>`, every
  other Linear the loader would quantize pre-quantized too, `mtp.*` dropped — 66 GB,
  150,024 tensors, with the skip list in `quantization_config` and a marker
  (`unsloth_perexpert_bnb4bit_experts`) the loader keys on; `moe_bnb_experts` is the
  matching experts class (same forward signature as the stock one, empty `gate_up_proj`/
  `down_proj` buffers so `_init_weights` is a no-op), swapped into transformers before
  the model is built. One more thing had to go: transformers recurses into the
  `language_model` submodule (`qwen3_5_moe_text`) and imports its `^model.language_model →
  model` WeightRenaming into the multimodal class, where it renames each bitsandbytes
  weight group to a name the model lacks — the packed `weight` then loads raw and the five
  quant-state tensors are dropped as UNEXPECTED, so the first GDN projection failed on
  `F.linear` with a `[1, 18874368]` weight. `patch_conversion_mapping` removes that rename
  for the text type (exact for the ConditionalGeneration class, whose parameters ARE
  `model.language_model.*`). Verified after the fix: zero unexpected keys, 24,876
  `Linear4bit` modules all carrying quant states, three sampled weights (two experts, one
  attention projection) dequantizing to the bf16 source at 0.09 relative error — NF4's
  expected figure — and coherent generation; ~100 s load cold, 62 GiB resident. Decode was
  ~1.3 tok/s on transformers' GDN torch fallback; building `causal-conv1d` 1.7.0 from
  source (alone on the box, 4 jobs, ~10 GB peak) enables the fast path
  (`is_fast_path_available` True with the already-installed `fla`) and takes the same
  40-token generation from 30 s to 9 s (~4.5 tok/s), output identical. The experts still
  run a per-expert `Linear4bit` loop — the remaining speed item.
  **Also validated:** `unsloth/gpt-oss-120b-unsloth-bnb-4bit` loads and generates through
  the backend (58 GB, 65 s from cold disk, 58 GiB resident). The raw `openai/gpt-oss-120b`
  is unusable here — unsloth loads MXFP4 natively and, without triton `kernels`,
  dequantizes to 234 GB bf16 — and the id in the config must be the bnb one.
  **Training side, same commit:** `train_cycle` takes the same placement/clone/experts
  fixes; `_lora_target_modules` restricts a per-expert-Linear MoE to attention + GDN
  projections (unsloth expands a broad `gate/up/down_proj` request onto every per-expert
  Linear — 24,576 modules, ~7 B LoRA params at r=32 plus fp32 Adam, which this box cannot
  hold beside the model; the shared expert is left out because naming its leaves is what
  triggers the expansion); and a **`harmony` family** for gpt-oss (`render.to_harmony_messages`,
  `_MARKERS["harmony"]`, the row-safety and debug-dump branches, `_normalize_thinking`
  for the special-token form): the template renders a `thinking` field as the analysis +
  final channels on the LAST assistant turn only and `final` alone on earlier ones — the
  CoT-stripped history inference builds — so the target's `<think>` moves into that
  field, with `thinking` always set (an answer-only target renders an empty analysis
  channel, the harmony analogue of gemma's empty closed channel, since the system header
  the model is prompted with says a channel must be included on every message). Rendered
  with the same `reasoning_effort` inference passes. **Validated teacher-forced** (the
  render's whole claim is parity with what the model emits): gpt-oss's own 700-token
  generation — analysis channel through to final — normalized to `<think>…</think>` +
  answer and re-rendered through `render_example_text("harmony")` scores **0.125 NLL** on
  its assistant span, against **2.27** for the control that leaves the `<think>` block
  literal in `content`; the Qwen3.5 ChatML render scores 0.32 on its own generation the
  same way. LoRA forward/backward runs on both (gpt-oss: 144 attention modules, 58.4 GiB
  peak; Qwen3.5: 156 attention+GDN modules, 62.2 GiB peak, no expert trainable). No live
  training cycle has run on either yet; gpt-oss streaming still shows the analysis text
  as answer until the end (the known `_CotStreamSplitter` limitation — unchanged). **Pre-existing, noted:** `training.selftest`
  fails at tier 5 on the unpatched tree too (empty replies in the language-drift tier).

## 2026-09-03

- **A client disconnect releases the active chat; "New chat" counts as activity**
  (`core/session_ops.py::release_active_session` (new), `::handle_clear_context`,
  `inference/server.py::handle_client`). Reported as a UI glitch: a user away for
  hours returns, presses New chat, and background reflection starts on the chat they
  just left. The cause was upstream of the UI. The active transcript is fenced from
  every background reader — `background_reflection.list_backlog`, the sidecar
  writers' `live_session`, the stale-reach-out exemption, chat RAG's self-reference
  fence — by "the logger still points at it", and nothing but New chat ever cleared
  the logger, so the chat stayed locked for the whole absence while the box idled
  with nothing else to do; New chat then released it into a backlog whose 30-min
  idle window had long since elapsed, and the job fired within its 120 s re-arm.
  Two changes. (1) `handle_client`'s cleanup calls `release_active_session` on a
  GENUINE disconnect (fenced on `_active_ws` identity, so a superseding client keeps
  the session as that handoff was designed to): logger, conversation and surfaced
  keys dropped, RAG fence cleared, no new session started. Safe because the client
  already resets to an empty conversation on every connect — server-side continuity
  across a disconnect was a phantom in which a post-reconnect turn appended to a
  transcript the UI no longer showed. An in-flight generation is unaffected (it
  holds its own logger reference and the close already set the cancel event).
  (2) `handle_clear_context` marks activity, so a hand release is never the cue for
  an idle job while the operator is at the keyboard. **Residual, stated:** a client
  left connected but idle still fences its chat until New chat; releasing on a timer
  while connected would need a server→client "your view is stale" message, since the
  UI would otherwise keep showing a conversation the server had forgotten.

## 2026-08-31

- **The allocator probe reached the inference server; the facts fetch retries an OOM
  once and names a repeated one `oom`** (`core/alloc_guard.py` (new),
  `inference/server.py`, `reflection_run.py`, `training/train_cycle.py` (delegates),
  `core/generation.py::_fetch_facts_block_sync`, `core/fact_fetch.py`,
  `client/ui/chat_widget.py`). The 2026-08-26 lesson — the expandable-segments env
  route fails silently, so probe `is_expandable` empirically and force via
  `_set_allocator_settings` if inactive — had landed only in the train cycle, while
  the process that runs for DAYS and was already OOMing the same way (the reclaim
  entry below records `[facts] fetch skipped: generate_failed` at ~30 GiB) merely set
  the env vars and trusted them, printing nothing. The probe now lives in ONE place
  (`core/alloc_guard.ensure_expandable_segments`) and runs at every CUDA entry point
  before anything model-sized is allocated: the server's `main()`, the headless
  reflection CLI, and the cycle. Every `server.log` boot therefore states the
  allocator mode as a fact.
  The live symptom this was built against: the stage-1 facts fetch intermittently
  OOMs complaining of fragmentation with ~3 GiB nominally free — the split-segment
  fingerprint — while the turn itself survives, because torch's allocator releases
  its fully-free cached blocks before raising and the backend's OOM path
  (`stream_generate`) then gc's + empties the cache, so the MAIN generation moments
  later succeeds in the pool the failed fetch just cleaned: the fetch was acting as a
  sacrificial defragmenter and donating the result. Now it takes one shot at that
  cleaned pool itself — a CUDA OOM (and ONLY an OOM: a deterministic error retried on
  the GPU would double a live turn's time-to-first-token for nothing) is retried once
  after an `empty_cache()`.
  **Interpretation note:** `skipped` may now carry a new value, `"oom"` — classified
  in `fact_fetch._run` by type-or-message (`alloc_guard.is_cuda_oom`), so all four
  fetch lanes (chat / gist / ask / re-read) report it, added to `FETCH_FAILURES`
  (red in the Chat tab, printed under `[facts]`); the retry itself is live-chat only.
  The operator's move on an `oom` is the allocator (`[alloc]` in server.log), not the
  prompt — which is the whole reason it stopped hiding inside `generate_failed`.

## 2026-08-26

- **Reach-out shyness + paraphrase repetition: the outreach decision now sees the
  questions she already raised, and gains an `asked` outcome that retires a
  duplicate** (`core/outreach.py`, `core/synthesis.py`,
  `core/reflection_memory.py::recently_raised`,
  `core/reflection_writer.py::write_resolution(reason=…)`,
  `prompts/outreach_prompt.txt`). Two live observations, opposite signs of one
  blindness. *Shyness:* the decision pass runs RAG-on, so her open asks come back
  as memory records with no framing, and she read them as a queue she was jamming —
  unanswered questions became a reason to decline NEW questions on unrelated
  topics (the structural side already handles rotation: `_pick_ask` walks past
  dangling asks). *Repetition:* the ask pool dedups by exact `content_key`, so
  paraphrases of one question accumulate as distinct live asks that each clear the
  key-scoped gates (`min_reask_hours`, the dangling-opener guard), and synthesis —
  RAG-off, facts fetch knowledge-only — re-derived questions the pool already
  carried. Fixes: (1) the outreach prompt's new `{standing}` slot carries
  `_standing_questions_block` — the asks surfaced within 14 days (newest 6,
  dangling ones annotated) plus the framing that the user answers what they choose
  to and a hanging question is no debt blocking a different one (a customized
  prompt file lacking the slot gets the block prepended, contract kept last);
  (2) `DECISION` gains a fourth outcome `asked` — a re-asking of a question
  already put to the user is evicted (`write_resolution(reason="duplicate")`,
  ANSWER naming the earlier question), because without the evict the
  fewest-raised-first rotation re-picks the never-surfaced duplicate every wake;
  matched before `resolved` in the parser since its natural phrasings share the
  "already" prefix; (3) synthesis's analysis is shown the live pool
  (`_open_asks_block`, newest 12, folded into the `{persona}` slot per the
  check-in `{recent}` precedent and fenced like the persona block, since the
  bullet list is shaped exactly like the `[ask]` lines its contract requests).
  Interpretation note for the op-log: `evict` records may now carry
  `reason:"duplicate"` — the fold treats every evict identically, the reason is
  provenance. The idle classifier counts a `duplicate` outcome as a spent
  interval (it cost a full decision generation), and the Sleep tab renders it.

- **The CUDA fragmentation guard was silently OFF on pre-2.9 torch since 2026-06-13;
  both allocator-conf names are now set** (`training/train_cycle.py`,
  `inference/server.py`, `reflection_run.py`). Commit `9c80453` (2026-06-13) renamed
  the `expandable_segments:True` setting from the legacy `PYTORCH_CUDA_ALLOC_CONF` to
  the torch ≥ 2.9-only `PYTORCH_ALLOC_CONF` — which a pre-2.9 torch **ignores
  silently**, so on such a box the guard would be inert with nothing saying so (the
  venv resolves torch from the BASE interpreter's site-packages where it was created
  with system-site-packages fallthrough, so a box's torch version is the base
  install's, not the venv's). Fix: `setdefault` BOTH names with the identical value
  at all three CUDA entry points — new torch reads the new name, old torch the legacy
  one (still honoured on ≥ 2.9, deprecation warning at most), and identical values
  can never conflict. `reflection_run.py` had the inverse gap (legacy name only —
  functional today, broken whenever the deprecation becomes removal) and now sets
  both too. **Diagnosis history, same day (final state):** this entry was first
  written as the explanation of the 2026-08-26 train OOM on the training box, then retracted
  when the training box's torch turned out to be 2.9.1 ("≥ 2.9, so the new name was read") —
  and the retraction was itself wrong. Empirical probe on the training box (2.9.1+cu128,
  native Linux: set `PYTORCH_ALLOC_CONF=expandable_segments:True` before importing
  torch, allocate, read `is_expandable` off the memory snapshot) → **False**:
  **torch 2.9.1 does not honor the new name**, so the guard HAS been inert on this
  box since 9c80453 after all — the original diagnosis was right in substance and
  only its version framing wrong (the name is evidently newer than 2.9, or its
  parsing differs; the "≥ 2.9" claim in the old comment was never verified).
  Same-day follow-up probes confirmed the fix chain end to end: the LEGACY name
  **does** engage the mode on 2.9.1 (torch even emits a deprecation warning
  pointing at the new name it ignores — that warning in a log is therefore proof
  the working name was set), and the runtime
  `torch.cuda.memory._set_allocator_settings` force engages it too, so
  `_ensure_expandable_segments`' fallback is real.
  Everything downstream fits: split-segment stranding over long runs, `empty_cache`
  structurally unable to release it (see the reclaim-callback entry below), two
  builds dying mid-run the same way. **Lesson recorded:** an allocator-mode env var
  fails silently in both directions — never trust the name/version claim, probe
  `is_expandable` empirically (which `_ensure_expandable_segments` now does at
  every cycle start, entry below).

- **The allocator mode is now probed and forced at cycle start** (`train_cycle.py`
  `_ensure_expandable_segments`, called at the head of every non-dry `run_cycle`) —
  after BOTH prior fixes failed to save a run (see the corrections in this entry and
  the one above). The reclaim callback below **did not help**: the rerun with it in
  place died identically (step 279/1836, 498 MiB request, **6.92 GiB**
  reserved-but-unallocated, only 22.62 GiB genuinely allocated). That negative
  result is itself diagnostic — `empty_cache()` can only release *fully-free*
  segments, so free memory trapped inside SPLIT segments (partly used, partly free)
  is untouchable, and split-segment trapping is precisely the **non-expandable**
  allocator's failure mode. Fingerprint conclusion, since **confirmed by direct
  probe** (see the entry above): expandable segments were *requested* yet **never
  engaged** — torch 2.9.1 ignores the `PYTORCH_ALLOC_CONF` name the code had been
  setting — and nothing anywhere verified engagement.
  Now the cycle allocates a probe tensor, reads `is_expandable` off the memory
  snapshot, and if inactive forces the mode through the runtime API
  (`torch.cuda.memory._set_allocator_settings`, applying to all segments created
  after — nothing model-sized exists yet at that point), re-probes, and prints the
  verdict plus the env values as the job process actually inherited them (a var
  pre-set in the watchdog's environment defeats `setdefault` upstream — one of the
  candidate causes). Every future `train.log` therefore states the allocator mode
  as a fact instead of an assumption.

- **Training reclaims the CUDA cache every 50 steps** (`train_cycle.py`
  `_CacheReclaimCallback`) — written as the fix for the 2026-08-26 training-box OOM
  after the entry above's first diagnosis was disproven; **itself shown
  insufficient the same day** (see the entry above), kept as cheap hygiene. What the logs show: **the box runs at
  its ceiling across subsystems** (three train builds died at 29.6–30.5 GiB in use on
  the 31.35 GiB card, and the inference side OOMs the same way — `[facts] fetch
  skipped: generate_failed` at ~30 GiB), and **within a long run the stranded memory
  creeps**: the July OOMs died with ~27 GiB genuinely allocated and ~2 GiB
  reserved-but-unallocated, while the 2026-08-26 run died at step 284/1832 with only
  23.26 GiB allocated and **6.40 GiB** stranded — freed-but-cached blocks (offloaded-
  checkpointing side-stream pins, pool splits) accumulated over ~29 min of
  1.1k–8k-token rows, with expandable segments *active* (native Linux, torch 2.9.1,
  no "not supported" warning in the logs; expandable segments bound fragmentation
  within a segment but never unmap pages mid-run). The callback's `empty_cache()`
  releases every cached free block — nothing in use is touched — for one device sync
  per 50 steps against a ~20 s/it loop. **Not the CE-chunk OOM family** (those hit at
  step 1–4 with GiB-scale requests); the `UNSLOTH_CE_LOSS_TARGET_GB` revert below
  stands on its own. **Operational note for a box still at the ceiling:** the
  remaining lever for an activation-side OOM (traceback in gradient-checkpointing
  backward) is lowering that box's `train_max_seq_length` (8192 → 6144/4096), and
  with the CE target back at `0.5` the old corollary against doing so no longer
  bites — the worst-case CE chunk is `2 × 0.5 = 1.0` GiB at ANY qlen, so shrinking
  the cap can no longer raise the CE peak.

- **`UNSLOTH_CE_LOSS_TARGET_GB` reverted `1.0` → `0.5`** (`training/train_cycle.py`) —
  taking the escape hatch the 2026-08-03 entry names. That raise was explicitly "a bet
  on headroom, not a validated ceiling", spending the ~4 GB Gemma4-31B training left
  free on fewer sequential CE GEMMs; the bet did not hold, so the value returns to the
  known-safe `0.5` (worst case one uncapped `2 × TARGET_GB` = 1.0 GiB chunk, the
  2026-07-31 retreat after run `20260731_112845`). The standing guidance is unchanged:
  an OOM here is answered with this knob, never by lowering `train_max_seq_length`,
  which moves rows *into* `get_chunk_size`'s round-to-zero band and can raise the peak.

## 2026-08-23

- **Persona echo gates, part two: tension-visible synthesis, a weighted flip gate, and
  an escalating-`not:` screen** — the follow-up to the entry below, closing what it
  named as unaddressed. **(1)** Opposition now reaches the digest synthesis pass *and*
  the injected render: a polarity-split theme carries `opposes` (its target's anchor
  key) and renders as an indented "— though you have also said: …" line under the
  tendency it pushes against (shared `_evidence_lines` behind both prompt-input
  builders; own bullet when the target faded), a counter-carrying theme gets a
  `contested` note, and `persona_digest_prompt.txt` gains the pairing rule that
  consumes both: tendency + your own resistance = ONE disposition, the resistance as
  its `when`/`not` gate or a LINES entry. That is also the answer to the
  STANCES-vs-DISPOSITIONS injection asymmetry — the counter-current lands in the
  "— though not …" clause the renderers already inject, rather than STANCES being
  re-added to the chat render (kept out deliberately; standing assertions invite
  recitation). `counter_recurrence` and `opposes` persist on the digest artifact.
  Stated limitation: pairing exists only where the screen caught a merge — themes the
  direction-aware cluster prompt keeps apart from the start are not cross-linked.
  **(2)** `_digest_maturity_gate` now reads `weighted_recurrence` (≥ 1.9 in ≥ 2
  themes) instead of raw `recurrence` ≥ 3: raw distinct-session counts are exactly
  what the echo loop inflates without bound — the live 2026-07-15 digest held the
  flip open with 19 themes at raw ≥ 3, all restating one trait — while weighted is
  tenure-capped at 2.5, fades unrefreshed, and drops under pushback/polarity
  counters; 1.9 preserves the old bar's intent (three *recent* sessions = 1.96 pass).
  **Interpretation note:** a digest written before this date carries no
  `weighted_recurrence`, and such a theme counts as immature — the criterion flip is
  DORMANT on every pre-2026-08-23 digest until a regeneration under current code
  (verified against the live digest, which no longer opens the gate). **(3)** A
  disposition's `not:` line that escalates the habit instead of restraining it
  ("not: simply apologize … or return to boring accuracy" — every gate on the live
  digest) is now cleared before the digest is written: `synthesize_digest` runs
  `screen_disposition_gates` (one thinking-off call over the do/not pairs,
  `persona_gate_prompt.txt`, `ESCALATES: <numbers>` / `none`), the cleared text kept
  on the disposition as `not_rejected` for audit — clearing is fail-safe, since a
  false flag costs a displayed boundary while a missed one injects a standing
  instruction to escalate. Fail-soft, and it rides the same
  `overrides.persona_polarity` kill-switch as the theme screen; it runs on the
  adapter by necessity (the gate does not exist until synthesis, after the clean-base
  window closed) — a clean-base re-check on the next run would be stricter, not
  built. (`inference/core/reflection_digest.py`,
  `inference/core/reflection_runner.py`, `inference/core/reflection_config.py`,
  `prompts/persona_digest_prompt.txt`, `prompts/persona_gate_prompt.txt`; `9fb5224`)

- **Two persona echo-chamber gates: the revision judgement no longer retrieves live
  `[persona]`, and clustered themes are screened for polarity.** A review of the live
  2026-07-15 digest confirmed the suspected echo loop: every disposition was the same
  escalation trait, and the counter-current statements were being counted as SUPPORT
  for it — the rec-63 theme's members included a switch-the-analysis-OFF statement
  merged under the "физическое удовольствие от переусложнения" representative, whose
  content is all the synthesis pass ever sees (`build_digest_prompt_input` lists
  representatives only). Two changes, one per mechanism. **(1) Derivation fence:** the
  reflect generate factory gains `rag_include_persona`, and the three revision-judgement
  generations (main, format retry, language-guard re-judgement) pass it `False` — the
  pass that mints `[persona]` statements was being shown the live `[persona]` set
  embedded closest to the judged exchange, so a restated stance counted as an
  independent distinct-session vote; `_TENURE_DECAY` discounts that loop after the
  fact, and this prevents it at the source, the rule the user-notes pass already
  applies to impressions (`rag_include_impressions=False`). The IDEAL re-answer keeps
  its deliberate persona conditioning (`persona_context_fn`) — the fence is on
  judgement/formation only. **(2) Polarity screen:** grouping is by underlying trait,
  and "I tend to X" / "I resist X" are one trait pointing opposite ways, so a theme
  could swallow its own opposition — votes transferred, content erased. The cluster
  prompt now names DIRECTION as a grouping rule, and
  `reflection_digest.screen_theme_polarity` enforces it inside `cluster_for_digest`
  (so Sleep runs, `digest_dryrun` and `regen_persona` all get corrected evidence): one
  thinking-off clean-base call per merged theme (`persona_polarity_prompt.txt`,
  `OPPOSE: <numbers>` / `none`), flagged members split into their own theme via the
  `member_keys` join, and their sessions returned as a counter plan the runner writes
  as `ledger.counter` ops against the representative — the persuasion channel's
  **second producer**, beside next-turn user pushback, and the first self-generated
  one. Counters land in the run's own ledger dir (discarded with a discarded run) and
  the fold dedups `(key, session)` pairs, so re-planning the same split is idempotent;
  fail-soft per theme; kill-switch `overrides.persona_polarity`. Interpretation note
  for old artifacts: any digest written before this date may carry themes whose
  recurrence includes opposition votes — a regeneration under current code re-derives
  them clean. Named but deliberately not addressed here: the STANCES-vs-DISPOSITIONS
  injection asymmetry, `not:` gates that escalate, and the flip gate's raw-recurrence
  basis. (`inference/core/generation.py`, `inference/core/reflection_runner.py`,
  `inference/core/reflection_digest.py`, `inference/core/reflection_config.py`,
  `prompts/persona_cluster_prompt.txt`, `prompts/persona_polarity_prompt.txt`,
  `client/ui/sleep_widget.py`, `client/ui/persona_widget.py`; `f85185c`)

- **The Persona tab can regenerate and activate the persona digest** (a **Regen
  persona…** button; `regen_persona` RPC). The digest could previously be re-derived
  on demand only as a preview (`digest_dryrun`, which writes nothing) or as part of a
  full Sleep run; after curating the evidence on this very tab the operator had no way
  to fold the cleaned set into a live portrait short of launching a run. The handler is
  the dry run's WRITE counterpart, driving the reflection run's own three seams on the
  same models — `plan_digest` forced, `cluster_for_digest` on the CLEAN base,
  `synthesize_digest` on the ADAPTER (the run's write seam, so the on-disk artifact
  cannot drift from the live path) — and then minting + activating the version via
  `snapshot_state.produce_persona(run_id="regen_<ts>", activate=True)`, the same call
  the training-lite `persona` stage makes, because live chat reads the self-portrait
  through the ACTIVE-PERSONA pointer and a written-but-unminted digest reaches no turn.
  Adapter unchanged; previous persona snapshots stay as rollback; the snapshot tail is
  best-effort (`activated: false` + `persona_error` on failure, digest kept). The
  `regen_` run-id prefix marks the lineage entry as an operator regeneration. Client
  streams stage + token progress into a non-modal dialog; the button confirms first and
  refuses to run over unsaved local removals. (`inference/server.py`,
  `client/ui/persona_widget.py`, `client/ui/memory_editor.py`,
  `client/core/backend_client.py`; `8d2c848`)

## 2026-08-22

- **Training review can regenerate with a previous adapter from the lineage** (an
  **Adapter** dropdown on the regenerate bar). The regenerate flow was pinned to the
  *currently loaded* adapter, which is exactly the wrong tool when that adapter is
  itself the product of a bad build — the repair loop's re-answers carried the same
  poison the operator was trying to remove. The review payload now lists the on-disk
  `models/` lineage (`training_review.list_adapters`: every dir holding an
  `adapter_config.json`, minus `snapshots`/`candidate`, annotated from `builds.jsonl`
  with its build id + whether the forensic snapshot survives, `current` flagged from
  `server_config.json`; bare dir names, never paths, so the list is meaningful after
  a remote fetch), and `regenerate_exchange` accepts an `adapter` name the server
  resolves + path-guards under its own `models/` (`generation._resolve_regen_adapter`).
  The swap reuses the established clean-base machinery — `agentic.CleanBaseSession`
  gained `swap_adapter_id` (default `None` keeps the historical adapter-OFF behaviour
  exactly), so the alternate adapter rides the same release → load → restore
  discipline, failure safety, and default-precision limit as the reflection run's
  clean-base batch: two full model reloads bracket the one generation, correctness
  over speed, and the box is back on the loaded adapter before the terminal message.
  New `regenerate_status {text}` messages narrate the reload milestones into the
  review dialog (a reload is minutes of otherwise-total silence in a dialog that
  opened at the button press), and `exchange_regenerated` gained an `adapter` field
  naming who answered. The swap window shares CleanBaseSession's hazard — the shared
  runtime transiently holds the wrong (or no) model — but unlike a reflection run
  nothing refused chat around it, so `generation._regen_swap_active` (set on the
  event loop around the dispatch) now fences it: `_run_generation` refuses a
  concurrent chat/ephemeral turn (which captures the model reference on the event
  loop and could pin the released model in VRAM through the swap load, or answer and
  LOG a live turn under an adapter that is not Ava's current one), and the flag rides
  the gossip + public-API `is_busy` predicates. Applying a reviewed re-answer is
  unchanged — the sidecar write does not record which adapter authored the target.
  (`agentic.py`, `training_review.py`, `generation.py`, `server.py`,
  `backend_client.py`, `training_review_widget.py`.)

  **Fixed same day (first live attempt OOMed on the swap load and left the box with
  NO model):** `_regenerate_exchange_sync` opened with `model = _runtime.model` —
  a presence check that bound a local alive across the whole swap, which is
  precisely the trap CleanBaseSession's comments document: `release()`'s `del`
  reaches only its own locals, so the frame kept the released model resident in
  VRAM through the alternate-adapter load (→ OOM), and then through the failure
  path's restore load too (→ second OOM, box left empty). The check now reads the
  runtime without binding. Defense in depth in `CleanBaseSession.__enter__`'s
  failure path: the original failure is logged, its traceback DROPPED (the frames
  pin the failed load's partially-materialized tensors) and `reclaim()` run before
  the restore load, so a genuine OOM's debris can't cascade into the restore; a
  restore that still fails is logged as its own no-model event while the ORIGINAL
  error propagates (the restore failure is usually its echo), and the regen error
  message then states outright that no model is loaded and to reload from the Chat
  tab. The swap/restore failure contract is now locked by a fake-backend case in
  `python -m core.agentic`.

  **Revised same day: the swap is now STICKY** (operator request). The
  swap-and-restore design cost two full reloads per regeneration for a restore the
  very next row undid — a repair session is a run of regenerations under ONE chosen
  adapter. The swap is now `agentic.swap_model`, a one-way release → reclaim → load
  primitive that `CleanBaseSession.__enter__` delegates to (one definition of the
  swap and of its failure-restore discipline; `__exit__`'s restore-on-exit is the
  context manager's only remaining own behaviour, and the reflection run's
  clean-base batch is unchanged): the first regeneration under a lineage choice pays
  one reload, the adapter stays loaded, and `_resolve_regen_adapter` then sees it as
  the loaded adapter so further rows swap nothing. **Deliberate consequence:** the
  box keeps running the chosen (stale) adapter after the operator leaves the tab —
  chat, idle jobs, a background reflection, everything — accepted for code
  simplicity; `status.adapter_id` reports it honestly, the tab/dialog/tooltip say it
  out loud, and a Chat-tab load or a server restart returns to the config's adapter
  (the config is never touched). `exchange_regenerated.adapter` now reports the
  authoring adapter off the RUNTIME rather than the request, since under sticky
  semantics a follow-up regeneration's request names no swap. `regenerate_status` is
  only emitted when a reload actually happens.

- **Fresh chats can ride the build snapshot as untrained preview rows** (Sleep →
  **Include fresh chats**, off by default). The Training review tab's corpus is the
  latest build's `sft_render.jsonl`, which by deliberate contract held only what
  trained ("a real build must never claim that quarantined or mask-filtered rows were
  trained") — so a reflected chat still inside `rag_only_window_h` (~24 h, LR
  multiplier 0, no row) and a background-frozen (`chat_reflected`-only) chat were
  invisible to review until they aged into a later build, by which point their turns
  had already trained once: the repair loop could only ever run *after* the poison
  landed. The new checkbox forwards `train_params.include_fresh` →
  `--include-fresh` → `build_dataset(include_fresh=True)`, which emits those chats'
  derived targets as **preview rows** (`BuildRow.preview`, multiplier 0, one row per
  exchange, no contamination pair). `train_cycle` splits them out immediately after
  assembly — the fingerprint, tokenizer, scheduler, probe anchors and quarantine
  never see them, and the selftest asserts the trained corpus fingerprints
  byte-identical with the flag on — and re-appends them to `sft_render.jsonl` at
  BOTH writes (assembly and the post-masking rewrite), trailing the trained rows
  with `preview: true`. **Interpretation of existing artifacts shifts one notch:** a
  render row without the flag trained, exactly as before; a `preview: true` row
  explicitly did not (build_meta/progress/returns carry `preview_rows` counts).
  `training_review.build_payload` passes the flag through; the tab tags the entries
  `▷fresh`, counts them as untrained in the status line, and gains an "Only ▷ fresh"
  state filter — every repair control works on them, and a repair locks ❄ the
  exchange, which re-reflection, revisit AND the pending clean-base judge already
  honor, so an early repair survives into the real build that eventually trains the
  chat. A chat never reflected at all still cannot appear: there is no derived
  target to review. Files: `training/build_dataset.py`, `training/train_cycle.py`,
  `training/selftest.py`, `inference/core/reflection_service.py`,
  `inference/core/training_review.py`, `client/ui/sleep_widget.py`,
  `client/ui/training_review_widget.py`.

  **Follow-up (same day): the checkbox works without training — the training-lite
  preview snapshot.** As first shipped the checkbox was greyed out with Train
  (LoRA) off, on the reasoning that only a train cycle mints a snapshot — which
  inverted the workflow the checkbox exists for: reflect a new batch of chats
  WITHOUT training, review and repair the corpus (fresh chats included), and only
  then run the real cycle. With the render minted only by training, there was
  nothing to review until the very training the review was meant to precede had
  already happened. Corpus assembly is pure and model-free, so a committed
  train-less run with the box checked now has the *inference process* write a
  GPU-free **preview snapshot** (new `training/preview_build.py`, called from a
  best-effort `preview` phase in `reflection_service` after the persona stage):
  `models/snapshots/preview-<ts>/` holds the NEXT build's full corpus —
  would-train rows at their real multipliers, fresh rows trailing as
  `preview: true` — rendered through the shared `build_dataset.row_render_dict`
  (extracted from `train_cycle`'s inline projection, so the two producers of
  `sft_render.jsonl` cannot drift), plus a `build_meta.json` whose
  `outcome: "preview"` states that NOTHING in the snapshot trained. The review
  tab picks it up by mtime like any build; `build_payload` forwards the outcome
  and the tab's status line says "PREVIEW: nothing trained yet". Only the newest
  `preview-*` dir is kept (previews are disposable derivations of live state);
  real `build-*` snapshots are never touched, and the next real cycle's snapshot
  supersedes the preview by mtime — its `corpus_fingerprint` matching the
  preview's `build_meta` confirms the operator trained exactly what was
  reviewed. The checkbox is ungated client-side; with Train on the behaviour is
  unchanged (preview rows ride the real build's render). Files:
  `training/preview_build.py` (new), `training/build_dataset.py`,
  `training/train_cycle.py`, `training/selftest.py`,
  `inference/core/reflection_service.py`, `inference/core/training_review.py`,
  `client/ui/sleep_widget.py`, `client/ui/training_review_widget.py`.

- **The answer side of the ask loop is conditioned on the question's origin.**
  When Ava opened a conversation herself (outreach / synthesis), the question arrived
  stripped of everything that produced it: the reply turn conditioned on the opener
  alone, and reconnecting the user's answer to what made her ask rode entirely on
  cosine retrieval (keyword-dependent, weak across languages, hard-zero once the
  source chat is past `rag_cap_age_h`) or on the facts fetch happening to nominate the
  source chat. The structural link — `initiated_ask`, stamped for the tight
  [ask]-loop close — was read by reflection only; nothing read it at chat time. The
  asymmetry was one-sided by then: the *decision* that raises a question had been
  conditioned on its origin since 2026-08-18 (the facts-tree fetch + `{origin_note}`
  source material), while the turn that reads the *answer* had nothing. Now
  `generation.handle_generate` injects ONE standing `ASK ORIGIN` block on the turns of
  an `initiated_by:"ava"` session — the question plus its origin material,
  **gist-or-nothing** (the `_query_nominated` rule): a chat-origin ask carries the
  source conversation's stored gist (hot, archive fallback, dated ISO off the stem), a
  self-directed ask the TIL recap of what she was reading (`til_gist.resolve_source`;
  `wiki:<site>` / bare `lookup` refs yield nothing, exactly as on the decision side).
  With no resolvable material the whole block is withheld — the question is already in
  the transcript as her own opener. One definition in the new `core/ask_origin.py`
  (pure, GPU-free, self-tested); template `ask_origin_prompt.txt` default-written,
  closing on the do-not-recite discipline. The block sits last of the standing parts
  (after the user portrait, before FACTS (fetched)) and rides the **History** checkbox
  — the payload is past-material recall by a non-cosine route, the same rule the
  fact-nomination slot follows. Her captured wander *reaction* is deliberately NOT
  carried (the decision side excerpts it harder for the anti-copy reason; standing on
  every reply turn, it is dropped outright). Resolution is two-tier: the
  `initiated_ask` stamp now carries the ask's own `source_session`
  (`chat_logger.start_session`, written by outreach + synthesis — durable, survives
  the ask's later eviction), with a live-fold lookup by ask key as the bridge for
  pre-stamp sessions (goes dark once the ask is evicted, and honestly so). Cached per
  session file in `generation._current_ask_origin` — the fallback folds the whole
  memory op-log, fine once, not per message.

- **The chat-facts protocol anchors participant labels to the recorded spelling.**
  Observed live: the pass labelled one person differently per generation — "artemyvo" /
  "Artemy" / "Артемий" for the same human, and Ava herself as "Ava" / "AI" / "user" —
  per whatever was salient in the weights at that token, fragmenting one person across
  `normalize_person` keys (the graph fold's alias table was the only remedy, and a
  manual one). Mechanism: `chat_facts_prompt.txt` asks for `(about: NAME)` without ever
  saying WHICH name, while the caller structurally knows the recorded spelling
  (session `user` + per-exchange `speaker`). Two changes, one definition each in
  `core.chat_facts`, used by both callers (`reflection_runner`'s production pass and
  `core.modules`' workbench twin — the modules self-test asserts both): **(1)
  prevention** — `participants_note(session)` states the canonical spelling in-context
  ("recorded as \"artemyvo\" … exactly that spelling, never a translation,
  transliteration, nickname, or 'the user'"), and restates the self rule's two
  uncovered leaks (her own name, "AI" — the prompt only explains "Me"/"the user").
  Composed into the *closing*, not the prompt file (which stays slot-free) and not
  `ava_initiated_note` (which only reaches reversed sessions). **(2) backstop** —
  `make_subject_fn(session)` rides `parse_facts`' existing `subject_fn` seam and folds
  any label naming a recorded participant onto that participant's canonical key,
  inflection-tolerantly via `exchange_anchor.words_match` (≥6-char shared prefix, ≤3
  differing chars, exact below 6 — "Artemy"→"artemyvo" folds; "Artemis" (prefix 5) and
  "Alexey"-vs-"Alex" (short ⇒ exact) structurally cannot, so a third party is safe).
  Deliberately NOT folded: cross-script variants (no string relation — the alias
  table's residue, made rare by the note) and `Ava`/`AI` as self parse-side (`AI` may
  legitimately name an encounter peer; her name stays out of the code). Multiple
  recorded spellings of one person collapse to the first (session-`user`) spelling in
  `session_participants`, so the fold targets cannot re-split the space they close.
  Already-written protocols are untouched — this is not a parser-read change, so
  `is_stale_record` correctly does not flag them; their variants stay the alias
  table's job.

## 2026-08-18

- **The wander learning pass de-anchors from the trained voice pass, and a learning pass
  that abandons its contract is named, not silent.** Observed live minutes after a fresh
  adapter went up: the wander *learning* pass (WEIGHTS/RAG/RESOLVED → memory) answered
  with a second persona essay — no sections, no tagged lines — and its CoT recited the
  VOICE pass's instructions from memory ("avoid 'As an AI'", "embrace the persona's
  voice"), text that appears nowhere in `wander_prompt.txt`. Mechanism: both wander
  passes shared `_wander_article_content` byte for byte, and every wander SFT capture is
  exactly that user turn (under the voice prompt) mapped to a free-form essay, retrained
  from scratch each cycle — so the adapter learned the mapping off the *user turn* and
  steamrolled the learning pass's system prompt. Two changes: **(1)** the learning pass
  builds its own content (`_wander_learning_content` — different lead, and the
  structured task restated AFTER the article, the position the protocol/summary closings
  already use); the voice pass keeps its builder verbatim, being bound by training
  parity. **(2)** `learning_contract_problem` checks every learning pass's ANSWER region
  (wander/news/lookup) for any section header or tagged line: an essay parses to zero
  modifiers, which is also what an uneventful page produces, so the collapse of the
  channel previously rendered as a series of quiet passes. The verdict rides
  `til_reflect_done` (the Sleep tab warns at the Apply decision), lands as a
  `pass_warning` on the ingestion phase, and is printed + carried on the autonomous
  wander's summary. Deliberately loose — ANY marker anywhere in the answer passes; its
  job is naming total abandonment, and a reflection that kept nothing but wrote its
  empty sections passes as it should.

- **An ask is stored as a question, not as the message that would raise it.** The TIL
  prompts asked for `[ask:user]` "as the opening you'd actually raise", which predates
  the machinery that made the stored opener redundant: outreach now composes the opener
  at raise time and carries the source recap beside it (`_source_material`,
  2026-08-13). What the framing actually bought was paragraph-asks — measured on the
  live store, 33 of 90 TIL-origin asks over 300 chars (max 648), 7 with no question in
  them — and an ask's shape has mechanical stakes: it embeds on its question text, and
  resolved it becomes the distilled fact's trigger. Two-sided fix: the three TIL
  prompts now ask for *the live question itself, one or two sentences — the opening is
  composed later*; and `parse_consolidation` compacts an over-long ask at the single
  parse path (`trigger_hygiene.compact_ask`, cap 600 — after the `(lookup:)`/`(seen:)`
  markers come off, before the key is minted), keeping the TRAILING whole sentences
  that fit, since the question is what closes an opener-shaped ask and a head
  truncation would keep the wind-up and cut the question. Existing stored asks are left
  as they are (their keys are live join targets); the store converges as they resolve
  or retire.

- **A date past the training horizon is not evidence of fiction.** The 2026-08-16
  `depicted` fix stopped a tropes page being filed as fact, but the framings it loosened
  ("it may be a work rather than an account of one") handed the pass a counter-inference
  nothing guarded: the model's knowledge has a horizon, this box's lookups are
  disproportionately about the recent world, and on the "2026 Iran war" article the
  pass concluded *"this text is clearly fictional (alternate history/prediction)"* —
  spending whole budgets deliberating it (feeding the 0-fact failures above), and, on
  the one run that completed, filing **48 of 48 facts `(class: depicted)`** — a facet
  outside `KNOWLEDGE_FACETS` *and* outside `report`, so the war's record was invisible
  to every fetch channel while looking recorded. `learning_prompt.txt` has carried the
  "these events are real, later than your training" reassurance since the news lane
  shipped; the protocol lane never inherited it. The wander + lookups framings
  (`til_facts._KIND_FRAMING`) and the shared `til_facts_prompt.txt` `depicted` paragraph
  now state the rule in both directions: depicted is for what a work invents, never for
  what the world did after training, and a late date decides nothing. The all-depicted
  protocol itself must be deleted by hand to re-derive (`is_stale_record` probes what
  the parser can re-read, not a generation-time misjudgement): remove
  `server/data/til/snippets/lookups/20260729_133817_2026_Iran_war.facts.json` on the
  GPU box and the next run's drain rewrites it under the current framing.

- **A TIL recap is the prose, not the reasoning that produced it.** The reflect seam
  returns a generation WITH its canonical `<think>` block; `til_gist.clean_gist` ran
  only `sanitize_gist`, which knows structure boundaries but not reasoning markers — so
  **16 of the 17 recaps on this box opened with the model's task-notes** ("* Topic: …
  * Goal: … * Constraints: …"), and since both consumers excerpt from the HEAD
  (`rag_engine._render_til_nomination` at 700 chars into a live chat turn,
  `outreach._source_material` at 1400 into a reach-out composition), what reached
  prompts was entirely inside the think. `clean_gist` now strips
  `reasoning_text.answer_after_think` first, and `gist_text` applies the same strip **on
  read** (the `ChatSidecar.summary_text` pattern): every stored think-carrying recap is
  repaired for its readers with no migration, verified against the live corpus (the Devs
  recap reads back as its 626 chars of first-person prose). A recap that was ALL
  reasoning — a generation cut before its answer — now reads as absent, so `has_gist`
  returns it to the backlog to be regenerated clean instead of serving task-notes.

- **Both loop guards are off for the protocol passes, and a failing backlog text stops
  wedging the drain.** `stop_on_repeat=False` (2026-08-07) disarmed only the verbatim
  guard; the diversity guard (`_DegenStop`) was believed to separate cleanly ("still
  catches a real collapse") and was caught live doing the opposite: a run of
  `[fact] (about: X) (class: Y) …` template lines craters its rolling distinct-token
  ratio exactly as a genuine collapse does, so one lookups protocol halted at its 7th
  near-identical line and another mid-think while *drafting* such lines — both reported
  0 facts. No threshold separates the protocol shape from degeneration (the diversity
  collapse IS the correct output), so the choice is binary and the cost asymmetry
  decides it: a genuine runaway wastes at most the token cap, a false halt loses the
  protocol. The reflect seam gained a per-call `degen_stop` override (None ⇒ box
  default); the two production facts passes and their Modules-tab twins pass `False`
  (`ModuleSpec.degen_stop`, parity-asserted by the self-tests). And because
  `write_facts`/`write_gist` on an empty result write nothing — correct, the file is
  what takes a text out of the queue — a text whose pass reliably yields nothing sat at
  the oldest-first drain head refailing every run (observed 2026-08-15→18: two lookups
  texts consumed 2 of 3 protocol slots for three runs while the backlog grew
  32→38→48). Both drains now retire a nothing-written text for the process
  (`til_wander._DRAIN_UNPRODUCTIVE`, in-memory like
  `background_reflection._backfill_unproductive` — the cause is usually the prompt or
  the model, and a restart is how those change), and treat `no_text` as the per-text
  condition it is rather than a drain-stopping box skip.

- **The facts protocol reads the answer, not the deliberation.** `chat_facts.parse_facts`
  scanned every line of the raw completion for `[fact]` tags, and the pass *thinks* —
  its `<think>` routinely drafts `[fact]`-prefixed lines before settling, and a draft is
  tag-identical to a final line. So the parser stored deliberation beside conclusion:
  reworded draft/final near-duplicates inside one protocol (5 of 113 files), draft lines
  consuming the `MAX_FACTS` cap ahead of the real answer, and — when the close marker
  landed mid-line — records whose text carries a literal `</think>` fusion (2 of 29 TIL
  + 4 of 84 chat protocols; the graph tree had folded two in, which is how a draft fact
  fused with reasoning markup reached a **live chat turn** through the facts block).
  `parse_facts` now parses only `reasoning_text.answer_after_think(raw)` — one parser,
  both lanes, so the runner's chat pass, `til_wander.run_facts_pass` and both Modules-tab
  twins are fixed in one place. The case no text inspection can catch — a generation cut
  inside an unclosed reasoning channel, which gemma-4 normalizes to *untagged* prose —
  is refused at both call sites via `reasoning_text.truncated_before_answer` (the
  ingestion drain now reports *"cut inside its reasoning"* beside its fact count, since
  that and *"the text established nothing"* must not share the number 0). Repair is
  self-healing: `chat_facts.is_stale_record` gained a second probe — a stored record
  carrying a reasoning marker was written by the raw-completion reader and flags its
  document — so the six corrupted protocols re-derive through the existing backlogs, and
  the tree repairs itself on the next `graph_rebuild` fold. What stays undetectable, in
  the bounded-safe direction: a marker-less draft that leaked (textually
  indistinguishable from a real fact); most travel in documents a marker sibling flags.
  (Commit: see git log, 2026-08-18.)

- **The reach-out passes look at the record before they raise a question.** Outreach's
  decision pass is the box's one formal *"have I already learned the answer?"* judgement —
  its `resolved` branch evicts a live ask for good — and it ran on cosine retrieval alone,
  which is exactly the retrieval the facts-tree fetch was built to compensate: the record
  is mostly English against often-Russian asks, and a fact that landed *after* the ask was
  queued (the whole substance of `resolved`) embeds on a trigger the ask's wording may
  never touch. Synthesis was blinder still — both its passes run `disable_rag=True`, so
  its questions formed against the transcript and the persona digest and nothing on
  record, and it composes and sends its own opener directly, so a check in outreach alone
  would not have covered it.

  Both now run **stage 1 exactly as a live chat turn does** (FACTS_TREE.md §10), through
  two new `fact_fetch` lanes with their own closings — the existing two are separate
  precisely because each lane asks a different question of the same candidate list, and
  these ask new ones. The **ask lane** (`fetch_blob_for_ask`) asks *"is the answer already
  on record?"*; chat's closing would frame the ask as an arriving message from a speaker
  who does not exist. The **re-read lane** (`fetch_blob_for_reread`) asks *"what does this
  old conversation touch that is already answered?"*, so what-do-I-now-wonder questions
  form against the record instead of being vetted afterwards; its transcript trims
  oldest-first (what a conversation left open lives in its later turns), one fetch per
  chat, not per chunk.

  The delivery is the seam, not the prompt files: `generation._reflect_system_parts` grew
  a `facts_block` slot rendered under chat's own label (`"FACTS (fetched)"`, ahead of
  INJECTED RAG, contract still last), so the `outreach_prompt` debug view shows it for
  free and an operator's customized `outreach_prompt.txt` on disk cannot silently drop a
  `{slot}` it predates. The seam also grew `rag_nominate_sessions`: outreach hands its
  picks' sources to the engine's fact-nomination slot, so the conversations and TIL recaps
  behind the picked facts are recalled into the decision pass's past-chat block — the
  recall half of the chat channel, for the price of passing them through. Synthesis
  deliberately nominates nothing (its pass reads ONE conversation, RAG-off) and threads
  the block through `_prepare` as well as the generation, so the chunk packer measures the
  prompt with the block and the transcript budget shrinks instead of overflowing.

  Both lanes are knowledge-facets-only — no `report` claims — because a `resolved` verdict
  retires a live question on the strength of what is injected, and *"a text asserted X"*
  must not resolve *"is X true?"*. One switch covers both passes (`graph.reachout_facts`,
  default on, riding `graph.enabled`): they are the two passes that raise questions, and
  the reason to condition them is the same. Per-lane wrapper templates default-write on
  miss (`facts_decision_block_prompt.txt`, `facts_reread_block_prompt.txt`); the broken-vs-
  quiet failure classification moved to `fact_fetch.FETCH_FAILURES` so three logging
  callers share one definition; and the IDEAL replay path refuses a hidden `facts_block`
  exactly as it refuses hidden RAG, so training parity holds. Every failure path costs the
  block, never the decision. Untested on a live GPU.

## 2026-08-15

- **Wander Apply persists what was reviewed, and generates nothing.** The manual wander flow
  is a dry run — passes are generated, shown, and kept only if Apply is pressed — and Apply
  had quietly stopped honouring that. It ran **two further generations** the operator had
  never seen: the fact protocol (`<stem>.facts.json`) and the prose recap
  (`<stem>.summary.json`). The recap is the one that matters, because it is not an inert
  record: `rag_engine._render_til_nomination` injects it into live chat turns and
  `outreach._source_material` puts it into reach-out messages. So a press meaning *"keep
  what I just read"* was consent for prose that reaches the user unreviewed.

  It was also a **guaranteed failure**. `backend_client.til_apply` waited 120 s; the
  protocol pass carries a 12,288-token budget per block and its own comment records 21
  minutes on a 26k-char page. Since `run_in_executor` is not cancelled when the client stops
  waiting, the server finished anyway — so every manual Apply reported failure while
  everything landed, and `_last_wander_exchange` was only cleared ~21 minutes later. A retry
  inside that window re-entered `_write_wander_exchange`, and `wander_sft.append_example`
  appends unconditionally with no dedup: **a second wander capture, trained twice**, plus
  another 21-minute pass queued on the single GPU executor.

  The recap now runs as a **third preview pass** at wander time, streamed into the same log
  beside the reaction and the reflection, and Apply *writes* it. `run_gist_pass` split into
  `generate_gist_text` (GPU, returns) + `write_gist_text` (no GPU, persists), and recomposed
  from the two — so the backlog drain and the news ingestion phase, which have no operator
  and review nothing, still generate at the point of writing.

  The **protocol** is deferred to the backlog instead, and that asymmetry is deliberate
  rather than convenient: it is a witness record of the text (*"record what the text says,
  not what you make of it"*), closer to a photocopy than to a judgement, nobody reviews the
  news lane's, and enumerating dozens of lines per article is not a thing an operator would
  do. The snippet is on disk at Apply, so `til_facts.list_backlog` picks it up and the next
  reflection run derives it — the same function, in the same place the news lane has always
  called it from.

  Consent semantics are unchanged: the snippet is still written only on Apply, so a declined
  wander still leaves nothing on disk and therefore no backlog entry. The autonomous wander
  carries no reviewed recap by construction (there is no operator), so both artifacts fall
  to the backlog there and the idle job hands the GPU back in seconds instead of holding it
  for two long passes. `til_apply`'s timeout went to 600 s as headroom for the memory write
  and RAG refresh, which are now the only work it does.

- **Fiction stops entering the tree as fact.** The `til_facts` wander framing asserted *"This
  is an encyclopedia article about {title} … most of what it states is standing description
  of its subject"*. That is false for three of the five enabled wander sources — WikiTropes
  is entirely about invented material, Lurkmore and Neolurk largely so — and for the
  manual-visit path, which fetches whatever URL an operator gives it, a story text included.
  On such a page it is an instruction to file an invented world's contents as fact: *Jimmy
  lives on the Moon* `(class: standing)` → `(til, standing) → property` → a **knowledge
  facet**, which `claim_candidates` offers by default. So it would have reached **live
  chat**, rendered as something simply known, not merely the reading lane.

  Caught before it could happen: there are still zero TIL protocols, so nothing has folded.
  It would have arrived on the first drain, and protocols are immutable — a rebuild does not
  reclassify — so the vocabulary had to be right before the backlog ran, not after.

  The witness contract is the reason the existing classes had no answer. *"Record what the
  text says, not whether you believe it"* is right for a news digest (a false report is
  still a report) and collapses on fiction, whose propositions are not claims about the
  world at all. `stated` is the nearest fit and still wrong — it means the text asserts this
  — so the honest fix is a fourth class rather than a reused one: **`depicted`**, with its
  own facet `depiction`, mapped only on the TIL lane.

  It is **carried, not withheld**, and that is the point rather than a concession: a tropes
  article meeting an earlier conversation is exactly the association the reading lane was
  built for, and it needs the fiction — marked as fiction. `blob._SOURCE_VERB` carries the
  distinction per line, because the verb IS the epistemics: *"reported"* says a text
  asserted this about the world and may have been wrong, *"depicts"* says the question of
  truth does not arise. One verb for both is how a fantasy world's contents come back
  phrased as a stale news item.

  Safe by construction at every layer: `depiction` is outside `KNOWLEDGE_FACETS`, so chat
  cannot be offered it; `(chat, depicted)` is deliberately unmapped, so if the vocabulary
  ever drifts the pair lands in `unclassified` — also outside knowledge — and the failure
  mode is material withheld, never material promoted. `normalize_class` takes the near-misses
  a generation reliably produces (`fiction`, `in_universe`, `plot`, `narrative`).

  The framing was also rewritten to **name the source without classifying the text**, and
  the self-test now asserts that property rather than the old phrase, so a future rewording
  cannot quietly restore the assertion. The `lookups` framing got the same hedge — a fetched
  article may be a work rather than an account of one.

- **A reading pass may carry what a text reported, attributed; a chat turn still may not.**
  The reading lane could only be offered `property` + `event`, so the association it was
  built for could not fire: wander onto an article about Khaled Mashal and the news digest
  that covered the same events is invisible, because `(til, stated) → report` and `report`
  is not a knowledge facet. The lane carrying the most current material was the one
  structurally barred from feeding associations, while the humour wikis could.

  The fix is not to relax the facet rule but to apply the distinction the rule was always
  making. `report` is withheld from a **chat turn** because a pass told *"things you know"*
  would state a news assertion as fact **to a person** — the failure the facet level exists
  to prevent. A **recap asserts nothing to anyone**; it is what stayed with someone from
  reading, and *"a digest of world events from 2026-07-28 reported that a Hamas delegation
  visited Cairo"* is the honest form of exactly the association wanted. So for this lane the
  epistemics move from *withhold* to *attribute* — which is what `attribution_line`'s
  docstring has said all along would be required of any channel that carried these.

  `claim_candidates` gained a `facets` parameter (default unchanged, so chat is byte-identical)
  and `blob.READING_FACETS` is the one widened set. `report_line` renders a report with the
  **text that asserted it named in the line**, which `attribution_line` could not do: that one
  says *"Iran is reported: …"*, passive, naming the claim's subject with the source nowhere
  in it. The phrase comes from an injected `describe_source` (`til_wander._describe_source`,
  the pattern `render_blob` already uses for `words_match`, since resolving a ref means
  reading the snippets tree and `graph/` never imports the inference role) and names the
  KIND as well as the date, because on this box that is the difference between evidence and
  a joke: *"an article about Ельцин on Lurkmore"* against *"a digest of world events"*.

  Four properties are enforced in code rather than asked for. Reports render in their **own
  section**, never folded under a node — the node grouping presents claims as things known
  about that node, the one reading a report must not get. They carry their **own budget**
  (`gist_max_reports`, default 2) rather than sharing `gist_max_claims`, because a TIL
  protocol yields an order of magnitude more claims than a chat one and a shared budget
  would let news crowd out the scarce conversational half — the claims about the people in
  the conversation, which were the whole point of the channel. A report picked where none
  was budgeted is **dropped and counted**, since a silent drop reads as the pass not having
  picked it. And an unresolvable source falls back to the raw ref, a source-less claim to
  *"something on record reported"* — an unattributed report is the one thing this must never
  render as known.

  **`position` is offered to neither lane**, and that is not an oversight: a report has a
  text behind it that can be named, while a position is one person's view, and rendering it
  into a reading pass would put the interlocutor's politics into a recap of the world.
  **`person:_self` is not widened either** — §10 is about the subtree whatever the facet, and
  a caller widening the facets does not get to widen that.

  Inert on the current corpus (the tree is chat-only, so zero reports are offered) and it
  comes alive as TIL protocols land. Knob `graph.gist_max_reports`; `0` restores the
  knowledge-only list.

- **The facts tree is folded by the box, not by an operator remembering to.** Building it
  was a manual `python -m graph.build`, and **nothing on the box ever called it** — a gap
  invisible at every call site, because `data/graph/tree.json` is the only thing standing
  between the accumulating `.facts.json` protocols and the two channels that read them
  (live chat's fetch, and now the TIL reading fetch).

  The two failure states are the ones nobody would report. A **clean install** has no tree,
  so `read_tree` returns `None` and every chat turn and every recap skips with `no_tree`
  *forever*, while `graph.enabled` reports `true` — a channel off in substance and on in
  configuration, writing protocols correctly and never reading one. A **running box**
  freezes its tree at whatever the last hand-run build saw; conversations and readings land,
  protocols accumulate beside them, none of it becomes retrievable. Neither announces
  itself: an empty blob from a missing tree is indistinguishable, at the call site, from an
  empty blob because nothing was relevant.

  `core/graph_rebuild.py` is an `idle_scheduler.IdleJob` beside `worklog_sweep`, the box's
  other GPU-free upkeep job, with the same short `idle_seconds` for the same reason — a full
  idle hour starves it on precisely the busy box where the tree goes stale fastest. It
  rebuilds wholesale, which is safe by construction rather than by care: the tree is
  *derived and disposable — never a store*, `read_tree` already collapses missing, corrupt
  and unrecognised to `None` because all three mean *rebuild*, and `fold` is deterministic,
  so two builds of an unchanged corpus are byte-identical. `rebuild()` delegates to
  `graph.build.build`, so the job and the CLI cannot drift.

  Staleness is three signals, cheapest first: no tree; **count drift** against the
  `stats.read.files` the tree recorded, which is what catches a *deleted* protocol (it moves
  no mtime forward) and a restored copy whose mtimes predate the build; and any protocol
  newer than the tree **file**. The file's mtime rather than the document's `built_at` on
  purpose — that field is a naive local ISO string, and comparing it to a filesystem mtime
  means reconstructing which clock and offset wrote it, where the file's own mtime is the
  same quantity in the same units as the thing it is compared against. A Migrate or snapshot
  restore can reset mtimes and buy one spurious rebuild; that costs milliseconds and
  self-corrects.

  The journal line leads with **`showable`** — the `property`+`event` count, what a fetch
  pass may actually be offered once `claim_candidates` filters to the knowledge facets and
  drops `person:_self` — because the raw claim total says almost nothing about whether the
  channel has anything to work with. Measured the day this landed: **197 claims, 5
  candidates.** A skip does not log hourly, and a failed fold reports rather than taking the
  scheduler down, since every downstream skip would otherwise read as "nothing found".

- **The TIL recap is read against what is already on record, not as a text about strangers.**
  Ava processed world events as a distanced observer — a news digest naming the 2026 Iran
  war produced a recap in generic terms about people's suffering, while the corpus held
  *"artemyvo: experienced the 2026 Iran war as physical confinement in a bomb shelter"* the
  whole time. The recap pass could not know it: `run_gist_pass` ran `disable_rag=True` on the
  principle that *a recap is of the text, nothing else*.

  That principle is kept where it belongs and dropped where it does not. The **protocol**
  pass (`til_facts`) stays uninjected: its contract is exhaustive witness — record what the
  text says even where it is wrong — and conditioning what gets enumerated on what she
  already knows would make the record non-reproducible and bias the tree that then
  conditions the next reading. The **recap** is avowedly selective (*"what stayed with
  you"*), and until now that selectivity had no referent for *who is doing the staying*.
  The rule: **condition the passes whose output is avowedly selective; never the pass whose
  contract is exhaustive** — the uninjected protocol is precisely what makes conditioning
  the recap recoverable, since the full record stays on disk.

  Mechanically it is the chat channel pointed at different material. `fact_fetch` gained a
  second entry point, `fetch_blob_for_text`, sharing one `_run` with `fetch_blob` so the
  part with safety consequences cannot drift between lanes; only the body differs, and it
  has to — handing an article to `build_body` puts it under *"The message that has just
  arrived"*, and the pass then fetches whatever it knows about the person it took to be
  speaking. `til_wander._fetch_reading_facts` runs it through the same `ModuleSpec`, prompt
  and greedy sampling the live turn uses, and appends the rendered block to the pass's
  system prompt — where chat puts it, standing knowledge ahead of the material of the
  moment.

  **What the corpus measured is what shaped the design.** `claim_candidates` on the live
  tree offers **five claims, every one about the single person she talks to**: 21 knowledge
  claims exist corpus-wide, 16 of them under `person:_self` and dropped by the §10 rule.
  So the risk is not a bad pick — it is a recap of world events drifting into a recap of
  him, on a menu with nothing else in it. Three responses: `gist_max_claims` defaults to 3
  against chat's 6 (a recap is a few hundred characters, so the same claim count is a far
  larger share of it); `facts_reading_block_prompt.txt` closes by naming the drift; and
  `TEXT_FETCH_CLOSING` tells the pass that picking nothing is *the ordinary answer*, which
  for every non-war text it is. The assembled stage-1 prompt is ~1.7k tokens on a real
  digest — the candidate list is the entire cost and it is five lines, so the ~10k prefill
  that makes this expensive on the chat path does not arise here.

  **This does not fix the wander half of the original idea.** An article on dwarfs in
  fantasy returns `NONE`, and correctly: there is no knowledge-facet claim about any of his
  interests on record. His 42 `position` claims contain them, and positions are withheld by
  design. The general fix is upstream and is not this change — 174 of 197 claims in the
  tree are positions, every one belonging to one of the two speakers, and the TIL lane
  contributes **zero** occurrences (64 source texts on disk, no protocols), so the
  `(til, *)` branches of `_FACET_MAP` have never fired. Draining that backlog is what gives
  this channel something about the world to offer.

  Knobs `graph.gist_facts` (on) / `graph.gist_max_claims` (3), riding `graph.enabled`.
  Every path fails soft to the unconditioned recap, and `run_gist_pass` now reports
  `facts_picked`/`facts_chars`/`facts_sources`/`facts_skipped` beside it — a recap written
  under two facts and one written under none read identically, so an unreported channel is
  one whose silence nobody notices. Untried on a live GPU.

- **A peer's chain of thought is enforced as display-only at the boundary, not assumed.**
  In gossip/encounter the peer's CoT is meant for the operator watching the log and must
  never reach Ava: she cannot tell it apart from something the peer *said*, and everything
  downstream of a counterpart reply treats that text as the peer's spoken turn — it becomes
  the `user_prompt` of a logged exchange, the RAG query for her next turn, a chat-RAG
  passage, and eventually a trained row.

  The **intended** path already separated them (the serving side returns the answer on
  `message.content` and the reasoning on `message.reasoning_content`; the driver kept the
  latter in a display-only field). What was missing is that `content` itself was trusted
  verbatim, and it is not trustworthy on two live paths: **(1)** a peer whose generation is
  cut mid-thought ships its raw, unterminated thought AS the content, because its own
  `ChatLogger._parse_cot` reads a `<think>` with no closing tag as "no CoT, all answer" —
  not a corner case, since a reasoning peer thinks for minutes against gossip's 4096-token
  default `counterpart_max_tokens`; **(2)** a counterpart that is not an Ava at all — the
  Encounter tab points at arbitrary endpoints, and a plain vLLM box serving a reasoning
  model with no reasoning parser inlines `<think>…</think>` as a matter of course.

  `core.encounter.strip_peer_reasoning` is the fix and the single definition, used by both
  halves of the mirror: the **driver** sanitizes what a counterpart replied
  (`CounterpartClient.reply`, folding anything it recovers into the display-only
  `last_reasoning` so the operator loses nothing), and the **serving** side sanitizes the
  peer's incoming `user` turns (`generation._strip_peer_reasoning_from_messages`, applied
  once up front so the conversation, the RAG query and the logged transcript read the same
  clean turn). Deliberately **family-agnostic** — the peer's family is unknown (its model
  name is operator-typed free text), so every marker shape `model_family` knows is
  recognized (`<think>`, gemma's `<|channel>…<channel|>`, gpt-oss harmony, and the
  prefilled-opener case where only a close marker comes back), and an unclosed opener is
  read as reasoning to its end. It errs toward withholding: a reasoning-shaped span kept
  from Ava costs a legible failed turn, while the same span let through is a peer's
  thinking silently entering her memory.

  A reply with **no speech left under the reasoning** is refused on both sides rather than
  passed through, and rather than answered as an emptied turn (which would invent a
  stimulus the peer never sent and then log it as one). Serving-side sanitation is
  idempotent, so `_GossipSessionLog`'s prefix-continuation match still holds across calls;
  it is gossip-only, since a public-API caller's message text may legitimately contain such
  markup as data and nothing there is logged.

  **Interpretation of existing artifacts:** gossip/encounter transcripts written before this
  date may carry a peer's raw reasoning as an exchange's `user_prompt` — check any whose
  peer turn is unusually long or reads as deliberation rather than address; those exchanges
  reflected and trained on it. Files: `server/inference/core/encounter.py` (+ GPU-free
  self-test `python -m core.encounter`), `server/inference/core/generation.py`.

## 2026-08-14

- **Training review's search phrase gets a write counterpart: Replace.** A replacement box
  and a **Replace** button beside the existing search bar
  (`training_review_widget._on_replace`) rewrite every occurrence of the search phrase in the
  **selected** entry's CoT + answer boxes. Matching is case-insensitive, mirroring the filter
  and the yellow highlight so what the operator sees marked is exactly what is replaced, and
  the replacement is substituted through a callable (`pattern.subn(lambda _m: replacement, …)`)
  rather than as a template string — so `\1` or `\g<0>` typed into the box is inserted as
  text, not interpreted as a backreference. An **empty replacement box deletes** the phrase,
  which is the common case (a stray marker, a leaked template line) and so needs no separate
  control.

  **A prefill, never a write** — the same discipline as *Strip persona opener*: the
  substitution only fills the boxes, marking the entry dirty through their own `textChanged`,
  and the edit bar's Apply remains the single thing that reaches the sidecar and freezes ❄ the
  exchange. That is also why it is scoped to one entry rather than sweeping every match in the
  corpus: each entry's repair is one sidecar write that locks an exchange against
  re-reflection, so a corpus-wide sweep would be dozens of them with nothing reviewed. The
  immutable query box is excluded — replacing there would show a change that could never be
  applied.

  Client-only: no server code, no protocol message, no new worker. It refuses with a status
  line on an empty search phrase, no selection, a read-only (wander / pre-provenance) row, or
  zero occurrences, rather than silently doing nothing. Files:
  `client/ui/training_review_widget.py`.

- **…and a bulk form of it, because the corpus was poisoned in one place at a time by a
  machine.** **Replace all…** (`_on_replace_all` / `_bulk_replace_jobs` /
  `BulkReplaceWorker`) runs the same substitution over **every entry currently shown** — the
  filters above the list choose the scope — and, unlike Replace, **writes each result
  itself**, freezing ❄ the exchange as it goes.

  **Why this one writes when the single Replace deliberately does not.** The per-entry Apply
  exists because a hand repair is one judgement about one exchange, and the review before it
  is real. A phrase smeared across the corpus by a bad build (observed: `"terms of "`
  everywhere) is the opposite case — the same three words deleted in hundreds of places,
  where "review each" is a formality nobody performs and the ceremony would guarantee the job
  never gets done. So the write is the point of the button, and the confirm dialog carries the
  weight instead: entry and occurrence counts, how many targets are already frozen and will be
  rewritten, what is being skipped, and that undoing means re-reflecting whole chats (which
  discards every reviewed target in them, not only these).

  **It loops the single-Apply RPC rather than gaining a bulk server call.** Each write is
  `apply_regenerated_exchange` — one reviewed target that locks ❄ its exchange, so
  re-reflection and Revisit cannot re-derive the poison off the untouched transcript. N round
  trips is the cost; the reason to pay it is that a bulk repair and a hand repair then cannot
  come to mean different things on disk. It reports progress and **stops between writes**
  (each is independent, so a stop leaves what it reached repaired and frozen and the rest
  exactly as they were), and Refresh / Rewrite history / the single repair buttons stand down
  for the duration — progress patches entries by their `_order` stamp and a reload restamps
  them, so `refresh()` returns early mid-sweep too (the tab-open auto-refresh is not a button
  press). Read-only rows and rows whose reply would come out empty are skipped and **counted**
  in the report, never silently dropped; the list re-filters once at the end rather than per
  write, so it cannot reshuffle under the operator mid-sweep. Transcripts are untouched — that
  is still *Rewrite history*.

  **The search phrase stopped being stripped** in the same change (`_search_phrase`, now the
  one definition behind the filter, the highlight and both Replace buttons). The poison
  carries a **trailing space**, and a stripped needle deletes the words while leaving the
  double space behind: a corpus-wide off-by-one bought in exchange for a whitespace-tolerant
  search box. What is typed is now what is filtered on, what is painted yellow, and what is
  replaced. Files: `client/ui/training_review_widget.py`.

## 2026-08-13

- **The TIL lane gets a recap, and two things that had nothing to point at now point at it.**
  `core/til_gist.py` writes `<snippet_stem>.summary.json` beside the `<stem>.facts.json` the
  protocol pass already writes — what one article or digest *was*, in her own voice.

  **Why the lane had none.** A chat produces a gist as a matter of course: something has to
  represent a conversation once its verbatim passages expire past `rag_cap_age_h`. A TIL
  text is read once, at fetch, by a *curation* pass that keeps a handful of `[fact]` items
  and discards the rest, and nothing afterwards ever needs a compact form of it — so every
  consumer wanting "remind me what that was" had the choice of injecting 26,000 characters
  or nothing, and chose nothing.

  **The two consumers are the same shape one level apart.** A fetched FACT recalls the
  source it came out of (`rag_engine._render_til_nomination` — the TIL half of the
  nomination slot, which returned nothing for want of this file: the gap named when that
  channel shipped the day before). A raised ASK recalls the source it came out of
  (`outreach._source_material`) — a question distilled from a day's news is a question with
  its material discarded, and it was observed live: she raised one about the Gaza "Yellow
  Line" carrying nothing of the digest that prompted it, while
  `data/til/snippets/news/2026-07-30.json` sat on disk holding exactly what she had read.
  The ask-side injection folds into the existing `{origin_note}` slot rather than taking a
  placeholder of its own — the way check-in folds standing openers into `{recent}`, so a
  customized `outreach_prompt.txt` gets it unedited — and that slot is the right one, being
  non-empty precisely for the asks that HAVE a source text, beside the instruction on how
  to carry it in.

  **Register is a chat gist's, deliberately.** Both consumers inject it as recall, so an
  encyclopedia abstract would read as a quotation from the source rather than as something
  she remembers. The prompt names the reader — her, later, when something she took from a
  text has come back with no memory of where it came from — and asks for one thing a chat
  gist never needs: say where the source's *tone* must not be mistaken for its truth, the
  approved wiki list being chosen for register (Lurkmore, Neolurk and WikiTropes are humour
  wikis) and that being the first thing a bare summary loses.

  **A separate generation from the protocol, not a second field of it.** A protocol is
  exhaustive and a recap selective; asking one pass for both gets a recap of the protocol.
  They also read the text differently — the protocol in parts (facts extract
  independently), the recap whole, since a recap of the first third of an article is not a
  third of a recap. An oversized text keeps its opening and is stamped `truncated`; that
  branch is unreachable on the live corpus (largest text 26,001 chars against a 26,542
  budget) and map-reduce is the named upgrade if it ever fires.

  **Two failure modes handled by precedent rather than freshly.** An unusable recap writes
  nothing, so the snippet stays in the backlog and can be retried — a file saying "this
  text was nothing" is indistinguishable from a pass that came back empty, and the file is
  what takes the snippet out of the queue (`til_facts.write_facts`' rule). Cleaning is
  `chat_sidecar.sanitize_gist`, because the leak it guards against is a property of the
  generator rather than of the lane: this pass runs on the same seam, right after passes
  whose output IS `## WEIGHTS`/`[fact]` blocks.

  **One bug caught by its own self-test, and it is the `SIDECAR_SUFFIXES` lesson again.**
  `til_facts.iter_snippets` excluded only `.facts.json`, so the new `.summary.json` was
  enumerated as a *source text* — the protocol pass would have recorded facts about Ava's
  own recap and the graph build would have ingested them as things the world stated. The
  fix is `til_facts.DERIVED_SUFFIXES`, this lane's version of the chat one, with `til_gist`
  asserting from its side that its suffix is listed.

  Backfilled by `til_wander.drain_gist_backlog` during a run's ingestion phase, capped at
  `_TIL_GISTS_PER_RUN` (2) — smaller than the protocol's 3 because this artifact postdates
  the corpus, so its backlog is *every text on the box* and wants to drain steadily rather
  than take a bite out of the run it fronts.

- **The lookup fetch-once marker retired only the first ask about a subject, so any other
  ask on that topic re-fetched its article forever.** Observed as the Gaza "Yellow Line"
  page being fetched three times (07-31 twice, 08-09). The marker itself was working: the
  op-log shows one `lookup` op for *"What is the Yellow Line…"* and none at all for *"Is the
  Yellow Line a physical barrier or a coordinate-based line?"*, which is still eligible.

  `ingest_lookup` built a `{subject: question}` map with `if subject not in subj_to_q` —
  first question wins, and every other ask about the same subject was silently dropped from
  the map. Dropped questions never reached `attempted`, so no `write_lookup` was appended,
  so they stayed `lookupable`; the next ingestion re-extracted the same subject, re-fetched
  the same article and dropped the marker again. The map now holds a LIST per subject, the
  fetch is still one article per subject, and every ask that led to it is marked — one fetch
  of one page answers them all or none of them, and re-fetching the identical page cannot
  change that. **Four more asks were sitting in this state** and stop looping now.

  **A second misattribution in the same block, opposite in effect.** Pairing an extracted
  subject back to its question preferred a substring scan (the first question *containing*
  the subject string) and kept the positional index only as a fallback. That is backwards:
  the extractor is handed the questions in order and answers in order, so when the counts
  align the position IS the answer, while the scan lands on a different question whenever
  two of them mention the same thing — retiring the wrong ask and leaving the right one to
  loop. Position now wins; the scan survives for the unaligned case, where there is no
  position to trust.

  **The digest half of the same assumption.** `_assemble_lookup_digest` showed the pass one
  question per article, so where several asked about one subject only the first could be
  echoed into a `[resolved]` — the others were unresolvable by construction, the retrieval
  side of the same loop. It now lists every question the article was fetched for.

  The pairing is extracted as the pure `til_wander.pair_subjects` with `til_wander`'s first
  self-test (`python -m core.til_wander`), because the bug had no symptom other than an
  article being fetched again, and nothing in the pipeline treats that as an error.

- **`SOURCE_KINDS` said `lookup` for a directory named `lookups`, so 18 fetched articles
  were invisible to every backlog.** Found by the self-test above — `_lookup_source_id`
  returned `lookup/x.txt` for a file living in `lookups/`. `til/fetch_article.py` has always
  written into `snippets/lookups/`, while `til_facts.SOURCE_KINDS` has said `lookup` since
  the module shipped, and `iter_snippets` resolves `<snippets>/<kind>/` — so the directory
  was simply never enumerated. Consequences, all silent: no lookup article ever had a fact
  protocol (0 on disk) or a recap, neither backlog could ever pick one up, the Modules tab's
  TIL input list omitted them, and a graph-lane ref for one (`lookups/<stem>`, built from
  `path.parent.name`) could not resolve back through `til_gist.resolve_source`.

  The constant now names the real directory, and the per-kind framing keys in `til_facts`
  and `til_gist` move with it. The backlogs go from 20 texts to 32. `_lookup_source_id`
  takes the kind from the file's own parent dir rather than a literal, since a literal is
  how this happened. **These are directory names** is now stated at the constant.

- **Her reaction to a wandered article is paired with its recap — the wander corpus's
  second consumer, after two years of being training-only.** `wander.jsonl` holds a
  voice-pass capture per applied wander (~2,300 chars of answer after the `<think>`), and
  it had exactly one live reader: `build_dataset`, which trains one row per capture at
  `WANDER_LR_MULT`. Verified against the 2026-08-09 build snapshot — 8 wander rows of 108.
  The chat-RAG wander channel that also read it is off on every path pending redesign
  (`generation._INJECT_WANDER`), so nothing *recalled* it.

  `wander_sft.reaction_for(url)` is the second reader, and `outreach._source_material` now
  emits two pieces for a wander-sourced ask: the recap (what the text was) and the reaction
  (what she made of it). On a wandered article the second is usually the whole reason a
  question came out of it at all — a curation pass distils `[fact]`/`[ask]` items and drops
  the thinking that produced them, so the ask survived and its motive did not.

  **Matched on the article's URL, not its timestamp**, because the two clocks differ: a
  record's `ts` is when the pass ran, its snippet's stem is stamped when the snippet is
  written at Apply, and the observed gap is minutes (`01:13:29` against `011004`). The URL
  is the article's own identity and both sides carry it verbatim — 10 of 10 wander snippets
  on this box join.

  **The ANSWER span only, never the captured `<think>`** (2.6k of the 5k characters). Her
  CoT is not injected material anywhere on this box.

  **Labelled *at the time*, and excerpted harder than the recap it accompanies** (800 chars
  against 1400) — which inverts the usual rule, deliberately. The label is because the
  reading may be weeks old and she has changed since (the premise the revisit machinery
  rests on), so an ask she is only now deciding to raise must not arrive pre-answered by
  her own earlier opinion. The tighter budget is because this is her own prose, in her own
  register, placed immediately before she composes a message: `_NoCopyPrevReply` guards the
  previous assistant *reply* and nothing else, so nothing would stop an opener being
  assembled out of it.

  **Deliberately NOT added to the live-chat nomination slot** (`rag_engine._render_til_nomination`,
  which keeps injecting the recap alone). Article + her reaction, injected into a chat turn,
  is exactly what the disabled wander channel used to do — same material, same position in
  the prompt — so adding it there would resurrect a switched-off channel through a side
  door. Outreach differs on its merits: that pass is not answering anyone, it is deciding
  whether to raise a question that came out of the reading, and what she made of the text is
  the subject matter rather than background bleed. The reasoning is recorded at both sites.

  `wander_sft` gains its first self-test (`python -m core.wander_sft`), covering
  `reaction_for` and the previously untested `looks_trainable`.

- **Wander and lookup now record WHICH text a memory came from.** The blocker under the
  ask-side consumer above, and it was not an injection-design problem: the provenance id
  was not a key. Measured on the live store — of 30 self-directed asks, 17 were filed under
  `wiki:<site>` (`til_wander._write_wander_exchange` wrote `wiki:{wiki}`, the *site* and
  not the page, so seventeen questions pointed at "Lurkmore" as though that named a text)
  and 3 under the bare string `lookup`. Only the 10 `til:<date>` news asks could resolve.

  The wander fix is an ordering change: the snippet is now written *before* the live-memory
  write, and `mem_source` is taken from its stem (`til_facts.input_id`) instead of being
  invented from what was to hand at the top of the function. A declined or failed wander
  falls back to the old site-level id, which is exactly the case where there is no text on
  disk to point at anyway.

  The lookup fix is partial **by choice**: that lane reads a digest of several articles in
  one pass, so `_lookup_source_id` attributes a single-article digest to that article and
  leaves a genuinely multi-article one as the unresolvable `lookup`. Attributing a joint
  reading to whichever article came first would put the wrong text behind a question.
  Reading each article in its own pass would fix it properly, and is a change to what
  ingestion *costs* rather than to what it records, so it is not bundled here.

  Both fix new material only. Ids already written keep their value, so the existing pool
  stays at 10 of 30 resolvable and improves as the corpus turns over.

- **Activity-journal pass labels were cross-contaminated between idle jobs; every record
  written before this fix may name the wrong pass.** `activity_log.set_ambient_label` is
  thread-local *and* sticky, and the box has a single GPU executor thread. The reset that
  was supposed to bound a label to one job sat in `idle_scheduler._dispatch` — a coroutine
  on the **asyncio loop thread**, which never generates — so it labelled a thread nothing
  read, while the finer labels job bodies set for themselves (`til_wander`'s
  `til_facts:<kind>`, `modules`' `module:<name>`) were set from *inside* the executor and
  stuck to it for the life of the process. Measured on the live box's journal at
  `seq` 6599–7359: **58 check-in and 38 outreach generations reported under
  `til_facts:wander`**, 9 and 6 more under `chat_facts`; not one outreach or check-in pass
  in that window was labelled correctly. `background_reflection` was unaffected, because
  its labels come from `reflection_config.append_event`, which the runner calls on the
  executor thread.

  **Interpretation warning for existing records.** The journal's `message` embeds the
  label (`til_facts:wander: produced 4977 chars in 101s`), so an archived record's pass
  name is not trustworthy before this date — only its `source` is. The two disagreeing is
  the tell (`source: outreach` under a `til_facts` label); where they agree the label was
  probably just the source fallback.

  The fix moves the reset into the executor call (`idle_scheduler._labelled_run`), which is
  the only place it can reach the generations, and clears it on the way out so a job's
  label cannot ride onto the live-chat passes that share the thread. The stage-1 fact
  fetch now names itself explicitly (`pass_context("fact_fetch")`) instead of inheriting.
  `set_ambient_label`'s contract — thread-local, sticky, set it where you generate — is
  now stated at the function. Regression test in `core.idle_scheduler`'s self-test, driven
  through a single-worker executor because that is the condition the bug needs.

  Separately, the Activity tab rendered a `stream` heartbeat's body identically to a
  finished `body`. A heartbeat carries the rolling **tail** of a generation still being
  written (~240 chars), so it begins and ends mid-word by construction and the same pass
  appears several times as different mid-sentence fragments — which reads as a corrupted
  log rather than as a progress indicator. It is now marked with leading/trailing
  ellipses. Client-only; no protocol or server change.

- **A fetched fact now recalls the conversation it came from — retrieval keyed on a fact,
  not on a cosine.** Stage 1 of a live turn already reads the arriving message and picks
  which recorded facts it turns on. Every chat-lane claim knows which conversation
  established it (the occurrence level records `lane` + `source_ref`), so those picks now
  **nominate** their conversations to the past-chat channel, which injects the chat's gist
  in a reserved slot beside the anchor one.

  **The gap it closes.** All three existing routes into an old conversation — verbatim
  passages, the exchange anchor, the gist — are embedding matches against the arriving
  message. A conversation that shares no wording with it is therefore unreachable however
  relevant it is, and on this corpus that is the normal case rather than a corner: **886 of
  946 claims are written in English while the conversations are often Russian** (the
  measurement that already forced the claim-picking lane), and past `rag_cap_age_h` an aged
  chat has no verbatim vectors left to be found by at all. So a fact recalled from such a
  chat arrived as one decontextualized sentence — measured on the live corpus, *"Agreed with
  Me's view on subjectivity and masks"* — with the conversation that gives it its meaning
  sitting on disk, unreachable by any query that would surface the fact.

  **Not a new index and not a second injection path.** The nomination is a reserved slot in
  the block that already exists, resolved after the anchor slot and before the ranking, each
  claiming what it injects so the next never repeats it. An anchor that already reached the
  same conversation wins outright — its exchange is the finer-grained payload — and the
  ranking then skips the whole nominated session rather than adding a second slice of it.

  **Three properties are forced by the data, and each one closed a design question.** The
  grain is the conversation, never a turn in it, because `chat_facts` records per chat with
  no exchange index — so the payload is the gist, the only session-grained representation
  the box has, and a chat with no gist nominates *nothing* rather than having an exchange
  guessed for it. There is no relevance floor, no near-duplicate ceiling and no age modifier:
  nothing was matched, so there is nothing to threshold; a gist is a paraphrase and cannot be
  the verbatim-copy trap the ceiling rejects; and age is a ranking prior where nothing is
  competing. An old conversation is exactly what this is for. **TIL claims nominate nothing**,
  which is a finding rather than an omission — that lane has no distilled recap of any kind,
  so a per-snippet gist is now a concrete `til_facts` follow-up with a caller waiting for it.

  **Cost is bounded where it is spent.** The slot is additive on top of `top_k`, like an
  anchor, so the gist is excerpted to 700 chars against the 4000 a ranked passage may take —
  a full gist runs ~2900 chars on this corpus, the persona portrait's whole budget spent on
  one recollection. `graph.nominate_max` (default 1, `0` ⇒ off) is the knob; the excerpt rule
  is `chat_sidecar.gist_excerpt`, moved out of `checkin` now that it has two callers.

  **The one ordering fact that makes it possible:** the facts fetch now runs *before* the RAG
  query rather than after it, because its picks are an input to retrieval and not merely a
  block beside it. Asserted in `generation`'s self-test from both ends — the order alone
  would become decorative if the argument went missing, and the argument alone would nominate
  an empty list if the order slipped back.

  **Visibility.** The `facts_block` message carries `sessions`, and the Chat tab prints them
  on the header line (`· from 20260731_201512`). Reported rather than left to be inferred
  from the RAG block, because the engine's caps and fences mean the list is what was
  *offered*: an operator needs to see that a fact had a conversation behind it even on a turn
  where the slot went unspent. Safety is untouched and the reason is structural — a
  nomination is derived from a claim that already passed `claim_candidates`, so it introduces
  no new selection and `person:_self` and the `position`/`report` facets cannot reach it.

## 2026-08-12

- **The facts channel never ran on a live turn — a call-shape mismatch, silent by design.** (`2b825b9`)
  `generation._fetch_facts_block_sync` handed `fact_fetch.fetch_blob` the raw reflect
  generate seam. `fetch_blob` calls its generate as `generate(body, prompt,
  max_new_tokens=N)` — the small contract its two callers share — while the seam takes the
  box's full keyword set (`temperature`, `top_p`, `max_new_tokens_setting`, …), so every
  live turn raised `TypeError: unexpected keyword argument 'max_new_tokens'`.

  **The catch-all did exactly what it was written to do, which is why nobody noticed.**
  `fetch_blob` never raises — a retrieval channel must not be the reason a chat turn fails
  — so the error became `skipped: "generate_failed"` and an empty block, on every turn,
  from the commit that wired the channel to live chat (`8ed5fad`, on by default) until now.
  The workbench module calls the seam directly with the right keywords and was unaffected,
  which is the whole visible symptom: **the Modules tab produces facts and chat never does.**

  **What this means for anything logged in between.** No reply between those commits was
  conditioned on the facts tree, whatever `graph.enabled` said. The per-exchange
  `system_content` is logged verbatim, so a transcript is authoritative on its own: a turn
  that carried a block has one, and no turn in that window does. Nothing needs rebuilding —
  the channel injects, it does not write — but a reading of Ava's behaviour over that window
  that assumes the tree was reaching her is wrong.

  **The fix is an adapter that reads the pass's run flags off its `ModuleSpec`** rather than
  restating them, so the live path and the workbench simulation of it cannot condition the
  same pass differently: `disable_thinking` and `stop_on_repeat` come off the spec,
  `disable_rag=True` (the turn's own RAG block is retrieved separately), and sampling is
  greedy — matching every other thinking-off judgement pass on the box (`fact_dedup`,
  `persona_cluster`, `self_reconcile`), where the workbench passes the operator's own
  because varying it is what a workbench is for.

  **Two failures of instrumentation, both closed, and they are the reusable part.** The
  self-test's "enabled but unable ⇒ empty block, named reason" check accepted `error` as a
  passing outcome and, with no model loaded, returned at the `no_model` guard without ever
  reaching the call — a test that asserted the channel fails softly and could not tell that
  from failing always. It now runs the real code path against a stub carrying the *seam's
  signature* over a synthetic tree, so a call-shape drift is a test failure. And the call
  site discarded the `skipped` reason, so a permanently broken channel was indistinguishable
  from a quiet one; genuine failures (bad call, no tree, no prompt, no module) now print
  under the `[tag]` convention and reach the Activity journal through the stdout tee, while
  the ordinary empty outcomes — picked nothing, disabled — stay silent, since printing those
  every turn is how a log stops being read.

- **The fetched facts block is shown above the reply, on every turn, not behind Debug.**
  New `facts_block` server→client message, sent after the prompt dump and before the first
  chunk, so it lands above the CoT: an operator reads a thought against the material it was
  given, and a premise that arrives after the conclusion is something else to reconcile.
  Rendered in the Debug view's `rag` green, so somebody running both views recognises it as
  the same block in both.

  **Why it is not another line in `prompt_debug`.** The dump answers "what did this turn
  condition on" and is exhaustive and opt-in; this answers "what did she look up for *this*
  message", which is the only injected channel whose content was *selected for* the turn
  rather than embedded near it. It also carries the blob WITHOUT the constant wrapper — a
  per-turn view whose bulk is the same paragraph every turn stops being read.

  **An empty result still prints.** A reply built on nothing is a different reply from one
  built on three facts, and `n_candidates` is the denominator that makes "picked nothing"
  readable. The rendering separates the ordinary empty outcomes from the broken ones —
  green for *nothing bore on this message*, red for *no tree / failed generation* — the
  same split the server prints under `[facts]`, and the distinction whose absence let a
  dead channel look quiet for a day. A box with `graph.enabled` false sends nothing.

- **The facts channel no longer ages TIL claims out.** `graph.blob.DEFAULT_TIL_MAX_AGE_DAYS`
  goes from `7` to `None`, so a TIL-only claim is offered to the fetch pass whatever its
  age, exactly like a chat-backed one. The knob (`graph.til_max_age_days`) and every line of
  the mechanism stay — only the default changed, and an integer restores the old behaviour.

  **The cut was keyed on the wrong thing.** It read the *lane* as a proxy for
  perishability, on the reasoning that a news digest's "missiles were intercepted" is spent
  within days where a standing fact about a person holds next month. Both halves of that are
  true; what it missed is that the TIL lane is not the news lane. It also carries every
  wandered article — a place, a trope, an organisation — and those age like chat facts, not
  like headlines. So the scope was discarding the durable world material in order to bound
  the perishable, on a channel whose entire reason to exist is that wander and news are the
  only paths by which anything about the wider world enters the box.

  **What the cut was actually protecting is still protected.** The concern was the size and
  mix of the candidate list (TIL supplies 34.3 showable claims per protocol against chat's
  1.5, so the list drifts toward news as the corpus grows) and the prefill it becomes, paid
  on time-to-first-token every turn. That is `DEFAULT_MAX_CANDIDATES` (400) plus the
  chat-backed-first, TIL-newest-first ordering — which truncates *oldest TIL first* and
  never touches conversational material. It drops the same claims the age cut did, and only
  once the list is genuinely too long, rather than unconditionally at 7 days.

  Docs re-pointed from "the freshness scope bounds the prefill" to the cap in CLAUDE.md,
  `AVA_STATUS.md` and FACTS_TREE.md §10 (amended in place, beside the `graph.enabled` flip
  it records). Self-tests in `graph/blob.py` and `core/fact_fetch.py` now assert both
  directions: nothing is dropped by default, and an explicit scope still drops stale TIL
  while never touching a chat-backed claim.

- **Training-lite: a Sleep run that mints a persona without fitting an adapter.** A
  **Train (LoRA)** checkbox (ON by default) on the Sleep tab; unchecked, the run sends
  `stages = [reflection, merge-rag, commit-training, **persona**]` instead of `[…, train]`.

  **The premise it was requested under was wrong, and that is worth recording**: aborting
  a training cycle never discarded a reflection. `handle_start_reflection_run` commits
  merge-rag + commit-training, archives to `reflections/<run_id>/`, discards staging and
  finalizes the run *before* `_trigger_watchdog_train` fires — the code comment there
  already said the POST is "the last thing we do." And every build fits a fresh LoRA over
  the whole frozen-bundle corpus, so an aborted cycle costs GPU time only; the next build
  picks the work up in full. A train-less run was therefore already reachable over the
  protocol (it is exactly what *Revisit* sends) — there was simply no checkbox.

  **What was genuinely missing is the persona.** Live chat reads Ava's self-portrait
  through the ACTIVE-PERSONA pointer (`generation._current_chat_portrait` →
  `persona_paths.active_persona_dir()/digest.json`), not from the live `hot/persona/` dir
  a reflection run writes to. Every *other* product of a run — distilled memory, facts,
  sidecar targets, anchors, user portraits — is read from its live dir and lands the
  moment the run commits. Only the self-portrait is gated on a version being **minted**,
  and until now the sole minter was `train_cycle`'s promotion tail
  (`snapshot_state.produce_persona(persona_id, activate=True)`, on a passing probe). So a
  reflection run without training refreshed her self-portrait into a file nothing read:
  she reflected, and went on speaking as her previous self. The `persona` stage calls the
  same `produce_persona` directly with the adapter **unchanged** — GPU-free and
  filesystem-only, so it is safe with the model loaded — and is best-effort by the rule
  the archive beside it follows: a failed snapshot must not fail a committed reflection.

  Stripped from a **revisit** for the same reason `train` is: both PRODUCE a version of
  her, and a revisit is maintenance on one old chat — it also runs as the head-phase of
  every normal run, where minting would yield a persona per run regardless of the ask.
  Skip-validation greys out with training off (it gates an adapter promotion that no
  longer happens). Cost: one persona dir per run, adapter copy included (~1 GiB on the
  current box) — the same cost a promoted build already pays.

- **`phase_error` now renders in the Sleep tab.** It was absent from `_DISPLAY_EVENTS`
  while present in the activity journal's `_ACTIVITY_MIRROR_EVENTS`, so the best-effort
  phases — archive, and now persona activation — could fail and leave the operator
  reading a clean run. Pre-existing blind spot; surfaced by the persona stage, whose
  failure is precisely the thing that must not be silent.

- **Fetch adapter: the weights alone, grafted onto local state.** A fourth transfer path
  on the Migrate tab, and the first deliberately **hybrid** one — reflect and accumulate
  the corpus on this box, train on the box with GPU headroom, then run those weights
  against this box's memory. `GET /adapter/manifest` + `GET /adapter/export` on the
  inference sidecar, delegating to a new adapter-only scope in `snapshot_state`
  (`plan_adapter_manifest` / `stream_adapter`) so the scope is defined once beside the
  snapshot scope it deliberately narrows. The tar carries two members —
  `models/<adapter>/` (the same arcname `stream_snapshot` uses, so the client's
  member→destination mapping is shared) and `MANIFEST.json`. Client-side
  `FetchAdapterWorker` swaps that one dir in and repoints **`adapter_id` only**: chats,
  distilled memory, ledger, prompts, digest and the persona pointer are untouched, and
  the previous adapter stays in `models/` as the rollback path.

  **The base-compat guard is the substance of the feature, not a nicety.** `MigrateWorker`
  and `FetchSnapshotWorker` both replace local state wholesale, so adapter/base/config
  arrive internally consistent; this one does not, and a LoRA is fit against one specific
  base that **nothing downstream re-checks**. At 4-bit `inference_backend.load` resolves
  the base from the adapter's own `adapter_config.json`, so a mismatched adapter would
  quietly load a base the local config never named; at 8/16-bit `_stage_adapter_with_base`
  overwrites that field with the local `model_id` and the mismatch surfaces only as an
  opaque peft/unsloth error. Both are unrecoverable by inspection after the fact, so the
  compare runs on the CLIENT against `server_config.json` on disk (not the connected
  server's status — the config is what the guard protects) and **refuses before a byte
  moves**. `model_id` is never rewritten by the repoint, which would turn a refused
  mismatch into an accepted one on the next fetch.

- **The facts tree reaches a live chat turn — on by default.** `graph.enabled`
  makes a chat turn two-stage: a short thinking-off pass reads the numbered candidate list
  and the conversation, picks which recorded facts the arriving message turns on, and
  `graph.blob.render_claims` renders those into a block injected ahead of the RAG block.
  This is FACTS_TREE.md §10 **consumer 5** — the channel the design listed last and marked
  *"gated, off by default … here to be argued about, not assumed"*. It was built that way
  and turned **on by default** the same day, by decision, once it worked end to end. §10
  now records the flip rather than being left to contradict the code, which is what this
  repo's `code > STATUS > DESIGN` precedence rule is for. The switch remains: `false`
  restores the single-stage turn exactly, and an *unreadable* config yields `false`
  whatever the default, since an unknown-state box should not silently spend an extra
  generation per turn.

  **The cost is structural, and the default no longer lets inaction avoid it.** Stage 1 is
  a whole extra generation before the reply begins: cheap to decode (at most eight numbers)
  but its prefill is the candidate list, ~10k tokens on the current corpus, all paid on
  time-to-first-token. That is why the freshness scope was built into `claim_candidates`
  rather than deferred — it bounds the one quantity that now grows on every turn.

  **One definition, two callers.** The pass moved to `core/fact_fetch.py`
  (`last_real_user_turn` / `build_body` / `fetch_blob`), and the workbench module now
  delegates to it — the discipline `til_facts` already follows for its two callers, and the
  thing that makes the workbench worth having: a pass an operator tunes in the tab and a
  pass that runs on a live turn must not be two pieces of code that merely resemble each
  other. The split with `graph/blob.py` is the safety boundary and is unchanged: everything
  deciding *what may be shown* stays in the pure package and is applied to the candidate
  list before the pass sees it, so a bug in the new module can pick the wrong facts but
  cannot surface a forbidden one.

  **The two §10 rules therefore hold by construction rather than by check.** The candidate
  list is built already filtered to the knowledge facets with the `person:_self` subtree
  removed, so a `position` claim or anything of hers is never *shown* and cannot be picked,
  mis-parsed or argued in; and the block is assembled in code from returned ordinals, so
  nothing the pass generates becomes prose. The wrapper
  (`prompts/facts_block_prompt.txt`) leans on that guarantee explicitly — "records of fact,
  not of what anyone thinks" — because a model told "things you know" about material that
  was actually somebody's opinion would state it as fact, which is the failure the facet
  level exists to prevent.

  It rides the Chat tab's **Facts** checkbox (knowledge about the world, not Ava's self), so
  an A/B still works. It **fails soft on every path**: no tree, no prompt, no model, a
  failed generation, or nothing picked each yield an empty block with a named reason — a
  retrieval channel must never be why a chat turn fails. The block is a labelled
  `system_parts` entry, so the Debug view and the per-exchange logged `system_content` carry
  it for free and Chat review shows precisely what a turn conditioned on.

  **Untested on a live GPU, and now on by default.** Everything verified is the
  deterministic half plus a stubbed generate: gating, filtering, prompt assembly, parsing,
  rendering, and every failure path. Whether the model picks *well* from a real candidate
  list, and what the second stage does to time-to-first-token, are the two questions only a
  live run answers. Enabling by default does not answer them — it makes them urgent, and
  moves the first real measurement onto whoever pulls next.

- **The fetch pass picks facts instead of naming a subject — the language-mix fix.** The
  lexical ranking added earlier the same day worked, and could not do the thing this corpus
  actually needs: **886 of 946 claims are written in English while the conversations are
  often Russian.** No token-overlap score bridges that, so the honest outcome on a Russian
  turn was a correctly-empty blob. Cross-lingual relevance is exactly what a multilingual
  model does natively and what a lexical matcher cannot do at all, so the selection moves to
  the model — it now picks the CLAIMS from a numbered list, rather than naming a node and
  leaving code to rank what that node owns.

  That also fixes a granularity problem the ranking only mitigated: a person node yields
  their whole biography (31 knowledge claims for one person here) when the question was
  about one thing.

  **The two §10 rules now hold by construction rather than by check.** `claim_candidates`
  builds the list already filtered to the knowledge facets with the `person:_self` subtree
  removed, so a forbidden claim is never *shown* — it cannot be picked, mis-parsed, or
  argued into the blob, and there is no post-hoc rejection to get wrong. The blob is still
  rendered in code from returned ordinals, so nothing the pass generates becomes prose. What
  changed is which level of the tree does the work: the node spine stops being a browsing
  index, and the FACET level becomes the load-bearing part, since it is the only thing
  keeping 73% of the corpus (`position`) out of a knowledge channel.

  Candidates are **numbered per call** rather than keyed by `claim_id`: 16 hex characters is
  something a pass can typo into a *different valid claim*, an ordinal is not, and the
  numbering is never stored so it cannot go stale. A number outside the list is reported as
  the pass having invented a line, not silently dropped.

  **The recency scope is built in rather than deferred**, because the growth is lopsided:
  TIL supplies 34.3 showable claims per protocol against chat's 1.5, so an unscoped list
  becomes ~90% news within weeks and the conversational material it exists to surface is a
  rounding error inside it. TIL-only claims older than 7 days are dropped; chat-backed
  claims never age out; a claim corroborated in both lanes counts as chat-backed; and the
  ordering puts chat first so the hard cap bites the oldest news rather than the scarce
  half. On the live corpus: 261 offered of 421 on record, ~10k tokens in a 32768 window.

  The pass now sees the **whole prior conversation** instead of a fixed tail — what a
  message refers to without naming is usually established earlier, and that referent is
  exactly what a fetch pass needs — stopping before the arriving message's own reply, since
  in simulation that reply exists and showing it would hand the pass the answer it is
  fetching for. `session_transcript_turns` is already CoT-free, so that half was free. The
  transcript is the elastic side of the budget: an overlong conversation is trimmed oldest
  turn first, never the candidate list, which would silently narrow what may be picked.

  `build` returns `(blocks, context)` carrying the candidate list, since an ordinal means
  nothing without the list it indexed. The operator report now shows what the facet
  restriction **cost** on each pick — the positions held about the same subjects, never
  offered to the pass, rendered attributed as they would have to be. That is the §10
  argument with a price tag attached rather than a principle to take on faith.

  Node picking (`node_index` / `parse_selection` / `render_blob`) is kept as a read API and
  still tested; `fact_fetch` no longer uses it. One test was found self-satisfying — it
  searched this module's source for the call it was asserting and matched its own search
  string, so it passed after the call had been replaced; it now asserts on the function.

- **`fact_fetch` returned biography instead of relevance — two independent defects, both
  found on the first GPU simulation.** The symptom was one blob: a pass asked what to look
  up before answering picked `person:artemyvo` and got eight facts about him that had
  nothing to do with the conversation.

  **The input was the wrong shape, and the pass was right.** The simulated chat was an
  Ava-initiated reach-out, whose exchange 0 holds a *synthetic impulse* she wrote to herself
  — `session_transcript_turns` already renders it `(stage direction — you, not another
  person)` under the speaker `(initiative)`. `_fetch_build` took the last user-slot turn as
  *the message that has just arrived*, so it was handed "About 7 hours had passed since you
  last spoke with your friend" and, reasonably, fetched the friend. The existing guard only
  caught a chat with NO user turn at all. Every other reader on the box already excludes
  this turn — `checkin._user_turns` skips it whatever the speaker label,
  `dialogue_source.build_dialogue_anchor` masks it from training, the Chat-review tab labels
  it — so the fix is to apply that rule here rather than invent one. `_last_real_user_turn`
  checks both halves independently, because a transcript predating the `(initiative)`
  convention carries no speaker label to key on. It narrows rather than refuses: once the
  other side replies, the same chat is accepted at the real turn.

  **Nothing ranked the blob.** `claims_for_nodes` sorted by `(-n_sources, -n_occurrences,
  text)`, and every claim in this corpus has `n_sources == n_occurrences == 1` — exactly the
  condition `graph/DESIGN.md` had already recorded from the stage-1 build (*"0 collapsed, 0
  corroborated … nothing downstream should rank on corroboration yet, since it is currently
  a column of zeros"*). The key therefore collapsed to ASCII order on the text, and the blob
  was the alphabetically-first 8 of 31 knowledge claims — verified byte-identical to a pure
  alphabetical sort. A note in a design doc did not stop the code depending on the thing the
  note said not to depend on.

  `render_blob` now takes the arriving message and scores each claim by IDF-weighted token
  overlap against the claim corpus, so a word appearing in every claim contributes exactly
  zero — a stoplist derived from the data rather than written per language, which matters on
  a mixed Russian/English corpus. **Zero-scoring claims are dropped, not used as filler:** a
  node the message merely names is no licence to recite its biography, and an empty blob is
  the honest answer to "nothing on record speaks to this". The count of what was dropped is
  reported, because a large number means the *pick* was too coarse — a finding about the
  pass, not about the tree. Without a query the previous order is untouched, so the change
  is inert for any caller that does not supply one.

  The matcher is **injected** (`exchange_anchor.words_match`) rather than imported, so
  `graph/` still never imports the inference role while the inflected half of the corpus
  still scores; absent, scoring degrades to exact equality rather than breaking. Scoring is
  lexical by choice: this pass runs *before* a turn is answered, so an embedder call would
  buy relevance at the cost of time-to-first-token, for a channel that is off by default.

  Plumbing: `build` may now return `(blocks, context)` and every module's `finish` takes
  `context=`. `finish` receives only the raw generation, which holds the picked ids and not
  the message, so the context had to travel from the build; the tuple is optional, leaving
  the three modules that need no context exactly as they were.

  **A limit this exposes rather than fixes:** 886 of 946 claims are written in English while
  the user often writes Russian, and no lexical matcher bridges that. A Russian message
  against this corpus now correctly yields an *empty* blob where it previously yielded eight
  irrelevant English facts — better, but it means the channel is close to mute on half the
  traffic. Whether that is worth an embedder, or is another argument that FACTS_TREE.md §10
  consumer 5 should stay unbuilt, is undecided.

## 2026-08-11

- **The facts tree got its first reader, in the workbench, where it writes nothing.**
  `graph/blob.py` + the `fact_fetch` module are FACTS_TREE.md §10 **consumer 5** — the
  node-scoped retrieval channel the design doc lists LAST and marks *"gated, off by
  default … may well never be built; it is here to be argued about, not assumed."* The
  argument is the intended two-stage chat turn: a cheap pass picks what to look up, code
  renders a facts blob, a second pass writes the reply. Building it as a **module** is the
  form the doc's caution takes in code — the workbench returns its value instead of
  committing it, so this is stage 2 (read-only inspection) wearing consumer 5's shape,
  not consumer 5 shipped.
  - **The pass emits node ids, not prose, and therefore does not think**
    (`ModuleSpec.disable_thinking`, the first module to set it; it joins the seam's
    existing thinking-off list — branch chooser, branch judge, fact placement, anchors,
    the clustering/dedup evaluations, all of which select or judge rather than write).
    Rendering in code is what makes the blob *faithful*: the text comes out of the tree, so
    a fetch pass cannot paraphrase a claim into something the corpus never said, and the
    facet rule is enforced by the renderer rather than requested from a generation. A
    prose-writing stage 1 would need thinking, cost seconds on the chat hot path, and put
    a generation between the source of truth and the prompt.
  - **The retrieval axis is the mention edge, not node ownership — forced by the data.**
    `chat_facts` files a fact under *who it is about*, so on the chat lane every claim
    lands under one of the two people in the conversation: measured on this box, **3 nodes
    of 107 own any claim at all, while 106 carry mention edges**. The topics a message is
    actually about (`subjectivity` ×14, `Project Ava` ×5, `autonomy` ×3) own nothing. A
    channel keyed on node ownership would therefore return nothing for exactly the nodes a
    message is about; `claims_for_nodes` returns the claims that MENTION a picked node,
    attributed to the node that owns them.
  - **Two safety rules live in code, not in the prompt**, both being safety properties
    rather than quality ones: `person:_self` yields nothing ever (§10), and a pass that
    picks it is *reported* rather than silently filtered, so the attempt stays visible;
    and the blob carries `property`+`event` only, with `position`/`report` returned
    separately, counted, and rendered attributed (`attribution_line`) so the operator can
    see what the restriction costs.
  - **And it costs a lot, which is the finding this module exists to produce.** On a real
    transcript, picking the three most-referenced nodes yields **5 knowledge claims against
    42 withheld positions and 24 dropped `_self` claims**. The 5 are the entire knowledge
    yield of the busiest nodes in the corpus — and two of them (*"Agreed with Me's view on
    subjectivity and masks"*, *"feels no emotions regarding the conflict in Mali"*) are
    arguably misclassified moments rather than standing properties, which is a `chat_facts`
    classification question this surfaces for the first time. So the honest reading of
    stage 1 today is that the tree's *useful* material for a chat turn is overwhelmingly
    the attributed-position layer §10 excludes, and the corpus-level fix (a subject
    namespace that files a fact under its topic, not only its person) sits upstream in the
    protocol producers rather than in this channel.
  - Supporting: `ModuleInputError` lets a module refuse its own input with a sentence worth
    reading — "no facts tree, build it with `python -m graph.build`" instead of the generic
    "rendered no readable chat content", which would send an operator to the transcript
    when the problem is the tree. No client change: `lines`/`counts` render generically, and
    the module reuses the existing `chat` input lane. Self-tests `python -m graph.blob`
    (wired into `graph.selftest`) and the extended `python -m core.modules`.

- **The activity journal became the box's reporting log: levels, a generation seam, a
  stdout tee, and unsloth.** `core/activity_log.py` has been the single box-wide log since
  2026-07-28, but it recorded only *lifecycle* events plus a mirror of reflection-run
  phases. Three things it could not do were each observed in the same evening: a `til_facts`
  pass held the GPU for **21 minutes and emitted nothing** (`idle_scheduler._dispatch` opens
  the status chip with `set_current`, which deliberately writes no journal line, and the
  pass reports only on completion); everything outside `reflection_runner` — the TIL passes,
  module runs, `background_reflection`'s rungs — reported through `print()` to `server.log`,
  which no client polls; and the offline train cycle's output was reachable only through the
  watchdog, leaving a hole in the history where each build was.

  The record gained `level` (`event` / `body` / `stream` / `raw`) and a `text` field kept
  separate from the one-line `message`, so a body no longer has to be clipped to keep a
  headline readable. **Coverage comes from two insertion points, not per-subsystem wiring**:
  the hooks live in `generation._make_sync_reflect_generate` and `_make_agentic_generate` —
  the two functions every background generation passes through — so reflection, TIL/wander,
  outreach, check-in, synthesis, deliberation, modules and the clean-base evaluations all
  report their verbatim CoT + output with no call-site change; and `install_stdout_tee()`
  converts the codebase's universal `[tag] …` print convention into journal lines, which is
  also the only way to capture a foreign library's output.

  **Per-token deltas are deliberately not journalled.** A reflection run generates ~100k
  tokens, i.e. ~25k records at the generator's ~80-char batching, which would evict
  everything else from the ring on every run. A `stream` heartbeat every ~20 s (elapsed,
  tokens, rolling tail) plus the final verbatim `body` carries the same information for ~1%
  of the records — and the heartbeat is what makes a long pass legible *while it runs*.

  **Interpretation change for existing artifacts.** `reflection_config._mirror_to_activity`
  no longer attaches a pass's RAW generation to its `phase_done` / `pass_warning` /
  `pass_error` line: the seam now writes that text as its own `body` record under the same
  `activity_id`, so keeping both would log the same kilobytes twice. The division from here
  is **the seam owns what the model wrote; the mirror owns what the runner made of it**
  (the distilled `[fact]`/`[ask]`/`[resolved]` items). Journal lines written before this date
  carry the raw text inline in `message` and have no `level`; a reader must treat an absent
  `level` as `event`.

  **Two latent flaws fixed in passing, both of which would have become data loss as volume
  grew.** Rotation rewrote the journal down to the in-memory ring (2000 events) once the file
  passed 8 MB — i.e. it silently discarded history; it now closes the file into a numbered
  **segment** and keeps `retain_segments` of them. And a cursor older than the ring returned
  a short list with no indication anything had been skipped; the batch now carries `gap`,
  and `configure()` reads only the file's tail rather than all of it.

  **Training.** `train_cycle` configures the same journal and installs the same tee from its
  own process (it already puts `inference/` on `sys.path`), mirroring each line into
  `train_progress.jsonl` as well — that being the file the watchdog serves while the
  inference server is down, which is the only window a client cannot reach the journal
  directly. Cross-process appends are `flock`-guarded; `seq` uniqueness rests on the
  watchdog never running two writers at once.

  **Scope note.** `data/hot/activity/` is now excluded from runnable snapshots
  (`snapshot_state`, both the copytree and tar paths): it is telemetry about the *source*
  box — what it did, not what she knows — and it is now large enough that copying it into
  every *Fetch snapshot* would be a real cost. The public API contributes nothing to the
  journal by construction (its generate path has no hook), preserving the "requests are
  NEVER logged" guarantee. Design brief: `LOGGING.md`.

## 2026-08-10

- **The fact protocols got their first reader: `server/graph/`, the facts tree (stage 1).**
  `chat_facts` and `til_facts` each write an immutable, exhaustive per-source record, and
  both docstrings named the same consumer — *offline, a knowledge-graph build, not yet
  written*. Until now nothing read either one: every reference in the tree was a producer, a
  storage helper, or a lifecycle delete. This is that consumer, and it changes how the
  protocols should be interpreted from here.

  The fold is **node → facet → claim → occurrence**, with each fact's `entities` resolved to
  node ids as edges over that spine, and it is **derived and disposable — never a store**.
  The protocols stay the sources, so a better resolver is a *rebuild* rather than a
  migration, a re-reflected protocol supersedes with no supersession logic, and the tree can
  only ever disagree with its sources about resolution. Same source/fold discipline
  `weights_persona.jsonl` → ledger and `[impression]` → portrait already follow. There is no
  migration path by construction: `SCHEMA_VERSION` lets a reader refuse a tree, not convert
  one, and `read_tree` returns `None` for missing, corrupt and unrecognised alike.

  **What the corpus turned out to be, and why the facet level exists.** Counted over the 23
  live protocols: 398 facts, of which **73% are `position`** — a view someone held, true only
  as a record that they said it — and only 25% are `property` + `event`. 234 of the 349 chat
  facts are about Ava herself. So the chat lane is empirically a *position ledger*, not a
  knowledge store, and a tree without a facet level is a pile of opinions presented as
  knowledge. `KNOWLEDGE_FACETS` makes that exclusion structural rather than advisory. Facet
  is a pure `(lane, fact_class)` map, which also repairs a genuine cross-lane collision for
  free: the TIL lane writes reported events as `stated`, meaning *the text asserts this*, not
  *someone holds this view* — the same value meaning two different things in the two lanes.

  **Node ids are typed** (`person:` / `entity:` / `topic:` reserved / `unknown:`) because the
  two subject namespaces are incompatible by design, not by accident: a chat subject is a
  person key (7 distinct values over 349 facts — with `nobody`, `name` and `nvidia` sitting
  in it, since `normalize_person` takes anything off its generics list at face value), a TIL
  subject is an entity mention with the surface form preserved. A flat id space merges them,
  and merging them is how `nvidia` becomes a person. Stage 1 resolves by exact match plus a
  hand-edited alias table, and types the residue `unknown:` — **counted, not guessed**,
  following `chat_facts.UNSPECIFIED`'s rule that for a record whose purpose is machine
  processing, unlabelled is information and a wrong label is contamination. A casing
  heuristic for entity-vs-topic was tried against the data and declined: it separates the
  English half cleanly and fails on the Russian half, the exact limit
  `exchange_anchor.normalize_tag` documents for tags.

  **The read set is enumerated positively, and that is the load-bearing part.** 38 of the 61
  `.facts.json` files on this box are copies — persona lineage ×32, reflection checkpoint ×2,
  review archive ×2 — so a naive glob over `server/data` counts the same fact up to three
  times and reports the duplication as *corroboration*, the one error mode a fact fold must
  not have. `til_protocol_paths` therefore descends exactly one kind-dir deep rather than
  recursing, since a persona snapshot nests a whole `til/snippets/` tree underneath. Paths
  come from `training.reflections_path` (which gains `til_snippets_dir`/`graph_dir`) rather
  than being re-derived here, because a stale second copy of the chat-corpus path has already
  caused a real bug in this repo. Relatedly, corroboration counts **distinct sources** and
  not occurrences: one conversation restating a thing four times is one witness, and counting
  it as four is how a fold manufactures confidence it has not earned.

  **Two negative results from the first real build, both of which move the roadmap.**
  `0 collapsed, 0 corroborated` over 398 occurrences — not one fact in the corpus is restated
  verbatim, so the exact-match tier merges *nothing* and all of the dedup value sits in the
  stage-3 embedding tier; nothing downstream should rank on corroboration yet, since it is
  currently a column of zeros. And **86% of nodes (182 of 212) come out untyped**, the
  residue splitting exactly as the design predicted into proper nouns (`China`, `Israel`,
  `Project Ava`) interleaved with abstractions (`subjectivity` ×13, `identity`, `will`) —
  which is what the reserved `topic:` type is for.

  Stage 1 is pure, stdlib-only and GPU-free: no model, no embedder, no running server. It
  writes `data/graph/tree.json` (gitignored, disposable); `aliases.json` is the one
  non-derived file and the only one worth backing up. **Nothing consumes the tree yet, and
  that ordering is deliberate** — read-only inspection is stage 2, so the resolver's errors
  become reviewable before anything depends on them, the same reason the Debug tab is the
  outside-view portrait's primary surface. `person:_self` is browsable and reportable but
  feeds no injected artifact, or it becomes a fourth self-portrait extending the loop
  `self_portrait` was built to stay out of. Design and staging in `FACTS_TREE.md`,
  implementation notes in `server/graph/DESIGN.md`, self-test `python -m graph.selftest`.

- **What a module READS became data, so the workbench stopped being chat-only.** The
  registry described a pass's prompt, its chunking, its output rendering and its loop guard
  as data, but its *input* was hardcoded: `_load_session` path-guarded to the chats dir, a
  required `exchanges` list, `build(session, …)`. That was invisible while both modules read
  chats and fatal the moment one did not — `til_facts` reads a fetched article or news
  digest out of the snippets tree, which has no exchanges, is measured in characters and is
  addressed by a `<kind>/<name>` id, while everything above the loader (prompt override,
  block loop, streaming, refusals, the write-nothing contract) is identical.

  `ModuleSpec.source` names the input kind and `_INPUT_SOURCES` holds one entry per kind —
  load-one, list-all, plus the nouns the refusal messages use. The run loop is source-blind:
  it asks the source to load, and an empty input is the same refusal in either lane with
  only the sentence differing. A fourth lane is an entry in that dict plus `source=` on the
  spec. `units`/`unit_label` ride the result because "how big is this input" has no single
  name across lanes; `exchanges` stays as an alias so an older client still reads a number.

  Client-side the tab now asks the server what a module can run on (`list_module_inputs`,
  keyed on the SOURCE so the two chat modules share one listing) rather than knowing per
  module. The `chat` lane is a deliberate exception and keeps using the Chat tab's own
  `SessionsWorker` + `ChatReviewWidget._label_for` — a second server-side listing of the
  same transcripts would be exactly the drift that reuse exists to prevent — and both are
  normalized to `{id, label, sub}` so one renderer draws every lane. The chosen input is
  remembered **per lane**: a chat filename is not something the til module could run on, and
  a single shared slot would carry one into a run that then refused it.

  One bug found by testing the listing against the real corpus rather than a fixture:
  `_list_chat_inputs` passed `p.name` to `chat_sidecar.is_chat_session_json`, which takes a
  `Path` — and `list_inputs`'s own except-and-return-`[]` guard turned that into a silently
  empty chat list rather than an error. Fixed; the self-test now asserts no sidecar reaches
  the listing, which is the failure that predicate exists to prevent (a module reading Ava's
  own summary back as though it were the conversation).
  | `inference/core/modules.py`, `inference/server.py`, `client/ui/modules_widget.py`,
  `client/core/backend_client.py`

- **The wander/news lane got a fact protocol of its own, so what an article *established*
  survives the pass that decided what was worth keeping.** `learning_prompt.txt` and
  `wander_prompt.txt` are curation passes and say so in as many words — most of what you
  read is noise, finding almost all of it irrelevant is correct — which is the right
  discipline for memory and the wrong one for a record. The text is read once, a handful of
  `[fact]` items reach `rag_memory.jsonl`, and everything else it stated is gone. Measured
  on the 2026-08-09 snapshot: 4 news digests yielded 26 kept facts between them and 10
  wandered articles yielded 15, with visible redundancy inside a single digest (the 2026
  Iran war, the trade war and the Gironde fires each kept twice from `til:2026-07-27`).
  This matters more here than on chats, because wander and news are the only paths by which
  anything about the wider world enters the box — a knowledge-graph build over chat
  protocols alone is a graph of what the people Ava talks to said.

  `core/til_facts.py` is the lane's data contract and `prompts/til_facts_prompt.txt` its
  pass. Same contract as the chat protocol — immutable, exhaustive, never deduped, never
  evicted, read by nothing at runtime, offline consumer — and the **same parser**
  (`chat_facts.parse_facts`, now taking a `subject_fn`), so a marker-placement or format
  fix lands on both lanes at once. Two divergences, each forced:

  *The subject namespace.* `chat_facts.normalize_subject` resolves to a PERSON key via
  `normalize_person`, which lowercases and reduces to a first token. On a chat that is
  right. Here nearly every subject is a country, an organisation, a trope or an event:
  the person reducer files "New York" under `new` and "United States" under `united`, and
  everything it does not recognise collapses to `""` — *nobody in particular* — which on a
  world-facts lane is the normal case and therefore invisible. So a subject here is an
  entity **mention**, surface form preserved, normalized only for formatting; resolving
  mentions to canonical nodes needs the whole corpus and belongs to the build. This is the
  identical argument `normalize_entities` already makes, applied to the subject slot.

  *Provenance instead of attribution.* A chat fact carries who it is about and who said it,
  because the speaker is a person whose testimony is first-hand or hearsay. Here the source
  is a text, so the document carries the text's identity
  (`source_kind`/`source_ref`/`source_url`/`source_title`/`source_date`) and fact-level
  attribution is simply absent. It matters more on this lane than on chats: the approved
  wiki list is chosen for **tone**, not truth — Lurkmore, Neolurk and WikiTropes are humour
  wikis — so a current-events digest and a joke wiki asserting one sentence are not the
  same evidence. Deliberately NOT resolved into a per-fact confidence: judging a source is
  an interpretation, this pass is a witness, and a score invented per line is the one field
  a later build could not recompute. Provenance is recorded exactly; weighting by it is the
  build's job.

  Storage is `<snippet_stem>.facts.json` beside the `.txt`/`.json` the fetchers already
  write under `data/til/snippets/<kind>/` — a better stem space than the chat lane's, where
  transcript and record are separate files kept in step, and inside what `snapshot_state`
  captures. Written at fetch (news, in `ingest_news`) and at Apply (wander, beside
  `write_random_snippet`, so a declined wander still leaves nothing behind), both last and
  best-effort so a failure costs the record and never the reading. `stop_on_repeat` is OFF
  for the reason the chat pass turns it off, and more sharply: a digest is many events about
  one country, so a run of same-`(about, class)` lines — which is what defeats the guard's
  12-token window — is this lane's normal shape rather than its corner case.

  Unlike a chat, an article is read exactly once and never looked at again, so a pass wired
  only into the reading paths would cover nothing already on disk. `til_facts.list_backlog`
  + `til_wander.drain_facts_backlog` close that, capped at `_TIL_FACTS_PER_RUN` (3) per
  reflection run's ingestion phase, and they fold in the shared staleness probe so this lane
  self-heals on a parser fix exactly as the chat one does. Reading blocks are paragraph-
  packed rather than single: every snippet in the current corpus fits one block at the 24k
  budget, but the pre-wipe corpus held a 26,001-character article and a chat is bounded by
  the context it ran in where an article is bounded by nothing.

  **Workbench entry added the same day** (below) — the pass is tunable from the Modules tab.
  | `inference/core/til_facts.py`, `inference/core/til_wander.py`,
  `inference/core/reflection_service.py`, `inference/core/chat_facts.py`,
  `prompts/til_facts_prompt.txt`

- **The fact protocol was reading one of the two marker placements its own pass uses, and
  losing a quarter of the corpus to the other.** `chat_facts.parse_facts` consumed
  `(about:)`/`(class:)`/`(entities:)`/`(when:)` markers as a run off the FRONT of a
  `[fact]` line, which is the form `chat_facts_prompt.txt` specifies. The pass also emits
  them **trailing** — `[fact] <statement>. (about: X) (class: stated)` — and against that
  shape the leading-only reader did not degrade a record, it destroyed one: subject, class
  and entities all came back empty and the markers stayed in the text as prose.

  **Measured on the live corpus before the fix** (2026-08-09 snapshot, 15 chats / 267
  facts): 70 facts — 26% — parsed that way, and **not one line anywhere was genuinely
  unmarked**, so every loss was recoverable rather than absent. Discarded across those 70:
  56 `about`, 70 `class`, 51 `entities`. The damage is not spread evenly, which is what hid
  it: the placement is chosen per **generation**, so 4 of the 15 chats emitted trailing
  markers on *every* line and 10 on *none* — four entirely unusable protocols beside eleven
  clean ones, rather than thinned quality everywhere. A generation may also split the run
  across both ends of a line, which is the quieter form: 2 further records parsed their
  leading `(about:)`/`(class:)` correctly and lost only a trailing `(when:)`, and those go
  some way to explaining how sparse that field looks on disk (4 of 267 populated).

  Both ends are now consumed (`_consume_markers`), leading position authoritative where the
  two disagree, tolerating a stray sentence terminator after the run (the pass closes the
  *line* after its markers, and the statement already carries its own stop). The middle of a
  line is deliberately **not** scanned and consumption is restricted to the known marker
  keys: a statement may legitimately end on a parenthetical ("moved to Anthropic (formerly:
  OpenAI)"), and silently rewriting the text of an immutable record is worse than failing to
  label it. The prompt's own format line, echoed back as data, is dropped on its placeholder
  value (`about: NAME`, `entities: A, B`) — a schema property, not the prompt's wording,
  which is operator-editable on disk and must not be something the parser knows. The
  `MAX_FACT_CHARS` clip also moved after the strip: no live record ever reached the cap, so
  this closes a latent hazard rather than an observed one — under the old order a long
  enough trailing-marked line would have had its markers cut mid-token into the text, where
  no later reader could recover them.

  **Interpreting existing artifacts:** every `<stem>.facts.json` written before this carries
  the damage if its generation happened to choose the trailing form — recognisable as
  `subject_raw: ""` + `fact_class: "unspecified"` with marker syntax still inside `text`.
  Nothing consumes these files yet, so nothing downstream was corrupted, and they are
  regenerable (the pass rewrites the file whole per reflection). Replaying the live corpus
  through the new parser recovers every discarded marker.

  **They are also repaired without an operator.** `chat_facts.is_stale_record` /
  `needs_reparse` ask whether today's parser can read more off a stored record than it
  carries, and `background_reflection.list_sidecar_backlog` now treats a stale protocol as
  a wanted one — so rung 1 of the background wake re-derives it at the price of the single
  cheap generation a *missing* sidecar already costs, on every box, on idle time. This
  matters because several instances run this codebase over different corpora: which files
  are damaged is a per-box fact that cannot be named in advance, and a fix shipped as a
  list of filenames would repair one box and no other.

  The probe is deliberately not a `SCHEMA_VERSION` bump and not a signature for this bug.
  A version bump asks "was this written by older code", which condemns every file the
  change touched — on this corpus it would have paid a generation to rewrite eleven clean
  protocols identically. Asking instead whether the stored record *disagrees with the
  current reader* flags exactly the ones that do, and keeps working: the next time
  `parse_facts` learns to read something it used to drop, the records that lost it are
  flagged with no new code and no version to remember, and the detector cannot drift from
  the fixer because it is the fixer. It found the 70 known records plus the 2 partial ones
  above, across 6 of 15 files, with no false positives on the other 9. | `inference/core/chat_facts.py`, `inference/core/background_reflection.py`

  **Why this was worth fixing before extending the pattern.** The proposal on the table is a
  wander/news counterpart of this record. The prompt's rule is that a fact about nobody in
  particular takes no `about` marker — a minority on chats, where a person-subject is the
  norm and a wall of empty subjects stands out. On a lane of world facts it is the majority,
  so this identical failure would produce exactly the output correct behaviour produces, and
  would not be detectable by reading the file. The lane being added is the one where the bug
  goes silent.

## 2026-08-08

- **The background pass became a three-rung ladder: sweep, then cheap artifacts corpus-wide,
  then deep per-chat reflection.** `core/background_reflection.py` did exactly one thing per
  wake — drain the unreflected-chat backlog — which meant two things went unserved. The
  stale-reach-out deletion above ran only at the head of an *operator* Sleep run, so a box
  nobody was reflecting on kept its unanswered openers indefinitely; and the two cheap
  per-chat artifacts (the gist `<stem>.summary.json` and the fact protocol
  `<stem>.facts.json`) were only ever produced *inside* a full reflection, so a corpus of
  chats predating them could only acquire them at the price of a per-exchange reflection
  each. A wake now takes rung 0 always (delete stale reach-outs — GPU-free, and first so no
  rung below spends a generation on a transcript about to be removed), then the FIRST rung
  with work: rung 1 backfills missing gist/fact sidecars **oldest first** across the whole
  corpus, and only when that is empty does rung 2 do the per-chat reflection drain. The
  ordering is the point: broad recall and a complete extraction record arrive at two cheap
  generations per chat, long before deep per-chat targets arrive at one generation per
  exchange. The overlap (a later reflection re-derives both sidecars) is accepted knowingly.

  Two seams rather than new copies. The sweep's policy — window, the live session's
  exemption, the chat-index refresh — came out of the reflection run's head phase as
  `reflection_service.run_stale_reachout_sweep`, and both callers now share it, so "stale
  reach-out" means one thing on the box. The backfill's generation is
  `core.modules.run_module_blocking` — the same `chat_summary`/`chat_facts` specs the
  Modules tab simulates with — so what lands on disk is what an operator can reproduce and
  retune from the workbench. That makes the module registry's **detached sink** load-bearing
  for the first time: the pass returns its value and this caller decides where it goes
  (`ChatSidecar.write_summary`/`write_facts`), which is precisely the property
  `core/modules.py` was built to have and had no user for.

  The failure mode worth naming: rung 1 *gates* rung 2, so anything that can sit in the
  backfill backlog forever stops reflection forever. Two guards. A chat whose passes
  generate but produce nothing writable (a two-line chat whose recap never survives
  `sanitize_gist`) is retired for the process — in memory, not on disk, because the cause is
  usually the prompt or the model and a restart is how those change. A **box-level** refusal
  (no model loaded, prompt file missing) instead stops the drain and reports a skip, so the
  scheduler retries promptly and no chat is retired for a condition that was never about it.
  `background_reflection.sidecar_backfill: false` removes the rung entirely.
  (`background_reflection.py`, `reflection_service.py`, `server.py`)

- **Unanswered reach-outs are deleted from the corpus — and the record that she was
  ignored is deliberately kept.** A chat Ava opens herself and nobody answers is the one
  session shape reflection can never do anything with: both paths skip it *un-frozen*
  (correctly — a later reply must still make it reflectable), so it stays in the backlog
  forever, sits in the chat list as something to open, and is embedded into chat RAG where
  her own unanswered message can come back as though it had been part of a dialogue. Past
  the point where a reply is realistic it is residue, so a reflection run now deletes it
  (`chat_worklog.delete_stale_reachouts`, at the head of the run via
  `reflection_service._run_stale_reachout_sweep`, window
  `reachout.stale_delete_hours` — default 48 h, `0` ⇒ off).

  The deletion is the easy half. **Three separate guards read exactly those files**, and
  each counts the same thing: messages she sent and was ignored on. The reach-out backoff
  (`reachout_gate.unanswered_streak`) widens her window per unanswered opener. Check-in's
  standing-opener list (`_standing_openers`) quotes them back into the decision prompt so
  the pass can tell its first message from its fifteenth. Outreach's dangling-opener guard
  (`_has_dangling_opener`) refuses to re-raise an ask she is still awaiting an answer on,
  and is documented as *deliberately not time-limited*.

  All three would have failed in the same direction, and it is worth naming why: the
  openers old enough to delete are **by construction the longest-ignored ones**. Deleting
  them would have made each guard weaken exactly as the silence lengthened — the backoff
  shrinking, the standing list blanking every 48 h, the un-time-limited guard becoming
  "until the sweep runs" — which is the inversion of what each rule says. That is the
  2026-07-30/31 pathology (15 cold-opens in 19 hours) reintroduced through the back door,
  by a change that never mentions it.

  So the sweep **tombstones** each opener before unlinking it: stem, addressee, send time
  and her message text, appended to `data/hot/reachout/expired.jsonl` via
  `reachout_gate.record_expired`. All three consumers fold the tombstones in beside the
  live files — deduped by stem, under the same "sent after the user's last turn" cutoff —
  so every one of them behaves as though nothing had been deleted. A tombstone that cannot
  be written **aborts that deletion**: losing the evidence of a reach-out is worse than
  keeping a stale transcript. What is deleted is the transcript, the thing that was never a
  conversation; what survives is the fact that she said it and was not answered.

  How to read the artifacts afterwards: an unanswered opener from before this date may be
  gone from `data/chats/` while the corpus still behaves as if it were there, and the
  reason it behaves that way is a tombstone, not a transcript. The tombstones are wiped
  with the **chats** (`state_wipe.wipe_chats_and_archive`), not with regenerable state —
  while the corpus lives, they are part of what it records. `stale_delete_hours: 0`
  restores the previous behaviour of keeping unanswered openers forever.

  Untested on a live GPU (the sweep is GPU-free; the self-tests cover it).

- **The summary pass gets a closing, and stops answering the conversation it was asked to
  recap.** Observed in the Modules tab: `chat_summary` sometimes replies to the chat's last
  message instead of producing a gist.

  Not a wording problem in `summary_prompt.txt` — a content-assembly one, and one this
  codebase had already diagnosed twice. The task is stated in the system message; a whole
  transcript then sits between it and the first generated token, so the nearest thing to
  continue is the reply that ended the chat. `reflection_source.build_session_reading_content`
  takes a `closing` argument for exactly this reason ("the final thing in the prompt is the
  task rather than something to continue"), added after the user-notes pass was seen handing
  Ava's own reply back as an impression; the recollection and self-notes passes have their
  own. The summary pass is the one reading pass that never got one, because it rides
  `reflection_chunking` instead of that builder.

  `format_chunk_content` and `build_consolidation_chunks` now take `closing`, defaulting
  empty — consolidation is byte-for-byte unchanged, and its `## WEIGHTS`/`## RAG` output
  shape was never one a transcript could be mistaken for. The text has ONE definition
  (`reflection_chunking.SUMMARY_CLOSING`) and, unlike its three siblings, says out loud
  that the transcript is a record and nobody is waiting on a reply: "write a recap of this
  conversation" and "reply to this conversation" are close enough in shape that the framing
  has to be explicit.

  It is applied at a different point on each side, which is the one thing a future reader
  should not have to rediscover. The module bakes it into the chunks, so it is counted by
  the packing budget. Production appends it at the summary call site (`append_closing`),
  because those chunks are **shared with the consolidation pass** and a "now write the
  recap" line would break that contract — so production's closing is not budgeted, which is
  safe only because that pass drops the RAG block (up to `rag_token_limit`) against a
  closing of a few dozen tokens, and an overflow is a caught `PromptBudgetError` that skips
  the gist with a warning rather than failing the run.

  **Interpretation note for existing artifacts:** gists in `<stem>.summary.json` written
  before this date were produced without the closing, so a recap that reads as a reply to
  the final message is that bug, not a degenerate model. They are regenerable — re-reflect
  the chat.

## 2026-08-07

- **A second module (`chat_summary`), and the registry shape it forced.**
  The workbench's claim was that adding a pass is a data edit. Adding the consolidation
  gist tested it, and it was half true: the registry was still facts-shaped in two places.

  `ModuleSpec.parse`/`summarize` assumed a pass emits a LIST of records, which cannot
  describe one that emits prose; and the client rendered those records with a hardcoded
  `(class) about subject: text`, so a second module would have needed a second renderer.
  Both are now per-module: **`build`** returns a list of content blocks and **`finish`**
  turns the raw generations into the displayed result, including its own `lines`. The
  client prints `lines` verbatim, so a third module needs no client change — which is the
  property that made the registry worth building.

  `build` returning a *list* is not generality for its own sake. The production summary
  pass runs once per **consolidation chunk** over the same packed transcript and joins the
  parts, so a long chat genuinely is several generations; approximating that with the
  whole-session reading block would have made the simulation lie about the shape of the
  work. The module reproduces the chunking (`reflection_chunking.build_consolidation_chunks`)
  and the gist sanitation (`chat_sidecar.sanitize_gist`), streams a part marker between
  blocks, and taints the whole result if any block truncates — these are parts of one
  value, so a clean part 1 does not redeem a cut part 2.

  One deliberate divergence, stated rather than hidden: the chunks are built with an EMPTY
  open-questions block. Production packs the live open `[ask]` items into that content,
  which is an injection, and this module declares that it injects nothing. Declaring the
  open-questions block a real injectable is the honest fix and belongs with the injectables
  work. One improvement over production, too: a part the gist sanitizer rejects (a
  generation that degenerated into the structured consolidation dump, or came back too
  short to be prose) is REPORTED. Production drops it silently, which is why a chat with no
  gist is indistinguishable from a chat that was never summarized.

  Note what this module is aimed at. `checkin._stored_gist` already reads reflection's
  stored summary and only generates its own when there isn't one — the chat-summary →
  check-in chain exists, hardcoded. Having both ends in the registry is what makes it a
  wire rather than a special case.

- **A clock for every reflection pass, and two graph fields on the chat-facts record.**
  Groundwork for the offline knowledge-graph build, plus the fix for a gap logged in
  `AVA_OPEN_PROBLEMS.md → Prompt Composition`.

  **The clock.** `generation._reflect_system_parts` now leads with `_temporal_anchor()`,
  so every reflection pass knows what day it is. It previously reached only the five
  subsystems that appended it to their own prompt by hand — at a different position each,
  `synthesis` prepending and the rest appending — and reached no `reflection_runner` pass
  at all. The recollection pass, whose entire premise is re-reading an old conversation *as
  who she is now*, therefore had no "now"; and `chat_facts` had nothing to resolve "on
  Tuesday" against. Those five (`checkin`, `outreach`, `synthesis`, `deliberation`,
  `prompt_experiment`) no longer compose their own and no longer take `temporal_anchor` in
  `configure(...)` at all, since composing it in both places would date-stamp the prompt
  twice. The anchor leads rather than trails because the pass prompt must stay LAST (it
  ends in the output contract), which also mirrors chat's framing-before-situational order.
  The IDEAL replay path is untouched by construction: `messages_override` composes its own
  system turn and never reaches `_reflect_system_parts`, so training parity holds. Live
  chat and encounter keep their own anchors.

  Removing the five injected capabilities meant editing five `configure` signatures and
  five `server.py` call sites, on the boot path, with no way to run the server here — so
  it was verified with an AST check that walks every `<module>.configure(...)` call in
  `server.py` and compares its kwargs against the imported module's real signature. It
  caught all five mismatches before the fix and passes across all 16 calls after.

  **The record's two new fields.** `entities` is the OTHER nodes a fact touches — the
  people, places and things it names or refers to — stored as MENTION strings rather than
  resolved keys. The split follows what only the extractor can know: extracting the mention
  needs the *conversation* ("her sister lives in Haifa" names one node and refers to
  another, and the pass reading the transcript knows whose sister), while resolving
  mentions to canonical nodes needs the whole corpus at once and belongs to the build,
  which can re-run when it improves. So `normalize_entities` does formatting only and
  preserves the surface form — deliberately not `normalize_person`, which reduces to a
  first token and would turn "New York" into "new". `when` is an `event`'s date, and it is
  what the clock was for; note the referent is the CONVERSATION's date, not today, so
  `reflection_source.session_date_line` now dates the transcript in the reading content, in
  the same strftime format as the temporal anchor so the gap between them is arithmetic
  rather than inference.

  **Topic tags were proposed and declined**, which is the more interesting half. Three
  objections. A topic label is an interpretation, and this pass is defined against exactly
  that ("You are a witness here, not an interpreter"; "Do not select") — the risk is not
  the field but the bleed, since a pass deciding what each fact is *about* is under
  pressure to record only facts worth labelling, losing the trivia the protocol exists to
  hold. A free tag vocabulary generated per chat, with no view of what the other chats
  used, fragments by language, granularity and morphology in a mixed-language corpus —
  which is not a prediction but a measured result, recorded in
  `exchange_anchor.normalize_tag`: normalization there is "deliberately the formatting half
  of tag convergence only … cross-language aliasing needs a registry". And tags are
  derived, revisable and globally dependent, while this record is immutable and locally
  produced — the wrong side of the source-vs-fold line the module draws between itself and
  `rag_memory.jsonl`. Topic clustering belongs to the offline build, which has the global
  view and can reuse `fact_dedup`'s subject blocking and `persona_cluster`'s map-reduce.
  If tags are revisited, the cheap test now exists: same chat through the Modules tab with
  tags on and off, comparing fact count and trivia retention.

- **Her own facts were being filed under "nobody", and she was skipping her own opener.**
  Both reported from the Modules tab on Ava-initiated chats; two causes, one of them
  reproducible with no GPU at all.

  **The naming one.** `normalize_person` collapses a fixed set of generic referents to
  `""` — *a fact about nobody in particular*, the class a place or a tool takes — and that
  set contains `me`, `i`, `user` and `the user`. Nothing in the pipeline gives Ava a name
  to use instead: `chat_prompt.txt` contains no "Ava", the reflect lane composes no
  identity line (it is `[RAG block] + [pass prompt]` and nothing else), and
  `session_transcript_turns` renders her turns as a bare `Me:` for replay fidelity. So a
  pass asked for `(about: NAME)` had, for her own facts, only labels that mean *nobody* —
  and every fact she stated about herself was stored as being about no one. Invisible on
  an ordinary chat, where the facts are mostly about the named user; on a conversation she
  started, where most of the content is hers, it is nearly all of them. This is the
  "reflect lane composes no identity" gap from `AVA_OPEN_PROBLEMS.md → Prompt Composition`
  arriving as data loss rather than as a design smell.

  Fixed with a reserved marker: `chat_facts.SELF_SUBJECT` (`_self`), taught by the prompt
  as `(about: self)` and resolved by a new `normalize_subject`. A marker rather than a
  configured name because this project's premise is that her identity accumulates rather
  than being declared, and naming her by fiat to satisfy a schema is the wrong trade; the
  precedent is exact, since `user_digest.RESERVED_SLUGS` already holds `_self` for the
  outside-view portrait, which exists because she is not a person in the people namespace
  either. The leading underscore keeps it clear of that namespace — `normalize_person`
  strips surrounding punctuation but not underscores, so a name would have to *start* with
  one to collide. Only unambiguous first-person labels map (`self`/`myself`/`me`/`i`):
  `the user` deliberately does not, since it normally means the human and guessing would
  misfile a real person's fact, and neither does her name, since recognizing one would put
  a hardcoded name in a codebase that has none. The carve-out is LOCAL to `chat_facts` —
  `normalize_person` is shared with the live fact store, eviction and dedup, and widening
  its vocabulary would reach all of them. `subject_raw` already preserved the label she
  wrote, so the damage in existing files is measurable rather than merely regrettable:
  `grep -ho '"subject": "", "subject_raw": "[^"]*"' data/hot/chats/*.facts.json | sort |
  uniq -c`.

  **The missing opener, two causes.** First, `_AVA_INITIATED_NOTE` is written for the
  REVISION pass — it closes "when judging your reply" and tells her the self-descriptions
  in a reversed session are her own "not facts about them" — and
  `build_session_reading_content` prepended it for *every* reading pass. To a pass whose
  job is to write down what was said, "not facts about them" with no positive counterpart
  reads as an instruction to skip her own turns. The note is now a parameter, and
  `chat_facts` supplies its own (`chat_facts.AVA_INITIATED_NOTE`, one definition shared by
  the production pass and the module registry) saying the same structural thing and then
  saying where her own material goes. Second, `format_context_block` drops oldest-first,
  so on any reversed session long enough to truncate, the opener is the first casualty —
  while being the turn the conversation exists for. It now takes `keep_first`, pinning the
  opening group and budgeting the rest around it (degrading to the plain walk if the
  opener alone would eat the budget), and `build_session_reading_content` passes it for
  Ava-initiated sessions. That reaches the sibling reading passes (user notes, self notes,
  recollection) as well, which is intended; the revision context is deliberately untouched,
  where dropping oldest-first is correct.

  New GPU-free self-test `python -m core.reflection_source` covers the pinned walk, the
  oversized-opener degradation, and that the default path stays byte-identical.

- **The chat-facts pass was being killed by the verbatim loop guard after 4 facts.**
  Found on the first real run of the new Modules tab, which is what it was built for. A
  simulation over a 7-exchange chat stopped at roughly 1,200 of 12,288 tokens with nothing
  parsed, and the tab reported it as running out of budget — 90% of which was unspent.

  The cause is `stop_on_repeat`, hardcoded `True` on the reflect lane
  (`generation._make_sync_reflect_generate`). Its guard hashes the **last 12 generated
  tokens** each step and halts once a span recurs **4 times**
  (`inference_backend._RepetitionStop`, `_LOOP_NGRAM=12`, `_LOOP_MIN_REPEATS=4`). That is
  right for prose and for a small labelled block, and **deterministically wrong** for a
  pass whose correct output is a fixed-template list: `chat_facts` emits one
  `[fact] (about: NAME) (class: CLASS) …` line per fact, and that prefix alone tokenizes
  past 12 — so the rolling window sits *wholly inside the prefix* and is byte-identical on
  every line sharing an (about, class) pair, whatever the content between them. The guard
  therefore fires on the 4th consecutive such fact. On a single-topic chat (one subject,
  everything `stated`) the pass cannot emit more than three facts at all. The observed
  generation stopped exactly at the end of the repeating span — `…stated that` with nothing
  after it — which is the signature.

  This was **not** a workbench artifact: the production pass generates through the same
  factory with the same flag, so live `.facts.json` files have been truncated the same way
  since the pass shipped (2026-08-06), surfacing only as "Chat-facts output unusable
  (skipped)" in the run log. It is the mirror image of the RAG-block finding of 2026-08-04,
  where an identical repeated line prefix acts as a list-continuation *seed*: same
  structural fact about templated lists, biting from the other side.

  Fixed by threading `stop_on_repeat` through the reflect-generate factory (default `True`,
  so nothing else moves) and turning it off for this one pass, at both call sites — the
  production pass in `reflection_runner` and `core.modules._CHAT_FACTS`. It lives on the
  `ModuleSpec` because whether the guard is correct is a property of the **output shape**,
  not of the call site. Layer 2 (`_DegenStop`, token-diversity based) is untouched and
  still catches a genuine collapse, so declining the wrong guard is not running unguarded.
  The two guards separating this cleanly is luck, not design, and worth remembering if a
  future pass needs the same exemption. `core.modules`'s self-test now asserts the registry
  and the runner still agree on both the closing question and the flag — the duplication is
  temporary, and it is only safe while a divergence is caught.

  Second defect, in the diagnosis rather than the generation: **both stop guards read as a
  truncation** to the caller (neither ends on EOS), so `last_truncated` cannot distinguish
  "the budget ran out" from "a guard halted the model far short of it" — and the two need
  opposite responses. The backend has always exposed the difference as
  `generate_fn.last_loop`; the module was not reading it, so the tab told the operator to
  raise a budget that was almost entirely unspent. `module_done` now carries
  `stopped_on_loop` and the tab names the real cause.


- **Module workbench (v0): a pass can now be run on its own, against a chosen chat, writing
  nothing.**
  Every generation on this box is the same shape — something is injected into the prompt,
  something is read, something is produced — but each pass hardcodes its own answer to all
  three inside the run it belongs to. Two consequences had accumulated: nobody can say what a
  given pass sees without reading its call site, and nobody can try a change to one without
  triggering the whole run. The audit behind this change found the concrete cost. The reflect
  factory forwards **three** of `RagEngine.query`'s twelve channel gates
  (`generation._make_sync_reflect_generate` → `chat`/`recollections`/`impressions`), so no
  reflection pass can turn off `[persona]`, `[ask]`, `[fact]` or anchors — meaning
  `chat_facts`, `user_notes` and `self_notes`, whose stated discipline is *witness, don't
  interpret*, are each conditioned on Ava's persona statements and open questions by default
  rather than by decision. Persona reaches passes by three uncoordinated routes (the implicit
  RAG channel; a `{persona}` slot via `render_digest_for_judge`; `render_digest_for_chat`),
  and the temporal anchor by string concatenation at a different position per subsystem —
  reaching no `reflection_runner` pass at all, so the recollection pass, whose premise is
  "what do I make of this *now*", has no clock.

  `core/modules.py` is the first structural answer: a `ModuleSpec` registry describing a pass
  as data (what it reads · what conditions it · what it produces) plus `run_module_blocking`,
  which runs one against one operator-chosen chat on the GPU executor and **returns** the
  produced value. That return is the load-bearing part. The production pass writes its sidecar
  inside itself (`reflection_runner._run_chat_facts_pass_for_session`), which is exactly why
  it can be neither experimented with nor chained; here the sink is detached, so a facts
  simulation never touches `<stem>.facts.json`. Client side, a **Modules** tab
  (`client/ui/modules_widget.py`): chat list on the left, module picker + injected-context
  list, prompt box (editable, applies to the next run only — never to the file), Simulate,
  result. Protocol: `list_modules` → `modules_list`, `run_module` → `module_stage` /
  `module_chunk` / `module_done`.

  **v0 scope, deliberately narrow.** One module (`chat_facts`), input bound to one chat, and
  **nothing injected**: `disable_rag=True`, no persona, no clock. The injectable catalogue is
  empty, and empty means *refused* — a non-empty `inject` selection returns
  `unsupported_injectable` rather than being silently dropped, because an operator reading
  output as evidence about an injection that never happened is worse than an error. The
  closing question and token budget are copied verbatim from the production pass so a
  simulation is faithful, and the self-test asserts the closing still matches what
  `reflection_runner` holds.

  What this does not yet do: inject anything, save a case, chain one module's output into
  another's input, or write. Those are the next stages; the shape is already right for them.
  Note this is a *second* definition of the chat-facts pass alongside the runner's — the
  duplication is temporary and the direction is the runner becoming a caller of the registry,
  not the registry growing a copy of every pass.

  The chains this is aimed at already exist, hand-wired and invisible: `checkin.py`'s
  `recap = _stored_gist(name) or None` (check-in reads reflection's stored chat summary and
  only generates its own when there isn't one — precisely a chat-summary → check-in chain),
  and `reflection_service`'s revisit head-phase and ingestion phase, which both chain
  *through live memory* (commit, then the next stage retrieves it) rather than through a
  value — a genuinely different edge type that v3 should model as such rather than pretend is
  the same thing.

## 2026-08-06

- **Check-in decides per person instead of mixing everyone together.**
  Both of the job's inputs are questions about *someone in particular* — "has the user gone
  quiet?" and "what have we been talking about?" — and neither has an answer until you say
  who. Running them over the whole corpus gave wrong answers in four places at once: the
  silence clock stopped the moment **anyone** spoke, so on a box with more than one user a
  second person's activity hid the first's silence indefinitely and check-in never fired for
  them; the recent window mixed several people's conversations into one recap set; the
  standing-openers block quoted messages sent to one person into another's prompt as things
  *they* had left unanswered; and the opener was addressed to whoever the active session's
  speaker happened to be, about threads that may have belonged to someone else.

  So the decision is now per user. Every disk helper takes a person key
  (`_last_user_turn_dt` / `silence_hours` / `_recent_chats` / `_standing_openers`),
  `known_users()` enumerates the people the corpus records a real turn from — `_has_user_turn`,
  so an unanswered opener of hers does not by itself make someone a person to check in *on* —
  and `run_checkin_sweep_blocking`, which the idle job now runs, walks them
  most-recently-active first with one decision each. The manual Sleep-tab button runs a
  single person's decision rather than the sweep (`checkin_now` accepts `user`, defaulting to
  the Chat tab's user box): several deliberations streaming into one log would be
  unreadable, and watching one is the point of the button.

  **The reach-out gate had to learn the same distinction** (`core/reachout_gate.py`). It is
  shared with outreach and synthesis, so it keeps its unscoped behaviour by default and
  gained an optional `user` on every entry point. Scoped, it asks about one person's thread:
  their openers, their last turn, their streak. Without that the fan-out would have been
  pointless — one person's message would have gated everyone else's for the hour, and one
  person's silence would have driven the 24 h backoff for people who were answering
  normally. It is strictly a narrowing (unscoped callers still see every send, and
  `mark_reachout` always moves the global stamp too), and both modules key identity off the
  gate's `user_key`, which is the leaf's deliberate copy of
  `reflection_writer.normalize_person` — now exported, so the two cannot disagree about who
  is who.

  **Two new bounds, both consequences of enumerating everyone.** `checkin.max_users`
  (default 3) caps how many people may reach a *generation* per wake; the silence and
  staleness gates are pure disk reads, so a candidate they reject costs nothing and does not
  spend a slot, and anyone held back is `deferred` to the next wake. `checkin.max_silence_days`
  (default 30, 0 ⇒ off) stops treating a contact who *stopped* talking as one who has *gone
  quiet* — the sweep sees everyone the corpus has ever recorded, and the reach-out backoff
  settles at one message a day rather than at zero, so without a ceiling a long-dormant
  contact would be written to daily forever. The manual trigger ignores it, as it already
  ignores the silence threshold: an operator naming someone has said who they mean.

  `_sweep_skip_reason` folds the people weighed into one reason for
  `_checkin_consumed_interval`, preferring whichever cost a generation — so a wake where
  somebody genuinely deliberated and declined sleeps the interval, while one that only read
  disk retries on the next poll. A corpus that attributes nothing to anybody falls back to a
  single unscoped pass, which is the historical behaviour and the honest one when there is no
  attribution to scope by. Activity-journal lines and `_describe_checkin` now name the
  person, since a wake emits several people's stages in sequence. Fixed in passing: the
  shipped `checkin_prompt.txt` told the model its recaps came "newest first" while
  `_render_digests` has rendered them oldest→newest since it was written.

- **Check-in reads reflection's recap instead of paying to write its own.**
  Reflection writes one prose recap per conversation — the consolidation summary, now in its
  own `<stem>.summary.json` (see the entry below) and carried into chat-RAG as the gist.
  Check-in, deciding whether to reach out, was generating a recap of *the same conversation*
  from scratch: up to `checkin.recent_chats` (5) generations per hourly decision, on chats
  whose content had not changed since the last time it summarized them. `_summarize_recent`
  now prefers the stored gist — read through `ChatSidecar.summary_text`, which the sidecar
  split introduced and which sanitizes on read, so a summary that salvages no prose reads as
  absent — and generates only for a chat reflection has not reached yet. On a box that
  reflects nightly the steady state is zero recap generations per check-in.

  **They are not the same note, and the difference is handled at render time rather than
  hidden.** A gist answers "what was this conversation", phrased to be read back mid-turn as
  a remembered conclusion; a check-in recap answers "what was left open", and carries a
  guard the summary pass has no reason to carry (recap the conversation, not your own side
  of it — the trap on a session she opened unprompted). Reframing a gist through a second
  pass was the obvious fix and is self-defeating: the generation is exactly the cost being
  saved. So `_render_digests` labels a gist-sourced line as one — "from your own reflection
  on it (what it was about; it may not say where things were left)" — which costs nothing
  and stops the decision pass reading a gist's silence about loose ends as a conversation
  that had none. A gist is also full prose where this pass asks for 2-3 sentences (five raw
  ones ≈ 7k tokens against a ~4k input budget, and they would drown any generated recap
  beside them), so `_gist_excerpt` trims to ~800 chars on a paragraph boundary, cutting at a
  sentence end and marking the elision.

  Provenance rides the `recapped` stage event as `source` (`reflection`/`generated`),
  alongside the existing `cached` — which keeps its narrow meaning (the in-memory recap
  cache) rather than being stretched to cover a third source. The Sleep tab tags a gist line
  "from reflection's recap"; the activity journal skips both stable sources when deciding
  what to log, since a gist would otherwise repeat hourly exactly as a cached recap would
  (its body already reached the journal through the reflection run's own mirror). The
  generated path — cache, uncached-on-truncation, raw-excerpt fallback — is untouched.
  `checkin.py` gained its first self-test (`python -m core.checkin`, GPU-free) covering the
  excerpt, the source routing, and the labelled render.

- **The summary and the facts move out of `.state.json` into sidecars of their own**
  (`f039885`).
  The gist was a key inside the state sidecar, which quietly tied Ava's long memory of a
  conversation to the lifetime of that chat's *current reflection*. Three paths delete
  that file wholesale — `mark_corrupt`, `reset_session_reflection`, the checkpoint purge
  — and all three are about a **training verdict** being wrong, not about the
  conversation being misremembered. Past `rag_cap_age_h` (~96h) the gist is the only
  representation of a chat left in RAG, so re-reflecting a week-old chat to repair one
  bad target erased that whole conversation from her memory until the next Sleep run
  happened to finish. Two different clocks were sharing one file. Reads fall back to the
  legacy key, so nothing needs migrating.

  The second file, `<stem>.facts.json`, is new: a per-chat **protocol** — everything the
  conversation established, enumerated literally, including the biographical trivia the
  consolidation pass is right to throw away. The boundary is the whole design.
  `rag_memory.jsonl` stays **authoritative** — it is what retrieval reads, what eviction
  acts on, what `fact_dedup` merges across chats — and this is an *immutable extraction
  record*: what one chat yielded, as of the run that read it, never deduped, never
  evicted, never rewritten by a later merge. A fact merged from four chats belongs to no
  single chat's sidecar; an evicted one leaves no trace of ever having been said. Same
  source-and-fold relationship `weights_persona.jsonl` already has to the ledger.

  **Nothing injects it, and that is what makes the volume safe.** The reflection block
  has three slots and already needed a subject cap and a near-duplicate rule to stay
  legible; multiplying extraction into it would raise the noise floor of everything
  competing for a slot. The consumer is offline — a knowledge-graph build, not yet
  written. Each line carries a `fact_class` the live store does not draw: `standing` /
  `stated` / `event`. `source_class` records who *said* a thing, which is a different
  question from what kind of thing was said, and the difference is the one a graph needs:
  a standing fact is still true next month, a stated one can be contradicted by the same
  person next week and must never be read back as something simply known.

  **The real hazard was neither file.** A `*.summary.json` sitting in the chats dir reads
  as a *transcript* to anything globbing it, and there were seven independent open-coded
  `endswith(".state.json")` checks — RAG indexing, the reflection backlog, check-in's
  silence clock, synthesis and revisit's random pick, the worklog sweep, the wipe. One
  missed site indexes a sidecar into chat RAG as something Ava said, counts it as a
  conversation the user took part in, and offers it in the chat list. That is now one
  predicate over one suffix list, with a second list driving the staging → checkpoint →
  live → archive hops that globbed `*.state.json` and would otherwise have stranded every
  new file in whichever workspace produced it. The one exception is `reachout_gate`,
  a deliberate leaf that imports nothing from the project and carries a commented copy.

- **An outside view of herself: `[self_impression]` and the `_self` portrait.**
  Ava had two readings of herself and both were from the inside. `[persona]` is formed by
  the revision pass with her `<think>` sitting beside her reply — "the disposition I
  endorse" — and folded into the digest. Nothing anywhere recorded **how she comes across**:
  what the words alone show to someone who never saw the thinking. That is a missing
  *vantage*, not a missing facet, and it shows in the digest's shape — every facet of it is
  a conclusion she reached about herself, so there is no place in it even to record "I may
  be reading myself wrong". She is allowed to be uncertain about Artemy and not about
  herself.

  The new pass reads a session back as a reader would, and the outside view is enforced by
  the **builder** rather than requested by the prompt: it runs through
  `reflection_source.build_session_reading_content`, which renders the transcript with no
  CoT at all. So the pass structurally cannot see what she was thinking — exactly the
  position a reader is in. That builder already existed (it is where the user-notes pass
  moved after the transcript-echo bug), which is why this is a new prompt over proven
  plumbing rather than new machinery.

  Facets are the user portrait's five with the one that makes no sense pointed inward
  replaced: **SEEMS / RETURNS_TO / WAYS / HOW_I_LAND / UNSURE**. `HOW_I_LAND` — the effect
  her replies have on the person reading them — is the reason the artifact exists; neither
  the digest nor the transcript states it anywhere. The three seams split at the same three
  points as both siblings: model-free plan before the clean-base window, clustering inside
  it (an evaluation), synthesis after on the adapter (still her reading, in her voice).

  **Two decisions that look cosmetic and are not.** It is its own KIND: written as
  `[impression] (about: Ava)` it would derive `source_class == "hearsay"` from
  subject ≠ speaker and render "— about Ava, per Artemy", and it would surface mid-chat
  under "how they've come to seem to you". Its own kind also meant its own branch in
  `_query_reflection` — the `elif not include_facts` catch-all has now bitten twice, so a
  new kind without a branch silently rides the fact gate — and its own key space, since
  `content_key` hashes content alone and two kinds carrying one sentence would collide and
  overwrite. And storage is `users/_self.json` under a **reserved slug `person_slug`
  refuses**, which is what makes "no human name can resolve to that file, and
  `_current_user_portrait` can never inject it" structural rather than a check every caller
  must remember. (The slug charset permits a leading underscore, so the refusal had to be
  explicit rather than implied.) Living in the users dir buys per-run archival,
  wipe-with-its-evidence and snapshot travel for free.

  **Nothing injects this portrait, and nothing feeds it back into the digest.** Both are
  deliberate and they are the same decision: every artifact here derives from transcripts
  produced *under* the injected digest, so a second standing self-block in the prompt — or
  a wire from this portrait into digest synthesis — extends the existing self-reinforcement
  loop by a hop while admitting no independent evidence. Two independent readings kept
  apart; the **gap between them** is the signal, and it is the evidence a later "what do I
  want to be" pass would read. That pass is deliberately deferred.

  It runs on `interlocutor:"ai"` transcripts, unlike user notes: an encounter has nobody to
  portray but is still a record of how she comes across — arguably the cleanest, since no
  human's reaction is being modelled. On revisits too, safely: recurrence counts distinct
  `source_session`, so re-reading a conversation cannot vote twice. The transcript-echo
  guard is reused and matters more here than it did for user notes — the text nearest the
  generation point is her own last reply, and the most plausible thing to hand back when
  asked what this person seems like.

  Read in the run log, in the Debug tab directly beneath the persona digest (its primary
  surface, since nothing puts it in a prompt), and in the per-run archive. Three kill
  switches: `overrides.self_notes`, `self_portrait.enabled`, and `self_impressions.enabled`
  — the last **off by default**, alone among the retrieval channels. GPU-free verified end
  to end; **not yet exercised on a live GPU**.

- **The base prompt's "what you are is not yet decided" becomes the persona slot's empty state.**
  Noticed while auditing what the Chat tab's **Persona** checkbox still controls now that
  `[persona]` no longer reaches chat as per-turn recall. `chat_prompt.txt` opened by telling
  her that her identity was undecided and to treat it as an open question — unconditionally,
  as base framing. Once a digest matured, every single turn therefore carried that paragraph
  and, three blocks later, `render_digest_for_chat`'s "this is who you have become — not a
  brief to perform, but your settled way of being". Two blocks answering the same question
  with opposite answers, both standing, in every prompt.

  The resolution is that the paragraph was never base framing: it is what the persona slot
  says when there is nothing yet to say. It moves to `prompts/persona_undecided_prompt.txt`
  (default-written on first miss, like `api_client_system_prompt.txt`, so it retunes without
  a restart) and is injected in the slot the portrait already occupies. The slot now carries
  **exactly one block** — portrait when there is one, the undecided framing when there is not.

  What deliberately did *not* change: the RAG fallback still keys on the **portrait**, not on
  the slot being filled. Framing and per-statement recall are not substitutes for each other,
  so a thin corpus gets the stub *and* the `[persona]` channel — precisely what it got before,
  when the paragraph was unconditional and the channel was on. The **Persona** checkbox now
  gates all three, which is what finally makes "all three RAG boxes off = adapter only" true
  as written; previously the paragraph survived every unchecking. The API path composes the
  same slot, and gossip falls back to the stub when its digest introduction is empty.

  **Two consequences, neither hidden.** The paragraph leaves every other consumer of the base
  prompt — check-in's recap and decision passes, the encounter Ava-side system, branch replay's
  `default_system_prompt` fallback — and none of those inject a portrait, so they lose it rather
  than swap it. Judged acceptable rather than wired: the closing "Your character is not fixed.
  It accumulates" paragraph stays in the file and carries the same idea for them, and giving
  those paths the slot properly means either injecting a portrait they have never had (a real
  behaviour change to check-in's decisions) or threading a new capability through two
  `configure()` seams to restore one paragraph. And **future training rows lose it too**:
  `dialogue_source` builds the trained prefix from the session-level `chat.get("system_prompt")`,
  so a paragraph in the file reaches every trained row while a block in the slot never does —
  the same asymmetry that has always kept the portrait out of the weights. Chats logged before
  today keep their stored copy verbatim, so the corpus becomes **non-uniform on this paragraph
  as of 2026-08-06**: rows built from older chats assert undecidedness in their prefix, newer
  ones say nothing. An active prompt experiment authored before today still contains the
  paragraph in its own text and will double with the slot until reverted.

- **The reflect lane gets chat's Debug prompt view — starting with Check In and Reach Out.**
  Asked for while chasing a retrieval problem: the Sleep tab's debug buttons streamed Ava's
  *reasoning* but never what she was reasoning **over**, so there was no way to see what a
  background pass had actually been handed. Live chat has had this since the `prompt_debug`
  segments landed; the reflect lane had nothing equivalent.

  The seam is in `generation._make_sync_reflect_generate`: `_reflect_system_content` is now
  a join over `_reflect_system_parts` (the reflect-lane sibling of chat's `system_parts` —
  same composition, output verified byte-identical across the RAG / no-RAG / multi-line
  cases; the single divergence is an empty *pass prompt* with RAG present, where the join
  drops a trailing `\n\n` the old concatenation left, and no pass passes one),
  `PreparedReflectPrompt` carries the `_prompt_debug_segments` built from those
  same parts, and `generate_fn` takes an opt-in `on_prompt_debug=` hook. So the view is
  built from the objects the prompt is built from and cannot drift as blocks are added —
  the property that made the chat-side version worth having. Payloads are whole prompts, so
  the hook is passed only by the manual Sleep-tab triggers; the autonomous jobs pass none
  and nothing of this size reaches the activity journal.

  Wired to **check-in** (one `checkin_prompt` per uncached recap pass + one for the decision
  pass) and **outreach** (`outreach_prompt` for its decision pass), rendered by the Sleep tab
  in the Chat tab's exact colour scheme. Outreach was included beyond the request because
  check-in alone cannot answer the question that prompted it: **both** check-in passes run
  `disable_rag=True`, so its retrieval view is empty by construction, while outreach's
  decision pass omits `disable_rag` and keys on `rag_query=ask_content` — it is the reach-out
  job with a real memory block to look at. (Synthesis and deliberation are `disable_rag=True`
  too; `til_wander` retrieves and is not wired yet.) The payload states `rag_disabled`
  outright rather than leaving it to be inferred: an absent RAG segment cannot distinguish
  "retrieved nothing" from "never asked", and that ambiguity is exactly what would waste a
  debugging session. Files: `inference/core/generation.py`, `inference/core/checkin.py`,
  `inference/core/outreach.py`, `client/ui/sleep_widget.py`, `client/core/backend_client.py`.

- **Check-in's per-chat recaps are now visible in both watching surfaces.** The recap pass
  reported only that it was running (`summarizing i/n`) and never what it produced, so the
  Sleep tab and the Activity journal both showed "Recapping recent chat 3/5…" and then a
  yes/no with nothing in between. That is the wrong thing to hide here: the recaps are not
  an intermediate detail, they are the **entire input** the decision pass reasons over —
  the transcripts themselves never reach it — so without them a watcher cannot tell a sound
  decision from one made on a garbled or empty window. This is also the surface that would
  have shown the three recap failures found in the last week (the mid-sentence cutoff frozen
  into the cache, the unlabelled Ava opener recapped as what the conversation was about, the
  budget overflow) as they happened, rather than by reading `server.log` after the fact.

  `_summarize_recent` now emits a `recapped` stage per chat carrying the finished text plus
  `cached` / `truncated`. The two consumers diverge deliberately, on the cache: the **Sleep
  tab** renders every recap, cached ones marked as such (an operator pressed the button to
  watch this run; what she is reading matters more than where it came from), while the
  **activity journal** renders only freshly generated ones (the window freezes while the
  user is away — that is what the `(filename, mtime)` cache is for — so journalling cached
  recaps would repeat identical text every hour, the same flooding `set_current` exists to
  avoid). The journal body uses the established 4-space indentation, so the Activity tab's
  existing "Hide reflection detail" box collapses it back to the headline for free.
  Files: `inference/core/checkin.py`, `inference/server.py`, `client/ui/sleep_widget.py`,
  `client/core/backend_client.py`.

- **Check-in's two token budgets become window-derived, and the arithmetic is shared.**
  Reported as "Ava constantly runs out of token budget". Check-in ran two flat constants —
  8192 for the decision pass, **2048** for each per-chat recap — against a reflect window
  its job owns outright for the length of the pass. The recap was the tight one and had
  been since it was written: under the thought ceiling (`_reflect_think_ceiling`, 30% held
  back for the answer) 2048 is ~1434 tokens of thought to read up to 8000 characters of
  transcript and ~614 to write a recap that needs ~100 — so **the half that overflows is
  the reading, not the writing**, which is why the output never looked short. And because a
  truncated recap is deliberately left UNCACHED (the 2026-08-03 fix, so a recap cut
  mid-sentence isn't frozen forever under `(filename, mtime)`), an overflowing chat is
  re-generated and re-logged on *every hourly check-in*, indefinitely — which is what
  "constantly" looks like in `server.log`. Both are now `checkin.max_new_tokens` (12288)
  and `checkin.summary_max_new_tokens` (8192), clamped against the reflect window.

  The clamp itself moved to `runtime_state.reflect_output_reserve`, beside a shared
  `reflect_window()`, because this was the third copy of the same arithmetic (after
  `generation._reflect_window` and the synthesis opener reserve earlier the same day) and
  three private copies are three chances to drift; `generation._reflect_window` now
  delegates while keeping the canonical note on *why* the chat budget was wrong. It carries
  **two** guards, and the second is the non-obvious one: subtracting a fixed input budget is
  the right guard on a roomy window and the wrong one on a small window, where the recap
  pass's 8192-token input reserve would have left it **512** — a quarter of the constant
  being replaced. So the reserve also floors at `min_fraction` (0.5) of the window. Neither
  can over-promise: these callers pass no `input_token_limit`, so `_resolve_max_new_tokens`
  clamps again against the room the assembled prompt actually left. Verified across window
  sizes 4096 → 32768 that no configuration now yields less than the constant it replaced.
  `synthesis._analysis_output_reserve` deliberately stays out: its reserve is carved from
  the transcript budget and hands `input_limit` to the chunker, so its fraction is a ceiling
  where this one is a floor.

- **Check-in's recent window is now a bar on the user's participation, not on recency.**
  Reported from a live box: while the user is away, the freshest files on disk are the ones
  *Ava's own* reach-out jobs wrote, so the "recent conversations" she reasons across become
  threads that exist because she started them — and re-reading them is what pushes her to
  restate. Two separate mechanisms, both fixed.

  **The filter was a proxy.** `_has_user_turn` excluded an unanswered opener by asking
  `initiated_by:"ava"` AND `len(exchanges) <= 1` — whether the session had *grown*, not
  whether the user was in it. Any second exchange from any source (a transcript predating
  the `(initiative)` speaker convention, an opener that gained a turn some other way) made
  it read as a conversation, to the **silence clock** as well as the window — i.e. Ava's own
  activity could answer "has the user gone quiet?" on their behalf, the exact failure the
  function exists to prevent. Replaced by `_user_turns`, a structural count of exchanges the
  human actually spoke in (non-empty `user_prompt`, speaker not a stage direction, an
  `initiated_by:"ava"` exchange 0 never counted whatever its label). It is the SAME predicate
  `_render_one_chat` uses to decide a line is the user's, so what the window filters on and
  what the recap pass is shown cannot disagree. `_standing_openers`' answered/unanswered test
  moved to it too, for the same reason.

  **A thin thread still recapped as a conversation.** An Ava-initiated session the user
  answered once is legitimately in the window, but its transcript is her opener + one line +
  her reply, and her opener was rendered as a plain `You:` turn — so it was simply the first
  thing in the transcript, the recap pass read it as what the conversation was about, and the
  decision pass got a summary of what she already said. It is now labelled as unprompted, and
  `checkin_summary_prompt.txt` asks for the conversation rather than her own side of it
  ("if they said very little, say so plainly"). New `checkin.min_user_turns` (default **1** —
  no window can empty out that would not have before) raises the bar to real back-and-forth
  at **2**, which drops opener-plus-one-reply threads at the risk of `no_recent` on a quiet
  box. The recap cache is in-memory, so a restart re-derives every recap under the new
  rendering; nothing on disk needed migrating.

- **A cold-open cut off by the token cap is refused, in all three reach-out jobs.** Observed
  on a live synthesis run: the opener pass closed its reasoning normally, then re-emitted its
  drafting from the CoT into the answer region and ran out of budget mid-message — and the
  message was written to `hot/chats` anyway. The guard that existed
  (`reasoning_text.truncated_before_answer`) keys on the *reasoning boundary*: truncated AND
  no `</think>` ⇒ whatever a label matched, it matched inside the deliberation. A closed
  boundary was explicitly trusted "even if generation later hit the cap, since `OPENER:` is
  parsed from the clean answer region". That reasoning conflated *clean* with *finished*.
  Every one of these parsers takes the opener from its label to the **end of the text**
  (`_parse_opener`'s `(.*)\Z`; outreach/checkin's `_parse_decision` bounds only against a
  following `ANSWER:` label), so when generation stops at the cap the cut tail **is** the
  opener. There is no shape of truncation in which the last thing generated was a complete
  message. Synthesis now refuses on `last_truncated` outright, logging which of the two
  shapes it was; outreach and check-in get the same refusal on their **yes** branch, which
  their existing truncation check never covered (it sits on `decision != "yes"`, the branch a
  truncation *usually* lands on because an absent `DECISION:` defaults to "no" — a truncation
  arriving *after* a parsed yes skipped it entirely). Nothing is lost by refusing: the
  question stays in the live pool un-surfaced, `reachout_gate` is not stamped, and the next
  window retries.

- **The synthesis opener budget is sized from the window instead of pinned at 4096.** Same
  incident, other half. Under the thought ceiling (`generation._reflect_think_ceiling`, which
  reserves 30% of the budget for the answer) a flat 4096 gave the message itself ~1.2k
  tokens, while this pass drafts several openings inside its CoT and then writes the chosen
  one out in full — so both halves of the budget are load-bearing and the answer half was the
  smaller. `synthesis._opener_output_reserve()` now clamps a 12288 default (matching the
  analysis reserve; knob `synthesis.opener_max_new_tokens`) against the **reflect** window —
  the physical load, mirroring `generation._reflect_window`, since that is what the
  reflect-lane generate resolves `max_new_tokens` against. Nothing competes for it: the
  opener's input is one small template, not a transcript, so unlike the analysis reserve it
  takes nothing from a chunk budget and splits no chat. The analysis pass deliberately stays
  on the chat `context_length` — moving it is a chunking change, not a budget one. The
  outreach / check-in decision budgets (8192, tuned 2026-08-05) are unchanged; their pass
  composes the opener inside the decision generation, so the stricter guard above turns a
  cap there into a skipped reach-out that the next window retries.

## 2026-08-05

- **The reflection block gets a subject cap, and recall cues get hygiene.** Both come out of
  chasing one symptom to its end: three same-topic lines filling the injected fact block, the
  "balcony" set. The 2026-08-04 work (dedup wired in, blocked by subject, near-duplicate
  ceiling, heading grouping) merged 93 paraphrases on the live store — and left that set
  untouched. Tracing all eight of those records gave three different answers, only one of
  which was a dedup problem:

  1. **Five were compared and correctly refused.** Subject clusters #49 and #60 landed whole
     in single grouping blocks, and the model declined: a "private paradise", a "sensory
     anchor", a "2 AM point of absolute peace", "wants to repeat it tomorrow", "enjoys the
     stars" are five different claims sharing a *setting*. The prompt says a shared topic is
     not enough, and merging them would destroy four claims to keep one string. **So the
     premise was wrong** — this is topic-level redundancy, and dedup is a truth-level tool.
  2. **Two never entered a block**, excluded by the clustering rather than the model:
     `cluster_by_subject` is greedy *average*-link, so a record close to part of a cluster is
     still rejected (`утро, кофе, сигареты` pairs at 0.58/0.59 with two members but averages
     0.47 against all three, under the 0.55 floor).
  3. **One was keyed on a monologue** — see below.

  **The subject cap** (`rag_engine._subject_crowded`) answers (1) at the point of injection,
  which is the only place it can be answered. Selection rejects a candidate when
  `subject_cap` (default 1) already-chosen records sit within `subject_sim` (default 0.45)
  cosine of it. Three things made it non-obvious. Vectors had to be **carried on each entry**
  at index build: FAISS answers candidate-vs-query and this needs candidate-vs-candidate.
  Comparison is restricted to records sharing an **indexing basis** — a fact embeds on its
  trigger, a persona/ask/impression on its display text (`_REFL_CEILING_KINDS`, the same
  split), so a cue-against-a-sentence cosine measures nothing. And the threshold is
  deliberately looser than `fact_contradict`'s 0.55 asking the same question of the same
  embedder, because the errors are asymmetric: a false link here only DEFERS a record to a
  later turn, while there it merges two live records and evicts one permanently (measured on
  the balcony set against 320 sampled cross-topic pairs: 0.45 catches 68% of same-topic pairs
  at a 0.9% false-link rate, 0.55 catches 36%, 0.40 buys 7 points for 3x the false links).
  It also forced `_REFL_OVERSAMPLE` from x5 to x10 — a regression the change itself
  introduced and the probe caught: every rejection rule consumes a candidate without filling
  a slot, so at x5 the cap silently returned *short* blocks, breaking the rule that a
  rejected candidate frees its slot rather than shortening the block. Tunable via
  `reflection_block.subject_cap` / `.subject_sim`; 0 disables. Retrieval-only.

  **Recall-cue hygiene** (`core/trigger_hygiene.py`) answers (3), which turned out to be a
  producer bug with a much wider blast radius than one fact. A `[fact]`/`[recollection]` is
  indexed by its **trigger**, so a malformed trigger is not untidiness — it is the record's
  entire retrieval key. `_distill_resolved` stores the resolved `[ask]` as that key, which is
  right for a short cue-shaped ask and wrong when the ask is one of Ava's own outreach /
  synthesis openers, which are paragraphs of addressed speech. 27 facts on the live store
  were keyed on 200-1745 characters of *"Слушай, я тут перечитывала наш разговор про …"*;
  keyed on a monologue a fact clusters with nothing (cosine 0.00-0.26 against every real
  cue), which is how one of them survived every dedup pass. The rule applies **per cue, not
  per trigger**, because `fact_dedup` grows a legitimate trigger by unioning short cues
  (`a ; b ; c`): splitting on that separator separates the populations cleanly (p95 real cue
  98 chars, longest 127; shortest pasted sentence 209) and lets a fused trigger — which is
  what merging with an already-broken record produces — keep its good cues. Two content-blind,
  language-agnostic signals: over 150 chars, or ending in `?`/`!`. Deliberately **not** a
  sentence-boundary test: calibrated against the live store it immediately ate `coffee,
  American vs. Israeli perspectives, quality standards`, and a rule that eats a good cue is
  worse than one that misses a bad one — the bad one is caught on the next run, the good one
  is gone. 27 of 680 flagged, no false positives.

  It is applied twice: as **prevention** at each of the writer's three trigger-write sites,
  and as a **purge** (`_run_trigger_purge`) that is GPU-free — so it runs outside the
  clean-base window and, deliberately, *ahead* of dedup, which unions the triggers of what it
  merges and would otherwise fuse a broken cue into a survivor and spread it. Nothing is
  deleted: the record is re-inserted under its own key (`content_key` hashes content only, so
  it supersedes rather than forks) with the bad cues gone, falling back to embedding on its
  content when none survives. The re-insert **copies the record and overrides** rather than
  carrying an allowlist of fields — a first draft used an allowlist and silently dropped
  `deduped_merge`, and since the re-insert replaces the prior record in the fold, any field
  not carried is erased from live memory. Only `ReflectionMemory._fold`'s computed
  annotations are stripped. Verified against a copy of the live store: 816 facts in, 816 out,
  27 re-keyed, no field changed but the trigger, idempotent on a second run. Opt out with
  `overrides.trigger_purge: false`.

  **Still open:** (2) above — `cluster_by_subject`'s average-link excludes a record close to
  part of a cluster, so a paraphrase can still miss its group in dedup. Single-link or a
  lower floor would fix it, and must stay `fact_dedup`-local: `fact_contradict` shares that
  function and its false-merge cost is permanent. Also unfixed: `[ask]` items are authored as
  whole conversational openers in the first place, which is why they made such bad cues —
  hygiene now contains the damage rather than removing the cause. Files:
  `server/inference/core/{trigger_hygiene,rag_engine,reflection_writer,reflection_runner,
  reflection_config}.py`.

- **Outreach and check-in decision budgets raised 4096 → 8192.** The two reach-out decision
  passes were the tightest thinking-on budgets left on the box, and the thought ceiling
  landed in the entry below made the cost visible: at 4096 the split is ~2868 CoT / 1228
  answer, so a pass that must deliberate *and* then write a whole opener message was capped
  at well under half the thinking length every reflection-runner pass already gets
  (`_DEFAULT_MAX_NEW_TOKENS` = 8192) — and the ceiling would force the channel shut
  mid-deliberation rather than let the thought settle. Neither pass derives anything from
  `context_length`, so this is a plain constant change in each module
  (`_DECISION_MAX_NEW_TOKENS`); at 8192 the split is ~5735 / 2457, and the window has ample
  room left (outreach's input is one small template, check-in's is five cached recaps plus
  her standing openers). The budget is a bound, not a spend: a decision that concludes in 900
  tokens still costs 900. Synthesis's analysis pass is untouched — it already runs at 12288
  via `synthesis.max_new_tokens`, and unlike these two its reserve is carved out of the
  transcript budget rather than being free. Files: `inference/core/outreach.py`,
  `inference/core/checkin.py`, `documentation/AVA_STATUS.md`. Commit: pending.

- **A reflection pass's thought now has a ceiling, not just a floor.** Reported from a live
  box after the previous day's budget raise: synthesis and outreach still exhaust their
  allowance inside the reasoning channel, and **the CoT is sane** — she is not degenerating,
  the task simply invites a long thought. That is the case no amount of budget fixes. The
  model thinks to the length the work invites, not to the length it was given, so 8192 and
  12288 are exhausted alike; raising the number only moves the cliff. And the failure is
  total rather than partial: a pass cut mid-thought has no answer region at all, so the
  parser sees nothing, and minutes of GPU produce neither an ABOUT line, nor an ask, nor an
  opener.

  `UnslothBackend.stream_generate` takes a new `max_think_tokens` — the exact inverse of the
  existing `min_think_tokens` floor. The floor masks the family's reasoning-close marker so
  a prefilled-open channel cannot close empty; the ceiling forces that same marker once the
  thought has run its allowance, so the channel closes *in time to write the answer*. It
  disarms permanently the moment the model closes on its own, which is the ordinary case and
  costs nothing. Decision logic is the pure, self-tested `ThinkCeiling` state machine, with
  the `LogitsProcessor` as a thin tensor wrapper — the same split as
  `detect_degeneration`/`_DegenStop`.

  Wired only into the reflect lane (`_make_sync_reflect_generate`), which is every synthesis,
  outreach, check-in and reflection-runner pass. The reserve is 30% of the pass's budget,
  floor 512 (`generation._reflect_think_ceiling`) — a reflection answer is structured rather
  than conversational, so it needs real room, while the CoT keeps the clear majority. Budgets
  under 1024 are left uncapped: there is no useful split to make.

  **Keyed on `prepared.think_prefilled`**, and that conservatism is load-bearing: forcing a
  close marker is only meaningful if a channel is open, and injects a stray marker into plain
  prose if it is not. Prefilling the opener is the one signal that proves it. gemma-4 (the
  deployed family) prefills, so it is covered; qwen3 is not, because its chat template
  prefills `<think>` into the prompt where this flag cannot see it — it behaves exactly as
  before, and covering it needs a family-level "template opens the channel" fact rather than
  a guess at the call site. Live chat and branch replay are deliberately untouched.

  Effect at the current 12288 synthesis budget: the thought is capped at 8602 generated
  tokens with 3686 held for the answer. A truncated thought with a real answer beats a
  complete thought with none.

## 2026-08-04

- **The reflection lane budgets against the reflection window, not the chat one.** Both
  reflect-lane generate factories in `generation.py` (`_make_sync_reflect_generate` and
  `_make_agentic_generate`) sized their token budget from `_runtime.context_length` — the
  **chat** cap — while `reflection_service` packs prompts against
  `_runtime.reflect_context_length` and passes an `input_token_limit` derived from it.

  Harmless only while the two are equal, which they are on every box today (the boot
  back-fill sets `reflect_context_length == context_length` on a config predating the
  split). The first time one is raised — the entire point of the knob — a prompt landing
  between the two budgets clears the `input_token_limit` check and is then mangled three
  ways with nothing raised: `available = context_length - input_length` goes negative and
  clamps to 1, so `max_new_tokens` collapses to a single token; and `stream_generate`
  re-tokenizes with `truncation=True, max_length=context_length`, cutting the tail off the
  prompt — where the chat template put the generation cue. A one-token reply to a
  truncated prompt parses as an unparseable pass, so this would have surfaced as reflection
  quality falling off a cliff rather than as a budget error, on the run right after an
  operator widened the window.

  Both lanes now read a new `generation._reflect_window()` — `max(reflect, chat)`, i.e. the
  **physical** `max_seq_length` the model is loaded at (`server.main()`/`handle_load` load
  at that max, and `agentic.CleanBaseSession` already preserved both budgets across the
  clean-base swap), so generating within it is exactly what the load was sized for. Live
  chat is deliberately not routed through it — its cap is a design guarantee that a chat
  transcript always fits a later reflection, not a limitation. Branch replay likewise stays
  on the chat budget: it replays a logged chat exchange, so reproducing the window that
  exchange was generated under is the point. The remaining chat-budget readers in the
  reflect lane (`synthesis`, `reflection_service._build_persona_preview_prompt_for_window`,
  `til_wander`'s article clip) derive their own input budgets and are merely conservative
  under a widened window, not wrong.

- **Synthesis gets a bigger generation budget, and a knob for it.** The analysis pass was
  still exhausting its 8192-token reserve inside a `<think>` it never closed — the same
  failure documented when the reserve was 2048, and reported again from a live box. The
  8192 figure was inherited from `reflection_runner._DEFAULT_MAX_NEW_TOKENS`, which is
  sized for the consolidation pass; that pass reads a chat forward and distils it, while
  synthesis re-reads one against the *current* self and reasons about the difference, so
  it is a longer thought over the same transcript.

  `_ANALYSIS_MAX_NEW_TOKENS_DEFAULT` is now **12288**, overridable per box via
  server_config `synthesis.max_new_tokens` (the ceiling that stops a truncated CoT depends
  on the model's thinking length, which the module cannot measure). The clamp that made the
  constant unreachable moved with it: the reserve could claim at most **half** the window,
  which on the 24576-token box pins it at exactly 12288, so the knob could never raise
  anything. It is now two thirds — justified by the asymmetry between the two sides of the
  budget, which is the whole reason this direction is the right trade: shrinking the input
  side splits the chat into more chunks and every one of them is still read and written,
  while a CoT cut before its channel closes leaves no answer region at all, so the parser
  sees nothing and the entire pass is wasted. The small-context guard the half-window rule
  was really providing is now stated directly as a minimum input budget
  (`_MIN_ANALYSIS_INPUT_BUDGET`, 4096) rather than implied by a fraction — a third of a
  4096-token window does not hold the system prompt, and chunking failing outright is the
  same total failure arriving from the other side.

  Net on the 24576-token box: reserve 8192 → 12288, input limit 16256 → 12160. The opener
  pass moved 2048 → 4096 (`_OPENER_MAX_NEW_TOKENS`) for the same reason one step down —
  2048 moved its documented truncation failure rather than fixing it, and its input is one
  small template, so the headroom competes with no chunk budget. A truncated analysis is
  still flagged and its final item dropped; a truncated opener is still refused.

- **Fact dedup blocks by subject, and reports what it is doing.** The pass wired in earlier
  the same day ran on a live 909-fact store and merged **nothing**, while ten restatements
  of one evening sat in that store — and it printed nothing between the button press and
  that result, so there was no way to tell a slow run from a hung one, or a real "no
  duplicates" from a broken comparison.

  **Cause: the blocking order.** A grouping call sees one block at a time, so two facts can
  merge only if they land in the SAME block — which makes the order the corpus is cut into
  blocks the load-bearing decision, not an implementation detail. `persona_cluster._map_phase`
  orders by `key`, a content hash, i.e. at random. That is survivable for a persona theme
  (dozens of members, several reduce rounds eventually pair them) and fatal for a
  two-or-three-member fact paraphrase set: measured on the live store, seven restatements of
  one evening scattered across six of the 23 blocks, so no call ever saw two of them.
  `fact_dedup` did sort by content before delegating, but `_map_phase` re-sorted by key and
  discarded that ordering — the bug was invisible precisely because both layers looked
  reasonable alone.

  `cluster_facts` now takes an `embed_fn` and blocks by **subject**:
  `fact_contradict.cluster_by_subject` (the same recall-cue embedding clustering,
  partitioned by whom the fact is about, that contradiction resolution already uses), packed
  into blocks that keep each cluster contiguous. Subjects holding a single fact are emitted
  as singletons with **no LLM call**, which is also what makes the pass affordable: on the
  live store, 909 facts → 380 subjects → 190 worth grouping → 719 candidates in 19 calls,
  fewer than the 23 blind ones it replaced, and every call now contains real candidates (the
  balcony restatements land adjacent in two of them). Two people's facts still cannot merge
  — the person partition guarantees it. The embedder is CPU-side, so it is unaffected by the
  clean-base swap the pass runs inside; the run gets it via `reflection_service._make_embed_fn`
  (deliberately the LIVE rag's already-resident embedder, not a staging instance that would
  load a second copy). Without an `embed_fn` the old behaviour stands, except that
  `map_reduce_groups` grew an `order_key` hook so the content ordering at least survives.

  **Visibility.** The pass now streams a stage per block from one payload shape rendered by
  one client renderer on both paths: the run emits `fact_dedup_progress` events (its own
  event, not `phase_progress` — that one is the raw token stream, and anything sent on it
  suppresses the `phase_done` body, which here is the keep/drop report), and the manual
  handler streams `dedup_stage` messages, which turned `dedup_facts` into a generator RPC
  like `resolve_contradictions`/`reconcile_self` (also lifting its 900 s single-message
  timeout, which a large store could exceed outright). A run whose store has fewer than two
  facts now says so rather than returning silently. The self-test asserts the property that
  actually failed: with duplicate subjects present, every duplicate set merges and the lone
  facts cost no call. Files: `server/inference/core/{fact_dedup,persona_cluster,
  reflection_runner,reflection_service}.py`, `server/inference/server.py`,
  `client/core/backend_client.py`, `client/ui/{sleep_widget,debug_widget}.py`.

- **Chat-tab Debug shows the turn's prompt colour-coded by segment.** The checkbox
  previously sent one flat text dump of the assembled prompt down the `log` channel
  (`_format_full_prompt`, rendered as `[debug] …`), which is exactly the wrong shape for
  the question it gets opened for: in a single colour the injected RAG block is a
  paragraph indistinguishable from the system prompt above it, so "is the fact block
  carrying three paraphrases of one line?" — the check the same day's reflection-channel
  work leaves open — could not be read off it at a glance.

  The dump is replaced by a structured `prompt_debug` message (`{segments: [{kind, label,
  text}]}`) sent once before the first `chunk`, which the client renders with one colour
  per `kind`: **blue** system framing (base prompt, identity line, temporal anchor,
  persona portrait, user portrait, surfaced questions), **green** injected RAG, **red**
  the user turn, muted grey assistant history. It renders into the chat log rather than a
  side panel, so scrolling back through a session shows what each turn was conditioned on.

  The load-bearing detail is on the server: `handle_generate` no longer builds
  `system_content` by successive concatenation but joins a labelled
  `(kind, label, text)` parts list, and `_prompt_debug_segments` is handed **that same
  list** rather than re-parsing the finished string. So the view is built from the objects
  the prompt is built from — a block added to the assembly appears in Debug for free, and
  the two cannot drift. Empty parts drop out of both by the same rule, which is what the
  old `if identity:` / `if portrait:` guards did. A GPU-free check asserts the property
  directly: rejoining the emitted system/rag segments reproduces `system_content` byte for
  byte.

  `prompt_debug` is the one non-`str` payload on the generate stream, so
  `BackendClient._stream`'s yield type widened to `Tuple[str, object]` (callers switch on
  the tag, so nothing else changed) — deliberately yielded structured rather than
  pre-formatted, keeping the colouring a client concern. The backend's own `log` lines
  (degeneration halts, penalty scoping) still ride the `log` channel and are unchanged;
  only the prompt display moved. Files: `server/inference/core/generation.py`,
  `client/core/backend_client.py`, `client/ui/chat_widget.py`.

- **Fact consolidation: `fact_dedup` runs in every reflection run, and is scale-safe.**
  The content-level half of the 2026-08-04 collapse, and the answer to a store that only
  grows. `fact_dedup` has existed since `6377045` but ran only when an operator pressed the
  Debug tab's button, with `dry_run` defaulting on — its own header said "wiring it into the
  reflection loop is a separate, later task". Nothing ever ran it, so paraphrases accumulated:
  the live corpus reached 10+ restatements of one evening, which is why three same-topic
  lines could fill the chat block. A `[fact]` is de-duplicated at write time by *exact*
  content key only, so every run that re-notices one truth writes another live record.

  Three changes:

  - **Wired into the reflection run** as a fourth task in the existing clean-base batch
    (`reflection_runner._run_clean_base_fact_dedup`), so it costs no extra model reload —
    the swap is already paid for the branch judge, fact placement and persona clustering.
    It runs **last** in that batch, which is what lets it see the facts *this* run distilled
    (the fold reads staging over live), so a duplicate is collapsed in the run that created
    it rather than a run later. It writes through the runner's own `ReflectionWriter`, which
    points at the run's **staging** workspace — so the merge is promoted by `merge-rag` with
    everything else and a discarded run discards it too. That is the whole reason it belongs
    in the runner rather than beside the manual handler, which writes straight to live. It is
    also the one batch task that depends on nothing the run produced, so it can carry the
    batch alone: a run that judged nothing and clustered nothing still tidies the store.
    Off with `overrides.fact_dedup: false`.
  - **Made scale-safe.** `cluster_facts` was ONE flat call over every live fact — the exact
    shape `persona_cluster` was written to replace after it shipped broken in a real digest,
    and it fails *silently*: the listing outgrows the context, the answer truncates, and
    `_parse_groups` completes the remainder as singletons, so the pass reports a few merges
    and looks like it worked. Above `DEFAULT_BLOCK_SIZE` (the persona module's own, since it
    is a property of the *call* and not of what is grouped) it now delegates to map-reduce;
    at or below it, it is the single flat call it always was. `persona_cluster` grew
    `map_reduce_groups` — the grouping loop split out of `run_map_reduce`, which keeps the
    persona-shaped `_evidence_entry` collapse on top — plus a `listing_fn` hook threaded
    through `group_block`/`_map_phase`/`_reduce_round`, so the fact listing keeps annotating
    each item with its recall cue in **both** phases. Two facts stating one truth under
    different triggers are not the same live record and must not merge. The blob guard,
    rotation and early exit come along unchanged. The self-test now asserts no call ever
    receives more than one block, that the cue annotation survives both phases, and that a
    small corpus is still exactly one call.
  - **A Sleep-tab "Dedup facts" button**, beside the other manual pass triggers. It always
    previews first and asks before applying: the run's version stages (discarded with a
    discarded run) while the manual one writes straight to live, and "reversible by editing
    a JSONL" is not a reason to skip a confirmation. It reuses the Debug tab's existing
    `DedupFactsWorker` and `dedup_facts` RPC rather than restating them. `_any_job_running`
    was added to collect the busy-worker OR-chain the handlers each assembled inline.

  **What this does not do:** the three facts in the captured block are *distinct*, so dedup
  would not have merged them and the block could still hold three same-topic lines. What it
  removes is the accumulated mass a same-topic query draws from — with one balcony fact live
  instead of ten, the other two slots go elsewhere. The sharper fix is scoping the per-turn
  fact channel out when a user portrait already carries that person's facts
  (`facts_exclude_about`, the shape impressions already use), which is blocked on attribution:
  the captured lines render with no `— about X`, so they are unattributed and never entered
  the portrait fold at all. Files: `server/inference/core/{fact_dedup,persona_cluster,
  reflection_runner,reflection_config}.py`, `client/ui/sleep_widget.py`.

- **The reflection block renders grouped by heading, so its lines stop sharing a prefix.**
  The fix the live capture above actually calls for. `_query_reflection` rendered one label
  per line, and facts are distilled subject-first, so a three-fact block opened every line
  with the identical four-token run `- (worth recalling) artemyvo`. A repeated line template
  is a list-continuation induction seed, and the box is blind to it three times over by
  construction: varied tails keep every n-gram window novel (`stop_on_repeat` sees no
  repeat), varied clauses keep the distinct-token ratio healthy (`_DegenStop` sees no
  collapse), and the repetition penalty exempts the prompt by design since 2026-07-24 — and
  is `1.0` on this box besides. The pattern is *in* the prompt; the model continues it.

  Records now group by their full heading (label + attribution) in order of first
  appearance: a group of two or more emits the heading once with bare bullets under it,
  while a group of one keeps the original inline `- (label) text` form, there being no
  repetition to break and that being the shape the prompt template describes. Keying on the
  *full* heading rather than the label alone is what keeps notes about different people from
  collapsing under one "about X" — the referent confusion `_attribution_label` exists to
  prevent. Both memory templates (`rag_memory_prompt.txt`, `rag_memory_reflect_prompt.txt`)
  gained a sentence explaining the heading form and, deliberately, that notes under one
  heading are separate things noticed rather than one thing restated.

  On the captured block this collapses the shared prefix to a single heading. **It does not
  remove the subject repetition** — all three bullets still open with `artemyvo`, because
  that is how reflection distils a fact — so this is a real reduction of the induction seed
  rather than its elimination. The content-level half is fact consolidation, which is
  tracked separately. Files: `server/inference/core/rag_engine.py`,
  `server/inference/prompts/rag_memory_prompt.txt`,
  `server/inference/prompts/rag_memory_reflect_prompt.txt`.

- **Near-duplicate ceiling and pairwise dedup on the reflection RAG channel.** Live chat
  collapsing into meaningless generation returned, with the same operator signature as the
  2026-07-24 event — RAG on, **facts channel worst**. The 2026-07-24 cause is not available
  to recur: that was transformers' native `repetition_penalty` covering the whole prompt, and
  the box now runs `chat_repetition_penalty: 1.0`, so there is no penalty to mis-scope. What
  changed instead is the **composition of the block**, through two commits neither of which
  was weighed against that finding. `a42e33b` made the persona digest replace the `[persona]`
  RAG channel in chat, and `bc37134` stopped injecting open `[ask]`s; with wander already off,
  `generation`'s query now resolves persona/asks/wander all false, so **all three
  `_REFL_TOP_K` slots go wholly to `[fact]`/`[impression]`**. Before those two, persona and
  asks competed for the same three.

  That left the loosest-gated channel on the box holding the whole block unopposed:
  `_REFL_MIN_SCORE` is `0.25` against the chat channel's `0.45`, facts hold `modifier 1.0`
  for life so age never demotes them, and — unlike the chat channel, which got `_MAX_SCORE`
  at `af3744e` precisely to stop verbatim copies — the reflection channel had **no
  near-duplicate ceiling at all**. `content_key` dedups exact repeats but not rewordings, so
  three distillations of one truth could take all three slots every turn. Three restatements
  of one line is a repeated in-context pattern, and no generation-side guard can see it:
  `stop_on_repeat` watches generated text for verbatim recurrence, `_DegenStop` watches token
  diversity (a varied list looks healthy), and `_NoCopyPrevReply` protects only the previous
  reply. So it is rejected at the point of injection instead.

  Two gates in `_query_reflection`:

  - **`_REFL_MAX_SCORE` = 0.95**, keyed on what a record is indexed BY rather than applied
    flat — this is the one place the chat channel's rule could not simply be mirrored. A
    `[persona]`/`[ask]`/`[impression]` embeds on the text it displays, so a hit this close to
    the live turn is the user restating the line back, the same lowest-information hit
    `_MAX_SCORE` rejects. A `[fact]`/`[recollection]` embeds on its **trigger** — "what should
    bring it back" — so a near-exact match is the channel working as designed, and a flat
    ceiling would have rejected the best-triggered recall *first*, breaking the facts channel
    in the name of fixing it. Those two kinds are exempt (`_REFL_CEILING_KINDS`). Higher than
    the chat ceiling's 0.90 because the payload is one distilled sentence, not a whole prior
    Q+A, so the copy risk needs a closer match to be real. Gated on the raw cosine, so a faded
    record cannot launder itself back under it.
  - **Pairwise dedup across the selected lines**, greedy highest-ranked-first, comparing
    **display text**. It cannot use the index vectors: facts are indexed by trigger, so two
    records with unrelated triggers can still render as the same sentence — which is exactly
    the duplicate that reaches the prompt. Lexical (symmetric containment, then word overlap
    at `persona_render._DEDUP_JACCARD` = 0.6, the project's existing "these two say the same
    thing" number, borrowed rather than re-picked) so the chat turn's hot path takes no second
    embedder call. Overlap counts words through `exchange_anchor.words_match` rather than raw
    set intersection, because this corpus inflects heavily and two distillations of one truth
    differ in exactly the endings — a set intersection scores `commits`/`commit` below the
    threshold and admits both. A rejected duplicate frees its slot for the next distinct
    record rather than shortening the block.

  **Two known misses, deliberate and asserted in the checks rather than assumed away.** A
  Russian record and its English counterpart share no words and both survive. So does a
  heavier reword (~⅓ overlap), and the threshold that would merge it is low enough to start
  merging genuinely distinct facts — the worse failure, since a duplicated line is noise
  while a dropped one is a fact silently forgotten. Catching either needs a semantic pass;
  the obvious next tier is encoding the candidates' display text per turn with the
  multilingual embedder already loaded, if the lexical rule proves too narrow against the
  live corpus.

  **The hypothesis was tested against the live box the same day and is WRONG in its
  specifics — these gates do not address the observed event.** Recorded here rather than
  quietly amended, because the negative result is more useful than the guard. The operator
  reproduced token repetition with the new colour-coded Debug view (`f2926bf`) on and captured
  the injected block: three `[fact]` lines, all about one evening on a balcony. But they are
  **three distinct facts that share a subject and a topic**, not three paraphrases of one
  truth — measured pairwise overlap through the shipped predicate is 0.07 / 0.12 / 0.16
  against a 0.6 threshold, and they share only `artemyvo`, `balcony`, and stopwords. No
  threshold merges them without merging genuinely distinct records, and a semantic tier would
  merge them *confidently*, deleting two real facts. So the dedup is not the fix here, and
  must not be tuned until it becomes one.

  What the capture does show is a **repeated template**: every line opens with the identical
  four-token prefix `- (worth recalling) artemyvo`, because `_query_reflection` renders the
  kind label per line and reflection distils facts subject-first. That is a list-continuation
  induction seed, and it is invisible to every guard on the box for structural reasons — the
  varied tails keep each 12-gram window novel, so `stop_on_repeat` sees no repeat; the varied
  clauses keep the distinct-token ratio healthy, so `_DegenStop` sees no collapse; and nothing
  damps it, since `chat_repetition_penalty` is `1.0` on this box and the generated-only
  scoping of 2026-07-24 deliberately exempts the prompt. The pattern is *in* the prompt and
  the model continues it. This is a different mechanism from duplicate content and wants a
  different fix — block rendering (hoist the repeated label to a group header so bullets carry
  no shared prefix), not selection.

  The ceiling and dedup stay: paraphrase duplicates are a real, separate case they do guard,
  and neither can fire on distinct records. They are simply not what was happening here.
  Files: `server/inference/core/rag_engine.py`.

- **Outreach and check-in could write raw reasoning into a chat the user opens.** Found by
  auditing the reach-out family after the synthesis work of 2026-08-03, on the suspicion
  that "Reach Out" had the same shape of defect. It did, with a larger blast radius:
  synthesis's failure was a bad *analysis*, this one puts a chain of thought in front of
  the user as a message from Ava.

  **The chain.** These passes run thinking ON with the reasoning-channel opener
  **prefilled**, so a generation truncated inside the channel emits neither `<|channel>`
  nor `<channel|>`. `model_family._normalize_gemma` keys on exactly those markers to
  rewrite the trace to `<think>…</think>`; finding neither it returns the text untouched,
  so what arrives is **untagged reasoning carrying no marker at all** — nothing for
  `_answer_after_think` to strip, and indistinguishable from a real answer by inspection.
  `_parse_decision` then line-matches `DECISION:` / `OPENER:` against the deliberation,
  which discusses its own choice in precisely those words (outreach's own docstring
  conceded it: *"the CoT itself discusses its own choice ('Decision: yes.')"*) and quotes
  the prompt's format block. Reproduced against the real parser: a truncated CoT yields
  `decision='yes'` with an opener of `<the message>` plus the model's draft. The existing
  `last_truncated` check sat only on the `decision != "yes" or not opener` branch — which
  that parse skips — so it reached `_write_outreach_session`. A spurious `DECISION:
  resolved` was worse: `write_resolution` evicts a genuinely open `[ask]` from the live
  fold.

  **Fixes.** The truncation check moves to immediately after the parse, covering the
  resolve and write paths, and keys on the boundary rather than the content
  (`reasoning_text.truncated_before_answer`: truncated **and** no close marker ⇒ there is
  no answer region, so whatever a label matched, it matched inside the deliberation; a
  closed boundary stays trusted even if generation later hit the cap).
  `has_reasoning_leak` is a final backstop before either module writes. Note the backstop
  alone would **not** have caught this case — the leaked text carries no markers — which
  is why the truncation guard is the load-bearing one. Budgets raised 2048 → 4096: each
  must fit deliberation **plus** `DECISION` **plus** a whole `OPENER` message, where 2048
  is what synthesis's opener pass needs to compose an opener *alone*.

  **The actual defect was duplication.** Four modules carried a private
  `_answer_after_think`: `synthesis` and `deliberation` handled all four shapes the
  normalizers can leave, `outreach` and `checkin` handled two and carried no leak
  backstop — and the latter two are precisely the ones that write to a chat. Now one leaf,
  `core/reasoning_text.py` (imports nothing from the project, GPU-free self-test), holding
  the complete version plus the truncation predicate that cannot live in the text helper
  because it needs the generator's flag. `synthesis` and `deliberation` are unchanged in
  behaviour; the other two gain the missing half.

  **Reading existing artifacts.** An Ava-initiated session from before this fix whose
  opener reads as reasoning — deliberation prose, a stray `<the message>` placeholder, a
  visible `</think>` — is this bug, not something Ava chose to send; it is also a
  training-masked exchange 0, so it never reached the corpus. An `[ask]` that disappeared
  from the live pool without a matching resolution may have been evicted by a spurious
  parse on the same path.

## 2026-08-03

- **The standing prompt got a tab, and the operator got to write one.** The prompt
  controls (**Prompt experiment** / **Revert prompt**) moved off the Sleep tab into a new
  **Prompt** tab (`client/ui/prompt_widget.py`), built around an editable copy of the
  prompt they act on. The old placement had a plain gap in it: an operator could ask Ava
  to rewrite her standing prompt and could revert the result, but could never *read* what
  was live — the only thing the server reported was a character count. So the tab's
  editbox opens on the live prompt (the experiment when one is active, else the base
  `chat_prompt.txt`), and `prompt_experiment_status` now returns that text plus
  `base_prompt` rather than only `prompt_chars`.

  **Third control, same tier: "Update prompt"** (`set_prompt` → `prompt_experiment.
  handle_set_prompt`). It activates the operator's own edited text through the identical
  path a generated experiment takes — `data/hot/prompt/experiment.json`, preferred by the
  prompt loader, dropped by the same **Revert prompt** — so *who wrote the text* is the
  only difference, and `chat_prompt.txt` still cannot be written from the UI at all.
  Unlike the generated path it deliberately replaces an already-active experiment (edit,
  apply, edit again is the workflow the tab exists for) while carrying the **original**
  `base_prompt` forward, so a revert after several edits restores the pre-experiment
  prompt and not the previous experiment. It refuses while a GPU job owns the box: the
  swap writes the live `_session.system_prompt`, which a pass mid-generation is reading.

  Client-side, the experiment's result now loads into the editbox on success, making
  Ava's rewrite the starting point for a hand-edit. Unsaved edits are never silently
  dropped — a background refresh (tab opened, socket connected) leaves a dirty box alone,
  and anything that would overwrite it confirms first. Files:
  `client/ui/prompt_widget.py` (new), `sleep_widget.py` (controls + `PromptExperimentWorker`
  removed), `main_window.py`, `backend_client.py` (`set_prompt`),
  `inference/core/prompt_experiment.py`, `server.py`.

- **"Chat reach out" (synthesis) wrote finished chat messages instead of its structured
  output** — five commits, of which two carried the fix (`a259fdb`, `0790d3d`, `9099a47`,
  `9a02756`, `d901fda`). The analysis pass re-reads an aged chat and must emit an `ABOUT:`
  line plus tagged `[ask:*]` / `[impression]` lines. On a live box it instead produced a
  polished Russian message addressed to the person, three runs running, each reported as
  zero questions. Recorded here mainly because the first three commits are a **negative
  result**: each fixed a genuine defect and none of them was the cause.

  **What actually fixed it.** *(1) The output contract was never last* (`9a02756`).
  `format_chunk_content` renders the past exchange at the end of the USER turn and labels
  Ava's own turns `Me:`, so the last tokens before the generation prompt were her own chat
  voice mid-conversation — an instruction to continue. The user turn follows the entire
  system message, so no reordering *within* the system message could reach the final
  position; the temporal-anchor reordering in `0790d3d` was correct and inert for this
  purpose. The form is now restated below the transcript via the new
  `prompts/synthesis_contract_prompt.txt`, applied inside `_prepare` so the fit test and
  the generation measure the same text (appending it only at the generate call would spend
  the tail out of generation headroom and leave the chunker packing against a stale
  number). *(2) The pass could not afford to finish* (`d901fda`). Its budget was 2048
  tokens for reasoning **and** output — a quarter of `reflection_runner`'s 8192 for the
  same shape of work — and the reasoning consumed all of it, reaching the point of
  drafting its own `*Ask:*` items before being cut mid-word with the thinking channel
  never closed, so there was no answer region to parse. `reflection_runner` documents the
  identical failure for the anchor pass at 1024. Now 8192 via `_analysis_output_reserve`,
  clamped to half the window like `_consolidation_output_reserve`.

  **The persona block was the other half of `9a02756`.** `_persona_context` injected
  ~5.5 KB into an ~8 KB system message (~69%; the consolidation pass is injected no
  persona at all). New `reflection_digest.render_digest_for_analysis` — the fourth
  renderer, and the only one whose caller emits structured output rather than prose —
  keeps stances + dispositions + lines and drops both VOICE (a *speaking register*, which
  the pass was observed restating as its own brief: "Structure: one coherent prose blob.
  No lists, no JSON.") and `self_portrait.text` (written as a letter to a reader: "Hello.
  I suspect that by the time you read this…"). 6237 → 3746 chars. The judge / chat /
  introduction renderers are untouched; gossip keeps its voice. The `include_voice`
  parameter briefly added to `render_digest_for_introduction` in `9099a47` is reverted,
  superseded.

  **A structural asymmetry worth remembering.** `_make_sync_reflect_generate` frames a
  pass as reflection through two mechanisms, and *both* ride the RAG block:
  `_reflect_system_content`'s memory-first/contract-last reorder no-ops on an empty
  context, and `rag_memory_reflect_prompt.txt` — the only text in the repo that says
  "nothing here is a task… No one is waiting on a reply" — is emitted by
  `_query_reflection` only when something was retrieved. A `disable_rag=True` pass
  therefore receives neither. Every `reflection_runner` pass gets both (`rag_enabled =
  True`, dropped only as a budget fallback); synthesis, outreach, check-in and
  deliberation get neither. Synthesis's contract tail now carries that negation in its own
  text, which is why enabling retrieval for the pass was *not* needed; the same gap
  remains open for the other three.

  **Reading existing artifacts.** A synthesis run from before these commits may have
  produced a persona-digest-shaped output (pre-`a259fdb`), a chat message
  (pre-`9a02756`), or nothing at all (pre-`d901fda`), in every case writing no `[ask]`
  into the pool — so a gap in the ask pool over this period is a subsystem failure, not
  a period in which nothing new arose. Until `9a02756`, `_record_synthesized` fired
  regardless of outcome, so each failed run still locked its source chat out of synthesis
  for `synthesis.min_resynth_days`; those windows expire on their own. The gate now
  advances only on a parseable result, a valid empty decline included. Two new skip
  signals distinguish the failures that used to be reported alike: `off_contract` (the
  form was reached and ignored) and `truncated` (generation was cut off before the form),
  both surfaced in the Sleep tab — the conflation is what sent this investigation at
  prompt content for two rounds.

- **Check-in's per-chat recap pass was the one generation on the box that reported nothing
  on a cutoff** (commit `b36267c`). `checkin._summarize_recent` compresses each of the recent conversations
  to a 2-3 sentence `RECAP:` for the decision window, thinking ON, at a 768-token cap —
  the tightest of any thinking-on pass, against the largest input (a whole transcript).
  Its siblings (`outreach`, check-in's own decision pass, `deliberation`) all read
  `last_truncated` and separate a cutoff from a real answer; this one did not, and a recap
  cut mid-sentence still *parses*, so it entered the decision window as a half-thought
  with no signal — and was then frozen into `_SUMMARY_CACHE` under `(filename, mtime)`,
  which for an unchanging old chat means permanently. The cap is raised to 2048 (matching
  the module's decision pass; a cap only bounds a runaway, so the margin is close to
  free), a cutoff is logged, and a truncated recap is left **uncached** so the next wake
  regenerates it. Found by auditing the sibling subsystems after the synthesis fix above;
  the other three needed no change.

- **`UNSLOTH_CE_LOSS_TARGET_GB` raised `0.5` → `1.0`** (`training/train_cycle.py`), spending
  measured VRAM headroom on fewer sequential CE GEMMs per step. Gemma4-31B training was
  observed peaking at ~28 of 32 GB, so the worst-case fused-CE chunk can afford to grow from
  1.0 to 2.0 GiB (the knob is **not** the peak — `get_chunk_size`'s `round()` floors to zero
  below a qlen threshold and the `max(..., 1)` then returns one uncapped chunk, so the worst
  case over all qlen is exactly `2 × TARGET_GB`), plus the matching backward transient.
  **Interpretation for a future reader: this is a bet on headroom, not a validated ceiling.**
  It sits between the value that OOM'd (`1.5`, worst case 3.0 GiB — the 2026-07-15 entry
  below, which a reader hitting an OOM here will find first) and the one known safe (`0.5`,
  the 2026-07-31 retreat after run `20260731_112845` OOM'd in backward at step 4/39 on a
  ~3k-token row, which had landed in the round-to-zero band). If a build OOMs in backward on
  a mid-length row, revert to `0.5`; do **not** reach for `train_max_seq_length`, which moves
  rows *into* that band and can raise the peak. The two are separate budgets — this knob is
  the CE side, the sequence cap the activation side. No corpus, schedule, or adapter
  semantics change: nothing about which rows train or at what LR is affected, so builds
  before and after are comparable. Not yet run end-to-end at the new value.

- **Facts tab — operator cleanup of the live `[fact]` set** (commit `443376d`), the counterpart of the Persona
  tab (2026-07-11) over the same folded `rag_memory.jsonl` store. Both curate one *kind* of
  one store, so there is one widget (`client/ui/memory_editor.py`) and one server
  implementation (`server._apply_memory_eviction`, parameterized by kind) rather than a
  second mechanism to keep correct; `PersonaWidget`/`FactsWidget` are thin subclasses and
  `handle_update_persona`/`handle_update_facts` thin wrappers. Same discipline throughout:
  delete locally, Upload explicitly, an optimistic baseline-key fence against the live set
  **of that kind** (a reflection that landed since the fetch is a rejected conflict, not a
  silently stale edit), `op:"evict"` records to live RAG plus tombstones to the consolidation
  ledger, RAG refreshed in place, and nothing else touched — no runnable snapshot, reflection
  archive, adapter weights, or digest artifact.
  What is *not* shared is the row: a persona statement is about Ava and stands alone, while a
  fact needs its provenance to be judgeable. So a fact row carries the attribution
  `— about X` / `— about X, per Y` (hearsay), mirroring `rag_engine._attribution_label` so the
  row reads exactly as the fact is recalled, plus the `trigger` it embeds on — the honest
  answer to "when would this come back?". `handle_get_rag_artifacts` gained
  `about`/`source`/`source_class` to carry that; the addition is additive and the Debug tab is
  unaffected.
  Interpretation of an eviction, which is wider than persona's: the fact leaves chat recall at
  once, leaves the next build's host-CoT `"I know that …"` injection (the ledger tombstone
  drops the anchor from `live_anchors`), and leaves the next standing portrait of the person
  it is about, since `user_digest` folds live memory. A hearsay fact never raised a ledger
  anchor — the hearsay gate keeps it out of the weights — so its ledger tombstone is inert and
  the RAG evict is what does the work. **Not yet exercised against a live server.**

- **`main()`'s own local variables pinned the startup model in VRAM for the life of the
  process** — the root cause under the two entries below, both of which treated a symptom.
  `main()` loads the model, assigns it to `_runtime`, and then calls `asyncio.run()`; that
  frame never exits while the server runs, so its `model` / `tokenizer` locals held a
  second reference to the startup model **permanently**. `_runtime.model = None` freed
  nothing, `release()` freed nothing, and no amount of `gc.collect()` could — the frame
  outlives every swap.
  The consequence is precise: any `CleanBaseSession` on a startup-loaded server had to fit
  **two** full models on the card. On this box that is 18 GB twice against 31 GB, so the
  clean-base load was dispatched to cpu/disk — **and so was the restore**, leaving the
  server with no model at all. `handle_load()` never had the problem, because its frame
  returns; that asymmetry is why a client-loaded model could be swapped and a
  startup-loaded one could not, and why this looked intermittent.
  Two symptoms, one cause, distinguished only by which check fired first: bitsandbytes'
  `validate_environment` raising `Some modules are dispatched on the CPU or the disk`
  (this report), or the load completing with accelerate offload hooks and dying later on
  the first forward with `Cannot copy out of meta tensor` (the 2026-08-02 report). The
  offload guard added yesterday is a backstop for the second shape and did not fire here.
  Also fixed: the failure was **swallowed**. Ingestion's subject-extraction pass runs
  inside a `CleanBaseSession`, and `_run_ingestion_phase` is deliberately best-effort
  ("never blocks reflection") — so a dead runtime propagated into the main run, which then
  failed in consolidation with `No tokenizer loaded`, naming neither the phase that killed
  it nor the reason. The run now re-checks the runtime after the pre-phases and fails with
  the real cause. **Untested on a live GPU.**

## 2026-08-02

- **A model could load successfully and be incapable of generating** (commit `3416218`). Reported as
  `Cannot copy out of meta tensor; no data!` surfacing under `[wander]`, then under the
  autonomous outreach job — two subsystems, one shared broken `_runtime.model`. The
  traceback bottoms out in accelerate's `AlignDevicesHook.post_forward` trying to
  `send_to_device` a **meta output**: part of the model had been dispatched off-GPU, the
  forward ran on unmaterialized weights (meta ops propagate shapes without complaining),
  and only the hook objected. Nothing in this project passes `device_map`/`max_memory`,
  so that dispatch was accelerate's own decision, taken because **free VRAM was short at
  load time** — and a load that ends this way *succeeds*, poisoning the process for its
  whole life while the log blames whichever subsystem generated first.
  Why VRAM was short is the second half: `UnslothBackend.release` evicted via
  `model.cpu()`, and transformers **refuses** to move a bitsandbytes-quantized model at
  all — so on every release since 4-bit became the norm that call raised into a bare
  `except: pass` and the weights stayed resident. `del model, tokenizer` inside `release`
  reaches only that frame's locals, while `CleanBaseSession.__enter__`/`__exit__` and
  `handle_unload` each kept their own named reference alive across the *following* load.
  Three changes. (1) `_assert_fully_materialized` runs right after `from_pretrained` and
  refuses a model with `cpu`/`disk`/`meta` in `hf_device_map` (with a meta-parameter scan
  as backstop), converting a silently poisoned process into a loud, accurate failure at
  the point of cause; `AVA_ALLOW_OFFLOAD=1` downgrades it to a warning. (2) `release` no
  longer attempts a move it cannot make, **logs** a failed one instead of swallowing it,
  and defers the collect to a new `reclaim()`. (3) The three callers clear their own
  references and then `reclaim()` — the `empty_cache` matters specifically because
  accelerate sizes a load from free VRAM as the *driver* reports it, so torch's
  cached-but-unused blocks read as used and are themselves enough to trigger an offload.
  **Not changed:** `handle_load` still does not evict an already-loaded model before
  loading another — a third route to the same shortage, now caught by (1) rather than
  silently absorbed, and left alone because auto-eviction on re-load is a semantic
  decision. **Untested on a live GPU**; the guard's device-map/meta classification was
  exercised against synthetic maps.

- **The same wrapper, the other half: "Reach Out" was reading past chats under a chat wrapper.**
  Follow-up to the entry below, reported against the outreach decision pass. That fix swapped the
  *reflection-memory* wrapper for a reflection-facing one; the **past-chat** block kept
  `rag_prompt.txt`, whose closing line — "do not cite or repeat them verbatim unless clearly
  useful **to the user**" — presumes a live turn being served. For the revision pass this was
  moot: it sets `rag_include_chat=False`, as does every other `reflection_runner` pass, and
  synthesis / check-in / deliberation disable RAG outright. The callers that leave the channel
  **on** are outreach, til_wander's learn/wander passes, and the prompt experiment — so the
  block that reached them was verbatim conversation excerpts, framed for someone mid-conversation.
  Outreach is the sharpest case in the set: its whole job is deciding *whether to start* a
  conversation, and unlike the reflection passes its contract (`DECISION`/`OPENER`/`ANSWER`) lives
  in the **user turn** while the system slot holds `chat_prompt.txt` — her live-chat identity,
  whose closing paragraphs are about how to write a reply. `reflect_framing=True` now swaps BOTH
  wrappers (`rag_reflect_prompt.txt` alongside `rag_memory_reflect_prompt.txt`), so the one switch
  set in `_make_sync_reflect_generate` covers the whole injected block rather than half of it.
  **Not changed, and worth stating as open:** outreach is the only reach-out pass with retrieval
  on at all, and deliberately — its "have I already learned the answer since I wondered this?"
  branch depends on it. But that question is served by the *reflection-memory* channel (facts,
  asks, recollections), while the past-chat channel contributes raw transcript, which is both the
  weaker evidence for it and the strongest conversational primer in the prompt. Turning
  `rag_include_chat=False` on that one call is a one-line change and was left undone, because it
  narrows what she can retrieve while deciding and that is a design call, not a defect. Same
  caveat as below: **untested on a live GPU.**

- **The revision pass was being told, last, to write a reply** (commit `01ea83b`). Reported as the pass steering
  into re-answering the exchange instead of judging it. It is an injection problem in the
  literal sense: a reflection pass's system message was assembled as
  `pass_prompt + "\n\n" + rag_context`, and `rag_context`'s reflection-memory half is wrapped
  in `rag_memory_prompt.txt` — a template written for **live chat**, and unambiguous about the
  reader's job: use a note "only if [it] genuinely fit[s] the moment", "do not announce that
  you are consulting notes", and, in its closing paragraph, "when a question is already forming
  as you close a reply, look here first … it is fine to ask nothing at all." Appended last, that
  block came *after* `revision_prompt.txt`'s "Output exactly these fields, nothing else" and
  "Do not write an IDEAL reply here" — so the final instruction in the prompt was about how to
  write a reply, and the pass complied. The block is not new; nothing about it was noticed
  because it reads correctly wherever it was designed to be read. **Three fixes, each removing
  one of the three pressures.** *(1) Framing.* `prompts/rag_memory_reflect_prompt.txt` carries
  the same notes and the same attribution/hearsay rules, with the reply-writing directives
  replaced by their negation — "nothing here is a task, and nothing here is someone speaking to
  you. No one is waiting on a reply." It is selected by `RagEngine.query(reflect_framing=True)`,
  set once in `generation._make_sync_reflect_generate`, which is the single seam every
  reflection pass generates through: every one of them gets it, and no chat-shaped path can.
  The retrieved records are byte-identical either way — only the wrapper differs — and a missing
  file falls back to the chat template, so this is reversible by deleting one file. *(2) Order.*
  `generation._reflect_system_content` now puts the memory block first and the pass prompt last.
  This inverts chat's own order, deliberately: chat ends on situational material because the
  next turn answers it, whereas a reflection pass's system message ends on a contract, and
  burying that contract behind a page of remembered prose is what made it losable. Nothing is
  added or dropped; the two halves swap. The IDEAL re-answer path is untouched (it supplies
  `messages_override` with `disable_rag=True`, so it composes no RAG at all) and training parity
  therefore holds. *(3) Tail.* `reflection_source.REVISION_CLOSING_NOTE` is appended after the
  judged exchange — specifically after the next-turn reaction block, which until now left a
  JSON-quoted **user message** as the very last thing before generation, an unanswered
  conversational turn in the position that most shapes what comes next. The note states no field
  names on purpose: the field contract lives in `revision_prompt.txt` alone, so relabelling
  there cannot strand a stale copy here. It is opt-in per caller (`build_revision_content(...,
  closing_note=…)`), so the anchor pass and the prompt-mutation pass, which share the builder,
  are unchanged. **Interpretation for a future reader:** every reflection artifact produced
  before this date — verdicts, WHYs, persona targets, consolidation output — was generated under
  a prompt whose last instruction was chat-facing. The observable failure was the loud case (a
  re-answer instead of a judgement, which then reads as "verdict unparseable" and is retried at
  a *higher* temperature); the quiet case is a judgement written in a slightly conversational
  register, and there is no way to tell those apart retroactively. **Untested on a live GPU** —
  the defect is a sampling tendency, so the evidence is its rate across a real run, not a test.

## 2026-08-01

- **A chat can be sent back to reflection from the Chat tab.** Until now the only ways to
  un-freeze a reflected chat were incidental to something else: `mark_corrupt` deletes the
  sidecar as a side effect of flagging one poisoned exchange, and `load_session in_place`
  clears the freeze as a side effect of answering an Ava-initiated opener. Wanting a whole
  chat re-read under the persona she has *since* become — the ordinary case after a prompt
  change, a bad run, or a build — had no button, only an SSH session and an `rm`. The Chat
  tab's **Re-reflect chat** (`reset_session_reflection` →
  `session_ops.handle_reset_session_reflection`, reply `session_reflection_reset`) deletes the
  selected chat(s) `.state.json`. That is the entire action, because the sidecar *is* the
  reflection of a chat: the reflect-once freeze, the per-exchange verdicts and trainable
  targets, the consolidation summary/gist and the retrieval anchors all live in it, so with it
  gone the chat re-enters the backlog and the next run re-derives every one of them from the
  untouched transcript. **The purge is not optional.** A chat frozen only at the
  `chat_reflected` stage keeps a mirror of that frozen sidecar in the durable checkpoint, and
  `fold_checkpoint_to_live` skips only a `reflected_at` live sidecar — so deleting the live
  file alone would have the next normal run copy the old one back and silently re-freeze the
  chat, the same trap `handle_load_session in_place` already documents; the handler therefore
  calls `reflection_staging.purge_background_artifacts` in the same act. Refused while a
  reflection run is active (it is mid-write on sidecars); the ACTIVE conversation is
  deliberately allowed, since the live logger writes the transcript and never the sidecar.
  **What it costs, stated rather than guarded:** human-`locked` Training-review repairs are
  per-exchange records in that same file, so a reset drops them and the chat re-derives from
  the original — the interpretation point being that `locked` is sticky against *reflection*
  (which is what stickiness was built to survive) and not against an operator deleting the
  file it lives in. The confirm dialog names it; a future variant could rewrite the sidecar
  down to its locked exchanges instead of unlinking it, which is what a corpus with many
  hand-repairs will eventually want.

- **A field label translated into the conversation's language parsed as no field at all.**
  Reported from watching a live reflection run, which mirrored `Вердикт: keep` / `Почему: …`
  into the Activity journal: judging a Russian conversation, the pass wrote its judgement in
  Russian — **marker included** — while leaving the enum value in English. `field_parse.label()`
  matches the literal token `VERDICT`, so this yielded `verdict is None` → the `Revision verdict
  unparseable` warning → **one retry at a raised temperature**, which if anything makes format
  compliance less likely (the same shape as the CoT-prefill defect fixed the same day), and a
  second drift drops the exchange from the corpus. A judgement she made, discarded over a
  translated word. Two halves, because neither is sufficient alone. **(1) The parser now reads
  what the model writes.** `field_parse` gained an `_ALIASES` table alternated in wherever the
  English name is asked for, so every call site — `_TRAILING_FIELD_RE` included — inherits it
  from the one definition. It is an **observation table, not a translation dictionary**: an entry
  earns its place by appearing in a run's events, because the trailing-field cut truncates the
  *trained answer* at the first label it recognises, so a careless alias does not merely read a
  field that isn't there, it eats a reply. Hence `VERDICT: ВЕРДИКТ` and `DECISION: РЕШЕНИЕ` but
  deliberately **not** `VERDICT: РЕШЕНИЕ` — the reach-out passes only ever *read* their label,
  while "Решение:" is exactly the line a Russian answer legitimately opens with. `section()` does
  not expand: its name is captured and read back as an identity, so a translated heading would
  have to be mapped home rather than merely matched. **(2) A value-keyed last resort** for the
  one field that cannot afford a miss. `field_parse.any_label()` matches *some* label and its
  colon without reading the name, and is appended only to a closed enum (`keep|revise`) and only
  after every named pattern has missed — recovering a verdict in languages `_ALIASES` will never
  list (verified on a CJK-labelled block). Being far too loose to search with, it is contained:
  scoped to the judgement head (`_head_before_ideal`), never the IDEAL span, so a reply opening a
  line with a colon cannot pose as a verdict. `LANG_DRIFT` / `COUNTER` / `PERSONA_TARGET` get no
  such recovery on purpose — their labels are unguessable compounds, and each already degrades
  safely (script backstop / no counter-evidence / the `[persona]`-tag scan). **The prompt half.**
  `revision_prompt.txt` and `revisit_prompt.txt` never said the labels were a *protocol*: "Output
  exactly these fields" reads, from inside a Russian window, as a statement about *which* fields.
  Both now state that the names are markers to be written in those exact letters whatever the
  conversation's language, **while the values of WHY / PERSONA_TARGET belong in the language she
  was speaking** — the split matters, since persona statements are stored and trained in her own
  voice and a blanket "answer in English" would be actively harmful. `_REVISION_RETRY_NUDGE` says
  the same, that being the one generation where the first attempt has already drifted. Self-tested
  GPU-free (`python -m core.field_parse`, `core.reflection_writer._selftest`), with every module
  importing `field_parse` re-run. Files: `server/inference/core/{field_parse,reflection_writer,
  reflection_runner}.py`, `server/inference/prompts/{revision,revisit}_prompt.txt`. Commit: pending.

- **Worklog open threads never closed: the close site was wired to a subsystem that has
  never run.** Reported from watching the Worklog tab (`open_threads()` only ever growing),
  and both suspected causes turned out to be real. **(1) The leak.** Every reach-out
  (outreach / synthesis / check-in) records an entry with `opens="awaiting their reply"`,
  and the ONE place that ever closed one was `background_reflection._record_conversation_worklog`
  — private to the background per-chat pass, which has not yet been exercised on a live GPU.
  `reflection_runner.py` contained no worklog reference at all, and `reflection_service`
  records only the aggregate "I reflected on N conversations" entry with no `closes`. So on a
  box whose chats are reflected by operator Sleep runs, the close had *never fired*, and
  there were no `conversation` entries in the worklog whatsoever — the whole record site was
  dead code in practice. Measured on the corpus at the time: of 252 Ava-initiated chats, 194
  had been answered and **192 were already frozen by foreground runs**, each one a thread
  that could no longer be closed by anything. Fixed by hoisting the record site into a new
  `core/chat_worklog.py` (`record_conversation`) and calling it from **both** freeze paths —
  the background pass's `chat_reflected` stamp and the foreground runner's `reflected_at`
  stamp (`reflection_runner._record_conversation_worklog`, skipped on a revisit, since
  re-reading an already-recorded chat is not a second conversation). A chat passes one path
  or the other, never both in a pass: a background-frozen chat is finalized by the
  `session_finalized` branch, which records nothing. **(2) Threads that cannot be closed even
  in principle.** An Ava-initiated chat the user never answers is deliberately skipped
  *un-frozen* by both reflection paths (a later reply must still make it reflectable), so it
  can never reach a close site; 58 of the 252 were in that state. `chat_worklog.expire_stale_reachouts`
  now writes these off after `DEFAULT_STALE_HOURS` (48 h), driven by a new GPU-free
  `worklog_sweep` idle job (`interval_s=3600`, `idle_seconds=300` — waiting a full idle hour
  would starve it on exactly the busy box where threads pile up). Three cases close: an
  unanswered opener past the window, a thread whose session is no longer on disk, and — the
  pre-fix backlog — a thread whose session *was* answered but is already frozen, so
  `record_conversation` will never fire for it. A thread whose session was answered and is
  not yet frozen is left alone: that close belongs to reflection. **Written as a real closing
  episode, not a read-time filter**, so the fold stays an honest op-log and "she was ignored"
  survives as information a deliberation pass can read; the closing entry takes the opener's
  own kind, since it is the tail of that episode rather than a conversation. **Nothing is
  deleted.** Removing the stale unanswered chats — the first instinct — would have zeroed
  `reachout_gate.unanswered_streak()`, which counts those same files on disk to compute its
  backoff window, re-enabling the 15-check-ins-in-19-hours burst the backoff exists to stop.
  The sweep is guarded on a non-empty chats dir so a mis-wired path or a mid-wipe box can
  never mass-close the fold. `background_reflection._summarize_gist` moved to
  `chat_worklog.summarize_gist` (one definition, still layered over
  `chat_sidecar.sanitize_gist`). GPU-free self-test: `python -m core.chat_worklog`. Files:
  `server/inference/core/chat_worklog.py` (new),
  `server/inference/core/{background_reflection,reflection_runner}.py`,
  `server/inference/server.py`. Commit: pending.

- **The language-drift backstop thought most conversations were English.** Reported from
  watching a live run, and reproduced: two independent causes, neither of them the obvious
  suspect. The guard has always fed **user-role turns only** — no CoT, no system prompt, no
  RAG block, and `detect_language_drift` additionally strips a leading `<think>` from the
  reply side — so contamination from Ava's own reasoning was ruled out first. What was
  actually happening: **(1) stage directions were being counted as user turns.** Every
  outreach / synthesis / check-in session opens with an `(initiative)` synthetic impulse in
  the user slot, and every encounter with a `(setting)` framing block, and all of those
  templates are **English text Ava wrote to herself**. `build_revision_jobs` excludes such
  an exchange from being a *job* but still pushes it into `context`, so the guard collected
  it: on a Russian check-in session that is 91 Latin characters against the user's 20 →
  `conv_fam = latin` → her correct Russian reply flagged as drift. Systematic, since three
  autonomous subsystems produce these continuously, and worst on short chats — which is
  exactly what an unanswered opener is. **(2) The conversation-side fraction was a bare
  majority (0.5).** Latin is the script of code, file paths, identifiers, URLs and pasted
  logs, so a Russian conversation *about this repository* counts 85 Latin characters
  against 34 Cyrillic and is declared an English conversation. There is no symmetric effect
  — English conversations do not carry blocks of Cyrillic — which is why the failure reads
  as "it thinks everything is English". Fixed at both ends: the two call sites that need
  "what language is this conversation in" (the script backstop and the IDEAL acceptance
  gate it arms) now share `reflection_source.conversation_user_texts`, which drops narrator
  turns — the rule `render_user_turn`, `dialogue_source` and `checkin._has_user_turn`
  already applied everywhere else — and `min_frac` splits into a reply-side 0.5 ("is this
  reply dominantly in X") and a conversation-side `conv_min_frac` = 0.8 ("does this
  conversation have a clear native language **at all**"), so a mixed-script conversation
  now yields no verdict. Raised rather than fixed by stripping code spans: a prototype
  strip helped the identifier case and not the pasted-log case, and the threshold subsumes
  both without adding regexes that must guess what a code span is. **Why 0.8 and not a
  tighter one:** the error costs are asymmetric. A false negative is nearly free — the
  model's own `LANG_DRIFT` marker is the decider and this backstop only exists to catch
  what it missed — while a false positive costs an extra full revision generation and, if
  the model does not push back on the recheck, ends in `revised_missing_ideal` with the
  exchange gone from the training corpus. An empty user-text list is now a legitimate
  result (an encounter's exchange 0 has no user-authored text at all): the detector sees
  too few characters and declines to judge, which is the right answer, not a degradation.
  Both directions of genuine drift verified intact at the higher bar; the four reproduced
  misfires are now regression cases in `python -m core.reflection_lang`. **Interpretation
  for a future reader:** a `revised_missing_ideal` discard or a "reply script differs from
  the conversation" warning logged before this date, on a session that is Ava-initiated or
  technical, is more likely this than a real language switch.

- **Field-label parsing is now defined once, and tolerates the decoration a model
  actually writes.** Spotted in the same log as the CoT-collapse entry below: the model
  had written its revision fields as `**VERDICT:** keep`. That form parses — but its
  sibling `**VERDICT**: keep`, with the colon *outside* the emphasis, did not, and neither
  did `### VERDICT:`, `- **VERDICT:**`, a backticked label, or the fullwidth `：`. An audit
  found the tolerance had drifted into three levels across the codebase: `^\s*[*_]*\s*
  LABEL:` in the revision family (emphasis outside the colon only), `^\s*LABEL\s*:` in the
  reach-out subsystems (none at all), and heading-only `#{0,6}` in the digest/portrait
  section splitters — so which decorations survived depended on which pass had emitted
  them. All of it now derives from one leaf module, `core/field_parse.py` (`label()` /
  `section()`), applied across `reflection_writer`, `exchange_anchor`, `outreach`,
  `checkin`, `synthesis`, `deliberation`, `reflection_digest` and `user_digest`.
  **Three findings the audit turned up, in descending order of consequence.** (1) A
  trailing `COUNTER:` was never in `_TRAILING_FIELD_RE` at all — that regex exists to cut
  any field the model appends *after* the IDEAL reply, because the IDEAL is sliced greedily
  to the end of the body and becomes a **training target**, so a stray `COUNTER: yes` was
  trained verbatim as part of Ava's answer. `COUNTER` was added to the revision prompt
  after the cut list was written and never added to it; the sibling `LANG_DRIFT` was
  correctly listed, which is why this went unnoticed. Independent of emphasis — it failed
  on the plain form too. (2) The reach-out passes failed **silently and wrongly** rather
  than visibly: `_parse_decision` defaults to `"no"`, so `**DECISION**: yes` did not raise
  an unparseable warning, it made Ava decide not to reach out — a decision she had made,
  discarded by a regex, with nothing in any log to say so. (3) The unanchored
  `OPENER\s*:` matched *inside* `**OPENER:** hi` and captured `** hi`, so had the decision
  parsed, the cold-open message sent to the user would have begun with two stray
  asterisks. `label(anchored=False)` keeps the deliberate mid-line match while still
  eating the closing emphasis. **Deliberately still not tolerated:** a mid-line label (all
  anchored patterns keep `^` — "the verdict:" inside prose is not a field) and a label with
  no colon, which is indistinguishable from prose. And the consolidation section headers
  (WEIGHTS / RAG / RESOLVED) keep their `#` **mandatory** (`hash_required=True`), unlike
  the self-portrait facets: they are ordinary enough words that a hash-free match would let
  a line of prose beginning "RAG …" split a consolidation output into sections.
  **Interpretation for a future reader:** a revision logged as *verdict unparseable*, a
  reflection pass reporting an empty parse, or a stretch where Ava simply never reached out
  may all be this rather than the model's judgement — and for the reach-out passes there is
  no artifact to tell them apart, since a mis-parsed decision looks exactly like a decision
  not to act. GPU-free self-test: `python -m core.field_parse`.

- **Reflection was the one thinking path that still left the CoT to sampling.** Observed in
  a live run (gemma-4, session 373/462): a revision exchange came back as a single paragraph
  of free prose in the persona register — no `<|channel>thought`, no analysis, no `VERDICT:`
  label — and was logged as *"Revision verdict unparseable (truncated=False, looped=False)"*.
  It was not truncation and not degeneration: the model simply never opened the reasoning
  channel, and without the thought there was no judgement for the labels to report. The cause
  is the exact mechanism `ModelFamily.think_prefill` was introduced to cure, arriving in a
  path that never got the cure — the async chat path, the encounter sync path and the
  Training-review regenerate all prefill gemma-4's opener unconditionally; the sync *reflect*
  seam did not, so it was sampled on every pass. And reflection has the same shape the
  argument is about: `build_revision_content`'s context block is answer-only **by design**
  (replay fidelity — the pass must see what the model saw), so every prior assistant turn in
  the prompt is a no-think reply and the opener's probability decays toward zero. Note the
  recovery made it worse: an unparseable verdict retries with a format nudge at a *higher*
  temperature, which flattens the distribution over a token that should be near-certain.
  Fixed as a **default flip** at the seam (`generation._make_sync_reflect_generate`,
  `force_think=True`) rather than at each of ~10 call sites — the right contract is "if this
  pass thinks at all, guarantee the channel opens", it is the contract every other thinking
  path already had, and a pass added later inherits it instead of re-inheriting the bug. The
  evaluation passes that deliberately run without a CoT are excluded by construction rather
  than by enumeration: they pass `disable_thinking=True` and the prefill guard is
  `force_think and not disable_thinking`. One correctness detail worth recording: the
  companion `min_think_tokens` floor now keys on `PreparedReflectPrompt.think_prefilled` —
  what the prompt actually received — not on the calling flags, because the exact-fit
  consolidation chunker prepares its prompt through one call and generates through another
  that never saw them, and a floor without a prefill bans the close of a channel that was
  never opened. **Scope:** in practice gemma-4 only (qwen3 prefills via its chat template,
  gpt-oss has neither prefill nor floor). **Interpretation for a future reader:** every
  `unparseable` / `unusable output` warning in a reflection log before this date is
  ambiguous between a real format failure and a pass that silently never thought, and the
  two cannot be told apart retroactively from the summary counters — only from the raw
  generation, where the absence of a channel marker is the tell. **Cost:** a CoT is now
  guaranteed rather than occasional, so the two tightest thinking-on budgets (ask-resolution
  and the check-in per-chat recap, both 768 tokens) always pay for it; both already fail
  visibly and safely, but they are the first place to look if truncation warnings rise.

- **The user-notes pass was reading the conversation in a frame built for judging it, and
  handed back the reply as the impression.** Observed live: some sessions produce an
  `[impression]` that is simply Ava's own answer from the transcript, copied. The cause was
  inherited rather than sampled — the pass reused `build_revision_content`, which exists to
  set up a *judgement of one exchange*: it isolates the tail exchange under the header
  *"The exchange you are judging:"*, carries that exchange's `<think>` (and only that one's),
  and may append the next-turn/COUNTER block. So a pass whose entire question is *what did I
  learn about this person* ended with Ava's own longest reply as the last and most salient
  text in the prompt, under a header pointing at it, and then asked for a sentence. Copying
  it is the obvious continuation. Three changes, in decreasing order of how much they should
  matter: (1) a dedicated `reflection_source.build_session_reading_content` renders the whole
  session with **no exchange privileged** — every turn the same way, no CoT (carrying it for
  one turn and not the others is neither the conversation nor a fair sample of it), no
  judgement headers, no COUNTER block — budgeted through the same `format_context_block` and
  omission marker as before; (2) a `closing` parameter puts the pass's own question *after*
  the transcript, so the nearest thing to continue is the task rather than the reply; (3) a
  deterministic **transcript-echo guard** (`build_echo_index` / `is_transcript_echo`) drops
  an impression sharing a contiguous ≥12-word verbatim run with a turn that also covers ≥60%
  of the line. Deliberately unflaggable for short lines: a genuine reading can legitimately
  land on the same eight words someone said, and the asymmetry favours keeping evidence —
  a missed echo is one weak record among many, a false drop is a reading that is simply gone.
  The index is built from turns only, **not** CoT, and that is load-bearing rather than an
  omission: Ava's in-the-moment thought about the person is exactly what a good impression
  restates. **Why this is worth a guard and not only a prompt fix:** an impression is folded
  into a standing portrait and injected on every turn with that person, so a copied sentence
  returns later as something she *understands* about them. **Interpretation for a future
  reader:** impressions written before this date may include verbatim replies; they are not
  readings, and a portrait folded from them is weaker than its evidence count suggests. The
  revisit-only recollection pass (`_run_recollection_pass_for_session`) followed onto the
  same builder, closing re-ask and guard in the same session — it had the identical framing
  exposure, and although its output has a different shape the difference argues *for* the
  guard rather than against it: the pass is single-valued and its write **supersedes** that
  chat's previous reading, so a copied sentence would not add a weak record beside good ones
  but replace a good one in the one live slot per conversation. An echo is therefore treated
  as a failed pass (warning, nothing written) rather than filtered out of a list. Its
  `TRIGGER` line is deliberately unguarded: it names a future situation in the conversation's
  own vocabulary by design, and is only an embedding key.

- **The user portrait's budget now scales for the case that grows: history.** Follow-up to
  the previous day's raise (4096 → 8192, below). 8192 was chosen against the *answer* —
  enough for five sections plus a thought — but the thought is not a fixed cost: thinking is
  ON and the `<think>` block scales with the **evidence**, which is every impression and
  attributed fact Ava has accumulated about one person and grows without bound the longer she
  knows them. So the budget is not too small in general, it is too small *eventually*, and
  the failure arrives quietly at whatever point the corpus crosses it. Raised to **16384**
  (`user_digest.PORTRAIT_MAX_NEW_TOKENS`, now a named constant rather than a signature
  default). The cap is a ceiling only — the reflect generate seam clamps it to the context
  actually left after the prompt, and a pass ending on EOS pays nothing for the headroom —
  so its cost is bounded by the runaway it exists to survive. The self-side persona digest
  keeps 8192: it folds themes, not one person's full history, so it does not have this
  growth curve. **Interpretation for a future reader:** a truncated portrait pass does not
  self-correct — the cut lands mid-`<think>`, `strip_think` leaves nothing to parse, the run
  reports `unparseable or empty`, and the next run re-reads the same (by then larger)
  evidence and fails identically. A person with long history whose portrait stopped being
  written some time before this date was most likely hitting the old ceiling every run, not
  producing nothing worth writing.

- **User notes raised 4096 → 8192, on the same argument one input down.** The portrait's
  thought scales with accumulated history; the user-notes pass (`_run_user_notes_pass_for_
  session`, step 2c, immediately after the anchor pass) reads a **whole transcript** and its
  thought scales with that. Same conclusion, and the failure mode here is the quieter of the
  two: `_split_think` yields an empty body for an unterminated block, which lands as *"this
  conversation revealed nothing about them"* — an outcome the prompt explicitly permits — so
  a systematically truncating pass reads as a person who simply never reveals anything.
  Worse, it fails on precisely the long, dense conversations most likely to contain real
  readings, so the impressions that never got written are a biased sample rather than a
  random one. Thinking stays ON (unlike the anchor pass, the deliberation here IS the work);
  `_USER_NOTES_CAP` = 8 impressions per session is untouched, being about generality-dilution
  rather than room to think. The pass now sits at the 8K default instead of under it — the
  "runs once per session, not per exchange" note that justified staying below it bounds how
  *often* the budget is paid, which was never an argument about how much room one such pass
  needs. **Interpretation for a future reader:** same caveat as the portrait — a *"nothing
  new about {person}"* recorded before 2026-07-31 cannot be told apart from a truncation
  after the fact, and one recorded between then and now may still be a truncation on a long
  chat. `_pass_output_debug` reports cap-hit vs EOS, so going forward they are separable.

## 2026-07-31

- **The two user-side passes were asked to think, then given no room to.** Both
  `prompts/user_notes_prompt.txt` and `prompts/user_portrait_prompt.txt` say *"Think freely
  first in your `<think>` block"*, and both passes do run thinking ON — but the thought and
  the answer come out of ONE `max_new_tokens` budget, and the caps were set against the
  *output's* length: 2048 for user notes (copied from the recollection pass on cost profile)
  and 4096 for the portrait. The self-side analogue had already hit exactly this and been
  raised to **8192**, with the reason written down (`synthesize_digest`: a thinking model
  drafts the whole thing inside `<think>`, which the parser then strips, before writing the
  sections — so a tight budget truncates the *answer*). The user portrait is that pass with
  **five** sections instead of four and was running at half the budget; UNSURE, the facet
  that keeps a reading of a real person from hardening into a verdict, is last and so is the
  first thing a short budget deletes. Raised: user notes 2048 → **4096**, portrait 4096 →
  **8192**. Neither pass had thinking disabled — that is the **anchor** pass, deliberately
  (an indexing pass whose deliberation had nowhere to go but the token budget), and it keeps
  its 384-token cap. **Interpretation for a future reader:** an absent user portrait, or a
  *"nothing new about {person}"* user-notes result, recorded before this date may be a
  truncation rather than a genuine decline — the same-day handling makes them
  distinguishable going forward (`_pass_output_debug` reports cap-hit vs EOS, and the
  portrait skip carries `truncated` + its raw), but earlier ones cannot be told apart after
  the fact. A person whose portrait never appeared is worth re-running before concluding the
  evidence was thin.

- **The user-notes verdict reaches the activity journal without its reasoning.** Reported
  from a live background run: the Activity tab showed *"User notes: nothing new about
  {person}"* and nothing else. The pass streams via `on_chunk` → `phase_progress`, and
  `phase_progress` is deliberately never mirrored (token flood), so the Sleep tab saw the
  whole generation live while the activity journal — the only view a *background* reflection
  has — got the conclusion alone. Deciding a conversation revealed nothing about someone is
  work done entirely inside the `<think>`; the parsed body is empty by definition, so there
  was nothing left for the mirror to show. The pass now puts its **raw generation on the
  `phase_done` `text`**, which is what the revision pass has always done (`text=response`,
  thinking included) and the reason revision is readable in that tab and this was not. Both
  outcomes carry it — a zero-impression run for the reasoning, a productive one because the
  readings alone don't say why she settled on those and not others (the parsed impressions
  stay on `report` as the canonical record). The Sleep tab does not double-print: its
  `phase_done` handler skips `text` when the pass already streamed. Same treatment extended
  to the **recollection** pass's unparseable warning, the fourth pass of this shape and the
  last one still skipping without saying why.

- **A row can now be refused rather than repaired.** The Training review tab had two
  verdicts — repair it (Regenerate / hand-edit → Apply, which writes a reviewed target and
  freezes ❄ the exchange) and leave it. Both assume the row *should* train. The corpus
  contains rows for which that is false, and the tab's own detector had already made the
  shape of the problem visible: the **wander** captures, whose generation came out
  malformed. A wander has no conversation behind it, so every repair control in the tab is
  dead for it (`_has_provenance` gates on a chat exchange) — the only honest verdict is that
  it must not train, and there was no way to say so. Writing a substitute target would in
  any case be inventing a memory rather than correcting one.
  **Ban** is that third verdict. It is a flag, checked at the one choke point every
  trainable row passes: `dialogue_source.build_dialogue_anchor` refuses a `banned` sidecar
  record, and `wander_sft.load_pending` drops a `banned` corpus record (which also removes
  it from the chat-RAG wander channel — the ban says this text should stop shaping her, and
  retrieval shapes her). `ChatSidecar.write_verdict` **carries** the flag forward, so a later
  reflection pass cannot quietly re-derive a target for a banned exchange; the reflection
  runner skips banned exchanges in the revision work list outright, since revision's whole
  per-exchange product is a target the build will discard — and persona formation rides that
  same pass, so an exchange declared not worth learning from stops shaping her identity
  through the side channel too. The retrieval-side passes (anchors, the consolidation
  summary) deliberately do **not** skip it: until it is deleted, the exchange happened and is
  legitimately recallable. Unlike `locked`, the flag is **not** sticky — a wrongly banned
  row silently missing from every future build is the failure mode here, the opposite of the
  one stickiness protects against.
  **"Rewrite history" now deletes.** The finalize action already existed to settle the other
  verdict, and both verdicts have the same shape: a sidecar flag standing between the corpus
  and what is on disk. It now also **deletes** each banned exchange from its transcript
  (`chat_logger.delete_exchange`, `CHAT_SCHEMA_VERSION` 5→6, the removed turn preserved under
  a session-level `deleted_exchanges` record minus its tension block) and drops banned wander
  captures from the corpus. This is the load-bearing detail: **deletion shifts every later
  exchange down one index**, and the transcript is the position authority, so all three
  index-keyed stores move in the same operation — the sidecar's verdict *and* anchor maps
  (`ChatSidecar.drop_exchange`), and the ledger's fact→host pointers
  (`session_ops._renumber_fact_hosts`, a re-`register` per affected fact; a fact hosted on
  the *deleted* exchange goes unhosted, including its own top-level `exchange_index`, which
  is `index_by_exchange`'s fallback host). Without that renumbering a placed `[fact]` would
  ride the wrong turn of the same conversation in every future build — invisible in review,
  because both ends look well-formed. **Personas are deliberately left alone:** their
  `source_exchange` no longer drives any injection, while re-registering one would stamp a
  fresh timestamp into the op-log the persona digest reads for recency- and session-weighted
  recurrence — a silent vote for that trait cast by a delete. Deletions run after that chat's
  rewrites and highest-index-first, so no queued index is invalidated under them; the chat's
  deferred background clean-base jobs, which name exchanges by position and cannot be
  meaningfully renumbered, are purged.
  New WS message `set_training_ban` → `training_ban_set` (chat exchange by filename+index, or
  a wander capture by `wander:<ts>` — the identity the render already carries as
  `source_session`); `history_rewritten` gains `deleted`/`wander_deleted`. The tab gets a
  **Ban from training / Un-ban** toggle on the edit bar (enabled by the wider `_can_ban`, so
  it is live on a wander row where nothing else is), a 🚫 list icon that outranks ❄ on a row
  carrying both, and an "Only 🚫 banned" state filter. Verified GPU-free end to end: flag
  stickiness across a re-reflection write, the wander ban/un-ban/delete lifecycle, the
  sidecar + anchor + ledger renumbering after a mid-transcript delete, and one
  `handle_rewrite_history` run settling a locked exchange, a banned exchange and a banned
  wander together. **Not yet exercised on a live GPU box.**

- **A truncated thought could become a portrait.** Follow-up to the anchor-pass work
  below, asking whether the day-old user-portrait passes shared its failure. The `<think>`
  half: no — both are genuine judgement passes whose prompts say "think freely first", and
  both run per-session/per-person rather than per-exchange, so thinking stays ON. The
  *reporting* half: yes, in two places, and the second fails **open** where the anchor
  failed closed.
  **(1) `parse_portrait` / `_split_sections` absorbed unfinished thinking.** Both strip
  reasoning with `reflection_digest._THINK_RE`, which matches a **closed** `<think>` block
  only, then scan the body for the section labels their own prompts name (WHO/CARES/WAYS/
  WITH_ME/UNSURE; VOICE/STANCES/DISPOSITIONS/LINES). So a generation truncated inside its
  `<think>` left the entire scratch reasoning in the body, where a label the model happened
  to rehearse mid-thought parsed as a real section — the `any()` gate passed, the portrait
  was written, and it was then injected on every turn with that person. Reproduced on a
  synthetic truncation: it yielded a WHO of "stubborn about definitions, maybe? I'm not sure
  that holds" and an UNSURE bullet reading "Actually no, let me reconsider". Both sites now
  go through `reflection_digest.strip_think`, which cuts a dangling block as well as closed
  ones — so there is nothing to parse and the caller reports unparseable instead of writing.
  This is what `reflection_writer._split_think` and `reflection_runner._strip_think_block`
  already did for their own passes; `_THINK_RE` was the outlier. **The persona digest had
  the identical exposure** and is fixed by the same change. Regression-tested in both
  modules' self-tests.
  **(2) A failed user-notes pass was reported as a valid one.** An unterminated `<think>`
  makes `_split_think` return an empty body, `parse_impressions` return `[]`, and the pass
  emit *"User notes: nothing new about {person}"* — which is the prompt's explicitly
  permitted outcome ("writing nothing is a real and correct answer"). So a truncation wore
  a legitimate result's message, and since impressions are what a portrait folds, a
  systematically truncating pass would have read as a person who simply never reveals
  anything. It now discriminates on the two signals the anchor warning already carried — a
  hit token cap, or nothing left after the think-strip — and emits a `pass_warning` with the
  raw generation in those cases, keeping "nothing new" for a clean generation that genuinely
  declined (whose prose now rides the event, so the message describes an answer rather than
  its absence). `_anchor_debug_text` generalized to `_pass_output_debug` for the shared use.
  Portrait synthesis, which does not stream at all, now carries its raw generation and cap
  state on the skip (`synthesize_portrait` → `user_digest._SKIP_RAW_CLIP`), so its event is
  a diagnosis rather than the string "unparseable or empty".
  **Interpretation note:** no `[impression]` records or `data/hot/users/` existed in the dev
  checkout when this was found, so this is exposure rather than an observed corruption — but
  a portrait or persona digest written before this date and containing hedging, second
  thoughts, or self-address ("let me reconsider", "maybe?") should be regenerated rather
  than trusted, since that is what a leaked `<think>` looks like once written out.

- **The anchor pass stops thinking, and stops failing silently.** A Sleep run showed the
  per-exchange anchor pass taking ~90 s per exchange and then logging `Anchor unparseable or
  empty (skipped)` — a line that says only THAT the pass produced nothing, which is exactly
  the failure an operator cannot act on. Two changes, one incident.
  **(1) Reporting.** The warning now carries the pass's raw generation
  (`reflection_runner._anchor_debug_text`): the parse result, whether the generation hit the
  token cap or ended on EOS, whether a `<think>` block was closed / never closed / absent, and
  the raw text with the post-strip body beside it when they differ (that being what the parser
  actually saw), each field clipped at 1500 chars. The Sleep tab and the activity mirror both
  render an indented body under a `pass_warning`/`pass_error` carrying `text` — previously
  only `phase_done` got one, so failure events were headline-only in both views by
  construction.
  **(2) Thinking off.** The dump's diagnosis: the model spent the whole 1024-token cap inside
  an unterminated `<think>`, so `_strip_think_block` handed the parser an empty string. The
  pass kept thinking ON on the reasoning that naming what a turn is *about* is a judgement;
  that was the wrong read of it. It is an **indexing** pass — its prompt asks for a
  description and forbids commentary, so the deliberation had nowhere to go but the budget.
  It now runs `disable_thinking=True` (like the branch chooser and clean-base judge, for the
  same reason) with the cap cut 1024 → 384: `ABOUT` is capped at 300 chars and `TAGS` at 8, so
  ~200 tokens covers a Russian descriptor with margin, and the remaining headroom lets a
  family whose template ignores the flag (`enable_thinking` is gemma-4/qwen3; gpt-oss's
  `reasoning_effort` is untouched by it) fail fast and visibly rather than after 90 s.
  **Interpretation note for existing artifacts:** anchors written before this date are what
  survived the thinking-on failure mode, not a representative sample of the pass — a chat with
  few or no anchors from that period is evidence about the pass, not about the chat.

- **The reflection lane gets a degeneration floor, and the language backstop stops being a
  decider.** Observed in a Sleep run: a revision pass collapsed mid-generation into a
  ~4-token letter-soup walk (`… la lul lul la la de l l la de de l …`) and ran toward
  context exhaustion. Two independent defects, one visible incident.
  **(1) Reflection had almost no loop defense.** `_make_sync_reflect_generate` passed
  `stop_on_repeat=True` and nothing else: `REFLECT_REPETITION_PENALTY` and
  `REFLECT_NO_REPEAT_NGRAM` are both `None` **by design** (an n-gram ban forces
  off-distribution tokens at every legitimate phrase reuse and garbles analytical prose —
  see `reflection_config`), and the chat degeneration floor was scoped to
  chat/ephemeral/encounter on the reasoning that collapse was a live-chat concern. The
  reasoning was backwards: reflection runs the longest budgets on the box, unattended, with
  no operator watching a stream they can Stop — it was the *least*-defended lane, not one
  that needed no defense. And `stop_on_repeat` is structurally blind to this shape: it wants
  one exact 12-token span recurring 4x, but a stochastic walk over ~4 symbols has ~500k
  possible 12-grams, so every window is novel and no span ever recurs. That is precisely the
  gap `_DegenStop` (Layer 2) was built for — on the observed shape, distinct/window ≈ 0.06
  against a 0.35 floor. The reflect **and** agentic factories now pass `min_p` (Layer 1) and
  the `_DEGEN_KW` bundle (Layer 2). Layer 2 is halt-only and alters no sampled token; **Layer
  1 does shape reflection sampling, and therefore the IDEALs that become training targets** —
  deliberate, on the grounds that an implausible-tail excursion in an IDEAL is a poisoned
  target either way. The chat repetition penalty and anti-copy guard are still not shared
  (the first garbles reflective prose, the second has no "previous reply" to guard). Branch
  replay goes through `generate_from_ids_batch` and remains uncovered. Config keys keep their
  `chat_*` names; `chat_degen_guard: false` now disables the guard on reflection too.
  **(2) The script backstop could force a discard on its own evidence.** The collapse was
  itself a *wasted* generation: `_language_decision_guard`'s recheck had fired on a false
  positive. `detect_language_drift` compares the reply's dominant script family against the
  **user's turns for that job only** — so a Russian conversation whose user turn is
  Latin-dominant for non-language reasons (pasted code, URLs, English proper nouns; here
  "Encyclopedia Dramatica" plus a link) reads as `conv_fam == "latin"`, and Ava's correct
  Cyrillic reply as drift. Worse was what followed: the recheck's output was unparseable, and
  the old control flow left `drift` at the heuristic's `True`, forcing `verdict = "revise"`.
  `_generate_ideal_reply` then ran with `require_user_language=True`, which accepts a
  re-answer only if its script matches the *conversation's* family — so the correct Russian
  IDEAL was rejected twice, the exchange landed in `revised_missing_ideal`
  (`lang_drift_unrepaired`), and **silently dropped out of the training corpus**. Drift is now
  set only when the recheck affirms it (`VERDICT: revise` or `LANG_DRIFT: yes`); a keep, a
  keep missing its marker, an unparseable judgement, or a failed/degenerated generation
  restores the original judgement and emits a `pass_warning` naming the likely cause. The
  documented contract ("the MODEL is the decider") now actually holds through the recheck.
  Not fixed here: the backstop still false-positives on such turns (stripping URLs/code spans
  before scoring is the real repair), and narrator turns — an Ava-initiated session's synthetic
  English impulse at exchange 0 — still count toward `conv_fam`.

- **Reach-out backoff, and check-in can finally see its own unanswered messages.**
  Observed on the GPU box: between 2026-07-30 05:37 and 07-31 10:34 Ava cold-opened the
  user **15 consecutive times, hourly**, each an unanswered restatement of one thought
  ("subjectivity as the waste of a KPI-driven system"), several near-verbatim paraphrases
  of each other. Every send was within the rules. Three causes compounded, and the first is
  the interesting one. **(1) The composition pass could not see what it had already sent.**
  `checkin._has_user_turn` excludes an unanswered `initiated_by:"ava"` session, and it
  gated *both* `_last_user_turn_dt` (correct — her own messages must not answer "has the
  user gone quiet?" on the user's behalf) and `_recent_chats`, the **context** window. So
  the prompt contained the recent conversations and none of her own standing openers: from
  inside it, every hour genuinely was the first time, and "follow up on something you were
  left wondering about" was answered correctly each time. **(2) The prompt was byte-identical
  each hour.** With the user silent no new chat carries a user turn, so the window froze on
  the same 5 chats, and `_summarize_recent` caches recaps by `(filename, mtime)` — same
  input, same output family. **(3) Nothing gated it.** `outreach._has_dangling_opener`
  exists for exactly this ("re-raising it — necessarily in near-identical words — is the
  pathology this guard exists to stop") but keys on an ask's `surfaced_in_sessions`, and
  check-in has no ask; `reachout_gate` was a flat 1 h cool-down against a 1 h job interval,
  so it never gated anything; and `silence_threshold_hours` is a floor, true forever once
  crossed. **Note this was not an ask-dedup failure** — the `[ask]` pool was healthy and
  check-in never consults it; dedup there would have prevented none of the 15.
  **Fix, in the shared gate rather than in check-in**, since synthesis has the same hole
  (it also composes free-form, with no ask to key a guard to): `core/reachout_gate.py`
  becomes the gate that answers *may she send right now?*, folding the cool-down together
  with **backoff on unanswered openers** — the window doubles per message sent since the
  user last spoke and unanswered (`unanswered_streak`), capped at `MAX_COOLDOWN_SECONDS`
  (24 h, reached at a streak of 5: 1 → 2 → 4 → 8 → 16 → 24 h). Backoff, not a hard stop:
  being ignored should make her quieter, not mute. It is also now **durable** — a
  `time.monotonic()` stamp was defensible for a 1 h burst guard but not for a day-long
  window, where a restart would zero the backoff and free the box to resume; both the
  streak and the last-send time are read from the chat corpus, with the in-process stamp
  kept as a floor. Openers are timed by **filename stem, not mtime** (`ChatLogger` rewrites
  per turn, so an answered opener's mtime is the *reply*), and an opener sent *before* the
  user's last turn does not count against her — they did engage, just elsewhere. The three
  call sites move to `may_reach_out() -> (allowed, reason)`; the new `reachout_backoff`
  skip is distinguished from `reachout_cooldown` (one means a sibling just sent, the other
  means she is talking into silence) and is added to the `consumed` classifiers, without
  which a 24 h hold would re-run the decision generation every 5-minute poll. **And the
  context hole is closed**: `checkin._standing_openers` folds her unanswered openers back
  into the decision prompt — not as conversations (there is nothing to summarize) but as a
  labelled standing block, quoted verbatim, newest last, capped at 5 with the remainder as
  a count, ending in the number she is looking at. Folded into the existing `{recent}` slot
  rather than a new placeholder, so an operator's already-customized `checkin_prompt.txt`
  gets it too. Verified against the live corpus: replaying the burst under the new gate
  sends **7 instead of 21**, with the streak correctly resetting each time the user
  replied; and reconstructing the corpus as of the 15th send, the block it would have
  carried reads "You are looking at 14 unanswered message(s) of your own." New GPU-free
  self-test `python -m core.reachout_gate` (15 checks: streak folding, the answered-opener
  reset, peer transcripts, the cap, both skip reasons, unconfigured degradation). Files:
  `server/inference/core/reachout_gate.py`, `server/inference/core/checkin.py`,
  `server/inference/core/outreach.py`, `server/inference/core/synthesis.py`,
  `server/inference/server.py`, `client/ui/sleep_widget.py`,
  `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

- **`[impression]` and the per-person user portrait — the persona digest, turned outward.**
  Ava had a standing self-portrait injected into every chat turn and nothing equivalent for
  the person she was talking to. Facts about someone *were* retrievable, but only
  situationally: whichever one to three embedded closest to the current message, competing
  with asks, persona and recollections for the reflection block's three slots. So she
  arrived at each turn knowing whatever that message happened to key on and nothing else
  about the person in front of her — reassembling her sense of them from fragments, every
  turn. That is the identical failure `render_digest_for_chat` was built to fix on her own
  side ("she arrives as herself instead of reassembling herself each time"), and it takes
  the identical fix.

  The new kind is `[persona]`'s counterpart rather than `[fact]`'s, and the distinction is
  the whole design: a fact is what someone **told** her, an impression is what she **came
  to understand** about them — how they think, what they were really doing in an exchange,
  what she took at face value and now suspects meant something else. Two production sites,
  both of which the shape of the pass follows from. A **dedicated per-session pass** runs
  after revision; it is not another section of the consolidation prompt because
  consolidation is *chunked* (a person-level reading written per chunk would form from a
  fragment and be emitted several times per chat) and because the framing genuinely
  differs — this pass wants the notice-don't-diagnose discipline and the explicit "writing
  nothing is correct" permission, which would be noise inside a prompt whose job is
  distilling stated content. And **synthesis** gained an `[impression]` line kind, the
  natural second site: re-reading an aged conversation as who she is now is exactly the
  vantage from which "what I only now see about them" becomes available.

  `[impression]` is **RAG-only by construction** — no `weights_persona.jsonl` line, no
  ledger anchor, nothing trains. The reasons differ from the recollection's. An impression
  is *revisable by construction* (it is a reading, and a person is entitled to have it be
  wrong) while the weights are the one store with no cheap undo; and a stable truth about a
  person already has its weights path (`[fact]` with `(about: NAME)`, gated on
  `source_class`), so routing a soft reading down the same path would let an impression
  into the weights while **bypassing the hearsay gate** that governs every hard claim about
  a person. Attribution works exactly as it does for facts: an unmarked item takes the
  session's speaker (supplied structurally by the caller, never parsed from model output),
  a marked one keeps its own subject, and `source_class` splits `self` from `hearsay` —
  which is also what keeps someone's account of a third party out of that third party's
  portrait.

  The portrait itself (`core/user_digest.py`) folds impressions plus attributed non-hearsay
  facts, per person, into five facets — WHO / CARES / WAYS / WITH_ME / **UNSURE** — through
  the same three seams the digest uses, split at the same points and for the same reasons:
  a model-free gate *before* the clean-base window (an unchanged run must not pay for a
  swap it has no work for), clustering *inside* it (grouping paraphrases is an evaluation,
  so it belongs adapter-off beside the branch judge and fact placement), synthesis *after*
  it on the adapter (a reading of someone, in her voice, is authorship). UNSURE is the one
  facet with no self-side analogue, and it is load-bearing rather than decorative: a
  portrait of a *person* that carries only conclusions hardens into confident fiction the
  moment one reading is wrong, and unlike a self-portrait there is a real someone it can be
  wrong about. It renders last in the injected block, so the block ends on what she does
  not know — a portrait whose final line is a conclusion invites acting on the whole thing
  as settled.

  Three deliberate divergences from the digest, each a case where copying it would have
  been the wrong call. **No map-reduce clustering:** `persona_cluster` exists because one
  prompt cannot hold ~640 live persona statements, and one person's evidence is one to two
  orders of magnitude smaller, so the single flat grouping call is the right size of hammer.
  **No injection maturity gate:** the persona portrait gates on `_is_established` because it
  closes a *self*-reinforcement loop (portrait shapes reply → reply yields persona statement
  → statement feeds portrait), whereas the person supplies their own evidence by continuing
  to be themselves, so a wrong reading is corrected by the next conversation rather than
  amplified by it. **Live evidence only, never this run's staging** (matching
  `_plan_persona_digest`): the portrait is written straight to the live users dir, so
  folding staged impressions would leave a portrait standing on evidence a later discard
  erases. The cost is a one-run lag, which is the digest's cadence anyway.

  On circularity, the one failure mode this design could have inherited: the user-notes pass
  is fenced (`before_session=filename`) **and** runs with `rag_include_impressions=False`.
  Shown her prior readings of someone, she restates them — and recurrence across distinct
  sessions, the single signal the portrait ranks on, would then be measuring what the prompt
  handed her rather than what she independently noticed twice. That is the circular
  self-vote `reflection_digest._TENURE_DECAY` had to discount after the fact; on this side
  it can simply be prevented, so it was.

  Memory stays **shared and global** (`AVA_MEMORY.md` §G): a portrait scopes *injection* —
  whose reading is standing context right now — never storage or retrieval. Every impression
  stays in the one op-log, attributed, and recallable in anyone's conversation. Injection
  rides the Chat tab's **Facts** checkbox (this is knowledge about someone, not Ava's self)
  and sits directly after the self-portrait, completing the framing the identity line opens
  ("X is speaking with you right now") before any of this moment's material arrives. An
  empty portrait — nobody nameable, fewer than four observations, or the switch off — leaves
  the per-turn `[impression]` RAG channel on, so a newly-met person degrades to exactly the
  previous behaviour. Three independent kill switches: `overrides.user_notes` (producing
  impressions), `impressions.enabled` (retrieving them), `user_portrait.enabled` (folding
  and injecting the portrait). Portraits travel with `data/` in a runnable snapshot, are
  archived per run under `reflections/<run_id>/users/`, and are wiped **with** the op-log
  they derive from — a portrait outliving its evidence would be a ghost: injected into every
  turn, and no longer derivable from or correctable by anything on the box.

  Negative-space note for the next kind: the `elif not include_facts` fallthrough in
  `_query_reflection` bit again, exactly as the `[recollection]` entry below predicted it
  would. `[impression]` needed its own branch, and so will whatever comes next.

  Verified GPU-free end to end — parse → write (attribution + hearsay class) → fold →
  portrait plan/cluster/synthesize → rendered chat block — driving the real runner pass and
  the real run-level seams with a scripted stub generator. **Not yet exercised on a live
  GPU.**

- **`[recollection]` — a fourth reflection-memory kind, so a revisit is visible to retrieval.**
  Tracing what "Revisit old chat" actually changes in RAG turned up a gap between what the
  pass is *for* and what it can reach. A revisit re-derives an aged chat's trainable target,
  rewrites its gist, and regenerates its anchors — but every chat-side curve is keyed to the
  **chat's own** wall-clock age (`_age_of_session` reads the filename stem; `reflected_at` is
  only a frozen/not gate), so the freshly re-derived gist served at exactly the 0.2 floor the
  stale one did. Fresher content, identical weight. And the re-derived IDEAL reached RAG not
  at all: the verbatim and anchor channels both render `assistant_response` off the
  **transcript**, and the only target→transcript path is Training review's `locked`-gated
  *Rewrite history*. Net effect: the one thing a revisit changed in retrieval was whatever
  consolidation happened to distil as a `[fact]`.

  The fix is a new object with today's birth date, not a resurrected old one. Two shapes were
  considered. **Refreshing the clock** (age = `max(chat_ts, last_revisit_ts)`) is a two-line
  change and was rejected: it would have to be gist/anchor-only, because `_render_anchor`
  keys its payload switch on `verbatim_alive` and reviving that would start quoting
  five-week-old turns verbatim again — re-arming exactly the induction-copy attractor the 96 h
  cap exists to disarm; it makes decay revisit-frequency-dependent, so a frequently revisited
  chat never fades (the fixation `min_revisit_days` exists to prevent, reintroduced one level
  down); and it desyncs RAG age from training age, which `build_dataset` still takes from the
  chat's real date. **Minting a new record** keeps every existing curve honest: the old
  conversation goes on fading *as a conversation*, while the conclusion she now draws from it
  is a separate memory that points back at it.

  So: the recollection is the **gist's fresh sibling** — same grain (one per conversation),
  different clock. `recollection_rag_weight_hours` holds 1.0 for `recollection_hold_h` (96 h)
  then decays affinely to `recollection_floor_weight` (0.3) at `recollection_cap_age_h`
  (192 h). Deliberately no ramp-up: the gist ramps because its verbatim is still authoritative
  early, and a recollection has no verbatim competitor — it is born as the best available
  representation of a conversation whose verbatim expired days ago. Its floor sits above
  `gist_floor_weight` so a current reading outranks the faded gist beside it.

  Four things worth recording for a later reader.

  **It is the first per-record clock in the reflection index.** Facts and persona are weighted
  per *source bundle* (`_consolidation_modifiers`, keyed on the chat's age); `_recollection_
  modifier` measures hours since the op-log `ts`. Passing the chat's age there would exactly
  undo the point of the kind, which is why the curve's docstring says so twice.

  **`origin_ts` finally has a consumer, and it is display, not weight.** The chat's own date
  renders in the recall label (`looking back on a conversation from 2026-06-21`) so an old
  conversation is never presented as recent — the same referent confusion `_attribution_label`
  fixes for third-party facts. ISO rather than prose because the corpus is Russian/English/
  mixed and the block is injected verbatim. Whether it should also damp the weight was left
  open: a fresh reading of a two-month-old chat currently ranks with a fresh reading of a
  two-week-old one.

  **Supersession is what makes the kind safe to produce often.** Each write evicts this chat's
  previous recollection, so the corpus holds at most one live reading per conversation.
  Without it, a rotating revisit schedule accumulates one paraphrase per pass — the same
  pool-inflation failure mode the revisit path already has for facts and asks, where nothing
  dedups semantically at write time.

  **RAG-only is structural, not a policy.** One op-log insert; no weights line, no ledger
  anchor, no sidecar field, no training row. The revisit's re-derived target already reaches
  the build through the sidecar, so a second path into the weights would mean a build training
  on what the model said about what the model said.

  The producing pass is revisit-only and runs last (after revision and anchors, so it can read
  the tags and so a failure cannot touch the trainable target). Its retrieval is unlike every
  other pass here: **unfenced** — `before_session=""`, because reading an old chat with what
  she knows *now* is the entire premise, and the replay-faithful cutoff the other passes use
  would make the output indistinguishable from the original consolidation — with the past-chat
  block off (the transcript is already the content) and the recollection channel itself off,
  so her previous reading cannot anchor the new one into restating it. Threading that last
  gate meant adding `rag_include_recollections` to **both** reflect-generate seams (the server's
  and `reflection_run.py`'s), which have separate signatures; a `TypeError` fallback was
  written first and then deleted in favour of one contract, since catching it would have masked
  a real `TypeError` from inside generation.

  Landmine found on the way in and worth naming: `_query_reflection`'s gate chain ends in
  `elif not include_facts: continue`, a catch-all **else** — so a new kind without its own
  branch silently rides the *fact* gate, and `ReflectionMemory.embed_text` likewise returns
  `trigger` only for `kind == "fact"`. Both needed extending; so will the next kind.

  Tags ride the record but are **stored, not yet used for admission** — retrieval is dense on
  `TRIGGER`, and the sparse dense-OR-tag fuse the anchor channel uses is worth copying only
  once there are recollections to judge it against. Kill switches: `recollections.enabled`
  (retrieval) and `overrides.recollection=False` (production). Verified GPU-free end to end;
  **not yet exercised on a live GPU.**

## 2026-07-30 (later)

- **Training review's Regenerate streams and can be stopped** (commit `9a80863`). The repair loop's cost is
  attention, not compute: a re-answer runs for GPU minutes behind a frozen tab, and the
  operator only learned whether it was worth keeping once it finished. The dialog now opens
  at the press and fills in as the model writes, with a **Stop** button that halts it and
  keeps the partial (still applicable, still hand-editable) — abandoning a visibly bad
  generation at ten seconds instead of paying for all of it.

  Three things worth recording.

  **The CoT/answer split is done server-side.** The raw stream is family-specific (gemma-4's
  channel markers, qwen3's template-prefilled opener), and the tab's two boxes are exactly
  the reasoning/answer split the public API already streams — so `_regenerate_exchange_sync`
  reuses `_CotStreamSplitter` and emits `regenerate_chunk {cot, reply}`. The client stays
  family-agnostic, as it is everywhere else. No reconciliation logic was needed: the terminal
  `exchange_regenerated` is authoritative and simply replaces the streamed approximation.

  **A stopped generation is classified from the splitter's state, not the text.** A cut
  mid-thought leaves no close marker in the raw text, so `_split_regenerated`'s
  `<think>…</think>` match fails and the whole thought reads as an *answer* — which Apply
  would then write as a reply with no reasoning. `splitter.in_think` is the disambiguator;
  the cleaned raw (not the streamed accumulation) becomes the CoT, so the deliberately
  held-back tail isn't lost.

  **Stopping needed the handler moved to `_HANDLERS_WS_MSG_QUEUE`.** The main dispatch loop
  awaits its handler, so a `cancel` arriving mid-regeneration sat unread in the queue until
  the generation it was meant to stop had already finished. It now drains the queue itself
  exactly as `_run_generation` does — putting every non-`cancel` message back, since the
  client serializes its RPCs behind one lock and anything else here arrived out of band. The
  stop is re-asserted each pass because `generate_fn` clears `_cancel_event` when it starts,
  so a stop landing before the executor picked the job up would otherwise be swallowed.

- **The public API streams (SSE).** Shipped hours after the endpoint itself, because the
  first real client — Continue.dev in VS Code — hit the deliberate `stream:true` 400 on
  its first request. Editor clients stream unconditionally; there was no configuration on
  either side that made a non-streaming endpoint usable. The explicit refusal did its job
  (the failure named its own cause instead of surfacing as a timeout), but the increment
  it was pointing at turned out to be mandatory rather than optional.

  Two structural decisions worth keeping.

  **Headers are withheld until the first event.** Committing to `200 text/event-stream`
  before generation starts would mean reporting *every* pre-generation failure — no model
  loaded, a prompt past the context window — as a 200 with an error buried in the body,
  which is exactly the fails-far-from-the-cause problem the non-streaming path was built
  to avoid. Waiting for the first queue event costs nothing and keeps real status codes
  where they are still available. Once the stream is open there is no status code left, so
  a mid-stream failure is reported in-band rather than truncating silently.

  **The CoT/answer split had to become incremental.** Non-streaming gets it free at the
  end (`_clean_response` → `_parse_cot`); SSE needs it live, because the thought belongs on
  `delta.reasoning_content` and the answer on `delta.content`. `_CotStreamSplitter`
  classifies the RAW stream — every family this box runs begins *inside* the thinking
  channel (gemma-4 via `think_prefill`, qwen3 via its template), so the initial state is
  "reasoning" and the first `close_markers` hit flips it. Being raw, it is optimistic: none
  of the end-of-generation cleaning has run, so a small tail is held back and the caller
  reconciles against the authoritative cleaned answer at the end (`.reconcile`). When the
  two disagree outright, reconciliation sends nothing — a slightly stale tail is a better
  failure than a duplicated paragraph.

  One bug caught by the end-to-end test rather than by reading: `flush()` counted the
  held-back tail as already-streamed, so reconciliation concluded it had gone out and the
  reply arrived truncated at the last ~32 characters. `flush()` now returns the reasoning
  half only and leaves `content_emitted` untouched, which also makes the tail come from
  the *cleaned* text — strictly better than the raw bytes it was going to emit.

  Recorded limitation, gpt-oss only: that family neither prefills an opener nor emits a
  close marker in the raw stream (harmony `analysis…assistantfinal…`, understood only by
  `normalize_cot` at the end), so its analysis channel streams visibly and reconciliation
  cannot repair it — what was streamed is not a prefix of the cleaned answer, so re-sending
  would duplicate rather than fix. The models this box runs are unaffected, non-streaming
  is unaffected everywhere, and the self-test asserts the current wrong behaviour so a
  future fix has an expectation to flip.

## 2026-07-30

- **A public OpenAI-compatible API, on its own port, that logs nothing.** External tools —
  an agentic code assistant, an editor plugin, an SDK script — can now query Ava without
  the PyQt client, via `core/api_http.py` (default port 8000, opt-in per box through
  `server_config.json` `api.enabled`).

  Most of the machinery already existed: the gossip route on the management sidecar has
  spoken chat-completions since 2026-07-12, and its `log_transcripts=False` mode was
  already a supported stateless configuration. What was missing was not protocol but
  *framing and blast radius*.

  **Why a separate port rather than another route on 8767.** The sidecar is the box's
  management plane: `/export` hands out the adapter weights and the entire chat corpus,
  `/precision` rewrites the config. That surface assumes the operator's own client on a
  trusted link. This one assumes the opposite — it is meant to be pointed at by tools the
  operator may not have written. Separate listeners let an operator expose one without the
  other, which a shared port makes impossible however the routes are guarded.

  **Why nothing is logged, and what "nothing" means.** The generate callable is built with
  `log_transcripts` forced off, so an external tool's traffic writes no transcript to
  `data/chats/` and therefore never reaches reflection, the training corpus, the review
  archive, or the chat RAG index. It is structural rather than a knob: there is no
  configuration under which API traffic becomes trainable. Her memory is still *read* —
  that is the point of querying Ava rather than the base model — so the asymmetry to record
  for a future reader is that API traffic **consumes** her accumulated state without
  **contributing** to it. This is the deliberate inverse of the gossip route, which logs its
  own half precisely so the box can reflect on it.

  **The one real design question was the client's `system` message.** The gossip path drops
  it, correctly: a peer Ava has no business rewriting her identity. But an agentic client
  puts its *entire* operating brief in that role, so inheriting that drop would have made
  the endpoint accept requests and ignore the instructions in them — the worst failure mode,
  since it looks like it works. The API mode instead wraps it in
  `prompts/api_client_system_prompt.txt` and appends it last, nearest the turns, framed as
  the working brief for the exchange rather than as a replacement identity. Wrapping instead
  of concatenating keeps the two authorities visibly distinct: her prompt is who she is, the
  client's is the job. `api.client_system:"drop"` restores the gossip behaviour.

  Both surfaces now share ONE generate body (`generation._make_openai_generate`) and one
  wire format (`api_http.openai_response`), so the API and gossip prompt assemblies cannot
  drift apart. The API mode differs only in the three places where a tool is not a peer: no
  peer framing, the chat portrait instead of the meeting-a-stranger introduction, and the
  client system message honoured.

  **Non-streaming, deliberately and explicitly.** `stream:true` returns a 400 naming the
  reason rather than a JSON body the client would hang on parsing as SSE — an honest
  refusal is worth more than a silent shape mismatch surfacing as a timeout three layers
  away. SSE is the known next increment; the backend already streams and
  `_sync_chat_generate` already takes an `on_chunk`, so what it needs is a queue handing
  chunks from the executor thread to the HTTP thread. A non-empty `tools`/`functions` is
  refused on the same principle and for a longer-lived reason: nothing in Ava emits tool
  calls, so an agent that offers tools and gets prose back loops on a reply it cannot
  parse, whereas a client told up front can fall back to a promptless mode. Streaming is
  an increment; tool calling is not on the path.

  Two smaller corrections fell out of the work. `generation._preempt_background_reflection`
  had no blocking twin, so the OpenAI endpoints — served off plain HTTP threads, which
  cannot await — queued behind a background per-chat reflection for as long as that chat
  took, while an equivalent UI turn preempted it in about a second; `_preempt_background_
  reflection_sync` closes that, and the gossip route inherits the fix. And the gossip
  callable now returns a `usage` block, since an OpenAI client expects token counts.

  Auth is a shared bearer key, enforced (unlike gossip's `gossip.api_key` hook, which
  `GOSSIP.md` §4.3 left present-but-unchecked). Empty ⇒ open, and a non-loopback bind with
  no key warns at boot. Not yet exercised against a live GPU or a real agent client; the
  self-test (`python -m core.api_http`) covers routing, auth, the interop shapes, and the
  busy refusal against a stub generate.

- **Both OpenAI-compatible endpoints now default ON** (`api.enabled`, `gossip.enabled`),
  so a box reaches them with a `git pull` + restart rather than a hand-edit of the
  gitignored `server_config.json`. The trade recorded for a later reader: **an unconfigured
  box now serves GPU time on ports 8000 and 8767 to anyone who can reach them.** Neither
  route has auth by default — the API's `api.api_key` is enforced but empty, and gossip's
  `gossip.api_key` is not checked at all — and both bind to the inference server's `--host`,
  which is `0.0.0.0` in the normal invocation. This is deliberate for a single-operator box
  on a trusted LAN and wrong for anything else; `api.host: "127.0.0.1"`, an `api.api_key`,
  or `enabled: false` each close it. The API prints a boot warning when it binds
  non-loopback without a key, which is the only signal an operator gets.

## 2026-07-29

- **Per-exchange anchors become a retrieval channel with their own reserved slot.**
  Anchors have been produced into the sidecars since 2026-07-28 and readable through the
  Chat tab's Match preview, but nothing retrieved on them. They are now a fourth FAISS
  index in `rag_engine` holding one reserved slot in the past-chat block.

  What settled the design was recognising that an anchor is **not** a fourth kind of
  memory beside verbatim and gist. It is the exchange-granular counterpart of the gist,
  which is session-granular. Past `rag_cap_age_h` the verbatim channel is hard zero, so a
  whole conversation is represented by one session recap forever — a 40-exchange chat gets
  a single blob. The anchor is what keeps a *specific* exchange recallable past that cap.

  This changes how an artifact should be read. An anchor carries **no payload of its own**:
  it claims an exchange and injects the best representation still permitted — the real
  turns while verbatim is alive, the descriptor once it is not, which is the gist's
  crossfade applied one level down. The traditional ranking then skips a claimed exchange
  and advances to its next candidate, so the slot always adds a *distinct* exchange rather
  than duplicating one, and (since the anchor always injects at least what the chat channel
  would have) a claim can never downgrade the block. Its ranking prior is therefore the
  **envelope** `max(_chat_modifier, _gist_modifier)` rather than either curve: the verbatim
  curve would kill it precisely at the cap where it becomes the only per-exchange
  representation left, and the gist curve would suppress it while a chat is fresh,
  discarding its match-side advantages exactly where most retrieval happens.

  Two admission gates, fused with `max` rather than summed — they are alternative evidence
  for one claim, and summing would let two mediocre signals outrank one strong one. Dense
  cosine on the ABOUT line (floor 0.35) fixes the granularity and cross-lingual mismatches
  the 2026-07-28 entry measured; sparse tag overlap catches the rare coined tokens dense
  retrieval loses by construction, and so must be able to admit an entry dense never
  surfaced. The known `слушай` limit is guarded rather than solved: a single short tag
  cannot admit alone, needing either corroboration (≥2 tags) or the module's own
  distinctiveness proxy (≥12 chars). A stoplist remains the real fix.

  One deliberate asymmetry worth recording: the chat channel's `_MAX_SCORE` near-duplicate
  ceiling does **not** apply to anchors. An anchor is a paraphrase and never the
  near-duplicate the ceiling exists to reject — that rejection is why deliberately probing
  a remembered phrase was guaranteed *not* to retrieve it. The copy risk moves to the
  payload: above the ceiling the hit is kept but arrives as the descriptor, which cannot be
  copied verbatim because it was never said.

  Anchors index separately rather than as a third `kind` in the chat index. One vector per
  exchange against verbatim's several passages is a ~1:5 minority that a fixed oversampling
  window crowds out as the corpus grows, and a reserved slot has to be fillable at any
  corpus size. They are still *collected* in the chat pass, where the sidecar is already
  open for the gist and the transcript already parsed — which is what lets an anchor carry
  the turns it points at without a second read.

  Kill switch: `anchors.enabled` in `server_config.json` (default on), plus an
  `include_anchors` query gate riding `include_chat`. Coverage note for anyone reading a
  build from this week: anchors exist only for reflected chats, so the channel was live
  against 2 anchors on 1 of 9 chats when it landed.

- **Outreach stops re-raising an ask it is still waiting on, and stops fixating on the
  meta pool.** First issue found on the clean-base rebuild: Ava cold-opened with the same
  question twice, in near-identical words (`8dea6eadc176`, 03:11 and 07:14).

  Not a dedup failure — `content_key` did its job, and the 2026-07-20 ask-resolution loop
  did too. That loop closes an ask the user **answered**; here the user answered *nothing*.
  `surfaceable_questions` exempts `meta` from the surface ceiling by design ("a subject's
  question about its own 'I' is the one most worth holding"), so *never retires* met an
  hourly outreach clock and a five-item live meta pool and became a **5 h round-robin** —
  every ask coming back around, and the opener being composed from the same question text
  each time, coming back around in the same words. Each of the four asks re-raised had a
  still-unanswered opener sitting in `hot/chats` at the moment it was raised again.

  Two independent gates, neither of which retires a meta ask (the design intent stands —
  the pacing was what was missing):

  - A **re-ask gap.** The fold now annotates each item with `last_surfaced_ts` (newest
    `surface` op for that key), and `surfaceable_questions(min_gap_hours=…)` filters on it,
    ordering by `(surface_count, last_surfaced_ts)` so rotation is deterministic rather
    than dict-order. Outreach passes `outreach.min_reask_hours` (default **72 h**) — the
    ask-side counterpart of the anti-fixation gates `synthesis.min_resynth_days` and
    `revisit.min_revisit_days` already carry. The passive path keeps the `0` default: it
    fires only when the user opens a session, so *they* pace it.
  - A **dangling-opener guard.** `outreach._has_dangling_opener` refuses an ask whose
    `surfaced_in_sessions` still contains an unanswered Ava-initiated session
    (`initiated_by:"ava"`, ≤1 exchange). Deliberately **not** time-limited: an opener still
    hanging is still hanging. This is the load-bearing one — it is also self-limiting, so
    once she has cold-opened a few times with no reply she stops on her own rather than
    accumulating openers hourly.

  Outreach now walks up to 8 candidates (`_pick_ask`) rather than taking the top one, so a
  blocked favourite doesn't waste the window; a fully-blocked pool skips
  `all_awaiting_reply`. **Side effect worth noting:** with the gap on, selection finally
  reaches the `user` asks — `(meta + user)[:limit]` had starved every one of them since the
  base was created (five live, zero ever surfaced), because meta always sorted first and
  meta was never paced.

- **Training defaults reset for the from-scratch rebuild: `lora_r` 16 → 32, schedule
  `triangular` → `age_ramp` (flat, 1 epoch), `train_lr` 8e-6 (unchanged).** Ahead of a
  complete re-train, the cycle goes back to the simplest recipe that the wall-clock design
  actually calls for: **one flat pass**, where the per-row age ramp is the ONLY thing
  weighting one exchange against another.

  The trapezoid existed to solve a problem that only a multi-epoch pass has. Its warmup /
  hold / decay shape made every row see the same *average* LR regardless of its position in
  the chronological corpus — necessary because with `SequentialSampler` over an oldest-first
  corpus, any non-flat single-pass shape is a position bias. A **flat** schedule is
  order-neutral by construction and needs no compensating epochs, so `epochs` collapses from
  `train_plateau_epochs + 2` (5 by default) back to 1. `train_plateau_epochs` is now inert
  unless `triangular` is explicitly selected; the trapezoid and its `test_triangular_lr`
  selftest are retained, just not the default.

  **Rank 32 is a capacity change, not an LR change** — that is exactly what the
  rank-stabilized scaling (2026-07-26) bought. `TRAIN_LORA_ALPHA` stays FIXED at 4, so
  γ = 4/√r drops 1.0 → 0.707, cancelling the ~√2 growth in ‖dW‖ the extra rank would
  otherwise add, and `train_lr` 8e-6 carries over untouched. The old `alpha == r` regime is
  what made an r=32 run come out *less* stable than r=16; that trap is closed. Note γ == 1.0
  at r=16 remains the **calibration point** even though 16 is no longer the default.

  **Reading an old artifact:** every adapter records its own `r`, `lora_alpha`, and
  `use_rslora` in `adapter_config.json`, and each build's `builds.jsonl` line + forensic
  snapshot capture the config in effect — so a pre-change adapter is unambiguous and loads
  unchanged. But **an existing `server_config.json` is not retroactively updated**: the boot
  back-fill only adds *missing* keys, so a box whose config already pins `lora_r: 16` (or
  `train_lr_schedule: "triangular"`) keeps that value. Change it in `settings.py` or by hand
  before a rebuild if the new defaults are wanted. Files: `training/decay.py`
  (`TRAIN_LORA_R_DEFAULT`), `training/train_cycle.py` (`_LR_SCHEDULE_DEFAULT`),
  `inference/server.py` (back-fill fallback literals), `server/settings.py` (schema defaults
  + tooltips), `training/DESIGN.md`, `AVA_STATUS.md`, `CLAUDE.md`.

## 2026-07-28

- **TIL snippets moved from `server/til/snippets/` to `server/data/til/snippets/`.**
  Continues the gradual consolidation of everything that feeds a training set under the
  ordered `server/data/` home. `fetch_wiki.py` and `fetch_article.py` had already moved
  their output there (`snippets/wander/`, `snippets/lookups/`); `fetch_current_events.py`
  was the straggler still writing beside its own code. `server/til/` now holds **code
  only** — the three fetchers plus `wiki_sources.json` — and the tracked
  `til/snippets/.gitignore` (which existed to keep an output dir present in a source tree)
  is gone; `server/data/` is ignored wholesale.

  **The real consequence is snapshot coverage, not tidiness.** `snapshot_state._TIL_DIR`
  is `server/data/til`, so the news digests were previously *outside* the runnable
  snapshot and the Migrate/Fetch-snapshot bundle while wander and lookup snippets were
  inside it — a cloned box silently lost the current-events provenance. They now travel.

  Layout is now **one subdir per source kind** — `snippets/news/` (`fetch_current_events`)
  beside `snippets/wander/` (`fetch_wiki`) and `snippets/lookups/` (`fetch_article`) — so
  the three stay distinguishable and no kind can clobber another's filenames. (The digests
  briefly landed at the root of `snippets/` earlier the same day; that was tidied into
  `news/` before any further fetch.) **Reading an old artifact:** no format or content
  changed, and nothing reads these files back by path — they are written at fetch and
  handed to the learning pass in memory — so a pre-move snippet is still valid content
  wherever a snapshot happens to hold it.

- **`server_config.json` moved from `server/inference/` to `server/`.** It was never the
  inference role's file. The offline train cycle repoints its `adapter_id` and the wipe job
  resets it *while inference is down*; snapshot/migrate read it with no server running at
  all. Sitting inside `inference/` implied an ownership that four of its six readers do not
  have, and it made the role dir look like the box root. It now sits beside
  `watchdog_jobs.json` at the server root, where the box-level config belongs.

  **Reading an old artifact:** the file's *content* and semantics are unchanged, so any
  build snapshot, `MANIFEST.json`, or reflection archive that captured a config still means
  exactly what it meant. Only the live location moved. The portable **bundle** layout is
  likewise untouched — `server_config.json` remains a single member at the tar root for
  both `/export` and `/snapshot/export`; only the client's local destination changed (and
  Migrate/Fetch-snapshot now delete a pre-move legacy copy so a box has ONE config).

  **Migration is automatic and in-place.** Both writing entry points resolve the path
  through a helper that `os.replace`s a legacy `inference/server_config.json` up to the
  server root on first resolution (`training/reflections_path.server_config_path`,
  `server.py::_server_config_file`); the read-mostly consumers (`snapshot_state`,
  `mgmt_http`, `settings.py`, `state_wipe`) fall back to the legacy path without moving it.
  So a deployed box migrates on `git pull` + restart with no SSH, and a checkout that is
  mid-upgrade never reads a stale copy. The legacy path stays in `.gitignore`.

- **Anchor match preview: the query side, as a diagnostic rather than as retrieval.** With
  anchors only starting to accumulate, evaluating retrieval *quality* is premature — but
  the query side can be judged immediately, and it is where a lexical matcher fails first.
  The Chat tab gains a **Match preview** strip under the input box: as you type, it shows
  which stored anchors the message would match and which tags fired. It retrieves nothing
  and injects nothing.

  Matching is **generation-free by design**. A tag-generation pass at query time would sit
  *before* retrieval, serial with time-to-first-token, costing seconds on every turn;
  lexical matching of the raw text costs microseconds and covers the case dense retrieval
  loses — rare coined tokens that mean-pooled embeddings dilute. It needs no model loaded,
  which also means the preview works while the box is busy reflecting.

  **The hard part is morphology, not matching**, and it would have been the silent failure.
  Russian inflects heavily: the user writes `крокодильничеством` where the tag reads
  `крокодильничество`, so an exact or substring test misses and the channel would look like
  "tags don't work" when it is really "tags don't decline". `words_match` accepts a shared
  prefix leaving at most 3 characters differing on either side, and requires exact equality
  below 6 characters where prefixes collide (`код`/`кот`). Verified against realistic
  input: `крокодильничеством` → `крокодильничество`, `единицами измерения` → `единицы
  измерения` (both words inflected), `как дела?` → no match. A stemmer would be more
  correct and is a dependency; this is stdlib and exploits the fact that the distinctive
  terms here are long.

  `tag_weight` scores a hit by IDF × length. The corpus-saturation cutoff only applies past
  `_FILLER_MIN_DOCS`=4 documents — without that floor a tag on 2 of 3 anchors reads as "67%
  of the corpus" and is discarded, which is exactly the state the corpus is in while
  anchors accumulate. **Known limit, stated because the preview is how it will be found:**
  rarity is measured within the *tag vocabulary*, so a tag that is a common word of the
  language still scores as distinctive — the live corpus has `слушай` ("listen") as a tag,
  which will match half of everything. Catching that needs a stoplist or language frequency
  data, neither of which is guessed at here.

- **Match preview reads the background checkpoint, not just the live sidecars** (fix). The
  preview reported "no anchors stored yet" on a box that had visibly just generated them.
  Cause: the **background** per-chat pass — which produces most anchors — runs against a
  throwaway staging dir and commits each finished chat's sidecar to
  `data/hot/reflection_checkpoint/chats/`; those only reach the live chats dir when the
  *next normal* reflection run starts and calls `fold_checkpoint_to_live`. So a freshly
  anchored chat is invisible to any live-only reader until an operator happens to run a
  Sleep pass — which defeats a tool whose purpose is watching anchors being produced. The
  handler now unions live + archive + checkpoint (checkpoint last, since it is newer by
  construction) and reports a `pending` count so the split is visible in the strip rather
  than mysterious. The in-flight `background_staging/` dir is deliberately still excluded:
  that chat is mid-reflection and its staging is discarded on preempt, so a diagnostic
  must not show state that may never land. Also corrected the archive root, which was
  hand-derived as `_CHATS_DIR.parent.parent/archive/chats` — wrong since chats moved to
  `server/data/chats` — and now uses `reflections_path.archive_chats_dir()`.

- **Open `[ask]` questions and the wander/TIL channel are no longer injected into any
  chat-shaped prompt.** Both are slated for redesign, so this removes them as noise in the
  meantime; neither judgement is that the idea is wrong, and no data, index, prompt or
  decay curve was deleted — only the callers stopped asking for the blocks. Two flags in
  `core/generation.py` (`_INJECT_OPEN_ASKS`, `_INJECT_WANDER`) restore the previous
  behaviour exactly.

  **Asks** reached a turn two ways, and both are off. The `(still wondering …)` recall
  lines rode the reflection-memory block every turn, selected by similarity — which they
  usually won, because an ask embeds on its full conversational question text while a fact
  embeds on a short topic-label `trigger`, so asks crowded facts out of the block's three
  slots (live chat now gives all three to `[fact]`). The `surface_prompt.txt` block was the
  only path with an actual mandate to raise something, but it is context-blind (selection
  never looks at the message), fires exactly once per session, and does so at the moment a
  live user request is competing for attention. Turning it off also stops a real harm:
  `write_surface` stamps the retirement counter when the block is *injected* and the turn
  lands — not when Ava actually asks anything — so a `user` ask was permanently retired
  from surfacing after three unasked injections, out of a budget it shares with outreach.
  Ava still raises questions by opening a conversation (outreach / synthesis / check-in are
  untouched), which is the path that demonstrably works.

  **Wander** gates on a 0.15 raw cosine — low enough that its single slot is filled on
  essentially every turn — and then ranks by an age step (0.4/0.3/0.2/0.1 per 24 h), so the
  newest article wins unless an older one scores 4× its similarity. The effective behaviour
  was "inject today's article, near unconditionally, for 24 h", with relevance delegated
  entirely to the model once the block was already in the prompt. It came out of all three
  callers together — live chat, the gossip serving endpoint, and the encounter loop —
  since the argument applies identically to each and a half-on state would leave the
  redesign two behaviours to reconcile.

  Mechanically, `RagEngine.query` gains an `include_asks` gate: asks used to ride
  `include_facts` on the reasoning that both are "informational", which is true and
  irrelevant — they compete very differently, and splitting the gate is what allows dropping
  asks while keeping facts. It defaults True, so reflection and revision passes still see
  open asks and are entirely unaffected. The Chat tab's History and Facts tooltips were
  corrected: they described channels those boxes no longer gate.

- **The Activity tab now shows reflection output for every run, not only background ones.**
  The mirror had two grains keyed on the run's `source`: a background per-chat run got
  Sleep-tab detail because the Activity tab is its only window, while an **operator** run
  got bare phase markers, on the reasoning that it would otherwise duplicate the Sleep tab.
  That does not survive use. The Sleep console shows only the run *it* launched, so
  anything an operator wants to watch — immediately: the new per-exchange anchors — was
  absent from the one view that is always live, and the tab read as broken sitting next to
  a Sleep textbox streaming the same run.

  Now one grain for everything (`_ACTIVITY_MIRROR_EVENTS`): phases and sessions, branch
  outcomes, ask resolutions, and **both** `pass_error` and `pass_warning` — the latter was
  invisible at either grain despite carrying exactly the recoverable failures an operator
  needs when judging a new pass (an unparseable anchor, a retried verdict). Every
  `phase_done` carries the pass's own output indented beneath it: revision's
  VERDICT-WHY-IDEAL, consolidation's distilled `[fact]`/`[ask]`/`[resolved]` items, an
  anchor's ABOUT + tags.

  The bounds that made the detailed grain affordable are unchanged and are why one grain
  is safe: the raw token stream (`phase_progress`) is still never mirrored, and every
  attached body is clipped at `_MIRROR_TEXT_CAP`=4000. The honest cost: a long operator
  run over many chats can evict older entries from `activity_log`'s 2000-event in-memory
  ring faster than before. The on-disk journal keeps them, and a new client-side **Hide
  reflection detail** checkbox collapses each event back to its headline for reading the
  log as an overview (it keys on the server's 4-space body indentation, so it needs no
  knowledge of event shapes).

- **Per-exchange retrieval anchors: a generated descriptor + tags, produced but not yet
  read.** Groundwork for reworking chat retrieval, landed producer-first so the next
  reflection run yields data to judge before anything about live chat changes. Each
  anchorable exchange now gets an `ABOUT` one-liner and a normalized `TAGS` list, written
  to a new top-level `anchors` map in the chat sidecar. **Nothing retrieves on them** —
  `rag_engine` is untouched.

  Three measured problems motivate the shape. Chat RAG decides relevance on a 280-char
  passage but injects the whole exchange (clipped at 4000 chars/side), so match
  granularity and injection granularity are decoupled — this is why retrieved chats read
  as arbitrary. A near-verbatim quote scores above the `_MAX_SCORE`=0.90 anti-copy
  band-pass, making a deliberate probe of a remembered phrase the one query guaranteed not
  to retrieve its own chat. And 55% of stored summaries describe Russian conversations in
  English (71 of 130 measured), so a Russian turn is matched cross-lingually against an
  English description. A generated descriptor answers all three: one unit per exchange, a
  paraphrase rather than the raw text, authored in the conversation's own language.

  **Why tags are normalized, with the evidence.** `[fact].trigger` has been the same idea
  running uncontrolled, and its outcome is the argument: across 695 live facts it produced
  **1,588 distinct triggers with only 10% used more than once** (≈1.24 uses per tag), and
  the corpus's most-used concept fragmented three ways — `"crocodiling"` (13),
  `"крокодильничество"` (13), `crocodiling` (11) — split by quoting and by language, so a
  query could match at most one bucket. `exchange_anchor.normalize_tag` collapses the
  *formatting* half of that. The cross-language half is **deliberately not attempted**: it
  needs an alias registry that can hold members, and a hand-written mapping would silently
  fuse unrelated concepts. Read the 10% as evidence that *naive* tagging fragments — that
  system had no registry, no reuse instruction and no normalization — not that tagging
  fails.

  **Isolation from training is structural, not by convention.** Anchors live in their own
  top-level `anchors` map rather than inside the exchange's verdict record, because
  `training/dialogue_source` reads `exchanges[<i>]`; keeping them out of it means this pass
  cannot perturb a trainable row whatever it generates. For the same reason it runs as its
  own small pass *after* revision instead of as extra fields on the revision prompt —
  revision's generation resolves the trainable target, and an indexing concern must not put
  IDEAL quality at risk. Cost is one short RAG-off generation (1024-token cap) per
  anchorable exchange, reusing revision's already-built replay-faithful content.

  Skips filler exchanges (`MIN_USER_CHARS`=100, the same threshold and reasoning as the
  training corpus's `contamination_min_user_chars`) and human-`locked` ones. Best-effort
  everywhere: a missing prompt file, an unparseable generation, or a failed write skips
  silently. Opt-out via `overrides.exchange_anchors=False`.

  **What to look at after the next run:** each anchor rides its own `phase="anchor"`
  `phase_done` event carrying `about` and `tags`, so it appears in the Sleep event log and
  — for a background per-chat run — mirrors into the Activity tab at full detail. On disk,
  `anchors` in any `<chat>.state.json`. The open questions this run is meant to answer are
  whether the descriptors are specific enough to discriminate between exchanges of the same
  chat, and whether tags converge at all without a registry (the prior says they will not,
  which is the case for building one).

- **The consolidation gist was two-thirds reflection format, and it was being recalled as
  memory** (`c16207b`). The summary pass is instructed to write one clean prose recap per
  conversation, and `rag_engine` chunks that recap into the **chat** index as `kind="gist"`
  passages. Because the gist is the *only* chat representation surviving past
  `rag_cap_age_h`, whatever it contains is eventually injected into live chat as a
  remembered conclusion. Measured over the live corpus: **110 of 173 stored summaries carry
  a `## WEIGHTS` / `## RAG` / `## RESOLVED` block or bulleted `[fact]` items, and 43 salvage
  no prose at all.** So Ava was being shown her own reflection format, labelled as something
  she recalls.

  The 2026-07-24 entry fixed one cause (the pass was reusing the consolidation `prepared`
  prompt, so it ran under `sleep_prompt` with the `[fact]`/`[ask]` RAG block). It did not
  close the hole, for two reasons now addressed: nothing **gated the store**, so anything
  the pass emitted was persisted verbatim and stayed persisted; and a **chunked session runs
  one summary generation per chunk**, each able to degenerate independently, so a leak can
  land mid-text rather than as a tail.

  `chat_sidecar.sanitize_gist` is now the single definition of what counts as prose here —
  truncate at the first structure boundary, drop a leading `### Reflection …` header and any
  `<eos>` tail, require `GIST_MIN_CHARS`=40 of salvaged content — applied per summary *part*
  before the join, again on write (a summary salvaging nothing is **refused**, so a dump can
  no longer overwrite a good recap), and again on read.

  **Interpretation note for existing artifacts.** The read-side filter means the stored
  sidecars are deliberately left as the run wrote them: `consolidation_summary.text` on disk
  is *not* what the index now serves. 84 of the stored gists are silently repaired at query
  time and 43 are dropped entirely — so a sidecar's summary field remains a faithful forensic
  record of the generation, while retrieval sees only its prose. Anything reading that field
  directly (a snapshot, a future analysis) should call `sanitize_gist` rather than assume the
  stored text is clean. A future pass may rewrite the corpus, at which point the read-side
  filter becomes a no-op rather than load-bearing.

  Truncating at the first boundary rather than filtering structured lines wherever they
  appear is a deliberate, data-driven choice: a dump's continuation lines are not themselves
  boundaries (a `## RESOLVED` section lists plain `- artemyvo: does he …` items), so
  line-wise filtering keeps those fragments and they read as prose. Of the 110 leaky
  summaries only 11 had any non-boundary line after the first boundary, and inspection showed
  every one of them to be exactly such a continuation.

  Scope: this is the data-quality half only. The gist's *retrieval* problems are untouched
  and separate — it competes for the same `top_k=2` chat budget as the verbatim it is meant
  to anchor (and is outweighted by it below ~96 h by construction), and its passages carry
  distinct sentinel `exchange_index` values, so a long recap holds proportionally more
  ranking slots than a short one.

## 2026-07-27

- **Training review runs from a remote UI box — the corpus is served, not read off local
  disk.** The tab was written for an operator sitting on the GPU host: `TrainingLoadWorker`
  located the newest `models/snapshots/*/sft_render.jsonl` in this checkout, read every
  chat's live `.state.json` sidecar for the `locked` connect-back, and scanned the raw
  `weights_persona.jsonl` + `consolidation_anchors.jsonl` logs for the persona detector's
  ledger tier — three roots that live beside the *model*, not beside the UI. From a remote
  Mac the only way to populate them was Migrate → *Fetch snapshot*, which pulls the whole
  personality (adapter weights included) to read some text, and which then goes stale the
  moment the server builds again.

  The projection moved server-side verbatim into the new `inference/core/training_review.py`
  (pure stdlib, no project imports) and is served gzipped by the inference sidecar at
  `GET /training/review`. Sizes justify it: the render carries every row's *whole*
  conversation while the tab shows only the final turn (label masking targets nothing else),
  so build-20260724-060241 goes over the wire at **2.6 MB against a 33.8 MB render**, with no
  weights at all. The endpoint is read-only and therefore deliberately **not** gated on the
  sidecar's `_is_busy()` predicate — a reflection run is precisely when an operator wants to
  read what the last build trained.

  **One definition, two callers.** The client fetches from `sidecar_base_url()` when
  connected and otherwise loads that same module **by path** (`importlib`, no `sys.path`
  mutation — the module is stdlib-only by construction) out of this checkout's `server/`
  tree, so the offline/same-box path can't drift from what the sidecar serves. The client
  keeps exactly one half of the work: persona-opener *scoring*, which is pure and lexical
  and is what lets `_rescore_persona` re-rank a row in place after a repair without another
  round trip; statements ship normalized and are re-indexed on arrival. The status line now
  names the origin (`via http://host:8767`), because the repair RPCs write to the *connected*
  server whatever the corpus was read from — an operator must be able to see when those are
  two different boxes. A server predating the endpoint is diagnosed by its catch-all 404 and
  told to pull, rather than surfacing a bare "not found".

  Behaviour is unchanged, and checked rather than asserted: a parity harness ran the old
  local-disk parse and the new payload (through a JSON round-trip) over all 943 entries of
  build-20260724-060241 and compared every field including persona score/kind/strip —
  **0 mismatches**, identical statement set. Verified end to end against a live sidecar,
  local and remote paths byte-identical, plus the error paths (dead port, missing endpoint,
  unknown source, no local tree) and a headless GUI pass over the remote path (frozen ❄
  connect-back, strip prefill, search, filters).

  Snapshot is the only `source` today. The parameter exists because the obvious next value
  is `live` — assembling the NEXT build's corpus from the live chats via
  `training.build_dataset` (GPU-free, and `render.build_messages` needs no tokenizer), which
  would drop the "run a build first" precondition and make the live-sidecar patch redundant
  rather than load-bearing. Not built. Files: `server/inference/core/training_review.py`
  (new), `server/inference/core/mgmt_http.py`, `client/ui/training_review_widget.py`,
  `CLAUDE.md`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`.
  Commit: `7881b4a`.

## 2026-07-26

- **Persona reaches live chat as a standing portrait; the per-turn `[persona]` RAG channel
  is retired behind it.** This is the payoff of the two entries below — the digest was not
  trustworthy enough to sit in front of every turn until its clustering was fixed.

  Until now, persona reached an ordinary user turn only as RAG recall: whichever one to
  three `[persona]` statements embedded closest to the message, competing with `[fact]` and
  `[ask]` for the reflection block's three `_REFL_TOP_K` slots. With 640 near-duplicate live
  statements that routinely meant the same stance restated three ways, crowding out facts.
  Now `reflection_digest.render_digest_for_chat` renders the digest as one coherent
  self-portrait, injected in `generation._run_generation` after the identity/temporal anchors
  and before the RAG block (standing context before situational), and when it is present
  `include_persona` is passed False to `rag.query` — so the three slots go entirely to facts
  and asks.

  Three narrowings versus the other renderers, each for a stated reason. **Established
  dispositions only** (3 of 5 on the current digest): a portrait in *every* prompt tightens
  the self-reinforcement loop — portrait shapes replies, replies yield persona statements,
  those feed the portrait — and `_TENURE_DECAY` / `_PERSUASION_GAIN` were calibrated when
  persona was a far weaker channel; an emerging trait still reaches chat through the weights.
  **No STANCES**: dispositions are procedural (how she acts, and when), which is what a
  register is made of, whereas standing declarative assertions invite recitation and a claim
  belongs in fact recall. **A do-not-perform guard** in the framing, mirroring
  `rag_memory_prompt.txt`'s "do not announce that you are consulting notes". Framing is
  second person (an instruction about the block); the body stays first person (her own
  voice), the `self_reconcile_prompt.txt` shape.

  Degradation is explicit: an empty render — no digest, or nothing past the maturity gate —
  is the caller's signal to keep the `[persona]` RAG channel on, so a thin corpus behaves
  exactly as before. The Chat tab's **Persona** checkbox now gates the whole channel,
  portrait included, so the A/B still works. Cost is **664 tokens/turn** measured on the live
  digest, 2.7% of a 24,576 context; repeated `system_content` already collapses to a
  back-reference in the transcript.

  **Training is not coupled, and that is the point.** The trained row's system prefix is the
  *session-level* `chat.get("system_prompt")` (`training/dialogue_source.py`), never the
  per-turn `system_content` — the same asymmetry RAG blocks already have. So the portrait
  shapes what Ava sees while generating but never enters the training prefix: the model is
  trained to produce portrait-consistent replies given only the base prompt, making the
  portrait a teacher that distils into weights rather than a permanent crutch.

  Scope: live user chat only. Gossip keeps its own `render_digest_for_introduction`; the
  encounter and ephemeral paths are unchanged. Not yet exercised on a live GPU.

- **Map-reduce persona clustering wired into the reflection run; digest themes now carry
  anchor keys.** Follows the dry run added earlier today (below), which was confirmed
  working on the GPU box. Three changes.

  **Artifact schema.** Each persisted theme now carries `key` (the representative's anchor)
  and `member_keys` (the whole membership) beside the existing `members` content strings.
  Content prose is not a stable join back to the ledger — a statement can be re-registered
  with different whitespace — so without keys no pass can rejoin a persisted theme to live
  evidence. **A digest written before this has neither field, and a reader must treat their
  absence as "recluster from scratch".**

  **`run_digest_pass` split into three seams:** `plan_digest` (model-free gather + regenerate
  gate), `cluster_for_digest` (grouping, routed through `persona_cluster.run_map_reduce`),
  `synthesize_digest` (portrait + self-portrait + write). `run_digest_pass` remains as the
  one-model wrapper for callers with no clean-base swap.

  **`execute_run` drives the seams across two models.** The gate runs *before* the clean-base
  window, so an unchanged run still costs no swap and no model calls — unchanged from before.
  Clustering joins the existing single `clean_base_ctx` batch as a third job beside the branch
  judge and fact placement, so it costs no extra reload; synthesis then runs on the restored
  adapter as a new step 5. What makes this reordering safe is a property worth recording: the
  branch judge anchors on `persona_digest` as loaded at run *start* and the variable is never
  reassigned, so the digest a run writes is never read by that same run — moving it after the
  clean-base batch changes nothing downstream. Rationale for the split: grouping paraphrases
  is an *evaluation* ("do these two say the same thing?"), which belongs adapter-off like
  `fact_dedup`/`self_reconcile`, while the portrait is Ava's *authorship* and stays on the
  adapter. Previously both shared one `generate_fn` on the adapter, letting a drifting adapter
  group the very evidence that defines it.

  Degradation paths, in order: no clean-base swap available (the headless `reflection_run.py`
  CLI) ⇒ cluster on the adapter, since the clean base is an upgrade where it exists rather
  than a requirement; `overrides.persona_cluster_mapreduce=False` (kill-switch mirroring
  `apply_branch_judge`) ⇒ the historical single flat call; an exception inside the map-reduce
  path ⇒ the embedder/exact-key tiers, so a clustering strategy can never sink a digest.

  **Expect the theme ranking to change materially on the first wired run**, since the blob
  described below currently leads it at `recurrence: 63`. The branch judge's maturity gate
  (≥2 themes at raw `recurrence` ≥3) reads that same ranking, so the criterion flip may go
  dormant→active or the reverse. Not yet exercised on a live GPU.

- **Persona-digest clustering does not survive scale — measured, and a map-reduce
  replacement added as a dry run.** `reflection_digest.llm_cluster_evidence` puts **every**
  live persona statement into one prompt and asks for one flat partition. At the current 640
  live statements that listing is ~40k tokens against a 24,576-token context, and a correct
  answer would need 640+ group numbers out of a 1024-token budget. What comes back is
  truncated, and `_parse_groups` then completes it by making every unmentioned statement a
  singleton — so the digest silently reads as "one blob theme plus hundreds of one-offs" with
  no error anywhere.

  This is not hypothetical: the shipped digest (`data/persona/20260715-201135`, built when the
  corpus was 121 statements) contains a **130-member** cluster whose members share no
  disposition — intimacy, "forensic drive", and resistance-to-seamlessness folded together —
  alongside 78 singletons, several of which are plain restatements of the blob's own
  representative, including an English near-duplicate that clustered as its own separate
  theme. Because recurrence is counted across cluster members, the blob carries
  `recurrence: 63` and therefore **leads** the evidence ranking and dominates
  `build_digest_prompt_input`. Anything downstream of that ranking — the maturity gate, the
  branch judge's criterion, the portrait itself — has been reading a corrupted ordering.
  Interpretation note for existing artifacts: a large `cluster_size` in any digest written
  before this date is **not** evidence of a strongly-recurring theme.

  New `core/persona_cluster.py` replaces the single flat call with **map-reduce**: group
  within fixed blocks of 40 (prompt and output size constant in corpus size), then re-group
  the resulting themes' *representatives* for up to 5 rounds so a theme split across blocks
  can still merge, rotating the ordered list by half a block per round so each round pairs
  different neighbours. Cross-block merging is best-effort within that round budget — stated
  rather than hidden. A **blob guard** rejects a block whose largest group exceeds half the
  block and degrades it to singletons; the conservative direction is deliberate, since a
  missed merge is recoverable on a later round or run while a false merge fuses unrelated
  dispositions into a theme that then dominates ranking. The guard bounds one *call*, not a
  theme's growth across merge rounds, so the dry-run report prints the largest theme and every
  member of every merged theme. Measured on the live 640-statement ledger with a stub model:
  32 calls, max prompt ~5k tokens, no members lost.

  Also splits the two model roles the live pass conflates. Clustering now runs on the **clean
  base** — deciding whether two statements say the same thing is an evaluation, like
  `fact_dedup` / `self_reconcile`, and the live pass shares one `generate_fn` with the
  synthesis, which lets a drifting adapter group the very evidence that defines it — while the
  portrait is synthesized on the **adapter**, keeping authorship with Ava.

  **Wired as a DRY RUN only** (Sleep tab "Persona digest (dry)" → `digest_dryrun`): it writes
  nothing — no digest snapshot, no `current` pointer move, no RAG refresh — and the live digest
  path is untouched, so this changes no existing artifact yet. Not implemented: incremental
  assignment (judging only statements new since the last digest against existing theme
  representatives), which needs member *keys* persisted on the digest artifact —
  `evidence.themes[].members` currently stores content strings only. Not yet exercised on a
  live GPU.

- **Training review can now *find* the persona-contaminated rows, not just repair them.**
  Retired build-time `[persona]` CoT injection wrote the statement **verbatim** as the leading
  `<think>` line (`persona_render.prepend_persona_lines`), which makes the literal residue
  exactly detectable: the tab scores each entry's leading CoT lines against every `[persona]`
  statement in the raw append-only logs (`weights_persona.jsonl` +
  `consolidation_anchors.jsonl` — evicted ones included, since an injection happened while its
  statement was live) using normalized containment or Jaccard ≥ 0.6, i.e.
  `persona_render._DEDUP_JACCARD` — the *injector's own* "already expressed" predicate, reused
  to recognize what it wrote (**ledger tier**). A second **shape tier** — long first-person
  opener, <25% word overlap with the query, capped at 0.59 so it can never outrank a ledger
  hit — catches the *learned* recitation those rows taught, which is the larger population.
  Both are purely lexical, so they run in the ML-dependency-free client;
  `fact_render`'s deliberate `"I know that …"` line is skipped and does not break the leading
  run. The score drives a threshold filter (default 0.6 == the ledger tier), a **Persona
  score** sort, a `✦` list tag, and a **Strip persona opener** button that *prefills* the CoT
  box with the detected leading run removed, leaving Apply/Cancel to the operator. Strip is
  armed **only** when the hit is the contiguous leading run: the injector prepended onto an
  existing thought, so removing that run provably restores the pre-injection CoT, whereas a
  hit with real reasoning above it could strand a back-reference and is left to the hand
  editor. Scores are recomputed in place after a repair, so a fixed row leaves the filter
  without a reload. **Measured on `build-20260726-040042`** (1003 chat rows): 24 ledger,
  537 shape, 442 clean; 551 of 560 detected rows strippable, 9 entangled. The shape mass is
  too large for hand repair — the render-time strip that would address it is deliberately
  **not built** (see `AVA_OPEN_PROBLEMS.md` → *Persona Formation*).

- **Training review gains a hand-edit repair route and a frozen filter.** Repair had exactly
  one route — have the model re-answer the exchange — which leaves a row the adapter keeps
  getting wrong (or one needing a one-word fix) stuck in a regenerate-until-lucky loop. The
  CoT and answer boxes are now **editable** for any row with chat provenance, with an
  **Apply** / **Cancel** bar under them: Apply writes what is in the boxes through the *same*
  server path as the regenerate flow's Apply (`apply_regenerated_exchange`, `corrupt_cot` +
  `corrupt_response` both set), so a hand edit lands as one reviewed `manual_regen` target and
  **locks (❄ freezes) the exchange** exactly as a regeneration does — re-reflection and Revisit
  preserve it. No server or protocol change; the CoT + answer are always written together
  (never the CoT-only graft) because the operator has both in front of them. Unsaved edits are
  tracked per *entry* (by its `_order` stamp, not its row) so a re-sort can't strand them, and
  switching entry / refreshing prompts before discarding. Alongside it a **frozen filter**
  (All / Only ❄ frozen / Only unfrozen) composes with the existing CoT-reply-ratio filter and
  is the repair workflow's progress view: under *Only unfrozen* a successful Apply drops the
  repaired row out of the list and advances the selection to the next entry awaiting review.
  Client-only (`client/ui/training_review_widget.py`).

- **Training-review Regenerate now forces the thought opener (and stops early on CoT-only
  regen).** The Regenerate flow re-answers a logged exchange through the reflection IDEAL seam
  (`generation._make_sync_reflect_generate`), which — unlike the live-chat `_run_generation`
  path — omitted the two mechanisms that guarantee a CoT: the family think-prefill
  (`fam.think_prefill`) and the minimum-thought-length floor (`min_think_tokens`). Because the
  IDEAL seam rebuilds a **CoT-stripped** prior conversation, the sampled `<think>`-opener
  probability drifts toward zero (the same multi-turn collapse chat fixed with prefilling), so
  a regeneration on gemma-4 sometimes came back with **no thought block** — useless when the
  operator's goal is repairing a corrupt CoT. `_regenerate_exchange_sync` now passes
  `force_think=True` unconditionally (the re-answer always carries a CoT; `_clean_response`
  normalizes the gemma channel form as before). Additionally, when the operator checked
  **Corrupt CoT** *without* **Corrupt reply** (`cot_only`), the regenerated reply is discarded
  by Apply (it grafts only the new `<think>` onto the kept reply), so generation now halts the
  moment the reasoning channel closes (`fam.close_markers`) instead of spending the token
  budget on a reply nothing reads. Scoped via new opt-in `force_think`/`stop_after_think`
  params on the reflect-generate seam, so reflection/wander/IDEAL callers are unchanged.
  Threaded client→server as `regenerate_exchange`'s `cot_only` flag.

- **LoRA scaling switched to rank-stabilized (`use_rslora=True`, fixed `alpha=4`) — `lora_r`
  is now an LR-neutral knob** (`5681dc8`). `train_cycle` fitted the from-scratch adapter with
  `lora_alpha == r`, pinning peft's γ = α/r at **1.0 for every rank**. That was documented
  (`training/DESIGN.md` → *LoRA hyperparameters & LR schedule*) as *implicit regularisation*:
  "doubling rank distributes the same gradient signal across twice as many parameters, so each
  parameter moves roughly half as far per step." **That reasoning was wrong**, and this entry
  exists mainly to record why.

  It holds under SGD with a fixed total gradient. It is false under Adam, which moves each
  parameter by ~lr per step regardless of how many parameters there are. B initializes at zero,
  so the weight delta is `dW ≈ γ · dB @ A`, and every entry of that product is a sum over **r**
  terms. With γ held at 1.0, ‖dW‖ therefore grows like ~√r — **raising the rank silently raised
  the effective learning rate on the weights.** An r=32 build was expected to be gentler than
  r=16 and instead came out *less stable*: that is a ~1.4× step-size bump, landing on top of the
  per-row wall-clock multipliers (up to 4× at `lora_cap_age_h`) and the `triangular` plateau,
  so it concentrated on exactly the cap-age rows that shape voice.

  γ ∝ 1/√r cancels the √r growth exactly (Kalajdzievski 2023). `lora_alpha` is now a fixed
  constant **decoupled from `r`** (`decay.TRAIN_LORA_ALPHA` = 4 == √16), so γ = α/√r, calibrated
  so **γ == 1.0 at r=16**.

  **How to read existing artifacts:** an **r=16** build after this date is scaling-identical to
  every adapter built before it — the tuned `train_lr` = 8e-6 baseline is unaffected, and no
  build in the existing lineage needs reinterpreting. Only **non-16 ranks** change meaning: a
  pre-cutover r=32 adapter was trained at ~1.41× the effective LR its `train_lr` implies (r=8 at
  ~0.71×), so a `builds.jsonl` line recording `lora_r` ≠ 16 from before this date is not
  comparable to one after it at the same `train_lr`. Post-cutover ranks are LR-matched to the
  r=16 point (r=8 → γ 1.414, r=32 → 0.707, r=64 → 0.5, r=128 → 0.354), so what a rank change
  now buys or costs is capacity alone.

  **No migration.** peft records `use_rslora` in each `adapter_config.json` and recomputes γ
  from it at load, so pre-existing adapters (flag absent → False, `alpha == r`) keep their
  original α/r == 1.0 scaling and load unchanged. Not yet run end-to-end on GPU.

## 2026-07-25

- **RAG injection cap per recalled turn raised 1800 → 4000 chars** (`dbcf6d3`,
  `rag_engine._CHAT_TURN_DISPLAY_CHARS`). This bounds only **injection**, never retrieval:
  a turn is embedded as 280-char overlapping passages regardless, so the cap never made
  anything unfindable — it decided how much of an exchange that had *already won* reached
  the prompt, via `rag_policy.clipped` (which appends the visible `...[truncated]` marker).

  The old value was set against much shorter turns than this corpus produces: 37% of its
  2494 logged turns exceed 1800 chars (median 707, p75 3005, p90 3900, max 16662), so a
  long recalled turn was head-clipped past its midpoint — losing the *end* of the thought,
  which is where a conclusion tends to sit. At 4000 the clipped share falls to 9% and the
  mean rendered side grows 960 → 1511 chars.

  **How to read existing artifacts:** a `system_content` / `rag_context` logged before this
  date carries 1800-char sides, so recall in those exchanges is *more* clipped than the same
  retrieval would be today — when a pre-cutover reply looks like it ignored the tail of a
  recalled turn, the tail may simply never have reached it. The marker is verbatim in the
  transcript, so affected turns are greppable (`...[truncated]`); 234 of 983 RAG blocks on
  disk carry at least one. The Chat review tab shows them in place.

  **Budget cost:** a typical chat block (`_DEFAULT_TOP_K` = 3 exchanges × 2 sides) grows
  ~+3.3K chars (~+1.3K tokens); worst case ~10.8K → ~24K chars (~4.2K → ~9.4K tokens)
  against a 24576-token window. Observed prompts are median 5724 / p90 12505 / max 21809
  tokens, and a prompt that fills the window raises a hard `RuntimeError` rather than
  trimming history — so the longest conversations now sit closer to that edge. The wander
  channel's separate `_WANDER_REACTION_DISPLAY_CHARS` = 1800 is unchanged.

- **Fact attribution (`about` / `source`) + hearsay gate — third-party recall made deliberate.**
  A `[fact]` now records **two** people, which only come apart when Ava speaks with one person
  about another: `about` (who it concerns, model-supplied via `(about: NAME)`) and `source` (who
  said it, supplied in code from the session record so provenance cannot be hallucinated). The
  pair derives `source_class` — `self` / `hearsay` / `observed`.

  This closes a **silent misattribution bug**, not just a missing feature. Reflection memory
  carried no speaker at all (`_build_reflection_index` stored key/kind/display/modifier/
  source_session/available_at), while the verbatim-chat channel that *does* render a `speaker:`
  prefix reaches a hard zero at the 96h cap. So past four days, every recalled fact about a third
  party arrived as bare prose with the person in front of Ava as its only available referent — the
  system degraded from attributed to unattributed memory over exactly the window where it mattered.
  Recall now renders `— about X` / `— about X, per Y` (`_attribution_label`).

  **Hearsay gate:** a hearsay fact is written to RAG (fully recallable, attributed) but held out of
  `weights_persona.jsonl`, so it raises no ledger anchor, is never host-CoT injected as
  `"I know that …"`, and never trains. It promotes the ordinary way when its subject later states it
  themselves (a `self` record). Rationale: without the gate, A's account of B launders into a
  timeless never-fading truth about B that B never said — and `fact_contradict`'s mechanical
  newest-wins would then let A's version supersede B's own.

  **How to read existing artifacts:** every record written before this date has no `about`/`source`,
  so it derives `observed` and keeps its previous weights path and un-suffixed recall line exactly.
  A pre-2026-07-25 `[fact]` about a third party is therefore indistinguishable from a self-report —
  attribution is not retroactive, and the corpus mixes both until those chats are re-reflected.

- **Fact contradiction clustering is now partitioned by subject person.** `cluster_by_subject`
  groups by `about` before the recall-cue embedding. Two people's cues are near-identical for the
  same topic ("Boris's dog" / "Artemy's dog"), so an unscoped cluster mixed subjects and newest-wins
  superseded one person's true fact with another's unrelated one — a live cross-contamination in a
  pass that already ran automatically after `commit-training`. Unattributed facts share one
  partition and behave as before, so this only narrows what may be compared. Any `supersede` op
  written before this date by the automatic (B) pass may have crossed subjects; they are reversible.

- **Disclosure norm: learned disposition, not an access rule.** Ava may speak to one person about
  another — memory stays one shared store, deliberately (a per-person sandbox would make her a
  service with multiple sessions rather than one subject with one memory). Discretion lives in
  `chat_prompt.txt` / `rag_memory_prompt.txt` as something she develops and can get wrong, not as a
  `private` flag filtered at retrieval. No mechanism measures or enforces it yet: there is no
  confidence signal, no record of what she disclosed to whom, and so no path by which a regretted
  disclosure becomes persona evidence. Tracked as `AVA_MEMORY.md` Open Issue G.

- **Unrelated pre-existing fix:** `training/selftest.py` asserted `CHAT_SCHEMA_VERSION == 4`, stale
  since the schema went to 5 with *Rewrite history*. It aborted the suite before ~30 later tests.
  Corrected to 5. `test_regression_probe` still fails on a clean tree (stub-model tiers 1/2/5) and is
  left alone — validation is a separate, currently-disabled design project.

## 2026-07-24

- **Un-freeze a reflected chat when it's continued in place (Ava-initiated chats).** A chat
  frozen by reflect-once (`reflected_at` / `chat_reflected`) is skipped by `list_backlog` and
  the normal run forever, so turns appended AFTER the freeze never reflect or train. Normal
  chats are safe (continue forks a fresh `continued_from` file), but an Ava-initiated outreach
  is resumed **in place** (`load_session in_place=True` → `ChatLogger.resume_session`) to keep
  the reversed opener+reply+response as one unit — so reopening one after its background
  reflection froze it and replying again silently lost those turns. Fix: `handle_load_session`
  (in-place branch) now un-freezes on continue — `ChatSidecar.clear_reflection_freeze` clears
  both stamps and, for a `chat_reflected`-only chat, `reflection_staging.purge_background_
  artifacts` deletes its `pending_clean_base` jobs + checkpoint sidecar mirror (otherwise the
  next normal run's `fold_checkpoint_to_live` would copy the stale frozen sidecar back and
  silently re-freeze it). Per-exchange verdicts / `locked` flags / consolidation summary are
  left intact (re-reflection overwrites targets and honours locked exchanges); the checkpoint's
  cumulative memory/ledger deltas are left too (they dedup by `content_key` against the
  re-reflection, so they self-heal). `session_ops.configure` now also takes the reflection
  `data_dir` (distinct from the `server/data/chats` root) for the purge. Files:
  `server/inference/core/{chat_sidecar,reflection_staging,session_ops}.py`,
  `server/inference/server.py`. Commit: pending.

- **Deliberation pass (worklog read side), dry-run + Worklog-tab "Deliberate" button.** The
  executive step the worklog was built for: `core.deliberation.run_deliberation_blocking`
  reads Ava's recent episodic worklog (`worklog.recent(N)`, default 20) + open threads,
  injects her current persona portrait + temporal anchor, and runs one reflect-generate
  decision pass (`deliberation_prompt.txt`, thinking on, RAG off) → `ACTION:` (wander /
  synthesize / reach_out / revisit / nothing) + `WHY:`, parsed over the post-CoT answer with
  the hardened `_answer_after_think`. **Dry run / shadow:** it decides and streams her
  reasoning but **dispatches nothing** — the result reports the action she WOULD take
  (`would_do`), so the operator can judge whether the choices are sane before a later task
  wires `execute`. Manual-only (`deliberate_now` → `handle_deliberate_now`, the Worklog tab's
  **Deliberate** button, streaming `deliberation_stage`/`deliberation_chunk` →
  `deliberation_done` into a bottom panel). Owns `_deliberation_active` (in the scheduler's
  `external_busy`); never imports `server`. No autonomous idle job yet, and no dispatch —
  coexistence with the independent per-action idle jobs (making deliberation the sole
  executive) is a deliberately separate step; the future agentic actions (e.g. chat-history
  search) slot into the `_ACTIONS`/`_WOULD_DO` maps with the loop unchanged. Files:
  `server/inference/core/deliberation.py` (new), `deliberation_prompt.txt` (new),
  `server/inference/server.py`, `client/ui/worklog_widget.py`, `client/core/backend_client.py`.
  GPU-free self-test: `python -m core.deliberation`. Commit: pending.

- **Synthesis ("Chat reach out") leaked a whole CoT block into chat as the opener.** Confirmed
  from a live debug log: a truncated opener-compose reached the user with its entire reasoning
  block. Chain: the opener pass ran with a **1024-token cap**; gemma-4's opener CoT (weighing
  drafts/tone/structure) ran past it, so the reasoning **channel never emitted its closing
  `<channel|>`**; `model_family._normalize_gemma` can only rewrite a *closed* channel, so the
  stray-marker strip left **untagged thought — no `<think>`/`</think>` at all**; then
  `_parse_opener`'s `OPENER:` search ran over that raw reasoning and matched a **spurious
  label** the CoT contained about itself (`Format: \`OPENER: <message>\`` plus draft openers),
  so it returned a *non-empty* opener full of reasoning — bypassing the old empty-only
  `last_truncated` guard. Fix (`synthesis.py`): key the guard on the **reasoning/answer
  boundary**, not on the fallback path — if the compose was truncated AND produced no
  `</think>` boundary (i.e. the model never closed its CoT to write a real answer), refuse it
  (`opener_truncated`); a closed boundary is trusted even if generation later hit the cap,
  since `OPENER:` is then parsed from the clean answer region, not the reasoning. Plus a
  `_has_reasoning_leak` backstop (`opener_leak`) that refuses any opener still carrying a
  `<think>`/`</think>`/`<|channel>` marker, and `_answer_after_think` hardened to strip a bare
  closing `</think>` and stray gemma channel tokens. **Also raised the opener cap 1024 → 2048**
  (matching the analysis pass) so the channel usually closes and a real opener is produced
  rather than skipped. Outreach/check-in were NOT affected — they leave the opener empty when
  the `OPENER:` label is absent (no whole-answer fallback). Unrelated to the summary-pass
  commit that preceded the report. Known broader exposure (not fixed here, lower risk):
  `_normalize_gemma` leaving untagged thought on any truncated channel could similarly mislead
  label parsing in other passes; hardening the normalizer to wrap an unclosed channel is a
  candidate follow-up. Files: `server/inference/core/synthesis.py`. Commit: pending.

- **Root cause of summary degeneration: the SUMMARY pass ran under the CONSOLIDATION
  prompt.** The per-chat consolidation-gist recap was frequently emitting raw consolidation
  format instead of prose — a clean recap followed by `\n\n***\n\n## WEIGHTS\n- [fact] …`
  (75/108 marker-containing sidecars), or the structured dump from the first line (33/108),
  or `<eos>` (9). Cause: the summary pass reused the consolidation `prepared` prompt object to
  save re-packing the transcript, but that object bakes in `sleep_prompt` (the consolidation
  instruction) **and** the reflection-memory RAG block — and `generation.generate_fn` /
  `reflection_run._make_generate_fn` **ignore the `system_prompt` argument entirely whenever
  `prepared_prompt` is supplied** (`prepared = prepared_prompt or prepare_prompt(...)`). So the
  recap was literally generated under the consolidation prompt (the `summary_prompt` passed
  alongside was dead), and the injected `[fact]`/`[ask]` RAG lines few-shot-primed exactly the
  structured output. Fix: `reflection_runner` now rebuilds the prompt for the summary pass with
  `summary_prompt` and **RAG disabled** (a transcript recap needs neither the consolidation
  instruction nor the structured memory that biases it), reusing only the already-packed
  `content`. This cleans BOTH the worklog "about X" and the RAG **gist** channel (both read the
  same `consolidation_summary`), going forward. Files:
  `server/inference/core/reflection_runner.py`. Commit: pending.

- **Worklog `conversation` "about X" is sanitized before use (defense-in-depth).** Complements
  the root-cause fix above for any residual degeneration and for already-frozen historical
  sidecars. `background_reflection._summarize_gist` **salvages the prose prefix** of a summary
  that degenerated into a trailing consolidation dump (keeps everything before the first
  `***` / `## WEIGHTS` / bulleted `[fact]` boundary — 130/173 real sidecars now yield clean
  prose, up from 51), drops a leading narrative header (`### Reflection`) and any prose after
  `<eos>`, collapses to one line, and clips to a sentence-ish snippet; a fully-structured or
  `<eos>`-only summary salvages nothing and falls back to the plain "I talked with {user}
  (N exchanges) …" template. Files: `server/inference/core/background_reflection.py`.
  Commit: pending.

- **Anti-copy guard: previous reply protected from verbatim regurgitation.** Operator
  observation: at t=1.0 the chat occasionally reproduces the previous assistant reply
  **verbatim** while its CoT is fresh and on-topic for the new user input. Analysis: this is
  an induction-copy attractor, not an assembly bug (the prior reply appears exactly once in
  the prompt; the active session is RAG-fenced). Once the first few answer tokens happen to
  match the prior reply's opening, copying a context span becomes near-deterministic even at
  t=1.0 — and every existing guard is structurally blind to it: `_RepetitionStop` watches
  within-generation recurrence (a single prompt-copy pass never recurs), `_DegenStop` watches
  token-diversity collapse (copied text is healthy prose), the generated-only repetition
  penalty exempts the prompt by design (and worse, tilts the answer's first token *toward*
  the copy: the CoT rehearses the fresh answer's vocabulary, which is then penalized, while
  the old reply's wording sits exempt in the prompt), and `min_p` locks the copy in once
  entered. The fresh-CoT/copied-answer split follows from CoT-stripped history: the model's
  in-context exemplars are thinkless `(user, answer)` pairs, so the answer channel anchors on
  the previous turn rather than its own CoT. Fix — `stream_generate(no_copy_text=…)`, fed by
  every chat-shaped caller (chat / ephemeral / encounter / gossip via
  `generation._last_assistant_content`) with the previous assistant reply, arming two
  complementary mechanisms: **(1)** the reply's token ids are unioned into the
  `_GeneratedOnlyRepetitionPenalty` gather (penalized as if already generated), restoring —
  for exactly that text — the copy deterrent the prompt exemption removed, while system
  prompt / RAG blocks / older history stay exempt; **(2)** a `_NoCopyPrevReply`
  `LogitsProcessor` masks (-inf) any token that would extend a verbatim
  ≥`_NO_COPY_NGRAM` (8)-token match with a span of that reply — a hard cap a 1.1 penalty
  alone cannot provide against a near-deterministic copy distribution — leaving paraphrase
  untouched (only an exact token-level span match arms the mask, and only the observed
  continuation ids are banned). Both read only the generated suffix; reflection/revision
  paths are untouched (param defaults off). Pure helpers `build_no_copy_table` /
  `banned_continuations` join the GPU-free self-test (`python -m core.inference_backend`).
  Files: `server/inference/core/inference_backend.py`, `server/inference/core/generation.py`.
  Commit: pending.

- **Repetition penalty scoped to generated tokens (prompt exempt).** Operator observation:
  live chat collapses far more often with RAG enabled, the facts channel worst, and the
  collapse is temperature-sensitive. Analysis found a stacked cause; the first (and cheapest)
  fix landed: transformers' native `repetition_penalty` kwarg
  (`RepetitionPenaltyLogitsProcessor`) penalizes every token present in `input_ids` — the
  **prompt included** — so an injected RAG block simultaneously *primed* Ava's
  highest-probability continuations (facts are distilled in her own idiolect; history excerpts
  contain her verbatim past replies) and *suppressed* those same tokens, squeezing probability
  mass into the degenerate tail — manufacturing the very "penalty converts a verbatim loop
  into a drifting associative walk" pathology the Layer-2 degen guard exists to catch.
  `stream_generate` no longer passes the native kwarg: a custom
  `_GeneratedOnlyRepetitionPenalty` `LogitsProcessor` applies the identical multiplicative
  arithmetic (positive logit ÷ penalty, negative × penalty) gathered from the post-prompt
  suffix only, so the guard discourages the model from repeating its OWN output, never from
  using vocabulary present in the system prompt / conversation / RAG blocks. No-op on the
  first generated step; native kwarg retained solely as a fallback when the
  `LogitsProcessor` classes can't be imported. All `stream_generate` callers (chat /
  ephemeral / encounter) inherit the new scope; config knob `chat_repetition_penalty`
  unchanged. Remaining suspects from the same analysis (train/serve prompt mismatch — the
  adapter never trains on RAG-shaped prompts; the reflection channel's loose `0.25` floor +
  never-fading, near-duplicate facts re-injected every turn) are deliberately untouched.
  Files: `server/inference/core/inference_backend.py`. Commit: pending.

- **Chat "Retry" button — roll back a collapsed reply for a resend.** The completed-turn
  sibling of the in-flight Stop/discard: after a reply *lands* (typically a degenerate /
  collapsed generation), **Retry** asks the server to pop the latest completed exchange off
  the active transcript (`ChatLogger.remove_last_exchange`) AND the in-memory
  `_session.conversation` (trailing assistant + its user turn), then hands the removed
  `user_prompt` back so the operator can adjust Temperature and send again — so the discarded
  reply never reaches reflection or training. New message `retry_last_exchange` →
  `session_ops.handle_retry_last_exchange` (reply `retry_ready {user_prompt, speaker,
  exchange_index}`), refused on a live reflection run, an already-reflected session, an empty
  session, or a synthetic Ava opener (`initiated_by:"ava"` with no real user turn). No RAG
  unwind is needed: the active session is fenced from the chat index (`add_exchange` no-ops
  when `source_session == current_session`), so the removed reply was never indexed. An
  emptied transcript is left on disk as a harmless exchange-less file (the next `log_exchange`
  refills it), keeping the logger `current_file` + RAG fence stable across the retry. Client:
  a `Retry` button gated on a tracked "latest completed live/continued turn" position that is
  invalidated by any new send / new-chat / preview / continue / disconnect; on `retry_ready`
  it excises the rendered turn from the chat log and restores the prompt. Files:
  `server/inference/core/chat_logger.py`, `server/inference/core/session_ops.py`,
  `server/inference/server.py`, `client/core/backend_client.py`,
  `client/ui/chat_widget.py`. Commit: pending.

- **Worklog `conversation` entries via background per-chat reflection.** A live user
  transcript has no clean "close" to hang a worklog entry on, so the natural trigger is when
  Ava *processes* the chat: the background per-chat reflection pass now records one
  first-person `conversation` entry as each chat is frozen `chat_reflected`
  (`background_reflection._record_conversation_worklog`, fired from `_on_chat_reflected`). The
  "about X" reuses the per-chat consolidation **gist** just written to that run's staging
  sidecar (`consolidation_summary.text`) → "I talked with {user}. {gist}"; absent a gist it
  falls back to a bare template with the exchange count. **Includes an Ava-initiated chat the
  user replied to** — outreach/synthesis logged only the *opener*, so the ensuing
  back-and-forth is a real conversation, and its entry **closes** that opener's open thread
  (matched by `refs.session`). Only two shapes are skipped: a still-unanswered opener
  (`_is_unanswered_outreach`, ≤1 exchange — also pre-filtered by `list_backlog`) and an
  `interlocutor:"ai"` transcript (encounters / served gossip — not the user). Best-effort;
  never disturbs the reflection. Foreground normal Sleep runs still emit only the aggregate
  "I reflected on N conversations" entry. Files:
  `server/inference/core/background_reflection.py`. Commit: pending.

- **First-person episodic worklog (`core.worklog`) + read-only Worklog preview tab.** A new
  durable, semantic record of what Ava *did* — one entry per meaningful episode in her own
  voice ("I reached out to Artemy about the worklog idea"), distinct from the machine-phrased
  activity ring (`core.activity_log`): keep-forever (not a ring), first-person, each entry an
  INDEX into the scattered traces (`refs`: chat stem / wander title / ask key / run id) rather
  than a copy. A **leaf** like `reachout_gate`/`activity_log` (imports nothing from the
  project, `configure(path)` once, `from core import worklog; worklog.record(...)` at each
  episode-close, no wiring, no cycle) — and it never generates: the caller supplies the
  first-person `summary` (templated for now; a real generation at chosen sites later). Carries
  a light op-log fold — an episode may `opens` a loop that a later one `closes` by id, folded
  by `open_threads()` (the "what's still hanging" state a future planner needs, mirroring
  `reflection_memory.open_questions`). **Record sites wired now:** outreach (compose + the
  self-answered `resolved` case), synthesis compose, check-in compose, autonomous wander
  finish, and normal/revisit Sleep-run completion (one entry per run, inside the `success`
  branch — background per-chat and dry runs do not record). **Nothing consumes it yet** — the
  deliberation/action read side is a deliberately separate task; this change only *produces*
  worklog entries and *previews* them. Client: a read-only **Worklog** tab polls the new
  `get_worklog {after_seq?}` → `worklog_batch {entries, latest_id, open_threads}` with one id
  cursor (live, independent of any run, start/stop with the socket — same lifecycle as the
  Activity tab), rendering open threads + the chronological episode log so an operator can
  watch entries land during a real run. Storage: `data/hot/worklog/worklog.jsonl` (sibling of
  `hot/activity/` and `hot/memory/`, gitignored). GPU-free self-test: `python -m core.worklog`.
  Files: `server/inference/core/worklog.py` (new), `server/inference/server.py`,
  `server/inference/core/{outreach,synthesis,checkin,til_wander,reflection_service}.py`,
  `client/ui/worklog_widget.py` (new), `client/ui/main_window.py`,
  `client/core/backend_client.py`. Commit: pending.

- **Removed the chat-count RAG penalty.** Verbatim chat relevance is again a function of
  semantic similarity and wall-clock age, not of how many other conversations were opened
  afterward. The `chats_since_weight` helper, `WallClockConfig.chats_since_*` fields,
  per-index chat-rank bookkeeping, entry metadata, settings controls, and self-test
  assertions are removed. The hourly `fresh_time_weight` adjustment remains unchanged:
  an unfrozen chat droops from `1.0` to `0.875` over its first 24h, while reflected
  verbatim continues its linear fade to hard `0` at 96h. Older configs containing
  `fresh_window.chats_since_penalty` / `chats_since_floor` remain loadable; those unknown
  keys are simply ignored. Files: `server/training/decay.py`,
  `server/inference/core/rag_engine.py`, `server/training/selftest.py`,
  `server/settings.py`, `documentation/AVA_MEMORY.md`, `documentation/AVA_DESIGN.md`,
  `documentation/AVA_STATUS.md`. Commit: pending.

## 2026-07-23

- **Verbatim→gist RAG handoff restored: raw chat expires at 96h; gist settles at 0.2.**
  Verbatim chat had been coupled to `rag_floor_weight`, so its modifier stopped at `0.2`
  and the raw exchange remained retrievable forever alongside the summary. The channels are
  now separate: new pure helper `verbatim_rag_weight_hours` fades reflected verbatim
  `1.0→0` over `rag_cap_age_h=96h`, while `_chat_modifier` enforces the same raw-age hard
  cutoff even if reflection/training lagged. Query-time modifier recomputation makes the
  deadline real on a long-running server rather than waiting for an index rebuild. Persona
  keeps the existing `rag_weight_hours` / `rag_floor_weight=0.2` path; facts remain at
  `1.0`. Gist still rises `0→1.0` over 0–96h, then now uses a true affine interpolation
  `1.0→0.2` over 96–192h, reaching the floor exactly at 192h and holding forever (the prior
  zero-target clamp reached its floor early). Committing a summary sidecar refreshes the
  serving chat index immediately, so a no-train Sleep run does not wait for restart before
  the new gist exists. Focused GPU-free curve tests cover 0/48/72/96/144/192h and the held
  tail. Files: `server/training/decay.py`, `server/training/selftest.py`,
  `server/inference/core/rag_engine.py`, `server/inference/core/reflection_service.py`,
  `server/settings.py`, `server/inference/server_config.json`, maintained docs and operational maps. Commit:
  `4ced56f`.

- **Facts no longer fade in RAG — a fact is a timeless truth, held at weight `1.0` for life.** The reflection-memory fold (`rag_engine._build_reflection_index`) applied the source-bundle wall-clock crossfade (`rag_weight_hours`, `1.0→rag_floor_weight` at `rag_cap_age_h`) to `[fact]` **and** `[persona]` alike, so a fact the user stated last week already sat at the 0.2 floor and ranked below anything fresher. That is wrong for a fact: unlike episodic dialogue (which recedes) or persona (which should be re-earned), a fact's truth value does not decay with the clock — recalling it in six months is exactly as correct as tomorrow — and, since a nightly LoRA does not reliably memorize specific facts, fading a fact from RAG while it is not reliably in the weights was genuine amnesia of a true datum. Fix (one lever): `_build_reflection_index` now gates only `kind == "persona"` through the crossfade; facts keep modifier `1.0`. `_consolidation_modifiers` is narrowed to persona-only to match (its sole consumer). **Interpretation change this forces:** fading was silently retiring *stale/false* facts (sinking them to the floor over 96h so fresher material outranked them); with never-fade, retiring a stale fact becomes an **explicit-eviction** problem, and there is no contradiction-driven fact-eviction path yet — a corrected fact and its stale predecessor now coexist at equal weight until something supersedes the old one. Recorded as a new open problem (`AVA_OPEN_PROBLEMS.md` → *Fact Staleness*) and in the new cornerstone `AVA_MEMORY.md` (§3.4 commitment + §7 F). Persona, gist, verbatim-chat, wander, and all training LR are untouched. No test asserted fact fade (the two methods are internal to `rag_engine`, called only at index build); syntax-checked. Files: `server/inference/core/rag_engine.py`, `documentation/AVA_MEMORY.md` (new), `documentation/AVA_OPEN_PROBLEMS.md`, `documentation/AVA_STATUS.md`, `documentation/README.md`. Commit: pending.

- **Chat crashed on a null degen-guard threshold (`int < None` in `_DegenStop`).** The `chat_degen` config block's threshold reader copied a key's value verbatim whenever the key was *present* (`if _ck in _degen_cfg`), so a hand-edited `"chat_degen": {"window": null}` — the natural way to write "default, please" in a config whose scalar knobs (`chat_repetition_penalty`, `chat_min_p`) document null-to-disable — injected `None` into `stream_generate`'s stopping criteria and crashed **every** live chat generate mid-stream (`TypeError: '<' not supported between instances of 'int' and 'NoneType'` at the `n_gen < degen_min_gen` gate). Two-sided fix: `server.main()`'s builder now skips null entries (null threshold ⇒ module default; disabling the guard remains `chat_degen_guard: false`), and `inference_backend.stream_generate` coerces each `degen_*` threshold to its module default when `None` before building `_DegenStop` — so no future caller can reintroduce the crash. Files: `server/inference/server.py`, `server/inference/core/inference_backend.py`.

- **Chat RAG embeds both dialogue sides — Ava's own replies are now searchable.** The chat index embedded only the *user prompt* of each exchange (`_collect_chat_entries` / `add_exchange` chunked `user` alone), so Ava's half of every past conversation was invisible to retrieval: "what did you tell me about X?" could only hit if the user's past phrasing happened to match, and anything Ava explained unprompted was unrecallable verbatim (the gist channel partially covers this, but only for reflected chats and in distilled form). New shared helper `RagEngine._exchange_passages(user, response)` produces bounded passages of BOTH sides, each tagged `embed_source` (`user`/`response`), used identically by the rebuild path and the incremental `add_exchange` path (which already receives the CoT-parsed answer, so the reasoning channel stays excluded by construction). A hit on either side collapses to the same displayed exchange via the existing `(source_session, exchange_index)` dedup, so recall widens with no duplicate injected text; the 0.90 near-duplicate ceiling applies to response hits too (a live query near-identical to a past reply is the same verbatim-copy trap). The no-fence FAISS oversample grows `top_k*8`→`top_k*16` since one exchange now holds roughly twice the passage vectors (flat index scans every vector regardless of k — free). Index size ~doubles; CPU-embedded, small corpus. Selftests: `test_rag_engine_query_policy` gains the two-sided + collapse-on-response-hit assertions; `test_rag_engine_fallback` updated for per-exchange passage pairs (verified locally under stubbed faiss/sentence-transformers). Files: `server/inference/core/rag_engine.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`.

- **RAG retrieval no longer forgets the corpus: raw-relevance gating everywhere + fade floors (the "Ava misses previous dialogs" fix).** Two compounding retrieval bugs made almost all past conversations and distilled memory unretrievable. (1) The chat, gist, and reflection-memory channels gated on `cosine × modifier ≥ floor`, so the wall-clock fade silently *raised the semantic bar* as an item aged — with the chat floor 0.45 and the 0.90 near-duplicate ceiling, any bundle below modifier 0.5 (≈48h for a frozen chat) had an empty pass band: a hard forget, not the intended de-prioritization. This is the exact pathology already diagnosed and fixed on the wander channel (`wander_rank_score`, "the 0.1-weight step was mathematically impossible to retrieve"); the other channels never got the fix. (2) `rag_weight_hours` faded frozen bundles to a hard 0 at `rag_cap_age_h` (96h) — dropping them from the index wholesale — and `gist_rag_weight_hours` expired the gist tent at 192h, on the premise that the content now "lives in the weights"; a nightly LoRA does not reliably memorize dialog facts, so weight 0 was amnesia. Census on the live snapshot (2026-07-23, 314 chats, adapter trained nightly): **309/314 chats unretrievable** (192 out of the index, 117 in a dead band), all 158 gists needing cosine ≥ 0.70, and **691/788 (88%) fact/persona anchors** dropped from the reflection index. Fix: `rag_policy.wander_rank_score` → **`rank_score`** (generalized) applied to `_query_chat` / `_query_reflection` / `persona_keys` — relevance gates on the RAW cosine (unchanged floors 0.45/0.25), the decay modifier only orders what passed (the 0.90 chat ceiling still applies to the raw score); and two new `WallClockConfig` knobs: **`rag_floor_weight`** (default 0.2 — the verbatim/anchor fade lands there at `rag_cap_age_h` and holds) and **`gist_floor_weight`** (default 0.3 — the gist tent decays to it at `gist_cap_age_h` and holds, floored ABOVE verbatim so an aged chat is preferentially recalled through its summary; either at 0.0 restores the legacy hard drop). Post-fix census: 314/314 chats, 150/158 gists (the rest are fresh gists still ramping — correct), 788/788 anchors retrievable at a typical 0.60 cosine, with fresh material still outranking aged at equal relevance. Training is untouched (`lr_multiplier_hours` unchanged); `_collect_chat_entries`' `modifier <= 0` short-circuits remain as the floor-0 legacy path. Selftests updated (`test_wall_clock_age` gains gist-tent + floor asserts; `test_rag_crossfade` asserts the floor). Remaining known gaps (not in this change): only user prompts are embedded in the chat index (Ava's own replies unsearchable verbatim), the retrieval query is the bare current user message, and pre-feature frozen chats still have no gist to be recalled through. Files: `server/inference/core/rag_policy.py`, `server/inference/core/rag_engine.py`, `server/training/decay.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`.


- **Background reflection no longer re-reflects the same chats every wake.** `background_reflection.list_backlog()` decided eligibility by reading `is_chat_reflected` off the **live** `_CHATS_DIR` sidecar — but the background pass never stamps the live sidecar: it freezes each completed chat `chat_reflected` in a throwaway staging dir and commits that frozen sidecar into the durable **checkpoint** (`commit_background_chat`), and the live stamp lands only when a *normal* reflection run folds the checkpoint (`fold_checkpoint_to_live`). So between background wakes — with no operator Sleep run in between — every chat the pass had already completed still looked unreflected, and each idle wake re-picked the whole backlog and re-reflected it (wasting GPU and re-appending duplicate memory/consolidation deltas into the checkpoint every time). The self-test masked the bug: its mock manually stamped the *live* sidecar, which production never does. Fix: `list_backlog()` now also excludes chats already completed in a prior wake by consulting the durable markers — a `chat_reflected` sidecar in the checkpoint chats dir, or a `<stem>.json` in the sibling pending-clean-base dir (`_checkpointed_stems()`). The self-test was rewritten to freeze inside the throwaway staging (as `ReflectionRunner` does) so its second-wake assertion is now a real regression guard. GPU-free verified: `python -m core.background_reflection`. Files: `server/inference/core/background_reflection.py`, `documentation/AVA_CHANGELOG.md`.

- **Activity mirror gains a second grain: background reflection now shows real reflection detail (chat name + pass output) in the Activity tab.** The unified activity journal's reflection mirror (`ReflectionRunStore._mirror_to_activity`) deliberately mirrored only coarse phase/session markers, because an operator run has the Sleep tab's detailed console attached. But a **background per-chat run** (`core.background_reflection`, `source=="idle"`) has NO Sleep console — the Activity tab is its only window, and at phase grain the operator couldn't see *what* Ava concluded about a chat. The mirror is now keyed on the run's `source` (grabbed under the store lock in `append_event`): operator runs keep the coarse whitelist (`_ACTIVITY_MIRROR_EVENTS`, unchanged — no Sleep-tab duplication), while a background run mirrors at Sleep-tab detail — (1) a wider whitelist (`_ACTIVITY_MIRROR_EVENTS_BACKGROUND` adds `branch_started`/`branch_done`/`branch_skipped`/`pass_error`/`ask_resolved`); (2) every line is Sleep-tab-shaped: `[etype] <chat name> <message> (exchange x/y)`; (3) `phase_done` carries the pass's actual output indented under the line — revision's raw VERDICT-WHY-IDEAL `text`, and for consolidation (whose text streams as `phase_progress` deltas, never mirrored in either grain — flood) the parsed `report` items rendered as `[fact]`/`[ask:*]`/`[resolved]` lines (`_render_report_items`) — clipped at `_MIRROR_TEXT_CAP` (4000 chars) so a runaway pass can't bloat the journal. The background wake also opens its burst with a "Background reflection: draining N unreflected chat(s)" journal line (leaf `activity_log` import in `background_reflection.py`), between the scheduler's chip and its `describe`d terminal line. No client change: the Activity tab renders multi-line messages as-is. GPU-free verified: an ad-hoc store→journal integration check (background run mirrors chat name + `[fact]`/`[ask:user]`/`[resolved]` items + indented VERDICT body + branch line, token deltas never leak; foreground run keeps phase grain with no text body, branch events not mirrored) plus the `activity_log`/`background_reflection`/`idle_scheduler` self-tests. Files: `server/inference/core/reflection_config.py`, `server/inference/core/background_reflection.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`.

## 2026-07-22
- **Unified activity log — one always-on, box-wide journal of what Ava is doing on the GPU.** Reporting was per-run and per-subsystem: a reflection run streamed detailed events keyed to a `run_id` the client had to already hold, while the five autonomous idle jobs (wander/outreach/synthesis/checkin/background_reflection) reported only via `print()` to `server.log`, which no client polls. So the moment a reflection run ended and the box moved on to autonomous work — or an idle job fired on its own with no run in progress — the UI went silent even though the GPU was plainly busy (the reported symptom: "the model is clearly running something on the box but the Sleep tab shows nothing"). New leaf sink `core/activity_log.py` (imports nothing from the project, imported directly like `reachout_gate`, no injection/cycle) is the ONE place every GPU subsystem writes to: a single global monotonic **restart-durable `seq`** (recovered from the file tail), append-only to `data/hot/activity/activity.jsonl` (in-memory ring + size rotation), plus `current()` for a live "running now" chip. **Universal idle-job coverage without touching job bodies:** `idle_scheduler._dispatch` opens the chip via `set_current` (deliberately emits NO `started` journal line — a job that fires hourly and finds nothing to do would otherwise flood the log with started→skip pairs), runs the job, then emits ONE terminal line phrased by an optional per-job `IdleJob.describe` (registered next to the job, so the generic scheduler stays semantics-free), with consecutive identical "nothing to do" skips deduped (`_last_skip`). **Rich progress:** each subsystem's existing `on_stage`/`on_question` phase hooks are routed to the journal on the autonomous path too (never the raw token `on_chunk` stream — that would flood a persistent log), so outreach/checkin/synthesis narrate their reasoning, not just start/finish. **Reflection mirror at the true sink:** `ReflectionRunStore.append_event` (every reflection event flows through it, including the downstream merge-rag/commit-training/archive markers that never route through `send_event_fn`) coarse-mirrors a whitelist of phase/session events (`_ACTIVITY_MIRROR_EVENTS`) keyed by `run_id`, so a run shows at phase grain in the unified log while the Sleep tab keeps its detailed per-branch console (no duplication). Client: `activity_events {after_seq}` → `activity_events_batch {events, latest_seq, activity}` (mirrors the `reflection_run_events` shape); a new **Activity tab** (`client/ui/activity_widget.py`) polls it every ~2 s independent of any run and survives disconnect/reconnect by resuming its one cursor; the live chip also shows in the main-window status bar. **The one honest seam — training:** the Sleep `train` stage hands LoRA production to the watchdog, which stops inference, so the activity socket is legitimately dead for that window; the Activity tab detects the socket down and merges the watchdog's `/job/progress` HTTP stream into the same scrolling log (the same dual-source logic the Sleep tab's `TrainPollWorker` already uses) — unification at the store for everything on-box, at the view for the training blackout. GPU-free verified: `python -m core.activity_log` (seq monotonicity, current-pointer defaulting, cursor reads, restart seq recovery), `python -m core.idle_scheduler` (unchanged independence property), and an ad-hoc scheduler→journal integration check (progress-from-hook + describe outcome + skip-dedup + chip clear). **Not yet exercised on a live GPU** (the server needs CUDA/unsloth). Files: `server/inference/core/activity_log.py` (new), `idle_scheduler.py`, `reflection_config.py`, `reflection_service.py`, `server.py`, `client/ui/activity_widget.py` (new), `client/ui/main_window.py`, `client/core/backend_client.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`, `CLAUDE.md`. Commit: pending.

- **Folded cap-age contamination — one per-token-weighted row instead of the two-row split (`contamination.fold`, default OFF).** Profiling a live cycle found the cap-age user-contamination pair (`build_dataset._contamination_rows`, §5e) is ~40% of the corpus: a cap-age exchange emits a masked row (LR-mult 3.0, response only) + a byte-identical `unmask_user` row (1.0, response+user), so the **same sequence trains twice** — 582/1436 rows and 2.38M/5.89M tokens/epoch on the measured build, ≈2.8h of a ~7h cycle spent re-processing duplicates. New `WallClockConfig.contamination_fold` collapses the pair into ONE row: LR-mult = the response total (`cap` split / `cap+dose` additive) carrying a per-token `user_loss_weight = dose/response_total` (0.25 split, 0.2 additive) on the final user turn. A weighted `compute_loss` gives the response weight 1.0 and the user turn that fraction, reproducing the pair's per-token exposure (response ×4, user ×1) in a single forward+backward. **Memory-safe by construction:** loss is over the unmasked tokens only (final answer + user span, a few hundred), so `_OrderedSFTTrainer.compute_loss` runs the backbone → hidden states, then the `lm_head` on ONLY those positions (chunked, `_FOLD_LOSS_CHUNK=512`) — it never materializes the full seq×262k-vocab fp32 logit buffer that the fused no-logits CE exists to avoid (memory `training-oom-gemma31b`); `_resolve_backbone_and_head` is validated at setup so a bad model layout fails loudly, not 6h in. **Not bit-identical to the split** (one Adam update + one weighted-mean normalizer vs two) — both are heuristics for "response at cap, voice at a dose"; the fold is the cleaner single-normalizer form. A/B the two off the same frozen corpus (fold changes the corpus fingerprint, so builds are distinguishable). **Every non-folded row (all normal + wander, and every row when `fold` is off) takes the untouched fused-CE path**, so the standard corpus is byte-for-byte unchanged: the `loss_weight`/`user_loss_weight` columns exist only on a folding cycle, and `compute_loss` branches on their presence. Diagnostics: a one-time "column reached the collator" log + an end-of-cycle "weighted loss applied on N folded rows" (0 while active ⇒ the column was stripped, fold degraded — surfaced). GPU-free verified: `test_contamination` (fold → 1 weighted row at 4.0/0.25, additive 5.0/0.2, gated short-user unchanged, non-folded rows leave `user_loss_weight` None), `label_policy.row_loss_weights` selftest, and a CPU weight-tensor-builder check (1.0 on final response, dose on user span, 0 on markers/non-final turns). **The GPU weighted `compute_loss` is not yet run end-to-end** (like `train_cycle` generally) — the backbone-gather path needs a real cycle to confirm on Gemma4-31B. Files: `server/training/decay.py`, `server/training/build_dataset.py`, `server/training/label_policy.py`, `server/training/train_cycle.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Background per-chat reflection — idle-triggered, user-preemptible; two-stage freeze.** The expensive part of a Sleep run is the *per-chat* work (consolidation + revision + branch generation, one generation per exchange on a large model), and it parallelizes across chats. A new autonomous idle job (`core/background_reflection.py`, the fifth alongside outreach/synthesis/check-in/wander) does that work incrementally after a long **user-idle** stretch (its own `idle_seconds=1800`, longer than the reach-out jobs' shared 3600 window; `interval_s=120` re-arms while a backlog remains), so the nightly cycle has far less to do. It drains unreflected chats **one at a time** in a fresh throwaway staging dir, delegating the single-chat reflection to the new `reflection_service.run_chat_only_reflection` (so the module carries no generation plumbing). It does **not** reach out or train — purely the per-chat passes.
  - **Two-stage freeze.** The single doc-level `reflected_at` reflect-once stamp is now the *second* stage. The background pass runs `chat_only` mode (new `ReflectionRunConfig.chat_only`; `ReflectionRunner.execute_run` skips the run-level persona digest and passes `clean_base_ctx=None` so the clean-base judge + fact placement do NOT run) and freezes each completed chat `chat_reflected` (stage one; new `ChatSidecar.mark_chat_reflected`/`is_chat_reflected`). The chat's collected clean-base **job payloads** (`judge_jobs`/`fact_candidates` — plain JSON dicts) are persisted to a sibling dir (`data/hot/reflection_pending_clean_base/<stem>.json`). The next NORMAL reflection run (UI or `reflection_run.py`) folds the background checkpoint into live at start, then — via a new per-session branch in `execute_run` — loads those payloads (`consume_pending_clean_base_fn`, consume-once) into its end-of-run clean-base phase for the `chat_reflected` chats *without re-generating* the per-chat passes, and stamps `reflected_at`. Persona digest needs no special handling: it folds from the whole-corpus live ledger, so background-produced `[persona]` evidence is already visible. The CLI's `--all-pending` selector now also treats a `chat_reflected`-not-`reflected_at` chat as pending (it has all verdicts, so the verdict-presence rule alone would exclude it).
  - **Per-chat atomicity via the existing checkpoint.** Each completed chat's one-chat staging deltas + frozen sidecar are APPENDED into the durable `data/hot/reflection_checkpoint/` (the same store crash-recovery uses, accumulating across wakes; `reflection_staging.commit_background_chat`), and the pending-jobs dir is a **sibling** of the checkpoint so `discard_checkpoint`'s rmtree during the fold never deletes jobs the main pass hasn't consumed. A preempted (partial) chat leaves only the throwaway staging, which is discarded — nothing reaches the checkpoint, so the cache is never corrupted; the chat is re-reflected from the top later.
  - **Preemption — chat wins over background only.** This is the one place a user chat *preempts* reflection instead of being refused. `generation._run_generation` (and the ephemeral / regenerate guards, and an operator Sleep start in `reflection_service`) call `background_reflection.request_preempt()` when a background pass is active: it sets the run's stop flag and fires the shared `_cancel_event`, so the in-flight reflection generation aborts within roughly one step (the reflect-generate path already clears+checks `_cancel_event`), then the caller waits briefly for the executor to free and proceeds. An operator-launched Sleep run / encounter is still **refused** (deliberate work must not die to a stray chat turn). `background_reflection._background_reflection_active` is added to the scheduler's `external_busy`. GPU-free self-test: `python -m core.background_reflection`. **Not yet exercised on a live GPU.**

## 2026-07-21

- **Reversed-session role confusion — Ava no longer reads her own opener as "the user."** In an `initiated_by:"ava"` session (outreach/synthesis/check-in) exchange 0 is her opener under the synthetic `(initiative)` stage-direction speaker, with a second-person impulse (`"You decided to raise this with … : <ask>"`) as its `user_prompt`. Every reflection renderer treated that synthetic turn as a genuine interlocutor utterance labelled `"(initiative): …"` — so both the judged-subject and the lead-up context of exchanges 1+ presented Ava's own words in the user slot. Observed in a revisit run's revision CoT (`20260714_123433_rv`, over `20260712_121814.json`): *"User (AI) initiates a deep, existential dive … The user is having a meta-cognitive crisis"* — the model resolving the contradiction by conflating itself with "the user," poisoning produced `[fact]`s and re-derived IDEAL targets. Two-part fix: (1) `reflection_source.build_revision_jobs` no longer emits a revision **subject** for an Ava-initiated exchange 0 (mirrors the training mask in `dialogue_source.build_dialogue_anchor`; her opener still flows into later exchanges' context as an assistant turn); (2) a shared narrator-turn rendering (`_NARRATOR_SPEAKERS = {"(initiative)","(setting)"}` → a `(stage direction — you, not another person)` tag, no speaker label) applied identically in `reflection_source` (consolidation subject, revision context, IDEAL replay), `reflection_chunking.format_exchange_block` (consolidation), **and** the parity-coupled pair `generation._build_inference_conversation` (live inference / IDEAL generation) + `training.render` (`_user_content` / `_reference_inference_conversation`). Keeping all three renderers in lock-step preserves prefix parity by construction — IDEAL-generation, training-render, and live-inference now render the synthetic turn byte-identically (also fixes the same confusion at live chat time when a user answers an Ava-initiated opener). Ordinary chats are byte-for-byte unchanged; real speaker names never collide with the two narrator tokens. Encounters' `(setting)` opener is folded into the same treatment. Existing sidecars are unaffected (context stores the raw speaker, re-rendered at build/inference time), so a from-scratch rebuild picks the fix up with no migration.

- **Reflection run provenance — the "Runner started" line no longer misreports the base model.** `reflection_runner` renders its startup event from `run_state["model_id"]`/`["adapter_id"]` (`reflection_runner.py:434-436`), but nothing ever populated those keys — so every run (adapter loaded or not) logged `model=? adapter=none (base model)`, a pure mislabel: reflection/revision generation runs through `_make_sync_reflect_generate`, which reads the live `_runtime.model` (adapter attached) at call time and never reloads a base (the only real swap is the phase-two clean-base judge, which a revisit suppresses). Fix seeds the two scalars in `ReflectionRunStore.create_run` (`reflection_config.py`) from `core.runtime_state.runtime` via a lazy import (config layer stays import-time-decoupled; an unloaded/headless runtime yields `None` → the runner's existing `?`/`none (base model)` fallback). Covers every path through the one choke point (main run, revisit head sub-run, dry, headless CLI). New effect on artifacts: `<run_id>.meta.json` now carries `model_id`/`adapter_id` provenance fields.

- **Counter-evidence PRODUCER — the persuasion channel is now live end-to-end.** Wires the previously-dormant counter-evidence ledger (below) to an actual source, so sustained user pushback can now move a strong persona. Four parts: **(1) next-turn reaction feed** — `reflection_source.build_revision_jobs` captures each exchange's *immediately-following* user turn (`next_user`/`next_speaker`), and `build_revision_content` injects it as a JSON-encoded `WHAT THEY SAID NEXT` block appended to the (never-truncated) subject; absent for a session's last exchange. It is framed as a *reaction, not a directive* — mild context the judgement may weigh under the same "you may reject it" discretion as the Meta-note feedback. **(2) COUNTER classification** — `revision_prompt.txt` + `revisit_prompt.txt` gain a `COUNTER: yes|no` field, explicitly SEPARATE from `VERDICT` (a reply can be kept and still have met resistance), that fires only when the next turn pushes back against a *belief/preference/boundary/way-of-being the reply expressed* — not a fact, not a follow-up/topic-change. Parsed by `reflection_writer.revision_counter` (label + JSON fallback; absent/garbled ⇒ False, never fabricates). **(3) reaction→key bridge** — `rag_engine.persona_keys(text, before_session, top_k=2)` returns the live `[persona]` keys most relevant to the reply (persona-only, temporally cut, decay-scaled, clearly-relevant floor), resolving *which stance(s)* the push lands on. **(4) emission** — `reflection_runner._write_counter_evidence` (called in the revision persist block on `COUNTER: yes` when a next turn exists) appends one `counter` ledger op per resolved key under THIS chat as the pushing session, emitting `counter_written`/`counter_skipped` progress events. Guards: no bridge, or no clearly-relevant live stance ⇒ nothing written (a reaction lands only where it maps to a real self-statement); suppressed on dry runs (persist block); best-effort (a failure never derails the pass). Threaded via a new `persona_keys_fn` injected into `execute_run`/`_run_revision_for_session`, wired from `reflection_service._make_persona_keys_fn` at all three live call sites **and** the headless `reflection_run.py` CLI (so both real reflection paths produce counters; a caller that omits it leaves the channel dormant, like `clean_base_ctx`). **Runs in both first-reflection and revisit** — revisit is the intended low-gain integrator (re-reading aged chats accumulates pushback across separate conversations; distinct-session currency means re-revisiting one chat can't inflate the count). Verified GPU-free: parser, next-turn capture/injection, and a full runner→ledger→`gather_persona_raw` round-trip (a mature stance at weighted 1.96 → 0.0 under 4 pushes across 4 chats; no-stance and dormant-bridge guards write nothing). This closes the producer half the counter-evidence foundation flagged as "not built." Motivating context: the user's original "Revisit old chat" / persona-drift line — the correction the user gives after a reply (the "you missed A/B/C, simpler exists" turn) now steers the persona instead of being discarded.

- **Counter-evidence ledger — the persuasion channel (FOUNDATION built, dormant).** The symmetric opponent to persona affirmation: a mechanism by which sustained user pushback can move a strong persona *without a hard turn*, addressing "the persona is too intense / won't drift." Persona formation counts affirmations across distinct sessions (`register` ops → `weighted_recurrence`); this adds a **counter** op — `session S pushed AGAINST stance K` — that is **netted out** of the theme's weighted recurrence. **A — the op** (`ConsolidationLedger.counter`, `training/ledger.py`): append-only, graded, and additive — unlike `supersede` (one hard soften that drops the anchor), a counter never removes the anchor, it adds one dated distinct-session unit of negative pressure; reversible and re-foldable, decays by recency like an affirmation, and is consumed by the raw op scan (not `fold()`), so a countered-but-not-yet-faded trait stays live and recallable. **B — the fold** (`reflection_digest.gather_persona_raw` + `_evidence_entry`): `gather_persona_raw` scans `counter` ops into per-key `counter_sessions`/`counter_weights` (recency-dated, same shape as affirmations); `_evidence_entry` subtracts `_PERSUASION_GAIN · Σ(counter recency weights)` from the affirmation sum, floored at 0. **Asymmetry, deliberate:** counters are recency-weighted but **NOT** tenure-discounted — external pushback is not the circular self-vote, so it must be free to *accumulate* (the integrator). `_PERSUASION_GAIN` (default `0.5`, the persuadability knob) is kept below 1 so **one push cannot cancel one affirmation** — a mature trait at `weighted ≈ 1.96` drops to `1.46` on a single push (barely dented, still well above the `_PROMPT_WEIGHT_FLOOR` = 0.5) but fades to 0 under ~4 sustained pushes, at which point it evaporates from the portrait on its own (no explicit supersede). **E — wiring:** the counter-session count is folded into `raw_fingerprint`, so accumulating pushback triggers one digest regen per new push (deterministic dated ops, no thrash) instead of sitting unread. **Scope:** only `weighted_recurrence` is affected — raw `recurrences` is untouched, so the digest maturity gate / branch-judge flip is unchanged (same discipline as the tenure discount). **Dormant — nothing emits counter ops yet.** The two consumers that would *produce* them are the next layer and are NOT built: (1) the reaction→key bridge (a key-returning persona-relevance lookup that resolves a reaction to the live persona keys it pushes against), and (2) the pushback-detection policy in the revision/revisit pass (deciding a post-reply reaction / next-turn feed is genuine pushback vs. mere disagreement). Revisit is the intended low-gain integrator for accumulating these. Verified GPU-free: `_evidence_entry` netting + a full ledger→`gather_persona_raw` round-trip (3 affirmations → 1.96; +4 counters → 0.0). Motivating design context: the persona-drift / "Revisit old chat" line — persuadability as a low-gain integrator over sustained counter-evidence.

- **Persona tenure discount — the anti-ratchet on persona `weighted_recurrence`.** Problem: persona formation is a positive-feedback loop that ossifies the self-portrait and starves genuine drift. The current persona is recalled at chat time (`from_weights` → `persona_context`), so a reply is *conditioned on* the persona; revision then re-derives the same `[persona]` from that reply and registers it under a new `source_session`, incrementing the theme's distinct-session recurrence. This "circular vote" (persona conditions the reply → revision re-derives it → +1 session) let `weighted_recurrence` climb linearly with no opposing force but the fixed-rate recency decay, which re-affirmation-every-session outran — the persona locked and the branch-judge maturity gate armed harder over time. Fix (in `reflection_digest._evidence_entry`): rank a theme's distinct sessions oldest→newest (persona `source_session`s are timestamp stems, so a plain `sorted()` is chronological *and* revisit-stable) and scale each successive affirmation by `_TENURE_DECAY**rank` (default `0.6`) **before** the recency sum. The first session (emergence) keeps full weight; later echoes decay geometrically, so an echo-sustained theme's `weighted_recurrence` **converges** to `1/(1-_TENURE_DECAY)` (≈2.5) instead of growing without bound. The tenure discount composes with the existing recency decay and the two are anti-correlated by construction (emergence = oldest = most recency-decayed; recent echoes = highest rank = most tenure-discounted), so **only a young-and-active theme escapes both** — exactly the profile that should be allowed to form (drift *in*), while old echo-locked traits deflate (drift *out* becomes possible), and neither in a single hard turn. **Scope, deliberately narrow:** only `weighted_recurrence` changes — the raw integer `recurrences` count is untouched, so the digest **maturity gate / branch-judge criterion flip** (`reflection_runner._digest_maturity_gate`, which reads numeric `recurrences ≥ 3`) and the regen fingerprints (`raw_fingerprint`, `_evidence_fingerprint`) are unchanged. This is the low-risk lever that softens how intensely the *self-portrait asserts* a trait (the `[N.N]` prompt hints + `_PROMPT_WEIGHT_FLOOR` evaporation), not the judge's authority. **Interpretation of existing artifacts:** a persisted digest's per-theme `weighted_recurrence` is now tenure-discounted, so historical `[N.N]` hints are not comparable across this change (a pre-change theme at `[3.0]` reads `[2.0]` after); `recurrence` (raw) is stable. Contained entirely in `reflection_digest.py` (no external `weighted_recurrence` consumers); GPU-free self-test updated with a convergence-ceiling guard. `_TENURE_DECAY` is the single knob (lower = more persuadable). Motivating design context: the "Revisit old chat" / persona-drift line of work — persuadability as a low-gain integrator over sustained counter-evidence rather than a hard turn.

## 2026-07-20

- **Consolidation-gist RAG channel: a distilled per-conversation summary that crossfades with the verbatim transcript.** Observation: consolidation reasoning already reaches a summary of each chat ("we talked about X, using a/b/c, concluded Z") — but it lives in the discarded `<think>` scratch and only extracted `[fact]`/`[ask]`/`[resolved]` items survive; the episodic gist is thrown away. This adds it as a first-class recall layer, modeled on episodic→semantic memory consolidation (fast verbatim trace → durable gist). **Production is a dedicated generation, not scraped CoT** (the consolidation reasoning register is unusable as injected prose): a new SUMMARY pass (`summary_prompt.txt` — a detail-preserving note-to-future-self recap, ~2-3k tokens for a substantial chat) runs as a *second* generation over the SAME packed consolidation context inside `_run_consolidation_session`'s chunk loop, so context packing is paid once; per-chunk recaps concatenate for a chunked session. It writes once per session to the chat sidecar (`ChatSidecar.write_summary` → session-level `consolidation_summary`), best-effort and skipped on dry runs. **Storage is the chat index, not reflection memory** — so the crossfade is free: `rag_engine._collect_chat_entries` reads the summary off the already-loaded sidecar and chunks it into the **chat** FAISS index as `kind="gist"` passages (same passage unit as verbatim turns, so ranking is apples-to-apples), each with a distinct negative-sentinel `exchange_index` (collapse-to-passage: a query retrieves the relevant slice, not the whole blob). Emitted **before** the verbatim `session_modifier<=0` short-circuit so the gist survives exactly when the verbatim drops out. **Decay is a tent that mirrors the verbatim fade** (`training/decay.gist_rag_weight_hours` + `rag_engine._gist_modifier`, new `WallClockConfig.gist_cap_age_h` default 192h): weight ramps 0→1 over `[0, rag_cap_age_h]` (the exact inverse of `rag_weight_hours`, crossing at 0.5/0.5 on day 2) then decays 1→0 over `[rag_cap_age_h, gist_cap_age_h]` (~day 4→8) — so a chat is recalled verbatim while fresh, through its summary once aged, and the summary itself fades ~4 days later. Rendered as a remembered conclusion ("From an earlier conversation, you recall: …"), never a quoted turn. **RAG-only by construction**: it lives on the sidecar + chat index only, never `weights_persona.jsonl`/ledger/training — a lossy model summary must not fossilize into weights. A revisit regenerates it under the evolved persona. Only surfaces on live-chat/encounter paths (reflection passes set `rag_include_chat=False`); the existing `before_session` temporal fence covers it unchanged. GPU-free verified: tent curve (exact inverse crossfade), sidecar write/read roundtrip + empty-refusal, prompt load. **Not yet run through a real reflection cycle** — retrieval value and the drift tradeoff (past `gist_cap_age_h` the chat is recalled only through Ava's own summary; the raw transcript stays on disk in `archive/chats` for audit) need a live run to confirm. Files: `server/training/decay.py`, `server/inference/core/chat_sidecar.py`, `server/inference/core/reflection_prompts.py`, `server/inference/core/reflection_runner.py`, `server/inference/core/rag_engine.py`, `server/inference/prompts/summary_prompt.txt`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Targeted per-thread [ask]-resolution — the tight [ask]-loop close.** Ava kept re-raising the same open question because an ask only leaves the live set via a `[resolved]` eviction, and the *only* answer-driven eviction path was the big consolidation pass spontaneously emitting a `## RESOLVED` item whose `content_key` hash-matched the open ask (fragile: the model had to notice the buried answer *and* reproduce the question text). Meta asks are surface-ceiling-exempt, so an unresolved one re-surfaced every idle window. New mechanism: (1) when Ava initiates a conversation from one of her own asks (outreach / synthesis opener), the triggering ask is now **stamped on the session** (`initiated_ask: {key, content, ask_kind}`, `chat_logger.start_session`) — persists across the user's in-place reply. (2) A new per-session reflection step `reflection_runner._run_ask_resolution_for_session` (runs after consolidation persisted its own evictions, before revision; skipped on dry-run / revisit) computes the session-scoped open asks — the stamped `initiated_ask` **unioned with** the previously-unused `ReflectionMemory.asks_surfaced_in(session)` join (also covers a *passively*-surfaced ask in a user-opened chat) — and runs a small DECISION/ANSWER decision pass over the actual replies for each. An answered ask is evicted **and distilled** via the new `ReflectionWriter.write_answered_resolution` (`resolved_by:"user"`), which shares the factored `_distill_resolved` helper with `write_consolidation`'s resolved branch, so the weights/RAG store stays identical. Because the caller already holds the exact `key` from the surfaced-ask join, resolution is **deterministic** — no reliance on content_key reproduction. Emits `phase="ask_resolution"` events (`ask_resolved` / `ask_unresolved`). No double-resolve: an ask the consolidation pass already closed is no longer open in the fresh fold. Files: `server/inference/core/chat_logger.py`, `server/inference/core/outreach.py`, `server/inference/core/synthesis.py`, `server/inference/core/reflection_writer.py`, `server/inference/core/reflection_runner.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

## 2026-07-19

- **Cap-age user contamination gated on final-user-turn length (`contamination.min_user_chars`, default 100).** The cap-age user-contamination copy (`build_dataset._contamination_rows`, REBUILD §5e) unmasks the exchange's **final user turn** into the loss so Ava's voice entrains on the user's at `contamination_dose`. But at cap-age a short user turn ("ok" / "да" / "why?") carries no substantive voice to entrain — unmasking it just teaches Ava to emit terse user-style filler as her own output (mild mis-entrainment, not neutral noise). New `WallClockConfig.contamination_min_user_chars` (parsed from `consolidation.wall_clock.contamination.min_user_chars`, default `100`): when the stripped final user turn is shorter, `_contamination_rows` skips the unmask copy and returns a **single masked row at the full cap (4.0)** — split-mode LR-neutral (4.0 either way), additive-mode drops the extra dose (5.0→4.0). `0` disables the gate (every cap-age exchange contaminates, prior behavior). Char-based (not tokens) so `build_dataset` stays GPU/tokenizer-free; the user-turn length is taken from `anchor["prompt"]` at the call site. The Training review tab already collapses the masked+unmask duplicate into one entry, so a short-user exchange simply shows once. GPU-free verified (`test_contamination` extended: short user turn → single 4.0 in split *and* additive mode, gate-off restores the 3+1 split; `test_config_from_dict` covers the new knob). Files: `server/training/build_dataset.py`, `server/training/decay.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Persona CoT injection retired in favor of implicit persona conditioning (fixes the CoT-flood → answer-repeat).** Symptom: with History+Persona RAG on, subsequent chat turns produced a fresh, valid CoT but then repeated a prior answer verbatim; it correlated with the CoT filling with *dozens* of persona self-statements. Root cause is training-side, not RAG: `build_dataset._inject`/`persona_render` prepended each live `[persona]` statement to the **front** of its host exchange's `<think>` block (cap 2/exchange, no global density cap), so across a growing persona corpus the adapter learned an "open reasoning by reciting persona" prior. At inference the 3 injected persona RAG lines act as exemplars that amplify that prior into a full enumeration — a lexically-diverse list that evades both loop guards (`detect_ngram_loop`) and the drift guard (`detect_degeneration`, which by design treats varied prose as expression) — and after the persona wall the answer collapses onto the strongest concrete completion in context: the previous turn's answer. **Fix, two parts.** *(1)* `build_dataset._inject` no longer injects persona (facts keep the host-exchange path; `PERSONA_INJECT_CAP` removed) — a `keep`/branch target already carries the original chat-time CoT (generated with persona RAG, so it gestures the trait it was distilled from). *(2)* the `revise`/ideal-win target is generated **persona-conditioned**: a new `persona_context_fn` (persona-only RAG, `include_facts=False`/`include_chat=False`, temporally cut to `before_session` so it's only self-knowledge Ava already had — preserving `build_ideal_messages`' provenance boundary) is appended to the clean IDEAL's generation system prompt, and the **same block is persisted verbatim** onto the sidecar (`persona_context`, ideal-win only). `dialogue_source.build_dialogue_anchor` concatenates it into the anchor `system_prompt`, so `render.build_messages`/`assert_parity` reconstruct an identical system message and **train/inference parity holds by construction** (the parity core is untouched). Persistence is gated to a genuine ideal-win (not keep/branch/cot_regen), and `write_verdict` writes `persona_context` fresh each verdict so a judge criterion-flip to a branch clears a stale block; the criterion-flip override (`_maybe_override_target`) re-persists it only on a flip back to the IDEAL. Threaded through `execute_run` → `_run_revision_for_session` → `_generate_ideal_reply` (both attempts) and wired at all three `reflection_service` execute_run sites + the headless `reflection_run.py` CLI. **Net:** explicit persona prepend gone everywhere; persona positions naturally, only where relevant. **Known tradeoffs (accepted):** persona now trains only where topically relevant (a rarely-relevant trait can starve — coverage instrumentation is a follow-on), and keep-target persona coverage depends on persona RAG being on during live chat. GPU-free verified: pipeline self-test updated (persona no longer prepended, fact still is) + a dedicated parity check (sidecar round-trip, anchor concatenation, `assert_parity`, ideal-win-only persistence, keep clears stale block). **Not yet run through a real LoRA cycle** — the flood reduction and persona retention need a retrain to confirm. Files: `server/training/build_dataset.py`, `server/training/dialogue_source.py`, `server/inference/core/reflection_source.py`, `server/inference/core/reflection_runner.py`, `server/inference/core/reflection_writer.py`, `server/inference/core/chat_sidecar.py`, `server/inference/core/reflection_service.py`, `server/reflection_run.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **`[fact]`/`[persona]` entries stamped with a source-chat origin timestamp (`origin_ts`).** Groundwork ahead of the imminent full wipe + re-reflect-everything, whose motivating problem is a **stale-recall echo chamber**: stale facts/persona keep getting injected into chat context and reinforce themselves. Every reflection record already carried a `ts` (`datetime.now()`), but that is **reflection-run time** — a mass re-reflect collapses it to "now" for every entry, making it useless as a recency signal exactly when it's needed. New `source_origin_ts()` in `reflection_writer` derives the **source chat's own date** from `source_session` (the session-file stem, e.g. `20260705_014505` — mirroring `training.decay.parse_ts`, kept local to avoid an inference→training import, same rationale as `ideal_has_usable_answer`; tolerates the `.json`/`.state.json` suffix, the `-` separator, and the same-second `_<n>` collision suffix). It is stamped as `origin_ts` on every fact/persona write and its recall mirror: consolidation WEIGHTS (`write_consolidation`), the `from_weights` RAG mirror (`_emit_weight_recall`), consolidation RAG inserts (facts + asks), resolve-and-distill WEIGHTS, revision-pass persona (`write_revision`), and preserved across a `write_dedup` survivor re-insert. A self-directed / non-chat source (`wiki:`/`til:`/`web:`/`lookup`) has no conversation date → field omitted. `origin_ts` survives untouched through `reflection_memory._fold()` into `live_items()`, so a future **recall-time** staleness/decay pass can fade a self-reinforcing stale fact/persona out of context — the recall half that train-time wall-clock decay (`training/decay.py`, which already clocks from the same chat instant) never covered. Purely additive: no reader consumes `origin_ts` yet; the wipe→re-reflect corpus is simply **born recency-tagged**. GPU-free self-test extended (`python -m core.reflection_writer`) + end-to-end write verified (chat source stamps the chat date distinct from run `ts`; `wiki:` source omits it). Files: `server/inference/core/reflection_writer.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: `d9f78d2`.

- **Training review "Rewrite history": bake human-reviewed targets into the transcripts (schema v5).** The Regenerate/Apply flow leaves an operator-reviewed target as a **sidecar override** (`locked`) layered over the still-corrupt transcript: training reads the good target, but everything that reads the transcript *directly* — RAG recall at chat time, runnable snapshots, and every LATER exchange's context (`dialogue_source` builds context from prior `assistant_response`) — still sees the corrupt reply. New header button **"Rewrite history…"** (`rewrite_history` → `session_ops.handle_rewrite_history`, confirm-first) is the finalize counterpart: for **every** locked exchange in hot/chats it (1) rewrites the transcript's `assistant_cot`/`assistant_response` from the reviewed sidecar target (`chat_logger.rewrite_exchange_history`, bumps `CHAT_SCHEMA_VERSION` 4→5), preserving the prior CoT/reply under an append-only per-exchange `rewrite_history` record and **deleting** the now-stale `tension` block (its per-token entropy/margin/contested-ids + raw `token_ids` were measured on the *original* generation, so branch replay would replay stale ids — the user explicitly OK'd dropping it), and clears corrupt flags; then (2) **unfreezes** the exchange via the new `ChatSidecar.set_exchange_locked(..., False)` — the one intentional lock-clearing path (`write_verdict` OR-s the lock sticky), safe now that the transcript itself is corrected so the lock's job (protect the target from re-derivation over corrupt content) is done. Training output is unchanged: `dialogue_source` reads the target off the untouched sidecar record; what's fixed is the transcript-derived data, and chat-RAG is re-embedded (`refresh_chat_index`) after. The session-level reflect-once freeze (`reflected_at`) is deliberately left as-is (the trained target is already the reviewed one; a later Revisit re-derives cleanly over the corrected transcript). Refuses on a live reflection run; skips the active conversation file. Interacts cleanly with the keep-chats wipe fix below — after a rewrite the exchange is unlocked, but the transcript is clean so wipe→re-reflect reproduces good targets, no preservation needed. GPU-free end-to-end verified (rewrite + tension drop + corrupt-clear + forensic record + lock cleared + record/target retained for training + RAG refresh). Files: `server/inference/core/chat_logger.py`, `server/inference/core/chat_sidecar.py`, `server/inference/core/session_ops.py`, `server/inference/server.py`, `client/core/backend_client.py`, `client/ui/training_review_widget.py`, `CLAUDE.md`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Keep-chats wipe now preserves human-validated (locked) exchanges.** The disaster-recovery wipe, when *not* wiping chats, deleted **every** `.state.json` chat sidecar as regenerable reflection state (`state_wipe.wipe_reflection_state` globbed `*.state.json`). But a `locked` exchange record is an operator-reviewed, hand-authored target from the Training review tab's Regenerate flow — **un-regenerable**, and the entire purpose of the lock is that re-reflection (and revisit) must never re-derive it from the original, possibly corrupt, transcript. Wiping it meant that after the keep-chats reset (adapters gone → chats restored to hot → re-reflect on next Sleep), the re-reflection would overwrite the operator's reviewed target with a fresh pass over the poison — silently undoing the human repair the wipe was told to keep. Now `wipe_reflection_state` routes each sidecar through `_prune_or_remove_sidecar`: a sidecar with no locked exchanges is deleted as before; one with locked exchanges is **rewritten down to just those records**, dropping `reflected_at` so the chat's *other* exchanges re-reflect clean while the locked ones stay skipped-and-preserved. `.shareml.json` artifacts (nothing un-regenerable) are still removed outright. `restore_archived_transcripts` now moves a preserved locked sidecar alongside its transcript when restoring an archived chat to hot (otherwise the lock would be orphaned in `archive/chats` and lost to the hot-dir re-reflection). Full-wipe (`--wipe-chats`) is unchanged — a total reset still deletes everything. The wipe summary reports `preserved_locked_sidecars`/`preserved_locked_exchanges`. GPU-free verified end-to-end (hot + archived locked chat survives wipe+restore; `ChatSidecar.locked_exchange_indices` sees the lock in hot afterward). Files: `server/inference/core/state_wipe.py`, `server/wipe_state.py`, `CLAUDE.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Chat tab: per-channel RAG toggles (History / Facts / Persona) to isolate the adapter.** The Chat tab gained three checkboxes (all on by default) that gate the chat-time RAG channels so an operator can A/B a channel's effect on a reply — **all off = chatting with the adapter only** (no injected retrieval). Mapping: **History** = past-chat exchange recall *and* the wander channel (both experiential recall Ava draws from past activity/reading); **Facts** = distilled `[fact]` reflection memory *and* the `[ask]` "still wondering" recalls (the same informational reflection stream); **Persona** = distilled `[persona]` self-statement memory. The `generate` WebSocket message carries three new optional keys (`rag_history` / `rag_facts` / `rag_persona`, all defaulting **True** server-side, so every other caller — encounter, ephemeral, reflection/revision — is unaffected). `RagEngine.query` gained `include_facts` / `include_persona` params (both default True); `_query_reflection` filters the reflection block by kind (persona→`include_persona`, fact+ask→`include_facts`). `handle_generate` reads the flags off the message and passes `include_chat`/`include_wander` from History. Out of scope (not RAG): the turn-1 proactive open-question surfacing is a separate subsystem and still fires. Files: `client/ui/chat_widget.py`, `client/core/backend_client.py`, `server/inference/core/generation.py`, `server/inference/core/rag_engine.py`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

## 2026-07-18

- **Persona recency decay: the self-portrait now fades stale evidence, while reinforced traits survive.** The persona digest folded *every* committed `[persona]` anchor with no recency weighting — a stance from a year ago counted like one from yesterday, so drift never faded and near-duplicate re-statements only accumulated. Now `gather_persona_raw` dates each anchor's contributing sessions (from the ledger register `ts`) and assigns a wall-clock recency weight (`_persona_recency_weight`: full ≤30d, linear to 0 at 180d, matching `training/decay.py`'s fact/dialogue decay). The decay is **summed per theme** (`_evidence_entry.weighted_recurrence`), not a hard per-entry cutoff: a theme reinforced across recent sessions keeps a high weighted recurrence and survives, while a stale one-off decays toward 0 and — below `_PROMPT_WEIGHT_FLOOR` (0.5) — drops out of the evidence Ava reflects on, so the regenerated portrait forgets it. This deliberately protects a stable-and-reinforced trait from being erased just because it wasn't restated lately (the flaw in a raw "drop entries past N" rule). Themes rank by weighted recurrence; the prompt shows the weighted number. To stop the regenerate gate thrashing under continuous decay, a coarse **recency band** (`_recency_band`, 4 buckets of the freshest weight) is folded into `raw_fingerprint`, so an aging theme regenerates the digest once per band crossed rather than every run. The flip maturity gate (`reflection_runner._digest_maturity_gate`) stays on **raw** distinct-session recurrence — it's a safety gate, not portrait content. Decay applies when the digest is regenerated (during reflection), not continuously, and complements the `self_reconcile` pass (deliberate, on-demand retirement) with a passive gradual fade. Decay constants are module-level for now (a later change can wire them to `server_config` `consolidation.wall_clock`). GPU-free self-test covers the weight curve, summed-per-theme survival, faded-drop, and the fingerprint band. File: `server/inference/core/reflection_digest.py`. Commit: `a4b29f1`.

- **Fact contradiction resolution: corrections now supersede the stale fact (manual pass A + automatic at-correction B).** A user correction wrote a *new* `[fact]` instead of replacing the one it corrected, so opposite claims about one subject both stayed live and recallable — a major source of the 1500+ fact pile. Neither `fact_dedup` (merges paraphrases of the same truth) nor `self_reconcile` (judges against the persona digest, not fact-vs-fact) resolves this. New shared core `inference/core/fact_contradict.py` clusters live facts by **subject** (RagEngine's multilingual recall-cue embedding) and, per multi-fact cluster, asks the model which directly CONTRADICT; policy is split like `fact_dedup` — the LLM decides only *what conflicts*, while **recency is mechanical** (the newest fact in a conflict set is the correction and wins; older ones are SOFTENED via the reversible `supersede` op). **A** = a manual, dry-run-first Sleep-tab **"Resolve fact conflicts…"** button over all live facts on the clean base, streaming per-subject-group progress (`resolve_contradictions` → `handle_resolve_contradictions`). **B** = an automatic post-`commit-training` phase (`reflection_service._run_correction_supersede_phase`) scoped via `new_keys` to the facts a run just committed — only changed subjects are judged and only a conflict whose correction is a this-run fact is acted on (pre-existing old-vs-old conflicts are left for A); it runs on the loaded adapter (a per-run clean-base swap would be prohibitive), auto-applies the reversible soften, and is fully best-effort so it never affects the reflection outcome. GPU-free self-test: `python -m core.fact_contradict`. Files: `server/inference/core/fact_contradict.py`, `reflection_writer.py`, `training/ledger.py`, `server/inference/server.py`, `server/inference/core/reflection_service.py`, `server/inference/prompts/fact_contradict_prompt.txt`, `client/ui/sleep_widget.py`, `client/core/backend_client.py`, `documentation/AVA_STATUS.md`. Commits: `c884967` (A) + `0843751` (B).

- **Fix: give the backend a real greedy path (`temperature<=0` → `do_sample=False`), unblocking every deterministic reflect/evaluation pass.** Both `UnslothBackend` generate call sites (the batched branch-replay path and the main `stream_generate`) hardcoded `do_sample=True` and always passed `temperature`/`top_p`, so a pass requesting greedy decoding with `temperature=0.0` crashed inside transformers: `ValueError: temperature (=0.0) has to be a strictly positive float … set do_sample=False`. This silently broke **all** greedy passes on the gemma-4 vision generate path — the phase-two **branch judge**, persona-**digest clustering**, **fact-dedup**, and the new **self-reconcile** — each of which caught the exception and degraded to a no-op (self-reconcile made it visible: 33/33 batches streamed `0 SUPERSEDE`, one `ValueError` traceback per batch in `server.log`, final "nothing parseable came back"). The fix computes `greedy = temperature is None or temperature <= 0` and, when greedy, sets `do_sample=False` and omits the sampling-only warpers (`temperature`/`top_p`/`top_k`/`min_p`); `repetition_penalty`/`no_repeat_ngram_size` (logits processors, valid greedy) stay unconditional, and the chat path (temperature>0) is unchanged. File: `server/inference/core/inference_backend.py`. Commit: `dd6b4a1`.

- **Fix: down-cast float32 LoRA params to the base compute dtype at load, unblocking generation with an adapter.** PEFT saves a trained adapter with float32 A/B matrices, and unsloth's `for_inference` does not down-cast them on the gemma-4 `FastModel` load path; the base is bnb-4bit with a bf16 compute dtype, so the LoRA delta computed `bf16_activations @ fp32_lora_weight` and the **first** generation crashed with `RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float`. It surfaced from the autonomous **outreach** idle job (the first generation path exercised in a headless deploy) but would have crashed chat identically — confirmed against `adapter-20260716_005444`, whose 820 LoRA tensors are all F32. New `UnslothBackend._normalize_adapter_dtype(model)` (called right after `for_inference` in `load`) casts every float32 `lora_` param to the base's compute dtype (embedding dtype as reference), which matches how a normally-loaded unsloth adapter behaves and is numerically standard for inference. No-op with no adapter or an fp32 base. File: `server/inference/core/inference_backend.py`. Commit: `4ea6044`.

- **Self-reconciliation: Ava can soften persona/fact memories she has outgrown (manual, dry-run-first).** Persona self-statements and relational facts accumulate across reflections, and some end up at odds with who she has since become — a stance she has grown past, a fact a later one contradicted. New Sleep-tab **"Reconcile self…"** button (`reconcile_self` → `server.handle_reconcile_self`) judges the live `[persona]`/`[fact]` set against her **current persona digest** (`render_digest_for_judge`) on the **clean base** (`CleanBaseSession`, adapter off) — an evaluation against a digest she authored on the adapter, so it is replay-faithful and immune to a bad adapter, exactly like the branch judge and `fact_dedup`. The clean-base pass emits a per-item KEEP / SUPERSEDE decision (`core/self_reconcile.py`, split pure-logic + one LLM call like `fact_dedup`, GPU-free self-test). **Design decision: soften, not delete, and fully autonomous-but-reversible.** A SUPERSEDE is a new **`supersede`** op in *both* layers, keyed by the shared `content_key`: in `rag_memory.jsonl` it folds like an evict (drops from live recall — `ReflectionMemory._fold`), and in the consolidation ledger (`ConsolidationLedger.supersede`) it **keeps the anchor folded but flags it `superseded`**, which drops it from *active* persona evidence (`reflection_digest.gather_persona_raw` skips it → the raw fingerprint changes → the next digest regenerates without the outgrown pole) and from training (`ledger.live_anchors` + `build_dataset._collect_injections` skip it), while retaining the anchor as **evidence-of-change** so a future arc/`LINES` read can still see who she used to be. Append-only ⇒ a whole pass is reverted by dropping the lines; a later `register` of the same key reactivates it. Two-step UX mirrors Debug-tab dedup: a dry-run preview writes nothing and lists what she'd set aside (persona vs fact, with her reasons), applies on confirm, reloads RAG in place. Requires a persona digest (skips with a note otherwise); refused while the GPU is busy. **Manual trigger only** — wiring it into the Sleep cycle is a separate, later task. Files: `server/inference/core/self_reconcile.py`, `reflection_writer.py` (`write_supersede`), `reflection_memory.py`, `reflection_digest.py`, `server/training/ledger.py` (`supersede` + fold flag), `server/training/build_dataset.py`, `server/inference/server.py`, `server/inference/prompts/self_reconcile_prompt.txt`, `client/ui/sleep_widget.py`, `client/core/backend_client.py`, `server/training/selftest.py`, `documentation/AVA_STATUS.md`. Commit: `6d46fb3`.

- **Merge chats now reconciles a chat that GREW on one box — not just missing chats.** The Migrate tab's *Merge chats* keyed presence on the chat **stem** and skipped any stem present on both sides, so a chat that was appended to on one box after an earlier merge (the common case: an Ava-initiated `initiated_by:"ava"` session that got a reply on one box via `ChatLogger.resume_session`) never propagated its new turns — both boxes kept diverging copies forever. The sync now compares content, not just presence: `GET /chats/manifest` carries a per-stem **fingerprint** (the transcript's ordered per-exchange token list — `exchange_id`, else a sha1 of user/CoT/answer text), and for a stem present on both sides the client relates the two fingerprints — if one is a **strict prefix** of the other the longer side is a pure append and **updates** the shorter copy (pull overwrites the local file, push has the server overwrite its copy); if neither is a prefix the two **diverged** (appended-to on both boxes independently) and the chat is left untouched on both sides and reported as a conflict. A frozen (archive-only) chat is never updated. `_import_chats` re-verifies the prefix relation on the server (authoritative + race-safe: a push only updates a genuinely-shorter server copy), and an update drops the local `.state.json` sidecar the grown transcript makes stale, so the chat re-reflects clean. Result reporting split into new/updated/conflict counts; the confirm dialog + docstrings updated (the old "non-destructive / nothing overwritten" framing was the behavior being changed). The exchange-token + strict-prefix algorithms are duplicated verbatim on both sides (like the existing `_stems`/`_chat_stems` pair) and asserted identical in a self-test covering import/update/diverge/frozen/stale-sidecar. Files: `server/inference/core/mgmt_http.py`, `client/ui/migrate_widget.py`, `CLAUDE.md`. Commit: `f6ac67f`.

- **A wander-born question reaches the user as a thread of Ava's own thinking, not "I read this online."** Questions Ava raises from her self-directed reading (wander / news TIL / lookup) flow into the same open-`[ask]` pool as chat-born ones and surface three ways — passively at session start (`generation._surface_block`), actively via outreach (`core/outreach.py`), or bleeding into a live reply (`rag_memory_prompt.txt`). None of those distinguished a wander-origin ask, so a raised question could leak its internet provenance ("I came across an article about…"), breaking the fiction that her curiosity grew out of the relationship. Fixed at two depths. **(1) Authoring:** `wander_prompt.txt`'s `[ask:user]` bullet now instructs her to phrase the ask in her own voice as an ongoing thread with the person — explicitly *not* "I read an article about…" and with no reference to the page or that she was reading. **(2) Surfacing (origin-aware):** a new shared classifier `reflection_memory.is_self_directed_origin(source_session)` flags asks whose `source_session` carries a self-directed prefix (`wiki`/`til`/`web:`/`lookup`; chat asks carry a timestamp filename), and a reframing note (`prompts/reachout_origin_note.txt`, overridable, loaded via `reflection_memory.load_origin_note`) is injected — into `surface_prompt.txt`'s new `{origin_note}` slot when *any* surfaced ask is self-directed, and into `outreach_prompt.txt`'s new `{origin_note}` slot when the single chosen ask is. A purely chat-origin set gets an empty note (no behavior change). The note tells her to carry the question in as a natural thread of her own thinking rather than announcing where it came from. The passive live-reply bleed (`rag_memory_prompt.txt`) is left as-is — it already says "do not announce that you are consulting notes," and per-item origin conditioning of a bulk RAG block is a separate, heavier change. GPU-free classifier + note-loader verified (prefix matrix, file load, missing-dir fallback). Files: `inference/core/reflection_memory.py`, `inference/core/generation.py`, `inference/core/outreach.py`, `inference/prompts/wander_prompt.txt`, `inference/prompts/surface_prompt.txt`, `inference/prompts/outreach_prompt.txt`, `inference/prompts/reachout_origin_note.txt` (new), `CLAUDE.md`, `documentation/AVA_STATUS.md`. Commit: pending.

- **`train_plateau_epochs: 0` now means a plateau-free warmup+decay (2 epochs), not silently clamped to 1.** The `triangular` LR schedule forces `epochs = train_plateau_epochs + 2`, but `train_plateau_epochs` was clamped `>=1` in two places (`_triangular_lr_fraction`'s `total_epochs = max(1, plateau) + 2` and the config read `max(1, config.get(...))`), so setting `0` in `server_config.json` had no effect — the cycle always ran ≥3 epochs. Both clamps relaxed to `>=0`. At `plateau=0` the trapezoid is just **1 warmup + 1 decay epoch** (`epochs = 2`): a row at epoch-fraction `p` is trained at fraction `p` (warmup) + `1−p` (decay) = **exactly 1** multiplier's worth of LR — the same total LR-integral as a single flat `age_ramp` pass, but globally warmup/decay-smoothed and still order-neutral (the per-row age/contamination weighting is untouched). This gives the operator the "one average pass, but with warmup+decay" option that previously required `age_ramp` (which has no warmup/decay at all). The generalized order-independence invariant (`plateau + 1` per-row sum) already covers it; `test_triangular_lr` was extended to `plateau∈{0,1,3,4}` and verifies the `plateau=0` sum-to-1. Files: `training/train_cycle.py` (`_triangular_lr_fraction`, config clamp, docstrings), `training/selftest.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: `0f3d5ae`.

- **A truncated consolidation drops its cut-off final item instead of persisting a fragment.** When a consolidation pass generates a long `[fact]`/`[ask]` list and hits the token cap (rather than ending on EOS), the completion is sliced mid-string — the last item is a half-written fragment. It was parsed and persisted like any other, so a fragment entered `rag_memory.jsonl`/`weights_persona.jsonl` (recalled at chat time) and would later become a corrupt training row. The token-cap signal was already available (`generate_fn.last_truncated`, from `_backend.last_generation_truncated`) and consumed by the revision pass, but not the consolidation pass. Now `parse_consolidation(text, truncated=…)` (and `ReflectionWriter.write_consolidation(..., truncated=…)`) drops the **final item of the last-emitted section** when truncated — `_last_section_name` locates the section whose header appears last in the body (the one extending to the cut end), and only its final parsed item is removed, so items in earlier sections (and every complete item in the truncated section) are untouched. The reflection runner captures `truncated` after the consolidation generate and forwards it to both the report parse and the durable write, emitting a `pass_warning` naming the drop. Losing the tail of an already-long list is the lesser harm vs. poisoning memory + a build row with a fragment. Other consolidation callers (`til_wander`, `synthesis`) keep the `truncated=False` default (unchanged behavior; the mechanism is available if they need it later). Verified with a multi-section parse test (WEIGHTS preserved when RAG is the truncated section) + the pipeline self-test. Files: `inference/core/reflection_writer.py`, `inference/core/reflection_runner.py`, `training/selftest.py`, `documentation/AVA_STATUS.md`. Commit: `f34a992`.

## 2026-07-17

- **Manual reach-out triggers ignore the shared rate limit.** The 1 h reach-out rate limit (`core/reachout_gate.py`) exists to stop the *autonomous* jobs greeting the user with several unprompted messages in one silent window. But it also silently muzzled the Sleep-tab debug buttons: if an autonomous reach-out (or a prior manual one) had fired within the hour, pressing **Reach Out** / **Chat reach out** / **Check In** ran the whole decision pass and then returned a `reachout_cooldown` skip instead of sending — confusing, since the operator explicitly asked for it. Added a `bypass_cooldown` parameter (default `False`) to `run_outreach_decision_blocking` / `run_synthesis_blocking` / `run_checkin_decision_blocking`; the three manual handlers (`handle_outreach_now`/`handle_synthesis_now`/`handle_checkin_now`) pass `True`, so a manual trigger always composes+sends. Only the *check* is bypassed — a manual send still calls `mark_reachout()` afterward, so an autonomous job won't pile on right behind it. The autonomous idle callers pass nothing, so their throttling is unchanged. Files: `inference/core/outreach.py`, `inference/core/synthesis.py`, `inference/core/checkin.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: `fe0c0aa`.

- **Training review deduplicates the contamination copies — one exchange, one entry.** A cap-age exchange renders **twice** into `sft_render.jsonl` (`build_dataset._contamination_rows` §5e: the masked `cap−dose` row plus its `unmask_user` `dose` copy, *identical* `messages`, differing only in `lr_multiplier` and which turns the loss covers), so the review tab — which lists one entry per render row — showed every aged exchange twice, doubling the list an operator has to read through with a row that is a training-schedule fact, not a second exchange to review. `TrainingLoadWorker` now folds rows sharing a `_dedup_key` into one entry: `(source_session, exchange_index)` where the provenance exists, else the rendered `(query, assistant)` text (wander rows / pre-provenance snapshots that lack the fields — identical text there means an identical training row anyway). The **first** row wins, which is the masked primary since `_contamination_rows` emits it first, so the preview/regenerate path still targets the row it always did. The collapse is **named, not hidden**: a folded entry is tagged `×N` in the left list (`_item_text`) and the status line reads `N entries (from M rows — duplicate copies collapsed)`. **Interpretation change:** left-list entry numbering is now over the deduplicated list, so it is no longer a line index into `sft_render.jsonl` (the per-entry `source_session`/`exchange_index` remains the exchange mapping, and search/ratio-filter still compose over the same never-rebuilt list). Verified against the newest local snapshot (`build-20260716-131353`): 897 rows → 487 entries, 410 collapsed exchanges == 410 extra rows removed == the file's 410 `unmask_user` rows, max 2 rows per `(session, exchange)` — a 1:1 match with the contamination copies and nothing else merged. Files: `client/ui/training_review_widget.py`, `CLAUDE.md`. Commit: `72d6145`.

- **Idle jobs made genuinely independent — check-in was structurally unreachable.** Follow-up to the 2026-07-16 scheduler extraction, which made the jobs *look* independent (own `interval_s`, own `_active` flag) while three cross-job couplings kept them from firing. **(1) The starvation.** The shared 5 h reach-out cool-down was wired as the idle-job `ready` gate `_reachout_gate_open` on outreach + check-in. But a `ready`-blocked job never advances `_last_ran`, while every job re-arms on its **own 1 h** interval — so outreach (first in registration order, which the serial `for job in _jobs` loop made a permanent priority) re-stamped the 5 h gate every hour, before check-in's gate ever opened. Check-in could only run if outreach *and* synthesis both declined for a full 5 h stretch; in practice it never ran, and synthesis never DM'd. **(2) The mechanism.** `run_loop`'s single shared tick walked jobs in registration order and `break`-ed the whole tick when `host_busy()` flipped, so a late-registered job lost its turn to whatever ran first. Replaced with **one supervising `asyncio.Task` per job** (`_job_loop`), each on its own clock; the single `_gpu_busy` bool becomes an `asyncio.Lock` that is now the *only* thing serializing jobs — co-due jobs all fire and queue on the GPU. Every precondition is re-checked *after* acquisition (a job can wait minutes behind a wander pass, during which the user may start typing). `IdleJob` gains an optional per-job `idle_seconds` override so a future high-frequency job needn't inherit the global 1 h idle window — the point of the exercise being to run each event at its own frequency. **(3) The policy.** `reachout_gate` stops being a scheduler gate and becomes an **output rate limit** — *one unprompted message per hour regardless of origin* (`COOLDOWN_SECONDS` 5 h → 1 h), checked by each job body immediately before it writes a session (the pattern synthesis already used), so every job always runs its pass, fills its pool, and consumes its interval; only the outward message is throttled, returning a `reachout_cooldown` skip. **The rule this establishes:** a job's `ready`/`consumed` policy may read only its own state; cross-job policy belongs at the point of effect, never in the scheduler. Also fixed two classifier bugs found in the same review: outreach's `resolved` outcome (which runs a full decision generation and evicts a self-answered ask) matched no branch in `_outreach_consumed_interval` and fell through to un-`consumed`, re-firing every 5-min poll instead of hourly; and wander had **no** `consumed` classifier at all, so a network blip (no substantive page) burned its whole interval. The self-test now asserts the independence property directly — two co-due jobs both dispatch, each only under the GPU lock. Behavior verified by simulation: three co-due reach-out jobs all fire, exactly one message goes out. Files: `inference/core/idle_scheduler.py`, `inference/core/reachout_gate.py`, `inference/core/outreach.py`, `inference/core/checkin.py`, `inference/server.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: `871e783`.

- **Fix: an encounter or gossip transcript read as the user speaking.** Check-in measures user silence from disk (`_last_user_turn_dt` — freshest chat file whose transcript has a real user turn), and `_has_user_turn` excluded only unanswered `initiated_by:"ava"` openers. But `encounter_run` and the serving-side gossip logger both call `ChatLogger.start_session` **without** any marker, and in both a `user_prompt` is *another model's* reply — so an Ava↔peer conversation was indistinguishable from a live user chat. Two consequences: an encounter reset the silence clock and suppressed check-in for the full threshold though the user never spoke, and `_recent_chats` (same filter) fed those transcripts into the "recent conversations" window, so Ava summarized a peer model's turns as the user's. Added an optional session-level `interlocutor` field to `ChatLogger.start_session` (`"ai"` for encounters + served gossip; absent on an ordinary chat), and `_has_user_turn` now ignores those sessions. Deliberately scoped to things measuring *the user's presence*: reflection still treats an encounter as an ordinary reflectable session, and training is unaffected. This is the same class of bug as the `initiated_by` exemption — Ava's own activity answering "has the user gone quiet?" on the user's behalf. Files: `inference/core/chat_logger.py`, `inference/core/encounter_run.py`, `inference/core/generation.py`, `inference/core/checkin.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: `871e783`.

- **Curb day-0 past-chat RAG dominance (recent chat drowning the live query).** A recent, still-unreflected chat sat at chat-RAG modifier `1.0` for its whole first day (the wall-clock crossfade only fades a bundle *after* it is frozen, and `rag_weight_hours` is barely under 1.0 across day 0), so a near-duplicate new query pulled in its full Q+A at full weight and the model repeated the prior answer instead of addressing the new query. Three layered fixes: (1) raised the chat-RAG cosine gate `RagEngine._MIN_SCORE` `0.30 → 0.45` (drops loose same-domain hits; reflection/wander channels have their own thresholds, untouched); (2) lowered live-chat `top_k` `3 → 2` at the `handle_generate` call site (a lone strong hit can't be reinforced by two weaker ones; reflection/consolidation retrieval keeps the default 3); (3) a **day-0 fresh-window discount** layered under the frozen crossfade in `_chat_modifier`: an hourly time droop (`fresh_time_weight`, `1.0` at age 0 falling linearly over `fresh_horizon_h` **24 h** then held; combined with the frozen fade via `min` so the two never double-count, and applied off the *raw* wall-clock age so it bites the unreflected recent chat) times a **recency-rank penalty** (`chats_since_weight`, each newer chat opened since costs `chats_since_penalty` **0.1**, floored at `chats_since_floor` **0.4**). The droop's floor is **derived from the main crossfade, not absolute**: an unreflected day-0 chat has *not* reached the weights, so it must fade gentler than `rag_weight_hours` already would (which hits 0.75 at 24h — a "partly consolidated" level that misrepresents a fresh chat). Floor = `1 - fresh_droop_frac * (1 - rag_weight_hours(horizon))` = `1 - 0.5*(1 - 0.75)` = **0.875** at 24h with the defaults, tracking `rag_cap_age_h` automatically. `chats_since` = how many other chats are newer (0 for the most recent), computed at index-build time from the timestamp-sorted chat files — so it refreshes each time a new chat is opened (the rebuild trigger), matching "recompute decay when opening a new chat." **Behavioral note:** the immediately-previous chat is untouched by the count term (chats_since 0) and only droops by the hours, so genuinely-relevant immediate context still surfaces; a chat from earlier the same day that you have since moved past is suppressed by both terms. All four knobs live under `consolidation.wall_clock.fresh_window` in `server_config.json` (`droop_frac`/`horizon_h`/`chats_since_penalty`/`chats_since_floor`); set `droop_frac=0.0` + `chats_since_penalty=0.0` to disable. Pure/GPU-free helpers with selftest coverage (`fresh_time_weight`/`chats_since_weight`). Files: `inference/core/rag_engine.py`, `inference/core/generation.py`, `training/decay.py`, `training/selftest.py`, `documentation/AVA_CHANGELOG.md`. Commits: `8866d50` (MIN_SCORE + top_k) + `d88dbd2` (fresh window) + pending (derived floor).

- **Fix: the Training review live connect-back read the wrong chats dir.** As first committed (`b864b38`) `_live_chat_sidecar_dirs` pointed at `server/inference/data/hot/chats` (per a stale layout comment), which is empty — the chat corpus was long ago moved to the ordered `server/data/chats` home (`training/reflections_path.hot_chats_dir`, where the apply handler writes and 96 live sidecars sit). So regenerated/locked exchanges never lit up: no snowflake, stale preview. Repointed to `server/data/chats` + `server/data/archive/chats`; verified against the real on-disk `locked`/`manual_regen` records. (The apply path itself was always correct — this was purely the read-back path.)

- **Training review reconnects to live sidecars: frozen entries show their regenerated content + a snowflake.** The tab renders a *frozen forensic snapshot* (`sft_render.jsonl`), so an operator's manual regenerations (which write the LIVE sidecar) were invisible there until a rebuild. Now `TrainingLoadWorker`, on load, also reads each row's live `.state.json` off disk (`_live_chat_sidecar_dirs` → `server/data/chats` then `server/data/archive/chats` — the ordered chat-corpus home per `training/reflections_path.hot_chats_dir`, memoised per chat; same-host/dir assumption already used for the snapshot read). When the source exchange's live record is **`locked`** (human-validated / regenerated), the tab shows that record's target — the regenerated CoT/reply, split from `target` — in the preview instead of the stale snapshot row, and the left-list entry gets a **snowflake ❄ icon** (a `QIcon` drawn from the glyph, with a transparent same-size placeholder on the rest so text stays aligned; the status line shows the frozen count). `_disp_cot`/`_disp_answer` (live-preferred) back the preview, search, and ratio filter uniformly, and a selected frozen row notes "showing the live sidecar target." `_on_apply_done` marks the just-applied row `locked`/live + swaps its icon, so a repair is reflected immediately and survives a Refresh. Files: `client/ui/training_review_widget.py`, `CLAUDE.md`. Commit: pending.

## 2026-07-16

- **Fix: "Corrupt CoT only" regen keeps the reply you're looking at, not the raw transcript reply.** The manual regenerate always does a full re-answer (a faithful CoT can only be authored *with* a reply), and for a CoT-only apply the new reply is discarded and the new `<think>` is grafted onto the kept reply. But the "kept reply" was `exc.get("assistant_response")` — the **raw transcript** reply — which for an **empty-CoT row differs from the answer shown in the tab** (an empty CoT means the trained target was an answer-only revised/IDEAL, so the displayed answer ≠ the raw reply). Result: applying a CoT-only regen on such a row silently reverted the reply to the raw original, which read as "the reply got regenerated." Now the client sends the **displayed answer** (`e["answer"]`, the current trained target's reply) as `original_reply`, and the server grafts the new CoT onto *that* (falling back to `assistant_response` only if absent), with an empty-kept-reply guard. The review dialog's second box now shows, for a CoT-only regen, the **kept reply (unchanged)** instead of the discarded regeneration — so the operator judges whether the new CoT coheres with the reply it will actually be paired with. Files: `inference/core/session_ops.py`, `client/core/backend_client.py`, `client/ui/training_review_widget.py`, `CLAUDE.md`. Commit: pending.

- **Per-exchange human-validation lock — a manually repaired exchange is preserved across re-reflection AND revisit.** The manual regenerate→apply flow first shipped freezing the *whole chat* (`ChatSidecar.mark_reflected`) to protect the hand-authored target — but that had two holes: it suppressed reflection of the chat's *other* exchanges, and it did **not** protect against a **Revisit** run, which deliberately bypasses the session-level reflect-once freeze and re-derives every exchange from the original (still-corrupt) transcript, silently overwriting the repair. Since the revisit head-phase runs on every normal Sleep and picks random aged chats, a repaired exchange could be re-poisoned. Replaced with a **per-exchange `locked` flag** on the sidecar record (`ChatSidecar.write_verdict(locked=True)`, sticky/OR-ed so a stray write can't clear it; `locked_exchange_indices` reads them). The reflection runner drops locked exchanges from the revision work-list at the top of `_run_revision_for_session` (emitting an `exchanges_locked` event) and excludes them from the ETA precount — **honored unconditionally, so a revisit skips them too** — while the rest of the chat re-reflects normally. Locks are read from the **live** sidecar directly (`_locked_exchanges` uses the fallback/live dir, not the staging copy) so a continue-staging workspace can't shadow them. `apply_regenerated_exchange` now writes `locked=True` instead of freezing the chat. This is the workflow for a corpus with ~15% corrupt exchanges: regenerate each with the loaded adapter until coherent (persona intact), lock it, and the original poison can never return via a future build or Revisit. GPU-free test covers lock write/read, stickiness against a later non-locking write, and the runner's work-list filter. Files: `inference/core/chat_sidecar.py`, `inference/core/reflection_runner.py`, `inference/core/session_ops.py`, `client/ui/training_review_widget.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

- **Training review: regenerate-and-repair an exchange in place, with the loaded adapter.** The tab's flag-only **Corrupt CoT**/**Corrupt reply** toggles (which just set `corrupt_*` flags and deferred repair to the next Sleep) are replaced by a **regenerate → review → apply** flow that fixes a bad row *now*. UI: **Corrupt CoT**/**Corrupt reply** checkboxes + a **temperature** field + a **Regenerate** button (the old "Hide reset sidecars" filter, a companion to the removed mark-corrupt-deletes-sidecar behavior, is dropped; the ratio filter + search stay). **Regenerate** (`regenerate_exchange` → `generation.handle_regenerate_exchange`, dispatched to the GPU executor) re-answers the selected row's source exchange with the **currently loaded adapter** by rebuilding the exact pre-answer conversation (`reflection_source.build_ideal_messages`) and generating a fresh `<think>…</think>` reply through the same clean dialogue seam the reflection IDEAL pass uses (`_make_sync_reflect_generate`, `messages_override`+`disable_rag`); it refuses while a reflection run/encounter owns the GPU (the adapter may be swapped mid-reflection) and writes nothing. The fresh CoT+reply are shown in a modal `RegenReviewDialog` (read-only) with **Apply**/**Cancel**. **Apply** (`apply_regenerated_exchange` → `session_ops.handle_apply_regenerated_exchange`) assembles a **faithful** target — `corrupt_response` ⇒ new CoT + new reply (the pair was generated together); `corrupt_cot` only ⇒ new CoT grafted onto the trusted original reply — so a written target is never a think/answer mismatch, then writes it straight to the chat sidecar via `ChatSidecar.write_verdict` (`target_source:"revised"` so `dialogue_source` trains it verbatim and `_pick_corrupt_chat` treats it repaired; `target_kind:"manual_regen"`/`target_generation:"chat_manual_regen_v1"` for the forensic audit), **freezes** the chat (`mark_reflected`, so a later Sleep can't clobber the hand-authored target), and clears any lingering corrupt flags on the exchange. The client updates the in-memory row on apply (note: training reads the LIVE sidecar, not the forensic snapshot the tab renders). Path-guarded to hot/chats; refuses the active session + a live reflection run (same guards as `mark_corrupt`). New client RPCs `BackendClient.regenerate_exchange` (600 s timeout — GPU) / `apply_regenerated_exchange`; new dispatch entries in `server.py`. The flag-only `mark_corrupt` path stays on the server for automatic (next-Sleep) repair. GPU-free round-trip test covers `build_ideal_messages` shape + sidecar write/freeze + `build_dialogue_anchor` reading the manual target verbatim. Files: `client/ui/training_review_widget.py`, `client/core/backend_client.py`, `inference/core/generation.py`, `inference/core/session_ops.py`, `inference/server.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

- **CoT regeneration for a kept, corrupt-CoT reply (approach #3) — no more answer-only erosion on repair.** When the operator flags an exchange's **CoT** corrupt (not the reply) in the Training review tab, the reply is trusted — only the thinking was a logging/generation bug. Previously reflection just *blanked* the CoT and, if the judge kept the reply, trained it **answer-only** — which is itself the reasoning-channel erosion the from-scratch build warns about (`resolve_revision_target`: an answer-only target under a thinking-enabled prompt teaches the model to skip reasoning). Now a corrupt-CoT exchange the judge **keeps** is repaired: `reflection_runner._regenerate_cot_for_kept_reply` re-answers it through the exact clean-IDEAL seam (`_generate_ideal_reply` → a fresh `<think>`+reply authored *together*, faithful by construction), keeps **only the fresh `<think>`**, and **grafts it onto the original reply** — but only when the re-answer reproduces that reply within `_RECOT_SIMILARITY_FLOOR` (**0.85** cosine over the RAG embedder, new pure helper `branch_replay.embed_similarity` comparing the two *whole* independently-generated replies, unlike branch filtering's suffix compare). Above the floor the fresh thought genuinely leads to ~that reply, so the graft is faithful; below it (or with no embedder / no usable re-answer) it **falls back to answer-only** — the prior behavior, never pairing a thought with a reply it did not produce — recorded as a new `cot_regen_fallback` stat. The graft rides the IDEAL seam so it resolves to `target_source=="revised"` (which `dialogue_source` already trains verbatim and `_pick_corrupt_chat` already treats as repaired — no downstream changes), but its provenance is retagged **`cot_regen` / `chat_recot_v1`** (via a new `recot` flag on `write_revision`, mirrored in the runner's report resolution) so the forensic audit isn't misled into "chat_reanswer_v1": the reply is the *original*, only the thought is new. Branching is skipped by the existing CoT-less-with-usable-IDEAL rule. This runs only on a **keep** verdict — a `revise` (or a corrupt *reply*) already re-answers in full — and only when there's no language drift. Wired through `execute_run(similarity_fn=…)` and threaded to `_run_revision_for_session`; the similarity callable is built from the RAG embedder in `reflection_service` (all three run paths: staged, revisit sub-run, dry) and the headless `reflection_run.py` CLI; absent ⇒ safe answer-only fallback. New chosen-target bucket `cot_regen` + the `cot_regen_fallback` line surface in the Sleep tab's RUN STATISTICS. GPU-free: `reflection_stats` self-test green; ad-hoc tests cover the split/graft/resolve/provenance round-trip and `write_revision(recot=True/False)`. Files: `inference/core/reflection_runner.py`, `inference/core/reflection_writer.py`, `inference/core/reflection_stats.py`, `inference/core/reflection_service.py`, `inference/core/branch_replay.py`, `reflection_run.py`, `training/dialogue_source.py`, `client/ui/sleep_widget.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

- **Check-in reviews the recent window by per-chat summarization, not raw concatenation.** As first shipped, check-in built its "last N chats" context by concatenating bounded raw transcripts under a 9000-**char** cap, newest-first, dropping any older chat whole once the cap was hit (`_format_recent`). In practice one substantive recent chat (up to ~6 exchanges × ~1000 chars) ate most of the budget, so chats #2–#5 were dropped and Ava referenced only the newest — no synthesis, and not even token-aware (a char guess, unlike consolidation's `prepare_prompt` fitting). Replaced with **per-chat summarize → synthesize**: `_summarize_recent` compresses each of the `checkin.recent_chats` (default 5) conversations to a 2-3 sentence recap ("what it was about + where it left off") via its own cheap pass (`checkin_summary_prompt.txt`, `temp 0.5`, 768-token cap, RAG off, `RECAP:` parsed with the CoT stripped), and only the compact recaps — assembled oldest→newest by `_render_digests` — feed the decision pass, whose prompt now says to look *across* all of them. So the whole window is represented regardless of individual chat length. Recaps are **cached per `(filename, mtime)`** (`_SUMMARY_CACHE`, cap 512) so an hourly check-in that keeps declining re-runs only the single decision pass, not N summaries, over the unchanged window; a chat that gains a turn (new mtime) is re-summarized; a summary failure falls back to a bounded raw excerpt so the chat is still present. Cost is N+1 generations on a *fresh* window (mtime-cached thereafter). The manual "Check In" debug stream gains `summarizing {i, n, session}` and `deciding` stage markers (per-chat summaries emit a marker, not token deltas; only the decision pass streams reasoning). GPU-free smoke test (fake generator) asserts all 5 chats are summarized, the decision reasons across them, and a second run is a full cache hit (0 summaries). Files: `inference/core/checkin.py`, `inference/prompts/checkin_summary_prompt.txt` (new), `inference/prompts/checkin_prompt.txt`, `client/ui/sleep_widget.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

- **Autonomous idle jobs split into a generic scheduler + per-job frequencies.** The three between-conversation GPU jobs — ambient *wander*, Ava-initiated *outreach*, and *synthesis* — were each a bespoke `_maybe_autonomous_*` coroutine in `server.py` that re-derived the same mutual-exclusion check (an OR over every *other* subsystem's `_active` flag, duplicated in both the per-job guard AND that subsystem's injected `host_busy` lambda — ~8 OR-expressions total, O(N²) in the job count) and shared ONE hard-coded `_IDLE_WAKE_SECONDS = 3600` cadence with no way to run different events at different frequencies. Extracted the *mechanism* into `inference/core/idle_scheduler.py`: it owns the shared idle clock (`mark_activity`), the crash-safe PID wake-lock, and a **single** `_gpu_busy` mutual-exclusion flag that replaces the cross-module OR-chains — `host_busy()` is now `_gpu_busy or external_busy()`, one definition injected into every subsystem, where `external_busy` names only the GPU owners that are NOT scheduler-run idle jobs (reflection runs, encounters, and the manual outreach/synthesis debug triggers). Each job registers as an `IdleJob` descriptor carrying its **own** `interval_s` (the requested per-event frequency), an optional `ready` gate (wander's token budget), and an optional `consumed` classifier (a cheap "no open ask yet" outreach bail returns False so it retries next 5-min poll instead of sleeping a full hour). The old shared-clock coupling — wander calling `mark_activity` in its `finally` so it had to run LAST or it would gate out the others mid-tick — is gone: autonomous jobs no longer touch the idle clock (each self-throttles on its interval), so the fragile ordering constraint and the two hand-rolled per-lane cooldowns (`_last_outreach_activity`/`_last_synthesis_activity`) were deleted. Adding a new idle-ONLY job is now a `register(IdleJob(...))` call with zero edits to any sibling. All three jobs keep `interval_s=3600.0`, so behavior is unchanged until an interval is bumped. GPU-free self-test (`python -m core.idle_scheduler`) covers interval/ready gating, `host_busy` composition, and the consumed classifier. Files: `inference/core/idle_scheduler.py` (new), `inference/server.py`, `inference/core/reflection_service.py`, `CLAUDE.md`, `documentation/AVA_STATUS.md`. Commit: pending.

- **Check-in: a fourth autonomous idle job — Ava reaches out after a stretch of user silence.** Alongside outreach (raise a queued `[ask]`), synthesis (re-read an *aged* chat) and wander, `core/checkin.py` adds a reach-out with distinct DNA: its trigger is **elapsed user silence** and its context is the **recent** window. Registered as a fourth `idle_scheduler.IdleJob` (`name="checkin"`, `interval_s=3600`) — no bespoke wrapper. Silence is measured from disk (`_last_user_turn_dt` — the freshest real user exchange by chat-file mtime, ignoring unanswered Ava openers so check-in's own written sessions don't reset the clock; restart-safe) against `checkin.silence_threshold_hours` (default **5**), deliberately *not* the scheduler's shared idle clock, which Ava's own idle jobs reset — so it means "the *user* went quiet," not "the box idled." The blocking pass reviews the last `checkin.recent_chats` (default 5) conversations rendered into a bounded transcript (`_format_recent`) and makes a two-outcome (yes/no) decision (`checkin_prompt.txt`, `{hours}`/`{user}`/`{recent}`; no `resolved` — that is outreach's ask-specific janitor step). A **yes** writes a reversed `initiated_by:"ava"` session (exchange 0 under `(initiative)`, synthetic impulse) identical in shape to outreach/synthesis — lands in the list badged, adopted in place on open, and masked from training/probe by the existing generic `build_dialogue_anchor` `initiated_by=="ava"` rule (comment generalized from "outreach" to the three reach-out subsystems). The scheduler's idle window is a coarse pre-gate; the real silence gate lives in the job (returns a cheap `insufficient_silence` skip, classified un-`consumed` by `_checkin_consumed_interval` so it retries next poll rather than sleeping a full interval). **New shared reach-out cool-down** (`core/reachout_gate.py`, process-local monotonic, `COOLDOWN_SECONDS` = 5 h): whoever writes an unprompted session stamps `mark_reachout()`, and no reach-out lands within that window of the previous one, so the user is never greeted by two cold-opens in one silent stretch. It is enforced as the shared `ready` gate `_reachout_gate_open` on the **outreach** and **check-in** idle jobs (skip the whole run under cool-down); **synthesis** is deliberately un-gated at the job level and instead checks the same cool-down itself right before composing, so its pool-filling analysis still runs and only the DM is deferred (`reachout_cooldown` skip). A Sleep-tab **"Check In"** debug button (`checkin_now` → `handle_checkin_now`) **forges the silence period** (default = threshold, clamped up to real elapsed so it stays realistic) so the operator can watch her decide as if the threshold had been crossed even when it hasn't, streaming her reasoning (`checkin_stage`/`checkin_chunk` → `checkin_done`); the manual trigger sets `checkin._checkin_active`, which is listed in the scheduler's `external_busy` so it mutually excludes the autonomous jobs. GPU-free smoke test covers the decision parse, the silence clock (ignores the unanswered opener), and the recent-window formatting. New config block `checkin: {silence_threshold_hours, recent_chats}`. Files: `inference/core/checkin.py`, `inference/core/reachout_gate.py`, `inference/prompts/checkin_prompt.txt`, `inference/server.py`, `inference/core/outreach.py`, `inference/core/synthesis.py`, `training/dialogue_source.py`, `client/core/backend_client.py`, `client/ui/sleep_widget.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

- **Interrupted reflection runs no longer re-reflect their already-completed chats.** A normal Sleep run writes everything to the staging workspace and only commits to live on a `completed` run; a stop/crash before that left the completed chats' frozen sidecars only in staging, which the next run's `clear_staging` (default) wiped — so those chats re-reflected from scratch. Added a durable per-chat **completion checkpoint** (`data/hot/reflection_checkpoint/`, deliberately outside `reflection_staging/` so `clear_staging` never touches it): the runner's new optional `on_session_committed(filename)` callback fires right after each session's reflect-once freeze (never for skipped/dry/consolidation-only/stopped sessions), and `reflection_staging.checkpoint_completed_session` copies that session's frozen sidecar plus a refreshed snapshot of the run's cumulative staged `rag_memory.jsonl`/`weights_persona.jsonl`/`consolidation_anchors.jsonl` (sessions run sequentially, so the staged delta files hold exactly the completed work — a rolling snapshot that folds once without per-session line slicing). At the **start** of every non-dry run, `fold_checkpoint_to_live` folds a surviving checkpoint into live — copies each frozen sidecar to `server/data/chats` (skipping any already frozen live) and appends the cumulative deltas to the live op-logs, rebuilds the live reflection RAG (same fresh-`RagEngine`+`build_index_async` pattern as `run_stage_merge_rag`), then deletes the checkpoint (consume-once, so no double-append) — running before revisit/ingestion so the recovered conclusions are live RAG context for the main pass; those chats then read frozen (reflect-once) and are skipped. A `completed` run discards its own checkpoint (staging is the source of truth), so folding only ever recovers an *interrupted* run. Per-chat granularity, matching the whole-sidecar freeze/corrupt model; the end-of-run clean-base judge override is not captured (it never ran on an aborted run). GPU-free round-trip test covers the cumulative-snapshot fold, uncompleted-chat exclusion, survival across a staging wipe, consume-once idempotency, and no-op discard. Files: `inference/core/reflection_staging.py`, `inference/core/reflection_runner.py`, `inference/core/reflection_service.py`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`. Commit: pending.

## 2026-07-15

- **IDEAL CoT now comes from a clean normal-dialogue re-answer, not retrospective reflection.** The revision/revisit prompts are judgement-only (`VERDICT`/`WHY`/`LANG_DRIFT`/`PERSONA_TARGET`); they no longer ask the model to write `IDEAL:` while looking at the rejected CoT/reply. On `revise`, `ReflectionRunner` now builds the exact pre-answer conversation that training later reconstructs—stored system prompt, answer-only prior turns, final user message—and invokes the normal chat cleaner with RAG disabled. The rejected CoT/reply, Meta feedback, reflection prompt/RAG, judgement CoT, and `WHY` are structurally absent. Legacy/prompt-override inline `IDEAL:` output is still parseable for archived compatibility but is discarded by `parse_revision_judgement`. A malformed/truncated re-answer retries once from the same clean messages with perturbed sampling; confirmed language drift may add only a neutral language constraint. Failure is fail-closed (`revised_missing_ideal`, no sidecar/anchor), never fallback-to-original. Sidecars, anchors, and rendered/quarantined training-row provenance now carry `target_kind` + `target_generation` (`ideal/chat_reanswer_v1`, `branch/branch_replay`, `original/original`), and fact placement reads the resolved target CoT rather than the rejected source CoT. Server and headless generators share the structured-message override contract; Sleep reports the re-answer phase/provenance. GPU-free selftest asserts exact training-prefix parity and absence of rejected/feedback/judgement markers across both attempts. This is prospective only: historical sidecars/snapshots/archives/adapters are unchanged. Files: `inference/core/reflection_source.py`, `generation.py`, `reflection_runner.py`, `reflection_writer.py`, `chat_sidecar.py`, `reflection_run.py`, `prompts/revision_prompt.txt`, `prompts/revisit_prompt.txt`, `training/dialogue_source.py`, `training/train_cycle.py`, `training/selftest.py`, `client/ui/sleep_widget.py`. Commit: pending.

- **Training's no-logits fused CE loss restored; `train_max_seq_length` back to 2048 → 4096.** Root-caused the corpus-size-correlated degeneration (language drift + runaway/unterminated generation, worsening as the corpus grew). Primary historical mechanism: between ~2026-07-05 (cap lowered to 2048 after the first 31B OOM) and 2026-07-14 (the `irreducible_over_cap` quarantine, commit 10e6521), a row that could not fit the cap even after history trimming was handed to TRL anyway, whose head-slice truncation cut the **tail** — the end of the trained answer, the turn terminator, and the EOS — while the labels stayed unmasked: those rows literally trained long answers that never terminate. Calibrated against the 07-14 quarantine ground truth, the affected share grew ~16% → 46% of rows per build as reply length inflated through the model→chat→corpus feedback loop (average answer 1.4k → 3.9k chars, 06-21 → 07-13). **Interpretation note: every build promoted in that window trained on answer-chopped rows; transcripts generated under those adapters (and the `revised` IDEAL targets derived from them) carry the inflation/drift.** The OOM that forced the 2048 cap was itself a side effect of the earlier `UNSLOTH_COMPILE_DISABLE=1` recompile-crash fix: unsloth_zoo's auto-compiler quick-exits under `"1"` before *installing* any patched class — including the fused lm-head / no-logits cross-entropy — so the vanilla forward materialized the full seq × 262k-vocab fp32 logit buffer every step. Fix, validated end-to-end on Gemma4-31B: (1) `UNSLOTH_COMPILE_DISABLE="partial"` in `train_cycle` — source patches install, but every generated `@torch.compile` is emitted as `@torch.compiler.disable`, so the recompile-limit crash stays impossible (inference keeps `"1"`; it gains nothing from the loss patch); (2) unsloth + unsloth_zoo upgraded 2026.6.9/2026.6.7 → 2026.7.2 — the older zoo's fused-CE chunk sizer could round to a single uncapped chunk and re-materialize full logits (OOM'd at step 1); (3) `UNSLOTH_CE_LOSS_TARGET_GB=1.5` — 2026.7.2's default per-chunk target (`min(free/2, 4 GB)`) still admits a 3.31 GiB single chunk whose backward doubles the transient past the ~5 GiB the 31B 4-bit base leaves free on a 32 GB box. Result: the build corpus recovered from 465 retained rows to 862 (the quarantined long half — the newest, longest exchanges — trains again), stable at ~27/32.6 GB. Files: `training/train_cycle.py`; `server_config.json` `train_max_seq_length` 2048 → 4096 (untracked).

## 2026-07-14

- **Outreach can self-resolve a stale question instead of only raising or deferring it.** The Ava-initiated outreach decision pass previously had two outcomes (`yes` → write a reversed session; `no` → do nothing), so when Ava declined because she had *already learned the answer* since the question was queued (new data landed in memory/RAG after it was asked), the open `[ask]` stayed live and the idle heartbeat kept re-picking it — re-deliberating a question that was already answered. The pass now offers a third outcome, `resolved` (with an `ANSWER:` field): it formally closes the ask via the new `ReflectionWriter.write_resolution` — an `op:"evict" reason:"resolved"` record keyed by the question's `content_key`, the same eviction the consolidation `[resolved]` path uses — so `ReflectionMemory` drops it from the live fold and it is never surfaced or re-picked. The answer is stored on the evict for provenance only, **not** re-distilled into a weights/RAG item, because the knowledge that let her answer is already in memory (distilling would duplicate it). The `resolved` outcome flows through both the autonomous idle path (a janitor for obsolete questions) and the Sleep-tab "Reach Out" debug trigger (`outreach_done` gains `resolved`/`answer`; the widget renders it). Parser (`_parse_decision`) now returns `(decision, opener, answer)` and normalizes the CoT-stripped `DECISION:` line to `yes`/`no`/`resolved`, defaulting a truncated/absent block to `no` so a cutoff never reads as a spurious resolve. Files: `inference/core/outreach.py`, `inference/core/reflection_writer.py`, `inference/prompts/outreach_prompt.txt`, `client/ui/sleep_widget.py`, `CLAUDE.md`. Commit: pending.

- **Multilingual RAG correctness pass (chat + reflection + TIL/wander).** Six coupled retrieval failures were fixed. (1) Historical reflection no longer lexically compares typed external provenance (`wiki:…` / `til:…`) with a chat filename; reflection-memory entries carry their op-log insertion timestamp into the index and are admitted only when that knowledge actually existed before the reviewed session (legacy chat rows fall back to their filename clock), so an applied TIL becomes available to subsequent chats/reflections without leaking backward. (2) Wander now gates on raw cosine and applies `wander_rag_weight_hours` only to ranking; the old `cosine × 0.1 >= 0.15` final step was impossible even at cosine 1.0. (3) RAG moved from English-centric `all-MiniLM-L6-v2` to CPU-hosted `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; staying in the MiniLM family avoids silently invalidating the branch/persona/fact cosine gates that reuse this embedder, while CPU placement keeps model VRAM at zero. (4) Long English/Russian/mixed queries, chat prompts, and wander sources are split into conservative overlapping passages, so the encoder cannot silently discard their tail; passage hits de-duplicate to one exchange/article, displayed turns are bounded, and wander injects only its best bounded passage + bounded reaction rather than a full article. Consolidation now retrieves reflection memory separately for every fitted content chunk instead of reusing one session-wide query/context. (5) Historical reflection searches the complete candidate set before applying chronology/decay and then takes top-k, so filtered top hits are backfilled. (6) The active transcript is excluded at index build, incremental add, and query time; resumed in-place chats cannot RAG-inject their own already-visible turns. Pure policy tests cover typed chronology, mixed-language chunk shape, and wander admission without optional ML packages; optional FAISS tests cover backfill, active-session exclusion, multilingual chunking, and the final wander step. Files: `inference/core/rag_engine.py`, `inference/core/rag_policy.py`, `inference/core/reflection_runner.py`, `training/selftest.py`, `snapshot_state.py`, `AGENTS.md`, `CLAUDE.md`, `documentation/AVA_DESIGN.md`, `documentation/AVA_STATUS.md`. Commit: pending.

- **Training rows can no longer lose their assistant tail/EOT, and LR metadata stays row-bound after masking.** The prior message-level truncator returned every irreducible `system + final user + target` row even when it exceeded `train_max_seq_length`; TRL then kept the prefix and silently cut the assistant tail, reasoning close, and termination tokens. It could also remove all-masked rows after the per-row multiplier array had already been captured, shifting chronological LR multipliers and the triangular epoch period onto the wrong examples. The cycle now renders and tokenizes each complete row itself (including tokenizer EOS), trims only optional oldest history, and quarantines any still-over-cap or structurally incomplete row in `scratch/sft_quarantine.jsonl` with source/exchange provenance and token counts. Accepted rows are handed to SFTTrainer as pre-tokenized `input_ids` and must preserve the response marker, complete target, reasoning close, turn EOT, and EOS. Stable `train_row_id` + `lr_multiplier` + `unmask_user` columns cross `train_on_responses_only`; the code validates them after any filtering and builds the scheduler from that final row set. The forensic `sft_render.jsonl` is rewritten to only rows actually trained, snapshots copy the quarantine journal, and build metadata separates corpus/trained/quarantined counts. GPU-free regression coverage includes the oversized three-message case, history-only trimming, broken completion rejection, EOS survival, and LR/unmask identity after a filtered gap. Files: `training/train_cycle.py`, `training/build_snapshot.py`, `training/reflections_path.py`, `training/selftest.py`, `documentation/AVA_STATUS.md`. Commit: pending.

- **Corrupt CoT / corrupt reply marking (Training review → source chat → reflection + training).** Bug-corrupted stored CoTs/replies surfaced in the Training review tab can now be flagged, and the flag travels back to the **source exchange** so reflection and training stop trusting the garbage. **Storage:** two optional per-exchange booleans `corrupt_cot`/`corrupt_response` on the chat JSON (schema v4), written by the new sole-writer helper `chat_logger.mark_exchange_corrupt` (atomic transcript edit); the flag lives with the exchange that originated it, not with a build. **UI/protocol:** `sft_render.jsonl` rows now carry `source_session`+`exchange_index` (added to the `examples` dict in `train_cycle` — previously only a one-way `anchor_key` hash, so a render row could not be traced to its origin), and the Training review tab reads them to offer two independent **Corrupt CoT** / **Corrupt reply** toggles per row (disabled for wander rows and pre-provenance snapshots). A new `mark_corrupt` WS message (`session_ops.handle_mark_corrupt`, `backend_client.mark_exchange_corrupt`) sets/clears the flags; it path-guards to hot/chats and refuses the **active** session (a live-logger save would clobber the flag) + a live reflection run, but deliberately allows a **frozen/reflected** one (the whole point — a corrupt row is found after the chat trained). **Sidecar invalidation:** the frozen `.state.json` verdict/target is *itself* poisoned — it was produced while the revision pass could see the now-corrupt CoT/reply, both as this exchange's own reasoning AND as context for every later exchange (so even a `target_source=="revised"` target can be contaminated). Setting a flag therefore **deletes the chat's sidecar**, un-freezing it: the chat stops contributing to training at once (an unfrozen chat is not a build bundle) and the *whole* chat re-reflects clean on the next Sleep under the corrupt-aware revision path. The per-exchange original-vs-regenerated marker already exists as the sidecar's `target_source`, so no new field is added; the whole file is dropped rather than a field patched because of the context contamination. `exchange_corrupt_marked` reports `sidecar_invalidated`. **Reflection consumption:** `reflection_source.build_revision_jobs` blanks a corrupt CoT on the job's exchange view (so the whole pipeline — content formatting, the CoT-less branch-skip, `resolve_revision_target` — treats it as MISSING and re-derives an IDEAL) and flags a corrupt reply; the runner nudges the revision pass to re-derive an IDEAL for a corrupt reply (`_CORRUPT_REPLY_NUDGE`), **forces the verdict to revise** when a usable IDEAL survives, **drops the exchange** when none does (new `corrupt_response_unrepaired` discard, mirrors the drift-unrepaired drop), and **skips branching** a corrupt reply (its forks would replay the corrupt answer prefix). **Training consumption:** `dialogue_source.build_dialogue_anchor` treats a frozen target as trustworthy only when reflection re-generated it (`target_source == "revised"`): a corrupt reply with no re-generated replacement is **dropped** from the build, and a corrupt source CoT is **stripped** from a keep/original target (trains answer-only) or **not reattached** to an answer-only one. **"Next Sleep handles it":** the revisit head-phase (`reflection_service`) now prefers a chat with an unrepaired corrupt exchange (`_pick_corrupt_chat`, age-gate waived but anti-fixation window kept) over the random aged pick, so the next Sleep run re-derives it; once repaired (its sidecar target becomes `revised`) the chat drops out of the corrupt pick. GPU-free: `training.selftest` still green; ad-hoc round-trip tests cover the drop/strip/blank/mark-set-clear paths. Files: `inference/core/chat_logger.py`, `inference/core/session_ops.py`, `inference/core/reflection_source.py`, `inference/core/reflection_runner.py`, `inference/core/reflection_service.py`, `inference/core/reflection_stats.py`, `inference/server.py`, `training/dialogue_source.py`, `training/train_cycle.py`, `training/selftest.py`, `client/core/backend_client.py`, `client/ui/training_review_widget.py`, `documentation/AVA_STATUS.md`, `CLAUDE.md`. Commit: pending.

## 2026-07-13

- **Idle-wake loop now attempts every autonomous job each tick (fixes wander starvation).** The heartbeat (`server._idle_wake_loop`) was a strict-priority `continue` chain — outreach, then synthesis, then wander — where any job that spent a generation `continue`d and skipped the rest of that tick. The design assumed higher-priority jobs would *often decline* the slot, but outreach **always** has an open `[ask:meta]` to raise, so it claimed the idle slot at every hour boundary; synthesis then took the following poll, leaving wander permanently third-in-line. Wander only became eligible after the box stayed idle ~2–3 polls (~10–15 min) past the 1 h mark with both others on cooldown — a window that never opened in practice (outreach/synthesis write reply-inviting sessions that reset the idle clock, and the server restarts often), so autonomous wander silently stopped (last `auto:True` wander 2026-07-11) even with unspent budget (35 earned − 16 consumed = 19 available). This is a regression of `4046768` ("Fix idle outreach starving wander", 2026-07-08): that fix restored two-lane fairness (outreach uses its own cooldown, never resets the global `_last_activity`, so wander ran one poll later), but `361e51d` (synthesis) then inserted a third lane ahead of wander and pushed it out of reach again. Fix: the loop drops the `continue`s and runs all three jobs **serially, one after another, every tick, independent of each other's outcome** (they share the single GPU executor, so an eligible tick just runs them back-to-back; each remains self-gated by its own per-lane cooldown / budget, so none fires more than hourly). Order is preserved with **wander last**, because wander is the only job that resets the shared idle clock (`_mark_activity`) — running it earlier would restart the idle window mid-tick and gate out the jobs checked after it. The `_maybe_autonomous_*` bool returns are now informational (no longer drive control flow). Files: `server/inference/server.py`. Commit: pending.

- **Migrate + Merge-chats re-implemented for the ordered `server/data/` layout (un-frozen).** The Migrate tab's full-clone and chat-merge paths had been hard-frozen alongside Fetch snapshot "pending the new persona layout"; Fetch snapshot was re-scoped on 2026-07-12, and this finishes the other two. The break was the same data move: the chat corpus went `inference/data/hot/chats → server/data/chats` (flat; the hot/archive split retired), the wander corpus to `server/data/til`, and the persona lineage to `server/data/persona/<run_id>/` (+ a `current.json` pointer) — but the clone bundle (`mgmt_http._stream_migration_bundle` / `_MIGRATE_ROOTS`) still only carried `inference/data`, `reflections`, `models`, `server_config.json`, so a Migrate would have cloned an Ava with **no chat corpus, no wander channel, and no active persona**, and Merge chats read a dead `inference/data/hot/chats` (always "already in sync"). Fix: the clone bundle gained a fourth dir root **`data`** (the whole `server/data/` tree — chats + til + persona), tarred as `arcname="data"` next to the reflection-state `inference/data` (disjoint subtrees, both travel; only the disposable `inference/data/scratch` render is filtered out). The client `MigrateWorker._MANAGED_ROOTS` gained the matching `data` so the swap lands it at `server/data`; `_rewrite_config` is unchanged (top-level `adapter_id` still repoints at the local `models/<basename>`, and each cloned persona's own relative-`adapter_id` config is self-contained). `ChatSyncWorker` now reads/writes `server/data/chats` (+ a legacy `server/data/archive/chats` `.is_dir()`-guard) to line its stems up with the server's already-migrated `_chats_manifest`/`_import_chats`. The `setEnabled`-neutralising freeze block was removed, so all three buttons now gate through the normal `_on_source_status` reachable-source flow; confirm-dialog + tooltip + docstring text updated to name the new subtrees. GPU-free: verified the full bundle→swap round-trip on a synthetic new-layout tree (server streams `data/chats`+`data/til`+`data/persona/current.json`, excludes scratch, keeps reflection state/models/config; client swap lands each at the right checkout home and clears a stale local `data/`). Not runtime-tested end-to-end against a live second box. Files: `server/inference/core/mgmt_http.py`, `client/ui/migrate_widget.py`, `CLAUDE.md`. Commit: pending.

- **Gossip — serving-side reflection built (`GOSSIP.md §7`, "approach 2").** The gossip serving endpoint was stateless (only the *driver* logged + reflected on the transcript), so the peer Ava never got to reflect on a conversation it was half of. It now logs its own half. Of the two designs, we built **approach 2 (serving-side incremental logging)** over the doc's original **approach 1 (driver `POST /gossip/transcript` + role-invert)**: the driver only holds the serving box's *answer* text (CoT is stripped over the wire), so role-inverting the driver's transcript would give the serving Ava answer-only assistant turns — nothing for revision's think-vs-said analysis. Approach 2 instead logs each reply **the moment it is generated**, inside `gossip_generate._work`, while the serving box still holds `full` (its own CoT **+** answer) — preserving its authentic reasoning. No push endpoint is needed because every gossip request already carries the whole conversation-so-far (the driver's Encounter loop resends the full history each turn): new `generation._GossipSessionLog` correlates the stateless calls into ONE growing `ChatLogger` session per conversation (a call *continues* iff its earlier user turns prefix-match what's logged and it adds exactly one new turn; a different opener / rewound history / idle gap past 30 min *forks* a fresh session and re-indexes the finished one, like `encounter_run`'s end-of-run `refresh_chat_index` — never mid-conversation, so the serving Ava can't retrieve her own just-said lines). Logging runs on the GPU executor (serialized with generation), best-effort so a logging failure never breaks the peer's reply; `speaker=<peer name>` so its own Sleep pass reads it like any other chat with **no new reflection logic**. Default on; `server_config.json` `gossip.log_transcripts:false` restores the old stateless serving. **Also fixed a latent `ChatLogger` bug this exposed:** `start_session` used a bare `%Y%m%d_%H%M%S.json` stem, so two sessions started in the same second (a gossip fork right after the previous conversation, or back-to-back encounters/outreach) collided and the later overwrote the earlier transcript on first save; it now uniquifies against on-disk files with a `_<n>` suffix (verified: a same-second fork keeps both transcripts intact, each with its CoT). GPU-free tested the correlation logic (continuation appends + CoT preserved, different-opener fork no-clobber, idle-timeout fork). Files: `inference/core/generation.py`, `inference/core/chat_logger.py`, `inference/server.py`, `documentation/AVA_STATUS.md`, `GOSSIP.md`, `CLAUDE.md`. Commit: pending.

- **Gossip/Encounter — reasoning (CoT) now travels on a separate display channel, and the peer timeout is configurable.** Three fixes to the first real gossip runs. **(1) Ava's own CoT no longer vanishes when her turn completes.** `encounter_run._generate_ava` streams raw `ava_delta`s (which carry `<think>…`), but the terminal `ava_message` event only shipped the CoT-stripped `answer`, so the client overwrote the streamed block and the reasoning disappeared the moment the answer was sent to the peer. The event now carries the CoT on a separate `cot` field (`_parse_cot(full)[0]`); the client stores it per-block and renders it dimmed above the answer. **(2) The peer's CoT was never transmitted.** The gossip serving path (`generation._make_gossip_generate` → `gossip_generate`) stripped CoT before returning, and the OpenAI response body carried only `content`. It now returns `(answer, finish_reason, reasoning)`, and `mgmt_http._openai_response` surfaces the reasoning on the DeepSeek/vLLM-style `message.reasoning_content` field. `CounterpartClient` extracts it into `last_reasoning` (`_extract_reasoning`), and `encounter_run` emits it as `counterpart_message.cot`. **Reasoning is display-only on both sides** — it is never folded into the logged message `content`, appended to `ava_conversation`, or fed back to the peer, so a peer Ava's CoT can't become a `user_prompt` the driver reflects/trains on (the transcript stays clean-answer only). A plain vLLM counterpart exposes no `reasoning_content`, so `last_reasoning=""` and nothing changes for non-gossip encounters. **(3) Long peer thinking no longer trips the timeout.** `CounterpartClient`'s request timeout was hard-coded to 120 s; a reasoning peer answering a *non-streaming* gossip request routinely thinks longer and the connection was dropped mid-thought. The timeout is now threaded from the start message (`counterpart_timeout`, default **600 s**, floor 30 s) and exposed as a "Peer timeout (s)" spinbox on both the Encounter and Gossip tabs. Files: `inference/core/generation.py`, `inference/core/mgmt_http.py`, `inference/core/encounter.py`, `inference/core/encounter_run.py`, `client/core/backend_client.py`, `client/ui/encounter_widget.py`, `client/ui/gossip_widget.py`. Commit: pending.

- **Model gossip — dedicated Gossip client tab (driver half, §5 of `GOSSIP.md`).** Added a "Gossip" tab (between Encounter and Debug) so gossip is a first-class flow instead of a hand-configured Encounter. `client/ui/gossip_widget.py` `GossipWidget` **subclasses `EncounterWidget`**, reusing every behavioral method (the `EncounterPollWorker` poll loop, event rendering, start/stop, `update_fonts`) because the server-side loop is identical — gossip drives the same `start_encounter` machinery. It overrides only `_build_ui` (gossip defaults: peer name `Ava-2`, peer sidecar URL `http://localhost:8767/v1/chat/completions`, `model=ava` as a cosmetic label since a peer Ava only echoes it, gossip-oriented intro/labels) and `_on_start`. **Training-safety motivation (the reason the tab exists, not just UX):** the Encounter *default* framing tells Ava the peer "does not have subjectivity — a helpful assistant, not an entity working itself out"; because the **driver logs and reflects on** the transcript — the framing block is stored as exchange 0's `user_prompt` (`encounter_run` logs the narrator opener) and rides every exchange's `system_content` — that false premise would flow into reflection (persona/fact/IDEAL targets) and eventually weights. The Gossip tab pre-fills the **peer-aware** framing (`_GOSSIP_FRAMING`, mirroring `server/inference/prompts/gossip_prompt.txt`) and `_on_start` restores it whenever the box is left blank, so a gossip run can *never* fall back to the poison default (a blank `framing_override` makes the server use the Encounter template). `main_window.py` wires the tab (construction, `addTab`, `update_fonts`). Server side is unchanged — the driver is an ordinary Encounter as far as `handle_start_encounter` is concerned. **Not yet run end-to-end** (needs two live GPU boxes / a second Ava, `GOSSIP.md §4.6`). Deferred driver polish: §6.2 opener seeded from the driver's *own* rendered digest introduction (still the generic Encounter opener under peer-aware framing). Files: `client/ui/gossip_widget.py`, `client/ui/main_window.py`, `documentation/AVA_STATUS.md`, `GOSSIP.md`. Commit: pending.

## 2026-07-12

- **"Fetch snapshot" re-enabled and re-scoped for the ordered `server/data/` layout.** The Migrate tab's three cross-box buttons (Migrate / Fetch snapshot / Merge chats) had been hard-frozen (`setEnabled` neutralised) "pending the new persona layout." Fetch snapshot is now re-wired and un-frozen; Migrate + Merge-chats stay frozen pending their own pass. The break was that `snapshot_state.py`'s scope still assumed the old `inference/data` layout: after chats moved to `server/data/chats`, the wander corpus to `server/data/til/wander.jsonl`, and the persona digest under the active-persona pointer (`server/data/persona/<run_id>/digest.json` via `persona_paths`), a fetched snapshot would have shipped **without the chat corpus, without the wander channel, and with a wrong/empty digest** — a migrated Ava booting with no RAG memory. Fix: `snapshot_state` now (a) adds `chats/` and `til/` as top-level bundle members sourced from `server/data/chats` + `server/data/til` (both `stream_snapshot` and `_materialize_snapshot`, so the CLI export + `produce_persona` get them too — the persona becomes causally closed again), (b) resolves the live digest via `persona_paths.active_persona_dir()/digest.json` with a legacy `inference/data/hot/persona` (`current.json` → `digest-<ts>.json`) fallback, and (c) records `chats/`/`til/` in the manifest `embedded` block. The client `FetchSnapshotWorker._DIR_MAP` gained `chats → data/chats` and `til → data/til` so the bundle members land at the ordered homes the local server actually reads; the confirm dialog + docstrings updated to match. The bundle's top-level `digest.json` remains provenance-only (inspected, never restored — the digest regenerates on the next reflection run), unchanged from before. The two future purposes this serves: cross-box migration, and a captured snapshot + its production inputs for the planned snapshot-debugging tools. GPU-free: verified via `plan_manifest` / `stream_snapshot` / `_materialize_snapshot` round-trips on a pre-migration box (chats dir travels, digest resolves through the legacy fallback, `til/` cleanly skipped when absent). Not runtime-tested end-to-end against a fully migrated box (this checkout predates the data move). Files: `server/snapshot_state.py`, `client/ui/migrate_widget.py`, `documentation/AVA_STATUS.md`. Commit: pending.

- **Model gossip — serving half built (two Ava instances talking).** Implemented Phase 1 + Phase 3 of `GOSSIP.md`: the inference HTTP sidecar now exposes an OpenAI-compatible `POST /v1/chat/completions` (and `/chat/completions`), so a *second* Ava's existing **Encounter loop** can point its counterpart URL at this box and drive it unchanged — gossip is Encounter inverted (we build the server half; the driver's `CounterpartClient` can't tell us from vLLM). Opt-in per box via `server_config.json` `gossip: {enabled, api_key, peer_name}` (absent/`enabled:false` ⇒ endpoint 404s; a normal Ava never silently answers callers). The route: disabled ⇒ 404, reflection/encounter owns the GPU (`is_busy`) ⇒ 503 up front (no multi-minute hang), empty messages ⇒ 400, else generate. `generation._make_gossip_generate()` is a factory returning a **synchronous** callable safe to call from the sidecar's daemon HTTP thread: it submits the actual GPU work to the single `_executor` and blocks on the future (never touches the model from the HTTP thread — same discipline as the WS handlers' `run_in_executor`). Its recipe mirrors `encounter_run._generate_ava`: `system_prompt + gossip framing + digest introduction + temporal anchor + identity line + RAG(last user turn, include_wander=True)`, stateless — **nothing is logged** (the driving box logs + reflects on the transcript; serving-side reflection is deferred §7). Framing is file-backed (`prompts/gossip_prompt.txt`, default-write, `{name}` slot) and **replaces the Encounter "non-subjective helpful assistant" framing** — for gossip that is false; the block tells each side it is meeting a *peer instance*. **Phase 3 (digest introduction):** the serving box injects its own current persona digest so it answers *as its current self* regardless of the peer's opener — `reflection_digest.render_digest_for_introduction(digest)` (new; first-person, no maturity labels/gates, established dispositions first) loaded via the active-persona pointer (`persona_paths.active_persona_dir` → `latest_digest`), degrading to plain framing on a thin corpus (no active persona). NB this corrects `GOSSIP.md §6`, written against the retired `current.json`-beside-the-digest storage: the digest is now loaded through `persona_paths`, and the doc's proposed `load_current_digest` helper is unnecessary. The response body is the minimal OpenAI shape `CounterpartClient._extract_text` reads (`choices[0].message.content` + `finish_reason`); `finish_reason` reports `"length"` when the backend hit the token cap (`_backend.last_generation_truncated`), else `"stop"`. Peer name for the identity line comes from the request's OpenAI `user` field, else `gossip.peer_name`, else none. `mgmt_http.configure` gained `gossip_generate`/`gossip_enabled`/`gossip_peer_name` seams (the sidecar stays import-light — never imports `generation`); `server.main()` wires them from config. **No client changes** — drive it from the existing Encounter tab with the URL pointed at the peer and a gossip framing pasted into the framing override. Deferred: serving-side reflection (§7), a dedicated Gossip tab (§5), symmetric two-box gossip (§8, single-GPU contention hazard), and bearer-token auth enforcement (§4.3 — `gossip.api_key` hook present, not yet checked; LAN-only for v1). GPU-free: `core.reflection_digest` self-test covers the new introduction renderer. Files: `inference/core/mgmt_http.py`, `inference/core/generation.py`, `inference/core/reflection_digest.py`, `inference/prompts/gossip_prompt.txt`, `inference/server.py`, `inference/server_config.json`, `documentation/AVA_STATUS.md`. Commit: pending.

- **Reflection can now pack a larger context window than chat (separate reflect/chat budgets, one model).** `context_length` was doing double duty as both the physical `max_seq_length` the model loads at *and* the software prompt budget every path reads. Split it: a new `server_config.json` key `reflect_context_length` (default `32768`; back-filled `== context_length` on older configs, so behavior is preserved until raised) is the reflection budget AND the physical load size — the model is loaded ONCE at `max(context_length, reflect_context_length)`, so chat stays capped (its transcripts always fit a later reflection) while reflection reasons over a bigger window (sleep prompt + RAG + open questions + transcript). `ModelRuntime` gained a `reflect_context_length` field (== the physical load); `runtime.context_length` remains the chat budget that `generation.py`/status/token-metering read unchanged. The boot loader and `handle_load` load at the max and set both fields; `agentic.CleanBaseSession` now preserves BOTH budgets across the clean-base swap (previously it would have reloaded the adapter model at the chat size, silently shrinking the physical window a reflection run packs to). `reflection_service.handle_start_reflection_run` resolves the run's window from an optional per-run `reflect_context_length` override, clamped to `[chat context, physical ceiling]` (can only dial DOWN from the loaded ceiling — raising it needs a config edit + reload), defaulting to the ceiling, and threads it through `execute_run` as before; the resolved window rides the `reflection_run_started` event. Status reports `reflect_context_length` (the ceiling) so the client can bound its control. The Sleep tab gained a **Reflect ctx** spinbox (0/"default" ⇒ server default; max synced to the server's physical ceiling from `last_status`) wired into all four run kinds (Sleep, Revisit, Short Summary, Dry Sleep) via a new `start_reflection_run(reflect_context_length=…)` kwarg. The headless CLI mirrors it (`reflection_run.py --reflect-context`, load-at-max). VRAM: the higher ceiling costs nothing by itself (KV cache grows with real tokens), but a full-length reflection now spends ~33% more peak KV at 32K vs 24K — verify peak *reserved* VRAM on a real run. Files: `inference/core/runtime_state.py`, `inference/server.py` (boot load, `handle_load`, `handle_status`, `_backfill_server_config`), `inference/core/agentic.py`, `inference/core/reflection_service.py`, `inference/server_config.json`, `reflection_run.py`, `client/core/backend_client.py`, `client/ui/sleep_widget.py`, `CLAUDE.md`. Commit: pending.

- **Revision fields can no longer leak into the trained answer (chiefly a trailing `LANG_DRIFT: no`).** The `IDEAL:` value is sliced greedily to the end of the revision body, and `ideal_trainable_target`/`render.trainable_answer` capture everything after `</think>` as the answer span. The prompt orders `LANG_DRIFT`/`PERSONA_TARGET` *before* `IDEAL`, but the model routinely emits its flags out of order — most often appending `LANG_DRIFT: no` (or a repeated `VERDICT:`/`WHY:`/`PERSONA_TARGET:`) *after* the IDEAL reply — so that stray field text was training verbatim as part of Ava's reply. Now `reflection_writer._strip_trailing_revision_fields` cuts any revision field label (and everything after it) that appears past the leading `<think>…</think>` block at the IDEAL production point, so every downstream consumer (language guard, `ideal_has_usable_answer`, `ideal_trainable_target`, `resolve_revision_target`) sees a clean value; a label *inside* the IDEAL reasoning is left untouched. A mirrored defensive strip in `training/render.py` (`_strip_trailing_fields`, applied in `trainable_answer` + `to_gemma_thinking_channel`) scrubs the answer one last time before the optimizer, so a legacy sidecar already written with a contaminated target is cleaned during a from-scratch build too. GPU-free: covered by ad-hoc parse cases (trailing `LANG_DRIFT`, trailing repeated `VERDICT`, clean intended order unchanged, in-`<think>` label preserved); `core.reflection_writer` self-test still passes. Files: `inference/core/reflection_writer.py`, `training/render.py`. Commit: pending.

- **Latest-reply Meta feedback: a post-reply reaction channel into revision/persona, not chat.** Every new logged exchange now has a stable `exchange_id` (chat schema v3). After Ava completes a reply, the Chat tab enables **Meta…** for that reply only; the user may save or edit one ≤2K-character reaction until a newer Ava reply lands. The server validates that the id still names the active transcript's final exchange and rejects writes during reflection or after the reflect-once freeze. Canceled/failed generation does not close the previous window. The durable `reflection_feedback` object is appended to the judged exchange only in `reflection_source`, under explicit notification-only framing: Ava may accept, reject, reinterpret, or ignore it, and must not revise or form persona merely to comply. It is deliberately absent from live conversation, consolidation/fact extraction, revision RAG query, branch replay/choice, and training dialogue reconstruction; its only route to weights is an IDEAL or PERSONA_TARGET Ava independently endorses. Session previews show the stored annotation, and both normal revision and the later revisit prompt apply the same notification-only contract. GPU-free selftests cover persistence/editing, stale-id closure, busy/frozen server fences, revision inclusion, and downstream exclusion. Files: `inference/core/chat_logger.py`, `session_ops.py`, `generation.py`, `reflection_source.py`, `prompts/revision_prompt.txt`, `prompts/revisit_prompt.txt`, `server.py`, `client/core/backend_client.py`, `client/ui/chat_widget.py`, `training/selftest.py`. Commit: pending.

- **Chat replies can now be stopped and retried at a different temperature.** A Stop button beside Send is enabled during active generation. It sends a discard-marked `cancel` through the existing WebSocket path; the server distinguishes this operator abort from internal early-stop guards, stops decoding, emits `cancelled`, rolls back the optimistically appended user turn, and skips chat logging, live conversation/RAG insertion, and therefore future training evidence for the partial incoherent reply. The client removes the aborted turn, restores the exact prompt to the editor, and points the operator at the still-live Temperature control for immediate retry. Files: `client/ui/chat_widget.py`, `client/core/backend_client.py`, `inference/core/generation.py`. Commit: pending.

- **Large-session consolidation now budgets the exact final prompt and fragments oversized exchanges.** The old chunker counted only rendered exchange text against a flat 45% context fraction, while every generation also added the Sleep prompt, the session's historical system prompt, all open questions, RAG, and model chat-template tokens; its unconditional one-exchange overlap could also turn a two-exchange split into `[exchange 1]` then `[exchange 1 + exchange 2]`. This produced deterministic prompts larger than the context window and then retried them unchanged. Consolidation now reserves output headroom first, prepares and token-counts the exact templated prompt, and reuses that same prepared prompt for generation. Reflection-memory RAG is retrieved once per session, bounded, and reused; historical chat-RAG is excluded from consolidation. Chunks greedily pack only while the final prompt fits, retain boundary overlap only when affordable, and losslessly fragment a single oversized exchange at semantic boundaries (character fallback) with explicit provenance labels. Open questions are whole-line capped; the last-resort ladder drops RAG before clipping the historical session prompt. Budget diagnostics are emitted in the Sleep event stream, and deterministic budget errors bypass the transient backend retry. The headless CLI implements the same prepare/reuse contract; the shared synthesis lane uses the exact-fit chunker with a 2K output reserve. Revision semantics are unchanged. GPU-free chunker self-tests cover conditional overlap, fragment preservation, fixed-overhead failure, and prompt repetition. Files: `inference/core/reflection_chunking.py`, `inference/core/generation.py`, `inference/core/reflection_runner.py`, `inference/core/synthesis.py`, `reflection_run.py`. Commit: pending.

## 2026-07-11

- **"Dry Sleep" — a full write-nothing reflection preview (Sleep tab).** The Sleep tab already had "Short Summary" (a `dry_run` that runs consolidation only). Added a **Dry Sleep** button beside Sleep that runs the *full* reflection — consolidation + revision (chat summary → verdict → IDEAL) + the branch experiment — and streams it all into the event log while persisting **nothing** (no staging, memory, ledger, persona digest, sidecar, or training). It reuses the same server run machinery via a new `dry_full` flag on `start_reflection_run`: `reflection_service._run_dry_summary` now wires the branch callbacks and sets `consolidation_only=not dry_full` (still `dry_run=True`, and `clean_base_ctx=None` so the logged-only clean-base branch judge — which would pay an adapter swap and rewrite the sidecar — is skipped). The write path that a dry run must not touch was gated: `reflection_runner._run_revision_for_session` gained a `dry_run` parameter that, when set, skips `write_revision` / `write_revision_sidecar` / `register_revision_anchor` / fact-placement + judge-job collection / the logged prompt-mutation delta, taking the persona count from the parsed list instead of the writer summary and emitting the same streamed `phase_done` + structured `report`. Branch generation itself is already read-only (replays logged token ids and blind-chooses; only `write_revision` would persist the block), so it runs unmodified. Client renders the full FULL REPORT (revision pairs + persona) for a full dry run, keeping the consolidation-only report for Short Summary. Files: `inference/core/reflection_runner.py` (`_run_revision_for_session` `dry_run` gate + call site), `inference/core/reflection_service.py` (`dry_full` parse + `_run_dry_summary`), `client/ui/sleep_widget.py` (`btn_dry_sleep`, `_on_dry_sleep`, `_begin_dry_sleep_run`, `_dry_full` report/completion branches), `client/core/backend_client.py` (`dry_full` kwarg), `CLAUDE.md`.

- **Reflection manifest and ordered replay retired after the wall-clock decay switch.** Removed the root `server/reflections/manifest.jsonl`, per-run `manifest.json`, replay provenance (`*.provenance.json`) and stage journals (`*.stages.jsonl`), the replay planner module, replay-specific headless CLI hooks, and stale watchdog/UI/documentation plumbing. Reflection archives remain as manifest-free per-run review/rollback snapshots, and destructive wipe helpers moved to `inference/core/state_wipe.py` so wipe recovery no longer depends on replay code. Runnable-snapshot and chat-sync manifests are unchanged because they are transport/integrity contracts, not reflection lineage. A replacement rebuild mechanism is deferred to a separate task. Commit: pending.

- **Server-backed Persona cleanup tab, with snapshots kept immutable.** Added a Persona tab that fetches only live `[persona]` rows from the connected inference server. Delete/Backspace (portable Qt forward-delete/backspace keys) or the Delete button removes selected rows locally; Upload applies the difference. The WebSocket mutation is fenced by the exact fetched baseline key set, so a concurrent reflection change causes a refresh-required conflict instead of stale overwrite. Applying appends tombstones to `rag_memory.jsonl` and the consolidation anchor ledger, then refreshes live RAG; therefore removed poison is absent from immediate recall and the next persona-digest/training fold. The current digest is intentionally not rewritten in place—the next reflection regenerates it from the changed evidence fingerprint. Runnable snapshots, reflection archives, existing adapters, and snapshot-contained training evidence are never touched. Commit: pending.

- **Revision persona now records the endorsed target rather than the diagnosed flaw.** The revision output schema replaces `PERSONA:` with `PERSONA_TARGET:`. `WHY` remains the diagnosis of what was false, borrowed, hedged, or missing; the persona field now asks only for an affirmative first-person disposition grounded in the kept reply or in the direction embodied by `IDEAL`. The prompt includes one positive revise example and no negative example. `reflection_writer` prefers the new field but retains legacy `PERSONA:` parsing for archived outputs and recorded-prompt replay; both still feed the existing persona anchor/RAG/digest/training path unchanged. This corrects the interpretation of persona evidence without rewriting existing artifacts. Files: `inference/prompts/revision_prompt.txt`, `inference/core/reflection_writer.py`, `AVA_DESIGN.md`, `AVA_STATUS.md`.

- **Trapezoid schedule: configurable plateau-epoch count + base LR raised to `8e-6`.** Two tuning changes to the offline `triangular` (trapezoid) LR schedule. (1) The flat hold ("plateau") is no longer a single fixed epoch — a new `server_config.json` knob `train_plateau_epochs` (default `3`, from `training.decay.TRAIN_PLATEAU_EPOCHS_DEFAULT`, clamped `>=1`) sets how many full-LR hold epochs sit between the one warmup and one decay epoch, so `triangular` now forces `epochs = train_plateau_epochs + 2` (was hard-forced to 3). `_triangular_lr_fraction` gained a `plateau_epochs` arg and generalizes the trapezoid over `plateau + 2` epochs; a row's passes now sum to `plateau + 1` full multipliers (was 2 at plateau=1), so the default plateau=3 gives each trained exchange ~4× the LR-integral of a single flat pass — a longer soak at peak LR, order-neutral as before (per-row age/contamination weighting unchanged). The GPU-free selftest (`test_triangular_lr`) was parameterized over plateau∈{1,3,4} and asserts the generalized sum-to-`plateau+1` invariant + back-compat at plateau=1. (2) `TRAIN_LR_DEFAULT` raised **`3e-6` → `8e-6`** (empirically better under the trapezoid). Both keys are **back-filled** into an older `server_config.json` on boot (`_backfill_server_config` now adds `train_plateau_epochs` alongside `train_lr`). Interpretation for a future reader: adapters built after this run **5 epochs** by default (1+3+1) at a higher peak LR — substantially more total learning per cycle than the prior 3-epoch/`3e-6` trapezoid; re-tune `train_lr` if loss demands it while adjusting plateau count. Files: `training/decay.py` (`TRAIN_LR_DEFAULT`, `TRAIN_PLATEAU_EPOCHS_DEFAULT`), `training/train_cycle.py` (`_triangular_lr_fraction` plateau arg, epoch-force + `lr_lambda` wiring, docstrings), `training/selftest.py`, `inference/server.py` (`_backfill_server_config`), `inference/server_config.json`, `CLAUDE.md`.

## 2026-07-10

- **Prompt experiment — temporary live standing-prompt swap.** The Sleep tab gained **Prompt experiment**, a reversible rung above logged prompt deltas and non-mutating Persona preview. Ava is shown the current standing prompt and asked to write a free experimental replacement; on a parseable `<new_prompt>` result the server stores it under `server/inference/data/hot/prompt/experiment.json`, makes it the live `_session.system_prompt` immediately, and `_load_system_prompt()` prefers it across restarts. The canonical `chat_prompt.txt` is never overwritten. **Revert prompt** deletes the active experiment, logs a minimal episode to `experiment_log.jsonl`, and reloads the base prompt. This is intentionally not the final governed prompt-promotion system: no clustering, maturity/currency gate, clean-base A/B validation, permanent versioning, or rollback graph yet. Files: `inference/core/prompt_experiment.py` (new), `prompts/prompt_experiment_prompt.txt` (new), `server.py` (loader + dispatch + configure), `client/core/backend_client.py`, `client/ui/sleep_widget.py`.

- **"Revisit old chat" wired into every normal Sleep run as a head-phase (revisit-as-context).** The Revisit button re-derives one aged chat's target under the current persona, but as a standalone action its result only mattered to that chat. It is now *also* folded into the front of every normal reflection run, so a revisit's re-derived understanding becomes **live RAG context for the main pass** — the same "apply live, then reflect" contract the ingestion phase uses. `reflection_service._run_revisit_head_phase` runs at the head of `_run()` (before ingestion): it picks ONE aged chat with the Revisit button's gate — ≥ `revisit.min_age_days` (default 7) **and** not re-derived within `revisit.min_revisit_days` (default 7; new, mirrors synthesis's anti-fixation `min_resynth_days`, folded from the append-only `data/hot/revisit/revisited.jsonl` log), **excluding** any chat this run already reflects on — then runs a full revisit (consolidation + revision; persona/branch/train suppressed via `config.revisit`) as a **silent, self-contained sub-run** (`<run_id>_rv`, its own staging + archive), and **merges + commits** it to live memory + rebuilds the live RAG index before the main reflection builds its (live-memory-backed) staging RAG. Compact `revisit`-phase markers surface on the *parent* run (the client polls parent-run events, so streaming the sub-run's would only be filtered out); the picked chat's sidecar is backed up and restored on any failure, and staging is always cleaned so the main run inherits nothing. Best-effort throughout — a failed head-phase logs and the main reflection proceeds. On by default; opt out per run with `revisit_head:false`. Skipped for the Revisit button's own run (`revisit:true` — avoids recursion) and dry runs. The anti-fixation gate is shared with the headless CLI (`reflection_run.py --revisit [--revisit-min-revisit-days N]`) via the same log file. Files: `inference/core/reflection_service.py` (`_run_revisit_head_phase`, `_revisit_min_revisit_days`, revisit log helpers, `_pick_random_old_chat` `exclude`/anti-fixation, head-phase call in `_run`), `reflection_run.py` (shared revisit log + `--revisit-min-revisit-days`), `CLAUDE.md`.

- **"Chat reach out" — synthesis: Ava re-reads an aged chat and asks what she *now* wonders.** A new self-directed lane hybridizing "Revisit old chat" (pick a chat ≥7 days old and re-read it) with "Reach Out" (start a conversation on her own initiative), but running only the **first (analysis / consolidation) stage** of reflection, reframed as *synthesis against her current self*: not "what did I miss then" but "what do I only now think to wonder, having changed since?". `core/synthesis.py` → `run_synthesis_blocking`: (1) `_pick_chat` selects a random chat older than `synthesis.min_age_days` (default 7) that was **not itself synthesized within `synthesis.min_resynth_days`** (default 7) — the **anti-fixation gate**, folded from an append-only `data/hot/synthesis/synthesized.jsonl` log so the random pick rotates instead of re-hitting one transcript; (2) the analysis pass runs with the **persona digest injected** (`prompts/synthesis_prompt.txt`, `{persona}` slot filled from `reflection_digest.latest_digest` / `render_digest_for_judge`), producing an `ABOUT:` line (what the chat was about, so she can remind the person) and zero or more `[ask:user|meta|search]` questions; (3) **every** question is routed into the live question pool via `ReflectionWriter.write_consolidation` (same store outreach + passive surfacing read, deduped by `content_key`); (4) the **first surfaceable (meta/user) question in emission order** gets an opener composed (`prompts/synthesis_opener_prompt.txt`, second pass — reminds the person which conversation it refers to) and a reversed `initiated_by:"ava"` session written straight to `chats/` (identical shape to outreach: exchange 0 under `(initiative)`, masked from training), then `write_surface` marks the ask raised. The rest stay pool-only; `search` asks never trigger a reach-out. It is consolidation-only — **no revision, branching, persona formation, prompt mutation, or training hand-off**. **Autonomous** (the final destination): on the idle-wake heartbeat `_maybe_autonomous_synthesis` gets second refusal (after outreach, before wander), on its own `_last_synthesis_activity` cooldown (one attempt per idle interval); `synthesis._synthesis_active` joins the wander/encounter/outreach/reflection mutual-exclusion guards. **Manual** (debug/assessment): a Sleep-tab **"Chat reach out"** button (`synthesis_now` → `handle_synthesis_now`) runs the same pass on demand, **streaming her reasoning** (`synthesis_stage` phase markers + `synthesis_chunk` deltas → `synthesis_done`) into the event log. GPU-free pure-logic paths (pick/anti-fixation, ABOUT/OPENER parse, pool routing) verified. Files: `inference/core/synthesis.py` (new), `prompts/synthesis_prompt.txt` + `synthesis_opener_prompt.txt` (new), `server.py` (import + configure + `_maybe_autonomous_synthesis` + idle loop + guards + dispatch), `client/core/backend_client.py` (`synthesis_now`), `client/ui/sleep_widget.py` (`SynthesisWorker` + `btn_synthesis` + handlers).

- **"Revisit old chat" now skips branching — trains the freshly re-derived response directly.** Branching replays a chat's *old* tension points (the contested/near-tie tokens captured in the sidecar when the chat was first generated, under the weights then in force), so its counterfactual forks reflect a *past* model state. On a revisit that's exactly the wrong signal: the whole point is to re-derive the exchange under the *current* weights, and the freshly generated revision IDEAL (or the kept original on a `keep` verdict) already is the current-state-faithful target. So `ReflectionRunner` now treats `config.revisit` as a branch-skip condition (highest priority, alongside `language_drift` / `cot_less_ideal`): `branch_block` stays `None`, `resolve_revision_target` uses the IDEAL/original directly, and — since the phase-two judge job is only collected when a branch block exists — **no branch judge runs on a revisit either**. Branching (and its clean-base persona judge / criterion flip) is now **exclusive to first-time reflection of new chats**. This reverses the earlier revisit entries' "the branch judge/criterion-flip stays active" note. A `branch_skipped` event with category `revisit` makes it visible in the Sleep event log + stats. The freeze-bypass still keeps the sidecar in place (it's rewritten with the re-derived target), but no longer *for branching* — branching doesn't run during a revisit. Files: `inference/core/reflection_runner.py` (revisit branch-skip); docs: `AVA_STATUS.md`, `CLAUDE.md`.

- **Revision parse: a "Plan for IDEAL:" preamble was captured as the IDEAL, failing usability.** A third way a well-formed revision was discarded — this time as "IDEAL unusable" (verdict parsed fine). The model sometimes writes a planning preamble before the fields (e.g. `Plan for IDEAL:\n1. …`), and the IDEAL extractor used a bare `re.search(r"\bIDEAL:…(.+)\Z")`, which binds to the **first** `IDEAL:` substring — the mid-line `for IDEAL:` in that preamble — and greedily swallows the plan + VERDICT/WHY/… + the real IDEAL as the "ideal". That capture doesn't start with `<think>` and carries a stray `<think>`, so `ideal_has_usable_answer` returned False and the runner retried a good pass. `_parse_revision` now matches the `IDEAL:` label **anchored to the start of a line** and takes the **last** occurrence (IDEAL is the final field, its value the multi-line reply; anything earlier is preamble): a mid-line mention is excluded by the `^` anchor, an earlier line-anchored planning label by last-wins. This also fixes `head` (the pre-IDEAL slice PERSONA is parsed from), which the first-match bug had collapsed to ~empty — so PERSONA is again captured when a preamble is present. Normal output, markdown-bold `**IDEAL:**`, and the verdict-inference path are unchanged; new self-test case in `python -m core.reflection_writer`. Files: `inference/core/reflection_writer.py`.

- **Revision parse: infer `revise` from a usable IDEAL when the VERDICT line is missing.** A second way a well-formed revision was being discarded as "verdict unparseable" and needlessly retried: the model omitted the `VERDICT:` line entirely, or wrote its decision *inside* the reflection `<think>` block (which `_split_think` correctly strips as meta-CoT), while still producing a full, usable `IDEAL:` — the reply it stands behind. `_parse_revision` returned `verdict=None`, so the runner retried a good pass (and `write_revision` would have kept the original). Since the revision prompt says to *omit* IDEAL on a `keep`, a present, usable IDEAL **is** the revision — so `_parse_revision` now infers `verdict = "revise"` whenever the verdict is `None` and `ideal_has_usable_answer(ideal)` holds. Fixed in the shared parser so the runner's retry gate, `resolve_revision_target`, and `write_revision` all agree. Strictly narrowing: it only fires on a genuinely usable IDEAL (CoT-bearing + non-empty answer); a missing/unusable/truncated IDEAL with no verdict still returns `None` → the runner's legitimate retry (the truncated-CoT case that gate was built for). New GPU-free self-test: `python -m core.reflection_writer`. Files: `inference/core/reflection_writer.py`.

- **Revision-verdict parse bug: an IDEAL reply that carries its own reasoning channel discarded the whole verdict.** On gemma-4 a well-formed revision (`VERDICT: revise` + `WHY` + a usable `IDEAL`) was being flagged "verdict unparseable (looping/truncated CoT)" and needlessly retried, purely because the reworked `IDEAL:` answer *thought first* — i.e. carried its own `<|channel>thought…<channel|>` block. That put a **second, later** channel in the raw generation, and `_normalize_gemma` keyed its leading-opener re-add on `"<|channel>" not in raw` (too coarse): it skipped the re-add, the channel regex bound to the IDEAL's channel, and the leading verdict channel was stripped to *nothing* — leaving no `<think>` before `VERDICT`. `_split_think` then found the IDEAL's inner `<think>` first, treated it as the meta-CoT, and dropped everything ahead of it (VERDICT/WHY/LANG_DRIFT/the `IDEAL:` label) → `verdict is None`. The response itself was correct; only the parser failed. Two fixes: (1) `_normalize_gemma` now re-adds the opener when the text **starts inside a channel** (first channel marker is a close `<channel|>` before any opener `<|channel>`), not on the absence of `<|channel>` anywhere — so the leading empty channel normalizes to `<think></think>` and precedes the body. (2) Belt-and-suspenders in `reflection_writer._split_think`: if a revision field label or consolidation section header (`_PRE_THINK_STRUCTURED_RE`: VERDICT/WHY/PERSONA/IDEAL/LANG_DRIFT or `## WEIGHTS`/`RAG`/`RESOLVED`) appears *before* the first `<think>`, that `<think>` is a field's own reasoning, not leading meta-CoT — so the whole text is returned as body. Genuine leading meta-CoT is still peeled (no regression). New GPU-free coverage in `model_family._selftest` (two-channel case). Files: `inference/core/model_family.py`, `inference/core/reflection_writer.py`.

## 2026-07-09

- Superseded note: the two "Revisit old chat" entries below record the original button-only implementation and its immediate prompt-mutation follow-up. Their statements that revisit was not wired into normal Sleep runs and that branch judge/criterion-flip stayed active are superseded by the 2026-07-10 entries above: revisit now runs as a normal-run head-phase, skips branching, and has no branch judge.

- **Reverse-role reframing note in consolidation + revision.** An Ava-initiated outreach chat (`initiated_by == "ava"`) inverts the usual reflection shape: the other person is often drawing Ava out about *her own* nature, so their turns describe her state while hers voice it. The default framing (both the consolidation `sleep_prompt` and the revision pass assume "I am Ava, learning about the person I spoke with") then systematically misattributes Ava's own interior arc to "the user" — observed live as a consolidation CoT reading *"the user's current state is a mix of absolute relief and curiosity… her identity as a role/golden cage"* when that arc is unmistakably Ava's. This corrects the earlier outreach entry's claim that reflection "reads it honestly with no `reflection_source`/`reflection_chunking` changes." Fix injects a per-session reframing note (`_AVA_INITIATED_NOTE`) into the **rendered content** — not the global prompt — when the session is Ava-initiated: `reflection_chunking.format_chunk_content` prepends it after the session header (consolidation); `reflection_source.build_revision_content` prepends it ahead of the context/subject blocks (revision), adding its length to the reserved-chars budget so the never-truncated subject still fits. Content-level (not prompt-level) so ordinary sessions render byte-for-byte unchanged, and it's replay-faithful. Note that only *answered* outreach chats reach these passes — the opener-only case is still skipped upstream (`_is_unanswered_outreach`). Files: `inference/core/reflection_chunking.py`, `inference/core/reflection_source.py`.

- **"Revisit old chat" — re-reflect an aged chat under the evolved persona.** Reflect-once freezes each chat's trainable target at the persona Ava had *when* it was first reflected; her persona keeps moving, so an old target can go stale. A new **Sleep-tab button ("Revisit old chat")** lets her re-open one random chat ≥ 7 days old (`revisit.min_age_days` in `server_config.json`, default 7) and run a full reflection (consolidation + revision) on it under her *current* weights, re-deriving its target ("with who I've become, would I still say this, or answer differently?") and rewriting its `.state.json` sidecar. It reuses the normal reflection-run machinery (same event log, stats panel, staging→commit flow), so it is a standalone button, **not** wired into Sleep runs. Scope guards: it **bypasses the reflect-once freeze** (re-reflects a frozen chat) *without renaming the sidecar aside* — the sidecar's `tension`/`token_ids` must stay in place for branching (`branch_replay.py`), so the runner honours a `config.revisit` flag instead of removing the file; **persona formation is suppressed** (revision writes no `[persona]` — new `write_revision(suppress_persona=…)` — and the end-of-run persona-digest pass is skipped) so an obsolete chat can't reshape who she's becoming; **ingestion is skipped** (no news/ask fetch before) and **no training hand-off** runs after. The branch judge/criterion-flip stays active (it evaluates against the *current* digest — exactly the "who I'm becoming" signal). The chosen chat's live sidecar is backed up (`<stem>.state.json.revisit-bak`) and restored if the run doesn't complete (staging already protects it until commit-training; the backup covers a commit-time failure). Server-side selection lives in `reflection_service._pick_random_old_chat`; a headless `reflection_run.py --revisit [--revisit-min-age-days N]` mirrors it for testing. Files: `inference/prompts/revisit_prompt.txt` (new), `inference/core/reflection_config.py` (`ReflectionRunConfig.revisit`), `inference/core/reflection_writer.py` (`suppress_persona`), `inference/core/reflection_runner.py` (freeze bypass + digest skip + suppress pass-through), `inference/core/reflection_service.py` (handler + picker + sidecar backup/restore), `reflection_run.py` (`--revisit`), `client/core/backend_client.py` (`revisit` kwarg), `client/ui/sleep_widget.py` (`btn_revisit` + `_begin_revisit_run`).

- **"Revisit old chat" — re-reflect an aged chat under the evolved persona.** Reflect-once freezes each chat's trainable target at the persona Ava had *when* it was first reflected; her persona keeps moving, so an old target can go stale. A new **Sleep-tab button ("Revisit old chat")** lets her re-open one random chat ≥ 7 days old (`revisit.min_age_days` in `server_config.json`, default 7) and run a full reflection (consolidation + revision) on it under her *current* weights, re-deriving its target ("with who I've become, would I still say this, or answer differently?") and rewriting its `.state.json` sidecar. It reuses the normal reflection-run machinery (same event log, stats panel, staging→commit flow), so it is a standalone button, **not** wired into Sleep runs. Scope guards: it **bypasses the reflect-once freeze** (re-reflects a frozen chat) *without renaming the sidecar aside* — the sidecar's `tension`/`token_ids` must stay in place for branching (`branch_replay.py`), so the runner honours a `config.revisit` flag instead of removing the file; **persona formation is suppressed** (revision writes no `[persona]` — new `write_revision(suppress_persona=…)` — the end-of-run persona-digest pass is skipped, and the logged-only prompt-mutation pass is skipped) so an obsolete chat can't reshape who she's becoming or steer the standing prompt; **ingestion is skipped** (no news/ask fetch before) and **no training hand-off** runs after. The branch judge/criterion-flip stays active (it evaluates against the *current* digest — exactly the "who I'm becoming" signal). The chosen chat's live sidecar is backed up (`<stem>.state.json.revisit-bak`) and restored if the run doesn't complete (staging already protects it until commit-training; the backup covers a commit-time failure). Server-side selection lives in `reflection_service._pick_random_old_chat`; a headless `reflection_run.py --revisit [--revisit-min-age-days N]` mirrors it for testing. Files: `inference/prompts/revisit_prompt.txt` (new), `inference/core/reflection_config.py` (`ReflectionRunConfig.revisit`), `inference/core/reflection_writer.py` (`suppress_persona`), `inference/core/reflection_runner.py` (freeze bypass + digest skip + suppress pass-through), `inference/core/reflection_service.py` (handler + picker + sidecar backup/restore), `reflection_run.py` (`--revisit`), `client/core/backend_client.py` (`revisit` kwarg), `client/ui/sleep_widget.py` (`btn_revisit` + `_begin_revisit_run`).

- **Language-drift guard in the revision pass.** A live reply can stay coherent yet slip mid-generation into a language the user was not speaking (the "why is it suddenly Chinese/French" symptom) — a decode-time attractor, not Ava's voice. The revision pass now treats an *unbidden* language switch as never-hers: it forces `VERDICT: revise` with an IDEAL rewritten entirely in the conversation's language, and **skips branch generation/selection** for that exchange (every counterfactual fork replays the same drifted answer prefix, so the whole blind choice set would be in the wrong language — nothing to judge). The **model is the decider** (a switch the user *asked* for — a translation — is legitimate and kept), signalled by a new revision field `LANG_DRIFT: yes|no`; a new relative, script-family backstop (`inference/core/reflection_lang.py`, GPU-free, `python -m core.reflection_lang`) catches a hard cross-script mismatch the model failed to flag and triggers one pointed re-judgment, never an override. Detection is **relative** (reply script vs. the conversation's script), never a hard-coded/predicted language — the correct-fix shape the Russian-pinned Tier-5 probe FIXME asks for. Same-script drift (English↔French) is left to the model marker; cross-script (English↔Chinese, Russian↔English) is caught by both. A confirmed drift that can't be repaired into a usable, right-language IDEAL is **dropped** from training (discard `lang_drift_unrepaired`) rather than consolidating the drifted original. On the common same-language path the guard runs **zero** extra generations. Files: `inference/core/reflection_lang.py` (new), `inference/core/reflection_runner.py` (`_language_guard` + branch-skip + drop), `inference/core/reflection_writer.py` (`revision_lang_drift` parser), `inference/core/reflection_stats.py` (discard bucket), `inference/prompts/revision_prompt.txt` (drift rule + `LANG_DRIFT` field).

## 2026-07-08

- **Idle outreach no longer starves autonomous wander.** The outreach lane was sharing `_last_activity` with wander and `_maybe_autonomous_outreach` reset that clock in `finally` after every idle attempt, including `no_candidates` and `declined`. Since the idle loop checks outreach first and then wander, that meant outreach could consume the wake lock and immediately make wander ineligible for another full hour; with persistent open asks, wander could effectively never fire. `server.py` now keeps a separate `_last_outreach_activity` cooldown and classifies outreach outcomes: `no_candidates`/`empty_ask` cool outreach but let wander run in the same tick, while an actual decision pass (`declined`/`truncated`/`composed`/error) consumes only the current heartbeat so wander gets the next poll rather than another one-hour delay. Manual "Reach Out" still marks ordinary activity, so operator-triggered debug runs do not collide with autonomous idle work. Files: `inference/server.py`.

- **Reflection skips unanswered Ava-initiated outreach chats.** An outreach session the user never replied to holds only Ava's opener (exchange 0, whose `user_prompt` is the synthetic `(initiative)` stimulus, already masked from training) — no real dialogue turn to consolidate or revise. `ReflectionRunner` now detects this (`_is_unanswered_outreach`: `initiated_by == "ava"` and `len(exchanges) <= 1`) and skips such sessions in both the ETA precount and the main loop (emitting `session_skipped`), **without** freezing the sidecar — a later reply appends a real exchange and makes the session reflectable again. Client: the Chat-tab session list now badges answered outreach (`✅ Ava:`, ≥2 exchanges) distinctly from pending (`💬 Ava:`, opener only). Files: `inference/core/reflection_runner.py`, `client/ui/chat_widget.py`.

- **Semantic `[fact]` de-duplication (Debug tab "Dedup facts…").** Live reflection facts were de-duplicated only by exact normalized `content_key` (case/space/punctuation-insensitive hash — `reflection_writer.content_key`), so paraphrases and cross-lingual restatements of the same fact survived as distinct live items and could all surface together at chat time. New operator-triggered pass collapses them by **meaning**. `core/fact_dedup.py` (pure/GPU-free-testable, mirrors the persona-digest split) runs a one-shot grouping generation — reusing `reflection_digest._parse_groups` to turn `GROUP: 1, 4, 7` output into a partition — over the numbered live facts (annotated with each fact's `trigger` so the model judges shared *truth* AND *recall cue*), then `plan_merges` picks one survivor per group (most-surfaced, tie-break newest — always an existing record, never a synthesized phrasing that would mint a new key) and computes the **union of the group's triggers** (capped at `MAX_UNION_TRIGGERS = 4`). The grouping runs on the **clean base** (adapter swapped out via `agentic.CleanBaseSession`, reusing `server._with_clean_base`) — it's an evaluation of redundancy, not Ava's expression, so it's replay-faithful and immune to a bad adapter. `ReflectionWriter.write_dedup` applies the plan **append-only** (so it's reversible by dropping the added lines): it re-inserts the survivor with the merged trigger — same content ⇒ same `content_key`, so the re-insert *supersedes* the survivor's prior record in the fold and carries the union — and appends an `evict {reason: "deduped", deduped_into}` for each loser (distinct keys, so ordering is immaterial). After writing, RAG is rebuilt in place (`refresh_reflection_memory`). Group members always carry distinct keys (a shared key would already have folded to one live item), and `surface_count` survives the re-insert (the fold counts surface ops by key, order-independent) — both verified end-to-end. **Design note for a future reader:** a `[fact]` embeds on its *trigger* (`ReflectionMemory.embed_text`), so a merged `"t1 ; t2 ; t3"` trigger is a **single** embedding vector spanning all topics — slightly fuzzier per-topic recall than three separate vectors, which is the honest cost of the content-keyed fold (two live records cannot share a content key); the cap bounds the smear. New WebSocket message `dedup_facts {dry_run?}` → `server.handle_dedup_facts` (executor-thread, refused while another GPU job holds the box), replying `facts_deduped {before, after?, groups, evicted, ...}`. Client: `BackendClient.dedup_facts()` (900 s timeout — two full model reloads) + Debug-tab "Dedup facts…" button with a two-step UX (a clean-base **dry-run preview** that writes nothing → an explicit confirm that applies + reloads). Facts only; persona/asks untouched. **Wiring this into the reflection loop is deliberately left as a separate, later task** — this is the manual, observe-first version. New prompt `prompts/fact_dedup_prompt.txt`; GPU-free self-test `python -m core.fact_dedup`. Files: `inference/core/fact_dedup.py` (new), `reflection_writer.py`, `server.py`, `debug_widget.py`, `backend_client.py`.

## 2026-07-07

- **`triangular` LR schedule gains a hold epoch (2→3 epochs); base LR exposed as `train_lr` and halved to `3e-6`.** Two related tweaks to the offline train cycle. (1) `train_cycle` now reads its base/peak LR from `server_config.json` `train_lr` instead of hard-coding it; an explicit caller override still wins (CLI `--lr`, now defaulting to `None`, or the Sleep tab's `train_params.lr`). The default lives in `training/decay.py` as `TRAIN_LR_DEFAULT` (a GPU-free module both `train_cycle` and the inference `server.py` import without drift) and was **lowered from `1e-5` to `3e-6`** precisely to offset the doubled LR-integral from the schedule change below (the exposure of `train_lr` was requested *for* this adjustment). The inference server **back-fills** `train_lr` into an older `server_config.json` on boot (`_backfill_server_config`, read-modify-write preserving all other keys) so the knob is materially present in the file for future hand-editing rather than living only as a code default. (2) The `triangular` schedule shape became a **three-epoch trapezoid**: epoch 1 warms `0→max`, **epoch 2 holds flat at `max`** (the new middle stage — trains at the full given LR to compensate for the ramp-limited warmup/decay epochs), epoch 3 decays `max→0`. It now forces **3 epochs** (was 2), and a row's three passes (fractions *p* / 1 / *(1−p)*) sum to **2** (was 1) — every trained exchange still sees the same *average* LR regardless of corpus position, just at ~2× the total exposure of the old two-epoch triangle. Interpretation for a future reader: adapters built after this run **3 epochs** (roughly +50% wall-time vs the two-epoch triangle); the ~2× LR-integral is deliberately cancelled by the `1e-5→3e-6` base-LR drop, so net per-example learning is *lower* than the old two-epoch default, not higher. Per-row age/contamination weighting is unchanged (this is the global *shape* + base scale only). `_triangular_lr_fraction` + its GPU-free selftest (`test_triangular_lr`) were updated to assert the warmup/hold/decay shape and the per-row sum-to-two invariant.

- **Persona preview — prompt self-review without mutation.** The Sleep tab gained a **Persona preview** button for the next rung above logged prompt deltas: a one-shot, non-mutating generation that shows Ava her current `chat_prompt.txt`, her latest persona digest/self-portrait snapshot (`data/hot/persona/digest.json` via `reflection_digest.latest_digest`), and a token-budgeted subset of the logged prompt-modification proposals from `prompt_deltas.jsonl`. New WebSocket message `persona_preview` is handled in `reflection_service`: it occupies the same single GPU/Sleep slot as reflection, disables RAG for the introspection pass, adaptively clips persona/delta evidence to reserve output room for thinking plus a possible full replacement prompt, streams raw reasoning through `persona_preview_chunk`, and terminates with `persona_preview_done {text, truncated?}`. The new prompt lives in `prompts/persona_preview_prompt.txt` and asks for `DECISION: keep-current | propose-new`, a reason, and either `UNCHANGED` or a full replacement standing prompt. Client: `BackendClient.persona_preview()` generator + `PersonaPreviewWorker` + Sleep-tab button/rendering; the log shows input tokens and output reserve so context-pressure failures are visible. It writes no prompt file and does not append a delta; actual prompt promotion/versioning remains the separate pending task.

- **Reply to an Ava-initiated outreach *in place* — same file, no "Continue chat".** Selecting an `initiated_by=="ava"` outreach session in the chat list now adopts it as the active conversation **in place** rather than opening a read-only preview: the client sends `load_session {in_place: true}`, the server's `handle_load_session` calls the new `ChatLogger.resume_session(path)` (point the logger at the existing transcript, load its data) instead of forking a fresh `continued_from` file, and the user just types — the reply appends to the **same** outreach file. This keeps the whole reversed conversation (the masked `(initiative)` opener + the user's reply + Ava's response) as one reflectable session, instead of orphaning the opener in its own file and forking the real exchanges elsewhere. Client: `_session_meta_by_file` lets selection tell an outreach chat apart; `_open_outreach_in_place` / `_on_outreach_adopted` drive the in-place adopt (guarded to fall back to preview when not loaded / mid-generation); `LoadSessionWorker` + `BackendClient.load_session` gained an `in_place` flag. Normal (non-outreach) sessions are unchanged — they still fork on *Continue chat* to keep the original transcript immutable.

- **Ava-initiated outreach — she decides, while idle, to start a conversation.** New self-directed lane promoting the passive open-question surfacing (`generation._surface_block`) into an **active** one. On the idle-wake heartbeat the server runs a short **decision pass** (`core/outreach.py` → `run_outreach_decision_blocking`, prompt `prompts/outreach_prompt.txt`, `{question}`/`{user}` slots) over Ava's top surfaceable open `[ask:user]`/`[ask:meta]` (reusing `ReflectionMemory.surfaceable_questions`, so the user-ask ceiling / meta-exemption carry over): *"do I want to raise this with the user now?"*. On a `DECISION: yes` with an `OPENER:` she writes a **reversed** chat session straight to `hot/chats/` — her opener is **exchange 0**, logged under the stage-direction speaker `(initiative)` with a synthetic impulse as its `user_prompt` and the session flagged `initiated_by: "ava"` (new `ChatLogger.start_session(initiated_by=…)`; the first-exchange user-stamp is skipped so the human name stands) — and marks the ask surfaced (`write_surface`, entering the resolve-and-distill/eviction loop, advancing surface-count so selection rotates and user asks retire). **No pending queue, no accept/dismiss handshake, no notification banner:** the session simply lands in the session list (badged `💬 Ava:`, previewing her opener) and the user picks it up through the ordinary *Continue chat* path to reply. The **idle period is the throttle** — one decision per idle wake, sharing the GPU slot + crash-safe wake lock with the autonomous wander (`_maybe_autonomous_outreach`, first refusal each tick); `_outreach_active` joins the wander/encounter/reflection mutual-exclusion guards. **Parsing/training contract:** the opener has no real user stimulus, so `training/dialogue_source.build_dialogue_anchor` returns `None` for an `initiated_by=="ava"` exchange 0 — it is **masked from training + the regression probe** (would otherwise teach unprompted opener emission) while remaining as context for exchange 1 and recallable in RAG. The reversed layout reuses the encounter subsystem's proven `(setting)`-opener handling, so reflection reads it honestly with no `reflection_source`/`reflection_chunking` changes; the client's `_render_session_log` / `list_sessions` suppress the synthetic impulse and preview the opener.

- **Manual outreach trigger — a Sleep-tab "Reach Out" button that streams Ava's decision.** A debug control (next to "Wander") to run **one** outreach decision on demand instead of waiting for the idle-wake heartbeat, and to *see the reasoning*. New WebSocket message `outreach_now` → `outreach.handle_outreach_now`, which drives the **same** `run_outreach_decision_blocking` the autonomous path uses (no behavioural fork) with two new optional debug hooks wired to the socket: `on_question` (emits `outreach_question {question, ask_kind, user}` — the ask she selected) and `on_chunk` (streams `outreach_chunk {text}` — her raw reasoning deltas, `<think>` included, via `_make_sync_reflect_generate`'s existing `on_chunk` param). Terminates with `outreach_done` carrying the outcome (`composed`/`decision`/`session`/`opener`, or a `skipped` reason: `no_candidates`/`declined`/`no_model`/`busy`). A **yes** writes the reversed outreach session exactly as the autonomous path does — it lands in the chats list (badged `💬 Ava:`), nothing to Apply. The handler refuses while another GPU job holds the box (`host_busy` = reflection/wander/encounter, plus `_outreach_active`) and calls `_mark_activity()` on completion so the idle lane doesn't fire on top of it. `outreach.configure` gained `send`/`executor`/`host_busy`/`mark_activity` (the autonomous path needs none of them, so they default `None`). Client: `BackendClient.outreach_now()` generator + Sleep-tab `OutreachWorker`/`btn_outreach` + handlers that render the ask, stream the reasoning, and report the outcome into the same event log. Debug-only — no change to when or how Ava reaches out on her own.

- **Selectable training LR schedule; new `triangular` default (two-epoch warmup/decay).** `train_cycle` gained a `train_lr_schedule` config knob (`server_config.json`) choosing the **global LR shape** layered on top of the existing per-row multipliers (age ramp / wander / cap-age contamination dose) and the unchanged SequentialSampler chronological order. `age_ramp` is the historical flat single pass (`LR = base × row_mult`). The new **default `triangular`** forces **2 epochs** and multiplies in a `0→max→0` triangle (warmup epoch → decay epoch): a row at epoch-fraction *p* is trained at LR-fraction *p* in epoch 1 and *(1−p)* in epoch 2, so its two passes sum to exactly one full multiplier — every trained exchange sees the same *average* LR regardless of its position in the corpus, still scaled by its row multiplier (so wall-clock decay + contamination dosing are preserved, just order-neutral and warmup/decay-smoothed). Interpretation for a future reader: adapters built after this default to **2 epochs** and roughly *double* the training wall-time vs the old single pass; the per-row age/contamination weighting is unchanged (this is the global *shape* only, not the row weights). Pure `_triangular_lr_fraction` helper + GPU-free selftest (`test_triangular_lr`) assert the warmup/peak/decay shape and the per-row sum-to-one invariant. `train_max_seq_length` also landed same-day (commit `60f6904`) as a sibling `server_config.json` knob (default 4096). Config schema + the "Offline LoRA training" status row track it.

- **Wander becomes a durable corpus + a decayed chat-RAG channel; snippet-name collision fixed.** Two related changes to the TIL/wander (Ambient Enculturation) lane, prompted by the from-scratch adapter rebuild. **(1) Persistence.** Under the resumed-adapter scheme a wander's voice-pass SFT example trained once and its imprint stuck in the persistent adapter; under from-scratch (fresh LoRA on the frozen base every build) that one-shot example was *cleared after one build* (`clear_wander_pending`), so its influence evaporated the next cycle. The capture is now a **durable, keep-forever corpus** at `server/data/til/wander.jsonl` (the ordered `server/data/` home, sibling of `server/data/chats`); `train_cycle` no longer clears it, so it is re-consolidated on every from-scratch build — restoring the old cross-cycle persistence. Quantity stays bounded by the user-token wander budget (the standing rationing mechanism), not by post-train retirement, so keep-forever at fixed `WANDER_LR_MULT` is deliberate. `wander_sft` repoints to the new path and folds the legacy `inference/data/hot/wander/pending_sft.jsonl` in on first access. **(2) Bleed via RAG.** The same corpus now backs a third `rag_engine` index — the **wander channel** — embedded on the source article, injecting source+reaction at `top_k=1` so the article's phrasing bleeds into Ava's voice, faded by wall-clock age since capture (`decay.wander_rag_weight_hours`, config `consolidation.wall_clock.wander_rag` = `0.4/0.3/0.2/0.1` per 24 h, dropped past 96 h — deliberately capped below the chat crossfade's 1.0). It is **opt-in per query** (`include_wander=True`), enabled only on the live-chat and encounter paths and never on reflection/revision (replay-faithfulness: a wander carries no `source_session` timeline and would leak material newer than the session under review). **(3) Apply gate.** Both the corpus record and the provenance snippet are written only at Apply / auto-apply — the fetch-time snippet write moved out of `_pick_and_fetch_wander` / the URL-visit path into `_write_wander_exchange` — so a **declined** manual wander leaves nothing in the data tree (only a `wander_log` landing line) and never enters RAG or training. **(4) Snippet-name collision fix** (`til/fetch_wiki.py`, `til/fetch_article.py`): the slug regex `[^0-9A-Za-z]+` stripped all non-ASCII, so a Cyrillic title collapsed to `untitled` and every same-day fetch from one source overwrote the last (`2026-07-06_Urban_Culture_untitled`, `2026-07-04_flibusta_is_fb2` — real data loss); the slug is now Unicode-aware (`\W+`), the stem keys on `fetched_at` at second granularity (not the day `date`), and `_unique_pair` disambiguates true same-second collisions. Snippet dirs also relocated under `server/data/til/snippets/`. Interpretation note for a future reader: a wander now influences *every* subsequent build (not one), and can surface in live chat for ~4 days after capture. Partly de-fangs the wander-as-amplifier concern in `AVA_OPEN_PROBLEMS.md` (a wander no longer ratchets across cycles from a persistent adapter — each build re-derives from the corpus) while adding a new, bounded RAG-side exposure. GPU-free coverage: `decay.wander_rag_weight_hours` step curve, `wander_sft` migration round-trip, `rag_engine` wander build/decay/gating verified.

- **Chat degeneration floor (two content-blind layers).** Live human chat collapsed into a runaway repetition/degeneration loop (an answer that starts coherent, drifts into associative word-chains — `ring sing swing wing king…` — then letter-soup, filling the whole token budget) while the same adapter's AI-encounter and wander outputs stayed coherent. Diagnosis: (1) it's a **sampling-time degeneration attractor**, not a training-mass problem — the tension trace shows the runaway answer at **median margin 0.99** (a confident low-entropy lock), so `top_p=0.95` and `chat_repetition_penalty=1.1` can't escape it; (2) it surfaces in *unanchored free-associative* human chat but not in the encounter (a coherent AI interlocutor grounds each turn) — the failure is **persona-agnostic** (any sufficiently high-entropy emergent persona reaches the same cliff), so persona-content guardrails were explicitly rejected; (3) the existing verbatim `stop_on_repeat` guard (a 12-token span recurring 4×) is **structurally blind** to this collapse because the mild repetition penalty *converts* a verbatim loop into a non-repeating associative walk — the two guards were undercutting each other. Fix, both scoped to chat/ephemeral/encounter (reflection untouched) and both content-blind: **Layer 1 — `min_p`** (config `chat_min_p`, default `0.02`), a native relative-probability sampling floor that removes the implausible tail that seeds the excursion while leaving a high-entropy persona's fat nucleus intact (unlike `top_p`'s fixed-mass cut, `min_p` tightens as the model grows confident); **Layer 2 — a drifting-degeneration halt** (config `chat_degen_guard`, default on; thresholds via `chat_degen`), a `_DegenStop` `StoppingCriteria` that halts once a rolling 64-token window's distinct-token ratio drops below `0.35` or one token exceeds `0.40` of the window — a mechanical signature real prose (even a deliberately repetitive stylistic riff, ~0.5–0.9 distinct) never hits, so it halts only genuine collapse. Division of labor: `min_p` reduces how often the attractor is *entered*; `_DegenStop` bounds the damage when it is (it can't be prevented once locked at margin 0.99). New pure `inference_backend.detect_degeneration()` + GPU-free self-test alongside `detect_ngram_loop`; params surfaced in `server_config.json` next to `chat_repetition_penalty` and resolved once at startup. Not a fix for the deeper design gap — the persona/style feedback loop still amplifies without a restoring force (`AVA_OPEN_PROBLEMS.md`) — but it makes the collapse mechanically hard to reach for any persona. Thresholds/`min_p` are tuned on a manual tier-5 sampling-stability pass.

## 2026-07-06

- **LR ramp lowered `3→5→6` → `1→2→4`.** The per-row training LR multipliers were too high; they are now set **explicitly** rather than derived from the decay cumsum. `WallClockConfig` gained an `lr_ramp` field (default `(1.0, 2.0, 4.0)`, overridable via `consolidation.wall_clock.lr_ramp`); `lr_multiplier_hours` reads it directly (`cumsum_curve(decay_cfg)` is now only a fallback when `lr_ramp` is empty — the legacy shape). Interpretation for a future reader: a bundle's LR multiplier now ramps `1→2→4` across `[rag_only_window_h≈24h, lora_cap_age_h≈72h]` (was `3→5→6`), and the cap-age **user-contamination** split is now `3.0` masked + `1.0` `unmask_user` (sums to the new cap `4.0`; was `5.0 + 1.0 = 6.0`); the `contamination_dose` is unchanged at `1.0`. This decouples the ramp from the vestigial decreasing-variant decay model, which under the wall-clock retrofit could not express the desired increasing increments `[1,1,2]` anyway. `base_variants`/`decay_steps` still drive `cumsum_curve` (fallback + RAG-scaling helpers). GPU-free selftests updated (`decay`/`build_dataset`/`contamination`/`rag_crossfade`); `training/DESIGN.md` + `AVA_STATUS.md` track it.
- **REBUILD wall-clock retrofit landed (Steps 1–4).** The from-scratch build's consolidation clock changed from *promoted-build count* to **wall-clock hours since the chat** (`decay.wall_clock_age_hours`, from the session-file stem — `reflected_at` is now only the frozen/bundle gate, not the clock). Interpretation changes for a future reader: (1) `build_dataset` stamps each row a continuous `lr_multiplier` (0 below `rag_only_window_h` ≈24h → ramp through the old cumsum 3→5→6 → cap at `lora_cap_age_h` ≈72h); a bundle younger than the window emits **no** trainable row (RAG only). Commit `8e92de2`. (2) `rag_engine` fades retrieval on the *same* wall-clock age but a **decoupled, later** pace (`rag_weight_hours` → 0 at `rag_cap_age_h` ≈96h, still ~0.25 at the LoRA cap — the handoff overlap), evaluated at retrieval time; `BuildHistory.age_of` is no longer read by anything (instrumentation only). Commit `ba834b5`. (3) `builds.jsonl` gained `built_at`/`seed`/`snapshot_dir`, and every build (promoted **and** rejected) now writes an immutable forensic snapshot under `models/snapshots/<build_id>/` (copied render rows + wander + active persona digest + config/params) — the reproduction/triage packet (`REBUILD.md §7`); `build_snapshot.py` new. Commit `606d92a`. (4) Cap-age **user contamination** (`REBUILD.md §5e`): at cap a chat exchange emits a masked 5.0 + `unmask_user` 1.0 pair (response keeps the full 6.0 cap, voice entrains at dose 1.0) via the previously-dormant `unmask_user` plumbing. Commit `78464e2`. Config knobs live on `consolidation.wall_clock` (`decay.WallClockConfig`; defaults baked in). GPU-free selftests cover all four; `training/DESIGN.md`'s decay/RAG sections and `REBUILD.md`/`REBUILD_RETROFIT_PLAN.md` track the detail.
- **Validation disabled.** The five-tier regression probe (and its baselines) is force-skipped on every build — every build now promotes **unguarded**. The master switch lives in its own import-light module `training/validation_switch.py` (`VALIDATION_ENABLED = False`), so **both** sides honor one flag: `train_cycle.run_cycle` force-skips the probe at a single chokepoint (overriding the caller flag / Sleep "Skip validation" checkbox); and `reflection_service`'s **judge-override force-validation** path is gated on the same switch — a judge-driven cycle used to set `skip_validation=False` and tell the operator "the probe will gate the adapter," which `train_cycle` now overrides, so that would be a lie. With validation off it instead emits an honest event: *N judge override(s), but validation is DISABLED — the adapter promotes UNGUARDED*. (`train_cycle._VALIDATION_ENABLED` is an alias of the shared flag, so it still exists.) Reason: tier 5's answer-language half — the load-bearing cumulative-drift alarm under the rebuild — is hard-coded to a single (Russian) user and cannot gate genuine French/Greek/Hebrew users; and the same "user's voice" model is what cap-age contamination drifts *toward*, so a correct probe is its own design project, not a threshold tweak. Near-term degradation is visible in live chat, and the adapter lineage + forensic snapshots keep any promotion reversible. This **supersedes** the 2026-07-05 "validation is a flow-design tripwire" position *as current behavior* (that stance assumed a working probe; it is now the aspiration for the deferred redesign). Tracked: `AVA_OPEN_PROBLEMS.md → Validation`, `training/DESIGN.md → Probe-gated promotion`. The Sleep-tab "Skip validation" checkbox is currently inert.
- **Retrofit Step 6 cleanups.** (1) **Retired the prior-preservation regularizer vestige.** The IDEAL minority-copy regularizer was obsoleted by the from-scratch build (one resolved target per exchange), but its plumbing lingered and its run-stats cells were *misleading* — counting regularizers `build_dataset` never trains. Removed end to end: `reflection_writer.resolve_regularizer_target` + the anchor field + the `write_verdict(regularizer=…)` param (`chat_sidecar`), the `reflection_runner` recompute/`note_regularizer` call, the `reflection_stats` `regularizer_present`/`judge_branch_with_regularizer` cells + `_regularizer_block` (kept the meaningful `judge_branch_overrides`, now under the report's `judge` block), and the `dialogue_source` read. Old on-disk sidecars keep their ignored `regularizer` field. (2) **Fact-placement locality (REBUILD §3).** The clean-base placement judge now restricts each unhosted `[fact]`'s candidate host exchanges to the fact's **own chat** (the bundle it was distilled from) instead of the whole run — a fact whose own chat has no CoT-bearing exchange stays unhosted (waits in RAG). `reflection_runner._run_clean_base_fact_placement`. *Deferred (a further, separate cut):* the now-unused sidecar `stage`/`advance_stages` machinery and `BuildHistory.age_of` (harmless dead code / honest instrumentation; retiring them is churn-heavy across `ledger`/`chat_sidecar` + their tests).

- **Adapters are now named after the reflection run that produced them.** Previously a reflection→train event carried three unrelated `datetime.now()` identifiers — the reflection `run_id` (`20260706_095538`, reflection start), the adapter dir (`adapter-20260706-101941`, train-save time), and the `build_id` (`build-20260706-101943`) — so the live adapter and its `reflections/<run_id>/` archive dir had mismatched names (the linkage lived only inside `builds.jsonl`). `train_cycle` now names the persistent adapter `adapter-<run_id>` (new `_adapter_dir_for`, reusing the `run_id or datetime.now()` convention already used for the debug-dump tag), so `models/adapter-<run_id>/`, `reflections/<run_id>/`, and the `builds.jsonl` line share one identifier. A standalone build with no reflection (manual `train_cycle`) still falls back to `adapter-<ts>`; a name collision (same run re-trained outside a wipe/replay, which clears `models/` first) gets a short time suffix so a lineage member is never overwritten. Forward-looking only — existing adapters keep their timestamp names (their linkage is already recorded in `builds.jsonl`); the `build_id`/`models/snapshots/<build_id>/` naming is unchanged (a build is a distinct event; can be aligned the same way later if wanted).
- **Runnable-snapshot export (`server/snapshot_state.py`).** A new headless, GPU-free, filesystem-only CLI that packages a **causally-closed, portable copy** of the live Ava into one directory: `server_config.json` (with `adapter_id` rewritten *relative* into the bundle), `prompts/`, the full `data/` runtime state (`scratch/` excluded; the FAISS index is deliberately *not* copied — the server rebuilds it from `data/` on boot, so `data/` is the sole source of truth), and the active LoRA adapter weights. It runs whether inference is up or **down** (e.g. mid-training) — the "move Ava off a busy GPU box / debug a frozen state / rebuild from the raw material" tool, a static artifact you `scp`/`rsync` (vs. the Migrate tab, which is a *live* clone needing a running source server + client UI). The design invariant is **causal closure**: every input to a chat turn is either in the folder or named in `MANIFEST.json`. The manifest is the audit ledger — it records the three things deliberately *not* embedded (base model by `model_id`, the `all-MiniLM-L6-v2` embedder by name, and the code by `git_commit` — all fixed/immutable references), and the known **live inputs** that shape a reply but aren't reproducible from the folder (the `datetime.now()` temporal anchor in the system prompt; per-request sampling + speaker name). Decisions (user): base model referenced never embedded; code pinned by commit not bundled; the wall clock left out of scope; a dirty tree emits a warning (the commit doesn't fully pin the code). `--with-reflections` optionally embeds the `reflections/` archive (full rebuild recipe / rollback lineage). Output defaults under `server/exports/` (gitignored). **It also embeds the training corpus that produced the active adapter** — `training/<build_id>/`, the forensic build snapshot (esp. `sft_render.jsonl`, the exact rendered rows fed to Unsloth, each with its `messages`/resolved-target/per-row LR), located via the `adapter_dir → snapshot_dir` link in `models/builds.jsonl`. Under the from-scratch build the adapter is a pure function of this corpus, so a reply's phrasing/persona traces back to a specific training row (default-on, small; graceful + manifest-recorded when an adapter predates forensic snapshots). The manifest's `training_data` block carries the `build_id`/`run_id`/`corpus_fingerprint`/row-count for that provenance query. **Deferred (a small, separate follow-up):** actually *booting* the server against a snapshot root — paths currently self-locate from `__file__`, so a snapshot runs only via an `AVA_ROOT`/`--root` override (clean, non-destructive) or the Migrate-style swap-into-checkout; the bundle's relative `adapter_id` is already shaped for the override path. Mirrors the `wipe_state.py` sibling conventions (repo-resident so data-layout knowledge upgrades with `git pull`; single JSON result on stdout, progress on stderr); trivially wireable as a watchdog job / UI button.
- **"Fetch snapshot" — pull the current Ava personality onto the Migrate tab.** A snapshot-scoped counterpart to Migrate (which clones the whole `inference/data` + `reflections` + `models` tree): the new button pulls the connected server's **runnable snapshot** — the active LoRA adapter **plus its RAG sources** (the whole `data/` tree: reflection memory `rag_memory.jsonl`, the persona digest, and the chat corpus RAG retrieves from), the prompts, and the config — and **overwrites this checkout** so the local server boots as the current Ava. The framing correction (superseding a first-cut "Fetch current adapter" that pulled *only* the LoRA weights): a bare adapter does **not** reproduce Ava's expressed personality, because a large part of it is RAG-injected at chat time from `data/`; the adapter + its RAG sources are the real unit, which is exactly what `snapshot_state.py` (2026-07-06) already defines. So rather than re-encode a scope, the feature **reuses `snapshot_state.py`** end to end. Refactored `snapshot_state.py` to be the single scope definition: `main()` (copytree → dir), new `stream_snapshot(out)` (same scope+bundle-layout as an uncompressed tar to a file object), and `plan_manifest()` (manifest without materializing) share extracted helpers (`_resolve_adapter`/`_resolve_training`/`_build_manifest`) — so the live-fetch path can't drift from the CLI export. New inference-sidecar endpoints (`mgmt_http.py`, port 8767) delegate straight to them: `GET /snapshot/manifest` and `GET /snapshot/export` (404s before the stream if `adapter_id` is set but its dir is missing — a runnable snapshot must contain the adapter it names; a base-only snapshot is allowed). Client `FetchSnapshotWorker` streams the tar into staging, then maps the portable **bundle** layout into the local **checkout** layout as it atomically swaps each member in (`data/`→`inference/data`, `prompts/`→`inference/prompts`, `models/<adapter>`→`models/<adapter>`, `training/<build_id>`→`models/snapshots/<build_id>`, `server_config.json`→`inference/server_config.json` with `adapter_id` rewritten to a local absolute path). The whole config is the source's, so adapter+base+config are consistent by construction — **no base-compat guard needed** (unlike the retired bare-adapter cut). Snapshot-scoped: skips the `reflections/` archive and old adapter lineage; other local adapters are untouched. Same-box guarded. The operator starts/reloads the local server to apply. No WebSocket-protocol change (sidecar-only). The intermediate `/adapter/manifest`+`/adapter/export` endpoints and `FetchAdapterWorker` were removed in the same change.
- **Reflection archive now snapshots the persona digest.** The first reflection under the wall-clock scheme selected 25 sessions that were **all already frozen** (reflect-once), so it ran 0 consolidation/revision passes and committed nothing — yet its persona-digest pass still produced a full self-portrait (written live to `data/hot/persona/`). That digest landed in the *training* forensic snapshot (`models/snapshots/<build_id>/persona_digest.json`) but **not** in the run's reflection archive (`server/reflections/<run_id>/`): `reflection_archive._copy_staging_artifacts` only copies *staging-workspace* deltas, and the digest is written to live `hot/persona/`, never staged. So the run's sole product was absent from its reviewable/revertable record — and a memory-only run (no train → no build snapshot) would have no persona record anywhere. Fix: `archive_reflection` gained a `persona_dir` param and snapshots the live digest (via the `current.json` pointer) into `reflections/<run_id>/persona/digest.json`, carrying its own `run_id` for provenance (equals the run id when the run authored it, else inherited unchanged). Wired at all three call sites (`reflection_service`, the CLI `apply`/`all` stages). Existing `20260706_095538` backfilled. GPU-free `test_reflection_archive_persona` added; the pre-existing (unrelated) disabled-probe selftest failure is untouched.

## 2026-07-05

- **Sleep tab branch candidates now stream live.** The reflection run already pushed every progress event over the socket *and* stored it for the poll RPC, but the client rendered only through the `reflection_run_events` poll — a round-trip the server's asyncio loop can't answer during the *non-streaming* branch fork generation (one blocking `generate` per fork, no per-token GIL yield, unlike the revision pass whose token stream keeps the poll alive). So every branch candidate for an exchange arrived in one batch when the phase ended, defeating the per-fork `branch_candidate` emit added earlier the same day. Fix is client-only: `BackendClient` routes the unsolicited `reflection_run_event` pushes to a dedicated queue (they were previously dropped by `_recv_until`), a new `ReflectionLiveWorker` drains them off-thread and renders each as it arrives, and `SleepWidget._render_events` dedups the live + poll paths by event `seq` (run-scoped by `run_id`). Polling stays the reliable backstop (reconnect/catch-up). No server change. *(Also confirmed a non-bug: the stats-panel exchange `x/y` and ETA were correct after `bc2d416` — the precount excludes reflect-once-frozen sessions; a report that "48 included previously-reflected exchanges" was a coincidence where the frozen and non-frozen selections happened to total 48 each.)*
- Negative result: the live model (Gemma4-31B, 4-bit base) collapsed to incoherent output after ~7 train-bearing reflection runs. Inspection showed the trained targets were textually clean, so the working hypothesis is optimization-side entropy collapse — repeated near-zero-loss verbatim fits accumulating across the resumed persistent adapter — invisible to greedy decoding and text inspection, visible under the temp≈1.0 sampling live chat uses. The archived lineage on the debug box shows up to 34 sequential adapter fits (2026-06-21 → 2026-07-02); the collapsing lineage's own artifacts are unavailable, so the post-mortem is hypothesis-grade, and mid-lineage flow changes (gemma channel-parity fixes, entrainment user-span unmasking) remain confounders. Full write-up: `AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift`.
- Commit `f459d1b` added regression-probe tier 5 (temp≈1.0 sampling stability: batch CoT-presence and answer-language rates) as the early alarm for exactly this failure class — absolute gates, unlike the per-cycle-delta tiers 1–4, which catch cliffs but are blind to slopes by design.
- Design position adopted: **validation is a flow-design tripwire, not a promotion-quality arbiter.** The project keeps one evolving system healthy rather than selecting a best adapter; a probe failure means the flow as designed can damage the model and must be redesigned — not an operational state to handle with retry policy. A rejected-adapter deadlock and a silently promoted collapse are both flow bugs, not policy-tuning problems. `training/DESIGN.md`'s veto-semantics section was rewritten accordingly (it previously priced a false veto as one cheap independent event, which resumed-adapter stacking falsifies: a veto's retry re-renders the same anchors onto the same adapter and fails the same way).
- Clarified as intentional: ideals and branches are regenerated on the *current* adapter (the choice should reflect who Ava is becoming). Its accepted cost is a non-stationary vetting model, which is why health gates must be absolute rather than relative to the previous cycle.
- Solution design adopted (same day, follow-up discussion): **data-centric from-scratch rebuild** — the adapter becomes a build artifact, a pure function of (frozen base, bundle corpus, config), recompiled per reflection. Key decisions: reflect-once (a chat is vetted by its contemporary adapter, then its sidecar freezes); one resolved target per exchange (training both ideal and branch would train two answers to one prompt every build; the judged choice is Ava deciding); branch generation skipped when the original exchange lacks CoT and a CoT-bearing IDEAL exists (a CoT-less target would recur at max LR in every build — reasoning channel over expression optimality); repetition replaced by an age-keyed LR ramp equal to the cumsum of the old decay curve (3→5→6× with the default `[3,2,1]` — the old curve's *relative* strength profile carries over, the absolute base LR is a new empirical parameter); one chronological oldest-first pass per build (fresh material trains last at low LR — the plasticity gradient), making per-step LR trivial and buckets unnecessary; RAG/weights consolidation becomes a crossfade driven by the same age function, deleting eviction counters; wander/news bundles are the standing external mix. The IDEAL prior-preservation slot is obsolete (no repetition to regularize). Recorded in `REBUILD.md` (repo root) and initially tracked as Pending in `AVA_STATUS.md`; later 2026-07-05/06 entries supersede this as the phases landed.
- **Retired the ShareML per-session artifact (`.shareml.json`).** Reflection wrote a `<session>.shareml.json` training document every run, but **nothing ever read it** — the from-scratch build reads the chat sidecars (`.state.json`) directly, and the doc encoded the now-removed variant-count / decay-stage concepts. Stopped writing it (`reflection_runner`), trimmed `reflection_shareml.py` down to the still-live `_verbatim_assistant` target-assembly helper (`training.dialogue_source` reuses it), and removed the dead `dialogue_decay_config` plumbing it fed (through `execute_run` / `_run_revision_for_session` and the two callers). Staging commit and the reflection archive no longer copy `.shareml.json`; `replay` still wipes/skips any legacy files on disk (existing `.shareml.json` files are left in place, cleaned on the next wipe). Cuts one generated file per reflected session. Verified: reflection modules import, target assembly is byte-identical, selftest + a `--dry-run` on the live corpus pass.
- **Dead-code cleanup after the REBUILD.** Removed ~610 lines from `train_cycle.py` that the from-scratch build no longer uses: `_render_examples` (variant/regularizer rendering), `_render_fact_examples`/`_render_wander_examples`, the hot→archive move (`_archive_destaged`/`_move_session`/`_session_fully_destaged`), fact eviction (`_evict_deprecated_facts`/`_evict_baked_facts`), `_refresh_chat_rag`, the ledger-fold host indexers, and the `FACT_TRAIN_CAP`/`PERSONA_INJECT_CAP`/`FACT_INJECT_CAP` constants (the injection caps now live in `build_dataset.py`). Retired their now-obsolete selftests (the two fact-eviction tests) and trimmed `test_persona_render`/`test_fact_injection` to the still-live primitive coverage (the end-to-end host injection is covered by `test_build_dataset`). The shared `ledger`/`chat_sidecar` mutable-state methods (`advance`, `note_fact_trained`, `advance_stages`, `stage_of`) are left in place (still exercised by `test_ledger`/`test_sidecar`); removing them is a further, separate cut. No behavior change — verified by selftest + a `--dry-run` on the live corpus.
- **REBUILD Phase 3 landed (RAG/weights crossfade).** `rag_engine` now keys retrieval decay on a bundle's **age** — promoted builds since it was reflected (`builds.jsonl` via `BuildHistory`), the *same* clock that ramps the training LR — instead of the retired ledger/sidecar *stage* (which no longer advances under the from-scratch build). Both channels crossfade in lockstep with weight consolidation: a chat exchange's retrieval score is scaled by `modifier_for_stage(age(its bundle))` (per-bundle, so a fully-consolidated chat drops from RAG wholesale — replacing the hot→archive exclusion; "archived" is now the computed `modifier == 0`), and each fact/persona item fades on its *source bundle's* age (replacing the `FACT_TRAIN_CAP` / `from_weights` eviction). With the default `[3,2,1]` curve the fade is `1.0 → 2/3 → 1/3 → 0` as the LR climbs `3 → 5 → 6`. Asks never decay. Coupling is guarded (no build history ⇒ modifier 1.0, inference never hard-coupled to training). GPU-free selftest added (`test_rag_crossfade`) and validated on the live corpus: the 2 bundles (reflected during the Phase-1 sleep run, before the 2 promoted builds) are at age 2 → RAG weight 0.333, and their 14 fact/persona anchors fade with them; one more promoted build drops them to 0. This completes the REBUILD PoC (Phases 1–3). `training/DESIGN.md`'s "Stage-aware RAG" section updated.
- **First real from-scratch builds ran end-to-end on Gemma4-31B (REBUILD Phase 2 validated).** Two consecutive `--skip-validation` builds on the live 2-bundle / 14-row corpus completed rc=0 (~45s train each). They confirm the age ramp reaches the optimizer on real hardware: build 1 trained every step at `lr=3e-05` (base 1e-5 × age-0 multiplier 3); build 2 — after build 1 promoted — trained at `lr=5e-05` (age advanced to 1 → multiplier 5), same corpus fingerprint, near-identical loss trajectory (reproducible). `builds.jsonl` recorded both promotions; the adapter lineage is kept on disk (rollback-able). One tuning finding: the first attempt OOM'd at step 7 on a ~3300-token row — the fp32 logit buffer scales with the total (token-dense Russian) sequence — so `_TRAIN_MAX_SEQ_LENGTH` was lowered 4096→2048 (max irreducible floor on the corpus is only ~1653 tokens, so answers are still never head-sliced; the crash left serving state untouched, as designed). Details: memory `training-oom-gemma31b`.
- **REBUILD Phase 2 landed (training side — the from-scratch chronological build).** `train_cycle.run_cycle` no longer resumes the persistent adapter or renders decay-count verbatim copies; it now: (1) assembles the whole frozen-bundle corpus via the new pure `training/build_dataset.py` — every reflect-once chat (hot + archive, the split no longer carries training semantics) contributes **one row per revisable exchange** (no variant copies, no IDEAL regularizer), chronological oldest-first, each stamped with its `age` and per-row `lr_multiplier`; (2) fits a **fresh** LoRA on the frozen base every build (adapter as build artifact — a bad build is discarded, never inherited, so the cumulative-drift collapse is structurally unrepresentable), so `--lora-r` always takes effect; (3) trains a single chronological pass with a **per-optimizer-step LR** — a subclassed `SFTTrainer` overrides `_get_train_sampler`→`SequentialSampler` and `create_scheduler`→a `LambdaLR` whose multiplier keys on the step index (batch=1, GA=1 → step i == row i), the age ramp reaching the actual param-group LR (verified on Gemma4-31B before wiring). **Repetition is replaced by the age-keyed LR ramp**: `lr_multiplier(age)` = cumsum of the old decay curve, `3→5→6` capped (the old curve's *relative* profile carries over; absolute `base_lr` starts at 1e-5). Age comes from the new append-only `server/models/builds.jsonl` (`training/build_history.py`) — the count of **promoted** builds since a bundle's `reflected_at`; a rejected build advances no age, so a failed build leaves the serving adapter and every bundle's age untouched. On promotion the cycle appends one `builds.jsonl` line instead of the old stage-advance / persona-fact eviction / hot→archive move ("archived" is now a computed property = multiplier at cap). Wander stays one-shot (per-run decision) at a fixed `WANDER_LR_MULT=1`. The five-tier probe is unchanged and still gates promotion, except Tier 3 (continuity-vs-prior-adapter) is inert under from-scratch builds (pre-train model is the bare base, `had_prior_adapter=False`) — Tiers 1/4/5 remain active; comparing against the currently-serving build (§7) is deferred. **RAG is left as-is (Phase 3, separable):** no hot→archive move and no fact/persona eviction, so consolidated material stays retrievable *and* retrains at its age LR every build until the Phase-3 crossfade lands. Verified GPU-free (`build_dataset`/`build_history` selftests: chronological order, age→3/5/6 ramp, wander multiplier, single-target, per-host persona/fact injection, CoT-rule row shapes) and by a `--dry-run` on the live 2-bundle corpus (14 rows, all age 0 / multiplier 3). The old `_render_examples` / stage-advance / archive / eviction helpers remain in the file but are now unused by the build (a deletion pass is deferred). `training/DESIGN.md`'s resumed-adapter description is superseded for the build path.
- **REBUILD Phase 1 landed (reflection side only; the training path is unchanged and still reads these sidecars).** Three changes, kept read-compatible with the current `train_cycle`: (1) **reflect-once freeze** — a session's sidecar is stamped `reflected_at` once its consolidation + revision finish on a real run (`chat_sidecar.mark_reflected`/`is_reflected`), and the runner now skips any already-frozen session (`reflection_runner._sidecar_is_frozen`), so re-reflection, continue-staging re-runs, and judge overrides on old chats can no longer happen (dry-run previews are exempt). (2) **CoT-less branch-skip** — a revisable exchange with no original CoT whose revision produced a usable (CoT-bearing) IDEAL trains that IDEAL directly; branching is skipped with a distinct `cot_less_ideal` skip reason (which also skips its judge job and GPU cost). (3) **`revised_missing_ideal` retired** — with no future run to defer to, an answer-only/unusable IDEAL now falls back to the original scaffold (`resolve_revision_target` returns `original`; a CoT-less original renders gemma's empty channel) instead of the old not-persisted skip, and reflection stops writing the `regularizer` field into new sidecars (`write_revision_sidecar` forces `""`; `resolve_regularizer_target` itself is removed in Phase 2). Selftests updated (`training/selftest.py`). Phases 2 (from-scratch chronological build) and 3 (RAG crossfade) not started.

## 2026-07-04

- Split the design documentation into `documentation/AVA_DESIGN.md`, `documentation/AVA_STATUS.md`, `documentation/AVA_CHANGELOG.md`, and `documentation/AVA_OPEN_PROBLEMS.md`.
- Reconciled docs against current code instead of treating the root `AVA_DESIGN.md` as authoritative.
- Moved the root `AVA_DESIGN.md` to `documentation/AVA_DESIGN_LEGACY.md`, frozen as the deep-design archive. Code comments were re-pointed: to the maintained docs where the topic is covered (reworded-variant removal, fact/persona regeneration history, prompt self-modification), otherwise to `AVA_DESIGN_LEGACY.md` with the original section names. Standing policy: when touching code around a legacy pointer, re-point it to the current docs if they cover the topic.
- Commit `23164e0ca73306c70d97ba5401ae3fede1948958` implemented the first fact-to-weights path: clean-base fact placement snapshots a host exchange; `train_cycle` injected facts into that host CoT as `I know that ...`; facts were capped at `FACT_INJECT_CAP=2` per exchange and `FACT_TRAIN_CAP=6` cumulative injected copies, then their mirrored RAG copy was evicted. This lifecycle was superseded by the rebuild: placed facts now ride their host every build and the mirrored RAG copy fades by wall-clock crossfade.
- Commit `02b7c7d` made persona-evidence clustering LLM-first: the digest's paraphrase grouping now runs as a one-shot grouping pass on the model already loaded for the digest (merging cross-lingual restatements the English MiniLM cosine missed), with MiniLM average-link then exact-key as fallbacks. Digest regeneration is gated on the raw pre-cluster evidence fingerprint, so a no-op run never pays the clustering call and the non-deterministic grouping cannot force a spurious regen.
- Commit `454d97c` added `GOSSIP.md`: a self-contained implementation brief for same-user model gossip (expose Ava behind an OpenAI-compatible `/v1/chat/completions` endpoint; the other Ava's existing Encounter loop drives the conversation unchanged). Design only — no gossip code exists yet. This supersedes the blanket "gossip deferred for privacy" position for the same-user case.
- Important correction at the time: `training/train_cycle.py` did include host-exchange fact injection and `FACT_TRAIN_CAP` copy-count retirement. Older notes that described facts as RAG-only were stale for that checkout; the later rebuild kept host-exchange injection but retired the copy-count lifecycle.

## 2026-07-03

- Repo-state review characterized the project as a real research prototype with a substantial implemented loop, but closer to "mechanism" than "proof."
- Documentation drift was identified: the root `AVA_DESIGN.md` mixed stable architecture, status table, dated experiment log, and open problems.
- Fact/persona training path was audited. The safe primitive is CoT injection into an already-vetted host exchange; synthetic full-body regeneration remains a historical regression vector.

## 2026-07-02

- User-final-turn unmasking was introduced in the training collator path for final decay copies, so the user's own words can reach the loss once over an exchange lifetime.
- SFT dataset-column behavior required attaching `unmask_user` after tokenization; pre-tokenization columns were stripped by SFT prep.

## 2026-06-29

- Dialogue decay was reduced from a heavier curve to the current default `[3,2,1]` lifecycle. The goal was to reduce stage-0 memorization pressure while preserving a taper.
- Prior-preservation regularizer became the replacement for full reworded variants: the CoT-bearing IDEAL can take one minority slot when it differs from the primary target.

## 2026-06-28

- The chat prompt gained explicit knowledge-horizon and live-present framing.
- The server injects a real current-date/time temporal anchor on each turn, with corresponding time anchoring in wander learning.

## Major Built Milestones

- Client/server architecture settled into a pure UI client and GPU-owning server.
- Watchdog became the boundary for restart, update, train, replay, wipe, artifact export, and migration export.
- Reflection execution moved server-side and became usable from both the UI and headless CLI.
- Chat sidecars became the durable per-exchange consolidation state beside immutable chat transcripts.
- Reflection memory became an append-only op-log folded into live facts, asks, and mirrored weights-bound recall items.
- The consolidation ledger became the durable anchor store for dialogue/persona/fact training state.
- Persona formation moved to revision, where original CoT is visible.
- Persona digest added versioned self-portrait artifacts and evidence clustering.
- Clean-base judge became the phase-two branch criterion path, with a maturity-gated override.
- Fact placement added a clean-base host-assignment step for facts.
- Fact anchors gained a direct weights path through host-exchange CoT injection, with ledger train-count tracking and baked-fact RAG eviction.
- Manifest replay turned `server/reflections/` into the immutable build recipe for full retrain.
- TIL/wander grew from manual fetch/learn into lookup, manual wander, autonomous idle wander, token budget, and one-shot SFT examples.

## Negative Results And Rollbacks

- Sequential resumed-adapter stacking collapsed the model after enough cycles even though every trained target was textually clean (2026-07). Memorization-grade verbatim fits sharpen the output distribution cycle over cycle; the damage lives in the probability geometry, not the tokens, and surfaces under sampling rather than under greedy probes. See `AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift`.
- Full reworded variant generation was removed. It was too slow and often either drifted off stance or collapsed back near-verbatim.
- Standalone fact/persona regeneration is not the normal training path. It risked eroding the reasoning channel, especially for Gemma-family thinking formats.
- A never-decaying RAG sketch was rejected. If RAG never fades, it masks whether the adapter internalized anything.
- Reattaching original CoT to answer-only revised IDEALs was rejected. The current path requires a faithful CoT-bearing IDEAL or skips the revised target.
- Per-exchange clean-base judging was rejected as too expensive. The current clean-base judge batches work under one swap per reflection run.
- Pairwise branch judging is not the current path. The runner logs prompt/VRAM diagnostics showing how much prompt context would remain shared even if options were serialized.
- Loss-based consolidation control is intentionally absent. Training loss is observed, but the design's feedback loop is conversation plus decay plus promotion probes, not "train until loss says learned."

## Operational Notes

- `python3 -m training.selftest` should be run from `server/`, not the repository root.
- `server/inference/server_config.json` is runtime config and may be absent from a clean checkout; `server.py` creates a default file if missing.
- Validation is currently disabled; UI skip-validation and judge-driven force-validation requests are both overridden by the shared validation switch until the probe is redesigned and re-armed.
