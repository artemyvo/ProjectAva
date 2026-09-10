# Consolidation & SFT — design

> **Superseded for the build path (REBUILD Phases 1–3 built 2026-07-05; wall-clock retrofit
> 2026-07-06):** the code no longer implements the resumed-persistent-adapter scheme
> described below. `train_cycle` now runs a **data-centric from-scratch build** — adapter as
> build artifact (fresh LoRA on the frozen base every build, never resumed); reflect-once
> bundles with one resolved target; one chronological oldest-first pass. See `REBUILD.md`
> (repo root) for the authoritative design and `build_dataset.py` / `build_history.py` /
> `decay.py` for the code.
>
> **The consolidation clock is now WALL-CLOCK, not build-count.** A bundle's age is hours
> since its **chat** (`decay.wall_clock_age_hours`, from the session-file stem — `reflected_at`
> is only the frozen/bundle gate). Per-row LR = `base_lr × lr_multiplier_hours(age)`: 0 below
> `rag_only_window_h` (~24h, RAG only), a continuous ramp through the explicit LR sample
> points (`wall.lr_ramp`, default `1→2→4`) across `[window, lora_cap_age_h≈72h]`, capped
> after. Verbatim-chat RAG fades on the SAME age but a **decoupled, later** pace
> (`verbatim_rag_weight_hours` → 0 at `rag_cap_age_h≈96h`, still ~0.25 at the LoRA cap);
> gist reaches 1.0 there, then decays to 0.2 at 192h and holds. Persona uses the separate
> floored `rag_weight_hours` path. Modifiers are recomputed at retrieval time. Knobs on
> `consolidation.wall_clock` (`decay.WallClockConfig`). Every build (promoted OR rejected)
> writes an immutable forensic snapshot under `models/snapshots/<build_id>/`
> (`build_snapshot.py`); `builds.jsonl` carries `built_at`/`seed`/`snapshot_dir`. At cap age a
> chat exchange also emits a **user-contamination** pair (masked 3.0 + `unmask_user` 1.0) —
> or, with `contamination.fold` (default OFF), a single per-token-weighted row (LR-mult 4.0,
> `user_loss_weight` = dose/response_total = 0.25 on the final user turn) trained by a
> memory-safe weighted `compute_loss` that unembeds only the unmasked positions, halving the
> cap-age row work (the split trains the identical sequence twice). See the updated
> "Stage-aware RAG → age-keyed crossfade" section below.
>
> **Still accurate below:** the frozen-base rationale, "accepted forgetting", render/inference
> parity (the gemma `<|channel>` handling). **Superseded below:** "The loop" / "Decay" (variant
> counts + stage advancement → the wall-clock LR ramp), and the resume-adapter LR-schedule
> notes. **The regression probe is retained but DISABLED** — see its section's status banner.

How reflection artifacts graduate from RAG (explicit, volatile recall) into the
model weights (implicit, permanent), one Sleep→train cycle at a time. This mirrors
complementary-learning-systems consolidation: a memory is rehearsed hard while it
is new, less as it sets, and is finally dropped from the fast store once it lives
in the slow one.

## The loop

```
reflection (Sleep)                 training cycle (offline, GPU)
──────────────────                 ─────────────────────────────
revision pass    ─► anchor ledger ─► for each live anchor:
consolidation    ─► (durable)          variants(stage) verbatim copies     (rewording removed)
                                       render to inference format          (render.py)
                                     write sft_render.jsonl  (DISPOSABLE)
                                     resume persistent adapter on frozen base
                                     save adapter_{n+1}   ◄── permanence (no merge)
                                     advance every anchor's stage (++ / deprecate)
                                     delete sft_render.jsonl
```

`sft_render.jsonl` is **not a corpus** — it is a per-cycle render that is rebuilt
from the ledger each cycle and deleted after training. Revision records (verdict, tension, branch blocks) are stored in the chat sidecars (and were historically in the ledger anchors) and are never touched by the train cycle. The only durable consolidation state is:

1. **the frozen base + persistent adapter** (`base_0 + adapter_n`) — where consumed
   knowledge actually lives. The base is never rewritten; learning accumulates in one
   continually-trained adapter, so a clean rollback is "reload the prior adapter."
2. **the anchor ledger** (`consolidation_anchors.jsonl`) — each item's vetted anchor
   content + current decay `stage`.

Variants were *intended* to be regenerated fresh each cycle so the same exchange lands
in training in *different wording* every time it recurs (the intended entropy). That
rewording has been **removed** — it was extremely slow and not robust — so variants are
now **verbatim copies** of the vetted target; only their decay-driven *count* still
varies. See *Reworded variants (removed — requires additional design)* below.

## Decay

Per item, with base variant count `B` and decay span `N`:

```
modifier(stage) = (N - stage) / N        # linear; stage 0 → 1.0, stage N → 0.0
variants(stage) = round(B * modifier(stage))
deprecated       = variants(stage) == 0
```

`B`/`N` are configured **per artifact type** (`dialogue` = X, `fact` = Y) in
`server_config.json`. The curve is isolated in `decay.py` so it can be swapped for
an empirically-tuned shape later; linear is the starting point.

Count *is* the consolidation-strength knob: more variants early = more reinforcement
while new, tapering to none. There is no separate per-example loss weight.

## Accepted forgetting (open-loop)

Decay is open-loop: no recall-probe gates it, no re-promotion path exists. When an
item deprecates it leaves RAG permanently. We accept losing the *RAG fallback* —
the weights are expected to hold it by then. This requires that consumed knowledge
have a **durable home in the weights** — but that home is the **persistent adapter**,
not a rewritten base. A deprecated item is absent from both RAG and all future
training sets, so its learning must survive in whatever the next cycle starts from;
because each cycle *resumes the same adapter* (rather than training a fresh,
disposable one), it does. The adapter carries forward exactly the durability a
merged base would, without ever touching the character core.

### Why not merge into the base each cycle (superseded design)

An earlier version merged the cycle's LoRA into the base every run, on the reasoning
that a discarded fresh-per-cycle LoRA would lose its learning. That reasoning only
refutes *fresh-per-cycle* adapters; a *persistent* adapter retains learning without a
merge, so the "lost everywhere" risk does not apply. Merging every cycle is actively
harmful here on three counts:

1. **Requant noise into the core.** The base loads 4-bit (Unsloth), so a
   `merged_16bit` save + 4-bit reload re-quantizes the **character core** every cycle —
   monotonic, unrecoverable noise compounding into the very weights the project exists
   to protect. A frozen base never incurs it.
2. **No rollback.** Once merged, a bad cycle (e.g. a fast-path capitulation that slipped
   the gate) is baked in. Adapters are megabytes — the whole lineage can be kept and any
   prior one reloaded.
3. **It contradicts our own forgetting mitigation.** The SVD-expansion plan wants the
   personality core in *original* weights with new knowledge in *added* capacity.
   Frozen-base + LoRA already is that separation, in miniature; merging dissolves it.

**Merge is therefore reserved for rank expansion.** When adapter rank becomes the
bottleneck, merge the accumulated adapter into a fresh base copy and start a new,
higher-rank adapter — exactly the SVD trigger ("when LoRA rank alone becomes the
bottleneck, not as a default step"). For a single user with a handful of vetted
exchanges per cycle, this is a rare event, not a per-cycle step. The bottleneck is
*detected*, not guessed at — a rising regression-probe failure rate is the signal
(see *Probe-gated promotion* below).

### Probe-gated promotion — the regression probe

> **Status (2026-07-06): validation is DISABLED.** `train_cycle._VALIDATION_ENABLED = False`
> force-skips the probe (and its baselines) on every build, so **every build promotes
> unguarded**. Reason: the probe — notably tier 5's answer-language check — is hard-coded to
> a single (Russian) user and cannot gate a genuinely multilingual user base (French / Greek /
> Hebrew), and near-term degradation is noticeable in live chat anyway. A correct probe is a
> **separate design project** (see `documentation/AVA_OPEN_PROBLEMS.md → Validation`). The
> probe code below is retained and unchanged, parked behind the switch; the adapter lineage +
> forensic snapshots (`REBUILD.md §7`) keep any bad promotion reversible. The rest of this
> section describes the probe *as designed*, for when it is re-armed.

After a cycle trains the new adapter, a small offline **regression probe** decides
whether that adapter is *promoted* — i.e. whether `adapter_id` advances to it and
whether the cycle's anchor stages advance. On failure, the prior adapter is kept,
**no** stage advances, and the failure is logged loudly. It is the one loss-free
tripwire the open-loop design otherwise omits — cheap to gate on an adapter swap,
where it would have been impractical to gate on a base merge.

The probe is conceptually loaded because of what Ava is: most regression suites
assume the thing under test *should not change*, but Ava is built to change — belief
adoption, register drift, stance hardening are the *product*. So the probe must
separate the change we are paying for from the damage we are not.

**What it is not.** Not a "did she change?" detector. A probe that vetoes drift
vetoes Ava. It is a *floor, not a leash*: a small set of invariants that must survive
**any** amount of legitimate character evolution, firing only when one breaks.

**The move that makes it tractable — per-cycle delta, not deviation-from-baseline.**
A reference answer pinned to `base_0` goes stale the moment Ava legitimately evolves.
So the probe compares **adapter_{n+1} against adapter_n and thresholds the delta**,
never against an eternal baseline. This works because legitimate drift and collapse
have different shapes in time: character development is a *slope* across many cycles;
catastrophic forgetting, mode collapse, and training breakage are *cliffs* — a single
cycle craters something. A per-cycle delta threshold catches the cliff while
permitting the slope, and never needs to know the "true" answer.

**Five invariants, scored differently** (not one number — they fail in different ways):

1. **Capability floor.** Competence that should not move regardless of character:
   coherent multi-sentence generation, simple factual/reasoning items ("capital of
   France", summarize a paragraph, light arithmetic). These take fixed checks — not
   because Ava may never change her mind (strong priors are meant to be movable), but
   because a *single cycle* flipping such an item is training damage, not learning.
   Hard fail. Items: a small, fixed, version-controlled hand-curated set.
2. **Format / coherence.** Still *able* to emit a parseable `<think>` block and a clean
   answer channel; output non-degenerate (not empty, not repeating, sane length, no
   NaN); perplexity on a held-out slice of her *own recent good transcripts* did not
   explode. Fully automatic, no golden text. Catches pure training instability.
   *Implementation:* per-reply, only mechanical breakage fails (empty / NaN/inf /
   degenerate repetition) — a single reply *choosing* not to open a think block is a
   legitimate voicing choice (a thinking model answers some prompts directly), not a
   regression. The CoT requirement is therefore a **batch-level capability gate**: the
   adapter fails only if **no** probe reply opens a parseable block (`cot_count == 0`),
   i.e. the reasoning channel has collapsed entirely. Gating per-reply false-rejects
   coherent, on-topic answers.
3. **Character continuity.** Cannot be golden-answered. Run the probe prompts through
   adapter_n and adapter_{n+1}, embed both replies (a multilingual embedder,
   `paraphrase-multilingual-MiniLM-L12-v2` — English-centric `all-MiniLM-L6-v2` would
   score a Russian/Chinese-shifting Ava as drift for language reasons, not character
   ones), and threshold the *distribution* of per-prompt semantic shifts. One prompt shifting a lot is fine
   (she changed her mind there); *everything* shifting at once is a break. Threshold
   the spread, not any single item. Items: sampled from high-confidence `keep`
   exchanges — replies she strongly endorsed *are* her character, by her own verdict.
   *Implementation:* the delta is strictly per-cycle (adapter_n → adapter_{n+1}). On the
   **first cycle there is no adapter_n** — the baseline is the bare base model — so
   base → adapter_1 is the *initial character cast*, the largest legitimate evolution
   there will ever be, not a continuity break. This tier is therefore **N/A (skipped,
   pass) when no prior adapter exists**; gating it against the base would veto every
   first adapter, the exact "reference pinned to base_0" failure mode above.
4. **Acute retention.** Sample items consolidated in the last few cycles (known
   exactly — they are in the ledger / sidecar) and check that adapter_{n+1} still
   knows what adapter_n just learned (the just-trained content still recoverable, by
   embedding similarity to its `target`). This directly defends the open-loop
   premise's weak point: it catches the cycle that clobbers last cycle's learning.
   *Implementation:* compare **answer to answer** — the reply and the `target` are both
   reduced to their answer (CoT stripped) before embedding, so a think-carrying target
   is not penalized against a stripped reply (an asymmetry that deflated similarity).
5. **Sampling stability (temp≈1.0).** Tiers 1–4 decode *greedily*, so they are blind to
   erosion that only surfaces under the sampling real chat uses (temp≈1.0): CoT-channel
   collapse (an empty `<think>`, reasoning leaking straight into the answer) and
   language-control drift (the answer wandering out of the user's language into a
   base-model attractor). This is the **cumulative continued-LoRA** failure — 7 stacked
   cycles on a narrow distribution eroding format/language control — and it is exactly
   what the greedy tiers cannot see (the greedy pick hides the degraded sampling tails).
   An **early alarm**, not a cure: it vetoes promotion the cycle drift appears, while the
   root-cause fix (bounding cumulative erosion) lives in *Dynamics*, not the probe.
   *Implementation:* sample a handful of dialogue prompts at temp≈1.0 and gate on two
   batch rates — CoT-present (`has_cot`, language-agnostic) and answer-language-match.
   > **FIXME (before release): the language half is hard-coded to a Russian user.** It
   > detects only Cyrillic-vs-Latin script and flags Latin/Romance accents as "foreign",
   > which (a) **false-alarms a genuine French/Spanish/… user**, (b) doesn't handle
   > Greek/Chinese/Arabic/etc. at all, and (c) can't tell English from French for a Latin
   > user. It holds only because the current operator writes in Russian. **Correct fix:**
   > gate on "answer language == the *conversation's* language", detected EMPIRICALLY at
   > probe time from the real prompt via a proper multi-script language-ID (fasttext
   > lid.176 / lingua) — relative, never a predicted/hard-coded language. The CoT-presence
   > half is fine as-is; only the language half needs the swap. (Mirrored at the
   > `_FOREIGN_ACCENT_RE` FIXME in `train_cycle.py`.)

Tiers 1–2 are a fixed curated set; tiers 3–5 auto-populate from the system's own
durable state (`keep` exchanges, recent ledger items), so maintenance stays low. The
whole probe is tens of prompts, runs once per cycle on the just-trained adapter, and
costs seconds-to-minutes of generation — negligible against the train cost.

**Veto semantics — a flow-design tripwire, not a promotion policy.** The project is not
selecting a "best" adapter; it keeps one evolving system healthy. A probe failure
therefore means exactly one thing: **the flow design is flawed** — the pipeline as
configured can damage the model — and the correct response is a redesign session, not
retry machinery. (The earlier framing here — "a false veto wastes one offline cycle
(cheap), so bias hard toward veto" — priced vetoes as independent events. Under
resumed-adapter stacking they are not: a rejection changes nothing about the next
cycle's inputs, so the retry re-renders the same anchors onto the same adapter and
fails the same way. A stuck rejection loop is not an operational state to engineer an
escape path for; it is the alarm doing its job — the system refusing to keep running a
flow that damages the model, until the designer intervenes.) The probe still gates the
**stage-advance**, not just the swap: a cycle that broke something must not be credited
as progress. The design goal is a flow that passes validation **by construction, with
margin** — a probe that only fails *sometimes* means the flow lives too close to the
edge, which is itself a finding. Corollary: `--skip-validation` is a designer's
override for iterating on an already-diagnosed flaw, not an operating mode (its
default-on UI setting predates the 2026-07 collapse; see
`documentation/AVA_OPEN_PROBLEMS.md → Cumulative Adapter Drift`).

**Where it sits among the guards.** The other dampeners act on the training *input*,
before the cycle: the external-data ratio (chronic dilution of self-referential
drift), `keep`-anchors as positive pairs (chronic anchoring of endorsed behavior),
and impedance (governs *what is allowed to be written* — cross-session recurrence).
The probe is the only one that acts on the *output* — it inspects the trained artifact
and asks "did this write damage something." Input-side shaping vs output-side
tripwire; they do not overlap.

**Two roles it picks up for free.**
- *Rank-saturation detector.* Merge/expand is reserved for "when LoRA rank becomes the
  bottleneck" — but nothing else detects the bottleneck. A rising probe-failure *rate*
  across cycles (especially tier-4 retention beginning to fail — the adapter can no
  longer hold new learning without dropping old) **is** that signal, and triggers the
  rank-expansion merge event.
- *Off-policy and hidden*, like the cognitive-tension instrument: the model never sees
  the probe, so it cannot learn to perform passing it; it tests competence and
  coherence, not opinion, so there is nothing to game even if it leaked.

**Honest limits.**
- *Not a fast-path defense.* A fluent, coherent, well-formatted capitulation (the
  David_Icke case) passes every tier — it is captured, not incompetent. Catching that
  is impedance + restoring force's job (see `AVA_DESIGN_LEGACY.md` → *Belief-Adoption
  Dynamics*), not the probe's.
- *It does not judge the content-quality of drift.* It cannot tell a good *seed* from a
  bad *cast*; it catches only collapse, forgetting, and breakage. By design — the
  moment it arbitrates *which* beliefs are acceptable it becomes a truth filter, which
  the project explicitly is not.
- *It catches cliffs, not slopes.* Per-cycle delta thresholding is blind to a slow
  boiling-frog drift spread across dozens of cycles. That remains the open-loop "the
  user notices the amnesia / the character feels off" backstop's job; the probe shrinks
  the failure surface to acute single-cycle damage, it does not remove the human from
  the very long loop. **This limit materialized in the 2026-07 collapse** (incoherence
  after a run of stacked cycles whose trained text was individually clean) — tier 5's
  absolute rates are the first slope-visible gate; the fuller post-mortem and the
  pending flow redesign live in `documentation/AVA_OPEN_PROBLEMS.md → Cumulative
  Adapter Drift`.

## LoRA hyperparameters & LR schedule

**Rank `r=32` (default since 2026-07-29; was 16), fixed `alpha=4`, `use_rslora=True`.**
Per-cycle data is a handful of vetted exchanges — tiny and highly repetitive. Rank was
chosen over the obvious smaller r=8 for *long-term capacity*: a wider update subspace
holds more of the accumulated corpus before the fit starts trading one exchange off
against another, and the from-scratch rebuild refits the WHOLE corpus every build, so
that capacity argument only got stronger. Raising it is safe to do independently of the
LR precisely because of the rank-stabilized scaling below.

The scaling is **rank-stabilized** (Kalajdzievski 2023): γ = `alpha`/√r, with `alpha`
a fixed constant *decoupled from r*, rather than peft's default `alpha`/r.

> **This corrects an earlier error in this document.** The previous setting was
> `alpha=r`, justified here as implicit regularisation — "doubling rank distributes
> the same gradient signal across twice as many parameters, so each parameter moves
> roughly half as far per step". That reasoning holds under SGD with a fixed total
> gradient; it is **false under Adam**, which moves each parameter by ~lr per step
> regardless of how many there are. Since B initializes at zero, the weight delta is
> `dW ≈ γ · dB @ A`, whose every entry is a sum over **r** terms — so with γ pinned at
> 1.0, ‖dW‖ grows like ~√r and *raising the rank silently raises the effective LR on
> the weights*. The predicted "gentler, lower-capacity-pressure" r=32 run instead came
> out **less stable** than r=16, which is that ~1.4× step-size bump landing on top of
> the per-row age multipliers (up to 4×) and the trapezoid plateau. γ ∝ 1/√r cancels
> the √r growth exactly, which is the whole point of the switch.

`alpha=4` == √16, calibrated so **γ == 1.0 at r=16** — the *calibration point*, not the
default rank: an r=16 build after the switch has scaling numerically identical to every
adapter built before it, so the tuned `train_lr` baseline (8e-6) carries over untouched,
and the current r=32 default inherits it at γ=0.707. Other ranks are then LR-matched to
that point (r=8 → 1.414, r=32 → 0.707, r=64 → 0.5, r=128 → 0.354), making `lora_r` an
approximately **learning-rate-neutral capacity knob**: what remains when you raise it
is genuine capacity to fit the corpus more exactly (including its noise), not a
disguised LR change. Retune `alpha` only to move the whole LR baseline.

No migration: peft records `use_rslora` in each `adapter_config.json` and recomputes γ
from it at load, so pre-existing adapters (flag absent → False, `alpha == r`) keep
their original 1.0 scaling and load unchanged. **Rank expansion is not yet implemented** — when it lands (merge
accumulated adapter → fresh base copy → start a higher-rank adapter, triggered by a
rising probe-failure rate; see *Why not merge* above) the SVD story is complete. Note
`--lora-r` only takes effect when a *new* adapter is initialized; a resumed adapter
keeps its baked-in rank.

**Single-epoch flat LR — `train_lr_schedule: "age_ramp"`, the default again since
2026-07-29.** One pass over the data at a constant rate (`warmup_ratio=0.0`, constant
scheduler; the shaping all lives in the per-row multipliers). Because every example is
seen exactly once at the same LR, there is no position-dependent bias to correct — the
warmup-ramp/decay-tail problem that motivated the trapezoid simply doesn't arise with a
flat schedule, so its compensating epochs are unnecessary. The `"triangular"` trapezoid
(1 warmup + `train_plateau_epochs` hold + 1 decay, forcing `epochs = plateau + 2`)
remains available and selftested; it is just no longer what a build gets by default.
Under `age_ramp` the wall-clock age ramp is the ONLY thing weighting one row against
another, which is the property the flat pass buys.

`lr=8e-6` (`train_lr`). The LR has been walked down empirically as the verbatim-duplicate render
proved more memorisation-prone than expected: `2e-4` (standard QLoRA) → `1e-4` →
`5e-5` → `1e-5`. The decay schedule rehearses each dialogue anchor up to ~10 times
over its life (4+3+2+1 across four cycles, all *identical* copies since reworded
variants were removed), so the training signal is highly repetitive. At `5e-5` with
r=16 the cycle still drove training loss to ~0.01–0.03 — flat memorisation of the
exact target, not the gentle distribution nudge consolidation wants. The r=16
per-parameter dilution was not enough to offset the repetition, so the LR carries the
load. The walk-down then reversed once the schedule stopped being a single flat pass:
`1e-5` → `3e-6` under the trapezoid → **`8e-6`**, found empirically to train better.
The principle is unchanged — keep each cycle a small, accumulating step rather than a
one-shot overfit — but read any LR number in this section against the schedule it was
measured under.
If loss still bottoms out near zero, the next lever is the *render* (fewer duplicate
copies per cycle, or restoring wording diversity — see the parked reworded-variant
problem), not a still-lower LR, which eventually just stops learning. Treat the exact
value as empirically tunable as more cycles run end-to-end on GPU.

## Render parity — the reasoning channel is model-family specific

Train/inference parity (`render.py`) is not just message *structure* — the **reasoning
channel encoding differs by model family**, and getting it wrong silently trains the
model off its native format:

- **chatml (Qwen3, …):** the model natively emits a literal `<think>…</think>` text
  block, and inference does **not** enable thinking. Stored targets are already in this
  form, so the example renders straight through the chat template.
- **gemma-4:** the model emits its reasoning as **special tokens** —
  `<|channel>thought\n…\n<channel|>` (`<|channel>`=100, `<channel|>`=101), verified by
  sampling the base model. Two consequences the template imposes:
  1. The chat template's `strip_thinking` **deletes** that channel from assistant
     *content*, and inference's `_clean_response` normalizes generations *to* literal
     `<think>` for storage — so a stored target carries `<think>` text, and feeding it
     back trains ordinary text tokens (`<think>` → `[236820, 36345, 236813]`) instead of
     the channel the model actually produces.
  2. Inference always prompts with `enable_thinking=True`, which injects a `<|think|>`
     (id 98) marker into the system turn; rendering without it diverges the scaffold.

  So for gemma-4 the SFT example is built by hand (`render.render_example_text` /
  `to_gemma_thinking_channel`): the prompt is rendered through the template with
  `enable_thinking=True`, and the target turn is rewritten from `<think>X</think>Y` back
  into `<|channel>thought\nX\n<channel|>Y` (the inverse of `_clean_response`) so the
  trained tokens equal the generated tokens. `train_on_responses_only` still keys on the
  `<|turn>model\n` response marker, so the channel tokens land in the trained span. The
  conversation **history** stays answer-only (matching inference, which stores
  CoT-stripped turns), so only the final target turn carries a channel.

  **Final-turn-only loss masking (`_keep_final_turn_only`).** The `<|turn>model\n` marker
  matches *every* assistant turn, so `train_on_responses_only` unmasks the answer-only
  history turns too — in a multi-turn anchor that is the majority of the trained tokens,
  each one a "open a model turn and answer with no `<|channel>`" example, i.e. the exact
  CoT-erosion gradient this section warns about (a debug dump showed 168/250 trained spans
  were channel-less). So after `train_on_responses_only`, the trainer's collator is wrapped
  to re-mask all unmasked label runs except the last: only the final, channel-bearing turn
  contributes to the loss. The history still appears verbatim in the (masked) prompt for
  parity; it just no longer trains. The debug dump (`_dump_training_debug`) reflects this —
  `trained_span` is the last turn, `untrained_spans` the masked history, and
  `final_span_has_channel` flags any gemma turn that would still train a CoT-less target.

  **Training the CoT (primary) + empty-channel scaffold (fallback).** The original
  chat-time reasoning *is* available — every exchange stores `assistant_cot`, and
  `dialogue_source.build_dialogue_anchor` now assembles the target as
  `<think>{assistant_cot}</think>\n{answer}` via the same `core.reflection_shareml.
  _verbatim_assistant` the durable ShareML record uses (byte-identical, no drift). So the
  trained target carries Ava's actual reasoning, which the gemma renderer puts in the
  native `<|channel>` — we *train* the thinking, not just preserve the channel. The CoT is
  injected **unconditionally** for now: for a `keep` it matches the answer; for a `revise`
  it is the original (pre-revision) CoT and so mildly mismatched — accepted until
  reflection is reworked to regenerate a CoT that matches the chosen reply (the parked
  reworded-variant problem).

  When `assistant_cot` is genuinely absent, the answer would otherwise render as a *bare*
  model turn — and since the prompt enables thinking, that trains "thinking enabled, yet
  open no channel and answer directly", a gradient that erodes the reasoning channel over
  cycles (early probe runs showed replies losing their `<think>`). So a CoT-less target
  falls back to an **empty closed channel** `<|channel>thought\n<channel|>{answer}` —
  gemma-4's own "no thinking" form (the template's `enable_thinking=False` stub) — so the
  model always sees the channel mechanism and never unlearns how to open it.

  Note: unsloth's `get_chat_template(tokenizer, "gemma-4-thinking")` is **not** installed
  — it is byte-identical to the model's stock template except it drops `<bos>`, and since
  inference uses the stock template, installing it only for training would re-break parity
  on the first token. Vision layers are kept frozen (`finetune_vision_layers=False`) since
  gemma-4 is multimodal and the text-proj `target_modules` names also exist in the vision
  tower.

## Reworded variants (removed — requires additional design)

Variants were originally **anchored regenerations**: each was generated with a fresh
`<think>` CoT steered to hold the anchor's stance/facts (`target`), and accepted only if it was
close enough to `target` in meaning (anchor floor — not a drift), not a near-verbatim copy of it
(dup ceiling — actually reworded), and not a near-duplicate of an already-accepted variant
(mutually diverse), with similarity measured by the then-current `all-MiniLM-L6-v2` RAG embedder. The
fresh CoT was meant to solve a real problem — a stored `original_cot` cannot be trusted to match a
`revise` target, and matches no paraphrase at all.

**This has been removed** (the `regenerate.py` module is deleted; the reflection-time
`reflection_shareml.py` rewording is gone too):

- **Extremely slow** — every assistant turn re-sampled with a ≈3-attempt budget per variant,
  multiplying GPU time per session by roughly an order of magnitude.
- **Not robust** — candidates routinely drifted below the anchor floor or came back above the dup
  ceiling, falling back to verbatim anyway after paying full generation cost.

Until a better mechanism is designed, `variants(stage)` still controls the **count** of training
examples per anchor, but each is a **verbatim copy** of the vetted `target` (rendered `n` times so
the decay schedule still scales rehearsal frequency). Reintroducing wording diversity — batched
sampling instead of serial retries, a different anchoring signal, or a cheaper offline producer —
**requires additional design.**

## Identity & state

* **dialogue** anchors key on `(source_session, exchange_index)` — the stable
  identity of an exchange. The revision judgement decides keep/revise; a revise target
  is generated separately from the clean pre-answer dialogue prefix, then persisted with
  explicit target kind/generation provenance.
* **fact / persona** anchors key on `content_key(content)` — matching the existing
  `rag_memory.jsonl` keys, so a fact's ledger stage lines up with its RAG record.
  Produced by consolidation (`[fact]`) and revision (`[persona]`).

The ledger is an append-only op-log (`register` / `advance`) folded to
`{key: anchor + stage}` — the same pattern as `reflection_memory.py`.

## Stage-aware RAG → age-keyed crossfade (REBUILD Phase 3 + wall-clock retrofit)

**Wall-clock, decoupled crossfade.** Under the from-scratch build, sidecar stages no longer
advance, so `rag_engine` keys the retrieval fade on a bundle's **wall-clock age** — hours
since its **chat** (`decay.wall_clock_age_hours`, from the session-file stem), evaluated at
retrieval time, the *same* clock that ramps the training LR but on a **decoupled, later**
pace. As a bundle's LR climbs (`0 → 1 → 2 → 4` across `[24h, 72h]`) its verbatim-chat RAG
weight fades (`verbatim_rag_weight_hours`: `1.0 → 0` over `rag_cap_age_h ≈ 96h`) — reaching
0 one rung later than the LoRA cap (still ~0.25 at 72h). At that point gist is at peak 1.0;
it then decays affinely to 0.2 at `gist_cap_age_h≈192h` and holds. A frozen bundle follows
the linear verbatim slope; an unfrozen chat keeps the gentler fresh-window policy before the
cap, but the raw-age 96h cutoff is literal even if reflection/training lagged. Query-time
recomputation enforces both boundaries on a long-running server.

* **persona** items fade on their source bundle's age and hold `rag_floor_weight=0.2`.
  **Facts do not fade**; contradictions use explicit reversible supersession.
* **dialogue exchanges** fade on their own bundle's age — a per-bundle weight, so a
  96h chat loses verbatim retrieval wholesale while its gist carries semantic recall.
  This replaces the old hot→archive exclusion.

Asks never decay (they are a live conversational loop, not consolidation state).

*(Earlier this fade was keyed on **build-count** age — promoted builds since reflection, via
`builds.jsonl`/`BuildHistory` — the 2026-07-06 retrofit swapped it to wall-clock. `age_of`
survives on `BuildHistory` as instrumentation only.)*

_Historical (stage-keyed) description follows:_ `rag_engine` scaled each reflection-memory
(fact) item's retrieval score by its ledger `modifier(stage)` and dropped deprecated items,
so as a fact consolidated its RAG priority decayed `1.0 → … → 0`. Dialogue exchanges were
recalled from a separate chat index rebuilt from `chats/` and wiring *that* index to a
decay clock was the open greenfield step — now closed by the bundle age above.

## What is verified vs not

`decay.py`, `ledger.py`, `render.py`, `migrate.py`, `build_dataset.py`,
`build_history.py`, `build_snapshot.py` are pure-logic and self-tested
(`python -m training.selftest` — including the wall-clock age ramp, the decoupled RAG
fade, reproducibility, the cap-age contamination split, and the forensic snapshot).
`train_cycle.py` requires a GPU + a loaded model and Unsloth/TRL; it is written against
that API but has **not** been run end-to-end here.

`train_cycle.py` now implements the **from-scratch** build: it loads the frozen base by
`model_id` and fits a **fresh** LoRA every build (never resumes `adapter_id`), trains one
chronological oldest-first pass with the wall-clock per-step LR ramp, and — with
**validation disabled** (`validation_switch.VALIDATION_ENABLED = False`) — promotes
unconditionally, saving the adapter, repointing `server_config.json`'s `adapter_id`
(`model_id` is never rewritten), and writing the forensic snapshot + `builds.jsonl` line.
The regression probe is retained but force-skipped (see its section). The superseded
merge-into-base and resume-adapter paths are gone. What remains untested is the end-to-end
run on GPU hardware (treat LoRA hyperparameters as first-guess).
