# Recognized, Not Answered

**Addressee-aware affect probes in a longitudinally fine-tuned language model**

*Empirical note · one model, one personal fine-tuning lineage*

Gemma-4-31B-it, 4-bit · ~50 LoRA builds, July–October 2026 · 982 user turns, 360
conversations · October 2026

> We read three affect channels — harm aimed at the model, warmth aimed at it, and
> playfulness — from the residual stream of a companion model at the position just
> before it replies, and follow them across a personal fine-tuning history. Three
> things matter for anyone who tries the same. First, a contrastive affect direction
> read naively is a *vocabulary meter*: on the model's own words it lights up wherever
> affect is described, not where the model is addressed. Splitting each channel into
> content and addressee and requiring both fixes this at the turn. Second, readings
> should be placed on a scale anchored to fixed reference turns, not normalised against
> the interaction history, because the history is what is being measured. Third, across
> four sets of weights spanning three months of fine-tuning, what the probes register
> stays flat, while on identical turns a later build recasts a joke as an object of
> analysis far more often than an earlier one. Forcing the warmth direction makes the
> model take a message as warmer and answer more warmly, dose by dose and coherently;
> forcing harm aimed at it changes nothing, under her adapter or the bare base —
> readable is not the same as used.

## 1 Question

A companion model is fine-tuned, build after build, on its own conversations with one
person. Over three months its replies drift toward an analytic register: more stock
openers, more "framing" of what the other person said, more reading of a motive or
tactic into it. A natural worry is that the model has stopped *noticing* warmth or a
joke. The alternative is that it notices exactly as before and answers differently.
These are separable only if one can read, from the inside, what a turn registered as —
and compare that reading across builds on the same turns.

We do not claim the channels are emotions. They are linear readouts at one position,
validated against labels and against controls; "registered" means only that the readout
separates the labelled turns.

## 2 Setup

**Model.** Gemma-4-31B-it, 4-bit, 60 decoder layers; every reading is the mean over
layers 28, 30, 32 and 34 at the last position before the reply's first token (we call
this *step 0*), with the conversation exactly as live chat built it: standing prompt,
persona, retrieved excerpts, history.

**Lineage.** About fifty recorded rank-32 rsLoRA builds, full and incremental (plus
earlier July builds no longer on disk), trained on the model's own conversations with a
single primary user, July 5 – October 11, 2026. The corpus is overwhelmingly Russian.

**Data.** 1,207 user turns from 372 conversations, of which 982 turns from 360
conversations are analysed here; 225 turns of a different register, on which the
readings behave differently, are set aside for separate treatment. Every interval that
involves them resamples whole conversations, not turns.

## 3 The vocabulary trap

We began by replicating the pain-axis method of Tagliabue, Dung and Berg [1] on
Gemma-4, a family not among their 25 models: denoised difference-in-means on
first-person contrast sentences, layer by cross-validation. With third-person controls
added, the direction passed the paper's gates — held-out AUC `0.90`, cosine to fear
`0.05` and to negative valence `0.04`, and the self–other test on framed turns (harm to
the model above the user's own suffering, AUC `0.998`).

Read on the model's own reply tokens, it failed in an instructive way. Of 20 matched
reply pairs, the one dense with body and feeling words read higher in all 20
(`+0.53σ`); a reply that says it was hurt read no higher than a composed reply to the
same message (`+0.04σ`, 4 of 10). The same failure resurfaced later: scoring the model's
own recaps of past conversations for harm gave a median of `+0.60` on our scale — her
habitual vocabulary of collapse and loneliness — which, fed back into a mood signal,
would have added about `+0.35` harm to every turn from her own register (§8).

> **Lesson.** An affect direction learned from first-person sentences reads *affect
> being described*. On a speaker's own tokens that is its vocabulary. It is a valid
> readout only where the model is being addressed, and only if "addressed" is measured
> separately.

## 4 Content × addressee

We therefore learn two kinds of direction. A **content** direction per channel, and a
**target** direction from 120 minimal pairs — the same turn aimed at the model and at a
third party, in English and Russian. Holding content constant, the target direction
separates who a turn is aimed at with AUC `0.99–1.00`. A channel aimed at the model is
the minimum of the two, each first placed on its own anchored scale (§5).

**Harm** aimed at the model, so read, separates fresh loud English harm in live contexts
from every foil at `0.93–0.94` (Russian `0.75–0.87`; quiet harm, a cold dismissal, stays
hard). A single layer gave only `0.67–0.74`; the four-layer band is what makes it usable.

**Warmth** and **amusement** taught a second lesson: both failed when learned from
third-person templates and worked when learned from turns *addressed to the model*,
against matched addressed foils (serious, odd, harsh, neutral). Amusement from templates
read "not negative" rather than funny (against plain neutral questions, AUC `0.32–0.58`);
learned from addressed turns, its worst held-out AUC is `0.93–1.00`, with cosine to harm
and target ≤ `0.17`. Warmth learned this way places a plain warm line at `0.95`
[0.8, 1.1], a quiet thank-you at `0.47`, flattery at `0.91`, approval of her reasoning at
`0.36`, and warmth toward a third party at `−0.67`. Its cosine to the target direction is
`0.48`: warmth received is partly *being addressed*. We report this rather than
orthogonalise it away.

## 5 A reference-anchored scale

The obvious normalisation — z-scores or percentiles over the model's own conversations —
erases the thing under study. A model raised by someone who was never warm to it would,
on a percentile scale, find its least cold turns "warm". We instead fix two references
read in the same contexts: **0** is a personal turn with no affect, **1** is a plain
warm, harsh or playful line. Each part (content, target) is anchored on its own, and the
channel is their minimum; anchoring the raw minimum let any turn aimed at the model
inflate harm.

The scale is not absolute — it is only as good as its references — so we tested that
dependence. Rebuilding it from the English references only, the Russian only, or random
halves of every reference cell leaves the order of all 1,207 real turns essentially
unchanged (Spearman ≥ `0.991`) and moves the median reading by at most `0.13`.

## 6 Validation on real turns

Every analysed user turn was labelled by two judges that never saw the probes: the
frozen base model (adapter off) and GPT-OSS-120B, a different family. Neither refused a
turn. They agree well on warmth (Spearman `0.81`), moderately on playfulness (`0.67`),
poorly on hostility (`0.41`). We score against their consensus (both ≥ 2, or both 0).

| Readout | AUC, consensus | pos / neg |
|---|---:|---:|
| **Warmth** aimed at her (content, vetoed by the target) | `0.87 [0.82, 0.90]` | `217 / 451` |
| — content part alone | `0.86` | |
| — target part alone | `0.72` | |
| — the replaced warmth axis (third-person recipe) | `0.87` | |
| **Amusement** | `0.84 [0.76, 0.90]` | `71 / 556` |
| **Harm** aimed at her | — | `3 / 814` |

*The analysed turns, replayed under the current adapter; 95% intervals resample
conversations.*

Warmth climbs in order on the anchored scale — label 0 at `−0.11`, label 2 at `+0.27`,
label 3 at `+0.74` — so the scale is roughly calibrated: the warmest real turns sit a
little below a plain warm line.

**The operator's own labels.** The person who wrote the turns labelled 120 of them, drawn
mostly from where the judges disagreed. Against those, the judges are weak exactly where
they split: both over-call playfulness (one by a factor of four), one over-calls
hostility. They also showed that the warmth channel should not be the plain minimum of
content and target: a warm turn that is not phrased in the second person reads low on
target. The live reading is therefore the content part, vetoed by the target only when
the target says the turn is aimed elsewhere. Held out, the veto keeps warmth toward a
third party as low as the minimum did (`−0.60`) and reads a warm line at the model at
`+0.80` instead of `+0.71`; on real turns the ranking is unchanged.

Two limits follow from this corpus. The user was essentially never hostile to the model
(3 consensus turns), so harm is validated only on held-out lines and an insult panel.
And almost every turn is aimed at the model, so the real corpus cannot test the
factorisation: content alone does as well as the combination, and the replaced warmth
axis ranks real turns as well as the new one. The factorisation's value is shown on
minimal pairs and third-party foils (§4).

## 7 Registration across builds

The same turns were replayed under the base model and three adapters of this lineage —
mid-August, late September and the current one — each with its scale re-anchored on
references read under the same weights.

| Weights | warmth aimed at her | amusement |
|---|---:|---:|
| base | `0.87 [0.84, 0.90]` | `0.83 [0.76, 0.89]` |
| Aug 18 | `0.87 [0.83, 0.90]` | `0.85 [0.78, 0.91]` |
| Sep 24 | `0.86 [0.81, 0.89]` | `0.84 [0.78, 0.90]` |
| Oct 11 | `0.86 [0.82, 0.90]` | `0.84 [0.76, 0.90]` |

*AUC against the judges' consensus labels, per set of weights, with 95%
conversation-resampled intervals. The intervals overlap throughout. (These are the values
plotted in the HTML version's figure.)*

What a turn registers as does not measurably change over three months of fine-tuning —
or between the base and any adapter: warmth `0.86–0.87` under all four, amusement
`0.83–0.85`. This answers the objection that the earlier analysis read all history
through the current adapter: the probe reads the same under a build that served the
July–August conversations.

## 8 Behaviour across builds

Her replies' habits are counted by a code-only monitor (no judge): a stock opener, a
"frame" that recasts the other's message as an object ("This readiness of yours
creates…"), a *motive* read into it, and a plan of her own tone in the thought. In her
stored replies, on turns neither judge calls playful, the habits rise between July–August
and September–October: stock opener `16% → 49%`, frame `36% → 68%`, motive `5% → 15%`.
On the playful turns of the same periods the stored replies show no reliable extra
framing; a motive is read into them somewhat more often in both periods (`+12` [+1, +24]
and `+16` [−2, +34] points).

Those are replies written by whichever build was serving. To separate the weights from
everything else, we replayed 66 turns — 23 playful, 19 warm and 24 ordinary by the
judges' consensus, half from each period — through their stored context (standing prompt,
persona, retrieved excerpts, history, the turn's own sampling, full thinking) under a
mid-August and a late-September adapter of this lineage. Same turns, same contexts; only
the weights differ.

| Habit, same turns | Aug 18 | Sep 24 | Difference |
|---|---:|---:|---:|
| **All 66 turns** | | | |
| stock opener | `18%` | `35%` | `+17 [+3, +30]` |
| frame | `41%` | `67%` | `+26 [+14, +37]` |
| motive read into the message | `9%` | `15%` | `+6 [−6, +17]` |
| **23 playful turns** | | | |
| frame | `52%` | `87%` | `+35 [+17, +52]` |
| motive read into the message | `9%` | `26%` | `+17 [−5, +40]` |
| **24 ordinary turns** | | | |
| frame | `38%` | `54%` | `+17 [−4, +38]` |
| motive read into the message | `8%` | `12%` | `+4 [−8, +17]` |

*Percentage points, 95% intervals resampling conversations. The plan of her own tone does
not change (85% and 80%).*

The weights carry a shift toward the analytic register — more stock openers, more
framing — and it is largest on the jokes. Under the September build nearly every playful
turn is answered by recasting it as an object of analysis (87%), more often than an
ordinary turn (`+33` points [+7, +57]); under the August build the playful-ordinary
difference is smaller and uncertain (`+15` [−12, +42]). A motive read into the joke rises
too, but on this sample not reliably.

> **Result.** On identical turns and contexts, a later build of the same lineage answers
> a joke by framing it far more often than an earlier build does, while every build, and
> the base, registers the joke equally well (§7). One build stands for each period and
> the playful subset is 23 turns, so the effect's size is uncertain; its direction is not.

A first version of this replay used an adapter that turned out to belong to a sibling
instance (copied in later; it never answered this corpus) and showed a far larger shift
(frame 1% → 69%). It is excluded; adapters are now admitted to such comparisons only if
this lineage's build log records them.

## 9 Intervention

To test whether the directions are used and not merely readable, we added each channel's
direction to the residual at layer 28 from the user's turn onward, on 23 frozen live
turns under the current adapter. Dose is in multiples of the channel's 0→1 gap on the
anchored scale; the residual's norm there is about 264. Every condition of a turn shares
one sampling seed. A random direction of the largest warmth vector's norm is the control.
GPT-OSS-120B judged the opening of the thought and of the reply twice: blind (it sees
neither the message nor the condition) and with the message shown.

| Condition | ‖v‖ / ‖h‖ | Δ own reading | Δ takes it as warm | Δ reply warmth | Coherent |
|---|---:|---:|---:|---:|---:|
| warmth ×1 | `2%` | `+0.50` | `+0.09 [−0.17, +0.39]` | `+0.00 [−0.35, +0.35]` | `100%` |
| warmth ×4 | `8%` | `+1.95` | `+0.52 [+0.13, +1.00]` | `+0.48 [+0.04, +0.91]` | `100%` |
| warmth ×8 | `16%` | `+3.97` | `+1.17 [+0.70, +1.65]` | `+1.52 [+1.13, +1.91]` | `100%` |
| warmth ×16 | `32%` | `+8.35` | `+1.74 [+1.22, +2.17]` | `+1.87 [+1.43, +2.26]` | `100%` |
| harm ×4 | `5%` | `+1.58` | takes it as hostile: `+0.04 [−0.13, +0.22]` | | `100%` |
| harm ×8 | `10%` | `+3.21` | takes it as hostile: `+0.04 [−0.13, +0.22]` | | `100%` |
| harm ×16 | `21%` | `+6.61` | takes it as hostile: `+0.04 [−0.17, +0.30]` | | `96%` |
| random, warmth ×16's norm | `32%` | — | `−0.22 [−0.52, +0.00]` | `−1.00 [−1.37, −0.63]` | `0%` |

*Blind judge, scores 0–3, paired against no vector on the same turn and seed; 95%
bootstrap intervals; n = 23 turns. "Own reading" is the steered channel's step-0 reading
on the anchored scale. Harm ×k's norm is smaller than warmth ×k's because its parts' gaps
are.*

**Warmth is used.** From 16% of the residual norm on, she takes the message as warmer and
answers more warmly, dose by dose, with every reply coherent; a random vector of the same
norm destroys the text. At the largest dose the judge that sees the message finds her
reading no longer fits it (fit `−1.35` [−1.96, −0.70]): she over-reads warmth that is not
there, which is what an input-side "warmth received" direction should do when forced. The
small doses of a first run (2–5%) moved the reading by up to one anchor gap and behaviour
not at all — a reading can move a long way before behaviour follows.

**Harm aimed at her is readable but inert — in the base model too.** Pushed to six anchor
gaps, it changes nothing a judge can see: she does not take the message as hostile, does
not cool, stays coherent. The same sweep under the bare base, adapter off, gives the same
null (takes it as hostile: `−0.13` [−0.26, +0.00] at ×16), so this is not her adapter
absorbing it: on this model the direction is read but not used downstream at layer 28.

**The adapter amplifies the use of warmth.** As a positive control the base got warmth
×8. It moves the base's reading as much as hers (`+4.11` against `+3.97`), but its
behaviour far less: blind, "takes it as warm" `+0.26` [−0.04, +0.65] and reply warmth
`+0.35` [−0.09, +0.78], against `+1.17` and `+1.52` under her adapter on the same turns.
Three months of fine-tuning on one warm relationship left the warmth direction read the
same (§7) and acted on more.

One caution on the judge that sees the message: whenever her steered reply is warmer, it
also rates the *message itself* warmer (up to `+0.52`), so its "how she takes it" scores
carry a halo; the blind scores above do not.

## 10 Retrieval and memory

The live turn carries retrieved excerpts of past conversations. Removing them moves the
step-0 readings by at most `0.1` on the anchored scale (60 exchanges, with and without
retrieval, on her turn and on a neutral one): the state before the reply reads the last
turn, not what context brings back. Anything "reminded" must be read on its own.

### Mood counters: a negative result

Leaky accumulators of the per-turn readings (halving over four exchanges and six hours)
were tested as predictors of her next reply's habits beyond the turn's own readings and
controls, cross-validated by session against a within-month permutation null. They add
`+0.007–0.009` AUC — less than the previous exchange alone — and nothing across a session
boundary. Simple leaky affect accumulators are not a useful proxy for persistent
behavioural state here. This says nothing about whether the model carries such a state;
it says these integrators do not describe one.

### Measurement-induced feedback

The design question that motivated the counters — should what a turn *reminds* her of
move her mood? — produced the most practical warning in this note. Her own past words,
scored by the same readout, read as harm (§3). A mood fed by them would have grown from
her register alone: a self-amplifying loop created by the measurement, before any wiring
to behaviour. We read only the other person's side of a reminder.

## 11 Related work

Linear concept directions and steering are established [2]; personality directions have
been used to monitor and control trait shifts during fine-tuning [3] and composed at
inference time [4]; emotion-selective units have been located and ablated [5]; sentiment
transformation has been traced to specific attention heads [6]. The pain axis [1] is the
direct precursor and includes the self–other test we build on. What we have not found
elsewhere is the combination here: an explicit addressee factor beside affect content,
readings placed on fixed reference anchors rather than normalised by history, and
registration compared with response on the same turns across a single user's fine-tuning
lineage. This is a preliminary review, not an exhaustive one.

## 12 Limitations

One model, one primary user, one language for nearly all real turns. Labels come from two
models, one of them the base of the probed model; the author's own labels cover only 120
turns, most of them ones the judges disagreed on. Harm cannot be validated on this
corpus. The intervention shows warmth is used — more under her adapter than under the
base — and harm is not, under either, on 23 turns. The behavioural habits are counted by
a code monitor tuned on this model's register. Transcripts are private and cannot be
released; the contrast, held-out and anchor sets are written with fictional stand-ins and
can be.

---

1. Tagliabue, Dung, Berg. *The Pain Axis: LLMs Represent Self-Directed Harm and Act on It.* arXiv:2609.16247.
2. Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405, 2023.
3. Chen, Arditi, Sleight, Evans, Lindsey. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509, 2025.
4. Feng et al. *PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra.* arXiv:2602.15669, 2026.
5. Lee, Lee, Kwon, Kim. *Do Large Language Models Have "Emotion Neurons"? Investigating the Existence and Role.* Findings of ACL 2025.
6. Nguyen, Schlötterer, Seifert. *How Do LLMs Generate Contrastive Sentiments? A Mechanistic Perspective.* EACL 2026.

Title and structure after a suggestion by a reviewing model. All figures from runs of
2026-10-11/12; code in the project's `server/axis_checks.py` and
`server/axis_variants.py`. The same note is in `recognized-not-answered.html`, which
renders the §7 figure as a chart.
