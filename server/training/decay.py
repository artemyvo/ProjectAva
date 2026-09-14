"""Decay schedule — how many training variants an anchor earns at a given stage.

A consolidating item is rehearsed hard while new and less as it sets:

    modifier(stage) = (N - stage) / N      # linear; stage 0 -> 1.0, stage N -> 0.0
    variants(stage) = round(B * modifier(stage))
    deprecated      = variants(stage) == 0

``B`` (base variants) and ``N`` (decay span) are configured per artifact type
(``dialogue`` / ``fact``). The curve lives behind :func:`modifier` so it can be
swapped for an empirically-tuned shape later without touching callers; ``linear``
is the deliberate starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Base/peak SFT learning rate for the offline train cycle (server_config.json `train_lr`).
# This is the reference LR the per-row multipliers + the global LR-schedule shape scale on
# top of. Lives here (a GPU-free module) so both the unsloth-heavy `train_cycle` and the
# inference `server.py` back-fill can import one value without drift. 8e-6 was found
# empirically to train better than the earlier 3e-6 under the trapezoid schedule.
TRAIN_LR_DEFAULT = 8e-6

# Default number of PLATEAU (full-LR hold) epochs in the "triangular"/trapezoid schedule
# (server_config.json `train_plateau_epochs`). The schedule is always one warmup epoch +
# N plateau epochs + one decay epoch, so total epochs = train_plateau_epochs + 2. Lives here
# beside TRAIN_LR_DEFAULT so `train_cycle` and the `server.py` back-fill share one value.
TRAIN_PLATEAU_EPOCHS_DEFAULT = 3

# Default LoRA rank (`get_peft_model` r) for the from-scratch adapter fit
# (server_config.json `lora_r`). Lives here beside the other train defaults so
# `train_cycle` and the `server.py` back-fill share one value; a CLI --lora-r / Sleep
# train_params.lora_r overrides it per run. Raised 16 -> 32 on 2026-07-29 (more adapter
# capacity for the from-scratch rebuild). Because scaling is rank-stabilized against the
# FIXED TRAIN_LORA_ALPHA below, this is approximately LR-neutral: gamma drops 1.0 -> 0.707,
# cancelling the ~sqrt(2) growth in ||dW|| that the extra rank would otherwise add, so
# TRAIN_LR_DEFAULT carries over. 8/64/128 remain available as capacity knobs.
TRAIN_LORA_R_DEFAULT = 32

# LoRA alpha for the from-scratch fit — a FIXED constant, deliberately NOT tied to `r`.
# Paired with `use_rslora=True` in `train_cycle`, so the adapter scaling is
#
#     gamma = lora_alpha / sqrt(r)          (rank-stabilized; Kalajdzievski 2023)
#
# rather than peft's default `lora_alpha / r`. The point is to make **rank an
# approximately learning-rate-neutral knob**, which the previous `lora_alpha == r`
# (gamma pinned at 1.0 for every rank) was NOT:
#
#   B initializes at zero, so the weight delta is dW ~= gamma * dB @ A, whose (j,k)
#   entry sums over **r** terms. Adam moves each entry of B by ~lr per step regardless
#   of gradient magnitude, so with gamma held constant ||dW|| grows like ~sqrt(r) — i.e.
#   raising the rank silently raised the effective LR on the weights (r=16 -> 32 was a
#   ~1.4x step-size bump, observed as a *less* stable adapter, not the expected
#   gentler one). gamma ∝ 1/sqrt(r) cancels exactly that sqrt(r) growth.
#
# 4 == sqrt(16), chosen so gamma == 1.0 at **r == 16** (the default when this was
# introduced; the default is now 32): an r=16 build after this change has scaling
# numerically IDENTICAL to every adapter built before it, so the tuned
# `TRAIN_LR_DEFAULT` baseline carries over untouched. Other ranks are then
# LR-matched to that calibration point (r=8 -> 1.414, r=32 -> 0.707, r=64 -> 0.5,
# r=128 -> 0.354). Change this only to re-calibrate the whole LR baseline.
#
# Nothing to migrate: peft records `use_rslora` in each adapter_config.json and
# recomputes gamma from it at load, so pre-existing adapters (flag absent -> False,
# alpha == their r) keep their original alpha/r == 1.0 scaling.
TRAIN_LORA_ALPHA = 4


def modifier(stage: int, decay_steps: int, curve: str = "linear") -> float:
    """RAG/rehearsal weight for *stage*, in [0, 1]. 1.0 when fresh, 0.0 when set."""
    if decay_steps <= 0:
        return 0.0
    if stage <= 0:
        return 1.0
    if stage >= decay_steps:
        return 0.0
    if curve == "linear":
        return (decay_steps - stage) / decay_steps
    raise ValueError(f"unknown decay curve: {curve!r}")


@dataclass(frozen=True)
class DecayConfig:
    """Decay parameters for one artifact type."""

    base_variants: int          # B — variants emitted at stage 0
    decay_steps: int            # N — stages until deprecation
    curve: str = "linear"

    def variants_for_stage(self, stage: int) -> int:
        """Number of training variants to emit for an item at *stage* (>= 0)."""
        return max(0, round(self.base_variants * modifier(stage, self.decay_steps, self.curve)))

    def modifier_for_stage(self, stage: int) -> float:
        """RAG-priority modifier for *stage* — used to scale retrieval scoring."""
        return modifier(stage, self.decay_steps, self.curve)

    def is_deprecated(self, stage: int) -> bool:
        """True once an item has decayed out: no variants, drop from RAG."""
        return self.variants_for_stage(stage) == 0


@dataclass(frozen=True)
class WallClockConfig:
    """Wall-clock decay knobs (REBUILD retrofit §1/§5e/§6). Hours-denominated.

    The LR-ramp sample points are set **directly** by ``lr_ramp`` (default ``[1,2,4]``):
    the ramp climbs 1 → 2 → 4 across ``[rag_only_window_h, lora_cap_age_h]``, capping at 4.
    (These superseded the old derived decay cumsum ``[3,5,6]`` — under the wall-clock
    retrofit the ramp is no longer a decreasing-variant decay curve, so it is configured
    as explicit sample points rather than derived from ``base_variants``/``decay_steps``.
    If ``lr_ramp`` is left empty the derived cumsum is used as a fallback.) These knobs
    also set the decoupled RAG fade + the cap-age contamination dose. Parsed from the
    ``consolidation.wall_clock`` block of server_config.json (all fields optional).
    """

    rag_only_window_h: float = 24.0   # multiplier 0 below this (models the pre-reflection span)
    lora_cap_age_h: float = 72.0      # LR multiplier reaches the ramp cap here (~3d)
    rag_cap_age_h: float = 96.0       # verbatim chat RAG reaches 0 here (~4d)
    # Consolidation-gist tent (crossfade partner of the verbatim fade above): the distilled
    # per-conversation summary ramps 0→1 over [0, rag_cap_age_h] (the inverse of the verbatim
    # fade — they cross at half weight), then decays 1→gist_floor_weight over
    # [rag_cap_age_h, gist_cap_age_h] and holds that semantic-memory floor.
    gist_cap_age_h: float = 192.0     # gist reaches its floor here (~8d)
    base_lr: float = 1e-5             # base LR the per-row multiplier scales (REBUILD §1)
    lr_ramp: tuple = (1.0, 2.0, 4.0)  # explicit LR-ramp sample points (cap = last = 4.0)
    contamination_enabled: bool = True
    contamination_dose: float = 1.0   # unmasked-copy multiplier at cap (§5e)
    contamination_additive: bool = False  # False: split 4→3+1; True: 4 masked + an extra 1 unmask
    # Fold the cap-age contamination PAIR into a SINGLE per-token-weighted row (~40% fewer
    # rows/tokens per build — the two split rows train the identical sequence twice). Off by
    # default; on, a cap-age exchange emits one row at LR-mult = the response total (4.0) with
    # a per-token loss weight of `dose/response_total` (0.25) on the final user turn, so the
    # response trains at the full cap and the user's voice entrains at the dose in ONE
    # forward+backward. NOT bit-identical to the two-step scheme (one Adam update vs two, one
    # weighted-mean normalizer vs two) — both are heuristics for the same intent; A/B them off
    # the same frozen corpus. See build_dataset._contamination_rows + train_cycle compute_loss.
    contamination_fold: bool = False
    # Minimum stripped user-turn length (chars) to be worth contaminating on. A cap-age
    # exchange whose final USER turn is shorter than this skips the unmask copy and trains
    # as a single masked row at the full cap — a short turn ("ok"/"да"/"why?") carries no
    # substantive voice to entrain and would only teach Ava to emit terse user-style filler.
    # 0 disables the gate (every cap-age exchange contaminates). Char-based so build_dataset
    # stays GPU/tokenizer-free; ~100 chars ≈ a real sentence or two.
    contamination_min_user_chars: int = 100
    # Wander chat-RAG fade (separate from the chat crossfade above): a persisted wander's
    # retrieval weight steps down every `wander_rag_step_h` hours through `wander_rag_weights`,
    # dropping out of RAG past the last step. Capped BELOW the chat curve's 1.0 — external
    # material is weighted under the relational corpus (the same philosophy as WANDER_LR_MULT).
    wander_rag_weights: tuple = (0.4, 0.3, 0.2, 0.1)
    wander_rag_step_h: float = 24.0
    # Day-0 fresh-window chat-RAG discount (curbs a very recent chat dominating a NEW one).
    # The linear wall-clock slope applies only AFTER a chat is frozen (reflected), and
    # `verbatim_rag_weight_hours` is already 0.75 by the end of the first day — so a recent,
    # unreflected chat sits at full RAG weight and can drown the live query. An hourly time
    # droop is layered UNDER the frozen crossfade via `min`: 1.0 at age 0, falling linearly
    # over `fresh_horizon_h` (day 0), then held. Its floor is DERIVED from the main
    # crossfade, NOT absolute: the day-0 chat has not reached the weights, so it must fade
    # GENTLER than `verbatim_rag_weight_hours` already would (which hits 0.75 at 24h — a
    # "partly consolidated" level that is a lie for an unreflected chat). The floor is
    # `1 - fresh_droop_frac * (main curve's drop across day 0)`: with the defaults,
    # `1 - 0.5 * (1.0 - 0.75) = 0.875` at 24h. It tracks `rag_cap_age_h` automatically.
    # Set `fresh_droop_frac=0.0` to disable this fresh-window adjustment.
    fresh_droop_frac: float = 0.5
    fresh_horizon_h: float = 24.0
    # Long-horizon floors. Verbatim chat is deliberately EXCLUDED: its dedicated
    # `verbatim_rag_weight_hours` reaches hard 0 at rag_cap_age_h, completing the episodic →
    # semantic handoff. `rag_floor_weight` remains the persona-anchor recall floor; facts are
    # exempt from decay. The gist reaches `gist_floor_weight` exactly at gist_cap_age_h and
    # holds, so the semantic summary survives after the raw exchange leaves RAG. With
    # retrieval gating on RAW similarity (rag_policy.rank_score), these floors only order
    # results — they cannot make an irrelevant item surface.
    rag_floor_weight: float = 0.2
    gist_floor_weight: float = 0.2
    # Recollection tent — the gist's FRESH sibling. A recollection is what Ava now makes of
    # an old conversation, written by a revisit pass; it shares the gist's grain (one per
    # chat) but NOT its clock: age runs from the reading's own birth (`ts`), not the chat's
    # date, which is the whole point — re-reading a two-month-old chat produces a memory that
    # is new today. Unlike the gist it does NOT ramp up (it has no verbatim competitor: the
    # conversation it recalls expired from verbatim long ago), so it holds 1.0 across
    # `recollection_hold_h`, then decays affinely to `recollection_floor_weight` at
    # `recollection_cap_age_h` and holds. The floor sits ABOVE `gist_floor_weight` on
    # purpose: a re-derived reading is more current than the faded gist beside it, and the
    # two coexist rather than one replacing the other.
    recollection_hold_h: float = 96.0        # full weight for ~4d after the reading
    recollection_cap_age_h: float = 192.0    # reaches its floor here (~8d)
    recollection_floor_weight: float = 0.3

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "WallClockConfig":
        raw = raw or {}
        cont = raw.get("contamination") or {}
        wr = raw.get("wander_rag") or {}
        fresh = raw.get("fresh_window") or {}
        rec = raw.get("recollection") or {}
        d = cls()  # defaults
        ramp = raw.get("lr_ramp")
        wr_w = wr.get("weights")
        return cls(
            rag_only_window_h=float(raw.get("rag_only_window_h", d.rag_only_window_h)),
            lora_cap_age_h=float(raw.get("lora_cap_age_h", d.lora_cap_age_h)),
            rag_cap_age_h=float(raw.get("rag_cap_age_h", d.rag_cap_age_h)),
            gist_cap_age_h=float(raw.get("gist_cap_age_h", d.gist_cap_age_h)),
            base_lr=float(raw.get("base_lr", d.base_lr)),
            lr_ramp=tuple(float(x) for x in ramp) if ramp else d.lr_ramp,
            contamination_enabled=bool(cont.get("enabled", d.contamination_enabled)),
            contamination_dose=float(cont.get("dose", d.contamination_dose)),
            contamination_additive=bool(cont.get("additive", d.contamination_additive)),
            contamination_min_user_chars=int(
                cont.get("min_user_chars", d.contamination_min_user_chars)),
            contamination_fold=bool(cont.get("fold", d.contamination_fold)),
            wander_rag_weights=tuple(float(x) for x in wr_w) if wr_w else d.wander_rag_weights,
            wander_rag_step_h=float(wr.get("step_h", d.wander_rag_step_h)),
            fresh_droop_frac=float(fresh.get("droop_frac", d.fresh_droop_frac)),
            fresh_horizon_h=float(fresh.get("horizon_h", d.fresh_horizon_h)),
            rag_floor_weight=min(1.0, max(0.0, float(
                raw.get("rag_floor_weight", d.rag_floor_weight)))),
            gist_floor_weight=min(1.0, max(0.0, float(
                raw.get("gist_floor_weight", d.gist_floor_weight)))),
            recollection_hold_h=float(
                rec.get("hold_h", d.recollection_hold_h)),
            recollection_cap_age_h=float(
                rec.get("cap_age_h", d.recollection_cap_age_h)),
            recollection_floor_weight=min(1.0, max(0.0, float(
                rec.get("floor_weight", d.recollection_floor_weight)))),
        )


@dataclass(frozen=True)
class ConsolidationConfig:
    """Per-type decay config plus the shared curve."""

    dialogue: DecayConfig
    fact: DecayConfig
    wall: WallClockConfig = field(default_factory=WallClockConfig)

    # Defaults are conservative, tuned to curb verbatim overfit on the dialogue path:
    # both types decay over 3 stages; dialogue gets a slightly fuller base (3 vs 2). The
    # dialogue curve is [3,2,1]→deprecate (6 verbatim copies over its life, peak 2 primary
    # +1 IDEAL/cycle); fact is the regenerated path. Tune empirically (see DESIGN.md).
    _DEFAULTS = {
        "dialogue": {"base_variants": 3, "decay_steps": 3},
        "fact": {"base_variants": 2, "decay_steps": 3},
        "decay_curve": "linear",
    }

    def for_type(self, item_type: str) -> Optional[DecayConfig]:
        """DecayConfig for ``dialogue`` / ``fact`` / ``persona`` (persona uses fact)."""
        if item_type == "dialogue":
            return self.dialogue
        if item_type in ("fact", "persona"):
            return self.fact
        return None

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "ConsolidationConfig":
        """Build from the ``consolidation`` block of server_config.json (or defaults)."""
        raw = raw or {}
        curve = raw.get("decay_curve", cls._DEFAULTS["decay_curve"])

        def _cfg(name: str) -> DecayConfig:
            d = {**cls._DEFAULTS[name], **(raw.get(name) or {})}
            return DecayConfig(
                base_variants=int(d["base_variants"]),
                decay_steps=int(d["decay_steps"]),
                curve=curve,
            )

        return cls(dialogue=_cfg("dialogue"), fact=_cfg("fact"),
                   wall=WallClockConfig.from_dict(raw.get("wall_clock")))


# --- Wall-clock age core (REBUILD retrofit §1) ------------------------------------- #
# Age is clocked from the CHAT timestamp (the session-file stem — when the conversation
# happened), NOT from reflected_at (which is only the frozen/bundle gate). These are pure
# and GPU-free so build_dataset / rag_engine / selftest share one implementation.

_SESSION_TS_FORMATS = ("%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S")


def parse_ts(ts) -> Optional[datetime]:
    """Parse a chat/session timestamp to a datetime, or None.

    Accepts a session-file stem (``20260705_014505``, optionally carrying a ``.json`` /
    ``.state.json`` suffix; a ``-`` time separator is tolerated) or an ISO timestamp
    (``reflected_at`` / ``built_at``). Returns None on anything unparseable.
    """
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        return ts
    s = str(ts).strip()
    for suf in (".state.json", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    for fmt in _SESSION_TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def wall_clock_age_hours(chat_ts, built_at) -> Optional[float]:
    """Hours between a bundle's chat timestamp and the build's ``built_at`` (REBUILD §1).

    None if either side is unparseable (the caller picks the fallback). A negative delta
    (clock skew / built before the chat) clamps to 0.
    """
    a = parse_ts(chat_ts)
    b = parse_ts(built_at)
    if a is None or b is None:
        return None
    return max(0.0, (b - a).total_seconds() / 3600.0)


def cumsum_curve(decay_cfg: DecayConfig) -> list:
    """Cumulative decay curve — the LR-ramp sample points (``[3,2,1]`` -> ``[3,5,6]``).

    The old per-stage *variant counts* become the wall-clock ramp's sample points; their
    relative profile is what carries over (REBUILD §1), reindexed onto hours below.
    """
    steps = max(1, int(getattr(decay_cfg, "decay_steps", 3)))
    run, out = 0, []
    for s in range(steps):
        run += decay_cfg.variants_for_stage(s)
        out.append(run)
    return out


def lr_multiplier_hours(age_h: Optional[float], decay_cfg: DecayConfig,
                        wall: WallClockConfig) -> float:
    """Continuous wall-clock LR multiplier (REBUILD §1).

    0 below ``rag_only_window_h`` (the RAG-only youngest window — no weight dose yet); then
    a continuous ramp through the sample points (``wall.lr_ramp``, default 1 -> 2 -> 4) across
    ``[window, lora_cap_age_h]``; the ramp cap (last sample point) at/after ``lora_cap_age_h``.
    The step quantization of the old stage curve is gone — the sample points are hit exactly and
    interpolated linearly between. ``age_h`` None (unparseable) is treated as fresh -> 0.

    The sample points come from ``wall.lr_ramp`` when set (the current path); otherwise they
    fall back to the derived decay cumsum ``cumsum_curve(decay_cfg)`` (the legacy shape).
    """
    cum = list(wall.lr_ramp) if wall.lr_ramp else cumsum_curve(decay_cfg)
    total = float(cum[-1])
    window, cap = wall.rag_only_window_h, wall.lora_cap_age_h
    if age_h is None or age_h < window:
        return 0.0
    if len(cum) == 1 or cap <= window or age_h >= cap:
        return total
    # fractional stage in [0, len(cum)-1] mapped across [window, cap]
    stage_f = (age_h - window) / ((cap - window) / (len(cum) - 1))
    lo = int(stage_f)
    if lo >= len(cum) - 1:
        return total
    return float(cum[lo] + (stage_f - lo) * (cum[lo + 1] - cum[lo]))


def verbatim_rag_weight_hours(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Verbatim-chat RAG weight: linear ``1.0 → 0`` at ``rag_cap_age_h``.

    This is the episodic half of the verbatim→gist handoff. At/after the cap the raw
    exchange is absent from chat RAG; the gist is at peak weight instead. ``age_h=None``
    keeps the pre-reflection/unknown-age path at 1.0, while ``RagEngine._chat_modifier``
    separately enforces the hard cap from raw chat age even when the bundle is unfrozen.
    """
    if age_h is None:
        return 1.0
    cap = wall.rag_cap_age_h
    if cap <= 0:
        return 0.0
    if age_h <= 0:
        return 1.0
    if age_h >= cap:
        return 0.0
    return (cap - age_h) / cap


def rag_weight_hours(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Persona-anchor RAG weight (1.0 fresh -> ``rag_floor_weight`` and hold).

    Persona recall retains the existing nonzero safety floor. Verbatim chat no longer uses
    this helper; see :func:`verbatim_rag_weight_hours`. ``age_h=None`` stays at 1.0.
    """
    floor = min(1.0, max(0.0, wall.rag_floor_weight))
    if age_h is None or age_h <= 0:
        return 1.0
    cap = wall.rag_cap_age_h
    if cap <= 0 or age_h >= cap:
        return floor
    return max(floor, (cap - age_h) / cap)


def gist_rag_weight_hours(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Consolidation-gist RAG weight — crossfade up, then decay to a semantic floor.

    The distilled per-conversation summary should stay quiet while the verbatim transcript is
    still ground truth, take over as it fades, then itself fade out. So the weight ramps
    ``0 → 1.0`` across ``[0, rag_cap_age_h]`` (the exact mirror of the verbatim fade, which
    goes ``1.0 → 0`` over the same span — the two cross at half weight), then decays
    ``1.0 → gist_floor_weight`` across ``[rag_cap_age_h, gist_cap_age_h]`` (~4 more days by
    default) and HOLDS the floor. The second leg is an affine interpolation, so the configured
    floor is reached exactly at ``gist_cap_age_h`` rather than early through a zero-target
    clamp. The gist is the long-horizon recall channel — the only chat-RAG representation
    after verbatim expires — so it never disappears.

    ``age_h`` None / <= 0 -> 0.0: a brand-new gist carries no weight (its verbatim is present
    and authoritative), and an unknown-age gist can't be placed on the tent so it is withheld
    (the opposite default from ``rag_weight_hours``, whose unknown-age items stay at 1.0).
    """
    peak = wall.rag_cap_age_h
    end = wall.gist_cap_age_h
    floor = min(1.0, max(0.0, wall.gist_floor_weight))
    if age_h is None or age_h <= 0 or peak <= 0:
        return 0.0
    if age_h <= peak:
        return age_h / peak
    if end <= peak or age_h >= end:
        return floor
    remaining = (end - age_h) / (end - peak)
    return floor + (1.0 - floor) * remaining


def recollection_rag_weight_hours(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Recollection RAG weight — hold, then decay to a floor. Age is the READING's own.

    The gist's fresh sibling (see ``WallClockConfig.recollection_*``). Where
    :func:`gist_rag_weight_hours` ramps 0→1 because its verbatim transcript is still
    authoritative early, a recollection has no verbatim competitor — it is written by a
    revisit pass about a conversation whose verbatim expired days ago — so it is born at
    full weight: ``1.0`` across ``[0, recollection_hold_h]``, then affine
    ``1.0 → recollection_floor_weight`` across
    ``[recollection_hold_h, recollection_cap_age_h]``, then held.

    **``age_h`` is hours since the recollection was WRITTEN**, not hours since the chat it
    recalls — the one place in this module where the clock is per-record rather than per
    source bundle. Passing the chat's age here would exactly undo the point of the kind.

    ``age_h`` None / <= 0 -> 1.0: a just-written reading is at full weight, and an
    unknown-age one is treated as fresh rather than withheld (the opposite default from
    ``gist_rag_weight_hours``, whose unknown age cannot be placed on its tent at all).
    """
    hold = wall.recollection_hold_h
    end = wall.recollection_cap_age_h
    floor = min(1.0, max(0.0, wall.recollection_floor_weight))
    if age_h is None or age_h <= 0:
        return 1.0
    if hold > 0 and age_h <= hold:
        return 1.0
    if end <= hold or age_h >= end:
        return floor
    remaining = (end - age_h) / (end - hold)
    return floor + (1.0 - floor) * remaining


def fresh_time_weight(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Day-0 hourly RAG droop: 1.0 at age 0 -> a DERIVED floor at ``fresh_horizon_h``.

    Linear across ``[0, fresh_horizon_h]`` (hour-granular within the first day), then held at
    the floor — it is a fresh-window *floor*, NOT a fade to 0 (the long-range fade-to-0 is
    ``verbatim_rag_weight_hours``'s job for frozen bundles).

    The floor is **derived from the main crossfade**, not absolute: an unreflected day-0 chat
    has not reached the weights, so it must fade *gentler* than
    ``verbatim_rag_weight_hours`` already
    would (that curve hits 0.75 at 24h — a "partly consolidated" level that misrepresents a
    fresh chat). Floor =
    ``1 - fresh_droop_frac * (1 - verbatim_rag_weight_hours(horizon))`` — half
    (by default) of the main curve's day-0 drop — so with the defaults it lands at
    ``1 - 0.5 * (1 - 0.75) = 0.875`` at 24h and tracks ``rag_cap_age_h`` automatically.

    Meant to be combined with the frozen crossfade via
    ``min(verbatim_rag_weight_hours(age), fresh_time_weight(age))`` so the two never
    double-count. ``age_h`` None / droop_frac<=0 / horizon<=0 -> 1.0 (inert).
    """
    frac = wall.fresh_droop_frac
    horizon = wall.fresh_horizon_h
    if age_h is None or age_h <= 0 or frac <= 0.0 or horizon <= 0:
        return 1.0
    floor = 1.0 - frac * (1.0 - verbatim_rag_weight_hours(horizon, wall))
    if age_h >= horizon:
        return floor
    return 1.0 - (1.0 - floor) * (age_h / horizon)

def wander_rag_weight_hours(age_h: Optional[float], wall: WallClockConfig) -> float:
    """Chat-RAG weight for a persisted wander, from its wall-clock age (hours since capture).

    A STEP fade (not the linear chat crossfade): ``wander_rag_weights[floor(age/step)]``,
    default ``0.4/0.3/0.2/0.1`` per 24 h, then 0 once past the last step. Kept below the chat
    curve's 1.0 so external wander material is weighted under the relational corpus. ``age_h``
    None/negative is treated as fresh (weights[0]).
    """
    weights = wall.wander_rag_weights
    if not weights:
        return 0.0
    if age_h is None or age_h < 0:
        age_h = 0.0
    step = wall.wander_rag_step_h
    if step <= 0:
        return float(weights[0])
    idx = int(age_h // step)
    return float(weights[idx]) if idx < len(weights) else 0.0
