"""From-scratch corpus assembly — REBUILD.md §5a (Phase 2).

``build_dataset`` is the pure, GPU-free heart of the rebuild: it gathers every
**frozen bundle** (a reflected chat = transcript + sidecar) plus the one-shot wander
queue, emits **one row per revisable exchange** (no decay-count variant copies, no
IDEAL regularizer), sorts them **chronologically oldest-first**, and stamps each row
with its **wall-clock age** (hours between the chat's own timestamp and the build's
``built_at``) and the resulting per-row ``lr_multiplier``. Repetition is gone; the
age-keyed LR ramp carries consolidation strength (REBUILD.md §1). Order + per-row
multiplier are what let ``train_cycle`` run a single sequential pass with a
per-optimizer-step LR (verified on Gemma4-31B).

Age is clocked from the **chat timestamp** (the session-file stem — when the
conversation happened), not from ``reflected_at`` (whose presence is only the
frozen/bundle gate). A bundle younger than ``rag_only_window_h`` gets multiplier 0 and
emits **no trainable row** at all — it lives in RAG only until it has aged enough to
consolidate (REBUILD.md §1). The age→multiplier math is the pure ``decay`` helpers
(``wall_clock_age_hours`` / ``lr_multiplier_hours``), shared with ``rag_engine``.

Kept model-free so it self-tests without a GPU: message assembly is
``render.build_messages`` (the gemma ``<|channel>`` rewrite happens later, at train
time, in ``render.render_example_text``). Persona/fact CoT injection reuses the exact
``persona_render`` / ``fact_render`` primitives ``train_cycle`` used, applied strictly
to the host exchange's own bundle (locality — REBUILD.md §3; the ledger remains the
store, but is read per-host so a persona/fact only rides the exchange it was distilled
from / placed on).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Reuse the inference layer's sidecar/anchor helpers (same trick as ledger.py).
_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from core.chat_sidecar import ChatSidecar, session_name_from_sidecar  # noqa: E402

from training import fact_render, persona_render, render               # noqa: E402
from training.decay import lr_multiplier_hours, wall_clock_age_hours   # noqa: E402
from training.dialogue_source import build_dialogue_anchor             # noqa: E402

# Fact injection cap — kept here so build_dataset stays free of any train_cycle import
# (train_cycle imports this module). PERSONA_INJECT_CAP was retired with explicit persona
# CoT injection (see _inject): persona now reaches the CoT implicitly, so there is no
# build-time persona prepend to cap.
FACT_INJECT_CAP = 2

# One-shot wander/news rows train at a fixed multiplier one rung below a fresh chat's
# 3x (REBUILD.md §5a) — external knowledge should not outweigh the relational corpus.
# Wander is kept one-shot for now (a per-run decision), so it does not age; it just
# rides this fixed multiplier for its single build.
WANDER_LR_MULT = 1


@dataclass
class BuildRow:
    """One trainable row of a build: a rendered conversation + its age/LR metadata."""

    messages: list
    age: float                       # wall-clock hours since the chat (0.0 for wander)
    lr_multiplier: float
    source: str                      # "chat" | "wander"
    source_session: str
    exchange_index: Optional[int]
    order_key: tuple                 # chronological sort key (ts, exchange_index)
    target: str = ""
    anchor: Optional[dict] = None    # dialogue anchor (chat rows) — feeds the probe
    persona_keys: tuple = ()
    fact_keys: tuple = ()
    unmask_user: bool = False        # cap-age contamination copy (§5e) — trains the user turn too
    # Render-only preview row (include_fresh): a chat too young for the LR ramp (or only
    # background-frozen), emitted at multiplier 0 so it reaches the snapshot render — and
    # thus the Training review tab — WITHOUT entering the trained corpus. The caller is
    # responsible for splitting these out before training.
    preview: bool = False
    # Folded contamination (contamination_fold): when set, this single row REPLACES the
    # masked+unmask split pair. It carries the response total on ``lr_multiplier`` and this
    # per-token weight (dose / response_total) for the final user turn; the collator turns it
    # into a loss-weight vector so one forward+backward does the work of the two split rows.
    # ``None`` on every non-folded row (the two-row path leaves it unset).
    user_loss_weight: Optional[float] = None


def index_by_exchange(anchors: list) -> dict:
    """Group persona/fact ledger anchors by their host ``(source_session, exchange)``.

    The host is the ``source_exchange`` snapshot (persona: the distilled-from exchange;
    fact: the placement judge's pick), falling back to the anchor's top-level fields.
    Buckets are freshest-first by registration ``ts`` so the newest self-statements inject
    first — the same join for personas and facts, so each rides only its own host exchange.
    """
    index: dict = {}
    for p in anchors:
        se = p.get("source_exchange") or {}
        sess = se.get("source_session") or p.get("source_session")
        idx = se.get("exchange_index")
        if idx is None:
            idx = p.get("exchange_index")
        if not sess or idx is None:
            continue
        try:
            key = (sess, int(idx))
        except (TypeError, ValueError):
            continue
        index.setdefault(key, []).append(p)
    for bucket in index.values():
        bucket.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return index


def _inject(target: str, fact_cands: list) -> tuple:
    """Inject up to the fact cap into *target*'s CoT (model-free).

    Returns ``(text_or_None, persona_keys, fact_keys)``. ``persona_keys`` is always
    empty: explicit persona injection was **retired**. Persona now reaches the trained
    CoT *implicitly* — a ``keep``/branch target carries the original chat-time thought
    (generated with persona RAG, so it already gestures the trait it was distilled from),
    and a ``revise``/IDEAL target is generated persona-conditioned (the IDEAL seam injects
    the exchange's relevant persona into the generation context and persists it into the
    anchor system prompt; see ``reflection_source.build_ideal_messages`` +
    ``dialogue_source.build_dialogue_anchor``). Prepending a synthetic persona line at the
    *start* of every host CoT taught an "open reasoning by reciting persona" prior that
    flooded the CoT at inference — the reason it was removed. Facts keep the injection path.

    ``None`` only when the target carries no usable answer span (unclosed / CoT-only /
    doubled-think — ``render``'s hard guard would trip mid-train). A CoT-*less* answer-only
    target IS trainable — it renders as gemma's empty-channel scaffold (REBUILD.md §4 row 3)
    — it just has no ``<think>`` to inject into, so it passes through un-injected.
    """
    target = (target or "").strip()
    if render.trainable_answer(target) is None:
        return None, (), ()
    if not fact_cands:
        return target, (), ()
    parts = persona_render.split_cot(target)
    if parts is None:
        return target, (), ()          # answer-only (empty channel) — nothing to inject
    thought = parts[0]
    sel_f, raw_f = persona_render.select_persona_injections(
        thought, fact_cands, cap=FACT_INJECT_CAP)
    stmts_f = [fact_render.fact_cot_line(s) for s in raw_f]
    if stmts_f:
        injected = persona_render.prepend_persona_lines(target, stmts_f) or target
    else:
        injected = target
    return injected, (), tuple(p["key"] for p in sel_f)


def _live_hostable_anchors(ledger) -> tuple[list, list]:
    """Live persona anchors and *placed* fact anchors from the ledger fold.

    Under the rebuild the ledger is read as an immutable store (no stage advance, no
    ``FACT_TRAIN_CAP``): every persona rides its host exchange, and every fact that the
    placement judge gave a ``source_exchange`` rides that host — in *every* build. RAG
    decay of these is Phase 3's crossfade, not here.
    """
    personas, facts = [], []
    for a in ledger.fold().values():
        if a.get("superseded"):
            continue   # softened by reconciliation — kept as evidence, not trained
        t = a.get("type")
        if t == "persona":
            personas.append(a)
        elif t == "fact" and (a.get("source_exchange") or {}).get("exchange_index") is not None:
            facts.append(a)
    return personas, facts


def _contamination_rows(mult: float, age_h: Optional[float], wall,
                        user_chars: Optional[int] = None) -> list:
    """The ``(lr_multiplier, unmask_user, user_loss_weight)`` rows to emit for one chat
    exchange (REBUILD §5e). ``user_loss_weight`` is ``None`` except on a *folded* row.

    Below cap, or with contamination disabled: a single masked row at *mult*. At cap
    (``age_h >= lora_cap_age_h``, so *mult* is the ramp cap = 4) with contamination on,
    Ava's *response* still trains at the full cap while her *voice* entrains on the user's at
    ``contamination_dose``. Two ways to realize that:

    **Split (default, two rows).** Emit a masked row + an unmask-user row that train the
    SAME rendered sequence twice:
      - split mode:    masked ``cap − dose`` (3.0) + unmasked ``dose`` (1.0) → sums to cap;
      - additive mode: masked ``cap`` (4.0) + an extra unmasked ``dose`` (1.0) on top.

    **Folded (``contamination_fold``, one row).** Emit a SINGLE row at LR-mult =
    ``response_total`` (the sum the response would have seen across the pair — ``cap`` in
    split mode, ``cap + dose`` in additive mode) carrying ``user_loss_weight =
    dose / response_total`` on the final user turn. The collator + a weighted ``compute_loss``
    give the response weight 1.0 and the user turn that fraction, so one forward+backward
    reproduces the pair's per-token exposure (response ×response_total, user ×dose) at ~half
    the compute. Not bit-identical (one Adam step / one normalizer vs two) — A/B against split.

    *user_chars*: stripped length of the exchange's final user turn. When it is below
    ``contamination_min_user_chars`` the unmask copy/weight is skipped and the exchange trains
    as a single masked row at the full cap (split-mode LR-neutral: 4.0 either way; additive
    drops the extra dose) — a short turn carries no substantive voice to entrain and would
    only teach Ava to emit terse user-style filler. ``min_user_chars`` 0 disables the gate.
    """
    at_cap = age_h is not None and age_h >= wall.lora_cap_age_h
    dose = wall.contamination_dose
    if not (at_cap and wall.contamination_enabled and dose > 0):
        return [(mult, False, None)]
    min_chars = getattr(wall, "contamination_min_user_chars", 0)
    if min_chars > 0 and (user_chars or 0) < min_chars:
        return [(mult, False, None)]        # short user turn — not worth contaminating on
    response_total = (mult + dose) if wall.contamination_additive else mult
    if getattr(wall, "contamination_fold", False):
        # One weighted row instead of the pair: response at response_total, user at dose.
        weight = (dose / response_total) if response_total > 0 else 0.0
        return [(response_total, True, weight)]
    if wall.contamination_additive:
        return [(mult, False, None), (dose, True, None)]
    return [(max(0.0, mult - dose), False, None), (dose, True, None)]


def build_dataset(*, chats_dirs, ledger, ccfg, built_at: Optional[str] = None,
                  wander_pending: Optional[list] = None,
                  fallback_chats_dir: Optional[Path] = None,
                  include_fresh: bool = False) -> list[BuildRow]:
    """Assemble the full chronological build corpus (REBUILD.md §5a).

    *chats_dirs*: dirs holding frozen bundles (hot + archive — the split no longer
        carries training semantics). Only sidecars stamped ``reflected_at`` (reflect-once,
        Phase 1) contribute; an unreflected chat is not a bundle yet.
    *ledger*: ``ConsolidationLedger`` — read immutably for persona/fact hosts.
    *ccfg*: ``ConsolidationConfig`` (the dialogue curve drives the LR ramp; ``ccfg.wall``
        the wall-clock windows).
    *built_at*: the build's ``as_of`` timestamp (ISO); every row's wall-clock age is
        ``built_at − chat_ts``. Defaults to now, but callers pass an explicit value so a
        build is reproducible (REBUILD §7). A bundle younger than ``rag_only_window_h``
        yields multiplier 0 and is skipped entirely (RAG-only window — no trainable row).
    *wander_pending*: one-shot wander records (kept one-shot for now).
    *include_fresh*: also emit **preview** rows (``BuildRow.preview``, multiplier 0) for
        the chats a build would otherwise leave out of the render entirely — a frozen
        bundle still inside ``rag_only_window_h`` (too young to train), and a
        ``chat_reflected``-only chat (background stage one: verdicts + targets exist,
        the run-level clean-base finish is pending). Their derived targets are thereby
        reviewable in the Training review tab *before* they age into a real build; a
        repair there locks the exchange, which every later pass honors. These rows must
        never train — the caller splits them out before building the optimizer corpus.

    Returns rows sorted oldest-first by (bundle timestamp, exchange index); wander rows
    sort by their own timestamp.
    """
    decay_cfg = ccfg.for_type("dialogue")
    wall = ccfg.wall
    if built_at is None:
        built_at = datetime.now().isoformat()
    # Persona anchors are no longer injected into training CoT (retired — see _inject);
    # only fact anchors keep the host-exchange injection path. Persona reaches the weights
    # implicitly via the original/IDEAL CoT. The persona lineage still lives in the ledger
    # for RAG recall + decay, so this only drops the *build-time* host lookup.
    _personas, facts = _live_hostable_anchors(ledger)
    fact_index = index_by_exchange(facts)

    rows: list[BuildRow] = []
    seen_sessions: set = set()
    for cdir in chats_dirs:
        cdir = Path(cdir)
        if not cdir.exists():
            continue
        sidecar = ChatSidecar(cdir, fallback_chats_dir=fallback_chats_dir)
        for state_path in sorted(cdir.glob("*.state.json")):
            session = session_name_from_sidecar(state_path)
            if session in seen_sessions:        # a session lives in exactly one dir
                continue
            doc = sidecar.load(session)
            frozen = bool(doc.get("reflected_at"))  # only frozen (reflect-once) bundles train
            if not frozen and not (include_fresh and doc.get("chat_reflected")):
                continue
            exchanges = doc.get("exchanges")
            if not isinstance(exchanges, dict):
                continue
            seen_sessions.add(session)
            # Wall-clock age from the CHAT timestamp (session stem), not reflected_at. All
            # exchanges of a chat share its age, so this is computed once per bundle.
            session_ts = Path(session).stem
            age = wall_clock_age_hours(session_ts, built_at)
            mult = lr_multiplier_hours(age, decay_cfg, wall)
            # A chat is trainable only when fully frozen AND past the RAG-only window;
            # otherwise it is a preview row (include_fresh) or skipped outright.
            preview = (not frozen) or mult <= 0.0
            if preview:
                if not include_fresh:           # RAG-only window (too young) or unparseable ts
                    continue                    # -> no trainable row; lives in RAG only
                mult = 0.0
            for key, rec in exchanges.items():
                if not isinstance(rec, dict):
                    continue
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                anchor = build_dialogue_anchor(
                    cdir, session, idx, rec, fallback_chats_dir=fallback_chats_dir)
                if anchor is None:
                    continue
                injected, pkeys, fkeys = _inject(
                    anchor.get("target", ""),
                    fact_index.get((session, idx), []))
                if injected is None:
                    continue
                try:
                    render.assert_parity(anchor, injected)   # train/inference parity guard
                except AssertionError as e:
                    print(f"build_dataset: parity skip {session}#{idx}: {e}", flush=True)
                    continue
                # One row per exchange, except a cap-age exchange splits into a masked +
                # unmasked pair for user contamination (§5e). Both share the same rendered
                # messages/target/anchor; only the LR multiplier + unmask flag differ.
                msgs = render.build_messages(anchor, injected)
                anchor_out = {**anchor, "target": injected}
                if preview:
                    # Render-only: one row, multiplier 0, no contamination pair. The
                    # caller keeps it out of the trained corpus (it exists to be
                    # reviewed/repaired before the chat ages into a real build).
                    rows.append(BuildRow(
                        messages=msgs, age=age if age is not None else 0.0,
                        lr_multiplier=0.0, source="chat",
                        source_session=session, exchange_index=idx,
                        order_key=(session_ts, idx), target=injected,
                        anchor=anchor_out, preview=True,
                        persona_keys=pkeys, fact_keys=fkeys))
                    continue
                user_chars = len((anchor.get("prompt") or "").strip())
                for lr_m, unmask, user_w in _contamination_rows(
                        mult, age, wall, user_chars):
                    rows.append(BuildRow(
                        messages=msgs, age=age if age is not None else 0.0,
                        lr_multiplier=lr_m, source="chat",
                        source_session=session, exchange_index=idx,
                        order_key=(session_ts, idx), target=injected,
                        anchor=anchor_out, unmask_user=unmask,
                        user_loss_weight=user_w,
                        persona_keys=pkeys, fact_keys=fkeys))

    # Wander/news — one-shot (per-run decision), fixed multiplier, sorted by own ts.
    for rec in (wander_pending or []):
        target = (rec.get("target") or "").strip()
        if render.trainable_answer(target) is None:
            continue
        anchor = {
            "key": f"wander:{rec.get('ts', '')}",
            "system_prompt": rec.get("system_prompt", ""),
            "context": [], "prompt": rec.get("prompt", ""), "speaker": "",
        }
        try:
            render.assert_parity(anchor, target)
        except AssertionError:
            continue
        rows.append(BuildRow(
            messages=render.build_messages(anchor, target),
            age=0.0, lr_multiplier=float(WANDER_LR_MULT), source="wander",
            source_session=anchor["key"], exchange_index=None,
            order_key=(str(rec.get("ts") or "~"), 0), target=target, anchor=anchor))

    rows.sort(key=lambda r: r.order_key)
    return rows


def row_render_dict(r: BuildRow) -> dict:
    """The ``sft_render.jsonl`` projection of one BuildRow.

    ONE definition shared by ``train_cycle``'s build and the GPU-free preview
    snapshot (``training/preview_build.py``), so the render schema — the file the
    Training review tab is built on — cannot drift between the two producers.
    ``source_session``/``exchange_index`` travel on each row so a render maps back
    to its origin exchange (the review tab's repair RPCs need them)."""
    ex = {"anchor_key": (r.anchor or {}).get("key") or r.source_session,
          "messages": r.messages, "variant": r.target,
          "age": r.age, "lr_multiplier": r.lr_multiplier, "source": r.source,
          "unmask_user": r.unmask_user, "user_loss_weight": r.user_loss_weight,
          "fact_key": bool(r.fact_keys),
          "source_session": r.source_session,
          "exchange_index": r.exchange_index,
          "target_source": (r.anchor or {}).get("target_source", ""),
          "target_kind": (r.anchor or {}).get("target_kind", ""),
          "target_generation": (r.anchor or {}).get("target_generation", "")}
    if r.preview:
        ex["preview"] = True        # render-only row — never trained (include_fresh)
    return ex


def corpus_fingerprint(rows: list) -> str:
    """A stable hash of a build's row identities + targets (reproducibility, §7)."""
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        # user_loss_weight distinguishes a folded row from the two-row split even at the
        # same (session, exchange, unmask) — so a folded vs unfolded corpus fingerprints apart.
        uw = "" if r.user_loss_weight is None else f"{r.user_loss_weight:.6f}"
        h.update(
            f"{r.source}|{r.source_session}|{r.exchange_index}|{int(r.unmask_user)}|{uw}|"
            .encode("utf-8"))
        h.update((r.target or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
