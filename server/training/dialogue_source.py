"""Build trainable dialogue anchors from chat transcripts + sidecar state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from core.chat_sidecar import ChatSidecar, session_name_from_sidecar  # noqa: E402
# Single source of truth for assembling <think>{cot}</think>{answer}: reuse the ShareML
# builder so the trained target is byte-identical to the durable record it mirrors.
from core.reflection_shareml import _verbatim_assistant  # noqa: E402

from training.decay import ConsolidationConfig
from training.ledger import dialogue_key


def _load_chat(
    chats_dir: Path,
    source_session: str,
    fallback_chats_dir: Optional[Path] = None,
) -> Optional[dict]:
    name = (source_session or "").strip()
    if not name or "/" in name or "\\" in name:
        return None
    # The staging workspace holds sidecars but not the raw transcripts (those stay
    # in hot/chats), so fall back to the live dir when the transcript is missing.
    for base in (chats_dir, fallback_chats_dir):
        if base is None:
            continue
        path = (base / name).resolve()
        if path.parent != Path(base).resolve() or path.suffix.lower() != ".json":
            continue
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def build_dialogue_anchor(
    chats_dir: Path,
    source_session: str,
    exchange_index: int,
    sidecar_rec: dict,
    fallback_chats_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Assemble a regeneration anchor from the chat JSON and one sidecar entry."""
    # Banned from training by the operator (Training review tab → Ban). The exchange stays
    # in the transcript and in RAG — it happened, and she may recall it — it simply never
    # becomes a trainable row again. Checked before anything else: the point of a ban is
    # that no reading of this exchange, however repaired-looking, is worth learning from.
    if sidecar_rec.get("banned"):
        return None
    chat = _load_chat(chats_dir, source_session, fallback_chats_dir=fallback_chats_dir)
    if chat is None:
        return None

    exchanges = chat.get("exchanges") or []
    if exchange_index < 0 or exchange_index >= len(exchanges):
        return None

    # An Ava-initiated session's exchange 0 (outreach / synthesis / check-in) is her own
    # opener under a stage-direction speaker, with a synthetic impulse as its
    # "user_prompt" (no real user stimulus). Training it would teach the model to emit
    # openers unprompted, so it is not a trainable anchor — it lives in the transcript +
    # RAG (recallable) and still forms the *context* for exchange 1. The parsing contract
    # is the session-level ``initiated_by`` flag those reach-out subsystems write.
    if str(chat.get("initiated_by") or "").strip() == "ava" and exchange_index == 0:
        return None

    exc = exchanges[exchange_index]
    target = (sidecar_rec.get("target") or "").strip()
    if not target:
        return None
    target_source = (sidecar_rec.get("target_source") or "").strip()

    # Corruption flags set from the Training review tab (chat_logger.mark_exchange_corrupt).
    # A frozen target is trustworthy only when reflection RE-GENERATED it (target_source
    # "revised" — a clean chat re-answer or branch win, authored fresh under current
    # weights); a
    # keep/original target rests on the exchange's own stored CoT + reply.
    corrupt_cot = bool(exc.get("corrupt_cot"))
    corrupt_response = bool(exc.get("corrupt_response"))
    # Corrupt reply + no re-generated replacement -> the target IS the corrupt answer.
    # Drop it (a later revisit re-derives an IDEAL, which lands as target_source "revised"
    # and trains normally). Better "not trained" than "trained on garbage".
    if corrupt_response and target_source != "revised":
        return None
    # Assemble the trainable assistant turn as <think>{cot}</think>{answer}, but only
    # ever pair an answer with a CoT that actually produced it (a mismatched thought is
    # what erodes reasoning over cycles):
    #   * target already carries a closed <think> block -> a faithful CoT was resolved
    #     upstream (`keep`/`original` keeps its own; a branch win reattaches the shared
    #     <think> prefix it was generated from; an IDEAL re-answer carries the thought
    #     the normal dialogue generation authored for it — see resolve_revision_target).
    #     Verbatim.
    #   * answer-only + a `revised` source -> a legacy answer-only IDEAL (pre clean-reanswer
    #     sidecars): free judge-written text with no faithful CoT. Train it answer-only
    #     (the gemma renderer emits the documented empty thinking channel) rather than
    #     injecting the original thought, which led to a *different* answer. New clean
    #     re-answer targets always carry a <think> and take the verbatim branch above.
    #   * answer-only, any other source (incl. legacy sidecars) -> reattach the captured
    #     chat-time CoT, the faithful reasoning behind the original reply.
    if "<think>" in target and "</think>" in target:
        # Already a complete, faithful target. If the operator flagged the SOURCE CoT
        # corrupt and this target rests on it (keep/original, not a re-generated
        # "revised"), strip the corrupt CoT so training never sees it — the answer trains
        # CoT-less (gemma's empty-channel scaffold). A "revised" target's <think> is a
        # freshly authored CoT, not the corrupt source, so it is kept — this includes a
        # CoT-regen graft (target_generation "chat_recot_v1"): the reply is the original,
        # but its <think> was regenerated by re-answering (reflection_runner approach #3),
        # so it is faithful and must NOT be stripped.
        if corrupt_cot and target_source != "revised":
            target = _verbatim_assistant("", target)
    elif target_source == "revised":
        pass  # IDEAL — no faithful CoT; keep answer-only
    elif corrupt_cot:
        pass  # answer-only; do NOT reattach the corrupt source CoT (treat it as missing)
    else:
        target = _verbatim_assistant(exc.get("assistant_cot"), target)

    session_user = (chat.get("user") or "").strip()
    context: list[dict] = []
    for prior in exchanges[:exchange_index]:
        context.append({
            "role": "user",
            "content": prior.get("user_prompt", "") or "",
            "speaker": (prior.get("speaker") or "").strip() or session_user,
        })
        context.append({
            "role": "assistant",
            "content": prior.get("assistant_response", "") or "",
        })

    speaker = (exc.get("speaker") or "").strip() or session_user
    try:
        stage = max(0, int(sidecar_rec.get("stage", 0)))
    except (TypeError, ValueError):
        stage = 0

    # Persona-conditioned IDEAL: the revise/ideal-win target was generated with the
    # exchange's relevant self-knowledge appended to the system prompt (retired the
    # build-time persona prepend — see build_dataset._inject). Re-append the SAME persisted
    # block so render.build_messages / assert_parity reconstruct the identical system
    # message the IDEAL was generated against (parity holds by construction). Empty for
    # keep/branch/manual targets (their CoT is the original, not this seam).
    base_system = chat.get("system_prompt", "") or ""
    persona_context = (sidecar_rec.get("persona_context") or "").strip()
    if persona_context:
        base_system = f"{base_system}\n\n{persona_context}" if base_system else persona_context

    return {
        "key": dialogue_key(source_session, exchange_index),
        "type": "dialogue",
        "system_prompt": base_system,
        "context": context,
        "prompt": exc.get("user_prompt", "") or "",
        "speaker": speaker,
        "target": target,
        "target_source": target_source,
        "target_kind": sidecar_rec.get("target_kind") or "",
        "target_generation": sidecar_rec.get("target_generation") or "",
        "verdict": sidecar_rec.get("verdict"),
        "source_session": source_session,
        "exchange_index": exchange_index,
        "stage": stage,
    }


def live_dialogue_anchors(
    chats_dir: Path,
    sidecar: ChatSidecar,
    config: ConsolidationConfig,
) -> list[dict]:
    """Non-deprecated vetted exchanges ready for anchored regeneration."""
    decay = config.for_type("dialogue")
    if decay is None:
        return []

    # Transcripts may live in the sidecar's fallback dir (hot) when reflecting off
    # the staging workspace, which carries sidecars but not raw transcripts.
    fallback_chats_dir = getattr(sidecar, "fallback_chats_dir", None)
    anchors: list[dict] = []
    for state_path in sorted(chats_dir.glob("*.state.json")):
        source_session = session_name_from_sidecar(state_path)
        doc = sidecar.load(source_session)
        exchanges = doc.get("exchanges")
        if not isinstance(exchanges, dict):
            continue
        for key, rec in exchanges.items():
            if not isinstance(rec, dict):
                continue
            try:
                exchange_index = int(key)
            except (TypeError, ValueError):
                continue
            try:
                stage = max(0, int(rec.get("stage", 0)))
            except (TypeError, ValueError):
                stage = 0
            if decay.is_deprecated(stage):
                continue
            anchor = build_dialogue_anchor(
                chats_dir, source_session, exchange_index, rec,
                fallback_chats_dir=fallback_chats_dir,
            )
            if anchor is not None:
                anchors.append(anchor)
    return anchors
