"""Consolidation anchor ledger — the durable state of what is being consolidated.

``sft_render.jsonl`` is rebuilt and deleted each training cycle; revision records
live in per-chat sidecar files. The *vetted* content a variant is regenerated from
cannot live in either disposable file — it lives here, in an
append-only op-log folded the same way as ``reflection_memory.py``:

    register  {key, type, ts, ...anchor payload}  — a new anchor enters at stage 0
    advance   {key, ts}                           — that anchor's stage += 1
    evict     {key, ts}                            — remove an anchor from live state
    supersede {key, ts, reason}                    — *soften*: keep the anchor folded but
                                                     flag it (dropped from live/training +
                                                     active persona evidence, retained as
                                                     evidence-of-change). Reconciliation's
                                                     reversible, non-destructive move.

Folded, this yields ``{key: anchor + stage}`` (a superseded anchor carries
``superseded: True``). A re-``register`` of an existing key
refreshes the payload but never resets the stage, so re-encountering an item does
not undo its consolidation progress.

Keys are stable per item:
  * dialogue  -> content_key("{source_session}#{exchange_index}")
  * fact/persona -> content_key(content)   (matches the rag_memory key, so a fact's
    stage here lines up with its RAG record)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# Reuse the inference layer's key derivation so fact keys line up with rag_memory.
_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))
from core.reflection_writer import content_key  # noqa: E402

from training.decay import ConsolidationConfig  # noqa: E402


LEDGER_FILE = "consolidation_anchors.jsonl"


def dialogue_key(source_session: str, exchange_index) -> str:
    return content_key(f"{source_session}#{exchange_index}")


def fact_key(content: str) -> str:
    return content_key(content)


class ConsolidationLedger:
    """Append-only anchor store, folded to live anchors with their current stage."""

    def __init__(self, consolidation_dir: Path) -> None:
        # Lives in data/hot/consolidation/.
        self.path = Path(consolidation_dir) / LEDGER_FILE

    # -- writes ------------------------------------------------------------ #

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def register(self, *, key: str, item_type: str, payload: dict) -> None:
        """Register (or refresh) an anchor. Stage is preserved across re-registers."""
        if not key:
            return
        self._append({
            "op": "register",
            "ts": datetime.now().isoformat(),
            "key": key,
            "type": item_type,
            **payload,
        })

    def register_dialogue(self, anchor: dict) -> str:
        """Register a dialogue anchor from a revision SFT record. Returns its key."""
        key = dialogue_key(anchor.get("source_session", ""), anchor.get("exchange_index"))
        self.register(key=key, item_type="dialogue", payload={
            "system_prompt": anchor.get("system_prompt", ""),
            "context": anchor.get("context", []) or [],
            "prompt": anchor.get("prompt", "") or anchor.get("user_prompt", ""),
            "speaker": anchor.get("speaker", ""),
            "target": anchor.get("target", ""),
            "verdict": anchor.get("verdict"),
            "source_session": anchor.get("source_session", ""),
            "exchange_index": anchor.get("exchange_index"),
            "lang": anchor.get("lang"),
        })
        return key

    def register_fact(self, *, content: str, item_type: str = "fact",
                      trigger: Optional[str] = None, source_session: str = "",
                      lang: Optional[str] = None,
                      exchange_index: Optional[int] = None,
                      source_exchange: Optional[dict] = None,
                      about: Optional[str] = None) -> str:
        """Register a fact/persona anchor. Returns its key (== rag_memory key).

        *source_exchange* (persona only) is the resolved revision anchor the statement
        was distilled from — system_prompt/context/prompt/speaker/target — snapshotted
        here so ``persona_render`` can inject the statement into that exchange's CoT
        without racing the source chat's hot→archive move. It is preserved across a
        re-register that omits it (see :meth:`fold`).

        *about* (facts only) is the person the fact concerns, carried through from
        consolidation so the anchor keeps its subject. Only ``self``/``observed`` facts
        ever reach this ledger — hearsay is held back at the writer (see
        ``ReflectionWriter.write_consolidation``) — so an anchor here is always
        something Ava may claim to know.
        """
        payload = {
            "content": content,
            "trigger": trigger,
            "source_session": source_session,
            "lang": lang,
        }
        if about:
            payload["about"] = about
        if exchange_index is not None:
            payload["exchange_index"] = exchange_index
        if source_exchange:
            payload["source_exchange"] = source_exchange
        key = fact_key(content)
        self.register(key=key, item_type=item_type, payload=payload)
        return key

    def advance(self, keys: Iterable[str]) -> int:
        """Bump the stage of each key by one (call after a successful train+merge)."""
        n = 0
        ts = datetime.now().isoformat()
        for key in keys:
            if not key:
                continue
            self._append({"op": "advance", "ts": ts, "key": key})
            n += 1
        return n

    def evict(self, keys: Iterable[str]) -> int:
        """Remove anchors from the folded live set using append-only tombstones."""
        n = 0
        ts = datetime.now().isoformat()
        for key in keys:
            if not key:
                continue
            self._append({"op": "evict", "ts": ts, "key": key})
            n += 1
        return n

    def supersede(self, keys: Iterable[str], *, reason: str = "") -> int:
        """*Soften* anchors — the reconciliation "keep as evidence-of-change" move.

        Unlike :meth:`evict` (which drops the anchor from the fold entirely), a
        ``supersede`` op keeps the anchor in ``fold()`` but flags it ``superseded``:
        it is excluded from ``live_anchors`` (so training no longer reinforces an
        outgrown persona / stale fact) and from the digest's *active* evidence
        (``reflection_digest.gather_persona_raw``), yet stays folded so a future
        evidence-of-change / ``LINES`` read can still see the arc. Append-only, so a
        reconciliation pass is reverted by dropping the lines it wrote. A later
        ``register`` of the same key reactivates it (clears the flag)."""
        n = 0
        ts = datetime.now().isoformat()
        for key in keys:
            if not key:
                continue
            self._append({"op": "supersede", "ts": ts, "key": key, "reason": reason})
            n += 1
        return n

    def counter(self, keys: Iterable[str], *, source_session: str = "",
                reason: str = "", run_id: str = "") -> int:
        """Append COUNTER-EVIDENCE against live persona items — a session pushed AGAINST
        each stance (the *persuasion* channel, symmetric to a ``register``'s affirmation).

        Unlike :meth:`supersede` (one hard soften that drops the anchor from active
        evidence + training), a counter is GRADED and ADDITIVE: it never removes the
        anchor, it adds one dated distinct-session unit of *negative* pressure that
        ``reflection_digest`` nets out of the theme's weighted recurrence at gain
        ``_PERSUASION_GAIN``. Persuasion therefore requires SUSTAINED pushes — a single
        counter barely dents a mature trait; enough of them drive it under the evaporation
        floor, where it fades from the portrait on its own (no explicit supersede needed).
        Append-only ⇒ reversible and re-foldable; a stale counter decays by recency like an
        affirmation. NOT surfaced by :meth:`fold` — it is consumed by
        ``reflection_digest.gather_persona_raw``'s raw op scan, exactly like ``register``
        ops are, not by the live-anchor fold (a countered-but-not-yet-faded trait stays
        live and recallable).

        The reaction→key bridge (which live keys a reaction pushes against) and the
        pushback-detection policy that *emit* these ops are the next layer, not here."""
        n = 0
        ts = datetime.now().isoformat()
        for key in keys:
            if not key:
                continue
            self._append({"op": "counter", "ts": ts, "key": key, "type": "persona",
                          "source_session": source_session, "reason": reason,
                          "run_id": run_id})
            n += 1
        return n

    def note_fact_trained(self, key: str, copies: int) -> None:
        """Record that a [fact] anchor was injected into *copies* trained CoT copies.

        Facts are trained the CoT-injection way (like persona) but, unlike persona, do NOT
        advance a decay *stage* — they ride whatever host exchange the placement judge
        assigned and are capped by a **cumulative injected-copy count** instead (see
        ``train_cycle.FACT_TRAIN_CAP``). ``fold`` sums these into ``train_count`` so the
        renderer can stop injecting a fact once it has had its share of weight exposure.
        Append-only, so later from-scratch builds observe the same count.
        """
        if not key or copies <= 0:
            return
        self._append({"op": "fact_trained", "ts": datetime.now().isoformat(),
                      "key": key, "n": int(copies)})

    # -- reads ------------------------------------------------------------- #

    def fold(self) -> dict[str, dict]:
        """Replay the op-log into ``{key: anchor record with ``stage`` attached}``."""
        anchors: dict[str, dict] = {}
        stages: dict[str, int] = {}
        train_counts: dict[str, int] = {}
        superseded: dict[str, dict] = {}
        if not self.path.exists():
            return anchors
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return anchors
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("key")
            if not key:
                continue
            op = rec.get("op")
            if op == "register":
                # Refresh payload but keep accrued stage. Preserve a persona's
                # source_exchange snapshot if an old/looser re-register drops it (e.g. a
                # consolidation-routed persona that carries no exchange snapshot).
                prev = anchors.get(key)
                if prev and prev.get("source_exchange") and not rec.get("source_exchange"):
                    rec["source_exchange"] = prev["source_exchange"]
                anchors[key] = rec
                stages.setdefault(key, 0)
                superseded.pop(key, None)   # re-registering reactivates a softened anchor
            elif op == "advance":
                stages[key] = stages.get(key, 0) + 1
            elif op == "fact_trained":
                train_counts[key] = train_counts.get(key, 0) + int(rec.get("n", 0) or 0)
            elif op == "supersede":
                superseded[key] = rec       # keep the anchor, flag it (evidence-of-change)
            elif op == "evict":
                anchors.pop(key, None)
                stages.pop(key, None)
                train_counts.pop(key, None)
                superseded.pop(key, None)
        for key, rec in anchors.items():
            rec["stage"] = stages.get(key, 0)
            rec["train_count"] = train_counts.get(key, 0)
            if key in superseded:
                rec["superseded"] = True
                rec["superseded_reason"] = superseded[key].get("reason", "")
                rec["superseded_ts"] = superseded[key].get("ts", "")
        return anchors

    def live_anchors(self, config: ConsolidationConfig) -> list[dict]:
        """Anchors still earning variants (not yet deprecated) under *config*."""
        out = []
        for rec in self.fold().values():
            if rec.get("superseded"):
                continue   # softened by reconciliation — no longer earns training
            cfg = config.for_type(rec.get("type", ""))
            if cfg is None:
                continue
            if not cfg.is_deprecated(rec.get("stage", 0)):
                out.append(rec)
        return out

    def live_facts(self, config: ConsolidationConfig) -> list[dict]:
        """Live ``[fact]`` anchors (``type == "fact"``, not deprecated) — the universe the
        fact-placement judge assigns hosts to and the renderer injects into CoT.

        Facts are gated on the cumulative injected-copy ``train_count`` (see
        ``train_cycle.FACT_TRAIN_CAP``), not a decay stage: they never call ``advance`` so
        their stage stays 0 and the deprecation filter is a no-op for them — the renderer's
        ``train_count`` check is what retires a fully-baked fact.
        """
        return [a for a in self.live_anchors(config) if a.get("type") == "fact"]

    def stage_of(self, key: str) -> Optional[int]:
        """Current stage of *key*, or None if unknown."""
        rec = self.fold().get(key)
        return None if rec is None else rec.get("stage", 0)
