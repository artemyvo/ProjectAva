"""One-shot migrations for consolidation state.

0. Relocate a pre-``data/`` flat layout (``inference/chats/`` + ``inference/
   reflections/``) into the lifetime-organized ``inference/data/`` tree.
1. Seed ``consolidation_anchors.jsonl`` from legacy reflection artifacts
   (``sft_pairs.jsonl`` dialogue rows, ``weights_persona.jsonl`` facts/persona).
2. Fold ledger **dialogue** anchors into per-chat sidecars (``chats/*.state.json``).
   Fact/persona anchors stay in the ledger.

All steps are **non-destructive** and idempotent.

    python -m training.migrate [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from core.chat_sidecar import ChatSidecar  # noqa: E402

from training.ledger import ConsolidationLedger, dialogue_key, fact_key
from training.reflections_path import (
    consolidation_dir,
    hot_chats_dir,
    memory_dir,
    scratch_dir,
)

SFT_PAIRS_FILE = "sft_pairs.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# --------------------------------------------------------------------------- #
# Step 0 — relocate the old flat layout into data/
# --------------------------------------------------------------------------- #

def relocate_legacy_layout(*, dry_run: bool = False) -> dict:
    """Move ``inference/chats`` + ``inference/reflections`` files into ``data/``.

    Routes each artifact to its new home by lifetime. Non-destructive: a file
    whose destination already exists is skipped, so re-running is safe.
    """
    old_chats = _INFERENCE / "chats"
    old_refl = _INFERENCE / "reflections"
    # filename -> new destination directory
    refl_routing = {
        "rag_memory.jsonl": memory_dir(),
        "weights_persona.jsonl": memory_dir(),
        SFT_PAIRS_FILE: consolidation_dir(),
        "consolidation_anchors.jsonl": consolidation_dir(),
        "sft_render.jsonl": scratch_dir(),
    }

    moves: list[tuple[Path, Path]] = []
    if old_chats.is_dir():
        for p in sorted(old_chats.glob("*.json")):  # transcripts + .state.json sidecars
            moves.append((p, hot_chats_dir() / p.name))
    if old_refl.is_dir():
        for name, dst_dir in refl_routing.items():
            src = old_refl / name
            if src.exists():
                moves.append((src, dst_dir / name))

    counts = {"moved": 0, "skipped": 0, "candidates": len(moves)}
    for src, dst in moves:
        if dst.exists():
            counts["skipped"] += 1
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        counts["moved"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Step 1 — legacy artifacts → ledger
# --------------------------------------------------------------------------- #

def migrate_artifacts_to_ledger(memory_path: Path, consolidation_path: Path,
                                *, dry_run: bool = False) -> dict:
    """Legacy step: sft_pairs (consolidation/) + weights_persona (memory/) → ledger."""
    ledger = ConsolidationLedger(consolidation_path)
    existing = set(ledger.fold().keys())
    counts = {"dialogue": 0, "fact": 0, "persona": 0, "skipped": 0}

    for row in _read_jsonl(consolidation_path / SFT_PAIRS_FILE):
        if row.get("target_source") == "revised_missing_ideal":
            counts["skipped"] += 1
            continue
        key = dialogue_key(row.get("source_session", ""), row.get("exchange_index"))
        if key in existing:
            counts["skipped"] += 1
            continue
        if not dry_run:
            ledger.register_dialogue(row)
        existing.add(key)
        counts["dialogue"] += 1

    for row in _read_jsonl(memory_path / "weights_persona.jsonl"):
        content = (row.get("content") or "").strip()
        if not content:
            counts["skipped"] += 1
            continue
        key = fact_key(content)
        if key in existing:
            counts["skipped"] += 1
            continue
        item_type = "persona" if row.get("weights_kind") == "persona" else "fact"
        if not dry_run:
            ledger.register_fact(
                content=content,
                item_type=item_type,
                trigger=row.get("trigger"),
                source_session=row.get("source_session", ""),
                lang=row.get("lang"),
            )
        existing.add(key)
        counts[item_type] += 1

    return counts


# --------------------------------------------------------------------------- #
# Step 2 — ledger dialogue → sidecars
# --------------------------------------------------------------------------- #

def migrate_ledger_dialogue_to_sidecars(
    consolidation_path: Path,
    chats_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Fold ledger dialogue anchors into ``chats/<session>.state.json`` sidecars."""
    ledger = ConsolidationLedger(consolidation_path)
    sidecar = ChatSidecar(chats_dir)
    counts = {"imported": 0, "skipped": 0}

    for rec in ledger.fold().values():
        if rec.get("type") != "dialogue":
            continue
        source_session = (rec.get("source_session") or "").strip()
        exchange_index = rec.get("exchange_index")
        target = (rec.get("target") or "").strip()
        if not source_session or exchange_index is None or not target:
            counts["skipped"] += 1
            continue
        try:
            exchange_index = int(exchange_index)
        except (TypeError, ValueError):
            counts["skipped"] += 1
            continue

        if sidecar.get_exchange(source_session, exchange_index) is not None:
            counts["skipped"] += 1
            continue

        stage = rec.get("stage", 0)
        last_trained = datetime.now().isoformat() if stage > 0 else None
        if dry_run:
            counts["imported"] += 1
            continue

        if sidecar.import_exchange(
            source_session=source_session,
            exchange_index=exchange_index,
            stage=stage,
            verdict=str(rec.get("verdict") or ""),
            target=target,
            run_id="ledger-migrate",
            last_trained=last_trained,
        ):
            counts["imported"] += 1
        else:
            counts["skipped"] += 1

    return counts


def migrate(
    memory_path: Path,
    consolidation_path: Path,
    chats_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    ledger_counts = migrate_artifacts_to_ledger(
        memory_path, consolidation_path, dry_run=dry_run,
    )
    sidecar_counts = migrate_ledger_dialogue_to_sidecars(
        consolidation_path, chats_dir, dry_run=dry_run,
    )
    return {"ledger": ledger_counts, "sidecar": sidecar_counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()
    verb = "would" if args.dry_run else ""

    reloc = relocate_legacy_layout(dry_run=args.dry_run)
    print(f"{verb} relocate old flat layout: {reloc['moved']} file(s) moved into data/ "
          f"(skipped {reloc['skipped']} of {reloc['candidates']})")

    counts = migrate(memory_dir(), consolidation_dir(), hot_chats_dir(), dry_run=args.dry_run)
    lc = counts["ledger"]
    sc = counts["sidecar"]
    print(f"{verb} register to ledger: {lc['dialogue']} dialogue, {lc['fact']} fact, "
          f"{lc['persona']} persona  (skipped {lc['skipped']})")
    print(f"{verb} import to sidecars: {sc['imported']} dialogue exchanges "
          f"(skipped {sc['skipped']})")
    print(f"ledger: {ConsolidationLedger(consolidation_dir()).path}")
    print(f"chats:  {hot_chats_dir()}")


if __name__ == "__main__":
    main()
