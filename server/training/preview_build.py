"""GPU-free preview snapshot of the NEXT build's training corpus.

The Training review tab reads the newest ``models/snapshots/*/sft_render.jsonl`` —
a file only ``train_cycle`` produced, at the END of a training cycle. That ordering
is backwards for the operator workflow "Include fresh chats" exists for: reflect a
new batch of chats WITHOUT training (Sleep's training-lite / persona path), review
and repair the derived targets — the fresh ones included — and only then run the
real cycle. With the render minted only by training, there was nothing to review
until the very training the review was meant to precede had already happened.

Corpus assembly is pure and model-free (``build_dataset`` + the shared
``row_render_dict`` projection), so this module lets the *inference process* write
a reviewable render right after a training-lite reflection commits: a snapshot dir
``models/snapshots/preview-<ts>/`` holding ``sft_render.jsonl`` (the full
next-build corpus: would-train rows at their real multipliers, plus the fresh
``preview: true`` rows) and a ``build_meta.json`` whose ``outcome: "preview"``
says out loud that NOTHING in this snapshot has trained. The review tab picks it
up by mtime exactly as it picks up a real build's; the next real ``train_cycle``
writes a newer snapshot and supersedes it.

Preview snapshots are disposable derivations of live state, so only the newest is
kept (older ``preview-*`` dirs are pruned on each write); real ``build-*``
snapshots are never touched. Deliberately import-light — no unsloth/torch — so it
is safe to import inside the inference server with the model loaded.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the inference core importable (same trick as build_dataset / ledger).
_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from training.build_dataset import (          # noqa: E402
    build_dataset, corpus_fingerprint, row_render_dict)
from training.decay import ConsolidationConfig  # noqa: E402
from training.ledger import ConsolidationLedger  # noqa: E402
from training.reflections_path import (       # noqa: E402
    SFT_RENDER_FILE, consolidation_dir, hot_chats_dir, load_server_config)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"   # server/models/
PREVIEW_PREFIX = "preview-"


def _prune_previews(snaps_root: Path, keep: str) -> int:
    """Delete every ``preview-*`` snapshot dir except *keep*; return the count.

    Previews are re-derivable from live state at any time, and each supersedes the
    last, so keeping a lineage of them would only bury the real ``build-*``
    snapshots in noise. Best-effort per dir."""
    pruned = 0
    if not snaps_root.is_dir():
        return 0
    for d in snaps_root.iterdir():
        if d.is_dir() and d.name.startswith(PREVIEW_PREFIX) and d.name != keep:
            try:
                shutil.rmtree(d)
                pruned += 1
            except OSError:
                pass
    return pruned


def build_preview_snapshot(*, run_id: Optional[str] = None,
                           models_dir: Optional[Path] = None,
                           chats_dirs: Optional[list] = None,
                           cons_dir: Optional[Path] = None,
                           config: Optional[dict] = None,
                           wander_pending: Optional[list] = None,
                           built_at: Optional[str] = None) -> dict:
    """Assemble the next build's corpus (fresh chats included) and snapshot it.

    Every parameter defaults to the live box state (the same non-staging resolution
    ``train_cycle`` uses); they are overridable so the selftest can run against a
    fixture tree. *wander_pending* ``None`` loads the live wander corpus — pass
    ``[]`` to exclude it.

    Returns ``{build_id, path, rows, trained_rows, preview_rows, pruned}`` where
    ``trained_rows`` means "would train in the next real build" — nothing here has
    trained. Raises on an unwritable snapshot dir; assembling nothing is not an
    error (an empty render is still an honest answer to "what would the next build
    hold?").
    """
    if config is None:
        config = load_server_config()
    ccfg = ConsolidationConfig.from_dict(config.get("consolidation"))
    if chats_dirs is None:
        chats_dirs = [hot_chats_dir()]
    if cons_dir is None:
        cons_dir = consolidation_dir()
    if models_dir is None:
        models_dir = _MODELS_DIR
    if wander_pending is None:
        from core.wander_sft import load_pending
        wander_pending = load_pending()
    if built_at is None:
        built_at = datetime.now().isoformat()

    ledger = ConsolidationLedger(Path(cons_dir))
    rows = build_dataset(chats_dirs=chats_dirs, ledger=ledger, ccfg=ccfg,
                         built_at=built_at, wander_pending=wander_pending,
                         include_fresh=True)
    trained = [r for r in rows if not r.preview]
    previews = [r for r in rows if r.preview]

    ts = built_at.replace("-", "").replace(":", "").replace("T", "-")[:15]
    build_id = f"{PREVIEW_PREFIX}{ts}"
    snaps_root = Path(models_dir) / "snapshots"
    snap = snaps_root / build_id
    snap.mkdir(parents=True, exist_ok=True)
    # Trained-eligible rows first (their real next-build order), preview rows trailing
    # — the same layout train_cycle writes, so the review tab reads both identically.
    with open(snap / SFT_RENDER_FILE, "w", encoding="utf-8") as fh:
        for r in trained + previews:
            fh.write(json.dumps(row_render_dict(r), ensure_ascii=False) + "\n")
    meta = {
        "build_id": build_id,
        # The load-bearing field: this snapshot trained NOTHING. build_payload
        # forwards it so the review tab can say so next to the build id.
        "outcome": "preview",
        "built_at": built_at, "run_id": run_id,
        "rows": len(rows), "trained_rows": len(trained),
        "preview_rows": len(previews),
        # Fingerprint of the would-train corpus: if the real cycle that follows
        # reports the same value, the operator trained exactly what was reviewed.
        "corpus_fingerprint": corpus_fingerprint(trained),
        "wall_clock": {
            "rag_only_window_h": ccfg.wall.rag_only_window_h,
            "lora_cap_age_h": ccfg.wall.lora_cap_age_h,
        },
    }
    (snap / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pruned = _prune_previews(snaps_root, keep=build_id)
    return {"build_id": build_id, "path": str(snap), "rows": len(rows),
            "trained_rows": len(trained), "preview_rows": len(previews),
            "pruned": pruned}
