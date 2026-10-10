"""Adopt an adapter trained on another box — INCREMENTAL_TRAINING.md §7 (experimental).

An adapter fetched from another Ava (Migrate tab → Fetch adapter) carries a dose ledger
about THAT box's chats. Resumed as-is by an incremental build, it would look as if this
box's whole history had never been trained, and the build would train all of it at full
dose at once (``train_cycle`` now refuses that: ``incremental.min_coverage``). Adoption is
the deliberate way in: it decides what the imported weights are taken to already hold.

    cd server && .venv/bin/python -m training.adopt_adapter \\
        --adapter models/<fetched adapter> --last-chats 50 [--strength 1.0] [--activate]

What it writes (never touching the fetched adapter, which stays the revert path):

* ``models/adapter-<ts>-adopt/`` — a copy of the fetched adapter. With ``--strength s``
  its ``lora_B`` matrices are multiplied by *s*, which scales the foreign weight delta
  (``ΔW ∝ B·A``) exactly; 1.0 copies it unchanged. The weights stay fp32.
* its ``dose_ledger.jsonl``: the imported ledger's entries kept (chats the foreign adapter
  really trained stay delivered — e.g. chats merged between the boxes), every other local
  row marked delivered at its current ``D(age)`` — older history is deliberately never
  trained into these weights — EXCEPT the rows of the newest ``--last-chats`` chats (by
  chat timestamp, among reflected chats), which are written owed (``delivered`` 0), so the
  ledger covers every local row and the build's foreign-adapter guard passes. Wander
  counts as older history. The imported ledger / lineage are kept beside it as
  ``*.foreign.*``.
* its ``lineage.json`` (``mode: adopted``, the foreign parent, N, strength) and a
  ``builds.jsonl`` line (``outcome: adopted``).
* with ``--activate``, ``server_config.json`` ``adapter_id`` → the copy.

The next incremental build (a Sleep run with training on, ``train_mode: incremental``)
then trains those N chats once, at the dose each is owed — mostly the full ``D(cap)``.
Revert: repoint ``adapter_id`` at the previous adapter. ``--dry-run`` prints the plan and
writes nothing. GPU-free; safe with the inference server up (the copy is read at the next
model load). The Migrate tab's **Adopt adapter…** runs this module with the server venv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_SERVER = Path(__file__).resolve().parent.parent
_INFERENCE = _SERVER / "inference"
for _p in (_SERVER, _INFERENCE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import dose_ledger                                   # noqa: E402
from training.decay import TRAIN_LORA_ALPHA, train_mode            # noqa: E402

ADOPT_SUFFIX = "-adopt"
STRENGTH_MAX = 4.0


def plan_adoption(rows: list, foreign_ledger: Optional[dict], last_chats: int, *,
                  built_at: str, build_id: str) -> tuple[dict, dict]:
    """The adopted adapter's ledger and a summary. Pure.

    *rows*: the local trainable ``BuildRow``s (preview rows excluded), each carrying its
    current ``D(age)`` on ``lr_multiplier``. The newest *last_chats* chat sessions (by
    their chat timestamp) are left owed; every other row is marked delivered at ``D(age)``
    unless the foreign ledger already has it (then its entry is kept as it is).
    """
    foreign = dict(foreign_ledger or {})
    sessions = sorted({r.order_key[0]: r.source_session
                       for r in rows if r.source == "chat"}.items())
    owed_list = [sess for _ts, sess in sessions[-last_chats:]] if last_chats > 0 else []
    owed_sessions = set(owed_list)
    ledger = {k: dict(v) for k, v in foreign.items()}
    summary = {"chats": len(sessions), "owed_chats": len(owed_sessions), "rows": len(rows),
               "owed_rows": 0, "owed_dose": 0.0, "kept_foreign": 0, "forgone_rows": 0,
               "foreign_entries": len(foreign),
               "owed_from": owed_list[0] if owed_list else None}
    for r in rows:
        key = dose_ledger.key_of_row(r)
        if key in foreign:
            summary["kept_foreign"] += 1
            if r.source_session in owed_sessions:
                owed = float(r.lr_multiplier) - float(foreign[key].get("delivered") or 0.0)
                if owed > 0:
                    summary["owed_rows"] += 1
                    summary["owed_dose"] += owed
            continue
        if r.source == "chat" and r.source_session in owed_sessions:
            # Written as KNOWN and owed (delivered 0), so the ledger still covers it — the
            # incremental build's foreign-adapter guard counts it as this box's own row.
            summary["owed_rows"] += 1
            summary["owed_dose"] += float(r.lr_multiplier)
            ledger[key] = {"key": key, "delivered": 0.0, "last_trained_at": None,
                           "build_id": build_id, "target_sha": dose_ledger.target_sha(r.target),
                           "adopted_owed": True}
            continue
        summary["forgone_rows"] += 1
        ledger[key] = {"key": key, "delivered": round(float(r.lr_multiplier), 6),
                       "last_trained_at": built_at, "build_id": build_id,
                       "target_sha": dose_ledger.target_sha(r.target), "adopted": True}
    summary["owed_dose"] = round(summary["owed_dose"], 3)
    return ledger, summary


def scale_lora_b(src_file: Path, dst_file: Path, strength: float) -> int:
    """Write *src_file* to *dst_file* with every ``lora_B`` tensor multiplied by *strength*.

    Returns the number of tensors scaled. CPU only; dtypes are kept (fp32 stays fp32)."""
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    with safe_open(str(src_file), framework="pt") as fh:
        metadata = fh.metadata() or {}
    state = load_file(str(src_file))
    scaled = 0
    for name in list(state):
        if "lora_B" in name:
            state[name] = (state[name] * strength).to(state[name].dtype).contiguous()
            scaled += 1
    save_file(state, str(dst_file), metadata=metadata or {"format": "pt"})
    return scaled


def write_adopted(src_dir: Path, dst_dir: Path, *, ledger: dict, lineage: dict,
                  strength: float) -> dict:
    """Copy *src_dir* to *dst_dir*, scale it, keep the foreign ledger/lineage as provenance,
    and write the adopted ledger + lineage. Refuses an existing *dst_dir*."""
    if dst_dir.exists():
        raise FileExistsError(f"{dst_dir} already exists")
    shutil.copytree(src_dir, dst_dir)
    try:
        for name, foreign_name in ((dose_ledger.LEDGER_FILE, "dose_ledger.foreign.jsonl"),
                                   (dose_ledger.LINEAGE_FILE, "lineage.foreign.json")):
            if (dst_dir / name).is_file():
                (dst_dir / name).rename(dst_dir / foreign_name)
        scaled = 0
        weights = dst_dir / "adapter_model.safetensors"
        if abs(strength - 1.0) > 1e-9:
            tmp = dst_dir / "adapter_model.scaled.tmp"
            scaled = scale_lora_b(src_dir / "adapter_model.safetensors", tmp, strength)
            tmp.replace(weights)
        dose_ledger.save(dst_dir, ledger)
        dose_ledger.save_lineage(dst_dir, lineage)
        return {"scaled_tensors": scaled,
                "dtypes": dose_ledger.safetensors_dtypes(weights)}
    except Exception:
        shutil.rmtree(dst_dir, ignore_errors=True)
        raise


def _resolve_adapter(arg: str, models_dir: Path) -> Path:
    p = Path(arg)
    if not p.is_absolute() and not p.exists():
        p = models_dir / arg
    return p.resolve()


def adopt(*, adapter: str, last_chats: int, strength: float = 1.0, activate: bool = False,
          dry_run: bool = False) -> dict:
    """Plan (and unless *dry_run*, write) the adoption. Returns a JSON-able summary."""
    from core.chat_sidecar import ChatSidecar  # noqa: F401  (import check: inference on path)
    from core.wander_sft import load_pending as load_wander_pending
    from training.build_dataset import build_dataset
    from training.build_history import BuildHistory
    from training.decay import ConsolidationConfig
    from training.ledger import ConsolidationLedger
    from training.reflections_path import (consolidation_dir, hot_chats_dir,
                                           load_server_config, save_server_config)

    models_dir = _SERVER / "models"
    config = load_server_config()
    model_id = str(config.get("model_id") or "")
    src = _resolve_adapter(adapter, models_dir)
    if not (src / "adapter_model.safetensors").is_file():
        raise SystemExit(f"not an adapter dir (no adapter_model.safetensors): {src}")
    if last_chats < 1:
        raise SystemExit("--last-chats must be at least 1")
    if not (0.0 < strength <= STRENGTH_MAX):
        raise SystemExit(f"--strength must be in (0, {STRENGTH_MAX}]")
    adapter_cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    lora_r = int(adapter_cfg.get("r") or 0)
    reasons = dose_ledger.lineage_mismatch(src, model_id=model_id, lora_r=lora_r,
                                           lora_alpha=TRAIN_LORA_ALPHA, use_rslora=True)
    local_r = int(config.get("lora_r") or lora_r)
    if local_r != lora_r:
        reasons.append(f"this box's lora_r is {local_r}, the adapter's {lora_r} — set "
                       f"server_config.json lora_r to {lora_r} to continue it")
    if reasons:
        raise SystemExit("cannot adopt: " + "; ".join(reasons))

    built_at = datetime.now().isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_id = f"adopt-{stamp}"
    ccfg = ConsolidationConfig.from_dict(config.get("consolidation"))
    rows = build_dataset(chats_dirs=[hot_chats_dir()],
                         ledger=ConsolidationLedger(consolidation_dir()), ccfg=ccfg,
                         built_at=built_at, wander_pending=load_wander_pending())
    rows = [r for r in rows if not r.preview]
    foreign = dose_ledger.load(src)
    ledger, summary = plan_adoption(rows, foreign, last_chats, built_at=built_at,
                                    build_id=build_id)
    dst = models_dir / f"adapter-{stamp}{ADOPT_SUFFIX}"
    mode_now = train_mode(config)
    summary.update({"source": str(src), "target": str(dst), "strength": strength,
                    "last_chats": last_chats, "lora_r": lora_r, "dry_run": dry_run,
                    "train_mode": mode_now, "activated": False})
    if dry_run:
        return summary

    lineage = {"mode": "adopted", "model_id": model_id, "lora_r": lora_r,
               "lora_alpha": TRAIN_LORA_ALPHA, "use_rslora": True,
               "parent_adapter": str(src), "foreign": True, "last_chats": last_chats,
               "strength": strength, "build_id": build_id, "built_at": built_at}
    summary.update(write_adopted(src, dst, ledger=ledger, lineage=lineage,
                                 strength=strength))
    BuildHistory(models_dir).append(
        outcome="adopted", rows=summary["owed_rows"], base_lr=0.0, build_id=build_id,
        built_at=built_at, adapter_dir=str(dst), mode="adopted", parent_adapter=str(src),
        probe_summary={})
    if activate:
        summary["previous_adapter"] = config.get("adapter_id")
        config["adapter_id"] = str(dst)
        save_server_config(config)
        summary["activated"] = True
    return summary


def format_summary(s: dict) -> str:
    lines = [
        ("DRY RUN — nothing written." if s.get("dry_run") else "Adopted."),
        f"  source:   {s['source']}",
        f"  target:   {s['target']}",
        f"  strength: {s['strength']}  (lora_B x{s['strength']}; r={s['lora_r']})",
        f"  chats:    {s['owed_chats']} newest of {s['chats']} reflected chats left owed"
        + (f" (from {s['owed_from']})" if s.get("owed_from") else ""),
        f"  rows:     {s['owed_rows']} owed (dose {s['owed_dose']}), {s['forgone_rows']} "
        f"marked delivered without training, {s['kept_foreign']} already in the imported "
        f"ledger ({s['foreign_entries']} foreign entries kept)",
    ]
    if s.get("dtypes"):
        lines.append(f"  weights:  {s['dtypes']}")
    if s.get("activated"):
        lines.append(f"  adapter_id -> {s['target']} (was {s.get('previous_adapter')})")
    elif not s.get("dry_run"):
        lines.append("  not activated: point server_config.json adapter_id at the target "
                     "to use it.")
    if s.get("train_mode") != "incremental":
        lines.append("  NOTE: train_mode is not 'incremental' — the next training Sleep would "
                     "be a full train and discard the adoption. Set train_mode: incremental.")
    else:
        lines.append("  Next: a Sleep run with training on trains the owed rows once "
                     "(skipped while they total < incremental.min_increment tokens).")
    lines.append("  Revert: repoint adapter_id at the previous adapter.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True,
                    help="the imported adapter dir (absolute, or a name under models/)")
    ap.add_argument("--last-chats", type=int, default=50,
                    help="how many of the newest local chats the next incremental build "
                         "trains onto it (default 50)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help=f"multiply the imported lora_B by this (0, {STRENGTH_MAX}]; "
                         f"1.0 = unchanged (default)")
    ap.add_argument("--activate", action="store_true",
                    help="point server_config.json adapter_id at the adopted copy")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args()
    summary = adopt(adapter=args.adapter, last_chats=args.last_chats,
                    strength=args.strength, activate=args.activate, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json
          else format_summary(summary))


if __name__ == "__main__":
    main()
