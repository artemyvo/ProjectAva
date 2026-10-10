"""Dose ledger — the incremental build's running total (INCREMENTAL_TRAINING.md §2).

Under ``train_mode: incremental`` a build resumes the serving adapter instead of fitting a
fresh LoRA, so it must know how much of each row's dose the adapter already holds. The
wall-clock ramp ``D(age)`` (``decay.lr_multiplier_hours``, as stamped on each
``BuildRow.lr_multiplier`` by ``build_dataset``) is read as the **cumulative** dose a row
should have received by its age; an incremental build trains each row by the increment

    increment(row) = D(age_now) − delivered(row)

and the ledger records ``delivered``. The total a row receives is then ``D(cap)`` however
many builds fell inside its window, a row that missed its window is trained once with
its remaining dose, and a row past cap (increment 0) leaves training and is free to fade.

The ledger lives **inside the adapter directory** (``dose_ledger.jsonl``) beside
``adapter_model.safetensors``: repointing ``adapter_id`` at an older adapter also points
at the ledger that matches its weights, and the ledger travels with the adapter. Every
build writes one — a full train a fresh ledger, an incremental build the parent's copied
forward plus its increments. The one exception is the transition: an adapter that
predates the ledger gets one seeded once from its build snapshot's render
(``seed_from_render``).

``lineage.json`` beside it records what an incremental step must match (model, rank,
alpha, rsLoRA) and the parent pointer. Pure / GPU-free: self-tested in
``training.selftest``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

LEDGER_FILE = "dose_ledger.jsonl"
LINEAGE_FILE = "lineage.json"

# Increments at or below this are treated as nothing owed (float noise from the ramp).
_EPS = 1e-6


# -- row identity ------------------------------------------------------------- #

def row_kind(source: str, unmask_user: bool, user_loss_weight) -> str:
    """``wander`` / ``contamination`` / ``dialogue``.

    Only the SPLIT contamination copy (``unmask_user`` with no per-token weight) is its own
    identity — it carries the user-voice dose. A FOLDED contamination row is the
    exchange's only row and carries the response dose, so it is the ``dialogue`` identity.
    """
    if source == "wander":
        return "wander"
    if unmask_user and user_loss_weight is None:
        return "contamination"
    return "dialogue"


def row_key(source: str, source_session: str, exchange_index, unmask_user: bool = False,
            user_loss_weight=None) -> str:
    """Stable identity of one training row across builds."""
    idx = "" if exchange_index is None else str(int(exchange_index))
    return f"{row_kind(source, unmask_user, user_loss_weight)}|{source_session}|{idx}"


def key_of_row(r) -> str:
    """``row_key`` of a ``build_dataset.BuildRow``."""
    return row_key(r.source, r.source_session, r.exchange_index, r.unmask_user,
                   r.user_loss_weight)


def key_of_example(ex: dict) -> str:
    """``row_key`` of an ``sft_render.jsonl`` row (``row_render_dict`` projection)."""
    return row_key(ex.get("source") or "chat", ex.get("source_session") or "",
                   ex.get("exchange_index"), bool(ex.get("unmask_user")),
                   ex.get("user_loss_weight"))


def target_sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# -- load / save -------------------------------------------------------------- #

def ledger_path(adapter_dir) -> Path:
    return Path(adapter_dir) / LEDGER_FILE


def load(adapter_dir) -> Optional[dict]:
    """``{key: entry}`` from the adapter's ledger, or ``None`` when it has none."""
    p = ledger_path(adapter_dir)
    if not p.is_file():
        return None
    out: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("key"):
            out[rec["key"]] = rec
    return out


def save(adapter_dir, ledger: dict) -> Path:
    """Write the whole ledger (sorted by key — one line per row identity)."""
    p = ledger_path(adapter_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for key in sorted(ledger):
            fh.write(json.dumps(ledger[key], ensure_ascii=False) + "\n")
    tmp.replace(p)
    return p


def load_lineage(adapter_dir) -> Optional[dict]:
    p = Path(adapter_dir) / LINEAGE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except Exception:
        return None


def save_lineage(adapter_dir, lineage: dict) -> Path:
    p = Path(adapter_dir) / LINEAGE_FILE
    p.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -- the increment ------------------------------------------------------------ #

def apply_increments(rows: list, ledger: dict) -> tuple[list, dict]:
    """Rewrite each row's ``lr_multiplier`` from its target ``D(age)`` to the increment owed.

    *rows*: ``BuildRow`` list (preview rows excluded by the caller), each carrying its
    target dose on ``lr_multiplier``. Rows owing nothing are dropped. Returns
    ``(rows_to_train, stats)``; the returned rows are copies (``dose_target`` is kept on
    the stats per key so ``advance`` can record it).

    A row whose target text changed since it was last trained is NOT retrained by the
    change alone — under incremental training a correction is not retroactive (the full
    train is). Those rows are counted (``changed_targets``) so the build log shows them.
    """
    out: list = []
    stats = {"rows": len(rows), "owed": 0, "done": 0, "new": 0, "changed_targets": 0,
             "dose_owed": 0.0, "targets": {}}
    for r in rows:
        key = key_of_row(r)
        prev = ledger.get(key) or {}
        delivered = float(prev.get("delivered") or 0.0)
        target = float(r.lr_multiplier)
        if prev and prev.get("target_sha") and prev["target_sha"] != target_sha(r.target):
            stats["changed_targets"] += 1
        inc = target - delivered
        if inc <= _EPS:
            stats["done"] += 1
            continue
        if not prev:
            stats["new"] += 1
        stats["owed"] += 1
        stats["dose_owed"] += inc
        stats["targets"][key] = target
        out.append(replace(r, lr_multiplier=inc))
    return out, stats


def coverage(rows: list, ledger: dict) -> float:
    """Share of *rows*' identities the ledger knows (1.0 when there are no rows).

    A ledger written on this box covers nearly every local row — everything but what
    arrived since the last build. An adapter trained on ANOTHER box carries a ledger about
    that box's chats, so it covers almost none of these; resuming it as-is would train the
    whole local history at full dose in one build. ``train_cycle`` refuses below
    ``incremental.min_coverage``; ``training.adopt_adapter`` is the way in.
    """
    keys = {key_of_row(r) for r in rows}
    if not keys:
        return 1.0
    return sum(1 for k in keys if k in ledger) / len(keys)


def advance(ledger: dict, trained: Iterable[dict], *, built_at: str, build_id: str,
            incremental: bool) -> dict:
    """The ledger after a build trained *trained* (the retained ``sft_render`` rows).

    Incremental: the parent ledger copied forward, each trained row's ``delivered`` raised
    by the increment it was trained with (its ``lr_multiplier``). Full train: a fresh
    ledger, ``delivered`` = the multiplier each row was fitted with. Rows that were
    quarantined or mask-filtered are not in *trained*, so they stay owed.
    """
    new = {k: dict(v) for k, v in ledger.items()} if incremental else {}
    for ex in trained:
        if ex.get("preview"):
            continue
        key = key_of_example(ex)
        mult = float(ex.get("lr_multiplier") or 0.0)
        prev = new.get(key) or {}
        delivered = (float(prev.get("delivered") or 0.0) + mult) if incremental else mult
        new[key] = {
            "key": key, "delivered": round(delivered, 6),
            "last_trained_at": built_at, "build_id": build_id,
            "target_sha": target_sha(ex.get("variant") or ""),
        }
    return new


def seed_from_render(render_path, *, built_at: str, build_id: str) -> dict:
    """A ledger for an adapter that predates ledgers, from its build snapshot's render.

    The snapshot's ``sft_render.jsonl`` is exactly the retained rows that build trained;
    each delivered its ``lr_multiplier`` (a full train's per-row dose). Preview rows never
    trained and are skipped.
    """
    rows = []
    for line in Path(render_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return advance({}, rows, built_at=built_at, build_id=build_id, incremental=False)


def find_parent_snapshot(builds: list, adapter_dir) -> Optional[dict]:
    """The promoted ``builds.jsonl`` line that produced *adapter_dir* (by dir name)."""
    name = Path(str(adapter_dir)).name
    for rec in reversed(builds):
        if rec.get("outcome") != "promoted" or not rec.get("adapter_dir"):
            continue
        if Path(str(rec["adapter_dir"])).name == name:
            return rec
    return None


# -- lineage fingerprint ------------------------------------------------------ #

def _base_name(model_id: str) -> str:
    """Normalize a base id for comparison: the unsloth bnb-4bit repo of a model is the
    same base (a 4-bit training load of ``unsloth/gemma-4-31B-it`` resolves to
    ``unsloth/gemma-4-31b-it-unsloth-bnb-4bit``, which is what PEFT records)."""
    s = (model_id or "").strip().lower().rstrip("/")
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def lineage_mismatch(adapter_dir, *, model_id: str, lora_r: int, lora_alpha,
                     use_rslora: bool = True) -> list:
    """Why an incremental step on *adapter_dir* would be invalid ([] = compatible).

    Reads the adapter's own ``lineage.json`` when present (exact ``model_id``), else
    PEFT's ``adapter_config.json`` (base compared modulo the bnb-4bit repo suffix).
    Target modules are checked later, when the parent's weights load (a module mismatch
    surfaces as missing / unexpected LoRA keys).
    """
    reasons: list = []
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"unreadable adapter_config.json ({exc})"]
    lineage = load_lineage(adapter_dir) or {}
    parent_model = lineage.get("model_id")
    if parent_model:
        if parent_model != model_id:
            reasons.append(f"model_id {parent_model!r} != {model_id!r}")
    elif _base_name(cfg.get("base_model_name_or_path", "")) != _base_name(model_id):
        reasons.append(f"base {cfg.get('base_model_name_or_path')!r} != {model_id!r}")
    if int(cfg.get("r") or 0) != int(lora_r):
        reasons.append(f"lora_r {cfg.get('r')} != {lora_r}")
    if lora_alpha is not None and float(cfg.get("lora_alpha") or 0) != float(lora_alpha):
        reasons.append(f"lora_alpha {cfg.get('lora_alpha')} != {lora_alpha}")
    if bool(cfg.get("use_rslora")) != bool(use_rslora):
        reasons.append(f"use_rslora {cfg.get('use_rslora')} != {use_rslora}")
    return reasons


def safetensors_dtypes(path) -> dict:
    """``{dtype: count}`` from a safetensors header (no torch needed)."""
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    counts: dict = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        counts[v.get("dtype")] = counts.get(v.get("dtype"), 0) + 1
    return counts
