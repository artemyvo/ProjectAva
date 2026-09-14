#!/usr/bin/env python3
"""Ava runnable-snapshot export — a causally-closed, portable copy of a live Ava.

Packages everything that determines what Ava *says* into one directory you can
``scp``/``rsync`` to another host and boot — so you can move her off a busy GPU box,
debug a state in isolation, or rebuild from the raw material. The guiding invariant is
**causal closure**: every input to a chat turn is either inside the folder or named in
``MANIFEST.json`` as an external, immutable reference. Nothing else touches a response.

What travels (state — embedded):
  - ``server_config.json``  model/adapter/precision/context + sampling guards
                            (its ``adapter_id`` is rewritten RELATIVE, into this bundle)
  - ``prompts/``            system / RAG / surface prompt text
  - ``data/``               inference-side runtime state (memory + consolidation +
                            reflection runs); ``scratch/`` excluded — it is the disposable
                            per-cycle render — and ``hot/activity/`` excluded, being the
                            source box's telemetry log (what it DID, not what she knows;
                            see ``_copy_data_tree``). The FAISS index is NOT copied: the server
                            rebuilds it in-memory from the RAG sources on boot, so the
                            travelled state is the sole source of truth (nothing derived travels).
  - ``chats/``              the live chat transcripts (RAG source), now under server/data/chats
  - ``til/``                the durable wander corpus + provenance (RAG source), server/data/til
  - ``models/<adapter>/``   the active LoRA adapter weights (the one ``adapter_id`` pointed at)
  - ``training/<build_id>/``  the forensic build snapshot that PRODUCED that adapter — most
                            importantly ``sft_render.jsonl``, the exact rendered rows fed to
                            Unsloth (plus ``build_meta.json``/``wander.json``/``persona_digest.json``).
                            This is the answer to "where did that reply come from?": the weights
                            are a pure function of these rows + the base, so a behaviour traces
                            back to the corpus that trained it. Absent only for an adapter that
                            predates forensic snapshots (recorded as such in the manifest).

What is referenced by immutable id, NOT embedded (named in MANIFEST → external_dependencies):
  - the base model            (downloaded from HF via ``model_id``)
  - the embedder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
                                  (downloaded via sentence-transformers — determines RAG retrieval)
  - the server code           (pinned by ``git_commit``; the target checkout must match)

Known live inputs that are NOT reproducible from the folder (named in MANIFEST → live_inputs):
  - the wall-clock temporal anchor (``datetime.now()`` injected into the system prompt), and
  - per-request sampling (temperature / top_p / max_new_tokens) + the speaker name — these
    are *the question*, not *Ava*.

GPU-free, stdlib-only, filesystem-only: it runs whether the inference server is up OR down
(e.g. mid-training), which is exactly when you want to hand a copy to another box. Like
``wipe_state.py`` it lives in the repo (so its data-layout knowledge upgrades with ``git
pull``) and prints a single JSON result to STDOUT; human progress goes to STDERR.

Beyond the CLI, this module is the **single definition of the snapshot scope**, reused
two ways so the live path can never drift from the artifact export:
  - ``main()``            copytrees the scope into a directory (the ``scp``-able artifact);
  - ``stream_snapshot()`` writes the same scope + layout as an uncompressed tar to a
    file object (the Migrate tab's *Fetch snapshot* — streamed over the inference HTTP
    sidecar so a client can overwrite its checkout with the live Ava);
  - ``plan_manifest()``   returns the manifest without materializing anything (the
    sidecar's cheap peek: what's in the snapshot + whether the adapter is present).

Usage:
    python snapshot_state.py [--out DIR] [--with-reflections]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent            # server/
_REPO_ROOT = _SERVER_DIR.parent                          # repo root
_INFERENCE_DIR = _SERVER_DIR / "inference"
_DATA_DIR = _INFERENCE_DIR / "data"
# Live RAG sources that MOVED out of inference/data under the ordered server/data/ layout
# (see server/inference/server.py + core/wander_sft.py). The snapshot must capture them at
# their new homes or a migrated Ava boots with no chat corpus / wander channel.
_CHATS_DIR = _SERVER_DIR / "data" / "chats"              # live chat transcripts (RAG source)
_TIL_DIR = _SERVER_DIR / "data" / "til"                  # durable wander corpus + provenance
_PROMPTS_DIR = _INFERENCE_DIR / "prompts"
_MODELS_DIR = _SERVER_DIR / "models"
_REFLECTIONS_DIR = _SERVER_DIR / "reflections"
_SERVER_CONFIG_FILE = _SERVER_DIR / "server_config.json"
_LEGACY_SERVER_CONFIG_FILE = _INFERENCE_DIR / "server_config.json"
_RAG_ENGINE_FILE = _INFERENCE_DIR / "core" / "rag_engine.py"

# The persona pointer + layout live in one authority; reach it via inference/core.
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))
from core import persona_paths  # noqa: E402  (import-light, stdlib-only)

SCHEMA_VERSION = 1


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_info() -> dict:
    """The repo's current commit + whether the tree is dirty (provenance for code closure)."""
    def _run(args: list[str]) -> str:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    try:
        commit = _run(["rev-parse", "HEAD"])
        dirty = bool(_run(["status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": None, "dirty": None, "error": str(e)}


def _embedder_id() -> str | None:
    """Read RagEngine._EMBED_MODEL from source without importing (faiss/ST are heavy).

    Text-parse keeps this GPU-free AND drift-free: the manifest names whatever the code
    actually uses, not a hand-copied constant that could go stale.
    """
    try:
        m = re.search(r"""_EMBED_MODEL\s*=\s*["']([^"']+)["']""",
                      _RAG_ENGINE_FILE.read_text(encoding="utf-8"))
        return m.group(1) if m else None
    except Exception:
        return None


def _find_build_snapshot(adapter_id: str) -> Path | None:
    """The forensic build snapshot whose adapter matches *adapter_id* — the rows fed to Unsloth.

    Reads ``models/builds.jsonl`` (the append-only build log) and returns the ``snapshot_dir``
    of the latest ``promoted`` line whose ``adapter_dir`` is this adapter. Under the from-scratch
    build the adapter is a pure function of its build's corpus, so this snapshot's
    ``sft_render.jsonl`` is exactly what trained the current weights. Returns None when no such
    snapshot exists (e.g. an adapter that predates forensic snapshots).
    """
    builds = _MODELS_DIR / "builds.jsonl"
    if not adapter_id or not builds.is_file():
        return None
    want = Path(adapter_id)
    chosen: dict | None = None
    try:
        for line in builds.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("snapshot_dir") and Path(rec.get("adapter_dir") or "") == want:
                chosen = rec   # last matching line wins (the build that actually produced it)
    except Exception:
        return None
    if not chosen:
        return None
    snap = Path(chosen["snapshot_dir"])
    return snap if snap.is_dir() else None


def _read_build_meta(snapshot_dir: Path) -> dict:
    try:
        return json.loads((snapshot_dir / "build_meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        pass
    return total


def _live_digest_path() -> Path:
    """The live persona ``digest.json`` — travels as the snapshot's top-level digest.json.

    New layout: the active persona (``server/data/persona/<run_id>/``, selected by
    ``current.json``) owns the flat, unversioned ``digest.json``. Falls back to the legacy
    ``inference/data/hot/persona`` dir (``current.json`` → ``digest-<ts>.json``, else a flat
    ``digest.json``) for a box that has not migrated. Always returns a Path; callers gate on
    ``.is_file()`` (a snapshot with no digest is allowed — it degrades to plain framing)."""
    pdir = persona_paths.active_persona_dir()
    if pdir is not None:
        d = pdir / persona_paths.DIGEST_FILE
        if d.is_file():
            return d
    legacy = _DATA_DIR / "hot" / "persona"
    ptr = legacy / "current.json"
    if ptr.is_file():
        try:
            rel = (json.loads(ptr.read_text(encoding="utf-8")) or {}).get("path")
            if rel and (legacy / rel).is_file():
                return legacy / rel
        except Exception:
            pass
    return legacy / "digest.json"


def _copy_data_tree(src: Path, dst: Path) -> None:
    """Copy data/ excluding the disposable scratch/ render, the persona digest dir and the
    activity journal.

    ``hot/persona`` holds only the self-portrait, which travels as the snapshot's top-level
    ``digest.json`` (see :func:`persona_paths.digest_path`) — so it is dropped here to avoid
    a redundant second copy.

    ``hot/activity`` is the box's telemetry log (``core.activity_log``). Since it began
    carrying every pass's verbatim generation and the raw stdout of the train cycle it is
    one of the largest things under ``data/``, and it is the one thing here that a
    restored Ava does not depend on in any way: it records what the SOURCE box did, not
    what she knows. Copying it would put tens of MB of logs into every *Fetch snapshot*
    and hand the destination a history that was never its own."""
    def _ignore(dirpath, names):
        dp = Path(dirpath)
        if dp == src:                      # top-level scratch/ (per-cycle render, regenerated)
            return {n for n in names if n == "scratch"}
        if dp == src / "hot":              # hot/persona → travels as top-level digest.json
            return {n for n in names if n in ("persona", "activity")}
        return set()
    shutil.copytree(src, dst, ignore=_ignore)


# ──────────────────────────────────────────────────────────────────────────────
# Shared snapshot planning — ONE definition of "what a snapshot contains", reused
# by the CLI (copytree → dir) and the inference sidecar (tar → socket). Keeping the
# scope here means the live-fetch path can never drift from the artifact export.
# ──────────────────────────────────────────────────────────────────────────────

def _config_file() -> Path:
    """The box config: ``server/server_config.json`` (moved up out of ``inference/``
    on 2026-07-28). Resolved per call, not at import, because a not-yet-migrated box
    may move the file under a long-lived server process; the legacy path is still
    READ so a snapshot of such a box works, but this exporter never moves it."""
    if not _SERVER_CONFIG_FILE.is_file() and _LEGACY_SERVER_CONFIG_FILE.is_file():
        return _LEGACY_SERVER_CONFIG_FILE
    return _SERVER_CONFIG_FILE


def _load_config() -> dict:
    path = _config_file()
    if not path.is_file():
        raise FileNotFoundError(f"no server_config.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_adapter(cfg: dict, *, strict: bool):
    """``(adapter_src, adapter_name, adapter_rel, present)`` for the active adapter.

    ``adapter_src`` is the on-disk dir (None when no ``adapter_id`` is set, or when
    it is set but missing and ``strict`` is False). ``adapter_rel`` is its
    bundle-relative path (``models/<name>``). ``strict`` = True (CLI / stream: a
    runnable snapshot must physically contain the adapter it names) raises when the
    dir is missing; ``strict`` = False (manifest peek) reports ``present=False``.
    """
    adapter_id = (cfg.get("adapter_id") or "").strip()
    if not adapter_id:
        return None, None, None, False
    src = Path(adapter_id)
    name = src.name
    if not src.is_dir():
        if strict:
            raise FileNotFoundError(f"adapter_id points at a missing dir: {src}")
        return None, name, f"models/{name}", False
    return src, name, f"models/{name}", True


def _resolve_training(adapter_id: str):
    """``(training_info, build_snap, build_id)`` — the forensic corpus for the adapter."""
    training_info: dict = {"included": False}
    build_snap: Path | None = None
    build_id: str | None = None
    if adapter_id:
        build_snap = _find_build_snapshot(adapter_id)
        if build_snap is not None:
            meta = _read_build_meta(build_snap)
            build_id = meta.get("build_id") or build_snap.name
            training_info = {
                "included": True,
                "path": f"training/{build_id}",
                "build_id": build_id,
                "run_id": meta.get("run_id"),
                "corpus_fingerprint": meta.get("corpus_fingerprint"),
                "rows": meta.get("rows"),
                "render_file": "sft_render.jsonl",
                "note": "exact rendered rows fed to Unsloth for the active adapter",
            }
        else:
            training_info = {
                "included": False,
                "reason": "no forensic build snapshot recorded for the active adapter "
                          "(predates snapshots, or builds.jsonl has no matching line)",
            }
    return training_info, build_snap, build_id


def _build_manifest(cfg: dict, adapter_rel: str | None, adapter_name: str | None,
                    training_info: dict, *, with_reflections: bool) -> dict:
    """The audit ledger of what is / isn't in the folder (see module docstring)."""
    git = _git_info()
    embedder = _embedder_id()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "ava-runnable-snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_host": socket.gethostname(),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "context_length": cfg.get("context_length"),
        "adapter": ({"path": adapter_rel, "name": adapter_name} if adapter_rel else None),
        "training_data": training_info,
        "embedded": {
            "server_config.json": True,
            "digest.json": _live_digest_path().is_file(),
            "prompts/": _PROMPTS_DIR.is_dir(),
            "data/": _DATA_DIR.is_dir(),
            "chats/": _CHATS_DIR.is_dir(),
            "til/": _TIL_DIR.is_dir(),
            "models/<adapter>/": bool(adapter_rel),
            "training/<build_id>/": bool(training_info.get("included")),
            "reflections/": bool(with_reflections and _REFLECTIONS_DIR.is_dir()),
        },
        # The things deliberately NOT in the folder — fixed public artifacts named by
        # immutable id so causal closure stays auditable.
        "external_dependencies": {
            "base_model": {"model_id": cfg.get("model_id") or "", "base_quant": cfg.get("base_quant"),
                           "source": "huggingface", "embedded": False},
            "embedder": {"name": embedder, "source": "sentence-transformers",
                         "device": "cpu", "embedded": False},
            "code": {"git_commit": git.get("commit"), "embedded": False,
                     "note": "target checkout must be at this commit"},
        },
        # Inputs that shape a reply but are NOT reproducible from the folder.
        "live_inputs": [
            "wall-clock temporal anchor (datetime.now() → system prompt); "
            "not reproducible across days",
            "per-request sampling (temperature/top_p/max_new_tokens) and speaker name",
        ],
    }


def plan_manifest(*, with_reflections: bool = False) -> dict:
    """The snapshot manifest WITHOUT materializing anything (the sidecar's peek).

    Adds ``adapter_present`` / ``model_id`` / ``base_quant`` at the top level so a
    client can decide whether an export is worth requesting (and drive its
    same-box / base-compat checks) from one cheap call.
    """
    cfg = _load_config()
    _, adapter_name, adapter_rel, present = _resolve_adapter(cfg, strict=False)
    training_info, _, _ = _resolve_training((cfg.get("adapter_id") or "").strip())
    manifest = _build_manifest(cfg, adapter_rel, adapter_name, training_info,
                               with_reflections=with_reflections)
    manifest["adapter_present"] = present
    manifest["model_id"] = cfg.get("model_id")
    manifest["base_quant"] = cfg.get("base_quant")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# Adapter-only scope — the WEIGHTS alone, without the state they were fit on
# ──────────────────────────────────────────────────────────────────────────────
# The third and narrowest scope in this module, and the only one that is deliberately
# NOT causally closed: `stream_snapshot` ships an adapter together with the RAG sources,
# prompts and config that make it behave as it does, so a fetched snapshot is internally
# consistent by construction. This ships the adapter ALONE, to be attached to a *different*
# box's memory — the "hybrid" case: reflect and accumulate the corpus here, train over
# there, run the remote weights against local state.
#
# That is exactly why the manifest leads with `model_id`. A LoRA is fit against one
# specific base, and nothing downstream re-checks it: at 4-bit `inference_backend.load`
# resolves the base from the adapter's OWN `adapter_config.json`, so a mismatched adapter
# silently loads a base the local config never named; at 8/16-bit `_stage_adapter_with_base`
# overwrites that field with the local `model_id` and the mismatch surfaces as an opaque
# peft/unsloth load error. Both failures are bad in the way a fetch cannot fix afterwards,
# so the compare belongs on the CLIENT, before a byte is streamed (see the Migrate tab's
# FetchAdapterWorker, which refuses on mismatch).

def plan_adapter_manifest() -> dict:
    """What the adapter-only export WOULD ship, without materializing it.

    Deliberately a superset of what the client strictly needs, because this call is the
    one chance to refuse cheaply: `model_id` + `base_quant` drive the base-compat guard,
    `adapter_present` the existence guard, and `training_data` / `git_commit` let an
    operator see WHICH build's weights they are about to graft onto local state.
    """
    cfg = _load_config()
    adapter_src, adapter_name, adapter_rel, present = _resolve_adapter(cfg, strict=False)
    training_info, _, build_id = _resolve_training((cfg.get("adapter_id") or "").strip())
    git = _git_info()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "ava-adapter-only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_host": socket.gethostname(),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        # The compat facts. `model_id` is the base this adapter was fit against; a client
        # whose own model_id differs must refuse rather than fetch.
        "model_id": cfg.get("model_id"),
        "base_quant": cfg.get("base_quant"),
        "adapter": ({"path": adapter_rel, "name": adapter_name} if adapter_rel else None),
        "adapter_present": present,
        "size_bytes": _dir_size_bytes(adapter_src) if (present and adapter_src) else 0,
        "build_id": build_id,
        "training_data": training_info,
        "embedded": {"models/<adapter>/": bool(present), "MANIFEST.json": True},
        # Named so a reader is never left inferring the scope from what didn't arrive.
        "excluded": ["data/", "chats/", "til/", "prompts/", "digest.json",
                     "server_config.json", "reflections/", "training/<build_id>/"],
        "note": "adapter weights only — NOT a runnable snapshot. The receiving box keeps "
                "its own memory, prompts, config and persona; only adapter_id is repointed.",
    }


def stream_adapter(out) -> dict:
    """Write the active adapter dir as one **uncompressed** tar to ``out``.

    Two members: ``models/<adapter-name>/`` (the same arcname `stream_snapshot` uses, so
    a client's member→destination mapping is shared) and ``MANIFEST.json``. No config
    travels — repointing `adapter_id` is the receiving box's job, since only it knows
    where the dir landed and what the rest of its config must keep saying.
    """
    cfg = _load_config()
    adapter_src, _adapter_name, adapter_rel, _present = _resolve_adapter(cfg, strict=True)
    if adapter_src is None or not adapter_rel:
        raise FileNotFoundError("no active adapter to export (server_config.json has no "
                                "adapter_id — this box is running the bare base model)")
    manifest = plan_adapter_manifest()
    with tarfile.open(fileobj=out, mode="w|") as tar:
        tar.add(str(adapter_src), arcname=adapter_rel)
        _tar_add_bytes(tar, "MANIFEST.json",
                       (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    return manifest


def _tar_add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _tar_ignore_scratch(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    # arcname "data/…": drop the disposable per-cycle render dir, the persona digest dir
    # (the digest travels as the snapshot's top-level digest.json instead), and the
    # activity journal (telemetry about the SOURCE box — see _copy_data_tree). Kept in
    # step with that function by hand; they are the two halves of one scope.
    parts = Path(info.name).parts
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "scratch":
        return None
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "hot" \
            and parts[2] in ("persona", "activity"):
        return None
    return info


def stream_snapshot(out, *, with_reflections: bool = False) -> dict:
    """Write the snapshot as one **uncompressed** tar to ``out`` (the live-fetch path).

    Same scope + bundle layout as the CLI export (``server_config.json`` with a
    RELATIVE ``adapter_id``, ``prompts/``, ``data/`` minus scratch, the active
    ``models/<adapter>/``, its forensic ``training/<build_id>/``, and — only with
    ``with_reflections`` — the ``reflections/`` archive), plus ``MANIFEST.json``.
    Uncompressed because the LoRA weights don't gzip; the config + manifest ride as
    in-memory members so the bundle is self-describing. Returns the manifest.

    ``strict`` adapter resolution: raises if ``adapter_id`` is set but its dir is
    missing (the sidecar pre-checks and 404s before the stream headers go out).
    """
    cfg = _load_config()
    adapter_src, adapter_name, adapter_rel, _ = _resolve_adapter(cfg, strict=True)
    training_info, build_snap, build_id = _resolve_training((cfg.get("adapter_id") or "").strip())
    snap_cfg = dict(cfg)
    snap_cfg["adapter_id"] = adapter_rel  # portable: resolve against the snapshot root
    manifest = _build_manifest(cfg, adapter_rel, adapter_name, training_info,
                               with_reflections=with_reflections)

    with tarfile.open(fileobj=out, mode="w|") as tar:
        _tar_add_bytes(tar, "server_config.json",
                       (json.dumps(snap_cfg, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        if _PROMPTS_DIR.is_dir():
            tar.add(str(_PROMPTS_DIR), arcname="prompts")
        if _DATA_DIR.is_dir():
            tar.add(str(_DATA_DIR), arcname="data", filter=_tar_ignore_scratch)
        if _CHATS_DIR.is_dir():
            tar.add(str(_CHATS_DIR), arcname="chats")
        if _TIL_DIR.is_dir():
            tar.add(str(_TIL_DIR), arcname="til")
        if _live_digest_path().is_file():
            tar.add(str(_live_digest_path()), arcname="digest.json")
        if adapter_src is not None and adapter_rel:
            tar.add(str(adapter_src), arcname=adapter_rel)
        if build_snap is not None and build_id:
            tar.add(str(build_snap), arcname=f"training/{build_id}")
        if with_reflections and _REFLECTIONS_DIR.is_dir():
            tar.add(str(_REFLECTIONS_DIR), arcname="reflections")
        _tar_add_bytes(tar, "MANIFEST.json",
                       (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return manifest


def _materialize_snapshot(out_dir: Path, *, with_reflections: bool = False) -> dict:
    """Copy the current live Ava into *out_dir* as a runnable snapshot; return a summary.

    *out_dir* must already exist. Shared by the CLI (:func:`main`) and
    :func:`produce_persona` so the on-disk scope + layout is defined exactly once."""
    summary: dict = {}
    cfg = _load_config()
    adapter_src, adapter_name, adapter_rel, _ = _resolve_adapter(cfg, strict=True)
    training_info, build_snap, build_id = _resolve_training((cfg.get("adapter_id") or "").strip())

    # ── config (adapter_id rewritten relative) ─────────────────────────────
    snap_cfg = dict(cfg)
    snap_cfg["adapter_id"] = adapter_rel  # portable: resolve against snapshot root
    if adapter_src is not None:
        _log(f"Copying adapter {adapter_name}…")
        shutil.copytree(adapter_src, out_dir / "models" / adapter_name)
        summary["adapter"] = adapter_name
    else:
        _log("No adapter_id in config — base-only snapshot.")
    (out_dir / "server_config.json").write_text(
        json.dumps(snap_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── prompts ────────────────────────────────────────────────────────────
    if _PROMPTS_DIR.is_dir():
        _log("Copying prompts…")
        shutil.copytree(_PROMPTS_DIR, out_dir / "prompts")
        summary["prompts"] = True

    # ── data/ (full state minus scratch + persona digest) ──────────────────
    if _DATA_DIR.is_dir():
        _log("Copying data/ (excluding scratch + persona digest)…")
        _copy_data_tree(_DATA_DIR, out_dir / "data")
        summary["data_bytes"] = _dir_size_bytes(out_dir / "data")

    # ── chats/ + til/ — RAG sources that live under server/data/ now ────────
    if _CHATS_DIR.is_dir():
        _log("Copying chats/ (RAG chat corpus)…")
        shutil.copytree(_CHATS_DIR, out_dir / "chats")
        summary["chats_bytes"] = _dir_size_bytes(out_dir / "chats")
    if _TIL_DIR.is_dir():
        _log("Copying til/ (durable wander corpus)…")
        shutil.copytree(_TIL_DIR, out_dir / "til")
        summary["til_bytes"] = _dir_size_bytes(out_dir / "til")

    # ── persona digest: the self-portrait, as the top-level digest.json ─────
    if _live_digest_path().is_file():
        _log("Copying persona digest → digest.json…")
        shutil.copy2(_live_digest_path(), out_dir / "digest.json")
        summary["digest"] = True

    # ── training corpus: the rows fed to Unsloth for the active adapter ─────
    if build_snap is not None and build_id:
        _log(f"Copying training corpus from build {build_id}…")
        shutil.copytree(build_snap, out_dir / "training" / build_id)
        summary["training_build_id"] = build_id
    elif training_info.get("reason"):
        _log(f"WARNING: no build snapshot found for adapter — {training_info['reason']}")

    # ── optional: reflections/ archive (rebuild recipe / rollback lineage) ──
    if with_reflections and _REFLECTIONS_DIR.is_dir():
        _log("Copying reflections/ archive…")
        shutil.copytree(_REFLECTIONS_DIR, out_dir / "reflections")
        summary["reflections_bytes"] = _dir_size_bytes(out_dir / "reflections")

    # ── MANIFEST: the ledger of what is / isn't in the folder ──────────────
    manifest = _build_manifest(cfg, adapter_rel, adapter_name, training_info,
                               with_reflections=with_reflections)
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["total_bytes"] = _dir_size_bytes(out_dir)
    if manifest.get("git_dirty"):
        _log("WARNING: working tree is dirty — git_commit does not fully pin the code.")
    return summary


def produce_persona(run_id: str, *, activate: bool = True,
                    with_reflections: bool = False) -> Path:
    """Materialize the current live Ava as a frozen persona at ``data/persona/<run_id>/``
    and (by default) point ``current.json`` at it — the reflection→persona hand-off.

    This is what makes "the persona snapshot is the product of a reflection run" literal:
    called at the end of a run (after the adapter + digest are on disk) so the new persona
    carries the just-trained adapter, its forensic corpus, the fresh ``digest.json``, and
    the run's data/ state. Idempotent: re-producing the same run_id replaces the dir.
    GPU-free / filesystem-only, so it runs in the offline train_cycle (inference down)."""
    out_dir = persona_paths.personas_root() / str(run_id)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _log(f"Producing persona → {out_dir}")
    summary = _materialize_snapshot(out_dir, with_reflections=with_reflections)
    if activate:
        persona_paths.set_active(str(run_id))
        _log(f"Activated persona {run_id} (data/persona/current.json).")
    summary["path"] = str(out_dir)
    summary["run_id"] = str(run_id)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Ava runnable-snapshot export")
    parser.add_argument("--out", default="",
                        help="Output dir (default: server/exports/ava-snapshot-<ts>/)")
    parser.add_argument("--with-reflections", action="store_true",
                        help="Also embed the reflections/ archive (full rebuild recipe + "
                             "rollback lineage; larger). Off by default — a runnable snapshot "
                             "needs only the active adapter.")
    args = parser.parse_args()

    summary: dict = {}
    ok = True
    err: str | None = None
    out_dir: Path | None = None
    try:
        out_dir = (Path(args.out).expanduser().resolve() if args.out
                   else _SERVER_DIR / "exports" / f"ava-snapshot-{_utc_now_stamp()}")
        if out_dir.exists():
            raise FileExistsError(f"output dir already exists: {out_dir}")
        out_dir.mkdir(parents=True)
        _log(f"Snapshot → {out_dir}")

        summary = _materialize_snapshot(out_dir, with_reflections=args.with_reflections)
        summary["path"] = str(out_dir)
    except Exception as e:
        ok = False
        err = str(e)
        _log(f"Snapshot error: {e}")

    result: dict = {"ok": ok, "summary": summary}
    if out_dir is not None:
        result["path"] = str(out_dir)
    if err:
        result["error"] = err
    print(json.dumps(result), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
