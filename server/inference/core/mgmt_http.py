"""Inference-side HTTP sidecar — the read-mostly management/data endpoints.

These endpoints used to live in the watchdog, but none of them needs the
inference server to be *down*: they are post-mortem/clone bundles and a config
write. Keeping them here — in the inference process, which upgrades with a plain
``git pull`` + relaunch — lets the watchdog stay a thin, never-upgraded process
supervisor. That is the whole point of the split: the watchdog is the root of the
process tree (nothing on the box can restart it without SSH), so it must contain
only generic, stable supervision logic; anything project-specific (data layout,
config schema, tar bundle shape) belongs in a component that git-pulls freely.

The sidecar runs on its own HTTP port (default 8767) alongside the WebSocket
server, in a daemon thread. Because it lives in the inference process it is only
reachable while inference is up — which is exactly when these endpoints are
meaningful (you never fetch a clone bundle mid-train, when the process is down;
the training progress that *is* served while inference is down stays on the
watchdog's generic job runner).

Endpoints (same request/response contracts the watchdog used to serve, so the
client only re-points its URL — see client/core/backend_client.py and the
Migrate/Chat workers):

  GET  /status         → {hostname, data_dir, base_quant} — box/config facts the
                         Migrate + Fetch-artifacts "same box?" checks read.
  GET  /artifacts      → streamed gzip .tar post-mortem bundle (LoRA weights
                         excluded — heavy, GPU-only, regenerable).
  GET  /export         → streamed uncompressed .tar clone bundle (weights KEPT).
  GET  /snapshot/manifest → the runnable-snapshot manifest (adapter + RAG sources
                         + prompts + config) without materializing anything.
  GET  /snapshot/export → streamed uncompressed .tar of the full runnable snapshot
                         (adapter + data/ RAG sources + prompts + config + forensic
                         training corpus) — the Migrate tab's "Fetch snapshot",
                         which overwrites a checkout with the live Ava. Reuses
                         server/snapshot_state.py so the scope can't drift.
  GET  /training/review → gzipped JSON entry list for the Training review tab (the
                         latest build snapshot's render, projected to the reviewed
                         final turn + the live sidecars' repaired targets). Lets the
                         tab run from a remote UI box without a Fetch snapshot: the
                         projection is a fraction of the render and moves no weights.
  GET  /chats/manifest → {hot:[stem...], archive:[stem...], fingerprints:{stem:[tok...]}}.
                         The fingerprint is the ordered per-exchange token list
                         (exchange_id, else a content hash) so the client can tell a
                         pure append from an independent divergence.
  GET  /chats/export   → streamed gzip .tar of every chat file (hot + archive).
  POST /chats/import   → gzip .tar of chat files; non-destructive merge into
                         hot/chats. A new stem is imported; an existing HOT stem is
                         UPDATED when the incoming transcript is a pure append onto
                         ours (ours a strict prefix); a divergence / frozen-archived /
                         equal stem is skipped.
  POST /precision      → {base_quant} persisted to server_config.json (takes
                         effect on the next server restart).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import socket
import tarfile
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ──────────────────────────────────────────────────────────────────────────────
# Self-located paths (mirror the repo layout under server/). The sidecar needs no
# wiring from server.py beyond the busy-getter — it locates every managed root
# from its own file position, the same discipline core/reflection_archive.py uses.
# ──────────────────────────────────────────────────────────────────────────────
_INFERENCE_DIR = Path(__file__).resolve().parent.parent   # server/inference
_SERVER_ROOT = _INFERENCE_DIR.parent                       # server/
_DATA_DIR = _INFERENCE_DIR / "data"
# The ordered server/data/ home for the RAG sources + personas that MOVED out of
# inference/data (chats/, til/wander, persona/<run_id>/ + its current.json pointer).
# A clone bundle must carry this whole tree or a migrated Ava boots with no chat
# corpus, no wander channel, and no active persona.
_SERVER_DATA_DIR = _SERVER_ROOT / "data"
_MODELS_DIR = _SERVER_ROOT / "models"                      # LoRA adapter lineage
_REFLECTIONS_DIR = _SERVER_ROOT / "reflections"            # archive tree (outside data/)
# Box config: server/server_config.json (moved up out of inference/ on 2026-07-28 —
# it is the box's config, repointed by the offline train cycle and the wipe job while
# inference is DOWN, so it belongs above the role dir). The inference server migrates
# a legacy checkout at boot; _config_file() still falls back so a request that beats
# that (or a bare sidecar import) reads the right file.
_SERVER_CONFIG_FILE = _SERVER_ROOT / "server_config.json"
_LEGACY_SERVER_CONFIG_FILE = _INFERENCE_DIR / "server_config.json"
# Live chats moved to the new ordered server/data/chats root (flat; archive retired).
_HOT_CHATS_DIR = _SERVER_ROOT / "data" / "chats"
_ARCHIVE_CHATS_DIR = _SERVER_ROOT / "data" / "archive" / "chats"  # nonexistent; .exists()-guarded

# Post-mortem log set (all under server/). Named here so /artifacts can fold them
# in even though they are written by different processes (inference/watchdog/train).
_LOG_FILES = (
    _SERVER_ROOT / "server.log",
    _SERVER_ROOT / "watchdog.log",
    _SERVER_ROOT / "train.log",
    _SERVER_ROOT / "train_progress.jsonl",
)

# Base-model load precisions /precision accepts (mirrors the client's Migrate-tab
# selector and inference_backend's load_in_4bit/8bit handling; absent ⇒ 4-bit).
_VALID_QUANTS = ("16bit", "8bit", "4bit")

# Injected once by server.py: a predicate that is True while an in-process
# reflection run owns the GPU/config, so /chats/import and /precision refuse to
# mutate the corpus / config out from under it (the same race the old watchdog
# /precision guarded against an offline train).
_is_busy = lambda: False  # noqa: E731

# Injected once by server.py for the model-gossip serving endpoint (GOSSIP.md). Kept a
# thin callable bundle so the sidecar stays import-light and never imports generation:
#   _gossip_generate(messages, *, temperature, top_p, max_tokens, peer_name)
#       -> (answer, finish_reason, reasoning, usage)  — submits GPU work to the executor
#          and blocks; `reasoning` is the <think> trace, surfaced on
#          message.reasoning_content; `usage` is the OpenAI token-count block. Shared with
#          the public API endpoint (core.api_http) via generation._make_openai_generate.
#   _gossip_enabled() -> bool          — per-box opt-in (absent config ⇒ disabled ⇒ 404).
#   _gossip_peer_name() -> str | None  — configured peer name (fallback when the request
#                                        carries none).
_gossip_generate = None
_gossip_enabled = lambda: False  # noqa: E731
_gossip_peer_name = lambda: None  # noqa: E731


def configure(*, is_busy=None, gossip_generate=None, gossip_enabled=None,
              gossip_peer_name=None) -> None:
    """Wire in the busy predicate and (optionally) the gossip serving capability.

    Called once from server startup. ``gossip_generate``/``gossip_enabled`` are the
    model-gossip serving endpoint's only coupling to the model (see GOSSIP.md §4)."""
    global _is_busy, _gossip_generate, _gossip_enabled, _gossip_peer_name
    if is_busy is not None:
        _is_busy = is_busy
    if gossip_generate is not None:
        _gossip_generate = gossip_generate
    if gossip_enabled is not None:
        _gossip_enabled = gossip_enabled
    if gossip_peer_name is not None:
        _gossip_peer_name = gossip_peer_name


# ──────────────────────────────────────────────────────────────────────────────
# Config (server_config.json is the inference server's domain — never the watchdog's)
# ──────────────────────────────────────────────────────────────────────────────

def _config_file() -> Path:
    """The live box config path, tolerating a not-yet-migrated checkout."""
    if not _SERVER_CONFIG_FILE.is_file() and _LEGACY_SERVER_CONFIG_FILE.is_file():
        return _LEGACY_SERVER_CONFIG_FILE
    return _SERVER_CONFIG_FILE


def _read_config() -> dict:
    """Best-effort read of server_config.json as a dict ({} on any failure)."""
    try:
        cfg = json.loads(_config_file().read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _read_base_quant() -> str | None:
    """Best-effort read of the current base_quant from server_config.json."""
    return _read_config().get("base_quant") or None


def _set_base_quant(value: str) -> dict:
    """Persist base_quant into server_config.json (takes effect next restart).

    The load precision is read at model-load time (the startup auto-load / a
    ``load`` message), so a persisted change only takes effect on the NEXT server
    restart — this deliberately does not touch the running, already-resident model.
    Atomic write (temp + os.replace): a torn write would corrupt the only pointer
    to base + adapter. Everything else in the config is kept verbatim.
    """
    if value not in _VALID_QUANTS:
        return {"ok": False,
                "error": f"invalid base_quant {value!r}; expected one of {_VALID_QUANTS}"}
    config_file = _config_file()
    try:
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            return {"ok": False, "error": "server_config.json is not a JSON object"}
    except Exception as e:
        return {"ok": False, "error": f"could not read server_config.json: {e}"}

    previous = cfg.get("base_quant")
    cfg["base_quant"] = value

    text = json.dumps(cfg, indent=2) + "\n"
    tfd, tpath = tempfile.mkstemp(dir=str(config_file.parent),
                                  prefix=".server_config.", suffix=".tmp")
    try:
        with os.fdopen(tfd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tpath, config_file)
    except Exception as e:
        try:
            os.unlink(tpath)
        except OSError:
            pass
        return {"ok": False, "error": f"could not write server_config.json: {e}"}

    print(f"[sidecar] base_quant set to {value} (was {previous or 'default 4bit'}); "
          f"takes effect on next inference server restart.", flush=True)
    return {"ok": True, "base_quant": value, "previous": previous,
            "note": "takes effect on next server restart"}


# ──────────────────────────────────────────────────────────────────────────────
# Bundle streaming (artifacts / clone) — the data-layout knowledge lives HERE
# ──────────────────────────────────────────────────────────────────────────────

# The artifact bundle's top-level subtrees, named relative to the local server/
# dir. The bundle ALWAYS carries a directory entry for each root (even when empty
# server-side) so the client can clear its stale local copy before extracting.
_ARTIFACT_ROOTS = ("inference/data", "reflections", "logs")
# The migration (clone) bundle's roots — weights-inclusive, config carried as a
# single-file member (not a dir root). ``data`` is the new ordered server/data/ home
# (chats + til/wander + personas); ``inference/data`` still carries the reflection
# working state. Both travel — they are disjoint subtrees.
_MIGRATE_ROOTS = ("inference/data", "data", "reflections", "models")


def _dir_member(name: str) -> tarfile.TarInfo:
    """An explicit (possibly empty) directory entry for the tarball."""
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = int(time.time())
    return info


def _stream_artifacts(out) -> None:
    """Stream the full post-mortem artifact set as one gzip tarball to ``out``.

    Three top-level subtrees, each rooted to mirror the repo layout under
    ``server/`` so the analyst's checkout reproduces the GPU box 1:1:
      - ``inference/data/`` — chats + reflection state (scratch/ excluded: the
        disposable per-cycle render);
      - ``reflections/``   — the reflection archive tree; per-run ``adapter/``
        subdirs EXCLUDED (hundreds of MB of LoRA weights);
      - ``logs/``          — server / watchdog / train logs + the
        structured progress journals.

    Each root gets a directory entry even when empty, so the client treats it as
    authoritative and clears stale local files. Written incrementally (streaming
    ``w|gz``) straight to the socket, never buffered whole in memory.
    """
    def _filter_data(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if len(parts) >= 3 and parts[2] == "scratch":
            return None
        return info

    def _filter_reflections(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if len(parts) >= 3 and parts[2] == "adapter":
            return None
        return info

    with tarfile.open(fileobj=out, mode="w|gz") as tar:
        for root in _ARTIFACT_ROOTS:
            tar.addfile(_dir_member(root))
        if _DATA_DIR.is_dir():
            tar.add(str(_DATA_DIR), arcname="inference/data", filter=_filter_data)
        if _REFLECTIONS_DIR.is_dir():
            tar.add(str(_REFLECTIONS_DIR), arcname="reflections",
                    filter=_filter_reflections)
        for f in _LOG_FILES:
            if f.is_file():
                tar.add(str(f), arcname=f"logs/{f.name}")


def _stream_migration_bundle(out) -> None:
    """Stream a weights-inclusive clone bundle as one **uncompressed** tar to ``out``.

    The transport behind the Migrate tab: pull a complete, still-running Ava (RAG,
    chats, reflection lineage, adapter weights, config) so a second host can adopt
    her. Unlike ``_stream_artifacts`` the LoRA weights are KEPT:
      - ``inference/data/`` — reflection working state (memory + consolidation +
        reflection runs; scratch/ excluded — the disposable per-cycle render);
      - ``data/``           — the ordered server/data/ home: the chat corpus
        (``chats/``), the durable wander corpus (``til/``), and the persona lineage
        (``persona/<run_id>/`` + its ``current.json`` pointer). These are RAG sources
        and the active-persona pointer, so a clone without them boots an amnesiac Ava;
      - ``reflections/``    — archive tree, adapters kept (rollback lineage);
      - ``models/``         — the full LoRA adapter lineage;
      - ``server_config.json`` — the pointer file the client rewrites for the box.

    **No gzip** (``w|`` streaming): the two hosts share a fast link, so gzip would
    only burn CPU on a multi-GB weights-laden bundle; raw tar lets the NIC be the
    bottleneck. Each dir root still gets an entry even when empty.
    """
    def _filter_inference_data(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        # arcname "inference/data/…": drop the disposable per-cycle scratch render.
        parts = Path(info.name).parts
        if len(parts) >= 3 and parts[2] == "scratch":
            return None
        return info

    with tarfile.open(fileobj=out, mode="w|") as tar:
        for root in _MIGRATE_ROOTS:
            tar.addfile(_dir_member(root))
        if _DATA_DIR.is_dir():
            tar.add(str(_DATA_DIR), arcname="inference/data",
                    filter=_filter_inference_data)
        if _SERVER_DATA_DIR.is_dir():
            tar.add(str(_SERVER_DATA_DIR), arcname="data")
        if _REFLECTIONS_DIR.is_dir():
            tar.add(str(_REFLECTIONS_DIR), arcname="reflections")
        if _MODELS_DIR.is_dir():
            tar.add(str(_MODELS_DIR), arcname="models")
        # Bundle layout is unchanged: a single member at the bundle root, whichever
        # side of the 2026-07-28 move this box's file is on.
        if _config_file().is_file():
            tar.add(str(_config_file()), arcname="server_config.json")


# ──────────────────────────────────────────────────────────────────────────────
# Runnable snapshot (Migrate tab: "Fetch snapshot" — adapter + RAG sources)
# ──────────────────────────────────────────────────────────────────────────────
# The full clone (/export) and the snapshot differ by scope, not transport: the
# snapshot is the causally-closed "current Ava personality" (active adapter + its
# RAG sources in data/ + prompts + config; NO reflections/ archive or old adapter
# lineage). Its scope lives in server/snapshot_state.py — the SAME definition the
# CLI export uses — so we import it lazily rather than re-encode the file list here.

def _snapshot_module():
    """Lazily import ``server/snapshot_state.py`` (the single snapshot-scope definition).

    Kept lazy + path-injected (snapshot_state lives at the server root, not on the
    inference package path) so the sidecar has no import-time coupling to it; it is
    stdlib-only, so the import is cheap and safe from the daemon thread.
    """
    import importlib
    import sys as _sys
    if str(_SERVER_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SERVER_ROOT))
    return importlib.import_module("snapshot_state")


# ──────────────────────────────────────────────────────────────────────────────
# Chat sync (Migrate tab: merge transcripts between two Avas)
# ──────────────────────────────────────────────────────────────────────────────
# A *merge*, not a clone: moves only the transcripts each side is missing (or has
# an OLDER copy of), never blindly overwrites or deletes. A chat is identified by
# its stem (timestamp before the first dot), shared by the transcript and its
# .state.json / .shareml.json companions. A stem is "present" if it exists in hot/
# OR archive/; a genuinely new chat is imported into hot/chats so it re-enters the
# RAG working set and is eligible for the next reflection run.
#
# A chat present on both boxes can still have *grown* on one side after an earlier
# merge — most commonly an Ava-initiated session that got a reply appended on one
# box only (ChatLogger.resume_session). We reconcile that by comparing the ordered
# per-exchange token lists ("fingerprints"): if one side's is a strict prefix of the
# other's, the longer side is a pure append and wins (the shorter is updated); if
# they diverge (both grew independently), we cannot safely merge and leave both.

def _chat_stems(directory: Path) -> set[str]:
    """Stems (timestamp before the first dot) of the chat files in *directory*."""
    stems: set[str] = set()
    if directory.is_dir():
        for f in directory.iterdir():
            if f.is_file() and f.name.endswith(".json"):
                stems.add(f.name.split(".", 1)[0])
    return stems


def _exchange_tokens(data: dict) -> list[str]:
    """Ordered per-exchange identity tokens for a transcript's exchange list.

    Prefers each exchange's ``exchange_id`` (stable across a file copy, so two boxes
    that synced the chat agree on the shared prefix); falls back to a content hash of
    the user/CoT/answer text for any legacy exchange that lacks an id. This is the
    unit the prefix comparison runs on — a pure append shares the whole shorter list
    as a prefix, an independent divergence does not.
    """
    tokens: list[str] = []
    for ex in (data.get("exchanges") or []):
        if not isinstance(ex, dict):
            tokens.append("")
            continue
        eid = ex.get("exchange_id")
        if eid:
            tokens.append(f"id:{eid}")
            continue
        h = hashlib.sha1()
        for k in ("user_prompt", "assistant_cot", "assistant_response"):
            h.update(str(ex.get(k, "")).encode("utf-8", "replace"))
            h.update(b"\x00")
        tokens.append("h:" + h.hexdigest())
    return tokens


def _tokens_from_bytes(raw: bytes) -> list[str]:
    """Fingerprint tokens for a transcript given its raw JSON bytes."""
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return _exchange_tokens(data) if isinstance(data, dict) else []


def _fingerprint_file(path: Path) -> list[str]:
    """Fingerprint tokens for a transcript on disk (empty on any read/parse error)."""
    try:
        return _tokens_from_bytes(path.read_bytes())
    except Exception:
        return []


def _is_strict_prefix(a: list[str], b: list[str]) -> bool:
    """True iff *a* is a strict prefix of *b* — i.e. *b* is a pure append onto *a*."""
    return len(a) < len(b) and b[: len(a)] == a


def _chats_manifest() -> dict:
    """The chat stems this box holds, split by location, plus per-stem fingerprints.

    The fingerprint (ordered exchange tokens of the ``<stem>.json`` transcript) lets
    the client decide, for a stem present on both boxes, whether one side is a pure
    append of the other (update) or the two diverged (skip). hot is listed last so
    its fingerprint wins over any stale archive copy sharing the stem.
    """
    hot = sorted(_chat_stems(_HOT_CHATS_DIR))
    archive = sorted(_chat_stems(_ARCHIVE_CHATS_DIR))
    fingerprints: dict[str, list[str]] = {}
    for directory, stems in ((_ARCHIVE_CHATS_DIR, archive), (_HOT_CHATS_DIR, hot)):
        for stem in stems:
            fingerprints[stem] = _fingerprint_file(directory / f"{stem}.json")
    return {"hot": hot, "archive": archive, "fingerprints": fingerprints}


def _stream_chats_bundle(out) -> None:
    """Stream a gzip tar of every chat file (hot + archive) to ``out``.

    Members are namespaced by origin (``hot/<file>`` / ``archive/<file>``); the
    client imports pulled chats into its own hot/chats regardless of origin (they
    need re-reflection there). Transcripts are small text, so streaming the whole
    set and letting the client filter to the stems it lacks is cheaper than a
    per-stem round trip.
    """
    with tarfile.open(fileobj=out, mode="w|gz") as tar:
        for loc, d in (("hot", _HOT_CHATS_DIR), ("archive", _ARCHIVE_CHATS_DIR)):
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and f.name.endswith(".json"):
                    tar.add(str(f), arcname=f"{loc}/{f.name}")


def _import_chats(blob: bytes) -> dict:
    """Merge chat files from a gzip tar into hot/chats. Never blindly overwrites.

    Authoritative + race-safe (we re-check disk here, not trusting the client's view):

      * stem NOT present on this box → imported (all its files land in hot/chats so it
        re-enters the RAG working set and is eligible for the next reflection run);
      * stem present in HOT and the incoming transcript is a **pure append** onto ours
        (ours a strict prefix of theirs) → updated (all its files overwritten; a local
        ``.state.json`` sidecar the incoming didn't replace is dropped, since the grown
        transcript makes any sidecar built against the shorter version stale);
      * stem present but frozen (only in archive), transcripts equal, or the two
        **diverged** (each grew independently) → skipped, both sides left as-is.

    Member names are flattened to a basename (guards path traversal in the tar), and a
    stem's files (transcript + companions) are decided and written together.
    """
    existing_hot = _chat_stems(_HOT_CHATS_DIR)
    existing_arch = _chat_stems(_ARCHIVE_CHATS_DIR)
    imported: set[str] = set()
    updated: set[str] = set()
    skipped: set[str] = set()
    _HOT_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    # Group the incoming members by stem so each chat is decided as a unit.
    groups: dict[str, list[tuple[str, bytes]]] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if not name.endswith(".json"):
                continue
            stem = name.split(".", 1)[0]
            if not stem:
                continue
            src = tar.extractfile(m)
            if src is None:
                continue
            groups.setdefault(stem, []).append((name, src.read()))

    for stem, files in groups.items():
        if stem not in existing_hot and stem not in existing_arch:
            for name, raw in files:
                (_HOT_CHATS_DIR / name).write_bytes(raw)
            imported.add(stem)
            continue

        # Present already. A frozen (archive-only) chat is never touched.
        if stem not in existing_hot:
            skipped.add(stem)
            continue

        transcript = next((raw for name, raw in files if name == f"{stem}.json"), None)
        if transcript is None:
            skipped.add(stem)
            continue
        ours = _fingerprint_file(_HOT_CHATS_DIR / f"{stem}.json")
        theirs = _tokens_from_bytes(transcript)
        if not _is_strict_prefix(ours, theirs):
            skipped.add(stem)  # equal, or diverged — cannot merge
            continue

        for name, raw in files:
            (_HOT_CHATS_DIR / name).write_bytes(raw)
        incoming_names = {name for name, _ in files}
        sidecar = _HOT_CHATS_DIR / f"{stem}.state.json"
        if f"{stem}.state.json" not in incoming_names and sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        updated.add(stem)

    return {"imported": sorted(imported), "updated": sorted(updated),
            "skipped": sorted(skipped)}


# ──────────────────────────────────────────────────────────────────────────────
# Model gossip — the OpenAI-compatible serving endpoint (GOSSIP.md §4)
# ──────────────────────────────────────────────────────────────────────────────
# The minimal /v1/chat/completions shape a peer Ava's Encounter CounterpartClient
# reads (encounter.py: choices[0].message.content + finish_reason). Non-streaming only.

def _openai_response(answer: str, model: str, finish_reason: str = "stop",
                     reasoning: str = "", usage: dict | None = None) -> dict:
    """The OpenAI chat-completion body the peer's CounterpartClient reads.

    Delegates to ``core.api_http.openai_response`` — the public API endpoint and this
    gossip route serve the same wire format, so they share one definition rather than two
    that drift. Only the id prefix differs (a gossip completion stays recognisable in a
    peer's logs)."""
    from core import api_http
    return api_http.openai_response(answer, model, finish_reason, reasoning, usage,
                                    id_prefix="gossip")


def _peer_name_from_request(req: dict) -> str | None:
    """Peer name for the identity line: the request's OpenAI ``user`` field, else the
    box's configured ``gossip.peer_name``, else None (no identity line)."""
    name = req.get("user")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return _gossip_peer_name()


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress default access logging

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_gzip(self, status: int, data: dict) -> None:
        """Send a JSON document gzipped, with a real Content-Length.

        For the few payloads that are large but bounded (the training-review entry
        list runs to megabytes of prose, which gzips to a fraction) — unlike the tar
        endpoints there is nothing to stream, so the client gets a length and a clean
        error instead of having to detect a truncated body."""
        body = gzip.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_headers(self, content_type: str) -> None:
        # Size isn't known up front (packed on the fly), so omit Content-Length
        # and signal end-of-body by closing the connection (client reads to EOF).
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/status":
            self._json(200, {
                "hostname": socket.gethostname(),
                "data_dir": str(_DATA_DIR.resolve()),
                "base_quant": _read_base_quant(),
            })

        elif parsed.path == "/artifacts":
            try:
                self._stream_headers("application/gzip")
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                try:
                    _stream_artifacts(self.wfile)
                except Exception:
                    # Headers already on the wire; drop the connection and let the
                    # client surface the truncated stream (it stages to a temp dir
                    # and only swaps in a complete bundle).
                    pass

        elif parsed.path == "/export":
            try:
                self._stream_headers("application/x-tar")
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                try:
                    _stream_migration_bundle(self.wfile)
                except Exception:
                    pass

        elif parsed.path == "/snapshot/manifest":
            try:
                self._json(200, _snapshot_module().plan_manifest())
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif parsed.path == "/snapshot/export":
            # Peek first: a runnable snapshot must physically contain the adapter it
            # references, so 404 cleanly if adapter_id is set but its dir is gone
            # (a base-only snapshot — adapter is None — is allowed).
            try:
                mod = _snapshot_module()
                m = mod.plan_manifest()
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            if m.get("adapter") is not None and not m.get("adapter_present"):
                self._json(404, {"error": "active adapter dir is missing; "
                                          "cannot export a runnable snapshot"})
                return
            try:
                self._stream_headers("application/x-tar")
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                try:
                    mod.stream_snapshot(self.wfile)
                except Exception:
                    pass

        elif parsed.path == "/adapter/manifest":
            # The compat peek for the Migrate tab's "Fetch adapter": model_id +
            # base_quant + adapter presence, so the client can refuse a base mismatch
            # before a byte of weights moves (nothing downstream re-checks it — see
            # snapshot_state's adapter-only scope note).
            try:
                self._json(200, _snapshot_module().plan_adapter_manifest())
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif parsed.path == "/adapter/export":
            # The weights ALONE — no data/, prompts, config or digest. Unlike
            # /snapshot/export this is deliberately not causally closed: the receiving
            # box keeps its own memory and persona and grafts these weights onto it.
            try:
                mod = _snapshot_module()
                m = mod.plan_adapter_manifest()
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            if m.get("adapter") is None:
                self._json(404, {"error": "no active adapter (this box runs the bare "
                                          "base model); nothing to export"})
                return
            if not m.get("adapter_present"):
                self._json(404, {"error": "active adapter dir is missing; "
                                          "cannot export the adapter"})
                return
            try:
                self._stream_headers("application/x-tar")
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                try:
                    mod.stream_adapter(self.wfile)
                except Exception:
                    pass

        elif parsed.path == "/training/review":
            # Read-only projection of the latest build's render + the live sidecars.
            # Deliberately NOT gated on _is_busy(): it mutates nothing, and a reflection
            # run is exactly when an operator wants to read what the last build trained.
            from core import training_review
            source = (parse_qs(parsed.query).get("source") or ["snapshot"])[0]
            try:
                payload = training_review.build_payload(source=source)
            except training_review.ReviewUnavailable as e:
                self._json(404, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                self._json_gzip(200, payload)

        elif parsed.path == "/chats/manifest":
            self._json(200, _chats_manifest())

        elif parsed.path == "/chats/export":
            try:
                self._stream_headers("application/gzip")
            except Exception as e:
                self._json(500, {"error": str(e)})
            else:
                try:
                    _stream_chats_bundle(self.wfile)
                except Exception:
                    pass

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/precision":
            # A config rewrite races a reflection run's train hand-off (which
            # repoints adapter_id at its end), so refuse while a run owns the config.
            if _is_busy():
                self._json(409, {"ok": False, "error": "a reflection run is in "
                                 "progress; cannot change precision"})
                return
            params = self._read_json_body()
            result = _set_base_quant(str(params.get("base_quant", "")))
            self._json(200 if result.get("ok") else 400, result)

        elif parsed.path == "/chats/import":
            # A reflection run reads chats from disk, so importing mid-run would
            # change the corpus under it. Refuse while a run is active.
            if _is_busy():
                self._json(409, {"error": "a reflection run is in progress; "
                                          "cannot import chats"})
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                result = _import_chats(body)
            except Exception as e:
                self._json(500, {"error": f"chat import failed: {e}"})
                return
            n = len(result.get("imported", []))
            n_upd = len(result.get("updated", []))
            if n or n_upd:
                print(f"[sidecar] chat merge: imported {n} new + updated {n_upd} "
                      f"appended transcript(s) into hot/chats (re-reflection pending); "
                      f"skipped {len(result.get('skipped', []))} unchanged/diverged.",
                      flush=True)
            result["ok"] = True
            self._json(200, result)

        elif parsed.path in ("/v1/chat/completions", "/chat/completions"):
            # Model gossip: answer a peer Ava's Encounter loop as if we were vLLM.
            if not _gossip_enabled() or _gossip_generate is None:
                self._json(404, {"error": "gossip endpoint disabled"})
                return
            if _is_busy():
                # A reflection run / encounter owns the single GPU worker; return a
                # clean 503 instead of queueing behind a many-minute job.
                self._json(503, {"error": "server busy"})
                return
            req = self._read_json_body()
            messages = req.get("messages") or []
            if not messages:
                self._json(400, {"error": "no messages"})
                return
            try:
                answer, finish_reason, reasoning, usage = _gossip_generate(
                    messages,
                    temperature=float(req.get("temperature", 1.0)),
                    top_p=float(req.get("top_p", 0.95)),
                    max_tokens=int(req.get("max_tokens", 1024)),
                    peer_name=_peer_name_from_request(req),
                )
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self._json(200, _openai_response(answer, req.get("model", "ava"),
                                             finish_reason, reasoning, usage))

        else:
            self._json(404, {"error": "not found"})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body) if body else {}
            return params if isinstance(params, dict) else {}
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def start(host: str, port: int) -> ThreadingHTTPServer:
    """Launch the sidecar HTTP server in a daemon thread and return it.

    Threaded so a long-running fetch (a multi-GB clone bundle) doesn't block a
    concurrent /status poll. The handlers touch only the filesystem + a cheap
    busy predicate, so concurrent requests are safe.
    """
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="mgmt-http-sidecar")
    t.start()
    print(f"[sidecar] management HTTP on http://{host}:{port}", flush=True)
    return httpd
