"""Migrate widget — clone a running Ava onto this box.

Pulls a **complete, still-running Ava** from the connected server — RAG, chats,
reflection lineage, LoRA adapter weights, and the ``server_config.json`` pointer —
and lays them down under this checkout's ``server/`` dir so a local inference
server can pick her up and continue. One host can then run "production" reflection
cycles while this one is a debugging clone.

The transport is the inference HTTP sidecar's ``GET /export`` (an uncompressed
clone tar; see ``server/inference/core/mgmt_http.py``), which — unlike the analyst
``/artifacts`` bundle — keeps the adapter weights and carries the config. These
data endpoints live on the inference sidecar (default port 8767), not the watchdog,
because they don't need inference down and so upgrade with a plain ``git pull``.
This widget:

  1. streams the bundle into a temp staging dir and atomically swaps the managed
     roots (``inference/data`` = reflection working state; ``data`` = the ordered
     ``server/data/`` home for the chat corpus + til/wander + persona lineage;
     ``reflections`` / ``models``) + ``server_config.json`` into the local ``server/``
     (same mirror discipline as *Fetch artifacts*);
  2. **rewrites the migrated config** — repoints the absolute ``adapter_id`` at the
     local ``models/`` copy, and sets ``base_quant`` from the precision selector
     (16 / 8 / 4-bit). The base ``model_id`` is kept verbatim so tokenizer + LoRA
     stay compatible; precision is a load-time choice on the new box.

It only *prepares* the box — it does not launch the local server. After a
successful migrate, start the local watchdog, point the client at
``ws://localhost:8765``, and load the model from the Chat tab.

The tab also carries a **Change precision** button: it reuses the same precision
selector but, instead of migrating anything, just persists ``base_quant`` on the
*connected* server (sidecar ``POST /precision``). The running server is not
bounced — the new precision is read at model-load time, so it takes effect on the
next server restart.

A **Merge chats** button does a two-way sync of chat transcripts between this
checkout and the connected server (``GET /chats/manifest`` + ``GET /chats/export``
to pull, ``POST /chats/import`` to push). Unlike Migrate (a full clone that
REPLACES the local subtrees), it only moves the chats each side is missing — or has
an OLDER copy of. A chat present on both boxes that GREW on one side (e.g. a reply
appended to an Ava-initiated session after an earlier merge) updates the shorter
copy; a chat appended to on both boxes independently diverges and is left untouched
(a conflict). The manifest carries per-stem fingerprints (ordered exchange tokens)
so a pure append is told apart from a divergence. Chats new to or updated on a box
must be re-reflected there (a Sleep pass), since the weights/RAG that consolidated
them on the origin box do not travel with the raw transcript.

Finally, a **Fetch snapshot** button pulls the connected server's *runnable
snapshot* — the active LoRA adapter PLUS its RAG sources (the whole ``data/`` tree:
reflection memory, persona digest, chat corpus), prompts, and config — and
overwrites this checkout with it, so the local server boots as the current Ava.
Server side it streams ``server/snapshot_state.py`` over the sidecar
(``GET /snapshot/manifest`` + ``GET /snapshot/export``), so the scope is the same
one the CLI snapshot export uses. It is snapshot-scoped (skips the reflections/
archive and old adapter lineage — lighter than a full Migrate) and replaces
``inference/data`` / ``inference/prompts`` / the active adapter / ``server_config.json``.
The whole config is the source's, so adapter+base+config stay consistent with no
compat guard needed. The local server is NOT started.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont

if TYPE_CHECKING:
    from ui.chat_widget import ChatWidget


# Precision tiers offered in the selector → the base_quant value written into the
# migrated server_config.json. Keeping the source model_id at any of these tiers
# preserves tokenizer + LoRA compatibility (precision is a load-time choice).
_QUANT_OPTIONS = [
    ("16-bit (bf16)", "16bit"),
    ("8-bit", "8bit"),
    ("4-bit", "4bit"),
]

# Must match mgmt_http._MIGRATE_ROOTS (the dir subtrees; server_config.json is a
# separate single-file member handled alongside them). ``data`` is the ordered
# server/data/ home (chat corpus + til/wander + persona lineage) that moved out of
# inference/data; ``inference/data`` still carries the reflection working state. The
# roots are disjoint subtrees under the local server/ dir, so swap order is irrelevant.
_MANAGED_ROOTS = ("inference/data", "data", "reflections", "models")


class MigrateSourceWorker(QThread):
    """Fetches the source watchdog /status (hostname, data_dir, base_quant) off-thread."""

    status_done = pyqtSignal(dict)

    def __init__(self, host: str, mgmt_port: int = 8766):
        super().__init__()
        self._host = host
        self._mgmt_port = mgmt_port

    def run(self) -> None:
        import urllib.request
        url = f"http://{self._host}:{self._mgmt_port}/status"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
                self.status_done.emit({"type": "success", "data": data})
        except Exception as exc:
            self.status_done.emit({"type": "error", "message": str(exc)})


class MigrateWorker(QThread):
    """Downloads the source's clone bundle and prepares the local server/ dir.

    Streams ``GET /export`` straight through a streaming (uncompressed) tar reader
    — the bundle carries the multi-GB adapter lineage, so it is never buffered
    whole in memory — extracts into a temp staging dir under the local ``server/``,
    atomically swaps in the managed roots + config, then rewrites the config for
    this box (local ``adapter_id``, selected ``base_quant``).
    """

    migrate_done = pyqtSignal(dict)

    def __init__(self, host: str, target_server_dir: Path, base_quant: str,
                 sidecar_port: int = 8767):
        super().__init__()
        self._host = host
        self._sidecar_port = sidecar_port
        self._target = target_server_dir  # local server/ dir
        self._base_quant = base_quant

    def run(self) -> None:
        import urllib.request
        import tarfile
        import shutil
        import tempfile

        url = f"http://{self._host}:{self._sidecar_port}/export"
        try:
            self._target.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".migrate-", dir=self._target))
            staging_root = staging.resolve()
            files = 0
            try:
                # Stream the tar directly off the socket (mode "r|" = uncompressed,
                # sequential). Validate each member's path before extracting — a
                # single-pass stream can't pre-scan all members like the buffered
                # /artifacts path does.
                with urllib.request.urlopen(url, timeout=1800) as resp:
                    with tarfile.open(fileobj=resp, mode="r|") as tar:
                        for member in tar:
                            dest = (staging / member.name).resolve()
                            if dest != staging_root and not str(dest).startswith(
                                str(staging_root) + os.sep
                            ):
                                raise ValueError(f"unsafe path in archive: {member.name}")
                            tar.extract(member, staging)
                            if member.isfile():
                                files += 1

                # Atomically replace each managed root: drop the stale local copy,
                # then swap in the freshly-staged tree.
                replaced = []
                for root in _MANAGED_ROOTS:
                    src = staging / root
                    if not src.exists():
                        continue
                    dst = self._target / root
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    elif dst.is_dir():
                        shutil.rmtree(dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    replaced.append(root)

                # server_config.json is a single-file member at the bundle root; it
                # lands at the SERVER root locally (moved up out of inference/ on
                # 2026-07-28 — the bundle layout itself is unchanged).
                staged_cfg = staging / "server_config.json"
                cfg_dst = self._target / "server_config.json"
                config_summary: dict = {}
                if staged_cfg.is_file():
                    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged_cfg), str(cfg_dst))
                    replaced.append("server_config.json")
                    # Drop a pre-move legacy copy so the box has ONE config, not a
                    # stale second pointer at the old inference/ location.
                    legacy_cfg = self._target / "inference" / "server_config.json"
                    if legacy_cfg.is_file():
                        legacy_cfg.unlink()
                    config_summary = self._rewrite_config(cfg_dst)
                else:
                    config_summary = {"warning": "no server_config.json in bundle"}
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            self.migrate_done.emit({
                "type": "success", "files": files, "roots": replaced,
                "config": config_summary,
            })
        except Exception as exc:
            self.migrate_done.emit({"type": "error", "message": str(exc)})

    def _rewrite_config(self, cfg_path: Path) -> dict:
        """Repoint adapter_id at the local models/ copy and set base_quant.

        The source stored adapter_id as an ABSOLUTE path on its own box; since we
        copied the whole models/ lineage the adapter basename is preserved, so the
        local path is ``<server>/models/<basename>``. model_id / context_length /
        consolidation are kept verbatim (tokenizer + LoRA compat).
        """
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        src_adapter = cfg.get("adapter_id")
        local_adapter = None
        adapter_present = False
        if src_adapter:
            base = os.path.basename(str(src_adapter).rstrip("/"))
            local_adapter = (self._target / "models" / base).resolve()
            cfg["adapter_id"] = str(local_adapter)
            adapter_present = local_adapter.exists()

        cfg["base_quant"] = self._base_quant

        # Atomic write — a torn write corrupts the only pointer to base + adapter.
        import tempfile as _tf
        text = json.dumps(cfg, indent=2) + "\n"
        tfd, tpath = _tf.mkstemp(dir=str(cfg_path.parent), prefix=".server_config.", suffix=".tmp")
        try:
            with os.fdopen(tfd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tpath, cfg_path)
        except Exception:
            try:
                os.unlink(tpath)
            except OSError:
                pass
            raise

        return {
            "model_id": cfg.get("model_id"),
            "adapter_id": cfg.get("adapter_id"),
            "adapter_present": adapter_present,
            "base_quant": cfg.get("base_quant"),
            "context_length": cfg.get("context_length"),
        }


class ChatSyncWorker(QThread):
    """Two-way MERGE of chat transcripts between this checkout and the source box.

    Unlike ``MigrateWorker`` (a full clone that REPLACES the local subtrees), this
    only moves the chats each side is missing — or has an OLDER copy of — and never
    blindly overwrites or deletes:

      1. read the local chat stems + per-stem fingerprints (hot + archive) off disk;
      2. fetch the source's stems + fingerprints (``GET /chats/manifest``);
      3. **pull** stems the source has but we lack, plus stems present on both where
         the source's transcript is a pure append onto ours (ours a strict prefix) —
         download the source bundle (``GET /chats/export``) and write those files into
         local ``hot/chats`` (updates overwrite our shorter copy);
      4. **push** stems we have but the source lacks, plus stems present on both where
         OUR transcript is a pure append onto theirs — tar them up and POST to
         ``/chats/import`` (the server re-verifies and updates its shorter copy).

    A chat is identified by its stem (the timestamp before the first dot); the
    transcript and its ``.state.json`` / ``.shareml.json`` companions share it and
    move together. The **fingerprint** is the transcript's ordered per-exchange token
    list (``exchange_id``, else a content hash); one side being a strict prefix of the
    other means a pure append (the longer wins). A chat appended to on BOTH boxes
    independently is a **divergence** we cannot safely merge — it is left untouched on
    both sides and reported as a conflict. Newly-merged / updated chats land in
    ``hot/chats`` — they must be re-reflected on the box that received them.
    """

    sync_done = pyqtSignal(dict)

    def __init__(self, host: str, local_server_dir: Path, sidecar_port: int = 8767):
        super().__init__()
        self._host = host
        self._sidecar_port = sidecar_port
        self._local = local_server_dir  # local server/ dir

    @staticmethod
    def _stems(directory: Path) -> set:
        stems = set()
        if directory.is_dir():
            for f in directory.iterdir():
                if f.is_file() and f.name.endswith(".json"):
                    stems.add(f.name.split(".", 1)[0])
        return stems

    @staticmethod
    def _tokens_from_data(data: dict) -> list:
        """Ordered per-exchange identity tokens (mirror of the server helper)."""
        tokens = []
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

    @classmethod
    def _tokens_from_bytes(cls, raw: bytes) -> list:
        try:
            data = json.loads(raw)
        except Exception:
            return []
        return cls._tokens_from_data(data) if isinstance(data, dict) else []

    @classmethod
    def _fingerprint(cls, path: Path) -> list:
        try:
            return cls._tokens_from_bytes(path.read_bytes())
        except Exception:
            return []

    @staticmethod
    def _is_strict_prefix(a: list, b: list) -> bool:
        """True iff *a* is a strict prefix of *b* (b is a pure append onto a)."""
        return len(a) < len(b) and b[: len(a)] == a

    def run(self) -> None:
        import urllib.request
        import tarfile

        base = f"http://{self._host}:{self._sidecar_port}"
        # Chats moved to the ordered server/data/chats home (flat; the archive/ split
        # is retired — server-side _chats_manifest reports [] for it). Mirror the
        # server's layout so stems line up: local hot == server/data/chats, and a
        # pulled chat lands there so it re-enters RAG on this box.
        hot = self._local / "data" / "chats"
        arch = self._local / "data" / "archive" / "chats"  # legacy; .is_dir()-guarded
        try:
            local_hot = self._stems(hot)
            local_arch = self._stems(arch)
            local_all = local_hot | local_arch
            local_fp = {}
            for d, stems in ((arch, local_arch), (hot, local_hot)):  # hot wins
                for stem in stems:
                    local_fp[stem] = self._fingerprint(d / f"{stem}.json")

            with urllib.request.urlopen(base + "/chats/manifest", timeout=30) as resp:
                manifest = json.loads(resp.read())
            remote_hot = set(manifest.get("hot", []))
            remote_arch = set(manifest.get("archive", []))
            remote_all = remote_hot | remote_arch
            remote_fp = manifest.get("fingerprints", {}) or {}

            to_pull = remote_all - local_all   # new remote-only
            to_push = local_all - remote_all   # new local-only

            # Present on both: reconcile by the prefix relation. A pure append wins;
            # an independent divergence is a conflict (left untouched on both sides).
            pull_update = set()   # remote appended onto our hot copy → overwrite local
            push_update = set()   # we appended onto the remote hot copy → push update
            conflicts = set()
            for stem in (local_all & remote_all):
                ours = local_fp.get(stem, [])
                theirs = remote_fp.get(stem, [])
                if self._is_strict_prefix(ours, theirs):
                    if stem in local_hot:  # never overwrite a frozen/archived local copy
                        pull_update.add(stem)
                elif self._is_strict_prefix(theirs, ours):
                    if stem in remote_hot:  # server only updates its hot copies
                        push_update.add(stem)
                elif ours != theirs:
                    conflicts.add(stem)

            # ---- PULL: fetch the source's chat bundle, write new + updated stems ----
            pulled_new = set()
            pulled_updated = set()
            want = to_pull | pull_update
            if want:
                with urllib.request.urlopen(base + "/chats/export", timeout=600) as resp:
                    blob = resp.read()
                hot.mkdir(parents=True, exist_ok=True)
                groups: dict = {}
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                    for m in tar.getmembers():
                        if not m.isfile():
                            continue
                        name = os.path.basename(m.name)
                        if not name.endswith(".json"):
                            continue
                        stem = name.split(".", 1)[0]
                        if stem not in want:
                            continue
                        src = tar.extractfile(m)
                        if src is None:
                            continue
                        groups.setdefault(stem, []).append((name, src.read()))
                for stem, files in groups.items():
                    if stem in to_pull:
                        for name, raw in files:
                            (hot / name).write_bytes(raw)
                        pulled_new.add(stem)
                        continue
                    # pull_update: re-verify against the freshly downloaded transcript
                    # (race-safe vs. the manifest) before overwriting our shorter copy.
                    transcript = next(
                        (raw for name, raw in files if name == f"{stem}.json"), None)
                    if transcript is None:
                        continue
                    ours = self._fingerprint(hot / f"{stem}.json")
                    if not self._is_strict_prefix(ours, self._tokens_from_bytes(transcript)):
                        continue
                    for name, raw in files:
                        (hot / name).write_bytes(raw)
                    # The transcript grew, so a local sidecar built against the shorter
                    # version is stale; drop one the incoming didn't replace.
                    incoming_names = {name for name, _ in files}
                    sidecar = hot / f"{stem}.state.json"
                    if f"{stem}.state.json" not in incoming_names and sidecar.exists():
                        try:
                            sidecar.unlink()
                        except OSError:
                            pass
                    pulled_updated.add(stem)

            # ---- PUSH: tar the new + updated stems, POST them for import ----
            pushed_imported: list = []
            pushed_updated: list = []
            pushed_skipped: list = []
            push_want = to_push | push_update
            if push_want:
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    for d in (hot, arch):
                        if not d.is_dir():
                            continue
                        for f in sorted(d.iterdir()):
                            if not (f.is_file() and f.name.endswith(".json")):
                                continue
                            if f.name.split(".", 1)[0] not in push_want:
                                continue
                            tar.add(str(f), arcname=f.name)
                req = urllib.request.Request(
                    base + "/chats/import", data=buf.getvalue(), method="POST",
                    headers={"Content-Type": "application/gzip"},
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    result = json.loads(resp.read())
                pushed_imported = result.get("imported", [])
                pushed_updated = result.get("updated", [])
                pushed_skipped = result.get("skipped", [])

            self.sync_done.emit({
                "type": "success",
                "local_total": len(local_all),
                "remote_total": len(remote_all),
                "pulled": sorted(pulled_new),
                "pulled_updated": sorted(pulled_updated),
                "pushed_imported": pushed_imported,
                "pushed_updated": pushed_updated,
                "pushed_skipped": pushed_skipped,
                "conflicts": sorted(conflicts),
            })
        except Exception as exc:
            self.sync_done.emit({"type": "error", "message": str(exc)})


class FetchSnapshotWorker(QThread):
    """Pull the source's **runnable snapshot** and overwrite this checkout with it.

    A snapshot-scoped counterpart to ``MigrateWorker``: it brings exactly what
    determines "the current Ava personality" — the active LoRA adapter **plus its
    RAG sources** (reflection memory + consolidation in ``data/``, the chat corpus in
    ``chats/``, the wander corpus in ``til/``, and the persona digest), the prompts,
    and the config — but NOT the reflections/ archive or the old adapter lineage
    (lighter than a full clone). The server side is ``server/snapshot_state.py``
    streamed over the inference sidecar (``GET /snapshot/export``), so the scope is
    the single one the CLI export uses.

    The snapshot's tar uses the portable **bundle** layout; this worker maps each
    member into the local **checkout** layout (the ordered server/data/ homes) as it
    swaps it in:

        bundle server_config.json  → server/server_config.json (adapter_id → abs)
        bundle data/               → inference/data/            (memory + consolidation)
        bundle chats/              → data/chats/                (chat corpus RAG source)
        bundle til/                → data/til/                  (wander corpus RAG source)
        bundle prompts/            → inference/prompts/
        bundle digest.json         → (provenance only — inspected, not restored; the
                                      persona digest regenerates on the next reflection run)
        bundle models/<adapter>/   → models/<adapter>/          (that adapter only)
        bundle training/<build_id>/→ models/snapshots/<build_id>/ (provenance)

    The whole snapshot config replaces the local one (model_id, base_quant, context,
    consolidation all come from the source), so — unlike the bare-adapter fetch —
    there is no base-compat guard: adapter, base, and config are internally consistent
    by construction. Only the named subtrees are replaced; other local adapters and
    the reflections/ archive are left untouched. The local server is NOT started.
    """

    fetch_done = pyqtSignal(dict)

    # Bundle-root subtree → local checkout path (relative to server/). models/ and
    # training/ are handled dynamically (their leaf name isn't known up front).
    # chats/ and til/ are RAG sources that live under server/data/ (moved out of
    # inference/data), so they map to the ordered server/data/ homes the local server reads.
    _DIR_MAP = {
        "data": "inference/data",
        "prompts": "inference/prompts",
        "chats": "data/chats",
        "til": "data/til",
    }

    def __init__(self, host: str, target_server_dir: Path, sidecar_port: int = 8767):
        super().__init__()
        self._host = host
        self._sidecar_port = sidecar_port
        self._target = target_server_dir  # local server/ dir

    def run(self) -> None:
        import urllib.request
        import tarfile
        import shutil
        import tempfile

        base = f"http://{self._host}:{self._sidecar_port}"
        try:
            # ---- 1. manifest peek ----
            with urllib.request.urlopen(base + "/snapshot/manifest", timeout=30) as resp:
                manifest = json.loads(resp.read())
            adapter = manifest.get("adapter") or None
            adapter_name = (adapter or {}).get("name")
            if adapter is not None and not manifest.get("adapter_present"):
                self.fetch_done.emit({
                    "type": "error",
                    "message": "the source's active adapter dir is missing — cannot "
                               "fetch a runnable snapshot.",
                })
                return

            # ---- 2. stream the snapshot into staging ----
            self._target.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".fetch-snapshot-", dir=self._target))
            staging_root = staging.resolve()
            files = 0
            try:
                with urllib.request.urlopen(base + "/snapshot/export", timeout=3600) as resp:
                    with tarfile.open(fileobj=resp, mode="r|") as tar:
                        for member in tar:
                            dest = (staging / member.name).resolve()
                            if dest != staging_root and not str(dest).startswith(
                                str(staging_root) + os.sep
                            ):
                                raise ValueError(f"unsafe path in archive: {member.name}")
                            tar.extract(member, staging)
                            if member.isfile():
                                files += 1

                replaced: list = []

                # 2a. fixed dir subtrees (RAG sources + prompts).
                for bundle, checkout in self._DIR_MAP.items():
                    src = staging / bundle
                    if src.is_dir():
                        self._replace(self._target / checkout, src)
                        replaced.append(checkout)

                # 2b. the active adapter — bundle models/<name> → models/<name> (only
                # that dir; sibling adapters in the local lineage are left alone).
                fetched_adapter = None
                staged_models = staging / "models"
                if staged_models.is_dir():
                    for d in sorted(staged_models.iterdir()):
                        if d.is_dir():
                            dst = self._target / "models" / d.name
                            self._replace(dst, d)
                            fetched_adapter = dst
                            replaced.append(f"models/{d.name}")

                # 2c. forensic training corpus → models/snapshots/<build_id>.
                staged_training = staging / "training"
                if staged_training.is_dir():
                    for d in sorted(staged_training.iterdir()):
                        if d.is_dir():
                            self._replace(self._target / "models" / "snapshots" / d.name, d)
                            replaced.append(f"models/snapshots/{d.name}")

                # 2d. config — replace wholesale, then rewrite adapter_id absolute.
                staged_cfg = staging / "server_config.json"
                cfg_dst = self._target / "server_config.json"
                config_summary: dict = {}
                if staged_cfg.is_file():
                    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged_cfg), str(cfg_dst))
                    replaced.append("server_config.json")
                    # Drop a pre-move legacy copy so the box has ONE config, not a
                    # stale second pointer at the old inference/ location.
                    legacy_cfg = self._target / "inference" / "server_config.json"
                    if legacy_cfg.is_file():
                        legacy_cfg.unlink()
                    config_summary = self._rewrite_config(cfg_dst, fetched_adapter)
                else:
                    config_summary = {"warning": "no server_config.json in snapshot"}
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            self.fetch_done.emit({
                "type": "success",
                "files": files,
                "roots": replaced,
                "adapter_name": adapter_name,
                "config": config_summary,
                "training": manifest.get("training_data") or {},
            })
        except Exception as exc:
            self.fetch_done.emit({"type": "error", "message": str(exc)})

    def _replace(self, dst: Path, src: Path) -> None:
        """Atomically swap the staged *src* tree in for *dst* (drop the stale copy)."""
        import shutil
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def _rewrite_config(self, cfg_path: Path, fetched_adapter: Optional[Path]) -> dict:
        """Rewrite the snapshot config's RELATIVE adapter_id to a local absolute path.

        The snapshot ships ``adapter_id = "models/<name>"`` (portable). We point it at
        the copy we just laid down. Everything else in the config is the source's and
        is kept verbatim (base model_id / base_quant / context / consolidation), so the
        adapter and its base stay consistent.
        """
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        adapter_present = False
        if cfg.get("adapter_id"):
            base = os.path.basename(str(cfg["adapter_id"]).rstrip("/"))
            local = fetched_adapter or (self._target / "models" / base)
            local = Path(local).resolve()
            cfg["adapter_id"] = str(local)
            adapter_present = local.exists()

        import tempfile as _tf
        text = json.dumps(cfg, indent=2) + "\n"
        tfd, tpath = _tf.mkstemp(dir=str(cfg_path.parent), prefix=".server_config.",
                                 suffix=".tmp")
        try:
            with os.fdopen(tfd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tpath, cfg_path)
        except Exception:
            try:
                os.unlink(tpath)
            except OSError:
                pass
            raise

        return {
            "model_id": cfg.get("model_id"),
            "adapter_id": cfg.get("adapter_id"),
            "adapter_present": adapter_present,
            "base_quant": cfg.get("base_quant"),
            "context_length": cfg.get("context_length"),
        }


class FetchAdapterWorker(QThread):
    """Pull ONLY the source's active LoRA adapter and point this box at it.

    The narrowest of the three transfer paths, and the only deliberately **hybrid**
    one. ``MigrateWorker`` and ``FetchSnapshotWorker`` both replace this box's state
    with the source's, so what arrives is internally consistent by construction. This
    one grafts foreign *weights* onto local *state*: the chat corpus, distilled memory,
    ledger, prompts, digest and persona pointer are all left exactly as they are, and
    only ``adapter_id`` moves. That is the point — reflect and accumulate here, train on
    the box with the GPU headroom, then run those weights against this box's memory.

    Because nothing else travels, the base-model compatibility that the other two get
    for free must be checked HERE, before any bytes move::

        remote MANIFEST model_id  ==  local server_config.json model_id   → proceed
                                  !=                                       → refuse

    A LoRA is fit against one specific base and nothing downstream re-checks that. At
    4-bit ``inference_backend.load`` takes the base from the adapter's own
    ``adapter_config.json``, so a mismatched adapter would quietly load a base this
    box's config never named; at 8/16-bit ``_stage_adapter_with_base`` overwrites that
    field with the local ``model_id`` and the mismatch surfaces only as an opaque
    peft/unsloth load failure. Neither is recoverable by inspection afterwards, so the
    guard refuses rather than warns and nothing is downloaded.

    On success the previous adapter dir is left in ``models/`` (the lineage is the
    rollback path: repoint ``adapter_id`` back by hand). The local server is NOT
    restarted — the fetched adapter is picked up on the next model load.
    """

    fetch_done = pyqtSignal(dict)

    def __init__(self, host: str, target_server_dir: Path, sidecar_port: int = 8767):
        super().__init__()
        self._host = host
        self._sidecar_port = sidecar_port
        self._target = target_server_dir  # local server/ dir

    def _local_model_id(self) -> Optional[str]:
        """This box's base ``model_id``, or None when the config is absent/unreadable.

        Read straight off disk rather than from the connected server's status, because
        the config is what the guard is protecting: the local server may not even be
        running, and if it is, it may be serving a model loaded before the last edit.
        """
        cfg_path = self._target / "server_config.json"
        if not cfg_path.is_file():
            # Pre-2026-07-28 layout; the same fallback every other reader keeps.
            legacy = self._target / "inference" / "server_config.json"
            cfg_path = legacy if legacy.is_file() else cfg_path
        try:
            return (json.loads(cfg_path.read_text(encoding="utf-8"))
                    .get("model_id") or "").strip() or None
        except Exception:
            return None

    def run(self) -> None:
        import urllib.request
        import tarfile
        import shutil
        import tempfile

        base = f"http://{self._host}:{self._sidecar_port}"
        try:
            # ---- 1. manifest peek: existence + base-compat, before any weights move ----
            with urllib.request.urlopen(base + "/adapter/manifest", timeout=30) as resp:
                manifest = json.loads(resp.read())

            adapter = manifest.get("adapter") or None
            if adapter is None:
                self.fetch_done.emit({
                    "type": "error",
                    "message": "the source has no active adapter (it is running the bare "
                               "base model) — there is nothing to fetch.",
                })
                return
            if not manifest.get("adapter_present"):
                self.fetch_done.emit({
                    "type": "error",
                    "message": "the source's adapter_id points at a missing dir — "
                               "nothing to fetch.",
                })
                return

            remote_model = (manifest.get("model_id") or "").strip()
            local_model = self._local_model_id()
            if not local_model:
                self.fetch_done.emit({
                    "type": "error",
                    "message": "cannot read this checkout's server_config.json model_id, "
                               "so the adapter's base cannot be verified. Refusing.",
                })
                return
            if remote_model != local_model:
                self.fetch_done.emit({
                    "type": "error",
                    "message": (
                        "base model MISMATCH — nothing was downloaded.\n\n"
                        f"  local  model_id: {local_model}\n"
                        f"  remote model_id: {remote_model or '(unset)'}\n\n"
                        "A LoRA is fit against one specific base; this adapter cannot be "
                        "applied here. Point both boxes at the same base model_id first."
                    ),
                })
                return

            adapter_name = adapter.get("name")
            adapter_rel = adapter.get("path") or f"models/{adapter_name}"

            # ---- 2. stream the adapter into staging ----
            self._target.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".fetch-adapter-", dir=self._target))
            staging_root = staging.resolve()
            files = 0
            try:
                with urllib.request.urlopen(base + "/adapter/export", timeout=3600) as resp:
                    with tarfile.open(fileobj=resp, mode="r|") as tar:
                        for member in tar:
                            dest = (staging / member.name).resolve()
                            if dest != staging_root and not str(dest).startswith(
                                str(staging_root) + os.sep
                            ):
                                raise ValueError(f"unsafe path in archive: {member.name}")
                            tar.extract(member, staging)
                            if member.isfile():
                                files += 1

                # ---- 3. swap the adapter dir in (only that one; the rest of the
                #         local lineage is the rollback path and is left alone) ----
                staged_models = staging / "models"
                fetched_adapter: Optional[Path] = None
                if staged_models.is_dir():
                    for d in sorted(staged_models.iterdir()):
                        if d.is_dir():
                            dst = self._target / "models" / d.name
                            if dst.is_symlink() or dst.is_file():
                                dst.unlink()
                            elif dst.is_dir():
                                shutil.rmtree(dst)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(d), str(dst))
                            fetched_adapter = dst
                if fetched_adapter is None:
                    raise ValueError("the adapter bundle contained no models/<adapter> dir")

                # ---- 4. repoint adapter_id ONLY (everything else stays local) ----
                config_summary = self._repoint_adapter(fetched_adapter)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            self.fetch_done.emit({
                "type": "success",
                "files": files,
                "adapter_name": adapter_name,
                "adapter_rel": adapter_rel,
                "size_bytes": manifest.get("size_bytes") or 0,
                "model_id": remote_model,
                "build_id": manifest.get("build_id"),
                "source_host": manifest.get("source_host"),
                "training": manifest.get("training_data") or {},
                "config": config_summary,
            })
        except Exception as exc:
            self.fetch_done.emit({"type": "error", "message": str(exc)})

    def _repoint_adapter(self, fetched_adapter: Path) -> dict:
        """Set the LOCAL config's ``adapter_id`` to the fetched dir, touching nothing else.

        The inverse of ``FetchSnapshotWorker._rewrite_config``: there, the whole config is
        the source's and only the (relative) adapter path needs localizing; here the whole
        config is *ours* and only the adapter changes. ``model_id`` in particular is left
        alone — it was already verified equal, and rewriting it would quietly turn a
        refused-mismatch into an accepted one on the next fetch.
        """
        cfg_path = self._target / "server_config.json"
        if not cfg_path.is_file():
            legacy = self._target / "inference" / "server_config.json"
            if legacy.is_file():
                cfg_path = legacy
            else:
                raise FileNotFoundError("no local server_config.json to repoint")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        previous = cfg.get("adapter_id")
        cfg["adapter_id"] = str(Path(fetched_adapter).resolve())

        import tempfile as _tf
        text = json.dumps(cfg, indent=2) + "\n"
        tfd, tpath = _tf.mkstemp(dir=str(cfg_path.parent), prefix=".server_config.",
                                 suffix=".tmp")
        try:
            with os.fdopen(tfd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tpath, cfg_path)
        except Exception:
            try:
                os.unlink(tpath)
            except OSError:
                pass
            raise

        return {
            "model_id": cfg.get("model_id"),
            "adapter_id": cfg.get("adapter_id"),
            "previous_adapter_id": previous,
            "base_quant": cfg.get("base_quant"),
            "context_length": cfg.get("context_length"),
        }


class PrecisionWorker(QThread):
    """Persists the connected server's base_quant off-thread (no migration).

    Wraps ``BackendClient.set_precision`` → inference sidecar ``POST /precision``,
    which rewrites ``server_config.json`` without restarting inference; the change
    takes effect on the next server restart.
    """

    precision_done = pyqtSignal(dict)

    def __init__(self, client, base_quant: str):
        super().__init__()
        self._client = client
        self._base_quant = base_quant

    def run(self) -> None:
        try:
            result = self._client.set_precision(self._base_quant)
        except Exception as exc:
            result = {"error": str(exc)}
        self.precision_done.emit(result)


class MigrateWidget(QWidget):
    """Migrate tab — clone the connected server's full state onto this box."""

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self.text_font = QFont("Courier")
        self._source_worker: Optional[MigrateSourceWorker] = None
        self._migrate_worker: Optional[MigrateWorker] = None
        self._sync_worker: Optional[ChatSyncWorker] = None
        self._fetch_worker: Optional[FetchSnapshotWorker] = None
        self._precision_worker: Optional[PrecisionWorker] = None
        self._same_box = False  # source == this checkout → clone would be a no-op
        self._source_ok = False  # last source inspection succeeded (server reachable)
        self._build_ui()

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._lbl_status = QLabel("Connect in the Chat tab, then Refresh to inspect the source.")
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_source)
        header.addWidget(self._lbl_status, stretch=1)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        # Quant selector + Change-precision / Migrate button row.
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Base precision:"))
        self.cmb_quant = QComboBox()
        for label, value in _QUANT_OPTIONS:
            self.cmb_quant.addItem(label, value)
        self.cmb_quant.setToolTip(
            "Base-model load precision. Used two ways: as the precision written into "
            "the migrated config (Migrate…), and as the precision to set on the "
            "connected server without migrating (Change precision…). The model_id is "
            "kept verbatim (tokenizer + LoRA compat); this only changes load precision. "
            "16-bit is the roomy-box option."
        )
        controls.addWidget(self.cmb_quant)
        self.btn_precision = QPushButton("Change precision…")
        self.btn_precision.setEnabled(False)
        self.btn_precision.setToolTip(
            "Set base_quant on the CONNECTED server without migrating any data. The "
            "running server is not restarted — the new precision takes effect on the "
            "next server restart."
        )
        self.btn_precision.clicked.connect(self._on_precision_clicked)
        controls.addWidget(self.btn_precision)
        controls.addStretch(1)
        self.btn_merge_chats = QPushButton("Merge chats…")
        self.btn_merge_chats.setEnabled(False)
        self.btn_merge_chats.setToolTip(
            "Two-way MERGE of chat transcripts with the connected server: pull chats "
            "it has that this checkout lacks, push chats this checkout has that it "
            "lacks. Non-destructive (nothing is overwritten or deleted). Merged chats "
            "land in the receiving box's hot/chats and must be re-reflected there "
            "(run a Sleep pass). Weights/RAG are NOT synced — only the raw transcripts."
        )
        self.btn_merge_chats.clicked.connect(self._on_merge_chats_clicked)
        controls.addWidget(self.btn_merge_chats)
        self.btn_fetch_snapshot = QPushButton("Fetch snapshot…")
        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_snapshot.setToolTip(
            "Pull the connected server's runnable snapshot — the active LoRA adapter "
            "PLUS its RAG sources (the whole data/ tree: reflection memory, persona "
            "digest, chat corpus), prompts, and config — and overwrite this checkout "
            "with it, so the local server boots as the current Ava. Snapshot-scoped: "
            "unlike Migrate it skips the reflections/ archive and the old adapter "
            "lineage. Replaces local inference/data, prompts, the active adapter, and "
            "server_config.json. The local server is NOT started."
        )
        self.btn_fetch_snapshot.clicked.connect(self._on_fetch_snapshot_clicked)
        controls.addWidget(self.btn_fetch_snapshot)
        self.btn_fetch_adapter = QPushButton("Fetch adapter…")
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_fetch_adapter.setToolTip(
            "Pull ONLY the connected server's active LoRA adapter and point this box at "
            "it — a HYBRID: this checkout keeps its own chat corpus, distilled memory, "
            "ledger, prompts, digest and persona, and just runs the remote weights. Use "
            "when reflection accumulates here but training happens on the GPU box.\n\n"
            "Refuses outright if the two boxes' base model_id differs — a LoRA is fit "
            "against one base, and nothing downstream re-checks it.\n\n"
            "Only server_config.json's adapter_id changes; the previous adapter stays in "
            "models/ as the rollback path. The local server is NOT restarted."
        )
        self.btn_fetch_adapter.clicked.connect(self._on_fetch_adapter_clicked)
        controls.addWidget(self.btn_fetch_adapter)
        self.btn_migrate = QPushButton("Migrate…")
        self.btn_migrate.setEnabled(False)
        self.btn_migrate.setToolTip(
            "Pull the complete server state (reflection working state, chat corpus, "
            "wander corpus, persona lineage, reflection archive, adapter weights, "
            "config) into this checkout and prepare it for a local server."
        )
        self.btn_migrate.clicked.connect(self._on_migrate_clicked)
        controls.addWidget(self.btn_migrate)
        layout.addLayout(controls)

        # All three transfer paths (Migrate / Merge chats / Fetch snapshot) now
        # understand the ordered server/data/ layout — the clone bundle carries
        # server/data (chats + til + persona lineage) and this widget maps it back into
        # the checkout — so nothing stays frozen. They enable/disable via the normal
        # source-inspection flow (_on_source_status) like any other reachable-source gate.

        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(self.text_font)
        self.txt_output.setPlaceholderText(
            "Source status and migration progress appear here."
        )
        layout.addWidget(self.txt_output, stretch=1)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.refresh_source()

    # ---------------------------------------------------------------- #
    # Source inspection                                                 #
    # ---------------------------------------------------------------- #

    def refresh_source(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            self.btn_migrate.setEnabled(False)
            return
        host = self._chat_widget.txt_host.text().strip() or "localhost"
        self._lbl_status.setText("Inspecting source…")
        self.btn_refresh.setEnabled(False)
        self._source_worker = MigrateSourceWorker(host, self._chat_widget._get_mgmt_port())
        self._source_worker.status_done.connect(self._on_source_status)
        self._source_worker.start()

    def _on_source_status(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        if not self._chat_widget._client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            self.btn_migrate.setEnabled(False)
            self.btn_precision.setEnabled(False)
            self.btn_merge_chats.setEnabled(False)
            self.btn_fetch_snapshot.setEnabled(False)
            self.btn_fetch_adapter.setEnabled(False)
            self._source_ok = False
            return

        if result.get("type") != "success":
            self._lbl_status.setText(f"Could not reach source watchdog: {result.get('message')}")
            self.btn_migrate.setEnabled(False)
            self.btn_precision.setEnabled(False)
            self.btn_merge_chats.setEnabled(False)
            self.btn_fetch_snapshot.setEnabled(False)
            self.btn_fetch_adapter.setEnabled(False)
            self._source_ok = False
            return

        self._source_ok = True
        data = result.get("data", {})
        server_host = data.get("hostname") or "?"
        server_data_dir = data.get("data_dir") or "?"
        # base_quant travels on the WebSocket chat status now (the inference server
        # owns server_config.json, not the watchdog), so read it from the connected
        # client's stashed status rather than the watchdog /status.
        server_quant = getattr(self._chat_widget, "_server_base_quant", "") or None
        # Preselect the selector to the source's current precision so both Migrate
        # and Change-precision default to "keep as-is".
        if server_quant:
            idx = self.cmb_quant.findData(server_quant)
            if idx >= 0:
                self.cmb_quant.setCurrentIndex(idx)

        # Same-box guard: cloning onto our own checkout would delete the source
        # while replacing it with itself. Refuse it.
        import socket
        local_data = self._chat_widget._local_inference_dir() / "data"
        self._same_box = (
            server_host == socket.gethostname()
            and server_data_dir != "?"
            and os.path.realpath(server_data_dir) == os.path.realpath(str(local_data))
        )

        model_id = getattr(self._chat_widget, "_server_model_id", "") or "—"
        adapter_id = getattr(self._chat_widget, "_server_adapter_id", "") or "(none)"
        host = self._chat_widget.txt_host.text().strip() or "localhost"

        lines = [
            "Source server:",
            f"  host:       {server_host}  ({host})",
            f"  data_dir:   {server_data_dir}",
            f"  model_id:   {model_id}",
            f"  adapter_id: {adapter_id}",
            f"  base_quant: {server_quant or '(default 4bit)'}",
            "",
            "Target (this checkout):",
            f"  server/:    {self._chat_widget._local_server_dir()}",
        ]
        # Change-precision acts on the connected server, so it's available whenever
        # the source is reachable — independent of the same-box migrate guard.
        self.btn_precision.setEnabled(True)
        # Chat merge needs a distinct remote box; a same-box merge is a no-op.
        self.btn_merge_chats.setEnabled(not self._same_box)
        # Fetching the adapter onto our own checkout would copy it onto itself and
        # repoint the config at its own copy — a no-op. Distinct box only.
        self.btn_fetch_snapshot.setEnabled(not self._same_box)
        self.btn_fetch_adapter.setEnabled(not self._same_box)
        if self._same_box:
            lines += [
                "",
                "⚠ Source shares this checkout's storage — a clone here is a no-op "
                "(it would replace the source with itself). Migrate is disabled.",
            ]
            self.btn_migrate.setEnabled(False)
            self._lbl_status.setText("Source == this box — nothing to migrate.")
        else:
            self.btn_migrate.setEnabled(True)
            self._lbl_status.setText("Ready to migrate.")
        self.txt_output.setPlainText("\n".join(lines))

    # ---------------------------------------------------------------- #
    # Migration                                                         #
    # ---------------------------------------------------------------- #

    def _on_migrate_clicked(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._same_box:
            return  # guarded above; belt-and-suspenders

        host = self._chat_widget.txt_host.text().strip() or "localhost"
        local_server = self._chat_widget._local_server_dir()
        quant_label = self.cmb_quant.currentText()
        base_quant = self.cmb_quant.currentData()

        reply = QMessageBox.warning(
            self,
            "Migrate Ava",
            f"Clone the full server state from {host} into\n{local_server}?\n\n"
            "This REPLACES the local inference/data (reflection state), data "
            "(chats + wander + persona lineage), reflections and models subtrees "
            "(and server_config.json) with the source's copy — including the LoRA "
            "adapter weights.\n\n"
            f"The base model will be configured to load at {quant_label} on this box "
            "(the source model_id is kept for tokenizer/LoRA compatibility).\n\n"
            "The local server is NOT started — you'll launch it afterward.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_migrate.setEnabled(False)
        self.btn_precision.setEnabled(False)
        self.btn_merge_chats.setEnabled(False)
        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Migrating — streaming the clone bundle…")
        self.txt_output.appendPlainText(f"\n[migrate] Pulling from {host} at {quant_label}…")
        self._migrate_worker = MigrateWorker(
            host, local_server, base_quant, self._chat_widget._get_sidecar_port()
        )
        self._migrate_worker.migrate_done.connect(self._on_migrate_done)
        self._migrate_worker.start()

    def _on_migrate_done(self, result: dict) -> None:
        self.btn_migrate.setEnabled(True)
        self.btn_precision.setEnabled(self._source_ok)
        self.btn_merge_chats.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_snapshot.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_adapter.setEnabled(self._source_ok and not self._same_box)
        self.btn_refresh.setEnabled(True)

        if result.get("type") != "success":
            self._lbl_status.setText(f"Migration failed: {result.get('message')}")
            self.txt_output.appendPlainText(
                f"✗ Migration failed: {result.get('message', 'Unknown error')}"
            )
            QMessageBox.critical(
                self, "Migration failed",
                f"The clone did not complete:\n\n{result.get('message', 'Unknown error')}",
            )
            return

        roots = ", ".join(result.get("roots", [])) or "—"
        cfg = result.get("config", {})
        local_server = self._chat_widget._local_server_dir()
        self._lbl_status.setText("Migration complete — start the local server to continue.")

        lines = [
            f"✓ Migrated {result.get('files', 0)} file(s) into {local_server}",
            f"  replaced: {roots}",
            "",
            "server_config.json:",
            f"  model_id:       {cfg.get('model_id')}",
            f"  adapter_id:     {cfg.get('adapter_id')}",
            f"  adapter_present:{cfg.get('adapter_present')}",
            f"  base_quant:     {cfg.get('base_quant')}",
            f"  context_length: {cfg.get('context_length')}",
        ]
        if cfg.get("warning"):
            lines.append(f"  ⚠ {cfg['warning']}")
        if cfg.get("adapter_id") and not cfg.get("adapter_present"):
            lines.append(
                "  ⚠ adapter dir not found locally after copy — check the models/ subtree."
            )
        lines += [
            "",
            "Next steps:",
            "  1. cd server && .venv/bin/python watchdog.py",
            "  2. point the client at ws://localhost:8765 (Chat tab)",
            "  3. load the model — the migrated config drives model_id + adapter + precision.",
        ]
        self.txt_output.appendPlainText("\n".join(lines))

        QMessageBox.information(
            self, "Migration complete",
            "Ava's full state is now in this checkout.\n\n"
            "Start the local server (cd server && .venv/bin/python watchdog.py), then "
            "connect the client to ws://localhost:8765 and load the model from the "
            "Chat tab. The base will load at the selected precision.",
        )

    # ---------------------------------------------------------------- #
    # Chat merge (two-way transcript sync, non-destructive)             #
    # ---------------------------------------------------------------- #

    def _on_merge_chats_clicked(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._same_box:
            return  # guarded above; belt-and-suspenders

        host = self._chat_widget.txt_host.text().strip() or "localhost"
        local_server = self._chat_widget._local_server_dir()

        reply = QMessageBox.question(
            self,
            "Merge chats",
            f"Two-way merge of chat transcripts between {host} and\n{local_server}?\n\n"
            "This pulls chats the server has that this checkout lacks, and pushes "
            "chats this checkout has that the server lacks. A chat present on both "
            "sides that GREW on one box (e.g. a reply appended to an Ava-initiated "
            "chat) updates the shorter copy; a chat appended to on both boxes "
            "independently is a conflict and is left untouched on both sides.\n\n"
            "Only the raw transcripts are synced (not weights or RAG). Chats new to or "
            "updated on a box must be re-reflected there (run a Sleep pass on the "
            "receiving box).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_merge_chats.setEnabled(False)
        self.btn_migrate.setEnabled(False)
        self.btn_precision.setEnabled(False)
        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Merging chats…")
        self.txt_output.appendPlainText(f"\n[merge] Syncing chats with {host}…")
        self._sync_worker = ChatSyncWorker(
            host, local_server, self._chat_widget._get_sidecar_port()
        )
        self._sync_worker.sync_done.connect(self._on_merge_chats_done)
        self._sync_worker.start()

    def _on_merge_chats_done(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_precision.setEnabled(self._source_ok)
        self.btn_migrate.setEnabled(self._source_ok and not self._same_box)
        self.btn_merge_chats.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_snapshot.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_adapter.setEnabled(self._source_ok and not self._same_box)

        if result.get("type") != "success":
            self._lbl_status.setText(f"Chat merge failed: {result.get('message')}")
            self.txt_output.appendPlainText(
                f"✗ Chat merge failed: {result.get('message', 'Unknown error')}"
            )
            QMessageBox.critical(
                self, "Chat merge failed",
                f"The merge did not complete:\n\n{result.get('message', 'Unknown error')}",
            )
            return

        pulled = result.get("pulled", [])
        pulled_upd = result.get("pulled_updated", [])
        imported = result.get("pushed_imported", [])
        pushed_upd = result.get("pushed_updated", [])
        skipped = result.get("pushed_skipped", [])
        conflicts = result.get("conflicts", [])
        n_pull = len(pulled) + len(pulled_upd)
        n_push = len(imported) + len(pushed_upd)
        self._lbl_status.setText(
            f"Chat merge complete — pulled {n_pull}, pushed {n_push}"
            + (f", {len(conflicts)} conflict(s)." if conflicts else ".")
        )
        lines = [
            "✓ Chat merge complete.",
            f"  local chats:   {result.get('local_total', '?')}",
            f"  server chats:  {result.get('remote_total', '?')}",
            f"  ← pulled to local hot/chats:  {len(pulled)} new"
            + (f", {len(pulled_upd)} updated (appended)" if pulled_upd else ""),
            f"  → pushed to server hot/chats: {len(imported)} new"
            + (f", {len(pushed_upd)} updated (appended)" if pushed_upd else ""),
        ]
        if skipped:
            lines.append(f"  (server left {len(skipped)} pushed stem(s) unchanged — already had them)")
        if conflicts:
            lines += [
                "",
                f"⚠ {len(conflicts)} chat(s) were appended to on BOTH boxes independently "
                "and could not be merged — left untouched on both sides:",
            ]
            lines += [f"    • {c}" for c in conflicts]
        if n_pull or n_push:
            lines += [
                "",
                "New / updated chats need re-reflection on the box that received them:",
            ]
            if pulled or pulled_upd:
                lines.append("  • local: run a Sleep pass here (start the local server first).")
            if imported or pushed_upd:
                lines.append("  • server: run a Sleep pass on it (Sleep tab).")
        elif not conflicts:
            lines.append("  Already in sync — nothing to move.")
        self.txt_output.appendPlainText("\n".join(lines))

    # ---------------------------------------------------------------- #
    # Fetch snapshot (adapter + RAG sources — overwrite the checkout)   #
    # ---------------------------------------------------------------- #

    def _on_fetch_snapshot_clicked(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._same_box:
            return  # guarded above; belt-and-suspenders

        host = self._chat_widget.txt_host.text().strip() or "localhost"
        local_server = self._chat_widget._local_server_dir()
        adapter_id = getattr(self._chat_widget, "_server_adapter_id", "") or "(none)"

        reply = QMessageBox.warning(
            self,
            "Fetch snapshot",
            f"Pull the connected server's runnable snapshot into\n{local_server}?\n\n"
            f"  source adapter_id: {adapter_id}\n\n"
            "This brings the current Ava personality — the active LoRA adapter PLUS "
            "its RAG sources (reflection memory, chat corpus, wander corpus, persona "
            "digest), the prompts, and the config — and REPLACES this checkout's "
            "inference/data, data/chats, data/til, inference/prompts, the active "
            "adapter, and server_config.json with the source's.\n\n"
            "Snapshot-scoped: it does NOT bring the reflections/ archive or the old "
            "adapter lineage (lighter than Migrate). Other local adapters are left "
            "untouched.\n\n"
            "The local server is NOT started — reload/launch it afterward to boot as "
            "the current Ava.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_migrate.setEnabled(False)
        self.btn_precision.setEnabled(False)
        self.btn_merge_chats.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Fetching snapshot…")
        self.txt_output.appendPlainText(f"\n[snapshot] Fetching runnable snapshot from {host}…")
        self._fetch_worker = FetchSnapshotWorker(
            host, local_server, self._chat_widget._get_sidecar_port()
        )
        self._fetch_worker.fetch_done.connect(self._on_fetch_snapshot_done)
        self._fetch_worker.start()

    def _on_fetch_snapshot_done(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_precision.setEnabled(self._source_ok)
        self.btn_migrate.setEnabled(self._source_ok and not self._same_box)
        self.btn_merge_chats.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_snapshot.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_adapter.setEnabled(self._source_ok and not self._same_box)

        if result.get("type") != "success":
            self._lbl_status.setText(f"Snapshot fetch failed: {result.get('message')}")
            self.txt_output.appendPlainText(
                f"✗ Snapshot fetch failed: {result.get('message', 'Unknown error')}"
            )
            QMessageBox.critical(
                self, "Snapshot fetch failed",
                f"The snapshot fetch did not complete:\n\n{result.get('message', 'Unknown error')}",
            )
            return

        roots = ", ".join(result.get("roots", [])) or "—"
        cfg = result.get("config", {})
        training = result.get("training", {})
        local_server = self._chat_widget._local_server_dir()
        self._lbl_status.setText("Snapshot fetched — start/reload the local server to boot as current Ava.")

        lines = [
            f"✓ Fetched runnable snapshot ({result.get('files', 0)} file(s)) into {local_server}",
            f"  replaced: {roots}",
            "",
            "server_config.json:",
            f"  model_id:        {cfg.get('model_id')}",
            f"  adapter_id:      {cfg.get('adapter_id')}",
            f"  adapter_present: {cfg.get('adapter_present')}",
            f"  base_quant:      {cfg.get('base_quant')}",
            f"  context_length:  {cfg.get('context_length')}",
        ]
        if cfg.get("warning"):
            lines.append(f"  ⚠ {cfg['warning']}")
        if cfg.get("adapter_id") and not cfg.get("adapter_present"):
            lines.append("  ⚠ adapter dir not found locally after copy — check models/.")
        if training.get("included"):
            lines.append(
                f"  training corpus: {training.get('build_id')} "
                f"({training.get('rows')} rows) → models/snapshots/"
            )
        lines += [
            "",
            "Next steps:",
            "  1. cd server && .venv/bin/python watchdog.py  (if not already running)",
            "  2. point the client at ws://localhost:8765 (Chat tab)",
            "  3. load/reload the model — the snapshot config drives model_id + adapter.",
        ]
        self.txt_output.appendPlainText("\n".join(lines))
        QMessageBox.information(
            self, "Snapshot fetched",
            "The current Ava snapshot (adapter + RAG sources + prompts + config) is "
            "now in this checkout.\n\n"
            "Start/reload the local server and load the model from the Chat tab to "
            "boot as the current Ava.",
        )

    # ---------------------------------------------------------------- #
    # Fetch adapter (weights only — hybrid: remote weights, local state) #
    # ---------------------------------------------------------------- #

    def _on_fetch_adapter_clicked(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._same_box:
            return  # guarded above; belt-and-suspenders

        host = self._chat_widget.txt_host.text().strip() or "localhost"
        local_server = self._chat_widget._local_server_dir()
        adapter_id = getattr(self._chat_widget, "_server_adapter_id", "") or "(none)"
        remote_model = getattr(self._chat_widget, "_server_model_id", "") or "(unknown)"

        reply = QMessageBox.warning(
            self,
            "Fetch adapter",
            f"Pull ONLY the connected server's LoRA adapter into\n{local_server}?\n\n"
            f"  source adapter_id: {adapter_id}\n"
            f"  source model_id:   {remote_model}\n\n"
            "This is a HYBRID, not a clone. Nothing of this box's state is replaced: "
            "the chat corpus, distilled memory, ledger, prompts, persona digest and "
            "persona pointer all stay exactly as they are. Only the weights arrive, "
            "and only server_config.json's adapter_id is repointed at them.\n\n"
            "The fetch is REFUSED if the two boxes' base model_id differs — a LoRA is "
            "fit against one specific base, and nothing downstream re-checks it.\n\n"
            "Your current adapter stays in models/ as the rollback path (repoint "
            "adapter_id back by hand). The local server is NOT restarted — reload the "
            "model from the Chat tab to run the fetched weights.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_migrate.setEnabled(False)
        self.btn_precision.setEnabled(False)
        self.btn_merge_chats.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Fetching adapter…")
        self.txt_output.appendPlainText(f"\n[adapter] Fetching LoRA adapter from {host}…")
        self._adapter_worker = FetchAdapterWorker(
            host, local_server, self._chat_widget._get_sidecar_port()
        )
        self._adapter_worker.fetch_done.connect(self._on_fetch_adapter_done)
        self._adapter_worker.start()

    def _on_fetch_adapter_done(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_precision.setEnabled(self._source_ok)
        self.btn_migrate.setEnabled(self._source_ok and not self._same_box)
        self.btn_merge_chats.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_snapshot.setEnabled(self._source_ok and not self._same_box)
        self.btn_fetch_adapter.setEnabled(self._source_ok and not self._same_box)

        if result.get("type") != "success":
            msg = result.get("message", "Unknown error")
            self._lbl_status.setText("Adapter fetch failed.")
            self.txt_output.appendPlainText(f"✗ Adapter fetch failed: {msg}")
            QMessageBox.critical(
                self, "Adapter fetch failed",
                f"The adapter fetch did not complete:\n\n{msg}",
            )
            return

        cfg = result.get("config", {})
        training = result.get("training", {})
        local_server = self._chat_widget._local_server_dir()
        size_mb = (result.get("size_bytes") or 0) / (1024 * 1024)
        self._lbl_status.setText(
            "Adapter fetched — reload the model to run the remote weights.")

        lines = [
            f"✓ Fetched adapter '{result.get('adapter_name')}' "
            f"({result.get('files', 0)} file(s), {size_mb:.1f} MiB) into {local_server}",
            f"  from:  {result.get('source_host') or '—'}",
            f"  build: {result.get('build_id') or '—'}",
            "",
            "server_config.json (adapter_id only — everything else untouched):",
            f"  model_id:       {cfg.get('model_id')}   [verified equal to source]",
            f"  adapter_id:     {cfg.get('adapter_id')}",
            f"  was:            {cfg.get('previous_adapter_id') or '(none)'}",
            f"  base_quant:     {cfg.get('base_quant')}",
            f"  context_length: {cfg.get('context_length')}",
            "",
            "This box keeps its OWN chat corpus, memory, ledger, prompts and persona — "
            "only the weights changed.",
        ]
        if training.get("included"):
            lines.append(
                f"  source training corpus: {training.get('build_id')} "
                f"({training.get('rows')} rows) — NOT fetched (weights only)."
            )
        lines += [
            "",
            "Next steps:",
            "  1. reload the model from the Chat tab (the new adapter_id is read at load)",
            "  2. to roll back: repoint adapter_id at the previous dir in models/.",
        ]
        self.txt_output.appendPlainText("\n".join(lines))
        QMessageBox.information(
            self, "Adapter fetched",
            f"'{result.get('adapter_name')}' is now this box's active adapter.\n\n"
            "Your local memory, chats and persona are unchanged. Reload the model from "
            "the Chat tab to run the fetched weights.",
        )

    # ---------------------------------------------------------------- #
    # Change precision (no migration)                                   #
    # ---------------------------------------------------------------- #

    def _on_precision_clicked(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return

        quant_label = self.cmb_quant.currentText()
        base_quant = self.cmb_quant.currentData()
        host = self._chat_widget.txt_host.text().strip() or "localhost"

        reply = QMessageBox.question(
            self,
            "Change base precision",
            f"Set the base-model load precision on {host} to {quant_label}?\n\n"
            "This rewrites server_config.json on the connected server but does NOT "
            "restart it — the new precision takes effect on the NEXT server restart "
            "(use the Chat tab's restart, or relaunch the watchdog).\n\n"
            "No data is migrated; the model_id is kept for tokenizer/LoRA compatibility.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_precision.setEnabled(False)
        self.btn_migrate.setEnabled(False)
        self.btn_merge_chats.setEnabled(False)
        self.btn_fetch_snapshot.setEnabled(False)
        self.btn_fetch_adapter.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Changing precision…")
        self.txt_output.appendPlainText(
            f"\n[precision] Setting base precision on {host} to {quant_label}…"
        )
        self._precision_worker = PrecisionWorker(client, base_quant)
        self._precision_worker.precision_done.connect(self._on_precision_done)
        self._precision_worker.start()

    def _on_precision_done(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_precision.setEnabled(self._source_ok)
        self.btn_migrate.setEnabled(not self._same_box and self._source_ok)
        self.btn_merge_chats.setEnabled(not self._same_box and self._source_ok)
        self.btn_fetch_snapshot.setEnabled(not self._same_box and self._source_ok)
        self.btn_fetch_adapter.setEnabled(not self._same_box and self._source_ok)

        if result.get("error") or not result.get("ok"):
            msg = result.get("error") or "the watchdog rejected the change"
            self._lbl_status.setText(f"Precision change failed: {msg}")
            self.txt_output.appendPlainText(f"✗ Precision change failed: {msg}")
            QMessageBox.critical(
                self, "Precision change failed",
                f"Could not change precision:\n\n{msg}",
            )
            return

        bq = result.get("base_quant")
        prev = result.get("previous") or "default 4bit"
        self._lbl_status.setText("Precision changed — restart the server to apply.")
        self.txt_output.appendPlainText(
            f"✓ base_quant set to {bq} (was {prev}).\n"
            "  Takes effect on the NEXT server restart (Chat tab → restart server, "
            "or relaunch the watchdog)."
        )
        QMessageBox.information(
            self, "Precision changed",
            f"Base precision set to {bq}.\n\n"
            "The running server was NOT restarted — restart the inference server "
            "(Chat tab → restart, or relaunch the watchdog) for it to take effect.",
        )

    # ---------------------------------------------------------------- #
    # Fonts                                                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self.txt_output.setFont(font)
