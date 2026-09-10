"""The document store (ASSOCIATIVE_MEMORY.md §1.4).

```
root/
  documents/<doc_id>/document.txt   the rendered text of THIS version (spans index into it)
                     source.txt     the text as handed to ingest (JSON for a chat)
                     meta.json      kind, key, title, version, date, language, scope facets, supersedes
                     chunks.jsonl   units (§1.5)
                     facts.json     the protocol (after the witness)
                     summary.json   the gist (optional)
                     status.json    pending | extracted | failed
  keys.json                          doc_key -> {kind, versions: [doc_id...], removed}
  state/                             activation, ledgers (later milestones)
  index/<build_id>/                  derived layers (§2.8); index/current -> build id
```

A *version* is immutable and a *key* is durable. Ingesting a known key makes a new version
(``supersedes`` → the previous), except for an append-only kind (chat), which grows in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .chunks import Unit
from . import kinds as kinds_mod

SCHEMA_VERSION = 1
APPEND_ONLY_KINDS = frozenset({"chat"})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def doc_id_for(key: str, version: str) -> str:
    return hashlib.sha1(f"{key}\x1f{version}".encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@dataclass
class Document:
    doc_id: str
    meta: dict
    text: str
    units: list[Unit]
    status: dict = field(default_factory=dict)
    facts: Optional[dict] = None
    summary: Optional[dict] = None

    @property
    def key(self) -> str:
        return self.meta["key"]

    @property
    def kind(self) -> str:
        return self.meta["kind"]

    def unit(self, chunk_id: str) -> Optional[Unit]:
        for u in self.units:
            if u.chunk_id == chunk_id:
                return u
        return None


class Store:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "documents").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self._keys: dict = _read_json(self.root / "keys.json", {}) or {}

    # ----- keys -------------------------------------------------------------------------
    def _save_keys(self) -> None:
        _write_json(self.root / "keys.json", self._keys)

    def keys(self) -> dict:
        return dict(self._keys)

    def versions_of(self, key: str) -> list[str]:
        return list((self._keys.get(key) or {}).get("versions") or [])

    def latest_id(self, key: str) -> Optional[str]:
        v = self.versions_of(key)
        return v[-1] if v else None

    # ----- ingest -----------------------------------------------------------------------
    def ingest(self, text: str, kind: str, meta: Optional[dict] = None, scope: Optional[dict] = None) -> str:
        """Store + chunk. Returns the doc_id. The witness is deferred (status pending)."""
        meta = dict(meta or {})
        spec = kinds_mod.get(kind)
        key = str(meta.get("key") or meta.get("url") or meta.get("path") or "")
        if not key:
            key = "doc-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        meta["key"] = key
        meta["kind"] = kind
        for k, v in (scope or {}).items():
            meta.setdefault(k, v)
        version = str(meta.get("version") or meta.get("date") or "")
        entry = self._keys.setdefault(key, {"kind": kind, "versions": [], "removed": False})
        entry["removed"] = False

        if kind in APPEND_ONLY_KINDS and entry["versions"]:
            return self._append(entry["versions"][-1], text, meta, spec)

        if not version:
            version = _now_iso() if not entry["versions"] else f"{_now_iso()}-{len(entry['versions'])}"
            meta["version"] = version
        doc_id = doc_id_for(key, version)
        position = None
        if doc_id in entry["versions"]:
            # Same key + version re-ingested: replaced in place, keeping its place in the chain.
            position = entry["versions"].index(doc_id)
            shutil.rmtree(self.root / "documents" / doc_id, ignore_errors=True)
        rendered, units = spec.split(text, meta)
        prev = (entry["versions"][position - 1] if position else None) if position is not None \
            else (entry["versions"][-1] if entry["versions"] else None)
        meta.update({"doc_id": doc_id, "version": version, "schema": SCHEMA_VERSION,
                     "ingested_at": _now_iso(), "supersedes": prev,
                     "append_only": kind in APPEND_ONLY_KINDS})
        meta.setdefault("title", self._title_of(units, key))
        meta.setdefault("language", _guess_language(rendered))
        d = self.root / "documents" / doc_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.txt").write_text(rendered, encoding="utf-8")
        (d / "source.txt").write_text(text, encoding="utf-8")
        _write_json(d / "meta.json", meta)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for u in units:
                fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        _write_json(d / "status.json", {"status": "pending", "witnesses": list(spec.witnesses), "updated_at": _now_iso()})
        if position is None:
            entry["versions"].append(doc_id)
        self._save_keys()
        return doc_id

    def _append(self, doc_id: str, text: str, meta: dict, spec) -> str:
        """Append-only growth of an existing document (chat): keep old chunk ids, add new."""
        d = self.root / "documents" / doc_id
        old_meta = _read_json(d / "meta.json", {}) or {}
        rendered, units = spec.split(text, {**old_meta, **meta, "key": old_meta.get("key")})
        old_units = list(self._read_units(doc_id))
        old_ids = {u.chunk_id for u in old_units}
        new_units = [u for u in units if u.chunk_id not in old_ids]
        old_text = (d / "document.txt").read_text(encoding="utf-8")
        if not rendered.startswith(old_text.rstrip("\n")):
            # Not a pure append: version it instead.
            entry = self._keys[old_meta["key"]]
            meta = {**meta, "version": f"{_now_iso()}-{len(entry['versions'])}"}
            return self._version_instead(text, meta, spec, entry)
        (d / "document.txt").write_text(rendered, encoding="utf-8")
        (d / "source.txt").write_text(text, encoding="utf-8")
        with (d / "chunks.jsonl").open("a", encoding="utf-8") as fh:
            for u in new_units:
                fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        status = _read_json(d / "status.json", {}) or {}
        if new_units:
            status.update({"status": "pending", "pending_chunks": (status.get("pending_chunks") or []) + [u.chunk_id for u in new_units],
                           "updated_at": _now_iso()})
            _write_json(d / "status.json", status)
        old_meta["appended_at"] = _now_iso()
        _write_json(d / "meta.json", old_meta)
        return doc_id

    def _version_instead(self, text: str, meta: dict, spec, entry: dict) -> str:
        key = meta["key"]
        version = meta["version"]
        doc_id = doc_id_for(key, version)
        rendered, units = spec.split(text, meta)
        meta.update({"doc_id": doc_id, "version": version, "schema": SCHEMA_VERSION, "ingested_at": _now_iso(),
                     "supersedes": entry["versions"][-1], "append_only": True})
        meta.setdefault("title", self._title_of(units, key))
        d = self.root / "documents" / doc_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.txt").write_text(rendered, encoding="utf-8")
        (d / "source.txt").write_text(text, encoding="utf-8")
        _write_json(d / "meta.json", meta)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for u in units:
                fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        _write_json(d / "status.json", {"status": "pending", "witnesses": list(spec.witnesses), "updated_at": _now_iso()})
        entry["versions"].append(doc_id)
        self._save_keys()
        return doc_id

    @staticmethod
    def _title_of(units: list[Unit], key: str) -> str:
        for u in units:
            if u.path:
                return u.path[0]
        return key

    # ----- read -------------------------------------------------------------------------
    def _read_units(self, doc_id: str) -> Iterator[Unit]:
        p = self.root / "documents" / doc_id / "chunks.jsonl"
        if not p.exists():
            return
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield Unit.from_dict(json.loads(line))

    def document(self, doc_id: str) -> Optional[Document]:
        d = self.root / "documents" / doc_id
        if not (d / "meta.json").exists():
            return None
        return Document(
            doc_id=doc_id, meta=_read_json(d / "meta.json", {}) or {},
            text=(d / "document.txt").read_text(encoding="utf-8"),
            units=list(self._read_units(doc_id)),
            status=_read_json(d / "status.json", {}) or {},
            facts=_read_json(d / "facts.json"), summary=_read_json(d / "summary.json"),
        )

    def meta(self, doc_id: str) -> Optional[dict]:
        return _read_json(self.root / "documents" / doc_id / "meta.json")

    def all_doc_ids(self) -> list[str]:
        out: list[str] = []
        for key, entry in list(self._keys.items()):
            out.extend(list(entry.get("versions") or []))
        return out

    def documents(self) -> Iterator[Document]:
        for doc_id in self.all_doc_ids():
            doc = self.document(doc_id)
            if doc is not None:
                yield doc

    def pending(self) -> list[str]:
        out = []
        for doc_id in self.all_doc_ids():
            st = _read_json(self.root / "documents" / doc_id / "status.json", {}) or {}
            if st.get("status") == "pending":
                out.append(doc_id)
        return out

    # ----- write-backs ------------------------------------------------------------------
    def write_facts(self, doc_id: str, protocol: dict, status: str = "extracted", report: Optional[dict] = None) -> None:
        d = self.root / "documents" / doc_id
        _write_json(d / "facts.json", protocol)
        st = _read_json(d / "status.json", {}) or {}
        st.update({"status": status, "updated_at": _now_iso(), "report": report or {}})
        st.pop("pending_chunks", None)
        _write_json(d / "status.json", st)

    def write_status(self, doc_id: str, status: str, report: Optional[dict] = None) -> None:
        d = self.root / "documents" / doc_id
        st = _read_json(d / "status.json", {}) or {}
        st.update({"status": status, "updated_at": _now_iso(), "report": report or {}})
        _write_json(d / "status.json", st)

    def write_summary(self, doc_id: str, summary: dict) -> None:
        _write_json(self.root / "documents" / doc_id / "summary.json", summary)

    def remove(self, key: str) -> None:
        entry = self._keys.get(key)
        if entry is None:
            return
        entry["removed"] = True
        self._save_keys()

    # ----- current set ------------------------------------------------------------------
    def current_doc_ids(self, scope: Optional[dict] = None) -> list[str]:
        """Newest version per key whose facets match *scope* (§1.2). A scope naming
        ``version`` selects that version exactly; absent facets are unscoped."""
        scope = dict(scope or {})
        wanted_version = scope.pop("version", None)
        out: list[str] = []
        for key, entry in list(self._keys.items()):      # an ingest on another thread may add a key
            if entry.get("removed"):
                continue
            chosen = None
            for doc_id in entry.get("versions") or []:
                meta = self.meta(doc_id) or {}
                if not _facets_match(meta, scope):
                    continue
                if wanted_version is not None and str(meta.get("version")) != str(wanted_version):
                    continue
                chosen = doc_id   # later versions override earlier ones
            if chosen:
                out.append(chosen)
        return out


def _facets_match(meta: dict, scope: dict) -> bool:
    for facet, value in scope.items():
        if value is None:
            continue
        have = meta.get(facet)
        if have is None:
            continue          # the document does not name this facet: unscoped on that axis
        if isinstance(have, (list, tuple, set)):
            if value not in have:
                return False
        elif str(have) != str(value):
            return False
    return True


def _guess_language(text: str) -> str:
    from .lex import dominant_script
    return {"cyr": "ru", "lat": "en"}.get(dominant_script(text[:4000]), "und")
