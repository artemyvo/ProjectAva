"""Feed a Library from Ava's own corpus — the live `server/data/` tree or a runnable-snapshot
export, which share one layout: `chats/<stem>.json` transcripts (with `<stem>.facts.json`
protocols and `<stem>.summary.json` gists beside them) and `til/snippets/<kind>/<stem>.json`
texts (with the same two sidecars). Transcripts become the chat kind, TIL snippets `article`
(lookups, wander) or `news`; Ava's `.facts.json` is imported as each document's protocol
(`witness.import_protocol`: every fact anchored to a chunk by overlap — lemmas OR L2 cells,
the cells placing an English fact's words over a Russian exchange) and her gist becomes the
document's summary. Nothing here generates: the passes that wrote those files keep running
on Ava's side, and the library reads what they wrote.

Two entry points over the same per-file functions:

  `import_export(lib, export_dir)`   one-shot over a snapshot export (the bench path);
  `sync(lib, chats_dirs, til_dir)`   incremental over the live tree — a per-file fingerprint
                                     (mtime, size, and the sidecars') in `state/feed.json`
                                     skips what has not changed, so an idle job can run it
                                     hourly for the cost of a stat per file.

    python -m assoc.ava_import <export_dir> <library_root> [--bge]
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from .library import Library
from .witness import import_protocol

_STEM_TS = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
TIL_KINDS = (("lookups", "article"), ("wander", "article"), ("news", "news"))
_SIDECARS = (".state.json", ".summary.json", ".facts.json", ".revisit-bak")


def _date_of_stem(stem: str) -> str:
    m = _STEM_TS.match(os.path.basename(stem))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}" if m else ""


def _chat_subject(raw: str) -> str:
    s = " ".join((raw or "").split()).strip().lower()
    if s in ("_self", "self", "me", "i", "ava"):
        return "person:_self"
    if not s or s in ("nobody", "", "user", "the user"):
        return ""
    return "person:" + s.split()[0]


def _til_subject(raw: str) -> str:
    s = " ".join((raw or "").split()).strip().strip("«»\"' ").lower()
    return f"entity:{s}" if s else ""


def _facts_of(path: Path, subject_fn) -> list[dict]:
    try:
        ff = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    return [{"subject": subject_fn(x.get("subject") or ""), "subject_raw": x.get("subject_raw") or x.get("subject") or "",
             "text": x.get("text") or "", "fact_class": x.get("fact_class") or "unspecified",
             "entities": list(x.get("entities") or []), "when": x.get("when") or "", "rel": None}
            for x in ff.get("facts") or [] if x.get("text")]


def _summary_of(lib: Library, doc_id: str, path: Path) -> None:
    if path.exists():
        try:
            s = json.load(open(path, encoding="utf-8"))
            lib.store.write_summary(doc_id, {"text": s.get("text") or "", "ts": s.get("ts") or "", "source": "ava"})
        except Exception:
            pass


def is_transcript(path: Path) -> bool:
    """Ava's own rule (`chat_sidecar.is_chat_session_json`): a chat's stem owns several
    `.json` files, and only the bare one is the transcript."""
    return path.suffix == ".json" and not path.name.endswith(_SIDECARS)


def import_chat_file(lib: Library, path: Path) -> Optional[tuple[str, list[dict]]]:
    """One transcript → (doc_id, Ava's facts to import as its protocol), or None if skipped."""
    try:
        c = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    exchanges = c.get("exchanges") or []
    if not exchanges:
        return None
    stem = str(path)[:-5]
    doc = {"user": c.get("user") or "", "exchanges": [
        {"user_prompt": e.get("user_prompt") or "", "assistant_response": e.get("assistant_response") or "",
         "speaker": e.get("speaker") or c.get("user") or "", "ts": c.get("timestamp") or ""} for e in exchanges]}
    meta = {"key": "chat/" + os.path.basename(stem), "date": (c.get("timestamp") or _date_of_stem(stem))[:19],
            "initiated_by": c.get("initiated_by") or "", "interlocutor": c.get("interlocutor") or "",
            "title": f"chat {os.path.basename(stem)}", "session": path.name}
    doc_id = lib.store.ingest(json.dumps(doc, ensure_ascii=False), "chat", meta)
    _summary_of(lib, doc_id, Path(stem + ".summary.json"))
    return doc_id, _facts_of(Path(stem + ".facts.json"), _chat_subject)


def import_til_file(lib: Library, path: Path, kind_dir: str, kind: str) -> Optional[tuple[str, list[dict]]]:
    """One TIL snippet (`<kind>/<stem>.json`) → (doc_id, facts), or None if skipped."""
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    stem = str(path)[:-5]
    text = j.get("text") or ""
    if not text.strip() and Path(stem + ".txt").exists():
        text = open(stem + ".txt", encoding="utf-8").read()
    if len(text.strip()) < 40:
        return None
    title = j.get("title") or os.path.basename(stem)
    meta = {"key": f"til/{kind_dir}/{os.path.basename(stem)}", "title": title,
            "date": (j.get("date") or j.get("fetched_at") or _date_of_stem(stem))[:19],
            "url": j.get("source_url") or "", "source": j.get("source") or j.get("wiki") or kind_dir,
            "language": j.get("lang") or "", "til_ref": f"{kind_dir}/{os.path.basename(stem)}"}
    doc_id = lib.store.ingest(f"# {title}\n\n{text}", kind, meta)
    _summary_of(lib, doc_id, Path(stem + ".summary.json"))
    return doc_id, _facts_of(Path(stem + ".facts.json"), _til_subject)


def _own_witness(lib: Library, doc_id: str) -> bool:
    doc = lib.store.document(doc_id)
    return bool(doc and (doc.facts or {}).get("witness") == "llm" and (doc.facts or {}).get("prompt_version") != "imported")


def _fingerprint(path: Path) -> str:
    parts = []
    stem = str(path)[:-5]
    for p in (path, Path(stem + ".facts.json"), Path(stem + ".summary.json")):
        try:
            st = p.stat()
            parts.append(f"{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append("-")
    return "|".join(parts)


def _import_protocols(lib: Library, pending: list[tuple[str, list[dict], str]], report: dict, log) -> None:
    """Phase 2: protocols, anchored with the current build's cells. Needs a build; the
    caller rebuilds first when there is none."""
    b = lib.build
    codebook = b.codebook if b is not None else None
    t1 = time.time()
    for i, (doc_id, facts, witness) in enumerate(pending):
        r = import_protocol(lib.store, doc_id, facts, witness=witness, codebook=codebook, embedder=lib.embedder)
        key = "chat_facts" if witness == "ava-chat-facts" else "til_facts"
        report[key] = report.get(key, 0) + r.get("facts", 0)
        report["ungrounded"] = report.get("ungrounded", 0) + r.get("ungrounded", 0)
        report["grounded_via_cells"] = report.get("grounded_via_cells", 0) + r.get("grounded_via_cells", 0)
        if (i + 1) % 50 == 0:
            log(f"[import]   protocols {i + 1}/{len(pending)} ({time.time() - t1:.0f}s)")


def sync(lib: Library, chats_dirs: list, til_dir: Optional[os.PathLike] = None, *, log=print,
         rebuild: bool = True) -> dict:
    """Incremental feed over the live tree. Returns counts; rebuilds when anything changed
    (or when the library has no build yet)."""
    t0 = time.time()
    state_path = lib.store.root / "state" / "feed.json"
    try:
        seen: dict = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        seen = {}
    report = {"scanned": 0, "chats": 0, "til": 0, "unchanged": 0, "skipped": 0, "doc_ids": []}
    pending: list[tuple[str, list[dict], str]] = []
    new_seen = dict(seen)

    for d in chats_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if not is_transcript(p):
                continue
            report["scanned"] += 1
            fp = _fingerprint(p)
            if seen.get(str(p)) == fp:
                report["unchanged"] += 1
                continue
            res = import_chat_file(lib, p)
            if res is None:
                report["skipped"] += 1
                continue
            report["chats"] += 1
            report["doc_ids"].append(res[0])
            if _own_witness(lib, res[0]):
                # The library witnessed this document itself (an earlier wake); Ava's
                # protocol is not imported over it. A grown transcript is pending again
                # (the store marks the new chunks), so the witness job revisits it.
                report["kept_own"] = report.get("kept_own", 0) + 1
            else:
                pending.append((res[0], res[1], "ava-chat-facts"))
            new_seen[str(p)] = fp
    if til_dir is not None:
        for kind_dir, kind in TIL_KINDS:
            for p in sorted((Path(til_dir) / kind_dir).glob("*.json")):
                if p.name.endswith((".facts.json", ".summary.json")):
                    continue
                report["scanned"] += 1
                fp = _fingerprint(p)
                if seen.get(str(p)) == fp:
                    report["unchanged"] += 1
                    continue
                res = import_til_file(lib, p, kind_dir, kind)
                if res is None:
                    report["skipped"] += 1
                    continue
                report["til"] += 1
                report["doc_ids"].append(res[0])
                pending.append((res[0], res[1], "ava-til-facts"))
                new_seen[str(p)] = fp

    changed = bool(pending)
    first = False
    report["first"] = False
    if changed:
        if lib.build is None:
            # First feed: the cell half of grounding needs a codebook, which needs a build.
            first = True
            report["first"] = True
            rep1 = lib.rebuild("full")
            log(f"[import] first build: {rep1['counts']} ({rep1['seconds']}s)")
        _import_protocols(lib, pending, report, log)
    if new_seen != seen:
        # Record every fingerprint that moved, not only those with a protocol to import: a
        # chat the library witnessed itself (`kept_own`) imports no protocol, and left
        # unrecorded it was re-ingested on every wake.
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(new_seen), encoding="utf-8")
    if rebuild:
        st = lib.staleness() if lib.build is not None else {"scope": "full", "reasons": ["no_build"]}
        report["staleness"] = st
        if changed or st.get("scope") != "none":
            # The first feed's protocols land on a codebook clustered before any claim
            # existed; a fast rebuild would only place their vocabulary and report drift.
            rep = lib.rebuild("full" if first else None)
            report["rebuild"] = {k: rep.get(k) for k in ("scope", "counts", "seconds", "contests", "families")}
    report["seconds"] = round(time.time() - t0, 1)
    return report


def import_export(lib: Library, export_dir: str | os.PathLike, *, log=print, chat_limit: Optional[int] = None,
                  til_limit: Optional[int] = None) -> dict:
    """One-shot over a snapshot export: `chats/` + `til/snippets/<kind>/`."""
    export = Path(export_dir)
    t0 = time.time()
    report = {"chats": 0, "chat_facts": 0, "til": 0, "til_facts": 0, "skipped": 0, "ungrounded": 0, "grounded_via_cells": 0}
    pending: list[tuple[str, list[dict], str]] = []
    chats = sorted(p for p in (export / "chats").glob("*.json") if is_transcript(p))
    for p in chats[: (chat_limit or 10**9)]:
        res = import_chat_file(lib, p)
        if res is None:
            report["skipped"] += 1
            continue
        report["chats"] += 1
        pending.append((res[0], res[1], "ava-chat-facts"))
    for kind_dir, kind in TIL_KINDS:
        metas = sorted(p for p in (export / "til" / "snippets" / kind_dir).glob("*.json")
                       if not p.name.endswith((".facts.json", ".summary.json")))
        for p in metas[: (til_limit or 10**9)]:
            res = import_til_file(lib, p, kind_dir, kind)
            if res is None:
                report["skipped"] += 1
                continue
            report["til"] += 1
            pending.append((res[0], res[1], "ava-til-facts"))
    log(f"[import] phase 1: {report['chats']} chats, {report['til']} til docs, {len(pending)} protocols to import ({time.time() - t0:.0f}s)")
    rep1 = lib.rebuild("full")
    log(f"[import] rebuild 1: {rep1['counts']} codebook {rep1['codebook']} ({rep1['seconds']}s)")
    t1 = time.time()
    _import_protocols(lib, pending, report, log)
    log(f"[import] phase 2: {report['chat_facts']} chat facts, {report['til_facts']} til facts, "
        f"{report['ungrounded']} ungrounded, {report['grounded_via_cells']} grounded via cells ({time.time() - t1:.0f}s)")
    rep2 = lib.rebuild("full")
    log(f"[import] rebuild 2: {rep2['counts']} dedup {rep2['dedup']} contests {rep2['contests']} families {rep2['families']} "
        f"senses {rep2['senses']['words']} edges {rep2['edges']} ({rep2['seconds']}s)")
    report["rebuild"] = {k: rep2[k] for k in ("counts", "dedup", "contests", "supersedes", "families", "alias_proposals", "codebook", "edges", "senses", "seconds")}
    report["seconds"] = round(time.time() - t0, 1)
    return report


if __name__ == "__main__":
    from .budget import Budget
    export_dir, root = sys.argv[1], sys.argv[2]
    embedder = None
    if "--bge" in sys.argv:
        from .dense import BgeM3Embedder
        import torch
        embedder = BgeM3Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    lib = Library(root, embedder=embedder, budget=Budget(total=8000))
    rep = import_export(lib, export_dir)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
