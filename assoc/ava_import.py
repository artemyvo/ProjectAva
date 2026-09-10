"""Import an Ava runnable-snapshot export (`chats/`, `til/`) into a Library: transcripts as
the chat kind, TIL snippets as article / news, Ava's `.facts.json` protocols imported as
each document's protocol (anchored to chunks by overlap — lemmas and L2 cells, since Ava's
chat facts are mostly English over Russian exchanges), Ava's gists as the documents'
summaries. Two phases: ingest + rebuild first (the codebook must exist for the cell half of
grounding), protocols second, rebuild again.

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


def import_export(lib: Library, export_dir: str | os.PathLike, *, log=print, chat_limit: Optional[int] = None,
                  til_limit: Optional[int] = None) -> dict:
    export = Path(export_dir)
    t0 = time.time()
    report = {"chats": 0, "chat_facts": 0, "til": 0, "til_facts": 0, "skipped": 0, "ungrounded": 0, "grounded_via_cells": 0}
    pending: list[tuple[str, list[dict], str]] = []

    # ----- phase 1: documents -------------------------------------------------------------
    chats = sorted(f for f in glob.glob(str(export / "chats" / "*.json")) if not f.endswith((".state.json", ".summary.json", ".facts.json")))
    for path in chats[: (chat_limit or 10**9)]:
        stem = path[:-5]
        try:
            c = json.load(open(path, encoding="utf-8"))
        except Exception:
            report["skipped"] += 1
            continue
        exchanges = c.get("exchanges") or []
        if not exchanges:
            report["skipped"] += 1
            continue
        doc = {"user": c.get("user") or "", "exchanges": [
            {"user_prompt": e.get("user_prompt") or "", "assistant_response": e.get("assistant_response") or "",
             "speaker": e.get("speaker") or c.get("user") or "", "ts": c.get("timestamp") or ""} for e in exchanges]}
        key = "chat/" + os.path.basename(stem)
        meta = {"key": key, "date": (c.get("timestamp") or _date_of_stem(stem))[:19], "initiated_by": c.get("initiated_by") or "",
                "interlocutor": c.get("interlocutor") or "", "title": f"chat {os.path.basename(stem)}"}
        doc_id = lib.store.ingest(json.dumps(doc, ensure_ascii=False), "chat", meta)
        report["chats"] += 1
        sp = Path(stem + ".summary.json")
        if sp.exists():
            try:
                s = json.load(open(sp, encoding="utf-8"))
                lib.store.write_summary(doc_id, {"text": s.get("text") or "", "ts": s.get("ts") or "", "source": "ava"})
            except Exception:
                pass
        fp = Path(stem + ".facts.json")
        if fp.exists():
            try:
                ff = json.load(open(fp, encoding="utf-8"))
                facts = [{"subject": _chat_subject(x.get("subject") or ""), "subject_raw": x.get("subject_raw") or x.get("subject") or "",
                          "text": x.get("text") or "", "fact_class": x.get("fact_class") or "unspecified",
                          "entities": list(x.get("entities") or []), "when": x.get("when") or "", "rel": None}
                         for x in ff.get("facts") or [] if x.get("text")]
                pending.append((doc_id, facts, "ava-chat-facts"))
            except Exception:
                pass

    for kind_dir, kind in (("lookups", "article"), ("wander", "article"), ("news", "news")):
        metas = sorted(f for f in glob.glob(str(export / "til" / "snippets" / kind_dir / "*.json")) if not f.endswith((".facts.json", ".summary.json")))
        for path in metas[: (til_limit or 10**9)]:
            stem = path[:-5]
            try:
                j = json.load(open(path, encoding="utf-8"))
            except Exception:
                report["skipped"] += 1
                continue
            text = j.get("text") or ""
            if not text.strip() and Path(stem + ".txt").exists():
                text = open(stem + ".txt", encoding="utf-8").read()
            if len(text.strip()) < 40:
                report["skipped"] += 1
                continue
            title = j.get("title") or os.path.basename(stem)
            key = f"til/{kind_dir}/{os.path.basename(stem)}"
            meta = {"key": key, "title": title, "date": (j.get("date") or j.get("fetched_at") or _date_of_stem(stem))[:19],
                    "url": j.get("source_url") or "", "source": j.get("source") or j.get("wiki") or kind_dir, "language": j.get("lang") or ""}
            doc_id = lib.store.ingest(f"# {title}\n\n{text}", kind, meta)
            report["til"] += 1
            sp = Path(stem + ".summary.json")
            if sp.exists():
                try:
                    s = json.load(open(sp, encoding="utf-8"))
                    lib.store.write_summary(doc_id, {"text": s.get("text") or "", "ts": s.get("ts") or "", "source": "ava"})
                except Exception:
                    pass
            fp = Path(stem + ".facts.json")
            if fp.exists():
                try:
                    ff = json.load(open(fp, encoding="utf-8"))
                    facts = [{"subject": _til_subject(x.get("subject") or ""), "subject_raw": x.get("subject_raw") or x.get("subject") or "",
                              "text": x.get("text") or "", "fact_class": x.get("fact_class") or "unspecified",
                              "entities": list(x.get("entities") or []), "when": x.get("when") or "", "rel": None}
                             for x in ff.get("facts") or [] if x.get("text")]
                    pending.append((doc_id, facts, "ava-til-facts"))
                except Exception:
                    pass
    log(f"[import] phase 1: {report['chats']} chats, {report['til']} til docs, {len(pending)} protocols to import ({time.time() - t0:.0f}s)")

    # ----- rebuild 1: chunks, glossary, codebook (no claims yet) ----------------------------
    rep1 = lib.rebuild("full")
    log(f"[import] rebuild 1: {rep1['counts']} codebook {rep1['codebook']} ({rep1['seconds']}s)")

    # ----- phase 2: protocols, anchored with the cells ---------------------------------------
    b = lib.build
    t1 = time.time()
    for i, (doc_id, facts, witness) in enumerate(pending):
        r = import_protocol(lib.store, doc_id, facts, witness=witness, codebook=b.codebook, embedder=lib.embedder)
        if witness == "ava-chat-facts":
            report["chat_facts"] += r.get("facts", 0)
        else:
            report["til_facts"] += r.get("facts", 0)
        report["ungrounded"] += r.get("ungrounded", 0)
        report["grounded_via_cells"] += r.get("grounded_via_cells", 0)
        if (i + 1) % 50 == 0:
            log(f"[import]   protocols {i + 1}/{len(pending)} ({time.time() - t1:.0f}s)")
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
