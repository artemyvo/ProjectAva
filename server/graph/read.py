"""The authoritative read set, and the occurrence rows loaded from it.

**This module's entire job is knowing which files count**, and it exists as its own module
because getting that wrong is silent. A ``.facts.json`` is copied by three separate
mechanisms — the persona lineage snapshots, the reflection checkpoint, and the per-run
review archive — and on this box at the time of writing, **38 of the 61 ``.facts.json``
files on disk are copies**. A build that globs the tree naively counts the same fact up to
three times and reports the duplication as corroboration, which is the one error mode a
fact fold must not have.

So the read set is enumerated positively, from the two dirs that hold originals, through
the shared resolver (``training.reflections_path``) rather than a second copy of the paths.
That resolver is not a stylistic preference: the live chat corpus moved to ``server/data/chats``
and stale copies of the old ``inference/data/hot/chats`` path have already caused a real bug
in this repo.

An **occurrence** is one line of one protocol — the leaf of the tree and the only level that
maps 1:1 onto an immutable source line. Everything above it is derived and can be rebuilt;
this is the level that carries provenance, so it keeps ``subject_raw`` verbatim and records
which lane and which file it came from.

GPU-free self-test: ``python -m graph.read``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from training import reflections_path

# Lane names. The lane is not cosmetic: it decides the subject namespace (a person key vs an
# entity mention) and, with `fact_class`, the facet — see `fold.facet_for`.
LANE_CHAT = "chat"
LANE_TIL = "til"

FACTS_SUFFIX = ".facts.json"

# A chat protocol's stem is its session timestamp; that IS the conversation's clock, and the
# only one most chat facts have (measured: 3 of 349 carry a `when`).
_STEM_TS_RE = re.compile(r"^(\d{8})_(\d{6})$")


def _iso_from_stem(stem: str) -> str:
    """``20260810_053413`` → ``2026-08-10``. Empty when the stem is not a timestamp."""
    m = _STEM_TS_RE.match(stem)
    if not m:
        return ""
    d = m.group(1)
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


# -- the authoritative set --------------------------------------------------- #

def chat_protocol_paths(chats_dir: Optional[Path] = None,
                        archive_dir: Optional[Path] = None) -> list:
    """Chat-lane protocols: the live corpus, plus the archive if this box has one.

    The archive is included where the copies are not, and the difference is the whole rule:
    an archived transcript is an *original* that was moved, while a snapshot's copy is a
    second pointer to a file still sitting in the live dir. (``archive_chats_dir`` is
    retired and normally absent; it is unioned under an ``.exists()`` guard so a box that
    still has one does not silently drop it.)
    """
    dirs = [chats_dir or reflections_path.hot_chats_dir(),
            archive_dir or reflections_path.archive_chats_dir()]
    out: list = []
    for d in dirs:
        if d and Path(d).is_dir():
            out.extend(sorted(Path(d).glob("*" + FACTS_SUFFIX)))
    return out


def til_protocol_paths(snippets_dir: Optional[Path] = None) -> list:
    """TIL-lane protocols: ``data/til/snippets/<kind>/<stem>.facts.json``.

    Enumerated one kind-dir deep rather than with ``rglob``, so a persona snapshot that
    nests a whole ``til/snippets/`` tree underneath cannot be swept in.
    """
    root = Path(snippets_dir or reflections_path.til_snippets_dir())
    if not root.is_dir():
        return []
    out: list = []
    for kind_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        out.extend(sorted(kind_dir.glob("*" + FACTS_SUFFIX)))
    return out


def protocol_paths(**kw) -> list:
    """Every protocol in the authoritative set, both lanes."""
    return ([(LANE_CHAT, p) for p in chat_protocol_paths(kw.get("chats_dir"),
                                                         kw.get("archive_dir"))]
            + [(LANE_TIL, p) for p in til_protocol_paths(kw.get("snippets_dir"))])


# -- loading ----------------------------------------------------------------- #

def _load_doc(path: Path) -> Optional[dict]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _occurrences_from_chat(doc: dict, path: Path) -> list:
    stem = path.name[: -len(FACTS_SUFFIX)]
    source_ref = doc.get("source_session") or (stem + ".json")
    # The session stem, not the document `ts`: `ts` is when the *pass ran*, which on a
    # backfilled protocol can be weeks after the conversation. Every chat-side curve on this
    # box keys on the conversation's own date, and so does this.
    asserted_at = _iso_from_stem(stem) or str(doc.get("ts") or "")[:10]
    out = []
    for i, f in enumerate(doc.get("facts") or []):
        if not isinstance(f, dict):
            continue
        out.append({
            "lane": LANE_CHAT,
            "source_ref": source_ref,
            "source_path": str(path),
            "line_index": i,
            "subject_key": str(f.get("subject") or ""),
            "subject_raw": str(f.get("subject_raw") or ""),
            "text": str(f.get("text") or ""),
            "fact_class": str(f.get("fact_class") or ""),
            "entities": [str(e) for e in (f.get("entities") or []) if str(e).strip()],
            "when": str(f.get("when") or ""),
            "asserted_at": asserted_at,
            "source_user": str(doc.get("source_user") or ""),
            "run_id": str(doc.get("run_id") or ""),
        })
    return out


def _occurrences_from_til(doc: dict, path: Path) -> list:
    stem = path.name[: -len(FACTS_SUFFIX)]
    kind = str(doc.get("source_kind") or path.parent.name or "")
    source_ref = f"{kind}/{stem}" if kind else stem
    asserted_at = str(doc.get("source_date") or "")[:10] or str(doc.get("ts") or "")[:10]
    out = []
    for i, f in enumerate(doc.get("facts") or []):
        if not isinstance(f, dict):
            continue
        out.append({
            "lane": LANE_TIL,
            "source_ref": source_ref,
            "source_path": str(path),
            "line_index": i,
            "subject_key": str(f.get("subject") or ""),
            "subject_raw": str(f.get("subject_raw") or ""),
            "text": str(f.get("text") or ""),
            "fact_class": str(f.get("fact_class") or ""),
            "entities": [str(e) for e in (f.get("entities") or []) if str(e).strip()],
            "when": str(f.get("when") or ""),
            "asserted_at": asserted_at,
            # Provenance instead of attribution — the source is a text, not a speaker. Kept
            # exactly as recorded and never reduced to a score: judging a source is
            # interpretation, and the approved wiki list is chosen for tone, not truth.
            "source_kind": kind,
            "source_ref_id": str(doc.get("source_ref") or ""),
            "source_url": str(doc.get("source_url") or ""),
            "source_title": str(doc.get("source_title") or ""),
            "run_id": str(doc.get("run_id") or ""),
        })
    return out


def load_occurrences(**kw) -> tuple:
    """Read the authoritative set. Returns ``(occurrences, stats)``.

    An occurrence with empty ``text`` is dropped — it carries nothing to state — but the
    count is reported rather than swallowed, since a protocol full of them means the parser
    and the generation disagree, which is a producer bug this fold would otherwise hide.
    """
    paths = kw.pop("paths", None)
    if paths is None:
        paths = protocol_paths(**kw)
    occurrences: list = []
    stats = {"files": 0, "files_unreadable": 0, "files_by_lane": {},
             "occurrences": 0, "dropped_empty": 0}
    for lane, path in paths:
        doc = _load_doc(Path(path))
        if doc is None:
            stats["files_unreadable"] += 1
            continue
        stats["files"] += 1
        stats["files_by_lane"][lane] = stats["files_by_lane"].get(lane, 0) + 1
        rows = (_occurrences_from_chat(doc, Path(path)) if lane == LANE_CHAT
                else _occurrences_from_til(doc, Path(path)))
        for r in rows:
            if not r["text"].strip():
                stats["dropped_empty"] += 1
                continue
            occurrences.append(r)
    stats["occurrences"] = len(occurrences)
    return occurrences, stats


# -- self-test --------------------------------------------------------------- #

def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        chats.mkdir()
        snips = root / "til" / "snippets" / "news"
        snips.mkdir(parents=True)

        (chats / "20260810_053413.facts.json").write_text(json.dumps({
            "source_session": "20260810_053413.json", "source_user": "artemyvo",
            "run_id": "r1", "ts": "2026-08-10T09:06:20",
            "facts": [
                {"subject": "_self", "subject_raw": "self", "text": "Said a thing.",
                 "fact_class": "stated", "entities": ["subjectivity"], "when": ""},
                {"subject": "artemyvo", "subject_raw": "artemyvo", "text": "",
                 "fact_class": "standing", "entities": [], "when": ""},
            ]}), encoding="utf-8")
        (snips / "2026-07-28.facts.json").write_text(json.dumps({
            "source_kind": "news", "source_ref": "wikipedia:current_events",
            "source_url": "http://x", "source_title": "T", "source_date": "2026-07-28",
            "run_id": "r1", "ts": "2026-08-10T08:46:44",
            "facts": [{"subject": "United States Central Command",
                       "subject_raw": "United States Central Command",
                       "text": "Missiles were intercepted.", "fact_class": "stated",
                       "entities": ["ballistic missiles"], "when": "2026-07-28"}]},
        ), encoding="utf-8")

        # A copy, in the shape the persona lineage actually produces: a nested tree holding
        # the same protocols. It must not be read.
        copy = root / "persona" / "20260810_082617" / "chats"
        copy.mkdir(parents=True)
        (copy / "20260810_053413.facts.json").write_text(
            (chats / "20260810_053413.facts.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        copy_til = root / "persona" / "20260810_082617" / "til" / "snippets" / "news"
        copy_til.mkdir(parents=True)
        (copy_til / "2026-07-28.facts.json").write_text(
            (snips / "2026-07-28.facts.json").read_text(encoding="utf-8"), encoding="utf-8")

        occ, stats = load_occurrences(chats_dir=chats,
                                      archive_dir=root / "archive" / "chats",
                                      snippets_dir=root / "til" / "snippets")

        assert stats["files"] == 2, stats                  # the copies are not in the set
        assert stats["dropped_empty"] == 1, stats          # empty text dropped, and counted
        assert len(occ) == 2, occ
        chat = [o for o in occ if o["lane"] == LANE_CHAT][0]
        til = [o for o in occ if o["lane"] == LANE_TIL][0]
        # The chat clock is the conversation's date, not the (later) pass date.
        assert chat["asserted_at"] == "2026-08-10", chat
        assert chat["source_ref"] == "20260810_053413.json"
        assert chat["subject_raw"] == "self"               # provenance kept verbatim
        assert til["asserted_at"] == "2026-07-28", til
        assert til["source_ref"] == "news/2026-07-28", til
        assert til["source_url"] == "http://x"

        # A missing archive dir is a no-op, not an error.
        assert chat_protocol_paths(chats, root / "nope") and True

        # Unreadable files are counted, never fatal.
        (chats / "20260101_000000.facts.json").write_text("{not json", encoding="utf-8")
        _, s2 = load_occurrences(chats_dir=chats, archive_dir=root / "archive" / "chats",
                                 snippets_dir=root / "til" / "snippets")
        assert s2["files_unreadable"] == 1, s2

    print("graph.read self-test OK")


if __name__ == "__main__":
    _selftest()
