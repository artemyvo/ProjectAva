"""A structure-first Markdown splitter shared by the prose kinds (§1.5 pass 1).

Cuts only where the text says it has a boundary: headings, paragraphs, fenced code blocks
(kept whole with their lead-in sentence), tables (kept whole; row groups as pieces), lists
(kept whole). A *section* is the finest heading level's content: heading + everything
until the next heading of any level. That section is the ``primary`` unit; its blocks are
``pieces``. A heading with no own content is an ``ancestor``. Fragments too small to stand
alone are merged upward into their section (they are still inside the section's text).
"""

from __future__ import annotations

import re
from typing import Optional

from ..chunks import Unit, make_unit

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MIN_PIECE_CHARS = 40
ROWGROUP_ROWS = 8


def _blocks(lines: list[str], line_starts: list[int]) -> list[dict]:
    """Split a section body into typed blocks with char spans: paragraph | code | table | list."""
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        start_line = i
        if _FENCE_RE.match(line.strip()):
            fence = line.strip()[:3]
            j = i + 1
            while j < n and not lines[j].strip().startswith(fence):
                j += 1
            j = min(j + 1, n)
            btype = "code"
        elif _TABLE_ROW_RE.match(line):
            j = i
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            btype = "table"
        elif _LIST_RE.match(line):
            j = i
            while j < n and (lines[j].strip() == "" and j + 1 < n and (_LIST_RE.match(lines[j + 1]) or lines[j + 1].startswith("  "))
                             or _LIST_RE.match(lines[j]) or (lines[j].startswith("  ") and lines[j].strip())):
                j += 1
            btype = "list"
        else:
            j = i
            while j < n and lines[j].strip() and not _FENCE_RE.match(lines[j].strip()) \
                    and not _TABLE_ROW_RE.match(lines[j]) and not _LIST_RE.match(lines[j]) and not _HEADING_RE.match(lines[j]):
                j += 1
            btype = "paragraph"
        if j == start_line:
            j = start_line + 1
        text = "\n".join(lines[start_line:j]).rstrip()
        span = (line_starts[start_line], line_starts[start_line] + len(text))
        blocks.append({"type": btype, "text": text, "span": span, "lines": (start_line, j)})
        i = j
    # A code block takes the paragraph immediately before it as its lead-in (never cut inside).
    merged: list[dict] = []
    for b in blocks:
        if b["type"] == "code" and merged and merged[-1]["type"] == "paragraph" \
                and b["lines"][0] - merged[-1]["lines"][1] <= 1:
            prev = merged.pop()
            b = {"type": "code", "text": prev["text"] + "\n" + b["text"], "span": (prev["span"][0], b["span"][1]),
                 "lines": (prev["lines"][0], b["lines"][1])}
        merged.append(b)
    return merged


def table_rows(text: str) -> tuple[list[str], list[str]]:
    """(header cells, data row lines) for a Markdown table block."""
    rows = [ln for ln in text.splitlines() if _TABLE_ROW_RE.match(ln)]
    if not rows:
        return [], []
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    data = [r for r in rows[1:] if not _TABLE_SEP_RE.match(r)]
    return header, data


def split_markdown(doc_key: str, text: str, *, title: Optional[str] = None, paragraph_primary: bool = False) -> list[Unit]:
    """*paragraph_primary*: each block is its own primary unit (news / article kinds, §1.4)
    instead of the section being primary with blocks as pieces (tech docs)."""
    lines = text.split("\n")
    line_starts: list[int] = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln) + 1

    # Walk headings; collect (heading path, body line range).
    sections: list[dict] = []
    path: list[str] = []
    levels: list[int] = []
    cur_start = 0
    cur_path: list[str] = [title] if title else []
    heading_line = None
    in_fence = False
    for i, ln in enumerate(lines):
        if _FENCE_RE.match(ln.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(ln)
        if not m:
            continue
        sections.append({"path": list(cur_path), "lines": (cur_start, i), "heading_line": heading_line})
        level, heading = len(m.group(1)), m.group(2).strip()
        heading_line = i
        while levels and levels[-1] >= level:
            levels.pop()
            path.pop()
        levels.append(level)
        path.append(heading)
        cur_path = ([title] if title else []) + list(path)
        cur_start = i + 1
    sections.append({"path": list(cur_path), "lines": (cur_start, len(lines)), "heading_line": heading_line})

    # A title equal to the document's own first heading would double the path root.
    units: list[Unit] = []
    seen_paths: dict[str, int] = {}
    for sec in sections:
        if title and len(sec["path"]) >= 2 and sec["path"][0].casefold() == sec["path"][1].casefold():
            sec["path"] = sec["path"][1:]
        a, b = sec["lines"]
        body_lines = lines[a:b]
        if not "".join(body_lines).strip():
            if sec["path"]:
                p = list(sec["path"])
                key = "/".join(p)
                if key in seen_paths:
                    continue
                seen_paths[key] = 1
                span = (line_starts[a] if a < len(line_starts) else len(text), line_starts[a] if a < len(line_starts) else len(text))
                units.append(make_unit(doc_key, role="ancestor", path=p, span=span, text="", unit_type="section"))
            continue
        p = list(sec["path"]) or ["(document)"]
        key = "/".join(p)
        if key in seen_paths:
            seen_paths[key] += 1
            p = p + [f"({seen_paths[key]})"]
        else:
            seen_paths[key] = 1
        blocks = _blocks(body_lines, line_starts[a:b])
        if not blocks:
            continue
        hl = sec.get("heading_line")
        sec_start = line_starts[hl] if hl is not None else blocks[0]["span"][0]
        sec_span = (sec_start, blocks[-1]["span"][1])
        sec_text = text[sec_span[0]:sec_span[1]]
        identity = " > ".join(p)
        if paragraph_primary:
            if len(blocks) == 1:
                units.append(make_unit(doc_key, role="primary", path=p, span=sec_span, text=sec_text,
                                       unit_type=blocks[0]["type"] if blocks[0]["type"] != "paragraph" else "section",
                                       keys={"heading": p[-1] if p else "", "identity": identity}))
                continue
            for bi, blk in enumerate(blocks):
                units.append(make_unit(doc_key, role="primary", path=p + [f"{blk['type']}{bi}"], span=blk["span"], text=blk["text"],
                                       unit_type=blk["type"], keys={"heading": p[-1] if p else "", "identity": identity}))
            continue
        primary = make_unit(doc_key, role="primary", path=p, span=sec_span, text=sec_text,
                            unit_type="section", keys={"heading": p[-1] if p else ""})
        units.append(primary)
        if len(blocks) == 1 and blocks[0]["type"] != "table":
            continue   # one block == the section itself; a piece would be a duplicate
        # Pieces: one per block, with tables further cut into row groups (header repeated).
        for bi, blk in enumerate(blocks):
            if blk["type"] == "table":
                header, data = table_rows(blk["text"])
                header_line = next((ln for ln in blk["text"].splitlines() if _TABLE_ROW_RE.match(ln)), "")
                sep_line = next((ln for ln in blk["text"].splitlines() if _TABLE_SEP_RE.match(ln)), "")
                # Row positions in the text.
                offset = blk["span"][0]
                row_pos: list[tuple[int, int, str]] = []
                cursor = 0
                for ln in blk["text"].split("\n"):
                    if ln in data and _TABLE_ROW_RE.match(ln):
                        row_pos.append((offset + cursor, offset + cursor + len(ln), ln))
                    cursor += len(ln) + 1
                if not row_pos:
                    continue
                for g in range(0, len(row_pos), ROWGROUP_ROWS):
                    grp = row_pos[g:g + ROWGROUP_ROWS]
                    gtext = "\n".join(r[2] for r in grp)
                    units.append(make_unit(
                        doc_key, role="piece", path=p + [f"table{bi}", f"rows{g // ROWGROUP_ROWS}"],
                        span=(grp[0][0], grp[-1][1]), text=gtext, unit_type="rowgroup", parent=primary,
                        keys={"identity": identity, "header": header, "header_line": header_line,
                              "sep_line": sep_line, "table_index": bi}))
                continue
            if len(blk["text"]) < _MIN_PIECE_CHARS and len(blocks) > 1 and blk["type"] == "paragraph":
                # A fragment: it stays inside the section text; no piece of its own.
                continue
            units.append(make_unit(
                doc_key, role="piece", path=p + [f"{blk['type']}{bi}"], span=blk["span"], text=blk["text"],
                unit_type=blk["type"], parent=primary, keys={"identity": identity}))
    return units
