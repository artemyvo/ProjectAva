"""Revision review — the server side of the Revisions tab (AVA_REWARD_LOOP.md §7 item 17).

Three read-mostly messages over the revisions ledger (`core.revision_ledger`):

  list_revisions      {verdict?, run_kind?, annotated?, text?, limit?}  → revisions_list
  get_revision        {session, exchange_index}                          → revision_detail
  annotate_revision   {session, exchange_index, exchange_id?, about_run_id?,
                       verdict_justified?, why_accurate?, tag?, note?}  → revision_annotated

The list is one row per judged exchange — its LATEST verdict event, how many times it
was judged, and whether a human has marked it. The detail is the full history of
verdict events for that exchange, the latest one's snapshot (the exchange as the judge
saw it), the original reply decoded into the chat's three colour channels from the
transcript's stored token series (`generation.stored_tension_spans`, injected — the
client has no tokenizer), the sidecar record as it stands now, and every annotation.

The tab's purpose is to judge the judge: the annotation asks whether the verdict was
justified and whether the WHY describes what went wrong, plus a failure tag. It never
asks whether the reply was liked, and what it writes (`revision_ledger.append_annotation`)
is read by nothing on Ava's side — P9 of the note. Never imports server: `configure`
injects send, the chats dir and the span decoder. `python -m core.revision_review`
self-tests the pure builders on a throwaway ledger.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from core import revision_ledger

_send = None
_CHATS_DIR: Optional[Path] = None
_tension_spans: Optional[Callable[[Any], Optional[dict]]] = None

WHY_HEAD = 160
LIST_DEFAULT_LIMIT = 500


def configure(*, send, chats_dir, tension_spans=None) -> None:
    """Wire the capabilities this module needs (once, from server.main())."""
    global _send, _CHATS_DIR, _tension_spans
    _send = send
    _CHATS_DIR = Path(chats_dir)
    _tension_spans = tension_spans


# ── pure builders ────────────────────────────────────────────────────────────

def _key(e: dict) -> tuple:
    return (str(e.get("source_session") or ""), int(e.get("exchange_index") or 0))


def index_events(events: list) -> tuple:
    """(verdicts_by_key newest-last, annotations_by_key, flags_by_key)."""
    verdicts: dict = defaultdict(list)
    annotations: dict = defaultdict(list)
    flags: dict = defaultdict(list)
    for e in events:
        kind = e.get("kind")
        if kind == "verdict":
            verdicts[_key(e)].append(e)
        elif kind == "annotation":
            annotations[_key(e)].append(e)
        elif kind in ("lock", "ban"):
            flags[_key(e)].append(e)
    return verdicts, annotations, flags


def _row(key: tuple, history: list, notes: list, flags: list) -> dict:
    latest = history[-1]
    snap = latest.get("snapshot") or {}
    why = str(latest.get("why") or "")
    last_note = notes[-1] if notes else None
    return {
        "session": key[0],
        "exchange_index": key[1],
        "exchange_id": latest.get("exchange_id") or snap.get("exchange_id"),
        "ts": latest.get("ts"),
        "run_id": latest.get("run_id"),
        "run_kind": latest.get("run_kind"),
        "verdict": latest.get("verdict"),
        "prev_verdict": (latest.get("prev") or {}).get("verdict"),
        "n_judged": len(history),
        "why_head": why if len(why) <= WHY_HEAD else why[:WHY_HEAD - 1] + "…",
        "has_why": bool(why),
        "target_kind": latest.get("target_kind"),
        "pass_info": latest.get("pass_info") or {},
        "locked": bool(latest.get("locked")),
        "banned": bool(latest.get("banned")),
        "speaker": snap.get("speaker") or snap.get("session_user"),
        "user_head": (str(snap.get("user_prompt") or "")[:120]),
        "model_id": snap.get("model_id"),
        "adapter_id": snap.get("adapter_id"),
        "has_series": bool((snap.get("tension") or {}).get("n_tokens")),
        "has_hidden": bool(snap.get("hidden")),
        "n_annotations": len(notes),
        "last_annotation": ({"ts": last_note.get("ts"), "verdict_justified": last_note.get("verdict_justified"),
                             "why_accurate": last_note.get("why_accurate"), "tag": last_note.get("tag")}
                            if last_note else None),
        "flags": [{"kind": f.get("kind"), "value": f.get("value"), "ts": f.get("ts")} for f in flags],
    }


def build_list(events: list, *, verdict: str = "", run_kind: str = "", annotated: str = "",
               text: str = "", limit: int = LIST_DEFAULT_LIMIT) -> dict:
    """Rows newest-first, filtered. `annotated`: "" / "yes" / "no"."""
    verdicts, annotations, flags = index_events(events)
    rows = []
    for key, history in verdicts.items():
        row = _row(key, history, annotations.get(key, []), flags.get(key, []))
        if verdict and row["verdict"] != verdict:
            continue
        if run_kind and row["run_kind"] != run_kind:
            continue
        if annotated == "yes" and not row["n_annotations"]:
            continue
        if annotated == "no" and row["n_annotations"]:
            continue
        if text:
            hay = " ".join(str(row.get(k) or "") for k in ("session", "why_head", "user_head", "speaker",
                                                             "run_id", "exchange_id")).lower()
            if text.lower() not in hay:
                continue
        rows.append(row)
    rows.sort(key=lambda r: (r.get("ts") or "", r["session"], r["exchange_index"]), reverse=True)
    total = len(rows)
    rows = rows[:max(1, int(limit))] if limit else rows
    run_kinds = sorted({str(e.get("run_kind") or "") for h in verdicts.values() for e in h} - {""})
    return {"items": rows, "total": total, "run_kinds": run_kinds,
            "tags": list(revision_ledger.ANNOTATION_TAGS),
            "summary": revision_ledger.summarize(events)}


def build_detail(events: list, session: str, exchange_index: int, *, spans: Optional[dict] = None,
                 sidecar_record: Optional[dict] = None) -> Optional[dict]:
    """Everything the tab shows for one exchange, or None when it was never judged."""
    verdicts, annotations, flags = index_events(events)
    key = (str(session), int(exchange_index))
    history = verdicts.get(key)
    if not history:
        return None
    latest = history[-1]
    return {
        "session": key[0], "exchange_index": key[1],
        "exchange_id": latest.get("exchange_id"),
        "latest": latest,
        "history": [{k: e.get(k) for k in ("ts", "run_id", "run_kind", "verdict", "why", "pass_info",
                                           "target_source", "target_kind", "target_sha", "prev", "locked",
                                           "banned")} for e in history],
        "annotations": annotations.get(key, []),
        "flags": flags.get(key, []),
        "tension_spans": spans,
        "sidecar": sidecar_record,
        "tags": list(revision_ledger.ANNOTATION_TAGS),
    }


# ── handlers ─────────────────────────────────────────────────────────────────

def _read_events() -> list:
    return revision_ledger.read_events(_CHATS_DIR) if _CHATS_DIR is not None else []


def _original_spans_and_record(session: str, exchange_index: int) -> tuple:
    """Blocking: decode the original reply's stored series into spans (a tokenizer may
    load on first use) and read the sidecar record as it stands."""
    spans = None
    record = None
    if _CHATS_DIR is None:
        return spans, record
    try:
        import json
        chat = (_CHATS_DIR / session).resolve()
        if chat.parent == _CHATS_DIR.resolve() and chat.exists():
            doc = json.loads(chat.read_text(encoding="utf-8"))
            exs = doc.get("exchanges") or []
            if 0 <= exchange_index < len(exs) and isinstance(exs[exchange_index], dict):
                block = exs[exchange_index].get("tension")
                if _tension_spans is not None and isinstance(block, dict):
                    spans = _tension_spans(block)
    except Exception as e:
        print(f"[review] spans for {session}#{exchange_index} skipped: {e}")
    try:
        from core.chat_sidecar import ChatSidecar
        record = ChatSidecar(_CHATS_DIR).get_exchange(session, int(exchange_index))
    except Exception:
        record = None
    return spans, record


async def handle_list_revisions(ws, msg: dict) -> None:
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(None, _read_events)
    result = build_list(
        events,
        verdict=str(msg.get("verdict") or ""),
        run_kind=str(msg.get("run_kind") or ""),
        annotated=str(msg.get("annotated") or ""),
        text=str(msg.get("text") or ""),
        limit=int(msg.get("limit") or LIST_DEFAULT_LIMIT),
    )
    result["type"] = "revisions_list"
    result["ledger"] = str(revision_ledger.ledger_path(_CHATS_DIR)) if _CHATS_DIR is not None else ""
    await _send(ws, result)


async def handle_get_revision(ws, msg: dict) -> None:
    session = str(msg.get("session") or "")
    try:
        index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "exchange_index required"})
        return
    if not session or "/" in session or "\\" in session:
        await _send(ws, {"type": "error", "message": "Invalid session."})
        return
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(None, _read_events)
    spans, record = await loop.run_in_executor(None, _original_spans_and_record, session, index)
    detail = build_detail(events, session, index, spans=spans, sidecar_record=record)
    if detail is None:
        await _send(ws, {"type": "error", "message": f"No verdict on record for {session} #{index + 1}."})
        return
    detail["type"] = "revision_detail"
    await _send(ws, detail)


def _tri(value) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("yes", "true", "1"):
        return True
    if s in ("no", "false", "0"):
        return False
    return None


async def handle_annotate_revision(ws, msg: dict) -> None:
    """A human calibration mark. About the judge's reading — see the module docstring."""
    session = str(msg.get("session") or "")
    try:
        index = int(msg.get("exchange_index"))
    except (TypeError, ValueError):
        await _send(ws, {"type": "error", "message": "exchange_index required"})
        return
    if not session or "/" in session or "\\" in session or _CHATS_DIR is None:
        await _send(ws, {"type": "error", "message": "Invalid session."})
        return
    vj = _tri(msg.get("verdict_justified"))
    wa = _tri(msg.get("why_accurate"))
    tag = str(msg.get("tag") or "").strip()
    note = str(msg.get("note") or "").strip()
    if vj is None and wa is None and not tag and not note:
        await _send(ws, {"type": "error", "message": "Nothing to record: mark at least one thing."})
        return
    ok = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: revision_ledger.append_annotation(
            _CHATS_DIR, source_session=session, exchange_index=index,
            exchange_id=(str(msg.get("exchange_id")) if msg.get("exchange_id") else None),
            author=str(msg.get("author") or "operator"),
            verdict_justified=vj, why_accurate=wa, tag=tag, note=note,
            about_run_id=(str(msg.get("about_run_id")) if msg.get("about_run_id") else None),
        ),
    )
    await _send(ws, {"type": "revision_annotated", "ok": bool(ok), "session": session,
                     "exchange_index": index})


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        rec = {"verdict": "revise", "target": "T", "run_id": "20260922_120000", "target_source": "revised",
               "target_kind": "ideal", "target_generation": "g", "persona_context": "", "locked": False,
               "banned": False, "stage": 0}
        chat = chats / "s1.json"
        chat.write_text(json.dumps({"user": "Pavel", "exchanges": [
            {"exchange_id": "e0", "user_prompt": "hello there", "assistant_response": "r0"},
            {"exchange_id": "e1", "user_prompt": "second", "assistant_response": "r1"}]}))
        revision_ledger.record_verdict(chats, source_session="s1.json", exchange_index=0, record=rec, prev=None,
                                       chat_path=chat, why="borrowed voice", pass_info={"truncated": False})
        rec2 = dict(rec, verdict="keep", run_id="bg_20260923_010000", target="r0")
        revision_ledger.record_verdict(chats, source_session="s1.json", exchange_index=0, record=rec2, prev=rec,
                                       chat_path=chat, why="already mine")
        revision_ledger.record_verdict(chats, source_session="s1.json", exchange_index=1, record=rec, prev=None,
                                       chat_path=chat, why="hedged", pass_info={"truncated": True})
        revision_ledger.record_flag(chats, "ban", source_session="s1.json", exchange_index=1, value=True, chat_path=chat)
        revision_ledger.append_annotation(chats, source_session="s1.json", exchange_index=1, exchange_id="e1",
                                          author="op", verdict_justified=False, why_accurate=True, tag="hedging")
        events = revision_ledger.read_events(chats)

        lst = build_list(events)
        assert lst["total"] == 2 and len(lst["items"]) == 2
        by_idx = {r["exchange_index"]: r for r in lst["items"]}
        assert by_idx[0]["verdict"] == "keep" and by_idx[0]["prev_verdict"] == "revise" and by_idx[0]["n_judged"] == 2
        assert by_idx[0]["run_kind"] == "background" and by_idx[0]["n_annotations"] == 0
        assert by_idx[1]["n_annotations"] == 1 and by_idx[1]["last_annotation"]["tag"] == "hedging"
        assert by_idx[1]["flags"][0]["kind"] == "ban" and by_idx[1]["pass_info"]["truncated"] is True
        assert by_idx[0]["user_head"] == "hello there" and by_idx[0]["speaker"] == "Pavel"
        assert lst["run_kinds"] == ["background", "reflection"]
        assert [r["exchange_index"] for r in build_list(events, verdict="revise")["items"]] == [1]
        assert [r["exchange_index"] for r in build_list(events, annotated="no")["items"]] == [0]
        assert [r["exchange_index"] for r in build_list(events, annotated="yes")["items"]] == [1]
        assert [r["exchange_index"] for r in build_list(events, text="HEDGED")["items"]] == [1]
        assert len(build_list(events, limit=1)["items"]) == 1 and build_list(events, limit=1)["total"] == 2

        d = build_detail(events, "s1.json", 0, spans={"cot": None, "answer": [["r0", 0.9, 0.0, None]]},
                         sidecar_record={"verdict": "keep"})
        assert d is not None and len(d["history"]) == 2 and d["latest"]["verdict"] == "keep"
        assert d["history"][0]["verdict"] == "revise" and d["latest"]["snapshot"]["user_prompt"] == "hello there"
        assert d["tension_spans"]["answer"][0][0] == "r0" and d["sidecar"] == {"verdict": "keep"}
        assert build_detail(events, "s1.json", 7) is None
        d1 = build_detail(events, "s1.json", 1)
        assert d1["annotations"][0]["tag"] == "hedging" and d1["flags"][0]["kind"] == "ban"
        assert _tri("yes") is True and _tri("no") is False and _tri("") is None and _tri(True) is True
    print("revision_review self-test: OK")


if __name__ == "__main__":
    _selftest()
