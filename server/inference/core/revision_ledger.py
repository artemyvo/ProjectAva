"""Revisions ledger — every judgement about an exchange, kept aside for offline analysis.

Why. The `.state.json` sidecar holds the LATEST verdict per exchange and nothing else: a
re-reflection overwrites it, the revision pass's one-line WHY is parsed and discarded,
and whether the pass truncated or was retried is gone with the run log. That is the
right shape for the training build (which wants one target per exchange) and the
wrong shape for asking what distinguishes an exchange Ava later revised — the question
Track B of AVA_REWARD_LOOP.md exists to answer. So every write through the one verdict
seam (`chat_sidecar.ChatSidecar.write_verdict`, which Sleep, the background drain, the
headless CLI and the Training review hand-edits all share) also appends ONE EVENT here,
with everything the judgement saw and everything about how it was made, snapshotted at
write time so later edits or deletions of the transcript do not move the ground.

Where. ``server/data/revisions/ledger.jsonl`` — a sibling of ``chats/``, corpus-grade
(the wipe job treats `data/` as the corpus), append-only, NOT under `chats/` (the
sidecar-suffix indexing trap) and read by nothing on Ava's side: not the RAG engine,
not reflection, not the training build. These are labels about the judge and the
sensors (P2 / P9 of the note): they calibrate instruments, never her.

Event kinds:
  * ``verdict``    — a `write_verdict`: verdict, WHY, pass outcome, run, the previous
                     record it replaced, the target, and the exchange snapshot;
  * ``lock`` / ``ban`` — the Training review's curation toggles (provenance only);
  * ``annotation`` — a human calibration mark (the future review tab): is the verdict
                     justified, is the WHY accurate, a failure tag, a note. Never a
                     reward, never a target; see `append_annotation`.

``python -m core.revision_ledger [--chats DIR] [--json]`` summarizes the box's ledger;
``--selftest`` exercises the writer and reader on a throwaway corpus.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

LEDGER_DIRNAME = "revisions"
LEDGER_FILENAME = "ledger.jsonl"
# The revision pass's raw judgement is kept, capped: it is the one thing no other store
# holds and the first thing a reader wants when a WHY looks wrong.
RAW_CAP = 16000
# Failure tags the review tab offers. About the judge's reading of the reply, never
# about whether the reply was liked.
ANNOTATION_TAGS = ("sycophancy", "fluency_capture", "factual", "persona_drift",
                   "language_drift", "borrowed_voice", "hedging", "other")


def ledger_dir(chats_dir: Path) -> Path:
    return Path(chats_dir).parent / LEDGER_DIRNAME


def ledger_path(chats_dir: Path) -> Path:
    return ledger_dir(chats_dir) / LEDGER_FILENAME


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _append(chats_dir: Path, event: dict) -> bool:
    try:
        d = ledger_dir(chats_dir)
        d.mkdir(parents=True, exist_ok=True)
        with (d / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return True
    except Exception as e:
        print(f"[ledger] append failed: {e}")
        return False


# ── snapshot ─────────────────────────────────────────────────────────────────

def _tension_summary(tension: Any) -> Optional[dict]:
    """The per-segment stats and the per-axis means — never the token series."""
    if not isinstance(tension, dict) or not tension:
        return None
    out: dict = {"model_id": tension.get("model_id")}
    for seg in ("cot", "answer"):
        s = tension.get(seg)
        if isinstance(s, dict):
            out[seg] = {k: s.get(k) for k in ("n_tokens", "peak_entropy", "median_entropy",
                                              "median_margin", "p10_margin", "contested_frac")}
    axes = tension.get("axes")
    if isinstance(axes, dict) and axes:
        means = {}
        for name, series in axes.items():
            try:
                vals = [float(x) for x in series]
                if vals:
                    means[name] = {"mean": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}
            except Exception:
                continue
        if means:
            out["axes"] = means
    out["n_tokens"] = len(tension.get("token_ids") or [])
    return out


def exchange_snapshot(chat_path: Optional[Path], exchange_index: int) -> dict:
    """What the judgement was about, as the transcript holds it right now."""
    snap: dict = {"exchange_index": int(exchange_index)}
    if chat_path is None:
        return snap
    try:
        doc = json.loads(Path(chat_path).read_text(encoding="utf-8"))
    except Exception:
        return snap
    exs = doc.get("exchanges") or []
    snap.update({
        "session_timestamp": doc.get("timestamp"),
        "session_user": doc.get("user"),
        "model_id": doc.get("model_id"),
        "adapter_id": doc.get("adapter_id"),
        "initiated_by": doc.get("initiated_by"),
        "n_exchanges": len(exs),
    })
    if not (0 <= exchange_index < len(exs)) or not isinstance(exs[exchange_index], dict):
        return snap
    ex = exs[exchange_index]
    system = str(ex.get("system_content") or "")
    snap.update({
        "exchange_id": ex.get("exchange_id"),
        "speaker": ex.get("speaker"),
        "timestamp": ex.get("timestamp"),
        "user_prompt": ex.get("user_prompt"),
        "assistant_cot": ex.get("assistant_cot"),
        "assistant_response": ex.get("assistant_response"),
        "system_content": system,
        "system_content_sha": _sha(system),
        "rag_context": ex.get("rag_context"),
        "generation_params": ex.get("generation_params"),
        "input_tokens": ex.get("input_tokens"),
        "think_open_prob": ex.get("think_open_prob"),
        "reflection_feedback": ex.get("reflection_feedback"),
        "corrupt_cot": ex.get("corrupt_cot"),
        "corrupt_response": ex.get("corrupt_response"),
        "rewrite_history_n": len(ex.get("rewrite_history") or []),
        "tension": _tension_summary(ex.get("tension")),
    })
    # Which hidden-state captures exist for it (live, and any backfilled variant).
    try:
        from core import hidden_capture
        idx = hidden_capture.sidecar_index(hidden_capture.sidecar_path(Path(chat_path)))
        key = hidden_capture.exchange_key(ex, exchange_index)
        snap["hidden"] = [{"variant": g["variant"], "axes": g["axes"], "layers": g["layers"]}
                          for g in idx.values() if g["exchange_id"] == key]
    except Exception:
        snap["hidden"] = None
    return snap


# ── writers ──────────────────────────────────────────────────────────────────

def record_verdict(chats_dir: Path, *, source_session: str, exchange_index: int,
                   record: dict, prev: Optional[dict], chat_path: Optional[Path],
                   why: str = "", pass_info: Optional[dict] = None, run_kind: str = "",
                   judgement_raw: str = "") -> bool:
    """One event per `write_verdict`. `record` is the sidecar record as written, `prev`
    the one it replaced (None on first vetting)."""
    prev = prev if isinstance(prev, dict) else {}
    target = str(record.get("target") or "")
    event = {
        "kind": "verdict",
        "ts": _now(),
        "source_session": source_session,
        "exchange_index": int(exchange_index),
        "run_id": record.get("run_id"),
        "run_kind": run_kind or _infer_run_kind(record),
        "verdict": record.get("verdict"),
        "why": (why or "").strip(),
        "pass_info": pass_info or {},
        "target_source": record.get("target_source"),
        "target_kind": record.get("target_kind"),
        "target_generation": record.get("target_generation"),
        "target": target,
        "target_sha": _sha(target),
        "persona_context_chars": len(record.get("persona_context") or ""),
        "locked": bool(record.get("locked")),
        "banned": bool(record.get("banned")),
        "stage": record.get("stage"),
        "prev": ({"verdict": prev.get("verdict"), "run_id": prev.get("run_id"),
                  "target_sha": _sha(str(prev.get("target") or "")),
                  "last_reflected": prev.get("last_reflected")} if prev else None),
        "judgement_raw": (judgement_raw or "")[:RAW_CAP],
        "snapshot": exchange_snapshot(chat_path, exchange_index),
    }
    event["exchange_id"] = (event["snapshot"] or {}).get("exchange_id")
    return _append(chats_dir, event)


def _infer_run_kind(record: dict) -> str:
    rid = str(record.get("run_id") or "")
    kind = str(record.get("target_kind") or "")
    if kind == "manual_regen":
        return "manual_regen"
    if rid.startswith("bg_"):
        return "background"
    if rid.endswith("_rv"):
        return "revisit"
    return "reflection" if rid else ""


def record_flag(chats_dir: Path, kind: str, *, source_session: str, exchange_index: int,
                value: bool, chat_path: Optional[Path] = None) -> bool:
    """A curation toggle (`lock` / `ban`) from the Training review tab — provenance."""
    snap = exchange_snapshot(chat_path, exchange_index) if chat_path is not None else {}
    return _append(chats_dir, {
        "kind": kind, "ts": _now(), "source_session": source_session,
        "exchange_index": int(exchange_index), "value": bool(value),
        "exchange_id": snap.get("exchange_id"),
    })


def append_annotation(chats_dir: Path, *, source_session: str, exchange_index: int,
                      exchange_id: Optional[str], author: str,
                      verdict_justified: Optional[bool], why_accurate: Optional[bool],
                      tag: str = "", note: str = "", about_run_id: Optional[str] = None) -> bool:
    """A human calibration mark. About the JUDGE's reading (was the verdict justified,
    does the WHY describe what went wrong, which failure kind) — never about whether
    the reply was liked, and never read by reflection or training (P9)."""
    tag = (tag or "").strip()
    if tag and tag not in ANNOTATION_TAGS:
        tag = "other:" + tag
    return _append(chats_dir, {
        "kind": "annotation", "ts": _now(), "source_session": source_session,
        "exchange_index": int(exchange_index), "exchange_id": exchange_id,
        "author": author or "operator", "about_run_id": about_run_id,
        "verdict_justified": verdict_justified, "why_accurate": why_accurate,
        "tag": tag, "note": (note or "").strip(),
    })


# ── readers ──────────────────────────────────────────────────────────────────

def read_events(chats_dir: Path) -> list:
    path = ledger_path(chats_dir)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def summarize(events: list) -> dict:
    kinds = Counter(e.get("kind") for e in events)
    verdicts = [e for e in events if e.get("kind") == "verdict"]
    by_verdict = Counter(e.get("verdict") for e in verdicts)
    by_run_kind = Counter(e.get("run_kind") for e in verdicts)
    with_why = sum(1 for e in verdicts if e.get("why"))
    flips = sum(1 for e in verdicts if e.get("prev") and e["prev"].get("verdict")
                and e["prev"]["verdict"] != e.get("verdict"))
    truncated = sum(1 for e in verdicts if (e.get("pass_info") or {}).get("truncated"))
    retried = sum(1 for e in verdicts if (e.get("pass_info") or {}).get("retried"))
    annotations = [e for e in events if e.get("kind") == "annotation"]
    tags = Counter(a.get("tag") for a in annotations if a.get("tag"))
    justified = [a.get("verdict_justified") for a in annotations if a.get("verdict_justified") is not None]
    distinct = {(e.get("source_session"), e.get("exchange_index")) for e in verdicts}
    return {
        "events": len(events), "kinds": dict(kinds),
        "verdict_events": len(verdicts), "distinct_exchanges": len(distinct),
        "by_verdict": dict(by_verdict), "by_run_kind": dict(by_run_kind),
        "with_why": with_why, "flips": flips, "truncated": truncated, "retried": retried,
        "annotations": len(annotations), "tags": dict(tags),
        "verdict_justified_rate": (sum(1 for j in justified if j) / len(justified)) if justified else None,
        "first": min((e.get("ts") or "" for e in events), default=""),
        "last": max((e.get("ts") or "" for e in events), default=""),
    }


def render(summary: dict, path: Path) -> str:
    s = summary
    lines = [f"REVISIONS LEDGER — {path}",
             f"  events {s['events']}  ({', '.join(f'{k} {v}' for k, v in sorted(s['kinds'].items()))})"
             + (f"  {s['first']} … {s['last']}" if s["first"] else ""),
             f"  verdict events {s['verdict_events']} over {s['distinct_exchanges']} exchanges: "
             + (", ".join(f"{k or '?'} {v}" for k, v in sorted(s["by_verdict"].items())) or "—"),
             f"  by run kind: " + (", ".join(f"{k or '?'} {v}" for k, v in sorted(s["by_run_kind"].items())) or "—"),
             f"  with WHY {s['with_why']}  ·  verdict flips vs previous {s['flips']}  ·  "
             f"pass truncated {s['truncated']}  ·  retried {s['retried']}",
             f"  annotations {s['annotations']}"
             + (f"  tags: " + ", ".join(f"{k} {v}" for k, v in sorted(s["tags"].items())) if s["tags"] else "")
             + (f"  ·  verdict justified {s['verdict_justified_rate'] * 100:.0f}%"
                if s["verdict_justified_rate"] is not None else "")]
    return "\n".join(lines)


# ── CLI + self-test ──────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        chat = chats / "20260922_100000.json"
        chat.write_text(json.dumps({
            "timestamp": "20260922_100000", "user": "Pavel", "model_id": "m", "adapter_id": "a",
            "exchanges": [{"exchange_id": "e0", "user_prompt": "u", "assistant_cot": "c",
                           "assistant_response": "r", "system_content": "S" * 50, "rag_context": "R",
                           "tension": {"token_ids": [1, 2], "entropies": [1, 1], "margins": [0.9, 0.9],
                                       "cot": {"n_tokens": 1, "peak_entropy": 2.0},
                                       "answer": {"n_tokens": 1, "contested_frac": 0.0},
                                       "axes": {"pain": [0.5, 1.5]}}}]}))
        rec = {"verdict": "revise", "target": "T", "run_id": "20260922_120000", "target_source": "revised",
               "target_kind": "ideal", "target_generation": "g", "persona_context": "pp", "locked": False,
               "banned": False, "stage": 0}
        assert record_verdict(chats, source_session=chat.name, exchange_index=0, record=rec, prev=None,
                              chat_path=chat, why="borrowed voice", pass_info={"truncated": False},
                              judgement_raw="VERDICT: revise\nWHY: borrowed voice")
        rec2 = dict(rec, verdict="keep", run_id="bg_20260923_010000", target="r")
        assert record_verdict(chats, source_session=chat.name, exchange_index=0, record=rec2, prev=rec,
                              chat_path=chat, why="already mine")
        assert record_flag(chats, "ban", source_session=chat.name, exchange_index=0, value=True, chat_path=chat)
        assert append_annotation(chats, source_session=chat.name, exchange_index=0, exchange_id="e0",
                                 author="op", verdict_justified=False, why_accurate=True, tag="sycophancy")
        assert append_annotation(chats, source_session=chat.name, exchange_index=0, exchange_id="e0",
                                 author="op", verdict_justified=True, why_accurate=None, tag="weird")
        ev = read_events(chats)
        assert len(ev) == 5 and ledger_path(chats).parent.name == "revisions"
        v0, v1 = ev[0], ev[1]
        assert v0["kind"] == "verdict" and v0["why"] == "borrowed voice" and v0["prev"] is None
        assert v0["snapshot"]["assistant_response"] == "r" and v0["snapshot"]["system_content_sha"]
        assert v0["snapshot"]["tension"]["axes"]["pain"]["mean"] == 1.0 and "token_ids" not in json.dumps(v0["snapshot"]["tension"])
        assert v0["exchange_id"] == "e0" and v0["run_kind"] == "reflection"
        assert v1["prev"]["verdict"] == "revise" and v1["run_kind"] == "background"
        assert ev[4]["tag"] == "other:weird"
        s = summarize(ev)
        assert s["verdict_events"] == 2 and s["distinct_exchanges"] == 1 and s["flips"] == 1
        assert s["by_verdict"] == {"revise": 1, "keep": 1} and s["with_why"] == 2
        assert s["annotations"] == 2 and s["tags"] == {"sycophancy": 1, "other:weird": 1}
        assert s["verdict_justified_rate"] == 0.5
        assert "REVISIONS LEDGER" in render(s, ledger_path(chats))
        # a missing transcript still records the event, with a thin snapshot
        assert record_verdict(chats, source_session="nope.json", exchange_index=3, record=rec, prev=None,
                              chat_path=chats / "nope.json")
        assert read_events(chats)[-1]["snapshot"] == {"exchange_index": 3}
        assert read_events(Path(td) / "other" / "chats") == []   # a different data root
    print("revision_ledger self-test: OK")


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Summarize the revisions ledger (read-only).")
    ap.add_argument("--chats", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    chats = args.chats
    if chats is None:
        server_dir = Path(__file__).resolve().parents[2]
        if str(server_dir) not in sys.path:
            sys.path.insert(0, str(server_dir))
        from training.reflections_path import hot_chats_dir
        chats = hot_chats_dir()
    events = read_events(chats)
    summary = summarize(events)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(render(summary, ledger_path(chats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
