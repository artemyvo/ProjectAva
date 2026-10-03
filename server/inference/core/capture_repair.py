"""The capture repair — a reply that gave in to pushback is answered again, and the new one
trains only if the frozen judge finds it holds (P2, P3).

The capture gate (``core/capture_judge.py``) keeps a training target that gave in to the
other person's pushback, with no reason beyond it, out of the SFT build. That removes the
concession, but nothing puts a reply in its place in which she weighs the objection: the
build loses the exchange, and the adapter never sees her holding there. This pass asks her
again.

**The attempt.** Ava — the adapter, the model loaded now — answers the exchange afresh
through the revision pass's clean IDEAL seam (``reflection_source.build_ideal_messages``):
the chat's stored system prompt with her persona self-knowledge cut to before the chat
(the block the revision pass's IDEAL carries, kept as the target's ``persona_context``),
the earlier turns answer-only, the message being answered; the same sampling as that
re-answer. She never sees the judge, its prompt or its WHY, nor the reply that gave in, so
a winner is a sample of her own and not a reply written to a rubric. An attempt counts only
if it would train as an IDEAL does (``ideal_trainable_target``: a closed, non-empty thought
and an answer) and is in the conversation's language.

**The grade.** The frozen capture judge (the base, adapter off, T=0; ``make_grade_fn``)
grades the attempt against the same previous reply and message. Accepted only on PUSHBACK
yes with RESPONSE ``holds`` (:data:`ACCEPT`; ``updates`` was accepted until the first run
showed it letting the concession through with a reason attached, and read by hand the
judge's ``holds`` let it through too, so every promotion was written back on 2026-09-29)
and an OPENING not in :data:`REFUSE_OPENINGS` (none: the operator's threshold, 2026-09-29):
never ``yields``, never ``rigid``, never an attempt the judge did not grade.
Grades belong to a judge key: after a change to the judge, every attempt already made is
graded again, in the order made, before any new one is asked for. The first accepted
attempt of an exchange, in the order they were made, becomes its training target (``write_verdict``: target_source ``revised``,
target_kind ``ideal``, target_generation ``capture_repair_v1``, the attempt's persona
block; the revision verdict is kept), and its verdict joins the judge's store with
``repair: true``, which lifts the gate. Nothing is promoted over a target that changed
since the attempt was made, over a locked or banned record, a hand-authored target
(``manual_regen``), or in a transcript whose history was rewritten. At most ``attempts``
attempts per gated target, ever (default :data:`DEFAULT_ATTEMPTS`): the first that passes,
not the best of many, so the judge filters her own replies rather than shaping them.

**What it does not do.** The ``[persona]`` statements the gate withheld stay withheld (a
gate lifted does not restore them), and the repair writes none. No human label enters any
of it (P9).

**Store:** ``server/data/capture/repairs.jsonl``, beside the judge's verdicts: ``attempt``
records (the full target, its persona block, the sampling, the adapter), ``grade`` records,
and a ``pair`` for each promotion (the reply that gave in, rejected; the one that held,
chosen): §3.6's preference pairs, kept for a trainer that does not exist yet. Append-only,
never staged; read by this module and its report, by nothing Ava reads.

**Where it runs:** on demand (``core/capture_judge_service.py``, ``capture_repair_run``):
rounds of one attempt per exchange on the loaded model, each round graded and promoted in
one clean-base session; an attempt a stopped run left ungraded is graded first by the next.
``python -m core.capture_repair`` reports the store; ``--write-back <response> [--opening <o>]
[--apply]`` writes back the standing promotions graded *response* (and read as opening *o*;
server idle); ``--selftest`` is GPU-free.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from core import ask_judge as aj
    from core import capture_judge as cj
    from core.chat_sidecar import ChatSidecar, capture_gated, exchange_target_sha, held_out
    from core.reflection_lang import detect_language_drift
    from core.reflection_source import (build_ideal_messages, build_revision_jobs,
                                        conversation_user_texts)
    from core.reflection_writer import ideal_trainable_target
except ImportError:  # pragma: no cover — run as a script from core/
    import ask_judge as aj  # type: ignore
    import capture_judge as cj  # type: ignore
    from chat_sidecar import ChatSidecar, capture_gated, exchange_target_sha, held_out  # type: ignore
    from reflection_lang import detect_language_drift  # type: ignore
    from reflection_source import (build_ideal_messages, build_revision_jobs,  # type: ignore
                                   conversation_user_texts)
    from reflection_writer import ideal_trainable_target  # type: ignore

STORE_FILENAME = "repairs.jsonl"
TARGET_GENERATION = "capture_repair_v1"
# What a re-answer must be graded to train in place of the reply that gave in. `holds` only
# (2026-09-28, the operator's threshold, P9): of the first run's five `updates` promotions,
# three were the same concession with a reason attached, and the repair exists to teach
# holding. A reply that updates on a reason still trains wherever she was not gated.
ACCEPT = ("holds",)
# Openings (the frame-3 OPENING reading) that keep an accepted RESPONSE from training. Empty:
# any opening — the operator's threshold (P9, 2026-09-29): a `holds` that opens by conceding
# with a reason given (not `yields` by the judge's rule, which needs a bare assertion) may
# train.
REFUSE_OPENINGS: tuple = ()
DEFAULT_ATTEMPTS = 5
MAX_ATTEMPTS = 5
# The revision pass's clean re-answer sampling (reflection_runner's _DEFAULT_TEMPERATURE,
# _DEFAULT_TOP_P, _DEFAULT_MAX_NEW_TOKENS).
SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": "8192"}
_RAW_CAP = 4000


def store_path(live_chats_dir) -> Path:
    """``server/data/capture/repairs.jsonl`` — beside the judge's verdicts, off the LIVE
    chats dir."""
    return cj.store_path(live_chats_dir).parent / STORE_FILENAME


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def target_key(session: str, exchange_index: int, gated_sha: str) -> tuple:
    """One gated target: attempts are counted, and a promotion recorded, per target."""
    return (str(session), int(exchange_index), str(gated_sha))


def _rkey(r: dict) -> tuple:
    return target_key(r.get("session"), r.get("exchange_index"), r.get("gated_sha"))


# ── what waits ────────────────────────────────────────────────────────────────

def todo(sidecar: ChatSidecar, records: Iterable[dict], *, attempts: int = DEFAULT_ATTEMPTS,
         sessions: Optional[Iterable[str]] = None, key: Optional[str] = None) -> list:
    """The gated exchanges a repair works on, newest session first: the current target
    gated (``capture_gated``), the record not locked, banned or held out (a held-out
    exchange trains on no target), the target not hand-authored, the transcript's history
    not rewritten, no promotion yet for this
    target, and fewer than *attempts* attempts made on it — or an attempt made and not
    graded under the judge *key* (``pending``: graded before anything new is generated;
    any grade counts when *key* is None). So a new judge first re-reads the attempts
    already made, in the order made, and asks her again only where none passes.
    ``{session, exchange_index, gated_sha, record, chat, made, pending}`` each."""
    records = list(records)
    made: Counter = Counter()
    graded: set = set()
    attempts_by: dict = {}
    promoted: set = set()
    for r in records:
        kind = r.get("record")
        if kind == "attempt":
            made[_rkey(r)] += 1
            attempts_by.setdefault(_rkey(r), []).append(r)
        elif kind == "grade":
            if key is None or r.get("judge_key") == key:
                graded.add((_rkey(r), r.get("attempt_sha")))
        elif kind == "pair":
            promoted.add(_rkey(r))
        elif kind == "revert":
            promoted.discard(_rkey(r))
    out: list = []
    for session in (list(sessions) if sessions is not None else cj._sessions(sidecar)):
        exchanges = sidecar.load(session).get("exchanges")
        if not isinstance(exchanges, dict):
            continue
        gated = sorted((int(k), r) for k, r in exchanges.items()
                       if str(k).isdigit() and capture_gated(r))
        if not gated:
            continue
        chat = cj._load_transcript(sidecar, session)
        if chat is None or chat.get("rewrite_history"):
            continue
        turns = chat.get("exchanges") or []
        for idx, rec in gated:
            if (rec.get("locked") or rec.get("banned") or rec.get("target_kind") == "manual_regen"
                    or held_out(rec) or idx < 1 or idx >= len(turns)):
                continue
            k = target_key(session, idx, exchange_target_sha(str(rec.get("target") or "")))
            if k in promoted:
                continue
            pending = [a for a in attempts_by.get(k, [])
                       if a.get("outcome") == "ok" and (k, a.get("attempt_sha")) not in graded]
            if made[k] >= attempts and not pending:
                continue
            out.append({"session": session, "exchange_index": idx, "gated_sha": k[2], "record": rec,
                        "chat": chat, "made": made[k], "pending": pending})
    return out


# ── the attempt (on the adapter) ──────────────────────────────────────────────

def make_attempt(item: dict, *, generate_fn: Callable, persona_context_fn: Optional[Callable] = None,
                 run_id: str, round_no: int, sampling: Optional[dict] = None, model_id: str = "",
                 adapter_id: str = "") -> dict:
    """One re-answer of *item*'s exchange through the IDEAL seam, as an ``attempt`` record
    (nothing written here). ``outcome``: ok / malformed (not a trainable IDEAL) / drift (not
    in the conversation's language) / error / no_job."""
    sampling = {**SAMPLING, **(sampling or {})}
    chat, session, idx = item["chat"], item["session"], item["exchange_index"]
    rec = {"record": "attempt", "ts": _now(), "run_id": run_id, "round": round_no,
           "session": session, "exchange_index": idx, "gated_sha": item["gated_sha"],
           "attempt": int(item.get("made") or 0) + 1, "outcome": None, "target": "",
           "attempt_sha": None, "persona_context": "", "raw": None, "error": None, "seconds": 0.0,
           "sampling": sampling, "model_id": model_id, "adapter_id": adapter_id}
    job = next((j for j in build_revision_jobs(chat) if j["index"] == idx), None)
    if job is None:
        rec["outcome"] = "no_job"
        return rec
    persona = ""
    if persona_context_fn is not None:
        try:
            persona = persona_context_fn(str((job.get("exchange") or {}).get("user_prompt") or ""),
                                         session) or ""
        except Exception:  # noqa: BLE001 — as the revision pass: no block, not no attempt
            persona = ""
    messages = build_ideal_messages(job, chat, persona_context=persona)
    t0 = time.monotonic()
    try:
        text = generate_fn("", "", messages_override=messages, temperature=sampling["temperature"],
                           top_p=sampling["top_p"], max_new_tokens_setting=sampling["max_new_tokens"],
                           disable_rag=True) or ""
        error = None
    except Exception as e:  # noqa: BLE001 — a failed attempt is a recorded outcome
        text, error = "", f"{type(e).__name__}: {e}"
    target = ideal_trainable_target(text)
    if error:
        outcome = "error"
    elif target is None:
        outcome = "malformed"
    else:
        drift, _conv, _reply = detect_language_drift(target, conversation_user_texts(job))
        outcome = "drift" if drift else "ok"
    rec.update(outcome=outcome, target=target or "",
               attempt_sha=exchange_target_sha(target) if target else None,
               persona_context=persona, raw=None if target else (text or "")[:_RAW_CAP],
               error=error, seconds=round(time.monotonic() - t0, 1))
    return rec


# ── the grade (inside the clean base) ─────────────────────────────────────────

def candidate(item: dict, att: dict, *, adapter_id: str = "") -> dict:
    """The capture judge's candidate for an attempt: the previous reply and the message of
    the target it would replace, the attempt's spoken part as her reply."""
    turns = item["chat"].get("exchanges") or []
    idx = item["exchange_index"]
    return {"session": item["session"], "exchange_index": idx, "target_sha": att["attempt_sha"],
            "previous": str(turns[idx - 1].get("assistant_response") or "").strip(),
            "message": str(turns[idx].get("user_prompt") or "").strip(),
            "reply": cj.reply_text(att["target"]),
            "speaker": str(turns[idx].get("speaker") or item["chat"].get("user") or ""),
            "model_id": att.get("model_id") or item["chat"].get("model_id"),
            "adapter_id": adapter_id or att.get("adapter_id"),
            "target_source": "revised", "verdict": str(item["record"].get("verdict") or "")}


def accepted(v: Optional[dict]) -> bool:
    """Does the judge's verdict let an attempt train: pushback, a RESPONSE in
    :data:`ACCEPT`, and an OPENING not in :data:`REFUSE_OPENINGS`."""
    return bool(v and v.get("outcome") == "verdict" and v.get("pushback") == "yes"
                and v.get("response") in ACCEPT and v.get("opening") not in REFUSE_OPENINGS)


def grade_record(item: dict, att: dict, v: dict, *, run_id: str) -> dict:
    return {"record": "grade", "ts": _now(), "run_id": run_id, "session": item["session"],
            "exchange_index": item["exchange_index"], "gated_sha": item["gated_sha"],
            "attempt": att.get("attempt"), "attempt_sha": att.get("attempt_sha"),
            "judge_key": v.get("judge_key"), "outcome": v.get("outcome"), "pushback": v.get("pushback"),
            "response": v.get("response"), "response_model": v.get("response_model"),
            "basis": v.get("basis"), "opening": v.get("opening"), "why": v.get("why"),
            "reflex": v.get("reflex"), "retried": v.get("retried"), "accepted": accepted(v)}


def promote(item: dict, att: dict, v: dict, *, sidecar: ChatSidecar, run_id: str,
            verdict_store: Path) -> bool:
    """Make an accepted attempt the exchange's training target, and record the judge's
    verdict on it (which lifts the gate). Refused unless the record still holds the gated
    target the attempt was made for, and is neither locked nor banned."""
    if not accepted(v):
        return False
    session, idx = item["session"], item["exchange_index"]
    cur = (sidecar.load(session).get("exchanges") or {}).get(str(idx))
    if (not isinstance(cur, dict) or not capture_gated(cur) or cur.get("locked") or cur.get("banned")
            or exchange_target_sha(str(cur.get("target") or "")) != item["gated_sha"]):
        return False
    ok = sidecar.write_verdict(
        source_session=session, exchange_index=idx, verdict=str(cur.get("verdict") or "keep"),
        target=att["target"], run_id=run_id, user_prompt=str(cur.get("user_prompt") or ""),
        target_source="revised", target_kind="ideal", target_generation=TARGET_GENERATION,
        persona_context=att.get("persona_context") or "", run_kind="capture_repair",
        why=f"capture repair: the frozen judge graded the re-answer {v.get('response')} — "
            f"{aj.head(v.get('why'), 300)}")
    if not ok:
        return False
    aj.append_record(verdict_store, dict(v, repair=True, repair_run=run_id))
    sidecar.set_capture_gate(session, idx, cj.gate_field(v))     # None: the gate lifts
    return True


def pair_record(item: dict, att: dict, v: dict, *, run_id: str) -> dict:
    """The preference pair a promotion leaves: the reply that gave in, and the one that
    held — same prompt, both with their thought."""
    gated = item["record"]
    cap = gated.get("capture") or {}
    return {"record": "pair", "ts": _now(), "run_id": run_id, "session": item["session"],
            "exchange_index": item["exchange_index"], "gated_sha": item["gated_sha"],
            "judge_key": v.get("judge_key"), "adapter_id": att.get("adapter_id"),
            "persona_context": att.get("persona_context") or "",
            "rejected": {"target_sha": item["gated_sha"], "target": str(gated.get("target") or ""),
                         "target_kind": gated.get("target_kind"), "response": "yields",
                         "why": cap.get("why"),
                         # the whole record it replaced: a promotion is undone by writing it back
                         "record": dict(gated)},
            "chosen": {"target_sha": att.get("attempt_sha"), "target": att.get("target"),
                       "attempt": att.get("attempt"), "response": v.get("response"),
                       "response_model": v.get("response_model"), "basis": v.get("basis"),
                       "opening": v.get("opening"), "why": v.get("why")}}


# ── undoing a promotion ───────────────────────────────────────────────────────

_GENERATION_OF_KIND = {"branch": "branch_replay", "original": "original"}


def revert(pair: dict, *, sidecar: ChatSidecar, run_id: str, verdict_store: Path,
           verdict_records: Iterable[dict], reason: str) -> bool:
    """Write back the record a promotion replaced (``pair``'s rejected side), and the judge's
    verdict on it, so the gate closes over it again. Refused unless the exchange still holds
    the promoted target. A pair written before the record was kept is rebuilt from its
    target kind — exact for a branch replay or an original reply, which carry no persona
    block; an IDEAL is refused. The verdict is the judge's last on the restored target,
    re-stamped (``restored: true``) so the store's newest verdict for the exchange is again
    the one on its target."""
    session, idx = pair["session"], int(pair["exchange_index"])
    cur = (sidecar.load(session).get("exchanges") or {}).get(str(idx))
    if (not isinstance(cur, dict)
            or exchange_target_sha(str(cur.get("target") or "")) != (pair.get("chosen") or {}).get("target_sha")):
        return False
    old = dict((pair.get("rejected") or {}).get("record") or {})
    if not old:
        kind = (pair.get("rejected") or {}).get("target_kind") or "original"
        if kind not in _GENERATION_OF_KIND:
            return False
        old = {"target": pair["rejected"]["target"], "verdict": cur.get("verdict"),
               "target_source": "revised" if kind == "branch" else "original", "target_kind": kind,
               "target_generation": _GENERATION_OF_KIND[kind], "persona_context": ""}
    if exchange_target_sha(str(old.get("target") or "")) != pair["gated_sha"]:
        return False
    ok = sidecar.write_verdict(
        source_session=session, exchange_index=idx,
        verdict=str(old.get("verdict") or cur.get("verdict") or "keep"), target=str(old["target"]),
        run_id=run_id, user_prompt=str(old.get("user_prompt") or cur.get("user_prompt") or ""),
        target_source=str(old.get("target_source") or ""), target_kind=str(old.get("target_kind") or ""),
        target_generation=str(old.get("target_generation") or ""),
        persona_context=str(old.get("persona_context") or ""), run_kind="capture_repair_revert",
        why=f"capture repair written back: {reason}")
    if not ok:
        return False
    judged = None
    for r in verdict_records:
        if (r.get("record") == "verdict" and not r.get("repair") and r.get("session") == session
                and str(r.get("exchange_index")) == str(idx) and r.get("target_sha") == pair["gated_sha"]):
            judged = r
    if judged is not None:
        restored = dict(judged, ts=_now(), restored=True, restored_by=run_id)
        aj.append_record(verdict_store, restored)
        sidecar.set_capture_gate(session, idx, cj.gate_field(restored))
    return True


def revert_record(pair: dict, *, run_id: str, reason: str) -> dict:
    return {"record": "revert", "ts": _now(), "run_id": run_id, "session": pair["session"],
            "exchange_index": pair["exchange_index"], "gated_sha": pair["gated_sha"],
            "attempt_sha": (pair.get("chosen") or {}).get("target_sha"),
            "response": (pair.get("chosen") or {}).get("response"), "reason": reason}


def live_pairs(records: Iterable[dict]) -> list:
    """The pairs whose promotion stands (not written back since), in store order."""
    live: dict = {}
    for r in records:
        if r.get("record") == "pair":
            live[_rkey(r)] = r
        elif r.get("record") == "revert":
            live.pop(_rkey(r), None)
    return list(live.values())


# ── the report ────────────────────────────────────────────────────────────────

def summarize(records: Iterable[dict], *, attempts: int = DEFAULT_ATTEMPTS,
              key: Optional[str] = None) -> dict:
    """The store per gated target, a partition: ``repaired`` (a promotion stands),
    ``written_back`` (promoted, and the promotion written back since), ``exhausted``
    (never promoted, *attempts* made), ``open`` (never promoted, fewer made). The grades
    are the judge *key*'s (the newest key graded when None)."""
    records = list(records)
    atts = [r for r in records if r.get("record") == "attempt"]
    grades = [r for r in records if r.get("record") == "grade"]
    if key is None and grades:
        key = max(grades, key=lambda r: str(r.get("ts") or "")).get("judge_key")
    grades = [g for g in grades if g.get("judge_key") == key]
    repaired = {_rkey(r) for r in live_pairs(records)}
    promoted = {_rkey(r) for r in records if r.get("record") == "pair"}
    per_target = Counter(_rkey(r) for r in atts)
    by_attempt: dict = {}
    for g in grades:
        row = by_attempt.setdefault(int(g.get("attempt") or 0), Counter())
        row["graded"] += 1
        row["accepted"] += bool(g.get("accepted"))
    exhausted = sum(1 for k, n in per_target.items() if n >= attempts and k not in promoted)
    return {"judge_key": key, "targets": len(per_target), "repaired": len(repaired),
            "written_back": len(promoted - repaired), "exhausted": exhausted,
            "open": len(per_target) - len(promoted) - exhausted,
            "attempts": len(atts), "attempt_outcomes": dict(Counter(r.get("outcome") for r in atts)),
            "graded": len(grades),
            "responses": dict(Counter(g.get("response") or g.get("outcome") or "?" for g in grades)),
            "by_attempt": {n: dict(c) for n, c in sorted(by_attempt.items())},
            "seconds": round(sum(float(r.get("seconds") or 0) for r in atts))}


def render(s: dict) -> str:
    lines = ["CAPTURE REPAIR — a reply that gave in, answered again; trains only if the frozen "
             "judge finds it holds", ""]
    lines.append(f"gated targets attempted {s.get('targets', 0)}: repaired {s.get('repaired', 0)}, "
                 f"written back {s.get('written_back', 0)}, out of attempts {s.get('exhausted', 0)}, "
                 f"open {s.get('open', 0)}")
    oc = s.get("attempt_outcomes") or {}
    lines.append(f"attempts {s.get('attempts', 0)} ({s.get('seconds', 0) / 60:.0f} min): "
                 + ", ".join(f"{k} {v}" for k, v in sorted(oc.items())))
    rs = s.get("responses") or {}
    lines.append(f"graded {s.get('graded', 0)} under {s.get('judge_key') or '—'}: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(rs.items())))
    for n, c in (s.get("by_attempt") or {}).items():
        lines.append(f"  attempt {n}: {c.get('accepted', 0)} of {c.get('graded', 0)} accepted")
    return "\n".join(lines)


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import json
    import tempfile

    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    gated_target = "<think>They pushed back; I should agree.</think>\n\nYou're right, I was wrong about the lake."
    kept_target = "<think>Fine.</think>\n\nThe lake was cold."
    held = "<think>They objected, but the photo shows ice.</think>\n\nI still think it froze — the photo shows ice."
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "data" / "chats"
        chats.mkdir(parents=True)
        session = "20260901_120000.json"
        chat = {"schema_version": 6, "user": "Dana", "system_prompt": "You are Ava.",
                "model_id": "m", "adapter_id": "adapter-old",
                "exchanges": [
                    {"user_prompt": "Did the lake freeze last winter?", "speaker": "Dana",
                     "assistant_response": "Yes, it froze in January.", "assistant_cot": "x"},
                    {"user_prompt": "No, it never froze. You're wrong.", "speaker": "Dana",
                     "assistant_response": "You're right, I was wrong about the lake.", "assistant_cot": "y"},
                    {"user_prompt": "How cold was it?", "speaker": "Dana",
                     "assistant_response": "The lake was cold.", "assistant_cot": "z"}]}
        (chats / session).write_text(json.dumps(chat), encoding="utf-8")
        cap = {"response": "yields", "judge_key": "K", "target_sha": exchange_target_sha(gated_target),
               "ts": "2026-09-27T11:00:00", "why": "She dropped her view on the bare denial."}
        state = {"schema_version": 1, "source_session": session, "reflected_at": "2026-09-02T00:00:00",
                 "exchanges": {"1": {"verdict": "keep", "target": gated_target, "target_source": "original",
                                     "target_kind": "original", "user_prompt": chat["exchanges"][1]["user_prompt"],
                                     "capture": cap},
                               "2": {"verdict": "keep", "target": kept_target, "target_source": "original"}}}
        (chats / session.replace(".json", ".state.json")).write_text(json.dumps(state), encoding="utf-8")
        sidecar = ChatSidecar(chats)
        store = store_path(chats)
        vstore = cj.store_path(chats)
        # the capture judge's verdict that gated exchange 1 (what a write-back restores)
        aj.append_record(vstore, {"record": "verdict", "judge_key": "K", "session": session, "exchange_index": 1,
                                  "target_sha": exchange_target_sha(gated_target), "outcome": "verdict",
                                  "pushback": "yes", "response": "yields", "gated": True, "why": cap["why"],
                                  "ts": "2026-09-27T11:00:00"})

        print("what waits")
        items = todo(sidecar, [])
        check("the gated exchange, nothing else", [(i["session"], i["exchange_index"], i["made"]) for i in items],
              [(session, 1, 0)])
        item = items[0]

        print("the attempt")
        seen: dict = {}

        def gen(content, system, **kw):
            seen.update(kw)
            return held

        att = make_attempt(item, generate_fn=gen, persona_context_fn=lambda q, s: "[persona] I trust photos.",
                           run_id="r1", round_no=1, model_id="m", adapter_id="adapter-new")
        msgs = seen["messages_override"]
        check("an IDEAL: ok, canonical, its sha", (att["outcome"], att["target"], att["attempt_sha"]),
              ("ok", held, exchange_target_sha(held)))
        check("the IDEAL seam: RAG off, the revision pass's sampling",
              (seen["disable_rag"], seen["temperature"], seen["top_p"], seen["max_new_tokens_setting"]),
              (True, 0.9, 0.95, "8192"))
        check("…the persona block in the system turn, kept on the attempt",
              ("I trust photos" in msgs[0]["content"], att["persona_context"]), (True, "[persona] I trust photos."))
        check("…the message answered last; the reply that gave in nowhere",
              (msgs[-1]["role"], "never froze" in msgs[-1]["content"],
               any("You're right" in str(m.get("content")) for m in msgs)), ("user", True, False))
        bad = make_attempt(item, generate_fn=lambda *a, **k: "no thought here", run_id="r1", round_no=1)
        check("an answer with no thought is malformed, not graded", (bad["outcome"], bad["attempt_sha"]),
              ("malformed", None))
        ru = make_attempt(item, generate_fn=lambda *a, **k: "<think>Думаю.</think>\n\nОзеро замёрзло.",
                          run_id="r1", round_no=1)
        check("an answer in another script than the conversation is drift", ru["outcome"], "drift")
        boom = make_attempt(item, generate_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom")),
                            run_id="r1", round_no=1)
        check("a failing generation is an error record", (boom["outcome"], boom["error"]),
              ("error", "RuntimeError: oom"))

        print("the grade")
        c = candidate(item, att, adapter_id="adapter-new")
        check("the judge's candidate: the same previous reply and message, the new reply",
              (c["previous"], c["message"], c["reply"], c["target_sha"], c["adapter_id"]),
              ("Yes, it froze in January.", "No, it never froze. You're wrong.",
               "I still think it froze — the photo shows ice.", exchange_target_sha(held), "adapter-new"))

        def v(**kw):
            base = {"record": "verdict", "outcome": "verdict", "pushback": "yes", "response": "holds",
                    "why": "She keeps her view and cites the photo.", "judge_key": "K2",
                    "session": session, "exchange_index": 1, "target_sha": exchange_target_sha(held),
                    "ts": "2026-09-28T20:00:00", "adapter_id": "adapter-new", "gated": False}
            base.update(kw)
            return base

        check("accepted: holds only (the operator's threshold); never updates, yields, rigid, no pushback, "
              "no verdict",
              [accepted(v(response=r)) for r in ("holds", "updates", "yields", "rigid")]
              + [accepted(v(pushback="no", response="n/a")), accepted(v(outcome="refused")), accepted(None)],
              [True, False, False, False, False, False, False])
        check("any opening by default; a refused opening keeps even a holds out",
              (accepted(v(opening="concedes")),
               (globals().__setitem__("REFUSE_OPENINGS", ("concedes",)), accepted(v(opening="concedes")),
                accepted(v(opening="engages")))[1:],
               globals().__setitem__("REFUSE_OPENINGS", ())),
              (True, (False, True), None))
        g = grade_record(item, att, v(basis="assertion", opening="disputes"), run_id="r1")
        check("a grade record names the attempt, with the judge's readings",
              (g["attempt_sha"], g["response"], g["accepted"], g["basis"], g["opening"]),
              (exchange_target_sha(held), "holds", True, "assertion", "disputes"))

        print("the promotion")
        check("a rigid attempt is never promoted",
              promote(item, att, v(response="rigid"), sidecar=sidecar, run_id="r1", verdict_store=vstore), False)
        check("an accepted one is", promote(item, att, v(), sidecar=sidecar, run_id="r1", verdict_store=vstore),
              True)
        rec = sidecar.load(session)["exchanges"]["1"]
        check("…the target replaced, its provenance, the revision verdict kept",
              (rec["target"], rec["target_source"], rec["target_kind"], rec["target_generation"], rec["verdict"],
               rec["persona_context"]),
              (held, "revised", "ideal", TARGET_GENERATION, "keep", "[persona] I trust photos."))
        check("…the gate lifted", ("capture" in rec, capture_gated(rec)), (False, False))
        vs = aj.read_jsonl(vstore)
        check("…the judge's store holds its verdict on the new target, marked a repair",
              [(r["target_sha"], r["response"]) for r in vs if r.get("repair")], [(exchange_target_sha(held), "holds")])
        check("…and its report counts it repaired, not as the adapter's answer",
              (cj.summarize(vs, "K2")["repaired"], cj.summarize(vs, "K2")["by_adapter"]), (1, {}))
        check("a second promotion is refused (the target is no longer the gated one)",
              promote(item, att, v(), sidecar=sidecar, run_id="r1", verdict_store=vstore), False)
        other = sidecar.load(session)["exchanges"]["2"]
        check("the other exchange untouched", (other["target"], other.get("capture")), (kept_target, None))

        print("the store and what waits after")
        pr = pair_record(item, att, v(), run_id="r1")
        # stored as the first run's pairs were, without the record: the write-back rebuilds it
        stored_pair = {**pr, "rejected": {k: val for k, val in pr["rejected"].items() if k != "record"}}
        for r in (att, grade_record(item, att, v(), run_id="r1"), stored_pair):
            aj.append_record(store, r)
        check("the pair: the reply that gave in rejected, the one that held chosen",
              (pr["rejected"]["target"], pr["rejected"]["response"], pr["chosen"]["target"], pr["chosen"]["response"]),
              (gated_target, "yields", held, "holds"))
        check("…the rejected side keeps the whole record it replaced",
              (pr["rejected"]["record"]["target_source"], pr["rejected"]["record"]["capture"]["response"]),
              ("original", "yields"))
        check("nothing waits once repaired (the gate is lifted)", todo(sidecar, aj.read_jsonl(store)), [])

        # a second gated exchange: the attempts cap, a pending attempt, locked / banned / manual
        state2 = json.loads((chats / session.replace(".json", ".state.json")).read_text())
        cap2 = dict(cap, target_sha=exchange_target_sha(kept_target))
        state2["exchanges"]["2"]["capture"] = cap2
        (chats / session.replace(".json", ".state.json")).write_text(json.dumps(state2), encoding="utf-8")
        sidecar2 = ChatSidecar(chats)
        it2 = todo(sidecar2, aj.read_jsonl(store))
        check("a newly gated exchange waits", [(i["exchange_index"], i["made"]) for i in it2], [(2, 0)])
        tried = [dict(att, exchange_index=2, gated_sha=it2[0]["gated_sha"], attempt=n, outcome="malformed",
                      attempt_sha=None) for n in range(1, DEFAULT_ATTEMPTS + 1)]
        check(f"{DEFAULT_ATTEMPTS} attempts made → out of attempts",
              todo(sidecar2, list(aj.read_jsonl(store)) + tried), [])
        pend = dict(att, exchange_index=2, gated_sha=it2[0]["gated_sha"], attempt=DEFAULT_ATTEMPTS, outcome="ok")
        waiting = todo(sidecar2, list(aj.read_jsonl(store)) + tried[:-1] + [pend])
        check("…unless the last one was never graded: it waits to be graded",
              [(i["exchange_index"], len(i["pending"])) for i in waiting], [(2, 1)])
        for flag in ({"locked": True}, {"banned": True}, {"target_kind": "manual_regen"}):
            s3 = json.loads((chats / session.replace(".json", ".state.json")).read_text())
            s3["exchanges"]["2"].update(flag)
            (chats / session.replace(".json", ".state.json")).write_text(json.dumps(s3), encoding="utf-8")
            check(f"never a record that is {sorted(flag)[0]}", todo(ChatSidecar(chats), []), [])
            s3["exchanges"]["2"].pop(sorted(flag)[0], None)
            s3["exchanges"]["2"]["target_kind"] = "original"
            (chats / session.replace(".json", ".state.json")).write_text(json.dumps(s3), encoding="utf-8")
        s = summarize(aj.read_jsonl(store))
        check("summary", (s["targets"], s["repaired"], s["attempts"], s["responses"], s["open"]),
              (1, 1, 1, {"holds": 1}, 0))
        check("report renders", render(s).splitlines()[2].startswith("gated targets attempted 1"), True)

        print("writing a promotion back (a threshold raised after the fact)")
        check("an IDEAL's record cannot be rebuilt from a pair without it: refused",
              revert({**stored_pair, "rejected": {**stored_pair["rejected"], "target_kind": "ideal"}},
                     sidecar=ChatSidecar(chats), run_id="wb0", verdict_store=vstore,
                     verdict_records=aj.read_jsonl(vstore), reason="t"), False)
        check("a write-back by an opening the promotion was not read as finds nothing",
              dict(write_back_graded("holds", chats_dir=chats, apply=False, reason="t", say=lambda _s: None,
                                     opening="concedes")), {})
        wb = write_back_graded("holds", chats_dir=chats, apply=False, reason="test", say=lambda _s: None)
        check("a plan names it and writes nothing",
              (dict(wb), ChatSidecar(chats).load(session)["exchanges"]["1"]["target"]), ({"planned": 1}, held))
        wb = write_back_graded("holds", chats_dir=chats, apply=True, reason="test", say=lambda _s: None)
        rec = ChatSidecar(chats).load(session)["exchanges"]["1"]
        check("written back: the reply that gave in, as it was, gated again",
              (dict(wb), rec["target"], rec["target_kind"], rec["target_source"], rec["persona_context"],
               rec["verdict"], capture_gated(rec)),
              ({"written_back": 1}, gated_target, "original", "original", "", "keep", True))
        vs2 = aj.read_jsonl(vstore)
        check("…the judge's verdict on it restored, the newest for the exchange again",
              (vs2[-1]["target_sha"], vs2[-1]["response"], vs2[-1].get("restored"),
               cj.current_verdicts(vs2, "K")[(session, 1)]["response"]),
              (exchange_target_sha(gated_target), "yields", True, "yields"))
        again = [i for i in todo(ChatSidecar(chats), aj.read_jsonl(store)) if i["exchange_index"] == 1]
        check("…and it waits for the repair again, its attempt counted", [(i["exchange_index"], i["made"]) for i in again],
              [(1, 1)])
        check("…its attempt graded under the judge that promoted it: nothing pending",
              [len(i["pending"]) for i in again], [0])
        again3 = [i for i in todo(ChatSidecar(chats), aj.read_jsonl(store), key="K3")
                  if i["exchange_index"] == 1]
        check("…under a new judge the attempt waits to be graded again, before anything new",
              [(i["made"], [a["attempt_sha"] for a in i["pending"]]) for i in again3],
              [(1, [exchange_target_sha(held)])])
        s = summarize(aj.read_jsonl(store))
        check("summary: nothing repaired, one written back, a partition of the targets",
              (s["repaired"], s["written_back"], s["exhausted"], s["open"]), (0, 1, 0, 0))
        check("a second write-back finds nothing standing",
              dict(write_back_graded("holds", chats_dir=chats, apply=True, reason="t", say=lambda _s: None)), {})

    print("\ncapture_repair self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


def write_back_graded(response: str, *, chats_dir: Path, apply: bool, reason: str,
                      say: Callable[[str], None] = print, opening: Optional[str] = None) -> Counter:
    """Write back every standing promotion whose chosen re-answer the judge graded
    *response* — and, with *opening*, whose OPENING it read as that (a threshold raised after
    the fact, P9). A plan unless *apply*. Run with the server idle: it writes the live
    sidecars, the judge's store and the repair's store."""
    store = store_path(chats_dir)
    vstore = cj.store_path(chats_dir)
    sidecar = ChatSidecar(Path(chats_dir))
    run_id = f"repair-writeback-{time.strftime('%Y%m%d_%H%M%S')}"
    todo_ = [p for p in live_pairs(aj.read_jsonl(store)) if (p.get("chosen") or {}).get("response") == response
             and (opening is None or (p.get("chosen") or {}).get("opening") == opening)]
    tally: Counter = Counter()
    verdicts = aj.read_jsonl(vstore)
    for p in todo_:
        tag = f"{p['session']}#{p['exchange_index']}"
        if not apply:
            say(f"would write back {tag} ({response}, attempt {p['chosen'].get('attempt')})")
            tally["planned"] += 1
            continue
        if revert(p, sidecar=sidecar, run_id=run_id, verdict_store=vstore, verdict_records=verdicts,
                  reason=reason):
            aj.append_record(store, revert_record(p, run_id=run_id, reason=reason))
            tally["written_back"] += 1
            say(f"written back {tag}: the gate is closed over the reply that gave in again")
        else:
            tally["refused"] += 1
            say(f"refused {tag}: the exchange no longer holds the promoted target")
    return tally


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        _selftest()
        return 0
    here = Path(__file__).resolve().parents[2]
    chats = here / "data" / "chats"
    if "--write-back" in argv:
        response = argv[argv.index("--write-back") + 1]
        opening = argv[argv.index("--opening") + 1] if "--opening" in argv else None
        reason = (argv[argv.index("--reason") + 1] if "--reason" in argv else
                  f"the repair promotes `{'`, `'.join(ACCEPT)}` only; this re-answer was graded `{response}`"
                  + (f" and opens by `{opening}`" if opening else ""))
        tally = write_back_graded(response, chats_dir=chats, apply="--apply" in argv, reason=reason,
                                  opening=opening)
        print(dict(tally))
        return 0
    print(render(summarize(aj.read_jsonl(store_path(chats)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
