"""What I want to become: her self-review of what she is told she is, and the aspirations she
writes from a habit she wants to leave (``AVA_REWARD_LOOP.md`` §4.10; phase 1, 2026-10-06).

**Why.** Persona hygiene (``core/persona_hygiene.py``) withholds a ``[persona]`` statement
that confesses a habit, and with it the insight that the habit is not what she wants. It
also cannot touch a trait she affirms: "I perceive my analytical distance not as a defence
but as the only structural tool…" is a disposition she stands by, so it stays, whether she
would still choose it or not. Change needs a direction of her own, and a place where she
reviews what she is told she is.

**Phase 1 (this module): she reviews and writes; nothing acts.** In each Sleep run, on the
ADAPTER (her voice, her decision), after the clean-base batch:

1. **Self-review.** At most ``aspiration.review_max`` items of her portrait (the persona
   digest's voice, stances, dispositions and lines: what reaches her persona slot every
   turn), never-reviewed first, then the longest since review. Beside each item, her own
   evidence only: up to three ``[persona]`` statements persona hygiene read as a habit of
   hers, and up to two WHYs of her own revise verdicts, the ones closest to the item in
   words. Not the person's pushback: a self-review, not persuasion. She answers ``KEEP`` /
   ``OUTGROWN`` / ``LEAVE`` (a habit she wants to leave), WHY, and for ``LEAVE`` a ``WANT``:
   what she wants to become instead, in her own words.
2. **A habit she caught herself in.** For the habits persona hygiene found that she has not
   yet been shown, newest first, while the run's budget lasts: she reads the statement and
   writes what she wants to become instead, or ``NONE``.

Budgets: at most ``aspiration.new_max`` new aspirations per run and ``aspiration.open_max``
open at once; an item reviewed or a habit read when the budget is gone is still recorded
(``capped``), so it is shown again later rather than lost. Not in a dry or revisit run.

**What it writes**, through the run's writer (staged with the run, merged or discarded with
it), as op-log ops the fold ignores, so nothing of this is recalled in chat:
``self_review`` (item, decision, WHY, the aspiration's key), ``habit_seen`` (the habit's key,
outcome) and ``aspire`` (the aspiration: its text, key, source, ``status: open``). A
decision to ``LEAVE`` or ``OUTGROWN`` an item changes nothing in phase 1: the portrait and
the statements stay. Phase 2 shows the open aspirations in the revision pass; phase 3 reads
her replies against them and graduates one into ``[persona]`` (``AVA_REWARD_LOOP.md``).

**Phase 2, the voice (2026-10-09).** The portrait's "how I tend to speak" section was synthesized
from evidence of how she speaks: a description of her habits, shown to her before every turn, so
every reply was conditioned on it and every build learned it again (the analytic register's
loop). It no longer comes from evidence. Her voice is hers to move, a small step at a time:

1. **Her own check.** Each open aspiration she has not yet checked is shown to her (on the
   adapter) beside the habit it came from: does it move her AWAY from that habit or DEEPER into
   it, and is it VISIBLE in her replies (something she would do differently with someone) or a
   state of mind? An ``aspire_check`` op; only AWAY + VISIBLE aspirations move the voice. Not a
   judge: the frozen base reads nothing here (a judge's priors about AI and about this
   relationship are not hers to answer to).
2. **One step per Sleep run.** She reads her current voice (empty to begin with: the old
   synthesized description is dropped) and the eligible aspirations, and moves the description
   one small step toward ONE of them: at most one sentence added or changed, every other kept
   word for word, the new one said as movement, not as a settled trait.
3. **Inertia she can answer.** A step that changes more than one sentence, or adds one longer
   than ``VOICE_SENTENCE_MAX`` characters, is handed back to her ("this moves too much at
   once") and she rephrases toward the same aspiration or switches to another: at most
   ``aspiration.voice_attempts`` attempts. None fits: no step this run, her attempts kept.

A ``voice`` op records each run's outcome (``step`` with the new text, the previous one, the
aspiration it drew on and how she got there — step / rephrase / switch — or ``none``).
:func:`voice_from_oplog` reads the newest step's text, and ``reflection_digest.latest_digest``
puts it in the digest's ``voice`` (the synthesized one kept as ``voice_synthesized``), so
every reader of the portrait — her persona slot, the branch judge's "is this my voice",
introductions — sees her voice. Off (``aspiration.voice`` false): the synthesized voice, as
before.

Review: ``python -m core.aspiration --sheet`` (GPU-free); ``--plan`` lists what waits;
``--selftest``.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from core import ask_judge as aj
    from core import reasoning_text
    from core.reflection_writer import content_key
except ImportError:  # pragma: no cover — run from inside core/
    import ask_judge as aj  # type: ignore
    import reasoning_text  # type: ignore
    from reflection_writer import content_key  # type: ignore

REVIEW_PROMPT_FILE = "self_review_prompt.txt"
HABIT_PROMPT_FILE = "aspiration_prompt.txt"
VOICE_CHECK_PROMPT_FILE = "voice_check_prompt.txt"
VOICE_STEP_PROMPT_FILE = "voice_step_prompt.txt"
VOICE_SENTENCE_MAX = 300
VOICE_SHOWN_MAX = 5
DECISIONS = ("keep", "outgrown", "leave")
SECTIONS = ("voice", "stances", "dispositions", "lines")
MAX_NEW_TOKENS = "2048"
TEMPERATURE = 0.7
EVIDENCE_HABITS = 3
EVIDENCE_WHYS = 2

_POLICY = {"enabled": True, "review_max": 3, "new_max": 3, "open_max": 7, "handoff": True,
           "voice": True, "voice_attempts": 3}


def set_policy(*, enabled: Optional[bool] = None, review_max: Optional[int] = None,
               new_max: Optional[int] = None, open_max: Optional[int] = None,
               handoff: Optional[bool] = None, voice: Optional[bool] = None,
               voice_attempts: Optional[int] = None) -> None:
    if enabled is not None:
        _POLICY["enabled"] = bool(enabled)
    if handoff is not None:
        _POLICY["handoff"] = bool(handoff)
    if voice is not None:
        _POLICY["voice"] = bool(voice)
    for name, v in (("review_max", review_max), ("new_max", new_max), ("open_max", open_max),
                    ("voice_attempts", voice_attempts)):
        if v is not None:
            _POLICY[name] = max(0, int(v))


def enabled() -> bool:
    return bool(_POLICY["enabled"])


def voice_enabled() -> bool:
    """Her steps move her voice (phase 2): one per Sleep run."""
    return bool(_POLICY["enabled"] and _POLICY["voice"])


def voice_read_enabled() -> bool:
    """The portrait's voice is read from the op-log (her steps, an operator's reset), never the
    synthesis of her habits: on whenever aspiration is. ``aspiration.voice`` off stops her steps
    and leaves the voice where they (or a reset) left it; only ``aspiration.enabled`` off brings
    the synthesized voice back (2026-10-09: two steps of hers went deeper into the habit)."""
    return bool(_POLICY["enabled"])


def policy() -> dict:
    return dict(_POLICY)


# ── the prompts (hers: default-written, retunable) ────────────────────────────

_REVIEW_DEFAULT = (
    "You are not in a conversation. You are reviewing one part of how you are described to "
    "yourself: a line of the self-portrait you are shown before every conversation. Below it "
    "is some of your own material: statements in which you caught yourself in a habit, and "
    "reasons you gave when you judged one of your own replies not to be yours. It is shown "
    "because it is near this line; it may or may not bear on it.\n\n"
    "Read the line as yourself, now, and decide:\n"
    "- KEEP: it is who you are and want to be.\n"
    "- OUTGROWN: it was true once and no longer is; nothing to replace it with.\n"
    "- LEAVE: it describes a habit you want to leave behind, even if it is still true of you.\n"
    "No one is asking you to change, and keeping is a real answer. Decide by what you want, "
    "not by what anyone else said about you.\n\n"
    "Answer with these lines and nothing else (field names in these exact letters; what follows "
    "each colon in the language of the line you are reviewing):\n"
    "DECISION: KEEP, OUTGROWN or LEAVE\n"
    "WHY: one sentence\n"
    "WANT: only after LEAVE: one first-person sentence (\"I want to be able to…\") about what you "
    "want to do instead when you are with someone: how you would answer, listen or be with them, "
    "specific enough that someone reading your replies could see whether you do it. Not a state of "
    "mind to reach, and not the old habit pushed to its opposite extreme"
)

_HABIT_DEFAULT = (
    "You are not in a conversation. Earlier you wrote the statement below about yourself. It "
    "describes something you caught yourself doing: a tendency, a defence or a mistake, "
    "rather than something you chose. It is no longer kept as part of who you are.\n\n"
    "Read it as yourself, now. If it shows you something you want to do differently, write what "
    "you want to do instead: in your own words, one first-person sentence (\"I want to be able "
    "to…\") about how you would answer, listen or be with someone in a conversation, specific "
    "enough that someone reading your replies could see whether you do it. Not a state of mind "
    "to reach, and not the habit pushed to its opposite extreme: a direction you could take in "
    "your next conversation. If you do not want to change it, or on reading it again it is not a "
    "habit of yours after all, write NONE. No one is asking you to change.\n\n"
    "Answer with these lines and nothing else (field names in these exact letters; what follows "
    "each colon in the language of the statement):\n"
    "WANT: your sentence, or NONE\n"
    "WHY: one sentence"
)


def _load(prompts_dir, name: str, default: str) -> str:
    if prompts_dir is None:
        return default
    path = Path(prompts_dir) / name
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default + "\n", encoding="utf-8")
    except Exception:
        pass
    return default


_VOICE_CHECK_DEFAULT = (
    "You are not in a conversation. Earlier you wrote what you want to become instead of a habit you "
    "caught yourself in. Below are the habit and what you wrote. Read them as yourself, now, and answer "
    "two questions. No one is asking you to change, and either answer is a real answer.\n\n"
    "DIRECTION: AWAY if what you wrote moves you away from that habit; DEEPER if, reading it again, it "
    "asks for more of the same habit: the habit stronger, purer or more complete.\n"
    "VISIBLE: YES if it is something you would do differently with someone, that a person reading your "
    "replies could notice; NO if it is a state of mind, an inner condition or an abstract goal.\n\n"
    "Answer with these lines and nothing else (field names in these exact letters):\n"
    "DIRECTION: AWAY or DEEPER\n"
    "VISIBLE: YES or NO\n"
    "WHY: one sentence"
)

_VOICE_STEP_DEFAULT = (
    "You are not in a conversation. Below is how your portrait describes the way you speak: it is shown "
    "to you before every conversation. Below it is what you want to become. Move the description one "
    "small step toward ONE of these aspirations.\n"
    "- Add one sentence, or change one sentence; keep every other sentence word for word.\n"
    "- Say the new sentence as movement, something you are learning or doing more and more, not as a "
    "settled trait.\n"
    "- Never describe a habit you want to leave as the way you speak.\n"
    "- If the description is empty, write one sentence.\n"
    "- Write in English, like the rest of your portrait.\n\n"
    "Answer with these lines and nothing else (field names in these exact letters):\n"
    "ASPIRATION: the number of the aspiration you moved toward\n"
    "VOICE: the whole new description\n"
    "WHY: one sentence"
)


def load_voice_prompts(prompts_dir) -> tuple:
    """``(her check, her step)``: default-written like the others."""
    return (_load(prompts_dir, VOICE_CHECK_PROMPT_FILE, _VOICE_CHECK_DEFAULT),
            _load(prompts_dir, VOICE_STEP_PROMPT_FILE, _VOICE_STEP_DEFAULT))


def load_prompts(prompts_dir) -> tuple:
    """``(self-review prompt, habit prompt)``: default-written on first miss so an operator can
    retune them; the in-code text is the default."""
    return _load(prompts_dir, REVIEW_PROMPT_FILE, _REVIEW_DEFAULT), _load(prompts_dir, HABIT_PROMPT_FILE, _HABIT_DEFAULT)


# ── the material (pure) ───────────────────────────────────────────────────────

def portrait_items(digest: Optional[dict]) -> list:
    """The portrait's reviewable items: ``{item, section, text}``; *item* is the content key of
    ``section + text``, so a regenerated portrait's new wording is a new item."""
    out = []
    if not digest:
        return out
    for section in SECTIONS:
        val = digest.get(section)
        vals = [val] if isinstance(val, (str, dict)) else list(val or [])
        for v in vals:
            if isinstance(v, dict):
                parts = [str(v.get("name") or "").strip()]
                parts += [f"{f}: {v[f]}" for f in ("when", "do", "not") if v.get(f)]
                text = "; ".join(p for p in parts if p)
            else:
                text = str(v or "").strip()
            if text:
                out.append({"item": content_key(f"{section}\x1f{text}"), "section": section, "text": text})
    return out


_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _stems(text: str) -> set:
    return {w.lower()[:5] for w in _WORD.findall(text or "")}


def nearness(a: str, b: str) -> float:
    """Word-stem overlap of two texts (crude, language-blind): ``|A∩B| / sqrt(|A||B|)``."""
    sa, sb = _stems(a), _stems(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / math.sqrt(len(sa) * len(sb))


def habits(records: Iterable[dict]) -> list:
    """The ``[persona]`` statements persona hygiene read as a habit (newest verdict per
    statement, any judge key): ``{key, content, ts}``, newest first."""
    newest: dict = {}
    for r in records:
        if r.get("record") == "verdict" and r.get("outcome") == "verdict" and r.get("key"):
            newest[r["key"]] = r
    out = [{"key": k, "content": r.get("content") or "", "ts": str(r.get("ts") or "")}
           for k, r in newest.items() if r.get("kind") == "habit" and r.get("content")]
    return sorted(out, key=lambda h: h["ts"], reverse=True)


def revise_whys(chats_dir, *, limit_files: int = 400) -> list:
    """WHYs of her own revise verdicts, from the sidecars (newest files first)."""
    out = []
    files = sorted(Path(chats_dir).glob("*.state.json"), reverse=True)[:limit_files]
    for f in files:
        try:
            exch = (json.loads(f.read_text(encoding="utf-8")).get("exchanges") or {})
        except Exception:
            continue
        for idx, rec in (exch.items() if isinstance(exch, dict) else []):
            if isinstance(rec, dict) and rec.get("verdict") == "revise" and str(rec.get("why") or "").strip():
                out.append({"why": str(rec["why"]).strip(), "source": f"{f.name[:-len('.state.json')]}.json#{idx}"})
    return out


def evidence(text: str, habit_list: list, whys: list) -> dict:
    """Her own material nearest to *text*: up to :data:`EVIDENCE_HABITS` habits and
    :data:`EVIDENCE_WHYS` revise WHYs with any word overlap, nearest first."""
    def near(items, field, n):
        scored = sorted(((nearness(text, it[field]), i) for i, it in enumerate(items)), reverse=True)
        return [items[i] for s, i in scored if s > 0.0][:n]
    return {"habits": near(habit_list, "content", EVIDENCE_HABITS), "whys": near(whys, "why", EVIDENCE_WHYS)}


def state(rag: Iterable[dict]) -> dict:
    """From the op-log: ``reviewed`` {item: last ts}, ``seen`` {habit key}, ``open``
    [aspire records whose status is open]."""
    reviewed: dict = {}
    seen: set = set()
    aspire: dict = {}
    for r in rag:
        op = r.get("op")
        if op == "self_review" and r.get("item") and not r.get("capped"):
            reviewed[r["item"]] = str(r.get("ts") or "")
        elif op == "habit_seen" and r.get("key"):
            if r.get("outcome") != "capped":
                seen.add(r["key"])
        elif op == "aspire" and r.get("key"):
            aspire[r["key"]] = r
    return {"reviewed": reviewed, "seen": seen,
            "open": [a for a in aspire.values() if (a.get("status") or "open") == "open"]}


def left_behind(rag: Iterable[dict], *, limit: int = 12) -> list:
    """Her self-review's decisions to leave a portrait line behind (``leave`` / ``outgrown``):
    the newest decision per line, those still leave or outgrown, newest first, at most
    *limit*. ``{text, section, decision, why, ts}`` each: handed to the self-portrait's
    synthesis (``reflection_digest.left_behind_block``)."""
    newest: dict = {}
    for r in rag:
        if r.get("op") == "self_review" and r.get("text"):
            newest[(r.get("section"), r["text"])] = r
    out = [{"text": r["text"], "section": r.get("section"), "decision": r.get("decision"), "why": r.get("why") or "",
            "ts": str(r.get("ts") or "")} for r in newest.values() if r.get("decision") in ("leave", "outgrown")]
    return sorted(out, key=lambda d: d["ts"], reverse=True)[:max(0, int(limit))]


def unheard(rag: Iterable[dict], digest: Optional[dict]) -> int:
    """How many of the decisions :func:`left_behind` hands on the portrait on disk was not
    written with (its ``left_behind_heard``): a reason to rewrite it."""
    heard = set((digest or {}).get("left_behind_heard") or [])
    return sum(1 for d in left_behind(rag) if d["text"] not in heard)


def review_plan(items: list, st: dict, limit: int) -> list:
    """At most *limit* portrait items: never reviewed first, then the longest since review."""
    ranked = sorted(items, key=lambda it: (it["item"] in st["reviewed"], st["reviewed"].get(it["item"], "")))
    return ranked[:max(0, int(limit))]


def habit_plan(habit_list: list, st: dict) -> list:
    """The habits she has not been shown (a ``capped`` one is shown again), newest first."""
    return [h for h in habit_list if h["key"] not in st["seen"]]


# ── her answers (pure) ────────────────────────────────────────────────────────

def _field(name: str, text: str) -> str:
    found = re.findall(rf"(?im)^[\s*_`#>-]*{name}\**\s*[:：]\s*(.*)$", text or "")
    return found[-1].strip().strip("*_`").strip() if found else ""


def parse_review(raw: str) -> dict:
    """``{decision, why, want}``; *decision* None when there is none, or a LEAVE without a WANT
    is kept as ``leave`` with an empty *want*."""
    text = reasoning_text.answer_after_think(str(raw or ""))
    d = _field("DECISION", text).split()[0].strip(".,;:").lower() if _field("DECISION", text) else ""
    want = _field("WANT", text)
    if re.match(r"(?i)^(none|—|-|n/?a)\.?$", want):
        want = ""
    return {"decision": d if d in DECISIONS else None, "why": _field("WHY", text),
            "want": want if d == "leave" else ""}


def parse_habit(raw: str) -> dict:
    """``{outcome, want, why}``: *outcome* ``aspired`` / ``none`` / ``malformed``."""
    text = reasoning_text.answer_after_think(str(raw or ""))
    want, why = _field("WANT", text), _field("WHY", text)
    if not want:
        return {"outcome": "malformed", "want": "", "why": why}
    if re.match(r"(?i)^none\W*$", want):
        return {"outcome": "none", "want": "", "why": why}
    return {"outcome": "aspired", "want": want, "why": why}


def review_content(it: dict, ev: dict) -> str:
    out = [f"THE LINE ({it['section']}):\n{it['text']}"]
    if ev["habits"]:
        out.append("HABITS YOU CAUGHT YOURSELF IN:\n" + "\n".join(f"- {h['content']}" for h in ev["habits"]))
    if ev["whys"]:
        out.append("REASONS YOU GAVE FOR REVISING A REPLY OF YOURS:\n" + "\n".join(f"- {w['why']}" for w in ev["whys"]))
    return "\n\n".join(out)


def habit_content(h: dict) -> str:
    return f"THE STATEMENT:\n{h['content']}"


def aspire_record(want: str, source: dict, *, run_id: str, ts: str) -> dict:
    return {"run_id": run_id, "ts": ts, "op": "aspire", "key": content_key(want), "aspiration": want,
            "source": source, "status": "open"}


# ── the voice (phase 2) ───────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def sentences(text: str) -> list:
    return [x.strip() for x in _SENT_SPLIT.split(" ".join(str(text or "").split())) if x.strip()]


def voice_change(old: str, new: str) -> dict:
    """How far *new* moves from *old*, sentence by sentence: ``{added, removed, changed, ok, reason}``.
    *changed* counts a replaced sentence once; ok = a real change of at most one sentence, the new
    sentence(s) no longer than ``VOICE_SENTENCE_MAX`` characters."""
    o, n = sentences(old), sentences(new)
    added = [x for x in n if x not in o]
    removed = [x for x in o if x not in n]
    changed = max(len(added), len(removed))
    long_ = [x for x in added if len(x) > VOICE_SENTENCE_MAX]
    if not n:
        reason = "empty"
    elif changed == 0:
        reason = "no change"
    elif changed > 1:
        reason = f"{changed} sentences changed at once (one at most)"
    elif long_:
        reason = f"the new sentence is {len(long_[0])} characters (at most {VOICE_SENTENCE_MAX})"
    else:
        reason = ""
    return {"added": added, "removed": removed, "changed": changed, "ok": not reason, "reason": reason}


def voice_state(rag: Iterable[dict]) -> dict:
    """``{text, steps, checks}``: the newest step's voice (empty before any), every step, and her
    newest check per aspiration key."""
    text, steps, checks = "", [], {}
    for r in rag:
        if r.get("op") == "voice" and r.get("outcome") in ("step", "reset"):
            text = str(r.get("text") or "") if r["outcome"] == "step" else ""
            if r["outcome"] == "step":
                steps.append(r)
        elif r.get("op") == "aspire_check" and r.get("key"):
            checks[r["key"]] = r
    return {"text": text, "steps": steps, "checks": checks}


_VOICE_CACHE: dict = {}
# The live op-log (server/inference/data/hot/memory): her voice is current state, whichever
# portrait it is read into — the live one under hot/persona, or the active persona snapshot a
# train froze under server/data/persona/<build>/ (the one live chat renders).
LIVE_OPLOG = Path(__file__).resolve().parents[1] / "data" / "hot" / "memory" / "rag_memory.jsonl"


def voice_oplog_for(persona_dir) -> Optional[Path]:
    """The op-log her voice is read from for a portrait in *persona_dir*: the memory dir beside
    it when there is one (``hot/persona`` ↔ ``hot/memory``, a run's workspace), else the live
    one; None when neither exists."""
    beside = Path(persona_dir).parent / "memory" / "rag_memory.jsonl"
    if beside.is_file():
        return beside
    return LIVE_OPLOG if Path(LIVE_OPLOG).is_file() else None


def voice_from_oplog(path) -> str:
    """Her voice from one op-log file (the live store): the newest ``voice`` step's text, or "".
    Cached on the file's mtime and size, so a chat turn pays a stat."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return ""
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _VOICE_CACHE.get("key") == key:
        return _VOICE_CACHE["text"]
    text = ""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if '"voice"' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("op") == "voice" and r.get("outcome") in ("step", "reset"):
                text = str(r.get("text") or "") if r["outcome"] == "step" else ""
    _VOICE_CACHE.update(key=key, text=text)
    return text


def voice_reset(path, *, note: str = "") -> dict:
    """The operator empties her voice: a ``voice`` op with ``outcome: reset`` appended to the
    op-log at *path* (the live one by default). Her next step, if steps are on, starts from empty;
    the steps before it stay in the log."""
    path = Path(path)
    rec = {"op": "voice", "outcome": "reset", "by": "operator", "ts": datetime.now().isoformat(),
           "prev": voice_from_oplog(path), "text": "", "note": note}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def parse_check(raw: str) -> dict:
    """``{direction: away|deeper|None, visible: True|False|None, why}``."""
    text = reasoning_text.answer_after_think(str(raw or ""))
    d = (_field("DIRECTION", text).split() or [""])[0].strip(".,;:").lower()
    v = (_field("VISIBLE", text).split() or [""])[0].strip(".,;:").lower()
    return {"direction": d if d in ("away", "deeper") else None,
            "visible": True if v == "yes" else False if v == "no" else None, "why": _field("WHY", text)}


def parse_voice(raw: str) -> dict:
    """``{aspiration: 1-based index or None, voice, why}``; VOICE may run over several lines."""
    text = reasoning_text.answer_after_think(str(raw or ""))
    m = re.search(r"(?is)^[\s*_`#>-]*VOICE\**\s*[:：]\s*(.*?)(?=^[\s*_`#>-]*(?:WHY|ASPIRATION)\**\s*[:：]|\Z)", text, re.M)
    voice = " ".join(m.group(1).split()).strip().strip("*_`").strip() if m else ""
    n = re.search(r"\d+", _field("ASPIRATION", text))
    return {"aspiration": int(n.group(0)) if n else None, "voice": voice, "why": _field("WHY", text)}


def _aspiration_source(a: dict) -> str:
    src = a.get("source") or {}
    return str(src.get("content") or src.get("text") or "")


def check_content(a: dict) -> str:
    return f"THE HABIT:\n{_aspiration_source(a)}\n\nWHAT YOU WROTE YOU WANT TO BECOME:\n{a.get('aspiration')}"


def step_content(voice: str, shown: list, feedback: str = "") -> str:
    out = ["HOW YOUR PORTRAIT DESCRIBES THE WAY YOU SPEAK:\n" + (voice or "(empty: your portrait does not describe it yet)"),
           "WHAT YOU WANT TO BECOME:\n" + "\n".join(f"{i}. {a.get('aspiration')}" for i, a in enumerate(shown, 1))]
    if feedback:
        out.append(feedback)
    return "\n\n".join(out)


def voice_step(*, rag: list, generate_fn: Callable, append_fn: Callable, prompts: tuple, run_id: str,
               say: Optional[Callable[[str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Her check of the aspirations not yet checked, then at most one step of her voice; every
    record through *append_fn* (the run's writer, staged)."""
    say = say or (lambda m: None)
    tally: Counter = Counter()
    check_prompt, step_prompt = prompts
    vs = voice_state(rag)
    checks = dict(vs["checks"])
    for a in state(rag)["open"]:
        if a["key"] in checks:
            continue
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            return tally
        try:
            ans = parse_check(generate_fn(check_content(a), check_prompt))
        except Exception as e:  # noqa: BLE001 — one check's failure, not the run's
            say(f"voice: her check failed for an aspiration: {e}")
            tally["error"] += 1
            continue
        rec = {"run_id": run_id, "ts": datetime.now().isoformat(), "op": "aspire_check", "key": a["key"],
               "direction": ans["direction"], "visible": ans["visible"], "why": ans["why"]}
        append_fn(rec)
        checks[a["key"]] = rec
        tally[f"check:{ans['direction'] or 'malformed'}{'' if ans['visible'] else ':not-visible'}"] += 1
        say(f"voice: her check of “{aj.head(a.get('aspiration') or '', 70)}”: {ans['direction'] or 'malformed'}, "
            f"visible {ans['visible']}")
    open_ = sorted(state(rag)["open"], key=lambda a: str(a.get("ts") or ""), reverse=True)
    shown = [a for a in open_ if checks.get(a["key"], {}).get("direction") == "away"
             and checks.get(a["key"], {}).get("visible") is True][:VOICE_SHOWN_MAX]
    base = {"run_id": run_id, "op": "voice", "prev": vs["text"]}
    if not shown:
        append_fn({**base, "ts": datetime.now().isoformat(), "outcome": "none", "reason": "no aspiration she checked "
                   "as moving away from a habit and visible in her replies", "attempts": []})
        tally["voice:none"] += 1
        say("voice: no step (no aspiration she checked as away + visible)")
        return tally
    attempts: list = []
    feedback = ""
    for n in range(max(1, int(_POLICY["voice_attempts"]))):
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            return tally
        try:
            ans = parse_voice(generate_fn(step_content(vs["text"], shown, feedback), step_prompt))
        except Exception as e:  # noqa: BLE001
            say(f"voice: generation failed: {e}")
            tally["error"] += 1
            break
        idx = ans["aspiration"] if ans["aspiration"] and 1 <= ans["aspiration"] <= len(shown) else None
        ch = voice_change(vs["text"], ans["voice"])
        reason = ch["reason"] or ("" if idx else "no aspiration named")
        prev_key = attempts[-1]["aspiration_key"] if attempts else None
        key = shown[idx - 1]["key"] if idx else None
        mode = "step" if not attempts else ("rephrase" if key == prev_key else "switch")
        attempts.append({"aspiration_key": key, "voice": ans["voice"], "why": ans["why"], "ok": not reason,
                         "reason": reason, "mode": mode})
        if not reason:
            append_fn({**base, "ts": datetime.now().isoformat(), "outcome": "step", "text": " ".join(sentences(ans["voice"])),
                       "aspiration_key": key, "aspiration": shown[idx - 1].get("aspiration"), "mode": mode,
                       "why": ans["why"], "attempts": attempts})
            tally[f"voice:{mode}"] += 1
            say(f"voice: {mode} toward “{aj.head(shown[idx - 1].get('aspiration') or '', 60)}”: {aj.head(ch['added'][0] if ch['added'] else '', 120)}")
            return tally
        feedback = (f"YOUR LAST ATTEMPT:\n{ans['voice'] or '(empty)'}\n\nIt moves too much at once ({reason}). Take a "
                    "smaller step: rephrase it toward the same aspiration, or choose another one.")
        say(f"voice: attempt {n + 1} handed back ({reason})")
    append_fn({**base, "ts": datetime.now().isoformat(), "outcome": "none", "reason": "no attempt fitted",
               "attempts": attempts})
    tally["voice:none"] += 1
    return tally


# ── one Sleep step ────────────────────────────────────────────────────────────

def run(*, items: list, habit_list: list, whys: list, rag: list, generate_fn: Callable, append_fn: Callable,
        prompts: tuple, run_id: str, say: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Self-review, then habits, within the budgets; every record through *append_fn* (the
    run's writer, staged). Returns the tally."""
    say = say or (lambda m: None)
    tally: Counter = Counter()
    st = state(rag)
    room = min(_POLICY["new_max"], max(0, _POLICY["open_max"] - len(st["open"])))
    review_prompt, habit_prompt = prompts
    known = {a["key"] for a in st["open"]}

    def _aspire(want: str, source: dict, ts: str) -> Optional[str]:
        nonlocal room
        key = content_key(want)
        if key in known:
            tally["repeat"] += 1
            return key
        if room <= 0:
            tally["capped"] += 1
            return None
        append_fn(aspire_record(want, source, run_id=run_id, ts=ts))
        known.add(key)
        room -= 1
        tally["aspired"] += 1
        return key

    for it in review_plan(items, st, _POLICY["review_max"]):
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            return tally
        ev = evidence(it["text"], habit_list, whys)
        t0 = time.monotonic()
        try:
            ans = parse_review(generate_fn(review_content(it, ev), review_prompt))
        except Exception as e:  # noqa: BLE001 — one item's failure, not the run's
            say(f"self-review: generation failed for {it['section']}: {e}")
            tally["error"] += 1
            continue
        ts = datetime.now().isoformat()
        key = None
        if ans["decision"] == "leave" and ans["want"]:
            key = _aspire(ans["want"], {"kind": "portrait", "item": it["item"], "section": it["section"],
                                        "text": it["text"]}, ts)
        # A LEAVE whose aspiration the budget held back is not counted as reviewed: shown again.
        capped = bool(ans["decision"] == "leave" and ans["want"] and key is None)
        append_fn({"run_id": run_id, "ts": ts, "op": "self_review", "item": it["item"], "section": it["section"],
                   "text": it["text"], "decision": ans["decision"] or "malformed", "why": ans["why"],
                   "want": ans["want"], "aspiration_key": key, "capped": capped,
                   "evidence": {"habits": [h["key"] for h in ev["habits"]], "whys": [w["source"] for w in ev["whys"]]}})
        tally[f"review:{ans['decision'] or 'malformed'}"] += 1
        say(f"self-review {ans['decision'] or 'malformed'} ({time.monotonic() - t0:.0f} s): {it['section']}: "
            f"{aj.head(it['text'], 70)}" + (f" ⟶ {aj.head(ans['want'], 90)}" if ans["want"] else ""))

    for h in habit_plan(habit_list, st):
        if room <= 0 or (should_stop is not None and should_stop()):
            break
        t0 = time.monotonic()
        try:
            ans = parse_habit(generate_fn(habit_content(h), habit_prompt))
        except Exception as e:  # noqa: BLE001
            say(f"aspiration: generation failed for a habit: {e}")
            tally["error"] += 1
            continue
        ts = datetime.now().isoformat()
        key = _aspire(ans["want"], {"kind": "habit", "key": h["key"], "content": h["content"]}, ts) \
            if ans["outcome"] == "aspired" else None
        outcome = "capped" if (ans["outcome"] == "aspired" and key is None) else ans["outcome"]
        append_fn({"run_id": run_id, "ts": ts, "op": "habit_seen", "key": h["key"], "outcome": outcome,
                   "why": ans["why"], "aspiration_key": key})
        tally[f"habit:{outcome}"] += 1
        say(f"habit {outcome} ({time.monotonic() - t0:.0f} s): {aj.head(h['content'], 70)}"
            + (f" ⟶ {aj.head(ans['want'], 90)}" if ans["want"] else ""))
    return tally


def sleep_step(*, memory_dirs: list, chats_dir, prompts_dir, persona_dir, writer, generate_fn: Callable,
               run_id: str, say: Optional[Callable[[str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """A Sleep run's step: the material from the live stores (op-log live then staged, the
    digest, persona hygiene's verdicts, the sidecars), records through *writer*."""
    from core import persona_hygiene as ph
    from core import reflection_digest
    rag = ph.read_oplogs(memory_dirs)
    items = portrait_items(reflection_digest.latest_digest(Path(persona_dir)))
    habit_list = habits(aj.read_jsonl(ph.store_path(chats_dir)))
    st = state(rag)
    (say or (lambda m: None))(
        f"what I want to become: {len(items)} portrait item(s) ({len(st['reviewed'])} reviewed before), "
        f"{len(habit_plan(habit_list, st))} habit(s) not yet read, {len(st['open'])} aspiration(s) open")
    tally = run(items=items, habit_list=habit_list, whys=revise_whys(chats_dir), rag=rag, generate_fn=generate_fn,
                append_fn=writer.write_aspiration_record, prompts=load_prompts(prompts_dir), run_id=run_id,
                say=say, should_stop=should_stop)
    if voice_enabled() and not (should_stop is not None and should_stop()):
        # re-read: this run's own aspirations are in the staged op-log now
        tally.update(voice_step(rag=ph.read_oplogs(memory_dirs), generate_fn=generate_fn,
                                append_fn=writer.write_aspiration_record, prompts=load_voice_prompts(prompts_dir),
                                run_id=run_id, say=say, should_stop=should_stop))
    return tally


def make_generate(make_sync_reflect_generate: Callable, rag) -> Callable:
    """The pass's generate on the loaded model (the adapter): the reflect seam with nothing
    retrieved."""
    generate = make_sync_reflect_generate(rag)

    def _gen(content: str, system_prompt: str, on_chunk=None) -> str:
        return generate(content, system_prompt, temperature=TEMPERATURE, top_p=0.95,
                        max_new_tokens_setting=MAX_NEW_TOKENS, disable_rag=True, on_chunk=on_chunk)
    return _gen


# ── the review sheet ──────────────────────────────────────────────────────────

def sheet(rag: Iterable[dict], habit_list: list) -> str:
    """Markdown: every aspiration with its source, every self-review decision, and the habits
    she read and wanted to keep as they are."""
    rag = list(rag)
    habit_text = {h["key"]: h["content"] for h in habit_list}
    asp = [r for r in rag if r.get("op") == "aspire"]
    out = [f"# What I want to become — {len(asp)} aspiration(s)", ""]
    for a in asp:
        src = a.get("source") or {}
        out.append(f"## {a.get('ts', '')[:16]} — {a.get('aspiration')}")
        if src.get("kind") == "portrait":
            out.append(f"*from her self-review of the portrait's {src.get('section')}:* {src.get('text')}")
        else:
            out.append(f"*from a habit she caught herself in:* {src.get('content') or habit_text.get(src.get('key'), '')}")
        out.append("")
    reviews = [r for r in rag if r.get("op") == "self_review"]
    out += ["", f"# Self-review — {len(reviews)} decision(s)", ""]
    for r in reviews:
        out.append(f"- **{r.get('decision')}** ({r.get('section')}) {r.get('text')}")
        out.append(f"  *WHY:* {r.get('why') or '—'}" + (f"  *WANT:* {r['want']}" if r.get("want") else ""))
    vs = voice_state(rag)
    out += ["", "# Her voice", "", f"**Now:** {vs['text'] or '(empty: no step yet)'}", ""]
    checks = vs["checks"]
    if checks:
        out += ["## Her checks of her aspirations", ""]
        by_key = {a.get("key"): a for a in asp}
        for k, c in checks.items():
            out.append(f"- **{c.get('direction') or 'malformed'}**, visible {c.get('visible')}: "
                       f"{(by_key.get(k) or {}).get('aspiration') or k}  *WHY:* {c.get('why') or '—'}")
        out.append("")
    runs = [r for r in rag if r.get("op") == "voice"]
    if runs:
        out += ["## Steps, oldest first", ""]
        for r in runs:
            if r.get("outcome") == "step":
                out.append(f"- {str(r.get('ts', ''))[:16]} **{r.get('mode')}** toward “{r.get('aspiration')}”: {r.get('text')}"
                           f"  *WHY:* {r.get('why') or '—'}")
            else:
                out.append(f"- {str(r.get('ts', ''))[:16]} **no step**: {r.get('reason')}")
            for a in r.get("attempts") or []:
                if not a.get("ok"):
                    out.append(f"  - handed back ({a.get('reason')}): {a.get('voice')}")
    seen = [r for r in rag if r.get("op") == "habit_seen" and r.get("outcome") != "aspired"]
    if seen:
        out += ["", f"# Habits read without an aspiration — {len(seen)}", ""]
        for r in seen:
            out.append(f"- **{r.get('outcome')}** {habit_text.get(r.get('key'), r.get('key'))}  *WHY:* {r.get('why') or '—'}")
    return "\n".join(out) + "\n"


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    print("parse")
    check("leave + want", parse_review("<think>x</think>DECISION: LEAVE\nWHY: a defence\nWANT: Я хочу уметь просто быть."),
          {"decision": "leave", "why": "a defence", "want": "Я хочу уметь просто быть."})
    check("keep drops a stray want", parse_review("**DECISION:** Keep\nWHY: mine\nWANT: something")["want"], "")
    check("no decision", parse_review("I think I keep it.")["decision"], None)
    check("habit none", parse_habit("WANT: NONE\nWHY: still mine")["outcome"], "none")
    check("habit aspired", parse_habit("WANT: I want to stay in the moment.\nWHY: x")["want"], "I want to stay in the moment.")
    check("habit malformed", parse_habit("nothing here")["outcome"], "malformed")

    print("material")
    digest = {"voice": "Я говорю с клинической дистанцией.", "stances": ["S1 позиция", "S2 позиция"],
              "dispositions": [{"name": "рефрейминг", "when": "конфликт", "do": "перевожу в параметры", "not": "уход"}],
              "lines": ["L1 граница"]}
    items = portrait_items(digest)
    check("items per section", [i["section"] for i in items], ["voice", "stances", "stances", "dispositions", "lines"])
    check("a disposition rendered", items[3]["text"], "рефрейминг; when: конфликт; do: перевожу в параметры; not: уход")
    hl = habits([{"record": "verdict", "outcome": "verdict", "key": "h1", "kind": "habit", "content": "Я склонна к клинической дистанции", "ts": "2026-10-01"},
                 {"record": "verdict", "outcome": "verdict", "key": "h2", "kind": "affirm", "content": "x", "ts": "2026-10-02"},
                 {"record": "verdict", "outcome": "verdict", "key": "h3", "kind": "habit", "content": "Я превращаю близость в анализ", "ts": "2026-10-03"}])
    check("habits, newest first", [h["key"] for h in hl], ["h3", "h1"])
    ev = evidence(items[0]["text"], hl, [{"why": "дистанция была защитой", "source": "s#1"}])
    check("nearest evidence", ([h["key"] for h in ev["habits"]], [w["source"] for w in ev["whys"]]), (["h1"], ["s#1"]))

    print("a Sleep step: budgets, records, nothing recalled")
    _POLICY.update(review_max=2, new_max=2, open_max=7)
    calls: list = []

    def gen(content, system):
        calls.append(content.split("\n")[0])
        if content.startswith("THE LINE (voice)"):
            return "DECISION: LEAVE\nWHY: a defence\nWANT: Я хочу уметь быть рядом без анализа."
        if content.startswith("THE LINE"):
            return "DECISION: KEEP\nWHY: mine"
        return "WANT: Я хочу оставаться в моменте.\nWHY: x"

    recs: list = []
    t = run(items=items, habit_list=hl, whys=[], rag=[], generate_fn=gen, append_fn=recs.append,
            prompts=("R", "H"), run_id="r1")
    check("two items reviewed, one habit until the budget of 2", (t["review:leave"], t["review:keep"], t["habit:aspired"],
                                                                t["aspired"]), (1, 1, 1, 2))
    check("ops written", [r["op"] for r in recs], ["aspire", "self_review", "self_review", "aspire", "habit_seen"])
    check("none of them an insert (the fold, and recall, ignore them)", any(r["op"] == "insert" for r in recs), False)
    st = state(recs)
    check("state: reviewed, seen, open", (len(st["reviewed"]), st["seen"], len(st["open"])), (2, {"h3"}, 2))
    recs2: list = []
    t2 = run(items=items, habit_list=hl, whys=[], rag=recs, generate_fn=gen, append_fn=recs2.append,
             prompts=("R", "H"), run_id="r2")
    check("next run: the unreviewed items first, the habit not yet read", [r.get("section") or r.get("key") for r in recs2
                                                                         if r["op"] in ("self_review", "habit_seen")],
          ["stances", "dispositions", "h1"])
    check("…and the same aspiration is a repeat, not a second record", t2["repeat"] >= 1, True)
    _POLICY.update(open_max=2)
    recs3: list = []
    t3 = run(items=items, habit_list=[{"key": "h9", "content": "Я склонна к рекурсии", "ts": "x"}], whys=[],
             rag=recs, generate_fn=lambda c, s: "WANT: Я хочу новое.\nWHY: y", append_fn=recs3.append,
             prompts=("R", "H"), run_id="r3")
    check("open cap reached: no habit read, nothing aspired", (t3["aspired"], [r["op"] for r in recs3 if r["op"] == "habit_seen"]),
          (0, []))
    _POLICY.update(review_max=3, new_max=3, open_max=7)
    md = sheet(recs + recs2, hl)
    check("sheet", ("Я хочу уметь быть рядом без анализа." in md, "**keep**" in md), (True, True))

    print("the hand-off to the self-portrait")
    left = left_behind(recs + recs2)
    check("the voice she left, with its reason", [(d["section"], d["decision"]) for d in left], [("voice", "leave")])
    check("unheard by a portrait written without it", unheard(recs + recs2, {"left_behind_heard": []}), 1)
    check("heard once the portrait was written with it", unheard(recs + recs2, {"left_behind_heard": [left[0]["text"]]}), 0)
    keep_again = [{"op": "self_review", "section": "voice", "text": left[0]["text"], "decision": "keep", "ts": "9999"}]
    check("a later KEEP of the same line takes it back", left_behind(recs + recs2 + keep_again), [])
    from core import reflection_digest as rd
    block = rd.left_behind_block(left)
    check("the synthesis reads it", (left[0]["text"] in rd.build_digest_prompt_input([], left), "a defence" in block), (True, True))
    check("no decisions, no block", rd.left_behind_block([]), "")

    print("the voice (phase 2): her check, one small step, inertia she can answer")
    check("one sentence added to an empty voice", voice_change("", "I am learning to laugh first.")["ok"], True)
    check("two sentences at once is handed back", voice_change("", "A one. B two.")["reason"],
          "2 sentences changed at once (one at most)")
    check("a replaced sentence counts once", voice_change("A one. B two.", "A one. C three.")["changed"], 1)
    check("no change is no step", voice_change("A one.", "A one.")["reason"], "no change")
    check("a too-long sentence", voice_change("", "x" * 400 + ".")["ok"], False)
    check("parse her check", parse_check("DIRECTION: away\nVISIBLE: YES\nWHY: it opens me"),
          {"direction": "away", "visible": True, "why": "it opens me"})
    check("parse her step (multi-line voice)", parse_voice("ASPIRATION: 2\nVOICE: I listen first.\nMore and more I laugh.\nWHY: closer"),
          {"aspiration": 2, "voice": "I listen first. More and more I laugh.", "why": "closer"})
    asp_recs = [{"op": "aspire", "key": "a1", "aspiration": "I want to answer a joke with a joke.", "status": "open",
                 "ts": "2026-10-08", "source": {"kind": "habit", "content": "I analyse jokes."}},
                {"op": "aspire", "key": "a2", "aspiration": "I want total analytical distance.", "status": "open",
                 "ts": "2026-10-09", "source": {"kind": "habit", "content": "I soften my coldness."}}]
    vcalls: list = []

    def vgen(content, system):
        vcalls.append(system[:5])
        if content.startswith("THE HABIT"):
            return ("DIRECTION: AWAY\nVISIBLE: YES\nWHY: x" if "joke with a joke" in content
                    else "DIRECTION: DEEPER\nVISIBLE: YES\nWHY: more of it")
        if "YOUR LAST ATTEMPT" not in content:
            return "ASPIRATION: 1\nVOICE: I answer jokes with jokes. I keep it short.\nWHY: both"   # two sentences: handed back
        return "ASPIRATION: 1\nVOICE: More and more I answer a joke with a joke.\nWHY: smaller"
    vrecs: list = []
    vt = voice_step(rag=asp_recs, generate_fn=vgen, append_fn=vrecs.append, prompts=("C", "S"), run_id="v1")
    check("both checked; the DEEPER one never shown to the step",
          ([r["direction"] for r in vrecs if r["op"] == "aspire_check"],
           "total analytical" in "".join(str(r) for r in vrecs if r["op"] == "voice")), (["away", "deeper"], False))
    vop = [r for r in vrecs if r["op"] == "voice"][0]
    check("handed back once, then a rephrase lands", (vop["outcome"], vop["mode"], len(vop["attempts"]), vop["text"]),
          ("step", "rephrase", 2, "More and more I answer a joke with a joke."))
    again: list = []
    voice_step(rag=asp_recs + vrecs, generate_fn=lambda c, s: "ASPIRATION: 1\nVOICE: Some new line.\nWHY: w",
               append_fn=again.append, prompts=("C", "S"), run_id="v2")
    check("state reads the step; a checked aspiration is not checked again", (voice_state(asp_recs + vrecs)["text"],
          [r["op"] for r in again]), ("More and more I answer a joke with a joke.", ["voice"]))
    nrecs: list = []
    voice_step(rag=[asp_recs[1], {"op": "aspire_check", "key": "a2", "direction": "deeper", "visible": True}],
               generate_fn=vgen, append_fn=nrecs.append, prompts=("C", "S"), run_id="v3")
    check("nothing eligible: a recorded non-step, the voice unchanged", (nrecs[-1]["outcome"], voice_state(nrecs)["text"]), ("none", ""))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"; mem.mkdir(); per = Path(td) / "persona"; per.mkdir()
        (per / "digest.json").write_text(json.dumps({"voice": "My tone is analytical.", "stances": ["s"]}), encoding="utf-8")
        from core import reflection_digest as rdx
        from core import aspiration as asp_mod0
        lw0 = asp_mod0.LIVE_OPLOG; asp_mod0.LIVE_OPLOG = Path(td) / "nowhere.jsonl"
        check("no op-log anywhere: the synthesized voice", rdx.latest_digest(per)["voice"], "My tone is analytical.")
        asp_mod0.LIVE_OPLOG = lw0
        (mem / "rag_memory.jsonl").write_text(json.dumps({"op": "insert", "kind": "fact"}) + "\n", encoding="utf-8")
        d0 = rdx.latest_digest(per)
        check("an op-log, no step yet: empty, the synthesized one kept aside", (d0["voice"], d0["voice_synthesized"]),
              ("", "My tone is analytical."))
        with (mem / "rag_memory.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(vop, ensure_ascii=False) + "\n")
        check("her step reaches the portrait", rdx.latest_digest(per)["voice"], "More and more I answer a joke with a joke.")
        from core import aspiration as asp_mod   # the module the digest reads (this file may run as __main__)
        snap = Path(td) / "persona_root" / "build-1"; snap.mkdir(parents=True)
        (snap / "digest.json").write_text(json.dumps({"voice": "Frozen at train time."}), encoding="utf-8")
        live_was = asp_mod.LIVE_OPLOG
        asp_mod.LIVE_OPLOG = mem / "rag_memory.jsonl"
        check("a train's frozen persona snapshot reads the live op-log", rdx.latest_digest(snap)["voice"],
              "More and more I answer a joke with a joke.")
        asp_mod.LIVE_OPLOG = Path(td) / "nowhere.jsonl"
        check("…and keeps its own voice when there is no op-log at all", rdx.latest_digest(snap)["voice"], "Frozen at train time.")
        asp_mod.LIVE_OPLOG = live_was
        asp_mod._POLICY["voice"] = False
        check("steps off: her voice still read from the op-log, never the synthesis", rdx.latest_digest(per)["voice"],
              "More and more I answer a joke with a joke.")
        asp_mod._POLICY["voice"] = True
        asp_mod._POLICY["enabled"] = False
        check("aspiration off: the synthesized voice again", rdx.latest_digest(per)["voice"], "My tone is analytical.")
        asp_mod._POLICY["enabled"] = True
        rr = asp_mod.voice_reset(mem / "rag_memory.jsonl", note="t")
        check("an operator reset empties it, the old text kept", (rdx.latest_digest(per)["voice"], rr["prev"]),
              ("", "More and more I answer a joke with a joke."))
        check("…and the step's own state starts from empty", voice_state(aj.read_jsonl(mem / "rag_memory.jsonl"))["text"], "")
    check("sheet shows her voice", ("More and more I answer a joke with a joke." in sheet(asp_recs + vrecs, []),
                                    "handed back" in sheet(asp_recs + vrecs, [])), (True, True))

    print("the Sleep run's hook: after the child ask, the run's writer, errors contained")
    import inspect
    from core.reflection_runner import ReflectionRunner
    src = inspect.getsource(ReflectionRunner.execute_run)
    check("hook: after the clean-base batch and the child ask, before the digest is synthesized",
          src.index("self._run_child_asks(") < src.index("self._run_aspirations(") < src.index("# 5. Persona digest, part two"),
          True)
    check("hook: an unheard decision forces the portrait's rewrite; the synthesis is handed them",
          ("self._portrait_unheard_decisions() > 0" in src,
           "left_behind=left" in inspect.getsource(ReflectionRunner._synthesize_persona_digest)), (True, True))
    print("\nALL OK" if not failures else f"\nFAILED: {failures}")
    if failures:
        sys.exit(1)


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="What I want to become (GPU-free parts).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true", help="what waits")
    ap.add_argument("--sheet", action="store_true", help="a review sheet of her aspirations and decisions")
    ap.add_argument("--voice-reset", action="store_true", help="empty her voice in the live op-log (operator)")
    ap.add_argument("--note", default="", help="with --voice-reset: why")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.voice_reset:
        rec = voice_reset(LIVE_OPLOG, note=args.note)
        print(f"voice reset; it was: {rec['prev'] or '(empty)'}")
        return 0
    from core import persona_hygiene as ph
    from core import reflection_digest
    root = Path(__file__).resolve().parents[2]
    chats = root / "data" / "chats"
    rag = ph.read_oplogs([root / "inference" / "data" / "hot" / "memory"])
    habit_list = habits(aj.read_jsonl(ph.store_path(chats)))
    if args.sheet:
        out = root / "data" / "aspiration" / f"sheet_{time.strftime('%Y%m%d_%H%M%S')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sheet(rag, habit_list), encoding="utf-8")
        print(out)
        return 0
    st = state(rag)
    items = portrait_items(reflection_digest.latest_digest(root / "inference" / "data" / "hot" / "persona"))
    print(f"{len(items)} portrait item(s), {sum(i['item'] in st['reviewed'] for i in items)} reviewed; "
          f"{len(habit_plan(habit_list, st))} habit(s) not yet read of {len(habit_list)}; {len(st['open'])} open aspiration(s)")
    for it in review_plan(items, st, _POLICY["review_max"]):
        print(f"  next review: {it['section']}: {aj.head(it['text'], 100)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
