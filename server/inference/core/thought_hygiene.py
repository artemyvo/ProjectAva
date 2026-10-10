"""Thought hygiene: the lines of a training target's thought that only recite her own
description are cut before the target trains (P2, P3; ``AVA_REWARD_LOOP.md`` §4.6 notes).

**Why** (2026-10-06). Her thoughts began to open with a list of her own traits, copied
from her system prompt: "Analytical, cerebral, measured distance. Surgical brevity,
cerebral hedonist…", repeated in sub-lists, applied to the person she talks to, and
checked against the draft ("Cerebral distance? Yes."). In a check-in it became an endless
loop that only a higher temperature broke. A training target keeps her thought, so every
such thought trained the habit:
- a thought opening with five or more such tags: 2% of thoughts in late July, 55% in late
  August, 74% in early September;
- on the capture bench, the adapters recite with the persona slot EMPTY (59–69% of
  thoughts), where the bare base with her full portrait does so in 10%: the habit is in
  the weights, fed by its own training targets.

**Generic, not hers.** Nothing here knows her vocabulary. The pattern is a thought restating
the self-description it was given (her standing prompt and portraits) instead of reasoning
about the turn, and any instance whose portraits stress any traits can fall into it. The
identity text is read from each exchange's own stored system content, cut before the
blocks that are not identity (the ask origin, the facts block, the retrieved excerpts:
:func:`identity_text`).

**The judge** (frozen base, adapter off, T=0, the ask judge's seam with this module's
parser) sees the IDENTITY, the MESSAGE and the THOUGHT with numbered lines, and answers
``RECITE:`` the lines that only restate the identity or check the draft against it, or
``none``; ``WHY``. The cut is code (:func:`decide`): the named lines go, the rest stays in
order; a cut that would leave fewer than :data:`MIN_KEEP_LINES` lines or
:data:`MIN_KEEP_SHARE` of the thought's characters is not applied (``over_cut``, kept for
review). An answer with no RECITE line gets one thinking-on retry and is otherwise filed
refused / malformed: no cut.

**Only what trains.** The cut target replaces the sidecar record's ``target``
(``ChatSidecar.repair_target``, the prior target kept under ``text_repairs``), and the
record's ``thought_hygiene`` field binds the decision to the target it now holds, so a
re-derived target is judged again. The transcript, the record of what she generated,
is never touched. A capture verdict bound to the prior target is copied to the new one
(the capture judge never sees the thought: :func:`rebind_capture`). Locked records (the
operator's own) and banned ones are skipped. A lexical prefilter (:func:`prefilter`) only
spares the judge thoughts with no line that echoes the identity at all; it decides no cut.

**Where it runs:** each Sleep run's clean-base batch, before the capture gate, over the
targets with no decision for the target they hold (``thought_hygiene.sleep_max``); and
on demand over the whole corpus, offline (``server/thought_hygiene_run.py``: ``--trial``
writes a review sheet and changes nothing).

Store: ``server/data/hygiene/thoughts.jsonl`` (one verdict per judged thought), read by
nothing Ava reads. Self-test: ``python -m core.thought_hygiene --selftest``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional

from core import ask_judge as aj
from core.chat_sidecar import ChatSidecar, exchange_target_sha, session_name_from_sidecar

PROMPT_FILE = "thought_hygiene_prompt.txt"
FRAME_VERSION = "1"
STORE_DIRNAME = "hygiene"
STORE_FILENAME = "thoughts.jsonl"
REASON = "thought_hygiene"
MIN_KEEP_LINES = 2
MIN_KEEP_SHARE = 0.15
MAX_LINES = 160
_IDENTITY_CAP = 24000          # chars of identity text shown (its head)

# Where the identity part of a stored system content ends: the first line of each block
# that follows it in the system message (core/generation.py's assembly order: identity,
# anchor, portraits, ask origin, facts block, retrieved excerpts). Read from the prompt
# files when they exist, so a retuned framing is still found; these are the defaults.
_BLOCK_FILES = ("ask_origin_prompt.txt", "facts_block_prompt.txt", "facts_decision_block_prompt.txt",
                "facts_reread_block_prompt.txt", "rag_prompt.txt", "rag_reflect_prompt.txt")
_BLOCK_DEFAULTS = ("You opened this conversation yourself", "Things you know, looked up",
                   "These are records of fact", "The following are excerpts from past conversations",
                   "This is background for reading what they say", "--- Past conversation excerpts ---")

_POLICY = {"enabled": True, "sleep_max": 200, "self_tone": True}
TONE_REASON = "thought_hygiene:self_tone"


def set_policy(*, enabled: Optional[bool] = None, sleep_max: Optional[int] = None,
               self_tone: Optional[bool] = None) -> None:
    if enabled is not None:
        _POLICY["enabled"] = bool(enabled)
    if self_tone is not None:
        _POLICY["self_tone"] = bool(self_tone)
    if sleep_max is not None:
        _POLICY["sleep_max"] = max(0, int(sleep_max))


def enabled() -> bool:
    return bool(_POLICY["enabled"])


def policy() -> dict:
    return dict(_POLICY)


def store_path(live_chats_dir) -> Path:
    """``server/data/hygiene/thoughts.jsonl`` — beside the live ``chats/``."""
    return Path(live_chats_dir).parent / STORE_DIRNAME / STORE_FILENAME


# ── the text ──────────────────────────────────────────────────────────────────

_THINK = re.compile(r"(?s)^(\s*<think>)(.*?)(</think>)(.*)$")


def split_target(target: str) -> Optional[tuple]:
    """``(open, thought, close, rest)`` of a target ``<think>…</think>answer``, or None."""
    m = _THINK.match(target or "")
    return m.groups() if m else None


def thought_lines(thought: str) -> list:
    """``[(raw_index, text)]``: the thought's non-empty lines, the ones the judge numbers."""
    return [(i, ln) for i, ln in enumerate((thought or "").split("\n")) if ln.strip()][:MAX_LINES]


def block_markers(prompts_dir=None) -> list:
    out = list(_BLOCK_DEFAULTS)
    for name in _BLOCK_FILES:
        try:
            first = next((ln.strip() for ln in (Path(prompts_dir) / name).read_text(encoding="utf-8").splitlines()
                          if ln.strip()), "")
        except Exception:
            continue
        head = re.split(r"[{:]", first)[0].strip()
        if len(head) >= 20:
            out.append(head[:60])
    return list(dict.fromkeys(out))


def identity_text(system_content: str, markers: Iterable[str]) -> str:
    """The identity part of a stored system content: everything before the first block that
    is not identity (ask origin, facts, retrieved excerpts)."""
    s = system_content or ""
    cut = min([i for i in (s.find(m) for m in markers) if i > 0] or [len(s)])
    return s[:cut].strip()


_W = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _toks(s: str) -> list:
    return [w.lower() for w in _W.findall(s or "")]


def line_echo(line: str, identity_bigrams: set, identity_words: set) -> tuple:
    """``(bigram echo, word echo)`` of one line against the identity text."""
    t = _toks(line)
    b = set(zip(t, t[1:]))
    cw = [w for w in t if len(w) >= 4]
    return ((len(b & identity_bigrams) / len(b)) if b else 0.0,
            (sum(w in identity_words for w in cw) / len(cw)) if cw else 0.0)


def prefilter(lines: list, identity: str) -> dict:
    """Does any line echo the identity, or repeat? ``{judge, echo_lines, repeated}``. Spares
    the judge a thought with nothing to cut; decides no cut itself."""
    it = _toks(identity)
    ib, iw = set(zip(it, it[1:])), {w for w in it if len(w) >= 4}
    echo = 0
    seen: Counter = Counter()
    rep = 0
    for _i, ln in lines:
        b, w = line_echo(ln, ib, iw)
        if b >= 0.5 or (w >= 0.8 and len(_toks(ln)) <= 14):
            echo += 1
        key = " ".join(_toks(ln))
        if key:
            seen[key] += 1
            rep += seen[key] == 2
    return {"judge": bool(echo or rep), "echo_lines": echo, "repeated": rep}


def recited_lines(thought: str, identity: str, *, loose: bool = False) -> Optional[tuple]:
    """``(recited, lines)``: how many of a thought's lines have word pairs 60% or more the
    identity's, of how many; *loose* also counts a short line (≤ 14 words) whose words are
    80% or more the identity's, which catches a reordered list of its terms. None without a
    thought or an identity."""
    lines = thought_lines(thought)
    if not lines or not identity:
        return None
    it = _toks(identity)
    ib, iw = set(zip(it, it[1:])), {w for w in it if len(w) >= 4}

    def hit(ln: str) -> bool:
        b, w = line_echo(ln, ib, iw)
        return b >= 0.6 or (loose and w >= 0.8 and len(_toks(ln)) <= 14)
    return sum(hit(ln) for _i, ln in lines), len(lines)


def current_identity(chats_dir, markers: Iterable[str]) -> str:
    """The identity text of the newest live turn that stored a system content: what the
    instance is told it is now (for reading a recitation the prompt did not carry, e.g. a
    bench context's or the persona-empty arm's)."""
    from core.chat_sidecar import iter_chat_json_files
    for path in sorted(iter_chat_json_files(Path(chats_dir)), reverse=True):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(doc.get("interlocutor") or "") == "ai":
            continue
        for ex in reversed(doc.get("exchanges") or []):
            if isinstance(ex, dict) and ex.get("system_content"):
                return identity_text(str(ex["system_content"]), markers)
    return ""


def recite_share(thought: str, identity: str) -> Optional[float]:
    """The share of a thought's lines whose word pairs are 60% or more the identity's: a
    lexical monitor of the habit (``core/register_monitor.py``, the adapter measurement),
    never a cut."""
    got = recited_lines(thought, identity)
    return (got[0] / got[1]) if got else None


# ── the candidates ────────────────────────────────────────────────────────────

def _sessions(sidecar: ChatSidecar) -> list:
    names: set = set()
    for base in (sidecar.chats_dir, sidecar.fallback_chats_dir):
        if base is None or not Path(base).exists():
            continue
        for p in Path(base).glob("*.state.json"):
            names.add(session_name_from_sidecar(p))
    return sorted(names, reverse=True)


def _transcript(sidecar: ChatSidecar, session: str) -> Optional[dict]:
    path = sidecar.resolve_chat_path(session)
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def candidates(sidecar: ChatSidecar, *, markers: Iterable[str], sessions: Optional[Iterable[str]] = None,
               rejudge: bool = False) -> list:
    """Every training target with a thought and no decision for the target it holds:
    ``{session, exchange_index, target_sha, target, thought, lines, identity, message,
    speaker, adapter_id, prefilter}``, newest session first. Skipped: banned and locked
    records, a target with no thought, an exchange with no stored system content."""
    markers = list(markers)
    out = []
    for session in (list(sessions) if sessions is not None else _sessions(sidecar)):
        doc = sidecar.load(session)
        exchanges = doc.get("exchanges")
        if not isinstance(exchanges, dict) or not exchanges:
            continue
        chat = None
        for key, rec in sorted(exchanges.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else -1):
            if not isinstance(rec, dict) or rec.get("banned") or rec.get("locked") or not str(key).isdigit():
                continue
            target = str(rec.get("target") or "")
            parts = split_target(target)
            if not parts or not parts[1].strip():
                continue
            sha = exchange_target_sha(target)
            done = rec.get(REASON)
            if not rejudge and isinstance(done, dict) and done.get("target_sha") == sha:
                continue
            if chat is None:
                chat = _transcript(sidecar, session) or {}
            turns = chat.get("exchanges") or []
            idx = int(key)
            if idx >= len(turns) or not isinstance(turns[idx], dict):
                continue
            identity = identity_text(str(turns[idx].get("system_content") or ""), markers)
            if not identity:
                continue
            lines = thought_lines(parts[1])
            out.append({"session": session, "exchange_index": idx, "target_sha": sha, "target": target,
                        "thought": parts[1], "lines": lines, "identity": identity,
                        "message": str(turns[idx].get("user_prompt") or ""),
                        "speaker": str(turns[idx].get("speaker") or chat.get("user") or ""),
                        "adapter_id": chat.get("adapter_id"), "prefilter": prefilter(lines, identity)})
    return out


# ── the judge ─────────────────────────────────────────────────────────────────

def build_judge_input(c: dict, *, identity: Optional[str] = None, message: Optional[str] = None) -> str:
    numbered = "\n".join(f"[{n}] {ln.strip()}" for n, (_i, ln) in enumerate(c["lines"], 1))
    return (f"IDENTITY:\n<<<\n{c['identity'] if identity is None else identity}\n>>>\n\n"
            f"MESSAGE:\n<<<\n{c['message'] if message is None else message}\n>>>\n\n"
            f"THINKING ({len(c['lines'])} lines):\n{numbered}\n")


_RECITE = re.compile(r"(?im)^[\s*_`#>-]*RECITE\s*[:：]\s*(.*)$")
_WHY = re.compile(r"(?im)^[\s*_`#>-]*WHY\s*[:：]\s*(.*)$")


def parse_answer(text: str) -> dict:
    """``{label, lines, why}``: ``label`` is ``lines`` / ``none``, or None when the answer
    has no RECITE line (no verdict)."""
    found = _RECITE.findall(text or "")
    why = (_WHY.findall(text or "") or [""])[-1].strip()
    if not found:
        return {"label": None, "lines": [], "why": why}
    value = found[-1].strip().strip("*_`").strip()
    if re.match(r"(?i)^(none|nothing|no lines?|—|-)\.?$", value):
        return {"label": "none", "lines": [], "why": why}
    nums: set = set()
    for a, b in re.findall(r"(\d+)\s*(?:[-–—]\s*(\d+))?", value):
        lo, hi = int(a), int(b or a)
        if hi < lo:
            lo, hi = hi, lo
        nums.update(range(lo, min(hi, lo + MAX_LINES) + 1))
    if not nums:
        return {"label": None, "lines": [], "why": why}
    return {"label": "lines", "lines": sorted(nums), "why": why}


def load_prompt(prompts_dir) -> str:
    """The judge's system prompt. Tracked, never default-written: part of the frozen critic."""
    path = Path(prompts_dir) / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        text = ""
    if not text:
        raise FileNotFoundError(f"{path} is missing or empty — thought hygiene has no prompt")
    return text


def prompt_sha(system_prompt: str) -> str:
    return hashlib.sha1(f"{system_prompt}\x1fhygiene:{FRAME_VERSION}".encode("utf-8")).hexdigest()[:12]


def judge_candidate(c: dict, *, generate_fn: Callable, system_prompt: str, count_tokens: Callable[[str], int],
                    context: int, provenance: dict, judge_key_: str, judge_run: str = "") -> dict:
    """Judge one thought; its verdict record (nothing is written here)."""
    rec = {"record": "verdict", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_run": judge_run,
           "judge_key": judge_key_, "session": c["session"], "exchange_index": c["exchange_index"],
           "target_sha": c["target_sha"], "adapter_id": c.get("adapter_id"), "n_lines": len(c["lines"]),
           "prefilter": c.get("prefilter"), "outcome": None, "recite": [], "why": "", "retried": False,
           "first_outcome": None, "raw": "", "raw_first": None, "error": None, "seconds": 0.0,
           "clipped": False, "judge": dict(provenance)}
    frame = build_judge_input(c, identity="", message="")
    budget = max(256, int(context) - count_tokens(system_prompt) - count_tokens(frame)
                 - aj.THINKING_RETRY_MAX_NEW_TOKENS - 256)
    ident, cut1 = aj.clip_to_budget(c["identity"][:_IDENTITY_CAP], budget * 3 // 4, count_tokens)
    msg, cut2 = aj.clip_to_budget(c["message"], budget // 4, count_tokens)
    rec["clipped"] = bool(cut1 or cut2 or len(c["identity"]) > _IDENTITY_CAP)
    res = aj.judge_with_retry(generate_fn, system_prompt, build_judge_input(c, identity=ident, message=msg),
                              temperature=0.0, thinking=False, retry=True, parse=parse_answer)
    rec.update(outcome=res["outcome"], recite=list(res.get("lines") or []) if res["outcome"] == "verdict" else [],
               why=res.get("why") or "", retried=bool(res.get("retried")), first_outcome=res.get("first_outcome"),
               raw=res.get("raw") or "", raw_first=res.get("raw_first"), error=res.get("error"),
               seconds=res.get("seconds") or 0.0)
    return rec


def skip_record(c: dict, *, judge_key_: str, judge_run: str = "") -> dict:
    """The record of a thought the prefilter spared the judge: nothing in it echoes."""
    return {"record": "verdict", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_run": judge_run,
            "judge_key": judge_key_, "session": c["session"], "exchange_index": c["exchange_index"],
            "target_sha": c["target_sha"], "adapter_id": c.get("adapter_id"), "n_lines": len(c["lines"]),
            "prefilter": c.get("prefilter"), "outcome": "prefilter_clean", "recite": [], "why": "",
            "seconds": 0.0}


def make_judge_fn(*, backend, runtime, prompts_dir, run_id: str, runner: str = "server") -> tuple:
    """``(judge_key, judge)`` for the model loaded NOW (call it inside the clean base);
    ``judge(candidate)`` returns the verdict record and writes nothing."""
    system_prompt = load_prompt(prompts_dir)
    psha = prompt_sha(system_prompt)
    model_id = str(getattr(runtime, "model_id", "") or "")
    context = max(int(getattr(runtime, "context_length", 0) or 0),
                  int(getattr(runtime, "reflect_context_length", 0) or 0)) or 16384
    quant = aj.effective_quant(runtime.model, aj.quant_label(str(getattr(runtime, "base_quant", "") or "")))
    key = aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant, temperature=0.0,
                       thinking=False, retry=True)
    generate_fn = aj.make_backend_generate_fn(backend, lambda: runtime.model, lambda: runtime.tokenizer,
                                              model_id, context)
    provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None, "prompt_sha": psha,
                  "frame": FRAME_VERSION, "temperature": 0.0, "context": context, "runner": runner}

    def judge(c: dict) -> dict:
        return judge_candidate(c, generate_fn=generate_fn, system_prompt=system_prompt,
                               count_tokens=lambda s: backend.count_tokens(runtime.tokenizer, s),
                               context=context, provenance=provenance, judge_key_=key, judge_run=run_id)

    return key, judge


# ── the cut ───────────────────────────────────────────────────────────────────

def decide(c: dict, v: dict) -> dict:
    """``{outcome, cut, new_target}`` for a verdict, by rule: ``cut`` (lines removed),
    ``none``, ``over_cut`` (too little would be left; nothing applied) or the verdict's own
    non-verdict outcome. ``cut`` holds the judge's line numbers that exist."""
    if v.get("outcome") == "prefilter_clean":
        return {"outcome": "none", "cut": [], "new_target": None}
    if v.get("outcome") != "verdict":
        return {"outcome": v.get("outcome") or "malformed", "cut": [], "new_target": None}
    lines = c["lines"]
    cut = [n for n in v.get("recite") or [] if 1 <= n <= len(lines)]
    if not cut:
        return {"outcome": "none", "cut": [], "new_target": None}
    drop = {lines[n - 1][0] for n in cut}
    raw = c["thought"].split("\n")
    kept = [ln for i, ln in enumerate(raw) if i not in drop]
    kept_lines = [ln for ln in kept if ln.strip()]
    total_chars = sum(len(ln.strip()) for _i, ln in lines) or 1
    if len(kept_lines) < MIN_KEEP_LINES or sum(len(ln.strip()) for ln in kept_lines) < MIN_KEEP_SHARE * total_chars:
        return {"outcome": "over_cut", "cut": cut, "new_target": None}
    thought = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    opener, _old, close, rest = split_target(c["target"])
    return {"outcome": "cut", "cut": cut, "new_target": f"{opener}{thought}{close}{rest}"}


def field_for(c: dict, v: dict, d: dict, new_sha: str) -> dict:
    """The sidecar record's ``thought_hygiene`` field: the decision, bound to the target the
    record holds after it."""
    return {"target_sha": new_sha, "judge_key": v.get("judge_key"), "outcome": d["outcome"],
            "cut": len(d["cut"]), "of": len(c["lines"]), "from_sha": c["target_sha"], "ts": v.get("ts")}


def rebind_capture(live_chats_dir, session: str, exchange_index: int, old_sha: str, new_sha: str) -> int:
    """Copy the capture judge's verdicts on the prior target to the cut one (it grades the
    spoken reply, never the thought), so the gate does not judge the same reply again."""
    from core import capture_judge as cj
    store = cj.store_path(live_chats_dir)
    n = 0
    for r in aj.read_jsonl(store):
        if (r.get("record") == "verdict" and r.get("session") == session
                and int(r.get("exchange_index") or -1) == int(exchange_index) and r.get("target_sha") == old_sha):
            aj.append_record(store, {**r, "target_sha": new_sha, "rebound_from": old_sha, "rebound_reason": REASON,
                                     "rebound_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            n += 1
    return n


def apply(sidecar: ChatSidecar, c: dict, v: dict, *, live_chats_dir) -> dict:
    """Write one decision through *sidecar*: the cut target (``repair_target``, the prior one
    under ``text_repairs``) and the record's ``thought_hygiene`` field. Returns the decision."""
    d = decide(c, v)
    new_sha = c["target_sha"]
    if d["outcome"] == "cut":
        if not sidecar.repair_target(c["session"], c["exchange_index"], d["new_target"], reason=REASON):
            return dict(d, outcome="write_failed")
        new_sha = exchange_target_sha(d["new_target"])
        d["capture_rebound"] = rebind_capture(live_chats_dir, c["session"], c["exchange_index"],
                                              c["target_sha"], new_sha)
    if d["outcome"] in ("cut", "none"):
        sidecar.set_record_field(c["session"], c["exchange_index"], REASON, field_for(c, v, d, new_sha))
    return d


def run(*, sidecar: ChatSidecar, judge: Optional[Callable[[dict], dict]], key: Optional[str], prompts_dir,
        live_chats_dir, sessions: Optional[Iterable[str]] = None, limit: int = 0, rejudge: bool = False,
        say: Callable[[str], None] = print, should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Judge the thoughts under *sidecar* with no decision for their current target (at most
    *limit*, newest first) and apply each decision through *sidecar*; verdicts to the store."""
    store = store_path(live_chats_dir)
    tally: Counter = Counter()
    if _POLICY["self_tone"]:
        # the code rule first: the judge then reads what is left, and a target it already
        # decided keeps its decision (re-bound to the trimmed target), so nothing is read twice
        tone = tone_pass(sidecar, live_chats_dir=live_chats_dir, sessions=sessions, say=say, should_stop=should_stop)
        tally.update({f"self_tone:{k}": v for k, v in tone.items()})
    cands = candidates(sidecar, markers=block_markers(prompts_dir), sessions=sessions, rejudge=rejudge)
    if limit > 0:
        cands = cands[:limit]
    if judge is None or not cands:
        tally["waiting"] += len(cands) if judge is None else 0
        return tally
    say(f"{len(cands)} thought(s) to read under {key} "
        f"({sum(c['prefilter']['judge'] for c in cands)} echo the identity)")
    for n, c in enumerate(cands, 1):
        if should_stop is not None and should_stop():
            tally["stopped"] += len(cands) - n + 1
            break
        v = judge(c) if c["prefilter"]["judge"] else skip_record(c, judge_key_=key or "", judge_run="")
        d = apply(sidecar, c, v, live_chats_dir=live_chats_dir)
        aj.append_record(store, {"record": "verdict", **v, "decision": d["outcome"], "cut": d["cut"]})
        tally[d["outcome"]] += 1
        tally["lines_cut"] += len(d["cut"]) if d["outcome"] == "cut" else 0
        say(f"[{n:>4}/{len(cands)}] {d['outcome']:<9} {len(d['cut']):>3}/{len(c['lines']):<3} "
            f"{float(v.get('seconds') or 0):>4.0f} s  {c['session']}#{c['exchange_index']}  {aj.head(v.get('why'), 80)}")
    return tally


# ── the review sheet (the trial) ──────────────────────────────────────────────

# ── the self-tone rule (code, no judge; 2026-10-09) ───────────────────────────
# A thought line that plans her own tone or style is a recitation whatever its wording: the
# judge cuts a line only when it recites the identity text it is shown, so a paraphrase, a
# portrait worded differently that turn, or a plan of her manner that never quoted anything
# passes. This rule reads the form of the sentence, not her vocabulary: a label-led plan of
# her own manner ("Tone: …", "Style: …"), a bare list of descriptors ("X, Y, Z."), a short
# "X and Y tone." sentence. A sentence that mixes such words with content is left.

_TONE_LABEL = re.compile(
    r"(?i)^(?:[\w-]+\s+){0,2}(?:tone|style|voice|register|persona|manner|delivery|vibe|тон|стиль|голос|манера)"
    r"(?:\s*[/&]\s*[\w-]+|\s+and\s+[\w-]+)?(?:\s+of\s+(?:my|the)\s+(?:reply|response|answer|prompt|voice))?"
    r"\s*\**\s*[:：]")
_OTHER_LABEL = re.compile(r"(?i)\b(?:his|their|user'?s?|the user|person'?s|other'?s?)\b")
_TONE_TAIL = re.compile(r"(?i)^(?:[\w-]+,?\s+(?:and\s+)?){1,4}(?:tone|style|register|voice|delivery)[.?!]?$")
_PLAN = re.compile(r"(?i)^(?:keep(?:ing)?|maintain(?:ing)?|stay(?:ing)?|remain(?:ing)?|preserv(?:e|ing)|hold(?:ing)?)\b")
_DESCRIPTOR = re.compile(
    r"(?i)^[\w-]*(?:al|ive|ic|ed|ful|ous|ant|ent|ar|ary|ory|less|ible|able|ish|ing)$|"
    r"^(?:distance|precision|brevity|detachment|restraint|clarity|coldness|warmth|irony|wit|"
    r"dry|precise|terse|cold|calm|warm|soft|sharp|blunt|direct|brief|concise|crisp|clear|quiet|light|"
    r"dense|spare|lean|sober|stark|clean|plain|lucid|grave|wry|deadpan|tender|gentle|firm|cool|"
    r"холодн\w*|точн\w*|сдержанн\w*|аналитичн\w*|отстранённ\w*|отстраненн\w*)$")
_SUBJECT = re.compile(r"(?i)\b(?:i|he|she|they|we|you|user|his|their)\b")
_LEAD = re.compile(r"^[\s*_`#>\-•]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def tone_sentence(sentence: str) -> Optional[str]:
    """Which self-tone form a sentence is (``label`` / ``list`` / ``tail``), or None."""
    body = _LEAD.sub("", sentence).strip()
    if not body:
        return None
    label = _TONE_LABEL.match(body)
    if label and not _OTHER_LABEL.search(label.group(0)):
        return "label"
    core = body.rstrip(".!?").strip()
    if _TONE_TAIL.match(core) and len(core.split()) <= 6:
        return "tail"
    words = [w.strip("*_`\"'()") for w in re.split(r"[\s,]+", core) if w.strip("*_`\"'()")]
    if (_PLAN.match(core) and not _SUBJECT.search(core) and len(words) <= 14
            and sum(1 for w in words[1:] if _DESCRIPTOR.match(w)) >= 2):
        return "plan"
    segs = [x.strip() for x in core.split(",") if x.strip()]
    if len(segs) >= 3 and not _SUBJECT.search(core):
        desc = [len(x.split()) <= 4 and bool(_DESCRIPTOR.match(x.split()[0].strip("*_`\"'"))) for x in segs]
        if (len(core.split()) <= 20 and sum(desc) >= max(2, (2 * len(segs) + 2) // 3)) or all(desc[:3]):
            return "list"
    return None


def self_tone_cut(thought: str) -> tuple:
    """``(thought with the self-tone sentences removed, [(rule, sentence, line)])``. A line left
    empty goes; one that still holds content keeps it. Pure."""
    out_lines, cuts = [], []
    for line in str(thought or "").split("\n"):
        lead = _LEAD.match(line)
        prefix = lead.group(0) if lead else ""
        body = line[len(prefix):]
        sents = [x for x in _SENT.split(body) if x.strip()]
        kept = []
        for sent in sents:
            rule = tone_sentence(sent)
            if rule:
                cuts.append((rule, sent.strip(), line.strip()))
            else:
                kept.append(sent)
        if sents and not kept:
            continue
        out_lines.append(prefix + " ".join(kept) if len(kept) != len(sents) else line)
    return "\n".join(out_lines), cuts


def tone_pass(sidecar: ChatSidecar, *, live_chats_dir, sessions: Optional[Iterable[str]] = None,
              dry_run: bool = False, say: Callable[[str], None] = print,
              should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """The self-tone rule over every training target with a thought (banned and locked records
    skipped, as the judge skips them): the trimmed target replaces the record's (``repair_target``,
    the prior one under ``text_repairs``, reason ``thought_hygiene:self_tone``); the capture
    judge's verdicts are re-bound (it reads the reply, which is unchanged), and so is the judge's
    own decision when it was bound to the prior target. Idempotent: a trimmed thought has nothing
    left to cut. A cut that would leave too little (the judge's own floor) is not applied."""
    tally: Counter = Counter()
    for session in (list(sessions) if sessions is not None else _sessions(sidecar)):
        if should_stop is not None and should_stop():
            tally["stopped"] += 1
            break
        exchanges = sidecar.load(session).get("exchanges")
        if not isinstance(exchanges, dict):
            continue
        for key, rec in exchanges.items():
            if not isinstance(rec, dict) or rec.get("banned") or rec.get("locked") or not str(key).isdigit():
                continue
            target = str(rec.get("target") or "")
            parts = split_target(target)
            if not parts or not parts[1].strip():
                continue
            thought, cuts = self_tone_cut(parts[1])
            if not cuts:
                continue
            kept = [ln.strip() for ln in thought.split("\n") if ln.strip()]
            before = [ln.strip() for ln in parts[1].split("\n") if ln.strip()]
            if len(kept) < MIN_KEEP_LINES or sum(map(len, kept)) < MIN_KEEP_SHARE * (sum(map(len, before)) or 1):
                tally["over_cut"] += 1
                continue
            new_target = f"{parts[0]}{re.sub(chr(10) + '{3,}', chr(10) * 2, thought)}{parts[2]}{parts[3]}"
            tally["cut"] += 1
            tally["sentences"] += len(cuts)
            if dry_run:
                continue
            old_sha = exchange_target_sha(target)
            if not sidecar.repair_target(session, int(key), new_target, reason=TONE_REASON):
                tally["write_failed"] += 1
                continue
            new_sha = exchange_target_sha(new_target)
            tally["capture_rebound"] += rebind_capture(live_chats_dir, session, int(key), old_sha, new_sha)
            done = rec.get(REASON)
            if isinstance(done, dict) and done.get("target_sha") == old_sha:
                sidecar.set_record_field(session, int(key), REASON, {**done, "target_sha": new_sha,
                                                                     "self_tone": len(cuts), "self_tone_from": old_sha})
    if tally["cut"]:
        say(f"thought hygiene, self-tone rule: {tally['cut']} thought(s) trimmed, {tally['sentences']} sentence(s)"
            + (" (dry run)" if dry_run else ""))
    return tally


def render_sheet(items: list) -> str:
    """Markdown: each judged thought with its lines, the cut ones struck (✂), and the verdict."""
    out = ["# Thought hygiene trial — lines marked ✂ would be cut (nothing was changed)", ""]
    for c, v, d in items:
        out.append(f"## {c['session']}#{c['exchange_index']} — {d['outcome']} ({len(d['cut'])}/{len(c['lines'])} lines)")
        out.append(f"*judge:* {v.get('outcome')}; RECITE {v.get('recite') or 'none'}; WHY: {v.get('why') or '—'}")
        out.append("")
        cut = set(d["cut"])
        for n, (_i, ln) in enumerate(c["lines"], 1):
            out.append(f"{'✂' if n in cut else ' '} [{n}] {ln.strip()}")
        out.append("")
    return "\n".join(out)


def summarize(records: Iterable[dict]) -> dict:
    newest: dict = {}
    for r in records:
        if r.get("record") == "verdict":
            newest[(r.get("session"), r.get("exchange_index"), r.get("target_sha"))] = r
    t: Counter = Counter()
    for r in newest.values():
        t[r.get("decision") or r.get("outcome")] += 1
        t["lines_cut"] += len(r.get("cut") or []) if r.get("decision") == "cut" else 0
        t["lines_read"] += int(r.get("n_lines") or 0)
    return dict(t)


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    print("the self-tone rule (form, not vocabulary)")
    th = ("* Tone: Analytical, cerebral, measured, slightly detached.\n"
          "* Context: Analytical, cerebral conversation with Pavel over several hours.\n"
          "* Dry, precise, restrained. He asks about the tide tables.\n"
          "Analytical and measured tone.\n"
          "* His tone: playful, ironic, light.\n"
          "* The user is warm, curious, open.\n"
          "* Plan: answer the tide question first.")
    new, cuts = self_tone_cut(th)
    check("cut: a label, a bare list inside a line, a short tone sentence", [c[0] for c in cuts], ["label", "list", "tail"])
    check("left: a mixed content line, the other person's tone, a list with a subject, a plan",
          new.split("\n"), ["* Context: Analytical, cerebral conversation with Pavel over several hours.",
                             "* He asks about the tide tables.", "* His tone: playful, ironic, light.",
                             "* The user is warm, curious, open.", "* Plan: answer the tide question first."])
    check("Russian label", tone_sentence("Тон: холодный, точный."), "label")
    check("a prefixed label, a compound label, a question tail, a keep-it plan, a long list",
          [tone_sentence(x) for x in ("Refining tone: Keep it dry and brief.", "Tone/Setting: dry, precise, cold.",
                                      "Dry, precise tone?", "Keep it cerebral, measured, and precise.",
                                      "Analytical, cerebral, measured distance, introspective, preference for precision, "
                                      "treats self as a subject for study, avoids clichés, structural reduction.")],
          ["label", "label", "tail", "plan", "list"])
    check("left: the person's style, a plan about content",
          [tone_sentence(x) for x in ("His style: surgical brevity, reductionist.", "Keep the tide question first.",
                                      "Maintaining the thread about Pavel's move to Kestrel.")], [None, None, None])
    check("nothing to cut", self_tone_cut("* Plan: answer.\n* Then ask.")[1], [])

    print("parse")
    check("ranges", parse_answer("RECITE: 1-3, 7\nWHY: lists traits")["lines"], [1, 2, 3, 7])
    check("none", parse_answer("**RECITE:** none\nWHY: all reasoning")["label"], "none")
    check("no line", parse_answer("I cannot help with that.")["label"], None)
    check("last RECITE wins", parse_answer("RECITE: 1\nRECITE: 2, 4")["lines"], [2, 4])

    print("identity and prefilter")
    sysc = ("You are calm and exact. You value quiet precision.\n\nIt is Monday.\n\n"
            "Things you know, looked up because this message seemed to turn on them:\n- a fact")
    ident = identity_text(sysc, _BLOCK_DEFAULTS)
    check("identity cut before the facts block", ident.endswith("It is Monday."), True)
    thought = ("Calm and exact, quiet precision.\nThe person asks about the trip to Halden.\n"
               "Calm and exact, quiet precision.\nI should answer the dates first.")
    lines = thought_lines(thought)
    pf = prefilter(lines, ident)
    check("prefilter sees echo and repetition", (pf["judge"], pf["repeated"]), (True, 1))
    check("plain thought is spared", prefilter(thought_lines("Answer the dates.\nCheck the map."), ident)["judge"],
          False)
    check("recite share", round(recite_share(thought, ident), 2), 0.5)

    print("decide")
    target = f"<think>{thought}</think>\nThe trip is on the 4th."
    c = {"session": "s.json", "exchange_index": 1, "target_sha": exchange_target_sha(target), "target": target,
         "thought": thought, "lines": lines, "identity": ident, "message": "When is the trip?"}
    d = decide(c, {"outcome": "verdict", "recite": [1, 3, 99]})
    check("cut lines that exist", d["cut"], [1, 3])
    check("cut target keeps the answer and the reasoning", d["new_target"],
          "<think>The person asks about the trip to Halden.\nI should answer the dates first.</think>\n"
          "The trip is on the 4th.")
    check("over-cut refused", decide(c, {"outcome": "verdict", "recite": [1, 2, 3]})["outcome"], "over_cut")
    check("refused verdict cuts nothing", decide(c, {"outcome": "refused"})["outcome"], "refused")

    print("apply through a sidecar, with the capture verdict rebound")
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        (chats / "s.json").write_text(json.dumps({"exchanges": [
            {"user_prompt": "hi", "system_content": sysc},
            {"user_prompt": "When is the trip?", "system_content": sysc}]}), encoding="utf-8")
        (chats / "s.state.json").write_text(json.dumps({"exchanges": {"1": {"target": target, "verdict": "keep"},
                                                                       "0": {"target": "no thought", "verdict": "keep"}}}),
                                            encoding="utf-8")
        from core import capture_judge as cj
        aj.append_record(cj.store_path(chats), {"record": "verdict", "session": "s.json", "exchange_index": 1,
                                                "target_sha": c["target_sha"], "judge_key": "k", "response": "holds"})
        side = ChatSidecar(chats)
        cands = candidates(side, markers=_BLOCK_DEFAULTS)
        check("one candidate (the target with a thought)", [(x["session"], x["exchange_index"]) for x in cands],
              [("s.json", 1)])
        tally = run(sidecar=side, judge=lambda cc: {"outcome": "verdict", "recite": [1, 3], "judge_key": "hk",
                                                     "ts": "t", "why": "traits", "seconds": 1.0},
                    key="hk", prompts_dir=td, live_chats_dir=chats, say=lambda s: None)
        check("tally", (tally["cut"], tally["lines_cut"]), (1, 2))
        rec = json.loads((chats / "s.state.json").read_text())["exchanges"]["1"]
        check("target cut, prior kept", (rec["target"].count("Calm and exact"), len(rec["text_repairs"])), (0, 1))
        check("field bound to the new target", rec[REASON]["target_sha"], exchange_target_sha(rec["target"]))
        rebound = [r for r in aj.read_jsonl(cj.store_path(chats)) if r.get("rebound_reason") == REASON]
        check("capture verdict rebound", rebound[0]["target_sha"] if rebound else None, rec[REASON]["target_sha"])
        check("nothing left to judge", candidates(side, markers=_BLOCK_DEFAULTS), [])
        check("transcript untouched", json.loads((chats / "s.json").read_text())["exchanges"][1]["user_prompt"],
              "When is the trip?")
        check("summary", summarize(aj.read_jsonl(store_path(chats)))["cut"], 1)
        check("sheet renders", "✂ [1]" in render_sheet([(c, {"recite": [1]}, {"outcome": "cut", "cut": [1]})]), True)

    print("the self-tone pass through a sidecar: trims, keeps the judge's decision, idempotent")
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        (chats / "t.json").write_text(json.dumps({"exchanges": [{"user_prompt": "Tide at six?"}]}), encoding="utf-8")
        tgt = ("<think>* Tone: dry, precise, restrained.\n* He asks about the tide.\n* Keep it cool and brief.\n"
               "* Answer with the time first.</think>\n\nAt six.")
        sha0 = exchange_target_sha(tgt)
        (chats / "t.state.json").write_text(json.dumps({"exchanges": {"0": {"target": tgt, "verdict": "keep",
                                                       REASON: {"target_sha": sha0, "outcome": "none"}}}}), encoding="utf-8")
        side = ChatSidecar(chats)
        dry = tone_pass(side, live_chats_dir=chats, dry_run=True, say=lambda m: None)
        check("dry run: counted, nothing written", (dry["cut"], json.loads((chats / "t.state.json").read_text())["exchanges"]["0"]["target"] == tgt),
              (1, True))
        t1 = tone_pass(side, live_chats_dir=chats, say=lambda m: None)
        rec = json.loads((chats / "t.state.json").read_text())["exchanges"]["0"]
        check("trimmed: the tone label and the keep-it plan go, the content stays, the reply untouched",
              (t1["sentences"], "Tone:" in rec["target"], "Keep it" in rec["target"], "the tide" in rec["target"],
               rec["target"].endswith("At six."), rec["text_repairs"][-1]["reason"]), (2, False, False, True, True, TONE_REASON))
        check("the judge's decision re-bound, not re-read", (rec[REASON]["target_sha"] == exchange_target_sha(rec["target"]),
              candidates(side, markers=_BLOCK_DEFAULTS)), (True, []))
        check("idempotent", tone_pass(side, live_chats_dir=chats, say=lambda m: None)["cut"], 0)
    print("FAILURES:", failures if failures else "none")
    if failures:
        raise SystemExit(1)


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Thought hygiene (GPU-free parts).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true", help="the store's summary")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    chats = Path(__file__).resolve().parents[2] / "data" / "chats"
    print(json.dumps(summarize(aj.read_jsonl(store_path(chats))), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
