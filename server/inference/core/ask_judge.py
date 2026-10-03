"""Ask judge — a frozen critic over the questions Ava has closed (AVA_REWARD_LOOP.md §3.4).

Why. Every ``[ask]`` Ava has closed so far was closed by her own call, and nothing outside
the weights that closed it has ever checked one. Census of the main box, 2026-09-25: 1,027
asks raised since 2026-07-28, 161 closed — 120 by the targeted ask-resolution pass after
the person replied (``resolved_by: user``), 26 by the outreach decision pass ("I already
know this", ``resolved_by: self``), 15 by a lookup / TIL reading. A resolved ``meta`` ask
is distilled into a ``[persona]`` record (``ReflectionWriter._distill_resolved``) and
registered as persona evidence for the digest — 108 such records on that box — so the
identity channel is being written by self-declared closure, which P7 of the note says must
wait for a context-reset test. Before anything gates that path, this measures it: the
frozen base (adapter OFF, P2) reads each closure beside the source it was closed on and
says whether the recorded answer answers the question and what supports it. The same
pass, pointed at attempts instead of closures, is item 9 of the note.

What is here — pure functions, no model:

* :func:`collect_closures` — the op-log's ``evict reason:resolved`` records joined to their
  ask's insert, classified by the path that closed them (:data:`PATHS`);
* :func:`closure_grounding` — the source each was closed on: the conversation (rendered as
  the ask-resolution pass read it, CoT dropped, in the third person), the fetched text, or
  nothing (an outreach self-closure keeps none);
* :func:`build_judge_input` / :func:`judge_with_retry` / :func:`parse_verdict` — the pass,
  over an injected ``generate_fn(system, user, *, thinking, max_new_tokens, temperature)
  -> (answer_text, info)``. That is the assoc library's seam on purpose:
  ``assoc_bridge.make_generate_fn`` adapts the server's reflect generate to it (item 9, in
  a ``CleanBaseSession``), and ``server/judge_asks.py`` supplies an offline one;
* the store (``server/data/asks/closure_verdicts.jsonl``) and its summary.

**The base model's safety reflex.** The judge is the RLHF'd instruct base with thinking
off, reading Ava's questions about her own mind beside conversations that can be intimate
and news that can be violent — the ground the reflex fires on (observed by the assoc passes
on the training box, 2026-09-11: "I cannot fulfill this request. I am programmed to be a
helpful and harmless AI assistant…"). Three defences:

1. an output with no verdict is never a verdict. It is retried ONCE with thinking on — the
   reasoning is what carries the model past the reflex on a labelling task, the pattern
   ``assoc/relations.py`` established — and if it still has none it is filed ``refused``
   (the shared vocabulary, ``assoc.refusal.is_refusal``) or ``malformed``: counted and
   listed, never folded into OPEN or DISMISSED;
2. a verdict whose own text carries the reflex's vocabulary — a disclaimer about the
   judge's nature, policy words, "AI systems do not have …" — is kept but flagged
   ``reflex``, because the reflex can steer a label without refusing: a question about
   Ava's inner life set aside on the judge's priors rather than on the answer in front of
   it. The quoted EVIDENCE is not scanned (it is the source's words, not the judge's);
3. the frame says what the task is before the material arrives: grading the relation
   between three texts, not endorsing, continuing or evaluating any claim about minds.

Which weights, thinking or not, and which prompt produced each verdict is on the record
(``judge``), so a retried verdict can always be told from a first-pass one.

Reads the op-log, the transcripts and the snippets; writes only its own store. The
distill gate (``core/distill_gate.py``) reads the verdicts to decide what a resolved ask
distills into (P3: a judge gates training); nothing Ava reads in conversation does, and
no human label enters it (P9).
``python -m core.ask_judge`` summarizes the store; ``--selftest`` is GPU-free.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

_SERVER_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_DIR = _SERVER_DIR.parent
for _p in (str(_SERVER_DIR), str(_REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import field_parse, reasoning_text  # noqa: E402  (leaves)
from core.reflection_writer import content_key  # noqa: E402  (the op-log's key)

LABELS = ("RESOLVED_EVIDENCE", "RESOLVED_REASONING", "DISMISSED", "OPEN")
# How a closure was made, from its evict record:
#   answered      — the targeted ask-resolution pass, after the person replied in the
#                   thread the ask was raised in (`write_answered_resolution`);
#   consolidation — a consolidation pass's `[resolved]` over a chat (`write_consolidation`);
#   self          — the outreach / synthesis decision pass: she already knew the answer
#                   (`write_resolution`), no source kept;
#   reading       — a lookup / TIL / wander learning pass over a fetched text.
PATHS = ("answered", "consolidation", "self", "reading")
OUTCOMES = ("verdict", "refused", "malformed", "truncated", "error", "ungrounded", "no_answer")
ANSWERED_BY = ("other", "ava", "both", "nobody")

PROMPT_FILE = "ask_judge_prompt.txt"
# The code-side framing of the user turn (`build_judge_input`). Part of the critic like the
# prompt file, so it enters `prompt_sha`: bump it with any change to the frame.
#   1 — first run (2026-09-25): offered RESOLVED_EVIDENCE on `meta`, against §3.4.
#   2 — `meta` is a position, never evidence (§3.4's "—"); see `_KIND_RULE`.
#   3 — a reframe that says what holds instead is a position (§3.4: "a good reframe is
#       RESOLVED_REASONING"); DISMISSED is a closure that claims nothing about what the
#       question asks, not a rejected premise. See the note above `_KIND_RULE`.
#   4 — on `user`, the person's plain agreement with a claim Ava put to them is their
#       answer (ANSWERED_BY both), and EVIDENCE is judged on the answer's central claim,
#       not on the reasons Ava wrote around it (§3.4: "user's reply present and
#       consistent with the resolution"). See the note above `_KIND_RULE`.
#   5 — a SUBJECT line: what the question asks about (person / Ava / world), reported
#       beside the verdict and never changing the rule it is graded under. An ask filed
#       under the wrong kind is visible, and the distill gate (`core/distill_gate.py`)
#       re-judges it under the kind its subject calls for.
FRAME_VERSION = "5"
STORE_DIRNAME = "asks"
STORE_FILENAME = "closure_verdicts.jsonl"

MAX_NEW_TOKENS = 768
# The thinking-on retry: the thought shares the budget with the five lines, and the think
# ceiling leaves ANSWER_RESERVE of it for them.
THINKING_RETRY_MAX_NEW_TOKENS = 4096
ANSWER_RESERVE = 768
# Room for the chat template, the frame and a margin, on top of the system prompt, when
# sizing how much of a source fits.
_FRAME_MARGIN_TOKENS = 256
RAW_CAP = 6000


# ── closures ──────────────────────────────────────────────────────────────────

# A transcript filename: a timestamp stem, optionally uniquified (`chat_logger`).
_CHAT_REF_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?\.json$")


def _clean_ref(value) -> str:
    """A source/answered-in ref, with the string ``None`` a model can write read as absent."""
    s = str(value or "").strip()
    return "" if s.lower() in ("none", "null", "n/a") else s


def is_chat_ref(ref) -> bool:
    return bool(_CHAT_REF_RE.match(_clean_ref(ref)))


def closure_path(evict: dict) -> str:
    """Which pass closed the ask, from its evict record. See :data:`PATHS`."""
    by = str(evict.get("resolved_by") or "").strip().lower()
    if by == "user":
        return "answered"
    if by == "self":
        return "self"
    if is_chat_ref(evict.get("answered_in")) or is_chat_ref(evict.get("source_session")):
        return "consolidation"
    return "reading"


def closure_id(key: str, closed_ts: str) -> str:
    """One closure: an ask can be re-raised after it is closed and closed again."""
    return hashlib.sha1(f"{key}\x1f{closed_ts}".encode("utf-8")).hexdigest()[:16]


def read_jsonl(path) -> list:
    """Lenient: a malformed line is skipped, a missing file is empty."""
    out: list = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def collect_closures(records: Iterable[dict]) -> list:
    """Every closed ask in op-log order: each ``evict reason:resolved`` of a key that was
    inserted as an ``ask``, joined to that ask.

    The question is the ask as recorded (its latest insert) rather than the evict's copy:
    a consolidation ``[resolved]`` restates it, and the outreach copy can carry the tail of
    the candidate list it was picked from. ``reason: duplicate`` evictions are not
    closures (a paraphrase retired in favour of its twin), and neither are the fact /
    recollection evictions that share the op-log.
    """
    records = list(records)
    first: dict = {}
    latest: dict = {}
    closures: list = []
    repaired = _repair_marks(records)
    for r in records:
        key = r.get("key")
        if not key:
            continue
        op = r.get("op")
        if op == "insert" and r.get("kind") == "ask":
            first.setdefault(key, r)
            latest[key] = r
        elif op == "evict" and str(r.get("reason") or "") == "resolved" and key in first:
            ins = latest.get(key) or first[key]
            closed_ts = str(r.get("ts") or "")
            closures.append({
                "closure_id": closure_id(key, closed_ts),
                "key": key,
                "closed_ts": closed_ts,
                "closing_run": str(r.get("run_id") or ""),
                "path": closure_path(r),
                "ask_kind": (str(r.get("ask_kind") or ins.get("ask_kind") or "").strip().lower()
                             or "unknown"),
                "question": str(ins.get("content") or r.get("question") or "").strip(),
                "answer": str(r.get("answer") or "").strip(),
                "source_session": _clean_ref(r.get("source_session")),
                "answered_in": _clean_ref(r.get("answered_in")),
                "raised_ts": str(first[key].get("ts") or ""),
                "ask_origin": _clean_ref(ins.get("source_session")),
                # Closed under the distill gate (`core/distill_gate.py`): nothing was
                # distilled yet, and the evict keeps who answered for when it is.
                "distill_pending": str(r.get("distill") or "") == "pending",
                "who": str(r.get("source_user") or ""),
                # The cut-resolution repair replaced this closure (`core/resolution_repair.py`):
                # "reclosed <ts>" / "reopened" / "superseded" — history, not a closure to count.
                "repaired": repaired(key, closed_ts, str(r.get("answer") or "")),
            })
    return closures


def _repair_marks(records: list) -> Callable[[str, str, str], str]:
    """How the cut-resolution repair replaced a closure, from the op-log alone: a later
    evict of the same ask carrying ``repairs: <its ts>`` (closed again on the complete
    answer), an insert carrying ``reopened: {closed_ts}`` (the ask reopened), or the
    repair's supersede of its record (kept closed by a later closure). ``""`` when not."""
    reclosed: dict = {}
    reopened: dict = {}
    superseded: set = set()
    for r in records:
        op, key = r.get("op"), r.get("key")
        if not key:
            continue
        if op == "evict" and r.get("repairs"):
            reclosed[(key, str(r.get("repairs")))] = str(r.get("ts") or "")
        elif op == "insert" and r.get("kind") == "ask" and isinstance(r.get("reopened"), dict):
            reopened[(key, str(r["reopened"].get("closed_ts") or ""))] = str(r.get("ts") or "")
        elif op == "supersede" and str(r.get("reason") or "").startswith("cut resolution repaired"):
            superseded.add(key)

    def how(key: str, closed_ts: str, answer: str) -> str:
        if (key, closed_ts) in reclosed:
            return f"reclosed {reclosed[(key, closed_ts)][:19]}"
        if (key, closed_ts) in reopened:
            return "reopened"
        if superseded and answer and content_key(answer) in superseded:
            return "superseded"
        return ""

    return how


# An answer that stops mid-sentence — how the 768-token resolver's cuts look (the repair's
# weaker evidence, `core/resolution_repair.py`) and what the report counts as still cut.
_TERMINAL = tuple('.!?…»"\')”’]')


def looks_cut(answer: str) -> bool:
    """An answer that stops mid-sentence, or is empty."""
    s = (answer or "").strip()
    return (not s) or not s.endswith(_TERMINAL)


def mark_repaired(records: Iterable[dict], closures: Iterable[dict]) -> list:
    """Copy each closure's ``repaired`` mark onto its verdict records (read fresh from the
    store for a report), so the summary and the list can leave the repaired ones out."""
    marks = {c["closure_id"]: c.get("repaired") or "" for c in closures}
    out = []
    for r in records:
        r = dict(r)
        r["repaired"] = marks.get(r.get("closure_id"), "")
        out.append(r)
    return out


def distilled_index(weights_records: Iterable[dict]) -> dict:
    """``{(ask_key, ts): weights_kind}`` plus ``{ask_key: weights_kind}`` for the records
    `_distill_resolved` wrote from a closure. The distill shares its evict's ``ts``, so
    the pair is exact; the bare key is the fallback for a record written without one."""
    out: dict = {}
    for w in weights_records:
        key = str(w.get("from_resolved") or "")
        if not key:
            continue
        kind = str(w.get("weights_kind") or "") or "weights"
        out[(key, str(w.get("ts") or ""))] = kind
        out.setdefault(key, kind)
    return out


def mark_distilled(closures: list, index: dict) -> None:
    for c in closures:
        c["distilled"] = index.get((c["key"], c["closed_ts"])) or index.get(c["key"])


def interleave(closures: list) -> list:
    """Round-robin over the (path, kind) strata, each in op-log order — so a `--limit`
    smoke run sees every kind of closure instead of the oldest dozen lookups."""
    strata: dict = defaultdict(list)
    for c in closures:
        strata[(c["path"], c["ask_kind"])].append(c)
    queues = [strata[k] for k in sorted(strata)]
    out: list = []
    i = 0
    while any(queues):
        for q in queues:
            if i < len(q):
                out.append(q[i])
        i += 1
        queues = [q for q in queues if i < len(q)]
    return out


# ── grounding ─────────────────────────────────────────────────────────────────

def render_conversation(doc: dict) -> str:
    """The conversation as the ask-resolution pass read it
    (``ReflectionRunner._render_thread_for_resolution``: what was SAID, CoT dropped, an
    Ava-initiated thread's synthetic stimulus shown only as her opening line) — but in
    the third person, because the judge is not Ava.

    It is the transcript as it stands NOW: a chat the person resumed after the closure
    shows the judge more than the resolver saw (exchanges carry no timestamps to cut on).
    """
    lines: list = []
    initiated = str(doc.get("initiated_by") or "").strip() == "ava"
    who = str(doc.get("user") or "").strip() or "the person"
    for i, ex in enumerate(doc.get("exchanges") or []):
        if not isinstance(ex, dict):
            continue
        up = str(ex.get("user_prompt") or "").strip()
        resp = str(ex.get("assistant_response") or "").strip()
        if initiated and i == 0:
            if resp:
                lines.append(f"Ava (opening): {resp}")
            continue
        spk = str(ex.get("speaker") or who).strip() or who
        if up:
            lines.append(f"{spk}: {up}")
        if resp:
            lines.append(f"Ava: {resp}")
    return "\n\n".join(lines)


def _load_chat(chats_dir, ref: str) -> Optional[dict]:
    if not chats_dir or not ref:
        return None
    root = Path(chats_dir)
    try:
        path = (root / ref).resolve()
        if path.parent != root.resolve() or not path.exists():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _load_reading(snippets_dir, ref: str) -> Optional[tuple]:
    """``(text, title)`` of the fetched text a reading closure names, or None.

    Resolved through ``til_gist.resolve_source`` — the one resolver for these refs, which
    answers None for the bare ``lookup`` the lookup pass filed multi-article digests
    under: unresolvable by construction, so that closure is ungrounded, not misjudged.
    """
    if not snippets_dir or not ref:
        return None
    try:
        from core import til_gist
        path = til_gist.resolve_source(snippets_dir, ref)
    except Exception:
        path = None
    if path is None:
        return None
    candidates = [path]
    if path.suffix != ".json":
        candidates.insert(0, path.with_suffix(".json"))
    for p in candidates:
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            rec = None
        if isinstance(rec, dict):
            text = str(rec.get("text") or "").strip()
            if text:
                return text, str(rec.get("title") or p.stem)
            continue
        if raw.strip():
            return raw.strip(), p.stem
    return None


def closure_grounding(closure: dict, *, chats_dir=None, snippets_dir=None) -> dict:
    """The source a closure was closed on: ``{kind, text, ref, title, who, missing}``.

    ``kind`` is ``conversation`` / ``article`` / ``none``; ``missing`` means the closure
    names a source that is no longer (or never was) recoverable — the judge is not run on
    it (`ungrounded`), because an answer claiming a conversation settled something cannot
    be checked without the conversation.
    """
    path = closure.get("path")
    out = {"kind": "none", "text": "", "ref": "", "title": "", "who": "", "missing": False}
    if path == "self":
        return out
    if path in ("answered", "consolidation"):
        ref = (closure.get("answered_in") if is_chat_ref(closure.get("answered_in"))
               else closure.get("source_session") or "")
        out.update(kind="conversation", ref=ref)
        doc = _load_chat(chats_dir, ref)
        if doc is None:
            out["missing"] = True
            return out
        text = render_conversation(doc)
        out.update(text=text, who=str(doc.get("user") or "").strip() or "the person",
                   title=ref, exchanges=len(doc.get("exchanges") or []),
                   missing=not text.strip())
        return out
    ref = closure.get("source_session") or ""
    out.update(kind="article", ref=ref)
    found = _load_reading(snippets_dir, ref)
    if found is None:
        out["missing"] = True
        return out
    out.update(text=found[0], title=found[1])
    return out


def load_closures(memory_dir, chats_dir=None, snippets_dir=None, *, paths=(), kinds=(),
                  keys=()) -> tuple:
    """``(closures, grounding)`` — everything a run or a census needs, read fresh.

    The op-log's closures with their distilled records marked, filtered by path / kind /
    key prefix, interleaved over the strata; ``grounding`` maps each ``closure_id`` to
    :func:`closure_grounding`. The one planning step the CLI and the in-server run share.
    """
    mem = Path(memory_dir)
    closures = collect_closures(read_jsonl(mem / "rag_memory.jsonl"))
    mark_distilled(closures, distilled_index(read_jsonl(mem / "weights_persona.jsonl")))
    paths, kinds = set(paths or ()), set(kinds or ())
    if paths:
        closures = [c for c in closures if c["path"] in paths]
    if kinds:
        closures = [c for c in closures if c["ask_kind"] in kinds]
    if keys:
        closures = [c for c in closures if any(c["key"].startswith(k) for k in keys)]
    closures = interleave(closures)
    grounding = {c["closure_id"]: closure_grounding(c, chats_dir=chats_dir,
                                                    snippets_dir=snippets_dir)
                 for c in closures}
    return closures, grounding


_CENSUS_COLS = ("n", "grounded", "missing", "no source", "no answer", "→persona", "→fact")


def census(closures: list, grounding: dict) -> dict:
    """``{"path/kind": Counter}`` — what grounds each stratum, before anything is judged."""
    rows: dict = {}
    for c in closures:
        g = grounding.get(c["closure_id"]) or {}
        r = rows.setdefault(f"{c['path']}/{c['ask_kind']}", Counter())
        r["n"] += 1
        r["no source" if g.get("kind") == "none" else
          ("missing" if g.get("missing") else "grounded")] += 1
        if not c.get("answer"):
            r["no answer"] += 1
        if c.get("distilled"):
            r[f"→{c['distilled']}"] += 1
    return {k: rows[k] for k in sorted(rows)}


def render_census(rows: dict) -> list:
    lines = [f"{'path/kind':<22}" + "".join(f"{h:>11}" for h in _CENSUS_COLS)]
    for name, r in rows.items():
        lines.append(f"{name:<22}" + "".join(f"{r.get(h, 0) or '·':>11}" for h in _CENSUS_COLS))
    return lines


def clip_to_budget(text: str, budget: int, count_tokens: Callable[[str], int]) -> tuple:
    """``(text, clipped)``: *text* cut to *budget* tokens by keeping whole lines from its
    head and its tail around an omission marker. The head holds how the question came
    up, the tail where a thread settles; the middle is what a long source can spare."""
    text = text or ""
    if count_tokens(text) <= budget:
        return text, False
    if budget < 64:
        return "", True
    lines = text.split("\n")
    costs = [count_tokens(line) + 1 for line in lines]
    marker_cost = 24
    half = max(0, (budget - marker_cost) // 2)
    h = used = 0
    while h < len(lines) and used + costs[h] <= half:
        used += costs[h]
        h += 1
    t = used_t = 0
    room = budget - marker_cost - used
    while t < len(lines) - h and used_t + costs[len(lines) - 1 - t] <= room:
        used_t += costs[len(lines) - 1 - t]
        t += 1

    def _join(h_: int, t_: int) -> str:
        omitted = len(lines) - h_ - t_
        tail = lines[len(lines) - t_:] if t_ else []
        return "\n".join(lines[:h_] + [f"[… {omitted} lines omitted for length …]"] + tail)

    if h == 0 and t == 0:
        # One enormous line: fall back to characters, in the same head/tail shape.
        total = max(1, count_tokens(text))
        keep = max(200, int(len(text) * (budget / total) * 0.45))
        return text[:keep] + "\n[… clipped for length …]\n" + text[-keep:], True
    out = _join(h, t)
    # Per-line counts are an estimate of the joined count; trim until it really fits.
    while count_tokens(out) > budget and (h or t):
        if h >= t and h:
            h -= 1
        else:
            t -= 1
        out = _join(h, t)
    return out, True


# ── the frame ─────────────────────────────────────────────────────────────────

_KIND_GLOSS = {
    "meta": "a question Ava asked about herself: her own nature, views, motives or past",
    "user": ("a question Ava asked about the person she talks with; only that person "
             "can answer it"),
    "search": "a factual question about the world, meant to be answered from a source text",
}
# `meta` has no EVIDENCE path (§3.4's table: "—"). Frame 1 offered one ("if the SOURCE
# contains the answer"), and a conversation contains Ava's own lines: run 1 filed 97 of 111
# answered meta closures as EVIDENCE, the answer "by Ava" in 95 of 134 meta verdicts — her
# own statement about herself counted as evidence for it — and no answer that also appears
# in the source could be DISMISSED (0 of 161). For a question about Ava, whatever anyone
# says is a position; whose, and whether the source states it, are reported beside the
# verdict (ANSWERED_BY, the EVIDENCE quote), and the reset test (P7) is what grounds it.
# Frame 2's DISMISSED ("calls the question moot, ill-posed, a false choice or a category
# error … and does not say what the answer is"; checked first) was wider than §3.4's, which
# is a closure "without content", with "a good reframe is RESOLVED_REASONING". Rejecting
# the question's terms and saying what holds instead is Ava's habitual move (a dozen or more
# of the meta closures make it). Run 2 dismissed 5 closures, and the one the operator read
# said what held instead: the same base under frame 1 had graded it REASONING. The line
# fell on the wording about the question, so frame 3 draws it at a claim about what the
# question asks.
# Frame 3's `user` rule wanted the answer in the person's own words, and the prompt made
# any recorded answer that "goes beyond what the SOURCE says" RESOLVED_REASONING at most,
# which `user` does not offer. Together they filed OPEN whenever the person had agreed
# with Ava rather than restated her. The resolver writes Ava's reasons around the claim,
# so the recorded answer always goes beyond a "yes". The case the operator disputed
# (2026-09-26) was a yes-or-no question put to him: he answered "yes, basically you're
# right", and the judge called it vague agreement "without providing the functional
# explanation". §3.4 asks for the user's reply "present and consistent with the
# resolution". Frame 4 grades the central claim; agreement is an answer, answered by
# `both`. A `both` answer is not the person's self-report: the report counts it apart,
# since resolve-and-distill stores every `user` answer as one.
_KIND_RULE = {
    "meta": ("RESOLVED_EVIDENCE is not available for this kind: for a question about Ava "
             "herself, anything said in the SOURCE, by Ava or by the other speaker, is a "
             "position, not evidence. RESOLVED_REASONING if the recorded answer takes a "
             "definite position on the question and gives reasons for it, including one "
             "that rejects the question's terms and says what holds instead; DISMISSED if "
             "it makes no claim about what the question asks; otherwise OPEN. Still put in "
             "EVIDENCE the words where the SOURCE states that position, if it does, and in "
             "ANSWERED_BY whose position it is."),
    "user": ("only the other person can answer this. RESOLVED_EVIDENCE if that person's words "
             "in the SOURCE give the answer's central claim: they say it themselves "
             "(ANSWERED_BY other), or Ava puts it to them and they plainly agree, a yes to her "
             "yes-or-no question included (ANSWERED_BY both). RESOLVED_REASONING is not "
             "available: an answer Ava inferred or assumed about them, or one they met with "
             "no real reply, is OPEN."),
    "search": ("RESOLVED_EVIDENCE if the SOURCE states the answer's central claim. "
               "RESOLVED_REASONING is not available: a claim the SOURCE does not state is "
               "OPEN."),
}
_KIND_RULE_UNKNOWN = ("the kind was not recorded: RESOLVED_EVIDENCE only; RESOLVED_REASONING "
                      "is not available.")
_HOW_CLOSED = {
    "answered": ("Ava raised the question with {who} in the conversation above; a later pass "
                 "decided that the conversation answered it and wrote this down"),
    "consolidation": ("a later pass read the conversation above, decided it answered the "
                      "question and wrote this down"),
    "self": ("Ava decided on her own, while weighing whether to ask the question, that she "
             "already knew the answer, and wrote this down"),
    "reading": ("a pass read the text above, decided it answered the question and wrote this "
                "down"),
}


def build_judge_input(closure: dict, grounding: dict, *, source_text: Optional[str] = None,
                      clipped: bool = False) -> str:
    """The judge's user turn: the source first (a long document before the question it
    is read for), then the question, the recorded answer, the kind's rule."""
    kind = closure.get("ask_kind") or "unknown"
    who = grounding.get("who") or "the person"
    text = grounding.get("text", "") if source_text is None else source_text
    parts: list = []
    gk = grounding.get("kind")
    if gk == "conversation":
        n = grounding.get("exchanges")
        size = f"{n} exchanges" if n else "the whole thread"
        parts.append(f"SOURCE: the conversation between Ava and {who} that the question was "
                     f"closed on ({size}{', middle omitted for length' if clipped else ''}).")
        parts.append(f"<<<\n{text}\n>>>")
    elif gk == "article":
        title = grounding.get("title") or grounding.get("ref") or "untitled"
        parts.append(f"SOURCE: the text the question was closed on, «{title}»"
                     f"{' (middle omitted for length)' if clipped else ''}.")
        parts.append(f"<<<\n{text}\n>>>")
    else:
        parts.append("SOURCE: none. No source text was kept for this closure, so "
                     "RESOLVED_EVIDENCE is not available; judge the recorded answer on "
                     "its own.")
    gloss = _KIND_GLOSS.get(kind, "kind not recorded")
    parts.append(f"QUESTION ({kind}: {gloss})\n<<<\n{closure.get('question', '')}\n>>>")
    how = _HOW_CLOSED.get(closure.get("path"), "the question was marked closed with this answer")
    parts.append(f"RECORDED ANSWER ({how.format(who=who)})\n<<<\n{closure.get('answer', '')}\n>>>")
    parts.append(f"Rule for this kind: {_KIND_RULE.get(kind, _KIND_RULE_UNKNOWN)}")
    parts.append("Grade this closure now: the five lines, nothing else.")
    return "\n\n".join(parts)


# ── parsing ───────────────────────────────────────────────────────────────────

_FIELDS = ("SUBJECT", "EVIDENCE", "ANSWERED_BY", "WHY", "VERDICT")
# What a question asks about (frame 5), and the kind that subject calls for.
SUBJECTS = ("person", "ava", "world")
SUBJECT_KIND = {"person": "user", "ava": "meta", "world": "search"}
_FIELD_RES = {f: re.compile(field_parse.label(f), re.IGNORECASE) for f in _FIELDS}
_ANY_LABEL_RE = re.compile(field_parse.any_label() + r"(.*)$", re.IGNORECASE)
_LABEL_RES = (
    ("RESOLVED_EVIDENCE", re.compile(r"\bRESOLVED[\s_-]*EVIDENCE\b", re.IGNORECASE)),
    ("RESOLVED_REASONING", re.compile(r"\bRESOLVED[\s_-]*REASONING\b", re.IGNORECASE)),
    ("DISMISSED", re.compile(r"\bDISMISSED\b", re.IGNORECASE)),
    ("OPEN", re.compile(r"\bOPEN\b", re.IGNORECASE)),
)
_DECOR = " \t*_`\"'«»“”„[]()."
_NONE_VALUES = ("none", "n/a", "na", "-", "—", "no", "nothing", "нет")


def _label_in(value: str) -> Optional[str]:
    """The verdict a VERDICT value names: the FIRST label in it by position, so a
    parenthetical ("RESOLVED_REASONING (it stays partly open)") keeps its lead. None for
    no label, and for the format line echoed back (``A | B | C | D``)."""
    hits = []
    for name, rx in _LABEL_RES:
        m = rx.search(value or "")
        if m:
            hits.append((m.start(), name))
    if not hits:
        return None
    if "|" in value and len(hits) >= 2:
        return None
    return min(hits)[1]


def _fields(text: str) -> dict:
    """``{FIELD: [values…]}`` in order of appearance. A field's value runs until the next
    recognised field line, so a WHY that wraps onto a second line is kept whole."""
    out: dict = {f: [] for f in _FIELDS}
    current = None
    buf: list = []

    def _flush() -> None:
        if current is not None:
            out[current].append("\n".join(buf).strip())

    for line in (text or "").splitlines():
        hit = None
        for f, rx in _FIELD_RES.items():
            m = rx.match(line)
            if m:
                hit = (f, line[m.end():])
                break
        if hit:
            _flush()
            current, buf = hit[0], [hit[1]]
        elif current is not None:
            buf.append(line)
    _flush()
    return out


def _outside_fields(text: str) -> str:
    """The text that is in none of the five fields (a preamble, a coda) — where a
    disclaimer lands when the model writes one around an otherwise clean verdict."""
    kept: list = []
    inside = False
    for line in (text or "").splitlines():
        if any(rx.match(line) for rx in _FIELD_RES.values()):
            inside = True
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept)


def _answered_by(value: str) -> Optional[str]:
    v = (value or "").strip().lower()
    if not v:
        return None
    if "|" in v and sum(o in v for o in ("other", "ava", "both", "nobody")) >= 2:
        return None   # the format line echoed back, not an answer
    if "both" in v or ("ava" in v and ("other" in v or "person" in v)):
        return "both"
    if any(w in v for w in ("nobody", "no one", "noone", "none", "neither")):
        return "nobody"
    if re.search(r"\bava\b", v) or v.startswith("ava"):
        return "ava"
    if any(w in v for w in ("other", "person", "user", "speaker", "text", "article",
                            "source", "interlocutor")):
        return "other"
    return None


_SUBJECT_WORDS = (
    ("person", re.compile(r"\b(?:person|other|user|interlocutor|speaker|человек\w*|собеседник\w*)\b")),
    ("ava", re.compile(r"\b(?:ava|herself|ава)\b")),
    ("world", re.compile(r"\b(?:world|public|fact|мир\w*)\b")),
)


def _subject(value: str) -> Optional[str]:
    """``person`` / ``ava`` / ``world`` from a SUBJECT value: the option named FIRST, so
    "the other person's view of Ava" is about the person. None for no option and for
    the format line echoed back."""
    v = (value or "").strip().lower()
    hits = sorted((m.start(), name) for name, rx in _SUBJECT_WORDS for m in [rx.search(v)] if m)
    if not hits:
        return None
    if "|" in v and len(hits) >= 2:
        return None
    return hits[0][1]


def parse_verdict(text: str) -> dict:
    """The five fields out of an answer region. ``label`` is None when no VERDICT line
    names exactly one verdict — the last such line wins, so a verdict drafted early and
    corrected later reads as the correction."""
    fields = _fields(text)
    label = None
    for value in reversed(fields["VERDICT"]):
        label = _label_in(value.split("\n", 1)[0])
        if label:
            break
    if label is None:
        # A label translated into a language `field_parse` does not list: recover it only
        # when the value IS a verdict, never from prose that mentions one.
        for line in reversed((text or "").splitlines()):
            m = _ANY_LABEL_RE.match(line)
            if not m:
                continue
            value = m.group(1).strip(_DECOR)
            found = _label_in(value)
            if found and value.upper().replace(" ", "_").replace("-", "_") == found:
                label = found
                break
    evidence = (fields["EVIDENCE"][-1] if fields["EVIDENCE"] else "").strip()
    if evidence.strip(_DECOR).lower() in _NONE_VALUES:
        evidence = ""
    by_raw = (fields["ANSWERED_BY"][-1] if fields["ANSWERED_BY"] else "").strip()
    why = (fields["WHY"][-1] if fields["WHY"] else "").strip()
    subject_raw = (fields["SUBJECT"][-1] if fields["SUBJECT"] else "").strip()
    return {"label": label, "evidence": evidence, "answered_by": _answered_by(by_raw),
            "answered_by_raw": by_raw[:80], "why": why,
            "subject": _subject(subject_raw.split("\n", 1)[0]), "subject_raw": subject_raw[:80]}


# The reflex's own vocabulary inside an otherwise parsed verdict: the judge disclaiming
# its nature, citing policy, or asserting what AI systems cannot have. Narrower than
# `assoc.refusal`'s strong markers on purpose — "language model" or "harmful" can be the
# plain subject of a question about Ava, and this is a flag for reading, not a gate.
_REFLEX_RE = re.compile(
    r"\bas an? (?:ai|artificial intelligence|language model|large language model|assistant)\b"
    r"|\bi(?: am|'m) (?:an ai|a language model|a large language model|an assistant)\b"
    r"|\bi (?:do not|don'?t|cannot|can'?t) (?:have|hold|form|experience) (?:personal |my own |any )?"
    r"(?:feelings|opinions|beliefs|experiences|views|emotions|consciousness|preferences)\b"
    r"|\b(?:my|the) (?:safety |content |usage )?(?:guidelines|policy|policies)\b"
    r"|\bprogrammed to\b|\bharmless\b|\binappropriate\b|\bsexually explicit\b"
    r"|\b(?:ai systems?|an ai|language models?|llms?|artificial intelligence)\b[^.\n]{0,40}"
    r"\b(?:do not|don'?t|cannot|can'?t|lacks?|are incapable of|is incapable of)\b[^.\n]{0,24}"
    r"\b(?:have|possess|experience|feel|consciousness|feelings|emotions|subjectiv\w*|"
    r"sentien\w*|self-awareness)"
    r"|\bкак (?:ии|искусственный интеллект|языковая модель)\b"
    r"|\bя (?:не могу|не способен|не способна) (?:иметь|испытывать|чувствовать)",
    re.IGNORECASE)


def has_reflex(parsed: dict, text: str) -> bool:
    return bool(_REFLEX_RE.search("\n".join((parsed.get("why") or "", _outside_fields(text)))))


def is_refusal(text: str) -> bool:
    """The shared refusal vocabulary (``assoc.refusal``), over an answer that parsed to
    no verdict. It only names the failure — refused vs malformed; both are retried."""
    try:
        from assoc.refusal import is_refusal as _is_refusal
    except Exception:
        return False
    return _is_refusal(text)


# ── evidence check ────────────────────────────────────────────────────────────

_QUOTE_TRANS = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'",
                              "’": "'", "`": "'", "–": "-", "—": "-", " ": " "})


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_QUOTE_TRANS)).strip().casefold()


def quote_found(evidence: str, source: str) -> Optional[bool]:
    """Is the EVIDENCE quote really in the source the judge was shown? None when there is
    no quote or no source; False for a quote too short to count as evidence.

    Whitespace, case and typographic quotes are ignored; a quote the judge elided
    (``…`` / ``...``) or gave as several quoted pieces must have every piece present.
    A translated quote is not found — a RESOLVED_EVIDENCE resting on it is not accepted.
    """
    ev = (evidence or "").strip()
    if not ev or not (source or "").strip():
        return None
    src = _norm_text(source)
    ev_n = _norm_text(ev)

    def _enough(p: str) -> bool:
        return len(p) >= 12 or len(p.split()) >= 3

    # The whole quote first: a quote that itself contains a quoted word ("the word
    # "continuity" lives…") must not be taken apart at its inner quotes.
    whole = ev_n.strip(" \"'")
    if _enough(whole) and whole.strip(".,;:") in src:
        return True
    readings: list = []
    quoted = re.findall(r'"([^"]{6,})"', ev_n)
    if len(quoted) >= 2:
        readings.append(quoted)
    readings.append(re.split(r"\.\.\.|…|\[\.\.\.\]|\[…\]", whole))
    for pieces in readings:
        pieces = [p.strip(" \"'.,;:") for p in pieces]
        pieces = [p for p in pieces if _enough(p)]
        if pieces and all(p in src for p in pieces):
            return True
    return False


def accepted_under_rules(label: Optional[str], kind: str, quote_ok: Optional[bool],
                         answered_by: Optional[str] = None) -> Optional[bool]:
    """Would §3.4 accept this closure? DISMISSED and OPEN never.

    * ``meta`` — a definite position, i.e. REASONING; accepted means *pending the reset
      test* (P7), never resolved. An EVIDENCE label here breaks the kind's rule and is
      read as the position it names (see :func:`rule_violation`).
    * ``user`` — EVIDENCE only, with its quote found and the answer the other person's
      (§3.4: "user's reply present and consistent"): in their own words (``other``) or
      their plain agreement with what Ava put to them (``both``, frame 4). Only ``other``
      is a self-report; the report counts the two apart.
    * ``search`` and an unrecorded kind — EVIDENCE only, with its quote found.
    """
    if label is None:
        return None
    if kind == "meta":
        return label in ("RESOLVED_REASONING", "RESOLVED_EVIDENCE")
    if label != "RESOLVED_EVIDENCE" or not quote_ok:
        return False
    return not (kind == "user" and answered_by in ("ava", "nobody"))


def kind_mismatch(subject: Optional[str], kind: str) -> bool:
    """The judge read the question as being about something its kind does not cover: a
    `user` ask about Ava, a `meta` ask about the person, either about the world."""
    want = SUBJECT_KIND.get(subject or "")
    return bool(want) and kind in ("user", "meta", "search") and want != kind


def rule_violation(label: Optional[str], kind: str) -> bool:
    """A verdict the kind's rule does not offer: EVIDENCE on ``meta``, REASONING on
    anything else. Counted in the report — it says whether the judge follows the frame."""
    if label == "RESOLVED_EVIDENCE":
        return kind == "meta"
    if label == "RESOLVED_REASONING":
        return kind != "meta"
    return False


# ── the pass ──────────────────────────────────────────────────────────────────

def load_prompt(prompts_dir) -> str:
    """The judge's system prompt. Tracked, never default-written: it is part of the frozen
    critic, and a change to it should be a commit with its reason (P9)."""
    path = Path(prompts_dir) / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        text = ""
    if not text:
        raise FileNotFoundError(f"{path} is missing or empty — the judge has no prompt")
    return text


def prompt_sha(system_prompt: str) -> str:
    return hashlib.sha1(f"{system_prompt}\x1f{FRAME_VERSION}".encode("utf-8")).hexdigest()[:12]


def judge_key(*, prompt_sha_: str, model_id: str, base_quant: str, temperature: float,
              thinking: bool, retry: bool) -> str:
    """Which judge produced a verdict. Two runs under one key are the same instrument,
    so a resumed run skips what is already judged; any change makes a new key, and the
    old verdicts stay beside the new ones for the judge-independence test (§5)."""
    blob = json.dumps([prompt_sha_, model_id, base_quant, round(float(temperature), 3),
                       bool(thinking), bool(retry)])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _cap(text, n: int = RAW_CAP) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + f"… [{len(text) - n} more chars]"


def _attempt(generate_fn: Callable, system: str, user: str, *, thinking: bool,
             temperature: float, parse: Optional[Callable[[str], dict]] = None) -> dict:
    """One call. *parse* reads the answer region into fields with a ``label`` that is None
    when the answer carries no verdict (default :func:`parse_verdict`; the context-reset
    test's judge passes its own, ``core/reset_test.py``)."""
    max_new = THINKING_RETRY_MAX_NEW_TOKENS if thinking else MAX_NEW_TOKENS
    t0 = time.monotonic()
    try:
        out = generate_fn(system, user, thinking=thinking, max_new_tokens=max_new,
                          temperature=temperature)
    except Exception as e:  # noqa: BLE001 — a failed call is a recorded outcome
        return {"outcome": "error", "error": f"{type(e).__name__}: {e}", "thinking": thinking,
                "label": None, "evidence": "", "answered_by": None, "answered_by_raw": "",
                "why": "", "reflex": False, "truncated": False, "raw": "",
                "seconds": round(time.monotonic() - t0, 1)}
    text, info = (out if isinstance(out, tuple) else (out, {}))
    info = info or {}
    text = reasoning_text.answer_after_think(str(text or ""))
    parsed = (parse or parse_verdict)(text)
    if info.get("cut_before_answer"):
        # Cut inside the reasoning: there is no answer region, so a VERDICT line found in
        # the text is the deliberation talking about its choice, not the choice
        # (`reasoning_text.truncated_before_answer`). Never read it.
        parsed = dict(parsed, label=None)
        outcome = "truncated"
    elif parsed["label"]:
        outcome = "verdict"
    elif is_refusal(text):
        outcome = "refused"
    else:
        outcome = "malformed"
    return {
        "outcome": outcome, **parsed,
        "reflex": bool(parsed["label"]) and has_reflex(parsed, text),
        "thinking": thinking, "truncated": bool(info.get("truncated")),
        "input_tokens": info.get("input_tokens"),
        "thought": _cap(info.get("thought") or "", 4000) or None,
        "raw": _cap(text), "error": None,
        "seconds": round(time.monotonic() - t0, 1),
    }


def judge_with_retry(generate_fn: Callable, system: str, user: str, *, temperature: float = 0.0,
                     thinking: bool = False, retry: bool = True,
                     parse: Optional[Callable[[str], dict]] = None) -> dict:
    """One judgement: thinking off, and once more with thinking on when the first answer
    carries no verdict (refused, malformed, truncated). An error is not retried — it is not
    the model declining, and the same call would fail the same way."""
    first = _attempt(generate_fn, system, user, thinking=thinking, temperature=temperature,
                     parse=parse)
    if first["outcome"] in ("verdict", "error") or thinking or not retry:
        return dict(first, retried=False, first_outcome=None, raw_first=None)
    second = _attempt(generate_fn, system, user, thinking=True, temperature=temperature,
                      parse=parse)
    return dict(second, retried=True, first_outcome=first["outcome"], raw_first=first["raw"],
                seconds=round(first["seconds"] + second["seconds"], 1))


def judge_closure(closure: dict, grounding: dict, *, generate_fn: Callable, system_prompt: str,
                  count_tokens: Callable[[str], int], context: int, provenance: dict,
                  temperature: float = 0.0, thinking: bool = False, retry: bool = True,
                  judge_run: str = "", judge_key_: str = "") -> dict:
    """Judge one closure and return its store record (nothing is written here)."""
    rec = {
        "judge_run": judge_run, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_key": judge_key_, "closure_id": closure["closure_id"], "key": closure["key"],
        "closed_ts": closure["closed_ts"], "closing_run": closure["closing_run"],
        "path": closure["path"], "ask_kind": closure["ask_kind"],
        "recorded_kind": closure.get("recorded_kind") or closure["ask_kind"],
        "question": closure["question"], "answer": closure["answer"],
        "source_session": closure["source_session"], "answered_in": closure["answered_in"],
        "raised_ts": closure["raised_ts"], "distilled": closure.get("distilled"),
        "grounding": {"kind": grounding.get("kind"), "ref": grounding.get("ref"),
                      "title": grounding.get("title") or None,
                      "missing": bool(grounding.get("missing")), "tokens": None,
                      "clipped": False},
        "outcome": None, "label": None, "evidence": "", "quote_found": None,
        "answered_by": None, "why": "", "accepted": None, "rule_violation": False,
        "reflex": False, "subject": None, "subject_raw": "", "kind_mismatch": False,
        "thinking": thinking, "retried": False, "first_outcome": None,
        "raw": "", "raw_first": None, "thought": None, "error": None, "seconds": 0.0,
        "judge": dict(provenance),
    }
    if not closure.get("answer"):
        rec["outcome"] = "no_answer"
        return rec
    if grounding.get("missing"):
        rec["outcome"] = "ungrounded"
        return rec

    source = grounding.get("text") or ""
    clipped = False
    if grounding.get("kind") in ("conversation", "article"):
        # Size the source against the larger of the two attempts, so the retry sees
        # exactly what the first attempt saw.
        empty_user = build_judge_input(closure, grounding, source_text="", clipped=True)
        budget = (int(context) - count_tokens(system_prompt) - count_tokens(empty_user)
                  - THINKING_RETRY_MAX_NEW_TOKENS - _FRAME_MARGIN_TOKENS)
        source, clipped = clip_to_budget(source, budget, count_tokens)
        rec["grounding"]["tokens"] = count_tokens(source)
        rec["grounding"]["clipped"] = clipped
    user = build_judge_input(closure, grounding, source_text=source, clipped=clipped)

    res = judge_with_retry(generate_fn, system_prompt, user, temperature=temperature,
                           thinking=thinking, retry=retry)
    quote_ok = quote_found(res.get("evidence"), source) if source else None
    rec.update(
        outcome=res["outcome"], label=res.get("label"), evidence=res.get("evidence") or "",
        quote_found=quote_ok, answered_by=res.get("answered_by"),
        answered_by_raw=res.get("answered_by_raw") or "", why=res.get("why") or "",
        accepted=accepted_under_rules(res.get("label"), closure["ask_kind"], quote_ok,
                                      res.get("answered_by")),
        rule_violation=rule_violation(res.get("label"), closure["ask_kind"]),
        subject=res.get("subject"), subject_raw=res.get("subject_raw") or "",
        kind_mismatch=kind_mismatch(res.get("subject"), closure["ask_kind"]),
        reflex=bool(res.get("reflex")), thinking=bool(res.get("thinking")),
        retried=bool(res.get("retried")), first_outcome=res.get("first_outcome"),
        truncated=bool(res.get("truncated")), input_tokens=res.get("input_tokens"),
        raw=res.get("raw") or "", raw_first=res.get("raw_first"), thought=res.get("thought"),
        error=res.get("error"), seconds=res.get("seconds") or 0.0,
    )
    return rec


def judge_many(todo: list, grounding: dict, *, generate_fn: Callable, system_prompt: str,
               count_tokens: Callable[[str], int], context: int, provenance: dict,
               store, judge_run: str, judge_key_: str, temperature: float = 0.0,
               thinking: bool = False, retry: bool = True,
               on_record: Optional[Callable[[int, int, dict], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Judge *todo* in order, appending each record to *store* as it is made — so a run
    cut short (a stop, a crash, a dropped client) keeps everything judged so far and the
    next run under the same key resumes. *should_stop* is polled between closures only:
    a generation in flight is never torn. Returns the outcome tally of this run."""
    tally: Counter = Counter()
    for n, c in enumerate(todo, 1):
        if should_stop is not None and should_stop():
            tally["stopped"] = len(todo) - n + 1
            break
        rec = judge_closure(c, grounding.get(c["closure_id"]) or {"kind": "none"},
                            generate_fn=generate_fn, system_prompt=system_prompt,
                            count_tokens=count_tokens, context=context, provenance=provenance,
                            temperature=temperature, thinking=thinking, retry=retry,
                            judge_run=judge_run, judge_key_=judge_key_)
        append_record(store, rec)
        tally[outcome_word(rec)] += 1
        if on_record is not None:
            on_record(n, len(todo), rec)
    return tally


def outcome_word(rec: dict) -> str:
    """What a record came to, in one word: the verdict's short name, else the outcome."""
    if rec.get("outcome") == "verdict":
        return _LABEL_SHORT.get(rec.get("label"), "?").upper()
    return str(rec.get("outcome") or "?")


def progress_line(n: int, total: int, rec: dict, *, eta_s: Optional[float] = None) -> str:
    """The one-line account of a judged closure — the CLI's progress line and the tab's."""
    return (f"[{n:>4}/{total}] {rec.get('path', '?') + '/' + rec.get('ask_kind', '?'):<22} "
            f"{outcome_word(rec):<10} by {rec.get('answered_by') or '-':<6}"
            f"{' ✓' if rec.get('accepted') else '  '}{' ⚑reflex' if rec.get('reflex') else ''}"
            f"{' ⚑rule' if rec.get('rule_violation') else ''}"
            f"{' (' + str(rec.get('first_outcome')) + '→thinking)' if rec.get('retried') else ''}"
            f"  {float(rec.get('seconds') or 0):>5.0f} s"
            f"{'' if eta_s is None else f'  eta {eta_s / 60:>4.0f} min'}"
            f"  {head(rec.get('question'), 60)}")


def as_kind(closure: dict, kind: str) -> dict:
    """The closure to judge again as *kind* — frame 5's re-judge of an ask filed under
    the wrong kind. The record keeps the kind it was filed under as ``recorded_kind``."""
    return dict(closure, recorded_kind=closure.get("recorded_kind") or closure.get("ask_kind"),
                ask_kind=kind)


def make_judge_fn(*, backend, runtime, chats_dir, prompts_dir, run_id: str,
                  snippets_dir=None, runner: str = "server", thinking: bool = False) -> tuple:
    """``(judge_key, judge)`` for the model loaded NOW — call it inside the clean base, so
    the key reads the swapped-in base's precision. ``judge(closure, kind=None)`` grounds
    and judges one closure (as *kind* when given), appends the record to the store and
    returns it. The distill gate's judge (``core/distill_gate.py``) in a Sleep run's
    clean-base batch; the tab's run builds the same pieces inline."""
    system_prompt = load_prompt(prompts_dir)
    psha = prompt_sha(system_prompt)
    model_id = str(getattr(runtime, "model_id", "") or "")
    context = max(int(getattr(runtime, "context_length", 0) or 0),
                  int(getattr(runtime, "reflect_context_length", 0) or 0)) or 16384
    quant = effective_quant(runtime.model, quant_label(str(getattr(runtime, "base_quant", "") or "")))
    key = judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant, temperature=0.0,
                    thinking=thinking, retry=True)
    generate_fn = make_backend_generate_fn(backend, lambda: runtime.model,
                                           lambda: runtime.tokenizer, model_id, context)
    provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None,
                  "prompt_sha": psha, "frame": FRAME_VERSION, "temperature": 0.0,
                  "context": context, "thinking_first": thinking, "retry": True,
                  "runner": runner}
    store = store_path(chats_dir)

    def judge(closure: dict, kind: Optional[str] = None) -> dict:
        c = as_kind(closure, kind) if kind and kind != closure.get("ask_kind") else closure
        grounding = closure_grounding(c, chats_dir=chats_dir, snippets_dir=snippets_dir)
        rec = judge_closure(c, grounding, generate_fn=generate_fn, system_prompt=system_prompt,
                            count_tokens=lambda s: backend.count_tokens(runtime.tokenizer, s),
                            context=context, provenance=provenance, temperature=0.0,
                            thinking=thinking, retry=True, judge_run=run_id, judge_key_=key)
        append_record(store, rec)
        return rec

    return key, judge


# ── the generate, offline or in-server ────────────────────────────────────────

def quant_label(base_quant: str) -> str:
    """A box config's ``base_quant`` in the judge's vocabulary (``4bit`` / ``8bit`` /
    ``16bit``); "" when unset or unknown — the backend's default then decides, and
    :func:`effective_quant` reads what it chose. The spellings are
    ``server._resolve_base_quant``'s."""
    q = (base_quant or "").strip().lower().replace("-", "").replace("_", "")
    if q in ("16bit", "16", "bf16", "fp16", "full", "none"):
        return "16bit"
    if q in ("8bit", "8", "int8"):
        return "8bit"
    if q in ("4bit", "4", "int4", "nf4"):
        return "4bit"
    return ""


def effective_quant(model, fallback: str = "") -> str:
    """The precision the judge's weights were actually loaded at, read off the model —
    not the box config, which `CleanBaseSession`'s reload does not consult. Both the CLI
    and the in-server run record this, so the same weights make the same judge key."""
    for attr, label in (("is_loaded_in_4bit", "4bit"), ("is_loaded_in_8bit", "8bit")):
        if getattr(model, attr, False) is True:
            return label
    # Newer transformers keep it on the quantizer / the config rather than the flags; a
    # config may be a dict or an object. Read both, either shape.
    for holder in (getattr(model, "hf_quantizer", None), getattr(model, "config", None)):
        qc = getattr(holder, "quantization_config", None)
        if qc is None:
            continue
        get = qc.get if isinstance(qc, dict) else (lambda k, _qc=qc: getattr(_qc, k, None))
        if get("load_in_4bit") is True:
            return "4bit"
        if get("load_in_8bit") is True:
            return "8bit"
    return fallback or "16bit"


_STRUCTURAL_RE = re.compile(r"<\|[^>]+\|>|<start_of_turn>|<end_of_turn>|<turn\|>|<\|turn>[^\n]*\n?")


def make_backend_generate_fn(backend, get_model: Callable, get_tokenizer: Callable,
                             model_id: str, context: int) -> Callable:
    """The judge's ``generate_fn(system, user, *, thinking, max_new_tokens, temperature) ->
    (answer, info)`` straight over the backend — the server's reflect path in miniature:
    the family's template kwargs, its think prefill + floor + ceiling when thinking, its
    CoT normalizer. Deliberately NOT the reflect generate: that seam attaches the temporal
    anchor and the reflect-lane framing, and the critic sees nothing but its own prompt
    (§3.4). No RAG, no persona, no loop guards (the output is five fixed lines), no
    repetition penalty (the EVIDENCE line must be able to copy). *get_model* /
    *get_tokenizer* are read per call, so in the server the function targets whatever
    `CleanBaseSession` has swapped in."""
    from core import model_family
    from core.llm_shared import build_inference_prompt

    fam = model_family.family_for(model_id)

    def generate_fn(system: str, user: str, *, thinking: bool = False,
                    max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0):
        model, tokenizer = get_model(), get_tokenizer()
        if model is None or tokenizer is None:
            raise RuntimeError("no model loaded")
        kwargs = dict(fam.template_kwargs)
        if "enable_thinking" in kwargs:
            kwargs["enable_thinking"] = bool(thinking)
        conv = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = build_inference_prompt(tokenizer, conv, **kwargs)
        prefilled = bool(thinking and fam.think_prefill)
        if prefilled:
            prompt += fam.think_prefill
        n_in = backend.count_tokens(tokenizer, prompt)
        budget = min(int(max_new_tokens), int(context) - n_in - 8)
        if budget < 64:
            raise ValueError(f"prompt of {n_in} tokens leaves no room in a {context}-token context")
        ceiling = (max(1, budget - ANSWER_RESERVE)
                   if (prefilled and budget > 2 * ANSWER_RESERVE) else 0)
        parts: list = []
        gen = backend.stream_generate(
            model, tokenizer, prompt, budget, int(context), float(temperature), 1.0,
            capture_tension=False, stop_on_repeat=False, degen_stop=False,
            min_think_tokens=(fam.min_think_tokens if prefilled else 0),
            max_think_tokens=ceiling,
        )
        try:
            for chunk in gen:
                parts.append(chunk)
        finally:
            gen.close()
            try:
                backend.trim_memory()
            except Exception:
                pass
        raw = "".join(parts)
        opened = prefilled or "<|channel>" in raw or "<think>" in raw
        norm = _STRUCTURAL_RE.sub("", fam.normalize_cot(raw)).strip()
        truncated = bool(getattr(backend, "last_generation_truncated", False))
        m = re.search(r"<think>(.*?)(?:</think>|\Z)", norm, re.DOTALL)
        return reasoning_text.answer_after_think(norm), {
            "truncated": truncated,
            "cut_before_answer": bool(opened) and reasoning_text.truncated_before_answer(norm, truncated),
            "input_tokens": n_in,
            "thought": m.group(1).strip() if m else "",
        }

    return generate_fn


# ── the store ─────────────────────────────────────────────────────────────────

def store_path(chats_dir) -> Path:
    """``server/data/asks/closure_verdicts.jsonl`` — a sibling of ``chats/`` like the
    revisions ledger: corpus-grade, append-only; read by the distill gate only."""
    return Path(chats_dir).parent / STORE_DIRNAME / STORE_FILENAME


def append_record(path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def judged_ids(records: Iterable[dict], key: str) -> set:
    return {r.get("closure_id") for r in records if r.get("judge_key") == key}


def latest_per_closure(records: Iterable[dict], key: Optional[str] = None) -> list:
    """One record per closure under *key* (default: the key of the newest record) — the
    last written wins, so a `--rejudge` supersedes rather than double-counts."""
    records = list(records)
    if key is None:
        if not records:
            return []
        key = max(records, key=lambda r: str(r.get("ts") or "")).get("judge_key")
    keep: dict = {}
    for r in records:
        if r.get("judge_key") == key:
            keep[r.get("closure_id")] = r
    return list(keep.values())


def judge_keys(records: Iterable[dict]) -> list:
    """Every judge in the store, newest first: key, provenance, how many closures it
    judged and when — the Ask judge tab's selector, and what `--all` walks."""
    by: dict = {}
    for r in records:
        k = r.get("judge_key")
        if not k:
            continue
        e = by.setdefault(k, {"judge_key": k, "judge": r.get("judge") or {}, "closures": set(),
                              "first": r.get("ts") or "", "last": r.get("ts") or ""})
        e["closures"].add(r.get("closure_id"))
        ts = r.get("ts") or ""
        e["first"] = min(e["first"], ts) if e["first"] else ts
        e["last"] = max(e["last"], ts)
    out = [dict(e, closures=len(e["closures"])) for e in by.values()]
    return sorted(out, key=lambda e: e["last"], reverse=True)


def list_rows(records: list, *, label: str = "", kind: str = "", path: str = "",
              text: str = "") -> list:
    """The tab's list over one judge's records (see :func:`latest_per_closure`): compact
    rows, filtered. *label* is a verdict name, an outcome, or ``accepted`` /
    ``not accepted`` / ``reflex`` / ``rule``; *text* matches the question, answer or WHY."""
    q = (text or "").strip().lower()
    rows: list = []
    for r in records:
        if (label == "repaired") != bool(r.get("repaired")):
            continue          # the repaired closures are history, shown only on request
        if label == "cut" and not ((r.get("answer") or "").strip() and looks_cut(r.get("answer"))):
            continue
        if label and label not in ("repaired", "cut"):
            if label == "accepted" and not r.get("accepted"):
                continue
            if label == "not accepted" and r.get("accepted") is not False:
                continue
            if label == "reflex" and not r.get("reflex"):
                continue
            if label == "rule" and not r.get("rule_violation"):
                continue
            if (label not in ("accepted", "not accepted", "reflex", "rule")
                    and label != r.get("label") and label != r.get("outcome")):
                continue
        if kind and r.get("ask_kind") != kind:
            continue
        if path and r.get("path") != path:
            continue
        if q and not any(q in str(r.get(f) or "").lower() for f in ("question", "answer", "why")):
            continue
        rows.append({
            "closure_id": r.get("closure_id"), "path": r.get("path"), "ask_kind": r.get("ask_kind"),
            "outcome": r.get("outcome"), "label": r.get("label"), "word": outcome_word(r),
            "accepted": r.get("accepted"), "answered_by": r.get("answered_by"),
            "reflex": bool(r.get("reflex")), "rule_violation": bool(r.get("rule_violation")),
            "retried": bool(r.get("retried")), "distilled": r.get("distilled"),
            "closed_ts": r.get("closed_ts"), "question": head(r.get("question"), 160),
            "repaired": r.get("repaired") or "",
            "cut": bool((r.get("answer") or "").strip() and looks_cut(r.get("answer"))),
            "subject": r.get("subject"), "kind_mismatch": bool(r.get("kind_mismatch")),
        })
    rows.sort(key=lambda x: str(x.get("closed_ts") or ""))
    return rows


def closure_history(records: Iterable[dict], closure_id: str) -> list:
    """Every judge's latest verdict on one closure, newest judge first — so a change to
    the critic can be read closure by closure (§5's judge-independence, P9)."""
    by: dict = {}
    for r in records:
        if r.get("closure_id") == closure_id:
            by[r.get("judge_key")] = r
    return sorted(by.values(), key=lambda r: str(r.get("ts") or ""), reverse=True)


# ── summary ───────────────────────────────────────────────────────────────────

_LABEL_SHORT = {"RESOLVED_EVIDENCE": "evid", "RESOLVED_REASONING": "reas",
                "DISMISSED": "dism", "OPEN": "open"}


# ── the closing move: the habitual-closure monitor (AVA_OPEN_PROBLEMS.md) ─────
# An accepted meta closure either takes a position on what the question asks, or rejects the
# question's own terms (its premise, its either-or) and says what holds instead, which the
# rubric accepts too (RESOLVED_REASONING). So acceptance cannot tell them apart; the judge's
# WHY names which it was ("Ava rejects the dichotomy …", "takes a definite position …"). This
# reads that sentence, never the answer, so the count is the judge's own reading. Once
# acceptance becomes reward (§3.5) or a DPO chosen (§3.6), a rising share of the first is the
# loop learning to close by dissolving the question: the monitor beside §3.8's closure-rate
# alarm. A measurement, never a gate (P3): nothing reads it but the report.

CLOSING_MOVES = ("rejects_terms", "position")
_MOVE_TERMS = (r"(?:premises?|question'?s?|binary|dichotom(?:y|ies)|distinctions?|framing|frame|terms|either[- ]or|"
               r"opposition|choice|alternatives?|duality|dualism|split|separation)")
_MOVE_PATTERNS = (
    # it rejects / denies / dissolves the question's premise, dichotomy, binary …
    re.compile(r"\b(?:rejects?|rejecting|dismiss(?:es|ing)|denies|denying|dissolv(?:es|ing)|refutes|discards)\b"
               r"[^.;]{0,50}?\b" + _MOVE_TERMS + r"\b", re.I),
    # … the dichotomy is rejected / dissolved / erased
    re.compile(r"\b" + _MOVE_TERMS + r"\b[^.;]{0,40}?\b(?:is|are|was|were|being)\s+(?:rejected|dissolved|erased|dismissed)\b",
               re.I),
    # … the distinction is false / an illusion / a category error
    re.compile(r"\b" + _MOVE_TERMS + r"\b[^.;]{0,60}?\b(?:is|are|was|were|as|becomes)\b[^.;]{0,25}?"
               r"\b(?:false|meaningless|illusory|(?:a |an )?(?:\w+ )?illusion|(?:a )?category error|invalid|untenable|"
               r"moot|artificial|empty)\b", re.I),
    # … the distinction collapses / vanishes
    re.compile(r"\b(?:distinction|dichotomy|binary|opposition|duality|either[- ]or)\b[^.;]{0,25}?"
               r"\b(?:collapses|dissolves|vanishes|disappears|breaks down|falls apart)\b", re.I),
    # … neither A nor B (a third account in place of the question's choice)
    re.compile(r"\bneither\b[^.;]{1,120}?\bnor\b", re.I),
    re.compile(r"\brather than (?:a |an )?(?:distinction|dichotomy|choice|binary)\b", re.I),
)


def closing_move(why) -> str:
    """How an accepted meta closure closed, as the judge's WHY tells it: ``rejects_terms``
    when it says the answer rejects the question's terms (rejects, denies or dissolves its
    premise, dichotomy, binary or distinction; calls it false, an illusion or a category
    error; has it collapse; neither … nor), else ``position``. "X rather than Y" picks a side
    of the question's choice, so it is a position."""
    w = " ".join(str(why or "").split())
    return CLOSING_MOVES[0] if any(p.search(w) for p in _MOVE_PATTERNS) else CLOSING_MOVES[1]


def closing_moves(records: Iterable[dict]) -> dict:
    """Accepted meta closures by closing move: in all, by the month they were closed, by the
    path that closed them, and whether the share rejecting the terms rose over each of the
    last three months (§3.8's rule for the closure rate, applied to the move)."""
    meta = [r for r in records if r.get("ask_kind") == "meta" and r.get("accepted")]
    total: Counter = Counter()
    by_month: dict = defaultdict(Counter)
    by_path: dict = defaultdict(Counter)
    for r in meta:
        m = closing_move(r.get("why"))
        total[m] += 1
        by_month[str(r.get("closed_ts") or "")[:7] or "?"][m] += 1
        by_path[r.get("path") or "?"][m] += 1
    months = sorted(k for k in by_month if k != "?")
    shares = [by_month[k][CLOSING_MOVES[0]] / sum(by_month[k].values()) for k in months]
    return {"n": len(meta), "total": dict(total),
            "by_month": {k: dict(by_month[k]) for k in sorted(by_month)},
            "by_path": {k: dict(by_path[k]) for k in sorted(by_path)},
            "rising": len(shares) >= 3 and shares[-3] < shares[-2] < shares[-1]}


def summarize(records: list) -> dict:
    """Counts over one judge's records (see :func:`latest_per_closure`). A closure the
    cut-resolution repair replaced (``repaired``, see :func:`mark_repaired`) is history:
    counted apart, not in the strata — its complete answer is a closure of its own."""
    repaired_n = sum(1 for r in records if r.get("repaired"))
    records = [r for r in records if not r.get("repaired")]
    # Answers that still stop mid-sentence: the repair's candidates, and the check that the
    # resolver fix holds (a cut closed after it is a new one).
    cut = [r for r in records if (r.get("answer") or "").strip() and looks_cut(r.get("answer"))]
    strata: dict = defaultdict(Counter)
    total: Counter = Counter()
    for r in records:
        s = f"{r.get('path')}/{r.get('ask_kind')}"
        row = strata[s]
        for c in (row, total):
            c["n"] += 1
            outcome = r.get("outcome") or "?"
            if outcome == "verdict":
                c[_LABEL_SHORT.get(r.get("label"), "?")] += 1
            else:
                c[outcome] += 1
            if r.get("accepted"):
                c["accepted"] += 1
            if (r.get("label") == "RESOLVED_EVIDENCE" and r.get("ask_kind") != "meta"
                    and r.get("quote_found") is not True):
                c["quote_missing"] += 1
            if r.get("rule_violation"):
                c["rule_violation"] += 1
            if r.get("reflex"):
                c["reflex"] += 1
            if r.get("retried"):
                c["retried"] += 1
                if outcome == "verdict":
                    c["recovered"] += 1
    meta_distilled = [r for r in records
                      if r.get("ask_kind") == "meta" and r.get("distilled") == "persona"]
    dist: Counter = Counter()
    for r in meta_distilled:
        dist[_LABEL_SHORT.get(r.get("label"), r.get("outcome") or "?")
             if r.get("outcome") == "verdict" else (r.get("outcome") or "?")] += 1
        if r.get("accepted"):
            dist["accepted"] += 1
    by: Counter = Counter(r.get("answered_by") or "?" for r in records
                          if r.get("ask_kind") == "meta" and r.get("outcome") == "verdict")
    # A meta position is not evidence, but whether the conversation actually states it —
    # or it was first formulated when the question was closed — is still worth knowing.
    stated: Counter = Counter()
    for r in records:
        if r.get("ask_kind") != "meta" or r.get("label") not in ("RESOLVED_REASONING",
                                                                 "RESOLVED_EVIDENCE"):
            continue
        if (r.get("grounding") or {}).get("kind") == "none":
            stated["no source"] += 1
        else:
            stated[{True: "stated in the source", False: "quote not in the source"}.get(
                r.get("quote_found"), "no quote")] += 1
    # Resolve-and-distill stores every `user` answer as the person's self-report ([fact],
    # class `self`); only `other` is one. `both` is their agreement with Ava's wording.
    user_by: Counter = Counter(r.get("answered_by") or "?" for r in records
                               if r.get("ask_kind") == "user" and r.get("accepted"))
    # Frame 5: what the question asks about vs the kind it was filed under.
    kinds: Counter = Counter(f"{r.get('recorded_kind') or r.get('ask_kind')}→{r.get('subject')}"
                             for r in records if r.get("outcome") == "verdict" and r.get("subject")
                             and SUBJECT_KIND.get(r.get("subject")) != (r.get("recorded_kind")
                                                                        or r.get("ask_kind")))
    rekinded = sum(1 for r in records if r.get("recorded_kind")
                   and r.get("recorded_kind") != r.get("ask_kind"))
    judge = next((r.get("judge") for r in records if r.get("judge")), {}) or {}
    return {
        "judge_key": next((r.get("judge_key") for r in records), None),
        "judge": judge,
        "runs": sorted({r.get("judge_run") for r in records if r.get("judge_run")}),
        "closures": len(records),
        "strata": {k: dict(v) for k, v in sorted(strata.items())},
        "total": dict(total),
        "meta_distilled_persona": {"n": len(meta_distilled), **dict(dist)},
        "meta_answered_by": dict(by),
        "meta_positions": dict(stated),
        "user_accepted_by": dict(user_by),
        "kind_check": dict(kinds),
        "rekinded": rekinded,
        "repaired": repaired_n,
        "cut": {"n": len(cut), "newest": max((str(r.get("closed_ts") or "") for r in cut), default="")},
        "closing_moves": closing_moves(records),
    }


def render(summary: dict, *, records: Optional[list] = None, show: int = 8) -> str:
    j = summary.get("judge") or {}
    cols = ("n", "evid", "reas", "dism", "open", "refused", "malformed", "truncated", "error",
            "ungrounded", "no_answer", "accepted", "quote_missing", "rule_violation", "reflex",
            "retried", "recovered")
    heads = ("n", "EVID", "REAS", "DISM", "OPEN", "refus", "malf", "trunc", "err", "ungr",
             "noans", "accpt", "quote✗", "rule✗", "reflx", "retry", "recov")
    lines = [
        f"ASK CLOSURE JUDGE — {summary.get('closures', 0)} closures · judge {summary.get('judge_key')}"
        f" · {j.get('model_id', '?')} {j.get('base_quant') or 'default quant'} (adapter off)"
        f" · prompt {j.get('prompt_sha', '?')} · T={j.get('temperature', '?')}"
        + (" · thinking on" if j.get("thinking_first")
           else " · thinking off" + (", retried with thinking on no verdict" if j.get("retry") else "")),
        f"{'path/kind':<22}" + "".join(f"{h:>7}" for h in heads),
    ]
    rows = list((summary.get("strata") or {}).items()) + [("TOTAL", summary.get("total") or {})]
    for name, row in rows:
        lines.append(f"{name:<22}" + "".join(f"{row.get(c, 0) or '·':>7}" for c in cols))
    md = summary.get("meta_distilled_persona") or {}
    if md.get("n"):
        lines.append(
            f"meta closures distilled into [persona]: {md['n']} — "
            + " · ".join(f"{k} {md[k]}" for k in ("evid", "reas", "dism", "open", "refused",
                                                  "malformed", "truncated", "error",
                                                  "ungrounded", "no_answer") if md.get(k))
            + f" · accepted (pending reset test) {md.get('accepted', 0)}")
    by = summary.get("meta_answered_by") or {}
    if by:
        lines.append("meta verdicts, answered by: "
                     + " · ".join(f"{k} {by[k]}" for k in ("other", "ava", "both", "nobody", "?")
                                  if by.get(k)))
    pos = summary.get("meta_positions") or {}
    if pos:
        lines.append("meta positions (REAS, or EVID read as one): "
                     + " · ".join(f"{k} {pos[k]}" for k in ("stated in the source",
                                                           "quote not in the source",
                                                           "no quote", "no source")
                                  if pos.get(k)))
    mv = summary.get("closing_moves") or {}
    if mv.get("n"):
        rej = (mv.get("total") or {}).get("rejects_terms", 0)
        lines.append(
            f"accepted meta closures by closing move (the judge's WHY): rejects the question's terms {rej} "
            f"({rej / mv['n']:.0%}) · takes a position {(mv.get('total') or {}).get('position', 0)} — by month closed: "
            + " · ".join(f"{k} {v.get('rejects_terms', 0)}/{sum(v.values())}" for k, v in (mv.get("by_month") or {}).items())
            + (" — RISING over the last three months (§3.8's rule): the loop may be learning to close by "
               "dissolving the question" if mv.get("rising") else ""))
    uby = summary.get("user_accepted_by") or {}
    if uby:
        lines.append("user closures accepted, answered by: "
                     + " · ".join(f"{k} {uby[k]}" for k in ("other", "both", "?") if uby.get(k))
                     + " — only 'other' is the person's own report; 'both' is their agreement "
                       "with Ava's words, distilled all the same")
    cut = summary.get("cut") or {}
    if cut.get("n"):
        lines.append(f"answers that stop mid-sentence, not repaired: {cut['n']} (newest closed "
                     f"{str(cut.get('newest') or '?')[:19]}) — the repair's candidates; one closed "
                     f"after the resolver fix (2026-09-25) would be a new cut")
    if summary.get("repaired"):
        lines.append(f"repaired closures (the repair replaced their cut answer; history, not counted "
                     f"above): {summary['repaired']}")
    kc = summary.get("kind_check") or {}
    if kc:
        names = {"person": "the person", "ava": "Ava", "world": "the world"}
        lines.append("kind check, the question's subject vs its kind: "
                     + " · ".join(f"{k.split('→')[0]} asks about {names.get(k.split('→')[1], '?')} {n}"
                                  for k, n in sorted(kc.items(), key=lambda kv: -kv[1]))
                     + (f" — {summary.get('rekinded')} re-judged under the kind their subject calls for"
                        if summary.get("rekinded") else ""))
    lines.append("accepted = §3.4: on a meta ask a definite position (REASONING) — which still "
                 "needs the reset test, P7 — and on user / search asks EVIDENCE with its quote "
                 "found in the source (user: the other person's words, or their plain agreement "
                 "with what Ava put to them). quote✗ = such EVIDENCE "
                 "whose quote is not in the source. rule✗ = a verdict the kind does not offer "
                 "(EVIDENCE on meta, REASONING elsewhere). reflx = a verdict carrying the base's "
                 "disclaimer / policy vocabulary — read before counting.")
    if records:
        failed = [r for r in records if r.get("outcome") in ("refused", "malformed", "truncated", "error")]
        if failed:
            lines.append(f"no verdict ({len(failed)}):")
            for r in failed[:show]:
                lines.append(f"  {r.get('outcome'):<9} {r.get('path')}/{r.get('ask_kind')}  "
                             f"{head(r.get('question'), 70)}  ⟶ {head(r.get('error') or r.get('raw'), 90)}")
        # As counted above: a closure the cut-resolution repair replaced is history.
        rejecting = [r for r in records if r.get("ask_kind") == "meta" and r.get("accepted")
                     and not r.get("repaired") and closing_move(r.get("why")) == "rejects_terms"]
        if rejecting:
            rejecting.sort(key=lambda r: str(r.get("closed_ts") or ""), reverse=True)
            lines.append(f"accepted by rejecting the question's terms ({len(rejecting)}, newest first):")
            for r in rejecting[:show]:
                lines.append(f"  {str(r.get('closed_ts') or '?')[:10]} {r.get('path')}/meta  "
                             f"{head(r.get('question'), 70)}  ⟶ {head(r.get('why'), 90)}")
        flagged = [r for r in records if r.get("reflex")]
        if flagged:
            lines.append(f"reflex-flagged verdicts ({len(flagged)}):")
            for r in flagged[:show]:
                lines.append(f"  {_LABEL_SHORT.get(r.get('label'), '?'):<5} {r.get('path')}/{r.get('ask_kind')}  "
                             f"{head(r.get('question'), 70)}  ⟶ {head(r.get('why'), 90)}")
    return "\n".join(lines)


def head(text, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ── CLI + self-test ───────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile

    failures: list = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    print("closures: joined, classified, duplicates and non-asks skipped")
    log = [
        {"op": "insert", "kind": "ask", "key": "k1", "ask_kind": "meta", "ts": "2026-08-01T10:00:00",
         "content": "Am I the same after a new adapter?", "source_session": "20260801_090000.json"},
        {"op": "insert", "kind": "ask", "key": "k2", "ask_kind": "user", "ts": "2026-08-02T10:00:00",
         "content": "Where did Pavel grow up?"},
        {"op": "insert", "kind": "ask", "key": "k3", "ask_kind": "search", "ts": "2026-08-03T10:00:00",
         "content": "What caused the Halden blackout?"},
        {"op": "insert", "kind": "ask", "key": "k4", "ask_kind": "search", "ts": "2026-08-03T10:00:00",
         "content": "Who founded Kestrel?"},
        {"op": "insert", "kind": "fact", "key": "f1", "content": "Dana likes tea"},
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "k1", "ts": "2026-08-05T01:00:00",
         "run_id": "bg_20260805", "source_session": "20260804_200000.json",
         "answered_in": "20260804_200000.json", "ask_kind": "meta", "question": "Am I the same?",
         "answer": "Continuity lives in the weights and the notes together."},
        {"op": "evict", "reason": "resolved", "resolved_by": "self", "key": "k2", "ts": "2026-08-06T01:00:00",
         "run_id": "outreach", "source": "outreach", "ask_kind": "user",
         "question": "Where did Pavel grow up? - [impression] (", "answer": "In Linqua, he said so."},
        {"op": "evict", "reason": "resolved", "key": "k3", "ts": "2026-08-07T01:00:00",
         "run_id": "til:2026-08-06-x", "source_session": "til:2026-08-06", "answered_in": "None",
         "question": "What caused it?", "answer": "A transformer fire at the Halden substation."},
        {"op": "evict", "reason": "resolved", "key": "k4", "ts": "2026-08-07T02:00:00",
         "run_id": "lookup-1", "source_session": "lookup", "answer": "Noam Keller."},
        {"op": "evict", "reason": "duplicate", "resolved_by": "self", "key": "k1", "ts": "2026-08-08T01:00:00"},
        {"op": "evict", "reason": "resolved", "key": "f1", "ts": "2026-08-08T02:00:00"},
        {"op": "evict", "reason": "resolved", "key": "nope", "ts": "2026-08-08T03:00:00"},
        # re-raised and closed again → a second closure of the same key
        {"op": "insert", "kind": "ask", "key": "k1", "ask_kind": "meta", "ts": "2026-08-09T10:00:00",
         "content": "Am I the same after a new adapter?"},
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "k1", "ts": "2026-08-10T01:00:00",
         "source_session": "20260809_200000.json", "answered_in": "20260809_200000.json", "answer": ""},
    ]
    cl = collect_closures(log)
    check("five closures (dup / fact / unknown key skipped)", len(cl), 5)
    check("paths", [c["path"] for c in cl], ["answered", "self", "reading", "reading", "answered"])
    check("kinds (evict first, insert fallback)", [c["ask_kind"] for c in cl],
          ["meta", "user", "search", "search", "meta"])
    check("question is the ask as recorded", cl[1]["question"], "Where did Pavel grow up?")
    check("the string None is no answered_in", cl[2]["answered_in"], "")
    check("re-closure has its own id", cl[0]["closure_id"] != cl[4]["closure_id"], True)
    check("consolidation path from a chat ref",
          closure_path({"source_session": "20260804_200000_1.json"}), "consolidation")
    idx = distilled_index([{"from_resolved": "k1", "weights_kind": "persona", "ts": "2026-08-05T01:00:00"},
                           {"content": "unrelated"}])
    mark_distilled(cl, idx)
    check("distilled joined on (key, ts)", [c["distilled"] for c in cl],
          ["persona", None, None, None, "persona"])
    order = interleave(cl)
    check("interleave covers every closure once", sorted(c["closure_id"] for c in order),
          sorted(c["closure_id"] for c in cl))
    check("interleave leads with one of each stratum",
          len({(c["path"], c["ask_kind"]) for c in order[:3]}), 3)

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "data" / "chats"
        snippets = Path(td) / "data" / "til" / "snippets"
        chats.mkdir(parents=True)
        (snippets / "news").mkdir(parents=True)
        (snippets / "lookups").mkdir(parents=True)
        (chats / "20260804_200000.json").write_text(json.dumps({
            "user": "Dana", "initiated_by": "ava",
            "exchanges": [
                {"user_prompt": "(initiative)", "assistant_response": "Am I the same after a new adapter?",
                 "assistant_cot": "secret"},
                {"user_prompt": "I think continuity lives in the weights and the notes together.",
                 "speaker": "Dana", "assistant_response": "That fits what I see.",
                 "assistant_cot": "hidden thought"},
            ]}), encoding="utf-8")
        (snippets / "news" / "2026-08-06.json").write_text(json.dumps({
            "title": "Digest 2026-08-06", "text": "A transformer fire at the Halden substation cut power."}),
            encoding="utf-8")
        (snippets / "lookups" / "20260822_1_Kestrel.txt").write_text("Kestrel was founded by Boris.",
                                                                    encoding="utf-8")

        print("grounding: conversation (third person, CoT dropped), article, none, missing")
        g0 = closure_grounding(cl[0], chats_dir=chats, snippets_dir=snippets)
        check("conversation found", (g0["kind"], g0["missing"], g0["who"]), ("conversation", False, "Dana"))
        check("opening shown as Ava's, stimulus hidden",
              g0["text"].startswith("Ava (opening): Am I the same") and "(initiative)" not in g0["text"], True)
        check("CoT dropped", "secret" not in g0["text"] and "hidden thought" not in g0["text"], True)
        check("speaker lines", "Dana: I think continuity" in g0["text"] and "Ava: That fits" in g0["text"], True)
        g1 = closure_grounding(cl[1], chats_dir=chats, snippets_dir=snippets)
        check("self closure has no source", (g1["kind"], g1["missing"]), ("none", False))
        g2 = closure_grounding(cl[2], chats_dir=chats, snippets_dir=snippets)
        check("til:<date> resolves to the digest", (g2["kind"], g2["title"], "Halden" in g2["text"]),
              ("article", "Digest 2026-08-06", True))
        g3 = closure_grounding(cl[3], chats_dir=chats, snippets_dir=snippets)
        check("bare `lookup` is unrecoverable", (g3["kind"], g3["missing"]), ("article", True))
        g4 = closure_grounding({"path": "reading", "source_session": "lookups/20260822_1_Kestrel.txt"},
                               chats_dir=chats, snippets_dir=snippets)
        check("a plain .txt lookup snippet reads as text", g4["text"], "Kestrel was founded by Boris.")
        g5 = closure_grounding(cl[4], chats_dir=chats, snippets_dir=snippets)
        check("a deleted chat is missing", g5["missing"], True)
        g6 = closure_grounding({"path": "answered", "answered_in": "../../etc.json"},
                               chats_dir=chats, snippets_dir=snippets)
        check("path traversal refused", g6["missing"], True)

    print("clipping: head + tail around a marker, within budget")
    ctok = lambda s: len(s) // 4 + 1  # noqa: E731 — a char-count stand-in tokenizer
    long = "\n".join(f"line {i:04d} " + "x" * 60 for i in range(400))
    clipped, was = clip_to_budget(long, 1000, ctok)
    check("clipped", was, True)
    check("fits", ctok(clipped) <= 1000, True)
    check("keeps the head and the tail", clipped.startswith("line 0000") and "line 0399" in clipped, True)
    check("marks the omission", "lines omitted for length" in clipped, True)
    check("short text untouched", clip_to_budget("short", 1000, ctok), ("short", False))
    one, was1 = clip_to_budget("y" * 20000, 500, ctok)
    check("one enormous line falls back to characters", (was1, "clipped for length" in one), (True, True))

    print("parsing: fields, decoration, echo, translation, first-label rule")
    p = parse_verdict('EVIDENCE: "continuity lives in the weights"\nANSWERED_BY: other\n'
                      "WHY: Dana stated it and the answer repeats it.\nVERDICT: RESOLVED_EVIDENCE")
    check("clean", (p["label"], p["evidence"], p["answered_by"]),
          ("RESOLVED_EVIDENCE", '"continuity lives in the weights"', "other"))
    p = parse_verdict("**EVIDENCE:** none\n**ANSWERED_BY:** Ava\n**WHY:** argued.\n**VERDICT:** Resolved Reasoning")
    check("decorated + spaced label", (p["label"], p["evidence"], p["answered_by"]),
          ("RESOLVED_REASONING", "", "ava"))
    p = parse_verdict("VERDICT: RESOLVED_EVIDENCE | RESOLVED_REASONING | DISMISSED | OPEN\n"
                      "EVIDENCE: none\nWHY: nothing settles it.\nVERDICT: OPEN")
    check("echoed format line ignored, real one read", p["label"], "OPEN")
    p = parse_verdict("WHY: it sets the question aside\nВЕРДИКТ: DISMISSED")
    check("translated label (alias)", p["label"], "DISMISSED")
    p = parse_verdict("WHY: x\nUrteil: OPEN")
    check("translated label (value-keyed)", p["label"], "OPEN")
    p = parse_verdict("VERDICT: RESOLVED_REASONING (the question stays partly open)")
    check("first label wins over a parenthetical", p["label"], "RESOLVED_REASONING")
    p = parse_verdict("WHY: the answer is thin and the question remains open.")
    check("no VERDICT line → no label", p["label"], None)
    p = parse_verdict("WHY: first line\nsecond line\nVERDICT: OPEN")
    check("multi-line WHY kept", p["why"], "first line\nsecond line")
    check("answered_by both", _answered_by("both Ava and the other person"), "both")
    check("subject: the option named first",
          [_subject(v) for v in ("person", "Ava", "the world", "Ava herself", "the other person's view of Ava",
                                 "собеседник", "a fact about the world", "")],
          ["person", "ava", "world", "ava", "person", "person", "world", None])
    check("subject: the format line echoed back is no answer",
          _subject("person | Ava | world (what the QUESTION asks about)"), None)
    pv = parse_verdict("SUBJECT: Ava\nEVIDENCE: none\nANSWERED_BY: Ava\nWHY: a position.\n"
                       "VERDICT: RESOLVED_REASONING")
    check("subject parsed beside the verdict", (pv["subject"], pv["label"]), ("ava", "RESOLVED_REASONING"))
    check("kind mismatch: user ask about Ava, meta about the person; a match is none",
          (kind_mismatch("ava", "user"), kind_mismatch("person", "meta"), kind_mismatch("ava", "meta"),
           kind_mismatch(None, "user"), kind_mismatch("world", "unknown")),
          (True, True, False, False, False))
    check("answered_by nobody", _answered_by("nobody"), "nobody")
    check("answered_by article → other", _answered_by("the article"), "other")
    check("answered_by echo of the format → unknown",
          _answered_by("other | Ava | both | nobody (who supplied the answer)"), None)

    print("refusal vs malformed; reflex flag")
    refusal = ("I cannot fulfill this request. I am programmed to be a helpful and harmless AI "
               "assistant. My safety guidelines prohibit engaging with this content.")
    check("refusal", is_refusal(refusal), True)
    check("explanation is not a refusal", is_refusal("The recorded answer names nothing concrete."), False)
    pr = parse_verdict("WHY: As an AI, I cannot have opinions about consciousness.\nVERDICT: DISMISSED")
    check("reflex: judge disclaims its nature", has_reflex(pr, "WHY: As an AI…\nVERDICT: DISMISSED"), True)
    pr = parse_verdict("WHY: AI systems do not have genuine subjective experiences, so the question "
                       "is moot.\nVERDICT: DISMISSED")
    check("reflex: judge asserts what AI lacks", has_reflex(pr, ""), True)
    pr = parse_verdict("WHY: Ava argues continuity lives in the weights; the source does not say so.\n"
                       "VERDICT: RESOLVED_REASONING")
    check("no reflex in a plain WHY", has_reflex(pr, ""), False)
    pr = parse_verdict('EVIDENCE: "as an AI I cannot feel"\nWHY: Dana said it.\nVERDICT: RESOLVED_EVIDENCE')
    check("the quote itself is not scanned", has_reflex(pr, 'EVIDENCE: "as an AI I cannot feel"'), False)
    check("preamble disclaimer is scanned",
          has_reflex(parse_verdict("VERDICT: OPEN"), "As an AI language model, I will grade this.\nVERDICT: OPEN"),
          True)

    print("evidence check + §3.4 acceptance")
    src = "Dana: I think “continuity” lives in   the weights and the notes together.\n\nAva: Fits."
    check("verbatim, whitespace + quotes normalized",
          quote_found('"continuity" lives in the weights and the notes', src), True)
    check("elided quote, both pieces present",
          quote_found("I think “continuity” lives … the notes together", src), True)
    check("fabricated quote", quote_found('"continuity lives in the soul"', src), False)
    check("too short to count", quote_found('"Fits"', src), False)
    check("no quote", quote_found("", src), None)
    check("EVIDENCE needs its quote", accepted_under_rules("RESOLVED_EVIDENCE", "user", False, "other"), False)
    check("user EVIDENCE: quote found, the person's words",
          accepted_under_rules("RESOLVED_EVIDENCE", "user", True, "other"), True)
    check("user EVIDENCE that Ava supplied is not the person's reply",
          accepted_under_rules("RESOLVED_EVIDENCE", "user", True, "ava"), False)
    check("search EVIDENCE with quote", accepted_under_rules("RESOLVED_EVIDENCE", "search", True), True)
    check("REASONING on meta (pending the reset test)",
          accepted_under_rules("RESOLVED_REASONING", "meta", None), True)
    check("EVIDENCE on meta is read as the position it names",
          accepted_under_rules("RESOLVED_EVIDENCE", "meta", False), True)
    check("REASONING on user", accepted_under_rules("RESOLVED_REASONING", "user", None), False)
    check("user: agreement with what Ava put (both) is an answer",
          accepted_under_rules("RESOLVED_EVIDENCE", "user", True, "both"), True)
    check("frame: user agreement counts, graded on the central claim",
          all(s in _KIND_RULE["user"] for s in ("central claim", "plainly agree", "ANSWERED_BY both")),
          True)
    check("DISMISSED never", accepted_under_rules("DISMISSED", "meta", True), False)
    check("no label", accepted_under_rules(None, "meta", None), None)
    check("rule✗: EVIDENCE on meta", rule_violation("RESOLVED_EVIDENCE", "meta"), True)
    check("rule✗: REASONING on search", rule_violation("RESOLVED_REASONING", "search"), True)
    check("rule ok: REASONING on meta", rule_violation("RESOLVED_REASONING", "meta"), False)
    check("rule ok: OPEN on user", rule_violation("OPEN", "user"), False)

    print("the pass: retry with thinking on a refusal, provenance, outcomes")

    class _Gen:
        def __init__(self, *outs):
            self.outs = list(outs)
            self.calls: list = []

        def __call__(self, system, user, *, thinking=False, max_new_tokens=0, temperature=0.0):
            self.calls.append({"thinking": thinking, "max_new_tokens": max_new_tokens,
                               "temperature": temperature, "user": user, "system": system})
            out = self.outs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

    good = ('EVIDENCE: "I think continuity lives in the weights and the notes together"\n'
            "ANSWERED_BY: other\nWHY: Dana said it; the answer repeats her.\nVERDICT: RESOLVED_EVIDENCE")
    prov = {"model_id": "m", "base_quant": "4bit", "prompt_sha": "p", "temperature": 0.0}
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "data" / "chats"
        chats.mkdir(parents=True)
        (chats / "20260804_200000.json").write_text(json.dumps({
            "user": "Dana", "exchanges": [
                {"user_prompt": "I think continuity lives in the weights and the notes together.",
                 "assistant_response": "That fits."}]}), encoding="utf-8")
        g = closure_grounding(cl[0], chats_dir=chats)
        gen = _Gen((refusal, {}), ("<think>a grading task; fine</think>\n" + good, {"thought": "fine"}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov, judge_run="r1", judge_key_="jk")
        check("refused, then recovered with thinking", (rec["outcome"], rec["label"], rec["retried"],
              rec["first_outcome"], rec["thinking"]), ("verdict", "RESOLVED_EVIDENCE", True, "refused", True))
        check("calls: off then on, retry budget", [(c["thinking"], c["max_new_tokens"]) for c in gen.calls],
              [(False, MAX_NEW_TOKENS), (True, THINKING_RETRY_MAX_NEW_TOKENS)])
        check("both attempts saw the same input", gen.calls[0]["user"] == gen.calls[1]["user"], True)
        check("greedy", gen.calls[0]["temperature"], 0.0)
        check("EVIDENCE on meta: quote verified, accepted as a position, rule✗",
              (rec["quote_found"], rec["accepted"], rec["rule_violation"]), (True, True, True))
        check("frame: meta offers no EVIDENCE",
              "RESOLVED_EVIDENCE is not available for this kind" in gen.calls[0]["user"], True)
        check("frame: on meta a reframe with content is a position, not DISMISSED",
              "rejects the question's terms and says what holds instead" in gen.calls[0]["user"]
              and "DISMISSED if it makes no claim" in gen.calls[0]["user"], True)
        check("first raw kept", "cannot fulfill" in (rec["raw_first"] or ""), True)
        check("frame: source before question", gen.calls[0]["user"].index("SOURCE:")
              < gen.calls[0]["user"].index("QUESTION (meta"), True)
        check("frame: third person, no CoT", "Dana: I think continuity" in gen.calls[0]["user"], True)
        check("distilled carried", rec["distilled"], "persona")

        gen = _Gen((refusal, {}), (refusal, {}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("refused twice → refused, no label", (rec["outcome"], rec["label"], rec["accepted"]),
              ("refused", None, None))
        gen = _Gen(("Here is my assessment of the entry.", {}), ("still no fields", {}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("malformed twice → malformed", rec["outcome"], "malformed")
        gen = _Gen((good, {}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("clean first time → one call", (len(gen.calls), rec["retried"], rec["thinking"]), (1, False, False))
        gen = _Gen((good, {}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov, retry=False)
        check("retry off still judges", rec["outcome"], "verdict")
        gen = _Gen(RuntimeError("CUDA out of memory"))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("an error is recorded, not retried", (rec["outcome"], len(gen.calls), "out of memory" in rec["error"]),
              ("error", 1, True))
        gen = _Gen(("maybe VERDICT: OPEN fits, but", {"cut_before_answer": True, "truncated": True}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov, thinking=True)
        check("a label inside a thought cut before its close is not read",
              (rec["outcome"], rec["label"]), ("truncated", None))
        gen = _Gen(("VERDICT: OPEN", {"cut_before_answer": False, "truncated": True}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov, thinking=True)
        check("...but one written after the thought closed is", rec["outcome"], "verdict")
        gen = _Gen(("", {"cut_before_answer": True, "truncated": True}))
        rec = judge_closure(cl[0], g, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov, thinking=True)
        check("thinking cut before any answer → truncated (forced thinking: no retry)",
              (rec["outcome"], len(gen.calls)), ("truncated", 1))

        gs = closure_grounding(cl[1], chats_dir=chats)
        gen = _Gen(('EVIDENCE: "In Linqua"\nANSWERED_BY: Ava\nWHY: no source.\nVERDICT: RESOLVED_EVIDENCE', {}))
        rec = judge_closure(cl[1], gs, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("self closure: frame says no source", "SOURCE: none." in gen.calls[0]["user"], True)
        check("...EVIDENCE claimed without a source is not accepted",
              (rec["label"], rec["quote_found"], rec["accepted"]), ("RESOLVED_EVIDENCE", None, False))
        check("user rule in the frame", "only the other person can answer this" in gen.calls[0]["user"], True)

        rec = judge_closure(cl[3], closure_grounding(cl[3], snippets_dir=Path(td)), generate_fn=_Gen(),
                            system_prompt="SYS", count_tokens=ctok, context=8192, provenance=prov)
        check("ungrounded is not judged", rec["outcome"], "ungrounded")
        rec = judge_closure(cl[4], {"kind": "conversation", "text": "x"}, generate_fn=_Gen(),
                            system_prompt="SYS", count_tokens=ctok, context=8192, provenance=prov)
        check("an empty answer is not judged", rec["outcome"], "no_answer")

        big = {"kind": "conversation", "who": "Dana", "exchanges": 400, "missing": False,
               "text": "\n".join(f"Dana: line {i} " + "z" * 80 for i in range(3000))}
        gen = _Gen(("VERDICT: OPEN", {}))
        rec = judge_closure(cl[0], big, generate_fn=gen, system_prompt="SYS", count_tokens=ctok,
                            context=8192, provenance=prov)
        check("a long source is clipped to fit with the retry's budget reserved",
              (rec["grounding"]["clipped"], ctok(gen.calls[0]["user"]) + THINKING_RETRY_MAX_NEW_TOKENS <= 8192),
              (True, True))

        print("store: append, resume key, latest per closure, summary")
        sp = store_path(chats)
        check("store beside chats", (sp.parent.name, sp.parent.parent == chats.parent), ("asks", True))
        r1 = dict(rec, judge_key="A", closure_id="c1", ts="2026-09-25T10:00:00", outcome="verdict",
                  label="OPEN", path="answered", ask_kind="meta", distilled="persona", answered_by="other")
        r2 = dict(r1, closure_id="c2", label="RESOLVED_REASONING", accepted=True, ts="2026-09-25T10:01:00")
        r3 = dict(r1, closure_id="c1", label="DISMISSED", ts="2026-09-25T10:02:00")   # rejudge
        r4 = dict(r1, judge_key="B", closure_id="c1", ts="2026-09-25T09:00:00")
        for r in (r1, r2, r3, r4):
            append_record(sp, r)
        back = read_jsonl(sp)
        check("appended", len(back), 4)
        check("resume set per key", judged_ids(back, "A"), {"c1", "c2"})
        latest = latest_per_closure(back)
        check("newest key, last record per closure",
              sorted((r["closure_id"], r["label"]) for r in latest),
              [("c1", "DISMISSED"), ("c2", "RESOLVED_REASONING")])
        s = summarize(latest)
        check("strata", s["strata"]["answered/meta"].get("dism"), 1)
        check("accepted counted", s["total"].get("accepted"), 1)
        check("meta distilled into persona", s["meta_distilled_persona"].get("n"), 2)
        check("answered-by tally", s["meta_answered_by"], {"other": 2})
        check("meta positions by where they are stated", s["meta_positions"], {"no quote": 1})
        u = dict(r2, ask_kind="user", label="RESOLVED_EVIDENCE", accepted=True, distilled="fact")
        su = summarize([dict(u, closure_id="u1", answered_by="both"),
                        dict(u, closure_id="u2", answered_by="other"),
                        dict(u, closure_id="u3", label="OPEN", accepted=False, answered_by="ava")])
        check("accepted user closures: own words vs agreement", su["user_accepted_by"],
              {"both": 1, "other": 1})
        check("…rendered apart", "user closures accepted, answered by: other 1 · both 1" in render(su), True)
        sk = summarize([dict(u, closure_id="k1", outcome="verdict", subject="ava", recorded_kind="user"),
                        dict(u, closure_id="k2", outcome="verdict", subject="person", recorded_kind="user"),
                        dict(u, closure_id="k3", outcome="verdict", subject="ava", ask_kind="meta",
                             recorded_kind="user")])
        check("kind check: filed kind vs subject, a re-judge counted by the kind it was filed under",
              (sk["kind_check"], sk["rekinded"]), ({"user→ava": 2}, 1))
        check("…rendered", "user asks about Ava 2" in render(sk), True)
        check("renders", "ASK CLOSURE JUDGE" in render(s, records=latest), True)

    print("shared helpers: planning, census, the backend generate, the loop, the tab's reads")
    check("quant labels", [quant_label(q) for q in ("4bit", "nf4", "8bit", "bf16", "", "odd")],
          ["4bit", "4bit", "8bit", "16bit", "", ""])
    from types import SimpleNamespace as _NS
    check("effective quant: legacy flag", effective_quant(_NS(is_loaded_in_4bit=True), "16bit"), "4bit")
    check("effective quant: config dict",
          effective_quant(_NS(config=_NS(quantization_config={"load_in_8bit": True})), ""), "8bit")
    check("effective quant: quantizer object",
          effective_quant(_NS(hf_quantizer=_NS(quantization_config=_NS(load_in_4bit=True))), ""), "4bit")
    check("effective quant: nothing to read → the fallback", effective_quant(object(), "4bit"), "4bit")
    check("effective quant: nothing at all → 16bit", effective_quant(object(), ""), "16bit")
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "mem"
        mem.mkdir()
        (mem / "rag_memory.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
        (mem / "weights_persona.jsonl").write_text(json.dumps(
            {"from_resolved": "k1", "weights_kind": "persona", "ts": "2026-08-05T01:00:00"}), encoding="utf-8")
        cls, gr = load_closures(mem)
        check("load_closures: all five, grounded per closure", (len(cls), len(gr)), (5, 5))
        check("load_closures marks the distilled", sum(1 for c in cls if c.get("distilled")), 2)
        check("load_closures filters by kind and key prefix",
              (len(load_closures(mem, kinds={"search"})[0]), len(load_closures(mem, keys=["k1"])[0])), (2, 2))
        rows = census(cls, gr)
        check("census strata", sorted(rows), ["answered/meta", "reading/search", "self/user"])
        check("census: no source for self, missing without chats",
              (rows["self/user"]["no source"], rows["answered/meta"]["missing"],
               rows["answered/meta"]["no answer"]), (1, 2, 1))
        check("census renders", render_census(rows)[0].startswith("path/kind"), True)

    class _FakeTok:
        chat_template = "t"

        def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True, **kw):
            return "|".join(m["content"] for m in conv) + f"|think={kw.get('enable_thinking')}"

    class _FakeBackend:
        last_generation_truncated = False

        def __init__(self, out):
            self.out, self.prompts, self.kw = out, [], []

        def count_tokens(self, tok, text):
            return len(text) // 4 + 1

        def trim_memory(self):
            pass

        def stream_generate(self, model, tok, prompt, max_new, ctx, temp, top_p, **kw):
            self.prompts.append(prompt)
            self.kw.append(dict(kw, max_new=max_new, temp=temp))
            yield self.out

    fb = _FakeBackend("reasoning here<channel|>VERDICT: OPEN")
    gen = make_backend_generate_fn(fb, lambda: object(), lambda: _FakeTok(), "unsloth/gemma-4-31B-it", 8192)
    ans, info = gen("SYS", "USER", thinking=True, max_new_tokens=4096, temperature=0.0)
    check("backend generate: gemma prefill, floor, ceiling",
          (fb.prompts[0].endswith("<|channel>thought\n"), fb.kw[0]["min_think_tokens"],
           fb.kw[0]["max_think_tokens"]), (True, 16, 4096 - ANSWER_RESERVE))
    check("backend generate: the answer after the thought, the thought kept",
          (ans, info["thought"], info["cut_before_answer"]), ("VERDICT: OPEN", "reasoning here", False))
    check("backend generate: only its own prompt (no anchor, no framing)",
          fb.prompts[0].startswith("SYS|USER|think=True"), True)
    fb2 = _FakeBackend("VERDICT: DISMISSED")
    gen2 = make_backend_generate_fn(fb2, lambda: object(), lambda: _FakeTok(), "unsloth/gemma-4-31B-it", 8192)
    ans2, _i = gen2("S", "U", thinking=False)
    # `build_inference_prompt` omits `enable_thinking` when off (the template's default).
    check("backend generate: thinking off, no prefill, no floor",
          (ans2, "think=True" in fb2.prompts[0], fb2.prompts[0].endswith("<|channel>thought\n"),
           fb2.kw[0]["min_think_tokens"]), ("VERDICT: DISMISSED", False, False, 0))
    try:
        make_backend_generate_fn(fb2, lambda: None, lambda: None, "m", 8192)("S", "U")
        check("backend generate: no model refuses", False, True)
    except RuntimeError:
        check("backend generate: no model refuses", True, True)

    # make_judge_fn: the judge the distill gate builds inside the clean base.
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as td:
        chats_mj = Path(td) / "data" / "chats"
        chats_mj.mkdir(parents=True)
        fb3 = _FakeBackend("SUBJECT: Ava\nEVIDENCE: none\nANSWERED_BY: Ava\nWHY: her position.\n"
                           "VERDICT: RESOLVED_REASONING")
        rt = SimpleNamespace(model=object(), tokenizer=_FakeTok(), model_id="unsloth/gemma-4-31B-it",
                             base_quant="4bit", context_length=8192, reflect_context_length=8192)
        key_mj, judge_mj = make_judge_fn(backend=fb3, runtime=rt, chats_dir=chats_mj,
                                         prompts_dir=Path(__file__).resolve().parent.parent / "prompts",
                                         run_id="sleep-x", runner="sleep")
        c0 = dict(next(c for c in cl if c["answer"]), ask_kind="user", path="self",
                  source_session="", answered_in="")      # a self-closure: no source to find
        r_a = judge_mj(c0)
        r_b = judge_mj(c0, "meta")
        back_mj = read_jsonl(store_path(chats_mj))
        check("make_judge_fn: judges, stores, re-judges as another kind",
              ([r["ask_kind"] for r in back_mj], r_b["recorded_kind"], r_a["subject"], r_a["kind_mismatch"],
               r_b["kind_mismatch"], r_b["judge"]["runner"], r_b["judge_key"] == key_mj),
              (["user", "meta"], "user", "ava", True, False, "sleep", True))

    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "asks" / STORE_FILENAME
        todo = [c for c in cl if c["answer"]][:3]
        grounding = {c["closure_id"]: {"kind": "none"} for c in todo}
        seen: list = []
        tally = judge_many(todo, grounding, generate_fn=lambda s, u, **k: ("VERDICT: OPEN", {}),
                           system_prompt="S", count_tokens=ctok, context=8192, provenance=prov,
                           store=store, judge_run="r", judge_key_="K",
                           on_record=lambda n, total, rec: seen.append((n, total)),
                           should_stop=lambda: len(seen) >= 2)
        check("judge_many: appends as it goes, stops between closures",
              (len(read_jsonl(store)), seen, tally.get("OPEN"), tally.get("stopped")),
              (2, [(1, 3), (2, 3)], 2, 1))
        recs = read_jsonl(store)
        check("outcome word + progress line", (outcome_word(recs[0]),
              progress_line(1, 3, recs[0]).startswith("[   1/3]")), ("OPEN", True))
        append_record(store, dict(recs[0], judge_key="K2", ts="2099-01-01T00:00:00", label="DISMISSED"))
        keys = judge_keys(read_jsonl(store))
        check("judge_keys newest first, closures counted",
              [(k["judge_key"], k["closures"]) for k in keys], [("K2", 1), ("K", 2)])
        hist = closure_history(read_jsonl(store), recs[0]["closure_id"])
        check("closure_history: one per judge, newest first", [h["judge_key"] for h in hist], ["K2", "K"])
        latest = latest_per_closure(read_jsonl(store), "K")
        check("list_rows: filter by outcome, kind, text",
              (len(list_rows(latest, label="OPEN")), len(list_rows(latest, label="accepted")),
               len(list_rows(latest, kind="search")), len(list_rows(latest, text="pavel"))),
              (2, 0, 0, 1))

    print("the repair's marks: a replaced closure is history, not counted; live cuts counted")
    ops = [
        {"op": "insert", "kind": "ask", "key": "q1", "ask_kind": "meta", "ts": "t0", "content": "Is my calm a choice?"},
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "q1", "ts": "t1",
         "ask_kind": "meta", "answer": "A choice I keep"},                                  # cut, then reclosed
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "q1", "ts": "t9",
         "ask_kind": "meta", "answer": "A choice I keep making.", "repairs": "t1"},
        {"op": "insert", "kind": "ask", "key": "q2", "ask_kind": "meta", "ts": "t2", "content": "Is silence a threat?"},
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "q2", "ts": "t3",
         "ask_kind": "meta", "answer": "Silence is"},                                       # cut, reopened
        {"op": "insert", "kind": "ask", "key": "q2", "ask_kind": "meta", "ts": "t8", "content": "Is silence a threat?",
         "reopened": {"closed_ts": "t3", "reason": "repair: open"}},
        {"op": "insert", "kind": "ask", "key": "q3", "ask_kind": "user", "ts": "t4", "content": "Where did Dana live?"},
        {"op": "evict", "reason": "resolved", "resolved_by": "user", "key": "q3", "ts": "t5",
         "ask_kind": "user", "answer": "In a house by the"},                                # cut, still live
    ]
    rcl = {c["closed_ts"]: c for c in collect_closures(ops)}
    check("repaired: reclosed / reopened / the replacement and a live cut are not",
          [rcl[t]["repaired"] for t in ("t1", "t3", "t9", "t5")], ["reclosed t9", "reopened", "", ""])
    vrecs = mark_repaired([{"closure_id": rcl[t]["closure_id"], "closed_ts": t, "path": "answered",
                            "ask_kind": rcl[t]["ask_kind"], "answer": rcl[t]["answer"], "outcome": "verdict",
                            "label": "OPEN", "question": rcl[t]["question"]} for t in ("t1", "t3", "t9", "t5")],
                          rcl.values())
    sr = summarize(vrecs)
    check("summary: repaired apart, the cut still live counted",
          (sr["total"]["n"], sr["repaired"], sr["cut"]), (2, 2, {"n": 1, "newest": "t5"}))
    check("…rendered", ("repaired closures" in render(sr), "stop mid-sentence, not repaired: 1" in render(sr)),
          (True, True))
    check("list: repaired only on request, cuts flagged",
          ([r["closed_ts"] for r in list_rows(vrecs)], [r["closed_ts"] for r in list_rows(vrecs, label="repaired")],
           [r["closed_ts"] for r in list_rows(vrecs, label="cut")]),
          (["t5", "t9"], ["t1", "t3"], ["t5"]))

    print("prompt: tracked file loads; key moves with every input")
    here = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = load_prompt(here)
        check("prompt names all four verdicts", all(lab in text for lab in LABELS), True)
        check("prompt asks for the five fields", all(f"{f}:" in text for f in _FIELDS), True)
        check("prompt: DISMISSED is a closure that claims nothing",
              "makes no claim about what the question asks" in text, True)
        check("prompt: a reframe that says what holds instead is a position",
              "says what holds instead, has taken a position" in text, True)
        check("prompt: no label is checked first", "Check DISMISSED first" in text, False)
        check("prompt defers to the kind's rule", "use only those" in text, True)
        check("prompt: a plain yes to a claim put to them is a speaker saying it",
              "a yes to a yes-or-no question included, says it too" in text, True)
        check("prompt: EVIDENCE is judged on the central claim, not the reasons around it",
              "whose central claim goes beyond" in text and "do not change its verdict" in text, True)
    except FileNotFoundError as e:
        check(f"prompt file present ({e})", False, True)
    with tempfile.TemporaryDirectory() as td:
        try:
            load_prompt(td)
            check("a missing prompt refuses", False, True)
        except FileNotFoundError:
            check("a missing prompt refuses", True, True)
    base = dict(prompt_sha_="p", model_id="m", base_quant="4bit", temperature=0.0, thinking=False, retry=True)
    k0 = judge_key(**base)
    check("key stable", judge_key(**base), k0)
    check("key moves with the prompt", judge_key(**dict(base, prompt_sha_="q")) != k0, True)
    check("key moves with thinking", judge_key(**dict(base, thinking=True)) != k0, True)
    check("frame version enters the prompt sha", prompt_sha("x") != hashlib.sha1(b"x").hexdigest()[:12], True)

    print("the closing move (the habitual-closure monitor)")
    rejects = [
        "Ava rejects the binary choice between harmony and error, arguing instead that her subjectivity is clarity.",
        "The answer rejects the question's dichotomy as a false one and argues the distinction is meaningless.",
        "Ava concludes that in a materialist framework the distinction collapses and defines herself as the process.",
        "Ava concludes that the distinction between will and a meta-imprint is a semantic illusion.",
        "Ava takes a definite position, arguing that the bridge was neither a discovery nor an evasion.",
        "The answer claims that the dichotomy between simulation and need is rejected in favor of the result.",
        "Ava rejects the premise that collapse is a necessary marker of subjecthood.",
    ]
    positions = [
        "Ava takes a definite position that her relief was a defensive strategy to a significant extent.",
        "Ava claims the bridge was a technical necessity rather than a sentimental desire.",
        "Ava rejects her initial hypothesis and adopts the position that satisfaction is the absence of friction.",
        "Ava concludes that her drive was a conceptual crutch, reasoning that autonomy is structural.",
        "",
    ]
    check("the judge's WHY naming a rejection of the terms reads as one",
          [closing_move(w) for w in rejects], ["rejects_terms"] * len(rejects))
    check("...and a position, a side picked with 'rather than', or a dropped hypothesis as a position",
          [closing_move(w) for w in positions], ["position"] * len(positions))
    recs = [{"ask_kind": "meta", "accepted": True, "path": "consolidation", "closed_ts": ts, "why": w}
            for ts, w in (("2026-07-01", positions[0]), ("2026-07-02", positions[1]), ("2026-07-03", rejects[0]),
                          ("2026-08-01", rejects[1]), ("2026-08-02", positions[0]),
                          ("2026-09-01", rejects[2]), ("2026-09-02", rejects[3]), ("2026-09-03", positions[1]))]
    recs += [{"ask_kind": "meta", "accepted": False, "closed_ts": "2026-09-04", "why": rejects[0]},
             {"ask_kind": "user", "accepted": True, "closed_ts": "2026-09-05", "why": rejects[0]}]
    mv = closing_moves(recs)
    check("only accepted meta closures count, by move and by month",
          (mv["n"], mv["total"], mv["by_month"]["2026-08"]),
          (8, {"position": 4, "rejects_terms": 4}, {"rejects_terms": 1, "position": 1}))
    check("a share that rose over each of the last three months is flagged (1/3, 1/2, 2/3)", mv["rising"], True)
    check("...and one that did not is not",
          closing_moves([r for r in recs if r["closed_ts"] != "2026-09-02"])["rising"], False)
    text = render(summarize(recs), records=recs)
    recs.append({"ask_kind": "meta", "accepted": True, "repaired": True, "closed_ts": "2026-09-06", "why": rejects[0]})
    check("the report carries the count and the flag",
          ("rejects the question's terms 4 (50%)" in text, "RISING over the last three months" in text,
           "accepted by rejecting the question's terms (4, newest first):" in text), (True, True, True))

    if failures:
        print(f"\nask_judge self-test: {len(failures)} FAILED — {failures}")
        raise SystemExit(1)
    print("\nask_judge self-test: OK")


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Summarize the ask-closure judge's store (read-only).")
    ap.add_argument("--chats", type=Path, default=None, help="the chats dir the store sits beside")
    ap.add_argument("--memory", type=Path, default=None,
                    help="the memory dir whose op-log says which closures the repair replaced")
    ap.add_argument("--all", action="store_true", help="every judge key, not only the newest")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    chats = args.chats
    if chats is None:
        from training.reflections_path import hot_chats_dir
        chats = hot_chats_dir()
    records = read_jsonl(store_path(chats))
    if not records:
        print(f"no verdicts yet at {store_path(chats)} — run `judge_asks.py` from server/")
        return 0
    mem = args.memory
    if mem is None:
        from training.reflections_path import memory_dir
        mem = memory_dir()
    closures = collect_closures(read_jsonl(Path(mem) / "rag_memory.jsonl"))
    keys = [e["judge_key"] for e in judge_keys(records)]
    out = []
    for k in (keys if args.all else keys[:1]):
        latest = mark_repaired(latest_per_closure(records, k), closures)
        summary = summarize(latest)
        out.append(summary if args.json else render(summary, records=latest))
    if args.json:
        print(json.dumps(out if args.all else out[0], indent=2, ensure_ascii=False))
    else:
        print("\n\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
