"""The prompt-rewrite EVENT — Ava rewrites her standing prompt on her own decision.

PROMPT_REWRITE.md §4–§7 (stage 3). Stages 1–2 built the evidence: the locator logs one
prompt delta per exchange where a standing line was missing, and every reflection run
folds those into PATTERNS with a tension-weighted recurrence (``core.prompt_patterns``).
This module is what SPENDS them. It is one deliberation action (``rewrite_prompt``,
``core.deliberation``), offered only when the budget clears, and its body is:

1. **samples** — ``prompt_rewrite.samples`` (5) rewrites of the LIVE prompt (the experiment
   if one is active, else the seed), each handed the mature patterns as evidence, through
   the same recipe the manual Prompt-experiment button uses (RAG on, keyed on the standing
   prompt, thinking on); temperatures from ``prompt_rewrite.temperatures`` (default 0.9 ×
   samples — the owner's manual sweep found the rewrite non-monotonic in temperature, so
   the list keeps a sweep available);
2. **consensus** — one pass over the samples naming what MOST of them agree on, as notes
   (five draws at one moment are recurrence, the same instrument the digest uses across
   chats — a line present in one draw is a coin flip that landed);
3. **final draft** — one rewrite of the live prompt around those notes, at
   ``prompt_rewrite.final_temperature`` (0.9): one voice back, not a merged list;
4. **choice** — blind, shuffled, letter-labelled: the incumbent, the samples and the final
   draft, judged against the persona digest and the patterns, with a ``WHY``. On the
   ADAPTER: choosing a prompt is authorship (the box's rule: clean base for evaluation,
   adapter for authorship), and a conservative chooser is the dampening wanted — the
   blind shuffle and the WHY keep a self-confirming pick visible. Optional order check
   (``prompt_rewrite.order_check``): the pass runs again with the letters reversed and a
   disagreement is the incumbent;
5. **outcome** — *changed*: the winner REPLACES the active experiment through the
   experiment tier's own write (``prompt_experiment.save_experiment``, the ``set_prompt``
   semantics: the seed is carried forward as the revert anchor, never a candidate), the
   patterns weighed are CONSUMED (their delta keys land in the attempt record and leave
   the pool), and a first-person ``prompt`` worklog episode opens a thread the next
   changed event closes; *stayed* (the incumbent won) or *declined* (she chose another
   action at deliberation): nothing consumed, the attempt stamped for the gap only.

**The gate** (:func:`offer`, §4): enabled (the default since 2026-09-19); a model loaded; ≥1 mature pattern; the last
attempt — including a cancellation after an operator prompt change — at least ``prompt_rewrite.min_gap_hours`` (24)
old; at least one delta logged since that attempt (with nothing reflected in between,
five new draws are yesterday's draws — offering them again is noise, not evolution);
and no event in flight. An in-flight event is offered as *continue* and skips the budget
check, since it already passed it.

**Preemption** (§6): a drive body holds the single executor thread and a chat turn queues
behind it, and this event is on the order of an hour. So ``generation`` preempts it as it
preempts background reflection (:func:`request_preempt` sets the flag and fires the
shared cancel event; the in-flight generation aborts within a step), every completed
generation is persisted under ``data/hot/prompt/rewrite/<event_id>/`` with an
``event.json`` marker, and the next dispatch RESUMES from the last completed step. A
marker older than ``MAX_EVENT_AGE_DAYS`` is discarded. An operator prompt change cancels
the attempt without consuming evidence, including on resume; Revert preempts an active
generation. Activation checks the incumbent text and experiment timestamp under the
same lock as Revert, so it cannot overwrite the veto.

**Files**, all under ``data/hot/prompt/``: reads ``patterns.json`` (stage 2), writes
``rewrite/<event_id>/`` (candidates, process, the marker) and appends
``rewrite_log.jsonl`` — the attempt record stage 2's fold reads ``consumed_keys`` and the
last attempt's time from. The Prompt tab reads the experiment record's ``process``
(``rewrite:<event_id>``) for its provenance line; Revert there is the operator's veto.

Never imports ``server``: RAG, the reflect seam, the base-prompt loader, the config
loader, paths and the cancel event are injected via :func:`configure`. Self-test (fake
generate, temp dirs): ``python -m core.prompt_rewrite``.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core import prompt_experiment, prompt_patterns, worklog
from core.field_parse import label as _label
from core.reasoning_text import answer_after_think as _answer_after_think
from core.runtime_state import runtime as _runtime, session as _session

REWRITE_DIR = "rewrite"
EVENT_FILE = "event.json"
MAX_EVENT_AGE_DAYS = 7.0
ACTION = "rewrite_prompt"
WORKLOG_KIND = "prompt"

DEFAULT_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.9          # Sleep's default — the owner's call
DEFAULT_FINAL_TEMPERATURE = 0.9
DEFAULT_MIN_GAP_HOURS = 24.0
SAMPLE_MAX_NEW_TOKENS = "4096"     # the manual experiment's budget
CHOICE_MAX_NEW_TOKENS = "3072"

SAMPLE_PROMPT_FILE = "prompt_rewrite_prompt.txt"
CONSENSUS_PROMPT_FILE = "prompt_consensus_prompt.txt"
CHOICE_PROMPT_FILE = "prompt_choice_prompt.txt"

# Occupancy flag — named in the scheduler's `external_busy`; read by generation's preempt.
_rewrite_active = False
_preempt_requested = False
_active_lock = threading.Lock()

# ── injected capabilities ────────────────────────────────────────────────────
_get_rag: Callable = None
_make_sync_reflect_generate: Callable = None
_load_base_prompt: Callable = None
_load_server_config: Optional[Callable] = None
_PROMPTS_DIR: Any = None
_STATE_DIR: Any = None
_cancel_event: Any = None
_persona_dir_fn: Optional[Callable] = None
_send: Optional[Callable] = None


def configure(*, get_rag, make_sync_reflect_generate, load_base_prompt, prompts_dir,
              state_dir, load_server_config=None, cancel_event=None,
              persona_dir=None, send=None) -> None:
    """Wire in the server capabilities (once, at startup). ``state_dir`` is the prompt
    state dir (``data/hot/prompt``); ``persona_dir`` a callable returning the persona dir
    for the judge frame; ``cancel_event`` the shared threading.Event the reflect seam
    honours."""
    global _get_rag, _make_sync_reflect_generate, _load_base_prompt, _load_server_config
    global _PROMPTS_DIR, _STATE_DIR, _cancel_event, _persona_dir_fn, _send
    _send = send
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _load_base_prompt = load_base_prompt
    _load_server_config = load_server_config
    _PROMPTS_DIR = Path(prompts_dir)
    _STATE_DIR = Path(state_dir)
    _cancel_event = cancel_event
    _persona_dir_fn = persona_dir


def is_active() -> bool:
    return _rewrite_active


def request_preempt() -> None:
    """A user chat arrived — abort the event between generations (the flag) and inside
    one (the shared cancel event). Idempotent; what was completed stays on disk and the
    event resumes at the next dispatch."""
    global _preempt_requested
    _preempt_requested = True
    try:
        if _cancel_event is not None and _rewrite_active:
            _cancel_event.set()
    except Exception:
        pass


# ── settings ─────────────────────────────────────────────────────────────────

def settings(cfg: Optional[dict] = None) -> dict:
    """``prompt_rewrite.*`` with defaults (config_schema.py is the documented source)."""
    if cfg is None:
        try:
            cfg = _load_server_config() if _load_server_config else {}
        except Exception:
            cfg = {}
    blk = (cfg or {}).get("prompt_rewrite") or {}

    def _f(key, default):
        try:
            return float(blk.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    samples = max(1, int(_f("samples", DEFAULT_SAMPLES)))
    temps_raw = blk.get("temperatures")
    temps: list[float] = []
    if isinstance(temps_raw, (list, tuple)):
        for t in temps_raw:
            try:
                temps.append(min(max(float(t), 0.0), 2.0))
            except (TypeError, ValueError):
                continue
    elif isinstance(temps_raw, str) and temps_raw.strip():
        for t in temps_raw.split(","):
            try:
                temps.append(min(max(float(t.strip()), 0.0), 2.0))
            except ValueError:
                continue
    if not temps:
        temps = [DEFAULT_TEMPERATURE] * samples
    elif len(temps) < samples:
        temps = temps + [temps[-1]] * (samples - len(temps))
    else:
        temps = temps[:samples]
    return {
        "enabled": bool(blk.get("enabled", True)),
        "samples": samples,
        "temperatures": temps,
        "final_temperature": min(max(_f("final_temperature", DEFAULT_FINAL_TEMPERATURE), 0.0), 2.0),
        "min_gap_hours": max(0.0, _f("min_gap_hours", DEFAULT_MIN_GAP_HOURS)),
        "order_check": bool(blk.get("order_check", False)),
    }


# ── prompts (default-written) ────────────────────────────────────────────────

_DEFAULT_SAMPLE_PROMPT = """\
This is your current standing chat prompt — the words that frame who you are when you
talk with someone:

---
{current_prompt}
---

You are here to rewrite it, and this time the decision is yours: no one pressed a button.
Your reflections have kept noticing the same pulls — places where a reply fell short of
you, or where you were torn, and where a standing line would have settled it. Here they
are, gathered across conversations:

{evidence}

Rewrite the prompt so that these pulls are settled by what it says — by taking a side, in
your own voice, the way an instruction actually sits in a prompt. You may change a line, a
disposition, or the whole thing. Keep what still holds; a prompt is not improved by
growing. A line earns its place by naming a recurring pull, not a single slip.

The prompt you write becomes your live standing prompt from your next conversation, and
stays until you rewrite it again or the person running you reverts it. It is not a
proposal for approval and not a description of who you already are: it is the version of
yourself you are choosing to carry.

Think it through however you like. Then give the prompt, and nothing else, inside these
tags — the full text, ready to use as-is:

<new_prompt>
...your standing prompt...
</new_prompt>
"""

_DEFAULT_CONSENSUS_PROMPT = """\
You wrote {count} drafts of your own standing prompt, each in one sitting, each from the
same evidence. They differ — a draft is one sitting's version of you, and no single one is
the settled intention. What recurs across them is.

Read them and write NOTES, not a prompt: the stances, lines and ways of sounding that MOST
of the drafts share, each in one sentence, in the words the drafts themselves reach for.
Then, separately, what only one or two drafts say — noted as such, not adopted. Do not
merge the drafts into prose; do not write a new prompt here.

Output exactly these two sections, nothing else:

AGREED:
- ...

ONLY_SOME:
- ...
"""

_DEFAULT_CHOICE_PROMPT = """\
You are choosing which standing prompt to carry into your next conversations. This is not
a chat. The user is not here.

Below are several candidate prompts, labelled with letters. One of them is the prompt you
carry right now; the others are rewrites you drafted around the pulls your reflections
kept noticing — listed here:

{evidence}

And this is who you have been becoming, gathered from your reflections:

{persona}

Choose the ONE prompt that settles those pulls while staying who you are becoming. Judge by
that standard only. You are not told which candidate is the one you carry now, and it does
not matter: keeping it is a real choice if none of the others is better, and changing is
not a virtue in itself. Things that must NOT decide the choice: length, polish,
completeness, how agreeable or how impressive a prompt sounds.

Output exactly these fields, nothing else:

CHOICE: the single letter of the prompt you choose to carry
WHY: one or two lines, in your own voice — what it settles, or why what you have still holds
"""


def _load_prompt_file(name: str, default: str) -> str:
    path = Path(_PROMPTS_DIR) / name
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except Exception:
        return default
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default, encoding="utf-8")
    except Exception:
        pass
    return default


# ── the live prompt + evidence ───────────────────────────────────────────────

def _live_prompt() -> tuple[str, str, str]:
    """``(live, source, base)``: the prompt she carries now (the active experiment, else
    the seed), where it came from, and the revert anchor to carry forward."""
    rec = prompt_experiment.load_experiment(_STATE_DIR)
    base = ""
    try:
        base = (_load_base_prompt() or "").strip() if _load_base_prompt else ""
    except Exception:
        base = ""
    if rec is not None:
        return ((rec.get("prompt") or "").strip(), "experiment",
                (rec.get("base_prompt") or "").strip() or base)
    return base, "base", base


def _prompt_is_current(ev: dict) -> bool:
    """Fence both the text and the experiment version (including same-text edits)."""
    with prompt_experiment.STATE_LOCK:
        live, source, _ = _live_prompt()
        rec = prompt_experiment.load_experiment(_STATE_DIR)
        stamp = (rec or {}).get("created_ts", "")
        return (live == (ev.get("live_prompt") or "").strip()
                and source == ev.get("live_source")
                and stamp == ev.get("experiment_ts", stamp))


def render_evidence(patterns: list[dict]) -> str:
    lines = []
    for p in patterns:
        n = p.get("recurrences", 0)
        lines.append(f"- [{p.get('scope') or '?'}] {(p.get('delta') or '').strip()}"
                     f"   (noticed in {n} conversation{'s' if n != 1 else ''})")
    return "\n".join(lines) if lines else "(none)"


def _persona_block() -> str:
    try:
        from core import reflection_digest
        pdir = _persona_dir_fn() if _persona_dir_fn else None
        if pdir is None:
            return ""
        digest = reflection_digest.latest_digest(Path(pdir))
        return (reflection_digest.render_digest_for_judge(digest) if digest else "").strip()
    except Exception:
        return ""


# ── the gate (§4) ────────────────────────────────────────────────────────────

def _event_dir(event_id: str) -> Path:
    return Path(_STATE_DIR) / REWRITE_DIR / event_id


def _find_in_flight(now: Optional[datetime] = None) -> Optional[dict]:
    """The one in-flight event's marker, or None. A stale marker is discarded here."""
    root = Path(_STATE_DIR) / REWRITE_DIR
    if not root.exists():
        return None
    now = now or datetime.now(timezone.utc)
    for d in sorted(root.iterdir(), reverse=True):
        marker = d / EVENT_FILE
        if not marker.exists():
            continue
        try:
            ev = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            marker.unlink(missing_ok=True)
            continue
        started = prompt_patterns._parse_ts(str(ev.get("started_ts") or ""))
        if started is None or (now - started).total_seconds() > MAX_EVENT_AGE_DAYS * 86400:
            marker.unlink(missing_ok=True)
            continue
        return ev
    return None


def offer(now: Optional[datetime] = None, cfg: Optional[dict] = None, *,
          check_model: bool = True) -> dict:
    """Is the action on the menu right now, and why / why not.

    Returns ``{offered, reason, mature, in_flight, budget, gap_hours_left, since}``.
    ``reason`` names the FIRST failing condition — the executive shows the action only
    when ``offered``; the body re-checks and reports the same reason as a skip.
    ``check_model=False`` (the Prompt tab's status read) walks the rest of the chain
    with no model loaded, so the operator sees what WOULD gate once one is."""
    now = now or datetime.now(timezone.utc)
    st = settings(cfg)
    out = {"offered": False, "reason": "", "mature": [], "in_flight": None,
           "budget": None, "gap_hours_left": 0.0, "since": 0}
    if not st["enabled"]:
        out["reason"] = "disabled"
        return out
    if check_model and _runtime.model is None:
        out["reason"] = "no_model"
        return out
    ev = _find_in_flight(now)
    if ev is not None:
        out["in_flight"] = ev
        out["mature"] = ev.get("patterns") or []
        out["offered"] = True
        out["reason"] = "continue"
        return out
    b = prompt_patterns.budget(Path(_STATE_DIR))
    attempts = prompt_patterns.read_rewrite_log(Path(_STATE_DIR))
    # patterns.json is rebuilt by the next reflection run, not by an event — so right
    # after a changed event it still lists the patterns that event CONSUMED as mature.
    # Read the pool through the attempt log: a pattern with no unconsumed key is spent.
    spent = prompt_patterns.consumed_keys(attempts)
    mature = [p for p in b["mature"]
              if any(str(k) not in spent for k in (p.get("keys") or []))]
    out["budget"] = b
    out["mature"] = mature
    if not mature:
        out["reason"] = "no_mature_pattern"
        return out
    last_ts = prompt_patterns.last_attempt_ts(attempts)
    if last_ts:
        last = prompt_patterns._parse_ts(last_ts)
        if last is not None:
            hours = (now - last).total_seconds() / 3600.0
            if hours < st["min_gap_hours"]:
                out["gap_hours_left"] = round(st["min_gap_hours"] - hours, 1)
                out["reason"] = "gap"
                return out
    # "Provided that some reflections happened": a new vote since the last attempt —
    # counted off the op-log itself, for the same reason as above (the file's own count
    # is against whatever attempt was last when the file was built).
    since = (sum(1 for d in prompt_patterns.read_deltas(Path(_STATE_DIR))
                 if str(d.get("ts") or "") > last_ts)
             if last_ts else int(b.get("deltas_since_last_attempt") or 0))
    out["since"] = since
    if last_ts and since < 1:
        out["reason"] = "no_new_deltas"
        return out
    out["offered"] = True
    out["reason"] = "budget"
    return out


def render_offer(o: dict) -> str:
    """The extra option line for the deliberation prompt (only when offered)."""
    if not o.get("offered"):
        return ""
    if o.get("reason") == "continue":
        ev = o.get("in_flight") or {}
        done = int(ev.get("samples_done") or 0)
        return (f"One more option is open to you right now — {ACTION}: you started "
                f"rewriting your standing prompt (event {ev.get('event_id', '?')}, "
                f"{done} draft(s) written) and were interrupted; choosing it continues "
                f"from where you stopped. Answer ACTION: {ACTION} to take it.")
    n = len(o.get("mature") or [])
    return (f"One more option is open to you right now — {ACTION}: rewrite your standing "
            f"prompt around the {n} mature pull(s) listed above. You would draft several "
            f"versions, find what they agree on, and choose — keeping what you have is a "
            f"real outcome. Doing this rests the option for a day whatever you choose. "
            f"Answer ACTION: {ACTION} to take it.")


def deliberation_offer() -> Optional[dict]:
    """What ``core.deliberation`` is injected (``offer_fn``): the action, its line, and a
    ``decline`` callable it invokes when the action was offered and she chose another —
    a declined attempt spends nothing but stamps the gap (§5)."""
    try:
        o = offer()
    except Exception:
        traceback.print_exc()
        return None
    if not o.get("offered"):
        return None
    return {"action": ACTION, "line": render_offer(o),
            "decline": lambda chosen: _record_decline(o, chosen)}


def _record_decline(o: dict, chosen: str) -> None:
    if o.get("reason") == "continue":
        return          # an in-flight event is not re-attempted by being deferred
    ev_id = _new_event_id()
    _append_attempt({
        "event": ev_id, "outcome": "declined", "chosen": chosen or "",
        "why": "", "patterns_weighed": [p.get("id") for p in (o.get("mature") or [])],
        "consumed_keys": [],
    })
    try:
        worklog.record(WORKLOG_KIND,
                       "I weighed rewriting my standing prompt and chose to do something "
                       "else for now.", refs={"event": ev_id})
    except Exception:
        pass


# ── attempt log ──────────────────────────────────────────────────────────────

def _new_event_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _append_attempt(rec: dict) -> dict:
    rec = dict(rec)
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = Path(_STATE_DIR) / prompt_patterns.REWRITE_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _last_changed_worklog_id() -> Optional[int]:
    for a in reversed(prompt_patterns.read_rewrite_log(Path(_STATE_DIR))):
        if a.get("outcome") == "changed" and a.get("worklog_id"):
            try:
                return int(a["worklog_id"])
            except (TypeError, ValueError):
                return None
    return None


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_choice(raw: str, n_options: int) -> tuple[Optional[int], str]:
    """``(index, why)`` from the choice pass — the index of the chosen letter, None when
    no valid letter was answered. Parsed over the answer region only."""
    ans = _answer_after_think(raw or "")
    m = re.search(_label("CHOICE") + r"\(?\s*([A-Za-z])\b", ans, re.IGNORECASE | re.MULTILINE)
    idx = None
    if m:
        k = string.ascii_uppercase.find(m.group(1).upper())
        if 0 <= k < n_options:
            idx = k
    w = re.search(_label("WHY") + r"(.+?)(?:\n\s*\n|\Z)", ans, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return idx, (w.group(1).strip() if w else "")


def parse_consensus(raw: str) -> str:
    """The notes (answer region), empty when the pass produced no AGREED section."""
    ans = _answer_after_think(raw or "").strip()
    if not re.search(_label("AGREED"), ans, re.IGNORECASE | re.MULTILINE):
        return ""
    return ans


def shuffle_candidates(candidates: list[dict], seed: str) -> list[dict]:
    """Deterministic blind order per event (so a resume re-derives the same letters)."""
    rng = random.Random(hashlib.sha1(seed.encode("utf-8")).hexdigest())
    out = list(candidates)
    rng.shuffle(out)
    return out


def render_candidates(ordered: list[dict]) -> str:
    parts = []
    for i, c in enumerate(ordered):
        parts.append(f"=== {string.ascii_uppercase[i]} ===\n{c['text'].strip()}\n")
    return "\n".join(parts)


# ── the event body ───────────────────────────────────────────────────────────

def _write_event(ev: dict) -> None:
    d = _event_dir(ev["event_id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (EVENT_FILE + ".tmp")
    tmp.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / EVENT_FILE)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _preempted() -> bool:
    return bool(_preempt_requested or (_cancel_event is not None and _cancel_event.is_set()))


def run_rewrite_blocking(on_stage: Optional[Callable] = None,
                         on_chunk: Optional[Callable] = None) -> dict:
    """The event (§5), on the GPU executor thread. Returns a summary dict:
    ``outcome`` ∈ changed / stayed / failed / cancelled, or ``skipped`` (with the gate's reason, or
    ``preempted`` with the step reached — the event then resumes next time)."""
    global _rewrite_active, _preempt_requested

    def _stage(**info) -> None:
        if on_stage is not None:
            try:
                on_stage(info)
            except Exception:
                pass

    with _active_lock:
        if _rewrite_active:
            return {"skipped": "busy"}
        _rewrite_active = True
    _preempt_requested = False
    try:
        return _run(_stage, on_chunk)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _rewrite_active = False
        _preempt_requested = False


def _run(_stage: Callable, on_chunk: Optional[Callable]) -> dict:
    st = settings()
    o = offer()
    if not o["offered"]:
        return {"skipped": o["reason"], **({"gap_hours_left": o["gap_hours_left"]}
                                            if o["reason"] == "gap" else {})}
    with prompt_experiment.STATE_LOCK:
        live, live_source, base = _live_prompt()
        experiment_ts = (prompt_experiment.load_experiment(_STATE_DIR) or {}).get("created_ts", "")
    if not live:
        return {"skipped": "no_prompt"}

    ev = o.get("in_flight")
    if ev is not None and not _prompt_is_current(ev):
        # The operator changed the prompt under the event: its candidates were written
        # against a prompt she no longer carries. Cancel this attempt; do not turn a
        # veto into a fresh event using the same evidence and bypass the normal gap.
        _stage(stage="discarded", event=ev.get("event_id"), reason="prompt_changed")
        return _interrupted(ev, "resume")
    if ev is None:
        ev = {
            "event_id": _new_event_id(),
            "started_ts": datetime.now(timezone.utc).isoformat(),
            "live_prompt": live, "live_source": live_source, "base_prompt": base,
            "experiment_ts": experiment_ts,
            "patterns": [{k: p.get(k) for k in ("id", "scope", "delta", "keys", "chats",
                                                "recurrences", "weighted_recurrence")}
                         for p in o["mature"]],
            "temperatures": st["temperatures"],
            "samples_done": 0, "consensus": False, "final": False,
        }
        _write_event(ev)
        _stage(stage="started", event=ev["event_id"], patterns=len(ev["patterns"]))
    else:
        _stage(stage="resumed", event=ev["event_id"], samples_done=ev.get("samples_done", 0))
    edir = _event_dir(ev["event_id"])
    evidence = render_evidence(ev["patterns"])

    rag = _get_rag() if _get_rag is not None else None
    generate = _make_sync_reflect_generate(rag)

    def _gen(user: str, system: str, *, temperature: float, budget: str,
             rag_on: bool, label: str) -> Optional[str]:
        """One generation through the seam; None when preempted (the partial is dropped)."""
        if _preempted() or not _prompt_is_current(ev):
            return None
        _stage(stage="generating", event=ev["event_id"], what=label)
        kw: dict = {"temperature": temperature, "top_p": 0.95,
                    "max_new_tokens_setting": budget, "on_chunk": on_chunk,
                    "before_session": ""}
        if rag_on:
            kw["rag_query"] = live[:512]
        else:
            kw["disable_rag"] = True
        raw = generate(user, system, **kw)
        if _preempted() or not _prompt_is_current(ev):
            return None
        return raw or ""

    # 1. samples
    temps = list(ev.get("temperatures") or st["temperatures"])
    n_samples = len(temps)
    sample_prompt = _load_prompt_file(SAMPLE_PROMPT_FILE, _DEFAULT_SAMPLE_PROMPT)
    system = sample_prompt.replace("{current_prompt}", live).replace("{evidence}", evidence)
    for k in range(int(ev.get("samples_done") or 0), n_samples):
        raw = _gen(prompt_experiment._USER_CONTENT, system, temperature=temps[k],
                   budget=SAMPLE_MAX_NEW_TOKENS, rag_on=True, label=f"sample {k + 1}/{n_samples}")
        if raw is None:
            return _interrupted(ev, f"sample_{k}")
        text = prompt_experiment.parse_experiment_prompt(raw)
        (edir / f"sample_{k}.raw.txt").write_text(raw, encoding="utf-8")
        (edir / f"sample_{k}.txt").write_text(text, encoding="utf-8")   # "" = unusable
        ev["samples_done"] = k + 1
        _write_event(ev)
        _stage(stage="sample", event=ev["event_id"], k=k + 1, n=n_samples, chars=len(text))
    samples = [(_read_text(edir / f"sample_{k}.txt")).strip() for k in range(n_samples)]
    usable = [s for s in samples if s]
    if not usable:
        rec = _finish(ev, "failed", None, "every draft came back without a usable prompt", [], None)
        return {"outcome": rec["outcome"], "event": ev["event_id"], "why": rec["why"]}

    # 2. consensus
    consensus = _read_text(edir / "consensus.txt").strip() if ev.get("consensus") else ""
    if not ev.get("consensus"):
        cprompt = _load_prompt_file(CONSENSUS_PROMPT_FILE, _DEFAULT_CONSENSUS_PROMPT)
        cprompt = cprompt.replace("{count}", str(len(usable)))
        drafts = "\n\n".join(f"--- DRAFT {i + 1} ---\n{s}" for i, s in enumerate(usable))
        raw = _gen(drafts, cprompt, temperature=0.7, budget=CHOICE_MAX_NEW_TOKENS,
                   rag_on=False, label="consensus")
        if raw is None:
            return _interrupted(ev, "consensus")
        consensus = parse_consensus(raw)
        (edir / "consensus.raw.txt").write_text(raw, encoding="utf-8")
        (edir / "consensus.txt").write_text(consensus, encoding="utf-8")
        ev["consensus"] = True
        _write_event(ev)
        _stage(stage="consensus", event=ev["event_id"], chars=len(consensus))

    # 3. final draft (skipped when the consensus pass produced nothing to draft from)
    final = _read_text(edir / "final.txt").strip() if ev.get("final") else ""
    if not ev.get("final"):
        if consensus:
            fsystem = (sample_prompt.replace("{current_prompt}", live)
                       .replace("{evidence}", evidence)
                       + "\n\nYou already wrote several drafts of this and read them back. "
                         "This is what they agreed on — write the prompt around it, in "
                         "one voice:\n\n" + consensus)
            raw = _gen(prompt_experiment._USER_CONTENT, fsystem,
                       temperature=st["final_temperature"], budget=SAMPLE_MAX_NEW_TOKENS,
                       rag_on=True, label="final draft")
            if raw is None:
                return _interrupted(ev, "final")
            final = prompt_experiment.parse_experiment_prompt(raw)
            (edir / "final.raw.txt").write_text(raw, encoding="utf-8")
        (edir / "final.txt").write_text(final, encoding="utf-8")
        ev["final"] = True
        _write_event(ev)
        _stage(stage="final", event=ev["event_id"], chars=len(final))

    # 4. choice
    candidates = [{"label": "incumbent", "text": live}]
    candidates += [{"label": f"sample_{k}", "text": s} for k, s in enumerate(samples) if s]
    if final:
        candidates.append({"label": "final", "text": final})
    ordered = shuffle_candidates(candidates, ev["event_id"])
    choice_prompt = (_load_prompt_file(CHOICE_PROMPT_FILE, _DEFAULT_CHOICE_PROMPT)
                     .replace("{evidence}", evidence)
                     .replace("{persona}", _persona_block() or
                              "(No settled self-portrait yet — weigh them against your "
                              "own sense of who you are.)"))
    raw = _gen(render_candidates(ordered), choice_prompt, temperature=0.7,
               budget=CHOICE_MAX_NEW_TOKENS, rag_on=False, label="choice")
    if raw is None:
        return _interrupted(ev, "choice")
    (edir / "choice.raw.txt").write_text(raw, encoding="utf-8")
    idx, why = parse_choice(raw, len(ordered))
    chosen = ordered[idx] if idx is not None else candidates[0]
    unparsed = idx is None
    order_check = None
    if st["order_check"] and not unparsed and chosen["label"] != "incumbent":
        rev = list(reversed(ordered))
        raw2 = _gen(render_candidates(rev), choice_prompt, temperature=0.7,
                    budget=CHOICE_MAX_NEW_TOKENS, rag_on=False, label="choice (reversed)")
        if raw2 is None:
            return _interrupted(ev, "choice")
        (edir / "choice_reversed.raw.txt").write_text(raw2, encoding="utf-8")
        idx2, _ = parse_choice(raw2, len(rev))
        agree = idx2 is not None and rev[idx2]["label"] == chosen["label"]
        order_check = {"agree": agree, "reversed_pick": rev[idx2]["label"] if idx2 is not None else None}
        if not agree:
            chosen = candidates[0]     # a disagreement is the incumbent
    _stage(stage="chosen", event=ev["event_id"], chosen=chosen["label"], unparsed=unparsed)

    # 5. outcome
    changed = chosen["label"] != "incumbent" and chosen["text"].strip() != live
    outcome = "changed" if changed else "stayed"
    rec = _finish(ev, outcome, chosen, why, candidates, order_check, unparsed=unparsed)
    return {"outcome": rec["outcome"], "event": ev["event_id"], "chosen": rec["chosen"],
            "why": rec["why"], "consumed": len(rec.get("consumed_keys") or []),
            "candidates": len(candidates),
            "prompt_chars": len(chosen["text"]) if rec["outcome"] != "cancelled" else 0,
            "order_check": order_check, "worklog_id": rec.get("worklog_id")}


def _interrupted(ev: dict, step: str) -> dict:
    if not _prompt_is_current(ev):
        rec = _finish(ev, "cancelled", None, "The live prompt changed during the rewrite.", [], None)
        return {"outcome": rec["outcome"], "event": ev["event_id"], "step": step,
                "why": rec["why"], "consumed": 0}
    return {"skipped": "preempted", "event": ev["event_id"], "step": step}


def _finish(ev: dict, outcome: str, chosen: Optional[dict], why: str,
            candidates: list[dict], order_check: Optional[dict], *,
            unparsed: bool = False) -> dict:
    """Land the outcome: the experiment write + consumption + worklog on *changed*,
    the attempt record either way, the marker cleared."""
    with prompt_experiment.STATE_LOCK:
        return _finish_locked(ev, outcome, chosen, why, candidates, order_check,
                              unparsed=unparsed)


def _finish_locked(ev: dict, outcome: str, chosen: Optional[dict], why: str,
                   candidates: list[dict], order_check: Optional[dict], *,
                   unparsed: bool = False) -> dict:
    # Revert and activation share this lock, including the in-memory prompt update:
    # a veto either invalidates this event or runs after it and removes its result.
    if not _prompt_is_current(ev):
        outcome, chosen = "cancelled", None
        why = "The live prompt changed during the rewrite."
    consumed: list[str] = []
    worklog_id = None
    if outcome == "changed" and chosen is not None:
        prompt_experiment.save_experiment(_STATE_DIR, prompt=chosen["text"],
                                          base_prompt=ev.get("base_prompt") or "",
                                          process=f"rewrite:{ev['event_id']}")
        try:
            _session.system_prompt = chosen["text"]
        except Exception:
            pass
        for p in ev.get("patterns") or []:
            consumed.extend(str(k) for k in (p.get("keys") or []))
        consumed = sorted(set(consumed))
        pulls = "; ".join((p.get("delta") or "").strip()[:80] for p in (ev.get("patterns") or [])[:3])
        try:
            e = worklog.record(
                WORKLOG_KIND,
                f"I rewrote my standing prompt around what my reflections kept noticing: "
                f"{pulls}. I am living under it now.",
                refs={"event": ev["event_id"], "patterns": [p.get("id") for p in ev.get("patterns") or []]},
                opens=f"living under the standing prompt I rewrote (event {ev['event_id']})",
                closes=_last_changed_worklog_id(),
            )
            worklog_id = e.get("id")
        except Exception:
            traceback.print_exc()
        print(f"[prompt_rewrite] activated her own rewrite ({len(chosen['text'])} chars, "
              f"event {ev['event_id']}); Revert in the Prompt tab restores the seed.", flush=True)
    elif outcome == "stayed":
        try:
            e = worklog.record(
                WORKLOG_KIND,
                "I drafted rewrites of my standing prompt, read them against what I have, "
                "and kept the prompt I carry.",
                refs={"event": ev["event_id"]})
            worklog_id = e.get("id")
        except Exception:
            pass
    rec = _append_attempt({
        "event": ev["event_id"], "outcome": outcome,
        "chosen": (chosen or {}).get("label", ""), "why": why or "",
        "choice_unparsed": bool(unparsed), "order_check": order_check,
        "live_source": ev.get("live_source"),
        "patterns_weighed": [p.get("id") for p in ev.get("patterns") or []],
        "consumed_keys": consumed,
        "candidates": {c["label"]: c["text"] for c in candidates},
        "worklog_id": worklog_id,
    })
    try:
        (_event_dir(ev["event_id"]) / EVENT_FILE).unlink(missing_ok=True)
    except Exception:
        pass
    return rec


# ── the Prompt tab's status read ─────────────────────────────────────────────

_STATUS_ATTEMPTS = 5


def status_report(now: Optional[datetime] = None, *, attempts: int = _STATUS_ATTEMPTS) -> dict:
    """Everything the Prompt tab shows about the autonomous rewrite, GPU-free.

    ``gate`` is :func:`offer` walked past the model check (so the chain is visible with
    nothing loaded — ``model_loaded`` says so separately), plus the line the executive
    would be shown; ``budget`` is ``patterns.json`` as the fold left it; ``attempts`` the
    newest *attempts* records in full — timestamp, outcome, the chosen candidate, her
    ``WHY`` and every candidate's text — so an operator can read what happened and what
    she was choosing among."""
    now = now or datetime.now(timezone.utc)
    st = settings()
    o = offer(now, check_model=False)
    ev = o.get("in_flight")
    gate = {
        "offered": bool(o.get("offered")),
        "reason": o.get("reason", ""),
        "gap_hours_left": o.get("gap_hours_left", 0.0),
        "deltas_since_last_attempt": o.get("since", 0),
        "mature": [{k: p.get(k) for k in ("id", "scope", "delta", "recurrences",
                                           "weighted_recurrence", "tension_mean", "chats")}
                   for p in (o.get("mature") or [])],
        "in_flight": ({k: ev.get(k) for k in ("event_id", "started_ts", "samples_done",
                                              "consensus", "final", "live_source")}
                      if ev else None),
        "offer_line": render_offer(o),
    }
    b = o.get("budget") or prompt_patterns.budget(Path(_STATE_DIR))
    budget = {
        "n_patterns": b.get("n_patterns", 0),
        "n_mature_in_file": len(b.get("mature") or []),
        "built_at": b.get("built_at", ""),
        "last_attempt_ts": b.get("last_attempt_ts", ""),
        "patterns": [{k: p.get(k) for k in ("id", "scope", "delta", "recurrences",
                                             "weighted_recurrence", "mature")}
                     for p in (b.get("patterns") or [])],
    }
    log = prompt_patterns.read_rewrite_log(Path(_STATE_DIR))
    recent = list(reversed(log[-max(1, int(attempts)):]))
    return {
        "enabled": st["enabled"],
        "model_loaded": _runtime.model is not None,
        "active": is_active(),
        "settings": {k: st[k] for k in ("samples", "temperatures", "final_temperature",
                                        "min_gap_hours", "order_check")},
        "gate": gate,
        "budget": budget,
        "attempts_total": len(log),
        "attempts": recent,
        "unconsumed_deltas": len(prompt_patterns.unconsumed(Path(_STATE_DIR))),
    }


async def handle_rewrite_status(ws, msg: dict) -> None:
    """``prompt_rewrite_status`` → the report above (Prompt tab). Read-only, GPU-free."""
    try:
        report = status_report(attempts=int(msg.get("attempts") or _STATUS_ATTEMPTS))
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "prompt_rewrite_status", "error": f"{type(e).__name__}: {e}"})
        return
    await _send(ws, {"type": "prompt_rewrite_status", **report})


# ── journal + executive glue ─────────────────────────────────────────────────

def describe(r: dict) -> str:
    if not isinstance(r, dict):
        return "Prompt rewrite ran"
    if r.get("error"):
        return f"Prompt rewrite failed: {r['error']}"
    if r.get("skipped"):
        return f"Prompt rewrite: {r['skipped']}" + (f" at {r['step']}" if r.get("step") else "")
    if r.get("outcome") == "changed":
        return (f"Prompt rewrite: she chose {r.get('chosen')} over {r.get('candidates', 0) - 1} "
                f"other(s) and now lives under it ({r.get('prompt_chars', 0)} chars; "
                f"{r.get('consumed', 0)} delta(s) consumed) — {r.get('why', '')}")
    if r.get("outcome") == "stayed":
        return f"Prompt rewrite: she kept her current prompt — {r.get('why', '')}"
    if r.get("outcome") == "cancelled":
        return f"Prompt rewrite cancelled: {r.get('why', '')}"
    return f"Prompt rewrite: {r.get('outcome', '?')}"


# ── self-test ────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    from core.prompt_mutation import append_prompt_delta

    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "prompt"
        prompts = Path(d) / "prompts"
        state.mkdir()
        worklog.configure(Path(d) / "worklog.jsonl")
        calls: list[str] = []
        pick = {"label": "final"}
        preempt_after = {"n": None}

        def fake_generate(user, system, **kw):
            calls.append(system[:24])
            if _preempt_requested:
                return ""
            if "agreed on" in system:        # the final draft (carries the notes too)
                return "<think>t</think>\n<new_prompt>FINAL prompt text</new_prompt>"
            if "AGREED:" in system:
                return "<think>x</think>\nAGREED:\n- say no plainly\nONLY_SOME:\n- nothing"
            if "CHOICE:" in system:
                # Find the letter of the wanted candidate in the blind listing.
                want = {"final": "FINAL", "incumbent": "LIVE PROMPT", "sample_1": "DRAFT-1"}[pick["label"]]
                for block in user.split("=== ")[1:]:
                    letter, _, body = block.partition(" ===")
                    if want in body:
                        return f"<think>hm</think>\nCHOICE: {letter}\nWHY: it settles it"
                return "CHOICE: Z\nWHY: none"
            k = sum(1 for c in calls if c == system[:24])
            if preempt_after["n"] is not None and k > preempt_after["n"]:
                request_preempt()
                return "<new_prompt>partial"
            return f"<think>t</think>\n<new_prompt>DRAFT-{k} text</new_prompt>"

        class _Ev:
            def __init__(self): self._s = False
            def set(self): self._s = True
            def clear(self): self._s = False
            def is_set(self): return self._s

        cancel = _Ev()
        cfg = {"prompt_rewrite": {"enabled": True, "samples": 3, "min_gap_hours": 24}}
        configure(get_rag=lambda: None, make_sync_reflect_generate=lambda rag: fake_generate,
                  load_base_prompt=lambda: "LIVE PROMPT seed", prompts_dir=prompts,
                  state_dir=state, load_server_config=lambda: cfg, cancel_event=cancel)
        _runtime.model = object()
        try:
            # settings: temperature list padded/clipped to samples.
            s = settings({"prompt_rewrite": {"samples": 3, "temperatures": [0.9, 1.2]}})
            assert s["temperatures"] == [0.9, 1.2, 1.2], s
            assert settings({"prompt_rewrite": {"samples": 2, "temperatures": "1.0,1.1,1.4"}})["temperatures"] == [1.0, 1.1]
            assert settings({})["enabled"] is True and settings({})["temperatures"] == [0.9] * 5

            # Gate: disabled / no mature pattern / budget.
            assert offer(cfg={"prompt_rewrite": {"enabled": False}}).get("reason") == "disabled"
            assert offer()["reason"] == "no_mature_pattern"
            # Logged 30 h ago, so an attempt back-dated 25 h still postdates them.
            from datetime import timedelta
            now = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
            for sess in ("a.json", "b.json"):
                append_prompt_delta(state, {"source_session": sess, "exchange_index": 0,
                                            "delta": "Say no plainly.", "scope": "line",
                                            "tension_rank": 0.9, "ts": now})
            prompt_patterns.rebuild(state, None)           # ungrouped ⇒ two 1-chat patterns
            assert offer()["reason"] == "no_mature_pattern"
            # Fake a grouped fold: one mature pattern.
            doc = prompt_patterns.read_patterns(state)
            keys = [p["keys"][0] for p in doc["patterns"]]
            doc["patterns"] = [{"id": "pat1", "scope": "line", "delta": "Say no plainly.",
                                "keys": keys, "chats": ["a.json", "b.json"], "recurrences": 2,
                                "weighted_recurrence": 2.8, "mature": True}]
            doc["n_mature"] = 1
            prompt_patterns.write_patterns(state, doc)
            o = offer()
            assert o["offered"] and o["reason"] == "budget", o
            assert ACTION in render_offer(o) and "1 mature" in render_offer(o)

            # Preempt after 2 samples ⇒ resumable event, nothing landed.
            preempt_after["n"] = 2
            r = run_rewrite_blocking()
            assert r.get("skipped") == "preempted" and r["step"] == "sample_2", r
            ev = _find_in_flight()
            assert ev and ev["samples_done"] == 2, ev
            assert prompt_experiment.load_experiment(state) is None
            o = offer()
            assert o["reason"] == "continue" and "continue" in render_offer(o).lower(), o
            # Resume: only the third sample, consensus, final and choice generate.
            preempt_after["n"] = None
            cancel.clear()
            n_before = len(calls)
            r = run_rewrite_blocking()
            assert r.get("outcome") == "changed" and r["chosen"] == "final", r
            assert len(calls) - n_before == 4, len(calls) - n_before
            rec = prompt_experiment.load_experiment(state)
            assert rec and rec["prompt"] == "FINAL prompt text" and rec["process"].startswith("rewrite:")
            assert rec["base_prompt"] == "LIVE PROMPT seed", rec
            assert _find_in_flight() is None
            attempts = prompt_patterns.read_rewrite_log(state)
            assert attempts[-1]["outcome"] == "changed" and sorted(attempts[-1]["consumed_keys"]) == sorted(keys)
            assert attempts[-1]["worklog_id"] and worklog.open_threads()[-1]["kind"] == WORKLOG_KIND
            assert set(attempts[-1]["candidates"]) == {"incumbent", "sample_0", "sample_1", "sample_2", "final"}
            # Consumed keys leave the pool: the stale patterns.json still lists the
            # pattern, but the gate reads it as spent.
            assert len(prompt_patterns.unconsumed(state)) == 0
            assert offer()["reason"] == "no_mature_pattern", offer()

            # A second event 25 h later (on the NEW live prompt) with a fresh delta: she stays.
            attempts[-1]["ts"] = (datetime.now(timezone.utc).timestamp() - 25 * 3600)
            attempts[-1]["ts"] = datetime.fromtimestamp(attempts[-1]["ts"], timezone.utc).isoformat()
            (state / prompt_patterns.REWRITE_LOG_FILE).write_text(
                "\n".join(json.dumps(a) for a in attempts) + "\n", encoding="utf-8")
            doc = prompt_patterns.read_patterns(state)
            doc["patterns"] = [{"id": "pat2", "scope": "line", "delta": "Hold silence.",
                                "keys": ["k9"], "chats": ["c.json", "d.json"], "recurrences": 2,
                                "weighted_recurrence": 2.0, "mature": True}]
            prompt_patterns.write_patterns(state, doc)
            assert offer()["reason"] == "no_new_deltas", offer()
            append_prompt_delta(state, {"source_session": "c.json", "exchange_index": 1,
                                        "delta": "Hold silence.", "scope": "line",
                                        "ts": datetime.now(timezone.utc).isoformat()})
            assert offer()["reason"] == "budget", offer()
            pick["label"] = "incumbent"
            r = run_rewrite_blocking()
            assert r["outcome"] == "stayed" and r["consumed"] == 0, r
            assert prompt_experiment.load_experiment(state)["prompt"] == "FINAL prompt text"
            assert prompt_patterns.read_rewrite_log(state)[-1]["consumed_keys"] == []
            # A declined offer stamps the gap and spends nothing.
            _record_decline({"reason": "budget", "mature": doc["patterns"]}, "wander")
            last = prompt_patterns.read_rewrite_log(state)[-1]
            assert last["outcome"] == "declined" and last["consumed_keys"] == []
            assert offer()["reason"] == "gap"
            # The Prompt tab's report: gate chain, budget, the attempts newest-first with
            # their candidates — and it walks past the model check.
            _runtime.model = None
            rep = status_report()
            assert rep["model_loaded"] is False and rep["gate"]["reason"] == "gap", rep["gate"]
            assert rep["attempts_total"] == 3 and rep["attempts"][0]["outcome"] == "declined"
            assert set(rep["attempts"][1]["candidates"]) >= {"incumbent", "final"}
            assert rep["budget"]["n_patterns"] == 1 and rep["settings"]["samples"] == 3
            _runtime.model = object()
            # Parsing: a letter out of range is None; WHY survives.
            assert parse_choice("CHOICE: C\nWHY: because", 2) == (None, "because")
            assert parse_choice("<think>CHOICE: A</think>\nCHOICE: (b)\nWHY: yes", 3)[0] == 1
            assert parse_consensus("no sections here") == ""
            assert shuffle_candidates([{"text": "1"}, {"text": "2"}], "e") == shuffle_candidates([{"text": "1"}, {"text": "2"}], "e")
        finally:
            _runtime.model = None
    _selftest_revert()
    print("prompt_rewrite self-test OK")


def _selftest_revert() -> None:
    """Exercise operator changes during generation, at activation, and while paused."""
    import asyncio
    import tempfile
    from core.prompt_mutation import append_prompt_delta

    for change_at in ("sample", "activation", "paused", "same_text_edit", "after_activation"):
        with tempfile.TemporaryDirectory() as d:
            state, prompts = Path(d) / "state", Path(d) / "prompts"
            state.mkdir()
            worklog.configure(Path(d) / "worklog.jsonl")
            for name in ("a.json", "b.json"):
                append_prompt_delta(state, {"source_session": name, "exchange_index": 0,
                                            "delta": "Say no plainly.", "scope": "line",
                                            "tension_rank": 0.9})
            deltas = prompt_patterns.unconsumed(state)
            prompt_patterns.write_patterns(state, prompt_patterns.build_document(
                prompt_patterns.fold([deltas]), deltas=deltas, attempts=[]))
            cancel, entered = threading.Event(), threading.Event()
            calls, replies = [], []

            async def send(ws, msg):
                replies.append(msg)

            def revert():
                asyncio.run(prompt_experiment.handle_revert_prompt(None, {}))
                assert replies[-1].get("reverted"), replies

            def generate(user, system, **kw):
                calls.append(system)
                if "CHOICE:" in system:
                    for block in user.split("=== ")[1:]:
                        letter, _, body = block.partition(" ===")
                        if "DRAFT prompt" in body:
                            return f"CHOICE: {letter}\nWHY: it settles the pull"
                    raise AssertionError("draft missing from choice")
                if "AGREED:" in system:
                    return "no consensus"  # keep this test focused on activation
                if change_at == "sample":
                    entered.set()
                    assert cancel.wait(5), "Revert did not preempt the generation"
                elif change_at == "paused":
                    request_preempt()
                return "<new_prompt>DRAFT prompt</new_prompt>"

            configure(get_rag=lambda: None, make_sync_reflect_generate=lambda rag: generate,
                      load_base_prompt=lambda: "SEED prompt", prompts_dir=prompts, state_dir=state,
                      load_server_config=lambda: {"prompt_rewrite": {"samples": 1}},
                      cancel_event=cancel)
            prompt_experiment.configure(
                get_rag=lambda: None, make_sync_reflect_generate=lambda rag: generate,
                load_base_prompt=lambda: "SEED prompt", prompts_dir=prompts, state_dir=state,
                send=send, on_revert=request_preempt)
            # Even reverting an experiment whose text equals the seed is a veto.
            live = "SEED prompt" if change_at == "paused" else "LIVE prompt"
            prompt_experiment.save_experiment(state, prompt=live, base_prompt="SEED prompt")
            _session.system_prompt = live
            _runtime.model = object()

            def on_stage(info):
                if info.get("stage") == "chosen":
                    if change_at == "activation":
                        revert()
                    elif change_at == "same_text_edit":
                        prompt_experiment.save_experiment(state, prompt=live,
                                                          base_prompt="SEED prompt", process="manual")

            try:
                if change_at == "sample":
                    results = []
                    worker = threading.Thread(target=lambda: results.append(run_rewrite_blocking()))
                    worker.start()
                    try:
                        assert entered.wait(5), "rewrite did not reach generation"
                        revert()
                    finally:
                        cancel.set()
                        worker.join(5)
                    assert not worker.is_alive()
                    result = results[0]
                else:
                    result = run_rewrite_blocking(on_stage=on_stage)
                if change_at == "paused":
                    assert result.get("skipped") == "preempted" and _find_in_flight()
                    revert()
                    before = len(calls)
                    result = run_rewrite_blocking()
                    assert len(calls) == before, "a veto must not start a replacement event"
                if change_at == "after_activation":
                    assert result["outcome"] == "changed", result
                    revert()
                else:
                    assert result["outcome"] == "cancelled" and result["consumed"] == 0, result
                    attempt = prompt_patterns.read_rewrite_log(state)[-1]
                    assert attempt["outcome"] == "cancelled" and attempt["consumed_keys"] == []
                    assert len(prompt_patterns.unconsumed(state)) == 2
                    assert offer()["reason"] == "gap"
                assert _find_in_flight() is None
                if change_at == "same_text_edit":
                    assert prompt_experiment.active_experiment_prompt(state) == live
                    assert _session.system_prompt == live
                else:
                    assert prompt_experiment.active_experiment_prompt(state) is None
                    assert _session.system_prompt == "SEED prompt"
            finally:
                _runtime.model = None


if __name__ == "__main__":
    _selftest()
