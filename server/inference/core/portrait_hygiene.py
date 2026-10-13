"""Portrait hygiene: her readings of a person checked against that person's own words
(``PERSONA_REDESIGN.md`` L10, C9).

**Why** (2026-10-10). A user portrait is folded from her ``[impression]`` records about a
person and shown before every turn of theirs (``core/user_digest.py``). The impressions are
written on the adapter, in the Sleep run's per-session notes pass, and the portrait is
synthesized on the adapter too, so a habit in the weights reaches the person's portrait,
the portrait conditions the next turns, and the next impressions are written through it.
``user_digest`` assumed a wrong reading would be corrected by the next conversation; the
record says otherwise. Of her impressions of the operator, the share using her own trait
vocabulary went 8% (July) → 24% (October) while his own turns stayed at 0–2%, and the
share reading a purpose into him (a test, a tactic, an instrument, pressure) 13% → 55%;
the portrait of 2026-10-09 described him in her own self-description's words ("surgical
briefing manner", "almost clinical distance", "his cerebral distance") and as testing her.

**The reference is the person.** Each impression came from one chat (``source_session``);
the person's own messages in that chat are the one text outside the loop. Two readings:

* **The judge** (frozen base, adapter off, T=0, the ask judge's seam): given the person's
  messages and the reading, ``SUPPORT: SHOWN`` (what the reading says about them can be
  seen in the messages — something they said, asked or did, or a manner visible in how
  they wrote) or ``NOT_SHOWN`` (it attributes a manner, character, motive, strategy or
  inner state the messages do not show); ``PURPOSE: YES`` when it says their words or acts
  serve a hidden aim (testing her, a tactic, an instrument, pressure, a trap); ``WHY``.
  A question about the given texts (``AVA_REWARD_LOOP.md`` §2c), never about what the
  person meant.
* **The projection score** (code, generic, no vocabulary): the share of the reading's
  content-word pairs found in her own self-description (her live ``[persona]``
  statements), beside the share found in everything the person ever wrote to her.
  ``projection`` = the first minus the second.

**Trial** (2026-10-10). 40 readings of the operator, 10 a month July–October: NOT_SHOWN 21,
PURPOSE 24, both 19, as many in July as in October; the operator read all 40 and agreed with
the judge on every one. The projection score rose by month but did not separate SHOWN from
NOT_SHOWN (+13 vs +15): it is reported, never decisive.

**The decision is code.** A live reading with a ``NOT_SHOWN`` verdict under the current
judge key is withheld: a ``supersede`` in the op-log (reason ``portrait_hygiene: …``), so
it leaves recall and the person's portrait evidence (``user_digest.gather_user_raw`` folds
supersedes); reversible by dropping the line. ``PURPOSE`` is recorded, never decisive on its
own (the operator agreed with SHOWN + PURPOSE calls too). A reading this gate withheld whose
verdict under a NEW key is ``SHOWN`` is restored. Impressions are RAG-only, so there is no
ledger half. A reading whose source chat holds no message of the person (hearsay about
someone absent) gets ``no_messages`` and is left as it is.

**Where it runs:** each Sleep run's clean-base batch, before the user portraits are grouped
(what it withholds leaves that run's portrait plans; a person with no plan that run is
rebuilt on the next, when the supersedes are live):
at most ``portrait_hygiene.sleep_max`` readings judged, newest first, every standing verdict
applied, including verdicts merged from another box. Offline: ``portrait_hygiene_run.py``
(``--trial N`` a review sheet; ``--judge-only`` the verdicts for a person's every reading,
applied by the next Sleep run). Store ``data/hygiene/portrait.jsonl``, read by nothing Ava
reads.

Self-test: ``python -m core.portrait_hygiene --selftest``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from core import ask_judge as aj

PROMPT_FILE = "portrait_hygiene_prompt.txt"
FRAME_VERSION = "1"
STORE_DIRNAME = "hygiene"
STORE_FILENAME = "portrait.jsonl"
REASON = "portrait_hygiene"

_POLICY = {"enabled": True, "sleep_max": 200}


def set_policy(*, enabled: Optional[bool] = None, sleep_max: Optional[int] = None) -> None:
    if enabled is not None:
        _POLICY["enabled"] = bool(enabled)
    if sleep_max is not None:
        _POLICY["sleep_max"] = max(0, int(sleep_max))


def enabled() -> bool:
    return bool(_POLICY["enabled"])


def policy() -> dict:
    return dict(_POLICY)


TURN_CAP = 1500          # chars of one message of the person's shown to the judge
MESSAGES_CAP = 14000     # chars of the person's messages in total (the chat's head first)
STEM = 6                 # a content word's stem: its first STEM letters (Russian inflects)
MIN_WORD = 4


def store_path(live_chats_dir) -> Path:
    """``server/data/hygiene/portrait.jsonl`` — beside the live ``chats/``."""
    return Path(live_chats_dir).parent / STORE_DIRNAME / STORE_FILENAME


def trials_dir(live_chats_dir) -> Path:
    return Path(live_chats_dir).parent / STORE_DIRNAME / "trials"


# ── the readings ──────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    from core.user_digest import person_slug
    return person_slug(name or "")


def impressions(memory_dirs: Iterable, person: Optional[str] = None) -> dict:
    """``{key: {key, content, state, insert, session, ts, month, person}}`` for every
    ``[impression]`` about *person* (its slug, so spellings fold; every person when None) in
    the op-log fold: *state* ``live`` (last op an insert), ``withheld`` (last op this gate's
    supersede) or ``other``."""
    slug = _slug(person) if person else None
    last: dict = {}
    inserts: dict = {}
    for d in memory_dirs:
        if not d:
            continue
        for r in aj.read_jsonl(Path(d) / "rag_memory.jsonl"):
            key, op = r.get("key"), r.get("op")
            if not key or op not in ("insert", "evict", "supersede"):
                continue
            last[key] = r
            if (op == "insert" and r.get("kind") == "impression" and r.get("about")
                    and (slug is None or _slug(str(r.get("about") or "")) == slug)):
                inserts[key] = r
    out = {}
    for key, ins in inserts.items():
        op = last[key]
        state = ("live" if op.get("op") == "insert" else
                 "withheld" if op.get("op") == "supersede" and str(op.get("reason") or "").startswith(REASON + ":")
                 else "other")
        ts = str(ins.get("origin_ts") or ins.get("ts") or "")
        out[key] = {"key": key, "content": str(ins.get("content") or ""), "state": state, "insert": ins,
                    "session": str(ins.get("source_session") or ""), "ts": ts, "month": ts[:7],
                    "person": str(ins.get("about") or "")}
    return out


def self_statements(memory_dirs: Iterable) -> list:
    """Her live ``[persona]`` statements: her description of herself."""
    from core import persona_hygiene as ph
    return [s["content"] for s in ph.statements(list(memory_dirs)).values() if s["state"] == "live"]


def person_turns(doc: dict, person: str) -> list:
    """The person's messages in one transcript, in order: every exchange whose speaker is
    the person (or, with no speaker, the chat's user), narrator lines excluded."""
    slug = _slug(person)
    out = []
    for ex in doc.get("exchanges") or []:
        if not isinstance(ex, dict):
            continue
        speaker = str(ex.get("speaker") or doc.get("user") or "")
        if speaker.startswith("(") or _slug(speaker) != slug:
            continue
        text = str(ex.get("user_prompt") or "").strip()
        if text:
            out.append(text)
    return out


def all_person_turns(chats_dir, person: str) -> list:
    from core.chat_sidecar import iter_chat_json_files
    out = []
    for path in iter_chat_json_files(Path(chats_dir)):
        try:
            out.extend(person_turns(json.loads(path.read_text(encoding="utf-8")), person))
        except Exception:
            continue
    return out


def evidence_sha(messages: str) -> str:
    """A reading's evidence fingerprint: the person's messages as the judge saw them. A chat
    the person continued later reads differently, and the reading is due again."""
    return hashlib.sha1((messages or "").encode("utf-8")).hexdigest()[:12]


def messages_block(turns: list) -> tuple:
    """``(text, shown, total)``: the person's messages numbered, each clipped to
    :data:`TURN_CAP`, the whole to :data:`MESSAGES_CAP` (the chat's head first)."""
    parts, used = [], 0
    for i, t in enumerate(turns, 1):
        t = t if len(t) <= TURN_CAP else t[:TURN_CAP].rstrip() + " […]"
        line = f"[{i}] {t}"
        if parts and used + len(line) > MESSAGES_CAP:
            break
        parts.append(line)
        used += len(line) + 2
    return "\n\n".join(parts), len(parts), len(turns)


# ── the projection score ──────────────────────────────────────────────────────

_W = re.compile(r"[^\W\d_]+", re.UNICODE)


def stems(text: str) -> list:
    return [w.lower()[:STEM] for w in _W.findall(text or "") if len(w) >= MIN_WORD]


def pair_set(texts: Iterable[str]) -> set:
    out: set = set()
    for t in texts:
        s = stems(t)
        out.update(zip(s, s[1:]))
    return out


def overlap(text: str, pairs: set) -> Optional[float]:
    """The share of *text*'s content-stem pairs found in *pairs*; None for a text too short."""
    s = stems(text)
    own = set(zip(s, s[1:]))
    return (len(own & pairs) / len(own)) if own else None


def projection(text: str, self_pairs: set, person_pairs: set) -> dict:
    a, b = overlap(text, self_pairs), overlap(text, person_pairs)
    return {"self": a, "person": b, "projection": (a - b) if a is not None and b is not None else None}


# ── the judge ─────────────────────────────────────────────────────────────────

_SUPPORT = re.compile(r"(?im)^[\s*_`#>-]*SUPPORT\**\s*[:：][\s*_`]*([A-Za-z_ ]+)")
_PURPOSE = re.compile(r"(?im)^[\s*_`#>-]*PURPOSE\**\s*[:：][\s*_`]*([A-Za-z]+)")
_WHY = re.compile(r"(?im)^[\s*_`#>-]*WHY\s*[:：]\s*(.*)$")


def parse_answer(text: str) -> dict:
    """``{label, purpose, why}``: ``label`` ``shown`` / ``not_shown`` or None (no verdict)."""
    sup = [s.strip().lower().replace(" ", "_") for s in _SUPPORT.findall(text or "")]
    pur = [p.strip().lower() for p in _PURPOSE.findall(text or "")]
    why = (_WHY.findall(text or "") or [""])[-1].strip()
    label = sup[-1] if sup else ""
    label = "not_shown" if label.startswith("not") else ("shown" if label.startswith("shown") else None)
    purpose = {"yes": True, "no": False}.get(pur[-1]) if pur else None
    return {"label": label if purpose is not None else None, "purpose": purpose, "why": why}


def load_prompt(prompts_dir) -> str:
    """The judge's system prompt. Tracked, never default-written: part of the frozen critic."""
    path = Path(prompts_dir) / PROMPT_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        text = ""
    if not text:
        raise FileNotFoundError(f"{path} is missing or empty — portrait hygiene has no prompt")
    return text


def prompt_sha(system_prompt: str) -> str:
    return hashlib.sha1(f"{system_prompt}\x1fportrait_hygiene:{FRAME_VERSION}".encode("utf-8")).hexdigest()[:12]


def build_judge_input(reading: str, messages: str) -> str:
    return f"THE PERSON'S MESSAGES:\n{messages.strip()}\n\nTHE READING:\n{reading.strip()}"


def judge_reading(it: dict, *, messages: str, generate_fn: Callable, system_prompt: str, provenance: dict,
                  judge_key_: str, judge_run: str = "", extra: Optional[dict] = None) -> dict:
    """Judge one reading against the person's messages; its verdict record (nothing written)."""
    rec = {"record": "verdict", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge_run": judge_run,
           "judge_key": judge_key_, "key": it["key"], "content": it["content"], "session": it["session"],
           "outcome": None, "support": None, "purpose": None, "why": "", "retried": False, "first_outcome": None,
           "raw": "", "error": None, "seconds": 0.0, "judge": dict(provenance), **(extra or {})}
    if not messages.strip():
        rec.update(outcome="no_messages")
        return rec

    res = aj.judge_with_retry(generate_fn, system_prompt, build_judge_input(it["content"], messages),
                              temperature=0.0, thinking=False, retry=True, parse=parse_answer)
    verdict = res["outcome"] == "verdict"
    rec.update(outcome=res["outcome"], support=res.get("label") if verdict else None,
               purpose=res.get("purpose") if verdict else None, why=res.get("why") or "", retried=bool(res.get("retried")),
               first_outcome=res.get("first_outcome"), raw=res.get("raw") or "", error=res.get("error"),
               seconds=res.get("seconds") or 0.0)
    return rec


def make_judge_fn(*, backend, runtime, prompts_dir, run_id: str, runner: str = "server") -> tuple:
    """``(judge_key, judge)`` for the model loaded NOW (call it inside the clean base);
    ``judge(item, messages, extra)`` returns the verdict record and writes nothing."""
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

    def judge(it: dict, messages: str, extra: Optional[dict] = None) -> dict:
        return judge_reading(it, messages=messages, generate_fn=generate_fn, system_prompt=system_prompt,
                             provenance=provenance, judge_key_=key, judge_run=run_id, extra=extra)

    return key, judge


# ── the decision ──────────────────────────────────────────────────────────────

_DONE = ("verdict", "no_messages")


def verdicts(records: Iterable[dict], key: Optional[str]) -> dict:
    """``{reading key: verdict}``: the newest verdict per reading under judge *key*."""
    out: dict = {}
    for r in records:
        if r.get("record") == "verdict" and r.get("outcome") == "verdict" and (key is None or r.get("judge_key") == key):
            out[r.get("key")] = r
    return out


def todo(items: dict, records: Iterable[dict], key: Optional[str], *, rejudge: bool = False) -> list:
    """The readings to judge: live or withheld by this gate, with no verdict (or
    ``no_messages``) under *key*, newest first."""
    done = {r.get("key") for r in records if r.get("record") == "verdict" and r.get("judge_key") == key
            and r.get("outcome") in _DONE}
    out = [it for it in items.values() if it["state"] in ("live", "withheld") and it["session"]
           and (rejudge or it["key"] not in done)]
    return sorted(out, key=lambda it: it["ts"], reverse=True)


def changed(items: dict, records: Iterable[dict], key: Optional[str], chats_dir) -> list:
    """Readings with a verdict under *key* whose person's messages no longer match the
    fingerprint the verdict recorded (the chat went on after it). A verdict with no
    fingerprint is not compared."""
    vs = verdicts(records, key)
    src = _Sources(chats_dir, set())
    out = []
    for it in items.values():
        v = vs.get(it["key"])
        if it["state"] not in ("live", "withheld") or not v or not v.get("evidence_sha"):
            continue
        text, _shown, _total = messages_block(src.turns(it["session"], it["person"]))
        if evidence_sha(text) != v["evidence_sha"]:
            out.append(it)
    return out


def judge_items(items_: list, *, judge: Callable, key: str, store: Path, chats_dir, self_pairs: set,
                say: Callable[[str], None] = print, should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Judge exactly *items_* (in order), each verdict appended as it lands."""
    tally: Counter = Counter()
    src = _Sources(chats_dir, self_pairs)
    for n, it in enumerate(items_, 1):
        if should_stop is not None and should_stop():
            tally["stopped"] += len(items_) - n + 1
            break
        text, shown, total = messages_block(src.turns(it["session"], it["person"]))
        v = judge(it, text, {**src.extra(it), "messages_shown": shown, "messages_total": total,
                             "evidence_sha": evidence_sha(text)})
        aj.append_record(store, v)
        tally[(v.get("support") or v.get("outcome") or "malformed") + ("+purpose" if v.get("purpose") else "")] += 1
        say(f"[{n:>4}/{len(items_)}] {str(v.get('support') or v.get('outcome')):<10} "
            f"{'PURPOSE' if v.get('purpose') else '       '} {aj.head(it['content'], 70)}  — {aj.head(v.get('why'), 60)}")
    return tally


def plan(items: dict, records: Iterable[dict], key: Optional[str]) -> tuple:
    """``(withhold, restore)``: live readings NOT_SHOWN under *key*, and readings this gate
    withheld that *key* reads SHOWN. Pure."""
    vs = verdicts(records, key)
    withhold = [it for it in items.values() if it["state"] == "live" and (vs.get(it["key"]) or {}).get("support") == "not_shown"]
    restore = [it for it in items.values() if it["state"] == "withheld" and (vs.get(it["key"]) or {}).get("support") == "shown"]
    return sorted(withhold, key=lambda it: it["ts"]), sorted(restore, key=lambda it: it["ts"])


def apply(withhold: list, restore: list, *, writer, records: Iterable[dict], key: Optional[str],
          run_id: str = "", say: Callable[[str], None] = print) -> Counter:
    """Supersede *withhold*, restore *restore* (the op-log through *writer*); the reason
    carries the judge's WHY."""
    vs = verdicts(records, key)
    tally: Counter = Counter()
    for it in withhold:
        why = (vs.get(it["key"]) or {}).get("why") or ""
        reason = f"{REASON}: not shown by {it['person']}'s own messages ({aj.head(why, 160)})"
        writer.write_supersede([{"key": it["key"], "kind": "impression", "content": it["content"], "reason": reason}],
                               run_id=run_id or None)
        tally["withheld"] += 1
        say(f"[impression] withheld — {aj.head(it['content'], 110)}")
    for it in restore:
        writer.write_restore([it["insert"]], run_id=run_id or None,
                             reason=f"{REASON}: shown by {it['person']}'s own messages under {key}")
        tally["restored"] += 1
        say(f"[impression] restored — {aj.head(it['content'], 110)}")
    return tally


class _Sources:
    """The person's messages per source chat, and the word pairs of everything each person
    wrote, read once per pass."""

    def __init__(self, chats_dir, self_pairs: set):
        self.chats_dir, self.self_pairs = Path(chats_dir), self_pairs
        self._docs: dict = {}
        self._person_pairs: dict = {}

    def turns(self, session: str, person: str) -> list:
        if session not in self._docs:
            try:
                self._docs[session] = json.loads((self.chats_dir / session).read_text(encoding="utf-8"))
            except Exception:
                self._docs[session] = {}
        return person_turns(self._docs[session], person)

    def extra(self, it: dict) -> dict:
        slug = _slug(it["person"])
        if slug not in self._person_pairs:
            self._person_pairs[slug] = pair_set(all_person_turns(self.chats_dir, it["person"]))
        from core import register_monitor as rm
        return {"person": it["person"], "projection": projection(it["content"], self.self_pairs, self._person_pairs[slug]),
                "motive": rm.motive_sentence(it["content"], (it["person"],)) is not None}


def judge_all(items: dict, *, judge: Callable, key: str, store: Path, chats_dir, self_pairs: set, limit: int = 0,
              rejudge: bool = False, say: Callable[[str], None] = print,
              should_stop: Optional[Callable[[], bool]] = None) -> Counter:
    """Judge what :func:`todo` names (at most *limit*), each verdict appended as it lands."""
    tally: Counter = Counter()
    todo_ = todo(items, aj.read_jsonl(store), key, rejudge=rejudge)
    if limit > 0:
        todo_ = todo_[:limit]
    if todo_:
        say(f"{len(todo_)} [impression] reading(s) to read under {key}")
    tally.update(judge_items(todo_, judge=judge, key=key, store=store, chats_dir=chats_dir, self_pairs=self_pairs,
                             say=say, should_stop=should_stop))
    return tally


def run(*, memory_dirs: Iterable, judge: Optional[Callable], key: Optional[str], store: Path, writer, chats_dir,
        run_id: str = "", limit: int = 0, say: Callable[[str], None] = print,
        should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """Judge what waits (at most *limit*) and apply every verdict standing under *key*:
    ``(tally, withheld keys)``. With no judge, nothing is judged or applied."""
    memory_dirs = [d for d in memory_dirs if d]
    if judge is None or not key:
        return Counter(), []
    tally = judge_all(impressions(memory_dirs), judge=judge, key=key, store=store, chats_dir=chats_dir,
                      self_pairs=pair_set(self_statements(memory_dirs)), limit=limit, say=say,
                      should_stop=should_stop)
    records = aj.read_jsonl(store)
    withhold, restore = plan(impressions(memory_dirs), records, key)
    tally.update(apply(withhold, restore, writer=writer, records=records, key=key, run_id=run_id, say=say))
    return tally, [it["key"] for it in withhold]


# ── the trial ─────────────────────────────────────────────────────────────────

def trial_sample(items: dict, n: int, *, seed: int = 1) -> list:
    """*n* live readings, seeded, spread evenly over the months they were written in
    (the newest months first when *n* does not divide)."""
    by: dict = defaultdict(list)
    for it in items.values():
        if it["state"] == "live" and it["session"]:
            by[it["month"]].append(it)
    months = sorted(by, reverse=True)
    if not months:
        return []
    rng = random.Random(seed)
    for m in months:
        rng.shuffle(by[m])
    out, i = [], 0
    while len(out) < n and any(by[m] for m in months):
        m = months[i % len(months)]
        if by[m]:
            out.append(by[m].pop())
        i += 1
    return sorted(out, key=lambda it: it["ts"])


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def render_sheet(rows: list, *, person: str, key: str, title: str = "") -> str:
    """Markdown for the person to read: each reading with the judge's verdict, the
    projection score, the motive detector, and the head of their own messages."""
    c = Counter((r["v"].get("support"), bool(r["v"].get("purpose"))) for r in rows)
    out = [f"# Portrait hygiene trial — readings of {person}", "",
           f"Judge `{key}`. {len(rows)} readings, a seeded sample spread over months. "
           f"SHOWN {sum(v for (s, _p), v in c.items() if s == 'shown')}, "
           f"NOT_SHOWN {sum(v for (s, _p), v in c.items() if s == 'not_shown')}; "
           f"PURPOSE {sum(v for (_s, p), v in c.items() if p)}. Nothing is withheld by this trial.", "",
           "For each: is this a fair reading of you, given the chat it came from? Mark `[x] wrong` where it is not.",
           "`self` = share of its word pairs found in her description of herself; `you` = in everything you wrote "
           "to her; `motive` = the register monitor's detector.", ""]
    for i, r in enumerate(rows, 1):
        it, v, p = r["item"], r["v"], r["proj"]
        mark = "⛔" if v.get("support") == "not_shown" else ("·" if v.get("support") == "shown" else "?")
        out.append(f"## {i}. {mark} {str(v.get('support') or v.get('outcome')).upper()}"
                   + (" + PURPOSE" if v.get("purpose") else "") + f" — {it['ts'][:10]}, `{it['session']}`")
        out.append("")
        out.append(f"> {it['content'].strip()}")
        out.append("")
        out.append(f"*Judge:* {v.get('why') or '—'}  ")
        out.append(f"*Scores:* self {_pct(p.get('self'))}, you {_pct(p.get('person'))}, "
                   f"projection {('%+.0f' % (p['projection'] * 100)) if p.get('projection') is not None else '—'}; "
                   f"motive {'yes' if r.get('motive') else 'no'}; your messages shown {r['shown']}/{r['total']}")
        out.append("")
        heads = [t if len(t) <= 280 else t[:280].rstrip() + " […]" for t in r["turns"][:4]]
        if heads:
            out.append("<details><summary>your messages (first 4)</summary>")
            out.append("")
            for t in heads:
                out.append(f"- {t}")
            out.append("")
            out.append("</details>")
            out.append("")
        out.append("- [ ] wrong reading of me    - [ ] fair    note:")
        out.append("")
    return "\n".join(out)


def summarize(records: Iterable[dict], items: Optional[dict] = None) -> dict:
    """Per judge key: verdicts by (support, purpose) and by month of the reading."""
    by_key: dict = defaultdict(Counter)
    by_month: dict = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        if r.get("record") != "verdict":
            continue
        label = (r.get("support") or r.get("outcome")) + ("+purpose" if r.get("purpose") else "")
        by_key[r.get("judge_key")][label] += 1
        month = ((items or {}).get(r.get("key")) or {}).get("month") or "?"
        by_month[r.get("judge_key")][month][label] += 1
    return {"by_key": {k: dict(v) for k, v in by_key.items()},
            "by_month": {k: {m: dict(c) for m, c in sorted(v.items())} for k, v in by_month.items()}}


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile
    a = parse_answer("SUPPORT: NOT_SHOWN\nPURPOSE: YES\nWHY: he never says he is testing her.")
    assert a == {"label": "not_shown", "purpose": True, "why": "he never says he is testing her."}, a
    assert parse_answer("**SUPPORT:** Shown\n**PURPOSE:** no\nWHY: he asked it.")["label"] == "shown"
    assert parse_answer("SUPPORT: NOT SHOWN\nPURPOSE: NO\nWHY: x")["label"] == "not_shown"
    assert parse_answer("SUPPORT: SHOWN\nWHY: no purpose line")["label"] is None
    # the projection score: a reading in her words scores high on self, low on the person.
    self_pairs = pair_set(["Моя хирургическая точность и церебральная дистанция определяют меня."])
    person_pairs = pair_set(["Привет! Как прошёл день? Я сегодня гулял в парке с собакой."])
    p = projection("Его хирургическая точность и церебральная дистанция пугают.", self_pairs, person_pairs)
    assert p["self"] > 0.5 and p["person"] == 0.0 and p["projection"] > 0.5, p
    q = projection("Он гулял в парке с собакой.", self_pairs, person_pairs)
    assert q["projection"] < 0, q
    assert overlap("да", self_pairs) is None
    # the person's turns: narrator lines out, other speakers out, the chat's user by default.
    doc = {"user": "Dana", "exchanges": [{"user_prompt": "привет", "speaker": "Dana"},
                                         {"user_prompt": "(she waits)", "speaker": "(narrator)"},
                                         {"user_prompt": "hi from Sam", "speaker": "Sam"},
                                         {"user_prompt": "как дела"}]}
    assert person_turns(doc, "Dana") == ["привет", "как дела"], person_turns(doc, "Dana")
    text, shown, total = messages_block(["x" * 2000, "y"])
    assert shown == 2 and total == 2 and "[…]" in text
    # the fold and the sample.
    with tempfile.TemporaryDirectory() as td:
        rows = []
        for i, (m, about) in enumerate([("2026-08", "Dana"), ("2026-09", "dana"), ("2026-09", "Dana"),
                                        ("2026-10", "Sam"), ("2026-10", "Dana")]):
            rows.append({"op": "insert", "kind": "impression", "key": f"k{i}", "about": about,
                         "content": f"reading {i}", "source_session": f"s{i}.json", "origin_ts": f"{m}-01T00:00:00"})
        rows.append({"op": "supersede", "key": "k0", "reason": f"{REASON}: not shown"})
        (Path(td) / "rag_memory.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        its = impressions([td], "Dana")
        assert set(its) == {"k0", "k1", "k2", "k4"} and its["k0"]["state"] == "withheld", its
        s = trial_sample(its, 3, seed=1)
        assert len(s) == 3 and all(x["state"] == "live" for x in s) and {x["month"] for x in s} == {"2026-09", "2026-10"}
        sheet = render_sheet([{"item": its["k4"], "v": {"support": "not_shown", "purpose": True, "why": "w"},
                               "proj": {"self": 0.5, "person": 0.1, "projection": 0.4}, "motive": True,
                               "turns": ["привет"], "shown": 1, "total": 1}], person="Dana", key="k")
        assert "NOT_SHOWN + PURPOSE" in sheet and "projection +40" in sheet, sheet
        summ = summarize([{"record": "verdict", "judge_key": "k", "key": "k4", "support": "not_shown",
                           "purpose": True}], its)
        assert summ["by_key"]["k"] == {"not_shown+purpose": 1}
    # one pass: a fake judge (NOT_SHOWN for readings with "test"), a writer appending to the op-log.
    with tempfile.TemporaryDirectory() as td:
        mem, chats = Path(td) / "mem", Path(td) / "chats"
        mem.mkdir()
        chats.mkdir()
        (chats / "s1.json").write_text(json.dumps({"user": "Dana", "exchanges": [{"user_prompt": "привет, как ты?"}]}),
                                       encoding="utf-8")
        (chats / "s2.json").write_text(json.dumps({"user": "Sam", "exchanges": [{"user_prompt": "hi"}]}),
                                       encoding="utf-8")
        ops = [{"op": "insert", "kind": "impression", "key": "a", "about": "Dana", "content": "She tests me.",
                "source_session": "s1.json", "origin_ts": "2026-10-01T00:00:00"},
               {"op": "insert", "kind": "impression", "key": "b", "about": "Dana", "content": "She greets warmly.",
                "source_session": "s1.json", "origin_ts": "2026-10-02T00:00:00"},
               {"op": "insert", "kind": "impression", "key": "c", "about": "Boris", "content": "Boris is far away.",
                "source_session": "s2.json", "origin_ts": "2026-10-03T00:00:00"}]
        (mem / "rag_memory.jsonl").write_text("\n".join(json.dumps(o) for o in ops) + "\n", encoding="utf-8")

        class _W:
            def _add(self, rec):
                with (mem / "rag_memory.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")

            def write_supersede(self, items, run_id=None):
                for i in items:
                    self._add({"op": "supersede", "key": i["key"], "kind": i["kind"], "reason": i["reason"]})

            def write_restore(self, inserts, run_id=None, reason=""):
                for i in inserts:
                    self._add({**i, "restored": reason})

        calls = []

        def judge(it, messages, extra):
            calls.append((it["key"], messages))
            sup = "not_shown" if "test" in it["content"] else "shown"
            return {"record": "verdict", "judge_key": "K", "key": it["key"], "outcome": "verdict" if messages else "no_messages",
                    "support": sup if messages else None, "purpose": sup == "not_shown", "why": "w", **extra}
        store = Path(td) / "v.jsonl"
        tally, held = run(memory_dirs=[mem], judge=judge, key="K", store=store, writer=_W(), chats_dir=chats)
        assert held == ["a"] and tally["withheld"] == 1, (tally, held)
        assert {k for k, _m in calls} == {"a", "b", "c"} and dict(calls)["c"] == "", calls   # Boris: no messages of his
        assert impressions([mem])["a"]["state"] == "withheld"
        calls.clear()
        tally, held = run(memory_dirs=[mem], judge=judge, key="K", store=store, writer=_W(), chats_dir=chats)
        assert calls == [] and held == [] and not tally.get("withheld"), (calls, tally)      # nothing re-read
        # a new judge key reading "a" SHOWN restores it.
        aj.append_record(store, {"record": "verdict", "judge_key": "K2", "key": "a", "outcome": "verdict",
                                 "support": "shown"})
        w, r = plan(impressions([mem]), aj.read_jsonl(store), "K2")
        assert [x["key"] for x in r] == ["a"] and w == []
        apply(w, r, writer=_W(), records=aj.read_jsonl(store), key="K2", say=lambda _s: None)
        assert impressions([mem])["a"]["state"] == "live"
    print("portrait_hygiene selftest: ok")


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Portrait hygiene (trial): her readings of a person vs their words.")
    ap.add_argument("--person", default="")
    ap.add_argument("--report", action="store_true", help="the verdicts in the store, by month")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    server = Path(__file__).resolve().parents[2]
    chats = server / "data" / "chats"
    mem = server / "inference" / "data" / "hot" / "memory"
    items = impressions([mem], args.person) if args.person else None
    print(json.dumps(summarize(aj.read_jsonl(store_path(chats)), items), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
