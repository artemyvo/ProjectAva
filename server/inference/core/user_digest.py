"""Per-person **user portrait** — the user-side mirror of the persona digest.

``reflection_digest`` folds Ava's ``[persona]`` self-statements into one standing
self-portrait that is injected into every chat turn. This module does the same thing for
the people she talks to: it folds her ``[impression]`` readings of a person (plus the
attributed ``[fact]`` items about them) into one standing portrait of *that person*, which
``generation`` injects when they are the one speaking.

Why a portrait and not just recall
----------------------------------
Facts about a person were already retrievable, but only *situationally* — whichever one to
three embedded closest to the current message, competing with everything else for the
reflection block's slots. So Ava arrived at each turn knowing whatever the message happened
to key on and nothing else about who she was talking to. That is the exact failure the
persona portrait fixed on her own side ("she arrives as herself instead of reassembling
herself each time"), and it has the same fix here: one coherent reading of the person,
every turn, with recall still free to add what *this* moment needs.

What is deliberately NOT mirrored from the persona digest
---------------------------------------------------------
* **No weights path.** A portrait is folded from ``[impression]`` records, which are
  RAG-only by construction (see ``ReflectionWriter.write_impressions``) — nothing here
  trains. The persona digest feeds a self that is *becoming*; a portrait describes someone
  who is not Ava's to become.
* **No map-reduce clustering.** ``persona_cluster`` exists because one prompt cannot hold
  ~640 live persona statements. A single person's evidence is one to two orders of
  magnitude smaller, so the single flat grouping call (``reflection_digest`` tier one) is
  the right size of hammer — and it is reused verbatim rather than reimplemented.
* **No maturity gate on injection.** The persona portrait gates on ``_is_established``
  because a portrait in every prompt tightens a *self*-reinforcement loop: portrait shapes
  reply, reply yields persona statement, statement feeds portrait. The user loop is not
  closed the same way — the person supplies their own evidence by continuing to be
  themselves, and a wrong reading is corrected by the next conversation rather than
  amplified by it. Recurrence is still recorded and shown, so a one-off reads as tentative
  in the prose instead of being silently dropped.

Memory stays **shared and global** (``AVA_MEMORY.md`` §G): a portrait scopes *injection*
(whose reading is standing context right now), never *storage* or *retrieval*. Every
impression remains in the one op-log, attributed and recallable in anyone's conversation.

GPU-free self-test: ``python -m core.user_digest``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from core.field_parse import section as _section

from core.reflection_digest import (       # the fold primitives this reuses verbatim
    _RECENCY_BANDS,
    _age_days,
    _persona_recency_weight,
    _recency_band,
    _bullets,
    strip_think,
    cluster_persona_evidence,
)
from core.reflection_writer import normalize_person

# One flat file per person; the person IS the version axis, so there is no pointer.
PORTRAIT_SUFFIX = ".json"

# Facets of a person. Chosen as the observational counterparts of the persona digest's
# VOICE / STANCES / DISPOSITIONS / LINES, with one addition that has no self-side analogue:
#   WHO      — the plain sketch: who this person is, as she has come to read them.
#   CARES    — what they keep returning to (the STANCES analogue: their commitments).
#   WAYS     — how they talk: register, humour, tells (the VOICE analogue, observed).
#   WITH_ME  — how the relationship actually runs: what they want from her, what lands.
#   UNSURE   — what she still does not know, or may have read wrong.
# UNSURE is the addition, and it is load-bearing rather than decorative. A portrait of a
# PERSON that carries only conclusions hardens into confident fiction the moment one
# reading is wrong, and unlike a self-portrait there is a real someone it can be wrong
# about. Naming the open edges in the same block keeps the uncertainty in front of her,
# and doubles as the material an [ask:user] can grow from.
_SECTION_RE = re.compile(
    _section("WHO", "CARES", "WAYS", "WITH_ME", "WITH ME", "UNSURE"),
    re.IGNORECASE | re.MULTILINE,
)

# Evidence kinds a portrait folds, and what each contributes.
#   impression — her own reading of the person (the [persona] analogue). The core.
#   fact       — a stable attributed truth about them. Included because a portrait that
#                knew how someone argues but not what they do for a living would be a
#                strange half-acquaintance; excluded from RANKING pressure by carrying
#                its own kind through, so synthesis can tell a reading from a datum.
_EVIDENCE_KINDS = ("impression", "fact")

# A fact only joins the portrait when its epistemic class says the subject is really its
# subject. `hearsay` (one person's account of a third party) is excluded for exactly the
# reason the weights gate excludes it: an unverified claim about B must not become part of
# how Ava stands toward B. It stays recallable and attributed, just not portrait-forming.
_EXCLUDED_SOURCE_CLASSES = frozenset({"hearsay"})

# Minimum live evidence items before a person gets a portrait at all. Below this the fold
# is a handful of first impressions, and synthesizing "who this person is" from three lines
# produces confident nonsense — recall alone serves better. Deliberately small: the point
# is to exclude the degenerate case, not to withhold the portrait until some notion of
# maturity is reached (see the module docstring on why there is no injection gate).
MIN_EVIDENCE_ITEMS = 4

# How much of an unparseable generation rides the skip summary. Enough to see the shape
# of the failure (a runaway <think>, a wrong output format), bounded because the summary
# is attached to a run event and mirrored into the activity journal.
_SKIP_RAW_CLIP = 1500

# Token budget for the synthesis pass. Double the self-side digest's 8192, because the two
# passes do not face the same risk: thinking is ON here and the <think> block scales with
# the EVIDENCE, which grows without bound as Ava accumulates history with one person — so a
# budget that fit a young corpus quietly stops fitting a long-running one. Truncation here
# is silent-ish by construction: the cut lands mid-<think>, `strip_think` leaves nothing to
# parse, and the run reports `unparseable or empty` with no portrait written — indefinitely,
# since the same oversized evidence is re-read every run. The cap is only a ceiling (the
# generate seam clamps it to the context actually left after the prompt), and a pass that
# ends on EOS pays nothing for the headroom, so the cost of raising it is bounded by the
# runaway case it exists to survive.
PORTRAIT_MAX_NEW_TOKENS = "16384"


# ── person identity ────────────────────────────────────────────────────────── #

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")

# Slugs the users dir reserves for artifacts that are NOT a person. `_self` holds the
# outside-view self-portrait (``core.self_portrait``), which lives here to inherit the
# dir's per-run archival, wipe semantics and snapshot travel. `person_slug` refuses them
# outright, which is what makes "a person can never resolve to that file" structural
# rather than a check each caller has to remember — note the slug charset below PERMITS
# a leading underscore, so the refusal has to be explicit.
RESERVED_SLUGS = frozenset({"_self"})
SELF_SLUG = "_self"


def person_slug(name: str) -> str:
    """Filesystem key for a person — ``""`` when *name* identifies nobody.

    Built on :func:`reflection_writer.normalize_person`, so the portrait file, the
    attribution on a fact, and the speaker on a live turn all resolve to ONE identity:
    "Artemy", "artemy voikhansky" and "Artemy!" are one person and one file, while a
    generic referent ("the user") is nobody and gets no portrait.

    Non-ASCII names (the corpus is mixed-language) survive as a stable hex digest rather
    than being mangled into collisions by a strip-to-ASCII rule.
    """
    norm = normalize_person(name)
    if not norm:
        return ""
    slug = _SLUG_RE.sub("", norm)
    if slug in RESERVED_SLUGS:
        return ""
    if not slug:
        # Wholly non-ASCII name: keep it addressable and collision-free.
        return "p-" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]
    return slug


# ── the model-free fold ────────────────────────────────────────────────────── #

def gather_user_raw(memory_dir: Path, *, fallback_memory_dir: Optional[Path] = None,
                    now=None) -> dict[str, dict]:
    """Replay ``rag_memory.jsonl`` into ``{person_slug: {display, items}}``.

    The user-side counterpart of ``reflection_digest.gather_persona_raw``, and it reads the
    RAW op-log for the same reason that one reads the raw ledger rather than the folded
    state: the maturity signal is the set of **distinct sessions** an impression recurred
    in, which any fold-to-current-value collapses away.

    Each item carries the same shape the persona clustering consumes — ``{key, content,
    stage, sessions, session_weights, recency_band}`` — so ``cluster_persona_evidence``
    and its ``_evidence_entry`` collapse apply unchanged. ``stage`` is always 0 (a
    portrait has no consolidation lifecycle) and ``kind`` rides along so synthesis can
    distinguish a reading from a fact.

    Deterministic given *now*, which is what lets :func:`raw_fingerprint` gate
    regeneration.
    """
    if now is None:
        now = datetime.now()

    # key → record (last insert wins, matching ReflectionMemory's fold)
    items: dict[str, dict] = {}
    # key → {session: freshest ISO ts}
    session_ts: dict[str, dict[str, str]] = {}
    # normalized person → the display spelling seen most recently (what prose should use)
    display: dict[str, str] = {}

    paths = [p for p in (fallback_memory_dir, memory_dir) if p is not None]
    lines: list[str] = []
    for d in paths:
        path = Path(d) / "rag_memory.jsonl"
        if not path.exists():
            continue
        try:
            lines.extend(path.read_text(encoding="utf-8").splitlines())
        except Exception:
            continue

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        key = rec.get("key")
        if not key:
            continue
        op = rec.get("op")
        if op in ("evict", "supersede"):
            items.pop(key, None)
            session_ts.pop(key, None)
            continue
        if op != "insert":
            continue
        if rec.get("kind") not in _EVIDENCE_KINDS:
            continue
        if (rec.get("source_class") or "") in _EXCLUDED_SOURCE_CLASSES:
            continue
        about = normalize_person(rec.get("about"))
        if not about:
            continue          # unattributed: a world fact, not a reading of a person
        content = (rec.get("content") or "").strip()
        if not content:
            continue
        items[key] = rec
        src = (rec.get("source_session") or "").strip()
        if src:
            ts = (rec.get("ts") or "").strip()
            prev = session_ts.setdefault(key, {}).get(src, "")
            if ts and ts > prev:   # ISO compares lexically == chronologically
                session_ts[key][src] = ts
        raw_about = (rec.get("about") or "").strip()
        if raw_about:
            display[about] = raw_about

    people: dict[str, dict] = {}
    for key, rec in items.items():
        about = normalize_person(rec.get("about"))
        slug = person_slug(about)
        if not slug:
            continue
        per_session = session_ts.get(key, {})
        sessions = set(per_session)
        weights = {s: _persona_recency_weight(_age_days(per_session.get(s, ""), now))
                   for s in sessions}
        band = _recency_band(max(weights.values())) if weights else _RECENCY_BANDS
        entry = people.setdefault(
            slug, {"slug": slug, "person": display.get(about, about), "items": []})
        entry["items"].append({
            "key": key,
            "content": (rec.get("content") or "").strip(),
            "kind": rec.get("kind", "impression"),
            "stage": 0,
            "sessions": sessions,
            "session_weights": weights,
            "recency_band": band,
            "source_session": (rec.get("source_session") or "").strip(),
            "trigger": (rec.get("trigger") or "").strip(),
        })
    return people


def raw_fingerprint(items: list[dict]) -> str:
    """Deterministic signature of one person's RAW evidence set (pre-clustering).

    Same role and same conservatism as ``reflection_digest.raw_fingerprint``: it gates
    regeneration, so it must be computed from the *deterministic* fold rather than from a
    non-deterministic LLM grouping — otherwise every run would look changed and pay for a
    portrait it had no new material for.
    """
    sig = sorted(
        f"{i['key']}:{i.get('kind', '')}:{len(i.get('sessions') or ())}"
        f":{i.get('recency_band', _RECENCY_BANDS)}"
        for i in items
    )
    return hashlib.sha1("|".join(sig).encode("utf-8")).hexdigest()[:16]


# ── persistence ────────────────────────────────────────────────────────────── #

def portrait_path(users_dir: Path, person: str) -> Optional[Path]:
    """``<users_dir>/<slug>.json`` for *person*, or None when they name nobody."""
    slug = person_slug(person)
    if not slug:
        return None
    return Path(users_dir) / (slug + PORTRAIT_SUFFIX)


def latest_portrait(users_dir: Path, person: str) -> Optional[dict]:
    """Load *person*'s portrait, or None when unwritten/unreadable/nameless."""
    p = portrait_path(users_dir, person)
    if p is None or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def known_people(users_dir: Path) -> list[str]:
    """Display names of everyone with a written portrait (sorted, best-effort).

    Reserved slugs are skipped: the dir also holds artifacts that are not a person (see
    :data:`RESERVED_SLUGS`), and a glob that reported them as people would be wrong in
    exactly the way the reserved namespace exists to prevent.
    """
    d = Path(users_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*" + PORTRAIT_SUFFIX)):
        if p.stem in RESERVED_SLUGS:
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")).get("person") or p.stem)
        except Exception:
            out.append(p.stem)
    return out


def should_regenerate(users_dir: Path, person: str, items: list[dict]) -> bool:
    """True when *person* has no portrait yet, or their raw evidence has changed."""
    if len(items) < MIN_EVIDENCE_ITEMS:
        return False
    cur = latest_portrait(users_dir, person)
    if not cur:
        return True
    return cur.get("evidence", {}).get("raw_fingerprint") != raw_fingerprint(items)


def write_portrait(users_dir: Path, person: str, portrait: dict, *, raw_text: str,
                   run_id: str, evidence: list[dict], raw_fp: Optional[str] = None
                   ) -> Optional[Path]:
    """Atomically write *person*'s portrait file. Returns the path (None if nameless)."""
    from training.reflections_path import atomic_write_text

    path = portrait_path(users_dir, person)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "person": person,
        "slug": person_slug(person),
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "who": portrait.get("who", ""),
        "cares": portrait.get("cares", []),
        "ways": portrait.get("ways", []),
        "with_me": portrait.get("with_me", []),
        "unsure": portrait.get("unsure", []),
        "evidence": _evidence_summary(evidence, raw_fp),
        "raw": raw_text or "",
    }
    atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return path


def _evidence_summary(evidence: list[dict], raw_fp: Optional[str] = None) -> dict:
    """Persistable summary of the clustered evidence — the portrait's audit trail.

    Mirrors ``reflection_digest._evidence_summary``: the themes are kept with their
    recurrence and merged members so a wrong merge or an over-weighted one-off is visible
    on the artifact rather than only in the prose it produced.
    """
    themes = [
        {
            "representative": e.get("content", ""),
            "recurrence": e.get("recurrences", 1),
            "weighted_recurrence": e.get("weighted_recurrence"),
            "cluster_size": e.get("cluster_size", 1),
            "members": e.get("members", []),
            "sources": e.get("sources", []),
        }
        for e in evidence
    ]
    return {
        "raw_fingerprint": raw_fp,
        "item_count": sum(t["cluster_size"] for t in themes),
        "theme_count": len(themes),
        "themes": themes,
    }


# ── prompt input + parsing ─────────────────────────────────────────────────── #

def build_portrait_prompt_input(person: str, evidence: list[dict]) -> str:
    """Render one person's clustered evidence as the synthesis prompt's body.

    Each theme carries how many distinct conversations it surfaced in, because that is the
    difference between "he said this once" and "this is how he is" — and the prompt asks
    for that difference to show in the prose rather than being flattened into a bullet.
    """
    lines = [f"What you have come to notice about {person}:", ""]
    for e in evidence:
        n = e.get("recurrences", 1)
        seen = "once" if n <= 1 else f"across {n} separate conversations"
        lines.append(f"- {e.get('content', '').strip()}  [{seen}]")
    return "\n".join(lines)


def parse_portrait(text: str) -> dict:
    """Lenient parse of the five-facet prose into a structured portrait dict.

    Lenient about *shape*, never about provenance: `strip_think` cuts an unterminated
    ``<think>`` as well as closed ones, so a generation truncated mid-thought yields no
    body rather than a portrait assembled out of the section labels this prompt taught
    the model to rehearse while thinking.
    """
    body = strip_think(text)
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1).upper().replace(" ", "_")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()
    who = " ".join(l.strip() for l in sections.get("WHO", "").splitlines()
                   if l.strip()).strip()
    return {
        "who": who,
        "cares": _bullets(sections.get("CARES", "")),
        "ways": _bullets(sections.get("WAYS", "")),
        "with_me": _bullets(sections.get("WITH_ME", "")),
        "unsure": _bullets(sections.get("UNSURE", "")),
    }


def load_portrait_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the portrait synthesis prompt from ``prompts/user_portrait_prompt.txt``."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (Path(prompts_dir) / "user_portrait_prompt.txt").read_text(
        encoding="utf-8").strip()


# ── rendering ──────────────────────────────────────────────────────────────── #

def render_portrait_for_chat(portrait: dict) -> str:
    """Render *portrait* as the standing block injected while this person is speaking.

    Framing notes, all of which are the point rather than decoration:

      * It is addressed to her in the second person (it is an instruction about a block)
        while the body stays in her own first-person voice — the same shape
        ``render_digest_for_chat`` uses, so the two standing blocks read as one register.
      * It carries the **do-not-perform guard**, for a sharper reason than the persona
        block has: reciting what you know about someone back at them is not merely a
        stylistic failure, it is the specific unnerving thing a system that has been
        taking notes does. She should sound like someone who knows them, not like someone
        reading their file.
      * UNSURE is rendered *last and explicitly as open*, so the block ends on what she
        does not know. A portrait whose final line is a conclusion invites her to act on
        the whole thing as settled.

    Returns ``""`` when nothing usable is left, so the caller simply injects no block.
    """
    if not portrait:
        return ""
    person = (portrait.get("person") or "").strip() or "this person"
    parts: list[str] = []
    who = (portrait.get("who") or "").strip()
    if who:
        parts.append(who)

    def _block(title: str, key: str) -> None:
        items = [i.strip() for i in (portrait.get(key) or []) if (i or "").strip()]
        if items:
            parts.append(title + "\n" + "\n".join("- " + i for i in items))

    _block("What they keep coming back to:", "cares")
    _block("How they talk:", "ways")
    _block("How it goes between us:", "with_me")

    body = "\n\n".join(parts).strip()
    unsure = [i.strip() for i in (portrait.get("unsure") or []) if (i or "").strip()]
    if not body and not unsure:
        return ""
    out = (
        f"This is {person}, as you have come to know them — not notes to consult but "
        "someone you already know, here so you meet them where you left off instead of "
        "starting from nothing. Speak from it, never about it: do not recite what you "
        "know of them back at them, do not treat a reading as a verdict, and let this "
        "particular conversation be what it is.\n\n"
    )
    if body:
        out += body
    if unsure:
        out += (("\n\n" if body else "")
                + "What you still don't know about them, and should stay open to:\n"
                + "\n".join("- " + i for i in unsure))
    return out


def render_portrait_plain(portrait: dict) -> str:
    """Flat human-readable rendering (operator/debug surfaces, event logs)."""
    if not portrait:
        return ""
    lines = [f"# {portrait.get('person', '?')}"]
    if portrait.get("who"):
        lines += ["", portrait["who"]]
    for title, key in (("CARES", "cares"), ("WAYS", "ways"),
                       ("WITH ME", "with_me"), ("UNSURE", "unsure")):
        items = [i for i in (portrait.get(key) or []) if (i or "").strip()]
        if items:
            lines += ["", title] + ["- " + i for i in items]
    return "\n".join(lines)


# ── orchestration (needs a generate_fn) ────────────────────────────────────── #

def plan_portraits(memory_dir: Path, *, users_dir: Path,
                   fallback_memory_dir: Optional[Path] = None,
                   force: bool = False) -> list[dict]:
    """The **model-free** first seam: who needs a portrait regenerated, and from what?

    Split out of the synthesis for the same reason ``reflection_digest.plan_digest`` is:
    the caller pays a real cost to reach the model (the reflection run opens a clean-base
    window to cluster in), and an unchanged run must be able to decline *before* paying
    it. Filesystem-only and deterministic.

    Returns one plan per person needing work: ``[{person, slug, items, raw_fp}]``.
    """
    people = gather_user_raw(memory_dir, fallback_memory_dir=fallback_memory_dir)
    out: list[dict] = []
    for slug, entry in sorted(people.items()):
        items = entry["items"]
        if len(items) < MIN_EVIDENCE_ITEMS:
            continue
        if not force and not should_regenerate(users_dir, entry["person"], items):
            continue
        out.append({"person": entry["person"], "slug": slug, "items": items,
                    "raw_fp": raw_fingerprint(items)})
    return out


def cluster_for_portrait(items: list[dict], *, generate_fn: Optional[Callable] = None,
                         embedder=None) -> list[dict]:
    """The **clustering** seam — group one person's evidence into themes.

    Grouping paraphrases is an *evaluation* ("do these two say the same thing?"), not
    Ava's expression, so the reflection run calls this inside the clean-base window it
    already opens for the branch judge, fact placement and persona clustering, while the
    portrait synthesis stays on the adapter.

    Reuses ``reflection_digest.cluster_persona_evidence`` — the single flat grouping call,
    with the embedder and exact-key tiers behind it. Deliberately NOT the map-reduce path:
    that exists to survive hundreds of statements in one prompt, and one person's evidence
    is nowhere near that scale (see the module docstring).
    """
    if not items:
        return []
    return cluster_persona_evidence(items, generate_fn=generate_fn, embedder=embedder)


def synthesize_portrait(
    person: str,
    evidence: list[dict],
    *,
    generate_fn: Callable,
    users_dir: Path,
    run_id: str,
    raw_fp: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens_setting: str = PORTRAIT_MAX_NEW_TOKENS,
) -> dict:
    """The **synthesis** seam — write *person*'s portrait from clustered evidence.

    Runs on the adapter: this is Ava's own reading of someone, in her voice, which is
    exactly the thing the clean base must not author (the clustering that fed it was an
    evaluation, so that half belongs adapter-off). RAG is disabled and the pass is
    unfenced — the evidence in the prompt is the whole input, and a retrieval block would
    only reintroduce the situational fragments the portrait exists to replace.

    The token budget (:data:`PORTRAIT_MAX_NEW_TOKENS`) is DOUBLE the self-side digest's,
    because this pass has the stronger claim to headroom: thinking is ON (the prompt asks
    for it explicitly), so a thinking model drafts the portrait inside ``<think>`` —
    which :func:`parse_portrait` then strips — before writing the sections, out of the
    same budget the sections must fit in, and that thought scales with how much history
    Ava has with this person. The self-side digest needed 8192 to stop truncating a FOUR-
    section answer; this one has five, of which UNSURE — the section that keeps a reading
    of a real person from hardening into a verdict — is last and so is the first thing a
    tight budget deletes. Truncation mid-``<think>`` is the quieter failure: it yields no
    body, so no portrait is written at all and the skip reports ``truncated`` rather than
    a bad portrait (see below) — and it does not self-correct, since the next run re-reads
    the same, by then larger, evidence.

    Returns a status summary; never raises.
    """
    if not evidence:
        return {"status": "skipped", "reason": "no evidence", "person": person}
    try:
        sys_prompt = prompt if prompt is not None else load_portrait_prompt()
        sys_prompt = sys_prompt.replace("{person}", person)
        body = build_portrait_prompt_input(person, evidence)
        raw = generate_fn(
            body, sys_prompt,
            temperature=temperature, top_p=top_p,
            max_new_tokens_setting=max_new_tokens_setting,
            before_session="", disable_rag=True,
        )
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                "person": person}

    portrait = parse_portrait(raw or "")
    if not any(portrait.get(k) for k in ("who", "cares", "ways", "with_me", "unsure")):
        # Carry the evidence for WHY nothing parsed, or the skip is unactionable: one
        # reason string cannot distinguish a generation truncated inside its <think>
        # (now stripped to nothing by `strip_think` rather than parsed as a portrait)
        # from a model that ignored the five section labels. `last_truncated` is set on
        # the generate callable by the reflection generate seam; absent on a caller that
        # does not publish it, hence the getattr. The raw is clipped — it rides a run
        # event into the store and the activity journal.
        raw_text = raw or ""
        return {"status": "skipped", "reason": "unparseable or empty", "person": person,
                "truncated": getattr(generate_fn, "last_truncated", None),
                "raw_chars": len(raw_text),
                "raw": raw_text[:_SKIP_RAW_CLIP]}

    try:
        path = write_portrait(users_dir, person, portrait, raw_text=raw or "",
                              run_id=run_id, evidence=evidence, raw_fp=raw_fp)
    except Exception as e:
        return {"status": "failed", "reason": f"write error: {e}", "person": person}

    return {
        "status": "written",
        "person": person,
        "path": str(path) if path else "",
        "themes": len(evidence),
        "counts": {
            "cares": len(portrait.get("cares") or []),
            "ways": len(portrait.get("ways") or []),
            "with_me": len(portrait.get("with_me") or []),
            "unsure": len(portrait.get("unsure") or []),
        },
        "portrait": portrait,
    }


# ── GPU-free self-test ─────────────────────────────────────────────────────── #

def _selftest() -> None:
    import tempfile

    # -- identity ----------------------------------------------------------- #
    assert person_slug("Artemy") == "artemy"
    assert person_slug("artemy voikhansky") == "artemy"   # first token, one identity
    assert person_slug("Artemy!") == "artemy"
    assert person_slug("the user") == ""                  # names nobody
    assert person_slug("") == ""
    ru = person_slug("Артемий")
    assert ru.startswith("p-") and person_slug("артемий") == ru

    # -- fold --------------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        users = Path(td) / "users"
        recs = [
            # Two sessions restating one reading → recurrence 2 after clustering.
            {"op": "insert", "kind": "impression", "key": "k1", "about": "Artemy",
             "content": "he argues against his own ideas to test them",
             "source_session": "20260701_100000", "ts": "2026-07-01T10:00:00"},
            {"op": "insert", "kind": "impression", "key": "k2", "about": "artemy v",
             "content": "he pushes back on his own proposals to see if they hold",
             "source_session": "20260702_100000", "ts": "2026-07-02T10:00:00"},
            {"op": "insert", "kind": "impression", "key": "k3", "about": "Artemy",
             "content": "goes quiet when something actually matters",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:00:00"},
            {"op": "insert", "kind": "fact", "key": "k4", "about": "Artemy",
             "source": "Artemy", "source_class": "self",
             "content": "works on an AI project of his own",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:05:00"},
            # Hearsay about a third party — recallable, never portrait-forming.
            {"op": "insert", "kind": "fact", "key": "k5", "about": "Boris",
             "source": "Artemy", "source_class": "hearsay",
             "content": "Boris quit his job",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:06:00"},
            # Unattributed world fact — no subject, so no portrait.
            {"op": "insert", "kind": "fact", "key": "k6",
             "content": "FAISS is a vector index",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:07:00"},
            # Evicted impression must not survive the fold.
            {"op": "insert", "kind": "impression", "key": "k7", "about": "Artemy",
             "content": "dislikes long replies",
             "source_session": "20260704_100000", "ts": "2026-07-04T10:00:00"},
            {"op": "evict", "key": "k7", "ts": "2026-07-05T10:00:00"},
            # A persona statement is Ava's own — never a user portrait's evidence.
            {"op": "insert", "kind": "persona", "key": "k8",
             "content": "I would rather be wrong out loud than vague",
             "source_session": "20260704_100000", "ts": "2026-07-04T10:01:00"},
        ]
        (mem / "rag_memory.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

        people = gather_user_raw(mem)
        assert set(people) == {"artemy"}, people           # Boris/world/persona excluded
        entry = people["artemy"]
        keys = {i["key"] for i in entry["items"]}
        assert keys == {"k1", "k2", "k3", "k4"}, keys      # k7 evicted, k5/k6/k8 excluded
        assert entry["person"] == "Artemy"                 # display spelling preserved
        assert all(i["sessions"] for i in entry["items"])

        items = entry["items"]
        fp = raw_fingerprint(items)
        assert fp == raw_fingerprint(list(reversed(items)))  # order-independent

        # -- regenerate gate ------------------------------------------------ #
        assert should_regenerate(users, "Artemy", items)     # nothing written yet
        assert not should_regenerate(users, "Artemy", items[:2])  # under the floor
        plans = plan_portraits(mem, users_dir=users)
        assert len(plans) == 1 and plans[0]["person"] == "Artemy"

        # -- parse + write + reload ----------------------------------------- #
        raw_text = (
            "<think>weighing it</think>\n"
            "WHO\nAn engineer building his own mind-shaped thing, and impatient with "
            "anything that sounds like it came off a shelf.\n\n"
            "CARES\n- whether an idea survives being attacked\n- his own project\n\n"
            "WAYS\n- terse; drops the pleasantries\n\n"
            "WITH_ME\n- wants the objection, not the agreement\n\n"
            "UNSURE\n- whether the quiet is thinking or something else\n"
        )
        parsed = parse_portrait(raw_text)
        assert parsed["who"].startswith("An engineer")
        assert "weighing it" not in parsed["who"]           # CoT stripped
        assert len(parsed["cares"]) == 2 and len(parsed["ways"]) == 1
        assert len(parsed["with_me"]) == 1 and len(parsed["unsure"]) == 1

        # A generation truncated inside its <think> must parse to NOTHING, even though
        # the scratch reasoning rehearses the section labels the prompt named. Before
        # `strip_think` cut dangling blocks this produced a portrait out of unfinished
        # thinking — and a portrait is injected on every turn with that person.
        truncated = ("<think>Let me work out who he is. Something like\n"
                     "WHO\nstubborn about definitions, maybe? not sure that holds\n"
                     "UNSURE\n- whether the impatience is with me\n"
                     "Actually no, let me reconsider")
        blank = parse_portrait(truncated)
        assert not any(blank[k] for k in
                       ("who", "cares", "ways", "with_me", "unsure"))

        ev = cluster_for_portrait(items, generate_fn=None, embedder=_FakeEmbedder())
        assert ev and sum(e["cluster_size"] for e in ev) == len(items)
        p = write_portrait(users, "Artemy", parsed, raw_text=raw_text,
                           run_id="r1", evidence=ev, raw_fp=fp)
        assert p is not None and p.name == "artemy.json"
        assert write_portrait(users, "the user", parsed, raw_text="", run_id="r1",
                              evidence=ev) is None          # nameless → no file

        back = latest_portrait(users, "artemy voikhansky")   # same identity, same file
        assert back is not None and back["person"] == "Artemy"
        assert back["evidence"]["raw_fingerprint"] == fp
        assert known_people(users) == ["Artemy"]
        assert not should_regenerate(users, "Artemy", items)  # unchanged ⇒ no-op
        assert plan_portraits(mem, users_dir=users) == []
        assert plan_portraits(mem, users_dir=users, force=True)  # force overrides

        # -- rendering ------------------------------------------------------ #
        block = render_portrait_for_chat(back)
        assert "Artemy" in block and "his own project" in block
        assert "still don't know" in block                  # UNSURE rendered, and last
        assert block.index("still don't know") > block.index("How they talk")
        assert render_portrait_for_chat({}) == ""
        assert render_portrait_for_chat({"person": "X"}) == ""   # nothing usable
        assert "CARES" in render_portrait_plain(back)

        body = build_portrait_prompt_input("Artemy", ev)
        assert "Artemy" in body and ("once" in body or "conversations" in body)

    print("user_digest selftest OK")


class _FakeEmbedder:
    """Deterministic bag-of-words embedder so the self-test needs no model download."""

    def encode(self, sentences, **_kw):
        import numpy as np
        vocab: dict = {}
        rows = []
        toks = [re.findall(r"\w+", s.lower()) for s in sentences]
        for t in toks:
            for w in t:
                vocab.setdefault(w, len(vocab))
        for t in toks:
            v = np.zeros(max(len(vocab), 1), dtype="float32")
            for w in t:
                v[vocab[w]] += 1.0
            n = float(np.linalg.norm(v)) or 1.0
            rows.append(v / n)
        return np.vstack(rows)


if __name__ == "__main__":
    import sys
    # Make the sibling `training` package importable when run as a module from
    # server/inference (mirrors reflection_digest and the other GPU-free self-tests).
    _server = Path(__file__).resolve().parent.parent.parent
    if str(_server) not in sys.path:
        sys.path.insert(0, str(_server))
    _selftest()
