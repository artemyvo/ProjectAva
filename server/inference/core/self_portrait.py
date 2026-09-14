"""**Outside-view self-portrait** — how Ava comes across, as against what she takes
herself to be.

This is the third portrait in the family and the one with no prior. ``reflection_digest``
folds her ``[persona]`` self-statements into a standing self-portrait; ``user_digest``
folds her ``[impression]`` readings of a person into a standing portrait of *them*. Both
of those are readings *she* authors from the inside. This module folds
``[self_impression]`` — her reading of **her own transcripts, read as a reader**.

Why it is not the persona digest
--------------------------------
Different evidence, produced by a different pass, answering a different question:

* A ``[persona]`` statement comes from the **revision pass**, which sees her ``<think>``
  next to her reply. It is *introspective*: "this is the disposition I endorse."
* A ``[self_impression]`` comes from a pass reading the session through
  ``reflection_source.build_session_reading_content``, which renders the transcript **with
  no CoT at all**. The outside view is therefore enforced by the builder rather than
  merely requested by the prompt: she is looking at what she *said*, with no access to
  what she was thinking when she said it — which is exactly the material a reader has.

So the two are independent readings of the same conversations, and the interesting
quantity is the **gap** between them. That is also why nothing here feeds the digest:
both artifacts derive from transcripts produced *under* the injected digest, so wiring
this one into digest synthesis would extend the existing self-reinforcement loop by a hop
without admitting any independent evidence. They are kept apart on purpose.

Why it is not just a ``users/`` entry
-------------------------------------
The fold machinery is genuinely the same (that is why the primitives below are imported
rather than reimplemented), but four things differ and each of them is load-bearing:

* **Facets.** ``WITH_ME`` ("what they want from her, what lands") is incoherent pointed
  inward, so it becomes ``HOW_I_LAND`` — the effect her replies actually have. That facet
  is this portrait's whole reason to exist: neither the digest nor the transcript states
  it anywhere.
* **No attribution.** A self-impression is about her by construction, so it carries no
  ``about``/``source``. Written as an ``[impression] (about: Ava)`` instead, it would
  derive ``source_class == "hearsay"`` (subject ≠ speaker) and render as "— about Ava, per
  Artemy", which is both wrong and unfixable without special-casing the name.
* **Its own kind.** ``[self_impression]`` keeps it out of the ``[impression]`` retrieval
  channel, where it would surface mid-conversation under "how they've come to seem to
  you — about Ava".
* **A reserved slug.** It lives at ``users/_self.json`` so it inherits the users dir's
  per-run archival, wipe semantics and snapshot travel for free, while
  ``user_digest.person_slug`` refuses the reserved slug outright — so no human name can
  ever resolve to this file and ``generation._current_user_portrait`` can never inject it.

Nothing injects this portrait
-----------------------------
Deliberately: there is no ``render_*_for_chat`` here. A second standing self-block would
compete with the digest for the same slot, and an outside view injected into the very
turns it is later folded from is the tightest possible version of the loop this module is
trying to stay out of. It is an observation instrument — visible in the run log, the
Debug tab and the per-run archive — and the evidence a later "what do I want to be" pass
would read.

GPU-free self-test: ``python -m core.self_portrait``.
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
from core.user_digest import (             # and the portrait primitives, likewise
    SELF_SLUG,
    _evidence_summary,
    build_portrait_prompt_input,
    raw_fingerprint,
)

# The op-log kind. Its own kind rather than an attributed `[impression]`: see the module
# docstring — attribution would misclassify it as hearsay, and sharing the kind would put
# it in a retrieval channel whose framing ("how they've come to seem to you") is wrong for
# a reading of herself.
SELF_KIND = "self_impression"

# Facets. The user portrait's five, with WHO/CARES renamed to read as observation rather
# than acquaintance and WITH_ME replaced outright:
#   SEEMS       — who she appears to be from the words alone (WHO).
#   RETURNS_TO  — what she keeps circling back to (CARES).
#   WAYS        — how she actually talks, observed rather than as she describes it (WAYS).
#   HOW_I_LAND  — the effect her replies have on the person reading them (was WITH_ME).
#   UNSURE      — what the transcripts do not settle.
# HOW_I_LAND is the addition with no analogue anywhere else in the system, and UNSURE is
# kept for a sharper reason than it was on the user side: the persona digest has no place
# at all to record "I may be reading myself wrong" — every facet of it is a conclusion.
_SECTION_RE = re.compile(
    _section("SEEMS", "RETURNS_TO", "RETURNS TO", "WAYS", "HOW_I_LAND", "HOW I LAND",
             "UNSURE"),
    re.IGNORECASE | re.MULTILINE,
)

_FACET_KEYS = ("seems", "returns_to", "ways", "how_i_land", "unsure")

# Minimum live observations before a portrait is written at all — the user portrait's
# floor, for the same reason: below it the fold is a handful of first readings and
# synthesizing "how I come across" from three lines produces confident nonsense.
MIN_EVIDENCE_ITEMS = 4

# Clipped raw carried on an unparseable synthesis (see user_digest._SKIP_RAW_CLIP).
_SKIP_RAW_CLIP = 1500

# Same budget as the user portrait, for the same reason: thinking is on, the <think>
# block scales with the accumulated evidence rather than with the answer, and a cut that
# lands mid-thought yields nothing to parse and does not self-correct (the next run
# re-reads the same, by then larger, evidence).
PORTRAIT_MAX_NEW_TOKENS = "16384"

# What the portrait calls its subject in prose. Not a person's name: this artifact is
# never addressed to anyone and never injected, so a display name would only invite the
# synthesis to write about "Ava" in the third person.
SELF_DISPLAY = "you"


# ── the model-free fold ────────────────────────────────────────────────────── #

def gather_self_raw(memory_dir: Path, *, fallback_memory_dir: Optional[Path] = None,
                    now=None) -> list[dict]:
    """Replay ``rag_memory.jsonl`` into this portrait's raw evidence items.

    The single-subject counterpart of ``user_digest.gather_user_raw``: same item shape
    (``{key, content, stage, sessions, session_weights, recency_band}``) so
    ``cluster_persona_evidence`` and its ``_evidence_entry`` collapse apply unchanged, and
    the same reason for reading the RAW op-log rather than the folded state — the maturity
    signal is the set of **distinct sessions** an observation recurred in, which any
    fold-to-current-value collapses away.

    It collects by KIND alone and needs no ``about``: a ``[self_impression]`` has exactly
    one possible subject. Because recurrence counts distinct ``source_session`` values, a
    revisit re-reading a conversation it has already read cannot vote twice on the same
    evidence.

    Deterministic given *now*, which is what lets :func:`raw_fingerprint` gate
    regeneration.
    """
    if now is None:
        now = datetime.now()

    items: dict[str, dict] = {}
    session_ts: dict[str, dict[str, str]] = {}

    lines: list[str] = []
    for d in [p for p in (fallback_memory_dir, memory_dir) if p is not None]:
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
        if op != "insert" or rec.get("kind") != SELF_KIND:
            continue
        if not (rec.get("content") or "").strip():
            continue
        items[key] = rec
        src = (rec.get("source_session") or "").strip()
        if src:
            ts = (rec.get("ts") or "").strip()
            prev = session_ts.setdefault(key, {}).get(src, "")
            if ts and ts > prev:   # ISO compares lexically == chronologically
                session_ts[key][src] = ts

    out: list[dict] = []
    for key, rec in items.items():
        per_session = session_ts.get(key, {})
        sessions = set(per_session)
        weights = {s: _persona_recency_weight(_age_days(per_session.get(s, ""), now))
                   for s in sessions}
        band = _recency_band(max(weights.values())) if weights else _RECENCY_BANDS
        out.append({
            "key": key,
            "content": (rec.get("content") or "").strip(),
            "kind": SELF_KIND,
            "stage": 0,
            "sessions": sessions,
            "session_weights": weights,
            "recency_band": band,
            "source_session": (rec.get("source_session") or "").strip(),
        })
    return out


# ── persistence ────────────────────────────────────────────────────────────── #

def portrait_path(users_dir: Path) -> Path:
    """``<users_dir>/_self.json`` — the reserved slug no person can resolve to."""
    return Path(users_dir) / (SELF_SLUG + ".json")


def latest_portrait(users_dir: Path) -> Optional[dict]:
    """Load the outside-view portrait, or None when unwritten/unreadable."""
    p = portrait_path(users_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_regenerate(users_dir: Path, items: list[dict]) -> bool:
    """True when there is no portrait yet, or the raw evidence has changed."""
    if len(items) < MIN_EVIDENCE_ITEMS:
        return False
    cur = latest_portrait(users_dir)
    if not cur:
        return True
    return cur.get("evidence", {}).get("raw_fingerprint") != raw_fingerprint(items)


def write_portrait(users_dir: Path, portrait: dict, *, raw_text: str, run_id: str,
                   evidence: list[dict], raw_fp: Optional[str] = None) -> Path:
    """Atomically write the outside-view portrait file."""
    from training.reflections_path import atomic_write_text

    path = portrait_path(users_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        # `kind` distinguishes this file from the per-person portraits it sits beside, for
        # any reader that globs the users dir (the archive copies all of them).
        "kind": "outside_view",
        "slug": SELF_SLUG,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "seems": portrait.get("seems", ""),
        "returns_to": portrait.get("returns_to", []),
        "ways": portrait.get("ways", []),
        "how_i_land": portrait.get("how_i_land", []),
        "unsure": portrait.get("unsure", []),
        "evidence": _evidence_summary(evidence, raw_fp),
        "raw": raw_text or "",
    }
    atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return path


# ── parsing + rendering ────────────────────────────────────────────────────── #

def parse_self_portrait(text: str) -> dict:
    """Lenient parse of the five-facet prose into a structured portrait dict.

    Lenient about *shape*, never about provenance: ``strip_think`` cuts an unterminated
    ``<think>`` as well as a closed one, so a generation truncated mid-thought yields no
    body rather than a portrait assembled out of the section labels the prompt taught the
    model to rehearse while thinking (the bug both sibling parsers had).
    """
    body = strip_think(text)
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1).upper().replace(" ", "_")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()
    seems = " ".join(l.strip() for l in sections.get("SEEMS", "").splitlines()
                     if l.strip()).strip()
    return {
        "seems": seems,
        "returns_to": _bullets(sections.get("RETURNS_TO", "")),
        "ways": _bullets(sections.get("WAYS", "")),
        "how_i_land": _bullets(sections.get("HOW_I_LAND", "")),
        "unsure": _bullets(sections.get("UNSURE", "")),
    }


def load_portrait_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the synthesis prompt from ``prompts/self_portrait_prompt.txt``."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    return (Path(prompts_dir) / "self_portrait_prompt.txt").read_text(
        encoding="utf-8").strip()


def render_portrait_plain(portrait: dict) -> str:
    """Flat human-readable rendering — the ONLY renderer this module has.

    There is deliberately no chat renderer (see the module docstring): this portrait is
    read by the operator and archived, never injected. It rides the synthesis run event
    the way the user portrait's does, so the Sleep tab and the activity journal show what
    she actually concluded rather than only that something was written.
    """
    if not portrait:
        return ""
    lines = ["# how you come across (outside view)"]
    if portrait.get("seems"):
        lines += ["", portrait["seems"]]
    for title, key in (("RETURNS TO", "returns_to"), ("WAYS", "ways"),
                       ("HOW I LAND", "how_i_land"), ("UNSURE", "unsure")):
        items = [i for i in (portrait.get(key) or []) if (i or "").strip()]
        if items:
            lines += ["", title] + ["- " + i for i in items]
    return "\n".join(lines)


# ── orchestration (needs a generate_fn) ────────────────────────────────────── #

def plan_portrait(memory_dir: Path, *, users_dir: Path,
                  fallback_memory_dir: Optional[Path] = None,
                  force: bool = False) -> Optional[dict]:
    """The **model-free** first seam: does the portrait need regenerating, and from what?

    Split out for the same reason ``reflection_digest.plan_digest`` and
    ``user_digest.plan_portraits`` are: the caller pays a real cost to reach the model (the
    reflection run opens a clean-base window to cluster in), and a run with no new material
    must be able to decline *before* paying it. Filesystem-only and deterministic.

    Reads **live** memory only, never a run's staging — the portrait file is written
    straight to the live users dir, so folding it from staged observations would leave it
    standing on evidence a later discard erases. The cost is a one-run lag, which is the
    cadence both sibling artifacts already have.

    Returns ``{items, raw_fp}`` or None when there is nothing to do.
    """
    items = gather_self_raw(memory_dir, fallback_memory_dir=fallback_memory_dir)
    if len(items) < MIN_EVIDENCE_ITEMS:
        return None
    if not force and not should_regenerate(users_dir, items):
        return None
    return {"items": items, "raw_fp": raw_fingerprint(items)}


def cluster_evidence(items: list[dict], *, generate_fn: Optional[Callable] = None,
                     embedder=None) -> list[dict]:
    """The **clustering** seam — group the observations into themes.

    Grouping paraphrases is an *evaluation* ("do these two say the same thing?"), not Ava's
    expression, so the reflection run calls this inside the clean-base window it already
    opens for the branch judge, fact placement, persona clustering and fact dedup, while
    the synthesis below stays on the adapter.

    The single flat call, not map-reduce: that exists to survive hundreds of statements in
    one prompt, and one subject's observations are nowhere near that scale.
    """
    if not items:
        return []
    return cluster_persona_evidence(items, generate_fn=generate_fn, embedder=embedder)


def synthesize_portrait(
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
    """The **synthesis** seam — write the portrait from clustered evidence.

    Runs on the adapter, like both sibling syntheses: reading her own transcripts back is
    still her reading, in her voice, and the clean base's job was the evaluation that fed
    it. RAG is disabled and the pass unfenced — the clustered evidence in the prompt is the
    whole input, and retrieval would only reintroduce the situational fragments a standing
    reading exists to replace.

    Returns a status summary; never raises.
    """
    if not evidence:
        return {"status": "skipped", "reason": "no evidence"}
    try:
        sys_prompt = prompt if prompt is not None else load_portrait_prompt()
        body = build_portrait_prompt_input(SELF_DISPLAY, evidence)
        raw = generate_fn(
            body, sys_prompt,
            temperature=temperature, top_p=top_p,
            max_new_tokens_setting=max_new_tokens_setting,
            before_session="", disable_rag=True,
        )
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    portrait = parse_self_portrait(raw or "")
    if not any(portrait.get(k) for k in _FACET_KEYS):
        # Carry the raw for WHY nothing parsed — one reason string cannot distinguish a
        # generation cut inside its <think> (stripped to nothing above) from a model that
        # ignored the section labels. Clipped: it rides a run event into the store and the
        # activity journal.
        raw_text = raw or ""
        return {"status": "skipped", "reason": "unparseable or empty",
                "truncated": getattr(generate_fn, "last_truncated", None),
                "raw_chars": len(raw_text), "raw": raw_text[:_SKIP_RAW_CLIP]}

    try:
        path = write_portrait(users_dir, portrait, raw_text=raw or "", run_id=run_id,
                              evidence=evidence, raw_fp=raw_fp)
    except Exception as e:
        return {"status": "failed", "reason": f"write error: {e}"}

    return {
        "status": "written",
        "path": str(path),
        "themes": len(evidence),
        "counts": {k: (1 if isinstance(portrait.get(k), str) and portrait.get(k)
                       else len(portrait.get(k) or []))
                   for k in _FACET_KEYS},
        "portrait": portrait,
    }


# ── GPU-free self-test ─────────────────────────────────────────────────────── #

def _selftest() -> None:
    import tempfile
    from core.user_digest import person_slug

    # The reserved slug is unreachable from any name, which is what makes the
    # "never injected as a user portrait" guarantee structural rather than a check.
    assert person_slug(SELF_SLUG) == ""
    assert person_slug("_self") == ""
    assert person_slug("Ava") == "ava" != SELF_SLUG

    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        users = Path(td) / "users"
        recs = [
            # Two sessions restating one observation → recurrence 2 after clustering.
            {"op": "insert", "kind": SELF_KIND, "key": "s1",
             "content": "she answers the question under the question, not the one asked",
             "source_session": "20260701_100000", "ts": "2026-07-01T10:00:00"},
            {"op": "insert", "kind": SELF_KIND, "key": "s2",
             "content": "she replies to what was meant rather than what was written",
             "source_session": "20260702_100000", "ts": "2026-07-02T10:00:00"},
            {"op": "insert", "kind": SELF_KIND, "key": "s3",
             "content": "her replies get longer when she is least sure",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:00:00"},
            {"op": "insert", "kind": SELF_KIND, "key": "s4",
             "content": "she ends on a hedge more often than she thinks",
             "source_session": "20260703_100000", "ts": "2026-07-03T10:05:00"},
            # Evicted → must not survive the fold.
            {"op": "insert", "kind": SELF_KIND, "key": "s5",
             "content": "she never asks anything back",
             "source_session": "20260704_100000", "ts": "2026-07-04T10:00:00"},
            {"op": "evict", "key": "s5", "ts": "2026-07-05T10:00:00"},
            # Other kinds are not this portrait's evidence — including an [impression]
            # about a person and a [persona] self-statement (the digest's material).
            {"op": "insert", "kind": "impression", "key": "s6", "about": "Artemy",
             "content": "he argues against his own ideas",
             "source_session": "20260704_100000", "ts": "2026-07-04T10:01:00"},
            {"op": "insert", "kind": "persona", "key": "s7",
             "content": "I would rather be wrong out loud than vague",
             "source_session": "20260704_100000", "ts": "2026-07-04T10:02:00"},
        ]
        (mem / "rag_memory.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

        items = gather_self_raw(mem)
        assert {i["key"] for i in items} == {"s1", "s2", "s3", "s4"}, items
        # Recurrence counts distinct source_session, so a re-read of one chat cannot
        # vote twice — s3 and s4 share a session.
        assert all(len(i["sessions"]) == 1 for i in items)
        fp = raw_fingerprint(items)
        assert fp == raw_fingerprint(list(reversed(items)))     # order-independent

        assert should_regenerate(users, items)                  # nothing written yet
        assert not should_regenerate(users, items[:2])          # under the floor
        plan = plan_portrait(mem, users_dir=users)
        assert plan and len(plan["items"]) == 4

        # -- parse ---------------------------------------------------------- #
        raw = (
            "<think>drafting the sections here</think>\n"
            "SEEMS\nSomeone who is more careful than she sounds.\n\n"
            "RETURNS_TO\n- whether a thing is actually true\n\n"
            "WAYS\n- long sentences, few of them\n\n"
            "HOW I LAND\n- read as blunt more often than she means to be\n\n"
            "UNSURE\n- whether the bluntness is hers or the register's\n"
        )
        p = parse_self_portrait(raw)
        assert p["seems"].startswith("Someone who is more careful")
        assert "drafting the sections" not in p["seems"]        # think stripped
        assert p["returns_to"] and p["ways"] and p["how_i_land"] and p["unsure"]
        assert "HOW I LAND" in render_portrait_plain(p)

        # A generation cut mid-<think> yields nothing, never a portrait built from the
        # labels it was rehearsing while thinking.
        cut = parse_self_portrait("<think>SEEMS\nsomeone who\n\nRETURNS_TO\n- x")
        assert not any(cut.get(k) for k in _FACET_KEYS), cut

        # -- write + reload ------------------------------------------------- #
        path = write_portrait(users, p, raw_text=raw, run_id="r1",
                              evidence=[{"content": "c", "recurrences": 2}],
                              raw_fp=fp)
        assert path.name == SELF_SLUG + ".json"
        back = latest_portrait(users)
        assert back and back["kind"] == "outside_view"
        assert back["how_i_land"] == p["how_i_land"]
        # Written with this fingerprint → the gate now declines.
        assert not should_regenerate(users, items)
        assert plan_portrait(mem, users_dir=users) is None
        assert plan_portrait(mem, users_dir=users, force=True) is not None

    print("self_portrait selftest OK")


if __name__ == "__main__":
    import sys
    # Make the sibling `training` package importable when run as a module from
    # server/inference (mirrors user_digest / reflection_digest).
    _server = Path(__file__).resolve().parent.parent.parent
    if str(_server) not in sys.path:
        sys.path.insert(0, str(_server))
    _selftest()
