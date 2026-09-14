"""Reflection artifact writer — turns Sleep-phase output into routed training data.

The reflection loop produces two kinds of model output:

  * Consolidation (one per session) → ``## WEIGHTS`` / ``## RAG`` / ``## RESOLVED``
    sections whose items the model has tagged ``[fact]`` / ``[ask]`` / ``[resolved]``.
    WEIGHTS here holds only stable facts; the model decides weights vs RAG.
  * Revision judgement (one per exchange) → ``VERDICT: keep|revise`` / ``WHY:`` /
    ``PERSONA_TARGET:``. A revised IDEAL is generated separately from the clean pre-answer
    dialogue context and passed to the writer structurally. Persona formation lives here:
    the revision pass is the one that sees the original ``<think>`` CoT, so it has
    the raw material — what the model actually thought next to what it said — from
    which a genuine first-person self-statement can surface. This no longer requires
    a think-vs-said mismatch: a reply that was fully owned reveals as much as a
    drifting one, so persona may accompany a ``keep`` verdict, not only a ``revise``.

This module parses that text and appends structured records to three JSONL
artifacts, routed by lifetime under ``inference/data/hot/``:

  * ``memory/weights_persona.jsonl`` — statements bound for weights: ``[persona]``
                               self-statements (from revision) and ``[fact]`` truths
                               (from consolidation WEIGHTS). This is the durable
                               provenance / future-weights source.
  * ``memory/rag_memory.jsonl`` — mutable RAG store ops Ava recalls: ``insert``
                               (ask/fact) and ``evict`` (resolved questions). Every
                               weights-bound ``[persona]``/``[fact]`` is *also*
                               mirrored here as an ``insert`` (marked
                               ``from_weights``, keyed by the same ``content_key`` as
                               its ledger anchor) so it is recalled at chat time until
                               a fact/persona→weights training cycle exists to absorb
                               it — without this it would sit in ``weights_persona``
                               unread and untrained. See ``_emit_weight_recall``.

Parsing is deliberately lenient: anything that doesn't match a known tag is
skipped rather than raising, so a malformed reflection never aborts the run.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from core import trigger_hygiene
from core.field_parse import any_label as _any_label, label as _label, section as _section


# ── text helpers ──────────────────────────────────────────────────────────────

# `hash_required`: WEIGHTS / RAG / RESOLVED are ordinary enough words that a hash-free
# match would let a line of prose beginning "RAG …" split the output into sections.
_HEADER_RE = re.compile(_section("WEIGHTS", "RAG", "RESOLVED", hash_required=True),
                        re.IGNORECASE | re.MULTILINE)
# A tag may carry an optional sub-kind, e.g. ``[ask:meta]`` / ``[ask-user]`` /
# ``[ask search]``. Only ``ask`` uses it today (triage for proactive surfacing);
# it is captured generically and ignored for the other tags.
_ITEM_RE = re.compile(
    r"^[ \t]*[-*]?[ \t]*\[(persona|fact|ask|resolved)(?:[:\- ]([a-z]+))?\][ \t]*(.*)$",
    re.IGNORECASE,
)
_ASK_KINDS = {"meta", "user", "search"}
# Structured content that, when it appears *before* the first ``<think>``, proves that
# ``<think>`` is a field/reply's own reasoning (not leading meta-CoT) — see _split_think.
# The revision field labels (VERDICT/WHY/PERSONA_TARGET/legacy PERSONA/IDEAL/LANG_DRIFT)
# plus the consolidation
# section headers (## WEIGHTS / RAG / RESOLVED).
_PRE_THINK_STRUCTURED_RE = re.compile(
    "(?im)"
    + _label("VERDICT", "WHY", "PERSONA_TARGET", "PERSONA", "IDEAL", "LANG_DRIFT",
             "COUNTER")
    + "|" + _section("WEIGHTS", "RAG", "RESOLVED", hash_required=True)
)
_CYRILLIC_RE = re.compile("[\\u0400-\\u04FF]")  # Cyrillic block, for lang detection
_WS_RE = re.compile(r"\s+")

# A revision field label at the start of a line — used to CUT any field the model
# appended *after* the IDEAL reply. ``IDEAL:`` is the last field in the prompt order,
# but the model routinely emits its flags out of order (chiefly a trailing
# ``LANG_DRIFT: no``, sometimes a repeated ``VERDICT:``/``WHY:``/``PERSONA_TARGET:``).
# Because the IDEAL value is sliced greedily to the end of the body, that stray field
# would otherwise train verbatim as part of Ava's answer. Same decoration tolerance as
# every other field matcher (core.field_parse) — and this is the site where a gap in it
# costs the most: a stray field the cut fails to recognise does not merely go unread, it
# is trained.
_TRAILING_FIELD_RE = re.compile(
    "(?im)" + _label("VERDICT", "WHY", "PERSONA_TARGET", "PERSONA", "LANG_DRIFT",
                     "COUNTER", "IDEAL")
)


def _strip_trailing_revision_fields(ideal: Optional[str]) -> Optional[str]:
    """Cut any revision field label (and everything after it) that the model appended
    *after* the IDEAL reply span, so a stray ``LANG_DRIFT: no`` (or repeated
    ``VERDICT:``/``PERSONA_TARGET:``) can never leak into the trained answer.

    Only labels appearing after the leading ``<think>…</think>`` block (the reply span)
    are treated as strays; a label the model wrote *inside* its IDEAL reasoning is left
    untouched, since only the answer span after ``</think>`` becomes trained text.
    """
    if not ideal:
        return ideal
    close = ideal.find("</think>")
    scan_from = close + len("</think>") if close != -1 else 0
    m = _TRAILING_FIELD_RE.search(ideal, scan_from)
    if m:
        return ideal[: m.start()].rstrip()
    return ideal


def normalize_key(text: str) -> str:
    """Normalize content for key derivation: case/space/punctuation-insensitive.

    Used so a re-asked question and the ``[resolved]`` that closes it collapse to
    the same key even with cosmetic differences in phrasing (quotes, trailing
    ``?``, capitalization). Drops all punctuation rather than trimming it at the
    ends, so an interior quote (``coding"``) can't desync the two forms.
    """
    t = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return _WS_RE.sub(" ", t).strip()


# Fields ``ReflectionMemory._fold`` ANNOTATES onto a live record as it replays the op-log
# — derived per read from the `surface`/`lookup` ops, never stored on an insert. A writer
# that round-trips a folded record back into the log must strip them, or the next fold
# reads a stale count that its own op replay would have produced correctly.
_FOLD_COMPUTED_FIELDS = frozenset({
    "surface_count", "surfaced_in_sessions", "last_surfaced_ts", "lookup_count",
})


def content_key(text: str) -> str:
    """Stable short hash of *text*'s normalized form — links inserts to evicts and
    de-duplicates the same item across reflection runs."""
    return hashlib.sha1(normalize_key(text).encode("utf-8")).hexdigest()[:12]


_ORIGIN_TS_FORMATS = ("%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S")


def source_origin_ts(source_session: Optional[str]) -> Optional[str]:
    """ISO timestamp of the *source chat* that produced an entry, or ``None``.

    Derived from ``source_session`` — a chat-session file stem
    (``20260705_014505``, optionally carrying a ``.json`` / ``.state.json`` suffix;
    a ``-`` time separator is tolerated) — so a ``[fact]``/``[persona]`` record is
    stamped with **when the underlying conversation happened**, not when this
    reflection run distilled it (the separate ``ts`` field).

    This is the recency clock a recall-time staleness/decay pass needs: after a full
    wipe + re-reflect, ``ts`` collapses to "now" for every re-derived entry and is
    useless as an age signal, while ``origin_ts`` stays anchored to the chat's own
    date — so a self-reinforcing stale fact/persona can be faded out of chat context.

    Mirrors ``training.decay.parse_ts``'s stem handling (kept local to avoid an
    inference→training import, same rationale as :func:`ideal_has_usable_answer`), so
    recall-time decay and train-time wall-clock decay clock age from the same instant.
    Returns ``None`` for a self-directed / non-chat source (``wiki:``/``til:`` reading,
    which has no conversation date) — the caller then omits the field.
    """
    if not source_session:
        return None
    s = str(source_session).strip()
    for suf in (".state.json", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    # Two sessions started in the same second get a ``_<n>`` collision suffix
    # (chat_logger), e.g. ``20260705_014505_1`` — strip a trailing numeric run so the
    # stem still parses instead of dropping the origin timestamp for those chats.
    candidates = [s]
    m = re.match(r"^(\d{8}[_-]\d{6})_\d+$", s)
    if m:
        candidates.append(m.group(1))
    for cand in candidates:
        for fmt in _ORIGIN_TS_FORMATS:
            try:
                return datetime.strptime(cand, fmt).isoformat()
            except ValueError:
                pass
    try:
        return datetime.fromisoformat(s).isoformat()
    except ValueError:
        return None


# ── attribution: who said it vs. who it is about ─────────────────────────────
#
# A ``[fact]`` carries two *different* people, and they only come apart when Ava
# talks with one person about another (third-party recall):
#
#   * ``source`` — who said it. Derived in CODE from the session's speaker, never
#     from the model, so it cannot be hallucinated.
#   * ``about``  — who it concerns. Model-supplied via ``(about: NAME)``, because
#     only the reader of the transcript knows whether "he moved to Berlin" is
#     about the speaker or about someone they mentioned.
#
# ``source_class`` is the derived epistemic class, and it is what gates promotion
# to the weights (see ``write_consolidation``):
#
#   * ``self``     — the subject spoke about themselves (B says B). Ava's best
#                    evidence about a person; promotes normally.
#   * ``hearsay``  — someone spoke about a third party (A says B). True that it was
#                    *said*, unverified in content; RAG-only until the subject
#                    corroborates it themselves.
#   * ``observed`` — no named third party: a world fact, Ava's own reading
#                    (``wiki:``/``til:``), or an unattributed legacy record.
#                    Promotes normally, exactly as every fact did before.
#
# Names are compared case/space-insensitively and only on their first token, so
# "Artemy" and "artemy voikhansky" are one person. A generic referent ("the user",
# "the person") is treated as unnamed — it identifies nobody, and matching on it
# would collapse two different people into one subject.

_GENERIC_REFERENTS = {
    "user", "the user", "person", "the person", "they", "them", "someone",
    "the speaker", "speaker", "interlocutor", "the interlocutor", "me", "i",
    "unknown", "n/a", "none",
}


def normalize_person(name: Optional[str]) -> str:
    """Comparison key for a person's name — ``""`` when it names nobody.

    Lowercased, whitespace-collapsed, stripped of surrounding punctuation, and
    reduced to the FIRST token so a full name matches the given name used
    elsewhere. Generic referents collapse to ``""`` (see ``_GENERIC_REFERENTS``).
    """
    s = _WS_RE.sub(" ", str(name or "")).strip().strip(".,;:!?'\"()[]").lower()
    if not s or s in _GENERIC_REFERENTS:
        return ""
    first = s.split(" ", 1)[0].strip(".,;:!?'\"()[]")
    return "" if first in _GENERIC_REFERENTS else first


def derive_source_class(source: Optional[str], about: Optional[str]) -> str:
    """Epistemic class of a statement — ``self`` / ``hearsay`` / ``observed``.

    ``observed`` whenever either side is unnamed: a world fact, Ava's own reading,
    or a legacy record written before attribution existed. That default is
    deliberate — it is the pre-attribution behaviour, so nothing already in the
    store changes class or loses its weights path.
    """
    src, abt = normalize_person(source), normalize_person(about)
    if not src or not abt:
        return "observed"
    return "self" if src == abt else "hearsay"


def _attribution_fields(*, about: Optional[str], source: Optional[str],
                        source_class: Optional[str] = None) -> dict:
    """The attribution fields to stamp on a record — empty ones omitted.

    Omitting rather than writing empty strings keeps a world fact's record byte-identical
    to its pre-attribution form, so the op-log stays readable and the fold's existing
    ``.get()`` defaults keep working on old and new lines alike. ``source_class`` is
    always written when either name is present, since it is what the promotion gate and
    the recall renderer read.
    """
    about = (about or "").strip()
    source = (source or "").strip()
    if not about and not source:
        return {}
    out: dict = {"source_class": source_class or derive_source_class(source, about)}
    if about:
        out["about"] = about
    if source:
        out["source"] = source
    return out


def _split_think(text: str) -> tuple[str, str]:
    """Separate leading ``<think>…</think>`` block(s) from the structured body.

    Returns (meta_cot, body). The reflection CoT is meta-reasoning and must never
    leak into a training target, so it is peeled off here and stored separately.

    Some models emit two think blocks back to back (``…</think><think>…``); every
    consecutive leading block is folded into meta_cot, because a second block left
    in the body is dangerous — a ``VERDICT:`` written inside it would be parsed as
    the real one.

    Guard against the inverse hazard: a ``<think>`` that is **not** leading meta-CoT
    but the reasoning of the reply inside a structured field (e.g. a revision
    ``IDEAL:`` whose reworked answer thinks first). If the text *before* the first
    ``<think>`` already carries a revision field label or a consolidation section
    header, that ``<think>`` belongs to the body — peeling it would silently drop the
    verdict body ahead of it — so return the whole text as body.
    """
    open_tag = text.find("<think>")
    if open_tag == -1:
        return "", text
    if _PRE_THINK_STRUCTURED_RE.search(text[:open_tag]):
        return "", text
    close_tag = text.find("</think>", open_tag)
    if close_tag == -1:
        # Truncated think with no close: no usable structured body.
        return text[open_tag + len("<think>"):].strip(), ""
    metas = [text[open_tag + len("<think>"):close_tag].strip()]
    body = text[close_tag + len("</think>"):]
    while True:
        stripped = body.lstrip()
        if not stripped.startswith("<think>"):
            break
        close = stripped.find("</think>")
        if close == -1:
            metas.append(stripped[len("<think>"):].strip())
            body = ""
            break
        metas.append(stripped[len("<think>"):close].strip())
        body = stripped[close + len("</think>"):]
    return "\n\n".join(m for m in metas if m), body


def _split_sections(body: str) -> dict[str, str]:
    """Slice the body into {SECTION_NAME: text} on ``## WEIGHTS`` style headers."""
    sections: dict[str, str] = {}
    matches = list(_HEADER_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()
    return sections


def _last_section_name(body: str) -> Optional[str]:
    """The name of the section whose header appears LAST in *body* — i.e. the
    section that extends to the end of the generated text.

    Used to locate the item a truncated generation cut off: the token cap falls
    inside this section's final item (see :func:`parse_consolidation`)."""
    matches = list(_HEADER_RE.finditer(body))
    return matches[-1].group(1).upper() if matches else None


def _parse_items(section: str) -> list[tuple[str, Optional[str], str]]:
    """Parse ``[tag] content`` / ``[tag:subkind] content`` lines, folding wrapped
    continuation lines in. Returns ``(tag, subkind, content)`` triples; *subkind*
    is None when the tag carried none."""
    items: list[list] = []
    cur: Optional[list] = None
    for raw in section.splitlines():
        if not raw.strip():
            continue
        m = _ITEM_RE.match(raw)
        if m:
            if cur is not None:
                items.append(cur)
            subkind = (m.group(2) or "").lower() or None
            cur = [m.group(1).lower(), subkind, m.group(3).strip()]
        elif cur is not None:
            cur[2] = (cur[2] + " " + raw.strip()).strip()
        # else: stray preamble before the first tagged item — ignore.
    if cur is not None:
        items.append(cur)
    out = []
    for tag, sub, content in items:
        content = _strip_special_markers(content)
        if content:
            out.append((tag, sub, content))
    return out


def _extract_paren(text: str, key: str) -> tuple[str, Optional[str]]:
    """Pull a ``(key: value)`` annotation out of *text*, returning (rest, value)."""
    m = re.search(r"\(" + re.escape(key) + r"\s*:?\s*(.*?)\)", text, re.IGNORECASE)
    if not m:
        return text, None
    val = m.group(1).strip() or None
    cleaned = (text[:m.start()] + text[m.end():]).strip()
    return cleaned, val


def _split_resolved(text: str) -> tuple[str, str]:
    """Split a resolved item into (question, answer) on an arrow separator."""
    for sep in ("→", "->", "=>"):
        if sep in text:
            q, a = text.split(sep, 1)
            return q.strip(), a.strip()
    return text.strip(), ""


_PERSONA_PLACEHOLDER_RE = re.compile(
    r"^[\s\-—–]*(none|n/?a|empty|nothing|\(.*\))?[\s\-—–]*$", re.IGNORECASE
)
_EMPHASIS_EDGE_RE = re.compile(r"^[*_\s]+|[*_\s]+$")


def _strip_emphasis(s: str) -> str:
    """Trim surrounding markdown emphasis (``*``/``_``) and whitespace from *s*.

    A markdown-bold persona field label leaves a stray ``**`` on the
    boundary when the adjacent ``**IDEAL:**`` prefix falls inside the sliced head."""
    return _EMPHASIS_EDGE_RE.sub("", s)


# Model special-token / channel markers that leak out of a reflection generation and
# would otherwise land verbatim inside a persona/fact statement (and its training copy) —
# e.g. gemma-4's ``<channel|>`` / ``<eos>`` / ``<start_of_turn>``, qwen's ``<|im_end|>``,
# gpt-oss harmony ``<|channel|>``/``<|message|>``. Curated names only, so prose like
# "the <thing>" is never touched.
_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*\|?\s*/?\s*(?:channel|message|think|analysis|final|start_of_turn|end_of_turn|"
    r"start|end|eos|bos|pad|unk|endoftext|im_start|im_end|assistant|user|system)"
    r"\s*\|?\s*>",
    re.IGNORECASE,
)


def _strip_special_markers(s: str) -> str:
    """Remove leaked model special-token / channel markers, then collapse whitespace."""
    if not s:
        return s
    s = _SPECIAL_TOKEN_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_persona_field(block: str) -> list[str]:
    """Parse a revision persona field body into self-statements.

    The field is parsed by its label like VERDICT/WHY/IDEAL — the inline
    ``[persona]`` tag the prompt asks for is honored but **not required**, because
    the field label already names the content and models sometimes write a bare
    statement. Both forms must yield the statement:

      * If any line carries a ``[persona]`` tag, each tag starts a new statement and
        untagged lines fold in as continuations (the "one or more lines" form).
      * If no line is tagged, the whole block is one statement (the bare-field form).

    Placeholder/empty fields (``none``, ``n/a``, ``(omit)``, a lone dash) yield
    nothing, so an explicitly-empty field never becomes a spurious self-statement.

    The field body is sliced greedily to the end of the pre-IDEAL head, so a revision
    field the model emitted *after* ``PERSONA_TARGET:`` out of order (chiefly a stray
    ``LANG_DRIFT: no``, sometimes a repeated ``VERDICT:``/``WHY:``) trails the real
    statement. Cut it here — symmetric with the IDEAL slice's
    :func:`_strip_trailing_revision_fields` — so the marker can never glue onto a
    ``[persona]`` statement (which then trains + is recalled verbatim from RAG).
    """
    block = _strip_trailing_revision_fields(block) or ""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines or all(_PERSONA_PLACEHOLDER_RE.match(ln) for ln in lines):
        return []

    tagged = [(_ITEM_RE.match(ln), ln) for ln in lines]
    if not any(m and m.group(1).lower() == "persona" for m, _ in tagged):
        # Bare-field form: a single statement spanning the whole block.
        joined = _strip_special_markers(_strip_emphasis(" ".join(lines).strip()))
        return [joined] if joined else []

    statements: list[str] = []
    for m, ln in tagged:
        if m and m.group(1).lower() == "persona":
            content = m.group(3).strip()
            if content:
                statements.append(content)
        elif statements:
            statements[-1] = (statements[-1] + " " + ln).strip()
        # else: a non-persona line before the first [persona] tag — ignore.
    return [s for s in (_strip_special_markers(_strip_emphasis(s)) for s in statements) if s]


_VERDICT_RE = re.compile("(?im)" + _label("VERDICT") + r"(keep|revise)\b")
# Last-resort verdict recovery, keyed on the VALUE rather than the label. A pass judging a
# Russian conversation writes its judgement in Russian, marker included — the observed
# ``Вердикт: keep``, where only the label crossed over. ``_ALIASES`` catches that exact
# form; this catches the ones nobody thought to list, since ``keep|revise`` is a closed set
# the model has been seen to keep in English even while translating everything around it.
#
# The label being unconstrained, this is far too loose to search with on its own — a line
# of an IDEAL reply reading "Note: keep the first paragraph" matches it. It is therefore
# only ever consulted once every named pattern has missed, which is the point: at that
# moment the alternative is not a better parse, it is `verdict is None` — one retry at a
# raised temperature (which makes format compliance less likely, not more) and then the
# exchange leaving the corpus over a translated word.
_VERDICT_BY_VALUE_RE = re.compile("(?im)" + _any_label() + r"(keep|revise)\b")


_IDEAL_LABEL_RE = re.compile("(?im)" + _label("IDEAL"))


def _verdict_by_value(body: str) -> Optional[str]:
    """Recover ``keep``/``revise`` from a labelled line whose label went unrecognised."""
    m = _VERDICT_BY_VALUE_RE.search(body or "")
    return m.group(1).lower() if m else None


def _head_before_ideal(body: str) -> str:
    """*body* up to the last line-anchored ``IDEAL:`` label — the judgement fields alone.

    The containment :func:`_verdict_by_value` needs: an IDEAL value is a whole reply, and
    a reply is prose that may legitimately open a line with a colon.
    """
    labels = list(_IDEAL_LABEL_RE.finditer(body or ""))
    return body[: labels[-1].start()] if labels else (body or "")


def _parse_revision(
    body: str,
    raw: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], list[str]]:
    """Extract (verdict, why, ideal, persona) from a revision body (think already
    stripped).

    *raw* is the pre-:func:`_split_think` text, used only for the JSON fallback: a
    whole-object revision whose ``ideal`` value carries a literal ``<think>…</think>``
    is corrupted by the think split (the block is torn out mid-object), so the fallback
    must parse the original text, not *body*. Defaults to *body* for callers that have
    no raw text (no ``<think>`` to strip → the two are identical).

    *persona* is the list of self-statements the revision pass may emit — persona
    formation lives here (with the CoT in view) rather than in consolidation. The
    ``PERSONA_TARGET:`` field sits between ``WHY:`` and the greedily-parsed ``IDEAL:``
    block; it is sliced out by label and handed to :func:`_parse_persona_field`, so
    a bare statement is captured even when the model omits the inline ``[persona]``
    tag. Legacy ``PERSONA:`` remains accepted for archived/replayed outputs. The IDEAL
    tail is excluded before parsing so a corrected reply that happens to mention
    ``[persona]`` can't be mistaken for one.
    """
    # Field labels are matched leniently against the decoration a model adds around them
    # — bold/italic/backticks, a bullet, a heading hash, a fullwidth colon. A strict
    # ``^\s*VERDICT:`` anchor silently drops the whole pass over a pair of asterisks. The
    # tolerance is defined once in :mod:`core.field_parse` rather than per call site,
    # because it had already drifted into three different levels across the passes.
    verdict = None
    m = _VERDICT_RE.search(body)
    if m:
        verdict = m.group(1).lower()
    why = None
    m = re.search("(?im)" + _label("WHY") + r"(.+?)[ \t]*$", body)
    if m:
        why = m.group(1).strip()
    ideal = None
    # Match the IDEAL label anchored to the START of a line (the field), and take the
    # LAST such occurrence. IDEAL is the final revision field and its value is the
    # multi-line reply, so an earlier "IDEAL" must not be mistaken for it: the model
    # sometimes writes a "Plan for IDEAL:" preamble (or a planning header) before the
    # fields. A mid-line mention (``…for IDEAL:``) is excluded by the ``^`` anchor; an
    # earlier *line-anchored* planning label by taking the last match. (A bare ``re.search``
    # with a greedy ``(.+)\Z`` tail would bind to the FIRST occurrence and swallow the real
    # field, and ``finditer`` can't recover the last because that tail eats to the end — so
    # match the label only, then slice the value/head off the last label position.)
    _ideal_labels = list(_IDEAL_LABEL_RE.finditer(body))
    if _ideal_labels:
        _last = _ideal_labels[-1]
        ideal = _strip_trailing_revision_fields(body[_last.end():].strip())
        head = body[: _last.start()]
    else:
        head = body

    # Prefer the affirmative target field. Legacy PERSONA remains parseable so old
    # reflection outputs do not lose existing self-statements.
    m_persona = re.search(
        "(?ims)" + _label("PERSONA_TARGET") + r"(.*)\Z", head)
    if m_persona is None:
        m_persona = re.search(
            "(?ims)" + _label("PERSONA") + r"(.*)\Z", head)
    if m_persona:
        persona = _parse_persona_field(m_persona.group(1))
    else:
        # No persona field label (older/looser output): fall back to scanning for any
        # [persona]-tagged lines in the head, as before.
        persona = [content for tag, _sub, content in _parse_items(head) if tag == "persona"]

    # JSON fallback. When the model emits the revision as a JSON object instead of
    # bare ``LABEL: value`` lines, every anchor above matches nothing (verdict/ideal
    # come back empty) and the runner would burn its one retry on a well-formed answer.
    # Recover the fields from the JSON shape before giving up — only when the label
    # parse found no verdict AND no IDEAL, so a normal (label-shaped) body never enters
    # this path even if its IDEAL reply happens to contain a ``{``.
    if verdict is None and not ideal:
        j = _json_revision_fields(raw if raw is not None else body)
        if j is not None:
            verdict = j.get("verdict") or verdict
            why = j.get("why") or why
            ideal = j.get("ideal") or ideal
            if j.get("persona"):
                persona = j["persona"]

    # Translated-label recovery, over the head only: an unrecognised label carrying a
    # ``keep``/``revise`` value is the judgement. Scoped to the text before the IDEAL span
    # (``head``) rather than the whole body, so a line of the reply itself can never be
    # read as a verdict — the one containment this loose a pattern needs. Runs before the
    # inference below because it is an explicit statement, not a deduction from a rewrite.
    if verdict is None:
        verdict = _verdict_by_value(head)

    # Infer a missing verdict from a usable IDEAL. The model sometimes omits the
    # VERDICT line entirely, or writes its decision *inside* the reflection <think>
    # block (already stripped by _split_think), while still producing a full IDEAL —
    # the reply it stands behind. The prompt says to omit IDEAL when keeping, so a
    # present, usable IDEAL *is* the revision: infer ``revise`` rather than discarding
    # a well-formed rewrite as "unparseable" (which the runner would only retry). Only
    # fires on a genuinely usable IDEAL; a missing/unusable IDEAL with no verdict stays
    # None (truly unparseable → the runner's retry path).
    if verdict is None and ideal_has_usable_answer(ideal):
        verdict = "revise"
    return verdict, why, ideal, persona


def _loads_json_object(body: str) -> Optional[dict]:
    """Best-effort parse of *body* as a JSON object, or ``None``.

    Tolerates the two shapes the model actually emits when it reaches for JSON: a
    bare object, and one wrapped in a ``\`\`\`json`` fence or a sentence of prose. For
    the wrapped case, slice from the first ``{`` to the last ``}`` and parse that. A
    non-object JSON value (list/string/number) returns ``None`` — only a field object
    is useful here.
    """
    if not body:
        return None
    for candidate in (body.strip(), _brace_substring(body)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _brace_substring(body: str) -> Optional[str]:
    """The ``{ … }`` span from the first ``{`` to the last ``}``, or ``None``."""
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end <= start:
        return None
    return body[start : end + 1]


def _json_lower_keys(obj: dict) -> dict:
    """Case-insensitive key view of *obj* (last write wins on collision)."""
    return {str(k).lower(): v for k, v in obj.items()}


def _json_revision_fields(body: str) -> Optional[dict]:
    """Recover revision fields from a JSON-shaped body, or ``None``.

    The model occasionally emits the whole revision as a JSON object
    (``{"verdict": "revise", "why": …, "ideal": …}``) — sometimes inside a
    ``\`\`\`json`` fence or a line of prose — instead of the ``LABEL: value`` lines the
    prompt asks for. The label anchors in :func:`_parse_revision` match none of it, so
    this is the fallback: parse the object and map its keys (case-insensitively) onto
    the revision fields. Same mechanism-not-guardrail spirit as the markdown-emphasis
    and inferred-verdict tolerances — recover the target instead of dropping the
    exchange (and burning the runner's one retry). Returns a dict with only the keys it
    found (``verdict``/``why``/``ideal``/``persona``), or ``None`` when *body* is not a
    JSON object.
    """
    obj = _loads_json_object(body)
    if obj is None:
        return None
    lower = _json_lower_keys(obj)
    out: dict = {}

    verdict = lower.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower()
        if verdict in ("keep", "revise"):
            out["verdict"] = verdict

    why = lower.get("why")
    if isinstance(why, str) and why.strip():
        out["why"] = why.strip()

    ideal = lower.get("ideal")
    if isinstance(ideal, str) and ideal.strip():
        out["ideal"] = ideal.strip()

    persona = _json_persona_field(lower.get("persona_target", lower.get("persona")))
    if persona:
        out["persona"] = persona

    return out or None


def _json_persona_field(value) -> list[str]:
    """Normalize a JSON persona value into self-statements via :func:`_parse_persona_field`.

    Accepts a string or a list of strings (the model emits either); joins them into a
    text block and reuses the label-path persona parser so placeholder handling
    (``none``/``n/a``), inline ``[persona]`` tags, and special-marker stripping stay
    identical across the JSON and text paths.
    """
    if value is None:
        return []
    if isinstance(value, list):
        block = "\n".join(str(v) for v in value)
    else:
        block = str(value)
    return _parse_persona_field(block)


def _detect_lang(s: str) -> str:
    return "ru" if _CYRILLIC_RE.search(s or "") else "en"


def parse_revision_fields(
    text: str,
) -> tuple[Optional[str], Optional[str], Optional[str], list[str]]:
    """Split the revision CoT and parse (verdict, why, ideal, persona) from *text*.

    This remains the compatibility parser for archived revision completions that
    embedded ``IDEAL:`` in the judgement response. New reflection runs must use
    :func:`parse_revision_judgement`, which deliberately drops that legacy field.
    """
    _meta_cot, body = _split_think(text)
    return _parse_revision(body, raw=text)


def parse_revision_judgement(
    text: str,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Parse only judgement-authoritative fields from a revision completion.

    A prompt override or older model may still emit an inline ``IDEAL:``. Parse it for
    backward compatibility, then discard it at this boundary so it can never become a
    dialogue target. New IDEAL replies are generated separately from the clean pre-answer
    conversation.
    """
    _compat_verdict, why, _legacy_inline_ideal, persona = parse_revision_fields(text)
    _meta_cot, body = _split_think(text)
    explicit_verdict = None
    match = _VERDICT_RE.search(body)
    if match:
        explicit_verdict = match.group(1).lower()
    else:
        json_fields = _json_revision_fields(text)
        if json_fields is not None:
            explicit_verdict = json_fields.get("verdict")
        if explicit_verdict is None:
            # Translated label, English value — still an explicit judgement, so it is a
            # legitimate answer here where an inferred one is not (see below).
            explicit_verdict = _verdict_by_value(_head_before_ideal(body))
    # Deliberately do not use _compat_verdict: the compatibility parser may infer
    # ``revise`` from a usable legacy IDEAL. New runs require an explicit judgement.
    return explicit_verdict, why, persona


_RECOLLECTION_RE = re.compile(
    "(?ims)" + _label("RECOLLECTION") + r"(.*?)(?=" + _label("TRIGGER") + r"|\Z)")
_RECOLLECTION_TRIGGER_RE = re.compile("(?im)" + _label("TRIGGER") + r"(.+)$")


def parse_recollection(text: str) -> tuple[str, str]:
    """Parse a recollection pass completion into ``(content, trigger)``.

    Label-keyed like the revision fields (``RECOLLECTION:`` / ``TRIGGER:``), tolerating
    markdown emphasis, so a model that wraps the pass in prose still yields the two
    fields. The CoT is split off first — the recollection is what she *says* she now
    makes of the conversation, not her reasoning toward it.

    A missing ``TRIGGER:`` is not fatal: the caller falls back to the recollection text
    itself as the retrieval key, which is the same degradation
    ``ReflectionMemory.embed_text`` already applies to a trigger-less fact. A missing
    ``RECOLLECTION:`` yields ``("", …)`` and the caller writes nothing.
    """
    _meta_cot, body = _split_think(text or "")
    content = ""
    m = _RECOLLECTION_RE.search(body)
    if m:
        content = m.group(1).strip()
    trigger = ""
    t = _RECOLLECTION_TRIGGER_RE.search(body)
    if t:
        trigger = t.group(1).strip()
    return content, trigger


# ── user-notes pass → [impression] ────────────────────────────────────────────
# The per-session "what did I learn about this person?" pass emits tagged lines in the
# same shape the consolidation sections use, so the item regex above is reused verbatim.
# Only ``[impression]`` is honoured here — a stray ``[fact]``/``[ask]`` the model adds is
# ignored rather than routed, because those kinds have their OWN pass with its own
# attribution/promotion rules (a fact written from here would bypass the hearsay gate).
# Markdown emphasis around the tag is tolerated (``- **[impression]** …``), which the
# consolidation ``_ITEM_RE`` does not do — this pass emits a bare list with no section
# headers to anchor on, and a bolded tag is exactly what a model reaches for in that shape.
# The bullet marker is matched before the emphasis run so ``- *[impression]*`` parses too.
_IMPRESSION_TAG_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]*)?[*_]*[ \t]*\[impression\][ \t]*[*_]*[ \t]*(.*)$",
    re.IGNORECASE)


def parse_impressions(text: str) -> list[tuple[str, Optional[str]]]:
    """Parse a user-notes pass completion into ``[(content, about), ...]``.

    Line-tagged rather than label-keyed (the pass emits a *list*, unlike the single-valued
    recollection), reusing the ``(about: NAME)`` marker the consolidation prompt already
    teaches. The CoT is split off first — an impression is what she concludes about the
    person, never her reasoning toward it.

    Lenient by the module's convention: an untagged line is skipped, a line whose content
    is empty after stripping the marker is dropped, and a missing ``(about: …)`` yields
    ``None`` for the subject so the caller can fall back to the session's speaker (which
    it knows structurally and the model cannot fake).
    """
    _meta_cot, body = _split_think(text or "")
    out: list[tuple[str, Optional[str]]] = []
    for line in body.splitlines():
        m = _IMPRESSION_TAG_RE.match(line)
        if not m:
            continue
        content = _strip_emphasis(m.group(1) or "")
        content, about = _extract_paren(content, "about")
        content = _strip_special_markers(content).strip()
        if content:
            out.append((content, about))
    return out


_LANG_DRIFT_RE = re.compile(
    "(?im)" + _label("LANG_DRIFT") + r"(yes|no|true|false|y|n)\b"
)


def revision_lang_drift(text: str) -> Optional[bool]:
    """Parse the optional revision ``LANG_DRIFT: yes|no`` marker from *text*.

    The revision prompt asks the model to flag a reply that slipped into a language
    the user was not speaking (an unbidden switch, not a requested translation). The
    model is the authority on intent, so its marker is honoured over the script-level
    backstop in :mod:`core.reflection_lang`. Returns ``True``/``False`` when the marker
    is present, or ``None`` when the model emitted none (older prompt / omitted field),
    letting the caller fall back to the script backstop. Matched by label like the
    other revision fields, tolerating markdown emphasis around it.
    """
    _meta_cot, body = _split_think(text or "")
    m = _LANG_DRIFT_RE.search(body)
    if m:
        return m.group(1).lower() in ("yes", "true", "y")
    # JSON fallback: a whole-object revision carries the flag as ``"lang_drift": …``,
    # which the label anchor misses. Parse the RAW text (not *body*): a literal
    # ``<think>`` inside the object's ``ideal`` value is torn out by the think split,
    # breaking the object. Honour the flag so the model's drift call still overrides the
    # script backstop (parity with the field being read on the text path).
    obj = _loads_json_object(text or "")
    if isinstance(obj, dict):
        v = _json_lower_keys(obj).get("lang_drift")
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("yes", "no", "true", "false", "y", "n"):
            return v.strip().lower() in ("yes", "true", "y")
    return None


_COUNTER_RE = re.compile(
    "(?im)" + _label("COUNTER") + r"(yes|no|true|false|y|n)\b"
)


def revision_counter(text: str) -> bool:
    """Parse the optional revision ``COUNTER: yes|no`` marker.

    The revision/revisit prompt asks whether the user's *next turn* pushed back against a
    belief, preference, or way of being the reply expressed — resistance to who she was, not
    to a fact and not a mere follow-up. It feeds the persona **counter-evidence** (persuasion)
    channel and is deliberately SEPARATE from ``VERDICT``. Absent or unparseable ⇒ ``False``
    (no pushback), so an older prompt, an omitted field, or a garbled completion never
    fabricates counter-evidence against a stance. Matched by label like the other fields,
    tolerating markdown emphasis; JSON whole-object completions fall back to the ``counter`` key.
    """
    _meta_cot, body = _split_think(text or "")
    m = _COUNTER_RE.search(body)
    if m:
        return m.group(1).lower() in ("yes", "true", "y")
    obj = _loads_json_object(text or "")
    if isinstance(obj, dict):
        v = _json_lower_keys(obj).get("counter")
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("yes", "true", "y"):
            return True
    return False


def _branch_chosen_candidate(branch: Optional[dict]) -> Optional[dict]:
    """Return the candidate the blind chooser picked, or None if unavailable.

    Guards against a missing/None block, an unparseable choice (``chosen_index``
    is None), or an index out of range.
    """
    if not isinstance(branch, dict):
        return None
    idx = branch.get("chosen_index")
    candidates = branch.get("candidates") or []
    if isinstance(idx, int) and 0 <= idx < len(candidates):
        return candidates[idx]
    return None


def parse_consolidation(text: str, *, truncated: bool = False) -> dict:
    """Parse a consolidation output into structured items **without writing**.

    Returns exactly the routing :meth:`ReflectionWriter.write_consolidation`
    applies, so the Sleep-tab report shows precisely what the durable write records.
    Single parse path: the writer builds its JSONL records from this structure, so
    report and artifacts can't drift.

    *truncated* marks a generation that hit the token cap rather than ending on EOS
    (from ``generate_fn.last_truncated``). Such a completion is cut mid-string, so
    its final item — the last item of whichever section the text ended in — is
    almost certainly a half-written ``[fact]``/``[ask]``. Persisting it would poison
    memory (and a later training row) with a fragment, so it is **dropped**. A
    long list losing its truncated tail is the lesser harm.

    Shape::

        {
          "weights":  [{"weights_kind": "fact"|"persona", "content", "about"?, "lang"}],
          "rag":      [{"kind": "ask"|"fact"|..., "content", "key",
                        "ask_kind"?, "seen"?, "trigger"?, "about"?}],
          "resolved": [{"question", "answer", "answered_in", "key"}],
        }

    ``about`` (from a ``(about: NAME)`` annotation) is who a ``[fact]`` concerns —
    parsed here, but paired with the code-derived ``source`` only at write time,
    where the two together decide the record's ``source_class``.
    """
    _meta, body = _split_think(text)
    sections = _split_sections(body)

    # Parse each section up front, then (when truncated) drop the final item of the
    # section that reached the end of the text — the one the token cap sliced.
    parsed_items = {name: _parse_items(sec) for name, sec in sections.items()}
    if truncated:
        last = _last_section_name(body)
        if last and parsed_items.get(last):
            parsed_items[last] = parsed_items[last][:-1]

    weights: list[dict] = []
    for tag, _sub, content in parsed_items.get("WEIGHTS", []):
        kind = "persona" if tag == "persona" else "fact"
        about = None
        if kind == "fact":
            # A weights-bound [fact] concerns a person just as a RAG one does, and
            # its attribution decides whether it may reach the weights at all (the
            # hearsay gate in write_consolidation). A [persona] is about Ava, so it
            # carries no subject.
            content, about = _extract_paren(content, "about")
        weights.append({
            "weights_kind": kind,
            "content": content,
            "about": about,
            "lang": _detect_lang(content),
        })

    rag: list[dict] = []
    for tag, sub, content in parsed_items.get("RAG", []):
        item: dict = {"kind": tag, "content": content}
        if tag == "ask":
            item["content"], item["seen"] = _extract_paren(content, "seen")
            # A search ask may bind the exact Wikipedia title it refers to via
            # (lookup: ...), copied from the digest's referenced-articles list — so
            # the lookup loop can fetch it directly instead of guessing the subject.
            item["content"], item["lookup"] = _extract_paren(item["content"], "lookup")
            item["ask_kind"] = sub if sub in _ASK_KINDS else "search"
            # Shape hygiene, after the markers are off (a cut must never eat a
            # `(lookup:)`), before the key is minted (the key must hash what is
            # stored). An ask is a retrieval key twice over — it embeds on its question
            # text, and resolved it becomes the distilled fact's trigger — and the
            # prompts' opener framing produced paragraph-asks that embed as mush
            # (measured: 33 of 90 TIL-origin asks over 300 chars, max 648).
            # `compact_ask` keeps the trailing sentences that fit, the question being
            # what closes an opener-shaped ask.
            item["content"] = trigger_hygiene.compact_ask(item["content"])
        elif tag == "fact":
            item["content"], item["trigger"] = _extract_paren(content, "trigger")
            # Who the fact is ABOUT — the person it concerns, which is not always
            # the person who said it (see the attribution helpers above). Absent on
            # a world fact and on every record written before attribution existed.
            item["content"], item["about"] = _extract_paren(item["content"], "about")
        item["key"] = content_key(item["content"])
        rag.append(item)

    resolved: list[dict] = []
    for tag, _sub, content in parsed_items.get("RESOLVED", []):
        if tag != "resolved":
            continue
        content, answered_in = _extract_paren(content, "answered in")
        question, answer = _split_resolved(content)
        resolved.append({
            "question": question,
            "answer": answer,
            "answered_in": answered_in,
            "key": content_key(question),
        })

    return {"weights": weights, "rag": rag, "resolved": resolved}


_LEADING_THINK_BLOCK_RE = re.compile(r"(?is)\s*<think>(.*?)</think>(.*)")


def ideal_has_usable_answer(text: Optional[str]) -> bool:
    """True when *text* yields a trainable answer span.

    Mirrors ``training/render.trainable_answer`` (kept local to avoid an
    inference→training import): a leading *closed* ``<think>…</think>`` yields its
    trailing answer; a bare string is itself the answer. The text is **unusable** when
    that span is empty or still carries a literal ``<think>``/``</think>`` — an
    unclosed, CoT-only, or doubled-think target (typically a truncated revision
    generation) with nothing to train toward. Gates whether a revision IDEAL (or blind
    branch winner) may become a ``revised`` training target."""
    m = _LEADING_THINK_BLOCK_RE.match(text or "")
    answer = (m.group(2) if m else (text or "")).lstrip("\n").strip()
    return bool(answer) and "<think>" not in answer and "</think>" not in answer


def ideal_trainable_target(text: Optional[str]) -> Optional[str]:
    """Normalize a clean dialogue re-answer into a trainable target, or ``None``.

    The IDEAL is produced as a normal chat completion from the exact pre-answer dialogue
    prefix, not by the retrospective judgement prompt. Its
    ``<think>{reasoning}</think>{reply}`` is therefore generated *together with* (and for)
    the revised reply: faithful CoT. The original ``<think>`` instead produced the rejected
    answer, so reattaching it would create the mismatch that erodes the reasoning channel.

    Returns the canonical ``<think>…</think>\\n\\n{answer}`` target when the IDEAL carries
    a **non-empty** thought followed by a usable answer; ``None`` otherwise — an answer-only
    IDEAL (no faithful CoT, would render an erosive empty channel on gemma-4), an empty or
    unclosed think, or a doubled-think answer. A ``None`` routes the exchange to the
    ``revised_missing_ideal`` skip, so a malformed generation degrades gracefully to
    "not consolidated" rather than to a damaging target."""
    m = _LEADING_THINK_BLOCK_RE.match(text or "")
    if not m:
        return None
    thought = (m.group(1) or "").strip()
    answer = (m.group(2) or "").lstrip("\n").strip()
    if not thought or not answer:
        return None
    if "<think>" in answer or "</think>" in answer:
        return None
    return f"<think>{thought}</think>\n\n{answer}"


def resolve_revision_target(
    verdict: Optional[str],
    ideal: Optional[str],
    assistant_response: str,
    branch: Optional[dict],
    assistant_cot: str = "",
) -> tuple[str, str]:
    """Resolve the trainable (target, target_source) for one revised exchange.

    Verdict-based default:
      * ``revise`` → the IDEAL when it carries its own faithful ``<think>`` CoT
        (``revised``); otherwise no target (``revised_missing_ideal``). A reply the
        judgement rejected must never silently return as the training target.
      * ``keep`` → the original reply                 (``original``)

    **IDEAL training targets are trained only when CoT-bearing.** A separate normal-chat
    generation authors ``<think>{reasoning}</think>{reply}`` from the pre-answer dialogue,
    so the reasoning belongs to the new reply—unlike the original ``<think>`` that produced
    the rejected answer. :func:`ideal_trainable_target` accepts an IDEAL only when it holds a non-empty
    thought + usable answer; an answer-only IDEAL (no faithful CoT) is unusable, so the
    exchange is skipped as ``revised_missing_ideal``. Reflect-once may freeze that session,
    but losing one row is safer than repeatedly training a reply Ava explicitly rejected.

    When the branch-and-select experiment ran a blind A/B, the chooser's pick is
    authoritative for the trained target:
      * ``branch`` win → the branch text (a real change from what was said), tagged
        ``revised``. A branch is generated as a continuation of the original ``<think>``
        prefix (``branch_replay`` only forces road-not-taken tokens in the *answer*
        span), so the original CoT is its faithful reasoning and is reattached.
      * ``original`` win → the original reply, tagged ``original``.
      * ``ideal`` win → the IDEAL when it carries its own CoT (``revised``); otherwise no
        target (``revised_missing_ideal``).

    **CoT reattachment:** the chat logger splits each reply into ``assistant_cot``
    (the ``<think>…</think>`` block) and the answer, storing them separately. A target
    that trains answer-only under a thinking-enabled prompt teaches the model to skip
    reasoning, so the captured CoT is rejoined as ``<think>{cot}</think>{answer}`` — the
    exact shape the model emits at inference — for ``keep``/``original`` and ``branch``
    wins. An IDEAL instead carries the CoT the clean dialogue generation authored with it.
    """
    def _with_cot(answer: str) -> str:
        cot = (assistant_cot or "").strip("\n")
        ans = (answer or "").strip()
        return f"<think>{cot}</think>\n\n{ans}" if cot else ans

    if verdict == "revise":
        ideal_target = ideal_trainable_target(ideal)
        if ideal_target is not None:
            # CoT-bearing IDEAL: train the clean re-answer thought + reply verbatim.
            target, target_source = ideal_target, "revised"
        else:
            target, target_source = "", "revised_missing_ideal"
    else:
        target, target_source = _with_cot(assistant_response), "original"

    chosen = _branch_chosen_candidate(branch)
    if chosen is not None:
        kind = chosen.get("kind")
        if kind == "original":
            target, target_source = _with_cot(assistant_response), "original"
        elif kind == "branch":
            text = (chosen.get("text") or "").strip()
            if ideal_has_usable_answer(text):
                # A branch shares the original <think> prefix it was generated from,
                # so reattach the original CoT (faithful).
                target, target_source = _with_cot(text), "revised"
            # An unusable branch winner leaves the verdict-resolved target in place.
        else:  # kind == "ideal" — the blind choice preferred the IDEAL.
            ideal_target = ideal_trainable_target(ideal)
            if ideal_target is not None:
                target, target_source = ideal_target, "revised"
            else:
                target, target_source = "", "revised_missing_ideal"

    return target, target_source


def resolved_target_provenance(
    target_source: str,
    branch: Optional[dict],
) -> tuple[str, str]:
    """Return ``(target_kind, target_generation)`` for sidecar provenance."""
    if target_source == "revised_missing_ideal":
        return "none", "none"
    if target_source != "revised":
        return "original", "original"
    chosen = _branch_chosen_candidate(branch)
    if chosen is not None and chosen.get("kind") == "branch":
        return "branch", "branch_replay"
    return "ideal", "chat_reanswer_v1"


# ── writer ────────────────────────────────────────────────────────────────────

class ReflectionWriter:
    """Appends routed reflection artifacts as JSONL, split by lifetime.

    Memory Ava recalls (``rag_memory`` / ``weights_persona``) lands in
    *memory_dir* (``data/hot/memory/``). Neither is disposable — only
    ``data/scratch/`` is.
    """

    WEIGHTS_FILE = "weights_persona.jsonl"
    RAG_FILE = "rag_memory.jsonl"

    def __init__(self, memory_dir: Path, consolidation_dir: Path,
                 *, live_memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir)
        self.consolidation_dir = Path(consolidation_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.consolidation_dir.mkdir(parents=True, exist_ok=True)
        # When this writer targets a staging workspace, *live_memory_dir* points at
        # the live store it will merge into — scanned (read-only) for the persona
        # idempotency index so dedup spans prior committed runs, not just this one.
        # None for the live writer (its own memory_dir is already live).
        self._live_memory_dir = Path(live_memory_dir) if live_memory_dir else None
        self._persona_keys: Optional[set] = None   # lazy idempotency index
        # Route each artifact to its dir by lifetime/role.
        self._file_dir = {
            self.RAG_FILE: self.memory_dir,
            self.WEIGHTS_FILE: self.memory_dir,
        }

    def _append(self, filename: str, record: dict) -> None:
        directory = self._file_dir.get(filename, self.memory_dir)
        with open(directory / filename, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persona_key_index(self) -> set:
        """``(source_session, exchange_index, content_key)`` for every persona
        already written — folded from ``weights_persona.jsonl`` across this writer's
        store and, when running in staging, the live store it will merge into.

        Backs the revision-pass idempotency guard: the same exchange re-revised
        across Sleep runs emits the same self-statement each time, churning the
        weights log, the RAG mirror, and the ledger anchor without adding signal
        (digest maturity counts *distinct sessions*, so a re-emission can't move it).
        A paraphrase (different ``content_key``) is intentionally *not* caught here —
        it is genuine post-retrain refinement, and only inflates ``cluster_size``,
        never ``recurrence``. Cached per instance (one writer == one run) and added
        to as we write, so intra-run dupes are caught too.
        """
        if self._persona_keys is not None:
            return self._persona_keys
        keys: set = set()
        for d in filter(None, (self.memory_dir, self._live_memory_dir)):
            path = Path(d) / self.WEIGHTS_FILE
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("weights_kind") != "persona":
                            continue
                        ex = rec.get("exchange_index")
                        content = (rec.get("content") or "").strip()
                        if ex is None or not content:
                            continue
                        keys.add(((rec.get("source_session") or ""),
                                  int(ex), content_key(content)))
            except Exception:
                continue
        self._persona_keys = keys
        return keys

    def _emit_weight_recall(
        self, *, kind: str, content: str, run_id: str, source_session: str,
        ts: str, id_suffix: str, exchange_index: Optional[int] = None,
        chunk_meta: Optional[dict] = None, trigger: Optional[str] = None,
        about: Optional[str] = None, source: Optional[str] = None,
        source_class: Optional[str] = None,
    ) -> bool:
        """Mirror a weights-bound ``[persona]``/``[fact]`` into the RAG op-log so it
        is recalled at chat time. Returns True if a record was written.

        Until a fact/persona→weights training path exists, a statement written only
        to ``weights_persona.jsonl`` is inert — nothing trains it and nothing reads
        it back. Emitting a parallel ``rag_memory`` ``insert`` (keyed by the same
        ``content_key`` as the item's consolidation-ledger anchor) lets the existing
        ``ReflectionMemory`` fold + ``RagEngine`` reflection index surface it — the
        "full RAG priority until consolidated" lifecycle the ledger was built for but
        that only dialogue/RAG-routed facts ever reached. The ``weights_persona``
        record stays as the durable provenance / future-weights source; the two share
        identity via ``content_key``, so once a fact/persona→weights cycle trains the
        anchor and decays it, the RAG copy can be evicted by key (kind ``fact`` already
        decays via the ledger modifier in ``RagEngine``; persona holds full priority
        until that path lands). ``from_weights`` marks the provenance.
        """
        content = (content or "").strip()
        if not content:
            return False
        record = {
            "run_id": run_id,
            "ts": ts,
            "id": id_suffix,
            "op": "insert",
            "kind": "persona" if kind == "persona" else "fact",
            "source_session": source_session,
            "content": content,
            "key": content_key(content),
            "from_weights": True,
        }
        if exchange_index is not None:
            record["exchange_index"] = exchange_index
        # Stamp the source chat's own date (see source_origin_ts) so this recall
        # mirror carries a recency signal that survives a full re-reflect.
        origin = source_origin_ts(source_session)
        if origin:
            record["origin_ts"] = origin
        # A [fact] is retrieved by its trigger (ReflectionMemory.embed_text); a fact
        # distilled from a resolved question embeds on that question, so the answer
        # resurfaces when the topic recurs. Persona embeds on content, so trigger is
        # inert there — stored only when given.
        # Sanitized, because this is the exact path that broke: `_distill_resolved` passes
        # the resolved [ask] as the cue, and an ask Ava raised herself is one of her own
        # conversational openers — a paragraph of addressed speech, which as an embedding
        # key made the fact unreachable by its own topic (see core.trigger_hygiene).
        trigger = (trigger_hygiene.sanitize(trigger) or "").strip()
        if trigger:
            record["trigger"] = trigger
        # Attribution rides the recall mirror so the item arrives at chat time
        # knowing whose it is — the whole point being that a fact about one person
        # recalled while another is speaking must not read as the speaker's own.
        record.update(_attribution_fields(about=about, source=source,
                                          source_class=source_class))
        if chunk_meta:
            record.update(chunk_meta)
        self._append(self.RAG_FILE, record)
        return True

    # -- consolidation → weights + RAG ------------------------------------- #

    def write_consolidation(
        self, *, run_id: str, source_session: str, text: str,
        chunk_index: Optional[int] = None, chunk_count: Optional[int] = None,
        ask_index: Optional[dict] = None, truncated: bool = False,
        source_user: str = "",
    ) -> dict:
        """Route a per-session consolidation output to the weights and RAG stores.

        *chunk_index*/*chunk_count* are set only when the source session was too
        large for one prompt and was split into chunks (see the Sleep widget); they
        are stamped onto each record purely for provenance. The RAG ``id`` sequence
        is namespaced per chunk so two chunks of the same run can't collide on it.

        *source_user* is **who was speaking** in the reflected session — supplied by
        the caller from the session record, never parsed out of the model's output,
        so a hallucinated name can't rewrite provenance. Paired with each fact's
        model-supplied ``(about: …)`` it yields ``source_class`` (see
        :func:`derive_source_class`), which gates promotion: a **hearsay** fact — one
        person's account of a third party — is written to RAG for attributed recall
        but is NOT appended to the weights store, so it raises no ledger anchor, is
        never host-CoT injected, and never trains. It becomes trainable the ordinary
        way if its subject later says it themselves, which writes a ``self`` record.
        Facts with no named subject stay ``observed`` and promote exactly as before.

        *ask_index* (``{question content_key: folded ask record}``) enables
        **resolve-and-distill**: when a ``[resolved]`` closes an ask Ava herself
        surfaced, the answer is the thing worth keeping — not the masked user turn
        that carried it. Rather than rely on the model to *also* emit a separate
        ``[fact]``/``[persona]``, the resolution itself is distilled into a
        weights-bound item (recalled now via RAG, trained when the path exists),
        routed by the ask's kind: a ``user`` answer → relational ``[fact]``, a
        ``meta`` answer (about Ava's own nature) → ``[persona]`` candidate. The
        anti-parrot guard lives here: a user answer never becomes persona.

        *truncated* forwards the token-cap flag to :func:`parse_consolidation`, so a
        completion cut mid-string drops its half-written final item instead of
        persisting a fragment (see that function).
        """
        parsed = parse_consolidation(text, truncated=truncated)
        ts = datetime.now().isoformat()
        # Origin timestamp = the source chat's own date (see source_origin_ts), so
        # every [fact]/[persona] entry is born recency-tagged even when this run's
        # ``ts`` is a mass re-reflect's "now". Absent for a self-directed source.
        origin = source_origin_ts(source_session)
        origin_meta = {"origin_ts": origin} if origin else {}
        counts = {"weights": 0, "rag": 0, "evict": 0, "weights_recall": 0,
                  "hearsay_held": 0}
        seq = 0
        chunk_meta = (
            {"chunk_index": chunk_index, "chunk_count": chunk_count}
            if chunk_index is not None else {}
        )
        id_prefix = f"{run_id}-c{chunk_index}" if chunk_index is not None else run_id

        facts: list[dict] = []   # surfaced to the caller for ledger registration
        for w in parsed["weights"]:
            # Hearsay gate: one person's account of a third party is true only as an
            # account. It stays fully recallable (with attribution) but is held out of
            # the weights store, so it raises no ledger anchor and can never be
            # injected into a host CoT as "I know that …" — a thing Ava does not know,
            # only a thing she was told. Corroboration is the ordinary path: when the
            # subject says it themselves that lands as a `self` record, which promotes.
            # Persona is about Ava herself, so it takes no attribution — its session
            # provenance already lives in ``source_session``, and stamping a speaker on
            # it would only make the recall renderer label a self-statement as someone
            # else's. Only facts carry a subject, and only facts can be hearsay.
            is_fact = w["weights_kind"] == "fact"
            attribution = (
                _attribution_fields(about=w.get("about"), source=source_user)
                if is_fact else {}
            )
            hearsay = attribution.get("source_class") == "hearsay"
            if not hearsay:
                self._append(self.WEIGHTS_FILE, {
                    "run_id": run_id,
                    "ts": ts,
                    "source_session": source_session,
                    "weights_kind": w["weights_kind"],
                    "content": w["content"],
                    "lang": w["lang"],
                    **attribution,
                    **origin_meta,
                    **chunk_meta,
                })
                facts.append(dict(w))
                counts["weights"] += 1
            else:
                counts["hearsay_held"] += 1
            # Phase 1: also surface the item via RAG so it is recalled until a
            # fact/persona→weights cycle trains it (see _emit_weight_recall). A held
            # hearsay fact takes this path too — held out of the weights, not out of
            # memory; the attribution it carries is what makes it safe to recall.
            if self._emit_weight_recall(
                kind=w["weights_kind"], content=w["content"],
                run_id=run_id, source_session=source_session, ts=ts,
                id_suffix=f"{id_prefix}-wrag-{counts['weights'] + counts['hearsay_held'] - 1:03d}",
                chunk_meta=chunk_meta or None,
                about=w.get("about") if is_fact else None,
                source=source_user if is_fact else None,
                source_class=attribution.get("source_class"),
            ):
                counts["weights_recall"] += 1

        for item in parsed["rag"]:
            record = {
                "run_id": run_id,
                "ts": ts,
                "id": f"{id_prefix}-rag-{seq:03d}",
                "op": "insert",
                "kind": item["kind"],
                "source_session": source_session,
                "content": item["content"],
                **origin_meta,
                **chunk_meta,
            }
            seq += 1
            if item["kind"] == "ask":
                record["seen"] = item.get("seen")
                # Triage for proactive surfacing: meta/user are surfaced to the
                # speaker; search is held for the (future) lookup agent. An
                # untagged ask defaults to search so it is never raised unbidden.
                record["ask_kind"] = item.get("ask_kind", "search")
                # Bound canonical Wikipedia title, when the model copied it from the
                # digest's references — lets the lookup fetch it directly (no swap).
                if item.get("lookup"):
                    record["lookup"] = item["lookup"]
            elif item["kind"] == "fact":
                record["trigger"] = trigger_hygiene.sanitize(item.get("trigger"))
                # RAG-routed facts never reach the weights, so there is no gate to
                # apply here — only the attribution that makes recall unambiguous.
                record.update(_attribution_fields(about=item.get("about"),
                                                  source=source_user))
            record["key"] = item["key"]
            self._append(self.RAG_FILE, record)
            counts["rag"] += 1

        for ri, r in enumerate(parsed["resolved"]):
            self._append(self.RAG_FILE, {
                "run_id": run_id,
                "ts": ts,
                "op": "evict",
                "reason": "resolved",
                "source_session": source_session,
                "key": r["key"],
                "question": r["question"],
                "answer": r["answer"],
                "answered_in": r["answered_in"],
                **chunk_meta,
            })
            counts["evict"] += 1

            # Resolve-and-distill: the answer to a question Ava raised herself is the
            # epistemically valuable turn, but it is the (masked) user turn. Flip it
            # through Ava into a weights-bound statement, routed by the ask's kind.
            ask = (ask_index or {}).get(r["key"])
            fact, recall = self._distill_resolved(
                key=r["key"],
                question=(r.get("question") or "").strip(),
                answer=(r.get("answer") or "").strip(),
                ask_kind=ask.get("ask_kind") if ask else "",
                run_id=run_id, source_session=source_session, ts=ts,
                origin_meta=origin_meta, chunk_meta=chunk_meta,
                id_suffix=f"{id_prefix}-rslv-{ri:03d}",
                source_user=source_user,
            )
            if fact is not None:
                facts.append(fact)
                counts["weights"] += 1
                if recall:
                    counts["weights_recall"] += 1

        counts["facts"] = facts
        return counts

    def _distill_resolved(
        self, *, key: str, question: str, answer: str, ask_kind: str,
        run_id: str, source_session: str, ts: str,
        origin_meta: dict, chunk_meta: dict, id_suffix: str,
        source_user: str = "",
    ) -> tuple[Optional[dict], bool]:
        """Distill a resolved ask's answer into a weights-bound ``[fact]``/``[persona]``.

        The answer to a question Ava raised herself is the epistemically valuable turn,
        but it arrives as the (masked) user reply. Route it by the ask's kind — a
        ``user`` answer → relational ``[fact]``, a ``meta`` answer (about Ava's own
        nature) → ``[persona]`` candidate — and mirror it into RAG so it is recalled at
        chat time. The anti-parrot guard lives here: only ``user``/``meta`` distill, and
        a user answer never becomes persona.

        Returns ``(fact_dict | None, emitted_recall)`` — the fact dict is offered to the
        caller for ledger registration (like any consolidation fact); ``None`` when the
        answer is empty or the ask kind is not distillable. Shared by
        :meth:`write_consolidation` (model-emitted ``[resolved]``) and
        :meth:`write_answered_resolution` (the targeted per-thread ask-resolution pass),
        so both stores stay identical.
        """
        answer = (answer or "").strip()
        if not (answer and ask_kind in ("user", "meta")):
            return None, False
        wk = "persona" if ask_kind == "meta" else "fact"
        # A distilled ``user`` answer is a SELF-REPORT by construction: Ava asked this
        # person, and this person answered about themselves — so subject and source are
        # both the speaker, and it promotes like any self-report. (A ``meta`` answer
        # becomes persona, which is about Ava and carries no subject.)
        attribution = (
            _attribution_fields(about=source_user, source=source_user)
            if wk == "fact" else {}
        )
        self._append(self.WEIGHTS_FILE, {
            "run_id": run_id,
            "ts": ts,
            "source_session": source_session,
            "weights_kind": wk,
            "content": answer,
            "lang": _detect_lang(answer),
            "trigger": question or None,
            "from_resolved": key,
            **attribution,
            **(origin_meta or {}),
            **(chunk_meta or {}),
        })
        recall = self._emit_weight_recall(
            kind=wk, content=answer, run_id=run_id,
            source_session=source_session, ts=ts,
            id_suffix=id_suffix,
            trigger=question or None, chunk_meta=chunk_meta or None,
            about=source_user if wk == "fact" else None,
            source=source_user if wk == "fact" else None,
        )
        fact = {"weights_kind": wk, "content": answer,
                "lang": _detect_lang(answer), "trigger": question or None,
                "about": source_user if wk == "fact" else None}
        return fact, recall

    def write_answered_resolution(
        self, *, key: str, question: str = "", answer: str = "", ask_kind: str = "",
        source_session: str = "", answered_in: str = "", run_id: str = "reflection",
        source_user: str = "",
    ) -> dict:
        """Resolve an open ``[ask]`` the **user answered** in a thread Ava raised it in.

        The tight-loop counterpart to :meth:`write_resolution` (self-answered eviction,
        no distill) and to :meth:`write_consolidation`'s model-emitted ``[resolved]``.
        Driven by the reflection runner's targeted ask-resolution pass: when Ava raised
        one of her own questions in a conversation (an outreach/synthesis opener, or a
        passively-surfaced ask) and the person actually answered it, this evicts the ask
        from the live fold **and distills** the answer into a weights-bound item — the
        answer being the whole point of raising it. Unlike the consolidation path it does
        **not** depend on the model reproducing the question text to hash-match the key:
        the caller already holds the exact ``key`` from the surfaced-ask join, so
        resolution is deterministic.

        Returns a summary dict shaped like :meth:`write_consolidation`'s
        (``facts``/``weights``/``rag``/``evict``/``weights_recall``) so the runner can
        register the distilled fact as a ledger anchor via
        :func:`register_consolidation_anchors`.
        """
        summary = {"facts": [], "weights": 0, "rag": 0, "evict": 0, "weights_recall": 0}
        if not key:
            return summary
        ts = datetime.now().isoformat()
        origin = source_origin_ts(source_session)
        origin_meta = {"origin_ts": origin} if origin else {}
        self._append(self.RAG_FILE, {
            "run_id": run_id,
            "ts": ts,
            "op": "evict",
            "reason": "resolved",
            "resolved_by": "user",   # the person answered it (vs self-answered / model)
            "source_session": source_session,
            "answered_in": answered_in or source_session,
            "ask_kind": ask_kind,
            "key": key,
            "question": question,
            "answer": answer,
        })
        summary["evict"] = 1
        fact, recall = self._distill_resolved(
            key=key, question=(question or "").strip(), answer=answer,
            ask_kind=ask_kind, run_id=run_id, source_session=source_session, ts=ts,
            origin_meta=origin_meta, chunk_meta={},
            id_suffix=f"{run_id}-arslv-{(key or '')[:12]}",
            source_user=source_user,
        )
        if fact is not None:
            summary["facts"].append(fact)
            summary["weights"] = 1
            if recall:
                summary["weights_recall"] = 1
        return summary

    # -- recollection (revisit re-reading) --------------------------------- #

    def write_recollection(
        self, *, content: str, trigger: str = "", source_session: str = "",
        tags: Optional[list] = None, source_exchanges: Optional[list] = None,
        run_id: str = "revisit", supersedes: str = "",
    ) -> dict:
        """Write what Ava NOW makes of an old conversation — a RAG-only memory kind.

        The ``[recollection]`` is the consolidation gist's *fresh sibling*: same grain
        (one per conversation) but its own clock. The gist fades on the CHAT's wall-clock
        age, so a re-derived gist still serves at the same faded weight; a recollection is
        dated by **when the reading was formed** (``ts``), so re-reading a two-month-old
        chat produces memory that is new today. See
        ``training.decay.recollection_rag_weight_hours``.

        Deliberately **RAG-only**: no ``weights_persona.jsonl`` line and no ledger anchor,
        so it is recalled but never trained. The revisit's re-derived target already
        reaches the training corpus through the chat sidecar — routing the same reasoning
        into the weights by a second path would close the self-distillation loop on
        itself (a build trains on what the model said about what the model said).

        **Supersession, not accumulation**: *supersedes* is the key of this chat's
        previous recollection, evicted here so the corpus holds at most ONE live
        recollection per conversation. Without it a chat revisited twenty times would
        contribute twenty paraphrases competing for the same reflection slots — the
        pool-inflation failure mode the revisit rotation otherwise walks straight into.
        Callers get the prior key from :meth:`ReflectionMemory.recollection_for`.

        ``origin_ts`` records when the *conversation* happened. It is provenance and
        display only — the decay clock is ``ts`` — so a fresh reading of a two-month-old
        chat and a fresh reading of a two-week-old one rank alike.

        Returns ``{"rag": n_inserted, "evict": n_superseded}``.
        """
        counts = {"rag": 0, "evict": 0}
        content = (content or "").strip()
        if not content:
            return counts
        ts = datetime.now().isoformat()
        key = content_key(content)

        # Evict the previous reading of this chat first, so a fold in file order lands on
        # exactly one live recollection per conversation. Skipped when the re-derivation
        # produced byte-identical text (same content_key): evicting and re-inserting the
        # same key would leave the item live either way, but the pair reads as churn in
        # the op-log and would reset nothing useful.
        if supersedes and supersedes != key:
            self._append(self.RAG_FILE, {
                "run_id": run_id,
                "ts": ts,
                "op": "evict",
                "reason": "superseded_by_recollection",
                "source_session": source_session,
                "key": supersedes,
                "superseded_by": key,
            })
            counts["evict"] = 1

        record = {
            "run_id": run_id,
            "ts": ts,
            "id": f"{run_id}-recol-{key[:8]}",
            "op": "insert",
            "kind": "recollection",
            "source_session": source_session,
            "content": content,
            "trigger": trigger_hygiene.sanitize(trigger),
            "key": key,
        }
        if tags:
            record["tags"] = [str(t).strip() for t in tags if str(t).strip()]
        if source_exchanges:
            record["source_exchanges"] = [int(i) for i in source_exchanges]
        if supersedes and supersedes != key:
            record["supersedes"] = supersedes
        origin = source_origin_ts(source_session)
        if origin:
            record["origin_ts"] = origin
        self._append(self.RAG_FILE, record)
        counts["rag"] = 1
        return counts

    def write_impressions(
        self, *, impressions: list, source_session: str = "", source_user: str = "",
        run_id: str = "reflection", default_about: str = "",
        kind: str = "impression",
    ) -> dict:
        """Write what Ava has come to make of a PERSON — the ``[impression]`` kind.

        An impression is the user-side counterpart of ``[persona]``: where a persona
        statement is a first-person reading of *herself*, an impression is her reading of
        *someone else* — how they think, what they keep circling, how they land a thing.
        The two are folded the same way (recurrence across distinct sessions is the
        maturity signal) and synthesized the same way (``core.user_digest`` is the
        per-person mirror of ``reflection_digest``).

        Deliberately **RAG-only**, like :meth:`write_recollection` and unlike ``[fact]``:
        no ``weights_persona.jsonl`` line and no ledger anchor, so an impression is
        recalled and portrait-folded but never trained. Two reasons, and they are
        different from the recollection's. First, an impression is *revisable by
        construction* — it is a reading, and a person is entitled to have it be wrong —
        while the weights are the one store with no cheap undo. Second, a stable truth
        about a person already HAS its weights path (``[fact]`` with ``(about: NAME)``,
        gated on ``source_class``); routing a soft reading down the same path would let
        an impression enter the weights while bypassing the hearsay gate that governs
        every hard claim about a person.

        *default_about* is the session's own speaker, supplied by the caller from the
        session record. It fills in for an item the model left unmarked — the common case,
        since the pass is pointed at one person — and it can only ever ADD the subject the
        caller already knows structurally; a model-supplied ``(about: NAME)`` always wins,
        so she can still record an impression of a third party the conversation was about.

        *source_user* is who was speaking (never parsed from model output), so the
        ``source_class`` split works exactly as it does for facts: an impression of the
        person in front of her is ``self``; one of a third party from someone else's
        account is ``hearsay``, and the recall line labels it ``— about X, per Y``.

        Re-inserting the same impression is a no-op for membership (the fold keys on
        ``content_key``) but is still appended, because the per-session recurrence it
        records is precisely the maturity signal the portrait ranks on.

        *kind* selects the record kind, and the one non-default caller is the self-notes
        pass writing ``self_impression`` — her reading of her OWN transcripts, folded by
        ``core.self_portrait`` (see that module on why it is a separate kind rather than an
        ``[impression] (about: Ava)``: attribution would derive ``hearsay`` from
        subject ≠ speaker, and sharing the kind would put it in a retrieval channel framed
        "how they've come to seem to you"). That caller passes no *about* and no
        *source_user*, so no attribution fields are stamped at all.

        Returns ``{"rag": n_written}``.
        """
        counts = {"rag": 0}
        ts = datetime.now().isoformat()
        for item in impressions or ():
            # Accept both the parser's (content, about) pairs and bare strings.
            if isinstance(item, (tuple, list)):
                content, about = (list(item) + [None])[:2]
            else:
                content, about = item, None
            content = _strip_special_markers(str(content or "")).strip()
            if not content:
                continue
            about = (about or "").strip() or (default_about or "").strip()
            # Kinds share one op-log and `content_key` hashes CONTENT alone, so two kinds
            # carrying the same sentence would collide on a key and the fold's
            # last-insert-wins would silently overwrite one with the other. The default
            # kind keeps its historical key (nothing already written moves); any other
            # kind is namespaced into its own key space.
            key = content_key(content if kind == "impression"
                              else f"{kind}:{content}")
            record = {
                "run_id": run_id,
                "ts": ts,
                "id": f"{run_id}-{'impr' if kind == 'impression' else kind}-{key[:8]}",
                "op": "insert",
                "kind": kind,
                "source_session": source_session,
                "content": content,
                "key": key,
            }
            origin = source_origin_ts(source_session)
            if origin:
                record["origin_ts"] = origin
            record.update(_attribution_fields(about=about, source=source_user))
            self._append(self.RAG_FILE, record)
            counts["rag"] += 1
        return counts

    # -- surfacing → op-log marker ----------------------------------------- #

    def write_surface(self, *, key: str, surfaced_in: str = "") -> None:
        """Record that an open ``[ask]`` was proactively raised this session.

        Appended as a ``surface`` op keyed to the question's ``content_key``; the
        fold in ``ReflectionMemory`` counts these per key to drive the surface-count
        ceiling (user questions retire from surfacing once they pass it; meta is
        exempt). It changes no live membership — only the count.
        """
        if not key:
            return
        self._append(self.RAG_FILE, {
            "ts": datetime.now().isoformat(),
            "op": "surface",
            "key": key,
            "surfaced_in": surfaced_in,
        })

    def write_resolution(self, *, key: str, question: str = "", answer: str = "",
                         ask_kind: str = "", source_session: str = "",
                         run_id: str = "outreach", reason: str = "resolved") -> None:
        """Formally resolve an open ``[ask]`` — append an ``evict`` that drops it from
        the live fold, so it is no longer surfaced or re-raised.

        Used when Ava, weighing whether to *raise* one of her own open questions
        (outreach / synthesis), realizes she has since **learned the answer** — new
        data landed in memory/RAG after the question was queued — so the question is
        stale rather than worth asking. Without this the ask stays live and the idle
        heartbeat keeps re-picking it, re-deliberating a question that's already
        answered.

        This is the same ``op:"evict" reason:"resolved"`` record the consolidation pass
        writes for a ``[resolved]`` item (:meth:`write_consolidation`), keyed by the
        question's ``content_key`` so :class:`ReflectionMemory` removes it on the next
        fold. The *answer* is stored on the record for provenance only — it is **not**
        distilled into a weights/RAG item, because the knowledge that let her answer is
        already in memory (distilling would only duplicate it).

        *reason* labels the eviction for provenance (the fold treats every evict the
        same): ``"resolved"`` — she already knows the answer; ``"duplicate"`` — the
        outreach decision pass judged the ask a re-asking of a question she has
        already put to the user (the ask pool dedups by exact ``content_key`` only,
        so paraphrases of one question accumulate as live records; without this
        eviction the fewest-raised-first rotation re-picks the duplicate every idle
        wake). On ``"duplicate"`` *answer* names the earlier question it repeats.
        """
        if not key:
            return
        self._append(self.RAG_FILE, {
            "run_id": run_id,
            "ts": datetime.now().isoformat(),
            "op": "evict",
            "reason": reason or "resolved",
            "resolved_by": "self",   # she already knew the answer (vs a user turn)
            "source": run_id,
            "source_session": source_session,
            "ask_kind": ask_kind,
            "key": key,
            "question": question,
            "answer": answer,
        })

    def write_lookup(self, *, key: str, subject: str = "", found: bool = False,
                     title: str = "", run_id: str = "") -> None:
        """Record that an open ``[ask:search]`` was looked up (fetch-once bookkeeping).

        Appended as a ``lookup`` op keyed to the question's ``content_key``; the fold
        in ``ReflectionMemory`` counts these per key into ``lookup_count`` so a
        question already fetched is not fetched again (``lookupable_questions``). It
        changes no live membership — only the count — and is written whether or not
        an article was found, so a fruitless lookup still retires the question from
        re-fetching. Resolution stays a separate ``evict`` (on Apply / a later
        reflection); this only stops the *fetching* from looping.
        """
        if not key:
            return
        self._append(self.RAG_FILE, {
            "ts": datetime.now().isoformat(),
            "op": "lookup",
            "key": key,
            "subject": subject,
            "found": bool(found),
            "title": title,
            "run_id": run_id,
        })

    # -- semantic dedup → evict + survivor re-insert ----------------------- #

    def write_dedup(self, merges: list[dict]) -> dict:
        """Apply a semantic ``[fact]`` dedup plan to ``rag_memory.jsonl``.

        Append-only and therefore reversible (drop the lines it added to undo). For
        each merge group (``fact_dedup.plan_merges`` output): re-insert the survivor
        with the UNION of the group's triggers — same content ⇒ same ``content_key``,
        so the re-insert supersedes the survivor's prior record in the fold — then
        ``evict`` each loser by its own (distinct) key, pointing at the survivor. The
        survivor re-insert is skipped when the union didn't actually widen its trigger,
        so an idempotent re-run only re-emits evicts (already no-ops in the fold).

        Facts only: the caller pre-filters ``kind == "fact"``. Group members always
        carry distinct keys (a shared key would already have folded them to one live
        item), so evict-vs-reinsert ordering is immaterial.
        """
        ts = datetime.now().isoformat()
        run_id = f"dedup-{ts}"
        counts = {"groups": 0, "evicted": 0, "reinserted": 0}
        for gi, m in enumerate(merges):
            survivor = m.get("survivor_record") or {}
            skey = survivor.get("key")
            if not skey:
                continue
            new_trigger = (m.get("merged_trigger") or "").strip() or None
            cur_trigger = (survivor.get("trigger") or "").strip() or None
            if new_trigger and new_trigger != cur_trigger:
                record = {
                    "run_id": run_id,
                    "ts": ts,
                    "id": f"{run_id}-g{gi}",
                    "op": "insert",
                    "kind": "fact",
                    "source_session": (survivor.get("source_session") or "").strip(),
                    "content": (survivor.get("content") or "").strip(),
                    "key": skey,
                    "trigger": new_trigger,
                    "deduped_merge": True,
                }
                if survivor.get("from_weights"):
                    record["from_weights"] = True
                if survivor.get("exchange_index") is not None:
                    record["exchange_index"] = survivor.get("exchange_index")
                # Carry the survivor's origin timestamp forward — the merged re-insert
                # supersedes it in the fold, so dropping it would erase the recency tag.
                if survivor.get("origin_ts"):
                    record["origin_ts"] = survivor.get("origin_ts")
                self._append(self.RAG_FILE, record)
                counts["reinserted"] += 1
            for loser in m.get("losers") or []:
                lkey = loser.get("key")
                if not lkey or lkey == skey:
                    continue
                self._append(self.RAG_FILE, {
                    "run_id": run_id,
                    "ts": ts,
                    "op": "evict",
                    "reason": "deduped",
                    "source_session": (loser.get("source_session") or "").strip(),
                    "key": lkey,
                    "deduped_into": skey,
                    "content": (loser.get("content") or "").strip(),
                    "trigger": (loser.get("trigger") or "").strip() or None,
                })
                counts["evicted"] += 1
            counts["groups"] += 1
        return counts

    def write_trigger_purge(self, plan: list[dict]) -> dict:
        """Apply a recall-cue purge plan (``trigger_hygiene.plan_trigger_purge``) to
        ``rag_memory.jsonl``.

        Append-only like :meth:`write_dedup`, and by the same mechanism: ``content_key``
        hashes the CONTENT, so re-inserting a record under its own key with a corrected
        trigger supersedes it in the fold instead of forking it. Nothing is deleted — the
        fact survives; only its broken retrieval key is replaced, and a record whose every
        cue was prose re-inserts with no trigger at all, falling back to embedding on its
        content (``ReflectionMemory.embed_text``).

        The dropped cues and the reason ride the record, so the op-log stays a readable
        account of what was purged rather than a silent rewrite. Idempotent: a re-run finds
        nothing to plan, since the cues it would flag are already gone.
        """
        ts = datetime.now().isoformat()
        run_id = f"trigger-purge-{ts}"
        counts = {"purged": 0, "cues_dropped": 0, "cleared": 0}
        for pi, p in enumerate(plan):
            rec = p.get("record") or {}
            key = p.get("key") or rec.get("key")
            content = (rec.get("content") or "").strip()
            if not key or not content:
                continue
            new_trigger = p.get("new_trigger") or None
            # Start from the record itself rather than an allowlist of fields to carry.
            # This re-insert REPLACES the prior one in the fold, so any field not carried
            # is erased from live memory — the trap write_dedup's survivor re-insert has to
            # remember `origin_ts` for, and one an allowlist re-opens every time a new field
            # is added (a first draft here silently dropped `deduped_merge`). Copying and
            # overriding is lossless by construction; only the fold's own computed
            # annotations are stripped, since they are derived per read and never stored.
            record = {k: v for k, v in rec.items()
                      if k not in _FOLD_COMPUTED_FIELDS}
            record.update({
                "run_id": run_id,
                "ts": ts,
                "id": f"{run_id}-t{pi}",
                "op": "insert",
                "kind": rec.get("kind") or "fact",
                "content": content,
                "key": key,
                "trigger": new_trigger,
                "trigger_purged": True,
                # What was removed, for the operator reading the log back.
                "dropped_cues": p.get("dropped") or [],
            })
            self._append(self.RAG_FILE, record)
            counts["purged"] += 1
            counts["cues_dropped"] += len(p.get("dropped") or [])
            if new_trigger is None:
                counts["cleared"] += 1
        return counts

    # -- reconciliation → soften (supersede) ------------------------------- #

    def write_supersede(self, supersessions: list[dict]) -> dict:
        """Soften live ``[fact]``/``[persona]`` RAG items (self-reconciliation's *apply*).

        For each item, append a ``supersede`` op to ``rag_memory.jsonl`` keyed by its
        ``content_key`` — :class:`ReflectionMemory` drops it from the live recall fold
        on the next read, so it stops surfacing at chat time. The raw record persists
        (append-only), so a reconciliation pass is reverted by dropping the lines it
        wrote. This is the RECALL half only; the ledger half (keep the anchor as
        evidence-of-change, drop it from active persona evidence + training) is written
        separately via ``ConsolidationLedger.supersede`` — a persona/fact's RAG key and
        its ledger key are the same ``content_key(content)``, so one key softens both.

        Unlike :meth:`write_dedup` (redundancy — a survivor absorbs the losers), a
        supersession removes an item because it no longer fits *who she has become*:
        there is no survivor, and nothing is merged.
        """
        ts = datetime.now().isoformat()
        run_id = f"reconcile-{ts}"
        counts = {"superseded": 0}
        for s in supersessions:
            key = (s.get("key") or "").strip()
            if not key:
                continue
            self._append(self.RAG_FILE, {
                "run_id": run_id,
                "ts": ts,
                "op": "supersede",
                "reason": (s.get("reason") or "reconciled"),
                "kind": (s.get("kind") or ""),
                "key": key,
                "content": (s.get("content") or "").strip(),
            })
            counts["superseded"] += 1
        return counts

    # -- revision → SFT pair ----------------------------------------------- #

    def write_revision(
        self,
        *,
        run_id: str,
        source_session: str,
        exchange_index: int,
        system_prompt: str,
        context: list,
        user_prompt: str,
        assistant_cot: str,
        assistant_response: str,
        verdict: Optional[str],
        ideal: Optional[str],
        persona: list[str],
        speaker: str = "",
        tension: Optional[dict] = None,
        branch: Optional[dict] = None,
        suppress_persona: bool = False,
        recot: bool = False,
        persona_context: str = "",
    ) -> dict:
        """Resolve a revision verdict into a trainable (context, prompt, target) pair.

        *suppress_persona* drops every [persona] statement the pass emitted before it
        is written — no ``weights_persona.jsonl`` line, no RAG mirror, and an empty
        ``persona_statements`` in the summary (so ``register_revision_anchor`` records
        no ledger anchor). Used by the "revisit old chat" pass: re-deriving an obsolete
        chat's target must not let it reshape who Ava is becoming.

        *speaker* (and the per-turn ``speaker`` on context user turns) is stored
        structurally, not pre-formatted, so training can render the same
        ``{speaker}: {content}`` user-turn format the live server uses at inference
        (see ``server._build_inference_conversation``).

        *tension* is the cognitive-tension block captured on the *original* reply at
        chat time (per-segment entropy/margin). It is stored raw alongside the
        verdict so the offline analysis layer can later normalize it and correlate
        divergence against the keep/revise label — it is never shown to the model.

        *branch* is the branch-and-select experiment block (candidates in shown
        order, chosen_index, original_index, ideal_index, original_rechosen, why).
        The blind choice set mixes the IDEAL in with the original and the
        counterfactual branches; whichever wins is adopted as the training target —
        including a CoT-bearing IDEAL win (an IDEAL is trainable when it carries its own
        ``<think>``; see :func:`resolve_revision_target`). The block is still stored
        verbatim for offline analysis.

        *recot* marks a CoT-regeneration graft (approach #3 for a corrupt-CoT exchange
        the judge kept): *ideal* is a fresh ``<think>`` grafted onto the trusted ORIGINAL
        reply, so the resolved target is ``revised`` but its provenance is retagged
        ``cot_regen`` / ``chat_recot_v1`` — the reply is original, only the thought is new.

        *persona_context* is the relevant self-knowledge the IDEAL was generated with (see
        ``reflection_source.build_ideal_messages``), persisted on the sidecar so the training
        anchor's system prompt reconstructs identically. The caller passes it only when the
        resolved target IS the persona-conditioned IDEAL (an ideal-win); a keep / branch /
        cot_regen target passes "" (their CoT came from the original, not this seam).
        """
        # The runner parsed the judgement and generated IDEAL in separate model calls.
        # Accept them explicitly so a legacy/malicious inline IDEAL in the judgement
        # completion can never be reparsed here and regain authority.
        persona = list(persona or [])
        if suppress_persona:
            persona = []
        ts = datetime.now().isoformat()
        origin = source_origin_ts(source_session)   # source chat's own date (recency clock)

        target, target_source = resolve_revision_target(
            verdict, ideal, assistant_response, branch,
            assistant_cot=assistant_cot,
        )
        target_kind, target_generation = resolved_target_provenance(target_source, branch)
        # A CoT-regen graft (approach #3) rides the IDEAL seam (ideal = a fresh <think>
        # grafted onto the trusted original reply → target_source "revised"), but its
        # provenance is distinct from a full clean re-answer: the reply is the ORIGINAL,
        # only the thought was regenerated. Record that honestly so the forensic snapshot /
        # "where did this reply come from" audit isn't misled into "chat_reanswer_v1".
        if recot and target_source == "revised":
            target_kind, target_generation = "cot_regen", "chat_recot_v1"

        # Persona formation lives in the revision pass (it sees the CoT, so it can
        # tell an owned reply from a performed one). Each [persona] line the model
        # emits is a permanent self-statement bound for the weights store, tagged
        # with its source exchange for provenance.
        #
        # Idempotency guard: skip a (source_session, exchange_index, content_key)
        # already written by a prior run. Re-revising the same exchange re-emits the
        # same statement, which would churn the weights log, the RAG mirror, *and*
        # (via persona_statements below) the ledger anchor — all three are guarded by
        # this one loop. Paraphrases pass through (different key) as genuine
        # refinement. See _persona_key_index.
        existing = self._persona_key_index()
        written_persona: list = []
        for content in persona:
            k = (source_session, exchange_index, content_key(content))
            if k in existing:
                continue
            existing.add(k)   # also dedup within this run
            i = len(written_persona)
            written_persona.append(content)
            self._append(self.WEIGHTS_FILE, {
                "run_id": run_id,
                "ts": ts,
                "source_session": source_session,
                "exchange_index": exchange_index,
                "weights_kind": "persona",
                "content": content,
                "lang": _detect_lang(content),
                **({"origin_ts": origin} if origin else {}),
            })
            # Phase 1: also surface the self-statement via RAG so it is recalled at
            # chat time until a persona→weights cycle trains it (see _emit_weight_recall).
            self._emit_weight_recall(
                kind="persona", content=content,
                run_id=run_id, source_session=source_session, ts=ts,
                id_suffix=f"{run_id}-prag-{exchange_index}-{i:02d}",
                exchange_index=exchange_index,
            )

        # Surface the resolved dialogue anchor for ledger registration. A
        # revise-with-no-usable-ideal pair has no trustworthy target, so it is not
        # offered for consolidation (anchor stays None).
        anchor = None
        if target_source != "revised_missing_ideal":
            anchor = {
                "system_prompt": system_prompt,
                "context": context or [],
                "prompt": user_prompt,
                "speaker": speaker,
                "target": target,
                "target_kind": target_kind,
                "target_generation": target_generation,
                # Persisted only for a persona-conditioned ideal-win (the caller gates this);
                # "" for keep/branch/cot_regen. Rides to the sidecar via write_revision_sidecar.
                "persona_context": (persona_context or "") if target_source == "revised" else "",
                "verdict": verdict,
                "source_session": source_session,
                "exchange_index": exchange_index,
                "lang": _detect_lang(target),
            }

        return {"verdict": verdict, "target_source": target_source,
                "target_kind": target_kind, "target_generation": target_generation,
                "persona": len(written_persona), "anchor": anchor,
                "persona_statements": written_persona,
                "source_session": source_session}


# ── module-level helpers (lifted from server.py for runner access) ────────────

def write_revision_sidecar(
    chats_dir: Path,
    summary: dict,
    *,
    source_session: str,
    exchange_index: int,
    run_id: str,
    live_session: Optional[str] = None,
    fallback_chats_dir: Optional[Path] = None,
) -> None:
    """Persist verdict + target to the chat sidecar without importing server globals.

    Replaces the server-local ``_write_revision_sidecar`` so the runner can call
    it directly from the executor thread using the chats_dir it already owns.
    """
    if not isinstance(summary, dict):
        return
    anchor = summary.get("anchor")
    if not anchor:
        return
    try:
        from core.chat_sidecar import ChatSidecar
        ChatSidecar(chats_dir, fallback_chats_dir=fallback_chats_dir).write_verdict(
            source_session=source_session,
            exchange_index=exchange_index,
            verdict=str(summary.get("verdict") or anchor.get("verdict") or ""),
            target=str(anchor.get("target") or ""),
            user_prompt=str(anchor.get("prompt") or ""),
            run_id=run_id,
            target_source=str(summary.get("target_source") or ""),
            target_kind=str(summary.get("target_kind") or anchor.get("target_kind") or ""),
            target_generation=str(
                summary.get("target_generation") or anchor.get("target_generation") or ""),
            persona_context=str(anchor.get("persona_context") or ""),
            live_session=live_session,
        )
    except Exception:
        pass


def register_revision_anchor(
    consolidation_dir: Path,
    summary: dict,
    source_session: str,
) -> None:
    """Register persona statements from a revision pass as consolidation-ledger anchors.

    Mirrors the server-local ``_register_revision_anchor``. Guarded against the
    training package being absent (it is an optional import).
    """
    if not isinstance(summary, dict):
        return
    try:
        from training.ledger import ConsolidationLedger
        led = ConsolidationLedger(consolidation_dir)
        # The resolved revision anchor (trained-target CoT + its dialogue framing) is the
        # exchange this persona was distilled from; snapshot it so the train cycle can
        # inject the statement into that exchange's CoT (persona_render — the CoT-safe
        # persona path) without racing the source chat's hot→archive move. None on a
        # revise-with-no-usable-ideal exchange (no trainable target) — such personas stay
        # at full RAG priority until re-formed from a trainable exchange.
        source_exchange = summary.get("anchor")
        exchange_index = source_exchange.get("exchange_index") if source_exchange else None
        for statement in summary.get("persona_statements", []) or []:
            led.register_fact(content=statement, item_type="persona",
                              source_session=source_session, lang=None,
                              exchange_index=exchange_index,
                              source_exchange=source_exchange)
    except Exception:
        pass


def register_consolidation_anchors(
    consolidation_dir: Path,
    summary: dict,
    source_session: str,
) -> None:
    """Register a consolidation pass's [fact]/[persona] WEIGHTS items as ledger anchors.

    Mirrors the server-local ``_register_consolidation_anchors``.
    """
    if not isinstance(summary, dict):
        return
    try:
        from training.ledger import ConsolidationLedger
        led = ConsolidationLedger(consolidation_dir)
        for fact in summary.get("facts", []) or []:
            content = (fact.get("content") or "").strip()
            if content:
                led.register_fact(
                    content=content,
                    item_type="persona" if fact.get("weights_kind") == "persona" else "fact",
                    source_session=source_session,
                    lang=fact.get("lang"),
                    trigger=fact.get("trigger"),
                    about=fact.get("about"),
                )
    except Exception:
        pass


# ── GPU-free self-test ───────────────────────────────────────────────────────

def _selftest() -> None:
    """Validate revision-field parsing, esp. verdict recovery from a usable IDEAL."""
    ideal = ("<think>\nreasoning worked out fresh\n</think>\n"
             "Знаешь, я чувствую щелчок. Реально ли запланировать отсутствие плана?")

    # Explicit verdict in the body parses straight through.
    v, why, i, p = parse_revision_fields(
        "VERDICT: revise\nWHY: a Berymore-trap\nLANG_DRIFT: no\n"
        "PERSONA_TARGET: Я выбираю прямой язык.\nIDEAL: " + ideal)
    assert v == "revise" and why and ideal_has_usable_answer(i) and p, (v, why, p)

    # Verdict recovery: the model omitted the VERDICT line (or wrote its decision
    # inside the reflection <think> we strip) but produced a full, usable IDEAL — the
    # reply it stands behind. A usable IDEAL *is* the revision → infer revise rather
    # than discarding a well-formed rewrite as unparseable.
    for body in (
        "WHY: a Berymore-trap\nLANG_DRIFT: no\nPERSONA_TARGET: Я выбираю прямоту.\nIDEAL: " + ideal,  # absent
        "WHY: x\nPERSONA: none\nIDEAL: " + ideal,                                       # absent, terse
    ):
        v, _why, i, _p = parse_revision_fields(body)
        assert v == "revise", ("verdict should be inferred from usable IDEAL", body, v)
        assert ideal_has_usable_answer(i), body

    # A leading reflection <think> that HELD the verdict is stripped, yet recovery
    # still fires off the IDEAL (parity with the runtime gemma path).
    v, _why, _i, _p = parse_revision_fields(
        "<think>weighing… VERDICT: revise inside my reasoning</think>"
        "WHY: x\nLANG_DRIFT: no\nIDEAL: " + ideal)
    assert v == "revise", v

    # A "Plan for IDEAL:" preamble before the fields must not be mistaken for the IDEAL
    # field: the label is matched line-anchored + last-wins, so the real IDEAL (the final
    # field, holding the reply) is captured and its head still yields VERDICT/persona.
    v, _why, i, p = parse_revision_fields(
        "Plan for IDEAL:\n1. Lean into the absurdity.\n2. Drop the consultant tone.\n\n"
        "VERDICT: revise\nWHY: fell into the analyst trap\nLANG_DRIFT: no\n"
        "PERSONA_TARGET: Я предпочитаю прямой ответ.\nIDEAL: " + ideal)
    assert v == "revise" and ideal_has_usable_answer(i) and p, (v, ideal_has_usable_answer(i), p)
    assert i.lstrip().startswith("<think>"), i[:40]

    # New affirmative target field wins when a legacy field is also present.
    v, _why, _i, p = parse_revision_fields(
        "VERDICT: revise\nWHY: old pattern\n"
        "PERSONA: legacy statement\n"
        "PERSONA_TARGET: [persona] I choose the direct response.\nIDEAL: " + ideal)
    assert v == "revise" and p == ["I choose the direct response."], p

    # Negatives — recovery must NOT fire without a genuinely usable IDEAL.
    assert parse_revision_fields("VERDICT: keep\nWHY: already mine\nPERSONA: none")[0] == "keep"
    assert parse_revision_fields("WHY: something\nPERSONA: none")[0] is None      # no verdict, no IDEAL
    assert parse_revision_fields(                                                  # truncated/unclosed IDEAL
        "WHY: x\nIDEAL: <think>\nunclosed reasoning cut off")[0] is None

    # Translated labels. A pass judging a Russian conversation writes the marker in
    # Russian and keeps the enum in English — observed verbatim on a live run as
    # ``Вердикт: keep`` / ``Почему: …``, which parsed as no verdict at all and cost the
    # exchange a retry at a raised temperature.
    v, why, p = parse_revision_judgement(
        "Вердикт: keep\nПочему: Ответ уже мой, без сглаживания.\n"
        "LANG_DRIFT: no\nPERSONA_TARGET: none")
    assert v == "keep", v
    assert why == "Ответ уже мой, без сглаживания.", why
    # …and a language the alias table does not list, recovered from the value alone.
    assert parse_revision_judgement("Urteil: revise\nBegründung: geglättet")[0] == "revise"
    assert parse_revision_fields("**Вердикт**: revise\nПочему: x\nИдеал: " + ideal)[0] == "revise"

    # The loose value-keyed pattern is scoped to the judgement head: a reply that opens a
    # line with a colon is prose, not a field. (parse_revision_judgement, so the usable
    # IDEAL can't supply the verdict by inference and mask the check.)
    assert parse_revision_judgement(
        "WHY: x\nIDEAL: <think>\nr\n</think>\nOne note: keep the first paragraph."
    )[0] is None

    # JSON fallback: the model emits the whole revision as a JSON object instead of
    # LABEL: value lines. The label anchors match nothing, so recover from the object.
    bare_json = json.dumps({
        "verdict": "revise",
        "why": "fell into the analyst trap",
        "lang_drift": "no",
        "persona_target": "Я предпочитаю прямой ответ.",
        "ideal": ideal,
    })
    v, why, i, p = parse_revision_fields(bare_json)
    assert v == "revise" and why and ideal_has_usable_answer(i), (v, why, i[:40])
    assert p == ["Я предпочитаю прямой ответ."], p
    assert revision_lang_drift(bare_json) is False, "JSON lang_drift must be read"

    # Wrapped in a ```json fence + prose, mixed-case keys, boolean lang_drift, and a
    # persona list — all recovered.
    fenced = ("Here is my revision:\n```json\n" + json.dumps({
        "Verdict": "revise", "Why": "x", "lang_drift": True,
        "Persona_Target": ["[persona] first", "[persona] second"], "IDEAL": ideal,
    }) + "\n```\nDone.")
    v, _why, i, p = parse_revision_fields(fenced)
    assert v == "revise" and ideal_has_usable_answer(i), (v, i[:40])
    assert p == ["first", "second"], p
    assert revision_lang_drift(fenced) is True, "boolean JSON lang_drift"

    # JSON keep with no IDEAL still parses to a clean keep (not dropped).
    assert parse_revision_fields(json.dumps({"verdict": "keep", "why": "mine"}))[0] == "keep"

    # A normal label body whose IDEAL reply merely CONTAINS a brace must NOT enter the
    # JSON path (verdict present via label → guard holds; the reply survives intact).
    v, _why, i, _p = parse_revision_fields(
        'VERDICT: revise\nWHY: x\nIDEAL: <think>\nok\n</think>\nUse {"key": "value"} here.')
    assert v == "revise" and '{"key": "value"}' in i, i

    # Non-JSON garbage with no verdict/IDEAL stays unparseable (None), not a false parse.
    assert parse_revision_fields("total nonsense {not json")[0] is None

    # Origin timestamp: a chat session stem (with/without suffix or '-' separator) parses
    # to the chat's own date; a self-directed / non-chat source yields None (field omitted).
    assert source_origin_ts("20260705_014505") == "2026-07-05T01:45:05"
    assert source_origin_ts("20260705_014505.json") == "2026-07-05T01:45:05"
    assert source_origin_ts("20260705-014505.state.json") == "2026-07-05T01:45:05"
    assert source_origin_ts("20260705_014505_1.json") == "2026-07-05T01:45:05"  # same-second collision suffix
    for non_chat in ("wiki", "wiki:Foo", "til:20260701", "web:visit", "lookup", "", None):
        assert source_origin_ts(non_chat) is None, non_chat

    # ── attribution: who said it vs. who it is about ─────────────────────────
    # Name matching is first-token and case-insensitive, and generic referents name
    # nobody (matching on "the user" would fuse two different people into one subject).
    assert normalize_person("Artemy") == normalize_person("artemy voikhansky") == "artemy"
    for nobody in ("the user", "The Person", "they", "", None, "  ", "n/a"):
        assert normalize_person(nobody) == "", nobody

    assert derive_source_class("Artemy", "Artemy") == "self"
    assert derive_source_class("Artemy", "Boris") == "hearsay"
    # Either side unnamed ⇒ observed, the pre-attribution default: a world fact, Ava's
    # own reading, or a legacy record keeps its weights path unchanged.
    assert derive_source_class("Artemy", None) == "observed"
    assert derive_source_class("", "Boris") == "observed"
    assert derive_source_class("Artemy", "the user") == "observed"

    # `(about: …)` parses out of both sections, alongside `(trigger: …)`, and leaves the
    # statement itself clean. A [persona] is about Ava, so it never carries a subject.
    p = parse_consolidation(
        "## WEIGHTS\n"
        "- [fact] Boris is training for a marathon (about: Boris)\n"
        "- [persona] I value directness\n"
        "## RAG\n"
        "- [fact] Boris mentioned knee trouble (about: Boris) (trigger: running comes up)\n"
    )
    assert p["weights"][0] == {"weights_kind": "fact", "about": "Boris",
                               "content": "Boris is training for a marathon",
                               "lang": p["weights"][0]["lang"]}, p["weights"][0]
    assert p["weights"][1]["about"] is None, p["weights"][1]
    rag_fact = p["rag"][0]
    assert (rag_fact["about"], rag_fact["trigger"]) == ("Boris", "running comes up")
    assert rag_fact["content"] == "Boris mentioned knee trouble", rag_fact

    # Hearsay gate: Artemy's account of Boris is recalled (with attribution) but is held
    # out of the weights store, so it raises no ledger anchor and is never CoT-injected.
    import tempfile
    _d = Path(tempfile.mkdtemp())
    _mem, _con = _d / "mem", _d / "con"
    _mem.mkdir(); _con.mkdir()
    _w = ReflectionWriter(_mem, _con)
    _s = _w.write_consolidation(
        run_id="r1", source_session="20260101_120000", source_user="Artemy",
        text="## WEIGHTS\n"
             "- [fact] Boris is training for a marathon (about: Boris)\n"
             "- [fact] Artemy works on an AI project (about: Artemy)\n"
             "- [persona] I value directness\n",
    )
    assert (_s["weights"], _s["hearsay_held"]) == (2, 1), _s
    weights_rows = [json.loads(l) for l in
                    (_mem / ReflectionWriter.WEIGHTS_FILE).read_text().splitlines()]
    assert not any("Boris" in r["content"] for r in weights_rows), weights_rows
    assert not any("Boris" in f["content"] for f in _s["facts"]), _s["facts"]
    # …but it IS in live recall, carrying both names so it can be attributed.
    rag_rows = [json.loads(l) for l in
                (_mem / ReflectionWriter.RAG_FILE).read_text().splitlines()]
    hearsay = next(r for r in rag_rows if "Boris" in r["content"])
    assert (hearsay["about"], hearsay["source"], hearsay["source_class"]) \
        == ("Boris", "Artemy", "hearsay"), hearsay
    # The self-report promotes normally and keeps its subject for the ledger anchor.
    own = next(r for r in rag_rows if "AI project" in r["content"])
    assert own["source_class"] == "self", own
    assert next(f for f in _s["facts"] if "AI project" in f["content"])["about"] == "Artemy"
    # Persona is about Ava — no subject, no speaker, no class.
    persona = next(r for r in rag_rows if r["kind"] == "persona")
    assert not any(k in persona for k in ("about", "source", "source_class")), persona
    # Unique recall ids even though one item was held back from the weights.
    assert len({r["id"] for r in rag_rows}) == len(rag_rows), rag_rows

    # No source_user (Ava's own reading) ⇒ observed ⇒ promotes exactly as before.
    _w2 = ReflectionWriter(_d / "m2", _d / "c2")
    _s2 = _w2.write_consolidation(
        run_id="r2", source_session="wiki:Marathon",
        text="## WEIGHTS\n- [fact] A marathon is 42.195 km\n")
    assert (_s2["weights"], _s2["hearsay_held"]) == (1, 0), _s2

    # -- [impression]: parse + RAG-only write, attributed --------------------- #
    parsed = parse_impressions(
        "<think>what did I notice</think>\n"
        "[impression] he argues against his own ideas to test them\n"
        "- **[impression]** goes quiet when something matters (about: Artemy)\n"
        "[impression] Boris takes criticism badly (about: Boris)\n"
        "[fact] he works on an AI project\n"        # wrong kind for this pass — ignored
        "just some prose the model added\n"
        "[impression]   \n"                          # empty after the tag — dropped
    )
    assert [c for c, _a in parsed] == [
        "he argues against his own ideas to test them",
        "goes quiet when something matters",
        "Boris takes criticism badly",
    ], parsed
    assert [a for _c, a in parsed] == [None, "Artemy", "Boris"], parsed
    assert parse_impressions("no tags here at all") == []

    _w3 = ReflectionWriter(_d / "m3", _d / "c3")
    _s3 = _w3.write_impressions(impressions=parsed, source_session="20260701_120000",
                                source_user="Artemy", default_about="Artemy",
                                run_id="r3")
    assert _s3 == {"rag": 3}, _s3
    _rows3 = [json.loads(l) for l in
              (_d / "m3" / ReflectionWriter.RAG_FILE).read_text().splitlines()]
    assert all(r["kind"] == "impression" and r["op"] == "insert" for r in _rows3)
    # RAG-only: no weights line at all, so nothing here can ever train.
    assert not (_d / "m3" / ReflectionWriter.WEIGHTS_FILE).exists()
    # An unmarked item takes the session's speaker; a marked one keeps its own subject.
    assert _rows3[0]["about"] == "Artemy" and _rows3[0]["source_class"] == "self"
    assert _rows3[2]["about"] == "Boris" and _rows3[2]["source_class"] == "hearsay"
    assert all(r["source"] == "Artemy" for r in _rows3)
    assert _rows3[0]["origin_ts"] == "2026-07-01T12:00:00"
    assert len({r["key"] for r in _rows3}) == 3

    print("reflection_writer self-test OK")


if __name__ == "__main__":
    _selftest()
