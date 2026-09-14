"""Decoration-tolerant matching of field labels in generation output.

Every pass on this box asks the model for labelled fields — ``VERDICT:`` / ``WHY:`` /
``IDEAL:``, ``DECISION:``, ``ABOUT:``/``TAGS:``, ``RECAP:``, ``ACTION:`` — and then
line-anchors a regex on the label to read the value back. The model, left to itself,
decorates: it bolds the label, bullets the field, makes it a heading, wraps it in
backticks, or writes the fullwidth colon a CJK-adjacent tokenizer prefers. None of that
changes what it *said*, and all of it can change whether we can read it.

That mattered more than it sounds. The parsers had drifted into three tolerance levels —
``^\\s*[*_]*\\s*LABEL:`` in the revision family (emphasis outside the colon only),
``^\\s*LABEL\\s*:`` in the reach-out subsystems (no emphasis at all), and
``#{0,6}`` heading-only in the digest sections — so which decorations survived depended
on which pass had emitted them. Worse, they failed *quietly and unevenly*: a completion
written ``**VERDICT**: keep`` parsed under none of them and was reported as an
unparseable verdict, while the very same generation's ``**LANG_DRIFT**: no`` slipped past
``_TRAILING_FIELD_RE`` and would have been trained verbatim as part of Ava's answer.

So the tolerance is defined once, here, and every parser is built from it. A leaf module
by design (imports nothing from the project, like ``activity_log`` / ``reachout_gate``):
it is depended on by the writer, the anchor pass, the digest/portrait section splitters
and the reach-out subsystems alike, so it must never depend back.

Decoration is not the only thing that changes a label without changing what was said.
A pass reflecting on a Russian conversation writes its judgement into that conversation's
language, labels included — ``Вердикт: keep``, ``Почему: …`` — where the enum value
survives in English and only the marker is translated. That parses as nothing, and for
the revision pass "nothing" means the exchange is retried at a higher temperature and
then dropped from the corpus. So the labels a model has actually been seen to translate
are alternated in alongside the English form (``_ALIASES``), and :func:`any_label`
recovers a field whose label is in a language nobody listed by keying on its *value*.

What is deliberately NOT tolerated: a label appearing mid-line (every pattern is
``^``-anchored — a passing mention of "the verdict:" inside prose is not a field), and a
label with no colon (``**VERDICT** keep``), which is indistinguishable from ordinary
prose. Sections are the exception on the colon, having never had one.

GPU-free self-test: ``python -m core.field_parse``.
"""
from __future__ import annotations

# Leading decoration, in the order a model writes it: an emphasis run may wrap the whole
# line (`**## WEIGHTS**`), then heading hashes, then a list bullet, then the emphasis that
# wraps the label alone (`- **VERDICT:**`). Every part is independently optional. The
# bullet alternatives all require trailing whitespace, so the `*` bullet cannot swallow
# the first star of a `**LABEL**` emphasis run.
_EMPH = r"[*_`]*[ \t]*"
_BULLET = r"(?:(?:[-*+•]|\d+[.)])[ \t]+)?"


def _lead(hashes: str) -> str:
    return rf"[ \t]*{_EMPH}{hashes}{_BULLET}{_EMPH}"


_LEAD = _lead(r"(?:\#{1,6}[ \t]*)?")
# Between the label and its colon: `**VERDICT**:` closes the emphasis before the colon,
# `**VERDICT:**` after it. Both are common; the old patterns only ever allowed the second.
_MID = r"[ \t]*[*_`]*[ \t]*"
# After the colon, before the value.
_TAIL = r"[*_`]*[ \t]*"
# ASCII and fullwidth colon. The corpus is mixed-language and the fullwidth form shows up
# in output from a tokenizer that has been reading CJK-adjacent text.
_COLON = r"[:：]"

# Translated labels, alternated in wherever the English name is asked for. This is an
# OBSERVATION table, not a translation dictionary: an entry earns its place by having been
# seen in a run's events, because every entry is also a word that must never open a line of
# ordinary prose. `_TRAILING_FIELD_RE` in the writer cuts the trained answer at the first
# label it recognises, so a careless alias does not merely read a field that isn't there —
# it truncates a reply. That is why the obvious `VERDICT: РЕШЕНИЕ` is absent while
# `DECISION: РЕШЕНИЕ` is present: the reach-out passes only ever *read* their label, and
# "Решение:" is exactly the line a Russian answer legitimately opens with.
#
# Coverage is therefore deliberately partial, and does not need to be complete. A field
# whose label is translated into something unlisted degrades the way it did before this
# table existed, except for the one field that cannot afford it — see :func:`any_label`.
_ALIASES: dict[str, tuple[str, ...]] = {
    "VERDICT": ("ВЕРДИКТ",),
    "WHY": ("ПОЧЕМУ",),
    "IDEAL": ("ИДЕАЛ",),
    "DECISION": ("РЕШЕНИЕ",),
}

# A label in a language nobody listed: anything short and punctuation-free standing where
# a label stands. Never used to *find* a field — only to recover one whose value is a known
# enum (`keep|revise`), and only after the named patterns have all missed. The charset ends
# the match at sentence punctuation so a clause cannot pose as a label, and the length cap
# keeps it to a marker rather than a sentence.
_UNKNOWN_LABEL = r"[^\n:：.!?…,;]{1,32}"


def _expand(names: tuple[str, ...]) -> list[str]:
    """*names* plus any translated forms of them, deduplicated, order-insensitive."""
    out: list[str] = []
    for name in names:
        for candidate in (name, *_ALIASES.get(name.upper(), ())):
            if candidate not in out:
                out.append(candidate)
    return out


def label(*names: str, anchored: bool = True) -> str:
    """Regex fragment matching a field label and its colon at the start of a line.

    Consumes the label, its decoration and the whitespace after the colon, so a caller
    appends only its value pattern (``(.+?)\\s*$``, ``(.*)\\Z``, …). Multiple *names* are
    alternated, longest first — so ``label("PERSONA_TARGET", "PERSONA")`` cannot match the
    shorter name against the longer label's prefix.

    *anchored* False drops the line anchor for the handful of callers that deliberately
    match a label mid-line (``OPENER:`` / ``ANSWER:`` / ``RECAP:``, whose value is the
    greedy tail). Those still need the decoration handling on the *closing* side: an
    unanchored ``OPENER\\s*:`` happily matches inside ``**OPENER:** hi`` and captures
    ``** hi``, which for the reach-out passes is a message Ava then sends to the user with
    two stray asterisks on the front.

    Adds no flags of its own: the caller supplies ``re.M`` (required when anchored — the
    ``^`` is per-line) and ``re.I``.
    """
    alt = "|".join(sorted(_expand(names), key=len, reverse=True))
    head = rf"^{_LEAD}" if anchored else ""
    return rf"{head}(?:{alt}){_MID}{_COLON}{_TAIL}"


def any_label(anchored: bool = True) -> str:
    """Regex fragment matching *some* field label and its colon — the name unread.

    The last resort for a field whose value is a closed set. ``Вердикт: keep`` translates
    the marker and keeps the enum, and a pass whose whole output is one such line is
    otherwise indistinguishable from a pass that answered in prose. Keying on the value
    recovers it in any language, including ones ``_ALIASES`` will never list.

    Only ever appended to an enum (``(keep|revise)``), and only after every named pattern
    has missed — the label is unconstrained, so on its own this matches far too much.
    """
    head = rf"^{_LEAD}" if anchored else ""
    return rf"{head}{_UNKNOWN_LABEL}{_MID}{_COLON}{_TAIL}"


def section(*names: str, hash_required: bool = False) -> str:
    """Regex matching a section HEADING line, capturing the name in group 1.

    Sections (``## WEIGHTS``, ``VOICE``, ``WHO``) are headings rather than fields: they
    carry no colon and their value is the block beneath them, so this matches the whole
    line and the caller slices between successive matches.

    *hash_required* keeps at least one ``#`` mandatory. It exists for the consolidation
    headers (WEIGHTS / RAG / RESOLVED), whose names are ordinary enough words that a
    hash-free match would let a line of prose beginning "RAG …" split the output into
    sections. The self-portrait facets (VOICE / STANCES / …) have no such collision and
    are routinely written bare, so they leave it off.

    Deliberately does NOT expand ``_ALIASES``: a section name is captured in group 1 and
    read back as an identity (the digest maps it to a facet key), so a translated heading
    would have to be mapped home rather than merely matched. A drifted section header
    still fails to split, as it always has.
    """
    alt = "|".join(sorted((n for n in names), key=len, reverse=True))
    hashes = r"\#{1,6}[ \t]*" if hash_required else r"(?:\#{1,6}[ \t]*)?"
    return rf"^{_lead(hashes)}({alt})\b.*$"


if __name__ == "__main__":  # pragma: no cover - GPU-free self-test
    import re

    L = re.compile(label("VERDICT") + r"(keep|revise)\b", re.I | re.M)
    # Every decoration a model has been seen to add, and combinations of them.
    for form in (
        "VERDICT: keep",
        "VERDICT : keep",
        "**VERDICT:** keep",          # emphasis closes after the colon
        "**VERDICT**: keep",          # …and before it — the form that used to be dropped
        "*VERDICT*: keep",
        "__VERDICT__: keep",
        "`VERDICT`: keep",
        "### VERDICT: keep",
        "## **VERDICT:** keep",
        "- **VERDICT:** keep",
        "* VERDICT: keep",
        "1. VERDICT: keep",
        "  *   VERDICT: keep",        # the indented bullet the CoT writes
        "VERDICT:keep",
        "VERDICT：keep",              # fullwidth colon
        "verdict: keep",
    ):
        assert L.search(form), form
    # …and what must still NOT match.
    for form in (
        "the VERDICT: keep",          # mid-line mention, not a field
        "VERDICT keep",               # no colon
        "VERDICTS: keep",             # different label
    ):
        assert not L.search(form), form

    # Longest-first alternation: PERSONA must not claim a PERSONA_TARGET line.
    P = re.compile(label("PERSONA_TARGET", "PERSONA") + r"(.*)$", re.I | re.M)
    m = P.search("**PERSONA_TARGET**: [persona] x")
    assert m and m.group(1).strip() == "[persona] x", m and m.group(1)

    # Unanchored: matches mid-line, and still eats the emphasis that closes after the
    # colon — the reach-out passes send this value to the user verbatim.
    U = re.compile(label("OPENER", anchored=False) + r"(.*)\Z", re.I | re.S)
    for form, want in (("OPENER: hi", "hi"),
                       ("**OPENER:** hi", "hi"),
                       ("**OPENER**: hi", "hi"),
                       ("…so, OPENER: hi", "hi")):
        m = U.search(form)
        assert m and m.group(1) == want, (form, m and m.group(1))

    # Multi-name alternation used as a cut (the trailing-field strip).
    T = re.compile(label("VERDICT", "WHY", "LANG_DRIFT", "IDEAL"), re.I | re.M)
    assert T.search("reply text\n**LANG_DRIFT**: no\n")
    assert not T.search("reply text\nno drift here\n")

    # Translated labels: the observed drift, at the same decoration tolerance, in either
    # case — and a listed alias must not admit an unlisted one.
    for form in ("Вердикт: keep", "ВЕРДИКТ: keep", "**Вердикт:** revise", "- вердикт: keep"):
        assert L.search(form), form
    assert not L.search("Решение: keep")          # unlisted for VERDICT, on purpose
    W = re.compile(label("WHY") + r"(.+?)[ \t]*$", re.I | re.M)
    m = W.search("Почему: ответ уже мой")
    assert m and m.group(1) == "ответ уже мой", m and m.group(1)

    # Value-keyed recovery: the label may be anything, the value may not.
    A = re.compile(any_label() + r"(keep|revise)\b", re.I | re.M)
    for form in ("Вердикт: keep", "判定: revise", "**Urteil**: keep", "Vurdering : revise"):
        assert A.search(form), form
    for form in (
        "I would keep it, honestly: it is already mine",  # a clause, not a label
        "Вердикт: оставить",                              # value translated too — not covered
        "keep",                                           # no label, no colon
    ):
        assert not A.search(form), form

    S = re.compile(section("VOICE", "STANCES"), re.I | re.M)
    for form in ("VOICE", "## VOICE", "**VOICE**", "### **VOICE:**", "- VOICE"):
        m = S.search(form)
        assert m and m.group(1).upper() == "VOICE", form
    H = re.compile(section("WEIGHTS", "RAG", hash_required=True), re.I | re.M)
    for form in ("## WEIGHTS", "### RAG", "## **WEIGHTS**", "**## WEIGHTS**"):
        assert H.search(form), form
    assert not H.search("RAG retrieval was noisy")   # bare prose is not a header
    assert not H.search("WEIGHTS")                   # …nor a bare name, without a hash
    print("field_parse self-test OK")
