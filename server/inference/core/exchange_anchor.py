"""Per-exchange retrieval ANCHORS: a one-line descriptor + normalized tags.

Producer-side only. This module holds the pure, GPU-free half — output parsing, tag
normalization and the filler gate — so the shapes can be tested on the client machine
and the reflection runner stays thin. **Nothing retrieves on these yet**: they are
generated during reflection, written to the chat sidecar, and otherwise inert. Wiring
them into `rag_engine` as an anchor channel is a deliberately separate step.

Why they exist
--------------
Chat RAG currently embeds raw dialogue split into 280-char passages and matches the
live user turn against them. Three consequences, all measured on the live corpus:

* Relevance is decided on a 280-char slice while the whole exchange (clipped at 4000
  chars/side) is injected — match granularity and injection granularity are decoupled,
  which is what makes retrieved chats look arbitrary.
* A near-verbatim quote scores above `RagEngine._MAX_SCORE` (0.90) and is dropped by
  the anti-induction-copy band-pass, so deliberately probing a remembered phrase is
  the one thing guaranteed not to retrieve it.
* Summaries of Russian conversations are written in English 55% of the time, so a
  Russian query is matched cross-lingually against an English description.

An anchor is a *generated descriptor* rather than raw text, which addresses all three:
it is one unit per exchange (so the matched unit is the payload target), it is a
paraphrase and therefore never near-identical to a live query, and it is authored in
the conversation's own language.

Tags, and why they are normalized here
--------------------------------------
Tags are the sparse counterpart to the descriptor's dense matching — the channel that
catches rare coined tokens (`крокодильничество`, `RESET`) that mean-pooled embeddings
dilute. `[fact].trigger` is the same idea, already running uncontrolled, and its
outcome is the argument for normalization: across 695 live facts it produced **1,588
distinct triggers with only 10% used more than once**, and the corpus's most-used
concept fragmented into three separate tags —

    "crocodiling"        13
    "крокодильничество"  13
    crocodiling          11

— split by quoting and by language. `normalize_tag` collapses the formatting half of
that (quotes, case, decoration, punctuation); the cross-language half needs an alias
registry and is NOT solved here.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from core.field_parse import label as _label

# Bump when the prompt or parse contract changes, so a stored anchor names the recipe
# that produced it and a later reader can tell generations apart.
ANCHOR_GENERATION = "exchange_anchor_v1"

# A user turn shorter than this carries no substance worth anchoring ("да", "ok",
# "why?"). Same threshold and same reasoning as the training corpus's
# `contamination_min_user_chars`: below it there is nothing to describe, and an
# anchor for it would be noise that still clears a retrieval floor.
MIN_USER_CHARS = 100

MAX_TAGS = 8           # keep a tag set readable and stop a runaway list
MAX_TAG_CHARS = 40     # a "tag" longer than this is a sentence, not a label
MAX_ABOUT_CHARS = 300  # one line; anything longer is the model narrating

_TAG_SPLIT_RE = re.compile(r"[,;\n]+")
# Decoration to strip from a tag's edges: quotes of several scripts, brackets, the
# social-media hash, list bullets, and trailing sentence punctuation.
_TAG_STRIP = " \t\r\n#*•·-–—_.!?:;\"'«»“”„‟‘’()[]{}"


def normalize_tag(raw: str) -> str:
    """Normalize one tag to its comparable form, or ``""`` if nothing usable remains.

    Lowercases, strips edge decoration (quotes/hash/bullets/punctuation) and collapses
    inner whitespace. This is deliberately the *formatting* half of tag convergence
    only: it merges `"crocodiling"` with `crocodiling`, but NOT `крокодильничество`
    with `crocodiling` — cross-language aliasing needs a registry that can hold members,
    and inventing a mapping here would silently fuse unrelated concepts.
    """
    t = (raw or "").strip().lower()
    if not t:
        return ""
    t = t.strip(_TAG_STRIP)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t) > MAX_TAG_CHARS:
        return ""
    # A bare number or single character is never a useful retrieval key.
    if len(t) < 2 or t.isdigit():
        return ""
    return t


def normalize_tags(raw_tags, *, max_tags: int = MAX_TAGS) -> list:
    """Normalize, de-duplicate (order-preserving) and cap a tag list."""
    out: list = []
    seen = set()
    for item in raw_tags or []:
        tag = normalize_tag(item)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= max_tags:
            break
    return out


def _label_value(text: str, label: str) -> str:
    """Value of a ``LABEL:`` line, matched by label like the revision pass's fields.

    Label-keyed rather than positional for the same reason `PERSONA:` is: the model
    reorders and omits fields, and a bare `TAGS: a, b` must still be captured when the
    surrounding format drifts. The LAST occurrence wins — when the model restates the
    block (a common shape after a self-correction) the final one is its answer.
    """
    matches = re.findall(
        _label(label) + r"(.+)$", text or "", re.IGNORECASE | re.MULTILINE
    )
    return matches[-1].strip() if matches else ""


def parse_anchor_output(raw: str) -> tuple:
    """Parse a generation into ``(about, tags)``; ``about`` is ``""`` when unusable.

    Tolerant by design — the pass is cheap and re-runs next reflection, so a partial
    parse (tags but no ABOUT, or vice versa) is kept rather than discarded.
    """
    text = (raw or "").strip()
    if not text:
        return "", []

    about = _label_value(text, "ABOUT")
    # Strip a wrapping quote pair the model sometimes adds around the description.
    about = about.strip().strip('"').strip("«»").strip()
    about = re.sub(r"\s+", " ", about).strip()
    if len(about) > MAX_ABOUT_CHARS:
        cut = about[:MAX_ABOUT_CHARS]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        about = (cut[: dot + 1] if dot > 60 else cut.rstrip()) + " …"

    tags = normalize_tags(_TAG_SPLIT_RE.split(_label_value(text, "TAGS")))
    return about, tags


def is_anchorable(exchange: dict, *, min_user_chars: int = MIN_USER_CHARS) -> bool:
    """Whether an exchange carries enough substance to be worth anchoring.

    Requires a real user turn AND a reply, and a user turn past *min_user_chars*. The
    synthetic opener of an Ava-initiated session is excluded by the caller (it has no
    genuine user turn), matching `reflection_source.build_revision_jobs`.
    """
    if not isinstance(exchange, dict):
        return False
    user = (exchange.get("user_prompt") or "").strip()
    reply = (exchange.get("assistant_response") or "").strip()
    if not user or not reply:
        return False
    return len(user) >= max(0, int(min_user_chars))


def has_content(about: str, tags) -> bool:
    """True when a parsed anchor is worth persisting (either half is enough)."""
    return bool((about or "").strip()) or bool(tags)


# --------------------------------------------------------------------- #
# Query side: match a live user turn against the stored tags              #
# --------------------------------------------------------------------- #
#
# Deliberately lexical and generation-free. A tag-generation pass at query time would sit
# *before* retrieval and so add seconds to time-to-first-token on every turn; matching the
# raw text costs microseconds and covers the case dense retrieval loses — rare coined
# tokens (`крокодильничество`, `RESET`) that mean-pooled embeddings dilute away.
#
# The hard part is morphology, not matching. Russian inflects heavily, so the user writes
# `крокодильничеством` while the tag reads `крокодильничество`; an exact or substring test
# misses, and the channel would look like "tags don't work" when it is really "tags don't
# decline". A stemmer would be correct and is a dependency; a shared-prefix test is neither
# correct nor free of false positives, but it is stdlib and it exploits the shape of the
# words that matter here — a distinctive coined term is long, so a long shared prefix is
# strong evidence, while short words are exactly the ones we do not want matching anyway.

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Below this, a word is too short to carry a distinctive prefix and must match exactly.
_MIN_PREFIX_WORD = 6
# Chars allowed to differ at the tail of a long word (inflectional endings).
_MAX_ENDING_DIFF = 3
# A tag on more than this fraction of the corpus discriminates nothing and is dropped.
_MAX_DOC_FRACTION = 0.5
# ...but only once it is on at least this many documents. Without the floor the fraction
# rule misfires on a thin corpus — a tag on 2 of 3 anchors is "67% of the corpus" and would
# be silently discarded, which is precisely the state the corpus is in while anchors are
# still accumulating.
_FILLER_MIN_DOCS = 4


def query_tokens(text: str) -> list:
    """Lowercased word tokens of a live user turn, in order, deduplicated."""
    seen = set()
    out = []
    for m in _WORD_RE.finditer((text or "").lower()):
        tok = m.group(0)
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def words_match(tag_word: str, token: str) -> bool:
    """Whether a tag word and a query token are the same word, modulo inflection.

    Exact for short words; for longer ones, a shared prefix that leaves at most
    ``_MAX_ENDING_DIFF`` characters differing on EITHER side — so `крокодильничество`
    matches `крокодильничеством` and `крокодильничества`, while `крокодил` (a different
    word that merely shares a stem) does not reach it from `крокодильничество` because
    the length gap exceeds the allowance.
    """
    a, b = (tag_word or "").lower(), (token or "").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < _MIN_PREFIX_WORD or len(b) < _MIN_PREFIX_WORD:
        return False
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    if common < _MIN_PREFIX_WORD:
        return False
    return (len(a) - common) <= _MAX_ENDING_DIFF and (len(b) - common) <= _MAX_ENDING_DIFF


def tag_matches(tag: str, tokens) -> bool:
    """Whether every content word of *tag* appears in *tokens* (modulo inflection).

    A multi-word tag ("единицы измерения") is a conjunction: partial matches are what
    turn a specific tag into a generic one.
    """
    words = [w for w in _WORD_RE.findall((tag or "").lower()) if len(w) > 1]
    if not words:
        return False
    return all(any(words_match(w, t) for t in tokens) for w in words)


def build_tag_stats(entries) -> dict:
    """``{tag: document_count}`` over anchor *entries* — the corpus side of the IDF weight."""
    counts: dict = {}
    for e in entries or []:
        for tag in set(e.get("tags") or []):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def tag_weight(tag: str, doc_count: int, corpus_size: int) -> float:
    """How much a matched tag is worth: distinctiveness × specificity.

    Two factors, both pointing the same way. **Rarity** — a tag on half the corpus
    discriminates nothing (plain IDF). **Length** — a long coined term is a far stronger
    signal than a common short noun, and length is the cheapest available proxy for
    "distinctive" without a frequency list of the language.

    Known limit, stated because the preview surface is how it will be found: this measures
    rarity *within the tag vocabulary*, which does not catch a tag that is a common word of
    the language. The live corpus has `слушай` ("listen") as a tag with 7 uses — rare among
    tags, so it scores as distinctive here, while actually being discourse filler that will
    match half the user's messages. Catching that needs a stoplist or language frequency
    data; neither is guessed at here.
    """
    if corpus_size <= 0 or doc_count <= 0:
        return 0.0
    if (doc_count >= _FILLER_MIN_DOCS
            and doc_count > max(1, int(corpus_size * _MAX_DOC_FRACTION))):
        return 0.0
    idf = math.log(1.0 + corpus_size / doc_count)
    length_bonus = min(2.0, len(tag) / 10.0)
    return idf * (0.5 + length_bonus)


def match_query(text: str, entries, *, limit: int = 20) -> list:
    """Rank anchor *entries* against a live user turn by tag overlap.

    *entries* are dicts with ``session`` / ``exchange_index`` / ``about`` / ``tags``.
    Returns the best matches, highest score first, each carrying the tags that fired so a
    reader can see WHY it matched — the point of the preview surface.

    Scores are not collapsed per chat here: this is the diagnostic view, where seeing two
    exchanges of one conversation match is information, not noise. A retrieval caller
    would collapse (see the wander channel's `ranked_by_source`).
    """
    tokens = query_tokens(text)
    if not tokens or not entries:
        return []
    stats = build_tag_stats(entries)
    corpus = len(entries)

    out = []
    for e in entries:
        hits = []
        score = 0.0
        for tag in e.get("tags") or []:
            if not tag_matches(tag, tokens):
                continue
            w = tag_weight(tag, stats.get(tag, 1), corpus)
            if w <= 0.0:
                continue
            hits.append(tag)
            score += w
        if not hits:
            continue
        out.append({
            "session": e.get("session", ""),
            "exchange_index": e.get("exchange_index"),
            "about": e.get("about", ""),
            "tags": list(e.get("tags") or []),
            "matched": hits,
            "score": round(score, 3),
        })
    out.sort(key=lambda r: (-r["score"], r["session"], r["exchange_index"] or 0))
    return out[:max(1, int(limit))]


def load_corpus(*chats_dirs) -> list:
    """Read every stored anchor out of the sidecars under *chats_dirs*.

    Returns a flat list of ``{session, exchange_index, about, tags}``. Later directories
    win on a repeated stem (same precedence as `rag_engine._collect_chat_entries`, where
    the live dir overrides the fallback). Best-effort per file: a malformed sidecar is
    skipped rather than failing the whole read.

    Uncached on purpose. The corpus is one small JSON per chat and the caller is a
    debounced preview, so a re-read costs milliseconds and can never serve a stale view —
    which matters while anchors are actively being produced by a background reflection
    running on the same box.
    """
    import json
    from pathlib import Path

    by_key: dict = {}
    for d in chats_dirs:
        if d is None:
            continue
        p = Path(d)
        if not p.exists():
            continue
        for sidecar in sorted(p.glob("*.state.json")):
            try:
                doc = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue
            anchors = doc.get("anchors")
            if not isinstance(anchors, dict):
                continue
            session = doc.get("source_session") or (
                sidecar.name[: -len(".state.json")] + ".json")
            for idx, rec in anchors.items():
                if not isinstance(rec, dict):
                    continue
                about = (rec.get("about") or "").strip()
                tags = [t for t in (rec.get("tags") or []) if (t or "").strip()]
                if not about and not tags:
                    continue
                try:
                    ex_index = int(idx)
                except (TypeError, ValueError):
                    continue
                by_key[(session, ex_index)] = {
                    "session": session,
                    "exchange_index": ex_index,
                    "about": about,
                    "tags": tags,
                }
    return list(by_key.values())


def _selftest() -> None:
    """GPU-free self-test. Run: ``python -m core.exchange_anchor``."""
    # Label parsing, order-independent and case-insensitive.
    about, tags = parse_anchor_output(
        "ABOUT: Разговор про крокодильничество и абсурд\n"
        "TAGS: крокодильничество, абсурд, язык"
    )
    assert about == "Разговор про крокодильничество и абсурд", about
    assert tags == ["крокодильничество", "абсурд", "язык"], tags
    about, tags = parse_anchor_output("tags: vinyl, RESET\nabout: A talk about boots")
    assert about == "A talk about boots" and tags == ["vinyl", "reset"], (about, tags)

    # The measured formatting fragmentation collapses.
    assert normalize_tag('"crocodiling"') == normalize_tag("crocodiling") == "crocodiling"
    assert normalize_tag("#Subjectivity") == "subjectivity"
    assert normalize_tag("  «Абсурд»  ") == "абсурд"
    # ...but cross-language aliasing is deliberately NOT attempted here.
    assert normalize_tag("крокодильничество") != normalize_tag("crocodiling")

    # Junk rejected; duplicates collapsed; cap honoured.
    assert normalize_tag("") == "" and normalize_tag("...") == "" and normalize_tag("7") == ""
    assert normalize_tag("x" * 100) == ""
    assert normalize_tag("я") == "", "a single character is not a retrieval key"
    assert normalize_tags(['"Vinyl"', "vinyl", " VINYL ", "boots"]) == ["vinyl", "boots"]
    assert len(normalize_tags([f"tag{i}" for i in range(50)])) == MAX_TAGS

    # Partial output is kept, not discarded.
    assert parse_anchor_output("TAGS: only, tags") == ("", ["only", "tags"])
    assert parse_anchor_output("ABOUT: only prose")[1] == []
    assert parse_anchor_output("") == ("", []) and not has_content("", [])
    # A restated block: the last occurrence wins.
    assert parse_anchor_output("ABOUT: first\nABOUT: second")[0] == "second"
    # An over-long ABOUT is clipped, not dropped.
    long_about = parse_anchor_output("ABOUT: " + "word " * 200)[0]
    assert long_about.endswith("…") and len(long_about) <= MAX_ABOUT_CHARS + 2

    # Filler gate.
    assert not is_anchorable({"user_prompt": "да", "assistant_response": "ok"})
    assert not is_anchorable({"user_prompt": "x" * 200, "assistant_response": ""})
    assert is_anchorable({"user_prompt": "x" * 200, "assistant_response": "reply"})

    # ── query side ────────────────────────────────────────────────────────────
    # Russian inflection is the whole reason this is not an exact match.
    assert words_match("крокодильничество", "крокодильничеством")
    assert words_match("крокодильничество", "крокодильничества")
    assert words_match("крокодильничество", "крокодильничество")
    assert words_match("измерения", "измерение")
    # ...but a shorter related word is NOT the same word.
    assert not words_match("крокодильничество", "крокодил")
    # Short words must match exactly, or "код"/"кот" style collisions creep in.
    assert words_match("vinyl", "vinyl") and not words_match("vinyl", "vinyls!")
    assert not words_match("код", "кот")

    # A multi-word tag is a conjunction — a partial hit is what makes a tag generic.
    toks = query_tokens("а что там было про единицы измерения?")
    assert tag_matches("единицы измерения", toks)
    assert not tag_matches("единицы времени", toks)

    entries = [
        {"session": "a.json", "exchange_index": 0, "about": "про крокодильничество",
         "tags": ["крокодильничество", "абсурд"]},
        {"session": "b.json", "exchange_index": 2, "about": "про абсурд вообще",
         "tags": ["абсурд"]},
        {"session": "c.json", "exchange_index": 1, "about": "не связано",
         "tags": ["виниловые ботинки"]},
    ]
    hits = match_query("помнишь наш разговор про крокодильничеством?", entries)
    assert [h["session"] for h in hits] == ["a.json"], hits
    assert hits[0]["matched"] == ["крокодильничество"], hits[0]
    # Both entries carrying a matched tag surface; the more distinctive one ranks first.
    hits = match_query("это про абсурд и крокодильничество", entries)
    assert [h["session"] for h in hits] == ["a.json", "b.json"], hits
    assert hits[0]["score"] > hits[1]["score"]
    assert match_query("совершенно другая тема", entries) == []
    assert match_query("", entries) == [] and match_query("абсурд", []) == []

    # A tag on most of the corpus is a discourse filler and scores nothing.
    filler = [{"session": f"{i}.json", "exchange_index": 0, "about": "",
               "tags": ["слушай"]} for i in range(4)]
    assert match_query("слушай, а что если", filler) == []

    print("exchange_anchor selftest: OK")


if __name__ == "__main__":
    _selftest()
