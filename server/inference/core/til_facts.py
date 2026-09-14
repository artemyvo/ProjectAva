"""Per-source fact-extraction record for the TIL lane — the protocol of one article.

The wander/news counterpart of `core.chat_facts`, and it exists for the same reason that
one does. The learning passes on this lane (`learning_prompt.txt`, `wander_prompt.txt`)
are *curation* passes: they are told in as many words that most of what they read will be
noise and that finding almost all of it irrelevant is correct, and they emit the handful
of WEIGHTS/RAG items worth keeping. That is the right discipline for memory and the wrong
one for a record — a digest of a day's world events is read once, six facts are kept, and
everything else the text established is gone the moment the pass ends. Measured on the
live corpus (2026-08-09): 4 news digests yielded 26 `[fact]` records between them, and 10
wandered articles yielded 15.

So this is the enumerated literal layer underneath that, exactly as the chat protocol is
the layer underneath consolidation, with the same contract: **immutable, exhaustive,
never deduped, never evicted, never merged**, and read by nothing at runtime. The
consumer is offline — the knowledge-graph build, which without this lane would be a graph
of what the people Ava talks to said, since wander and news are the only paths by which
anything about the wider world enters the box.

**What it shares with the chat protocol.** The parser (`chat_facts.parse_facts`, called
with this module's subject resolver), the class vocabulary (`standing`/`stated`/`event`),
the entity-mention rules, the truncation contract, and the staleness probe. One parser
across both lanes, because a marker placement or a format change is a property of the
pass, not of the material.

**Where it diverges, and why each divergence is forced.**

1. *The subject namespace.* `chat_facts.normalize_subject` resolves to a PERSON key via
   `reflection_writer.normalize_person`, which lowercases and reduces to a first token so
   "Artemy Voikhansky" matches "Artemy". On a chat that is right — the subjects are the
   handful of people the box knows, plus Ava. Here nearly every subject is a country, an
   organisation, a fictional trope or an event, none of which live in that namespace: the
   person reducer would file "New York" under `new` and "United States" under `united`,
   and everything it did not recognise would collapse to `""`, meaning *nobody in
   particular*. That collapse would be invisible, because on a world-facts lane an
   unowned fact is the normal case. So a subject here is an **entity mention**, kept in
   its surface form and normalized only for formatting — the identical argument
   `chat_facts.normalize_entities` already makes for the `entities` field, applied to the
   subject slot because on this lane the subject IS one of those nodes. Resolving mentions
   to canonical nodes needs the whole corpus at once and belongs to the build.

2. *Provenance instead of attribution.* A chat fact carries `about`/`source`/`source_class`
   — who it concerns and who said it — because the speaker is a person whose testimony can
   be first-hand or hearsay. Here the source is a *text*, so the record carries the text's
   identity instead (`source_kind`/`source_ref`/`source_url`/`source_title`/`source_date`)
   and the fact-level attribution is simply absent. This matters more on this lane than
   the chat one: the approved wiki list is chosen for **tone**, not for truth — Lurkmore,
   Neolurk and WikiTropes are humour wikis — so a Current-events digest and a joke wiki
   asserting the same sentence are not the same evidence.

   Deliberately NOT resolved into a per-fact confidence or authority score. Judging how
   much a source is to be believed is an interpretation, and this pass is defined as a
   witness; a score invented per line would also be the one field of the record a later
   build could not recompute. Provenance is recorded exactly, and weighting by it is the
   build's job — the same split as mention resolution above.

Storage: ``<snippet_stem>.facts.json``, beside the ``.txt``/``.json`` the fetchers already
write under ``server/data/til/snippets/<kind>/``. That tree IS the stem space for this
lane — better than the chat case, where the transcript and its record are separate files
that have to be kept in step — and it is inside what `snapshot_state` captures, so the
records travel with the corpus that produced them.

GPU-free self-test: ``python -m core.til_facts``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from core import chat_facts
from core.chat_facts import FACT_CLASSES, UNSPECIFIED, class_counts, needs_reparse  # re-export


SCHEMA_VERSION = 1

# The lanes this record can come from. **These are directory names**, and that is the whole
# contract: `iter_snippets` resolves `<snippets>/<kind>/`, so a value here that does not name
# a real dir silently enumerates nothing.
#
# `lookups` was written `lookup` from the day this module shipped, while
# `til/fetch_article.py` has always written into `snippets/lookups/`. Nothing failed — the
# backlogs simply never saw that directory, so **18 fetched articles on this box had no
# protocol and no recap**, and never would have. Nor could a graph-lane ref for one
# (`lookups/<stem>`, built from `path.parent.name`) resolve back through
# `til_gist.resolve_source`. Caught by a self-test written for an unrelated fix; the tell
# was `_lookup_source_id` returning `lookup/x.txt` for a file living in `lookups/`.
SOURCE_KINDS = ("news", "wander", "lookups")

# A subject is a mention, so it takes the entity ceiling rather than a person key's. Longer
# than that is a clause, which means the pass wrote a sentence into the marker.
MAX_SUBJECT_CHARS = chat_facts.MAX_ENTITY_CHARS

# Sidecar suffix, mirroring `chats/<stem>.facts.json`.
FACTS_SUFFIX = ".facts.json"

# Every DERIVED file that lives in a snippets kind-dir under a snippet's stem — this
# lane's `chat_sidecar.SIDECAR_SUFFIXES`, and load-bearing for the same reason. A snippet
# owns its stem, `iter_snippets` is the one place that decides what counts as a *source
# text*, and a suffix missing from here is not cosmetic: the recap or the protocol gets
# enumerated as an article, so the protocol pass records facts about Ava's own summary and
# the graph build ingests them as things the world stated. Caught exactly that way when
# `.summary.json` arrived (`core.til_gist`, whose self-test asserts its suffix is listed).
# The strings are literal rather than imported from each owner, because that import would
# be a cycle — `til_gist` reads this module.
DERIVED_SUFFIXES = (FACTS_SUFFIX, ".summary.json")

_WS_RE = re.compile(r"\s+")

# Labels that name no particular entity. Kept SHORT and literal on purpose: this is a
# formatting rule, not a judgement about what the sentence was about. Anything not on this
# list is taken at face value as a mention, because a subject the pass wrote and this
# module refused to record is a fact silently filed under nobody — the exact failure this
# lane's whole subject treatment exists to avoid.
_NO_SUBJECT = frozenset({
    "", "-", "—", "n/a", "na", "none", "nobody", "no one", "nothing",
    "unknown", "unspecified", "general", "the world", "world",
})


def normalize_subject(about_raw: Optional[str]) -> str:
    """Subject key for a TIL protocol line — an entity MENTION, or ``""``.

    Formatting normalization only (collapse whitespace, strip decoration, cap length) with
    the **surface form preserved**, exactly as :func:`chat_facts.normalize_entities` treats
    the other nodes a fact touches. Deliberately not `normalize_person`: see this module's
    docstring for why a person key destroys a world-facts corpus.

    A handful of literal non-answers map to ``""`` (*a fact about nothing in particular* —
    a general observation the article makes). Everything else is taken at face value.
    """
    s = _WS_RE.sub(" ", str(about_raw or "")).strip().strip(chat_facts._ENTITY_STRIP).strip()
    if not s or s.casefold() in _NO_SUBJECT:
        return ""
    return s[:MAX_SUBJECT_CHARS].strip()


def parse_facts(raw: str, *, truncated: bool = False) -> list:
    """Parse a TIL facts-pass generation. The chat parser, with this lane's subject rule."""
    return chat_facts.parse_facts(raw, truncated=truncated, subject_fn=normalize_subject)


# ── the reading content ─────────────────────────────────────────────────────── #

# Per-kind framing. The lane matters to the reader in a way it does not to the writer of a
# chat protocol: a current-events digest is a list of things that happened on a date, while
# a wiki article is a standing description of one subject, and asking for "every fact" of
# each without saying which is which produces a digest read as biography and an article
# read as news.
_KIND_FRAMING = {
    "news": ("This is a digest of world events from a single day, {date}. Most of its "
             "lines report something that happened — expect (class: event), and resolve "
             "each one's date against {date} rather than against today."),
    # Deliberately does NOT call this an encyclopedia article. Three of the five enabled
    # wander sources are wikis about invented material (WikiTropes entirely; Lurkmore and
    # Neolurk largely), and the manual-visit path fetches whatever URL an operator gives it
    # — a story text included. The old wording asserted the genre AND that "most of what it
    # states is standing description of its subject", which on a fiction page is an
    # instruction to file an invented world's contents as fact.
    "wander": ("This is a text titled {title}, from {ref}. It may be an encyclopedia "
               "article, an essay, a piece of internet folklore, or a work of fiction — "
               "read what is actually in front of you rather than assuming which. Where it "
               "describes what happens inside a story, a game or an invented world, that "
               "is (class: depicted). Where it reports that someone said, claimed or "
               "believes something, that is (class: stated) — the record is that it was "
               "said, not that it is true. One thing genre cannot be read off: dates. "
               "Your own knowledge has a horizon and the world has kept moving past it, "
               "so an account of events later than the ones you carry — a war, an "
               "election, a release you have never heard of — is an account, not fiction; "
               "a date beyond your training is not evidence of invention."),
    # The horizon note is load-bearing HERE most of all: this lane fetches articles to
    # answer questions Ava wrote down, so its texts are disproportionately about the
    # recent world — and it is where the failure was observed live (2026-08-18): a pass
    # over the "2026 Iran war" article spent its whole budget deliberating "this is
    # clearly fictional (alternate history/prediction)", and the one run that completed
    # filed 48 of 48 facts (class: depicted) — a facet no fetch channel is ever offered,
    # so the war's record was invisible while looking recorded. `learning_prompt.txt` has
    # carried the same reassurance since the news lane shipped; the protocol lane simply
    # never inherited it.
    "lookups": ("This is a text titled {title}, fetched to answer a question you had "
                "written down. It may not be the kind of source you expected, and it may "
                "be a work rather than an account of one — read what is in front of you. "
                "But remember that your own knowledge has a horizon and the world has "
                "kept moving past it: real events have happened since your training that "
                "you are reading about for the first time, and an account of them is an "
                "account, not fiction; a date beyond your training is not evidence of "
                "invention."),
}

_CLOSING = (
    "— end of the text —\n\n"
    "Now write the protocol: every fact this text states, one per [fact] line, each "
    "marked with what it is about and whether it is standing, stated or an event. "
    "Record what the text says, not what you make of it and not whether you believe it."
)


def framing_for(kind: str, record: dict) -> str:
    """The one-paragraph note that tells the pass what KIND of text it is reading."""
    template = _KIND_FRAMING.get(str(kind or "").strip(), _KIND_FRAMING["wander"])
    rec = record or {}
    return template.format(
        date=rec.get("date") or "the date given above",
        title=rec.get("title") or "its subject",
        ref=rec.get("wiki") or rec.get("source") or "a wiki",
    )


def pack_paragraphs(text: str, budget: int) -> list:
    """Split *text* on paragraph boundaries and pack greedily into ``budget``-sized parts.

    Shared with :mod:`core.til_gist`, which packs the same material to a different end —
    where a block break falls is a property of the TEXT, not of the pass reading it, and
    two copies of this loop would be two places for a seam rule to drift.

    A single paragraph longer than the budget is hard-split (rare, and losing the tail of
    a long paragraph is worse than a seam inside it).
    """
    budget = max(500, int(budget))
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]
    packed: list = []
    cur = ""
    for p in paras:
        while len(p) > budget:                       # a paragraph bigger than one block
            if cur:
                packed.append(cur)
                cur = ""
            packed.append(p[:budget])
            p = p[budget:]
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= budget:
            cur += "\n\n" + p
        else:
            packed.append(cur)
            cur = p
    if cur:
        packed.append(cur)
    return packed


def build_reading_blocks(record: dict, kind: str, max_chars: int) -> list:
    """The source text as one or more blocks, each ready to hand to the pass.

    Several blocks rather than one, because this lane's material genuinely does not fit:
    the live corpus holds wandered articles from 690 to 26,001 characters (median 6.4k)
    against a news digest's 1.9k–9.4k. A chat protocol reads its session in one block
    because a transcript is already bounded by the context the chat ran in; an article is
    bounded by nothing.

    Split on paragraph boundaries and packed greedily, so a block break falls between
    facts rather than through one.
    """
    text = str((record or {}).get("text") or "").strip()
    if not text:
        return []
    budget = max(500, int(max_chars))
    head = framing_for(kind, record).strip()
    title = str((record or {}).get("title") or "").strip()

    packed = pack_paragraphs(text, budget)

    blocks: list = []
    for i, body in enumerate(packed, 1):
        part = f" (part {i} of {len(packed)})" if len(packed) > 1 else ""
        header = f"{head}\n\nTEXT{part}"
        if title:
            header += f" — {title}"
        blocks.append(f"{header}:\n\n{body}\n\n{_CLOSING}")
    return blocks


# ── the record on disk ──────────────────────────────────────────────────────── #

def sidecar_path(snippet_path) -> Path:
    """``…/<stem>.facts.json`` for a snippet's ``.txt`` or ``.json`` path."""
    p = Path(snippet_path)
    stem = p.name
    for suffix in (".txt", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return p.parent / (stem + FACTS_SUFFIX)


def build_doc(*, record: dict, kind: str, facts: list, run_id: str) -> dict:
    """The document written to ``<stem>.facts.json``. Provenance first, then the records."""
    from datetime import datetime
    rec = record or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_kind": str(kind or "").strip(),
        # What the text IS, verbatim from the fetcher's own provenance — never inferred,
        # and never reduced to a score. See the module docstring.
        "source_ref": rec.get("wiki") or rec.get("source") or "",
        "source_url": rec.get("source_url") or "",
        "source_title": rec.get("title") or "",
        "source_date": rec.get("date") or "",
        "source_lang": rec.get("lang") or "",
        "facts": list(facts or []),
        "run_id": str(run_id or ""),
        "ts": datetime.now().isoformat(),
    }


def write_facts(snippet_path, *, record: dict, kind: str, facts: list,
                run_id: str) -> Optional[Path]:
    """Write the protocol beside its snippet. Overwrites; returns the path, or None.

    Overwrite-in-full for the same reason the chat protocol overwrites: the record is this
    text's extraction *as of the pass that read it*, so a re-derivation under a better
    prompt or a fixed parser replaces it wholesale. Immutability here is against dedup and
    eviction churn — nothing merges or evicts inside this file — not against re-derivation.
    """
    if not facts:
        return None
    path = sidecar_path(snippet_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = build_doc(record=record, kind=kind, facts=facts, run_id=run_id)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def read_facts(snippet_path) -> dict:
    """The protocol beside a snippet, or ``{}`` when there is none / it is unreadable."""
    try:
        return json.loads(sidecar_path(snippet_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def has_facts(snippet_path) -> bool:
    """True when a usable, non-stale protocol already exists for this snippet.

    Staleness is `chat_facts.needs_reparse` — one probe across both lanes, so a parser
    improvement re-derives TIL protocols exactly as it re-derives chat ones, and neither
    can be forgotten when the other is fixed.
    """
    doc = read_facts(snippet_path)
    return bool(doc.get("facts")) and not needs_reparse(doc)


def iter_snippets(snippets_dir, kind: str) -> list:
    """The ``.json`` provenance files under ``snippets/<kind>/``, oldest first by name.

    **The one place that decides what is a source text** — see `DERIVED_SUFFIXES`. Every
    derived file in the dir shares its snippet's stem and ends in `.json`, so the filter
    here is what keeps a recap or a protocol from being read back as an article.
    """
    d = Path(snippets_dir) / str(kind or "")
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json")
                  if not p.name.endswith(DERIVED_SUFFIXES))


def block_budget_chars(window_tokens: int) -> int:
    """Character budget for ONE reading block, from the window the pass will run in.

    Its own function rather than `til_wander._wander_max_chars` (which they were briefly
    sharing) because the two answer different questions that merely happen to land on
    similar numbers: that one caps how much of an article to FETCH and keep, this one caps
    how much to put in front of the pass at a time — so a text over the cap is split here
    and truncated there. Same conservative ~2.7 chars/token, sized for Cyrillic's denser
    tokenization so a Russian page still fits its token budget.
    """
    tokens = max(1024, int(window_tokens or 0))
    return max(2000, min(60_000, int(tokens * 0.30 * 2.7)))


def input_id(path, kind: str) -> str:
    """The stable `<kind>/<name>` id the workbench addresses a snippet by."""
    return f"{kind}/{Path(path).name}"


def load_input(snippets_dir, ident: str) -> tuple:
    """``(record, kind)`` for a `<kind>/<name>` id. Path-guarded to the snippets tree."""
    root = Path(snippets_dir).resolve()
    kind, _, name = str(ident or "").partition("/")
    if kind not in SOURCE_KINDS or not name:
        raise ValueError(f"bad input id: {ident!r}")
    path = (root / kind / name).resolve()
    if path.parent != (root / kind).resolve():
        raise ValueError("path traversal")
    if not path.exists():
        raise FileNotFoundError(ident)
    return json.loads(path.read_text(encoding="utf-8")), kind


def list_inputs(snippets_dir, kinds: tuple = SOURCE_KINDS) -> list:
    """Every fetched text the protocol pass can run on, NEWEST first.

    The workbench's input list, and deliberately not :func:`list_backlog` — an operator
    tuning the prompt wants the texts whose protocol they can already read and compare
    against, which are exactly the ones the backlog has finished with. `has_facts` rides
    along so the list can say which is which.
    """
    out: list = []
    for kind in kinds:
        for p in iter_snippets(snippets_dir, kind):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "id": input_id(p, kind),
                "kind": kind,
                "label": (rec.get("title") or p.stem)[:120],
                "sub": " · ".join(x for x in (
                    kind,
                    rec.get("wiki") or rec.get("source") or "",
                    rec.get("date") or "",
                    f"{len(rec.get('text') or '')} chars") if x),
                "chars": len(rec.get("text") or ""),
                "has_facts": has_facts(p),
            })
    return sorted(out, key=lambda r: r["id"], reverse=True)


def list_backlog(snippets_dir, kinds: tuple = SOURCE_KINDS) -> list:
    """Snippets with no protocol (or a stale one), oldest first. ``[(path, kind), …]``.

    The TIL counterpart of `background_reflection.list_sidecar_backlog`, and the reason
    this lane needs one at all: unlike a chat — which is reflected once, soon after it
    happens, by a pass that runs anyway — an article is read exactly once, at fetch, and
    never looked at again. Without a backlog the protocol would exist only for material
    fetched after the day this shipped, and every article already on disk would stay
    unrecorded forever.
    """
    out: list = []
    for kind in kinds:
        for p in iter_snippets(snippets_dir, kind):
            if not has_facts(p):
                out.append((p, kind))
    return sorted(out, key=lambda t: (t[0].parent.name, t[0].name))


# --------------------------------------------------------------------------- #
# GPU-free self-test                                                           #
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import tempfile

    def check(label, got, want):
        status = "ok " if got == want else "FAIL"
        print(f"{status}  {label}: got {got!r}, want {want!r}")
        assert got == want, label

    # ---- the subject namespace, which is the whole reason this module exists ------- #
    check("a multi-word place keeps both words", normalize_subject("New York"), "New York")
    check("...where the person rule would have kept one",
          chat_facts.normalize_subject("New York"), "new")
    check("case and surface form are preserved for the resolver",
          normalize_subject("United States"), "United States")
    check("decoration stripped", normalize_subject('  "Nvidia".  '), "Nvidia")
    check("whitespace collapsed", normalize_subject("Strait  of\nHormuz"),
          "Strait of Hormuz")
    check("a literal non-answer means nothing in particular",
          [normalize_subject(x) for x in ("", "  ", "none", "N/A", "the world")],
          ["", "", "", "", ""])
    check("an unrecognized subject is taken at face value, never dropped",
          normalize_subject("Роскомнадзор"), "Роскомнадзор")
    check("a clause in the marker is capped, not stored whole",
          len(normalize_subject("x" * (MAX_SUBJECT_CHARS + 40))), MAX_SUBJECT_CHARS)

    # ---- the parser is the chat one, so every placement/marker rule comes free ----- #
    facts = parse_facts(
        "[fact] (about: Singapore) (class: event) (when: 2026) Tightened monetary policy.\n"
        "[fact] Struck an ADNOC vessel. (about: Houthi movement) (class: event) "
        "(entities: ADNOC, Red Sea).\n"
        "[fact] (about: New York) (class: standing) Has a subway.")
    check("count", len(facts), 3)
    check("leading markers", (facts[0]["subject"], facts[0]["fact_class"],
                              facts[0]["when"]), ("Singapore", "event", "2026"))
    check("trailing markers, inherited from the shared parser",
          (facts[1]["subject"], facts[1]["entities"]),
          ("Houthi movement", ["ADNOC", "Red Sea"]))
    check("...and the subject rule is this lane's", facts[2]["subject"], "New York")
    check("class vocabulary is shared across lanes", FACT_CLASSES,
          chat_facts.FACT_CLASSES)
    check("truncation drops the half-written tail",
          len(parse_facts("[fact] (about: A) x.\n[fact] (about: B) y", truncated=True)), 1)

    # ---- reading blocks ------------------------------------------------------------ #
    rec = {"title": "Силач", "wiki": "Neolurk", "date": "2026-08-09", "lang": "ru",
           "text": "\n\n".join(f"Paragraph {i} " + "z" * 200 for i in range(10))}
    one = build_reading_blocks(rec, "wander", 100_000)
    check("a short text is one block", len(one), 1)
    check("...carrying the framing", "Силач" in one[0] and "Neolurk" in one[0], True)
    # The framing must NAME the source without CLASSIFYING the text. Three of the five
    # enabled wander sources are wikis about invented material and the manual-visit path
    # takes any URL, so asserting "this is an encyclopedia article" is how a story's
    # contents get filed as standing fact. Asserted as a property, not as a phrase, so a
    # future rewording does not have to touch this line.
    check("...without asserting what kind of text it is",
          "encyclopedia article about" in one[0], False)
    check("...and offering the fiction class", "(class: depicted)" in one[0], True)
    # The counter-inference the fiction class made available (2026-08-18, live): a pass
    # reading a real post-cutoff event — the 2026 Iran war — concluded "clearly
    # fictional (alternate history/prediction)" and filed 48/48 facts depicted, a facet
    # no fetch channel is offered. Both framings that admit fiction must carry the
    # horizon note; asserted as the property (a date is not evidence of invention), on
    # the phrase both templates share.
    check("...while saying a late date is not evidence of invention",
          "not evidence of invention" in one[0], True)
    check("the lookups framing carries the horizon note too",
          "not evidence of invention" in framing_for("lookups", rec), True)
    check("news needs none — its framing dates the digest",
          "not evidence of invention" in framing_for("news", {"date": "2026-08-08"}),
          False)
    check("...and the closing", one[0].endswith(_CLOSING), True)
    many = build_reading_blocks(rec, "wander", 800)
    check("a long text is several", len(many) > 1, True)
    check("every block carries the closing", all(b.endswith(_CLOSING) for b in many), True)
    check("...and says which part it is", "part 1 of" in many[0], True)
    check("the whole text survives the split",
          all(f"Paragraph {i} " in "".join(many) for i in range(10)), True)
    # A paragraph bigger than one block is split rather than dropped.
    huge = build_reading_blocks({"title": "T", "text": "q" * 5000}, "wander", 900)
    check("an oversized paragraph is hard-split, not lost",
          "".join(huge).count("q"), 5000)
    check("empty text ⇒ no blocks", build_reading_blocks({"text": "  "}, "news", 900), [])
    # The framing is per-kind: a digest is dated events, an article is standing description.
    check("news framing anchors the date",
          "2026-08-08" in framing_for("news", {"date": "2026-08-08"}), True)
    check("an unknown kind degrades to the article framing",
          framing_for("nonsense", rec), framing_for("wander", rec))

    # ---- the record on disk -------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as td:
        snips = Path(td) / "snippets"
        (snips / "news").mkdir(parents=True)
        snippet = snips / "news" / "2026-08-08.json"
        snippet.write_text("{}", encoding="utf-8")
        (snips / "news" / "2026-08-08.txt").write_text("x", encoding="utf-8")

        check("sidecar sits beside the snippet",
              sidecar_path(snippet).name, "2026-08-08.facts.json")
        check("...derived identically from either member",
              sidecar_path(snips / "news" / "2026-08-08.txt"), sidecar_path(snippet))
        check("no protocol yet", has_facts(snippet), False)
        check("backlog finds it", [p.name for p, _ in list_backlog(snips)],
              ["2026-08-08.json"])

        newsrec = {"title": "Portal:Current events", "source": "wikipedia:current_events",
                   "source_url": "https://en.wikipedia.org/x", "date": "2026-08-08"}
        written = write_facts(snippet, record=newsrec, kind="news", run_id="r1",
                              facts=parse_facts(
                                  "[fact] (about: Singapore) (class: event) Tightened policy."))
        check("written", written is not None and written.exists(), True)
        doc = read_facts(snippet)
        check("provenance is the text's identity, not a person's",
              (doc["source_kind"], doc["source_ref"], doc["source_date"]),
              ("news", "wikipedia:current_events", "2026-08-08"))
        check("no fact-level attribution on this lane",
              [k for k in doc["facts"][0] if k in ("about", "source", "source_class")], [])
        check("protocol now present", has_facts(snippet), True)
        check("...so the backlog is empty", list_backlog(snips), [])

        # Nothing to record is not an artifact: an empty protocol would read as "this text
        # established nothing" when it means "the pass came back empty", and it would take
        # the snippet out of the backlog so it could never be retried.
        empty_snip = snips / "news" / "2026-08-09.json"
        empty_snip.write_text("{}", encoding="utf-8")
        check("an empty result writes no file",
              write_facts(empty_snip, record={}, kind="news", facts=[], run_id="r"), None)
        check("...and the snippet stays in the backlog",
              [p.name for p, _ in list_backlog(snips)], ["2026-08-09.json"])

        # Staleness is the shared probe, so a parser fix reaches this lane too.
        stale = [{"subject": "", "subject_raw": "", "fact_class": UNSPECIFIED,
                  "entities": [], "when": "",
                  "text": "Tightened policy. (about: Singapore) (class: event)"}]
        write_facts(snippet, record=newsrec, kind="news", facts=stale, run_id="r2")
        check("a stale protocol counts as absent", has_facts(snippet), False)
        check("...and comes back in the backlog",
              [p.name for p, _ in list_backlog(snips)],
              ["2026-08-08.json", "2026-08-09.json"])

        check("a protocol file is never mistaken for a snippet",
              [p.name for p in iter_snippets(snips, "news")],
              ["2026-08-08.json", "2026-08-09.json"])
        check("an absent kind dir is empty, not an error", iter_snippets(snips, "wander"), [])

    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
