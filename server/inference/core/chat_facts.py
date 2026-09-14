"""Per-chat fact-extraction record — the immutable "protocol" of one conversation.

The fourth thing reflection derives from a chat, beside the per-exchange verdicts
(``.state.json``), the prose recap (``.summary.json``) and the distilled memory that
lands in ``rag_memory.jsonl``. This one is the *enumerated literal layer* underneath
all of them: everything factual the conversation established, written down plainly,
in the register of a protocol rather than a memoir.

**Why it is not just more ``[fact]`` records.** The consolidation pass's ``[fact]``
output is *curated* — it is bound for the weights, so it emits what is worth
remembering forever, and it is deduped, evicted, superseded and merged across chats
by ``fact_dedup``. That makes the live store the right answer to "what does Ava
believe now" and the wrong place to answer "what did this conversation establish".
A fact merged from four chats belongs to no single chat; an evicted one leaves no
trace of having been said at all.

So the two stores have different jobs, and the split is deliberate:

* ``rag_memory.jsonl`` stays **authoritative** — it is what retrieval reads, what
  eviction acts on, what the Facts tab curates. Nothing here changes that.
* this record is an **immutable extraction record** — what THIS chat yielded, as of
  the run that read it. Never deduped, never evicted, never rewritten by a later
  merge. Its consumer is offline: a knowledge-graph build, an audit, a re-derivation
  of the live store after a wipe.

Same relationship ``weights_persona.jsonl`` already has to the ledger: a source and
a fold, not two copies of one truth.

**Nothing injects this.** No RAG channel reads it, by design. The injected reflection
block has three slots and already needs a subject cap and a near-duplicate rule to
stay legible; multiplying extraction volume into that block would raise the noise
floor of everything competing for it. Extraction volume is only safe *because* the
consumer is offline.

Storage: ``chats/<stem>.facts.json``, written by ``ChatSidecar.write_facts``, which
is the sole writer of files in the chats dir's sidecar space.

GPU-free self-test: ``python -m core.chat_facts``.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from core.reasoning_text import REASONING_LEAK_MARKERS, answer_after_think


SCHEMA_VERSION = 1

# What KIND of claim a line is — the distinction the live fact store does not draw and
# a graph build cannot work without.
#
# `source_class` (self/hearsay/observed) already records who *said* a thing, which is a
# different question from what kind of thing was said. "Artemy drinks Reviseur XO on the
# balcony at night" and "Artemy said Trump is manufacturing the Hormuz crisis deliberately"
# are both first-hand, both true as records — but the first is a standing property of a
# person and the second is a position held by that person at a moment. Collapsing them
# loses the only property that matters downstream: the first is still true next month,
# the second can be contradicted by the same person next week and must never be recalled
# as something Ava simply knows.
#
# `depicted` (2026-08-16) is the TIL lane's fourth, and it names the case the other three
# have no answer for: **a text that is not asserting anything about the world at all.** The
# witness contract ("record what the text says, not whether you believe it") is right for a
# news digest — a false report is still a report — and it collapses on fiction, where the
# propositions are not claims about the world in the first place. Three of the five enabled
# wander sources are wikis about invented material (WikiTropes entirely, Lurkmore and
# Neolurk largely), so without this a protocol over "Мир Элдерлингов" files a fantasy
# world's contents as `standing`, `fold` maps that to `property`, and live chat is offered
# it as something simply known. Routed instead to its own facet, carried only where it can
# be marked as depiction. Never produced on the chat lane — `chat_facts_prompt.txt` does not
# mention it — but it lives here because both lanes share this parser and vocabulary.
FACT_CLASSES = ("standing", "stated", "event", "depicted")

# Written when the pass omitted or garbled the marker. Deliberately its own value rather
# than a guessed default: for a record whose whole purpose is later machine processing,
# "unlabelled" is information and a wrong label is contamination.
UNSPECIFIED = "unspecified"

# The subject key for a fact about AVA HERSELF, and the reason it must exist.
#
# `normalize_person` collapses a fixed set of generic referents — "me", "i", "user",
# "the user" among them — to "", which in this schema means *a fact about nobody in
# particular* (a place, a tool, a piece of the world). Nothing in the pipeline gives Ava
# a name: `chat_prompt.txt` names nobody, the reflect lane composes no identity line, and
# `session_transcript_turns` renders her turns as a bare "Me:" for replay fidelity. So a
# pass asked for `(about: NAME)` had, for her own facts, only labels that mean "nobody" —
# and every fact she stated about herself was filed as a fact about no one. Invisible on
# an ordinary chat (the facts are mostly about the named user); on a conversation she
# started, where most of the content is hers, it is nearly all of them.
#
# A reserved marker rather than a configured name: this project's premise is that her
# identity accumulates rather than being declared, so naming her by fiat to satisfy a
# schema would be the wrong trade. The precedent is exact — `user_digest.RESERVED_SLUGS`
# holds `_self` for the outside-view portrait, which exists because she is not a person in
# the people namespace either. The leading underscore is what keeps it unforgeable:
# `normalize_person` strips surrounding punctuation but not underscores, so no natural
# name can normalize onto this key.
#
# Deliberately local to this module: `normalize_person` is shared with the live fact
# store, eviction and dedup, and widening its vocabulary would reach all of them.
SELF_SUBJECT = "_self"

# Raw `about:` labels that mean Ava herself. First-person only — the pass writing the
# label IS her, so "me"/"I" in its output are unambiguous. "user"/"the user" are NOT here:
# they normally mean the human, and a pass that reached for them while meaning herself is
# a prompt problem to fix rather than a mapping to guess at. Her *name* is deliberately
# absent too: recognizing a name here would put one hardcoded string in a codebase that
# has none, and would misfile the facts of any human who shares it. The prompt teaches
# `self`; a stray `(about: Ava)` stays an ordinary person key rather than being guessed at.
_SELF_REFERENTS = frozenset({"self", "myself", "me", "i"})

# Reversed-session framing for THIS pass, replacing `reflection_source._AVA_INITIATED_NOTE`.
#
# That default is written for the revision pass: it closes "when judging your reply" and
# tells her the self-descriptions in a session she started are her own "not facts about
# them". Every reading pass inherited it, and for a pass whose job is to write down what
# was said, "not facts about them" with no positive counterpart reads as an instruction to
# skip her own turns — which is what she did, dropping her own opener from the protocol.
# This says the same structural thing (the roles are inverted) and then says where her own
# material goes.
#
# It lives here, in the pure module that owns this pass's data contract, because both
# callers need it — `reflection_runner`'s production pass and `core.modules`'s registry
# entry — and a second copy is a second thing to keep in step.
AVA_INITIATED_NOTE = (
    "NOTE — this is a conversation YOU started, on your own initiative. The roles are "
    "inverted from an ordinary chat: the opening message is yours, and much of what this "
    "conversation established is about YOU rather than about the other person. That does "
    "not make it less of a record. Write your own facts down exactly as you would "
    "theirs, marked (about: self), starting with what you said in the opening message."
)

# Per-fact ceilings. Both are runaway guards, not editorial limits.
MAX_ENTITIES = 8
MAX_ENTITY_CHARS = 60      # a place or an organisation name; longer is a clause
MAX_WHEN_CHARS = 60

_ENTITY_SPLIT_RE = re.compile(r"[,;]+")
# Decoration to strip off an entity mention's edges — quotes of several scripts, brackets,
# bullets, trailing sentence punctuation. Mirrors `exchange_anchor._TAG_STRIP`'s intent;
# NOT its lowercasing, since a mention's surface form is what a later resolver matches on.
_ENTITY_STRIP = " \t\"'`«»„“”‘’()[]{}<>#*•-–—.,;:!?"

_FACT_LINE_RE = re.compile(r"^\s*[-*•]?\s*\[fact\]\s*", re.IGNORECASE)
# `(key: value)` markers, consumed as a run off the FRONT and off the BACK of the content.
# Both ends, because the pass uses both placements — see `_consume_markers` for the
# measurement that forced this and for why the middle of a line is deliberately not scanned.
_MARKER_LEAD_RE = re.compile(r"^\s*\(\s*([A-Za-z_]+)\s*:\s*([^)]*)\)\s*")
# The trailing form tolerates a stray sentence terminator after the run — the pass writes
# `… documents. (class: event) (entities: Nvidia, Taiwan, China).`, closing the LINE rather
# than the statement, and the statement already carries its own full stop. Safe because
# consumption is restricted to the known keys below: an ordinary parenthetical ending a
# sentence is not one of them and survives as text.
_MARKER_TAIL_RE = re.compile(r"\s*\(\s*([A-Za-z_]+)\s*:\s*([^)]*)\)\s*[.;,]?\s*$")
# The marker vocabulary, as key → record field. Consumption is restricted to these, so a
# sentence that merely ENDS in a parenthetical — "…moved to Anthropic (formerly: OpenAI)" —
# keeps it as text instead of having it silently eaten as an unrecognized marker. An
# unknown key stops the run rather than being dropped: nothing this pass writes is
# recoverable later, so preserving an unparsed scrap beats discarding it.
_MARKER_FIELDS = {
    "about": "about", "subject": "about", "who": "about",
    "class": "class", "kind": "class", "type": "class",
    "entities": "entities", "with": "entities", "also": "entities",
    "when": "when", "date": "when",
}
_WS_RE = re.compile(r"\s+")

# The prompt's own output-format line, echoed back as though it were a fact (observed
# twice in the live corpus: `(about: NAME) … the fact, in one plain sentence.`). The
# discriminator is the placeholder VALUE — a schema-level property — and deliberately not
# the prompt's wording, which is operator-editable on disk and must not be something the
# parser knows. Case-sensitive, which is what keeps the rule provably narrow: the prompt
# writes the placeholder in caps, and no person is written under the name "NAME".
_PLACEHOLDER_ABOUT = frozenset({"NAME"})
_PLACEHOLDER_ENTITIES = ("A", "B")

# A protocol line is a statement, not a paragraph. The cap is loose enough not to clip a
# genuinely compound fact and tight enough that a pass which drifts into prose is caught.
MAX_FACT_CHARS = 400
# Per-conversation ceiling. Exhaustive is the point, so this sits far above what the
# curated consolidation pass emits — it is a runaway guard, not an editorial limit.
MAX_FACTS = 60


def normalize_class(value: Optional[str]) -> str:
    """Map a model-supplied class marker onto :data:`FACT_CLASSES`, else UNSPECIFIED."""
    s = _WS_RE.sub(" ", str(value or "")).strip().strip(".,;:!?'\"()[]").lower()
    if s in FACT_CLASSES:
        return s
    # Tolerate the near-misses a generation reliably produces around a closed vocabulary.
    aliases = {
        "standing_fact": "standing", "fact": "standing", "property": "standing",
        "habit": "standing", "trait": "standing", "biography": "standing",
        "opinion": "stated", "claim": "stated", "view": "stated", "belief": "stated",
        "statement": "stated", "position": "stated",
        "events": "event", "happening": "event", "action": "event",
        "fiction": "depicted", "fictional": "depicted", "in_universe": "depicted",
        "narrative": "depicted", "plot": "depicted", "depiction": "depicted",
    }
    return aliases.get(s, UNSPECIFIED)


def normalize_subject(about_raw: Optional[str]) -> str:
    """Subject key for a protocol line: a person, :data:`SELF_SUBJECT`, or ``""``.

    ``normalize_person`` with one carve-out — a first-person label resolves to Ava
    instead of to "nobody". See :data:`SELF_SUBJECT` for why that carve-out has to exist
    and why it is local to this module.
    """
    from core.reflection_writer import normalize_person

    s = _WS_RE.sub(" ", str(about_raw or "")).strip().strip(".,;:!?'\"()[]").lower()
    if s in _SELF_REFERENTS:
        return SELF_SUBJECT
    return normalize_person(about_raw)


def session_participants(session: Optional[dict]) -> list[str]:
    """Ordered distinct recorded spellings of this session's other participant(s).

    The session-level ``user`` first — the spelling the box files this person under
    everywhere else — then any per-exchange ``speaker`` that names someone new. A stage
    direction ("(initiative)", "(setting)") occupies the speaker slot without naming
    anyone and is skipped through the one definition of that rule
    (``reflection_source.is_narrator_speaker``); so is any label ``normalize_person``
    reduces to nobody. Dedup is by person key with inflection tolerance
    (``exchange_anchor.words_match``), first spelling wins — so a session recording
    "artemyvo" on one exchange and "Artemy Voikhansky" on another yields ONE participant
    under the first spelling, rather than two fold targets that would re-split the
    subject space this function exists to close.
    """
    from core.reflection_source import is_narrator_speaker
    from core.reflection_writer import normalize_person
    from core.exchange_anchor import words_match

    sess = session or {}
    candidates = [sess.get("user") or ""]
    for ex in sess.get("exchanges") or []:
        candidates.append((ex or {}).get("speaker") or "")

    names: list[str] = []
    keys: list[str] = []
    for raw_name in candidates:
        name = _WS_RE.sub(" ", str(raw_name or "")).strip()
        if not name or is_narrator_speaker(name):
            continue
        key = normalize_person(name)
        if not key:
            continue
        if any(key == k or words_match(key, k) for k in keys):
            continue
        names.append(name)
        keys.append(key)
    return names


def participants_note(session: Optional[dict]) -> str:
    """Per-session labelling note: the canonical ``(about:)`` spellings, stated in-context.

    The pass is asked for ``(about: NAME)`` while the prompt never says WHICH name — so
    the model picks whatever is salient at that token: the transcript's speaker label one
    line, the name its weights associate with the person the next, the conversation's own
    language the third ("artemyvo" / "Artemy" / "Артемий" observed for one person across
    one corpus). The caller structurally knows the recorded spelling; this puts it in
    front of the pass, which is the strongest lever — a parse-time fold can only
    reconcile string-related variants, while the note stops cross-script drift at the
    source. Composed into the reading content's *closing* (never the prompt file, which
    stays slot-free like its siblings) so it lands on EVERY session — the
    ``ava_initiated_note`` slot only reaches reversed ones.

    Also restates the self rule with the two leaks the prompt's "Me"/"the user"
    explanation does not cover: her own name and "AI". Returns "" when no participant is
    nameable (then the pass behaves exactly as before).
    """
    names = session_participants(session)
    if not names:
        return ""
    if len(names) == 1:
        who = (f'NOTE ON NAMES — the other participant in this conversation is recorded '
               f'as "{names[0]}". A fact about them takes (about: {names[0]}) — exactly '
               f'that spelling')
    else:
        listed = ", ".join(f'"{n}"' for n in names)
        who = (f"NOTE ON NAMES — the participants in this conversation are recorded as "
               f"{listed}. A fact about one of them takes their recorded name in the "
               f"about marker, exactly as written here")
    return (
        f"{who}, even when the conversation is in another language: never a "
        "translation, a transliteration, a nickname, or \"the user\". A fact about "
        "yourself still takes (about: self) — never your own name, never \"AI\". Anyone "
        "else keeps the name the conversation itself uses for them."
    )


def make_subject_fn(session: Optional[dict]) -> Callable[[Optional[str]], str]:
    """Session-aware subject resolver — :func:`normalize_subject` plus a participant fold.

    The parse-side backstop behind :func:`participants_note`: a label that names a
    recorded participant resolves to that participant's canonical key, whatever spelling
    the pass reached for. "Names a participant" is decided conservatively — the label's
    person key must equal a recorded key or match it modulo inflection
    (``exchange_anchor.words_match``: a shared prefix of ≥6 chars leaving ≤3 differing on
    either side, exact below that). That folds "Artemy" / "Artemyvo" / "artemyvo's" onto
    a recorded "artemyvo", and Russian case endings onto a Cyrillic-recorded name, while
    a third party is structurally safe: "Alexey" cannot reach a recorded "Alex" (short
    names demand exactness) and "Artemis" cannot reach "artemy" (the shared prefix is 5).
    What it deliberately does NOT fold is a cross-script variant ("Артемий" against a
    recorded "artemyvo") — no string relation exists, that residue belongs to the graph
    build's alias table, and the note upstream is what makes it rare.

    Self labels and nobody-labels pass through :func:`normalize_subject` untouched, so
    the ``_self`` carve-out and the refuse-to-guess rule for generic referents hold
    exactly as before.
    """
    from core.reflection_writer import normalize_person
    from core.exchange_anchor import words_match

    keys = [normalize_person(n) for n in session_participants(session)]
    keys = [k for k in keys if k]

    def resolve(about_raw: Optional[str]) -> str:
        base = normalize_subject(about_raw)
        if not base or base == SELF_SUBJECT:
            return base
        for key in keys:
            if base == key or words_match(base, key):
                return key
        return base

    return resolve


def normalize_entities(raw: Optional[str]) -> list:
    """The other entities a fact touches, as MENTION strings — the graph's other nodes.

    Stored as mentions rather than resolved keys, and that split is the point. Extracting
    the mention needs the *conversation*: "her sister lives in Haifa" is one node the
    sentence names and one it only refers to, and the pass reading the transcript knows
    which person "her" was. That knowledge is gone by the time an offline build sees the
    row, so it has to be captured now. RESOLVING mentions to canonical nodes needs the
    opposite thing — the whole corpus at once, to know that "Haifa" here and "חיפה" there
    are one place — so it belongs to the build, which can also re-run when it gets better.

    Hence formatting normalization only (split, strip decoration, collapse whitespace,
    dedup case-insensitively, cap) with the **surface form preserved**: a later resolver
    matches on what was written, and lowercasing "Haifa" throws away a signal for nothing.
    Deliberately NOT ``normalize_person``, which reduces to a first token — right for a
    person key, and it would turn "New York" into "new".
    """
    out: list = []
    seen: set = set()
    for piece in _ENTITY_SPLIT_RE.split(str(raw or "")):
        name = _WS_RE.sub(" ", piece).strip().strip(_ENTITY_STRIP).strip()
        if not name or len(name) > MAX_ENTITY_CHARS:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= MAX_ENTITIES:
            break
    return out


def _consume_markers(body: str, marks: dict) -> str:
    """Strip the leading and trailing marker runs off *body*, filling *marks*.

    **Both ends, because the pass writes both.** The prompt specifies the leading form,
    and reading only that position was not a partial loss but a total one: the markers
    stayed in the text, so subject, class and entities all came back empty and the record
    carried marker syntax as prose. Measured on the live corpus (2026-08-09, 15 chats /
    267 facts): 70 facts — 26% — parsed that way, and **not one line anywhere was
    genuinely unmarked**, so every loss was recoverable. The placement is chosen per
    GENERATION, not per line: 4 of the 15 chats emitted trailing markers on every single
    line and 10 on none, which is why the damage arrived as whole unusable protocols
    rather than as evenly-thinned quality.

    The MIDDLE of a line is deliberately not scanned. Only the two run positions are
    observed, and a global sweep would strip a legitimate mid-sentence parenthetical —
    silently rewriting a fact's text, which is the one thing an immutable record must not
    do. Restricting consumption to :data:`_MARKER_FIELDS` guards the same risk at the
    ends, where a sentence can plausibly finish on one.

    Precedence: the specified leading position wins; a trailing marker only fills a field
    the leading run left empty. Within a run the rightmost wins, which is the behaviour
    the leading loop always had.
    """
    while True:
        m = _MARKER_LEAD_RE.match(body)
        if not m:
            break
        field = _MARKER_FIELDS.get(m.group(1).lower())
        if field is None:
            break
        marks[field] = m.group(2).strip()
        body = body[m.end():]

    while True:
        m = _MARKER_TAIL_RE.search(body)
        if not m:
            break
        field = _MARKER_FIELDS.get(m.group(1).lower())
        if field is None:
            break
        if not marks.get(field):
            marks[field] = m.group(2).strip()
        body = body[:m.start()]

    return body


def parse_facts(raw: str, *, truncated: bool = False,
                subject_fn: Optional[Callable[[Optional[str]], str]] = None) -> list[dict]:
    """Parse a facts-pass generation into protocol records. Lenient by construction.

    *subject_fn* resolves the `(about:)` label to a subject key, defaulting to this
    module's person-shaped :func:`normalize_subject`. It is a parameter because the OTHER
    protocol lane — `core.til_facts`, reading articles and news digests — shares every part
    of this parser except that one step: its subjects are countries, organisations and
    things rather than the handful of people the box knows, and reducing those to a person
    key would file most of a digest under a first token ("New York" → "new"). One parser,
    because a second copy is a second thing to fix when a placement or a marker changes;
    one parameter, because the namespace a subject lives in is genuinely per-lane.

    Reads ``[fact]``-tagged lines, each optionally carrying ``(about: NAME)`` /
    ``(class: standing)`` markers, in either order and at either end of the statement::

        [fact] (about: Artemy) (class: standing) Wears band t-shirts to the office.
        [fact] Wears band t-shirts to the office. (about: Artemy) (class: standing)

    Both placements are read because the pass produces both — see :func:`_consume_markers`
    for the measurement, and for why the middle of a line is left alone.

    A line missing a marker still yields a record — the text is the part worth keeping
    and an unlabelled fact is recoverable later, where a dropped one is not. Everything
    outside ``[fact]`` lines (the pass's own preamble, a trailing summary) is ignored
    rather than parsed, so a chatty generation degrades to fewer facts instead of to
    garbage records.

    **Only the ANSWER region is read** (``reasoning_text.answer_after_think``). The pass
    thinks, and its thinking routinely DRAFTS ``[fact]``-prefixed lines before settling —
    a draft line is tag-identical to a final one, so a line scan over the raw completion
    stored the deliberation beside the conclusion: reworded draft/final near-duplicates,
    draft lines consuming the ``MAX_FACTS`` cap ahead of the real answer, and — when the
    close marker landed mid-line — records whose text carries a literal ``</think>``
    fusion (observed on 2 of 29 TIL and 4 of 84 chat protocols, two of which the graph
    tree had folded in and served to live chat). One caveat is the caller's, not this
    function's: a generation cut *inside* an unclosed reasoning channel can normalize to
    untagged prose carrying no marker at all (the gemma-4 shape), which no text
    inspection can detect — the caller must check
    ``reasoning_text.truncated_before_answer(raw, generate_fn.last_truncated)`` and
    refuse the parse entirely when it fires.

    *truncated* — the generation hit its token cap, so the final line may be cut
    mid-statement. It is dropped, matching ``parse_consolidation``'s contract: half a
    fact is worse than no fact, and this one cannot be repaired by a later pass because
    the record is immutable.
    """
    lines = answer_after_think(str(raw or "")).splitlines()
    tagged: list[str] = []
    for line in lines:
        if not _FACT_LINE_RE.match(line):
            continue
        tagged.append(_FACT_LINE_RE.sub("", line).strip())

    if truncated and tagged:
        tagged.pop()

    out: list[dict] = []
    seen: set[str] = set()
    for body in tagged:
        # However many markers this line carries, in whatever order, at either end.
        marks: dict = {}
        body = _consume_markers(body, marks)
        about_raw = marks.get("about", "")
        cls_raw = marks.get("class", "")
        entities_raw = marks.get("entities", "")
        when_raw = marks.get("when", "")

        entities = normalize_entities(entities_raw)
        # The prompt's format line, echoed as data. Dropped rather than stored: a record
        # whose subject is the literal placeholder describes nothing that was said.
        if about_raw.strip() in _PLACEHOLDER_ABOUT or tuple(entities) == _PLACEHOLDER_ENTITIES:
            continue

        # Clipped only AFTER the markers are off. Latent rather than observed (no live
        # record ever reached MAX_FACT_CHARS), but under the old order a long enough
        # trailing-marked line would have had its marker run cut mid-token and left in the
        # text as an unmatchable fragment — unreadable then and unrecoverable after.
        text = _WS_RE.sub(" ", body).strip().strip("-–—•* ").strip()
        if not text:
            continue
        text = text[:MAX_FACT_CHARS].strip()

        # Exact-duplicate guard only. Paraphrase collapsing is the live store's job
        # (`fact_dedup`), and doing it here would make the record a judgement about what
        # is the same fact rather than a record of what was said.
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "subject": (subject_fn or normalize_subject)(about_raw),
            "subject_raw": _WS_RE.sub(" ", about_raw).strip(),
            "text": text,
            "fact_class": normalize_class(cls_raw),
            # The other nodes this fact touches, and (for an event) when it happened.
            # Both absent on a line that carried no marker — an omitted field is
            # information, the same contract `fact_class` keeps with UNSPECIFIED.
            "entities": entities,
            "when": _WS_RE.sub(" ", when_raw).strip()[:MAX_WHEN_CHARS],
        })
        if len(out) >= MAX_FACTS:
            break
    return out


def class_counts(facts: list[dict]) -> dict:
    """``{fact_class: n}`` over *facts* — the one-line shape of a protocol, for events."""
    counts: dict = {}
    for f in facts or []:
        c = (f or {}).get("fact_class") or UNSPECIFIED
        counts[c] = counts.get(c, 0) + 1
    return counts


def is_stale_record(rec: Optional[dict]) -> bool:
    """True when TODAY's parser would read more off this record than it carries.

    The staleness test is the parser itself: run the marker consumption over the record's
    stored ``text`` and ask whether anything comes off. A record written by a parser that
    understood its line has clean text — the markers are already in their fields — so
    anything still extractable is a field the writing parser could not see.

    **Why this rather than a ``SCHEMA_VERSION`` bump.** A version bump answers "was this
    written by older code", which is not the question worth asking here: it condemns every
    file the change touched, and regenerating one costs a GPU generation, so on a corpus
    where a parser fix affected a quarter of the records it would pay for the other three
    quarters to be rewritten identically. This asks the exact question instead — "does the
    stored record disagree with what we can read now" — and flags only the records that do.

    It is also self-maintaining, which a hardcoded signature for one known bug would not
    be. Nothing here describes the trailing-marker defect; it describes the *relationship*
    between a stored record and the current reader. The next time :func:`parse_facts` learns
    to read something it previously dropped, the records that lost that something are
    flagged by this function with no new code and no version to remember to bump — and the
    detector cannot drift from the fixer, because it IS the fixer, asked a question.

    Bounded in the safe direction. It cannot flag a record whose markers are genuinely
    absent (nothing to extract), and it cannot flag one whose text merely *contains*
    marker-shaped prose mid-sentence, since :func:`_consume_markers` reads only the two run
    positions. What it misses is a marker no regex can recover — one cut mid-token before
    its closing paren. That is moot in practice for the reason the damage is coarse to
    begin with: a generation that placed its markers where the old reader could not see
    them did so on *every* line of the pass, so such a record is carried to repair by its
    siblings flagging the document.

    A second probe, same philosophy: a record whose stored text carries a **reasoning
    marker** (``</think>`` and kin) was written by a reader that scanned the raw
    completion — today's reader parses only the answer region and could not have written
    it, so the document is stale by the same "does the store disagree with the current
    reader" question. This is what re-derives the six protocols observed carrying
    draft/final ``</think>`` fusions. What it cannot flag — bounded in the same safe
    direction as the marker probe — is a draft fact that leaked WITHOUT a marker (the
    silent duplication case): such a record is textually indistinguishable from a real
    one, and most travel in documents a marker sibling flags anyway.
    """
    text = str((rec or {}).get("text") or "")
    low = text.lower()
    if any(marker in low for marker in REASONING_LEAK_MARKERS):
        return True
    marks: dict = {}
    _consume_markers(text, marks)
    return bool(marks)


def needs_reparse(doc: Optional[dict]) -> bool:
    """True when any record in a ``<stem>.facts.json`` document is stale.

    Document-level because the pass rewrites the file whole: there is no repairing one
    record in place, and no reason to want to — the fix is one cheap generation that
    re-derives the protocol under the current reader.
    """
    return any(is_stale_record(r) for r in ((doc or {}).get("facts") or []))


# --------------------------------------------------------------------------- #
# GPU-free self-test                                                           #
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    def check(label, got, want):
        status = "ok " if got == want else "FAIL"
        print(f"{status}  {label}: got {got!r}, want {want!r}")
        assert got == want, label

    raw = (
        "Here is what the conversation established:\n"
        "[fact] (about: Artemy) (class: standing) Wears heavy metal band t-shirts "
        "when going to the office.\n"
        "- [fact] (class: standing) (about: Artemy voikhansky) Drinks Reviseur XO and "
        "smokes Camel at night on the balcony.\n"
        "[fact] (about: Artemy) (class: stated) Trump is deliberately creating "
        "uncertainty around the Hormuz crisis to be able to strike.\n"
        "[fact] The server box has 24 GB of VRAM.\n"
        "[fact] (about: the user) (class: nonsense) Something unlabelled.\n"
        "Some trailing prose that is not a fact.\n"
    )
    facts = parse_facts(raw)
    check("count", len(facts), 5)
    check("subject normalized", facts[0]["subject"], "artemy")
    check("full name → first token", facts[1]["subject"], "artemy")
    check("markers in either order", facts[1]["fact_class"], "standing")
    check("stance is `stated`", facts[2]["fact_class"], "stated")
    check("no markers ⇒ no subject", facts[3]["subject"], "")
    check("no markers ⇒ unspecified", facts[3]["fact_class"], UNSPECIFIED)
    check("generic referent names nobody", facts[4]["subject"], "")
    check("unknown class ⇒ unspecified", facts[4]["fact_class"], UNSPECIFIED)
    check("bullet prefix stripped",
          facts[1]["text"].startswith("Drinks Reviseur XO"), True)

    # Truncation drops the half-written tail.
    check("truncated drops last", len(parse_facts(raw, truncated=True)), 4)

    # Exact duplicates collapse; paraphrases deliberately do not.
    dup = ("[fact] (about: A) X happened.\n"
           "[fact] (about: A) x happened.\n"
           "[fact] (about: A) X took place.\n")
    check("exact dup collapsed", len(parse_facts(dup)), 2)

    check("no fact lines ⇒ empty", parse_facts("nothing tagged here"), [])
    check("empty input ⇒ empty", parse_facts(""), [])
    check("class alias", normalize_class("Opinion"), "stated")
    check("class passthrough", normalize_class("event"), "event")
    check("class garbage", normalize_class(""), UNSPECIFIED)
    # The TIL lane's fourth class. Parsed here because both lanes share this parser; the
    # chat prompt never asks for it, and `fold` leaves `(chat, depicted)` unmapped.
    check("fiction is its own class", normalize_class("depicted"), "depicted")
    check("...with the near-misses a generation produces",
          [normalize_class(s) for s in ("fiction", "in_universe", "Plot")],
          ["depicted", "depicted", "depicted"])
    check("counts", class_counts(facts),
          {"standing": 2, "stated": 1, UNSPECIFIED: 2})

    # A runaway generation is capped rather than stored whole.
    many = "".join(f"[fact] (about: A) fact number {i}.\n" for i in range(MAX_FACTS + 20))
    check("runaway capped", len(parse_facts(many)), MAX_FACTS)

    # Only the ANSWER region is parsed. The pass drafts [fact]-prefixed lines inside its
    # <think>, and a draft is tag-identical to a final line — so a raw-completion scan
    # stored deliberation as protocol (observed live: draft/final near-duplicates, and
    # records fused through a mid-line `</think>`).
    drafted = ("<think>Let me draft.\n"
               "[fact] (about: A) draft version of the fact.\n"
               "Actually, sharper:</think>\n"
               "[fact] (about: A) final version of the fact.\n")
    got = parse_facts(drafted)
    check("draft lines inside <think> are not records", len(got), 1)
    check("...and the answer's line is the one kept",
          got[0]["text"], "final version of the fact.")
    # The observed corruption shape: the close marker lands mid-line, fusing the last
    # draft with the first answer line. Splitting on the marker (not the line) recovers
    # the answer line clean.
    fused = ("<think>[fact] (about: A) draft.</think>[fact] (about: B) real fact.\n")
    got = parse_facts(fused)
    check("a mid-line </think> fusion yields the answer's record only", len(got), 1)
    check("...with clean text", got[0]["text"], "real fact.")
    check("...and no marker residue", "think" in got[0]["text"].lower(), False)
    # An unclosed <think> (a truncated thought) yields nothing — its drafts must not
    # masquerade as a protocol. NB the gemma-4 truncated channel can normalize to
    # UNTAGGED prose, which no parse can detect: that case is the caller's, via
    # `reasoning_text.truncated_before_answer`.
    check("an unclosed <think> yields no records",
          parse_facts("[fact] (about: A) early.\n<think>[fact] (about: B) draft"),
          parse_facts("[fact] (about: A) early.\n"))

    # A stored record carrying a reasoning marker is stale: it was written by the
    # raw-completion reader, and today's answer-region reader could not produce it.
    check("a </think> fusion in a stored record flags the document",
          needs_reparse({"facts": [{"text": "x.</think>[fact] (about: A) y."}]}), True)
    check("a clean stored record does not",
          needs_reparse({"facts": [{"text": "Lies at the mouth of the Volga."}]}), False)

    # Facts about Ava herself resolve to the reserved subject rather than to "nobody".
    # Every label below normalizes to "" through `normalize_person` — which is what filed
    # her own facts as being about no one, most of a reversed session's content included.
    selfies = ("[fact] (about: self) (class: standing) Keeps a wander log.\n"
               "[fact] (about: me) (class: stated) Thinks the block needs a subject cap.\n"
               "[fact] (about: I) (class: event) Started this conversation unprompted.\n"
               "[fact] (about: Myself) (class: standing) Reads current events on her own.")
    got = parse_facts(selfies)
    check("first-person labels resolve to the reserved subject",
          sorted({f["subject"] for f in got}), [SELF_SUBJECT])
    check("...and the raw label is preserved for audit",
          [f["subject_raw"] for f in got], ["self", "me", "I", "Myself"])
    # "the user" is NOT mapped: it normally means the human, and guessing would misfile a
    # real person's fact. A pass reaching for it while meaning herself is a prompt problem.
    check("a generic referent still means nobody",
          parse_facts("[fact] (about: the user) x.")[0]["subject"], "")
    check("a named person is untouched",
          parse_facts("[fact] (about: Artemy) x.")[0]["subject"], "artemy")
    # The leading underscore is what keeps the reserved key clear of the people namespace:
    # `normalize_person` strips surrounding punctuation but not underscores, so a name has
    # to *start with one* to collide — which no name a person is written under does. Note
    # this is a namespace guarantee, not an impossibility: a literal "_self …" would
    # collide, and nothing rejects it.
    from core.reflection_writer import normalize_person as _np
    check("a name that merely looks self-referring does not collide",
          [_np(n) for n in ("Self", "Ava", "Artemy Voikhansky")], ["self", "ava", "artemy"])
    # Same reserved slug the self-portrait already uses, for the same reason.
    from core.user_digest import RESERVED_SLUGS
    check("reuses the existing reserved slug", SELF_SUBJECT in RESERVED_SLUGS, True)

    # ---- session-aware participant fold (see participants_note / make_subject_fn) -- #
    # The pass writes whatever participant label is salient — the recorded speaker, the
    # name in its weights, the conversation's language — so one person fragments across
    # keys. The caller knows the recorded spelling; these fold the label back onto it.
    sess = {"user": "artemyvo", "exchanges": [
        {"speaker": "(initiative)", "user_prompt": "x", "assistant_response": "y"},
        {"speaker": "artemyvo", "user_prompt": "x", "assistant_response": "y"},
        {"speaker": "Artemy Voikhansky", "user_prompt": "x", "assistant_response": "y"},
    ]}
    check("participants: one person, first spelling wins",
          session_participants(sess), ["artemyvo"])
    check("a stage direction names nobody",
          session_participants({"user": "", "exchanges": [{"speaker": "(initiative)"}]}),
          [])
    check("a generic speaker names nobody",
          session_participants({"user": "the user", "exchanges": []}), [])
    two = {"user": "Alex", "exchanges": [{"speaker": "Maria"}]}
    check("distinct people both listed", session_participants(two), ["Alex", "Maria"])

    note = participants_note(sess)
    check("note names the recorded spelling",
          ('"artemyvo"' in note, "(about: artemyvo)" in note), (True, True))
    check("note restates the self rule", "(about: self)" in note, True)
    check("no participant ⇒ no note", participants_note({"user": ""}), "")

    fn = make_subject_fn(sess)
    check("recorded spelling resolves to itself", fn("artemyvo"), "artemyvo")
    check("case variant folds", fn("Artemyvo"), "artemyvo")
    check("inflection-close variant folds", fn("Artemy"), "artemyvo")
    check("cross-script variant is honest residue (alias table's job)",
          fn("Артемий"), "артемий")
    check("self carve-out unaffected", fn("me"), SELF_SUBJECT)
    check("a generic referent still means nobody", fn("the user"), "")
    check("a third party is untouched", fn("Maria"), "maria")
    check("a near-name third party cannot fold (prefix < 6)", fn("Artemis"), "artemis")
    fn2 = make_subject_fn(two)
    check("short names demand exactness", fn2("Alexey"), "alexey")
    check("each participant folds to its own key",
          (fn2("alex"), fn2("MARIA")), ("alex", "maria"))
    folded = parse_facts("[fact] (about: Artemy) (class: standing) x.",
                         subject_fn=make_subject_fn(sess))
    check("parse_facts takes the session resolver", folded[0]["subject"], "artemyvo")
    check("...and the raw label is preserved for audit",
          folded[0]["subject_raw"], "Artemy")

    # entities — the graph's OTHER nodes, kept as mention strings (see normalize_entities).
    graphy = parse_facts(
        "[fact] (about: Artemy) (class: standing) (entities: sister, Haifa) "
        "His sister lives in Haifa.\n"
        "[fact] (about: Artemy) (class: event) (when: 2026-06-16) Shipped the change.\n"
        "[fact] (about: Artemy) (class: standing) Nothing else is named here.")
    check("entities parsed and split", graphy[0]["entities"], ["sister", "Haifa"])
    check("surface form is preserved for the resolver",
          normalize_entities("new york, HAIFA"), ["new york", "HAIFA"])
    check("deduped case-insensitively, first form wins",
          normalize_entities("Haifa, haifa, HAIFA"), ["Haifa"])
    check("decoration stripped", normalize_entities('"Haifa", #tel-aviv.'),
          ["Haifa", "tel-aviv"])
    check("runaway entity list capped",
          len(normalize_entities(",".join(f"e{i}" for i in range(MAX_ENTITIES + 5)))),
          MAX_ENTITIES)
    check("a clause is not an entity", normalize_entities("x" * (MAX_ENTITY_CHARS + 1)), [])
    check("no marker ⇒ empty, not absent", graphy[2]["entities"], [])

    # when — only meaningful on an event, but stored as given rather than second-guessed.
    check("when parsed", graphy[1]["when"], "2026-06-16")
    check("an unresolvable phrase is kept verbatim",
          parse_facts("[fact] (when: last spring) (class: event) x.")[0]["when"],
          "last spring")
    check("no when marker ⇒ empty", graphy[0]["when"], "")

    # Markers stay order-free and optional, as they were before these two were added.
    jumbled = parse_facts(
        "[fact] (entities: Haifa) (when: 2026-06-16) (class: event) (about: Artemy) x.")[0]
    check("markers in any order", (jumbled["subject"], jumbled["fact_class"],
                                   jumbled["entities"], jumbled["when"]),
          ("artemy", "event", ["Haifa"], "2026-06-16"))

    # ---- marker PLACEMENT (see `_consume_markers`) --------------------------------- #
    # The pass writes its markers at either end, choosing per generation. Reading only
    # the leading position lost 26% of the live corpus outright — subject, class and
    # entities empty, marker syntax left in the text — so both ends are consumed.
    tail = parse_facts(
        "[fact] Drinks Reviseur XO on the balcony. (about: Artemy) (class: standing) "
        "(entities: Reviseur XO, balcony)")[0]
    check("trailing markers are read", (tail["subject"], tail["fact_class"]),
          ("artemy", "standing"))
    check("...with their entities", tail["entities"], ["Reviseur XO", "balcony"])
    check("...and stripped out of the text", tail["text"],
          "Drinks Reviseur XO on the balcony.")
    check("a trailing marker leaves no syntax behind", "(about:" in tail["text"], False)

    both = parse_facts("[fact] (about: Artemy) x happened. (class: event)")[0]
    check("markers at both ends compose", (both["subject"], both["fact_class"],
                                           both["text"]), ("artemy", "event", "x happened."))
    # The specified (leading) position is authoritative where the two disagree.
    conflict = parse_facts("[fact] (about: Artemy) x. (about: someone-else)")[0]
    check("leading wins over trailing", conflict["subject"], "artemy")

    # A sentence may legitimately END in a parenthetical. Consumption is restricted to the
    # known marker keys, so an unknown one stays text rather than being silently eaten.
    kept = parse_facts("[fact] (about: Artemy) Moved to Anthropic (formerly: OpenAI)")[0]
    check("an unknown trailing key is preserved as text",
          kept["text"], "Moved to Anthropic (formerly: OpenAI)")
    check("...and the real marker still parsed", kept["subject"], "artemy")
    # The pass terminates the LINE after its marker run; the statement keeps its own stop.
    dotted = parse_facts(
        "[fact] Detained in Taiwan. (class: event) (entities: Nvidia, Taiwan).")[0]
    check("a stray stop after the marker run is consumed",
          (dotted["fact_class"], dotted["entities"], dotted["text"]),
          ("event", ["Nvidia", "Taiwan"], "Detained in Taiwan."))

    # Only the two run positions are read; the middle of a line is never rewritten.
    mid = parse_facts("[fact] (about: A) He said (class: joke) and left.")[0]
    check("a mid-sentence parenthetical is left alone",
          mid["text"], "He said (class: joke) and left.")

    # The prompt's own format line, echoed back as data.
    echo = parse_facts(
        "[fact] (about: NAME) (class: standing) (entities: A, B) the fact, in one plain "
        "sentence.\n"
        "[fact] (about: Artemy) (class: standing) A real one.")
    check("the prompt's format line is not a fact", len(echo), 1)
    check("...and the real line survives it", echo[0]["subject"], "artemy")
    check("the placeholder rule is case-sensitive and narrow",
          parse_facts("[fact] (about: Name) x.")[0]["subject"], "name")

    # A long trailing-marked line is clipped only after the markers come off, so the
    # record can never keep a half-marker fragment (`… (clas`) the way the old reader did.
    long_tail = parse_facts(
        "[fact] " + "y" * (MAX_FACT_CHARS + 50) + " (about: Artemy) (class: stated)")[0]
    check("clip happens after the markers are stripped",
          (long_tail["subject"], len(long_tail["text"])), ("artemy", MAX_FACT_CHARS))

    # ---- staleness (see `is_stale_record`) ----------------------------------------- #
    # The probe is the parser asked a question: can today's reader take anything off a
    # stored record's text? If so, the parser that wrote it could not see that field.
    fresh = parse_facts("[fact] (about: Artemy) (class: standing) Wears band t-shirts.")[0]
    check("a record today's parser wrote is not stale", is_stale_record(fresh), False)
    # What a leading-only parser produced from a trailing-marked line: fields empty, the
    # markers still sitting in the text.
    damaged = {"subject": "", "subject_raw": "", "fact_class": UNSPECIFIED, "entities": [],
               "when": "", "text": "Wears band t-shirts. (about: Artemy) (class: standing)"}
    check("a record written by a reader that missed its markers is stale",
          is_stale_record(damaged), True)
    check("...and re-parsing its text recovers the fields",
          (parse_facts("[fact] " + damaged["text"])[0]["subject"],
           parse_facts("[fact] " + damaged["text"])[0]["fact_class"]),
          ("artemy", "standing"))
    # Bounded in the safe direction: an unmarked fact is not stale (nothing to recover),
    # and marker-shaped prose mid-sentence is not stale (only the run positions are read).
    check("a genuinely unmarked record is not stale",
          is_stale_record({"text": "The server box has 24 GB of VRAM."}), False)
    check("a mid-sentence parenthetical is not staleness",
          is_stale_record({"text": "He said (class: joke) and left."}), False)
    check("an unknown trailing key is not staleness",
          is_stale_record({"text": "Moved to Anthropic (formerly: OpenAI)"}), False)
    check("junk in ⇒ not stale", (is_stale_record(None), is_stale_record({})), (False, False))

    # Document level, because the pass rewrites the file whole.
    check("a document is stale if any record is",
          needs_reparse({"facts": [fresh, damaged]}), True)
    check("...and clean when none is", needs_reparse({"facts": [fresh]}), False)
    check("an empty or absent document is not stale",
          (needs_reparse({"facts": []}), needs_reparse(None)), (False, False))

    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
