"""The node id space: typed ids, formatting normalization, and the alias table.

This module is where the two lanes' **incompatible subject namespaces** are reconciled, and
it is worth stating what that incompatibility actually is, because it is not a bug in either
producer — each is right for its lane:

* the chat lane's subject is a **person key** (``normalize_person``: lowercased, reduced to a
  first token). Measured on this box: **7 distinct values over 349 facts** — ``_self`` 234,
  ``artemyvo`` 96, ``""`` 9, plus ``nobody``, ``claude``, ``name``, ``nvidia``. The last
  three show the namespace is not even internally clean: ``normalize_person`` takes anything
  not on its generic-referent list at face value, so a company lands in it as a person.
* the TIL lane's subject is an **entity mention**, surface form preserved — because a person
  key would file "New York" under ``new`` and "United States" under ``united``, and file
  everything unrecognised under ``""``, which on a world-facts lane is the *normal* case and
  therefore invisible.

A flat id space merges these, and merging them is how ``nvidia`` becomes a person. So ids
are **typed**, and the type is part of the id.

**The residue is typed ``unknown:``, not guessed.** Stage 1 has no classifier, and only two
signals are free and reliable: a mention matching a known person key is a person, and a
mention that appears as a TIL subject is an entity. Everything else — 146 distinct chat
mentions on this box, mixing proper nouns ("Israel", "Project Ava") with abstractions
("subjectivity" ×13, "identity", "will") — goes to ``unknown:`` and is *counted*. That
follows ``chat_facts.UNSPECIFIED``'s rule exactly: for a record whose purpose is later
machine processing, unlabelled is information and a wrong label is contamination. It also
makes the resolver's gap the headline number of a stage-1 build, which is the point of
building stage 1 at all.

**A casing heuristic was considered and declined.** "Israel" vs "subjectivity" separates
cleanly by capitalization in the measured English data — and the corpus is mixed
Russian/English, where it does not (Russian does not capitalize common nouns the same way,
German capitalizes all of them). ``exchange_anchor.normalize_tag`` documents this exact
limit for tags. A rule that silently misfires on half the corpus is worse than an honest
``unknown:``.

``topic:`` is **reserved and unpopulated** until a classifier exists (FACTS_TREE.md §12).

GPU-free self-test: ``python -m graph.nodes``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

# Node types. `person` and `entity` are assigned structurally; `unknown` is the honest
# residue; `topic` is reserved for the classifier tier and is never produced by stage 1.
TYPE_PERSON = "person"
TYPE_ENTITY = "entity"
TYPE_TOPIC = "topic"
TYPE_UNKNOWN = "unknown"
NODE_TYPES = (TYPE_PERSON, TYPE_ENTITY, TYPE_TOPIC, TYPE_UNKNOWN)

# Ava's own node. Reserved upstream already (`chat_facts.SELF_SUBJECT`,
# `user_digest.RESERVED_SLUGS`) — inherited here rather than re-invented, so the tree, the
# protocols and the portrait dir all agree on which key means her.
SELF_KEY = "_self"
SELF_NODE = f"{TYPE_PERSON}:{SELF_KEY}"

ALIASES_FILE = "aliases.json"

_WS_RE = re.compile(r"\s+")
# Same decoration set the producers strip, so a mention normalizes identically on both sides.
_STRIP = " \t\"'`«»„“”‘’()[]{}<>#*•-–—.,;:!?"

# Subject/mention strings that name nobody. The union of both lanes' non-answers: the chat
# lane's `normalize_person` already maps its generics to "", but a *mention* is not passed
# through it, so the same words arrive raw in the `entities` slot.
NO_NODE = frozenset({
    "", "-", "—", "n/a", "na", "none", "nobody", "no one", "nothing", "unknown",
    "unspecified", "general", "the world", "world", "someone", "anyone", "everyone",
    "them", "they", "it", "user", "the user",
})

# First-person labels, which name Ava. A deliberate copy of `chat_facts._SELF_REFERENTS`
# rather than an import: this package is offline and never imports the inference role (the
# same reason `reachout_gate` carries its own copy of a small predicate). Keeping the two in
# step matters, because `chat_facts.normalize_subject` already makes exactly this decision
# for the *subject* slot — and the mention slot is not passed through it, so without this the
# same word means two different things in two slots of one record.
#
# Measured on this box: `self` is the single most frequent chat-lane mention (22), against
# `_self` as the subject key of 234 facts. They are one node.
#
# `the user` is deliberately NOT here (it normally means the human, and guessing would
# misfile a real person's fact) — it is in NO_NODE instead, where the chat lane's own
# normalizer already sends it.
SELF_REFERENTS = frozenset({"_self", "self", "myself", "me", "i"})

MAX_LABEL_CHARS = 60


def clean_surface(raw: Optional[str]) -> str:
    """Formatting normalization only, surface form preserved.

    The same treatment ``chat_facts.normalize_entities`` gives a mention: collapse
    whitespace, strip decoration, cap length — and deliberately **no** case folding, because
    a later resolver matches on what was written and lowercasing "Haifa" discards a signal
    for nothing.
    """
    s = _WS_RE.sub(" ", str(raw or "")).strip().strip(_STRIP).strip()
    return s[:MAX_LABEL_CHARS].strip()


def match_key(raw: Optional[str]) -> str:
    """The key two surface forms are compared on: cleaned, casefolded, NFKC.

    Unicode-normalized because the corpus is mixed-script and the same name can arrive
    composed or decomposed; that difference is never meaningful and always invisible.
    """
    s = clean_surface(raw)
    if not s:
        return ""
    return unicodedata.normalize("NFKC", s).casefold()


def is_no_node(raw: Optional[str]) -> bool:
    """Does this name nobody at all? (A general observation, not a claim about a thing.)"""
    return match_key(raw) in NO_NODE


def node_id(node_type: str, key: str) -> str:
    """``person:artemyvo``. The type is part of the id, so two lanes can never collide."""
    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown node type: {node_type!r}")
    return f"{node_type}:{key}"


def parse_node_id(ident: str) -> tuple:
    """``person:artemyvo`` → ``("person", "artemyvo")``. Splits on the FIRST colon only,
    since an entity key may itself contain one ("Portal:Current events")."""
    s = str(ident or "")
    t, _, k = s.partition(":")
    return (t, k) if t in NODE_TYPES else ("", s)


def is_self(ident: str) -> bool:
    return ident == SELF_NODE


# -- the alias table --------------------------------------------------------- #

def load_aliases(path: Optional[Path]) -> dict:
    """``{node_id: [surface, ...]}`` → a lookup ``{match_key: node_id}``.

    Hand-edited, and the only non-derived file the tree owns — the operator's escape hatch
    when the structural rules are wrong, and the manual answer to cross-lingual merging
    until there is enough evidence to measure an automatic one (FACTS_TREE.md §12).

    An alias is authoritative: it wins over every structural rule below, including the
    lane's own subject namespace. That is the point of having one.
    """
    if not path or not Path(path).is_file():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for ident, surfaces in raw.items():
        t, _ = parse_node_id(str(ident))
        if not t:
            continue                       # an untyped id is a typo, not a node
        if isinstance(surfaces, str):
            surfaces = [surfaces]
        for s in (surfaces or []):
            k = match_key(s)
            if k:
                out[k] = str(ident)
        # An id's own key resolves to itself, so listing it as its own alias is unnecessary.
        _, key = parse_node_id(str(ident))
        k = match_key(key)
        if k:
            out.setdefault(k, str(ident))
    return out


# -- resolution (stage 1: exact + alias + structural type) ------------------- #

class Resolver:
    """Maps subjects and mentions onto typed node ids.

    Stage 1 is tiers 1–2 of the four in FACTS_TREE.md §5b: **exact** (casefolded surface
    match) and **alias** (the hand-edited table plus the person keys the box already knows
    for free). Tiers 3–4 — embedding-blocked candidates and LLM adjudication — plug in later
    behind ``resolve_mention`` without changing anything above it.

    Precision over recall, deliberately: a missed merge leaves two nodes a later rebuild can
    join, while a false merge attributes one person's position to another and is invisible
    once the surface forms are gone. Every node therefore keeps its surface forms.
    """

    def __init__(self, aliases: Optional[dict] = None):
        self.aliases = dict(aliases or {})
        # Structural knowledge, gathered from the corpus itself in `learn`.
        self.person_keys: set = {SELF_KEY}
        self.entity_keys: dict = {}        # match_key -> canonical surface
        self._labels: dict = {}            # node_id -> {surface: count}

    # -- learning ------------------------------------------------------------ #

    def learn(self, occurrences) -> None:
        """Gather the two free, reliable type signals off the corpus before resolving.

        Both are *structural* — they read what the producers already decided, rather than
        guessing from the text: a chat-lane subject went through ``normalize_person`` and so
        names a person; a TIL-lane subject is an entity mention by that lane's construction.
        """
        for o in occurrences:
            key = str(o.get("subject_key") or "")
            if not key or is_no_node(key):
                continue
            if o.get("lane") == "chat":
                self.person_keys.add(match_key(key) or key)
            else:
                mk = match_key(key)
                if mk:
                    self.entity_keys.setdefault(mk, clean_surface(key))

    # -- resolution ---------------------------------------------------------- #

    def resolve_subject(self, occurrence: dict) -> Optional[str]:
        """The node a fact is *about*. ``None`` when it is about nobody in particular.

        On the TIL lane an unowned fact is ordinary (a general observation the article
        makes), so ``None`` is a normal outcome and not a failure to record.
        """
        key = str(occurrence.get("subject_key") or "")
        raw = str(occurrence.get("subject_raw") or "") or key
        if not key or is_no_node(key):
            return None
        alias = self.aliases.get(match_key(key)) or self.aliases.get(match_key(raw))
        if alias:
            self._note_label(alias, raw or key)
            return alias
        if occurrence.get("lane") == "chat":
            mk = match_key(key)
            ident = (SELF_NODE if mk in SELF_REFERENTS
                     else node_id(TYPE_PERSON, mk or key))
        else:
            ident = node_id(TYPE_ENTITY, clean_surface(key))
        self._note_label(ident, raw or key)
        return ident

    def resolve_mention(self, mention: str) -> Optional[str]:
        """A node named in a fact's ``entities`` — one of the tree's edges.

        Order is alias → known person → known entity → ``unknown:``. The last is the honest
        residue this stage exists to measure, not a fallback that pretends to have decided.
        """
        surface = clean_surface(mention)
        if not surface or is_no_node(surface):
            return None
        mk = match_key(surface)
        alias = self.aliases.get(mk)
        if alias:
            self._note_label(alias, surface)
            return alias
        if mk in SELF_REFERENTS:
            self._note_label(SELF_NODE, surface)
            return SELF_NODE
        if mk in self.person_keys:
            ident = SELF_NODE if mk == SELF_KEY else node_id(TYPE_PERSON, mk)
        elif mk in self.entity_keys:
            ident = node_id(TYPE_ENTITY, self.entity_keys[mk])
        else:
            # Deliberately keyed on the *cleaned surface*, not the casefolded match key: an
            # unknown node is a candidate for a later tier, and that tier needs what was
            # actually written.
            ident = node_id(TYPE_UNKNOWN, surface)
        self._note_label(ident, surface)
        return ident

    # -- labels -------------------------------------------------------------- #

    def _note_label(self, ident: str, surface: str) -> None:
        s = clean_surface(surface)
        if not s:
            return
        self._labels.setdefault(ident, {})
        self._labels[ident][s] = self._labels[ident].get(s, 0) + 1

    def label_for(self, ident: str) -> str:
        """The best surface form for a node: the one most often written for it.

        Ties break on the longer form, then alphabetically — deterministic, so two builds of
        an unchanged corpus produce byte-identical output.
        """
        forms = self._labels.get(ident) or {}
        if not forms:
            _, key = parse_node_id(ident)
            return key
        return sorted(forms.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]

    def surface_forms(self, ident: str) -> list:
        """Every surface form seen for a node, most frequent first — what makes a false
        merge reviewable after the fact."""
        forms = self._labels.get(ident) or {}
        return [s for s, _ in sorted(forms.items(),
                                     key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))]


# -- self-test --------------------------------------------------------------- #

def _selftest() -> None:
    import tempfile

    assert node_id(TYPE_PERSON, "artemyvo") == "person:artemyvo"
    assert parse_node_id("person:artemyvo") == ("person", "artemyvo")
    # An entity key may contain a colon; only the first splits.
    assert parse_node_id("entity:Portal:Current events") == ("entity", "Portal:Current events")
    assert parse_node_id("garbage") == ("", "garbage")
    assert is_self(SELF_NODE)

    assert clean_surface("  «Israel».  ") == "Israel"
    assert clean_surface(None) == ""
    assert match_key("ISRAEL") == match_key("israel") == "israel"
    assert is_no_node("nobody") and is_no_node("") and is_no_node("The User")
    assert not is_no_node("Israel")

    occ = [
        {"lane": "chat", "subject_key": "_self", "subject_raw": "self"},
        {"lane": "chat", "subject_key": "artemyvo", "subject_raw": "artemyvo"},
        {"lane": "chat", "subject_key": "nobody", "subject_raw": "nobody"},
        {"lane": "til", "subject_key": "United States Central Command",
         "subject_raw": "United States Central Command"},
    ]
    r = Resolver()
    r.learn(occ)

    assert r.resolve_subject(occ[0]) == SELF_NODE
    assert r.resolve_subject(occ[1]) == "person:artemyvo"
    assert r.resolve_subject(occ[2]) is None                 # names nobody
    assert r.resolve_subject(occ[3]) == "entity:United States Central Command"

    # Mentions: known person / known entity / honest residue.
    assert r.resolve_mention("Artemyvo") == "person:artemyvo"
    # The measured P2 collision: `self` is the top chat-lane MENTION while `_self` is the
    # subject key of 234 facts. Both first-person spellings land on the one node, matching
    # the carve-out `chat_facts.normalize_subject` already makes for the subject slot.
    for form in ("self", "_self", "Myself", "me", "I"):
        assert r.resolve_mention(form) == SELF_NODE, form
    # ...but "the user" still means the human, and is never guessed onto her.
    assert r.resolve_mention("the user") is None
    assert r.resolve_mention("United States Central Command") == "entity:United States Central Command"
    assert r.resolve_mention("subjectivity") == "unknown:subjectivity"
    assert r.resolve_mention("  ") is None
    # The residue keeps what was written — the later tier needs the surface form.
    assert r.resolve_mention("Project Ava") == "unknown:Project Ava"

    # Aliases win over every structural rule, and carry the type.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / ALIASES_FILE
        p.write_text(json.dumps({
            "person:artemyvo": ["Artemy", "Artemy Voikhansky", "Артемий"],
            "entity:United States": ["USA", "US"],
        }), encoding="utf-8")
        al = load_aliases(p)
        r2 = Resolver(al)
        r2.learn(occ)
        assert r2.resolve_mention("Артемий") == "person:artemyvo"
        assert r2.resolve_mention("artemy voikhansky") == "person:artemyvo"
        assert r2.resolve_mention("USA") == "entity:United States"
        # An id's own key resolves to itself without being listed.
        assert r2.resolve_mention("United States") == "entity:United States"
        assert load_aliases(Path(td) / "missing.json") == {}

    # Label choice is frequency-ranked and deterministic.
    r3 = Resolver()
    for _ in range(3):
        r3.resolve_mention("Israel")
    r3.resolve_mention("israel")
    assert r3.label_for("unknown:Israel") == "Israel"
    assert r3.surface_forms("unknown:Israel")[0] == "Israel"
    assert r3.label_for("person:never-seen") == "never-seen"

    print("graph.nodes self-test OK")


if __name__ == "__main__":
    _selftest()
