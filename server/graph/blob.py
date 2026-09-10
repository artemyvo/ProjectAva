"""The retrieval side of the tree: what a fetch pass may choose from, and what the tree is
allowed to say back.

This is the read half of FACTS_TREE.md §10 **consumer 5** — the node-scoped retrieval
channel the design doc lists last and marks *"gated, off by default … here to be argued
about, not assumed"*. Nothing here injects anything: the module is pure, it renders a blob
and a report, and the only caller so far is the workbench module ``core.modules.fact_fetch``,
which writes nothing and returns its value. It lives beside the tree rather than in
``core/`` so that if a channel is ever built, the channel and the simulation share ONE
definition of what a node yields — the discipline ``til_facts`` already follows for its two
callers, and the one that keeps a workbench honest.

**Two rules are enforced here, in code, not in the prompt**, because a prompt cannot be
relied on for either and both are safety properties rather than quality ones:

1. ``person:_self`` yields nothing, ever (§10, *"One thing the tree must not become"*). The
   self subtree is the largest on the box and three artifacts already fold Ava's
   self-evidence; a fourth that quietly answered "who is Ava" would extend the exact
   self-reinforcement loop ``self_portrait`` was built to stay out of. A pass that picks it
   is not refused — the pick is *reported*, so an operator sees the pass tried.

2. Facet decides register, and the split is not cosmetic. ``property``/``event`` are things
   simply known and render as statements. ``position``/``report`` are records that *someone
   said* a thing, and §10 restricts the channel to the knowledge facets — so they are
   withheld from the blob and returned separately, counted. That separation is the whole
   reason the facet level exists (``fold.py``), and it is also the measurement this module
   was written to expose: on the corpus at the time of writing **88% of claims are
   ``position``**, so a knowledge-only blob is nearly empty and the rule's cost is a number
   an operator can look at rather than a thing they have to take on faith.

**The retrieval axis is the mention edge, not node ownership**, and that is forced by the
data rather than chosen. ``chat_facts`` files a fact under *who it is about*, so in a
two-person conversation every chat claim lands under one of two person nodes: on this box
3 nodes of 107 own any claim at all, while 106 carry mention edges. The topics — what a
message is actually *about* — exist only as mentions. So a pass picks topic nodes and this
module returns the claims that MENTION them, attributed to the node that owns them.

GPU-free self-test: ``python -m graph.blob`` (also run by ``python -m graph.selftest``).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional

from graph import fold as F
from graph import nodes as G
from graph import store as S

# The label the fetch pass answers under. Kept here beside the parser rather than in the
# prompt file alone, so the contract has one definition and the self-test can assert the
# shipped prompt still asks for it.
SELECTION_LABEL = "NODES"
NONE_MARKER = "NONE"

# How many nodes the catalogue offers. A pass cannot pick what it was not shown, and the
# whole index is currently ~100 lines of a few tokens each, so this is a guard against a
# corpus that grows rather than a budget that binds today. Past it the index is ordered by
# how much each node can actually yield, so what is dropped is the least useful tail.
DEFAULT_INDEX_LIMIT = 200

# Rendered blob caps. A blob is prompt real estate competing with the persona portrait, the
# user portrait and the reflection block, so it is bounded here rather than at the caller.
DEFAULT_MAX_CLAIMS = 24
DEFAULT_MAX_PER_NODE = 8

# ── the claim lane: the model selects, instead of code scoring ────────────────
#
# The node catalogue asks a pass to name a SUBJECT and then has code decide which of that
# subject's claims to show. Two measured problems with that, and one is unfixable in code:
#
#   * a person node yields their whole biography (31 knowledge claims for one person here),
#     so the pick is far coarser than the question;
#   * relevance had to be scored lexically, and **886 of 946 claims are written in English
#     while the conversations are often Russian**. No lexical matcher bridges that; the
#     honest result was a correctly-empty blob on Russian turns.
#
# Cross-lingual relevance is the thing a multilingual model does natively and a token
# overlap cannot do at all. So this lane hands the pass the candidate CLAIMS and lets it
# choose among them. The safety properties are unchanged and are kept by construction
# rather than by instruction: the candidate list is built already filtered to the knowledge
# facets with the `person:_self` subtree removed, so a forbidden claim is never *shown*, and
# the blob is still rendered from ids in code, so nothing the pass writes becomes text.
#
# What the tree is doing here is different from what the node catalogue used it for, and
# worth naming: the spine stops being a browsing index and the FACET level does the work —
# it is the only thing keeping 73% of the corpus (`position`) out of a knowledge channel.

# The label the claim-picking pass answers under. Distinct from SELECTION_LABEL so a prompt
# written for one lane cannot be parsed as the other.
CLAIM_SELECTION_LABEL = "FACTS"

# How stale a TIL claim may be — `None` (the default) means a TIL claim is never dropped by
# age, exactly like a chat-backed one. It shipped at 7 days on the reasoning that a news
# digest's "missiles were intercepted" is spent within days where a standing fact about a
# person holds next month; what that reasoning missed is that the TIL lane carries far more
# than news (an article about a place, a trope, an organisation ages like a chat fact, not
# like a headline), so an age cut keyed on the LANE was throwing away the durable material
# to bound the perishable. What the cut was really protecting is the size and mix of the
# candidate list — TIL supplies 34.3 showable claims per protocol against chat's 1.5 — and
# that job belongs to `DEFAULT_MAX_CANDIDATES` plus the chat-backed-first ordering below,
# which drops the OLDEST TIL first and never touches conversational material. Set an integer
# here (or `graph.til_max_age_days` in the config) to restore a scope; the mechanism is
# intact, only its default changed.
DEFAULT_TIL_MAX_AGE_DAYS: Optional[int] = None

# Hard ceiling on the candidate list, after the age cut. A prompt this pass prefills is paid
# before the user sees a token, so it is bounded rather than left to the corpus.
DEFAULT_MAX_CANDIDATES = 400

# The facet set the TIL **reading** lane offers: knowledge, plus `report` (what a text
# asserted) and `depiction` (what a work depicts), both carried only because `report_line`
# names the text they came from. See `claim_candidates` for why a recap may carry these
# where a chat turn may not, and why `position` is offered to neither.
#
# `depiction` is here rather than withheld because it is the material the reading lane was
# most wanted for: wandering onto a tropes article and having it meet an earlier
# conversation is the association the channel exists for, and it needs the fiction — marked
# as fiction, which is what `_SOURCE_VERB` does.
READING_FACETS = tuple(F.KNOWLEDGE_FACETS) + (F.FACET_REPORT, F.FACET_DEPICTION)

# How a carried-but-not-knowledge claim names its relation to its source. The verb IS the
# epistemics: "reported" says a text asserted this about the world and may have been wrong,
# "depicts" says the question of truth does not arise. Collapsing them to one verb is how a
# fantasy world's contents come back phrased as a stale news item.
_SOURCE_VERB = {F.FACET_REPORT: "reported", F.FACET_DEPICTION: "depicts"}

# How many claims a pass may pick. Above the node lane's 6 because the unit is finer.
MAX_PICKS = 8

_INT_RE = re.compile(r"\d+")

_ID_RE = re.compile(r"^([a-z]+):(.+)$")

# Tokens shorter than this never carry a topic. Language-agnostic and content-blind, so it
# holds for the mixed Russian/English corpus where a stopword list would not.
_MIN_TOKEN_CHARS = 3
_WORD_RE = re.compile(r"\w+", re.UNICODE)


# -- relevance --------------------------------------------------------------- #
#
# A picked node yields everything it owns or is mentioned by, which for a person is their
# whole biography — on this box `person:artemyvo` yields 31 knowledge claims, and the blob
# caps at 8. WHICH 8 was, until this existed, decided by a sort key that is constant on this
# corpus: `(-n_sources, -n_occurrences, text)` where every claim has n_sources == 1 and
# n_occurrences == 1 (`graph/DESIGN.md`: *"0 collapsed, 0 corroborated … nothing downstream
# should rank on corroboration yet, since it is currently a column of zeros"*). The key
# therefore collapsed to ASCII order on the text, and the blob was the alphabetically-first
# 8 of 31 — the observed "results seem irrelevant".
#
# So relevance has to come from the arriving message, and it has to be computed HERE rather
# than asked of a generation: the blob is rendered in code precisely so nothing can
# paraphrase a claim into something no source said, and picking which claims to show is part
# of that rendering.
#
# Scoring is lexical, which is the right size of tool for a pass that runs *before* a turn
# is answered: an embedder call would add latency to time-to-first-token for a channel that
# is off by default. Weighting is IDF over the claim corpus itself, so a word appearing in
# every claim contributes exactly zero — a stoplist derived from the data instead of one
# written per language.


def tokenize(text: str) -> list:
    """Lowercased, deduplicated word tokens. Mirrors ``exchange_anchor.query_tokens``.

    A deliberate copy rather than an import: this package never imports the inference role
    (the rule ``nodes.SELF_REFERENTS`` follows for the same reason). The *matcher* is
    injected instead — see :func:`render_blob` — because that one is genuinely hard to get
    right for an inflected language and must not be duplicated.
    """
    seen: set = set()
    out: list = []
    for m in _WORD_RE.finditer((text or "").lower()):
        tok = m.group(0)
        if len(tok) < _MIN_TOKEN_CHARS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def build_token_stats(doc: dict) -> dict:
    """Document frequency of every token across the tree's claims, for IDF weighting."""
    df: dict = {}
    claims = doc.get("claims") or {}
    for c in claims.values():
        for tok in tokenize(c.get("text") or ""):
            df[tok] = df.get(tok, 0) + 1
    return {"df": df, "n": len(claims)}


def _idf(token: str, stats: dict) -> float:
    n = max(1, int(stats.get("n") or 0))
    d = int((stats.get("df") or {}).get(token, 0))
    # Unseen in the claim corpus can only happen for a query token matched by inflection;
    # treat it as maximally rare rather than as free.
    return math.log(n + 1) if d <= 0 else math.log(n / d)


def score_claim(text: str, query_tokens: list, stats: dict,
                match_fn: Optional[Callable] = None) -> float:
    """How much of the arriving message this claim actually speaks to.

    Summed IDF of the claim's OWN tokens that match some query token — the claim's tokens,
    not the query's, so an inflected match is weighted by the word the tree actually holds.
    Zero means nothing in the message points at this claim.
    """
    if not query_tokens:
        return 0.0
    m = match_fn or (lambda a, b: a == b)
    total = 0.0
    for tok in tokenize(text):
        if any(m(tok, q) for q in query_tokens):
            total += _idf(tok, stats)
    return total


# -- the catalogue ----------------------------------------------------------- #

def selectable_nodes(doc: dict) -> list:
    """Nodes a fetch pass may pick, most-yielding first.

    A node is selectable when it can yield anything at all — it owns claims, or claims
    mention it. ``person:_self`` is excluded here rather than at render time so it is not
    offered in the first place; §10 forbids it feeding an injected artifact, and the
    cheapest way to honour that is to keep it out of the menu.
    """
    out = []
    for ident, n in (doc.get("nodes") or {}).items():
        if G.is_self(ident):
            continue
        owned = int(n.get("n_claims") or 0)
        mentions = len(n.get("mentioned_in") or [])
        if not owned and not mentions:
            continue
        out.append((owned + mentions, owned, mentions, ident, n))
    out.sort(key=lambda t: (-t[0], t[3]))
    return [{"id": t[3], "node": t[4], "owned": t[1], "mentions": t[2]} for t in out]


def node_index(doc: dict, limit: int = DEFAULT_INDEX_LIMIT) -> str:
    """The catalogue text a fetch pass chooses from: one line per selectable node.

    Surface forms are included because they are what was actually *written* — a message
    saying "субъектность" has to be able to find the node it resolved to, and the resolver
    already keeps every form it saw (``nodes.Resolver.surface_forms``).
    """
    rows = selectable_nodes(doc)
    total = len(rows)
    lines = []
    for r in rows[:limit]:
        n = r["node"]
        label = n.get("label") or r["id"]
        forms = [f for f in (n.get("surface_forms") or []) if f and f != label]
        alias = f"  (also: {', '.join(forms[:4])})" if forms else ""
        held = []
        if r["owned"]:
            held.append(f"{r['owned']} about it")
        if r["mentions"]:
            held.append(f"{r['mentions']} mention it")
        lines.append(f"{r['id']}  —  {label}{alias}  [{'; '.join(held)}]")
    if total > limit:
        lines.append(f"[... {total - limit} further nodes omitted, least-referenced first]")
    return "\n".join(lines)


# -- the pass's answer ------------------------------------------------------- #

def parse_selection(raw: str, doc: dict) -> dict:
    """Parse a fetch pass's answer into resolved node ids.

    Deliberately lenient in one direction only. A written id that exists is taken as-is; a
    bare name is run through ``store.find_nodes``, which searches surface forms, so a pass
    naming what was *said* still lands on the node it resolved to. Anything that matches
    nothing is returned in ``unknown`` rather than dropped — a pass reaching for a node the
    tree does not have is a finding about the corpus, not noise to hide.
    """
    text = str(raw or "")
    # The label may not appear at all (a pass that just lists), so fall back to the whole
    # text rather than to nothing.
    m = re.search(rf"^\s*{SELECTION_LABEL}\s*:?\s*$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        text = text[m.end():]
    else:
        m = re.search(rf"{SELECTION_LABEL}\s*:", text, re.IGNORECASE)
        if m:
            text = text[m.end():]

    picked: list = []
    unknown: list = []
    self_picked = False
    for line in text.splitlines():
        cand = line.strip().strip("-*•\t ").strip()
        cand = re.sub(r"^\d+[.)]\s*", "", cand).strip().strip("\"'`,;")
        if not cand:
            continue
        if cand.upper() == NONE_MARKER:
            break
        if len(cand) > G.MAX_LABEL_CHARS * 2:
            continue                      # a sentence, not an id — the pass drifted
        ident = _resolve(cand, doc)
        if ident is None:
            unknown.append(cand)
            continue
        if G.is_self(ident):
            # Not refused silently: §10 keeps the self subtree out of every injected
            # artifact, and an operator needs to see that the pass reached for it.
            self_picked = True
            continue
        if ident not in picked:
            picked.append(ident)
    return {"picked": picked, "unknown": unknown, "self_picked": self_picked}


def _resolve(cand: str, doc: dict) -> Optional[str]:
    nodes = doc.get("nodes") or {}
    if cand in nodes:
        return cand
    if _ID_RE.match(cand):
        # A well-formed id that does not exist: try its key as a name before giving up, so
        # a pass inventing `topic:` for something the tree typed `unknown:` still lands.
        cand = _ID_RE.match(cand).group(2)
        if cand in nodes:
            return cand
    hits = S.find_nodes(doc, cand, limit=1)
    return hits[0]["id"] if hits else None


# -- what the tree says back ------------------------------------------------- #

def claims_for_nodes(doc: dict, idents: list) -> list:
    """Every claim a picked node yields — the ones it owns AND the ones that mention it.

    The mention half is the load-bearing one (see the module docstring): topics own no
    claims on the chat lane, so ownership alone would return nothing for exactly the nodes
    a message is about. Claims owned by ``person:_self`` are dropped here, whichever way
    they were reached — §10 is about the subtree, not about the path taken to it.
    """
    claims = doc.get("claims") or {}
    seen: set = set()
    out: list = []
    self_dropped = 0
    for ident in idents:
        n = S.node(doc, ident)
        if not n:
            continue
        cids: list = []
        for cid_list in (n.get("facets") or {}).values():
            cids.extend(cid_list)
        cids.extend(n.get("mentioned_in") or [])
        for cid in cids:
            c = claims.get(cid)
            if c is None or cid in seen:
                continue
            seen.add(cid)
            if G.is_self(c.get("node") or ""):
                self_dropped += 1
                continue
            out.append(dict(c, _via=ident))
    out.sort(key=lambda c: (-int(c.get("n_sources") or 0),
                            -int(c.get("n_occurrences") or 0), c.get("text") or ""))
    return [out, self_dropped]


def _age_days(iso: str, now: str) -> Optional[int]:
    """Whole days between two ``YYYY-MM-DD`` strings, or ``None`` if either is unusable."""
    import datetime as _dt
    try:
        a = _dt.date(*(int(p) for p in str(iso).split("-")[:3]))
        b = _dt.date(*(int(p) for p in str(now).split("-")[:3]))
    except Exception:
        return None
    return (b - a).days


def _til_only(claim: dict) -> bool:
    """True when every occurrence of this claim came from the TIL lane.

    A claim corroborated from BOTH lanes is treated as chat-backed and never aged out —
    somebody said it in conversation too, which is what the age cut is protecting.
    """
    lanes = claim.get("lanes") or []
    return bool(lanes) and all(l == "til" for l in lanes)


def claim_candidates(doc: dict, *, now: str,
                     facets: Optional[tuple] = None,
                     til_max_age_days: Optional[int] = DEFAULT_TIL_MAX_AGE_DAYS,
                     max_candidates: int = DEFAULT_MAX_CANDIDATES) -> dict:
    """The claims a fetch pass may choose from, already filtered and scoped.

    Filtered **here** rather than checked after the pick, so the §10 rules hold by
    construction: a claim outside *facets* and anything under ``person:_self`` is never
    shown, and therefore cannot be picked, mis-parsed or argued into the blob.

    *facets* defaults to ``fold.KNOWLEDGE_FACETS`` — ``property`` + ``event``, the claims
    that can be stated as simply known. **The `person:_self` exclusion is not part of it and
    is never relaxed**: that rule is about the self subtree whatever the facet, and a caller
    widening the facets does not get to widen that.

    A caller may add ``FACET_REPORT`` (see `READING_FACETS`), and exactly one does: the TIL
    **reading** lane. The asymmetry is the same one that governs the rest of this package.
    `report` is withheld from a chat turn because a pass told *"things you know"* would state
    a news assertion as fact **to a person** — the failure the facet level exists to prevent.
    A recap asserts nothing to anyone; it is what stayed with someone from reading, and *"a
    digest of 2026-07-28 reported that a delegation was in Cairo"* is the honest form of
    precisely the association a reading pass is for. So the epistemics move from *withhold*
    to *attribute*, which is what `report_line` does and what `attribution_line`'s docstring
    has always said would be required of any channel that carried these.

    ``position`` is deliberately NOT offered to any caller. A report has a *text* behind it
    that can be named; a position is one person's view, and rendering it into a reading pass
    would put the interlocutor's politics into a recap of the world — the exact contamination
    the reading lane's wrapper prompt is written to prevent.

    The candidate list is uniform: nothing marks which entries are reports. The pass's job is
    relevance, and how a claim may be *stated* is settled downstream in code, where it cannot
    be argued with. Marking them would invite the pass to weigh source reliability, which is
    a provenance question this pass has no basis to answer and no mandate to.

    ``now`` is required rather than read off the clock, for the reason ``store.build_doc``
    takes ``built_at``: a build that reads the wall clock is neither reproducible nor
    testable, and this one decides what an operator sees.

    Ordering is chat-backed first, then TIL newest-first, because that is also the order in
    which the cap should bite — truncation drops the oldest news, never the conversational
    material, which is the scarce half (1.5 claims per chat protocol against TIL's 34.3).
    With ``til_max_age_days`` at its ``None`` default that ordering is the ONLY thing keeping
    the list's mix honest, which is what it was always doing under the cap anyway.
    """
    allowed = tuple(facets) if facets else F.KNOWLEDGE_FACETS
    offered = [c for c in (doc.get("claims") or {}).values()
               if c.get("facet") in allowed
               and not G.is_self(c.get("node") or "")]

    kept: list = []
    dropped_stale = 0
    for c in offered:
        age = _age_days(c.get("last_asserted") or "", now)
        if (til_max_age_days is not None and _til_only(c)
                and age is not None and age > til_max_age_days):
            dropped_stale += 1
            continue
        kept.append((c, age))

    # Undated sorts last within its group rather than first: an unknown date is not evidence
    # of freshness.
    kept.sort(key=lambda ca: (_til_only(ca[0]),
                              ca[1] if ca[1] is not None else 10 ** 6,
                              ca[0].get("text") or ""))
    dropped_capped = max(0, len(kept) - max_candidates)
    claims = [c for c, _ in kept[:max_candidates]]
    return {
        "claims": claims,
        "n": len(claims),
        # Kept as the STRICT knowledge count whatever `facets` was, so its meaning does not
        # shift under a caller that widened the list — `n` is how many were offered.
        "total_knowledge": sum(1 for c in (doc.get("claims") or {}).values()
                               if c.get("facet") in F.KNOWLEDGE_FACETS
                               and not G.is_self(c.get("node") or "")),
        "n_reports": sum(1 for c in claims if c.get("facet") == F.FACET_REPORT),
        "facets": list(allowed),
        "dropped_stale": dropped_stale,
        "dropped_capped": dropped_capped,
        "now": now,
        "til_max_age_days": til_max_age_days,
    }


def claim_index(doc: dict, candidates: dict) -> str:
    """The numbered catalogue the pass picks from — one claim per line.

    Numbered rather than keyed by ``claim_id``: an id is 16 hex characters that a pass has
    to copy exactly, and an ordinal is two or three digits it cannot typo into a *different*
    valid claim. The numbering is per-call and never stored, so it cannot go stale.

    Each line carries the owning node's label, because a claim's subject is what makes it
    findable ("artemyvo: ...") and the text alone often does not name it.
    """
    lines = []
    for i, c in enumerate(candidates.get("claims") or [], 1):
        n = S.node(doc, c.get("node") or "") or {}
        when = f" ({c['when']})" if c.get("when") else ""
        lines.append(f"[{i}] {n.get('label') or c.get('node')}: "
                     f"{(c.get('text') or '').strip().rstrip('.')}{when}.")
    return "\n".join(lines)


def parse_claim_selection(raw: str, candidates: dict) -> dict:
    """Parse a claim-picking answer into claims. Lenient in the same ways as its sibling.

    Reads the numbers after :data:`CLAIM_SELECTION_LABEL`, falling back to the whole text
    when the pass omitted the label — a pass that answered correctly but dropped a header
    should not read as having picked nothing. An out-of-range number is *reported*, not
    silently dropped: it means the pass invented a line, which is a finding about the pass.
    """
    claims = candidates.get("claims") or []
    text = str(raw or "")
    body = text
    if CLAIM_SELECTION_LABEL in text:
        body = text.rsplit(CLAIM_SELECTION_LABEL, 1)[1]
    if NONE_MARKER in body.upper().split():
        return {"picked": [], "numbers": [], "out_of_range": [], "none": True}

    picked: list = []
    numbers: list = []
    bad: list = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _INT_RE.search(s)
        if not m:
            continue
        k = int(m.group(0))
        if k in numbers or k in bad:
            continue
        if 1 <= k <= len(claims):
            numbers.append(k)
            picked.append(claims[k - 1])
        else:
            bad.append(k)
        if len(picked) >= MAX_PICKS:
            break
    return {"picked": picked, "numbers": numbers, "out_of_range": bad,
            "none": not picked and not bad}


def report_line(doc: dict, claim: dict, describe_source: Optional[Callable] = None) -> str:
    """A ``report`` claim rendered with the text that asserted it named in the line.

    The difference from :func:`attribution_line` is the agent. That one says *"Iran is
    reported: …"* — passive, naming the claim's SUBJECT, with the source nowhere in it. For
    a reading pass the source is the whole point: the wiki list is chosen for tone and not
    for truth, so a humour wiki and a Current-events digest asserting one sentence are not
    the same evidence, and a line that cannot say which is not attribution.

    *describe_source* is INJECTED, for the reason `render_blob` takes ``words_match``
    injected: resolving a ``(lane, ref)`` to a human phrase means reading the snippets tree,
    and ``graph/`` never imports the inference role. Absent — or returning nothing — the raw
    ref is used, which is degraded but still names a file rather than claiming nobody said
    it.
    """
    src = sources(doc, [claim])
    phrase = ""
    if src:
        lane, ref = src[0]
        if describe_source is not None:
            try:
                phrase = str(describe_source(lane, ref) or "").strip()
            except Exception:
                phrase = ""
        phrase = phrase or ref
    when = f" ({claim['when']})" if claim.get("when") else ""
    text = (claim.get("text") or "").strip().rstrip(".")
    verb = _SOURCE_VERB.get(claim.get("facet") or "", "reported")
    if not phrase:
        # No occurrence carries a source. Say so rather than dropping the hedge: an
        # unattributed report is the one thing this channel must never render as known.
        return f"something on record {verb}{when}: {text}."
    return f"{phrase} {verb}{when}: {text}."


def render_claims(doc: dict, claims: list, *, max_claims: int = DEFAULT_MAX_CLAIMS,
                  max_per_node: int = DEFAULT_MAX_PER_NODE,
                  max_reports: int = 0,
                  describe_source: Optional[Callable] = None) -> dict:
    """Render the blob for claims the pass picked directly.

    No facet gate and no self-drop here, deliberately — :func:`claim_candidates` already
    removed both, and re-checking would suggest the filter was optional. The blob text is
    still built in code from the picked records, so nothing generated becomes prose.

    ``report`` claims are rendered **separately and attributed** (:func:`report_line`), never
    folded into the node grouping above them: the grouping presents claims as things known
    about a node, which is exactly the reading a report must not get.

    They also carry their OWN budget rather than sharing ``max_claims``, and that is the
    point of the parameter. The lanes are wildly unequal — a TIL protocol yields an order of
    magnitude more claims than a chat one — so a shared budget would let a news-heavy pick
    crowd out the handful of claims about the people in the conversation, which are the
    scarce half and usually the reason the channel is worth having. ``max_reports=0`` is the
    default and makes this function byte-identical to before for every existing caller.
    """
    sourced = (F.FACET_REPORT, F.FACET_DEPICTION)
    knowledge = [c for c in claims if c.get("facet") not in sourced]
    reports = [c for c in claims if c.get("facet") in sourced]

    text, kept = _render_grouped(doc, knowledge, max_claims, max_per_node)
    shown_reports = reports[:max(0, int(max_reports))]
    if shown_reports:
        lines = [report_line(doc, c, describe_source) for c in shown_reports]
        block = "\n".join(f"  - {ln}" for ln in lines)
        # One heading for both facets, with the verb carrying the distinction per line: a
        # heading per facet would split two or three lines into two labelled sections, and
        # the line already says which it is.
        head = "From the texts named — what they said, not what is so:"
        text = f"{text}\n\n{head}\n{block}".strip()
    return {"text": text, "claims": knowledge[:kept] + shown_reports,
            "n_rendered": kept + len(shown_reports),
            "n_knowledge": kept, "n_reports": len(shown_reports),
            # A report picked by a caller that budgeted none is DROPPED, and counted — a
            # silent drop here would look identical to the pass not having picked it.
            "n_reports_dropped": len(reports) - len(shown_reports),
            "n_picked": len(claims)}


# -- provenance: which conversations the picked claims came out of ----------- #
#
# A claim is a decontextualized sentence — the conversation that produced it is the thing a
# reader would actually want next, and the tree is the only structure that can say which one
# it was. The occurrence level already carries it (`read.py` records `lane` + `source_ref`
# per line), so this is a join over data the fold already keeps, not a new field.
#
# What a consumer does with the answer belongs to the consumer: this returns the ordered
# list and applies no budget of its own, because "how many conversations may be recalled in
# one prompt" is a question about prompt real estate, which this package knows nothing about.

def sources(doc: dict, claims: list, lanes: tuple = ("chat", "til")) -> list:
    """The sources the given claims came out of, best-first, deduplicated.

    Returns ``[(lane, source_ref), …]`` — **typed**, because the two lanes' refs are not
    interchangeable and a caller has to resolve them differently: a chat ref is a session
    filename under the chats dir, a TIL ref is a ``<kind>/<stem>`` under the snippets tree.
    A flat list of strings would make that the caller's problem to re-derive by pattern,
    which is the mistake `nodes.py` documents one level up.

    Ordered by the claims' own order — a fetch pass is asked for its picks most-important
    first, so the source behind its first pick is the one worth recalling if only one can
    be. Within a single claim, its most RECENT telling wins: a claim corroborated across
    two sources is one thing said twice, and the fresher one is both the better memory and
    the one whose other material is still retrievable beside it. Where a claim was
    established in BOTH lanes, the chat occurrence is preferred at equal date — somebody
    said it in conversation, which is the more particular memory of the two.
    """
    order = {lane: i for i, lane in enumerate(("chat", "til"))}
    seen: set = set()
    out: list = []
    for c in claims or []:
        rows = [o for o in S.occurrences_for(doc, c or {})
                if (o.get("lane") or "") in lanes and (o.get("source_ref") or "").strip()]
        rows.sort(key=lambda o: (str(o.get("asserted_at") or ""),
                                 -order.get(o.get("lane") or "", 9)), reverse=True)
        for o in rows:
            key = (o.get("lane") or "", o["source_ref"].strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def chat_sources(doc: dict, claims: list) -> list:
    """Just the chat sessions, as bare refs. The narrow form of :func:`sources`."""
    return [ref for lane, ref in sources(doc, claims, lanes=("chat",))]


def _render_grouped(doc: dict, claims: list, max_claims: int, max_per_node: int) -> tuple:
    """Group claims under their owning node and render, honouring both caps."""
    by_node: dict = {}
    kept = 0
    for c in claims:
        if kept >= max_claims:
            break
        bucket = by_node.setdefault(c.get("node") or "", [])
        if len(bucket) >= max_per_node:
            continue
        bucket.append(c)
        kept += 1
    lines: list = []
    for ident, group in by_node.items():
        n = S.node(doc, ident) or {}
        lines.append(f"{n.get('label') or ident}:")
        for c in group:
            when = f" ({c['when']})" if c.get("when") else ""
            mark = " [corroborated]" if int(c.get("n_sources") or 0) > 1 else ""
            lines.append(f"  - {(c.get('text') or '').rstrip('.')}{when}.{mark}")
        lines.append("")
    return "\n".join(lines).strip(), kept


def render_blob(doc: dict, idents: list, *, query: str = "",
                words_match: Optional[Callable] = None,
                max_claims: int = DEFAULT_MAX_CLAIMS,
                max_per_node: int = DEFAULT_MAX_PER_NODE) -> dict:
    """Render the injectable blob for a set of picked nodes, plus what was withheld.

    The blob carries the knowledge facets only (§10 consumer 5: *"Restricted to property
    and event facets"*). ``position``/``report`` claims are returned under ``withheld``,
    counted and sampled but never in the text — they are records that someone said a thing,
    and the failure this whole facet level guards against is one being read back as
    something simply known.

    ``query`` is the message the turn is about to answer. Given one, claims are ranked by
    relevance to it and **claims scoring zero are dropped rather than used as filler** — a
    node the message merely names is not a licence to recite its biography, and an empty
    blob is the honest answer to "nothing on record speaks to this". Without a query the
    previous order is kept unchanged, so an existing caller behaves exactly as before.

    ``words_match`` is injected (``exchange_anchor.words_match``) rather than imported: the
    corpus is half Russian, where an exact-token test misses `оптимизации` against
    `оптимизация`, and that matcher is the box's one tuned answer to it. Absent, scoring
    falls back to exact equality — degraded, not wrong.
    """
    picked = [i for i in (idents or []) if not G.is_self(i)]
    all_claims, self_dropped = claims_for_nodes(doc, picked)

    knowledge = [c for c in all_claims if c.get("facet") in F.KNOWLEDGE_FACETS]
    withheld = [c for c in all_claims if c.get("facet") not in F.KNOWLEDGE_FACETS]

    q_tokens = tokenize(query)
    dropped_irrelevant = 0
    if q_tokens and knowledge:
        stats = build_token_stats(doc)
        scored = [(score_claim(c.get("text") or "", q_tokens, stats, words_match), c)
                  for c in knowledge]
        keep = [(s, c) for s, c in scored if s > 0.0]
        dropped_irrelevant = len(scored) - len(keep)
        # Relevance first; the old key stays as the tie-break so ordering is still total
        # and deterministic when two claims answer the message equally well.
        keep.sort(key=lambda sc: (-sc[0], -int(sc[1].get("n_sources") or 0),
                                  -int(sc[1].get("n_occurrences") or 0),
                                  sc[1].get("text") or ""))
        knowledge = [dict(c, _score=round(s, 3)) for s, c in keep]

    text, kept = _render_grouped(doc, knowledge, max_claims, max_per_node)

    return {
        "text": text,
        "picked": picked,
        "claims": knowledge[:kept] if kept else [],
        "n_knowledge": len(knowledge),
        "n_rendered": kept,
        "withheld": withheld,
        "n_withheld": len(withheld),
        "withheld_by_facet": _count_by_facet(withheld),
        "self_dropped": self_dropped,
        # Knowledge claims the picked nodes hold that the message points at nothing in.
        # Reported rather than silent: a large number here means the pick was too coarse
        # (a person rather than a topic), which is a finding about the pass, not the tree.
        "dropped_irrelevant": dropped_irrelevant,
        "scored": bool(q_tokens),
    }


def _count_by_facet(claims: list) -> dict:
    out: dict = {}
    for c in claims:
        f = c.get("facet") or F.FACET_UNCLASSIFIED
        out[f] = out.get(f, 0) + 1
    return out


def attribution_line(doc: dict, claim: dict) -> str:
    """How a withheld claim would have to be rendered if it were ever carried.

    Not used by ``render_blob`` — the blob excludes these. It exists so the workbench can
    show an operator what the excluded material would look like *correctly* attributed,
    which is the argument §10 invites rather than a channel that quietly starts making it.
    """
    n = S.node(doc, claim.get("node") or "") or {}
    who = n.get("label") or claim.get("node") or "someone"
    facet = claim.get("facet")
    verb = ("holds" if facet == F.FACET_POSITION
            else "is depicted" if facet == F.FACET_DEPICTION else "is reported")
    return f"{who} {verb}: {claim.get('text', '')}"


# -- GPU-free self-test ------------------------------------------------------ #

def _boom_source(lane, ref):
    """A `describe_source` that raises — the render must survive it (self-test only)."""
    raise RuntimeError("resolver on fire")


def _selftest() -> None:
    """Run: ``python -m graph.blob``."""
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    # A tree shaped like the real one: person-owned claims, topic nodes reached only by
    # mention, and a fat `_self` subtree that must never surface.
    doc = {
        "nodes": {
            "person:artemyvo": {
                "id": "person:artemyvo", "type": "person", "label": "artemyvo",
                "surface_forms": ["artemyvo", "Artemy"], "n_claims": 3, "n_occurrences": 3,
                "facets": {"property": ["c1"], "position": ["c2"], "event": ["c5"]},
                "mentioned_in": [],
            },
            "person:_self": {
                "id": "person:_self", "type": "person", "label": "self",
                "surface_forms": ["self"], "n_claims": 1, "n_occurrences": 1,
                "facets": {"property": ["c3"]}, "mentioned_in": [],
            },
            "unknown:subjectivity": {
                "id": "unknown:subjectivity", "type": "unknown", "label": "subjectivity",
                "surface_forms": ["subjectivity", "субъектность"], "n_claims": 0,
                "n_occurrences": 0, "facets": {}, "mentioned_in": ["c1", "c2", "c3"],
            },
        },
        "claims": {
            "c1": {"claim_id": "c1", "node": "person:artemyvo", "facet": "property",
                   "text": "is creating Project Ava.", "n_sources": 2, "n_occurrences": 3,
                   "mentions": ["unknown:subjectivity"], "when": ""},
            "c2": {"claim_id": "c2", "node": "person:artemyvo", "facet": "position",
                   "text": "thinks a KPI system wastes subjectivity.", "n_sources": 1,
                   "n_occurrences": 1, "mentions": ["unknown:subjectivity"], "when": ""},
            "c3": {"claim_id": "c3", "node": "person:_self", "facet": "property",
                   "text": "prefers long replies.", "n_sources": 1, "n_occurrences": 1,
                   "mentions": ["unknown:subjectivity"], "when": ""},
            "c5": {"claim_id": "c5", "node": "person:artemyvo", "facet": "event",
                   "text": "sat out the 2026 war in a shelter", "n_sources": 1,
                   "n_occurrences": 1, "mentions": [], "when": "2026-06"},
        },
        "occurrences": [],
    }

    print("catalogue")
    idx = node_index(doc)
    check("the self node is never offered", "person:_self" in idx, False)
    check("a topic reachable only by mention IS offered",
          "unknown:subjectivity" in idx, True)
    check("surface forms are listed, so what was written can be found",
          "субъектность" in idx, True)
    check("a node that yields nothing is not offered",
          len(selectable_nodes({"nodes": {"unknown:x": {"id": "unknown:x", "facets": {},
                                                        "n_claims": 0, "mentioned_in": []}},
                                "claims": {}})), 0)

    print("\nparsing the pass's answer")
    sel = parse_selection("NODES:\n- unknown:subjectivity\n- person:artemyvo", doc)
    check("ids parsed", sel["picked"], ["unknown:subjectivity", "person:artemyvo"])
    check("a bare surface form resolves through find_nodes",
          parse_selection("NODES:\nсубъектность", doc)["picked"], ["unknown:subjectivity"])
    check("an unlabelled answer still parses",
          parse_selection("person:artemyvo", doc)["picked"], ["person:artemyvo"])
    check("NONE yields nothing", parse_selection("NODES:\nNONE", doc)["picked"], [])
    check("an unknown name is reported, not dropped",
          parse_selection("NODES:\nquantum tractors", doc)["unknown"], ["quantum tractors"])
    picked_self = parse_selection("NODES:\nperson:_self\nperson:artemyvo", doc)
    check("picking the self node is refused...", picked_self["picked"], ["person:artemyvo"])
    check("...and reported, so the attempt is visible", picked_self["self_picked"], True)

    print("\nrendering")
    out = render_blob(doc, ["unknown:subjectivity"])
    check("a mention edge reaches the owner's claim",
          "is creating Project Ava" in out["text"], True)
    check("the blob is attributed to the node that owns the claim",
          out["text"].startswith("artemyvo:"), True)
    check("a position is withheld from the blob",
          "KPI system" in out["text"], False)
    check("...and counted", out["withheld_by_facet"], {"position": 1})
    check("a self claim reached by mention is dropped", out["self_dropped"], 1)
    check("...and never in the text", "long replies" in out["text"], False)
    check("corroboration is marked", "[corroborated]" in out["text"], True)
    ev = render_blob(doc, ["person:artemyvo"])
    check("an event carries its when", "(2026-06)" in ev["text"], True)
    check("the caps bound the blob",
          render_blob(doc, ["person:artemyvo"], max_claims=1)["n_rendered"], 1)
    check("with no query the old order is kept untouched", ev["scored"], False)

    print("\nrelevance (the arriving message ranks what is shown)")
    # Both of artemyvo's knowledge claims are offerable; only one answers each message.
    war = render_blob(doc, ["person:artemyvo"], query="how did you get through the war?")
    check("the claim the message points at is kept",
          "sat out the 2026 war" in war["text"], True)
    check("...and the unrelated one is dropped, not used as filler",
          "Project Ava" in war["text"], False)
    check("...and the drop is counted", war["dropped_irrelevant"], 1)
    proj = render_blob(doc, ["person:artemyvo"], query="how is Project Ava going?")
    check("a different message selects the other claim",
          "is creating Project Ava" in proj["text"], True)
    check("...dropping the first", "2026 war" in proj["text"], False)

    # The failure that produced "results seem irrelevant": every claim on this corpus has
    # n_sources == n_occurrences == 1, so the old key collapsed to alphabetical order.
    flat = {"claim_id": "cN", "node": "person:artemyvo", "facet": "property",
            "n_sources": 1, "n_occurrences": 1, "mentions": [], "when": ""}
    many = dict(doc)
    many["claims"] = dict(doc["claims"], **{
        f"z{i}": dict(flat, claim_id=f"z{i}", text=t) for i, t in enumerate([
            "Aardvark unrelated filler one.", "Beetle unrelated filler two.",
            "Cormorant unrelated filler three.", "keeps bees on the roof in summer.",
            "tends a beekeeping operation on the roof."])})
    many["nodes"] = dict(doc["nodes"], **{"person:artemyvo": dict(
        doc["nodes"]["person:artemyvo"],
        facets={"property": ["c1", "z0", "z1", "z2", "z3", "z4"], "position": ["c2"],
                "event": ["c5"]})})
    bees = render_blob(many, ["person:artemyvo"], query="do you still keep bees?",
                       max_per_node=3)
    check("relevance beats alphabetical order", "keeps bees" in bees["text"], True)
    check("...and the alphabetically-first filler is not shown",
          "Aardvark" in bees["text"], False)
    check("...leaving only what the message asked about", bees["n_rendered"], 1)

    print("\nrelevance: the injected matcher")
    # Exact equality is the fallback; an inflected corpus needs the real matcher, which the
    # caller injects (`exchange_anchor.words_match`). Simulated here by a prefix matcher.
    def _inflect(a, b):
        return a == b or (min(len(a), len(b)) >= 5 and a[:5] == b[:5])

    plain = render_blob(many, ["person:artemyvo"], query="beekeepers")
    check("exact equality alone misses an inflected form", plain["n_rendered"], 0)
    inflected = render_blob(many, ["person:artemyvo"], query="beekeepers",
                            words_match=_inflect)
    check("the injected matcher finds it", "beekeeping" in inflected["text"], True)

    print("\nclaim lane: the candidate list is filtered BEFORE the pass sees it")
    # The two §10 rules hold by construction here — a forbidden claim is never shown, so it
    # cannot be picked, mis-parsed, or argued into the blob.
    cands = claim_candidates(doc, now="2026-08-12")
    texts = [c["text"] for c in cands["claims"]]
    check("a position is not offered", any("KPI" in t for t in texts), False)
    check("a self claim is not offered", any("long replies" in t for t in texts), False)
    check("knowledge is offered", any("Project Ava" in t for t in texts), True)
    check("...and so is an event", any("2026 war" in t for t in texts), True)
    # `total_knowledge` is the OFFERABLE universe (self already excluded — §10 is not a
    # tunable, so it is not reported as a drop). The counts must reconcile, or the operator
    # line "N of M, X stale" is arithmetic nobody can check.
    check("the counts reconcile",
          cands["n"] + cands["dropped_stale"] + cands["dropped_capped"],
          cands["total_knowledge"])

    print("\nclaim lane: TIL is not aged out by default, and can be scoped on request")
    aged = {"claims": {
        "t1": {"claim_id": "t1", "node": "entity:X", "facet": "event", "n_sources": 1,
               "n_occurrences": 1, "text": "a digest said a thing", "lanes": ["til"],
               "last_asserted": "2026-08-01", "mentions": [], "when": ""},
        "t2": {"claim_id": "t2", "node": "entity:X", "facet": "event", "n_sources": 1,
               "n_occurrences": 1, "text": "a fresh digest", "lanes": ["til"],
               "last_asserted": "2026-08-11", "mentions": [], "when": ""},
        "c9": {"claim_id": "c9", "node": "person:artemyvo", "facet": "property",
               "n_sources": 1, "n_occurrences": 1, "text": "an old chat fact",
               "lanes": ["chat"], "last_asserted": "2026-01-01", "mentions": [], "when": ""},
        "b1": {"claim_id": "b1", "node": "entity:X", "facet": "property", "n_sources": 2,
               "n_occurrences": 2, "text": "said in both lanes", "lanes": ["til", "chat"],
               "last_asserted": "2026-01-01", "mentions": [], "when": ""},
    }, "nodes": {}}
    got = [c["text"] for c in claim_candidates(aged, now="2026-08-12")["claims"]]
    check("old TIL is kept by default", "a digest said a thing" in got, True)
    check("fresh TIL is kept", "a fresh digest" in got, True)
    check("an old CHAT fact is never aged out", "an old chat fact" in got, True)
    check("a claim corroborated in both lanes counts as chat-backed",
          "said in both lanes" in got, True)
    check("nothing is dropped as stale by default",
          claim_candidates(aged, now="2026-08-12")["dropped_stale"], 0)
    # The mechanism is intact — only its default changed — so an operator who sets a scope
    # gets the old behaviour back, chat still exempt from it.
    scoped = claim_candidates(aged, now="2026-08-12", til_max_age_days=7)
    scoped_got = [c["text"] for c in scoped["claims"]]
    check("an explicit scope drops stale TIL",
          "a digest said a thing" in scoped_got, False)
    check("...but never a chat-backed claim", "an old chat fact" in scoped_got, True)
    check("...and the drop is counted", scoped["dropped_stale"], 1)
    check("chat-backed sorts ahead of TIL, so the cap bites news first",
          [c["text"] for c in
           claim_candidates(aged, now="2026-08-12", max_candidates=2)["claims"]],
          ["an old chat fact", "said in both lanes"])

    print("\nclaim lane: the catalogue and the answer")
    idx = claim_index(doc, cands)
    check("lines are numbered from 1", idx.startswith("[1] "), True)
    check("each line names the owning subject", "artemyvo:" in idx, True)
    n_lines = len([l for l in idx.split("\n") if l.strip()])
    check("one line per candidate", n_lines, cands["n"])

    sel = parse_claim_selection("FACTS:\n2\n1", cands)
    check("numbers parse in the order given", sel["numbers"], [2, 1])
    check("...into the claims they indexed",
          sel["picked"][0]["text"], cands["claims"][1]["text"])
    check("decoration is tolerated",
          parse_claim_selection("FACTS:\n- [2]\n1.", cands)["numbers"], [2, 1])
    check("a missing label still parses",
          parse_claim_selection("2", cands)["numbers"], [2])
    check("NONE picks nothing", parse_claim_selection("FACTS:\nNONE", cands)["picked"], [])
    check("...and reads as a deliberate refusal",
          parse_claim_selection("FACTS:\nNONE", cands)["none"], True)
    oor = parse_claim_selection("FACTS:\n999", cands)
    check("an invented number is reported, not silently dropped",
          oor["out_of_range"], [999])
    check("...and is NOT a refusal", oor["none"], False)
    check("duplicate picks collapse",
          parse_claim_selection("FACTS:\n1\n1\n1", cands)["numbers"], [1])
    check("the pick cap holds",
          len(parse_claim_selection("FACTS:\n" + "\n".join(str(i) for i in range(1, 30)),
                                    cands)["numbers"]) <= MAX_PICKS, True)

    print("\nclaim lane: rendering what was picked")
    rc = render_claims(doc, parse_claim_selection("FACTS:\n1\n2", cands)["picked"])
    check("the blob is grouped under the owning subject",
          rc["text"].startswith("artemyvo:"), True)
    check("nothing withheld can appear", "KPI" in rc["text"], False)
    check("an empty pick renders empty", render_claims(doc, [])["text"], "")

    print("\nprovenance: which conversations the picks came out of")
    prov = {
        "nodes": doc["nodes"],
        "claims": {
            # Corroborated across two conversations, told again more recently.
            "p1": {"claim_id": "p1", "node": "person:artemyvo", "facet": "property",
                   "text": "keeps bees.", "occurrences": [0, 1], "n_sources": 2,
                   "n_occurrences": 2, "mentions": [], "when": ""},
            # Same conversation as p1's older telling: the dedup must hold across claims.
            "p2": {"claim_id": "p2", "node": "person:artemyvo", "facet": "event",
                   "text": "moved house.", "occurrences": [1], "n_sources": 1,
                   "n_occurrences": 1, "mentions": [], "when": ""},
            # TIL only: nothing on the other end of the pointer.
            "p3": {"claim_id": "p3", "node": "entity:X", "facet": "property",
                   "text": "an article said a thing.", "occurrences": [2], "n_sources": 1,
                   "n_occurrences": 1, "mentions": [], "when": ""},
        },
        "occurrences": [
            {"lane": "chat", "source_ref": "20260810_100000.json",
             "asserted_at": "2026-08-10"},
            {"lane": "chat", "source_ref": "20260701_100000.json",
             "asserted_at": "2026-07-01"},
            {"lane": "til", "source_ref": "news/2026-07-28", "asserted_at": "2026-07-28"},
        ],
    }
    check("a claim resolves to the conversation it came out of",
          chat_sources(prov, [prov["claims"]["p2"]]), ["20260701_100000.json"])
    check("the most recent telling leads",
          chat_sources(prov, [prov["claims"]["p1"]]),
          ["20260810_100000.json", "20260701_100000.json"])
    check("pick order decides which conversation is first",
          chat_sources(prov, [prov["claims"]["p2"], prov["claims"]["p1"]]),
          ["20260701_100000.json", "20260810_100000.json"])
    check("one conversation is named once however many claims came from it",
          len(chat_sources(prov, [prov["claims"]["p1"], prov["claims"]["p2"]])), 2)
    check("a TIL claim points at no conversation",
          chat_sources(prov, [prov["claims"]["p3"]]), [])
    check("no picks, no sources", chat_sources(prov, []), [])
    check("a claim whose occurrences are gone degrades to nothing",
          chat_sources({"occurrences": []}, [prov["claims"]["p1"]]), [])

    # The typed form: the two lanes' refs resolve differently, so the lane travels with
    # the ref rather than being re-derived from its shape by every caller.
    check("a TIL claim names its text, typed",
          sources(prov, [prov["claims"]["p3"]]), [("til", "news/2026-07-28")])
    check("both lanes come back, best-first",
          sources(prov, [prov["claims"]["p3"], prov["claims"]["p2"]]),
          [("til", "news/2026-07-28"), ("chat", "20260701_100000.json")])
    both = {**prov, "claims": {"b": {"claim_id": "b", "node": "person:artemyvo",
                                     "facet": "property", "text": "said in both.",
                                     "occurrences": [1, 3], "n_sources": 2,
                                     "n_occurrences": 2, "mentions": [], "when": ""}},
            "occurrences": prov["occurrences"] + [
                {"lane": "til", "source_ref": "news/2026-07-01",
                 "asserted_at": "2026-07-01"}]}
    check("at equal date the conversation is the more particular memory",
          sources(both, [both["claims"]["b"]])[0], ("chat", "20260701_100000.json"))

    print("\nrelevance: scoring")
    stats = build_token_stats(many)
    check("a token in every claim is weightless", round(_idf("the", {"df": {"the": 8},
                                                                    "n": 8}), 6), 0.0)
    check("an unmatched message scores zero",
          score_claim("is creating Project Ava.", tokenize("quantum tractors"), stats), 0.0)
    check("short tokens are not topics", tokenize("a to the of"), ["the"])
    check("picking the self node directly yields nothing",
          render_blob(doc, ["person:_self"])["text"], "")

    print("\nattribution of the withheld material")
    check("a position renders as a record of who holds it",
          attribution_line(doc, doc["claims"]["c2"]),
          "artemyvo holds: thinks a KPI system wastes subjectivity.")

    print("\nthe reading lane's report channel")
    # A TIL report claim with a real occurrence behind it, so `sources` can name the text.
    rdoc = {
        "nodes": {"entity:USCENTCOM": {"id": "entity:USCENTCOM", "label": "USCENTCOM",
                                       "facets": {"report": ["r1"]}, "mentioned_in": []},
                  "person:a": {"id": "person:a", "label": "a",
                               "facets": {"property": ["k1"]}, "mentioned_in": []}},
        "claims": {
            "r1": {"claim_id": "r1", "node": "entity:USCENTCOM", "facet": "report",
                   "text": "All missiles were intercepted.", "when": "2026-07-28",
                   "lanes": ["til"], "last_asserted": "2026-07-28", "n_sources": 1,
                   "n_occurrences": 1, "mentions": [], "occurrences": [0]},
            "k1": {"claim_id": "k1", "node": "person:a", "facet": "property",
                   "text": "was in a bomb shelter.", "when": "", "lanes": ["chat"],
                   "last_asserted": "2026-07-31", "n_sources": 1, "n_occurrences": 1,
                   "mentions": [], "occurrences": [1]}},
        "occurrences": [{"lane": "til", "source_ref": "news/2026-07-28",
                         "asserted_at": "2026-07-28"},
                        {"lane": "chat", "source_ref": "20260731_134223.json",
                         "asserted_at": "2026-07-31"}]}

    check("the default list still withholds reports — chat is untouched",
          [c["claim_id"] for c in claim_candidates(rdoc, now="2026-08-15")["claims"]],
          ["k1"])
    wide = claim_candidates(rdoc, now="2026-08-15", facets=READING_FACETS)
    check("the reading lane is offered both", sorted(c["claim_id"] for c in wide["claims"]),
          ["k1", "r1"])
    check("...and counts the reports it added", wide["n_reports"], 1)
    check("...while total_knowledge keeps its strict meaning", wide["total_knowledge"], 1)
    check("position is offered to neither", F.FACET_POSITION in READING_FACETS, False)
    # §10 is about the subtree, not the facet — widening must not reach it.
    sdoc = {**rdoc, "claims": {"s1": {**rdoc["claims"]["r1"], "node": G.SELF_NODE}}}
    check("widening the facets does NOT widen the self rule",
          claim_candidates(sdoc, now="2026-08-15", facets=READING_FACETS)["claims"], [])

    picked = [rdoc["claims"]["k1"], rdoc["claims"]["r1"]]
    out = render_claims(rdoc, picked, max_claims=3)
    check("a report picked with no budget is dropped, and counted",
          (out["n_reports"], out["n_reports_dropped"]), (0, 1))
    check("...leaving the knowledge blob exactly as before", "intercepted" in out["text"],
          False)

    out = render_claims(rdoc, picked, max_claims=3, max_reports=2,
                        describe_source=lambda lane, ref: "a digest of world events "
                                                          "from 2026-07-28")
    check("the report is attributed to the text that asserted it",
          "a digest of world events from 2026-07-28 reported (2026-07-28): "
          "All missiles were intercepted." in out["text"], True)
    check("...and is NOT folded under its node like knowledge",
          out["text"].index("From the texts named") > out["text"].index("was in a bomb"),
          True)
    check("both are counted separately", (out["n_knowledge"], out["n_reports"]), (1, 1))
    check("a resolver that returns nothing falls back to the ref, never to silence",
          "news/2026-07-28 reported" in
          render_claims(rdoc, [rdoc["claims"]["r1"]], max_reports=1,
                        describe_source=lambda l, r: "")["text"], True)
    check("a raising resolver does not take the render down",
          "reported" in render_claims(rdoc, [rdoc["claims"]["r1"]], max_reports=1,
                                      describe_source=_boom_source)["text"], True)
    check("a report with no source at all still hedges",
          report_line({"claims": {}, "occurrences": [], "nodes": {}},
                      {"facet": "report", "text": "x.", "occurrences": []}),
          "something on record reported: x.")

    # Fiction: carried, but the verb refuses to let it read as a stale news item.
    fdoc = {**rdoc, "claims": {"f1": {**rdoc["claims"]["r1"], "claim_id": "f1",
                                      "facet": "depiction", "when": "",
                                      "text": "Jimmy lives on the Moon with his parents."}}}
    check("a depiction is offered to the reading lane",
          [c["claim_id"] for c in claim_candidates(fdoc, now="2026-08-15",
                                                   facets=READING_FACETS)["claims"]], ["f1"])
    check("...and never to chat",
          claim_candidates(fdoc, now="2026-08-15")["claims"], [])
    check("...rendered as depiction, not as a report",
          report_line(fdoc, fdoc["claims"]["f1"],
                      lambda l, r: "an article about Robatt on Neolurk"),
          "an article about Robatt on Neolurk depicts: Jimmy lives on the Moon with "
          "his parents.")
    check("...and it shares the sourced section rather than splitting it",
          render_claims(fdoc, [fdoc["claims"]["f1"]], max_reports=2)["n_reports"], 1)
    check("the withheld-material view names it too",
          attribution_line(fdoc, fdoc["claims"]["f1"]).split(":")[0],
          "USCENTCOM is depicted")

    print("\nthe shipped prompt still asks for the contract this parses")
    from pathlib import Path
    p = (Path(__file__).resolve().parent.parent / "inference" / "prompts"
         / "fact_fetch_prompt.txt")
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        # The live lane is the claim lane; the prompt must ask for ITS label, or the parser
        # reads a correct answer as an empty one.
        check("the answer label", f"{CLAIM_SELECTION_LABEL}:" in txt, True)
        check("the empty answer", NONE_MARKER in txt, True)
        # The whole reason this lane exists: the corpus is largely English while the
        # conversations often are not, so the pass must be told not to match on wording.
        check("...and tells the pass to judge across languages",
              "different languages" in txt, True)
    else:
        print(f"  --   prompt not on disk ({p.name}) — skipped")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
