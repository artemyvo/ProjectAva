"""Modules — the workbench seam: run ONE pass against ONE input and see what it produced.

Every generation on this box is the same shape — *something is injected into the prompt*,
*something is read* (usually a chat), *something is produced* — but each pass hardcodes its
own answer to all three, scattered across `reflection_runner`, `checkin`, `synthesis`,
`outreach`, `deliberation`. The result is that nobody can say what a given pass sees
without reading its call site, and nobody can try a change to one without triggering the
run it lives inside.

A **module** is that shape made explicit and runnable on demand:

    context (what conditions it) + input (what it reads) + output (what it produces)

**Scope so far.** Three modules — ``chat_facts`` (the per-chat extraction protocol),
``chat_summary`` (the consolidation gist) and ``til_facts`` (the per-article/news
protocol) — input bound to one operator-chosen item,
**nothing injected** (``disable_rag=True``, no persona, no clock), and the output is
**returned, never written**. That last part is the load-bearing one: the real passes write
their sidecars inside themselves, which is why they can neither be experimented with nor
chained. Here the sink is detached — a module run yields a value the operator looks at.
Attaching a sink (and wiring one module's value to another's input) is a later stage; the
shape is already right for it, and ``chat_summary``'s output feeding ``checkin`` is the
chain that already exists hardcoded (``checkin._stored_gist``).

The second module is what firmed up the shape. ``build`` returns a LIST of content blocks,
because the summary pass genuinely runs once per consolidation chunk and joins the parts;
and ``finish`` turns the raw generations into the displayed result, so a module that
produces prose and one that produces records are both describable without the client
knowing which is which.

The **third** added the last dimension the shape was missing. ``til_facts`` reads a fetched
article or news digest out of the snippets tree rather than a transcript out of the chats
dir — no exchanges, measured in characters, addressed by a `<kind>/<name>` id — while
everything above the loader is identical. So *what a pass reads* became data too
(``ModuleSpec.source`` + ``_INPUT_SOURCES``), and the client asks the server what a module
can run on (``list_module_inputs``) instead of knowing per module. Adding a fourth lane is
an entry in that dict plus ``source=`` on the spec.

What is deliberately NOT here yet: injectables (the catalogue is empty and a non-empty
selection is REFUSED rather than ignored — a UI must never believe it injected something
that did nothing), saved cases, chaining, and any path that writes. The registry is one
dict, so growing it is a data edit.

Manual-trigger only, mirroring the other decision-pass subsystems: a blocking pass on the
GPU executor thread + a handler that streams. Never imports ``server`` — every capability
is injected once via :func:`configure`. GPU-free self-test: ``python -m core.modules``.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.runtime_state import runtime as _runtime, reflect_window as _reflect_window
from core import activity_log
from core import chat_facts, fact_fetch, reasoning_text

# Set while a module run occupies the single GPU executor thread. Owned here; the
# scheduler's `external_busy` and sibling host_busy getters may read it.
_module_run_active = False

# ── Injected server capabilities (populated by configure()) ──
_get_rag: Callable = None
_make_sync_reflect_generate: Callable = None
_PROMPTS_DIR: Any = None
_CHATS_DIR: Any = None
_TIL_SNIPPETS_DIR: Any = None
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None


def configure(*, get_rag, make_sync_reflect_generate, prompts_dir, chats_dir,
              til_snippets_dir=None, send=None, executor=None, host_busy=None,
              mark_activity=None) -> None:
    """Wire in the server capabilities the module workbench depends on (once, at startup)."""
    global _get_rag, _make_sync_reflect_generate, _PROMPTS_DIR, _CHATS_DIR
    global _TIL_SNIPPETS_DIR, _send, _executor, _host_busy, _mark_activity
    from pathlib import Path
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _PROMPTS_DIR = Path(prompts_dir)
    _CHATS_DIR = Path(chats_dir)
    # Absent ⇒ the `til` source lists nothing and refuses to load, which is the honest
    # degrade for a box whose snippets tree a caller did not wire.
    _TIL_SNIPPETS_DIR = Path(til_snippets_dir) if til_snippets_dir else None
    _send = send
    _executor = executor
    _host_busy = host_busy
    _mark_activity = mark_activity


# ── the registry ──────────────────────────────────────────────────────────────

class ModuleInputError(Exception):
    """A module's ``build`` refusing its input, with a sentence saying why.

    ``build`` returning ``[]`` already means "nothing to read", but the generic message
    that produces ("rendered no readable chat content") is a lie for a module whose input
    is missing for a reason worth naming — ``fact_fetch`` needs a built facts tree, and an
    operator told the transcript was unreadable would go looking in the wrong place.
    """


@dataclass(frozen=True)
class ModuleSpec:
    """One pass, described as data: what it reads, what conditions it, what it yields.

    *injectables* is the catalogue of context blocks this module MAY be given. Empty for
    both modules so far — and empty means refused, not ignored: a selection the server
    would silently drop is worse than an error, because the operator would read the output
    as evidence about an injection that never happened.

    *stop_on_repeat* is the verbatim loop guard (see the note at
    ``generation._make_sync_reflect_generate``'s ``stream_generate`` call). It belongs on
    the module rather than at the call site because whether it is correct is a property of
    the OUTPUT SHAPE: a pass that emits prose wants it, a pass that emits a fixed-template
    list is killed by it after 4 lines.

    *degen_stop* is the diversity guard (Layer 2, ``_DegenStop``), on the spec for the
    identical reason and proven necessary the same way (2026-08-18, live): a run of
    template lines sharing an (about, class) prefix craters the rolling distinct-token
    ratio exactly as a real collapse does, so the facts modules turn BOTH guards off and
    let the token cap bound a genuine runaway.

    **``build`` returns a LIST of content blocks**, because "how the material is cut up"
    is a real per-pass property rather than an implementation detail. ``chat_facts`` reads
    the whole session as one block; the summary pass runs once per *consolidation chunk*
    and joins the results, so a long chat genuinely is several generations. One block is
    the common case, not the only one.

    **``finish`` turns the raw generations into the displayable result** — parsing,
    counting and rendering in one place, per module. It replaces the earlier
    facts-shaped ``parse``/``summarize`` pair, which could not describe a pass whose
    output is prose. It returns a dict carrying at least ``records`` / ``count`` /
    ``lines``; the client prints ``lines`` verbatim, so adding a module needs no client
    change — which is the property that makes the registry worth having.

    *source* names the INPUT KIND this module reads — which list of things it can be run
    against, and how one of them is loaded. It exists because the registry's second lane
    arrived: `til_facts` reads a fetched article or news digest out of the snippets tree,
    not a transcript out of the chats dir, and everything above the loader (the prompt
    override, the block loop, the streaming, the refusal rules, the write-nothing
    contract) is identical for both. So the difference is declared as data like the rest
    of the pass, rather than branched on inside the run — which is also what lets the
    client ask "what can this module run on?" instead of knowing the answer per module.
    """
    name: str
    label: str
    prompt_file: str
    max_new_tokens: str
    build: Callable[..., list]
    finish: Callable[..., dict]
    describe_output: str
    injectables: tuple[str, ...] = ()
    stop_on_repeat: bool = True
    # None ⇒ the box default (Layer 2 on unless configured off); False ⇒ off for this
    # pass. See the class docstring.
    degen_stop: Optional[bool] = None
    source: str = "chat"
    # Whether the pass thinks. Data on the spec for the same reason `stop_on_repeat` is: it
    # is a property of the JOB, not of the call site. `generation._make_sync_reflect_generate`
    # defaults thinking ON and lists the passes that turn it off — branch chooser, branch
    # judge, fact placement, anchors, the clustering/dedup evaluations — and they share one
    # shape: a selection or a judgement over material already in front of them, whose output
    # is a label or a list rather than a written answer. A module of that shape belongs on
    # that list, and saying so here keeps the reason next to the pass.
    disable_thinking: bool = False


# ── chat_facts: the per-chat extraction protocol ──────────────────────────────
# Production twin: `reflection_runner._run_chat_facts_pass_for_session`. The closing
# question and token budget are taken from there verbatim so a simulation is faithful to
# what the real run would have generated (the self-test asserts they still agree).

_CHAT_FACTS_CLOSING = (
    "— end of the conversation —\n\n"
    "Now write the protocol: every fact this conversation established, one per [fact] "
    "line, each marked with who it is about and whether it is standing, stated or an "
    "event. Record what was said, not what it suggests about anyone."
)


def _facts_build(session: dict, window: int, tokenizer) -> list:
    from core.reflection_source import build_session_reading_content
    # Production composes the participants note into the closing (the canonical (about:)
    # spellings stated in-context) and parses with the session-aware subject fold, so the
    # simulation does both — the session rides the build context to `finish`, which is
    # what that seam exists for.
    names_note = chat_facts.participants_note(session)
    closing = (f"{names_note}\n\n{_CHAT_FACTS_CLOSING}" if names_note
               else _CHAT_FACTS_CLOSING)
    content = build_session_reading_content(
        session, window, closing=closing,
        ava_initiated_note=chat_facts.AVA_INITIATED_NOTE)
    return ([content], {"session": session}) if content else []


def _facts_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    session = (context or {}).get("session") or {}
    facts = chat_facts.parse_facts(raws[0] if raws else "", truncated=truncated,
                                   subject_fn=chat_facts.make_subject_fn(session))
    counts = chat_facts.class_counts(facts)
    lines = []
    for i, rec in enumerate(facts, 1):
        subject = rec.get("subject") or "—"
        klass = rec.get("fact_class") or chat_facts.UNSPECIFIED
        line = f"{i:3d}. ({klass}) about {subject}: {rec.get('text', '')}"
        extra = []
        if rec.get("entities"):
            extra.append("entities: " + ", ".join(rec["entities"]))
        if rec.get("when"):
            extra.append("when: " + rec["when"])
        if extra:
            line += "\n      [" + " · ".join(extra) + "]"
        lines.append(line)
    return {"records": facts, "counts": counts, "count": len(facts), "lines": lines}


# ── chat_summary: the per-chat consolidation gist ─────────────────────────────
# Production twin: the SUMMARY pass inside `reflection_runner._run_consolidation_...`,
# which runs once per consolidation chunk over the SAME packed transcript and joins the
# parts. Reproduced here rather than approximated with the whole-session reading block,
# because the chunking is what makes a long chat several generations instead of one.
#
# ONE deliberate divergence from production: the chunks are built with an EMPTY
# open-questions block. Production packs the live open `[ask]` items into that content,
# which is an injection — and this module declares that it injects nothing. Declaring the
# open-questions block as a real injectable is the honest fix and belongs with the rest of
# the injectables work; until then the divergence is stated rather than hidden.
#
# The closing (`reflection_chunking.SUMMARY_CLOSING`) is the same text on both sides but
# applied at a different point: baked into the chunks here, appended at the call site in
# production, whose chunks are shared with consolidation and would break under it. So the
# module's is counted by the packing budget and production's is not — a difference in how
# a long chat splits, never in what the pass is asked to do.

def _summary_build(session: dict, window: int, tokenizer) -> list:
    from core.reflection_chunking import build_consolidation_chunks, SUMMARY_CLOSING
    # The closing is baked into the chunks here (production appends it at the call site
    # instead — its chunks are shared with consolidation), so it is counted by the fit
    # test. Same text either way: `reflection_chunking.SUMMARY_CLOSING`.
    chunks = build_consolidation_chunks(session, window, "", tokenizer,
                                        closing=SUMMARY_CLOSING)
    return [c["content"] for c in chunks if (c.get("content") or "").strip()]


def _summary_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    from core.chat_sidecar import sanitize_gist
    from core import reasoning_text
    parts = []
    dropped = 0
    for raw in raws:
        body = reasoning_text.answer_after_think(raw or "")
        cleaned = sanitize_gist(body)
        if cleaned:
            parts.append(cleaned)
        elif (raw or "").strip():
            # `sanitize_gist` refused it: the generation degenerated into the structured
            # consolidation dump, or came back too short to be a recap. Worth reporting —
            # production silently drops these, so a chat with no gist looks like a chat
            # that was never summarized.
            dropped += 1
    text = "\n\n".join(parts)
    lines = text.splitlines() if text else []
    if dropped:
        lines.append("")
        lines.append(f"[{dropped} generated part(s) rejected by the gist sanitizer — "
                     f"structured dump or too short to be prose]")
    return {"records": [], "text": text, "counts": {}, "count": len(parts),
            "rejected": dropped, "lines": lines}


_CHAT_FACTS = ModuleSpec(
    name="chat_facts",
    label="Facts (chat protocol)",
    prompt_file="chat_facts_prompt.txt",
    max_new_tokens="12288",
    build=_facts_build,
    finish=_facts_finish,
    describe_output="[fact] lines → <stem>.facts.json (NOT written in a simulation)",
    injectables=(),
    # BOTH guards off, deterministically necessary. Every output line is
    # `[fact] (about: NAME) (class: CLASS) …`; that prefix tokenizes past the verbatim
    # guard's 12-token window (fires on the 4th consecutive line sharing an
    # (about, class) pair — a single-topic chat cannot get past 3 facts), and a run of
    # such lines also craters the diversity guard's distinct-token ratio (proven live on
    # the TIL twin, 2026-08-18). The token cap bounds a genuine runaway.
    stop_on_repeat=False,
    degen_stop=False,
)

_CHAT_SUMMARY = ModuleSpec(
    name="chat_summary",
    label="Summary (chat gist)",
    prompt_file="summary_prompt.txt",
    # Production uses the RUN's max_new_tokens, whose default is this
    # (`reflection_runner._DEFAULT_MAX_NEW_TOKENS`).
    max_new_tokens="8192",
    build=_summary_build,
    finish=_summary_finish,
    describe_output="prose recap → <stem>.summary.json (NOT written in a simulation)",
    injectables=(),
    # ON — the default, and correct here: this pass emits prose, which is exactly the
    # output shape the verbatim loop guard is built for.
    stop_on_repeat=True,
)

# ── til_facts: the per-source extraction protocol (wander / news) ─────────────
# Production twin: `til_wander.run_facts_pass`. Same prompt file, same budget, same
# `stop_on_repeat=False`, and the SAME block builder — so a simulation here is faithful
# and, unlike the chat pair, there is no second definition of the pass to keep in step:
# both callers go through `til_facts.build_reading_blocks` / `til_facts.parse_facts`.
#
# This is the module that forced `ModuleSpec.source`. It reads a fetched article or news
# digest out of `data/til/snippets/<kind>/`, which is not a transcript and has no
# exchanges — and it is the pass that most needs the workbench, since its material is
# dense, multilingual and wildly variable in length, and it had no interactive way to be
# tuned at all.

def _til_facts_build(record: dict, window: int, tokenizer) -> list:
    from core import til_facts
    return til_facts.build_reading_blocks(
        record, str(record.get("_kind") or "wander"),
        til_facts.block_budget_chars(window))


def _til_facts_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    from core import til_facts
    facts: list = []
    for i, raw in enumerate(raws):
        # Only the LAST block can be cut by the cap; an earlier one ended on its own.
        last = (i == len(raws) - 1)
        facts.extend(til_facts.parse_facts(raw or "",
                                           truncated=bool(truncated) and last))
    lines = []
    for i, rec in enumerate(facts, 1):
        subject = rec.get("subject") or "—"
        klass = rec.get("fact_class") or til_facts.UNSPECIFIED
        line = f"{i:3d}. ({klass}) about {subject}: {rec.get('text', '')}"
        extra = []
        if rec.get("entities"):
            extra.append("entities: " + ", ".join(rec["entities"]))
        if rec.get("when"):
            extra.append("when: " + rec["when"])
        if extra:
            line += "\n      [" + " · ".join(extra) + "]"
        lines.append(line)
    return {"records": facts, "counts": til_facts.class_counts(facts),
            "count": len(facts), "lines": lines}


_TIL_FACTS = ModuleSpec(
    name="til_facts",
    label="Facts (article/news protocol)",
    prompt_file="til_facts_prompt.txt",
    max_new_tokens="12288",          # matches `til_wander.run_facts_pass`
    build=_til_facts_build,
    finish=_til_facts_finish,
    describe_output=("[fact] lines → <snippet>.facts.json "
                     "(NOT written in a simulation)"),
    injectables=(),
    # BOTH guards off, for the chat protocol's reason and more sharply — see
    # `til_wander.run_facts_pass`: a digest is many events about one country, so a run of
    # lines sharing an (about, class) prefix is this lane's normal shape rather than its
    # corner case (and it is where the diversity guard was caught firing, live).
    stop_on_repeat=False,
    degen_stop=False,
    source="til",
)

# ── fact_fetch: stage 1 of a two-stage chat turn ──────────────────────────────
# **No production twin — this one is the argument, not a simulation of something running.**
# Every other module here mirrors a pass that already runs somewhere; this one is a pass
# that does not exist yet, built in the workbench first precisely because the workbench
# writes nothing and injects nothing. It is FACTS_TREE.md §10 **consumer 5** — the
# node-scoped retrieval channel the design doc lists LAST and marks *"gated, off by
# default … may well never be built; it is here to be argued about, not assumed"*. Running
# it here is how the argument gets real output in front of it.
#
# The shape: given the conversation and the message that just arrived, pick the nodes worth
# looking up, and render what the tree says about them. In a live turn that blob would be
# injected and a second pass would write the reply. Here the blob is returned and looked at.
#
# Thinking is OFF, and the output contract is a list of ids rather than prose. Both follow
# from the same decision: the pass SELECTS, and the rendering is done in code
# (`graph.blob.render_blob`). That is what keeps the blob faithful — the text comes out of
# the tree, so a fetch pass cannot paraphrase a claim into something the corpus never said,
# and the facet rule (`position` is a record that someone SAID a thing, never knowledge)
# is enforced by the renderer rather than requested from a generation. A prose-writing
# stage 1 would need thinking, cost seconds on the chat hot path, and put a generation
# between the source of truth and the prompt.
#
# The input is a chat, and the pass is simulated at its FINAL user turn — the same turn
# training targets under final-turn masking, and the one whose reply the whole transcript
# leads up to.

def _load_tree():
    """The facts tree, or None. Read fresh: one JSON read is nothing beside a generation,
    and a cached tree in a workbench whose whole point is rebuilding and re-running would
    be a stale answer that looks like a live one."""
    import sys as _sys
    from pathlib import Path
    server_dir = Path(__file__).resolve().parent.parent.parent
    if str(server_dir) not in _sys.path:
        _sys.path.insert(0, str(server_dir))
    from graph import store
    return store.read_tree()


def _fetch_build(session: dict, window: int, tokenizer) -> list:
    from core.reflection_source import session_transcript_turns, render_user_turn
    import sys as _sys
    from pathlib import Path
    server_dir = Path(__file__).resolve().parent.parent.parent
    if str(server_dir) not in _sys.path:
        _sys.path.insert(0, str(server_dir))
    from graph import blob as graph_blob

    doc = _load_tree()
    if doc is None:
        raise ModuleInputError(
            "No facts tree on this box. The tree is derived and disposable — build it "
            "with `python -m graph.build` from server/, then run this again.")

    turns = session_transcript_turns(session)
    last_user = fact_fetch.last_real_user_turn(session, turns)
    if last_user is None:
        raise ModuleInputError(
            "This chat has no real user turn to fetch for. A fetch pass answers 'what "
            "should I look up before replying to THIS message', and the only user-slot "
            "turns here are synthetic — an Ava-initiated opener's exchange 0 is a stage "
            "direction she wrote to herself, not something that arrived. Pick a chat with "
            "a message from the other side.")

    # The candidate list is scoped and filtered before the pass sees it — see
    # `graph.blob.claim_candidates`. `now` is passed rather than read there, so the same
    # corpus and date always produce the same list.
    import datetime as _dt
    cands = graph_blob.claim_candidates(doc, now=_dt.date.today().isoformat())
    index = graph_blob.claim_index(doc, cands)
    if not index.strip():
        raise ModuleInputError(
            "The tree holds no fact worth offering. Everything on record is either a "
            "position someone stated (withheld from a knowledge channel by design), under "
            "the self subtree, or older than the freshness window. Reflect some chats and "
            "rebuild the tree with `python -m graph.build`.")

    msg = turns[last_user]
    # The whole prior conversation, the arriving message, and the task — assembled by the
    # SHARED builder, so the pass an operator tunes here and the one a live turn runs are
    # the same pass rather than two that resemble each other.
    body = fact_fetch.build_body(index, turns[:last_user], msg.get("speaker", ""),
                                 msg["content"], window)
    # `finish` receives the candidate list because the generation holds only ordinals, and
    # an ordinal means nothing without the list it indexed.
    return [body], {"candidates": cands, "query": msg["content"]}


# Mirrors `graph.fold.KNOWLEDGE_FACETS`, read lazily so this module keeps importing no
# graph symbol at import time (server/ reaches sys.path only via `_load_tree`).
_KNOWLEDGE_FACETS = ("property", "event")


def _withheld_for(doc: dict, picked: list) -> list:
    """The `position`/`report` claims about the same subjects the picked facts belong to.

    Never offered to the pass and never injectable — this is only the *price tag* on §10's
    knowledge-facet restriction, shown so an operator can judge the rule against what it
    excludes rather than on principle. On this corpus 73% of claims are positions, so the
    number is usually larger than the blob.
    """
    subjects = {c.get("node") for c in (picked or []) if c.get("node")}
    if not subjects:
        return []
    out = [c for c in (doc.get("claims") or {}).values()
           if c.get("node") in subjects and c.get("facet") not in _KNOWLEDGE_FACETS]
    out.sort(key=lambda c: (c.get("node") or "", c.get("text") or ""))
    return out


def _fetch_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    import sys as _sys
    from pathlib import Path
    server_dir = Path(__file__).resolve().parent.parent.parent
    if str(server_dir) not in _sys.path:
        _sys.path.insert(0, str(server_dir))
    from graph import blob as graph_blob

    doc = _load_tree() or {"nodes": {}, "claims": {}, "occurrences": []}
    cands = (context or {}).get("candidates") or {"claims": []}
    sel = graph_blob.parse_claim_selection(raws[0] if raws else "", cands)
    out = graph_blob.render_claims(doc, sel["picked"])

    lines: list = []
    lines.append("PICKED: " + (", ".join(f"[{k}]" for k in sel["numbers"])
                               if sel["numbers"] else "(nothing)"))
    if sel["out_of_range"]:
        # A number outside the list means the pass invented a line rather than reading one.
        # A finding about the pass, so it is named rather than quietly dropped.
        lines.append("OUT OF RANGE: " + ", ".join(str(k) for k in sel["out_of_range"][:8])
                     + f" (the list offered 1..{len(cands.get('claims') or [])})")
    lines.append(f"OFFERED: {cands.get('n', 0)} fact(s) of "
                 f"{cands.get('total_knowledge', 0)} on record"
                 + (f" · {cands['dropped_stale']} dropped as stale "
                    f"(TIL older than {cands.get('til_max_age_days')}d)"
                    if cands.get("dropped_stale") else "")
                 + (f" · {cands['dropped_capped']} dropped by the cap"
                    if cands.get("dropped_capped") else ""))
    lines.append("")
    lines.append("── the blob (what a live turn would inject) ──")
    lines.append("")
    if out["text"]:
        lines.append(out["text"])
    elif sel["none"]:
        # A deliberate refusal is a correct answer here and must read as one — the prompt
        # says most messages need nothing, so "picked nothing" is not a failed run.
        lines.append("(nothing picked — the pass judged the message needs no fact "
                     "looked up, which the prompt names as a correct answer)")
    else:
        lines.append("(empty)")
    lines.append("")

    # What the knowledge-facet restriction cost on THIS pick: the positions and reports the
    # same subjects hold, which the candidate list never offered. This is the argument §10
    # invites, and an operator judging whether the restriction is right has to see its
    # price rather than take it on faith.
    withheld = _withheld_for(doc, sel["picked"])
    lines.append(f"── withheld from the list: {len(withheld)} claim(s) about the same "
                 f"subjects ──")
    if withheld:
        lines.append("(records of what someone said, not knowledge — never offered to the "
                     "pass. Rendered here attributed, as they would have to be.)")
        lines.append("")
        for c in withheld[:12]:
            lines.append("  - " + graph_blob.attribution_line(doc, c))
        if len(withheld) > 12:
            lines.append(f"  ... and {len(withheld) - 12} more")

    return {"records": out["claims"], "count": out["n_rendered"],
            "counts": {"picked": len(sel["picked"]), "offered": cands.get("n", 0),
                       "on_record": cands.get("total_knowledge", 0),
                       "stale": cands.get("dropped_stale", 0),
                       "withheld": len(withheld),
                       "out_of_range": len(sel["out_of_range"])},
            "blob": out["text"], "picked": sel["numbers"], "lines": lines}


_FACT_FETCH = ModuleSpec(
    name="fact_fetch",
    label="Fetch facts (stage 1 of a chat turn)",
    prompt_file="fact_fetch_prompt.txt",
    # Small on purpose. The answer is at most 6 ids; a budget that would fit prose is a
    # budget a pass can drift into using.
    max_new_tokens="512",
    build=_fetch_build,
    finish=_fetch_finish,
    describe_output="picked nodes → a facts blob (returned, never injected)",
    injectables=(),
    # OFF, for chat_facts' reason: the output is a fixed-template list whose lines share a
    # `person:`/`unknown:` prefix, which is most of the guard's 12-token window.
    stop_on_repeat=False,
    # The whole design decision, in one field — see the block comment above.
    disable_thinking=True,
    source="chat",
)

MODULES: dict[str, ModuleSpec] = {
    _CHAT_FACTS.name: _CHAT_FACTS,
    _CHAT_SUMMARY.name: _CHAT_SUMMARY,
    _TIL_FACTS.name: _TIL_FACTS,
    _FACT_FETCH.name: _FACT_FETCH,
}


# ── input sources ─────────────────────────────────────────────────────────────
# What a module can be run against, per `ModuleSpec.source`. Each entry answers the two
# questions the run and the client respectively ask: load ONE by id, and list what there
# is. Adding a third lane is an entry here plus `source=` on the spec — the run loop, the
# streaming, the refusals and the write-nothing contract are all source-blind.
#
# `units`/`unit_label` exist because "how big is this input" has no single name across
# lanes: a transcript is measured in exchanges and an article in characters, and the
# reading stage line has to say which without the client knowing what it is looking at.

def _load_chat_input(ident: str) -> tuple:
    """One chat transcript by filename. Path-guarded to the chats dir."""
    path = (_CHATS_DIR / ident).resolve()
    if path.parent != _CHATS_DIR.resolve():
        raise ValueError("path traversal")
    if not path.exists():
        raise FileNotFoundError(ident)
    doc = json.loads(path.read_text(encoding="utf-8"))
    n = len(doc.get("exchanges") or [])
    return doc, {"units": n, "unit_label": "exchanges", "empty": not n,
                 "empty_reason": f"{ident} has no exchanges."}


def _load_til_input(ident: str) -> tuple:
    """One fetched article / news digest by `<kind>/<name>` id. Guarded to the snippets tree."""
    from core import til_facts
    if _TIL_SNIPPETS_DIR is None:
        raise FileNotFoundError("no snippets dir configured")
    record, kind = til_facts.load_input(_TIL_SNIPPETS_DIR, ident)
    # The kind decides the pass's framing (a dated digest of events vs a standing
    # description of one subject), and it is a property of WHERE the snippet lives rather
    # than of the record, so it is carried on the doc the builder receives.
    record["_kind"] = kind
    n = len(record.get("text") or "")
    return record, {"units": n, "unit_label": "characters", "empty": not n,
                    "empty_reason": f"{ident} has no text."}


def _list_chat_inputs() -> list:
    from core.chat_sidecar import is_chat_session_json
    if _CHATS_DIR is None or not _CHATS_DIR.is_dir():
        return []
    out = []
    for p in sorted(_CHATS_DIR.glob("*.json"), reverse=True):
        # The ONE definition of "this file is a transcript" — a chat's stem owns several
        # sidecars, and a listing that offered one would let a module read Ava's own
        # summary back as though it were the conversation. Takes a Path, not a name.
        if not is_chat_session_json(p):
            continue
        out.append({"id": p.name, "label": p.stem, "sub": "chat"})
    return out


def _list_til_inputs() -> list:
    from core import til_facts
    if _TIL_SNIPPETS_DIR is None:
        return []
    return til_facts.list_inputs(_TIL_SNIPPETS_DIR)


_INPUT_SOURCES: dict = {
    "chat": {"load": _load_chat_input, "list": _list_chat_inputs,
             "noun": "chat", "select": "Select a chat first."},
    "til": {"load": _load_til_input, "list": _list_til_inputs,
            "noun": "text", "select": "Select an article or digest first."},
}


def list_inputs(source: str) -> list:
    """What a module of this *source* can be run against. ``[]`` for an unknown source."""
    entry = _INPUT_SOURCES.get(str(source or "").strip())
    if entry is None:
        return []
    try:
        return entry["list"]()
    except Exception:
        traceback.print_exc()
        return []


def load_module_prompt(spec: ModuleSpec) -> str:
    """The module's prompt as it stands on disk, or "" when the file is missing/empty."""
    try:
        if _PROMPTS_DIR is not None:
            path = _PROMPTS_DIR / spec.prompt_file
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def catalogue() -> list[dict]:
    """The registry as the client sees it: one entry per module, prompt text included."""
    return [
        {
            "name": spec.name,
            "label": spec.label,
            "prompt_file": spec.prompt_file,
            "prompt": load_module_prompt(spec),
            "injectables": list(spec.injectables),
            "output": spec.describe_output,
            "max_new_tokens": spec.max_new_tokens,
            # Which input list this module runs against — the client asks the server for
            # it (`list_module_inputs`) rather than knowing per module.
            "source": spec.source,
        }
        for spec in MODULES.values()
    ]


# ── the run ───────────────────────────────────────────────────────────────────

def run_module_blocking(
    *,
    module: str,
    filename: str,
    prompt: str = "",
    inject: tuple = (),
    temperature: float = 0.7,
    top_p: float = 0.95,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run one module against one chat and return what it produced. **Writes nothing.**

    Runs on the single GPU executor thread (blocking). *prompt* overrides the module's
    on-disk prompt for this run only — the whole point of the workbench is to try one
    without editing the file. Nothing is injected: the pass sees the module prompt and the
    transcript, and that is all.
    """
    global _module_run_active

    def _stage(**info) -> None:
        if on_stage is not None:
            try:
                on_stage(info)
            except Exception:
                pass

    spec = MODULES.get(str(module or "").strip())
    if spec is None:
        return {"skipped": "unknown_module", "message": f"No such module: {module!r}"}
    if _runtime.model is None:
        return {"skipped": "no_model"}

    # Refuse rather than ignore — see ModuleSpec.injectables.
    unsupported = [i for i in (inject or ()) if i not in spec.injectables]
    if unsupported:
        return {"skipped": "unsupported_injectable",
                "message": ("This module injects nothing yet; cannot honour: "
                            + ", ".join(map(str, unsupported)))}

    src = _INPUT_SOURCES.get(spec.source)
    if src is None:
        return {"skipped": "unknown_source",
                "message": f"Module {spec.name!r} declares an unknown input source "
                           f"{spec.source!r}."}

    _module_run_active = True
    try:
        try:
            session, meta = src["load"](filename)
        except Exception as e:
            return {"skipped": "bad_session",
                    "message": f"Could not read {filename}: {type(e).__name__}: {e}"}
        if meta.get("empty"):
            # Kept under the historical key: an empty input is the same refusal whatever
            # the lane, and only the sentence describing it is per-source.
            return {"skipped": "empty_session", "message": meta["empty_reason"]}

        system_prompt = (prompt or "").strip() or load_module_prompt(spec)
        if not system_prompt:
            return {"skipped": "no_prompt",
                    "message": f"No prompt supplied and {spec.prompt_file} is missing."}

        window = _reflect_window()
        _stage(stage="reading", session=filename, units=meta["units"],
               unit_label=meta["unit_label"], exchanges=meta["units"])
        try:
            built = spec.build(session, window, _runtime.tokenizer)
            # A build may return blocks alone, or `(blocks, context)` when its `finish`
            # needs something the generation does not carry back. Optional so the three
            # modules that need no context stay exactly as they were.
            blocks, build_context = (built if isinstance(built, tuple) else (built, {}))
        except ModuleInputError as e:
            # The module refusing its own input, with a reason worth naming — see the class.
            return {"skipped": "no_content", "message": str(e)}
        if not blocks:
            return {"skipped": "no_content",
                    "message": f"{filename} rendered no readable {src['noun']} content."}

        rag = _get_rag() if _get_rag is not None else None
        activity_log.set_ambient_label(f"module:{spec.name}")
        generate = _make_sync_reflect_generate(rag)

        raws: list = []
        truncated = False
        stopped_on_loop = False
        cut_before_answer = False
        for i, content in enumerate(blocks, 1):
            _stage(stage="generating", session=filename, module=spec.name,
                   part=i, parts=len(blocks))
            if len(blocks) > 1 and on_chunk is not None and i > 1:
                # The blocks are separate generations joined into one value; without a
                # marker the streamed text reads as a single runaway.
                on_chunk(f"\n\n── part {i} of {len(blocks)} ──\n\n")
            raw = generate(
                content, system_prompt,
                temperature=float(temperature), top_p=float(top_p),
                max_new_tokens_setting=spec.max_new_tokens,
                # v0: nothing injected. `disable_rag` skips retrieval entirely, so the
                # assembled system message is the module prompt alone.
                disable_rag=True,
                stop_on_repeat=spec.stop_on_repeat,
                degen_stop=spec.degen_stop,
                # Also turns off the channel prefill: the seam gates it on
                # `force_think and not disable_thinking`, so a thinking-off pass is not
                # handed an opener it was told not to use.
                disable_thinking=spec.disable_thinking,
                on_chunk=on_chunk,
            )
            raws.append(raw or "")
            # Any block hitting a wall taints the whole result: these are parts of ONE
            # value, so a clean part 1 does not make a cut part 2 acceptable.
            part_truncated = bool(getattr(generate, "last_truncated", None))
            truncated = truncated or part_truncated
            stopped_on_loop = stopped_on_loop or bool(getattr(generate, "last_loop", None))
            cut_before_answer = cut_before_answer or reasoning_text.truncated_before_answer(
                raw or "", part_truncated)

        result = spec.finish(raws, truncated=truncated, context=build_context)
        _stage(stage="parsed", count=int(result.get("count") or 0))

        return {
            "ok": True,
            "module": spec.name,
            "source": spec.source,
            "session": filename,
            # How big the input was, in whatever unit its lane measures (a transcript in
            # exchanges, an article in characters). `exchanges` is kept as the historical
            # alias so an older client reads a number rather than nothing.
            "units": meta["units"],
            "unit_label": meta["unit_label"],
            "exchanges": meta["units"],
            "parts": len(blocks),
            "raw": "\n\n".join(raws),
            **result,
            "truncated": truncated,
            "stopped_on_loop": stopped_on_loop,
            # A pass cut inside its <think> has no answer region at all, so an empty
            # result means "the pass failed", not "this chat established nothing" — a
            # distinction the operator must not have to infer from a bare zero.
            "cut_before_answer": cut_before_answer,
            # The LAST part's accounting — the callable is re-stamped per generation. On a
            # single-block module (the common case) that is the whole run; on a chunked one
            # the client labels it as the last part rather than implying it is a total.
            "input_tokens": getattr(generate, "last_input_tokens", None),
            "max_new_tokens": getattr(generate, "last_max_new_tokens", None),
            "context_length": getattr(generate, "last_context_length", None),
            "written": False,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        _module_run_active = False


# ── handlers ──────────────────────────────────────────────────────────────────

async def handle_list_modules(ws, msg: dict) -> None:
    """Answer ``list_modules`` with the registry + each module's current prompt text."""
    await _send(ws, {"type": "modules_list", "modules": catalogue()})


async def handle_list_module_inputs(ws, msg: dict) -> None:
    """Answer ``list_module_inputs`` with what a module of this *source* can run against.

    A separate RPC rather than a field on ``modules_list`` because the two have different
    sizes and lifetimes: the registry is three small entries and the input list grows with
    the corpus. It is keyed on the SOURCE, not the module, so two modules reading the same
    kind of thing share one listing and one fetch.
    """
    source = str(msg.get("source") or "chat")
    await _send(ws, {"type": "module_inputs", "source": source,
                     "inputs": list_inputs(source)})


async def handle_run_module(ws, msg: dict) -> None:
    """Run one module against one chat and stream it back (Modules tab "Simulate").

    Writes nothing. Protocol: ``module_stage`` (phase markers) + ``module_chunk``
    (generation deltas, ``<think>`` included), finishing with ``module_done``.
    """
    loop = asyncio.get_event_loop()
    if _module_run_active or (_host_busy is not None and _host_busy()):
        await _send(ws, {"type": "module_done", "skipped": "busy",
                         "message": "Another GPU job is in progress — try again once it "
                                    "finishes."})
        return
    if _runtime.model is None:
        await _send(ws, {"type": "module_done", "skipped": "no_model",
                         "message": "No model loaded — load one from the Chat tab first."})
        return

    module = str(msg.get("module") or "")
    filename = str(msg.get("filename") or "")
    if not filename:
        spec = MODULES.get(module)
        src = _INPUT_SOURCES.get(spec.source) if spec else None
        await _send(ws, {"type": "module_done", "skipped": "no_session",
                         "message": (src or {}).get("select", "Select an input first.")})
        return
    prompt = str(msg.get("prompt") or "")
    inject = tuple(msg.get("inject") or ())
    temperature = float(msg.get("temperature", 0.7))
    top_p = float(msg.get("top_p", 0.95))

    def _on_chunk(delta: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "module_chunk", "text": delta}), loop)

    def _on_stage(info: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            _send(ws, {"type": "module_stage", **info}), loop)

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: run_module_blocking(
                module=module, filename=filename, prompt=prompt, inject=inject,
                temperature=temperature, top_p=top_p,
                on_chunk=_on_chunk, on_stage=_on_stage),
        )
    except Exception as e:
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    if _mark_activity is not None:
        try:
            _mark_activity()
        except Exception:
            pass
    payload = {"type": "module_done"}
    payload.update(result)
    await _send(ws, payload)


# ── GPU-free self-test ─────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise the registry + the no-model/unknown-module refusals.

    Run: ``python -m core.modules``."""
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    print("registry")
    check("the registry", sorted(MODULES),
          ["chat_facts", "chat_summary", "fact_fetch", "til_facts"])
    check("nothing injects yet",
          [s.injectables for s in MODULES.values()], [()] * len(MODULES))
    check("every module can render its own result without client help",
          all(callable(s.build) and callable(s.finish) for s in MODULES.values()), True)

    print("\nfinish: facts")
    facts = _facts_finish(["[fact] (about: Artemy) (class: standing) (entities: Haifa) "
                           "His sister lives in Haifa.\n"
                           "[fact] (about: self) (class: stated) Prefers short replies."],
                          truncated=False)
    check("records parsed", facts["count"], 2)
    check("counts by class", facts["counts"], {"standing": 1, "stated": 1})
    check("entities surface in the rendered line",
          "entities: Haifa" in "\n".join(facts["lines"]), True)
    check("the reserved self subject renders readably",
          "about _self" in "\n".join(facts["lines"]), True)
    # The session rides the build context so `finish` can fold participant labels onto
    # the recorded spelling, exactly as the production parse does.
    folded = _facts_finish(
        ["[fact] (about: Artemy) (class: standing) Keeps odd hours."],
        truncated=False,
        context={"session": {"user": "artemyvo", "exchanges": []}})
    check("a participant label folds to the recorded spelling",
          folded["records"][0]["subject"], "artemyvo")
    check("no session context ⇒ the plain resolver",
          facts["records"][0]["subject"], "artemy")

    print("\nfinish: summary")
    gist = _summary_finish(
        ["<think>weighing it</think>They talked about the retrieval change and where the "
         "block budget should go, and left the subject-cap question open for later.",
         "<think>part two</think>The second half turned to how the gist is stored and why "
         "it outlives the state sidecar, which settled the split."],
        truncated=False)
    check("parts joined", gist["count"], 2)
    check("thinking stripped", "<think>" in gist["text"], False)
    check("prose survives", gist["text"].startswith("They talked about"), True)
    rejected = _summary_finish(["## WEIGHTS\n[fact] a\n[fact] b"], truncated=False)
    check("a structured dump is refused by the sanitizer", rejected["count"], 0)
    check("...and the refusal is reported, not silent", rejected["rejected"], 1)
    check("...visibly", "rejected by the gist sanitizer" in "\n".join(rejected["lines"]),
          True)

    print("\nrefusals (no model loaded)")
    check("unknown module", run_module_blocking(module="nope", filename="x.json")
          .get("skipped"), "unknown_module")
    check("no model", run_module_blocking(module="chat_facts", filename="x.json")
          .get("skipped"), "no_model")

    print("\nround-trip: the spec still agrees with the production pass")
    # The registry duplicates `reflection_runner`'s chat-facts pass until the runner
    # becomes a caller of it. Duplication is only safe if a divergence is caught, so the
    # two facts that would silently change the output — the closing question and the loop
    # guard — are asserted against the runner's source here.
    from pathlib import Path
    runner = (Path(__file__).parent / "reflection_runner.py").read_text(encoding="utf-8")
    tail = "Record what was said, not what it suggests about anyone."
    check("closing question still lives in reflection_runner", tail in runner, True)
    check("and in the spec", tail in _CHAT_FACTS_CLOSING, True)
    check("the loop guard is off in the spec", _CHAT_FACTS.stop_on_repeat, False)
    check("...and off in the production pass",
          "stop_on_repeat=False" in runner, True)
    check("the diversity guard is off in the spec", _CHAT_FACTS.degen_stop, False)
    check("...and off in the production pass",
          "degen_stop=False" in runner, True)
    # The reversed-session note has ONE definition (core.chat_facts), so this asserts both
    # sides reach for the shared source rather than that two copies still match.
    check("...and the production pass uses the shared reversed-session note",
          "chat_facts_mod.AVA_INITIATED_NOTE" in runner, True)
    # Likewise the participants note and the session-aware subject fold: one definition
    # each (core.chat_facts), asserted used on both sides rather than copied.
    check("...and the shared participants note",
          "chat_facts_mod.participants_note(session)" in runner, True)
    check("...and the shared session subject fold",
          "chat_facts_mod.make_subject_fn(session)" in runner, True)
    # chat_summary's budget mirrors the run default its production twin inherits.
    check("summary budget matches the run default",
          f'_DEFAULT_MAX_NEW_TOKENS = "{_CHAT_SUMMARY.max_new_tokens}"' in runner, True)
    check("...and it reads the summary prompt the runner loads",
          _CHAT_SUMMARY.prompt_file, "summary_prompt.txt")
    # The closing has ONE definition (reflection_chunking.SUMMARY_CLOSING), so — as with
    # the reversed-session note above — this asserts both sides reach for the shared
    # source rather than that two copies still match. They apply it differently on
    # purpose (baked into the chunks here, appended at the call site there, since
    # production's chunks are shared with consolidation), which is exactly the kind of
    # split that invites a second copy of the text later.
    check("the production summary pass appends the shared closing",
          "append_closing(content, SUMMARY_CLOSING)" in runner, True)
    check("...and the module bakes the same one into its chunks",
          "closing=SUMMARY_CLOSING" in Path(__file__).read_text(encoding="utf-8"), True)
    from core.reflection_chunking import (
        SUMMARY_CLOSING as _closing, append_closing as _append, format_chunk_content)
    check("the closing names the transcript a record",
          "record, not a turn addressed to you" in _closing, True)
    check("consolidation is unchanged — no closing by default",
          format_chunk_content({"exchanges": []}, [], 1, 1, 1, 1).rstrip()
          .endswith(_closing), False)
    check("...and the summary chunk ends on the task",
          _append("Artemy: so what do you think?\nMe: I think so.",
                  _closing).rstrip().endswith(_closing), True)

    print("\nthe reflect seam accepts the flag")
    import inspect
    from core import generation
    src = inspect.getsource(generation._make_sync_reflect_generate)
    check("generate_fn takes stop_on_repeat", "stop_on_repeat: bool = True" in src, True)
    check("...and forwards it, not a literal",
          "stop_on_repeat=stop_on_repeat" in src, True)
    check("generate_fn takes degen_stop",
          "degen_stop: Optional[bool] = None" in src, True)
    check("...and a caller's False overrides the box default",
          '"degen_stop": bool(degen_stop)' in src, True)

    print("\ninput sources")
    check("every module declares a source the registry can serve",
          sorted({s.source for s in MODULES.values()} - set(_INPUT_SOURCES)), [])
    check("every source can load and list",
          all(callable(e.get("load")) and callable(e.get("list"))
              for e in _INPUT_SOURCES.values()), True)
    check("the catalogue carries it, so the client need not know per module",
          sorted({m["source"] for m in catalogue()}), ["chat", "til"])
    check("an unknown source lists nothing rather than raising",
          list_inputs("nonsense"), [])
    # Unconfigured is the honest degrade, not a crash: a box that never wired the
    # snippets tree has no articles, and saying so is better than a traceback.
    _saved, globals()["_TIL_SNIPPETS_DIR"] = _TIL_SNIPPETS_DIR, None
    check("an unwired til tree lists nothing", list_inputs("til"), [])
    globals()["_TIL_SNIPPETS_DIR"] = _saved

    print("\ntil_facts mirrors its production twin (til_wander.run_facts_pass)")
    from core import til_wander, til_facts
    prod = inspect.getsource(til_wander.run_facts_pass)
    check("same prompt file", f'"{_TIL_FACTS.prompt_file}"' in prod, True)
    check("same generation budget",
          f'max_new_tokens_setting="{_TIL_FACTS.max_new_tokens}"' in prod, True)
    check("both turn the verbatim loop guard off",
          (_TIL_FACTS.stop_on_repeat, "stop_on_repeat=False" in prod), (False, True))
    check("...and the diversity guard with it",
          (_TIL_FACTS.degen_stop, "degen_stop=False" in prod), (False, True))
    check("both build blocks through the one definition",
          "til_facts.build_reading_blocks" in prod, True)
    # The kind decides the framing and rides on the doc, so a mis-wired loader would
    # silently read a digest as an article.
    blocks = _TIL_FACTS.build({"_kind": "news", "date": "2026-08-08", "text": "x y z"},
                              24576, None)
    check("the loader's kind reaches the builder",
          "digest of world events" in (blocks[0] if blocks else ""), True)
    # `finish` is source-blind like the chat pair: raws in, display lines out.
    fin = _TIL_FACTS.finish(
        ["[fact] (about: New York) (class: standing) Has a subway."], truncated=False)
    check("...and parses with the TIL subject rule, not the chat one",
          fin["records"][0]["subject"], "New York")
    check("the display line is rendered by the module",
          fin["lines"][0].endswith("about New York: Has a subway."), True)
    # Only the LAST block can be cut by the cap; dropping an earlier block's final fact
    # would silently lose one per seam on every multi-block article.
    two = _TIL_FACTS.finish(["[fact] (about: A) one.\n[fact] (about: B) two.",
                             "[fact] (about: C) three.\n[fact] (about: D) four."],
                            truncated=True)
    check("truncation drops one fact in total, from the last block only",
          [r["subject"] for r in two["records"]], ["A", "B", "C"])

    print("\nfact_fetch: the seam accepts a thinking-off pass")
    check("the spec carries it as data", _FACT_FETCH.disable_thinking, True)
    check("...and it is the only module that does",
          [s.name for s in MODULES.values() if s.disable_thinking], ["fact_fetch"])
    check("the reflect seam takes the flag", "disable_thinking: bool = False" in src, True)
    check("...and the run forwards the spec's value, not a literal",
          "disable_thinking=spec.disable_thinking"
          in Path(__file__).read_text(encoding="utf-8"), True)
    # The prefill is what a thinking-off pass must not get; the seam gates it on both
    # flags, so this asserts the condition the module relies on rather than restating it.
    check("...and the seam's prefill is gated on it",
          "if force_think and not disable_thinking and fam.think_prefill" in
          inspect.getsource(generation._make_sync_reflect_generate), True)

    print("\nfact_fetch: build refuses its input with a reason, not a generic message")
    # `build` returning [] would surface as "rendered no readable chat content", which for
    # this module points an operator at the transcript when the problem is the tree.
    def _refusal(session, **kw):
        try:
            _fetch_build(session, 24576, None)
            return None
        except ModuleInputError as e:
            return str(e)
        except Exception as e:
            return f"WRONG EXCEPTION: {type(e).__name__}: {e}"
    # `graph/` reaches sys.path through the module's own helper (server/ is not on it for a
    # process rooted at inference/), so call that before importing what it makes importable.
    _load_tree()
    import graph.store as _gstore
    _real_read, _gstore.read_tree = _gstore.read_tree, lambda *a, **k: None
    check("no tree ⇒ a sentence naming the tree",
          (_refusal({"exchanges": [{"user_prompt": "hi"}]}) or "")
          .startswith("No facts tree on this box."), True)
    _gstore.read_tree = _real_read
    check("...and the run turns that into a skip, not a traceback",
          "except ModuleInputError as e:" in Path(__file__).read_text(encoding="utf-8"),
          True)

    print("\nfact_fetch: a synthetic impulse is not a message that arrived")
    # The observed failure: on an Ava-initiated reach-out the only user-slot turn is the
    # stage direction she wrote to herself, and this pass took it as the arriving message
    # and fetched the friend it names. Every other reader on the box already excludes it.
    reachout = {"initiated_by": "ava", "exchanges": [
        {"speaker": "(initiative)", "user_prompt": "About 7 hours had passed since you "
         "last spoke with your friend.", "assistant_response": "I've been thinking..."}]}
    check("an Ava-initiated opener is refused",
          "no real user turn" in (_refusal(reachout) or ""), True)
    # ...and refused by BOTH halves independently, since a transcript predating the
    # `(initiative)` convention carries no speaker label to key on.
    check("...by the exchange-0 rule alone (no speaker label)",
          fact_fetch.last_real_user_turn({"initiated_by": "ava"},
                               [{"role": "user", "speaker": "", "content": "impulse"},
                                {"role": "assistant", "content": "x"}]), None)
    check("...and by the speaker rule alone (no initiated_by)",
          fact_fetch.last_real_user_turn({}, [{"role": "user", "speaker": "(setting)",
                                     "content": "framing"}]), None)
    # The point of the rule is that it narrows, not that it refuses: once the other side
    # actually replies, the same reach-out chat becomes a valid input.
    answered = {"initiated_by": "ava", "exchanges": [
        dict(reachout["exchanges"][0]),
        {"speaker": "artemyvo", "user_prompt": "yes, go on", "assistant_response": "ok"}]}
    check("an ANSWERED reach-out is accepted, at the real turn",
          fact_fetch.last_real_user_turn(answered, [
              {"role": "user", "speaker": "(initiative)", "content": "impulse"},
              {"role": "assistant", "content": "opener"},
              {"role": "user", "speaker": "artemyvo", "content": "yes, go on"}]), 2)

    print("\nfact_fetch: the model selects claims, and the list travels to finish")
    src_self = Path(__file__).read_text(encoding="utf-8")
    sess = {"exchanges": [
        {"speaker": "artemyvo", "user_prompt": "first thing", "assistant_response": "r1"},
        {"speaker": "artemyvo", "user_prompt": "second thing", "assistant_response": "r2"},
        {"speaker": "artemyvo", "user_prompt": "the message", "assistant_response": "r3"}]}
    built = _fetch_build(sess, 24576, None)
    check("build returns (blocks, context)", isinstance(built, tuple), True)
    body, ctx = built[0][0], built[1]
    # `finish` sees only ordinals, which mean nothing without the list they indexed.
    check("...carrying the candidate list", isinstance(ctx.get("candidates"), dict), True)
    check("the pass is asked for numbers, not node ids", "FACTS:" in body, True)

    print("\nfact_fetch: the WHOLE prior conversation, minus the answer being simulated")
    check("an early turn is present, not just a tail", "first thing" in body, True)
    check("...and the turn before the message", "second thing" in body, True)
    check("the arriving message is called out", "the message" in body, True)
    # The reply to the arriving message is what the fetch is FOR; showing it in simulation
    # would hand the pass the answer.
    check("its reply is withheld", "r3" in body, False)
    check("...while earlier replies are context", "r1" in body and "r2" in body, True)

    print("\nfact_fetch: the run's plumbing")
    check("the run unpacks the tuple shape",
          "blocks, build_context = (built if isinstance(built, tuple)" in src_self, True)
    check("...and forwards context to finish",
          "spec.finish(raws, truncated=truncated, context=build_context)" in src_self, True)
    for _spec in MODULES.values():
        check(f"{_spec.name}.finish accepts context",
              "context" in inspect.signature(_spec.finish).parameters, True)

    print("\nfact_fetch: the renderer is the one definition (graph.blob)")
    from graph import blob as _blob
    # Asserted on the FUNCTION, not on this file's text: the earlier form searched the
    # source for the call and matched the search string itself, so it passed after the
    # call it was checking had been replaced.
    check("the module renders through graph.blob, never inline",
          "graph_blob.render_claims(" in inspect.getsource(_fetch_finish), True)
    check("...and builds the candidate list there too",
          "graph_blob.claim_candidates(" in inspect.getsource(_fetch_build), True)
    # The blob itself is never assembled here — it is taken whole from the renderer. (The
    # withheld list IS rendered here, but that is an operator report, never injected.)
    check("the blob text comes whole from the renderer",
          'lines.append(out["text"])' in inspect.getsource(_fetch_finish), True)
    # The two §10 rules now hold by CONSTRUCTION — the list is filtered before the pass
    # sees it, so a forbidden claim cannot be picked rather than being rejected after.
    check("the facet gate lives in the candidate builder",
          "KNOWLEDGE_FACETS" in inspect.getsource(_blob.claim_candidates), True)
    check("...and so does the self-subtree rule",
          "is_self" in inspect.getsource(_blob.claim_candidates), True)
    # `finish` is display-only, so it must survive a tree that is gone since `build` ran.
    _gstore.read_tree = lambda *a, **k: None
    empty = _fetch_finish(["NODES:\nNONE"], truncated=False)
    check("a vanished tree yields an empty report, not a crash", empty["count"], 0)
    _gstore.read_tree = _real_read

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
