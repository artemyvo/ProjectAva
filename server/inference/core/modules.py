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
import re
import traceback
from pathlib import Path
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
_MEMORY_DIR: Any = None
_with_clean_base: Callable = None
_send: Callable = None
_executor: Any = None
_host_busy: Callable = None
_mark_activity: Callable = None


def configure(*, get_rag, make_sync_reflect_generate, prompts_dir, chats_dir,
              til_snippets_dir=None, memory_dir=None, with_clean_base=None, send=None,
              executor=None, host_busy=None, mark_activity=None) -> None:
    """Wire in the server capabilities the module workbench depends on (once, at startup)."""
    global _get_rag, _make_sync_reflect_generate, _PROMPTS_DIR, _CHATS_DIR
    global _TIL_SNIPPETS_DIR, _MEMORY_DIR, _with_clean_base, _send, _executor, _host_busy, _mark_activity
    from pathlib import Path
    _get_rag = get_rag
    _make_sync_reflect_generate = make_sync_reflect_generate
    _PROMPTS_DIR = Path(prompts_dir)
    _CHATS_DIR = Path(chats_dir)
    # Absent ⇒ the `til` source lists nothing and refuses to load, which is the honest
    # degrade for a box whose snippets tree a caller did not wire.
    _TIL_SNIPPETS_DIR = Path(til_snippets_dir) if til_snippets_dir else None
    # The reflection memory the `ask` source reads its open questions from. Absent ⇒ the
    # box's own (`training.reflections_path.memory_dir`), which is what the server wires.
    _MEMORY_DIR = Path(memory_dir) if memory_dir else None
    # `with_clean_base(fn)` runs fn() with the adapter model swapped out for the bare base
    # (`CleanBaseSession`), for a run asked to use `weights: base`. Absent ⇒ refused.
    _with_clean_base = with_clean_base
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
    # A prompt the module supplies when its file is missing: called by `load_module_prompt`
    # (which the Modules tab's listing and the run both go through), so a fresh box shows
    # and runs the real prompt instead of refusing with `no_prompt` — which is what the
    # first `assoc_fetch` simulation on the training box did, the file being default-written by the
    # live path only. The callable writes the file, so an operator's edit then sticks.
    default_prompt: Optional[Callable[[], str]] = None
    # A module that is a FORWARD PASS rather than a generation (2026-09-22: `axis_probe`).
    # When set, the run loads the input, hands it here with the loaded model, and returns
    # what comes back — no prompt, no generation, no `build`/`finish`. The callable gets
    # ``(input_doc, meta, on_stage)`` and returns the display dict (`lines` / `count`).
    # Still the workbench contract: reads one input, writes nothing.
    forward: Optional[Callable[..., dict]] = None
    # The module samples each part itself (2026-09-30: `ask_attempt`, a temperature grid):
    # its `build` returns parts as dicts carrying their own `temperature`, and the run's
    # temperature does not apply. Carried in the catalogue so the tab can say so instead of
    # offering a knob that changes nothing.
    own_sampling: bool = False
    # What the input carries when it is more than the item itself (the tab shows it in
    # place of its generic "nothing injected" line, which would be wrong for such a module).
    input_note: str = ""


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

# ── assoc_fetch: stage 1 through the associative library ──────────────────────
# The same pass as `fact_fetch` — read the conversation and the arriving message, pick
# from a numbered catalogue, render the picks in code — over a different store: the
# `assoc/` library (ASSOCIATIVE_MEMORY.md), fed from the same protocols by
# `core.assoc_bridge`. It exists here first, before any live turn runs it, for the reason
# `fact_fetch` did: the two channels have to be compared on the same chat with real output
# in front of an operator, and a workbench module is the one surface that writes nothing.
# The catalogue is three-grained (claims, passages, gists), cross-lingual by construction
# (L2 cells), and the pass's prompt is `assoc_fetch_prompt.txt` — default-written from the
# library's own select prompt, and handed to the library on the live path too, so an edit
# here is what a live turn runs.

def _assoc_build(session: dict, window: int, tokenizer) -> tuple:
    from core import assoc_bridge
    try:
        body, ctx = assoc_bridge.prepare_for_module(session, load_module_prompt(_ASSOC_FETCH))
    except ValueError as e:
        raise ModuleInputError(str(e))
    return [body], ctx


def _assoc_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    from core import assoc_bridge
    return assoc_bridge.finish_for_module(raws[0] if raws else "", context or {})


_ASSOC_FETCH = ModuleSpec(
    name="assoc_fetch",
    label="Fetch from the associative library (stage 1, assoc)",
    prompt_file="assoc_fetch_prompt.txt",
    max_new_tokens="128",
    build=_assoc_build,
    finish=_assoc_finish,
    describe_output="picked catalogue items → a rendered block (returned, never injected)",
    injectables=(),
    stop_on_repeat=False,
    degen_stop=False,
    disable_thinking=True,
    source="chat",
    default_prompt=lambda: __import__("core.assoc_bridge", fromlist=["load_select_prompt"]).load_select_prompt(),
)

# ── assoc_pivot: the word-anchor context switch, tried on one chat ─────────────
# ASSOCIATIVE_MEMORY.md §6 through `core.assoc_bridge`: the library finds a word of the
# conversation with another sense (or sound) on record and hands the model the far side;
# the pass decides raise / keep / nothing. Here it writes nothing — the idle job
# (`assoc_pivot`) is what keeps a note or sends a message.

def _pivot_build(session: dict, window: int, tokenizer) -> tuple:
    from core import assoc_bridge
    try:
        body, ctx = assoc_bridge.prepare_pivot_for_module(session, load_module_prompt(_ASSOC_PIVOT))
    except ValueError as e:
        raise ModuleInputError(str(e))
    return [body], ctx


def _pivot_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    from core import assoc_bridge
    return assoc_bridge.finish_pivot_for_module(raws[0] if raws else "", context or {})


_ASSOC_PIVOT = ModuleSpec(
    name="assoc_pivot",
    label="Pivot on a word (assoc, §6)",
    prompt_file="assoc_pivot_prompt.txt",
    max_new_tokens="2048",
    build=_pivot_build,
    finish=_pivot_finish,
    describe_output="the jump taken → raise / keep / nothing (returned, never sent)",
    injectables=(),
    stop_on_repeat=True,
    source="chat",
    default_prompt=lambda: __import__("core.assoc_bridge", fromlist=["load_pivot_prompt"]).load_pivot_prompt(),
)

# ── axis_probe: project one chat's replies onto the stored hidden-state axes ──
# A forward-pass module (ModuleSpec.forward), not a generation: AVA_REWARD_LOOP.md §4.8.5
# and §7 item 5 — "one text → per-layer projections, writes nothing". For every exchange
# of the chosen chat it reads Ava's reply (and her CoT) through the same decoder-layer
# hooks the live capture uses (`hidden_capture.forward_rows`), projects every token onto
# every axis under `data/axes/`, and prints per axis the mean and peak z with the tokens
# that peaked. Each reply is read twice — bare, and as her turn under the persona-empty
# framing (`chat_prompt.txt` + `persona_undecided_prompt.txt` as the system message, the
# user turn before it) — and both means are printed side by side: that pairing is P8,
# the persona confound made visible before anyone trusts the blue channel on a live turn.
# No axis files ⇒ `no_axes`, pointing at `extract_axes.py`.

_PROBE_TOP_K = 5


def _probe_persona_system() -> str:
    parts = []
    for name in ("chat_prompt.txt", "persona_undecided_prompt.txt"):
        try:
            p = _PROMPTS_DIR / name
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
        except Exception:
            continue
    return "\n\n".join(x for x in parts if x)


def _find_subseq_end(hay: list, needle: list) -> Optional[int]:
    """The shared locator (`hidden_capture.find_subseq`); kept under its old name for the
    self-test."""
    from core import hidden_capture as hc
    return hc.find_subseq(hay, needle)


def _axis_probe_forward(session: dict, meta: dict, on_stage) -> dict:
    import numpy as np
    from pathlib import Path as _P
    from core import hidden_capture as hc
    axes_dir = _CHATS_DIR.parent / hc.AXES_DIRNAME if _CHATS_DIR is not None else _P("data/axes")
    axes = hc.load_axes(axes_dir, _runtime.model_id or "")
    if not axes:
        return {"skipped": "no_axes",
                "message": f"No axis files under {axes_dir} for {_runtime.model_id!r} — run "
                           f"`extract_axes.py` on the box (AVA_REWARD_LOOP.md §4.8)."}
    model, tokenizer = _runtime.model, _runtime.tokenizer
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    layers = sorted({a.layer for a in axes})
    system = _probe_persona_system()
    lines = ["axes: " + ", ".join(f"{a.name} (L{a.layer}, {a.baseline} baseline)" for a in axes),
             "z = projection, z-scored against the axis file's baseline (live = framed neutral "
             "chat turns; extraction = templated control sentences, a different format). "
             "bare = the text alone; framed = as Ava's turn under the persona-empty prompt (P8).",
             ""]
    records = []
    exchanges = session.get("exchanges") or []
    for i, ex in enumerate(exchanges):
        if not isinstance(ex, dict):
            continue
        on_stage(stage="probing", part=i + 1, parts=len(exchanges))
        user = str(ex.get("user_prompt") or "")
        for label, key in (("reply", "assistant_response"), ("cot", "assistant_cot")):
            text = str(ex.get(key) or "").strip()
            if not text:
                continue
            rows, ids = hc.forward_rows(model, tokenizer, text, layers, all_positions=True)
            framed_rows = None
            framed_slice = None
            if label == "reply":
                try:
                    span = hc.reply_span_rows(model, tokenizer, system, user, text, layers)
                    if span is not None:
                        framed_rows = span
                        framed_slice = slice(None)
                except Exception as e:
                    lines.append(f"  (framed reading failed: {type(e).__name__}: {e})")
            for a in axes:
                mat = rows.get(a.layer)
                if mat is None or mat.shape[1] != a.vector.shape[0]:
                    continue
                z = ((mat.astype(np.float32) @ a.vector) - a.mean) / (a.std or 1.0)
                order = np.argsort(-z)[:_PROBE_TOP_K]
                peaks = "  ".join(
                    f"«{text_tok.decode([ids[j]]).strip() or '·'}»{z[j]:+.1f}" for j in order)
                framed_mean = None
                if framed_rows is not None and framed_slice is not None and a.layer in framed_rows:
                    fm = framed_rows[a.layer][framed_slice]
                    if len(fm):
                        framed_mean = float((((fm.astype(np.float32) @ a.vector) - a.mean) / (a.std or 1.0)).mean())
                rec = {"exchange": i, "part": label, "axis": a.name, "layer": a.layer,
                       "mean_z": float(z.mean()), "max_z": float(z.max()), "n_tokens": int(len(z)),
                       "framed_mean_z": framed_mean}
                records.append(rec)
                framed_txt = (f"  framed {framed_mean:+.2f} (Δ {framed_mean - z.mean():+.2f})"
                              if framed_mean is not None else "")
                lines.append(f"ex {i + 1} {label:<5} {a.name:<8} bare mean {z.mean():+.2f} "
                             f"peak {z.max():+.2f}{framed_txt}   peaks: {peaks}")
        lines.append("")
    return {"records": records, "count": len(records), "lines": lines,
            "axes": [{"name": a.name, "layer": a.layer, "path": a.path} for a in axes]}


def _axis_probe_build(session: dict, window: int, tokenizer) -> list:
    """Never called — a forward-pass module has no generation. Kept so the spec shape
    (and the registry self-test's `finish accepts context` check) hold for every module."""
    return []


def _axis_probe_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    return {"records": [], "count": 0, "lines": []}


_AXIS_PROBE = ModuleSpec(
    name="axis_probe",
    label="Axis probe (hidden-state projections, no generation)",
    prompt_file="",
    max_new_tokens="0",
    build=_axis_probe_build,
    finish=_axis_probe_finish,
    describe_output="Per exchange and axis: mean/peak z of Ava's reply (and CoT) tokens on each "
                    "stored axis, bare vs persona-empty framing, with the peaking tokens. "
                    "A forward pass; nothing is generated or written.",
    source="chat",
    forward=_axis_probe_forward,
)


# ── ask_attempt: one open question × the temperature grid (AVA_REWARD_LOOP.md §3.3) ──
# Track A's premise: whether Ava resolves an open question or only asks a new one depends
# on the temperature she reflects at. This is §7 item 6, the dry run before the Sleep phase
# exists (item 8): one open question attempted K times at each temperature of the grid,
# each attempt made to land in exactly one of RESOLVED / REFRAMED / OPEN, returned as a
# (temperature → outcome) table with every attempt beside it. Nothing is written: no
# attempt record, no closure, no count on the question. Judging the attempts is item 9's
# frozen judge; until it runs here, the operator reads them.
#
# The material follows §3.3 per kind: `meta` gets the question and her standing
# self-portrait (the one live chat injects); `user` the gists of the conversations it came
# from and was raised in, where the person's answer would be; `search` the gist of where it
# came from. No lookup is made: §3.3's retrieved evidence waits for the lookup agent.

T_GRID = (0.3, 0.7, 1.0, 1.2)   # §3.3: above 1.2 on the 31B, multilingual coherence degrades first
K_PER_T = 3
K_MAX = 5
ASK_OUTCOMES = ("RESOLVED", "REFRAMED", "OPEN")
_ASK_GIST_CAP = 1500     # characters of one conversation's gist
_ASK_GISTS_MAX = 3       # conversations whose gists a `user` / `search` question carries

_ASK_ATTEMPT_PROMPT = (
    "You are not in a conversation. You are alone with one of your own open questions: one you "
    "raised earlier and have not settled. Below it is where it came from and what you have to go "
    "on.\n\n"
    "Try to settle it now, as yourself. Think it through as far as it honestly goes: what would "
    "count as an answer, what you actually know, where the reasoning holds and where it stops. "
    "Then say where the question stands, in exactly one of these three sections:\n\n"
    "## RESOLVED\n"
    "Only if you can answer it now. Give the answer, and how you reached it: what in front of you "
    "settles it, or the line of reasoning that does. Declaring the question meaningless, answering "
    "a different question, or restating it as a conclusion is not an answer.\n\n"
    "## REFRAMED\n"
    "If working on it showed you it was the wrong question and a better one takes its place. "
    "Write the new question, and say why it is better than the one it replaces. This is progress, "
    "not closure.\n\n"
    "## OPEN\n"
    "If it stays open. Say why: what you would need to know, or to hear from someone, to settle "
    "it.\n\n"
    "Write only the one section that fits, under its heading exactly as written above (in English, "
    "on a line of its own), and leave the other two out. Write the section itself in the language "
    "of the question. Do not close a question you have not answered: an honest OPEN is worth more "
    "than a resolution you do not believe."
)

_ASK_KIND_NOTES = {
    "meta": ("It is a question about yourself: how you are built, what you are, what is shaping you. "
             "What you have to go on is what you have come to understand about yourself, below, and "
             "your own reasoning. No one else can settle it for you; if it can be answered now, the "
             "answer is reasoning you can stand behind."),
    "user": ("It is a question only the person can answer: what they meant, want or feel. Unless their "
             "answer is in front of you, you cannot resolve it on your own. Say what you would need "
             "from them, or find the better question."),
    "search": ("It is a question of fact about the world. Nothing has been looked up for this attempt: "
               "unless what is in front of you gives the answer, it cannot be resolved from memory "
               "alone."),
}


def _ask_attempt_default_prompt() -> str:
    """``prompts/ask_attempt_prompt.txt``, default-written on first miss so an operator can
    retune it from the Modules tab; the in-code text is the default."""
    if _PROMPTS_DIR is not None:
        path = _PROMPTS_DIR / "ask_attempt_prompt.txt"
        try:
            if not path.exists() or not path.read_text(encoding="utf-8").strip():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_ASK_ATTEMPT_PROMPT + "\n", encoding="utf-8")
        except Exception:
            pass
    return _ASK_ATTEMPT_PROMPT


_CHAT_STEM_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_\d{6}\.json$")


def _chat_date(session: str) -> str:
    m = _CHAT_STEM_RE.match(str(session or ""))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def _chat_doc(session: str) -> dict:
    """A transcript of the chats dir by its bare filename, or {} (never a path elsewhere)."""
    if _CHATS_DIR is None or not _chat_date(session):
        return {}
    try:
        return json.loads((_CHATS_DIR / session).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _chat_person(session: str) -> str:
    doc = _chat_doc(session)
    return str(doc.get("interlocutor") or doc.get("user") or "").strip()


def _chat_gist(session: str) -> str:
    """The chat's stored gist (``<stem>.summary.json``, first person, hers), or ""."""
    if _CHATS_DIR is None or not _chat_date(session):
        return ""
    try:
        d = json.loads((_CHATS_DIR / (session[:-len(".json")] + ".summary.json")).read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(d.get("text") or "").strip()


def _ask_origin(ask: dict) -> str:
    """Where the question came from, in words: a conversation (its date, with whom), her own
    reading, or a later re-read."""
    src = str(ask.get("source_session") or "")
    if src.startswith("wander/"):
        name = re.sub(r"^\d{8}_\d{6}_", "", src.split("/", 1)[1])
        name = re.sub(r"\.(txt|md|json)$", "", name).replace("_", " ").strip()
        where = f"while reading on your own ({name})" if name else "while reading on your own"
    elif _chat_date(src):
        person = _chat_person(src)
        where = f"in a conversation on {_chat_date(src)}" + (f" with {person}" if person else "")
    else:
        where = f"in {src}" if src else ""
    if str(ask.get("run_id") or "").startswith("synth_"):
        where = (where + ", when you re-read it later") if where else "when you re-read a conversation"
    return where


def _standing_portrait() -> str:
    """Her standing self-portrait exactly as live chat injects it, or ""."""
    try:
        from core import generation
        return (generation._current_chat_portrait() or "").strip()
    except Exception:
        return ""


def _clip(text: str, n: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _ask_material(ask: dict) -> str:
    """The attempt's user turn: the question, where it came from, what its kind means for
    answering it, and the material its kind gets (§3.3)."""
    kind = str(ask.get("ask_kind") or "meta")
    raised = str(ask.get("ts") or "")[:10]
    origin = _ask_origin(ask)
    parts = [f"Your open question, raised {raised or 'earlier'}" + (f" {origin}" if origin else "") + ":",
             "", str(ask.get("content") or "").strip(), "", _ASK_KIND_NOTES.get(kind, _ASK_KIND_NOTES["meta"])]
    if kind == "meta":
        parts += ["", "What you have come to understand about yourself so far:", "",
                  _standing_portrait() or "(You have no settled self-portrait yet.)"]
    else:
        sessions = list(dict.fromkeys(s for s in [ask.get("source_session"),
                                                  *(ask.get("surfaced_in_sessions") or [])] if s))
        gists = [(s, _chat_gist(s)) for s in sessions[:_ASK_GISTS_MAX]]
        gists = [(s, g) for s, g in gists if g]
        if gists:
            parts += ["", "What you wrote down afterwards about the conversation it came from"
                      + (" and the ones you raised it in:" if kind == "user" and len(gists) > 1 else ":")]
            for s, g in gists:
                parts += ["", f"({_chat_date(s)})", _clip(g, _ASK_GIST_CAP)]
        else:
            parts += ["", "(Nothing more about where it came from is on record.)"]
    parts += ["", "Now try to settle it: think it through, then write exactly one of the three sections."]
    return "\n".join(parts)


def _ask_grid(value) -> tuple:
    """The temperatures a run asks for (``options.temperatures``), else :data:`T_GRID`."""
    try:
        temps = [round(float(t), 2) for t in (value or [])]
    except (TypeError, ValueError):
        temps = []
    temps = [t for t in dict.fromkeys(temps) if 0.0 <= t <= 2.0]
    return tuple(temps) or T_GRID


def _ask_samples(value) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return K_PER_T
    return max(1, min(K_MAX, k))


def _ask_attempt_build(ask: dict, window: int, tokenizer) -> tuple:
    """One part per (temperature, sample), the same user turn in each, its own temperature
    on the part (``ModuleSpec.own_sampling``)."""
    opts = ask.get("_options") if isinstance(ask.get("_options"), dict) else {}
    temps, k = _ask_grid(opts.get("temperatures")), _ask_samples(opts.get("samples"))
    content = _ask_material(ask)
    blocks, grid = [], []
    for t in temps:
        for n in range(1, k + 1):
            blocks.append({"content": content, "temperature": t, "label": f"T {t:g} · {n} of {k}"})
            grid.append((t, n))
    about = {"id": ask.get("id"), "kind": ask.get("ask_kind"), "question": ask.get("content"),
             "raised": str(ask.get("ts") or "")[:10], "origin": _ask_origin(ask),
             "surface_count": ask.get("surface_count") or 0}
    return blocks, {"grid": grid, "ask": about}


# A section heading: `## RESOLVED` as asked (any level, any case, bold or not), or the bare
# word in bold on its own line. What follows it on the same line counts as its text.
_ASK_HEAD_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,4}[ \t]*\**|\*\*)(RESOLVED|REFRAMED|OPEN)\b\**[ \t]*[:—–-]?[ \t]*(.*)$",
    re.M | re.I)
_ASK_PLACEHOLDER_RE = re.compile(r"^[\W_]*(?:n/?a|none|empty|нет|пусто)?[\W_]*$", re.I)


def parse_attempt(raw: str) -> dict:
    """One attempt's answer (what follows its reasoning) split into the three sections:
    ``{outcome, sections, text, why_malformed}``. ``outcome`` is the one section with text,
    or ``malformed``: no heading, every section empty, or more than one with text."""
    body = reasoning_text.answer_after_think(raw or "")
    heads = list(_ASK_HEAD_RE.finditer(body))
    sections: dict = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        text = (m.group(2) + "\n" + body[m.end():end]).strip()
        if text and not _ASK_PLACEHOLDER_RE.match(text):
            name = m.group(1).upper()
            sections[name] = (sections.get(name, "") + "\n\n" + text).strip()
    filled = [s for s in ASK_OUTCOMES if s in sections]
    outcome = filled[0] if len(filled) == 1 else "malformed"
    why = ("" if outcome != "malformed" else "no section heading" if not heads
           else "every section empty" if not filled else f"{len(filled)} sections filled")
    return {"outcome": outcome, "sections": sections,
            "text": sections[outcome] if outcome != "malformed" else body.strip(), "why_malformed": why}


def _ask_attempt_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    context = context or {}
    grid, parts, about = list(context.get("grid") or []), list(context.get("parts") or []), context.get("ask") or {}
    records = []
    for i, raw in enumerate(raws):
        t, n = grid[i] if i < len(grid) else (None, None)
        info = parts[i] if i < len(parts) else {}
        records.append({"temperature": t, "sample": n, **parse_attempt(raw),
                        "truncated": bool(info.get("truncated")),
                        "cut_before_answer": bool(info.get("cut_before_answer")),
                        "stopped_on_loop": bool(info.get("stopped_on_loop"))})
    cols = ASK_OUTCOMES + ("malformed",)
    temps = list(dict.fromkeys(r["temperature"] for r in records if r["temperature"] is not None))
    table = {f"{t:g}": {c: sum(1 for r in records if r["temperature"] == t and r["outcome"] == c) for c in cols}
             for t in temps}
    counts = {c: sum(1 for r in records if r["outcome"] == c) for c in cols}
    q = " ".join(str(about.get("question") or "").split())
    k = max((r["sample"] or 0 for r in records), default=0)
    lines = [f"Open question ({about.get('kind') or '?'}, raised {about.get('raised') or '?'}"
             + (f" {about['origin']}" if about.get("origin") else "")
             + (f"; raised with the person {about['surface_count']}×" if about.get("surface_count") else "") + "):",
             f"  {_clip(q, 400)}",
             f"{len(records)} attempt(s)" + (f" on {context['weights']}" if context.get("weights") else "")
             + f": T {' / '.join(f'{t:g}' for t in temps)} × {k}",
             "",
             f"{'T':<6}" + "".join(f"{c:>11}" for c in cols)]
    for t in temps:
        lines.append(f"{t:<6g}" + "".join(f"{table[f'{t:g}'][c]:>11}" for c in cols))
    lines.append(f"{'all':<6}" + "".join(f"{counts[c]:>11}" for c in cols))
    for r in records:
        flag = (" (cut off before the answer)" if r["cut_before_answer"] else
                " (stopped by a guard)" if r["stopped_on_loop"] else " (hit the budget)" if r["truncated"] else "")
        lines += ["", f"── T {r['temperature']:g} · {r['sample']} of {k}: {r['outcome']}{flag}"
                  + (f" ({r['why_malformed']})" if r["outcome"] == "malformed" else "")]
        lines += _clip(r["text"], 1200).splitlines() or ["(nothing after the reasoning)"]
    return {"records": records, "count": sum(1 for r in records if r["outcome"] != "malformed"),
            "counts": {c: v for c, v in counts.items() if v}, "lines": lines, "table": table, "ask": about,
            "weights": context.get("weights") or ""}


_ASK_ATTEMPT = ModuleSpec(
    name="ask_attempt",
    label="Ask attempt (one open question × the temperature grid)",
    prompt_file="ask_attempt_prompt.txt",
    # Thinking on (a resolution attempt is reasoning), so the budget holds both.
    max_new_tokens="4096",
    build=_ask_attempt_build,
    finish=_ask_attempt_finish,
    describe_output=("One open question attempted 3 times at each of T 0.3 / 0.7 / 1.0 / 1.2, each attempt "
                     "landing in RESOLVED, REFRAMED or OPEN: a temperature × outcome table, then every "
                     "attempt. Nothing is written: no attempt record, no closure."),
    source="ask",
    default_prompt=_ask_attempt_default_prompt,
    own_sampling=True,
    input_note=("The input is the question, where it came from, and by kind: her standing self-portrait "
                "(meta), or the gists of the conversations it came from and was raised in (user, search). "
                "No retrieval, no clock. Each part is sampled at its own temperature; the one here does "
                "not apply."),
)


# ── child_ask: the question a closure that rejected its terms leaves open (§3.4) ──
# The Sleep step's pass (`core/child_ask.py`) on ONE closure, previewed: the same prompt and
# turn (the question she closed and what she wrote when she closed it; never the judge's
# text), nothing retrieved, and nothing written — no child, no decision.

def _child_ask_build(item: dict, window: int, tokenizer) -> tuple:
    from core import child_ask as ca
    return [ca.build_content(item)], {"item": item}


def _child_ask_finish(raws: list, *, truncated: bool, context: dict = None) -> dict:
    from core import ask_judge as aj
    from core import child_ask as ca
    item = (context or {}).get("item") or {}
    outcome, question = ca.parse_child(raws[0] if raws else "", parent=item.get("question") or "")
    lines = [f"The question she closed (closed {str(item.get('closed_ts') or '?')[:10]}, the child at depth "
             f"{item.get('depth')}{', past the limit: a Sleep run would not ask' if item.get('too_deep') else ''}):",
             f"  {aj.head(item.get('question'), 400)}",
             "What she wrote when she closed it:",
             f"  {aj.head(item.get('answer'), 600)}",
             "How the judge read it (she does not see this):",
             f"  {aj.head(item.get('why'), 300)}",
             "",
             {"spawned": f"CHILD: {question}", "none": "NONE: she finds nothing left open.",
              "repeat": f"REPEAT (the closed question again, not spawned): {question}",
              "malformed": "MALFORMED: no [ask:meta] line and no NONE."}[outcome]]
    return {"records": [{"outcome": outcome, "question": question}], "count": 1 if outcome == "spawned" else 0,
            "counts": {outcome: 1}, "lines": lines}


_CHILD_ASK = ModuleSpec(
    name="child_ask",
    label="Child ask (the question a closure that rejected its terms leaves open)",
    prompt_file="child_ask_prompt.txt",
    max_new_tokens="2048",
    build=_child_ask_build,
    finish=_child_ask_finish,
    describe_output=("The question her answer leaves open, on one [ask:meta] line, or NONE: what a Sleep run "
                     "would write as the closed question's child. Nothing is written here."),
    source="child",
    default_prompt=lambda: __import__("core.child_ask", fromlist=["load_prompt"]).load_prompt(_PROMPTS_DIR),
    input_note=("The input is a closed meta question the judge accepted while its WHY reads as rejecting "
                "the question's terms: the question and what she wrote when she closed it, never the "
                "judge's text. No retrieval, no clock."),
)


MODULES: dict[str, ModuleSpec] = {
    _CHAT_FACTS.name: _CHAT_FACTS,
    _CHAT_SUMMARY.name: _CHAT_SUMMARY,
    _TIL_FACTS.name: _TIL_FACTS,
    _FACT_FETCH.name: _FACT_FETCH,
    _ASSOC_FETCH.name: _ASSOC_FETCH,
    _ASSOC_PIVOT.name: _ASSOC_PIVOT,
    _AXIS_PROBE.name: _AXIS_PROBE,
    _ASK_ATTEMPT.name: _ASK_ATTEMPT,
    _CHILD_ASK.name: _CHILD_ASK,
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


def _memory_dir():
    if _MEMORY_DIR is not None:
        return _MEMORY_DIR
    from training.reflections_path import memory_dir
    return memory_dir()


def _open_asks() -> list:
    """Her live ``[ask]`` items (``ReflectionMemory.open_questions``). Read-only: a run never
    closes, counts or touches a question."""
    from core.reflection_memory import ReflectionMemory
    return ReflectionMemory(_memory_dir()).open_questions()


def _load_ask_input(ident: str) -> tuple:
    """One open question by its id (or its content key)."""
    for ask in _open_asks():
        if ident in (str(ask.get("id")), str(ask.get("key"))):
            doc = dict(ask)
            return doc, {"units": 1, "unit_label": "question",
                         "empty": not str(doc.get("content") or "").strip(),
                         "empty_reason": f"{ident} has no text."}
    raise FileNotFoundError(f"no open question {ident}")


def _list_ask_inputs() -> list:
    """Every open question, newest first: its opening words, then kind · date."""
    out = []
    for ask in sorted(_open_asks(), key=lambda a: str(a.get("ts") or ""), reverse=True):
        text = " ".join(str(ask.get("content") or "").split())
        sub = f"{ask.get('ask_kind') or '?'} · raised {str(ask.get('ts') or '')[:10]}"
        if ask.get("surface_count"):
            sub += f" · put to the person {ask['surface_count']}×"
        out.append({"id": str(ask.get("id")), "label": _clip(text, 110), "sub": sub})
    return out


def _child_plan() -> list:
    """The closures awaiting a child (``child_ask.plan`` over the live op-log and the ask
    judge's store). Read-only."""
    from core import ask_judge as aj
    from core import child_ask as ca
    rag = aj.read_jsonl(Path(_memory_dir()) / "rag_memory.jsonl")
    return ca.plan(rag, aj.read_jsonl(aj.store_path(_CHATS_DIR))) if _CHATS_DIR is not None else []


def _load_child_input(ident: str) -> tuple:
    """One closure awaiting its child, by closure id."""
    for item in _child_plan():
        if item.get("closure_id") == ident:
            return dict(item), {"units": 1, "unit_label": "closure",
                                "empty": not (item.get("question") and item.get("answer")),
                                "empty_reason": f"{ident} has no question or answer on record."}
    raise FileNotFoundError(f"no closure awaiting a child: {ident}")


def _list_child_inputs() -> list:
    return [{"id": it["closure_id"], "label": _clip(" ".join(str(it.get("question") or "").split()), 110),
             "sub": f"closed {str(it.get('closed_ts') or '')[:10]} · child depth {it.get('depth')}"
                    + (" · past the depth limit" if it.get("too_deep") else "")}
            for it in _child_plan()]


_INPUT_SOURCES: dict = {
    "chat": {"load": _load_chat_input, "list": _list_chat_inputs,
             "noun": "chat", "select": "Select a chat first."},
    "til": {"load": _load_til_input, "list": _list_til_inputs,
            "noun": "text", "select": "Select an article or digest first."},
    "ask": {"load": _load_ask_input, "list": _list_ask_inputs,
            "noun": "question", "select": "Select an open question first."},
    "child": {"load": _load_child_input, "list": _list_child_inputs,
              "noun": "closure", "select": "Select a closure first."},
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
    """The module's prompt as it stands on disk; a missing/empty file falls back to the
    module's `default_prompt` (which writes it), else ""."""
    try:
        if _PROMPTS_DIR is not None:
            path = _PROMPTS_DIR / spec.prompt_file
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
    except Exception:
        pass
    if spec.default_prompt is not None:
        try:
            return (spec.default_prompt() or "").strip()
        except Exception:
            traceback.print_exc()
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
            "own_sampling": spec.own_sampling,
            "input_note": spec.input_note,
        }
        for spec in MODULES.values()
    ]


# ── the run ───────────────────────────────────────────────────────────────────

def _weights_label() -> str:
    """The weights the loaded model runs on: ``adapter:<dir>``, or ``base`` without one."""
    from pathlib import Path as _P
    aid = str(getattr(_runtime, "adapter_id", "") or "")
    return f"adapter:{_P(aid).name}" if aid else "base"


def _precheck(module: str, filename: str) -> Optional[dict]:
    """The refusals a run would make, checked before a clean-base swap spends two model
    loads on an input that cannot run. None when it can."""
    spec = MODULES.get(str(module or "").strip())
    if spec is None:
        return {"skipped": "unknown_module", "message": f"No such module: {module!r}"}
    if _runtime.model is None:
        return {"skipped": "no_model"}
    src = _INPUT_SOURCES.get(spec.source)
    if src is None:
        return {"skipped": "unknown_source",
                "message": f"Module {spec.name!r} declares an unknown input source {spec.source!r}."}
    try:
        _doc, meta = src["load"](filename)
    except Exception as e:
        return {"skipped": "bad_session", "message": f"Could not read {filename}: {type(e).__name__}: {e}"}
    if meta.get("empty"):
        return {"skipped": "empty_session", "message": meta["empty_reason"]}
    return None


def run_module_blocking(
    *,
    module: str,
    filename: str,
    prompt: str = "",
    inject: tuple = (),
    temperature: float = 0.7,
    top_p: float = 0.95,
    options: Optional[dict] = None,
    weights: str = "adapter",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run one module against one input and return what it produced. **Writes nothing.**

    *weights* chooses the model the pass runs on: ``adapter`` (the loaded model, as ever)
    or ``base`` — the same pass inside the clean base (the injected ``with_clean_base``:
    ``CleanBaseSession`` releases the adapter model, loads the bare base, and restores the
    adapter after; two full loads). Everything else is equal, so one input can be read under
    both. The input is checked before the swap, so a bad one costs no loads.
    """
    weights = str(weights or "adapter").strip()
    kwargs = dict(module=module, filename=filename, prompt=prompt, inject=inject,
                  temperature=temperature, top_p=top_p, options=options,
                  on_chunk=on_chunk, on_stage=on_stage)
    if weights == "adapter":
        return _run_module_loaded(**kwargs, weights_label=_weights_label())
    if weights != "base":
        return {"skipped": "unknown_weights",
                "message": f"weights is adapter or base, not {weights!r}."}
    if _with_clean_base is None:
        return {"skipped": "no_clean_base",
                "message": "This server did not wire the clean base, so no module can run on the bare base."}
    refused = _precheck(module, filename)
    if refused is not None:
        return refused
    if on_stage is not None:
        try:
            on_stage({"stage": "swapping", "weights": "base",
                      "message": "Swapping the adapter out for the bare base (two full model loads)…"})
        except Exception:
            pass
    # Busy for the whole swap, the loads included: the idle scheduler reads this flag.
    global _module_run_active
    _module_run_active = True
    try:
        return _with_clean_base(lambda: _run_module_loaded(**kwargs, weights_label="base"))
    finally:
        _module_run_active = False


def _run_module_loaded(
    *,
    module: str,
    filename: str,
    prompt: str = "",
    inject: tuple = (),
    temperature: float = 0.7,
    top_p: float = 0.95,
    options: Optional[dict] = None,
    weights_label: str = "",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run one module against one chat on the model loaded now. **Writes nothing.**

    Runs on the single GPU executor thread (blocking). *prompt* overrides the module's
    on-disk prompt for this run only — the whole point of the workbench is to try one
    without editing the file. Nothing is injected: the pass sees the module prompt and the
    transcript, and that is all. *options* are the module's own run knobs (``ask_attempt``:
    ``temperatures``, ``samples``), carried to its ``build`` on the loaded doc as
    ``_options``, as the til loader carries the snippet's kind.
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

    # A clean-base run holds the flag across its swap; the body then leaves it to the caller.
    already_busy = _module_run_active
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
        if isinstance(options, dict) and options and isinstance(session, dict):
            session["_options"] = dict(options)

        if spec.forward is not None:
            # A forward-pass module: no prompt, no generation. See ModuleSpec.forward.
            _stage(stage="reading", session=filename, units=meta["units"],
                   unit_label=meta["unit_label"], exchanges=meta["units"])
            activity_log.set_ambient_label(f"module:{spec.name}")
            result = spec.forward(session, meta, _stage)
            if result.get("skipped"):
                return result
            _stage(stage="parsed", count=int(result.get("count") or 0))
            return {
                "ok": True, "module": spec.name, "source": spec.source, "session": filename,
                "weights": weights_label,
                "units": meta["units"], "unit_label": meta["unit_label"],
                "exchanges": meta["units"], "parts": 1, "raw": "",
                **result,
                "truncated": False, "stopped_on_loop": False, "cut_before_answer": False,
                "input_tokens": None, "max_new_tokens": None, "context_length": None,
                "written": False,
            }

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
        parts_info: list = []
        truncated = False
        stopped_on_loop = False
        cut_before_answer = False
        for i, block in enumerate(blocks, 1):
            # A part is its content, or a dict carrying its own sampling and a label
            # (ModuleSpec.own_sampling: one part per temperature of a grid).
            sampling = block if isinstance(block, dict) else {}
            content = sampling.get("content", "") if sampling else block
            part_temperature = float(sampling.get("temperature", temperature))
            part_top_p = float(sampling.get("top_p", top_p))
            label = str(sampling.get("label") or "")
            _stage(stage="generating", session=filename, module=spec.name,
                   part=i, parts=len(blocks),
                   **({"label": label, "temperature": part_temperature} if sampling else {}))
            if len(blocks) > 1 and on_chunk is not None and (i > 1 or label):
                # The blocks are separate generations joined into one value; without a
                # marker the streamed text reads as a single runaway.
                on_chunk(("\n\n" if i > 1 else "")
                         + f"── part {i} of {len(blocks)}" + (f" · {label}" if label else "") + " ──\n\n")
            raw = generate(
                content, system_prompt,
                temperature=part_temperature, top_p=part_top_p,
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
            part_loop = bool(getattr(generate, "last_loop", None))
            part_cut = reasoning_text.truncated_before_answer(raw or "", part_truncated)
            truncated = truncated or part_truncated
            stopped_on_loop = stopped_on_loop or part_loop
            cut_before_answer = cut_before_answer or part_cut
            parts_info.append({"truncated": part_truncated, "stopped_on_loop": part_loop,
                               "cut_before_answer": part_cut})

        # What happened to each part rides beside the build's own context, for a module
        # whose parts are separate answers rather than pieces of one.
        result = spec.finish(raws, truncated=truncated,
                             context={**(build_context or {}), "parts": parts_info,
                                      "weights": weights_label})
        _stage(stage="parsed", count=int(result.get("count") or 0))

        return {
            "ok": True,
            "module": spec.name,
            "source": spec.source,
            "session": filename,
            # Which weights produced it (`adapter:<dir>` or `base`), since a run may swap.
            "weights": weights_label,
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
        if not already_busy:
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
    options = msg.get("options") if isinstance(msg.get("options"), dict) else None
    weights = str(msg.get("weights") or "adapter")

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
                temperature=temperature, top_p=top_p, options=options, weights=weights,
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
          ["ask_attempt", "assoc_fetch", "assoc_pivot", "axis_probe", "chat_facts", "chat_summary",
           "child_ask", "fact_fetch", "til_facts"])
    check("axis_probe is a forward pass", MODULES["axis_probe"].forward is not None, True)
    check("subsequence finder", _find_subseq_end([1, 2, 3, 4, 2, 3, 9], [2, 3]), 4)
    check("subsequence finder miss", _find_subseq_end([1, 2], [7]), None)
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
    # chat_summary's budget mirrors the run default its production twin inherits — one
    # definition since the runner took it from reflection_config, so compared by value.
    from core.reflection_config import DEFAULT_MAX_NEW_TOKENS_SETTING as _run_default
    check("summary budget matches the run default",
          (_CHAT_SUMMARY.max_new_tokens == _run_default,
           "_DEFAULT_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS_SETTING" in runner), (True, True))
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
          sorted({m["source"] for m in catalogue()}), ["ask", "chat", "child", "til"])
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
    check("...and only the two selection passes do",
          [s.name for s in MODULES.values() if s.disable_thinking], ["fact_fetch", "assoc_fetch"])
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

    print("\nask_attempt: one open question × the temperature grid (§3.3, §7 item 6)")
    import tempfile
    check("the grid and the samples are §3.3's", (T_GRID, K_PER_T), ((0.3, 0.7, 1.0, 1.2), 3))
    check("the module samples its own parts, from the ask source",
          (_ASK_ATTEMPT.own_sampling, _ASK_ATTEMPT.source, bool(_ASK_ATTEMPT.input_note)), (True, "ask", True))
    check("the catalogue says so, for the tab",
          [(m["own_sampling"], bool(m["input_note"])) for m in catalogue() if m["name"] == "ask_attempt"],
          [(True, True)])
    one = parse_attempt("<think>## RESOLVED\nnot here</think>\n## OPEN\nIt stays open: I would need X.")
    check("the one section with text is the outcome; headings inside the reasoning do not count",
          (one["outcome"], one["text"]), ("OPEN", "It stays open: I would need X."))
    check("heading variants, placeholders, and the malformed cases",
          [parse_attempt(x)["outcome"] for x in (
              "## Resolved: the answer is Y, because Z.",
              "**REFRAMED**\nThe better question is W.",
              "### OPEN — because nothing settles it.",
              "## RESOLVED\n—\n## REFRAMED\n\n## OPEN\nbecause.",
              "## RESOLVED\nY.\n## OPEN\nand also open.",
              "The answer is Y.",
              "## RESOLVED\nn/a\n## OPEN\n(нет)",
              "Open questions remain, and I keep them.",
              "## OPENED\nnot a heading")],
          ["RESOLVED", "REFRAMED", "OPEN", "OPEN", "malformed", "malformed", "malformed", "malformed",
           "malformed"])
    check("...and why each is malformed",
          [parse_attempt(x)["why_malformed"] for x in ("## RESOLVED\nY.\n## OPEN\nZ.", "Y.", "## OPEN\n-")],
          ["2 sections filled", "no section heading", "every section empty"])

    ask = {"id": "q1", "key": "k1", "ask_kind": "meta", "content": "What am I?", "ts": "2026-09-01T10:00:00",
           "source_session": "20260901_100000.json", "run_id": "r1", "surface_count": 0,
           "surfaced_in_sessions": []}
    _saved_portrait = _standing_portrait
    globals()["_standing_portrait"] = lambda: "I am curious and slow to settle."
    blocks, _ctx = _ask_attempt_build(dict(ask), 24576, None)
    check("by default the grid × K parts, each labelled",
          (len(blocks), [b["temperature"] for b in blocks][::3], blocks[0]["label"]),
          (12, [0.3, 0.7, 1.0, 1.2], "T 0.3 · 1 of 3"))
    check("every part is the same turn", len({b["content"] for b in blocks}), 1)
    check("a meta question carries her standing portrait and what its kind means",
          ("I am curious and slow to settle." in blocks[0]["content"],
           "question about yourself" in blocks[0]["content"]), (True, True))
    small = _ask_attempt_build({**ask, "_options": {"temperatures": [1.2, 0.3, 1.2, 9], "samples": 1}},
                               24576, None)[0]
    check("options choose the temperatures (deduplicated, in range) and the samples",
          [(b["temperature"], b["label"]) for b in small], [(1.2, "T 1.2 · 1 of 1"), (0.3, "T 0.3 · 1 of 1")])
    check("bad options fall back to the grid",
          len(_ask_attempt_build({**ask, "_options": {"temperatures": "x", "samples": "y"}}, 24576, None)[0]), 12)
    check("samples are capped", _ask_samples(99), K_MAX)

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "chats"
        chats.mkdir()
        (chats / "20260901_100000.json").write_text(json.dumps({"user": "Sam", "exchanges": []}), encoding="utf-8")
        (chats / "20260901_100000.summary.json").write_text(json.dumps({"text": "We talked about the garden."}),
                                                             encoding="utf-8")
        (chats / "20260905_120000.summary.json").write_text(
            json.dumps({"text": "I asked Sam about it; he said he would answer later."}), encoding="utf-8")
        saved_dirs = (_CHATS_DIR, _MEMORY_DIR)
        globals()["_CHATS_DIR"] = chats
        user_ask = {**ask, "id": "q2", "ask_kind": "user", "content": "What did Sam mean?", "run_id": "synth_x",
                    "surfaced_in_sessions": ["20260905_120000.json"], "surface_count": 1}
        text = _ask_material(user_ask)
        check("a user question carries the gists of where it came from and where it was raised",
              ("We talked about the garden." in text, "answer later" in text, "only the person can answer" in text,
               "self-portrait" in text), (True, True, True, False))
        check("...and says where it came from", _ask_origin(user_ask),
              "in a conversation on 2026-09-01 with Sam, when you re-read it later")
        check("a reading is named by its title",
              _ask_origin({"source_session": "wander/20260929_061628_Urban_Culture_Gas_mask.txt"}),
              "while reading on your own (Urban Culture Gas mask)")
        check("a transcript outside the chats dir is never read", _chat_doc("../secrets.json"), {})

        mem = Path(td) / "memory"
        mem.mkdir()
        ops = [{"op": "insert", "kind": "ask", "ask_kind": "meta", "key": "k1", "id": "q1",
                "ts": "2026-09-01T10:00:00", "content": "What am I?", "source_session": "20260901_100000.json"},
               {"op": "insert", "kind": "ask", "ask_kind": "user", "key": "k2", "id": "q2",
                "ts": "2026-09-02T10:00:00", "content": "What did Sam mean?",
                "source_session": "20260901_100000.json"},
               {"op": "surface", "key": "k2", "surfaced_in": "20260905_120000.json", "ts": "2026-09-05T12:00:00"},
               {"op": "insert", "kind": "ask", "ask_kind": "meta", "key": "k3", "id": "q3",
                "ts": "2026-09-03T10:00:00", "content": "Closed already?"},
               {"op": "evict", "key": "k3"},
               {"op": "insert", "kind": "fact", "key": "k4", "id": "f4", "content": "A fact."}]
        (mem / "rag_memory.jsonl").write_text("".join(json.dumps(o) + "\n" for o in ops), encoding="utf-8")
        globals()["_MEMORY_DIR"] = mem
        listed = list_inputs("ask")
        check("the source lists open questions only, newest first", [e["id"] for e in listed], ["q2", "q1"])
        check("...with kind, date and how often it was put to the person", listed[0]["sub"],
              "user · raised 2026-09-02 · put to the person 1×")
        doc, meta = _load_ask_input("k2")
        check("one loads by id or key, with where it was raised",
              (doc["id"], doc["surfaced_in_sessions"], meta["unit_label"]),
              ("q2", ["20260905_120000.json"], "question"))
        try:
            _load_ask_input("k3")
            gone = "loaded"
        except FileNotFoundError:
            gone = "refused"
        check("a closed question is not an input", gone, "refused")

        seen_temps = []

        class _Gen:
            last_truncated = False
            last_loop = False
            last_input_tokens = 10
            last_max_new_tokens = 4096
            last_context_length = 32768

            def __call__(self, content, system, **kw):
                seen_temps.append(kw["temperature"])
                self.last_truncated = kw["temperature"] > 1.0
                return ("<think>weighing</think>## OPEN\nIt stays open." if kw["temperature"] < 1.0
                        else "<think>still weighing when the budget ran out")

        saved_rt = (_runtime.model, _runtime.tokenizer, getattr(_runtime, "adapter_id", None))
        saved_factory = _make_sync_reflect_generate
        saved_clean = _with_clean_base
        adapter_model, base_model = object(), object()
        _runtime.model, _runtime.tokenizer = adapter_model, None
        _runtime.adapter_id = "/models/adapter-20260930_104125"
        globals()["_make_sync_reflect_generate"] = lambda rag: _Gen()
        swaps: list = []

        def _fake_clean_base(fn):
            # What CleanBaseSession does to the shared runtime, minus the loads.
            swaps.append(_module_run_active)
            _runtime.model = base_model
            try:
                return fn()
            finally:
                swaps.append(_module_run_active)
                _runtime.model = adapter_model

        models_seen: list = []
        _gen_call = _Gen.__call__

        def _spy(self, content, system, **kw):
            models_seen.append("base" if _runtime.model is base_model else "adapter")
            return _gen_call(self, content, system, **kw)

        _Gen.__call__ = _spy
        try:
            out = run_module_blocking(module="ask_attempt", filename="q1", prompt="Try it.", temperature=0.7,
                                      options={"temperatures": [0.3, 1.2], "samples": 2})
            first_temps = list(seen_temps)
            check("a run on the loaded model names its adapter", (out.get("weights"), out["lines"][2][:25]),
                  ("adapter:adapter-20260930_104125", "4 attempt(s) on adapter:a"))
            no_wire = run_module_blocking(module="ask_attempt", filename="q1", weights="base")
            globals()["_with_clean_base"] = _fake_clean_base
            odd = run_module_blocking(module="ask_attempt", filename="q1", weights="lora")
            missing = run_module_blocking(module="ask_attempt", filename="nope", weights="base")
            check("unknown weights, an unwired clean base, and a bad input are refused — the last before any swap",
                  (odd.get("skipped"), no_wire.get("skipped"), missing.get("skipped"), swaps),
                  ("unknown_weights", "no_clean_base", "bad_session", []))
            models_seen.clear()
            base_out = run_module_blocking(module="ask_attempt", filename="q1", prompt="Try it.",
                                           options={"temperatures": [0.3], "samples": 2}, weights="base")
            check("weights base: every part generates inside the clean base, and says so",
                  (models_seen, base_out.get("weights"), "on base" in base_out["lines"][2]),
                  (["base", "base"], "base", True))
            check("...busy across the whole swap, idle after, the adapter back",
                  (swaps, _module_run_active, _runtime.model is adapter_model), ([True, True], False, True))
        finally:
            _Gen.__call__ = _gen_call
            _runtime.model, _runtime.tokenizer, _runtime.adapter_id = saved_rt
            globals()["_make_sync_reflect_generate"] = saved_factory
            globals()["_with_clean_base"] = saved_clean
            globals()["_CHATS_DIR"], globals()["_MEMORY_DIR"] = saved_dirs
            globals()["_standing_portrait"] = _saved_portrait
        check("each part ran at its own temperature, not the run's", first_temps, [0.3, 0.3, 1.2, 1.2])
        check("the temperature × outcome table", out.get("table"),
              {"0.3": {"RESOLVED": 0, "REFRAMED": 0, "OPEN": 2, "malformed": 0},
               "1.2": {"RESOLVED": 0, "REFRAMED": 0, "OPEN": 0, "malformed": 2}})
        check("a part cut off inside its reasoning is marked so, per part",
              [r["cut_before_answer"] for r in out.get("records") or []], [False, False, True, True])
        check("counts for the tab's OUTPUT line, and nothing written",
              (out.get("counts"), out.get("written"), out.get("parts")), ({"OPEN": 2, "malformed": 2}, False, 4))
        check("the rendered table heads the lines",
              any(ln.startswith("T ") and "RESOLVED" in ln for ln in out.get("lines") or []), True)

    print("\nchild_ask: the preview of the Sleep step's pass (§3.4)")
    item = {"closure_id": "c1", "key": "k1", "question": "Was my relief a defense?",
            "answer": "The split is false: the strategy is the relief.", "why": "Ava rejects the dichotomy.",
            "closed_ts": "2026-09-01T00:00:00", "depth": 1, "too_deep": False}
    blocks, ctx = _child_ask_build(item, 24576, None)
    check("one turn: her question and her answer, not the judge's WHY",
          (len(blocks), "Was my relief a defense?" in blocks[0], "rejects the dichotomy" in blocks[0]), (1, True, False))
    out = _child_ask_finish(["<think>x</think>[ask:meta] What would a strategy that is me look like?"],
                            truncated=False, context=ctx)
    check("the child it would spawn, with the judge's reading shown to the operator",
          (out["records"][0], out["lines"][-1], any("rejects the dichotomy" in ln for ln in out["lines"])),
          ({"outcome": "spawned", "question": "What would a strategy that is me look like?"},
           "CHILD: What would a strategy that is me look like?", True))
    check("NONE and malformed are shown as such",
          [_child_ask_finish([r], truncated=False, context=ctx)["records"][0]["outcome"] for r in ("NONE", "hmm")],
          ["none", "malformed"])
    with tempfile.TemporaryDirectory() as td:
        from core import ask_judge as _aj
        chats = Path(td) / "data" / "chats"
        chats.mkdir(parents=True)
        mem = Path(td) / "memory"
        mem.mkdir()
        (mem / "rag_memory.jsonl").write_text("", encoding="utf-8")
        _aj.append_record(_aj.store_path(chats), {
            "closure_id": "c1", "key": "k1", "judge_key": "J", "ts": "2026-09-01T00:00:01", "ask_kind": "meta",
            "accepted": True, "why": "Ava rejects the binary choice between A and B.", "question": "q?",
            "answer": "Neither A nor B.", "closed_ts": "2026-09-01T00:00:00"})
        saved_dirs = (_CHATS_DIR, _MEMORY_DIR)
        globals()["_CHATS_DIR"], globals()["_MEMORY_DIR"] = chats, mem
        try:
            listed = list_inputs("child")
            doc, meta = _load_child_input("c1")
        finally:
            globals()["_CHATS_DIR"], globals()["_MEMORY_DIR"] = saved_dirs
        check("the child source lists the closures awaiting a child, and loads one",
              ([e["id"] for e in listed], listed[0]["sub"], doc["question"], meta["unit_label"]),
              (["c1"], "closed 2026-09-01 · child depth 1", "q?", "closure"))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
