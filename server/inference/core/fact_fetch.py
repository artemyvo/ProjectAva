"""The fact-fetch pass: which facts are worth having in front of you before this message.

**One definition, two callers.** The workbench module (`core.modules.fact_fetch`) and the
live chat channel (`generation._facts_block`) run the same pass over the same candidate
list with the same prompt — the discipline `til_facts` already follows for its two callers,
and the reason the workbench is worth having at all: a pass an operator tunes in the tab and
a pass that runs on a live turn must not be two pieces of code that merely resemble
each other.

The split with `graph.blob` is deliberate and is the safety boundary. Everything that
decides *what may be shown* — the knowledge-facet filter, the `person:_self` exclusion, the
freshness scope, the rendering of the blob itself — lives there, in the pure package, and is
applied to the candidate list **before** the pass sees it. This module only assembles the
prompt around that list and interprets the answer. So a bug here can pick the wrong facts;
it cannot surface a forbidden one.

Why the model selects rather than code scoring it: **886 of 946 claims on this box are
written in English while the conversations are often Russian**. A lexical score cannot
cross that and returns an empty blob; judging what a fact is *about* is what a multilingual
model does natively. See `graph/blob.py`'s claim-lane note for the measurements.

GPU-free self-test: ``python -m core.fact_fetch``.
"""

from __future__ import annotations

from typing import Callable, Optional

# Instructional tail, appended after the material. Lives here rather than in the prompt file
# because it must come *after* the transcript — the prompt file is the system message, and
# the last thing in the content has to be the task rather than something to continue.
FETCH_CLOSING = (
    "— end of the conversation —\n\n"
    "Now list the numbers of the facts worth having in front of you before this message "
    "is answered, under the label FACTS:, one number per line, most important first. "
    "Judge what a fact is about, not which words it shares with the message — the list "
    "and the conversation may be in different languages. NONE if nothing needs looking up."
)

# Chars of prompt per token of window, used to size the transcript. Deliberately
# conservative (real mixed EN/RU runs ~3.5): overshooting the budget costs a truncated
# prompt, undershooting costs a few turns of context.
_CHARS_PER_TOKEN = 3

# The `skipped` reasons that mean the channel is BROKEN rather than simply quiet. Owned
# here — the module whose pass produces most of them — so the callers that log failures
# (live chat, outreach, synthesis) share one classification instead of each deciding
# afresh which reasons are worth a line. A box missing its tree is in this set on
# purpose: `python -m graph.build` is the act that turns the channel on in substance,
# and an operator who thinks it is already running has no other way to find out.
# (`no_module` is emitted by the callers, not by `_run`; it lives here because the set
# is about the channel, not about who noticed.) `oom` is the one failure that is not
# about the channel's own plumbing: the fetch's candidate-list prefill is often the
# first big allocation of a turn on a box running at its VRAM ceiling, so it is where a
# fragmented allocator pool surfaces first — named apart from `generate_failed` because
# the operator's next move is different (check the `[alloc]` verdict in server.log, not
# the pass), and because the live path retries an OOM once before it lands here (see
# `generation._fetch_facts_block_sync`).
FETCH_FAILURES = frozenset({
    "error", "generate_failed", "candidates_failed", "no_tree", "no_prompt", "no_module",
    "oom",
})


def last_real_user_turn(session: dict, turns: list) -> Optional[int]:
    """Index of the last turn that is a message from the OTHER side, or ``None``.

    The box already has this rule and applies it wherever a reader asks "did the user
    actually say something" — ``checkin._user_turns`` skips an ``initiated_by:"ava"``
    exchange 0 whatever its speaker label, ``dialogue_source.build_dialogue_anchor`` masks
    it from training, the Chat-review tab labels it. This pass was the one place that did
    not, so on an Ava-initiated reach-out it took her own stage direction ("About 7 hours
    had passed since you last spoke with your friend…") as *the message that has just
    arrived* and, reasonably, fetched the friend.

    Both halves are needed: the speaker check catches the ``(initiative)``/``(setting)``
    convention, and the exchange-0 check catches transcripts written before it existed.
    """
    from core.reflection_source import is_narrator_speaker

    ava_opened = (session.get("initiated_by") or "").strip() == "ava"
    exchange = -1
    last = None
    for i, t in enumerate(turns):
        if t.get("role") != "user":
            continue
        exchange += 1              # user turns arrive one per exchange, in order
        if ava_opened and exchange == 0:
            continue
        if is_narrator_speaker(t.get("speaker")):
            continue
        if not (t.get("content") or "").strip():
            continue
        last = i
    return last


# Instructional tail for the READING lane (news / wander / lookups), the sibling of
# `FETCH_CLOSING` above. Separate wording rather than a shared one, because the two passes
# ask different questions of the same list: a chat turn is about to answer somebody, so the
# test is "does the message turn on this fact"; a reading pass is about to write down what
# it read, so the test is "does this text touch something already on record".
#
# The last sentence is the anti-narrowing rule and is the load-bearing one. Every eligible
# candidate on this box is a claim about the ONE person Ava talks to, so an instruction to
# pick "what matters to you" would reliably return his biography for any text at all, and
# the recap would drift from being about the events to being about him.
TEXT_FETCH_CLOSING = (
    "— end of the text —\n\n"
    "Now list the numbers of the facts worth having in mind while you write down what you "
    "just read, under the label FACTS:, one number per line, most important first. Judge "
    "what a fact is about, not which words it shares with the text — the list and the text "
    "may be in different languages. Pick a fact only where this text genuinely touches it: "
    "somewhere it names, someone it concerns, something it would mean for them. NONE if it "
    "touches none of them, which is the ordinary answer."
)


def build_text_body(index: str, framing: str, title: str, text: str,
                    window: int) -> str:
    """Assemble the reading lane's content: the numbered list, then the text.

    The counterpart of :func:`build_body`, and it differs in the one way that matters: there
    is no speaker and no conversation, so the material is presented as a *text that was
    read* rather than as a turn somebody took. Handing an article to `build_body` would put
    it under "The message that has just arrived", and the pass would reasonably fetch
    whatever it knew about the person it took to be speaking.

    The text is the elastic half of the budget and is trimmed from the END, where a
    conversation is trimmed from the start: a reply's referent is usually established in the
    oldest turns, while an article says what it is about in its opening. The list is never
    cut, for the same reason as there — narrowing what may be picked without saying so.
    """
    def _assemble(body: str) -> str:
        parts = ["The facts on record, numbered:", "", index, ""]
        if framing.strip():
            parts += [framing.strip(), ""]
        header = "TEXT" + (f" — {title.strip()}" if title.strip() else "")
        parts += [f"{header}:", "", body, "", TEXT_FETCH_CLOSING]
        return "\n".join(parts)

    body = str(text or "").strip()
    out = _assemble(body)
    budget = max(4000, int(window) * _CHARS_PER_TOKEN)
    # Floor of 500 chars of text: below that the pass is judging the framing line alone,
    # which is not a reading, and an empty-ish body should surface as a poor pick rather
    # than as a silently truncated one.
    while len(out) > budget and len(body) > 500:
        body = body[:max(500, int(len(body) * 0.8))].rstrip()
        out = _assemble(body)
    return out


# Instructional tail for the ASK lane — outreach's decision pass, weighing one of Ava's
# own open questions. A third closing rather than a reuse, for the reason the first two
# are already separate: each lane asks a different question of the same list. Chat's is
# "does the arriving message turn on this" (a lie here — nobody sent anything, and a pass
# handed the ask as an arriving message reasonably fetches facts about the person it takes
# to be speaking); the reading lane's is "does this text touch the record" in a
# recap-writing register. This lane's question is *"is the answer already on record?"* —
# the `resolved` branch of the decision downstream, made retrievable. The facts that
# matter most are exactly the ones that answer the question, which is why they are asked
# for by name.
ASK_FETCH_CLOSING = (
    "— end of the question —\n\n"
    "Now list the numbers of the facts worth having in front of you while you decide "
    "whether to raise this question — above all, any that answer it, or part of it, "
    "under the label FACTS:, one number per line, most important first. Judge what a "
    "fact is about, not which words it shares with the question — the list and the "
    "question may be in different languages. NONE if nothing on record touches it."
)


def build_ask_body(index: str, question: str, window: int) -> str:
    """Assemble the ask lane's content: the numbered list, then the question.

    The smallest of the four bodies, and the only one with no elastic half: an ask is a
    distilled line or short paragraph, bounded by the consolidation pass that wrote it,
    so there is nothing to trim and no budget loop. The *window* parameter is kept for
    signature symmetry with its siblings and for the day an ask stops being short.
    """
    del window
    return "\n".join([
        "The facts on record, numbered:", "", index, "",
        "The question you have been carrying:", "",
        str(question or "").strip(), "",
        ASK_FETCH_CLOSING,
    ])


# Instructional tail for the RE-READ lane — synthesis's analysis pass, re-reading an aged
# conversation as who she is now. The test differs from all three above: the pass
# downstream is about to form questions ("what do I only now think to wonder"), so the
# facts worth having are the ones that bear on which of those questions are ALREADY
# answered — a question the record can answer is not worth raising, and knowing that
# while the questions form beats vetting them afterwards.
REREAD_FETCH_CLOSING = (
    "— end of the conversation —\n\n"
    "Now list the numbers of the facts worth having in mind while you consider what "
    "this conversation leaves you wondering today, under the label FACTS:, one number "
    "per line, most important first — above all, any that already answer something it "
    "left open. Judge what a fact is about, not which words it shares with the "
    "conversation — the list and the conversation may be in different languages. NONE "
    "if nothing on record touches it, which is an ordinary answer."
)


def build_reread_body(index: str, turns: list, framing: str, window: int) -> str:
    """Assemble the re-read lane's content: the list, the framing, the old conversation.

    The transcript is rendered exactly as :func:`build_body` renders prior turns (the
    narrator-aware ``render_user_turn`` / ``Me:`` convention) and is the elastic half of
    the budget, trimmed **oldest-turn-first** like a conversation rather than end-first
    like an article: what the later turns wonder about is usually established earlier,
    but the *later* turns are where a conversation left things — and "what did this
    leave open" is the question the pass downstream is asking. The list is never cut.

    *framing* says whose conversation this is and when it happened — without it the pass
    reads an undated transcript of strangers, and a fact's relevance to "what has changed
    since" needs the since.
    """
    from core.reflection_source import render_user_turn

    rendered = []
    for t in turns:
        if t.get("role") == "user":
            rendered.append(render_user_turn(t.get("speaker", ""), t.get("content", "")))
        else:
            rendered.append(f"Me: {t.get('content', '')}")

    def _assemble() -> str:
        parts = ["The facts on record, numbered:", "", index, ""]
        if framing.strip():
            parts += [framing.strip(), ""]
        parts += ["The conversation:", "", "\n\n".join(rendered), "",
                  REREAD_FETCH_CLOSING]
        return "\n".join(parts)

    body = _assemble()
    budget = max(4000, int(window) * _CHARS_PER_TOKEN)
    while len(rendered) > 1 and len(body) > budget:
        rendered.pop(0)
        body = _assemble()
    return body


def build_body(index: str, prior_turns: list, speaker: str, message: str,
               window: int) -> str:
    """Assemble the pass's content: the numbered list, the conversation, the message.

    *prior_turns* is the WHOLE conversation before the arriving message, not a fixed tail:
    what a message refers to without naming is usually established earlier, and that
    referent is exactly what a fetch pass needs. The caller stops it before the arriving
    message's own reply — in simulation that reply exists, and showing it would hand the
    pass the answer it is fetching for.

    The transcript is the elastic half of the budget. An overlong conversation is trimmed
    oldest-turn-first; the candidate list is never cut, since that would silently narrow
    what the pass is allowed to pick without saying so.
    """
    from core.reflection_source import render_user_turn

    rendered = []
    for t in prior_turns:
        if t.get("role") == "user":
            rendered.append(render_user_turn(t.get("speaker", ""), t.get("content", "")))
        else:
            rendered.append(f"Me: {t.get('content', '')}")

    def _assemble() -> str:
        parts = ["The facts on record, numbered:", "", index, ""]
        if rendered:
            parts += ["The conversation so far:", "", "\n\n".join(rendered), ""]
        parts += ["The message that has just arrived:", "",
                  render_user_turn(speaker, message), "", FETCH_CLOSING]
        return "\n".join(parts)

    body = _assemble()
    budget = max(4000, int(window) * _CHARS_PER_TOKEN)
    while rendered and len(body) > budget:
        rendered.pop(0)
        body = _assemble()
    return body


def fetch_blob(*, doc: dict, prior_turns: list, speaker: str, message: str,
               generate: Callable, prompt: str, window: int, now: str,
               max_new_tokens: int = 512, til_max_age_days: Optional[int] = None,
               max_claims: Optional[int] = None) -> dict:
    """Run the whole pass and return the blob plus what it took to get there.

    The chat-side entry point. The workbench cannot use it — the module contract splits
    ``build`` and ``finish`` around the run loop's generation — so the two share the parts
    below rather than this wrapper.

    Never raises: a live turn must not fail because a retrieval channel did. Every failure
    path returns an empty ``text``, with ``skipped`` naming which one.
    """
    return _run(doc=doc,
                make_body=lambda index: build_body(index, prior_turns, speaker,
                                                   message, window),
                generate=generate, prompt=prompt, now=now,
                max_new_tokens=max_new_tokens, til_max_age_days=til_max_age_days,
                max_claims=max_claims)


def fetch_blob_for_text(*, doc: dict, framing: str, title: str, text: str,
                        generate: Callable, prompt: str, window: int, now: str,
                        max_new_tokens: int = 512, til_max_age_days: Optional[int] = None,
                        max_claims: Optional[int] = None, max_reports: int = 0,
                        describe_source: Optional[Callable] = None) -> dict:
    """The reading lane's entry point: which facts to have in mind while recapping a text.

    Same pass, same candidate list, same safety boundary as :func:`fetch_blob` — only the
    body differs (:func:`build_text_body`). Returns the identical shape, so a caller can
    report it with the chat channel's vocabulary.

    ``max_reports > 0`` widens the candidate list to `graph.blob.READING_FACETS` — knowledge
    plus ``report``, what a text asserted — and renders those attributed. The two travel
    together on ONE parameter deliberately: offering reports the renderer has no budget for
    would put them in front of the pass only to drop whatever it picked, and budgeting for
    reports that were never offered would silently do nothing. `position` is not offered at
    either setting.
    """
    from graph import blob as graph_blob

    return _run(doc=doc,
                make_body=lambda index: build_text_body(index, framing, title, text,
                                                        window),
                generate=generate, prompt=prompt, now=now,
                max_new_tokens=max_new_tokens, til_max_age_days=til_max_age_days,
                max_claims=max_claims,
                facets=(graph_blob.READING_FACETS if max_reports > 0 else None),
                render_kw={"max_reports": int(max_reports),
                           "describe_source": describe_source})


def fetch_blob_for_ask(*, doc: dict, question: str, generate: Callable, prompt: str,
                       window: int, now: str, max_new_tokens: int = 512,
                       til_max_age_days: Optional[int] = None,
                       max_claims: Optional[int] = None) -> dict:
    """The ask lane's entry point: which facts bear on a question she is carrying.

    Feeds outreach's decision pass — the one place on the box that formally decides
    "have I already learned the answer?" (its ``resolved`` branch evicts the ask for
    good). Until this lane existed that judgement ran on cosine retrieval alone, which
    is exactly what the fetch channel was built to compensate: the record is mostly
    English against often-Russian material, and a fact that landed *after* the ask was
    queued — the whole substance of ``resolved`` — embeds on a trigger the ask's wording
    may never touch.

    Same pass, same candidate list, same safety boundary as :func:`fetch_blob`; only the
    body differs (:func:`build_ask_body`). Knowledge facets only — no ``report`` claims:
    a ``resolved`` verdict retires a live question on the strength of what is injected,
    and "a text asserted X" resolving "is X true?" is the epistemic slip the facet level
    exists to prevent. Returns the identical shape, so a caller reports it with the chat
    channel's vocabulary and hands ``sources`` to the nomination slot unchanged.
    """
    return _run(doc=doc,
                make_body=lambda index: build_ask_body(index, question, window),
                generate=generate, prompt=prompt, now=now,
                max_new_tokens=max_new_tokens, til_max_age_days=til_max_age_days,
                max_claims=max_claims)


def fetch_blob_for_reread(*, doc: dict, turns: list, framing: str, generate: Callable,
                          prompt: str, window: int, now: str, max_new_tokens: int = 512,
                          til_max_age_days: Optional[int] = None,
                          max_claims: Optional[int] = None) -> dict:
    """The re-read lane's entry point: which facts bear on an aged conversation.

    Feeds synthesis's analysis pass, so its questions *form* against the record instead
    of being vetted against it afterwards — synthesis composes and sends its own opener
    directly (it does not route through outreach's decision), so a "do I already know
    this" check anywhere downstream would not cover it. Knowledge facets only, for the
    ask lane's reason one step earlier: what gets suppressed here is a question, and a
    text's assertion should not be what decides a question is not worth asking.

    Same pass, same candidate list, same safety boundary; only the body differs
    (:func:`build_reread_body`).
    """
    return _run(doc=doc,
                make_body=lambda index: build_reread_body(index, turns, framing, window),
                generate=generate, prompt=prompt, now=now,
                max_new_tokens=max_new_tokens, til_max_age_days=til_max_age_days,
                max_claims=max_claims)


def _run(*, doc: dict, make_body: Callable, generate: Callable, prompt: str, now: str,
         max_new_tokens: int, til_max_age_days: Optional[int],
         max_claims: Optional[int], facets: Optional[tuple] = None,
         render_kw: Optional[dict] = None) -> dict:
    """Everything the two entry points share: candidates → pick → render → report.

    Factored so the lanes cannot drift on the part that has safety consequences. What may
    be *shown* is decided in `graph.blob` before either body is built; what is *asked* is
    the only thing either caller chooses.
    """
    from graph import blob as graph_blob

    kw = {}
    if til_max_age_days is not None:
        kw["til_max_age_days"] = int(til_max_age_days)
    if facets:
        kw["facets"] = tuple(facets)
    try:
        cands = graph_blob.claim_candidates(doc, now=now, **kw)
    except Exception as e:
        return {"text": "", "skipped": "candidates_failed", "error": str(e)}
    if not cands.get("claims"):
        return {"text": "", "skipped": "no_candidates", "candidates": cands}

    index = graph_blob.claim_index(doc, cands)
    body = make_body(index)
    try:
        raw = generate(body, prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        # A CUDA OOM is named apart from every other generation failure: it says
        # nothing about this pass and everything about the box (fragmentation, the
        # VRAM ceiling), so the label must send the operator to the allocator rather
        # than to the prompt. Classified by type-or-message (`alloc_guard.is_cuda_oom`)
        # since a wrapped OOM can arrive as a plain RuntimeError carrying the text.
        from core.alloc_guard import is_cuda_oom
        return {"text": "", "skipped": "oom" if is_cuda_oom(e) else "generate_failed",
                "error": str(e), "candidates": cands}

    sel = graph_blob.parse_claim_selection(raw or "", cands)
    rkw = {"max_claims": int(max_claims)} if max_claims else {}
    rkw.update({k: v for k, v in (render_kw or {}).items() if v is not None})
    out = graph_blob.render_claims(doc, sel["picked"], **rkw)
    return {
        "text": out["text"],
        "picked": sel["numbers"],
        "claims": out["claims"],
        "n_rendered": out["n_rendered"],
        # The sources behind the claims that actually made it into the blob — not behind
        # everything picked, since a claim the render caps dropped conditions nothing and
        # its source should not be recalled on its behalf. What the caller does with them
        # is its own decision (`rag_engine` nominates them for recall); this only answers
        # "where did these come from", which is a question about the tree.
        #
        # `[(lane, ref)]`, typed: a chat ref is a session filename and a TIL ref a
        # `<kind>/<stem>` under the snippets tree, and they resolve through different
        # stores. `sessions` is kept beside it as the chat-only view, because the client
        # renders session names and has no business knowing the snippet layout.
        "sources": graph_blob.sources(doc, out["claims"]),
        "sessions": graph_blob.chat_sources(doc, out["claims"]),
        "out_of_range": sel["out_of_range"],
        "none": sel["none"],
        "candidates": cands,
        "raw": raw,
        "skipped": "" if out["text"] else ("picked_nothing" if sel["none"] else "empty"),
    }


# -- self-test --------------------------------------------------------------- #

def _selftest() -> None:
    import sys
    from pathlib import Path
    server_dir = Path(__file__).resolve().parent.parent.parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    print("the arriving-message rule")
    reach = {"initiated_by": "ava"}
    check("an Ava-initiated exchange 0 is not a message",
          last_real_user_turn(reach, [{"role": "user", "speaker": "", "content": "impulse"},
                                      {"role": "assistant", "content": "x"}]), None)
    check("a stage-direction speaker is not either",
          last_real_user_turn({}, [{"role": "user", "speaker": "(setting)",
                                    "content": "framing"}]), None)
    check("an answered reach-out resolves to the real turn",
          last_real_user_turn(reach, [
              {"role": "user", "speaker": "(initiative)", "content": "impulse"},
              {"role": "assistant", "content": "opener"},
              {"role": "user", "speaker": "artemyvo", "content": "reply"}]), 2)

    print("\nthe assembled body")
    turns = [{"role": "user", "speaker": "artemyvo", "content": "first"},
             {"role": "assistant", "content": "answered first"}]
    body = build_body("[1] x: a fact.", turns, "artemyvo", "the message", 24576)
    check("the numbered list leads", body.startswith("The facts on record"), True)
    check("prior turns are present", "first" in body and "answered first" in body, True)
    check("the arriving message is last before the task",
          body.index("the message") < body.index(FETCH_CLOSING), True)
    check("the task is the final thing", body.rstrip().endswith(
        "NONE if nothing needs looking up."), True)
    check("the pass is asked for numbers", "FACTS:" in body, True)

    print("\nbudget: the transcript is trimmed, never the list")
    fat = [{"role": "user", "speaker": "u", "content": "x" * 2000} for _ in range(40)]
    small = build_body("[1] keep: this line.", fat, "u", "m", 1000)
    check("the list survives a squeeze", "[1] keep: this line." in small, True)
    check("...and the body respects the floor", len(small) <= 4000 + 2100, True)
    check("the arriving message always survives", "m" in small, True)

    print("\nthe reading lane's body")
    tbody = build_text_body("[1] x: a fact.", "This is a digest of world events.",
                            "2026-07-27", "the article text", 24576)
    check("the numbered list leads here too", tbody.startswith("The facts on record"), True)
    check("the framing and title are present",
          "digest of world events" in tbody and "2026-07-27" in tbody, True)
    check("the text is presented as a text, not as somebody's turn",
          "message that has just arrived" not in tbody, True)
    check("the task is the final thing", tbody.rstrip().endswith(
        "which is the ordinary answer."), True)

    print("\nbudget: the text is trimmed from the end, never the list")
    tsmall = build_text_body("[1] keep: this line.", "f", "t", "y" * 40000, 1000)
    check("the list survives a squeeze", "[1] keep: this line." in tsmall, True)
    check("...and the body respects the floor", len(tsmall) <= 4000 + 1200, True)
    check("...and the opening of the text is what survives",
          tsmall.count("y") >= 500, True)

    print("\nthe ask lane's body")
    abody = build_ask_body("[1] x: a fact.", "Was the answer ever recorded?", 24576)
    check("the numbered list leads here too", abody.startswith("The facts on record"), True)
    check("the question is present and framed as hers",
          "Was the answer ever recorded?" in abody
          and "question you have been carrying" in abody, True)
    check("the question is never presented as an arriving message",
          "message that has just arrived" not in abody, True)
    check("the task is the final thing", abody.rstrip().endswith(
        "NONE if nothing on record touches it."), True)

    print("\nthe re-read lane's body")
    old_turns = [{"role": "user", "speaker": "artemyvo", "content": "first thing said"},
                 {"role": "assistant", "content": "what I answered"}]
    rbody = build_reread_body("[1] x: a fact.", old_turns,
                              "An old conversation of yours with artemyvo.", 24576)
    check("the numbered list leads here too", rbody.startswith("The facts on record"), True)
    check("the framing and both sides of the conversation are present",
          "An old conversation" in rbody and "first thing said" in rbody
          and "Me: what I answered" in rbody, True)
    check("the task is the final thing", rbody.rstrip().endswith(
        "which is an ordinary answer."), True)

    print("\nbudget: the re-read transcript is trimmed oldest-first, never the list")
    rfat = ([{"role": "user", "speaker": "u", "content": "OLDEST " + "x" * 2000}]
            + [{"role": "user", "speaker": "u", "content": "y" * 2000} for _ in range(38)]
            + [{"role": "user", "speaker": "u", "content": "NEWEST"}])
    rsmall = build_reread_body("[1] keep: this line.", rfat, "f", 1000)
    check("the list survives a squeeze", "[1] keep: this line." in rsmall, True)
    check("...the oldest turn is what went", "OLDEST" in rsmall, False)
    check("...and the newest survives", "NEWEST" in rsmall, True)

    print("\nfetch_blob never raises")
    doc = {"nodes": {"person:a": {"id": "person:a", "label": "a"}},
           "claims": {"c1": {"claim_id": "c1", "node": "person:a", "facet": "property",
                             "text": "a fact", "n_sources": 1, "n_occurrences": 1,
                             "lanes": ["chat"], "last_asserted": "2026-08-11",
                             "mentions": [], "when": "", "occurrences": [0]}},
           "occurrences": [{"lane": "chat", "source_ref": "20260811_120000.json",
                            "asserted_at": "2026-08-11"}]}
    common = dict(doc=doc, prior_turns=[], speaker="u", message="m", prompt="p",
                  window=24576, now="2026-08-12")

    def boom(*a, **k):
        raise RuntimeError("gpu on fire")

    r = fetch_blob(generate=boom, **common)
    check("a failing generation degrades to no block", (r["text"], r["skipped"]),
          ("", "generate_failed"))

    def boom_oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 498.00 MiB")

    r = fetch_blob(generate=boom_oom, **common)
    check("an OOM degrades the same way but is NAMED as the allocator's failure",
          (r["text"], r["skipped"]), ("", "oom"))
    check("...and the named reason is in the broken-channel set",
          "oom" in FETCH_FAILURES, True)
    r = fetch_blob(generate=lambda *a, **k: "FACTS:\n1", **common)
    check("a good pick renders", "a fact" in r["text"], True)
    check("...and reports what it picked", r["picked"], [1])
    # The nomination input: a rendered claim carries the conversation it came out of, so the
    # caller can recall it without re-reading the tree.
    check("...and the conversation behind it", r["sessions"], ["20260811_120000.json"])
    check("...typed, since the two lanes resolve through different stores",
          r["sources"], [("chat", "20260811_120000.json")])
    check("a refusal nominates nothing",
          fetch_blob(generate=lambda *a, **k: "FACTS:\nNONE", **common)["sessions"], [])
    r = fetch_blob(generate=lambda *a, **k: "FACTS:\nNONE", **common)
    check("a refusal is empty and named", (r["text"], r["skipped"]),
          ("", "picked_nothing"))
    r = fetch_blob(generate=lambda *a, **k: "FACTS:\n1",
                   **{**common, "doc": {"nodes": {}, "claims": {}, "occurrences": []}})
    check("an empty tree is a skip, not a crash", r["skipped"], "no_candidates")

    # The reading lane goes through the same `_run`, so it inherits every guarantee above.
    tcommon = dict(doc=doc, framing="f", title="t", text="an article", prompt="p",
                   window=24576, now="2026-08-12")
    check("a failing generation degrades to no block here too",
          fetch_blob_for_text(generate=boom, **tcommon)["skipped"], "generate_failed")
    tr = fetch_blob_for_text(generate=lambda *a, **k: "FACTS:\n1", **tcommon)
    check("a good pick renders", "a fact" in tr["text"], True)
    check("...and carries its source, so a recap can be traced to it",
          tr["sources"], [("chat", "20260811_120000.json")])
    check("picking nothing is empty and named — the expected answer for most texts",
          fetch_blob_for_text(generate=lambda *a, **k: "FACTS:\nNONE",
                              **tcommon)["skipped"], "picked_nothing")

    # The ask and re-read lanes go through the same `_run` as well.
    acommon = dict(doc=doc, question="was it ever answered?", prompt="p",
                   window=24576, now="2026-08-12")
    check("the ask lane degrades on a failing generation",
          fetch_blob_for_ask(generate=boom, **acommon)["skipped"], "generate_failed")
    ar = fetch_blob_for_ask(generate=lambda *a, **k: "FACTS:\n1", **acommon)
    check("an ask-lane pick renders and carries its sources",
          ("a fact" in ar["text"], ar["sources"]),
          (True, [("chat", "20260811_120000.json")]))
    rcommon = dict(doc=doc, turns=old_turns, framing="f", prompt="p",
                   window=24576, now="2026-08-12")
    check("the re-read lane degrades on a failing generation",
          fetch_blob_for_reread(generate=boom, **rcommon)["skipped"], "generate_failed")
    rr = fetch_blob_for_reread(generate=lambda *a, **k: "FACTS:\n1", **rcommon)
    check("a re-read-lane pick renders and carries its sources",
          ("a fact" in rr["text"], rr["sources"]),
          (True, [("chat", "20260811_120000.json")]))

    # The attributed-report channel: `max_reports` must widen the OFFER and the RENDER
    # together, or the pass is shown claims nothing will carry.
    rdoc = {**doc, "claims": {**doc["claims"],
                              "c2": {"claim_id": "c2", "node": "person:a",
                                     "facet": "report", "text": "a wire report",
                                     "n_sources": 1, "n_occurrences": 1, "lanes": ["til"],
                                     "last_asserted": "2026-08-11", "mentions": [],
                                     "when": "", "occurrences": [0]}}}
    seen: list = []

    def spy(body, prompt, *, max_new_tokens):
        seen.append(body)
        return "FACTS:\n1\n2"

    r = fetch_blob_for_text(**{**tcommon, "doc": rdoc, "generate": spy})
    check("reports are not offered at the default", "a wire report" in seen[-1], False)
    r = fetch_blob_for_text(**{**tcommon, "doc": rdoc, "generate": spy},
                            max_reports=2, describe_source=lambda l, ref: "a digest")
    check("...and are, once budgeted", "a wire report" in seen[-1], True)
    check("...rendered attributed, never as knowledge",
          "a digest reported: a wire report." in r["text"], True)
    # Freshness is a caller knob, defaulting to no scope at all: an old TIL claim is offered
    # exactly like an old chat one, and a live turn can still tighten it.
    stale = {**doc, "claims": {"c1": {**doc["claims"]["c1"], "lanes": ["til"],
                                      "last_asserted": "2026-01-01"}}}
    r = fetch_blob(generate=lambda *a, **k: "FACTS:\n1", **{**common, "doc": stale})
    check("an old TIL claim is offered by default", "a fact" in r["text"], True)
    r = fetch_blob(generate=lambda *a, **k: "FACTS:\n1",
                   **{**common, "doc": stale, "til_max_age_days": 7})
    check("an explicit scope still drops it", r["skipped"], "no_candidates")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
