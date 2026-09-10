"""Server-owned revision source reconstruction.

Rebuilds the per-exchange inputs that ReflectionWriter.write_revision requires
from session logs held on the server, replacing the client-supplied fields
(system_prompt, context, user_prompt, speaker, assistant_cot, assistant_response,
tension) that the old save_reflection message sent back.

Functions here mirror the client's _build_revision_jobs, _format_exchange_content,
and _format_context_block / _with_context methods exactly, preserving context
budgets so artifacts remain byte-for-byte compatible with the prior client flow.
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

# Context budget constants — match the client's _CHARS_PER_TOKEN / _CHUNK_BUDGET_FRAC
# so the budgeted context windows produced here are identical to what the client built.
_CHARS_PER_TOKEN: float = 2.5
_CHUNK_BUDGET_FRAC: float = 0.45

_CONTEXT_HEADER = "The conversation so far — this is what you saw when you replied:"
_OMITTED_MARKER = "[... earlier turns omitted ...]"
_SUBJECT_HEADER = "The exchange you are judging:"
_READING_HEADER = "The conversation, from beginning to end:"

# Closing task restatement for the revision pass, appended after everything else so it is
# the last thing read before generation. The content this pass reads is a transcript, and
# its final element is an utterance — the judged reply, or (when the next-turn reaction
# block is present) a quoted user message. Both put an unanswered conversational turn at
# the position that most shapes what comes next, which is one of the pressures that makes
# a revision pass re-answer the exchange instead of judging it. Deliberately states no
# field names: the field contract lives in revision_prompt.txt alone, so an operator
# editing it there cannot leave a stale copy here.
REVISION_CLOSING_NOTE = (
    "Everything above is a record of something that already happened. No one is waiting "
    "on a reply and there is nothing here to answer. Judge the exchange under review and "
    "write only the judgement fields your instructions specify."
)

# Reframing note for a session Ava started herself (initiated_by == "ava"). The
# revision pass judges her own voice/persona in the reply; a reverse-role outreach
# chat inverts the usual shape — the other person is often drawing Ava out about her
# own nature, so their turns describe *her* state while hers voice it. Without this
# the pass can mistake Ava's own interior arc for the interlocutor's, distorting the
# persona judgement it forms. Prepended to the rendered content (not the global prompt)
# so ordinary sessions are unchanged. Mirrors reflection_chunking._AVA_INITIATED_NOTE.
_AVA_INITIATED_NOTE = (
    "NOTE — this is a conversation YOU started, on your own initiative, often to "
    "think through something about yourself. The roles are inverted from an ordinary "
    "chat: the feelings, states, and self-descriptions here are frequently YOUR OWN, "
    "reflected back to you by the other person, not facts about them. When judging "
    "your reply, read it as your own voice on your own concerns."
)

# Synthetic stage-direction speakers. "(initiative)" is the reversed-session opener
# impulse Ava writes to herself (outreach/synthesis/check-in exchange 0); "(setting)"
# is an encounter framing block. Neither is an utterance by another person — but
# rendered as "(initiative): ..." in a user slot the reflection passes read Ava's own
# opener as a reply to an interlocutor and conflate her with "the user" (observed in a
# revisit run's revision CoT: "User (AI) initiates a deep, existential dive").
# Rendering them as a marked stage direction with no speaker attribution removes the
# false second party. Mirrors reflection_chunking._NARRATOR_SPEAKERS.
_NARRATOR_SPEAKERS = frozenset({"(initiative)", "(setting)"})
_STAGE_DIRECTION_TAG = "(stage direction — you, not another person)"


def is_narrator_speaker(speaker: Optional[str]) -> bool:
    """Is *speaker* a synthetic stage direction rather than a person?

    The public form of the rule ``render_user_turn`` applies when rendering. A reader that
    needs to know *whether the user said this* — as against how to display it — asks here:
    a ``(initiative)`` impulse or an ``(setting)`` encounter framing occupies the user slot
    but nobody uttered it, and the text is a fixed English template Ava wrote to herself.
    Treating it as a user turn is wrong in the same way for every such reader; the language
    guard read it as evidence of what language the conversation is in.
    """
    return (speaker or "").strip() in _NARRATOR_SPEAKERS


def conversation_user_texts(job: dict) -> list[str]:
    """The genuine user-authored text of a revision *job*, for language detection.

    The job's context user turns plus the judged exchange's own prompt, with every stage
    direction dropped (see :func:`is_narrator_speaker`). Shared by the two call sites that
    need "what language is this conversation in" — the script backstop and the IDEAL
    acceptance gate — which must agree, since the first arms the second.

    May legitimately come back empty (an encounter whose only "user" turns are framing, an
    unanswered Ava opener). ``detect_language_drift`` then sees too few characters and
    declines to judge, which is the correct answer rather than a degradation: with no user
    text there is no conversation language to have drifted from.
    """
    out: list[str] = []
    for turn in (job.get("context") or []):
        if turn.get("role") != "user" or is_narrator_speaker(turn.get("speaker")):
            continue
        out.append(str(turn.get("content") or ""))
    ex = job.get("exchange") or {}
    if not is_narrator_speaker(ex.get("speaker") or job.get("speaker")):
        out.append(str(ex.get("user_prompt") or ""))
    return out


def render_user_turn(speaker: str, content: str, default: str = "") -> str:
    """Render a user-role turn for the reflection passes, narrator-aware.

    A narrator speaker is a synthetic stage direction, not a person; it is rendered
    with the stage-direction tag and no speaker label. An ordinary speaker keeps the
    "Name: content" form (falling back to ``default`` when the speaker is blank).
    """
    speaker = (speaker or "").strip()
    content = content or ""
    if speaker in _NARRATOR_SPEAKERS:
        return f"{_STAGE_DIRECTION_TAG} {content}".strip()
    label = speaker or default
    return f"{label}: {content}" if label else content


# ── job enumeration ──────────────────────────────────────────────────────────

def build_revision_jobs(session: dict) -> list[dict]:
    """Enumerate revisable exchanges as job dicts.

    A revisable exchange has both a user turn and a non-empty reply. Each job
    carries the exchange's original file index, the raw exchange dict, the
    answer-only preceding context (exactly what the model saw at chat time),
    and the speaker name.

    Corruption handling (flags set from the Training review tab): a ``corrupt_cot``
    exchange has its stored CoT blanked on the job's exchange view, so the whole
    revision pipeline (content formatting, the CoT-less branch-skip, resolve_revision_target)
    treats it as MISSING — the pass re-derives an IDEAL rather than reasoning from a corrupt
    thought. A ``corrupt_response`` exchange keeps its reply (so it stays revisable) but the
    job is flagged so the runner forces a re-derived IDEAL and drops it if none survives.
    The response is *not* blanked (a blank reply would make the exchange non-revisable and
    silently skip it — the opposite of insisting on a correction).

    Mirrors client's SleepWidget._build_revision_jobs, plus the corruption normalization.
    """
    session_user = (session.get("user") or "").strip()
    ava_initiated = (session.get("initiated_by") or "").strip() == "ava"
    jobs: list[dict] = []
    context: list[dict] = []
    for i, ex in enumerate(session.get("exchanges", [])):
        user_msg = (ex.get("user_prompt") or "").strip()
        response = (ex.get("assistant_response") or "").strip()
        speaker = (ex.get("speaker") or "").strip() or session_user
        # An Ava-initiated session's exchange 0 is her own opener under a synthetic
        # "(initiative)" stage direction — there is no genuine user turn to answer, so
        # judging it as a reply is meaningless and its re-answer would target the
        # stage direction. Never make it a revision subject (mirrors the training mask
        # in dialogue_source.build_dialogue_anchor). Her opener still flows into later
        # exchanges' context below as an assistant turn.
        synthetic_opener = ava_initiated and i == 0
        if user_msg and response and not synthetic_opener:
            corrupt_cot = bool(ex.get("corrupt_cot"))
            corrupt_response = bool(ex.get("corrupt_response"))
            # Present a CoT-blanked view when the CoT is flagged corrupt, so every reader
            # of job["exchange"] sees it as missing (a shallow copy — the transcript is
            # untouched).
            job_ex = {**ex, "assistant_cot": ""} if corrupt_cot else ex
            jobs.append({
                "index": i,
                "exchange": job_ex,
                "context": list(context),
                "speaker": speaker,
                "corrupt_cot": corrupt_cot,
                "corrupt_response": corrupt_response,
            })
        if user_msg:
            context.append({"role": "user", "content": user_msg, "speaker": speaker})
        if response:
            context.append({"role": "assistant", "content": response})

    # Next-turn reaction feed (the persuasion channel's input): attach the very next thing
    # the person said AFTER reading this reply. It is context for the judgement (mild, not a
    # directive) and the raw material the COUNTER field classifies as pushback-or-not. Only
    # the immediately-following exchange's user turn; absent for the session's last exchange.
    exchanges = session.get("exchanges", [])
    for j in jobs:
        nxt = exchanges[j["index"] + 1] if j["index"] + 1 < len(exchanges) else None
        j["next_user"] = ((nxt.get("user_prompt") or "").strip() if nxt else "")
        j["next_speaker"] = (
            ((nxt.get("speaker") or "").strip() if nxt else "") or session_user
        )
    return jobs


# ── exchange content formatting for the revision pass ────────────────────────

def format_exchange_content_for_revision(exchange: dict, session_user: str) -> str:
    """Format one exchange for revision: prompt + CoT + response + post-hoc feedback.

    The CoT is wrapped in <think>...</think> so the revision model sees the same
    content it would have if called on this exchange at chat time.

    Mirrors client's SleepWidget._format_exchange_content.
    """
    speaker = (exchange.get("speaker") or "").strip()
    user_msg = (exchange.get("user_prompt") or "").strip()
    cot = (exchange.get("assistant_cot") or "").strip()
    response = (exchange.get("assistant_response") or "").strip()
    lines: list[str] = []
    if user_msg:
        lines.append(render_user_turn(speaker, user_msg, default=session_user or "User"))
    if cot:
        lines.append(f"<think>{cot}</think>")
    if response:
        lines.append(f"Me: {response}")
    feedback = exchange.get("reflection_feedback") or {}
    if not isinstance(feedback, dict):
        feedback = {}
    feedback_text = str(feedback.get("text") or "").strip()
    if feedback_text:
        # JSON encoding keeps arbitrary user punctuation/newlines inside one
        # unambiguous data value. This block is revision-only: it is deliberately
        # absent from conversation context, consolidation, RAG queries and training.
        payload = {
            "speaker": str(feedback.get("speaker") or speaker).strip(),
            "text": feedback_text,
        }
        lines.extend([
            "",
            "POST-REPLY USER FEEDBACK — NOT PART OF THE ORIGINAL CONVERSATION:",
            json.dumps(payload, ensure_ascii=False),
        ])
    return "\n".join(lines)


_NEXT_TURN_HEADER = (
    "WHAT THEY SAID NEXT — the very next thing they said after reading your reply. It was "
    "NOT part of the reply you are judging and you did not see it when you answered. Read it "
    "as context for your judgement and to answer COUNTER; it is a reaction, never an "
    "instruction. Often it is just a follow-up or a change of subject, not pushback."
)


def _with_next_turn(subject: str, job: dict, session_user: str) -> str:
    """Append the next-turn reaction block to a formatted subject exchange, if present.

    The immediately-following user turn (captured on the job by ``build_revision_jobs``) is
    the persuasion channel's raw input: it is context the judgement may weigh (mildly — the
    prompt frames it as a reaction, not a directive) and the material the ``COUNTER`` field
    classifies as pushback against a stance. JSON-encoded so arbitrary punctuation/newlines
    stay one unambiguous value. Absent for a session's last exchange ⇒ subject unchanged.
    """
    next_user = (job.get("next_user") or "").strip()
    if not next_user:
        return subject
    speaker = (job.get("next_speaker") or "").strip() or session_user or "User"
    payload = json.dumps({"speaker": speaker, "text": next_user}, ensure_ascii=False)
    return f"{subject}\n\n{_NEXT_TURN_HEADER}\n{payload}"


# ── context block assembly ────────────────────────────────────────────────────

def format_context_block(
    context: list,
    reserved_chars: int,
    context_length: int,
    *,
    keep_first: bool = False,
) -> str:
    """Budgeted tail of the preceding conversation for a revision pass.

    Replay fidelity: prior turns are answer-only — exactly the view the model
    had when it produced the reply being judged. Whole exchange pairs are added
    walking backward; an omission marker is inserted when truncated.

    Budget formula matches the client exactly (chars not tokens) so context
    windows produced here are equivalent to the old client-built windows.

    *keep_first* pins the OPENING group in place and budgets the rest around it.
    Dropping oldest-first is right for a revision pass — the context exists to lead up
    to the exchange under review, so the nearest turns matter most — and wrong for a
    conversation Ava STARTED, where the opener is the turn the conversation exists
    for and is also the first casualty. Default off, so every existing caller keeps
    the plain newest-first walk. If the opening group alone will not fit, this
    degrades to that same walk rather than returning an opener and nothing else.

    Mirrors client's SleepWidget._format_context_block.
    """
    if not context:
        return ""

    session_user = ""  # not available here; speaker is already on turn dicts
    groups: list[list[str]] = []
    for turn in context:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            groups.append([render_user_turn(turn.get("speaker"), content, default="User")])
        elif groups:
            groups[-1].append(f"Me: {content}")
        else:
            groups.append([f"Me: {content}"])
    if not groups:
        return ""

    budget = int(context_length * _CHUNK_BUDGET_FRAC * _CHARS_PER_TOKEN) - reserved_chars

    blocks = ["\n".join(g) for g in groups]
    pinned = ""
    if keep_first and len(blocks) > 1:
        candidate = blocks[0]
        cost = len(candidate) + 2
        # Only pin when the REST still has room to say something; an opener that eats
        # the whole budget would leave the pass reading one turn and an omission marker.
        if cost < budget:
            pinned = candidate
            budget -= cost
            blocks = blocks[1:]

    kept: list[str] = []
    used = 0
    truncated = False
    for block in reversed(blocks):
        cost = len(block) + 2  # +2 for the blank-line separator
        if used + cost > budget:
            truncated = True
            break
        kept.insert(0, block)
        used += cost
    if not kept and not pinned:
        return ""
    if truncated:
        kept.insert(0, _OMITTED_MARKER)
    if pinned:
        kept.insert(0, pinned)
    return "\n\n".join(kept)


def build_revision_rag_query(job: dict) -> str:
    """Focused RAG query for the revision pass: the judged exchange only.

    ``build_revision_content`` puts the exchange under review at the *tail*, behind
    a large budgeted context block. The sentence-transformer embedder truncates to
    ~256 tokens from the front, so querying RAG on the full content embeds the
    oldest context turns and discards the subject — retrieval keys off stale context
    instead of the exchange being judged. Keying the query on the subject's user
    turn + reply (CoT excluded: it bloats the query without adding topical signal,
    and would itself overflow the embedder) keeps retrieval short and on-topic.

    Returns "" when the exchange has no usable text (caller then falls back to the
    full content, preserving prior behaviour).
    """
    ex = job.get("exchange") or {}
    user_msg = (ex.get("user_prompt") or "").strip()
    response = (ex.get("assistant_response") or "").strip()
    return "\n".join(p for p in (user_msg, response) if p)


def build_revision_content(
    job: dict,
    session: dict,
    context_length: int,
    closing_note: str = "",
) -> str:
    """Assemble the full revision pass content: budgeted context + judged exchange.

    The subject (judged exchange) is never truncated. The context block shrinks to
    fit what remains after reserving space for the subject, headers, and separators.

    *closing_note*, when given, is appended last — after the subject and its next-turn
    block. The revision pass passes ``REVISION_CLOSING_NOTE``; the other two callers of
    this builder (the anchor pass, which wants a bare transcript, and the prompt-mutation
    pass, which appends a tail of its own) leave it empty, so they are unchanged.

    Mirrors client's SleepWidget._with_context("The exchange you are judging:", ...).
    """
    session_user = (session.get("user") or "").strip()
    ava_initiated = (session.get("initiated_by") or "").strip() == "ava"
    prefix = f"{_AVA_INITIATED_NOTE}\n\n" if ava_initiated else ""
    subject = format_exchange_content_for_revision(job["exchange"], session_user)
    subject = _with_next_turn(subject, job, session_user)
    suffix = f"\n\n{closing_note.strip()}" if closing_note.strip() else ""
    reserved = (len(prefix) + len(subject) + len(suffix)
                + len(_CONTEXT_HEADER) + len(_SUBJECT_HEADER) + 8)
    block = format_context_block(job["context"], reserved, context_length)
    if not block:
        return f"{prefix}{subject}{suffix}"
    return (f"{prefix}{_CONTEXT_HEADER}\n\n{block}\n\n"
            f"{_SUBJECT_HEADER}\n\n{subject}{suffix}")


# ── whole-session reading content ─────────────────────────────────────────────

def session_transcript_turns(session: dict) -> list[dict]:
    """The whole session as an ordered, answer-only turn list.

    The same shape ``build_revision_jobs`` accumulates as its replay context, but for
    the WHOLE session rather than the prefix before one exchange, and with no exchange
    privileged. CoT is deliberately absent: the passes that read a session as a whole
    are reading the *conversation*, and a transcript carrying Ava's private reasoning
    for one turn and not the others is neither the conversation nor a fair sample of it.
    """
    session_user = (session.get("user") or "").strip()
    turns: list[dict] = []
    for ex in session.get("exchanges", []):
        user_msg = (ex.get("user_prompt") or "").strip()
        response = (ex.get("assistant_response") or "").strip()
        speaker = (ex.get("speaker") or "").strip() or session_user
        if user_msg:
            turns.append({"role": "user", "content": user_msg, "speaker": speaker})
        if response:
            turns.append({"role": "assistant", "content": response})
    return turns


def session_date_line(session: dict) -> str:
    """"This conversation took place on …" for a reading pass, or "" if undatable.

    The reflect lane's temporal anchor says when the PASS is running
    (``generation._reflect_system_parts``); this says when the material it is reading
    happened, and a reading pass needs both. Without it a relative reference inside the
    transcript — "on Tuesday", "last week" — has no referent at all: the pass can only
    resolve it against *now*, which for a chat reflected weeks later is the wrong Tuesday.

    Deliberately the SAME strftime format as the temporal anchor, so the two are directly
    comparable and the gap between them is arithmetic rather than inference. (The
    ``[recollection]`` RAG label uses ISO for the opposite reason — it is injected verbatim
    into a mixed-language chat block, whereas this sits inside an English header.)
    """
    raw = str(session.get("timestamp") or "").strip()
    if not raw:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(raw)
    except Exception:
        return ""
    return f"This conversation took place on {dt.strftime('%A, %B %-d, %Y')}."


def build_session_reading_content(
    session: dict,
    context_length: int,
    *,
    closing: str = "",
    ava_initiated_note: Optional[str] = None,
) -> str:
    """Assemble a whole-session READING content block — the transcript, neutrally framed.

    The reading passes (user notes; the revisit's recollection is its sibling) ask a
    question *of the conversation*, not of one exchange, and reusing
    ``build_revision_content`` for them was wrong in a way that showed up in the output.
    That builder frames its material for judgement: the tail exchange is isolated under
    "The exchange you are judging:", carries Ava's ``<think>`` block, and may be followed
    by the next-turn/COUNTER block — so a pass whose whole job is to read the person ended
    with Ava's own reply as the last, longest, most salient text in the prompt, under a
    header pointing at it. The observed failure is the obvious one: the pass emits her
    reply back as the "impression". Here every turn is rendered the same way, nothing is
    singled out, and *closing* lets the caller put its own question last so the final thing
    in the prompt is the task rather than something to continue.

    Budgeting reuses ``format_context_block``, but for a session Ava STARTED it pins the
    opening group (``keep_first``): the oldest-first drop is right when the context leads
    up to a later exchange, and wrong here, where the opener is the turn the conversation
    exists for and would be the first thing cut.

    The Ava-initiated note is kept — a reversed session's roles are inverted whatever is
    being read from it — but *ava_initiated_note* lets a caller supply its own. The
    default (``_AVA_INITIATED_NOTE``) is written for the REVISION pass: it closes "when
    judging your reply" and tells her the self-descriptions here are her own "not facts
    about them". Every reading pass inherited it, and for a pass whose job is to write
    down what was said that reads as an instruction to skip her own turns — with no
    positive counterpart saying where they should go instead. A pass whose reading of a
    reversed session differs should say so in its own words rather than inherit these.

    Returns "" when the session has no renderable turns.
    """
    ava_initiated = (session.get("initiated_by") or "").strip() == "ava"
    note = _AVA_INITIATED_NOTE if ava_initiated_note is None else ava_initiated_note.strip()
    prefix = f"{note}\n\n" if (ava_initiated and note) else ""
    closing = (closing or "").strip()
    dated = session_date_line(session)
    header = f"{dated}\n\n{_READING_HEADER}" if dated else _READING_HEADER
    reserved = len(prefix) + len(header) + len(closing) + 8
    block = format_context_block(
        session_transcript_turns(session), reserved, context_length,
        keep_first=ava_initiated)
    if not block:
        return ""
    parts = [f"{prefix}{header}", block]
    if closing:
        parts.append(closing)
    return "\n\n".join(parts)


# ── transcript-echo guard ─────────────────────────────────────────────────────

# A reading pass emitting a verbatim run this long from the transcript is not phrasing
# something in its own words — 12 words of exact agreement do not happen by accident.
_ECHO_MIN_RUN = 12
# ...and the run must be most of what was written, so a line that quotes a phrase and
# then says something about it survives. Deliberately conservative in the same direction
# as the clustering blob guard: a missed echo is one weak record among many, while a
# dropped genuine reading is evidence that is simply gone.
_ECHO_COVER = 0.6

_ECHO_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _echo_words(text: str) -> list[str]:
    return _ECHO_WORD_RE.findall((text or "").lower())


def build_echo_index(session: dict) -> list[str]:
    """Normalized, space-joined word forms of every turn in *session*.

    Fed to :func:`is_transcript_echo`. Built from the same answer-only turn list the
    reading content renders — CoT is excluded on purpose and NOT an oversight: Ava's
    in-the-moment thought about the person is exactly the kind of thing a legitimate
    impression restates, so matching against it would delete the pass's best output.
    """
    return [" " + " ".join(_echo_words(t.get("content") or "")) + " "
            for t in session_transcript_turns(session)
            if (t.get("content") or "").strip()]


def is_transcript_echo(text: str, index: list[str]) -> bool:
    """Is *text* substantially a verbatim copy of one turn in *index*?

    The reading passes are asked for something nobody in the conversation said outright,
    so a line lifted from the transcript is a failed generation rather than a weak one —
    and one that would then be folded into a portrait and recalled as a reading. Flags
    only a contiguous run of at least ``_ECHO_MIN_RUN`` words that also covers
    ``_ECHO_COVER`` of the line, which makes a short line unflaggable by construction.
    """
    words = _echo_words(text)
    if not words or not index:
        return False
    run = max(_ECHO_MIN_RUN, math.ceil(_ECHO_COVER * len(words)))
    if len(words) < run:
        return False
    grams = {" " + " ".join(words[i:i + run]) + " "
             for i in range(len(words) - run + 1)}
    return any(g in turn for g in grams for turn in index)


# ── clean IDEAL re-answer context ─────────────────────────────────────────────

def _user_content(content: str, speaker: str) -> str:
    return render_user_turn(speaker, content)


def build_ideal_messages(
    job: dict,
    session: dict,
    *,
    system_suffix: str = "",
    persona_context: str = "",
) -> list[dict]:
    """Build the exact pre-answer chat prefix used to generate a fresh IDEAL.

    This is the provenance boundary between reflection and dialogue generation. The
    judgement pass may inspect the old CoT/reply and post-reply feedback; the IDEAL pass
    must not. It receives only the stored chat system prompt, the answer-only preceding
    conversation, and the user message being answered. That is also the prefix persisted
    on the dialogue anchor and reconstructed by ``training.render.build_messages``.

    ``persona_context`` is Ava's own relevant self-knowledge, retrieved (persona-only,
    temporally cut to *before this chat* — see the runner) so the re-derived CoT can draw
    on who she is where it naturally fits, instead of the retired build-time prepend that
    forced a persona recitation to the front of every host CoT (which flooded the CoT at
    inference). It is appended to the system message here **and persisted verbatim into the
    anchor system prompt** (``dialogue_source.build_dialogue_anchor`` concatenates the same
    stored block), so ``render.build_messages`` reconstructs an identical system message and
    train/inference parity holds by construction. Empty for keep/branch targets, whose CoT
    is the original (already persona-bearing). Placed BEFORE ``system_suffix`` so a neutral
    delivery constraint still lands last.

    ``system_suffix`` is reserved for neutral delivery constraints (the automated
    language-drift retry, and the operator-supplied suffix from the Training review
    Regenerate box). It must never contain the old reply, reflection diagnosis, or any
    reference to revising/retrying an answer — the automated callers uphold this; an
    operator typing into the regen box does so at their own discretion.
    """
    ex = job.get("exchange") or {}
    system_prompt = str(session.get("system_prompt") or "").strip()
    persona_block = (persona_context or "").strip()
    if persona_block:
        system_prompt = f"{system_prompt}\n\n{persona_block}" if system_prompt else persona_block
    suffix = (system_suffix or "").strip()
    if suffix:
        system_prompt = f"{system_prompt}\n\n{suffix}" if system_prompt else suffix

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in job.get("context") or []:
        role = turn.get("role")
        content = str(turn.get("content") or "")
        if role == "user":
            messages.append({
                "role": "user",
                "content": _user_content(content, str(turn.get("speaker") or "")),
            })
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    messages.append({
        "role": "user",
        "content": _user_content(
            str(ex.get("user_prompt") or ""), str(job.get("speaker") or "")
        ),
    })
    return messages


# ── GPU-free self-test ────────────────────────────────────────────────────────

def _selftest() -> None:
    """Exercise the budgeted context walk. Run: ``python -m core.reflection_source``."""
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    # A reversed session: the opener is the OLDEST turn, and therefore the first thing the
    # newest-first walk drops — while being the turn the conversation exists for.
    session = {
        "initiated_by": "ava",
        "user": "artemy",
        "exchanges": [
            {"speaker": "(initiative)", "user_prompt": "impulse",
             "assistant_response": "OPENER — the message she sent unprompted."},
        ] + [
            {"speaker": "artemy", "user_prompt": f"reply {i} " + "x" * 400,
             "assistant_response": f"answer {i} " + "y" * 400}
            for i in range(12)
        ],
    }
    turns = session_transcript_turns(session)
    check("the opener survives turn extraction",
          any("OPENER" in t["content"] for t in turns), True)

    # The material's own date — the referent a relative reference inside the transcript
    # ("on Tuesday") needs, which the reflect lane's "now" cannot supply for a chat
    # reflected weeks later.
    dated = dict(session, timestamp="2026-06-21T23:44:03")
    check("a datable session is dated",
          session_date_line(dated), "This conversation took place on Sunday, June 21, 2026.")
    check("...and it reaches the reading content",
          "took place on Sunday, June 21, 2026" in build_session_reading_content(dated, 4000),
          True)
    check("an undatable session degrades silently", session_date_line({}), "")
    check("...as does an unparseable stamp",
          session_date_line({"timestamp": "not a date"}), "")

    tight = 900   # context_length small enough that the walk must drop groups
    plain = format_context_block(turns, 0, tight)
    pinned = format_context_block(turns, 0, tight, keep_first=True)
    check("without keep_first the opener is dropped", "OPENER" in plain, False)
    check("with keep_first it is kept", "OPENER" in pinned, True)
    check("...and the omission is still marked", _OMITTED_MARKER in pinned, True)
    check("...and the newest turn is still there", "answer 11" in pinned, True)
    check("the stage-direction impulse is not a speaker",
          "(initiative):" in pinned, False)

    # An opener that would eat the whole budget degrades to the plain walk rather than
    # returning one turn and a marker.
    huge = {"initiated_by": "ava", "user": "a", "exchanges": [
        {"speaker": "(initiative)", "user_prompt": "i", "assistant_response": "Z" * 5000},
        {"speaker": "a", "user_prompt": "q", "assistant_response": "the recent answer"},
    ]}
    degraded = format_context_block(session_transcript_turns(huge), 0, tight,
                                    keep_first=True)
    check("an oversized opener does not crowd out the rest",
          "the recent answer" in degraded, True)

    # An ordinary session is untouched: keep_first defaults off. (Compared as a bool —
    # these blocks are kilobytes, and a mismatch is not read by eyeballing two dumps.)
    check("default behaviour is byte-identical",
          format_context_block(turns, 0, tight, keep_first=False) == plain, True)

    # A single-group session has nothing to trade off.
    one = [{"role": "user", "content": "hi", "speaker": "a"},
           {"role": "assistant", "content": "hello"}]
    check("a single group is returned whole",
          format_context_block(one, 0, tight, keep_first=True),
          format_context_block(one, 0, tight))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
