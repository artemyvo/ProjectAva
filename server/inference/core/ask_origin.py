"""Answer-side ask origin — the material behind an Ava-initiated session's question.

When Ava opens a conversation herself (outreach / synthesis), the question arrives in
that session stripped of everything that produced it: the ask's ``source_session``, the
conversation or article it was distilled from, the recap that was in front of her when
she composed the opener. The *decision* side got its conditioning on 2026-08-18 (the
facts-tree fetch + ``{origin_note}`` source material in `outreach`); this is the ANSWER
side's counterpart. When the user adopts the session and replies, the reply turn used
to condition on the opener alone — reconnecting the answer to what made her ask rode
entirely on cosine retrieval (keyword-dependent, weak across languages, hard-zero once
the source chat is past ``rag_cap_age_h``) or on the facts fetch happening to nominate
the source chat. The structural link — ``initiated_ask``, stamped on the session for
the tight [ask]-loop close — was read by reflection only; nothing read it at chat time.

So `generation.handle_generate` injects ONE standing block on the turns of an
``initiated_by:"ava"`` session: the question plus its origin material, **gist or
nothing** (the `rag_engine._query_nominated` rule) — a chat-origin ask carries the
source conversation's stored gist, a self-directed ask the TIL recap of what she was
reading (`til_gist.resolve_source`; the ``wiki:<site>`` / bare-``lookup`` refs are
unresolvable by construction and yield nothing, exactly as on the decision side). No
origin material ⇒ NO block at all: the question itself is already in the transcript as
her own opener, and a block that only restates it is noise.

Deliberately NOT carried: her captured wander *reaction* (`wander_sft.reaction_for`),
which the decision pass does inject. There it sits before she composes ONE opener; here
it would sit before every reply of a whole conversation — her own old prose, in her own
register, with only the previous-reply anti-copy window between it and the replies. The
decision side already excerpts it harder than the recap for that reason; this side
takes the argument to its conclusion and drops it.

Resolving the ask's origin is two-tier: the ``source_session`` stamped into the
session's ``initiated_ask`` (durable — it survives the ask's later resolution/eviction),
falling back to a live-fold lookup by ask key for sessions written before the stamp
existed. The fallback goes dark once the ask is evicted; the stamp is the fix, the
fallback the bridge for the existing corpus.

Pure and GPU-free — callers hand in every directory. Self-test:
``python -m core.ask_origin``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.chat_sidecar import ChatSidecar, gist_excerpt
from core.reflection_memory import ReflectionMemory, is_self_directed_origin
from core import til_gist

# Same excerpt budget as the decision side's source material
# (`outreach._SOURCE_MATERIAL_CHARS`): the two are the same recall one stage apart, and
# a bigger allowance on the answer side would mean the *reply* turns carry more of the
# source than the turn that decided to raise it.
ORIGIN_CHARS = 1400

_BLOCK_FILE = "ask_origin_prompt.txt"

# {question} / {origin} slots. The closing discipline is this block's whole reason to be
# phrased at all: it lands on every reply turn of the session, so without it the origin
# material reads as something to work into the answer rather than as the context the
# answer is read against.
_DEFAULT_BLOCK = (
    "You opened this conversation yourself, to raise a question you had been "
    "carrying:\n\n"
    "{question}\n\n"
    "{origin}\n\n"
    "This is background for reading what they say, not material for the reply: do not "
    "recite it back to them, and do not present it as something you just looked up. "
    "If their answer settles the question, let it be settled."
)


def load_block_template(prompts_dir) -> str:
    """The block template (``ask_origin_prompt.txt``), default-written on first miss.

    Same contract as `generation._load_persona_undecided` / the facts-block wrappers:
    the default is materialized to disk so an operator can retune the framing without a
    code change, and a hand-emptied file falls back to the default rather than
    injecting a bare gist with no discipline around it.
    """
    path = Path(prompts_dir) / _BLOCK_FILE
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_DEFAULT_BLOCK + "\n", encoding="utf-8")
    except Exception:
        pass
    return _DEFAULT_BLOCK


def resolve_source_session(initiated_ask: dict, memory_dir) -> str:
    """The ask's origin ref — the stamp when present, else a live-fold lookup by key.

    The stamp (``initiated_ask.source_session``, written since this module landed) is
    authoritative and durable. The fallback serves the sessions written before it: find
    the still-open ask by ``key`` in the live fold and read its ``source_session``. An
    ask that has since been resolved/evicted is gone from the fold, so the fallback
    honestly returns ``""`` — the block simply doesn't render for those sessions.
    """
    stamp = str((initiated_ask or {}).get("source_session") or "").strip()
    if stamp:
        return stamp
    key = str((initiated_ask or {}).get("key") or "").strip()
    if not key:
        return ""
    try:
        for item in ReflectionMemory(Path(memory_dir)).open_questions():
            if item.get("key") == key:
                return str(item.get("source_session") or "").strip()
    except Exception:
        pass
    return ""


def _session_date(stem_or_name: str) -> str:
    """ISO date off a chat filename stem (``20260729_021044[.json]``), or ``""``.

    ISO, not prose — the corpus is mixed-language and the block is injected verbatim
    (the `rag_engine._recollection_label` argument).
    """
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_", str(stem_or_name or ""))
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def origin_text(source_session: str, *, chats_dir, archive_chats_dir,
                snippets_dir) -> str:
    """The origin material with its lane lead-in, or ``""`` when nothing resolves.

    Two lanes, mirroring where an ask can come from:

    * a **chat** ref (a transcript filename) — the source conversation's stored
      consolidation gist, read through `ChatSidecar.summary_text` (hot first, archive
      fallback), dated off the stem;
    * a **self-directed** ref (``til:<date>`` / ``<kind>/<stem>``) — the article's or
      digest's TIL recap. ``wiki:<site>`` and the bare ``lookup`` return ``None`` from
      the resolver and therefore ``""`` here — the honest consequence of the producers'
      id quality, same as on the decision side (see `til_gist.resolve_source`).
    """
    ref = str(source_session or "").strip()
    if not ref:
        return ""
    try:
        if is_self_directed_origin(ref):
            path = til_gist.resolve_source(Path(snippets_dir), ref)
            if path is None:
                return ""
            gist = gist_excerpt(til_gist.gist_text(path), ORIGIN_CHARS)
            if not gist:
                return ""
            return ("It first stirred while you were reading on your own. What you "
                    "were reading, as you remember it:\n\n" + gist)
        sidecar = ChatSidecar(Path(chats_dir),
                              fallback_chats_dir=Path(archive_chats_dir))
        gist = gist_excerpt(sidecar.summary_text(ref), ORIGIN_CHARS)
        if not gist:
            return ""
        when = _session_date(ref)
        dated = f" from {when}" if when else ""
        return (f"It grew out of a conversation{dated}, which you remember "
                f"as:\n\n{gist}")
    except Exception:
        return ""


def origin_block(initiated_ask: dict, *, memory_dir, chats_dir, archive_chats_dir,
                 snippets_dir, prompts_dir) -> str:
    """The injectable ASK ORIGIN block for one session, or ``""``.

    Gist-or-nothing: with no resolvable origin material the whole block is withheld —
    the question is already in the transcript as her own opener, so a block that only
    restated it would spend a standing slot on noise. A missing question (a malformed
    stamp) likewise yields nothing.
    """
    question = str((initiated_ask or {}).get("content") or "").strip()
    if not question:
        return ""
    ref = resolve_source_session(initiated_ask or {}, memory_dir)
    origin = origin_text(ref, chats_dir=chats_dir,
                         archive_chats_dir=archive_chats_dir,
                         snippets_dir=snippets_dir)
    if not origin:
        return ""
    return (load_block_template(prompts_dir)
            .replace("{question}", question)
            .replace("{origin}", origin))


# ── self-test ────────────────────────────────────────────────────────────────── #

def _selftest() -> None:  # pragma: no cover - exercised via `python -m`
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chats = root / "chats"
        archive = root / "archive"
        snippets = root / "snippets"
        memory = root / "memory"
        prompts = root / "prompts"
        for d in (chats, archive, snippets / "news", memory, prompts):
            d.mkdir(parents=True)

        # Chat-origin: gist in the hot dir, dated off the stem.
        stem = "20260729_021044.json"
        (chats / stem).write_text("{}", encoding="utf-8")
        (chats / "20260729_021044.summary.json").write_text(
            json.dumps({"text": "We argued about crocodiles, and whether a metaphor "
                                "can be worn out by the person who coined it."}),
            encoding="utf-8")
        ask = {"key": "k1", "content": "Why crocodiles?", "source_session": stem}
        block = origin_block(ask, memory_dir=memory, chats_dir=chats,
                             archive_chats_dir=archive, snippets_dir=snippets,
                             prompts_dir=prompts)
        assert "Why crocodiles?" in block and "worn out" in block, block
        assert "2026-07-29" in block, "chat origin should be dated off the stem"
        assert (prompts / _BLOCK_FILE).exists(), "template default-written on first miss"

        # Archive fallback: gist beside an archived transcript only.
        stem2 = "20260101_000000.json"
        (archive / stem2).write_text("{}", encoding="utf-8")
        (archive / "20260101_000000.summary.json").write_text(
            json.dumps({"text": "An archived talk about winter plans and the odd "
                                "comfort of postponing them together."}),
            encoding="utf-8")
        t = origin_text(stem2, chats_dir=chats, archive_chats_dir=archive,
                        snippets_dir=snippets)
        assert "An archived talk" in t, t

        # Self-directed: a news digest resolves via til:<date>; wiki:<site> never does.
        (snippets / "news" / "2026-08-01.txt").write_text("digest", encoding="utf-8")
        (snippets / "news" / "2026-08-01.summary.json").write_text(
            json.dumps({"text": "A day of drone strikes and one odd treaty nobody "
                                "expected to survive the week, which I keep thinking "
                                "about."}),
            encoding="utf-8")
        t = origin_text("til:2026-08-01", chats_dir=chats, archive_chats_dir=archive,
                        snippets_dir=snippets)
        assert "odd treaty" in t and "reading on your own" in t, t
        assert origin_text("wiki:lurkmore", chats_dir=chats, archive_chats_dir=archive,
                           snippets_dir=snippets) == ""

        # Gist-or-nothing: a chat with no stored gist withholds the whole block.
        stem3 = "20260810_120000.json"
        (chats / stem3).write_text("{}", encoding="utf-8")
        assert origin_block({"key": "k", "content": "Q?", "source_session": stem3},
                            memory_dir=memory, chats_dir=chats,
                            archive_chats_dir=archive, snippets_dir=snippets,
                            prompts_dir=prompts) == ""

        # Fallback lookup by key for a pre-stamp session; eviction darkens it.
        rag_log = memory / "rag_memory.jsonl"
        rag_log.write_text(json.dumps({
            "op": "insert", "kind": "ask", "key": "k9", "content": "Why crocodiles?",
            "ask_kind": "user", "source_session": stem,
        }) + "\n", encoding="utf-8")
        assert resolve_source_session({"key": "k9", "content": "Why crocodiles?"},
                                      memory) == stem
        with rag_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"op": "evict", "key": "k9"}) + "\n")
        assert resolve_source_session({"key": "k9"}, memory) == ""
        # The stamp survives what the fallback does not.
        assert resolve_source_session(
            {"key": "k9", "source_session": stem}, memory) == stem

        # A hand-emptied template file falls back to the built-in default.
        (prompts / _BLOCK_FILE).write_text("", encoding="utf-8")
        assert load_block_template(prompts) == _DEFAULT_BLOCK

    print("core.ask_origin self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
