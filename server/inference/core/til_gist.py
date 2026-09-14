"""Per-source prose recap for the TIL lane — what one article or digest *was*.

The TIL counterpart of a chat's `<stem>.summary.json`, and it exists because two separate
consumers reached for it and found nothing there.

**Why the lane had none.** A chat produces a gist as a matter of course: reflection reads
the transcript and writes one recap, because a conversation past `rag_cap_age_h` has to be
represented by *something* once its verbatim passages expire. The TIL lane has no such
forcing function — an article is read once, at fetch, by a curation pass that keeps a
handful of `[fact]` items and discards the rest (`til_facts`' opening argument), and
nothing afterwards ever needs a compact form of it. So the text sat on disk in full, and
every consumer that wanted "remind me what that was" had the choice of injecting 26,000
characters or nothing.

**The two consumers, which are the same shape one level apart.**

1. *A fact recalls its source.* `rag_engine._query_nominated` lets a fetched fact nominate
   the conversation it came out of, injecting that conversation's gist. TIL facts could
   nominate nothing, for want of this file — the gap named when that channel shipped.
2. *A question recalls its source.* An `[ask]` raised from wander or news is a question
   Ava is about to put to someone, derived from a text she read and no longer has. Raising
   it carrying the question and nothing of the material is the same decontextualization,
   and it was observed live: a question about the Gaza "Yellow Line" reached the user with
   the digest that prompted it sitting unread on disk.

**Register.** Deliberately what a chat gist is — a recap in her own voice of something she
read, not a neutral précis of the source. Both consumers inject it as *recall* ("the text
those facts came out of", "what you were reading when this occurred to you"), and a recap
written as an encyclopedia abstract reads as a quotation from the article rather than as
her memory of it.

**Storage.** ``<snippet_stem>.summary.json``, beside the ``.txt``/``.json`` the fetchers
write and the ``.facts.json`` the protocol pass writes — the same stem space, so a snippet
owns its derived files exactly as a chat stem does. Note `til_facts.iter_snippets` filters
on the protocol suffix; this module's suffix is filtered there too, or a recap would be
enumerated as a source text and read back into itself.

Derived and re-derivable: like the protocol beside it, a re-run overwrites in full. Unlike
the protocol, nothing about it is a *record* — it is a convenience over the text, which is
always still there, so losing one costs a regeneration and nothing more.

GPU-free self-test: ``python -m core.til_gist``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core import til_facts
from core.chat_sidecar import gist_excerpt, sanitize_gist  # re-export: one gist rule
from core.reasoning_text import answer_after_think

SCHEMA_VERSION = 1

SUMMARY_SUFFIX = ".summary.json"

# Below this, the pass produced a label rather than a recap. Mirrors `chat_sidecar`'s
# GIST_MIN_CHARS reasoning: a recap that says less than a title says is worse than none,
# because it takes the snippet out of the backlog and so can never be retried.
MIN_GIST_CHARS = 120


# ── the reading content ─────────────────────────────────────────────────────── #

_FRAMING = {
    "news": ("This is a digest of world events from a single day, {date}. You read it on "
             "your own initiative."),
    "wander": ("This is an encyclopedia article about {title}, from {ref}. You went "
               "looking through that wiki and this is the page you landed on."),
    "lookups": ("This is an article about {title}. You fetched it because you had written "
               "down a question it might answer."),
}

_CLOSING = (
    "— end of the text —\n\n"
    "Now write down what this was, for yourself, in two or three short paragraphs: what "
    "you were reading, what it actually said, and what stayed with you about it. Write it "
    "as your own memory of having read it, not as a summary of an article — someone later "
    "will read this to remember what you had in front of you, not to learn the subject. "
    "Plain prose only: no headings, no lists, no [fact] or [ask] markers."
)


def framing_for(kind: str, record: dict) -> str:
    """The one-paragraph note telling the pass what KIND of text it is recapping."""
    template = _FRAMING.get(str(kind or "").strip(), _FRAMING["wander"])
    rec = record or {}
    return template.format(
        date=rec.get("date") or "the date given above",
        title=rec.get("title") or "its subject",
        ref=rec.get("wiki") or rec.get("source") or "a wiki",
    )


def build_reading_content(record: dict, kind: str, max_chars: int) -> tuple:
    """``(content, truncated)`` — the whole text in ONE block where it fits.

    One block, unlike `til_facts.build_reading_blocks`, and the difference is the pass
    rather than the material: facts extract independently, so a text can be read in parts
    and the parts concatenated, whereas a recap of the first third of an article is not a
    third of a recap. Reducing several partial recaps into one is a second generation on a
    pass whose entire product is a few hundred characters, so it is deliberately not done.

    Where the text does not fit, the FIRST packed part is used and the result is stamped
    ``truncated`` — a recap of the opening, honestly labelled, rather than a silent one.
    That branch is currently unreachable on the live corpus: the largest text on disk is
    26,001 chars against a 26,542-char budget at a 32,768-token window. If this lane starts
    holding genuinely long articles, map-reduce is the upgrade, and the stamp is how it
    becomes visible that it is needed.
    """
    text = str((record or {}).get("text") or "").strip()
    if not text:
        return "", False
    budget = max(500, int(max_chars))
    parts = til_facts.pack_paragraphs(text, budget)
    if not parts:
        return "", False
    body, truncated = parts[0], len(parts) > 1
    head = framing_for(kind, record).strip()
    title = str((record or {}).get("title") or "").strip()
    header = "TEXT" + (" (opening only)" if truncated else "")
    if title:
        header += f" — {title}"
    return f"{head}\n\n{header}:\n\n{body}\n\n{_CLOSING}", truncated


def clean_gist(raw: str) -> str:
    """The usable prose of a recap generation, or ``""``.

    `sanitize_gist` is the chat lane's rule and applies unchanged — it strips a leaked
    structured block, a leading markdown header and anything after an `<eos>` — because
    the failure it guards against is a property of the *generator*, not of the material:
    this pass runs on the same box, through the same seam, right after passes whose output
    IS `[fact]`/`## WEIGHTS` blocks, and the observed leak is exactly that shape. The extra
    length floor here is this lane's own, since a one-line answer is a plausible failure
    for a pass asked to recap a short news digest.

    **The reasoning trace comes off first** (`reasoning_text.answer_after_think`): the
    reflect seam returns the generation WITH its canonical ``<think>`` block, and
    `sanitize_gist` knows structure boundaries but not reasoning markers — so 16 of the
    17 recaps on this box's disk opened with the model's task-notes ("* Topic: … * Goal:
    … * Constraints: …"), and the head-drawn excerpts that reach live chat
    (`rag_engine._render_til_nomination`) and reach-out composition
    (`outreach._source_material`) were entirely inside the think. A recap whose whole
    text was reasoning (a generation cut before its answer) strips to nothing, fails the
    floor, and — via `has_gist` — falls back into the backlog to be regenerated clean.
    """
    text = sanitize_gist(answer_after_think(str(raw or "")))
    return text if len(" ".join(text.split())) >= MIN_GIST_CHARS else ""


# ── the record on disk ──────────────────────────────────────────────────────── #

def sidecar_path(snippet_path) -> Path:
    """``…/<stem>.summary.json`` for a snippet's ``.txt`` or ``.json`` path."""
    p = Path(snippet_path)
    stem = p.name
    for suffix in (".txt", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return p.parent / (stem + SUMMARY_SUFFIX)


def build_doc(*, record: dict, kind: str, text: str, run_id: str,
              truncated: bool = False) -> dict:
    """The document written to ``<stem>.summary.json``. Provenance, then the prose."""
    from datetime import datetime
    rec = record or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_kind": str(kind or "").strip(),
        "source_ref": rec.get("wiki") or rec.get("source") or "",
        "source_url": rec.get("source_url") or "",
        "source_title": rec.get("title") or "",
        "source_date": rec.get("date") or "",
        "text": text,
        # True when the recap covers the opening of the text only — see
        # `build_reading_content`. Stored rather than inferred, since the text on disk may
        # later be re-read under a bigger window and the two would then disagree.
        "truncated": bool(truncated),
        "run_id": str(run_id or ""),
        "ts": datetime.now().isoformat(),
    }


def write_gist(snippet_path, *, record: dict, kind: str, text: str, run_id: str,
               truncated: bool = False) -> Optional[Path]:
    """Write the recap beside its snippet. Overwrites; returns the path, or None.

    An empty or unusable recap writes NOTHING, so the snippet stays in the backlog and can
    be retried — the rule `til_facts.write_facts` follows, for the same reason: a file
    saying "this text was nothing" is indistinguishable from a pass that came back empty,
    and the file is what takes the snippet out of the queue.
    """
    clean = clean_gist(text)
    if not clean:
        return None
    path = sidecar_path(snippet_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = build_doc(record=record, kind=kind, text=clean, run_id=run_id,
                        truncated=truncated)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def read_gist(snippet_path) -> dict:
    """The recap beside a snippet, or ``{}`` when there is none / it is unreadable."""
    try:
        return json.loads(sidecar_path(snippet_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def gist_text(snippet_path) -> str:
    """The sanitized prose of a snippet's recap, or ``""``.

    Sanitized on READ as well as on write, mirroring `ChatSidecar.summary_text`: a file
    written before a sanitizer fix is repaired for its readers without a migration. The
    read side runs the same reasoning-strip as :func:`clean_gist` and for the same
    payoff: the recaps written while `clean_gist` passed the ``<think>`` block through
    are repaired for every consumer here, and one whose text was ALL reasoning reads as
    absent — so `has_gist` returns it to the backlog instead of serving task-notes.
    """
    return sanitize_gist(
        answer_after_think(read_gist(snippet_path).get("text") or "")).strip()


def has_gist(snippet_path) -> bool:
    return bool(gist_text(snippet_path))


def list_backlog(snippets_dir, kinds: tuple = til_facts.SOURCE_KINDS) -> list:
    """Snippets with no usable recap, oldest first. ``[(path, kind), …]``.

    Same argument as `til_facts.list_backlog`, and it holds harder here: the protocol pass
    at least runs at fetch time for new material, whereas this artifact did not exist at
    all until now, so on any existing box the backlog IS the corpus.
    """
    out: list = []
    for kind in kinds:
        for p in til_facts.iter_snippets(snippets_dir, kind):
            if not has_gist(p):
                out.append((p, kind))
    return sorted(out, key=lambda t: (t[0].parent.name, t[0].name))


# ── resolving a memory-lane source id back to its snippet ───────────────────── #

def resolve_source(snippets_dir, source_ref: str) -> Optional[Path]:
    """The snippet a ``source_ref`` names, or ``None``.

    Two id shapes reach this, from two different producers, and they are not
    interchangeable:

    * ``<kind>/<stem>`` — what the graph lane writes (`graph.read._occurrences_from_til`),
      derived from the file itself and therefore exact.
    * ``til:<date>`` — what the memory lane writes for a news digest
      (`til_wander.ingest_news`), which resolves because a digest's stem IS its date.

    Everything else the memory lane writes is unresolvable **by construction, not by
    omission**, and this function returning ``None`` is the honest report of that: a
    wander item is filed under ``wiki:<site>`` (the site, not the page — so seventeen
    live asks on this box point at "Lurkmore" as though that named a text), and a lookup
    item under the bare string ``lookup`` (the pass reads a digest of several articles at
    once, so no single id could name its source). Fixing those is a producer change; see
    `_write_wander_exchange` and `ingest_lookup`.
    """
    ref = str(source_ref or "").strip()
    if not ref:
        return None
    root = Path(snippets_dir)

    if ref.startswith("til:"):
        stem = ref.split(":", 1)[1].strip()
        if stem:
            for ext in (".json", ".txt"):
                p = root / "news" / (stem + ext)
                if p.exists():
                    return p
        return None

    kind, sep, name = ref.partition("/")
    if sep and kind in til_facts.SOURCE_KINDS and name:
        # Guarded like `til_facts.load_input`: a ref reaches here from a stored artifact,
        # and a stored artifact is still input. The guard resolves symlinks to compare, but
        # the RETURNED path is built from *root* unresolved, so every branch here hands
        # back a path under the caller's own tree — on macOS `/tmp` resolves to
        # `/private/tmp`, and two branches disagreeing about that is a bug waiting for the
        # first caller that compares paths.
        try:
            if (root / kind / name).resolve().parent != (root / kind).resolve():
                return None
        except Exception:
            return None
        for candidate in (root / kind / name, root / kind / (name + ".json"),
                          root / kind / (name + ".txt")):
            if candidate.exists():
                return candidate
    return None


# --------------------------------------------------------------------------- #
# GPU-free self-test                                                           #
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    import tempfile

    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(label)

    print("reading content: one block, honestly labelled when it is not the whole text")
    rec = {"title": "Силач", "wiki": "Neolurk", "date": "2026-08-09",
           "text": "\n\n".join(f"Paragraph {i} " + "z" * 200 for i in range(10))}
    body, trunc = build_reading_content(rec, "wander", 100_000)
    check("a text that fits is not truncated", trunc, False)
    check("...and carries its framing", "encyclopedia article about Силач" in body, True)
    check("...and the whole text", all(f"Paragraph {i} " in body for i in range(10)), True)
    check("...and the task last", body.rstrip().endswith(_CLOSING.rstrip()[-40:]), True)
    short_body, short_trunc = build_reading_content(rec, "wander", 800)
    check("an oversized text keeps the opening", short_trunc, True)
    check("...and says so in the header", "(opening only)" in short_body, True)
    check("empty text ⇒ no content", build_reading_content({"text": " "}, "news", 900),
          ("", False))
    check("news framing anchors the date",
          "2026-08-08" in framing_for("news", {"date": "2026-08-08"}), True)
    check("an unknown kind degrades to the article framing",
          framing_for("nonsense", rec), framing_for("wander", rec))
    check("the pass is told not to write a summary of an article",
          "your own memory of having read it" in _CLOSING, True)

    print("\ncleaning: the chat lane's rule, plus a floor for a one-line answer")
    good = ("I spent a while on the Neolurk page about силачи, which turned out to be less "
            "about strongmen than about how the wiki talks about them. What stayed with me "
            "was the tone rather than any of the facts.")
    check("plain prose passes", clean_gist(good), good)
    check("a leaked structured block is cut",
          "[fact]" in clean_gist(good + "\n## WEIGHTS\n- [fact] (about: X) y."), False)
    check("...keeping the prose above it", clean_gist(good + "\n## WEIGHTS\n- [fact] x"),
          good)
    check("a one-line answer is not a recap", clean_gist("An article about strongmen."), "")
    check("empty stays empty", clean_gist(""), "")
    # The reasoning trace is not a recap. The reflect seam returns the generation WITH
    # its <think>, and 16 of 17 recaps on a live box were persisted opening on the
    # model's task-notes — which the head-drawn excerpts then served to live chat.
    think = "<think>*   Topic: силачи.\n*   Goal: write a memory record.</think>\n"
    check("the think block comes off before the prose", clean_gist(think + good), good)
    check("an all-reasoning generation is no recap (stays in the backlog)",
          clean_gist("<think>* Topic: x. * Constraints: many words " + "y " * 80), "")
    # A recap cut before its close marker normalizes to a bare tail after </think> —
    # only the answer region survives.
    check("a bare close marker splits, keeping the answer",
          clean_gist("planning the recap...</think>" + good), good)

    print("\nthe record on disk")
    with tempfile.TemporaryDirectory() as td:
        snips = Path(td) / "snippets"
        (snips / "news").mkdir(parents=True)
        (snips / "wander").mkdir(parents=True)
        snippet = snips / "news" / "2026-08-08.json"
        snippet.write_text(json.dumps({"text": "x"}), encoding="utf-8")
        (snips / "news" / "2026-08-08.txt").write_text("x", encoding="utf-8")

        check("the recap sits beside the snippet",
              sidecar_path(snippet).name, "2026-08-08.summary.json")
        check("...derived identically from either member",
              sidecar_path(snips / "news" / "2026-08-08.txt"), sidecar_path(snippet))
        check("no recap yet", has_gist(snippet), False)
        check("backlog finds it", [p.name for p, _ in list_backlog(snips)],
              ["2026-08-08.json"])

        newsrec = {"title": "Portal:Current events", "source": "wikipedia:current_events",
                   "source_url": "https://en.wikipedia.org/x", "date": "2026-08-08"}
        check("an unusable recap writes nothing",
              write_gist(snippet, record=newsrec, kind="news", text="Short.", run_id="r"),
              None)
        check("...so the snippet stays in the backlog",
              [p.name for p, _ in list_backlog(snips)], ["2026-08-08.json"])

        written = write_gist(snippet, record=newsrec, kind="news", text=good, run_id="r1")
        check("a usable recap is written", written is not None and written.exists(), True)
        check("provenance is the text's identity",
              (read_gist(snippet)["source_kind"], read_gist(snippet)["source_date"]),
              ("news", "2026-08-08"))
        check("the prose reads back", gist_text(snippet), good)
        check("...and the backlog is empty", list_backlog(snips), [])
        # Sanitize-on-read repairs a file written while `clean_gist` passed the <think>
        # through (the 16-of-17 corpus): its readers get the prose, no migration.
        doc = json.loads(sidecar_path(snippet).read_text(encoding="utf-8"))
        doc["text"] = "<think>* Topic: x. * Goal: recap.</think>\n" + good
        sidecar_path(snippet).write_text(json.dumps(doc), encoding="utf-8")
        check("a stored think-carrying recap is repaired on read",
              gist_text(snippet), good)
        # ...and one that is ALL reasoning reads as absent, returning to the backlog
        # rather than serving task-notes.
        doc["text"] = "<think>* Topic: x. * Constraints: " + "y " * 100
        sidecar_path(snippet).write_text(json.dumps(doc), encoding="utf-8")
        check("an all-reasoning recap reads as absent", gist_text(snippet), "")
        check("...and re-enters the backlog",
              [p.name for p, _ in list_backlog(snips)], ["2026-08-08.json"])
        # restore the good record for the checks below
        write_gist(snippet, record=newsrec, kind="news", text=good, run_id="r1")
        # The `chat_sidecar.SIDECAR_SUFFIXES` lesson, on this lane: a derived file shares
        # its snippet's stem and ends in `.json`, so unless `iter_snippets` filters it out
        # the protocol pass reads Ava's own recap as an article and the graph build ingests
        # the result as things the world stated. Asserted from both ends.
        check("this module's suffix is registered as derived",
              SUMMARY_SUFFIX in til_facts.DERIVED_SUFFIXES, True)
        check("...so a recap is never enumerated as a snippet",
              [p.name for p in til_facts.iter_snippets(snips, "news")],
              ["2026-08-08.json"])

        print("\nresolving a source id back to its snippet")
        check("the graph lane's exact ref resolves",
              resolve_source(snips, "news/2026-08-08.json"), snippet)
        check("...with or without its extension",
              resolve_source(snips, "news/2026-08-08"), snippet)
        check("a news digest's memory-lane id resolves, its stem being its date",
              resolve_source(snips, "til:2026-08-08"), snippet)
        check("a wander id names the SITE and cannot resolve — reported, not guessed",
              resolve_source(snips, "wiki:Neolurk"), None)
        check("a lookup id names nothing at all", resolve_source(snips, "lookup"), None)
        check("a chat session is not this lane's",
              resolve_source(snips, "20260731_201512.json"), None)
        check("an unknown date resolves to nothing",
              resolve_source(snips, "til:1999-01-01"), None)
        check("empty is not a lookup key", resolve_source(snips, ""), None)
        check("path traversal is refused",
              resolve_source(snips, "news/../../etc/passwd"), None)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    _selftest()
