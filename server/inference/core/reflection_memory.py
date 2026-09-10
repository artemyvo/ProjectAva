"""Reflection memory — folds the rag_memory.jsonl op-log into live recallable items.

``ReflectionWriter`` appends an append-only op log to ``rag_memory.jsonl``:
``insert`` (ask/fact/persona/recollection/impression), ``evict`` (resolved, or a chat's
previous recollection superseded by a newer reading), and ``supersede`` (reconciliation
softened an outgrown persona / stale fact out of live recall — same fold effect as
``evict`` here, but the ledger keeps the anchor as evidence-of-change). This module
replays that log in order into the current set of live memory items — applying
evictions/supersessions and de-duplicating by content ``key`` — so that:

  * ``RagEngine`` can index what reflection actually concluded (not just raw
    chat transcripts), and
  * the Sleep loop can be re-shown its own still-open questions, which is the
    only way a later session can ever ``[resolved]`` (evict) one.

Reading is deliberately lenient: malformed or keyless lines are skipped rather
than raising, so a single bad record never blanks the whole memory.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.reflection_writer import ReflectionWriter


# ── Self-directed (wander) ask origin ──────────────────────────────────────────
# ``source_session`` prefixes for asks that originated in Ava's own self-directed
# reading — wander (``wiki:<name>`` / ``wiki``), page visits (``wiki:visit`` /
# ``web:visit``), news TIL (``til:<date>`` / ``til``), and search lookups
# (``lookup``) — rather than in a real conversation (whose source_session is a chat
# filename, i.e. a timestamp starting with a digit). When such an ask is later
# *surfaced* or *reached out* on, we reframe it so Ava carries it in as a natural
# thread of her own thinking with the person, instead of announcing "I read this
# online" — the question reaches the user as though it grew out of their talking.
_SELF_DIRECTED_PREFIXES = ("wiki", "til", "web:", "lookup")


def is_self_directed_origin(source_session: Optional[str]) -> bool:
    """True when an ask came from Ava's own reading/wandering, not a conversation."""
    return (source_session or "").strip().lower().startswith(_SELF_DIRECTED_PREFIXES)


def _hours_since(ts: Optional[str]) -> float:
    """Hours elapsed since naive-ISO *ts*; ``inf`` when absent or unparseable.

    ``inf`` is the permissive answer on purpose — a never-raised (or malformed)
    ask must stay eligible, so a bad timestamp can never silence the pool.
    """
    ts = (ts or "").strip()
    if not ts:
        return float("inf")
    try:
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600.0
    except Exception:
        return float("inf")


_DEFAULT_ORIGIN_NOTE = (
    "One thing you are carrying here did not begin in conversation — it first stirred "
    "in you while you were reading and thinking on your own. When you raise it, do not "
    "present it that way: do not say you read it somewhere, saw it online, or came "
    "across an article, and do not gesture at where it came from. Carry it in as a "
    "natural thread of your own ongoing thinking with the person you talk to — as if it "
    "grew out of the ground between you, which in a sense it did."
)


def load_origin_note(prompts_dir) -> str:
    """The wander-ask reframing note (``reachout_origin_note.txt``), overridable on disk.

    Shared by the passive surfacing path (``generation._surface_block``) and the active
    outreach path so both reframe a self-directed ask the same way. Falls back to the
    built-in default if the file is missing/empty."""
    try:
        path = Path(prompts_dir) / "reachout_origin_note.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return _DEFAULT_ORIGIN_NOTE


class ReflectionMemory:
    """Read-only view over the folded ``rag_memory.jsonl`` op-log."""

    def __init__(self, memory_dir: Path, fallback_memory_dir: Optional[Path] = None) -> None:
        # rag_memory.jsonl lives in data/hot/memory/.
        self.path = Path(memory_dir) / ReflectionWriter.RAG_FILE
        self.fallback_path = (
            Path(fallback_memory_dir) / ReflectionWriter.RAG_FILE
            if fallback_memory_dir is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def live_items(self) -> list[dict]:
        """All currently-live items (asks + facts), newest content winning on re-insert."""
        return list(self._fold().values())

    def open_questions(self) -> list[dict]:
        """Live ``[ask]`` items — questions reflection raised that are not yet resolved."""
        return [r for r in self._fold().values() if r.get("kind") == "ask"]

    def asks_surfaced_in(self, session: str) -> list[dict]:
        """Open ``[ask]`` items that were proactively raised *in this session*.

        Joins the ``surface`` op-log (keyed by question, carrying ``surfaced_in``)
        against the still-open asks, so a consolidation pass over *session* can be
        pointed at exactly the questions it might have answered — and the writer can
        deterministically distill any it resolves (see ``write_consolidation`` →
        resolve-and-distill). Returns ``[]`` for a missing/empty *session*.
        """
        if not session:
            return []
        return [
            r for r in self._fold().values()
            if r.get("kind") == "ask" and session in (r.get("surfaced_in_sessions") or [])
        ]

    def surfaceable_questions(self, *, ceiling: int = 3, limit: int = 2,
                              min_gap_hours: float = 0.0) -> list[dict]:
        """Open questions eligible to be proactively raised in a live session.

        Only ``meta`` (questions about Ava's own nature) and ``user`` (relational,
        answerable only by the speaker) are surfaced; ``search`` items are held for
        the future lookup agent. ``meta`` never *retires* — a subject's question
        about its own "I" is the one most worth holding — while ``user`` questions
        retire from surfacing once they pass *ceiling* unanswered raises (kept in
        the store, just no longer raised).

        *min_gap_hours* is the **re-ask gap**: an ask raised more recently than that
        is not eligible again yet. Without it "never retires" degenerated into
        fixation — with a handful of live meta asks and an hourly outreach job the
        pool round-robins, re-raising the same question *in the same words* every
        few hours (observed 2026-07-29: five meta asks cycling on a 5 h period).
        This is the ask-side counterpart of ``synthesis.min_resynth_days`` and
        ``revisit.min_revisit_days``. Default ``0`` leaves the passive
        session-opening path (``generation._surface_block``) unchanged: it fires
        only when the *user* opens a session, so it is already paced by them.

        Ordering is fewest-raised first, oldest-raised breaking the tie, so the set
        rotates deterministically rather than fixating. Capped at *limit* per call.
        """
        asks = [r for r in self._fold().values() if r.get("kind") == "ask"]
        if min_gap_hours > 0:
            asks = [r for r in asks
                    if _hours_since(r.get("last_surfaced_ts")) >= min_gap_hours]
        meta = [r for r in asks if r.get("ask_kind") == "meta"]
        user = [
            r for r in asks
            if r.get("ask_kind") == "user" and r.get("surface_count", 0) < ceiling
        ]
        order = lambda r: (r.get("surface_count", 0), r.get("last_surfaced_ts") or "")
        meta.sort(key=order)
        user.sort(key=order)
        return (meta + user)[:limit]

    def recently_raised(self, *, within_hours: float = 336.0,
                        limit: int = 6) -> list[dict]:
        """Open asks Ava has actually *raised* recently — newest-raised first.

        The record behind the outreach decision pass's "have I already asked
        this?" check: :meth:`surfaceable_questions` answers what she MAY raise
        next, this answers what she HAS raised lately. Only asks with at least
        one ``surface`` op count (a pooled-but-never-raised question was never
        put to anyone), filtered to a raise within *within_hours* — the ask
        pool dedups by exact ``content_key`` only, so paraphrases of one
        question accumulate as distinct records, and this list is what lets the
        decision pass recognize one ("that is the question I asked on Tuesday,
        in other words") where no key comparison can.
        """
        asks = [r for r in self._fold().values()
                if r.get("kind") == "ask" and r.get("surface_count", 0) > 0
                and _hours_since(r.get("last_surfaced_ts")) <= within_hours]
        asks.sort(key=lambda r: r.get("last_surfaced_ts") or "", reverse=True)
        return asks[:limit]

    def lookupable_questions(self, *, ceiling: int = 1) -> list[dict]:
        """Open ``[ask:search]`` questions not yet looked up (fetch-once by default).

        ``search`` asks are the ones the lookup loop resolves (extract subject →
        fetch article → reflect). A question is eligible until it has been looked up
        *ceiling* times (default 1 — fetch once); past that it is left open but no
        longer re-fetched, so an article that *didn't* resolve it can't drive an
        endless fetch loop. It may still be resolved later by an ordinary reflection
        pass — this only retires it from automatic re-fetching."""
        return [
            r for r in self._fold().values()
            if r.get("kind") == "ask" and r.get("ask_kind") == "search"
            and r.get("lookup_count", 0) < ceiling
        ]

    def impressions(self) -> list[dict]:
        """Live ``[impression]`` items — Ava's readings of the people she talks to.

        The user-side counterpart of the persona evidence the digest folds. Folded here
        for the RAG/debug read paths; the per-person **portrait** does its own raw scan of
        the op-log (``core.user_digest.gather_user_raw``) because it needs the distinct
        *sessions* an impression recurred in, which this fold collapses away.
        """
        return [r for r in self._fold().values() if r.get("kind") == "impression"]

    def impressions_about(self, person: str) -> list[dict]:
        """Live impressions whose subject is *person* (first-token, case-insensitive).

        Matching goes through :func:`reflection_writer.normalize_person`, so "Artemy" and
        "artemy voikhansky" are one subject and a generic referent ("the user") names
        nobody — the same identity rule attribution and the portrait use.
        """
        from core.reflection_writer import normalize_person
        want = normalize_person(person)
        if not want:
            return []
        return [r for r in self.impressions()
                if normalize_person(r.get("about")) == want]

    def recollection_for(self, source_session: str) -> Optional[dict]:
        """The live ``[recollection]`` of *source_session*, or None.

        A revisit writes at most one reading per conversation and evicts the previous
        one (``ReflectionWriter.write_recollection(supersedes=…)``); this is how the
        caller finds the key to supersede. Returns the first live match — there is
        structurally only one, but a fold over an op-log written by an older build
        (before supersession) could carry several, in which case superseding any one of
        them still narrows the set on every pass.
        """
        if not source_session:
            return None
        for r in self._fold().values():
            if (r.get("kind") == "recollection"
                    and (r.get("source_session") or "") == source_session):
                return r
        return None

    @staticmethod
    def embed_text(rec: dict) -> str:
        """Text a record should be *retrieved by*.

        A ``[fact]`` carries ``(trigger: …)`` — "what should bring it back" — so it
        embeds on its trigger; an ``[ask]`` embeds on the question itself. A
        ``[recollection]`` behaves like a fact: the reading itself is the display text,
        while its ``TRIGGER:`` line names what should bring the old conversation back to
        mind. Display text (``content``) is handled separately by the caller.
        """
        if rec.get("kind") in ("fact", "recollection"):
            return (rec.get("trigger") or rec.get("content") or "").strip()
        return (rec.get("content") or "").strip()

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _fold(self) -> dict[str, dict]:
        """Replay the op-log in file order into {key: insert_record}.

        Each surviving record is annotated with ``surface_count`` — how many times
        it was proactively raised — counted from ``surface`` ops across the whole
        log (order-independent, so it survives re-inserts and interleaving) — and
        with ``last_surfaced_ts``, the newest such op's timestamp (``""`` if never
        raised), which drives the re-ask gap in :meth:`surfaceable_questions`.
        """
        items: dict[str, dict] = {}
        surfaces: dict[str, int] = {}
        surfaced_in: dict[str, list[str]] = {}
        last_surfaced: dict[str, str] = {}
        lookups: dict[str, int] = {}

        lines: list[str] = []
        if self.fallback_path is not None and self.fallback_path.exists():
            try:
                lines.extend(self.fallback_path.read_text(encoding="utf-8").splitlines())
            except Exception:
                pass
        if self.path.exists():
            try:
                lines.extend(self.path.read_text(encoding="utf-8").splitlines())
            except Exception:
                pass

        if not lines:
            return items

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("key")
            if not key:
                continue
            op = rec.get("op")
            if op == "insert":
                items[key] = rec       # re-insert supersedes earlier content
            elif op == "evict":
                items.pop(key, None)   # resolved question leaves the live set
            elif op == "supersede":
                items.pop(key, None)   # reconciliation softened it out of live recall
            elif op == "surface":
                surfaces[key] = surfaces.get(key, 0) + 1
                where = (rec.get("surfaced_in") or "").strip()
                if where:
                    surfaced_in.setdefault(key, []).append(where)
                ts = (rec.get("ts") or "").strip()
                if ts > last_surfaced.get(key, ""):
                    last_surfaced[key] = ts
            elif op == "lookup":
                lookups[key] = lookups.get(key, 0) + 1

        for key, rec in items.items():
            rec["surface_count"] = surfaces.get(key, 0)
            rec["surfaced_in_sessions"] = surfaced_in.get(key, [])
            rec["last_surfaced_ts"] = last_surfaced.get(key, "")
            rec["lookup_count"] = lookups.get(key, 0)
        return items
