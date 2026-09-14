"""Per-chat consolidation sidecar — durable verdict/target/stage beside each transcript.

Consolidation state lives in ``chats/<timestamp>.state.json`` next to
``chats/<timestamp>.json``. The chat JSON has exactly one writer (``ChatLogger``);
this module is the sole writer for the sidecar (reflection/training layer only).

See ``AVA_DESIGN_LEGACY.md`` → *Consolidation — Chat-Centric Decay* → *The sidecar*.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


SCHEMA_VERSION = 1

# Every non-transcript file that lives in the chats dir under a transcript's stem.
#
# THE load-bearing list of this module. A chat's stem owns several files
# (`<ts>.json` + its sidecars), and roughly a dozen call sites enumerate the dir
# looking for *transcripts* — RAG indexing, the reflection backlog, check-in's
# silence clock, synthesis/revisit's random pick, the session list. A suffix
# missing from here is not a cosmetic bug: the sidecar is read as a conversation.
# It gets indexed into chat RAG as if Ava had said it, counted as a chat the user
# took part in, and offered to the operator in the chat list.
#
# So: adding a sidecar means adding its suffix HERE and nowhere else. Every check
# in the tree routes through `is_chat_session_json` — except `reachout_gate`,
# which is a deliberate leaf that imports nothing from the project and carries its
# own copy (see the note there).
SIDECAR_SUFFIXES = (
    ".state.json",     # per-exchange verdicts / trainable targets / anchors
    ".summary.json",   # the consolidation gist — Ava's long memory of the chat
    ".facts.json",     # the immutable per-chat fact-extraction record
    ".shareml.json",   # legacy export artifact
)

# The subset a reflection run PRODUCES, and therefore the subset that has to travel
# staging → checkpoint → live. `reflection_staging` promotes exactly these; a suffix
# added above but not here is written into the staging workspace and then discarded
# with it. (`.shareml.json` is an export artifact, not a reflection product.)
REFLECTION_SIDECAR_SUFFIXES = (
    ".state.json",
    ".summary.json",
    ".facts.json",
)


def is_chat_session_json(path: Path) -> bool:
    """True for ``chats/<timestamp>.json``, false for any of its sidecars.

    The single definition of "this file is a transcript". See
    :data:`SIDECAR_SUFFIXES` for why that matters more than it looks.
    """
    return (path.suffix.lower() == ".json"
            and not path.name.endswith(SIDECAR_SUFFIXES))


def iter_chat_json_files(chats_dir: Path) -> Iterator[Path]:
    """Yield transcript JSON files under *chats_dir*, excluding sidecars."""
    for path in sorted(Path(chats_dir).glob("*.json")):
        if is_chat_session_json(path):
            yield path


def sidecar_path_for(chat_json: Path) -> Path:
    """``foo.json`` → ``foo.state.json`` in the same directory."""
    return chat_json.with_name(chat_json.stem + ".state.json")


def summary_path_for(chat_json: Path) -> Path:
    """``foo.json`` → ``foo.summary.json`` in the same directory."""
    return chat_json.with_name(chat_json.stem + ".summary.json")


def facts_path_for(chat_json: Path) -> Path:
    """``foo.json`` → ``foo.facts.json`` in the same directory."""
    return chat_json.with_name(chat_json.stem + ".facts.json")


def chat_products(chats_dir: Path, stem: str) -> list[Path]:
    """Existing reflection products for *stem* in *chats_dir* (state / summary / facts).

    A chat's stem owns several files and a run may write any of them, so every
    staging → checkpoint → live → archive hop has to move the SET rather than the state
    sidecar alone. Those hops each globbed ``*.state.json``, which silently stranded the
    summary in whichever workspace produced it.
    """
    out = []
    for suffix in REFLECTION_SIDECAR_SUFFIXES:
        p = Path(chats_dir) / f"{stem}{suffix}"
        if p.exists():
            out.append(p)
    return out


def iter_chat_products(chats_dir: Path) -> Iterator[Path]:
    """Yield every reflection product file in *chats_dir*, any stem."""
    for suffix in REFLECTION_SIDECAR_SUFFIXES:
        yield from sorted(Path(chats_dir).glob(f"*{suffix}"))


def session_name_from_sidecar(sidecar_path: Path) -> str:
    """``foo.state.json`` → ``foo.json``."""
    name = sidecar_path.name
    if name.endswith(".state.json"):
        return name[: -len(".state.json")] + ".json"
    return name


# --------------------------------------------------------------------- #
# Consolidation-summary (gist) sanitation                                #
# --------------------------------------------------------------------- #
#
# The summary pass is instructed to write pure prose, but in practice it very often
# appends — or emits outright — the structured consolidation dump instead
# (`<prose>\n\n## WEIGHTS\n- [fact] …\n## RAG\n…\n## RESOLVED\n…`). Measured on the
# 2026-07 corpus: 110 of 173 stored summaries carried such a block, 34 of them from
# the very first line. That matters because `rag_engine` chunks this text into the
# chat-RAG index as `kind="gist"` passages — the ONLY chat representation surviving
# past `rag_cap_age_h` — so a leaked dump is injected into live chat as if it were a
# remembered conclusion, teaching Ava to recall her own reflection format.
#
# Both the writer (below) and the reader (`rag_engine._collect_chat_entries`) run the
# text through `sanitize_gist`: the writer so nothing new is stored dirty, the reader
# so the already-stored corpus is repaired without a migration or a destructive rewrite
# of sidecars.

GIST_MIN_CHARS = 40   # salvaged prose shorter than this carries nothing worth recalling

# Section words that mark the start of a leaked structured block.
_GIST_STRUCT_WORDS = ("rag", "weights", "resolved", "final verdict")


def gist_is_structure_boundary(line: str) -> bool:
    """True when *line* begins the leaked consolidation block (a `***` separator, a
    `## WEIGHTS`/`**RAG**`/`VERDICT:` section header, or a bulleted `[fact]`/`[ask]` item)."""
    s = line.strip()
    if not s:
        return False
    if s in ("***", "---", "* * *", "___"):
        return True
    low = s.lower()
    body = low.lstrip("-*•· ").strip()            # a bulleted structured item
    if body.startswith(("[fact]", "[ask", "[persona]", "[resolved]")):
        return True
    hdr = low.lstrip("#*• ").rstrip("*: ").strip()  # a markdown/bold section header
    if hdr in _GIST_STRUCT_WORDS or hdr.startswith(_GIST_STRUCT_WORDS):
        return True
    if low.startswith(("weights:", "rag:", "verdict:", "resolved:")):
        return True
    return False


def sanitize_gist(text: str) -> str:
    """Return the usable PROSE of a consolidation summary, or ``""`` if there is none.

    Keeps everything before the first structure boundary, drops a leading narrative
    header (`### Reflection on Session …`) and any text after an `<eos>` marker, and
    requires ``GIST_MIN_CHARS`` of real content to remain.

    Truncating at the FIRST boundary (rather than filtering structured lines wherever
    they occur) is deliberate: the dump's own continuation lines are not themselves
    boundaries — a `## RESOLVED` section lists plain `- artemyvo: did he …` items — so
    line-wise filtering would keep those fragments and read as prose. On the measured
    corpus only 11 of 110 leaky summaries had any non-boundary line after the first
    boundary, and inspection showed every one of them to be exactly such a continuation.

    Paragraph structure is preserved (the reader chunks this into passages); only the
    length check collapses whitespace.
    """
    t = (text or "").strip()
    if not t:
        return ""
    low = t.lower()
    if "<eos>" in low:                       # keep only prose before an EOS marker
        t = t[: low.find("<eos>")].rstrip()
        if not t:
            return ""
    kept: list[str] = []
    for line in t.splitlines():
        if gist_is_structure_boundary(line):
            break
        kept.append(line)
    while kept and kept[0].lstrip().startswith("#"):   # drop a leading "### Reflection"
        kept.pop(0)
    while kept and not kept[0].strip():
        kept.pop(0)
    out = "\n".join(kept).strip()
    if len(" ".join(out.split())) < GIST_MIN_CHARS:
        return ""
    return out


def gist_excerpt(text: str, cap: int) -> str:
    """Trim a stored gist to *cap* chars, ending on a finished thought.

    Keeps whole paragraphs while they fit (a gist is written in paragraphs, and one of
    them entire reads better than a cut across two); a single paragraph longer than the
    whole allowance is cut at the last sentence end inside it. An elision is marked, so a
    recap that stops early is not read as a conversation that had nothing more to it.

    Lives here rather than in either of its callers because it is a gist-shaping rule, and
    both of them are already importing this module for ``summary_text``: check-in reduces
    a chat to recap size for its decision window (``checkin._gist_excerpt``), and the
    fact-nomination slot clips one to a single injected passage
    (``rag_engine._render_nomination``). Two copies of "how do you shorten a gist" is two
    things to keep in step, and the second caller is what proved it.
    """
    t = (text or "").strip()
    if not t:
        return ""
    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    full = "\n\n".join(paras)
    if len(full) <= cap:
        return full
    kept: list[str] = []
    total = 0
    for p in paras:
        if kept and total + len(p) > cap:
            break
        kept.append(p)
        total += len(p)
        if total >= cap:
            break
    out = "\n\n".join(kept).strip()
    if len(out) > cap:
        head = out[:cap]
        cut = max(head.rfind(ch) for ch in (".", "!", "?", "…"))
        out = head[: cut + 1].strip() if cut >= cap // 3 else head.rstrip()
    return out if out == full else out + " …"


class ChatSidecar:
    """Read/write per-exchange consolidation state for one chats/ directory."""

    def __init__(self, chats_dir: Path, fallback_chats_dir: Optional[Path] = None) -> None:
        self.chats_dir = Path(chats_dir)
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_chats_dir = Path(fallback_chats_dir) if fallback_chats_dir is not None else None

    # ------------------------------------------------------------------ #
    # Path resolution                                                     #
    # ------------------------------------------------------------------ #

    def resolve_chat_path(self, source_session: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        """Return the chat JSON path for *source_session*, or None if unsafe."""
        name = (source_session or "").strip()
        if (
            not name
            or "/" in name
            or "\\" in name
            or name in (".", "..")
            or name.endswith(SIDECAR_SUFFIXES)
        ):
            return None
        target_dir = Path(base_dir) if base_dir is not None else self.chats_dir
        path = (target_dir / name).resolve()
        if path.parent != target_dir.resolve():
            return None
        if path.suffix.lower() != ".json":
            return None
        return path

    def resolve_sidecar_path(self, source_session: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        chat = self.resolve_chat_path(source_session, base_dir=base_dir)
        return sidecar_path_for(chat) if chat is not None else None

    def resolve_summary_path(self, source_session: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        chat = self.resolve_chat_path(source_session, base_dir=base_dir)
        return summary_path_for(chat) if chat is not None else None

    def resolve_facts_path(self, source_session: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        chat = self.resolve_chat_path(source_session, base_dir=base_dir)
        return facts_path_for(chat) if chat is not None else None

    def is_live_session(self, source_session: str, live_session: Optional[str]) -> bool:
        """True when *source_session* is the active chat logger file."""
        if not live_session:
            return False
        chat = self.resolve_chat_path(source_session)
        if chat is None:
            return False
        try:
            return chat.resolve() == (self.chats_dir / live_session).resolve()
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #

    def load(self, source_session: str) -> dict:
        """Return the sidecar document, or an empty shell if none exists yet."""
        path = self.resolve_sidecar_path(source_session)
        if path is None or not path.exists():
            if self.fallback_chats_dir is not None:
                fb_path = self.resolve_sidecar_path(source_session, base_dir=self.fallback_chats_dir)
                if fb_path is not None and fb_path.exists():
                    try:
                        data = json.loads(fb_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            exchanges = data.get("exchanges")
                            if not isinstance(exchanges, dict):
                                data["exchanges"] = {}
                            data.setdefault("schema_version", SCHEMA_VERSION)
                            data.setdefault("source_session", source_session)
                            return data
                    except Exception:
                        pass
            return self._empty_doc(source_session)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_doc(source_session)
        if not isinstance(data, dict):
            return self._empty_doc(source_session)
        exchanges = data.get("exchanges")
        if not isinstance(exchanges, dict):
            data["exchanges"] = {}
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("source_session", source_session)
        return data

    def get_exchange(self, source_session: str, exchange_index: int) -> Optional[dict]:
        """Per-exchange record, or None if never vetted."""
        key = str(exchange_index)
        rec = self.load(source_session).get("exchanges", {}).get(key)
        return dict(rec) if isinstance(rec, dict) else None

    def stage_of(self, source_session: str, exchange_index: int) -> int:
        """Consolidation stage for an exchange (0 when unprocessed)."""
        rec = self.get_exchange(source_session, exchange_index)
        if rec is None:
            return 0
        try:
            return max(0, int(rec.get("stage", 0)))
        except (TypeError, ValueError):
            return 0

    def is_reflected(self, source_session: str) -> bool:
        """True once this session has been fully reflected (reflect-once freeze).

        Set once by :meth:`mark_reflected` when a reflection run finishes a session's
        consolidation + revision. A frozen sidecar is immutable: the runner skips it (no
        re-reflection, continue-staging re-run, or judge override on an old chat). See
        ``REBUILD.md`` §3.
        """
        return bool(self.load(source_session).get("reflected_at"))

    def is_chat_reflected(self, source_session: str) -> bool:
        """True once the session's PER-CHAT passes are done (first stage of the two-stage
        freeze).

        Set by :meth:`mark_chat_reflected` when the background per-chat reflection pass
        finishes a chat's consolidation + revision + branch generation. It blocks
        re-*generation* (the expensive part) but NOT the nightly run-level passes: a chat
        that is ``chat_reflected`` and not yet ``reflected_at`` is still picked up by a
        normal reflection run, which runs the clean-base judge + fact placement over its
        persisted job payloads and then stamps ``reflected_at``. See
        ``core.background_reflection``.
        """
        return bool(self.load(source_session).get("chat_reflected"))

    # ------------------------------------------------------------------ #
    # Writes — reflection layer                                           #
    # ------------------------------------------------------------------ #

    def write_verdict(
        self,
        *,
        source_session: str,
        exchange_index: int,
        verdict: str,
        target: str,
        run_id: str,
        user_prompt: str = "",
        target_source: str = "",
        target_kind: str = "",
        target_generation: str = "",
        persona_context: str = "",
        locked: bool = False,
        live_session: Optional[str] = None,
    ) -> bool:
        """Persist a revision vetting outcome. Stage is **not** advanced here.

        Returns False when the write is skipped (invalid session, live session,
        or untrustworthy target). Returns True on success.

        *locked* marks the exchange **human-validated** (a manually regenerated,
        operator-reviewed target from the Training review tab). A locked exchange is
        preserved across re-reflection AND revisit — the runner skips re-deriving it — so
        the original poison can't contaminate a future build. Once set it is sticky (a
        later ``write_verdict`` cannot silently clear it): only an explicit locked=True
        write ever sets it, and it is OR-ed with the prior value.
        """
        if target_source == "revised_missing_ideal":
            return False
        if not (target or "").strip():
            return False
        if self.is_live_session(source_session, live_session):
            return False

        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False

        data = self.load(source_session)
        key = str(exchange_index)
        exchanges = data.setdefault("exchanges", {})
        prev = exchanges.get(key) if isinstance(exchanges.get(key), dict) else {}

        exchanges[key] = {
            "stage": prev.get("stage", 0),
            "verdict": (verdict or "").strip().lower(),
            "target": target,
            "user_prompt": user_prompt or prev.get("user_prompt", ""),
            "target_source": target_source or prev.get("target_source", ""),
            "target_kind": target_kind or prev.get("target_kind", ""),
            "target_generation": target_generation or prev.get("target_generation", ""),
            # Persona self-knowledge folded into the IDEAL generation's system prompt for a
            # revise/ideal-win target, persisted verbatim so build_dialogue_anchor can
            # reconstruct the identical system message (train/inference parity). Written
            # FRESH each verdict (not carried from prev): a keep/branch/manual target passes
            # "" and correctly clears any stale block — e.g. a judge criterion-flip to a
            # branch, whose CoT is the original and was never persona-conditioned.
            "persona_context": persona_context or "",
            "run_id": run_id,
            "last_reflected": datetime.now().isoformat(),
            "last_trained": prev.get("last_trained"),
            # Sticky once set: OR with the prior value so a stray non-locking write can
            # never un-validate an exchange the operator hand-authored.
            "locked": bool(locked) or bool(prev.get("locked")),
            # Carried, never set here: an operator's training ban survives re-reflection
            # (which happily re-derives a fine-looking target for an exchange that must not
            # train at all). Only set_exchange_banned writes it.
            "banned": bool(prev.get("banned")),
        }
        return self._save(path, data)

    def write_anchor(
        self, *, source_session: str, exchange_index: int,
        about: str, tags: list, run_id: str, generation: str = "",
        live_session: Optional[str] = None,
    ) -> bool:
        """Persist a per-exchange retrieval ANCHOR — a one-line descriptor + tags.

        Stored under a **separate top-level ``anchors`` map**, not inside the exchange's
        verdict record. That isolation is deliberate: `exchanges[<i>]` is what
        `training/dialogue_source` reads to build a trainable row, so keeping anchors out
        of it guarantees this pass cannot perturb training no matter what it generates.
        ``about`` and ``tags`` are kept as separate fields of the record for the same
        reason — they are two different retrieval channels (dense descriptor, sparse tags)
        and will be indexed independently.

        Producer-side only: nothing reads these yet. Overwrites any prior anchor for the
        exchange (a later reflection or a revisit re-derives it under the evolved persona,
        exactly like the consolidation summary). Refuses the live session and an
        unresolvable path; a record with neither an ``about`` nor a tag is not written.
        """
        about = (about or "").strip()
        tags = [t for t in (tags or []) if (t or "").strip()]
        if not about and not tags:
            return False
        if self.is_live_session(source_session, live_session):
            return False
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False
        data = self.load(source_session)
        anchors = data.setdefault("anchors", {})
        if not isinstance(anchors, dict):
            anchors = {}
            data["anchors"] = anchors
        anchors[str(exchange_index)] = {
            "about": about,
            "tags": tags,
            "generation": generation,
            "run_id": run_id,
            "ts": datetime.now().isoformat(),
        }
        return self._save(path, data)

    def set_exchange_locked(self, source_session: str, exchange_index: int, locked: bool) -> bool:
        """Deliberately set/clear the human-validation lock on one exchange.

        ``write_verdict`` OR-s ``locked`` (sticky) so a stray reflection write can never
        un-validate a hand-authored target. This is the one *intentional* path that can
        CLEAR it: the Training review "Rewrite history" action bakes the reviewed target
        into the transcript itself (``chat_logger.rewrite_exchange_history``) and then
        unfreezes the exchange — the lock's job (protect the reviewed target from being
        re-derived over corrupt content) is done once the content itself is corrected.

        Edits the real sidecar in ``chats_dir`` (never the fallback). Returns True on
        success, False on an unresolvable path, a missing sidecar, or a missing record.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None or not path.exists():
            return False
        data = self.load(source_session)
        ex = data.get("exchanges")
        key = str(exchange_index)
        if not isinstance(ex, dict) or not isinstance(ex.get(key), dict):
            return False
        ex[key]["locked"] = bool(locked)
        return self._save(path, data)

    def set_exchange_banned(self, source_session: str, exchange_index: int,
                            banned: bool) -> bool:
        """Ban (or un-ban) one exchange from ever contributing a training row.

        The Training review tab's counterpart to a repair: some exchanges cannot be fixed
        into something worth learning from — a bug-mangled generation, an experiment that
        went nowhere — and the honest answer is that they should not train, not that they
        should train from a hand-written substitute. ``dialogue_source.build_dialogue_anchor``
        refuses a banned record outright, so the row leaves the corpus on the very next
        build, and ``write_verdict`` carries the flag forward so a later reflection pass
        cannot quietly re-derive a target for it.

        Unlike ``locked`` this is NOT sticky — it is a plain operator toggle, since the
        mistake it guards against (a wrongly banned exchange staying silently out of every
        future build) is the opposite of the one ``locked`` guards against.

        Creates the per-exchange record (and the sidecar) when the exchange was never vetted:
        an unreflected exchange has no record yet, and it must still be bannable. Such a
        record carries no target, so it contributes nothing beyond the flag itself. Edits the
        real sidecar in ``chats_dir`` (never the fallback). Returns True on success.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False
        data = self.load(source_session)
        exchanges = data.setdefault("exchanges", {})
        if not isinstance(exchanges, dict):
            exchanges = {}
            data["exchanges"] = exchanges
        key = str(exchange_index)
        rec = exchanges.get(key)
        if not isinstance(rec, dict):
            rec = {"stage": 0, "verdict": "", "target": "", "user_prompt": "",
                   "target_source": "", "run_id": "training-review-ban",
                   "last_reflected": datetime.now().isoformat(), "last_trained": None}
            exchanges[key] = rec
        if banned:
            rec["banned"] = True
        else:
            rec.pop("banned", None)
        return self._save(path, data)

    def banned_exchange_indices(self, source_session: str) -> set:
        """Exchange indices the operator banned from training (record ``banned``)."""
        doc = self.load(source_session)
        ex = doc.get("exchanges")
        out: set = set()
        if isinstance(ex, dict):
            for k, rec in ex.items():
                if isinstance(rec, dict) and rec.get("banned"):
                    try:
                        out.add(int(k))
                    except (TypeError, ValueError):
                        pass
        return out

    def drop_exchange(self, source_session: str, exchange_index: int) -> bool:
        """Delete one exchange's sidecar state and RENUMBER the ones after it.

        The finalize half of a ban ("Rewrite history"): once the exchange itself is gone from
        the transcript, every later exchange has shifted down one position, so the sidecar's
        two index-keyed maps — the per-exchange verdict records and the top-level ``anchors``
        map — must shift with it or they would describe the wrong turns. This is the reason
        deletion is a deliberate finalize step rather than something the ban does directly:
        it invalidates positions, and everything that holds one has to be moved in the same
        breath (the ledger's fact hosts are the caller's job — see
        ``session_ops._renumber_fact_hosts``).

        Returns True when the sidecar was rewritten.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None or not path.exists():
            return False
        data = self.load(source_session)
        touched = False
        for field in ("exchanges", "anchors"):
            table = data.get(field)
            if not isinstance(table, dict):
                continue
            shifted: dict = {}
            for k, rec in table.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    shifted[k] = rec        # unparseable key — leave it verbatim
                    continue
                if idx == exchange_index:
                    touched = True
                    continue                # the deleted exchange's own state
                shifted[str(idx - 1 if idx > exchange_index else idx)] = rec
                if idx > exchange_index:
                    touched = True
            data[field] = shifted
        if not touched:
            return False
        return self._save(path, data)

    def locked_exchange_indices(self, source_session: str) -> set:
        """Exchange indices the operator hand-validated (record ``locked``).

        These are preserved across re-reflection and revisit — the reflection runner
        skips re-deriving them so their reviewed target can't be overwritten by a fresh
        pass over the original (corrupt) transcript. Reads via ``load`` so a staged run
        sees the live locks through the fallback dir."""
        doc = self.load(source_session)
        ex = doc.get("exchanges")
        out: set = set()
        if isinstance(ex, dict):
            for k, rec in ex.items():
                if isinstance(rec, dict) and rec.get("locked"):
                    try:
                        out.add(int(k))
                    except (TypeError, ValueError):
                        pass
        return out

    # ------------------------------------------------------------------ #
    # Writes — training layer                                             #
    # ------------------------------------------------------------------ #

    def advance_stages(
        self,
        source_session: str,
        exchange_indices: list[int],
        *,
        trained_at: Optional[str] = None,
    ) -> int:
        """Bump stage for each exchange after a successful train cycle.

        Returns the number of exchanges advanced. Skips unknown indices.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return 0

        data = self.load(source_session)
        exchanges = data.get("exchanges", {})
        if not isinstance(exchanges, dict):
            return 0

        ts = trained_at or datetime.now().isoformat()
        n = 0
        for idx in exchange_indices:
            key = str(idx)
            rec = exchanges.get(key)
            if not isinstance(rec, dict):
                continue
            try:
                stage = max(0, int(rec.get("stage", 0)))
            except (TypeError, ValueError):
                stage = 0
            rec["stage"] = stage + 1
            rec["last_trained"] = ts
            n += 1

        if n:
            self._save(path, data)
        return n

    def import_exchange(
        self,
        *,
        source_session: str,
        exchange_index: int,
        stage: int = 0,
        verdict: str = "",
        target: str = "",
        user_prompt: str = "",
        target_source: str = "",
        run_id: str = "migrated",
        last_trained: Optional[str] = None,
    ) -> bool:
        """Seed one exchange from a legacy ledger anchor. Skips if already present."""
        if not (target or "").strip():
            return False
        if self.get_exchange(source_session, exchange_index) is not None:
            return False

        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False

        data = self.load(source_session)
        key = str(exchange_index)
        data.setdefault("exchanges", {})[key] = {
            "stage": max(0, int(stage)),
            "verdict": (verdict or "").strip().lower(),
            "target": target,
            "user_prompt": user_prompt,
            "target_source": target_source or ("revised" if verdict == "revise" else "original"),
            "run_id": run_id,
            "last_reflected": datetime.now().isoformat(),
            "last_trained": last_trained,
        }
        return self._save(path, data)

    def advance_by_session(self, advances: dict[str, list[int]], *, trained_at: Optional[str] = None) -> int:
        """Advance many sessions at once. Keys are session filenames."""
        total = 0
        for session, indices in advances.items():
            total += self.advance_stages(session, indices, trained_at=trained_at)
        return total

    def mark_reflected(self, source_session: str, *, reflected_at: Optional[str] = None) -> bool:
        """Stamp the session as fully reflected (reflect-once freeze — ``REBUILD.md`` §3).

        Sets a doc-level ``reflected_at`` once; a second call is a no-op (the stamp is set
        once, by the contemporary adapter, and never rewritten). Returns True when the
        sidecar is stamped (already or now), False only on an unresolvable path.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False
        data = self.load(source_session)
        if data.get("reflected_at"):
            return True
        data["reflected_at"] = reflected_at or datetime.now().isoformat()
        return self._save(path, data)

    def mark_chat_reflected(self, source_session: str,
                            *, chat_reflected: Optional[str] = None) -> bool:
        """Stamp the session as PER-CHAT reflected — stage one of the two-stage freeze.

        Set by the background per-chat pass (``core.background_reflection``) once a chat's
        consolidation + revision + branch generation are done and its clean-base job
        payloads are checkpointed. Sets a doc-level ``chat_reflected`` timestamp once; a
        second call is a no-op. Unlike ``reflected_at`` this does NOT hide the chat from a
        normal reflection run — that run finishes it (clean-base judge + fact placement)
        and stamps ``reflected_at``. Returns True when stamped (already or now), False only
        on an unresolvable path.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False
        data = self.load(source_session)
        if data.get("chat_reflected"):
            return True
        data["chat_reflected"] = chat_reflected or datetime.now().isoformat()
        return self._save(path, data)

    def clear_reflection_freeze(self, source_session: str) -> bool:
        """Un-freeze a session so it re-enters the reflection backlog — the un-do of
        :meth:`mark_reflected` / :meth:`mark_chat_reflected`.

        Used when an already-reflected chat is CONTINUED (an Ava-initiated session resumed
        in place — see ``session_ops.handle_load_session``): the appended turns are invisible
        to reflection while the freeze stands, so clearing both stamps lets a later run
        re-reflect the whole (now-longer) conversation. Per-exchange verdicts / ``locked``
        flags / the consolidation summary are LEFT intact — re-reflection overwrites the
        per-exchange targets and honours locked exchanges, so nothing needs deleting here.
        Returns True iff a stamp was actually present and removed. NB: for a chat frozen only
        at the ``chat_reflected`` stage, the caller must also purge its background checkpoint
        artifacts (``reflection_staging.purge_background_artifacts``) so the next normal run's
        checkpoint fold cannot re-freeze it.
        """
        path = self.resolve_sidecar_path(source_session)
        if path is None:
            return False
        data = self.load(source_session)
        if not (data.get("reflected_at") or data.get("chat_reflected")):
            return False
        data.pop("reflected_at", None)
        data.pop("chat_reflected", None)
        return self._save(path, data)

    # ------------------------------------------------------------------ #
    # Consolidation summary — its OWN file (`<stem>.summary.json`)         #
    # ------------------------------------------------------------------ #
    #
    # It used to be a `consolidation_summary` key inside `.state.json`, which tied
    # Ava's long memory of a conversation to the lifetime of that chat's *current
    # reflection*. Three paths delete the state sidecar wholesale — `mark_corrupt`,
    # `reset_session_reflection` ("Re-reflect chat"), and the checkpoint fold — and
    # every one of them is about a TRAINING VERDICT being wrong, not about the
    # conversation being misremembered. Past `rag_cap_age_h` (~96h) the gist is the
    # only representation of a chat left in RAG, so re-reflecting a week-old chat to
    # fix one bad target silently erased the whole conversation from her memory until
    # the next Sleep run happened to complete.
    #
    # Two different clocks, so two files: the summary's lifetime is the
    # conversation's, the state sidecar's is this reflection's.

    def read_summary(self, source_session: str) -> dict:
        """Return the stored consolidation-summary doc, or ``{}`` if there is none.

        Reads the dedicated `.summary.json` first and falls back to the legacy
        ``consolidation_summary`` key inside `.state.json` — the corpus predates the
        split, and a chat whose state sidecar still holds the only copy must keep
        being recalled. No migration pass: the legacy copy is read where it lies and
        superseded the next time this chat is reflected.

        The text is sanitized on READ as well as on write (see :func:`sanitize_gist`):
        most of the stored corpus predates the write-side gate and carries a leaked
        ``## WEIGHTS`` dump, and this is the one channel that survives to be injected
        into live chat as a remembered conclusion.
        """
        for base in (self.chats_dir, self.fallback_chats_dir):
            if base is None:
                continue
            path = self.resolve_summary_path(source_session, base_dir=base)
            if path is None or not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            text = sanitize_gist(data.get("text") or "")
            if text:
                return {**data, "text": text}
        legacy = self.load(source_session).get("consolidation_summary")
        if isinstance(legacy, dict):
            text = sanitize_gist(legacy.get("text") or "")
            if text:
                return {**legacy, "text": text}
        return {}

    def summary_text(self, source_session: str) -> str:
        """The sanitized prose of this chat's consolidation summary, or ``""``."""
        return (self.read_summary(source_session).get("text") or "").strip()

    def write_summary(
        self, *, source_session: str, summary: str, run_id: str,
        live_session: Optional[str] = None,
    ) -> bool:
        """Persist the per-conversation consolidation SUMMARY — a distilled prose recap.

        Session-level (not per-exchange): a dedicated reflection generation writes one clean
        recap of the whole conversation here, which ``rag_engine`` chunks into the chat-RAG
        index as ``kind="gist"`` passages carried on the slow gist tent
        (``gist_rag_weight_hours``) — the crossfade partner of the verbatim fade, so a chat is
        recalled verbatim while fresh and through its summary once aged. RAG-only by
        construction: it never touches the weights/ledger stores. Overwrites any prior summary
        (a revisit re-derives it under the evolved persona). Refuses the live session (a
        concurrent ChatLogger owns nothing here, but the reflect-once contract still applies)
        and an unresolvable path.

        The text is run through :func:`sanitize_gist` first, so a pass that appended (or
        emitted only) the structured consolidation dump stores its prose alone. A summary
        that salvages nothing is REFUSED rather than stored dirty: no gist at all is
        strictly better than a `## WEIGHTS` block injected into live chat as a memory, and
        an existing good summary is left in place instead of being overwritten by garbage.
        """
        clean = sanitize_gist(summary)
        if not clean:
            return False
        if self.is_live_session(source_session, live_session):
            return False
        path = self.resolve_summary_path(source_session)
        if path is None:
            return False
        return self._save(path, {
            "source_session": source_session,
            "text": clean,
            "run_id": run_id,
            "ts": datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------ #
    # Fact-extraction record — its OWN file (`<stem>.facts.json`)          #
    # ------------------------------------------------------------------ #
    #
    # The immutable protocol of one conversation. See `core.chat_facts` for what it
    # is for and why it is not a second copy of the live fact store.

    def read_facts(self, source_session: str) -> dict:
        """Return the stored fact-extraction doc, or ``{}`` if there is none."""
        for base in (self.chats_dir, self.fallback_chats_dir):
            if base is None:
                continue
            path = self.resolve_facts_path(source_session, base_dir=base)
            if path is None or not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict) and isinstance(data.get("facts"), list):
                return data
        return {}

    def write_facts(
        self, *, source_session: str, facts: list, run_id: str,
        source_user: str = "", live_session: Optional[str] = None,
    ) -> bool:
        """Persist this chat's fact-extraction record. Overwrites any prior one.

        *facts* are :func:`core.chat_facts.parse_facts` records. *source_user* is who
        was speaking — supplied by the caller from the session record, never parsed out
        of model output, exactly as ``write_consolidation`` takes it: it is the one piece
        of provenance a generation must not be able to invent.

        Overwrite-in-full rather than append: the record is this chat's extraction *as
        of the run that read it*, so a re-reflection under an evolved persona replaces
        it wholesale. Immutability is against the LIVE store's dedup/eviction churn —
        nothing merges or evicts inside this file — not against re-derivation.

        An empty *facts* list is refused, so a failed pass leaves the previous record
        standing instead of blanking it.
        """
        if not facts:
            return False
        if self.is_live_session(source_session, live_session):
            return False
        path = self.resolve_facts_path(source_session)
        if path is None:
            return False
        return self._save(path, {
            "source_session": source_session,
            "source_user": source_user,
            "facts": list(facts),
            "run_id": run_id,
            "ts": datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_doc(source_session: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_session": source_session,
            "exchanges": {},
        }

    def _save(self, path: Path, data: dict) -> bool:
        """Atomic write: temp file in the same directory, then rename."""
        data["schema_version"] = SCHEMA_VERSION
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=path.stem + ".", suffix=".tmp", dir=path.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            return True
        except Exception:
            return False
