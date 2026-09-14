"""Chat session logger — persists conversations to JSON files in the chats/ directory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

# Chat-file format version. Stamped on every session so a later migration can
# detect the format and skip re-running an (expensive) regeneration. Files with
# no `schema_version` key are implicitly version 1 (the legacy unversioned format).
#   1 — legacy: no version field.
#   2 — adds session-level `schema_version`/`adapter_id` and per-exchange
#       `system_content` (exact assembled system message) + `input_tokens`.
#   3 — adds stable per-exchange `exchange_id` and optional post-reply
#       `reflection_feedback` for the revision/persona pass.
#   4 — adds optional per-exchange `corrupt_cot`/`corrupt_response` flags (set from
#       the Training review tab) marking a stored CoT/reply as corrupt so reflection
#       treats it as missing/insists on a re-derived IDEAL and training drops it.
#   5 — the Training review "Rewrite history" action may rewrite `assistant_cot`/
#       `assistant_response` in place from a human-reviewed sidecar target (baking the
#       correction into ground truth), recording the prior content under an append-only
#       per-exchange `rewrite_history` list and REMOVING the now-stale `tension` block
#       (its per-token logits were measured on the original generation).
#   6 — the same action may DELETE an exchange the operator banned from training
#       (`delete_exchange`), shifting every later exchange down one index and recording
#       the removed turn under a session-level append-only `deleted_exchanges` list.
CHAT_SCHEMA_VERSION = 6
REFLECTION_FEEDBACK_MAX_CHARS = 2000


class ChatLogger:
    """Manages a single chat session file and appends exchanges to it."""

    def __init__(self, chats_dir: Path) -> None:
        self.chats_dir = chats_dir
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None
        self._session_data: dict = {}

    @property
    def current_file(self) -> Optional[Path]:
        """Path of the active session file, or None if no session has started."""
        return self._current_file

    @property
    def exchange_count(self) -> int:
        """Number of exchanges logged to the active session so far."""
        return len(self._session_data.get("exchanges", []))

    @property
    def initiated_by(self) -> str:
        """Who opened the active session ("ava" for a self-initiated outreach, else "")."""
        return str(self._session_data.get("initiated_by") or "").strip()

    @property
    def initiated_ask(self) -> dict:
        """The stamped ``initiated_ask`` of the active session, or ``{}``.

        Read at chat time by `generation`'s ASK ORIGIN injection (`core.ask_origin`)
        as well as by reflection's join — a copy, so a caller can't mutate the stamp."""
        stamp = self._session_data.get("initiated_ask")
        return dict(stamp) if isinstance(stamp, dict) else {}

    def remove_last_exchange(self) -> Optional[dict]:
        """Pop the most-recent exchange off the active session file and re-save.

        The Chat-tab **Retry** action: a just-completed reply the operator wants to redo
        (typically a degenerate/collapsed generation) is rolled off the transcript so it
        never reaches reflection or training, and its user prompt can be resent under
        adjusted sampling. Returns the removed exchange dict (so the caller can restore
        the prompt), or ``None`` when there is nothing to remove. An emptied transcript is
        left on disk (an exchange-less file, harmless — the next log_exchange re-fills it),
        so the logger's ``current_file`` and RAG fence stay stable across the retry.
        """
        exchanges = self._session_data.get("exchanges") or []
        if not exchanges:
            return None
        removed = exchanges.pop()
        self._save()
        return removed

    def start_session(
        self,
        system_prompt: str = "",
        user: str = "",
        model_id: str = "",
        notes: str = "",
        adapter_id: Optional[str] = None,
        continued_from: Optional[str] = None,
        initiated_by: Optional[str] = None,
        interlocutor: Optional[str] = None,
        initiated_ask: Optional[dict] = None,
    ) -> None:
        """Begin a new chat session; creates a new timestamped JSON file.

        *adapter_id* identifies the persistent LoRA adapter (atop the frozen base
        named by *model_id*) that produced this session, once that training design
        lands; it is reserved now (``None`` on a merged base) so the format is
        stable and post-rework logging only fills it in — no second regeneration.
        *continued_from* names the prior session file when this session is a
        continuation rather than a fresh conversation.
        *initiated_by* marks who opened the conversation when it wasn't the user:
        ``"ava"`` for a self-initiated outreach session, where exchange 0 is Ava's
        own opener (logged under a stage-direction speaker) rather than a reply to a
        user turn. The flag is the parsing contract downstream reads — reflection
        treats the opener honestly and training masks it (it has no user stimulus).
        *interlocutor* marks who is on the other side when it is not the human user:
        ``"ai"`` for an encounter or a served gossip conversation, where every
        ``user_prompt`` is another model's reply. Absent ⇒ the human. It exists because
        such a transcript is otherwise indistinguishable from a live conversation, and
        anything measuring *the user's* presence (check-in's silence clock and its
        recent-window review) would read Ava talking to a peer as the user talking to
        Ava. Reflection deliberately ignores it: an encounter is hers to reflect on
        exactly like any other session.
        *initiated_ask* stamps the open ``[ask]`` that triggered a self-initiated
        session — ``{"key", "content", "ask_kind", "source_session"}`` — so reflection
        can join this thread back to the exact question it was raised to answer and
        decide, from the actual replies, whether it was resolved (the tight [ask]-loop
        close). ``source_session`` (the ask's own origin ref — the chat it was
        distilled from, or a ``til:``/``<kind>/<stem>`` reading ref) is stamped here
        rather than looked up later because the live ask is TRANSIENT: once it is
        resolved/evicted the fold no longer answers "where did this question come
        from?", and the answer-side ASK ORIGIN injection (`core.ask_origin`) needs it
        for the whole life of the session. Persists across an in-place reply
        (``resume_session`` loads the whole dict).
        """
        now = datetime.now()
        # Second-granularity stem. Two sessions started in the same second (e.g. a
        # gossip conversation forking right after the previous one, or an encounter
        # started back-to-back) would otherwise share a filename and the later one
        # would overwrite the earlier transcript on its first save. Uniquify against
        # what is already on disk with a short suffix, preserving the timestamp stem.
        stem = now.strftime("%Y%m%d_%H%M%S")
        candidate = self.chats_dir / (stem + ".json")
        suffix = 1
        while candidate.exists():
            candidate = self.chats_dir / f"{stem}_{suffix}.json"
            suffix += 1
        self._current_file = candidate
        self._session_data = {
            "schema_version": CHAT_SCHEMA_VERSION,
            "timestamp": now.isoformat(),
            "system_prompt": system_prompt,
            "user": user,
            "model_id": model_id,
            "adapter_id": adapter_id,
            "notes": notes,
            "exchanges": [],
        }
        if continued_from:
            self._session_data["continued_from"] = continued_from
        if initiated_by:
            self._session_data["initiated_by"] = initiated_by
        if interlocutor:
            self._session_data["interlocutor"] = interlocutor
        if initiated_ask and initiated_ask.get("key"):
            self._session_data["initiated_ask"] = {
                "key": initiated_ask.get("key"),
                "content": initiated_ask.get("content") or "",
                "ask_kind": initiated_ask.get("ask_kind") or "",
                "source_session": initiated_ask.get("source_session") or "",
            }
        # The file is *not* written here: an exchange-less session (e.g. the user
        # presses Clear context, or loads a past session just to view it) would
        # otherwise leave an orphaned transcript on disk. The first log_exchange
        # persists it; until then the session lives only in memory.

    def resume_session(self, path: Path) -> None:
        """Adopt an existing session file so new exchanges append to it *in place*.

        Unlike ``start_session`` (which always forks a fresh timestamped file), this
        points the logger at an existing transcript and loads its data, so the next
        ``log_exchange`` appends to the same file rather than starting a new one. Used
        to continue an Ava-initiated outreach session in its own file — the reply is a
        direct continuation of that one reversed conversation, so it belongs together
        (and keeps the masked opener + the real exchanges as a single reflectable unit).
        """
        path = Path(path)
        self._current_file = path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("exchanges", [])
        self._session_data = data

    def set_notes(self, notes: str) -> None:
        """Update the free-form intent/notes field on the active session."""
        if self._current_file is None:
            return
        self._session_data["notes"] = notes
        # Don't materialise the file for notes alone — that would re-create the
        # orphan we avoid in start_session. The notes ride along on the first
        # log_exchange save; a notes-only session never hits disk.
        if self._session_data.get("exchanges"):
            self._save()

    def log_exchange(
        self,
        user_prompt: str,
        assistant_full_response: str,
        rag_context: str = "",
        tension: Optional[dict] = None,
        speaker: str = "",
        generation_params: Optional[dict] = None,
        system_content: str = "",
        input_tokens: Optional[int] = None,
        think_open_prob: Optional[float] = None,
        exchange_id: Optional[str] = None,
    ) -> None:
        """Parse and append one user/assistant exchange to the current session.

        *speaker* is the name of the user who sent *user_prompt*; it is stored so
        the model can later attribute remembered facts to whoever told her.
        *generation_params* records what produced this specific reply (temperature,
        top_p, max_new_tokens_setting) so per-exchange overrides remain traceable.
        *system_content* is the *exact* assembled system message the model saw for
        this turn (base prompt + identity line + RAG block + any first-turn surfaced
        questions). It is logged verbatim because the surfaced-questions block depends
        on the then-current open-question store and cannot be reconstructed later —
        this is what lets a Sleep judge or branch replay see exactly what the speaker
        saw. *input_tokens* is the prompt token count at generation time.
        """
        if self._current_file is None:
            self.start_session()

        # Stamp the session-level user from the first attributed exchange. The
        # header should name whoever actually speaks first — not a value set
        # provisionally at session start (e.g. via clear_context) by an operator
        # who then switched the active user before sending anything. Only the
        # first exchange sets it; later speaker changes are tracked per-exchange.
        # An ``initiated_by`` session is exempt: exchange 0 is Ava's own opener under
        # a stage-direction speaker (e.g. "(initiative)"), so stamping from it would
        # misname the session — the human set at start_session must stand.
        if speaker and not self._session_data["exchanges"] and not self._session_data.get("initiated_by"):
            self._session_data["user"] = speaker

        cot, response = self._parse_cot(assistant_full_response)
        entry: dict = {
            "exchange_id": exchange_id or uuid4().hex,
            "user_prompt": user_prompt,
            "speaker": speaker,
            "assistant_cot": cot,
            "assistant_response": response,
            "system_content": system_content,
            "rag_context": rag_context,
        }
        if input_tokens is not None:
            entry["input_tokens"] = input_tokens
        if tension is not None:
            entry["tension"] = tension
        # First-step probability the model put on opening a thinking block (gemma-4
        # `<|channel>`) — the CoT-decision diagnostic, charted offline across turns.
        if think_open_prob is not None:
            entry["think_open_prob"] = think_open_prob
        if generation_params is not None:
            entry["generation_params"] = generation_params
        self._session_data["exchanges"].append(entry)
        self._save()

    def set_latest_reflection_feedback(
        self,
        exchange_id: str,
        text: str,
        *,
        speaker: str = "",
    ) -> dict:
        """Create or edit feedback on the latest completed exchange only.

        The exchange id is an optimistic fence supplied by the Chat tab. Once a
        newer reply lands, the id no longer names the final exchange and the old
        feedback window is permanently closed. The feedback remains in the raw
        transcript for normal reflection and any later deliberate revisit.
        """
        exchanges = self._session_data.get("exchanges") or []
        if not exchanges:
            raise ValueError("No completed Ava reply is available for feedback.")
        latest = exchanges[-1]
        if not exchange_id or latest.get("exchange_id") != exchange_id:
            raise ValueError("That reply is no longer the latest Ava reply.")
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("Meta feedback cannot be empty.")
        if len(cleaned) > REFLECTION_FEEDBACK_MAX_CHARS:
            raise ValueError(
                f"Meta feedback is limited to {REFLECTION_FEEDBACK_MAX_CHARS} characters."
            )

        now = datetime.now().isoformat()
        prior = latest.get("reflection_feedback") or {}
        if not isinstance(prior, dict):
            prior = {}
        feedback = {
            "id": prior.get("id") or uuid4().hex,
            "speaker": str(speaker or "").strip(),
            "text": cleaned,
            "created_at": prior.get("created_at") or now,
            "updated_at": now,
        }
        latest["reflection_feedback"] = feedback
        self._session_data["schema_version"] = CHAT_SCHEMA_VERSION
        if not self._save():
            raise OSError("Could not save Meta feedback to the active transcript.")
        return dict(feedback)

    # ---------------------------------------------------------------- #
    # Helpers                                                           #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _parse_cot(response: str) -> Tuple[str, str]:
        """
        Split response into (cot, answer).

        Rules:
        - No <think> tag → cot="", answer=full response.
        - <think>…</think> present → cot=inner text, answer=text after tag.
        - <think> without </think> (malformed) → cot="", answer=full response.
        """
        open_tag = response.find("<think>")
        if open_tag == -1:
            return "", response

        close_tag = response.find("</think>", open_tag)
        if close_tag == -1:
            # Malformed: missing closing tag — treat whole response as answer
            return "", response

        cot = response[open_tag + len("<think>") : close_tag]
        answer = response[close_tag + len("</think>") :].strip()
        return cot, answer

    def _save(self) -> bool:
        if self._current_file is None:
            return False
        try:
            with open(self._current_file, "w", encoding="utf-8") as fh:
                json.dump(self._session_data, fh, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False


# ------------------------------------------------------------------- #
# Out-of-band transcript edits                                        #
# ------------------------------------------------------------------- #

def _atomic_write_session(session_path: Path, data: dict) -> None:
    """Atomically write a session dict to *session_path* (temp file + rename).

    Shared by the out-of-band transcript editors (``mark_exchange_corrupt`` /
    ``rewrite_exchange_history``). Raises ``OSError`` on failure.
    """
    import os
    import tempfile
    fd, tmp_name = tempfile.mkstemp(
        prefix=session_path.stem + ".", suffix=".tmp", dir=session_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, session_path)
    except Exception as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise OSError(f"Could not write transcript: {exc}")


def rewrite_exchange_history(
    session_path: Path,
    exchange_index: int,
    *,
    new_cot: str,
    new_response: str,
    run_id: str = "",
    target_kind: str = "",
) -> dict:
    """Bake a human-reviewed target into a PAST transcript, in place (schema v5).

    The Training review "Rewrite history" action replaces one exchange's stored
    ``assistant_cot``/``assistant_response`` with the operator-reviewed content (from
    the locked sidecar target), so the transcript itself — the ground truth RAG
    indexes, snapshots carry, and later exchanges use as context — becomes correct,
    instead of relying on a sidecar override layered over corrupt content. Training is
    unaffected (it already reads the reviewed target off the sidecar); this fixes
    everything that reads the transcript directly.

    Side effects on the exchange:
      * the prior ``assistant_cot``/``assistant_response`` are preserved under an
        append-only ``rewrite_history`` list (forensics + manual recovery);
      * the ``tension`` block is REMOVED — it holds per-token logits (entropy/margin/
        contested ids + the raw ``token_ids`` branch replay seeds from) measured on the
        ORIGINAL generation, so it no longer describes the rewritten text and would make
        branch replay replay stale ids;
      * any ``corrupt_cot``/``corrupt_response`` flags are cleared (the rewrite resolves
        them).

    ``ChatLogger`` is the sole writer of a transcript, so this edit lives here. Meant for
    a *frozen* (already-reflected) session surfaced in the review tab; the caller must
    refuse the live session (a later ``log_exchange`` save would clobber the edit). Atomic
    write. Returns ``{"exchange_index", "cot_changed", "response_changed",
    "tension_removed"}``. Raises ``ValueError`` on a bad index, ``OSError`` on read/write.
    """
    session_path = Path(session_path)
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OSError(f"Could not read transcript: {exc}")
    if not isinstance(data, dict):
        raise ValueError("Transcript is not a valid session file.")
    exchanges = data.get("exchanges")
    if not isinstance(exchanges, list) or not (0 <= exchange_index < len(exchanges)):
        raise ValueError(f"No exchange {exchange_index} in {session_path.name}.")
    entry = exchanges[exchange_index]
    if not isinstance(entry, dict):
        raise ValueError(f"Exchange {exchange_index} is malformed.")

    prev_cot = entry.get("assistant_cot", "") or ""
    prev_response = entry.get("assistant_response", "") or ""
    had_tension = "tension" in entry

    record = {
        "at": datetime.now().isoformat(),
        "run_id": run_id or "",
        "target_kind": target_kind or "",
        "prev_cot": prev_cot,
        "prev_response": prev_response,
        "had_tension": had_tension,
    }
    history = entry.get("rewrite_history")
    if not isinstance(history, list):
        history = []
    history.append(record)
    entry["rewrite_history"] = history

    entry["assistant_cot"] = new_cot
    entry["assistant_response"] = new_response
    # The per-token tension series was measured on the original generation and no longer
    # matches the rewritten text — drop it so nothing (esp. branch replay) reads stale ids.
    entry.pop("tension", None)
    # Resolved by the rewrite; leaving them would make reflection blank/re-derive the target
    # we just baked in.
    entry.pop("corrupt_cot", None)
    entry.pop("corrupt_response", None)

    data["schema_version"] = CHAT_SCHEMA_VERSION
    _atomic_write_session(session_path, data)

    return {
        "exchange_index": exchange_index,
        "cot_changed": prev_cot != new_cot,
        "response_changed": prev_response != new_response,
        "tension_removed": had_tension,
    }


def delete_exchange(session_path: Path, exchange_index: int) -> dict:
    """Remove one exchange from a PAST transcript for good (schema v6).

    The finalize half of a training **ban**. A ban alone keeps the exchange in the
    transcript and only bars it from the corpus, which is the right default: it happened,
    later turns were answered in its light, and RAG may legitimately recall it. But for the
    case the ban exists to serve — a generation that came out malformed — leaving it in
    place means every later exchange keeps reasoning from garbage and RAG keeps it
    retrievable, so "Rewrite history" (which already bakes reviewed targets into ground
    truth) takes the last step and deletes it.

    The removed turn is preserved under a session-level append-only ``deleted_exchanges``
    list — with its own former index, so the shift is reconstructible — minus its ``tension``
    block, whose per-token logit series is both the bulk of an exchange's bytes and
    meaningless once the turn it measured is gone (``had_tension`` records that it existed).

    **This shifts every later exchange down one index.** The transcript is the position
    authority, so every index-keyed store built on it must be moved in the same operation:
    the sidecar's verdict + anchor maps (``ChatSidecar.drop_exchange``) and the ledger's
    fact hosts (``session_ops._renumber_fact_hosts``). ``ChatLogger`` is the sole writer of
    a transcript, so the edit itself lives here; the caller must refuse the live session (a
    later ``log_exchange`` save would resurrect the deleted turn from memory). Atomic write.

    Returns ``{"exchange_index", "remaining", "had_tension"}``. Raises ``ValueError`` on a
    bad index, ``OSError`` on read/write.
    """
    session_path = Path(session_path)
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OSError(f"Could not read transcript: {exc}")
    if not isinstance(data, dict):
        raise ValueError("Transcript is not a valid session file.")
    exchanges = data.get("exchanges")
    if not isinstance(exchanges, list) or not (0 <= exchange_index < len(exchanges)):
        raise ValueError(f"No exchange {exchange_index} in {session_path.name}.")

    removed = exchanges.pop(exchange_index)
    had_tension = isinstance(removed, dict) and "tension" in removed
    if isinstance(removed, dict):
        removed = {k: v for k, v in removed.items() if k != "tension"}

    graveyard = data.get("deleted_exchanges")
    if not isinstance(graveyard, list):
        graveyard = []
    graveyard.append({
        "at": datetime.now().isoformat(),
        "reason": "banned_from_training",
        "exchange_index": exchange_index,
        "had_tension": had_tension,
        "exchange": removed,
    })
    data["deleted_exchanges"] = graveyard

    data["schema_version"] = CHAT_SCHEMA_VERSION
    _atomic_write_session(session_path, data)

    return {"exchange_index": exchange_index,
            "remaining": len(exchanges),
            "had_tension": had_tension}


def mark_exchange_corrupt(
    session_path: Path,
    exchange_index: int,
    *,
    corrupt_cot: Optional[bool] = None,
    corrupt_response: Optional[bool] = None,
) -> dict:
    """Set/clear the corruption flags on one exchange of a PAST transcript.

    The Training review tab flags a stored CoT and/or reply as corrupt (a
    logging/generation bug). The flag lives on the source exchange in the chat JSON
    so it travels with the exchange that originated it: reflection then treats a
    corrupt CoT as missing and insists on a re-derived IDEAL for a corrupt reply,
    and the training build drops the corrupt content unless an IDEAL replaced it
    (see ``dialogue_source`` / ``reflection_runner``).

    ``ChatLogger`` is the sole writer of a transcript, so this edit lives here too.
    It is meant for *frozen* (already-reflected) sessions surfaced in the review tab;
    the caller must refuse the live session (a subsequent ``log_exchange`` save would
    clobber the flag). Each flag is written only when a bool is passed (``None`` leaves
    it untouched), so the two toggles are independent. Atomic write via a temp file.

    Returns ``{"exchange_index", "corrupt_cot", "corrupt_response"}`` (the resulting
    per-exchange state). Raises ``ValueError`` on a bad index and ``OSError`` on a
    failed write.
    """
    session_path = Path(session_path)
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OSError(f"Could not read transcript: {exc}")
    if not isinstance(data, dict):
        raise ValueError("Transcript is not a valid session file.")
    exchanges = data.get("exchanges")
    if not isinstance(exchanges, list) or not (0 <= exchange_index < len(exchanges)):
        raise ValueError(f"No exchange {exchange_index} in {session_path.name}.")

    entry = exchanges[exchange_index]
    if not isinstance(entry, dict):
        raise ValueError(f"Exchange {exchange_index} is malformed.")
    if corrupt_cot is not None:
        if corrupt_cot:
            entry["corrupt_cot"] = True
        else:
            entry.pop("corrupt_cot", None)
    if corrupt_response is not None:
        if corrupt_response:
            entry["corrupt_response"] = True
        else:
            entry.pop("corrupt_response", None)

    data["schema_version"] = CHAT_SCHEMA_VERSION
    _atomic_write_session(session_path, data)

    return {
        "exchange_index": exchange_index,
        "corrupt_cot": bool(entry.get("corrupt_cot")),
        "corrupt_response": bool(entry.get("corrupt_response")),
    }
