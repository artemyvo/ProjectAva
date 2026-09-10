"""WebSocket client connecting the UI to the inference backend server."""
from __future__ import annotations

import json
import queue
import threading
from typing import Generator, Optional, Tuple


class BackendClient:
    """Thread-safe WebSocket connection to the Ava inference server."""

    def __init__(self) -> None:
        self._ws = None
        self._send_lock = threading.Lock()
        # One in-flight request/response (or stream) at a time — concurrent workers
        # otherwise steal each other's replies from the shared recv queue.
        self._rpc_lock = threading.Lock()
        self._recv_queue: queue.Queue = queue.Queue()
        # Server-pushed reflection progress events (unsolicited, type
        # "reflection_run_event") land here instead of _recv_queue so a lightweight
        # listener can render them the instant they arrive. The Sleep tab's poll RPC
        # is the reliable backstop, but it's starved during the non-streaming branch
        # fork generation, so live pushes are what make branch candidates stream.
        self._reflection_events_q: queue.Queue = queue.Queue()
        self._recv_thread: Optional[threading.Thread] = None
        self.is_loaded: bool = False
        self.last_status: dict = {}
        # URL of the last connect() — used to derive the watchdog HTTP management
        # endpoint (same host, mgmt port) for polling train progress while the
        # inference WebSocket server is stopped for training.
        self.server_url: Optional[str] = None
        self.mgmt_port: int = 8766
        # The inference HTTP sidecar (same host) serves the read-mostly, project-
        # specific endpoints that used to live on the watchdog: /precision plus the
        # artifacts/export/chats bundles (those streams are fetched by the widgets'
        # own workers, which derive this port too).
        self.sidecar_port: int = 8767
        # WebSocket close code from the last drop (4000 = superseded by another
        # client). None until a close is observed; reset on each connect.
        self.last_close_code: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Connection management                                                #
    # ------------------------------------------------------------------ #

    def connect(self, url: str) -> None:
        from websockets.sync.client import connect as ws_connect
        # max_size=None disables the 1 MB per-message frame cap; session_data
        # replies carry full transcripts that already exceed it (~2 MB seen).
        # Must mirror the server's serve(max_size=None) — both sides cap recv.
        #
        # ping_timeout=None mirrors the server too: during a reflection run the
        # server does long GPU passes that starve its asyncio loop, so it can be
        # slow to pong. Without this the client's own 20s keepalive would tear
        # down the link mid-run from this side — the same freeze, just initiated
        # locally instead of by the server.
        self._ws = ws_connect(url, max_size=None, ping_timeout=None)
        self.server_url = url
        self.last_close_code = None  # fresh link — clear any prior supersede/drop code
        self._recv_queue = queue.Queue()  # discard stale messages from any prior connection
        self._reflection_events_q = queue.Queue()  # same, for live reflection pushes
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def is_connected(self) -> bool:
        return self._ws is not None

    # ------------------------------------------------------------------ #
    # Background receiver thread                                           #
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        try:
            while self._ws is not None:
                raw = self._ws.recv()
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "loaded":
                    self.is_loaded = True
                    self.last_status.update({
                        "memory": msg.get("memory", self.last_status.get("memory", "")),
                        "model_id": msg.get("model_id", ""),
                        "adapter_id": msg.get("adapter_id"),
                    })
                elif t == "unloaded":
                    self.is_loaded = False
                elif t == "status":
                    self.last_status.update(msg)
                    self.is_loaded = bool(msg.get("loaded", False))
                elif t == "chunk":
                    # Live generation speed rides each streamed chunk; fold it into
                    # last_status so the main-window status line shows tok/s while the
                    # reply is still streaming (the reader thread sees chunks first).
                    tps = msg.get("tokens_per_sec")
                    if tps is not None:
                        self.last_status["tokens_per_sec"] = tps
                elif t == "token_stats":
                    # The server pushes fresh token-economy counters after each live
                    # turn (and answers get_token_stats RPCs). Fold into last_status
                    # so the status-bar imprint meter stays live without polling.
                    econ = msg.get("token_economy")
                    if econ is not None:
                        self.last_status["token_economy"] = econ
                elif t == "reflection_run_status":
                    # The Sleep tab polls run status every 2 s; the server now
                    # carries fresh VRAM on it. Fold that into last_status so the
                    # main-window memory readout stays live during a reflection
                    # run (no `done`/`status` messages flow then).
                    mem = msg.get("memory")
                    if mem:
                        self.last_status["memory"] = mem
                elif t == "done":
                    self.last_status.update({
                        "exchange_id": msg.get("exchange_id"),
                        "input_tokens": msg.get("input_tokens", self.last_status.get("input_tokens", 0)),
                        "memory": msg.get("memory", self.last_status.get("memory", "")),
                        # Tension is set fresh on every done (None if not captured) so the
                        # chat-tab chip can't accidentally show stale numbers from a prior reply.
                        "tension": msg.get("tension"),
                        # Per-token decoded [text, margin] spans for live coloring of the
                        # reply (CoT + answer). Render-only; reset each done so a reply with
                        # no capture clears any prior coloring.
                        "tension_spans": msg.get("tension_spans"),
                        # First-step probability the model put on opening a thinking block
                        # (gemma-4 `<|channel>`) — drives the "Thinking: NN%" header. Reset
                        # each done so a reply with no capture shows no stale diagnostic.
                        "think_open_prob": msg.get("think_open_prob"),
                        # Finalized generation speed (tok/s) for the status-line readout.
                        "tokens_per_sec": msg.get("tokens_per_sec"),
                    })
                if t == "reflection_run_event":
                    # Live, unsolicited reflection progress push. No _recv_until ever
                    # waits on this type (it would just be discarded from _recv_queue),
                    # so route it to its own queue for the Sleep tab's live listener —
                    # this is the only delivery that keeps up during branch generation,
                    # when the poll RPC is GIL-starved. The listener dedups by seq
                    # against the poll backstop.
                    self._reflection_events_q.put(msg)
                else:
                    self._recv_queue.put(msg)
        except Exception as e:
            # Capture the WebSocket close code if the server initiated a clean
            # close. Code 4000 means another client superseded this one (handoff);
            # callers use it to detach instead of auto-reconnecting into a war.
            self.last_close_code = getattr(getattr(e, "rcvd", None), "code", None)
            self._recv_queue.put(
                {"type": "connection_error", "message": str(e), "code": self.last_close_code}
            )
            self._ws = None
            self.is_loaded = False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _send(self, msg: dict) -> None:
        with self._send_lock:
            if self._ws is None:
                raise ConnectionError("Not connected to inference server.")
            self._ws.send(json.dumps(msg))

    def _recv_until(self, expected: set, timeout: float = 120.0) -> dict:
        """Block until a message of one of the expected types arrives (or an error)."""
        while True:
            try:
                msg = self._recv_queue.get(timeout=timeout)
            except queue.Empty:
                return {"type": "error", "message": "Timed out waiting for server response."}
            if msg["type"] in expected or msg["type"] in ("error", "connection_error"):
                return msg

    def drain_reflection_events(self, timeout: float = 1.0) -> list:
        """Return server-pushed reflection events, blocking up to *timeout* for the
        first one then draining the rest without blocking.

        These are the live ``reflection_run_event`` pushes (see ``_recv_loop``). A
        Sleep-tab listener calls this in a loop so branch candidates render as they
        arrive, independent of the poll RPC. Never touches the socket or the RPC
        lock, so it can't be starved by an in-flight poll during GPU work."""
        events: list = []
        try:
            events.append(self._reflection_events_q.get(timeout=timeout))
        except queue.Empty:
            return events
        while True:
            try:
                events.append(self._reflection_events_q.get_nowait())
            except queue.Empty:
                break
        return events

    # ------------------------------------------------------------------ #
    # Operations                                                           #
    # ------------------------------------------------------------------ #

    def load_model(self, model_id: str, context_length: int, adapter_id: Optional[str] = None) -> dict:
        with self._rpc_lock:
            self._send({"type": "load", "model_id": model_id, "context_length": context_length, "adapter_id": adapter_id})
            return self._recv_until({"loaded"}, timeout=600.0)

    def unload_model(self) -> dict:
        with self._rpc_lock:
            self._send({"type": "unload"})
            return self._recv_until({"unloaded"}, timeout=300.0)

    def request_status(self) -> dict:
        with self._rpc_lock:
            self._send({"type": "status"})
            return self._recv_until({"status"})

    def clear_context(self, user: str = "") -> dict:
        with self._rpc_lock:
            self._send({"type": "clear_context", "user": user})
            return self._recv_until({"context_cleared"})

    def set_session_notes(self, notes: str) -> dict:
        with self._rpc_lock:
            self._send({"type": "set_session_notes", "notes": notes})
            return self._recv_until({"session_notes_saved"})

    def set_reflection_feedback(
        self, exchange_id: str, text: str, speaker: str = ""
    ) -> dict:
        with self._rpc_lock:
            self._send({
                "type": "set_reflection_feedback",
                "exchange_id": exchange_id,
                "text": text,
                "speaker": speaker,
            })
            return self._recv_until({"reflection_feedback_saved"})

    def retry_last_exchange(self) -> dict:
        """Roll the latest completed exchange off the active session so it can be redone.

        The Chat-tab "Retry" action for a collapsed/degenerate reply: the server drops the
        last exchange (and its user turn) from the active transcript + in-memory
        conversation so it never reaches reflection/training, and returns the user prompt
        so the operator can resend it under adjusted sampling. The completed-turn sibling
        of ``cancel(discard=True)`` (which rolls back a reply still streaming). Returns
        ``retry_ready {user_prompt, speaker, exchange_index}`` or an error (reflection
        active / already reflected / nothing to retry / synthetic opener)."""
        with self._rpc_lock:
            self._send({"type": "retry_last_exchange"})
            return self._recv_until({"retry_ready"})

    def list_sessions(self) -> dict:
        with self._rpc_lock:
            self._send({"type": "list_sessions"})
            return self._recv_until({"sessions_list"})

    def get_session(self, filename: str) -> dict:
        with self._rpc_lock:
            self._send({"type": "get_session", "filename": filename})
            return self._recv_until({"session_data"})

    def load_session(self, filename: str, in_place: bool = False) -> dict:
        """Adopt a past session as the active conversation.

        Normally forks a fresh session file (``continued_from``); with *in_place* the
        server resumes the SAME file so replies append to it (used to answer an
        Ava-initiated outreach in its own transcript)."""
        with self._rpc_lock:
            self._send({"type": "load_session", "filename": filename, "in_place": in_place})
            return self._recv_until({"session_loaded"})

    def delete_session(self, filename: str) -> dict:
        """Delete a past chat transcript (+ its sidecar) from the server."""
        with self._rpc_lock:
            self._send({"type": "delete_session", "filename": filename})
            return self._recv_until({"session_deleted"})

    def reset_session_reflection(self, filename: str) -> dict:
        """Delete a past chat's sidecar so the chat re-reflects from scratch.

        Drops the reflect-once freeze along with everything reflection derived for
        that chat (verdicts, trainable targets — human-``locked`` ones included —
        the consolidation summary and the retrieval anchors); the transcript is
        untouched. Chat tab's "Re-reflect chat"."""
        with self._rpc_lock:
            self._send({"type": "reset_session_reflection", "filename": filename})
            return self._recv_until({"session_reflection_reset"})

    def mark_exchange_corrupt(
        self,
        filename: str,
        exchange_index: int,
        *,
        corrupt_cot: Optional[bool] = None,
        corrupt_response: Optional[bool] = None,
    ) -> dict:
        """Flag a past exchange's stored CoT and/or reply as corrupt on the server.

        The flag lands on the source exchange in the chat JSON so reflection treats a
        corrupt CoT as missing / insists on a re-derived IDEAL for a corrupt reply, and
        the next training build drops the corrupt content. Each flag is only sent when a
        bool is passed (``None`` leaves the server-side value untouched), so the two
        toggles are independent. Returns ``exchange_corrupt_marked`` with the new state."""
        msg: dict = {
            "type": "mark_corrupt",
            "filename": filename,
            "exchange_index": int(exchange_index),
        }
        if corrupt_cot is not None:
            msg["corrupt_cot"] = bool(corrupt_cot)
        if corrupt_response is not None:
            msg["corrupt_response"] = bool(corrupt_response)
        with self._rpc_lock:
            self._send(msg)
            return self._recv_until({"exchange_corrupt_marked"})

    def stream_regenerate_exchange(
        self, filename: str, exchange_index: int, *, temperature: float = 0.9,
        system_suffix: str = "", cot_only: bool = False, adapter: str = "",
    ) -> Generator[Tuple[str, dict], None, None]:
        """Re-answer a past exchange with the currently loaded adapter (Training review).

        ``adapter`` (a bare directory name under the server's ``models/``, from the
        review payload's adapter lineage) has a DIFFERENT adapter answer instead — the
        server swaps it in STICKY: one full model reload the first time, after which it
        stays loaded (further regenerations under the same choice swap nothing, and the
        box keeps running that adapter until another load). ``("status", {"text": …})``
        items narrate the reload when one happens; empty = whatever is loaded.

        ``system_suffix`` is an optional operator delivery constraint that shapes only this
        regeneration (appended to the system prompt), not the trainable prefix. ``cot_only``
        (Corrupt CoT checked without Corrupt reply) halts generation once the reasoning
        channel closes, since Apply grafts only the new ``<think>`` onto the kept reply. The
        thought opener is always prefilled so the re-answer carries a CoT. Writes nothing —
        the separate ``apply_regenerated_exchange`` persists a reviewed result.

        Yields ``("chunk", {"cot": str, "reply": str})`` deltas as the re-answer is produced
        (the server classifies them, so the family's raw markers never reach the UI), then
        exactly one ``("done", exchange_regenerated)`` or ``("error", {...})``. The terminal
        payload is authoritative — it comes from the cleaned text, which the streamed
        approximation may differ from at the tail.

        Stop an in-flight regeneration with :meth:`cancel`; it is sent outside the RPC lock,
        so it reaches the server while this generator still holds it. The server then returns
        the partial as a normal ``done`` carrying ``cancelled: true``."""
        with self._rpc_lock:
            self._send({
                "type": "regenerate_exchange",
                "filename": filename,
                "exchange_index": int(exchange_index),
                "temperature": float(temperature),
                "system_suffix": str(system_suffix),
                "cot_only": bool(cot_only),
                "adapter": str(adapter or ""),
            })
            try:
                while True:
                    msg = self._recv_until(
                        {"regenerate_chunk", "regenerate_status",
                         "exchange_regenerated"}, timeout=600.0)
                    t = msg.get("type")
                    if t == "regenerate_chunk":
                        yield "chunk", {"cot": msg.get("cot", ""),
                                        "reply": msg.get("reply", "")}
                    elif t == "regenerate_status":
                        yield "status", {"text": msg.get("text", "")}
                    elif t == "exchange_regenerated":
                        yield "done", msg
                        return
                    else:   # error / connection_error / timeout
                        yield "error", msg
                        return
            except GeneratorExit:
                # The consumer walked away mid-generation (dialog closed): stop the GPU
                # work rather than leaving it to run out the token budget unread.
                self.cancel()
                raise

    def apply_regenerated_exchange(
        self, filename: str, exchange_index: int, *,
        corrupt_cot: bool, corrupt_response: bool, new_cot: str, new_reply: str,
        original_reply: str = "",
    ) -> dict:
        """Persist a reviewed regeneration to the chat sidecar (Training review Apply).

        Writes a fresh trainable target for the exchange (a regenerated reply carries its
        new CoT; a CoT-only regen grafts the new thought onto the *kept* reply), locks the
        exchange, and clears its corrupt flags. ``original_reply`` is the reply to keep for
        a CoT-only regen — the answer the tab is displaying (the current trained target),
        which the server prefers over the raw transcript reply. Returns
        ``exchange_regenerated_applied``."""
        with self._rpc_lock:
            self._send({
                "type": "apply_regenerated_exchange",
                "filename": filename,
                "exchange_index": int(exchange_index),
                "corrupt_cot": bool(corrupt_cot),
                "corrupt_response": bool(corrupt_response),
                "new_cot": new_cot,
                "new_reply": new_reply,
                "original_reply": original_reply,
            })
            return self._recv_until({"exchange_regenerated_applied"})

    def set_training_ban(self, target: str, exchange_index: Optional[int],
                         banned: bool) -> dict:
        """Ban (or un-ban) one Training review row from future training.

        *target* is the row's ``source_session``: a chat transcript filename (with its
        ``exchange_index``) or ``wander:<ts>`` for a wander capture (index ``None``). The
        flag lands on the chat sidecar / the wander corpus record, and the build drops the
        row from then on; the source itself survives until "Rewrite history" deletes it, so
        the ban is reversible until that point. Returns ``training_ban_set``."""
        with self._rpc_lock:
            self._send({
                "type": "set_training_ban",
                "target": target,
                "exchange_index": (None if exchange_index is None else int(exchange_index)),
                "banned": bool(banned),
            })
            return self._recv_until({"training_ban_set"})

    def rewrite_history(self) -> dict:
        """Bake every human-reviewed (locked) target into its transcript and unfreeze it.

        Training review "Rewrite history": for every locked exchange in hot/chats, rewrite the
        stored CoT/reply with the reviewed sidecar target (making the transcript correct),
        drop that exchange's now-stale tension block, clear its corrupt flags, and remove the
        per-exchange lock; then re-embed chat RAG. Training output is unchanged (it already
        reads the reviewed target off the sidecar); what changes is the transcript-derived data
        (RAG recall, snapshots, later-exchange context). In the same pass every **banned**
        exchange is DELETED from its transcript (later exchanges shift down a position, and
        the sidecar/ledger stores keyed by position move with them) and every banned wander
        capture is dropped from the corpus. Returns ``history_rewritten {rewritten, deleted,
        wander_deleted, chats, skipped, errors}``. May re-embed the whole chat corpus, so a
        wide timeout."""
        with self._rpc_lock:
            self._send({"type": "rewrite_history"})
            return self._recv_until({"history_rewritten"}, timeout=600.0)

    def get_rag_artifacts(self) -> dict:
        """Fetch all live reflection RAG artifacts (fact/persona/ask) for the Debug tab."""
        with self._rpc_lock:
            self._send({"type": "get_rag_artifacts"})
            return self._recv_until({"rag_artifacts"})

    def match_anchors(self, text: str, limit: int = 20) -> dict:
        """Preview which stored exchange anchors *text* would match (Chat tab strip).

        Diagnostic only — the server retrieves and injects nothing. Matching is lexical,
        so this needs no model loaded and is cheap enough to call on a debounce while the
        operator is still typing.
        """
        with self._rpc_lock:
            self._send({"type": "match_anchors", "text": text, "limit": limit})
            return self._recv_until({"anchor_matches"})

    def update_persona(self, baseline_keys: list[str], retained_keys: list[str]) -> dict:
        """Evict removed live persona entries on the connected server.

        ``baseline_keys`` provides an optimistic-concurrency fence: if reflection changed
        the server's persona set since the tab fetched it, the server rejects the upload
        instead of applying a stale editor view.
        """
        with self._rpc_lock:
            self._send({
                "type": "update_persona",
                "baseline_keys": baseline_keys,
                "retained_keys": retained_keys,
            })
            return self._recv_until({"persona_updated"})

    def update_facts(self, baseline_keys: list[str], retained_keys: list[str]) -> dict:
        """Evict removed live ``[fact]`` entries on the connected server.

        The fact counterpart of :meth:`update_persona`, with the same optimistic-
        concurrency fence: ``baseline_keys`` is the live fact set as the tab fetched it,
        so a reflection that landed since then makes the server reject the upload rather
        than apply a stale editor view.
        """
        with self._rpc_lock:
            self._send({
                "type": "update_facts",
                "baseline_keys": baseline_keys,
                "retained_keys": retained_keys,
            })
            return self._recv_until({"facts_updated"})

    def dedup_facts(self, dry_run: bool = True):
        """Semantic de-dup of live [fact] RAG items on the clean base (Sleep/Debug tabs).

        Generator yielding ``(kind, msg)`` so a caller can render the blocked clean-base
        grouping **in real time** (a large store is dozens of sequential calls — the pass
        ran silently for tens of minutes before this, which is indistinguishable from a
        hang):
          * ``("dedup_stage", {stage:"clustered", facts, subjects, multi, singletons,
            candidates})`` — the subject blocking, emitted once before any grouping call.
          * ``("dedup_stage", {stage:"block", i, n, items, groups, merged, rejected})``
            — one per grouping block as it completes.
          * ``("facts_deduped", {before, after?, groups, evicted, dry_run | skipped, ...})``
            — terminal; ``dry_run`` previews the merges, else they were applied.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        With ``dry_run`` (default) nothing is written; otherwise the duplicates are evicted
        to one survivor (which inherits the union of the group's triggers) and RAG reloads.
        One logical RPC (holds ``_rpc_lock``); run on a worker thread. Very wide per-message
        timeout: the clean-base swap is two full model reloads."""
        with self._rpc_lock:
            self._send({"type": "dedup_facts", "dry_run": bool(dry_run)})
            while True:
                msg = self._recv_until({"dedup_stage", "facts_deduped"}, timeout=5400.0)
                t = msg.get("type")
                yield t, msg
                if t in ("facts_deduped", "error", "connection_error"):
                    return

    def reconcile_self(self, dry_run: bool = True):
        """Reconcile live [persona]/[fact] items against the current persona digest (Sleep tab).

        Generator yielding ``(kind, msg)`` so the Sleep tab can render the batched
        clean-base judgment **in real time** (the set is thousands of items → dozens of
        sequential calls):
          * ``("reconcile_stage", {stage:"start", items, n_persona, n_fact, batches, ...})``
            — the plan, emitted once before judging begins.
          * ``("reconcile_stage", {stage:"batch", i, n, n_items, n_supersede, running_total})``
            — one per batch as it completes.
          * ``("self_reconciled", {before, after?, report, superseded, dry_run | skipped, ...})``
            — terminal; ``dry_run`` previews what she'd set aside, else it was applied.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        With ``dry_run`` (default) nothing is written; otherwise it softens (supersedes) the
        flagged items — dropped from recall + active evidence/training but kept as
        evidence-of-change, reversibly — and reloads RAG. One logical RPC (holds
        ``_rpc_lock``); run on a worker thread. Very wide per-message timeout since a single
        batch is a full clean-base generation."""
        with self._rpc_lock:
            self._send({"type": "reconcile_self", "dry_run": bool(dry_run)})
            while True:
                msg = self._recv_until({"reconcile_stage", "self_reconciled"},
                                       timeout=5400.0)
                t = msg.get("type")
                yield t, msg
                if t in ("self_reconciled", "error", "connection_error"):
                    return

    def digest_dryrun(self, block_size=None, temperature=None):
        """DRY RUN of the persona digest — cluster, synthesize, show, write nothing
        (Sleep tab "Persona digest (dry)").

        Generator yielding ``(kind, msg)`` so the Sleep tab can watch a long two-model
        pass: the live [persona] evidence is clustered into themes on the CLEAN base
        (adapter off — grouping paraphrases is an evaluation) using the map-reduce pass
        that survives a large corpus, then the self-portrait is synthesized on the
        ADAPTER (authorship stays with Ava).
          * ``("digest_dryrun_stage", {stage, ...})`` — phase + per-block progress
            (gathered / clean_base_enter / map / reduce_started / reduce / clustered /
            synthesizing).
          * ``("digest_dryrun_chunk", {text})`` — synthesis deltas (``<think>`` included).
          * ``("digest_dryrun_done", {items, stats, themes, digest, rendered, current |
            skipped, message?})`` — terminal.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        Nothing is persisted — no digest snapshot, no ``current`` pointer move, no RAG
        refresh — so there is nothing to Apply. One logical RPC (holds ``_rpc_lock``); run
        on a worker thread. Very wide per-message timeout: the clean-base swap is two full
        model reloads and the clustering is many sequential calls."""
        with self._rpc_lock:
            req = {"type": "digest_dryrun"}
            if block_size is not None:
                req["block_size"] = int(block_size)
            if temperature is not None:
                req["temperature"] = float(temperature)
            self._send(req)
            while True:
                msg = self._recv_until(
                    {"digest_dryrun_stage", "digest_dryrun_chunk", "digest_dryrun_done"},
                    timeout=5400.0)
                t = msg.get("type")
                yield t, msg
                if t in ("digest_dryrun_done", "error", "connection_error"):
                    return

    def regen_persona(self, block_size=None, temperature=None):
        """Regenerate the persona digest AND activate it (Persona tab "Regen persona").

        The WRITE counterpart of :meth:`digest_dryrun`: the same cluster-on-the-CLEAN-base
        → synthesize-on-the-ADAPTER pass, but the digest is written to the live persona
        store exactly as a Sleep run writes it, and the run's Ava version is then minted
        train-lessly (``produce_persona(activate=True)`` — the same call the Sleep run's
        ``persona`` stage makes) so the refreshed self-portrait reaches the next live
        chat turn. The adapter is unchanged; the previous persona snapshot stays on disk
        as the rollback path.

        Generator yielding ``(kind, msg)``:
          * ``("regen_persona_stage", {stage, ...})`` — phase + per-block progress
            (gathered / clean_base_enter / map / reduce_started / reduce / clustered /
            synthesizing / digest_written / producing_persona).
          * ``("regen_persona_chunk", {text})`` — synthesis deltas (``<think>`` included).
          * ``("regen_persona_done", {run_id, items, themes, counts, digest, rendered,
            activated, persona_path | skipped, message?})`` — terminal. ``activated``
            false with ``persona_error`` means the digest was written but the previous
            persona stayed active (the snapshot step failed).
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``); run on a worker thread. Very wide
        per-message timeout: the clean-base swap is two full model reloads, and the
        persona snapshot at the end is a full corpus copy."""
        with self._rpc_lock:
            req = {"type": "regen_persona"}
            if block_size is not None:
                req["block_size"] = int(block_size)
            if temperature is not None:
                req["temperature"] = float(temperature)
            self._send(req)
            while True:
                msg = self._recv_until(
                    {"regen_persona_stage", "regen_persona_chunk", "regen_persona_done"},
                    timeout=5400.0)
                t = msg.get("type")
                yield t, msg
                if t in ("regen_persona_done", "error", "connection_error"):
                    return

    def resolve_contradictions(self, dry_run: bool = True):
        """Resolve contradictory live [fact] items (Sleep tab "Resolve fact conflicts").

        Generator yielding ``(kind, msg)`` so the Sleep tab can render the per-subject
        contradiction check live:
          * ``("contradict_stage", {stage:"start", facts, groups, ...})`` — the plan.
          * ``("contradict_stage", {stage:"group", i, n, group_size, n_superseded, running_total})``
            — one per subject group as it is judged on the clean base.
          * ``("contradictions_resolved", {before, after?, report, superseded | skipped, ...})``
            — terminal; ``dry_run`` previews, else the stale facts were softened.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        Keeps the NEWEST fact in each conflict set (the correction) and supersedes the
        older ones — dropped from recall + active evidence/training but kept as
        evidence-of-change, reversibly. One logical RPC (holds ``_rpc_lock``); run on a
        worker thread. Wide per-message timeout (dozens of sequential clean-base calls)."""
        with self._rpc_lock:
            self._send({"type": "resolve_contradictions", "dry_run": bool(dry_run)})
            while True:
                msg = self._recv_until({"contradict_stage", "contradictions_resolved"},
                                       timeout=5400.0)
                t = msg.get("type")
                yield t, msg
                if t in ("contradictions_resolved", "error", "connection_error"):
                    return

    def get_wander_log(self) -> dict:
        """Fetch the log of pages Ava wandered into (title + url) for the Debug tab."""
        with self._rpc_lock:
            self._send({"type": "get_wander_log"})
            return self._recv_until({"wander_log"})

    def get_worklog(self, after_seq: int = 0) -> dict:
        """Fetch first-person episodic worklog entries with id > after_seq (0 = everything).

        Backs the Worklog preview tab: Ava's durable, semantic record of what she did
        (one entry per meaningful episode, in her own voice), distinct from the activity
        journal. Poll with the last-seen id to stream new entries during a live run."""
        with self._rpc_lock:
            self._send({"type": "get_worklog", "after_seq": int(after_seq)})
            return self._recv_until({"worklog_batch"})

    def get_prompt_deltas(self) -> dict:
        """Fetch the logged-only prompt-mutation proposals (standing-prompt deltas) for the Debug tab."""
        with self._rpc_lock:
            self._send({"type": "get_prompt_deltas"})
            return self._recv_until({"prompt_deltas"})

    def get_token_stats(self) -> dict:
        """Fetch the running user-produced token count for the Debug tab."""
        with self._rpc_lock:
            self._send({"type": "get_token_stats"})
            return self._recv_until({"token_stats"})

    def til_fetch(self, date: str = "") -> dict:
        """Ask the server to fetch a day's Wikipedia Current events digest into server/til.

        Manual debug/research hook. The date defaults to yesterday server-side
        (today's portal page isn't populated yet); pass ``YYYY-MM-DD`` to override.
        Returns a ``til_fetched`` summary + preview. (Fetch only — no reflection;
        use :meth:`til_learn` to also run the dry-run learning pass.)
        """
        with self._rpc_lock:
            payload: dict = {"type": "til_fetch"}
            if date:
                payload["date"] = date
            self._send(payload)
            # Network fetch + retry/backoff can take a few seconds; allow generous slack.
            return self._recv_until({"til_fetched"}, timeout=60.0)

    def til_learn(self, date: str = "", reflect: bool = True):
        """Fetch a TIL digest and stream a DRY-RUN learning reflection over it.

        Generator yielding ``(kind, msg)`` for the Sleep tab's "Learn" button:
          * ``("til_fetched", {...})`` — the fetched digest summary + preview.
          * ``("til_reflect_chunk", {text})`` — streamed reflection deltas.
          * ``("til_reflect_done", {text, report, skipped?})`` — terminal: the
            cleaned reflection and the parsed ``{weights, rag, resolved}`` modifiers
            the run *would* route (``skipped`` set if no model / failure).
          * ``("error"/"connection_error", {...})`` — terminal failure.

        The whole sequence is one logical RPC (holds ``_rpc_lock`` throughout), so
        it must run on a worker thread, never the GUI thread. Writes nothing.
        """
        with self._rpc_lock:
            payload: dict = {"type": "til_fetch", "reflect": bool(reflect)}
            if date:
                payload["date"] = date
            self._send(payload)
            while True:
                msg = self._recv_until(
                    {"til_fetched", "til_reflect_chunk", "til_reflect_done"},
                    timeout=300.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("til_reflect_done", "error", "connection_error"):
                    return
                if t == "til_fetched" and not reflect:
                    return

    def til_lookup(self):
        """Resolve open [ask:search] questions by looking them up on Wikipedia.

        Generator yielding ``(kind, msg)`` for the Sleep tab's "Resolve Questions"
        button — the lookup loop end to end:
          * ``("til_lookup_collected", {count, questions})`` — open search asks gathered.
          * ``("til_lookup_log", {message})`` — clean-base swap / progress notes.
          * ``("til_lookup_subjects", {subjects})`` — subjects the clean base extracted.
          * ``("til_lookup_fetched", {subject, title?, via?, missing?, ...})`` — per subject.
          * ``("til_reflect_chunk", {text})`` — streamed learning-pass deltas.
          * ``("til_reflect_done", {text, report, skipped?})`` — terminal; same shape as
            Learn, so the result feeds :meth:`til_apply` unchanged.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock`` throughout) — run on a worker thread.
        The clean-base swap (two model loads) plus per-subject fetches make this
        slow; the generous recv timeout reflects that."""
        with self._rpc_lock:
            self._send({"type": "til_lookup"})
            while True:
                msg = self._recv_until(
                    {"til_lookup_collected", "til_lookup_log", "til_lookup_subjects",
                     "til_lookup_fetched", "til_reflect_chunk", "til_reflect_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("til_reflect_done", "error", "connection_error"):
                    return

    def til_wander(self, lang: str = "", source: str = "", url: str = ""):
        """Reflect on a random page, or a user-supplied page URL.

        Generator yielding ``(kind, msg)`` for the Sleep tab's "Wander"/"Visit" controls:
          * ``("til_wandered", {wiki, lang, title, url, chars})`` — the page picked/fetched.
          * ``("til_reflect_chunk", {text})`` — streamed learning-pass deltas.
          * ``("til_reflect_done", {text, report, skipped?})`` — terminal; same shape as
            Learn, so the result feeds :meth:`til_apply` unchanged.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        ``lang``/``source`` optionally narrow the source pick (else a random enabled
        one). ``url`` bypasses the random pick and asks the server to fetch that page
        directly. One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the
        network fetch + learning pass make it slow, hence the generous timeout."""
        with self._rpc_lock:
            payload: dict = {"type": "til_wander"}
            if lang:
                payload["lang"] = lang
            if source:
                payload["source"] = source
            if url:
                payload["url"] = url
            self._send(payload)
            while True:
                msg = self._recv_until(
                    {"til_wandered", "til_reflect_chunk", "til_reflect_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("til_reflect_done", "error", "connection_error"):
                    return

    def til_apply(self, text: str, date: str = "", kind: str = "") -> dict:
        """Persist a completed dry-run learning pass.

        For Learn/Lookup (default), ``text`` is the cleaned reflection produced by the
        terminal ``til_reflect_done``; the server re-parses it and routes the same WEIGHTS
        / RAG / RESOLVED modifiers it previewed into RAG memory + the weights store + the
        ledger, then refreshes the reflection index.

        With ``kind="wander"`` the server instead commits the most-recent wander exchange
        to the SFT *learning dataset* (one one-shot Ambient-Enculturation example) — it
        does not write RAG/weights memory, and ``text`` is unused (the server holds the
        stashed exchange). Returns ``til_applied`` ``{counts, source, skipped?}``.
        Blocking — run off the GUI thread.
        """
        with self._rpc_lock:
            payload: dict = {"type": "til_apply", "text": text}
            if date:
                payload["date"] = date
            if kind:
                payload["kind"] = kind
            self._send(payload)
            # 600 s, not 120. Apply is GPU-free by design — it persists material the
            # preview already generated — but it still parses/writes the modifiers and
            # rebuilds the reflection index, which on a large store is not instant. The
            # old 120 s dated from when Apply also GENERATED the protocol and recap, at
            # which point no timeout could have been right: the protocol alone runs to
            # ~21 minutes, so every manual Apply reported failure while the server
            # finished anyway, and a retry inside that window double-wrote the wander
            # capture. That generation moved out; this is headroom for what is left.
            return self._recv_until({"til_applied"}, timeout=600.0)

    def outreach_now(self):
        """Manually trigger one Ava-initiated outreach decision (Sleep-tab debug).

        Generator yielding ``(kind, msg)`` for the Sleep tab's "Reach Out" button —
        the same decision pass the idle-wake heartbeat runs, with Ava's reasoning
        streamed back so the operator can watch her deliberate:
          * ``("outreach_question", {question, ask_kind, user})`` — the open ask she's weighing.
          * ``("outreach_prompt", {pass, segments, input_tokens, rag_tokens, rag_disabled,
            max_new_tokens, context_length})`` — the FULL model-facing prompt of the
            decision pass, in the same labelled ``{kind, label, text}`` segments the Chat
            tab's Debug checkbox renders. This is the reach-out job that actually
            retrieves, so its ``rag`` segment is the injected memory block.
          * ``("outreach_chunk", {text})`` — streamed reasoning deltas (``<think>`` included).
          * ``("outreach_done", {composed?, decision?, session?, opener? | skipped, message?})``
            — terminal; a yes wrote the reversed outreach session into the chats list.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the decision
        pass is a full generation, hence the generous timeout."""
        with self._rpc_lock:
            self._send({"type": "outreach_now"})
            while True:
                msg = self._recv_until(
                    {"outreach_question", "outreach_prompt", "outreach_chunk",
                     "outreach_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("outreach_done", "error", "connection_error"):
                    return

    def synthesis_now(self):
        """Manually trigger one synthesis pass (Sleep-tab "Chat reach out", debug).

        Generator yielding ``(kind, msg)`` — the same pass the idle-wake heartbeat runs:
        Ava re-reads an aged chat and asks what she now wonders, streaming her reasoning
        back so the operator can watch:
          * ``("synthesis_stage", {stage, ...})`` — phase markers (picked / analyzing /
            no_question / composing).
          * ``("synthesis_chunk", {text})`` — streamed reasoning deltas (``<think>`` included).
          * ``("synthesis_done", {composed?, raised?, session?, question?, opener?, about?,
            asks? | skipped, message?})`` — terminal; a raised question wrote a reversed
            session into the chats list and the rest landed in the question pool.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the pass is up to
        two full generations, hence the generous timeout."""
        with self._rpc_lock:
            self._send({"type": "synthesis_now"})
            while True:
                msg = self._recv_until(
                    {"synthesis_stage", "synthesis_chunk", "synthesis_done"},
                    timeout=900.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("synthesis_done", "error", "connection_error"):
                    return

    def checkin_now(self, sim_hours=None, user=None):
        """Manually simulate one check-in decision (Sleep-tab "Check In", debug).

        Generator yielding ``(kind, msg)`` — the same per-person pass the idle-wake sweep
        runs, but with the silence period **forged** so the operator can watch Ava decide as
        if the threshold had been crossed. ``sim_hours`` overrides the forged elapsed hours
        (the server defaults it to the configured threshold, clamped up to the real
        silence). ``user`` names whose check-in it is — their silence clock, their recent
        conversations, their opener; the server defaults it to the active session's speaker.
        The autonomous job decides for every person separately, so this runs ONE of those
        decisions rather than the whole sweep, whose several deliberations would interleave
        into one unreadable stream. Ava's reasoning is streamed back:
          * ``("checkin_stage", {stage, …})`` — phase markers: ``considering``
            (``{hours, chats, user, simulated}`` — the window she's weighing),
            ``summarizing`` (``{i, n, session}`` — one chat being recapped),
            ``recapped`` (``{i, n, session, when, user, recap, cached, truncated}`` — that
            chat's finished recap; these recaps are the decision pass's entire input),
            ``deciding`` (``{chats, user, standing}``).
          * ``("checkin_prompt", {pass, segments, input_tokens, rag_tokens, rag_disabled,
            max_new_tokens, context_length})`` — one per generation (each recap pass, then
            the decision pass): the FULL model-facing prompt, split into the same labelled
            ``{kind, label, text}`` segments the Chat tab's Debug checkbox renders, so one
            renderer serves both. ``rag_disabled`` says the pass asked for no retrieval,
            which an absent RAG segment alone cannot distinguish from an empty one.
          * ``("checkin_chunk", {text})`` — streamed reasoning deltas (``<think>`` included).
          * ``("checkin_done", {composed?, decision?, session?, opener?, silence_hours? |
            skipped, message?})`` — terminal; a yes wrote the reversed check-in session
            into the chats list.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the decision pass
        is a full generation, hence the generous timeout."""
        with self._rpc_lock:
            req = {"type": "checkin_now"}
            if sim_hours is not None:
                req["sim_hours"] = sim_hours
            if user:
                req["user"] = str(user)
            self._send(req)
            while True:
                msg = self._recv_until(
                    {"checkin_stage", "checkin_prompt", "checkin_chunk", "checkin_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("checkin_done", "error", "connection_error"):
                    return

    def deliberate_now(self, n=None):
        """Manually run one deliberation pass (Worklog-tab "Deliberate", DRY RUN).

        Generator yielding ``(kind, msg)`` — Ava reads her recent episodic worklog and
        decides what she would do next; her reasoning is streamed so the operator can watch.
        DRY RUN: nothing is dispatched — the terminal message reports the action she WOULD
        take.
          * ``("deliberation_stage", {stage, ...})`` — phase markers (reading / deciding /
            decided).
          * ``("deliberation_chunk", {text})`` — streamed reasoning deltas (``<think>`` included).
          * ``("deliberation_done", {decided?, action?, why?, would_do?, episodes?,
            open_threads?, truncated? | skipped, message?})`` — terminal.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the decision pass is
        a full generation, hence the generous timeout."""
        with self._rpc_lock:
            req = {"type": "deliberate_now"}
            if n is not None:
                req["n"] = int(n)
            self._send(req)
            while True:
                msg = self._recv_until(
                    {"deliberation_stage", "deliberation_chunk", "deliberation_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("deliberation_done", "error", "connection_error"):
                    return

    def list_modules(self) -> dict:
        """The module registry for the Modules tab (each entry carries its prompt text)."""
        with self._rpc_lock:
            self._send({"type": "list_modules"})
            return self._recv_until({"modules_list"})

    def list_module_inputs(self, source: str) -> dict:
        """What a module of this input *source* can be run against (Modules tab).

        Keyed on the source rather than the module, so the two chat-reading modules share
        one listing. `chat` returns transcripts, `til` the fetched articles / news digests
        under the snippets tree.
        """
        with self._rpc_lock:
            self._send({"type": "list_module_inputs", "source": source})
            return self._recv_until({"module_inputs"})

    def run_module(self, module: str, filename: str, *, prompt: str = "",
                   inject=(), temperature: float = 0.7, top_p: float = 0.95):
        """Run one module against one chat (Modules tab "Simulate"). **Writes nothing.**

        Generator yielding ``(kind, msg)``:
          * ``("module_stage", {stage, ...})`` — phase markers (reading / generating / parsed).
          * ``("module_chunk", {text})`` — generation deltas (``<think>`` included).
          * ``("module_done", {ok, raw, records, counts, truncated, ... | skipped, message?})``
            — terminal.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the pass is a full
        generation over a whole transcript, hence the generous timeout."""
        with self._rpc_lock:
            self._send({
                "type": "run_module", "module": module, "filename": filename,
                "prompt": prompt, "inject": list(inject),
                "temperature": temperature, "top_p": top_p,
            })
            while True:
                msg = self._recv_until(
                    {"module_stage", "module_chunk", "module_done"}, timeout=900.0)
                t = msg.get("type")
                yield t, msg
                if t in ("module_done", "error", "connection_error"):
                    return

    def persona_preview(self, temperature: float = 0.7, top_p: float = 0.95):
        """Run a non-mutating persona/prompt self-review from the Sleep tab."""
        with self._rpc_lock:
            self._send({
                "type": "persona_preview",
                "temperature": temperature,
                "top_p": top_p,
            })
            while True:
                msg = self._recv_until(
                    {"persona_preview_context", "persona_preview_chunk",
                     "persona_preview_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("persona_preview_done", "error", "connection_error"):
                    return

    def prompt_experiment(self, temperature: float = 0.9, top_p: float = 0.95):
        """Run one prompt-experiment generation from the Sleep tab.

        Generator yielding ``(kind, msg)`` for the "Prompt experiment" button — Ava
        rewrites her standing prompt freely and it becomes live (temporarily) on success:
          * ``("prompt_experiment_chunk", {text})`` — streamed reasoning/process deltas.
          * ``("prompt_experiment_done", {activated?, prompt_chars? | skipped, message?})``
            — terminal; on ``activated`` the experimental prompt is now live until reverted.
          * ``("error"/"connection_error", {...})`` — terminal failure.

        One logical RPC (holds ``_rpc_lock``) — run on a worker thread; the pass is a full
        generation, hence the generous timeout."""
        with self._rpc_lock:
            self._send({
                "type": "prompt_experiment",
                "temperature": temperature,
                "top_p": top_p,
            })
            while True:
                msg = self._recv_until(
                    {"prompt_experiment_chunk", "prompt_experiment_done"},
                    timeout=600.0,
                )
                t = msg.get("type")
                yield t, msg
                if t in ("prompt_experiment_done", "error", "connection_error"):
                    return

    def set_prompt(self, prompt: str) -> dict:
        """Activate an operator-authored standing prompt (Prompt tab "Update prompt").

        The hand-written sibling of :meth:`prompt_experiment`: the text goes into the same
        temporary tier and is reverted by the same :meth:`revert_prompt`, so
        ``chat_prompt.txt`` is never overwritten. Replaces an already-active experiment.
        """
        with self._rpc_lock:
            self._send({"type": "set_prompt", "prompt": prompt})
            return self._recv_until({"prompt_updated"})

    def revert_prompt(self) -> dict:
        """Revert an active prompt experiment back to the base standing prompt."""
        with self._rpc_lock:
            self._send({"type": "revert_prompt"})
            return self._recv_until({"prompt_revert_done"})

    def prompt_experiment_status(self) -> dict:
        """Fetch the live standing prompt and whether it is a temporary experiment.

        Replies ``{active, prompt, base_prompt, prompt_chars, created_ts?}`` — ``prompt``
        is what is actually live (the experiment when active, else the base prompt), which
        is what the Prompt tab's editbox opens on."""
        with self._rpc_lock:
            self._send({"type": "prompt_experiment_status"})
            return self._recv_until({"prompt_experiment_status"})

    # ------------------------------------------------------------------ #
    # Server-owned reflection run APIs (Phase 2+)                         #
    # ------------------------------------------------------------------ #

    def start_reflection_run(
        self,
        sessions: list,
        *,
        source: str = "ui",
        debug: bool = False,
        overrides: Optional[dict] = None,
        stages: Optional[list[str]] = None,
        clear_staging: bool = True,
        dry_run: bool = False,
        dry_full: bool = False,
        train_params: Optional[dict] = None,
        revisit: bool = False,
        reflect_context_length: Optional[int] = None,
    ) -> dict:
        """Create a server-owned reflection run. Returns reflection_run_started.

        ``dry_run`` requests a write-nothing preview: the server generates and reports
        the distilled artifacts but persists nothing (no staging, memory, ledger, RAG,
        or downstream stages). By default it is a consolidation-only "short reflection
        summary". ``dry_full`` extends it to the full "Dry Sleep" — consolidation +
        revision + branch experiment — still writing nothing.

        ``train_params`` (e.g. ``{lora_r, epochs, lr, model_id, skip_validation}``) is
        forwarded to the watchdog's ``POST /train`` when the run includes the ``train``
        stage; ignored otherwise.

        ``revisit`` requests the "revisit old chat" pass: with ``sessions`` empty the
        server picks one random chat >= 7 days old, re-reflects it under the current
        persona to re-derive its target (suppressing persona formation), skips ingestion,
        and never hands off to training. The ack carries ``chosen_session``.

        ``reflect_context_length`` sets the reflection window (max_seq_length the run
        packs to). Omit to use the server's configured default; a value is clamped
        server-side to [chat context, the physical ceiling the model was loaded at].
        """
        with self._rpc_lock:
            payload = {
                "type": "start_reflection_run",
                "sessions": sessions,
                "source": source,
                "debug": debug,
                "overrides": overrides or {},
                "clear_staging": clear_staging,
                "dry_run": dry_run,
            }
            if dry_full:
                payload["dry_full"] = True
            if revisit:
                payload["revisit"] = True
            if reflect_context_length is not None:
                payload["reflect_context_length"] = int(reflect_context_length)
            if stages is not None:
                payload["stages"] = stages
            if train_params is not None:
                payload["train_params"] = train_params
            self._send(payload)
            return self._recv_until({"reflection_run_started"})

    def reflection_run_status(self, run_id: str) -> dict:
        """Query current state and progress counters for a run."""
        with self._rpc_lock:
            self._send({"type": "reflection_run_status", "run_id": run_id})
            return self._recv_until({"reflection_run_status"})

    def stop_reflection_run(self, run_id: str) -> dict:
        """Request a graceful stop for a running or pending run."""
        with self._rpc_lock:
            self._send({"type": "stop_reflection_run", "run_id": run_id})
            return self._recv_until({"reflection_run_stop_requested"})

    def list_reflection_runs(self) -> dict:
        """List all known reflection runs (most-recent first)."""
        with self._rpc_lock:
            self._send({"type": "list_reflection_runs"})
            return self._recv_until({"reflection_runs_list"})

    def reflection_run_events(self, run_id: str, after_seq: int = 0) -> dict:
        """Fetch events for a run with seq > after_seq. Pass 0 to get all events."""
        with self._rpc_lock:
            self._send({
                "type": "reflection_run_events",
                "run_id": run_id,
                "after_seq": after_seq,
            })
            return self._recv_until({"reflection_run_events_batch"})

    def activity_events(self, after_seq: int = 0) -> dict:
        """Fetch unified activity-journal events with seq > after_seq (0 = the ring).

        The single, always-on stream every autonomous/GPU subsystem writes to
        (wander/outreach/synthesis/checkin/background_reflection + a coarse reflection
        mirror). The Activity tab polls this with one cursor, independent of any run, so
        the box is never silent while it works. The batch also carries the live
        ``activity`` chip (the open activity, or None). See server core.activity_log."""
        with self._rpc_lock:
            self._send({"type": "activity_events", "after_seq": after_seq})
            return self._recv_until({"activity_events_batch"})

    # ------------------------------------------------------------------ #
    # Encounter — Ava converses with a fellow (OpenAI-compatible) AI       #
    # ------------------------------------------------------------------ #

    def start_encounter(
        self,
        *,
        name: str,
        url: str,
        model: str,
        turns: int = 6,
        counterpart_system: str = "",
        framing_override: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        counterpart_temperature: float = 1.0,
        counterpart_top_p: float = 0.95,
        counterpart_max_tokens: int = 4096,
        counterpart_timeout: float = 600.0,
        api_key: str = "",
    ) -> dict:
        """Start a server-owned encounter. Returns the ``encounter_started`` reply.

        Ava (the local model) talks to the OpenAI-compatible endpoint at *url* for
        *turns* counterpart replies; the server logs the dialogue as an ordinary
        chat session and streams progress into the encounter event buffer.
        """
        with self._rpc_lock:
            self._send({
                "type": "start_encounter",
                "name": name,
                "url": url,
                "model": model,
                "turns": turns,
                "counterpart_system": counterpart_system,
                "framing_override": framing_override,
                "temperature": temperature,
                "top_p": top_p,
                "counterpart_temperature": counterpart_temperature,
                "counterpart_top_p": counterpart_top_p,
                "counterpart_max_tokens": counterpart_max_tokens,
                "counterpart_timeout": counterpart_timeout,
                "api_key": api_key,
            })
            return self._recv_until({"encounter_started"})

    def encounter_status(self) -> dict:
        """Current encounter status snapshot (cheap, no event payload)."""
        with self._rpc_lock:
            self._send({"type": "encounter_status"})
            return self._recv_until({"encounter_status"})

    def encounter_events(self, after_seq: int = 0) -> dict:
        """Buffered encounter events with seq > after_seq (0 = all)."""
        with self._rpc_lock:
            self._send({"type": "encounter_events", "after_seq": after_seq})
            return self._recv_until({"encounter_events_batch"})

    def stop_encounter(self) -> dict:
        """Ask the server to halt the active encounter after the current turn."""
        with self._rpc_lock:
            self._send({"type": "stop_encounter"})
            return self._recv_until({"encounter_stop_requested"})

    # ------------------------------------------------------------------ #
    # Watchdog HTTP management API (generic offline-job runner)            #
    # ------------------------------------------------------------------ #
    # During an offline job (Sleep "train" stage, replay, wipe) the watchdog stops
    # THIS WebSocket server to free the GPU, so reflection_run_events can't reach
    # us. The watchdog (HTTP, mgmt port) stays up and serves the generic job status
    # + progress; the Sleep tab polls these over plain HTTP for the job window,
    # then reconnects the socket.

    def _base_url_for(self, port: int) -> Optional[str]:
        """Derive http://<host>:<port> from the connected ws:// server URL."""
        if not self.server_url:
            return None
        from urllib.parse import urlparse
        parsed = urlparse(self.server_url)
        host = parsed.hostname or "127.0.0.1"
        scheme = "https" if parsed.scheme in ("wss", "https") else "http"
        return f"{scheme}://{host}:{port}"

    def watchdog_base_url(self) -> Optional[str]:
        """Derive http://<host>:<mgmt_port> (the watchdog) from the server URL."""
        return self._base_url_for(self.mgmt_port)

    def sidecar_base_url(self) -> Optional[str]:
        """Derive http://<host>:<sidecar_port> (the inference HTTP sidecar)."""
        return self._base_url_for(self.sidecar_port)

    def _http_get(self, base: Optional[str], path: str, timeout: float = 5.0) -> dict:
        """GET a JSON document from *base*+*path*. Returns {'error': ...} on failure."""
        if not base:
            return {"error": "no server URL to derive management endpoint"}
        import urllib.error
        import urllib.request
        url = base.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                if isinstance(data, dict):
                    data.setdefault("http_status", e.code)
                    return data
                return {"error": f"HTTP {e.code}: {body}"}
            except Exception:
                return {"error": f"HTTPError {e.code}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _watchdog_get(self, path: str, timeout: float = 5.0) -> dict:
        """GET a JSON document from the watchdog. Returns {'error': ...} on failure."""
        return self._http_get(self.watchdog_base_url(), path, timeout)

    def train_status(self) -> dict:
        """Watchdog offline-job status (cheap, no event payload).

        Returns ``{name, running, started_at, finished_at, returncode, stage?,
        last_message?, last_status?}`` or ``{error}``. The watchdog flips
        ``running`` true synchronously when it accepts ``POST /job/train`` — before
        it stops the inference server — so this is a race-free way to confirm that a
        dropped inference socket is the training hand-off (vs. a transient blip).
        """
        return self._watchdog_get("/job/status")

    def train_progress(self, after_seq: int = 0) -> dict:
        """Structured offline-job progress events with seq > after_seq.

        Returns ``{running, returncode, events:[...]}`` or ``{error}``. Each event:
        ``{seq, ts, run_id, stage, status, message, data?}``.
        """
        return self._watchdog_get(f"/job/progress?after_seq={int(after_seq)}")

    def _http_post(self, base: Optional[str], path: str,
                   payload: Optional[dict] = None, timeout: float = 10.0) -> dict:
        """POST a JSON body to *base*+*path*. Returns {'error': ...} on failure."""
        if not base:
            return {"error": "no server URL to derive management endpoint"}
        import urllib.error
        import urllib.request
        url = base.rstrip("/") + path
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                if isinstance(data, dict):
                    data.setdefault("http_status", e.code)
                    return data
                return {"error": f"HTTP {e.code}: {body}"}
            except Exception:
                return {"error": f"HTTPError {e.code}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _watchdog_post(self, path: str, payload: Optional[dict] = None,
                       timeout: float = 10.0) -> dict:
        """POST a JSON body to the watchdog. Returns {'error': ...} on failure."""
        return self._http_post(self.watchdog_base_url(), path, payload, timeout)

    def _sidecar_post(self, path: str, payload: Optional[dict] = None,
                      timeout: float = 10.0) -> dict:
        """POST a JSON body to the inference HTTP sidecar. Returns {'error'} on failure."""
        return self._http_post(self.sidecar_base_url(), path, payload, timeout)

    def set_precision(self, base_quant: str) -> dict:
        """Persist the connected server's base-model load precision (base_quant).

        Rewrites ``server_config.json`` on the connected box via the inference HTTP
        sidecar (``POST /precision``) WITHOUT restarting the inference server — the
        precision is read at model-load time, so the change takes effect on the
        *next* server restart. *base_quant* is one of ``"16bit"`` / ``"8bit"`` /
        ``"4bit"``. Returns ``{ok, base_quant, previous, note}`` or ``{error}``.
        """
        return self._sidecar_post("/precision", {"base_quant": base_quant})

    def wipe_state(self, wipe_chats: bool = False, model_id: str = "") -> dict:
        """Kick off a watchdog state wipe (DESTRUCTIVE disaster recovery).

        The watchdog stops this WebSocket server, deletes all regenerable artifacts
        (LoRA adapters, RAG indexes, reflection memory/ledger/runs/staging); when
        *wipe_chats* is set it also deletes the raw transcripts + reflection archive;
        then it repoints the base *model_id* (clearing the active adapter) and
        relaunches. Runs through the generic offline-job runner (``POST /job/wipe``,
        a synchronous job) — this client forms the ``wipe_state.py`` flags. Returns
        once the relaunch is issued, so reconnect from the Chat tab after the model
        reloads. Returns ``{ok, summary, model_id, adapter_id}`` or ``{error}``.
        """
        args: list[str] = []
        if wipe_chats:
            args += ["--wipe-chats"]
        if model_id:
            args += ["--model-id", model_id]
        # Allow time for the kill (waits up to ~10s) + file deletion; the slow model
        # load happens in the relaunched subprocess and is not awaited here.
        return self._watchdog_post("/job/wipe", {"args": args}, timeout=120.0)

    def _stream(
        self,
        payload: dict,
        temperature: float,
        top_p: float,
        max_new_tokens_setting: str,
        context_length: int,
        debug: bool,
    ) -> Generator[Tuple[str, object], None, None]:
        """Shared streaming loop used by both generate variants."""
        with self._rpc_lock:
            self._send({
                **payload,
                "max_new_tokens_setting": max_new_tokens_setting,
                "context_length": context_length,
                "temperature": temperature,
                "top_p": top_p,
                "debug": debug,
            })
            try:
                while True:
                    msg = self._recv_until(
                        {"chunk", "done", "cancelled", "error", "log", "prompt_debug",
                         "facts_block"},
                        timeout=300.0,
                    )
                    t = msg["type"]
                    if t == "chunk":
                        yield "chunk", msg.get("text", "")
                    elif t == "log":
                        yield "log", msg.get("message", "")
                    elif t == "prompt_debug":
                        # The one non-str payload on this stream: a list of labelled
                        # prompt segments the Chat tab renders colour-coded (see
                        # generation._prompt_debug_segments). Yielded structured rather
                        # than pre-formatted so the colouring stays a client concern.
                        yield "prompt_debug", msg.get("segments", [])
                    elif t == "facts_block":
                        # Stage 1's pick for this turn (facts tree → the injected block),
                        # arriving once before the first chunk so the Chat tab renders it
                        # above the reply. The whole message is yielded — the counts are
                        # what make an empty block readable, so this is not a text field
                        # with metadata attached.
                        yield "facts_block", dict(msg)
                    elif t == "done":
                        yield "done", msg.get("response", "")
                        return
                    elif t == "cancelled":
                        yield "cancelled", ""
                        return
                    elif t in ("error", "connection_error"):
                        yield "error", msg.get("message", "Unknown error.")
                        return
            except GeneratorExit:
                self.cancel()
                raise

    def stream_generate(
        self,
        message: str,
        user: str = "",
        max_new_tokens_setting: str = "75%",
        context_length: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        debug: bool = False,
        rag_history: bool = True,
        rag_facts: bool = True,
        rag_persona: bool = True,
    ) -> Generator[Tuple[str, object], None, None]:
        """Session-managed generation: server owns conversation history, RAG, and logging.

        *user* is the name of the person sending this message, so the server can
        tell Ava who is talking and attribute what is stored to its speaker.

        *rag_history* / *rag_facts* / *rag_persona* gate the three chat-time RAG
        channels (Chat tab checkboxes, all on by default); all off ⇒ adapter only.
        """
        yield from self._stream(
            {
                "type": "generate", "message": message, "user": user,
                "rag_history": rag_history, "rag_facts": rag_facts,
                "rag_persona": rag_persona,
            },
            temperature, top_p, max_new_tokens_setting, context_length, debug,
        )

    def cancel(self, *, discard: bool = False) -> None:
        """Stop the active generation.

        ``discard=True`` is the Chat-tab user action: the server rolls the turn
        back instead of committing the partial assistant text.  Internal stream
        guards keep the historical finalize-the-clean-prefix behaviour.
        """
        try:
            self._send({"type": "cancel", "discard": discard})
        except Exception:
            pass
