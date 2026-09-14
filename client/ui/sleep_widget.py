"""Sleep (reflection) widget — thin launcher and event-log viewer.

The server owns all reflection workflow: consolidation, judgement, clean re-answer,
and branch
experiment. The client starts a run, polls for events and progress, and stops
if requested. No prompt text, no pass orchestration, no artifact writes.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QCheckBox,
    QLineEdit,
    QMessageBox,
    QDoubleSpinBox,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QSplitter,
)
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor

# The Debug tab's dedup worker, reused rather than restated: both tabs drive the same
# `dedup_facts` RPC, and a second copy is a second thing to keep in step with it.
from ui.debug_widget import DedupFactsWorker

if TYPE_CHECKING:
    from ui.chat_widget import ChatWidget

    from core.backend_client import BackendClient


class ReflectionPollWorker(QThread):
    """Polls a server-owned reflection run off the GUI thread.

    The Sleep panel used to call ``reflection_run_events`` / ``reflection_run_status``
    straight from a ``QTimer`` on the main thread. Those are *blocking* socket RPCs,
    and the server's asyncio loop is GIL-starved during long GPU passes, so each poll
    parked the Qt event loop for seconds at a time — the UI froze (scrolling included)
    for the whole run while every other window stayed responsive.

    This worker does the blocking I/O on its own thread and hands results back through
    queued signals; the main thread only renders. Reconnect/superseded handling lives
    here too, since ``connect()`` is itself blocking. All run RPCs (including the user
    stop) flow through this one thread, so they never contend for ``_rpc_lock`` on the
    GUI thread.
    """

    events_ready = pyqtSignal(list)   # new events (seq > cursor), in order
    status_ready = pyqtSignal(dict)   # reflection_run_status payload
    connection_lost = pyqtSignal()    # socket dropped; reconnecting (emitted once)
    reconnected = pyqtSignal()        # socket re-established
    superseded = pyqtSignal()         # another device took over (close code 4000)
    server_down_for_train = pyqtSignal()  # socket down + watchdog confirms training started

    # Consecutive failed reconnects to tolerate before probing the watchdog for a
    # training hand-off. A transient blip reconnects within a tick, so this only
    # trips when the inference server is genuinely down (the watchdog stopped it).
    _TRAIN_HANDOFF_DOWN_THRESHOLD = 2

    def __init__(
        self,
        client: "BackendClient",
        run_id: str,
        server_url: str,
        *,
        start_seq: int = 0,
        interval: float = 2.0,
        train_expected: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._run_id = run_id
        self._server_url = server_url
        self._after_seq = start_seq
        self._interval = interval
        self._train_expected = train_expected
        self._down_count = 0
        self._stop = threading.Event()
        self._stop_run_requested = False
        self._reported_disconnect = False

    def stop(self) -> None:
        """Stop polling without touching the server-side run."""
        self._stop.set()

    def stop_run(self) -> None:
        """Stop polling and ask the server to halt the run (user pressed Stop).

        The stop RPC is sent from this thread on the way out, so the GUI thread
        never blocks on it (and it can't deadlock against an in-flight poll that
        already holds ``_rpc_lock``)."""
        self._stop_run_requested = True
        self._stop.set()

    def run(self) -> None:
        self._poll_once()  # fill the panel immediately, not after the first interval
        while not self._stop.wait(self._interval):
            self._poll_once()
        if self._stop_run_requested:
            try:
                self._client.stop_reflection_run(self._run_id)
            except Exception:
                pass

    def _poll_once(self) -> None:
        client = self._client
        if not client.is_connected():
            # Close code 4000 = another device took over this session. Detach
            # instead of reconnecting, which would start an eviction war.
            if client.last_close_code == 4000:
                self.superseded.emit()
                self._stop.set()
                return
            if not self._reconnect():
                # The inference socket is down. During a train-staged run this is
                # almost always the watchdog stopping us to free the GPU for the
                # LoRA cycle — but the "completed" status that normally triggers the
                # train-poll hand-off rides the same socket and is easily missed in
                # the sub-second window before the server dies. So once the socket is
                # confirmed persistently down, ask the watchdog directly: if a train
                # cycle is running, hand off (a transient blip reconnects before the
                # threshold and never reaches here).
                self._down_count += 1
                if (self._train_expected
                        and self._down_count >= self._TRAIN_HANDOFF_DOWN_THRESHOLD
                        and self._training_in_progress()):
                    self.server_down_for_train.emit()
                    self._stop.set()
                return  # still down — retry next tick (the run keeps going server-side)
        self._down_count = 0  # connected (or just reconnected) — clear the down streak

        try:
            result = client.reflection_run_events(self._run_id, after_seq=self._after_seq)
        except Exception:
            return
        if result.get("type") == "reflection_run_events_batch":
            events = result.get("events") or []
            if events:
                for ev in events:
                    seq = ev.get("seq", 0)
                    if seq > self._after_seq:
                        self._after_seq = seq
                self.events_ready.emit(events)

        try:
            status = client.reflection_run_status(self._run_id)
        except Exception:
            return
        if status.get("type") == "reflection_run_status":
            self.status_ready.emit(status)

    def _reconnect(self) -> bool:
        """Re-establish the dropped socket so polling can resume. Returns success.

        The run continues on the server throughout, so the only state to recover is
        the connection; subsequent polls replay events after the cursor. Reported to
        the panel once per disconnect so a recovered run reads as one continuous log."""
        if not self._reported_disconnect:
            self.connection_lost.emit()
            self._reported_disconnect = True
        try:
            self._client.connect(self._server_url)
        except Exception:
            return False
        self._reported_disconnect = False
        self.reconnected.emit()
        return True

    def _training_in_progress(self) -> bool:
        """True if the watchdog reports a LoRA cycle is running (the train hand-off).

        The watchdog flips ``running`` true synchronously when it accepts ``POST
        /train`` (before stopping inference), so by the time our socket is down this
        is already set — distinguishing the training stop from a network blip (where
        the watchdog reports ``running`` false) or a total outage (watchdog also
        unreachable → ``error`` → we keep reconnecting, since we couldn't show
        training progress anyway)."""
        try:
            st = self._client.train_status()
        except Exception:
            return False
        if not isinstance(st, dict) or st.get("error"):
            return False
        if st.get("running"):
            return True
        # Defensive: a started-but-not-finished cycle (rare timing) also counts.
        return bool(st.get("started_at") and not st.get("finished_at"))


class ReflectionLiveWorker(QThread):
    """Renders server-pushed reflection events the instant they arrive, off-thread.

    The reflection run streams every progress event live over the socket, but the
    Sleep tab historically consumed them only through :class:`ReflectionPollWorker`'s
    ``reflection_run_events`` RPC. That RPC is a round-trip, and the server's asyncio
    loop is GIL-starved during the *non-streaming* branch fork generation (one
    blocking ``generate`` per fork, no per-token yield), so the poll can't be answered
    until the whole branch phase ends — every candidate then lands in one batch.

    This worker drains the live pushes (``BackendClient.drain_reflection_events``),
    which never touch the socket or the RPC lock and so aren't starved: a candidate
    renders as soon as the server flushes it. Polling stays the reliable backstop
    (reconnect/catch-up); the widget dedups the two paths by event ``seq``."""

    events_ready = pyqtSignal(list)   # live events (may overlap the poll; deduped by seq)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                events = self._client.drain_reflection_events(timeout=1.0)
            except Exception:
                events = []
            if events and not self._stop.is_set():
                self.events_ready.emit(events)


class TrainPollWorker(QThread):
    """Polls the watchdog's /train/progress over HTTP during the LoRA train stage.

    The Sleep "train" stage hands LoRA production to the watchdog, which stops THIS
    WebSocket inference server to free the GPU — so the reflection event stream is
    dead for the whole training window. The watchdog (HTTP mgmt port) stays up and
    serves structured progress (stages + validation-tier replies); this worker
    polls it on its own thread and feeds the same panel, then signals completion so
    the panel can finalize and the socket can reconnect to the relaunched server.

    Completion is "training was observed running (or a terminal 'done' event landed)
    and is now stopped". A startup grace bounds the case where we attach after the
    cycle already finished (or the watchdog is unreachable) so the UI never hangs.
    """

    train_events_ready = pyqtSignal(list)   # new progress events (seq-ordered)
    train_status_ready = pyqtSignal(dict)   # {running, returncode}
    train_finished = pyqtSignal(dict)        # {returncode, observed}

    def __init__(self, client: "BackendClient", *, interval: float = 1.5,
                 startup_grace_s: float = 60.0, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._interval = interval
        self._startup_grace_s = startup_grace_s
        self._stop = threading.Event()
        self._after_seq = 0

    def stop(self) -> None:
        """Stop polling. Training itself keeps running on the watchdog (no halt API)."""
        self._stop.set()

    def run(self) -> None:
        start = time.monotonic()
        seen_running = False
        saw_done = False
        returncode = None
        first = True
        while first or not self._stop.wait(self._interval):
            first = False
            prog = self._client.train_progress(after_seq=self._after_seq)
            if not isinstance(prog, dict) or prog.get("error"):
                # Watchdog not reachable yet (server may still be coming down).
                # Tolerate through the startup grace; bail after if never seen.
                if not seen_running and (time.monotonic() - start) > self._startup_grace_s:
                    break
                continue

            events = prog.get("events") or []
            if events:
                for ev in events:
                    seq = ev.get("seq", 0)
                    if seq > self._after_seq:
                        self._after_seq = seq
                    if ev.get("stage") == "done":
                        saw_done = True
                self.train_events_ready.emit(events)

            running = bool(prog.get("running"))
            returncode = prog.get("returncode")
            if running:
                seen_running = True
            self.train_status_ready.emit({"running": running, "returncode": returncode})

            # Done once we've actually observed the cycle (running or a terminal
            # event) and it has since stopped. The relaunched server is up by then.
            if (seen_running or saw_done) and not running:
                break
            # Never observed it start within the grace window — give up so the panel
            # doesn't wait forever (cycle already over before we attached, or no
            # watchdog). Events (if any) were still rendered above.
            if not seen_running and not saw_done and (time.monotonic() - start) > self._startup_grace_s:
                break

        self.train_finished.emit({"returncode": returncode, "observed": seen_running})


class TilFetchWorker(QThread):
    """Fetches a TIL digest and streams a dry-run learning reflection off the GUI thread.

    Gradual step toward the "Today I Learned" source: the Sleep tab's "Learn" button
    asks the server to fetch a day's news digest into ``server/til`` and then run a
    DRY-RUN learning reflection over it (writes nothing). This worker drives the
    streaming ``til_learn`` RPC on its own thread and relays each stage back through
    queued signals; all rendering happens on the GUI thread.
    """

    digest_ready = pyqtSignal(dict)    # the fetched digest summary + preview
    reflect_chunk = pyqtSignal(str)    # streamed reflection delta
    reflect_done = pyqtSignal(dict)    # terminal: {text, report, skipped?}
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.til_learn(reflect=True):
                if kind == "til_fetched":
                    self.digest_ready.emit(msg)
                elif kind == "til_reflect_chunk":
                    self.reflect_chunk.emit(msg.get("text", ""))
                elif kind == "til_reflect_done":
                    self.reflect_done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Learn failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class TilApplyWorker(QThread):
    """Persists a completed Learn pass's findings off the GUI thread.

    The Sleep tab's "Apply learning" button hands the cleaned reflection text from
    the preceding dry-run Learn pass back to the server, which routes the same
    modifiers it previewed into Ava's live memory (RAG/weights/ledger). This worker
    drives that one blocking ``til_apply`` RPC and relays the result.
    """

    applied = pyqtSignal(dict)         # {counts, source, skipped?}
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", text: str, date: str,
                 kind: str = "", parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._text = text
        self._date = date
        self._kind = kind   # "wander" → SFT learning dataset; "" → live memory

    def run(self) -> None:
        try:
            result = self._client.til_apply(self._text, self._date, self._kind)
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))
            return
        if not isinstance(result, dict) or result.get("type") != "til_applied":
            self.error_occurred.emit(
                (result or {}).get("message", "apply failed") if isinstance(result, dict)
                else "apply failed"
            )
            return
        self.applied.emit(result)


class TilLookupWorker(QThread):
    """Drives the lookup loop off the GUI thread: resolve open [ask:search] questions.

    The Sleep tab's "Resolve Questions" button asks the server to extract subjects
    from Ava's open search questions (on the clean base), fetch each subject's
    Wikipedia article, and run a dry-run learning pass over the answers. This worker
    relays each stage through queued signals; the terminal ``reflect_done`` shares
    the Learn result shape, so the same Apply Learning path persists it.
    """

    collected = pyqtSignal(dict)        # {count, questions}
    log = pyqtSignal(dict)              # {message} — clean-base swap / progress
    subjects = pyqtSignal(dict)         # {subjects}
    fetched = pyqtSignal(dict)          # {subject, title?, via?, missing?, ...}
    reflect_chunk = pyqtSignal(str)     # streamed learning-pass delta
    reflect_done = pyqtSignal(dict)     # terminal: {text, report, skipped?}
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.til_lookup():
                if kind == "til_lookup_collected":
                    self.collected.emit(msg)
                elif kind == "til_lookup_log":
                    self.log.emit(msg)
                elif kind == "til_lookup_subjects":
                    self.subjects.emit(msg)
                elif kind == "til_lookup_fetched":
                    self.fetched.emit(msg)
                elif kind == "til_reflect_chunk":
                    self.reflect_chunk.emit(msg.get("text", ""))
                elif kind == "til_reflect_done":
                    self.reflect_done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Look-up failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class TilWanderWorker(QThread):
    """Drives a 'wander' off the GUI thread.

    The Sleep tab's "Wander" button asks the server to pick a random enabled wiki
    source, fetch a random page, and run a dry-run learning pass over it. The
    terminal ``reflect_done`` shares the Learn result shape, so the same Apply
    Learning path persists it. The "Visit" control provides an explicit URL but
    otherwise runs the same sequence.
    """

    wandered = pyqtSignal(dict)        # {wiki, lang, title, url, chars}
    reflect_chunk = pyqtSignal(str)
    reflect_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", lang: str = "", source: str = "",
                 url: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._lang = lang
        self._source = source
        self._url = url

    def run(self) -> None:
        try:
            for kind, msg in self._client.til_wander(self._lang, self._source, self._url):
                if kind == "til_wandered":
                    self.wandered.emit(msg)
                elif kind == "til_reflect_chunk":
                    self.reflect_chunk.emit(msg.get("text", ""))
                elif kind == "til_reflect_done":
                    self.reflect_done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Wander failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class OutreachWorker(QThread):
    """Drives one manual outreach-decision pass off the GUI thread.

    The Sleep tab's "Reach Out" button asks the server to run the same decision pass its
    idle-wake heartbeat runs — Ava picks her top open ask and decides whether to raise it
    now — while streaming her reasoning back so the operator can watch her deliberate. On
    a yes the server writes the reversed outreach session directly (it lands in the chats
    list); there is nothing to Apply.
    """

    question = pyqtSignal(dict)        # {question, ask_kind, user}
    prompt_debug = pyqtSignal(dict)    # {pass, segments, …} the assembled decision prompt
    chunk = pyqtSignal(str)            # streamed reasoning delta
    done = pyqtSignal(dict)            # terminal outreach_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.outreach_now():
                if kind == "outreach_question":
                    self.question.emit(msg)
                elif kind == "outreach_prompt":
                    self.prompt_debug.emit(msg)
                elif kind == "outreach_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "outreach_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Outreach failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class SynthesisWorker(QThread):
    """Drives one manual synthesis pass off the GUI thread.

    The Sleep tab's "Chat reach out" button asks the server to run the same pass its
    idle-wake heartbeat runs — Ava re-reads an aged chat and asks what she now wonders —
    while streaming her reasoning back so the operator can watch. A surfaceable question
    writes a reversed session directly (it lands in the chats list) and the rest go to
    the question pool; there is nothing to Apply.
    """

    stage = pyqtSignal(dict)           # {stage, ...} phase markers
    chunk = pyqtSignal(str)            # streamed reasoning delta
    done = pyqtSignal(dict)            # terminal synthesis_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.synthesis_now():
                if kind == "synthesis_stage":
                    self.stage.emit(msg)
                elif kind == "synthesis_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "synthesis_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Synthesis failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class CheckinWorker(QThread):
    """Drives one manual (simulated) check-in decision off the GUI thread.

    The Sleep tab's "Check In" button asks the server to run the same pass its idle-wake
    sweep runs — after a stretch of user silence Ava reviews her recent chats and
    decides whether to reach out on her own accord — but with the silence period forged
    so it fires regardless of the real elapsed time. Her reasoning is streamed back so
    the operator can watch; a yes writes the reversed session directly (it lands in the
    chats list). There is nothing to Apply.

    Scoped to ONE person (*user*, the name from the Chat tab's box): silence and "what have
    we been talking about" are questions about somebody in particular, and the autonomous
    job now asks them per person. This runs one of those decisions rather than the whole
    sweep, so what streams into the log is a single deliberation.
    """

    stage = pyqtSignal(dict)           # {stage, hours, chats, ...} phase markers
    prompt_debug = pyqtSignal(dict)    # {pass, segments, …} one per generation
    chunk = pyqtSignal(str)            # streamed reasoning delta
    done = pyqtSignal(dict)            # terminal checkin_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", sim_hours=None, user=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._sim_hours = sim_hours
        self._user = user

    def run(self) -> None:
        try:
            for kind, msg in self._client.checkin_now(sim_hours=self._sim_hours,
                                                      user=self._user):
                if kind == "checkin_stage":
                    self.stage.emit(msg)
                elif kind == "checkin_prompt":
                    self.prompt_debug.emit(msg)
                elif kind == "checkin_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "checkin_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Check-in failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class PersonaPreviewWorker(QThread):
    """Runs a non-mutating prompt self-review off the GUI thread."""

    context = pyqtSignal(dict)        # assembled input sizes
    chunk = pyqtSignal(str)           # streamed reasoning delta
    done = pyqtSignal(dict)           # terminal persona_preview_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", temperature: float, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._temperature = temperature

    def run(self) -> None:
        try:
            for kind, msg in self._client.persona_preview(temperature=self._temperature):
                if kind == "persona_preview_context":
                    self.context.emit(msg)
                elif kind == "persona_preview_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "persona_preview_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Persona preview failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class ReconcileSelfWorker(QThread):
    """Runs the clean-base self-reconciliation off the GUI thread, streaming progress.

    Slow: the server swaps the adapter out (two full model reloads) inside a
    CleanBaseSession and judges Ava's live [persona]/[fact] items against her current
    persona digest in BATCHES (thousands of items → dozens of sequential calls). The
    server streams a ``reconcile_stage`` event per batch so the Sleep-tab log can show
    it live; ``progress`` carries those, ``reconcile_done`` the terminal
    ``self_reconciled`` payload (which echoes ``dry_run`` so the caller can branch).
    """

    progress = pyqtSignal(dict)        # reconcile_stage events (start / per-batch)
    reconcile_done = pyqtSignal(dict)  # terminal self_reconciled
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", dry_run: bool, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._dry_run = dry_run

    def run(self) -> None:
        try:
            for kind, msg in self._client.reconcile_self(self._dry_run):
                if kind == "reconcile_stage":
                    self.progress.emit(msg)
                elif kind == "self_reconciled":
                    self.reconcile_done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Reconcile failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class DigestDryRunWorker(QThread):
    """Runs the persona-digest DRY RUN off the GUI thread, streaming progress.

    Slow and two-phase: the server swaps the adapter out (two full model reloads) to
    cluster Ava's live [persona] evidence into themes on the clean base — many sequential
    grouping calls over blocks — then swaps back and synthesizes the self-portrait on the
    adapter, whose tokens stream in. ``progress`` carries the ``digest_dryrun_stage``
    events, ``chunk`` the synthesis deltas, ``done`` the terminal payload. Nothing is
    written server-side, so there is no Apply path.
    """

    progress = pyqtSignal(dict)        # digest_dryrun_stage events
    chunk = pyqtSignal(str)            # synthesis delta
    done = pyqtSignal(dict)            # terminal digest_dryrun_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", block_size: int, temperature: float,
                 parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._block_size = block_size
        self._temperature = temperature

    def run(self) -> None:
        try:
            for kind, msg in self._client.digest_dryrun(
                    block_size=self._block_size, temperature=self._temperature):
                if kind == "digest_dryrun_stage":
                    self.progress.emit(msg)
                elif kind == "digest_dryrun_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "digest_dryrun_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Digest dry run failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class ResolveContradictionsWorker(QThread):
    """Runs the clean-base fact-contradiction resolution off the GUI thread, streaming.

    Groups live [fact] items by subject and, per group, asks the clean base which
    contradict; keeps the newest (the correction) and supersedes the older. The server
    streams a ``contradict_stage`` per subject group so the Sleep log shows progress;
    ``progress`` carries those, ``done`` the terminal ``contradictions_resolved`` payload
    (which echoes ``dry_run`` so the caller can branch).
    """

    progress = pyqtSignal(dict)
    done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", dry_run: bool, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._dry_run = dry_run

    def run(self) -> None:
        try:
            for kind, msg in self._client.resolve_contradictions(self._dry_run):
                if kind == "contradict_stage":
                    self.progress.emit(msg)
                elif kind == "contradictions_resolved":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Resolve conflicts failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class SleepWidget(QWidget):
    """Reflection panel: launch a server-owned run and display its event log."""

    _DIVIDER = "─" * 60

    # Events the output panel shows; others are progress-label only.
    _DISPLAY_EVENTS = frozenset({
        "run_started",
        "session_started",
        "session_skipped",
        "phase_started",
        "phase_progress",
        "phase_done",
        # A phase that failed without failing the RUN (archive, persona activation) —
        # best-effort steps whose whole point is that they don't take a committed
        # reflection down. That also made them invisible here, so an operator whose
        # persona never activated saw a clean run. The activity journal already
        # mirrors this event (`_ACTIVITY_MIRROR_EVENTS`); this is the Sleep-tab half.
        "phase_error",
        # Per-block persona clustering. The digest's grouping is now many sequential
        # clean-base calls — the longest silent stretch in a run — so it reports.
        "persona_cluster_progress",
        # Per-block fact dedup, for the same reason: it is the last thing a run does, on
        # the whole live [fact] store, and it too is many sequential clean-base calls.
        "fact_dedup_progress",
        "pass_error",
        "pass_warning",
        "pass_progress",
        "branching_disabled",
        "branch_started",
        "branch_candidate",
        "branch_choosing",
        "branch_done",
        "branch_skipped",
        "entrainment_started",
        "entrainment_forced",
        "entrainment_skipped",
        "run_completed",
        "run_stopped",
        "run_failed",
    })

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self.text_font = QFont("Courier")

        # Reflection-specific sampling temperature — independent from the chat
        # tab's value. Defaults just under chat's 1.0 so reflection samples the
        # recommended distribution rather than collapsing onto a greedy mode.
        self.temperature: float = 0.9

        # Run state — set by _begin_server_run, cleared on completion.
        self._server_run_id: str = ""
        self._poll_worker: Optional[ReflectionPollWorker] = None
        # Renders live event pushes alongside the poll worker (see ReflectionLiveWorker):
        # the only path that keeps branch candidates streaming while the poll is
        # GIL-starved. Runs for a run's lifetime, torn down with the poll worker.
        self._live_worker: Optional[ReflectionLiveWorker] = None
        # Train stage: when "train" is a requested stage, the run completes (over
        # the socket) and hands LoRA production to the watchdog, which stops this
        # server. _train_requested drives the handoff into HTTP-polling the watchdog
        # for live training progress; _train_poll_worker does that off-thread.
        self._train_requested: bool = False
        # Dry run: a preview that writes nothing. When set, completion finalizes the
        # panel here instead of handing off to the watchdog train poll. _dry_full
        # distinguishes the two shapes: False = "Short Summary" (consolidation only),
        # True = "Dry Sleep" (full consolidation + revision + branch experiment).
        self._dry_run: bool = False
        self._dry_full: bool = False
        self._train_poll_worker: Optional[TrainPollWorker] = None
        # "Learn" (manual debug): fetch a Wikipedia digest + dry-run learning pass
        # off-thread. _til_stream_open tracks whether a streamed reflection delta
        # left the cursor mid-line (so the modifiers block starts on a fresh line).
        self._til_worker: Optional[TilFetchWorker] = None
        self._til_stream_open: bool = False
        # "Apply learning" persists the most recent Learn pass's findings. Held until
        # the operator applies them (or starts a fresh Learn). _til_apply_worker does
        # the blocking write off-thread; _til_pending_* carry what to apply.
        self._til_apply_worker: Optional[TilApplyWorker] = None
        self._til_pending_text: str = ""
        self._til_pending_date: str = ""
        # Which flow produced the pending findings: "learn" | "lookup" | "wander".
        # Wander applies route to the SFT learning dataset; the rest write live memory.
        self._til_pending_source: str = ""
        # "Resolve Questions" runs the lookup loop (extract → fetch → reflect) and
        # shares the Apply path via the same pending findings.
        self._til_lookup_worker: Optional[TilLookupWorker] = None
        # "Wander" reflects on a random approved-wiki page (serendipity source);
        # shares the Apply path too.
        self._til_wander_worker: Optional[TilWanderWorker] = None
        # "Reach Out" manually runs one Ava-initiated outreach decision pass (debug),
        # streaming her reasoning into the log. Writes nothing to apply — a yes lands a
        # session in the chats list directly (server-side), like the autonomous path.
        self._outreach_worker: Optional["OutreachWorker"] = None
        self._outreach_stream_open: bool = False
        # Whether the "her reasoning:" header has been written for the current outreach
        # run. The header is emitted by whichever comes first — the prompt-debug block or
        # the first reasoning delta — so it lands under the prompt when one is streamed and
        # still appears if it isn't.
        self._outreach_reasoning_header: bool = False
        # "Chat reach out" manually runs one synthesis pass (debug): Ava re-reads an aged
        # chat and asks what she now wonders, streaming her reasoning into the log. Writes
        # nothing to apply — a surfaceable question lands a reversed session in the chats
        # list directly (server-side), the rest go to the question pool, like the
        # autonomous path.
        self._synthesis_worker: Optional["SynthesisWorker"] = None
        self._synthesis_stream_open: bool = False
        # "Check In" manually runs one (simulated) check-in decision pass (debug): after a
        # forged stretch of user silence Ava reviews her recent chats and decides whether
        # to reach out on her own accord, streaming her reasoning into the log. Writes
        # nothing to apply — a yes lands a reversed session in the chats list directly
        # (server-side), like the autonomous path.
        self._checkin_worker: Optional["CheckinWorker"] = None
        self._checkin_stream_open: bool = False
        # "Persona preview" asks Ava to review her standing prompt against her
        # current persona digest + logged prompt deltas. It streams thinking and
        # writes nothing; actual prompt mutation is intentionally out of scope.
        self._persona_preview_worker: Optional["PersonaPreviewWorker"] = None
        self._fact_dedup_worker: Optional["DedupFactsWorker"] = None
        self._persona_preview_stream_open: bool = False
        # "Reconcile self" judges Ava's live [persona]/[fact] items against her current
        # persona digest on the clean base and, on confirm, SOFTENS (supersedes) the ones
        # she has grown past / that are now stale — reversibly. Dry-run preview first.
        self._reconcile_worker: Optional["ReconcileSelfWorker"] = None
        # "Resolve fact conflicts" clusters live [fact]s by subject, asks the clean base
        # which contradict, keeps the newest (the correction) and softens the older —
        # fixing corrections that wrote a contradicting fact. Dry-run preview first.
        self._contradict_worker: Optional["ResolveContradictionsWorker"] = None
        # "Persona digest (dry)" re-derives the self-portrait end to end — themes grouped
        # on the clean base, portrait written on the adapter — and only PRINTS it. There
        # is no apply path: the server persists nothing, so there is no state to mirror.
        self._digest_dry_worker: Optional["DigestDryRunWorker"] = None
        # NB: the standing-prompt controls (Prompt experiment / Update prompt / Revert)
        # live in the **Prompt** tab (ui/prompt_widget.py), beside the editable prompt
        # they act on — not here.
        self._revision_filenames: list[str] = []
        self._stopping: bool = False
        self._last_event_seq: int = 0
        # Event seqs already rendered — the live and poll paths overlap, so both feed
        # _render_events and this set drops the duplicates (seqs can interleave, so a
        # high-water mark alone isn't enough).
        self._rendered_seqs: set[int] = set()
        # _stream_open: a raw delta left the output cursor mid-line (needs a
        # newline before the next framed line). _pass_streamed: the current
        # pass's generated text was already shown live (so phase_done must not
        # re-dump it). Tracked separately because branch events can interleave
        # between a revision stream and its phase_done.
        self._stream_open: bool = False
        self._pass_streamed: bool = False
        # Branch sub-pass runs inside revision but isn't a run *phase*, so the
        # run-status poll can't see it. Drive a branch-active flag off the
        # branch_started/branch_done events so the status line reflects the
        # (minutes-long) counterfactual generation instead of looking stalled.
        self._branch_active: bool = False
        self._branch_ex_idx: Optional[int] = None
        self._branch_ex_total: Optional[int] = None

        # Accumulated full-report data, built from the structured `report` payload
        # on each phase_done event. Rendered as a "FULL REPORT" section when the run
        # finishes — see _render_report.
        self._report_rag: list[dict] = []        # consolidation RAG inserts (fact/ask)
        self._report_resolved: list[dict] = []   # consolidation resolved evictions
        self._report_weights: list[dict] = []    # consolidation WEIGHTS facts/persona
        self._report_persona: list[dict] = []     # revision persona self-statements
        self._sessions_worker = None

        self._build_ui()

    # ---------------------------------------------------------------- #
    # Session listing (via server)                                      #
    # ---------------------------------------------------------------- #

    def _sync_reflect_ctx_ceiling(self) -> None:
        """Match the Reflect-ctx spinbox max to the server's physical window.

        The server reports ``reflect_context_length`` (the max_seq_length the model was
        loaded at) on every ``status`` poll, cached in ``client.last_status``. We cap the
        spinbox at that so the UI can't offer more than the load physically allows. Any
        value is still clamped server-side, so a stale ceiling is harmless.
        """
        spin = getattr(self, "spin_reflect_ctx", None)
        if spin is None:
            return
        ceiling = self._chat_widget._client.last_status.get("reflect_context_length")
        try:
            ceiling = int(ceiling)
        except (TypeError, ValueError):
            return
        if ceiling > 0 and spin.maximum() != ceiling:
            cur = spin.value()
            spin.setMaximum(ceiling)
            if cur > ceiling:
                spin.setValue(ceiling)

    def refresh_sessions(self) -> None:
        """Fetch past session list from the server using SessionsWorker."""
        client = self._chat_widget._client
        if not client.is_connected():
            return
        self._sync_reflect_ctx_ceiling()
        if self._sessions_worker is not None and self._sessions_worker.isRunning():
            return
        from ui.chat_widget import SessionsWorker
        self._sessions_worker = SessionsWorker(client)
        self._sessions_worker.sessions_ready.connect(self._on_sessions_ready)
        self._sessions_worker.error_occurred.connect(self._on_sessions_error)
        self._sessions_worker.start()

    def _on_sessions_ready(self, sessions: list) -> None:
        self.lst_sessions.clear()
        for s in reversed(sessions):  # most recent first
            item = QListWidgetItem(self._chat_widget._session_list_label(s))
            item.setData(Qt.ItemDataRole.UserRole, s["filename"])
            self.lst_sessions.addItem(item)
        self._update_sleep_button()

    def _on_sessions_error(self, err: str) -> None:
        self._lbl_status.setText(f"Failed to load sessions: {err}")

    def _update_sleep_button(self) -> None:
        selected = self.lst_sessions.selectedItems()
        has_selection = len(selected) >= 1
        running = (self._poll_worker is not None or self._train_poll_worker is not None)
        connected = self._chat_widget._client.is_connected()
        til_busy = self._til_worker is not None and self._til_worker.isRunning()
        lookup_busy = self._til_lookup_worker is not None and self._til_lookup_worker.isRunning()
        wander_busy = self._til_wander_worker is not None and self._til_wander_worker.isRunning()
        apply_busy = self._til_apply_worker is not None and self._til_apply_worker.isRunning()
        outreach_busy = self._outreach_worker is not None and self._outreach_worker.isRunning()
        synthesis_busy = (self._synthesis_worker is not None
                          and self._synthesis_worker.isRunning())
        checkin_busy = (self._checkin_worker is not None
                        and self._checkin_worker.isRunning())
        persona_busy = (self._persona_preview_worker is not None
                        and self._persona_preview_worker.isRunning())
        reconcile_busy = (self._reconcile_worker is not None
                          and self._reconcile_worker.isRunning())
        contradict_busy = (self._contradict_worker is not None
                           and self._contradict_worker.isRunning())
        digest_dry_busy = (self._digest_dry_worker is not None
                           and self._digest_dry_worker.isRunning())
        fact_dedup_busy = (self._fact_dedup_worker is not None
                           and self._fact_dedup_worker.isRunning())
        self.btn_sleep.setEnabled(connected and has_selection and not running and not persona_busy)
        if hasattr(self, "btn_dry_sleep"):
            self.btn_dry_sleep.setEnabled(connected and has_selection and not running and not persona_busy)
        if hasattr(self, "btn_summary"):
            self.btn_summary.setEnabled(connected and has_selection and not running and not persona_busy)
        # Learn / Resolve Questions run regardless of session selection; gate them on
        # a connection and no other job in flight (incl. each other and the apply).
        idle = (connected and not running and not til_busy and not lookup_busy
                and not wander_busy and not apply_busy and not outreach_busy
                and not synthesis_busy and not checkin_busy and not persona_busy
                and not reconcile_busy and not contradict_busy and not digest_dry_busy
                and not fact_dedup_busy)
        if hasattr(self, "btn_learn"):
            self.btn_learn.setEnabled(idle)
        if hasattr(self, "btn_lookup"):
            self.btn_lookup.setEnabled(idle)
        if hasattr(self, "btn_wander"):
            self.btn_wander.setEnabled(idle)
        if hasattr(self, "btn_outreach"):
            self.btn_outreach.setEnabled(idle)
        if hasattr(self, "btn_synthesis"):
            self.btn_synthesis.setEnabled(idle)
        if hasattr(self, "btn_checkin"):
            self.btn_checkin.setEnabled(idle)
        if hasattr(self, "btn_persona_preview"):
            self.btn_persona_preview.setEnabled(idle)
        if hasattr(self, "btn_reconcile"):
            self.btn_reconcile.setEnabled(idle)
        if hasattr(self, "btn_contradict"):
            self.btn_contradict.setEnabled(idle)
        if hasattr(self, "btn_digest_dry"):
            self.btn_digest_dry.setEnabled(idle)
        if hasattr(self, "btn_fact_dedup"):
            self.btn_fact_dedup.setEnabled(idle)
        if hasattr(self, "edt_visit_url"):
            self.edt_visit_url.setEnabled(idle)
        if hasattr(self, "btn_visit"):
            self.btn_visit.setEnabled(idle and bool(self.edt_visit_url.text().strip()))
        # "Apply learning" lights up only once a learning pass leaves findings
        # pending, and never while another job runs.
        if hasattr(self, "btn_apply_learning"):
            self.btn_apply_learning.setEnabled(idle and bool(self._til_pending_text))

    def _any_job_running(self) -> bool:
        """True when any reflection / learning / tool job holds the box.

        The same set the individual handlers assemble inline; collected here because a
        new tool button should not have to restate it (and a restatement is how one gets
        forgotten). Deliberately reuses the button-enable logic's own notion of busy.
        """
        workers = (
            self._til_worker, self._til_lookup_worker, self._til_wander_worker,
            self._til_apply_worker, self._outreach_worker, self._synthesis_worker,
            self._checkin_worker, self._persona_preview_worker, self._reconcile_worker,
            self._contradict_worker, self._digest_dry_worker, self._fact_dedup_worker,
        )
        if any(w is not None and w.isRunning() for w in workers):
            return True
        return self._poll_worker is not None or self._train_poll_worker is not None

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel: past sessions list widget
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Past Sessions:"))
        self.lst_sessions = QListWidget()
        self.lst_sessions.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.lst_sessions.itemSelectionChanged.connect(self._update_sleep_button)
        left_layout.addWidget(self.lst_sessions, 1)

        # Right panel: status, text log, and controls widget
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._lbl_status = QLabel(
            "Select past sessions on the left, then press Sleep to begin reflection."
        )
        right_layout.addWidget(self._lbl_status)

        # Live stats panel — a compact one-liner refreshed from each status poll
        # (elapsed · rough ETA · global exchange x/y · peak VRAM · discards · t/s).
        # Hidden until a run reports stats; the long branch/judge phases stream
        # nothing, so this is the run's heartbeat.
        self._lbl_stats = QLabel("")
        self._lbl_stats.setWordWrap(True)
        self._lbl_stats.setStyleSheet("color: palette(mid);")
        self._lbl_stats.setVisible(False)
        right_layout.addWidget(self._lbl_stats)

        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setPlaceholderText("Reflection event log will appear here...")
        self.txt_output.setFont(self.text_font)
        right_layout.addWidget(self.txt_output)

        # Controls: three rows, grouped by what they do, each ending in something
        # that absorbs the leftover width (a stretch, or the Visit URL box). The
        # panel has to fit a 13" laptop, so no row may exceed roughly 1000 px:
        #   1. the reflection run + the flags that modify it
        #   2. the autonomous jobs (learning / reach-out) + the sampling params
        #   3. the persona/prompt tools + the ad-hoc page visit
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 4, 0, 0)
        controls_layout.setSpacing(4)
        btn_row_run = QHBoxLayout()
        btn_row_jobs = QHBoxLayout()
        btn_row_tools = QHBoxLayout()
        self.btn_sleep = QPushButton("Sleep")
        self.btn_sleep.setEnabled(False)
        self.btn_sleep.setToolTip(
            "Run a full reflection cycle on the selected chat(s): "
            "Reflection → Merge RAG → Commit Training → Train (LoRA)."
        )
        self.btn_sleep.clicked.connect(self._on_sleep)
        btn_row_run.addWidget(self.btn_sleep)

        self.btn_dry_sleep = QPushButton("Dry Sleep")
        self.btn_dry_sleep.setEnabled(False)
        self.btn_dry_sleep.setToolTip(
            "Dry-run preview of a FULL reflection: run consolidation + revision + the "
            "branch experiment on the selected chat(s) — the chat summary, the revised "
            "IDEAL target per exchange, and the counterfactual branches — and show it all "
            "in the log below. Writes NOTHING: no staging, memory, ledger, persona digest, "
            "sidecar, or training. Use it to debug exactly what a real Sleep run would "
            "reflect and produce before committing to one. (The clean-base branch judge is "
            "not run — it is logged-only and would rewrite the sidecar.)"
        )
        self.btn_dry_sleep.clicked.connect(self._on_dry_sleep)
        btn_row_run.addWidget(self.btn_dry_sleep)

        self.btn_summary = QPushButton("Short Summary")
        self.btn_summary.setEnabled(False)
        self.btn_summary.setToolTip(
            "Dry-run preview: run ONLY the consolidation (first) reflection phase "
            "on the selected chat(s) and show the distilled summary — the [fact], "
            "[ask], [resolved] artifacts that would be written to RAG. "
            "([persona] is a revision-phase artifact and is not produced here.) "
            "Writes nothing: no staging, memory, ledger, or training. Use it to "
            "debug what a real Sleep run would record before committing to one."
        )
        self.btn_summary.clicked.connect(self._on_short_summary)
        btn_row_run.addWidget(self.btn_summary)

        self.btn_learn = QPushButton("Learn")
        self.btn_learn.setToolTip(
            "Fetch yesterday's Wikipedia Current events digest into server/til and "
            "run a DRY-RUN learning reflection over it — previews the findings Ava "
            "would keep, writing nothing. Press Apply Learning afterward to persist "
            "them into her live memory."
        )
        self.btn_learn.clicked.connect(self._on_learn)
        btn_row_jobs.addWidget(self.btn_learn)

        self.btn_lookup = QPushButton("Resolve Questions")
        self.btn_lookup.setToolTip(
            "Resolve Ava's open [ask:search] questions by looking them up: a clean "
            "base model (adapter off) extracts the subject of each question, the "
            "matching Wikipedia article is fetched, and Ava runs a dry-run learning "
            "pass over the answers. Press Apply Learning afterward to keep what she "
            "concludes — a resolved question is evicted, closing the loop."
        )
        self.btn_lookup.clicked.connect(self._on_lookup)
        btn_row_jobs.addWidget(self.btn_lookup)

        self.btn_wander = QPushButton("Wander")
        self.btn_wander.setToolTip(
            "Serendipity: reflect on a RANDOM page from a wiki you've approved in "
            "server/til/wiki_sources.json (only 'enabled' sources are drawn from). "
            "Ava reads a random article and reflects on it (dry run, free — costs no "
            "tokens). Press Apply Learning to apply that exchange two ways: add it to the "
            "SFT learning dataset (one one-shot example trained next cycle, then retired) "
            "AND write its findings to memory (RAG + weights). Button-only: wander is not "
            "part of the autonomous Sleep cycle."
        )
        self.btn_wander.clicked.connect(self._on_wander)
        btn_row_jobs.addWidget(self.btn_wander)

        self.btn_outreach = QPushButton("Reach Out")
        self.btn_outreach.setToolTip(
            "Debug: manually run one Ava-initiated outreach decision — the same pass "
            "the idle-wake heartbeat runs on its own. Ava picks her top surfaceable open "
            "question and deliberates on whether to raise it with you now; her reasoning "
            "streams into the log below. On a 'yes' she writes a reversed chat session "
            "(she speaks first) straight to the chats list — open it from the Chat tab's "
            "session list (badged 'Ava:') to reply. Nothing to Apply; the session lands "
            "directly. No session selection needed."
        )
        self.btn_outreach.clicked.connect(self._on_outreach)
        btn_row_jobs.addWidget(self.btn_outreach)

        self.btn_synthesis = QPushButton("Chat reach out")
        self.btn_synthesis.setToolTip(
            "Debug: manually run one synthesis pass — the same pass the idle-wake "
            "heartbeat runs on its own. Ava picks one of her older chats (>7 days, not "
            "recently revisited), re-reads it as who she is now, and surfaces any "
            "questions that newly arise; her reasoning streams into the log below. Every "
            "question lands in her open-question pool; the first she can raise directly, "
            "she opens a fresh chat about (reminding you which conversation it was) — a "
            "reversed session straight to the chats list (badged 'Ava:'). Nothing to "
            "Apply. No session selection needed."
        )
        self.btn_synthesis.clicked.connect(self._on_synthesis)
        btn_row_jobs.addWidget(self.btn_synthesis)

        self.btn_checkin = QPushButton("Check In")
        self.btn_checkin.setToolTip(
            "Debug/simulate: manually run one check-in decision — the same per-person pass "
            "the idle-wake sweep runs once someone has been silent for the configured "
            "threshold (checkin.silence_threshold_hours, default 5h). Here the silence "
            "period is FORGED so it fires regardless of how long it's really been: Ava "
            "reviews her last few conversations WITH THAT PERSON and decides, on her own "
            "accord, whether there's anything she wants to say to them. The person is "
            "whoever the Chat tab's user box names (blank ⇒ the active session's speaker); "
            "the autonomous job walks every user separately, this runs one of them. Streams "
            "her reasoning; a yes opens a fresh chat (a reversed session straight to the "
            "chats list, badged 'Ava:'). Nothing to Apply. No session selection needed."
        )
        self.btn_checkin.clicked.connect(self._on_checkin)
        btn_row_jobs.addWidget(self.btn_checkin)

        self.btn_revisit = QPushButton("Revisit old chat")
        self.btn_revisit.setToolTip(
            "Ava remembers one random past chat (>= 7 days old) and re-reflects on it "
            "under her CURRENT persona: 'with who I've become, would I still say this, or "
            "answer differently?'. Re-derives that chat's trainable target and rewrites "
            "its sidecar, producing the same reflection log below. Persona formation is "
            "suppressed (an obsolete chat must not reshape who she's becoming), no "
            "news/ask ingestion runs first, and it does NOT trigger a training cycle. No "
            "session selection needed — the server picks the chat; the old sidecar is "
            "restored if the run fails."
        )
        self.btn_revisit.clicked.connect(self._on_revisit)
        btn_row_run.addWidget(self.btn_revisit)

        self.btn_persona_preview = QPushButton("Persona preview")
        self.btn_persona_preview.setToolTip(
            "Ask Ava to review her current standing chat prompt against her persona "
            "self-presentation and logged prompt-delta proposals. Streams her thinking "
            "and final keep/change recommendation into this log. Preview only: no "
            "prompt file is changed."
        )
        self.btn_persona_preview.clicked.connect(self._on_persona_preview)
        btn_row_tools.addWidget(self.btn_persona_preview)

        self.btn_reconcile = QPushButton("Reconcile self…")
        self.btn_reconcile.setToolTip(
            "Judge Ava's live [persona]/[fact] memories against who she has become — her "
            "current persona digest — on the clean base (adapter off). Runs a PREVIEW "
            "first (writes nothing) listing which she would set aside: persona statements "
            "she has grown past, facts a later understanding has made stale. On confirm "
            "it SOFTENS them (supersede, not delete): dropped from recall and from active "
            "evidence/training, but kept as evidence-of-change. Append-only, so reversible. "
            "The server swaps the model twice, so this takes a while. No session selection "
            "needed; needs a persona digest (run a reflection first)."
        )
        self.btn_reconcile.clicked.connect(self._on_reconcile)
        btn_row_tools.addWidget(self.btn_reconcile)

        self.btn_contradict = QPushButton("Resolve fact conflicts…")
        self.btn_contradict.setToolTip(
            "Find live [fact] memories that CONTRADICT each other — the 'user corrected "
            "me but I wrote a new fact instead of replacing the stale one' case. Groups "
            "facts by subject, asks the clean base which directly conflict, and keeps the "
            "NEWEST in each conflict (the correction), softening the older ones "
            "(supersede, not delete — reversible, kept as evidence-of-change). Runs a "
            "PREVIEW first (writes nothing) listing old → correction. Judged in per-"
            "subject groups on the clean base, so a large memory takes several minutes. "
            "No session selection needed."
        )
        self.btn_contradict.clicked.connect(self._on_contradict)
        btn_row_tools.addWidget(self.btn_contradict)

        self.btn_digest_dry = QPushButton("Persona digest (dry)")
        self.btn_digest_dry.setToolTip(
            "DRY RUN of the persona digest — writes NOTHING. Re-derives Ava's "
            "self-portrait from scratch the way the reflection run would, but with each "
            "half on the model it belongs on: her live [persona] statements are grouped "
            "into themes on the CLEAN BASE (adapter off — deciding whether two statements "
            "say the same thing is a judgement, not self-expression), then the portrait "
            "is written on the ADAPTER, so the voice stays hers. Grouping runs in fixed "
            "blocks with a merge pass over the results, so it works on a large persona "
            "store instead of overflowing one prompt. The themes, the portrait, and how it "
            "compares to the active digest are printed below; no digest snapshot is saved "
            "and recall is untouched. The model is swapped twice, so this takes a while. "
            "No session selection needed."
        )
        self.btn_digest_dry.clicked.connect(self._on_digest_dryrun)
        btn_row_tools.addWidget(self.btn_digest_dry)

        self.btn_fact_dedup = QPushButton("Dedup facts")
        self.btn_fact_dedup.setToolTip(
            "Collapse paraphrases in Ava's live [fact] store — the fact-side counterpart "
            "of persona clustering, and the same pass a normal reflection run now does at "
            "the end of its clean-base window. Facts are de-duplicated at write time only "
            "by EXACT wording, so every run that re-notices one truth adds another live "
            "record for it; those restatements then surface together in chat. Grouping "
            "runs on the CLEAN BASE (adapter off — whether two facts say the same thing "
            "is a judgement, not self-expression), in fixed blocks with a merge pass over "
            "the results, so a large store works instead of overflowing one prompt. Each "
            "group keeps one record (the one she actually surfaces) and gives it the union "
            "of the group's recall cues, so nothing becomes unfindable.\n\n"
            "Runs as a DRY RUN first: the proposed merges are printed below and nothing is "
            "written. Confirm to apply. The model is swapped, so this takes a while. "
            "No session selection needed."
        )
        self.btn_fact_dedup.clicked.connect(self._on_fact_dedup)
        btn_row_tools.addWidget(self.btn_fact_dedup)

        self.btn_apply_learning = QPushButton("Apply Learning")
        self.btn_apply_learning.setEnabled(False)
        self.btn_apply_learning.setToolTip(
            "Persist the findings from the most recent Learn pass. Routes the same "
            "[fact]/[ask]/[resolved] modifiers previewed above into Ava's live memory "
            "(RAG + weights store + ledger) and refreshes recall, so she actually "
            "remembers what she learned. Enabled only after a Learn pass produced "
            "something to keep."
        )
        self.btn_apply_learning.clicked.connect(self._on_apply_learning)
        btn_row_jobs.addWidget(self.btn_apply_learning)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            "Halt the reflection run. Passes already saved are kept; the current "
            "pass finishes on the server before halting."
        )
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row_run.addWidget(self.btn_stop)

        btn_row_run.addStretch()

        # Sampling params close out the jobs row: the temperature applies to every
        # reflection pass on this tab, not just the run above.
        btn_row_jobs.addStretch()
        lbl_temp = QLabel("Temp:")
        lbl_temp.setToolTip("Sampling temperature for reflection passes.")
        btn_row_jobs.addWidget(lbl_temp)
        self.spn_temperature = QDoubleSpinBox()
        self.spn_temperature.setRange(0.0, 2.0)
        self.spn_temperature.setSingleStep(0.05)
        self.spn_temperature.setDecimals(2)
        self.spn_temperature.setValue(self.temperature)
        self.spn_temperature.setMaximumWidth(80)
        self.spn_temperature.setToolTip(
            "Sampling temperature for reflection passes. Default 0.9 (just under the "
            "chat tab's 1.0); lower = more consistent judgement; higher = more "
            "variation. Independent from the chat tab."
        )
        self.spn_temperature.valueChanged.connect(self._on_temperature_changed)
        btn_row_jobs.addWidget(self.spn_temperature)

        # Reflection window (max_seq_length a run packs to). Reflection may reason over
        # a larger window than chat, since a chat transcript + the sleep scaffolding
        # (prompt + RAG + open questions) is bigger than the chat alone. The ceiling is
        # the physical size the model was loaded at (server-reported); this can only
        # dial DOWN from it. 0 ⇒ use the server default.
        lbl_ctx = QLabel("Ctx:")
        lbl_ctx.setToolTip("Reflection window (max_seq_length) this run packs to.")
        btn_row_jobs.addWidget(lbl_ctx)
        self.spin_reflect_ctx = QSpinBox()
        self.spin_reflect_ctx.setRange(0, 32768)
        self.spin_reflect_ctx.setSingleStep(2048)
        self.spin_reflect_ctx.setValue(0)
        self.spin_reflect_ctx.setSpecialValueText("default")
        self.spin_reflect_ctx.setToolTip(
            "Reflection window: the token budget (max_seq_length) a reflection run packs "
            "to — the sleep prompt + RAG + open questions + the chat transcript. It can be "
            "larger than the chat context so a whole transcript still leaves headroom for "
            "the scaffolding. The maximum is the physical window the model was loaded at "
            "(server-set via reflect_context_length in server_config.json — raising THAT "
            "needs a reload); this control only dials down from it. 'default' uses the "
            "server's configured reflection window. A value below the chat context is "
            "floored to it server-side."
        )
        btn_row_jobs.addWidget(self.spin_reflect_ctx)

        # The tools row ends with the ad-hoc page visit, whose URL box takes the slack.
        self.edt_visit_url = QLineEdit()
        self.edt_visit_url.setPlaceholderText("https://example.org/wiki/Page")
        self.edt_visit_url.setToolTip(
            "Fetch this page and run the same two-pass Wander preview over it. "
            "Press Apply Learning afterward to keep the exchange and findings."
        )
        self.edt_visit_url.textChanged.connect(lambda _text: self._update_sleep_button())
        self.edt_visit_url.returnPressed.connect(self._on_visit)
        btn_row_tools.addWidget(self.edt_visit_url, 1)

        self.btn_visit = QPushButton("Visit")
        self.btn_visit.setEnabled(False)
        self.btn_visit.setToolTip(
            "Fetch the URL in the box and run the usual Wander sequence on that page."
        )
        self.btn_visit.clicked.connect(self._on_visit)
        btn_row_tools.addWidget(self.btn_visit)

        # Run flags close out the run row (they modify what Sleep does).
        self.chk_train = QCheckBox("Train (LoRA)")
        self.chk_train.setChecked(True)
        self.chk_train.setToolTip(
            "Run the offline LoRA cycle at the end of this Sleep run (ON by default).\n\n"
            "Unchecked = 'training-lite': reflection still runs in full and is still "
            "COMMITTED — distilled memory, facts, sidecar targets, anchors and user "
            "portraits all land in live memory exactly as they would with training on. "
            "What is skipped is only the weight update: no watchdog hand-off, so the "
            "inference server is never stopped and restarted, and the adapter stays "
            "whatever it is now.\n\n"
            "Instead the run mints a persona snapshot (data/persona/<run_id>/) and "
            "activates it, which is what puts her refreshed self-portrait in front of a "
            "live chat turn — with training on, that snapshot is produced by the train "
            "cycle after a passing probe.\n\n"
            "Note the reverse was never true: aborting a training cycle does NOT discard "
            "the reflection. The run finalizes on disk before the watchdog is even asked "
            "to train, and every build is fit from scratch over the whole corpus, so an "
            "aborted cycle costs GPU time only — the next build picks the work up in full."
        )
        btn_row_run.addWidget(self.chk_train)

        self.chk_skip_validation = QCheckBox("Skip validation")
        self.chk_skip_validation.setChecked(True)
        self.chk_skip_validation.setToolTip(
            "Promote the trained LoRA adapter without running the regression probe "
            "(the capability / format / character-continuity / acute-retention gate "
            "and its pre-train baselines). Faster, and never false-rejects a legitimate "
            "cycle — but there is then no single-cycle damage tripwire. The adapter "
            "lineage still keeps a bad promotion reversible (reload the prior adapter). "
            "On by default for UI-initiated training; uncheck to gate the promotion."
        )
        btn_row_run.addWidget(self.chk_skip_validation)
        # Skip-validation gates the ADAPTER promotion, so it means nothing without a
        # training cycle; grey it out rather than leaving a live-looking no-op control.
        self.chk_train.toggled.connect(self.chk_skip_validation.setEnabled)
        self.chk_skip_validation.setEnabled(self.chk_train.isChecked())

        self.chk_fresh = QCheckBox("Include fresh chats")
        self.chk_fresh.setChecked(False)
        self.chk_fresh.setToolTip(
            "Also render the chats NOT yet eligible for weights — a reflected chat "
            "still inside the RAG-only window (< rag_only_window_h, ~24h) and a "
            "background-frozen (chat_reflected) one — as preview rows at LR "
            "multiplier 0, so their derived targets appear in the Training review tab "
            "(tagged ▷fresh) and a bad turn can be regenerated/repaired + locked ❄ "
            "BEFORE it ages into a real training cycle.\n\n"
            "With Train (LoRA) ON: the preview rows are appended to the real build's "
            "sft_render.jsonl (the trained corpus itself is unchanged).\n\n"
            "With Train (LoRA) OFF (training-lite): the run additionally writes a "
            "GPU-free PREVIEW snapshot of the next build's corpus "
            "(models/snapshots/preview-<ts>/ — nothing trains, no restart), so the "
            "review→repair pass can happen between reflecting the new chats and the "
            "real training cycle. The repairs lock their exchanges, and the real "
            "build then trains the repaired targets.\n\n"
            "A chat never reflected at all cannot appear either way — it has no "
            "derived target to review yet. Off by default."
        )
        btn_row_run.addWidget(self.chk_fresh)
        # Deliberately NOT greyed with Train (unlike Skip-validation): the train-less
        # path is the checkbox's main workflow — reflect without training, review and
        # repair the corpus (fresh chats included) off the preview snapshot, then run
        # the real cycle on the repaired targets.

        self.chk_regen_persona = QCheckBox("Regen persona")
        self.chk_regen_persona.setChecked(False)
        self.chk_regen_persona.setToolTip(
            "Force the persona-digest pass to regenerate at the end of this run, even "
            "if the persona evidence hasn't changed since the last digest. Normally the "
            "digest only regenerates when new persona evidence shifts its fingerprint; "
            "check this to rebuild the self-portrait from the current evidence on demand "
            "(e.g. after a prompt or token-budget change). Off by default."
        )
        btn_row_run.addWidget(self.chk_regen_persona)

        self.chk_apply_judge = QCheckBox("Apply judge")
        self.chk_apply_judge.setChecked(True)
        self.chk_apply_judge.setToolTip(
            "Criterion flip (ON by default): let the clean-base persona judge SET the "
            "trainable target (override the blind chooser) when its 'who I'm becoming' pick "
            "differs. Gated on digest maturity — numeric cross-session recurrence, not the "
            "model's label — so on a thin/narrow corpus it stays dormant (logged-only) and "
            "gains authority as the persona digest matures. When any target is overridden, "
            "validation (the regression probe) is forced regardless of Skip-validation. "
            "Uncheck = kill-switch: fall back to the blind chooser (logged-only)."
        )
        btn_row_run.addWidget(self.chk_apply_judge)

        self.chk_skip_branching = QCheckBox("Skip branching")
        self.chk_skip_branching.setChecked(False)
        self.chk_skip_branching.setToolTip(
            "Skip the branching phase of reflection: no counterfactual branch generation "
            "and no branch judge. The trainable target is resolved directly — the kept "
            "original reply on a 'keep' verdict, or the revised IDEAL when one was "
            "generated on a 'revise'. Faster (no branch GPU passes), at the cost of the "
            "'road-not-taken' exploration. Off by default."
        )
        btn_row_run.addWidget(self.chk_skip_branching)

        controls_layout.addLayout(btn_row_run)
        controls_layout.addLayout(btn_row_jobs)
        controls_layout.addLayout(btn_row_tools)
        right_layout.addLayout(controls_layout)

        # Use QSplitter to allow resizing Left (sessions) and Right (logs) panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

    def _on_temperature_changed(self, value: float) -> None:
        self.temperature = float(value)

    # ---------------------------------------------------------------- #
    # Font updates (called by main window)                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self.txt_output.setFont(font)

    # ---------------------------------------------------------------- #
    # Sleep entry points                                                #
    # ---------------------------------------------------------------- #

    def _on_sleep(self) -> None:
        if not self._can_start_reflection():
            return
        selected = self.lst_sessions.selectedItems()
        filenames = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected])
        if not filenames:
            QMessageBox.information(
                self, "Nothing to reflect on",
                "No selected chat sessions found. Please select one or more past sessions from the list on the left."
            )
            return
        self._begin_server_run(filenames)

    def _on_short_summary(self) -> None:
        """Start a consolidation-only dry-run preview on the selected chat(s)."""
        if not self._can_start_reflection():
            return
        selected = self.lst_sessions.selectedItems()
        filenames = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected])
        if not filenames:
            QMessageBox.information(
                self, "Nothing to summarize",
                "No selected chat sessions found. Please select one or more past "
                "sessions from the list on the left."
            )
            return
        self._begin_summary_run(filenames)

    def _on_dry_sleep(self) -> None:
        """Start a full dry-run reflection (consolidation + revision + branch)."""
        if not self._can_start_reflection():
            return
        selected = self.lst_sessions.selectedItems()
        filenames = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected])
        if not filenames:
            QMessageBox.information(
                self, "Nothing to reflect on",
                "No selected chat sessions found. Please select one or more past "
                "sessions from the list on the left."
            )
            return
        self._begin_dry_sleep_run(filenames)

    # ---------------------------------------------------------------- #
    # "Learn" — fetch a Wikipedia digest into server/til (manual debug) #
    # ---------------------------------------------------------------- #

    def _on_learn(self) -> None:
        """Fetch a Current events digest, then dry-run a learning reflection over it.

        Manual debug/research: streams Ava's reflection into the event log and shows
        the modifiers ([fact]/[ask]/[resolved]) she *would* produce — WITHOUT writing
        anything. Lets us see how she handles external information before any of it is
        allowed to touch RAG, memory, or the weights pipeline."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._til_worker is not None and self._til_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "A reflection, training, or persona-preview job is in progress. Try Learn "
                "once it finishes.",
            )
            return

        # A fresh Learn supersedes any not-yet-applied findings.
        self._til_pending_text = ""
        self._til_pending_date = ""
        self._til_pending_source = ""

        self._append_section_header("Learn — Wikipedia Current events (dry run)")
        self._append_text("[learn] Fetching yesterday's digest into server/til…\n")
        self._lbl_status.setText("Fetching Wikipedia digest…")
        self._til_stream_open = False

        worker = TilFetchWorker(client, parent=self)
        worker.digest_ready.connect(self._on_til_digest)
        worker.reflect_chunk.connect(self._on_til_reflect_chunk)
        worker.reflect_done.connect(self._on_til_reflect_done)
        worker.error_occurred.connect(self._on_til_error)
        worker.finished.connect(worker.deleteLater)
        self._til_worker = worker
        self._update_sleep_button()
        worker.start()

    def _on_til_digest(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._til_worker:
            return
        date = result.get("date", "?")
        self._til_pending_date = result.get("date", "")
        self._append_text(
            f"[learn] {date}: {result.get('chars', 0):,} chars, "
            f"{result.get('sources', 0)} source(s) → {result.get('path', '')}\n"
        )
        self._append_text(f"[learn] {result.get('source_url', '')}\n\n")
        preview = (result.get("preview") or "").strip() or "(empty digest)"
        indented = "\n".join("    " + ln for ln in preview.splitlines())
        self._append_text(indented + "\n\n")
        self._append_text(
            "[learn] Reflecting on this (dry run — nothing will be written):\n\n"
        )
        self._lbl_status.setText(f"Reflecting on the {date} digest (dry run)…")

    def _on_til_reflect_chunk(self, delta: str) -> None:
        if self.sender() is not None and self.sender() is not self._til_worker:
            return
        if delta:
            self._append_text(delta)
            self._til_stream_open = True

    def _on_til_reflect_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._til_worker:
            return
        self._til_worker = None
        self._finalize_learning_result(
            result, tag="learn", source="learn",
            complete_msg="Learn complete (dry run). Press Apply Learning to keep these findings.",
            empty_msg="Learn complete (dry run — Ava kept nothing, nothing to apply).",
        )

    def _finalize_learning_result(self, result: dict, *, tag: str, source: str,
                                  complete_msg: str, empty_msg: str) -> None:
        """Render a dry-run learning result and arm Apply Learning.

        Shared by all three flows (Learn over a news digest, Resolve Questions over
        looked-up articles, Wander over a random page): close any open stream, render the
        proposed modifiers, and set the status line. *source* ("learn"/"lookup"/"wander")
        decides where a later Apply goes — wander commits the exchange to the SFT learning
        dataset, the rest write live memory. For learn/lookup, Apply is armed only when
        the pass distilled findings (otherwise the memory write is a no-op); for **wander**
        the value is the exchange itself, so Apply is armed whenever any text was produced,
        findings or not."""
        if self._til_stream_open:
            self._append_text("\n")
            self._til_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            self._append_text(f"\n[{tag}] Reflection skipped: {skipped}\n")
            self._lbl_status.setText(f"{tag.capitalize()}: reflection skipped ({skipped}).")
            self._update_sleep_button()
            return

        report = result.get("report") or {}
        self._render_til_modifiers(report)

        # A pass that abandoned the WEIGHTS/RAG/RESOLVED contract outright (wrote an
        # essay, or never reached its answer) parses to zero modifiers — the same
        # rendering an uneventful page produces. The server names the difference; say
        # it where the operator is deciding whether to Apply.
        contract_problem = (result.get("contract_problem") or "").strip()
        if contract_problem:
            self._append_text(
                f"\n[{tag}] ⚠ The facts/asks pass FAILED its contract: "
                f"{contract_problem}. The zero modifiers above are a failed pass, "
                f"not a quiet page.\n")

        # Hold the cleaned reflection for "Apply learning". Learn/lookup arm only on real
        # findings (else the memory write is a no-op); wander arms on any produced text,
        # since Apply commits the exchange itself to the SFT dataset, findings or not.
        has_findings = bool(
            report.get("weights") or report.get("rag") or report.get("resolved")
        )
        text = (result.get("text") or "").strip()
        keep = bool(text) and (has_findings or source == "wander")
        if keep:
            self._til_pending_text = text
            self._til_pending_source = source
            self._lbl_status.setText(complete_msg)
        else:
            self._til_pending_text = ""
            self._til_pending_source = ""
            self._lbl_status.setText(empty_msg)
        self._update_sleep_button()

    def _render_til_modifiers(self, report: dict) -> None:
        """Render the parsed WEIGHTS/RAG/RESOLVED modifiers a real learning pass would route."""
        weights = report.get("weights") or []
        rag = report.get("rag") or []
        resolved = report.get("resolved") or []

        self._append_text(
            "\n── proposed modifiers (dry run — NOT written) ──\n"
        )
        if not (weights or rag or resolved):
            self._append_text("  (Ava kept nothing from this digest)\n\n")
            return

        self._append_text(f"WEIGHTS — {len(weights)} item(s):\n")
        for w in weights:
            kind = w.get("weights_kind", "fact")
            self._append_text(f"  [{kind}] {self._one_line(w.get('content', ''))}\n")
        if not weights:
            self._append_text("  (none)\n")

        self._append_text(f"RAG — {len(rag)} item(s):\n")
        for item in rag:
            kind = item.get("kind", "?")
            label = kind
            if kind == "ask":
                ask_kind = item.get("ask_kind")
                label = f"ask:{ask_kind}" if ask_kind else "ask"
            line = f"  [{label}] {self._one_line(item.get('content', ''))}"
            trigger = item.get("trigger")
            if kind == "fact" and trigger:
                line += f"  (trigger: {self._one_line(trigger, 80)})"
            self._append_text(line + "\n")
        if not rag:
            self._append_text("  (none)\n")

        if resolved:
            self._append_text(f"RESOLVED — {len(resolved)} item(s):\n")
            for r in resolved:
                q = self._one_line(r.get("question", ""), 120)
                a = self._one_line(r.get("answer", ""), 120)
                arrow = f" → {a}" if a else ""
                self._append_text(f"  [resolved] {q}{arrow}\n")
        self._append_text("\n")

    def _on_til_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._til_worker:
            return
        self._til_worker = None
        if self._til_stream_open:
            self._append_text("\n")
            self._til_stream_open = False
        self._append_text(f"[learn] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Learn failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Resolve Questions" — look up open [ask:search] questions          #
    # ---------------------------------------------------------------- #

    def _on_lookup(self) -> None:
        """Run the lookup loop: extract subjects → fetch articles → reflect (dry run)."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._til_lookup_worker is not None and self._til_lookup_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, or apply job is in progress. Try "
                "Resolve Questions once it finishes.",
            )
            return

        # A fresh lookup supersedes any not-yet-applied findings.
        self._til_pending_text = ""
        self._til_pending_date = datetime.now().strftime("%Y-%m-%d")
        self._til_pending_source = ""
        self._til_stream_open = False

        self._append_section_header("Resolve Questions — Wikipedia lookup (dry run)")
        self._append_text("[lookup] Collecting open [ask:search] questions…\n")
        self._lbl_status.setText("Resolving open questions…")

        worker = TilLookupWorker(client, parent=self)
        worker.collected.connect(self._on_lookup_collected)
        worker.log.connect(self._on_lookup_log)
        worker.subjects.connect(self._on_lookup_subjects)
        worker.fetched.connect(self._on_lookup_fetched)
        worker.reflect_chunk.connect(self._on_lookup_chunk)
        worker.reflect_done.connect(self._on_lookup_done)
        worker.error_occurred.connect(self._on_lookup_error)
        worker.finished.connect(worker.deleteLater)
        self._til_lookup_worker = worker
        self._update_sleep_button()
        worker.start()

    def _lookup_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._til_lookup_worker

    def _on_lookup_collected(self, msg: dict) -> None:
        if not self._lookup_is_current():
            return
        n = msg.get("count", 0)
        if not n:
            self._append_text("[lookup] No open [ask:search] questions to resolve.\n")
            return
        self._append_text(f"[lookup] {n} open question(s) to look up:\n")
        for q in msg.get("questions") or []:
            self._append_text(f"    • {self._one_line(q)}\n")
        self._lbl_status.setText(f"Looking up {n} question(s) — extracting subjects…")

    def _on_lookup_log(self, msg: dict) -> None:
        if not self._lookup_is_current():
            return
        self._append_text(f"[lookup] {msg.get('message', '')}\n")

    def _on_lookup_subjects(self, msg: dict) -> None:
        if not self._lookup_is_current():
            return
        subjects = msg.get("subjects") or []
        bound = msg.get("bound_count", 0)
        n = len(subjects)
        if bound >= n:
            head = f"[lookup] {n} subject(s) from bound titles (no model swap):"
        elif bound:
            head = (f"[lookup] {n} subject(s) — {bound} bound, "
                    f"{n - bound} extracted on the clean base:")
        else:
            head = f"[lookup] Extracted {n} subject(s) on the clean base:"
        self._append_text(head + "\n")
        for s in subjects:
            self._append_text(f"    → {s}\n")
        self._lbl_status.setText("Fetching Wikipedia articles…")

    def _on_lookup_fetched(self, msg: dict) -> None:
        if not self._lookup_is_current():
            return
        subject = msg.get("subject", "")
        if msg.get("missing"):
            self._append_text(f"[lookup] ✗ no article for {subject!r}\n")
            return
        title = msg.get("title", "")
        via = msg.get("via", "")
        chars = msg.get("chars", 0)
        redir = msg.get("redirected_from")
        rtag = f" (redirected from {redir!r})" if redir else ""
        head = f"{subject!r} → {title!r}" if title.lower() != subject.lower() else f"{title!r}"
        self._append_text(f"[lookup] ✓ {head} via {via}{rtag}: {chars:,} chars\n")

    def _on_lookup_chunk(self, delta: str) -> None:
        if not self._lookup_is_current():
            return
        if delta:
            self._append_text(delta)
            self._til_stream_open = True

    def _on_lookup_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._til_lookup_worker:
            return
        self._til_lookup_worker = None
        self._finalize_learning_result(
            result, tag="lookup", source="lookup",
            complete_msg="Look-up complete (dry run). Press Apply Learning to keep these findings.",
            empty_msg="Look-up complete (dry run — nothing resolved or kept).",
        )

    def _on_lookup_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._til_lookup_worker:
            return
        self._til_lookup_worker = None
        if self._til_stream_open:
            self._append_text("\n")
            self._til_stream_open = False
        self._append_text(f"[lookup] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Look-up failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Wander" — reflect on a random approved-wiki page                  #
    # ---------------------------------------------------------------- #

    def _on_wander(self) -> None:
        """Reflect on a random page from a user-approved wiki (dry run)."""
        self._start_wander()

    def _on_visit(self) -> None:
        """Reflect on a user-supplied URL using the wander sequence."""
        url = self.edt_visit_url.text().strip() if hasattr(self, "edt_visit_url") else ""
        if not url:
            return
        self._start_wander(url=url)

    def _start_wander(self, url: str = "") -> None:
        """Start either a random wander or a direct URL visit."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._til_wander_worker is not None and self._til_wander_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_lookup_worker is not None and self._til_lookup_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, or apply job is in progress. Try "
                "again once it finishes.",
            )
            return

        # A fresh wander supersedes any not-yet-applied findings.
        self._til_pending_text = ""
        self._til_pending_date = datetime.now().strftime("%Y-%m-%d")
        self._til_pending_source = ""
        self._til_stream_open = False

        if url:
            self._append_section_header("Visit — supplied page (dry run)")
            self._append_text(f"[visit] Fetching {url}\n")
            self._lbl_status.setText("Visiting…")
        else:
            self._append_section_header("Wander — random approved-wiki page (dry run)")
            self._append_text("[wander] Picking a random page from an approved wiki…\n")
            self._lbl_status.setText("Wandering…")

        worker = TilWanderWorker(client, url=url, parent=self)
        worker.wandered.connect(self._on_wandered)
        worker.reflect_chunk.connect(self._on_wander_chunk)
        worker.reflect_done.connect(self._on_wander_done)
        worker.error_occurred.connect(self._on_wander_error)
        worker.finished.connect(worker.deleteLater)
        self._til_wander_worker = worker
        self._update_sleep_button()
        worker.start()

    def _wander_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._til_wander_worker

    def _on_wandered(self, msg: dict) -> None:
        if not self._wander_is_current():
            return
        wiki = msg.get("wiki", "?")
        lang = msg.get("lang", "")
        title = msg.get("title", "?")
        url = msg.get("url", "")
        chars = msg.get("chars", 0)
        prefix = "visit" if msg.get("mode") == "visit" else "wander"
        tag = f"{wiki} ({lang})" if lang else wiki
        self._append_text(f"[{prefix}] {tag}: {title!r} — {chars:,} chars\n")
        if url:
            self._append_text(f"[{prefix}] {url}\n")
        self._append_text(f"[{prefix}] Reflecting (dry run — nothing will be written):\n\n")
        self._lbl_status.setText(f"Reflecting on {title!r} (dry run)…")

    def _on_wander_chunk(self, delta: str) -> None:
        if not self._wander_is_current():
            return
        if delta:
            self._append_text(delta)
            self._til_stream_open = True

    def _on_wander_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._til_wander_worker:
            return
        self._til_wander_worker = None
        self._finalize_learning_result(
            result, tag="wander", source="wander",
            complete_msg="Wander complete. Press Apply Learning to add this exchange to "
                         "the SFT learning dataset.",
            empty_msg="Wander complete — no text produced, nothing to apply.",
        )

    def _on_wander_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._til_wander_worker:
            return
        self._til_wander_worker = None
        if self._til_stream_open:
            self._append_text("\n")
            self._til_stream_open = False
        self._append_text(f"[wander] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Wander failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Reach Out" — manual Ava-initiated outreach decision (debug)       #
    # ---------------------------------------------------------------- #

    def _on_outreach(self) -> None:
        """Manually run one outreach decision pass, streaming Ava's reasoning."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._outreach_worker is not None and self._outreach_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_lookup_worker is not None and self._til_lookup_worker.isRunning())
                   or (self._til_wander_worker is not None and self._til_wander_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._synthesis_worker is not None and self._synthesis_worker.isRunning())
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, or apply job is in progress. Try "
                "again once it finishes.",
            )
            return

        self._outreach_stream_open = False
        self._outreach_reasoning_header = False
        self._append_section_header("Reach Out — Ava-initiated outreach decision (debug)")
        self._append_text(
            "[outreach] Asking Ava whether she wants to raise one of her open "
            "questions right now…\n")
        self._lbl_status.setText("Deciding whether to reach out…")

        worker = OutreachWorker(client, parent=self)
        worker.question.connect(self._on_outreach_question)
        worker.prompt_debug.connect(self._on_outreach_prompt)
        worker.chunk.connect(self._on_outreach_chunk)
        worker.done.connect(self._on_outreach_done)
        worker.error_occurred.connect(self._on_outreach_error)
        worker.finished.connect(worker.deleteLater)
        self._outreach_worker = worker
        self._update_sleep_button()
        worker.start()

    def _outreach_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._outreach_worker

    # ---------------------------------------------------------------- #
    # "Dedup facts" — clean-base semantic [fact] dedup (manual)          #
    # ---------------------------------------------------------------- #

    def _on_fact_dedup(self) -> None:
        """Preview, then optionally apply, a semantic dedup of the live [fact] store.

        The manual sibling of the pass a reflection run now performs at the end of its
        clean-base window. Always previews first: a merge evicts live records, and unlike
        the run's version (which stages, and so is discarded with a discarded run) this
        one writes straight to live memory. It is reversible — the op-log is append-only
        — but "reversible by editing a JSONL" is not a reason to skip the confirmation.
        """
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._fact_dedup_worker is not None and self._fact_dedup_worker.isRunning():
            return
        if self._any_job_running():
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, or apply job is in progress. Try "
                "again once it finishes.",
            )
            return

        self._append_section_header("Dedup facts — semantic [fact] merge (clean base)")
        self._append_text(
            "[dedup] Grouping live facts by meaning on the clean base — dry run, "
            "nothing will be written yet…\n")
        self._lbl_status.setText("Deduplicating facts (dry run)…")
        self._start_fact_dedup(client, dry_run=True)

    def _start_fact_dedup(self, client, *, dry_run: bool) -> None:
        worker = DedupFactsWorker(client, dry_run)
        worker.progress.connect(self._on_fact_dedup_progress)
        worker.dedup_done.connect(self._on_fact_dedup_done)
        worker.error_occurred.connect(self._on_fact_dedup_error)
        worker.finished.connect(worker.deleteLater)
        self._fact_dedup_worker = worker
        self._update_sleep_button()
        worker.start()

    def _on_fact_dedup_progress(self, ev: dict) -> None:
        """Render the grouping as it advances.

        The pass is dozens of sequential clean-base calls over a real store, and it used
        to print nothing between the button press and the result — so a slow run and a
        hung one looked identical, and a zero-merge result carried no evidence of what had
        actually been compared.
        """
        stage = ev.get("stage")
        if stage == "clustered":
            self._append_text(
                f"[dedup] {ev.get('facts', 0)} live fact(s) → {ev.get('subjects', 0)} "
                f"subject(s); {ev.get('multi', 0)} hold more than one fact "
                f"({ev.get('candidates', 0)} candidate(s) to group), "
                f"{ev.get('singletons', 0)} are one of a kind.\n")
        elif stage == "block":
            flag = "  ⚠ rejected (one blob) — kept apart" if ev.get("rejected") else ""
            self._append_text(
                f"    block {ev.get('i', 0)}/{ev.get('n', 0)}: {ev.get('items', 0)} "
                f"fact(s) → {ev.get('groups', 0)} group(s), "
                f"{ev.get('merged', 0)} merge(s){flag}\n")

    def _on_fact_dedup_done(self, result: dict) -> None:
        self._fact_dedup_worker = None
        skipped = result.get("skipped")
        if skipped:
            self._append_text(f"[dedup] Skipped ({skipped}): "
                              f"{result.get('message', '')}\n")
            self._lbl_status.setText("Idle")
            self._update_sleep_button()
            return

        groups = result.get("groups") or []
        before = result.get("before", 0)
        if result.get("dry_run"):
            if not groups:
                self._append_text(
                    f"[dedup] {before} live fact(s); no paraphrase groups found — "
                    "nothing to merge.\n")
                self._lbl_status.setText("Idle")
                self._update_sleep_button()
                return
            evicted = sum(len(g.get("evicted") or []) for g in groups)
            for g in groups:
                self._append_text(f"  keep: {g.get('survivor', '')}\n")
                for e in g.get("evicted") or []:
                    self._append_text(f"    drop: {e}\n")
                trigger = g.get("merged_trigger")
                if trigger:
                    self._append_text(f"    recalled when: {trigger}\n")
            self._append_text(
                f"[dedup] {len(groups)} group(s) would merge, evicting {evicted} of "
                f"{before} fact(s).\n")
            self._lbl_status.setText("Idle")
            self._update_sleep_button()
            reply = QMessageBox.question(
                self, "Apply fact dedup?",
                f"Apply this merge?\n\n{len(groups)} group(s) collapse, evicting "
                f"{evicted} of {before} live fact(s). Each surviving fact inherits the "
                f"union of its group's recall cues, so nothing becomes unfindable.\n\n"
                "This writes to live memory (append-only, so it is reversible by "
                "dropping the lines it adds) and refreshes recall. The model is "
                "swapped again, so it takes a while.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._append_text("[dedup] Not applied.\n")
                return
            self._append_text("[dedup] Applying — re-running on the clean base…\n")
            self._lbl_status.setText("Applying fact dedup…")
            self._start_fact_dedup(self._chat_widget._client, dry_run=False)
            return

        after = result.get("after", before)
        self._append_text(
            f"[dedup] Applied — {result.get('evicted', 0)} paraphrase(s) evicted; "
            f"{before} → {after} live fact(s). Recall refreshed.\n")
        self._lbl_status.setText("Idle")
        self._update_sleep_button()

    def _on_fact_dedup_error(self, message: str) -> None:
        self._fact_dedup_worker = None
        self._append_text(f"[dedup] Failed: {message}\n")
        self._lbl_status.setText("Idle")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Persona preview" — non-mutating prompt self-review               #
    # ---------------------------------------------------------------- #

    def _on_persona_preview(self) -> None:
        """Ask Ava to preview whether her standing prompt should change."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if (self._persona_preview_worker is not None
                and self._persona_preview_worker.isRunning()):
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_lookup_worker is not None and self._til_lookup_worker.isRunning())
                   or (self._til_wander_worker is not None and self._til_wander_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._outreach_worker is not None and self._outreach_worker.isRunning())
                   or (self._synthesis_worker is not None and self._synthesis_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, outreach, or apply job is in progress. "
                "Try again once it finishes.",
            )
            return

        self._persona_preview_stream_open = False
        self._append_section_header("Persona preview — prompt self-review (dry run)")
        self._append_text(
            "[persona preview] Showing Ava her current chat prompt, persona "
            "self-presentation, and logged prompt deltas. This writes nothing.\n")
        self._lbl_status.setText("Running persona preview…")

        worker = PersonaPreviewWorker(client, temperature=self.temperature, parent=self)
        worker.context.connect(self._on_persona_preview_context)
        worker.chunk.connect(self._on_persona_preview_chunk)
        worker.done.connect(self._on_persona_preview_done)
        worker.error_occurred.connect(self._on_persona_preview_error)
        worker.finished.connect(worker.deleteLater)
        self._persona_preview_worker = worker
        self._update_sleep_button()
        worker.start()

    def _persona_preview_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._persona_preview_worker

    def _on_persona_preview_context(self, msg: dict) -> None:
        if not self._persona_preview_is_current():
            return
        prompt_chars = msg.get("current_prompt_chars", 0)
        persona_chars = msg.get("persona_chars", 0)
        delta_chars = msg.get("prompt_delta_chars", 0)
        input_tokens = msg.get("input_tokens")
        input_budget = msg.get("input_token_budget")
        output_reserve = msg.get("output_token_reserve")
        tier = msg.get("clip_tier")
        budget = ""
        if input_tokens is not None and input_budget is not None and output_reserve is not None:
            budget = (f"; input {input_tokens:,}/{input_budget:,} tokens, "
                      f"output reserve {output_reserve:,}")
            if tier:
                budget += f", clipped tier {tier}"
        self._append_text(
            "[persona preview] Context assembled: "
            f"prompt {prompt_chars:,} chars, persona {persona_chars:,} chars, "
            f"prompt deltas {delta_chars:,} chars{budget}.\n\n")

    def _on_persona_preview_chunk(self, delta: str) -> None:
        if not self._persona_preview_is_current():
            return
        if delta:
            self._append_text(delta)
            self._persona_preview_stream_open = True

    def _on_persona_preview_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._persona_preview_worker:
            return
        self._persona_preview_worker = None
        had_stream = self._persona_preview_stream_open
        if had_stream:
            self._append_text("\n")
            self._persona_preview_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or skipped
            self._append_text(f"\n[persona preview] Skipped: {reason}\n")
            self._lbl_status.setText("Persona preview skipped.")
        elif result.get("error"):
            self._append_text(f"\n[persona preview] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Persona preview failed.")
        else:
            if not had_stream:
                text = (result.get("text") or "").strip()
                if text:
                    self._append_text(text + "\n")
            if result.get("truncated"):
                self._append_text(
                    "\n[persona preview] Warning: generation hit the token cap; "
                    "the recommendation may be incomplete.\n")
            self._append_text(
                "\n[persona preview] Complete. No prompt was changed; actual prompt "
                "mutation remains a separate task.\n")
            self._lbl_status.setText("Persona preview complete — no prompt changed.")
        self._update_sleep_button()

    def _on_persona_preview_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._persona_preview_worker:
            return
        self._persona_preview_worker = None
        if self._persona_preview_stream_open:
            self._append_text("\n")
            self._persona_preview_stream_open = False
        self._append_text(f"[persona preview] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Persona preview failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Reconcile self" — soften outgrown persona / stale facts          #
    # ---------------------------------------------------------------- #

    def _on_reconcile(self) -> None:
        """Two-step self-reconciliation: run a clean-base PREVIEW (writes nothing),
        show which persona/fact items Ava would set aside, apply only on confirm."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._reconcile_worker is not None and self._reconcile_worker.isRunning():
            return
        resp = QMessageBox.information(
            self,
            "Reconcile self (preview)",
            "This asks Ava, on the clean base (adapter off), to judge her live persona "
            "and fact memories against who she has become — her current persona digest — "
            "and flag the ones she has grown past or that are now stale.\n\n"
            "It runs a PREVIEW first: nothing is changed until you confirm. Softening is "
            "'set aside, not delete' (kept as evidence-of-change) and append-only, so it "
            "is reversible.\n\n"
            "The memory is judged in batches on the clean base, so with a large memory "
            "this can take many minutes (watch server.log '[reconcile]' lines for "
            "per-batch progress). The window may appear busy meanwhile.\n\n"
            "Run preview?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if resp != QMessageBox.StandardButton.Ok:
            return
        self._start_reconcile(dry_run=True)

    def _start_reconcile(self, *, dry_run: bool) -> None:
        client = self._chat_widget._client
        self._append_section_header(
            "Reconcile self — persona/fact cleanup against the current self"
            + ("" if dry_run else " (applying)"))
        self._append_text(
            "[reconcile] Applying — softening the flagged items…\n" if not dry_run
            else "[reconcile] Sent — swapping to the clean base (per-batch progress "
                 "will stream below as each batch is judged)…\n")
        self._lbl_status.setText(
            "Reconciling self (applying)…" if not dry_run
            else "Reconciling self (preview — swapping to the clean base)…")

        worker = ReconcileSelfWorker(client, dry_run, parent=self)
        worker.progress.connect(self._on_reconcile_progress)
        worker.reconcile_done.connect(self._on_reconcile_done)
        worker.error_occurred.connect(self._on_reconcile_error)
        worker.finished.connect(worker.deleteLater)
        self._reconcile_worker = worker
        self._update_sleep_button()
        worker.start()

    def _on_reconcile_progress(self, ev: dict) -> None:
        """Render streamed batch progress into the event log as it arrives."""
        if self.sender() is not None and self.sender() is not self._reconcile_worker:
            return
        stage = ev.get("stage")
        if stage == "start":
            n = ev.get("batches", 0)
            self._append_text(
                f"[reconcile] Judging {ev.get('items', 0)} live item(s) "
                f"({ev.get('n_persona', 0)} persona, {ev.get('n_fact', 0)} fact) "
                f"against the current self, in {n} batch(es) of up to "
                f"{ev.get('batch_size', 0)} on the clean base…\n")
            self._lbl_status.setText(f"Reconciling — 0/{n} batches…")
        elif stage == "batch":
            i, n = ev.get("i", 0), ev.get("n", 0)
            total = ev.get("running_total", 0)
            self._append_text(
                f"  batch {i}/{n}: {ev.get('n_items', 0)} judged → "
                f"{ev.get('n_supersede', 0)} to set aside "
                f"(running total: {total})\n")
            self._lbl_status.setText(f"Reconciling — {i}/{n} batches, {total} flagged…")

    def _on_reconcile_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._reconcile_worker:
            return
        self._reconcile_worker = None

        skipped = result.get("skipped")
        if skipped:
            msg = result.get("message") or skipped
            self._append_text(f"[reconcile] Skipped: {msg}\n")
            self._lbl_status.setText(f"Reconcile skipped: {msg}")
            QMessageBox.information(self, "Reconcile self", msg)
            self._update_sleep_button()
            return

        report = result.get("report")
        before = result.get("before", 0)
        note = result.get("note")

        if result.get("dry_run"):
            if not report or report.get("superseded", 0) == 0:
                reason = note or "nothing to set aside — all items still fit"
                self._append_text(f"[reconcile] Preview — {reason}.\n")
                self._lbl_status.setText(f"Reconcile preview — {reason}.")
                QMessageBox.information(
                    self, "Reconcile self",
                    f"No items flagged among {before} live persona/fact item(s) "
                    f"({reason}).")
                self._update_sleep_button()
                return
            self._append_text(self._format_reconcile_preview(report))
            n = report.get("superseded", 0)
            confirm = QMessageBox.question(
                self,
                "Apply reconciliation?",
                f"Ava would set aside {n} of {before} live persona/fact item(s) "
                f"(details in the panel above).\n\n"
                "Soften these (supersede, not delete) and reload RAG? They stay as "
                "evidence-of-change and the op-log is append-only, so this is "
                "reversible.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._start_reconcile(dry_run=False)
            else:
                self._append_text("[reconcile] Preview shown — not applied.\n")
                self._lbl_status.setText("Reconcile preview shown — not applied.")
                self._update_sleep_button()
            return

        # Applied.
        n = result.get("superseded", 0)
        after = result.get("after", "?")
        self._append_text(self._format_reconcile_preview(report, applied=True))
        self._append_text(
            f"[reconcile] Applied — {n} item(s) set aside, {after} remaining live. "
            "RAG was reloaded in place.\n")
        self._lbl_status.setText(
            f"Reconcile applied — {n} set aside, {after} remaining.")
        QMessageBox.information(
            self, "Reconcile complete",
            f"Set aside {n} item(s) ({before} → {after} live persona/fact items). "
            "They are kept as evidence-of-change and dropped from recall + active "
            "evidence/training; the change is reversible via the append-only op-log.")
        self._update_sleep_button()

    def _on_reconcile_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._reconcile_worker:
            return
        self._reconcile_worker = None
        self._append_text(f"[reconcile] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Reconcile failed: {err}")
        QMessageBox.critical(
            self, "Reconcile failed",
            f"The reconciliation did not complete:\n\n{err}")
        self._update_sleep_button()

    @staticmethod
    def _format_reconcile_preview(report: dict, *, applied: bool = False) -> str:
        header = ("Set aside (applied):" if applied
                  else "Proposed to set aside (preview — not yet applied):")
        lines = [header, ""]
        persona = report.get("persona") or []
        facts = report.get("facts") or []
        if persona:
            lines.append("Persona (grown past):")
            for p in persona:
                lines.append(f"  - {p.get('content', '')}")
                reason = (p.get("reason") or "").strip()
                if reason:
                    lines.append(f"      reason: {reason}")
            lines.append("")
        if facts:
            lines.append("Facts (now stale / contradicted):")
            for f in facts:
                lines.append(f"  - {f.get('content', '')}")
                trigger = (f.get("trigger") or "").strip()
                if trigger:
                    lines.append(f"      (recalled when: {trigger})")
                reason = (f.get("reason") or "").strip()
                if reason:
                    lines.append(f"      reason: {reason}")
            lines.append("")
        kept = report.get("kept")
        if kept is not None:
            lines.append(f"Kept: {kept} item(s) still fit who she is.")
            lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # "Resolve fact conflicts" — supersede stale contradicted facts     #
    # ---------------------------------------------------------------- #

    def _on_contradict(self) -> None:
        """Two-step contradiction resolution: clean-base PREVIEW, apply on confirm."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.")
            return
        if self._contradict_worker is not None and self._contradict_worker.isRunning():
            return
        resp = QMessageBox.information(
            self,
            "Resolve fact conflicts (preview)",
            "This finds live facts that CONTRADICT each other — the case where a "
            "correction wrote a new fact instead of replacing the stale one. It groups "
            "facts by subject, asks the clean base which directly conflict, and keeps "
            "the newest (the correction).\n\n"
            "It runs a PREVIEW first: nothing changes until you confirm. Softening is "
            "'set aside, not delete' and append-only, so reversible.\n\n"
            "Judged per subject group on the clean base, so a large memory can take "
            "several minutes (per-group progress streams below).\n\n"
            "Run preview?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if resp != QMessageBox.StandardButton.Ok:
            return
        self._start_contradict(dry_run=True)

    def _start_contradict(self, *, dry_run: bool) -> None:
        client = self._chat_widget._client
        self._append_section_header(
            "Resolve fact conflicts — supersede stale contradicted facts"
            + ("" if dry_run else " (applying)"))
        self._append_text(
            "[contradict] Applying — softening the stale facts…\n" if not dry_run
            else "[contradict] Sent — clustering facts by subject; per-group progress "
                 "will stream below…\n")
        self._lbl_status.setText(
            "Resolving fact conflicts (applying)…" if not dry_run
            else "Resolving fact conflicts (preview — swapping to the clean base)…")

        worker = ResolveContradictionsWorker(client, dry_run, parent=self)
        worker.progress.connect(self._on_contradict_progress)
        worker.done.connect(self._on_contradict_done)
        worker.error_occurred.connect(self._on_contradict_error)
        worker.finished.connect(worker.deleteLater)
        self._contradict_worker = worker
        self._update_sleep_button()
        worker.start()

    def _on_contradict_progress(self, ev: dict) -> None:
        if self.sender() is not None and self.sender() is not self._contradict_worker:
            return
        stage = ev.get("stage")
        if stage == "start":
            n = ev.get("groups", 0)
            self._append_text(
                f"[contradict] {ev.get('facts', 0)} live fact(s) → {n} multi-fact "
                f"subject group(s) to check on the clean base…\n")
            self._lbl_status.setText(f"Resolving conflicts — 0/{n} groups…")
        elif stage == "group":
            i, n = ev.get("i", 0), ev.get("n", 0)
            total = ev.get("running_total", 0)
            nsup = ev.get("n_superseded", 0)
            if nsup:
                self._append_text(
                    f"  group {i}/{n}: {ev.get('group_size', 0)} facts → "
                    f"{nsup} superseded (running total: {total})\n")
            self._lbl_status.setText(
                f"Resolving conflicts — {i}/{n} groups, {total} superseded…")

    def _on_contradict_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._contradict_worker:
            return
        self._contradict_worker = None

        skipped = result.get("skipped")
        if skipped:
            m = result.get("message") or skipped
            self._append_text(f"[contradict] Skipped: {m}\n")
            self._lbl_status.setText(f"Resolve conflicts skipped: {m}")
            QMessageBox.information(self, "Resolve fact conflicts", m)
            self._update_sleep_button()
            return

        report = result.get("report")
        before = result.get("before", 0)
        note = result.get("note")

        if result.get("dry_run"):
            if not report or report.get("superseded", 0) == 0:
                reason = note or "no contradictions found — all facts are consistent"
                self._append_text(f"[contradict] Preview — {reason}.\n")
                self._lbl_status.setText(f"Resolve conflicts — {reason}.")
                QMessageBox.information(
                    self, "Resolve fact conflicts",
                    f"No conflicts found among {before} live fact(s) ({reason}).")
                self._update_sleep_button()
                return
            self._append_text(self._format_contradict_preview(report))
            n = report.get("superseded", 0)
            confirm = QMessageBox.question(
                self, "Apply resolution?",
                f"Found {n} stale fact(s) contradicted by a newer one among {before} "
                f"fact(s) (details above).\n\n"
                "Supersede the stale ones (keeping the correction) and reload RAG? "
                "Softening is reversible via the append-only op-log.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._start_contradict(dry_run=False)
            else:
                self._append_text("[contradict] Preview shown — not applied.\n")
                self._lbl_status.setText("Resolve conflicts preview shown — not applied.")
                self._update_sleep_button()
            return

        n = result.get("superseded", 0)
        after = result.get("after", "?")
        self._append_text(self._format_contradict_preview(report, applied=True))
        self._append_text(
            f"[contradict] Applied — {n} stale fact(s) superseded, {after} remaining. "
            "RAG was reloaded in place.\n")
        self._lbl_status.setText(
            f"Resolve conflicts applied — {n} superseded, {after} remaining.")
        QMessageBox.information(
            self, "Resolve complete",
            f"Superseded {n} stale fact(s) ({before} → {after} live facts). The "
            "corrections were kept; the stale ones are set aside as evidence-of-change "
            "and dropped from recall + training. Reversible via the op-log.")
        self._update_sleep_button()

    def _on_contradict_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._contradict_worker:
            return
        self._contradict_worker = None
        self._append_text(f"[contradict] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Resolve conflicts failed: {err}")
        QMessageBox.critical(
            self, "Resolve failed",
            f"The conflict resolution did not complete:\n\n{err}")
        self._update_sleep_button()

    @staticmethod
    def _format_contradict_preview(report: dict, *, applied: bool = False) -> str:
        header = ("Superseded (applied):" if applied
                  else "Proposed to supersede (preview — not yet applied):")
        lines = [header, ""]
        for it in (report.get("items") or []):
            lines.append(f"  stale : {it.get('content', '')}")
            trigger = (it.get("trigger") or "").strip()
            if trigger:
                lines.append(f"          (about: {trigger})")
            lines.append(f"  keep  : {it.get('survivor', '')}")
            lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # "Persona digest (dry)" — re-derive the self-portrait, write none  #
    # ---------------------------------------------------------------- #

    def _on_digest_dryrun(self) -> None:
        """Re-derive the persona digest end to end and print it. Writes nothing.

        Themes are grouped on the clean base (a judgement, not self-expression) and the
        portrait is written on the adapter (the voice stays hers). There is no confirm /
        apply step because the server persists nothing — the point is to read the result
        before deciding whether this path should replace the live one.
        """
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._digest_dry_worker is not None and self._digest_dry_worker.isRunning():
            return
        resp = QMessageBox.information(
            self,
            "Persona digest (dry run)",
            "This re-derives Ava's self-portrait from her live [persona] statements:\n\n"
            "  1. Her statements are grouped into themes on the CLEAN BASE (adapter "
            "off), in fixed blocks with a merge pass over the results — so a large "
            "persona store is handled in many small judgements instead of one prompt "
            "that cannot hold it.\n"
            "  2. The portrait is then written on the ADAPTER, so the voice is hers.\n\n"
            "NOTHING IS SAVED: no digest snapshot, no change to the active digest, no "
            "change to recall. The themes and the portrait are printed below.\n\n"
            "The model is swapped twice and the grouping is many sequential calls, so "
            "this can take a while; the window may appear busy meanwhile.\n\n"
            "Run it?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if resp != QMessageBox.StandardButton.Ok:
            return

        self._append_section_header(
            "Persona digest (DRY RUN) — cluster on the clean base, synthesize on the adapter")
        self._append_text(
            "[digest-dry] Sent — gathering live persona evidence, then swapping to the "
            "clean base to group it (per-block progress streams below)…\n")
        self._lbl_status.setText("Persona digest (dry) — swapping to the clean base…")

        worker = DigestDryRunWorker(client, self._digest_block_size(),
                                    self.temperature, parent=self)
        worker.progress.connect(self._on_digest_dryrun_progress)
        worker.chunk.connect(self._on_digest_dryrun_chunk)
        worker.done.connect(self._on_digest_dryrun_done)
        worker.error_occurred.connect(self._on_digest_dryrun_error)
        worker.finished.connect(worker.deleteLater)
        self._digest_dry_worker = worker
        self._update_sleep_button()
        worker.start()

    @staticmethod
    def _digest_block_size() -> int:
        """Statements per grouping call. The server's own default (40) is the tuned
        value; kept as a hook so a box with a tighter context can lower it."""
        return 40

    def _on_digest_dryrun_progress(self, ev: dict) -> None:
        """Render clustering progress as it streams in."""
        if self.sender() is not None and self.sender() is not self._digest_dry_worker:
            return
        stage = ev.get("stage")
        if stage == "gathered":
            self._append_text(
                f"[digest-dry] {ev.get('items', 0)} live persona statement(s) to group, "
                f"in blocks of {ev.get('block_size', 0)}.\n")
            self._lbl_status.setText(
                f"Persona digest (dry) — {ev.get('items', 0)} statements…")
        elif stage == "clean_base_enter":
            self._append_text(
                "[digest-dry] Swapping the adapter out (two full model reloads)…\n")
        elif stage == "map":
            i, n = ev.get("i", 0), ev.get("n", 0)
            flag = "  ⚠ rejected (looked like one blob) — kept apart" if ev.get("rejected") else ""
            self._append_text(
                f"  block {i}/{n}: {ev.get('items', 0)} statements → "
                f"{ev.get('themes', 0)} theme(s){flag}\n")
            self._lbl_status.setText(
                f"Persona digest (dry) — grouping block {i}/{n}…")
        elif stage == "reduce_started":
            self._append_text(
                f"  merge round {ev.get('round', 0)}: re-grouping "
                f"{ev.get('themes', 0)} theme(s) across blocks…\n")
            self._lbl_status.setText(
                f"Persona digest (dry) — merge round {ev.get('round', 0)}…")
        elif stage == "reduce":
            self._append_text(
                f"  merge round {ev.get('round', 0)}: {ev.get('before', 0)} → "
                f"{ev.get('after', 0)} theme(s) ({ev.get('merged', 0)} merged)\n")
        elif stage == "clustered":
            self._append_text(
                f"[digest-dry] Clustered → {ev.get('themes', 0)} theme(s) "
                f"({ev.get('map_themes', 0)} after blocks, before merging) in "
                f"{ev.get('calls', 0)} model call(s); "
                f"{ev.get('rejected_blocks', 0)} block(s) rejected by the blob guard.\n")
        elif stage == "polarity_split":
            self._append_text(
                f"  ⇄ polarity: {ev.get('opposed', 0)} opposing statement(s) split out "
                f"of theme “{ev.get('theme', '')}”\n")
        elif stage == "polarity":
            self._append_text(
                f"[digest-dry] Polarity screen: {ev.get('themes_screened', 0)} merged "
                f"theme(s) checked, {ev.get('themes_split', 0)} split "
                f"({ev.get('members_flagged', 0)} opposing statement(s))."
                + (f" {ev.get('calls_failed', 0)} check(s) failed.\n"
                   if ev.get("calls_failed") else "\n"))
        elif stage == "synthesizing":
            faded = ev.get("themes_faded", 0)
            tail = f" ({faded} faded below the prompt floor and dropped)" if faded else ""
            self._append_text(
                f"[digest-dry] Adapter back on — writing the portrait from "
                f"{ev.get('themes_in_prompt', 0)} theme(s){tail}:\n\n")
            self._lbl_status.setText("Persona digest (dry) — writing the portrait…")

    def _on_digest_dryrun_chunk(self, text: str) -> None:
        if self.sender() is not None and self.sender() is not self._digest_dry_worker:
            return
        self._append_text(text)

    def _on_digest_dryrun_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._digest_dry_worker:
            return
        self._digest_dry_worker = None

        skipped = result.get("skipped")
        if skipped:
            m = result.get("message") or skipped
            self._append_text(f"\n[digest-dry] Skipped: {m}\n")
            self._lbl_status.setText(f"Persona digest (dry) skipped: {m}")
            QMessageBox.information(self, "Persona digest (dry run)", m)
            self._update_sleep_button()
            return

        self._append_text("\n" + self._format_digest_dryrun(result))
        stats = result.get("stats") or {}
        themes = result.get("themes") or []
        self._lbl_status.setText(
            f"Persona digest (dry) — {result.get('items', 0)} statements → "
            f"{len(themes)} themes. Nothing was saved.")

        cur = result.get("current")
        cur_line = ""
        if cur:
            cur_line = (f"\n\nThe active digest has {cur.get('themes', 0)} theme(s) "
                        f"(largest cluster {cur.get('largest_cluster', 0)}).")
        QMessageBox.information(
            self, "Persona digest (dry run)",
            f"{result.get('items', 0)} live persona statement(s) → {len(themes)} theme(s) "
            f"in {stats.get('calls', 0)} model call(s)."
            f"{cur_line}\n\nNothing was saved — the active digest, recall and training "
            "inputs are untouched. The themes and the portrait are in the panel.")
        self._update_sleep_button()

    def _on_digest_dryrun_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._digest_dry_worker:
            return
        self._digest_dry_worker = None
        self._append_text(f"\n[digest-dry] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Persona digest (dry) failed: {err}")
        QMessageBox.critical(
            self, "Persona digest failed",
            f"The digest dry run did not complete:\n\n{err}\n\nNothing was written.")
        self._update_sleep_button()

    @staticmethod
    def _format_digest_dryrun(result: dict) -> str:
        """The readable report: what merged, what the portrait says, how it compares."""
        themes = result.get("themes") or []
        stats = result.get("stats") or {}
        digest = result.get("digest") or {}
        out: list[str] = ["", "=" * 60, "THEMES (clustered on the clean base)", "=" * 60, ""]

        merged = [t for t in themes if t.get("cluster_size", 1) > 1]
        singles = len(themes) - len(merged)
        biggest = max([t.get("cluster_size", 1) for t in themes] or [0])
        out.append(
            f"{result.get('items', 0)} statement(s) → {len(themes)} theme(s): "
            f"{len(merged)} merged, {singles} standing alone. "
            f"{stats.get('calls', 0)} model call(s), "
            f"{stats.get('rejected_blocks', 0)} block(s) rejected by the blob guard.")
        # The blob guard only bounds ONE call, and a merge round joins whole themes — so a
        # theme can still accrete across rounds without any single call seeing an oversized
        # group. This is the number that says whether that happened; read theme 1's members.
        out.append(f"Largest theme: {biggest} statement(s).")
        rounds = stats.get("rounds") or []
        if rounds:
            out.append("Merge rounds: " + ", ".join(
                f"{r.get('before')}→{r.get('after')}" for r in rounds))
        out.append("")

        for i, t in enumerate(themes, 1):
            size = t.get("cluster_size", 1)
            out.append(f"{i:3d}. [{t.get('recurrence', 1)} session(s), "
                       f"weight {t.get('weighted_recurrence', '?')}, "
                       f"{size} statement(s)]")
            out.append(f"     {t.get('content', '')}")
            if size > 1:
                # The members are the point of the preview: this is where a wrong merge
                # is visible. Show them all rather than an elided sample.
                for m in t.get("members") or []:
                    if m != t.get("content"):
                        out.append(f"       · {m}")
            out.append("")

        out += ["=" * 60, "PORTRAIT (synthesized on the adapter)", "=" * 60, ""]
        voice = (digest.get("voice") or "").strip()
        out.append("VOICE:")
        out.append(f"  {voice}" if voice else "  (none parsed)")
        out.append("")
        for label, key in (("STANCES", "stances"), ("LINES", "lines")):
            items = [s for s in (digest.get(key) or []) if (s or "").strip()]
            out.append(f"{label}:")
            out += [f"  - {s.strip()}" for s in items] or ["  (none parsed)"]
            out.append("")
        out.append("DISPOSITIONS:")
        disps = digest.get("dispositions") or []
        if not disps:
            out.append("  (none parsed)")
        for d in disps:
            out.append(f"  - {(d.get('name') or '?').strip()} "
                       f"[{(d.get('maturity') or '?').strip()}]")
            for field in ("when", "do", "not"):
                val = (d.get(field) or "").strip()
                if val:
                    out.append(f"      {field}: {val}")
        out.append("")

        cur = result.get("current")
        if cur:
            counts = cur.get("counts") or {}
            out += ["=" * 60, "COMPARED TO THE ACTIVE DIGEST", "=" * 60, ""]
            out.append(f"  active  : run {cur.get('run_id')} ({cur.get('created')})")
            out.append(f"            {cur.get('themes', 0)} theme(s), "
                       f"largest cluster {cur.get('largest_cluster', 0)}, "
                       f"{counts.get('stances', 0)} stance(s), "
                       f"{counts.get('dispositions', 0)} disposition(s), "
                       f"{counts.get('lines', 0)} line(s)")
            biggest = max([t.get("cluster_size", 1) for t in themes] or [0])
            out.append(f"  dry run : {len(themes)} theme(s), largest cluster {biggest}, "
                       f"{len(digest.get('stances') or [])} stance(s), "
                       f"{len(digest.get('dispositions') or [])} disposition(s), "
                       f"{len(digest.get('lines') or [])} line(s)")
            out.append("")
        out.append("Nothing was written — the active digest, recall, and the next "
                   "build's inputs are unchanged.")
        out.append("")
        return "\n".join(out)

    def _on_outreach_question(self, msg: dict) -> None:
        if not self._outreach_is_current():
            return
        question = (msg.get("question") or "").strip()
        ask_kind = msg.get("ask_kind", "") or "ask"
        user = msg.get("user", "") or "the user"
        self._append_text(
            f"[outreach] Considering her open [{ask_kind}] question (would address "
            f"{user}):\n    {question}\n")
        self._lbl_status.setText("Ava is deliberating…")

    def _emit_outreach_reasoning_header(self) -> None:
        """Write the 'her reasoning:' header once, whoever gets here first.

        The prompt-debug block belongs between the question and the reasoning it produced,
        so the header can no longer be written with the question. Emitting it from both
        possible successors keeps it correct if the prompt block never arrives (an older
        server, or a hook that raised).
        """
        if self._outreach_reasoning_header:
            return
        self._outreach_reasoning_header = True
        self._append_text(
            "[outreach] Her reasoning on whether to raise it now:\n\n")

    def _on_outreach_prompt(self, msg: dict) -> None:
        """The assembled decision prompt, rendered exactly as the Chat tab's Debug view."""
        if not self._outreach_is_current():
            return
        if self._outreach_stream_open:
            self._append_text("\n")
            self._outreach_stream_open = False
        self._append_prompt_debug(f"outreach · {msg.get('pass') or '?'}", msg)
        self._emit_outreach_reasoning_header()

    def _on_outreach_chunk(self, delta: str) -> None:
        if not self._outreach_is_current():
            return
        if delta:
            self._emit_outreach_reasoning_header()
            self._append_text(delta)
            self._outreach_stream_open = True

    def _on_outreach_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._outreach_worker:
            return
        self._outreach_worker = None
        if self._outreach_stream_open:
            self._append_text("\n")
            self._outreach_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or {
                "no_candidates": "She has no surfaceable open question to raise right now.",
                "declined": "She decided not to raise anything this time.",
                "truncated": "She ran past the token budget — before reaching a decision, "
                             "or mid-message after it. Nothing was written; the question "
                             "is still in her pool (try again).",
                "no_model": "No model loaded.",
                "busy": "Another GPU job is in progress.",
                "empty_ask": "The selected question was empty.",
            }.get(skipped, skipped)
            self._append_text(f"\n[outreach] — no session written: {reason}\n")
            self._lbl_status.setText("Reach Out complete — nothing sent.")
        elif result.get("error"):
            self._append_text(f"\n[outreach] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Reach Out failed.")
        elif result.get("composed"):
            opener = (result.get("opener") or "").strip()
            session = result.get("session", "")
            self._append_text(
                "\n[outreach] ✓ She decided to reach out. Her opener:\n"
                f"    {opener}\n")
            self._append_text(
                f"[outreach] Wrote session {session} — it now appears in the Chat tab's "
                "session list (badged 'Ava:'); open it there to reply.\n")
            self._lbl_status.setText("Reach Out complete — Ava started a chat.")
        elif result.get("resolved"):
            answer = (result.get("answer") or "").strip()
            self._append_text(
                "\n[outreach] ✓ She had already learned the answer, so instead of "
                "raising it she resolved the question:\n")
            if answer:
                self._append_text(f"    {answer}\n")
            self._append_text(
                "[outreach] The open question was formally resolved (evicted) — it "
                "won't be surfaced or re-considered again.\n")
            self._lbl_status.setText("Reach Out complete — question resolved.")
        elif result.get("duplicate"):
            answer = (result.get("answer") or "").strip()
            self._append_text(
                "\n[outreach] ✓ She recognized this as a question she has already put "
                "to the user, so instead of repeating it she retired the duplicate:\n")
            if answer:
                self._append_text(f"    (it repeats) {answer}\n")
            self._append_text(
                "[outreach] The duplicate ask was evicted — the originally-raised "
                "question stays open and keeps governing.\n")
            self._lbl_status.setText("Reach Out complete — duplicate question retired.")
        else:
            # Decision was 'no' but reported without a skip reason.
            self._append_text("\n[outreach] — she decided not to reach out this time.\n")
            self._lbl_status.setText("Reach Out complete — nothing sent.")
        self._update_sleep_button()

    def _on_outreach_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._outreach_worker:
            return
        self._outreach_worker = None
        if self._outreach_stream_open:
            self._append_text("\n")
            self._outreach_stream_open = False
        self._append_text(f"[outreach] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Reach Out failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Chat reach out" — manual synthesis pass over an aged chat (debug) #
    # ---------------------------------------------------------------- #

    def _on_synthesis(self) -> None:
        """Manually run one synthesis pass, streaming Ava's reasoning."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._synthesis_worker is not None and self._synthesis_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_lookup_worker is not None and self._til_lookup_worker.isRunning())
                   or (self._til_wander_worker is not None and self._til_wander_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._outreach_worker is not None and self._outreach_worker.isRunning())
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, outreach, or apply job is in progress. "
                "Try again once it finishes.",
            )
            return

        self._synthesis_stream_open = False
        self._append_section_header(
            "Chat reach out — synthesis over an aged chat (debug)")
        self._append_text(
            "[synthesis] Asking Ava to re-read one of her older chats and surface any "
            "questions that newly arise…\n")
        self._lbl_status.setText("Picking an old chat to re-read…")

        worker = SynthesisWorker(client, parent=self)
        worker.stage.connect(self._on_synthesis_stage)
        worker.chunk.connect(self._on_synthesis_chunk)
        worker.done.connect(self._on_synthesis_done)
        worker.error_occurred.connect(self._on_synthesis_error)
        worker.finished.connect(worker.deleteLater)
        self._synthesis_worker = worker
        self._update_sleep_button()
        worker.start()

    def _synthesis_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._synthesis_worker

    def _on_synthesis_stage(self, msg: dict) -> None:
        if not self._synthesis_is_current():
            return
        stage = msg.get("stage", "")
        if self._synthesis_stream_open:
            self._append_text("\n")
            self._synthesis_stream_open = False
        if stage == "picked":
            self._append_text(
                f"[synthesis] Re-reading {msg.get('session', '?')} "
                f"(with {msg.get('user', 'someone')}).\n")
            self._lbl_status.setText("Reflecting on the old chat…")
        elif stage == "analyzing":
            parts = msg.get("parts", 1)
            extra = f" in {parts} parts" if parts and parts > 1 else ""
            self._append_text(
                f"[synthesis] Reading as who she is now{extra}. Her reasoning:\n\n")
            self._lbl_status.setText("Ava is reflecting…")
        elif stage == "no_question":
            if msg.get("off_contract") and msg.get("truncated"):
                # She never reached the form: generation hit the token cap mid-thought.
                # Distinct from ignoring the form, and the fix is a different one.
                self._append_text(
                    "[synthesis] Cut off: generation hit the token cap before the form "
                    "was written (no ABOUT, no ## RAG). Not a decline and not a refusal "
                    "of the form — she ran out of budget mid-thought.\n")
            elif msg.get("off_contract"):
                # Not a decline — the pass reached the form and wrote something else
                # (typically a chat message). Naming it keeps a model failure from
                # reading as a considered "nothing new rose".
                self._append_text(
                    "[synthesis] Off-contract: no ABOUT line and no ## RAG section came "
                    "back. She did not decline — the output form was ignored.\n")
            else:
                self._append_text(
                    f"[synthesis] Nothing new rose this time "
                    f"({msg.get('asks', 0)} question(s) in total).\n")
        elif stage == "composing":
            q = (msg.get("question") or "").strip()
            self._append_text(
                f"\n[synthesis] A new question arose (would address "
                f"{msg.get('user', 'them')}):\n    {q}\n"
                "[synthesis] Composing an opener that reminds them of the chat:\n\n")
            self._lbl_status.setText("Composing her opener…")

    def _on_synthesis_chunk(self, delta: str) -> None:
        if not self._synthesis_is_current():
            return
        if delta:
            self._append_text(delta)
            self._synthesis_stream_open = True

    def _on_synthesis_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._synthesis_worker:
            return
        self._synthesis_worker = None
        if self._synthesis_stream_open:
            self._append_text("\n")
            self._synthesis_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or {
                "no_candidate": "No chat old enough (and not recently revisited) to "
                                "re-read right now.",
                "no_model": "No model loaded.",
                "busy": "Another GPU job is in progress.",
                "unreadable": "The selected chat could not be read.",
                "opener_empty": "She raised a question but produced no opener — nothing "
                                "sent (the question is still in her pool).",
                "opener_truncated": "Her opener ran past the token budget — nothing sent "
                                    "(the question is still in her pool).",
            }.get(skipped, skipped)
            self._append_text(f"\n[synthesis] — {reason}\n")
            self._lbl_status.setText("Chat reach out complete — nothing sent.")
        elif result.get("error"):
            self._append_text(f"\n[synthesis] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Chat reach out failed.")
        elif result.get("composed"):
            opener = (result.get("opener") or "").strip()
            session = result.get("session", "")
            asks = result.get("asks", 0)
            pooled = max(0, int(asks) - 1)
            self._append_text(
                "\n[synthesis] ✓ She reached out. Her opener:\n"
                f"    {opener}\n")
            self._append_text(
                f"[synthesis] Wrote session {session} — it appears in the Chat tab's "
                "session list (badged 'Ava:'); open it there to reply.\n")
            if pooled:
                self._append_text(
                    f"[synthesis] {pooled} further question(s) added to her pool for later.\n")
            self._lbl_status.setText("Chat reach out complete — Ava started a chat.")
        elif result.get("analyzed"):
            asks = result.get("asks", 0)
            self._append_text(
                f"\n[synthesis] — re-read complete; {asks} question(s) added to her "
                "pool, none surfaceable to raise directly this time.\n")
            self._lbl_status.setText("Chat reach out complete — nothing sent.")
        else:
            self._append_text("\n[synthesis] — nothing came of this pass.\n")
            self._lbl_status.setText("Chat reach out complete — nothing sent.")
        self._update_sleep_button()

    def _on_synthesis_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._synthesis_worker:
            return
        self._synthesis_worker = None
        if self._synthesis_stream_open:
            self._append_text("\n")
            self._synthesis_stream_open = False
        self._append_text(f"[synthesis] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Chat reach out failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Check In" — manual (simulated) silence-triggered reach-out       #
    # ---------------------------------------------------------------- #

    def _on_checkin(self) -> None:
        """Manually simulate one check-in decision, streaming Ava's reasoning."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._checkin_worker is not None and self._checkin_worker.isRunning():
            return
        running = (self._poll_worker is not None or self._train_poll_worker is not None
                   or (self._til_worker is not None and self._til_worker.isRunning())
                   or (self._til_lookup_worker is not None and self._til_lookup_worker.isRunning())
                   or (self._til_wander_worker is not None and self._til_wander_worker.isRunning())
                   or (self._til_apply_worker is not None and self._til_apply_worker.isRunning())
                   or (self._outreach_worker is not None and self._outreach_worker.isRunning())
                   or (self._synthesis_worker is not None and self._synthesis_worker.isRunning())
                   or (self._persona_preview_worker is not None
                       and self._persona_preview_worker.isRunning()))
        if running:
            QMessageBox.warning(
                self, "Busy",
                "Another reflection, learning, outreach, or apply job is in progress. "
                "Try again once it finishes.",
            )
            return

        # Whose check-in this is. The autonomous job decides per person; the button runs one
        # of those decisions, for whoever the Chat tab says is at the keyboard. Left blank
        # there, the server falls back to the active session's speaker.
        target = self._chat_widget._current_user()

        self._checkin_stream_open = False
        self._append_section_header(
            "Check In — simulated silence-triggered reach-out (debug)")
        self._append_text(
            f"[checkin] Forging a stretch of silence and asking Ava to review her recent "
            f"conversations with {target or 'the current user'} and decide whether to "
            f"reach out on her own accord…\n")
        self._lbl_status.setText("Reviewing recent chats…")

        worker = CheckinWorker(client, user=target or None, parent=self)
        worker.stage.connect(self._on_checkin_stage)
        worker.prompt_debug.connect(self._on_checkin_prompt)
        worker.chunk.connect(self._on_checkin_chunk)
        worker.done.connect(self._on_checkin_done)
        worker.error_occurred.connect(self._on_checkin_error)
        worker.finished.connect(worker.deleteLater)
        self._checkin_worker = worker
        self._update_sleep_button()
        worker.start()

    def _checkin_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._checkin_worker

    # Same palette and the same segment contract as the Chat tab's Debug view
    # (`chat_widget._PROMPT_DEBUG_COLORS`): blue = the framing Ava is given, green = the
    # memory retrieved into it, red = the turn she is answering, muted grey = history.
    # Duplicated rather than imported because importing the chat widget here would make
    # the dependency run both ways (chat_widget already reaches into this tab), and the
    # contract this shares with it is the server's segment `kind`, not a Python symbol.
    _PROMPT_DEBUG_COLORS = {
        "system": QColor(38, 88, 190),
        "rag": QColor(20, 125, 60),
        "user": QColor(190, 40, 40),
        "assistant": QColor(140, 140, 140),
    }

    def _append_prompt_debug(self, title: str, msg: dict) -> None:
        """Render one pass's full model-facing prompt, colour-coded by segment kind.

        The reflect-lane counterpart of the Chat tab's Debug view, and deliberately the
        same rendering: an operator comparing what chat retrieves against what a
        background pass retrieves should not also have to translate between two layouts.
        Lands in the event log in the order the passes ran, so the prompt sits directly
        above the reasoning it produced.
        """
        segments = msg.get("segments") or []
        cursor = self.txt_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        rule = QTextCharFormat()
        rule.setForeground(QColor(150, 150, 150))
        # The header carries what the segments cannot: an absent RAG segment is ambiguous
        # between "retrieved nothing" and "never asked", and that is the first thing to
        # establish when retrieval is what's under suspicion.
        if msg.get("rag_disabled"):
            rag_note = "RAG: DISABLED by this pass (nothing retrieved)"
        else:
            rag_note = f"RAG: {int(msg.get('rag_tokens') or 0)} tok"
        cursor.insertText(
            f"\n───── prompt · {title} ─────\n"
            f"{int(msg.get('input_tokens') or 0)} input tok · {rag_note} · "
            f"budget {int(msg.get('max_new_tokens') or 0)} of "
            f"{int(msg.get('context_length') or 0)}\n",
            rule)

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", ""))
            if not text.strip():
                continue
            color = self._PROMPT_DEBUG_COLORS.get(
                str(seg.get("kind", "")), QColor(90, 90, 90))
            label = QTextCharFormat()
            label.setForeground(color)
            label.setFontWeight(QFont.Weight.Bold)
            cursor.insertText(f"\n{seg.get('label', '?')}\n", label)
            body = QTextCharFormat()
            body.setForeground(color)
            cursor.insertText(text.rstrip() + "\n", body)

        cursor.insertText("\n───── end prompt ─────\n\n", rule)
        # Leave the cursor neutral, or everything appended after this inherits the last
        # segment's colour.
        neutral = QTextCharFormat()
        neutral.setForeground(self.txt_output.palette().text().color())
        cursor.setCharFormat(neutral)
        self.txt_output.setTextCursor(cursor)
        self.txt_output.ensureCursorVisible()

    def _on_checkin_prompt(self, msg: dict) -> None:
        if not self._checkin_is_current():
            return
        if self._checkin_stream_open:
            self._append_text("\n")
            self._checkin_stream_open = False
        self._append_prompt_debug(f"checkin · {msg.get('pass') or '?'}", msg)

    def _on_checkin_stage(self, msg: dict) -> None:
        if not self._checkin_is_current():
            return
        if self._checkin_stream_open:
            self._append_text("\n")
            self._checkin_stream_open = False
        stage = msg.get("stage")
        if stage == "considering":
            hours = msg.get("hours", "?")
            chats = msg.get("chats", 0)
            user = msg.get("user", "them")
            sim = " (forged period)" if msg.get("simulated") else ""
            self._append_text(
                f"[checkin] ~{hours}h since {user} last spoke{sim}; reviewing "
                f"{chats} recent conversation(s).\n")
            self._lbl_status.setText("Summarizing recent chats…")
        elif stage == "summarizing":
            i = msg.get("i", 0)
            n = msg.get("n", 0)
            session = msg.get("session", "?")
            self._append_text(
                f"[checkin] Recapping recent chat {i}/{n} ({session})…\n")
            self._lbl_status.setText(f"Summarizing recent chat {i}/{n}…")
        elif stage == "recapped":
            # The recaps ARE the decision pass's whole input — it reasons across these and
            # sees nothing else of the window — so show each one as it lands. A cached
            # recap is shown too (marked as such): the operator pressed the button to watch
            # this run, and what she is reading matters more than where it came from.
            i = msg.get("i", 0)
            n = msg.get("n", 0)
            recap = (msg.get("recap") or "").strip()
            when = msg.get("when") or "?"
            with_whom = msg.get("user") or "them"
            tags = []
            # Where this recap came from. A stored gist costs no generation at all — it is
            # reflection's own recap of the same conversation — so an operator watching the
            # pass should be able to tell one from a fresh generation without inferring it
            # from how fast the line appeared.
            if msg.get("source") == "reflection":
                tags.append("from reflection's recap")
            if msg.get("cached"):
                tags.append("cached")
            if msg.get("truncated"):
                tags.append("hit the token cap")
            suffix = f" [{', '.join(tags)}]" if tags else ""
            self._append_text(
                f"[checkin] Recap {i}/{n} — {when} with {with_whom}{suffix}:\n"
                f"    {recap}\n")
        elif stage == "deciding":
            standing = int(msg.get("standing") or 0)
            if standing:
                # Her own unanswered openers are quoted back into the prompt; say so, since
                # it is the difference between her first message into a silence and her Nth.
                self._append_text(
                    f"[checkin] {standing} unanswered message(s) of her own are in the "
                    f"prompt alongside the recaps.\n")
            self._append_text(
                "[checkin] Reasoning across the recaps to decide whether to reach out. "
                "Her reasoning:\n\n")
            self._lbl_status.setText("Ava is deciding whether to reach out…")

    def _on_checkin_chunk(self, delta: str) -> None:
        if not self._checkin_is_current():
            return
        if delta:
            self._append_text(delta)
            self._checkin_stream_open = True

    def _on_checkin_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._checkin_worker:
            return
        self._checkin_worker = None
        if self._checkin_stream_open:
            self._append_text("\n")
            self._checkin_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or {
                "no_model": "No model loaded.",
                "busy": "Another GPU job is in progress.",
                "no_history": "No prior conversations to review yet.",
                "no_recent": "No recent conversations to review.",
                "insufficient_silence": "Not enough silence has passed (autonomous gate).",
                "declined": "She chose not to reach out this time — silence is a fine answer.",
                "truncated": "She ran past the token budget — before deciding, or "
                             "mid-message after it. Nothing sent.",
            }.get(skipped, skipped)
            self._append_text(f"\n[checkin] — {reason}\n")
            self._lbl_status.setText("Check-in complete — nothing sent.")
        elif result.get("error"):
            self._append_text(f"\n[checkin] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Check-in failed.")
        elif result.get("composed"):
            opener = (result.get("opener") or "").strip()
            session = result.get("session", "")
            self._append_text(
                "\n[checkin] ✓ She reached out. Her opener:\n"
                f"    {opener}\n")
            self._append_text(
                f"[checkin] Wrote session {session} — it appears in the Chat tab's "
                "session list (badged 'Ava:'); open it there to reply.\n")
            self._lbl_status.setText("Check-in complete — Ava started a chat.")
        else:
            self._append_text("\n[checkin] — nothing came of this pass.\n")
            self._lbl_status.setText("Check-in complete — nothing sent.")
        self._update_sleep_button()

    def _on_checkin_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._checkin_worker:
            return
        self._checkin_worker = None
        if self._checkin_stream_open:
            self._append_text("\n")
            self._checkin_stream_open = False
        self._append_text(f"[checkin] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Check-in failed: {err}")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # "Apply learning" — persist the last Learn pass's findings         #
    # ---------------------------------------------------------------- #

    def _on_apply_learning(self) -> None:
        """Apply the most recent learning pass.

        Wander commits the previewed exchange to the SFT learning dataset (one one-shot
        Ambient-Enculturation example); Learn/Lookup write the distilled findings to Ava's
        live memory."""
        client = self._chat_widget._client
        if not self._til_pending_text:
            return  # button shouldn't be enabled, but guard anyway
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._til_apply_worker is not None and self._til_apply_worker.isRunning():
            return

        is_wander = self._til_pending_source == "wander"
        if is_wander:
            self._append_text(
                "\n[apply] Applying this wander — to the SFT learning dataset "
                "(one one-shot example) AND to Ava's memory (RAG + weights + ledger)…\n"
            )
            self._lbl_status.setText("Applying wander (learning dataset + memory)…")
        else:
            self._append_text(
                "\n[apply] Writing the learned findings to Ava's memory "
                "(RAG + weights + ledger)…\n"
            )
            self._lbl_status.setText("Applying learning…")

        worker = TilApplyWorker(
            client, self._til_pending_text, self._til_pending_date,
            kind="wander" if is_wander else "", parent=self
        )
        worker.applied.connect(self._on_learning_applied)
        worker.error_occurred.connect(self._on_learning_apply_error)
        worker.finished.connect(worker.deleteLater)
        self._til_apply_worker = worker
        self._update_sleep_button()
        worker.start()

    def _on_learning_applied(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._til_apply_worker:
            return
        self._til_apply_worker = None

        skipped = result.get("skipped")
        if skipped:
            self._append_text(f"[apply] Nothing applied: {skipped}\n")
            self._lbl_status.setText(f"Apply learning skipped ({skipped}).")
            self._update_sleep_button()
            return

        counts = result.get("counts") or {}
        if result.get("source") == "wander" or "wander_examples" in counts:
            n = counts.get("wander_examples", 0)
            rag = counts.get("rag", 0)
            weights = counts.get("weights", 0)
            evict = counts.get("evict", 0)
            self._append_text(
                f"[apply] ✓ Wander applied — {n} exchange added to the SFT learning "
                f"dataset (trained once next cycle, then retired), and {rag} RAG "
                f"insert(s), {weights} weights item(s), {evict} resolved/evicted into "
                f"memory. Recall refreshed.\n"
            )
            if n and not counts.get("wander_trainable", True):
                self._append_text(
                    "[apply] ⚠ The captured reaction isn't a clean single-think target "
                    "(e.g. double-think / no answer) — it was stored but train_cycle will "
                    "skip it. Nothing will train from this one.\n"
                )
            self._lbl_status.setText(
                f"Wander applied (dataset +{n}; memory {rag} RAG, {weights} weights, "
                f"{evict} resolved)."
            )
        else:
            rag = counts.get("rag", 0)
            weights = counts.get("weights", 0)
            evict = counts.get("evict", 0)
            self._append_text(
                f"[apply] ✓ Applied — {rag} RAG insert(s), {weights} weights item(s), "
                f"{evict} resolved/evicted. Recall refreshed; Ava remembers this now.\n"
            )
            self._lbl_status.setText(
                f"Learning applied ({rag} RAG, {weights} weights, {evict} resolved)."
            )
        # Consumed — clear so a second press can't double-write the same findings.
        self._til_pending_text = ""
        self._til_pending_date = ""
        self._til_pending_source = ""
        self._update_sleep_button()

    def _on_learning_apply_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._til_apply_worker:
            return
        self._til_apply_worker = None
        self._append_text(f"[apply] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Apply learning failed: {err}")
        # Keep the pending findings so the operator can retry.
        self._update_sleep_button()

    def _can_start_reflection(self) -> bool:
        if not self._chat_widget.model_loaded:
            QMessageBox.warning(
                self, "Not connected", "Please connect to the server in the Chat tab first."
            )
            return False
        if self._poll_worker is not None:
            QMessageBox.warning(self, "Running", "A reflection run is already in progress.")
            return False
        if self._train_poll_worker is not None:
            QMessageBox.warning(self, "Training", "A LoRA training cycle is still in progress.")
            return False
        if (self._persona_preview_worker is not None
                and self._persona_preview_worker.isRunning()):
            QMessageBox.warning(self, "Running", "A persona preview is already in progress.")
            return False
        return True

    def _reset_run_state(self, filenames: list[str]) -> None:
        """Reset per-run tracking and report accumulators."""
        self._stopping = False
        self._train_requested = False
        self._dry_run = False
        self._dry_full = False
        self._server_run_id = ""
        self._last_event_seq = 0
        self._rendered_seqs = set()
        self._stream_open = False
        self._pass_streamed = False
        self._branch_active = False
        self._branch_ex_idx = None
        self._branch_ex_total = None
        self._report_rag = []
        self._report_resolved = []
        self._report_weights = []
        self._report_pairs = []
        self._report_persona = []
        self._revision_filenames = list(filenames)
        if hasattr(self, "_lbl_stats"):
            self._lbl_stats.setText("")
            self._lbl_stats.setVisible(False)

    def _reflect_ctx_override(self) -> Optional[int]:
        """The Reflect-ctx spinbox value, or None when left at 'default' (0).

        The server clamps any value to [chat context, physical ceiling], so sending a
        stale UI number is safe.
        """
        spin = getattr(self, "spin_reflect_ctx", None)
        if spin is None:
            return None
        v = int(spin.value())
        return v if v > 0 else None

    def _begin_server_run(
        self, filenames: list[str], temperature: Optional[float] = None,
        continue_staging: Optional[bool] = None,
    ) -> None:
        """Start a server-owned reflection run (consolidation + revision + branch)."""
        self.txt_output.clear()
        self.btn_sleep.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_run_state(filenames)

        # Reflection → Merge RAG → Commit Training, then ONE of two ways to mint the
        # resulting Ava version:
        #   Train on  → "train": hand off to the watchdog; train_cycle fits the adapter
        #               and produces+activates the persona itself on a passing probe.
        #   Train off → "persona": the server produces+activates the persona directly
        #               with the current adapter (training-lite). Without this the run's
        #               fresh self-portrait would stay invisible to chat, which reads the
        #               digest through the active-persona pointer rather than live memory.
        train = bool(self.chk_train.isChecked())
        stages = ["reflection", "merge-rag", "commit-training",
                  "train" if train else "persona"]
        self._train_requested = train

        continue_staging_checked = False if continue_staging is None else continue_staging
        clear_staging = not continue_staging_checked
        effective_temperature = temperature if temperature is not None else self.temperature

        if effective_temperature != self.temperature:
            self.temperature = effective_temperature
            self.spn_temperature.blockSignals(True)
            self.spn_temperature.setValue(effective_temperature)
            self.spn_temperature.blockSignals(False)

        overrides: dict = {}
        if effective_temperature != 0.9:
            sampling = {
                "temperature": effective_temperature,
                "top_p": 0.95,   # Gemma 4 recommended (family top_k applied server-side)
                "max_new_tokens_setting": "75%",
            }
            overrides["sleep_sampling"] = sampling
            overrides["revision_sampling"] = sampling

        # "Regen persona": force the digest pass to rebuild even on unchanged evidence.
        if self.chk_regen_persona.isChecked():
            overrides["force_persona_digest"] = True

        # Criterion flip (on by default, maturity-gated server-side). Send the explicit
        # bool so unchecking is a real kill-switch (False), not just an omitted default.
        overrides["apply_branch_judge"] = self.chk_apply_judge.isChecked()

        # "Skip branching": drop the branch-generation + judge phase entirely; the
        # trainable target resolves to the kept original or the revised IDEAL.
        skip_branching = self.chk_skip_branching.isChecked()
        if skip_branching:
            overrides["skip_branching"] = True

        # UI-initiated training skips the regression probe by default (the "Skip
        # validation" footer checkbox), forwarded to the watchdog's POST /train via
        # the run's train_params. "Include fresh chats" rides the same dict: preview
        # rows (untrained, lr 0) for not-yet-eligible chats in the build snapshot.
        train_params = {"skip_validation": self.chk_skip_validation.isChecked(),
                        "include_fresh": self.chk_fresh.isChecked()}

        try:
            result = self._chat_widget._client.start_reflection_run(
                filenames,
                source="ui",
                stages=stages,
                overrides=overrides,
                clear_staging=clear_staging,
                train_params=train_params,
                reflect_context_length=self._reflect_ctx_override(),
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Run start failed",
                f"Could not start reflection run: {e}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        if result.get("type") != "reflection_run_started":
            err = result.get("message", "Unknown error")
            QMessageBox.warning(
                self, "Run start failed",
                f"Server rejected reflection run: {err}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        self._server_run_id = result["run_id"]
        n = len(filenames)
        stages_tag = f" [stages: {'+'.join(stages)}]"
        temp_tag = f", temp {effective_temperature:.2f}"
        branch_tag = ", no-branch" if skip_branching else ", branching"
        valid_tag = ", no-validate" if train_params["skip_validation"] else ", validated"
        if train_params.get("include_fresh"):
            # Train on: preview rows ride the real build's render. Train off: the run
            # writes a GPU-free preview snapshot for pre-training review.
            valid_tag += ", +fresh-preview" if train else ", +preview-snapshot"
        self._lbl_status.setText(
            f"Reflection running: {n} session{'s' if n != 1 else ''} "
            f"(run {self._server_run_id}{temp_tag}{branch_tag}){stages_tag}..."
        )
        self._append_section_header(
            f"Reflection — {n} session{'s' if n != 1 else ''}"
            f" (server{temp_tag}{branch_tag}{valid_tag}{stages_tag})"
        )

        self._start_poll_worker()

    def _on_revisit(self) -> None:
        """Revisit one random old chat (server picks it). No session selection needed."""
        if not self._can_start_reflection():
            return
        self._begin_revisit_run()

    def _begin_revisit_run(self) -> None:
        """Re-reflect one random aged chat under the current persona.

        The server picks a random chat >= its configured min age, re-derives that chat's
        trainable target ('would I answer differently now?'), and rewrites its sidecar —
        persona formation suppressed, no ingestion before, no training after. It streams
        the same reflection events as a normal Sleep run, so we reuse the poll worker.
        """
        self.txt_output.clear()
        self.btn_sleep.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_run_state([])
        # Reflection → Merge RAG → Commit Training (NO train, NO ingestion).
        stages = ["reflection", "merge-rag", "commit-training"]
        self._train_requested = False
        # Keep the criterion flip in step with the Sleep tab's checkbox: the clean-base
        # judge reading the CURRENT persona digest is exactly the "would I answer
        # differently now" signal, so it stays active here.
        overrides = {"apply_branch_judge": self.chk_apply_judge.isChecked()}

        try:
            result = self._chat_widget._client.start_reflection_run(
                [],
                source="ui",
                stages=stages,
                overrides=overrides,
                clear_staging=True,
                revisit=True,
                reflect_context_length=self._reflect_ctx_override(),
            )
        except Exception as e:
            QMessageBox.warning(self, "Revisit failed", f"Could not start revisit: {e}")
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        if result.get("type") != "reflection_run_started":
            err = result.get("message", "Unknown error")
            QMessageBox.warning(self, "Revisit failed", f"Server rejected revisit: {err}")
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        self._server_run_id = result["run_id"]
        chosen = result.get("chosen_session")
        self._revision_filenames = [chosen] if chosen else []
        label_chat = chosen or "(unknown chat)"
        stages_tag = f" [stages: {'+'.join(stages)}]"
        self._lbl_status.setText(
            f"Revisiting old chat {label_chat} (run {self._server_run_id}){stages_tag}..."
        )
        self._append_section_header(
            f"Revisit old chat — {label_chat} "
            f"(server, persona formation suppressed{stages_tag})"
        )

        self._start_poll_worker()

    def _begin_summary_run(self, filenames: list[str]) -> None:
        """Start a consolidation-only, write-nothing reflection preview.

        Reuses the server reflection-run machinery (and its poll/event stream) but
        with ``dry_run`` set: the server runs only the consolidation phase and
        persists nothing — no staging, memory, ledger, RAG refresh, or training.
        The streamed summary and the FULL REPORT show exactly what a real run
        would distil into RAG, so the operator can debug it first."""
        self.txt_output.clear()
        self.btn_sleep.setEnabled(False)
        self.btn_summary.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_run_state(filenames)
        self._dry_run = True

        effective_temperature = self.temperature
        overrides: dict = {}
        if effective_temperature != 0.9:
            sampling = {
                "temperature": effective_temperature,
                "top_p": 0.95,   # Gemma 4 recommended (family top_k applied server-side)
                "max_new_tokens_setting": "75%",
            }
            overrides["sleep_sampling"] = sampling

        try:
            result = self._chat_widget._client.start_reflection_run(
                filenames,
                source="ui",
                stages=["reflection"],
                overrides=overrides,
                dry_run=True,
                reflect_context_length=self._reflect_ctx_override(),
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Summary start failed",
                f"Could not start summary run: {e}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        if result.get("type") != "reflection_run_started":
            err = result.get("message", "Unknown error")
            QMessageBox.warning(
                self, "Summary start failed",
                f"Server rejected summary run: {err}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        self._server_run_id = result["run_id"]
        n = len(filenames)
        temp_tag = f", temp {effective_temperature:.2f}"
        self._lbl_status.setText(
            f"Short summary (dry run): {n} session{'s' if n != 1 else ''} "
            f"(run {self._server_run_id}{temp_tag}) — consolidation only, no writes..."
        )
        self._append_section_header(
            f"Short summary — {n} session{'s' if n != 1 else ''}"
            f" (dry run, consolidation only{temp_tag})"
        )
        self._append_text(
            "[dry run] Generating the consolidation summary only. Nothing will be "
            "written to RAG, memory, the ledger, or training — this is a preview.\n"
        )

        self._start_poll_worker()

    def _begin_dry_sleep_run(self, filenames: list[str]) -> None:
        """Start a full, write-nothing reflection preview on the selected chat(s).

        Reuses the server reflection-run machinery (and its poll/event stream) with
        ``dry_run`` + ``dry_full`` set: the server runs the full consolidation +
        judgement + clean re-answer + branch experiment
        (chat summary → judgement → normal-dialogue IDEAL → counterfactual branches)
        and persists nothing — no staging, memory, ledger, persona digest, sidecar, or
        training. The streamed passes and the FULL REPORT show exactly what a real Sleep
        run would reflect and produce, so the operator can debug it first."""
        self.txt_output.clear()
        self.btn_sleep.setEnabled(False)
        self.btn_dry_sleep.setEnabled(False)
        self.btn_summary.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._reset_run_state(filenames)
        self._dry_run = True
        self._dry_full = True

        effective_temperature = self.temperature
        overrides: dict = {}
        if effective_temperature != 0.9:
            sampling = {
                "temperature": effective_temperature,
                "top_p": 0.95,   # Gemma 4 recommended (family top_k applied server-side)
                "max_new_tokens_setting": "75%",
            }
            overrides["sleep_sampling"] = sampling
            overrides["revision_sampling"] = sampling

        # Honor "Skip branching" in the full dry preview too.
        skip_branching = self.chk_skip_branching.isChecked()
        if skip_branching:
            overrides["skip_branching"] = True

        try:
            result = self._chat_widget._client.start_reflection_run(
                filenames,
                source="ui",
                stages=["reflection"],
                overrides=overrides,
                dry_run=True,
                dry_full=True,
                reflect_context_length=self._reflect_ctx_override(),
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Dry Sleep start failed",
                f"Could not start dry sleep run: {e}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        if result.get("type") != "reflection_run_started":
            err = result.get("message", "Unknown error")
            QMessageBox.warning(
                self, "Dry Sleep start failed",
                f"Server rejected dry sleep run: {err}",
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)
            return

        self._server_run_id = result["run_id"]
        n = len(filenames)
        temp_tag = f", temp {effective_temperature:.2f}"
        branch_desc = ("judgement + re-answer (branching skipped)" if skip_branching
                       else "judgement + re-answer + branch")
        self._lbl_status.setText(
            f"Dry Sleep (dry run): {n} session{'s' if n != 1 else ''} "
            f"(run {self._server_run_id}{temp_tag}) — full reflection, no writes..."
        )
        self._append_section_header(
            f"Dry Sleep — {n} session{'s' if n != 1 else ''}"
            f" (dry run, consolidation + {branch_desc}{temp_tag})"
        )
        self._append_text(
            "[dry run] Running the full reflection (consolidation → judgement → clean "
            "re-answer → branch experiment). Nothing will be written to RAG, memory, the ledger, persona "
            "digest, sidecars, or training — this is a preview.\n"
        )

        self._start_poll_worker()

    # ---------------------------------------------------------------- #
    # Polling                                                           #
    # ---------------------------------------------------------------- #

    def _start_poll_worker(self) -> None:
        """Spin up the off-thread poller for the current run and wire its signals.

        All rendering happens in the connected slots on the GUI thread; the worker
        only does the blocking socket I/O. Parented to this widget and set to
        deleteLater on finish so finished workers don't leak across runs."""
        worker = ReflectionPollWorker(
            self._chat_widget._client,
            self._server_run_id,
            self._chat_widget._server_url,
            start_seq=self._last_event_seq,
            interval=2.0,
            train_expected=self._train_requested,
            parent=self,
        )
        worker.events_ready.connect(self._render_events)
        worker.status_ready.connect(self._render_status)
        worker.connection_lost.connect(self._on_connection_lost)
        worker.reconnected.connect(self._on_reconnected)
        worker.superseded.connect(self._handle_superseded)
        worker.server_down_for_train.connect(self._on_server_down_for_train)
        worker.finished.connect(worker.deleteLater)
        self._poll_worker = worker
        worker.start()

        # Live push listener — renders events as the server flushes them, which is
        # the only path that keeps up during GIL-starved branch generation. Started
        # once per run (idempotent across _start_poll_worker re-entry on reconnect);
        # torn down with the poll worker.
        if self._live_worker is None:
            live = ReflectionLiveWorker(self._chat_widget._client, parent=self)
            live.events_ready.connect(self._render_events)
            live.finished.connect(live.deleteLater)
            self._live_worker = live
            live.start()

    def _teardown_poll_worker(self, *, stop_run: bool = False) -> None:
        """Stop tracking the current run. Never blocks the GUI thread.

        ``stop_run`` routes the server-side halt through the worker thread (user
        pressed Stop); otherwise the run keeps going server-side and we only stop
        polling (handoff/supersede/terminal status)."""
        live = self._live_worker
        self._live_worker = None
        if live is not None:
            live.stop()
        worker = self._poll_worker
        self._poll_worker = None
        if worker is None:
            return
        if stop_run:
            worker.stop_run()
        else:
            worker.stop()

    # ---------------------------------------------------------------- #
    # LoRA training progress (watchdog HTTP poll)                       #
    # ---------------------------------------------------------------- #

    def _start_train_poll(self) -> None:
        """Poll the watchdog for live LoRA training progress over HTTP.

        The reflection run is finalized and the watchdog is stopping this server to
        run the offline cycle; its HTTP mgmt API stays up. Stop here only halts the
        poll — the cycle itself can't be aborted mid-flight from the UI."""
        if self._train_poll_worker is not None:
            return  # already polling training (reached via both hand-off paths)
        client = self._chat_widget._client
        # Respect a custom mgmt port (same source the chat tab's restart/artifacts use).
        try:
            client.mgmt_port = self._chat_widget._get_mgmt_port()
        except Exception:
            pass
        self._append_section_header("LoRA training (watchdog)")
        self._append_text(
            "[train] Handed off to the watchdog. The inference server is stopping to "
            "free the GPU; polling training progress over HTTP "
            f"({client.watchdog_base_url() or 'watchdog'})...\n"
        )
        self._lbl_status.setText("Training (LoRA) on the watchdog — server restarting...")
        self.btn_stop.setEnabled(True)
        worker = TrainPollWorker(client, interval=1.5, parent=self)
        worker.train_events_ready.connect(self._render_train_events)
        worker.train_status_ready.connect(self._on_train_poll_status)
        worker.train_finished.connect(self._on_train_finished)
        worker.finished.connect(worker.deleteLater)
        self._train_poll_worker = worker
        worker.start()

    def _render_train_events(self, events: list) -> None:
        """Render watchdog train-progress events (GUI thread, queued signal)."""
        if self.sender() is not None and self.sender() is not self._train_poll_worker:
            return
        for ev in events:
            stage = ev.get("stage", "")
            status = ev.get("status", "info")
            message = ev.get("message", "")
            tag = f"[train:{stage}]" if stage else "[train]"
            line = f"{tag} {message}".rstrip()
            if status in ("pass", "promoted"):
                line += "  ✓"
            elif status in ("fail", "rejected", "error"):
                line += "  ✗"
            self._append_text(line + "\n")
            # Tier 1/2 validation events carry the actual generated reply — show it
            # indented so the operator can read what the trained model produced.
            data = ev.get("data") or {}
            reply = data.get("reply")
            if reply:
                for ln in str(reply).splitlines() or [""]:
                    self._append_text("    " + ln + "\n")

    def _on_train_poll_status(self, status: dict) -> None:
        if self.sender() is not None and self.sender() is not self._train_poll_worker:
            return
        if status.get("running"):
            self._lbl_status.setText("Training (LoRA) on the watchdog — in progress...")

    def _on_train_finished(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._train_poll_worker:
            return
        self._train_poll_worker = None
        self._train_requested = False
        rc = result.get("returncode")
        observed = result.get("observed")
        if not observed:
            self._append_text(
                "[train] Could not observe a training cycle — the watchdog was "
                "unreachable, or the cycle finished before polling attached. "
                "See server/train.log.\n"
            )
            self._lbl_status.setText("Training status unknown (see server/train.log).")
        elif rc == 0:
            self._append_text("[train] ✓ Training cycle finished (rc=0).\n")
            self._lbl_status.setText("Training complete.")
        else:
            self._append_text(
                f"[train] ✗ Training cycle exited rc={rc}. See server/train.log.\n")
            self._lbl_status.setText("Training finished with errors (see server/train.log).")
        # The watchdog relaunches the inference server, but the WebSocket may take a
        # few seconds to bind. Match the Restart-server UX: tell the operator to
        # reconnect from the Chat tab rather than racing an immediate auto-connect.
        self._append_text(
            "[train] The inference server is relaunching. Reconnect from the Chat "
            "tab once it is back up to resume chatting / reflecting.\n"
        )
        self.btn_stop.setEnabled(False)
        self._update_sleep_button()

    def _on_connection_lost(self) -> None:
        if self.sender() is not None and self.sender() is not self._poll_worker:
            return
        self._append_text(
            "\n[connection lost] Reconnecting — the run continues on the server.\n"
        )
        self._lbl_status.setText(
            "Connection lost — reconnecting (run continues on the server)..."
        )

    def _on_reconnected(self) -> None:
        if self.sender() is not None and self.sender() is not self._poll_worker:
            return
        self._append_text("[reconnected] Resuming event log.\n")

    def _on_server_down_for_train(self) -> None:
        """Reflection poll confirmed the inference server stopped for the LoRA cycle.

        The normal hand-off rides the "completed" status over the WebSocket, but that
        socket dies within the same instant the watchdog stops us to train, so the
        status is easily missed. This fallback fires when the poll worker confirms
        (via the watchdog) that training is underway, so the panel switches to the
        live training/validation progress instead of spinning on reconnect."""
        if self.sender() is not None and self.sender() is not self._poll_worker:
            return
        if self._train_poll_worker is not None:
            return  # the completed-status path already handed off
        if not self._train_requested:
            return
        self._teardown_poll_worker()
        self._append_text(
            "\n[train] Inference server went down for the training hand-off; "
            "switching to live training progress.\n"
        )
        self._start_train_poll()

    @staticmethod
    def _fmt_event_time(ev: dict) -> str:
        """Local-time '[HH:MM:SS] ' prefix from the event's ISO ts.

        Gives each stage line an absolute wall-clock stamp so unattended runs
        can be analysed after the fact. Empty string if the ts is absent or
        unparseable (older servers), so the line still renders.
        """
        ts = ev.get("ts")
        if not ts:
            return ""
        try:
            # ts is UTC-aware ISO; astimezone() with no arg renders local time.
            return datetime.fromisoformat(ts).astimezone().strftime("[%H:%M:%S] ")
        except Exception:
            return ""

    # ---------------------------------------------------------------- #
    # Sequential handoff (desktop <-> laptop)                           #
    # ---------------------------------------------------------------- #

    def on_connected(self) -> None:
        """A client just connected — adopt any reflection run already in progress.

        A run is server-owned and survives a handoff, so when this device connects
        (e.g. after switching from the laptop) we discover an in-flight run, replay
        its event log into the panel, and resume polling. No-op when we are already
        tracking a run locally or none is running."""
        if self._poll_worker is not None:
            return  # already tracking a run (started or adopted here)
        client = self._chat_widget._client
        if not client.is_connected():
            return
        try:
            result = client.list_reflection_runs()
        except Exception:
            return
        if result.get("type") != "reflection_runs_list":
            return
        active = next(
            (r for r in (result.get("runs") or []) if r.get("status") == "running"),
            None,
        )
        if active is not None:
            self._adopt_running_run(active)

    def on_disconnected(self) -> None:
        """Intentional disconnect (e.g. handing off to another device).

        Stop polling and auto-reconnecting — the run continues server-side and is
        re-adopted on the next connect. Without this, the poll loop's _reconnect
        would immediately undo the disconnect, defeating the handoff."""
        if self._poll_worker is not None:
            self._teardown_poll_worker()
            if self._server_run_id:
                self._append_text(
                    "\n[detached] Disconnected — the run continues on the server "
                    "and will reattach on the next connect.\n"
                )
            self.btn_stop.setEnabled(False)
        self._server_run_id = ""

    def _handle_superseded(self) -> None:
        """Another device connected and took over this session. Stop tracking —
        the run continues there and on the server; this viewport is now stale."""
        self._teardown_poll_worker()
        self._append_text(
            "\n[superseded] Another device took over this session. "
            "The run continues there and on the server.\n"
        )
        self._lbl_status.setText("Superseded by another device — detached.")
        self.btn_stop.setEnabled(False)
        self._server_run_id = ""

    def _adopt_running_run(self, run: dict) -> None:
        """Attach to an in-progress server run and rebuild its panel by replaying
        the full event log (after_seq=0 returns everything; the report is derived
        from those same events, so the view is reconstructed faithfully)."""
        run_id = run.get("run_id") or ""
        if not run_id:
            return
        sessions = run.get("selected_sessions") or []
        self.txt_output.clear()
        self.btn_sleep.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._reset_run_state(sessions)
        self._server_run_id = run_id

        self._lbl_status.setText(
            f"Attached to reflection run {run_id} (in progress)..."
        )
        self._append_section_header(f"Attached to in-progress run {run_id}")
        self._append_text("(replaying event log from the server)\n")

        # The worker polls immediately on start (after_seq=0 replays everything).
        self._start_poll_worker()

    def _render_events(self, events: list) -> None:
        """Render a batch of events from the poll worker OR the live push listener.

        Runs on the GUI thread (queued signal). Ignores stale batches from a worker
        we've already stopped tracking (e.g. a tick that raced a user stop). The two
        paths overlap — the live listener renders a candidate as it arrives, the poll
        re-reports it from the store — so seqs already shown are dropped."""
        sender = self.sender()
        if sender is not None and sender not in (self._poll_worker, self._live_worker):
            return
        for ev in events:
            # The live queue can carry a tail of the previous run's pushes; seqs are
            # only unique within a run, so drop anything not tagged for the run we're
            # tracking before it reaches the dedup set (the poll path is already
            # run-scoped by its RPC).
            ev_run = ev.get("run_id")
            if self._server_run_id and ev_run and ev_run != self._server_run_id:
                continue
            seq = ev.get("seq", 0)
            if seq:
                if seq in self._rendered_seqs:
                    continue
                self._rendered_seqs.add(seq)
                if seq > self._last_event_seq:
                    self._last_event_seq = seq
            self._render_reflection_event(ev)

    def _render_reflection_event(self, ev: dict) -> None:
        """Render one structured reflection event into the output panel."""
        etype = ev.get("event", "")
        if etype not in self._DISPLAY_EVENTS:
            return

        if etype == "phase_progress":
            delta = ev.get("text") or ""
            if delta:
                self._append_text(delta)
                self._stream_open = True
                self._pass_streamed = True
            return

        if self._stream_open:
            self._append_text("\n")
            self._stream_open = False

        if etype == "fact_dedup_progress":
            # Carries no `message` — it is the same payload as the manual "Dedup facts"
            # button's dedup_stage events, so ONE renderer serves both (the run stages the
            # merge, the button writes it live). Handled before the generic header line,
            # which would otherwise print a bare `[fact_dedup_progress]` above each.
            self._on_fact_dedup_progress(ev)
            return

        msg = ev.get("message", "")
        sess = ev.get("session", "")
        ex_idx = ev.get("exchange_index")
        ex_total = ev.get("exchange_total")
        chunk = ev.get("chunk")
        chunk_total = ev.get("chunk_total")

        parts = [f"{self._fmt_event_time(ev)}[{etype}]"]
        if sess:
            parts.append(sess)
        if msg:
            parts.append(msg)
        if ex_idx is not None and ex_total:
            parts.append(f"(exchange {ex_idx}/{ex_total})")
        if chunk is not None and chunk_total and chunk_total > 1:
            parts.append(f"(part {chunk}/{chunk_total})")
        self._append_text(" ".join(parts) + "\n")

        if etype == "branch_started":
            self._branch_active = True
            self._branch_ex_idx = ex_idx
            self._branch_ex_total = ex_total
        elif etype in ("branch_done", "branch_skipped"):
            self._branch_active = False

        if etype == "branch_candidate":
            cand_idx = ev.get("candidate_index")
            cand_total = ev.get("candidate_total")
            text = (ev.get("text") or "").strip()
            if cand_idx is not None and cand_total:
                self._append_text(f"    candidate {cand_idx + 1}/{cand_total}:\n")
            if text:
                body = "\n".join("      " + ln for ln in text.splitlines())
                self._append_text(body + "\n")

        if etype in ("pass_warning", "pass_error"):
            # A failure that carries the pass's own output (today: the anchor pass's
            # raw generation on an unparseable parse) renders it under the line — the
            # message alone says only THAT it failed, never what came back.
            text = (ev.get("text") or "").strip()
            if text:
                body = "\n".join("    " + ln for ln in text.splitlines())
                self._append_text(body + "\n")

        if etype in ("phase_started", "session_started"):
            self._pass_streamed = False

        if etype == "persona_cluster_progress":
            stage = ev.get("stage")
            if stage == "map":
                flag = "  ⚠ rejected (one blob) — kept apart" if ev.get("rejected") else ""
                self._append_text(
                    f"    block {ev.get('i', 0)}/{ev.get('n', 0)}: "
                    f"{ev.get('items', 0)} statements → {ev.get('themes', 0)} "
                    f"theme(s){flag}\n")
            elif stage == "reduce":
                self._append_text(
                    f"    merge round {ev.get('round', 0)}: {ev.get('before', 0)} → "
                    f"{ev.get('after', 0)} theme(s) ({ev.get('merged', 0)} merged)\n")
            return

        if etype == "phase_done":
            if not self._pass_streamed:
                text = (ev.get("text") or "").strip()
                if text:
                    indented = "\n".join("    " + ln for ln in text.splitlines())
                    self._append_text(indented + "\n\n")
            self._pass_streamed = False
            self._collect_report(ev)

        if etype == "branch_done":
            options = ev.get("options") or []
            chooser_cot = (ev.get("chooser_cot") or "").strip()
            why = (ev.get("why") or "").strip()

            if options:
                self._append_text("    options (blind order):\n")
                for opt in options:
                    letter = opt.get("letter", "?")
                    kind = opt.get("kind", "")
                    mark = "  ← chosen" if opt.get("chosen") else ""
                    text = (opt.get("text") or "").strip()
                    self._append_text(f"      {letter}) [{kind}]{mark}\n")
                    body = "\n".join("        " + ln for ln in text.splitlines())
                    self._append_text(body + "\n")
            if chooser_cot:
                self._append_text("    reasoning:\n")
                body = "\n".join("      " + ln for ln in chooser_cot.splitlines())
                self._append_text(body + "\n")
            if why:
                self._append_text("    why: " + why + "\n")

            probe = ev.get("probe") or {}
            probe_lines = self._format_branch_probe(probe)
            if probe_lines:
                self._append_text("    probe:\n")
                for ln in probe_lines:
                    self._append_text("      " + ln + "\n")

            if options or chooser_cot or why or probe_lines:
                self._append_text("\n")

    @staticmethod
    def _format_branch_probe(probe: dict) -> list[str]:
        """Render the branch VRAM/token probe as a few human-readable lines.

        Returns [] when the probe is absent (e.g. a server without the probe, or
        a CPU run where the VRAM fields are None)."""
        if not probe:
            return []
        lines: list[str] = []

        def _gb(v):
            return f"{v:.2f} GB" if isinstance(v, (int, float)) else "n/a"

        gen = probe.get("gen_peak_reserved_gb")
        chooser = probe.get("chooser_peak_reserved_gb")
        if gen is not None or chooser is not None:
            lines.append(
                f"peak VRAM (reserved) — generation {_gb(gen)}, chooser {_gb(chooser)}"
            )
            # Also surface allocated peaks if present (closer to true working set).
            gen_a = probe.get("gen_peak_alloc_gb")
            ch_a = probe.get("chooser_peak_alloc_gb")
            if gen_a is not None or ch_a is not None:
                lines.append(
                    f"peak VRAM (allocated) — generation {_gb(gen_a)}, chooser {_gb(ch_a)}"
                )

        total = probe.get("chooser_prompt_tokens")
        if total:
            opt = probe.get("chooser_option_tokens", 0)
            ctx = probe.get("chooser_context_tokens", 0)
            frac = probe.get("chooser_option_frac")
            pct = f" ({frac:.0%})" if isinstance(frac, (int, float)) else ""
            n_opts = probe.get("n_options")
            n_tag = f" across {n_opts} options" if n_opts else ""
            lines.append(
                f"chooser prompt {total} tok = {opt} option text{pct}{n_tag} "
                f"+ {ctx} shared context"
            )
        return lines

    def _render_status(self, result: dict) -> None:
        """Render run status pushed from the poll worker (GUI thread, queued signal).

        Ignores stale payloads from a worker we've already stopped tracking. On a
        terminal status the worker has already emitted (and we've already rendered)
        the final event batch ahead of this status signal, so no extra drain is
        needed before tearing the worker down."""
        if self.sender() is not None and self.sender() is not self._poll_worker:
            return

        status = result.get("status", "")
        phase = result.get("phase", "")
        sess_idx = result.get("session_index", 0)
        sess_total = result.get("session_total", 0)
        skipped = result.get("skipped_passes", 0)

        # Live stats heartbeat — independent of phase, updated every poll.
        self._render_stats_panel(result.get("stats") or {})

        if status == "running":
            if self._branch_active:
                # Branch generation + blind choice take minutes with nothing
                # streamed; keep the operator informed it's still working.
                msg = f"Branch experiment: session {sess_idx}/{sess_total}"
                if self._branch_ex_total:
                    msg += f", exchange {self._branch_ex_idx}/{self._branch_ex_total}"
                msg += " — generating counterfactuals & choosing (minutes)"
                self._lbl_status.setText(msg)
                return
            if phase == "consolidation":
                chunk_idx = result.get("chunk_index", 0)
                chunk_total = result.get("chunk_total", 0)
                msg = f"Consolidation: session {sess_idx}/{sess_total}"
                if chunk_total > 1:
                    msg += f", part {chunk_idx}/{chunk_total}"
            elif phase == "revision":
                ex_idx = result.get("exchange_index", 0)
                ex_total = result.get("exchange_total", 0)
                msg = f"Revision: session {sess_idx}/{sess_total}"
                if ex_total:
                    msg += f", exchange {ex_idx}/{ex_total}"
            elif phase == "ideal_generation":
                ex_idx = result.get("exchange_index", 0)
                ex_total = result.get("exchange_total", 0)
                msg = f"Clean IDEAL re-answer: session {sess_idx}/{sess_total}"
                if ex_total:
                    msg += f", exchange {ex_idx}/{ex_total}"
            else:
                msg = f"Running ({phase or 'initializing'})..."
            self._lbl_status.setText(msg + "...")

        elif status == "completed":
            self._teardown_poll_worker()
            summary = result.get("summary") or {}
            con_passes = summary.get("consolidation_passes", 0)
            rev_passes = summary.get("revision_passes", 0)
            skipped_note = f", {skipped} pass(es) skipped" if skipped else ""
            took = self._fmt_duration(summary.get("elapsed_seconds"))
            took_note = f" Took {took}." if took else ""
            if self._dry_run:
                n = len(self._revision_filenames)
                if self._dry_full:
                    self._append_text(
                        f"[done] Consolidation: {con_passes} pass(es). "
                        f"Revision: {rev_passes} exchange(s){skipped_note} "
                        f"— dry run, nothing written.{took_note}\n"
                    )
                    self._render_report()
                    self._render_detailed_stats(summary.get("detailed_report") or {})
                    self._lbl_status.setText(
                        f"Dry Sleep complete (dry run, nothing written). "
                        f"({n} session{'s' if n != 1 else ''} reviewed{skipped_note})"
                        + (f" — {took}" if took else "")
                    )
                else:
                    self._append_text(
                        f"[done] Consolidation: {con_passes} pass(es)"
                        f"{skipped_note} — dry run, nothing written.{took_note}\n"
                    )
                    self._render_report()
                    self._lbl_status.setText(
                        f"Short summary complete (dry run, nothing written). "
                        f"({n} session{'s' if n != 1 else ''} summarized{skipped_note})"
                        + (f" — {took}" if took else "")
                    )
                self._update_sleep_button()
                self.btn_stop.setEnabled(False)
                return
            committed = bool(summary.get("mutations_applied", False))
            stage_tag = " — committed to production" if committed else " — staged only"
            self._append_text(
                f"[done] Consolidation: {con_passes} pass(es). "
                f"Revision: {rev_passes} exchange(s){skipped_note}{stage_tag}.{took_note}\n"
            )
            self._render_report()
            self._render_detailed_stats(summary.get("detailed_report") or {})
            n = len(self._revision_filenames)
            # When the train stage was requested, the run finalizes here but the
            # watchdog is about to stop this server and run the offline LoRA cycle.
            # Hand off to the watchdog HTTP poller for live training progress
            # instead of finalizing the panel now.
            if self._train_requested:
                self._start_train_poll()
                return
            self._lbl_status.setText(
                f"Reflection complete. ({n} session{'s' if n != 1 else ''} reviewed"
                f"{skipped_note}{stage_tag})"
                + (f" — {took}" if took else "")
            )
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)

        elif status in ("failed", "stopped"):
            self._teardown_poll_worker()
            summary = result.get("summary") or {}
            err = summary.get("error", "")
            took = self._fmt_duration(summary.get("elapsed_seconds"))
            msg = f"Reflection {status}"
            if err:
                msg += f": {err}"
            if took:
                msg += f" (after {took})"
            self._append_text(f"[{status}] {msg}\n")
            if status == "stopped":
                # A stopped run keeps whatever passes already finished — show the
                # partial report so the operator sees what was recorded before halt.
                self._render_report()
                self._render_detailed_stats(summary.get("detailed_report") or {})
            self._lbl_status.setText(msg)
            self._update_sleep_button()
            self.btn_stop.setEnabled(False)

    # ---------------------------------------------------------------- #
    # Stop                                                              #
    # ---------------------------------------------------------------- #

    def _on_stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.btn_stop.setEnabled(False)

        # If a LoRA training cycle is being polled, Stop only detaches the poll —
        # the watchdog has no mid-cycle halt, so training continues server-side.
        if self._train_poll_worker is not None:
            self._train_poll_worker.stop()
            self._train_poll_worker = None
            self._train_requested = False
            self._stopping = False
            self._append_text(
                "\n[stopped] Stopped watching training. The cycle continues on the "
                "watchdog; see server/train.log for the outcome.\n"
            )
            self._lbl_status.setText("Stopped watching training (cycle continues).")
            self._update_sleep_button()
            return

        # Route the server-side halt through the worker thread so the GUI never
        # blocks on the stop RPC (and can't deadlock against an in-flight poll
        # that already holds the client's _rpc_lock).
        self._teardown_poll_worker(stop_run=bool(self._server_run_id))
        self._stopping = False
        self._append_text("\n[stopped] Reflection run halted by user.\n")
        self._lbl_status.setText("Reflection stopped.")
        self._update_sleep_button()

    # ---------------------------------------------------------------- #
    # Output helpers                                                    #
    # ---------------------------------------------------------------- #

    def _append_section_header(self, title: str) -> None:
        cursor = self.txt_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"\n{self._DIVIDER}\n{title}\n{self._DIVIDER}\n\n")
        self.txt_output.setTextCursor(cursor)
        self.txt_output.ensureCursorVisible()

    def _append_text(self, text: str) -> None:
        cursor = self.txt_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.txt_output.setTextCursor(cursor)
        self.txt_output.ensureCursorVisible()

    # ---------------------------------------------------------------- #
    # Full report                                                       #
    # ---------------------------------------------------------------- #

    def _collect_report(self, ev: dict) -> None:
        """Accumulate the structured `report` payload from a phase_done event.

        The server emits the same routing it persists, so the report can't drift
        from the artifacts written to the staging workspace. Rendered by
        _render_report at the end.
        """
        report = ev.get("report")
        if not isinstance(report, dict):
            return
        phase = ev.get("phase", "")
        if phase == "consolidation":
            self._report_weights.extend(report.get("weights") or [])
            self._report_rag.extend(report.get("rag") or [])
            self._report_resolved.extend(report.get("resolved") or [])
        elif phase == "revision":
            if report.get("target_source") != "revised_missing_ideal":
                self._report_pairs.append(report)
            for stmt in report.get("persona") or []:
                self._report_persona.append({
                    "content": stmt,
                    "source_session": report.get("source_session", ""),
                    "exchange_index": report.get("exchange_index"),
                })

    @staticmethod
    def _one_line(text: str, limit: int = 200) -> str:
        """Collapse *text* to a single clipped line for the report."""
        s = " ".join((text or "").split())
        return s if len(s) <= limit else s[:limit].rstrip() + " …"

    def _render_stats_panel(self, stats: dict) -> None:
        """Refresh the live one-line stats panel from a status poll's ``stats`` block.

        Rough ETA is the headline ("~5h left — leave the box unattended"); the rest
        is the at-a-glance heartbeat for the long, silent branch/judge phases."""
        if not isinstance(stats, dict) or not stats:
            return
        parts = []
        elapsed = self._fmt_duration(stats.get("elapsed_seconds"))
        if elapsed:
            parts.append(f"elapsed {elapsed}")
        eta = self._fmt_duration(stats.get("eta_seconds"))
        parts.append(f"ETA ~{eta}" if eta else "ETA estimating…")
        total = stats.get("exchanges_total") or 0
        if total:
            parts.append(f"exchange {stats.get('exchanges_done', 0)}/{total}")
        vram = stats.get("vram_peak_reserved_gb")
        if vram:
            parts.append(f"peak VRAM {vram} GB")
        discards = stats.get("discards_total") or 0
        if discards:
            parts.append(f"{discards} discarded")
        tps = stats.get("tokens_per_sec")
        if tps:
            parts.append(f"{tps} tok/s")
        self._lbl_stats.setText("  ·  ".join(parts))
        self._lbl_stats.setVisible(True)

    @staticmethod
    def _fmt_duration(seconds) -> str:
        """Human-friendly elapsed time: "45s", "3m 07s", "1h 04m". "" if unknown."""
        if seconds is None:
            return ""
        try:
            s = int(round(max(0.0, float(seconds))))
        except (TypeError, ValueError):
            return ""
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"

    def _render_report(self) -> None:
        """Append the FULL REPORT section: RAG facts accumulated + weights pairs."""
        if self._dry_run:
            self._append_section_header(
                "FULL REPORT (dry run — preview only, nothing written)"
            )
        else:
            self._append_section_header("FULL REPORT (recorded to staging workspace)")

        # ── Facts accumulated in RAG ────────────────────────────────────────── #
        n_rag = len(self._report_rag)
        n_resolved = len(self._report_resolved)
        self._append_text(
            f"Facts accumulated in RAG — {n_rag} insert(s), "
            f"{n_resolved} resolved/evicted:\n"
        )
        if not n_rag and not n_resolved:
            self._append_text("  (none)\n")
        for item in self._report_rag:
            kind = item.get("kind", "?")
            label = kind
            if kind == "ask":
                ask_kind = item.get("ask_kind")
                label = f"ask:{ask_kind}" if ask_kind else "ask"
            content = self._one_line(item.get("content", ""))
            line = f"  [{label}] {content}"
            trigger = item.get("trigger")
            if kind == "fact" and trigger:
                line += f"  (trigger: {self._one_line(trigger, 80)})"
            self._append_text(line + "\n")
        for r in self._report_resolved:
            q = self._one_line(r.get("question", ""), 120)
            a = self._one_line(r.get("answer", ""), 120)
            arrow = f" → {a}" if a else ""
            self._append_text(f"  [resolved] {q}{arrow}  (evicts the open ask)\n")

        # ── Bound for weights: consolidation facts/persona ──────────────────── #
        if self._report_weights:
            self._append_text(
                f"\nConsolidation → weights store — {len(self._report_weights)} item(s):\n"
            )
            for w in self._report_weights:
                kind = w.get("weights_kind", "fact")
                self._append_text(
                    f"  [{kind}] {self._one_line(w.get('content', ''))}\n"
                )

        # A consolidation-only dry run (Short Summary) has no revision pairs/persona
        # statements to show — stop after the RAG/weights preview. A full Dry Sleep
        # ran revision + branch, so it falls through and renders those below (they were
        # generated but not written).
        if self._dry_run and not self._dry_full:
            self._append_text("\n")
            return

        # ── Pairs generated for weights training (revision) ─────────────────── #
        n_pairs = len(self._report_pairs)
        self._append_text(
            f"\nPairs generated for weights training — {n_pairs} exchange(s):\n"
        )
        if not n_pairs:
            self._append_text("  (none)\n")
        for p in self._report_pairs:
            sess = p.get("source_session", "")
            idx = p.get("exchange_index")
            verdict = p.get("verdict") or "?"
            tsrc = p.get("target_source") or "?"
            tkind = p.get("target_kind") or "?"
            tgen = p.get("target_generation") or "?"
            branch_tag = " +branch" if p.get("has_branch") else ""
            self._append_text(
                f"  {sess}#{idx}  verdict={verdict}  target={tsrc} "
                f"kind={tkind} generation={tgen}{branch_tag}\n"
            )
            why = p.get("why")
            if why:
                self._append_text(f"      why: {self._one_line(why)}\n")
            self._append_text(
                f"      prompt: {self._one_line(p.get('prompt', ''))}\n"
            )
            self._append_text(
                f"      target: {self._one_line(p.get('target', ''))}\n"
            )

        # ── Persona self-statements (revision → weights) ────────────────────── #
        if self._report_persona:
            self._append_text(
                f"\nPersona self-statements → weights store — "
                f"{len(self._report_persona)} item(s):\n"
            )
            for s in self._report_persona:
                self._append_text(
                    f"  [persona] {self._one_line(s.get('content', ''))}\n"
                )
        self._append_text("\n")

    def _render_detailed_stats(self, report: dict) -> None:
        """Render the detailed end-of-run statistics report (the ``detailed_report``
        block from the run summary; also persisted to ``<run_id>.report.json`` and
        carried into the reflection archive). The VRAM section is the headline for
        the "does this fit in 24 GB?" question."""
        if not isinstance(report, dict) or not report:
            return
        self._append_section_header("RUN STATISTICS")

        totals = report.get("totals") or {}
        timing = report.get("timing") or {}
        outcomes = report.get("outcomes") or {}
        vram = report.get("vram") or {}

        # ── Outcomes ─────────────────────────────────────────────────────────── #
        verdict = outcomes.get("verdict") or {}
        chosen = outcomes.get("chosen") or {}
        self._append_text(
            f"Exchanges: {totals.get('exchanges_done', 0)}/{totals.get('exchanges', 0)} "
            f"processed, {totals.get('exchanges_without_cot', 0)} without CoT; "
            f"{totals.get('consolidation_passes', 0)} consolidation pass(es).\n"
        )
        self._append_text(
            f"Verdicts — keep {verdict.get('keep', 0)}, revise {verdict.get('revise', 0)}.\n"
        )
        self._append_text(
            "Trained target — "
            f"original {chosen.get('original', 0)}, "
            f"ideal-win {chosen.get('ideal_win', 0)}, "
            f"branch-win {chosen.get('branch_win', 0)}, "
            f"cot-regen {chosen.get('cot_regen', 0)}, "
            f"judge-override {chosen.get('judge_override', 0)}.\n"
        )
        _cot_fb = outcomes.get("cot_regen_fallback", 0)
        if _cot_fb:
            self._append_text(
                f"CoT-regen fallback — {_cot_fb} corrupt-CoT reply(ies) kept answer-only "
                "(re-answer diverged too far to graft a faithful thought).\n"
            )

        # ── Discards (categorized) ───────────────────────────────────────────── #
        discarded = outcomes.get("discarded") or {}
        self._append_text(
            f"Discarded — {outcomes.get('discards_total', 0)} total: "
            f"consolidation-gen {discarded.get('consolidation_gen_error', 0)}, "
            f"revision-gen {discarded.get('revision_gen_error', 0)}, "
            f"persist {discarded.get('persist_error', 0)}, "
            f"missing-ideal {discarded.get('revised_missing_ideal', 0)}, "
            f"language-unrepaired {discarded.get('lang_drift_unrepaired', 0)}, "
            f"corrupt-unrepaired {discarded.get('corrupt_response_unrepaired', 0)}, "
            f"branch-unparseable {discarded.get('branch_unparseable', 0)}.\n"
        )
        retries = outcomes.get("retries") or {}
        branch = outcomes.get("branch") or {}
        skipped = branch.get("skipped") or {}
        skip_str = (", ".join(f"{k}: {v}" for k, v in skipped.items())
                    if skipped else "none")
        self._append_text(
            f"Retries — {retries.get('attempted', 0)} attempted, "
            f"{retries.get('recovered', 0)} recovered.\n"
        )
        self._append_text(
            f"Branch — {branch.get('eligible', 0)} eligible; skipped ({skip_str}).\n"
        )
        judge = outcomes.get("judge") or {}
        oq = outcomes.get("open_questions") or {}
        self._append_text(
            f"Judge — {judge.get('judged', 0)} judged, "
            f"{judge.get('overrides', 0)} override(s). "
            f"Open questions — {oq.get('resurfaced', 0)} resurfaced, "
            f"{oq.get('resolved', 0)} resolved. "
            f"Persona — {outcomes.get('persona_written', 0)} written.\n"
        )

        # ── Timing / throughput ──────────────────────────────────────────────── #
        per_phase = timing.get("per_phase") or {}
        if per_phase:
            self._append_text("\nTime per phase:\n")
            for phase, slot in per_phase.items():
                tps = slot.get("tokens_per_sec")
                tps_str = f", {tps} tok/s" if tps else ""
                self._append_text(
                    f"  {phase}: {self._fmt_duration(slot.get('seconds'))} "
                    f"over {slot.get('passes', 0)} pass(es) "
                    f"(avg {slot.get('avg_seconds', 0)}s{tps_str})\n"
                )

        # ── VRAM (the 24 GB question) ────────────────────────────────────────── #
        self._append_text("\nPeak VRAM:\n")
        reserved = vram.get("peak_reserved_gb")
        alloc = vram.get("peak_allocated_gb")
        self._append_text(
            f"  reserved {reserved if reserved is not None else 'n/a'} GB"
            f" (allocated {alloc if alloc is not None else 'n/a'} GB)"
            + (f" at {vram['peak_at_context_tokens']} context tokens"
               if vram.get("peak_at_context_tokens") else "") + "\n"
        )
        per_phase_peak = vram.get("per_phase_peak_gb") or {}
        if per_phase_peak:
            self._append_text(
                "  by phase: "
                + ", ".join(f"{k} {v} GB" for k, v in per_phase_peak.items())
                + "\n"
            )
        dev = vram.get("device_total_gb")
        headroom = vram.get("headroom_vs_device_gb")
        if dev:
            self._append_text(
                f"  device total {dev} GB"
                + (f", headroom {headroom} GB" if headroom is not None else "") + "\n"
            )
        fit = vram.get("would_fit_24gb")
        if fit is not None:
            self._append_text(
                f"  fits in 24 GB (RTX 3090): {'YES' if fit else 'NO'} "
                f"(peak reserved {reserved} GB)\n"
            )
        ctx = vram.get("context_length")
        quant = vram.get("quant")
        meta = []
        if vram.get("model_id"):
            meta.append(f"model {vram['model_id']}")
        if quant:
            meta.append(quant)
        if ctx:
            meta.append(f"ctx {ctx}")
        if meta:
            self._append_text("  config: " + ", ".join(meta) + "\n")
        self._append_text("\n")
