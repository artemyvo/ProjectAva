"""Encounter tab — Ava meets a fellow (non-subjective) AI.

The server runs the whole Ava↔counterpart loop (see ``handle_start_encounter``);
this widget only configures it, kicks it off, and renders the dialogue as it
streams in. It polls ``encounter_events`` / ``encounter_status`` off the GUI
thread (the same pattern the Sleep tab uses for reflection runs), so the long
silent GPU turns never freeze the UI.

The produced transcript is an ordinary chat session on the server, attributed to
the counterpart by name — so the planned chat-deletion / no-reflect-flag features
apply to it without any special-casing here.
"""

from __future__ import annotations

import html
import threading
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QPlainTextEdit, QTextEdit, QFormLayout, QHBoxLayout, QVBoxLayout, QGroupBox,
)

try:  # type-only; avoids a hard import cycle at runtime
    from core.backend_client import BackendClient
except Exception:  # pragma: no cover
    BackendClient = object  # type: ignore


# Speaker colors for the transcript. Kept readable on both light and dark themes.
_AVA_COLOR = "#3b7dd8"
_COUNTERPART_COLOR = "#b5651d"
_SYSTEM_COLOR = "#888888"
_ERROR_COLOR = "#c0392b"


class EncounterPollWorker(QThread):
    """Polls the active encounter off the GUI thread; renders happen in slots."""

    events_ready = pyqtSignal(list)   # new events (seq-ordered)
    status_ready = pyqtSignal(dict)   # encounter_status payload
    connection_lost = pyqtSignal()    # socket dropped (emitted once)
    reconnected = pyqtSignal()

    def __init__(
        self,
        client: "BackendClient",
        server_url: str,
        *,
        start_seq: int = 0,
        interval: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._server_url = server_url
        self._after_seq = start_seq
        self._interval = interval
        self._stop = threading.Event()
        self._stop_run_requested = False
        self._reported_disconnect = False

    def stop(self) -> None:
        self._stop.set()

    def stop_run(self) -> None:
        """Stop polling and ask the server to halt the encounter (user pressed Stop)."""
        self._stop_run_requested = True
        self._stop.set()

    def run(self) -> None:
        self._poll_once()
        while not self._stop.wait(self._interval):
            self._poll_once()
        if self._stop_run_requested:
            try:
                self._client.stop_encounter()
            except Exception:
                pass

    def _poll_once(self) -> None:
        client = self._client
        if not client.is_connected():
            if not self._reconnect():
                return
        self._drain_events()
        try:
            status = client.encounter_status()
        except Exception:
            return
        if status.get("type") != "encounter_status":
            return
        self.status_ready.emit(status)
        # On a terminal status, drain once more: the final ava_message / finished
        # event may have been appended between this cycle's event fetch and the
        # status read, and we must not stop before rendering it.
        if not status.get("active") and status.get("status") in ("completed", "stopped", "failed"):
            self._drain_events()
            self._stop.set()

    def _drain_events(self) -> None:
        try:
            result = self._client.encounter_events(after_seq=self._after_seq)
        except Exception:
            return
        if result.get("type") != "encounter_events_batch":
            return
        events = result.get("events") or []
        if not events:
            return
        for ev in events:
            seq = ev.get("seq", 0)
            if seq > self._after_seq:
                self._after_seq = seq
        self.events_ready.emit(events)

    def _reconnect(self) -> bool:
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


class EncounterWidget(QWidget):
    """Configure and watch an Ava↔fellow-AI encounter."""

    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._poll_worker: Optional[EncounterPollWorker] = None
        self._last_event_seq = 0
        # Ordered transcript blocks + an index for streaming updates.
        self._blocks: list[dict] = []
        self._block_by_key: dict = {}
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(
            "Ava encounters a fellow AI over an OpenAI-compatible endpoint. She "
            "speaks in her own voice; the counterpart's turns are attributed to it "
            "by name. The whole exchange is logged as a normal chat session."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_SYSTEM_COLOR};")
        root.addWidget(intro)

        cfg = QGroupBox("Counterpart")
        form = QFormLayout(cfg)
        self._name = QLineEdit("Spark")
        self._url = QLineEdit("http://spark:8000")
        self._model = QLineEdit()
        self._model.setPlaceholderText("model name the endpoint expects (e.g. gpt-4o-mini)")
        self._turns = QSpinBox()
        self._turns.setRange(1, 50)
        self._turns.setValue(6)
        self._turns.setToolTip("Number of counterpart replies (Ava answers each one).")
        self._cp_system = QPlainTextEdit()
        self._cp_system.setPlaceholderText("Optional system prompt for the counterpart (blank = endpoint default).")
        self._cp_system.setFixedHeight(40)
        self._api_key = QLineEdit()
        self._api_key.setPlaceholderText("Optional API key (Bearer)")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Name:", self._name)
        form.addRow("Endpoint URL:", self._url)
        form.addRow("Model:", self._model)
        form.addRow("Turns:", self._turns)
        form.addRow("Counterpart system:", self._cp_system)
        form.addRow("API key:", self._api_key)

        adv = QGroupBox("Framing && sampling")
        advl = QVBoxLayout(adv)
        self._framing = QPlainTextEdit()
        self._framing.setPlaceholderText(
            "Optional framing override. Use {name} for the counterpart's name. "
            "Blank = the server's default encounter framing."
        )
        self._framing.setFixedHeight(44)
        advl.addWidget(QLabel("Framing (how Ava is told who she's meeting):"))
        advl.addWidget(self._framing)
        samp = QHBoxLayout()
        self._ava_temp = self._mk_temp(1.0)
        self._cp_temp = self._mk_temp(1.0)
        samp.addWidget(QLabel("Ava temp:"))
        samp.addWidget(self._ava_temp)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Counterpart temp:"))
        samp.addWidget(self._cp_temp)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Counterpart max tokens:"))
        self._cp_max = QSpinBox()
        self._cp_max.setRange(64, 16384)
        self._cp_max.setSingleStep(256)
        self._cp_max.setValue(4096)
        samp.addWidget(self._cp_max)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Peer timeout (s):"))
        self._cp_timeout = QSpinBox()
        self._cp_timeout.setRange(30, 3600)
        self._cp_timeout.setSingleStep(30)
        self._cp_timeout.setValue(600)
        self._cp_timeout.setToolTip(
            "How long to wait for one counterpart reply. A reasoning peer (gossip) can "
            "think for minutes over a non-streaming request; raise this if the "
            "connection drops mid-thought."
        )
        samp.addWidget(self._cp_timeout)
        samp.addStretch(1)
        advl.addLayout(samp)

        # Buttons sit in the right column, tucked just under the (short) Framing
        # box — using the space freed because Framing has fewer fields than the
        # Counterpart column.
        controls = QHBoxLayout()
        self._start_btn = QPushButton("Start encounter")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        self._clear_btn = QPushButton("Clear transcript")
        self._clear_btn.clicked.connect(self._clear_transcript)
        controls.addWidget(self._start_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._clear_btn)
        controls.addStretch(1)

        # Right column: short Framing box on top, buttons beneath it, then a
        # trailing stretch so both hug the top and the empty space falls below —
        # this keeps Framing from being stretched to the full height of the taller
        # Counterpart column.
        right_col = QVBoxLayout()
        right_col.addWidget(adv)
        right_col.addLayout(controls)
        right_col.addStretch(1)

        # Counterpart config (left) and the Framing+buttons column (right) share
        # one row, so the upper area stays compact and the transcript gets the rest.
        top_row = QHBoxLayout()
        top_row.addWidget(cfg, 1)
        top_row.addLayout(right_col, 1)
        root.addLayout(top_row)

        self._status = QLabel("Idle.")
        self._status.setStyleSheet(f"color: {_SYSTEM_COLOR};")
        root.addWidget(self._status)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        root.addWidget(self._transcript, 1)

    def _mk_temp(self, value: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 2.0)
        sb.setSingleStep(0.05)
        sb.setDecimals(2)
        sb.setValue(value)
        return sb

    # ------------------------------------------------------------------ #
    # Start / stop                                                        #
    # ------------------------------------------------------------------ #

    def _on_start(self) -> None:
        if not self._chat_widget.model_loaded:
            self._set_status("Load a model from the Chat tab first.", error=True)
            return
        url = self._url.text().strip()
        model = self._model.text().strip()
        if not url:
            self._set_status("Endpoint URL is required.", error=True)
            return
        if not model:
            self._set_status("Model name is required.", error=True)
            return
        name = self._name.text().strip() or "the assistant"

        self._clear_transcript()
        self._set_controls_running(True)
        self._set_status(f"Starting encounter with {name}…")

        client = self._chat_widget._client
        try:
            result = client.start_encounter(
                name=name,
                url=url,
                model=model,
                turns=self._turns.value(),
                counterpart_system=self._cp_system.toPlainText(),
                framing_override=self._framing.toPlainText().strip(),
                temperature=self._ava_temp.value(),
                counterpart_temperature=self._cp_temp.value(),
                counterpart_max_tokens=self._cp_max.value(),
                counterpart_timeout=float(self._cp_timeout.value()),
                api_key=self._api_key.text(),
            )
        except Exception as e:
            self._set_status(f"Failed to start: {e}", error=True)
            self._set_controls_running(False)
            return
        if result.get("type") != "encounter_started":
            self._set_status(result.get("message", "Failed to start encounter."), error=True)
            self._set_controls_running(False)
            return

        self._last_event_seq = 0
        self._start_poll_worker()

    def _on_stop(self) -> None:
        self._set_status("Stopping after the current turn…")
        self._teardown_poll_worker(stop_run=True)
        # We detach from polling here, so we won't see the server's terminal status.
        # Re-enable Start; a too-eager restart while the server finishes the current
        # turn is harmless — it returns a "busy" error the widget surfaces.
        self._set_controls_running(False)

    # ------------------------------------------------------------------ #
    # Polling                                                             #
    # ------------------------------------------------------------------ #

    def _start_poll_worker(self) -> None:
        worker = EncounterPollWorker(
            self._chat_widget._client,
            self._chat_widget._server_url,
            start_seq=self._last_event_seq,
            interval=1.0,
            parent=self,
        )
        worker.events_ready.connect(self._render_events)
        worker.status_ready.connect(self._render_status)
        worker.connection_lost.connect(lambda: self._set_status("Connection lost — reconnecting…", error=True))
        worker.reconnected.connect(lambda: self._set_status("Reconnected."))
        worker.finished.connect(worker.deleteLater)
        self._poll_worker = worker
        worker.start()

    def _teardown_poll_worker(self, *, stop_run: bool = False) -> None:
        worker = self._poll_worker
        self._poll_worker = None
        if worker is None:
            return
        if stop_run:
            worker.stop_run()
        else:
            worker.stop()

    # ------------------------------------------------------------------ #
    # Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _render_events(self, events: list) -> None:
        for ev in events:
            seq = ev.get("seq", 0)
            if seq > self._last_event_seq:
                self._last_event_seq = seq
            self._apply_event(ev)
        self._rerender()

    def _apply_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "ava_start":
            self._ensure_block("ava", ev.get("index", 0), "Ava")
        elif t == "ava_delta":
            blk = self._ensure_block("ava", ev.get("index", 0), "Ava")
            if not blk.get("_final"):
                blk["text"] += ev.get("text", "")
        elif t == "ava_message":
            blk = self._ensure_block("ava", ev.get("index", 0), "Ava")
            blk["text"] = ev.get("text", "")
            blk["cot"] = ev.get("cot", "")
            blk["_final"] = True
            blk["streaming"] = False
        elif t == "counterpart_start":
            blk = self._ensure_block("counterpart", ev.get("index", 0), ev.get("name", "Counterpart"))
            if not blk["text"]:
                blk["text"] = "…"
        elif t == "counterpart_message":
            blk = self._ensure_block("counterpart", ev.get("index", 0), ev.get("name", "Counterpart"))
            blk["text"] = ev.get("text", "")
            blk["cot"] = ev.get("cot", "")
            blk["truncated"] = bool(ev.get("truncated"))
            blk["_final"] = True
            blk["streaming"] = False
        elif t == "info":
            self._add_system(ev.get("text", ""))
        elif t == "error":
            self._add_system(ev.get("text", ""), error=True)
        elif t == "finished":
            status = ev.get("status", "finished")
            err = ev.get("error") or ""
            done = ev.get("turns_done", 0)
            msg = f"Encounter {status} ({done} turns)."
            if err:
                msg += f" {err}"
            self._add_system(msg, error=(status == "failed"))

    def _ensure_block(self, role: str, index: int, name: str) -> dict:
        key = (role, index)
        blk = self._block_by_key.get(key)
        if blk is None:
            blk = {"role": role, "index": index, "name": name, "text": "",
                   "streaming": True, "_final": False}
            self._block_by_key[key] = blk
            self._blocks.append(blk)
        return blk

    def _add_system(self, text: str, error: bool = False) -> None:
        if not text:
            return
        self._blocks.append({"role": "error" if error else "system", "text": text})

    def _rerender(self) -> None:
        parts = []
        for blk in self._blocks:
            role = blk["role"]
            text = html.escape(blk.get("text", "")).replace("\n", "<br>")
            if role == "ava":
                head = f'<b style="color:{_AVA_COLOR}">Ava</b>'
            elif role == "counterpart":
                head = f'<b style="color:{_COUNTERPART_COLOR}">{html.escape(blk.get("name", "Counterpart"))}</b>'
                if blk.get("streaming"):
                    head += f' <span style="color:{_SYSTEM_COLOR}">(thinking…)</span>'
            elif role == "error":
                parts.append(f'<p style="color:{_ERROR_COLOR}"><i>{text}</i></p>')
                continue
            else:  # system
                parts.append(f'<p style="color:{_SYSTEM_COLOR}"><i>{text}</i></p>')
                continue
            cursor = "" if blk.get("_final") else f' <span style="color:{_SYSTEM_COLOR}">▌</span>'
            trunc = ""
            if blk.get("truncated"):
                trunc = (f'<br><span style="color:{_SYSTEM_COLOR}"><i>'
                         f'[reply hit the counterpart max-tokens cap — raise '
                         f'"Counterpart max tokens" to let it finish]</i></span>')
            # Reasoning trace (Ava's, or a peer Ava's over gossip): shown dimmed above
            # the answer once the turn is final. Display-only — never part of the logged
            # transcript. A plain vLLM counterpart sends no CoT, so this stays empty.
            cot = html.escape(blk.get("cot", "")).replace("\n", "<br>")
            cot_html = ""
            if cot:
                cot_html = (f'<span style="color:{_SYSTEM_COLOR}"><i>{cot}</i></span><br>')
            parts.append(f'<p>{head}: {cot_html}{text}{cursor}{trunc}</p>')
        self._transcript.setHtml("".join(parts))
        sb = self._transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _render_status(self, status: dict) -> None:
        st = status.get("status", "idle")
        if not status.get("active") and st in ("completed", "stopped", "failed"):
            self._set_controls_running(False)
            self._teardown_poll_worker(stop_run=False)
            if st != "failed":
                self._set_status(f"Encounter {st}.")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _clear_transcript(self) -> None:
        self._blocks = []
        self._block_by_key = {}
        self._transcript.clear()

    def _set_controls_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)

    def _set_status(self, text: str, error: bool = False) -> None:
        color = _ERROR_COLOR if error else _SYSTEM_COLOR
        self._status.setStyleSheet(f"color: {color};")
        self._status.setText(text)

    def update_fonts(self, font) -> None:
        """Match the app-wide text font (called by MainWindow on font changes)."""
        self._transcript.setFont(font)
