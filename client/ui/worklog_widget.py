"""Worklog tab — a live preview of Ava's first-person episodic worklog.

The worklog (server-side ``core.worklog``) is Ava's durable, semantic record of what she
did, one entry per meaningful episode in her own voice ("I reached out to Artemy about the
worklog idea") — distinct from the machine-phrased Activity journal. Nothing acts on it
yet; wiring it to a deliberation/action loop is a separate task. This tab exists to *watch
it being produced*: it polls ``get_worklog`` with a single id cursor while connected, so as
outreach / wander / reflection / check-in / synthesis close episodes during a real run,
their first-person entries appear here.

Read-only. Like the Activity tab it polls off the GUI thread (the RPC is blocking) and
starts/stops with the socket. Because the worklog is small (one entry per episode, not per
token) it keeps every entry in memory and rerenders the whole view each poll — so the
"open threads" summary at the top always reflects the current fold.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit, QSplitter,
)
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import QThread, pyqtSignal, Qt


_KIND_GLYPH = {
    "conversation": "💬", "wander": "🧭", "outreach": "📨",
    "synthesis": "🔀", "checkin": "👋", "reflection": "🌙", "encounter": "🤝",
}


def _fmt_time(ts) -> str:
    """ISO timestamp → local HH:MM:SS (best-effort)."""
    try:
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "------- --:--:--"


class WorklogPollWorker(QThread):
    """Polls ``get_worklog`` off the GUI thread and hands the batch back via a signal.

    The RPC is blocking (it parks the Qt event loop), so it must never run on the main
    thread. Tracks one id cursor; the widget merges new entries into its full map."""

    batch_ready = pyqtSignal(dict)   # the raw worklog_batch payload

    def __init__(self, client, *, interval: float = 2.0, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._interval = interval
        self._after_seq = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._poll_once()
        while not self._stop.wait(self._interval):
            self._poll_once()

    def _poll_once(self) -> None:
        client = self._client
        if not client.is_connected():
            return
        try:
            batch = client.get_worklog(after_seq=self._after_seq)
        except Exception:
            return
        if not isinstance(batch, dict) or batch.get("type") != "worklog_batch":
            return
        # Advance the cursor past the highest id we just received.
        for e in batch.get("entries") or []:
            self._after_seq = max(self._after_seq, int(e.get("id", 0) or 0))
        if not self._stop.is_set():
            self.batch_ready.emit(batch)


class DeliberateWorker(QThread):
    """Drives one manual (dry-run) deliberation off the GUI thread.

    The Worklog tab's "Deliberate" button asks the server to run the deliberation pass: Ava
    reads her recent episodic worklog and decides what she would do next. DRY RUN — nothing
    is dispatched; her reasoning is streamed back and the terminal message reports the action
    she WOULD take, so the operator can judge whether the decision loop is sane."""

    stage = pyqtSignal(dict)     # {stage, ...} phase markers
    chunk = pyqtSignal(str)      # streamed reasoning delta (<think> included)
    done = pyqtSignal(dict)      # terminal deliberation_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client, parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.deliberate_now():
                if kind == "deliberation_stage":
                    self.stage.emit(msg)
                elif kind == "deliberation_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "deliberation_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Deliberation failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class WorklogWidget(QWidget):
    """Always-live preview of Ava's episodic worklog + the current open-thread fold.

    Also hosts the **Deliberate** button: a dry-run of the deliberation pass (the worklog
    read side) — Ava reads her recent worklog and decides what she would do next, streaming
    her reasoning into the bottom panel. Nothing is executed yet."""

    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._worker: Optional[WorklogPollWorker] = None
        self._delib_worker: Optional[DeliberateWorker] = None
        self._entries: dict[int, dict] = {}      # id -> entry (full history in memory)
        self._open_threads: list[dict] = []
        self._rendered = False                   # first batch always paints (even if empty)
        self.text_font = QFont("Courier")
        self._build_ui()
        try:
            if self._chat_widget._client.is_connected():
                self.on_connected()
        except Exception:
            pass

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._lbl = QLabel("Worklog — Ava's first-person record of what she did.")
        header.addWidget(self._lbl, 1)
        self._btn_deliberate = QPushButton("Deliberate")
        self._btn_deliberate.setToolTip(
            "Dry run: have Ava read her recent worklog and decide what she would do next "
            "(wander / synthesize / reach out / revisit / nothing). Streams her reasoning; "
            "nothing is executed yet."
        )
        self._btn_deliberate.clicked.connect(self._on_deliberate)
        header.addWidget(self._btn_deliberate)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setToolTip("Re-poll the worklog now (it also polls live while "
                                     "connected).")
        self._btn_refresh.clicked.connect(self._refresh_now)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        split = QSplitter(Qt.Orientation.Vertical)

        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        self._out.setPlaceholderText(
            "Ava's episodic worklog will appear here as she closes episodes "
            "(reach-outs, wanders, reflections) during a run.\n\n"
            "This is a read-only preview — nothing acts on the worklog yet."
        )
        split.addWidget(self._out)

        self._delib = QPlainTextEdit()
        self._delib.setReadOnly(True)
        self._delib.setFont(self.text_font)
        self._delib.setPlaceholderText(
            "Press Deliberate to have Ava read her recent worklog and decide what she "
            "would do next. Her reasoning (and the action she would take) appears here — "
            "dry run, nothing is executed."
        )
        split.addWidget(self._delib)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._out.setFont(font)
        self._delib.setFont(font)

    # ── deliberation (dry run) ───────────────────────────────────────────────────
    def _on_deliberate(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._delib.setPlainText("Not connected — connect in the Chat tab first.")
            return
        if self._delib_worker is not None and self._delib_worker.isRunning():
            return
        self._btn_deliberate.setEnabled(False)
        self._delib.setPlainText("Ava is reading her recent worklog…\n\n")
        worker = DeliberateWorker(client)
        worker.stage.connect(self._on_delib_stage)
        worker.chunk.connect(self._on_delib_chunk)
        worker.done.connect(self._on_delib_done)
        worker.error_occurred.connect(self._on_delib_error)
        worker.finished.connect(lambda: self._btn_deliberate.setEnabled(True))
        self._delib_worker = worker
        worker.start()

    def _on_delib_stage(self, info: dict) -> None:
        stage = info.get("stage", "")
        if stage == "reading":
            self._delib_append(
                f"[reading {info.get('episodes', 0)} episode(s), "
                f"{info.get('open_threads', 0)} open thread(s)]")
        elif stage == "deciding":
            self._delib_append("[deciding — her reasoning follows]\n")

    def _on_delib_chunk(self, delta: str) -> None:
        # Insert through a detached cursor rather than moving the *visible* one: moving the
        # visible cursor drags the viewport with it, so a user scrolled up to re-read the
        # reasoning would be pulled back to the end on every streamed token.
        at_bottom = self._delib_at_bottom()
        cur = QTextCursor(self._delib.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(delta)
        if at_bottom:
            self._delib_pin()

    def _on_delib_done(self, result: dict) -> None:
        self._btn_deliberate.setEnabled(True)
        if result.get("skipped"):
            self._delib_append(
                f"\n\n— skipped: {result.get('message') or result['skipped']}")
        elif result.get("error"):
            self._delib_append(f"\n\n— error: {result['error']}")
        else:
            action = result.get("action", "?")
            why = (result.get("why") or "").strip()
            would = result.get("would_do", "")
            trunc = "  (⚠ generation truncated)" if result.get("truncated") else ""
            self._delib_append(
                f"\n\n{'─' * 48}\n"
                f"→ DECISION: {action}{trunc}\n"
                f"  why: {why}\n"
                f"  would: {would}\n"
                f"  (dry run — not executed)")

    def _on_delib_error(self, err: str) -> None:
        self._btn_deliberate.setEnabled(True)
        self._delib_append(f"\n\n— error: {err}")

    def _delib_at_bottom(self) -> bool:
        sb = self._delib.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _delib_append(self, text: str) -> None:
        """Append a line, following the tail only if the user hasn't scrolled away."""
        at_bottom = self._delib_at_bottom()
        self._delib.appendPlainText(text)
        if at_bottom:
            self._delib_pin()

    def _delib_pin(self) -> None:
        sb = self._delib.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── lifecycle (wired to the chat widget's connect/disconnect signals) ───────
    def on_connected(self) -> None:
        if self._worker is not None:
            return
        worker = WorklogPollWorker(self._chat_widget._client)
        worker.batch_ready.connect(self._on_batch)
        self._worker = worker
        worker.start()

    def on_disconnected(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

    def _refresh_now(self) -> None:
        # Cheapest reliable refresh: bounce the poller so it re-reads from cursor 0.
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl.setText("Not connected — connect in the Chat tab first.")
            return
        self.on_disconnected()
        self._entries.clear()
        self._rendered = False
        self.on_connected()

    # ── rendering ────────────────────────────────────────────────────────────────
    def _on_batch(self, batch: dict) -> None:
        # Only rerender when the poll actually brought something new. A rerender replaces
        # the whole document, which drops the user's text selection — at a 2 s poll that
        # made the pane unusable to read from while idle.
        changed = False
        for e in batch.get("entries") or []:
            eid = int(e.get("id", 0) or 0)
            if eid and self._entries.get(eid) != e:
                self._entries[eid] = e
                changed = True
        open_threads = batch.get("open_threads") or []
        if open_threads != self._open_threads:
            self._open_threads = open_threads
            changed = True
        if changed or not self._rendered:
            self._rendered = True
            self._rerender()

    def _rerender(self) -> None:
        n = len(self._entries)
        n_open = len(self._open_threads)
        self._lbl.setText(f"Worklog — {n} episode(s), {n_open} open thread(s).")

        sections: list[str] = []
        if self._open_threads:
            lines = [f"=== open threads  ({n_open}) ==="]
            for t in self._open_threads:
                lines.append(f"  ⟳ {(t.get('opens') or '').strip()}"
                             f"   (#{t.get('id')}, {t.get('kind')})")
            sections.append("\n".join(lines))

        if self._entries:
            lines = [f"=== episodes  ({n}) ==="]
            for eid in sorted(self._entries):
                lines.append(self._format_entry(self._entries[eid]))
            sections.append("\n".join(lines))

        # setPlainText resets the viewport to the top, so a poll landing while the user is
        # reading older entries would snap the pane back to its start. Restore where they
        # were (entries only ever append below, so the absolute offset stays meaningful),
        # and re-pin to the newest entry only if they were already at the bottom.
        sb = self._out.verticalScrollBar()
        prev = sb.value()
        at_bottom = prev >= sb.maximum() - 4
        self._out.setPlainText("\n\n".join(sections))
        sb.setValue(sb.maximum() if at_bottom else min(prev, sb.maximum()))

    @staticmethod
    def _format_entry(e: dict) -> str:
        glyph = _KIND_GLYPH.get(e.get("kind", ""), "•")
        t = _fmt_time(e.get("ts"))
        kind = e.get("kind", "?")
        summary = (e.get("summary") or "").strip()
        head = f"  {glyph} [{t}] #{e.get('id')} ({kind})"
        lines = [head, f"      {summary}"]
        opens = (e.get("opens") or "").strip()
        if opens:
            lines.append(f"      ⟳ opens: {opens}")
        closes = e.get("closes")
        if closes:
            lines.append(f"      ✓ closes #{closes}")
        refs = e.get("refs") or {}
        if refs:
            ref_str = ", ".join(f"{k}={v}" for k, v in refs.items())
            lines.append(f"      ({ref_str})")
        return "\n".join(lines)
