"""Activity tab — the single, always-on view of what Ava is doing on the GPU box.

Reflection runs have always streamed detailed progress into the Sleep tab, but the five
autonomous idle jobs (wander / outreach / synthesis / checkin / background_reflection)
reported only to ``server.log`` — so the instant a reflection run ended and the box moved
on to autonomous work, or an idle job fired on its own, the UI went silent even though the
GPU was plainly busy. This tab is the unified surface for the server's one activity
journal (``core.activity_log``): it polls ``activity_events`` with a single cursor,
independent of any run, so nothing Ava does on-box is invisible.

Two sources are merged into ONE scrolling log:
  * the WebSocket ``activity_events`` stream (all autonomous jobs + a coarse reflection
    mirror), while the inference socket is up; and
  * the watchdog's ``/job/progress`` (over HTTP), while the socket is DOWN because the
    Sleep "train" stage handed LoRA production to the watchdog and stopped inference — the
    one window the activity socket is legitimately dead. This mirrors the Sleep tab's own
    dual-source train poll, so training shows up in the same timeline as everything else.

The live "current activity" is also surfaced as a status-bar chip (``current_activity_text``,
read by the main window's existing status poll).
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QPlainTextEdit,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QThread, pyqtSignal


_SOURCE_LABEL = {
    "wander": "Wander", "outreach": "Outreach", "synthesis": "Synthesis",
    "checkin": "Check-in", "background_reflection": "Background reflection",
    "reflection": "Reflection", "encounter": "Encounter", "training": "Training",
    "til": "TIL", "idle": "Idle", "server": "Server", "module": "Module",
    "worklog_sweep": "Worklog sweep", "deliberation": "Deliberation",
}
_KIND_GLYPH = {
    "started": "•", "progress": "→", "note": "·", "result": "✓",
    "finished": "✓", "skipped": "·", "failed": "✗",
}
# The level says what KIND of record this is, and that is more useful in the gutter than
# the kind: a heartbeat, a finished generation and a raw stdout line are three different
# things to be scanning for. `event` keeps the kind glyph (its lifecycle meaning is the
# information); the other three take their own.
_LEVEL_GLYPH = {"body": "✎", "stream": "⋯", "raw": "│"}
_BODY_INDENT = "    "


def _fmt_time(ts) -> str:
    """Render an event timestamp (ISO string from activity_log, or epoch float from the
    watchdog train journal) as local HH:MM:SS. Best-effort."""
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts)
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is not None:
                dt = dt.astimezone()
        else:
            return "--:--:--"
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


class ActivityPollWorker(QThread):
    """Polls the unified activity journal (and, while inference is down for training, the
    watchdog's train progress) off the GUI thread. Blocking socket/HTTP RPCs park the Qt
    event loop, so they must never run on the main thread — this hands results back through
    queued signals; the widget only renders."""

    events_ready = pyqtSignal(list)          # list of render dicts (see _render)
    current_changed = pyqtSignal(object)     # dict | None — the live chip

    def __init__(self, client, *, interval: float = 2.0, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._interval = interval
        self._after_seq = 0          # activity-journal cursor
        self._train_after_seq = 0    # independent train-progress cursor
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._poll_once()   # fill immediately, not after the first interval
        while not self._stop.wait(self._interval):
            self._poll_once()

    def _poll_once(self) -> None:
        client = self._client
        if client.is_connected():
            try:
                batch = client.activity_events(after_seq=self._after_seq)
            except Exception:
                return
            if batch.get("type") != "activity_events_batch":
                return
            out = []
            if batch.get("gap"):
                # The cursor fell off the end of what the server can still serve (a long
                # disconnect, or a burst that rotated past the journal's window). Say so:
                # a truncated history rendered as continuity is worse than a short one.
                out.append({"key": ("g", self._after_seq), "ts": None, "source": "server",
                            "kind": "note", "phase": None, "level": "event", "text": "",
                            "message": "— earlier activity is no longer in the journal —"})
            for ev in batch.get("events") or []:
                seq = ev.get("seq", 0)
                if seq > self._after_seq:
                    self._after_seq = seq
                out.append(self._render(ev))
            if out and not self._stop.is_set():
                self.events_ready.emit(out)
            self.current_changed.emit(batch.get("activity"))
        else:
            # Inference socket down — merge the watchdog's train progress if a LoRA cycle
            # is running (the Sleep "train" hand-off stops inference to free the GPU).
            self._poll_training()

    def _poll_training(self) -> None:
        try:
            prog = self._client.train_progress(after_seq=self._train_after_seq)
        except Exception:
            return
        if not isinstance(prog, dict) or prog.get("error"):
            return
        out = []
        for ev in prog.get("events") or []:
            seq = ev.get("seq", 0)
            if seq > self._train_after_seq:
                self._train_after_seq = seq
            out.append(self._render_train(ev))
        if out and not self._stop.is_set():
            self.events_ready.emit(out)
        if prog.get("running"):
            self.current_changed.emit({"source": "training", "message": "LoRA training"})

    @staticmethod
    def _render(ev: dict) -> dict:
        return {"key": ("a", ev.get("seq", 0)), "ts": ev.get("ts"),
                "source": ev.get("source", ""), "kind": ev.get("kind", "note"),
                "phase": ev.get("phase"), "message": ev.get("message", ""),
                # `text` is the payload the server keeps out of `message` so the headline
                # never has to be clipped to bound the body: a pass's verbatim CoT +
                # output (`body`), or a heartbeat's rolling tail (`stream`).
                "level": ev.get("level", "event"), "text": ev.get("text") or ""}

    @staticmethod
    def _render_train(ev: dict) -> dict:
        status = str(ev.get("status") or "")
        stage = str(ev.get("stage") or "")
        # `log` is the raw stdout of the train cycle — unsloth's own output, mirrored here
        # by the activity tee because during training the watchdog's progress journal is
        # the only thing a client can reach (the inference socket is down). Render it as
        # the raw line it is rather than as a "log: …" stage.
        if stage == "log":
            return {"key": ("t", ev.get("seq", 0)), "ts": ev.get("ts"),
                    "source": "training", "kind": "note", "phase": None,
                    "message": str(ev.get("message") or ""), "level": "raw", "text": ""}
        kind = ("failed" if status == "error"
                else "finished" if stage == "done" else "progress")
        msg = str(ev.get("message") or stage)
        if stage and stage.lower() not in msg.lower():
            msg = f"{stage}: {msg}"
        return {"key": ("t", ev.get("seq", 0)), "ts": ev.get("ts"),
                "source": "training", "kind": kind, "phase": stage, "message": msg,
                "level": "event", "text": ""}


class ActivityWidget(QWidget):
    """Always-live scrolling log of the unified activity journal + a current-activity chip.

    Polls continuously from the moment the client connects (not only when the tab is
    focused), so switching to it shows the running history immediately."""

    _MEM_CAP = 2500   # events retained for the hide-filter rebuild (they carry bodies now)

    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._worker: Optional[ActivityPollWorker] = None
        self._events: list[dict] = []
        self._seen: set = set()
        self._current: Optional[dict] = None
        self._build_ui()
        # If the client is already connected when this tab is built, start polling now.
        try:
            if self._chat_widget._client.is_connected():
                self.on_connected()
        except Exception:
            pass

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self._chip = QLabel("Idle")
        self._chip.setStyleSheet("font-weight: bold; padding: 2px 6px;")
        header.addWidget(self._chip, 1)
        self._hide_cb = QCheckBox("Hide routine skips")
        self._hide_cb.setToolTip(
            "Hide the hourly 'nothing to do' lines (e.g. no open question to raise).")
        self._hide_cb.toggled.connect(lambda _c: self._rerender())
        header.addWidget(self._hide_cb)
        # Every generation on the box reports what it produced (its verbatim CoT + output)
        # and, while it is still running, a heartbeat carrying the rolling tail of what it
        # is writing — indented under the headline. Watching that is the point of the tab,
        # so it is ON by default; this collapses it back to one line per event when the log
        # is being read as an overview rather than watched.
        self._brief_cb = QCheckBox("Hide pass output")
        self._brief_cb.setToolTip(
            "Collapse each entry to its headline, hiding the indented body: a pass's\n"
            "generated CoT + output, a live heartbeat's tail, and reflection's\n"
            "distilled items / verdicts / anchors.")
        self._brief_cb.toggled.connect(lambda _c: self._rerender())
        header.addWidget(self._brief_cb)
        # Raw stdout of the server process and — the reason this is on by default — of the
        # offline train cycle, which is how unsloth's own output (loss, warnings,
        # tracebacks) reaches this view at all.
        self._raw_cb = QCheckBox("Hide raw output")
        self._raw_cb.setToolTip(
            "Hide raw stdout lines (server prints, and unsloth's output during a\n"
            "training cycle). Leaves the semantic events and pass output.")
        self._raw_cb.toggled.connect(lambda _c: self._rerender())
        header.addWidget(self._raw_cb)
        clear = QPushButton("Clear")
        clear.setToolTip("Clear the on-screen log (does not affect the server journal).")
        clear.clicked.connect(self._clear)
        header.addWidget(clear)
        layout.addLayout(header)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        # Blocks are LINES, not events, and an event may now carry a whole generation —
        # so this is sized for bodies rather than for one line per event.
        self._log.setMaximumBlockCount(60000)
        layout.addWidget(self._log, 1)

    def update_fonts(self, font: QFont) -> None:
        self._log.setFont(font)

    # ── lifecycle (wired to the chat widget's connect/disconnect signals) ───────
    def on_connected(self) -> None:
        if self._worker is not None:
            return
        client = self._chat_widget._client
        worker = ActivityPollWorker(client)
        worker.events_ready.connect(self._on_events)
        worker.current_changed.connect(self._set_current)
        self._worker = worker
        worker.start()

    def on_disconnected(self) -> None:
        self._teardown_worker()
        self._set_current(None)

    def _teardown_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

    # ── rendering ───────────────────────────────────────────────────────────────
    def _on_events(self, events: list) -> None:
        appended = False
        for ev in events:
            key = ev.get("key")
            if key in self._seen:
                continue
            self._seen.add(key)
            self._events.append(ev)
            if self._visible(ev):
                self._log.appendPlainText(self._line(ev))
                appended = True
        if len(self._events) > self._MEM_CAP:
            self._events = self._events[-self._MEM_CAP:]
            # _seen may hold keys for dropped events; rebuild it lazily on next rerender.
        if appended:
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _visible(self, ev: dict) -> bool:
        if self._hide_cb.isChecked() and ev.get("kind") == "skipped":
            return False
        if self._raw_cb.isChecked() and ev.get("level") == "raw":
            return False
        return True

    def _line(self, ev: dict) -> str:
        t = _fmt_time(ev.get("ts"))
        level = ev.get("level", "event")
        glyph = _LEVEL_GLYPH.get(level) or _KIND_GLYPH.get(ev.get("kind", "note"), "·")
        src = _SOURCE_LABEL.get(ev.get("source", ""), ev.get("source") or "?")
        msg = (ev.get("message") or "").strip()
        # The body arrives in its own field (`text`) rather than packed into the headline,
        # so it can be indented here and folded away by "Hide pass output" without the
        # server having had to clip it to keep the headline readable. Reflection's mirror
        # still writes its distilled items as indented continuation lines INSIDE `message`
        # (a derived rendering, not a generation), so the same indent rule folds both.
        body = (ev.get("text") or "").strip()
        if body and not self._brief_cb.isChecked():
            lines = body.splitlines()
            # A `stream` body is not a record — it is the rolling TAIL of a generation
            # still being written (`activity_log.stream_tail_chars`, ~240 chars), so it
            # begins and ends mid-word by construction. Rendered identically to a finished
            # `body` it reads as a corrupted entry: the same pass then appears several
            # times, each a different randomly-cut fragment. Mark both ends so it reads as
            # what it is — a window onto a generation in progress, whose full text arrives
            # on the `✎` line that closes the pass.
            if level == "stream" and lines:
                lines = [f"…{lines[0]}"] + lines[1:]
                lines[-1] = lines[-1] + "…"
            msg = msg + "\n" + "\n".join(_BODY_INDENT + ln for ln in lines)
        if self._brief_cb.isChecked() and "\n" in msg:
            msg = "\n".join(
                ln for ln in msg.splitlines() if not ln.startswith(_BODY_INDENT)
            ).strip()
        # Most journal messages already self-prefix with their source ("Wander: read …",
        # "wander skipped: …"); only prepend the source column when they don't.
        if msg.lower().startswith((ev.get("source") or "").lower()) or \
                msg.lower().startswith(src.lower()):
            return f"[{t}] {glyph} {msg}"
        return f"[{t}] {glyph} {src}: {msg}"

    def _rerender(self) -> None:
        self._log.clear()
        lines = [self._line(ev) for ev in self._events if self._visible(ev)]
        if lines:
            self._log.setPlainText("\n".join(lines))
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear(self) -> None:
        self._events.clear()
        self._seen.clear()
        self._log.clear()

    # ── current-activity chip ────────────────────────────────────────────────────
    def _set_current(self, cur) -> None:
        self._current = cur if isinstance(cur, dict) else None
        self._chip.setText(self._chip_text() or "Idle")

    def _chip_text(self) -> str:
        if not self._current:
            return ""
        src = _SOURCE_LABEL.get(self._current.get("source", ""),
                                self._current.get("source") or "")
        return f"● {src} — running" if src else ""

    def current_activity_text(self) -> str:
        """Short label for the main-window status chip, or '' when the box is idle."""
        if not self._current:
            return ""
        src = _SOURCE_LABEL.get(self._current.get("source", ""),
                                self._current.get("source") or "")
        return f"Ava: {src}" if src else ""
