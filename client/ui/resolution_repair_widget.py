"""Repair cut resolutions — a page of the Ask judge tab (server side `core/resolution_repair.py`).

Before the 2026-09-25 fix, the targeted ask-resolution pass ran out of tokens mid-ANSWER on
43% of its passes and closed the ask on the fragment, which was distilled into a
`[persona]` or `[fact]` record. This page lists those closures — found in the activity
journal (the pass hit its cap) or, older than the journal, ending mid-sentence — and
repairs them in two deliberate steps:

* **Dry run** re-reads each conversation with the fixed pass on the live model, in the
  background (this page polls it; chat is refused while it runs; Stop ends it after the
  conversation in flight). It writes nothing to Ava's memory — only the plan.
* **Apply…** asks for confirmation, then, per re-read item: supersedes the cut record
  (out of recall, the digest's evidence and training; kept as evidence and reversible)
  and closes the ask again on the complete answer — or, when the conversation does not
  answer the question after all, reopens the ask. Under the distill gate (on by default)
  the complete answer becomes a `[persona]` / `[fact]` only once the frozen judge accepts
  it, at the next Sleep run (the "Distill gate" page). Every change goes to
  `repair_log.jsonl`.

Select an item to see the cut answer beside the complete one and exactly what Apply will
write for it.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter, QCheckBox, QMessageBox,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal

from ui.chat_widget import ChatWidget

_POLL_MS = 2000
_COLOURS = {
    "resolved": QColor(140, 220, 140),
    "open": QColor(255, 190, 120),
    "refused": QColor(255, 140, 140),
    "error": QColor(255, 140, 140),
    "missing_chat": QColor(255, 140, 140),
    "applied": QColor(150, 150, 150),
}


class _RpcWorker(QThread):
    """Run one blocking client call off the GUI thread; emit its dict."""
    done = pyqtSignal(dict)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs) or {})
        except Exception as e:  # noqa: BLE001 — surfaced to the page as a status line
            self.done.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


class ResolutionRepairWidget(QWidget):
    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._plan_worker: Optional[_RpcWorker] = None
        self._run_worker: Optional[_RpcWorker] = None
        self._status_worker: Optional[_RpcWorker] = None
        self._stop_worker: Optional[_RpcWorker] = None
        self._apply_worker: Optional[_RpcWorker] = None
        self._items: list = []
        self._summary: dict = {}
        self._gated = False
        self._selected: Optional[str] = None
        self._cursor = 0
        self._active = False
        self.text_font = QFont("Courier")
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        about = QLabel(
            "Closures the resolver made on a half-written answer before the 2026-09-25 fix. "
            "<b>Dry run</b> re-reads each conversation with the fixed pass and changes nothing. "
            "<b>Apply</b> supersedes each cut record (out of recall and training, kept as evidence, "
            "reversible) and closes the ask again on the complete answer, which the distill gate "
            "grades at the next Sleep run — or reopens the ask when the conversation does not "
            "answer it.")
        about.setWordWrap(True)
        layout.addWidget(about)

        row = QHBoxLayout()
        self._btn_run = QPushButton("Dry run")
        self._btn_run.setToolTip("Re-read every conversation not re-read yet with the fixed resolver "
                                 "(the live model, ~1–2 min each). Writes only the plan. Chat is "
                                 "refused while it runs.")
        self._btn_run.clicked.connect(self._start)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("End the dry run after the conversation in flight; what was "
                                  "re-read is kept.")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        self._btn_apply = QPushButton("Apply…")
        self._btn_apply.setToolTip("Write the dry run's outcomes — asks for confirmation first.")
        self._btn_apply.clicked.connect(self._apply)
        self._btn_apply.setEnabled(False)
        self._chk_redo = QCheckBox("redo")
        self._chk_redo.setToolTip("Re-read items that already have a dry-run result (not applied ones).")
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        for w in (self._btn_run, self._btn_stop, self._btn_apply, self._chk_redo, self._btn_refresh):
            row.addWidget(w)
        self._lbl_status = QLabel("")
        row.addWidget(self._lbl_status, 1)
        layout.addLayout(row)
        self._lbl_summary = QLabel("")
        self._lbl_summary.setWordWrap(True)
        layout.addWidget(self._lbl_summary)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selected)
        hsplit.addWidget(self._lst)
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        ChatWidget.paint_black_pane(self._out)
        self._out.setPlaceholderText("Select a closure on the left.")
        hsplit.addWidget(self._out)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([440, 700])
        vsplit.addWidget(hsplit)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(self.text_font)
        self._log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        ChatWidget.paint_black_pane(self._log)
        self._log.setPlaceholderText("Dry-run progress appears here.")
        vsplit.addWidget(self._log)
        vsplit.setSizes([520, 140])
        layout.addWidget(vsplit, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._out.setFont(font)
        self._log.setFont(font)

    def _client(self):
        return self._chat_widget._client

    # ── plan ───────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        client = self._client()
        if not client.is_connected():
            self._lbl_status.setText("Not connected (connect in the Chat tab).")
            return
        if self._plan_worker is not None and self._plan_worker.isRunning():
            return
        self._lbl_status.setText(self._lbl_status.text() or "Reading the plan…")
        self._plan_worker = _RpcWorker(client.repair_plan)
        self._plan_worker.done.connect(self._on_plan)
        self._plan_worker.start()
        if not self._timer.isActive():
            self._poll()      # adopt a dry run started before this page looked

    def _on_plan(self, msg: dict) -> None:
        if msg.get("type") != "repair_plan":
            self._lbl_status.setText(f"✗ {msg.get('message', 'could not read the plan')}")
            return
        self._items = list(msg.get("items") or [])
        self._summary = msg.get("summary") or {}
        self._gated = bool(msg.get("gated"))
        s = self._summary
        states = s.get("states") or {}
        self._lbl_summary.setText(
            f"{s.get('items', 0)} closure(s) with a cut answer ("
            + ", ".join(f"{k} {v}" for k, v in sorted((s.get("found_by") or {}).items())) + ") — "
            + " · ".join(f"{k} {v}" for k, v in sorted(states.items()))
            + f".   Apply would supersede {s.get('will_supersede', 0)} cut record(s), "
            + (f"close {s.get('will_distill', 0)} again on the complete answer (the distill gate "
               f"grades them at the next Sleep run)" if self._gated else
               f"distill {s.get('will_distill', 0)} complete answer(s)")
            + f", reopen {s.get('will_reopen', 0)} ask(s).")
        if not self._active and self._lbl_status.text() == "Reading the plan…":
            self._lbl_status.setText("")
        self._set_buttons()
        self._populate()

    def _populate(self) -> None:
        previous = self._selected
        self._lst.blockSignals(True)
        self._lst.clear()
        for it in self._items:
            state = _state(it)
            found = (it.get("found") or {}).get("by", "?")
            item = QListWidgetItem(f"{state:<12} {it.get('ask_kind') or '?':<5} {found:<13} "
                                   f"{_head(it.get('question'), 110)}")
            item.setData(Qt.ItemDataRole.UserRole, it.get("id"))
            if state in _COLOURS:
                item.setForeground(_COLOURS[state])
            self._lst.addItem(item)
        self._lst.blockSignals(False)
        if previous is not None:
            for i in range(self._lst.count()):
                if self._lst.item(i).data(Qt.ItemDataRole.UserRole) == previous:
                    self._lst.setCurrentRow(i)
                    break

    def _on_selected(self) -> None:
        items = self._lst.selectedItems()
        if not items:
            return
        self._selected = items[0].data(Qt.ItemDataRole.UserRole)
        self._render()

    def _render(self) -> None:
        it = next((x for x in self._items if x.get("id") == self._selected), None)
        if it is None:
            return
        self._out.clear()
        cur = self._out.textCursor()
        neutral = QTextCharFormat()
        muted = QTextCharFormat()
        muted.setForeground(ChatWidget._MUTED_GREY)
        warn = QTextCharFormat()
        warn.setForeground(QColor(255, 140, 140))
        good = QTextCharFormat()
        good.setForeground(QColor(140, 220, 140))

        def line(text: str = "", fmt: QTextCharFormat = neutral) -> None:
            cur.insertText(text + "\n", fmt)

        found = it.get("found") or {}
        line(f"{it.get('ask_kind')}  ·  closed {str(it.get('closed_ts') or '')[:19]} by "
             f"{it.get('closing_run') or '?'}  ·  chat {it.get('chat') or '?'}", muted)
        if found.get("by") == "journal":
            line(f"found: the resolver hit its {found.get('max_new_tokens') or 768}-token cap "
                 f"(journal {str(found.get('ts') or '')[:19]})", muted)
        else:
            line("found: older than the journal; the recorded answer stops mid-sentence", muted)
        line()
        line("QUESTION", muted)
        line(str(it.get("question") or ""))
        line()
        line("RECORDED ANSWER (cut)", muted)
        line(str(it.get("old_answer") or "(empty)"), warn)
        old = it.get("old_record") or {}
        line(f"→ distilled as [{old.get('kind')}] {old.get('key')}" if old.get("key")
             else "→ nothing was distilled from it", muted)
        line()

        res = it.get("result") or {}
        outcome = res.get("outcome")
        if not outcome:
            line("RE-READ: not yet — run the dry run.", muted)
        else:
            line(f"RE-READ {str(res.get('ts') or '')[:19]}  ·  adapter "
                 f"{str(res.get('adapter') or '?').split('/')[-1]}"
                 + (f"  ·  {res.get('seconds')} s" if res.get("seconds") else ""), muted)
            if outcome == "resolved":
                line("RESOLVED — the complete answer:", good)
                line(str(res.get("answer") or ""))
            elif outcome == "unchanged":
                line("UNCHANGED — the recorded answer was complete after all; nothing to repair.", muted)
            elif outcome == "open":
                line("OPEN — re-read, the conversation does not answer the question.", warn)
            elif outcome == "refused":
                line(f"REFUSED — the fixed pass could not finish either ({res.get('reason')}).", warn)
            else:
                line(f"{outcome.upper()} — {res.get('reason') or ''}", warn)
        line()

        applied = it.get("applied")
        if applied:
            line(f"APPLIED {str(applied.get('ts') or '')[:19]} ({applied.get('run_id')}): "
                 f"{applied.get('action')} — wrote {applied.get('wrote')}", muted)
        elif outcome in ("resolved", "open", "refused"):
            plan = []
            if old.get("key"):
                plan.append(f"supersede [{old.get('kind')}] {old.get('key')} (recall + ledger, reversible)")
            if outcome == "resolved" and self._gated:
                plan.append("close the ask on the complete answer — the distill gate grades it at "
                            "the next Sleep run")
            elif outcome == "resolved":
                plan.append(f"distill the complete answer as "
                            f"[{'persona' if it.get('ask_kind') == 'meta' else 'fact'}]")
            elif it.get("closed_later"):
                plan.append("keep the ask closed (a later closure answers it)")
            else:
                plan.append("reopen the ask")
            line("APPLY WILL: " + " · ".join(plan), good if outcome == "resolved" else warn)
        cur.setCharFormat(neutral)
        self._out.verticalScrollBar().setValue(0)

    # ── dry run ────────────────────────────────────────────────────────────────
    def _start(self) -> None:
        client = self._client()
        if not client.is_connected() or (self._run_worker is not None and self._run_worker.isRunning()):
            return
        self._btn_run.setEnabled(False)
        self._btn_apply.setEnabled(False)
        self._lbl_status.setText("Starting…")
        self._run_worker = _RpcWorker(client.repair_run, redo=self._chk_redo.isChecked())
        self._run_worker.done.connect(self._on_started)
        self._run_worker.start()

    def _on_started(self, msg: dict) -> None:
        if msg.get("type") != "repair_started" or msg.get("skipped"):
            running = msg.get("skipped") == "running"
            self._active = running
            self._lbl_status.setText(f"✗ {msg.get('message', msg.get('skipped', 'could not start'))}")
            self._set_buttons()
            if running:
                self._timer.start()
            return
        self._active = True
        self._log.clear()
        self._lbl_status.setText(f"Dry run {msg.get('run_id')} — re-reading… (chat is refused until it ends)")
        self._set_buttons()
        self._timer.start()

    def _stop(self) -> None:
        client = self._client()
        if not client.is_connected() or (self._stop_worker is not None and self._stop_worker.isRunning()):
            return
        self._btn_stop.setEnabled(False)
        self._stop_worker = _RpcWorker(client.repair_stop)
        self._stop_worker.done.connect(lambda _m: self._lbl_status.setText(
            "Stopping after the conversation in flight…"))
        self._stop_worker.start()

    def _poll(self) -> None:
        client = self._client()
        if not client.is_connected():
            self._timer.stop()
            return
        if self._status_worker is not None and self._status_worker.isRunning():
            return
        self._status_worker = _RpcWorker(client.repair_status, self._cursor)
        self._status_worker.done.connect(self._on_status)
        self._status_worker.start()

    def _on_status(self, msg: dict) -> None:
        if msg.get("type") != "repair_status":
            return
        new = list(msg.get("lines") or [])
        if new:
            self._cursor = int(msg.get("latest_seq") or self._cursor)
        for ln in new:
            self._log.append(str(ln.get("text") or ""))
        if msg.get("active"):
            if not self._active:
                self._timer.start()
            self._active = True
            tally = msg.get("tally") or {}
            eta = msg.get("eta_s")
            self._lbl_status.setText(
                f"{msg.get('phase')} · {msg.get('done', 0)}/{msg.get('total', 0)}"
                + (f" · eta {int(eta) // 60} min" if eta else "")
                + (" · " + " ".join(f"{k} {v}" for k, v in sorted(tally.items())) if tally else "")
                + (" · stopping…" if msg.get("stop") else "")
                + " · chat refused while it runs")
            self._set_buttons()
            return
        was_active = self._active
        self._active = False
        self._timer.stop()
        self._set_buttons()
        if not was_active:
            return
        phase = str(msg.get("phase") or "")
        self._lbl_status.setText(f"✗ failed: {msg.get('error')}" if phase == "failed" else
                                 f"Dry run {phase}: {msg.get('done', 0)}/{msg.get('total', 0)} re-read — "
                                 f"review, then Apply…")
        self.refresh()

    # ── apply ──────────────────────────────────────────────────────────────────
    def _apply(self) -> None:
        s = self._summary
        if not s.get("applicable"):
            return
        answer = QMessageBox.question(
            self, "Apply the repair",
            f"Apply the dry run to {s.get('applicable')} closure(s)?\n\n"
            f"• supersede {s.get('will_supersede', 0)} cut record(s) — out of recall, persona "
            f"evidence and training; kept in the ledger as evidence and reversible\n"
            + (f"• close {s.get('will_distill', 0)} ask(s) again on the complete answer — distilled "
               f"only if the frozen judge accepts it, at the next Sleep run\n" if self._gated else
               f"• distill {s.get('will_distill', 0)} complete answer(s)\n")
            + f"• reopen {s.get('will_reopen', 0)} ask(s) the conversation does not answer\n\n"
            f"Every change is logged in repair_log.jsonl.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        client = self._client()
        if not client.is_connected() or (self._apply_worker is not None and self._apply_worker.isRunning()):
            return
        self._btn_apply.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._lbl_status.setText("Applying…")
        self._apply_worker = _RpcWorker(client.repair_apply)
        self._apply_worker.done.connect(self._on_applied)
        self._apply_worker.start()

    def _on_applied(self, msg: dict) -> None:
        if msg.get("type") != "repair_applied" or msg.get("skipped"):
            self._lbl_status.setText(f"✗ {msg.get('message', msg.get('skipped', 'apply failed'))}")
        else:
            tally = msg.get("tally") or {}
            self._lbl_status.setText("Applied: " + (", ".join(f"{k} {v}" for k, v in sorted(tally.items()))
                                                    or "nothing to apply") + f" ({msg.get('run_id')})")
        self._set_buttons()
        self.refresh()

    def _set_buttons(self) -> None:
        busy = self._active or (self._apply_worker is not None and self._apply_worker.isRunning())
        self._btn_run.setEnabled(not busy)
        self._chk_redo.setEnabled(not busy)
        self._btn_stop.setEnabled(self._active)
        self._btn_apply.setEnabled(not busy and bool(self._summary.get("applicable")))


def _state(it: dict) -> str:
    if it.get("applied"):
        return "applied"
    return str((it.get("result") or {}).get("outcome") or "pending")


def _head(text, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"
