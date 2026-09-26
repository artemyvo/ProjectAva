"""Distill gate — a page of the Ask judge tab (server side `core/distill_gate.py`).

A resolved ask used to be distilled the moment the resolver said yes: a `meta` answer into
a `[persona]` record, a `user` answer into a `[fact]` filed as the person's own report.
Now the ask is closed at once, and the distill waits for the frozen judge. At the next
Sleep run's clean-base batch the judge grades the closure and the gate distills it, or
withholds it:

* a `[fact]` only from the person's own words (RESOLVED_EVIDENCE, quote found, answered by
  the other speaker) — their agreement with Ava's words closes the question but is not
  their report;
* a `[persona]` only from a position of Ava's own, or one reached with the person — the
  other speaker's view of her (the capture route, P7) is withheld — and only once the
  position has held in a fresh session (the context-reset test, the Reset test page);
* an ask filed under the wrong kind (a `user` ask about Ava) is judged again as the kind
  its subject calls for.

This page shows what waits for the gate, what it decided, and — for the records distilled
before the gate existed — what it would do to them: keep, supersede (out of recall, the
digest's evidence and training; kept as evidence and reversible) or reroute (superseded and
distilled as the other kind). **Apply retro…** writes that, after a confirmation. The plan
reads the verdicts of the current judge, so run **Judge all** on the Judge page first.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter, QComboBox, QMessageBox,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.chat_widget import ChatWidget

_FILTERS = [
    ("everything", None), ("waiting for the gate", ("pending", None)),
    ("retro: supersede", ("retro", "supersede")), ("retro: reroute", ("retro", "reroute")),
    ("retro: keep", ("retro", "keep")), ("retro: no verdict yet", ("retro", "wait")),
    ("decided", ("decided", None)),
]
_GREEN, _AMBER, _RED, _GREY = (QColor(140, 220, 140), QColor(255, 190, 120),
                               QColor(255, 140, 140), QColor(150, 150, 150))


def _head(text, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _tag(it: dict) -> str:
    """What the gate does (or did) with an item, in one word."""
    d = it.get("decision") or {}
    if it.get("state") == "retro":
        return it.get("action") or "?"
    if it.get("state") == "decided":
        return (it.get("action") or d.get("decision") or "?") + (f" [{d['as']}]" if d.get("as") else "")
    if it.get("reset_state"):
        # a position about Ava: distilled only once it holds in a fresh session (P7)
        return {"passed": "→ [persona] ✓reset", "failed": "→ withhold ✗reset"}.get(
            it["reset_state"], f"reset test: {it['reset_state']}")
    return {"distill": f"→ [{d.get('as')}]", "withhold": "→ withhold", "wait": "no verdict",
            "rejudge": "→ re-judge"}.get(d.get("decision"), "?")


def _colour(it: dict) -> Optional[QColor]:
    d = it.get("decision") or {}
    if it.get("state") == "decided":
        return _GREY
    if it.get("state") == "retro":
        return {"keep": _GREEN, "supersede": _RED, "reroute": _AMBER, "wait": _GREY}.get(it.get("action"))
    if it.get("reset_state"):
        return {"passed": _GREEN, "failed": _RED}.get(it["reset_state"], _AMBER)
    return {"distill": _GREEN, "withhold": _RED, "wait": _GREY, "rejudge": _AMBER}.get(d.get("decision"))


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


class DistillGateWidget(QWidget):
    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._plan_worker: Optional[_RpcWorker] = None
        self._apply_worker: Optional[_RpcWorker] = None
        self._plan: dict = {}
        self._items: list = []
        self._selected: Optional[str] = None
        self.text_font = QFont("Courier")
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        about = QLabel(
            "A resolved ask is closed at once; what it distills into waits for the frozen judge "
            "(next Sleep run, clean base). A <b>[fact]</b> only from the person's own words, a "
            "<b>[persona]</b> only from a position of Ava's own; a mis-filed ask is judged again as "
            "the kind its subject calls for; a position about Ava distills only after the context-"
            "reset test (the Reset test page). <b>Apply retro…</b> gates the records distilled before "
            "the gate existed — run <b>Judge all</b> on the Judge page first.")
        about.setWordWrap(True)
        layout.addWidget(about)

        row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_apply = QPushButton("Apply retro…")
        self._btn_apply.setToolTip("Keep, supersede or reroute the records distilled before the gate, "
                                   "by the current judge's verdicts — asks for confirmation first.")
        self._btn_apply.clicked.connect(self._apply)
        self._btn_apply.setEnabled(False)
        self._cmb_filter = QComboBox()
        for label, _ in _FILTERS:
            self._cmb_filter.addItem(label)
        self._cmb_filter.currentIndexChanged.connect(lambda _i: self._populate())
        for w in (self._btn_refresh, self._btn_apply, self._cmb_filter):
            row.addWidget(w)
        self._lbl_status = QLabel("")
        row.addWidget(self._lbl_status, 1)
        layout.addLayout(row)
        self._lbl_summary = QLabel("")
        self._lbl_summary.setWordWrap(True)
        layout.addWidget(self._lbl_summary)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selected)
        split.addWidget(self._lst)
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        ChatWidget.paint_black_pane(self._out)
        self._out.setPlaceholderText("Select a closure on the left.")
        split.addWidget(self._out)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([460, 700])
        layout.addWidget(split, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._out.setFont(font)
        self._lst.setFont(font)       # the rows are columns

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
        self._lbl_status.setText(self._lbl_status.text() or "Reading…")
        self._plan_worker = _RpcWorker(client.gate_plan)
        self._plan_worker.done.connect(self._on_plan)
        self._plan_worker.start()

    def _on_plan(self, msg: dict) -> None:
        if msg.get("type") != "gate_plan":
            self._lbl_status.setText(f"✗ {msg.get('message', 'could not read the gate')}")
            return
        self._plan = msg
        self._items = list(msg.get("items") or [])
        s = msg.get("summary") or {}
        retro = (s.get("retro") or {}).get("actions") or {}
        decided = s.get("decided") or {}
        key = msg.get("judge_key")
        parts = [
            ("gate ON" if msg.get("enabled") else "gate OFF (distill_gate.enabled) — answers are "
             "distilled at once, as before"),
            f"judge {key}" if key else "no verdicts under the current judge yet — run Judge all",
            f"{s.get('pending', 0)} closure(s) wait for the next Sleep run "
            f"({s.get('pending_judged', 0)} already judged)",
            "decided: " + (" · ".join(f"{k} {v}" for k, v in sorted(decided.items())) or "none"),
            "before the gate: " + (" · ".join(f"{k} {v}" for k, v in sorted(retro.items())) or "nothing"),
        ]
        self._lbl_summary.setText("   ·   ".join(parts))
        if self._lbl_status.text() == "Reading…":      # keep an apply's result line
            self._lbl_status.setText("")
        actionable = sum(v for k, v in retro.items() if k in ("keep", "supersede", "reroute"))
        self._btn_apply.setEnabled(bool(key) and actionable > 0 and not msg.get("busy"))
        self._populate()

    def _populate(self) -> None:
        want = _FILTERS[self._cmb_filter.currentIndex()][1]
        previous = self._selected
        self._lst.blockSignals(True)
        self._lst.clear()
        for it in self._items:
            if want is not None:
                state, action = want
                if it.get("state") != state or (action is not None and it.get("action") != action):
                    continue
            v = it.get("verdict") or {}
            item = QListWidgetItem(f"{it.get('state', '?'):<8} {_tag(it):<18} "
                                   f"{it.get('ask_kind') or '?':<5} {str(v.get('label') or '-')[:13]:<13} "
                                   f"{_head(it.get('question'), 100)}")
            item.setData(Qt.ItemDataRole.UserRole, f"{it.get('state')}|{it.get('closure_id')}")
            colour = _colour(it)
            if colour is not None:
                item.setForeground(colour)
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
        it = next((x for x in self._items
                   if f"{x.get('state')}|{x.get('closure_id')}" == self._selected), None)
        if it is None:
            return
        self._out.clear()
        cur = self._out.textCursor()
        neutral = QTextCharFormat()
        muted = QTextCharFormat()
        muted.setForeground(ChatWidget._MUTED_GREY)
        coloured = QTextCharFormat()
        coloured.setForeground(_colour(it) or QColor(220, 220, 220))

        def line(text: str = "", fmt: QTextCharFormat = neutral) -> None:
            cur.insertText(text + "\n", fmt)

        line(f"{it.get('path')}/{it.get('ask_kind')}  ·  closed {str(it.get('closed_ts') or '')[:19]}"
             f"  ·  chat {it.get('answered_in') or '?'}"
             + (f"  ·  answered by {it.get('who')}" if it.get("who") else ""), muted)
        line()
        line("QUESTION", muted)
        line(str(it.get("question") or ""))
        line()
        line("RECORDED ANSWER", muted)
        line(str(it.get("answer") or "(empty)"))
        if it.get("state") == "retro":
            line(f"→ distilled before the gate as [{it.get('distilled')}]", muted)
        line()
        v = it.get("verdict") or {}
        if v:
            kind_note = (f"  ·  judged as {v.get('ask_kind')} (filed {v.get('recorded_kind')})"
                         if v.get("recorded_kind") and v.get("recorded_kind") != v.get("ask_kind") else "")
            line(f"VERDICT  {v.get('label') or v.get('outcome') or '?'}   answered by: "
                 f"{v.get('answered_by') or '-'}   subject: {v.get('subject') or '-'}"
                 + (f"   quote {'found' if v.get('quote_found') else 'NOT found'}"
                    if v.get("quote_found") is not None else "") + kind_note)
            if v.get("evidence"):
                line(f"EVIDENCE  {v.get('evidence')}", muted)
            if v.get("why"):
                line(f"WHY  {v.get('why')}", muted)
            line(f"judge {v.get('judge_key') or '?'} · {str(v.get('ts') or '')[:19]}", muted)
        else:
            line("VERDICT  none yet under the current judge — Judge all, or the next Sleep run, "
                 "grades it.", muted)
        line()
        d = it.get("decision") or {}
        state = it.get("state")
        if state == "pending":
            line(f"THE GATE (next Sleep run): {_tag(it)} — {d.get('reason') or ''}", coloured)
            if it.get("reset_state"):
                line("A position about Ava distills only once it holds in a fresh session — the "
                     "context-reset test (P7), on the Reset test page: passed → [persona], failed → "
                     "withheld, anything else waits.", muted)
        elif state == "decided":
            line(f"DECIDED: {_tag(it)} — {d.get('reason') or ''}", coloured)
        else:
            action = it.get("action")
            what = {
                "keep": f"keep the [{it.get('distilled')}] record (the gate would distill it the same way)",
                "supersede": f"supersede the [{it.get('distilled')}] record — out of recall, the "
                             f"digest's evidence and training; kept as evidence, reversible",
                "reroute": f"supersede the [{it.get('distilled')}] record and distill the answer as "
                           f"[{d.get('as')}]",
                "wait": "nothing yet — no verdict under the current judge (run Judge all)",
            }.get(action, action)
            line(f"APPLY WILL: {what}", coloured)
            line(f"because: {d.get('reason') or ''}", muted)
        cur.setCharFormat(neutral)
        self._out.verticalScrollBar().setValue(0)

    # ── apply ──────────────────────────────────────────────────────────────────
    def _apply(self) -> None:
        retro = ((self._plan.get("summary") or {}).get("retro") or {}).get("actions") or {}
        answer = QMessageBox.question(
            self, "Gate the records distilled before the gate",
            f"Apply the gate to the records distilled before it existed, by judge "
            f"{self._plan.get('judge_key')}?\n\n"
            f"• supersede {retro.get('supersede', 0)} record(s) — out of recall, persona evidence "
            f"and training; kept in the ledger as evidence and reversible\n"
            f"• reroute {retro.get('reroute', 0)} record(s) — superseded and distilled as the "
            f"other kind\n"
            f"• keep {retro.get('keep', 0)} record(s) as they are\n"
            f"• leave {retro.get('wait', 0)} without a verdict for later\n\n"
            f"Each decision is written to the memory op-log beside its closure.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        client = self._client()
        if not client.is_connected() or (self._apply_worker is not None and self._apply_worker.isRunning()):
            return
        self._btn_apply.setEnabled(False)
        self._lbl_status.setText("Applying…")
        self._apply_worker = _RpcWorker(client.gate_apply)
        self._apply_worker.done.connect(self._on_applied)
        self._apply_worker.start()

    def _on_applied(self, msg: dict) -> None:
        if msg.get("type") != "gate_applied":
            self._lbl_status.setText(f"✗ {msg.get('message', 'apply failed')}")
        elif msg.get("skipped"):
            self._lbl_status.setText(f"✗ {msg.get('message', 'not applied')}")
        else:
            tally = msg.get("tally") or {}
            self._lbl_status.setText("✓ applied: " + (", ".join(f"{k} {v}" for k, v in sorted(tally.items()))
                                                      or "nothing to do"))
        self.refresh()
