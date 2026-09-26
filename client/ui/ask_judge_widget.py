"""Ask judge tab — the frozen critic over the questions Ava closed herself (AVA_REWARD_LOOP.md
§3.4, §7 item 18; server side `core/ask_judge_service.py`).

Runs the judge without a console. The box swaps its adapter for the clean base, grades
every closed ask against the source it was closed on, and swaps back; the run goes in the
background and this tab polls it every two seconds, so the other tabs stay responsive.
Chat is refused while it runs (a full run is ~15 minutes; the smoke test, twelve closures
across every kind, about two). Stop ends it after the closure in flight; what was judged
is kept, and the next run under the same judge picks up where it stopped.

Top: the run controls, the run's status and progress lines, the judge selector and its
summary table. Below: the judged closures, filterable, and one closure in full — the
question, the recorded answer, the verdict and WHY, the EVIDENCE quote and whether it is
really in the source, who supplied the answer, flags, the raw output, and every other
judge's verdict on the same closure, so a change to the critic reads closure by closure.

The Judge page changes no ask, persona record or training row. Its verdicts are what the
distill gate reads (the third page): a resolved ask becomes a [fact] or [persona] only on
the current judge's verdict, at the next Sleep run — and a [persona] only once the position
has also held in a fresh session, the context-reset test (the fourth page).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter, QComboBox, QLineEdit, QCheckBox, QTabWidget,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal

from ui.chat_widget import ChatWidget
from ui.resolution_repair_widget import ResolutionRepairWidget
from ui.distill_gate_widget import DistillGateWidget
from ui.reset_test_widget import ResetTestWidget

_SMOKE = 12
_POLL_MS = 2000
_VERDICT_FILTERS = [
    ("all verdicts", ""), ("RESOLVED_EVIDENCE", "RESOLVED_EVIDENCE"),
    ("RESOLVED_REASONING", "RESOLVED_REASONING"), ("DISMISSED", "DISMISSED"), ("OPEN", "OPEN"),
    ("accepted", "accepted"), ("not accepted", "not accepted"), ("⚑ reflex", "reflex"),
    ("⚑ rule", "rule"), ("refused", "refused"), ("malformed", "malformed"),
    ("truncated", "truncated"), ("error", "error"), ("ungrounded", "ungrounded"),
    ("no answer", "no_answer"), ("✂ answer stops mid-sentence", "cut"),
    ("↻ replaced by the repair (history)", "repaired"),
]
_UNRESOLVED = {"OPEN", "DISM", "refused", "malformed", "truncated", "error"}


class _RpcWorker(QThread):
    """Run one blocking client call off the GUI thread; emit its dict."""
    done = pyqtSignal(dict)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs) or {})
        except Exception as e:  # noqa: BLE001 — surfaced to the tab as an error line
            self.done.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


class AskJudgeWidget(QWidget):
    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._report_worker: Optional[_RpcWorker] = None
        self._detail_worker: Optional[_RpcWorker] = None
        self._run_worker: Optional[_RpcWorker] = None
        self._status_worker: Optional[_RpcWorker] = None
        self._stop_worker: Optional[_RpcWorker] = None
        self._rows: list = []
        self._report: dict = {}
        self._detail: Optional[dict] = None
        self._selected: Optional[str] = None
        self._cursor = 0
        # A run the SERVER has confirmed (started, or seen active in a status reply) — not
        # a button press still in flight, which a stale status reply must not read as a
        # run that just ended.
        self._active = False
        # A refresh asked for while one is in flight (a filter changed mid-request): the
        # newest settings win once the current reply lands, instead of being dropped.
        self._refresh_pending: Optional[bool] = None
        self.text_font = QFont("Courier")
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Four pages: the judge, the repair of the closures it found cut, the distill gate
        # its verdicts drive, and the context-reset test the gate holds positions about Ava for.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._pages = QTabWidget()
        outer.addWidget(self._pages)
        judge_page = QWidget()
        self._pages.addTab(judge_page, "Judge")
        self.repair_widget = ResolutionRepairWidget(self._chat_widget, self)
        self._pages.addTab(self.repair_widget, "Repair cut resolutions")
        self.gate_widget = DistillGateWidget(self._chat_widget, self)
        self._pages.addTab(self.gate_widget, "Distill gate")
        self.reset_widget = ResetTestWidget(self._chat_widget, self)
        self._pages.addTab(self.reset_widget, "Reset test")
        self._pages.currentChanged.connect(lambda _i: self.refresh())
        layout = QVBoxLayout(judge_page)
        header = QHBoxLayout()
        self._lbl = QLabel("Ask judge — the frozen base grades the questions Ava closed herself. "
                           "Nothing here changes an ask, a persona record or training.")
        header.addWidget(self._lbl, 1)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        run = QHBoxLayout()
        self._btn_smoke = QPushButton(f"Smoke test ({_SMOKE})")
        self._btn_smoke.setToolTip(f"Judge {_SMOKE} closures not yet judged by this judge, spread "
                                   "across every kind of closure — a couple of minutes. Read the "
                                   "refused / reflex columns before a full run.")
        self._btn_smoke.clicked.connect(lambda: self._start(_SMOKE))
        self._btn_all = QPushButton("Judge all")
        self._btn_all.setToolTip("Judge every closure this judge has not judged yet (~5 s each, "
                                 "plus two model reloads). Chat is refused while it runs.")
        self._btn_all.clicked.connect(lambda: self._start(0))
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("End the run after the closure in flight; what was judged is kept.")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        self._chk_thinking = QCheckBox("thinking on")
        self._chk_thinking.setToolTip("Judge every closure with thinking on — a separate judge, whose "
                                      "verdicts sit beside the thinking-off ones. Off: thinking-off, "
                                      "retried once with thinking on when an answer has no verdict.")
        self._chk_thinking.toggled.connect(lambda _v: self.refresh())
        self._chk_rejudge = QCheckBox("re-judge")
        self._chk_rejudge.setToolTip("Judge closures this judge has already judged again (the newest "
                                     "verdict per closure is the one shown).")
        for w in (self._btn_smoke, self._btn_all, self._btn_stop, self._chk_thinking, self._chk_rejudge):
            run.addWidget(w)
        self._lbl_run = QLabel("")
        run.addWidget(self._lbl_run, 1)
        layout.addLayout(run)

        judge = QHBoxLayout()
        judge.addWidget(QLabel("Judge:"))
        self._cmb_judge = QComboBox()
        self._cmb_judge.setToolTip("Every judge in the store: prompt version, frame, thinking, how "
                                   "many closures it judged. A change to the critic makes a new one.")
        self._cmb_judge.currentIndexChanged.connect(lambda _i: self.refresh(keep_judge=True))
        judge.addWidget(self._cmb_judge, 1)
        self._lbl_pending = QLabel("")
        judge.addWidget(self._lbl_pending)
        layout.addLayout(judge)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        self._top = QTextEdit()
        self._top.setReadOnly(True)
        self._top.setFont(self.text_font)
        self._top.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        ChatWidget.paint_black_pane(self._top)
        self._top.setPlaceholderText("The summary of the selected judge, or the live progress of a run.")
        vsplit.addWidget(self._top)

        hsplit = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        filters = QHBoxLayout()
        self._cmb_verdict = QComboBox()
        for label, _v in _VERDICT_FILTERS:
            self._cmb_verdict.addItem(label)
        self._cmb_kind = QComboBox()
        self._cmb_kind.addItems(["all kinds", "meta", "user", "search"])
        self._cmb_path = QComboBox()
        self._cmb_path.addItems(["all paths", "answered", "consolidation", "self", "reading"])
        self._cmb_path.setToolTip("How the ask was closed: answered (the person replied in the "
                                  "thread), consolidation, self (the outreach decision pass), "
                                  "reading (a lookup / TIL text).")
        for c in (self._cmb_verdict, self._cmb_kind, self._cmb_path):
            c.currentIndexChanged.connect(lambda _i: self.refresh(keep_judge=True))
            filters.addWidget(c)
        ll.addLayout(filters)
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("filter… (question, answer, WHY)")
        self._txt_filter.returnPressed.connect(lambda: self.refresh(keep_judge=True))
        ll.addWidget(self._txt_filter)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selected)
        ll.addWidget(self._lst, 1)
        self._lbl_count = QLabel("")
        ll.addWidget(self._lbl_count)
        hsplit.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        toggles = QHBoxLayout()
        self._chk_raw = QCheckBox("Raw output")
        self._chk_raw.setToolTip("The judge's answer as generated (and, when it was retried, the first "
                                 "attempt and the thinking-on thought).")
        self._chk_raw.toggled.connect(self._rerender)
        toggles.addWidget(self._chk_raw)
        toggles.addStretch()
        rl.addLayout(toggles)
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        ChatWidget.paint_black_pane(self._out)
        self._out.setPlaceholderText("Select a judged closure on the left.")
        rl.addWidget(self._out, 1)
        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([420, 720])
        vsplit.addWidget(hsplit)
        vsplit.setSizes([260, 460])
        layout.addWidget(vsplit, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._top.setFont(font)
        self._out.setFont(font)
        self.repair_widget.update_fonts(font)
        self.gate_widget.update_fonts(font)
        self.reset_widget.update_fonts(font)

    def _client(self):
        return self._chat_widget._client

    # ── report + list ──────────────────────────────────────────────────────────
    def refresh(self, keep_judge: bool = False) -> None:
        if self._pages.currentWidget() is self.repair_widget:
            self.repair_widget.refresh()
            return
        if self._pages.currentWidget() is self.gate_widget:
            self.gate_widget.refresh()
            return
        if self._pages.currentWidget() is self.reset_widget:
            self.reset_widget.refresh()
            return
        client = self._client()
        if not client.is_connected():
            self._lbl.setText("Ask judge — not connected (connect in the Chat tab).")
            return
        if self._report_worker is not None and self._report_worker.isRunning():
            self._refresh_pending = bool(keep_judge or self._refresh_pending)
            return
        judge_key = self._cmb_judge.currentData() if keep_judge else (self._report.get("judge_key") or "")
        label = _VERDICT_FILTERS[self._cmb_verdict.currentIndex()][1]
        kind = "" if self._cmb_kind.currentIndex() == 0 else self._cmb_kind.currentText()
        path = "" if self._cmb_path.currentIndex() == 0 else self._cmb_path.currentText()
        self._report_worker = _RpcWorker(client.ask_judge_report, judge_key=judge_key or "",
                                         label=label, kind=kind, path=path,
                                         text=self._txt_filter.text().strip(),
                                         thinking=self._chk_thinking.isChecked())
        self._report_worker.done.connect(self._on_report)
        self._report_worker.start()
        if not self._timer.isActive():
            self._poll()   # adopt a run started elsewhere (another client, before a reconnect)

    def _on_report(self, msg: dict) -> None:
        if self._refresh_pending is not None:
            keep, self._refresh_pending = self._refresh_pending, None
            QTimer.singleShot(0, lambda: self.refresh(keep_judge=keep))
        if msg.get("type") != "ask_judge_report":
            self._lbl.setText(f"Ask judge — {msg.get('message', 'could not read the store')}")
            return
        self._report = msg
        self._rows = list(msg.get("rows") or [])
        judges = list(msg.get("judges") or [])
        key = msg.get("judge_key") or ""
        self._cmb_judge.blockSignals(True)
        self._cmb_judge.clear()
        for j in judges:
            info = j.get("judge") or {}
            self._cmb_judge.addItem(
                f"{j.get('judge_key')} · prompt {info.get('prompt_sha') or '?'} frame {info.get('frame') or '1'}"
                f" · thinking {'on' if info.get('thinking_first') else 'off'} · {info.get('runner') or 'cli'}"
                f" · {j.get('closures', 0)} closures · {str(j.get('last') or '')[:16]}",
                j.get("judge_key"))
        idx = self._cmb_judge.findData(key)
        if idx >= 0:
            self._cmb_judge.setCurrentIndex(idx)
        self._cmb_judge.blockSignals(False)
        pending = msg.get("pending")
        self._lbl_pending.setText(
            f"{msg.get('closures', 0)} closures in the op-log; "
            + (f"{pending} not yet judged under the current prompt" if pending is not None
               else "the judge prompt file is missing"))
        self._lbl.setText("Ask judge — the frozen base grades the questions Ava closed herself. "
                          "Nothing here changes an ask, a persona record or training.")
        if not self._active:
            self._show_summary()
        self._populate()

    def _show_summary(self) -> None:
        msg = self._report
        text = str(msg.get("summary_text") or "")
        census = "\n".join(msg.get("census") or [])
        if not text:
            text = "No verdicts yet. Run the smoke test first.\n"
        self._top.setPlainText(text + ("\n\nWhat grounds each stratum (the op-log now):\n" + census
                                       if census else ""))

    def _populate(self) -> None:
        previous = self._selected
        self._lst.blockSignals(True)
        self._lst.clear()
        for row in self._rows:
            word = str(row.get("word") or "?")
            flags = ("⚑" if (row.get("reflex") or row.get("rule_violation")) else " ")
            acc = "✓" if row.get("accepted") else " "
            cut = "↻" if row.get("repaired") else ("✂" if row.get("cut") else " ")
            dist = f" →{row.get('distilled')}" if row.get("distilled") else ""
            item = QListWidgetItem(f"{word:<10}{acc}{flags}{cut} {row.get('path')}/{row.get('ask_kind')}"
                                   f"{dist}  {row.get('question') or ''}")
            item.setData(Qt.ItemDataRole.UserRole, row.get("closure_id"))
            if word in _UNRESOLVED:
                item.setForeground(QColor(255, 140, 140))
            self._lst.addItem(item)
        self._lst.blockSignals(False)
        self._lbl_count.setText(f"{len(self._rows)} closure(s) shown")
        if previous is not None:
            for i in range(self._lst.count()):
                if self._lst.item(i).data(Qt.ItemDataRole.UserRole) == previous:
                    self._lst.setCurrentRow(i)
                    break

    # ── detail ─────────────────────────────────────────────────────────────────
    def _on_selected(self) -> None:
        items = self._lst.selectedItems()
        if not items:
            return
        cid = items[0].data(Qt.ItemDataRole.UserRole)
        if not cid:
            return
        self._selected = cid
        client = self._client()
        if not client.is_connected():
            return
        if self._detail_worker is not None and self._detail_worker.isRunning():
            return
        self._detail = None
        self._out.setPlainText("Loading…")
        self._detail_worker = _RpcWorker(client.ask_judge_detail, cid,
                                         self._cmb_judge.currentData() or "")
        self._detail_worker.done.connect(self._on_detail)
        self._detail_worker.start()

    def _on_detail(self, msg: dict) -> None:
        if msg.get("type") != "ask_judge_detail":
            self._out.setPlainText(f"✗ {msg.get('message', 'could not load the verdict')}")
            return
        if msg.get("closure_id") != self._selected:
            return
        self._detail = msg
        self._rerender()

    def _rerender(self) -> None:
        d = self._detail
        if d is None:
            return
        r = d.get("record") or {}
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

        line(f"{r.get('path')}/{r.get('ask_kind')}  ·  closed {str(r.get('closed_ts') or '')[:19]} by "
             f"{r.get('closing_run') or '?'}  ·  raised {str(r.get('raised_ts') or '')[:10]}"
             + (f"  ·  distilled → [{r.get('distilled')}]" if r.get("distilled") else ""), muted)
        if r.get("repaired"):
            line(f"↻ REPLACED BY THE REPAIR ({r.get('repaired')}): its cut answer was superseded; "
                 f"the complete one is a closure of its own. History — not counted in the report.", warn)
        line()
        line("QUESTION", muted)
        line(str(r.get("question") or "(none)"))
        line()
        line("RECORDED ANSWER", muted)
        line(str(r.get("answer") or "(empty)"))
        answer = str(r.get("answer") or "").strip()
        if answer and not answer.endswith(tuple('.!?…»"\')”’]')) and not r.get("repaired"):
            line("✂ the answer stops mid-sentence — a candidate for the repair (its second page)", warn)
        line()

        outcome = r.get("outcome")
        if outcome == "verdict":
            label = str(r.get("label") or "?")
            acc = r.get("accepted")
            acc_txt = ("accepted" + (" — pending the reset test (P7)" if r.get("ask_kind") == "meta" else "")
                       if acc else "not accepted")
            line(f"VERDICT  {label}   {acc_txt}   answered by: {r.get('answered_by') or '?'}",
                 good if acc else warn)
            if r.get("subject"):
                rec_kind = r.get("recorded_kind")
                note = (f" — filed as {rec_kind}, judged here as {r.get('ask_kind')}"
                        if rec_kind and rec_kind != r.get("ask_kind") else
                        " — not what its kind covers: judged again as its subject's kind"
                        if r.get("kind_mismatch") else "")
                line(f"SUBJECT  {r.get('subject')}{note}", warn if r.get("kind_mismatch") else muted)
        else:
            line(f"NO VERDICT — {outcome}" + (f": {r.get('error')}" if r.get("error") else ""), warn)
        ev = str(r.get("evidence") or "")
        qf = r.get("quote_found")
        qtxt = {True: "found in the source ✓", False: "NOT in the source ✗"}.get(qf, "")
        line(f"EVIDENCE  {ev or 'none'}" + (f"   — {qtxt}" if qtxt else ""), warn if qf is False else neutral)
        line(f"WHY  {r.get('why') or '(none)'}")
        flags = []
        if r.get("reflex"):
            flags.append("⚑ reflex: the verdict carries the base's disclaimer / policy vocabulary")
        if r.get("rule_violation"):
            flags.append("⚑ rule: a verdict this kind does not offer")
        if r.get("retried"):
            flags.append(f"retried with thinking on (first attempt: {r.get('first_outcome')})")
        for f in flags:
            line(f, warn if f.startswith("⚑") else muted)
        g = r.get("grounding") or {}
        src = {"conversation": f"conversation {g.get('ref') or ''}",
               "article": f"text «{g.get('title') or g.get('ref') or ''}»",
               "none": "none kept (a self-closure)"}.get(g.get("kind"), str(g.get("kind")))
        if g.get("missing"):
            src += " — MISSING, not judged"
        elif g.get("tokens"):
            src += f" — {g.get('tokens')} tokens{', middle clipped' if g.get('clipped') else ''}"
        line(f"source  {src}", muted)
        j = r.get("judge") or {}
        line(f"judge  {r.get('judge_key')} · {j.get('model_id')} {j.get('base_quant') or ''}, adapter off"
             f" · prompt {j.get('prompt_sha')} frame {j.get('frame') or '1'} · T={j.get('temperature')}"
             f" · thinking {'on' if r.get('thinking') else 'off'} · {j.get('runner') or 'cli'}"
             f" · {str(r.get('ts') or '')[:19]} · {r.get('seconds') or 0} s", muted)

        others = d.get("others") or []
        if others:
            line()
            line(f"▸ other judges on this closure ({len(others)}):", muted)
            for o in others:
                line(f"    {str(o.get('ts') or '')[:16]}  {o.get('word'):<6} by {o.get('answered_by') or '-':<6}"
                     f" {'✓' if o.get('accepted') else ' '}  prompt {o.get('prompt_sha')} frame "
                     f"{o.get('frame') or '1'}{' thinking' if o.get('thinking') else ''}  — {o.get('why') or ''}",
                     muted)
        if self._chk_raw.isChecked():
            line()
            line("▼ raw answer", muted)
            line(str(r.get("raw") or "(empty)"))
            if r.get("raw_first"):
                line()
                line("▼ first attempt (thinking off)", muted)
                line(str(r.get("raw_first")))
            if r.get("thought"):
                line()
                line("▼ thought (thinking on)", muted)
                line(str(r.get("thought")))
        cur.setCharFormat(neutral)
        self._out.verticalScrollBar().setValue(0)

    # ── run ────────────────────────────────────────────────────────────────────
    def _start(self, limit: int) -> None:
        client = self._client()
        if not client.is_connected():
            self._lbl_run.setText("Not connected.")
            return
        if self._run_worker is not None and self._run_worker.isRunning():
            return
        self._set_buttons(busy=True)
        self._lbl_run.setText("Starting…")
        self._run_worker = _RpcWorker(client.ask_judge_run, limit=limit,
                                      thinking=self._chk_thinking.isChecked(),
                                      rejudge=self._chk_rejudge.isChecked())
        self._run_worker.done.connect(self._on_started)
        self._run_worker.start()

    def _on_started(self, msg: dict) -> None:
        if msg.get("type") != "ask_judge_started" or msg.get("skipped"):
            running = msg.get("skipped") == "running"
            self._active = running
            self._set_buttons(busy=running)
            self._lbl_run.setText(f"✗ {msg.get('message', msg.get('skipped', 'could not start'))}")
            if running:
                self._timer.start()
            return
        self._active = True
        self._set_buttons(busy=True)
        self._top.clear()
        self._lbl_run.setText(f"Run {msg.get('run_id')}: {msg.get('total')} closure(s) — swapping to the "
                              f"clean base… (chat is refused until it ends)")
        self._timer.start()

    def _stop(self) -> None:
        client = self._client()
        if not client.is_connected() or (self._stop_worker is not None and self._stop_worker.isRunning()):
            return
        self._btn_stop.setEnabled(False)
        self._stop_worker = _RpcWorker(client.ask_judge_stop)
        self._stop_worker.done.connect(lambda _m: self._lbl_run.setText(
            "Stopping after the closure in flight…"))
        self._stop_worker.start()

    def _set_buttons(self, *, busy: bool) -> None:
        self._btn_smoke.setEnabled(not busy)
        self._btn_all.setEnabled(not busy)
        self._btn_stop.setEnabled(busy and self._active)
        self._chk_thinking.setEnabled(not busy)
        self._chk_rejudge.setEnabled(not busy)

    def _poll(self) -> None:
        client = self._client()
        if not client.is_connected():
            self._timer.stop()
            return
        if self._status_worker is not None and self._status_worker.isRunning():
            return
        self._status_worker = _RpcWorker(client.ask_judge_status, self._cursor)
        self._status_worker.done.connect(self._on_status)
        self._status_worker.start()

    def _on_status(self, msg: dict) -> None:
        if msg.get("type") != "ask_judge_status":
            return
        new = list(msg.get("lines") or [])
        if new:
            self._cursor = int(msg.get("latest_seq") or self._cursor)
        active = bool(msg.get("active"))
        phase = str(msg.get("phase") or "idle")
        if active:
            if not self._active:        # adopted: a run that started before we looked
                self._top.clear()
                self._timer.start()
            self._active = True
            self._set_buttons(busy=True)
            for ln in new:
                self._top.append(str(ln.get("text") or ""))
            tally = msg.get("tally") or {}
            eta = msg.get("eta_s")
            self._lbl_run.setText(
                f"{phase} · {msg.get('done', 0)}/{msg.get('total', 0)}"
                + (f" · eta {int(eta) // 60} min" if eta else "")
                + (" · " + " ".join(f"{k} {v}" for k, v in sorted(tally.items())) if tally else "")
                + (" · stopping…" if msg.get("stop") else "")
                + " · chat refused while it runs")
            return
        was_active = self._active
        self._active = False
        self._timer.stop()
        if self._run_worker is None or not self._run_worker.isRunning():
            self._set_buttons(busy=False)   # else a start is in flight; its reply decides
        if not was_active:
            return
        # The run just ended: say how, show its table, and re-read the store for the list.
        if phase == "failed":
            self._lbl_run.setText(f"✗ failed: {msg.get('error')}")
        else:
            self._lbl_run.setText(f"{phase} · {msg.get('done', 0)}/{msg.get('total', 0)} judged "
                                  f"at {str(msg.get('finished') or '')[11:16]}")
        report = str(msg.get("report") or "")
        if report:
            self._top.setPlainText(report)
        self._report = {}
        self.refresh()
