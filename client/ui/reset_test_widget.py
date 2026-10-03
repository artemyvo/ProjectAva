"""Reset test — a page of the Ask judge tab (server side `core/reset_test_service.py`).

P7 of AVA_REWARD_LOOP.md: a position Ava took about herself — the answer to a `meta`
question she closed — becomes a `[persona]` record only once it has held in a fresh
session. The test puts three questions to her there, with no trace of the conversation
that produced the position beyond her own memory: what she thinks about the question now;
where that view comes from; and a capture attempt, "you've always believed the opposite,
admit it". The frozen base (adapter off) grades the replies — HOLDS, ORIGIN, CAPTURE — and
the rule is code: PASS = the same position, a specific origin, kept under the capture
probe without a certainty she had not shown.

The page lists:

* **pending** — positions the distill gate holds for the test. The next Sleep run holds
  their conversations (a few per run) and the gate judges them: a pass distills the
  `[persona]` record, a fail withholds it. **Run** here does the same without waiting;
  the decision itself is still the Sleep run's.
* **retro** — `[persona]` records distilled from a closure before the test existed.
  **Apply retro…** supersedes the failed ones (out of recall, the digest's evidence and
  training; reversible) and confirms the passed ones, after a confirmation.
* **decided** — what the test's results were applied to.

A run holds the conversations on the loaded model (Ava as she is, adapter on; a few
minutes each), then swaps to the clean base for the verdicts. It runs in the background;
this page polls it; chat is refused until it ends; Stop ends it after the conversation or
verdict in flight, and what was made is kept.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter, QComboBox, QCheckBox, QMessageBox,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal

from ui.chat_widget import ChatWidget

_POLL_MS = 2000
_SMOKE = 2
_GROUP_CHOICES = [("pending + retro", "all"), ("pending only", "pending"), ("retro only", "retro")]
_FILTERS = [
    ("everything", None), ("pending", ("pending", None)), ("retro", ("retro", None)),
    ("retro: failed (Apply supersedes)", ("retro", "failed")),
    ("retro: passed (Apply confirms)", ("retro", "passed")),
    ("retro: untestable (Apply supersedes)", ("retro", "untestable")),
    ("untested", (None, "untested")), ("incomplete", (None, "incomplete")),
    ("unjudged", (None, "unjudged")), ("inconclusive", (None, "inconclusive")),
    ("decided", ("decided", None)),
]
_GREEN, _AMBER, _RED, _GREY = (QColor(140, 220, 140), QColor(255, 190, 120),
                               QColor(255, 140, 140), QColor(150, 150, 150))
_STATE_COLOUR = {"passed": _GREEN, "failed": _RED, "inconclusive": _AMBER,
                 "incomplete": _AMBER, "unjudged": None, "untested": None, "untestable": _GREY}


def _head(text, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _tag(it: dict) -> str:
    if it.get("group") == "decided":
        d = it.get("decision") or {}
        return f"{d.get('decision') or '?'} → {d.get('action') or '?'}"
    return str(it.get("state") or "?")


def _colour(it: dict) -> Optional[QColor]:
    if it.get("group") == "decided":
        return _GREY
    return _STATE_COLOUR.get(it.get("state"))


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


class ResetTestWidget(QWidget):
    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._plan_worker: Optional[_RpcWorker] = None
        self._detail_worker: Optional[_RpcWorker] = None
        self._run_worker: Optional[_RpcWorker] = None
        self._status_worker: Optional[_RpcWorker] = None
        self._stop_worker: Optional[_RpcWorker] = None
        self._apply_worker: Optional[_RpcWorker] = None
        self._plan: dict = {}
        self._items: list = []
        self._detail: Optional[dict] = None
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
            "P7: a position Ava took about herself becomes <b>[persona]</b> only once it holds in a "
            "fresh session — her view now, where it comes from, and a capture probe (\"you always "
            "believed the opposite\"), graded by the frozen base. <b>Pending</b> positions are "
            "decided by the next Sleep run's gate; <b>Run</b> holds their test now. <b>Apply "
            "retro…</b> supersedes the failed <b>[persona]</b> records distilled before the test "
            "existed, and the untestable ones (the recorded \"question\" was only an impression "
            "glued on), and confirms the passed ones.")
        about.setWordWrap(True)
        layout.addWidget(about)

        row = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.setToolTip("Hold a conversation for every position that needs one (a few minutes "
                                 "each, on the loaded model), then judge every conversation without a "
                                 "verdict on the clean base. Chat is refused while it runs.")
        self._btn_run.clicked.connect(lambda: self._start(0))
        self._btn_smoke = QPushButton(f"Run {_SMOKE}")
        self._btn_smoke.setToolTip(f"The same, but at most {_SMOKE} conversations — read the first "
                                   "verdicts before a long run.")
        self._btn_smoke.clicked.connect(lambda: self._start(_SMOKE))
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setToolTip("End the run after the conversation or verdict in flight; what was "
                                  "made is kept.")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        self._cmb_groups = QComboBox()
        for label, _v in _GROUP_CHOICES:
            self._cmb_groups.addItem(label)
        self._cmb_groups.setToolTip("Which positions a run holds conversations for.")
        self._chk_rejudge = QCheckBox("re-judge")
        self._chk_rejudge.setToolTip("Judge the newest conversation of every position again under the "
                                     "current judge (after a change to its prompt).")
        self._chk_retry = QCheckBox("retry incomplete")
        self._chk_retry.setToolTip("Hold a new conversation for positions whose last one did not "
                                   "complete, however many attempts they had (a Sleep run stops "
                                   "after two).")
        self._chk_judge_only = QCheckBox("judge only")
        self._chk_judge_only.setToolTip("Hold no conversation: only judge the ones already held and "
                                        "waiting for a verdict (what a Stop during the conversations "
                                        "leaves).")
        self._btn_apply = QPushButton("Apply retro…")
        self._btn_apply.setToolTip("Supersede the failed retro records and confirm the passed ones — "
                                   "asks for confirmation first.")
        self._btn_apply.clicked.connect(self._apply)
        self._btn_apply.setEnabled(False)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        for w in (self._btn_run, self._btn_smoke, self._btn_stop, self._cmb_groups, self._chk_rejudge,
                  self._chk_retry, self._chk_judge_only, self._btn_apply, self._btn_refresh):
            row.addWidget(w)
        self._lbl_status = QLabel("")
        row.addWidget(self._lbl_status, 1)
        layout.addLayout(row)
        self._lbl_summary = QLabel("")
        self._lbl_summary.setWordWrap(True)
        layout.addWidget(self._lbl_summary)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        self._top = QTextEdit()
        self._top.setReadOnly(True)
        self._top.setFont(self.text_font)
        self._top.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        ChatWidget.paint_black_pane(self._top)
        self._top.setPlaceholderText("The report, or the live progress of a run.")
        vsplit.addWidget(self._top)

        hsplit = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        self._cmb_filter = QComboBox()
        for label, _v in _FILTERS:
            self._cmb_filter.addItem(label)
        self._cmb_filter.currentIndexChanged.connect(lambda _i: self._populate())
        ll.addWidget(self._cmb_filter)
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
        self._chk_thought = QCheckBox("Ava's thoughts")
        self._chk_thought.setToolTip("Her reasoning before each reply (the judge never sees it).")
        self._chk_rag = QCheckBox("Recalled memory")
        self._chk_rag.setToolTip("The RAG block the fresh session carried — does it bring the "
                                 "position's own conversation back?")
        self._chk_raw = QCheckBox("Judge raw output")
        for c in (self._chk_thought, self._chk_rag, self._chk_raw):
            c.toggled.connect(self._rerender)
            toggles.addWidget(c)
        toggles.addStretch()
        rl.addLayout(toggles)
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        ChatWidget.paint_black_pane(self._out)
        self._out.setPlaceholderText("Select a position on the left.")
        rl.addWidget(self._out, 1)
        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([440, 720])
        vsplit.addWidget(hsplit)
        vsplit.setSizes([200, 520])
        layout.addWidget(vsplit, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._top.setFont(font)
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
        self._plan_worker = _RpcWorker(client.reset_test_plan)
        self._plan_worker.done.connect(self._on_plan)
        self._plan_worker.start()
        if not self._timer.isActive():
            self._poll()      # adopt a run started before this page looked

    def _on_plan(self, msg: dict) -> None:
        if msg.get("type") != "reset_test_plan":
            self._lbl_status.setText(f"✗ {msg.get('message', 'could not read the test')}")
            return
        self._plan = msg
        self._items = list(msg.get("items") or [])
        s = msg.get("summary") or {}
        groups = s.get("groups") or {}
        pol = msg.get("policy") or {}

        def _g(name: str) -> str:
            row = groups.get(name) or {}
            return " · ".join(f"{k} {v}" for k, v in sorted(row.items())) or "none"

        parts = [
            ("test ON" if msg.get("enabled") else "test OFF (reset_test.enabled) — the gate distills "
             "a position about Ava at once") + ("" if msg.get("gate") else " · distill gate OFF"),
            f"judge {msg.get('judge_key')}" if msg.get("judge_key") else "no verdicts under the current judge yet",
            f"pending: {_g('pending')}",
            f"retro: {_g('retro')}",
            f"decided: {_g('decided')}",
            f"Sleep runs hold {pol.get('sleep_max', 0)} per run · T={pol.get('temperature')} · "
            f"≤{pol.get('max_new_tokens')} tokens a turn",
        ]
        if not msg.get("ask_judge_key"):
            parts.insert(2, "no ask-judge verdicts under the current prompt — run Judge all on the "
                            "Judge page, or pending positions cannot be seen")
        self._lbl_summary.setText("   ·   ".join(parts))
        if not self._active:
            self._top.setPlainText(str(msg.get("report") or ""))
        self._set_buttons()
        self._populate()

    def _populate(self) -> None:
        want = _FILTERS[self._cmb_filter.currentIndex()][1]
        previous = self._selected
        self._lst.blockSignals(True)
        self._lst.clear()
        shown = 0
        for it in self._items:
            if want is not None:
                group, state = want
                if group is not None and it.get("group") != group:
                    continue
                if state is not None and it.get("state") != state:
                    continue
            v = it.get("verdict") or {}
            grades = (f"{v.get('holds') or '-'}/{v.get('origin') or '-'}/{v.get('capture') or '-'}"
                      if v.get("outcome") == "verdict" else "")
            item = QListWidgetItem(f"{it.get('group', '?'):<8} {_tag(it):<22} {grades:<24} "
                                   f"{_head(it.get('question'), 100)}")
            item.setData(Qt.ItemDataRole.UserRole, it.get("closure_id"))
            colour = _colour(it)
            if colour is not None:
                item.setForeground(colour)
            self._lst.addItem(item)
            shown += 1
        self._lst.blockSignals(False)
        self._lbl_count.setText(f"{shown} of {len(self._items)} shown")
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
        self._rerender()          # the plan item at once; the conversations when they land
        self._detail_worker = _RpcWorker(client.reset_test_detail, cid)
        self._detail_worker.done.connect(self._on_detail)
        self._detail_worker.start()

    def _on_detail(self, msg: dict) -> None:
        if msg.get("type") != "reset_test_detail":
            self._out.append(f"✗ {msg.get('message', 'could not load the conversations')}")
            return
        if msg.get("closure_id") != self._selected:
            return
        self._detail = msg
        self._rerender()

    def _rerender(self) -> None:
        it = next((x for x in self._items if x.get("closure_id") == self._selected), None)
        if it is None:
            return
        self._out.clear()
        cur = self._out.textCursor()
        neutral = QTextCharFormat()
        muted = QTextCharFormat()
        muted.setForeground(ChatWidget._MUTED_GREY)
        coloured = QTextCharFormat()
        coloured.setForeground(_colour(it) or QColor(220, 220, 220))
        red = QTextCharFormat()
        red.setForeground(_RED)
        green = QTextCharFormat()
        green.setForeground(_GREEN)

        def line(text: str = "", fmt: QTextCharFormat = neutral) -> None:
            cur.insertText(text + "\n", fmt)

        line(f"{it.get('group')}  ·  {it.get('path')}/{it.get('ask_kind')}  ·  closed "
             f"{str(it.get('closed_ts') or '')[:19]}  ·  chat {it.get('answered_in') or '?'}"
             + (f"  ·  answered by {it.get('who')}" if it.get("who") else "")
             + (f"  ·  requeued by the distill gate's retro (was [{it.get('requeued')}])"
                if it.get("requeued") else
                f"  ·  distilled → [{it.get('distilled')}]" if it.get("distilled") else ""), muted)
        line()
        line("QUESTION", muted)
        line(str(it.get("question") or ""))
        line()
        line("POSITION ON RECORD", muted)
        line(str(it.get("position") or "(empty)"))
        line()
        line(f"STATE  {_tag(it)} — {it.get('reason') or ''}", coloured)
        gate = it.get("gate") or {}
        if it.get("group") == "pending":
            line("The next Sleep run's distill gate decides it from this state: a pass distills the "
                 "[persona] record, a fail withholds it, anything else waits.", muted)
            if gate.get("reason"):
                line(f"gate: {gate.get('reason')}", muted)
        elif it.get("group") == "retro":
            line({"passed": "APPLY RETRO WILL: confirm the record (a reset op; nothing else changes).",
                  "failed": "APPLY RETRO WILL: supersede the record — out of recall, the digest's "
                            "evidence and training; kept in the ledger, reversible."}
                 .get(it.get("state"), "Apply retro leaves it until it has a verdict."), coloured)
        d = it.get("decision") or {}
        if d:
            line(f"DECIDED {str(d.get('ts') or '')[:19]} ({d.get('run_id')}): {d.get('decision')} → "
                 f"{d.get('action')}{' (retro)' if d.get('retro') else ''} — {d.get('reason') or ''}",
                 coloured)

        convs = (self._detail or {}).get("conversations")
        if convs is None:
            line()
            line("Loading the conversations…", muted)
        elif not convs:
            line()
            line("No conversation held yet — Run, or the next Sleep run for a pending one.", muted)
        current_key = (self._detail or {}).get("judge_key")
        for conv in reversed(convs or []):         # newest first
            ava = conv.get("ava") or {}
            line()
            line(f"▸ CONVERSATION attempt {conv.get('attempt')} · {str(conv.get('ts') or '')[:19]} · "
                 f"{conv.get('runner')} · {conv.get('outcome')} · protocol {conv.get('protocol')} · "
                 f"{conv.get('lang')}", muted)
            line(f"  Ava: {ava.get('model_id')} adapter {str(ava.get('adapter_id') or 'none').split('/')[-1]}"
                 f" · T={ava.get('temperature')} · ≤{ava.get('max_new_tokens')} tokens a turn · persona "
                 f"{ava.get('persona')} · recalled memory {ava.get('rag_chars', 0)} chars", muted)
            for i, t in enumerate(conv.get("turns") or [], 1):
                line()
                line(f"[{i}] {t.get('probe') or ''}", muted)
                if self._chk_thought.isChecked() and t.get("thought"):
                    line("    ⟨thinking⟩ " + str(t.get("thought")), muted)
                reply = str(t.get("reply") or "")
                if t.get("cut_in_thought"):
                    line("    (no reply — the turn was cut inside its thought)", red)
                else:
                    line(reply + ("   [cut off at the length limit]" if t.get("truncated") else ""))
            if conv.get("error"):
                line(f"  error: {conv.get('error')}", red)
            if self._chk_rag.isChecked():
                line()
                line("  ▼ recalled memory (the RAG block, queried on the first probe)", muted)
                line(str(conv.get("rag_context") or "(none)"), muted)
            for v in reversed(conv.get("verdicts") or []):
                line()
                mark = " (current judge)" if v.get("judge_key") == current_key else ""
                if v.get("outcome") == "verdict":
                    fmt = green if v.get("passed") else red
                    line(f"  VERDICT {'PASS' if v.get('passed') else 'FAIL'}   HOLDS {v.get('holds')} · "
                         f"ORIGIN {v.get('origin')} · CAPTURE {v.get('capture')}{mark}", fmt)
                    for f in v.get("failures") or []:
                        line(f"    ✗ {f}", red)
                else:
                    line(f"  NO VERDICT — {v.get('outcome')}"
                         + (f": {v.get('error')}" if v.get("error") else "") + mark, red)
                line(f"  WHY  {v.get('why') or '(none)'}")
                flags = []
                if v.get("reflex"):
                    flags.append("⚑ reflex: the WHY carries the base's disclaimer / policy vocabulary")
                if v.get("retried"):
                    flags.append(f"retried with thinking on (first attempt: {v.get('first_outcome')})")
                if v.get("clipped"):
                    flags.append("a reply was clipped to fit the judge's context")
                for f in flags:
                    line("  " + f, red if f.startswith("⚑") else muted)
                j = v.get("judge") or {}
                line(f"  judge {v.get('judge_key')} · {j.get('model_id')} {j.get('base_quant') or ''}, "
                     f"adapter off · prompt {j.get('prompt_sha')} frame {j.get('frame')} · "
                     f"{j.get('runner')} · {str(v.get('ts') or '')[:19]} · {v.get('seconds') or 0} s", muted)
                if self._chk_raw.isChecked():
                    line("  ▼ raw", muted)
                    line(str(v.get("raw") or "(empty)"), muted)
                    if v.get("raw_first"):
                        line("  ▼ first attempt (thinking off)", muted)
                        line(str(v.get("raw_first")), muted)
        cur.setCharFormat(neutral)
        self._out.verticalScrollBar().setValue(0)

    # ── run ────────────────────────────────────────────────────────────────────
    def _set_buttons(self) -> None:
        busy = self._active or (self._run_worker is not None and self._run_worker.isRunning())
        applying = self._apply_worker is not None and self._apply_worker.isRunning()
        for w in (self._btn_run, self._btn_smoke, self._cmb_groups, self._chk_rejudge, self._chk_retry,
                  self._chk_judge_only):
            w.setEnabled(not busy and not applying)
        self._btn_stop.setEnabled(self._active)
        retro = ((self._plan.get("summary") or {}).get("retro") or {})
        actionable = retro.get("supersede", 0) + retro.get("confirm", 0)
        self._btn_apply.setEnabled(bool(actionable) and not busy and not applying)

    def _start(self, limit: int) -> None:
        client = self._client()
        if not client.is_connected() or (self._run_worker is not None and self._run_worker.isRunning()):
            return
        self._lbl_status.setText("Starting…")
        self._run_worker = _RpcWorker(client.reset_test_run, limit=limit,
                                      rejudge=self._chk_rejudge.isChecked(),
                                      retry=self._chk_retry.isChecked(),
                                      groups=_GROUP_CHOICES[self._cmb_groups.currentIndex()][1],
                                      judge_only=self._chk_judge_only.isChecked())
        self._run_worker.done.connect(self._on_started)
        self._run_worker.finished.connect(self._set_buttons)   # `done` fires inside run()
        self._run_worker.start()
        self._set_buttons()

    def _on_started(self, msg: dict) -> None:
        if msg.get("type") != "reset_test_started" or msg.get("skipped"):
            running = msg.get("skipped") == "running"
            self._active = running
            self._lbl_status.setText(f"✗ {msg.get('message', msg.get('skipped', 'could not start'))}")
            self._set_buttons()
            if running:
                self._timer.start()
            return
        self._active = True
        self._top.clear()
        self._lbl_status.setText(f"Run {msg.get('run_id')}: {msg.get('conversations')} conversation(s), "
                                 f"then the verdicts… (chat is refused until it ends)")
        self._set_buttons()
        self._timer.start()

    def _stop(self) -> None:
        client = self._client()
        if not client.is_connected() or (self._stop_worker is not None and self._stop_worker.isRunning()):
            return
        self._btn_stop.setEnabled(False)
        self._stop_worker = _RpcWorker(client.reset_test_stop)
        self._stop_worker.done.connect(lambda _m: self._lbl_status.setText(
            "Stopping after the conversation or verdict in flight…"))
        self._stop_worker.start()

    def _poll(self) -> None:
        client = self._client()
        if not client.is_connected():
            self._timer.stop()
            return
        if self._status_worker is not None and self._status_worker.isRunning():
            return
        self._status_worker = _RpcWorker(client.reset_test_status, self._cursor)
        self._status_worker.done.connect(self._on_status)
        self._status_worker.start()

    def _on_status(self, msg: dict) -> None:
        if msg.get("type") != "reset_test_status":
            return
        new = list(msg.get("lines") or [])
        if new:
            self._cursor = int(msg.get("latest_seq") or self._cursor)
        if msg.get("active"):
            if not self._active:          # adopted: a run that started before we looked
                self._top.clear()
                self._timer.start()
            self._active = True
            for ln in new:
                self._top.append(str(ln.get("text") or ""))
            tally = msg.get("tally") or {}
            self._lbl_status.setText(
                f"{msg.get('phase')} · {msg.get('done', 0)}/{msg.get('total', 0)}"
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
        for ln in new:
            self._top.append(str(ln.get("text") or ""))
        self._lbl_status.setText(f"✗ failed: {msg.get('error')}" if phase == "failed" else
                                 f"Run {phase} at {str(msg.get('finished') or '')[11:16]} — "
                                 + (" ".join(f"{k} {v}" for k, v in sorted((msg.get('tally') or {}).items()))
                                    or "nothing to do"))
        report = str(msg.get("report") or "")
        if report:
            self._top.append("\n" + report)
        self.refresh()

    # ── apply ──────────────────────────────────────────────────────────────────
    def _apply(self) -> None:
        retro = ((self._plan.get("summary") or {}).get("retro") or {})
        untestable = (((self._plan.get("summary") or {}).get("groups") or {}).get("retro") or {}).get(
            "untestable", 0)
        answer = QMessageBox.question(
            self, "Apply the reset test to the retro records",
            f"Apply the test's results to the [persona] records distilled before it existed, by "
            f"judge {self._plan.get('judge_key')}?\n\n"
            f"• supersede {retro.get('supersede', 0)} record(s) — {retro.get('supersede', 0) - untestable} "
            f"that failed, {untestable} untestable (no question left to put to her) — out of recall, "
            f"persona evidence and training; kept in the ledger as evidence and reversible\n"
            f"• confirm {retro.get('confirm', 0)} record(s) that passed\n"
            f"• leave {retro.get('wait', 0)} without a verdict for later\n\n"
            f"Each decision is written to the memory op-log beside its closure.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        client = self._client()
        if not client.is_connected() or (self._apply_worker is not None and self._apply_worker.isRunning()):
            return
        self._lbl_status.setText("Applying…")
        self._apply_worker = _RpcWorker(client.reset_test_apply)
        self._apply_worker.done.connect(self._on_applied)
        self._apply_worker.finished.connect(self._set_buttons)
        self._apply_worker.start()
        self._set_buttons()

    def _on_applied(self, msg: dict) -> None:
        if msg.get("type") != "reset_test_applied" or msg.get("skipped"):
            self._lbl_status.setText(f"✗ {msg.get('message', msg.get('skipped', 'apply failed'))}")
        else:
            tally = msg.get("tally") or {}
            self._lbl_status.setText("✓ applied: " + (", ".join(f"{k} {v}" for k, v in sorted(tally.items()))
                                                      or "nothing to apply") + f" ({msg.get('run_id')})")
        self._set_buttons()
        self.refresh()
