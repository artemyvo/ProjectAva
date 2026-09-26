"""Revisions tab — judge the judge (AVA_REWARD_LOOP.md §7 item 17, P9).

One row per exchange the revision pass has judged, from the revisions ledger
(``data/revisions/ledger.jsonl``, `core/revision_ledger.py` on the server): the latest
verdict, the one it replaced, the pass's WHY, how the pass ended, how many times the
exchange has been judged, and whether a human has marked it. Selecting a row shows the
original reply painted in the Chat tab's three channels (decoded server-side from the
transcript's stored token series, as the Chat review tab does), the revised target
beside it, the judgement (VERDICT / WHY / pass outcome / run), the signals the original
carried (tension summary, axis means, hidden captures), the full judgement history and
every annotation so far.

The two controls at the bottom are about the JUDGE, not the reply: was the verdict
justified, does the WHY describe what went wrong, and which failure kind was it. They
write an ``annotation`` event to the ledger and nothing else — no verdict, no target,
no lock, no ban. Nothing Ava reads or trains on sees them; they calibrate the judge
prompt, the axes and the probes. The Training review tab remains the place for a
curation decision (ban / lock / rewrite). Read-only otherwise.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter, QComboBox, QLineEdit, QCheckBox,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.chat_widget import ChatWidget

_SUB = "─" * 78


class _RpcWorker(QThread):
    """Run one blocking client call off the GUI thread; emit its dict."""
    done = pyqtSignal(dict)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs) or {})
        except Exception as e:  # noqa: BLE001 — surfaced to the tab as an error row
            self.done.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


def _tri_text(v) -> str:
    return "—" if v is None else ("yes" if v else "no")


class RevisionsWidget(QWidget):
    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._list_worker: Optional[_RpcWorker] = None
        self._detail_worker: Optional[_RpcWorker] = None
        self._annotate_worker: Optional[_RpcWorker] = None
        self._items: list = []
        self._detail: Optional[dict] = None
        self._selected: Optional[tuple] = None
        self.text_font = QFont("Courier")
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self._lbl = QLabel("Revisions — every judged exchange, with the judge's WHY. Mark whether the "
                           "judge was right; nothing here reaches training.")
        header.addWidget(self._lbl, 1)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: filters + the list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        filters = QHBoxLayout()
        self._cmb_verdict = QComboBox()
        self._cmb_verdict.addItems(["all verdicts", "revise", "keep"])
        self._cmb_verdict.setToolTip("The latest verdict on the exchange.")
        self._cmb_run = QComboBox()
        self._cmb_run.addItem("all runs")
        self._cmb_run.setToolTip("Which pass wrote the latest verdict: reflection (Sleep), background "
                                 "(the idle drain), revisit, branch_judge, manual_regen.")
        self._cmb_ann = QComboBox()
        self._cmb_ann.addItems(["all", "unannotated", "annotated"])
        self._cmb_ann.setToolTip("Whether a human has marked the judgement yet.")
        for c in (self._cmb_verdict, self._cmb_run, self._cmb_ann):
            c.currentIndexChanged.connect(lambda _i: self.refresh())
            filters.addWidget(c)
        ll.addLayout(filters)
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("filter… (session, WHY, speaker, user message)")
        self._txt_filter.returnPressed.connect(self.refresh)
        ll.addWidget(self._txt_filter)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selected)
        ll.addWidget(self._lst, 1)
        self._lbl_count = QLabel("")
        ll.addWidget(self._lbl_count)
        split.addWidget(left)

        # Right: the detail pane + controls
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Show:"))
        self._chk_channels = QCheckBox("Channels")
        self._chk_channels.setChecked(True)
        self._chk_channels.setToolTip("Paint the original reply in the Chat tab's three channels (red = "
                                      "tension, green = relief, blue = pain), decoded from the token "
                                      "series stored with the exchange. Untick for plain text.")
        self._chk_raw = QCheckBox("Raw judgement")
        self._chk_raw.setToolTip("The revision pass's full output as the writer parsed it.")
        self._chk_system = QCheckBox("System content")
        self._chk_system.setToolTip("The exact system message the original turn was conditioned on, "
                                    "as snapshotted at judgement time.")
        for chk in (self._chk_channels, self._chk_raw, self._chk_system):
            chk.toggled.connect(self._rerender)
            toggles.addWidget(chk)
        toggles.addStretch()
        rl.addLayout(toggles)
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        ChatWidget.paint_black_pane(self._out)
        self._out.setPlaceholderText("Select a judged exchange on the left.")
        rl.addWidget(self._out, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Verdict justified:"))
        self._cmb_justified = QComboBox()
        self._cmb_justified.addItems(["—", "yes", "no"])
        self._cmb_justified.setToolTip("Given the exchange, was keep / revise the right call? About "
                                       "the judge's decision, not about liking the reply.")
        controls.addWidget(self._cmb_justified)
        controls.addWidget(QLabel("WHY accurate:"))
        self._cmb_why = QComboBox()
        self._cmb_why.addItems(["—", "yes", "no"])
        self._cmb_why.setToolTip("Does the WHY name what actually went wrong (or right)?")
        controls.addWidget(self._cmb_why)
        controls.addWidget(QLabel("Tag:"))
        self._cmb_tag = QComboBox()
        self._cmb_tag.addItem("—")
        self._cmb_tag.setToolTip("The failure kind, if any: what the judge saw or should have seen.")
        controls.addWidget(self._cmb_tag)
        self._txt_note = QLineEdit()
        self._txt_note.setPlaceholderText("note (optional)")
        controls.addWidget(self._txt_note, 1)
        self._btn_save = QPushButton("Save annotation")
        self._btn_save.setToolTip("Append a calibration mark to the revisions ledger. It changes no "
                                  "verdict, target, lock or ban and is read by nothing Ava trains on.")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setEnabled(False)
        controls.addWidget(self._btn_save)
        rl.addLayout(controls)
        self._lbl_status = QLabel("")
        rl.addWidget(self._lbl_status)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 760])
        layout.addWidget(split, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._out.setFont(font)

    # ── list ───────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl.setText("Revisions — not connected (connect in the Chat tab).")
            return
        if self._list_worker is not None and self._list_worker.isRunning():
            return
        verdict = {"all verdicts": "", "revise": "revise", "keep": "keep"}[self._cmb_verdict.currentText()]
        run_kind = "" if self._cmb_run.currentIndex() == 0 else self._cmb_run.currentText()
        annotated = {"all": "", "unannotated": "no", "annotated": "yes"}[self._cmb_ann.currentText()]
        self._lbl.setText("Revisions — loading…")
        self._list_worker = _RpcWorker(client.list_revisions, verdict=verdict, run_kind=run_kind,
                                       annotated=annotated, text=self._txt_filter.text().strip())
        self._list_worker.done.connect(self._on_list)
        self._list_worker.start()

    def _on_list(self, msg: dict) -> None:
        if msg.get("type") != "revisions_list":
            self._lbl.setText(f"Revisions — {msg.get('message', 'could not load the ledger')}")
            return
        self._items = list(msg.get("items") or [])
        # run-kind filter options follow the ledger
        current = self._cmb_run.currentText()
        self._cmb_run.blockSignals(True)
        self._cmb_run.clear()
        self._cmb_run.addItem("all runs")
        for k in msg.get("run_kinds") or []:
            self._cmb_run.addItem(str(k))
        idx = self._cmb_run.findText(current)
        self._cmb_run.setCurrentIndex(idx if idx >= 0 else 0)
        self._cmb_run.blockSignals(False)
        tags = list(msg.get("tags") or [])
        if self._cmb_tag.count() <= 1:
            for t in tags:
                self._cmb_tag.addItem(t)
        s = msg.get("summary") or {}
        self._lbl.setText(f"Revisions — {s.get('verdict_events', 0)} verdict events over "
                          f"{s.get('distinct_exchanges', 0)} exchanges; {s.get('annotations', 0)} annotations. "
                          f"Mark whether the judge was right; nothing here reaches training.")
        self._populate()

    def _populate(self) -> None:
        previous = self._selected
        self._lst.blockSignals(True)
        self._lst.clear()
        for row in self._items:
            mark = "✓ " if row.get("n_annotations") else "  "
            prev = row.get("prev_verdict")
            verdict = str(row.get("verdict") or "?")
            vtxt = f"{prev}→{verdict}" if prev and prev != verdict else verdict
            n = row.get("n_judged") or 1
            why = str(row.get("why_head") or "").replace("\n", " ")
            label = (f"{mark}{(row.get('ts') or '')[:16]}  {str(row.get('session') or '').replace('.json', '')}"
                     f" #{int(row.get('exchange_index') or 0) + 1}  {vtxt}"
                     f"{'  ×' + str(n) if n > 1 else ''}  [{row.get('run_kind') or '?'}]"
                     f"{'  ' + why if why else ''}")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (row.get("session"), int(row.get("exchange_index") or 0)))
            if verdict == "revise":
                item.setForeground(QColor(255, 140, 140))
            self._lst.addItem(item)
        self._lst.blockSignals(False)
        self._lbl_count.setText(f"{len(self._items)} judged exchange(s) shown")
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
        key = items[0].data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        self._selected = tuple(key)
        client = self._chat_widget._client
        if not client.is_connected():
            return
        if self._detail_worker is not None and self._detail_worker.isRunning():
            return
        self._detail = None
        self._out.setPlainText(f"Loading {key[0]} #{key[1] + 1}…")
        self._btn_save.setEnabled(False)
        self._detail_worker = _RpcWorker(client.get_revision, key[0], key[1])
        self._detail_worker.done.connect(self._on_detail)
        self._detail_worker.start()

    def _on_detail(self, msg: dict) -> None:
        if msg.get("type") != "revision_detail":
            self._out.setPlainText(f"✗ {msg.get('message', 'could not load the judgement')}")
            return
        if (msg.get("session"), int(msg.get("exchange_index") or 0)) != self._selected:
            return
        self._detail = msg
        self._btn_save.setEnabled(True)
        self._rerender()

    def _rerender(self) -> None:
        d = self._detail
        if d is None:
            return
        latest = d.get("latest") or {}
        snap = latest.get("snapshot") or {}
        spans = d.get("tension_spans") if self._chk_channels.isChecked() else None
        spans = spans if isinstance(spans, dict) else {}
        chat = self._chat_widget
        self._out.clear()
        cur = self._out.textCursor()
        neutral = QTextCharFormat()
        muted = QTextCharFormat()
        muted.setForeground(ChatWidget._MUTED_GREY)
        warn = QTextCharFormat()
        warn.setForeground(QColor(255, 140, 140))

        def line(text: str = "", fmt: QTextCharFormat = neutral) -> None:
            cur.insertText(text + "\n", fmt)

        verdict = str(latest.get("verdict") or "?")
        prev = (latest.get("prev") or {}).get("verdict")
        line(f"{d.get('session')}  exchange {int(d.get('exchange_index') or 0) + 1}"
             f"  ·  {snap.get('speaker') or snap.get('session_user') or ''}"
             f"  ·  {snap.get('model_id') or ''} + {str(snap.get('adapter_id') or 'base').split('/')[-1]}", muted)
        line(f"VERDICT {verdict}" + (f"  (was {prev})" if prev and prev != verdict else "")
             + f"   run {latest.get('run_id') or '?'} [{latest.get('run_kind') or '?'}]   {latest.get('ts') or ''}",
             warn if verdict == "revise" else neutral)
        why = str(latest.get("why") or "").strip()
        line(f"WHY  {why or '(none recorded)'}")
        pi = latest.get("pass_info") or {}
        if pi:
            line("pass  " + "  ".join(f"{k}={v}" for k, v in pi.items()), muted)
        flags = []
        if latest.get("locked"):
            flags.append("locked")
        if latest.get("banned"):
            flags.append("banned")
        if flags:
            line("⚑ " + ", ".join(flags), warn)
        line()

        if self._chk_system.isChecked():
            line("▼ system content (as snapshotted at judgement)", muted)
            line(str(snap.get("system_content") or "(not recorded)"))
            line()
        line(f"▶ {snap.get('speaker') or 'user'}:", muted)
        line(str(snap.get("user_prompt") or "(empty)"))
        line()

        cot = str(snap.get("assistant_cot") or "").strip()
        if spans.get("cot"):
            line("▶ original CoT (channels):", muted)
            chat._insert_colored_spans(cur, spans["cot"])
            cur.insertText("\n", neutral)
        elif cot:
            line("▶ original CoT:", muted)
            line(cot)
        if spans.get("answer"):
            line("▶ original reply (channels):", muted)
            chat._insert_colored_spans(cur, spans["answer"])
            cur.insertText("\n", neutral)
            chat._append_channel_means(cur, neutral, spans=spans)
            cur.insertText("\n", neutral)
        else:
            line("▶ original reply:", muted)
            line(str(snap.get("assistant_response") or "(empty)"))
        line()

        target = str(latest.get("target") or "")
        line(f"▶ revised target  [{latest.get('target_source') or '?'} / {latest.get('target_kind') or '?'}]:", muted)
        if verdict == "keep":
            line("(kept — the target is the original reply)", muted)
        elif target:
            line(target)
        else:
            line("(no target on record)", muted)
        line()

        t = snap.get("tension") or {}
        if t:
            bits = []
            for seg in ("cot", "answer"):
                s = t.get(seg) or {}
                if s:
                    bits.append(f"{seg}: n={s.get('n_tokens')} contested={_pct(s.get('contested_frac'))} "
                                f"p10_margin={_f(s.get('p10_margin'))} peak_H={_f(s.get('peak_entropy'))}")
            line("▸ tension  " + "   ".join(bits), muted)
            axes = t.get("axes") or {}
            if axes:
                line("▸ axes     " + "   ".join(f"{k}: mean {_f(v.get('mean'))} max {_f(v.get('max'))}"
                                               for k, v in axes.items()), muted)
        hid = snap.get("hidden")
        if hid:
            line("▸ hidden   " + ", ".join(f"{h.get('variant') or 'live'} L{h.get('layers')} axes {h.get('axes')}"
                                          for h in hid), muted)
        elif hid == []:
            line("▸ hidden   none captured for this exchange", muted)
        fb = snap.get("reflection_feedback")
        if fb:
            line(f"▸ meta feedback  {fb}", muted)
        line()

        hist = d.get("history") or []
        if len(hist) > 1:
            line(f"▸ judged {len(hist)} times:", muted)
            for h in hist:
                line(f"    {h.get('ts') or ''}  {h.get('verdict')}  [{h.get('run_kind') or '?'}] "
                     f"{h.get('run_id') or ''}  — {str(h.get('why') or '')[:100]}", muted)
            line()
        anns = d.get("annotations") or []
        line(f"▸ annotations ({len(anns)})", muted)
        for a in anns:
            line(f"    {a.get('ts') or ''}  {a.get('author') or ''}  verdict justified: "
                 f"{_tri_text(a.get('verdict_justified'))}  WHY accurate: {_tri_text(a.get('why_accurate'))}"
                 f"  tag: {a.get('tag') or '—'}  {a.get('note') or ''}")
        if self._chk_raw.isChecked():
            line()
            line("▼ raw judgement", muted)
            line(str(latest.get("judgement_raw") or "(not recorded)"))
        cur.setCharFormat(neutral)
        self._out.verticalScrollBar().setValue(0)

    # ── annotate ───────────────────────────────────────────────────────────────
    def _on_save(self) -> None:
        d = self._detail
        if d is None or self._selected is None:
            return
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected.")
            return
        vj = {"—": None, "yes": True, "no": False}[self._cmb_justified.currentText()]
        wa = {"—": None, "yes": True, "no": False}[self._cmb_why.currentText()]
        tag = "" if self._cmb_tag.currentIndex() == 0 else self._cmb_tag.currentText()
        note = self._txt_note.text().strip()
        if vj is None and wa is None and not tag and not note:
            self._lbl_status.setText("Mark at least one thing before saving.")
            return
        if self._annotate_worker is not None and self._annotate_worker.isRunning():
            return
        latest = d.get("latest") or {}
        self._btn_save.setEnabled(False)
        self._lbl_status.setText("Saving…")
        self._annotate_worker = _RpcWorker(
            client.annotate_revision, self._selected[0], self._selected[1],
            exchange_id=d.get("exchange_id"), about_run_id=latest.get("run_id"),
            verdict_justified=vj, why_accurate=wa, tag=tag, note=note)
        self._annotate_worker.done.connect(self._on_saved)
        self._annotate_worker.start()

    def _on_saved(self, msg: dict) -> None:
        self._btn_save.setEnabled(True)
        if msg.get("type") != "revision_annotated" or not msg.get("ok"):
            self._lbl_status.setText(f"✗ {msg.get('message', 'annotation not saved')}")
            return
        self._lbl_status.setText("Annotation saved to the ledger.")
        self._cmb_justified.setCurrentIndex(0)
        self._cmb_why.setCurrentIndex(0)
        self._cmb_tag.setCurrentIndex(0)
        self._txt_note.clear()
        # Re-fetch: the detail shows the new mark, the list shows the ✓.
        self._on_selected()
        self.refresh()


def _f(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"
