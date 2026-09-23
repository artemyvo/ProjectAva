"""Prompt widget — the standing-prompt surface (view, hand-edit, experiment, revert).

Ava's live standing prompt has two tiers: the canonical ``chat_prompt.txt`` on the server,
and a temporary **experiment** in the mutable ``hot/prompt`` tier that the prompt loader
prefers while it is active (see ``server/inference/core/prompt_experiment.py``). This tab
is the one place both are visible and changeable, and every write it makes lands in the
temporary tier — ``chat_prompt.txt`` is never overwritten from here, so *any* change is
undone by **Revert prompt**.

Three ways to change the live prompt, all reverted the same way:

* **Update prompt** — apply what is in the editbox (the operator's own text).
* **Prompt experiment** — hand the current prompt to Ava and let her rewrite it freely;
  her process streams into the log below and the result becomes live on success.
* **Revert prompt** — drop the experiment and restore the base standing prompt.

The editbox opens on whatever is actually live: the experimental prompt when one is
active, else the base prompt. It is refreshed from the server (never remembered
client-side), so a restart or a reconnect shows the real state. Unsaved edits are never
silently discarded — a refresh that would clobber them asks first.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QPlainTextEdit,
    QLabel,
    QDoubleSpinBox,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont

if TYPE_CHECKING:
    from ui.chat_widget import ChatWidget
    from core.backend_client import BackendClient


class PromptStatusWorker(QThread):
    """Fetches the live standing prompt + experiment state off the GUI thread."""

    ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            result = self._client.prompt_experiment_status()
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))
            return
        if result.get("type") == "prompt_experiment_status":
            self.ready.emit(result)
        else:
            self.error_occurred.emit(result.get("message", "Failed to fetch the prompt"))


class RewriteStatusWorker(QThread):
    """Fetches the autonomous rewrite's gate + attempt log off the GUI thread."""

    ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            result = self._client.prompt_rewrite_status()
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))
            return
        if result.get("type") == "prompt_rewrite_status" and not result.get("error"):
            self.ready.emit(result)
        else:
            self.error_occurred.emit(result.get("error") or result.get("message")
                                     or "Failed to fetch the rewrite status")


class SetPromptWorker(QThread):
    """Applies the operator's edited prompt as the live experiment, off the GUI thread."""

    done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", prompt: str, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._prompt = prompt

    def run(self) -> None:
        try:
            self.done.emit(self._client.set_prompt(self._prompt))
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))


class RevertPromptWorker(QThread):
    """Ends the active experiment and restores the base prompt, off the GUI thread."""

    done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            self.done.emit(self._client.revert_prompt())
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))


class PromptExperimentWorker(QThread):
    """Drives one prompt-experiment generation off the GUI thread.

    The "Prompt experiment" button asks Ava to rewrite her standing prompt freely; on
    success the server activates it (temporarily, until reverted) and this worker's
    ``done`` reports it. Her process streams into the log as she writes it.
    """

    chunk = pyqtSignal(str)           # streamed reasoning/process delta
    done = pyqtSignal(dict)           # terminal prompt_experiment_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", temperature: float, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._temperature = temperature

    def run(self) -> None:
        try:
            for kind, msg in self._client.prompt_experiment(temperature=self._temperature):
                if kind == "prompt_experiment_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "prompt_experiment_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Prompt experiment failed"))
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))


class PromptWidget(QWidget):
    """Prompt tab — read, hand-edit, experiment on, and revert the standing prompt."""

    _DIVIDER = "─" * 60

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self.text_font = QFont("Courier")

        # Server truth, re-read on every refresh (never remembered across a restart):
        # whether the live prompt is a temporary experiment, and what a revert restores.
        self._active: bool = False
        self._base_prompt: str = ""
        # The text as last loaded from / written to the server — the baseline the dirty
        # check compares against, so "unsaved edits" means exactly that.
        self._baseline: str = ""

        self._status_worker: Optional[PromptStatusWorker] = None
        self._set_worker: Optional[SetPromptWorker] = None
        self._revert_worker: Optional[RevertPromptWorker] = None
        self._experiment_worker: Optional[PromptExperimentWorker] = None
        self._experiment_stream_open: bool = False

        self._build_ui()

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        header = QHBoxLayout()
        self._lbl_status = QLabel("Standing prompt — press Refresh to load.")
        header.addWidget(self._lbl_status, 1)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip(
            "Re-read the live standing prompt from the server. If you have unsaved "
            "edits in the box below you are asked before they are replaced."
        )
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setFont(self.text_font)
        self.txt_prompt.setPlaceholderText(
            "The live standing prompt loads here — the experimental one if an "
            "experiment is active, otherwise the base prompt."
        )
        self.txt_prompt.textChanged.connect(self._on_text_changed)
        splitter.addWidget(self.txt_prompt)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(self.text_font)
        self.txt_log.setPlaceholderText(
            "Ava's process during a prompt experiment, and the outcome of each change, "
            "appear here."
        )
        splitter.addWidget(self.txt_log)

        # The autonomous rewrite (core/prompt_rewrite.py): why the `rewrite_prompt` action
        # is or is not on Ava's menu right now (the currency gate, condition by condition),
        # the pattern budget, and the last attempts in full — timestamp, outcome, what she
        # chose, her WHY, and every candidate she was choosing among. Read-only; refreshed
        # with the tab, independent of unsaved edits in the prompt box (it never touches it).
        self.txt_rewrite = QPlainTextEdit()
        self.txt_rewrite.setReadOnly(True)
        self.txt_rewrite.setFont(self.text_font)
        self.txt_rewrite.setPlaceholderText(
            "Autonomous rewrite: the gate's verdict, the pattern budget and the last "
            "attempts (variants + her choice) load here on Refresh."
        )
        splitter.addWidget(self.txt_rewrite)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        layout.addWidget(splitter, 1)

        btn_row = QHBoxLayout()

        self.btn_update = QPushButton("Update prompt")
        self.btn_update.setToolTip(
            "Apply the text above as Ava's live standing prompt, right now. It goes into "
            "the same temporary tier a prompt experiment uses — live for her next "
            "conversations, across chats and restarts, until you press Revert prompt. "
            "chat_prompt.txt is never overwritten, so this is always undoable."
        )
        self.btn_update.clicked.connect(self._on_update_prompt)
        btn_row.addWidget(self.btn_update)

        self.btn_experiment = QPushButton("Prompt experiment")
        self.btn_experiment.setToolTip(
            "A free experiment: ask Ava to rewrite her standing chat prompt however she "
            "likes, knowing it is temporary. Her process streams into the log below. On "
            "success the new prompt becomes LIVE for her next conversations — across "
            "chats and restarts — until you press Revert prompt, then it reverts to the "
            "original. chat_prompt.txt is never overwritten. Disabled while an "
            "experiment is active."
        )
        self.btn_experiment.clicked.connect(self._on_prompt_experiment)
        btn_row.addWidget(self.btn_experiment)

        self.btn_revert = QPushButton("Revert prompt")
        self.btn_revert.setEnabled(False)
        self.btn_revert.setToolTip(
            "End the active prompt experiment — Ava's or your own — and restore her "
            "original standing prompt. Enabled only while one is active. The experience "
            "is kept as a minimal episode; only the temporary prompt is dropped."
        )
        self.btn_revert.clicked.connect(self._on_revert_prompt)
        btn_row.addWidget(self.btn_revert)

        btn_row.addStretch()

        lbl_temp = QLabel("Temp:")
        lbl_temp.setToolTip("Sampling temperature for the prompt-experiment generation.")
        btn_row.addWidget(lbl_temp)
        self.spn_temperature = QDoubleSpinBox()
        self.spn_temperature.setRange(0.0, 2.0)
        self.spn_temperature.setSingleStep(0.05)
        self.spn_temperature.setDecimals(2)
        self.spn_temperature.setValue(0.9)
        self.spn_temperature.setMaximumWidth(80)
        self.spn_temperature.setToolTip(
            "Sampling temperature for the prompt-experiment generation. Affects only "
            "Ava's rewrite — not the prompt you type."
        )
        btn_row.addWidget(self.spn_temperature)

        layout.addLayout(btn_row)

        self._update_buttons()

    # ---------------------------------------------------------------- #
    # State                                                             #
    # ---------------------------------------------------------------- #

    @property
    def temperature(self) -> float:
        return float(self.spn_temperature.value())

    def _is_dirty(self) -> bool:
        return self.txt_prompt.toPlainText() != self._baseline

    def _busy(self) -> bool:
        """Is one of this tab's own RPCs in flight?"""
        for w in (self._status_worker, self._set_worker, self._revert_worker,
                  self._experiment_worker):
            if w is not None and w.isRunning():
                return True
        return False

    def _update_buttons(self) -> None:
        connected = self._chat_widget._client.is_connected()
        idle = connected and not self._busy()
        self.btn_refresh.setEnabled(idle)
        self.btn_update.setEnabled(idle)
        # Experiment / revert are mutually exclusive on the active state: you can only
        # start one when none is live, and only revert when one is.
        self.btn_experiment.setEnabled(idle and not self._active)
        self.btn_revert.setEnabled(idle and self._active)

    def _on_text_changed(self) -> None:
        self._render_status()

    def _render_status(self) -> None:
        if self._active:
            set_by = getattr(self, "_set_by", "") or ""
            if set_by == "ava":
                ev = getattr(self, "_event", "") or ""
                state = (f"Prompt set by AVA HERSELF (rewrite event {ev}) — LIVE; "
                         f"Revert restores the base")
            elif set_by == "operator":
                state = "Hand-written prompt is LIVE (temporary — Revert restores the base)"
            else:
                state = "Experimental prompt is LIVE (temporary — Revert restores the base)"
        else:
            state = "Base standing prompt (no experiment active)"
        chars = len(self.txt_prompt.toPlainText())
        dirty = " · unsaved edits — press \"Update prompt\" to apply" if self._is_dirty() else ""
        self._lbl_status.setText(f"{state} · {chars:,} chars{dirty}")

    def _append_log(self, text: str) -> None:
        if not text:
            return
        cursor = self.txt_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.txt_log.setTextCursor(cursor)
        self.txt_log.ensureCursorVisible()

    def _append_header(self, title: str) -> None:
        prefix = "\n" if self.txt_log.toPlainText() else ""
        self._append_log(f"{prefix}{self._DIVIDER}\n{title}\n{self._DIVIDER}\n")

    # ---------------------------------------------------------------- #
    # Refresh                                                           #
    # ---------------------------------------------------------------- #

    def on_connected(self) -> None:
        """Load the live prompt when the socket comes up (wired in MainWindow)."""
        self.refresh()

    def on_disconnected(self) -> None:
        self._update_buttons()

    def _on_refresh_clicked(self) -> None:
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        """Re-read the live standing prompt from the server.

        Unsaved edits are protected: a background refresh (tab opened, socket connected)
        leaves the box alone, and an explicit Refresh press asks before replacing them.
        """
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect from the Chat tab.")
            self._update_buttons()
            return
        if self._status_worker is not None and self._status_worker.isRunning():
            return
        # The rewrite panel never touches the prompt box, so it refreshes whatever the
        # edit state — the dirty guard below protects the box, not this.
        self._refresh_rewrite_status(client)
        if self._is_dirty():
            if not force:
                self._render_status()
                return
            confirm = QMessageBox.question(
                self, "Discard edits?",
                "The prompt box has unsaved edits. Reload the live prompt from the "
                "server and discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        worker = PromptStatusWorker(client, parent=self)
        worker.ready.connect(self._on_status_ready)
        worker.error_occurred.connect(self._on_status_error)
        worker.finished.connect(worker.deleteLater)
        self._status_worker = worker
        self._update_buttons()
        worker.start()

    # ---------------------------------------------------------------- #
    # Autonomous rewrite panel                                          #
    # ---------------------------------------------------------------- #

    def _refresh_rewrite_status(self, client) -> None:
        worker = getattr(self, "_rewrite_worker", None)
        if worker is not None and worker.isRunning():
            return
        worker = RewriteStatusWorker(client, parent=self)
        worker.ready.connect(self._on_rewrite_status_ready)
        worker.error_occurred.connect(
            lambda err: self.txt_rewrite.setPlainText(f"Rewrite status unavailable: {err}"))
        worker.finished.connect(worker.deleteLater)
        self._rewrite_worker = worker
        worker.start()

    @staticmethod
    def _fmt_ts(ts: str) -> str:
        return (ts or "").replace("T", " ")[:19] or "?"

    def _on_rewrite_status_ready(self, r: dict) -> None:
        self._rewrite_worker = None
        lines: list[str] = []
        gate = r.get("gate") or {}
        st = r.get("settings") or {}
        budget = r.get("budget") or {}
        reason_text = {
            "budget": "on her menu — a mature pattern funds it",
            "continue": "on her menu — an interrupted event resumes",
            "disabled": "prompt_rewrite.enabled is false",
            "no_model": "no model loaded",
            "no_mature_pattern": "no mature pattern (spent ones excluded)",
            "gap": f"last attempt too recent — {gate.get('gap_hours_left', 0)} h of the "
                   f"{st.get('min_gap_hours', 24)} h gap left",
            "no_new_deltas": "nothing reflected since the last attempt (no new delta)",
        }.get(gate.get("reason", ""), gate.get("reason", "?"))
        verdict = "OFFERED" if gate.get("offered") else "NOT OFFERED"
        lines.append(f"AUTONOMOUS REWRITE — {verdict}: {reason_text}")
        lines.append(
            f"  enabled={'yes' if r.get('enabled') else 'NO'} · "
            f"model={'loaded' if r.get('model_loaded') else 'NOT loaded'} · "
            f"event running={'yes' if r.get('active') else 'no'} · "
            f"samples={st.get('samples')} @ {st.get('temperatures')} · "
            f"final={st.get('final_temperature')} · gap={st.get('min_gap_hours')} h · "
            f"order_check={'on' if st.get('order_check') else 'off'}")
        lines.append(
            f"  budget: {budget.get('n_patterns', 0)} pattern(s) in patterns.json "
            f"({budget.get('n_mature_in_file', 0)} mature there, "
            f"{len(gate.get('mature') or [])} still unspent) · "
            f"{r.get('unconsumed_deltas', 0)} unspent delta(s) · "
            f"{gate.get('deltas_since_last_attempt', 0)} new since the last attempt · "
            f"folded {self._fmt_ts(budget.get('built_at', ''))} · "
            f"last attempt {self._fmt_ts(budget.get('last_attempt_ts', '')) if budget.get('last_attempt_ts') else 'never'}")
        if gate.get("in_flight"):
            ev = gate["in_flight"]
            lines.append(
                f"  IN FLIGHT: event {ev.get('event_id')} started "
                f"{self._fmt_ts(ev.get('started_ts', ''))} — {ev.get('samples_done', 0)} draft(s) "
                f"done, consensus={'yes' if ev.get('consensus') else 'no'}, "
                f"final={'yes' if ev.get('final') else 'no'}")
        pats = budget.get("patterns") or []
        if pats:
            lines.append("  patterns (weight = tension-weighted recurrence over distinct chats):")
            for p_ in pats[:12]:
                tag = "MATURE" if p_.get("mature") else "forming"
                lines.append(f"    - [{p_.get('scope') or '?'}] {tag} w={p_.get('weighted_recurrence')} "
                             f"chats={p_.get('recurrences')}: {(p_.get('delta') or '')[:200]}")
        if gate.get("offer_line"):
            lines.append("  the line she would see: " + gate["offer_line"])

        attempts = r.get("attempts") or []
        lines.append("")
        if not attempts:
            lines.append("ATTEMPTS: none yet.")
        else:
            lines.append(f"ATTEMPTS: {r.get('attempts_total', len(attempts))} total; newest first.")
        for i, a in enumerate(attempts):
            outcome = str(a.get("outcome") or "?").upper()
            head = (f"— {self._fmt_ts(a.get('ts', ''))} · event {a.get('event') or '?'} · "
                    f"{outcome} · chosen: {a.get('chosen') or '—'}")
            if a.get("choice_unparsed"):
                head += " · (choice unparsed → kept the current prompt)"
            if a.get("order_check"):
                oc = a["order_check"]
                head += (f" · order check {'agreed' if oc.get('agree') else 'DISAGREED'}"
                         f" (reversed pick: {oc.get('reversed_pick')})")
            lines.append(head)
            if a.get("why"):
                lines.append(f"  WHY: {a['why']}")
            if a.get("patterns_weighed"):
                lines.append(f"  patterns weighed: {', '.join(map(str, a['patterns_weighed']))}"
                             f" · consumed keys: {len(a.get('consumed_keys') or [])}")
            cands = a.get("candidates") or {}
            if i == 0 and cands:
                lines.append(f"  candidates ({len(cands)}):")
                order = ["incumbent"] + sorted(k for k in cands if k.startswith("sample_"))                     + (["final"] if "final" in cands else [])
                for label in order:
                    if label not in cands:
                        continue
                    mark = "  ← CHOSEN" if label == a.get("chosen") else ""
                    lines.append(f"  ===== {label}{mark} =====")
                    lines.append((cands[label] or "").strip() or "(empty)")
                    lines.append("")
            elif cands:
                lines.append(f"  candidates: {len(cands)} (texts shown for the newest attempt only)")
        self.txt_rewrite.setPlainText("\n".join(lines))

    def _on_status_ready(self, result: dict) -> None:
        self._status_worker = None
        self._active = bool(result.get("active"))
        self._base_prompt = result.get("base_prompt", "") or ""
        # Provenance (2026-09-19): `set_by` ava/operator/experiment + the rewrite event id,
        # so an operator can see that Ava changed her own prompt rather than assuming the
        # experiment button was pressed. Revert is the veto either way.
        self._set_by = result.get("set_by", "") or ""
        self._event = result.get("event", "") or ""
        prompt = result.get("prompt", "") or ""
        self.txt_prompt.blockSignals(True)
        self.txt_prompt.setPlainText(prompt)
        self.txt_prompt.blockSignals(False)
        self._baseline = prompt
        self._render_status()
        self._update_buttons()

    def _on_status_error(self, err: str) -> None:
        self._status_worker = None
        self._lbl_status.setText(f"Failed to load the prompt: {err}")
        self._update_buttons()

    # ---------------------------------------------------------------- #
    # "Update prompt" — apply the operator's own text                   #
    # ---------------------------------------------------------------- #

    def _on_update_prompt(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._busy():
            return
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(
                self, "Empty prompt",
                "The prompt box is empty. Type a prompt (or press Refresh to reload the "
                "live one) before applying.",
            )
            return

        body = ("Apply this text as Ava's live standing prompt? It replaces the active "
                "experimental prompt."
                if self._active else
                "Apply this text as Ava's live standing prompt? It becomes a temporary "
                "experiment — chat_prompt.txt is not overwritten.")
        confirm = QMessageBox.question(
            self, "Update prompt",
            body + "\n\nPress \"Revert prompt\" at any time to restore the original.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._append_header("Update prompt — applying your edited standing prompt")
        self._lbl_status.setText("Applying prompt…")
        worker = SetPromptWorker(client, prompt, parent=self)
        worker.done.connect(self._on_set_done)
        worker.error_occurred.connect(self._on_set_error)
        worker.finished.connect(worker.deleteLater)
        self._set_worker = worker
        self._update_buttons()
        worker.start()

    def _on_set_done(self, result: dict) -> None:
        self._set_worker = None
        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or skipped
            self._append_log(f"[prompt] Not applied: {reason}\n")
            self._lbl_status.setText(f"Prompt not applied: {reason}")
        elif result.get("error"):
            self._append_log(f"[prompt] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Update failed.")
        elif result.get("activated"):
            self._active = True
            self._baseline = self.txt_prompt.toPlainText()
            chars = result.get("prompt_chars", 0)
            replaced = " (replacing the previous experiment)" if result.get("replaced") else ""
            self._append_log(
                f"[prompt] ✓ Your prompt is now LIVE ({chars:,} chars){replaced}. It "
                "stays active — across chats and restarts — until you press \"Revert "
                "prompt\".\n")
            self._render_status()
        else:
            self._append_log(f"[prompt] Update returned: {result}\n")
        self._update_buttons()

    def _on_set_error(self, err: str) -> None:
        self._set_worker = None
        self._append_log(f"[prompt] ✗ Update failed: {err}\n")
        self._lbl_status.setText(f"Update failed: {err}")
        self._update_buttons()

    # ---------------------------------------------------------------- #
    # "Prompt experiment" — Ava rewrites her own standing prompt        #
    # ---------------------------------------------------------------- #

    def _on_prompt_experiment(self) -> None:
        """Ask Ava to rewrite her standing prompt as a free (temporary) experiment."""
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._busy():
            return
        if self._active:
            QMessageBox.information(
                self, "Experiment active",
                "A prompt experiment is already active. Press \"Revert prompt\" to end "
                "it before starting another.",
            )
            return
        if self._is_dirty():
            confirm = QMessageBox.question(
                self, "Discard edits?",
                "The prompt box has unsaved edits. Ava rewrites the prompt that is "
                "currently LIVE, not your edits, and her result replaces the box "
                "contents. Continue and discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        self._experiment_stream_open = False
        self._append_header("Prompt experiment — Ava rewrites her standing prompt")
        self._append_log(
            "[prompt experiment] Handing Ava her current standing prompt to rewrite "
            "freely. On success the new prompt goes live for her next conversations — "
            "until you Revert. chat_prompt.txt is never overwritten.\n\n")
        self._lbl_status.setText("Running prompt experiment…")

        worker = PromptExperimentWorker(client, temperature=self.temperature, parent=self)
        worker.chunk.connect(self._on_experiment_chunk)
        worker.done.connect(self._on_experiment_done)
        worker.error_occurred.connect(self._on_experiment_error)
        worker.finished.connect(worker.deleteLater)
        self._experiment_worker = worker
        self._update_buttons()
        worker.start()

    def _experiment_is_current(self) -> bool:
        return self.sender() is None or self.sender() is self._experiment_worker

    def _on_experiment_chunk(self, delta: str) -> None:
        if not self._experiment_is_current():
            return
        if delta:
            self._append_log(delta)
            self._experiment_stream_open = True

    def _on_experiment_done(self, result: dict) -> None:
        if self.sender() is not None and self.sender() is not self._experiment_worker:
            return
        self._experiment_worker = None
        if self._experiment_stream_open:
            self._append_log("\n")
            self._experiment_stream_open = False

        skipped = result.get("skipped")
        if skipped:
            reason = result.get("message") or skipped
            self._append_log(f"\n[prompt experiment] Skipped: {reason}\n")
            self._lbl_status.setText("Prompt experiment skipped.")
        elif result.get("error"):
            self._append_log(f"\n[prompt experiment] ✗ Error: {result['error']}\n")
            self._lbl_status.setText("Prompt experiment failed.")
        elif result.get("activated"):
            self._active = True
            chars = result.get("prompt_chars", 0)
            self._append_log(
                f"\n[prompt experiment] ✓ Experimental prompt is now LIVE ({chars:,} "
                "chars). It stays active — across chats and restarts — until you press "
                "\"Revert prompt\".\n")
            self._lbl_status.setText("Prompt experiment active — Revert to end it.")
            # Her new prompt IS the live one now, so the editbox must show it (and
            # become the baseline for any further hand-editing).
            self._load_into_box(result.get("prompt"))
        else:
            self._append_log(
                "\n[prompt experiment] Finished without activating a prompt.\n")
            self._lbl_status.setText("Prompt experiment finished.")
        self._update_buttons()

    def _on_experiment_error(self, err: str) -> None:
        if self.sender() is not None and self.sender() is not self._experiment_worker:
            return
        self._experiment_worker = None
        if self._experiment_stream_open:
            self._append_log("\n")
            self._experiment_stream_open = False
        self._append_log(f"[prompt experiment] ✗ Failed: {err}\n")
        self._lbl_status.setText(f"Prompt experiment failed: {err}")
        self._update_buttons()

    def _load_into_box(self, prompt: Optional[str]) -> None:
        """Show a server-side prompt change in the editbox and re-baseline it.

        Takes the text from the terminal message when it carried one; otherwise re-reads
        it from the server (the box is not dirty at this point — it was just replaced by
        a change the operator asked for — so the plain refresh path is safe)."""
        text = (prompt or "").strip()
        if not text:
            self.refresh(force=True)
            return
        self.txt_prompt.blockSignals(True)
        self.txt_prompt.setPlainText(text)
        self.txt_prompt.blockSignals(False)
        self._baseline = text
        self._render_status()

    # ---------------------------------------------------------------- #
    # "Revert prompt" — restore the base standing prompt                #
    # ---------------------------------------------------------------- #

    def _on_revert_prompt(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(
                self, "Not connected",
                "Please connect to the server in the Chat tab first.",
            )
            return
        if self._busy():
            return
        confirm = QMessageBox.question(
            self, "Revert prompt experiment",
            "End the active prompt experiment and restore Ava's original standing "
            "prompt? The experimental prompt is dropped (the experience is kept as a "
            "minimal episode).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._lbl_status.setText("Reverting…")
        worker = RevertPromptWorker(client, parent=self)
        worker.done.connect(self._on_revert_done)
        worker.error_occurred.connect(self._on_revert_error)
        worker.finished.connect(worker.deleteLater)
        self._revert_worker = worker
        self._update_buttons()
        worker.start()

    def _on_revert_done(self, result: dict) -> None:
        self._revert_worker = None
        if result.get("skipped") == "none_active":
            self._active = False
            self._append_log("[prompt] No experiment was active.\n")
            self._lbl_status.setText("No prompt experiment was active.")
        elif result.get("reverted"):
            self._active = False
            self._append_log("[prompt] ✓ Reverted to the original standing prompt.\n")
            self._lbl_status.setText("Prompt experiment reverted.")
        else:
            self._append_log(f"[prompt] Revert returned: {result}\n")
        # The live prompt just changed under the box either way — reload it. The
        # experimental text the operator may have been editing is gone by design.
        self._baseline = self.txt_prompt.toPlainText()
        self.refresh()
        self._update_buttons()

    def _on_revert_error(self, err: str) -> None:
        self._revert_worker = None
        self._append_log(f"[prompt] ✗ Revert failed: {err}\n")
        self._lbl_status.setText(f"Revert failed: {err}")
        self._update_buttons()

    # ---------------------------------------------------------------- #
    # Fonts                                                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self.txt_prompt.setFont(font)
        self.txt_log.setFont(font)
        self.txt_rewrite.setFont(font)
