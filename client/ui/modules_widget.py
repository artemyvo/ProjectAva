"""Modules tab — run ONE pass against ONE input and look at what it produced.

Every generation on this box is the same shape (something is injected, something is read,
something is produced), but each pass hardcodes its own answer to all three inside the run
it belongs to — so a pass can neither be inspected nor tried on its own. This tab is the
workbench: pick a module, pick an input, edit the prompt, press **Simulate**, read the
output. It is a *dry run in the strict sense* — the server returns the produced value and
writes nothing (see ``core.modules``), so a facts simulation never touches the chat's
``.facts.json``.

**Scope so far.** Three modules — ``chat_facts`` (the per-chat extraction protocol),
``chat_summary`` (the consolidation gist) and ``til_facts`` (the per-article/news
protocol) — one item as input, and **nothing injected**: the injected-context list is
empty on purpose and the server refuses a selection rather than ignoring one. Chaining a
module's output into another's input is a later stage; what makes it possible is already
here (the value comes back instead of being written).

**The input list follows the module.** A module declares which KIND of thing it reads
(``ModuleSpec.source``) and this tab asks the server what that kind offers
(``list_module_inputs``) rather than knowing per module — so ``til_facts`` shows the
fetched articles and news digests where the chat modules show transcripts, and a fourth
lane would need no change here. The `chat` lane is the exception, and deliberately: it
keeps using the Chat tab's own ``SessionsWorker`` + ``ChatReviewWidget._label_for``, so
the chat list here cannot drift from the one beside it. The chosen input is remembered
per lane, since a chat filename is not a thing the til module could run on.

Rendering is module-agnostic: the server sends display ``lines`` produced by the module's
own ``finish``, so adding a module to the registry needs no change in this file.

Needs a connection **and** a loaded model (Simulate is a real generation).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit, QSplitter,
    QListWidget, QListWidgetItem, QLineEdit, QComboBox, QDoubleSpinBox, QMessageBox,
)
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from ui.chat_widget import SessionsWorker
# The chat list is literally the same list, so its label builder is reused rather than
# re-implemented — a second copy would be free to drift in exactly the way this tab exists
# to stop.
from ui.chat_review_widget import ChatReviewWidget

# Per-source display strings. The server names the source; this file names it for a
# reader. A source the server grows before this client is updated falls back to the
# generic wording rather than showing a raw key.
_SOURCE_LABELS = {
    "chat": "Chats (input)",
    "til": "Articles + news digests (input)",
}
_SOURCE_NOUNS = {"chat": "chat", "til": "text"}


class ModulesListWorker(QThread):
    """Fetches the module registry (name + current prompt text) off the GUI thread."""

    ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, client, parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            result = self._client.list_modules()
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the label
            self.error_occurred.emit(str(e))
            return
        if isinstance(result, dict) and result.get("type") == "modules_list":
            self.ready.emit(list(result.get("modules") or []))
        else:
            self.error_occurred.emit(
                (result or {}).get("message", "Failed to list modules"))


class ModuleInputsWorker(QThread):
    """Fetch the input list for one module *source* off the GUI thread.

    Only the non-chat sources come through here. The `chat` source deliberately keeps
    using the Chat tab's own `SessionsWorker` + `ChatReviewWidget._label_for`, so the
    chat list in this tab cannot drift from the one beside it; a second server-side
    listing of the same transcripts would be exactly that drift.
    """

    ready = pyqtSignal(str, list)          # source, inputs
    error_occurred = pyqtSignal(str)

    def __init__(self, client, source: str, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._source = source

    def run(self) -> None:
        try:
            result = self._client.list_module_inputs(self._source)
            self.ready.emit(self._source, list(result.get("inputs") or []))
        except Exception as e:
            self.error_occurred.emit(str(e))


class SimulateWorker(QThread):
    """Runs one module against one chat off the GUI thread, streaming the generation.

    The server writes nothing — the produced value comes back in the terminal message."""

    stage = pyqtSignal(dict)     # {stage, ...} phase markers
    chunk = pyqtSignal(str)      # generation delta (<think> included)
    done = pyqtSignal(dict)      # terminal module_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client, *, module: str, filename: str, prompt: str,
                 temperature: float, top_p: float, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._module = module
        self._filename = filename
        self._prompt = prompt
        self._temperature = temperature
        self._top_p = top_p

    def run(self) -> None:
        try:
            for kind, msg in self._client.run_module(
                    self._module, self._filename, prompt=self._prompt,
                    temperature=self._temperature, top_p=self._top_p):
                if kind == "module_stage":
                    self.stage.emit(msg)
                elif kind == "module_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "module_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Module run failed"))
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))


class ModulesWidget(QWidget):
    """Workbench for one pass: chat list on the left, prompt + result on the right."""

    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._sessions_worker: Optional[SessionsWorker] = None
        self._modules_worker: Optional[ModulesListWorker] = None
        self._inputs_worker: Optional[ModuleInputsWorker] = None
        self._sim_worker: Optional[SimulateWorker] = None
        self._sessions: list[dict] = []
        self._modules: list[dict] = []
        # Input lists per source, so switching module back and forth does not refetch.
        # `chat` is filled by SessionsWorker, everything else by ModuleInputsWorker.
        self._inputs: dict = {}
        # The selected input PER SOURCE: the lanes hold different kinds of thing, so one
        # shared slot would carry a chat filename into a til run and be refused.
        self._selected: dict = {}
        self._filename: str = ""
        self._prompt_dirty = False
        self._streaming = False
        self.text_font = QFont("Courier")
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._lbl = QLabel("Modules — run one pass against one input. Writes nothing.")
        header.addWidget(self._lbl, 1)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setToolTip(
            "Re-fetch the input list and the module registry (including each module's "
            "prompt as it stands on disk). An edited prompt box is left alone.")
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: the chat list — the module's INPUT.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        self._lbl_input = QLabel("Chats (input)")
        left_layout.addWidget(self._lbl_input)
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("filter…")
        self._txt_filter.setToolTip("Show only inputs whose list label contains this text.")
        self._txt_filter.textChanged.connect(self._populate_list)
        left_layout.addWidget(self._txt_filter)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._lst, 1)
        split.addWidget(left)

        # Right: module + injected context, then prompt over result.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.addWidget(QLabel("Module:"))
        self._cmb_module = QComboBox()
        self._cmb_module.setToolTip("Which pass to run. Its prompt loads below.")
        self._cmb_module.currentIndexChanged.connect(self._on_module_changed)
        top.addWidget(self._cmb_module)
        top.addSpacing(12)
        top.addWidget(QLabel("Temp:"))
        self._spn_temp = QDoubleSpinBox()
        self._spn_temp.setRange(0.0, 2.0)
        self._spn_temp.setSingleStep(0.05)
        self._spn_temp.setValue(0.7)
        self._spn_temp.setToolTip("Sampling temperature for the simulated pass.")
        top.addWidget(self._spn_temp)
        top.addStretch()
        right_layout.addLayout(top)

        # The injected-context list. Empty in v0 — and the emptiness is the statement:
        # the pass below sees the prompt and the transcript, nothing else.
        self._lbl_inject = QLabel()
        self._lbl_inject.setWordWrap(True)
        self._lbl_inject.setToolTip(
            "What is added to the prompt beyond the module's own prompt and the input. "
            "Nothing is injected yet — no persona, no clock, no retrieved memory.")
        right_layout.addWidget(self._lbl_inject)

        vsplit = QSplitter(Qt.Orientation.Vertical)

        prompt_box = QWidget()
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_row = QHBoxLayout()
        self._lbl_prompt = QLabel("Prompt")
        prompt_row.addWidget(self._lbl_prompt, 1)
        self._btn_reset = QPushButton("Reset to file")
        self._btn_reset.setToolTip("Discard edits and reload the module's prompt from disk.")
        self._btn_reset.clicked.connect(self._reset_prompt)
        prompt_row.addWidget(self._btn_reset)
        self._btn_run = QPushButton("Simulate")
        self._btn_run.setToolTip(
            "Run this module against the selected input with the prompt as edited here.\n"
            "Needs a loaded model. Writes nothing — the output is shown below only.")
        self._btn_run.clicked.connect(self._on_simulate)
        prompt_row.addWidget(self._btn_run)
        prompt_layout.addLayout(prompt_row)
        self._txt_prompt = QPlainTextEdit()
        self._txt_prompt.setFont(self.text_font)
        self._txt_prompt.setPlaceholderText(
            "The module's prompt. Edit freely — an edit applies to the next Simulate only "
            "and never touches the file on disk.")
        self._txt_prompt.textChanged.connect(self._on_prompt_edited)
        prompt_layout.addWidget(self._txt_prompt, 1)
        vsplit.addWidget(prompt_box)

        result_box = QWidget()
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(0, 0, 0, 0)
        self._lbl_result = QLabel("Result")
        result_layout.addWidget(self._lbl_result)
        self._txt_result = QPlainTextEdit()
        self._txt_result.setReadOnly(True)
        self._txt_result.setFont(self.text_font)
        self._txt_result.setPlaceholderText(
            "Select an input, then press Simulate.\n\n"
            "The generation streams here as it is written; when it finishes this is "
            "replaced by the parsed output — what the module would have produced.")
        result_layout.addWidget(self._txt_result, 1)
        vsplit.addWidget(result_box)
        vsplit.setSizes([300, 400])

        right_layout.addWidget(vsplit, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 760])
        layout.addWidget(split, 1)

        self._render_injectables([])
        self._update_buttons()

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._txt_prompt.setFont(font)
        self._txt_result.setFont(font)

    # ── refresh ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-fetch the chat list + module registry (tab open and Refresh button)."""
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl.setText("Modules — not connected (connect in the Chat tab).")
            return
        if self._sessions_worker is None or not self._sessions_worker.isRunning():
            worker = SessionsWorker(client)
            worker.sessions_ready.connect(self._on_sessions_ready)
            worker.error_occurred.connect(self._on_sessions_error)
            self._sessions_worker = worker
            worker.start()
        if self._modules_worker is None or not self._modules_worker.isRunning():
            mworker = ModulesListWorker(client)
            mworker.ready.connect(self._on_modules_ready)
            mworker.error_occurred.connect(self._on_modules_error)
            self._modules_worker = mworker
            mworker.start()
        self._fetch_inputs(self._current_source())

    def _current_source(self) -> str:
        """The input kind the selected module reads. Defaults to `chat` for an older
        server whose catalogue carries no `source` field."""
        return str((self._current_module() or {}).get("source") or "chat")

    def _fetch_inputs(self, source: str) -> None:
        """Fetch a non-chat source's input list. `chat` comes from SessionsWorker."""
        if source == "chat":
            return
        client = self._chat_widget._client
        if not client.is_connected():
            return
        if self._inputs_worker is not None and self._inputs_worker.isRunning():
            return
        worker = ModuleInputsWorker(client, source)
        worker.ready.connect(self._on_inputs_ready)
        worker.error_occurred.connect(self._on_inputs_error)
        self._inputs_worker = worker
        worker.start()

    def _on_inputs_ready(self, source: str, inputs: list) -> None:
        self._inputs[source] = list(inputs)
        if source == self._current_source():
            self._populate_list()
        self._status()

    def _on_inputs_error(self, error: str) -> None:
        self._lbl.setText(f"Modules — could not list inputs: {error}")

    def _on_sessions_ready(self, sessions: list) -> None:
        self._sessions = list(sessions)
        # Rendered through the Chat tab's own label builder, so this list and the one
        # beside it cannot disagree about what a chat is called.
        self._inputs["chat"] = [
            {"id": m.get("filename", ""), "label": ChatReviewWidget._label_for(m),
             "sub": ""}
            for m in reversed(sessions)          # most recent first, as in the Chat tab
        ]
        self._populate_list()
        self._status()

    def _on_sessions_error(self, error: str) -> None:
        self._lbl.setText(f"Modules — could not list chats: {error}")

    def _on_modules_ready(self, mods: list) -> None:
        self._modules = list(mods)
        previous = self._cmb_module.currentData()
        self._cmb_module.blockSignals(True)
        self._cmb_module.clear()
        for m in self._modules:
            self._cmb_module.addItem(m.get("label") or m.get("name", "?"), m.get("name"))
        if previous:
            idx = self._cmb_module.findData(previous)
            if idx >= 0:
                self._cmb_module.setCurrentIndex(idx)
        self._cmb_module.blockSignals(False)
        self._on_module_changed(keep_dirty=True)
        self._status()

    def _on_modules_error(self, error: str) -> None:
        self._lbl.setText(f"Modules — could not list modules: {error}")

    def _populate_list(self) -> None:
        """Render the current source's inputs. One renderer for every lane: the entries
        are already `{id, label, sub}` whether they came from the chat listing or the
        server, so a new input kind needs no branch here."""
        source = self._current_source()
        needle = self._txt_filter.text().strip().lower()
        previous = self._selected.get(source, "")
        self._lst.blockSignals(True)
        self._lst.clear()
        for entry in self._inputs.get(source) or []:
            label = entry.get("label") or entry.get("id", "")
            sub = entry.get("sub") or ""
            text = f"{label}\n    {sub}" if sub else label
            if needle and needle not in text.lower():
                continue
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, entry.get("id", ""))
            # A protocol this source has already recorded is the useful thing to simulate
            # against — the operator can compare the run to what is on disk.
            if entry.get("has_facts"):
                item.setToolTip("A protocol is already recorded for this text.")
            self._lst.addItem(item)
        self._lst.blockSignals(False)
        if previous:
            for i in range(self._lst.count()):
                if self._lst.item(i).data(Qt.ItemDataRole.UserRole) == previous:
                    self._lst.setCurrentRow(i)
                    break

    # ── module + prompt ────────────────────────────────────────────────────────
    def _current_module(self) -> dict:
        name = self._cmb_module.currentData()
        for m in self._modules:
            if m.get("name") == name:
                return m
        return {}

    def _on_module_changed(self, *_args, keep_dirty: bool = False) -> None:
        """Load the selected module's prompt + injectables into the panel.

        An edited prompt box survives a background refresh (``keep_dirty``) — losing a
        prompt an operator was midway through writing is the one thing this tab must not
        do — but switching module deliberately replaces it, since the text belongs to the
        module that was selected."""
        mod = self._current_module()
        self._lbl_prompt.setText(
            f"Prompt — {mod.get('prompt_file', '(none)')}" if mod else "Prompt")
        self._render_injectables(mod.get("injectables") or [])
        # The input list belongs to the module's SOURCE, so switching to a module that
        # reads something else swaps the list under it (and fetches it if this is the
        # first time that lane has been asked for).
        source = self._current_source()
        self._lbl_input.setText(_SOURCE_LABELS.get(source, "Inputs"))
        if source not in self._inputs:
            self._fetch_inputs(source)
        self._populate_list()
        self._filename = self._selected.get(source, "")
        if not (keep_dirty and self._prompt_dirty):
            self._txt_prompt.blockSignals(True)
            self._txt_prompt.setPlainText(mod.get("prompt", ""))
            self._txt_prompt.blockSignals(False)
            self._prompt_dirty = False
        self._update_buttons()

    def _render_injectables(self, injectables: list) -> None:
        if injectables:
            names = ", ".join(str(i) for i in injectables)
            self._lbl_inject.setText(f"Injected context: {names}")
        else:
            self._lbl_inject.setText(
                "Injected context: nothing — the pass sees this prompt and the input "
                "transcript, and nothing else (no persona, no clock, no retrieved memory).")

    def _on_prompt_edited(self) -> None:
        self._prompt_dirty = True
        self._status()

    def _reset_prompt(self) -> None:
        mod = self._current_module()
        if not mod:
            return
        if self._prompt_dirty:
            reply = QMessageBox.question(
                self, "Reset prompt",
                f"Discard your edits and reload {mod.get('prompt_file', 'the prompt')} "
                "from disk?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._txt_prompt.blockSignals(True)
        self._txt_prompt.setPlainText(mod.get("prompt", ""))
        self._txt_prompt.blockSignals(False)
        self._prompt_dirty = False
        self._status()

    def _on_selection_changed(self) -> None:
        items = self._lst.selectedItems()
        self._filename = items[0].data(Qt.ItemDataRole.UserRole) if items else ""
        # Remembered per lane, so switching module away and back restores the input that
        # was chosen there rather than clearing it or carrying a chat into a til run.
        self._selected[self._current_source()] = self._filename
        self._update_buttons()
        self._status()

    def _update_buttons(self) -> None:
        ready = bool(self._filename) and bool(self._current_module()) \
            and not self._streaming
        self._btn_run.setEnabled(ready)
        self._btn_reset.setEnabled(bool(self._current_module()) and not self._streaming)

    def _status(self) -> None:
        if self._streaming:
            return
        source = self._current_source()
        noun = _SOURCE_NOUNS.get(source, "input")
        bits = [f"{len(self._inputs.get(source) or [])} {noun}(s)"]
        bits.append(f"{noun}: {self._filename}" if self._filename
                    else f"no {noun} selected")
        if self._prompt_dirty:
            bits.append("prompt edited (not saved to disk)")
        self._lbl.setText("Modules — " + " · ".join(bits) + " · writes nothing.")

    # ── simulate ───────────────────────────────────────────────────────────────
    def _on_simulate(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl.setText("Modules — not connected (connect in the Chat tab).")
            return
        mod = self._current_module()
        if not mod or not self._filename:
            return
        if self._sim_worker is not None and self._sim_worker.isRunning():
            return

        self._streaming = True
        self._update_buttons()
        self._txt_result.setPlainText("")
        self._lbl.setText(
            f"Modules — running {mod.get('name')} on {self._filename}…")

        worker = SimulateWorker(
            client,
            module=str(mod.get("name")),
            filename=self._filename,
            prompt=self._txt_prompt.toPlainText(),
            temperature=float(self._spn_temp.value()),
            top_p=0.95,
        )
        worker.stage.connect(self._on_stage)
        worker.chunk.connect(self._on_chunk)
        worker.done.connect(self._on_done)
        worker.error_occurred.connect(self._on_error)
        self._sim_worker = worker
        worker.start()

    def _on_stage(self, info: dict) -> None:
        stage = str(info.get("stage") or "")
        if stage == "reading":
            self._lbl.setText(
                f"Modules — reading {info.get('session')} "
                f"({info.get('exchanges')} exchange(s))…")
        elif stage == "generating":
            # A chunked module (the summary pass over a long chat) is several generations
            # joined into one value; without the counter a multi-minute run looks stuck.
            parts = int(info.get("parts") or 1)
            where = f" — part {info.get('part')} of {parts}" if parts > 1 else ""
            self._lbl.setText(f"Modules — generating ({info.get('module')}){where}…")
        elif stage == "parsed":
            self._lbl.setText(f"Modules — parsed {info.get('count')} item(s).")

    def _at_bottom(self) -> bool:
        sb = self._txt_result.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _pin(self) -> None:
        sb = self._txt_result.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_chunk(self, text: str) -> None:
        # Insert through a DETACHED cursor: moving the visible one drags the viewport, so
        # an operator scrolled up to re-read the thought would be yanked back to the end
        # on every streamed token (same idiom as the Worklog tab's deliberation panel).
        at_bottom = self._at_bottom()
        cur = QTextCursor(self._txt_result.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text)
        if at_bottom:
            self._pin()

    def _on_error(self, message: str) -> None:
        self._streaming = False
        self._update_buttons()
        self._lbl.setText(f"Modules — failed: {message}")

    def _on_done(self, msg: dict) -> None:
        self._streaming = False
        self._update_buttons()
        if msg.get("skipped") or msg.get("error"):
            reason = msg.get("message") or msg.get("error") or msg.get("skipped")
            self._txt_result.setPlainText(f"— not run: {reason}")
            self._lbl.setText(f"Modules — not run ({msg.get('skipped') or 'error'}).")
            self._status()
            return
        self._txt_result.setPlainText(self._render_result(msg))
        self._lbl.setText(
            f"Modules — {msg.get('module')} on {msg.get('session')}: "
            f"{msg.get('count')} item(s). Nothing was written.")

    @staticmethod
    def _render_result(msg: dict) -> str:
        """The produced value, then the raw generation it was parsed out of.

        Module-agnostic on purpose: the server sends ``lines`` already rendered by the
        module's own ``finish``, so adding a module to the registry needs no change here.
        A facts-shaped renderer lived here until the second module arrived and had prose
        to show instead of records."""
        counts = msg.get("counts") or {}
        parts = int(msg.get("parts") or 1)
        count = msg.get("count")
        token_scope = "last part · " if parts > 1 else ""
        lines = [
            f"MODULE   {msg.get('module')}",
            # The unit is per lane — a transcript is measured in exchanges, an article in
            # characters — and the server says which, so this renderer stays lane-blind.
            # `exchanges` is the historical alias, read when talking to an older server.
            f"INPUT    {msg.get('session')}  "
            f"({msg.get('units', msg.get('exchanges'))} "
            f"{msg.get('unit_label') or 'exchange'}(s)"
            + (f", read in {parts} parts" if parts > 1 else "") + ")",
            f"OUTPUT   {count} item(s)"
            + (f"  [{', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}]"
               if counts else ""),
            f"TOKENS   {token_scope}input {msg.get('input_tokens')} · budget "
            f"{msg.get('max_new_tokens')} · window {msg.get('context_length')}",
            "WRITTEN  no — this was a simulation",
        ]
        # Why the generation ended, and it matters which: BOTH a stop-guard halt and a
        # real budget overrun come back as `truncated`, because neither ends on EOS — but
        # they need opposite responses, and reporting a guard halt as "raise the budget"
        # sends the operator to a knob that changes nothing (observed: a pass stopped at
        # ~1.2k of 12k tokens by the repetition guard, reported as out of budget).
        if msg.get("stopped_on_loop"):
            cap = msg.get("max_new_tokens")
            lines.append(
                "WARNING  a stop guard halted the model — it did NOT run out of budget"
                + (f" (cap was {cap})" if cap else "") + ". Either the repetition guard "
                "saw a repeated span or the degeneration guard saw diversity collapse. "
                "Read the tail of the raw generation below: if it stops at the end of a "
                "line prefix this module repeats, the guard is wrong for the module's "
                "output shape and belongs off (ModuleSpec.stop_on_repeat).")
        elif msg.get("cut_before_answer"):
            # An empty result here means the pass FAILED, not that the chat established
            # nothing — worth saying outright rather than leaving to be inferred.
            lines.append(
                "WARNING  the generation was cut off before it finished thinking, so "
                "there was no answer to parse. Raise the budget or shorten the input.")
        elif msg.get("truncated"):
            lines.append("WARNING  the generation hit its token budget (output may be "
                         "incomplete).")
        lines.append("")
        lines.append("── produced ──")
        produced = msg.get("lines") or []
        lines.extend(produced if produced else ["(nothing)"])
        lines.append("")
        lines.append("── raw generation ──")
        lines.append(msg.get("raw") or "")
        return "\n".join(lines)
