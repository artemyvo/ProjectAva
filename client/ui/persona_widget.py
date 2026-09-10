"""Persona tab — curate live persona evidence on the connected server.

Rows are fetched from the folded ``rag_memory.jsonl`` view. Deleting only edits the
local list; Upload appends server-side tombstones to live RAG and consolidation state.
Runnable snapshots, reflection archives, adapter weights, and digest artifacts are never
modified by the *editor* half of this tool.

The editor machinery is shared with the Facts tab (``ui.memory_editor``); this module
holds only what is persona-specific — including the one control here that deliberately
breaks the editor's never-touch-the-digest rule: **Regen persona…** (``regen_persona``
RPC), which re-derives the digest from the live evidence exactly as a Sleep run does
(cluster on the CLEAN base, synthesize on the ADAPTER) and then mints + activates the
persona version (``produce_persona(activate=True)``, the Sleep run's own ``persona``
stage call) so the refreshed self-portrait reaches the next live chat turn. It sits on
this tab because it is the natural next act after curating the evidence the digest folds:
evict the stale rows, then regenerate the portrait from what remains.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ui.memory_editor import MemoryEditorWidget

if TYPE_CHECKING:
    from core.backend_client import BackendClient


class RegenPersonaWorker(QThread):
    """Runs the persona regeneration off the GUI thread, streaming progress.

    Slow and two-phase like the Sleep tab's digest dry run (the server swaps the adapter
    out — two full model reloads — to cluster, then swaps back to synthesize), plus a
    third GPU-free tail: the persona snapshot (a full corpus copy) that activates the
    result. ``progress`` carries the ``regen_persona_stage`` events, ``chunk`` the
    synthesis deltas, ``done`` the terminal payload.
    """

    progress = pyqtSignal(dict)        # regen_persona_stage events
    chunk = pyqtSignal(str)            # synthesis delta
    done = pyqtSignal(dict)            # terminal regen_persona_done
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", parent=None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:
        try:
            for kind, msg in self._client.regen_persona():
                if kind == "regen_persona_stage":
                    self.progress.emit(msg)
                elif kind == "regen_persona_chunk":
                    self.chunk.emit(msg.get("text", ""))
                elif kind == "regen_persona_done":
                    self.done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(
                        msg.get("message", "Persona regeneration failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the dialog
            self.error_occurred.emit(str(e))


class RegenPersonaDialog(QDialog):
    """Non-modal progress window for one persona regeneration.

    Non-modal because the pass holds the server for many minutes (two model reloads +
    many clustering calls + the snapshot copy) and the rest of the client stays usable
    meanwhile. Closing the window does NOT stop the server-side pass — the run finishes
    and activates regardless — and the dialog says so, so a closed window is never
    mistaken for a cancel.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Regen persona")
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        self.lbl_status = QLabel("Starting…")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)
        note = QLabel("Closing this window does not stop the run — the server "
                      "finishes and activates the new persona regardless.")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        layout.addLayout(row)

    def append(self, text: str) -> None:
        """Append raw text at the end without yanking a scrolled-up reader."""
        cursor = self.log.textCursor()
        at_end = cursor.atEnd()
        cursor2 = QTextCursor(self.log.document())
        cursor2.movePosition(QTextCursor.MoveOperation.End)
        cursor2.insertText(text)
        if at_end:
            self.log.moveCursor(QTextCursor.MoveOperation.End)

    def append_line(self, text: str) -> None:
        self.append(text + "\n")


class PersonaWidget(MemoryEditorWidget):
    """Edit a fetched live-persona set, then explicitly upload removals."""

    KIND = "persona"
    NOUN = "persona"
    ROW_NOUN = ("live persona entry", "live persona entries")
    REPLY_TYPE = "persona_updated"
    SEARCH_HINT = "Search persona text…"
    UPLOAD_NOTE = "The next reflection will rebuild the persona digest from clean evidence."

    def __init__(self, chat_widget, parent=None):
        self._regen_worker: Optional[RegenPersonaWorker] = None
        self._regen_dialog: Optional[RegenPersonaDialog] = None
        super().__init__(chat_widget, parent)

    def upload_rpc(self, client: "BackendClient", baseline: list[str],
                   retained: list[str]) -> dict:
        return client.update_persona(baseline, retained)

    def format_row(self, artifact: dict) -> str:
        return f"[persona] {artifact['content'].strip()}"

    def eviction_warning(self, removed: int) -> str:
        return (
            f"Evict {removed} persona entr{'y' if removed == 1 else 'ies'} from the connected "
            "server's live recall and next-training evidence?\n\n"
            "No runnable snapshot or archived reflection bundle will be modified."
        )

    # -- Regen persona ------------------------------------------------------ #

    def _build_ui(self) -> None:
        super()._build_ui()
        # Left of the stretch, apart from Delete/Upload: those edit the fetched list,
        # this one talks to the model and writes a new active persona.
        self.btn_regen = QPushButton("Regen persona…")
        self.btn_regen.setToolTip(
            "Re-derive Ava's self-portrait from the live [persona] evidence (cluster on "
            "the clean base, synthesize on the adapter — the same pass a Sleep run "
            "performs) and ACTIVATE it as the current persona snapshot. The adapter is "
            "unchanged; the previous persona snapshot stays on disk as rollback."
        )
        self.btn_regen.clicked.connect(self._on_regen_persona)
        self.buttons_row.insertWidget(0, self.btn_regen)

    def _on_regen_persona(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            QMessageBox.warning(self, "Not connected",
                                "Please connect to the server in the Chat tab first.")
            return
        if self._regen_worker is not None and self._regen_worker.isRunning():
            if self._regen_dialog is not None:
                self._regen_dialog.show()
                self._regen_dialog.raise_()
            return
        if self._dirty:
            QMessageBox.warning(
                self, "Unsaved edits",
                "You have local persona removals that were not uploaded.\n\n"
                "Upload (or refresh away) your edits first, so the regenerated digest "
                "folds the evidence you actually mean it to.")
            return
        resp = QMessageBox.question(
            self,
            "Regenerate persona?",
            "This re-derives Ava's self-portrait from her live [persona] statements and "
            "ACTIVATES the result:\n\n"
            "  1. Statements are grouped into themes on the CLEAN BASE (adapter off).\n"
            "  2. The portrait is written on the ADAPTER — a new digest lands in the "
            "live persona store, exactly as a Sleep run writes it.\n"
            "  3. A new persona snapshot (data/persona/<run_id>) is produced and made "
            "the ACTIVE persona, so the refreshed self-portrait reaches the very next "
            "chat turn.\n\n"
            "The adapter and her memory are unchanged; the previous persona snapshot "
            "stays on disk as the rollback path.\n\n"
            "The model is swapped twice and the snapshot copies the corpus, so this "
            "takes a while. Run it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        dialog = RegenPersonaDialog(self)
        dialog.lbl_status.setText(
            "Sent — gathering live persona evidence, then swapping to the clean base "
            "to group it (per-block progress streams below)…")
        dialog.show()
        self._regen_dialog = dialog

        worker = RegenPersonaWorker(client, parent=self)
        worker.progress.connect(self._on_regen_progress)
        worker.chunk.connect(self._on_regen_chunk)
        worker.done.connect(self._on_regen_done)
        worker.error_occurred.connect(self._on_regen_error)
        worker.finished.connect(worker.deleteLater)
        self._regen_worker = worker
        self.btn_regen.setEnabled(False)
        self.lbl_status.setText("Regenerating persona — see the progress window…")
        worker.start()

    def _on_regen_progress(self, ev: dict) -> None:
        if self.sender() is not self._regen_worker:
            return
        d = self._regen_dialog
        if d is None:
            return
        stage = ev.get("stage")
        if stage == "gathered":
            d.append_line(f"[regen] {ev.get('items', 0)} live persona statement(s) to "
                          f"group, in blocks of {ev.get('block_size', 0)} "
                          f"(run {ev.get('run_id', '?')}).")
            d.lbl_status.setText(f"Grouping {ev.get('items', 0)} statements…")
        elif stage == "clean_base_enter":
            d.append_line("[regen] Swapping the adapter out (two full model reloads)…")
        elif stage == "map":
            i, n = ev.get("i", 0), ev.get("n", 0)
            flag = ("  ⚠ rejected (looked like one blob) — kept apart"
                    if ev.get("rejected") else "")
            d.append_line(f"  block {i}/{n}: {ev.get('items', 0)} statements → "
                          f"{ev.get('themes', 0)} theme(s){flag}")
            d.lbl_status.setText(f"Grouping block {i}/{n} on the clean base…")
        elif stage == "reduce_started":
            d.append_line(f"  merge round {ev.get('round', 0)}: re-grouping "
                          f"{ev.get('themes', 0)} theme(s) across blocks…")
            d.lbl_status.setText(f"Merge round {ev.get('round', 0)}…")
        elif stage == "reduce":
            d.append_line(f"  merge round {ev.get('round', 0)}: {ev.get('before', 0)} → "
                          f"{ev.get('after', 0)} theme(s) ({ev.get('merged', 0)} merged)")
        elif stage == "clustered":
            d.append_line(f"[regen] Clustered → {ev.get('themes', 0)} theme(s) in "
                          f"{ev.get('calls', 0)} model call(s); "
                          f"{ev.get('rejected_blocks', 0)} block(s) rejected by the "
                          f"blob guard.")
        elif stage == "polarity_split":
            d.append_line(f"  ⇄ polarity: {ev.get('opposed', 0)} opposing statement(s) "
                          f"split out of theme “{ev.get('theme', '')}”")
        elif stage == "polarity":
            split = ev.get("themes_split", 0)
            d.append_line(f"[regen] Polarity screen: {ev.get('themes_screened', 0)} "
                          f"merged theme(s) checked, {split} split "
                          f"({ev.get('members_flagged', 0)} opposing statement(s))."
                          + (f" {ev.get('calls_failed', 0)} check(s) failed."
                             if ev.get("calls_failed") else ""))
        elif stage == "synthesizing":
            d.append_line(f"[regen] Adapter back on — writing the portrait from "
                          f"{ev.get('themes', 0)} theme(s):\n")
            d.lbl_status.setText("Writing the portrait on the adapter…")
        elif stage == "digest_written":
            counts = ev.get("counts") or {}
            d.append_line(f"\n[regen] Digest written — {counts}")
        elif stage == "producing_persona":
            d.append_line(f"[regen] Producing + activating persona snapshot "
                          f"data/persona/{ev.get('run_id', '?')} (corpus copy, "
                          f"GPU-free)…")
            d.lbl_status.setText("Producing the persona snapshot…")

    def _on_regen_chunk(self, text: str) -> None:
        if self.sender() is not self._regen_worker:
            return
        if self._regen_dialog is not None:
            self._regen_dialog.append(text)

    def _on_regen_done(self, result: dict) -> None:
        if self.sender() is not self._regen_worker:
            return
        self._regen_worker = None
        self.btn_regen.setEnabled(True)
        d = self._regen_dialog

        skipped = result.get("skipped")
        if skipped:
            m = result.get("message") or skipped
            if d is not None:
                d.append_line(f"\n[regen] Skipped: {m}")
                d.lbl_status.setText(f"Skipped: {m}")
            self.lbl_status.setText(f"Persona regeneration skipped: {m}")
            return

        run_id = result.get("run_id", "?")
        if result.get("activated"):
            summary = (f"Persona {run_id} is ACTIVE — the refreshed self-portrait "
                       f"reaches the next chat turn.")
        else:
            summary = (f"Digest was written, but the persona snapshot FAILED — the "
                       f"previous persona stays active: "
                       f"{result.get('persona_error', 'unknown error')}")
        if d is not None:
            rendered = (result.get("rendered") or "").strip()
            if rendered:
                d.append_line("\n" + "=" * 60)
                d.append_line("PORTRAIT AS A CHAT TURN WILL SEE IT")
                d.append_line("=" * 60)
                d.append_line(rendered)
            d.append_line(f"\n[regen] {result.get('items', 0)} statement(s) → "
                          f"{result.get('themes', 0)} theme(s). {summary}")
            d.lbl_status.setText(summary)
        self.lbl_status.setText(summary)

    def _on_regen_error(self, err: str) -> None:
        if self.sender() is not self._regen_worker:
            return
        self._regen_worker = None
        self.btn_regen.setEnabled(True)
        if self._regen_dialog is not None:
            self._regen_dialog.append_line(f"\n[regen] ✗ Failed: {err}")
            self._regen_dialog.lbl_status.setText(f"Failed: {err}")
        self.lbl_status.setText(f"Persona regeneration failed: {err}")

    def update_fonts(self, font: QFont) -> None:
        super().update_fonts(font)
        if self._regen_dialog is not None:
            self._regen_dialog.log.setFont(font)
