"""Shared editor for one kind of Ava's live reflection memory.

The Persona and Facts tabs curate two kinds of the same folded ``rag_memory.jsonl``
view, so the machinery is one widget: fetch the live rows of a kind, delete locally,
then upload the retained set explicitly. Deleting only edits the local list; Upload
appends server-side tombstones to live RAG and consolidation state. Runnable
snapshots, reflection archives, adapter weights, and digest artifacts are never
modified by this tool.

A subclass supplies what actually differs between the kinds — the artifact kind it
folds, the nouns the status line uses, how a row is rendered, and the upload RPC.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from core.backend_client import BackendClient
    from ui.chat_widget import ChatWidget


class MemoryListWidget(QListWidget):
    """Multi-select list with portable forward-delete and backspace handling."""

    delete_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MemoryFetchWorker(QThread):
    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, client: "BackendClient", noun: str):
        super().__init__()
        self._client = client
        self._noun = noun

    def run(self) -> None:
        result = self._client.get_rag_artifacts()
        if result.get("type") == "rag_artifacts":
            self.ready.emit(result)
        else:
            self.failed.emit(result.get("message", f"Failed to fetch {self._noun}."))


class MemoryUploadWorker(QThread):
    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, client: "BackendClient", widget: "MemoryEditorWidget",
                 baseline: list[str], retained: list[str]):
        super().__init__()
        self._client = client
        self._widget = widget
        self._baseline = baseline
        self._retained = retained

    def run(self) -> None:
        result = self._widget.upload_rpc(self._client, self._baseline, self._retained)
        if result.get("type") == self._widget.REPLY_TYPE:
            self.ready.emit(result)
        else:
            self.failed.emit(result.get("message", f"Failed to upload {self._widget.NOUN}."))


class MemoryEditorWidget(QWidget):
    """Edit a fetched live-memory set of one kind, then explicitly upload removals."""

    #: The ``rag_artifacts`` kind this editor folds ("persona" / "fact").
    KIND = ""
    #: Plural noun for status lines ("persona entries" reads as "persona").
    NOUN = ""
    #: Singular/plural row nouns, e.g. ("persona entry", "persona entries").
    ROW_NOUN = ("entry", "entries")
    #: Reply message type of the upload RPC.
    REPLY_TYPE = ""
    #: Sentence appended to the status line after a successful upload.
    UPLOAD_NOTE = ""
    #: Search-box placeholder.
    SEARCH_HINT = "Search…"

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._fetch_worker: Optional[MemoryFetchWorker] = None
        self._upload_worker: Optional[MemoryUploadWorker] = None
        self._baseline_keys: list[str] = []
        self._dirty = False
        self._build_ui()

    # -- subclass hooks ---------------------------------------------------- #

    def upload_rpc(self, client: "BackendClient", baseline: list[str],
                   retained: list[str]) -> dict:
        """Send the retained set to the server. Called off the GUI thread."""
        raise NotImplementedError

    def format_row(self, artifact: dict) -> str:
        """Render one live artifact as its list row."""
        raise NotImplementedError

    def eviction_warning(self, removed: int) -> str:
        """Body of the Upload confirmation dialog."""
        raise NotImplementedError

    # -- UI ---------------------------------------------------------------- #

    def _rows(self, n: int) -> str:
        return f"{n} {self.ROW_NOUN[0] if n == 1 else self.ROW_NOUN[1]}"

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.lbl_status = QLabel(
            f"Open this tab while connected to load {self.ROW_NOUN[1]}."
        )
        layout.addWidget(self.lbl_status)

        search = QHBoxLayout()
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText(self.SEARCH_HINT)
        self.edit_search.setClearButtonEnabled(True)
        self.edit_search.returnPressed.connect(self.search_next)
        self.btn_search = QPushButton("Search")
        self.btn_search.setToolTip("Jump to the next entry containing this text (Enter also works).")
        self.btn_search.clicked.connect(self.search_next)
        search.addWidget(self.edit_search, stretch=1)
        search.addWidget(self.btn_search)
        layout.addLayout(search)

        self.list_items = MemoryListWidget()
        self.list_items.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_items.setAlternatingRowColors(True)
        self.list_items.delete_requested.connect(self.delete_selected)
        self.list_items.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.list_items, stretch=1)

        # Kept on self so a subclass can add its own kind-specific controls to the
        # same row (the Persona tab's "Regen persona…" lives left of the stretch).
        self.buttons_row = buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip("Remove selected rows from this editor (Del or Backspace also works).")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_upload = QPushButton("Upload")
        self.btn_upload.setToolTip(
            f"Apply these removals to the connected server's live {self.NOUN} state."
        )
        self.btn_upload.clicked.connect(self.upload_changes)
        buttons.addWidget(self.btn_delete)
        buttons.addWidget(self.btn_upload)
        layout.addLayout(buttons)
        self._update_buttons()

    # -- fetch ------------------------------------------------------------- #

    def refresh_items(self) -> None:
        client = self._chat_widget._client
        if not client.is_connected():
            self.lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._dirty:
            return  # Do not silently discard an edit when tabs are switched.
        if self._fetch_worker is not None and self._fetch_worker.isRunning():
            return
        self.lbl_status.setText(f"Loading {self.NOUN} from connected server…")
        self._set_busy(True)
        self._fetch_worker = MemoryFetchWorker(client, self.NOUN)
        self._fetch_worker.ready.connect(self._on_fetched)
        self._fetch_worker.failed.connect(self._on_failed)
        self._fetch_worker.start()

    def _on_fetched(self, payload: dict) -> None:
        items = [
            a for a in payload.get("artifacts", [])
            if a.get("kind") == self.KIND and (a.get("content") or "").strip()
            and (a.get("key") or "").strip()
        ]
        self.list_items.clear()
        self._baseline_keys = []
        for artifact in items:
            key = artifact["key"].strip()
            item = QListWidgetItem(self.format_row(artifact))
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.list_items.addItem(item)
            self._baseline_keys.append(key)
        self._dirty = False
        self.lbl_status.setText(f"{self._rows(len(items))} loaded.")
        self._set_busy(False)

    def _on_failed(self, message: str) -> None:
        self.lbl_status.setText(f"{self.NOUN.capitalize()} operation failed: {message}")
        self._set_busy(False)

    # -- edit -------------------------------------------------------------- #

    def search_next(self) -> None:
        needle = self.edit_search.text().strip().lower()
        if not needle:
            return
        count = self.list_items.count()
        if count == 0:
            self.lbl_status.setText("No entries to search.")
            return
        start = self.list_items.currentRow()
        # Search forward from the row after the current one, wrapping around.
        for offset in range(1, count + 1):
            row = (start + offset) % count
            if needle in self.list_items.item(row).text().lower():
                self.list_items.setCurrentRow(row)
                self.list_items.scrollToItem(
                    self.list_items.item(row),
                    QAbstractItemView.ScrollHint.PositionAtCenter,
                )
                self.lbl_status.setText(f"Found at entry {row + 1} of {count}.")
                return
        self.lbl_status.setText(f"No entry contains “{self.edit_search.text().strip()}”.")

    def delete_selected(self) -> None:
        selected = self.list_items.selectedItems()
        if not selected:
            return
        for item in selected:
            self.list_items.takeItem(self.list_items.row(item))
        self._dirty = True
        self.lbl_status.setText(
            f"{len(selected)} row(s) removed locally — press Upload to apply on the server."
        )
        self._update_buttons()

    # -- upload ------------------------------------------------------------ #

    def upload_changes(self) -> None:
        if not self._dirty:
            return
        client = self._chat_widget._client
        if not client.is_connected():
            self.lbl_status.setText("Not connected — reconnect before uploading.")
            return
        retained = [
            self.list_items.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_items.count())
        ]
        removed = len(self._baseline_keys) - len(retained)
        answer = QMessageBox.question(
            self,
            f"Upload {self.NOUN} cleanup?",
            self.eviction_warning(removed),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.lbl_status.setText(f"Uploading {self.NOUN} cleanup to connected server…")
        self._set_busy(True)
        self._upload_worker = MemoryUploadWorker(client, self, self._baseline_keys, retained)
        self._upload_worker.ready.connect(self._on_uploaded)
        self._upload_worker.failed.connect(self._on_failed)
        self._upload_worker.start()

    def _on_uploaded(self, result: dict) -> None:
        if not result.get("ok"):
            message = result.get("message", f"{self.NOUN.capitalize()} upload was rejected.")
            if result.get("conflict"):
                QMessageBox.warning(self, f"{self.NOUN.capitalize()} changed", message)
                self._dirty = False
                self._set_busy(False)
                self.refresh_items()
                return
            self.lbl_status.setText(message)
            self._set_busy(False)
            return
        removed = int(result.get("removed", 0))
        self._baseline_keys = [
            self.list_items.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_items.count())
        ]
        self._dirty = False
        self.lbl_status.setText(f"Uploaded: {self._rows(removed)} evicted. {self.UPLOAD_NOTE}")
        self._set_busy(False)

    # -- misc -------------------------------------------------------------- #

    def _set_busy(self, busy: bool) -> None:
        self.list_items.setEnabled(not busy)
        self.btn_delete.setEnabled(not busy and bool(self.list_items.selectedItems()))
        self.btn_upload.setEnabled(not busy and self._dirty)

    def _update_buttons(self) -> None:
        busy = ((self._fetch_worker is not None and self._fetch_worker.isRunning()) or
                (self._upload_worker is not None and self._upload_worker.isRunning()))
        self.btn_delete.setEnabled(not busy and bool(self.list_items.selectedItems()))
        self.btn_upload.setEnabled(not busy and self._dirty)

    def update_fonts(self, font: QFont) -> None:
        self.list_items.setFont(font)
        self.edit_search.setFont(font)
