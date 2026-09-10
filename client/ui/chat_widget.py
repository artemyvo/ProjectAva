"""Chat widget — UI-only: no logging, no RAG, no prompt management (all server-side)."""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QTextEdit,
    QSplitter,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
    QSpinBox,
    QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor, QTextCharFormat, QColor

import colorsys

from core.backend_client import BackendClient


_DEFAULT_SERVER_URL = "ws://localhost:8765"
_META_FEEDBACK_MAX_CHARS = 2000


# ------------------------------------------------------------------ #
# Worker threads                                                       #
# ------------------------------------------------------------------ #

class ConnectWorker(QThread):
    """Connects to the server and fetches its status in a background thread."""

    connect_done = pyqtSignal(dict)

    def __init__(self, client: BackendClient, url: str):
        super().__init__()
        self._client = client
        self._url = url

    def run(self) -> None:
        try:
            self._client.connect(self._url)
        except Exception as exc:
            self.connect_done.emit({"type": "error", "message": str(exc)})
            return
        result = self._client.request_status()
        self.connect_done.emit(result)


class ChatWorker(QThread):
    """Streams a single user message to the backend server.

    The server owns conversation history, RAG, and logging.
    """

    response_ready = pyqtSignal(str)
    response_chunk = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    generation_cancelled = pyqtSignal(str)
    debug_info = pyqtSignal(str)
    prompt_debug = pyqtSignal(list)
    facts_block = pyqtSignal(dict)

    def __init__(
        self,
        client: BackendClient,
        message: str,
        max_new_tokens_setting: str = "75%",
        context_length: int = 32768,
        debug_enabled: bool = False,
        user: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        rag_history: bool = True,
        rag_facts: bool = True,
        rag_persona: bool = True,
    ):
        super().__init__()
        self._client = client
        self._payload = message
        self.max_new_tokens_setting = max_new_tokens_setting
        self.context_length = context_length
        self.debug_enabled = debug_enabled
        self._user = user
        self.temperature = temperature
        self.top_p = top_p
        self.rag_history = rag_history
        self.rag_facts = rag_facts
        self.rag_persona = rag_persona

    def _emit_debug(self, message: str) -> None:
        if self.debug_enabled:
            self.debug_info.emit(message)

    def _should_early_stop_stream(self, text: str) -> bool:
        if not text:
            return False
        patterns = [
            r"(?is)\n\s*(user|system)\b[^\n]{0,32}\n\s*\n",
            r"(?is)\n\s*\n(?:what|why|how|who|when|where|is|are|do|does|did|can|could|would|should|will)"
            r"\b[^\n?]{0,220}\?\s*\n+\s*assistant\b",
        ]
        return any(re.search(p, text) for p in patterns)

    def run(self):
        gen = self._client.stream_generate(
            self._payload,
            user=self._user,
            max_new_tokens_setting=self.max_new_tokens_setting,
            context_length=self.context_length,
            temperature=self.temperature,
            top_p=self.top_p,
            debug=self.debug_enabled,
            rag_history=self.rag_history,
            rag_facts=self.rag_facts,
            rag_persona=self.rag_persona,
        )

        raw_tail = ""
        early_stop = False

        try:
            for t, text in gen:
                if t == "chunk":
                    raw_tail = (raw_tail + text)[-3000:]
                    if not early_stop and self._should_early_stop_stream(raw_tail):
                        early_stop = True
                        self._emit_debug("Early stop triggered.")
                        self._client.cancel()
                    if not early_stop:
                        self.response_chunk.emit(text)
                elif t == "prompt_debug":
                    # Structured, not a string: a list of labelled prompt segments the
                    # Chat tab renders colour-coded. Arrives once, before the first
                    # chunk, so it lands above the reply.
                    if self.debug_enabled and text:
                        self.prompt_debug.emit(list(text))
                elif t == "facts_block":
                    # NOT gated on Debug, unlike the prompt dump above: this is the one
                    # channel selected FOR this message, so it is what the reply rests on
                    # rather than a diagnostic. The server only sends it when the fetch
                    # ran, so a box with the channel off shows nothing.
                    if isinstance(text, dict):
                        self.facts_block.emit(text)
                elif t == "log":
                    self._emit_debug(text)
                elif t == "error":
                    self.error_occurred.emit(text)
                    return
                elif t == "done":
                    self.response_ready.emit(text or "(No response generated.)")
                    return
                elif t == "cancelled":
                    self.generation_cancelled.emit(self._payload)
                    return
        finally:
            gen.close()


class SessionsWorker(QThread):
    """Fetches the past-session list from the server in a background thread."""

    sessions_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: BackendClient):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.list_sessions()
        if result.get("type") == "sessions_list":
            self.sessions_ready.emit(result.get("sessions", []))
        else:
            self.error_occurred.emit(result.get("message", "Failed to list sessions"))


class AnchorMatchWorker(QThread):
    """Asks the server which stored exchange anchors the drafted message would match.

    Diagnostic only — the server retrieves and injects nothing. Matching is lexical
    server-side, so this needs no loaded model and returns fast enough to run while the
    operator is still typing.
    """

    matches_ready = pyqtSignal(dict)

    def __init__(self, client: BackendClient, text: str):
        super().__init__()
        self._client = client
        self._text = text

    def run(self) -> None:
        try:
            result = self._client.match_anchors(self._text)
        except Exception:
            return
        if result.get("type") == "anchor_matches":
            self.matches_ready.emit(result)


class LoadSessionWorker(QThread):
    """Loads a past session into the server's active conversation state."""

    session_loaded = pyqtSignal(dict, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: BackendClient, filename: str, in_place: bool = False):
        super().__init__()
        self._client = client
        self._filename = filename
        self._in_place = in_place

    def run(self) -> None:
        result = self._client.load_session(self._filename, in_place=self._in_place)
        if result.get("type") == "session_loaded":
            # For an in-place resume the active file is the same one; emit that.
            fn = result.get("filename") or self._filename
            self.session_loaded.emit(result.get("data", {}), fn)
        else:
            self.error_occurred.emit(result.get("message", "Failed to load session"))


class DeleteSessionWorker(QThread):
    """Deletes one or more past chat transcripts (+ sidecars) on the server."""

    finished_deleting = pyqtSignal(list, list)  # (deleted: [str], failed: [(name, msg)])

    def __init__(self, client: BackendClient, filenames: list):
        super().__init__()
        self._client = client
        self._filenames = list(filenames)

    def run(self) -> None:
        deleted: list = []
        failed: list = []
        for fn in self._filenames:
            try:
                result = self._client.delete_session(fn)
            except Exception as exc:  # connection dropped mid-batch
                failed.append((fn, str(exc)))
                continue
            if result.get("type") == "session_deleted":
                deleted.append(fn)
            else:
                failed.append((fn, result.get("message", "Failed to delete session")))
        self.finished_deleting.emit(deleted, failed)


class ResetReflectionWorker(QThread):
    """Deletes the sidecars of one or more past chats so they re-reflect from scratch."""

    finished_resetting = pyqtSignal(list, list, list)  # (reset, unreflected, failed)

    def __init__(self, client: BackendClient, filenames: list):
        super().__init__()
        self._client = client
        self._filenames = list(filenames)

    def run(self) -> None:
        reset: list = []        # had a sidecar; it was removed
        unreflected: list = []  # nothing to remove — already awaiting reflection
        failed: list = []
        for fn in self._filenames:
            try:
                result = self._client.reset_session_reflection(fn)
            except Exception as exc:  # connection dropped mid-batch
                failed.append((fn, str(exc)))
                continue
            if result.get("type") == "session_reflection_reset":
                (reset if result.get("had_sidecar") else unreflected).append(fn)
            else:
                failed.append((fn, result.get("message", "Failed to reset reflection")))
        self.finished_resetting.emit(reset, unreflected, failed)


class MetaFeedbackWorker(QThread):
    """Persists revision-only feedback on the latest completed Ava reply."""

    feedback_saved = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self, client: BackendClient, exchange_id: str, text: str, speaker: str
    ):
        super().__init__()
        self._client = client
        self._exchange_id = exchange_id
        self._text = text
        self._speaker = speaker

    def run(self) -> None:
        result = self._client.set_reflection_feedback(
            self._exchange_id, self._text, self._speaker
        )
        if result.get("type") == "reflection_feedback_saved":
            self.feedback_saved.emit(
                result.get("exchange_id", self._exchange_id),
                result.get("feedback") or {},
            )
        else:
            self.error_occurred.emit(result.get("message", "Failed to save Meta feedback"))


class RetryWorker(QThread):
    """Rolls the latest completed exchange off the active session (Chat "Retry").

    The completed-turn sibling of the Stop/discard flow: the server drops the last
    exchange from the active transcript + conversation and hands back the user prompt
    so the operator can resend it under adjusted sampling.
    """

    retry_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: BackendClient):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.retry_last_exchange()
        if result.get("type") == "retry_ready":
            self.retry_ready.emit(result)
        else:
            self.error_occurred.emit(result.get("message", "Failed to retry the last reply"))


class PreviewSessionWorker(QThread):
    """Read-only fetch of a past session for the chat log preview."""

    session_ready = pyqtSignal(dict, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: BackendClient, filename: str):
        super().__init__()
        self._client = client
        self._filename = filename

    def run(self) -> None:
        result = self._client.get_session(self._filename)
        if result.get("type") == "session_data":
            self.session_ready.emit(result.get("data", {}), self._filename)
        else:
            self.error_occurred.emit(result.get("message", "Failed to load session preview"))


class RestartWorker(QThread):
    """Sends a restart request to the watchdog HTTP API in a background thread."""

    restart_done = pyqtSignal(dict)

    def __init__(self, host: str, mgmt_port: int = 8766):
        super().__init__()
        self._host = host
        self._mgmt_port = mgmt_port

    def run(self) -> None:
        import urllib.request
        import json
        url = f"http://{self._host}:{self._mgmt_port}/restart"
        req = urllib.request.Request(url, data=b"", method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", "0")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read()
                data = json.loads(body)
                self.restart_done.emit({"type": "success", "data": data})
        except Exception as exc:
            self.restart_done.emit({"type": "error", "message": str(exc)})


class PullWorker(QThread):
    """Asks the watchdog to `git pull` and restart the inference server.

    The watchdog stops inference, fast-forwards its checkout, and relaunches —
    so the operator pushes code from this machine and lets the server pull it.
    """

    pull_done = pyqtSignal(dict)

    def __init__(self, host: str, mgmt_port: int = 8766):
        super().__init__()
        self._host = host
        self._mgmt_port = mgmt_port

    def run(self) -> None:
        import urllib.request
        import json
        url = f"http://{self._host}:{self._mgmt_port}/pull"
        req = urllib.request.Request(url, data=b"", method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", "0")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                self.pull_done.emit({"type": "success", "data": data})
        except Exception as exc:
            self.pull_done.emit({"type": "error", "message": str(exc)})


class MgmtStatusWorker(QThread):
    """Fetches the watchdog /status (hostname + data_dir) in a background thread.

    Used to decide whether "Fetch artifacts" makes sense: if the server reports
    the same hostname and the same resolved data dir as this client's local
    checkout, the two share one repository on storage and the fetch is a no-op.
    """

    status_done = pyqtSignal(dict)

    def __init__(self, host: str, mgmt_port: int = 8766):
        super().__init__()
        self._host = host
        self._mgmt_port = mgmt_port

    def run(self) -> None:
        import urllib.request
        import json
        url = f"http://{self._host}:{self._mgmt_port}/status"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
                self.status_done.emit({"type": "success", "data": data})
        except Exception as exc:
            self.status_done.emit({"type": "error", "message": str(exc)})


class FetchArtifactsWorker(QThread):
    """Downloads the server's full artifact set and reproduces it under the local
    ``server/`` dir, preserving the GPU box's directory structure.

    The inference HTTP sidecar's GET /artifacts returns a gzip tarball rooted at
    ``server/`` with three managed subtrees (``inference/data/``, ``reflections/``,
    ``logs/``). To make the fetch a faithful mirror — and not an accreting pile of
    leftovers from runs the server has since wiped — the worker extracts into a temp
    staging dir, then for each managed root **deletes the local copy and moves the
    freshly-fetched one into place**. The server guarantees a directory entry for
    every managed root (even when empty), so a root emptied server-side still clears
    locally.
    """

    # The bundle's top-level subtrees — must match mgmt_http._ARTIFACT_ROOTS.
    MANAGED_ROOTS = ("inference/data", "reflections", "logs")

    fetch_done = pyqtSignal(dict)

    def __init__(self, host: str, target_server_dir: Path, sidecar_port: int = 8767):
        super().__init__()
        self._host = host
        self._sidecar_port = sidecar_port
        self._target = target_server_dir  # local server/ dir

    def run(self) -> None:
        import urllib.request
        import io
        import tarfile
        import shutil
        import tempfile
        url = f"http://{self._host}:{self._sidecar_port}/artifacts"
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                blob = resp.read()
            self._target.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".artifacts-", dir=self._target))
            staging_root = staging.resolve()
            try:
                count = 0
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                    members = tar.getmembers()
                    for m in members:
                        # Guard against path traversal in tar entries.
                        dest = (staging / m.name).resolve()
                        if dest != staging_root and not str(dest).startswith(
                            str(staging_root) + os.sep
                        ):
                            raise ValueError(f"unsafe path in archive: {m.name}")
                    tar.extractall(staging)
                    count = sum(1 for m in members if m.isfile())

                # Atomically replace each managed root: drop the stale local copy,
                # then swap in the freshly-staged tree.
                replaced = []
                for root in self.MANAGED_ROOTS:
                    src = staging / root
                    if not src.exists():
                        continue
                    dst = self._target / root
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    elif dst.is_dir():
                        shutil.rmtree(dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    replaced.append(root)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            self.fetch_done.emit({
                "type": "success", "files": count, "bytes": len(blob),
                "roots": replaced,
            })
        except Exception as exc:
            self.fetch_done.emit({"type": "error", "message": str(exc)})


# ------------------------------------------------------------------ #
# Multi-line user input                                                #
# ------------------------------------------------------------------ #

class UserInputEdit(QTextEdit):
    """QTextEdit that emits send_requested on Alt+Enter; plain Enter inserts a newline."""

    send_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
            event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self.send_requested.emit()
        else:
            super().keyPressEvent(event)


# ------------------------------------------------------------------ #
# Chat widget                                                          #
# ------------------------------------------------------------------ #

class ChatWidget(QWidget):
    """Standalone chat panel for interacting with a remote LLM backend."""

    # Connection lifecycle — the Sleep tab uses these to adopt/detach an
    # in-progress reflection run across a desktop<->laptop handoff (the run is
    # server-owned; only the viewport moves).
    connected = pyqtSignal()
    disconnected = pyqtSignal()

    def __init__(self, parent=None, server_url: Optional[str] = None):
        super().__init__(parent)
        self.model_loaded = False
        self.chat_worker: Optional[ChatWorker] = None
        self._connect_worker: Optional[ConnectWorker] = None
        self.conversation_history: list = []  # local display tracking only
        self.max_new_tokens_setting = "75%"
        self.temperature = 1.0
        self.top_p = 0.95
        self.context_length = 32768
        # Chat-time RAG channel gates (all on by default). Off ⇒ that channel is not
        # injected; all off ⇒ chatting with the adapter only.
        self.rag_history = True
        self.rag_facts = True
        self.rag_persona = True
        # Brightest RGB channel (0-255) the red↔green tension tint may reach. In HSV
        # the value component IS the max channel, so this caps it exactly; full-bright
        # tints (~0xCC) wash out on a white background, hence the darker default.
        self.tension_color_max = 0x7F
        self._server_model_id: str = ""
        self._server_adapter_id: str = ""
        self._server_base_quant: str = ""
        self.debug_enabled = False
        self._streaming_assistant_active = False
        self._streaming_response_start_pos: Optional[int] = None
        self._streaming_turn_start_pos: Optional[int] = None
        self._pending_debug_lines: list[str] = []
        self._meta_exchange_id: Optional[str] = None
        self._meta_text: str = ""
        self._meta_worker: Optional[MetaFeedbackWorker] = None
        # Retry (roll back a collapsed reply): the chat-log position where the latest
        # completed turn begins (its "You:" line), so Retry can excise it, and the worker
        # driving the server-side rollback. None ⇒ nothing to retry (fresh chat / preview).
        self._last_completed_turn_start_pos: Optional[int] = None
        self._retry_worker: Optional[RetryWorker] = None
        self._sessions_worker: Optional[SessionsWorker] = None
        self._load_session_worker: Optional[LoadSessionWorker] = None
        self._preview_worker: Optional[PreviewSessionWorker] = None
        self._delete_worker: Optional[DeleteSessionWorker] = None
        self._reflect_reset_worker: Optional[ResetReflectionWorker] = None
        # live = normal chat; preview = read-only past session in the log;
        # continued = server conversation restored from a past session.
        self._session_mode = "live"
        self._active_filename: Optional[str] = None
        self._sessions_select_filename: Optional[str] = None
        self._session_meta_by_file: dict = {}
        self.text_font = QFont("Courier")

        self._client = BackendClient()
        self._load_config(server_url)
        self._build_ui()

    # ---------------------------------------------------------------- #
    # Config persistence                                                #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _config_path() -> Path:
        # config.json lives at the project root, two levels above client/ui/
        return Path(__file__).resolve().parents[2] / "config.json"

    @staticmethod
    def _url_from_host(host: str) -> str:
        return f"ws://{host.strip() or 'localhost'}:8765"

    def _load_config(self, explicit_url: Optional[str]) -> None:
        self._server_url = _DEFAULT_SERVER_URL
        self._saved_host = "localhost"
        self._saved_user = ""
        try:
            p = self._config_path()
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                saved_host = data.get("server_host", "")
                if saved_host:
                    self._saved_host = saved_host
                    self._server_url = self._url_from_host(saved_host)
                self._saved_user = data.get("user_name", "") or ""
        except Exception:
            pass
        # --server CLI flag always wins over saved config
        if explicit_url is not None:
            self._server_url = explicit_url
            try:
                self._saved_host = explicit_url.split("//", 1)[1].split(":")[0]
            except Exception:
                pass

    def save_config(self) -> None:
        try:
            p = self._config_path()
            with open(p, "w") as f:
                json.dump(
                    {
                        "server_host": self.txt_host.text().strip() or "localhost",
                        "user_name": self._current_user(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def _current_user(self) -> str:
        """The name of the person currently using this client (may be empty)."""
        return self.txt_user.text().strip()

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        dir_layout = QHBoxLayout()

        dir_layout.addWidget(QLabel("Host:"))
        self.txt_host = QLineEdit(self._saved_host)
        self.txt_host.setMaximumWidth(160)
        self.txt_host.textChanged.connect(self._on_host_changed)
        dir_layout.addWidget(self.txt_host)

        dir_layout.addWidget(QLabel("User:"))
        self.txt_user = QLineEdit(self._saved_user)
        self.txt_user.setMaximumWidth(120)
        self.txt_user.setPlaceholderText("your name")
        self.txt_user.setToolTip("Name Ava sees you as — lets her tell users apart.")
        dir_layout.addWidget(self.txt_user)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect_or_disconnect)
        dir_layout.addWidget(self.btn_connect)

        self.btn_clear_context = QPushButton("New chat")
        self.btn_clear_context.clicked.connect(self._on_clear_context)
        self.btn_clear_context.setEnabled(False)
        dir_layout.addWidget(self.btn_clear_context)

        self.btn_restart_server = QPushButton("Restart server")
        self.btn_restart_server.clicked.connect(self._on_restart_server)
        dir_layout.addWidget(self.btn_restart_server)

        self.btn_update_server = QPushButton("Update and restart server")
        self.btn_update_server.setToolTip(
            "Pull the latest code on the server (git pull) and restart the "
            "inference server. Push your changes first, then click this."
        )
        self.btn_update_server.clicked.connect(self._on_update_server)
        dir_layout.addWidget(self.btn_update_server)

        self.btn_fetch_artifacts = QPushButton("Fetch artifacts")
        self.btn_fetch_artifacts.setToolTip(
            "Download the server's chats + reflection data into this local "
            "repository for offline analysis (preserves directory structure)."
        )
        self.btn_fetch_artifacts.setEnabled(False)
        self.btn_fetch_artifacts.clicked.connect(self._on_fetch_artifacts)
        dir_layout.addWidget(self.btn_fetch_artifacts)

        dir_layout.addWidget(QLabel("Model:"))
        self.lbl_model = QLabel("—")
        dir_layout.addWidget(self.lbl_model)

        dir_layout.addWidget(QLabel("Adapter:"))
        self.lbl_adapter = QLabel("—")
        self.lbl_adapter.setToolTip("LoRA adapter revision currently attached "
                                    "(\"base\" = frozen base model, no adapter).")
        dir_layout.addWidget(self.lbl_adapter)

        dir_layout.addStretch()
        layout.addLayout(dir_layout)

        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Max new tokens:"))
        self.txt_max_new_tokens = QLineEdit(self.max_new_tokens_setting)
        self.txt_max_new_tokens.setMaximumWidth(100)
        self.txt_max_new_tokens.textChanged.connect(self._on_max_new_tokens_changed)
        params_layout.addWidget(self.txt_max_new_tokens)

        params_layout.addWidget(QLabel("Temperature:"))
        self.spn_temperature = QDoubleSpinBox()
        self.spn_temperature.setRange(0.0, 2.0)
        self.spn_temperature.setSingleStep(0.05)
        self.spn_temperature.setDecimals(2)
        self.spn_temperature.setValue(self.temperature)
        self.spn_temperature.setMaximumWidth(80)
        self.spn_temperature.setToolTip(
            "Sampling temperature. 0 = deterministic; 1.0 = default (Gemma 4 recommended, "
            "paired with top_p 0.95 + top_k 64); >1.0 = more chaotic."
        )
        self.spn_temperature.valueChanged.connect(self._on_temperature_changed)
        params_layout.addWidget(self.spn_temperature)

        params_layout.addWidget(QLabel("Tint max:"))
        self.spn_tint_max = QSpinBox()
        self.spn_tint_max.setRange(0x20, 0xFF)
        self.spn_tint_max.setValue(self.tension_color_max)
        self.spn_tint_max.setMaximumWidth(70)
        self.spn_tint_max.setToolTip(
            "Brightness cap (0-255) for the green/red per-token tension colors. "
            "Lower = darker, easier to read on a white background; 127 (0x7F) is "
            "a good default. Applies to newly rendered replies."
        )
        self.spn_tint_max.valueChanged.connect(self._on_tint_max_changed)
        params_layout.addWidget(self.spn_tint_max)

        self.chk_debug = QCheckBox("Debug")
        self.chk_debug.setToolTip(
            "Show the full prompt this turn conditions on, above the reply, "
            "colour-coded by segment: blue = system framing (prompt, identity, "
            "temporal anchor, persona/user portraits), green = the injected RAG "
            "block, red = the user turn. Assistant history is muted grey."
        )
        self.chk_debug.setChecked(self.debug_enabled)
        self.chk_debug.toggled.connect(self._on_debug_toggled)
        params_layout.addWidget(self.chk_debug)

        params_layout.addSpacing(16)
        params_layout.addWidget(QLabel("RAG:"))
        self.chk_rag_history = QCheckBox("History")
        self.chk_rag_history.setToolTip(
            "Inject past-chat recall into this chat. Off = no experiential recall. "
            "(The wander/TIL channel this box also used to gate is off everywhere "
            "pending redesign.)"
        )
        self.chk_rag_history.setChecked(self.rag_history)
        self.chk_rag_history.toggled.connect(self._on_rag_history_toggled)
        params_layout.addWidget(self.chk_rag_history)

        self.chk_rag_facts = QCheckBox("Facts")
        self.chk_rag_facts.setToolTip(
            "Inject distilled [fact] reflection memory into this chat. (Open [ask] "
            "recalls, which used to share this box, are off everywhere pending "
            "redesign — so all three memory slots now go to facts.)"
        )
        self.chk_rag_facts.setChecked(self.rag_facts)
        self.chk_rag_facts.toggled.connect(self._on_rag_facts_toggled)
        params_layout.addWidget(self.chk_rag_facts)

        self.chk_rag_persona = QCheckBox("Persona")
        self.chk_rag_persona.setToolTip(
            "Inject who Ava is into this chat. Normally that is her persona DIGEST — "
            "one standing self-portrait (voice, her established dispositions, and the "
            "lines she won't cross) present every turn, instead of whichever one or two "
            "[persona] statements happened to match the message. If there is no digest "
            "yet, or nothing in it has matured, the slot instead carries the 'what you "
            "are is not yet decided' framing, and the old per-statement [persona] recall "
            "comes back. Unchecking drops all of it. All three RAG boxes off = "
            "chatting with the adapter only."
        )
        self.chk_rag_persona.setChecked(self.rag_persona)
        self.chk_rag_persona.toggled.connect(self._on_rag_persona_toggled)
        params_layout.addWidget(self.chk_rag_persona)

        params_layout.addStretch()
        layout.addLayout(params_layout)

        notes_layout = QHBoxLayout()
        notes_layout.addWidget(QLabel("Session notes:"))
        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText(
            "Free-form intent — what you're testing in this session (saved on Enter / focus loss)"
        )
        self.txt_notes.editingFinished.connect(self._on_notes_committed)
        notes_layout.addWidget(self.txt_notes, 1)
        layout.addLayout(notes_layout)

        # Horizontal splitter: [sessions panel | chat area]
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: past sessions list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.addWidget(QLabel("Past chats"))
        self.lst_sessions = QListWidget()
        self.lst_sessions.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.lst_sessions, 1)
        self.btn_continue = QPushButton("Continue chat")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self._on_continue_chat)
        left_layout.addWidget(self.btn_continue)
        self.btn_reflect_again = QPushButton("Re-reflect chat")
        self.btn_reflect_again.setEnabled(False)
        self.btn_reflect_again.setToolTip(
            "Delete the selected chat(s) reflection sidecar so the next Sleep run "
            "reflects the chat again from scratch. The transcript is untouched, but "
            "everything reflection derived from it — the reflect-once freeze, the "
            "per-exchange verdicts and trainable targets (human-locked repairs "
            "included), the consolidation summary and the retrieval anchors — is "
            "dropped and re-derived under the current persona."
        )
        self.btn_reflect_again.clicked.connect(self._on_reflect_again)
        left_layout.addWidget(self.btn_reflect_again)
        self.btn_delete_chat = QPushButton("Delete chat")
        self.btn_delete_chat.setEnabled(False)
        self.btn_delete_chat.setToolTip(
            "Permanently delete the selected chat(s) from the server and drop "
            "them from memory retrieval."
        )
        self.btn_delete_chat.clicked.connect(self._on_delete_chat)
        left_layout.addWidget(self.btn_delete_chat)
        h_splitter.addWidget(left_panel)

        # Right panel: chat log + input (vertical splitter)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        v_splitter = QSplitter(Qt.Orientation.Vertical)

        self.txt_chat_log = QTextEdit()
        self.txt_chat_log.setReadOnly(True)
        self.txt_chat_log.setPlaceholderText("Chat log will appear here...")
        self.txt_chat_log.setFont(self.text_font)
        v_splitter.addWidget(self.txt_chat_log)

        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)

        input_row = QHBoxLayout()
        self.txt_user_input = UserInputEdit()
        self.txt_user_input.setPlaceholderText("Enter your message here… (Alt+Enter to send)")
        self.txt_user_input.send_requested.connect(self._on_send_message)
        self.txt_user_input.setFont(self.text_font)
        fm = self.txt_user_input.fontMetrics()
        self.txt_user_input.setMinimumHeight(fm.lineSpacing() * 6 + 16)
        input_row.addWidget(self.txt_user_input, 1)

        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self._on_send_message)
        self.btn_send.setEnabled(False)
        input_row.addWidget(self.btn_send)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip(
            "Stop and discard this reply, restore your prompt, then adjust "
            "Temperature and retry."
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_generation)
        input_row.addWidget(self.btn_stop)

        self.btn_retry = QPushButton("Retry")
        self.btn_retry.setToolTip(
            "Discard the latest reply (e.g. a collapsed/degenerate one), restore your "
            "prompt, then adjust Temperature and send again. It is rolled off the "
            "transcript so it never reaches reflection or training."
        )
        self.btn_retry.setEnabled(False)
        self.btn_retry.clicked.connect(self._on_retry)
        input_row.addWidget(self.btn_retry)

        self.btn_meta = QPushButton("Meta…")
        self.btn_meta.setToolTip(
            "Add notification-only feedback for Ava to consider when reflecting "
            "on her latest reply. Available only until the next reply lands."
        )
        self.btn_meta.setEnabled(False)
        self.btn_meta.clicked.connect(self._on_meta_feedback)
        input_row.addWidget(self.btn_meta)

        input_layout.addLayout(input_row)

        # ── anchor-match preview ─────────────────────────────────────────────────
        # Shows which stored exchange anchors the message being typed WOULD match. It is
        # a read-only diagnostic: nothing here is retrieved or injected into the prompt.
        # It exists to make the query side of anchor retrieval judgeable against real
        # text before any of it is wired into chat — in particular whether inflected
        # forms match their tags, which is where a lexical matcher fails first.
        anchor_row = QHBoxLayout()
        anchor_row.setContentsMargins(0, 2, 0, 0)
        self.chk_anchor_preview = QCheckBox("Match preview")
        self.chk_anchor_preview.setToolTip(
            "As you type, show which stored exchange anchors (ABOUT + tags) your message "
            "would match. Diagnostic only — nothing is retrieved or injected."
        )
        self.chk_anchor_preview.setChecked(False)
        self.chk_anchor_preview.toggled.connect(self._on_anchor_preview_toggled)
        anchor_row.addWidget(self.chk_anchor_preview)
        self.lbl_anchor_matches = QLabel("")
        self.lbl_anchor_matches.setWordWrap(True)
        self.lbl_anchor_matches.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_anchor_matches.setStyleSheet("color: palette(mid);")
        anchor_row.addWidget(self.lbl_anchor_matches, 1)
        input_layout.addLayout(anchor_row)

        # Debounce: the RPC is cheap, but one per keystroke is still noise on the wire.
        self._anchor_timer = QTimer(self)
        self._anchor_timer.setSingleShot(True)
        self._anchor_timer.setInterval(450)
        self._anchor_timer.timeout.connect(self._request_anchor_matches)
        self._anchor_worker: Optional[AnchorMatchWorker] = None
        self.txt_user_input.textChanged.connect(self._on_draft_changed)

        v_splitter.addWidget(input_container)
        v_splitter.setStretchFactor(0, 4)
        v_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(v_splitter)
        h_splitter.addWidget(right_widget)

        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setSizes([200, 700])

        layout.addWidget(h_splitter, 1)

        self.txt_user_input.setEnabled(False)
        self.lst_sessions.itemSelectionChanged.connect(self._on_session_selection_changed)

    # ---------------------------------------------------------------- #
    # Font updates (called by main window)                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, text_font: QFont) -> None:
        self.text_font = text_font
        self.txt_chat_log.setFont(text_font)
        self.txt_user_input.setFont(text_font)

    # ---------------------------------------------------------------- #
    # Parameter callbacks                                               #
    # ---------------------------------------------------------------- #

    def _on_host_changed(self, text: str) -> None:
        self._server_url = self._url_from_host(text)

    def _on_max_new_tokens_changed(self, text: str) -> None:
        self.max_new_tokens_setting = text.strip() or "75%"

    def _on_temperature_changed(self, value: float) -> None:
        self.temperature = float(value)

    def _on_tint_max_changed(self, value: int) -> None:
        self.tension_color_max = int(value)

    def _on_notes_committed(self) -> None:
        if not self._client.is_connected():
            return
        try:
            self._client.set_session_notes(self.txt_notes.text())
        except Exception as exc:
            self.txt_chat_log.append(f"[notes] Failed to save session notes: {exc}\n")

    def _on_debug_toggled(self, enabled: bool) -> None:
        self.debug_enabled = enabled
        self.txt_chat_log.append(
            "[debug] Prompt display "
            + ("enabled — each turn shows its full prompt before the reply "
               "(blue = system framing, green = injected RAG, red = user turn)"
               if enabled else "disabled")
            + ".\n"
        )

    def _on_rag_history_toggled(self, enabled: bool) -> None:
        self.rag_history = enabled

    def _on_rag_facts_toggled(self, enabled: bool) -> None:
        self.rag_facts = enabled

    def _on_rag_persona_toggled(self, enabled: bool) -> None:
        self.rag_persona = enabled

    # ---------------------------------------------------------------- #
    # Connect / disconnect                                              #
    # ---------------------------------------------------------------- #

    def _on_connect_or_disconnect(self) -> None:
        if self._client.is_connected():
            self._do_disconnect()
        else:
            self._do_connect()

    def _do_connect(self) -> None:
        self.save_config()
        self.btn_connect.setText("Connecting…")
        self.btn_connect.setEnabled(False)
        self.txt_host.setEnabled(False)
        self.txt_chat_log.clear()
        self.txt_chat_log.append(f"Connecting to {self._server_url}…\n")

        self._connect_worker = ConnectWorker(self._client, self._server_url)
        self._connect_worker.connect_done.connect(self._on_connect_done)
        self._connect_worker.start()

    def _on_connect_done(self, result: dict) -> None:
        if result.get("type") in ("error", "connection_error"):
            self.txt_chat_log.append(f"✗ Connection failed: {result.get('message', 'Unknown error')}\n")
            self.btn_connect.setText("Connect")
            self.btn_connect.setEnabled(True)
            self.txt_host.setEnabled(True)
            return

        self.context_length = int(result.get("context_length", 32768))
        self._server_model_id = result.get("model_id") or ""
        self._server_adapter_id = result.get("adapter_id") or ""
        # base_quant travels on the WebSocket status (the inference server owns the
        # config now, not the watchdog), so the Migrate tab reads it from here.
        self._server_base_quant = result.get("base_quant") or ""
        # Pin the client's HTTP ports (watchdog job runner + inference sidecar) from
        # config now that we're connected, so precision/artifacts/job calls resolve.
        self._client.mgmt_port = self._get_mgmt_port()
        self._client.sidecar_port = self._get_sidecar_port()
        self.model_loaded = bool(result.get("loaded", False))
        self.conversation_history = []
        self._session_mode = "live"
        self._active_filename = None
        self._close_meta_window()

        model_label = self._server_model_id.split("/")[-1] if self._server_model_id else "—"
        self.lbl_model.setText(model_label)
        self._update_adapter_label()

        self.btn_connect.setText("Disconnect")
        self.btn_connect.setEnabled(True)
        self.txt_host.setEnabled(False)

        if self.model_loaded:
            self.btn_clear_context.setEnabled(True)
            self.txt_user_input.setEnabled(True)
            self.txt_chat_log.append(
                f"✓ Connected. Model: {self._server_model_id} (ctx={self.context_length})\n"
            )
            self.txt_chat_log.append(
                "Select a past chat to preview it, then click \"Continue chat\" "
                "to restore its context before sending.\n"
            )
        else:
            self.btn_send.setEnabled(False)
            self.btn_clear_context.setEnabled(False)
            self.txt_user_input.setEnabled(False)
            self.txt_chat_log.append("Connected, but no model is loaded on the server.\n")

        self._update_continue_button()

        self._refresh_sessions()
        # Decide whether "Fetch artifacts" is meaningful for this connection
        # (disabled when the server shares this checkout's storage).
        self._refresh_fetch_artifacts_state()
        # Let the Sleep tab adopt any reflection run already in progress on the
        # server (e.g. started on the other device before this handoff).
        self.connected.emit()

    def _update_adapter_label(self) -> None:
        """Show the attached LoRA adapter revision (its directory name), or
        "base" when the server is running the frozen base with no adapter."""
        adapter_id = (self._server_adapter_id or "").strip()
        if not self.model_loaded:
            self.lbl_adapter.setText("—")
        elif adapter_id:
            self.lbl_adapter.setText(adapter_id.rstrip("/").split("/")[-1])
        else:
            self.lbl_adapter.setText("base")

    def _do_disconnect(self) -> None:
        if self.chat_worker is not None and self.chat_worker.isRunning():
            QMessageBox.warning(
                self, "Generation in progress",
                "Wait for generation to finish before disconnecting."
            )
            return
        if self._meta_worker is not None and self._meta_worker.isRunning():
            QMessageBox.warning(
                self, "Meta feedback is saving",
                "Wait for the Meta feedback save to finish before disconnecting."
            )
            return
        if self._retry_worker is not None and self._retry_worker.isRunning():
            QMessageBox.warning(
                self, "Retry in progress",
                "Wait for the retry to finish before disconnecting."
            )
            return
        self._client.disconnect()
        self._client.last_status.clear()
        self.model_loaded = False
        self._server_model_id = ""
        self._server_adapter_id = ""
        self._server_base_quant = ""
        self.conversation_history = []
        self._session_mode = "live"
        self._active_filename = None
        self._last_completed_turn_start_pos = None
        self._close_meta_window()
        self.lbl_model.setText("—")
        self.lbl_adapter.setText("—")
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)
        self.txt_host.setEnabled(True)
        self.btn_send.setEnabled(False)
        self.btn_clear_context.setEnabled(False)
        self.txt_user_input.setEnabled(False)
        self.btn_fetch_artifacts.setEnabled(False)
        self.btn_fetch_artifacts.setText("Fetch artifacts")
        self.lst_sessions.clear()
        self._update_continue_button()
        self.txt_chat_log.append("Disconnected.\n")
        self.disconnected.emit()

    def _get_mgmt_port(self) -> int:
        try:
            p = Path.home() / ".avadeploy.json"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                return int(data.get("mgmt_port", 8766))
        except Exception:
            pass
        return 8766

    def _get_sidecar_port(self) -> int:
        """The inference HTTP sidecar port (artifacts/export/chats/precision).

        Mirrors ``_get_mgmt_port`` — overridable via ~/.avadeploy.json's
        ``sidecar_port`` (default 8767, matching the watchdog's --http-port).
        """
        try:
            p = Path.home() / ".avadeploy.json"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                return int(data.get("sidecar_port", 8767))
        except Exception:
            pass
        return 8767

    def _on_restart_server(self) -> None:
        host = self.txt_host.text().strip() or "localhost"
        reply = QMessageBox.question(
            self,
            "Restart Server",
            f"Are you sure you want to restart the inference server on {host}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_restart_server.setText("Restarting…")
        self.btn_restart_server.setEnabled(False)
        self.txt_chat_log.append(f"[server] Requesting restart on {host}…\n")

        if self._client.is_connected():
            self._do_disconnect()

        self._restart_worker = RestartWorker(host, self._get_mgmt_port())
        self._restart_worker.restart_done.connect(self._on_restart_done)
        self._restart_worker.start()

    def _on_restart_done(self, result: dict) -> None:
        self.btn_restart_server.setText("Restart server")
        self.btn_restart_server.setEnabled(True)

        if result.get("type") == "success":
            self.txt_chat_log.append("✓ Restart command accepted by watchdog. Inference server is relaunching.\n")
        else:
            self.txt_chat_log.append(f"✗ Restart failed: {result.get('message', 'Unknown error')}\n")

    def _on_update_server(self) -> None:
        host = self.txt_host.text().strip() or "localhost"
        reply = QMessageBox.question(
            self,
            "Update and Restart Server",
            f"Pull the latest code (git pull) and restart the inference server on {host}?\n\n"
            "Make sure you have pushed your changes first.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_update_server.setText("Updating…")
        self.btn_update_server.setEnabled(False)
        self.txt_chat_log.append(f"[server] Requesting git pull + restart on {host}…\n")

        if self._client.is_connected():
            self._do_disconnect()

        self._pull_worker = PullWorker(host, self._get_mgmt_port())
        self._pull_worker.pull_done.connect(self._on_update_done)
        self._pull_worker.start()

    def _on_update_done(self, result: dict) -> None:
        self.btn_update_server.setText("Update and restart server")
        self.btn_update_server.setEnabled(True)

        if result.get("type") == "success":
            data = result.get("data", {})
            output = (data.get("output") or "").strip()
            if output:
                self.txt_chat_log.append(output + "\n")
            if data.get("ok"):
                self.txt_chat_log.append("✓ Server pulled the latest code. Inference server is relaunching.\n")
            else:
                self.txt_chat_log.append(
                    "✗ git pull failed on the server; inference server relaunched with the existing code.\n"
                )
        else:
            self.txt_chat_log.append(f"✗ Update failed: {result.get('message', 'Unknown error')}\n")

    # ---------------------------------------------------------------- #
    # Fetch artifacts (chats + reflection data → local repository)      #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _local_server_dir() -> Path:
        # server/ lives two levels above client/ui/ → server
        return Path(__file__).resolve().parents[2] / "server"

    @classmethod
    def _local_inference_dir(cls) -> Path:
        return cls._local_server_dir() / "inference"

    def _refresh_fetch_artifacts_state(self) -> None:
        """Query the watchdog for its data dir and decide if a fetch is meaningful.

        When the server reports the same hostname and the same resolved data dir
        as this checkout, the two share one repository on storage — the fetch is a
        no-op, so the button stays disabled.
        """
        host = self.txt_host.text().strip() or "localhost"
        self.btn_fetch_artifacts.setEnabled(False)
        self.btn_fetch_artifacts.setText("Checking…")
        self._mgmt_status_worker = MgmtStatusWorker(host, self._get_mgmt_port())
        self._mgmt_status_worker.status_done.connect(self._on_mgmt_status_done)
        self._mgmt_status_worker.start()

    def _on_mgmt_status_done(self, result: dict) -> None:
        self.btn_fetch_artifacts.setText("Fetch artifacts")
        if not self._client.is_connected():
            return  # disconnected while the check was in flight

        same_repo = False
        if result.get("type") == "success":
            data = result.get("data", {})
            server_host = data.get("hostname")
            server_data_dir = data.get("data_dir")
            if server_host and server_data_dir:
                local_data = self._local_inference_dir() / "data"
                same_repo = (
                    server_host == socket.gethostname()
                    and os.path.realpath(server_data_dir) == os.path.realpath(str(local_data))
                )

        if same_repo:
            self.btn_fetch_artifacts.setEnabled(False)
            self.btn_fetch_artifacts.setToolTip(
                "Server shares this repository's storage — artifacts are already local."
            )
        else:
            self.btn_fetch_artifacts.setEnabled(True)
            self.btn_fetch_artifacts.setToolTip(
                "Download the server's chats, reflection archive + logs into this "
                "local repository for offline analysis. Mirrors the server — the "
                "inference/data, reflections and logs subtrees are replaced "
                "(stale files from wiped runs are removed)."
            )

    def _on_fetch_artifacts(self) -> None:
        host = self.txt_host.text().strip() or "localhost"
        local_server = self._local_server_dir()
        reply = QMessageBox.question(
            self,
            "Fetch artifacts",
            f"Download chats, reflection archive + logs from {host} into\n"
            f"{local_server}?\n\n"
            "This mirrors the server: the local inference/data, reflections and "
            "logs subtrees are replaced with the fetched copy (stale files from "
            "wiped runs are removed).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_fetch_artifacts.setEnabled(False)
        self.btn_fetch_artifacts.setText("Fetching…")
        self.txt_chat_log.append(f"[artifacts] Fetching from {host}…\n")

        self._fetch_artifacts_worker = FetchArtifactsWorker(
            host, local_server, self._get_sidecar_port()
        )
        self._fetch_artifacts_worker.fetch_done.connect(self._on_fetch_artifacts_done)
        self._fetch_artifacts_worker.start()

    def _on_fetch_artifacts_done(self, result: dict) -> None:
        self.btn_fetch_artifacts.setText("Fetch artifacts")
        self.btn_fetch_artifacts.setEnabled(self._client.is_connected())

        if result.get("type") == "success":
            roots = ", ".join(result.get("roots", [])) or "—"
            self.txt_chat_log.append(
                f"✓ Fetched {result.get('files', 0)} file(s) "
                f"({result.get('bytes', 0):,} bytes) into "
                f"{self._local_server_dir()} [{roots}]\n"
            )
        else:
            self.txt_chat_log.append(
                f"✗ Fetch failed: {result.get('message', 'Unknown error')}\n"
            )

    # ---------------------------------------------------------------- #
    # Status accessors (used by main window)                           #
    # ---------------------------------------------------------------- #

    def get_context_usage(self) -> Tuple[int, int, int]:
        total = max(1, int(self.context_length))
        if not self.model_loaded:
            return 0, total, 0
        used = self._client.last_status.get("input_tokens", 0)
        used = max(0, min(total, used))
        return used, total, int(used * 100 / total)

    def get_memory_status(self) -> str:
        return self._client.last_status.get("memory", "disconnected")

    def get_generation_speed(self) -> Optional[float]:
        """Last measured generation speed in tok/s (live during a reply, finalized
        after), or None if nothing has been generated yet this session."""
        return self._client.last_status.get("tokens_per_sec")

    def get_token_economy(self) -> Optional[dict]:
        """The user-token imprint meter last reported by the server:
        {accumulated, consumed, wanders_consumed, available, tokens_per}, or None
        if nothing has been reported yet (fresh on connect + after each live turn)."""
        return self._client.last_status.get("token_economy")

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected()

    # ---------------------------------------------------------------- #
    # Chat log helpers                                                  #
    # ---------------------------------------------------------------- #

    def _append_chat_text(self, text: str) -> Tuple[int, int]:
        cursor = self.txt_chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        start_pos = cursor.position()
        cursor.insertText(text)
        end_pos = cursor.position()
        self.txt_chat_log.setTextCursor(cursor)
        self.txt_chat_log.ensureCursorVisible()
        return start_pos, end_pos

    # ---------------------------------------------------------------- #
    # Message handling                                                  #
    # ---------------------------------------------------------------- #

    def _on_clear_context(self) -> None:
        if self._meta_worker is not None and self._meta_worker.isRunning():
            QMessageBox.warning(
                self, "Meta feedback is saving",
                "Wait for the Meta feedback save to finish before starting a new chat."
            )
            return
        if self._client.is_connected():
            self._client.clear_context(user=self._current_user())
        self.conversation_history = []
        self._session_mode = "live"
        self._active_filename = None
        self._last_completed_turn_start_pos = None
        self.txt_chat_log.clear()
        self.txt_chat_log.append("Started a new chat.\n")
        self.txt_user_input.clear()
        self.txt_notes.clear()
        self._close_meta_window()
        self.txt_user_input.setFocus()
        self._refresh_sessions()

    # ── anchor-match preview (diagnostic; never touches the prompt) ────────────────
    def _on_anchor_preview_toggled(self, on: bool) -> None:
        if on:
            self._on_draft_changed()
        else:
            self._anchor_timer.stop()
            self.lbl_anchor_matches.setText("")

    def _on_draft_changed(self) -> None:
        if not self.chk_anchor_preview.isChecked():
            return
        if not self.txt_user_input.toPlainText().strip():
            self.lbl_anchor_matches.setText("")
            self._anchor_timer.stop()
            return
        self._anchor_timer.start()

    def _request_anchor_matches(self) -> None:
        if not self.chk_anchor_preview.isChecked():
            return
        text = self.txt_user_input.toPlainText().strip()
        if not text or not self._client.is_connected():
            return
        # One in flight at a time: a stale reply for an older draft is worse than a
        # slightly late one, and the debounce already caps the rate.
        if self._anchor_worker is not None and self._anchor_worker.isRunning():
            return
        worker = AnchorMatchWorker(self._client, text)
        worker.matches_ready.connect(self._on_anchor_matches)
        self._anchor_worker = worker
        worker.start()

    def _on_anchor_matches(self, result: dict) -> None:
        if not self.chk_anchor_preview.isChecked():
            return
        corpus = result.get("corpus") or {}
        total = corpus.get("anchors", 0)
        pending = corpus.get("pending", 0)
        matches = result.get("matches") or []
        if not total:
            self.lbl_anchor_matches.setText(
                "no anchors stored yet — they are produced by reflection")
            return
        # Anchors a normal Sleep run has not folded out of the background checkpoint yet.
        note = f" ({pending} pending fold)" if pending else ""
        if not matches:
            self.lbl_anchor_matches.setText(f"no match ({total} anchors{note})")
            return
        # Show the tags that actually fired, plus where the best hit came from: the tag
        # answers "why did this match", the chat answers "is that the right memory".
        fired: list = []
        for m in matches:
            for tag in m.get("matched") or []:
                if tag not in fired:
                    fired.append(tag)
        head = matches[0]
        where = f"{head.get('session', '')}#{head.get('exchange_index')}"
        about = (head.get("about") or "").strip()
        if len(about) > 90:
            about = about[:90].rstrip() + "…"
        summary = f"{len(matches)} of {total}{note} · " + " ".join(f"#{t}" for t in fired[:8])
        if about:
            summary += f"  →  {where}: {about}"
        self.lbl_anchor_matches.setText(summary)

    def _on_send_message(self) -> None:
        if not self.model_loaded:
            QMessageBox.warning(self, "Not connected", "Please connect to the server first.")
            return
        if self._session_mode == "preview":
            QMessageBox.information(
                self,
                "Continue chat first",
                "This is a read-only preview of a past session.\n\n"
                "Click \"Continue chat\" to restore its context on the server, "
                "then send your next message.",
            )
            return
        message = self.txt_user_input.toPlainText().strip()
        if not message:
            return

        user = self._current_user()
        cursor = self.txt_chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._streaming_turn_start_pos = cursor.position()
        self.txt_chat_log.append(f"{user or 'You'}: {message}\n")
        self.conversation_history.append({"role": "user", "content": message})
        self.txt_user_input.clear()

        # The "Assistant: " marker is printed lazily (on the first chunk / on
        # response), not here — so any start-of-turn debug output (e.g. the full
        # prompt shown when Debug is checked) renders before the reply, not after it.
        self._streaming_assistant_active = False
        self._streaming_response_start_pos = None
        self._pending_debug_lines = []
        # A new turn supersedes any prior retry target; it becomes valid again only if
        # this reply lands successfully (set in _on_response_ready).
        self._last_completed_turn_start_pos = None

        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_retry.setEnabled(False)
        self.btn_meta.setEnabled(False)
        self.txt_user_input.setEnabled(False)

        self.chat_worker = ChatWorker(
            self._client,
            message,
            max_new_tokens_setting=self.max_new_tokens_setting,
            context_length=self.context_length,
            debug_enabled=self.debug_enabled,
            user=user,
            temperature=self.temperature,
            top_p=self.top_p,
            rag_history=self.rag_history,
            rag_facts=self.rag_facts,
            rag_persona=self.rag_persona,
        )

        self.chat_worker.response_chunk.connect(self._on_response_chunk)
        self.chat_worker.response_ready.connect(self._on_response_ready)
        self.chat_worker.error_occurred.connect(self._on_response_error)
        self.chat_worker.generation_cancelled.connect(self._on_generation_cancelled)
        self.chat_worker.debug_info.connect(self._on_debug_info)
        self.chat_worker.prompt_debug.connect(self._on_prompt_debug)
        self.chat_worker.facts_block.connect(self._on_facts_block)
        self.chat_worker.finished.connect(self._on_generation_finished)
        self.chat_worker.start()
        self._update_continue_button()

    def _on_meta_feedback(self) -> None:
        exchange_id = self._meta_exchange_id
        if not exchange_id:
            return
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Meta feedback",
            (
                "Your reaction will be shown to Ava only when she reflects on "
                "this reply. It is notification-only; she may accept, reject, "
                "reinterpret, or ignore it."
            ),
            self._meta_text,
        )
        text = text.strip()
        if not accepted:
            return
        if not text:
            QMessageBox.information(
                self, "Meta feedback", "Enter a comment, or press Cancel."
            )
            return
        if len(text) > _META_FEEDBACK_MAX_CHARS:
            QMessageBox.information(
                self,
                "Meta feedback",
                f"Meta feedback is limited to {_META_FEEDBACK_MAX_CHARS} characters.",
            )
            return
        self._meta_worker = MetaFeedbackWorker(
            self._client, exchange_id, text, self._current_user()
        )
        self._meta_worker.feedback_saved.connect(self._on_meta_feedback_saved)
        self._meta_worker.error_occurred.connect(self._on_meta_feedback_error)
        self._meta_worker.finished.connect(self._on_meta_feedback_finished)
        self._meta_worker.start()
        self._update_continue_button()

    def _on_meta_feedback_saved(self, exchange_id: str, feedback: dict) -> None:
        if exchange_id != self._meta_exchange_id:
            return
        self._meta_text = str(feedback.get("text") or "")
        self.btn_meta.setText("Edit Meta…")
        self.btn_meta.setToolTip(
            "Meta feedback saved for reflection. You may edit it until the next "
            "Ava reply lands."
        )

    def _on_meta_feedback_error(self, error: str) -> None:
        QMessageBox.warning(self, "Meta feedback", error)

    def _on_meta_feedback_finished(self) -> None:
        self._update_continue_button()

    def _close_meta_window(self) -> None:
        self._meta_exchange_id = None
        self._meta_text = ""
        if hasattr(self, "btn_meta"):
            self.btn_meta.setText("Meta…")
            self.btn_meta.setEnabled(False)
            self.btn_meta.setToolTip(
                "Add notification-only feedback for Ava to consider when reflecting "
                "on her latest reply. Available only until the next reply lands."
            )

    def _on_stop_generation(self) -> None:
        if self.chat_worker is None or not self.chat_worker.isRunning():
            return
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stopping…")
        self._client.cancel(discard=True)

    def _on_retry(self) -> None:
        """Roll the latest completed reply off the transcript and restore its prompt.

        The completed-turn counterpart of Stop/discard: for a reply that already landed
        (e.g. a collapsed/degenerate one), ask the server to drop it from the active
        transcript + conversation, then excise it from the chat log and put the prompt
        back so the operator can adjust Temperature and send again."""
        if self._last_completed_turn_start_pos is None:
            return
        if self.chat_worker is not None and self.chat_worker.isRunning():
            return
        if self._retry_worker is not None and self._retry_worker.isRunning():
            return
        self.btn_retry.setText("Retrying…")
        self.btn_retry.setEnabled(False)
        self._retry_worker = RetryWorker(self._client)
        self._retry_worker.retry_ready.connect(self._on_retry_ready)
        self._retry_worker.error_occurred.connect(self._on_retry_error)
        self._retry_worker.finished.connect(self._on_retry_finished)
        self._retry_worker.start()
        self._update_continue_button()

    def _on_retry_ready(self, result: dict) -> None:
        # The server has rolled the exchange off the transcript + conversation. Excise
        # the rendered turn (from its "You:" line to the end of the log) and restore the
        # prompt so it can be resent under adjusted sampling.
        start = self._last_completed_turn_start_pos
        if start is not None:
            cursor = self.txt_chat_log.textCursor()
            cursor.setPosition(start)
            cursor.movePosition(
                QTextCursor.MoveOperation.End,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()
            cursor.insertText(
                "↺ Reply discarded. Prompt restored; adjust Temperature and "
                "send again.\n\n"
            )
            self.txt_chat_log.setTextCursor(cursor)
            self.txt_chat_log.ensureCursorVisible()
        # Mirror the server-side rollback in the local display history (user + assistant).
        if self.conversation_history and self.conversation_history[-1]["role"] == "assistant":
            self.conversation_history.pop()
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()
        self.txt_user_input.setPlainText(str(result.get("user_prompt") or ""))
        self._last_completed_turn_start_pos = None
        self._close_meta_window()

    def _on_retry_error(self, error: str) -> None:
        QMessageBox.warning(self, "Retry", error)

    def _on_retry_finished(self) -> None:
        self.btn_retry.setText("Retry")
        # _update_continue_button re-enables the input now the worker has released the
        # RPC lock; focus it so the restored prompt is immediately editable.
        self._update_continue_button()
        if self.txt_user_input.isEnabled():
            self.txt_user_input.setFocus()

    def _on_generation_cancelled(self, message: str) -> None:
        start = self._streaming_turn_start_pos
        if start is not None:
            cursor = self.txt_chat_log.textCursor()
            cursor.setPosition(start)
            cursor.movePosition(
                QTextCursor.MoveOperation.End,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()
            cursor.insertText(
                "⏹ Generation stopped. Prompt restored; adjust Temperature "
                "and send again.\n\n"
            )
            self.txt_chat_log.setTextCursor(cursor)
            self.txt_chat_log.ensureCursorVisible()
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()
        self.txt_user_input.setPlainText(message)
        self._streaming_assistant_active = False
        self._streaming_response_start_pos = None
        self._streaming_turn_start_pos = None
        self._pending_debug_lines = []

    def _on_response_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._streaming_assistant_active:
            _, assistant_start = self._append_chat_text("Assistant: ")
            self._streaming_assistant_active = True
            self._streaming_response_start_pos = assistant_start
        self._append_chat_text(chunk)

    def _on_response_ready(self, response: str) -> None:
        if self._streaming_assistant_active:
            start = self._streaming_response_start_pos
            if start is not None:
                cursor = self.txt_chat_log.textCursor()
                cursor.setPosition(start)
                cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                self._insert_think_prob_line(cursor)
                self._insert_reply(cursor, response)
                self._append_tension_chip(cursor)
                cursor.insertText("\n\n")
                self.txt_chat_log.setTextCursor(cursor)
                self.txt_chat_log.ensureCursorVisible()
            else:
                cursor = self.txt_chat_log.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self._insert_think_prob_line(cursor)
                cursor.insertText(response)
                self._append_tension_chip(cursor)
                cursor.insertText("\n\n")
            self._streaming_assistant_active = False
            self._streaming_response_start_pos = None
        else:
            self.txt_chat_log.append(f"Assistant: {response}\n")

        self.conversation_history.append({"role": "assistant", "content": response})
        exchange_id = self._client.last_status.get("exchange_id")
        if exchange_id:
            self._meta_exchange_id = str(exchange_id)
            self._meta_text = ""
            self.btn_meta.setText("Meta…")
            self.btn_meta.setToolTip(
                "Add notification-only feedback for Ava to consider during reflection. "
                "This window closes when the next Ava reply lands."
            )
        else:
            self._close_meta_window()

        if self._pending_debug_lines:
            for line in self._pending_debug_lines:
                self.txt_chat_log.append(f"[debug] {line}\n")
            self._pending_debug_lines = []

        # The reply landed — mark this completed turn as the retry target (its "You:"
        # line start, captured before _on_generation_finished clears the streaming
        # pos). _update_continue_button (called from _on_generation_finished) enables
        # the Retry button off this. Only meaningful for live/continued turns.
        if self._session_mode in ("live", "continued"):
            self._last_completed_turn_start_pos = self._streaming_turn_start_pos

    # ---------------------------------------------------------------- #
    # Per-token tension coloring                                        #
    # ---------------------------------------------------------------- #

    def _tinted_qcolor(self, hue: float, saturation: float) -> QColor:
        """HSV → QColor with value capped by the UI's "Tint max" setting.

        HSV's value component is exactly the brightest RGB channel, so scaling the
        0-255 cap into value guarantees no channel exceeds it — the knob that keeps
        the red/green tension tints readable on a white background.
        """
        v = self.tension_color_max / 255.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, v)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def _confidence_to_qcolor(self, margin: float) -> QColor:
        """Per-token text color: margin 1 (decisive) = green, 0 (near-tie) = red.

        Driven off the top1-top2 probability margin (already in [0,1], so no empirical
        clip) — the inverse of friction, on the same hue axis as the tension chip so the
        two read consistently. Most tokens sit near margin 1 (green); the near-ties pop
        red against that field — exactly where the model nearly said something else.
        """
        f = 1.0 - max(0.0, min(1.0, float(margin)))   # friction = 1 - confidence
        hue = (1.0 - f) * (120.0 / 360.0)             # decisive=green → contested=red
        return self._tinted_qcolor(hue, 0.65)

    def _insert_colored_spans(self, cursor: QTextCursor, spans: list) -> None:
        """Insert ``[text, margin]`` spans, each tinted by its generation confidence."""
        for entry in spans or []:
            try:
                text, margin = entry[0], entry[1]
            except (TypeError, IndexError):
                continue
            if not text:
                continue
            fmt = QTextCharFormat()
            fmt.setForeground(self._confidence_to_qcolor(margin))
            cursor.insertText(text, fmt)

    def _insert_think_prob_line(self, cursor: QTextCursor) -> None:
        """Insert the ``Thinking: NN%`` diagnostic header above the reply.

        The percentage is the probability the model put on opening a thinking block at
        its first generated token (gemma-4 `<|channel>`) — a low value means it barely
        wanted to reason, i.e. the missing-CoT problem is serious; a high value means it
        usually thinks and a CoT-less reply was unlucky sampling. Green (wants to think)
        → red (won't). Omitted for families whose opener isn't sampled (qwen3) or when
        capture is off.
        """
        prob = self._client.last_status.get("think_open_prob")
        if prob is None:
            return
        fmt = QTextCharFormat()
        # prob=1 → green (hue 120°), prob=0 → red (hue 0°): high = wants to think.
        hue = max(0.0, min(1.0, float(prob))) * (120.0 / 360.0)
        fmt.setForeground(self._tinted_qcolor(hue, 0.55))
        cursor.insertText(f"Thinking: {float(prob) * 100:.0f}%\n", fmt)

    def _insert_reply(self, cursor: QTextCursor, response: str) -> None:
        """Insert the assistant reply at *cursor*, per-token colored by generation
        confidence when tension spans are available — the CoT shown above the reply,
        each delimited by a muted label. Green = decisive token, red = the model nearly
        said something else. Display-only: conversation history keeps the plain answer.
        Falls back to plain text when capture is off (no spans).
        """
        spans = self._client.last_status.get("tension_spans") or {}
        cot = spans.get("cot")
        answer = spans.get("answer")
        if not (cot or answer):
            cursor.insertText(response)
            return
        muted = QTextCharFormat()
        muted.setForeground(QColor(150, 150, 150))
        if cot:
            cursor.insertText("⟨thinking⟩\n", muted)
            self._insert_colored_spans(cursor, cot)
            cursor.insertText("\n⟨reply⟩\n", muted)
        if answer:
            self._insert_colored_spans(cursor, answer)
        else:
            cursor.insertText(response)

    # ---------------------------------------------------------------- #
    # Tension chip (debug overlay; not for production)                  #
    # ---------------------------------------------------------------- #

    # Contested fraction (in [0,1]) at which a segment is painted full red. Half the
    # tokens being near-ties is already extreme friction, so the scale tops out there.
    _FRICTION_FULL_SCALE = 0.5

    def _friction_to_qcolor(self, contested_frac: float) -> QColor:
        """Map a segment's contested fraction to a chip color.

        More contested (more friction) = warmer/redder; decisive = green. Drives off
        contested_frac rather than the old logit margin, so it is model-agnostic and
        does not need an empirical clip. Red-green axis is debug-only; not for production.
        """
        f = max(0.0, min(1.0, float(contested_frac) / self._FRICTION_FULL_SCALE))
        hue = (1.0 - f) * (120.0 / 360.0)  # 0 friction = green, full scale = red
        return self._tinted_qcolor(hue, 0.55)

    def _append_tension_chip(self, cursor: QTextCursor) -> None:
        """Insert a one-line tension chip beneath the just-rendered reply.

        Renders CoT and answer as separately-colored segments side by side, each tinted
        by its own contested fraction. That contrast is the point: friction trapped in
        the CoT (CoT warm, answer green) reads differently from friction that surfaced
        into the answer (answer warm). Numbers shown per segment: contested fraction,
        median + 10th-percentile probability margin, and peak entropy.
        """
        tension = self._client.last_status.get("tension")
        if not tension:
            return
        rows = [(label, tension.get(key) or {})
                for label, key in (("CoT", "cot"), ("ans", "answer"))]
        rows = [(label, seg) for label, seg in rows if seg.get("n_tokens")]
        if not rows:
            return

        neutral = QTextCharFormat()
        cursor.insertText("\n")
        cursor.insertText("friction · ", neutral)
        for i, (label, seg) in enumerate(rows):
            if i:
                cursor.insertText("   ", neutral)
            frac = float(seg.get("contested_frac") or 0.0)
            fmt = QTextCharFormat()
            fmt.setForeground(self._friction_to_qcolor(frac))
            cursor.insertText(
                f"{label} contested={frac * 100:.0f}% "
                f"med={seg.get('median_margin', 0):.2f} "
                f"p10={seg.get('p10_margin', 0):.2f} "
                f"H={seg.get('peak_entropy', 0):.2f}",
                fmt,
            )

        # Per-segment trace of the most-contested tokens — where the model nearly went
        # the other way. Each: position in the reply, the chosen token, its margin.
        for label, seg in rows:
            contested = seg.get("contested") or []
            if not contested:
                continue
            fmt = QTextCharFormat()
            fmt.setForeground(self._friction_to_qcolor(float(seg.get("contested_frac") or 0.0)))
            picks = "  ".join(
                f"@{c.get('position')}«{(c.get('token') or '').strip() or '·'}»{c.get('margin', 0):.2f}"
                for c in contested
            )
            cursor.insertText(f"\n  {label} forks · {picks}", fmt)

        # Reset format so subsequent inserts (the trailing \n\n) don't carry the chip color.
        cursor.setCharFormat(neutral)

    def _on_response_error(self, error: str) -> None:
        if self._streaming_assistant_active:
            self._append_chat_text(f"\n\n✗ {error}\n")
            self._streaming_assistant_active = False
            self._streaming_response_start_pos = None
        else:
            self.txt_chat_log.append(f"\n✗ {error}\n")
        # Remove the user message we optimistically added to local history
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
            self.conversation_history.pop()
        if self._pending_debug_lines:
            for line in self._pending_debug_lines:
                self.txt_chat_log.append(f"[debug] {line}\n")
            self._pending_debug_lines = []

    # Debug prompt view: one colour per segment kind. Blue = the system framing Ava is
    # given, green = the memory retrieval injected into it, red = what the user actually
    # said. Those three are what an operator is trying to tell apart when a turn goes
    # wrong — most immediately, whether the injected RAG block is carrying near-duplicate
    # lines — and in a single-colour dump the block is a paragraph indistinguishable from
    # the prompt around it. Assistant history stays muted: it is context, not this turn.
    _PROMPT_DEBUG_COLORS = {
        "system": QColor(38, 88, 190),
        "rag": QColor(20, 125, 60),
        "user": QColor(190, 40, 40),
        "assistant": QColor(140, 140, 140),
    }

    def _on_prompt_debug(self, segments: list) -> None:
        """Render the full prompt for this turn, colour-coded by segment kind.

        Runs before the reply streams, so it lands above it. Renders straight into the
        chat log rather than a side panel: the prompt belongs in the transcript at the
        point it was used, so scrolling back through a session shows what each turn was
        actually conditioned on.
        """
        if not segments:
            return
        cursor = self.txt_chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        rule = QTextCharFormat()
        rule.setForeground(QColor(150, 150, 150))
        cursor.insertText("\n───── prompt ─────\n", rule)

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", ""))
            if not text.strip():
                continue
            color = self._PROMPT_DEBUG_COLORS.get(
                str(seg.get("kind", "")), QColor(90, 90, 90)
            )
            label = QTextCharFormat()
            label.setForeground(color)
            label.setFontWeight(QFont.Weight.Bold)
            cursor.insertText(f"\n{seg.get('label', '?')}\n", label)
            body = QTextCharFormat()
            body.setForeground(color)
            cursor.insertText(text.rstrip() + "\n", body)

        cursor.insertText("\n───── end prompt ─────\n\n", rule)
        # Leave the widget's own cursor in a neutral format, or the reply that follows
        # inherits the last segment's colour.
        neutral = QTextCharFormat()
        neutral.setForeground(self.txt_chat_log.palette().text().color())
        cursor.setCharFormat(neutral)
        self.txt_chat_log.setTextCursor(cursor)
        self.txt_chat_log.ensureCursorVisible()

    # What each `skipped` reason means to somebody reading a reply. The distinction that
    # matters is between "she looked and there was nothing to take" (ordinary) and "the
    # channel is broken" (act on it) — the same split the server logs under `[facts]`.
    _FACTS_SKIP_TEXT = {
        "picked_nothing": "nothing picked — no fact on record bore on this message",
        "no_candidates": "nothing offered — the tree holds no showable fact",
        "empty": "nothing rendered from the pick",
        "no_model": "not run — no model loaded",
        "no_tree": "NOT RUN — no facts tree on this box (run `python -m graph.build`)",
        "no_prompt": "NOT RUN — the fetch prompt is missing",
        "no_module": "NOT RUN — the fetch module is missing",
        "generate_failed": "FAILED — the fetch pass could not generate",
        "candidates_failed": "FAILED — the candidate list could not be built",
        # Twice, since the live path retries once. Not a fact about the pass but about
        # the box (fragmentation / the VRAM ceiling) — the thing to check is the
        # `[alloc] expandable_segments` verdict the server prints at boot.
        "oom": "FAILED — CUDA out of memory twice (check [alloc] in server.log)",
        "error": "FAILED",
    }

    def _on_facts_block(self, info: dict) -> None:
        """Render stage 1's pick above the reply it is about to condition.

        Lands before the CoT because that is the point of it: the operator is reading the
        thought against the material it was given, and material that arrives after the
        conclusion is something else to reconcile rather than a premise. Rendered in the
        Debug view's `rag` green, since it IS an injected retrieval block — one an operator
        who has both views on should recognise as the same thing in both.

        Shown on every turn the fetch ran, empty result included: a reply built on nothing
        is a different reply from one built on three facts, and only the counts say which
        this was.
        """
        green = self._PROMPT_DEBUG_COLORS["rag"]
        skipped = str(info.get("skipped") or "")
        text = str(info.get("text") or "").rstrip()

        picked = info.get("picked") or []
        n_cands = int(info.get("n_candidates") or 0)
        head = f"facts fetched: {int(info.get('n_rendered') or 0)}"
        if n_cands:
            head += f" of {n_cands} offered"
        if picked:
            head += f" (picked #{', #'.join(str(p) for p in picked)})"
        # The conversations those facts came out of, which the picks nominated to the
        # past-chat channel. Named here rather than left to be spotted in the Debug
        # dump: this is the one retrieval on the box whose relevance was decided by a
        # fact rather than by a cosine, and the whole question an operator has about it
        # is whether the conversation it reached back into was the right one.
        sessions = [str(s) for s in (info.get("sessions") or []) if s]
        if sessions:
            head += (" · from " + ", ".join(s.replace(".json", "") for s in sessions[:3])
                     + (f" +{len(sessions) - 3}" if len(sessions) > 3 else ""))

        cursor = self.txt_chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        rule = QTextCharFormat()
        rule.setForeground(QColor(150, 150, 150))
        cursor.insertText(f"\n───── {head} ─────\n", rule)

        if skipped:
            note = self._FACTS_SKIP_TEXT.get(skipped, skipped)
            if info.get("error"):
                note += f": {info['error']}"
            muted = QTextCharFormat()
            # A failure is not a quiet outcome and must not be dressed as one — that
            # equivalence is exactly what hid a dead channel for a day.
            muted.setForeground(green if skipped in ("picked_nothing", "no_candidates",
                                                     "empty") else QColor(190, 40, 40))
            cursor.insertText(note + "\n", muted)
        elif text:
            body = QTextCharFormat()
            body.setForeground(green)
            cursor.insertText(text + "\n", body)

        cursor.insertText("─" * 20 + "\n\n", rule)
        # Leave a neutral format behind, or the reply inherits this block's colour.
        neutral = QTextCharFormat()
        neutral.setForeground(self.txt_chat_log.palette().text().color())
        cursor.setCharFormat(neutral)
        self.txt_chat_log.setTextCursor(cursor)
        self.txt_chat_log.ensureCursorVisible()

    def _on_debug_info(self, info: str) -> None:
        if self._streaming_assistant_active:
            self._pending_debug_lines.append(info)
            return
        self.txt_chat_log.append(f"[debug] {info}\n")

    def _on_generation_finished(self) -> None:
        self._streaming_assistant_active = False
        self._streaming_response_start_pos = None
        self._streaming_turn_start_pos = None
        self.btn_stop.setText("Stop")
        self.btn_stop.setEnabled(False)
        self._update_continue_button()
        self.txt_user_input.setFocus()
        if self._session_mode == "continued":
            self._refresh_sessions(select_filename=self._active_filename)

    # ---------------------------------------------------------------- #
    # Session list helpers                                              #
    # ---------------------------------------------------------------- #

    def _refresh_sessions(self, select_filename: Optional[str] = None) -> None:
        if not self._client.is_connected():
            return
        if self._sessions_worker is not None and self._sessions_worker.isRunning():
            return
        self._sessions_select_filename = select_filename
        self._sessions_worker = SessionsWorker(self._client)
        self._sessions_worker.sessions_ready.connect(self._on_sessions_ready)
        self._sessions_worker.error_occurred.connect(self._on_sessions_error)
        self._sessions_worker.start()

    def _session_list_label(self, meta: dict) -> str:
        first = meta.get("first_message", "").strip()
        if not first:
            first = meta.get("timestamp", meta.get("filename", ""))
        label = (first[:68] + "…") if len(first) > 68 else first
        user = meta.get("user", "").strip()
        if user:
            label = f"[{user}] {label}"
        continued_from = meta.get("continued_from", "").strip()
        if continued_from:
            label = f"↳ {label}"
        # Ava-initiated outreach: she opened this one — flag it so the user can tell it
        # apart and pick it up to reply. Split pending (only her opener, awaiting a
        # reply) from answered (the user replied, so ≥2 exchanges) with distinct icons.
        if meta.get("initiated_by", "").strip() == "ava":
            answered = int(meta.get("exchange_count", 0) or 0) > 1
            label = f"{'✅' if answered else '💬'} Ava: {label}"
        return label

    def _render_session_log(self, data: dict, header: str, footer: str = "") -> None:
        self.txt_chat_log.clear()
        self.txt_chat_log.append(header + "\n\n")
        session_user = (data.get("user") or "").strip()
        ava_initiated = str(data.get("initiated_by") or "").strip() == "ava"
        for i, ex in enumerate(data.get("exchanges", [])):
            user = ex.get("user_prompt", "")
            resp = ex.get("assistant_response", "")
            # An Ava-initiated session's exchange 0 is her opener; its user_prompt is the
            # synthetic "(initiative)" stimulus, so show only her message.
            if i == 0 and ava_initiated:
                self.txt_chat_log.append(f"Assistant: {resp}\n\n")
                continue
            speaker = (ex.get("speaker") or "").strip() or session_user or "You"
            self.txt_chat_log.append(f"{speaker}: {user}\n")
            self.txt_chat_log.append(f"Assistant: {resp}\n\n")
            feedback = ex.get("reflection_feedback") or {}
            if not isinstance(feedback, dict):
                feedback = {}
            feedback_text = str(feedback.get("text") or "").strip()
            if feedback_text:
                self.txt_chat_log.append(
                    f"[Meta feedback after this reply: {feedback_text}]\n\n"
                )
        if footer:
            self.txt_chat_log.append(footer + "\n\n")

    def _on_sessions_ready(self, sessions: list) -> None:
        select_filename = getattr(self, "_sessions_select_filename", None)
        self._sessions_select_filename = None
        # Keep the per-session metadata so selection can tell an Ava-initiated outreach
        # apart (it opens for reply in place rather than a read-only preview).
        self._session_meta_by_file = {s["filename"]: s for s in sessions}
        self.lst_sessions.clear()
        for s in reversed(sessions):  # most recent first
            item = QListWidgetItem(self._session_list_label(s))
            item.setData(Qt.ItemDataRole.UserRole, s["filename"])
            self.lst_sessions.addItem(item)
        if select_filename:
            for i in range(self.lst_sessions.count()):
                item = self.lst_sessions.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == select_filename:
                    self.lst_sessions.setCurrentItem(item)
                    break
        self._update_continue_button()

    def _on_sessions_error(self, _error: str) -> None:
        pass  # silently ignore — session list is non-critical

    def _on_session_selection_changed(self) -> None:
        self._update_continue_button()
        selected = self.lst_sessions.selectedItems()
        if len(selected) != 1:
            return
        filename = selected[0].data(Qt.ItemDataRole.UserRole)
        if not filename:
            return
        if self._session_mode == "continued" and filename == self._active_filename:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        # An Ava-initiated outreach chat opens for reply directly — adopted in place
        # (same file) so the user just types, no "Continue chat" step. Falls back to a
        # read-only preview if we can't adopt right now (not loaded / mid-generation).
        meta = self._session_meta_by_file.get(filename, {})
        generating = self.chat_worker is not None and self.chat_worker.isRunning()
        if (meta.get("initiated_by") == "ava" and self.model_loaded
                and self._client.is_connected() and not generating):
            self._open_outreach_in_place(filename)
            return
        self._start_session_preview(filename)

    def _start_session_preview(self, filename: str) -> None:
        if not self._client.is_connected() or not self.model_loaded:
            return
        if self.chat_worker is not None and self.chat_worker.isRunning():
            return
        self._session_mode = "preview"
        self._active_filename = filename
        self.conversation_history = []
        self._last_completed_turn_start_pos = None
        self._close_meta_window()
        self.txt_notes.clear()
        self.txt_chat_log.clear()
        self.txt_chat_log.append("Loading preview…\n")
        self._update_continue_button()

        self._preview_worker = PreviewSessionWorker(self._client, filename)
        self._preview_worker.session_ready.connect(self._on_preview_ready)
        self._preview_worker.error_occurred.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_ready(self, data: dict, filename: str) -> None:
        if self._session_mode != "preview" or self._active_filename != filename:
            return
        n = len(data.get("exchanges", []))
        self._render_session_log(
            data,
            f"Preview of {filename} ({n} exchange{'s' if n != 1 else ''}).",
            "Read-only — click \"Continue chat\" to restore this context on the server.",
        )
        self._update_continue_button()

    def _on_preview_error(self, error: str) -> None:
        if self._session_mode != "preview":
            return
        self.txt_chat_log.clear()
        self.txt_chat_log.append(f"✗ Failed to load preview: {error}\n")
        self._update_continue_button()

    def _update_continue_button(self) -> None:
        selected = self.lst_sessions.selectedItems()
        has_single = len(selected) == 1
        has_any = len(selected) >= 1
        generating = self.chat_worker is not None and self.chat_worker.isRunning()
        # No typing while previewing (read-only) or mid-adopt of an outreach session.
        input_blocked = self._session_mode in ("preview", "loading_outreach")
        deleting = self._delete_worker is not None and self._delete_worker.isRunning()
        resetting = (self._reflect_reset_worker is not None
                     and self._reflect_reset_worker.isRunning())
        meta_saving = self._meta_worker is not None and self._meta_worker.isRunning()
        retrying = self._retry_worker is not None and self._retry_worker.isRunning()
        self.lst_sessions.setEnabled(not generating and not meta_saving and not retrying)
        self.btn_continue.setEnabled(
            self.model_loaded and has_single and not generating and not meta_saving
            and not retrying
        )
        self.btn_delete_chat.setEnabled(
            self._client.is_connected() and has_any and not generating
            and not deleting and not meta_saving and not retrying and not resetting
        )
        # Re-reflect is a server-side file op like delete — it needs a connection but
        # not a loaded model, and must not race the other session-mutating workers.
        self.btn_reflect_again.setEnabled(
            self._client.is_connected() and has_any and not generating
            and not deleting and not meta_saving and not retrying and not resetting
        )
        self.btn_reflect_again.setText("Resetting…" if resetting else "Re-reflect chat")
        can_type = (self.model_loaded and not generating and not input_blocked
                    and not meta_saving and not retrying)
        self.btn_send.setEnabled(can_type)
        self.txt_user_input.setEnabled(can_type)
        self.btn_meta.setEnabled(can_type and bool(self._meta_exchange_id))
        # Retry is available once a completed live/continued turn is on the log (its
        # start position is tracked) and nothing else is busy. It rolls that turn off
        # the transcript and restores its prompt for a resend under adjusted sampling.
        self.btn_retry.setEnabled(
            can_type
            and self._session_mode in ("live", "continued")
            and self._last_completed_turn_start_pos is not None
        )
        self.btn_retry.setText("Retrying…" if retrying else "Retry")

    def _on_reflect_again(self) -> None:
        """Put the selected chat(s) back in the reflection backlog (delete the sidecar).

        The sidecar IS the reflection of a chat, so removing it is the whole action:
        the next Sleep run (or the background per-chat pass) re-reads the transcript
        under the current persona and re-derives the verdicts, targets, summary and
        anchors. The confirm names the one destructive part — a hand-repaired
        (locked) target lives in that same file and goes with it."""
        selected = self.lst_sessions.selectedItems()
        filenames = [it.data(Qt.ItemDataRole.UserRole) for it in selected]
        filenames = [f for f in filenames if f]
        if not filenames:
            return
        n = len(filenames)
        detail = filenames[0] if n == 1 else f"{n} chats"
        reply = QMessageBox.question(
            self,
            "Re-reflect chat",
            f"Reset the reflection of {detail}?\n\n"
            "The transcript is kept — but the chat's reflection sidecar is deleted, "
            "so it re-reflects from scratch on the next Sleep run. This drops its "
            "reflect-once freeze, per-exchange verdicts and trainable targets "
            "(including any human-locked Training-review repairs), consolidation "
            "summary and retrieval anchors. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.btn_reflect_again.setEnabled(False)
        self._reflect_reset_worker = ResetReflectionWorker(self._client, filenames)
        self._reflect_reset_worker.finished_resetting.connect(self._on_reflect_again_done)
        self._reflect_reset_worker.start()
        self._update_continue_button()

    def _on_reflect_again_done(self, reset: list, unreflected: list, failed: list) -> None:
        for fn in reset:
            self.txt_chat_log.append(
                f"[chat] {fn} will be reflected again — sidecar removed.\n")
        for fn in unreflected:
            self.txt_chat_log.append(
                f"[chat] {fn} had no reflection yet — already in the backlog.\n")
        for fn, message in failed:
            self.txt_chat_log.append(f"[chat] Could not reset {fn}: {message}\n")
        self._update_continue_button()

    def _on_delete_chat(self) -> None:
        selected = self.lst_sessions.selectedItems()
        filenames = [it.data(Qt.ItemDataRole.UserRole) for it in selected]
        filenames = [f for f in filenames if f]
        if not filenames:
            return
        n = len(filenames)
        detail = filenames[0] if n == 1 else f"{n} chats"
        reply = QMessageBox.question(
            self,
            "Delete chat",
            f"Permanently delete {detail}?\n\n"
            "This removes the transcript(s) from the server and drops them from "
            "memory retrieval. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Selecting an Ava-initiated outreach chat adopts it *in place* as the
        # active server-side conversation (the logger writes to that same file),
        # so the server refuses to delete it while it's active. Declining such a
        # chat is a legitimate delete — clear the server context first (what the
        # server's error instructs) so the logger releases the file, then delete.
        if (self._session_mode == "continued"
                and self._active_filename in filenames
                and self._session_meta_by_file.get(
                    self._active_filename, {}).get("initiated_by") == "ava"):
            if self._client.is_connected():
                self._client.clear_context(user=self._current_user())
            self._session_mode = "live"
            self._active_filename = None
            self.conversation_history = []
            self._last_completed_turn_start_pos = None
            self.txt_chat_log.clear()
        self.btn_delete_chat.setEnabled(False)
        self._delete_worker = DeleteSessionWorker(self._client, filenames)
        self._delete_worker.finished_deleting.connect(self._on_delete_done)
        self._delete_worker.start()

    def _on_delete_done(self, deleted: list, failed: list) -> None:
        for fn in deleted:
            self.txt_chat_log.append(f"[chat] Deleted {fn}.\n")
        for fn, message in failed:
            self.txt_chat_log.append(f"[chat] Could not delete {fn}: {message}\n")
        # If we were previewing one of the deleted transcripts, drop back to a
        # clean live view so we're not showing a file that no longer exists. A
        # *continued* conversation is left alone — its server-side context and new
        # session file are independent of the original transcript.
        if self._session_mode == "preview" and self._active_filename in deleted:
            self._session_mode = "live"
            self._active_filename = None
            self.conversation_history = []
            self.txt_chat_log.clear()
        self._refresh_sessions()
        self._update_continue_button()

    def _on_continue_chat(self) -> None:
        selected = self.lst_sessions.selectedItems()
        if len(selected) != 1:
            return
        item = selected[0]
        filename = item.data(Qt.ItemDataRole.UserRole)
        if not filename:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        self.btn_continue.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.txt_user_input.setEnabled(False)
        self._close_meta_window()
        self.txt_chat_log.clear()
        self.txt_chat_log.append("Restoring session context on server…\n")

        self._load_session_worker = LoadSessionWorker(self._client, filename)
        self._load_session_worker.session_loaded.connect(self._on_session_loaded)
        self._load_session_worker.error_occurred.connect(self._on_session_load_error)
        self._load_session_worker.start()

    def _on_session_loaded(self, data: dict, filename: str) -> None:
        self._session_mode = "continued"
        self._active_filename = filename
        self.conversation_history = []
        # Restored prior exchanges aren't retryable via this mechanism (Retry only rolls
        # back a turn generated live in this session); it re-arms on the next live reply.
        self._last_completed_turn_start_pos = None
        # Continuing creates a fresh server-side session file — notes don't carry over.
        self.txt_notes.clear()
        notes = (data.get("notes") or "").strip()
        n = len(data.get("exchanges", []))
        header = (
            f"Continuing {filename} ({n} exchange{'s' if n != 1 else ''} loaded into context)."
        )
        footer = (
            "Server context restored — your next message continues this conversation. "
            "New exchanges are saved to a separate session file."
        )
        if notes:
            footer = f"Original session notes (not carried over): {notes}\n\n" + footer
        self._render_session_log(data, header, footer)
        for ex in data.get("exchanges", []):
            user = ex.get("user_prompt", "")
            resp = ex.get("assistant_response", "")
            self.conversation_history.append({"role": "user", "content": user})
            self.conversation_history.append({"role": "assistant", "content": resp})
        self._update_continue_button()
        self.txt_user_input.setFocus()

    def _on_session_load_error(self, error: str) -> None:
        self._session_mode = "live"
        self._active_filename = None
        self.conversation_history = []
        self.txt_chat_log.clear()
        self.txt_chat_log.append(f"✗ Failed to continue session: {error}\n")
        self._update_continue_button()

    def _open_outreach_in_place(self, filename: str) -> None:
        """Adopt an Ava-initiated outreach session as the active conversation *in place*
        (same file), so the user can reply immediately without a "Continue chat" step."""
        if self._load_session_worker is not None and self._load_session_worker.isRunning():
            return
        self._session_mode = "loading_outreach"
        self._active_filename = filename
        self.conversation_history = []
        self._last_completed_turn_start_pos = None
        self._close_meta_window()
        self.txt_notes.clear()
        self.txt_chat_log.clear()
        self.txt_chat_log.append("Opening Ava's message…\n")
        self._update_continue_button()
        self._load_session_worker = LoadSessionWorker(self._client, filename, in_place=True)
        self._load_session_worker.session_loaded.connect(self._on_outreach_adopted)
        self._load_session_worker.error_occurred.connect(self._on_session_load_error)
        self._load_session_worker.start()

    def _on_outreach_adopted(self, data: dict, filename: str) -> None:
        """Ava's outreach session is now active in place; render it and enable reply.
        New exchanges append to this same file (one reflectable conversation)."""
        self._session_mode = "continued"
        self._active_filename = filename
        self.conversation_history = []
        self.txt_notes.clear()
        n = len(data.get("exchanges", []))
        header = f"Ava reached out ({filename})."
        footer = (
            "Type your reply below — it continues this same conversation, saved to this "
            "chat file."
        )
        self._render_session_log(data, header, footer)
        for ex in data.get("exchanges", []):
            self.conversation_history.append(
                {"role": "user", "content": ex.get("user_prompt", "")})
            self.conversation_history.append(
                {"role": "assistant", "content": ex.get("assistant_response", "")})
        self._update_continue_button()
        self.txt_user_input.setFocus()
