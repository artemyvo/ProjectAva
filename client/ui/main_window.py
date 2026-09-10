"""Main application window for Ava Chat."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QMainWindow, QStatusBar, QLabel, QTabWidget
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import QTimer, Qt

from ui.chat_widget import ChatWidget
from ui.chat_review_widget import ChatReviewWidget
from ui.sleep_widget import SleepWidget
from ui.prompt_widget import PromptWidget
from ui.activity_widget import ActivityWidget
from ui.worklog_widget import WorklogWidget
from ui.modules_widget import ModulesWidget
from ui.encounter_widget import EncounterWidget
from ui.gossip_widget import GossipWidget
from ui.debug_widget import DebugWidget
from ui.persona_widget import PersonaWidget
from ui.facts_widget import FactsWidget
from ui.migrate_widget import MigrateWidget
from ui.training_review_widget import TrainingReviewWidget

# File-based error logging
log_path = Path.home() / ".avachat" / "error.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_path),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = _log_uncaught_exception


class MainWindow(QMainWindow):
    """Top-level window for Ava Chat."""

    _MIN_TEXT_FONT_SIZE = 8
    _MAX_TEXT_FONT_SIZE = 24

    def __init__(self, server_url: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Ava Chat")
        self.setGeometry(100, 100, 960, 720)
        self.setMinimumSize(640, 480)

        self._text_font_size = 11

        self._create_menu()

        kwargs = {"server_url": server_url} if server_url else {}
        self.chat_widget = ChatWidget(self, **kwargs)
        self.chat_review_widget = ChatReviewWidget(self.chat_widget, self)
        self.sleep_widget = SleepWidget(self.chat_widget, self)
        self.prompt_widget = PromptWidget(self.chat_widget, self)
        self.activity_widget = ActivityWidget(self.chat_widget, self)
        self.worklog_widget = WorklogWidget(self.chat_widget, self)
        self.modules_widget = ModulesWidget(self.chat_widget, self)
        self.encounter_widget = EncounterWidget(self.chat_widget, self)
        self.gossip_widget = GossipWidget(self.chat_widget, self)
        self.debug_widget = DebugWidget(self.chat_widget, self)
        self.persona_widget = PersonaWidget(self.chat_widget, self)
        self.facts_widget = FactsWidget(self.chat_widget, self)
        self.migrate_widget = MigrateWidget(self.chat_widget, self)
        self.training_review_widget = TrainingReviewWidget(self.chat_widget, self)

        self._tabs = QTabWidget()
        self._tabs.addTab(self.chat_widget, "Chat")
        self._tabs.addTab(self.chat_review_widget, "Chat review")
        self._tabs.addTab(self.sleep_widget, "Sleep")
        self._tabs.addTab(self.prompt_widget, "Prompt")
        self._tabs.addTab(self.activity_widget, "Activity")
        self._tabs.addTab(self.worklog_widget, "Worklog")
        self._tabs.addTab(self.modules_widget, "Modules")
        self._tabs.addTab(self.encounter_widget, "Encounter")
        self._tabs.addTab(self.gossip_widget, "Gossip")
        self._tabs.addTab(self.debug_widget, "Debug")
        self._tabs.addTab(self.persona_widget, "Persona")
        self._tabs.addTab(self.facts_widget, "Facts")
        self._tabs.addTab(self.migrate_widget, "Migrate")
        self._tabs.addTab(self.training_review_widget, "Training review")
        self.setCentralWidget(self._tabs)

        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Sequential handoff: on connect the Sleep tab adopts any in-progress
        # reflection run; on intentional disconnect it detaches (stops polling).
        self.chat_widget.connected.connect(self.sleep_widget.on_connected)
        self.chat_widget.disconnected.connect(self.sleep_widget.on_disconnected)

        # The Prompt tab shows the LIVE standing prompt, which lives server-side and can
        # change under it (an experiment activated elsewhere, a restart) — so it reloads
        # from the server on connect rather than trusting anything remembered here.
        self.chat_widget.connected.connect(self.prompt_widget.on_connected)
        self.chat_widget.disconnected.connect(self.prompt_widget.on_disconnected)

        # The Activity tab polls the unified activity journal continuously (independent of
        # any run) so the box is never silent while it works; start/stop with the socket.
        self.chat_widget.connected.connect(self.activity_widget.on_connected)
        self.chat_widget.disconnected.connect(self.activity_widget.on_disconnected)

        # The Worklog tab previews Ava's first-person episodic worklog; like Activity it
        # polls live (independent of any run) and starts/stops with the socket.
        self.chat_widget.connected.connect(self.worklog_widget.on_connected)
        self.chat_widget.disconnected.connect(self.worklog_widget.on_disconnected)

        self._create_status_bar()

        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(2000)
        self._update_status_bar()

    # ---------------------------------------------------------------- #
    # Menu                                                              #
    # ---------------------------------------------------------------- #

    def _create_menu(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("Edit")

        inc_font = QAction("Increase Font Size", self)
        inc_font.setShortcut("Ctrl++")
        inc_font.triggered.connect(self._increase_font_size)
        edit_menu.addAction(inc_font)

        dec_font = QAction("Decrease Font Size", self)
        dec_font.setShortcut("Ctrl+-")
        dec_font.triggered.connect(self._decrease_font_size)
        edit_menu.addAction(dec_font)

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) == self.chat_review_widget:
            self.chat_review_widget.refresh()
        elif self._tabs.widget(index) == self.sleep_widget:
            self.sleep_widget.refresh_sessions()
        elif self._tabs.widget(index) == self.prompt_widget:
            self.prompt_widget.refresh()
        elif self._tabs.widget(index) == self.modules_widget:
            self.modules_widget.refresh()
        elif self._tabs.widget(index) == self.debug_widget:
            self.debug_widget.refresh_artifacts()
        elif self._tabs.widget(index) == self.persona_widget:
            self.persona_widget.refresh_items()
        elif self._tabs.widget(index) == self.facts_widget:
            self.facts_widget.refresh_items()
        elif self._tabs.widget(index) == self.training_review_widget:
            self.training_review_widget.refresh()

    # ---------------------------------------------------------------- #
    # Font management                                                   #
    # ---------------------------------------------------------------- #

    def _make_text_font(self) -> QFont:
        f = QFont("Courier")
        f.setPointSize(self._text_font_size)
        return f

    def _apply_fonts(self):
        font = self._make_text_font()
        self.chat_widget.update_fonts(font)
        self.chat_review_widget.update_fonts(font)
        self.sleep_widget.update_fonts(font)
        self.prompt_widget.update_fonts(font)
        self.encounter_widget.update_fonts(font)
        self.gossip_widget.update_fonts(font)
        self.debug_widget.update_fonts(font)
        self.persona_widget.update_fonts(font)
        self.facts_widget.update_fonts(font)
        self.migrate_widget.update_fonts(font)
        self.training_review_widget.update_fonts(font)
        self.activity_widget.update_fonts(font)
        self.worklog_widget.update_fonts(font)
        self.modules_widget.update_fonts(font)

    def _increase_font_size(self):
        if self._text_font_size < self._MAX_TEXT_FONT_SIZE:
            self._text_font_size += 1
            self._apply_fonts()

    def _decrease_font_size(self):
        if self._text_font_size > self._MIN_TEXT_FONT_SIZE:
            self._text_font_size -= 1
            self._apply_fonts()

    # ---------------------------------------------------------------- #
    # Status bar                                                        #
    # ---------------------------------------------------------------- #

    def _create_status_bar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._ctx_label = QLabel("")
        self._activity_label = QLabel("")   # live autonomous-activity chip
        self._activity_label.setStyleSheet("font-weight: bold;")
        self._tokens_label = QLabel("")
        self._mem_label = QLabel("checking...")
        self._model_label = QLabel("Model: not loaded")

        bar.addWidget(self._ctx_label)
        bar.addPermanentWidget(self._activity_label)
        bar.addPermanentWidget(self._tokens_label)
        bar.addPermanentWidget(self._mem_label)
        bar.addPermanentWidget(self._model_label)

    def _update_status_bar(self):
        # Memory (delegated to backend — VRAM for CUDA)
        try:
            self._mem_label.setText(self.chat_widget.get_memory_status())
        except Exception:
            self._mem_label.setText("memory: unavailable")

        # Connection + model state
        try:
            cw = self.chat_widget
            if not cw.is_connected:
                self._model_label.setText("disconnected")
            elif cw.model_loaded:
                self._model_label.setText("remote · loaded")
            else:
                self._model_label.setText("remote · no model")
        except Exception:
            self._model_label.setText("disconnected")

        # Live autonomous-activity chip (driven by the Activity tab's continuous poll of
        # the unified activity journal), so what Ava is doing on the box is visible from
        # any tab. Blank when the box is idle.
        try:
            self._activity_label.setText(self.activity_widget.current_activity_text())
        except Exception:
            self._activity_label.setText("")

        # User-token imprint meter: accumulated (earned) vs consumed by wanders.
        try:
            cw = self.chat_widget
            econ = cw.get_token_economy() if cw.is_connected else None
            if econ:
                acc = int(econ.get("accumulated", 0))
                consumed = int(econ.get("consumed", 0))
                self._tokens_label.setText(
                    f"User tokens: {acc:,} acc / {consumed:,} used"
                )
            else:
                self._tokens_label.setText("")
        except Exception:
            self._tokens_label.setText("")

        # Context usage + generation speed (only while a model is loaded)
        try:
            cw = self.chat_widget
            if cw.model_loaded and cw.conversation_history:
                used, total, pct = cw.get_context_usage()
                text = f"Context: {used}/{total} tokens ({pct}%)"
                tps = cw.get_generation_speed()
                if tps:
                    text += f"  ·  {tps:.1f} tok/s"
                self._ctx_label.setText(text)
            else:
                self._ctx_label.setText("")
        except Exception:
            self._ctx_label.setText("")

    # ---------------------------------------------------------------- #
    # Window lifecycle                                                   #
    # ---------------------------------------------------------------- #

    def closeEvent(self, event):
        self.chat_widget.save_config()
        event.accept()

