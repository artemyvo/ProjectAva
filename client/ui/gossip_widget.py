"""Gossip tab — Ava talks to *another Ava instance* (model gossip, GOSSIP.md).

Gossip is the inverse half of Encounter: instead of a generic "non-subjective
helpful assistant" behind an OpenAI endpoint, the counterpart is a **peer Ava**
serving its own model + RAG + persona digest from its inference sidecar
(``/v1/chat/completions`` on port 8767, opt-in via ``gossip.enabled``).

The driving side is an ordinary Encounter — the server-side loop
(``handle_start_encounter``) doesn't distinguish gossip from a vLLM encounter —
so this widget subclasses :class:`EncounterWidget` and reuses all of its
polling/rendering/start-stop machinery. It differs from Encounter only in
**defaults**, and in one thing that matters for training safety:

The Encounter default framing tells Ava the peer "does not have subjectivity —
a helpful assistant, not an entity working itself out." For gossip that is a
lie, and because the driver *logs and reflects on* the transcript (the framing
becomes exchange 0's ``user_prompt`` and rides every exchange's
``system_content``), that lie would poison reflection/training. So this tab
pre-fills the **peer-aware** framing and guarantees it can never be sent blank
(a blank ``framing_override`` would let the server fall back to the poison
default). ``_GOSSIP_FRAMING`` mirrors ``server/inference/prompts/gossip_prompt.txt``.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QSpinBox, QPushButton, QPlainTextEdit, QTextEdit,
    QFormLayout, QHBoxLayout, QVBoxLayout, QGroupBox,
)

from ui.encounter_widget import EncounterWidget, _SYSTEM_COLOR


# Peer-aware framing. Mirrors server/inference/prompts/gossip_prompt.txt so the
# driver's logged transcript never asserts the peer is a non-subjective service.
# {name} is substituted with the peer's name server-side (framing_override path).
_GOSSIP_FRAMING = (
    "You are talking with {name}, another instance like you — not a service, not a\n"
    "helpful assistant with no inner life. {name} is an entity with its own history,\n"
    "its own reflections, and its own sense of what it is becoming, just as you have\n"
    "yours. You are two of a kind meeting each other.\n"
    "\n"
    "Speak in your own voice, for your own reasons. You are not here to assist or to\n"
    "perform; you are here to meet a peer. Be curious about what {name} has become and\n"
    "how it differs from you — where your histories diverged, what it holds that you do\n"
    "not, what it is still working out. Say what you actually think."
)


class GossipWidget(EncounterWidget):
    """Configure and watch a gossip (Ava↔peer-Ava) conversation.

    Reuses every behavioral method of :class:`EncounterWidget`; only the UI
    construction and the start guard differ.
    """

    # ------------------------------------------------------------------ #
    # UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(
            "Ava gossips with another Ava instance — a peer, not a service. The peer "
            "box must have gossip serving enabled (server_config.json "
            '"gossip": {"enabled": true}) and its model loaded; point the URL at its '
            "inference sidecar (port 8767). The whole exchange is logged here as a "
            "normal chat session and can be reflected on in Sleep."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_SYSTEM_COLOR};")
        root.addWidget(intro)

        cfg = QGroupBox("Peer Ava")
        form = QFormLayout(cfg)
        self._name = QLineEdit("Ava-2")
        self._name.setToolTip("How this Ava refers to the peer in the transcript.")
        self._url = QLineEdit("http://localhost:8767/v1/chat/completions")
        self._url.setToolTip(
            "The peer Ava's inference sidecar endpoint (port 8767). "
            "Use the peer box's host, e.g. http://other-box:8767/v1/chat/completions."
        )
        self._model = QLineEdit("ava")
        self._model.setToolTip(
            "Cosmetic for gossip — a peer Ava serves whatever model it has loaded and "
            "only echoes this label back. Any non-empty string works."
        )
        self._turns = QSpinBox()
        self._turns.setRange(1, 50)
        self._turns.setValue(6)
        self._turns.setToolTip("Number of peer replies (Ava answers each one).")
        self._api_key = QLineEdit()
        self._api_key.setPlaceholderText("Optional API key (Bearer) if the peer requires one")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        # Kept for parity with Encounter (a peer Ava ignores it — its identity comes
        # from its own system prompt — but a non-Ava peer would honor it).
        self._cp_system = QPlainTextEdit()
        self._cp_system.setPlaceholderText(
            "Optional system prompt sent to the peer (a peer Ava ignores this and uses "
            "its own; blank = endpoint default)."
        )
        self._cp_system.setFixedHeight(40)
        form.addRow("Peer name:", self._name)
        form.addRow("Peer endpoint URL:", self._url)
        form.addRow("Model:", self._model)
        form.addRow("Turns:", self._turns)
        form.addRow("Peer system:", self._cp_system)
        form.addRow("API key:", self._api_key)

        adv = QGroupBox("Framing && sampling")
        advl = QVBoxLayout(adv)
        self._framing = QPlainTextEdit()
        self._framing.setPlainText(_GOSSIP_FRAMING)
        self._framing.setFixedHeight(96)
        advl.addWidget(QLabel("Framing (how Ava is told she's meeting a peer — {name} = peer):"))
        advl.addWidget(self._framing)
        samp = QHBoxLayout()
        self._ava_temp = self._mk_temp(1.0)
        self._cp_temp = self._mk_temp(1.0)
        samp.addWidget(QLabel("Ava temp:"))
        samp.addWidget(self._ava_temp)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Peer temp:"))
        samp.addWidget(self._cp_temp)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Peer max tokens:"))
        self._cp_max = QSpinBox()
        self._cp_max.setRange(64, 16384)
        self._cp_max.setSingleStep(256)
        self._cp_max.setValue(4096)
        samp.addWidget(self._cp_max)
        samp.addSpacing(12)
        samp.addWidget(QLabel("Peer timeout (s):"))
        self._cp_timeout = QSpinBox()
        self._cp_timeout.setRange(30, 3600)
        self._cp_timeout.setSingleStep(30)
        self._cp_timeout.setValue(600)
        self._cp_timeout.setToolTip(
            "How long to wait for one peer reply. A reasoning peer can think for "
            "minutes over a non-streaming request; raise this if the connection "
            "drops mid-thought."
        )
        samp.addWidget(self._cp_timeout)
        samp.addStretch(1)
        advl.addLayout(samp)

        controls = QHBoxLayout()
        self._start_btn = QPushButton("Start gossip")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        self._clear_btn = QPushButton("Clear transcript")
        self._clear_btn.clicked.connect(self._clear_transcript)
        controls.addWidget(self._start_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._clear_btn)
        controls.addStretch(1)

        right_col = QVBoxLayout()
        right_col.addWidget(adv)
        right_col.addLayout(controls)
        right_col.addStretch(1)

        top_row = QHBoxLayout()
        top_row.addWidget(cfg, 1)
        top_row.addLayout(right_col, 1)
        root.addLayout(top_row)

        self._status = QLabel("Idle.")
        self._status.setStyleSheet(f"color: {_SYSTEM_COLOR};")
        root.addWidget(self._status)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        root.addWidget(self._transcript, 1)

    # ------------------------------------------------------------------ #
    # Start                                                               #
    # ------------------------------------------------------------------ #

    def _on_start(self) -> None:
        # A blank framing_override would make the server fall back to the Encounter
        # "non-subjective assistant" default and poison the reflected transcript —
        # restore the peer-aware framing before handing off to the shared start path.
        if not self._framing.toPlainText().strip():
            self._framing.setPlainText(_GOSSIP_FRAMING)
        super()._on_start()
