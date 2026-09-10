"""Chat review tab — a past chat rendered with everything the model actually saw.

The Chat tab's read-only preview shows a transcript the way a *reader* sees it (speaker
line, reply line). This tab is the same transcript the way the *model* saw it: for every
exchange, the exact assembled system message (base prompt + identity/temporal anchors +
the injected RAG block + any first-turn surfaced questions — logged verbatim per turn as
``system_content``, because it cannot be reconstructed later), the raw retrieved
``rag_context``, the thinking block (``assistant_cot``) next to the answer, and the
per-turn bookkeeping: generation params, input tokens, the tension summary + its
contested-token trace, meta feedback, corruption flags, rewrite history.

It loosely mimics the Chat tab's shape (chat list on the left, conversation on the
right) but chats nowhere: read-only, and it never touches server state beyond the two
read RPCs (``list_sessions`` / ``get_session``). Section toggles let the operator collapse
the bulky parts (the system message repeats near-verbatim every turn) without a refetch —
the fetched session dict is cached and re-rendered locally.

Scope note: this renders the **transcript** (``data/hot/chats/<ts>.json``). The reflection
sidecar (``<ts>.state.json`` — verdicts and trainable targets) is a separate artifact; the
Training review tab is the surface for that.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QListWidget, QListWidgetItem, QSplitter, QCheckBox, QLineEdit,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# The list/fetch workers are generic read-only RPC threads; reuse them rather than
# cloning (the Chat tab owns them, and never imports this module — no cycle).
from ui.chat_widget import SessionsWorker, PreviewSessionWorker

_RULE = "═" * 78
_SUB = "─" * 78


def _fmt_float(v, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


class ChatReviewWidget(QWidget):
    """Read-only inspection of one past chat, with every per-turn internal shown."""

    def __init__(self, chat_widget, parent=None) -> None:
        super().__init__(parent)
        self._chat_widget = chat_widget
        self._sessions_worker: Optional[SessionsWorker] = None
        self._fetch_worker: Optional[PreviewSessionWorker] = None
        self._sessions: list[dict] = []          # raw sessions_list metadata
        self._data: Optional[dict] = None        # currently rendered session JSON
        self._filename: str = ""                 # ...and its filename
        self.text_font = QFont("Courier")
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._lbl = QLabel("Chat review — a past chat with everything the model saw.")
        header.addWidget(self._lbl, 1)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setToolTip("Re-fetch the chat list (and the open chat) from "
                                     "the server.")
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: the chat list (+ a substring filter — the corpus grows into the hundreds).
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.addWidget(QLabel("Chats"))
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("filter…")
        self._txt_filter.setToolTip("Show only chats whose list label contains this text "
                                    "(matches the speaker, the opening message, the "
                                    "timestamp).")
        self._txt_filter.textChanged.connect(self._populate_list)
        left_layout.addWidget(self._txt_filter)
        self._lst = QListWidget()
        self._lst.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._lst, 1)
        split.addWidget(left)

        # Right: section toggles + the rendered transcript.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Show:"))
        self._chk_system = QCheckBox("System message")
        self._chk_system.setToolTip(
            "The exact assembled system message for each turn (base prompt + anchors + "
            "injected RAG + surfaced questions), as logged verbatim at generation time. "
            "It differs turn to turn (the temporal anchor and the retrieved passages "
            "move), so it is shown in full; a turn that repeats the previous message "
            "byte for byte collapses to a back-reference. This is also the bulkiest "
            "section — untick it to read the conversation itself."
        )
        self._chk_rag = QCheckBox("RAG block")
        self._chk_rag.setToolTip(
            "The raw retrieved context for the turn (rag_context). The system message "
            "already embeds this exact block, so while that section is shown this one "
            "collapses to a pointer rather than repeating it — untick \"System message\" "
            "to read the retrieved block on its own."
        )
        self._chk_cot = QCheckBox("CoT")
        self._chk_cot.setToolTip("Ava's thinking block (assistant_cot) for each reply.")
        self._chk_tension = QCheckBox("Tension")
        self._chk_tension.setToolTip(
            "Per-segment tension stats (entropy / margin / contested fraction) and the "
            "contested-token trace with the road-not-taken alternative."
        )
        self._chk_meta = QCheckBox("Params + meta")
        self._chk_meta.setToolTip(
            "Sampling params, input tokens, think-open probability, meta feedback, "
            "corruption flags, and rewrite history."
        )
        for chk in (self._chk_system, self._chk_rag, self._chk_cot,
                    self._chk_tension, self._chk_meta):
            chk.setChecked(True)
            chk.toggled.connect(self._rerender)
            toggles.addWidget(chk)
        toggles.addStretch()
        right_layout.addLayout(toggles)

        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(self.text_font)
        self._out.setPlaceholderText(
            "Select a chat on the left to review it.\n\n"
            "Read-only: this shows the transcript exactly as logged — the assembled "
            "system message with its injected RAG, the chain of thought, the reply, and "
            "the per-turn bookkeeping. Nothing here is sent to the model."
        )
        right_layout.addWidget(self._out, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 700])
        layout.addWidget(split, 1)

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self._out.setFont(font)

    # ── chat list ──────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-fetch the chat list (called on tab open and from the Refresh button)."""
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl.setText("Chat review — not connected (connect in the Chat tab).")
            return
        if self._sessions_worker is not None and self._sessions_worker.isRunning():
            return
        self._lbl.setText("Chat review — loading chat list…")
        worker = SessionsWorker(client)
        worker.sessions_ready.connect(self._on_sessions_ready)
        worker.error_occurred.connect(self._on_sessions_error)
        self._sessions_worker = worker
        worker.start()

    def _on_sessions_ready(self, sessions: list) -> None:
        self._sessions = list(sessions)
        self._populate_list()
        self._lbl.setText(
            f"Chat review — {len(self._sessions)} chat(s). "
            "The chat open in the Chat tab is not listed until it is closed."
        )

    def _on_sessions_error(self, error: str) -> None:
        self._lbl.setText(f"Chat review — could not list chats: {error}")

    def _populate_list(self) -> None:
        """Rebuild the list widget from the cached metadata, honouring the filter."""
        needle = self._txt_filter.text().strip().lower()
        previous = self._filename
        self._lst.blockSignals(True)
        self._lst.clear()
        for meta in reversed(self._sessions):     # most recent first, as in the Chat tab
            label = self._label_for(meta)
            if needle and needle not in label.lower():
                continue
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, meta.get("filename", ""))
            self._lst.addItem(item)
        self._lst.blockSignals(False)
        # Keep the open chat selected across a refresh / filter change when it survives.
        if previous:
            for i in range(self._lst.count()):
                if self._lst.item(i).data(Qt.ItemDataRole.UserRole) == previous:
                    self._lst.setCurrentRow(i)
                    break

    @staticmethod
    def _label_for(meta: dict) -> str:
        """List label: timestamp + speaker + opening message (Ava-initiated flagged)."""
        stem = str(meta.get("filename", "")).replace(".json", "")
        first = str(meta.get("first_message", "")).strip().replace("\n", " ")
        if len(first) > 60:
            first = first[:60] + "…"
        user = str(meta.get("user", "")).strip()
        label = f"{stem}  [{user}] {first}" if user else f"{stem}  {first}"
        if str(meta.get("continued_from", "")).strip():
            label = f"↳ {label}"
        if str(meta.get("initiated_by", "")).strip() == "ava":
            answered = int(meta.get("exchange_count", 0) or 0) > 1
            label = f"{'✅' if answered else '💬'} {label}"
        return label

    # ── fetch one chat ─────────────────────────────────────────────────────────
    def _on_selection_changed(self) -> None:
        items = self._lst.selectedItems()
        if not items:
            return
        filename = items[0].data(Qt.ItemDataRole.UserRole)
        if not filename:
            return
        client = self._chat_widget._client
        if not client.is_connected():
            self._out.setPlainText("Not connected — connect in the Chat tab first.")
            return
        if self._fetch_worker is not None and self._fetch_worker.isRunning():
            return
        self._filename = filename
        self._data = None
        self._out.setPlainText(f"Loading {filename}…")
        worker = PreviewSessionWorker(client, filename)
        worker.session_ready.connect(self._on_session_ready)
        worker.error_occurred.connect(self._on_session_error)
        self._fetch_worker = worker
        worker.start()

    def _on_session_ready(self, data: dict, filename: str) -> None:
        if filename != self._filename:
            return                      # a newer selection won the race
        self._data = data
        self._rerender()

    def _on_session_error(self, error: str) -> None:
        self._data = None
        self._out.setPlainText(f"✗ Could not load {self._filename}: {error}")

    # ── rendering ──────────────────────────────────────────────────────────────
    def _rerender(self) -> None:
        """Re-render the cached session (toggles change the view, never refetch)."""
        if self._data is None:
            return
        self._out.setPlainText(self._render(self._data, self._filename))
        self._out.verticalScrollBar().setValue(0)

    def _render(self, data: dict, filename: str) -> str:
        lines: list[str] = []
        lines.extend(self._render_header(data, filename))

        exchanges = data.get("exchanges") or []
        ava_initiated = str(data.get("initiated_by") or "").strip() == "ava"
        session_user = str(data.get("user") or "").strip()
        prev_system = ""
        for i, ex in enumerate(exchanges):
            lines.extend(self._render_exchange(
                ex, i, len(exchanges),
                session_user=session_user,
                opener=(i == 0 and ava_initiated),
                prev_system=prev_system,
            ))
            prev_system = str(ex.get("system_content") or "")
        if not exchanges:
            lines.append("(this chat has no exchanges)")
        return "\n".join(lines)

    def _render_header(self, data: dict, filename: str) -> list[str]:
        exchanges = data.get("exchanges") or []
        rows = [
            ("file", filename),
            ("timestamp", data.get("timestamp", "")),
            ("user", data.get("user", "")),
            ("model", data.get("model_id", "")),
            ("adapter", data.get("adapter_id") or "base (no adapter)"),
            ("schema", f"v{data.get('schema_version', 1)}"),
            ("exchanges", str(len(exchanges))),
        ]
        for key, label in (("continued_from", "continued from"),
                           ("initiated_by", "initiated by"),
                           ("interlocutor", "interlocutor"),
                           ("notes", "notes")):
            value = str(data.get(key) or "").strip()
            if value:
                rows.append((label, value))

        lines = [_RULE, f"SESSION  {filename}", _RULE]
        width = max(len(k) for k, _ in rows)
        for key, value in rows:
            lines.append(f"  {key.ljust(width)} : {value}")

        ask = data.get("initiated_ask") or {}
        if isinstance(ask, dict) and ask.get("key"):
            lines.append(f"  {'raised ask'.ljust(width)} : "
                         f"[ask:{ask.get('ask_kind') or '?'}] {ask.get('content') or ''}")
            lines.append(f"  {''.ljust(width)}   (key {ask.get('key')})")

        if self._chk_system.isChecked():
            base = str(data.get("system_prompt") or "").strip()
            lines.append("")
            lines.append("── session system_prompt (the base prompt, before per-turn "
                         "assembly) ──")
            lines.append(base or "(none recorded)")
        lines.append("")
        return lines

    def _render_exchange(
        self,
        ex: dict,
        index: int,
        total: int,
        *,
        session_user: str,
        opener: bool,
        prev_system: str,
    ) -> list[str]:
        speaker = str(ex.get("speaker") or "").strip() or session_user or "user"
        lines = [_SUB, f"EXCHANGE {index + 1}/{total}    id {ex.get('exchange_id', '—')}"]

        if self._chk_meta.isChecked():
            lines.extend(self._render_meta(ex))
        flags = self._flags(ex)
        if flags:
            lines.append(f"  ⚑ {flags}")
        lines.append("")

        if self._chk_system.isChecked():
            system = str(ex.get("system_content") or "")
            lines.append("▼ system message (exact — base prompt + anchors + injected RAG "
                         "+ surfaced questions)")
            if not system:
                lines.append("(not recorded — pre-v2 transcript)")
            elif system == prev_system:
                lines.append(f"(identical to exchange {index}) — "
                             f"{len(system)} chars")
            else:
                lines.append(system)
            lines.append("")

        if self._chk_rag.isChecked():
            rag = str(ex.get("rag_context") or "").strip()
            system = str(ex.get("system_content") or "")
            lines.append("▼ retrieved RAG context (rag_context)")
            if not rag:
                lines.append("(nothing retrieved for this turn)")
            elif self._chk_system.isChecked() and rag and rag in system:
                # The assembled system message EMBEDS this exact block, so printing it
                # again reads as a double injection when it is one. Point at it instead;
                # untick "System message" to read the retrieved block on its own.
                lines.append(f"(the same {len(rag)} chars are embedded in the system "
                             f"message above — injected once, not twice; untick "
                             f"\"System message\" to read the block on its own)")
            else:
                lines.append(rag)
            lines.append("")

        if opener:
            # An Ava-initiated session's exchange 0 has no real user turn: its
            # user_prompt is the synthetic impulse that prompted her to open. Shown
            # (it is exactly the kind of internal this tab exists for), but labelled
            # so it is never mistaken for something the user said.
            lines.append(f"▶ synthetic impulse, logged under {speaker} — no user turn "
                         f"(masked from training)")
        else:
            lines.append(f"▶ {speaker}:")
        lines.append(str(ex.get("user_prompt") or "").strip() or "(empty)")
        lines.append("")

        if self._chk_cot.isChecked():
            cot = str(ex.get("assistant_cot") or "").strip()
            lines.append("▶ CoT (assistant_cot):")
            lines.append(cot or "(no thinking block)")
            lines.append("")

        lines.append("▶ Ava:")
        lines.append(str(ex.get("assistant_response") or "").strip() or "(empty)")
        lines.append("")

        if self._chk_tension.isChecked():
            lines.extend(self._render_tension(ex.get("tension")))

        if self._chk_meta.isChecked():
            lines.extend(self._render_feedback(ex))
            lines.extend(self._render_rewrites(ex))
        return lines

    @staticmethod
    def _render_meta(ex: dict) -> list[str]:
        params = ex.get("generation_params") or {}
        bits = []
        if params:
            # The spin boxes hand back binary-float noise (1.2000000000000002); the
            # operator cares about the setting, not its representation.
            bits.append(
                f"temp {_fmt_float(params.get('temperature'), 2)} · "
                f"top_p {_fmt_float(params.get('top_p'), 2)} · "
                f"max_new {params.get('max_new_tokens_setting', '—')}"
            )
        if ex.get("input_tokens") is not None:
            bits.append(f"input {ex['input_tokens']} tok")
        if ex.get("think_open_prob") is not None:
            bits.append(f"p(open think) {_fmt_float(ex['think_open_prob'])}")
        return [f"  {'  |  '.join(bits)}"] if bits else []

    @staticmethod
    def _flags(ex: dict) -> str:
        flags = []
        if ex.get("corrupt_cot"):
            flags.append("CoT flagged corrupt")
        if ex.get("corrupt_response"):
            flags.append("reply flagged corrupt")
        history = ex.get("rewrite_history") or []
        if history:
            flags.append(f"rewritten from a reviewed target ({len(history)} prior "
                         f"version(s) kept)")
        return " · ".join(flags)

    def _render_tension(self, tension) -> list[str]:
        if not isinstance(tension, dict) or not tension:
            return ["▸ tension: (not captured for this exchange)", ""]
        lines = ["▸ tension"]
        for name in ("cot", "answer"):
            seg = tension.get(name)
            if not isinstance(seg, dict):
                lines.append(f"    {name:<7}: (none)")
                continue
            lines.append(
                f"    {name:<7}: n={seg.get('n_tokens', '—')}  "
                f"peak_H {_fmt_float(seg.get('peak_entropy'))}  "
                f"med_H {_fmt_float(seg.get('median_entropy'))}  "
                f"med_margin {_fmt_float(seg.get('median_margin'))}  "
                f"p10_margin {_fmt_float(seg.get('p10_margin'))}  "
                f"contested {_fmt_pct(seg.get('contested_frac'))}"
            )
            for row in (seg.get("contested") or []):
                lines.append(
                    f"        pos {row.get('position', '—')}: "
                    f"{row.get('token', '')!r} ↔ {row.get('alt_token', '')!r}  "
                    f"(margin {_fmt_float(row.get('margin'))}, "
                    f"H {_fmt_float(row.get('entropy'))})"
                )
        n_tokens = len(tension.get("token_ids") or [])
        if n_tokens:
            lines.append(f"    series : {n_tokens} token ids + entropies/margins "
                         f"retained (branch-replay prefix)")
        lines.append("")
        return lines

    @staticmethod
    def _render_feedback(ex: dict) -> list[str]:
        feedback = ex.get("reflection_feedback")
        if not isinstance(feedback, dict):
            return []
        text = str(feedback.get("text") or "").strip()
        if not text:
            return []
        who = str(feedback.get("speaker") or "").strip()
        head = f"▸ meta feedback{f' from {who}' if who else ''}:"
        return [head, f"    {text}", ""]

    @staticmethod
    def _render_rewrites(ex: dict) -> list[str]:
        history = ex.get("rewrite_history") or []
        if not isinstance(history, list) or not history:
            return []
        lines = [f"▸ rewrite history ({len(history)} prior version(s))"]
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            when = str(entry.get("at") or "").strip() or "unknown time"
            kind = str(entry.get("target_kind") or "").strip()
            run = str(entry.get("run_id") or "").strip()
            head = f"    [{i}] {when}"
            if kind:
                head += f"  ({kind})"
            if run:
                head += f"  run {run}"
            lines.append(head)
            if entry.get("had_tension"):
                lines.append("        (the original tension block was dropped by the "
                             "rewrite — its token ids no longer match)")
            prior_cot = str(entry.get("prev_cot") or "").strip()
            prior_reply = str(entry.get("prev_response") or "").strip()
            if prior_cot:
                lines.append(f"        prior CoT  : {prior_cot}")
            if prior_reply:
                lines.append(f"        prior reply: {prior_reply}")
        lines.append("")
        return lines
