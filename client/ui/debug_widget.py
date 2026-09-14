"""Debug widget — inspect Ava's live reflection RAG artifacts.

A read-only view over everything the server currently recalls from the
reflection memory op-log: ``[fact]`` truths, ``[persona]`` self-statements, and
open ``[ask]`` questions. The server folds ``rag_memory.jsonl`` into its live
item set and returns it; this panel renders that set grouped by type. It also
shows the **wander log** — the pages Ava wandered into (title + link), newest
first — as its own category. Read-only — it never mutates server state.
"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QLabel,
    QMessageBox,
    QInputDialog,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont

if TYPE_CHECKING:
    from ui.chat_widget import ChatWidget
    from core.backend_client import BackendClient


# Display order + labels for the artifact kinds the reflection store emits.
_KIND_ORDER = ["persona", "fact", "ask", "recollection", "impression", "self_impression"]
_KIND_LABELS = {
    "persona": "[persona]",
    "fact": "[fact]",
    "ask": "[ask]",
    # Her reading of a PERSON — folded into that person's standing portrait.
    "impression": "[impression]",
    # Her reading of HERSELF from the outside: what a transcript shows to a reader with
    # no access to the <think> she wrote it from. Folded into the outside-view portrait
    # rendered above; not retrieved in chat by default. See core.self_portrait.
    "self_impression": "[self_impression]",
    # What Ava now makes of a re-read conversation, written by a revisit pass. Listed
    # last because it is the newest kind and the one an operator is checking rather
    # than living with — see ReflectionWriter.write_recollection.
    "recollection": "[recollection]",
}

# Known base models offered (editable) in the post-wipe model picker. The current
# server model is prepended at runtime; the field stays free-text for any HF id.
_KNOWN_BASE_MODELS = [
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3.6-27B",
    "unsloth/gemma-4-31B-it",
]


class RagArtifactsWorker(QThread):
    """Fetches the live RAG artifact set from the server off the GUI thread."""

    artifacts_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient"):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.get_rag_artifacts()
        if result.get("type") == "rag_artifacts":
            self.artifacts_ready.emit(result)
        else:
            self.error_occurred.emit(result.get("message", "Failed to fetch RAG artifacts"))


class TokenStatsWorker(QThread):
    """Fetches the running token-economy counters from the server off the GUI thread."""

    stats_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient"):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.get_token_stats()
        if result.get("type") == "token_stats":
            self.stats_ready.emit(result)
        else:
            self.error_occurred.emit(result.get("message", "Failed to fetch token stats"))


class WanderLogWorker(QThread):
    """Fetches the log of pages Ava wandered into off the GUI thread."""

    log_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient"):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.get_wander_log()
        if result.get("type") == "wander_log":
            self.log_ready.emit(result.get("entries", []))
        else:
            self.error_occurred.emit(result.get("message", "Failed to fetch wander log"))


class PromptDeltaWorker(QThread):
    """Fetches the logged-only prompt-mutation proposals off the GUI thread."""

    deltas_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient"):
        super().__init__()
        self._client = client

    def run(self) -> None:
        result = self._client.get_prompt_deltas()
        if result.get("type") == "prompt_deltas":
            self.deltas_ready.emit(result.get("deltas", []))
        else:
            self.error_occurred.emit(result.get("message", "Failed to fetch prompt deltas"))


class WipeWorker(QThread):
    """Runs the destructive watchdog state wipe off the GUI thread.

    The wipe stops the inference server, deletes regenerable state (and optionally
    chats + manifest), repoints the base model, and relaunches. Synchronous on the
    server, so this thread blocks until the relaunch is issued.
    """

    wipe_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", wipe_chats: bool, model_id: str):
        super().__init__()
        self._client = client
        self._wipe_chats = wipe_chats
        self._model_id = model_id

    def run(self) -> None:
        result = self._client.wipe_state(self._wipe_chats, self._model_id)
        if result.get("error"):
            self.error_occurred.emit(str(result.get("error")))
        else:
            self.wipe_done.emit(result)


class DedupFactsWorker(QThread):
    """Runs the clean-base semantic fact-dedup off the GUI thread, streaming progress.

    Slow: the server swaps the adapter out (two full model reloads) inside a
    CleanBaseSession, then groups the store one subject block at a time — dozens of
    sequential calls on a real corpus. ``progress`` carries the ``dedup_stage`` events
    (subject blocking, then one per block) so a panel can show it live; ``dedup_done`` the
    terminal ``facts_deduped`` payload (which echoes ``dry_run`` so the caller can branch).
    ``dry_run`` previews the proposed merges without writing; otherwise it evicts the
    duplicates and reloads RAG.
    """

    progress = pyqtSignal(dict)        # dedup_stage events (clustered / per-block)
    dedup_done = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, client: "BackendClient", dry_run: bool):
        super().__init__()
        self._client = client
        self._dry_run = dry_run

    def run(self) -> None:
        try:
            for kind, msg in self._client.dedup_facts(self._dry_run):
                if kind == "dedup_stage":
                    self.progress.emit(msg)
                elif kind == "facts_deduped":
                    self.dedup_done.emit(msg)
                else:  # error / connection_error
                    self.error_occurred.emit(msg.get("message", "Dedup failed"))
        except Exception as e:  # noqa: BLE001 — surface any RPC failure to the panel
            self.error_occurred.emit(str(e))


class DebugWidget(QWidget):
    """Debug tab — dumps all active reflection RAG artifacts into a textbox."""

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self.text_font = QFont("Courier")
        self._worker: Optional[RagArtifactsWorker] = None
        self._token_worker: Optional[TokenStatsWorker] = None
        self._wander_worker: Optional[WanderLogWorker] = None
        self._prompt_delta_worker: Optional[PromptDeltaWorker] = None
        self._wipe_worker: Optional[WipeWorker] = None
        self._dedup_worker: Optional[DedupFactsWorker] = None
        # Latest fetched slices, combined into the textbox by _rerender. Several async
        # workers (artifacts + wander log + prompt deltas) write the same box, so each
        # updates its slice and re-renders from all rather than overwriting the others.
        self._artifacts: list = []
        self._persona_digest: Optional[dict] = None
        self._outside_view: Optional[dict] = None
        self._wander_log: list = []
        self._prompt_deltas: list = []
        self._build_ui()

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._lbl_status = QLabel("RAG artifacts — press Refresh to load.")
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_artifacts)
        self.btn_dedup = QPushButton("Dedup facts…")
        self.btn_dedup.setToolTip(
            "Semantic de-duplication of live [fact] items on the clean base (adapter "
            "off): group paraphrases / cross-lingual restatements of the same fact, "
            "preview the merges, then evict duplicates to one survivor (union of "
            "triggers) and reload RAG. Append-only, so reversible."
        )
        self.btn_dedup.clicked.connect(self._on_dedup_clicked)
        self.btn_wipe = QPushButton("Wipe…")
        self.btn_wipe.setToolTip(
            "Disaster recovery: stop the server and delete all regenerable state "
            "(LoRA adapters, RAG, reflection data); optionally also chats + the "
            "reflection review archive; then reload a chosen base model."
        )
        self.btn_wipe.clicked.connect(self._on_wipe_clicked)
        header.addWidget(self._lbl_status, stretch=1)
        header.addWidget(self.btn_refresh)
        header.addWidget(self.btn_dedup)
        header.addWidget(self.btn_wipe)
        layout.addLayout(header)

        # Token-economy metrics row (above the textbox). Only "User tokens" for now;
        # external-token / Curiosity-Token fields will join here later.
        stats_row = QHBoxLayout()
        self.lbl_user_tokens = QLabel("User tokens: —")
        stats_row.addWidget(self.lbl_user_tokens)
        stats_row.addStretch(1)
        layout.addLayout(stats_row)

        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(self.text_font)
        self.txt_output.setPlaceholderText(
            "Active reflection RAG artifacts ([persona] / [fact] / [ask]) and the "
            "wander log (pages Ava wandered into) will appear here."
        )
        layout.addWidget(self.txt_output, stretch=1)

    # ---------------------------------------------------------------- #
    # Fetching + rendering                                              #
    # ---------------------------------------------------------------- #

    def refresh_artifacts(self) -> None:
        """Fetch the live RAG artifacts and token counters from the server (off-thread)."""
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._worker is None or not self._worker.isRunning():
            self._lbl_status.setText("Loading artifacts…")
            self.btn_refresh.setEnabled(False)
            self._worker = RagArtifactsWorker(client)
            self._worker.artifacts_ready.connect(self._on_artifacts_ready)
            self._worker.error_occurred.connect(self._on_artifacts_error)
            self._worker.start()
        if self._token_worker is None or not self._token_worker.isRunning():
            self._token_worker = TokenStatsWorker(client)
            self._token_worker.stats_ready.connect(self._on_token_stats_ready)
            self._token_worker.error_occurred.connect(self._on_token_stats_error)
            self._token_worker.start()
        if self._wander_worker is None or not self._wander_worker.isRunning():
            self._wander_worker = WanderLogWorker(client)
            self._wander_worker.log_ready.connect(self._on_wander_log_ready)
            self._wander_worker.error_occurred.connect(self._on_wander_log_error)
            self._wander_worker.start()
        if self._prompt_delta_worker is None or not self._prompt_delta_worker.isRunning():
            self._prompt_delta_worker = PromptDeltaWorker(client)
            self._prompt_delta_worker.deltas_ready.connect(self._on_prompt_deltas_ready)
            self._prompt_delta_worker.error_occurred.connect(self._on_prompt_deltas_error)
            self._prompt_delta_worker.start()

    def _on_artifacts_ready(self, payload: dict) -> None:
        self.btn_refresh.setEnabled(True)
        artifacts = payload.get("artifacts", [])
        self._artifacts = artifacts
        self._persona_digest = payload.get("persona_digest")
        self._outside_view = payload.get("outside_view")
        self._lbl_status.setText(f"{len(artifacts)} active artifact(s).")
        self._rerender()

    def _on_artifacts_error(self, err: str) -> None:
        self.btn_refresh.setEnabled(True)
        self._lbl_status.setText(f"Failed to load artifacts: {err}")

    def _on_wander_log_ready(self, entries: list) -> None:
        self._wander_log = entries
        self._rerender()

    def _on_wander_log_error(self, err: str) -> None:
        # Non-fatal: leave the artifacts view intact, just note the wander log failed.
        self._wander_log = []
        self._rerender()

    def _on_prompt_deltas_ready(self, deltas: list) -> None:
        self._prompt_deltas = deltas
        self._rerender()

    def _on_prompt_deltas_error(self, err: str) -> None:
        # Non-fatal: leave the rest of the view intact, just drop the prompt-delta slice.
        self._prompt_deltas = []
        self._rerender()

    def _on_token_stats_ready(self, stats: dict) -> None:
        econ = stats.get("token_economy") or {}
        acc = int(econ.get("accumulated", stats.get("user_tokens", 0)))
        consumed = int(econ.get("consumed", 0))
        wanders = int(econ.get("wanders_consumed", 0))
        available = int(econ.get("available", 0))
        self.lbl_user_tokens.setText(
            f"User tokens: {acc:,} accumulated / {consumed:,} consumed "
            f"({wanders} wander(s), {available} available)"
        )

    def _on_token_stats_error(self, err: str) -> None:
        self.lbl_user_tokens.setText("User tokens: —")

    def _rerender(self) -> None:
        """Render the latest artifacts + wander-log slices into the textbox."""
        sections: list[str] = []
        portrait_block = self._render_persona_digest(self._persona_digest)
        if portrait_block:
            sections.append(portrait_block)
        # Directly beneath it, deliberately: the two are independent readings of the same
        # conversations — one from the inside, one from the words alone — and the thing
        # worth looking at is where they disagree.
        outside_block = self._render_outside_view(self._outside_view)
        if outside_block:
            sections.append(outside_block)
        sections.append(self._render(self._artifacts))
        wander_block = self._render_wander(self._wander_log)
        if wander_block:
            sections.append(wander_block)
        delta_block = self._render_prompt_deltas(self._prompt_deltas)
        if delta_block:
            sections.append(delta_block)
        self.txt_output.setPlainText("\n\n".join(sections))

    @staticmethod
    def _render_persona_digest(digest: Optional[dict]) -> str:
        """Render Ava's current authored self-portrait, if the server has one."""
        if digest is None:
            return ""
        portrait = digest.get("self_portrait") or {}
        status = (portrait.get("status") or "unknown").strip()
        text = (portrait.get("text") or "").strip()
        error = (portrait.get("error") or "").strip()
        version = (digest.get("version") or "").strip()
        created = (digest.get("created") or "").strip().replace("T", " ")
        evidence = digest.get("evidence") or {}
        themes = evidence.get("persona_count")
        counts = digest.get("counts") or {}
        meta_parts = [
            f"version={version}" if version else "",
            f"created={created}" if created else "",
            f"themes={themes}" if themes is not None else "",
            " / ".join(
                f"{k}={counts.get(k, 0)}"
                for k in ("voice", "stances", "dispositions", "lines")
                if counts.get(k, 0)
            ),
            f"status={status}" if status else "",
        ]
        meta = ", ".join(p for p in meta_parts if p)
        lines = ["=== persona self-portrait ==="]
        if meta:
            lines.append(f"  {meta}")
        if text:
            lines.append("")
            lines.append(text)
        elif error:
            lines.append(f"  (self-portrait unavailable: {error})")
        else:
            lines.append("  (no self-portrait generated yet)")
        return "\n".join(lines)

    @staticmethod
    def _render_outside_view(view: Optional[dict]) -> str:
        """Render the outside-view self-portrait, if the server has one.

        The sibling of :meth:`_render_persona_digest`, and this tab is its PRIMARY
        surface: nothing injects this portrait into a prompt (see `core.self_portrait`),
        so the only places it is ever read are here and the reflection run log.
        """
        if view is None:
            return ""
        status = (view.get("status") or "unknown").strip()
        if status == "missing":
            return ""
        text = (view.get("text") or "").strip()
        error = (view.get("error") or "").strip()
        created = (view.get("created") or "").strip().replace("T", " ")
        counts = view.get("counts") or {}
        evidence = view.get("evidence") or {}
        meta_parts = [
            f"created={created}" if created else "",
            f"observations={evidence.get('item_count')}"
            if evidence.get("item_count") is not None else "",
            f"themes={evidence.get('theme_count')}"
            if evidence.get("theme_count") is not None else "",
            " / ".join(f"{k}={v}" for k, v in counts.items() if v),
        ]
        meta = ", ".join(p for p in meta_parts if p)
        lines = ["=== outside view (how she comes across) ==="]
        if meta:
            lines.append(f"  {meta}")
        if text:
            lines.append("")
            lines.append(text)
        elif error:
            lines.append(f"  (outside view unavailable: {error})")
        return "\n".join(lines)

    def _render(self, artifacts: list) -> str:
        """Group artifacts by type and render them as plain text."""
        if not artifacts:
            return "(no active RAG artifacts)"

        grouped: dict[str, list[dict]] = {}
        for a in artifacts:
            grouped.setdefault((a.get("kind") or "other"), []).append(a)

        # Known kinds first (in display order), then any unexpected kinds.
        ordered_kinds = [k for k in _KIND_ORDER if k in grouped]
        ordered_kinds += sorted(k for k in grouped if k not in _KIND_ORDER)

        blocks: list[str] = []
        for kind in ordered_kinds:
            items = grouped[kind]
            label = _KIND_LABELS.get(kind, f"[{kind}]")
            lines = [f"=== {label}  ({len(items)}) ==="]
            for a in items:
                lines.append(self._format_item(kind, a))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _render_wander(entries: list) -> str:
        """Render the wander log as its own category: page title + link, newest first.

        The title is shown alongside the URL because a non-English wiki link is not
        human-readable on its own."""
        if not entries:
            return ""
        lines = [f"=== wandered pages  ({len(entries)}) ==="]
        for e in entries:
            title = (e.get("title") or "").strip() or "(untitled)"
            url = (e.get("url") or "").strip()
            wiki = (e.get("wiki") or "").strip()
            ts = (e.get("ts") or "").strip().replace("T", " ")
            tag = "auto" if e.get("auto") else "manual"
            meta = ", ".join(p for p in (tag, ts, wiki) if p)
            head = f"  • {title}"
            if meta:
                head += f"   ({meta})"
            lines.append(head)
            if url:
                lines.append(f"      {url}")
        return "\n".join(lines)

    @staticmethod
    def _render_prompt_deltas(deltas: list) -> str:
        """Render the logged-only prompt-mutation proposals, newest first.

        Each is a standing-prompt change Ava's reflection suggested on a drifted
        exchange — logged only, never applied. Shown as its own category so it reads
        as a proposal queue, not as live state."""
        if not deltas:
            return ""
        lines = [f"=== prompt deltas — logged-only, not applied  ({len(deltas)}) ==="]
        for d in deltas:
            scope = (d.get("scope") or "?").strip() or "?"
            delta = (d.get("delta") or "").strip()
            drift = (d.get("drift") or "").strip()
            missing = (d.get("missing") or "").strip()
            sess = (d.get("source_session") or "").strip()
            ex = d.get("exchange_index")
            ts = (d.get("ts") or "").strip().replace("T", " ")
            meta = ", ".join(p for p in (
                f"[{scope}]",
                f"session={sess}" if sess else "",
                f"exchange={ex}" if ex is not None else "",
                ts,
            ) if p)
            lines.append(f"  • {delta or '(no line)'}")
            if meta:
                lines.append(f"      {meta}")
            if drift:
                lines.append(f"      drift:   {drift}")
            if missing:
                lines.append(f"      missing: {missing}")
        return "\n".join(lines)

    @staticmethod
    def _format_item(kind: str, a: dict) -> str:
        content = (a.get("content") or "").strip()
        annotations: list[str] = []
        if kind == "ask" and a.get("ask_kind"):
            annotations.append(f"ask_kind={a['ask_kind']}")
        if kind == "fact" and a.get("trigger"):
            annotations.append(f"trigger={a['trigger']}")
        if a.get("from_weights"):
            annotations.append("from_weights")
        if a.get("surface_count"):
            annotations.append(f"surfaced×{a['surface_count']}")
        if a.get("source_session"):
            annotations.append(f"session={a['source_session']}")
        suffix = f"   ({', '.join(annotations)})" if annotations else ""
        return f"  • {content}{suffix}"

    # ---------------------------------------------------------------- #
    # Wipe (disaster recovery)                                          #
    # ---------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    # Semantic fact dedup                                               #
    # ---------------------------------------------------------------- #

    def _on_dedup_clicked(self) -> None:
        """Two-step semantic fact dedup: run a clean-base PREVIEW (writes nothing),
        show the proposed merges, and apply only on an explicit confirm."""
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return
        if self._dedup_worker is not None and self._dedup_worker.isRunning():
            return
        resp = QMessageBox.information(
            self,
            "Dedup facts (preview)",
            "This groups Ava's live [fact] memories by meaning on the clean base "
            "(adapter off) to find duplicate phrasings of the same fact.\n\n"
            "It runs a PREVIEW first — nothing is changed until you confirm the "
            "merges. The server swaps the model twice, so this can take a while.\n\n"
            "Run preview?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if resp != QMessageBox.StandardButton.Ok:
            return
        self._start_dedup(dry_run=True)

    def _start_dedup(self, *, dry_run: bool) -> None:
        client = self._chat_widget._client
        self.btn_dedup.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.btn_wipe.setEnabled(False)
        self._lbl_status.setText(
            "Deduping facts (applying)…" if not dry_run
            else "Deduping facts (preview — swapping to the clean base)…"
        )
        self._dedup_worker = DedupFactsWorker(client, dry_run)
        self._dedup_worker.progress.connect(self._on_dedup_progress)
        self._dedup_worker.dedup_done.connect(self._on_dedup_done)
        self._dedup_worker.error_occurred.connect(self._on_dedup_error)
        self._dedup_worker.start()

    def _on_dedup_progress(self, ev: dict) -> None:
        """Show the grouping advancing — this tab has one status line, so the block
        counter goes there (the Sleep tab renders the same events into its log)."""
        if ev.get("stage") == "clustered":
            self._lbl_status.setText(
                f"Deduping facts — {ev.get('facts', 0)} fact(s) → "
                f"{ev.get('multi', 0)} subject(s) worth grouping "
                f"({ev.get('candidates', 0)} candidate(s))…")
        elif ev.get("stage") == "block":
            self._lbl_status.setText(
                f"Deduping facts — block {ev.get('i', 0)}/{ev.get('n', 0)} "
                f"({ev.get('merged', 0)} merge(s) so far in this block)…")

    def _on_dedup_done(self, result: dict) -> None:
        self.btn_dedup.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.btn_wipe.setEnabled(True)

        skipped = result.get("skipped")
        if skipped:
            msg = result.get("message") or skipped
            self._lbl_status.setText(f"Dedup skipped: {msg}")
            QMessageBox.information(self, "Dedup facts", msg)
            return

        groups = result.get("groups") or []
        before = result.get("before", 0)

        if result.get("dry_run"):
            if not groups:
                note = result.get("note") or "no duplicate facts found"
                self._lbl_status.setText(f"Dedup preview — {note}.")
                QMessageBox.information(
                    self, "Dedup facts",
                    f"No merges proposed among {before} live fact(s) ({note})."
                )
                return
            self.txt_output.setPlainText(self._format_dedup_preview(groups))
            n_evict = sum(len(g.get("evicted") or []) for g in groups)
            confirm = QMessageBox.question(
                self,
                "Apply dedup?",
                f"Found {len(groups)} duplicate group(s) among {before} fact(s); "
                f"{n_evict} fact(s) would be evicted (details in the panel).\n\n"
                "Apply these merges and reload RAG? The op-log is append-only, so "
                "this is reversible by editing rag_memory.jsonl.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._start_dedup(dry_run=False)
            else:
                self._lbl_status.setText("Dedup preview shown — not applied.")
            return

        # Applied.
        evicted = result.get("evicted", 0)
        after = result.get("after", "?")
        self._lbl_status.setText(
            f"Dedup applied — {evicted} fact(s) evicted, {after} remaining."
        )
        self.txt_output.setPlainText(self._format_dedup_preview(groups, applied=True))
        QMessageBox.information(
            self,
            "Dedup complete",
            f"Merged {len(groups)} group(s); {evicted} duplicate fact(s) evicted "
            f"({before} → {after} live facts). RAG was reloaded in place.\n\n"
            "Press Refresh to see the updated artifact list.",
        )
        self.refresh_artifacts()

    def _on_dedup_error(self, err: str) -> None:
        self.btn_dedup.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.btn_wipe.setEnabled(True)
        self._lbl_status.setText(f"Dedup failed: {err}")
        QMessageBox.critical(self, "Dedup failed", f"The dedup did not complete:\n\n{err}")

    @staticmethod
    def _format_dedup_preview(groups: list, *, applied: bool = False) -> str:
        header = ("Applied merges:" if applied
                  else "Proposed merges (preview — not yet applied):")
        lines = [header, ""]
        for i, g in enumerate(groups, 1):
            lines.append(f"Group {i} — keep:")
            lines.append(f"    {g.get('survivor', '')}")
            trigger = g.get("merged_trigger")
            if trigger:
                lines.append(f"    (recalled when: {trigger})")
            lines.append("  evict:")
            for e in (g.get("evicted") or []):
                lines.append(f"    - {e}")
            lines.append("")
        return "\n".join(lines)

    def _on_wipe_clicked(self) -> None:
        """Guided destructive wipe: confirm regenerable delete → ask about chats +
        manifest → choose a new base model → final confirm → execute off-thread.

        Nothing is sent to the server until every prompt is answered, so cancelling
        at any step aborts cleanly with no partial state.
        """
        client = self._chat_widget._client
        if not client.is_connected():
            self._lbl_status.setText("Not connected — connect in the Chat tab first.")
            return

        # 1. Confirm the always-on regenerable wipe.
        resp = QMessageBox.warning(
            self,
            "Wipe regenerable state",
            "This permanently deletes everything Ava can regenerate:\n\n"
            "  •  all LoRA adapters\n"
            "  •  RAG indexes\n"
            "  •  reflection memory, ledger, run logs, staging\n\n"
            "Your chat transcripts are kept for now (you'll be asked about them "
            "next). The inference server will restart. This cannot be undone.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        # 2. Ask whether the total reset should also remove chats and review history.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Delete chats and reflection archive?")
        box.setText(
            "Also delete the raw chat transcripts and the reflection review archive?\n\n"
            "Keep them unless "
            "you want a total reset — a brand-new Ava with no memory of any past "
            "conversation."
        )
        keep_btn = box.addButton("Keep chats", QMessageBox.ButtonRole.AcceptRole)
        delete_btn = box.addButton("Delete everything", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked is cancel_btn:
            return
        wipe_chats = clicked is delete_btn

        # 3. Choose the base model to load after the wipe (current model preselected).
        current = (getattr(self._chat_widget, "_server_model_id", "") or "").strip()
        options: list[str] = []
        if current:
            options.append(current)
        for m in _KNOWN_BASE_MODELS:
            if m not in options:
                options.append(m)
        model_id, ok = QInputDialog.getItem(
            self,
            "Choose base model",
            "Base model to load after the wipe (editable — any HF id):",
            options,
            0,
            True,
        )
        if not ok:
            return
        model_id = (model_id or "").strip()

        # 4. Final summary confirm — this is destructive and restarts the server.
        lines = [
            "About to wipe state and restart the server:",
            "",
            "  •  delete adapters + RAG + reflection data",
        ]
        lines.append("  •  DELETE all chats + reflection archive" if wipe_chats
                     else "  •  keep chats + reflection archive")
        lines.append(f"  •  reload base model: {model_id or '(unchanged)'}")
        lines += ["", "Proceed?"]
        final = QMessageBox.warning(
            self, "Confirm wipe", "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if final != QMessageBox.StandardButton.Yes:
            return

        # Execute off-thread (the server blocks until the relaunch is issued).
        self.btn_wipe.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self._lbl_status.setText("Wiping state and restarting the server…")
        self._wipe_worker = WipeWorker(client, wipe_chats, model_id)
        self._wipe_worker.wipe_done.connect(self._on_wipe_done)
        self._wipe_worker.error_occurred.connect(self._on_wipe_error)
        self._wipe_worker.start()

    def _on_wipe_done(self, result: dict) -> None:
        self.btn_wipe.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        model_id = result.get("model_id") or "(unchanged)"
        summary = result.get("summary") or {}
        self.txt_output.setPlainText(
            "State wiped.\n\n"
            f"Base model now configured: {model_id}\n\n"
            f"Wipe summary:\n{json.dumps(summary, indent=2, ensure_ascii=False)}"
        )
        self._lbl_status.setText("Wipe complete — server is reloading the model.")
        QMessageBox.information(
            self,
            "Wipe complete",
            "The server is restarting and loading the base model "
            f"({model_id}). This can take a while for a large model.\n\n"
            "Reconnect from the Chat tab once it is back up.",
        )

    def _on_wipe_error(self, err: str) -> None:
        self.btn_wipe.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self._lbl_status.setText(f"Wipe failed: {err}")
        QMessageBox.critical(self, "Wipe failed", f"The wipe did not complete:\n\n{err}")

    # ---------------------------------------------------------------- #
    # Fonts                                                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        self.txt_output.setFont(font)
