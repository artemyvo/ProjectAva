#!/usr/bin/env python3
"""Ava server settings editor — a small PyQt6 GUI over ``server/server_config.json``.

Most of the server's knobs live in that one JSON file, but only a handful are ever
written into a fresh config; the rest are read with in-code defaults and are only
adjustable by hand-editing JSON (or not adjustable at all without knowing they exist).
This tool surfaces the full, documented set — base model + quantisation, chat/generation
guards, the offline SFT (training) parameters, the wall-clock consolidation/decay curve,
check-in, and gossip — in one editable window.

Design contract:
  * It is a **round-trip editor**. The whole JSON is loaded and kept; on Save only the
    keys this GUI manages are written back (nested groups merged in place), so any key
    it does not know about is preserved verbatim.
  * Every default shown matches the server's own in-code default (see the citations in
    each field's tooltip), so an unset field displays what the running server actually
    uses, and saving makes that value explicit.
  * A timestamped ``.bak`` copy is written before each Save.

Run it with any Python that has PyQt6 (the client venv has it):

    cd server
    python settings.py                       # edits server/server_config.json
    python settings.py --config /path/to/server_config.json

No server restart happens here — settings take effect the next time the inference
server (re)starts / reloads the model, exactly as a hand-edit would.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QFormLayout, QVBoxLayout,
        QHBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QScrollArea,
        QFileDialog, QMessageBox, QPlainTextEdit, QDialog, QDialogButtonBox,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "PyQt6 is required. Run this with the client virtualenv, e.g.\n"
        "    source ../.venv/bin/activate && python settings.py\n"
    )
    raise


_SERVER_DIR = Path(__file__).resolve().parent
# The box config moved up out of inference/ on 2026-07-28; a not-yet-migrated
# checkout is still editable at the legacy path (this editor never moves it).
DEFAULT_CONFIG_PATH = _SERVER_DIR / "server_config.json"
if not DEFAULT_CONFIG_PATH.exists() and (_SERVER_DIR / "inference" / "server_config.json").exists():
    DEFAULT_CONFIG_PATH = _SERVER_DIR / "inference" / "server_config.json"

# Known HF base-model ids the picker offers (editable — any id may be typed). Mirrors
# client/ui/debug_widget.py::_KNOWN_BASE_MODELS.
KNOWN_MODELS = [
    "unsloth/gemma-4-31B-it",
    "unsloth/Qwen3.6-27B",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
]


# ──────────────────────────────────────────────────────────────────────────────
# Field schema
# ──────────────────────────────────────────────────────────────────────────────

# kinds:
#   str        free text                      combo   fixed+editable choices
#   path       free text + Browse             bool    checkbox
#   int        integer                        float   float (accepts 8e-6 etc.)
#   opt_int    integer or empty→null          opt_float float or empty→null
#   floatlist  comma-separated floats

@dataclass
class Field:
    path: str                     # dotted key path into the config dict
    label: str
    kind: str
    default: Any = None
    choices: Optional[list[str]] = None
    tooltip: str = ""


@dataclass
class Tab:
    name: str
    intro: str
    fields: list[Field] = field(default_factory=list)


SCHEMA: list[Tab] = [
    Tab(
        "Model & precision",
        "Base weights, quantisation, and the context windows. The base <b>model_id</b> is "
        "frozen once weights exist; training only ever repoints <b>adapter_id</b>. "
        "Changes take effect on the next server restart / model reload.",
        [
            Field("model_id", "Base model id", "combo", "unsloth/gemma-4-31B-it",
                  choices=KNOWN_MODELS,
                  tooltip="HuggingFace repo id of the frozen base model. Any id may be "
                          "typed. Raising precision to 8/16-bit needs the full-precision "
                          "base repo to be downloadable (a bnb-4bit repo can't be upcast)."),
            Field("base_quant", "Base load precision", "combo", "",
                  choices=["", "16bit", "8bit", "4bit"],
                  tooltip="Base-model load precision. Empty ⇒ the backend's historical "
                          "4-bit default. With an adapter present, 8/16-bit loads the "
                          "full-precision base and attaches the LoRA on top. "
                          "(server.py::_resolve_base_quant)"),
            Field("adapter_id", "Active LoRA adapter", "path", "",
                  tooltip="Absolute path to the active LoRA adapter dir. Set/rotated by "
                          "the training cycle; edit to roll back to an earlier persona's "
                          "adapter. Empty ⇒ base model only."),
            Field("context_length", "Chat context length", "int", 24576,
                  tooltip="Chat / default token budget. Chat transcripts are capped here "
                          "so they always fit a later reflection window."),
            Field("reflect_context_length", "Reflection context length", "int", 24576,
                  tooltip="Reflection window AND the physical max_seq_length the model "
                          "loads at (model is loaded once at max(context, reflect)). Raise "
                          "to let a reflection run pack a bigger window than chat. "
                          "Back-filled == context_length on older configs."),
        ],
    ),
    Tab(
        "Chat / generation",
        "Live-chat degeneration guards (chat / ephemeral / encounter paths only — never "
        "reflection). All are content-blind mechanisms, so they hold for any emergent "
        "persona. See server.py main() and core/inference_backend.py.",
        [
            Field("chat_repetition_penalty", "Repetition penalty", "opt_float", None,
                  tooltip="Mild live-chat repetition penalty. 1.0 / empty(null) disables "
                          "it (the halt-only stop_on_repeat guard is always on). Default "
                          "null. Only values > 1.0 take effect."),
            Field("chat_min_p", "min_p sampling floor", "opt_float", 0.02,
                  tooltip="Layer-1 relative-probability sampling floor: cuts the "
                          "implausible tail that seeds a collapse while leaving a "
                          "high-entropy persona's nucleus intact. Default 0.02; empty/≤0 "
                          "disables."),
            Field("chat_degen_guard", "Drifting-degeneration guard", "bool", True,
                  tooltip="Layer-2 halt for an associative-walk / letter-soup runaway "
                          "(distinct-token-ratio / single-token-dominance over a rolling "
                          "window) that the verbatim stop_on_repeat guard is blind to. "
                          "Default on."),
            Field("chat_degen.window", "  degen: window", "opt_int", None,
                  tooltip="Optional override of the degeneration guard's rolling-window "
                          "size. Empty ⇒ backend default."),
            Field("chat_degen.min_gen", "  degen: min tokens", "opt_int", None,
                  tooltip="Optional override: minimum generated tokens before the guard "
                          "may fire. Empty ⇒ backend default."),
            Field("chat_degen.distinct_ratio", "  degen: distinct ratio", "opt_float", None,
                  tooltip="Optional override: distinct-token ratio below which the window "
                          "is judged degenerate. Empty ⇒ backend default."),
            Field("chat_degen.top_freq", "  degen: top-token freq", "opt_float", None,
                  tooltip="Optional override: single-token dominance fraction that trips "
                          "the guard. Empty ⇒ backend default."),
        ],
    ),
    Tab(
        "Training (SFT)",
        "Offline from-scratch LoRA build parameters (training/train_cycle.py, defaults in "
        "training/decay.py). The adapter is rebuilt on the frozen base every cycle. "
        "Per-row wall-clock multipliers (Consolidation tab) scale on top of these.",
        [
            Field("lora_r", "LoRA rank (r)", "int_combo", 32,
                  choices=[8, 16, 32, 64, 128],
                  tooltip="Adapter capacity: get_peft_model r for the from-scratch LoRA "
                          "fit. Default 32 (decay.TRAIN_LORA_R_DEFAULT); higher = more "
                          "capacity + VRAM. Scaling is rank-stabilized (use_rslora, "
                          "gamma = alpha/sqrt(r) with alpha fixed at 4), calibrated so "
                          "gamma == 1.0 at r=16 — so changing rank adds/removes capacity "
                          "WITHOUT changing the effective learning rate, and train_lr "
                          "stays valid across ranks. A CLI --lora-r / Sleep "
                          "train_params.lora_r overrides at run time."),
            Field("train_lr", "Base / peak SFT LR", "float", 8e-6,
                  tooltip="Base/peak learning rate the per-row multipliers + schedule "
                          "shape scale on top of. Default 8e-6 (decay.TRAIN_LR_DEFAULT). "
                          "A CLI --lr / Sleep train_params.lr overrides at run time."),
            Field("train_lr_schedule", "LR schedule shape", "combo", "age_ramp",
                  choices=["age_ramp", "triangular"],
                  tooltip="Global LR shape layered on the per-row multipliers. "
                          "'age_ramp' = DEFAULT: flat single pass (LR = base × row_mult), "
                          "one epoch, the age ramp alone weighting the rows. 'triangular' "
                          "= warmup + plateau + decay trapezoid (order-neutral, forces "
                          "epochs = plateau+2)."),
            Field("train_plateau_epochs", "Plateau (hold) epochs", "int", 3,
                  tooltip="Full-LR hold epochs in the triangular schedule (forces total "
                          "epochs = this + 2). 0 = minimal trapezoid (warmup+decay only, "
                          "2 epochs). Default 3 (decay.TRAIN_PLATEAU_EPOCHS_DEFAULT). "
                          "IGNORED under the default 'age_ramp' schedule."),
            Field("train_max_seq_length", "Training max seq length", "int", 8192,
                  tooltip="Offline-training sequence cap (clamped to context_length). "
                          "Drives the answer-preserving message-level truncation (drop "
                          "oldest history turns, keep system + final exchange + target); "
                          "a row still over cap is quarantined. It no longer bounds the "
                          "fused CE-loss chunk — that is pinned by "
                          "UNSLOTH_CE_LOSS_TARGET_GB regardless of sequence length — so "
                          "lower it for an activation-side OOM only. Training-only — "
                          "inference always uses full context_length. Default 8192."),
        ],
    ),
    Tab(
        "Consolidation / decay",
        "Wall-clock consolidation curve (consolidation.wall_clock, parsed by "
        "training/decay.py::WallClockConfig). Controls how a chat's age scales its "
        "training weight and the verbatim-to-gist RAG handoff. Hours are wall-clock "
        "hours since the chat vs the build time.",
        [
            Field("consolidation.decay_curve", "Decay curve", "combo", "linear",
                  choices=["linear"],
                  tooltip="Shared decay curve name (ConsolidationConfig)."),
            Field("consolidation.wall_clock.rag_only_window_h", "RAG-only window (h)", "float", 24.0,
                  tooltip="LR multiplier is 0 below this age (models the pre-reflection "
                          "span — RAG only, no trainable row). Default 24h."),
            Field("consolidation.wall_clock.lora_cap_age_h", "LoRA cap age (h)", "float", 72.0,
                  tooltip="Age at which the per-row LR multiplier reaches the ramp cap "
                          "(~3d). Default 72h."),
            Field("consolidation.wall_clock.rag_cap_age_h", "RAG handoff age (h)", "float", 96.0,
                  tooltip="Age at which verbatim-chat RAG reaches hard 0, gist reaches "
                          "peak weight, and persona reaches its floor (~4d). Default 96h."),
            Field("consolidation.wall_clock.gist_cap_age_h", "Gist cap age (h)", "float", 192.0,
                  tooltip="Age at which the per-conversation gist/summary reaches its "
                          "permanent floor (~8d). Default 192h."),
            Field("consolidation.wall_clock.rag_floor_weight", "Persona RAG floor", "float", 0.2,
                  tooltip="Permanent retrieval floor for persona anchors. Verbatim chat "
                          "does not use this floor. Default 0.2."),
            Field("consolidation.wall_clock.gist_floor_weight", "Gist RAG floor", "float", 0.2,
                  tooltip="Permanent gist/summary retrieval floor reached exactly at the "
                          "gist cap age. Default 0.2."),
            Field("consolidation.wall_clock.base_lr", "Wall-clock base LR", "float", 1e-5,
                  tooltip="Base LR the per-row wall-clock multiplier scales (REBUILD §1). "
                          "Default 1e-5. Note: train_lr (Training tab) is the actual peak "
                          "SFT LR used by train_cycle."),
            Field("consolidation.wall_clock.lr_ramp", "LR ramp sample points", "floatlist",
                  [1.0, 2.0, 4.0],
                  tooltip="Explicit LR-ramp sample points across [rag_only_window, "
                          "lora_cap]; the cap is the last value. Comma-separated. "
                          "Default 1.0, 2.0, 4.0."),
            Field("consolidation.wall_clock.contamination.enabled", "Contamination: enabled",
                  "bool", True,
                  tooltip="Cap-age user-contamination: a cap-age exchange also emits a "
                          "user-turn-unmasked copy so voice entrains at a small dose. "
                          "Default on. (§5e)"),
            Field("consolidation.wall_clock.contamination.dose", "Contamination: dose", "float", 1.0,
                  tooltip="Unmasked-copy LR multiplier at cap (the response keeps the full "
                          "cap; voice entrains at this dose). Default 1.0."),
            Field("consolidation.wall_clock.contamination.additive", "Contamination: additive",
                  "bool", False,
                  tooltip="False: split cap → (cap−dose) masked + dose unmasked. True: keep "
                          "cap masked + add an extra dose unmasked copy. Default False."),
            Field("consolidation.wall_clock.contamination.min_user_chars", "Contamination: min user chars",
                  "int", 100,
                  tooltip="Minimum user-turn length (chars) for an exchange to emit a "
                          "contamination copy. Default 100."),
            Field("consolidation.wall_clock.contamination.fold", "Contamination: fold pair",
                  "bool", False,
                  tooltip="Fold the cap-age contamination PAIR into ONE per-token-weighted "
                          "row (~40% fewer rows). Default False."),
            Field("consolidation.wall_clock.wander_rag.weights", "Wander RAG weights", "floatlist",
                  [0.4, 0.3, 0.2, 0.1],
                  tooltip="Age-stepped retrieval weights for the wander channel. "
                          "Comma-separated. Default 0.4, 0.3, 0.2, 0.1."),
            Field("consolidation.wall_clock.wander_rag.step_h", "Wander RAG step (h)", "float", 24.0,
                  tooltip="Hours per wander-RAG weight step. Default 24h."),
            Field("consolidation.wall_clock.fresh_window.droop_frac", "Fresh: droop frac", "float", 0.5,
                  tooltip="Freshness droop fraction for a just-happened chat. Default 0.5; "
                          "set 0 to disable the fresh-window adjustment."),
            Field("consolidation.wall_clock.fresh_window.horizon_h", "Fresh: horizon (h)", "float", 24.0,
                  tooltip="Freshness horizon in hours. Default 24h."),
        ],
    ),
    Tab(
        "Check-in",
        "The autonomous check-in idle job (core/checkin.py): after the USER has been "
        "quiet long enough, Ava reviews recent chats and may reach out on her own. "
        "Absent ⇒ the defaults below.",
        [
            Field("checkin.silence_threshold_hours", "User-silence threshold (h)", "float", 5.0,
                  tooltip="How long the user must be quiet (measured from disk, not the "
                          "idle clock) before an autonomous check-in may fire. Default 5h."),
            Field("checkin.recent_chats", "Recent chats reviewed", "int", 5,
                  tooltip="How many recent conversations she summarises to decide whether "
                          "to reach out. Default 5."),
        ],
    ),
    Tab(
        "Gossip",
        "Model-gossip serving (GOSSIP.md). When enabled, the inference HTTP sidecar "
        "exposes an OpenAI-compatible /v1/chat/completions so a peer Ava can drive this "
        "box as if it were vLLM.",
        [
            Field("gossip.enabled", "Gossip serving enabled", "bool", True,
                  tooltip="ON by default since 2026-07-30 (a pulled box is reachable by a "
                          "peer with no config edit). False ⇒ the endpoint 404s. The route "
                          "carries no auth — turn it off on an untrusted network."),
            Field("gossip.log_transcripts", "Log served transcripts", "bool", True,
                  tooltip="Serving-side reflection: log this box's own half of the gossip "
                          "(with its CoT) so its Sleep pass can reflect on it too. Default "
                          "on; false restores stateless serving."),
            Field("gossip.peer_name", "Peer name", "str", "",
                  tooltip="Optional display name for the peer Ava. Empty ⇒ unset."),
        ],
    ),
    Tab(
        "Public API",
        "OpenAI-compatible API for external tools (core/api_http.py) — an agentic code "
        "assistant, an editor plugin, an SDK script. Its own port, separate from the "
        "management sidecar. Requests are NEVER logged: nothing written to chats/, so "
        "nothing reaches reflection or the training corpus. Non-streaming.",
        [
            Field("api.enabled", "Public API enabled", "bool", True,
                  tooltip="ON by default since 2026-07-30 (a pulled box is queryable by an "
                          "external tool with no config edit). False ⇒ the listener never "
                          "starts. Set an API key, or bind to 127.0.0.1, on a shared "
                          "network — with neither, the port is open."),
            Field("api.port", "Port", "int", 8000,
                  tooltip="Listener port. Default 8000 (the --api-port flag is the "
                          "fallback when this is unset)."),
            Field("api.host", "Bind address", "str", "",
                  tooltip="Empty ⇒ the server's --api-host, else --host. Use 127.0.0.1 to "
                          "keep it local to the box."),
            Field("api.api_key", "API key (bearer)", "str", "",
                  tooltip="Shared secret required as `Authorization: Bearer <key>`. EMPTY "
                          "⇒ no auth: anyone who can reach the port can spend GPU time as "
                          "Ava. Set one whenever the bind address is not 127.0.0.1."),
            Field("api.model_name", "Model name", "str", "ava",
                  tooltip="The id reported by GET /v1/models and echoed in responses — "
                          "what the client's model picker shows."),
            Field("api.max_tokens", "Default max tokens", "str", "75%",
                  tooltip="Used when a request omits max_tokens. Accepts an integer or a "
                          "percentage of the remaining context (chat's own convention)."),
            Field("api.client_system", "Client system prompt", "combo", "append",
                  choices=["append", "drop"],
                  tooltip="append: wrap the calling tool's system message in "
                          "prompts/api_client_system_prompt.txt and place it last (an "
                          "agent's whole operating brief lives there). drop: ignore it, "
                          "gossip-style — she only ever answers as herself."),
            Field("api.inject_rag", "Inject RAG memory", "bool", True,
                  tooltip="Retrieve her memory (past chats, facts, persona) for each "
                          "request, as live chat does. Off ⇒ prompt + adapter only, which "
                          "is usually what a code assistant wants."),
            Field("api.inject_persona", "Inject persona portrait", "bool", True,
                  tooltip="Inject the standing persona digest, as live chat does. Off ⇒ "
                          "no portrait and no [persona] RAG channel."),
        ],
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Nested dict helpers
# ──────────────────────────────────────────────────────────────────────────────

def dget(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dhas(d: dict, path: str) -> bool:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def dset(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# Field <-> widget binding
# ──────────────────────────────────────────────────────────────────────────────

class FieldError(ValueError):
    pass


class Binding:
    """Wraps one Field's widget and knows how to read/write its config value."""

    def __init__(self, spec: Field, widget: QWidget, getter: Callable[[], Any],
                 setter: Callable[[Any], None]):
        self.spec = spec
        self.widget = widget
        self._get = getter
        self._set = setter

    def load(self, cfg: dict) -> None:
        val = dget(cfg, self.spec.path, self.spec.default)
        self._set(val)

    def collect(self) -> Any:
        """Return the parsed value for this field. Raises FieldError on bad input."""
        return self._get()


def _make_binding(spec: Field, browse_parent: QWidget) -> Binding:
    kind = spec.kind

    if kind == "bool":
        w = QCheckBox()
        return Binding(spec, w, lambda: w.isChecked(),
                       lambda v: w.setChecked(bool(v)))

    if kind == "combo":
        w = QComboBox()
        w.setEditable(True)
        for c in (spec.choices or []):
            w.addItem(c)

        def cget() -> str:
            return w.currentText().strip()

        def cset(v: Any) -> None:
            w.setCurrentText("" if v is None else str(v))

        return Binding(spec, w, cget, cset)

    if kind == "int_combo":
        # Fixed set of integer choices; stored as an int so the config key stays
        # type-consistent with the server's back-fill.
        w = QComboBox()
        w.setEditable(False)
        for c in (spec.choices or []):
            w.addItem(str(c))

        def iget() -> int:
            raw = w.currentText().strip()
            try:
                return int(raw)
            except ValueError:
                raise FieldError(f"{spec.label}: '{raw}' is not a valid integer")

        def iset(v: Any) -> None:
            text = "" if v is None else str(int(v)) if isinstance(v, (int, float)) else str(v)
            idx = w.findText(text)
            if idx < 0 and text != "":
                w.addItem(text)          # tolerate an out-of-list value already in the config
                idx = w.findText(text)
            w.setCurrentIndex(max(0, idx))

        return Binding(spec, w, iget, iset)

    # Everything else is a line edit (kept simple + validated on save).
    le = QLineEdit()

    def num_get() -> Any:
        raw = le.text().strip()
        nullable = kind in ("opt_int", "opt_float")
        if raw == "":
            if nullable:
                return None
            raise FieldError(f"{spec.label}: value required")
        try:
            if kind in ("int", "opt_int"):
                return int(raw)
            if kind in ("float", "opt_float"):
                return float(raw)
            if kind == "floatlist":
                parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
                return [float(p) for p in parts]
        except ValueError:
            raise FieldError(f"{spec.label}: '{raw}' is not a valid {kind}")
        return raw  # str / path

    def num_set(v: Any) -> None:
        if v is None:
            le.setText("")
        elif isinstance(v, list):
            le.setText(", ".join(_fmt_num(x) for x in v))
        elif isinstance(v, float):
            le.setText(_fmt_num(v))
        else:
            le.setText(str(v))

    binding = Binding(spec, le, num_get, num_set)

    if kind == "path":
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le, 1)
        btn = QPushButton("Browse…")

        def browse() -> None:
            start = le.text().strip() or str(Path.cwd())
            path = QFileDialog.getExistingDirectory(browse_parent, "Select adapter dir", start)
            if path:
                le.setText(path)

        btn.clicked.connect(browse)
        h.addWidget(btn)
        binding.widget = row  # the composite goes into the form

    return binding


def _fmt_num(x: Any) -> str:
    """Compact numeric formatting: keep ints int-looking, avoid trailing noise, keep
    scientific notation readable for tiny LRs."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-4 or abs(x) >= 1e7):
            return f"{x:g}"
        s = repr(x)
        return s
    return str(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class SettingsWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.raw_config: dict = {}
        self.bindings: list[Binding] = []

        self.setWindowTitle(f"Ava Server Settings — {config_path}")
        self.resize(760, 720)

        central = QWidget()
        root = QVBoxLayout(central)

        header = QLabel(f"Editing <code>{config_path}</code>")
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        root.addWidget(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_tabs()

        # Buttons
        bar = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        bar.addWidget(self.status, 1)

        preview_btn = QPushButton("Preview JSON")
        preview_btn.clicked.connect(self._preview)
        bar.addWidget(preview_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        bar.addWidget(reload_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        bar.addWidget(save_btn)

        root.addLayout(bar)
        self.setCentralWidget(central)

        self._reload()

    def _build_tabs(self) -> None:
        mono = QFont("monospace")
        for tab in SCHEMA:
            page = QWidget()
            outer = QVBoxLayout(page)

            intro = QLabel(tab.intro)
            intro.setTextFormat(Qt.TextFormat.RichText)
            intro.setWordWrap(True)
            intro.setStyleSheet("color: palette(mid-text); padding: 4px 0 8px 0;")
            outer.addWidget(intro)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            form = QFormLayout(inner)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

            for spec in tab.fields:
                binding = _make_binding(spec, self)
                self.bindings.append(binding)
                lbl = QLabel(spec.label)
                if spec.tooltip:
                    lbl.setToolTip(spec.tooltip)
                    binding.widget.setToolTip(spec.tooltip)
                if isinstance(binding.widget, QLineEdit):
                    binding.widget.setFont(mono)
                form.addRow(lbl, binding.widget)

            scroll.setWidget(inner)
            outer.addWidget(scroll, 1)
            self.tabs.addTab(page, tab.name)

    # ── actions ────────────────────────────────────────────────────────────
    def _reload(self) -> None:
        cfg: dict = {}
        if self.config_path.exists():
            try:
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    raise ValueError("top-level JSON is not an object")
            except Exception as e:
                QMessageBox.critical(self, "Load error",
                                     f"Could not read {self.config_path}:\n{e}")
                cfg = {}
        else:
            self.status.setText("File does not exist yet — Save will create it.")
        self.raw_config = cfg
        for b in self.bindings:
            b.load(cfg)
        n_unmanaged = self._count_unmanaged(cfg)
        extra = f"  ({n_unmanaged} unmanaged key(s) preserved)" if n_unmanaged else ""
        self.status.setText(f"Loaded.{extra}")

    def _managed_top_keys(self) -> set[str]:
        return {b.spec.path.split(".")[0] for b in self.bindings}

    def _count_unmanaged(self, cfg: dict) -> int:
        return sum(1 for k in cfg if k not in self._managed_top_keys())

    def _collect_into(self, base: dict) -> Optional[dict]:
        """Apply every binding's value onto a deep copy of *base*. Returns None (and
        pops a dialog) if any field fails validation."""
        out = copy.deepcopy(base)
        errors: list[str] = []
        for b in self.bindings:
            try:
                val = b.collect()
            except FieldError as e:
                errors.append(str(e))
                continue
            dset(out, b.spec.path, val)
        if errors:
            QMessageBox.warning(self, "Invalid values",
                                "Fix these before saving:\n\n• " + "\n• ".join(errors))
            return None
        return out

    def _preview(self) -> None:
        out = self._collect_into(self.raw_config)
        if out is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Pending server_config.json")
        dlg.resize(560, 640)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setFont(QFont("monospace"))
        edit.setPlainText(json.dumps(out, indent=2))
        lay.addWidget(edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _save(self) -> None:
        out = self._collect_into(self.raw_config)
        if out is None:
            return
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config_path.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = self.config_path.with_suffix(self.config_path.suffix + f".{stamp}.bak")
                shutil.copy2(self.config_path, backup)
            else:
                backup = None
            self.config_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not write config:\n{e}")
            return
        self.raw_config = out
        note = f"  (backup: {backup.name})" if backup else "  (new file created)"
        self.status.setText(f"Saved {datetime.now():%H:%M:%S}.{note}  "
                            f"Restart the server for it to take effect.")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ava server settings editor (GUI).")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help=f"Path to server_config.json (default: {DEFAULT_CONFIG_PATH})")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    win = SettingsWindow(args.config.resolve())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
