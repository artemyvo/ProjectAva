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
    uses, and saving makes that value explicit. The schema itself lives in
    ``config_schema.py`` and is the same list the inference server back-fills into an
    older config at boot — so after one server run every knob here is already in the
    file, and this editor is a typed view over it rather than the only way to find one.
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

# The schema (tabs → fields, defaults, tooltips) and the dotted-path helpers live in
# config_schema.py — PyQt-free, so the inference server's boot back-fill walks the SAME
# list this editor renders (one definition of "the documented knob set").
sys.path.insert(0, str(_SERVER_DIR))
from config_schema import (  # noqa: E402
    Field, Tab, SCHEMA, KNOWN_MODELS, dget, dhas, dset, config_text)


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
            if kind in ("floatlist", "strlist"):
                return []
            raise FieldError(f"{spec.label}: value required")
        try:
            if kind in ("int", "opt_int"):
                return int(raw)
            if kind in ("float", "opt_float"):
                return float(raw)
            if kind == "floatlist":
                parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
                return [float(p) for p in parts]
            if kind == "strlist":
                return [p.strip() for p in raw.split(",") if p.strip() != ""]
        except ValueError:
            raise FieldError(f"{spec.label}: '{raw}' is not a valid {kind}")
        return raw  # str / path

    def num_set(v: Any) -> None:
        if v is None:
            le.setText("")
        elif isinstance(v, (list, tuple)):
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
