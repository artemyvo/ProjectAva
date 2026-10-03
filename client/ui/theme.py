"""The app-wide dark theme: one palette every tab inherits.

Why dark everywhere, not just the chat pane: the Chat tab paints its log black with
neutral-grey body text, and the per-token channels only ever ADD light to that grey
(`ChatWidget.paint_black_pane`). Framed by a light window, the eye adapts to the bright
controls and the grey body reads as murky and the lifts as faint — the colouring is only
legible when nothing around it is brighter than it needs to be. So the chrome sits a step
above black (WINDOW), the text fields a half-step below it (BASE), and the control text
is a soft grey rather than white; colour is kept for state and selection.

Fusion is the style on every platform because it is the one built-in style that draws
entirely from the palette — the native macOS and Windows styles keep drawing some
controls with the system's light appearance whatever palette they are handed. The
colour-scheme hint is set too, so the window frame and title bar follow on platforms
that honour it (Qt ≥ 6.8).

Widgets reference the named colours below rather than inlining hex values, so a
retune happens here once.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# Surfaces, darkest to lightest. The chat pane itself is pure black (#000), below all of these.
BASE = "#121214"          # text fields, lists, tables, trees
ALT_BASE = "#18181b"      # alternate rows
WINDOW = "#1c1c1f"        # window and tab chrome
BUTTON = "#2a2a2e"        # buttons, combo boxes, headers
BORDER = "#3a3a3f"        # bevel highlight, tooltip / menu edges

# Text, brightest to faintest.
TEXT = "#c8c8cc"          # body and control text — soft, never pure white
MUTED = "#85858b"         # secondary labels, status lines, captions
DISABLED = "#56565c"      # disabled controls and placeholders

# Accents.
HIGHLIGHT = "#2f5584"     # selection background — a muted blue, not a glare
HIGHLIGHTED_TEXT = "#f2f2f4"
LINK = "#7fb0ff"
LINK_VISITED = "#b49aff"
ERROR = "#ff7b72"
WARNING = "#f0b860"
OK = "#7ed492"

_STYLESHEET = f"""
QToolTip {{
    color: {TEXT};
    background-color: {BUTTON};
    border: 1px solid {BORDER};
    padding: 4px;
}}
"""


def dark_palette() -> QPalette:
    """The full dark palette, all three colour groups (Disabled overridden where it differs)."""
    pal = QPalette()
    role = QPalette.ColorRole
    for r, c in (
        (role.Window, WINDOW),
        (role.WindowText, TEXT),
        (role.Base, BASE),
        (role.AlternateBase, ALT_BASE),
        (role.ToolTipBase, BUTTON),
        (role.ToolTipText, TEXT),
        (role.PlaceholderText, DISABLED),
        (role.Text, TEXT),
        (role.Button, BUTTON),
        (role.ButtonText, TEXT),
        (role.BrightText, ERROR),
        (role.Highlight, HIGHLIGHT),
        (role.HighlightedText, HIGHLIGHTED_TEXT),
        (role.Link, LINK),
        (role.LinkVisited, LINK_VISITED),
        # Fusion's bevels and frames are drawn from these five.
        (role.Light, BORDER),
        (role.Midlight, "#313136"),
        (role.Mid, "#26262a"),
        (role.Dark, "#101012"),
        (role.Shadow, "#000000"),
    ):
        pal.setColor(r, QColor(c))

    dis = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(dis, r, QColor(DISABLED))
    pal.setColor(dis, role.Button, QColor(WINDOW))
    pal.setColor(dis, role.Base, QColor(WINDOW))
    pal.setColor(dis, role.Highlight, QColor(BUTTON))
    pal.setColor(dis, role.HighlightedText, QColor(DISABLED))
    return pal


def apply_dark_theme(app: QApplication) -> None:
    """Fusion + the dark palette + the few rules the palette cannot express."""
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:      # Qt < 6.8: the frame keeps the system look, the rest is dark
        pass
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(_STYLESHEET)
