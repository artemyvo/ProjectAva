"""Ava Chat — UI entry point."""
import argparse
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Ava Chat UI")
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="Inference server WebSocket URL (e.g. ws://192.168.1.10:8765). "
             "Overrides config.json server_url.",
    )
    args = parser.parse_args()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
    app = QApplication(sys.argv)
    window = MainWindow(server_url=args.server)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
