"""Structured progress log for the offline train cycle.

``train_cycle`` is launched by the watchdog as a subprocess *while the inference
WebSocket server is down* (the base model must be unloaded to free the GPU). The
Sleep tab therefore can't watch training over the normal reflection event stream
— that server is dead for the whole training window. Instead the cycle appends
structured JSONL events here; the watchdog serves them via ``GET /train/progress``
and the Sleep tab polls that endpoint live during training.

The file is a sibling of ``train.log`` (``server/train_progress.jsonl``) so the
watchdog can locate it without importing training internals. It is truncated at
the start of each cycle, so it always describes the *current* (or most recent)
run. Each line is one event::

    {"seq": 1, "ts": 1781809966.3, "run_id": "...", "stage": "render",
     "status": "info", "message": "...", "data": {...}}

``stage`` is a coarse phase (``start``/``render``/``load``/``baseline``/``train``/
``probe``/``save``/``done``); ``status`` is ``info``/``pass``/``fail``/``error``/
``promoted``/``rejected``. Writing never raises — progress is best-effort telemetry
and must never break the cycle it observes.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

# Sibling of train.log (server/train_progress.jsonl). train_cycle runs with
# cwd=server and lives at server/training/, so parent.parent is server/.
PROGRESS_FILE = Path(__file__).resolve().parent.parent / "train_progress.jsonl"


class TrainProgress:
    """Append-only structured progress emitter for one train cycle."""

    def __init__(self, path: Path = PROGRESS_FILE, run_id: Optional[str] = None,
                 reset: bool = True) -> None:
        self._path = Path(path)
        self._run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()
        if reset:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text("", encoding="utf-8")
            except Exception:
                pass

    def emit(self, stage: str, message: str = "", status: str = "info",
             **data) -> None:
        """Append one event. Best-effort: a write failure is swallowed."""
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "ts": time.time(),
                "run_id": self._run_id,
                "stage": stage,
                "status": status,
                "message": message,
            }
            if data:
                rec["data"] = data
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass


def read_events(path: Path = PROGRESS_FILE, after_seq: int = 0) -> list[dict]:
    """Return progress events with ``seq > after_seq`` (oldest first).

    Tolerant of a missing/partially-written file — used by the watchdog's
    ``GET /train/progress`` handler, which may read mid-write.
    """
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if int(rec.get("seq", 0)) > after_seq:
                    out.append(rec)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return out


def last_event(path: Path = PROGRESS_FILE) -> Optional[dict]:
    """The most recent event, or None. Used for one-line status summaries."""
    events = read_events(path, after_seq=0)
    return events[-1] if events else None
