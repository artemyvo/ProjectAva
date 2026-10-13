"""Which adapter answered an exchange, and whether it is one of this lineage's own builds (2026-10-12).

The operator pushes the training data on purpose by answering a chat with a foreign adapter —
the sibling instance's builds copied into ``models/`` on 2026-10-09, loaded through the
Training review's swap (which stays until the model is reloaded): the crocodile chat
(2026-10-08) and the lace-dress chat (2026-10-12). Those replies are meant to train; they are
not her behaviour, and a reading of how she answers (the register monitor, the habit drift's
live view, the governor's live reply, the registered-vs-answered gap) must not count them.

The rule (the operator's, 2026-10-12): an adapter is this lineage's own iff ``models/builds.jsonl``
records it — every promoted build there has its snapshot. Anything else is foreign.

* ``own_adapters(models_dir)`` — the names of the recorded builds (cached on the file's mtime).
* ``serving_adapter(doc, index, hidden=None)`` — the adapter that answered exchange *index*:
  the live capture's ``meta.adapter_id`` when the transcript's ``.hidden.npz`` carries one
  (stamped per exchange, so a swap mid-chat is seen), else the session's ``adapter_id``
  (stamped when the chat started); ``""`` when neither is stamped (early July).
* ``is_own(doc, index, …)`` — True / False, or None when the exchange carries no stamp
  (counted, kept: the earliest transcripts predate stamping and the sibling's builds).

Read-only; imports nothing of the project but ``hidden_capture``'s key convention.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Optional

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_OWN_CACHE: dict = {"key": None, "names": frozenset()}
_META_CACHE: dict = {}


def _name(adapter: str) -> str:
    return Path(str(adapter or "").rstrip("/")).name


def own_adapters(models_dir: Optional[Path] = None) -> frozenset:
    """The adapter directory names ``builds.jsonl`` records (this lineage's builds)."""
    path = Path(models_dir or _MODELS_DIR) / "builds.jsonl"
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return frozenset()
    if _OWN_CACHE["key"] == key:
        return _OWN_CACHE["names"]
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        n = _name(rec.get("adapter_dir") or "")
        if n:
            names.add(n)
    _OWN_CACHE.update(key=key, names=frozenset(names))
    return _OWN_CACHE["names"]


def _live_meta(hidden: Path) -> dict:
    """``{exchange_id: adapter name}`` from the live capture groups' metas (cached on mtime)."""
    try:
        key = (str(hidden), hidden.stat().st_mtime_ns)
    except OSError:
        return {}
    hit = _META_CACHE.get(str(hidden))
    if hit and hit[0] == key:
        return hit[1]
    out: dict = {}
    try:
        import numpy as np
        with zipfile.ZipFile(hidden) as zf:
            for n in zf.namelist():
                group, _, member = n.partition("/")
                if member != "meta.npy" or "~" in group:
                    continue
                try:
                    meta = json.loads(np.load(io.BytesIO(zf.read(n)), allow_pickle=False).item())
                except Exception:
                    continue
                out[group] = _name(meta.get("adapter_id") or "")
    except Exception:
        out = {}
    _META_CACHE[str(hidden)] = (key, out)
    return out


def serving_adapter(doc: dict, index: int, hidden: Optional[Path] = None) -> str:
    """The adapter name that answered exchange *index* of *doc* ("" when unstamped)."""
    exs = (doc or {}).get("exchanges") or []
    if hidden is not None and 0 <= index < len(exs) and isinstance(exs[index], dict):
        ex_id = str(exs[index].get("exchange_id") or f"ex{index}")
        name = _live_meta(Path(hidden)).get(ex_id)
        if name:
            return name
    return _name((doc or {}).get("adapter_id") or "")


def is_own(doc: dict, index: int, hidden: Optional[Path] = None, models_dir: Optional[Path] = None) -> Optional[bool]:
    """Whether exchange *index* was answered by one of this lineage's builds; None if unstamped."""
    name = serving_adapter(doc, index, hidden)
    if not name:
        return None
    return name in own_adapters(models_dir)


def hidden_for(transcript: Path) -> Path:
    """The ``.hidden.npz`` beside a transcript (it may not exist)."""
    p = Path(transcript)
    return p.with_name(p.stem + ".hidden.npz")


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        md = Path(d)
        (md / "builds.jsonl").write_text(json.dumps({"adapter_dir": str(md / "adapter-own")}) + "\n", encoding="utf-8")
        doc = {"adapter_id": str(md / "adapter-own"), "exchanges": [{"exchange_id": "e0"}, {"exchange_id": "e1"}]}
        assert is_own(doc, 0, models_dir=md) is True
        assert is_own({"adapter_id": "/x/adapter-foreign", "exchanges": [{}]}, 0, models_dir=md) is False
        assert is_own({"exchanges": [{}]}, 0, models_dir=md) is None
        # a swap mid-chat: the live capture's meta overrides the session stamp
        import numpy as np
        hz = md / "c.hidden.npz"
        with zipfile.ZipFile(hz, "w") as zf:
            buf = io.BytesIO()
            np.save(buf, np.array(json.dumps({"adapter_id": "/m/adapter-foreign"})))
            zf.writestr("e1/meta.npy", buf.getvalue())
            buf = io.BytesIO()
            np.save(buf, np.array(json.dumps({"adapter_id": ""})))
            zf.writestr("e1~base/meta.npy", buf.getvalue())
        assert serving_adapter(doc, 1, hz) == "adapter-foreign" and is_own(doc, 1, hz, md) is False
        assert serving_adapter(doc, 0, hz) == "adapter-own" and is_own(doc, 0, hz, md) is True
    print("lineage self-test: OK")


if __name__ == "__main__":
    _selftest()
