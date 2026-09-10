"""Training review widget — inspect the exact rows fed to Unsloth, and repair them.

A viewer over the latest build's training render
(``server/models/snapshots/<build_id>/sft_render.jsonl`` — the forensic corpus the
current adapter was fitted on). Each line of that file is one training row; the model
is trained on the row's **final** exchange (label masking keeps only the last turn),
so this tab surfaces exactly that: the final user query, the assistant's
chain-of-thought, and the answer.

**The corpus is fetched from the connected server** (``GET /training/review`` on the
inference HTTP sidecar), which is what lets the tab run from a remote UI box: the
projection it serves is a fraction of the render's size and carries no weights, so
reviewing a build no longer requires a Migrate → *Fetch snapshot* pull. With no
connection it falls back to this checkout's own ``server/`` tree, using the same
builder (``inference/core/training_review.py``, loaded by path) so the two can't
diverge. Repairs always target the connected server.

Left: a list of every training entry in file order (== training order), keyed by the
first 20 characters of the reply. A cap-age exchange renders **twice** (the masked row
plus its user-contamination copy — same messages, different LR multiplier / unmask flag),
which is a training-schedule fact, not a second exchange to review: identical rows are
collapsed to one entry tagged ``×N``. Right: three boxes — query (immutable), CoT and answer
(**editable**) — over a **regenerate bar**: pick which part(s) are corrupt (CoT / reply), set a
temperature, optionally type a **system-prompt suffix** that steers only this
regeneration (not the trainable prefix — e.g. away from a reflexive trailing question),
and hit **Regenerate** to have the *currently loaded adapter* re-answer the exchange. An
**Adapter dropdown** beside the temperature can instead pick any adapter from the server's
lineage (the review payload's ``adapters`` list, annotated with its build + whether the
build snapshot survives): the server swaps that adapter in — **sticky**: one full model
reload, narrated into the dialog via ``regenerate_status`` milestones, and it then STAYS
loaded, so a repair session's further regenerations under the same choice reload nothing
(the box keeps running that adapter afterwards until another load; a server restart
returns to the config's) — which is the "repair with the last known GOOD adapter"
path for when the loaded adapter is itself what keeps re-answering badly. A
pop-up opens at once and shows the CoT + reply *as they are generated*, with a **Stop**
button that halts the model and keeps what it wrote; **Apply** writes the repaired target
straight into the chat sidecar on the connected server.

Repair has two routes to the same sidecar write, so a row too broken to regenerate well can
still be fixed by hand: **Regenerate** (the model re-answers) and a **hand edit** — type
straight into the CoT / answer boxes and hit the edit bar's **Apply** (**Cancel** reverts to
the stored content) — the search bar's **Replace** is a power tool for that second route,
rewriting every occurrence of the search phrase in those two boxes (an empty replacement box
deletes it) and leaving the same Apply to persist it. **Replace all…** is its corpus-scale
form for a poison sequence a bad build smeared everywhere: it runs the same substitution over
every entry currently *shown* and writes each result itself, freezing ❄ as it goes, because
per-entry review is a formality when the same three words are being deleted in 400 places.
The search phrase is matched **exactly as typed** — a trailing space counts, which is what
lets `"terms of "` be deleted without leaving a double space behind. Either route writes one reviewed target and **locks (❄ freezes)** the
exchange, so re-reflection and Revisit preserve it. The frozen filter is the workflow's
progress view: *Only unfrozen* is what's left to repair, *Only ❄ frozen* what has been.

Finding the work is the third piece: a **persona-opener detector** scores every entry on how
much its CoT begins by reciting identity rather than engaging the question — the residue of
the retired build-time `[persona]` CoT injection and of the recitation habit it taught. The
score drives a filter, a sort, a ``✦`` list tag, and a **Strip persona opener** button that
prefills the CoT box with the offending leading lines removed, leaving the operator to review
and Apply. See the detection block below for the two tiers and why one of them is exact.

Needs a connection to a server that has run at least one training build; with none, it
falls back to a snapshot in this checkout if one is there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QPainter, QColor, QTextCursor, QTextCharFormat,
    QFontMetrics,
)

if TYPE_CHECKING:
    from ui.chat_widget import ChatWidget


class _StatusLabel(QLabel):
    """A status label whose text can never move the layout around it.

    The tab's status lines are whole sentences ("Applied to sidecar (new CoT + new reply);
    this exchange is now locked … it trains from the LIVE sidecar, not this snapshot."), and
    a plain QLabel reports its full text width as its *minimum* — so writing one after an
    Apply raised the right pane's minimum width and shoved the splitter, i.e. the tab
    visibly reflowed on a status message. This one asks for no width at all, elides to
    whatever the layout gives it, and keeps the full text in its tooltip.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full_text = text

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._elide()

    def text(self) -> str:
        return self._full_text

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        # Height from the real hint, width zero: the label may be squeezed to nothing
        # rather than pushing its siblings (or the splitter) aside.
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, max(0, self.width())))


# ---------------------------------------------------------------------- #
# Corpus fetch                                                            #
# ---------------------------------------------------------------------- #
#
# The entry list is BUILT SERVER-SIDE (`inference/core/training_review.py`): finding the
# latest build's render, projecting each row down to its reviewed final turn, collapsing
# duplicate rows and patching the live sidecars' repaired targets over the frozen copy all
# read files that live next to the model, not next to the UI. Doing it there is what lets
# this tab run from a remote box — and keeps ONE definition of the projection, since the
# same-box path below loads that same file instead of a second copy of the logic.
#
# What stays here is the part that is genuinely the UI's: persona-opener scoring. It is
# pure and lexical, and keeping it client-side is what lets a repaired row be rescored in
# place (`_rescore_entry`) without another round trip.

_REVIEW_PATH = "/training/review"


def _fetch_review_remote(base_url: str, source: str, timeout: float = 300.0) -> dict:
    """GET the entry list from the inference HTTP sidecar (`training/review`)."""
    import gzip
    import urllib.error
    import urllib.request

    url = f"{base_url}{_REVIEW_PATH}?source={source}"
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (json.loads(exc.read().decode("utf-8")) or {}).get("error", "")
        except Exception:
            pass
        # A server predating this endpoint answers with the sidecar's catch-all
        # ("not found"); the endpoint's own 404 always carries a real explanation.
        if exc.code == 404 and detail in ("", "not found"):
            raise RuntimeError(
                f"The server at {base_url} has no {_REVIEW_PATH} endpoint.\n"
                "It is running an older build — git pull + restart it, or open this tab "
                "on a checkout that has a local snapshot."
            ) from exc
        raise RuntimeError(detail or f"HTTP {exc.code} from {url}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach the inference sidecar at {base_url} ({exc}).\n"
            "It serves the training corpus on the sidecar port (default 8767) — check "
            "that the port is reachable from this box."
        ) from exc
    return json.loads(raw.decode("utf-8"))


def _fetch_review_local(server_dir: Path, source: str) -> dict:
    """Build the entry list from this checkout, using the server's own builder.

    Loaded by file path rather than imported as a package: the module is stdlib-only by
    construction, so this pulls in no server dependencies, adds nothing to ``sys.path``,
    and cannot drift from what the sidecar serves."""
    import importlib.util

    mod_path = server_dir / "inference" / "core" / "training_review.py"
    if not mod_path.is_file():
        raise RuntimeError(
            "Not connected, and this checkout has no server/ tree to read a snapshot "
            "from.\nConnect from the Chat tab — the training corpus is served by the "
            "inference sidecar."
        )
    spec = importlib.util.spec_from_file_location("_ava_training_review", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.build_payload(source=source, server_root=server_dir)
    except module.ReviewUnavailable as exc:
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------- #
# Persona-opener detection                                                #
# ---------------------------------------------------------------------- #
#
# Old builds injected a `[persona]` statement VERBATIM as the leading line of its host
# exchange's <think> block (`training/persona_render.prepend_persona_lines`). That path is
# retired, but the rows it produced trained a model that now opens its *own* reasoning by
# reciting identity — and those replies came back through reflection as targets. Both
# populations share one signature: the FIRST line(s) of the CoT are a first-person
# self-statement that does not engage the user's question.
#
# Two tiers, both purely lexical (the client carries no ML dependency):
#   * ledger — the line matches a `[persona]` statement still on disk. The injector copied
#     the statement verbatim, so containment is proof, and `_LEDGER_JACCARD` mirrors
#     `persona_render._DEDUP_JACCARD` (0.6) — the injector's own "already expressed" test,
#     reused here to recognize what it wrote.
#   * shape  — no ledger match, but the line is a long first-person declaration with almost
#     no word overlap with the query: the learned recitation. Weaker evidence, so it is
#     capped (`_SHAPE_CAP`) at the ledger tier's floor and can never outrank a ledger hit.
#
# `fact_render`'s "I know that …" line is CURRENT, deliberate injection — it is skipped, and
# does not break the leading run (a fact line may sit above a persona line).

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_FIRST_PERSON_RE = re.compile(r"^(I|I'm|My|Я|Мне|Моя|Мой|Моё)\b", re.IGNORECASE | re.UNICODE)
_FACT_LINE_RE = re.compile(r"^(I know that|Я знаю[,\s])", re.IGNORECASE | re.UNICODE)

_PERSONA_SCAN_LINES = 3          # the injector's cap was 2; scan one more
_POSITION_WEIGHT = (1.0, 0.85, 0.7)   # a tiebreaker, not a suppressor: line 2 still scores
_LEDGER_JACCARD = 0.6            # == persona_render._DEDUP_JACCARD
_SHAPE_CAP = 0.59                # ceiling of the shape tier, kept UNDER the ledger floor so
                                 # that at equal position a ledger hit always outranks a shape one
_SHAPE_MIN_WORDS = 8
_SHAPE_MAX_QUERY_OVERLAP = 0.25
_PERSONA_MARK = 0.35             # score at/above which the list tags the entry as detected


def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower())}


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _nonempty_lines(cot: str) -> list:
    """The CoT's non-blank lines, paired with their index in ``splitlines()``."""
    return [(i, ln.strip()) for i, ln in enumerate((cot or "").splitlines()) if ln.strip()]


def _index_statements(normalized: list) -> list:
    """The wire form (normalized statement strings) as ``(normalized, word_set)`` pairs.

    The corpus builder ships every `[persona]` statement ever written — the RAW
    append-only logs, evicted ones included, since an injection happened while its
    statement was live — already normalized; the word sets are the detector's own
    working form."""
    out = []
    for n in normalized or ():
        if isinstance(n, (list, tuple)):     # already indexed (a rescore round-trip)
            out.append((n[0], set(n[1])))
        elif n:
            out.append((n, set(n.split())))
    return out


def _ledger_similarity(line: str, statements: list) -> float:
    """How strongly *line* matches any stored persona statement, in ``[0, 1]``.

    1.0 when a statement is contained in the line (what verbatim injection produces);
    otherwise the best word-set Jaccard — the same lexical test the injector used to decide
    a statement was already expressed in a thought."""
    if not statements:
        return 0.0
    ln_norm = _norm(line)
    if not ln_norm:
        return 0.0
    ln_words = set(ln_norm.split())
    best = 0.0
    for s_norm, s_words in statements:
        if s_norm in ln_norm:
            return 1.0
        union = len(s_words | ln_words)
        if union:
            j = len(s_words & ln_words) / union
            if j > best:
                best = j
    return best


def _line_persona_value(line: str, query_words: set, statements: list) -> tuple:
    """``(value, kind)`` for one CoT line — ``kind`` is "ledger", "shape" or "" (neither).

    A ledger match scores its similarity (>= `_LEDGER_JACCARD`); a shape match scores
    `_SHAPE_CAP` scaled by how detached the line is from the question, so it always lands at
    or below the ledger floor."""
    sim = _ledger_similarity(line, statements)
    if sim >= _LEDGER_JACCARD:
        return sim, "ledger"
    lw = _words(line)
    if len(lw) < _SHAPE_MIN_WORDS or not _FIRST_PERSON_RE.match(line):
        return 0.0, ""
    overlap = (len(lw & query_words) / len(lw)) if lw else 1.0
    if overlap >= _SHAPE_MAX_QUERY_OVERLAP:
        return 0.0, ""
    return _SHAPE_CAP * (1.0 - overlap), "shape"


def _score_persona_opener(query: str, cot: str, statements: list) -> dict:
    """Rank how much *cot* opens by reciting persona, and what could be stripped.

    Returns ``{score, kind, line, matched, strip}``. ``strip`` is the list of
    ``splitlines()`` indices forming the **contiguous leading run** of persona lines — the
    only shape that is mechanically removable, because the injector *prepended* onto an
    existing thought, so dropping that run restores the pre-injection CoT verbatim. A hit
    further down (with real reasoning above it) yields an empty ``strip``: that one is
    entangled and belongs to the hand editor, not a button."""
    lines = _nonempty_lines(cot)[:_PERSONA_SCAN_LINES]
    if not lines:
        return {"score": 0.0, "kind": "", "line": -1, "matched": "", "strip": []}
    qw = _words(query)
    best = (0.0, "", -1, "")
    strip: list = []
    run_open = True
    for pos, (raw_idx, text) in enumerate(lines):
        if _FACT_LINE_RE.match(text):
            continue                      # deliberate fact injection: not persona, not a barrier
        value, kind = _line_persona_value(text, qw, statements)
        if value <= 0.0:
            run_open = False
            continue
        if run_open:
            strip.append(raw_idx)
        weighted = value * _POSITION_WEIGHT[min(pos, len(_POSITION_WEIGHT) - 1)]
        if weighted > best[0]:
            best = (weighted, kind, pos, text)
    return {"score": round(best[0], 3), "kind": best[1], "line": best[2],
            "matched": best[3], "strip": strip}


def _strip_lines(cot: str, indices: list) -> str:
    """*cot* with the given ``splitlines()`` indices removed (leading blanks trimmed)."""
    drop = set(indices or ())
    kept = [ln for i, ln in enumerate((cot or "").splitlines()) if i not in drop]
    while kept and not kept[0].strip():
        kept.pop(0)
    return "\n".join(kept).strip()


class TrainingLoadWorker(QThread):
    """Fetches the training corpus and scores it off the GUI thread.

    The corpus comes from the connected server's inference sidecar when there is a
    connection — that is the remote-operation path, and it needs no snapshot on this box —
    and otherwise from this checkout's own server/ tree (same builder, loaded by path).
    Either way the payload arrives already projected; what happens here is the
    persona-opener scoring, which is where the per-entry work actually is."""

    loaded = pyqtSignal(dict)   # {build_id, path, origin, entries:[…], persona_statements}
    failed = pyqtSignal(str)

    def __init__(self, server_dir: Path, base_url: Optional[str] = None,
                 source: str = "snapshot"):
        super().__init__()
        self._server_dir = server_dir
        self._base_url = base_url
        self._source = source

    def run(self) -> None:
        try:
            if self._base_url:
                payload = _fetch_review_remote(self._base_url, self._source)
                origin = self._base_url
            else:
                payload = _fetch_review_local(self._server_dir, self._source)
                origin = "this checkout"
            entries = payload.get("entries") or []
            # Indexed once, reused for every entry's persona-opener score (and handed to
            # the widget so a repaired row can be rescored without a reload).
            statements = _index_statements(payload.get("persona_statements"))
            for e in entries:
                # Score the DISPLAYED CoT (the live target for a repaired row), and only
                # for entries traceable to a chat exchange — a wander row's "I just read…"
                # opener is legitimate, not injected persona.
                traceable = bool(e.get("source_session")) and e.get("exchange_index") is not None
                disp_cot = e.get("live_cot") if e.get("has_live") else e.get("cot")
                e["persona"] = (_score_persona_opener(e.get("query", ""), disp_cot or "",
                                                      statements)
                                if traceable
                                else {"score": 0.0, "kind": "", "line": -1,
                                      "matched": "", "strip": []})
            self.loaded.emit({
                "build_id": payload.get("build_id", "?"),
                # "preview" = a training-lite preview snapshot: NOTHING in it trained.
                "outcome": payload.get("outcome", ""),
                "path": payload.get("path", ""),
                "origin": origin,
                "entries": entries,
                "rows": payload.get("rows", len(entries)),
                "persona_statements": statements,
                # The server's adapter lineage — the regenerate bar's Adapter dropdown
                # (regenerate with a previous, last-known-good adapter).
                "adapters": payload.get("adapters") or [],
            })
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class RegenerateWorker(QThread):
    """Ask the server to re-answer one exchange with the loaded adapter (off the GUI thread).

    Streams: `delta` carries the (cot, reply) text produced so far this step, so the review
    dialog fills in as the model writes instead of appearing minutes later fully formed.
    `status` carries the server's swap milestones when an alternate `adapter` was chosen
    (loading/restoring an adapter is minutes of otherwise-total silence in the dialog).
    `done` carries the server's authoritative final payload."""

    delta = pyqtSignal(str, str)  # (cot_delta, reply_delta)
    status = pyqtSignal(str)      # adapter-swap milestone text
    done = pyqtSignal(dict)       # server's exchange_regenerated payload
    failed = pyqtSignal(str)

    def __init__(self, client, filename: str, exchange_index: int, temperature: float,
                 system_suffix: str = "", cot_only: bool = False, adapter: str = ""):
        super().__init__()
        self._client = client
        self._filename = filename
        self._exchange_index = exchange_index
        self._temperature = temperature
        self._system_suffix = system_suffix
        self._cot_only = cot_only
        self._adapter = adapter

    def stop(self) -> None:
        """Ask the server to halt the generation and return what it has.

        Sent straight down the socket (outside the RPC lock this worker holds), so it lands
        while the generation is still running; the stream then terminates normally with a
        partial payload."""
        try:
            self._client.cancel()
        except Exception:
            pass

    def run(self) -> None:
        try:
            for kind, payload in self._client.stream_regenerate_exchange(
                self._filename, self._exchange_index, temperature=self._temperature,
                system_suffix=self._system_suffix, cot_only=self._cot_only,
                adapter=self._adapter):
                if kind == "chunk":
                    self.delta.emit(payload.get("cot", ""), payload.get("reply", ""))
                elif kind == "status":
                    self.status.emit(payload.get("text", ""))
                elif kind == "done":
                    self.done.emit(payload)
                    return
                else:
                    self.failed.emit(payload.get("message", "Regeneration failed."))
                    return
            self.failed.emit("Regeneration ended without a result.")
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class ApplyRegenWorker(QThread):
    """Persist a reviewed regeneration to the sidecar (off the GUI thread)."""

    done = pyqtSignal(dict)      # server's exchange_regenerated_applied payload
    failed = pyqtSignal(str)

    def __init__(self, client, filename: str, exchange_index: int, *,
                 corrupt_cot: bool, corrupt_response: bool,
                 new_cot: str, new_reply: str, original_reply: str = ""):
        super().__init__()
        self._client = client
        self._filename = filename
        self._exchange_index = exchange_index
        self._corrupt_cot = corrupt_cot
        self._corrupt_response = corrupt_response
        self._new_cot = new_cot
        self._new_reply = new_reply
        self._original_reply = original_reply

    def run(self) -> None:
        try:
            result = self._client.apply_regenerated_exchange(
                self._filename, self._exchange_index,
                corrupt_cot=self._corrupt_cot, corrupt_response=self._corrupt_response,
                new_cot=self._new_cot, new_reply=self._new_reply,
                original_reply=self._original_reply)
            if result.get("type") == "error":
                self.failed.emit(result.get("message", "Apply failed."))
                return
            self.done.emit(result)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class BulkReplaceWorker(QThread):
    """Apply a prepared bulk search-and-replace, one sidecar write per exchange.

    Deliberately the SAME per-exchange RPC the single Apply uses
    (``apply_regenerated_exchange``), looped off the GUI thread, rather than a new bulk
    server call: each write is one reviewed target that locks ❄ its exchange, and going
    through the existing path means a bulk repair and a hand repair cannot end up meaning
    different things on disk. The cost is N round trips, which is why this reports progress
    and can be stopped between writes — a half-finished run leaves the exchanges it did
    reach repaired and frozen, and the rest untouched (each write is independent).
    """

    progress = pyqtSignal(int, int, dict)   # attempted, total, {order, cot, reply} ({} on failure)
    done = pyqtSignal(int, list, bool)      # applied, [(order, message)], stopped

    def __init__(self, client, jobs: list):
        super().__init__()
        self._client = client
        self._jobs = jobs
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        applied = 0
        failures: list = []
        total = len(self._jobs)
        for i, job in enumerate(self._jobs):
            if self._stop:
                break
            payload: dict = {}
            try:
                result = self._client.apply_regenerated_exchange(
                    job["filename"], job["exchange_index"],
                    corrupt_cot=True, corrupt_response=True,
                    new_cot=job["new_cot"], new_reply=job["new_reply"],
                    original_reply=job["new_reply"])
                if result.get("type") == "error":
                    failures.append((job["order"], result.get("message", "Apply failed.")))
                else:
                    applied += 1
                    payload = {
                        "order": job["order"],
                        "cot": str(result.get("new_cot") or job["new_cot"]),
                        "reply": str(result.get("new_reply") or job["new_reply"]),
                    }
            except Exception as exc:  # pragma: no cover - defensive
                failures.append((job["order"], str(exc)))
            self.progress.emit(i + 1, total, payload)
        self.done.emit(applied, failures, self._stop)


class BanWorker(QThread):
    """Set/clear one row's training ban on the server (off the GUI thread)."""

    done = pyqtSignal(dict)      # server's training_ban_set payload
    failed = pyqtSignal(str)

    def __init__(self, client, target: str, exchange_index, banned: bool):
        super().__init__()
        self._client = client
        self._target = target
        self._exchange_index = exchange_index
        self._banned = banned

    def run(self) -> None:
        try:
            result = self._client.set_training_ban(
                self._target, self._exchange_index, self._banned)
            if result.get("type") == "error":
                self.failed.emit(result.get("message", "Ban failed."))
                return
            self.done.emit(result)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class RewriteHistoryWorker(QThread):
    """Ask the server to bake all locked targets into their transcripts (off the GUI thread)."""

    done = pyqtSignal(dict)      # server's history_rewritten payload
    failed = pyqtSignal(str)

    def __init__(self, client):
        super().__init__()
        self._client = client

    def run(self) -> None:
        try:
            result = self._client.rewrite_history()
            if result.get("type") == "error":
                self.failed.emit(result.get("message", "Rewrite history failed."))
                return
            self.done.emit(result)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class RegenReviewDialog(QDialog):
    """Modal review of a fresh re-answer, opened AT the regeneration and filled live.

    It appears the moment Regenerate is pressed — a re-answer takes GPU minutes, and an
    operator cleaning a dataset needs to see the thought forming (and abandon a bad one)
    rather than stare at a frozen tab until a finished result pops up. `append_delta`
    feeds it, `finish` swaps in the server's authoritative text, `fail` reports an error.

    Always shows the new CoT. The second box reflects what Apply will actually write:
    for a reply regen it is the fresh reply; for a CoT-only regen it is the KEPT reply
    (unchanged), so the operator judges whether the new CoT coheres with the reply it will
    be grafted onto — not the discarded regeneration. For a CoT-only regen the streaming
    deltas therefore go to the CoT box only (the server halts at the reasoning close
    anyway, so nothing streams to the answer side).

    Buttons are stateful: while generating, **Stop** halts the model and keeps whatever it
    produced (a partial CoT is still applicable, and hand-editable afterwards) and Cancel
    abandons; once finished, Apply / Cancel behave as before. Apply is armed only when the
    generation has ended, so a half-written thought can't be persisted by a stray click."""

    stop_requested = pyqtSignal()

    def __init__(self, parent, *, kept_reply: str, corrupt_cot: bool,
                 corrupt_response: bool, font: QFont, state: Optional[dict] = None,
                 adapter: str = ""):
        super().__init__(parent)
        self._corrupt_response = corrupt_response
        self._saved_state: Optional[dict] = None
        self._finished = False
        self.setWindowTitle("Regenerating exchange…")
        self.setModal(True)
        self.resize(720, 560)
        layout = QVBoxLayout(self)

        if corrupt_response:
            self._note = "Apply will write: new CoT + new reply (a fresh, faithful pair)."
            reply_label = "New reply"
        else:
            self._note = ("Apply will write: new CoT grafted onto the KEPT reply below "
                          "(the reply is not changed; the regenerated reply is discarded).")
            reply_label = "Reply (kept — unchanged)"
        # `adapter` names an ALTERNATE lineage adapter chosen for this re-answer. The
        # swap is STICKY server-side (`agentic.swap_model`): the first regeneration
        # with it pays one full model reload and it then stays loaded, so the client
        # cannot know whether THIS press needs a reload — the opening note is phrased
        # conditionally, and the server's `regenerate_status` milestones (set_status)
        # narrate the reload when one actually happens.
        if adapter:
            self._note = (f"Answered by adapter {adapter} — it STAYS loaded for "
                          "further regenerations (reload from the Chat tab to go "
                          "back). " + self._note)
            opening = (f"Making sure adapter {adapter} is loaded (a full model reload "
                       "if it isn't already — it then stays loaded), then generating…")
        else:
            opening = "Generating with the loaded adapter… (this runs on the GPU)"
        self._lbl_note = QLabel(opening)
        self._lbl_note.setWordWrap(True)
        layout.addWidget(self._lbl_note)

        self._split = QSplitter(Qt.Orientation.Vertical)
        self._cot_box = self._make_box("New chain of thought (CoT)", "", font)
        # A CoT-only regen never streams an answer, so the kept reply can be shown straight
        # away as the context the new thought has to cohere with.
        self._reply_box = self._make_box(
            reply_label, "" if corrupt_response else kept_reply, font)
        self._split.addWidget(self._cot_group)
        self._split.addWidget(self._reply_group)
        layout.addWidget(self._split, 1)

        buttons = QDialogButtonBox()
        self.btn_stop = buttons.addButton("Stop", QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_apply = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_stop.setToolTip("Halt the model and keep what it has written so far.")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_apply.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._restore_state(state)

    # -- window geometry, carried across regenerations ------------------ #
    #
    # A repair session is a run of regenerations, and the operator sizes this window (and
    # the CoT/reply divider) for the material they are reading. Rebuilding the dialog per
    # regeneration threw that away and snapped back to 720x560 centred, so the caller hands
    # the last dialog's state back in here.

    def _restore_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        geometry = state.get("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        sizes = state.get("split")
        if sizes:
            self._split.setSizes(list(sizes))

    def save_state(self) -> dict:
        """The dialog's geometry + splitter sizes, for the next regeneration to restore.

        Cached on `done()`: by the time the caller asks, `exec()` has returned and the dialog
        is already hidden, so reading it live would report the pre-close geometry on some
        platforms."""
        if self._saved_state is not None:
            return self._saved_state
        return {"geometry": self.saveGeometry(), "split": self._split.sizes()}

    def done(self, result: int) -> None:  # noqa: N802 (Qt naming)
        self._saved_state = {"geometry": self.saveGeometry(), "split": self._split.sizes()}
        super().done(result)

    def _make_box(self, label: str, text: str, font: QFont) -> QPlainTextEdit:
        group = QWidget()
        gl = QVBoxLayout(group)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.addWidget(QLabel(label))
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setFont(font)
        box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        box.setPlainText(text)
        gl.addWidget(box, 1)
        if label.startswith("New chain"):
            self._cot_group = group
        else:
            self._reply_group = group
        return box

    @staticmethod
    def _append(box: QPlainTextEdit, text: str) -> None:
        """Append at the end and follow it, without disturbing a manual scroll-back."""
        if not text:
            return
        bar = box.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        cursor = box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if follow:
            bar.setValue(bar.maximum())

    def append_delta(self, cot: str, reply: str) -> None:
        self._append(self._cot_box, cot)
        # A CoT-only regen shows the kept reply, not the one being generated and discarded.
        if self._corrupt_response:
            self._append(self._reply_box, reply)

    def _on_stop(self) -> None:
        self.btn_stop.setEnabled(False)
        self._lbl_note.setText("Stopping — waiting for the model to halt…")
        self.stop_requested.emit()

    def set_status(self, text: str) -> None:
        """A server-side swap milestone (adapter loading/restoring) — progress, not text.

        Shown right up to `finish`/`fail` (which get the last word): after a Stop the
        milestone that matters most is exactly this one — 'restoring the loaded
        adapter…' is why a stopped generation still takes minutes to come back."""
        if text and self._finished is False:
            self._lbl_note.setText(text)

    def finish(self, result: dict) -> None:
        """Generation ended: swap in the authoritative text and arm Apply."""
        self._finished = True
        self.btn_stop.setEnabled(False)
        self.setWindowTitle("Review regenerated exchange")
        # The streamed text is an optimistic split of the raw generation; the payload is the
        # cleaned one that Apply will actually write, so it gets the last word.
        self._cot_box.setPlainText(str(result.get("new_cot") or ""))
        if self._corrupt_response:
            self._reply_box.setPlainText(str(result.get("new_reply") or ""))
        note = self._note
        if result.get("cancelled"):
            note = "STOPPED early — this is a partial generation. " + note
        self._lbl_note.setText(note)
        self.btn_apply.setEnabled(True)
        self.btn_apply.setDefault(True)

    def fail(self, message: str) -> None:
        self._finished = True
        self.btn_stop.setEnabled(False)
        self.setWindowTitle("Regeneration failed")
        self._lbl_note.setText(f"Regenerate failed: {message}")


class TrainingReviewWidget(QWidget):
    """Training review tab — inspect the latest snapshot's training rows and repair them."""

    def __init__(self, chat_widget: "ChatWidget", parent=None):
        super().__init__(parent)
        self._chat_widget = chat_widget
        self.text_font = QFont("Courier")
        self._worker: Optional[TrainingLoadWorker] = None
        self._regen_worker: Optional[RegenerateWorker] = None
        self._apply_worker: Optional[ApplyRegenWorker] = None
        self._rewrite_worker: Optional[RewriteHistoryWorker] = None
        # Bulk search-and-replace (Replace all). Its own worker + in-flight flag, and while it
        # runs the controls that could swap `self._entries` under it are disabled — progress
        # patches entries by their `_order` stamp, which a Refresh would restamp.
        self._bulk_worker: Optional[BulkReplaceWorker] = None
        self._bulk_in_flight: bool = False
        self._ban_worker: Optional[BanWorker] = None
        self._ban_pending_row: Optional[int] = None
        # Explicit in-flight flag rather than `_ban_worker.isRunning()`, for the same reason
        # as `_apply_in_flight`: the done/failed signal reaches the GUI thread while the
        # worker is still winding down, and a stale "still running" read would leave the Ban
        # button disabled after a successful ban.
        self._ban_in_flight: bool = False
        self._pending: Optional[dict] = None   # context for the in-flight regenerate→apply
        # The live regenerate dialog (None while none is open) and the server's terminal
        # payload for it. The dialog is opened at the start of the generation and streamed
        # into, so the result arrives while it is already on screen.
        self._regen_dlg: Optional["RegenReviewDialog"] = None
        self._regen_result: Optional[dict] = None
        # Last regenerate dialog's geometry + CoT/reply divider, replayed into the next one
        # so a run of regenerations keeps the window the operator sized (see
        # `RegenReviewDialog.save_state`).
        self._regen_dlg_state: Optional[dict] = None
        self._entries: list = []
        self._ratio_active: bool = False
        self._ratio_threshold: float = 0.3
        # Frozen filter: "all" | "frozen" (only human-validated ❄ rows) | "unfrozen"
        # (what is still left to repair). Composes with the ratio filter in `_apply_hiding`.
        self._frozen_filter: str = "all"
        # Search filter: the lowercased phrase currently typed into the search box ("" when
        # empty). A non-empty phrase HIDES every entry not containing it (query / CoT /
        # reply), composing with the other filters in `_apply_hiding`; the Search button
        # then cycles through the survivors.
        self._search_needle: str = ""
        # Yellow highlight painted under every occurrence of the search phrase in the three
        # preview boxes. The foreground is pinned too, so the text stays readable whatever
        # the palette's default text colour is (a light-on-yellow theme would wash out).
        self._search_hl_fmt = QTextCharFormat()
        self._search_hl_fmt.setBackground(QColor("#ffe66b"))
        self._search_hl_fmt.setForeground(QColor("#000000"))
        # Persona-opener filter: show only entries scoring at/above the threshold.
        self._persona_active: bool = False
        self._persona_threshold: float = 0.6
        self._persona_statements: list = []
        # Hand-edit state. `_edit_order` is the `_order` stamp of the entry carrying unsaved
        # box edits (None when clean) — keyed by the entry, not the row, so a re-sort can't
        # strand or silently drop them; `_edit_baseline` is the (cot, answer) the boxes were
        # populated with, i.e. what Cancel reverts to and what "dirty" is measured against.
        self._edit_order: Optional[int] = None
        self._edit_baseline: tuple = ("", "")
        self._suppress_edit_signals: bool = False
        # Explicit in-flight flag rather than `_apply_worker.isRunning()`: the done/failed
        # signal can reach the GUI thread while the worker thread is still winding down, and
        # a stale "still running" read would leave Apply disabled.
        self._apply_in_flight: bool = False
        # Sort state: "order" == file/training order (each entry's stamped `_order`);
        # "reply_len" / "persona" sort by reply length / persona-opener score, with
        # `_sort_desc` the direction the sort combo picks outright.
        self._sort_mode: str = "order"
        self._sort_desc: bool = True
        # List icons: a snowflake for frozen (locked / human-validated) entries, a ban glyph
        # for rows barred from training, and a transparent placeholder of the same size for
        # the rest, so text stays aligned. Banned outranks frozen on a row carrying both —
        # a repaired-then-banned exchange is not going to train either way, and "won't
        # train" is the fact an operator scanning the list needs.
        self._frost_icon = self._make_frost_icon()
        self._ban_icon = self._make_ban_icon()
        self._blank_icon = self._make_blank_icon()
        self._build_ui()

    # ---------------------------------------------------------------- #
    # Icons + display helpers                                           #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _make_frost_icon(size: int = 16) -> QIcon:
        """A small snowflake, drawn from the ❄ glyph so no image asset is needed."""
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        f = p.font()
        f.setPointSize(int(size * 0.72))
        p.setFont(f)
        p.setPen(QColor(90, 160, 220))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "❄")
        p.end()
        return QIcon(pm)

    @staticmethod
    def _make_ban_icon(size: int = 16) -> QIcon:
        """A small 🚫, drawn from the glyph like the snowflake so no asset is needed."""
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        f = p.font()
        f.setPointSize(int(size * 0.72))
        p.setFont(f)
        p.setPen(QColor(200, 90, 90))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "🚫")
        p.end()
        return QIcon(pm)

    @staticmethod
    def _make_blank_icon(size: int = 16) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        return QIcon(pm)

    @staticmethod
    def _persona_score(e: dict) -> float:
        return float((e.get("persona") or {}).get("score") or 0.0)

    @classmethod
    def _item_text(cls, row: int, e: dict) -> str:
        """The left-list label: position, reply key, how many rows the exchange collapses
        (cap-age contamination copy), and — when it scores — the persona-opener tag."""
        copies = e.get("copies", 1)
        suffix = f"  ×{copies}" if copies > 1 else ""
        score = cls._persona_score(e)
        if score >= _PERSONA_MARK:
            suffix += f"  ✦{score:.2f}"
        # A fresh preview row (Sleep → "Include fresh chats"): in the snapshot render at
        # LR multiplier 0 so it can be repaired early — this build did NOT train it.
        if e.get("preview"):
            suffix += "  ▷fresh"
        prefix = "wander " if e.get("kind") == "wander" else ""
        return f"{row + 1:>4}. {prefix}{e['key']}{suffix}"

    def _row_icon(self, e: dict) -> QIcon:
        if e.get("banned"):
            return self._ban_icon
        return self._frost_icon if e.get("locked") else self._blank_icon

    @staticmethod
    def _can_ban(e: Optional[dict]) -> bool:
        """A row can be banned when it can be traced back to a source to flag.

        Deliberately wider than `_has_provenance` (which gates repair): a **wander** row has
        no chat exchange to re-answer or hand-edit, so every repair control is dead for it —
        yet a malformed wander is exactly the row that most needs taking out of the corpus.
        Its capture record in the wander corpus is enough to ban, so banning asks only for a
        `source_session`, plus an integer index when it is a chat exchange."""
        if e is None or not e.get("source_session"):
            return False
        if e.get("kind") == "wander":
            return True
        return isinstance(e.get("exchange_index"), int)

    @staticmethod
    def _disp_cot(e: dict) -> str:
        """The CoT to show: the live sidecar target for a frozen/regenerated row, else the
        snapshot render's."""
        return e["live_cot"] if e.get("has_live") else e["cot"]

    @staticmethod
    def _disp_answer(e: dict) -> str:
        return e["live_answer"] if e.get("has_live") else e["answer"]

    @classmethod
    def _haystack(cls, e: dict) -> str:
        """The lowercased query + CoT + reply the search filter matches against.

        Cached on the entry: the filter runs over every entry on each keystroke, and the
        three fields together are kilobytes per row. A repair that rewrites the displayed
        content drops the cache (`_invalidate_haystack`)."""
        hay = e.get("_hay")
        if hay is None:
            hay = f"{e['query']}\n{cls._disp_cot(e)}\n{cls._disp_answer(e)}".lower()
            e["_hay"] = hay
        return hay

    @staticmethod
    def _invalidate_haystack(e: dict) -> None:
        e.pop("_hay", None)

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        # The controls live on TWO compact rows (status/actions, then search+filter+sort)
        # rather than one bar each: the entry list and the three content boxes are what the
        # operator actually reads, so the chrome above them is kept to a minimum. Labels that
        # only restate a control's placeholder/tooltip are dropped for the same reason.
        outer = QVBoxLayout(self)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        # Status/feedback labels elide instead of demanding their text's width — see
        # `_StatusLabel`: a long line here used to reflow the tab.
        self._lbl_status = _StatusLabel(
            "Training review — press Refresh to load the latest snapshot.")
        self._lbl_search = _StatusLabel("")
        self._lbl_filter = _StatusLabel("")
        self.btn_rewrite = QPushButton("Rewrite history…")
        self.btn_rewrite.setToolTip(
            "Bake every human-reviewed (❄ frozen) exchange's reviewed answer into its chat "
            "transcript, drop its stale tension data, and unfreeze it. Edits the original "
            "transcripts (ground truth) on the server.")
        self.btn_rewrite.clicked.connect(self._on_rewrite_history)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._lbl_status, 1)
        # The search / filter feedback rides the status row, so the control row below carries
        # only controls.
        header.addWidget(self._lbl_search)
        header.addWidget(self._lbl_filter)
        header.addWidget(self.btn_rewrite)
        header.addWidget(self.btn_refresh)
        outer.addLayout(header)

        # One control row: search, the four composing filters, and sort.
        #
        # Search FILTERS as it is typed: a non-empty phrase leaves only the entries whose
        # query / CoT / reply contains it (case-insensitive), composing with the ratio /
        # frozen / persona filters. The Search button then cycles through those survivors,
        # wrapping around. Rows are only ever hidden, never removed, so entry numbering and
        # the source-exchange index mapping stay intact.
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.edt_search = QLineEdit()
        self.edt_search.setPlaceholderText("search query / CoT / reply")
        self.edt_search.setClearButtonEnabled(True)
        self.edt_search.setToolTip(
            "Filter the entry list to exchanges containing this phrase (query, CoT or "
            "reply; case-insensitive). Press Enter / Search to step through the matches.\n\n"
            "Matched exactly as typed — leading/trailing spaces count, so a phrase can be "
            "replaced together with the space beside it.")
        self.edt_search.textChanged.connect(self._on_search_text_changed)
        self.edt_search.returnPressed.connect(self._on_search)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self._on_search)
        controls.addWidget(self.edt_search, 1)
        controls.addWidget(self.btn_search)

        # Replace — the search phrase's write counterpart, scoped to the SELECTED entry's
        # editable CoT / answer boxes. A **prefill, never a write** (the same discipline as
        # Strip persona opener): it marks the entry dirty and leaves the edit bar's Apply to
        # be what actually reaches the sidecar. Deliberately one entry at a time rather than
        # a corpus-wide sweep — each entry's repair is a sidecar write that freezes ❄ an
        # exchange, so a sweep would be dozens of unreviewed ones. An EMPTY replacement box
        # deletes the phrase, which is the common case (a stray marker, a leaked template
        # line) and needs no separate control.
        self.edt_replace = QLineEdit()
        self.edt_replace.setPlaceholderText("replace with (blank = delete)")
        self.edt_replace.setClearButtonEnabled(True)
        self.edt_replace.setToolTip(
            "Replace every occurrence of the search phrase in THIS entry's CoT + answer "
            "boxes (case-insensitive, like the search). Leave this box empty to delete the "
            "phrase instead.\n\nThis only fills the boxes — review them, then Apply (which "
            "writes the target and freezes ❄ the exchange). The query box is immutable and "
            "is never touched.")
        self.edt_replace.returnPressed.connect(self._on_replace)
        self.btn_replace = QPushButton("Replace")
        self.btn_replace.setToolTip(self.edt_replace.toolTip())
        self.btn_replace.clicked.connect(self._on_replace)
        controls.addWidget(self.edt_replace, 1)
        controls.addWidget(self.btn_replace)
        # Replace all — the same substitution across every CURRENTLY SHOWN entry, and the one
        # control here that WRITES: a poison sequence smeared across the corpus by a bad build
        # is not a row-at-a-time repair, and the per-entry review Replace leaves to Apply is
        # not review at all when the same three words are being deleted 400 times. So this
        # applies each result straight to its sidecar and **freezes ❄ the exchange**, exactly
        # as Apply does — confirm-first, since that is a few hundred writes.
        self.btn_replace_all = QPushButton("Replace all…")
        self.btn_replace_all.setToolTip(
            "Replace the search phrase across every entry currently SHOWN (the filters above "
            "choose the scope), writing each result to its chat sidecar and freezing ❄ the "
            "exchange — no per-entry Apply.\n\nFor a poison sequence smeared across the "
            "corpus by a bad build. Read-only rows (wander / pre-provenance) are skipped and "
            "reported. Transcripts are untouched — that is “Rewrite history”.")
        self.btn_replace_all.clicked.connect(self._on_replace_all)
        controls.addWidget(self.btn_replace_all)

        # Ratio filter — hide "normal" rows and keep only the runaway replies. A row is
        # hidden when its CoT length is at least `ratio` of its reply length (raw character
        # counts, no tokenization); the survivors are the tiny-CoT / huge-reply exchanges
        # left by the missing-<eos> training bug (and the IDEALs / long replies it spawned).
        controls.addWidget(QLabel("CoT/reply <"))
        self.edt_ratio = QLineEdit("0.3")
        self.edt_ratio.setFixedWidth(50)
        self.edt_ratio.setToolTip(
            "CoT/reply ratio filter: keep only entries whose CoT is shorter than this "
            "fraction of the reply (press Filter to apply).")
        self.edt_ratio.returnPressed.connect(self._on_filter)
        controls.addWidget(self.edt_ratio)
        # Frozen filter — the repair workflow's progress view: "Only unfrozen" is the
        # remaining work, "Only ❄ frozen" what has already been reviewed. Applies
        # immediately (no Filter press) and composes with the ratio filter. Self-describing
        # items, so it needs no label of its own.
        self.cmb_frozen = QComboBox()
        self.cmb_frozen.addItem("All entries", "all")
        self.cmb_frozen.addItem("Only ❄ frozen", "frozen")
        self.cmb_frozen.addItem("Only unfrozen", "unfrozen")
        self.cmb_frozen.addItem("Only 🚫 banned", "banned")
        self.cmb_frozen.addItem("Only ▷ fresh", "fresh")
        self.cmb_frozen.setToolTip(
            "Show only human-validated (❄ frozen) entries, only the ones still awaiting "
            "review, only the ones banned (🚫) from training, only the fresh preview "
            "rows (▷ — rendered untrained, at LR 0, so they can be repaired before they "
            "age into a real build), or all of them.\n"
            "Note that “Only unfrozen” still includes banned rows — a ban is a verdict on "
            "the row, not a review of its content.")
        self.cmb_frozen.currentIndexChanged.connect(self._on_frozen_filter)
        controls.addWidget(self.cmb_frozen)
        # Persona-opener filter — surfaces CoTs that begin by reciting identity instead of
        # engaging the question (see the detection block at module scope). Its own toggle, so
        # it is independent of the ratio filter's Filter button.
        self.chk_persona = QCheckBox("✦ persona ≥")
        persona_tip = (
            "Show only entries whose CoT opens by reciting persona.\n"
            "≥ 0.6 — the ledger tier: the line matches a stored [persona] statement (the "
            "retired build-time injection and near-verbatim derivations of it). Small and "
            "high-confidence.\n"
            "0.35–0.6 — the shape tier: no stored match, but a long first-person opener "
            "with almost no word overlap with the question — the recitation habit learned "
            "from those rows. Large; expect to judge each one.")
        self.chk_persona.setToolTip(persona_tip)
        self.chk_persona.toggled.connect(self._on_persona_filter)
        controls.addWidget(self.chk_persona)
        self.edt_persona = QLineEdit("0.6")
        self.edt_persona.setFixedWidth(50)
        self.edt_persona.setToolTip(persona_tip)
        self.edt_persona.returnPressed.connect(self._on_persona_filter)
        controls.addWidget(self.edt_persona)
        self.btn_filter = QPushButton("Filter")
        self.btn_filter.clicked.connect(self._on_filter)
        self.btn_clear_filter = QPushButton("Clear filters")
        self.btn_clear_filter.clicked.connect(self._on_clear_filter)
        controls.addWidget(self.btn_filter)
        controls.addWidget(self.btn_clear_filter)

        # Sort — one combo instead of three buttons, so direction is picked outright rather
        # than by re-pressing. Sorting reorders `self._entries` and the list widget together,
        # so the row→entry mapping every other control relies on stays intact.
        self.cmb_sort = QComboBox()
        self.cmb_sort.addItem("Sort: training order", ("order", True))
        self.cmb_sort.addItem("Sort: reply length ↓", ("reply_len", True))
        self.cmb_sort.addItem("Sort: reply length ↑", ("reply_len", False))
        self.cmb_sort.addItem("Sort: persona score ↓", ("persona", True))
        self.cmb_sort.addItem("Sort: persona score ↑", ("persona", False))
        self.cmb_sort.setToolTip(
            "Order the entry list: file/training order (by exchange index), by reply length, "
            "or by the persona-opener score — worst offenders first.")
        self.cmb_sort.currentIndexChanged.connect(self._on_sort_changed)
        controls.addWidget(self.cmb_sort)
        outer.addLayout(controls)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: the training-entry list (file order == training order).
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Training entries:"))
        self.lst_entries = QListWidget()
        self.lst_entries.currentRowChanged.connect(self._on_row_changed)
        left_layout.addWidget(self.lst_entries, 1)
        split.addWidget(left)

        # Right: a regenerate control bar over the three content boxes (query immutable,
        # CoT + answer editable), with the hand-edit Apply/Cancel bar underneath.
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Regenerate bar: mark which part(s) of the SELECTED entry are corrupt, choose a
        # temperature, and Regenerate has the currently loaded adapter re-answer the source
        # exchange. The result is reviewed in a pop-up before it touches the sidecar.
        regen_bar = QHBoxLayout()
        regen_bar.setSpacing(6)
        regen_bar.addWidget(QLabel("Regenerate:"))
        self.chk_corrupt_cot = QCheckBox("Corrupt CoT")
        self.chk_corrupt_cot.toggled.connect(self._update_regen_enabled)
        self.chk_corrupt_reply = QCheckBox("Corrupt reply")
        self.chk_corrupt_reply.toggled.connect(self._update_regen_enabled)
        regen_bar.addWidget(self.chk_corrupt_cot)
        regen_bar.addWidget(self.chk_corrupt_reply)
        regen_bar.addWidget(QLabel("Temp:"))
        self.edt_temp = QLineEdit("0.9")
        self.edt_temp.setFixedWidth(50)
        regen_bar.addWidget(self.edt_temp)
        # Which adapter answers: the loaded one (default, no cost), or any adapter from
        # the server's lineage — the "regenerate with the last known good one" path when
        # the current adapter is itself the thing producing bad re-answers. The swap is
        # STICKY: the first regeneration with a non-loaded choice pays one full model
        # reload and the adapter then stays loaded, so a repair session's further
        # regenerations under the same choice cost nothing extra — at the accepted
        # price that the box keeps running that adapter afterwards (chat and the idle
        # jobs included) until another load. Tooltip and dialog both say so. Populated
        # per refresh from the review payload's `adapters` (item data = the adapter dir
        # name; "" = whatever is loaded).
        regen_bar.addWidget(QLabel("Adapter:"))
        self.cmb_adapter = QComboBox()
        self.cmb_adapter.setToolTip(
            "Which adapter re-answers the exchange. \"Loaded adapter\" uses the model as "
            "it is (no extra cost). Choosing another adapter from the lineage swaps it in "
            "with ONE full model reload — and it then STAYS loaded, so repairing a run of "
            "rows under the same choice reloads only once. The box keeps running that "
            "adapter afterwards (chat included) until you pick another here or reload "
            "from the Chat tab; a server restart returns to the config's adapter. Use it "
            "to repair rows with a last-known-good adapter when the current one keeps "
            "re-answering badly.")
        self.cmb_adapter.addItem("Loaded adapter", "")
        regen_bar.addWidget(self.cmb_adapter)
        self.btn_regenerate = QPushButton("Regenerate")
        self.btn_regenerate.clicked.connect(self._on_regenerate)
        regen_bar.addWidget(self.btn_regenerate)
        self._lbl_regen = _StatusLabel("")
        regen_bar.addWidget(self._lbl_regen, 1)
        # Optional operator delivery constraint appended to the system prompt for THIS
        # regeneration only (not the trainable prefix) — e.g. steering the re-answer away
        # from a reflexive trailing question. Blank = a faithful re-answer, as before.
        # Rides the regenerate row (it belongs to the same action) rather than costing the
        # panel a second full-width bar; the long explanation lives in the tooltip.
        self.edt_suffix = QLineEdit()
        self.edt_suffix.setPlaceholderText("optional system-prompt suffix")
        self.edt_suffix.setToolTip(
            "Appended to the system prompt for THIS regeneration only — a delivery "
            "constraint, not part of the trainable prefix (e.g. don't append a question "
            "merely to continue the conversation). Blank = a faithful re-answer.")
        self.edt_suffix.setClearButtonEnabled(True)
        # A floor, so a long regenerate/apply status beside it can shrink the field but never
        # collapse it — the row's width demand stays the same whatever the status says.
        self.edt_suffix.setMinimumWidth(180)
        regen_bar.addWidget(self.edt_suffix, 1)
        right_layout.addLayout(regen_bar)

        right = QSplitter(Qt.Orientation.Vertical)
        self.txt_query = self._make_box("User query")
        self.txt_cot = self._make_box("Chain of thought (CoT)")
        self.txt_answer = self._make_box("Answer (trained reply)")
        for grp in (self._query_group, self._cot_group, self._answer_group):
            right.addWidget(grp)
        right_layout.addWidget(right, 1)

        # Hand-edit bar — the second route to the same sidecar write as Regenerate, for a row
        # the model can't re-answer well. The CoT / answer boxes are editable for any row with
        # chat provenance; Apply persists what is in them as the reviewed target (locking the
        # exchange), Cancel reverts them to the stored content.
        edit_bar = QHBoxLayout()
        edit_bar.addWidget(QLabel("Edit:"))
        self._lbl_edit = _StatusLabel("")
        edit_bar.addWidget(self._lbl_edit, 1)
        # Strip the detected persona opener. A PREFILL, never an auto-apply: it drops the
        # leading run out of the CoT box and leaves the operator to review and Apply.
        self._lbl_persona = QLabel("")
        edit_bar.addWidget(self._lbl_persona)
        self.btn_strip_persona = QPushButton("Strip persona opener")
        self.btn_strip_persona.clicked.connect(self._on_strip_persona)
        edit_bar.addWidget(self.btn_strip_persona)
        # Ban — the third verdict beside "repair it" and "leave it": this row should not
        # train, and no hand-written substitute for it should either. Sits on the edit bar
        # because it answers the same question the operator is asking of the row in front of
        # them, but it is enabled independently: a wander row can be banned and nothing else.
        self.btn_ban = QPushButton("Ban from training")
        self.btn_ban.clicked.connect(self._on_toggle_ban)
        edit_bar.addWidget(self.btn_ban)
        self.btn_apply_edits = QPushButton("Apply")
        self.btn_apply_edits.setToolTip(
            "Write the edited CoT + answer to the chat sidecar as this exchange's trainable "
            "target and freeze (lock) the exchange, so re-reflection and Revisit preserve it.")
        self.btn_apply_edits.clicked.connect(self._on_apply_edits)
        self.btn_cancel_edits = QPushButton("Cancel")
        self.btn_cancel_edits.setToolTip("Discard the edits and restore the stored CoT + answer.")
        self.btn_cancel_edits.clicked.connect(self._on_cancel_edits)
        edit_bar.addWidget(self.btn_apply_edits)
        edit_bar.addWidget(self.btn_cancel_edits)
        right_layout.addLayout(edit_bar)

        for box in (self.txt_cot, self.txt_answer):
            box.textChanged.connect(self._on_edit_changed)

        split.addWidget(right_col)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        outer.addWidget(split, 1)
        self._set_regen_controls_enabled(False)
        self._sync_edit_controls(None)

    def _make_box(self, label: str) -> QPlainTextEdit:
        """A labelled text box; stashes the group widget for the splitter.

        Read-only at construction: the query box stays that way, while the CoT + answer boxes
        are unlocked per selected entry by `_sync_edit_controls` (only rows with chat
        provenance can be written back)."""
        group = QWidget()
        gl = QVBoxLayout(group)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.addWidget(QLabel(label))
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setFont(self.text_font)
        box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        # Repaint this box's search highlight whenever its content changes — programmatic
        # fills and hand edits alike, since an edit shifts every later match's offsets.
        # Deliberately NOT routed through `_suppress_edit_signals`: that guard exists to
        # keep a programmatic fill from registering as an edit, and a fill is exactly when
        # the highlight must be recomputed.
        box.textChanged.connect(lambda b=box: self._highlight_box(b))
        gl.addWidget(box, 1)
        # Stash the group widget so _build_ui can drop it into the vertical splitter.
        if label.startswith("User"):
            self._query_group = group
        elif label.startswith("Chain"):
            self._cot_group = group
        else:
            self._answer_group = group
        return box

    # ---------------------------------------------------------------- #
    # Loading                                                           #
    # ---------------------------------------------------------------- #

    def refresh(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        # A reload restamps every entry's `_order`, which is what an in-flight bulk replace
        # patches its results back by — so the tab-open auto-refresh has to stand down too,
        # not just the (disabled) button.
        if self._bulk_in_flight:
            return
        if not self._confirm_discard_edits("reload the snapshot"):
            return
        # Prefer the connected server: the corpus lives beside the model, so fetching it
        # is what lets this tab run from a box that has no snapshot of its own. With no
        # connection we fall back to this checkout (the same-box / offline case).
        client = self._chat_widget._client
        base_url = (client.sidecar_base_url()
                    if client is not None and client.is_connected() else None)
        self._lbl_status.setText("Fetching training corpus from the server…" if base_url
                                 else "Loading latest local snapshot…")
        self.btn_refresh.setEnabled(False)
        self._worker = TrainingLoadWorker(self._chat_widget._local_server_dir(), base_url)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _populate_adapters(self, adapters: list) -> None:
        """Rebuild the regenerate bar's Adapter dropdown from the payload's lineage.

        Keeps the operator's selection across a refresh when that adapter is still
        listed (a repair session is a run of regenerations under ONE chosen adapter,
        and a refresh mid-session must not silently reset it to the loaded one).
        Item data is the bare adapter dir name the server re-resolves; "" = loaded."""
        prev = self.cmb_adapter.currentData() or ""
        self.cmb_adapter.blockSignals(True)
        self.cmb_adapter.clear()
        self.cmb_adapter.addItem("Loaded adapter", "")
        for a in adapters or []:
            name = str(a.get("name") or "")
            if not name:
                continue
            label = name
            if a.get("build_id"):
                label += f" — {a['build_id']}"
            if a.get("current"):
                label += " (current)"
            if not a.get("has_snapshot") and not a.get("current"):
                label += " (no build snapshot)"
            self.cmb_adapter.addItem(label, name)
        if prev:
            idx = self.cmb_adapter.findData(prev)
            if idx >= 0:
                self.cmb_adapter.setCurrentIndex(idx)
        self.cmb_adapter.blockSignals(False)

    def _on_loaded(self, result: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self._populate_adapters(result.get("adapters") or [])
        self._entries = result.get("entries", [])
        # Stamp each entry with its load position (file/training order) so "By order" can
        # restore it and selection survives a re-sort.
        for i, e in enumerate(self._entries):
            e["_order"] = i
        self._persona_statements = result.get("persona_statements") or []
        self._ratio_active = False
        self._search_needle = ""
        self.edt_search.blockSignals(True)
        self.edt_search.clear()
        self.edt_search.blockSignals(False)
        self._frozen_filter = "all"
        self.cmb_frozen.blockSignals(True)
        self.cmb_frozen.setCurrentIndex(0)
        self.cmb_frozen.blockSignals(False)
        self._persona_active = False
        self.chk_persona.blockSignals(True)
        self.chk_persona.setChecked(False)
        self.chk_persona.blockSignals(False)
        self._edit_order = None
        self._edit_baseline = ("", "")
        self._sort_mode = "order"
        self._sort_desc = True
        self._sync_sort_combo()
        self._rebuild_list()
        n = len(self._entries)
        rows = result.get("rows", n)
        frozen = sum(1 for e in self._entries if e.get("locked"))
        frozen_note = f", {frozen} frozen ❄" if frozen else ""
        banned = sum(1 for e in self._entries if e.get("banned"))
        if banned:
            frozen_note += f", {banned} banned 🚫"
        fresh = sum(1 for e in self._entries if e.get("preview"))
        if fresh:
            frozen_note += f", {fresh} fresh preview ▷ (untrained)"
        # Rows > entries means cap-age exchanges rendered a contamination copy each; the
        # list shows the exchange once, so name the collapse instead of hiding it.
        dup_note = f" (from {rows} rows — duplicate copies collapsed)" if rows > n else ""
        # Name where the corpus came from: the repairs below write to the *connected*
        # server, so an operator must be able to see at a glance whether what they are
        # reading came from that box or from a stale local checkout.
        origin = result.get("origin") or ""
        origin_note = f" via {origin}" if origin and origin != "this checkout" else ""
        # A training-lite preview snapshot holds the NEXT build's corpus — the whole
        # point is repairing it before training, so say plainly that nothing trained.
        preview_note = (" — PREVIEW: nothing trained yet"
                        if result.get("outcome") == "preview" else "")
        self._lbl_status.setText(
            f"Snapshot {result.get('build_id', '?')}{origin_note}{preview_note} — "
            f"{n} training "
            f"{'entry' if n == 1 else 'entries'}{dup_note}{frozen_note}."
        )
        self._clear_boxes()
        self._lbl_regen.setText("")
        self._lbl_search.setText("")
        self._lbl_filter.setText("")
        self._set_regen_controls_enabled(False)
        self._sync_edit_controls(None)
        if self._entries:
            self.lst_entries.setCurrentRow(0)

    def _on_failed(self, message: str) -> None:
        self.btn_refresh.setEnabled(True)
        self._entries = []
        self._ratio_active = False
        self._search_needle = ""
        self.edt_search.blockSignals(True)
        self.edt_search.clear()
        self.edt_search.blockSignals(False)
        self._edit_order = None
        self.lst_entries.clear()
        self._clear_boxes()
        self._lbl_status.setText(message.splitlines()[0])
        self.txt_query.setPlainText(message)
        self._lbl_regen.setText("")
        self._lbl_search.setText("")
        self._lbl_filter.setText("")
        self._set_regen_controls_enabled(False)
        self._sync_edit_controls(None)

    def _clear_boxes(self) -> None:
        """Empty the three boxes and re-baseline, so clearing never reads as a hand edit."""
        self._suppress_edit_signals = True
        for box in (self.txt_query, self.txt_cot, self.txt_answer):
            box.clear()
        self._suppress_edit_signals = False
        self._edit_baseline = ("", "")
        self._edit_order = None

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._entries):
            self._set_regen_controls_enabled(False)
            self._sync_edit_controls(None)
            return
        e = self._entries[row]
        # Re-selecting the entry being hand-edited (e.g. after a re-sort carried the
        # selection across) must not clobber the unsaved edits with the stored content.
        if self._edit_order is not None and e.get("_order") == self._edit_order:
            self._sync_regen_controls(e)
            self._sync_edit_controls(e)
            return
        if not self._confirm_discard_edits("switch to another entry"):
            self._restore_edit_selection()
            return
        self.txt_query.setPlainText(e["query"])
        self._populate_edit_boxes(e)
        self._sync_regen_controls(e)
        self._sync_edit_controls(e)

    def _rebuild_list(self) -> None:
        """Rebuild the entry list from `self._entries` (current order), re-applying the
        active filters. Signals are blocked during the clear/add so `_on_row_changed`
        doesn't fire mid-rebuild against a transient selection."""
        self.lst_entries.blockSignals(True)
        self.lst_entries.clear()
        for i, e in enumerate(self._entries):
            item = QListWidgetItem(self._item_text(i, e))
            item.setIcon(self._row_icon(e))
            self.lst_entries.addItem(item)
        self.lst_entries.blockSignals(False)
        if (self._ratio_active or self._search_needle or self._frozen_filter != "all"
                or self._persona_active):
            self._apply_hiding()

    # ---------------------------------------------------------------- #
    # Sorting                                                          #
    # ---------------------------------------------------------------- #

    def _on_sort_changed(self, *_) -> None:
        """Sort mode/direction picked from the combo (mode "order" restores file order)."""
        data = self.cmb_sort.currentData()
        if not data:
            return
        self._sort_mode, self._sort_desc = data
        self._apply_sort()

    def _apply_sort(self) -> None:
        """Reorder `self._entries` per the current sort mode and rebuild the list.

        Reordering `self._entries` and the list widget together keeps the row→entry mapping
        every other control (search / filter / regenerate) depends on. Python's sort is
        stable, so equal reply lengths keep their prior relative order. The selection is
        carried to the same entry by its stamped `_order`."""
        self._sync_sort_combo()
        if not self._entries:
            return
        cur = self.lst_entries.currentRow()
        cur_order = (self._entries[cur].get("_order")
                     if 0 <= cur < len(self._entries) else None)
        if self._sort_mode == "reply_len":
            self._entries.sort(key=lambda e: len(self._disp_answer(e)),
                               reverse=self._sort_desc)
        elif self._sort_mode == "persona":
            self._entries.sort(key=self._persona_score, reverse=self._sort_desc)
        else:
            self._entries.sort(key=lambda e: e.get("_order", 0))
        self._rebuild_list()
        if cur_order is not None:
            for i, e in enumerate(self._entries):
                if e.get("_order") == cur_order:
                    self.lst_entries.setCurrentRow(i)
                    it = self.lst_entries.item(i)
                    if it is not None:
                        self.lst_entries.scrollToItem(it)
                    break

    def _sync_sort_combo(self) -> None:
        """Point the sort combo at the active mode/direction without re-triggering a sort
        (so a programmatic reset — e.g. a fresh load — can't recurse through `_apply_sort`)."""
        for i in range(self.cmb_sort.count()):
            mode, desc = self.cmb_sort.itemData(i)
            if mode == self._sort_mode and (mode == "order" or desc == self._sort_desc):
                self.cmb_sort.blockSignals(True)
                self.cmb_sort.setCurrentIndex(i)
                self.cmb_sort.blockSignals(False)
                return

    # ---------------------------------------------------------------- #
    # Search                                                           #
    # ---------------------------------------------------------------- #

    def _search_phrase(self) -> str:
        """The search phrase EXACTLY as typed — no strip.

        One definition for the filter, the highlight and both Replace buttons, so what is
        marked yellow is precisely what a replace would rewrite. It is deliberately not
        stripped: the phrase this was built for is a poison sequence with a **trailing space**
        (`"terms of "`), and stripping it would delete the words and leave the double space
        behind — a whitespace-tolerant search box is not worth a corpus-wide off-by-one."""
        return self.edt_search.text()

    def _on_search_text_changed(self, text: str) -> None:
        """Search box edited — re-filter the list to the entries containing the phrase.

        This is the filter half of the search: an empty box shows everything, a non-empty
        one hides every non-matching entry (`_apply_hiding` does the actual hiding, so the
        four filters compose). Stepping through the survivors stays on the Search button."""
        needle = text.lower()
        if needle == self._search_needle:
            return
        self._search_needle = needle
        self._lbl_search.setText("")
        self._apply_hiding()
        self._highlight_search()

    # Cap on painted occurrences per box: a one-letter phrase against a long reply would
    # otherwise build thousands of selections on every keystroke. Past the cap the box is
    # still shown, just not fully painted — the phrase is that unselective anyway.
    _MAX_HIGHLIGHTS = 500

    def _highlight_search(self) -> None:
        """Repaint the search highlight in all three preview boxes (after a needle change)."""
        for box in (self.txt_query, self.txt_cot, self.txt_answer):
            self._highlight_box(box)

    def _highlight_box(self, box: QPlainTextEdit) -> None:
        """Paint a yellow background under every occurrence of the search phrase in `box`.

        View-level `ExtraSelections` only — the document is never touched, so highlighting
        can never mark a row dirty, reach `_is_dirty`, or end up in a written target."""
        needle = self._search_needle
        selections = []
        if needle:
            text = box.toPlainText().lower()
            start = text.find(needle)
            while start >= 0 and len(selections) < self._MAX_HIGHLIGHTS:
                cursor = QTextCursor(box.document())
                cursor.setPosition(start)
                cursor.setPosition(start + len(needle), QTextCursor.MoveMode.KeepAnchor)
                sel = QTextEdit.ExtraSelection()
                sel.cursor = cursor
                sel.format = self._search_hl_fmt
                selections.append(sel)
                start = text.find(needle, start + len(needle))
        box.setExtraSelections(selections)

    def _on_search(self) -> None:
        """Select the next entry whose query / CoT / reply contains the phrase.

        The list is already filtered to the matches as the phrase was typed; this steps
        through them, advancing from the current selection and wrapping around."""
        phrase = self._search_phrase()
        if not phrase:
            self._lbl_search.setText("Enter a phrase.")
            return
        if not self._entries:
            self._lbl_search.setText("No entries loaded.")
            return
        needle = phrase.lower()
        # Only cycle through currently-visible rows, so the other filters and search compose.
        matches = [
            i for i, e in enumerate(self._entries)
            if needle in self._haystack(e) and not self.lst_entries.item(i).isHidden()
        ]
        if not matches:
            self._lbl_search.setText(f"No match for “{phrase}”.")
            return
        cur = self.lst_entries.currentRow()
        nxt = next((i for i in matches if i > cur), matches[0])
        self.lst_entries.setCurrentRow(nxt)
        item = self.lst_entries.currentItem()
        if item is not None:
            self.lst_entries.scrollToItem(item)
        self._lbl_search.setText(
            f"Match {matches.index(nxt) + 1}/{len(matches)} (entry {nxt + 1})")

    def _on_replace(self) -> None:
        """Replace the search phrase in the selected entry's CoT + answer boxes.

        A **prefill, never a write**: this rewrites the two editable boxes (which marks the
        entry dirty through their `textChanged`), and the edit bar's Apply is still what
        persists the target and freezes ❄ the exchange. The query box is excluded — it is
        immutable and never written back, so replacing in it would show the operator a change
        that could not be applied.

        Matching is case-insensitive, mirroring the search filter + highlight, and the
        replacement is inserted literally: the substitution runs through a callable so `\\1`
        and friends in the replacement box are text, not backreferences. An empty replacement
        box deletes the phrase."""
        phrase = self._search_phrase()
        if not phrase:
            self._lbl_search.setText("Enter a search phrase to replace.")
            return
        e = self._current_entry()
        if e is None:
            self._lbl_search.setText("Select an entry to replace in.")
            return
        if not self._has_provenance(e):
            self._lbl_search.setText(
                "This entry is read-only — no source exchange to write a repair back to.")
            return
        replacement = self.edt_replace.text()
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        total = 0
        for box in (self.txt_cot, self.txt_answer):
            new, n = pattern.subn(lambda _m: replacement, box.toPlainText())
            if n:
                # Marks the entry dirty via textChanged (and repaints the highlight).
                box.setPlainText(new)
                total += n
        if not total:
            self._lbl_search.setText(
                f"No “{phrase}” in this entry's CoT / answer (the query is not editable).")
            return
        plural = "" if total == 1 else "s"
        self._lbl_search.setText(
            f"{'Removed' if not replacement else 'Replaced'} {total} occurrence{plural}.")
        self._lbl_edit.setText(
            f"{'Removed' if not replacement else 'Replaced'} {total} occurrence{plural} of "
            f"“{phrase}” — review the boxes, then Apply (writes the target and freezes ❄ the "
            "exchange) or Cancel.")

    def _bulk_replace_jobs(self, phrase: str, replacement: str) -> tuple:
        """Plan a bulk replace: one job per shown, repairable, matching entry.

        Returns `(jobs, skipped_readonly, skipped_empty)`. Scope is the entries currently
        SHOWN, so the filters above the list choose it — narrowing by frozen state or by the
        search phrase itself is how an operator bounds a sweep. Each job carries the entry's
        `_order` stamp rather than its row, so progress can find it again after a re-sort.

        Two exclusions, both reported rather than silent: a **read-only** row (wander /
        pre-provenance) has no source exchange to write back to, and a row whose reply would
        come out **empty** is refused for the same reason the single Apply refuses it — a
        trainable target needs a reply, and writing an empty one would replace poison with a
        worse artifact."""
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        sub = lambda text: pattern.subn(lambda _m: replacement, text)  # noqa: E731
        jobs: list = []
        skipped_readonly = 0
        skipped_empty = 0
        for i, e in enumerate(self._entries):
            item = self.lst_entries.item(i)
            if item is None or item.isHidden():
                continue
            cot, n_cot = sub(self._disp_cot(e))
            answer, n_ans = sub(self._disp_answer(e))
            if not (n_cot or n_ans):
                continue
            if not self._has_provenance(e):
                skipped_readonly += 1
                continue
            if not answer.strip():
                skipped_empty += 1
                continue
            jobs.append({
                "order": e.get("_order"),
                "filename": e["source_session"],
                "exchange_index": e["exchange_index"],
                "new_cot": cot.strip(),
                "new_reply": answer.strip(),
                "hits": n_cot + n_ans,
            })
        return jobs, skipped_readonly, skipped_empty

    def _on_replace_all(self) -> None:
        """Replace the search phrase across every shown entry, applying + freezing each.

        The one control on this bar that WRITES. Replace leaves the write to a per-entry
        Apply because a hand repair is one judgement about one exchange; a poison sequence
        smeared across the corpus by a bad build is the opposite case — the same three words
        deleted in 400 places, where per-entry review is a formality nobody performs. So each
        result goes straight to its sidecar through the same `apply_regenerated_exchange` the
        single Apply uses, which **locks ❄ the exchange** so re-reflection and Revisit cannot
        re-derive the poison from the untouched transcript.

        Confirms first (it is a few hundred writes), and the button becomes **Stop** while it
        runs: each write is independent, so stopping leaves what it reached repaired and
        frozen and the rest exactly as they were."""
        if self._bulk_in_flight:
            # The button is a Stop while a sweep is running.
            if self._bulk_worker is not None:
                self._bulk_worker.stop()
                self.btn_replace_all.setEnabled(False)
                self._lbl_status.setText("Stopping the bulk replace after the current write…")
            return
        phrase = self._search_phrase()
        if not phrase:
            self._lbl_search.setText("Enter a search phrase to replace.")
            return
        if not self._entries:
            self._lbl_search.setText("No entries loaded.")
            return
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_status.setText("Not connected — connect from the Chat tab to replace.")
            return
        # A bulk job rewrites from each entry's STORED content, so unsaved box edits would be
        # silently overwritten on the way past.
        if not self._confirm_discard_edits("replace across every shown entry"):
            return
        replacement = self.edt_replace.text()
        jobs, skipped_readonly, skipped_empty = self._bulk_replace_jobs(phrase, replacement)
        if not jobs:
            notes = []
            if skipped_readonly:
                notes.append(f"{skipped_readonly} read-only")
            if skipped_empty:
                notes.append(f"{skipped_empty} would leave an empty reply")
            tail = f" ({', '.join(notes)} — skipped)" if notes else ""
            self._lbl_status.setText(
                f"Bulk replace: no repairable shown entry contains “{phrase}”{tail}.")
            return
        hits = sum(j["hits"] for j in jobs)
        locked_orders = {e.get("_order") for e in self._entries if e.get("locked")}
        frozen = sum(1 for j in jobs if j["order"] in locked_orders)
        n = len(jobs)
        notes = []
        if skipped_readonly:
            notes.append(f"{skipped_readonly} matching read-only row"
                         f"{'s' if skipped_readonly != 1 else ''} (wander / no source "
                         f"exchange) will be skipped.")
        if skipped_empty:
            notes.append(f"{skipped_empty} row{'s' if skipped_empty != 1 else ''} would be "
                         "left with an empty reply and will be skipped.")
        if frozen:
            notes.append(f"{frozen} of them {'is' if frozen == 1 else 'are'} already frozen ❄ "
                         "— their reviewed target will be rewritten.")
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle("Replace in every shown entry?")
        confirm.setText(
            f"{'Delete' if not replacement else 'Replace'} “{phrase}”"
            f"{'' if not replacement else f' with “{replacement}”'} in {n} "
            f"{'entry' if n == 1 else 'entries'}?")
        confirm.setInformativeText(
            f"{hits} occurrence{'s' if hits != 1 else ''} across the entries currently shown "
            f"({len(self._entries)} loaded — narrow with the filters above to change the "
            "scope).\n\n"
            "Unlike Replace, this WRITES: each entry's repaired CoT + answer go straight to "
            "the chat sidecar on the connected server as that exchange's trainable target, "
            "and the exchange is frozen ❄ (human-validated) so re-reflection and Revisit "
            "preserve it.\n\n"
            + ("\n".join(notes) + "\n\n" if notes else "") +
            "The transcripts themselves are untouched (that is “Rewrite history”). To undo, "
            "re-reflect the affected chats from the Chat tab — which discards every reviewed "
            "target in those chats, not only these.")
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return
        self._bulk_in_flight = True
        self._set_bulk_controls_running(True)
        self._lbl_status.setText(f"Bulk replace: 0/{n} written…")
        self._bulk_worker = BulkReplaceWorker(client, jobs)
        self._bulk_worker.progress.connect(self._on_bulk_progress)
        self._bulk_worker.done.connect(self._on_bulk_done)
        self._bulk_worker.start()

    def _set_bulk_controls_running(self, running: bool) -> None:
        """Lock down the controls that could swap `self._entries` mid-sweep, and turn the
        Replace-all button into a Stop.

        Progress patches entries by their `_order` stamp; a Refresh restamps them (and a
        second write path could interleave with the sweep's), so Refresh / Rewrite history /
        the single repair buttons stand down until it finishes."""
        self.btn_replace_all.setEnabled(True)
        self.btn_replace_all.setText("Stop" if running else "Replace all…")
        for w in (self.btn_refresh, self.btn_rewrite, self.btn_replace, self.btn_regenerate,
                  self.btn_apply_edits, self.btn_cancel_edits, self.btn_ban,
                  self.btn_strip_persona):
            w.setEnabled(False)
        if not running:
            self.btn_refresh.setEnabled(True)
            self.btn_rewrite.setEnabled(True)
            self.btn_replace.setEnabled(True)
            self._update_regen_enabled()
            self._sync_edit_controls(self._current_entry())

    def _on_bulk_progress(self, attempted: int, total: int, payload: dict) -> None:
        """One bulk write landed — reflect it in the in-memory entry, exactly as a single
        Apply does (live target + ❄ lock + list key/icon + stale caches dropped)."""
        self._lbl_status.setText(f"Bulk replace: {attempted}/{total} written…")
        order = payload.get("order")
        if order is None:
            return
        for row, e in enumerate(self._entries):
            if e.get("_order") != order:
                continue
            e["live_cot"] = str(payload.get("cot") or "")
            e["live_answer"] = str(payload.get("reply") or "")
            e["has_live"] = True
            e["locked"] = True
            e["key"] = " ".join(e["live_answer"].split())[:20] or "(empty reply)"
            self._invalidate_haystack(e)
            self._rescore_persona(e)
            item = self.lst_entries.item(row)
            if item is not None:
                item.setText(self._item_text(row, e))
                item.setIcon(self._row_icon(e))
            if self.lst_entries.currentRow() == row:
                self._populate_edit_boxes(e)
            return

    def _on_bulk_done(self, applied: int, failures: list, stopped: bool) -> None:
        self._bulk_in_flight = False
        self._set_bulk_controls_running(False)
        # Every touched row's content, frozen state and persona verdict changed, so an active
        # filter's verdict on them did too. Re-filtering LAST (not per write) keeps the list
        # from reshuffling under the operator mid-sweep.
        if self._frozen_filter != "all" or self._persona_active or self._search_needle:
            self._apply_hiding()
        self._sync_edit_controls(self._current_entry())
        parts = [f"Bulk replace{' STOPPED' if stopped else ''}: wrote + froze ❄ {applied} "
                 f"{'exchange' if applied == 1 else 'exchanges'}."]
        if failures:
            parts.append(f"{len(failures)} failed ({failures[0][1]}).")
        if stopped:
            parts.append("The rest were left untouched.")
        parts.append("They train from the LIVE sidecar, not this snapshot.")
        self._lbl_status.setText("  ".join(parts))

    # ---------------------------------------------------------------- #
    # Ratio filter                                                     #
    # ---------------------------------------------------------------- #

    def _on_filter(self) -> None:
        """Activate the CoT/reply ratio filter, then apply hiding.

        Raw character counts (no tokenization). Survivors of the ratio test are the
        tiny-CoT / huge-reply exchanges — the shape the missing-<eos> bug leaves behind."""
        if not self._entries:
            self._lbl_filter.setText("No entries loaded.")
            return
        try:
            threshold = float(self.edt_ratio.text().strip())
        except ValueError:
            self._lbl_filter.setText("Enter a number.")
            return
        self._ratio_active = True
        self._ratio_threshold = threshold
        self._apply_hiding()

    def _on_frozen_filter(self, *_) -> None:
        """Frozen-state filter changed — apply immediately (it needs no threshold input)."""
        self._frozen_filter = self.cmb_frozen.currentData() or "all"
        self._apply_hiding()

    def _on_persona_filter(self, *_) -> None:
        """Persona-opener filter toggled, or its threshold re-entered."""
        try:
            self._persona_threshold = float(self.edt_persona.text().strip())
        except ValueError:
            self._lbl_filter.setText("Persona threshold must be a number.")
            return
        self._persona_active = self.chk_persona.isChecked()
        self._apply_hiding()

    def _apply_hiding(self, *_) -> None:
        """Hide rows failing the active filters. The list is only hidden, never rebuilt, so
        entry numbering and the source-exchange index mapping stay intact.

        All four filters (search phrase, ratio, frozen state, persona opener) compose: a row
        must pass every active one to stay visible."""
        thr = self._ratio_threshold
        frozen_mode = self._frozen_filter
        p_thr = self._persona_threshold
        needle = self._search_needle
        shown = 0
        for i, e in enumerate(self._entries):
            item = self.lst_entries.item(i)
            if item is None:
                continue
            cot_len = len(self._disp_cot(e))
            ans_len = len(self._disp_answer(e))
            hide = self._ratio_active and cot_len >= thr * ans_len
            if not hide and needle:
                hide = needle not in self._haystack(e)
            if not hide and frozen_mode != "all":
                if frozen_mode == "banned":
                    hide = not e.get("banned")
                elif frozen_mode == "fresh":
                    hide = not e.get("preview")
                else:
                    locked = bool(e.get("locked"))
                    hide = (not locked) if frozen_mode == "frozen" else locked
            if not hide and self._persona_active:
                hide = self._persona_score(e) < p_thr
            item.setHidden(hide)
            if not hide:
                shown += 1
        notes = []
        if needle:
            notes.append(f"“{self._search_phrase()}”")
        if self._ratio_active:
            notes.append(f"CoT/reply < {thr:g}")
        if frozen_mode == "frozen":
            notes.append("frozen ❄ only")
        elif frozen_mode == "unfrozen":
            notes.append("unfrozen only")
        elif frozen_mode == "banned":
            notes.append("banned 🚫 only")
        elif frozen_mode == "fresh":
            notes.append("fresh ▷ only")
        if self._persona_active:
            notes.append(f"persona ✦ ≥ {p_thr:g}")
        self._lbl_filter.setText(
            f"{shown}/{len(self._entries)} shown ({', '.join(notes)})" if notes else "")
        self._select_visible_row()

    def _on_clear_filter(self) -> None:
        """Show every row again — clears the search, ratio, frozen and persona filters."""
        self._ratio_active = False
        self._search_needle = ""
        self.edt_search.blockSignals(True)
        self.edt_search.clear()
        self.edt_search.blockSignals(False)
        self._lbl_search.setText("")
        # The boxes keep their content here (no entry change), so the stale highlight has to
        # be repainted away explicitly.
        self._highlight_search()
        self._frozen_filter = "all"
        self.cmb_frozen.blockSignals(True)
        self.cmb_frozen.setCurrentIndex(0)
        self.cmb_frozen.blockSignals(False)
        self._persona_active = False
        self.chk_persona.blockSignals(True)
        self.chk_persona.setChecked(False)
        self.chk_persona.blockSignals(False)
        for i in range(self.lst_entries.count()):
            item = self.lst_entries.item(i)
            if item is not None:
                item.setHidden(False)
        self._lbl_filter.setText("")

    def _select_visible_row(self) -> None:
        """Keep the selection on a visible row after filtering (move to the first
        visible entry if the current one just got hidden)."""
        cur = self.lst_entries.currentRow()
        item = self.lst_entries.item(cur) if cur >= 0 else None
        if item is not None and not item.isHidden():
            return
        for i in range(self.lst_entries.count()):
            it = self.lst_entries.item(i)
            if it is not None and not it.isHidden():
                self.lst_entries.setCurrentRow(i)
                self.lst_entries.scrollToItem(it)
                return

    # ---------------------------------------------------------------- #
    # Hand edit (CoT / answer)                                         #
    # ---------------------------------------------------------------- #

    def _current_entry(self) -> Optional[dict]:
        """The selected entry, or None when nothing (valid) is selected."""
        row = self.lst_entries.currentRow()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    def _populate_edit_boxes(self, e: dict) -> None:
        """Load an entry's stored CoT + answer into the editable boxes and re-baseline.

        The baseline is what Cancel reverts to and what "dirty" is measured against, so it is
        stamped here — the one place the boxes are filled programmatically. Box signals are
        suppressed so the fill itself doesn't register as an edit."""
        cot, answer = self._disp_cot(e), self._disp_answer(e)
        self._suppress_edit_signals = True
        self.txt_cot.setPlainText(cot)
        self.txt_answer.setPlainText(answer)
        self._suppress_edit_signals = False
        self._edit_baseline = (cot, answer)
        self._edit_order = None

    def _is_dirty(self) -> bool:
        return (self.txt_cot.toPlainText(), self.txt_answer.toPlainText()) != self._edit_baseline

    def _on_edit_changed(self) -> None:
        """Track whether the boxes diverge from the stored content, keyed by ENTRY.

        Keying the dirty flag on the entry's `_order` (not the row) means a re-sort that
        moves the row can't strand the edits on the wrong entry."""
        if self._suppress_edit_signals:
            return
        row = self.lst_entries.currentRow()
        if not (0 <= row < len(self._entries)):
            return
        e = self._entries[row]
        self._edit_order = e.get("_order") if self._is_dirty() else None
        self._sync_edit_controls(e)

    def _sync_edit_controls(self, e: Optional[dict]) -> None:
        """Enable the editable boxes + Apply/Cancel for the selected entry.

        Editing writes to the chat sidecar by (source_session, exchange_index), so it needs
        the same provenance regeneration does; a wander / pre-provenance row stays read-only.
        Apply and Cancel are live only while there is something unsaved to apply or revert."""
        editable = bool(e is not None and self._has_provenance(e))
        self.txt_cot.setReadOnly(not editable)
        self.txt_answer.setReadOnly(not editable)
        # The persona verdict + strip button track the same selection, so they ride along.
        self._sync_persona_controls(e)
        # Ban rides the same bar but not the same gate: it survives on a row nothing else
        # here can touch (see `_can_ban`).
        self._sync_ban_control(e)
        dirty = editable and self._edit_order is not None
        self.btn_apply_edits.setEnabled(dirty and not self._apply_in_flight)
        self.btn_cancel_edits.setEnabled(dirty)
        if not editable:
            self._lbl_edit.setText(
                "(wander capture — nothing to repair here; it can only be banned)"
                if (e is not None and e.get("kind") == "wander")
                else "(read-only — no source exchange to write back to)")
        elif dirty:
            self._lbl_edit.setText(
                "Unsaved edits — Apply writes them as this exchange's target and freezes ❄ it.")
        else:
            self._lbl_edit.setText("Edit the CoT / answer boxes above to repair this entry by hand.")

    def _sync_ban_control(self, e: Optional[dict]) -> None:
        """Point the Ban button at the selected row's current state.

        The button is a toggle showing what pressing it would DO, not what the row is (the
        list's 🚫 says that) — so a banned row offers "Un-ban"."""
        can = self._can_ban(e)
        self.btn_ban.setEnabled(can and not self._ban_in_flight)
        banned = bool(e is not None and e.get("banned"))
        self.btn_ban.setText("Un-ban" if banned else "Ban from training")
        if not can:
            self.btn_ban.setToolTip(
                "This row carries no source to flag — re-fetch the snapshot to enable banning.")
        elif banned:
            self.btn_ban.setToolTip(
                "Lift the ban: this row trains again from the next build.")
        elif e is not None and e.get("kind") == "wander":
            self.btn_ban.setToolTip(
                "Bar this wander capture from every future build, and from the chat-RAG "
                "wander channel.\n\nThe record stays on disk (reversible) until “Rewrite "
                "history”, which deletes it for good.")
        else:
            self.btn_ban.setToolTip(
                "Bar this exchange from every future build — for a row not worth learning "
                "from at all, where writing a substitute target would be inventing a memory "
                "rather than correcting one.\n\nThe exchange stays in the transcript and in "
                "RAG (reversible) until “Rewrite history”, which deletes it from the "
                "transcript.")

    def _on_toggle_ban(self) -> None:
        """Ban / un-ban the selected row on the server."""
        row = self.lst_entries.currentRow()
        if not (0 <= row < len(self._entries)):
            return
        e = self._entries[row]
        if not self._can_ban(e):
            self._sync_ban_control(e)
            return
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_edit.setText("Not connected — connect from the Chat tab to ban a row.")
            return
        if self._ban_in_flight:
            return
        banned = not bool(e.get("banned"))
        idx = e.get("exchange_index") if e.get("kind") != "wander" else None
        self._ban_pending_row = row
        self._ban_in_flight = True
        self.btn_ban.setEnabled(False)
        self._lbl_edit.setText("Banning this row from training…" if banned
                               else "Lifting the training ban…")
        self._ban_worker = BanWorker(client, e["source_session"], idx, banned)
        self._ban_worker.done.connect(self._on_ban_done)
        self._ban_worker.failed.connect(self._on_ban_failed)
        self._ban_worker.start()

    def _on_ban_done(self, result: dict) -> None:
        self._ban_in_flight = False
        row = self._ban_pending_row
        self._ban_pending_row = None
        banned = bool(result.get("banned"))
        if isinstance(row, int) and 0 <= row < len(self._entries):
            e = self._entries[row]
            e["banned"] = banned
            item = self.lst_entries.item(row)
            if item is not None:
                item.setIcon(self._row_icon(e))
        # The banned-state filter's verdict on this row just changed; under "Only banned"
        # (or its inverse) that advances the selection to the next row awaiting a decision.
        if self._frozen_filter != "all":
            self._apply_hiding()
        self._sync_edit_controls(self._current_entry())
        self._lbl_edit.setText(
            "Banned 🚫 — this row is out of the corpus from the next build. “Rewrite "
            "history” deletes the source for good."
            if banned else
            "Ban lifted — this row trains again from the next build.")

    def _on_ban_failed(self, message: str) -> None:
        self._ban_in_flight = False
        self._ban_pending_row = None
        self._sync_edit_controls(self._current_entry())
        self._lbl_edit.setText(f"Ban failed: {message}")

    def _sync_persona_controls(self, e: Optional[dict]) -> None:
        """Show the selected entry's persona-opener verdict and arm the strip button.

        Strip is available only when the detected lines form the CONTIGUOUS LEADING RUN of
        the CoT (`_score_persona_opener` computes it): the injector prepended onto an
        existing thought, so dropping that run restores the pre-injection CoT verbatim. A hit
        with real reasoning above it is entangled and belongs to the hand editor."""
        p = (e or {}).get("persona") or {}
        score = float(p.get("score") or 0.0)
        editable = bool(e is not None and self._has_provenance(e))
        strip = list(p.get("strip") or ())
        can_strip = editable and bool(strip) and score >= _PERSONA_MARK
        self.btn_strip_persona.setEnabled(can_strip)
        if score < _PERSONA_MARK:
            self._lbl_persona.setText("")
            self.btn_strip_persona.setToolTip(
                "No persona opener detected in this CoT.")
            return
        kind = p.get("kind") or "?"
        self._lbl_persona.setText(f"✦{score:.2f} {kind}")
        matched = (p.get("matched") or "")[:160]
        where = f"line {int(p.get('line', -1)) + 1}"
        if can_strip:
            n = len(strip)
            tip = (f"Remove the leading {n} CoT line{'s' if n != 1 else ''} detected as a "
                   f"persona opener ({kind} match, {where}):\n\n{matched}\n\n"
                   "This only fills the CoT box — review it, then Apply.")
        elif not strip:
            tip = (f"Detected a persona opener ({kind} match, {where}) but NOT at the start of "
                   "the CoT — there is real reasoning above it, so removing it mechanically "
                   f"could strand a back-reference. Edit it by hand:\n\n{matched}")
        else:
            tip = "This row has no source exchange to write back to."
        self.btn_strip_persona.setToolTip(tip)

    def _rescore_persona(self, e: dict) -> None:
        """Recompute an entry's persona score against its CURRENT displayed CoT.

        Called after a repair lands, so a stripped opener stops being reported (and drops out
        of the persona filter) without a snapshot reload."""
        if not (e.get("source_session") and isinstance(e.get("exchange_index"), int)):
            return
        e["persona"] = _score_persona_opener(
            e.get("query", ""), self._disp_cot(e), self._persona_statements)

    def _on_strip_persona(self) -> None:
        """Drop the detected persona opener from the CoT box — a prefill, not a write."""
        e = self._current_entry()
        if e is None or not self._has_provenance(e):
            return
        strip = list(((e.get("persona") or {}).get("strip")) or ())
        if not strip:
            return
        current = self.txt_cot.toPlainText()
        stripped = _strip_lines(current, strip)
        if stripped == current:
            self._lbl_edit.setText("Nothing to strip — the CoT box no longer matches the detection.")
            return
        if not stripped:
            self._lbl_edit.setText(
                "Refusing to strip: that would leave the CoT empty (the opener is the whole "
                "thought). Regenerate this row instead.")
            return
        self.txt_cot.setPlainText(stripped)      # marks the entry dirty via textChanged
        n = len(strip)
        self._lbl_edit.setText(
            f"Stripped {n} persona line{'s' if n != 1 else ''} from the CoT — review it, then "
            "Apply (writes the target and freezes ❄ the exchange) or Cancel.")

    def _confirm_discard_edits(self, action: str) -> bool:
        """Ask before losing unsaved edits. True == proceed (edit state cleared)."""
        if self._edit_order is None:
            return True
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle("Discard edits?")
        confirm.setText(f"Discard the unsaved edits to this entry and {action}?")
        confirm.setInformativeText(
            "The edited CoT / answer have not been applied, so nothing was written to the "
            "chat sidecar.")
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Discard:
            return False
        self._edit_order = None
        return True

    def _restore_edit_selection(self) -> None:
        """Put the selection back on the entry holding unsaved edits (after a declined
        discard). Signals are blocked so this doesn't re-enter `_on_row_changed`."""
        for i, e in enumerate(self._entries):
            if e.get("_order") == self._edit_order:
                self.lst_entries.blockSignals(True)
                self.lst_entries.setCurrentRow(i)
                self.lst_entries.blockSignals(False)
                item = self.lst_entries.item(i)
                if item is not None:
                    self.lst_entries.scrollToItem(item)
                return

    def _on_cancel_edits(self) -> None:
        """Revert the boxes to the stored CoT + answer (nothing was written)."""
        row = self.lst_entries.currentRow()
        if not (0 <= row < len(self._entries)):
            return
        e = self._entries[row]
        self._populate_edit_boxes(e)
        self._sync_edit_controls(e)
        self._lbl_edit.setText("Edits discarded — restored the stored CoT + answer.")

    def _on_apply_edits(self) -> None:
        """Write the hand-edited CoT + answer to the sidecar as this exchange's target.

        Same server path as the regenerate flow's Apply (``apply_regenerated_exchange``): the
        pair is written together as one reviewed target and the exchange is **locked
        (frozen)**, so re-reflection and Revisit preserve it. The CoT + answer travel as a
        pair — an operator editing one box has the other in front of them — so this always
        writes both (``corrupt_response``), never the CoT-only graft. What is in the boxes is
        exactly what is written: for a not-yet-frozen row that starts from the snapshot
        render's content (what the tab has been showing), not a re-read of the live sidecar."""
        row = self.lst_entries.currentRow()
        if not (0 <= row < len(self._entries)):
            return
        e = self._entries[row]
        if not self._has_provenance(e):
            self._sync_edit_controls(e)
            return
        new_cot = self.txt_cot.toPlainText().strip()
        new_reply = self.txt_answer.toPlainText().strip()
        if not new_reply:
            self._lbl_edit.setText("The answer box is empty — a trainable target needs a reply.")
            return
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_edit.setText("Not connected — connect from the Chat tab to apply.")
            return
        if self._apply_worker is not None and self._apply_worker.isRunning():
            return
        self._pending = {
            "row": row,
            "mode": "edit",
            "filename": e["source_session"],
            "exchange_index": e["exchange_index"],
            "corrupt_cot": True,
            "corrupt_response": True,
            "kept_reply": new_reply,
        }
        self._apply_in_flight = True
        self.btn_apply_edits.setEnabled(False)
        self._lbl_edit.setText("Applying the edited target to the sidecar…")
        self._apply_worker = ApplyRegenWorker(
            client, e["source_session"], e["exchange_index"],
            corrupt_cot=True, corrupt_response=True,
            new_cot=new_cot, new_reply=new_reply, original_reply=new_reply)
        self._apply_worker.done.connect(self._on_apply_done)
        self._apply_worker.failed.connect(self._on_apply_failed)
        self._apply_worker.start()

    # ---------------------------------------------------------------- #
    # Regenerate + apply                                               #
    # ---------------------------------------------------------------- #

    def _has_provenance(self, e: dict) -> bool:
        """A row can be regenerated only if it carries chat provenance (source_session +
        integer exchange_index). Wander rows and old snapshots (pre-provenance) can't be
        traced to a source exchange, so their controls stay disabled."""
        return bool(e.get("source_session")) and isinstance(e.get("exchange_index"), int)

    def _set_regen_controls_enabled(self, enabled: bool) -> None:
        self.chk_corrupt_cot.setEnabled(enabled)
        self.chk_corrupt_reply.setEnabled(enabled)
        self.edt_temp.setEnabled(enabled)
        self.edt_suffix.setEnabled(enabled)
        self.cmb_adapter.setEnabled(enabled and self.cmb_adapter.count() > 1)
        self._update_regen_enabled()

    def _update_regen_enabled(self, *_) -> None:
        """Regenerate is live only when the row has provenance and at least one part is
        checked (and the checkboxes themselves are enabled)."""
        checks_on = self.chk_corrupt_cot.isChecked() or self.chk_corrupt_reply.isChecked()
        self.btn_regenerate.setEnabled(self.chk_corrupt_cot.isEnabled() and checks_on)

    def _sync_regen_controls(self, e: dict) -> None:
        """Enable/disable the regenerate controls for the selected entry."""
        has_prov = self._has_provenance(e)
        # Reset the checkboxes on row change (they select what to regenerate, not a
        # persistent state) without firing their toggled handler.
        for chk in (self.chk_corrupt_cot, self.chk_corrupt_reply):
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._set_regen_controls_enabled(has_prov)
        if not has_prov:
            self._lbl_regen.setText("(no source exchange — re-fetch snapshot to enable regeneration)")
        elif e.get("has_live"):
            self._lbl_regen.setText("❄ frozen (human-validated) — showing the live sidecar "
                                    "target; regenerate again to replace it.")
        else:
            self._lbl_regen.setText("")

    def _on_regenerate(self) -> None:
        row = self.lst_entries.currentRow()
        if row < 0 or row >= len(self._entries):
            return
        e = self._entries[row]
        if not self._has_provenance(e):
            self._sync_regen_controls(e)
            return
        corrupt_cot = self.chk_corrupt_cot.isChecked()
        corrupt_response = self.chk_corrupt_reply.isChecked()
        if not (corrupt_cot or corrupt_response):
            self._lbl_regen.setText("Check Corrupt CoT and/or Corrupt reply first.")
            return
        try:
            temperature = float(self.edt_temp.text().strip())
        except ValueError:
            self._lbl_regen.setText("Temperature must be a number.")
            return
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_regen.setText("Not connected — connect from the Chat tab to regenerate.")
            return
        if self._regen_worker is not None and self._regen_worker.isRunning():
            return
        system_suffix = self.edt_suffix.text().strip()
        # Which adapter answers: "" = the loaded one; a name = an alternate from the
        # lineage the server swaps in for this generation (two full model reloads).
        adapter = str(self.cmb_adapter.currentData() or "")
        # Remember what the operator asked to regenerate; the Apply step reuses it.
        # `kept_reply` is the answer currently shown (the trained target's reply) — for a
        # CoT-only regen the server grafts the new CoT onto THIS, not the raw transcript
        # reply, so "keep the reply" keeps the one the operator is looking at.
        self._pending = {
            "row": row,
            "filename": e["source_session"],
            "exchange_index": e["exchange_index"],
            "corrupt_cot": corrupt_cot,
            "corrupt_response": corrupt_response,
            "kept_reply": e.get("answer", ""),
        }
        self.btn_regenerate.setEnabled(False)
        self._lbl_regen.setText(
            f"Regenerating with adapter {adapter}… (loads it first if needed; it stays "
            "loaded)"
            if adapter else
            "Regenerating with the loaded adapter… (this runs on the GPU)")
        # CoT-only regen: the fresh reply is discarded (Apply grafts only the new <think>
        # onto the kept reply), so tell the server to halt once the CoT closes.
        cot_only = corrupt_cot and not corrupt_response
        # The dialog opens NOW and fills in as the model writes — a re-answer takes GPU
        # minutes, so waiting for a finished result before showing anything leaves the
        # operator with a frozen tab and no way to abandon an obviously bad generation.
        self._regen_result = None
        dlg = RegenReviewDialog(
            self,
            kept_reply=str(self._pending.get("kept_reply") or ""),
            corrupt_cot=corrupt_cot,
            corrupt_response=corrupt_response,
            font=self.text_font,
            state=self._regen_dlg_state,
            adapter=adapter,
        )
        self._regen_dlg = dlg
        worker = RegenerateWorker(
            client, e["source_session"], e["exchange_index"], temperature, system_suffix,
            cot_only=cot_only, adapter=adapter)
        # Signals land on the widget, not the dialog: the dialog is destroyed when the
        # operator dismisses it, and a worker that outlives it would otherwise deliver into
        # a dead object. The widget forwards only while `_regen_dlg` is live.
        worker.delta.connect(self._on_regen_delta)
        worker.status.connect(self._on_regen_status)
        worker.done.connect(self._on_regen_done)
        worker.failed.connect(self._on_regen_failed)
        dlg.stop_requested.connect(worker.stop)
        self._regen_worker = worker
        worker.start()

        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        self._regen_dlg = None
        # However it ended (Apply / Cancel / dismissed mid-generation), the size it ended at
        # is the one the next regeneration opens with.
        self._regen_dlg_state = dlg.save_state()
        if worker.isRunning():
            # Dismissed mid-generation — stop the GPU work rather than let it run out the
            # token budget for a result nobody will read.
            worker.stop()
        if accepted and self._regen_result is not None:
            self._apply_regen(self._regen_result)
        elif not accepted:
            self._lbl_regen.setText("Regeneration discarded.")

    def _on_regen_delta(self, cot: str, reply: str) -> None:
        if self._regen_dlg is not None:
            self._regen_dlg.append_delta(cot, reply)

    def _on_regen_status(self, text: str) -> None:
        # Adapter-swap milestones — shown in the dialog note AND the bar's status label,
        # since the restore milestone can outlive a dismissed dialog.
        if self._regen_dlg is not None:
            self._regen_dlg.set_status(text)
        if text:
            self._lbl_regen.setText(text)

    def _on_regen_done(self, result: dict) -> None:
        self._update_regen_enabled()
        self._regen_result = result
        text = ("Regeneration stopped early — review the partial."
                if result.get("cancelled")
                else "Review the regenerated exchange.")
        # The server reports the authoring adapter off the RUNTIME (sticky swap: the
        # request alone can't name it), so the bar always says whose answer this is.
        if result.get("adapter"):
            text += f" Answered by {result['adapter']} (loaded)."
        self._lbl_regen.setText(text)
        if self._regen_dlg is not None:
            self._regen_dlg.finish(result)

    def _on_regen_failed(self, message: str) -> None:
        self._update_regen_enabled()
        self._lbl_regen.setText(f"Regenerate failed: {message}")
        if self._regen_dlg is not None:
            self._regen_dlg.fail(message)

    def _apply_regen(self, result: dict) -> None:
        pending = getattr(self, "_pending", None)
        if not pending:
            return
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_regen.setText("Not connected — connect from the Chat tab to apply.")
            return
        if self._apply_worker is not None and self._apply_worker.isRunning():
            return
        self._apply_in_flight = True
        self.btn_regenerate.setEnabled(False)
        self._lbl_regen.setText("Applying to the sidecar…")
        self._apply_worker = ApplyRegenWorker(
            client, pending["filename"], pending["exchange_index"],
            corrupt_cot=pending["corrupt_cot"],
            corrupt_response=pending["corrupt_response"],
            new_cot=str(result.get("new_cot") or ""),
            new_reply=str(result.get("new_reply") or ""),
            original_reply=str(pending.get("kept_reply") or ""),
        )
        self._apply_worker.done.connect(self._on_apply_done)
        self._apply_worker.failed.connect(self._on_apply_failed)
        self._apply_worker.start()

    def _on_apply_done(self, result: dict) -> None:
        """Shared completion for both repair routes (regenerate-Apply and hand-edit-Apply).

        Reflect the applied target in the in-memory entry + the boxes, so the tab shows the
        repaired, now-frozen exchange without a full snapshot re-fetch."""
        self._apply_in_flight = False
        self._update_regen_enabled()
        pending = getattr(self, "_pending", None) or {}
        is_edit = pending.get("mode") == "edit"
        row = pending.get("row")
        applied_cot = str(result.get("new_cot") or "")
        applied_reply = str(result.get("new_reply") or "")
        if isinstance(row, int) and 0 <= row < len(self._entries):
            e = self._entries[row]
            # The applied target is now the LIVE (frozen) content: mark it so the row
            # shows the repaired CoT/reply and gets the snowflake, surviving a re-open.
            e["live_cot"] = applied_cot
            e["live_answer"] = applied_reply
            e["has_live"] = True
            e["locked"] = True
            e["key"] = " ".join(applied_reply.split())[:20] or "(empty reply)"
            # The displayed CoT/reply changed, so the cached search haystack is stale.
            self._invalidate_haystack(e)
            # The CoT changed, so its persona verdict did too — a stripped opener must stop
            # being reported (and drop out of the persona filter) without a reload.
            self._rescore_persona(e)
            item = self.lst_entries.item(row)
            if item is not None:
                item.setText(self._item_text(row, e))
                item.setIcon(self._row_icon(e))
            if self._edit_order == e.get("_order"):
                # Those edits just became the stored content — no longer unsaved.
                self._edit_order = None
            if self.lst_entries.currentRow() == row:
                # Re-baseline: the applied content is the new stored content, so the boxes
                # are clean again (Apply/Cancel go back to disabled).
                self._populate_edit_boxes(e)
        # The row's content, frozen state and persona verdict just changed, so an active
        # filter's verdict on it did too — re-apply. Under "Only unfrozen" (or the persona
        # filter) this is the repair loop: the repaired row drops out and the selection
        # advances to the next entry still awaiting review.
        if self._frozen_filter != "all" or self._persona_active or self._search_needle:
            self._apply_hiding()
        # Sync against whatever is SELECTED now (the operator may have moved on mid-apply, or
        # the re-filter may have advanced the selection), not the entry that was applied.
        self._sync_edit_controls(self._current_entry())
        if is_edit:
            self._lbl_edit.setText(
                "Applied the edited CoT + answer to the sidecar; this exchange is now frozen ❄ "
                "(human-validated) — re-reflection and Revisit will preserve it. Note: it "
                "trains from the LIVE sidecar, not this snapshot.")
            return
        src = "new reply" if result.get("reply_source") == "regenerated" else "original reply"
        self._lbl_regen.setText(
            f"Applied to sidecar (new CoT + {src}); this exchange is now locked "
            "(human-validated) — re-reflection and Revisit will preserve it. Note: it trains "
            "from the LIVE sidecar, not this snapshot.")

    def _on_apply_failed(self, message: str) -> None:
        self._apply_in_flight = False
        self._update_regen_enabled()
        pending = getattr(self, "_pending", None) or {}
        self._sync_edit_controls(self._current_entry())
        if pending.get("mode") == "edit":
            # The edits are still in the boxes (and still dirty) — the operator can retry.
            self._lbl_edit.setText(f"Apply failed: {message}")
            return
        self._lbl_regen.setText(f"Apply failed: {message}")

    # ---------------------------------------------------------------- #
    # Rewrite history                                                  #
    # ---------------------------------------------------------------- #

    def _on_rewrite_history(self) -> None:
        """Bake all human-reviewed (locked) targets into their transcripts, then unfreeze.

        A finalize step: after regenerating + locking corrupt exchanges, this makes the
        corrections permanent ground truth (fixing RAG recall, snapshots, and later-exchange
        context — training already reads the reviewed target). Destructive to the original
        transcripts, so it confirms first."""
        client = self._chat_widget._client
        if client is None or not client.is_connected():
            self._lbl_status.setText("Not connected — connect from the Chat tab to rewrite history.")
            return
        if self._rewrite_worker is not None and self._rewrite_worker.isRunning():
            return
        frozen = sum(1 for e in self._entries if e.get("locked"))
        banned = sum(1 for e in self._entries if e.get("banned"))
        # The server settles ALL locked/banned rows on the box, which may exceed what this
        # (possibly filtered / stale) snapshot shows — say so honestly.
        counts = []
        if frozen:
            counts.append(f"{frozen} frozen ❄ {'exchange' if frozen == 1 else 'exchanges'}")
        if banned:
            counts.append(f"{banned} banned 🚫 {'row' if banned == 1 else 'rows'}")
        shown = f"This snapshot shows {' and '.join(counts)}. " if counts else ""
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle("Rewrite history?")
        confirm.setText("Rewrite history — bake in reviewed answers and delete banned rows?")
        confirm.setInformativeText(
            shown +
            "For every human-reviewed (locked) exchange on the server, this rewrites the "
            "original chat transcript with the reviewed answer, deletes that exchange's "
            "tension data, and unfreezes it.\n\n"
            "For every banned (🚫) row it goes further and DELETES the source: a banned "
            "exchange is removed from its transcript (the later exchanges shift down one "
            "position), and a banned wander capture is dropped from the wander corpus.\n\n"
            "This edits the original transcripts (ground truth) and cannot be automatically "
            "undone. Prior content is kept in each exchange's rewrite_history, and a deleted "
            "exchange under the session's deleted_exchanges, for manual recovery. Rebuild + "
            "re-fetch a snapshot afterwards to see it here.")
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return
        self.btn_rewrite.setEnabled(False)
        self._lbl_status.setText(
            "Rewriting history (baking reviewed targets in, deleting banned rows)…")
        self._rewrite_worker = RewriteHistoryWorker(client)
        self._rewrite_worker.done.connect(self._on_rewrite_done)
        self._rewrite_worker.failed.connect(self._on_rewrite_failed)
        self._rewrite_worker.start()

    def _on_rewrite_done(self, result: dict) -> None:
        self.btn_rewrite.setEnabled(True)
        n = int(result.get("rewritten", 0))
        deleted = int(result.get("deleted", 0))
        wander_deleted = int(result.get("wander_deleted", 0))
        chats = result.get("chats") or []
        skipped = result.get("skipped") or []
        errors = result.get("errors") or []
        if not (n or deleted or wander_deleted):
            self._lbl_status.setText(
                "Rewrite history: nothing to do — no human-reviewed (locked) or banned rows found.")
            return
        parts = []
        if n:
            parts.append(f"Rewrote {n} exchange{'s' if n != 1 else ''} across "
                         f"{len(chats)} chat{'s' if len(chats) != 1 else ''} and unfroze them "
                         "(transcripts updated; tension dropped).")
        if deleted:
            parts.append(f"Deleted {deleted} banned exchange"
                         f"{'s' if deleted != 1 else ''} from their transcripts.")
        if wander_deleted:
            parts.append(f"Dropped {wander_deleted} banned wander "
                         f"capture{'s' if wander_deleted != 1 else ''} from the corpus.")
        if skipped:
            parts.append(f"{len(skipped)} skipped")
        if errors:
            parts.append(f"{len(errors)} error{'s' if len(errors) != 1 else ''}")
        parts.append("Rebuild + re-fetch a snapshot to see it here.")
        self._lbl_status.setText("  ".join(parts))

    def _on_rewrite_failed(self, message: str) -> None:
        self.btn_rewrite.setEnabled(True)
        self._lbl_status.setText(f"Rewrite history failed: {message}")

    # ---------------------------------------------------------------- #
    # Fonts                                                             #
    # ---------------------------------------------------------------- #

    def update_fonts(self, font: QFont) -> None:
        self.text_font = font
        for box in (self.txt_query, self.txt_cot, self.txt_answer):
            box.setFont(font)
