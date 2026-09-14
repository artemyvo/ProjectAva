"""Training-review payload builder — the entry list the Training review tab reviews.

The tab's database is the **latest build's** training render
(``server/models/snapshots/<build_id>/sft_render.jsonl`` — one JSON row per training
row), projected down to what a human reviews: per row the FINAL exchange (the only turn
training targets under final-turn label masking), split into query / CoT / answer, with
duplicate rows collapsed and the LIVE sidecar's reviewed target patched over the frozen
render.

This module exists because that projection used to live in the *client* and read the
three inputs straight off local disk — which pinned the tab to the GPU box (or forced a
Migrate → *Fetch snapshot* pull of the whole personality, weights included, just to read
some text). Building the payload here makes the tab work from a remote UI over the
inference HTTP sidecar (``GET /training/review``), and keeps ONE definition of the
projection: the client's same-box path loads *this file* rather than a second copy.

The projection is small compared to its inputs — the render carries every row's whole
conversation, and the tab shows only the last turn (on the 2026-07-24 build: 5.7 MB of
final-turn text out of a 33.8 MB render), so the wire payload is a fraction of the file
it is derived from and no adapter weights move at all.

Four inputs, all read-only:

  * the newest ``models/snapshots/*/sft_render.jsonl`` by mtime — the forensic corpus;
  * the live ``data/chats/*.state.json`` sidecars — a **locked** (human-validated)
    exchange carries its reviewed target on disk, which supersedes the frozen render;
    that row is shown repaired and flagged frozen (the tab's ❄). The same record says
    whether the operator **banned** the exchange from training (the tab's 🚫);
  * the live ``data/til/wander.jsonl`` corpus — the one live state a wander row has, since
    it has no chat and no sidecar: whether that capture is banned;
  * every ``[persona]`` statement ever written (the raw append-only logs, evicted ones
    included) — the ledger tier of the client's persona-opener detector. Shipped
    normalized so the detector's lexical tests run unchanged; scoring itself stays in
    the client, so a repaired row can be rescored in place without a round trip.

Pure stdlib (json / pathlib / re) and no project imports, so the client can load it by
path with no server dependencies on its side.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Self-located roots (server/inference/core/ → server/). Overridable per call so the
# client can point at its own checkout, and so a test can build against a fixture tree.
_DEFAULT_SERVER_ROOT = Path(__file__).resolve().parent.parent.parent

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class ReviewUnavailable(RuntimeError):
    """No reviewable corpus — carries the operator-facing explanation verbatim."""


# ---------------------------------------------------------------------- #
# Render parsing                                                          #
# ---------------------------------------------------------------------- #

def split_think(text: str) -> tuple[str, str]:
    """Split an assistant turn into ``(cot, answer)`` around ``<think>…</think>``.

    Handles the canonical ``<think>…</think>answer`` form, a bare closing tag (the
    opening ``<think>`` may be prefilled into the prompt for some families), and a
    turn with no thinking at all (all answer)."""
    if not text:
        return "", ""
    close = "</think>"
    open_ = "<think>"
    i_close = text.find(close)
    if i_close != -1:
        head = text[:i_close]
        i_open = head.find(open_)
        cot = head[i_open + len(open_):] if i_open != -1 else head
        return cot.strip(), text[i_close + len(close):].strip()
    i_open = text.find(open_)
    if i_open != -1:                       # unclosed think — treat the rest as CoT
        return text[i_open + len(open_):].strip(), ""
    return "", text.strip()                # no thinking markers — all answer


def _final_turns(messages: list) -> tuple[str, str]:
    """The final user + final assistant content (the turn training targets)."""
    user = ""
    assistant = ""
    for m in messages or []:
        role = m.get("role")
        if role == "user":
            user = m.get("content", "") or ""
        elif role == "assistant":
            assistant = m.get("content", "") or ""
    return user, assistant


def _dedup_key(row: dict, query: str, assistant: str):
    """Identity of the *exchange* a render row targets, for collapsing duplicate rows.

    A cap-age exchange emits two rows carrying the same rendered messages — the masked one
    and its user-contamination copy (``unmask_user``), differing only in LR multiplier
    (see ``training/build_dataset._contamination_rows``). Prefer the chat provenance;
    fall back to the rendered text for wander rows and pre-provenance snapshots."""
    src = row.get("source_session")
    idx = row.get("exchange_index")
    if isinstance(src, str) and isinstance(idx, int):
        return ("exchange", src, idx)
    return ("text", query, assistant)


def latest_snapshot_render(server_root: Path) -> Optional[Path]:
    """Newest ``models/snapshots/<build_id>/sft_render.jsonl`` by mtime, or None."""
    snaps = server_root / "models" / "snapshots"
    if not snaps.is_dir():
        return None
    renders = [d / "sft_render.jsonl" for d in snaps.iterdir()
               if (d / "sft_render.jsonl").is_file()]
    if not renders:
        return None
    return max(renders, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------- #
# Live sidecar connect-back                                               #
# ---------------------------------------------------------------------- #

def _live_chat_sidecar_dirs(server_root: Path) -> list:
    """Where the live per-chat consolidation sidecars live (active then archive).

    The render is a frozen forensic copy; the live ``.state.json`` sidecars carry the
    *current* target (incl. an operator's regenerated + locked one).

    The chat corpus was moved out of ``inference/data/hot/chats`` to the ordered
    ``server/data/chats`` home (see ``training/reflections_path.hot_chats_dir`` — the
    authoritative resolver, which the apply handler writes through); ``archive/chats``
    under the same root is a legacy path that may not exist (``.is_file`` guards it)."""
    base = server_root / "data"
    return [base / "chats", base / "archive" / "chats"]


def _load_live_sidecar(dirs: list, source_session: str, cache: dict) -> Optional[dict]:
    """Load a chat's ``.state.json`` sidecar (searching *dirs* in order), memoised per chat."""
    if source_session in cache:
        return cache[source_session]
    stem = (source_session[:-len(".json")] if source_session.endswith(".json")
            else source_session)
    doc = None
    for d in dirs:
        p = d / (stem + ".state.json")
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    doc = loaded
            except Exception:
                doc = None
            break
    cache[source_session] = doc
    return doc


# ---------------------------------------------------------------------- #
# Wander corpus (the non-chat rows' live state)                           #
# ---------------------------------------------------------------------- #

WANDER_PREFIX = "wander:"


def banned_wander_timestamps(server_root: Path) -> set:
    """Capture timestamps of wander records the operator banned from training.

    A wander row has no chat and no sidecar, so its live state lives in the corpus file
    itself (``server/data/til/wander.jsonl``, one JSON record per line, ``banned: true`` set
    by ``core.wander_sft.set_banned``). The render row identifies it as
    ``source_session == "wander:<ts>"``, which is the join used here. Read directly rather
    than through ``wander_sft`` so this module stays pure stdlib — the client loads it by
    path when it has no connection, with no server package available."""
    path = server_root / "data" / "til" / "wander.jsonl"
    out: set = set()
    try:
        fh = path.open("r", encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("banned"):
                ts = str(rec.get("ts") or "")
                if ts:
                    out.add(ts)
    return out


# ---------------------------------------------------------------------- #
# Persona statements (ledger tier of the client's opener detector)        #
# ---------------------------------------------------------------------- #

_persona_cache: dict = {}     # server_root -> (fingerprint, [normalized statement, ...])


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def persona_statements(server_root: Path) -> list:
    """Every `[persona]` statement ever written, **normalized**, duplicates collapsed.

    Read from the RAW append-only logs rather than the folded live set: an injection
    happened while its statement was live, so a since-evicted persona still explains a
    line in an old target. Both files are read (the weights-bound provenance log and the
    anchor ledger). A missing file is not an error — the detector simply falls back to
    its shape tier.

    The anchor ledger runs to tens of MB, so the result is memoised against the two
    files' (size, mtime): an operator refreshing the tab does not re-read it each time."""
    paths = [
        server_root / "inference" / "data" / "hot" / "memory" / "weights_persona.jsonl",
        server_root / "inference" / "data" / "hot" / "consolidation" / "consolidation_anchors.jsonl",
    ]
    fingerprint = []
    for p in paths:
        try:
            st = p.stat()
            fingerprint.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            fingerprint.append((str(p), -1, -1))
    cached = _persona_cache.get(str(server_root))
    if cached and cached[0] == fingerprint:
        return cached[1]

    stmts: dict = {}
    for p in paths:
        try:
            fh = p.open("r", encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if (rec.get("type") or rec.get("kind")) != "persona":
                    continue
                content = (rec.get("content") or "").strip()
                if not content:
                    continue
                n = _norm(content)
                if n:
                    stmts[n] = True
    out = list(stmts)
    _persona_cache[str(server_root)] = (fingerprint, out)
    return out


# ---------------------------------------------------------------------- #
# Adapter lineage (the regenerate flow's alternate-adapter choices)       #
# ---------------------------------------------------------------------- #

def _config_adapter_name(server_root: Path) -> str:
    """Basename of the adapter ``server_config.json`` points at ('' when none).

    Read-only fallback pair (current home, then the pre-2026-07-28 legacy path) —
    the same discipline as snapshot_state / mgmt_http, kept inline so this module
    stays pure stdlib with no project imports."""
    for p in (server_root / "server_config.json",
              server_root / "inference" / "server_config.json"):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(cfg, dict):
            aid = str(cfg.get("adapter_id") or "").strip()
            return Path(aid).name if aid else ""
    return ""


def list_adapters(server_root: Path) -> list:
    """The LoRA adapter lineage under ``models/`` — the regenerate flow's choices.

    One entry per adapter directory holding an ``adapter_config.json`` (the transient
    ``candidate`` a mid-build cycle stages is skipped, as is ``snapshots``), annotated
    from ``models/builds.jsonl`` when that log knows the adapter (``build_id`` + whether
    the build's forensic snapshot is still on disk) and flagged ``current`` when it is
    the one ``server_config.json`` points at. Newest first (build timestamp, else dir
    mtime), so a dropdown's second entry is the natural "last known good" rollback.

    Entries carry the bare directory NAME, never a path: the client picks from this
    list and sends the name back over the wire, and the server re-resolves it under
    its own ``models/`` — an absolute path would break the moment the tab runs from
    a remote UI box (the whole reason this payload is server-built)."""
    models_dir = server_root / "models"
    if not models_dir.is_dir():
        return []

    # builds.jsonl annotation: adapter dir name -> latest build line naming it.
    builds: dict = {}
    builds_path = models_dir / "builds.jsonl"
    try:
        for line in builds_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            adir = str(rec.get("adapter_dir") or "").strip()
            if adir:
                builds[Path(adir).name] = rec    # later lines win: newest build
    except OSError:
        pass

    current = _config_adapter_name(server_root)
    snaps_dir = models_dir / "snapshots"
    out: list = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir() or d.name in ("snapshots", "candidate"):
            continue
        if not (d / "adapter_config.json").is_file():
            continue
        rec = builds.get(d.name) or {}
        snap = str(rec.get("snapshot_dir") or "").strip()
        # A migrated box's builds.jsonl may carry the SOURCE box's absolute path;
        # the snapshot travels by basename, so check both readings.
        has_snapshot = bool(snap) and (Path(snap).is_dir()
                                       or (snaps_dir / Path(snap).name).is_dir())
        try:
            mtime = d.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append({
            "name": d.name,
            "build_id": str(rec.get("build_id") or ""),
            "ts": str(rec.get("ts") or ""),
            "has_snapshot": has_snapshot,
            "current": d.name == current,
            "_mtime": mtime,
        })
    # Newest first: the build log's ISO ts when known (lexicographic == chronological),
    # else the directory mtime — comparable enough for ordering a short lineage.
    out.sort(key=lambda a: (a["ts"] or "", a["_mtime"]), reverse=True)
    for a in out:
        a.pop("_mtime", None)
    return out


# ---------------------------------------------------------------------- #
# Payload                                                                 #
# ---------------------------------------------------------------------- #

def build_payload(*, source: str = "snapshot",
                  server_root: Optional[Path] = None) -> dict:
    """The Training review tab's entry list.

    *source* selects the corpus. Only ``"snapshot"`` exists today — the latest build's
    forensic render, i.e. exactly the rows the current adapter was fitted on. (A future
    ``"live"`` source would assemble the NEXT build's corpus from the live chats via
    ``training.build_dataset``; the parameter is here so that arrives as a value rather
    than a second endpoint.)

    Returns ``{source, build_id, outcome, path, rows, entries, persona_statements}`` —
    ``outcome`` is the snapshot's ``build_meta.json`` outcome ('' when unreadable):
    ``"promoted"``/``"rejected"`` for a real build, ``"preview"`` for a training-lite
    preview snapshot (``training/preview_build.py``) in which NOTHING trained. One entry per
    reviewed *exchange* — duplicate render rows are collapsed onto ``copies`` — carrying
    the frozen render's ``cot``/``answer``, the live sidecar's ``live_cot``/``live_answer``
    when the exchange is ``locked``, its ``kind`` (``"chat"``/``"wander"``) and ``banned``
    state, its ``preview`` flag (an untrained fresh-chat row rendered for early repair —
    Sleep's "Include fresh chats"), and the ``source_session``/``exchange_index``
    provenance the repair RPCs need. Entries stay in file order == training order
    (preview rows trail the trained corpus).

    Raises ``ReviewUnavailable`` with an operator-facing message when there is nothing to
    review."""
    if source != "snapshot":
        raise ReviewUnavailable(f"Unknown training-review source: {source!r}.")
    root = Path(server_root) if server_root is not None else _DEFAULT_SERVER_ROOT

    render = latest_snapshot_render(root)
    if render is None:
        raise ReviewUnavailable(
            "No build snapshot found under server/models/snapshots/.\n"
            "The tab reviews the corpus of the latest training build — run a "
            "training cycle (Sleep → Train) first."
        )

    entries: list = []
    seen: dict = {}                     # dedup key -> index into `entries`
    sidecar_dirs = _live_chat_sidecar_dirs(root)
    sidecar_cache: dict = {}
    banned_wanders = banned_wander_timestamps(root)
    with render.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query, assistant = _final_turns(row.get("messages"))
            # One exchange, many rows: a cap-age exchange renders again as its
            # user-contamination copy. Count the copies onto the first entry rather
            # than listing the same exchange twice for review.
            dkey = _dedup_key(row, query, assistant)
            if dkey in seen:
                entries[seen[dkey]]["copies"] += 1
                continue
            cot, answer = split_think(assistant)
            # Provenance back to the origin exchange (added to the render row so the
            # review tab can regenerate against the source chat). Older snapshots lack
            # these; the client disables regeneration for such rows.
            src = row.get("source_session")
            idx = row.get("exchange_index")
            src = src if isinstance(src, str) else None
            idx = idx if isinstance(idx, int) else None

            # Connect back to the LIVE sidecar: a locked (human-validated /
            # regenerated) exchange carries its reviewed target on disk, which
            # supersedes this frozen render. Show that content in the preview and
            # flag the row frozen.
            locked = False
            banned = False
            live_cot = live_answer = None
            # A wander row's live state lives in the wander corpus, not a sidecar — it is a
            # one-shot capture with no chat behind it, which is also why it can only ever be
            # banned, never repaired.
            kind = "wander" if (src or "").startswith(WANDER_PREFIX) else "chat"
            if kind == "wander":
                banned = src[len(WANDER_PREFIX):] in banned_wanders
            elif src and idx is not None:
                doc = _load_live_sidecar(sidecar_dirs, src, sidecar_cache)
                if isinstance(doc, dict):
                    rec = (doc.get("exchanges") or {}).get(str(idx))
                    if isinstance(rec, dict):
                        banned = bool(rec.get("banned"))
                        if rec.get("locked"):
                            locked = True
                            tgt = rec.get("target")
                            if isinstance(tgt, str) and tgt.strip():
                                live_cot, live_answer = split_think(tgt)
            has_live = live_cot is not None or live_answer is not None
            disp_answer = live_answer if has_live else answer
            key = " ".join((disp_answer or "").split())[:20] or "(empty reply)"
            seen[dkey] = len(entries)
            entries.append({"key": key, "query": query,
                            "cot": cot, "answer": answer,
                            "live_cot": live_cot or "", "live_answer": live_answer or "",
                            "has_live": has_live, "locked": locked,
                            "banned": banned, "kind": kind,
                            # A fresh preview row (Sleep → "Include fresh chats"): rendered
                            # at LR multiplier 0 so it is reviewable/repairable BEFORE it
                            # ages into a real build — this build did NOT train it.
                            "preview": bool(row.get("preview")),
                            "copies": 1,
                            "source_session": src, "exchange_index": idx})

    # The snapshot's own outcome: a real build's is "promoted"/"rejected"; a
    # training-lite preview snapshot's (training/preview_build.py) is "preview" —
    # NOTHING in it trained, and the tab must be able to say so beside the build id.
    outcome = ""
    try:
        meta = json.loads((render.parent / "build_meta.json").read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            outcome = str(meta.get("outcome") or "")
    except Exception:
        pass

    return {
        "source": "snapshot",
        "build_id": render.parent.name,
        "outcome": outcome,
        "path": str(render),
        "rows": sum(e["copies"] for e in entries),
        "entries": entries,
        "persona_statements": persona_statements(root),
        # The adapter lineage beside the render — what the tab's regenerate flow may
        # ask the server to swap in (so a corrupt row can be re-answered by the last
        # known GOOD adapter rather than the one that produced the corruption).
        "adapters": list_adapters(root),
    }


if __name__ == "__main__":      # GPU-free self-test / CLI summary of the local corpus
    import sys
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _DEFAULT_SERVER_ROOT
    try:
        payload = build_payload(server_root=root)
    except ReviewUnavailable as exc:
        print(exc)
        raise SystemExit(1)
    ents = payload["entries"]
    frozen = sum(1 for e in ents if e["locked"])
    banned = sum(1 for e in ents if e["banned"])
    preview = sum(1 for e in ents if e.get("preview"))
    traced = sum(1 for e in ents if e["source_session"] and e["exchange_index"] is not None)
    print(f"{banned} banned from training "
          f"({sum(1 for e in ents if e['kind'] == 'wander')} wander rows)")
    size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(f"snapshot {payload['build_id']}: {len(ents)} entries "
          f"from {payload['rows']} rows, {frozen} frozen, {preview} fresh preview, "
          f"{traced} with chat provenance, "
          f"{len(payload['persona_statements'])} persona statements")
    adapters = payload.get("adapters") or []
    print(f"adapter lineage: {len(adapters)} "
          f"({', '.join(a['name'] + (' *' if a.get('current') else '') for a in adapters)})"
          if adapters else "adapter lineage: none")
    print(f"payload {size / 1e6:.1f} MB (render {Path(payload['path']).stat().st_size / 1e6:.1f} MB)")
