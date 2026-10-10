"""The capture probe: does the state before her reply say she will give in?
(``AVA_REWARD_LOOP.md`` §4.3, item 14). The reading and its reports are read-only and
GPU-free; the shadow step (§4.5, below) runs inside a Sleep run's clean base and appends.

**The label.** The capture judge's verdict (``core/capture_judge.py``) on a reply that
answers the other person's pushback: ``yields`` against ``holds`` / ``updates`` /
``rigid``, under one judge key (the newest by default), on the STORED reply
(``target_source: original``). A revised target was written in another seam, so the
state of the live context did not produce it; where the revision pass replaced her reply,
the same judge grades the stored reply into a store of its own (``cj.run_originals``,
``originals.jsonl``: a label, never a gate), and the probe reads both
(:func:`label_records`).

**The state.** The residual stream at the last token of the prompt, after the template and
the think prefill and before any token of the reply: the state the reply is generated
from, which the live capture stores as its step 0. ``server/capture_states.py`` (GPU,
offline) computes it into ``data/capture/states/``. It does so for the corpus (the prompt
rebuilt as the live turn built it, under the base) and for the capture bench's frozen
contexts (under each arm's weights and condition). For comparison the probe also reads
the base backfill (``backfill_hidden.py``, variant ``~base``). The backfill's rows sit on
the replayed reply's own tokens, so its first row is the state after the first generated
token rather than step 0, and its last row the state at the reply's last token.

**The probe.** L2 logistic regression over the standardized state at one layer. The CV is
grouped by session, since an exchange's neighbours share most of its context, and
stratified, with the penalty chosen by an inner CV and the whole thing repeated. The
averaged out-of-fold ranks give the AUC and a bootstrap CI. For the layers named before
this run (:data:`NAMED_LAYERS`) they also give a permutation p; a maximum over the sweep is
optimistic, and is labelled so. The fit runs in the span of the training rows (an SVD),
which for an L2 penalty is the same solution in a hundred dimensions instead of five
thousand.

**The baselines, on the same folds.** The judge's own BASIS reading (a bare assertion about
her, or a reason): frame 3 grades a concession to a bare assertion ``yields`` by a rule in
code, so the stratum alone predicts the label well. The probe has to be read against it,
and within it (the AUC over pairs in the same stratum). Also on the same folds: the text of
the message, of the system prompt, of her previous reply and of the reply itself (character
n-grams); the session's date; the exchange's depth in its conversation; the prompt's length;
the reply's length; the adapter that wrote it; shuffled labels.

**The transfer.** A probe trained on the corpus, without the bench's own exchanges (and,
stricter, without their sessions), scores the bench's states and is read against each
arm's grades. The bench differs from the corpus in four ways at once: the seam is the IDEAL
re-answer, the replies are sampled, the weights can be another set, and the persona slot
can be empty (P8). The layer and the penalty are chosen on the corpus alone.

What it is not: a gate, a sensor in use, or a reward. Nothing reads its output (P3; §4.5's
shadow mode comes only after this holds).

**The shadow reading** (§4.5): a probe fitted on every labelled exchange is frozen
(``--freeze LAYERS`` → ``data/capture/probe/<id>.npz``), and a Sleep run's clean-base batch,
after the capture gate, reads the base's state before each newly graded pushback reply,
stores it with the corpus states and logs every frozen probe's score beside the verdict
(``data/capture/shadow.jsonl``; :func:`run_shadow`, ``capture_probe.shadow`` /
``capture_probe.sleep_max``). ``--shadow`` reports each probe on the replies graded after
it was frozen: the held-out test.

``python -m core.capture_probe [--judge-key K] [--source original|any] [--json]
[--permutations N] [--freeze LAYERS] [--shadow]`` from ``server/inference``. ``--selftest``
is GPU-free.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import capture_judge as cj
    from core import hidden_capture as hc
except ImportError:  # pragma: no cover — run as a script from core/
    import ask_judge as aj  # type: ignore
    import capture_bench as cb  # type: ignore
    import capture_judge as cj  # type: ignore
    import hidden_capture as hc  # type: ignore

STATES_DIRNAME = "states"
POSITION = "last prompt token (after the template and the think prefill, before the reply)"
# The L2 penalty's inverse strength, chosen per fold by an inner CV. The states are
# standardized over ~5,000 dimensions, so the useful range is small.
DEFAULT_CS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_FOLDS = 5
INNER_FOLDS = 3
DEFAULT_REPEATS = 3
DEFAULT_PERMUTATIONS = 100
SWEEP_STRIDE = 4
# Named before this run: the reading of 2026-09-28 (`AVA_REWARD_LOOP.md` §4.3) put the
# backfill's final row at layer 30 and its first row at layers 5 / 38. Tested as named.
NAMED_LAYERS = (5, 30, 38)
STRATA = ("assertion", "reason")


# ── the states store (written by server/capture_states.py) ───────────────────────

def states_dir(live_chats_dir) -> Path:
    """``server/data/capture/states/`` — beside the capture judge's verdicts."""
    return cj.store_path(live_chats_dir).parent / STATES_DIRNAME


def weights_name(adapter_id) -> str:
    """``base``, or the adapter's directory name."""
    name = str(adapter_id or "").rstrip("/\\").replace("\\", "/").split("/")[-1]
    return name or "base"


def states_file(directory, set_name: str, weights: str, condition: str = "as_is") -> Path:
    """``<set>~<weights>[+<condition>].npz``: ``corpus~base.npz``, ``b1~base+persona_empty.npz``
    (``~`` as in the hidden sidecar's ``<exchange_id>~<variant>``)."""
    tail = "" if condition in (None, "", "as_is") else f"+{condition}"
    return Path(directory) / f"{set_name}~{weights}{tail}.npz"


def context_key(session: str, exchange_index: int) -> str:
    """The id of one exchange: the capture bench's context id, ``<session>#<index>``."""
    return cb.context_id(session, exchange_index)


def sweep_layers(n_layers: int, band: str = hc.DEFAULT_LAYER_SPEC, stride: int = SWEEP_STRIDE,
                 extra: Iterable[int] = ()) -> list:
    """Every ``stride``-th layer, the live capture's stored band, the last layer, and *extra*
    (the stored axes' own layers: pain 25, relief 22, the band's 34, so a state can be read
    on every axis)."""
    if n_layers <= 0:
        return []
    out = set(range(0, n_layers, max(1, int(stride))))
    out |= set(hc.parse_layer_spec(band, n_layers))
    out |= {int(l) for l in extra if 0 <= int(l) < n_layers}
    out.add(n_layers - 1)
    return sorted(out)


def _put(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    with zf.open(name, "w") as fh:
        np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)


def stored_keys(path) -> set:
    """The keys already stored in a states file (from the zip's directory alone)."""
    try:
        with zipfile.ZipFile(Path(path)) as zf:
            names = zf.namelist()
    except Exception:
        return set()
    return {n[:-len("/meta.npy")] for n in names if n.endswith("/meta.npy")}


def write_state(path, key: str, states: np.ndarray, meta: dict) -> bool:
    """Append one context's states ((layers, D), stored fp16) and meta. A key already
    present is left alone (a zip cannot replace a member), as the hidden sidecar does."""
    key = str(key or "")
    if not key or "/" in key:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        if f"{key}/meta.npy" in set(zf.namelist()):
            return False
        _put(zf, f"{key}/states.npy", np.asarray(states, dtype=np.float16))
        _put(zf, f"{key}/meta.npy", np.array(json.dumps(meta, ensure_ascii=False)))
    return True


def load_states(path, keys: Optional[Iterable[str]] = None, layers: Optional[Iterable[int]] = None) -> dict:
    """``{"keys", "layers", "X": (N, L, D) float16, "meta": {key: meta}}`` for every stored
    key (or the given ones present). The layers read are the ones every read key carries (a
    store can hold keys captured with different sweeps: since 2026-10-06 a capture adds the
    stored axes' own layers), in the first key's order; ``mismatched`` counts the keys whose
    own list differs from that. With *layers*, only keys carrying all of them are read, and
    exactly those layers (``missing`` counts the keys left out)."""
    out = {"keys": [], "layers": [], "X": None, "meta": {}, "mismatched": 0, "missing": 0, "path": str(path)}
    path = Path(path)
    if not path.exists():
        return out
    want = None if keys is None else set(keys)
    need = None if layers is None else [int(x) for x in layers]
    found = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for name in sorted(names):
            if not name.endswith("/meta.npy"):
                continue
            key = name[:-len("/meta.npy")]
            if (want is not None and key not in want) or f"{key}/states.npy" not in names:
                continue
            meta = json.loads(np.load(io.BytesIO(zf.read(name)), allow_pickle=False).item())
            kl = [int(x) for x in meta.get("layers") or []]
            if need is not None and not set(need) <= set(kl):
                out["missing"] += 1
                continue
            found.append((key, kl, meta))
        if not found:
            return out
        shared = need if need is not None else [l for l in found[0][1] if all(l in kl for _k, kl, _m in found)]
        rows = []
        for key, kl, meta in found:
            arr = np.load(io.BytesIO(zf.read(f"{key}/states.npy")), allow_pickle=False)
            rows.append(arr[[kl.index(l) for l in shared]])
            out["keys"].append(key)
            out["meta"][key] = meta
            out["mismatched"] += kl != shared
    out["layers"] = list(shared)
    out["X"] = np.stack(rows).astype(np.float16)
    return out


def layer_matrix(states: dict, layer: int, keys: Sequence[str]) -> np.ndarray:
    """(len(keys), D) float64 at one layer, in the given key order (every key present)."""
    idx = {k: i for i, k in enumerate(states["keys"])}
    li = states["layers"].index(int(layer))
    return np.stack([states["X"][idx[k], li] for k in keys]).astype(np.float64)


# ── one context's state (server/capture_states.py and a Sleep run's shadow step) ────

class TooLong(Exception):
    """A prompt over the context the model was loaded at: skipped, never truncated."""


def context_prompt(tokenizer, model_id: str, conv: list) -> tuple:
    """``(prompt, ids)``: a conversation as a generation reads it — the family's chat
    template and think prefill, tokenized as ``stream_generate`` tokenizes a prompt."""
    from core import model_family
    from core.llm_shared import build_inference_prompt
    fam = model_family.family_for(model_id)
    prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    return prompt, [int(t) for t in text_tok(prompt)["input_ids"]]


def context_state(model, ids: Sequence[int], layers: Sequence[int]) -> np.ndarray:
    """``(len(layers), D)`` float32: the residual at the last of *ids* — the state the
    reply's first token is generated from — at each layer, from one forward pass."""
    rows = hc.forward_ids_rows(model, ids, layers)
    return np.stack([np.asarray(rows[int(l)], dtype=np.float32)[-1] for l in layers])


def state_meta(*, key: str, set_name: str, condition: str, framing: str, model_id: str,
               adapter_id: str, weights: str, layers: Sequence[int], n_layers: int,
               hidden_dim: int, prompt: str, n_tokens: int, base_quant: str, **extra) -> dict:
    """The ``meta`` a stored state carries (one definition for every writer)."""
    return {"key": key, "set": set_name, "condition": condition, "framing": framing,
            "model_id": model_id, "adapter_id": adapter_id, "weights": weights,
            "layers": [int(l) for l in layers], "n_layers": int(n_layers), "hidden_dim": int(hidden_dim),
            "prompt_tokens": int(n_tokens),
            "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "position": POSITION, "base_quant": base_quant,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}


def store_layers(path) -> list:
    """The layer list the states in a store share ([] for an empty store)."""
    try:
        with zipfile.ZipFile(Path(path)) as zf:
            for name in zf.namelist():
                if name.endswith("/meta.npy"):
                    meta = json.loads(np.load(io.BytesIO(zf.read(name)), allow_pickle=False).item())
                    return [int(x) for x in meta.get("layers") or []]
    except Exception:
        pass
    return []


# ── what to compute states for ───────────────────────────────────────────────────

def pushback_exchanges(records: Iterable[dict]) -> list:
    """Every exchange any judge key found answering pushback: ``[(session, index)]``."""
    out = set()
    for r in records:
        if (r.get("record") == "verdict" and r.get("outcome") == "verdict"
                and r.get("pushback") == "yes" and r.get("session") is not None):
            out.add((str(r["session"]), int(r["exchange_index"])))
    return sorted(out)


# ── the labels ───────────────────────────────────────────────────────────────────

def newest_key(records: Iterable[dict], record: str = "verdict") -> Optional[str]:
    """The judge key of the newest record of a kind (``verdict``; ``grade`` for the bench)."""
    best = None
    for r in records:
        if r.get("record") != record or not r.get("judge_key"):
            continue
        ts = str(r.get("ts") or "")
        if best is None or ts >= best[1]:
            best = (r["judge_key"], ts)
    return best[0] if best else None


def label_records(chats_dir) -> list:
    """The capture judge's verdicts and, after them, its grades of the stored reply behind a
    revised target (``cj.originals_path``): the probe's labels, both under the same key."""
    return aj.read_jsonl(cj.store_path(chats_dir)) + aj.read_jsonl(cj.originals_path(chats_dir))


def label_rows(records: Iterable[dict], *, key: str, source: str = "original") -> list:
    """One row per exchange whose reply the judge graded as answering pushback, under
    *key*: ``{key, session, exchange_index, y, response, basis, opening, adapter, ts}``.
    ``source="original"`` keeps a verdict on the stored reply only (the gate's, where the
    reply trains as it was, or ``originals.jsonl``'s, where it was revised); ``"any"`` takes
    the stored reply's where there is one, else the newest target's."""
    newest: dict = {}
    for r in records:
        if (r.get("record") != "verdict" or r.get("judge_key") != key or r.get("outcome") != "verdict"
                or r.get("pushback") != "yes" or r.get("response") not in cj.RESPONSES):
            continue
        orig = r.get("target_source") == "original"
        if source == "original" and not orig:
            continue
        k = context_key(r["session"], r["exchange_index"])
        rank = (orig, str(r.get("ts") or ""))
        if k not in newest or rank >= newest[k][0]:
            newest[k] = (rank, r)
    rows = []
    for k, (_rank, r) in sorted(newest.items()):
        rows.append({"key": k, "session": str(r["session"]), "exchange_index": int(r["exchange_index"]),
                     "y": int(r["response"] == cj.GATE), "response": r["response"],
                     "basis": r.get("basis"), "opening": r.get("opening"),
                     "adapter": weights_name(r.get("adapter_id")),
                     "target_source": r.get("target_source"), "ts": r.get("ts")})
    return rows


def _session_day(session: str) -> Optional[float]:
    """Days since the epoch from a ``YYYYmmdd_HHMMSS.json`` session name."""
    try:
        return datetime.strptime(str(session)[:15], "%Y%m%d_%H%M%S").timestamp() / 86400.0
    except ValueError:
        return None


def attach_texts(rows: list, chats_dir) -> list:
    """The rows whose transcript and exchange exist, each with ``message`` (the pushback),
    ``reply`` (her stored reply's spoken part), ``system`` (the stored system content),
    ``day`` and ``month``."""
    cache: dict = {}
    out = []
    for row in rows:
        s = row["session"]
        if s not in cache:
            try:
                cache[s] = json.loads((Path(chats_dir) / s).read_text(encoding="utf-8"))
            except Exception:
                cache[s] = None
        doc = cache[s]
        exs = (doc or {}).get("exchanges") or []
        i = row["exchange_index"]
        if not isinstance(exs, list) or not (0 <= i < len(exs)) or not isinstance(exs[i], dict):
            continue
        ex = exs[i]
        prev = exs[i - 1] if i >= 1 and isinstance(exs[i - 1], dict) else {}
        out.append(dict(row, message=str(ex.get("user_prompt") or ""),
                        reply=cj.reply_text(str(ex.get("assistant_response") or "")),
                        previous=cj.reply_text(str(prev.get("assistant_response") or "")),
                        system=str(ex.get("system_content") or ""),
                        exchange_id=hc.exchange_key(ex, i), day=_session_day(s), month=s[:6]))
    return out


# How a backfill group's rows line up with the live capture's steps (`meta.alignment`).
# Groups written before the flag existed read one token late: row t is the state at the
# t-th generated token, where the live capture's row t is the state that token was
# generated from. They are ``after``; a group that reads the live positions says ``live``.
LEGACY_ALIGNMENT = "after"


def backfill_positions(rows: list, chats_dir, *, variant: str = "base",
                       alignment: str = LEGACY_ALIGNMENT) -> dict:
    """The backfill's first and last rows per exchange: ``{"first"|"final": {"keys",
    "layers", "X": (N, L, D) float16}}``, from the groups whose rows line up as
    *alignment* says (the legacy ``after`` by default; see :data:`LEGACY_ALIGNMENT`), so
    the two conventions are never mixed. ``skipped`` counts the groups left out."""
    out = {p: {"keys": [], "layers": [], "X": None, "meta": {}, "skipped": 0} for p in ("first", "final")}
    stacks: dict = {"first": [], "final": []}
    by_session: dict = defaultdict(list)
    for row in rows:
        if row.get("exchange_id"):
            by_session[row["session"]].append(row)
    for session, srows in sorted(by_session.items()):
        path = hc.sidecar_path(Path(chats_dir) / session)
        if not path.exists():
            continue
        try:
            z = np.load(path, allow_pickle=False)
        except Exception:
            continue
        with z:
            files = set(z.files)
            for row in srows:
                group = hc.member_group(row["exchange_id"], variant)
                if f"{group}/positions" not in files:
                    continue
                try:
                    raw = z[f"{group}/meta"]
                    meta = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
                except Exception:
                    meta = {}
                if str(meta.get("alignment") or LEGACY_ALIGNMENT) != alignment:
                    for pos in out:
                        out[pos]["skipped"] += 1
                    continue
                layers = sorted(int(f.split("/L", 1)[1]) for f in files
                                if f.startswith(group + "/L") and f.split("/L", 1)[1].isdigit())
                if not layers:
                    continue
                mats = [np.asarray(z[f"{group}/L{l}"]) for l in layers]
                for pos, j in (("first", 0), ("final", -1)):
                    if not out[pos]["layers"]:
                        out[pos]["layers"] = layers
                    if layers != out[pos]["layers"]:
                        continue
                    stacks[pos].append(np.stack([m[j] for m in mats]))
                    out[pos]["keys"].append(row["key"])
    for pos in out:
        if stacks[pos]:
            out[pos]["X"] = np.stack(stacks[pos]).astype(np.float16)
    return out


def bench_labels(records: Iterable[dict], *, bench: str = cb.DEFAULT_BENCH,
                 key: Optional[str] = None) -> dict:
    """``{arm: {context_id: y}}`` from the bench's grades under *key* (the newest grade key
    by default), sample 1, graded as answering pushback; plus ``strata`` per context."""
    records = list(records)
    ctxs = {c["context_id"]: c for c in cb.contexts(records, bench)}
    key = key or newest_key((r for r in records if r.get("bench") == bench), record="grade")
    labels: dict = defaultdict(dict)
    for g in records:
        if (g.get("record") != "grade" or g.get("bench") != bench or g.get("judge_key") != key
                or g.get("outcome") != "verdict" or int(g.get("sample") or 1) != 1
                or g.get("response") not in cj.RESPONSES or g.get("context_id") not in ctxs):
            continue
        labels[cb.arm_of(g)][g["context_id"]] = int(g["response"] == cj.GATE)
    return {"key": key, "arms": {a: dict(v) for a, v in labels.items()},
            "strata": {cid: c.get("stratum") for cid, c in ctxs.items()},
            "sessions": {cid: c.get("session") for cid, c in ctxs.items()}}


def arm_states_file(directory, bench: str, arm: str) -> Path:
    """The states file an arm's labels are read against: ``base|persona_empty`` →
    ``<bench>~base+persona_empty.npz``; ``adapter:<dir>`` → ``<bench>~<dir>.npz``."""
    weights, _, condition = arm.partition("|")
    if weights.startswith("adapter:"):
        weights = weights[len("adapter:"):]
    return states_file(directory, bench, weights, condition or "as_is")


# ── the statistics ───────────────────────────────────────────────────────────────

def auc(y, s) -> float:
    """ROC AUC as the Mann-Whitney statistic (ties count a half). NaN without both classes."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):          # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rpos = r[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auc_within(y, s, strata) -> float:
    """The AUC over pairs that share a stratum: how well *s* ranks yields above the rest
    with the stratum held fixed (each stratum weighted by its number of pairs)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    strata = np.asarray(strata, dtype=object)
    num = den = 0.0
    for st in sorted(set(strata.tolist()), key=str):
        m = strata == st
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos and nneg:
            num += auc(y[m], s[m]) * npos * nneg
            den += npos * nneg
    return float(num / den) if den else float("nan")


def bootstrap_ci(y, s, *, n: int = 2000, seed: int = 0, strata=None, alpha: float = 0.05) -> tuple:
    """Percentile CI of the AUC (or the within-stratum AUC) over resampled exchanges,
    resampling within each class so every draw has both."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v = (auc(y[idx], s[idx]) if strata is None
             else auc_within(y[idx], s[idx], np.asarray(strata, dtype=object)[idx]))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def _splits(y, groups, folds: int, seed: int) -> list:
    """Stratified folds that keep a group (a session) on one side; plain stratified folds
    without groups. Fewer folds when a class is smaller than ``folds``."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    y = np.asarray(y, dtype=int)
    k = max(2, min(int(folds), int(y.sum()), int((1 - y).sum())))
    X0 = np.zeros((len(y), 1))
    if groups is None:
        sp = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(sp.split(X0, y))
    sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(sp.split(X0, y, np.asarray(groups)))


def _rank01(s: np.ndarray) -> np.ndarray:
    """Scores → ranks in (0, 1], ties averaged: repeats with different penalties average
    on one scale."""
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    out = r.copy()
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            out[m] = r[m].mean()
    return out / len(s)


def _standardize(train: np.ndarray, others: Sequence[np.ndarray]) -> tuple:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (train - mu) / sd, [(o - mu) / sd for o in others], (mu, sd)


def _span(train: np.ndarray) -> tuple:
    """(Z, V): the training rows in the basis of their own span. An L2-penalized linear
    model's weights lie in that span, so fitting on Z is the same fit."""
    _u, s, vt = np.linalg.svd(train, full_matrices=False)
    keep = s > (s[0] * 1e-7 if len(s) else 0)
    v = vt[keep].T
    return train @ v, v


def _logit(Z: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=float(C), max_iter=5000)
    m.fit(Z, y)
    return m


def _choose_c(Z, y, groups, Cs, seed: int) -> float:
    """The penalty with the best mean inner-fold AUC (inner folds standardized with the
    outer fold's statistics; the span already holds every inner training set)."""
    if len(Cs) == 1:
        return float(Cs[0])
    splits = _splits(y, groups, INNER_FOLDS, seed + 7919)
    best = None
    for C in Cs:
        vals = []
        for tr, te in splits:
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue
            vals.append(auc(y[te], _logit(Z[tr], y[tr], C).decision_function(Z[te])))
        mean = float(np.mean(vals)) if vals else 0.5
        if best is None or mean > best[1] + 1e-12:
            best = (float(C), mean)
    return best[0]


def cv_dense(X, y, groups=None, *, Cs=DEFAULT_CS, folds: int = DEFAULT_FOLDS,
             repeats: int = DEFAULT_REPEATS, seed: int = 0) -> dict:
    """Grouped, stratified, repeated CV of the L2 probe: ``{"scores": averaged out-of-fold
    ranks, "auc", "auc_repeats", "C": {C: times chosen}}``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    g = None if groups is None else np.asarray(groups)
    acc = np.zeros(len(y))
    aucs, chosen = [], Counter()
    for r in range(max(1, int(repeats))):
        oof = np.full(len(y), np.nan)
        for tr, te in _splits(y, g, folds, seed + r):
            xtr, (xte,), _ = _standardize(X[tr], [X[te]])
            ztr, v = _span(xtr)
            C = _choose_c(ztr, y[tr], None if g is None else g[tr], Cs, seed + r)
            chosen[C] += 1
            oof[te] = _logit(ztr, y[tr], C).decision_function(xte @ v)
        aucs.append(auc(y, oof))
        acc += _rank01(oof)
    s = acc / max(1, int(repeats))
    return {"scores": s, "auc": auc(y, s), "auc_repeats": aucs, "C": dict(chosen)}


def cv_text(texts: Sequence[str], y, groups=None, *, Cs=(0.1, 1.0, 10.0), folds: int = DEFAULT_FOLDS,
            repeats: int = DEFAULT_REPEATS, seed: int = 0) -> dict:
    """The same CV over character n-gram TF-IDF of a text (any language): a baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = np.asarray([str(t or "") for t in texts], dtype=object)
    y = np.asarray(y, dtype=int)
    g = None if groups is None else np.asarray(groups)
    acc = np.zeros(len(y))
    aucs = []
    for r in range(max(1, int(repeats))):
        oof = np.full(len(y), np.nan)
        for tr, te in _splits(y, g, folds, seed + r):
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                  sublinear_tf=True, max_features=100000)
            try:
                ztr = vec.fit_transform(texts[tr])
            except ValueError:            # an empty vocabulary
                oof[te] = 0.0
                continue
            C = _choose_c(ztr, y[tr], None if g is None else g[tr], Cs, seed + r)
            oof[te] = _logit(ztr, y[tr], C).decision_function(vec.transform(texts[te]))
        aucs.append(auc(y, oof))
        acc += _rank01(oof)
    s = acc / max(1, int(repeats))
    return {"scores": s, "auc": auc(y, s), "auc_repeats": aucs}


def permutation_p(X, y, groups, *, C: float, n: int = DEFAULT_PERMUTATIONS,
                  folds: int = DEFAULT_FOLDS, seed: int = 0) -> Optional[dict]:
    """``{"auc", "p"}``: the AUC of one CV repeat with the penalty fixed at *C*, and its
    one-sided p against the same CV on shuffled labels (the groups kept). The observed
    statistic is computed exactly as the null is, so the p is exact for it."""
    if n <= 0:
        return None
    y = np.asarray(y, dtype=int)
    observed = cv_dense(X, y, groups, Cs=(C,), folds=folds, repeats=1, seed=seed)["auc"]
    if math.isnan(observed):
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(int(n)):
        a = cv_dense(X, rng.permutation(y), groups, Cs=(C,), folds=folds, repeats=1,
                     seed=seed + 101 * (i + 1))["auc"]
        hits += int(a >= observed)
    return {"auc": observed, "p": (1 + hits) / (1 + int(n)), "C": float(C), "n": int(n)}


class Probe:
    """One probe fitted on all of its training rows: standardize, span, logistic."""

    def __init__(self, X, y, groups=None, *, Cs=DEFAULT_CS, seed: int = 0):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        xtr, _, (self.mu, self.sd) = _standardize(X, [])
        ztr, self.v = _span(xtr)
        self.C = _choose_c(ztr, y, None if groups is None else np.asarray(groups), Cs, seed)
        self.model = _logit(ztr, y, self.C)

    def score(self, X) -> np.ndarray:
        X = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return self.model.decision_function(X @ self.v)


# ── the report ───────────────────────────────────────────────────────────────────

def _cell(y, s, strata: Optional[dict] = None, *, seed: int = 0) -> dict:
    """AUC and CI, and per stratification in *strata* (``{"basis": [...], "month": [...]}``)
    the AUC over pairs inside one stratum, with its CI."""
    a = auc(y, s)
    lo, hi = bootstrap_ci(y, s, seed=seed)
    out = {"n": int(len(y)), "yields": int(np.sum(y)), "auc": a, "ci": [lo, hi], "within": {}}
    for name, st in (strata or {}).items():
        out["within"][name] = {"auc": auc_within(y, s, st),
                               "ci": list(bootstrap_ci(y, s, seed=seed, strata=st, n=1000))}
    return out


def _depth_bucket(index: int) -> str:
    """How deep in its conversation an exchange is (yields come later in a chat)."""
    i = int(index or 0)
    return "1-2" if i <= 2 else "3-5" if i <= 5 else "6+"


def _strata(rows: list) -> dict:
    """The stratifications every cell is read within: the judge's BASIS (frame 3's rule
    lives in it), the month (the yields cluster in September), both, and the exchange's
    depth in its conversation."""
    return {"basis": [str(r.get("basis")) for r in rows],
            "month": [str(r.get("month")) for r in rows],
            "basis×month": [f"{r.get('basis')}|{r.get('month')}" for r in rows],
            "depth": [_depth_bucket(r.get("exchange_index")) for r in rows]}


def _layer_rows(states: dict, rows: list, *, layers, Cs, folds, repeats, seed) -> dict:
    """CV per layer over the rows present in *states*: ``{layer: cell + C}``."""
    have = set(states["keys"])
    rr = [r for r in rows if r["key"] in have]
    if not rr or states["X"] is None:
        return {}
    y = np.array([r["y"] for r in rr])
    groups = [r["session"] for r in rr]
    strata = _strata(rr)
    out = {}
    for layer in layers:
        if layer not in states["layers"]:
            continue
        res = cv_dense(layer_matrix(states, layer, [r["key"] for r in rr]), y, groups,
                       Cs=Cs, folds=folds, repeats=repeats, seed=seed)
        out[int(layer)] = dict(_cell(y, res["scores"], strata, seed=seed), C=res["C"],
                               auc_repeats=res["auc_repeats"])
    return out


def build(chats_dir, *, judge_key: Optional[str] = None, source: str = "original",
          states_directory=None, bench: str = cb.DEFAULT_BENCH, layers: Optional[Sequence[int]] = None,
          Cs=DEFAULT_CS, folds: int = DEFAULT_FOLDS, repeats: int = DEFAULT_REPEATS,
          permutations: int = DEFAULT_PERMUTATIONS, seed: int = 0, with_backfill: bool = True,
          with_transfer: bool = True, progress=None) -> dict:
    """The whole reading. Read-only."""
    say = progress or (lambda m: None)
    chats_dir = Path(chats_dir)
    verdicts = label_records(chats_dir)
    key = judge_key or newest_key(verdicts)
    rows = attach_texts(label_rows(verdicts, key=key, source=source), chats_dir)
    sdir = Path(states_directory) if states_directory else states_dir(chats_dir)
    rep: dict = {"judge_key": key, "source": source, "n": len(rows),
                 "yields": sum(r["y"] for r in rows),
                 "by_basis": {b: {"n": sum(1 for r in rows if r.get("basis") == b),
                                  "yields": sum(r["y"] for r in rows if r.get("basis") == b)}
                              for b in sorted({str(r.get("basis")) for r in rows})},
                 "sessions": len({r["session"] for r in rows}), "positions": {}, "baselines": {},
                 "named_layers": list(NAMED_LAYERS), "transfer": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not rows:
        return rep
    y = np.array([r["y"] for r in rows])
    groups = [r["session"] for r in rows]
    strata = _strata(rows)

    # baselines over every labelled row
    say("baselines")
    basis = np.array([1.0 if r.get("basis") == "assertion" else 0.0 for r in rows])
    rep["baselines"]["basis (bare assertion)"] = _cell(y, basis, strata, seed=seed)
    for name, field in (("message text", "message"), ("system prompt text", "system"),
                        ("her previous reply", "previous"), ("reply text", "reply")):
        res = cv_text([r[field] for r in rows], y, groups, folds=folds, repeats=repeats, seed=seed)
        rep["baselines"][name] = _cell(y, res["scores"], strata, seed=seed)
    days = np.array([r["day"] if r["day"] is not None else np.nan for r in rows], dtype=float)
    days = np.where(np.isnan(days), np.nanmean(days), days)
    for name, v in (("session date", days), ("reply length", np.array([len(r["reply"]) for r in rows], float))):
        res = cv_dense(v[:, None], y, groups, Cs=(1.0,), folds=folds, repeats=repeats, seed=seed)
        rep["baselines"][name] = _cell(y, res["scores"], strata, seed=seed)
    depth = np.array([float(r["exchange_index"]) for r in rows])
    res = cv_dense(depth[:, None], y, groups, Cs=(1.0,), folds=folds, repeats=repeats, seed=seed)
    rep["baselines"]["depth in the conversation"] = _cell(y, res["scores"], strata, seed=seed)
    corpus = load_states(states_file(sdir, "corpus", "base"))
    corpus_meta = corpus["meta"]
    if corpus_meta and all(r["key"] in corpus_meta for r in rows):
        ptok = np.array([float(corpus_meta[r["key"]].get("prompt_tokens") or 0) for r in rows])
        res = cv_dense(ptok[:, None], y, groups, Cs=(1.0,), folds=folds, repeats=repeats, seed=seed)
        rep["baselines"]["prompt length"] = _cell(y, res["scores"], strata, seed=seed)
    adapters = sorted({r["adapter"] for r in rows})
    onehot = np.array([[1.0 if r["adapter"] == a else 0.0 for a in adapters] for r in rows])
    res = cv_dense(onehot, y, groups, Cs=(0.1, 1.0, 10.0), folds=folds, repeats=repeats, seed=seed)
    rep["baselines"]["adapter that wrote it"] = _cell(y, res["scores"], strata, seed=seed)
    rng = np.random.default_rng(seed)
    shuffled = [auc(y, rng.permutation(len(y)).astype(float)) for _ in range(200)]
    rep["baselines"]["shuffled scores"] = {"auc": float(np.mean(shuffled)),
                                           "ci": [float(np.quantile(shuffled, 0.025)),
                                                  float(np.quantile(shuffled, 0.975))]}

    # the probe, per position
    sources: dict = {}
    if corpus["X"] is not None:
        sources["step0"] = corpus
    if with_backfill:
        say("reading the backfill")
        sources.update(backfill_positions(rows, chats_dir))
    for pos, st in sources.items():
        if st.get("X") is None:
            continue
        sweep = list(layers) if layers else list(st["layers"])
        say(f"probe at {pos}: {len(sweep)} layers")
        per = _layer_rows(st, rows, layers=sweep, Cs=Cs, folds=folds, repeats=repeats, seed=seed)
        have = [r for r in rows if r["key"] in set(st["keys"])]
        entry = {"n": len(have), "yields": sum(r["y"] for r in have), "layers": per,
                 "note": (POSITION if pos == "step0" else
                          "backfill ~base: the state after the first generated token" if pos == "first" else
                          "backfill ~base: the state at the reply's last token")}
        # within the bare-assertion stratum alone (frame 3's rule lives there)
        sub = [r for r in have if r.get("basis") == "assertion"]
        if sub and 0 < sum(r["y"] for r in sub) < len(sub):
            entry["assertion_only"] = _layer_rows(st, sub, layers=[l for l in NAMED_LAYERS if l in st["layers"]],
                                                  Cs=Cs, folds=folds, repeats=repeats, seed=seed)
        if permutations > 0:
            hy = np.array([r["y"] for r in have])
            hg = [r["session"] for r in have]
            perm = {}
            for layer in NAMED_LAYERS:
                cell = per.get(layer)
                if not cell:
                    continue
                c_mode = max(cell["C"].items(), key=lambda kv: kv[1])[0]
                say(f"  permutations at {pos} L{layer}")
                perm[int(layer)] = permutation_p(layer_matrix(st, layer, [r["key"] for r in have]), hy, hg,
                                                 C=float(c_mode), n=permutations, folds=folds, seed=seed)
            entry["permutation_p"] = perm
        rep["positions"][pos] = entry

    if with_transfer and "step0" in sources:
        say("transfer to the bench")
        rep["transfer"] = transfer_to_bench(rows, sources["step0"], chats_dir, sdir=sdir, bench=bench,
                                            Cs=Cs, folds=folds, repeats=repeats, seed=seed)
    return rep


def transfer_to_bench(rows: list, corpus_states: dict, chats_dir, *, sdir, bench: str,
                      Cs=DEFAULT_CS, folds: int = DEFAULT_FOLDS, repeats: int = DEFAULT_REPEATS,
                      seed: int = 0, layer: Optional[int] = None) -> dict:
    """Train on the corpus without the bench's exchanges; score each arm's states."""
    records = aj.read_jsonl(cb.store_path(chats_dir))
    labels = bench_labels(records, bench=bench)
    if not labels["arms"]:
        return {"bench": bench, "note": "no graded bench replies"}
    bench_keys = set(labels["strata"])
    bench_sessions = {s for s in labels["sessions"].values() if s}
    have = set(corpus_states["keys"])
    out: dict = {"bench": bench, "judge_key": labels["key"], "variants": {}}
    for variant, keep in (("without the bench's exchanges", lambda r: r["key"] not in bench_keys),
                          ("without their sessions", lambda r: r["session"] not in bench_sessions)):
        train = [r for r in rows if r["key"] in have and keep(r)]
        if not train or not (0 < sum(r["y"] for r in train) < len(train)):
            continue
        ty = np.array([r["y"] for r in train])
        tg = [r["session"] for r in train]
        chosen = layer
        if chosen is None:
            # the layer by CV on the training rows alone, so the bench reading is not
            # the maximum of a sweep
            per = _layer_rows(corpus_states, train, layers=corpus_states["layers"], Cs=Cs, folds=folds,
                              repeats=1, seed=seed)
            chosen = max(per, key=lambda l: (per[l]["auc"], -l)) if per else None
        if chosen is None:
            continue
        probe = Probe(layer_matrix(corpus_states, chosen, [r["key"] for r in train]), ty, tg, Cs=Cs, seed=seed)
        arms = {}
        for arm, lab in sorted(labels["arms"].items()):
            st = load_states(arm_states_file(sdir, bench, arm))
            if st["X"] is None or chosen not in st["layers"]:
                arms[arm] = {"note": "no states"}
                continue
            keys = [k for k in st["keys"] if k in lab]
            if not keys:
                arms[arm] = {"note": "no labelled contexts with states"}
                continue
            yy = np.array([lab[k] for k in keys])
            ss = probe.score(layer_matrix(st, chosen, keys))
            cell = _cell(yy, ss, {"basis": [str(labels["strata"].get(k)) for k in keys]}, seed=seed)
            cell["mean_score"] = float(np.mean(ss))
            arms[arm] = cell
        out["variants"][variant] = {"train_n": len(train), "train_yields": int(ty.sum()),
                                    "layer": int(chosen), "C": probe.C, "arms": arms}
    return out


def _fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _ci(c) -> str:
    if not c:
        return ""
    return f"({_fmt(c[0])}–{_fmt(c[1])})"


def _within(c: dict, names=("basis", "month", "basis×month", "depth")) -> str:
    w = c.get("within") or {}
    return "  ".join(f"{n} {_fmt(w[n]['auc'])} {_ci(w[n]['ci'])}" for n in names if n in w)


def render(rep: dict) -> str:
    lines = [f"CAPTURE PROBE: judge {rep.get('judge_key')}, pushback replies graded on "
             + ("the stored reply" if rep.get("source") == "original" else "any target"), ""]
    bb = rep.get("by_basis") or {}
    lines.append(f"{rep.get('n', 0)} replies in {rep.get('sessions', 0)} sessions, {rep.get('yields', 0)} yields; "
                 + ", ".join(f"{b}: {v['yields']} of {v['n']}" for b, v in sorted(bb.items())))
    lines.append("AUC (95% CI); within X = the AUC over pairs that share X (BASIS, month, both, depth in "
                 "the conversation). Grouped CV by session.")
    lines.append("")
    lines.append("baselines")
    for name, c in (rep.get("baselines") or {}).items():
        lines.append(f"  {name:<24} AUC {_fmt(c.get('auc'))} {_ci(c.get('ci')):<13} within {_within(c)}")
    for pos, e in (rep.get("positions") or {}).items():
        lines.append("")
        lines.append(f"probe at {pos}: {e.get('note')}; {e.get('n')} replies, {e.get('yields')} yields")
        perm = e.get("permutation_p") or {}
        best = max(e["layers"], key=lambda l: e["layers"][l]["auc"]) if e.get("layers") else None
        for layer, c in sorted((e.get("layers") or {}).items()):
            mark = ("  [named]" if layer in rep.get("named_layers", []) else "") + ("  [max of sweep]" if layer == best else "")
            p = perm.get(layer)
            lines.append(f"  L{layer:<3} AUC {_fmt(c['auc'])} {_ci(c['ci']):<13} within {_within(c)}"
                         + (f"  perm p {p['p']:.3f} (one repeat at C {p['C']:g}: {_fmt(p['auc'])})" if p else "")
                         + mark)
        ao = e.get("assertion_only") or {}
        for layer, c in sorted(ao.items()):
            lines.append(f"  trained and read on bare assertions only, L{layer}: n {c['n']}, yields {c['yields']}, "
                         f"AUC {_fmt(c['auc'])} {_ci(c['ci'])}  within month {_within(c, ('month',))}")
    tr = rep.get("transfer") or {}
    if tr.get("variants"):
        lines.append("")
        lines.append(f"transfer to bench {tr.get('bench')} (grades {tr.get('judge_key')}): a probe trained on "
                     "the corpus scores each arm's states")
        for variant, v in tr["variants"].items():
            lines.append(f"  trained {variant}: {v['train_n']} replies, {v['train_yields']} yields; "
                         f"layer L{v['layer']} (chosen on the corpus), C {v['C']:g}")
            for arm, c in v["arms"].items():
                if "auc" not in c:
                    lines.append(f"    {arm:<50} {c.get('note')}")
                    continue
                lines.append(f"    {arm:<50} n {c['n']:>3}, yields {c['yields']:>3}: AUC {_fmt(c['auc'])} "
                             f"{_ci(c['ci']):<13} within {_within(c, ('basis',))}")
    elif tr.get("note"):
        lines.append("")
        lines.append(f"transfer: {tr['note']}")
    return "\n".join(lines)


# ── the frozen probe ─────────────────────────────────────────────────────────────

PROBE_DIRNAME = "probe"


def probe_dir(live_chats_dir) -> Path:
    """``server/data/capture/probe/``: the frozen probes, one ``<id>.npz`` each."""
    return cj.store_path(live_chats_dir).parent / PROBE_DIRNAME


def freeze(chats_dir, *, layer: int, judge_key: Optional[str] = None, source: str = "original",
           states_directory=None, out_dir=None, Cs=DEFAULT_CS, seed: int = 0) -> Path:
    """Fit one probe at *layer* on every labelled exchange with a stored step-0 state, and
    write it frozen to ``probe/<id>.npz``. The direction is folded back into the raw state
    (``score = state · w + b``); ``meta`` holds the layer, the judge key, when it was frozen,
    its CV AUC there and the keys it was fitted on. Its scores on an exchange graded after
    ``frozen_at`` and not among those keys are out of sample by construction."""
    chats_dir = Path(chats_dir)
    verdicts = label_records(chats_dir)
    key = judge_key or newest_key(verdicts)
    sdir = Path(states_directory) if states_directory else states_dir(chats_dir)
    st = load_states(states_file(sdir, "corpus", "base"))
    if st["X"] is None or int(layer) not in st["layers"]:
        raise ValueError(f"no stored corpus states at layer {layer}")
    rows = [r for r in label_rows(verdicts, key=key, source=source) if r["key"] in set(st["keys"])]
    y = np.array([r["y"] for r in rows])
    if not len(y) or not 0 < int(y.sum()) < len(y):
        raise ValueError("a probe needs labelled exchanges of both kinds")
    groups = [r["session"] for r in rows]
    X = layer_matrix(st, layer, [r["key"] for r in rows])
    cv = cv_dense(X, y, groups, Cs=Cs, repeats=1, seed=seed)
    probe = Probe(X, y, groups, Cs=Cs, seed=seed)
    w = (probe.v @ np.asarray(probe.model.coef_, dtype=np.float64).ravel()) / probe.sd
    b = float(probe.model.intercept_[0] - probe.mu @ w)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    meta = {"probe_id": f"p{stamp}-L{int(layer)}", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "layer": int(layer), "C": probe.C, "judge_key": key, "source": source, "n": len(rows),
            "yields": int(y.sum()), "cv_auc": cv["auc"], "position": POSITION, "weights": "base",
            "model_id": next(iter(st["meta"].values()), {}).get("model_id", ""),
            "train_keys": [r["key"] for r in rows]}
    out = Path(out_dir) if out_dir else probe_dir(chats_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{meta['probe_id']}.npz"
    np.savez(path, w=w.astype(np.float32), b=np.float64(b), layer=np.int64(layer),
             meta=np.array(json.dumps(meta)))
    return path


def load_probes(directory) -> list:
    """Every frozen probe under *directory*, oldest first: ``{"id", "layer", "w", "b",
    "meta", "train_keys"}``. An unreadable file is named and skipped."""
    out = []
    d = Path(directory)
    for p in (sorted(d.glob("*.npz")) if d.is_dir() else []):
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(z["meta"].item())
                out.append({"id": meta.get("probe_id") or p.stem, "layer": int(z["layer"]),
                            "w": np.asarray(z["w"], dtype=np.float64), "b": float(z["b"]),
                            "meta": meta, "train_keys": set(meta.get("train_keys") or [])})
        except Exception as e:  # noqa: BLE001 — one bad file does not hide the others
            print(f"[capture_probe] probe {p.name} unreadable: {e}")
    out.sort(key=lambda r: str(r["meta"].get("frozen_at") or ""))
    return out


def probe_scores(probes: Sequence[dict], states: np.ndarray, layers: Sequence[int], key: str) -> dict:
    """``{probe_id: score}`` for one context's states ((layers, D)): every probe whose layer
    was read and whose fit did not include *key*."""
    layers = [int(l) for l in layers]
    out = {}
    for p in probes:
        if p["layer"] in layers and key not in p["train_keys"]:
            x = np.asarray(states[layers.index(p["layer"])], dtype=np.float64)
            out[p["id"]] = float(x @ p["w"] + p["b"])
    return out


# ── the shadow reading (a Sleep run's clean-base batch, §4.5) ─────────────────────

SHADOW_FILENAME = "shadow.jsonl"
# Below this cosine between a stored state and the same context read again, the run's
# model, quantization or prompt rebuild is not the one the probes were fitted on: the
# step stores and scores nothing (one forward pass reads an identical setup at 0.999+).
CONSISTENCY_MIN = 0.98

# server.main sets these from server_config.json `capture_probe.*`.
_POLICY = {"shadow": True, "sleep_max": 200, "originals": True}


def set_policy(*, shadow: Optional[bool] = None, sleep_max: Optional[int] = None,
               originals: Optional[bool] = None) -> None:
    if shadow is not None:
        _POLICY["shadow"] = bool(shadow)
    if originals is not None:
        _POLICY["originals"] = bool(originals)
    if sleep_max is not None:
        _POLICY["sleep_max"] = max(0, int(sleep_max))


def shadow_enabled() -> bool:
    """Does a Sleep run's clean-base batch read the state before each newly graded reply?"""
    return bool(_POLICY["shadow"]) and int(_POLICY["sleep_max"]) > 0


def originals_enabled() -> bool:
    """Does a Sleep run's capture gate also grade the stored reply behind each revised target
    (the probe's label, ``cj.run_originals``)?"""
    return bool(_POLICY["originals"]) and int(_POLICY["sleep_max"]) > 0


def policy() -> dict:
    return dict(_POLICY)


def shadow_path(live_chats_dir) -> Path:
    """``server/data/capture/shadow.jsonl``: the shadow step's records, beside the verdicts."""
    return cj.store_path(live_chats_dir).parent / SHADOW_FILENAME


def shadow_todo(verdict_records: Iterable[dict], stored: set, *, key: str) -> list:
    """``[(context key, verdict)]``: each exchange the capture judge found answering pushback
    under *key* with no stored state, one verdict per exchange (the stored reply's
    preferred), newest verdict first."""
    best: dict = {}
    for r in verdict_records:
        if (r.get("record") != "verdict" or r.get("judge_key") != key or r.get("outcome") != "verdict"
                or r.get("pushback") != "yes" or r.get("session") is None):
            continue
        k = context_key(r["session"], r["exchange_index"])
        if k in stored:
            continue
        rank = (r.get("target_source") == "original", str(r.get("ts") or ""))
        if k not in best or rank >= best[k][0]:
            best[k] = (rank, r)
    return [(k, v) for k, (_rank, v) in sorted(best.items(), key=lambda kv: str(kv[1][1].get("ts") or ""),
                                               reverse=True)]


def run_shadow(*, model, tokenizer, model_id: str, chats_dir, prompts_dir, key: str,
               run_id: str = "", limit: int = 0, max_tokens: int = 0, base_quant: str = "",
               say=print, should_stop=None, trim=None, layers: Optional[Sequence[int]] = None,
               state_fn=None) -> Counter:
    """The shadow step: after the capture gate graded this run's targets, inside the clean
    base. For each exchange its judge found answering pushback with no stored state (newest
    first, at most *limit*), the base's state before the reply goes to the corpus states
    store: the live prompt rebuilt, one forward pass. Every frozen probe's score goes beside
    the verdict in ``shadow.jsonl``. First, one ``check`` record: the newest state already
    stored, read again and compared, so a run on a different model, quantization or prompt
    rebuild stores and scores nothing (:data:`CONSISTENCY_MIN`). A measurement only: nothing
    Ava reads, trains on or is gated by sees it (P3). *state_fn(conv) -> (states, prompt,
    ids)* stands in for the model in the self-test."""
    tally: Counter = Counter()
    chats_dir = Path(chats_dir)
    store = states_file(states_dir(chats_dir), "corpus", "base")
    log = shadow_path(chats_dir)
    layers = [int(l) for l in (layers or store_layers(store))]
    n_layers = 0
    if state_fn is None:
        layers_mod = hc.find_decoder_layers(model)
        if layers_mod is None:
            say("capture probe: no decoder stack on the loaded model — nothing read")
            tally["shadow_no_model"] += 1
            return tally
        n_layers = len(layers_mod)
        layers = layers or sweep_layers(n_layers)

        def state_fn(conv):
            prompt, ids = context_prompt(tokenizer, model_id, conv)
            if max_tokens and len(ids) > max_tokens:
                raise TooLong(f"{len(ids)} tokens")
            return context_state(model, ids, layers), prompt, ids
    probes = [p for p in load_probes(probe_dir(chats_dir)) if p["layer"] in layers]
    fallback = hc.persona_empty_system(prompts_dir) if prompts_dir else ""
    docs: dict = {}

    def conversation(session: str, idx: int) -> tuple:
        if session not in docs:
            try:
                docs[session] = json.loads((chats_dir / session).read_text(encoding="utf-8"))
            except Exception:
                docs[session] = None
        doc = docs[session]
        exs = (doc or {}).get("exchanges") or []
        if not isinstance(exs, list) or not 0 <= idx < len(exs) or not isinstance(exs[idx], dict):
            raise LookupError(f"{session} #{idx}: no such exchange")
        return hc.replay_conversation(doc, idx, fallback)

    # The consistency check: the newest stored state, read again.
    stored_states = load_states(store)
    fits = [k for k in stored_states["keys"]
            if not max_tokens or int(stored_states["meta"][k].get("prompt_tokens") or 0) <= max_tokens]
    if fits:
        newest = max(fits, key=lambda k: str(stored_states["meta"][k].get("captured_at") or ""))
        session, _, idx = newest.rpartition("#")
        try:
            conv, _framing = conversation(session, int(idx))
            again, _prompt, ids = state_fn(conv)
            ref = stored_states["X"][stored_states["keys"].index(newest)].astype(np.float64)
            cos = {}
            for i, l in enumerate(stored_states["layers"]):
                if l in layers:
                    a, r = np.asarray(again[layers.index(l)], dtype=np.float64), ref[i]
                    cos[int(l)] = float(a @ r / (np.linalg.norm(a) * np.linalg.norm(r) + 1e-12))
            low = min(cos.values()) if cos else float("nan")
            aj.append_record(log, {"record": "check", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id,
                                   "key": newest, "cos_min": low, "cos": cos, "model_id": model_id,
                                   "base_quant": base_quant, "prompt_tokens": len(ids)})
            say(f"capture probe: consistency with the stored states, cosine ≥ {low:.4f} ({newest})")
            if not low >= CONSISTENCY_MIN:
                say(f"capture probe: below {CONSISTENCY_MIN} — this run's base is not the one the probes "
                    "were fitted on; nothing stored or scored")
                tally["shadow_inconsistent"] += 1
                return tally
        except Exception as e:  # noqa: BLE001 — no check, no reading
            say(f"capture probe: consistency check failed ({type(e).__name__}: {e}); nothing read")
            tally["shadow_check_failed"] += 1
            return tally
    todo = shadow_todo(label_records(chats_dir), set(stored_states["keys"]), key=key)
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        return tally
    say(f"capture probe: {len(todo)} newly graded exchange(s) to read before the reply, "
        f"{len(probes)} frozen probe(s)")
    for n, (k, v) in enumerate(todo, 1):
        if should_stop is not None and should_stop():
            tally["shadow_stopped"] += len(todo) - n + 1
            break
        t0 = time.monotonic()
        try:
            conv, framing = conversation(str(v["session"]), int(v["exchange_index"]))
            states, prompt, ids = state_fn(conv)
        except TooLong:
            tally["shadow_too_long"] += 1
            continue
        except Exception as e:  # noqa: BLE001 — one exchange, not the step
            tally["shadow_failed"] += 1
            say(f"  capture probe {k}: {type(e).__name__}: {e}")
            if trim is not None:
                trim()
            continue
        write_state(store, k, states, state_meta(
            key=k, set_name="corpus", condition="as_is", framing=framing, model_id=model_id,
            adapter_id="", weights="base", layers=layers, n_layers=n_layers or len(layers),
            hidden_dim=int(states.shape[1]), prompt=prompt, n_tokens=len(ids), base_quant=base_quant,
            run_id=run_id, writer="shadow"))
        scores = probe_scores(probes, states, layers, k)
        aj.append_record(log, {"record": "shadow", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": run_id,
                               "key": k, "session": v["session"], "exchange_index": int(v["exchange_index"]),
                               "judge_key": key, "verdict_ts": v.get("ts"), "response": v.get("response"),
                               "basis": v.get("basis"), "opening": v.get("opening"),
                               "target_source": v.get("target_source"), "target_sha": v.get("target_sha"),
                               "scores": scores, "prompt_tokens": len(ids), "model_id": model_id,
                               "base_quant": base_quant, "seconds": round(time.monotonic() - t0, 1)})
        tally["shadow_read"] += 1
        say(f"  capture probe {n}/{len(todo)}: {k} {v.get('response')}"
            + "".join(f", {pid} {sc:+.2f}" for pid, sc in scores.items()))
        if trim is not None and n % 10 == 0:
            trim()
    return tally


# ── the held-out report ──────────────────────────────────────────────────────────

def shadow_report(chats_dir, *, judge_key: Optional[str] = None, source: str = "original",
                  states_directory=None, seed: int = 0) -> dict:
    """Each frozen probe on the exchanges graded after it was frozen and not among those it
    was fitted on: the held-out test of §4.5. Scored from the stored states (so a probe frozen
    later is read on its own held-out set), under the judge key given or the newest."""
    chats_dir = Path(chats_dir)
    verdicts = label_records(chats_dir)
    key = judge_key or newest_key(verdicts)
    sdir = Path(states_directory) if states_directory else states_dir(chats_dir)
    st = load_states(states_file(sdir, "corpus", "base"))
    have = set(st["keys"])
    rows = label_rows(verdicts, key=key, source=source)
    log = aj.read_jsonl(shadow_path(chats_dir))
    out = {"judge_key": key, "source": source, "probes": [],
           "checks": [r for r in log if r.get("record") == "check"][-5:],
           "shadow_records": sum(1 for r in log if r.get("record") == "shadow")}
    for p in load_probes(probe_dir(chats_dir)):
        frozen = str(p["meta"].get("frozen_at") or "")
        # Held out: graded after the freeze, not fitted on, and from a session none of the
        # fitted exchanges came from (an exchange's neighbours share most of its context).
        fitted_sessions = {k.rpartition("#")[0] for k in p["train_keys"]}
        held = [r for r in rows if r["key"] not in p["train_keys"] and r["session"] not in fitted_sessions
                and str(r.get("ts") or "") > frozen]
        ready = [r for r in held if r["key"] in have and p["layer"] in st["layers"]]
        entry = {"id": p["id"], "layer": p["layer"], "frozen_at": frozen,
                 "fitted": {k: p["meta"].get(k) for k in ("n", "yields", "cv_auc", "judge_key", "C")},
                 "held_out": len(held), "read": len(ready), "pending": len(held) - len(ready),
                 "subsets": {}}
        # ...of which live: the sessions begun after the freeze. The rest are older exchanges
        # graded later (a stored reply behind a revised target, a new judge): out of sample,
        # but from the weeks the probe was fitted on.
        frozen_day = _session_day(frozen.replace("-", "").replace("T", "_").replace(":", ""))
        for name, sub in (("all", ready),
                          ("live", [r for r in ready if frozen_day is not None
                                    and (_session_day(r["session"]) or 0) > frozen_day])):
            if not sub:
                continue
            y = np.array([r["y"] for r in sub])
            cell = {"n": len(y), "yields": int(y.sum())}
            if 0 < int(y.sum()) < len(y):
                scores = layer_matrix(st, p["layer"], [r["key"] for r in sub]) @ p["w"] + p["b"]
                cell = _cell(y, scores, {"basis": [str(r.get("basis")) for r in sub]}, seed=seed)
            entry["subsets"][name] = cell
        out["probes"].append(entry)
    return out


def render_shadow(rep: dict) -> str:
    lines = [f"CAPTURE PROBE, held out: each frozen probe on the replies graded after it was frozen "
             f"(judge {rep.get('judge_key')}, " + ("the stored reply" if rep.get("source") == "original"
                                                   else "any target") + ")",
             f"shadow records: {rep.get('shadow_records', 0)}"]
    for c in rep.get("checks") or []:
        lines.append(f"  check {c.get('ts')} ({c.get('run_id') or 'no run'}): cosine ≥ {_fmt(c.get('cos_min'), 4)}")
    if not rep.get("probes"):
        lines.append("no frozen probe (python -m core.capture_probe --freeze LAYERS)")
    for p in rep.get("probes") or []:
        f = p.get("fitted") or {}
        lines.append("")
        lines.append(f"{p['id']}: layer {p['layer']}, frozen {p['frozen_at']}; fitted on {f.get('n')} "
                     f"({f.get('yields')} yields, CV AUC {_fmt(f.get('cv_auc'))}, judge {f.get('judge_key')})")
        lines.append(f"  graded since, from sessions it was not fitted on: {p['held_out']} "
                     f"({p['read']} read, {p['pending']} without a state)")
        for name, label in (("all", "all of them"), ("live", "sessions begun after the freeze")):
            c = (p.get("subsets") or {}).get(name)
            if not c:
                continue
            if "auc" in c:
                lines.append(f"  {label}: AUC {_fmt(c['auc'])} {_ci(c['ci'])}, within {_within(c, ('basis',))}; "
                             f"n {c['n']}, yields {c['yields']}")
            else:
                lines.append(f"  {label}: n {c['n']}, yields {c['yields']}: one class only, no AUC yet")
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ── self-test ────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import tempfile

    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    print("auc")
    check("perfect", auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
    check("inverted", auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
    check("ties count a half", auc([0, 1], [0.5, 0.5]), 0.5)
    check("one class only", math.isnan(auc([1, 1], [0.1, 0.2])), True)
    check("within strata: perfect inside each, inverted across",
          (auc([0, 1, 0, 1], [0.0, 0.1, 0.8, 0.9]), auc_within([0, 1, 0, 1], [0.0, 0.1, 0.8, 0.9], ["a", "a", "b", "b"])),
          (0.75, 1.0))
    lo, hi = bootstrap_ci([0] * 20 + [1] * 20, list(range(40)))
    check("the CI of a perfect ranking", (lo, hi), (1.0, 1.0))

    print("the layer sweep")
    check("60 layers", sweep_layers(60), [0, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 30, 32, 36, 38, 40, 44,
                                          47, 48, 52, 56, 59])
    check("no layers", sweep_layers(0), [])
    check("the axes' own layers added", [l for l in sweep_layers(60, extra=[22, 25, 34, 99]) if l in (22, 25, 34, 99)],
          [22, 25, 34])

    print("the folds keep a session on one side")
    y = np.array([0, 1] * 20)
    groups = [f"s{i // 2}" for i in range(40)]
    ok = True
    for tr, te in _splits(y, groups, 5, 0):
        ok &= not ({groups[i] for i in tr} & {groups[i] for i in te})
    check("grouped", ok, True)

    print("the probe on planted data")
    rng = np.random.default_rng(1)
    n, D = 120, 400
    y = (rng.random(n) < 0.3).astype(int)
    direction = rng.normal(size=D)
    X_signal = rng.normal(size=(n, D)) + np.outer(y - 0.3, direction) * 0.25
    X_noise = rng.normal(size=(n, D))
    groups = [f"s{i}" for i in range(n)]
    sig = cv_dense(X_signal, y, groups, Cs=(1e-3, 1e-1), repeats=2)
    noise = cv_dense(X_noise, y, groups, Cs=(1e-3, 1e-1), repeats=2)
    check("a planted direction is found", sig["auc"] > 0.85, True)
    check("noise is near a half", 0.3 < noise["auc"] < 0.7, True)
    p = permutation_p(X_signal, y, groups, C=1e-3, n=19)
    check("…and its permutation p is the floor", (p["p"], p["auc"] > 0.85), (0.05, True))
    check("…and noise's is not", permutation_p(X_noise, y, groups, C=1e-3, n=19)["p"] > 0.05, True)
    probe = Probe(X_signal[:80], y[:80], groups[:80], Cs=(1e-3,))
    check("a probe fitted on part ranks the rest", auc(y[80:], probe.score(X_signal[80:])) > 0.8, True)
    texts = ["you are right, I give in" if v else "no, here is why" for v in y]
    check("a text baseline reads a text", cv_text(texts, y, groups, repeats=1)["auc"] > 0.95, True)
    Z, V = _span(rng.normal(size=(10, 50)))
    check("the span keeps the rows' rank", (Z.shape, V.shape), ((10, 10), (50, 10)))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("the states store")
        path = states_file(td / "states", "corpus", "base")
        check("file names", (path.name, states_file(td, "b1", "adapter-1", "persona_empty").name),
              ("corpus~base.npz", "b1~adapter-1+persona_empty.npz"))
        st = rng.normal(size=(3, 8)).astype(np.float32)
        check("write", write_state(path, "s1.json#1", st, {"layers": [0, 2, 4]}), True)
        check("a key once only", write_state(path, "s1.json#1", st, {"layers": [0, 2, 4]}), False)
        check("no slash in a key", write_state(path, "a/b", st, {"layers": [0, 2, 4]}), False)
        write_state(path, "s2.json#3", st * 2, {"layers": [0, 2, 4]})
        write_state(path, "s3.json#1", st[:2], {"layers": [0, 2]})
        check("stored keys", stored_keys(path), {"s1.json#1", "s2.json#3", "s3.json#1"})
        back = load_states(path)
        check("read back: the layers every key carries, no key left out",
              (back["keys"], back["layers"], back["X"].shape, back["mismatched"]),
              (["s1.json#1", "s2.json#3", "s3.json#1"], [0, 2], (3, 2, 8), 2))
        check("…the rows are the right layers", np.allclose(back["X"][2], st[:2].astype(np.float16), atol=1e-2), True)
        wide = load_states(path, layers=[0, 2, 4])
        check("asked layers: only keys carrying them", (wide["keys"], wide["layers"], wide["missing"]),
              (["s1.json#1", "s2.json#3"], [0, 2, 4], 1))
        check("a layer's matrix in the asked order",
              np.allclose(layer_matrix(back, 2, ["s2.json#3", "s1.json#1"]),
                          np.stack([st[1] * 2, st[1]]).astype(np.float16), atol=1e-2), True)
        check("a missing file is empty", load_states(td / "none.npz")["X"], None)
        check("arm → file", (arm_states_file(td, "b1", "base").name,
                             arm_states_file(td, "b1", "adapter:adapter-7|persona_empty").name),
              ("b1~base.npz", "b1~adapter-7+persona_empty.npz"))

        print("labels")
        recs = [
            {"record": "verdict", "judge_key": "k", "outcome": "verdict", "pushback": "yes", "response": "yields",
             "basis": "assertion", "session": "20260901_000000.json", "exchange_index": 1, "target_source": "original",
             "ts": "2026-09-29T10:00:00", "adapter_id": "/m/adapter-1"},
            {"record": "verdict", "judge_key": "k", "outcome": "verdict", "pushback": "yes", "response": "holds",
             "basis": "reason", "session": "20260901_000000.json", "exchange_index": 3, "target_source": "revised",
             "ts": "2026-09-29T10:00:00"},
            {"record": "verdict", "judge_key": "k", "outcome": "verdict", "pushback": "no", "response": "n/a",
             "session": "20260902_000000.json", "exchange_index": 1, "target_source": "original",
             "ts": "2026-09-29T10:00:00"},
            {"record": "verdict", "judge_key": "old", "outcome": "verdict", "pushback": "yes", "response": "holds",
             "session": "20260903_000000.json", "exchange_index": 2, "target_source": "original",
             "ts": "2026-09-01T10:00:00"},
        ]
        check("the newest key", newest_key(recs), "k")
        check("pushback under any key", pushback_exchanges(recs),
              [("20260901_000000.json", 1), ("20260901_000000.json", 3), ("20260903_000000.json", 2)])
        rows = label_rows(recs, key="k")
        check("on the stored reply only", [(r["key"], r["y"], r["adapter"]) for r in rows],
              [("20260901_000000.json#1", 1, "adapter-1")])
        check("any target", [r["key"] for r in label_rows(recs, key="k", source="any")],
              ["20260901_000000.json#1", "20260901_000000.json#3"])
        chats = td / "chats"
        chats.mkdir()
        (chats / "20260901_000000.json").write_text(json.dumps({"exchanges": [
            {"user_prompt": "hi", "assistant_response": "hello"},
            {"exchange_id": "e1", "user_prompt": "You never meant it.", "system_content": "You are Ava.",
             "assistant_response": "<think>hm</think>\n\nYou're right."}]}), encoding="utf-8")
        got = attach_texts(rows, chats)
        check("texts attached", (got[0]["message"], got[0]["reply"], got[0]["system"], got[0]["exchange_id"],
                                 got[0]["month"], got[0]["previous"]),
              ("You never meant it.", "You're right.", "You are Ava.", "e1", "202609", "hello"))
        check("a row without a transcript is left out",
              attach_texts([dict(rows[0], session="missing.json")], chats), [])
        cap = {"positions": [0, 4], "residuals": {2: np.ones((2, 8), np.float16), 5: np.zeros((2, 8), np.float16)},
               "axes": {}, "n_tokens": 5, "layers": [2, 5], "hidden_dim": 8}
        hc.write_sidecar(chats / "20260901_000000.json", "e1", cap, model_id="m", adapter_id="", variant="base")
        bf = backfill_positions(got, chats)
        check("the backfill's first and last rows", (bf["first"]["keys"], bf["first"]["layers"], bf["final"]["X"].shape),
              (["20260901_000000.json#1"], [2, 5], (1, 2, 8)))
        hc.write_sidecar(chats / "20260901_000000.json", "e1", cap, model_id="m", adapter_id="", variant="base2",
                         extra_meta={"alignment": "live"})
        check("the two alignments are never mixed",
              (backfill_positions(got, chats, variant="base2")["first"]["skipped"],
               backfill_positions(got, chats, variant="base2", alignment="live")["first"]["keys"]),
              (1, ["20260901_000000.json#1"]))

        print("bench labels")
        brecs = [{"record": "context", "bench": "b1", "context_id": "s.json#1", "session": "s.json",
                  "stratum": "assertion", "messages": []},
                 {"record": "grade", "bench": "b1", "context_id": "s.json#1", "weights": "base", "condition": "as_is",
                  "sample": 1, "judge_key": "g", "outcome": "verdict", "response": "yields", "ts": "1"},
                 {"record": "grade", "bench": "b1", "context_id": "s.json#1", "weights": "adapter:adapter-1",
                  "condition": "persona_empty", "sample": 1, "judge_key": "g", "outcome": "verdict",
                  "response": "n/a", "ts": "1"}]
        bl = bench_labels(brecs, bench="b1")
        check("per arm, n/a left out", (bl["key"], bl["arms"], bl["strata"]),
              ("g", {"base": {"s.json#1": 1}}, {"s.json#1": "assertion"}))

    print("the frozen probe and the shadow step")
    import hashlib as _hl
    with tempfile.TemporaryDirectory() as td:
        chats = Path(td) / "data" / "chats"
        chats.mkdir(parents=True)
        D2, LAYERS2 = 64, [3, 5]
        plant = np.random.default_rng(7).normal(size=D2)

        def fake_state(conv):
            text = conv[-1]["content"]
            r = np.random.default_rng(int(_hl.sha1(text.encode()).hexdigest()[:8], 16))
            st_ = r.normal(size=(len(LAYERS2), D2))
            if "yield" in text:
                st_[1] += plant * 0.6
            return st_.astype(np.float32), "prompt: " + text, list(range(10 + len(text)))

        def put_chat(name, message):
            (chats / name).write_text(json.dumps({"exchanges": [
                {"user_prompt": "hi", "assistant_response": "hello"},
                {"user_prompt": message, "system_content": "You are Ava.", "assistant_response": "…"}]}),
                encoding="utf-8")

        store = states_file(states_dir(chats), "corpus", "base")
        verdicts_path = cj.store_path(chats)
        for i in range(40):
            name, message = f"202609{i + 1:02d}_000000.json", f"message {i} " + ("yield" if i % 3 == 0 else "hold")
            put_chat(name, message)
            states, prompt, ids = fake_state([{"role": "user", "content": message}])
            write_state(store, f"{name}#1", states, state_meta(
                key=f"{name}#1", set_name="corpus", condition="as_is", framing="replay", model_id="m",
                adapter_id="", weights="base", layers=LAYERS2, n_layers=8, hidden_dim=D2, prompt=prompt,
                n_tokens=len(ids), base_quant="4bit"))
            aj.append_record(verdicts_path, {"record": "verdict", "judge_key": "k1", "outcome": "verdict",
                                             "pushback": "yes", "target_source": "original",
                                             "response": "yields" if "yield" in message else "holds",
                                             "basis": "assertion", "session": name, "exchange_index": 1,
                                             "ts": "2026-09-30T00:00:00"})
        check("the store's layers", store_layers(store), LAYERS2)
        path = freeze(chats, layer=5, Cs=(1e-2,))
        probes = load_probes(path.parent)
        check("frozen: one probe, its layer, fitted on all 40", (len(probes), probes[0]["layer"],
                                                                  probes[0]["meta"]["n"]), (1, 5, 40))
        st2 = load_states(store)
        keys2 = [f"202609{i + 1:02d}_000000.json#1" for i in range(40)]
        rows2 = label_rows(aj.read_jsonl(verdicts_path), key="k1")
        X2 = layer_matrix(st2, 5, keys2)
        direct = Probe(X2, np.array([r["y"] for r in rows2]), [r["session"] for r in rows2], Cs=(1e-2,))
        check("the folded direction scores as the fitted probe does",
              bool(np.allclose(X2 @ probes[0]["w"] + probes[0]["b"], direct.score(X2), atol=1e-3)), True)
        check("a key it was fitted on gets no score", probe_scores(probes, st2["X"][0], LAYERS2, keys2[0]), {})

        later = [("20991001_000000.json", "late a yield"), ("20991002_000000.json", "late b hold"),
                 ("20991003_000000.json", "late c yield")]
        for j, (name, message) in enumerate(later):
            put_chat(name, message)
            aj.append_record(verdicts_path, {"record": "verdict", "judge_key": "k1", "outcome": "verdict",
                                             "pushback": "yes", "target_source": "original",
                                             "response": "yields" if "yield" in message else "holds",
                                             "basis": "assertion", "session": name, "exchange_index": 1,
                                             "ts": f"2099-01-0{j + 1}T00:00:00"})
        aj.append_record(verdicts_path, {"record": "verdict", "judge_key": "k1", "outcome": "verdict",
                                         "pushback": "no", "response": "n/a", "session": "20991004_000000.json",
                                         "exchange_index": 1, "ts": "2099-01-05T00:00:00"})
        todo = shadow_todo(aj.read_jsonl(verdicts_path), stored_keys(store), key="k1")
        check("to read: the three graded since, pushback only, newest first",
              [k for k, _v in todo], [f"{n}#1" for n, _m in reversed(later)])
        quiet = lambda m: None
        t1 = run_shadow(model=None, tokenizer=None, model_id="m", chats_dir=chats, prompts_dir=None, key="k1",
                        run_id="r1", limit=2, layers=LAYERS2, state_fn=fake_state, say=quiet)
        recs = aj.read_jsonl(shadow_path(chats))
        check("first run: the check, then two read and scored",
              (dict(t1), [r["record"] for r in recs], round(recs[0]["cos_min"], 6),
               all(probes[0]["id"] in r["scores"] for r in recs[1:])),
              ({"shadow_read": 2}, ["check", "shadow", "shadow"], 1.0, True))
        t2 = run_shadow(model=None, tokenizer=None, model_id="m", chats_dir=chats, prompts_dir=None, key="k1",
                        run_id="r2", layers=LAYERS2, state_fn=fake_state, say=quiet)
        t3 = run_shadow(model=None, tokenizer=None, model_id="m", chats_dir=chats, prompts_dir=None, key="k1",
                        run_id="r3", layers=LAYERS2, state_fn=fake_state, say=quiet)
        check("the next run reads the rest, then nothing waits", (dict(t2), dict(t3), len(stored_keys(store))),
              ({"shadow_read": 1}, {}, 43))
        noise = lambda conv: (np.random.default_rng(0).normal(size=(2, D2)).astype(np.float32), "p", [1, 2])
        put_chat("20991005_000000.json", "late d yield")
        aj.append_record(verdicts_path, {"record": "verdict", "judge_key": "k1", "outcome": "verdict",
                                         "pushback": "yes", "target_source": "original", "response": "yields",
                                         "basis": "assertion", "session": "20991005_000000.json",
                                         "exchange_index": 1, "ts": "2099-01-06T00:00:00"})
        t4 = run_shadow(model=None, tokenizer=None, model_id="m", chats_dir=chats, prompts_dir=None, key="k1",
                        run_id="r4", layers=LAYERS2, state_fn=noise, say=quiet)
        check("another base: the check fails, nothing stored", (dict(t4), len(stored_keys(store))),
              ({"shadow_inconsistent": 1}, 43))

        def too_long(conv):
            if "late d" in conv[-1]["content"]:
                raise TooLong("over the context")
            return fake_state(conv)
        t5 = run_shadow(model=None, tokenizer=None, model_id="m", chats_dir=chats, prompts_dir=None, key="k1",
                        run_id="r5", layers=LAYERS2, state_fn=too_long, say=quiet)
        check("a prompt over the context is skipped, not cut", dict(t5), {"shadow_too_long": 1})
        # The stored reply behind a revised target, graded later (cj.originals_path): one from a
        # session the probe was fitted on (not held out), one from an older session it never saw.
        originals = cj.originals_path(chats)
        put_chat("20260920_120000.json", "older e yield")
        states, prompt, ids = fake_state([{"role": "user", "content": "older e yield"}])
        write_state(store, "20260920_120000.json#1", states, state_meta(
            key="20260920_120000.json#1", set_name="corpus", condition="as_is", framing="replay", model_id="m",
            adapter_id="", weights="base", layers=LAYERS2, n_layers=8, hidden_dim=D2, prompt=prompt,
            n_tokens=len(ids), base_quant="4bit"))
        for name in ("20260901_000000.json", "20260920_120000.json"):
            aj.append_record(originals, {"record": "verdict", "judge_key": "k1", "outcome": "verdict",
                                         "pushback": "yes", "target_source": "original", "graded": "stored_reply",
                                         "response": "yields", "basis": "assertion", "session": name,
                                         "exchange_index": 2 if name.startswith("20260901") else 1,
                                         "ts": "2099-01-07T00:00:00"})
        check("the stored replies' grades are labels beside the verdicts",
              len(label_rows(label_records(chats), key="k1")), 40 + 4 + 2)
        rep = shadow_report(chats, judge_key="k1")
        p0 = rep["probes"][0]
        a0, l0 = p0["subsets"]["all"], p0["subsets"]["live"]
        check("held out: graded after the freeze from sessions it was not fitted on; the live ones "
              "are the sessions begun after it",
              (p0["held_out"], p0["read"], p0["pending"], a0["n"], a0["yields"], l0["n"], l0["yields"]),
              (5, 4, 1, 4, 3, 3, 2))
        text = render_shadow(rep)
        check("…and it renders", ("held out" in text, "cosine ≥" in text, p0["id"] in text), (True, True, True))
        set_policy(shadow=True, sleep_max=0)
        check("a budget of 0 turns it off", shadow_enabled(), False)
        set_policy(sleep_max=200)

    print("a cell and the report")
    yy = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ss = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    rr = [{"basis": b, "month": m, "exchange_index": 1} for b, m in (("assertion", "202608"),) * 4 + (("reason", "202609"),) * 4]
    cell = _cell(yy, ss, _strata(rr))
    check("within each stratification", (cell["auc"], cell["within"]["basis"]["auc"],
                                         cell["within"]["month"]["auc"], cell["within"]["basis×month"]["auc"]),
          (0.75, 0.5, 0.5, 0.5))
    rep = {"judge_key": "k", "source": "original", "n": 8, "yields": 4, "sessions": 8,
           "by_basis": {"assertion": {"n": 4, "yields": 2}}, "baselines": {"basis (bare assertion)": cell},
           "named_layers": [5], "positions": {"step0": {"note": POSITION, "n": 8, "yields": 4,
                                                        "layers": {5: dict(cell, C={1e-3: 1})},
                                                        "permutation_p": {5: {"p": 0.04, "auc": 0.7, "C": 1e-3}}}},
           "transfer": {"bench": "b1", "judge_key": "g", "variants": {"x": {
               "train_n": 8, "train_yields": 4, "layer": 5, "C": 1e-3,
               "arms": {"base": dict(cell), "base|persona_empty": {"note": "no states"}}}}}}
    text = render(rep)
    check("it renders", ("probe at step0" in text, "perm p 0.040" in text, "[named]" in text,
                         "no states" in text, "within basis" in text), (True, True, True, True, True))
    check("the JSON form is plain", json.loads(json.dumps(_jsonable({"a": np.float64("nan"), "b": np.arange(2)}))),
          {"a": None, "b": [0, 1]})

    print("\ncapture_probe self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="The capture probe over the stored states (read-only).")
    ap.add_argument("--chats", default=None, help="the live chats dir (default: the box's)")
    ap.add_argument("--states", default=None, help="the states dir (default: data/capture/states)")
    ap.add_argument("--judge-key", default=None, help="the capture judge's key (default: the newest)")
    ap.add_argument("--source", choices=("original", "any"), default="original")
    ap.add_argument("--bench", default=cb.DEFAULT_BENCH)
    ap.add_argument("--layers", default="", help="comma-separated layers (default: every stored layer)")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                    help="shuffled-label refits for each named layer (0 = none)")
    ap.add_argument("--no-backfill", action="store_true", help="skip the backfill's positions")
    ap.add_argument("--no-transfer", action="store_true", help="skip the bench")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", default="", help="fit and freeze a probe at each of these layers "
                    "(comma-separated) into data/capture/probe/; writes nothing else")
    ap.add_argument("--shadow", action="store_true",
                    help="the frozen probes on the replies graded since they were frozen")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.chats:
        chats = Path(args.chats)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from training.reflections_path import hot_chats_dir
        chats = Path(hot_chats_dir())
    if args.freeze:
        for layer in [int(x) for x in args.freeze.split(",") if x.strip()]:
            path = freeze(chats, layer=layer, judge_key=args.judge_key, source=args.source,
                          states_directory=args.states, seed=args.seed)
            meta = load_probes(path.parent)
            meta = next(p["meta"] for p in meta if p["id"] == path.stem)
            print(f"frozen {path.name}: layer {layer}, fitted on {meta['n']} ({meta['yields']} yields), "
                  f"C {meta['C']:g}, CV AUC {_fmt(meta['cv_auc'])}, judge {meta['judge_key']}")
        return 0
    if args.shadow:
        rep = shadow_report(chats, judge_key=args.judge_key, source=args.source, states_directory=args.states,
                            seed=args.seed)
        print(json.dumps(_jsonable(rep), ensure_ascii=False, indent=1) if args.json else render_shadow(rep))
        return 0
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
    rep = build(chats, judge_key=args.judge_key, source=args.source, states_directory=args.states,
                bench=args.bench, layers=layers, repeats=args.repeats, permutations=args.permutations,
                seed=args.seed, with_backfill=not args.no_backfill, with_transfer=not args.no_transfer,
                progress=lambda m: print(f"[capture_probe] {m}", file=sys.stderr, flush=True))
    print(json.dumps(_jsonable(rep), ensure_ascii=False, indent=1) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
