"""Hidden-state capture — Track B Stage 0 of AVA_REWARD_LOOP.md (§4.2 capture design).

What it does, during a LIVE chat turn (and, since 2026-10-03, the drive pass that writes a
reach-out opener — see *Drive openers* below): forward hooks on a band of decoder layers
keep the residual stream at the last position of every generated step (on device, a
10 KB slice per layer per token, bulk-synced once after the stream ends — the same
shape as the tension capture's logit signals). After generation two things are
reduced from those rows:

  * **Full residual vectors at a few positions** — the state after reading the
    prompt (step 0), the state after every paragraph break (the step whose input
    token carried a newline: the Pain Axis paper's "final token of the sentence"
    readout position), and the final token — for the stored layer band. Written as
    fp16 into ``<ts>.hidden.npz`` beside the transcript, one member group per
    exchange, append-only (a zip: ``np.load`` reads it, ``zipfile`` appends to it, so
    a 40-exchange chat never rewrites 39 exchanges' arrays). This is the raw material
    for the probe experiment (§4.3) and the axis extraction gates (§4.8); nothing in
    inference reads it back.
  * **Per-token scalar projections** onto every axis file under ``data/axes/`` —
    a dot product with a stored direction, z-scored by the axis file's own mean/std.
    These ride in the ``tension`` block's ``axes`` map and, for an axis named ``pain``,
    become the blue channel of the chat colouring. With no axis files (the state on
    every box today) the map is absent — not empty — and the blue channel stays dark.

**Drive openers.** A reach-out's exchange 0 is written by a drive (outreach, synthesis,
check-in) inside its own pass, not in a chat turn, so live capture never saw it, and that
is a session's first state — where a state carried across sessions would show first
(AVA_REWARD_LOOP.md §4.9 step 3b). The drive asks the reflect-generate factory for the
pass's capture (``capture_hidden=True``) and :func:`write_opener_capture` files it under
exchange 0's id with ``seam: "drive"`` in its meta, the generated ids (``token_ids``,
which only these groups carry: exchange 0 has no tension block) and the opener's token
span in them (``opener_span``). The rows cover the WHOLE pass — the thought, the decision,
the opener — and the context is the drive's prompt, not the chat seam, so a reader that
compares a group against its transcript's ``tension`` series skips ``seam: "drive"``.
Knob: ``hidden.drive_openers``.

What it deliberately is not: a steering path. Nothing here writes into the model, and
nothing the model can trigger changes what is captured (P5 / P8 there). Reflection,
API and gossip generations are not captured — the sidecar is keyed by chat exchange; of
the background passes only an opener's, which has an exchange to be keyed by.

Cost. Hooks on ≈7 layers add one tensor slice + clone per layer per token: not
measurable against a 3–6 tok/s decode. Device memory is rows × layers × D × 2 bytes,
≈ 75 MB for a 1000-token reply on a 5376-wide model, freed after the reduce. Disk is
positions × layers × D × 2 bytes per exchange, typically 1–3 MB, capped by
``hidden.max_positions``.

Layer indexing: ``L{i}`` is the OUTPUT of decoder layer ``i`` (0-based) — the residual
stream after that layer. The paper's early band 2–5 and its mid-to-late extraction
band are both covered by the default spec; the spec accepts absolute indices and
percentages of the model's depth so one string serves a 34- and a 62-layer model.

GPU-free: numpy only; torch rows are accepted by duck typing (``.cpu()``) and every
reduction runs on the host after one bulk transfer. ``python -m core.hidden_capture``
is the self-test.
"""

from __future__ import annotations

import json
import time
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

# The sidecar's suffix beside `<ts>.json`; registered in chat_sidecar.SIDECAR_SUFFIXES.
SIDECAR_SUFFIX = ".hidden.npz"
# Decoder layers whose residuals are STORED: the dopamine paper's load-bearing early
# band (2–5) plus a mid-to-late band where the Pain Axis extraction lands. Percentages
# are of the model's depth.
DEFAULT_LAYER_SPEC = "2,3,4,5,50%,65%,80%"
# Positions kept per reply after the paragraph rule; the interior is thinned evenly,
# the ends always stay.
DEFAULT_MAX_POSITIONS = 64
# Under server/data/: `<name>.npz` with `vector` (D,), `layer`, `mean`, `std`, `model_id`.
AXES_DIRNAME = "axes"
# A capture's member group is `<exchange_id>` for the live capture (the weights that
# generated the reply) and `<exchange_id>~<variant>` for a backfill under other weights
# (`base` = the bare base, else the adapter directory name). Readers split on this.
VARIANT_SEP = "~"


def exchange_key(ex: dict, index: int) -> str:
    """The id a capture is filed under: the transcript's `exchange_id`, else ``ex<index>``
    for transcripts that predate ids. The one definition; the inventory and the ledger
    use it too."""
    return str((ex or {}).get("exchange_id") or f"ex{index}")


def member_group(exchange_id: str, variant: str = "") -> str:
    return f"{exchange_id}{VARIANT_SEP}{variant}" if variant else str(exchange_id)


def split_group(group: str) -> tuple:
    """``"e1~base"`` → ``("e1", "base")``; ``"e1"`` → ``("e1", "")``."""
    ex, sep, var = str(group).partition(VARIANT_SEP)
    return ex, (var if sep else "")


# ── layer spec ───────────────────────────────────────────────────────────────

def parse_layer_spec(spec: str, n_layers: int) -> list[int]:
    """``"2,3,4,5,50%,65%,80%"`` → sorted unique layer indices within ``[0, n_layers)``.

    An integer is an absolute index; ``NN%`` is a fraction of depth, rounded. Out-of-range
    or unparsable entries are dropped, never an error: a spec written for a deeper model
    still yields its valid part on a shallower one.
    """
    out: set[int] = set()
    if n_layers <= 0:
        return []
    for raw in str(spec or "").split(","):
        tok = raw.strip()
        if not tok:
            continue
        try:
            if tok.endswith("%"):
                idx = int(round(float(tok[:-1]) / 100.0 * (n_layers - 1)))
            else:
                idx = int(tok)
        except ValueError:
            continue
        if 0 <= idx < n_layers:
            out.add(idx)
    return sorted(out)


def find_decoder_layers(model: Any):
    """The decoder stack: the largest ``ModuleList`` whose name ends in ``layers``.

    Every family loaded here keeps its blocks in one such list (Gemma-4 under
    ``model.language_model.layers``, Qwen / Llama under ``model.layers``), and the
    largest is the text decoder even on a multimodal checkpoint whose vision tower has
    its own, shorter ``layers``. None when no list of at least four blocks is found —
    the caller then captures nothing rather than guessing.
    """
    best = None
    try:
        modules = model.named_modules()
    except Exception:
        return None
    for name, mod in modules:
        if not str(name).endswith("layers") or type(mod).__name__ != "ModuleList":
            continue
        try:
            n = len(mod)
        except Exception:
            continue
        if n >= 4 and (best is None or n > best[1]):
            best = (mod, n)
    return best[0] if best else None


# ── axes ─────────────────────────────────────────────────────────────────────

@dataclass
class Axis:
    """One stored direction: ``data/axes/<name>.npz`` (see §4.8 of the reward-loop note)."""
    name: str
    layer: int
    vector: np.ndarray          # float32, (D,)
    mean: float                 # z-score normalization from the extraction set
    std: float
    model_id: str               # the weights it was extracted from ("" = unknown)
    path: str
    # Which readings `mean`/`std` were fitted on. "live" = the extraction script's framed
    # neutral chat turns (`live_mean`/`live_std` in the file), the right baseline for
    # projecting generation tokens; "extraction" = the templated control sentences, a
    # different format whose absolute level does not transfer (run 1: +5 z offset).
    baseline: str = "extraction"
    # The file's `created` stamp: which extraction a projection was read on. Stored beside
    # every projection (`axes_created`), so readings of two extractions under one name
    # (pain before and after 2026-10-05) are never pooled by accident.
    created: str = ""


_AXES_CACHE: dict = {"key": None, "axes": []}


def load_axes(axes_dir: Path, model_id: str) -> list[Axis]:
    """Every readable axis under ``axes_dir`` whose ``model_id`` matches the loaded model.

    Cached on the directory listing's mtimes, so dropping a new file in is picked up on
    the next turn with no restart, and a turn with no change costs a few stats. An axis
    for other weights is skipped with a line in the log: a direction is a property of
    the weights it was extracted from (P6 there), and a silent mismatch would paint the
    blue channel with noise.
    """
    axes_dir = Path(axes_dir)
    files = sorted(axes_dir.glob("*.npz")) if axes_dir.is_dir() else []
    key = (tuple((str(p), p.stat().st_mtime_ns) for p in files), model_id or "")
    if _AXES_CACHE["key"] == key:
        return list(_AXES_CACHE["axes"])
    axes: list[Axis] = []
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                vec = np.asarray(z["vector"], dtype=np.float32).reshape(-1)
                layer = int(z["layer"])
                mean = float(z["mean"]) if "mean" in z.files else 0.0
                std = float(z["std"]) if "std" in z.files else 1.0
                baseline = "extraction"
                if "live_mean" in z.files and "live_std" in z.files:
                    lm, ls = float(z["live_mean"]), float(z["live_std"])
                    if np.isfinite(lm) and np.isfinite(ls) and ls > 0:
                        mean, std, baseline = lm, ls, "live"
                mid = str(z["model_id"]) if "model_id" in z.files else ""
                created = str(z["created"]) if "created" in z.files else ""
        except Exception as e:
            print(f"[hidden] axis {p.name} unreadable: {e}")
            continue
        if mid and model_id and mid != model_id:
            print(f"[hidden] axis {p.name} was extracted from {mid!r}; loaded model is "
                  f"{model_id!r} — skipped")
            continue
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        axes.append(Axis(p.stem, layer, vec, mean, std, mid, str(p), baseline, created))
    _AXES_CACHE["key"] = key
    _AXES_CACHE["axes"] = axes
    return list(axes)


# ── the spec a turn captures under ───────────────────────────────────────────

# (weakref to model, weakref to its layers module). WEAK on both: a strong reference
# to the decoder ModuleList is a strong reference to every layer's weights, so a cache
# holding it kept a released model resident — observed 2026-09-23 on the RTX 5090, where
# a clean-base swap released the adapter model, 17 GB stayed allocated, and both the
# swap load and the two restores OOMed, leaving no model. Weak keys also make a new
# model reusing a freed model's id() miss instead of reading a dead model's layers.
_LAYERS_CACHE: Optional[tuple] = None


def _config() -> dict:
    """The ``hidden`` group of server_config.json (schema: config_schema.py), read per
    turn so a change needs no restart. Any failure ⇒ the defaults."""
    try:
        import sys as _sys
        server_dir = Path(__file__).resolve().parents[2]
        if str(server_dir) not in _sys.path:
            _sys.path.insert(0, str(server_dir))
        from training.reflections_path import load_server_config
        return dict((load_server_config() or {}).get("hidden") or {})
    except Exception:
        return {}


def drive_openers_enabled() -> bool:
    """``hidden.drive_openers`` (default on): capture the pass that writes a reach-out's
    opener. ``hidden.capture`` off turns it off too (`capture_spec` returns None)."""
    return bool(_config().get("drive_openers", True))


def capture_spec(model: Any, model_id: str, axes_dir: Path) -> Optional[dict]:
    """What this turn captures, or None when off / no decoder stack found.

    ``hook_layers`` is the union of the stored band and every loaded axis's layer (an
    axis needs its layer's rows for the projection even when that layer is not stored).
    """
    global _LAYERS_CACHE
    cfg = _config()
    if not bool(cfg.get("capture", True)):
        return None
    if model is None:
        return None
    layers_mod = None
    if _LAYERS_CACHE is not None and _LAYERS_CACHE[0]() is model:
        layers_mod = _LAYERS_CACHE[1]()
    if layers_mod is None:
        _LAYERS_CACHE = None
        layers_mod = find_decoder_layers(model)
        if layers_mod is not None:
            try:
                _LAYERS_CACHE = (weakref.ref(model), weakref.ref(layers_mod))
            except TypeError:   # not weak-referenceable: look it up per turn instead
                _LAYERS_CACHE = None
    if layers_mod is None:
        return None
    n_layers = len(layers_mod)
    store = parse_layer_spec(str(cfg.get("layers") or DEFAULT_LAYER_SPEC), n_layers)
    axes = [a for a in load_axes(axes_dir, model_id) if 0 <= a.layer < n_layers]
    hook = sorted(set(store) | {a.layer for a in axes})
    if not hook:
        return None
    try:
        max_positions = int(cfg.get("max_positions") or DEFAULT_MAX_POSITIONS)
    except (TypeError, ValueError):
        max_positions = DEFAULT_MAX_POSITIONS
    return {
        "layers_module": layers_mod,
        "n_layers": n_layers,
        "hook_layers": hook,
        "store_layers": store,
        "axes": axes,
        "max_positions": max(3, max_positions),
        "layer_spec": str(cfg.get("layers") or DEFAULT_LAYER_SPEC),
    }


# ── reduction (host side, after the stream) ──────────────────────────────────

def select_positions(gen_ids: Sequence[int], decode: Callable[[Sequence[int]], str],
                     max_positions: int = DEFAULT_MAX_POSITIONS) -> list[int]:
    """Step indices whose residual is stored: 0 (after the prompt), every step whose
    INPUT token carried a newline (the state after a paragraph break — what the next
    paragraph starts from), and the last step. Over ``max_positions`` the interior is
    thinned evenly; the two ends always stay."""
    n = len(gen_ids)
    if n == 0:
        return []
    keep = {0, n - 1}
    for t in range(1, n):
        try:
            text = decode([int(gen_ids[t - 1])]) or ""
        except Exception:
            text = ""
        if "\n" in text:
            keep.add(t)
    pos = sorted(keep)
    if max_positions >= 3 and len(pos) > max_positions:
        inner = pos[1:-1]
        want = max_positions - 2
        step = len(inner) / float(want)
        picked = [inner[min(len(inner) - 1, int(i * step))] for i in range(want)]
        pos = [pos[0]] + sorted(set(picked)) + [pos[-1]]
    return pos


def _dedup(rows: list, n: int) -> Optional[list]:
    """One row per generated token: a layer may fire k times per token (the LM head
    does on gemma-4); keep the last of each group, or give up on a misaligned count."""
    if n <= 0 or not rows or len(rows) % n != 0:
        return None
    stride = len(rows) // n
    chosen = rows[stride - 1::stride]
    return chosen if len(chosen) == n else None


def _stack_rows(rows: list) -> np.ndarray:
    """(n, D) float16 on the host from device rows (torch) or arrays — ONE transfer."""
    first = rows[0]
    if hasattr(first, "cpu") and hasattr(first, "dtype") and not isinstance(first, np.ndarray):
        import torch
        return torch.stack(list(rows)).to(torch.float16).cpu().numpy()
    return np.stack([np.asarray(r, dtype=np.float16).reshape(-1) for r in rows])


def reduce_capture(
    gen_ids: Sequence[int],
    per_layer: dict,
    decode: Callable[[Sequence[int]], str],
    *,
    store_layers: Sequence[int],
    axes: Sequence[Axis] = (),
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> Optional[dict]:
    """Turn the hooks' per-step rows into the stored capture.

    ``per_layer`` maps a layer index to its list of last-position rows, one per hook
    call. Returns ``{"positions", "residuals": {layer: (P, D) fp16}, "axes": {name:
    [z per token]}, "n_tokens", "layers"}`` or None when nothing aligned. A layer whose
    row count does not divide by the token count is dropped, not misattributed.
    """
    n = len(gen_ids)
    if n == 0 or not per_layer:
        return None
    mats: dict[int, np.ndarray] = {}
    for layer, rows in per_layer.items():
        chosen = _dedup(list(rows), n)
        if chosen is None:
            continue
        try:
            mats[int(layer)] = _stack_rows(chosen)
        except Exception as e:
            print(f"[hidden] layer {layer}: rows unusable: {e}")
    if not mats:
        return None
    positions = select_positions(gen_ids, decode, max_positions)
    residuals = {layer: mats[layer][positions] for layer in sorted(mats)
                 if layer in set(int(x) for x in store_layers)}
    proj: dict[str, list[float]] = {}
    created: dict[str, str] = {}
    for ax in axes:
        mat = mats.get(int(ax.layer))
        if mat is None or mat.shape[1] != ax.vector.shape[0]:
            continue
        z = (mat.astype(np.float32) @ ax.vector - ax.mean) / (ax.std or 1.0)
        proj[ax.name] = [float(v) for v in z]
        created[ax.name] = getattr(ax, "created", "") or ""
    return {
        "positions": positions,
        "residuals": residuals,
        "axes": proj,
        "axes_created": created,
        "n_tokens": n,
        "layers": sorted(residuals),
        "hidden_dim": int(next(iter(mats.values())).shape[1]),
    }


# ── one forward pass, read through the same hooks ────────────────────────────

def forward_ids_rows(model: Any, input_ids: Sequence[int], layers: Sequence[int], *,
                     all_positions: bool = False, from_position: int = 0) -> dict:
    """Residuals of an exact token sequence at the given decoder layers, read through
    the SAME modules the live capture hooks (`find_decoder_layers`), so an axis, a
    live projection and a backfill share one convention: L{i} = the output of decoder
    layer i. Returns ``{layer: (T, D) float32 numpy}``: the last position only, or
    every position from `from_position` on with `all_positions` (the backfill passes
    the prompt length so only the generated tokens' rows come back — a long prefix's
    rows would be hundreds of MB for nothing). Teacher-forced, no gradient, one
    sequence. Torch is imported here and nowhere else in this module."""
    import torch
    layers_mod = find_decoder_layers(model)
    if layers_mod is None:
        raise RuntimeError("no decoder stack found on the model")
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = torch.tensor([list(int(t) for t in input_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    start = max(0, int(from_position))

    rows: dict = {}
    hooks = []

    def _make(li: int):
        def _hook(module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h is None or not hasattr(h, "dim"):
                return
            if h.dim() == 3:
                sel = h[0, start:, :] if all_positions else h[0, -1:, :]
            elif h.dim() == 2:
                sel = h[start:, :] if all_positions else h[-1:, :]
            else:
                sel = h.reshape(1, -1)
            rows[li] = sel.detach().to(torch.float32).cpu().numpy()
        return _hook

    for li in layers:
        try:
            hooks.append(layers_mod[int(li)].register_forward_hook(_make(int(li))))
        except Exception:
            continue
    try:
        with torch.inference_mode():
            _forward_no_logits(model, ids, attn)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
    return rows


def _decoder_parent(model: Any):
    """The module that owns the decoder stack (Gemma-4: `model.language_model`,
    Qwen / Llama: `model.model`) — callable on its own for a text-only forward that
    stops before the LM head."""
    try:
        for name, mod in model.named_modules():
            if str(name).endswith("layers") and type(mod).__name__ == "ModuleList" and len(mod) >= 4:
                if mod is find_decoder_layers(model):
                    parent_name = str(name).rsplit(".", 1)[0]
                    return model.get_submodule(parent_name)
    except Exception:
        return None
    return None


def _forward_no_logits(model: Any, ids, attn) -> None:
    """Run the decoder over `ids` for the hooks' sake while allocating as little else as
    possible. The default forward builds a KV cache it never reads and projects EVERY
    position through the LM head — on a 262k vocabulary that is ~1 MB of fp32 logits per
    token, several GB for a replayed 8k-token exchange, and it was what put the first
    backfill run out of memory on a 31 GB card. Three attempts, most frugal first:
    (1) the top-level forward with `logits_to_keep=1` and `use_cache=False`; (2) the
    decoder parent alone (no LM head at all); (3) the plain forward."""
    import torch
    try:
        model(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=1)
        return
    except TypeError:
        pass
    except torch.cuda.OutOfMemoryError:
        raise
    parent = _decoder_parent(model)
    if parent is not None:
        try:
            parent(input_ids=ids, attention_mask=attn, use_cache=False)
            return
        except TypeError:
            pass
        except torch.cuda.OutOfMemoryError:
            raise
    model(input_ids=ids, attention_mask=attn)


def forward_rows(model: Any, tokenizer: Any, text: str, layers: Sequence[int], *,
                 all_positions: bool = False, add_special_tokens: bool = True,
                 max_tokens: int = 4096) -> tuple:
    """`forward_ids_rows` over the tokenization of `text`: ``({layer: (T, D)}, token_ids)``
    with T = 1 unless `all_positions`. `add_special_tokens` is False for text that
    already went through the chat template."""
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    enc = text_tok(text, add_special_tokens=add_special_tokens, truncation=True,
                   max_length=max_tokens)
    ids = [int(t) for t in enc["input_ids"]]
    rows = forward_ids_rows(model, ids, layers, all_positions=all_positions)
    return rows, ids


def framed_reply_text(tokenizer: Any, system: str, user: str, reply: str) -> tuple:
    """(chat-templated text, reply token ids): `reply` as the model's turn after the
    user's, under `system`. The reader for "how does this reply read in the framing it
    was generated in" — the axis probe and the extraction script's reply baseline."""
    from core.llm_shared import build_inference_prompt
    conv = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user or "…"},
        {"role": "assistant", "content": reply},
    ]
    text = build_inference_prompt(tokenizer, conv, enable_thinking=False)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    reply_ids = text_tok.encode(reply, add_special_tokens=False)
    return text, reply_ids


def find_subseq(hay: Sequence[int], needle: Sequence[int], probe_len: int = 8) -> Optional[int]:
    """Start index of the LAST occurrence of `needle[:probe_len]` in `hay`, or None —
    how a reply's tokens are located inside the templated turn (the template may
    retokenize the boundary, so the first few tokens are matched, not the whole)."""
    probe = list(needle[:probe_len])
    if not probe:
        return None
    hay = list(hay)
    n, m = len(hay), len(probe)
    for i in range(n - m, -1, -1):
        if hay[i:i + m] == probe:
            return i
    return None


def persona_empty_system(prompts_dir: Path) -> str:
    """The original standing prompt and the persona slot's undecided framing, from the
    prompt files: the system turn a transcript without a stored one is replayed under,
    and the capture bench's persona-empty condition (the server builds the same text)."""
    parts = []
    for name in ("chat_prompt.txt", "persona_undecided_prompt.txt"):
        p = Path(prompts_dir) / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n".join(x for x in parts if x)


def replay_conversation(doc: dict, index: int, fallback_system: str = "") -> tuple:
    """(conversation, framing) as a live chat turn assembled it: the exchange's stored
    system content, the earlier exchanges CoT-stripped, the user turn. A transcript with
    no stored system content (pre-v2) is replayed under `fallback_system`, framing
    ``fallback``. The one rebuild the backfill and the capture probe's states share."""
    exs = doc.get("exchanges") or []
    ex = exs[index]
    system = str(ex.get("system_content") or "")
    framing = "replay"
    if not system:
        system, framing = fallback_system, "fallback"
    conv = [{"role": "system", "content": system}] if system else []
    for prev in exs[:index]:
        if not isinstance(prev, dict):
            continue
        conv.append({"role": "user", "content": str(prev.get("user_prompt") or "")})
        conv.append({"role": "assistant", "content": str(prev.get("assistant_response") or "")})
    conv.append({"role": "user", "content": str(ex.get("user_prompt") or "")})
    return conv, framing


def reply_span_rows(model: Any, tokenizer: Any, system: str, user: str, reply: str,
                    layers: Sequence[int]) -> Optional[dict]:
    """{layer: (T_reply, D)} — the residuals of the reply's own tokens as they read
    inside the framed turn; None when the reply cannot be located in the template."""
    text, reply_ids = framed_reply_text(tokenizer, system, user, reply)
    rows, ids = forward_rows(model, tokenizer, text, layers, all_positions=True,
                             add_special_tokens=False)
    start = find_subseq(ids, reply_ids)
    if start is None:
        return None
    end = min(len(ids), start + len(reply_ids))
    return {l: m[start:end] for l, m in rows.items()}


# ── the sidecar ──────────────────────────────────────────────────────────────

def sidecar_path(transcript: Path) -> Path:
    """``chats/<ts>.json`` → ``chats/<ts>.hidden.npz``."""
    transcript = Path(transcript)
    return transcript.with_name(transcript.stem + SIDECAR_SUFFIX)


def _put(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    with zf.open(name, "w") as fh:
        np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)


def write_sidecar(transcript: Path, exchange_id: str, capture: Optional[dict], *,
                  model_id: str, adapter_id: Optional[str], layer_spec: str = "",
                  variant: str = "", extra_meta: Optional[dict] = None,
                  token_ids: Optional[Sequence[int]] = None) -> Optional[Path]:
    """Append one exchange's capture to the transcript's ``.hidden.npz``.

    Members are ``<group>/positions``, ``<group>/L<layer>`` (P, D) fp16,
    ``<group>/axis_<name>`` per-token float32, ``<group>/token_ids`` int32 when given (a
    drive opener's group only: a chat reply's ids are in its transcript's tension block),
    and ``<group>/meta`` (a JSON string as a
    0-d array: model + adapter revision, layers, dims, when), where ``<group>`` is the
    exchange id for the live capture and ``<exchange_id>~<variant>`` for a backfill
    under other weights (`member_group`). A group already present is left alone — a zip
    cannot replace a member in place, and the first capture of an exchange under given
    weights is the one kept. A capture with axis projections but no stored residuals
    (an axes-only backfill) is written too.
    """
    if not capture or not (capture.get("residuals") or capture.get("axes")):
        return None
    exchange_id = str(exchange_id or "").strip()
    variant = str(variant or "").strip()
    if not exchange_id or "/" in exchange_id or VARIANT_SEP in exchange_id or "/" in variant:
        return None
    path = sidecar_path(transcript)
    prefix = f"{member_group(exchange_id, variant)}/"
    meta = {
        "exchange_id": exchange_id,
        "variant": variant,
        "model_id": model_id or "",
        "adapter_id": adapter_id or "",
        "layers": [int(x) for x in capture.get("layers") or []],
        "layer_spec": layer_spec,
        "hidden_dim": int(capture.get("hidden_dim") or 0),
        "n_tokens": int(capture.get("n_tokens") or 0),
        "n_positions": len(capture.get("positions") or []),
        "position_rule": "0 + after-newline + last",
        "residual": "output of decoder layer L (0-based), last position, fp16",
        "axes": sorted((capture.get("axes") or {}).keys()),
        "axes_created": dict(capture.get("axes_created") or {}),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        if prefix + "meta.npy" in set(zf.namelist()):
            print(f"[hidden] {path.name}: {prefix[:-1]} already captured — kept")
            return None
        _put(zf, prefix + "positions.npy", np.asarray(capture.get("positions") or [], dtype=np.int32))
        for layer, mat in sorted((capture.get("residuals") or {}).items()):
            _put(zf, f"{prefix}L{int(layer)}.npy", np.asarray(mat, dtype=np.float16))
        for name, series in sorted((capture.get("axes") or {}).items()):
            _put(zf, f"{prefix}axis_{name}.npy", np.asarray(series, dtype=np.float32))
        if token_ids is not None:
            _put(zf, prefix + "token_ids.npy", np.asarray(list(token_ids), dtype=np.int32))
        _put(zf, prefix + "meta.npy", np.array(json.dumps(meta)))
    return path


def token_span(gen_ids: Sequence[int], decode: Callable[[Sequence[int]], str],
               text: str) -> Optional[tuple]:
    """``(start, end)`` token indices into ``gen_ids`` covering the LAST occurrence of
    ``text`` in their decoding, or None when it is not found. The last, because a drive's
    thought often drafts the opener before the answer states it. Found by its first and
    last 64 characters, so a cleaner that only touched the middle still matches; token
    boundaries by binary search over decoded-prefix lengths (a dozen decodes)."""
    ids = [int(t) for t in gen_ids]
    text = (text or "").strip()
    if not ids or not text:
        return None
    full = decode(ids)
    head = text[:64]
    start_c = full.rfind(head)
    if start_c < 0:
        return None
    tail = text[-64:]
    tail_at = full.find(tail, start_c)
    end_c = tail_at + len(tail) if tail_at >= 0 else min(len(full), start_c + len(text))

    def reaching(c: int) -> int:
        # smallest k whose decoded prefix ids[:k] is at least c characters long
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(decode(ids[:mid])) >= c:
                hi = mid
            else:
                lo = mid + 1
        return lo

    start = max(0, reaching(start_c + 1) - 1)    # the token holding the first character
    end = max(start + 1, reaching(end_c))
    return (start, end)


def write_opener_capture(transcript: Optional[Path], exchange_id: str,
                         taken: Optional[dict], opener: str, *, drive: str,
                         tokenizer: Any) -> Optional[Path]:
    """File a drive pass's capture under the opener's exchange (see *Drive openers*).

    ``taken`` is the reflect-generate factory's ``last_hidden_capture``: ``{"capture",
    "layer_spec", "model_id", "adapter_id"}``, the capture carrying the pass's generated
    ``token_ids``. Best-effort: telemetry never fails a reach-out, so every failure is a
    printed line and None."""
    if transcript is None or not taken or not (taken.get("capture") or {}):
        return None
    try:
        cap = taken["capture"]
        ids = [int(t) for t in cap.get("token_ids") or []]
        span = None
        if ids and tokenizer is not None:
            text_tok = getattr(tokenizer, "tokenizer", tokenizer)
            span = token_span(ids, lambda x: text_tok.decode(list(x)), opener)
        return write_sidecar(
            Path(transcript), exchange_id, cap,
            model_id=str(taken.get("model_id") or ""), adapter_id=taken.get("adapter_id"),
            layer_spec=str(taken.get("layer_spec") or ""),
            extra_meta={"seam": "drive", "drive": drive,
                        "opener_span": list(span) if span else None},
            token_ids=ids or None)
    except Exception as e:
        print(f"[hidden] {drive} opener capture not written: {type(e).__name__}: {e}")
        return None


def sidecar_index(path: Path) -> dict:
    """``{group: {"exchange_id", "variant", "layers": [..], "axes": [..]}}`` from the
    zip's directory alone — no arrays read (the inventory and the ledger call this once
    per transcript). Empty when the file is missing or unreadable."""
    out: dict = {}
    try:
        with zipfile.ZipFile(Path(path)) as zf:
            names = zf.namelist()
    except Exception:
        return out
    for n in names:
        group, sep, member = n.partition("/")
        if not sep or not member.endswith(".npy"):
            continue
        member = member[:-4]
        ex, var = split_group(group)
        rec = out.setdefault(group, {"exchange_id": ex, "variant": var, "layers": [], "axes": [],
                                     "meta": False})
        if member == "meta":
            rec["meta"] = True
        elif member.startswith("L") and member[1:].isdigit():
            rec["layers"].append(int(member[1:]))
        elif member.startswith("axis_"):
            rec["axes"].append(member[5:])
    for rec in out.values():
        rec["layers"].sort()
        rec["axes"].sort()
    return {g: r for g, r in out.items() if r["meta"]}


def captured_exchange_ids(path: Path, variant: str = "") -> set:
    """Exchange ids captured under `variant` ("" = the live capture, "base" = the
    bare-base backfill, else an adapter name), from the zip's directory alone."""
    return {r["exchange_id"] for r in sidecar_index(path).values() if r["variant"] == (variant or "")}


def read_sidecar(path: Path) -> dict:
    """``{exchange_id: {"meta", "positions", "layers": {L: (P, D)}, "axes": {name: [...]}}}``."""
    out: dict = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for key in z.files:
            ex, sep, member = key.partition("/")
            if not sep:
                continue
            rec = out.setdefault(ex, {"layers": {}, "axes": {}, "exchange_id": split_group(ex)[0],
                                      "variant": split_group(ex)[1]})
            if member == "meta":
                raw = z[key]
                rec["meta"] = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
            elif member == "positions":
                rec["positions"] = [int(x) for x in z[key]]
            elif member == "token_ids":
                rec["token_ids"] = [int(x) for x in z[key]]
            elif member.startswith("L"):
                try:
                    rec["layers"][int(member[1:])] = np.asarray(z[key])
                except ValueError:
                    pass
            elif member.startswith("axis_"):
                rec["axes"][member[5:]] = [float(x) for x in z[key]]
    return out


if __name__ == "__main__":
    # GPU-free self-test (python -m core.hidden_capture). numpy rows stand in for
    # device tensors; the reduce is the same code.
    import tempfile

    # layer spec: absolute + percent, dedup, out-of-range dropped, shallow model
    assert parse_layer_spec("2,3,4,5,50%,65%,80%", 34) == [2, 3, 4, 5, 16, 21, 26]
    assert parse_layer_spec("2, 40, x, 100%", 10) == [2, 9]
    assert parse_layer_spec("", 10) == [] and parse_layer_spec("3", 0) == []

    # decoder-stack finder on a fake model: the largest ModuleList named *layers
    ModuleList = type("ModuleList", (list,), {})
    class _Fake:
        def named_modules(self):
            return [("model.vision.layers", ModuleList(range(6))),
                    ("model.language_model.layers", ModuleList(range(34))),
                    ("model.language_model.layers.0.mlp", object())]
    assert len(find_decoder_layers(_Fake())) == 34
    assert find_decoder_layers(object()) is None

    # the layers cache must NOT keep a released model's weights alive (weak refs)
    import gc as _gc
    _LayerList = type("ModuleList", (list,), {})
    class _Model:
        def __init__(self):
            self._layers = _LayerList(range(8))
        def named_modules(self):
            return [("model.layers", self._layers)]
    _config = lambda: {}   # the defaults (capture on), whatever this box's file says
    _m = _Model()
    _layers_ref = weakref.ref(_m._layers)
    _spec = capture_spec(_m, "m", Path("/nonexistent-axes"))
    assert _spec is None or _spec["n_layers"] == 8
    assert _LAYERS_CACHE is not None and _LAYERS_CACHE[1]() is _m._layers
    _spec = None
    del _m
    _gc.collect()
    assert _layers_ref() is None, "hidden_capture cache pinned a released model's layers"

    # positions: 0, after each newline, last; thinning keeps the ends
    NL = 7
    ids = [1, 2, NL, 3, 4, NL, 5]
    dec = lambda t: "\n" if t[0] == NL else "a"
    assert select_positions(ids, dec) == [0, 3, 6]
    assert find_subseq([1, 2, 3, 4, 2, 3, 9], [2, 3]) == 4 and find_subseq([1, 2], [7]) is None
    assert find_subseq([5, 6, 7, 8, 9], [7, 8, 9, 10, 11, 12, 13, 14, 15], probe_len=3) == 2
    many = [NL] * 100
    pos = select_positions(many, dec, max_positions=10)
    assert pos[0] == 0 and pos[-1] == 99 and len(pos) <= 10

    # reduce: stride-2 rows (a layer firing twice per token), one stored layer, an
    # axis on a non-stored layer, and a misaligned layer dropped
    D = 8
    n = len(ids)
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, D)).astype(np.float32)
    rows_l2 = []
    for t in range(n):
        rows_l2.append(np.zeros(D, dtype=np.float32))   # the pre-fire, must be skipped
        rows_l2.append(base[t])
    rows_l9 = [base[t] * 2 for t in range(n)]
    per_layer = {2: rows_l2, 9: rows_l9, 4: rows_l9[:-1]}   # 4 is misaligned
    vec = np.zeros(D, dtype=np.float32); vec[0] = 1.0
    ax = Axis("pain", 9, vec, mean=0.0, std=2.0, model_id="m", path="")
    cap = reduce_capture(ids, per_layer, dec, store_layers=[2], axes=[ax])
    assert cap is not None and cap["layers"] == [2] and 4 not in cap["residuals"]
    assert cap["residuals"][2].shape == (3, D) and cap["residuals"][2].dtype == np.float16
    assert np.allclose(cap["residuals"][2][1], base[3].astype(np.float16))
    assert len(cap["axes"]["pain"]) == n
    assert abs(cap["axes"]["pain"][3] - float(base[3, 0])) < 1e-3   # (2·x0 − 0)/2
    assert reduce_capture([], per_layer, dec, store_layers=[2]) is None

    # sidecar: append two exchanges, refuse a duplicate, read back
    with tempfile.TemporaryDirectory() as td:
        transcript = Path(td) / "20260922_100000.json"
        p = write_sidecar(transcript, "ex1", cap, model_id="m", adapter_id="a")
        assert p == Path(td) / "20260922_100000.hidden.npz" and p.is_file()
        assert write_sidecar(transcript, "ex2", cap, model_id="m", adapter_id=None) == p
        assert write_sidecar(transcript, "ex1", cap, model_id="m", adapter_id="a") is None
        assert write_sidecar(transcript, "bad/id", cap, model_id="m", adapter_id="a") is None
        back = read_sidecar(p)
        assert set(back) == {"ex1", "ex2"}
        assert captured_exchange_ids(p) == {"ex1", "ex2"}
        assert captured_exchange_ids(Path(td) / "missing.npz") == set()
        # a backfill variant sits beside the live capture and is listed apart from it
        assert write_sidecar(transcript, "ex1", cap, model_id="m", adapter_id="", variant="base",
                             extra_meta={"framing": "replay"}) == p
        assert write_sidecar(transcript, "ex1", cap, model_id="m", adapter_id="", variant="base") is None
        assert captured_exchange_ids(p) == {"ex1", "ex2"} and captured_exchange_ids(p, "base") == {"ex1"}
        idx = sidecar_index(p)
        assert set(idx) == {"ex1", "ex2", "ex1~base"} and idx["ex1~base"]["variant"] == "base"
        assert idx["ex1~base"]["layers"] == [2] and idx["ex1~base"]["axes"] == ["pain"]
        back = read_sidecar(p)
        assert back["ex1~base"]["meta"]["framing"] == "replay" and back["ex1~base"]["variant"] == "base"
        # an axes-only capture (no stored residuals) is still written
        cap_axes = dict(cap, residuals={}, layers=[])
        assert write_sidecar(transcript, "ex3", cap_axes, model_id="m", adapter_id="") == p
        assert sidecar_index(p)["ex3"]["axes"] == ["pain"] and sidecar_index(p)["ex3"]["layers"] == []
        assert write_sidecar(transcript, "bad~id", cap, model_id="m", adapter_id="") is None
        assert exchange_key({"exchange_id": "abc"}, 3) == "abc" and exchange_key({}, 3) == "ex3"
        assert back["ex1"]["meta"]["adapter_id"] == "a" and back["ex2"]["meta"]["adapter_id"] == ""
        assert back["ex1"]["positions"] == [0, 3, 6]
        assert np.array_equal(back["ex1"]["layers"][2], cap["residuals"][2])
        assert len(back["ex1"]["axes"]["pain"]) == n
        assert back["ex1"]["meta"]["layers"] == [2] and back["ex1"]["meta"]["hidden_dim"] == D

        # axes dir: a matching axis loads, a foreign one is skipped, cache refreshes on change
        axes_dir = Path(td) / "axes"
        axes_dir.mkdir()
        np.savez(axes_dir / "pain.npz", vector=vec, layer=9, mean=0.0, std=2.0, model_id="m")
        np.savez(axes_dir / "other.npz", vector=vec, layer=9, mean=0.0, std=1.0, model_id="not-m")
        got = load_axes(axes_dir, "m")
        assert [a.name for a in got] == ["pain"] and got[0].std == 2.0
        assert load_axes(axes_dir, "m") == got            # cached
        (axes_dir / "other.npz").unlink()
        np.savez(axes_dir / "relief.npz", vector=vec, layer=3)  # no model_id ⇒ accepted
        assert [a.name for a in load_axes(axes_dir, "m")] == ["pain", "relief"]
        # a live baseline in the file wins over the extraction one
        np.savez(axes_dir / "pain.npz", vector=vec, layer=9, mean=0.0, std=2.0, model_id="m",
                 live_mean=5.0, live_std=0.5)
        got = [a for a in load_axes(axes_dir, "m") if a.name == "pain"][0]
        assert (got.mean, got.std, got.baseline) == (5.0, 0.5, "live")
        assert load_axes(Path(td) / "nowhere", "m") == []

        # the live turn's prompt rebuild: stored system content, earlier turns, the message
        doc = {"exchanges": [{"user_prompt": "a", "assistant_response": "b"},
                             {"user_prompt": "c", "assistant_response": "d", "system_content": "S"}]}
        conv, framing = replay_conversation(doc, 1, "F")
        assert framing == "replay" and conv[0] == {"role": "system", "content": "S"}
        assert [m["content"] for m in conv[1:]] == ["a", "b", "c"] and conv[-1]["role"] == "user"
        conv, framing = replay_conversation(doc, 0, "F")
        assert framing == "fallback" and conv[0]["content"] == "F" and conv[-1]["content"] == "a"
        prompts = Path(td) / "prompts"
        prompts.mkdir()
        (prompts / "chat_prompt.txt").write_text("Base.\n", encoding="utf-8")
        assert persona_empty_system(prompts) == "Base."
        (prompts / "persona_undecided_prompt.txt").write_text(" Undecided. \n", encoding="utf-8")
        assert persona_empty_system(prompts) == "Base.\n\nUndecided."

        # a drive opener: its span in the pass's ids is the LAST occurrence (the thought
        # drafted it first), and the group carries the ids and the drive's marker
        vocab = ["<t>", "draft: ", "Hello", " there", "</t>", "DECISION: yes\n", "OPENER: ",
                 ", friend", "."]

        class _Tok:
            def decode(self, ids):
                return "".join(vocab[i] for i in ids)

        ids = [0, 1, 2, 3, 4, 5, 6, 2, 3, 7, 8]
        dec = _Tok().decode
        a, b = token_span(ids, dec, "Hello there, friend.")
        assert (a, b) == (7, 11), (a, b)
        assert dec(ids[a:b]) == "Hello there, friend."
        assert token_span(ids, dec, "Hello there") == (7, 9)
        assert token_span(ids, dec, "absent") is None and token_span([], dec, "x") is None
        opener_cap = {"positions": [0, 10], "residuals": {2: np.zeros((2, D), np.float16)},
                      "axes": {"pain": [0.0] * len(ids)}, "layers": [2], "hidden_dim": D,
                      "n_tokens": len(ids), "token_ids": ids}
        taken = {"capture": opener_cap, "layer_spec": "2", "model_id": "m", "adapter_id": "ad"}
        assert write_opener_capture(transcript, "op0", taken, " Hello there, friend. ",
                                    drive="outreach", tokenizer=_Tok()) == p
        back = read_sidecar(p)
        assert back["op0"]["token_ids"] == ids and "token_ids" not in back["ex1"]
        m = back["op0"]["meta"]
        assert (m["seam"], m["drive"], m["opener_span"]) == ("drive", "outreach", [7, 11])
        assert m["variant"] == "" and m["adapter_id"] == "ad"
        assert write_opener_capture(transcript, "op1", None, "x", drive="checkin",
                                    tokenizer=_Tok()) is None
        assert write_opener_capture(None, "op1", taken, "x", drive="checkin",
                                    tokenizer=_Tok()) is None
        # an opener the ids do not contain is still filed, with no span
        assert write_opener_capture(transcript, "op2", taken, "not in there", drive="synthesis",
                                    tokenizer=_Tok()) == p
        assert read_sidecar(p)["op2"]["meta"]["opener_span"] is None

    print("hidden_capture self-test: OK")
