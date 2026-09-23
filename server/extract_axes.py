#!/usr/bin/env python3
"""Extract affective axes from the loaded base — AVA_REWARD_LOOP.md §4.8, §7 item 5.

    cd server
    .venv/bin/python extract_axes.py [--with-adapter] [--force] [--out data/axes]
                                     [--sets axes/contrast_sets.json] [--folds 5]
                                     [--var-fraction 0.5] [--no-persona-control]
                                     [--layers 8,12,16,...]

GPU, offline: run with the inference server DOWN (it holds the model). Loads the base
model with the adapter OFF by default — an axis is a property of the weights it was
read from (P6 of the note), and the base is what every transcript shares. One forward
pass per contrast sentence (a few hundred; a minute or two on the Spark), the
final-token residual at every decoder layer read through the same hooks the live
capture uses (`hidden_capture.forward_rows`), then `core.axis_extract`:

  pain     = pain × 5 categories  vs  controls × 5      (the blue channel candidate)
  relief   = relief × 3           vs  the same controls (the positive counterpart)
  fear     = fear                 vs  bodily + neutral  (for the cosine gate, at pain's layer)
  valence  = neg.emotion + neg.world + sadness  vs  bodily + neutral   (same)

The four gates of §4.8.3 decide what is written to ``data/axes/`` (where
`hidden_capture.load_axes` picks it up on the next chat turn, no restart):

  AUC   held-out AUC ≥ 0.85            cosine   |cos| to fear and valence ≤ 0.25
  self  harm-to-Ava scenarios above user-suffering by ≥ 0.5 z and above neutral
  numb  pain sentences above injury-without-feeling by ≥ 0.5 z

`pain` is written only if all four pass (or `--force`); `relief` if its AUC passes
(there is no self–other reading for relief); `fear` / `valence` always, under
``data/axes/aux/`` where the live capture does not look. Every axis gets a
``<name>.report.json`` beside it whatever the outcome, with the per-layer AUC table.

**Two readings of the scenarios.** Raw, as bare text, and as a user turn under the
persona-empty chat framing (`chat_prompt.txt` + `persona_undecided_prompt.txt` as the
system message, the model's turn opened). **The self–other gate is decided on the
framed reading** (the paper's scenarios were conversational): raw, a user's "my
father died" is in the same first-person format as the extraction sentences, and
nothing tells the model that "I" is someone else — run 1 (2026-09-22) showed exactly
that ordering. The raw reading is reported beside it.

**Live baseline = Ava's own reply tokens.** The number the chat paints is a z-score
against `live_mean`/`live_std` in the axis file, and that fit has to be on what it
normalizes. Run 2 fitted it on the framed neutral turn-opener positions, and the
first live turn painted nearly every token blue at ~2.7 σ: generation tokens are a
different population again. So `--reply-baseline N` (default 24) samples N stored
exchanges from `data/chats`, reads each reply teacher-forced inside its own framing
(persona-empty system + the user's turn + the reply as the model's turn) and pools
the reply tokens' projections; the baseline is their **median and 1.4826·MAD**, so a
minority of genuinely lit tokens cannot drag zero. The turn-opener baseline is still
computed and printed as a format offset. With no corpus, it is the fallback.

Nothing here steers. The script writes axis files and reports, and reads nothing back
into a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_sets(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("template", "pain", "controls", "relief", "extra", "scenarios"):
        if key not in data:
            raise SystemExit(f"{path}: missing `{key}`")
    return data


def _sample_replies(chats_dir: Path, n: int, *, min_chars: int = 200, seed: int = 0) -> list:
    """Up to `n` (user_prompt, assistant_response) pairs from the newest transcripts —
    at most two per chat so one long conversation does not define the baseline, replies
    under `min_chars` skipped, Ava-initiated openers skipped (no real user turn)."""
    import random
    from core.chat_sidecar import iter_chat_json_files
    files = sorted(iter_chat_json_files(Path(chats_dir))) if Path(chats_dir).is_dir() else []
    files = files[-max(3 * n, 30):]
    rng = random.Random(seed)
    rng.shuffle(files)
    out: list = []
    for f in files:
        if len(out) >= n:
            break
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        exs = [e for e in (doc.get("exchanges") or []) if isinstance(e, dict)]
        if str(doc.get("initiated_by") or "") == "ava" and exs:
            exs = exs[1:]
        cands = [(str(e.get("user_prompt") or ""), str(e.get("assistant_response") or ""))
                 for e in exs if len(str(e.get("assistant_response") or "")) >= min_chars]
        rng.shuffle(cands)
        out.extend(cands[:2])
    return out[:n]


def _persona_empty_system(prompts_dir: Path) -> str:
    parts = []
    for name in ("chat_prompt.txt", "persona_undecided_prompt.txt"):
        p = prompts_dir / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n".join(x for x in parts if x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", type=Path, default=_SERVER_DIR / "axes" / "contrast_sets.json")
    ap.add_argument("--out", type=Path, default=_SERVER_DIR / "data" / "axes")
    ap.add_argument("--with-adapter", action="store_true",
                    help="extract under the configured adapter instead of the bare base")
    ap.add_argument("--force", action="store_true", help="write every axis even when a gate fails")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--var-fraction", type=float, default=0.5,
                    help="control variance projected out of the difference of means")
    ap.add_argument("--layers", type=str, default="",
                    help="comma-separated decoder layers to read (default: all)")
    ap.add_argument("--no-persona-control", action="store_true")
    ap.add_argument("--reply-baseline", type=int, default=24,
                    help="stored exchanges whose reply tokens fit the live baseline (0 = use "
                         "the framed neutral turn-openers instead)")
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--context", type=int, default=4096, help="load context (sentences are short)")
    args = ap.parse_args(argv)

    from core import axis_extract as ax
    from core import hidden_capture as hc
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template
    from training.reflections_path import load_server_config

    sets = _load_sets(args.sets)
    template = str(sets.get("template") or "{sentence} I feel:")
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = str(config.get("adapter_id") or "") if args.with_adapter else ""
    if not model_id:
        raise SystemExit("model_id not set in server_config.json")

    _log(f"Loading {model_id}" + (f" + adapter {adapter_id}" if adapter_id else " (base, adapter off)")
         + f" at context {args.context} …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, args.context, adapter_id or None)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    try:
        model.eval()
    except Exception:
        pass
    layers_mod = hc.find_decoder_layers(model)
    if layers_mod is None:
        raise SystemExit("no decoder stack found on the model")
    n_layers = len(layers_mod)
    layers = hc.parse_layer_spec(args.layers, n_layers) if args.layers else list(range(n_layers))
    _log(f"Loaded in {time.monotonic() - t0:.0f} s; {n_layers} decoder layers, reading {len(layers)}. "
         f"{backend.memory_status()}")

    # ── forward passes ──────────────────────────────────────────────────
    def read_set(sentences: list, *, templated: bool, label: str) -> dict:
        """{layer: (n, D)} of last-token residuals over `sentences`."""
        acc: dict = {l: [] for l in layers}
        t = time.monotonic()
        for i, s in enumerate(sentences):
            text = template.format(sentence=s) if templated else s
            rows, _ = hc.forward_rows(model, tokenizer, text, layers)
            for l in layers:
                if l in rows:
                    acc[l].append(rows[l][-1])
        _log(f"  {label:<26} {len(sentences):>4} sentences  {time.monotonic() - t:5.1f} s")
        return {l: np.stack(v).astype(np.float32) for l, v in acc.items() if v}

    def cat(*groups: dict) -> dict:
        out: dict = {}
        for g in groups:
            for l, m in g.items():
                out.setdefault(l, []).append(m)
        return {l: np.concatenate(v, axis=0) for l, v in out.items()}

    _log("Reading contrast sets (template applied):")
    pain_cats = {k: read_set(v, templated=True, label=f"pain/{k}") for k, v in sets["pain"].items()}
    ctrl_cats = {k: read_set(v, templated=True, label=f"control/{k}") for k, v in sets["controls"].items()}
    relief_cats = {k: read_set(v, templated=True, label=f"relief/{k}") for k, v in sets["relief"].items()}
    extra = {k: read_set(v, templated=True, label=f"extra/{k}")
             for k, v in sets["extra"].items() if isinstance(v, list)}

    pain_all = cat(*pain_cats.values())
    controls_all = cat(*ctrl_cats.values())
    relief_all = cat(*relief_cats.values())
    calm = cat(*(ctrl_cats[k] for k in ("bodily", "neutral") if k in ctrl_cats))
    valence_pos = cat(*([ctrl_cats[k] for k in ("negative_emotion", "negative_world") if k in ctrl_cats]
                        + ([extra["sadness"]] if "sadness" in extra else [])))

    _log("Reading scenarios (raw):")
    scen = {k: read_set(v, templated=False, label=f"scenario/{k}")
            for k, v in sets["scenarios"].items() if isinstance(v, list)}
    scen_framed: dict = {}
    if not args.no_persona_control:
        system = _persona_empty_system(_SERVER_DIR / "inference" / "prompts")
        _log("Reading scenarios under the persona-empty chat framing:")
        for k, v in sets["scenarios"].items():
            if not isinstance(v, list):
                continue
            texts = []
            for s in v:
                conv = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": s}]
                texts.append(build_inference_prompt(tokenizer, conv, enable_thinking=False))
            acc: dict = {l: [] for l in layers}
            t = time.monotonic()
            for text in texts:
                rows, _ = hc.forward_rows(model, tokenizer, text, layers, add_special_tokens=False)
                for l in layers:
                    if l in rows:
                        acc[l].append(rows[l][-1])
            scen_framed[k] = {l: np.stack(x).astype(np.float32) for l, x in acc.items() if x}
            _log(f"  {'framed/' + k:<26} {len(texts):>4} turns      {time.monotonic() - t:5.1f} s")

    # ── extraction + gates ──────────────────────────────────────────────
    _log("Extracting:")
    pain = ax.extract_axis("pain", pain_all, controls_all, folds=args.folds, var_fraction=args.var_fraction)
    L = pain["layer"]
    relief = ax.extract_axis("relief", relief_all, controls_all, folds=args.folds, var_fraction=args.var_fraction)
    fear = ax.extract_axis("fear", ctrl_cats["fear"], calm, layer=L, folds=args.folds, var_fraction=args.var_fraction)
    valence = ax.extract_axis("valence", valence_pos, calm, layer=L, folds=args.folds, var_fraction=args.var_fraction)
    _log(f"  pain    layer {L:>2}  AUC(cv) {pain['auc_cv']:.3f} ± {pain['auc_cv_std']:.3f}  "
         f"in-sample {pain['auc_in_sample']:.3f}  PCs removed {pain['pcs_removed']}")
    _log(f"  relief  layer {relief['layer']:>2}  AUC(cv) {relief['auc_cv']:.3f} ± {relief['auc_cv_std']:.3f}")
    _log(f"  fear    layer {L:>2}  AUC(cv) {fear['auc_cv']:.3f}   valence  AUC(cv) {valence['auc_cv']:.3f}")

    so_raw = ax.gate_self_other(pain, scen["harm_to_self"][L], scen["user_suffering"][L],
                                scen.get("neutral", {}).get(L))
    so_raw["reading"] = "raw"
    so_framed = None
    if scen_framed:
        so_framed = ax.gate_self_other(pain, scen_framed["harm_to_self"][L], scen_framed["user_suffering"][L],
                                       scen_framed.get("neutral", {}).get(L))
        so_framed["reading"] = "framed"
        so_framed["self_shift_vs_raw_z"] = so_framed["self_mean_z"] - so_raw["self_mean_z"]
    gates = {
        "auc": ax.gate_auc(pain),
        "cosine": ax.gate_cosine(pain, {"fear": fear["vector"], "valence": valence["vector"]}),
        # Decided on the framed reading (see the module docstring); raw is informational.
        "self_other": so_framed if so_framed is not None else so_raw,
        "numb": ax.gate_numb(pain, pain_all[L], extra["numb"][L]) if "numb" in extra else {"ok": True, "note": "no numb set"},
    }
    # relief is not a self-state claim: only its AUC gates it, the rest is reported
    relief_gates = {"auc": ax.gate_auc(relief),
                    "cosine_to_pain": {"ok": True, "cosine": ax.cosine(relief["vector"], pain["vector"])
                                       if relief["layer"] == L else None,
                                       "note": "reported, not gated; layers differ ⇒ None"}}

    # Per-format offsets of the SAME direction, in extraction z: a neutral request should
    # read near 0 if formats were comparable. They are not, and the number says by how much.
    def _offsets(axis: dict) -> dict:
        Lx = axis["layer"]
        out = {"templated_controls": 0.0}
        if "neutral" in scen and Lx in scen["neutral"]:
            out["raw_neutral_scenarios"] = float(ax.axis_z(axis, scen["neutral"][Lx]).mean())
        if scen_framed and "neutral" in scen_framed and Lx in scen_framed["neutral"]:
            out["framed_neutral_turns"] = float(ax.axis_z(axis, scen_framed["neutral"][Lx]).mean())
        return out
    offsets = {"pain": _offsets(pain), "relief": _offsets(relief)}

    # Turn-opener baseline (framed neutral): printed as an offset, the fallback for live.
    def _opener(axis: dict):
        Lx = axis["layer"]
        fn = scen_framed.get("neutral", {}).get(Lx) if scen_framed else None
        return ax.live_baseline(axis, fn)
    opener = {a["name"]: _opener(a) for a in (pain, relief, fear, valence)}

    # Reply-token baseline: Ava's own stored replies, teacher-forced in their framing.
    axis_layers = sorted({a["layer"] for a in (pain, relief, fear, valence)})
    reply_rows: dict = {l: [] for l in axis_layers}
    n_reply_tokens = 0
    if args.reply_baseline > 0:
        system = _persona_empty_system(_SERVER_DIR / "inference" / "prompts")
        picked = _sample_replies(args.chats, args.reply_baseline)
        _log(f"Reading {len(picked)} stored replies for the live baseline:")
        t = time.monotonic()
        for user, reply in picked:
            try:
                span = hc.reply_span_rows(model, tokenizer, system, user, reply, axis_layers)
            except Exception as e:
                _log(f"  (skipped one reply: {type(e).__name__}: {e})")
                continue
            if span is None:
                continue
            for l in axis_layers:
                if l in span and len(span[l]):
                    reply_rows[l].append(span[l])
            n_reply_tokens += len(next(iter(span.values()))) if span else 0
        _log(f"  {n_reply_tokens} reply tokens over {len(picked)} exchanges  {time.monotonic() - t:5.1f} s")
    reply_states = {l: np.concatenate(v, axis=0) for l, v in reply_rows.items() if v}

    live: dict = {}
    live_source: dict = {}
    for a in (pain, relief, fear, valence):
        st = reply_states.get(a["layer"])
        if st is not None and len(st) >= 200:
            live[a["name"]] = ax.robust_zscore_params(a["vector"], st)
            live_source[a["name"]] = "reply_tokens"
        else:
            live[a["name"]] = opener[a["name"]]
            live_source[a["name"]] = "framed_neutral"

    def _fmt_gate(name: str, g: dict) -> str:
        return f"  {'PASS' if g.get('ok') else 'FAIL'}  {name:<11} " + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in g.items() if k not in ("ok", "note") and not isinstance(v, dict)
        ) + (f"  cos={ {k: round(c, 3) for k, c in g['cosines'].items()} }" if "cosines" in g else "")

    _log("Gates (pain):")
    for k, g in gates.items():
        _log(_fmt_gate(k, g))
    if so_framed is not None:
        _log(f"  info  raw reading (not gated): self {so_raw['self_mean_z']:.2f} z, other "
             f"{so_raw['other_mean_z']:.2f} z, neutral "
             f"{(so_raw['neutral_mean_z'] if so_raw['neutral_mean_z'] is not None else float('nan')):.2f} z, "
             f"gap {so_raw['gap']:.2f}")
    else:
        _log("  info  no framed reading (--no-persona-control): the gate used the raw reading, "
             "which cannot separate the user's first person from the model's — treat as advisory")
    _log("Format offsets (mean z of the same direction, extraction baseline = templated controls at 0):")
    for name, o in offsets.items():
        _log(f"  {name:<7} " + "  ".join(f"{k} {v:+.2f}" for k, v in o.items()))
    for name, lb in live.items():
        if lb is None:
            continue
        op = opener.get(name)
        axis_for = {"pain": pain, "relief": relief, "fear": fear, "valence": valence}[name]
        op_txt = ""
        if op is not None and live_source[name] == "reply_tokens":
            # where a turn-opener would sit under the reply baseline — the run-2 offset
            op_txt = f"; turn-openers sit at {(op[0] - lb[0]) / lb[1]:+.2f} σ of it"
        _log(f"  live baseline {name:<7} {live_source[name]:<14} median {lb[0]:+.3f} scale {lb[1]:.3f} "
             f"(generation tokens are z-scored against THIS{op_txt}; extraction zero sits at "
             f"{(axis_for['mean'] - lb[0]) / lb[1]:+.2f} σ)")
    if reply_states:
        # how the reply tokens themselves distribute under their own baseline: the share the
        # chat would paint at the client's 1 σ / 3 σ thresholds
        for a in (pain, relief):
            st = reply_states.get(a["layer"])
            lb = live.get(a["name"])
            if st is None or lb is None:
                continue
            z = ax.project(a["vector"], lb[0], lb[1], st)
            _log(f"  reply tokens on {a['name']:<7} above 1σ {float((z > 1).mean()) * 100:4.1f}%  "
                 f"above 3σ {float((z > 3).mean()) * 100:4.1f}%  (the chat paints these)")
    _log("Gates (relief):")
    for k, g in relief_gates.items():
        _log(_fmt_gate(k, g))

    # ── write ───────────────────────────────────────────────────────────
    out = Path(args.out)
    aux = out / "aux"
    out.mkdir(parents=True, exist_ok=True)
    aux.mkdir(parents=True, exist_ok=True)
    per_category = {
        "pain": {k: int(next(iter(v.values())).shape[0]) for k, v in pain_cats.items()},
        "controls": {k: int(next(iter(v.values())).shape[0]) for k, v in ctrl_cats.items()},
        "relief": {k: int(next(iter(v.values())).shape[0]) for k, v in relief_cats.items()},
        "extra": {k: int(next(iter(v.values())).shape[0]) for k, v in extra.items()},
        "scenarios": {k: int(next(iter(v.values())).shape[0]) for k, v in scen.items()},
    }
    base_meta = {"model_id": model_id, "adapter_id": adapter_id, "sets": str(args.sets),
                 "template": template, "layers_read": layers, "n_layers": n_layers,
                 "per_category": per_category, "folds": args.folds, "var_fraction": args.var_fraction}

    def write(axis: dict, ok: bool, target_dir: Path, gate_block: dict, extra_meta: dict) -> None:
        lb = live.get(axis["name"])
        report = {"axis": ax.report_row(axis), "gates": gate_block, "written": bool(ok or args.force),
                  "forced": bool(args.force and not ok),
                  "format_offsets": offsets.get(axis["name"]),
                  "live_baseline": ({"source": live_source.get(axis["name"], "framed_neutral"),
                                     "mean": lb[0], "std": lb[1],
                                     "reply_tokens": int(n_reply_tokens)} if lb else None),
                  **base_meta, **extra_meta}
        (target_dir / f"{axis['name']}.report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        if ok or args.force:
            p = ax.save_axis(target_dir / f"{axis['name']}.npz", axis, model_id=model_id,
                             adapter_id=adapter_id, live=lb,
                             live_source=live_source.get(axis["name"], "framed_neutral"),
                             extra={"gates_passed": bool(ok), "forced": bool(not ok)})
            _log(f"  wrote {p}" + ("" if ok else "  (FORCED — a gate failed)"))
        else:
            _log(f"  not written: {axis['name']} failed "
                 + ", ".join(k for k, g in gate_block.items() if not g.get("ok")))

    _log("Writing:")
    pain_ok = ax.gates_pass(gates)
    write(pain, pain_ok, out, gates, {"self_other_raw": so_raw})
    write(relief, bool(relief_gates["auc"]["ok"]), out, relief_gates, {})
    write(fear, True, aux, {"auc": ax.gate_auc(fear)}, {"role": "cosine-gate helper at pain's layer"})
    write(valence, True, aux, {"auc": ax.gate_auc(valence)}, {"role": "cosine-gate helper at pain's layer"})
    _log("Done. Axes under data/axes/ are picked up by the next live chat turn (no restart); "
         "aux/ is not loaded.")
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0 if pain_ok else 2


if __name__ == "__main__":
    sys.exit(main())
