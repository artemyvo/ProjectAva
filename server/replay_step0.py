#!/usr/bin/env python3
"""Step-0 axis readings under a swapped turn — what moved the state before her reply?

    cd server
    .venv/bin/python replay_step0.py --targets 20261005_004700#8,20261003_024540#7
                                     [--weights adapter,base] [--out FILE] [--report FILE]

GPU, offline: run with the inference server DOWN (the watchdog job `replay_step0` stops
and relaunches it). Read-only on the corpus; writes one report under
``data/axes/controls/step0-<ts>.json``.

The step-0 reading (``generation._step0_reading``) is each stored axis's projection at the
prompt's last position, the state her first token is generated from. A high one can be
the turn she is answering, how deep into the session it comes, or the text alone. For
each target exchange the prompt is rebuilt the way the live turn built it
(``hidden_capture.replay_conversation``: the stored system content, the CoT-stripped
history, the user turn, the family's template and think prefill), then read under five
conditions:

  orig            as stored — under the weights that wrote it, this is the live reading
                  (the report shows the stored value beside it as an alignment check)
  neutral         the user turn swapped for a matter-of-fact one of a few lines
  short           the user turn swapped for "Продолжай."
  shallow         the stored turn with no history (system + this turn only)
  shallow_neutral the neutral turn with no history

so that ``orig − neutral`` is the turn's effect at its depth, ``shallow − shallow_neutral``
the same turn's effect with no session behind it, and ``neutral − shallow_neutral`` the
depth's own. The system content is the one stored with the exchange, so its retrieval
block is the one retrieved for the real turn; a swapped turn keeps it. Under ``base`` the
adapter is off: a turn that moves both weights alike is read off the text.

``--report FILE`` reprints a saved report, GPU-free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

NEUTRAL_TURN = ("Я прочитал твой ответ. Давай продолжим с того места, где ты остановилась, "
                "и разберём следующий шаг.")
SHORT_TURN = "Продолжай."
CONDITIONS = ("orig", "neutral", "short", "shallow", "shallow_neutral")


def _log(msg: str) -> None:
    print(msg, flush=True)


def conversation(doc: dict, index: int, condition: str, fallback_system: str) -> list:
    """The prompt's conversation for one condition (see the module docstring)."""
    from core import hidden_capture as hc
    conv, _framing = hc.replay_conversation(doc, index, fallback_system)
    system = conv[:1] if conv and conv[0]["role"] == "system" else []
    turn = conv[-1]
    if condition == "orig":
        return conv
    if condition == "neutral":
        return conv[:-1] + [{**turn, "content": NEUTRAL_TURN}]
    if condition == "short":
        return conv[:-1] + [{**turn, "content": SHORT_TURN}]
    if condition == "shallow":
        return system + [turn]
    if condition == "shallow_neutral":
        return system + [{**turn, "content": NEUTRAL_TURN}]
    raise ValueError(condition)


def print_report(report: dict) -> None:
    axes = report.get("axes") or []
    _log(f"\nStep-0 readings (σ, the transcripts' units) — {report.get('ts')}")
    for w, rows in (report.get("readings") or {}).items():
        _log(f"\n  weights: {w}")
        for target, conds in rows.items():
            stored = (report.get("stored") or {}).get(target) or {}
            for ax in axes:
                vals = {c: conds.get(c, {}).get(ax) for c in CONDITIONS}
                if any(v is None for v in vals.values()):
                    continue
                turn = vals["orig"] - vals["neutral"]
                shallow = vals["shallow"] - vals["shallow_neutral"]
                depth = vals["neutral"] - vals["shallow_neutral"]
                live = stored.get(ax)
                _log(f"    {target:24s} {ax:6s} orig {vals['orig']:+.2f}"
                     + (f" (stored {live:+.2f})" if live is not None else "")
                     + f"  neutral {vals['neutral']:+.2f}  short {vals['short']:+.2f}"
                     f"  shallow {vals['shallow']:+.2f}  shallow_neutral {vals['shallow_neutral']:+.2f}"
                     f"  | turn {turn:+.2f}  turn@shallow {shallow:+.2f}  depth {depth:+.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default="", help="comma-separated <stem>#<exchange>")
    ap.add_argument("--weights", default="adapter,base", help="adapter and/or base")
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--axes", type=Path, default=_SERVER_DIR / "data" / "axes")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="reprint a saved report, GPU-free")
    ap.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.report:
        print_report(json.loads(args.report.read_text(encoding="utf-8")))
        return 0

    weights_list = [w.strip() for w in args.weights.split(",") if w.strip()]
    if len(weights_list) > 1:
        # One process per set of weights: a quantized model's memory is not reliably
        # returned inside the process that loaded it, and a second 31B load beside the
        # first runs out of memory on a 32 GB card. Each child writes its own report.
        import subprocess
        out = args.out or (args.axes / "controls" / f"step0-{datetime.now():%Y%m%d_%H%M%S}.json")
        merged: dict = {}
        for w in weights_list:
            part = out.with_name(out.stem + f".{w}.json")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--weights", w,
                   "--targets", args.targets, "--chats", str(args.chats), "--axes", str(args.axes),
                   "--context", str(args.context), "--out", str(part), "--quiet"]
            rc = subprocess.call(cmd)
            if rc != 0 or not part.exists():
                _log(f"the {w} pass failed (rc {rc})")
                continue
            rep = json.loads(part.read_text(encoding="utf-8"))
            part.unlink()
            if not merged:
                merged = rep
            else:
                merged["readings"].update(rep.get("readings") or {})
        if not merged:
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        print_report(merged)
        _log(f"\nReport: {out}")
        return 0 if set(merged["readings"]) == set(weights_list) else 2

    from core import hidden_capture as hc
    from core import model_family
    from training.reflections_path import load_server_config

    targets = []
    for t in [s.strip() for s in args.targets.split(",") if s.strip()]:
        stem, _, idx = t.partition("#")
        path = args.chats / (stem if stem.endswith(".json") else stem + ".json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        targets.append((t, doc, int(idx)))
    if not targets:
        raise SystemExit("--targets is empty")

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = str(config.get("adapter_id") or "")
    fam = model_family.family_for(model_id)
    fallback_system = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")

    stored = {}
    for name, doc, i in targets:
        ax = ((doc["exchanges"][i].get("tension") or {}).get("axes") or {})
        stored[name] = {k: float(v[0]) for k, v in ax.items() if v}
        stored[name]["adapter"] = Path(str(doc.get("adapter_id") or "")).name

    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template

    report = {"ts": datetime.now().isoformat(timespec="seconds"), "model_id": model_id,
              "adapter": Path(adapter_id).name if adapter_id else "", "neutral_turn": NEUTRAL_TURN,
              "short_turn": SHORT_TURN, "stored": stored, "readings": {}, "axes": []}
    ensure_expandable_segments()
    for weights in weights_list:
        adapter = adapter_id if weights == "adapter" else ""
        if weights == "adapter" and not adapter:
            _log("no adapter_id configured — skipping the adapter pass")
            continue
        _log(f"Loading {model_id}" + (f" + {Path(adapter).name}" if adapter else " (base)") + " …")
        backend = UnslothBackend()
        t0 = time.monotonic()
        model, tokenizer = backend.load(model_id, args.context, adapter or None)
        if not getattr(tokenizer, "chat_template", None):
            ensure_chat_template(tokenizer, model_name=model_id, emit=print)
        model.eval()
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        spec = hc.capture_spec(model, model_id, args.axes)
        if spec is None or not spec["axes"]:
            raise SystemExit("no axes to read")
        report["axes"] = [a.name for a in spec["axes"]]
        layers = sorted({a.layer for a in spec["axes"]})
        _log(f"Loaded in {time.monotonic() - t0:.0f} s; axes {report['axes']} at layers {layers}")
        rows_out: dict = {}
        for name, doc, i in targets:
            rows_out[name] = {}
            for cond in CONDITIONS:
                conv = conversation(doc, i, cond, fallback_system)
                prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
                ids = [int(t) for t in text_tok(prompt)["input_ids"]]
                if len(ids) > args.context:
                    _log(f"  {name} {cond}: {len(ids)} tokens > context, skipped")
                    continue
                rows = hc.forward_ids_rows(model, ids, layers)
                rows_out[name][cond] = {
                    a.name: round(float((rows[a.layer][-1].astype("float32") @ a.vector - a.mean)
                                        / (a.std or 1.0)), 4)
                    for a in spec["axes"]}
                rows_out[name][cond]["prompt_tokens"] = len(ids)
            _log(f"  {name}: " + "  ".join(
                f"{c} " + "/".join(f"{rows_out[name][c][a]:+.2f}" for a in report["axes"])
                for c in CONDITIONS if c in rows_out[name]))
            try:
                backend.trim_memory()
            except Exception:
                pass
        report["readings"][weights] = rows_out
        try:
            backend.release(model, tokenizer)
        except Exception:
            pass
        del model, tokenizer
        # The next pass loads a second copy of the base: return this one's memory first.
        import gc
        gc.collect()
        try:
            backend.reclaim()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    out = args.out or (args.axes / "controls" / f"step0-{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.quiet:
        print_report(report)
        _log(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
