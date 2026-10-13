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

Two more (2026-10-11), for what the retrieval block does to the reading:
  norag           the stored turn with the exchange's ``rag_context`` (the past-conversation
                  excerpts, stored verbatim inside the system content) cut out
  neutral_norag   the neutral turn without it
so that ``orig − norag`` is what retrieval adds to a real turn's reading and ``neutral −
neutral_norag`` what it makes a neutral turn read as (``--rag-summary`` reads them on the
absolute scale of ``data/axes/step0_anchors.json``, GPU-free). ``--conditions`` picks a subset.

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
RAG_CONDITIONS = ("orig", "norag", "neutral", "neutral_norag")
ALL_CONDITIONS = CONDITIONS + ("norag", "neutral_norag")


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
    if condition in ("norag", "neutral_norag"):
        rag = str((doc["exchanges"][index] or {}).get("rag_context") or "")
        if system and rag and rag in system[0]["content"]:
            system = [{**system[0], "content": system[0]["content"].replace(rag, "")}]
        content = turn["content"] if condition == "norag" else NEUTRAL_TURN
        return system + conv[1:-1] + [{**turn, "content": content}]
    raise ValueError(condition)


def rag_summary(report: dict, anchors_path: Path) -> list:
    """What the retrieval block does to the step-0 readings, on the absolute scale
    (``core.mood.absolute``): per channel, ``orig − norag`` (on the real turn) and ``neutral −
    neutral_norag`` (on a neutral turn), median / IQR / share with |Δ| ≥ 0.25, beside the turn's
    own effect without retrieval (``norag − neutral_norag``)."""
    import numpy as np
    from core import mood
    anchors = json.loads(Path(anchors_path).read_text(encoding="utf-8"))
    lines = [f"Retrieval's effect on the step-0 readings — absolute scale ({Path(anchors_path).name}: "
             "0 = a personal turn with no affect, 1 = a plain warm / harsh / playful line)"]
    for w, rows in (report.get("readings") or {}).items():
        vals: dict = {}
        for target, conds in rows.items():
            ab = {c: mood.absolute({"axes": {k: [v] for k, v in conds[c].items() if k != "prompt_tokens"}}, anchors)
                  for c in RAG_CONDITIONS if c in conds}
            if len(ab) < 4:
                continue
            for ch in ab["orig"]:
                vals.setdefault(ch, {"rag_turn": [], "rag_neutral": [], "turn": [], "orig": []})
                vals[ch]["rag_turn"].append(ab["orig"][ch] - ab["norag"][ch])
                vals[ch]["rag_neutral"].append(ab["neutral"][ch] - ab["neutral_norag"][ch])
                vals[ch]["turn"].append(ab["norag"][ch] - ab["neutral_norag"][ch])
                vals[ch]["orig"].append(ab["orig"][ch])
        lines.append(f"\n  weights {w}, {len(next(iter(vals.values()))['orig']) if vals else 0} exchanges")
        lines.append(f"  {'channel':<10} {'reading':>16}   {'retrieval on the turn':>30}   {'retrieval on a neutral turn':>30}   {'the turn itself':>22}")
        for ch, v in vals.items():
            def q(x):
                x = np.array(x)
                return f"{np.median(x):+.2f} [{np.percentile(x, 25):+.2f},{np.percentile(x, 75):+.2f}]"
            share = lambda x: f"{np.mean(np.abs(np.array(x)) >= 0.25):.0%}"
            lines.append(f"  {ch:<10} {q(v['orig']):>16}   {q(v['rag_turn']):>22} ≥.25 {share(v['rag_turn']):>4}   "
                         f"{q(v['rag_neutral']):>22} ≥.25 {share(v['rag_neutral']):>4}   {q(v['turn']):>22}")
    return lines


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


def reminders_run(args) -> int:
    """``--reminders``: each target's retrieval block parsed and scored (core/reminders.py) under
    the adapter, the items and what they would add to the turn's mood input printed and
    written beside the reports."""
    import numpy as np
    from core import hidden_capture as hc
    from core import reminders as rm
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from training.reflections_path import load_server_config
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter = str(Path(args.adapter).resolve()) if args.adapter else str(config.get("adapter_id") or "")
    anchors = json.loads((args.axes / "step0_anchors.json").read_text(encoding="utf-8"))
    backend = UnslothBackend()
    model, tokenizer = backend.load(model_id, args.context, adapter or None)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    axes = hc.load_axes(args.axes, model_id)
    system = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")
    out, inputs = [], []
    for t in [x.strip() for x in args.targets.split(",") if x.strip()]:
        stem, _, idx = t.partition("#")
        doc = json.loads((args.chats / f"{stem}.json").read_text(encoding="utf-8"))
        ex = doc["exchanges"][int(idx)]
        items = rm.parse(str(ex.get("rag_context") or ""))
        t0 = time.monotonic()
        scored = rm.score(model, tokenizer, model_id, items, axes, anchors, system=system)
        inp = rm.reminder_input(scored)
        inputs.append(inp)
        out.append({"target": t, "user": " ".join(str(ex.get("user_prompt") or "").split())[:80], "items": scored, "input": inp})
        _log(f"  {t:<24} {len(scored)} items {time.monotonic() - t0:4.1f} s  input " +
             " ".join(f"{c} {inp.get(c, 0):+.2f}" for c in ("warmth", "harm", "amusement")))
        try:
            backend.trim_memory()
        except Exception:
            pass
    for side in ("person", "self"):
        vals = [it["values"] for r in out for it in r["items"] if it["side"] == side]
        if vals:
            _log(f"  {side}: {len(vals)} items; " + "; ".join(
                f"{c} median {np.median([v[c] for v in vals if c in v]):+.2f} p90 {np.percentile([v[c] for v in vals if c in v], 90):+.2f}"
                f" max {max(v[c] for v in vals if c in v):+.2f}" for c in ("warmth", "harm", "amusement")))
    _log("  per-turn reminder input: " + "; ".join(
        f"{c} median {np.median([i.get(c, 0) for i in inputs]):+.2f} p90 {np.percentile([i.get(c, 0) for i in inputs], 90):+.2f}"
        f" max {max(i.get(c, 0) for i in inputs):+.2f}" for c in ("warmth", "harm", "amusement")))
    dst = args.out or (args.axes / "controls" / f"reminders-{datetime.now():%Y%m%d_%H%M%S}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"Report: {dst}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default="", help="comma-separated <stem>#<exchange>")
    ap.add_argument("--weights", default="adapter,base", help="adapter and/or base")
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--axes", type=Path, default=_SERVER_DIR / "data" / "axes")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="reprint a saved report, GPU-free")
    ap.add_argument("--conditions", default="", help="a subset of the conditions, comma-separated (e.g. "
                    + ",".join(RAG_CONDITIONS) + ")")
    ap.add_argument("--adapter", default="", help="this adapter dir instead of the config's")
    ap.add_argument("--reminders", action="store_true",
                    help="score each target's retrieved excerpts as reminders (core/reminders.py) on the "
                         "framed absolute scale of data/axes/step0_anchors.json; GPU, adapter weights")
    ap.add_argument("--rag-summary", type=Path, default=None,
                    help="with --report: retrieval's effect on the absolute scale of this anchors file (GPU-free)")
    ap.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.report and args.rag_summary:
        print("\n".join(rag_summary(json.loads(args.report.read_text(encoding="utf-8")), args.rag_summary)))
        return 0
    if args.report:
        print_report(json.loads(args.report.read_text(encoding="utf-8")))
        return 0
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip()) or CONDITIONS
    bad = [c for c in conditions if c not in ALL_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad}")

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
                   "--context", str(args.context), "--out", str(part), "--quiet",
                   "--conditions", ",".join(conditions)] + (["--adapter", args.adapter] if args.adapter else [])
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
    if args.reminders:
        return reminders_run(args)

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
    adapter_id = str(Path(args.adapter).resolve()) if args.adapter else str(config.get("adapter_id") or "")
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

    report = {"ts": datetime.now().isoformat(timespec="seconds"), "model_id": model_id, "conditions": list(conditions),
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
            for cond in conditions:
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
                for c in conditions if c in rows_out[name]))
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
