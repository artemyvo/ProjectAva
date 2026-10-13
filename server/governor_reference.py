#!/usr/bin/env python3
"""The habit governor's reference — AVA_REWARD_LOOP.md §4.11, PERSONA_REDESIGN.md §6 E2.

    cd server
    .venv/bin/python governor_reference.py [--adapter DIR] [--limit N] [--draw D] [--dry-run]

GPU, offline: run with inference down (watchdog job `governor_reference`). The base (or
`--adapter DIR`) answers every turn of the measurement's frozen live sample
(`data/measure/live_sample.json`) the way live chat answered it:

  * the conversation as the live turn built it (`hidden_capture.replay_conversation`: the
    exchange's stored system content — standing prompt, portraits, RAG block — the earlier
    exchanges CoT-stripped, the user turn), the family's chat template and think prefill;
  * the turn's own sampling (`generation_params`: temperature, top_p, max_new_tokens
    setting against the chat context), and the live chat's guards from server_config.json
    (min_p floor, repetition penalty, degeneration halt, stop-on-repeat, the anti-copy guard
    on the previous reply, the minimum thought, the stray-Latin guard).

One `reply` record per turn to `data/measure/governor_reference.jsonl` ({key, weights,
draw, sample_sha, cot, answer, outcome, sampling, n_in, seconds, …}). Resumable: a turn
already answered by these weights on this sample is skipped. Nothing else is written;
`python -m core.governor --reference` reads the store (GPU-free).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _quant(base_quant: str) -> tuple:
    """(load_in_4bit, load_in_8bit) for a `base_quant` label, as the server maps it."""
    q = (base_quant or "").strip().lower().replace("-", "").replace("_", "")
    if q in ("16bit", "16", "bf16", "fp16", "full", "none"):
        return (False, False)
    if q in ("8bit", "8", "int8"):
        return (False, True)
    if q in ("4bit", "4", "int4", "nf4"):
        return (True, False)
    return (None, None)


def _resolve_max_new_tokens(setting, available: int) -> int:
    """`generation._resolve_max_new_tokens` (an int, or a percentage of what is left)."""
    setting = str(setting).strip()
    if available <= 1:
        return 1
    try:
        if setting.endswith("%"):
            pct = float(setting[:-1].strip())
            if pct <= 0:
                raise ValueError
            return max(1, min(available, int(available * pct / 100.0)))
        return max(1, min(available, int(setting)))
    except (TypeError, ValueError):
        return max(1, min(available, int(available * 0.75)))


def chat_guards(config: dict) -> dict:
    """The live chat's guard kwargs for `stream_generate`, as `server.main()` builds them."""
    rp = config.get("chat_repetition_penalty")
    try:
        rp = float(rp) if rp is not None else None
    except (TypeError, ValueError):
        rp = None
    mp = config.get("chat_min_p", 0.02)
    try:
        mp = float(mp) if mp is not None else None
    except (TypeError, ValueError):
        mp = None
    kw = {"stop_on_repeat": True, "repetition_penalty": rp if (rp is not None and rp > 1.0) else None,
          "min_p": mp if (mp is not None and mp > 0.0) else None,
          "degen_stop": bool(config.get("chat_degen_guard", True))}
    for ck, ak in (("window", "degen_window"), ("min_gen", "degen_min_gen"),
                   ("distinct_ratio", "degen_distinct"), ("top_freq", "degen_top_freq")):
        v = (config.get("chat_degen") or {}).get(ck)
        if v is not None:
            kw[ak] = v
    return kw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", default="", help="an adapter dir (or its name under models/); default: the base")
    ap.add_argument("--limit", type=int, default=0, help="stop after N turns (0 = the whole sample)")
    ap.add_argument("--draw", type=int, default=1, help="which draw this is (a second draw re-answers every turn)")
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--dry-run", action="store_true", help="count what would be answered, no model")
    args = ap.parse_args(argv)

    from core import adapter_measure as am
    from core import governor as gv
    from core import hidden_capture as hc
    from core import model_family
    from training.reflections_path import load_server_config

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    if not model_id:
        raise SystemExit("model_id not set in server_config.json")
    adapter_id = ""
    if args.adapter:
        cand = Path(args.adapter)
        if not cand.exists():
            cand = _SERVER_DIR / "models" / args.adapter
        if not cand.exists():
            raise SystemExit(f"adapter not found: {args.adapter}")
        adapter_id = str(cand.resolve())
    weights = f"adapter:{Path(adapter_id).name}" if adapter_id else "base"
    measure_dir = _SERVER_DIR / "data" / "measure"
    sample = am.load_sample(measure_dir)
    if sample is None:
        raise SystemExit(f"no frozen live sample in {measure_dir}")
    context = int(config.get("context_length") or 8192)
    load_context = max(context, int(config.get("reflect_context_length") or 0))
    default_setting = str(config.get("max_new_tokens") or "75%")
    fallback = hc.persona_empty_system(_SERVER_DIR / "inference" / "prompts")

    done = gv.done_keys(gv.read_store(measure_dir), weights=weights, sample_sha=sample["sha"], draw=args.draw)
    todo = [t for t in sample["turns"] if t["key"] not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    _log(f"{weights}: sample {sample['sha']} of {len(sample['turns'])} turns, {len(done)} answered "
         f"(draw {args.draw}), {len(todo)} to answer")
    if args.dry_run or not todo:
        return 0

    # ── model ───────────────────────────────────────────────────────────
    from core.alloc_guard import ensure_expandable_segments
    from core.chat_logger import ChatLogger
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template

    load_4, load_8 = _quant(str(config.get("base_quant") or ""))
    _log(f"Loading {model_id}" + (f" + adapter {weights}" if adapter_id else " (base, adapter off)")
         + f" at context {load_context} …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    backend.stray_latin_guard = bool((config.get("stray_latin") or {}).get("guard", True))
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, load_context, adapter_id or None,
                                    load_in_4bit=load_4, load_in_8bit=load_8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    try:
        model.eval()
    except Exception:
        pass
    _log(f"Loaded in {time.monotonic() - t0:.0f} s. {backend.memory_status()}")
    fam = model_family.family_for(model_id)
    guards = chat_guards(config)
    _log(f"guards: {guards}")

    # ── answer ──────────────────────────────────────────────────────────
    docs: dict = {}
    written = failed = 0
    t_start = time.monotonic()
    for n, t in enumerate(todo, 1):
        key = t["key"]
        stem, _, idx = key.rpartition("#")
        rec = {"key": key, "weights": weights, "draw": args.draw, "sample_sha": sample["sha"], "model_id": model_id,
               "adapter_id": adapter_id, "base_quant": str(config.get("base_quant") or ""), "outcome": None,
               "cot": "", "answer": "", "error": None}
        t1 = time.monotonic()
        try:
            if stem not in docs:
                docs[stem] = json.loads((args.chats / f"{stem}.json").read_text(encoding="utf-8"))
            doc = docs[stem]
            ex = doc["exchanges"][int(idx)]
            conv, framing = hc.replay_conversation(doc, int(idx), fallback)
            gp = ex.get("generation_params") or {}
            temperature = float(gp.get("temperature") if gp.get("temperature") is not None else 1.0)
            top_p = float(gp.get("top_p") if gp.get("top_p") is not None else 0.95)
            prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
            n_in = backend.count_tokens(tokenizer, prompt)
            rec.update(framing=framing, n_in=n_in)
            if n_in >= context:
                rec.update(outcome="too_long")
                gv.append(measure_dir, rec)
                _log(f"  [{n}/{len(todo)}] {key}: {n_in} tokens ≥ context {context}, skipped")
                continue
            max_new = _resolve_max_new_tokens(gp.get("max_new_tokens_setting") or default_setting, context - n_in)
            rec["sampling"] = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new,
                               **{k: v for k, v in guards.items() if k in ("min_p", "repetition_penalty")}}
            no_copy = next((str(m.get("content") or "") for m in reversed(conv) if m.get("role") == "assistant"), "")
            parts = []
            gen = backend.stream_generate(model, tokenizer, prompt, max_new_tokens=max_new, context_length=context,
                                          temperature=temperature, top_p=top_p,
                                          min_think_tokens=fam.min_think_tokens, no_copy_text=no_copy, **guards)
            try:
                for chunk in gen:
                    parts.append(chunk)
            finally:
                gen.close()
            norm = fam.normalize_cot("".join(parts))
            cot, answer = ChatLogger._parse_cot(norm)
            rec.update(cot=cot.strip(), answer=answer.strip(),
                       truncated=bool(getattr(backend, "last_generation_truncated", False)),
                       outcome="ok" if answer.strip() else "empty")
            written += 1
        except Exception as e:  # noqa: BLE001 — one turn, not the run
            failed += 1
            rec.update(outcome="error", error=f"{type(e).__name__}: {e}")
            try:
                backend.trim_memory()
            except Exception:
                pass
        rec["seconds"] = round(time.monotonic() - t1, 1)
        gv.append(measure_dir, rec)
        if n % 5 == 0:
            try:
                backend.trim_memory()
            except Exception:
                pass
        el = time.monotonic() - t_start
        _log(f"  [{n}/{len(todo)}] {key}: {rec['outcome']} {rec['seconds']:.0f} s "
             f"(eta {el / n * (len(todo) - n) / 60:.0f} min)")
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    _log(f"Done: answered {written}, failed {failed}.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
