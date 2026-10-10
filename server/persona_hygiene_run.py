#!/usr/bin/env python3
"""Persona hygiene over the whole persona store, offline (``core/persona_hygiene.py``).

    cd server
    .venv/bin/python persona_hygiene_run.py --dry-run          # what waits (GPU-free)
    .venv/bin/python persona_hygiene_run.py --judge-only       # verdicts to the store, nothing applied
    .venv/bin/python persona_hygiene_run.py --apply-only       # the standing verdicts applied (GPU-free)
    .venv/bin/python persona_hygiene_run.py [--limit N]        # the retro pass: judge, then apply

GPU for judging, offline: the inference server must not hold the model (the watchdog job
``persona_hygiene`` stops and relaunches it). Loads the base with the adapter OFF, as the
judge runs in a Sleep run's clean base.

The judge reads every ``[persona]`` statement that is live (or withheld by this gate) with
no verdict under its key. ``--judge-only`` stops there: Ava's stores are untouched, and the
review sheet (``python -m core.persona_hygiene --sheet``) reads the verdicts. Applying
supersedes each live statement read as HABIT, and restores one this gate withheld that a
new judge reads as AFFIRM, through the LIVE op-log and ledger: run it with inference down.
A Sleep run applies the standing verdicts too, so verdicts merged in from another box (the
same judge key: the same prompt, base and quantization) are applied at the next Sleep.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rejudge", action="store_true", help="judge statements that already have a verdict")
    ap.add_argument("--judge-only", action="store_true", help="write verdicts, apply nothing")
    ap.add_argument("--apply-only", action="store_true", help="apply the standing verdicts (GPU-free)")
    ap.add_argument("--key", default="", help="--apply-only: the judge key (default: the newest in the store)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from core import persona_hygiene as ph
    from core.reflection_writer import ReflectionWriter
    from training.ledger import ConsolidationLedger
    from training.reflections_path import (consolidation_dir, default_prompts_dir, hot_chats_dir,
                                           load_server_config, memory_dir)

    chats, prompts = Path(hot_chats_dir()), Path(default_prompts_dir())
    mem, cons = Path(memory_dir()), Path(consolidation_dir())
    store = ph.store_path(chats)
    sts = ph.statements([mem])
    live = sum(s["state"] == "live" for s in sts.values())
    print(f"{live} live [persona] statement(s), {sum(s['state'] == 'withheld' for s in sts.values())} withheld "
          f"by this gate; store {store}", flush=True)

    def _apply(key: str) -> None:
        records = aj.read_jsonl(store)
        withhold, restore = ph.plan(ph.statements([mem]), records, key)
        print(f"under {key}: {len(withhold)} to withhold, {len(restore)} to restore", flush=True)
        if args.dry_run or not (withhold or restore):
            return
        run_id = f"persona-hygiene-{time.strftime('%Y%m%d_%H%M%S')}"
        tally = ph.apply(withhold, restore, writer=ReflectionWriter(mem, cons), ledger=ConsolidationLedger(cons),
                         records=records, key=key, run_id=run_id, say=lambda s: print(s, flush=True))
        print(f"\napplied {run_id}: {dict(tally)}", flush=True)

    if args.apply_only:
        key = args.key or next((r.get("judge_key") for r in reversed(aj.read_jsonl(store))
                                if r.get("record") == "verdict"), "")
        if not key:
            print("no verdicts in the store", flush=True)
            return 1
        _apply(key)
        return 0

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    if args.dry_run:
        print("(the judge key needs the loaded model: --dry-run counts every statement with no verdict yet "
              f"under any key: {sum(1 for s in ph.todo(sts, [], None))} live or withheld)", flush=True)
        return 0
    context = max(int(config.get("reflect_context_length") or 0), int(config.get("context_length") or 0)) or 16384
    system_prompt = ph.load_prompt(prompts)
    psha = ph.prompt_sha(system_prompt)

    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from judge_asks import _quant_flags

    l4, l8 = _quant_flags(str(config.get("base_quant") or ""))
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, context, None, load_in_4bit=l4, load_in_8bit=l8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    print(f"loaded {model_id} (adapter off) in {time.monotonic() - t0:.0f} s. {backend.memory_status()}", flush=True)
    quant = aj.effective_quant(model, aj.quant_label(str(config.get("base_quant") or "")))
    key = aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant, temperature=0.0, thinking=False,
                       retry=True)
    generate_fn = aj.make_backend_generate_fn(backend, lambda: model, lambda: tokenizer, model_id, context)
    provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None, "prompt_sha": psha,
                  "frame": ph.FRAME_VERSION, "temperature": 0.0, "context": context, "runner": "retro"}
    run_id = f"persona-hygiene-{time.strftime('%Y%m%d_%H%M%S')}"

    def judge(s: dict) -> dict:
        return ph.judge_statement(s, generate_fn=generate_fn, system_prompt=system_prompt, provenance=provenance,
                                  judge_key_=key, judge_run=run_id)

    tally = ph.judge_all(sts, judge=judge, key=key, store=store, limit=args.limit, rejudge=args.rejudge,
                         say=lambda s: print(s, flush=True))
    print(f"\njudged {run_id} under {key}: {dict(tally)}", flush=True)
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    if not args.judge_only:
        _apply(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
