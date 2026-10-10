#!/usr/bin/env python3
"""Thought hygiene over the whole corpus, offline (``core/thought_hygiene.py``).

    cd server
    .venv/bin/python thought_hygiene_run.py --trial 30 [--seed 1]   # a review sheet, changes nothing
    .venv/bin/python thought_hygiene_run.py [--limit N]             # the retro pass: judge and cut
    .venv/bin/python thought_hygiene_run.py --dry-run               # what is waiting (GPU-free)
    .venv/bin/python thought_hygiene_run.py --tone-only [--dry-run] # the self-tone rule alone (GPU-free)

GPU, offline: the inference server must not hold the model (the watchdog job
``thought_hygiene`` stops and relaunches it). Loads the base with the adapter OFF, as the
judge runs in a Sleep run's clean base.

``--trial N`` judges a stratified sample of N thoughts (by half-month, those whose lines
echo the identity first) and writes ``data/hygiene/trials/<ts>.md`` + ``.json``: every
line, the ones the judge would cut marked ✂. Nothing in the corpus changes. The retro
pass judges every training target with a thought and no decision for the target it holds,
and writes each decision through the live sidecars (the cut target replaces the record's
``target``, the prior one kept under ``text_repairs``). The transcripts are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def stratified(cands: list, n: int, seed: int) -> list:
    """*n* thoughts across the half-months, two thirds from those the prefilter sends to the
    judge (the ones a cut is possible in), the rest from the ones it spares."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for c in cands:
        s = c["session"]
        by[(s[:6] + ("a" if int(s[6:8]) <= 15 else "b"), c["prefilter"]["judge"])].append(c)
    want_echo = max(1, (2 * n) // 3)
    out = []
    for echo, quota in ((True, want_echo), (False, n - want_echo)):
        keys = sorted(k for k in by if k[1] is echo)
        i = 0
        pools = {k: rng.sample(by[k], len(by[k])) for k in keys}
        while quota > 0 and any(pools.values()):
            k = keys[i % len(keys)]
            if pools[k]:
                out.append(pools[k].pop())
                quota -= 1
            i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trial", type=int, default=0, help="judge a stratified sample of N thoughts; change nothing")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rejudge", action="store_true", help="judge targets that already have a decision")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tone-only", action="store_true",
                    help="the self-tone rule over the corpus, no judge (GPU-free; with inference down, it writes the sidecars)")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from core import thought_hygiene as th
    from core.chat_sidecar import ChatSidecar
    from training.reflections_path import default_prompts_dir, hot_chats_dir, load_server_config

    chats = Path(hot_chats_dir())
    prompts = Path(default_prompts_dir())
    sidecar = ChatSidecar(chats)
    if args.tone_only:
        tally = th.tone_pass(sidecar, live_chats_dir=chats, dry_run=args.dry_run, say=lambda m: print(m, flush=True))
        print(f"self-tone rule{' (dry run)' if args.dry_run else ''}: {dict(tally)}", flush=True)
        return 0
    cands = th.candidates(sidecar, markers=th.block_markers(prompts), rejudge=args.rejudge)
    echo = sum(c["prefilter"]["judge"] for c in cands)
    print(f"{len(cands)} thought(s) without a decision for their target; {echo} echo the identity "
          f"(the judge reads those), {len(cands) - echo} spared", flush=True)
    if args.dry_run:
        return 0
    todo = stratified(cands, args.trial, args.seed) if args.trial else (cands[:args.limit] if args.limit else cands)
    if not todo:
        return 0

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    context = max(int(config.get("reflect_context_length") or 0), int(config.get("context_length") or 0)) or 16384
    system_prompt = th.load_prompt(prompts)
    psha = th.prompt_sha(system_prompt)

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
                  "frame": th.FRAME_VERSION, "temperature": 0.0, "context": context,
                  "runner": "trial" if args.trial else "retro"}
    run_id = f"hygiene-{time.strftime('%Y%m%d_%H%M%S')}"

    def judge(c: dict) -> dict:
        return th.judge_candidate(c, generate_fn=generate_fn, system_prompt=system_prompt,
                                  count_tokens=lambda s: backend.count_tokens(tokenizer, s), context=context,
                                  provenance=provenance, judge_key_=key if not args.trial else f"trial-{psha}",
                                  judge_run=run_id)

    if args.trial:
        items = []
        for n, c in enumerate(todo, 1):
            v = judge(c) if c["prefilter"]["judge"] else th.skip_record(c, judge_key_=f"trial-{psha}")
            d = th.decide(c, v)
            items.append((c, v, d))
            print(f"[{n:>3}/{len(todo)}] {d['outcome']:<9} {len(d['cut']):>3}/{len(c['lines']):<3} "
                  f"{float(v.get('seconds') or 0):>4.0f} s  {c['session']}#{c['exchange_index']}  "
                  f"{aj.head(v.get('why'), 80)}", flush=True)
        out = chats.parent / th.STORE_DIRNAME / "trials"
        out.mkdir(parents=True, exist_ok=True)
        stem = out / time.strftime("%Y%m%d_%H%M%S")
        stem.with_suffix(".md").write_text(th.render_sheet(items), encoding="utf-8")
        stem.with_suffix(".json").write_text(json.dumps(
            [{"session": c["session"], "exchange_index": c["exchange_index"], "lines": [ln for _i, ln in c["lines"]],
              "verdict": {k: v.get(k) for k in ("outcome", "recite", "why", "raw", "retried", "seconds")},
              "decision": {k: d[k] for k in ("outcome", "cut")}} for c, v, d in items],
            ensure_ascii=False, indent=1), encoding="utf-8")
        cut = [d for _c, _v, d in items if d["outcome"] == "cut"]
        print(f"\nsheet: {stem.with_suffix('.md')}\ncut {len(cut)} of {len(items)} thoughts, "
              f"{sum(len(d['cut']) for d in cut)} line(s); "
              + ", ".join(f"{k} {sum(d['outcome'] == k for _c, _v, d in items)}"
                          for k in ("none", "over_cut", "refused", "malformed", "truncated", "error")), flush=True)
        return 0

    tally = th.run(sidecar=sidecar, judge=judge, key=key, prompts_dir=prompts, live_chats_dir=chats,
                   limit=args.limit, rejudge=args.rejudge, say=lambda s: print(s, flush=True))
    print(f"\nretro pass {run_id}: {dict(tally)}", flush=True)
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
