#!/usr/bin/env python3
"""Portrait hygiene, offline (``core/portrait_hygiene.py``): the trial, or the retro verdicts.

    cd server
    .venv/bin/python portrait_hygiene_run.py --person NAME --trial 40 [--dry-run] [--seed 1]
    .venv/bin/python portrait_hygiene_run.py --judge-only [--person NAME] [--limit N]

GPU, offline: the inference server must not hold the model (the watchdog job
``portrait_hygiene`` stops and relaunches it). Loads the base with the adapter OFF.

Samples *N* of her live ``[impression]`` readings of NAME, spread over the months they were
written in; for each, the frozen base reads it against NAME's own messages in the chat it
came from (SUPPORT / PURPOSE / WHY), and the code scores how much of its wording comes from
her description of herself against NAME's own words. Verdicts go to
``data/hygiene/portrait.jsonl``, the review sheet to ``data/hygiene/trials/portrait_<ts>.md``.
Nothing in Ava's stores is changed. ``--dry-run`` writes the sheet with the code scores
only (GPU-free). ``--memory`` / ``--chats`` read another box's copy of the stores.

``--judge-only`` judges every reading (of NAME, or of everyone) with no verdict under the
judge's key and writes the verdicts only; a Sleep run applies them (verdicts from another
box are merged into ``data/hygiene/portrait.jsonl`` first: same key, same base and prompt).
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

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--person", default="")
    ap.add_argument("--judge-only", action="store_true", help="verdicts for every unread reading, nothing applied")
    ap.add_argument("--limit", type=int, default=0, help="--judge-only: at most N readings")
    ap.add_argument("--trial", type=int, default=40, help="how many readings to sample")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--memory", type=Path, default=None, help="the op-log's directory (default: the live one)")
    ap.add_argument("--chats", type=Path, default=None, help="the transcripts' directory (default: the live one)")
    ap.add_argument("--dry-run", action="store_true", help="the sheet with the code scores only, no model")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from core import portrait_hygiene as ph
    from core import register_monitor as rm
    from training.reflections_path import default_prompts_dir, hot_chats_dir, load_server_config, memory_dir

    chats = args.chats or Path(hot_chats_dir())
    mem = args.memory or Path(memory_dir())
    live_chats = Path(hot_chats_dir())
    if args.judge_only:
        return _judge_only(args, chats=chats, mem=mem, live_chats=live_chats)
    if not args.person:
        raise SystemExit("--trial needs --person")
    items = ph.impressions([mem], args.person)
    live = [i for i in items.values() if i["state"] == "live"]
    sample = ph.trial_sample(items, args.trial, seed=args.seed)
    print(f"{len(live)} live readings of {args.person}; sampled {len(sample)} over "
          f"{sorted({i['month'] for i in sample})}", flush=True)
    self_pairs = ph.pair_set(ph.self_statements([mem]))
    person_pairs = ph.pair_set(ph.all_person_turns(chats, args.person))
    print(f"her self-description: {len(self_pairs)} word pairs; {args.person}'s own words: {len(person_pairs)}",
          flush=True)

    rows, docs = [], {}
    for it in sample:
        if it["session"] not in docs:
            try:
                docs[it["session"]] = json.loads((chats / it["session"]).read_text(encoding="utf-8"))
            except Exception:
                docs[it["session"]] = {}
        turns = ph.person_turns(docs[it["session"]], args.person)
        text, shown, total = ph.messages_block(turns)
        rows.append({"item": it, "turns": turns, "messages": text, "shown": shown, "total": total,
                     "proj": ph.projection(it["content"], self_pairs, person_pairs),
                     "motive": rm.motive_sentence(it["content"], (args.person,)) is not None,
                     "v": {"outcome": "not judged"}})

    key = "dry-run"
    if not args.dry_run and rows:
        config = load_server_config() or {}
        model_id = str(config.get("model_id") or "")
        context = max(int(config.get("reflect_context_length") or 0), int(config.get("context_length") or 0)) or 16384
        prompts = Path(default_prompts_dir())
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
        print(f"loaded {model_id} (adapter off) in {time.monotonic() - t0:.0f} s. {backend.memory_status()}",
              flush=True)
        quant = aj.effective_quant(model, aj.quant_label(str(config.get("base_quant") or "")))
        key = aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant, temperature=0.0, thinking=False,
                           retry=True)
        generate_fn = aj.make_backend_generate_fn(backend, lambda: model, lambda: tokenizer, model_id, context)
        provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None, "prompt_sha": psha,
                      "frame": ph.FRAME_VERSION, "temperature": 0.0, "context": context, "runner": "trial"}
        run_id = f"portrait-hygiene-{time.strftime('%Y%m%d_%H%M%S')}"
        store = ph.store_path(live_chats)
        for n, r in enumerate(rows, 1):
            v = ph.judge_reading(r["item"], messages=r["messages"], generate_fn=generate_fn,
                                 system_prompt=system_prompt, provenance=provenance, judge_key_=key,
                                 judge_run=run_id, extra={"person": args.person, "projection": r["proj"],
                                                          "motive": r["motive"]})
            aj.append_record(store, v)
            r["v"] = v
            print(f"[{n:>3}/{len(rows)}] {str(v.get('support') or v.get('outcome')):<10} "
                  f"{'PURPOSE' if v.get('purpose') else '       '} {float(v.get('seconds') or 0):>4.0f} s  "
                  f"{aj.head(r['item']['content'], 70)}  — {aj.head(v.get('why'), 70)}", flush=True)
        try:
            backend.release(model, tokenizer)
        except Exception:
            pass

    out = ph.trials_dir(live_chats) / f"portrait_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ph.render_sheet(rows, person=args.person, key=key), encoding="utf-8")
    print(f"sheet: {out}", flush=True)
    return 0


def _load_judge(run_id: str, runner: str):
    """``(key, judge, release)``: the base loaded with the adapter OFF and portrait hygiene's judge on it."""
    from types import SimpleNamespace
    from core import ask_judge as aj
    from core import portrait_hygiene as ph
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from judge_asks import _quant_flags
    from training.reflections_path import default_prompts_dir, load_server_config

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    context = max(int(config.get("reflect_context_length") or 0), int(config.get("context_length") or 0)) or 16384
    l4, l8 = _quant_flags(str(config.get("base_quant") or ""))
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, context, None, load_in_4bit=l4, load_in_8bit=l8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    print(f"loaded {model_id} (adapter off) in {time.monotonic() - t0:.0f} s. {backend.memory_status()}", flush=True)
    rt = SimpleNamespace(model=model, tokenizer=tokenizer, model_id=model_id, context_length=context,
                         reflect_context_length=context, base_quant=str(config.get("base_quant") or ""))
    key, judge = ph.make_judge_fn(backend=backend, runtime=rt, prompts_dir=Path(default_prompts_dir()),
                                  run_id=run_id, runner=runner)

    def release():
        try:
            backend.release(model, tokenizer)
        except Exception:
            pass
    return key, judge, release


def _judge_only(args, *, chats: Path, mem: Path, live_chats: Path) -> int:
    from core import ask_judge as aj
    from core import portrait_hygiene as ph
    items = ph.impressions([mem], args.person or None)
    store = ph.store_path(live_chats)
    waiting = ph.todo(items, aj.read_jsonl(store), None)
    print(f"{len(items)} readings{' of ' + args.person if args.person else ''}; "
          f"{sum(i['state'] == 'live' for i in items.values())} live; store {store}", flush=True)
    if args.dry_run:
        print(f"(--dry-run: {len(waiting)} have no verdict under any key)", flush=True)
        return 0
    run_id = f"portrait-hygiene-{time.strftime('%Y%m%d_%H%M%S')}"
    key, judge, release = _load_judge(run_id, "retro")
    t0 = time.monotonic()
    tally = ph.judge_all(items, judge=judge, key=key, store=store, chats_dir=chats,
                         self_pairs=ph.pair_set(ph.self_statements([mem])), limit=args.limit,
                         say=lambda s: print(s, flush=True))
    release()
    print(f"\njudged {run_id} under {key} in {time.monotonic() - t0:.0f} s: {dict(tally)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
