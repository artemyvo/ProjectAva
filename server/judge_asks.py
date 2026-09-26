#!/usr/bin/env python3
"""Judge the questions Ava has closed, with the frozen base — AVA_REWARD_LOOP.md §3.4, retro.

    cd server
    .venv/bin/python judge_asks.py [--dry-run] [--limit N] [--path answered,self,reading]
                                   [--kind meta,user,search] [--key KEY ...] [--thinking]
                                   [--no-retry] [--rejudge] [--context N] [--temperature 0]
    cd server/inference && ../.venv/bin/python -m core.ask_judge [--all] [--json]   # report

GPU, offline: run with the inference server DOWN (it holds the model). Loads the base with
the adapter OFF — always. The judge is the frozen critic (P2 of the note): there is no flag
that puts the adapter under it.

Which closures. Every ``evict reason:resolved`` of an ``[ask]`` in ``rag_memory.jsonl``,
joined to its ask and read beside the source it was closed on (``core.ask_judge``: the
conversation for a closure the person answered, the fetched text for a lookup / TIL
reading, nothing for an outreach self-closure). A closure whose source is gone — a deleted
chat, a multi-article lookup filed under the bare ``lookup`` — is recorded ``ungrounded``
and not judged. Closures are taken in a round-robin over (path, kind), so ``--limit 12``
is a smoke test that touches every kind of closure.

**Run the smoke test first.** The judge is the instruct base with thinking off, and this
corpus — Ava's questions about her own mind, conversations that can be intimate, war news
— is where its safety reflex fires. An answer with no verdict is retried once with
thinking on; one that still has none is filed ``refused`` / ``malformed``, never as a
verdict; a verdict carrying the reflex's vocabulary is flagged ``reflex``. The table at
the end lists both. If the smoke test shows mostly refusals or flags, ``--thinking`` runs
every closure with thinking on (a separate judge key, so the two runs sit side by side).

Resumable: a closure already judged under the same judge key (prompt + frame + weights +
quant + temperature + thinking/retry policy) is skipped, so a run cut short continues;
``--rejudge`` judges it again, and the report reads the newest record per closure.

Writes only ``server/data/asks/closure_verdicts.jsonl``, which nothing on Ava's side reads.
Changes no ask, no persona record, no training row.
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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _quant_flags(base_quant: str) -> tuple:
    """``server._resolve_base_quant``: the box's precision, so the judge's weights are the
    ones the server runs (unknown ⇒ the backend's historical 4-bit default)."""
    q = (base_quant or "").strip().lower().replace("-", "").replace("_", "")
    if q in ("16bit", "16", "bf16", "fp16", "full", "none"):
        return (False, False)
    if q in ("8bit", "8", "int8"):
        return (False, True)
    if q in ("4bit", "4", "int4", "nf4"):
        return (True, False)
    return (None, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list the closures and what grounds them; no model")
    ap.add_argument("--limit", type=int, default=0, help="judge at most N closures (0 = all)")
    ap.add_argument("--path", default="", help="comma list of answered,consolidation,self,reading")
    ap.add_argument("--kind", default="", help="comma list of meta,user,search")
    ap.add_argument("--key", action="append", default=[],
                    help="only this ask key (repeatable; a prefix is enough)")
    ap.add_argument("--thinking", action="store_true",
                    help="thinking on for every closure (a separate judge key)")
    ap.add_argument("--no-retry", action="store_true",
                    help="no thinking-on retry for an answer without a verdict")
    ap.add_argument("--rejudge", action="store_true",
                    help="judge closures this judge key has already judged")
    ap.add_argument("--temperature", type=float, default=0.0, help="judge temperature (§3.4: 0–0.2)")
    ap.add_argument("--context", type=int, default=0,
                    help="load context (default: the box's reflect/chat context, max of the two)")
    ap.add_argument("--memory", type=Path, default=None, help="dir holding rag_memory.jsonl")
    ap.add_argument("--chats", type=Path, default=None, help="the transcripts dir")
    ap.add_argument("--snippets", type=Path, default=None, help="the TIL snippets tree")
    ap.add_argument("--prompts", type=Path, default=None, help="the prompts dir")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from training.reflections_path import (default_prompts_dir, hot_chats_dir, load_server_config,
                                           memory_dir, til_snippets_dir)

    mem = args.memory or memory_dir()
    chats = args.chats or hot_chats_dir()
    snippets = args.snippets or til_snippets_dir()
    prompts = args.prompts or default_prompts_dir()
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    base_quant = str(config.get("base_quant") or "")
    context = int(args.context or max(int(config.get("reflect_context_length") or 0),
                                      int(config.get("context_length") or 0)) or 16384)

    # ── plan ────────────────────────────────────────────────────────────
    paths = {p.strip() for p in args.path.split(",") if p.strip()}
    kinds = {k.strip() for k in args.kind.split(",") if k.strip()}
    if paths - set(aj.PATHS):
        raise SystemExit(f"--path: unknown {sorted(paths - set(aj.PATHS))}; choose from {aj.PATHS}")
    closures, grounding = aj.load_closures(mem, chats, snippets, paths=paths, kinds=kinds,
                                           keys=args.key)
    # A closure the cut-resolution repair replaced is history; its replacement is judged.
    closures = [c for c in closures if not c.get("repaired")]
    try:
        system_prompt = aj.load_prompt(prompts)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    psha = aj.prompt_sha(system_prompt)
    store = aj.store_path(chats)

    def _key(quant: str) -> str:
        return aj.judge_key(prompt_sha_=psha, model_id=model_id, base_quant=quant,
                            temperature=args.temperature, thinking=args.thinking,
                            retry=not args.no_retry)

    def _todo(key: str) -> list:
        done = set() if args.rejudge else aj.judged_ids(aj.read_jsonl(store), key)
        todo = [c for c in closures if c["closure_id"] not in done]
        return todo[:args.limit] if args.limit > 0 else todo

    quant = aj.quant_label(base_quant)
    jkey = _key(quant)
    todo = _todo(jkey)
    _log(f"{len(closures)} closures selected from {Path(mem) / 'rag_memory.jsonl'}; "
         f"{len(todo)} to judge now under judge {jkey}.")
    for line in aj.render_census(aj.census(closures, grounding)):
        _log("  " + line)
    # Sources longer than the window will be clipped (head + tail); estimated here at the
    # conservative ~2.7 chars/token, measured with the tokenizer during the run.
    est_budget = context - 1500 - aj.THINKING_RETRY_MAX_NEW_TOKENS - 256
    long = sum(1 for c in closures if len(grounding[c["closure_id"]].get("text") or "") / 2.7 > est_budget)
    _log(f"  judge: {model_id or '?'} {quant or '(default quant)'}, adapter OFF · prompt {psha} "
         f"· T={args.temperature} · thinking {'on' if args.thinking else 'off'}"
         f"{'' if args.no_retry or args.thinking else ', retried with thinking on no verdict'} "
         f"· context {context} (≈{long} source(s) will be clipped)")
    _log(f"  store: {store}")
    if args.dry_run or not todo:
        return 0
    if not model_id:
        raise SystemExit("model_id not set in server_config.json")

    # ── model ───────────────────────────────────────────────────────────
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template

    l4, l8 = _quant_flags(base_quant)
    _log(f"Loading {model_id} (base, adapter off, {quant or 'default quant'}) at context {context} …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, context, None, load_in_4bit=l4, load_in_8bit=l8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    try:
        model.eval()
    except Exception:
        pass
    _log(f"Loaded in {time.monotonic() - t0:.0f} s. {backend.memory_status()}")
    # The key names the weights actually loaded (the in-server run reads them the same way).
    loaded = aj.effective_quant(model, quant)
    if loaded != quant:
        _log(f"  loaded at {loaded}, not {quant or 'the default'}: judge key recomputed")
        quant, jkey = loaded, _key(loaded)
        todo = _todo(jkey)
    generate_fn = aj.make_backend_generate_fn(backend, lambda: model, lambda: tokenizer,
                                              model_id, context)
    provenance = {"model_id": model_id, "base_quant": quant or None, "adapter": None,
                  "prompt_sha": psha, "frame": aj.FRAME_VERSION, "temperature": args.temperature,
                  "context": context, "thinking_first": bool(args.thinking),
                  "retry": not args.no_retry, "runner": "cli"}

    # ── judge ───────────────────────────────────────────────────────────
    t_start = time.monotonic()

    def _on_record(n: int, total: int, rec: dict) -> None:
        eta = (time.monotonic() - t_start) / n * (total - n)
        _log(aj.progress_line(n, total, rec, eta_s=eta))

    try:
        tally = aj.judge_many(
            todo, grounding, generate_fn=generate_fn, system_prompt=system_prompt,
            count_tokens=lambda s: backend.count_tokens(tokenizer, s), context=context,
            provenance=provenance, store=store, judge_run=time.strftime("%Y%m%d_%H%M%S"),
            judge_key_=jkey, temperature=args.temperature, thinking=args.thinking,
            retry=not args.no_retry, on_record=_on_record)
        _log(f"\nThis run: {dict(tally)} in {(time.monotonic() - t_start) / 60:.1f} min.\n")
    except KeyboardInterrupt:
        _log("\nInterrupted — what was judged is in the store; run again to continue.\n")
    latest = aj.latest_per_closure(aj.read_jsonl(store), jkey)
    _log(aj.render(aj.summarize(latest), records=latest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
