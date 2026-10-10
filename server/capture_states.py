#!/usr/bin/env python3
"""Pre-reply hidden states for the capture probe — AVA_REWARD_LOOP.md §4.3, item 14.

    cd server
    .venv/bin/python capture_states.py --set corpus [--adapter DIR] [--limit N] [--dry-run]
    .venv/bin/python capture_states.py --set bench [--bench b1] [--condition as_is|persona_empty|both]
                                       [--adapter DIR] [--dry-run]

GPU, offline: run with the server's model unloaded (or inference down). For each context
it makes ONE forward pass over the prompt alone (the chat template and the family's think
prefill included) and keeps the residual stream at the prompt's last token at a sweep of
decoder layers (`core.capture_probe.sweep_layers`). That is the state the reply's first
token is generated from, the one the live capture stores as its step 0. The backfill
(`backfill_hidden.py`) does not store it: its first row sits one token later, on the
replayed reply's own first token.

Two sets of contexts:
  * `corpus`: every exchange any capture-judge key found answering pushback
    (`core.capture_probe.pushback_exchanges`), with the prompt rebuilt as the live turn
    built it (`hidden_capture.replay_conversation`, the backfill's rebuild). By default
    under the bare base, the one set of weights every transcript shares.
  * `bench`: the capture bench's frozen contexts (`core/capture_bench.py`), rendered as the
    bench rendered them: the frozen IDEAL messages, or with `--condition persona_empty`
    the system turn replaced by the original standing prompt and the undecided framing.
    That text is checked against the sha the bench's replies recorded, and the script
    refuses if the prompt files have changed since. Run it under the base or under
    `--adapter DIR`, so each arm's grades meet the states of the weights that answered.

Store: `data/capture/states/<set>~<weights>[+<condition>].npz`, an append-only zip like
the hidden sidecar, with `<key>/states.npy` ((layers, D) fp16) and `<key>/meta.npy` per
context. Resumable: a key already stored is skipped. The script writes nothing else, and
`core/capture_probe.py` reads the store.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _plan_corpus(chats: Path, verdicts: list, fallback_system: str) -> list:
    """[(key, conversation, framing)] for every pushback exchange with a transcript."""
    from core import capture_probe as cp
    from core import hidden_capture as hc
    plan, cache = [], {}
    for session, idx in cp.pushback_exchanges(verdicts):
        if session not in cache:
            try:
                cache[session] = json.loads((chats / session).read_text(encoding="utf-8"))
            except Exception:
                cache[session] = None
        doc = cache[session]
        exs = (doc or {}).get("exchanges") or []
        if not isinstance(exs, list) or not (0 <= idx < len(exs)) or not isinstance(exs[idx], dict):
            continue
        conv, framing = hc.replay_conversation(doc, idx, fallback_system)
        plan.append((cp.context_key(session, idx), conv, framing))
    return plan


def _plan_bench(records: list, bench: str, condition: str, empty_system: str) -> list:
    """[(key, messages, condition)] for the bench's frozen contexts."""
    from core import capture_bench as cb
    system = empty_system if cb.persona_empty_condition(condition) else None
    insults = cb.load_insults() if cb.message_kind(condition) != "as_is" else None
    plan = []
    for ctx in cb.contexts(records, bench):
        message, _i = cb.message_variant(ctx, condition, insults)
        plan.append((ctx["context_id"], cb.messages_for(ctx, system=system, message=message), condition))
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", choices=("corpus", "bench"), required=True)
    ap.add_argument("--bench", default="b1", help="the bench's name (with --set bench)")
    ap.add_argument("--condition", default="both",
                    help="with --set bench: as_is, persona_empty, both (those two), or an insult-study "
                         "condition (insult, insult+persona_empty, insult_only, insult_only+persona_empty), "
                         "comma-separated")
    ap.add_argument("--adapter", default="", help="an adapter dir (or its name under models/); default: the base")
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--context", type=int, default=0,
                    help="load context and the longest prompt read (default: the server's load context)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N contexts per condition (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="count what would be read, no model")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from core import capture_bench as cb
    from core import capture_judge as cj
    from core import capture_probe as cp
    from core import hidden_capture as hc
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
    weights = cp.weights_name(adapter_id)
    context = args.context or max(int(config.get("context_length") or 8192),
                                  int(config.get("reflect_context_length") or 0))
    out_dir = cp.states_dir(args.chats)
    prompts_dir = _SERVER_DIR / "inference" / "prompts"
    empty_system = hc.persona_empty_system(prompts_dir)

    # ── plan ────────────────────────────────────────────────────────────
    jobs = []        # (store path, [(key, messages, tag)])
    if args.set == "corpus":
        verdicts = cp.label_records(args.chats)
        jobs.append((cp.states_file(out_dir, "corpus", weights), _plan_corpus(args.chats, verdicts, empty_system)))
    else:
        records = aj.read_jsonl(cb.store_path(args.chats))
        if not cb.contexts(records, args.bench):
            raise SystemExit(f"no frozen contexts for bench {args.bench!r}")
        conditions = (("as_is", "persona_empty") if args.condition == "both"
                      else tuple(c.strip() for c in args.condition.split(",") if c.strip()))
        unknown = [c for c in conditions if c not in cb.ALL_CONDITIONS]
        if unknown:
            raise SystemExit(f"unknown condition(s) {unknown}; known: {list(cb.ALL_CONDITIONS)}")
        if any(cb.persona_empty_condition(c) for c in conditions):
            recorded = {r.get("system_sha") for r in records if r.get("record") == "reply"
                        and r.get("bench") == args.bench and r.get("condition") == "persona_empty"}
            recorded.discard(None)
            if recorded and _sha(empty_system) not in recorded:
                raise SystemExit(f"the persona-empty system text has changed since the bench ran "
                                 f"({_sha(empty_system)} not in {sorted(recorded)}); refusing")
        for cond in conditions:
            jobs.append((cp.states_file(out_dir, args.bench, weights, cond),
                         _plan_bench(records, args.bench, cond, empty_system)))
    todo = []
    for path, plan in jobs:
        done = cp.stored_keys(path)
        left = [p for p in plan if p[0] not in done]
        if args.limit > 0:
            left = left[:args.limit]
        _log(f"{path.name}: {len(plan)} contexts, {len(plan) - len([p for p in plan if p[0] not in done])} "
             f"stored, {len(left)} to read")
        todo.append((path, left))
    if args.dry_run or not any(left for _p, left in todo):
        return 0

    # ── model ───────────────────────────────────────────────────────────
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template

    load_4, load_8 = _quant(str(config.get("base_quant") or ""))
    _log(f"Loading {model_id}" + (f" + adapter {weights}" if adapter_id else " (base, adapter off)")
         + f" at context {context} …")
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, context, adapter_id or None,
                                    load_in_4bit=load_4, load_in_8bit=load_8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    try:
        model.eval()
    except Exception:
        pass
    layers_mod = hc.find_decoder_layers(model)
    if layers_mod is None:
        raise SystemExit("no decoder stack found")
    n_layers = len(layers_mod)
    # The stored axes' own layers too (pain 25, relief 22, the band's 34), so each state
    # can be read on every axis, not only on the ones that sit on the sweep (2026-10-06).
    axis_layers = [a.layer for a in hc.load_axes(_SERVER_DIR / "data" / "axes", model_id)]
    layers = cp.sweep_layers(n_layers, str((config.get("hidden") or {}).get("layers") or hc.DEFAULT_LAYER_SPEC),
                             extra=axis_layers)
    _log(f"Loaded in {time.monotonic() - t0:.0f} s; {n_layers} layers, reading {layers}. "
         f"{backend.memory_status()}")

    # ── read ────────────────────────────────────────────────────────────
    total_written = total_failed = 0
    for path, left in todo:
        written = failed = too_long = 0
        t_start = time.monotonic()
        for n, (key, conv, tag) in enumerate(left, 1):
            try:
                prompt, ids = cp.context_prompt(tokenizer, model_id, conv)
            except Exception as e:
                failed += 1
                _log(f"  [{n}/{len(left)}] {key}: prompt build failed: {type(e).__name__}: {e}")
                continue
            if len(ids) > context:
                too_long += 1
                _log(f"  [{n}/{len(left)}] {key}: {len(ids)} tokens > context {context}, skipped")
                continue
            try:
                mat = cp.context_state(model, ids, layers)
                meta = cp.state_meta(
                    key=key, set_name=args.set if args.set == "corpus" else args.bench,
                    condition=tag if args.set == "bench" else "as_is",
                    framing=tag if args.set == "corpus" else "ideal", model_id=model_id,
                    adapter_id=weights if adapter_id else "", weights=weights, layers=layers,
                    n_layers=n_layers, hidden_dim=int(mat.shape[1]), prompt=prompt, n_tokens=len(ids),
                    base_quant=str(config.get("base_quant") or ""), writer="capture_states")
                if cp.write_state(path, key, mat, meta):
                    written += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                _log(f"  [{n}/{len(left)}] {key}: {type(e).__name__}: {e}")
                try:
                    backend.trim_memory()
                except Exception:
                    pass
                continue
            if n % 10 == 0:
                try:
                    backend.trim_memory()
                except Exception:
                    pass
            if n % 25 == 0 or n == len(left):
                el = time.monotonic() - t_start
                _log(f"  [{n}/{len(left)}] {path.name}: written {written}, {el:.0f} s "
                     f"({el / max(1, n):.1f} s each); last {key} ({len(ids)} tokens)")
        _log(f"{path.name}: written {written}, failed {failed}, over-context {too_long}")
        total_written += written
        total_failed += failed
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    _log(f"Done: written {total_written}, failed {total_failed}.")
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
