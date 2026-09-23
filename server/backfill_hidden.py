#!/usr/bin/env python3
"""Backfill hidden-state captures over the stored corpus — AVA_REWARD_LOOP.md §4.3 step 1.

    cd server
    .venv/bin/python backfill_hidden.py [--with-adapter] [--limit N] [--only-labeled]
                                        [--axes-only] [--context 16384] [--dry-run]

GPU, offline: run with the inference server DOWN. The live capture (`core/hidden_capture`)
writes residuals only for turns generated since it landed; every earlier exchange has
its exact generated token series in the transcript's `tension.token_ids` and nothing
else. This replays each of them teacher-forced — the prompt rebuilt the way the live
turn built it (the stored `system_content`, the CoT-stripped history, the user turn,
the family's chat template and think prefill), then the stored generated ids appended
raw — and reduces the generated positions exactly as the live hook does
(`reduce_capture`: residuals at step 0 / paragraph breaks / last token for the stored
layer band, plus per-token projections onto every axis under `data/axes/`). The result
is appended to the transcript's `.hidden.npz` under a **variant** group,
``<exchange_id>~base`` by default, beside any live capture: the live one is under the
weights that wrote the reply, this one under the bare base, which is the one set of
weights every transcript shares (the label inventory found the corpus spread over 13
adapters). `--with-adapter` files it under the adapter's directory name instead.

What this is and is not. The states are what the BASE reads at each generated token
given the same prefix — the right material for a probe trained across the corpus, and
for the axis projections that let the inventory and the ledger see the whole history.
They are not the states the adapter had while generating; those exist only in live
captures. A transcript without `system_content` (pre-v2) is replayed under the
persona-empty framing and marked ``framing: fallback`` in its meta.

Resumable: exchanges already captured under the variant are skipped, so a run cut short
continues where it stopped. Read-only on everything but the `.hidden.npz` sidecars.
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


def _persona_empty_system(prompts_dir: Path) -> str:
    parts = []
    for name in ("chat_prompt.txt", "persona_undecided_prompt.txt"):
        p = prompts_dir / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n".join(x for x in parts if x)


def _conversation(doc: dict, index: int, fallback_system: str) -> tuple:
    """(conversation, framing) as the live turn assembled it: the exchange's exact
    system content, the prior exchanges CoT-stripped, the user turn."""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chats", type=Path, default=_SERVER_DIR / "data" / "chats")
    ap.add_argument("--axes", type=Path, default=_SERVER_DIR / "data" / "axes")
    ap.add_argument("--with-adapter", action="store_true",
                    help="replay under the configured adapter (variant = its directory name)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N exchanges (0 = all)")
    ap.add_argument("--only-labeled", action="store_true",
                    help="only exchanges with a keep/revise verdict in their sidecar")
    ap.add_argument("--axes-only", action="store_true",
                    help="store no residuals, only the per-token axis projections")
    ap.add_argument("--context", type=int, default=16384,
                    help="load context; an exchange whose prefix + reply exceeds it is skipped")
    ap.add_argument("--oldest-first", action="store_true", help="default is newest first")
    ap.add_argument("--dry-run", action="store_true", help="count what would be replayed, no model")
    args = ap.parse_args(argv)

    from core import hidden_capture as hc
    from core import model_family
    from core.chat_sidecar import iter_chat_json_files, sidecar_path_for
    from training.reflections_path import load_server_config

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    adapter_id = str(config.get("adapter_id") or "") if args.with_adapter else ""
    if not model_id:
        raise SystemExit("model_id not set in server_config.json")
    variant = "base"
    if adapter_id:
        variant = Path(adapter_id.rstrip("/\\")).name or "adapter"

    # ── plan ────────────────────────────────────────────────────────────
    files = sorted(iter_chat_json_files(Path(args.chats)))
    if not args.oldest_first:
        files.reverse()
    plan: list = []          # (chat_path, doc, index, key)
    skipped_no_series = skipped_done = 0
    for chat in files:
        try:
            doc = json.loads(chat.read_text(encoding="utf-8"))
        except Exception:
            continue
        exs = doc.get("exchanges") or []
        if not isinstance(exs, list):
            continue
        done = hc.captured_exchange_ids(hc.sidecar_path(chat), variant)
        verdicts = {}
        if args.only_labeled:
            try:
                side = json.loads(sidecar_path_for(chat).read_text(encoding="utf-8"))
                verdicts = side.get("exchanges") or {}
            except Exception:
                verdicts = {}
        for i, ex in enumerate(exs):
            if not isinstance(ex, dict):
                continue
            tension = ex.get("tension") if isinstance(ex.get("tension"), dict) else {}
            if not tension.get("token_ids"):
                skipped_no_series += 1
                continue
            key = hc.exchange_key(ex, i)
            if key in done:
                skipped_done += 1
                continue
            if args.only_labeled:
                rec = verdicts.get(str(i)) if isinstance(verdicts, dict) else None
                v = str((rec or {}).get("verdict") or "").lower()
                if v not in ("keep", "revise") or (rec or {}).get("banned"):
                    continue
            plan.append((chat, doc, i, key))
    if args.limit > 0:
        plan = plan[:args.limit]
    _log(f"{len(files)} transcripts; {len(plan)} exchanges to replay under variant {variant!r} "
         f"(already captured {skipped_done}, no stored series {skipped_no_series})")
    if args.dry_run or not plan:
        return 0

    # ── model ───────────────────────────────────────────────────────────
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import build_inference_prompt, ensure_chat_template

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
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    fam = model_family.family_for(model_id)
    spec = hc.capture_spec(model, model_id, Path(args.axes))
    if spec is None:
        raise SystemExit("hidden.capture is off or no decoder stack was found — nothing to do")
    store_layers = [] if args.axes_only else list(spec["store_layers"])
    hook_layers = sorted(set(store_layers) | {a.layer for a in spec["axes"]})
    if not hook_layers:
        raise SystemExit("no layers to read (axes-only with no axis files?)")
    _log(f"Loaded in {time.monotonic() - t0:.0f} s; {spec['n_layers']} layers, storing "
         f"{store_layers or 'none'}, axes {[a.name for a in spec['axes']] or 'none'}. "
         f"{backend.memory_status()}")
    fallback_system = _persona_empty_system(_SERVER_DIR / "inference" / "prompts")

    def _decode(ids):
        return text_tok.decode(list(ids))

    # ── replay ──────────────────────────────────────────────────────────
    written = failed = too_long = 0
    n_tokens = 0
    t_start = time.monotonic()
    for n, (chat, doc, i, key) in enumerate(plan, 1):
        ex = doc["exchanges"][i]
        gen_ids = [int(t) for t in ex["tension"]["token_ids"]]
        try:
            conv, framing = _conversation(doc, i, fallback_system)
            prompt = build_inference_prompt(tokenizer, conv, **fam.template_kwargs) + fam.think_prefill
            # Mirror stream_generate's tokenization (default special-token handling).
            prefix_ids = [int(t) for t in text_tok(prompt)["input_ids"]]
        except Exception as e:
            failed += 1
            _log(f"  [{n}/{len(plan)}] {chat.name} #{i}: prompt rebuild failed: {type(e).__name__}: {e}")
            continue
        total = len(prefix_ids) + len(gen_ids)
        if total > args.context:
            too_long += 1
            _log(f"  [{n}/{len(plan)}] {chat.name} #{i}: {total} tokens > context {args.context}, skipped")
            continue
        try:
            rows = hc.forward_ids_rows(model, prefix_ids + gen_ids, hook_layers,
                                       all_positions=True, from_position=len(prefix_ids))
            per_layer = {l: [m[t] for t in range(m.shape[0])] for l, m in rows.items()
                         if m.shape[0] == len(gen_ids)}
            cap = hc.reduce_capture(gen_ids, per_layer, _decode, store_layers=store_layers,
                                    axes=spec["axes"], max_positions=spec["max_positions"])
            if cap is None:
                failed += 1
                _log(f"  [{n}/{len(plan)}] {chat.name} #{i}: rows did not align, skipped")
                continue
            p = hc.write_sidecar(chat, key, cap, model_id=model_id, adapter_id=adapter_id,
                                 layer_spec=spec["layer_spec"], variant=variant,
                                 extra_meta={"framing": framing, "backfill": True,
                                             "prefix_tokens": len(prefix_ids),
                                             "history_exchanges": i})
            if p is None:
                failed += 1
                continue
            written += 1
            n_tokens += len(gen_ids)
        except Exception as e:
            failed += 1
            _log(f"  [{n}/{len(plan)}] {chat.name} #{i}: {type(e).__name__}: {e}")
            try:
                backend.trim_memory()
            except Exception:
                pass
            continue
        if n % 10 == 0 or n == len(plan):
            el = time.monotonic() - t_start
            _log(f"  [{n}/{len(plan)}] written {written}, {n_tokens} tokens, {el:.0f} s "
                 f"({n_tokens / max(el, 1e-6):.0f} tok/s); last: {chat.name} #{i} "
                 f"({len(prefix_ids)}+{len(gen_ids)} tokens, {framing})")
    _log(f"Done: written {written}, failed {failed}, over-context {too_long}, "
         f"{time.monotonic() - t_start:.0f} s. `python -m core.label_inventory` now counts them "
         f"under `base`.")
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
