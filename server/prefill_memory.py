#!/usr/bin/env python3
"""Prefill memory of the clean base by prompt length — AVA_OPEN_PROBLEMS "Clean-Base
Out-of-Memory on Long Prompts" (2026-10-10).

    cd server
    .venv/bin/python prefill_memory.py [--lengths 1024,2048,4096,6144,8192,12288] [--adapter DIR]

GPU, offline (watchdog job `prefill_memory`, inference down). Loads the base as the judges
do (adapter off, the box's quantization and context), then for each length:

  * a prefill forward of a real-text prompt of that many tokens (``no_grad``, ``use_cache``,
    as ``generate`` prefills), with the peak above the loaded model;
  * every ``scaled_dot_product_attention`` call during it, its own peak above what was
    allocated when it started, grouped by the layer's head_dim (512 = gemma-4's global
    layers, 256 = sliding): the math backend materializes a fp32 score matrix that grows
    as L² (32 heads × L² × 4 bytes ≈ 1.68 GiB at L ≈ 3,750), the memory-efficient one does
    not. Which of the two the global layers take is the question;
  * then the same prompt through ``stream_generate`` (4 new tokens), the production path.

An OOM at a length is recorded and the run goes on. Writes
``data/measure/prefill_memory_<ts>.json`` and prints a table; changes nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GiB = float(1 << 30)


def _text_corpus(chats: Path, want_chars: int) -> str:
    """Real conversation text (user turns and replies, oldest chats first) until *want_chars*."""
    parts, n = [], 0
    for path in sorted(chats.glob("2026*.json")):
        if path.name.endswith((".state.json", ".summary.json", ".facts.json")):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ex in doc.get("exchanges") or []:
            if not isinstance(ex, dict):
                continue
            for t in (ex.get("user_prompt"), ex.get("assistant_response")):
                if t:
                    parts.append(str(t))
                    n += len(t)
        if n >= want_chars:
            break
    return "\n\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lengths", default="1024,2048,3072,4096,6144,8192,12288")
    ap.add_argument("--adapter", default="", help="an adapter dir (default: the bare base, as the judges run)")
    ap.add_argument("--generate", default="2048,4096,8192", help="lengths also run through stream_generate")
    args = ap.parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    gen_lengths = {int(x) for x in args.generate.split(",") if x.strip()}

    import torch
    import torch.nn.functional as F
    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from judge_asks import _quant_flags
    from training.reflections_path import hot_chats_dir, load_server_config

    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    context = max(int(config.get("context_length") or 8192), int(config.get("reflect_context_length") or 0))
    l4, l8 = _quant_flags(str(config.get("base_quant") or ""))
    ensure_expandable_segments()
    backend = UnslothBackend()
    adapter = args.adapter or None
    model, tokenizer = backend.load(model_id, context, adapter, load_in_4bit=l4, load_in_8bit=l8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"loaded {model_id}{' + ' + adapter if adapter else ' (base)'}: {loaded / GiB:.2f} GiB allocated, "
          f"{(total - torch.cuda.memory_reserved()) / GiB:.2f} GiB free of {total / GiB:.2f}", flush=True)

    # The SDPA spy: each call's own peak above what was allocated when it started, by head_dim.
    calls: dict = defaultdict(list)
    orig_sdpa = F.scaled_dot_product_attention

    def spy(query, key, value, *a, **kw):
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = orig_sdpa(query, key, value, *a, **kw)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - start
        calls[int(query.shape[-1])].append({
            "q": list(query.shape), "k": list(key.shape), "mask": a[0] is not None if a else kw.get("attn_mask") is not None,
            "enable_gqa": bool(kw.get("enable_gqa", False)), "dtype": str(query.dtype), "peak_gib": peak / GiB})
        return out

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = _text_corpus(Path(hot_chats_dir()), max(lengths) * 6)
    all_ids = [int(t) for t in tok(text)["input_ids"]]
    print(f"corpus: {len(all_ids)} tokens of real conversation text", flush=True)
    results = []
    for L in lengths:
        if L > len(all_ids):
            print(f"L={L}: corpus too short", flush=True)
            continue
        row = {"L": L}
        ids = torch.tensor([all_ids[:L]], device="cuda")
        calls.clear()
        F.scaled_dot_product_attention = spy
        try:
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
            torch.cuda.synchronize()
            row["prefill_peak_gib"] = (torch.cuda.max_memory_allocated() - base) / GiB
            del out
        except torch.OutOfMemoryError as e:
            row["prefill_oom"] = str(e)[:160]
        finally:
            F.scaled_dot_product_attention = orig_sdpa
            backend.trim_memory()
        row["attention"] = {str(hd): {"calls": len(v), "max_call_peak_gib": max(c["peak_gib"] for c in v),
                                      "mask": v[0]["mask"], "enable_gqa": v[0]["enable_gqa"],
                                      "q": v[0]["q"], "k": v[0]["k"], "dtype": v[0]["dtype"]}
                            for hd, v in calls.items() if v}
        if L in gen_lengths:
            try:
                prompt = tok.decode(all_ids[:L])
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for _chunk in backend.stream_generate(model, tokenizer, prompt, 4, context, 0.0, 1.0):
                    pass
                torch.cuda.synchronize()
                row["generate_peak_gib"] = (torch.cuda.max_memory_allocated() - base) / GiB
            except torch.OutOfMemoryError as e:
                row["generate_oom"] = str(e)[:160]
            except Exception as e:  # noqa: BLE001 — the production path's own error, recorded
                row["generate_error"] = f"{type(e).__name__}: {e}"[:200]
            finally:
                backend.trim_memory()
        results.append(row)
        att = "  ".join(f"hd{hd}: {a['calls']} calls, max {a['max_call_peak_gib']:.2f} GiB"
                        f"{' mask' if a['mask'] else ''}{' gqa' if a['enable_gqa'] else ''}"
                        for hd, a in sorted(row["attention"].items()))
        print(f"L={L:>6}  prefill {row.get('prefill_peak_gib', float('nan')):.2f} GiB"
              f"{' OOM' if 'prefill_oom' in row else ''}  generate "
              f"{row.get('generate_peak_gib', float('nan')):.2f} GiB"
              f"{' OOM' if 'generate_oom' in row else ''}{' ERR' if 'generate_error' in row else ''}  | {att}",
              flush=True)
    out_path = _SERVER_DIR / "data" / "measure" / f"prefill_memory_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({"model_id": model_id, "adapter": adapter, "loaded_gib": loaded / GiB,
                                    "total_gib": total / GiB, "torch": torch.__version__, "results": results},
                                   indent=1), encoding="utf-8")
    print(f"written {out_path}", flush=True)
    try:
        backend.release(model, tokenizer)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
