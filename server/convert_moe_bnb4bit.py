#!/usr/bin/env python
"""Convert a bf16 Qwen3.5-MoE checkpoint into a per-expert bitsandbytes NF4 checkpoint.

The problem it solves (see ``inference/core/moe_bnb_experts.py``): transformers stores
MoE experts as fused 3D parameters that bitsandbytes never quantizes, so
``load_in_4bit`` on Qwen3.5-122B-A10B keeps 232 GB of experts in bf16 — which
cannot exist on a 121 GB unified-memory box. This writes the layout unsloth ships
for gpt-oss (``experts.gate_up_projs.<i>`` / ``experts.down_projs.<i>`` as
pre-quantized ``Linear4bit``), plus every other Linear the loader would quantize,
so a load is a pure deserialization: ~66 GB read, nothing quantized at load time.

Streams shard by shard (one fused tensor at a time on the GPU), so peak RSS stays
a few GB whatever the source size. Output is a self-contained local model dir:
point ``server_config.json``'s ``model_id`` at it.

Usage (from server/, GPU needed):
    .venv/bin/python convert_moe_bnb4bit.py --src unsloth/Qwen3.5-122B-A10B \
        --out models/base/Qwen3.5-122B-A10B-bnb-4bit [--plan] [--layers N] [--shard-gb 4]

``--plan`` is GPU-free: it prints what would be quantized / copied / skipped and exits.
``--layers N`` converts only the first N decoder layers' experts (a smoke test of the
pipeline; the result is NOT a loadable model).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "inference"))

EXPERT_FUSED_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<which>gate_up_proj|down_proj)$")
SIDE_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
    "generation_config.json", "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json",
)
# What stays bf16. unsloth's own list (lm_head, routers, vision projectors, …) plus
# the Qwen3.5 pieces that are tiny or gate-like, where NF4 buys nothing and costs
# accuracy: the shared-expert gate (H→1), the GDN decay/beta projections (H→64),
# and the whole vision tower (0.9 GB — not worth quantizing, and unsloth freezes it).
EXTRA_SKIP = ["embed_tokens", "shared_expert_gate", "in_proj_a", "in_proj_b", "visual"]


def resolve_snapshot(src: str) -> str:
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download
    return snapshot_download(src, local_files_only=True)


def skip_list() -> list:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    import unsloth  # noqa: F401  (unsloth_zoo refuses to import before it)
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
    out = list(SKIP_QUANTIZATION_MODULES)
    for k in EXTRA_SKIP:
        if k not in out:
            out.append(k)
    return out


def plan(snapshot: str, skips: list):
    """Which checkpoint keys become quantized Linear weights, using the SAME module
    walk + skip rule transformers applies at load (``should_convert_module``)."""
    import torch
    from transformers import AutoConfig
    from transformers.quantizers.quantizers_utils import should_convert_module
    from core import moe_bnb_experts
    moe_bnb_experts.install()
    cfg = AutoConfig.from_pretrained(snapshot)
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    with torch.device("meta"):
        model = Qwen3_5MoeForConditionalGeneration(cfg)
    linear_keys = set()
    perexpert_keys = set()
    for name, mod in model.named_modules():
        if type(mod) is torch.nn.Linear:
            if f".{moe_bnb_experts.GATE_UP_LIST}." in name or f".{moe_bnb_experts.DOWN_LIST}." in name:
                perexpert_keys.add(name + ".weight")
            elif should_convert_module(name, skips):
                linear_keys.add(name + ".weight")
    model_keys = set(model.state_dict().keys())
    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    ckpt = set(idx.keys())
    fused = {k for k in ckpt if EXPERT_FUSED_RE.match(k)}
    mtp = {k for k in ckpt if k.startswith("mtp.")}
    quantize = linear_keys & ckpt
    copy = ckpt - fused - mtp - quantize
    not_in_model = copy - model_keys
    return dict(cfg=cfg, idx=idx, fused=fused, mtp=mtp, quantize=quantize, copy=copy,
                not_in_model=not_in_model, perexpert=perexpert_keys,
                missing_from_ckpt=(model_keys - ckpt - perexpert_keys
                                   - {k for k in model_keys if k.endswith(("gate_up_proj", "down_proj"))}))


def quantize_2d(w, device):
    """bf16 [out, in] → (packed uint8 data, quant-state dict) exactly as bnb serializes
    a Linear4bit (nf4, double quant, uint8 storage) — the form transformers'
    ``Bnb4bitDeserialize`` rebuilds with ``Params4bit.from_prequantized``."""
    import torch
    from bitsandbytes.nn import Params4bit
    p = Params4bit(w.contiguous(), requires_grad=False, compress_statistics=True,
                   quant_type="nf4", quant_storage=torch.uint8, module=None).to(device)
    out = {"": p.data.detach().cpu()}
    for k, v in p.quant_state.as_dict(packed=True).items():
        out[k] = v.detach().cpu()
    return out


class ShardWriter:
    def __init__(self, out_dir: str, shard_bytes: int):
        self.out_dir = out_dir; self.shard_bytes = shard_bytes
        self.buf = {}; self.buf_bytes = 0; self.n = 0; self.weight_map = {}; self.total = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, key: str, t):
        self.buf[key] = t; nb = t.numel() * t.element_size()
        self.buf_bytes += nb; self.total += nb
        if self.buf_bytes >= self.shard_bytes:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        from safetensors.torch import save_file
        self.n += 1
        name = f"model-{self.n:05d}.safetensors"
        save_file(self.buf, os.path.join(self.out_dir, name), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = name
        self.buf = {}; self.buf_bytes = 0

    def finish(self):
        self.flush()
        # Rename to the conventional -of- form now the count is known.
        wm = {}
        for k, name in self.weight_map.items():
            i = int(name.split("-")[1].split(".")[0])
            wm[k] = f"model-{i:05d}-of-{self.n:05d}.safetensors"
        for i in range(1, self.n + 1):
            os.rename(os.path.join(self.out_dir, f"model-{i:05d}.safetensors"),
                      os.path.join(self.out_dir, f"model-{i:05d}-of-{self.n:05d}.safetensors"))
        json.dump({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(wm.items()))},
                  open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w"), indent=2)


def write_config(snapshot: str, out_dir: str, skips: list, source_id: str = ""):
    from core import moe_bnb_experts
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    cfg["quantization_config"] = {
        "_load_in_4bit": True, "_load_in_8bit": False,
        "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
        "llm_int8_enable_fp32_cpu_offload": False, "llm_int8_has_fp16_weight": False,
        "llm_int8_skip_modules": skips, "llm_int8_threshold": 6.0,
        "load_in_4bit": True, "load_in_8bit": False, "quant_method": "bitsandbytes",
    }
    cfg[moe_bnb_experts.PEREXPERT_MARKER] = True
    # Provenance: the source id this was derived from, deterministically, by this
    # script — what a snapshot manifest should cite as the base (local dirs have no id).
    cfg["ava_converted_from"] = source_id
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for f in SIDE_FILES:
        p = os.path.join(snapshot, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="unsloth/Qwen3.5-122B-A10B")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "base", "Qwen3.5-122B-A10B-bnb-4bit"))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="smoke test: only this many layers of experts")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    args = ap.parse_args()

    snapshot = resolve_snapshot(args.src)
    skips = skip_list()
    P = plan(snapshot, skips)
    print(f"source: {snapshot}")
    print(f"skip list: {skips}")
    print(f"fused expert tensors: {len(P['fused'])}  | linear weights to quantize: {len(P['quantize'])}  "
          f"| copied as-is: {len(P['copy'])}  | dropped (mtp): {len(P['mtp'])}")
    print(f"per-expert Linear weights in model: {len(P['perexpert'])}")
    if P["not_in_model"]:
        print(f"WARNING {len(P['not_in_model'])} checkpoint keys have no model parameter (copied anyway): "
              f"{sorted(P['not_in_model'])[:8]}")
    if P["missing_from_ckpt"]:
        print(f"WARNING {len(P['missing_from_ckpt'])} model parameters absent from checkpoint: "
              f"{sorted(P['missing_from_ckpt'])[:8]}")
    print("quantize sample:", sorted(P["quantize"])[:6])
    if args.plan:
        return

    import torch
    from safetensors import safe_open
    from core import moe_bnb_experts
    device = torch.device("cuda")
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    writer = ShardWriter(args.out, int(args.shard_gb * 2**30))
    idx = P["idx"]
    shards = sorted(set(idx.values()))
    t0 = time.time(); done_bytes = 0; n_exp = 0
    for si, shard in enumerate(shards):
        with safe_open(os.path.join(snapshot, shard), "pt") as f:
            for key in f.keys():
                if key in P["mtp"]:
                    continue
                m = EXPERT_FUSED_RE.match(key)
                if m:
                    layer = int(re.search(r"layers\.(\d+)\.", key).group(1))
                    if args.layers is not None and layer >= args.layers:
                        continue
                    which = m.group("which"); prefix = m.group("prefix")
                    lst = moe_bnb_experts.GATE_UP_LIST if which == "gate_up_proj" else moe_bnb_experts.DOWN_LIST
                    # clone() first: a copy straight out of the mmap runs at ~0.16 GB/s on
                    # GB10, from anonymous memory at full speed (see core/fast_load.py).
                    t = f.get_tensor(key).clone().to(device, non_blocking=False)
                    for e in range(t.shape[0]):
                        q = quantize_2d(t[e], device)
                        base = f"{prefix}.{lst}.{e}.weight"
                        for k, v in q.items():
                            writer.add(base if k == "" else f"{base}.{k}", v)
                        n_exp += 1
                    done_bytes += t.numel() * t.element_size()
                    del t
                elif key in P["quantize"]:
                    t = f.get_tensor(key).clone().to(device)
                    q = quantize_2d(t, device)
                    for k, v in q.items():
                        writer.add(key if k == "" else f"{key}.{k}", v)
                    done_bytes += t.numel() * t.element_size(); del t
                else:
                    t = f.get_tensor(key).clone()
                    writer.add(key, t); done_bytes += t.numel() * t.element_size()
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[{el:7.0f}s] shard {si+1}/{len(shards)} done | {done_bytes/1e9:.0f} GB in | "
              f"{writer.total/1e9:.1f} GB out | experts {n_exp} | {done_bytes/1e9/max(el,1):.2f} GB/s", flush=True)
    writer.finish()
    write_config(snapshot, args.out, skips, source_id=args.src)
    print(f"DONE: {args.out}  ({writer.total/1e9:.1f} GB, {writer.n} shards, {len(writer.weight_map)} tensors, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
