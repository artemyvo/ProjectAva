# Model loading on the DGX Spark — READ BEFORE TOUCHING THE LOADER

**If a model load on this box takes minutes instead of seconds, or "1 tensor/s", it is one
of the three things below. None of them is thermal. All three were misdiagnosed for weeks.**

Box: NVIDIA DGX Spark, GB10, 121 GB unified memory shared by CPU and GPU, aarch64,
torch 2.10.0+cu130, transformers 5.5.0, unsloth 2026.8.19. Measured 2026-09-07.

Re-verify the numbers any time (30 s, needs a safetensors shard on disk):

```bash
cd server/inference && ../.venv/bin/python -m core.fast_load --bench
```

---

## 1. Copying a tensor out of the safetensors mmap runs at 0.16 GB/s

| Copy | GB/s |
| --- | --- |
| host → device, ordinary heap tensor | 58 |
| host → device, straight out of the mmap'd safetensors slice (page cache **warm**) | **0.16** |
| the same slice `clone()`d to the heap first, end to end | 4.7–6.5 |

transformers' loader does the slow thing: `core_model_loading._materialize_copy` slices the
mmap (`tensor[...]` keeps the file as backing storage) and calls `.to("cuda")` on it. The
driver then services file-backed pages one at a time. Profiled: a warm 18 GB Gemma load
spent **95 of its 112 weight-loading seconds** inside bitsandbytes' per-tensor `.cpu()`
round trip, which was simply waiting on those copies. Disk was never the bottleneck
(cold read 1.2 GB/s; warm and cold loads took the same time).

**The fix** is one `clone()` before the device move — a plain memcpy from the page cache —
so the H2D copy runs from anonymous memory at full speed:
`server/inference/core/fast_load.py::install`, called by `inference_backend.load` and
`training/train_cycle.py`. Gemma-4-31B: 144 s → 15 s. Applied on unified memory only
(`AVA_FAST_LOAD=1` forces it, `AVA_FAST_LOAD=0` disables). The same file sets
`HF_HUB_OFFLINE` for a model already on disk: 84 Hub HEAD requests, 12 s, per load.

**What will break it:** a transformers upgrade that renames or restructures
`_materialize_copy`. `install()` then returns False and the backend prints a
`!!! FAST LOAD NOT ACTIVE` banner at every load on this box. If you see that banner,
loads are back to minutes — re-point the patch, do not shrug it off.

## 2. CUDA's "free memory" here excludes the page cache

`torch.cuda.mem_get_info()` reports the pool's *MemFree*. A model load fills the page cache
with its own shards, so after one load CUDA reports ~16 GiB free of 121 while ~97 GiB is
instantly reclaimable. accelerate plans the `device_map` from that number, decides the
model does not fit, and offloads to "CPU" — which is the same memory — and bitsandbytes
refuses: `Some modules are dispatched on the CPU or the disk`. A bf16 load can instead
*succeed* half-materialized and die later on `Cannot copy out of meta tensor`.

**The fix:** pin `device_map={"": 0}` whenever the CUDA total equals the system total
(`server/inference/core/unified_memory.py`). Offload is meaningless on this box.

## 3. bitsandbytes never quantizes fused MoE experts

transformers' bnb path converts `nn.Linear` only. Qwen3.5-122B-A10B's experts are fused
3-D `nn.Parameter`s holding 232 of its 250 GB, so `load_in_4bit` on the bf16 repo keeps
them bf16 — a model that cannot exist in 121 GB. It crawled into swap and **froze the box**
(twice). Never load `unsloth/Qwen3.5-122B-A10B` directly here.

**The fix:** a one-time conversion to unsloth's per-expert `Linear4bit` layout
(`server/convert_moe_bnb4bit.py` → `server/models/base/Qwen3.5-122B-A10B-bnb-4bit`, 66 GB)
plus the matching experts class (`server/inference/core/moe_bnb_experts.py`), which also
removes a transformers key-renaming that otherwise drops every bnb quant state in the
multimodal class. `server_config.json`'s `model_id` points at the converted dir.

gpt-oss-120b has the same shape of problem with a different cause: the raw
`openai/gpt-oss-120b` is MXFP4, which unsloth dequantizes to 234 GB bf16 without triton
`kernels`. Use `unsloth/gpt-oss-120b-unsloth-bnb-4bit` (58 GB) and nothing else.

---

## The box freezes from memory pressure, not heat

Both hard freezes were the unified pool overcommitted into the 16 GB swap file: once from
(3), once from an 8-way `nvcc` build (causal-conv1d) left running beside a 60 GB load.
Rules that hold now:

- One memory-heavy thing at a time: a model load, a conversion, a compile. Never a source
  build beside a loaded model.
- A memory watchdog on long GPU jobs (kill below ~10 GiB available / above ~8 GiB swap).
- Two or three GB of swap use during a load is normal here (idle processes paged out
  while the page cache is hot); a *climbing* swap figure is the freeze approaching.
- Lowering `vm.swappiness` or disabling the swap file (so the OOM killer acts instead of
  the box thrashing) is an operator decision; the code does not touch it.

## Numbers as validated (2026-09-07)

| Model | Load | Resident |
| --- | --- | --- |
| `unsloth/gemma-4-31B-it` (bnb-4bit twin) | 15 s | 18 GiB |
| `unsloth/gpt-oss-120b-unsloth-bnb-4bit` | 65 s cold | 58 GiB |
| `models/base/Qwen3.5-122B-A10B-bnb-4bit` | 85–100 s cold | 62 GiB |

Full history and the profile: `documentation/AVA_CHANGELOG.md` → 2026-09-07.
