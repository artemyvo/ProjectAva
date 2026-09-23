"""Unified-memory (DGX Spark / GB10) load placement.

On a Grace-Blackwell box the GPU and the CPU share ONE physical pool. Two
consequences bite every model load:

1. ``torch.cuda.mem_get_info()`` reports the pool's *MemFree*, which excludes the
   page cache — and a model load fills the page cache with its own shards. After
   one load CUDA "sees" ~16 GiB free out of 121 while 97 GiB is instantly
   reclaimable. accelerate plans the ``device_map`` from that number, decides the
   model cannot fit, and offloads to CPU/disk; bitsandbytes then refuses the load
   ("Some modules are dispatched on the CPU or the disk") or, worse, a bf16 load
   succeeds half-materialized (see ``UnslothBackend._assert_fully_materialized``).

2. Offload is meaningless here anyway: "CPU" memory IS the GPU's memory. The only
   placement that makes sense is everything on device 0 and letting the allocator
   pull from the shared pool.

So on such a box every load pins ``device_map={"": 0}`` and skips accelerate's
free-memory planning. Detected by the CUDA total being (within tolerance) the
system's MemTotal — a discrete GPU reports its own VRAM, an order of magnitude
below system RAM.

Since 2026-09-17 a single-GPU DISCRETE box is pinned too (see ``load_kwargs``):
the planner's conservative estimate refused a model that fit, and an offloaded
load is a failed load on this project anyway.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def _meminfo_total_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def is_unified_memory(tolerance: float = 0.15) -> bool:
    """True when the CUDA device's total memory is the system's total memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        _free, total = torch.cuda.mem_get_info()
    except Exception:
        return False
    sys_total = _meminfo_total_bytes()
    if not sys_total or not total:
        return False
    return abs(total - sys_total) <= tolerance * sys_total


def _single_cuda_device() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() == 1
    except Exception:
        return False


def load_kwargs() -> Dict[str, object]:
    """Extra ``from_pretrained`` kwargs for this box: pinned single-device placement
    (``device_map={"": 0}``) on unified memory AND on any single-GPU box; nothing on a
    multi-GPU one. ``AVA_LOAD_PLACEMENT=planner`` restores the loader's own planning
    on a discrete GPU.

    Pinning was Spark-only until 2026-09-17. On a discrete card the planner (unsloth's
    head-aware map over accelerate) sizes the model from its own estimate against a
    fraction of free VRAM and, when short, offloads part of it to CPU/disk — which
    ``UnslothBackend._assert_fully_materialized`` then refuses, so on this project an
    offloaded load is a failed load either way. The estimate is conservative enough to
    refuse a model that fits: the RTX 5090 box loaded gemma-4-31B (~23 GB resident)
    with 28.1 GiB free at every clean-base swap for months, and one day the planner
    put ``lm_head`` on meta at that same 28.1 GiB, taking the reflection run and then
    the box down. Pinned, the load either fits or raises a real CUDA OOM at the point
    of failure — the shape the swap's restore path is built for — and the guard stays
    as the backstop."""
    if is_unified_memory():
        return {"device_map": {"": 0}}
    if _single_cuda_device() and os.environ.get("AVA_LOAD_PLACEMENT", "").lower() != "planner":
        return {"device_map": {"": 0}}
    return {}


def describe() -> str:
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        sys_total = _meminfo_total_bytes() or 0
        return (f"unified_memory={is_unified_memory()} cuda_total={total/2**30:.1f}GiB "
                f"cuda_free={free/2**30:.1f}GiB sys_total={sys_total/2**30:.1f}GiB")
    except Exception as e:
        return f"unified_memory=? ({e})"
