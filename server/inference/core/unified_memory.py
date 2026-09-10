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
"""

from __future__ import annotations

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


def load_kwargs() -> Dict[str, object]:
    """Extra ``from_pretrained`` kwargs for this box: pinned single-device placement
    on unified memory, nothing elsewhere."""
    if is_unified_memory():
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
