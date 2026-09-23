"""Load-time fixes for the DGX Spark (GB10, unified memory).

Measured 2026-09-07 on this box (torch 2.10.0+cu130, transformers 5.5.0):

    H2D copy, ordinary heap tensor            58 GB/s
    H2D copy straight out of an mmap'd
      safetensors slice (page cache WARM)      0.16 GB/s
    same slice .clone()'d to the heap first    6.5 GB/s end to end

transformers' loader (``core_model_loading._materialize_copy``) does exactly the
slow thing: ``tensor = slice[...]`` keeps the mmap as backing storage and
``tensor.to("cuda")`` then copies from file-backed pages, which the driver
handles page by page here. That single call was 95 of the 112 s a warm-cache
18 GB Gemma load spent on weights (profile in AVA_CHANGELOG 2026-09-07), and it is
what turned the previous 250 GB attempts into "1 tensor/s". The fix is one
``clone()`` before the device move — a plain memcpy from the page cache — so the
H2D copy runs from anonymous memory at full speed. Cost: one extra host memcpy
per tensor, the tensor's size in transient RAM, freed as soon as it lands.

Applied only on a unified-memory box (``core.unified_memory``) unless
``AVA_FAST_LOAD=1`` forces it, and never when ``AVA_FAST_LOAD=0``. Idempotent.

Also: ``hf_offline_if_cached`` — a fully cached model spends ~12 s of every
load on Hub HEAD requests (84 of them, profiled); when the snapshot is complete
on disk, set ``HF_HUB_OFFLINE`` for the load so the resolver reads the cache.

**That is an optimization and must never be why a load fails.** The check can only
see the id it is handed, while the loader resolves more behind its back — a LoRA
dir's base (followed here, see ``snapshot_is_cached``) and unsloth's bnb-4bit twin
of a named repo (not knowable from here) — so a file nobody knew to look for can be
missing with offline on. ``inference_backend.load`` therefore retries the load
online once when an offline attempt dies on a missing cache entry. Both halves were
added 2026-09-21, after a cleared HF cache left the box refusing to download its own
base model: the guard was pointed at the adapter dir, which is trivially "on disk",
and reported the load cached while the base it sits on was gone.
"""

from __future__ import annotations

import os
from typing import Optional

_PATCHED_ATTR = "_ava_fast_load_original"


def _should_patch() -> bool:
    flag = os.environ.get("AVA_FAST_LOAD")
    if flag == "0":
        return False
    if flag == "1":
        return True
    from core import unified_memory
    return unified_memory.is_unified_memory()


def install() -> bool:
    """Replace ``transformers.core_model_loading._materialize_copy`` with a version
    that clones out of the mmap before a device move. Returns True when active."""
    if not _should_patch():
        return False
    import torch
    import transformers.core_model_loading as cml
    if getattr(cml, _PATCHED_ATTR, None) is not None:
        return True
    original = getattr(cml, "_materialize_copy", None)
    if original is None:
        # A transformers upgrade moved the copy site. Say so LOUDLY: on this box the
        # difference is a 15 s load versus a 2-3 minute one (see SPARK_LOADING.md).
        print("!!! FAST LOAD NOT ACTIVE: transformers.core_model_loading._materialize_copy is "
              "gone (transformers upgrade?). Loads on this box will run at ~0.16 GB/s until "
              "core/fast_load.py is re-pointed at the new mmap->device copy site. See "
              "SPARK_LOADING.md.", flush=True)
        return False

    def _materialize_copy(tensor, device=None, dtype=None):
        tensor = tensor[...]
        if device is not None:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type == "cuda" and tensor.device.type == "cpu":
                # Out of the mmap and onto the heap: the slow path is file-backed pages.
                tensor = tensor.clone()
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    setattr(cml, _PATCHED_ATTR, original)
    cml._materialize_copy = _materialize_copy
    return True


def uninstall() -> None:
    import transformers.core_model_loading as cml
    original = getattr(cml, _PATCHED_ATTR, None)
    if original is not None:
        cml._materialize_copy = original
        setattr(cml, _PATCHED_ATTR, None)


def adapter_base(path: str) -> Optional[str]:
    """The base model a local LoRA dir resolves against, or None if it is not one.

    A LoRA dir is a complete local snapshot of *itself* and nothing else: the weights
    it sits on top of are named in its ``adapter_config.json`` and fetched from the Hub
    at load time. So "is this on disk?" asked of an adapter has to be asked of its base
    too — see ``snapshot_is_cached``."""
    try:
        import json
        with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as fh:
            base = json.load(fh).get("base_model_name_or_path")
    except Exception:
        return None
    return base if isinstance(base, str) and base.strip() else None


def snapshot_is_cached(model_id: str, _seen: Optional[set] = None) -> bool:
    """True when ``model_id`` and everything it loads through are on disk.

    A local dir is trivially present, but a LoRA adapter dir is only half the load:
    unsloth resolves the base from its ``adapter_config.json``, so the base decides
    whether this load needs the Hub. Reporting an adapter "cached" with its base
    missing is what turned a cleared HF cache into a refusal to download instead of a
    download (2026-09-21)."""
    _seen = _seen or set()
    if model_id in _seen:
        return True
    _seen.add(model_id)
    if os.path.isdir(model_id):
        base = adapter_base(model_id)
        return base is None or snapshot_is_cached(base, _seen)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


class hf_offline_if_cached:
    """Context manager: ``HF_HUB_OFFLINE=1`` for the duration when the model is
    already on disk, so the loader skips its per-file Hub round trips."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._prev: Optional[str] = None
        self.active = False

    def __enter__(self):
        if os.environ.get("HF_HUB_OFFLINE") is None and snapshot_is_cached(self.model_id):
            self._prev = None
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.active = True
        return self

    def __exit__(self, *exc):
        if self.active:
            os.environ.pop("HF_HUB_OFFLINE", None)
        return False


def bench(path: Optional[str] = None, n_tensors: int = 40) -> dict:
    """Measure the three copy paths on a real safetensors shard: straight out of the
    mmap, via clone(), and a heap baseline. `python -m core.fast_load --bench [shard]`.
    Re-run this whenever a load gets slow, before theorizing (SPARK_LOADING.md)."""
    import glob, time
    import torch
    from safetensors import safe_open
    if path is None:
        cands = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*/snapshots/*/*.safetensors")),
                       key=os.path.getsize)
        cands = [c for c in cands if os.path.getsize(c) > 512 * 2**20] or cands
        if not cands:
            raise SystemExit("no safetensors shard found under ~/.cache/huggingface/hub; pass a path")
        path = cands[0]
    out = {"shard": path}
    N = 512 * 2**20
    host = torch.empty(N, dtype=torch.uint8); dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    dev.copy_(host); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3): dev.copy_(host)
    torch.cuda.synchronize(); out["heap_h2d_gbps"] = N * 3 / (time.perf_counter() - t) / 1e9
    with safe_open(path, "pt") as sf:
        keys = [k for k in sf.keys() if "quant" not in k and "absmax" not in k][:n_tensors]
        def run(fn):
            tot = 0; t = time.perf_counter()
            for k in keys:
                x = sf.get_tensor(k); tot += x.numel() * x.element_size(); fn(x)
            torch.cuda.synchronize(); return tot / (time.perf_counter() - t) / 1e9, tot
        out["mmap_h2d_gbps"], tot = run(lambda x: x.to("cuda"))
        out["clone_h2d_gbps"], _ = run(lambda x: x.clone().to("cuda"))
        out["bytes"] = tot
    return out


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--bench"]
        r = bench(args[0] if args else None)
        print(f"shard: {r['shard']}  ({r['bytes']/2**20:.0f} MiB sampled)")
        print(f"host->device, heap tensor         : {r['heap_h2d_gbps']:7.2f} GB/s")
        print(f"host->device, straight from mmap  : {r['mmap_h2d_gbps']:7.2f} GB/s   <- what an unpatched loader does")
        print(f"host->device, clone() then copy   : {r['clone_h2d_gbps']:7.2f} GB/s   <- what fast_load makes it do")
        ratio = r["clone_h2d_gbps"] / max(r["mmap_h2d_gbps"], 1e-9)
        print(f"clone/mmap ratio: {ratio:.0f}x  -> fast_load {'MATTERS on this box' if ratio > 3 else 'is not needed on this box'}")
        sys.exit(0)
    # The offline guard, sealed from the Hub: a plain dir is cached, an adapter dir is
    # cached only when its base is, and the guard stays off when it is not.
    import json, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "plain"); os.makedirs(plain)
        assert adapter_base(plain) is None and snapshot_is_cached(plain)
        lora = os.path.join(tmp, "lora"); os.makedirs(lora)
        with open(os.path.join(lora, "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": "ava/does-not-exist-anywhere"}, fh)
        assert adapter_base(lora) == "ava/does-not-exist-anywhere"
        assert not snapshot_is_cached(lora), "an adapter whose base is gone is NOT cached"
        with open(os.path.join(lora, "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": plain}, fh)
        assert snapshot_is_cached(lora), "an adapter whose base is a local dir IS cached"
        with open(os.path.join(lora, "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": lora}, fh)   # self-reference: must terminate
        assert snapshot_is_cached(lora)
        assert os.environ.get("HF_HUB_OFFLINE") is None
        with hf_offline_if_cached(plain) as off:
            assert off.active and os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("HF_HUB_OFFLINE") is None
        with open(os.path.join(lora, "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": "ava/does-not-exist-anywhere"}, fh)
        with hf_offline_if_cached(lora) as off:
            assert not off.active, "offline must stay OFF when the base still has to be fetched"
            assert os.environ.get("HF_HUB_OFFLINE") is None

    os.environ["AVA_FAST_LOAD"] = "1"
    import torch
    import transformers.core_model_loading as cml
    assert install() and install()
    x = torch.arange(6.0).reshape(2, 3)
    y = cml._materialize_copy(x, device="cpu")
    assert torch.equal(x, y)
    uninstall(); assert getattr(cml, _PATCHED_ATTR) is None
    print("fast_load selftest OK")
