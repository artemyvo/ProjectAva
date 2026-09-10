"""CUDA allocator guard: verify expandable-segments mode is ACTUALLY on, and name OOMs.

ONE definition, two processes. The inference server calls :func:`ensure_expandable_segments`
at the head of ``main()`` (before any model is loaded), and the offline train cycle's
``training/train_cycle._ensure_expandable_segments`` delegates here (train_cycle already
puts ``inference/`` on ``sys.path`` for the activity tee). The probe used to live only in
the train cycle, which meant the inference process — the one that runs for days and OOMs
on the facts fetch — merely *set the env vars and trusted them*, with no ``[alloc]`` line
in ``server.log`` to say whether the guard was live.

Why trusting the env is not enough (the 2026-08-26 training-box investigation, recorded in
full in ``documentation/AVA_CHANGELOG.md``):

- torch 2.9.1 was VERIFIED to silently ignore the new ``PYTORCH_ALLOC_CONF`` name (env
  set before import, ``is_expandable`` False on the memory snapshot) — only the legacy
  ``PYTORCH_CUDA_ALLOC_CONF`` engages the mode there. Both names are set at every CUDA
  entry point, but "set" and "engaged" were observed to diverge.
- ``setdefault`` yields to a value pre-set in the parent environment (the watchdog's, a
  shell profile's), and unsloth — imported right after, which is what initializes CUDA —
  has been known to write allocator env vars of its own in some versions.
- The failure is indistinguishable from ordinary fragmentation until a long run dies of
  it: free memory strands inside SPLIT segments (partly used, partly free), which
  ``empty_cache()`` structurally cannot release — the every-50-steps reclaim callback
  was shown NOT to help. Two 8192-cap builds died mid-run with 6.4–6.9 GiB
  reserved-but-unallocated; the inference side shows the same fingerprint as facts-fetch
  OOMs complaining of fragmentation with ~3 GiB nominally free.

So: probe empirically (allocate, read ``is_expandable`` off the snapshot); if inactive,
force it through the runtime API (``torch.cuda.memory._set_allocator_settings`` —
applies to segments created afterwards, which is why the call site must precede anything
model-sized); re-probe; print the verdict + the env values as this process actually
inherited them. Best-effort at every step — the probe must never take a process down.

Also home to the OOM classifiers (:func:`is_cuda_oom` / :func:`is_oom_message`), so the
callers that retry or relabel an OOM (the facts fetch's one-shot retry in
``core/generation.py``, the ``"oom"`` skip classification in ``core/fact_fetch.py``)
share one definition of "that was the allocator, not the pass".

GPU-free self-test: ``python -m core.alloc_guard`` (runs without torch installed).
"""

from __future__ import annotations

import os
from typing import Callable, Optional


def ensure_expandable_segments(log: Callable[[str], None] = None) -> Optional[bool]:
    """Probe whether the CUDA allocator runs in expandable-segments mode; force if not.

    Returns True/False for the final probed state, or None when there is no CUDA (or no
    torch) to probe — and says so in the log either way, because the env-var route fails
    silently and the failure is indistinguishable from fragmentation until something
    dies of it. Call BEFORE anything model-sized is allocated: the runtime force applies
    only to segments created afterwards.
    """
    emit = log or (lambda s: print(s, flush=True))
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        def _probe() -> bool:
            t = torch.empty(8 * 1024 * 1024, device="cuda")  # 32 MiB → large pool
            try:
                snap = torch.cuda.memory_snapshot()
                segs = snap.get("segments", []) if isinstance(snap, dict) else snap
                return any(bool(s.get("is_expandable")) for s in segs)
            finally:
                del t
                torch.cuda.empty_cache()

        env = {k: os.environ.get(k)
               for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
        active = _probe()
        forced = False
        if not active:
            try:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                active = _probe()
                forced = active
            except Exception as set_err:
                emit(f"[alloc] runtime _set_allocator_settings failed: {set_err}")
        state = "ACTIVE (forced at runtime)" if forced else (
            "ACTIVE" if active else "INACTIVE")
        emit(f"[alloc] expandable_segments {state} — env {env}")
        if not active:
            emit("[alloc] WARNING: allocator is in non-expandable mode; long-lived "
                 "processes will strand freed memory in split segments and OOM with "
                 "gigabytes nominally free (see the 2026-08-26 changelog entries)")
        return active
    except Exception as probe_err:
        emit(f"[alloc] allocator probe failed: {probe_err}")
        return None


def is_oom_message(text: str) -> bool:
    """Does this error text describe a CUDA out-of-memory?

    Pure string classification, for the pure modules (``fact_fetch``) that see only the
    stringified error a catch-all stored. Deliberately narrow: a generic "out of memory"
    alone is not matched, because a CPU MemoryError relabeled as a GPU problem would
    send an operator chasing the wrong allocator.
    """
    t = (text or "").lower()
    return ("cuda out of memory" in t
            or "hip out of memory" in t
            or ("out of memory" in t and ("cuda" in t or "gpu" in t or "vram" in t)))


def is_cuda_oom(exc: BaseException) -> bool:
    """Is this exception a CUDA out-of-memory (typed when torch is importable, by
    message otherwise)?

    The message fallback matters twice over: the GPU-free self-tests run without torch,
    and an OOM re-raised through wrappers can arrive as a plain RuntimeError carrying
    the allocator's text (``inference_backend.stream_generate`` severs the traceback and
    re-raises the lean error, but the type survives; other layers may not be so
    careful).
    """
    if exc is None:
        return False
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return is_oom_message(str(exc))


def _selftest() -> None:
    # The classifiers are the GPU-free half; the probe is exercised only for harmlessness.
    assert is_oom_message("CUDA out of memory. Tried to allocate 498.00 MiB")
    assert is_oom_message("torch.OutOfMemoryError: HIP out of memory")
    assert is_oom_message("RuntimeError: CUDA error: out of memory")  # cuda + oom words
    assert not is_oom_message("out of memory")            # untyped — could be the CPU
    assert not is_oom_message("MemoryError: allocation failed")
    assert not is_oom_message("")
    assert not is_oom_message(None)

    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"))
    assert not is_cuda_oom(RuntimeError("gpu on fire"))
    assert not is_cuda_oom(TypeError("unexpected keyword argument 'max_new_tokens'"))
    assert not is_cuda_oom(None)

    # The probe must never raise, torch or no torch, CUDA or no CUDA — it is called at
    # the head of two entry points whose job is to boot, not to be taken down by a guard.
    lines: list = []
    state = ensure_expandable_segments(log=lines.append)
    assert state in (None, True, False)
    # Without CUDA it returns None silently; with CUDA it must have said something.
    if state is not None:
        assert any("[alloc]" in ln for ln in lines), lines

    print("core.alloc_guard selftest OK")


if __name__ == "__main__":
    _selftest()
