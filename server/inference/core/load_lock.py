"""One model load at a time in this process — transformers' loader is not thread-safe.

``from_pretrained`` (transformers 5.x) swaps ``PreTrainedModel.tie_weights`` and
``init_weights`` — and the torch init functions — for empty stubs at CLASS / module level
while it loads (``initialization.no_tie_weights`` / ``no_init_weights``), and on the way out
puts back whatever it found there. Two loads that overlap in two threads can put back each
other's stub: the first to enter saves the real function, the second saves the first one's
stub; the first leaves and restores the real one, the second leaves and restores the stub.
From then on no model this process loads ties its weights. Gemma-4's checkpoint has no
``lm_head`` (it is the embedding matrix, tied), so ``lm_head`` stays on the meta device,
unsloth's automatic embedding offload moves the (now untied) embedding to RAM on top, and
the backend's materialization check refuses the load — every later load too, the restore
of the previous model included, until the process restarts.

Observed 2026-09-30: a Sleep run with nothing new to reflect reached its clean-base swap
within a second of the staging RAG's index build loading its MiniLM embedder in a
background thread; the swap, both restores and the run were lost, and the model-loss hook
restarted the server. Read again, the 2026-09-17 failure (the same refusal with 28 GiB
free, blamed on accelerate's placement planner) has the same shape. Two concurrent CPU
loads of the MiniLM leave ``tie_weights`` stubbed in about half the attempts.

:data:`LOCK` serializes the loads this process makes — the backend's model, the RAG
embedders, the persona digest's, the associative library's. :func:`heal` puts the real
functions back if a load outside the lock left a stub anyway; the backend runs it, under
the lock, before each load. The real functions are remembered then (:func:`capture`,
never a stub), from transformers as it is already imported: this module imports nothing
itself, so it cannot import transformers ahead of unsloth.

Leaf module: imports nothing from the project, and transformers only in its self-test.
``python -m core.load_lock`` is the GPU-free self-test.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional

# Re-entrant: a loader may call another loader (a backend load that builds a RAG view).
LOCK = threading.RLock()

_GENUINE: dict = {}          # (id(owner), attr) -> (owner, attr, function)
_STUB = "<locals>.empty_func"


def is_stub(fn) -> bool:
    """Is *fn* one of the empty functions transformers' loader patches in while it loads?"""
    return _STUB in str(getattr(fn, "__qualname__", "") or "")


def _targets() -> list:
    """The (owner, attribute) pairs transformers' loader patches: the model class's
    ``tie_weights`` / ``init_weights``, and the torch init functions ``no_init_weights``
    empties. Read from transformers as already imported (empty before it is)."""
    mu = sys.modules.get("transformers.modeling_utils")
    model_cls = getattr(mu, "PreTrainedModel", None) if mu is not None else None
    if model_cls is None:
        return []
    out = [(model_cls, "tie_weights"), (model_cls, "init_weights")]
    init = sys.modules.get("transformers.initialization")
    for name in getattr(init, "TORCH_MODULES_TO_PATCH", ()) or ():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for fname in getattr(init, "TORCH_INIT_FUNCTIONS", {}) or {}:
            if hasattr(mod, fname):
                out.append((mod, fname))
    return out


def _label(owner, attr: str) -> str:
    return f"{getattr(owner, '__name__', type(owner).__name__)}.{attr}"


def capture() -> int:
    """Remember the real loader functions (skipping any that is a stub right now); the
    count remembered. :func:`heal` calls it under the lock, where no guarded load runs."""
    n = 0
    for owner, attr in _targets():
        fn = getattr(owner, attr, None)
        if fn is None or is_stub(fn):
            continue
        _GENUINE.setdefault((id(owner), attr), (owner, attr, fn))
        n += 1
    return n


def heal(log: Optional[Callable[[str], None]] = None) -> list:
    """Put back every real loader function a racing load left as a stub; the names put
    back (empty when nothing was wrong). Remembers the real ones first (:func:`capture`).
    Only while no load holds :data:`LOCK` is a stub a leftover — callers hold the lock."""
    capture()
    healed = []
    for owner, attr in _targets():
        fn = getattr(owner, attr, None)
        if fn is None or not is_stub(fn):
            continue
        genuine = _GENUINE.get((id(owner), attr))
        if genuine is None:
            continue
        setattr(owner, attr, genuine[2])
        healed.append(_label(owner, attr))
    if healed and log is not None:
        log(f"[load] a concurrent from_pretrained had left {', '.join(healed)} stubbed — put back "
            f"(core/load_lock.py)")
    return healed


def stubbed() -> list:
    """The loader functions that are stubs right now (a diagnostic; nothing is changed)."""
    return [_label(o, a) for o, a in _targets() if is_stub(getattr(o, a, None))]


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    import os
    import tempfile

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    failures: list = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            failures.append(name)

    from transformers import BertConfig, BertModel
    from transformers.modeling_utils import PreTrainedModel

    print("capture and stub detection")
    check("the real functions are remembered once transformers is imported",
          capture() >= 2 and len(_GENUINE) >= 2, True)
    check("nothing is stubbed now", stubbed(), [])

    def no_tie_weights():
        def empty_func(*args, **kwargs):
            pass
        return empty_func

    real = PreTrainedModel.tie_weights
    PreTrainedModel.tie_weights = no_tie_weights()
    check("a leftover stub is seen", (is_stub(PreTrainedModel.tie_weights), stubbed()),
          (True, ["PreTrainedModel.tie_weights"]))
    said: list = []
    check("heal puts the real one back, and says so",
          (heal(said.append), PreTrainedModel.tie_weights is real, bool(said)),
          (["PreTrainedModel.tie_weights"], True, True))
    check("…and a second heal has nothing to do", heal(), [])

    print("two loads in two threads, with and without the lock")
    with tempfile.TemporaryDirectory() as td:
        cfg = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=32)
        BertModel(cfg).save_pretrained(td)

        def race(locked: bool, trials: int) -> int:
            stuck = 0
            for _ in range(trials):
                barrier = threading.Barrier(2)

                def load():
                    barrier.wait()
                    if locked:
                        with LOCK:
                            BertModel.from_pretrained(td)
                    else:
                        BertModel.from_pretrained(td)

                threads = [threading.Thread(target=load) for _ in range(2)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                if stubbed():
                    stuck += 1
                    heal()
            return stuck

        unlocked = race(False, 40)
        print(f"  (without the lock: a stub left behind after {unlocked} of 40 concurrent pairs)")
        check("with the lock: never", race(True, 40), 0)

    print("\nload_lock self-test: " + ("OK" if not failures else f"{len(failures)} FAILED — {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
