"""Python 3.14 compatibility shim for ``datasets`` 4.3.0.

CPython 3.14 changed ``pickle.Pickler._batch_setitems`` to take the owning
object as a second positional argument (``_batch_setitems(items, obj)`` — see
CPython gh-100112). ``datasets.utils._dill.Pickler`` overrides that method with
the pre-3.14 ``(self, items)`` signature and forwards only ``items`` to
``super()``, so any dataset fingerprinting (``Dataset.from_dict`` →
``generate_fingerprint`` → dill) crashes on 3.14 with:

    TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given

Upstream fixed this in datasets 4.4.0 by making the override
``(self, items, *args, **kwargs)`` and forwarding the extra args
(huggingface/datasets #7839). But unsloth 2026.7.2 pins ``datasets<4.4.0``, so
we cannot take that release without breaking the blessed training stack. This
shim re-applies the identical upstream fix to the installed 4.3.0, preserving
the module's key-order-ignoring behaviour, so the pin and Python 3.14 coexist.

Import this module for its side effect **before** any ``datasets`` fingerprint
runs (i.e. before ``Dataset.from_dict``). Idempotent and a no-op on Python
< 3.14 or on a ``datasets`` that already carries the fix.
"""

from __future__ import annotations

import sys


def apply() -> bool:
    """Patch datasets' dill Pickler for Python 3.14. Returns True if patched."""
    if sys.version_info < (3, 14):
        return False

    try:
        from datasets.utils import _dill
    except Exception:
        return False

    Pickler = getattr(_dill, "Pickler", None)
    if Pickler is None:
        return False

    # Already fixed upstream (>=4.4.0) — detect by an accepted extra positional arg.
    import inspect

    try:
        params = inspect.signature(Pickler._batch_setitems).parameters
    except (TypeError, ValueError):
        params = {}
    has_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
    )
    # (self, items) == 2 named params and no *args → the broken pre-4.4.0 form.
    if has_varargs or len(params) > 2:
        return False

    _orig = Pickler._batch_setitems

    def _batch_setitems(self, items, *args, **kwargs):  # noqa: ANN001
        # Reproduce the original key-order-ignoring sort, then forward whatever
        # extra positional args (e.g. 3.14's ``obj``) the caller passed on.
        legacy = getattr(self, "_legacy_no_dict_keys_sorting", False)
        if not legacy:
            try:
                items = sorted(items)
            except Exception:  # TypeError, decimal.InvalidOperation, etc.
                from datasets.fingerprint import Hasher

                items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        # Skip datasets' own (broken) override; go straight to dill's Pickler,
        # which on 4.3.0 already accepts the 3.14 signature.
        import dill

        return dill.Pickler._batch_setitems(self, items, *args, **kwargs)

    _batch_setitems._ava_py314_patch = True  # marker for idempotency/inspection
    Pickler._batch_setitems = _batch_setitems
    return not getattr(_orig, "_ava_py314_patch", False)


_PATCHED = apply()
