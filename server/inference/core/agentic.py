"""Agentic tasks Ava dispatches to a CLEAN base model (adapter + RAG OFF).

SKETCH — the pure task layer + self-test run GPU-free today; ``CleanBaseSession``
is correct-by-construction but its load/release path must be exercised once on the
GPU box before it is wired into the live reflection loop (see module self-test and
AVA_DESIGN_LEGACY.md).

Why a clean base
----------------
Ava's identity lives in the LoRA adapter, and the persona pipeline deliberately
trains dispositions, refusals, and a voice ("buy yourself a calculator and don't
bother me"). That is correct for Ava-the-self, but it makes the adapter-bearing
model an unreliable *tool*: a mechanical sub-task ("extract the subject of this
question") may be answered in character, deflected, or refused, with no clean
parse. So tool-shaped steps run against the frozen base with the adapter swapped
out entirely — the obedient instruction-follower "before Ava became Ava".

This also keeps the lookup deterministic w.r.t. the frozen base: extraction does
not drift with whichever adapter generation happens to be loaded.

Layering
--------
* ``CleanBaseSession`` — context manager that swaps the loaded adapter model for
  the bare base and restores it on exit. Full unload + reload (correctness over
  speed; runs offline during reflection, no user waiting). It does NOT rely on a
  PEFT adapter-toggle, which unsloth may have merged away after ``for_inference``.
* ``AgenticTask`` / ``TASKS`` — pure (prompt, parse) units depending only on a
  ``generate`` callable, so they are GPU-free testable with a fake generate.
* ``run_task`` — run one registered task against a provided generate fn.

Adding a task later means registering an ``AgenticTask`` — no new GPU plumbing.

GPU-free self-test: ``python -m core.agentic``.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


# ──────────────────────────────────────────────────────────────────────────────
# Model-loss escalation
# ──────────────────────────────────────────────────────────────────────────────

# Called when a swap has released the loaded model and could NOT bring one back — not
# the swap target and not the previous model, after the retry below. Injected by
# ``server.main()`` (``configure``); it requests a watchdog restart, which relaunches
# the process on the config's ``adapter_id`` — the last adapter that was promoted,
# i.e. the last known good one. This module stays GPU-free and import-light, so the
# hook is a plain callable returning a one-line note of what it did (or ``None``).
_on_model_lost: Optional[Callable[[str], Optional[str]]] = None
_last_model_loss: Optional[dict] = None


def configure(*, on_model_lost: Optional[Callable[[str], Optional[str]]] = None) -> None:
    global _on_model_lost
    _on_model_lost = on_model_lost


def last_model_loss() -> Optional[dict]:
    """``{"reason", "recovery"}`` of the most recent unrecovered swap failure in this
    process, or ``None``. ``recovery`` is the note the escalation hook returned — what
    a caller reporting "no model is loaded" should tell the operator to expect."""
    return dict(_last_model_loss) if _last_model_loss else None


def _model_lost(reason: str, log: Callable[[str], None]) -> None:
    global _last_model_loss
    _last_model_loss = {"reason": reason, "recovery": None}
    log(f"[agentic] NO MODEL IS LOADED: {reason}")
    if _on_model_lost is None:
        log("[agentic] no model-loss handler configured — reload from the client.")
        return
    try:
        note = _on_model_lost(reason)
    except Exception as exc:
        note = f"model-loss handler failed: {type(exc).__name__}: {exc}"
    if note:
        _last_model_loss["recovery"] = str(note)
        log(f"[agentic] {note}")


def _drop_tracebacks(exc: BaseException) -> None:
    """Sever the traceback from *exc* and every exception chained under it.

    A traceback pins its frames, and a load that failed in ``backend.load`` has the
    partially-materialized model in exactly those frames — tens of GB that stay
    resident until the traceback is gone. Dropping it once on the outer exception is
    not enough: a load that failed while handling an earlier failure carries that
    earlier one as ``__context__``, with its own traceback and its own frames."""
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur.__traceback__ = None
        cur = cur.__cause__ or cur.__context__


# ──────────────────────────────────────────────────────────────────────────────
# Generate contract
# ──────────────────────────────────────────────────────────────────────────────

class GenerateFn(Protocol):
    """The subset of the server's reflect-generate the tasks rely on.

    ``server._make_sync_reflect_generate(...)`` already matches this signature, so
    the server passes its existing generate straight through — once the clean base
    is swapped into the shared ``ModelRuntime`` it reads that state at call time and
    therefore runs on the bare base automatically.
    """

    def __call__(
        self,
        content: str,
        system_prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens_setting: str,
        disable_rag: bool = False,
    ) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# Clean-base swap (GPU)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanBase:
    """Handle to the currently-loaded swapped model (returned by the swap paths)."""
    model: Any
    tokenizer: Any
    model_id: str
    context_length: int


def _reclaim_backend(backend: Any) -> None:
    """Collect + empty the CUDA cache once the caller's own references are gone.

    Optional on the backend (``InferenceBackend.reclaim`` is a no-op default), so a
    duck-typed backend that predates it still satisfies the contract."""
    fn = getattr(backend, "reclaim", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


def _load_model_into_state(backend: Any, state: Any, model_id: str,
                           context_length: int, adapter_id: Optional[str],
                           reflect_context_length: Optional[int],
                           prepare: Callable[[Any, str], None]) -> CleanBase:
    """Load ``model_id`` (+ optional adapter) and write it into the shared runtime.

    The physical model is loaded at max(chat, reflect); BOTH runtime budgets are
    preserved so a swap never silently shrinks the physical window (which reflection
    may be packing to)."""
    reflect_ctx = max(context_length,
                      context_length if reflect_context_length is None
                      else int(reflect_context_length))
    load_ctx = max(context_length, reflect_ctx)
    model, tok = backend.load(model_id, load_ctx, adapter_id)
    prepare(tok, model_id)
    state.model, state.tokenizer = model, tok
    state.model_id, state.adapter_id = model_id, adapter_id
    state.context_length = context_length
    state.reflect_context_length = reflect_ctx
    return CleanBase(model, tok, model_id, load_ctx)


def swap_model(backend: Any, model_state: Any, *, adapter_id: Optional[str],
               prepare: Optional[Callable[[Any, str], None]] = None,
               on_log: Optional[Callable[[str], None]] = None) -> CleanBase:
    """ONE-WAY swap: release the loaded model, load the same base with *adapter_id*
    (``None`` = bare base), and leave it loaded. Nothing restores the previous model —
    the shared runtime simply now holds the new one, honestly reported (``status``
    serves ``model_state.adapter_id``); loading from the client or a server restart
    (the config's ``adapter_id`` is never touched) is what undoes it.

    This is the Training-review regenerate flow's swap: a repair session is a RUN of
    regenerations under one chosen lineage adapter, so restoring after each re-answer
    (the first design) cost two full reloads per row for a restore the very next row
    undid. ``CleanBaseSession.__enter__`` delegates here, so both swaps share ONE
    definition of release → reclaim → load and one failure discipline: a failed load
    restores the PREVIOUS model — the original failure logged and its traceback
    DROPPED first (those frames pin the failed load's partially-materialized tensors),
    then, OUTSIDE the except block, a reclaim and the restore (``_restore_previous``:
    once more after a second reclaim if the first attempt fails), so an OOM's debris
    can't cascade into the restore — and re-raises the original error; a restore that
    ALSO fails is escalated through the model-loss hook (``configure``) while the
    original error still propagates (the restore failure is usually its echo).

    Executor-thread only, exclusive GPU — and the caller must hold NO reference to the
    outgoing model in its own frame (``release()``'s ``del`` reaches only its own
    locals; a frame-held reference keeps the released model resident through the load)."""
    prepare = prepare or (lambda tok, mid: None)
    log = on_log or (lambda m: None)
    st = model_state
    if st.model is None:
        raise RuntimeError("swap_model: no model loaded to swap from")
    prev_adapter = st.adapter_id
    model_id = st.model_id
    ctx = st.context_length
    reflect_ctx = st.reflect_context_length

    log("[agentic] releasing loaded model to free VRAM…")
    old_model, old_tok = st.model, st.tokenizer
    st.model, st.tokenizer = None, None
    backend.release(old_model, old_tok)
    # Drop OUR references before loading — see the release() contract above.
    old_model = old_tok = None
    _reclaim_backend(backend)

    if adapter_id:
        log(f"[agentic] loading {model_id} with adapter {adapter_id}…")
    else:
        log(f"[agentic] loading clean base {model_id} (adapter OFF)…")
    failed: Optional[BaseException] = None
    try:
        return _load_model_into_state(backend, st, model_id, ctx, adapter_id,
                                      reflect_ctx, prepare)
    except Exception as exc:
        log("[agentic] swap load failed; restoring the previous model…")
        traceback.print_exc()
        _drop_tracebacks(exc)
        failed = exc
    # The restore runs OUTSIDE the except block, deliberately. Inside it, the
    # interpreter's own exception state still references the traceback — on Python
    # 3.10 as a separate ``exc_info`` slot that ``exc.__traceback__ = None`` does not
    # touch — so the failed load's frames, and the ~20 GB of weights they hold, stayed
    # resident through the restore. Observed 2026-09-17 on the RTX 5090: the clean-base
    # load was refused by the offload guard with 28 GiB free, and the restore then ran
    # with 10 GiB free and was refused the same way, leaving the box with no model.
    _restore_previous(backend, st, model_id, ctx, prev_adapter, reflect_ctx, prepare, log)
    raise failed


def _restore_previous(backend: Any, st: Any, model_id: str, ctx: int,
                      adapter_id: Optional[str], reflect_ctx: Optional[int],
                      prepare: Callable[[Any, str], None],
                      log: Callable[[str], None]) -> bool:
    """Reload the previous model after a failed swap; ``True`` when a model is loaded.

    Reclaim → load, and on failure reclaim again — outside that failure's except block,
    for the reason given at the call site — and try ONCE more, since the first attempt
    may have run against the swap's debris. A second failure is escalated through the
    configured model-loss hook (a watchdog restart onto the config's adapter); nothing is
    raised here, the caller decides what its own contract says."""
    for attempt in (1, 2):
        _reclaim_backend(backend)
        try:
            _load_model_into_state(backend, st, model_id, ctx, adapter_id, reflect_ctx,
                                   prepare)
            return True
        except Exception as exc:
            log(f"[agentic] restore attempt {attempt} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            _drop_tracebacks(exc)
            reason = (f"restoring {model_id} (adapter {adapter_id or 'OFF'}) failed twice "
                      f"after a failed swap: {type(exc).__name__}: {exc}")
    _model_lost(reason, log)
    return False


class CleanBaseSession:
    """Swap the loaded adapter model for the bare base, restore on exit.

    Correctness over performance: the adapter model is fully released (VRAM freed)
    and the base reloaded from disk with ``adapter_id=None``; the reverse runs on
    exit. No reliance on PEFT adapter toggling. Two full model loads per session
    make this slow — acceptable because agentic tasks run offline during reflection.

    ``swap_adapter_id`` generalizes the session from "adapter OFF" to "a DIFFERENT
    adapter ON": the same release → load → restore discipline, but the interim load
    attaches the named adapter instead of the bare base. Used by the Training-review
    regenerate flow so an operator can re-answer a corrupt exchange with a *previous*
    (last-known-good) adapter from the lineage while the box returns to the current
    one afterward. Default ``None`` keeps the historical clean-base behaviour exactly.

    Contract:
      * MUST run on the single GPU executor thread with exclusive access — no
        concurrent ``generate`` / reflection pass (both read the ``ModelRuntime``).
      * ``backend`` duck-types ``UnslothBackend``: ``load(model_id, ctx, adapter_id)
        -> (model, tokenizer)`` and ``release(model, tokenizer)``, plus an optional
        ``reclaim()`` called once this frame's references to the released model are
        gone (a quantized model cannot be moved to CPU, so dropping the last reference
        is the only eviction — and ``release``'s own ``del`` cannot reach ours).
      * ``model_state`` is the server's shared ``ModelRuntime`` (core.runtime_state);
        its attributes are mutated in place so the existing generate path picks up
        the swapped model.
      * ``prepare(tokenizer, model_id)`` (optional) mirrors the server's
        ``_do_load`` post-load fixups (pad token, ensure_chat_template).

    Failure safety: if loading the clean base fails mid-enter, the original adapter
    model is reloaded before the error propagates (retried once after a real reclaim);
    if THAT fails too — on enter or on exit — the model-loss hook is called, which on a
    watchdog-run box requests a restart onto the config's adapter. See ``swap_model``.
    """

    def __init__(
        self,
        backend: Any,
        model_state: Any,   # the shared ModelRuntime (core.runtime_state); mutated in place
        *,
        prepare: Optional[Callable[[Any, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        swap_adapter_id: Optional[str] = None,
    ) -> None:
        self._backend = backend
        self._state = model_state
        self._prepare = prepare or (lambda tok, mid: None)
        self._log = on_log or (lambda m: None)
        self._swap_adapter_id = swap_adapter_id
        self._saved_spec: Optional[dict] = None

    def _load_into_state(self, model_id: str, context_length: int,
                         adapter_id: Optional[str],
                         reflect_context_length: Optional[int] = None) -> CleanBase:
        return _load_model_into_state(self._backend, self._state, model_id,
                                      context_length, adapter_id,
                                      reflect_context_length, self._prepare)

    def _reclaim(self) -> None:
        _reclaim_backend(self._backend)

    def __enter__(self) -> CleanBase:
        st = self._state
        if st.model is None:
            raise RuntimeError("CleanBaseSession: no model loaded to swap from")

        # Spec only — never retain the live objects, so releasing actually frees them.
        self._saved_spec = {
            "model_id": st.model_id,
            "adapter_id": st.adapter_id,
            "context_length": st.context_length,
            "reflect_context_length": st.reflect_context_length,
        }
        # The swap itself — release → reclaim → load, failure-restore included (a
        # failed load restores the previous model and re-raises the original error) —
        # is the shared one-way primitive; this context manager adds only the saved
        # spec and the restore-on-exit.
        try:
            return swap_model(self._backend, st, adapter_id=self._swap_adapter_id,
                              prepare=self._prepare, on_log=self._log)
        except Exception:
            self._saved_spec = None
            raise

    def __exit__(self, exc_type, exc, tb) -> bool:
        st = self._state
        spec = self._saved_spec or {}
        self._saved_spec = None

        self._log("[agentic] releasing swapped-in model…" if self._swap_adapter_id
                  else "[agentic] releasing clean base…")
        clean_model, clean_tok = st.model, st.tokenizer
        st.model, st.tokenizer = None, None
        try:
            self._backend.release(clean_model, clean_tok)
        except Exception:
            pass
        clean_model = clean_tok = None   # ...and our own refs — see __enter__
        self._reclaim()

        adapter_id = spec.get("adapter_id")
        self._log(
            f"[agentic] reloading adapter model {spec.get('model_id')} "
            f"(adapter {'ON' if adapter_id else 'OFF'})…"
        )
        # Same restore discipline as a failed swap (retry once after a real reclaim,
        # escalate on a second failure) — and then let the failure propagate: a server
        # with no model is a loud, actionable error, not something to swallow.
        ok = _restore_previous(self._backend, st, spec["model_id"], spec["context_length"],
                               adapter_id, spec.get("reflect_context_length"),
                               self._prepare, self._log)
        if not ok:
            loss = last_model_loss() or {}
            raise RuntimeError(
                f"CleanBaseSession: restoring the previous model failed — no model is "
                f"loaded. {loss.get('recovery') or 'Reload from the client.'}")
        return False  # never suppress the body's exception


# ──────────────────────────────────────────────────────────────────────────────
# Task framework (pure — GPU-free testable)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgenticTask:
    """One mechanical task run against the clean base.

    ``build_user(payload) -> str`` renders the user message; ``parse(raw) -> Any``
    turns the model's cleaned answer into a structured result. Sampling defaults
    are cold (low temperature) because these are tool calls, not Ava's judgement.
    """
    name: str
    system: str
    build_user: Callable[[Any], str]
    parse: Callable[[str], Any]
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: str = "512"

    def run(self, generate: GenerateFn, payload: Any) -> Any:
        raw = generate(
            self.build_user(payload), self.system,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens_setting=self.max_new_tokens,
            disable_rag=True,   # tools never read Ava's memory
        )
        return self.parse(raw or "")


def run_task(generate: GenerateFn, task_name: str, payload: Any) -> Any:
    """Run a registered task against ``generate``. Raises KeyError on unknown task.

    The caller is responsible for having entered a ``CleanBaseSession`` first (so
    ``generate`` is bound to the bare base). Kept separate from the swap so task
    parsing is unit-testable without a GPU.
    """
    task = TASKS[task_name]
    return task.run(generate, payload)


# ──────────────────────────────────────────────────────────────────────────────
# Task: extract_subjects — pull Wikipedia-lookup subjects out of [ask:search] text
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "You are a precise subject-extraction tool. For each numbered question, identify "
    "the single entity a Wikipedia article would be titled after — a person, place, "
    "organization, or named event — using its canonical full name (e.g. 'Abelardo de "
    "la Espriella', not 'de la Espriella' or 'the winner').\n\n"
    "CRITICAL: Do NOT judge whether the article exists, is recent, or is a future "
    "event. Your own knowledge may be out of date, and a SEPARATE step checks "
    "Wikipedia — so always extract the most likely article title from the wording, "
    "even for something you have never heard of (a question about the '2026 Iran war' "
    "yields '2026 Iran war'; a '21st Century ROAD to Housing Act' yields that exact "
    "name). Use NONE ONLY when the question names no lookupable entity at all — e.g. "
    "a question purely about your own feelings or an abstract idea.\n\n"
    "You may reason briefly first, but your FINAL output MUST be exactly one line per "
    "question, in the original order, each line formatted EXACTLY as:\n"
    "    SUBJECT: <canonical name>\n"
    "or, only when there is genuinely no named entity:\n"
    "    SUBJECT: NONE\n\n"
    "Put the SUBJECT: lines LAST, with nothing after them. Do not answer the "
    "questions, do not explain on the SUBJECT: lines, do not add any other text there."
)


def _build_extract_user(asks: list[str]) -> str:
    lines = ["Questions:"]
    for i, q in enumerate(asks, 1):
        lines.append(f"{i}. {' '.join((q or '').split())}")
    lines.append("")
    lines.append(
        f"Now output exactly {len(asks)} line(s), one per question in order, each "
        "'SUBJECT: <name>' or 'SUBJECT: NONE'."
    )
    return "\n".join(lines)


# Only lines of the strict form 'SUBJECT: <value>' are accepted — any reasoning the
# model emits around them is ignored, so a chatty/CoT model can't leak prose as
# subjects (the failure that turned 'Result: NONE.' into a bogus 'Result, New York').
_SUBJECT_RE = re.compile(r"^\s*SUBJECT\s*:\s*(.+?)\s*$", re.IGNORECASE)

# A canonical article title is short; anything longer is almost certainly leaked
# reasoning that happened to start with the label, so reject it.
_MAX_SUBJECT_LEN = 100


def _parse_subjects(raw: str) -> list[str]:
    """Strict: take only 'SUBJECT: <name>' lines, drop NONE / empty / over-long, dedup.

    Returns [] when the model emitted no SUBJECT: line at all — better to extract
    nothing than to scrape a reasoning dump and fetch garbage."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        m = _SUBJECT_RE.match(line)
        if not m:
            continue
        val = m.group(1).strip().strip('"').strip("'").strip()
        if not val or val.upper() == "NONE" or len(val) > _MAX_SUBJECT_LEN:
            continue
        key = val.casefold()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


TASKS: dict[str, AgenticTask] = {
    "extract_subjects": AgenticTask(
        name="extract_subjects",
        system=_EXTRACT_SYSTEM,
        build_user=_build_extract_user,
        parse=_parse_subjects,
        temperature=0.2,   # extraction wants determinism
        max_new_tokens="512",   # headroom so a brief preamble can't truncate the block
    ),
    # Future tasks register here, e.g.:
    #   "shape_search_query", "rank_article_relevance", "summarize_article".
}


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    # Fake generate: a chatty/CoT model that reasons in prose first, then emits the
    # strict SUBJECT: block — the real-world failure mode (reasoning must be ignored).
    def fake_generate(content, system_prompt, **kw):
        assert kw.get("disable_rag") is True, "agentic tasks must run RAG-off"
        # Extract-don't-judge: future-dated subjects ARE extracted (fetch decides
        # existence); NONE only for a genuinely subject-less question.
        return (
            "Let me reason through each question.\n"
            "Result: NONE.\n"          # reasoning prose — must NOT be scraped
            "Goal: extract subjects.\n"  # echoed instruction — must NOT be scraped
            "\n"
            "SUBJECT: 2026 Iran war\n"                 # future event still extracted
            "SUBJECT: Keir Starmer\n"
            'SUBJECT: "2026 European heatwaves"\n'   # quoted → unquoted
            "SUBJECT: NONE\n"                         # genuinely no named entity
            "SUBJECT: Abelardo de la Espriella\n"
            "subject: abelardo de la espriella\n"     # case-insensitive dup → deduped
        )

    asks = [
        "What were the causes of the '2026 Iran war'?",
        "What prompted Keir Starmer's resignation?",
        "How bad are the 2026 European heatwaves?",
        "What does it mean that I learn this way?",   # no named entity → NONE
        "Who is Abelardo de la Espriella?",
    ]
    subjects = run_task(fake_generate, "extract_subjects", asks)
    assert subjects == [
        "2026 Iran war",
        "Keir Starmer",
        "2026 European heatwaves",
        "Abelardo de la Espriella",
    ], subjects

    # Parser edge cases: only SUBJECT: lines count; reasoning/garbage is dropped.
    assert _parse_subjects("") == []
    assert _parse_subjects("Goal: blah\nAnalysis: x\nResult: NONE.") == []  # no SUBJECT:
    assert _parse_subjects("SUBJECT: NONE\nsubject:  none ") == []
    assert _parse_subjects("SUBJECT: " + "x" * 200) == []                   # over-long
    assert _parse_subjects("SUBJECT: Tokyo\nSUBJECT: Kyoto") == ["Tokyo", "Kyoto"]

    # ── CleanBaseSession swap/restore semantics, fake backend (no GPU) ──────────
    # Locks the failure contract the Training-review adapter swap leans on: a failed
    # swap load restores the previous model and re-raises the ORIGINAL error; a
    # restore that also fails still surfaces the original error, with the state
    # honestly empty rather than half-written.
    import contextlib, io

    class _FakeState:
        def __init__(self):
            self.model, self.tokenizer = "m0", "t0"
            self.model_id, self.adapter_id = "base", "A"
            self.context_length, self.reflect_context_length = 1024, 2048

    class _FakeBackend:
        def __init__(self, fail_on=(), fail_times=None):
            # fail_on: adapters whose load ALWAYS fails; fail_times: {adapter: n} — the
            # first n loads of that adapter fail, later ones succeed (a restore that
            # works once the failed swap's debris is actually gone).
            self.fail_on = set(fail_on)
            self.fail_times = dict(fail_times or {})
            self.loads, self.releases, self.reclaims = [], 0, 0

        def load(self, model_id, ctx, adapter_id=None, **kw):
            self.loads.append(adapter_id)
            if adapter_id in self.fail_on:
                raise RuntimeError(f"oom loading {adapter_id}")
            if self.fail_times.get(adapter_id, 0) > 0:
                self.fail_times[adapter_id] -= 1
                raise RuntimeError(f"oom loading {adapter_id} (transient)")
            return (f"model:{adapter_id}", "tok")

        def reclaim(self):
            self.reclaims += 1

        def release(self, model, tokenizer):
            self.releases += 1

    # One-way sticky swap (the Training-review regenerate path): loads the target and
    # LEAVES it loaded — no restore — so a repair session's next regeneration with the
    # same choice needs no reload at all.
    st, be = _FakeState(), _FakeBackend()
    cb = swap_model(be, st, adapter_id="B")
    assert st.adapter_id == "B" and st.model == "model:B", (st.adapter_id, st.model)
    assert cb.model_id == "base"
    assert be.loads == ["B"] and be.releases == 1, (be.loads, be.releases)

    # ...and its failed load restores the previous model, original error propagating.
    st, be = _FakeState(), _FakeBackend(fail_on={"B"})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            swap_model(be, st, adapter_id="B")
            raise AssertionError("swap should have failed")
        except RuntimeError as e:
            assert "oom loading B" in str(e), e
    assert st.adapter_id == "A" and st.model == "model:A", (st.adapter_id, st.model)

    # Clean round trip: enter loads the swap target, exit restores the original.
    st, be = _FakeState(), _FakeBackend()
    with CleanBaseSession(be, st, swap_adapter_id="B") as cb:
        assert st.adapter_id == "B" and st.model == "model:B", (st.adapter_id, st.model)
        assert cb.model_id == "base"
    assert st.adapter_id == "A" and st.model == "model:A", (st.adapter_id, st.model)
    assert be.loads == ["B", "A"] and be.releases == 2, (be.loads, be.releases)

    # And with no swap target the historical clean-base behaviour: adapter OFF.
    st, be = _FakeState(), _FakeBackend()
    with CleanBaseSession(be, st):
        assert st.adapter_id is None and st.model == "model:None"
    assert st.adapter_id == "A"

    # Swap load fails -> previous model restored, ORIGINAL error propagates.
    st, be = _FakeState(), _FakeBackend(fail_on={"B"})
    with contextlib.redirect_stderr(io.StringIO()):   # the logged traceback is expected
        try:
            with CleanBaseSession(be, st, swap_adapter_id="B"):
                raise AssertionError("enter should have failed")
        except RuntimeError as e:
            assert "oom loading B" in str(e), e
    assert st.adapter_id == "A" and st.model == "model:A", (st.adapter_id, st.model)
    assert be.loads == ["B", "A"], be.loads

    # A restore that fails ONCE (the swap's debris still resident) is retried after a
    # second reclaim and succeeds: the previous model is back, the hook never fires.
    global _last_model_loss
    _last_model_loss = None
    hook_calls: list = []
    configure(on_model_lost=lambda reason: (hook_calls.append(reason) or
                                            "restart requested (fake)"))
    st, be = _FakeState(), _FakeBackend(fail_on={"B"}, fail_times={"A": 1})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            swap_model(be, st, adapter_id="B")
            raise AssertionError("swap should have failed")
        except RuntimeError as e:
            assert "oom loading B" in str(e), e
    assert st.adapter_id == "A" and st.model == "model:A", (st.adapter_id, st.model)
    assert be.loads == ["B", "A", "A"], be.loads
    assert be.reclaims >= 3, be.reclaims            # release, then one per attempt
    assert hook_calls == [] and last_model_loss() is None

    # Restore fails twice -> original error STILL propagates (the restore failure is
    # its echo, logged not raised), the state is honestly empty, and the model-loss
    # hook fired exactly once with the recovery note recorded for callers to report.
    st, be = _FakeState(), _FakeBackend(fail_on={"A", "B"})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            with CleanBaseSession(be, st, swap_adapter_id="B"):
                raise AssertionError("enter should have failed")
        except RuntimeError as e:
            assert "oom loading B" in str(e), e
    assert st.model is None and st.tokenizer is None, (st.model, st.tokenizer)
    assert be.loads == ["B", "A", "A"], be.loads
    assert len(hook_calls) == 1 and "failed twice" in hook_calls[0], hook_calls
    loss = last_model_loss()
    assert loss and loss["recovery"] == "restart requested (fake)", loss

    # The exit-side restore has the same discipline: retried once, and a second
    # failure raises (a server with no model must not be silent) naming the recovery.
    hook_calls.clear()
    st, be = _FakeState(), _FakeBackend(fail_times={"A": 1})
    with contextlib.redirect_stderr(io.StringIO()):
        with CleanBaseSession(be, st, swap_adapter_id="B"):
            pass
    assert st.adapter_id == "A" and st.model == "model:A", (st.adapter_id, st.model)
    assert be.loads == ["B", "A", "A"] and hook_calls == [], (be.loads, hook_calls)
    st, be = _FakeState(), _FakeBackend(fail_on={"A"})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            with CleanBaseSession(be, st, swap_adapter_id="B"):
                pass
            raise AssertionError("exit should have failed")
        except RuntimeError as e:
            assert "no model is loaded" in str(e) and "restart requested (fake)" in str(e), e
    assert st.model is None and len(hook_calls) == 1, (st.model, hook_calls)
    # A hook that itself raises is contained: the loss is still recorded.
    configure(on_model_lost=lambda reason: (_ for _ in ()).throw(ValueError("boom")))
    st, be = _FakeState(), _FakeBackend(fail_on={"A", "B"})
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            swap_model(be, st, adapter_id="B")
        except RuntimeError:
            pass
    assert "boom" in (last_model_loss() or {}).get("recovery", ""), last_model_loss()
    configure(on_model_lost=None)
    _last_model_loss = None

    # Traceback severing walks the __context__ chain (a restore that failed while
    # handling the swap failure carries the swap failure, and ITS frames, underneath).
    try:
        try:
            raise RuntimeError("inner")
        except RuntimeError:
            raise ValueError("outer")
    except ValueError as outer:
        _drop_tracebacks(outer)
        assert outer.__traceback__ is None and outer.__context__.__traceback__ is None

    print("agentic self-test: OK")


if __name__ == "__main__":
    _selftest()
