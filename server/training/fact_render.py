"""Fact anchors → weights, the CoT-safe way (injection), plus the parked regeneration path.

A fact anchor is a bare self-statement — ``"Artemy is building me as a single-user
companion"`` — not a ``prompt → response`` pair, and **not** safe to train answer-only: an
answer-only target renders an empty reasoning channel on gemma-4 and erodes CoT over cycles
(the exact regression ``train_cycle``/``render.has_cot`` now guard against).

**Live path — CoT injection (this is what runs).** Mirroring ``persona_render``, a fact
reaches the weights by being injected as an ``"I know that …"`` line into the ``<think>``
of a real dialogue exchange that is *already being trained this cycle* — the host the
reflection-run **fact-placement judge** assigned it (``source_exchange``). Structurally a
dialogue example (the proven-safe vector), so no CoT-channel erosion; model-free, so it
runs in the render phase and is unit-tested without a GPU. :func:`fact_cot_line` builds the
injected line; ``train_cycle`` does the selection/dedup (reusing ``persona_render``) and the
cumulative-copy cap (``FACT_TRAIN_CAP``). See documentation/AVA_DESIGN.md → *Consolidation And Training*.

**Parked path — response regeneration (below, unused).** The original design posed the
statement back to Ava for a fresh ``<think>…</think>`` reply and trained that. It is slow,
not robust, and — via the synthetic single-turn self-interview — was the sole CoT-erosion
vector (see the memory ``fact-persona-cot-erosion``), so it was superseded by injection and
is kept only for reference / ``selftest`` coverage. ``train_cycle._render_fact_examples`` is
not called.
"""

from __future__ import annotations

from typing import Optional


# ── live path: CoT injection ───────────────────────────────────────────────── #

def fact_cot_line(statement: str) -> str:
    """The ``<think>``-line form of a fact — ``"I know that <statement>."``.

    Frames the fact as settled knowledge Ava holds (not a thing to look up), so the injected
    line reads as grounding for the reasoning that follows rather than a recital. Idempotent
    on an already-framed statement, tolerant of trailing punctuation, and a no-op on empty
    input. Dedup/selection against the CoT is the caller's job (``persona_render``'s
    incremental selector), keyed on the raw *statement*, so the framing here never defeats it.
    """
    s = (statement or "").strip()
    if not s:
        return ""
    # Keep the statement's own casing (it often starts with a name — "Artemy …"), so this
    # only prefixes and ensures terminal punctuation; already-framed statements pass through.
    core = s if s[:11].lower() == "i know that" else f"I know that {s}"
    return core if core.endswith((".", "!", "?")) else core + "."


# Embedder anchoring. A regenerated reply is accepted only when its answer is at least
# this semantically close to the statement (it really expresses the belief) and not so
# close it is a near-verbatim echo (which trains rote recall of one phrasing). Same
# all-MiniLM family RAG uses; first-guess thresholds (see DESIGN.md).
ANCHOR_FLOOR = 0.45
ANCHOR_CEILING = 0.97

# Per-type elicitation phrasings. The statement is given as context and Ava is asked to
# speak *from* it in her own voice with her own reasoning; rotating phrasings across an
# anchor's variant count keeps a single prompt from being overfit. The prompt is loss-
# masked (response-only), so these never become generation targets — only conditioning.
_PERSONA_PROMPTS = [
    'Something you have come to hold about yourself: "{s}". Say it in your own voice — '
    "what it means to you, and why it is yours.",
    'You once put it like this: "{s}". Speak to that now — is it still how you see things?',
    'This is true of who you are: "{s}". Put it the way you would actually say it.',
]
_FACT_PROMPTS = [
    'Something you know and no longer need to look up: "{s}". Show that you hold it — '
    "speak to it as settled knowledge.",
    'You have learned this: "{s}". Bring it to mind and say what it means in context.',
    'A stable truth for you: "{s}". Reason about it briefly, then state it plainly.',
]


def elicitation_prompt(statement: str, item_type: str, variant: int = 0) -> str:
    """The user-turn text that poses *statement* back to Ava for regeneration.

    *variant* selects (round-robin) among the per-type phrasings so the same statement
    is elicited differently across its decay-driven variant copies.
    """
    pool = _PERSONA_PROMPTS if item_type == "persona" else _FACT_PROMPTS
    return pool[variant % len(pool)].format(s=(statement or "").strip())


def build_fact_anchor(
    statement: str,
    item_type: str,
    *,
    key: str,
    stage: int,
    system_prompt: str,
    speaker: str = "",
    variant: int = 0,
) -> dict:
    """An anchor shaped like a dialogue anchor (so ``render.build_messages`` works),
    with an empty ``target`` to be filled by the regenerated reply.

    ``context`` is empty: a self-statement has no prior turns. The ``prompt`` is the
    elicitation; the ``target`` is set later from the model's ``<think>…</think>`` reply.
    """
    return {
        "key": key,
        "type": item_type,                       # "fact" | "persona"
        "system_prompt": system_prompt or "",
        "context": [],
        "prompt": elicitation_prompt(statement, item_type, variant),
        "speaker": speaker or "",
        "target": "",
        "statement": (statement or "").strip(),
        "stage": max(0, int(stage)),
        "variant": variant,
    }


def candidate_accepted(
    statement: str, answer: str, embedder, *,
    floor: float = ANCHOR_FLOOR, ceiling: float = ANCHOR_CEILING,
) -> bool:
    """True when a regenerated *answer* genuinely expresses *statement* without echoing it.

    Pure given an *embedder* with the sentence-transformers ``encode`` interface, so a
    fake embedder makes this unit-testable. Returns False on any embedding failure
    (best-effort: a candidate we cannot score is not trained).
    """
    statement = (statement or "").strip()
    answer = (answer or "").strip()
    if not statement or not answer:
        return False
    try:
        vecs = embedder.encode([statement, answer], convert_to_numpy=True,
                               normalize_embeddings=True)
        sim = float(vecs[0] @ vecs[1])
    except Exception:
        return False
    return floor <= sim < ceiling


def assemble_example(anchor: dict, full_target: str) -> Optional[dict]:
    """Build a render example from an anchor and a regenerated ``<think>…</think>`` reply.

    Returns ``None`` when *full_target* has no usable answer span (so the caller drops it
    rather than rendering a degenerate target). Mirrors the dialogue example shape used by
    ``train_cycle._render_examples`` / ``render.build_messages``.
    """
    from training import render
    target = (full_target or "").strip()
    if not target or not render.has_cot(target) or render.trainable_answer(target) is None:
        return None
    a = dict(anchor)
    a["target"] = target
    render.assert_parity(a, target)              # train/inference parity guard
    return {"anchor_key": anchor["key"],
            "messages": render.build_messages(a, target),
            "variant": target,
            "fact_key": anchor["key"]}
