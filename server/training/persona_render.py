"""Render persona anchors by injecting the self-statement at the start of its
triggering exchange's CoT — the CoT-safe alternative to response regeneration.

Why this exists (see documentation/AVA_CHANGELOG.md → *Negative Results And Rollbacks*, and the project
memory ``fact-persona-cot-erosion``): the previous fact/persona path posed each
statement back to Ava in a **synthetic single-turn self-interview** and trained the
regenerated reply. That example shape — empty context, a narrow repeated elicitation
prompt, regenerated targets — eroded the chain-of-thought channel on gemma-4 (the
turn-start "open a reasoning channel" prior drifts). A controlled cycle with fact/
persona forced empty kept CoT fully intact, isolating that path as the sole erosion
vector.

This module trains a persona statement the **safe** way: it reuses the real exchange
the persona was distilled from (a normal dialogue training example — real system
prompt, context, user turn, answer) and only **injects the statement as a leading line
inside that exchange's ``<think>`` block**. The result is structurally a *dialogue*
example (the proven-safe vector), so the turn-start channel-open prior is reinforced
exactly like dialogue; the only change is content *inside* the channel, which does not
perturb that prior. No model generation is involved, so the whole path is GPU-free — it
runs in the model-free render phase, is exercised on ``--dry-run``, and is unit-tested
in ``selftest`` without a model.

The source exchange is the resolved revision **anchor** (``resolve_revision_target``'s
trained target — ``keep`` keeps the original CoT, ``revise`` carries the IDEAL/branch
win), snapshotted onto the persona ledger anchor at formation time so the render does
not race the hot→archive move of the source chat.

**Dedup.** A persona is often distilled from CoT that already gestures at it, so
re-injecting it would be a redundant near-duplicate that just thickens identity
boilerplate. :func:`statement_in_cot` skips injection when the statement already appears
in the target's thought (lexical by default — normalized containment / word-overlap;
pass an embedder for semantic/paraphrase dedup). Dedup is a *quality* guard, not the
erosion fix — safety comes from the dialogue-shaped example regardless.
"""

from __future__ import annotations

import re
from typing import Optional

# A leading closed ``<think>thought</think>`` block plus everything after it — the same
# split render uses to find the answer span.
_LEADING_THINK_RE = re.compile(r"(?s)\s*<think>(.*?)</think>(.*)", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# A persona statement counts as already-present in the CoT when a CoT line's word set
# overlaps the statement's by at least this Jaccard ratio (lexical dedup). Verbatim /
# near-verbatim derivations are the common duplicate; paraphrase is left to the optional
# embedder path.
_DEDUP_JACCARD = 0.6
# Cosine at/above which an embedder treats a CoT line as expressing the statement.
_DEDUP_SIM = 0.80


def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower())}


def split_cot(target: str) -> Optional[tuple[str, str]]:
    """``(thought, rest)`` of a ``<think>thought</think>rest`` target, or ``None`` when
    it carries no closed think block with both a non-empty thought and an answer to keep.

    *rest* is captured verbatim (it includes the original ``</think>`` → answer
    separator) so reconstruction does not alter the answer span's formatting.
    """
    m = _LEADING_THINK_RE.match(target or "")
    if not m:
        return None
    thought = (m.group(1) or "").strip()
    rest = m.group(2) or ""
    if not thought or not rest.strip():
        return None
    return thought, rest


def statement_in_cot(statement: str, thought: str, *, embedder=None,
                     jaccard: float = _DEDUP_JACCARD, sim: float = _DEDUP_SIM) -> bool:
    """True when *statement* is already expressed in *thought* (so skip re-injection).

    Lexical by default: normalized substring containment, or any CoT line whose word set
    overlaps the statement's by >= *jaccard*. When an *embedder* (sentence-transformers
    ``encode`` interface) is supplied, a CoT line with cosine >= *sim* is also treated as
    a duplicate, catching paraphrase the lexical pass misses. An empty statement is
    "already present" (nothing to inject).
    """
    s = (statement or "").strip()
    if not s:
        return True
    s_norm = " ".join(_WORD_RE.findall(s.lower()))
    t_norm = " ".join(_WORD_RE.findall((thought or "").lower()))
    if s_norm and s_norm in t_norm:
        return True
    s_words = _words(s)
    if not s_words:
        return True
    lines = [ln for ln in re.split(r"[\n.;!?]+", thought or "") if ln.strip()]
    for ln in lines:
        lw = _words(ln)
        union = len(s_words | lw)
        if union and len(s_words & lw) / union >= jaccard:
            return True
    if embedder is not None and lines:
        try:
            vecs = embedder.encode([s] + lines, convert_to_numpy=True,
                                   normalize_embeddings=True)
            base = vecs[0]
            if any(float(base @ v) >= sim for v in vecs[1:]):
                return True
        except Exception:
            pass
    return False


def inject_persona(target: str, statement: str) -> Optional[str]:
    """Prepend *statement* as the first line of *target*'s ``<think>`` block.

    Returns the rewritten ``<think>…</think>answer`` target, or ``None`` when *target*
    has no usable CoT to inject into (answer-only / unclosed — those would erode the
    channel and are left to RAG recall instead) or *statement* is empty.
    """
    parts = split_cot(target)
    s = (statement or "").strip()
    if parts is None or not s:
        return None
    thought, rest = parts
    return f"<think>{s}\n{thought}</think>{rest}"


def prepend_persona_lines(target: str, statements: list) -> Optional[str]:
    """Prepend each of *statements* (in order) as leading lines of *target*'s ``<think>``
    block.

    Unlike :func:`inject_persona` this applies a *pre-selected*, already-deduped batch (the
    caller vets them with :func:`select_persona_injections`), so the first statement becomes
    the first CoT line and priority order is preserved. Returns the rewritten
    ``<think>…</think>answer`` target, or ``None`` when *target* has no usable CoT or the
    batch is empty.
    """
    parts = split_cot(target)
    stmts = [s.strip() for s in (statements or []) if (s or "").strip()]
    if parts is None or not stmts:
        return None
    thought, rest = parts
    block = "\n".join(stmts + [thought])
    return f"<think>{block}</think>{rest}"


def select_persona_injections(thought: str, personas: list, *, cap: int, embedder=None):
    """Choose up to *cap* persona anchors to inject into *thought*.

    *personas* is a priority-ordered list of ledger anchor dicts (each carrying
    ``content``). A candidate is skipped when its statement is already expressed in the CoT
    built so far — deduped **incrementally** so two paraphrases of the same idea don't both
    land (the running thought is grown with each accepted line before the next is tested).
    Returns ``(selected_anchors, statements)`` in injection order; empty when nothing
    survives. Pure/GPU-free unless an *embedder* is supplied for paraphrase dedup.
    """
    selected: list = []
    stmts: list = []
    running = thought or ""
    for p in personas or []:
        s = (p.get("content") or "").strip()
        if not s or statement_in_cot(s, running, embedder=embedder):
            continue
        selected.append(p)
        stmts.append(s)
        running = f"{s}\n{running}"
        if len(selected) >= cap:
            break
    return selected, stmts


def build_persona_example(anchor: dict, statement: str, *, embedder=None) -> Optional[dict]:
    """Build a dialogue-shaped SFT example for *statement* by injecting it into the CoT
    of *anchor* (a resolved revision/dialogue anchor: system_prompt, context, prompt,
    speaker, target).

    Returns ``None`` when there is nothing safe/useful to train: the anchor's target has
    no usable CoT, or the statement is already expressed in that CoT (dedup). The returned
    dict mirrors the dialogue render example shape plus the rewritten ``target``.
    """
    from training import render

    target = (anchor.get("target") or "").strip()
    parts = split_cot(target)
    if parts is None:
        return None
    thought, _ = parts
    if statement_in_cot(statement, thought, embedder=embedder):
        return None
    new_target = inject_persona(target, statement)
    if (not new_target or not render.has_cot(new_target)
            or render.trainable_answer(new_target) is None):
        return None
    render.assert_parity(anchor, new_target)          # train/inference parity guard
    return {"messages": render.build_messages(anchor, new_target),
            "variant": new_target, "target": new_target}
