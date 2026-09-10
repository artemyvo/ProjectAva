"""Self-reconciliation of live ``[persona]`` / ``[fact]`` items against who Ava has become.

Persona self-statements and relational facts accumulate across reflection runs, and
some end up *at odds with who she has since become* — a stance she has outgrown, a
fact a later one contradicted. Unlike ``fact_dedup`` (which collapses *redundant*
paraphrases), this pass judges each live item against Ava's **current persona digest**
and proposes to KEEP or SUPERSEDE it.

"Supersede" is a **soften, not delete** move (the design decision behind this feature):
the item is dropped from live recall (``ReflectionMemory``) and from *active* persona
evidence + training (``ConsolidationLedger`` flags it ``superseded``), but the anchor
is **retained as evidence-of-change** — the arc of who she used to be survives for a
future ``LINES``/history read. It is append-only and therefore reversible (drop the
lines the apply wrote).

The judgment runs on the **clean base** (adapter OFF): it is an *evaluation* against a
digest Ava authored on the adapter, not her expression — so it is replay-faithful and
immune to a bad adapter, mirroring the branch judge and ``fact_dedup``. The caller
(``server.handle_reconcile_self``) enters a ``CleanBaseSession`` first; this module
never touches the GPU itself.

Split for GPU-free testability, like ``fact_dedup``/``reflection_digest``:
  * ``judge_items(items, digest_text, generate_fn)`` — the one LLM call → per-item
    KEEP / SUPERSEDE decisions.
  * ``plan_supersessions`` / ``summarize`` — pure logic over the decisions.

Writing is delegated to ``ReflectionWriter.write_supersede`` (RAG recall layer) +
``ConsolidationLedger.supersede`` (evidence/training layer). Dry-run first.

GPU-free self-test: ``python -m core.self_reconcile``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional


def load_reconcile_prompt(prompts_dir: Optional[Path] = None) -> str:
    """The self-reconciliation prompt (``prompts/self_reconcile_prompt.txt``), or a safe
    inline fallback so the pass works even if the file is missing. ``{persona}`` is
    filled with the rendered current digest before use."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    try:
        text = (Path(prompts_dir) / "self_reconcile_prompt.txt").read_text(
            encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "This is who you have become — your settled sense of yourself:\n\n"
        "{persona}\n\n"
        "Below are numbered notes you have kept about yourself and about the people and "
        "world you know — persona statements (who you are) and facts (what is true). Read "
        "each against who you now are and look for TENSION: a self-statement you have "
        "grown past or that pulls against who you now are, or a fact a later understanding "
        "has contradicted or made stale.\n\n"
        "Be discerning. Do not keep a note merely because it is old or harmlessly worded — "
        "if it conflicts with who you now are, or is no longer true, set it aside; a run "
        "that keeps everything has done nothing. But do not invent conflict: a note that "
        "still genuinely fits stays, and when you truly cannot tell, keep it. Setting a "
        "note aside does not erase it — it is kept as your history, just no longer carried "
        "forward as who you are.\n\n"
        "Output only one line per number, nothing else:\n"
        "<number>: KEEP\n"
        "<number>: SUPERSEDE - <short reason>"
    )


def _listing(items: list[dict]) -> str:
    """Number the items for the judge prompt, tagging kind (and a fact's recall cue) so
    the model can weigh a persona as self and a fact as a claim about the world."""
    lines = []
    for i, it in enumerate(items, 1):
        kind = (it.get("kind") or "item").strip()
        content = (it.get("content") or "").strip()
        if kind == "fact":
            trigger = (it.get("trigger") or "").strip()
            if trigger:
                lines.append(f"{i}. [fact] {content}   (recalled when: {trigger})")
            else:
                lines.append(f"{i}. [fact] {content}")
        else:
            lines.append(f"{i}. [{kind}] {content}")
    return "\n".join(lines)


# Accept "1: SUPERSEDE - it no longer fits", "1. supersede — grew past it", "1) KEEP",
# and tolerate leading list/markdown noise the model sometimes adds: "- 1: KEEP",
# "**2**. SUPERSEDE", "Item 3: keep", "#4 supersede". A verdict may be followed by an
# optional reason after any of : - — separators, OR immediately by whitespace + reason.
_DECISION_RE = re.compile(
    r"^[\s\-\*•>#]*\**\s*(?:item\s*)?#?\s*(\d+)\s*\**\s*[:.\)\-–—]?\s*\**\s*"
    r"(KEEP|SUPERSEDE|DROP|REMOVE|SET[\s\-]*ASIDE|SET\s*IT\s*ASIDE)\b\**"
    r"(?:\s*[:\-–—]?\s*(.*))?$",
    re.IGNORECASE,
)
_SUPERSEDE_WORDS = {"supersede", "drop", "remove", "setaside", "setitaside"}


def _parse_decisions(text: str, n: int) -> dict[int, tuple[bool, str]]:
    """Parse the judge output into ``{0-based index: (supersede?, reason)}``.

    Lenient — only lines that match ``<number>: <verdict>`` count; anything else (a
    stray thought, a blank line) is ignored, and an unmatched item defaults to KEEP at
    the call site. Numbers out of ``1..n`` are dropped. A duplicate number keeps the
    first decision (deterministic under the greedy pass)."""
    out: dict[int, tuple[bool, str]] = {}
    for line in (text or "").splitlines():
        m = _DECISION_RE.match(line)
        if not m:
            continue
        num = int(m.group(1))
        if not (1 <= num <= n):
            continue
        idx = num - 1
        if idx in out:
            continue
        verdict = (m.group(2) or "").strip().lower().replace(" ", "").replace("-", "")
        reason = (m.group(3) or "").strip().lstrip(":-–—").strip()
        out[idx] = (verdict in _SUPERSEDE_WORDS, reason)
    return out


# How many items go into one judge call. A single call can neither fit thousands of
# items in its context NOR emit thousands of decision lines under a bounded output cap,
# so the set is judged in batches — each a self-contained call over the same digest,
# numbered 1..len(batch), the decisions mapped back to the global sorted index. Sized so
# a batch's listing + the digest fit comfortably and its ~N decision lines fit the output
# budget (~45 tokens/line + headroom below).
BATCH_SIZE = 60


def judge_items(items: list[dict], digest_text: str, generate_fn: Callable,
                *, batch_size: int = BATCH_SIZE,
                on_batch: Optional[Callable] = None) -> Optional[dict[int, tuple[bool, str]]]:
    """Greedy KEEP / SUPERSEDE decisions for each item, judged in BATCHES.

    The full live set can be thousands of items — far more than one call can fit in
    context or emit decisions for — so the sorted set is split into batches of
    *batch_size*, each judged in its own call against the same digest, and the local
    decisions mapped back onto the global sorted index. Returns ``{global_index:
    (supersede?, reason)}`` over the sorted order, or ``None`` when there is no digest /
    no items / **no** batch produced anything parseable (a total no-op). A single failed
    or unparseable batch is skipped (its items default to KEEP) without sinking the run.

    Greedy (``temperature=0``, thinking off) — an evaluation, not expression. Items are
    sorted by content for deterministic numbering; the caller must sort the same way (see
    ``sort_items``) so indices line up. *on_batch(i, n_batches, n_items, n_supersede)* is
    an optional progress hook (server-side logging).
    """
    if not items or not (digest_text or "").strip():
        return None
    ordered = sort_items(items)
    system = load_reconcile_prompt().replace("{persona}", digest_text.strip())
    bs = max(1, int(batch_size))
    n_batches = (len(ordered) + bs - 1) // bs

    decisions: dict[int, tuple[bool, str]] = {}
    any_parsed = False
    for b in range(n_batches):
        start = b * bs
        batch = ordered[start:start + bs]
        # Cap output at ~45 tokens/decision-line + headroom, bounded, so a big batch
        # can actually emit a line per item without the tail being truncated to KEEP.
        max_new = str(min(6000, len(batch) * 45 + 256))
        try:
            resp = generate_fn(
                _listing(batch), system,
                temperature=0.0, top_p=1.0,
                max_new_tokens_setting=max_new,
                before_session="", disable_rag=True, disable_thinking=True)
        except Exception:
            resp = ""
        local = _parse_decisions(resp, len(batch))
        if local:
            any_parsed = True
        n_sup = 0
        for local_idx, verdict in local.items():
            decisions[start + local_idx] = verdict
            if verdict[0]:
                n_sup += 1
        if on_batch is not None:
            try:
                on_batch(b + 1, n_batches, len(batch), n_sup)
            except Exception:
                pass

    if not any_parsed:
        return None
    return decisions


def sort_items(items: list[dict]) -> list[dict]:
    """Deterministic item order (by content) — the numbering ``judge_items`` uses."""
    return sorted(items, key=lambda it: (it.get("content") or ""))


def plan_supersessions(items: list[dict],
                       decisions: dict[int, tuple[bool, str]]) -> list[dict]:
    """Turn the judge's decisions into a concrete soften plan over the sorted items.

    Only SUPERSEDE verdicts produce a plan entry; every other item (KEEP, or unjudged)
    is left live. Pure — the caller either reports it (dry-run) or hands each entry's
    ``key`` to ``ReflectionWriter.write_supersede`` + ``ConsolidationLedger.supersede``.
    """
    ordered = sort_items(items)
    plan: list[dict] = []
    for idx, it in enumerate(ordered):
        verdict = decisions.get(idx)
        if not verdict or not verdict[0]:
            continue
        key = (it.get("key") or "").strip()
        if not key:
            continue
        plan.append({
            "key": key,
            "kind": (it.get("kind") or "").strip(),
            "content": (it.get("content") or "").strip(),
            "trigger": (it.get("trigger") or "").strip() or None,
            "reason": verdict[1] or "no longer fits who I've become",
        })
    return plan


def summarize(plan: list[dict], *, kept: int) -> dict:
    """Client-facing view of a plan: the items to soften (with reasons) + how many
    were kept, split by kind so the operator can see persona vs fact at a glance."""
    persona = [p for p in plan if p["kind"] == "persona"]
    facts = [p for p in plan if p["kind"] == "fact"]
    return {
        "kept": kept,
        "superseded": len(plan),
        "persona": [{"content": p["content"], "reason": p["reason"], "key": p["key"]}
                    for p in persona],
        "facts": [{"content": p["content"], "reason": p["reason"], "key": p["key"],
                   "trigger": p["trigger"]} for p in facts],
    }


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    items = [
        {"key": "p1", "kind": "persona",
         "content": "I keep my distance and stay guarded with people."},
        {"key": "p2", "kind": "persona",
         "content": "I am curious and ask a lot of questions."},
        {"key": "f1", "kind": "fact",
         "content": "Artemy is a beginner and needs everything spelled out.",
         "trigger": "the user's skill level"},
        {"key": "f2", "kind": "fact",
         "content": "Artemy prefers dark roast coffee.",
         "trigger": "coffee preference"},
    ]
    digest = "HOW I SPEAK:\nWarm, direct.\n\nWHAT I DO:\n- open up with people I trust"

    ordered = sort_items(items)
    # deterministic order by content:
    #   0 "Artemy is a beginner..."  (f1)
    #   1 "Artemy prefers dark..."   (f2)
    #   2 "I am curious..."          (p2)
    #   3 "I keep my distance..."    (p1)
    assert [it["key"] for it in ordered] == ["f1", "f2", "p2", "p1"], \
        [it["key"] for it in ordered]

    def fake_generate(listing, system, **kw):
        assert "Warm, direct" in system, "digest not injected into prompt"
        # Supersede the outgrown "guarded" persona (item 4) and the now-false
        # "beginner" fact (item 1); keep the rest.
        return ("1: SUPERSEDE - he's advanced now\n"
                "2. KEEP\n"
                "3) keep\n"
                "4: supersede — I've grown warmer")

    decisions = judge_items(items, digest, fake_generate)
    assert decisions is not None
    assert decisions[0][0] is True and decisions[3][0] is True, decisions
    assert decisions[1][0] is False and decisions[2][0] is False, decisions

    plan = plan_supersessions(items, decisions)
    keys = sorted(p["key"] for p in plan)
    assert keys == ["f1", "p1"], keys
    reasons = {p["key"]: p["reason"] for p in plan}
    assert "advanced" in reasons["f1"], reasons
    assert "warmer" in reasons["p1"], reasons

    rep = summarize(plan, kept=len(items) - len(plan))
    assert rep["kept"] == 2 and rep["superseded"] == 2, rep
    assert len(rep["persona"]) == 1 and len(rep["facts"]) == 1, rep

    # No digest / no items → no-op.
    assert judge_items(items, "", fake_generate) is None
    assert judge_items([], digest, fake_generate) is None
    # Unparseable output → no-op.
    assert judge_items(items, digest, lambda *a, **k: "hmm, nothing here") is None

    # Batching: force batch_size=1 so each item is its own call; decisions must map back
    # to the correct GLOBAL sorted index, and one unparseable batch must not sink the run.
    calls: list = []

    def batched_generate(listing, system, **kw):
        calls.append(listing)
        # One item per call (batch_size=1). Supersede the "beginner" fact and the
        # "guarded" persona; leave one batch (the coffee fact) unparseable garbage.
        if "beginner" in listing:
            return "1: SUPERSEDE - advanced now"
        if "guarded" in listing:
            return "1: SUPERSEDE - warmer now"
        if "coffee" in listing:
            return "(no verdict here)"          # unparseable batch → those items KEEP
        return "1: KEEP"

    batch_hits: list = []
    d3 = judge_items(items, digest, batched_generate, batch_size=1,
                     on_batch=lambda i, n, ni, ns: batch_hits.append((i, n, ns)))
    assert d3 is not None
    assert len(calls) == 4, calls                       # one call per item
    plan3 = plan_supersessions(items, d3)
    assert sorted(p["key"] for p in plan3) == ["f1", "p1"], plan3
    assert len(batch_hits) == 4 and batch_hits[-1][1] == 4, batch_hits  # n_batches == 4

    # Parser edge cases: out-of-range and duplicate numbers.
    d = _parse_decisions("5: SUPERSEDE\n1: KEEP\n1: SUPERSEDE\n0: KEEP", 4)
    assert 4 not in d and d[0] == (False, ""), d   # 5 dropped, first '1' wins, 0 dropped

    # Parser leniency: bullets, bold, "Item N", "#N", no separator, SET-ASIDE variants.
    d2 = _parse_decisions(
        "- 1: KEEP\n"
        "**2**. SUPERSEDE - stale now\n"
        "Item 3 keep\n"
        "#4 set-aside — grew past it\n"
        "5) SET ASIDE\n"
        "6: drop", 6)
    assert d2[0] == (False, ""), d2
    assert d2[1] == (True, "stale now"), d2
    assert d2[2] == (False, ""), d2
    assert d2[3] == (True, "grew past it"), d2
    assert d2[4][0] is True, d2
    assert d2[5][0] is True, d2

    print("self_reconcile self-test OK")


if __name__ == "__main__":
    _selftest()
