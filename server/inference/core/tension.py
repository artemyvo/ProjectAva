"""Cognitive-tension reduction.

Turns a per-generated-token series of (entropy, margin) into per-segment summary
statistics, split between the CoT (`<think>`) and the answer. The raw signals are
captured during generation (see UnslothBackend); this module is the pure reduction
and is GPU-free / unit-testable.

Design notes:
  * Per token we keep two signals — Shannon entropy of the next-token distribution
    (how spread out the model's options were, in nats) and the top1-top2 *probability*
    margin (how decisively it favoured the winner, bounded [0,1] so it is comparable
    across models and temperatures). They disagree informatively, so both.
  * Entropy is summarized as peak (a localized spike) + median (the diffuse baseline);
    their gap distinguishes "spiked once" from "uniformly contested".
  * Margin is summarized for *friction*, which lives at the low end: a low margin means
    the top two tokens were near-ties. We keep the median, the 10th-percentile (a robust
    low end — one lone near-tie can't drag it the way `min` would), and the contested
    fraction (share of tokens below CONTESTED_MARGIN). We deliberately do NOT keep peak
    margin: the most-decisive token is the *least* frictional, i.e. anti-signal.
  * No divergence field here: CoT and answer live in different token regimes (and in
    our logs different languages), so divergence must be normalized against a corpus
    baseline that does not exist at log time. It is computed offline in the
    validation step. We store raw per-segment stats + provenance instead.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence


# ── per-token signal from one step's raw logits ──────────────────────────────

def step_signals_tensor(logits, target_id=None):
    """(entropy, probability-margin, top2-token-ids, target-prob) on the logits' device.

    Kept on-device so a per-token capture loop can defer the GPU->CPU sync to a
    single bulk transfer after generation, rather than syncing every token.
    `top2_ids` (shape (2,)) are the two most-probable token ids at this step —
    the raw material for the road-not-taken: with sampling, the *generated* token
    may be neither of them, so both are kept and the alternative is resolved
    against the generated id at reduction time (see `_contested_trace`).

    `target_id` (optional): when given, the 4th element is the probability mass this
    step's distribution put on that specific token id (a 0-dim tensor), else None.
    Folded into the same softmax so it costs ~nothing — used to read how strongly
    the model wanted to open a thinking block (e.g. gemma-4's ``<|channel>``) at the
    first generated step.
    """
    import torch

    v = logits.detach().float().reshape(-1)
    logp = torch.log_softmax(v, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum()
    top2 = torch.topk(p, 2)
    margin = top2.values[0] - top2.values[1]  # probability margin, in [0, 1]
    target_prob = p[target_id] if target_id is not None else None
    return entropy, margin, top2.indices, target_prob


def step_signals(logits) -> tuple[float, float]:
    """(entropy in nats, top1-top2 probability margin in [0,1]) for one logit vector."""
    entropy, margin, _, _ = step_signals_tensor(logits)
    return float(entropy), float(margin)


# ── boundary detection (CoT end → answer start) ──────────────────────────────

def _find_subseq(seq: Sequence[int], sub: Sequence[int]) -> int:
    """First index where `sub` occurs in `seq`, or -1. Empty `sub` → -1."""
    if not sub:
        return -1
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if list(seq[i:i + m]) == list(sub):
            return i
    return -1


def find_think_end(token_ids: Sequence[int], close_markers: Sequence[Sequence[int]]) -> Optional[int]:
    """Index where the answer starts (just past the thinking-close marker), or None.

    `close_markers` is a list of candidate token-id sequences for the close of the
    thinking block — model-family specific (`</think>` for Qwen, the channel-close
    for Gemma-4). The first that matches wins. None → no CoT split (whole reply is
    treated as the answer).
    """
    for marker in close_markers:
        idx = _find_subseq(token_ids, marker)
        if idx != -1:
            return idx + len(marker)
    return None


# ── reduction ────────────────────────────────────────────────────────────────

# A token is "contested" when the top two next-token probabilities are within this
# of each other — a near-tie. Prob-space, so it is model/temperature agnostic; tunable.
CONTESTED_MARGIN = 0.10

# How many of the most-contested tokens to record per segment in the `contested` trace.
CONTESTED_TRACE_K = 5


def _peak(xs: Sequence[float]) -> Optional[float]:
    return max(xs) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated q-quantile (q in [0,1]); None for empty input."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _contested_frac(margins: Sequence[float]) -> Optional[float]:
    """Share of tokens whose probability margin is below CONTESTED_MARGIN."""
    if not margins:
        return None
    return sum(1 for m in margins if m < CONTESTED_MARGIN) / len(margins)


def _contested_trace(
    entropies: Sequence[float],
    margins: Sequence[float],
    token_ids: Sequence[int],
    offset: int,
    decode: Optional[Callable[[Sequence[int]], str]],
    top2_ids: Optional[Sequence[Sequence[int]]] = None,
) -> list:
    """Up to CONTESTED_TRACE_K most-contested (lowest-margin) tokens in the segment.

    Only genuine near-ties (margin < CONTESTED_MARGIN) are listed, most-contested
    first — so a decisive segment yields an empty trace rather than its five least
    decisive tokens. `position` is the absolute index into the generated token series,
    letting a row be cross-referenced against the decoded reply. This is where "the
    model nearly said something else" lives, which no scalar can show.

    When `top2_ids` (per-token (top1, top2) ids, aligned with the segment) is given,
    each row also carries the road-not-taken: `alt_token_id` is the most-probable
    token that is NOT the one actually generated (with sampling the generated token
    may be neither of the top two, in which case the alternative is top-1). This is
    the branch-point material for branch-and-select revision.
    """
    idx = [i for i, m in enumerate(margins) if m < CONTESTED_MARGIN]
    idx.sort(key=lambda i: margins[i])
    rows = []
    for i in idx[:CONTESTED_TRACE_K]:
        tid = int(token_ids[i])
        row = {
            "position": offset + i,
            "token_id": tid,
            "token": decode([tid]) if decode is not None else None,
            "margin": margins[i],
            "entropy": entropies[i],
        }
        if top2_ids is not None:
            t1, t2 = int(top2_ids[i][0]), int(top2_ids[i][1])
            alt = t2 if t1 == tid else t1
            row["alt_token_id"] = alt
            row["alt_token"] = decode([alt]) if decode is not None else None
        rows.append(row)
    return rows


def _segment(
    entropies: Sequence[float],
    margins: Sequence[float],
    token_ids: Optional[Sequence[int]] = None,
    *,
    offset: int = 0,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    top2_ids: Optional[Sequence[Sequence[int]]] = None,
) -> Optional[dict]:
    if not entropies:
        return None
    seg = {
        "peak_entropy": _peak(entropies),
        "median_entropy": _median(entropies),
        "median_margin": _median(margins),
        "p10_margin": _quantile(margins, 0.10),
        "contested_frac": _contested_frac(margins),
        "n_tokens": len(entropies),
    }
    if token_ids is not None:
        seg["contested"] = _contested_trace(
            entropies, margins, token_ids, offset, decode, top2_ids,
        )
    return seg


def summarize(
    entropies: Sequence[float],
    margins: Sequence[float],
    answer_start: Optional[int],
    *,
    model_id: str,
    raw_logits: bool,
    token_ids: Optional[Sequence[int]] = None,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    top2_ids: Optional[Sequence[Sequence[int]]] = None,
) -> Optional[dict]:
    """Build the `tension` block from the per-token series.

    `answer_start` is the index in the series where the answer begins (CoT is
    everything before it). None → no CoT block; the whole series is the answer.
    `token_ids` (aligned with the signal series) and `decode` are optional; when given,
    each segment gains a `contested` trace of its most near-tie tokens, and the block
    carries the full generated `token_ids` series — branch replay needs the exact raw
    prefix, which cannot be recovered by re-tokenizing the cleaned reply text (cleaning
    strips channel/turn tokens; trace `position`s index this raw series). Alongside it
    the block carries the full per-token `entropies` and `margins` series (and, when
    `top2_ids` is given, the per-token `top2_ids`) — all aligned with `token_ids` — so
    any later reduction (a different threshold, a deeper trace, distribution analysis)
    is recoverable offline without regenerating. `top2_ids` (per-token (top1, top2) ids)
    additionally puts the road-not-taken (`alt_token_id`/`alt_token`) on each contested
    row. `decode` maps a token-id list to text (e.g. `tokenizer.decode`), kept as a
    callable so this module stays GPU/tokenizer-free. Returns None if there is no usable
    signal at all.
    """
    if not entropies:
        return None

    if answer_start is None or answer_start <= 0:
        cot = None
        ans_e, ans_m = entropies, margins
        ans_t = token_ids
        ans_a = top2_ids
        ans_off = 0
    else:
        answer_start = min(answer_start, len(entropies))
        cot_t = token_ids[:answer_start] if token_ids is not None else None
        cot_a = top2_ids[:answer_start] if top2_ids is not None else None
        cot = _segment(
            entropies[:answer_start], margins[:answer_start], cot_t,
            decode=decode, top2_ids=cot_a,
        )
        ans_e, ans_m = entropies[answer_start:], margins[answer_start:]
        ans_t = token_ids[answer_start:] if token_ids is not None else None
        ans_a = top2_ids[answer_start:] if top2_ids is not None else None
        ans_off = answer_start

    block = {
        "cot": cot,
        "answer": _segment(
            ans_e, ans_m, ans_t, offset=ans_off, decode=decode, top2_ids=ans_a,
        ),
        "model_id": model_id,
        "raw_logits": raw_logits,
    }
    if token_ids is not None:
        # Full per-token raw series, all aligned with `token_ids` (length n_tokens).
        # The per-segment summary stats and the contested trace above are *derived*
        # from exactly these arrays; we persist the arrays themselves because the
        # signal is a property of the (weights, prefix) pair at generation time and
        # cannot be recomputed once the weights drift. Keeping them means any later
        # reduction is recoverable offline without regenerating — a different
        # CONTESTED_MARGIN, a deeper trace than CONTESTED_TRACE_K, distribution /
        # localization analysis, or divergence variants. `top2_ids` (per-token
        # (top1, top2) ids) additionally allows branching at any position, not only
        # the pre-selected contested ones.
        block["token_ids"] = [int(t) for t in token_ids]
        block["entropies"] = [float(e) for e in entropies]
        block["margins"] = [float(m) for m in margins]
        if top2_ids is not None:
            block["top2_ids"] = [[int(x) for x in pair] for pair in top2_ids]
    return block
