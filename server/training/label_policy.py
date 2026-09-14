"""Training-label surgery (pure, torch-free).

The masking pipeline is: ``train_on_responses_only`` unmasks *every* assistant turn,
then the training collator applies the policy here. Two things happen, in order, on the
label row:

  1. **keep-final-turn** — re-mask every unmasked run except the last. In a multi-turn
     dialogue anchor the non-final assistant turns are rendered CoT-less (to mirror how
     inference stores history); training them erodes the reasoning channel over cycles, so
     only the final assistant turn should carry loss. (This is the exact behaviour of the
     old ``train_cycle._keep_final_turn_only`` — proven equivalent in the selftest.)

  2. **user unmask** (optional) — on the last training run of an exchange, re-*unmask* the
     **entire final user turn** so the user's own words reach the loss on that one copy.
     This is the blunt replacement for the old lexical-entrainment machinery: rather than
     harvesting distinctive spans and forcing them, the render simply flags the exchange's
     final (deprecating) decay copy, and the whole user message contributes to the loss
     once — user words reach the weights once per the exchange's total decay copies (six on
     the default ``[3,2,1]`` dialogue curve). The user content is located by the same chat
     markers ``train_on_responses_only`` trusts: everything strictly between the final
     ``instruction_part`` marker and the ``response_part`` that opens the final assistant
     turn — so the system prompt (before the marker) and the template control tokens are
     never unmasked.

Kept torch-free (operates on plain ``list[int]`` rows) so the risky masking logic is
provable without a GPU. ``train_cycle`` supplies the tensor<->list adapter.

GPU-free self-test: ``python -m training.label_policy``.
"""

from __future__ import annotations

from typing import Optional, Sequence


def find_subsequences(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    """Every start offset at which *needle* occurs as a contiguous sub-sequence of
    *haystack* (overlapping matches included). Empty when the needle is empty or longer
    than the haystack."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return []
    needle = list(needle)
    return [i for i in range(n - m + 1) if list(haystack[i:i + m]) == needle]


def _final_run_start(labels: Sequence[int]) -> Optional[int]:
    """First index of the final unmasked run (the final assistant turn), or ``None`` when
    the row is fully masked (nothing to train)."""
    idx = [i for i, l in enumerate(labels) if l != -100]
    if not idx:
        return None
    gap_ks = [k for k in range(len(idx) - 1) if idx[k + 1] - idx[k] > 1]
    if gap_ks:
        return idx[gap_ks[-1] + 1]
    return idx[0]


def _final_user_span(
    input_ids: Sequence[int],
    instruction_ids: Sequence[int],
    response_ids: Sequence[int],
    last_run_start: int,
) -> Optional[tuple[int, int]]:
    """``(content_start, content_end)`` of the final user turn's content, or ``None`` when
    the markers can't be located.

    The final user turn is delimited by the **last** ``instruction_part`` marker that opens
    before the final assistant run and the **first** ``response_part`` marker after it. The
    returned span is the content strictly between the two markers — the user's words, with
    neither marker's tokens nor the surrounding system/assistant text."""
    if not instruction_ids or not response_ids:
        return None
    instr_len = len(instruction_ids)
    instr_starts = [s for s in find_subsequences(input_ids, instruction_ids)
                    if s + instr_len <= last_run_start]
    if not instr_starts:
        return None
    content_start = instr_starts[-1] + instr_len
    resp_starts = [s for s in find_subsequences(input_ids, response_ids) if s >= content_start]
    if not resp_starts:
        return None
    content_end = resp_starts[0]
    if content_end <= content_start:
        return None
    return content_start, content_end


def row_label_policy(
    labels: Sequence[int],
    input_ids: Sequence[int],
    unmask_user: bool = False,
    instruction_ids: Optional[Sequence[int]] = None,
    response_ids: Optional[Sequence[int]] = None,
) -> list[int]:
    """Apply keep-final-turn masking (always) + the final-user-turn unmask (when
    ``unmask_user``) to one row. Returns a new label list; does not mutate the input.

    With ``unmask_user`` false this is exactly ``_keep_final_turn_only`` on a single row
    (asserted in the selftest). With it true and the chat markers supplied, the whole final
    user turn is re-unmasked (label := the actual input id) so it lands in the loss."""
    labels = list(labels)
    last_run_start = _final_run_start(labels)
    if last_run_start is None:
        return labels

    # keep-final-turn: mask everything before the final assistant run.
    for i in range(last_run_start):
        labels[i] = -100

    # blunt user unmask: re-unmask the final user turn's content (marker-delimited, so the
    # system prompt and template control tokens are left masked).
    if unmask_user:
        span = _final_user_span(list(input_ids), instruction_ids or [],
                                response_ids or [], last_run_start)
        if span is not None:
            content_start, content_end = span
            for pos in range(content_start, content_end):
                labels[pos] = input_ids[pos]
    return labels


def row_loss_weights(
    labels: Sequence[int],
    input_ids: Sequence[int],
    unmask_user: bool,
    instruction_ids: Optional[Sequence[int]],
    response_ids: Optional[Sequence[int]],
    user_loss_weight: float,
) -> list[float]:
    """Per-token loss weight for a **folded contamination** row (contamination_fold):
    ``1.0`` on the final assistant run, ``user_loss_weight`` on the final user turn's
    content, ``0.0`` everywhere else. Returns a new float list; does not mutate anything.

    Computed from the RAW ``labels`` (as ``train_on_responses_only`` leaves them — every
    assistant turn unmasked) so it must be called BEFORE ``row_label_policy`` re-masks the
    non-final turns; it locates the same final assistant run and final user span that policy
    does. The pairing with the two-row split: the split trains the response across both rows
    (Σ mult = response_total, folded onto ``lr_multiplier``) and the user only on the unmask
    row (mult ``dose``), so at the row's LR the response weight is 1.0 and the user weight is
    ``dose / response_total`` — passed in as ``user_loss_weight``. Torch-free (self-tested)."""
    weights = [0.0] * len(labels)
    last_run_start = _final_run_start(labels)
    if last_run_start is None:
        return weights                       # fully masked — nothing trains
    # Final assistant run: every unmasked position from its start to the end (the earlier
    # runs sit before last_run_start and stay at 0.0, matching keep-final-turn masking).
    for i in range(last_run_start, len(labels)):
        if labels[i] != -100:
            weights[i] = 1.0
    if unmask_user:
        span = _final_user_span(list(input_ids), instruction_ids or [],
                                response_ids or [], last_run_start)
        if span is not None:
            for pos in range(*span):
                weights[pos] = float(user_loss_weight)
    return weights


# --------------------------------------------------------------------------- #
# GPU-free self-test: python -m training.label_policy                          #
# --------------------------------------------------------------------------- #
def _ref_keep_final_turn_only(labels: list[int]) -> list[int]:
    """Reference reimplementation of train_cycle._keep_final_turn_only on a plain list,
    used only to prove row_label_policy(unmask_user=False) is behaviourally identical."""
    labels = list(labels)
    idx = [i for i, l in enumerate(labels) if l != -100]
    if not idx:
        return labels
    gap_ks = [k for k in range(len(idx) - 1) if idx[k + 1] - idx[k] > 1]
    if not gap_ks:
        return labels
    last_run_start = idx[gap_ks[-1] + 1]
    for i in range(last_run_start):
        labels[i] = -100
    return labels


def _selftest() -> None:
    # --- find_subsequences ----------------------------------------------------------- #
    assert find_subsequences([1, 2, 3, 2, 3], [2, 3]) == [1, 3]
    assert find_subsequences([1, 2, 3], [9]) == []
    assert find_subsequences([1], [1, 2]) == []          # needle longer than haystack
    assert find_subsequences([5, 5, 5], [5, 5]) == [0, 1]  # overlapping

    # --- parity with _keep_final_turn_only (unmask_user=False) ----------------------- #
    cases = [
        [-100, -100, 40, 41, -100, -100, 50, 51],   # two runs -> keep the last
        [-100, 40, 41, 42],                          # single run -> unchanged
        [-100, -100, -100],                          # all masked -> unchanged
        [40, 41],                                    # run at row start, single -> unchanged
        [-100, 40, -100, 41, -100, 42],              # three 1-tok runs -> keep only the last
    ]
    for row in cases:
        got = row_label_policy(row, row, False)
        assert got == _ref_keep_final_turn_only(row), (row, got)

    # --- user unmask, restricted to the FINAL user turn ------------------------------ #
    # markers: instruction = [7, 1], response = [8, 2]
    #  idx:    0    1    2    3   4  5    6    7    8    9   10  11  12  13
    #  seg:  sys  I0   I0  usrA a1 a1   R    R    I1   I1  usrB  R   R   a2
    instr = [7, 1]
    resp = [8, 2]
    #             sys  <I>       usrA  <a1>   <R>      <I>       usrB  <R>       a2
    input_ids = [10,  7, 1,      77,   40, 41, 8, 2,   7, 1,     88,   8, 2,     90]
    labels    = [-100, -100, -100, -100, 50, 51, -100, -100, -100, -100, -100, -100, -100, 60]

    out = row_label_policy(labels, input_ids, True, instr, resp)
    # final assistant run intact
    assert out[13] == 60, out
    # final user turn content (position 10 = usrB) unmasked to the actual input id
    assert out[10] == 88, out
    # the EARLIER user turn (position 3 = usrA) stays masked
    assert out[3] == -100, out
    # the non-final assistant run (4,5) is masked by keep-final-turn
    assert out[4] == -100 and out[5] == -100, out
    # marker tokens themselves stay masked
    assert out[8] == -100 and out[9] == -100, out   # instruction marker
    assert out[11] == -100 and out[12] == -100, out  # response marker

    # single-turn: window would include the system prompt, but the marker delimits the user
    #             sys sys  <I>      usr  usr  <R>      a
    input_ids2 = [10, 11,  7, 1,    77, 78,   8, 2,    50, 51]
    labels2    = [-100, -100, -100, -100, -100, -100, -100, 50, 51]
    # pad labels2 to len(input_ids2)
    labels2 = [-100, -100, -100, -100, -100, -100, -100, -100, 50, 51]
    out2 = row_label_policy(labels2, input_ids2, True, instr, resp)
    assert out2[4] == 77 and out2[5] == 78, out2   # user content unmasked
    assert out2[0] == -100 and out2[1] == -100, out2  # system prompt stays masked
    assert out2[8] == 50 and out2[9] == 51, out2   # assistant intact

    # unmask_user with no markers found is a safe no-op beyond keep-final-turn
    out3 = row_label_policy(labels, input_ids, True, [999], [998])
    assert out3 == _ref_keep_final_turn_only(labels), out3

    # --- row_loss_weights (folded contamination) ------------------------------------- #
    # Reuse the two-user-turn fixture: final assistant run at 13, final user span at 10.
    w = row_loss_weights(labels, input_ids, True, instr, resp, 0.25)
    assert len(w) == len(labels)
    assert w[13] == 1.0                      # final assistant run weighted 1.0
    assert w[10] == 0.25                     # final user turn weighted at the dose fraction
    assert w[3] == 0.0                       # earlier (non-final) user turn: 0.0
    assert w[4] == 0.0 and w[5] == 0.0       # non-final assistant run: 0.0 (keep-final-turn)
    assert w[8] == 0.0 and w[9] == 0.0       # instruction marker tokens: 0.0
    assert w[11] == 0.0 and w[12] == 0.0     # response marker tokens: 0.0
    assert w[0] == 0.0                       # system/pre-run: 0.0
    # unmask_user False -> response-only weights (no user span lifted).
    w2 = row_loss_weights(labels, input_ids, False, instr, resp, 0.25)
    assert w2[13] == 1.0 and w2[10] == 0.0
    # fully-masked row -> all zeros.
    assert row_loss_weights([-100, -100], [1, 2], True, instr, resp, 0.25) == [0.0, 0.0]
    # markers absent -> response run still weighted, user span left at 0.0 (safe no-op).
    w3 = row_loss_weights(labels, input_ids, True, [999], [998], 0.25)
    assert w3[13] == 1.0 and w3[10] == 0.0

    print("label_policy selftest OK")


if __name__ == "__main__":
    _selftest()
