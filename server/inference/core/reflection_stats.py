"""Reflection-run statistics accumulator (pure, GPU-free).

One ``RunStats`` is created per reflection run in ``ReflectionRunner.execute_run``
and threaded through the phase methods (exactly like ``store`` / ``send_event_fn``).
It centralizes everything the Sleep tab's live stats panel and the end-of-run
detailed report need, so the metrics can't drift across the scattered phase code:

  * **progress / ETA** — a precount of total revisable exchanges (the backbone
    clock) + a running global exchange counter; ETA is deliberately *rough*
    ("~5h left, leave the box unattended"), extrapolated from observed
    seconds-per-exchange against the exchanges remaining.
  * **outcomes** — verdict split (keep/revise), the chosen-target breakdown
    (original / ideal-win / branch-win / judge-override), categorized discards
    (the three distinct failure modes, no longer lumped), retries, branch
    eligibility, RAG/persona deltas, open-question resurface/resolve.
  * **timing / throughput** — per-phase wall time + generated-token throughput.
  * **VRAM** — run-level and per-phase peaks (both allocated *and* reserved;
    reserved is the honest "will it fit in 24 GB" footprint), pinned to the
    context length that produced the peak.

This object holds no GPU logic: the runner reads CUDA's allocator counters and
feeds the numbers in via ``observe_vram`` so this module stays importable and
testable without a GPU (``python -m core.reflection_stats``).
"""

from __future__ import annotations

import time
from typing import Optional


# How rough ETA is computed: exchanges are the master clock. Generation cost is
# dominated by the per-exchange judgement + clean re-answer + branch + judge work,
# and the exchange
# count is the one quantity we can precount exactly, so seconds-per-completed-
# exchange × exchanges-remaining is a stable rough estimate. Consolidation time is
# folded into the same rate (it inflates the estimate slightly early, then settles).
class RunStats:
    def __init__(
        self,
        *,
        total_exchanges: int = 0,
        exchanges_without_cot: int = 0,
        open_questions_resurfaced: int = 0,
        branching_enabled: bool = False,
        model_id: Optional[str] = None,
        adapter_id: Optional[str] = None,
        context_length: Optional[int] = None,
        quant: Optional[str] = None,
        device_total_gb: Optional[float] = None,
        started_monotonic: Optional[float] = None,
    ) -> None:
        self._t0 = started_monotonic if started_monotonic is not None else time.monotonic()
        self._first_exchange_at: Optional[float] = None

        # provenance / footprint context (for the 24 GB question)
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.context_length = context_length
        self.quant = quant
        self.device_total_gb = device_total_gb
        self.branching_enabled = branching_enabled

        # precount + progress
        self.total_exchanges = int(total_exchanges)
        self.exchanges_without_cot = int(exchanges_without_cot)
        self.exchanges_done = 0
        self.consolidation_passes = 0
        self.consolidation_chunks = 0

        # outcomes
        self.verdict = {"keep": 0, "revise": 0}
        # cot_regen: a corrupt-CoT exchange the judge KEPT whose <think> was regenerated
        # by re-answering and grafted onto the (trusted) original reply — see
        # reflection_runner._regenerate_cot_for_kept_reply.
        self.chosen = {"original": 0, "ideal_win": 0, "branch_win": 0,
                       "cot_regen": 0, "judge_override": 0}
        # A corrupt-CoT kept exchange whose re-answer diverged too far from the original
        # reply (below the similarity floor), so the CoT could not be faithfully
        # regenerated and the reply trained answer-only (prior behavior). Not a discard —
        # a valid "original" target is still produced — tracked only for visibility.
        self.cot_regen_fallback = 0
        self.discarded = {
            "consolidation_gen_error": 0,
            "revision_gen_error": 0,
            "persist_error": 0,
            "revised_missing_ideal": 0,
            "branch_unparseable": 0,
            # A confirmed language-drift exchange whose clean re-answer could not be generated in
            # the conversation's language — dropped rather than train the drifted reply.
            "lang_drift_unrepaired": 0,
            # An exchange whose stored reply was operator-flagged corrupt and could not be
            # reconstructed into a usable clean re-answer — dropped rather than keep the corrupt reply.
            "corrupt_response_unrepaired": 0,
        }
        self.retries = {"attempted": 0, "recovered": 0}
        self.branch = {"eligible": 0, "skipped": {}}  # reason-category -> count
        self.rag = {"inserts": 0, "evicts": 0}
        self.persona_written = 0
        self.open_questions = {"resurfaced": int(open_questions_resurfaced),
                               "resolved": 0}
        self.judge = {"judged": 0, "overrides": 0}

        # Judge-override accounting: how often the branch judge's criterion flip re-pointed
        # the trainable target to a *branch* (a cell of interest for the flip's effect).
        # (The old prior-preservation regularizer cells were retired with the IDEAL
        # minority-copy slot — the from-scratch build trains one target per exchange.)
        self.judge_branch_overrides = 0       # judge overrides whose pick was a branch

        # timing / throughput: phase -> {"secs", "count", "tokens"}
        self.phase_time: dict[str, dict] = {}
        self._gen_secs_total = 0.0
        self._gen_tokens_total = 0

        # VRAM
        self.vram_peak_alloc_gb = 0.0
        self.vram_peak_reserved_gb = 0.0
        self.vram_phase_peak_gb: dict[str, float] = {}   # phase -> reserved gb
        self.vram_peak_context_tokens = 0

    # ── progress / timing ──────────────────────────────────────────────── #

    def record_phase(self, phase: str, secs: float, tokens: int = 0) -> None:
        """Accumulate one generation pass's wall time + generated tokens for *phase*."""
        slot = self.phase_time.setdefault(phase, {"secs": 0.0, "count": 0, "tokens": 0})
        slot["secs"] += max(0.0, float(secs))
        slot["count"] += 1
        slot["tokens"] += max(0, int(tokens))
        self._gen_secs_total += max(0.0, float(secs))
        self._gen_tokens_total += max(0, int(tokens))

    def note_exchange_done(self) -> None:
        if self._first_exchange_at is None:
            self._first_exchange_at = time.monotonic()
        self.exchanges_done += 1

    def note_consolidation_pass(self) -> None:
        self.consolidation_passes += 1

    # ── outcome counters ───────────────────────────────────────────────── #

    def note_verdict(self, verdict: Optional[str]) -> None:
        if verdict in self.verdict:
            self.verdict[verdict] += 1

    def note_chosen(self, kind: str) -> None:
        if kind in self.chosen:
            self.chosen[kind] += 1

    def note_cot_regen_fallback(self) -> None:
        """A corrupt-CoT kept exchange whose CoT could not be faithfully regenerated
        (re-answer diverged from the original reply); it trained answer-only."""
        self.cot_regen_fallback += 1

    def note_discard(self, kind: str) -> None:
        if kind in self.discarded:
            self.discarded[kind] += 1

    def note_retry(self, *, recovered: bool) -> None:
        self.retries["attempted"] += 1
        if recovered:
            self.retries["recovered"] += 1

    def note_branch_eligible(self) -> None:
        self.branch["eligible"] += 1

    def note_branch_skipped(self, category: str) -> None:
        self.branch["skipped"][category] = self.branch["skipped"].get(category, 0) + 1

    def note_rag(self, inserts: int = 0, evicts: int = 0) -> None:
        self.rag["inserts"] += max(0, int(inserts))
        self.rag["evicts"] += max(0, int(evicts))

    def note_resolved(self, n: int = 1) -> None:
        self.open_questions["resolved"] += max(0, int(n))

    def note_persona(self, n: int = 1) -> None:
        self.persona_written += max(0, int(n))

    def note_judge(self, *, judged: int = 0, overrides: int = 0) -> None:
        self.judge["judged"] += max(0, int(judged))
        self.judge["overrides"] += max(0, int(overrides))

    def note_judge_branch_override(self) -> None:
        """A judge override that flipped the trainable target to a branch."""
        self.judge_branch_overrides += 1

    def trained_pairs(self) -> int:
        """Persisted trainable targets — the denominator for the 'cases' percentages.
        (judge_override re-points one of these, it is not additive.)"""
        return (self.chosen["original"] + self.chosen["ideal_win"]
                + self.chosen["branch_win"] + self.chosen["cot_regen"])

    # ── VRAM ───────────────────────────────────────────────────────────── #

    def observe_vram(self, peak, phase: str = "", context_tokens: int = 0) -> None:
        """Fold one stage's CUDA peak (``(alloc_gb, reserved_gb)`` or None) into the
        run-level and per-phase high-water marks. The reserved figure is the honest
        footprint vs the device total. *context_tokens* pins the run peak to the
        sequence length that produced it (VRAM scales with seq len)."""
        if not peak:
            return
        alloc_gb, reserved_gb = peak
        if reserved_gb is not None:
            if reserved_gb > self.vram_peak_reserved_gb:
                self.vram_peak_reserved_gb = reserved_gb
                self.vram_peak_context_tokens = int(context_tokens or 0)
            if phase:
                prev = self.vram_phase_peak_gb.get(phase, 0.0)
                self.vram_phase_peak_gb[phase] = max(prev, reserved_gb)
        if alloc_gb is not None:
            self.vram_peak_alloc_gb = max(self.vram_peak_alloc_gb, alloc_gb)

    # ── derived ────────────────────────────────────────────────────────── #

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._t0)

    def eta_seconds(self) -> Optional[float]:
        """Rough seconds-remaining, or None until there's signal.

        Clock = revisable exchanges. Once at least one has completed, extrapolate
        the average seconds-per-exchange (measured from run start, so consolidation
        is folded in) over the exchanges still to go. Intentionally coarse — the
        end-of-run batched judge isn't modelled, so the true finish runs a little
        past this; fine for "leave the GPU unattended ~N hours"."""
        if self.total_exchanges <= 0 or self.exchanges_done <= 0:
            return None
        remaining = self.total_exchanges - self.exchanges_done
        if remaining <= 0:
            return 0.0
        rate = self.elapsed_seconds() / self.exchanges_done
        return rate * remaining

    def tokens_per_sec(self) -> Optional[float]:
        """Run-wide generated-token throughput across ALL recorded phases.

        Honest only while every record_phase call passes its generated tokens:
        a generation phase that logs seconds with tokens=0 dilutes this rate.
        (The branch phases once did exactly that — the panel read ~10 tok/s
        during the consolidation-only start and decayed toward ~4 as untokened
        branch seconds accumulated, looking like a GPU slowdown that wasn't.)
        """
        if self._gen_secs_total <= 0:
            return None
        return self._gen_tokens_total / self._gen_secs_total

    def discards_total(self) -> int:
        return sum(self.discarded.values())

    # ── outputs ────────────────────────────────────────────────────────── #

    def to_status(self) -> dict:
        """Compact live snapshot for the Sleep tab's stats panel (pushed via the
        run-state ``stats`` field, polled by ``reflection_run_status``)."""
        eta = self.eta_seconds()
        tps = self.tokens_per_sec()
        return {
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "exchanges_done": self.exchanges_done,
            "exchanges_total": self.total_exchanges,
            "vram_peak_reserved_gb": round(self.vram_peak_reserved_gb, 2) or None,
            "discards_total": self.discards_total(),
            "tokens_per_sec": round(tps, 1) if tps is not None else None,
        }

    def build_report(self) -> dict:
        """The full end-of-run report, persisted to ``<run_id>.report.json`` (and
        carried into the reflection archive). Self-contained for later analysis."""
        peak_reserved = round(self.vram_peak_reserved_gb, 2) or None
        headroom = None
        would_fit_24 = None
        if peak_reserved is not None:
            would_fit_24 = peak_reserved <= 24.0
        if peak_reserved is not None and self.device_total_gb:
            headroom = round(self.device_total_gb - peak_reserved, 2)

        per_phase = {}
        for phase, slot in self.phase_time.items():
            cnt = slot["count"] or 1
            per_phase[phase] = {
                "seconds": round(slot["secs"], 1),
                "passes": slot["count"],
                "avg_seconds": round(slot["secs"] / cnt, 2),
                "tokens_per_sec": (round(slot["tokens"] / slot["secs"], 1)
                                   if slot["secs"] > 0 else None),
            }

        return {
            "totals": {
                "exchanges": self.total_exchanges,
                "exchanges_done": self.exchanges_done,
                "exchanges_without_cot": self.exchanges_without_cot,
                "consolidation_passes": self.consolidation_passes,
            },
            "outcomes": {
                "verdict": dict(self.verdict),
                "chosen": dict(self.chosen),
                "cot_regen_fallback": self.cot_regen_fallback,
                "discarded": dict(self.discarded),
                "discards_total": self.discards_total(),
                "retries": dict(self.retries),
                "branch": {"eligible": self.branch["eligible"],
                           "skipped": dict(self.branch["skipped"])},
                "rag": dict(self.rag),
                "persona_written": self.persona_written,
                "open_questions": dict(self.open_questions),
                "judge": {**dict(self.judge),
                          "branch_overrides": self.judge_branch_overrides},
            },
            "timing": {
                "elapsed_seconds": round(self.elapsed_seconds(), 1),
                "per_phase": per_phase,
                "tokens_per_sec": (round(self.tokens_per_sec(), 1)
                                   if self.tokens_per_sec() is not None else None),
            },
            "vram": {
                "peak_allocated_gb": round(self.vram_peak_alloc_gb, 2) or None,
                "peak_reserved_gb": peak_reserved,
                "per_phase_peak_gb": {k: round(v, 2)
                                      for k, v in self.vram_phase_peak_gb.items()},
                "peak_at_context_tokens": self.vram_peak_context_tokens or None,
                "device_total_gb": (round(self.device_total_gb, 2)
                                    if self.device_total_gb else None),
                "headroom_vs_device_gb": headroom,
                "would_fit_24gb": would_fit_24,
                "model_id": self.model_id,
                "adapter_id": self.adapter_id,
                "quant": self.quant,
                "context_length": self.context_length,
            },
        }


def _selftest() -> None:
    s = RunStats(total_exchanges=4, exchanges_without_cot=1, branching_enabled=True,
                 model_id="m", context_length=32768, device_total_gb=24.0)
    s.note_consolidation_pass()
    s.record_phase("consolidation", 2.0, tokens=200)
    s.note_verdict("revise")
    s.note_chosen("branch_win")
    s.note_branch_eligible()
    s.note_branch_skipped("no eligible candidates")
    s.observe_vram((10.0, 12.5), phase="branch_gen", context_tokens=8000)
    s.observe_vram((11.0, 13.0), phase="branch_choose", context_tokens=9000)
    s.record_phase("branch_gen", 30.0, tokens=400)
    s.note_exchange_done()
    s.note_discard("revised_missing_ideal")
    s.note_judge(judged=1, overrides=1)
    s.note_judge_branch_override()
    rep = s.build_report()
    assert rep["vram"]["peak_reserved_gb"] == 13.0
    assert rep["vram"]["peak_at_context_tokens"] == 9000
    assert rep["vram"]["would_fit_24gb"] is True
    assert rep["outcomes"]["chosen"]["branch_win"] == 1
    assert rep["outcomes"]["discarded"]["revised_missing_ideal"] == 1
    assert rep["outcomes"]["judge"]["overrides"] == 1
    assert rep["outcomes"]["judge"]["branch_overrides"] == 1
    assert s.to_status()["exchanges_total"] == 4
    assert s.eta_seconds() is not None
    print("reflection_stats selftest OK:", s.to_status())


if __name__ == "__main__":
    _selftest()
