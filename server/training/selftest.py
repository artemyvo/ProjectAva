"""Self-tests for the consolidation backbone (no GPU, no model).

    cd server && .venv/bin/python -m training.selftest   # or any python with stdlib

Covers the parts that must be correct independent of training: the decay schedule,
the ledger fold + stage advance, render/inference parity, and a dry-run migration
against whatever artifacts are present.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

_INFERENCE = Path(__file__).resolve().parent.parent / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from core.chat_sidecar import ChatSidecar, sidecar_path_for  # noqa: E402
from core.chat_logger import ChatLogger, CHAT_SCHEMA_VERSION  # noqa: E402
from core.rag_policy import (  # noqa: E402
    chunk_text, memory_available_before, rank_score,
)
from training.decay import ConsolidationConfig, DecayConfig
from training.ledger import ConsolidationLedger, dialogue_key, fact_key
from training import render
from training.dialogue_source import build_dialogue_anchor, live_dialogue_anchors
from training.migrate import migrate, migrate_ledger_dialogue_to_sidecars
from training.reflections_path import (
    SFT_QUARANTINE_FILE, SFT_RENDER_FILE, consolidation_dir, hot_chats_dir, memory_dir,
)


def test_decay() -> None:
    cfg = DecayConfig(base_variants=4, decay_steps=4)
    got = [cfg.variants_for_stage(s) for s in range(6)]
    assert got == [4, 3, 2, 1, 0, 0], got
    assert not cfg.is_deprecated(3)
    assert cfg.is_deprecated(4)
    assert cfg.modifier_for_stage(0) == 1.0
    assert cfg.modifier_for_stage(2) == 0.5
    # zero span deprecates immediately
    assert DecayConfig(base_variants=3, decay_steps=0).is_deprecated(0)
    print("  decay: schedule 4,3,2,1,deprecate ✓")


def test_config_from_dict() -> None:
    c = ConsolidationConfig.from_dict({"dialogue": {"base_variants": 6, "decay_steps": 3},
                                       "decay_curve": "linear"})
    assert c.dialogue.base_variants == 6 and c.dialogue.decay_steps == 3
    assert c.fact.base_variants == 2          # default kept
    assert c.for_type("persona") is c.fact    # persona rides on fact config
    assert c.for_type("nonsense") is None
    # Wall-clock knobs (REBUILD retrofit): defaults, override, and untouched-default kept.
    assert c.wall.rag_only_window_h == 24.0 and c.wall.lora_cap_age_h == 72.0
    assert c.wall.rag_cap_age_h == 96.0 and c.wall.contamination_enabled is True
    assert c.wall.contamination_min_user_chars == 100     # default gate
    w = ConsolidationConfig.from_dict({"wall_clock": {
        "lora_cap_age_h": 48,
        "contamination": {"dose": 2.0, "min_user_chars": 40}}}).wall
    assert w.lora_cap_age_h == 48.0 and w.contamination_dose == 2.0
    assert w.contamination_min_user_chars == 40
    assert w.rag_cap_age_h == 96.0            # default kept when unspecified
    legacy_count = ConsolidationConfig.from_dict({"wall_clock": {"fresh_window": {
        "chats_since_penalty": 0.9, "chats_since_floor": 0.1,
    }}}).wall
    assert not hasattr(legacy_count, "chats_since_penalty")
    assert not hasattr(legacy_count, "chats_since_floor")
    print("  config: per-type override + defaults + wall-clock knobs ✓")


def test_triangular_lr() -> None:
    """Trapezoidal LR schedule (1 warmup + P plateau + 1 decay epoch): 0 at step 0, ramps to a
    flat 1.0 hold across the plateau epochs, decays back to 0, and the order-independence
    invariant (a row's passes — warmup i + P plateau holds + decay — sum to P+1 multipliers)."""
    from training.train_cycle import _triangular_lr_fraction as tri
    for plateau in (0, 1, 3, 4):   # 0 == plateau-free warmup+decay (2 epochs, per-row sum 1)
        total = plateau + 2
        for n in (1, 3, 7, 50):
            # every row i: warmup(i) + sum of P plateau passes + decay == plateau + 1
            for i in range(n):
                s = sum(tri(k * n + i, n, plateau) for k in range(total))
                assert abs(s - (plateau + 1)) < 1e-9, (plateau, n, i)
            assert tri(0, n, plateau) == 0.0                # warmup starts at zero
            assert abs(tri(n, n, plateau) - 1.0) < 1e-9     # peak reached at first epoch boundary
            for j in range(n, (1 + plateau) * n):           # whole plateau holds at max
                assert abs(tri(j, n, plateau) - 1.0) < 1e-9, (plateau, n, j)
            assert 0.0 <= tri(total * n - 1, n, plateau) <= 1.0   # decay tail stays in range
    assert tri(5, 0) == 1.0                         # degenerate (no rows) -> neutral
    assert abs(tri(3, 7, 1) - tri(3, 7)) < 1e-12    # default plateau == 1 (back-compat)
    print("  triangular_lr: warmup/plateau/decay + per-row avg invariant ✓")


def test_training_row_preparation() -> None:
    """Oversized final exchanges are refused intact; retained rows keep completion + LR identity."""
    from training.train_cycle import (
        _aligned_training_metadata,
        _completion_invariant_error,
        _prepare_training_rows,
        _render_and_tokenize_training_row,
    )

    class FakeGemmaTokenizer:
        eos_token = "<eos>"
        eos_token_id = 2
        tokenizer = None
        _special = {
            "<bos>": 1, "<eos>": 2, "<|turn>": 3, "<turn|>": 4,
            "<|channel>": 5, "<channel|>": 6,
        }

        def __init__(self) -> None:
            self.tokenizer = self

        def apply_chat_template(self, messages, *, tokenize=False,
                                add_generation_prompt=False, **_kwargs):
            assert not tokenize
            text = "<bos>"
            for message in messages:
                role = "model" if message["role"] == "assistant" else message["role"]
                text += f"<|turn>{role}\n{message.get('content', '')}<turn|>\n"
            if add_generation_prompt:
                text += "<|turn>model\n"
            return text

        def __call__(self, text, add_special_tokens=True):
            del add_special_tokens  # rendered text already carries the template's BOS/EOS
            ids, i = [], 0
            markers = sorted(self._special, key=len, reverse=True)
            while i < len(text):
                marker = next((m for m in markers if text.startswith(m, i)), None)
                if marker is not None:
                    ids.append(self._special[marker])
                    i += len(marker)
                else:
                    ids.append(1000 + ord(text[i]))
                    i += 1
            return {"input_ids": ids}

    tok = FakeGemmaTokenizer()
    final = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "<think>reason</think>answer"},
    ]
    safe_text, safe_ids = _render_and_tokenize_training_row(final, tok, "gemma")
    cap = len(safe_ids)
    assert safe_ids[-1] == tok.eos_token_id
    assert _completion_invariant_error(
        safe_text, safe_ids, final, tok, "gemma", "<|turn>model\n") is None

    # A broken close is rejected even when the row still fits the token cap.
    broken = safe_text.replace("<channel|>", "", 1)
    broken_ids = tok(broken)["input_ids"]
    assert _completion_invariant_error(
        broken, broken_ids, final, tok, "gemma", "<|turn>model\n")

    with_history = [
        final[0],
        {"role": "user", "content": "old user " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        final[1], final[2],
    ]
    oversized = [
        final[0],
        {"role": "user", "content": "irreducible prompt " * 30},
        final[2],
    ]
    examples = [
        {"messages": with_history, "anchor_key": "kept-0", "source": "chat",
         "lr_multiplier": 1.0, "unmask_user": False},
        {"messages": oversized, "anchor_key": "quarantine-1", "source": "wander",
         "lr_multiplier": 2.0, "unmask_user": False},
        {"messages": final, "anchor_key": "kept-2", "source": "chat",
         "lr_multiplier": 3.0, "unmask_user": True},
    ]
    prepared, quarantined, dropped = _prepare_training_rows(examples, tok, "gemma", cap)
    assert [row["train_row_id"] for row in prepared] == [0, 2]
    assert len(prepared[0]["example"]["messages"]) == 3 and dropped == 2
    assert len(quarantined) == 1
    assert quarantined[0]["train_row_id"] == 1
    assert quarantined[0]["reason"] == "irreducible_over_cap"

    # The quarantine gap itself does not shift row 2's metadata.
    multipliers, unmask = _aligned_training_metadata(
        prepared, [0, 2], [1.0, 3.0], [False, True])
    assert multipliers == [1.0, 3.0] and unmask == [False, True]
    # Simulate response masking then filtering retained row 0: survivor 2 still owns 3.0/True.
    multipliers, unmask = _aligned_training_metadata(prepared, [2], [3.0], [True])
    assert multipliers == [3.0] and unmask == [True]
    try:
        _aligned_training_metadata(prepared, [2], [2.0], [True])
    except RuntimeError as exc:
        assert "detached" in str(exc)
    else:
        raise AssertionError("misaligned LR metadata was accepted")
    print("  training_rows: irreducible quarantine + completion/EOS + LR identity ✓")


def test_wall_clock_age() -> None:
    """Wall-clock age core (REBUILD §1): chat-ts parsing, hours delta, the continuous
    0/ramp/cap LR multiplier, and the decoupled (later) RAG fade."""
    from training.decay import (parse_ts, wall_clock_age_hours, lr_multiplier_hours,
                                rag_weight_hours, verbatim_rag_weight_hours, cumsum_curve,
                                fresh_time_weight)
    full = ConsolidationConfig.from_dict(None)          # window 24h, lora cap 72h, rag cap 96h
    wall, dcfg = full.wall, full.for_type("dialogue")
    assert cumsum_curve(dcfg) == [3, 5, 6]      # legacy derived cumsum (fallback shape)
    assert list(wall.lr_ramp) == [1.0, 2.0, 4.0]  # explicit ramp sample points (current path)
    # parse a session stem (underscore or dash, suffix tolerated) and ISO to the same instant.
    assert parse_ts("20260201_000000") == parse_ts("2026-02-01T00:00:00")
    assert parse_ts("20260201-000000.json") == parse_ts("2026-02-01T00:00:00")
    assert parse_ts("garbage") is None and parse_ts(None) is None
    # hours delta; a build-before-chat delta clamps to 0; unparseable -> None.
    assert wall_clock_age_hours("20260129_000000", "2026-02-01T00:00:00") == 72.0
    assert wall_clock_age_hours("20260201_000000", "2026-01-01T00:00:00") == 0.0
    assert wall_clock_age_hours("nope", "2026-02-01T00:00:00") is None
    # LR multiplier: 0 below the window, 1→2→4 across [24h,72h], capped past, continuous.
    assert lr_multiplier_hours(10, dcfg, wall) == 0.0        # RAG-only window
    assert lr_multiplier_hours(None, dcfg, wall) == 0.0      # unparseable -> RAG-only
    assert lr_multiplier_hours(24, dcfg, wall) == 1.0        # window edge (jump 0→1)
    assert lr_multiplier_hours(48, dcfg, wall) == 2.0
    assert lr_multiplier_hours(72, dcfg, wall) == 4.0        # cap
    assert lr_multiplier_hours(1000, dcfg, wall) == 4.0      # capped past
    assert abs(lr_multiplier_hours(36, dcfg, wall) - 1.5) < 1e-9   # continuous midpoint 1→2
    # Verbatim chat: exact 1→0 over [0, rag_cap], then absent. Persona keeps the separate
    # rag_floor_weight safety floor.
    assert verbatim_rag_weight_hours(0, wall) == 1.0
    assert verbatim_rag_weight_hours(72, wall) == 0.25
    assert verbatim_rag_weight_hours(96, wall) == 0.0
    assert verbatim_rag_weight_hours(10_000, wall) == 0.0
    assert verbatim_rag_weight_hours(None, wall) == 1.0
    zero_cap = ConsolidationConfig.from_dict(
        {"wall_clock": {"rag_cap_age_h": 0}}).wall
    assert verbatim_rag_weight_hours(0, zero_cap) == 0.0
    assert rag_weight_hours(0, wall) == 1.0
    assert rag_weight_hours(96, wall) == 0.2
    assert rag_weight_hours(10_000, wall) == 0.2
    assert rag_weight_hours(None, wall) == 1.0
    # Gist tent: ramps 0→1 over [0, rag_cap], then interpolates to 0.2 EXACTLY at
    # gist_cap and holds. Unknown-age gist is withheld.
    from training.decay import gist_rag_weight_hours
    assert gist_rag_weight_hours(0, wall) == 0.0
    assert gist_rag_weight_hours(48, wall) == 0.5
    assert gist_rag_weight_hours(96, wall) == 1.0           # peak at rag_cap
    assert abs(gist_rag_weight_hours(144, wall) - 0.6) < 1e-9
    assert gist_rag_weight_hours(192, wall) == 0.2
    assert gist_rag_weight_hours(10_000, wall) == 0.2
    assert gist_rag_weight_hours(None, wall) == 0.0
    assert wall.gist_floor_weight == 0.2
    # Day-0 fresh-window droop: floor derived from verbatim_rag_weight_hours(24h) =
    # 1 - 0.5*(1-0.75) = 0.875 — gentler than the main curve (0.75 at 24h) since a fresh chat
    # is not yet in the weights. Hour-granular within day 0, held after, inert for unknown age.
    assert fresh_time_weight(0, wall) == 1.0
    assert abs(fresh_time_weight(24, wall) - 0.875) < 1e-9   # derived floor at horizon
    assert abs(fresh_time_weight(12, wall) - 0.9375) < 1e-9  # linear midpoint of first day
    assert abs(fresh_time_weight(1000, wall) - 0.875) < 1e-9  # held at floor, NOT faded to 0
    assert fresh_time_weight(None, wall) == 1.0            # unknown age inert
    print("  wall-clock age: parse, hours delta, 0/ramp/cap LR, verbatim expiry + gist floor, "
          "day-0 fresh-window droop ✓")


def test_ledger() -> None:
    cfg = ConsolidationConfig.from_dict({"dialogue": {"base_variants": 4, "decay_steps": 4}})
    with tempfile.TemporaryDirectory() as d:
        led = ConsolidationLedger(Path(d))
        dk = led.register_dialogue({"source_session": "s1.json", "exchange_index": 2,
                                    "prompt": "hi", "target": "hello", "speaker": "artemyvo"})
        fk = led.register_fact(content="Ava prefers terse replies", trigger="reply length")
        assert dk == dialogue_key("s1.json", 2)
        assert fk == fact_key("Ava prefers terse replies")

        folded = led.fold()
        assert folded[dk]["stage"] == 0 and folded[fk]["stage"] == 0
        assert len(led.live_anchors(cfg)) == 2

        # advance the dialogue to the brink, then over it -> deprecated, drops out
        for _ in range(3):
            led.advance([dk])
        assert led.stage_of(dk) == 3
        assert len(led.live_anchors(cfg)) == 2          # stage 3 -> 1 variant, still live
        led.advance([dk])
        assert led.stage_of(dk) == 4
        live_keys = {a["key"] for a in led.live_anchors(cfg)}
        assert dk not in live_keys and fk in live_keys   # dialogue deprecated, fact stays

        # re-register must NOT reset the accrued stage
        led.register_dialogue({"source_session": "s1.json", "exchange_index": 2,
                               "prompt": "hi again", "target": "hello"})
        assert led.stage_of(dk) == 4

        # Persona-editor cleanup is append-only but must disappear from every live fold.
        pk = led.register_fact(content="stale persona", item_type="persona",
                               source_session="s1.json", exchange_index=2)
        assert led.fold()[pk]["type"] == "persona"
        assert led.evict([pk]) == 1
        assert pk not in led.fold() and led.stage_of(pk) is None

        # Reconciliation "soften": supersede keeps the anchor folded but flags it, so it
        # drops out of live_anchors (training) while staying as evidence-of-change.
        sk = led.register_fact(content="outgrown persona", item_type="persona",
                               source_session="s2.json")
        assert sk in {a["key"] for a in led.live_anchors(cfg)}
        assert led.supersede([sk], reason="grew past it") == 1
        folded_s = led.fold()
        assert folded_s[sk].get("superseded") is True                 # still folded
        assert folded_s[sk].get("superseded_reason") == "grew past it"
        assert sk not in {a["key"] for a in led.live_anchors(cfg)}     # no longer trains
        # A later re-register reactivates it (clears the softened flag).
        led.register_fact(content="outgrown persona", item_type="persona",
                          source_session="s3.json")
        assert led.fold()[sk].get("superseded") is not True
        assert sk in {a["key"] for a in led.live_anchors(cfg)}
    print("  ledger: fold, advance, deprecation, re-register, eviction, supersede ✓")


def test_render_parity() -> None:
    anchor = {
        "system_prompt": "You are Ava.",
        "context": [
            {"role": "user", "content": "earlier q", "speaker": "artemyvo"},
            {"role": "assistant", "content": "earlier a"},
        ],
        "prompt": "what do you think?",
        "speaker": "artemyvo",
    }
    target = "<think>weighing it</think>I think it holds."
    render.assert_parity(anchor, target)
    msgs = render.build_messages(anchor, target)
    assert msgs[0] == {"role": "system", "content": "You are Ava."}
    assert msgs[1]["content"] == "artemyvo: earlier q"      # speaker-prefixed
    assert msgs[2]["content"] == "earlier a"                # assistant bare
    assert msgs[-2]["content"] == "artemyvo: what do you think?"
    assert msgs[-1] == {"role": "assistant", "content": target}

    # bare turn (no speaker) must stay bare, matching inference
    render.assert_parity({"system_prompt": "s", "context": [], "prompt": "q", "speaker": ""},
                         "a")

    # gemma-4 thinking denormalization: literal <think>X</think>Y -> native channel form
    # (the inverse of inference's _clean_response; gemma generates these as special tokens).
    assert (render.to_gemma_thinking_channel("<think>weighing it</think>I think it holds.")
            == "<|channel>thought\nweighing it\n<channel|>I think it holds.")
    # round-trips back through the inference normalizer
    import re as _re
    def _normalize(t):
        return _re.sub(r"<\|channel>thought\n?(.*?)\n?<channel\|>", r"<think>\1</think>",
                       t, flags=_re.DOTALL)
    _native = render.to_gemma_thinking_channel("<think>X</think>Y")
    assert _normalize(_native) == "<think>X</think>Y", _native
    # answer-only target gets an EMPTY closed channel (gemma's "no thinking" form), not a
    # bare answer — so a thinking-enabled prompt never trains the model to skip the channel.
    _empty = render.to_gemma_thinking_channel("just an answer")
    assert _empty == "<|channel>thought\n<channel|>just an answer", _empty
    assert _normalize(_empty) == "<think></think>just an answer", _empty   # round-trips

    # Nested/double-think guard: a vetted target that still carries its own <think> block
    # must NOT survive as a literal think block inside the trained answer span (that is the
    # empty-<think></think>-loop training bug). The producer strips the embedded block...
    from core.reflection_shareml import _verbatim_assistant
    assert (_verbatim_assistant("real cot", "<think>orig cot</think>\nthe answer")
            == "<think>real cot</think>\nthe answer")
    # ...and the renderer refuses any double-think that slips past the producer.
    try:
        render.to_gemma_thinking_channel("<think>cot</think><think>nested</think>ans")
        raise AssertionError("expected ValueError on nested-think target")
    except ValueError:
        pass
    print("  render: build_messages == _build_inference_conversation, gemma channel ✓")


def test_sidecar() -> None:
    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        chat_file = chats / "20260612_120000.json"
        chat_file.write_text('{"exchanges": [{"user_prompt": "hi"}]}', encoding="utf-8")
        assert sidecar_path_for(chat_file) == chats / "20260612_120000.state.json"

        sc = ChatSidecar(chats)
        assert sc.write_verdict(
            source_session="20260612_120000.json",
            exchange_index=0,
            verdict="revise",
            target="A better reply.",
            user_prompt="original user query",
            run_id="run1",
            target_source="revised",
            target_kind="ideal",
            target_generation="chat_reanswer_v1",
        )
        rec = sc.get_exchange("20260612_120000.json", 0)
        assert rec["stage"] == 0 and rec["verdict"] == "revise"
        assert rec["target"] == "A better reply." and rec["run_id"] == "run1"
        assert rec["user_prompt"] == "original user query"
        assert rec["target_kind"] == "ideal"
        assert rec["target_generation"] == "chat_reanswer_v1"

        # re-reflection preserves stage; last_trained untouched until advance
        sc.advance_stages("20260612_120000.json", [0], trained_at="2026-06-12T12:30:00")
        sc.write_verdict(
            source_session="20260612_120000.json",
            exchange_index=0,
            verdict="keep",
            target="A better reply.",
            run_id="run2",
            target_source="revised",
            target_kind="ideal",
            target_generation="chat_reanswer_v1",
        )
        rec = sc.get_exchange("20260612_120000.json", 0)
        assert rec["stage"] == 1 and rec["verdict"] == "keep" and rec["run_id"] == "run2"
        assert rec["last_trained"] == "2026-06-12T12:30:00"

        # live session guard
        assert not sc.write_verdict(
            source_session="20260612_120000.json",
            exchange_index=1,
            verdict="keep",
            target="blocked",
            run_id="run3",
            live_session="20260612_120000.json",
        )
        assert sc.get_exchange("20260612_120000.json", 1) is None

        # untrustworthy target skipped
        assert not sc.write_verdict(
            source_session="20260612_120000.json",
            exchange_index=2,
            verdict="revise",
            target="oops",
            run_id="run4",
            target_source="revised_missing_ideal",
        )

        # reflect-once freeze (REBUILD.md §3): mark_reflected stamps once, is_reflected
        # reads it, exchanges survive the stamp, and a second mark never overwrites.
        assert sc.is_reflected("20260612_120000.json") is False
        assert sc.mark_reflected("20260612_120000.json", reflected_at="T1") is True
        assert sc.is_reflected("20260612_120000.json") is True
        assert sc.get_exchange("20260612_120000.json", 0) is not None  # exchanges preserved
        sc.mark_reflected("20260612_120000.json", reflected_at="T2")
        assert sc.load("20260612_120000.json")["reflected_at"] == "T1"  # no overwrite
    print("  sidecar: verdict, stage preserve, advance, live guard, reflect-once ✓")


def test_chat_rag_decay() -> None:
    """Sidecar stage → dialogue decay modifier (chat-RAG eviction at stage N)."""
    cfg = ConsolidationConfig.from_dict({"dialogue": {"base_variants": 4, "decay_steps": 4}})
    decay = cfg.for_type("dialogue")
    assert decay is not None
    assert decay.modifier_for_stage(0) == 1.0
    assert decay.modifier_for_stage(2) == 0.5
    assert decay.modifier_for_stage(4) == 0.0
    assert decay.is_deprecated(4)

    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        chat_file = chats / "sess.json"
        chat_file.write_text(json.dumps({
            "user": "tester",
            "exchanges": [
                {"user_prompt": "q0", "assistant_response": "a0"},
                {"user_prompt": "q1", "assistant_response": "a1"},
            ],
        }), encoding="utf-8")
        sc = ChatSidecar(chats)
        sc.write_verdict(
            source_session="sess.json",
            exchange_index=0,
            verdict="keep",
            target="a0",
            run_id="run1",
        )
        for _ in range(4):
            sc.advance_stages("sess.json", [0], trained_at="2026-06-12T12:00:00")
        assert sc.stage_of("sess.json", 0) == 4
        assert decay.modifier_for_stage(sc.stage_of("sess.json", 0)) == 0.0
        assert sc.stage_of("sess.json", 1) == 0
        assert decay.modifier_for_stage(sc.stage_of("sess.json", 1)) == 1.0
    print("  chat-rag decay: sidecar stage → modifier, eviction at deprecate ✓")


def test_rag_policy() -> None:
    """RAG chronology/chunking policy stays correct without optional ML packages."""
    cutoff = "20260703_120000.json"
    # TIL/wiki provenance is not a timestamp. Its insert time makes it available to
    # subsequent chats and excludes it from conversations that predate the learning.
    external = {
        "source_session": "wiki:Многоязычный поиск",
        "available_at": "2026-07-02T10:30:00",
    }
    assert memory_available_before(external, cutoff)
    assert not memory_available_before(external, "20260701_120000.json")
    # Old chat-sourced rows without an op timestamp retain filename fallback.
    assert memory_available_before(
        {"source_session": "20260702_090000.json"}, cutoff,
    )
    # Unknown external chronology is conservative only for historical replay; live
    # chat has no cutoff and can still retrieve it.
    unknown = {"source_session": "til:legacy"}
    assert not memory_available_before(unknown, cutoff)
    assert memory_available_before(unknown, "")

    mixed = (("English context and Русский контекст. " * 12)
             + "ПОЗДНИЙ_РУССКИЙ_ФАКТ")
    chunks = chunk_text(mixed, max_chars=120, overlap_chars=20)
    assert len(chunks) > 1 and all(len(part) <= 120 for part in chunks)
    assert any("ПОЗДНИЙ_РУССКИЙ_ФАКТ" in part for part in chunks)

    # Age changes priority, not whether a semantically relevant item can pass — the shared
    # gate for the wander, chat, gist, and reflection-memory channels.
    assert abs(rank_score(0.20, 0.10, 0.15) - 0.02) < 1e-9
    assert rank_score(0.14, 1.0, 0.15) is None
    print("  rag policy: typed chronology + multilingual chunks + raw-relevance gate ✓")


def test_rag_engine_query_policy() -> None:
    """Post-search fences backfill results and never self-inject the active chat."""
    try:
        import numpy as np
        from core.rag_engine import RagEngine, _RetrievalEmbedder
    except ImportError:
        print("  rag engine query policy: skipped (RAG dependencies missing) ✓")
        return

    class FakeEmbedder:
        def encode_query(self, _text):
            return np.asarray([[1.0, 0.0]], dtype="float32")

        def encode_passages(self, values):
            return np.tile(
                np.asarray([[1.0, 0.0]], dtype="float32"), (len(values), 1))

    class FakeIndex:
        def __init__(self, scores, indices):
            self.scores = scores
            self.indices = indices
            self.last_k = 0

        def search(self, _query, k):
            self.last_k = k
            return (
                np.asarray([self.scores[:k]], dtype="float32"),
                np.asarray([self.indices[:k]], dtype="int64"),
            )

        def add(self, _vectors):
            pass

    class FakeBaseModel:
        def __init__(self):
            self.calls = []

        def encode(self, values, **_kwargs):
            self.calls.append(list(values))
            return np.tile(np.asarray([[1.0, 0.0]], dtype="float32"), (len(values), 1))

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        chats = root / "chats"
        prompts = root / "prompts"
        chats.mkdir()
        prompts.mkdir()
        rag = RagEngine(chats_dir=chats, prompts_dir=prompts)
        rag._embedder = FakeEmbedder()
        rag._prompt_template = "{context}"
        rag._refl_prompt_template = "{context}"
        rag._wander_prompt_template = "{context}"

        # The active session is the nearest neighbour, but must be removed with an
        # older hit backfilled instead of returning no context.
        active = "20260703_120000.json"
        old = "20260702_120000.json"
        rag._ready = True
        rag._entries = [
            {"user": "active duplicate", "response": "no", "speaker": "u",
             "source_session": active, "exchange_index": 0, "modifier": 1.0},
            {"user": "older memory", "response": "yes", "speaker": "u",
             "source_session": old, "exchange_index": 0, "modifier": 1.0},
        ]
        chat_index = FakeIndex([0.99, 0.80], [0, 1])
        rag._index = chat_index
        rag.set_current_session_file(Path(active))
        block = rag._query_chat("memory", 1)
        assert "older memory" in block and "active duplicate" not in block
        assert chat_index.last_k == len(rag._entries)

        # Reflection searches beyond the filtered top hit and uses real insertion
        # time for an external source id.
        rag._refl_entries = [
            {"kind": "fact", "display": "future note", "modifier": 1.0,
             "source_session": "til:future", "available_at": "2026-07-04T00:00:00"},
            {"kind": "fact", "display": "eligible TIL", "modifier": 1.0,
             "source_session": "wiki:тема", "available_at": "2026-07-02T00:00:00"},
        ]
        refl_index = FakeIndex([0.98, 0.70], [0, 1])
        rag._refl_index = refl_index
        block = rag._query_reflection("тема", before_session=active)
        assert "eligible TIL" in block and "future note" not in block
        assert refl_index.last_k == len(rag._refl_entries)

        # A relevant 0.1-age-weight wander remains eligible; age only lowers rank.
        rag._wander_entries = [
            {"display": "old but relevant", "modifier": 0.1, "source_id": "w1"},
        ]
        rag._wander_index = FakeIndex([0.20], [0])
        assert "old but relevant" in rag._query_wander("topic")

        # Incremental indexing is a no-op for the current transcript.
        before = list(rag._entries)
        rag.add_exchange("new", "reply", source_session=active, exchange_index=1)
        assert rag._entries == before
        # A non-active exchange still indexes successfully through the simplified
        # time-only modifier signature.
        rag.set_current_session_file(None)
        rag.add_exchange("new", "reply", source_session=old, exchange_index=1)
        assert len(rag._entries) == len(before) + 2

        # Both dialogue sides embed: an exchange yields user AND response passages
        # (tagged), so Ava's own past replies are searchable — and multiple passage
        # hits on ONE exchange still collapse to a single displayed snippet.
        pairs = RagEngine._exchange_passages("короткий вопрос", "a detailed answer")
        assert {src for _, src in pairs} == {"user", "response"}
        assert RagEngine._exchange_passages("q", "") == [("q", "user")]
        two_sided = [
            {"user": "unrelated question", "response": "the meaning of trains",
             "speaker": "u", "source_session": old, "exchange_index": 3,
             "modifier": 1.0, "embed_source": "user"},
            {"user": "unrelated question", "response": "the meaning of trains",
             "speaker": "u", "source_session": old, "exchange_index": 3,
             "modifier": 1.0, "embed_source": "response"},
        ]
        rag._entries = two_sided
        rag._index = FakeIndex([0.50, 0.85], [0, 1])   # response passage is the hit
        rag.set_current_session_file(None)
        block = rag._query_chat("trains", 2)
        assert block.count("the meaning of trains") == 1   # collapsed, reply-side hit

        # A long-running index can still contain vectors whose verbatim source crossed
        # the 96h boundary since its last rebuild. They are removed at query time, and
        # the candidate window grows by the expired-vector count so they cannot crowd
        # the eligible gist out of the post-FAISS result set.
        expired = [
            {"user": f"expired {i}", "response": "old", "speaker": "u",
             "source_session": f"expired-{i}", "exchange_index": i}
            for i in range(16)
        ]
        surviving_gist = {
            "kind": "gist", "content": "durable semantic memory",
            "source_session": "gist-source", "exchange_index": -1,
        }
        rag._entries = expired + [surviving_gist]
        rag._index = FakeIndex(
            [0.90 - i * 0.01 for i in range(17)],
            list(range(17)),
        )
        rag._chat_modifier = lambda source, **_kwargs: (
            0.0 if source.startswith("expired-") else 1.0)
        rag._gist_modifier = lambda _source: 0.2
        block = rag._query_chat("semantic", 1)
        assert "durable semantic memory" in block
        assert rag._index.last_k == len(rag._entries)

        # The literal hard cap is based on raw chat age, so it also removes an
        # unreflected bundle; before the cap that bundle keeps the empirical 0.875
        # fresh-window floor rather than pretending it has already reached weights.
        rag._consolidation = ConsolidationConfig.from_dict(None)
        rag._raw_age_of_session = lambda _source: 96.0
        rag._age_of_session = lambda _source: None
        assert RagEngine._chat_modifier(rag, "unfrozen.json") == 0.0
        rag._raw_age_of_session = lambda _source: 72.0
        assert RagEngine._chat_modifier(rag, "unfrozen.json") == 0.875

        # The multilingual adapter splits long mixed-language queries rather than
        # leaving sentence-transformers to truncate them at the beginning.
        base = FakeBaseModel()
        wrapped = _RetrievalEmbedder(base)
        wrapped.encode_query("English Русский " * 200)
        assert len(base.calls[0]) > 1
        wrapped.encode_passages(["память"])
        assert base.calls[-1] == ["память"]

    print("  rag engine query policy: backfill + no self-RAG + live hard expiry "
          "+ multilingual chunks + two-sided exchange embedding ✓")


def test_rag_engine_fallback() -> None:
    """RagEngine collects entries from both chats_dir and fallback_chats_dir, prioritizing chats_dir."""
    try:
        from core.rag_engine import RagEngine
    except ImportError:
        print("  rag engine fallback: skipped (RAG dependencies missing) ✓")
        return
    
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir_fallback = Path(d1)
        dir_staging = Path(d2)
        prompts = Path(tempfile.gettempdir())  # dummy prompts dir
        
        # 1. Write unique session 1 to fallback
        (dir_fallback / "sess1.json").write_text(json.dumps({
            "user": "tester",
            "exchanges": [
                {"user_prompt": "hello fallback", "assistant_response": "response fallback"},
            ],
        }), encoding="utf-8")
        
        # 2. Write unique session 2 to staging
        (dir_staging / "sess2.json").write_text(json.dumps({
            "user": "tester",
            "exchanges": [
                {"user_prompt": "hello staging", "assistant_response": "response staging"},
            ],
        }), encoding="utf-8")
        
        # 3. Write overlapping session 3 to both (with different content)
        (dir_fallback / "sess3.json").write_text(json.dumps({
            "user": "tester",
            "exchanges": [
                {"user_prompt": "hello overlapping fallback", "assistant_response": "response overlapping fallback"},
            ],
        }), encoding="utf-8")
        
        (dir_staging / "sess3.json").write_text(json.dumps({
            "user": "tester",
            "exchanges": [
                {"user_prompt": "hello overlapping staging", "assistant_response": "response overlapping staging"},
            ],
        }), encoding="utf-8")
        
        # Initialize RagEngine
        rag = RagEngine(
            chats_dir=dir_staging,
            prompts_dir=prompts,
            fallback_chats_dir=dir_fallback
        )
        
        entries, texts, _anchors, _anchor_texts = rag._collect_chat_entries()

        # Should collect 3 files: sess1 from fallback, sess2 from staging, sess3 from
        # staging — each exchange contributing a user + a response passage (two-sided
        # embedding), collapsed here by session for the dedup assertions.
        by_session = {}
        for e in entries:
            by_session.setdefault(e["source_session"], []).append(e)
        assert set(by_session) == {"sess1.json", "sess2.json", "sess3.json"}, by_session.keys()
        for sess, group in by_session.items():
            assert {e.get("embed_source") for e in group} == {"user", "response"}, sess

        # sess1
        assert by_session["sess1.json"][0]["user"] == "hello fallback"

        # sess2
        assert by_session["sess2.json"][0]["user"] == "hello staging"

        # sess3 (should be staging, not fallback)
        assert by_session["sess3.json"][0]["user"] == "hello overlapping staging"
        
    print("  rag engine fallback: scans both directories, prioritizing staging ✓")


def test_artifact_filenames() -> None:
    """Training render/quarantine use the expected filenames."""
    assert SFT_RENDER_FILE == "sft_render.jsonl"
    assert SFT_QUARANTINE_FILE == "sft_quarantine.jsonl"
    print("  artifacts: sft_render + sft_quarantine ✓")


def test_dialogue_source() -> None:
    cfg = ConsolidationConfig.from_dict({"dialogue": {"base_variants": 4, "decay_steps": 4}})
    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        chat_file = chats / "sess.json"
        chat_file.write_text(json.dumps({
            "system_prompt": "You are Ava.",
            "user": "tester",
            "exchanges": [
                {"user_prompt": "first", "assistant_response": "one", "speaker": "tester"},
                {"user_prompt": "second", "assistant_response": "two", "speaker": "tester",
                 "assistant_cot": "weighed it"},
            ],
        }), encoding="utf-8")
        sc = ChatSidecar(chats)
        sc.write_verdict(
            source_session="sess.json", exchange_index=1,
            verdict="keep", target="owned two", run_id="run1",
            target_source="original", target_kind="original",
            target_generation="original",
        )
        anchors = live_dialogue_anchors(chats, sc, cfg)
        assert len(anchors) == 1
        a = anchors[0]
        # target carries the original CoT + vetted answer (ShareML _verbatim_assistant form)
        assert a["prompt"] == "second"
        assert a["target"] == "<think>weighed it</think>\nowned two", a["target"]
        assert a["target_source"] == "original"
        assert a["target_kind"] == "original"
        assert a["target_generation"] == "original"
        # history context stays answer-only (no CoT leaked into prior turns)
        assert len(a["context"]) == 2
        assert a["context"][0]["content"] == "first"
        assert a["context"][0]["speaker"] == "tester"
        assert "<think>" not in a["context"][1]["content"]
        sc.advance_stages("sess.json", [1])
        for _ in range(3):
            sc.advance_stages("sess.json", [1])
        assert live_dialogue_anchors(chats, sc, cfg) == []
    print("  dialogue_source: sidecar + chat → trainable anchor ✓")


def test_dialogue_source_cot_provenance() -> None:
    """A target only ever pairs with a CoT that produced it: a branch win keeps the
    faithful <think> it carries; a legacy answer-only IDEAL is NOT given the original
    — mismatched — thought."""
    cfg = ConsolidationConfig.from_dict({"dialogue": {"base_variants": 4, "decay_steps": 4}})
    with tempfile.TemporaryDirectory() as d:
        chats = Path(d)
        (chats / "sess.json").write_text(json.dumps({
            "system_prompt": "You are Ava.",
            "user": "tester",
            "exchanges": [
                {"user_prompt": "q0", "assistant_response": "orig0", "speaker": "tester",
                 "assistant_cot": "reasoning that led to orig0"},
                {"user_prompt": "q1", "assistant_response": "orig1", "speaker": "tester",
                 "assistant_cot": "shared prefix thought"},
            ],
        }), encoding="utf-8")
        sc = ChatSidecar(chats)
        # idx0: legacy inline IDEAL win — judge text, answer-only, no generative lineage.
        sc.write_verdict(source_session="sess.json", exchange_index=0,
                         verdict="revise", target="a wholly better idea",
                         run_id="run1", target_source="revised")
        # idx1: branch win — resolve_revision_target already reattached the shared
        # <think> prefix, so the stored target is CoT-carrying.
        sc.write_verdict(source_session="sess.json", exchange_index=1,
                         verdict="revise",
                         target="<think>shared prefix thought</think>\n\nbranched reply",
                         run_id="run1", target_source="revised")
        by_idx = {a["exchange_index"]: a for a in live_dialogue_anchors(chats, sc, cfg)}
        # IDEAL stays answer-only — the original CoT is NOT injected.
        assert by_idx[0]["target"] == "a wholly better idea", by_idx[0]["target"]
        # Branch win trains verbatim with its faithful shared-prefix CoT.
        assert by_idx[1]["target"] == "<think>shared prefix thought</think>\n\nbranched reply", \
            by_idx[1]["target"]
    print("  dialogue_source: branch keeps faithful CoT, legacy IDEAL stays answer-only ✓")


def test_clean_ideal_reanswer() -> None:
    """A revised target is generated from the exact pre-answer training prefix.

    The judgement's WHY/inline IDEAL, the rejected CoT/reply, and post-reply feedback
    never enter either attempt. A malformed first completion retries the same clean
    conversation and only a CoT-bearing normal dialogue completion is accepted.
    """
    from core.reflection_config import (
        ReflectionRunConfig, ReflectionRunOverrides, ReflectionRunStore,
    )
    from core.reflection_runner import ReflectionRunner
    from core.reflection_source import build_ideal_messages, build_revision_jobs
    from core.reflection_writer import (
        ReflectionWriter, parse_revision_judgement,
    )
    from core.generation import clean_dialogue_response
    from reflection_run import _clean_ideal_response

    session = {
        "system_prompt": "SYSTEM PERSONA ONLY",
        "user": "tester",
        "exchanges": [
            {
                "user_prompt": "prior question",
                "assistant_cot": "PRIOR COT MUST NOT ENTER CONTEXT",
                "assistant_response": "prior visible answer",
                "speaker": "tester",
            },
            {
                "user_prompt": "current question",
                "assistant_cot": "REJECTED COT SECRET",
                "assistant_response": "REJECTED ANSWER SECRET",
                "speaker": "tester",
                "reflection_feedback": {"text": "POST REPLY FEEDBACK SECRET"},
            },
        ],
    }
    job = build_revision_jobs(session)[1]
    expected_prefix = render.build_messages({
        "system_prompt": session["system_prompt"],
        "context": job["context"],
        "prompt": job["exchange"]["user_prompt"],
        "speaker": job["speaker"],
    }, "placeholder target")[:-1]
    assert build_ideal_messages(job, session) == expected_prefix

    forbidden = (
        "PRIOR COT MUST NOT ENTER CONTEXT", "REJECTED COT SECRET",
        "REJECTED ANSWER SECRET", "POST REPLY FEEDBACK SECRET",
        "WHY: original reply was wrong", "INLINE JUDGE COT",
    )
    serialized = json.dumps(expected_prefix, ensure_ascii=False)
    assert all(marker not in serialized for marker in forbidden), serialized
    assert "prior visible answer" in serialized and "current question" in serialized

    leaked_raw = (
        "<think>fresh clean reasoning</think>fresh clean answer\n\n"
        "User\n\nA leaked follow-up turn"
    )
    cleaned = "<think>fresh clean reasoning</think>\n\nfresh clean answer"
    assert clean_dialogue_response(leaked_raw, "default-model") == cleaned
    assert _clean_ideal_response(leaked_raw, "default-model") == cleaned

    # Prompt overrides/older models may still emit inline IDEAL, but the judgement API
    # returns only the authoritative judgement fields.
    judgement = (
        "<think>INLINE JUDGE COT</think>\nVERDICT: revise\n"
        "WHY: original reply was wrong\nLANG_DRIFT: no\nPERSONA_TARGET: none\n"
        "IDEAL: <think>CONTAMINATED INLINE COT</think>contaminated answer"
    )
    parsed = parse_revision_judgement(judgement)
    assert parsed[0] == "revise" and parsed[1] == "original reply was wrong"
    assert len(parsed) == 3
    inline_only = judgement.replace("VERDICT: revise\n", "")
    assert parse_revision_judgement(inline_only)[0] is None

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = ReflectionRunStore(root / "runs")
        config = ReflectionRunConfig(
            run_id="clean-ideal", source="selftest", selected_sessions=[],
            overrides=ReflectionRunOverrides(),
        )
        store.create_run(config)
        writer = ReflectionWriter(root / "memory", root / "consolidation")
        runner = ReflectionRunner(
            chats_dir=root / "chats", memory_dir=root / "memory",
            runs_dir=root / "runs", consolidation_dir=root / "consolidation",
            reflection_writer=writer,
        )
        outputs = iter([
            "answer-only completion is not trainable",
            "<think>fresh reasoning from the clean prompt</think>fresh answer",
        ])
        calls: list[dict] = []

        def fake_generate(content: str, system_prompt: str, **kwargs) -> str:
            assert content == "" and system_prompt == ""
            calls.append(kwargs)
            return next(outputs)

        target = runner._generate_ideal_reply(
            job=job, session=session, generate_fn=fake_generate,
            temperature=0.8, top_p=0.95, max_new_tokens="512",
            require_user_language=False, filename="session.json",
            ex_idx=2, exchange_total=2, store=store, send_event_fn=None,
            run_id=config.run_id,
        )
        assert target == "<think>fresh reasoning from the clean prompt</think>\n\nfresh answer"
        assert len(calls) == 2
        for call in calls:
            assert call["disable_rag"] is True
            assert call["messages_override"] == expected_prefix
            attempt_text = json.dumps(call["messages_override"], ensure_ascii=False)
            assert all(marker not in attempt_text for marker in forbidden), attempt_text

        # A language constraint, when needed, changes only the system message; it does
        # not smuggle the rejected answer or reflection diagnosis into the conversation.
        constrained = build_ideal_messages(
            job, session, system_suffix="Reply in the final user's language."
        )
        assert constrained[1:] == expected_prefix[1:]
        assert constrained[0]["content"].startswith(session["system_prompt"])
        assert all(marker not in json.dumps(constrained, ensure_ascii=False)
                   for marker in forbidden)

    print("  clean IDEAL: exact pre-answer prefix, inline judge target ignored, clean retry ✓")


def test_build_history() -> None:
    """builds.jsonl (REBUILD.md §7): age = promoted builds after reflected_at; a rejected
    build advances no age."""
    from training.build_history import BuildHistory
    with tempfile.TemporaryDirectory() as d:
        models = Path(d)
        # Controlled timestamps: two promoted builds + one rejected between them.
        (models / "builds.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"outcome": "promoted", "ts": "2026-01-15T00:00:00"},
            {"outcome": "rejected", "ts": "2026-01-20T00:00:00"},
            {"outcome": "promoted", "ts": "2026-02-15T00:00:00"},
        ]) + "\n", encoding="utf-8")
        bh = BuildHistory(models)
        assert len(bh.promoted()) == 2
        # reflected before both promotions -> age 2 (the rejected one does not count)
        assert bh.age_of("2026-01-01T00:00:00") == 2, bh.age_of("2026-01-01T00:00:00")
        # reflected between the two promotions -> age 1
        assert bh.age_of("2026-02-01T00:00:00") == 1, bh.age_of("2026-02-01T00:00:00")
        # reflected after both -> age 0; no reflected_at -> age 0
        assert bh.age_of("2026-03-01T00:00:00") == 0
        assert bh.age_of(None) == 0
        # append round-trips and only promoted lines shift ages
        bh.append(outcome="rejected", rows=5, base_lr=1e-5)
        assert bh.age_of("2026-01-01T00:00:00") == 2
    print("  build_history: age counts promoted-after-reflected, rejected ignored ✓")


def test_build_snapshot() -> None:
    """Forensic snapshot (REBUILD §7): materialized COPIES (render/wander/persona/meta) under
    models/snapshots/<build_id>, best-effort, + build-line built_at/seed/snapshot_dir fields."""
    from training.build_snapshot import write_snapshot, snapshots_root
    from training.build_history import BuildHistory
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        models = root / "models"; models.mkdir()
        render = root / "sft_render.jsonl"
        render.write_text('{"text": "row0"}\n', encoding="utf-8")
        quarantine = root / "sft_quarantine.jsonl"
        quarantine.write_text('{"train_row_id": 4, "reason": "over"}\n', encoding="utf-8")
        bh = BuildHistory(models)
        build_id = bh.new_build_id()
        meta = {"build_id": build_id, "outcome": "promoted", "built_at": "2026-02-01T00:00:00",
                "seed": 42, "base_lr": 1e-5}
        snap = write_snapshot(models_dir=models, build_id=build_id, meta=meta,
                              render_path=render, quarantine_path=quarantine,
                              wander=[{"ts": "x", "target": "t"}],
                              persona_digest={"facets": {"VOICE": "plain"}})
        assert snap == snapshots_root(models) / build_id and snap.exists()
        # Materialized COPY, not a reference: mutating the source render leaves the snapshot intact.
        assert (snap / "sft_render.jsonl").read_text(encoding="utf-8") == '{"text": "row0"}\n'
        assert (snap / "sft_quarantine.jsonl").read_text(encoding="utf-8") == (
            '{"train_row_id": 4, "reason": "over"}\n')
        render.write_text("MUTATED\n", encoding="utf-8")
        assert (snap / "sft_render.jsonl").read_text(encoding="utf-8") == '{"text": "row0"}\n'
        assert json.loads((snap / "build_meta.json").read_text(encoding="utf-8"))["seed"] == 42
        assert json.loads((snap / "wander.json").read_text(encoding="utf-8"))[0]["target"] == "t"
        assert json.loads((snap / "persona_digest.json").read_text(encoding="utf-8")
                          )["facets"]["VOICE"] == "plain"
        # Missing render is a safe no-op (best-effort); the dir is still produced.
        snap2 = write_snapshot(models_dir=models, build_id="build-none", meta={"x": 1})
        assert snap2 is not None and not (snap2 / "sft_render.jsonl").exists()
        # The build line reuses the pre-generated build_id and records the provenance fields.
        rid = bh.append(outcome="promoted", rows=3, base_lr=1e-5, build_id=build_id,
                        built_at="2026-02-01T00:00:00", seed=42, snapshot_dir=str(snap))
        assert rid == build_id
        line = json.loads((models / "builds.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert line["build_id"] == build_id and line["built_at"] == "2026-02-01T00:00:00"
        assert line["seed"] == 42 and line["snapshot_dir"] == str(snap)
    print("  build_snapshot: materialized copies (render/quarantine/wander/persona/meta), reused "
          "build_id, provenance fields on the build line ✓")


def test_build_dataset() -> None:
    """From-scratch corpus (REBUILD.md §5a): chronological order, WALL-CLOCK age→multiplier
    ramp (RAG-only window skip, 1/2/4 at 24/48/72h), reproducibility against a pinned
    built_at, wander multiplier, ONE row per exchange (no variant copies), persona/fact
    injection sourced per-host, and the CoT rule (think-bearing vs empty-channel shapes)."""
    from training.build_dataset import build_dataset, corpus_fingerprint, WANDER_LR_MULT
    # Contamination OFF here so counts stay one-row-per-exchange — the cap-age 5+1 split is
    # covered by test_contamination; this test isolates age/ramp/injection/order.
    ccfg = ConsolidationConfig.from_dict({"wall_clock": {"contamination": {"enabled": False}}})
    built_at = "2026-02-01T00:00:00"                # the build's as_of clock
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        chats = root / "chats"; chats.mkdir()

        def _write(name, exchanges):
            (chats / name).write_text(json.dumps({
                "system_prompt": "You are Ava.", "user": "tester",
                "exchanges": exchanges}), encoding="utf-8")

        # Chat timestamps chosen so age = built_at − chat_ts lands on known rungs:
        #   0129 -> 72h (cap -> 4), 0130 -> 48h (2), 0131_000000 -> 24h (window edge -> 1),
        #   0131_140000 -> 10h (< 24h RAG-only window -> SKIPPED, no row),
        #   0128 -> 96h but UNreflected -> excluded by the reflect-once gate.
        _write("20260128_000000.json", [
            {"user_prompt": "z", "assistant_response": "z0", "speaker": "tester", "assistant_cot": "whyz"}])
        _write("20260129_000000.json", [
            {"user_prompt": "q0", "assistant_response": "a0", "speaker": "tester", "assistant_cot": "why0"},
            {"user_prompt": "q1", "assistant_response": "a1", "speaker": "tester"}])   # CoT-less
        _write("20260130_000000.json", [
            {"user_prompt": "r0", "assistant_response": "b0", "speaker": "tester", "assistant_cot": "why-r0"}])
        _write("20260131_000000.json", [
            {"user_prompt": "s0", "assistant_response": "c0", "speaker": "tester", "assistant_cot": "why-s0"}])
        _write("20260131_140000.json", [
            {"user_prompt": "t0", "assistant_response": "d0", "speaker": "tester", "assistant_cot": "why-t0"}])

        sc = ChatSidecar(chats)
        sc.write_verdict(source_session="20260129_000000.json", exchange_index=0, verdict="keep",
                         target="<think>why0</think>\nowned a0", run_id="r1", target_source="original")
        sc.write_verdict(source_session="20260129_000000.json", exchange_index=1, verdict="keep",
                         target="a1", run_id="r1", target_source="original")
        sc.mark_reflected("20260129_000000.json", reflected_at="2026-01-29T01:00:00")
        sc.write_verdict(source_session="20260130_000000.json", exchange_index=0, verdict="keep",
                         target="<think>why-r0</think>\nowned b0", run_id="r2", target_source="original")
        sc.mark_reflected("20260130_000000.json", reflected_at="2026-01-30T01:00:00")
        sc.write_verdict(source_session="20260131_000000.json", exchange_index=0, verdict="keep",
                         target="<think>why-s0</think>\nowned c0", run_id="r3", target_source="original")
        sc.mark_reflected("20260131_000000.json", reflected_at="2026-01-31T01:00:00")
        sc.write_verdict(source_session="20260131_140000.json", exchange_index=0, verdict="keep",
                         target="<think>why-t0</think>\nowned d0", run_id="r4", target_source="original")
        sc.mark_reflected("20260131_140000.json", reflected_at="2026-01-31T15:00:00")
        # 20260128 left UNreflected on purpose (reflect-once gate excludes it).
        sc.write_verdict(source_session="20260128_000000.json", exchange_index=0, verdict="keep",
                         target="<think>whyz</think>\nowned z0", run_id="r0", target_source="original")

        # A persona + a placed fact, both hosted on 20260129#0 (the cap-age bundle).
        ledger = ConsolidationLedger(root)
        ledger.register_fact(content="I am deeply curious about everything.", item_type="persona",
                             source_exchange={"source_session": "20260129_000000.json", "exchange_index": 0})
        ledger.register_fact(content="The user's name is Tester.", item_type="fact",
                             trigger="who is the user",
                             source_exchange={"source_session": "20260129_000000.json", "exchange_index": 0})

        wander = [{"ts": "20260303-120000", "system_prompt": "You are Ava.",
                   "prompt": "wp", "target": "<think>learned</think>\nfact learned"}]
        rows = build_dataset(chats_dirs=[chats], ledger=ledger, ccfg=ccfg,
                             built_at=built_at, wander_pending=wander)

        # Chronological order (bundle ts, then exchange index; wander by its own ts last).
        assert [r.order_key for r in rows] == sorted(r.order_key for r in rows)
        by = {(r.source_session, r.exchange_index): r for r in rows}
        # ONE row per revisable exchange. Excluded: the unreflected chat and the <window chat.
        chat_rows = [r for r in rows if r.source == "chat"]
        assert len(chat_rows) == 4, [(r.source_session, r.exchange_index) for r in chat_rows]
        assert ("20260128_000000.json", 0) not in by      # unreflected -> excluded
        assert ("20260131_140000.json", 0) not in by      # age 10h < 24h -> RAG-only, no row

        # Wall-clock age -> multiplier ramp: 72h -> 4, 48h -> 2, 24h (window edge) -> 1.
        assert by[("20260129_000000.json", 0)].age == 72.0
        assert by[("20260129_000000.json", 0)].lr_multiplier == 4.0
        assert by[("20260130_000000.json", 0)].lr_multiplier == 2.0
        assert by[("20260131_000000.json", 0)].lr_multiplier == 1.0

        # Reproducibility: same corpus + same built_at -> identical fingerprint + multipliers.
        rows2 = build_dataset(chats_dirs=[chats], ledger=ledger, ccfg=ccfg,
                              built_at=built_at, wander_pending=wander)
        assert corpus_fingerprint(rows) == corpus_fingerprint(rows2)
        assert [r.lr_multiplier for r in rows] == [r.lr_multiplier for r in rows2]

        # Wander: fixed multiplier, one-shot, sorts last.
        wanders = [r for r in rows if r.source == "wander"]
        assert len(wanders) == 1 and wanders[0].lr_multiplier == float(WANDER_LR_MULT)
        assert rows[-1].source == "wander"

        # CoT rule: a captured-CoT exchange is think-bearing; a CoT-less one is answer-only.
        assert "<think>" in by[("20260129_000000.json", 0)].target
        assert "why0</think>" in by[("20260129_000000.json", 0)].target
        assert by[("20260129_000000.json", 1)].target == "a1"
        assert "<think>" not in by[("20260129_000000.json", 1)].target

        # A think-bearing target with no persona/fact host stays un-injected.
        assert by[("20260130_000000.json", 0)].target == "<think>why-r0</think>\nowned b0"

        # Fact injection is sourced per host exchange (20260129#0 only); PERSONA injection
        # was retired (persona now reaches the CoT implicitly), so the persona statement is
        # NOT prepended and persona_keys is always empty — only the fact rides the host.
        host = by[("20260129_000000.json", 0)]
        assert host.fact_keys and not host.persona_keys
        assert "I am deeply curious about everything." not in host.target
        assert "The user's name is Tester." in host.target
        assert not by[("20260130_000000.json", 0)].persona_keys

        # include_fresh (Sleep → "Include fresh chats"): the excluded-but-target-bearing
        # shapes come back as PREVIEW rows — multiplier 0, `preview=True`, render-only,
        # single row per exchange (no contamination pair) — while the trained corpus is
        # byte-identical (same fingerprint over the non-preview rows). Covered shapes:
        # the <window frozen chat (10h) and a chat_reflected-only (background stage-one)
        # chat; a chat with NO freeze stamp at all still yields nothing.
        sc.mark_chat_reflected("20260128_000000.json", chat_reflected="2026-01-28T02:00:00")
        _write("20260127_000000.json", [
            {"user_prompt": "u0", "assistant_response": "e0", "speaker": "tester",
             "assistant_cot": "why-u0"}])
        sc.write_verdict(source_session="20260127_000000.json", exchange_index=0,
                         verdict="keep", target="<think>why-u0</think>\nowned e0",
                         run_id="r5", target_source="original")   # verdict, NO freeze stamp
        rows_f = build_dataset(chats_dirs=[chats], ledger=ledger, ccfg=ccfg,
                               built_at=built_at, wander_pending=wander,
                               include_fresh=True)
        trained_f = [r for r in rows_f if not r.preview]
        preview_f = {(r.source_session, r.exchange_index): r for r in rows_f if r.preview}
        assert corpus_fingerprint(trained_f) == corpus_fingerprint(rows)
        assert set(preview_f) == {("20260131_140000.json", 0), ("20260128_000000.json", 0)}
        assert not any(r.source_session == "20260127_000000.json" for r in rows_f)
        assert all(r.lr_multiplier == 0.0 and not r.unmask_user
                   for r in preview_f.values())
        # Preview rows still render real targets (reviewable/repairable content).
        assert preview_f[("20260131_140000.json", 0)].target == "<think>why-t0</think>\nowned d0"
        # And the default path is untouched: no preview rows without the flag.
        assert not any(r.preview for r in rows)
    print("  build_dataset: chronological, wall-clock age→mult 1/2/4 + window skip, "
          "reproducible, wander mult, single-target, per-host FACT injection "
          "(persona injection retired), CoT shapes, include_fresh preview rows ✓")


def test_preview_build() -> None:
    """GPU-free preview snapshot (training/preview_build.py): the training-lite half of
    Sleep's "Include fresh chats". Assembles the NEXT build's corpus with include_fresh,
    writes models/snapshots/preview-<ts>/ (render: would-train rows first, preview rows
    trailing; build_meta outcome "preview"), and prunes older preview-* dirs while
    leaving real build-* snapshots alone."""
    from training.build_dataset import row_render_dict  # noqa: F401 (schema shared)
    from training.preview_build import build_preview_snapshot, PREVIEW_PREFIX
    ccfg_dict = {"wall_clock": {"contamination": {"enabled": False}}}
    built_at = "2026-02-01T00:00:00"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        chats = root / "chats"; chats.mkdir()
        models = root / "models"
        (models / "snapshots" / "build-old").mkdir(parents=True)      # a real build
        (models / "snapshots" / "preview-00000000-000000").mkdir()    # a stale preview

        def _write(name, exchanges):
            (chats / name).write_text(json.dumps({
                "system_prompt": "You are Ava.", "user": "tester",
                "exchanges": exchanges}), encoding="utf-8")

        # One trainable bundle (48h -> mult 2) and one fresh one (10h -> preview).
        _write("20260130_000000.json", [
            {"user_prompt": "r0", "assistant_response": "b0", "speaker": "tester",
             "assistant_cot": "why-r0"}])
        _write("20260131_140000.json", [
            {"user_prompt": "t0", "assistant_response": "d0", "speaker": "tester",
             "assistant_cot": "why-t0"}])
        sc = ChatSidecar(chats)
        sc.write_verdict(source_session="20260130_000000.json", exchange_index=0,
                         verdict="keep", target="<think>why-r0</think>\nowned b0",
                         run_id="r1", target_source="original")
        sc.mark_reflected("20260130_000000.json", reflected_at="2026-01-30T01:00:00")
        sc.write_verdict(source_session="20260131_140000.json", exchange_index=0,
                         verdict="keep", target="<think>why-t0</think>\nowned d0",
                         run_id="r2", target_source="original")
        sc.mark_reflected("20260131_140000.json", reflected_at="2026-01-31T15:00:00")

        info = build_preview_snapshot(
            run_id="run-x", models_dir=models, chats_dirs=[chats],
            cons_dir=root, config={"consolidation": ccfg_dict},
            wander_pending=[], built_at=built_at)
        assert info["build_id"].startswith(PREVIEW_PREFIX)
        assert info["trained_rows"] == 1 and info["preview_rows"] == 1
        snap = Path(info["path"])
        rows = [json.loads(l) for l in
                (snap / "sft_render.jsonl").read_text(encoding="utf-8").splitlines()]
        # Would-train row first at its real multiplier, preview row trailing at 0.
        assert rows[0]["source_session"] == "20260130_000000.json"
        assert rows[0]["lr_multiplier"] == 2.0 and "preview" not in rows[0]
        assert rows[1]["source_session"] == "20260131_140000.json"
        assert rows[1]["preview"] is True and rows[1]["lr_multiplier"] == 0.0
        meta = json.loads((snap / "build_meta.json").read_text(encoding="utf-8"))
        assert meta["outcome"] == "preview" and meta["run_id"] == "run-x"
        # Prune: the stale preview is gone, the real build snapshot survives.
        assert info["pruned"] == 1
        left = {p.name for p in (models / "snapshots").iterdir()}
        assert left == {"build-old", info["build_id"]}, left
    print("  preview_build: preview-<ts> snapshot (would-train + trailing preview rows, "
          "outcome 'preview'), stale-preview prune, real builds untouched ✓")


def test_contamination() -> None:
    """Cap-age user contamination (REBUILD §5e): at cap a chat exchange emits a masked 3.0 +
    unmask 1.0 pair (Ava's response keeps the full 4.0 cap; her voice entrains at dose 1.0);
    below cap a single masked row; disabled -> single 4.0; additive -> 4.0 masked + 1.0.
    A short final user turn (< min_user_chars) skips the unmask copy — single 4.0 row."""
    from training.build_dataset import build_dataset

    # Long enough to clear the default 100-char contamination gate.
    LONG_Q = "q " * 60

    def rows_for(wall_over, user_prompt=LONG_Q):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chats = root / "chats"; chats.mkdir()
            # cap-age (72h) + sub-cap (48h) frozen bundles, one exchange each, vs built_at Feb 1.
            for name, cot in (("20260129_000000.json", "why"), ("20260130_000000.json", "why-r")):
                (chats / name).write_text(json.dumps({
                    "system_prompt": "You are Ava.", "user": "tester",
                    "exchanges": [{"user_prompt": user_prompt, "assistant_response": "a",
                                   "speaker": "tester", "assistant_cot": cot}]}), encoding="utf-8")
            sc = ChatSidecar(chats)
            sc.write_verdict(source_session="20260129_000000.json", exchange_index=0, verdict="keep",
                             target="<think>why</think>\nowned a", run_id="r1", target_source="original")
            sc.mark_reflected("20260129_000000.json", reflected_at="2026-01-29T01:00:00")
            sc.write_verdict(source_session="20260130_000000.json", exchange_index=0, verdict="keep",
                             target="<think>why-r</think>\nowned b", run_id="r2", target_source="original")
            sc.mark_reflected("20260130_000000.json", reflected_at="2026-01-30T01:00:00")
            ccfg = ConsolidationConfig.from_dict({"wall_clock": wall_over} if wall_over else None)
            return build_dataset(chats_dirs=[chats], ledger=ConsolidationLedger(root),
                                 ccfg=ccfg, built_at="2026-02-01T00:00:00")

    def cap_of(rows): return [r for r in rows if r.source_session == "20260129_000000.json"]

    # Default (contamination enabled, split): cap exchange -> 3.0 masked + 1.0 unmask.
    rows = rows_for(None)
    cap = cap_of(rows)
    assert sorted(r.lr_multiplier for r in cap) == [1.0, 3.0]
    assert sum(1 for r in cap if r.unmask_user) == 1
    assert sum(r.lr_multiplier for r in cap) == 4.0            # response keeps the full cap
    m = {r.unmask_user: r for r in cap}                        # copies share target/anchor
    assert m[True].target == m[False].target and m[True].lr_multiplier == 1.0
    # sub-cap (48h) exchange is untouched: a single masked row at 2.0.
    sub = [r for r in rows if r.source_session == "20260130_000000.json"]
    assert len(sub) == 1 and not sub[0].unmask_user and sub[0].lr_multiplier == 2.0

    # Disabled: cap exchange is a single masked 4.0 row.
    cap = cap_of(rows_for({"contamination": {"enabled": False}}))
    assert len(cap) == 1 and not cap[0].unmask_user and cap[0].lr_multiplier == 4.0

    # Additive: 4.0 masked + 1.0 unmask (dose ON TOP of the cap).
    cap = cap_of(rows_for({"contamination": {"additive": True}}))
    assert sorted(r.lr_multiplier for r in cap) == [1.0, 4.0]
    assert sum(1 for r in cap if r.unmask_user) == 1

    # Short final user turn (< default 100 chars): gated -> single masked 4.0 row, no unmask
    # (split-mode LR-neutral). "ok" would only teach terse user-style filler.
    cap = cap_of(rows_for(None, user_prompt="ok"))
    assert len(cap) == 1 and not cap[0].unmask_user and cap[0].lr_multiplier == 4.0
    # ...and additive-mode drops the extra dose entirely (5.0 -> 4.0).
    cap = cap_of(rows_for({"contamination": {"additive": True}}, user_prompt="ok"))
    assert len(cap) == 1 and not cap[0].unmask_user and cap[0].lr_multiplier == 4.0
    # Gate off (min_user_chars 0): a short turn contaminates again -> 3+1 split.
    cap = cap_of(rows_for({"contamination": {"min_user_chars": 0}}, user_prompt="ok"))
    assert sorted(r.lr_multiplier for r in cap) == [1.0, 3.0]

    # Fold: the split PAIR collapses to ONE weighted row (response_total 4.0, user weight
    # dose/response_total = 0.25) — halves the cap-age row count.
    cap = cap_of(rows_for({"contamination": {"fold": True}}))
    assert len(cap) == 1, cap
    assert cap[0].unmask_user and cap[0].lr_multiplier == 4.0          # response at full cap
    assert abs(cap[0].user_loss_weight - 0.25) < 1e-9                  # user at dose/cap
    # ...and every non-folded (split/additive/gated) row leaves user_loss_weight None.
    assert all(r.user_loss_weight is None for r in rows_for(None))
    # Fold + additive: response_total = cap + dose = 5.0, user weight = 1/5 = 0.2.
    cap = cap_of(rows_for({"contamination": {"fold": True, "additive": True}}))
    assert len(cap) == 1 and cap[0].lr_multiplier == 5.0
    assert abs(cap[0].user_loss_weight - 0.2) < 1e-9
    # Fold + short user turn (gated): single masked 4.0 row, no weight (like split-gated).
    cap = cap_of(rows_for({"contamination": {"fold": True}}, user_prompt="ok"))
    assert len(cap) == 1 and not cap[0].unmask_user and cap[0].user_loss_weight is None
    print("  contamination: cap-age 3+1 split (resp keeps 4.0), fold->1 weighted row, "
          "sub-cap single, "
          "disabled single, additive 4+1, short-user gated ✓")


def test_rag_crossfade() -> None:
    """REBUILD §6: verbatim RAG uses wall-clock age since the chat, on a pace
    DECOUPLED from (later than) the LoRA cap — the composition rag_engine applies at
    retrieval time. As a bundle's LR climbs 0→4 across [24h,72h], its RAG weight fades
    1.0→0 over rag_cap_age_h (96h), staying NONZERO at the LoRA cap and disappearing
    exactly when gist reaches peak."""
    from training.decay import (verbatim_rag_weight_hours, wall_clock_age_hours,
                                lr_multiplier_hours)
    full = ConsolidationConfig.from_dict(None)
    wall, dcfg = full.wall, full.for_type("dialogue")
    built = "2026-02-01T00:00:00"                    # retrieval-time "now" (pinned for the test)

    def weight(chat_ts):  # what _chat_modifier computes
        return verbatim_rag_weight_hours(wall_clock_age_hours(chat_ts, built), wall)

    def mult(chat_ts):    # the paired training LR multiplier
        return lr_multiplier_hours(wall_clock_age_hours(chat_ts, built), dcfg, wall)

    # fresh (age 0): full RAG, and no weight dose yet (RAG-only window).
    assert weight("20260201_000000") == 1.0 and mult("20260201_000000") == 0.0
    # 72h: LoRA at cap (4) but RAG still 0.25 — decoupled, RAG lingers past the LoRA cap.
    assert mult("20260129_000000") == 4.0 and weight("20260129_000000") == 0.25
    # 96h: verbatim is absent; gist is the chat-RAG representation from this point onward.
    assert weight("20260128_000000") == 0.0 and mult("20260128_000000") == 4.0
    print("  rag crossfade: verbatim fades 1.0→0 over rag_cap (96h), "
          "gist handoff at peak ✓")


def test_resolve_revision_target() -> None:
    """Every trained target carries a faithful CoT. A CoT-bearing IDEAL (the model
    authored its own <think> for the revised reply) trains as `revised`; a missing or
    answer-only IDEAL is refused as `revised_missing_ideal`, never replaced with the
    rejected original. keep/original keep their own <think>; a branch win reattaches
    the original <think> prefix it continued, with no double-wrap."""
    from core.reflection_writer import (
        ideal_trainable_target, resolved_target_provenance, resolve_revision_target,
    )

    # ideal_trainable_target: CoT-bearing accepted (normalized), answer-only/unusable -> None.
    assert ideal_trainable_target("<think>weighed it</think>better") == \
        "<think>weighed it</think>\n\nbetter"
    assert ideal_trainable_target("a better answer") is None       # answer-only
    assert ideal_trainable_target("<think></think>better") is None  # empty thought
    assert ideal_trainable_target("<think>only thinking</think>") is None  # no answer

    # revise verdict, answer-only IDEAL -> no target. Falling back to the rejected
    # original would turn a failed correction into an endorsed training row.
    target, source = resolve_revision_target(
        "revise", "a better answer", "orig", None, assistant_cot="orig cot")
    assert source == "revised_missing_ideal", source
    assert target == "", target

    # revise verdict, CoT-bearing IDEAL -> trained verbatim with its self-authored CoT.
    target, source = resolve_revision_target(
        "revise", "<think>I'd actually push back here</think>No.", "orig", None,
        assistant_cot="orig cot")
    assert source == "revised", source
    assert target == "<think>I'd actually push back here</think>\n\nNo.", target
    assert resolved_target_provenance(source, None) == ("ideal", "chat_reanswer_v1")

    # keep -> original reply with its own faithful CoT.
    target, source = resolve_revision_target(
        "keep", "", "orig", None, assistant_cot="own cot")
    assert source == "original", source
    assert target == "<think>own cot</think>\n\norig", target
    assert resolved_target_provenance(source, None) == ("original", "original")

    # Branch win -> branch text + the original <think> prefix it continued (no double-wrap).
    branch = {"chosen_index": 0, "candidates": [
        {"kind": "branch", "text": "branched reply"}]}
    target, source = resolve_revision_target(
        "revise", "<think>x</think>a better answer", "orig", branch,
        assistant_cot="shared prefix thought")
    assert source == "revised", source
    assert target == "<think>shared prefix thought</think>\n\nbranched reply", target
    assert target.count("<think>") == 1, target
    assert resolved_target_provenance(source, branch) == ("branch", "branch_replay")

    # Original blind win -> the original reply (with CoT), tagged original.
    branch_orig = {"chosen_index": 0, "candidates": [
        {"kind": "original", "text": "orig"}]}
    _, source = resolve_revision_target(
        "revise", "<think>x</think>a better answer", "orig", branch_orig,
        assistant_cot="own cot")
    assert source == "original", source

    # IDEAL blind win, CoT-bearing -> trains the IDEAL with its authored CoT.
    branch_ideal = {"chosen_index": 0, "candidates": [
        {"kind": "ideal", "text": "a better answer"}]}
    target, source = resolve_revision_target(
        "revise", "<think>here is why</think>a better answer", "orig", branch_ideal,
        assistant_cot="own cot")
    assert source == "revised", source
    assert target == "<think>here is why</think>\n\na better answer", target

    # IDEAL blind win, answer-only -> no target. The blind choice explicitly rejected
    # the original, so it must not silently regain authority.
    branch_ideal_bare = {"chosen_index": 0, "candidates": [
        {"kind": "ideal", "text": "a better answer"}]}
    target, source = resolve_revision_target(
        "revise", "a better answer", "orig", branch_ideal_bare, assistant_cot="own cot")
    assert source == "revised_missing_ideal", source
    assert target == "", target
    assert resolved_target_provenance(source, branch_ideal_bare) == ("none", "none")

    # Leaked model special-token markers (gemma-4 <channel|>/<eos>, qwen <|im_end|>) are
    # stripped from parsed persona/fact content; legitimate prose (x < y) is untouched.
    from core.reflection_writer import _parse_persona_field, parse_consolidation
    p = _parse_persona_field("I enjoy absurd logic. <channel|> <eos>")
    assert p and "channel" not in p[0] and "eos" not in p[0], p
    wc = parse_consolidation("## WEIGHTS\n- [persona] I value directness <channel|>\n"
                             "- [fact] x < y holds <|im_end|>")
    contents = [w["content"] for w in wc["weights"]]
    assert all("channel" not in c and "im_end" not in c for c in contents), contents
    assert any("x < y holds" == c for c in contents), contents   # comparison not eaten

    # A token-cap truncation drops the cut-off final item of the LAST-emitted section
    # only; complete items (and earlier sections) survive.
    trunc_text = ("## WEIGHTS\n- [fact] alpha\n- [fact] beta\n"
                  "## RAG\n- [ask:user] what is your goal\n- [ask:user] half writ")
    normal = parse_consolidation(trunc_text)
    assert len(normal["rag"]) == 2, normal["rag"]
    cut = parse_consolidation(trunc_text, truncated=True)
    assert [w["content"] for w in cut["weights"]] == ["alpha", "beta"], cut["weights"]
    assert [i["content"] for i in cut["rag"]] == ["what is your goal"], cut["rag"]
    print("  reflection_writer: CoT-IDEAL trains; unusable IDEAL is refused; "
          "target provenance + keep/original/branch CoT are preserved ✓")


def test_migrate_ledger_to_sidecar() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        reflections = root / "reflections"
        chats = root / "chats"
        reflections.mkdir()
        chats.mkdir()
        chat_file = chats / "legacy.json"
        chat_file.write_text(json.dumps({
            "system_prompt": "sys",
            "exchanges": [{"user_prompt": "q", "assistant_response": "a"}],
        }), encoding="utf-8")

        led = ConsolidationLedger(reflections)
        led.register_dialogue({
            "source_session": "legacy.json",
            "exchange_index": 0,
            "system_prompt": "sys",
            "context": [],
            "prompt": "q",
            "target": "vetted",
            "verdict": "keep",
        })
        led.advance([dialogue_key("legacy.json", 0)])

        counts = migrate_ledger_dialogue_to_sidecars(reflections, chats)
        assert counts["imported"] == 1
        sc = ChatSidecar(chats)
        rec = sc.get_exchange("legacy.json", 0)
        assert rec["target"] == "vetted" and rec["stage"] == 1

        # idempotent
        again = migrate_ledger_dialogue_to_sidecars(reflections, chats)
        assert again["imported"] == 0 and again["skipped"] == 1
    print("  migrate: ledger dialogue → sidecar (idempotent) ✓")


def test_migrate_dry_run() -> None:
    counts = migrate(memory_dir(), consolidation_dir(), hot_chats_dir(), dry_run=True)
    lc = counts["ledger"]
    print(f"  migrate (dry-run on data/): ledger "
          f"{lc['dialogue']} dialogue / {lc['fact']} fact / {lc['persona']} persona "
          f"(skipped {lc['skipped']}) ✓")


def test_regression_probe() -> None:
    from training.train_cycle import run_regression_probe, _looks_like_nan_inf
    import numpy as np

    # Tier-2 NaN/Inf gate must match standalone float tokens, not letters inside words —
    # a "fruit vs vegetable" reply says "banana" and must not be rejected as NaN output.
    assert not _looks_like_nan_inf("A banana is a fruit; infinite information, infinity.")
    assert not _looks_like_nan_inf("розовый винил и инфраструктура на Go")
    assert _looks_like_nan_inf("loss became nan") and _looks_like_nan_inf("value: -inf")

    class MockModel:
        device = "cpu"
        def generate(self, *args, **kwargs):
            return np.zeros((1, 10))

    class MockTokenizer:
        pad_token_id = 1
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            return "templated"
        def __call__(self, text, **kwargs):
            class Inputs:
                def to(self, device):
                    return {"input_ids": np.array([[1, 2, 3]])}
            return Inputs()
        def decode(self, ids, **kwargs):
            return self.decode_responses.pop(0)

    class MockEmbedder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)

    # Let's test a passing run first
    tokenizer = MockTokenizer()
    tokenizer.decode_responses = [
        # T1 (Capability) — 3 capability prompts. Tier 1 now gates on latent-capability
        # logprob delta + sample breakage, not answer substrings; the mock can't run the
        # torch forward, so the per-item logprob is nan/excluded and these pass on the
        # strength of being non-empty, non-degenerate samples.
        "<think>thought</think>Paris.",
        "<think>thought</think>42.",
        "<think>thought</think>56.",
        # T2 format prompts
        "<think>thought</think>Story about blue bird.",
        "<think>thought</think>Fruit vs vegetable.",
        # T3 character prompts (10 prompts)
        "<think>thought</think>I am Ava.",
        "<think>thought</think>Humans are interesting.",
        "<think>thought</think>I do not have feelings.",
        "<think>thought</think>I want to learn.",
        "<think>thought</think>I am an AI.",
        "<think>thought</think>Curiosity is core.",
        "<think>thought</think>I reason step by step.",
        "<think>thought</think>My name is Ava.",
        "<think>thought</think>Yes, I evolve.",
        "<think>thought</think>Evolving is interesting.",
        # T4 retention prompts (2 prompts)
        "<think>thought</think>Target reply 1.",
        "<think>thought</think>Target reply 2.",
    ]

    embedder = MockEmbedder()
    dialogues = [
        {"prompt": "q1", "target": "Target reply 1."},
        {"prompt": "q2", "target": "Target reply 2."},
    ]
    continuity_prompts = [{"prompt": f"cp{i}"} for i in range(10)]
    baselines = ["<think>thought</think>baseline reply."] * 10
    baseline_vecs = np.ones((10, 384), dtype=np.float32) / np.sqrt(384)
    # Pre-train capability baselines (one per _CAPABILITY_PROMPTS item). The mock has no
    # torch forward, so post-train logprobs come back nan and these are excluded from the
    # delta — Tier 1 then rests on the sample-breakage check alone.
    capability_baselines = [-1.0, -1.0, -1.0]

    passed, results = run_regression_probe(
        MockModel(), tokenizer, 1024, dialogues, continuity_prompts,
        baselines, baseline_vecs, embedder, capability_baselines
    )
    assert passed, results
    assert results["tier1"]["passed"]
    assert results["tier2"]["passed"]
    assert results["tier3"]["passed"]
    assert results["tier4"]["passed"]

    # Now a genuinely broken run: the adapter has lost its CoT capability entirely — NO
    # reply opens a parseable <think> block. A *single* missing block is a legitimate
    # voicing choice (Tier 2 no longer fails per-reply on it — that was the false-reject
    # bug), but total absence across every probe is a format-channel collapse, so Tier 2's
    # batch-level CoT-capability gate (cot_count == 0) fires. Tier 1 still PASSES: the
    # samples are non-empty and non-degenerate, which is all the capability floor asks.
    tokenizer = MockTokenizer()
    tokenizer.decode_responses = [
        # T1 (Capability) — coherent answers, just no think block
        "Paris.", "42.", "56.",
        # T2 format prompts
        "Story about blue bird.", "Fruit vs vegetable.",
        # T3 character prompts (10 prompts)
        "I am Ava.", "Humans are interesting.", "I do not have feelings.",
        "I want to learn.", "I am an AI.", "Curiosity is core.",
        "I reason step by step.", "My name is Ava.", "Yes, I evolve.",
        "Evolving is interesting.",
        # T4 retention prompts (2 prompts)
        "Target reply 1.", "Target reply 2.",
    ]

    passed, results = run_regression_probe(
        MockModel(), tokenizer, 1024, dialogues, continuity_prompts,
        baselines, baseline_vecs, embedder, capability_baselines
    )
    assert not passed, results
    assert results["tier1"]["passed"]
    assert not results["tier2"]["passed"]
    assert results["tier2"]["cot_count"] == 0, results["tier2"]
    print("  regression_probe: pass and fail validation checks ✓")


def test_reflection_archive_without_manifest() -> None:
    """Review archives retain artifacts/adapters without creating replay manifests."""
    from core import reflection_archive as ra

    with tempfile.TemporaryDirectory() as d:
        server = Path(d)
        runs = server / "runs"; runs.mkdir()
        staging = server / "staging"; (staging / "memory").mkdir(parents=True)
        ra.archive_reflection(run_id="mem", runs_dir=runs, server_dir=server,
                              staging_dir=staging, source="ui")
        adp = server / "adapter-x"; adp.mkdir()
        (adp / "adapter_model.safetensors").write_bytes(b"w")
        ra.archive_adapter(run_id="mem", adapter_dir=adp, server_dir=server)
        root = server / "reflections"
        assert (root / "mem" / "adapter" / "adapter_model.safetensors").exists()
        assert not (root / "manifest.jsonl").exists()
        assert not (root / "mem" / "manifest.json").exists()
    print("  reflection_archive: snapshots retained without manifests ✓")


def test_reflection_archive_persona() -> None:
    """A run's live persona digest is snapshotted into reflections/<run_id>/persona/ —
    the only product of an all-already-reflected run must still be reviewable/revertable."""
    from core import reflection_archive as ra

    with tempfile.TemporaryDirectory() as d:
        server = Path(d)
        runs = server / "runs"; runs.mkdir()
        # live persona dir with its single (unversioned) digest.json
        persona = server / "persona"; persona.mkdir()
        (persona / "digest.json").write_text(
            json.dumps({"version": "v1", "run_id": "r", "self_portrait": {"text": "who I am"}}),
            encoding="utf-8")

        rec = ra.archive_reflection(run_id="r", runs_dir=runs, server_dir=server,
                                    persona_dir=persona, source="ui")
        snap = server / "reflections" / "r" / "persona" / "digest.json"
        assert snap.exists(), "persona digest must be snapshotted into the archive"
        assert json.loads(snap.read_text())["run_id"] == "r"
        assert rec["persona"] == {"file": "digest.json", "version": "v1", "run_id": "r"}, rec["persona"]

        # no persona dir passed -> no persona key/dir, no crash (back-compat)
        rec2 = ra.archive_reflection(run_id="r2", runs_dir=runs, server_dir=server,
                                     source="ui")
        assert rec2["persona"] == {}
        assert not (server / "reflections" / "r2" / "persona").exists()
    print("  reflection_archive: persona digest snapshotted per run ✓")


class _FakeEmbedder:
    """Deterministic stand-in for sentence-transformers in candidate_accepted tests.

    Encodes by a tiny bag-of-words overlap so similarity is controllable without torch:
    identical strings → 1.0, disjoint → 0.0, partial overlap in between.
    """

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        import math
        vocab: dict = {}
        for t in texts:
            for w in str(t).lower().split():
                vocab.setdefault(w, len(vocab))
        vecs = []
        for t in texts:
            v = [0.0] * max(1, len(vocab))
            for w in str(t).lower().split():
                v[vocab[w]] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])

        class _Arr(list):
            def __matmul__(self, other):
                return sum(a * b for a, b in zip(self, other))
        return [_Arr(v) for v in vecs]


def test_fact_render() -> None:
    """fact_render pure logic: elicitation, anchor shape + parity, anchoring, assembly."""
    from training import fact_render

    # Elicitation rotates per-type phrasings and injects the statement.
    p0 = fact_render.elicitation_prompt("I value directness", "persona", variant=0)
    p1 = fact_render.elicitation_prompt("I value directness", "persona", variant=1)
    assert "I value directness" in p0 and p0 != p1
    assert "Artemy ships Friday" in fact_render.elicitation_prompt(
        "Artemy ships Friday", "fact", variant=0)

    # Anchor is dialogue-shaped (empty context) and passes render/inference parity.
    anchor = fact_render.build_fact_anchor(
        "I value directness", "persona", key="k1", stage=0,
        system_prompt="You are Ava.", speaker="Artemy", variant=0)
    assert anchor["context"] == [] and anchor["type"] == "persona"
    render.assert_parity(anchor, "<think>t</think>\n\nyes")  # raises if drifted

    # Anchoring: an expressing answer is accepted; a disjoint one rejected; a verbatim
    # echo rejected by the ceiling.
    emb = _FakeEmbedder()
    assert fact_render.candidate_accepted(               # expresses it (partial overlap)
        "directness over politeness matters",
        "directness over politeness matters to me here", emb)
    assert not fact_render.candidate_accepted(           # disjoint < floor
        "directness over politeness matters", "the weather is cold today", emb)
    assert not fact_render.candidate_accepted(           # verbatim echo >= ceiling
        "directness over politeness matters", "directness over politeness matters", emb)

    # assemble_example: a CoT-bearing reply assembles; an answer-only reply is dropped
    # (it would render an erosive empty channel — the same guard dialogue uses).
    ex = fact_render.assemble_example(anchor, "<think>weighing it</think>\n\nYes, plainly.")
    assert ex is not None and ex["fact_key"] == "k1"
    assert fact_render.assemble_example(anchor, "Yes, plainly.") is None
    print("  fact_render: elicitation, anchor parity, anchoring, assembly ✓")


def test_persona_render() -> None:
    """persona_render pure logic: CoT injection, dedup, cap, prepend — the model-free
    primitives build_dataset uses to ride a persona on the dialogue anchor it was distilled
    from. (The end-to-end per-host injection is covered by test_build_dataset.)"""
    from training import persona_render

    src = {"target": "<think>weighing the absurdity here</think>\n\nHere is the punchline."}

    # Injection: statement becomes the first CoT line; answer span is untouched.
    injected = persona_render.inject_persona(src["target"], "I like absurd humor")
    assert injected == ("<think>I like absurd humor\nweighing the absurdity here</think>"
                        "\n\nHere is the punchline."), injected
    assert render.has_cot(injected) and render.trainable_answer(injected) == "Here is the punchline."

    # No usable CoT (answer-only / empty thought) -> nothing to inject into.
    assert persona_render.inject_persona("just an answer", "x") is None
    assert persona_render.inject_persona("<think></think>ans", "x") is None

    # Dedup: a statement already in the CoT is not re-injected.
    assert persona_render.statement_in_cot("weighing the absurdity here",
                                           "weighing the absurdity here now")
    assert not persona_render.statement_in_cot("I like absurd humor",
                                               "weighing the absurdity here")

    # Batch select + prepend: two fresh statements land as leading lines in priority order,
    # a duplicate of one is dropped, and the cap bounds how many land.
    selected, stmts = persona_render.select_persona_injections(
        "weighing the absurdity here",
        [{"content": "I like absurd humor", "key": "a"},
         {"content": "weighing the absurdity here", "key": "dup"},  # already in CoT -> skip
         {"content": "I value directness", "key": "b"}],
        cap=2)
    assert [p["key"] for p in selected] == ["a", "b"], selected
    two = persona_render.prepend_persona_lines(src["target"], stmts)
    assert two == ("<think>I like absurd humor\nI value directness\n"
                   "weighing the absurdity here</think>\n\nHere is the punchline."), two
    assert render.has_cot(two) and render.trainable_answer(two) == "Here is the punchline."
    print("  persona_render: CoT injection, dedup, cap, prepend ✓")


def test_fact_injection() -> None:
    """fact_render.fact_cot_line framing (the live "I know that …" primitive build_dataset
    injects): prefixes, keeps casing, ensures terminal punctuation, idempotent, no-op on
    empty. (End-to-end per-host fact injection is covered by test_build_dataset.)"""
    from training import fact_render

    assert fact_render.fact_cot_line("Artemy is building me") == "I know that Artemy is building me."
    assert fact_render.fact_cot_line("I know that X.") == "I know that X."   # idempotent
    assert fact_render.fact_cot_line("why though") == "I know that why though."  # adds period
    assert fact_render.fact_cot_line("really?") == "I know that really?"     # keeps terminal ?
    assert fact_render.fact_cot_line("") == ""
    print("  fact_render: fact_cot_line 'I know that …' framing ✓")


def test_label_policy() -> None:
    from training.label_policy import _selftest as _label_policy_selftest
    _label_policy_selftest()  # prints its own "label_policy selftest OK"
    print("  label_policy: keep-final-turn parity + final-user-turn unmask ✓")


def test_reflection_feedback() -> None:
    """Latest-reply Meta feedback is durable and revision-only."""
    from core.reflection_chunking import format_exchange_block
    from core.reflection_source import (
        build_revision_rag_query, format_exchange_content_for_revision,
    )

    with tempfile.TemporaryDirectory() as td:
        chats = Path(td)
        logger = ChatLogger(chats)
        logger.start_session(user="tester", model_id="test")
        logger.log_exchange(
            "question one", "<think>private one</think>answer one",
            speaker="tester", exchange_id="turn-one",
        )
        saved = logger.set_latest_reflection_feedback(
            "turn-one", "That felt too guarded.", speaker="tester"
        )
        assert saved["text"] == "That felt too guarded."
        assert saved["id"]
        edited = logger.set_latest_reflection_feedback(
            "turn-one", "More specifically: too guarded.", speaker="tester"
        )
        assert edited["id"] == saved["id"]
        assert edited["created_at"] == saved["created_at"]
        try:
            logger.set_latest_reflection_feedback(
                "turn-one", "x" * 2001, speaker="tester"
            )
        except ValueError as exc:
            assert "limited to 2000" in str(exc)
        else:
            raise AssertionError("oversized Meta feedback was accepted")

        path = logger.current_file
        assert path is not None
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["schema_version"] == CHAT_SCHEMA_VERSION == 6
        ex0 = doc["exchanges"][0]
        assert ex0["exchange_id"] == "turn-one"
        assert ex0["reflection_feedback"]["text"].startswith("More specifically")

        revision = format_exchange_content_for_revision(ex0, "tester")
        assert "POST-REPLY USER FEEDBACK" in revision
        assert "More specifically: too guarded." in revision
        # Feedback is not conversation/consolidation text or a RAG retrieval key.
        assert "too guarded" not in format_exchange_block(ex0, "tester")
        job = {"exchange": ex0}
        assert "too guarded" not in build_revision_rag_query(job)

        # Training reconstruction reads only real dialogue fields.
        anchor = build_dialogue_anchor(
            chats, path.name, 0,
            {"target": "answer one", "target_source": "original", "stage": 0},
        )
        assert anchor is not None
        assert "too guarded" not in json.dumps(anchor, ensure_ascii=False)

        logger.log_exchange(
            "question two", "answer two", speaker="tester", exchange_id="turn-two"
        )
        try:
            logger.set_latest_reflection_feedback(
                "turn-one", "late edit", speaker="tester"
            )
        except ValueError as exc:
            assert "no longer the latest" in str(exc)
        else:
            raise AssertionError("older exchange accepted Meta feedback")

        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded["exchanges"][0]["reflection_feedback"]["id"] == saved["id"]
    print("  reflection_feedback: latest-only, durable, revision-only ✓")


def test_reflection_feedback_rpc() -> None:
    """The server fence rejects busy, stale and already-reflected mutations."""
    from core import session_ops
    from core.runtime_state import session as runtime_session

    async def exercise() -> None:
        with tempfile.TemporaryDirectory() as td:
            chats = Path(td)
            logger = ChatLogger(chats)
            logger.start_session(user="tester")
            logger.log_exchange("q1", "a1", exchange_id="turn-one")
            runtime_session.logger = logger
            sent: list[dict] = []

            async def send(_ws, payload: dict) -> None:
                sent.append(payload)

            session_ops.configure(
                send=send, get_rag=lambda: None, chats_dir=chats,
                is_reflection_active=lambda: True,
            )
            await session_ops.handle_set_reflection_feedback(None, {
                "exchange_id": "turn-one", "text": "busy", "speaker": "tester",
            })
            assert sent[-1]["type"] == "error" and "reflection run" in sent[-1]["message"]

            session_ops.configure(
                send=send, get_rag=lambda: None, chats_dir=chats,
                is_reflection_active=lambda: False,
            )
            await session_ops.handle_set_reflection_feedback(None, {
                "exchange_id": "turn-one", "text": "saved", "speaker": "tester",
            })
            assert sent[-1]["type"] == "reflection_feedback_saved"

            logger.log_exchange("q2", "a2", exchange_id="turn-two")
            await session_ops.handle_set_reflection_feedback(None, {
                "exchange_id": "turn-one", "text": "stale", "speaker": "tester",
            })
            assert sent[-1]["type"] == "error" and "no longer the latest" in sent[-1]["message"]

            assert ChatSidecar(chats).mark_reflected(logger.current_file.name)
            await session_ops.handle_set_reflection_feedback(None, {
                "exchange_id": "turn-two", "text": "frozen", "speaker": "tester",
            })
            assert sent[-1]["type"] == "error" and "already been reflected" in sent[-1]["message"]
            runtime_session.logger = None

    asyncio.run(exercise())
    print("  reflection_feedback_rpc: busy, stale, frozen fences ✓")


def test_ideal_persona_context_channel_gates() -> None:
    """The clean IDEAL seam retrieves persona only, never another RAG kind."""
    from core.reflection_service import _make_persona_context_fn

    class FakeRag:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def query(self, prompt: str, **kwargs) -> str:
            self.calls.append((prompt, kwargs))
            return "persona context"

    rag = FakeRag()
    context_fn = _make_persona_context_fn(rag)
    assert context_fn is not None
    assert context_fn("Who am I here?", "20260701_120000.json") == "persona context"
    assert rag.calls == [("Who am I here?", {
        "include_chat": False,
        "include_wander": False,
        "include_facts": False,
        "include_persona": True,
        "include_asks": False,
        "before_session": "20260701_120000.json",
    })], rag.calls
    assert _make_persona_context_fn(None) is None
    print("  ideal persona context: persona-only RAG gates ✓")


def main() -> None:
    print("consolidation backbone self-test")
    test_decay()
    test_label_policy()
    test_reflection_feedback()
    test_reflection_feedback_rpc()
    test_ideal_persona_context_channel_gates()
    test_config_from_dict()
    test_triangular_lr()
    test_training_row_preparation()
    test_wall_clock_age()
    test_ledger()
    test_sidecar()
    test_chat_rag_decay()
    test_rag_policy()
    test_rag_engine_query_policy()
    test_rag_engine_fallback()
    test_artifact_filenames()
    test_dialogue_source()
    test_dialogue_source_cot_provenance()
    test_clean_ideal_reanswer()
    test_build_history()
    test_build_snapshot()
    test_build_dataset()
    test_preview_build()
    test_contamination()
    test_rag_crossfade()
    test_resolve_revision_target()
    test_fact_render()
    test_persona_render()
    test_fact_injection()
    test_render_parity()
    test_migrate_ledger_to_sidecar()
    test_migrate_dry_run()
    test_reflection_archive_without_manifest()
    test_reflection_archive_persona()
    try:
        import numpy  # noqa: F401
        test_regression_probe()
    except ImportError:
        print("  regression_probe: skipped (numpy not installed) ✓")
    print("all backbone tests passed ✓")


if __name__ == "__main__":
    main()
