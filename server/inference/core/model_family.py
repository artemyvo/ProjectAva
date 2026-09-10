"""Model-family chat / chain-of-thought (CoT) profiles.

Different model families implement thinking differently, both in how the chat
template is primed and in how the raw generation encodes the reasoning trace.
Centralizing that knowledge here keeps the rest of the server family-agnostic:
callers ask a profile for the chat-template kwargs, normalize a raw generation to
the canonical ``<think>…</think>\\nanswer`` form, and read the thinking-close
markers — without sprinkling ``"gemma-4" in model_id`` checks everywhere.

Families
--------
- **gemma-4** — emits a native reasoning *channel*
  ``<|channel>thought\\n…\\n<channel|>``; thinking is turned on via the chat
  template's ``enable_thinking`` (a ``<|think|>`` system directive). The channel
  close marker is ``<channel|>``. Normalization rewrites the channel to
  ``<think>…</think>``.
- **qwen3** (Qwen3 / Qwen3.5 / Qwen3.6) — ChatML + native ``<think>…</think>``,
  but the chat template **pre-fills the opening ``<think>\\n`` into the prompt**.
  With ``skip_prompt=True`` streaming, the *generated* text therefore starts
  *inside* the think block and carries only the closing ``</think>`` — no opener.
  Normalization re-adds the opener so downstream ``<think>…</think>`` parsing is
  uniform. ``enable_thinking`` controls whether the template prefills ``<think>``
  (think) or ``<think>\\n\\n</think>`` (no-think). Close marker ``</think>``.
- **gpt-oss** — harmony channels surface as plain text
  ``analysis…assistantfinal…`` (no ``<think>`` tags); ``reasoning_effort`` drives
  depth. Normalization converts the analysis/final split to ``<think>…</think>``.
- **default** — any other model: assume a literal, self-contained
  ``<think>…</think>`` already in the generation; no template priming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple


# ── per-family CoT normalizers (raw generation → canonical <think>…</think>) ──

_GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought\n?(.*?)\n?<channel\|>", re.DOTALL)
_GEMMA_STRAY_RE = re.compile(r"<\|channel>[^\n]*\n?|<channel\|>")
_GPTOSS_RE = re.compile(r"(?s)analysis(.*?)(?:assistant)?final(.*)")


def _normalize_gemma(raw: str) -> str:
    """Rewrite gemma-4's ``<|channel>thought…<channel|>`` to ``<think>…</think>``.

    When the channel opener is *prefilled* into the prompt (``GEMMA4.think_prefill``,
    the mechanism-1 cure), generation begins *inside* the channel under skip_prompt
    streaming, so the raw text carries the closing ``<channel|>`` with no opener.
    Re-add the opener first so the channel regex below matches — the gemma analogue of
    qwen3's prefilled-``<think>`` re-add in ``_normalize_think_tags``. Idempotent: a
    self-generated channel (reflection path, base model) already has the opener and
    skips this branch.

    The re-add keys on whether the text *starts inside a channel* — the first channel
    marker is a close (``<channel|>``) with no opener before it — NOT on the mere
    absence of ``<|channel>`` anywhere. A revision reply that carries its own reasoning
    channel (e.g. an ``IDEAL:`` block whose reworked answer thinks first) puts a *later*
    ``<|channel>`` in the raw; the old ``not in raw`` guard then skipped the re-add, the
    regex bound to the IDEAL's channel, and the leading ``<channel|>`` was stripped to
    nothing — leaving no ``<think>`` before ``VERDICT`` so ``_split_think`` mistook the
    IDEAL's inner think for the meta-CoT and dropped the whole verdict body.
    """
    close = raw.find("<channel|>")
    opener = raw.find("<|channel>")
    if close != -1 and (opener == -1 or close < opener):
        raw = "<|channel>thought\n" + raw
    if "<|channel>" in raw:
        raw = _GEMMA_CHANNEL_RE.sub(r"<think>\1</think>", raw)
        # Strip any remaining channel markers (non-thought or stray).
        raw = _GEMMA_STRAY_RE.sub("", raw)
    return raw


def _normalize_gptoss(raw: str) -> str:
    """Convert gpt-oss harmony ``analysis…(assistant)final…`` to ``<think>…</think>``.

    The ``<|…|>`` role/channel tokens are stripped per-chunk upstream, leaving the
    bare role NAMES ("analysis"/"final") as plain text; "analysis…assistantfinal…".
    """
    if raw.startswith("analysis") and "final" in raw:
        m = _GPTOSS_RE.match(raw)
        if m:
            think, answer = m.group(1).strip(), m.group(2).strip()
            return f"<think>{think}</think>\n{answer}" if think else answer
    return raw


def _normalize_think_tags(raw: str) -> str:
    """Qwen / default: re-add the opening ``<think>`` the template prefilled.

    Qwen3.x templates prime the prompt with ``<think>\\n``, so the generation only
    contains the closing ``</think>``. Prepend the opener when a close exists with
    no opener. Idempotent: a self-contained ``<think>…</think>`` is left untouched.
    """
    if "</think>" in raw and "<think>" not in raw:
        return "<think>" + raw
    return raw


# ── family profile ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelFamily:
    """The family-specific knobs for priming and parsing chain-of-thought."""

    name: str
    # kwargs spread into build_inference_prompt / apply_chat_template.
    template_kwargs: Dict[str, object] = field(default_factory=dict)
    # text markers that close a thinking block (for tension boundary detection).
    close_markers: Tuple[str, ...] = ("</think>",)
    # special-token strings that *open* a thinking block, in resolution-preference
    # order. The first generated token's probability mass on this token is the
    # "Thinking: NN%" diagnostic — how strongly the model wanted to reason before
    # answering. Empty / unresolvable → no diagnostic (e.g. qwen3, where the opener
    # is prefilled into the prompt and so is never sampled at generation time).
    think_open_markers: Tuple[str, ...] = ()
    # text appended to the *prompt* after the generation prompt to force the reasoning
    # channel / think block open every turn. This is the mechanism-1 cure for multi-turn
    # CoT collapse: with CoT stripped from history, prior assistant turns show the model
    # only no-think replies and the *sampled* opener probability decays toward zero, so
    # turn 2+ answers without thinking. Prefilling the opener removes it from sampling
    # entirely (the model continues from inside the channel). qwen3 gets this for free
    # from its chat template's ``enable_thinking``; gemma-4 must prefill by hand. Empty =
    # nothing prefilled (qwen3/gpt-oss/default).
    think_prefill: str = ""
    # minimum number of generated tokens before the reasoning-close marker (``close_markers``)
    # is allowed — the companion to ``think_prefill``. Prefilling the opener guarantees the
    # channel *opens*, but the model can still emit the close token immediately (an empty
    # ``<|channel>thought\n<channel|>`` — "thinking on, no thought"), which is what the
    # answer-only training scaffold biases it toward. A logits floor forbidding the close
    # for this many steps guarantees a non-empty CoT. 0 = no floor (qwen3 already thinks
    # reliably; gpt-oss/default unaffected).
    min_think_tokens: int = 0
    _normalize: Callable[[str], str] = _normalize_think_tags
    # Recommended sampling for this family (the model card / Unsloth defaults). The
    # backend applies ``rec_top_k`` automatically when a caller doesn't override it
    # (top-k isn't exposed in the protocol); ``rec_temperature`` / ``rec_top_p`` seed
    # the chat-UI + reflection runtime defaults. 0 / leaving HF default == no top-k
    # truncation.
    rec_temperature: float = 1.0
    rec_top_p: float = 1.0
    rec_top_k: int = 0

    def normalize_cot(self, raw: str) -> str:
        """Rewrite a raw generation to the canonical ``<think>…</think>`` form."""
        return self._normalize(raw)


GEMMA4 = ModelFamily(
    name="gemma-4",
    template_kwargs={"enable_thinking": True},
    close_markers=("<channel|>", "</think>"),
    # The channel opener is prefilled into the prompt (think_prefill below), so it is no
    # longer sampled at generation time — the "Thinking: NN%" diagnostic is moot (CoT is
    # now structural) and is omitted, like qwen3 whose opener is likewise prefilled. The
    # decision we used to diagnose (open the channel vs answer directly) is the thing the
    # prefill now forces; see git history / AVA_DESIGN_LEGACY.md for the diagnostic's role.
    think_open_markers=(),
    # gemma-4 must prefill its channel opener to keep CoT alive across multi-turn chats
    # (see ModelFamily.think_prefill). Matches the trained span emitted by
    # training.render.to_gemma_thinking_channel, so train/inference stay in parity.
    think_prefill="<|channel>thought\n",
    # Forbid the channel close (`<channel|>`) for the first 16 generated tokens so a
    # prefilled-open channel can't be closed empty (see ModelFamily.min_think_tokens).
    min_think_tokens=16,
    _normalize=_normalize_gemma,
    # Gemma 4 (per the unsloth/gemma-4-31B card + Gemma 3 lineage): T=1.0, top_p=0.95,
    # top_k=64. The project shipped with T=0.7 / top_p=0.9 and no top-k — too greedy
    # for this family, which is tuned to sample over a wider, top-k-bounded set.
    rec_temperature=1.0,
    rec_top_p=0.95,
    rec_top_k=64,
)

GPTOSS = ModelFamily(
    name="gpt-oss",
    template_kwargs={"reasoning_effort": "high"},
    close_markers=("</think>",),
    _normalize=_normalize_gptoss,
    rec_temperature=1.0,
    rec_top_p=1.0,
    rec_top_k=0,
)

QWEN3 = ModelFamily(
    name="qwen3",
    template_kwargs={"enable_thinking": True},
    close_markers=("</think>",),
    _normalize=_normalize_think_tags,
    # Qwen3 thinking-mode recommendation: T=0.6, top_p=0.95, top_k=20.
    rec_temperature=0.6,
    rec_top_p=0.95,
    rec_top_k=20,
)

# A self-contained `<think>…</think>` is generated literally, so the opener is sampled.
DEFAULT = ModelFamily(name="default", think_open_markers=("<think>",))


def family_for(model_id: str) -> ModelFamily:
    """Resolve the chat/CoT profile for a model id (matches the loaded model)."""
    mid = (model_id or "").lower()
    if "gemma-4" in mid:
        return GEMMA4
    if "gpt-oss" in mid:
        return GPTOSS
    if "qwen3" in mid:  # qwen3, qwen3.5, qwen3.6, qwen3_5, …
        return QWEN3
    return DEFAULT


# ── GPU-free self-test ───────────────────────────────────────────────────────

def _selftest() -> None:
    """Validate per-family CoT normalization against representative raw output."""
    # gemma-4: channel tokens → <think>…</think>
    g = family_for("unsloth/gemma-4-31B-it")
    assert g is GEMMA4
    out = g.normalize_cot("<|channel>thought\nweigh options\n<channel|>The answer.")
    assert out == "<think>weigh options</think>The answer.", out
    # recommended sampling carried on the family (applied by the backend / defaults)
    assert (g.rec_temperature, g.rec_top_p, g.rec_top_k) == (1.0, 0.95, 64), (
        g.rec_temperature, g.rec_top_p, g.rec_top_k)
    assert family_for("mistralai/Mistral-7B").rec_top_k == 0  # default: no top-k
    # gemma-4 prefills its channel opener (mechanism-1 cure), so — like qwen3 — the
    # opener is never sampled and there is no think-open diagnostic marker.
    assert g.think_prefill == "<|channel>thought\n", g.think_prefill
    assert g.think_open_markers == (), g.think_open_markers
    assert family_for("unsloth/Qwen3-4B").think_open_markers == ()
    assert family_for("unsloth/Qwen3-4B").think_prefill == ""
    # prefilled gemma generation: closing channel with no opener → opener re-added,
    # then rewritten to <think>; empty channel (immediate close) round-trips too.
    assert g.normalize_cot("reason\n<channel|>ans") == "<think>reason</think>ans"
    assert g.normalize_cot("<channel|>ans") == "<think></think>ans"
    # A reply that carries its OWN channel (e.g. a revision IDEAL whose reworked answer
    # thinks first) puts a *later* <|channel> in the raw. The leading empty channel must
    # still normalize to <think></think> — keyed on the text starting inside a channel,
    # not on the mere absence of <|channel> — so <think> precedes the structured body and
    # _split_think can't mistake the IDEAL's inner think for the meta-CoT.
    two_channel = (
        "<channel|>VERDICT: revise\nIDEAL:\n"
        "<|channel>thought\nrework tone\n<channel|>the answer"
    )
    assert g.normalize_cot(two_channel) == (
        "<think></think>VERDICT: revise\nIDEAL:\n<think>rework tone</think>the answer"
    ), g.normalize_cot(two_channel)

    # qwen3.6: prefilled opener missing in generation → re-added
    q = family_for("unsloth/Qwen3.6-27B")
    assert q is QWEN3
    assert q.template_kwargs == {"enable_thinking": True}
    out = q.normalize_cot("weigh options</think>\n\nThe answer.")
    assert out == "<think>weigh options</think>\n\nThe answer.", out
    # idempotent on a self-contained block
    full = "<think>weigh</think>\n\nThe answer."
    assert q.normalize_cot(full) == full

    # gpt-oss: analysis/final harmony → <think>…</think>
    o = family_for("openai/gpt-oss-20b")
    assert o is GPTOSS
    out = o.normalize_cot("analysisweigh optionsassistantfinalThe answer.")
    assert out == "<think>weigh options</think>\nThe answer.", out

    # default: unknown model, self-contained think block untouched, no priming
    d = family_for("mistralai/Mistral-7B")
    assert d is DEFAULT and d.template_kwargs == {}
    assert d.normalize_cot("plain answer, no think") == "plain answer, no think"

    print("model_family self-test OK")


if __name__ == "__main__":
    _selftest()
