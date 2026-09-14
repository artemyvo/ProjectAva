"""OpenAI-compatible counterpart client for the Encounter feature.

Ava (the local GPU model) meets a "fellow AI" — a non-subjective helpful
assistant served behind an OpenAI-compatible ``/v1/chat/completions`` endpoint
(e.g. a vLLM box at ``http://spark:8000``). This module is the thin caller for
that endpoint plus a small holder for the counterpart's side of the dialogue.

It owns no GPU and no Ava state: the encounter loop in ``server.py`` drives the
turn-taking, calling :meth:`CounterpartClient.reply` with each of Ava's messages
and feeding the returned text back to Ava as a labeled ``user`` turn. The mirror
conversation (Ava-as-user / counterpart-as-assistant) lives here so the endpoint
sees a coherent dialogue from its own point of view.

Stdlib only (``urllib``) — the server already depends on nothing for HTTP, and
the client tier stays ML-free, so we avoid pulling the ``openai`` package in.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional, Tuple


# ── Peer-reasoning boundary sanitizer ────────────────────────────────────────
#
# The peer's chain of thought is display-only: an operator watches it in the
# Encounter/Gossip log, but Ava must never see it. It is not hers, she has no way
# to tell it apart from something the peer *said*, and every consumer downstream
# of a counterpart reply treats that text as the peer's spoken turn — it becomes
# the ``user_prompt`` of a logged exchange, the RAG query for Ava's next turn, a
# chat-RAG passage, and eventually a trained row.
#
# The well-behaved path already keeps the two apart: a peer Ava's gossip endpoint
# returns the answer on ``message.content`` and the reasoning on the separate
# ``message.reasoning_content``. But ``content`` is not trustworthy on its own:
#
#   * a peer whose generation is cut mid-thought sends the raw, unterminated
#     thought AS the content — its own ``ChatLogger._parse_cot`` reads a
#     ``<think>`` with no close as "no CoT, all answer". Not a corner case: a
#     reasoning peer thinks for minutes against a 4096-token default cap.
#   * a counterpart that is not an Ava at all — a plain vLLM box serving a
#     reasoning model with no reasoning parser configured — inlines
#     ``<think>…</think>`` in ``content`` as a matter of course.
#
# So the split is re-established HERE, at the boundary where foreign text enters,
# rather than trusted to whatever produced it. Deliberately family-agnostic: the
# peer's family is unknown (its model name is operator-typed free text), so every
# marker shape ``core.model_family`` knows about is recognized rather than one
# profile's. Errs toward withholding — an unrecognised-but-reasoning-shaped span
# kept from Ava costs a legible failed turn; the same span let through is a peer's
# thinking silently entering her memory.
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Any gemma channel, not just ``thought``: withholding a non-thought channel is
# the safe direction, and the peer's profile is unknown here.
_GEMMA_BLOCK_RE = re.compile(r"<\|channel>[^\n]*\n?(.*?)<channel\|>", re.DOTALL)
_HARMONY_RE = re.compile(r"(?s)\Aanalysis(.*?)(?:assistant)?final(.*)\Z")
_REASONING_CLOSERS = ("</think>", "<channel|>")
_REASONING_OPENERS = ("<think>", "<|channel>")


def strip_peer_reasoning(text: str) -> Tuple[str, str]:
    """Split a counterpart reply into ``(reasoning, answer)``.

    Returns the reasoning spans found (joined, for display) and the text with every
    one of them removed. A reply that is *entirely* an unterminated thought yields
    an empty answer, which :meth:`CounterpartClient.reply` reports as a failed turn —
    an honest error beats feeding Ava the peer's raw thinking.
    """
    if not text:
        return "", text or ""
    found: list[str] = []

    # gpt-oss harmony ("analysis…assistantfinal…"), matched only when the reply
    # *begins* with the channel name — mirroring model_family._normalize_gptoss, so
    # a reply that merely discusses analysis stays prose.
    harmony = _HARMONY_RE.match(text)
    if harmony:
        found.append(harmony.group(1).strip())
        text = harmony.group(2)

    def _collect(match) -> str:
        found.append(match.group(1).strip())
        return ""

    text = _THINK_BLOCK_RE.sub(_collect, text)
    text = _GEMMA_BLOCK_RE.sub(_collect, text)

    # A close marker with no opener before it: the peer's channel opener was
    # prefilled into its prompt, so the trace it streamed back carries only the
    # close (see model_family._normalize_gemma / _normalize_think_tags).
    for closer in _REASONING_CLOSERS:
        idx = text.find(closer)
        if idx != -1:
            found.append(text[:idx].strip())
            text = text[idx + len(closer):]

    # An opener with no close: the generation was cut mid-thought. Everything from
    # the opener on is thinking, and there is no answer after it.
    for opener in _REASONING_OPENERS:
        idx = text.find(opener)
        if idx != -1:
            found.append(text[idx + len(opener):].strip())
            text = text[:idx]

    return "\n\n".join(p for p in found if p), text.strip()


def normalize_endpoint(url: str) -> str:
    """Resolve a user-supplied base URL to the chat-completions endpoint.

    Accepts the forgiving forms an operator is likely to type:
      * ``http://spark:8000``                 → ``…/v1/chat/completions``
      * ``http://spark:8000/v1``              → ``…/v1/chat/completions``
      * ``http://spark:8000/v1/chat/completions`` → unchanged
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("counterpart endpoint URL is empty")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


class CounterpartError(RuntimeError):
    """Raised when the counterpart endpoint is unreachable or returns an error."""


class CounterpartClient:
    """Holds the fellow AI's side of the conversation and calls its endpoint.

    *url* is the operator-supplied base (or full) endpoint; *model* the model
    name the endpoint expects; *system_prompt* an optional system message for the
    counterpart (left blank to use the endpoint's own default persona).
    """

    def __init__(
        self,
        url: str,
        model: str,
        *,
        system_prompt: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        api_key: str = "",
    ) -> None:
        self.endpoint = normalize_endpoint(url)
        self.model = (model or "").strip()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = (api_key or "").strip()
        # ``finish_reason`` of the most recent reply ("stop" | "length" | …). A
        # "length" finish means the endpoint hit *our* max_tokens cap mid-reply —
        # surfaced to the UI so a truncation reads as our limit, not the model
        # refusing or hitting a content wall.
        self.last_finish_reason: Optional[str] = None
        # Reasoning trace of the most recent reply: ``message.reasoning_content`` when the
        # endpoint exposes one (a peer Ava's gossip endpoint does; a plain vLLM box usually
        # does not), plus anything ``strip_peer_reasoning`` pulled back out of ``content``.
        # Display-only — it is NEVER appended to the mirror conversation or returned from
        # :meth:`reply`, so a peer's CoT can't reach Ava or her transcript.
        self.last_reasoning: str = ""
        # Mirror conversation from the counterpart's POV: Ava's turns are `user`,
        # its own replies are `assistant`. The optional system message leads.
        self._messages: list[dict] = []
        if system_prompt.strip():
            self._messages.append({"role": "system", "content": system_prompt.strip()})

    def reply(self, ava_message: str) -> str:
        """Send Ava's latest message, return the counterpart's reply text.

        Appends both turns to the mirror conversation so multi-turn context is
        preserved. Raises :class:`CounterpartError` on any transport/parse error
        so the loop can surface a legible failure rather than hang.
        """
        self._messages.append({"role": "user", "content": ava_message})
        payload = {
            "model": self.model,
            "messages": self._messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise CounterpartError(
                f"counterpart endpoint returned HTTP {e.code}: {detail or e}"
            ) from e
        except urllib.error.URLError as e:
            raise CounterpartError(
                f"could not reach counterpart endpoint {self.endpoint}: {e.reason}"
            ) from e
        except Exception as e:  # JSON decode, timeout, …
            raise CounterpartError(
                f"counterpart request to {self.endpoint} failed: "
                f"{type(e).__name__}: {e}"
            ) from e

        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise CounterpartError(
                f"counterpart response had no choices: {json.dumps(body)[:300]}"
            )
        self.last_finish_reason = choices[0].get("finish_reason")
        reasoning = self._extract_reasoning(body)
        # Re-establish the reasoning/answer split at the boundary rather than trusting
        # the peer to have done it (see strip_peer_reasoning): a peer cut mid-thought
        # ships its raw thinking AS the content, and a non-Ava reasoning peer inlines
        # it by default. Both would otherwise reach Ava as the peer's spoken turn.
        inline_reasoning, text = strip_peer_reasoning(self._extract_text(body))
        if inline_reasoning:
            reasoning = f"{reasoning}\n\n{inline_reasoning}".strip() if reasoning else inline_reasoning
        # Set before any raise below, so the operator can still inspect what was withheld.
        self.last_reasoning = reasoning
        if not text.strip():
            if inline_reasoning:
                raise CounterpartError(
                    "counterpart replied with reasoning only and no answer — its thinking "
                    "was withheld; raise the peer's max_tokens if it was cut mid-thought"
                )
            raise CounterpartError("counterpart returned an empty reply")
        # The stripped text is what goes into the mirror conversation too, so the peer's
        # own CoT is not echoed back to it on the next turn either.
        self._messages.append({"role": "assistant", "content": text})
        return text

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Pull the assistant message text out of an OpenAI-style response."""
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise CounterpartError(
                f"counterpart response had no choices: {json.dumps(body)[:300]}"
            ) from e
        message = choice.get("message") or {}
        content = message.get("content")
        # Some servers stream the final in `text` (legacy completions) — accept it.
        if content is None:
            content = choice.get("text")
        if isinstance(content, list):
            # Some servers return content as a list of parts ({type,text}).
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return content or ""

    @staticmethod
    def _extract_reasoning(body: dict) -> str:
        """Pull the assistant reasoning trace out of an OpenAI-style response, if any.

        Reads ``message.reasoning_content`` (DeepSeek/vLLM/ our gossip endpoint). Returns
        "" when the endpoint doesn't expose reasoning — the common vLLM case."""
        try:
            message = body["choices"][0].get("message") or {}
        except (KeyError, IndexError, TypeError):
            return ""
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, list):
            reasoning = "".join(
                part.get("text", "") for part in reasoning if isinstance(part, dict)
            )
        return reasoning if isinstance(reasoning, str) else ""


# ── GPU-free self-test: python -m core.encounter ─────────────────────────────

if __name__ == "__main__":
    def _check(label, got, want):
        assert got == want, f"{label}: got {got!r}, want {want!r}"
        print(f"  ok  {label}")

    print("strip_peer_reasoning:")
    # The ordinary case: a peer that split correctly is left alone.
    _check("clean answer untouched",
           strip_peer_reasoning("Just an answer."), ("", "Just an answer."))
    _check("empty", strip_peer_reasoning(""), ("", ""))
    # Inlined CoT — a non-Ava reasoning peer with no reasoning parser.
    _check("closed <think> block",
           strip_peer_reasoning("<think>hmm</think>\nAnswer."), ("hmm", "Answer."))
    _check("gemma channel",
           strip_peer_reasoning("<|channel>thought\nhmm\n<channel|>Answer."),
           ("hmm", "Answer."))
    _check("harmony",
           strip_peer_reasoning("analysis hmm assistantfinal Answer."),
           ("hmm", "Answer."))
    # Prefilled opener: the trace comes back carrying only the close marker.
    _check("close with no opener",
           strip_peer_reasoning("hmm</think>Answer."), ("hmm", "Answer."))
    _check("gemma close with no opener",
           strip_peer_reasoning("hmm<channel|>Answer."), ("hmm", "Answer."))
    # The case this guard exists for: a peer cut mid-thought ships the raw thought
    # as its content, and must yield NO answer rather than an answer of thinking.
    _check("unterminated <think> is all reasoning",
           strip_peer_reasoning("<think>hmm, and then"), ("hmm, and then", ""))
    _check("unterminated gemma channel is all reasoning",
           strip_peer_reasoning("<|channel>thought\nhmm, and then"),
           ("thought\nhmm, and then", ""))
    _check("two blocks both collected",
           strip_peer_reasoning("<think>a</think>Mid.<think>b</think>Tail."),
           ("a\n\nb", "Mid.Tail."))
    # Prose that merely mentions the words must not be eaten.
    _check("prose mentioning analysis is not a channel",
           strip_peer_reasoning("My analysis is that the final answer is 4."),
           ("", "My analysis is that the final answer is 4."))

    print("normalize_endpoint:")
    for raw, want in (
        ("http://spark:8000", "http://spark:8000/v1/chat/completions"),
        ("http://spark:8000/v1", "http://spark:8000/v1/chat/completions"),
        ("http://spark:8000/v1/chat/completions", "http://spark:8000/v1/chat/completions"),
    ):
        _check(raw, normalize_endpoint(raw), want)

    print("\nAll encounter self-tests passed.")
