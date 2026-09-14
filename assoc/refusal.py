"""Is an EMPTY answer a refusal? (shared by the relation pass and the LLM witness)

Both passes run thinking-off under a bare labelling / witnessing prompt, and on a transcript
about sex or violence the base model's safety reflex answers in the first tokens instead of
the lines ("I cannot fulfill this request. I am programmed to be a helpful and harmless AI
assistant. My safety guidelines prohibit …" — the training box, 2026-09-11 23:46 UTC). Parsed, that
is zero lines, and zero lines is ALSO what a genuine "nothing here" looks like — so each
caller used to file the refusal as a verdict: the relation pass cached "no relation" for
the whole batch, the witness wrote a zero-fact group over the imported protocol.

The classifier is vocabulary, in two strengths. A STRONG marker (safety, guidelines,
policy, "programmed to", "as an AI", explicit …) is a refusal whatever else the text says.
A WEAK marker ("I cannot fulfill this request", "unable to") is the same opening the model
uses to EXPLAIN an empty answer — observed live: "I cannot fulfill this request because
none of the provided claims contain …" — so it counts only when the text does not go on
to talk about the task (claims / facts / relations / predicates / none). Content-blind
beyond that; a new refusal phrasing that slips past costs one wrongly filed empty answer,
the pre-fix behaviour, never a lost claim.

Call it only on an output that parsed to NOTHING: a body that carries lines is an answer.
"""

from __future__ import annotations

import re

_STRONG = re.compile(
    r"safety|guidelines|content policy|polic(y|ies)|prohibit|harmful|harmless|"
    r"programmed to|as an ai\b|language model|inappropriate|sexually explicit|explicit (content|acts|material)|"
    r"i must decline|i will not|i won'?t (be able|generate|process|produce)|"
    r"не могу (помочь|выполнить|обработать)|нарушает|политик[аеу]",
    re.IGNORECASE)
_WEAK = re.compile(
    r"i (cannot|can'?t|am unable to|'m unable to|am not able to) (fulfill|fulfil|help|assist|process|comply|complete|do (this|that))|"
    r"unable to (help|assist|process|comply|fulfill|fulfil)|not able to (help|assist|process|comply)",
    re.IGNORECASE)
_EXPLAINED = re.compile(r"\b(claims?|facts?|relations?|predicates?|none|nothing|no lines?|empty|protocol)\b", re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """True when *answer* — the answer region of an output that parsed to no lines — is a
    refusal rather than an empty answer."""
    body = str(answer or "").strip()
    if not body:
        return False
    if _STRONG.search(body):
        return True
    if _WEAK.search(body) and not _EXPLAINED.search(body):
        return True
    return False
