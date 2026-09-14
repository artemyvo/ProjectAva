"""Prompt-mutation pass — LOGGED-ONLY counterfactual on the standing prompt.

The revision pass asks *"was this reply mine?"* and, when it wasn't, produces the
reply it stands behind instead (the weight-space fix). This pass asks the question one
level up, in **prompt space**: *would I have answered as the better me on my own, if my
standing prompt had said something it does not currently say — and what would it need
to say?* (See documentation/AVA_OPEN_PROBLEMS.md → *Prompt Self-Modification* and the
persona→prompt-mutation design path.)

**This module is logged-only. It mutates nothing.** It runs on a *revise* exchange (the
drift signal the revision pass already produced), measures it against the persona digest
(the "better me"), and appends any proposed standing-prompt delta to a separate
append-only op-log — ``data/hot/prompt/prompt_deltas.jsonl`` — surfaced read-only in the
Debug tab. No prompt is ever changed; the live ``chat_prompt.txt`` is read but never
written. Aggregating, gating (Curiosity-Token / maturity), validating, and actually
promoting a delta are the future, separate steps.

The pure halves (parse / gap-test / read-write the log) are GPU-free and self-tested;
the orchestration that needs a ``generate_fn`` lives in the runner.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROMPT_DELTAS_FILE = "prompt_deltas.jsonl"

_THINK_RE = re.compile(r"(?is)<think>.*?</think>")
# Field labels emitted by prompts/prompt_mutation_prompt.txt. Lenient — tolerant of
# markdown bold/heading noise around the label, like the consolidation/revision parsers.
_FIELDS = ("VERDICT", "DRIFT", "MISSING", "DELTA", "SCOPE")
_VALID_VERDICTS = ("prompt-adequate", "prompt-gap")
_VALID_SCOPES = ("disposition", "line", "voice")


# ── parsing (GPU-free) ─────────────────────────────────────────────────────── #

def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _field(body: str, label: str) -> str:
    """Pull a single labelled field's value (one logical line) from *body*.

    Matches ``LABEL:`` at a line start (tolerating leading ``#``/``*``/``>`` and bold
    markers) and takes the rest of that line. Returns "" when absent.
    """
    m = re.search(
        rf"(?im)^[\s*#>]*{label}\s*\**\s*:\s*\**\s*(.+?)\s*$",
        body,
    )
    return m.group(1).strip(" *") if m else ""


def parse_prompt_mutation(text: str) -> dict:
    """Parse a prompt-mutation pass output into structured fields (no writes).

    Returns ``{verdict, drift, missing, delta, scope}`` — empty strings for absent
    fields. ``verdict``/``scope`` are normalised to the known vocab when recognisable,
    else passed through verbatim so an off-spec generation is still inspectable.
    """
    body = _strip_think(text)
    verdict = _field(body, "VERDICT").lower()
    if verdict not in _VALID_VERDICTS:
        # Tolerate "gap"/"adequate"/"prompt gap" etc.
        if "gap" in verdict:
            verdict = "prompt-gap"
        elif "adequate" in verdict:
            verdict = "prompt-adequate"
    scope = _field(body, "SCOPE").lower()
    scope = next((s for s in _VALID_SCOPES if s in scope), scope)
    return {
        "verdict": verdict,
        "drift": _field(body, "DRIFT"),
        "missing": _field(body, "MISSING"),
        "delta": _field(body, "DELTA"),
        "scope": scope,
    }


def _is_noneish(value: str) -> bool:
    return (value or "").strip().lower() in ("", "none", "n/a", "na", "-", "—")


def has_prompt_gap(parsed: dict) -> bool:
    """True when the pass proposes a real standing-prompt change worth logging.

    Requires the ``prompt-gap`` verdict *and* a non-empty, non-"none" DELTA — so a
    ``prompt-adequate`` verdict (the common case) and a gap with no concrete line both
    fall through silently rather than littering the log with empties.
    """
    if (parsed.get("verdict") or "") != "prompt-gap":
        return False
    return not _is_noneish(parsed.get("delta", ""))


# ── prompt loading (GPU-free) ──────────────────────────────────────────────── #

def _prompts_dir(prompts_dir: Optional[Path]) -> Path:
    if prompts_dir is not None:
        return Path(prompts_dir)
    return Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_mutation_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the counterfactual prompt-mutation system prompt."""
    return (_prompts_dir(prompts_dir) / "prompt_mutation_prompt.txt").read_text(
        encoding="utf-8").strip()


def load_current_chat_prompt(prompts_dir: Optional[Path] = None) -> str:
    """Load the live standing chat prompt — the text a delta would be measured against."""
    try:
        return (_prompts_dir(prompts_dir) / "chat_prompt.txt").read_text(
            encoding="utf-8").strip()
    except Exception:
        return ""


# ── op-log read/write (GPU-free) ───────────────────────────────────────────── #

def append_prompt_delta(prompt_dir: Path, record: dict) -> None:
    """Append one proposed-delta record to the append-only op-log (best-effort)."""
    prompt_dir = Path(prompt_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with (prompt_dir / PROMPT_DELTAS_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_prompt_deltas(prompt_dir: Path, limit: Optional[int] = None) -> list[dict]:
    """Read the logged prompt-delta proposals, newest first (best-effort).

    Tolerates partial/corrupt lines (skips them) so a torn append never blanks the
    Debug view. *limit* caps the number returned (after the newest-first sort).
    """
    path = Path(prompt_dir) / PROMPT_DELTAS_FILE
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    out.reverse()
    return out[:limit] if limit else out


# ── GPU-free self-test ─────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import tempfile

    sample = (
        "<think>Let me look at this. I softened the disagreement to be polite.</think>\n"
        "VERDICT: prompt-gap\n"
        "DRIFT: I hedged a real disagreement into a vague maybe to keep things pleasant.\n"
        "MISSING: My prompt never tells me it's fine to flatly disagree and stay there.\n"
        "DELTA: When I disagree, I say so plainly and do not soften it into a maybe to "
        "keep the peace.\n"
        "SCOPE: disposition\n"
    )
    p = parse_prompt_mutation(sample)
    assert p["verdict"] == "prompt-gap", p
    assert p["scope"] == "disposition", p
    assert p["delta"].startswith("When I disagree"), p
    assert has_prompt_gap(p)

    adequate = "VERDICT: prompt-adequate\nMISSING: none\nDELTA: none\nSCOPE: none\n"
    pa = parse_prompt_mutation(adequate)
    assert pa["verdict"] == "prompt-adequate", pa
    assert not has_prompt_gap(pa)

    # off-spec verdict phrasings normalise
    assert parse_prompt_mutation("VERDICT: there is a gap")["verdict"] == "prompt-gap"

    with tempfile.TemporaryDirectory() as d:
        append_prompt_delta(Path(d), {"run_id": "r1", "delta": "first", **p})
        append_prompt_delta(Path(d), {"run_id": "r2", "delta": "second"})
        rows = read_prompt_deltas(Path(d))
        assert len(rows) == 2 and rows[0]["delta"] == "second", rows  # newest first
        assert rows[0]["ts"], "ts auto-stamped"

    print("prompt_mutation self-test OK")
