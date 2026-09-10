"""Server-owned reflection prompt loader — single source of truth for all reflection passes.

Loads prompt text from the canonical files under server/inference/prompts/ and exposes
a resolved prompt set so callers never reach into the filesystem themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class ReflectionPromptSet:
    sleep_prompt: str
    revision_prompt: str
    branch_prompt: str
    # The dedicated per-conversation SUMMARY pass (consolidation-gist). Optional: an older
    # checkout without summary_prompt.txt yields "" and the runner simply skips the pass.
    summary_prompt: str = ""


_cached: Optional[ReflectionPromptSet] = None


def load_reflection_prompts(prompts_dir: Optional[Path] = None) -> ReflectionPromptSet:
    """Load and cache the three canonical reflection prompts.

    Reads from *prompts_dir* when supplied (useful in tests); otherwise uses the
    standard server/inference/prompts/ directory. Result is module-level cached
    after the first call so repeated calls within one process are free.
    """
    global _cached
    if _cached is not None and prompts_dir is None:
        return _cached
    d = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
    summary_path = d / "summary_prompt.txt"
    result = ReflectionPromptSet(
        sleep_prompt=(d / "sleep_prompt.txt").read_text(encoding="utf-8").strip(),
        revision_prompt=(d / "revision_prompt.txt").read_text(encoding="utf-8").strip(),
        branch_prompt=(d / "branch_prompt.txt").read_text(encoding="utf-8").strip(),
        summary_prompt=(summary_path.read_text(encoding="utf-8").strip()
                        if summary_path.exists() else ""),
    )
    if prompts_dir is None:
        _cached = result
    return result
