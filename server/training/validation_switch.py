"""Master switch for training-time validation (the regression probe).

DISABLED for now — a **separate design project** (see
`documentation/AVA_OPEN_PROBLEMS.md → Validation` and `training/DESIGN.md → Probe-gated
promotion`). The five-tier probe — notably tier 5's answer-language check — is hard-coded
to a single (Russian) user and cannot gate a genuinely multilingual user base (French /
Greek / Hebrew); that same "user's voice" model is what cap-age contamination
(`REBUILD.md §5e`) drifts toward, so a correct probe is its own project, not a threshold
tweak. Until then every build promotes UNGUARDED; the adapter lineage + forensic snapshots
(`REBUILD.md §7`) keep any bad promotion reversible.

This is the single source of truth, kept in its OWN tiny module (no GPU/model imports) so
**both** sides can read it without dragging in `train_cycle` (which imports unsloth):
  * the training side (`train_cycle.run_cycle`) force-skips the probe when this is False;
  * the inference side (`reflection_service`) must not tell the operator that a
    judge-override cycle will be validated when it won't.

Flip to True to re-arm the probe (and revisit the multilingual redesign first).
"""
from __future__ import annotations

VALIDATION_ENABLED = False
