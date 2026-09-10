# `server/reflections/` — reflection review archive

A reviewable, revertable snapshot of each completed Sleep run. This tree is not live
serving state and is not a replay recipe. Everything except this README is gitignored.

## Layout

```
server/reflections/
  README.md
  <run_id>/
    run/
      <run_id>.meta.json
      <run_id>.events.jsonl
      <run_id>.summary.json
      <run_id>.report.json
    artifacts/
      rag_memory.jsonl
      weights_persona.jsonl
      consolidation_anchors.jsonl
      chats/
      archive/chats/
    persona/
      digest.json
    users/
      <person-slug>.json
    adapter/
```

`inference/core/reflection_archive.py` writes the tree in two best-effort phases:

1. At commit, `archive_reflection()` copies the run log, staged deltas, the current
   persona digest, and the current per-person user portraits before staging is discarded.
   Every portrait is copied, not only ones this run rewrote: the archive's contract is a
   snapshot of the live state at commit time, and a portrait left unchanged this run is
   still part of what Ava was working from.
2. After an offline train succeeds, `archive_adapter()` copies the produced adapter
   into the same run directory.

No root index, per-run manifest, replay provenance, or stage journal is stored. The
ordered replay mechanism was retired with the switch to wall-clock decay.

## Reverting

For a train-bearing run, its `adapter/` directory can still be inspected or selected
manually as a rollback aid. The `artifacts/` and `run/` directories document what the
reflection changed. They are review material, not an automatically applicable rebuild
recipe.
