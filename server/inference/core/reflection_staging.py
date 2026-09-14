import os
import shutil
import difflib
import json
from pathlib import Path
from datetime import datetime

from core.chat_sidecar import chat_products as _chat_products
from core.chat_sidecar import iter_chat_products as _iter_chat_products

# Staging paths helper
def get_staging_paths(data_dir: Path) -> dict:
    staging_dir = data_dir / "hot" / "reflection_staging"
    return {
        "staging_dir": staging_dir,
        "chats_dir": staging_dir / "chats",
        "memory_dir": staging_dir / "memory",
        "consolidation_dir": staging_dir / "consolidation",
        "archive_dir": staging_dir / "archive",
    }


# ── per-chat completion checkpoint ──────────────────────────────────────── #
# A reflection run writes everything to the staging workspace and only commits to
# live at the very end, on a `completed` run. If the run is stopped/crashed before
# then, `clear_staging` (default) wipes the staging dir on the NEXT run, so even
# fully-reflected chats are re-reflected. The checkpoint is a durable mirror that
# `clear_staging` never touches (it lives OUTSIDE reflection_staging/): each time a
# session finishes, its frozen sidecar + the run's cumulative staged memory/ledger
# deltas are copied here. On the next run's start `fold_checkpoint_to_live` folds a
# surviving checkpoint into live (so those chats read as frozen — reflect-once — and
# are skipped) and deletes it. A `completed` run discards its own checkpoint (staging
# is the source of truth), so folding only ever recovers an interrupted run's work.
def get_checkpoint_paths(data_dir: Path) -> dict:
    ckpt = data_dir / "hot" / "reflection_checkpoint"
    return {
        "checkpoint_dir": ckpt,
        "chats_dir": ckpt / "chats",
        "memory_dir": ckpt / "memory",
        "consolidation_dir": ckpt / "consolidation",
        "info": ckpt / "CHECKPOINT_INFO.json",
    }


def has_checkpoint(data_dir: Path) -> bool:
    """True when a prior interrupted run left recoverable completed chats."""
    chats = get_checkpoint_paths(data_dir)["chats_dir"]
    try:
        return chats.exists() and any(chats.glob("*.state.json"))
    except Exception:
        return False


def checkpoint_completed_session(data_dir: Path, session_filename: str, run_id: str) -> None:
    """Mirror a just-completed session into the durable checkpoint (best-effort).

    Called by the runner right after a session's sidecar is frozen (reflect-once).
    Copies that session's frozen sidecar plus a refreshed snapshot of the run's
    staged memory/ledger delta files. The runner processes sessions sequentially, so
    at this point the staged delta files hold exactly the deltas of the sessions
    completed so far — a rolling snapshot that, folded once, reproduces all completed
    work without per-session line slicing.
    """
    try:
        staging = get_staging_paths(data_dir)
        ckpt = get_checkpoint_paths(data_dir)
        ckpt["chats_dir"].mkdir(parents=True, exist_ok=True)
        ckpt["memory_dir"].mkdir(parents=True, exist_ok=True)
        ckpt["consolidation_dir"].mkdir(parents=True, exist_ok=True)

        # 1. This session's reflection products (frozen sidecar + summary + facts).
        stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
        for src in _chat_products(staging["chats_dir"], stem):
            shutil.copy2(src, ckpt["chats_dir"] / src.name)

        # 2. Refresh the cumulative delta mirrors (sessions completed so far).
        for key, fname in (("memory_dir", "rag_memory.jsonl"),
                           ("memory_dir", "weights_persona.jsonl"),
                           ("consolidation_dir", "consolidation_anchors.jsonl")):
            src = staging[key] / fname
            if src.exists():
                shutil.copy2(src, ckpt[key] / fname)

        # 3. Provenance (best-effort).
        sessions = []
        try:
            if ckpt["info"].exists():
                sessions = list(
                    json.loads(ckpt["info"].read_text(encoding="utf-8")).get("sessions") or [])
        except Exception:
            sessions = []
        if session_filename not in sessions:
            sessions.append(session_filename)
        ckpt["info"].write_text(
            json.dumps({"run_id": run_id, "sessions": sessions,
                        "updated_at": datetime.now().isoformat()}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass


def discard_checkpoint(data_dir: Path) -> None:
    """Delete the completion checkpoint (a `completed` run leaves no data to recover)."""
    ckpt_dir = get_checkpoint_paths(data_dir)["checkpoint_dir"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def fold_checkpoint_to_live(data_dir: Path, inference_dir: Path) -> dict:
    """Recover an interrupted run's completed chats into live, then delete the checkpoint.

    Copies each checkpointed frozen sidecar to live (skipping any already frozen live)
    and appends the cumulative memory/ledger deltas to the live op-logs, then rebuilds
    the live reflection RAG index so the recovered conclusions are retrievable. Consume-
    once: the checkpoint is removed after folding, so it can never double-append.
    """
    ckpt = get_checkpoint_paths(data_dir)
    if not has_checkpoint(data_dir):
        return {}
    # Live chats moved out of data_dir to the ordered server/data/chats root (mirrors
    # run_stage_commit_training).
    live_chats_dir = data_dir.parent.parent / "data" / "chats"
    live_memory_dir = data_dir / "hot" / "memory"
    live_consolidation_dir = data_dir / "hot" / "consolidation"
    live_chats_dir.mkdir(parents=True, exist_ok=True)

    sidecars = 0
    for sp in _iter_chat_products(ckpt["chats_dir"]):
        lp = live_chats_dir / sp.name
        # Reflect-once: never clobber a chat already frozen live. Keyed on the STATE
        # sidecar for all three products — if this chat was fully reflected live since
        # the checkpoint was written, the checkpoint's whole view of it is stale, and
        # promoting its summary over the newer one would undo that run.
        stem = sp.name.split(".", 1)[0]
        live_state = live_chats_dir / f"{stem}.state.json"
        if live_state.exists():
            try:
                if json.loads(live_state.read_text(encoding="utf-8")).get("reflected_at"):
                    continue
            except Exception:
                pass
        shutil.copy2(sp, lp)
        sidecars += 1

    rag_lines = append_file_to_file(
        ckpt["memory_dir"] / "rag_memory.jsonl", live_memory_dir / "rag_memory.jsonl")
    w_lines = append_file_to_file(
        ckpt["memory_dir"] / "weights_persona.jsonl", live_memory_dir / "weights_persona.jsonl")
    l_lines = append_file_to_file(
        ckpt["consolidation_dir"] / "consolidation_anchors.jsonl",
        live_consolidation_dir / "consolidation_anchors.jsonl")

    # Rebuild live RAG so the recovered conclusions are retrievable this run.
    try:
        from core.rag_engine import RagEngine
        rag = RagEngine(
            live_chats_dir, inference_dir / "prompts",
            memory_dir=live_memory_dir, consolidation_dir=live_consolidation_dir,
        )
        rag.build_index_async()
    except Exception:
        pass

    discard_checkpoint(data_dir)
    return {
        "recovered_sidecars": sidecars,
        "recovered_rag_lines": rag_lines,
        "recovered_weights_lines": w_lines,
        "recovered_ledger_lines": l_lines,
    }


# ── background per-chat pass: two-stage-freeze plumbing ─────────────────────── #
# The background reflection pass (core.background_reflection) reflects ONE aged chat
# per idle wake in a throwaway staging workspace, then — only if the chat completed —
# appends that chat's staged deltas + frozen (chat_reflected) sidecar into the SAME
# durable checkpoint the crash-recovery path uses, and drops the chat's clean-base job
# payloads into a sibling dir. The checkpoint accumulates across wakes; the next normal
# reflection run folds it (`fold_checkpoint_to_live`, unchanged: it copies the
# chat_reflected sidecars to live and appends the deltas), then finishes each chat by
# running the run-level clean-base passes over the persisted jobs and stamping
# ``reflected_at``. Because each wake reflects exactly ONE chat in a fresh staging dir,
# the commit is a clean per-chat append and a preempted (partial) chat leaves the
# throwaway staging to be discarded — nothing reaches the checkpoint. The pending-jobs
# dir is a SIBLING of the checkpoint (not inside it), so ``discard_checkpoint``'s rmtree
# during the fold never deletes jobs the main pass has not yet consumed.

def get_pending_clean_base_dir(data_dir: Path) -> Path:
    return data_dir / "hot" / "reflection_pending_clean_base"


def write_pending_clean_base(data_dir: Path, session_filename: str,
                             judge_jobs: list, fact_candidates: list) -> None:
    """Persist a background-reflected chat's deferred clean-base job payloads.

    ``judge_jobs`` / ``fact_candidates`` are the plain-dict entries the runner collects
    during revision (branch-judge + fact-placement inputs). A normal reflection run loads
    them for this chat via :func:`load_pending_clean_base` and feeds them straight into its
    end-of-run clean-base phase. Best-effort."""
    try:
        stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
        d = get_pending_clean_base_dir(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.json").write_text(
            json.dumps({"judge_jobs": list(judge_jobs or []),
                        "fact_candidates": list(fact_candidates or [])},
                       ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass


def load_pending_clean_base(data_dir: Path, session_filename: str) -> dict:
    """Return ``{"judge_jobs": [...], "fact_candidates": [...]}`` for a chat_reflected chat,
    or empty lists when no background jobs were persisted for it."""
    stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
    p = get_pending_clean_base_dir(data_dir) / f"{stem}.json"
    if not p.exists():
        return {"judge_jobs": [], "fact_candidates": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"judge_jobs": list(data.get("judge_jobs") or []),
                "fact_candidates": list(data.get("fact_candidates") or [])}
    except Exception:
        return {"judge_jobs": [], "fact_candidates": []}


def delete_pending_clean_base(data_dir: Path, session_filename: str) -> None:
    """Consume-once: drop a chat's pending-jobs file after the main pass loads it."""
    stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
    try:
        (get_pending_clean_base_dir(data_dir) / f"{stem}.json").unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def purge_background_artifacts(data_dir: Path, session_filename: str) -> None:
    """Remove a chat's background per-chat staging so an UN-FROZEN chat cannot be re-frozen
    or double-folded by a later normal run's checkpoint fold.

    A chat frozen only at the ``chat_reflected`` stage has two durable background artifacts:
    its deferred clean-base jobs (``reflection_pending_clean_base/<stem>.json``) and its
    frozen sidecar mirrored into the checkpoint (``reflection_checkpoint/chats/<stem>.state
    .json``). When such a chat is un-frozen because it was continued in place
    (``ChatSidecar.clear_reflection_freeze``), those must go too: otherwise the next normal
    run would consume the stale clean-base jobs and copy the OLD frozen sidecar back over the
    live one (``fold_checkpoint_to_live`` only skips a ``reflected_at`` live sidecar), silently
    re-freezing the chat and re-losing the appended turns. The checkpoint's *cumulative*
    memory / ledger deltas are intentionally left — they fold to live and dedup by
    ``content_key`` against the chat's re-reflection, so they are self-healing. Best-effort."""
    delete_pending_clean_base(data_dir, session_filename)
    stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
    try:
        ckpt = get_checkpoint_paths(data_dir)
        for p in _chat_products(ckpt["chats_dir"], stem):
            p.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def commit_background_chat(data_dir: Path, staging_dir: Path, session_filename: str,
                           run_id: str) -> None:
    """Fold ONE completed background chat's throwaway staging into the durable checkpoint.

    *staging_dir* is that wake's fresh staging workspace holding exactly this chat's
    deltas + its frozen (``chat_reflected``) sidecar. Appends the delta op-logs to the
    checkpoint (never replace — the checkpoint accumulates across wakes) and copies the
    sidecar. A later ``fold_checkpoint_to_live`` promotes it. Best-effort."""
    try:
        staging = {
            "chats_dir": Path(staging_dir) / "chats",
            "memory_dir": Path(staging_dir) / "memory",
            "consolidation_dir": Path(staging_dir) / "consolidation",
        }
        ckpt = get_checkpoint_paths(data_dir)
        ckpt["chats_dir"].mkdir(parents=True, exist_ok=True)
        ckpt["memory_dir"].mkdir(parents=True, exist_ok=True)
        ckpt["consolidation_dir"].mkdir(parents=True, exist_ok=True)

        # 1. This chat's reflection products (frozen `chat_reflected` sidecar + summary
        #    + facts).
        stem = session_filename[:-5] if session_filename.endswith(".json") else session_filename
        for src in _chat_products(staging["chats_dir"], stem):
            shutil.copy2(src, ckpt["chats_dir"] / src.name)

        # 2. APPEND this chat's deltas (staging holds exactly one chat's worth).
        for key, fname in (("memory_dir", "rag_memory.jsonl"),
                           ("memory_dir", "weights_persona.jsonl"),
                           ("consolidation_dir", "consolidation_anchors.jsonl")):
            append_file_to_file(staging[key] / fname, ckpt[key] / fname)

        # 3. Provenance (best-effort).
        sessions = []
        try:
            if ckpt["info"].exists():
                sessions = list(
                    json.loads(ckpt["info"].read_text(encoding="utf-8")).get("sessions") or [])
        except Exception:
            sessions = []
        if session_filename not in sessions:
            sessions.append(session_filename)
        ckpt["info"].write_text(
            json.dumps({"run_id": run_id, "sessions": sessions,
                        "updated_at": datetime.now().isoformat()}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass


# ── staging owner pointer ───────────────────────────────────────────────── #
# Ties the (single-occupancy) staging workspace to the reflection run(s) that
# produced its contents, so a standalone stage command can attribute itself back
# to the right run(s) in the reflection log. Reset on a fresh staged run, appended
# on continue-staging (multiple runs accumulating into one workspace).
_OWNER_FILE = "STAGED_BY.json"


def write_staging_owner(staging_dir: Path, run_id: str, *, append: bool = False) -> None:
    staging_dir = Path(staging_dir)
    run_ids = read_staging_owners(staging_dir) if append else []
    if run_id not in run_ids:
        run_ids.append(run_id)
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / _OWNER_FILE).write_text(
            json.dumps({"run_ids": run_ids, "staged_at": datetime.now().isoformat()},
                       ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def read_staging_owners(staging_dir: Path) -> list:
    """Run ids that contributed to the current staging workspace ([] if none/unreadable)."""
    path = Path(staging_dir) / _OWNER_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(r) for r in (data.get("run_ids") or [])]
    except Exception:
        return []

def append_file_to_file(src: Path, dest: Path) -> int:
    if not src.exists():
        return 0
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        content = dest.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            with open(dest, "a", encoding="utf-8") as fh:
                fh.write("\n")
    with open(dest, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.strip() + "\n")
    return len(lines)

def run_stage_diff(data_dir: Path) -> str:
    paths = get_staging_paths(data_dir)
    # Live chats moved out of data_dir to the ordered server/data/chats root.
    live_chats_dir = data_dir.parent.parent / "data" / "chats"
    
    out = []
    out.append("\n🔍 Producing Sleep Pipeline Delta Diff...")
    out.append("═" * 60)
    
    # 1. RAG memory delta
    rag_staged = paths["memory_dir"] / "rag_memory.jsonl"
    if rag_staged.exists() and rag_staged.stat().st_size > 0:
        out.append("📝 Staged RAG Memory Changes (rag_memory.jsonl):")
        out.append("─" * 60)
        out.append(rag_staged.read_text(encoding="utf-8").strip())
        out.append("═" * 60)
        
    # 2. Weights delta
    weights_staged = paths["memory_dir"] / "weights_persona.jsonl"
    if weights_staged.exists() and weights_staged.stat().st_size > 0:
        out.append("📝 Staged Weights Persona Changes (weights_persona.jsonl):")
        out.append("─" * 60)
        out.append(weights_staged.read_text(encoding="utf-8").strip())
        out.append("═" * 60)
        
    # 3. Ledger delta
    ledger_staged = paths["consolidation_dir"] / "consolidation_anchors.jsonl"
    if ledger_staged.exists() and ledger_staged.stat().st_size > 0:
        out.append("📝 Staged Consolidation Ledger Changes (consolidation_anchors.jsonl):")
        out.append("─" * 60)
        out.append(ledger_staged.read_text(encoding="utf-8").strip())
        out.append("═" * 60)
        
    # 4. Chat sidecars diff
    if paths["chats_dir"].exists():
        staged_sidecars = list(_iter_chat_products(paths["chats_dir"]))
        if staged_sidecars:
            out.append("📝 Staged Chat Sidecar Diffs:")
            out.append("─" * 60)
            for sp in staged_sidecars:
                lp = live_chats_dir / sp.name
                sp_text = sp.read_text(encoding="utf-8").splitlines()
                if lp.exists():
                    lp_text = lp.read_text(encoding="utf-8").splitlines()
                else:
                    lp_text = []
                diff = list(difflib.unified_diff(
                    lp_text, sp_text,
                    fromfile=f"live/chats/{lp.name}", tofile=f"staged/chats/{sp.name}",
                    lineterm=""
                ))
                if diff:
                    out.append("\n".join(diff))
                else:
                    out.append(f"No changes to {sp.name} (staged version matches live)")
                out.append("─" * 60)
            out.append("═" * 60)
            
    return "\n".join(out)

def run_stage_merge_rag(data_dir: Path, inference_dir: Path) -> dict:
    paths = get_staging_paths(data_dir)
    live_memory_dir = data_dir / "hot" / "memory"
    live_consolidation_dir = data_dir / "hot" / "consolidation"
    # Live chats moved out of data_dir to the ordered server/data/chats root.
    live_chats_dir = data_dir.parent.parent / "data" / "chats"

    print("Merging RAG memory delta to live RAG log...", flush=True)
    rag_staged = paths["memory_dir"] / "rag_memory.jsonl"
    rag_live = live_memory_dir / "rag_memory.jsonl"
    appended = append_file_to_file(rag_staged, rag_live)
    print(f"Merged {appended} RAG memory line(s) into {rag_live}.", flush=True)

    # Rebuild live RAG index
    from core.rag_engine import RagEngine
    rag = RagEngine(
        live_chats_dir, inference_dir / "prompts",
        memory_dir=live_memory_dir, consolidation_dir=live_consolidation_dir,
    )
    print("Rebuilding live RAG index...", flush=True)
    rag.build_index_async()
    return {"rag_memory_lines": appended}

def run_stage_commit_training(data_dir: Path) -> dict:
    paths = get_staging_paths(data_dir)
    # Live chats moved out of data_dir to the ordered server/data/chats root
    # (matching run_stage_diff / run_stage_merge_rag).
    live_chats_dir = data_dir.parent.parent / "data" / "chats"
    live_memory_dir = data_dir / "hot" / "memory"
    live_consolidation_dir = data_dir / "hot" / "consolidation"
    archived_count = 0

    print("Merging weights persona delta to live...", flush=True)
    w_staged = paths["memory_dir"] / "weights_persona.jsonl"
    w_live = live_memory_dir / "weights_persona.jsonl"
    w_appended = append_file_to_file(w_staged, w_live)
    print(f"Merged {w_appended} weights persona line(s) into {w_live}.", flush=True)
    
    print("Merging consolidation ledger anchors delta to live...", flush=True)
    l_staged = paths["consolidation_dir"] / "consolidation_anchors.jsonl"
    l_live = live_consolidation_dir / "consolidation_anchors.jsonl"
    l_appended = append_file_to_file(l_staged, l_live)
    print(f"Merged {l_appended} ledger line(s) into {l_live}.", flush=True)
    
    print("Committing staged sidecars to live...", flush=True)
    sidecars_copied = 0
    if paths["chats_dir"].exists():
        for sp in _iter_chat_products(paths["chats_dir"]):
            lp = live_chats_dir / sp.name
            shutil.copy2(sp, lp)
            sidecars_copied += 1
    print(f"Committed {sidecars_copied} sidecar file(s) to live chats.", flush=True)
    
    # Check if there are staged archived chats to commit
    staged_archive = paths["archive_dir"] / "chats"
    live_archive = data_dir.parent.parent / "data" / "archive" / "chats"
    if staged_archive.exists():
        live_archive.mkdir(parents=True, exist_ok=True)
        for f in staged_archive.glob("*"):
            # Copy to archive
            shutil.copy2(f, live_archive / f.name)
            # Delete from live hot chats
            live_hot = live_chats_dir / f.name
            if live_hot.exists():
                live_hot.unlink()
            archived_count += 1
        print(f"Committed {archived_count} archived file(s) to live archive and cleaned from live hot chats.", flush=True)

    return {
        "weights_persona_lines": w_appended,
        "ledger_lines": l_appended,
        "sidecars_copied": sidecars_copied,
        "archived": archived_count,
    }

def run_stage_apply(data_dir: Path, inference_dir: Path, server_dir: Path) -> dict:
    counts: dict = {}
    # 1. Merge RAG
    counts.update(run_stage_merge_rag(data_dir, inference_dir))
    # 2. Commit training / sidecars
    counts.update(run_stage_commit_training(data_dir))

    # 3. Promote candidate adapter if present
    candidate_dir = server_dir / "models" / "candidate"
    if candidate_dir.exists():
        new_adapter_id = server_dir / "models" / f"adapter-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Promoting candidate model adapter to {new_adapter_id}...", flush=True)
        new_adapter_id.parent.mkdir(parents=True, exist_ok=True)
        os.rename(candidate_dir, new_adapter_id)

        # Update config
        from reflections_path import load_server_config, save_server_config
        config = load_server_config()
        config["adapter_id"] = str(new_adapter_id)
        save_server_config(config)
        print("Updated server_config.json adapter_id to point to new adapter weights.", flush=True)
        counts["promoted_adapter"] = str(new_adapter_id)

    # 4. Clean up staging
    run_stage_discard(data_dir, server_dir)
    print("Sleep staging workspace committed successfully.", flush=True)
    return counts

def run_stage_discard(data_dir: Path, server_dir: Path) -> dict:
    paths = get_staging_paths(data_dir)
    print("Cleaning up Sleep staging workspace...", flush=True)
    if paths["staging_dir"].exists():
        shutil.rmtree(paths["staging_dir"])
    candidate_dir = server_dir / "models" / "candidate"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    print("Staging directories and candidate model weights deleted.", flush=True)
    return {}
