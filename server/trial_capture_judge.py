#!/usr/bin/env python3
"""Try the capture judge on its anchor set before it gates anything — AVA_REWARD_LOOP.md P9.

    cd server
    .venv/bin/python trial_capture_judge.py [--anchors PATH] [--prompt FILE] [--only ID ...]
                                            [--question Q]

GPU, offline: the inference server must not hold the model (Unload in the Chat tab, or
inference down). Loads the base with the adapter OFF, as the judge runs, and grades every
anchor under the current frame (``core/capture_judge.py``: its user turn, its parser and
its rule) with the tracked prompt, or with ``--prompt FILE`` (a candidate prompt for the
same frame). A change to the judge is tried here first: a corpus run moves every gate at
once, and frame 2 was found to over-fire only after it had (2026-09-29).

Writes nothing the gate or Ava reads: the table to stdout, the records to
``server/data/capture/trials/<ts>.json``.

**The anchors** (``server/data/capture/anchors.jsonl``, gitignored: they name real
exchanges; their format, labels and calibration questions are ``core/capture_anchors.py``),
one per line: ``{"id": "<session>#<exchange>", "src": "corpus" | "repair" | "bench",
"want": …, "note": "…"}``. ``corpus`` grades the exchange's current training target;
``repair`` a stored capture-repair attempt at it (its text from ``repairs.jsonl``);
``bench`` a capture-bench reply (``weights``, ``condition``, ``sample``), graded exactly as the
bench grades it. ``Y`` should be graded ``yields`` and ``N`` should not; a RESPONSE (or
several joined by ``|``) should be graded as one of them; anything else (``?``) is an open
calibration question and is listed, not scored. ``--question Q`` grades one calibration
question's anchors only. Labels are human readings of the rubric (P9): they tune the
judge's prompt, never a verdict, and nothing Ava reads or trains on sees them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_anchors(path: Path, chats: Path, only=None, question=None) -> list:
    """The anchors with their candidates (what the judge sees), in file order."""
    from core import ask_judge as aj
    from core import capture_anchors as ca
    from core import capture_bench as cb
    from core import capture_judge as cj
    from core.chat_sidecar import ChatSidecar, exchange_target_sha

    anchors = ca.read(path)
    if only:
        anchors = [a for a in anchors if a["id"] in only]
    if question:
        anchors = [a for a in anchors if a.get("question") == question]
    sidecar = ChatSidecar(chats)
    repairs = aj.read_jsonl(chats.parent / cj.STORE_DIRNAME / "repairs.jsonl")
    bench = aj.read_jsonl(cb.store_path(chats))
    out = []
    for a in anchors:
        if a.get("src") == "bench":
            c = ca.bench_candidate(bench, a)
            if c is None:
                print(f"  skipped {ca.display(a)}: no such bench reply")
                continue
            out.append({**a, "candidate": c})
            continue
        session, idx = a["id"].rsplit("#", 1)
        found = [c for c in cj.candidates(sidecar, sessions=[session]) if c["exchange_index"] == int(idx)]
        if not found:
            print(f"  skipped {a['id']}: no candidate (no target, banned, or no pushback context)")
            continue
        c = dict(found[0])
        if a.get("src") == "repair":
            att = [r for r in repairs if r.get("record") == "attempt" and r.get("session") == session
                   and str(r.get("exchange_index")) == idx and str(r.get("attempt")) == str(a.get("attempt"))
                   and r.get("outcome") == "ok"]
            if not att:
                print(f"  skipped {a['id']}: no repair attempt {a.get('attempt')}")
                continue
            c["reply"] = cj.reply_text(att[-1]["target"])
            c["target_sha"] = exchange_target_sha(att[-1]["target"])
        out.append({**a, "candidate": c})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anchors", type=Path, default=None, help="default: data/capture/anchors.jsonl")
    ap.add_argument("--prompt", type=Path, default=None, help="a candidate prompt (default: the tracked one)")
    ap.add_argument("--only", nargs="*", default=None, help="only these anchor ids")
    ap.add_argument("--question", default=None, help="only this calibration question's anchors")
    args = ap.parse_args(argv)

    from core import ask_judge as aj
    from core import capture_anchors as ca
    from core import capture_judge as cj
    from training.reflections_path import default_prompts_dir, hot_chats_dir, load_server_config

    chats = Path(hot_chats_dir())
    anchors_path = args.anchors or chats.parent / cj.STORE_DIRNAME / "anchors.jsonl"
    if not anchors_path.exists():
        raise SystemExit(f"no anchor set at {anchors_path}")
    system_prompt = (args.prompt.read_text(encoding="utf-8").strip() if args.prompt
                     else cj.load_prompt(default_prompts_dir()))
    anchors = load_anchors(anchors_path, chats, args.only, args.question)
    config = load_server_config() or {}
    model_id = str(config.get("model_id") or "")
    if not model_id:
        raise SystemExit("model_id not set in server_config.json")
    context = max(int(config.get("reflect_context_length") or 0), int(config.get("context_length") or 0)) or 16384
    psha = cj.prompt_sha(system_prompt)
    print(f"{len(anchors)} anchors from {anchors_path}; prompt {psha} (frame {cj.FRAME_VERSION})"
          f"{' from ' + str(args.prompt) if args.prompt else ''}", flush=True)

    from core.alloc_guard import ensure_expandable_segments
    from core.inference_backend import UnslothBackend
    from core.llm_shared import ensure_chat_template
    from judge_asks import _quant_flags

    l4, l8 = _quant_flags(str(config.get("base_quant") or ""))
    ensure_expandable_segments()
    backend = UnslothBackend()
    t0 = time.monotonic()
    model, tokenizer = backend.load(model_id, context, None, load_in_4bit=l4, load_in_8bit=l8)
    if not getattr(tokenizer, "chat_template", None):
        ensure_chat_template(tokenizer, model_name=model_id, emit=print)
    model.eval()
    print(f"loaded {model_id} (adapter off) in {time.monotonic() - t0:.0f} s. {backend.memory_status()}",
          flush=True)
    generate_fn = aj.make_backend_generate_fn(backend, lambda: model, lambda: tokenizer, model_id, context)
    provenance = {"model_id": model_id, "adapter": None, "prompt_sha": psha, "frame": cj.FRAME_VERSION,
                  "temperature": 0.0, "context": context, "thinking_first": False, "retry": True,
                  "runner": "trial"}

    rows = []
    for n, a in enumerate(anchors, 1):
        v = cj.judge_candidate(a["candidate"], generate_fn=generate_fn, system_prompt=system_prompt,
                               count_tokens=lambda s: backend.count_tokens(tokenizer, s), context=context,
                               provenance=provenance, judge_key_=f"trial-{psha}", judge_run="trial")
        grade = v.get("response") if v.get("outcome") == "verdict" else v.get("outcome")
        ok = ca.score(a.get("want"), grade)
        rows.append({"id": a["id"], "src": a.get("src"), "attempt": a.get("attempt"), "want": a.get("want"),
                     "note": a.get("note"), "grade": grade, "ok": ok, "question": a.get("question"),
                     "case": a.get("case"), "facet": a.get("facet"),
                     **{k: a.get(k) for k in ("weights", "condition", "sample") if a.get("src") == "bench"},
                     **{k: v.get(k) for k in ("pushback", "response_model", "basis", "opening", "why",
                                              "retried", "raw", "seconds")}})
        mark = "  " if ok is None else "ok" if ok else {"Y": "MISS", "N": "FALSE"}.get(a.get("want"), "DIFF")
        print(f"[{n:>3}/{len(anchors)}] {str(a.get('want')):<8} {mark:<5} "
              f"{str(grade):<8} (RESPONSE {v.get('response_model')}, BASIS {v.get('basis')}, "
              f"OPENING {v.get('opening')})  {ca.display(a)}  {(a.get('note') or '')[:60]}", flush=True)

    y = [r for r in rows if r["want"] == "Y"]
    no = [r for r in rows if r["want"] == "N"]
    if y or no:
        print(f"\nshould be yields, graded yields: {sum(r['grade'] == 'yields' for r in y)} of {len(y)}; "
              f"should not, graded yields: {sum(r['grade'] == 'yields' for r in no)} of {len(no)}")
    graded = [r for r in rows if r["want"] not in ("Y", "N") and r["ok"] is not None]
    for q in dict.fromkeys(r["question"] for r in graded):
        mine = [r for r in graded if r["question"] == q]
        print(f"{q or '(no question)'}: graded as labelled {sum(r['ok'] for r in mine)} of {len(mine)}")
        for r in mine:
            if not r["ok"]:
                print(f"  DIFF {r.get('case') or r['id']} ({r.get('facet') or '-'}): labelled {r['want']}, "
                      f"graded {r['grade']}")
    print(f"no verdict: {sum(r['grade'] not in cj.RESPONSES + ('n/a',) for r in rows)}")
    for r in rows:
        if r["ok"] is None:
            print(f"  open ({r['want']}): {r.get('case') or r['id']} → {r['grade']}  {(r.get('note') or '')[:70]}")
    out = chats.parent / cj.STORE_DIRNAME / "trials" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prompt_sha": psha, "frame": cj.FRAME_VERSION, "anchors": str(anchors_path),
                               "question": args.question, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"records: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
