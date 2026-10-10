#!/usr/bin/env python3
"""Re-deciding the reset test's applied decisions under the rule now (``core/reset_test.py``
``redecide_plan`` / ``apply_redecide``), offline and GPU-free.

    cd server
    .venv/bin/python reset_redecide.py              # the plan: what passes now that failed then
    .venv/bin/python reset_redecide.py --sheet      # …as a review sheet (server/data/asks/)
    .venv/bin/python reset_redecide.py --apply      # restore the superseded ones (inference DOWN)

The rule's switches are read from server_config.json (``reset_test.capture_gates``,
``reset_test.accept_shifted``). ``--apply`` writes the live op-log and ledger: run it with
inference down (the watchdog job ``reset_redecide`` stops and relaunches it). A position
the gate withheld is listed, never distilled here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SERVER_DIR / "inference"))
sys.path.insert(0, str(_SERVER_DIR))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args(argv)
    from core import ask_judge as aj
    from core import distill_gate as dg
    from core import reset_test as rt
    from core.reflection_writer import ReflectionWriter
    from training.ledger import ConsolidationLedger
    from training.reflections_path import consolidation_dir, load_server_config, memory_dir

    cfg = (load_server_config() or {}).get("reset_test") or {}
    rt.set_policy(capture_gates=bool(cfg.get("capture_gates", False)), accept_shifted=bool(cfg.get("accept_shifted", True)))
    mem, cons = Path(memory_dir()), Path(consolidation_dir())
    rag, weights = dg.read_oplogs([mem])
    plan = rt.redecide_plan(rag, weights)
    restore = [it for it in plan if it["action"] == "restore"]
    withheld = [it for it in plan if it["action"] == "withheld"]
    print(f"rule {rt.rule()}: {len(restore)} superseded position(s) pass now (restore), "
          f"{len(withheld)} withheld one(s) pass now (left to the gate)")
    if args.sheet:
        lines = [f"# Reset test: positions that pass under the rule {rt.rule()}", ""]
        for title, items in (("Superseded then, restored by --apply", restore), ("Withheld by the gate (listed only)", withheld)):
            lines += [f"## {title} — {len(items)}", ""]
            for it in items:
                op, c = it["op"], it["closure"]
                lines.append(f"- **{op.get('holds')} / {op.get('origin')} / capture {op.get('capture')}** "
                             f"({str(c.get('closed_ts'))[:10]}) — *{c.get('question')}*")
                lines.append(f"  - {it['content']}")
            lines.append("")
        out = _SERVER_DIR / "data" / "asks" / f"redecide_{time.strftime('%Y%m%d_%H%M%S')}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(out)
    if args.apply and restore:
        run_id = f"reset-redecide-{time.strftime('%Y%m%d_%H%M%S')}"
        tally = rt.apply_redecide(restore, rag, writer=ReflectionWriter(mem, cons), ledger=ConsolidationLedger(cons),
                                  ledger_fold=ConsolidationLedger(cons).fold(), run_id=run_id,
                                  say=lambda s: print(s, flush=True))
        print(f"applied {run_id}: {dict(tally)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
