"""The "Aha!" (ASSOCIATIVE_MEMORY.md §4, §4.1): standing needs, the two-sided spread with
the minimum across sides, the ledger keyed on (need, resource, path-type set), and the
judge — thinking on, four verdicts plus one sentence, an unparseable answer an `error`
that is never recorded.

`aha()` is arithmetic; `judge()` is the model. The application schedules the second.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .graph import EdgeTable
from .spread import HOPS, Reached, render_path, spread

AHA_MIN = 0.015             # convergence floor on min(need side, stimulus side): above the cell route alone
CELL_SEED_TOTAL = 0.15      # the need claim's cell postings seed weakly, sharing this much in total
                            # (§4: "the seed-time cell route (weak)") — a fixed share, so the route's
                            # weight does not depend on how many postings an embedder happens to return
MAX_JUDGEMENTS = 3
PROMPT_FILE = Path(__file__).parent / "prompts" / "judge_prompt.txt"
VERDICTS = ("connect", "satisfies", "no", "stale")


@dataclass
class Candidate:
    need: str                     # need node id
    need_claim: str               # claim id that raised it
    resource: str                 # claim id
    strength: float
    need_side: float
    stimulus_side: float
    path_types: list[str]
    paths: dict = field(default_factory=dict)        # {"need": [...], "stimulus": [...]} rendered
    passages: dict = field(default_factory=dict)     # {"need": text, "resource": text}
    bridges: list = field(default_factory=list)      # [(node, line)]
    prior: list = field(default_factory=list)
    verdict: Optional[str] = None
    link: str = ""
    scope: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("need", "need_claim", "resource", "strength", "need_side", "stimulus_side",
                                                 "path_types", "paths", "passages", "bridges", "prior", "verdict", "link", "scope")}


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self.entries.append(json.loads(line))
                except Exception:
                    pass

    @staticmethod
    def key(need: str, resource: str, path_types: list[str]) -> str:
        return f"{need}\x1f{resource}\x1f{','.join(sorted(set(path_types)))}"

    def prior_for(self, need: str) -> list[dict]:
        return [e for e in self.entries if e.get("need") == need]

    def blocked(self, need: str, resource: str, path_types: list[str]) -> bool:
        k = self.key(need, resource, path_types)
        for e in self.entries:
            if e.get("key") == k and e.get("verdict") in ("no", "connect", "satisfies", "stale"):
                return True
            if e.get("need") == need and e.get("resource") == resource and e.get("verdict") in ("satisfies", "stale"):
                return True
        return False

    def record(self, cand: Candidate, verdict: str, link: str, *, now: float) -> None:
        e = {"ts": now, "key": self.key(cand.need, cand.resource, cand.path_types), "need": cand.need, "resource": cand.resource,
             "path_types": sorted(set(cand.path_types)), "verdict": verdict, "link": link, "strength": round(cand.strength, 5)}
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _inferred_only(paths: list[dict]) -> bool:
    edges = [e for p in paths for e in p["edges"]]
    return bool(edges) and all(e[1].startswith("inferred:") for e in edges)


def need_seeds(table: EdgeTable, need_node: str, claims: dict, codebook=None) -> dict[str, float]:
    """The need node, the claim that raised it, its subject/entity nodes, plus that claim's
    cell postings at seed time (§4). Pass ``codebook=None`` for the CORE seeds alone — the
    novelty rule's hop distance is measured from those, never from a cell posting."""
    info = table.needs.get(need_node) or {}
    cid = info.get("claim_id")
    seeds: dict[str, float] = {need_node: 1.0}
    if cid:
        seeds[f"claim:{cid}"] = 1.0
    for other, etype, _ev in table.edges(need_node):
        if etype == "need":
            seeds[other] = 1.0
    if codebook is not None and cid and cid in claims:
        hits = [h for h in codebook.search(claims[cid]["text"], limit=12)[:12]]
        share = CELL_SEED_TOTAL / len(hits) if hits else 0.0
        for h in hits:
            node = f"claim:{h['id']}" if codebook.ids_kind.get(h["id"]) == "claim" else f"chunk:{h['id']}"
            if node not in seeds:
                seeds[node] = share
    return seeds


def find_candidates(table: EdgeTable, claims: dict, need_maps: dict[str, Reached], stimulus: Reached, *,
                    ledger: Ledger, scope: Optional[dict] = None, aha_min: float = AHA_MIN) -> list[Candidate]:
    out: list[Candidate] = []
    for need_node, nmap in need_maps.items():
        info = table.needs.get(need_node) or {}
        own = f"claim:{info.get('claim_id')}"
        for node, rs in stimulus.r.items():
            if not node.startswith("claim:") or node == own:
                continue
            rn = nmap.r.get(node, 0.0)
            if rn <= 0.0 or rs <= 0.0:
                continue
            # A resource is something the stimulus brings NEW to the need: reached from the
            # stimulus in fewer hops than from the need, and outside the need's own one-hop
            # neighbourhood. Without this the bridge facts — between the two, reached strongly
            # from both sides — outrank every genuine resource.
            hn, hs = nmap.hops.get(node, 99), stimulus.hops.get(node, 99)
            if hn < 2 or hs >= hn:
                continue
            strength = min(rn, rs)
            if strength < aha_min:
                continue
            cid = node.split(":", 1)[1]
            if cid not in claims:
                continue
            types = sorted({e[1].split(":", 1)[0] if not e[1].startswith("rel:") else e[1] for p in nmap.paths.get(node, []) + stimulus.paths.get(node, []) for e in p["edges"]})
            # An inferred edge (§4, B2) is never the SOLE path on either side.
            if any(_inferred_only(side.paths.get(node, [])) for side in (nmap, stimulus)):
                continue
            if ledger.blocked(need_node, cid, types):
                continue
            out.append(Candidate(need=need_node, need_claim=str(info.get("claim_id")), resource=cid, strength=strength,
                                 need_side=rn, stimulus_side=rs, path_types=types,
                                 paths={"need": [render_path(p) for p in nmap.paths.get(node, [])],
                                        "stimulus": [render_path(p) for p in stimulus.paths.get(node, [])]},
                                 prior=ledger.prior_for(need_node), scope=dict(scope or {})))
    out.sort(key=lambda c: -c.strength)
    return out


def fill_candidate(cand: Candidate, claims: dict, store, table: EdgeTable, need_maps: dict[str, Reached], stimulus: Reached) -> Candidate:
    """Attach the two passages and the bridge nodes' claim lines (§4.1 items 1–3)."""
    from .render import claim_line

    def passage(cid: str) -> str:
        c = claims.get(cid) or {}
        occ = (c.get("occurrences") or [{}])[0]
        doc = store.document(occ.get("doc_id")) if occ.get("doc_id") else None
        u = doc.unit(occ.get("chunk_id")) if doc and occ.get("chunk_id") else None
        head = f"[{occ.get('title') or occ.get('key')} — {str(occ.get('asserted_at') or '')[:10]}"
        head += f", {occ.get('speaker')}]" if occ.get("speaker") else "]"
        return head + "\n" + (u.text if u else c.get("text", ""))

    cand.passages = {"need": passage(cand.need_claim), "resource": passage(cand.resource)}
    bridges: list = []
    seen: set[str] = set()
    for side in (need_maps.get(cand.need), stimulus):
        if side is None:
            continue
        for p in side.paths.get(f"claim:{cand.resource}", []):
            for frm, et, to, _s in p["edges"]:
                for node in (frm, to):
                    if node in seen or node.startswith(("claim:", "chunk:", "doc:", "need:")):
                        continue
                    seen.add(node)
                    lines = [claim_line(claims[e[2]]) for e in table.edges(node) if e[1] in ("about", "mentions") and e[2] in claims][:3]
                    if lines:
                        bridges.append((node, " / ".join(lines)))
    cand.bridges = bridges
    return cand


# ----- the judge --------------------------------------------------------------------------------

def _judge_user(cand: Candidate, *, now_line: str, doing: str) -> str:
    parts = [f"NEED (as said):\n{cand.passages.get('need', '')}\n",
             f"RESOURCE (as said):\n{cand.passages.get('resource', '')}\n",
             "PATHS from the stimulus to the resource:\n" + "\n".join("  " + p for p in cand.paths.get("stimulus", [])) + "\n",
             "PATHS from the need to the resource:\n" + "\n".join("  " + p for p in cand.paths.get("need", [])) + "\n"]
    if cand.bridges:
        parts.append("On the bridges:\n" + "\n".join(f"  {n}: {l}" for n, l in cand.bridges) + "\n")
    parts.append(f"CLOCKS: {now_line}\n")
    if cand.prior:
        parts.append("EARLIER VERDICTS on this need:\n" + "\n".join(f"  {e.get('verdict')} — resource {e.get('resource')}: {e.get('link', '')}" for e in cand.prior[-6:]) + "\n")
    parts.append(f"NOW: {doing or 'idle'}\n")
    parts.append("Answer:\n\nVERDICT: <connect | satisfies | no | stale>\nLINK: <one sentence, or —>\n")
    return "\n".join(parts)


def parse_verdict(raw: str) -> tuple[Optional[str], str]:
    text = str(raw or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    verdict, link = None, ""
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("verdict:"):
            v = low.split(":", 1)[1].strip().strip("`*. ").split()[0] if low.split(":", 1)[1].strip() else ""
            if v in VERDICTS:
                verdict = v
        elif low.startswith("link:"):
            link = s.split(":", 1)[1].strip()
    return verdict, link


def judge(cand: Candidate, generate_fn: Callable, ledger: Ledger, *, now: float, now_line: str = "", doing: str = "",
          max_new_tokens: int = 2048) -> Candidate:
    system = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""
    user = _judge_user(cand, now_line=now_line, doing=doing)
    verdict, link = None, ""
    for _attempt in range(2):
        try:
            raw = generate_fn(system, user, thinking=True, max_new_tokens=max_new_tokens, temperature=0.0)
            if isinstance(raw, tuple):
                raw = raw[0]
        except Exception:  # noqa: BLE001
            raw = ""
        verdict, link = parse_verdict(raw)
        if verdict:
            break
    if not verdict:
        cand.verdict, cand.link = "error", ""      # never recorded (§4.1, correction c3)
        return cand
    cand.verdict, cand.link = verdict, link
    ledger.record(cand, verdict, link, now=now)
    return cand
