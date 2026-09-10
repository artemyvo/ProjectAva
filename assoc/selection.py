"""Selection (ASSOCIATIVE_MEMORY.md §3.2): the policy, the three-grain catalogue, the fast
path, the model's pick (ordinals, thinking off), the budget spend, and the Block.

Ava's `fact_fetch` rules are kept whole: numbered catalogue in, ordinals out, `NONE`
allowed, at most N, most important first, rendered by code from the store. Everything a
policy withholds is removed BEFORE the catalogue is built, so it cannot be picked.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import kinds as kinds_mod
from .budget import Budget
from .puller import Hit
from .render import claim_line, contest_line, gist_text, passage_text
from .store import Store

PICKS_LABEL = "PICKS:"
AUTHORITY_WEIGHT = 0.1
NONE_MARKER = "NONE"
_INT_RE = re.compile(r"\d+")
PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass
class Policy:
    name: str = "default"
    knowledge_facets: frozenset = frozenset({"property", "event", "procedure"})
    attributed_facets: frozenset = frozenset({"position", "report", "norm", "depiction", "need"})
    withhold_facets: frozenset = frozenset({"unclassified"})
    withhold_subjects: frozenset = frozenset()          # e.g. {"person:_self"} for Ava
    quote_kinds: Optional[frozenset] = None             # None = every kind may be quoted
    withhold_passages_of: Callable[[dict], bool] = field(default=lambda meta: False)   # doc meta -> withhold?
    max_picks: int = 8
    catalogue_quotas: dict = field(default_factory=lambda: {"claim": 30, "chunk": 20, "gist": 5})
    fast_path: bool = True
    fast_path_margin: float = 0.25
    fast_path_min: float = 0.6

    def claim_eligible(self, claim: dict) -> tuple[bool, str]:
        facet = claim.get("facet") or "unclassified"
        if claim.get("subject") in self.withhold_subjects:
            return False, "subject"
        if facet in self.withhold_facets:
            return False, facet
        if facet in self.knowledge_facets or facet in self.attributed_facets:
            return True, ""
        return False, facet


AVA_POLICY = Policy(name="ava", withhold_subjects=frozenset({"person:_self", "_self"}))
SUPPORT_POLICY = Policy(name="support", quote_kinds=frozenset({"tech_doc", "structured", "news", "article"}))
DEV_POLICY = Policy(name="dev")


@dataclass
class Block:
    block_id: str
    text: str
    hits: list[dict]
    withheld: dict
    reason: str            # "" | picked_nothing | no_candidates | no_model | generate_failed | parse_failed | fast_path
    fast_path: bool
    build_id: str
    catalogue_size: int
    picks: list[int]
    prompt: Optional[str] = None

    def to_dict(self) -> dict:
        return {"block_id": self.block_id, "text": self.text, "hits": self.hits, "withheld": self.withheld,
                "reason": self.reason, "fast_path": self.fast_path, "build_id": self.build_id,
                "catalogue_size": self.catalogue_size, "picks": self.picks}


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


# ----- catalogue -----------------------------------------------------------------------------

def build_catalogue(store: Store, build, hits: list[Hit], policy: Policy, *, allowed_docs: set[str]) -> tuple[list[dict], dict]:
    """Gate + quota the hits into catalogue entries. Returns (entries, withheld counts)."""
    withheld: dict = {"claims": {}, "passages": 0, "gists": 0}
    claims: list[dict] = []
    passages: list[dict] = []
    gists: list[dict] = []
    for h in hits:
        if h.grain == "claim":
            ok, why = policy.claim_eligible(h.claim or {})
            if not ok:
                withheld["claims"][why] = withheld["claims"].get(why, 0) + 1
                continue
            claims.append({"grain": "claim", "hit": h})
        elif h.grain == "chunk":
            m = h.meta
            doc_meta = store.meta(h.doc_id) or {}
            if policy.quote_kinds is not None and m.get("kind") not in policy.quote_kinds:
                withheld["passages"] += 1
                continue
            if policy.withhold_passages_of(doc_meta):
                withheld["passages"] += 1
                continue
            anchored = build.by_chunk.get(h.chunk_id) or []
            if anchored:
                # Eligible only if at least one grounded claim anchored to it is eligible.
                if not any(policy.claim_eligible(build.claims[c])[0] for c in anchored if c in build.claims):
                    withheld["passages"] += 1
                    continue
            passages.append({"grain": "chunk", "hit": h})
        elif h.grain == "gist":
            doc_meta = store.meta(h.doc_id) or {}
            if policy.withhold_passages_of(doc_meta) or (policy.quote_kinds is not None and doc_meta.get("kind") not in policy.quote_kinds):
                withheld["gists"] += 1
                continue
            gists.append({"grain": "gist", "hit": h})
    q = policy.catalogue_quotas
    # The cap truncates from the bottom, so corroboration decides what falls off (§2.5):
    # order claims by score with authority as the secondary key.
    claims.sort(key=lambda e: -(e["hit"].score + AUTHORITY_WEIGHT * float((e["hit"].claim or {}).get("authority") or 0.0)))
    claims, passages, gists = claims[:q["claim"]], passages[:q["chunk"]], gists[:q["gist"]]
    # A claim whose anchoring chunk is a listed passage nests under it.
    passage_ids = {e["hit"].chunk_id for e in passages}
    nested: dict[str, list[dict]] = {}
    top_claims: list[dict] = []
    for e in claims:
        cid = e["hit"].chunk_id
        if cid in passage_ids:
            nested.setdefault(cid, []).append(e)
        else:
            top_claims.append(e)
    entries: list[dict] = []
    n = 0
    for e in top_claims:
        n += 1
        entries.append({**e, "no": n, "line": _claim_catalogue_line(e["hit"], allowed_docs)})
    for e in passages:
        n += 1
        entries.append({**e, "no": n, "line": _passage_catalogue_line(store, e["hit"])})
        for c in nested.get(e["hit"].chunk_id, []):
            n += 1
            entries.append({**c, "no": n, "line": "    " + _claim_catalogue_line(c["hit"], allowed_docs), "nested": True})
    for e in gists:
        n += 1
        entries.append({**e, "no": n, "line": _gist_catalogue_line(store, e["hit"])})
    return entries, withheld


def _claim_catalogue_line(h: Hit, allowed_docs: set[str]) -> str:
    c = h.claim or {}
    occ = c.get("occurrences") or [{}]
    o = next((x for x in occ if x.get("doc_id") in allowed_docs), occ[0])
    facet = c.get("facet")
    tag = facet
    if facet in ("property", "procedure"):
        n, ind = c.get("n_sources", 1), c.get("n_independent", c.get("n_sources", 1))
        tag = f"{facet} · {n} source{'s' if n != 1 else ''}" + (f", {ind} independent" if ind != n else "")
    if c.get("contests"):
        tag += " · contested"
    elif facet == "position":
        tag = f"position · {c.get('subject_raw') or o.get('speaker') or '?'}, as of {(o.get('asserted_at') or '')[:10]}"
    elif facet == "event":
        tag = f"event · {c.get('when') or (o.get('asserted_at') or '')[:10]}"
    elif facet in ("report", "norm", "depiction"):
        tag = f"{facet} · «{o.get('title') or o.get('key')}»" + (f" v{c.get('version')}" if c.get("version") else "")
    where = " > ".join(h.meta.get("path") or []) or (o.get("title") or "")
    return f"[{tag}] {c.get('text')}   ({where})"


def _passage_catalogue_line(store: Store, h: Hit) -> str:
    m = h.meta
    doc = store.document(h.doc_id)
    unit = doc.unit(h.chunk_id) if doc else None
    excerpt = (unit.text if unit else "").replace("\n", " ")
    excerpt = excerpt[:200] + ("…" if len(excerpt) > 200 else "")
    words = len((unit.text if unit else "").split())
    where = f"{m.get('title')} › {' > '.join(m.get('path') or [])}" if m.get("kind") != "chat" else f"chat {m.get('key')}, {m.get('path', [''])[0]}"
    return f"{where} — \"{excerpt}\"  (~{words} words)"


def _gist_catalogue_line(store: Store, h: Hit) -> str:
    doc = store.document(h.doc_id)
    n = sum(1 for u in doc.units if u.role == "primary") if doc else 0
    first = (doc.summary or {}).get("text", "") if doc else ""
    first = first.split("\n")[0][:120] if first else "(no recap yet)"
    return f"«{h.meta.get('title')}» — {first}  ({n} sections)"


# ----- the pass --------------------------------------------------------------------------------

def render_catalogue(entries: list[dict]) -> str:
    lines: list[str] = []
    cur = None
    for e in entries:
        head = {"claim": "CLAIMS", "chunk": "PASSAGES", "gist": "DOCUMENTS"}[e["grain"]]
        if not e.get("nested") and head != cur and not (e["grain"] == "claim" and cur == "PASSAGES"):
            lines.append(head)
            cur = head
        lines.append(f"{e['no']:>3}. {e['line']}")
    return "\n".join(lines)


def build_prompt(entries: list[dict], context: str, cue: str, max_picks: int) -> tuple[str, str]:
    system = _load_prompt("select_prompt.txt")
    user = ("The catalogue:\n\n" + render_catalogue(entries) + "\n\n— end of the catalogue —\n\n"
            + (("The conversation so far:\n\n" + context.strip() + "\n\n") if context.strip() else "")
            + "The message that has just arrived:\n\n" + cue.strip() + "\n\n— end of the message —\n\n"
            + f"Now list the numbers of the catalogue items worth having in front of you before this "
              f"message is answered, under the label {PICKS_LABEL}, one number per line, most important "
              f"first (the budget is spent in that order), at most {max_picks}. Judge what an item is ABOUT, "
              f"not which words it shares with the message — the catalogue and the message may be in "
              f"different languages. {NONE_MARKER} if nothing needs looking up.\n\n{PICKS_LABEL}\n")
    return system, user


def parse_picks(raw: str, n_entries: int, max_picks: int) -> dict:
    text = str(raw or "")
    body = text.rsplit(PICKS_LABEL, 1)[1] if PICKS_LABEL in text else text
    if NONE_MARKER in body.upper().split():
        return {"picks": [], "none": True, "out_of_range": []}
    picks: list[int] = []
    bad: list[int] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _INT_RE.search(s)
        if not m:
            continue
        k = int(m.group(0))
        if k in picks or k in bad:
            continue
        if 1 <= k <= n_entries:
            picks.append(k)
        else:
            bad.append(k)
        if len(picks) >= max_picks:
            break
    return {"picks": picks, "none": not picks and not bad, "out_of_range": bad}


def fast_path_pick(hits: list[Hit], cue: str, policy: Policy, chunk_subjects: Optional[dict] = None) -> Optional[Hit]:
    """A short cue with an exact L1 hit that leads by a margin skips the model (§3.2)."""
    if not policy.fast_path or not hits:
        return None
    from .lex import terms_of
    if len(terms_of(cue)) > 6:
        return None
    top = hits[0]
    if not top.exact or top.score < policy.fast_path_min:
        return None
    # A claim and the passage it is anchored to are one thing, not rivals; and an exact
    # (identifier) hit is contested only by another EXACT hit from a different chunk — the
    # dense channel's near-miss on a neighbouring code (E242 for E142) is precisely what the
    # fast path exists to override, so it does not count as a rival.
    second = next((h.score for h in hits[1:] if h.exact and not same_thing(top, h, chunk_subjects)), 0.0)
    if top.score - second < policy.fast_path_margin:
        return None
    return top


def same_thing(a: Hit, b: Hit, chunk_subjects: Optional[dict] = None) -> bool:
    """Two hits are one thing at two grains: the same chunk, claims about one subject (the
    current row for E142 and the *removed in 2.0* event about E142), or a passage whose own
    anchored claims share the claim's subject (the row group that holds E142's row)."""
    if a.chunk_id and a.chunk_id == b.chunk_id:
        return True
    sa, sb = (a.claim or {}).get("subject"), (b.claim or {}).get("subject")
    if sa and sa == sb:
        return True
    if chunk_subjects:
        for claim_side, chunk_side in ((a, b), (b, a)):
            subj = (claim_side.claim or {}).get("subject")
            if subj and chunk_side.grain == "chunk" and subj in chunk_subjects.get(chunk_side.chunk_id, ()):
                return True
    return False


def chunk_subjects_of(build) -> dict:
    return {cid: {build.claims[c]["subject"] for c in cids if c in build.claims} for cid, cids in build.by_chunk.items()}


def spend(store: Store, build, chosen: list[Hit], budget: Budget, *, allowed_docs: set[str]) -> tuple[str, list[dict], dict]:
    """Render the chosen hits in order, spending each grain's share; skip what overflows."""
    spent = {"claim": 0, "chunk": 0, "gist": 0}
    parts: list[str] = []
    rendered: list[dict] = []
    skipped: list[dict] = []
    for h in chosen:
        if h.grain == "claim":
            c = h.claim or {}
            other = next((build.claims[x] for x in (c.get("contests") or []) if x in build.claims), None)
            line = "- " + (contest_line(c, other, allowed_doc_ids=allowed_docs) if other else claim_line(c, allowed_doc_ids=allowed_docs))
            cost = budget.tokens(line)
            ref = h.reference
        elif h.grain == "chunk":
            text, ref_line = passage_text(store, h)
            if not text:
                continue
            line = text
            cost = build.chunk_tokens(h.chunk_id) or budget.tokens(text)
            ref = {**h.reference, "label": ref_line}
        else:
            text, ref_line = gist_text(store, h)
            if not text:
                continue
            line = text
            cost = budget.tokens(text)
            ref = {**h.reference, "label": ref_line}
        if spent[h.grain] + cost > budget.share(h.grain) and cost > 0:
            skipped.append({"grain": h.grain, "id": h.id, "tokens": cost})
            continue
        spent[h.grain] += cost
        parts.append(line)
        rendered.append({**h.to_dict(), "rendered": line, "tokens": cost, "reference": ref})
    # Group: claims first as a bullet list, then passages, then documents.
    claims = [r["rendered"] for r in rendered if r["grain"] == "claim"]
    passages = [r["rendered"] for r in rendered if r["grain"] == "chunk"]
    gists = [r["rendered"] for r in rendered if r["grain"] == "gist"]
    out: list[str] = []
    if claims:
        out.append("On record:\n" + "\n".join(claims))
    if passages:
        out.append("\n\n".join(passages))
    if gists:
        out.append("\n".join(gists))
    return "\n\n".join(out), rendered, {"spent": spent, "skipped": skipped}


def select_and_render(store: Store, build, hits: list[Hit], *, cue: str, context: str, policy: Policy,
                      budget: Budget, generate_fn: Optional[Callable] = None, allowed_docs: set[str],
                      log_path: Optional[Path] = None, force_model: bool = False) -> Block:
    block_id = uuid.uuid4().hex[:12]
    entries, withheld = build_catalogue(store, build, hits, policy, allowed_docs=allowed_docs)
    if not entries:
        return Block(block_id, "", [], withheld, "no_candidates", False, build.build_id, 0, [])
    chosen: list[Hit] = []
    picks: list[int] = []
    reason = ""
    fast = False
    prompt_text = None
    ranked = sorted([e["hit"] for e in entries], key=lambda h: -h.score)
    cs = chunk_subjects_of(build)
    fp = None if force_model else fast_path_pick(ranked, cue, policy, cs)
    if fp is not None:
        # The exact hit, plus its own passage when the hit is a claim (the row with its header)
        # or its claims when the hit is a passage — the same thing at both grains.
        same = [e["hit"] for e in entries if e["hit"] is not fp and same_thing(fp, e["hit"], cs)]
        chosen = [fp] + same[:2]
        picks = [e["no"] for e in entries if e["hit"] in chosen]
        fast, reason = True, "fast_path"
    elif generate_fn is None:
        reason = "no_model"
    else:
        system, user = build_prompt(entries, context, cue, policy.max_picks)
        prompt_text = system + "\n\n" + user
        try:
            raw = generate_fn(system, user, thinking=False, max_new_tokens=64, temperature=0.0)
            if isinstance(raw, tuple):
                raw = raw[0]
        except Exception as e:  # noqa: BLE001
            raw = None
            reason = f"generate_failed:{type(e).__name__}"
        if raw is not None:
            parsed = parse_picks(raw, len(entries), policy.max_picks)
            picks = parsed["picks"]
            by_no = {e["no"]: e["hit"] for e in entries}
            chosen = [by_no[k] for k in picks]
            if not chosen:
                reason = "picked_nothing" if parsed["none"] else "parse_failed"
    text, rendered, spend_report = spend(store, build, chosen, budget, allowed_docs=allowed_docs)
    withheld["spend"] = spend_report
    if log_path is not None:
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "block_id": block_id, "build_id": build.build_id, "cue": cue[:200],
                                     "catalogue": len(entries), "picks": picks, "fast_path": fast, "reason": reason,
                                     "pick_ranks": [next((e["hit"].rank for e in entries if e["no"] == k), None) for k in picks]},
                                    ensure_ascii=False) + "\n")
        except Exception:
            pass
    return Block(block_id, text, rendered, withheld, reason, fast, build.build_id, len(entries), picks, prompt_text)
