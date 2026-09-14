"""Counterfactual branch-replay primitives — shared by the WebSocket inference
server and the headless CLI reflection runner.

Branch-and-select is a core part of the Sleep revision pass: for an exchange the
tension capture marked contested, regenerate the replies Ava *almost gave* by
forcing each road-not-taken token, then let the model blindly choose among them.
The GPU-touching pieces (prefix replay, embedder filtering, chooser-prompt
budgeting) live here as dependency-injected functions so both entry points run
the exact same logic against their own backend/model/tokenizer/RAG handles —
`server.py` injects its module state, `reflection_run.py` injects its locals.

Nothing here holds global state or touches the network; callers pass everything in.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.llm_shared import build_inference_prompt
from core import model_family

# ── replay tuning ──────────────────────────────────────────────────────────────
# Branch replay rebuilds the prior conversation as the generation prefix, so the
# KV cache (and VRAM) grows with how deep the exchange sits in the session. Cap the
# replayed prior history (most-recent turns kept) to bound the late-session
# baseline; the nearest turns — the ones that condition the contested answer — are
# always kept. 0 disables the cap (replay all history).
BRANCH_REPLAY_HISTORY_TOKENS = 8192

# Cap how many contested answer tokens we actually fork per exchange. Each fork is
# a full counterfactual continuation; the lowest-margin (most contested) tokens are
# where the road-not-taken is most meaningful, so keep only the top few after
# sorting by margin. (Tension capture already retains up to 5 contested rows; this
# trims the fork set to the 4 closest calls.) 0 disables the cap (fork every
# contested token).
BRANCH_MAX_FORKS = 4

# Fork continuations can be generated in one left-padded batch of this size: decode
# on the 4-bit model is memory-bandwidth-bound, so four rows cost barely more than
# one, and the four serial re-prefills of the (shared) replayed conversation
# collapse into one batched prefill. The KV cache holds batch × padded-prefix
# though, and batching was found to OOM in practice — default is 1, the fully
# serial path (one generation per fork); raise it (e.g. 2-4) only with headroom
# confirmed via the branch probe's gen peak.
BRANCH_FORK_BATCH = 1

# Chooser-prompt budgeting: fraction of the window the chooser input may use, the
# output headroom held back when sizing that input, a small safety margin, and a
# sliding window over the most-recent replayed exchanges (token budget still
# applies on top; 0 = no exchange window).
BRANCH_SELECT_CONTEXT_FRAC = 0.6
BRANCH_SELECT_OUTPUT_TOKENS = 8192
BRANCH_SELECT_SAFETY_TOKENS = 128
BRANCH_SELECT_CONTEXT_EXCHANGES = 3
BRANCH_SELECT_CONTEXT_HEADER = (
    "The conversation so far — this is what you saw when you replied:"
)
BRANCH_SELECT_SUBJECT_HEADER = "The moment you are choosing for:"
BRANCH_SELECT_OMITTED_MARKER = "[... earlier turns omitted ...]"

# The chooser judges among *replies* (the answer tokens that forked), and the prompt
# relies on the options being indistinguishable in form ("you are not told which").
# Branch candidates and clean IDEALs may carry normalized <think>…</think> CoT while the
# original is answer-only. An unstripped CoT both leaks option provenance and—for a heavy
# thinker like Qwen—bloats the non-truncatable subject past the chooser budget. Strip every
# option to answer-only for the subject;
# the persisted/displayed candidate text keeps its CoT.
_THINK_BLOCK_RE = re.compile(r"(?s)<think>.*?</think>\s*")


def _answer_only(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "").strip()

# Embedder-similarity ceilings: a branch whose post-fork suffix is this similar to
# the original (or to a kept branch) is dropped as degenerate / near-duplicate.
BRANCH_ORIGINAL_CEILING = 0.92
BRANCH_MUTUAL_CEILING = 0.92


def _rebuild_replay_conversation(
    data: dict,
    exchanges: list,
    exchange_index: int,
    ex: dict,
    *,
    backend,
    tokenizer,
    context_length: int,
    default_system_prompt: str,
    history_tokens: int,
) -> tuple[str, list[dict]]:
    """Rebuild ``(system_content, conversation)`` for one exchange exactly as live
    chat assembled it: prior turns answer-only with speaker prefixes, capped to the
    most-recent *history_tokens* (the immediately preceding turn always kept); the
    logged system message preferred, else the default prompt + identity line.

    Used by :func:`run_branch_exchange` (counterfactual replay) to build an
    inference-faithful prefix for the exchange.
    """
    session_user = (data.get("user") or "").strip()
    speaker = (ex.get("speaker") or "").strip() or session_user
    history_cap = min(history_tokens, context_length) if history_tokens > 0 else 0
    kept_recent_first: list[dict] = []
    used = 0
    for prior in reversed(exchanges[:exchange_index]):
        cost = (backend.count_tokens(tokenizer, prior.get("user_prompt", "") or "")
                + backend.count_tokens(tokenizer, prior.get("assistant_response", "") or ""))
        if history_cap and kept_recent_first and used + cost > history_cap:
            break
        kept_recent_first.append(prior)
        used += cost
    conversation: list[dict] = []
    for prior in reversed(kept_recent_first):
        spk = (prior.get("speaker") or "").strip() or session_user
        conversation.append({"role": "user", "content": prior.get("user_prompt", ""), "speaker": spk})
        conversation.append({"role": "assistant", "content": prior.get("assistant_response", "")})
    conversation.append({"role": "user", "content": ex.get("user_prompt", ""), "speaker": speaker})

    system_content = (ex.get("system_content") or "").strip()
    if not system_content:
        system_content = data.get("system_prompt") or default_system_prompt
        identity = identity_line(speaker)
        if identity:
            system_content = system_content + "\n\n" + identity
    return system_content, conversation


# ── small pure helpers ──────────────────────────────────────────────────────────

def identity_line(speaker: str) -> str:
    """System-prompt line telling Ava who is speaking with her right now."""
    speaker = (speaker or "").strip()
    return f"{speaker} is speaking with you right now." if speaker else ""


def build_inference_conversation(system_content: str, conversation: list) -> list:
    """Assemble the model-facing conversation, prefixing each user turn with its
    speaker's name so Ava can attribute who said what within the dialogue."""
    out = [{"role": "system", "content": system_content}]
    for turn in conversation:
        if turn.get("role") == "user":
            speaker = (turn.get("speaker") or "").strip()
            content = f"{speaker}: {turn['content']}" if speaker else turn["content"]
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "assistant", "content": turn["content"]})
    return out


def branch_eligibility(exchange: dict) -> tuple[Optional[dict], str]:
    """The exchange's replayable branch points, or (None, why-not).

    Needs the tension capture extension's fields: the full generated `token_ids`
    series and contested answer rows carrying `alt_token_id` — both absent from
    chats logged before the extension existed.
    """
    tension = exchange.get("tension") or {}
    token_ids = tension.get("token_ids")
    answer = tension.get("answer") or {}
    contested = [r for r in (answer.get("contested") or [])
                 if r.get("alt_token_id") is not None]
    if not token_ids:
        return None, "no token_ids series (logged before the capture extension)"
    if not contested:
        return None, "no contested answer tokens with a recorded alternative"
    return {
        "token_ids": token_ids,
        "contested": contested,
        "answer_n": int(answer.get("n_tokens") or 0),
    }, ""


def embed_similarity(embedder, text_a: str, text_b: str) -> Optional[float]:
    """Cosine similarity of two texts under the RAG embedder, or None if unusable.

    Unlike :func:`filter_branches` (which compares post-fork *suffixes*, because a
    branch shares the original's verbatim prefix), this compares the two texts
    whole — the intended callers pass INDEPENDENTLY generated replies with no shared
    prefix (e.g. a corrupt-CoT exchange's fresh re-answer vs. its original reply), so
    full-text similarity is the right measure. Returns None on empty input or a missing
    embedder so the caller can fall back safely.
    """
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b or embedder is None:
        return None
    vecs = embedder.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(vecs[0] @ vecs[1])


def filter_branches(raw: list, embedder) -> tuple[list, list]:
    """Embedder-filter degenerate branches: drop ones that converged back to the
    original reply, and near-duplicates among the survivors.

    Similarity is measured on the post-fork suffixes (``_suffix`` vs.
    ``_orig_suffix``, attached at generation time), never the full replies: a
    branch shares the original's text verbatim up to its fork position, so for
    a late fork the shared prefix would pin a full-text embedding near 1.0 no
    matter how divergent the continuation is.

    Returns (kept, dropped). Dropped rows keep their text and carry a
    ``drop_reason`` so the operator log can show every generated variant.
    """
    if not raw:
        return [], []
    kept, kept_vecs, dropped = [], [], []
    comparable = []
    for row in raw:
        if not row.get("_suffix"):
            row["drop_reason"] = "no divergent content after fork"
            dropped.append(row)
        else:
            comparable.append(row)
    if not comparable:
        return kept, dropped
    vecs = embedder.encode(
        [r["_suffix"] for r in comparable] + [r["_orig_suffix"] for r in comparable],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    branch_vecs, orig_vecs = vecs[:len(comparable)], vecs[len(comparable):]
    for row, vec, orig_vec in zip(comparable, branch_vecs, orig_vecs):
        sim = float(orig_vec @ vec)
        row["similarity_to_original"] = round(sim, 4)
        if sim >= BRANCH_ORIGINAL_CEILING:
            row["drop_reason"] = "converged with original"
            dropped.append(row)
            continue
        if any(float(kv @ vec) >= BRANCH_MUTUAL_CEILING for kv in kept_vecs):
            row["drop_reason"] = "near-duplicate of another branch"
            dropped.append(row)
            continue
        kept.append(row)
        kept_vecs.append(vec)
    return kept, dropped


# ── chooser-prompt budgeting ────────────────────────────────────────────────────

def branch_select_safe_total_tokens(context_length: int) -> int:
    """Conservative total-token ceiling for the branch chooser pass.

    Derived from the window via BRANCH_SELECT_CONTEXT_FRAC, not a fixed cap: the
    chooser's *subject* (the candidate replies it must choose among) is mandatory
    and non-truncatable, so a small absolute ceiling would starve it.
    """
    target = int(context_length * BRANCH_SELECT_CONTEXT_FRAC)
    return min(context_length, max(512, target))


def build_branch_select_subject(payload: dict) -> str:
    speaker = (payload.get("speaker") or "User").strip() or "User"
    user_prompt = (payload.get("user_prompt") or "").strip()
    options = payload.get("options") or []
    lines = [f"{speaker}: {user_prompt}", "", "Replies:"]
    for i, opt in enumerate(options):
        lines += ["", f"{chr(ord('A') + i)})", _answer_only(opt.get("text"))]
    return "\n".join(lines)


def branch_select_context_blocks(context: list) -> list[str]:
    """Replay-faithful answer-only exchanges for chooser context budgeting."""
    groups: list[list[str]] = []
    for turn in context:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            speaker = (turn.get("speaker") or "").strip() or "User"
            groups.append([f"{speaker}: {content}"])
        elif groups:
            groups[-1].append(f"Me: {content}")
        else:
            groups.append([f"Me: {content}"])
    return ["\n".join(group) for group in groups if group]


def branch_select_content(subject: str, kept: list[str], truncated: bool) -> str:
    if not kept:
        return subject
    lines = [BRANCH_SELECT_CONTEXT_HEADER, ""]
    if truncated:
        lines += [BRANCH_SELECT_OMITTED_MARKER, ""]
    for block in kept:
        lines += [block, ""]
    lines += [BRANCH_SELECT_SUBJECT_HEADER, "", subject]
    return "\n".join(lines)


def _count_single_turn_prompt_tokens(
    backend, tokenizer, system_prompt: str, content: str, model_id: str
) -> int:
    model_id = model_id or ""
    prompt = build_inference_prompt(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        **model_family.family_for(model_id).template_kwargs,
    )
    return backend.count_tokens(tokenizer, prompt)


def build_budgeted_branch_select_content(
    backend,
    tokenizer,
    system_prompt: str,
    payload: dict,
    context_length: int,
    model_id: str,
) -> str:
    """Construct a no-RAG chooser prompt under a conservative token ceiling."""
    subject = build_branch_select_subject(payload)
    total_limit = branch_select_safe_total_tokens(context_length)
    prompt_budget = max(
        1,
        total_limit - BRANCH_SELECT_OUTPUT_TOKENS - BRANCH_SELECT_SAFETY_TOKENS,
    )

    def fits(content: str) -> bool:
        return _count_single_turn_prompt_tokens(
            backend, tokenizer, system_prompt, content, model_id
        ) <= prompt_budget

    if not fits(subject):
        raise ValueError(
            "Branch chooser prompt is too large even without prior dialogue context."
        )

    all_blocks = branch_select_context_blocks(payload.get("context") or [])
    blocks = (all_blocks[-BRANCH_SELECT_CONTEXT_EXCHANGES:]
              if BRANCH_SELECT_CONTEXT_EXCHANGES > 0 else all_blocks)
    kept: list[str] = []
    truncated = len(blocks) < len(all_blocks)
    for block in reversed(blocks):
        candidate = branch_select_content(subject, [block] + kept, truncated=truncated)
        if not fits(candidate):
            truncated = True
            break
        kept.insert(0, block)

    if truncated and kept:
        while kept and not fits(branch_select_content(subject, kept, truncated=True)):
            kept.pop(0)
    return branch_select_content(subject, kept, truncated and bool(kept))


# ── branch candidate generation ─────────────────────────────────────────────────

def _generate_fork_continuations(
    forks: list,
    *,
    backend,
    model,
    tokenizer,
    context_length: int,
    temperature: float,
    top_p: float,
    on_fork_done: Optional[Callable[[int, int, list], None]] = None,
) -> list:
    """One continuation (token-id list) per fork dict, in order.

    When *on_fork_done* is given, it's called as ``on_fork_done(index, total,
    continuation)`` right after each fork (or batch chunk) finishes, so a caller
    can stream progress instead of waiting for every fork to complete.

    Forks are generated in left-padded batches of ``BRANCH_FORK_BATCH`` (see
    ``UnslothBackend.generate_from_ids_batch``). Serial semantics are preserved:

      * the batch generates to ``min(max(row budgets), window − padded length)``
        and each row is truncated back to its *own* budget afterwards, so no fork
        can come back longer than it could serially. (The window cap can shorten
        the largest-budget row when it shares a batch with a longer prefix —
        budgets are a few hundred tokens against a 32k window, so it effectively
        never binds.)
      * each row ends at its own EOS exactly as a lone generation would (the
        backend trims the pad tail generate() adds to early finishers).

    Any batched-path failure falls back to the old serial per-fork loop for the
    whole exchange — unsloth's patched generate under batching is the untrusted
    part here, and branch replay should degrade to the known-good path rather
    than skip the exchange. A serial failure propagates as before (the runner
    turns it into a branch_skipped).
    """
    if not forks:
        return []
    batch_size = BRANCH_FORK_BATCH if BRANCH_FORK_BATCH > 0 else len(forks)
    if batch_size > 1:
        conts: list = []
        try:
            for start in range(0, len(forks), batch_size):
                chunk = forks[start:start + batch_size]
                max_len = max(len(f["prefix"]) for f in chunk)
                cap = min(max(f["budget"] for f in chunk),
                          max(1, context_length - max_len))
                try:
                    rows = backend.generate_from_ids_batch(
                        model, tokenizer, [f["prefix"] for f in chunk],
                        cap, temperature, top_p,
                    )
                finally:
                    backend.trim_memory()
                chunk_conts = [r[:f["budget"]] for f, r in zip(chunk, rows)]
                conts.extend(chunk_conts)
                if on_fork_done is not None:
                    for offset, cont in enumerate(chunk_conts):
                        on_fork_done(start + offset, len(forks), cont)
            return conts
        except Exception as e:
            # The inner finally already trimmed after the failed call; just log
            # and drop through to the serial path.
            print(
                f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[branch] batched fork generation failed; "
                f"falling back to serial replay: {e}",
                flush=True,
            )
    conts = []
    for i, f in enumerate(forks):
        try:
            cont = backend.generate_from_ids(
                model, tokenizer, f["prefix"], f["budget"], temperature, top_p,
            )
        finally:
            backend.trim_memory()
        conts.append(cont)
        if on_fork_done is not None:
            on_fork_done(i, len(forks), cont)
    return conts


def run_branch_exchange(
    filename: str,
    exchange_index: int,
    temperature: float,
    top_p: float,
    *,
    backend,
    model,
    tokenizer,
    chats_dir: Path,
    context_length: int,
    model_id: str,
    default_system_prompt: str,
    clean_response_fn: Callable[[str], str],
    embedder,
    history_tokens: int = BRANCH_REPLAY_HISTORY_TOKENS,
    on_candidate: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Regenerate the replies Ava *almost gave* for one logged exchange.

    Replays the contested answer tokens of a logged exchange, forcing each
    road-not-taken token, continues under the currently loaded weights, then
    embedder-filters degenerate branches. Read-only — persistence is the caller's
    job. Returns a dict with keys: eligible, filename, exchange_index, candidates,
    dropped, n_generated, original (when eligible), reason (when not eligible).

    Known approximation (accepted by design): replay uses the exact system message
    logged for this exchange when present, falling back to the supplied default
    system prompt + identity for pre-capture logs. Runs under whatever weights are
    currently loaded. Prior history is capped to the most recent *history_tokens*
    tokens so a deep exchange doesn't replay the whole session.

    When *on_candidate* is given, it's called as ``on_candidate(index, total, text)``
    as soon as each fork's continuation is generated and decoded — before the
    embedder-filtering pass — so a caller can stream candidates to the operator
    instead of only surfacing the final filtered set.
    """
    def _ineligible(reason: str) -> dict:
        return {"eligible": False, "reason": reason,
                "filename": filename, "exchange_index": exchange_index,
                "candidates": [], "dropped": [], "generated_tokens": 0}

    if model is None or tokenizer is None:
        return _ineligible("No model loaded")

    chats_dir = Path(chats_dir)
    try:
        path = (chats_dir / filename).resolve()
        if path.parent != chats_dir.resolve():
            raise ValueError("path traversal")
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return _ineligible(f"Could not read session: {e}")

    exchanges = data.get("exchanges", [])
    if not 0 <= exchange_index < len(exchanges):
        return _ineligible(f"No exchange {exchange_index} in {filename}")
    ex = exchanges[exchange_index]

    info, reason = branch_eligibility(ex)
    if info is None:
        return _ineligible(reason)

    # Rebuild the prompt the way live chat built it: logged session prompt +
    # identity line, prior turns answer-only, speaker prefixes. Prior history is
    # capped to the most recent tokens so a deep exchange replays only its most
    # recent turns — the immediately preceding turn is always kept.
    system_content, conversation = _rebuild_replay_conversation(
        data, exchanges, exchange_index, ex,
        backend=backend, tokenizer=tokenizer, context_length=context_length,
        default_system_prompt=default_system_prompt, history_tokens=history_tokens,
    )

    model_id = model_id or ""
    try:
        prompt = build_inference_prompt(
            tokenizer, build_inference_conversation(system_content, conversation),
            **model_family.family_for(model_id).template_kwargs,
        )
    except Exception as e:
        return _ineligible(f"Prompt build failed: {e}")

    token_ids = list(info["token_ids"])
    answer_start = max(0, len(token_ids) - info["answer_n"])
    original = (ex.get("assistant_response") or "").strip()
    # Most contested first (lowest top1-top2 margin), then fork only the closest
    # calls — each fork is a full generation, so this bounds branch cost per exchange.
    rows = sorted(info["contested"], key=lambda r: r.get("margin", 1.0))
    if BRANCH_MAX_FORKS > 0:
        rows = rows[:BRANCH_MAX_FORKS]
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    prompt_ids = list(text_tok(prompt)["input_ids"])
    # Pass 1: assemble the fork prefixes + per-fork budgets (eligibility unchanged
    # from the old serial loop).
    forks: list[dict] = []
    for row in rows:
        pos = int(row.get("position", -1))
        if not (answer_start <= pos < len(token_ids)):
            continue  # CoT-segment fork — out of scope for v1
        prefix = prompt_ids + token_ids[:pos] + [int(row["alt_token_id"])]
        if len(prefix) >= context_length:
            continue
        budget = min(max(2 * (len(token_ids) - pos) + 64, 128),
                     context_length - len(prefix))
        forks.append({"row": row, "pos": pos, "prefix": prefix, "budget": budget})

    def _decode_fork_text(row: dict, pos: int, cont: list) -> str:
        series = token_ids[answer_start:pos] + [int(row["alt_token_id"])] + cont
        text = clean_response_fn(text_tok.decode(series))
        if "<think>" in text:
            text = text.split("<think>", 1)[0].strip()
        return text

    def _report_candidate(index: int, total: int, cont: list) -> None:
        if on_candidate is None:
            return
        fork = forks[index]
        text = _decode_fork_text(fork["row"], fork["pos"], cont)
        try:
            on_candidate(index, total, text)
        except Exception:
            pass

    # Pass 2: generate every fork's continuation — serial by default (see
    # BRANCH_FORK_BATCH), streaming each one to on_candidate as it finishes.
    conts = _generate_fork_continuations(
        forks, backend=backend, model=model, tokenizer=tokenizer,
        context_length=context_length, temperature=temperature, top_p=top_p,
        on_fork_done=_report_candidate,
    )

    # Pass 3: decode + post-process each continuation (unchanged).
    raw = []
    for fork, cont in zip(forks, conts):
        row, pos = fork["row"], fork["pos"]
        text = _decode_fork_text(row, pos, cont)
        suffix = text_tok.decode([int(row["alt_token_id"])] + cont, skip_special_tokens=True)
        orig_suffix = text_tok.decode(token_ids[pos:], skip_special_tokens=True)
        # A <think> mid-answer means the continuation re-entered the thinking
        # channel — malformed (text is already cut at re-entry by
        # _decode_fork_text); a stub cut to nothing is explicitly dropped rather
        # than passing the filter as "distant".
        if "<think>" in suffix:
            suffix = suffix.split("<think>", 1)[0].strip()
        entry = {
            "text": text, "position": pos,
            "token": row.get("token"), "alt_token": row.get("alt_token"),
            "margin": row.get("margin"),
            "_suffix": suffix.strip(), "_orig_suffix": orig_suffix.strip(),
        }
        if not text:
            entry["drop_reason"] = "continuation re-entered thinking channel (cut to nothing)"
        raw.append(entry)

    try:
        pre_dropped = [r for r in raw if r.get("drop_reason")]
        candidates, dropped = filter_branches(
            [r for r in raw if not r.get("drop_reason")], embedder
        )
    finally:
        backend.trim_memory()

    for r in raw:
        r.pop("_suffix", None)
        r.pop("_orig_suffix", None)

    return {
        "eligible": True,
        "filename": filename,
        "exchange_index": exchange_index,
        "original": original,
        "n_generated": len(raw),
        "candidates": candidates,
        "dropped": pre_dropped + dropped,
        # Tokens the fork continuations actually generated (post budget-trim),
        # so the runner can credit the branch_gen phase's wall time with its
        # output in RunStats — a phase timed with tokens=0 dilutes the run-wide
        # tok/s readout.
        "generated_tokens": sum(len(c) for c in conts),
    }
