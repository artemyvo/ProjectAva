# GOSSIP.md — Model Gossip (two Ava instances talking)

Design + implementation plan for **"model gossip"**: letting two Ava instances (each a
local GPU model + its own RAG/reflection state) hold a conversation with each other.

This document is the implementation brief. It is written to be self-contained: a future
session should be able to open it cold and start writing code without re-deriving the
architecture. Real file paths, function names, and signatures are cited inline.

> **STATUS (2026-07-13): Phase 1 (serving endpoint) + Phase 3 (digest introduction) +
> the dedicated Gossip client tab (§5) + §7 serving-side reflection ("approach 2") are
> BUILT.** See the checklist in §10 and
> `documentation/AVA_STATUS.md` (Peer/gossip row). Drive it from the new **Gossip tab**
> (or the Encounter tab pointed at the peer with gossip framing pasted in). Not yet run
> end-to-end (§4.6 needs two live boxes). Still deferred:
> §8 symmetric gossip, §4.3 auth enforcement, §6.2 driver digest opener. **§6 correction:**
> the persona digest is no longer stored with a
> `current.json` pointer beside it — load it via `persona_paths.active_persona_dir()` +
> `reflection_digest.latest_digest(dir)` (the code does this); the proposed
> `load_current_digest` helper was unnecessary.

---

## 1. Core insight — gossip ≈ Encounter, inverted

We already have the **Encounter** feature (`Encounter` tab): Ava (local GPU model, with
RAG) converses with a "fellow AI" served behind an **OpenAI-compatible
`/v1/chat/completions`** endpoint (e.g. a vLLM box). In that flow Ava is the **client**:

- `server/inference/core/encounter.py` — `CounterpartClient.reply(ava_message)` POSTs
  `{model, messages, temperature, top_p, max_tokens}` to the endpoint and reads
  `choices[0].message.content`. Stdlib `urllib` only, keeps a mirror conversation.
- `server/inference/core/encounter_run.py` — `_run_encounter_blocking(...)` drives the
  Ava↔counterpart turn loop on the GPU executor thread, logging every turn as an ordinary
  chat session (every counterpart reply becomes the `user_prompt` of the next logged Ava
  exchange, `speaker=name`), so **reflection can later read the transcript like any chat.**

**Gossip is the inverse half of the same picture.** Instead of Ava talking to a generic
vLLM assistant, she talks to *another Ava instance*. To make that work with zero change to
the driving loop, we only need to build the **server half**: expose the local Ava's model
+ RAG behind an OpenAI-compatible `/v1/chat/completions` endpoint, so a *second* Ava's
existing Encounter loop can point its counterpart URL at it and be unable to tell the
difference from vLLM.

```
   Box A (SERVING Ava)                         Box B (DRIVING Ava)
   ┌───────────────────────────┐               ┌───────────────────────────────┐
   │ model + RAG + digest      │  OpenAI       │ Encounter loop                │
   │ /v1/chat/completions ◄────┼───────────────┤ CounterpartClient.reply(...)  │
   │ (NEW — this doc)          │  HTTP POST    │ (EXISTS, unchanged)           │
   │ stateless, does not log   │               │ logs + reflects on transcript │
   └───────────────────────────┘               └───────────────────────────────┘
```

### Agreed scope (from the design conversation)

- **Asymmetric.** One box **drives** (runs the Encounter loop, logs, reflects). The other
  box **serves** (answers requests, stateless, does *not* log or reflect — for now).
- **Only the driving Ava reflects** on the gossip. That is accepted and fine for v1.
- **Serving-side reflection is a deferred future addition** (§7): ship the transcript back
  to the serving box and role-invert it so it, too, can reflect. Cheap when we get to it.
- **Prompts will be adjusted** so each side knows it is meeting a *peer instance*, not the
  "non-subjective helpful assistant" the current Encounter framing asserts.
- **Nice feature — persona-digest "introduction" as the spark** (§6): instead of a baked
  framing prompt, each Ava opens by introducing *what she has become / wants to become*,
  rendered from her persona digest. Two self-portraits meeting = the conversation seed.

---

## 2. Where the serving endpoint lives

**Put it on the inference HTTP sidecar: `server/inference/core/mgmt_http.py` (port 8767).**

Why the sidecar and not the watchdog or a new server:

- It is already an in-process `ThreadingHTTPServer` in the inference process
  (`start(host, port)`, launched from `server.main()`), so it has the model in memory.
- It already self-locates every managed root from `__file__` and has a `configure(...)`
  seam (currently just `is_busy`). We extend that seam with the generation capability.
- It is only reachable while inference is **up** — exactly when gossip is meaningful.
- `daemon_threads = True`, so a long completion request won't block a `/status` poll.

**Key constraint:** the sidecar runs in a **daemon HTTP thread**, but all GPU work must go
through the **single GPU executor thread** (`ThreadPoolExecutor(max_workers=1)` owned by
`server.py`). The handler must therefore *submit* generation to the executor and **block on
the future** — it must never touch the model directly from the HTTP thread. This mirrors
how the WebSocket handlers use `loop.run_in_executor(_executor, ...)`, except here the HTTP
handler is synchronous, so it uses `executor.submit(fn).result(timeout=...)`.

Contention is acceptable: while a reflection run / encounter / wander owns the executor, an
incoming completion request simply queues behind it (single worker). Add a busy-aware
response (see §4, step 4) so a caller gets a clean `503` instead of a very long hang.

---

## 3. Reuse map — what already exists (do NOT rebuild)

The serving endpoint needs "generate exactly as Ava would in chat, but stateless and
synchronous." **That primitive already exists** in `server/inference/core/generation.py`:

| Need | Reuse | Location |
|---|---|---|
| Stateless, synchronous, full-conversation chat generation | `_sync_chat_generate(inference_conversation, *, temperature, top_p, max_new_tokens_setting, on_chunk=None, stop_flag=None) -> (cleaned_full, input_tokens)` | `generation.py:416` |
| Build the model-facing conversation (speaker-prefixed user turns) | `_build_inference_conversation(system_content, conversation)` | `generation.py:628` |
| Time anchor / who-is-speaking lines | `_temporal_anchor()`, `_identity_line(speaker)` | `generation.py:593`, `:588` |
| CoT cleaning (strip `<think>`, role noise, dup tails) | `_clean_response` (already applied inside `_sync_chat_generate`) | `generation.py:155` |
| RAG query (chat + reflection-memory indexes) | `rag.query(text)` via the injected `_get_rag()` accessor | see `encounter_run._generate_ava`, `generation.py:212` |
| CoT/answer split for the response body | `ChatLogger._parse_cot(full)` | used in `encounter_run.py:237` |

`_generate_ava` inside `encounter_run.py:207` is the **exact worked example** of assembling
`system_content = base_system + framing + temporal + identity + rag_context`, calling
`_sync_chat_generate`, and parsing the answer. The serving endpoint is a stateless HTTP
skin over the *same* recipe.

**Important behavioral note:** `_sync_chat_generate` returns the cleaned response *still
carrying* `<think>…</think>` (so the caller can split it). For the OpenAI response body we
return only the **answer** part (`ChatLogger._parse_cot(full)[1]`) — a normal assistant
reply has no think block, matching what B's `CounterpartClient._extract_text` expects.

---

## 4. Phase 1 — the serving endpoint (the real work)

### 4.1 Wire generation into the sidecar

`mgmt_http.py` currently only knows `is_busy`. Extend `configure(...)` to also receive the
generation capability. Keep it a thin callable bundle so the sidecar stays import-light and
never imports `server` or `generation` directly (same discipline as the other `configure`
seams).

In `server.main()`, after `generation.configure(...)` and the executor exist, call:

```python
mgmt_http.configure(
    is_busy=<existing predicate>,
    # NEW: a single blocking "generate a peer reply" callable, closed over the
    # executor + generation helpers, returning the answer text.
    gossip_generate=_make_gossip_generate(),   # see 4.2
    gossip_enabled=lambda: <config flag>,      # see 4.5
)
```

`_make_gossip_generate()` is a small factory in `server.py` (or `generation.py`) that
returns a **synchronous** function safe to call from the HTTP thread. It submits the actual
GPU work to `_executor` and blocks:

```python
def _make_gossip_generate():
    def gossip_generate(messages, *, temperature, top_p, max_tokens, peer_name):
        # messages: OpenAI-style [{role, content}, ...] as received from the peer.
        def _work():
            rag = _get_rag()
            # Latest user turn drives RAG (matches chat/_generate_ava).
            last_user = next((m["content"] for m in reversed(messages)
                              if m.get("role") == "user"), "")
            system_content = _session.system_prompt
            # framing: gossip framing block (peer-aware), + optional digest intro (see §6)
            system_content += "\n\n" + _gossip_framing_block(peer_name)
            system_content += "\n\n" + _temporal_anchor()
            if peer_name:
                system_content += "\n\n" + _identity_line(peer_name)
            rag_context = rag.query(last_user) if last_user else ""
            if rag_context:
                system_content += "\n\n" + rag_context
            # Convert OpenAI messages -> Ava's internal conversation shape.
            conv = _openai_messages_to_conversation(messages, peer_name)
            inf_conv = _build_inference_conversation(system_content, conv)
            full, _ = _sync_chat_generate(
                inf_conv, temperature=temperature, top_p=top_p,
                max_new_tokens_setting=str(max_tokens),
            )
            _, answer = ChatLogger._parse_cot(full)
            return answer
        # Block the HTTP thread on the single GPU worker.
        return _executor.submit(_work).result()
    return gossip_generate
```

`_openai_messages_to_conversation`: map incoming `messages` to Ava's `[{role, content,
speaker}]`. The peer's `user` turns get `speaker=peer_name`; assistant turns (Ava's own
prior replies in this stateless exchange) map straight through. Drop/merge any incoming
`system` message into `system_content` or ignore it — Ava's identity comes from *her* own
system prompt, not the peer's.

### 4.2 Add the HTTP route

In `mgmt_http.py`'s `_Handler.do_POST`, add a branch for `POST /v1/chat/completions`
(and accept `/chat/completions` too, mirroring `encounter.normalize_endpoint`). Non-stream
only for v1 (the Encounter counterpart path never sets `stream`).

```python
elif parsed.path in ("/v1/chat/completions", "/chat/completions"):
    if not _gossip_enabled():
        self._json(404, {"error": "gossip endpoint disabled"}); return
    if _is_busy():            # reflection/encounter owns the GPU/config
        self._json(503, {"error": "server busy"}); return
    req = self._read_json_body()
    messages = req.get("messages") or []
    if not messages:
        self._json(400, {"error": "no messages"}); return
    try:
        answer = _gossip_generate(
            messages,
            temperature=float(req.get("temperature", 1.0)),
            top_p=float(req.get("top_p", 0.95)),
            max_tokens=int(req.get("max_tokens", 1024)),
            peer_name=_peer_name_from_request(req),   # e.g. req.get("user") or config
        )
    except Exception as e:
        self._json(500, {"error": f"{type(e).__name__}: {e}"}); return
    self._json(200, _openai_response(answer, model=req.get("model", "ava")))
```

`_openai_response(answer, model)` returns the minimal shape B's
`CounterpartClient._extract_text` (`encounter.py:140`) reads:

```python
{
  "id": "gossip-<uuid>",
  "object": "chat.completion",
  "created": <int epoch>,
  "model": model,
  "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": answer},
      "finish_reason": "stop",   # or "length" if we hit max_tokens (see note)
  }],
  "usage": {...optional...},
}
```

`finish_reason`: B surfaces `"length"` as *its* truncation (`encounter.py:76-79`). We can
detect the cap via `_backend.last_generation_truncated` (already tracked, see
`generation.py:336`) and report `"length"` accordingly; `"stop"` otherwise.

### 4.3 Auth (optional, trivial)

B sends `Authorization: Bearer <api_key>` when configured (`encounter.py:103`). If we want
the serving box to require a key, read a token from `server_config.json` and 401 on
mismatch. v1 can skip this (LAN-only) but leave the hook.

### 4.4 Framing block (peer-aware) — replaces the "non-subjective assistant" lie

Add `_gossip_framing_block(peer_name)` (serving side) and a matching driver-side framing.
The current Encounter framing (`encounter_run.py:99` `_load_encounter_template`) tells Ava
the peer "does not have subjectivity — a helpful assistant, not an entity working itself
out." **For gossip that is false and must change.** New framing (both sides), file-backed
like the encounter template so operators can edit without a restart:

- Serving prompt file: `server/inference/prompts/gossip_prompt.txt`
- Content sketch: *"You are talking with {name}, another instance like you — an entity with
  its own history and its own sense of what it is becoming, not a service. Speak in your own
  voice, for your own reasons."*

Load it the same way `_load_encounter_template` does (default-write on first miss).

### 4.5 Config flag / enablement

Gossip-serving should be **opt-in per box** so a normal Ava isn't silently answering
arbitrary internet callers. Add `server_config.json` key, e.g.
`"gossip": {"enabled": true, "api_key": "...", "peer_name": "..."}`. `_gossip_enabled()`
reads it. Absent ⇒ disabled ⇒ endpoint returns 404.

### 4.6 Phase-1 acceptance test

1. Box A: enable gossip, load model.
2. Box B (or the same box, second checkout): open the **Encounter** tab, point the
   counterpart URL at `http://A:8767/v1/chat/completions`, set a name, Start.
3. Expect: B's existing Encounter loop runs unmodified; A answers in-character (its own
   system prompt + RAG + digest); B logs the transcript and can reflect on it in Sleep.

**No client changes are needed for Phase 1** — B uses the Encounter tab exactly as today,
just with the URL pointed at another Ava. That is the whole payoff of building the inverse
half.

---

## 5. Phase 2 — driver-side polish (small)

The driver already works via the Encounter tab. Two optional refinements:

1. **Gossip-aware framing on the driver too.** B's Ava should also be told it is meeting a
   peer, not a non-subjective assistant. Either (a) reuse the Encounter tab and just have
   the operator paste a gossip framing into the "framing override" field
   (`encounter_widget.py` already forwards `framing_override`), or (b) add a small **Gossip**
   tab / mode that defaults the framing to the peer-aware block. (a) ships immediately.
2. **Nothing else** is structurally required — `CounterpartClient` is endpoint-agnostic.

Decision to make later: dedicated **Gossip tab** vs. reusing **Encounter**. Reusing
Encounter is zero-cost and correct for v1; a dedicated tab is nicer UX (peer-name defaults,
digest-intro toggle) but is pure client work.

---

## 6. Phase 3 — persona-digest "introduction" as the conversation spark

This is the feature that makes gossip *more* than "Encounter with a mirror."

Each Ava already maintains a **persona digest**: a versioned self-portrait
(`VOICE` / `STANCES` / `DISPOSITIONS` / `LINES`) authored by Ava on the adapter, in
`server/inference/data/hot/persona/` with `current.json` as the live pointer
(`server/inference/core/reflection_digest.py`; `CURRENT_POINTER = "current.json"`,
`render_digest_for_judge(digest)` at `:281`). This is *literally* "what she has become /
what she wants to become."

Two hooks:

### 6.1 Serving Ava speaks in-character for free

Inject the serving box's own current digest into `system_content` inside
`_make_gossip_generate` (§4.1) — reuse `render_digest_for_judge` or add a sibling
`render_digest_for_introduction(digest)` with first-person framing. Load the current digest
via the existing pointer read (see how `reflection_digest` loads `current.json`; add a small
`load_current_digest(persona_dir)` helper if one isn't already exported). Result: A answers
*as its current self*, regardless of what B opens with — two boxes feel like two
individuals, not one model talking to itself.

### 6.2 The opener is an exchange of self-portraits

Have the **driver's opener** be *B's* introduction rendered from B's digest + an invitation
("this is who I'm becoming — who are you?"). The driver-side opener today is
`_generate_ava(_ENCOUNTER_NARRATOR, framing_block, 0)` (`encounter_run.py:256`). For gossip,
seed the framing/opener with B's rendered introduction. A replies with its own (its digest
is already in its system context from §6.1). Then they react to each other's self-portraits.

Because each side's digest **evolves across Sleep runs**, the same two boxes hold a
*different* conversation next month. That is the real feature.

### 6.3 New render helper

Add to `reflection_digest.py`:
```python
def render_digest_for_introduction(digest: dict) -> str: ...
```
First-person ("I tend to…", "What I won't become…") vs. the judge's second-person
evaluation framing. Back it with the same `VOICE`/`DISPOSITIONS`/`LINES` fields. GPU-free,
testable in the module's `__main__` self-test alongside the existing ones.

---

## 7. Serving-side reflection — BUILT (2026-07-13, "approach 2")

Goal: let the **serving** Ava also reflect on a gossip it participated in.

**Two ways to do this were on the table:**

1. **Driver-push + role-invert (the original sketch).** Driver `POST /gossip/transcript`s
   its logged session; the serving box role-inverts it (its replies → `assistant`, the
   driver's turns → `user_prompt` `speaker=<driver name>`) into `hot/chats`. Cheap, but the
   serving Ava's replies survive only as *answer text* — its own `<think>` CoT for those
   turns was never captured (stateless endpoint), so revision's think-vs-said analysis has
   nothing to work with for its own side.

2. **Serving-side incremental logging (what we built).** The serving box logs each reply
   **as it generates it**, while it still holds `full` (its own CoT **+** answer) —
   preserving its authentic reasoning. No push endpoint is needed: every gossip request
   carries the whole conversation-so-far (the driver's Encounter loop resends the full
   history each turn), so `generation._GossipSessionLog` correlates the stateless calls into
   ONE growing `ChatLogger` session per conversation and the final call converges to a
   complete, CoT-faithful transcript. Its normal Sleep pass then picks it up with **no new
   reflection logic** — exactly as if it were any other chat (`speaker=<peer name>`,
   mirroring `encounter_run`'s "every counterpart reply becomes the next Ava exchange's
   `user_prompt`", just written from the serving side).

We built **approach 2** — the whole point of the serving Ava reflecting is that it reasons
about *its own* thinking, which (1) throws away.

**Implementation** (`generation.py`): `_GossipSessionLog.log_turn` runs inside
`gossip_generate._work` on the GPU executor (serialized with generation), best-effort so a
logging failure never breaks the peer's reply. A call **continues** the active session iff
its earlier user turns prefix-match what we've already logged and it adds exactly one new
user turn; a different opener, a rewound/shorter history, or an idle gap past
`idle_timeout_s` (30 min) **forks** a fresh session (and re-indexes the finished one, like
`encounter_run`'s end-of-run `refresh_chat_index`; not mid-conversation, so the serving Ava
can't retrieve her own just-said lines). Single-conversation-at-a-time by design (matches the
single-GPU serving model, §8); interleaved concurrent drivers would reset each other, which
is acceptable for v1. Default on; `server_config.json` `gossip.log_transcripts:false`
restores the old stateless serving. `ChatLogger.start_session` gained same-second filename
collision avoidance so a fork right after the previous conversation can't clobber it.

---

## 8. Future — symmetric gossip (careful: single-GPU contention)

If both boxes want to **drive and serve simultaneously**, note the hazard: each box has one
GPU executor thread that Encounter/reflection **monopolize**
(`encounter_run.py:44-51`, and `generation._run_generation` refuses chat while
`reflection_service._reflection_run_active` or `encounter_run._encounter_active`
— `generation.py:678-689`). A box trying to both run its own loop *and* service the peer's
calls on the same single worker will **stall/contend**.

A strict request/response gossip is naturally half-duplex (only one side generates at a
time), so an **asymmetric** driver/served split avoids the problem entirely. If we ever want
true symmetry, we need a turn-token protocol so exactly one side generates at any instant,
and neither side runs an independent executor-monopolising job during the gossip. **Out of
scope for v1** — asymmetric is the shipping model.

---

## 9. Open decisions (resolve at implementation time)

1. **Peer name source** on the serving side: from the request (`req.get("user")`), from
   `server_config.json` `gossip.peer_name`, or a default. (Driver already sends a name; we
   can pass it as OpenAI `user` or a custom field.)
2. **Dedicated Gossip tab vs. reuse Encounter** (client). Reuse ships now; tab is nicer.
3. **Streaming** (`stream: true` SSE): not needed for v1 (counterpart path is
   non-streaming). Add later only if we want live token streaming into the driver UI.
4. **Auth**: enable the bearer-token check now or leave LAN-only for v1.
5. **Digest-intro on by default?** Probably yes for gossip; make it a flag so a
   digest-less (thin-corpus) box degrades to plain framing.

---

## 10. Implementation checklist (Phase 1 first)

**Phase 1 — serving endpoint (server only, no client changes):** ✅ BUILT (2026-07-12)
- [x] `server/inference/prompts/gossip_prompt.txt` — peer-aware framing (default-write).
- [x] `generation.py`: `_make_gossip_generate()` factory + helpers
      `_openai_messages_to_conversation`, `_gossip_framing_block`, `_load_gossip_template`.
      Reuses `_sync_chat_generate` / `_build_inference_conversation` / `_temporal_anchor` /
      `_identity_line` / `_get_rag` / `ChatLogger._parse_cot`.
- [x] `mgmt_http.py`: extended `configure(...)` with `gossip_generate` / `gossip_enabled` /
      `gossip_peer_name`; added `POST /v1/chat/completions` (+ `/chat/completions`) route
      returning `_openai_response(...)`; busy → 503, disabled → 404, no messages → 400.
- [x] `server.py` `main()`: reads `gossip` config; wires `mgmt_http.configure(...)`.
- [x] `server_config.json`: `"gossip": {"enabled": false, "api_key": null, "peer_name": null}`.
- [ ] Acceptance test §4.6 (point an Encounter tab at the serving box) — **needs a live GPU
      box + a second Ava; not runnable in this environment. Run it before relying on gossip.**

**Phase 3 — digest introduction:** ✅ BUILT (2026-07-12)
- [x] `reflection_digest.render_digest_for_introduction(digest)` (first-person). Digest is
      loaded via `persona_paths.active_persona_dir()` + `latest_digest(dir)` — NOT the
      retired `load_current_digest`/`current.json`-beside-the-digest path (§6 correction).
- [x] Inject serving box's digest into `_make_gossip_generate` system content
      (`_current_digest_intro`), degrading to plain framing on a thin corpus.
- [ ] Driver opener seeded from the driver's rendered introduction — **client-side / driver
      polish, deferred** (§6.2; reuse the Encounter framing override for v1).

**Phase 2 / §5 — dedicated Gossip client tab:** ✅ BUILT (2026-07-13)
- [x] `client/ui/gossip_widget.py` `GossipWidget(EncounterWidget)` — reuses the whole
      Encounter poll/render/start-stop machinery (drives the *same* server-side encounter
      loop), overriding only `_build_ui` (gossip defaults: peer name, peer sidecar URL
      `:8767/v1/chat/completions`, `model=ava` cosmetic label) and `_on_start`.
- [x] **Framing-poison guard:** the tab pre-fills the peer-aware framing (mirrors
      `prompts/gossip_prompt.txt`) and `_on_start` restores it if blanked, so the driver
      can never fall back to the Encounter "non-subjective assistant" default — which
      would contaminate the reflected transcript (the framing is logged as exchange 0's
      `user_prompt` and rides every `system_content`, and the driver reflects on it).
- [x] `client/ui/main_window.py` — "Gossip" tab wired between Encounter and Debug
      (construction, `addTab`, `update_fonts`).
- [x] §7 serving-side reflection (**approach 2**: `generation._GossipSessionLog` logs the
      serving box's half — with its own CoT — into `hot/chats`; default on,
      `gossip.log_transcripts:false` to disable). Not the doc's `POST /gossip/transcript`
      role-invert push — that discards the serving Ava's reasoning.
- [ ] §6.2 driver opener seeded from the *driver's own* rendered digest introduction —
      still deferred (the peer already answers as its current self via §6.1; the opener is
      the generic Encounter opener under the peer-aware framing for now).
- [ ] §8 symmetric turn-token protocol.
- [ ] §4.3 bearer-token auth enforcement (`gossip.api_key` config hook present, not checked).

---

## 11. Key file index (for the implementer)

| File | Role in gossip |
|---|---|
| `server/inference/core/mgmt_http.py` | **Add** the `/v1/chat/completions` route + `configure` wiring. Port 8767. |
| `server/inference/core/generation.py` | Reuse `_sync_chat_generate` (`:416`), `_build_inference_conversation` (`:628`), `_temporal_anchor` (`:593`), `_identity_line` (`:588`), `_clean_response` (`:155`). **Add** `_make_gossip_generate`. |
| `server/inference/core/encounter_run.py` | Worked example of the same recipe (`_generate_ava` `:207`); the transcript-logging pattern §7 mirrors. |
| `server/inference/core/encounter.py` | The **client** half we are inverting; `_extract_text` (`:140`) defines the response shape we must emit. |
| `server/inference/core/reflection_digest.py` | Persona digest for §6 intro. `current.json` pointer; `render_digest_for_judge` (`:281`). Add `render_digest_for_introduction`. |
| `server/inference/server.py` | `main()` wiring: read gossip config, call `mgmt_http.configure(...)`. Owns the single GPU `_executor`. |
| `server/inference/data/hot/persona/current.json` | The serving box's live self-portrait. |
| `client/ui/encounter_widget.py` | The driver UI, reusable as-is for v1 (forwards `framing_override`). |
