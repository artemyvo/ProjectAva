"""Public OpenAI-compatible API — the endpoint external tools talk to.

A second, deliberately separate HTTP listener (default port **8000**) alongside the
management sidecar (`core/mgmt_http.py`, 8767) and the WebSocket server (8765). It exists
so that anything speaking the OpenAI chat-completions protocol — an agentic code
assistant such as OpenClaw, an editor plugin, a LangChain/OpenAI-SDK script, a shell
one-liner — can query *Ava* (her prompt, her persona portrait, her memory, her adapter)
without going through the PyQt client.

**Why its own port rather than another route on 8767.** The sidecar is the box's
management plane: `/export` hands out the full adapter weights and the entire chat
corpus, `/precision` rewrites the config. That surface is meant for the operator's own
client on a trusted link. This one is meant to be *pointed at by tools*, possibly ones
the operator did not write. Separate ports keep the two blast radii separate: an operator
can bind this to a LAN address, or put a key on it, without also exposing the corpus.
(The pre-existing gossip route on 8767 is left where it is — it is peer-Ava plumbing with
its own framing and its own opt-in, not a general API. Both now share one generate body,
`generation._make_openai_generate`, so their prompt assembly can't drift.)

**Nothing here is logged.** The generate callable is built with `log_transcripts=False`
(`generation._make_api_generate`), so an external tool's traffic writes no transcript to
`data/chats/`, and therefore never reaches reflection, the training corpus, the review
archive, or the chat RAG index. Her memory is still *read* — that is the point of
querying Ava rather than the base model — but the conversation leaves no trace and cannot
shape a later build. This is the difference from the gossip route, which logs its half on
purpose so the box can reflect on it.

Endpoints:

  POST /v1/chat/completions  (also /chat/completions)
      The OpenAI chat-completion, streaming (`stream: true`, Server-Sent Events) or not.
      Streaming is what editor clients such as Continue.dev require — they stream
      unconditionally — and it is also what keeps a minute-long generation from looking
      like a hung request. Accepts `messages`,
      `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `model`, `user`.
      Content parts (`[{"type": "text", ...}]`) are flattened, so clients that always send
      the structured form work. Returns `usage`, and Ava's `<think>` trace on the separate
      `message.reasoning_content` field (DeepSeek/vLLM convention) so a client can display
      her reasoning without it contaminating `content`.
  GET  /v1/models  (also /models)
      One model entry. Many clients probe this on connect and refuse to start without it.
  GET  /health
      `{ok, model, busy}` — a cheap liveness/occupancy check that touches no GPU.

Auth is a shared bearer key (`api.api_key`): absent ⇒ open. The listener is **on by
default** (`api.enabled`, default true since 2026-07-30) so a box that pulls and restarts
is immediately queryable — which means an unconfigured box is serving GPU time on this
port to anyone who can reach it. `start()` warns at boot when a non-loopback bind has no
key; `api.host: "127.0.0.1"` limits it to the box, `api.enabled: false` turns it off.
CORS is permissive so browser-based tools can call it.

Concurrency: one GPU, one worker. A request preempts a background per-chat reflection
(interactive work wins) but is refused with 503 while an operator reflection run or an
encounter owns the box — a clean refusal beats a many-minute hang.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────────────────────────
# Injected once by server.py. Kept a thin callable bundle so this module stays
# import-light and never imports `generation` (same discipline as mgmt_http).
#   _generate(messages, *, temperature, top_p, max_tokens, peer_name)
#       -> (answer, finish_reason, reasoning, usage)   — submits GPU work to the
#          single executor and blocks.
#   _is_busy()  -> bool   — True while a reflection run / encounter owns the GPU.
#
# There is no `enabled` predicate: `api.enabled` (default true) decides whether server.py
# calls `start()` at all, so a disabled box has no listener rather than a listener that
# refuses — one fewer state to reason about than the gossip route has.
# ──────────────────────────────────────────────────────────────────────────────
_generate = None
_is_busy = lambda: False        # noqa: E731
_api_key: str | None = None
_model_name = "ava"
_default_max_tokens = "75%"


def configure(*, generate=None, is_busy=None, api_key=None,
              model_name=None, default_max_tokens=None) -> None:
    """Wire the serving capability + policy. Called once from server startup."""
    global _generate, _is_busy, _api_key, _model_name, _default_max_tokens
    if generate is not None:
        _generate = generate
    if is_busy is not None:
        _is_busy = is_busy
    if api_key is not None:
        _api_key = api_key or None
    if model_name is not None:
        _model_name = model_name or "ava"
    if default_max_tokens is not None:
        _default_max_tokens = str(default_max_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI wire format
# ──────────────────────────────────────────────────────────────────────────────

def openai_response(answer: str, model: str, finish_reason: str = "stop",
                    reasoning: str = "", usage: dict | None = None,
                    id_prefix: str = "chatcmpl") -> dict:
    """The chat-completion body an OpenAI client reads.

    ``reasoning`` (Ava's ``<think>`` trace) travels on the separate, DeepSeek/vLLM-style
    ``message.reasoning_content`` field — so a client can display her CoT without it
    contaminating ``content`` (the only field fed back into the dialogue). Shared with the
    gossip route on the management sidecar so the two can't drift apart."""
    message = {"role": "assistant", "content": answer}
    if reasoning:
        message["reasoning_content"] = reasoning
    body = {
        "id": f"{id_prefix}-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "ava",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        body["usage"] = usage
    return body


def _chunk_body(model: str, cid: str, delta: dict, finish_reason=None,
                usage: dict | None = None) -> dict:
    """One `chat.completion.chunk` — the SSE frame shape an OpenAI client accumulates."""
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "ava",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _flatten_content(content) -> str:
    """Normalize a message's ``content`` to plain text.

    OpenAI's newer shape is a list of typed parts (``[{"type": "text", "text": ...}]``) and
    several agent clients emit it unconditionally, even for pure text. Non-text parts
    (images, audio) are dropped — this box has no multimodal path — rather than failing the
    request, so a client that attaches one still gets an answer to the text it sent."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "\n".join(p for p in out if p)
    return ""


def _normalize_messages(raw) -> list:
    """Coerce incoming messages to ``[{role, content: str}]``, dropping unusable turns."""
    out = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("system", "user", "assistant", "developer"):
            continue
        # `developer` is the newer OpenAI spelling of a system instruction.
        if role == "developer":
            role = "system"
        text = _flatten_content(m.get("content"))
        if not text.strip():
            continue
        out.append({"role": role, "content": text})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: agent clients reuse one connection

    def log_message(self, *_):
        pass  # suppress default access logging

    # ── plumbing ──
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, err_type: str = "invalid_request_error",
               code: str | None = None) -> None:
        """OpenAI-shaped error body — clients surface ``error.message`` verbatim."""
        self._json(status, {"error": {"message": message, "type": err_type,
                                      "param": None, "code": code}})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body) if body else {}
            return params if isinstance(params, dict) else {}
        except Exception:
            return {}

    def _authorized(self) -> bool:
        """Shared-bearer-key check. No configured key ⇒ open (the operator's choice)."""
        if not _api_key:
            return True
        header = self.headers.get("Authorization", "") or ""
        if header.startswith("Bearer "):
            return header[len("Bearer "):].strip() == _api_key
        return (self.headers.get("x-api-key", "") or "").strip() == _api_key

    # ── routes ──
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path == "/health":
            self._json(200, {"ok": True, "model": _model_name, "busy": bool(_is_busy())})
            return

        if path in ("/v1/models", "/models"):
            if not self._authorized():
                self._error(401, "Invalid API key.", "authentication_error", "invalid_api_key")
                return
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": _model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ava",
                }],
            })
            return

        self._error(404, f"Unknown endpoint: {path}", "invalid_request_error", "not_found")

    def do_POST(self) -> None:
        # Drain the body FIRST, before any early return. This handler keeps connections
        # alive (HTTP/1.1), so unread request bytes would be parsed as the next request
        # line on the same socket — a 401 or a 404 would corrupt every following call on
        # that connection, and agent clients reuse one connection for everything.
        req = self._read_json_body()

        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"Unknown endpoint: {path}",
                        "invalid_request_error", "not_found")
            return
        if not self._authorized():
            self._error(401, "Invalid API key.", "authentication_error", "invalid_api_key")
            return
        if _generate is None:
            self._error(503, "Serving capability not wired.", "server_error")
            return
        if _is_busy():
            # A reflection run / encounter owns the single GPU worker. Refuse cleanly
            # rather than queue behind a job that can run for many minutes.
            self._error(503, "Server busy: a reflection run or encounter owns the GPU.",
                        "server_error", "busy")
            return

        if req.get("tools") or req.get("functions"):
            # Nothing in Ava produces tool calls,
            # so an agent that offers tools and silently gets prose back loops on a reply
            # it cannot parse. Told up front, a client can fall back to a promptless mode.
            self._error(400, "Tool/function calling is not supported on this endpoint — "
                             "retry without `tools`.", "invalid_request_error",
                        "tools_unsupported")
            return
        try:
            n = int(req.get("n", 1) or 1)
        except (TypeError, ValueError):
            n = 1
        if n != 1:
            self._error(400, "Only n=1 is supported.", "invalid_request_error")
            return

        messages = _normalize_messages(req.get("messages"))
        if not messages:
            self._error(400, "`messages` must be a non-empty list of {role, content}.",
                        "invalid_request_error", "missing_messages")
            return
        if not any(m["role"] == "user" for m in messages):
            self._error(400, "`messages` must contain at least one user turn.",
                        "invalid_request_error", "missing_user_turn")
            return

        def _num(key, default):
            try:
                v = req.get(key)
                return default if v is None else float(v)
            except (TypeError, ValueError):
                return default

        # `max_completion_tokens` is the newer OpenAI spelling; accept both. Absent ⇒ the
        # box default, which may be a "75%"-of-remaining-context string (chat's own
        # convention) rather than a fixed integer.
        max_tokens = req.get("max_tokens", req.get("max_completion_tokens"))
        if max_tokens is None:
            max_tokens = _default_max_tokens
        else:
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                max_tokens = _default_max_tokens

        caller = req.get("user")
        caller = caller.strip() if isinstance(caller, str) and caller.strip() else None

        model = req.get("model") or _model_name
        kwargs = dict(temperature=_num("temperature", 1.0), top_p=_num("top_p", 0.95),
                      max_tokens=max_tokens, peer_name=caller)

        if req.get("stream"):
            want_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            self._stream_completion(messages, kwargs, model, want_usage)
            return

        started = time.time()
        try:
            answer, finish_reason, reasoning, usage = _generate(messages, **kwargs)
        except RuntimeError as e:
            # The two legible failures from the generate path. An oversized prompt is the
            # client's problem and gets OpenAI's own `context_length_exceeded` code, which
            # clients handle by trimming and retrying; no model loaded is the box's problem
            # and gets a 503 the client should back off on.
            if "context" in str(e).lower():
                self._error(400, str(e), "invalid_request_error", "context_length_exceeded")
            else:
                self._error(503, str(e), "server_error", "model_unavailable")
            return
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}", "server_error")
            return

        print(f"[api] {usage.get('prompt_tokens', 0)}→"
              f"{usage.get('completion_tokens', 0)} tok in "
              f"{time.time() - started:.1f}s ({finish_reason})", flush=True)
        self._json(200, openai_response(answer, model, finish_reason, reasoning, usage))

    # ── streaming ──
    def _stream_completion(self, messages, kwargs, model: str, want_usage: bool) -> None:
        """Serve the completion as Server-Sent Events.

        Required by editor clients (Continue.dev and friends stream unconditionally), and
        the reason it is not merely a nicety: a generation that runs for a minute looks
        like a hung request until the first token arrives.

        Two threads are unavoidable here. `_generate` blocks on the single GPU executor,
        and only *this* thread may write the socket, so generation runs on a helper thread
        and pushes deltas through a queue that this thread drains into the wire.

        Headers are deliberately withheld until the first event arrives. Committing to
        `200 text/event-stream` up front would mean reporting every failure — no model
        loaded, a prompt past the context window — as a 200 with an error buried in the
        body, which is precisely the "fails far from the cause" problem the non-streaming
        path avoids. Waiting one event costs nothing and keeps real status codes for the
        errors that happen before generation starts.
        """
        q: "queue.Queue" = queue.Queue()
        done = object()
        box: dict = {}

        def on_delta(kind, text):
            q.put((kind, text))

        def run():
            try:
                box["result"] = _generate(messages, on_delta=on_delta, **kwargs)
            except BaseException as e:          # re-raised on this thread below
                box["error"] = e
            finally:
                q.put(done)

        started = time.time()
        threading.Thread(target=run, daemon=True, name="ava-api-stream").start()

        first = q.get()
        if first is done and "error" in box:
            e = box["error"]
            if isinstance(e, RuntimeError) and "context" in str(e).lower():
                self._error(400, str(e), "invalid_request_error", "context_length_exceeded")
            elif isinstance(e, RuntimeError):
                self._error(503, str(e), "server_error", "model_unavailable")
            else:
                self._error(500, f"{type(e).__name__}: {e}", "server_error")
            return

        cid = f"chatcmpl-{uuid.uuid4().hex}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length is possible, and chunked framing would have to be hand-rolled
        # on BaseHTTPRequestHandler; ending the body at EOF is the simpler correct option.
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True

        def send(obj) -> bool:
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False   # client hung up (cancelled edit, closed tab)

        send(_chunk_body(model, cid, {"role": "assistant", "content": ""}))

        event = first
        alive = True
        while event is not done:
            kind, text = event
            if alive:
                key = "reasoning_content" if kind == "reasoning" else "content"
                alive = send(_chunk_body(model, cid, {key: text}))
            event = q.get()

        if "error" in box and alive:
            # Failed mid-stream: the client already has a partial reply, so there is no
            # status code left to use. Say so in-band rather than truncating silently.
            e = box["error"]
            send({"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})

        usage = None
        finish_reason = "stop"
        if "result" in box:
            _, finish_reason, _, usage = box["result"]
        if alive:
            send(_chunk_body(model, cid, {}, finish_reason,
                             usage if (want_usage and usage) else None))
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        u = usage or {}
        print(f"[api] stream {u.get('prompt_tokens', 0)}→{u.get('completion_tokens', 0)} "
              f"tok in {time.time() - started:.1f}s ({finish_reason}"
              f"{'' if alive else ', client gone'})", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def start(host: str, port: int) -> ThreadingHTTPServer:
    """Launch the API server in a daemon thread and return it.

    Threaded so a `/health` or `/v1/models` probe answers while a completion is blocked on
    the GPU worker — several clients poll one while waiting on the other."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="ava-openai-api")
    t.start()
    port = httpd.server_address[1]   # resolve an ephemeral (port=0) bind before printing
    shown = socket.gethostname() if host in ("0.0.0.0", "::") else host
    print(f"[api] OpenAI-compatible API on http://{host}:{port}/v1 "
          f"(model={_model_name}, auth={'key' if _api_key else 'OPEN'}, "
          f"reachable as {shown}:{port}); requests are NOT logged", flush=True)
    if not _api_key and host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[api] WARNING: bound to {host} with no api.api_key — anyone who can "
              f"reach this port can spend GPU time as Ava.", flush=True)
    return httpd


# ──────────────────────────────────────────────────────────────────────────────
# GPU-free self-test:  python -m core.api_http
# ──────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Drive every route against a stub generate — no model, no GPU, no config."""
    import urllib.error
    import urllib.request

    seen: dict = {}

    def fake_generate(messages, *, temperature, top_p, max_tokens, peer_name,
                      on_delta=None):
        seen.update(messages=messages, temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, peer_name=peer_name)
        if on_delta is not None:
            on_delta("reasoning", "thinking...")
            on_delta("content", "hel")
            on_delta("content", "lo")
        return ("hello", "stop", "thinking...",
                {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13})

    configure(generate=fake_generate, is_busy=lambda: False,
              api_key="secret", model_name="ava-test", default_max_tokens="75%")
    httpd = start("127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, key="secret", method=None):
        req = urllib.request.Request(
            base + path, method=method or ("POST" if body is not None else "GET"),
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    failures = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
        if not cond:
            failures.append(name)

    print("api_http self-test")

    st, body = call("/health", key=None)
    check("health is open + reports model", st == 200 and body.get("model") == "ava-test",
          f"{st} {body}")

    st, body = call("/v1/models")
    check("models lists one entry", st == 200 and body["data"][0]["id"] == "ava-test",
          f"{st} {body}")

    st, _ = call("/v1/models", key="wrong")
    check("bad key is 401", st == 401, str(st))

    st, _ = call("/v1/models", key=None)
    check("missing key is 401", st == 401, str(st))

    # The shape an agent client sends: system brief + structured content parts.
    st, body = call("/v1/chat/completions", {
        "model": "ava",
        "user": "openclaw",
        "messages": [
            {"role": "developer", "content": "You are a coding agent."},
            {"role": "user", "content": [{"type": "text", "text": "ping"},
                                         {"type": "image_url", "image_url": {"url": "x"}}]},
        ],
        "temperature": 0.7,
    })
    check("completion returns OpenAI body",
          st == 200 and body["choices"][0]["message"]["content"] == "hello"
          and body["object"] == "chat.completion", f"{st} {body}")
    check("reasoning on its own field",
          body.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
          == "thinking...", str(body))
    check("usage passed through", body.get("usage", {}).get("total_tokens") == 13,
          str(body.get("usage")))
    check("developer role normalized to system",
          seen["messages"][0] == {"role": "system", "content": "You are a coding agent."},
          str(seen["messages"]))
    check("content parts flattened, image dropped",
          seen["messages"][1]["content"] == "ping", str(seen["messages"]))
    check("caller name forwarded", seen["peer_name"] == "openclaw", str(seen["peer_name"]))
    check("temperature forwarded", seen["temperature"] == 0.7, str(seen["temperature"]))
    check("absent max_tokens falls back to box default", seen["max_tokens"] == "75%",
          str(seen["max_tokens"]))

    # Streaming: the shape Continue.dev and every editor client consume.
    def stream_call(body, key="secret"):
        req = urllib.request.Request(
            base + "/v1/chat/completions", method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {key}"} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read().decode()

    st, ctype, raw = stream_call({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "stream_options": {"include_usage": True}})
    frames = [ln[len("data: "):] for ln in raw.splitlines() if ln.startswith("data: ")]
    objs = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
    check("stream returns SSE", st == 200 and "text/event-stream" in ctype,
          f"{st} {ctype}")
    check("stream terminates with [DONE]", frames and frames[-1].strip() == "[DONE]",
          str(frames[-1:]))
    check("first frame opens the assistant role",
          objs[0]["choices"][0]["delta"].get("role") == "assistant", str(objs[:1]))
    check("content deltas concatenate to the answer",
          "".join(o["choices"][0]["delta"].get("content", "") for o in objs) == "hello",
          str(objs))
    check("reasoning streams on its own delta field",
          any(o["choices"][0]["delta"].get("reasoning_content") for o in objs), str(objs))
    check("final frame carries finish_reason",
          objs[-1]["choices"][0]["finish_reason"] == "stop", str(objs[-1:]))
    check("stream_options.include_usage adds usage",
          objs[-1].get("usage", {}).get("total_tokens") == 13, str(objs[-1:]))
    check("chunk objects are typed for the client",
          all(o["object"] == "chat.completion.chunk" for o in objs), str(objs[:1]))
    check("one completion id across the stream",
          len({o["id"] for o in objs}) == 1, str({o["id"] for o in objs}))

    # A pre-generation failure must still be a real status code, not a 200 with an
    # error buried in the event stream.
    configure(generate=lambda messages, **_: (_ for _ in ()).throw(
        RuntimeError("Input is 99999 tokens but the context window is 24576.")))
    st, ctype, raw = stream_call({"messages": [{"role": "user", "content": "hi"}],
                                  "stream": True})
    check("stream failing before the first token is a real 400",
          st == 400 and "application/json" in ctype
          and json.loads(raw)["error"]["code"] == "context_length_exceeded",
          f"{st} {ctype} {raw[:120]}")
    configure(generate=fake_generate)

    st, body = call("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "read_file"}}]})
    check("tool calling refused up front",
          st == 400 and body["error"]["code"] == "tools_unsupported", f"{st} {body}")

    st, body = call("/v1/chat/completions", {"messages": []})
    check("empty messages is 400", st == 400, f"{st} {body}")

    st, body = call("/v1/chat/completions",
                    {"messages": [{"role": "system", "content": "only a brief"}]})
    check("system-only conversation is 400", st == 400, f"{st} {body}")

    st, _ = call("/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hi"}]}, key="wrong")
    check("completion honours the key", st == 401, str(st))

    st, _ = call("/nope")
    check("unknown route is 404", st == 404, str(st))

    # An oversized prompt is the caller's problem, not the box's: OpenAI's own
    # `context_length_exceeded` on a 400, which clients handle by trimming.
    def raiser(messages, **_):
        raise RuntimeError("Input is 99999 tokens but the context window is 24576.")

    configure(generate=raiser)
    st, body = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    check("oversized prompt is a 400 context_length_exceeded",
          st == 400 and body["error"]["code"] == "context_length_exceeded", f"{st} {body}")

    configure(generate=lambda messages, **_: (_ for _ in ()).throw(RuntimeError("No model loaded")))
    st, body = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    check("no model loaded is a 503", st == 503, f"{st} {body}")

    # Busy: an operator reflection run / encounter owns the GPU.
    configure(is_busy=lambda: True)
    st, body = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    check("busy box refuses with 503", st == 503 and body["error"]["code"] == "busy",
          f"{st} {body}")

    httpd.shutdown()
    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _selftest()
