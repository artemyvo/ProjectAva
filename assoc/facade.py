"""The network facade (ASSOCIATIVE_MEMORY.md §7): application-side, stdlib only.

    POST /v1/chat/completions   OpenAI-compatible; runs the two-stage loop (inject → answer)
    POST /context               {messages | cue, scope?, policy?} → Block (context-provider shape)
    POST /ingest                {text, kind, meta?, scope?} → {doc_id}
    POST /extract               run the deferred witness over pending documents
    POST /rebuild               {scope?} → report
    POST /touch                 {ids, scope?, why?} → the activation nodes warmed
    POST /needs                 {scope?} → standing needs; {register: text} adds one; {close: need} closes one
    POST /aha                   {stimulus?: [ids], cue?, scope?} → candidates (arithmetic only)
    POST /judge                 {need, resource, doing?} → the verdict on a candidate from the last /aha
    GET  /health, GET /staleness, GET /v1/models

Session identity for a stateless client is DERIVED by prefix match on the turns (Ava's
`_GossipSessionLog` rule): the same conversation resent with one more turn is the same
session, and that session is the `conversation` facet of the scope. Tenancy comes from the
`X-Tenant` header (or `scope.tenant` in the body) and is never inferred from content.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .library import Library
from .selection import Policy, SUPPORT_POLICY, DEV_POLICY, AVA_POLICY

POLICIES = {"default": Policy(), "support": SUPPORT_POLICY, "dev": DEV_POLICY, "ava": AVA_POLICY}


class Sessions:
    """Prefix-matched conversation identity for stateless clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self._turns: dict[str, list[str]] = {}      # session_id -> turn fingerprints

    @staticmethod
    def _fp(m: dict) -> str:
        return hashlib.sha1(f"{m.get('role')}\x1f{m.get('content')}".encode("utf-8")).hexdigest()[:12]

    def resolve(self, messages: list[dict]) -> str:
        fps = [self._fp(m) for m in messages if m.get("role") in ("user", "assistant")]
        with self._lock:
            best, best_len = None, -1
            for sid, turns in self._turns.items():
                n = min(len(turns), len(fps))
                if n and turns[:n] == fps[:n] and n > best_len and (len(turns) <= len(fps)):
                    best, best_len = sid, n
            if best is None:
                best = uuid.uuid4().hex[:12]
            self._turns[best] = fps
            return best


class Facade:
    def __init__(self, lib: Library, *, generate_fn: Optional[Callable] = None, answer_fn: Optional[Callable] = None,
                 judge_fn: Optional[Callable] = None,
                 api_key: str = "", model_name: str = "assoc", system_prompt: str = "", default_policy: str = "default",
                 log: Optional[Callable[[str], None]] = None):
        """*generate_fn* drives `select` (thinking off); *answer_fn(messages) -> str` writes
        the reply for /v1/chat/completions; *judge_fn* is the aha judge (thinking on,
        the answering model or stronger). All injected; the facade owns no model."""
        self.lib = lib
        self.generate_fn = generate_fn
        self.answer_fn = answer_fn
        self.judge_fn = judge_fn
        self._pending: dict = {}
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_policy = default_policy
        self.sessions = Sessions()
        self.log = log or (lambda s: None)

    # ----- the loops --------------------------------------------------------------------
    def context(self, body: dict, tenant: Optional[str]) -> dict:
        messages = body.get("messages") or []
        cue = body.get("cue") or next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        if isinstance(cue, list):
            cue = " ".join(p.get("text", "") for p in cue if isinstance(p, dict))
        scope = dict(body.get("scope") or {})
        if tenant:
            scope["tenant"] = tenant
        if messages:
            scope.setdefault("conversation", self.sessions.resolve(messages))
        policy = POLICIES.get(body.get("policy") or self.default_policy, POLICIES["default"])
        context = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages[:-1][-6:] if isinstance(m.get("content"), str))
        block = self.lib.inject(str(cue), context=context, scope=scope, policy=policy, generate_fn=self.generate_fn)
        out = block.to_dict()
        out["scope"] = scope
        return out

    def chat(self, body: dict, tenant: Optional[str]) -> dict:
        if self.answer_fn is None:
            raise FacadeError(503, "no answering model configured")
        ctx = self.context(body, tenant)
        messages = list(body.get("messages") or [])
        system_parts = [p for p in (self.system_prompt, ("What is on record, fetched for this message:\n\n" + ctx["text"]) if ctx["text"] else "") if p]
        client_system = [m for m in messages if m.get("role") in ("system", "developer")]
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        for m in client_system:
            system_parts.append(str(m.get("content") or ""))
        full = ([{"role": "system", "content": "\n\n".join(system_parts)}] if system_parts else []) + turns
        reply = self.answer_fn(full)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "assoc": {"block_id": ctx["block_id"], "reason": ctx["reason"], "hits": [{"grain": h["grain"], "reference": h["reference"]} for h in ctx["hits"]],
                      "scope": ctx["scope"]},
        }

    # ----- server -----------------------------------------------------------------------
    def serve(self, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
        facade = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: N802
                facade.log(fmt % args)

            def _auth(self) -> bool:
                if not facade.api_key:
                    return True
                h = self.headers.get("Authorization", "")
                return h == f"Bearer {facade.api_key}" or self.headers.get("x-api-key") == facade.api_key

            def _send(self, code: int, obj: dict) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Tenant, x-api-key")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                if self.path in ("/health", "/v1/health"):
                    b = facade.lib.build
                    return self._send(200, {"ok": True, "model": facade.model_name, "build": b.build_id if b else None,
                                            "documents": len(facade.lib.store.all_doc_ids()), "pending": len(facade.lib.store.pending())})
                if self.path in ("/v1/models", "/models"):
                    return self._send(200, {"object": "list", "data": [{"id": facade.model_name, "object": "model", "owned_by": "assoc"}]})
                if self.path == "/staleness":
                    return self._send(200, facade.lib.staleness())
                return self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if not self._auth():
                    return self._send(401, {"error": "unauthorized"})
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except Exception:
                    return self._send(400, {"error": "bad json"})
                tenant = self.headers.get("X-Tenant") or (body.get("scope") or {}).get("tenant")
                try:
                    if self.path in ("/v1/chat/completions", "/chat/completions"):
                        if body.get("tools") or body.get("functions"):
                            return self._send(400, {"error": "tools are not supported"})
                        return self._send(200, facade.chat(body, tenant))
                    if self.path == "/context":
                        return self._send(200, facade.context(body, tenant))
                    if self.path == "/ingest":
                        scope = dict(body.get("scope") or {})
                        if tenant:
                            scope["tenant"] = tenant
                        doc_id = facade.lib.ingest(str(body.get("text") or ""), str(body.get("kind") or "article"),
                                                   dict(body.get("meta") or {}), scope)
                        return self._send(200, {"doc_id": doc_id})
                    if self.path == "/extract":
                        return self._send(200, {"reports": facade.lib.extract_pending(facade.generate_fn, limit=body.get("limit"))})
                    if self.path == "/rebuild":
                        return self._send(200, facade.lib.rebuild(body.get("scope")))
                    if self.path == "/outcome":
                        facade.lib.outcome(str(body.get("block_id")), str(body.get("signal")))
                        return self._send(200, {"ok": True})
                    if self.path == "/touch":
                        scope = dict(body.get("scope") or {})
                        if tenant:
                            scope["tenant"] = tenant
                        return self._send(200, {"nodes": facade.lib.touch(list(body.get("ids") or []), scope=scope, why=str(body.get("why") or "touch"))})
                    if self.path == "/needs":
                        scope = dict(body.get("scope") or {})
                        if tenant:
                            scope["tenant"] = tenant
                        if body.get("register"):
                            return self._send(200, {"need": facade.lib.register_need(str(body["register"]), scope=scope, subject=str(body.get("subject") or ""))})
                        if body.get("close"):
                            facade.lib.close_need(str(body["close"]), str(body.get("reason") or "closed"))
                            return self._send(200, {"ok": True})
                        return self._send(200, {"needs": facade.lib.needs(scope)})
                    if self.path == "/aha":
                        scope = dict(body.get("scope") or {})
                        if tenant:
                            scope["tenant"] = tenant
                        cands = facade.lib.aha(stimulus=body.get("stimulus"), cue=body.get("cue"), scope=scope)
                        facade._pending = {c.need + "\x1f" + c.resource: c for c in cands}
                        return self._send(200, {"candidates": [c.to_dict() for c in cands]})
                    if self.path == "/judge":
                        key = str(body.get("need")) + "\x1f" + str(body.get("resource"))
                        cand = getattr(facade, "_pending", {}).get(key)
                        if cand is None:
                            return self._send(404, {"error": "no such candidate; call /aha first"})
                        if facade.judge_fn is None:
                            raise FacadeError(503, "no judging model configured")
                        out = facade.lib.judge(cand, facade.judge_fn, doing=str(body.get("doing") or ""))
                        return self._send(200, out.to_dict())
                except FacadeError as e:
                    return self._send(e.code, {"error": e.message})
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return self._send(404, {"error": "not found"})

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


class FacadeError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message
