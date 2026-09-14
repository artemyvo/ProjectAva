"""The HTTP facade (§7): the two-stage loop over the wire, prefix-matched sessions, tenant
fencing from the header, the context-provider endpoint, ingest/rebuild round trip."""

import json
import urllib.request

import pytest

from assoc.facade import Facade, Sessions
from assoc.bench import fixtures as fx


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def server(corpus):
    seen = []

    def answer_fn(messages):
        seen.append(messages)
        return "ok: " + messages[-1]["content"][:20]

    facade = Facade(corpus, generate_fn=None, answer_fn=answer_fn, api_key="k", model_name="assoc-test",
                    system_prompt="You are a support bot.", default_policy="support")
    srv = facade.serve(port=0)
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}", seen
    srv.shutdown()


def test_sessions_prefix_match():
    s = Sessions()
    a = s.resolve([{"role": "user", "content": "hi"}])
    b = s.resolve([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "E142?"}])
    c = s.resolve([{"role": "user", "content": "something else"}])
    assert a == b and c != a


def test_health_models_and_auth(server):
    base, _ = server
    h = _get(base + "/health")
    assert h["ok"] and h["documents"] > 0
    assert _get(base + "/v1/models")["data"][0]["id"] == "assoc-test"
    code, _ = _post(base + "/context", {"cue": "E142"})
    assert code == 401


def test_context_endpoint_fast_path(server):
    base, _ = server
    code, out = _post(base + "/context", {"cue": "E142", "scope": {"version": "2.0", "platform": "linux"}},
                      {"Authorization": "Bearer k"})
    assert code == 200 and out["fast_path"] and "retry_backoff" in out["text"]
    assert out["hits"][0]["reference"]["key"] == "docs/linux/errors.md"


def test_chat_completion_runs_two_stages(server):
    base, seen = server
    body = {"model": "assoc-test", "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "E142"}],
            "scope": {"version": "2.0", "platform": "linux"}}
    code, out = _post(base + "/v1/chat/completions", body, {"Authorization": "Bearer k", "X-Tenant": "acme"})
    assert code == 200 and out["choices"][0]["message"]["content"].startswith("ok:")
    assert out["assoc"]["reason"] == "fast_path" and out["assoc"]["scope"]["tenant"] == "acme"
    sys_msg = seen[-1][0]
    assert sys_msg["role"] == "system" and "You are a support bot." in sys_msg["content"] and "retry_backoff" in sys_msg["content"]
    assert "Be terse." in sys_msg["content"]
    code, out = _post(base + "/v1/chat/completions", {**body, "tools": [{"type": "function"}]}, {"Authorization": "Bearer k"})
    assert code == 400


def test_ingest_and_rebuild_round_trip(server):
    base, _ = server
    code, out = _post(base + "/ingest", {"text": fx.NEWS_ARTICLE, "kind": "news", "meta": {"key": "news/via-http", "date": "2026-08-21"}},
                      {"Authorization": "Bearer k"})
    assert code == 200 and out["doc_id"]
    assert _get(base + "/staleness")["scope"] == "fast"
    code, rep = _post(base + "/rebuild", {}, {"Authorization": "Bearer k"})
    assert code == 200 and rep["delta"]["documents"] == 1
    code, out = _post(base + "/context", {"cue": "Starling memory-pooling"}, {"Authorization": "Bearer k"})
    assert code == 200 and any(h["reference"]["key"] == "news/via-http" for h in out["hits"]) or out["reason"] == "no_model"


def test_needs_touch_and_aha_endpoints(server):
    base, _ = server
    code, out = _post(base + "/needs", {"register": "find a lab manager job in Haifa", "subject": "person:artemy"}, {"Authorization": "Bearer k", "X-Tenant": "acme"})
    assert code == 200 and out["need"].startswith("need:reg-")
    code, out2 = _post(base + "/needs", {}, {"Authorization": "Bearer k", "X-Tenant": "acme"})
    assert any(n["need"] == out["need"] for n in out2["needs"])
    code, t = _post(base + "/touch", {"ids": ["entity:noam keller"], "why": "mentioned"}, {"Authorization": "Bearer k"})
    assert code == 200 and t["nodes"] == ["entity:noam keller"]
    code, a = _post(base + "/aha", {"stimulus": ["entity:noam keller"]}, {"Authorization": "Bearer k"})
    assert code == 200 and isinstance(a["candidates"], list)
    code, j = _post(base + "/judge", {"need": "x", "resource": "y"}, {"Authorization": "Bearer k"})
    assert code == 404
