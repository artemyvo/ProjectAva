"""bench/code — the parser witness end to end (§9): exact protocol, `what calls X`, the
identifier as the bridge between chat and code, chunk identity across a save, a long
function split by block with its signature on each piece, secrets absent, todo → need."""

from assoc.library import Library
from assoc.budget import Budget
from assoc.bench import fixtures as fx


def _claims(corpus, key):
    return [c for c in corpus.build.claims.values() if any(o["key"] == key for o in c["occurrences"])]


def test_protocol_matches_hand_written(corpus):
    texts = {c["text"] for c in _claims(corpus, "demo/client.py")}
    expected = {
        "def resolve(host)", "demo/client.py defines resolve (function)", "resolve calls socket.gethostbyname",
        "class Client", "def connect(self)", "Client.connect calls resolve", "Client.connect calls socket.create_connection",
        "Client.connect calls ConnectError", "demo/client.py imports import os", "TODO: cache results",
        "Resolve a host name to an address. TODO: cache results",
    }
    missing = expected - texts
    assert not missing, missing
    c_texts = {c["text"] for c in _claims(corpus, "demo/conn.c")}
    assert {"open_conn calls gw_resolve", "open_conn calls gw_connect", "main calls open_conn", "main calls printf",
            'demo/conn.c imports #include "gateway.h"'} <= c_texts


def test_what_calls_x_is_one_hop(corpus):
    hits = corpus.pull("what calls resolve?", limit=10)
    claim_texts = [(h.claim or {}).get("text") for h in hits if h.grain == "claim"]
    assert "Client.connect calls resolve" in claim_texts[:3]
    rels = [c["rel"] for c in _claims(corpus, "demo/client.py") if c.get("rel") and c["rel"][0] == "calls"]
    assert ["calls", "ident:demo:Client.connect", "ident:demo:resolve"] in rels


def test_identifier_bridges_chat_and_code(fresh_root):
    lib = Library(fresh_root)
    lib.ingest(fx.PY_MODULE, "code", {"key": "demo/client.py", "project": "demo", "version": "a"})
    lib.ingest('{"user":"Dev","exchanges":[{"user_prompt":"Should Client.connect retry when resolve fails?","assistant_response":"Probably once."}]}',
               "chat", {"key": "dev-chat", "date": "2026-09-01"})
    lib.rebuild()
    hits = lib.pull("Client.connect", limit=10)
    kinds = {h.meta.get("kind") for h in hits}
    assert {"code", "chat"} <= kinds
    assert hits[0].exact


def test_chunk_identity_survives_a_save(fresh_root):
    lib = Library(fresh_root)
    d1 = lib.ingest(fx.PY_MODULE, "code", {"key": "demo/client.py", "project": "demo", "version": "a"})
    edited = fx.PY_MODULE.replace('"""Connection helpers for the demo client."""', '"""Connection helpers for the demo client."""\nimport sys')
    edited = edited.replace("    def close(self):\n        return None", "    def close(self):\n        self.host = None\n        return None")
    d2 = lib.ingest(edited, "code", {"key": "demo/client.py", "project": "demo", "version": "b"})
    a = {u.path[0]: u for u in lib.store.document(d1).units if u.role in ("primary", "piece") and u.unit_type == "symbol"}
    b = {u.path[0]: u for u in lib.store.document(d2).units if u.role in ("primary", "piece") and u.unit_type == "symbol"}
    for name in ("resolve", "Client.connect", "Client.__init__"):
        assert a[name].chunk_id == b[name].chunk_id and a[name].span != b[name].span, name
    assert a["Client.close"].chunk_id != b["Client.close"].chunk_id
    assert lib.store.meta(d2)["supersedes"] == d1


def test_long_function_split_by_block_with_signature(fresh_root):
    body = "\n".join(f"    x{i} = compute_{i}(x{i - 1} if {i} else 0)  # step {i} of the long function" for i in range(120))
    src = f'def long_one(seed):\n    """A very long function."""\n{body}\n    return x119\n'
    lib = Library(fresh_root, budget=Budget(total=400))
    lib.ingest(src, "code", {"key": "big.py", "project": "demo", "version": "a"})
    rep = lib.rebuild()
    assert rep["fits"]["oversize"] == 1 and rep["fits"]["split"] == 1
    b = lib.build
    pieces = [u for u in lib.store.document(lib.store.latest_id("big.py")).units if u.role == "piece"]
    inj = [u for u in pieces if b.injectable(u.chunk_id)]
    assert inj and all(u.identity.startswith("def long_one(seed)") for u in inj)
    hits = lib.pull("compute_57", limit=5)
    assert hits and hits[0].grain == "chunk" and b.injectable(hits[0].chunk_id)


def test_secret_absent_from_protocol_and_passages(corpus):
    doc = corpus.store.document(corpus.store.latest_id("demo/client.py"))
    assert "sk-THISISASECRETKEY" not in doc.text
    assert all("sk-THIS" not in f["text"] for f in doc.facts["facts"])
    assert "sk-THISISASECRETKEY" in (corpus.store.root / "documents" / doc.doc_id / "source.txt").read_text()


def test_todo_is_a_need(corpus):
    needs = {n["text"] for n in corpus.needs()}
    assert "TODO: cache results" in needs and "TODO: read the timeout from the config" in needs
