"""The code kind (§1.3): tree-sitter splitter by symbol + the PARSER witness (no model).

Units: one primary per top-level symbol (function / class / method, with its docstring);
a class's methods are pieces of the class; a long function's top-level statements are
pieces with the signature + docstring as identity. Facts: ``signature``, ``defines``,
``imports``, ``calls``, ``docstring``, ``todo`` — exact, grounded by construction, each
anchored to the symbol's span. Symbol namespace: ``ident:<project>:<qualified name>``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from . import KindSpec, register, FACET_PROPERTY, FACET_NEED, FACET_UNCLASSIFIED
from ..chunks import Unit, make_unit

_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.*)", re.IGNORECASE)
_SECRET_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"][^'\"]{8,}['\"]|['\"](?:sk|ghp|xox[abp]|AKIA)[A-Za-z0-9_\-]{12,}['\"]")
_LANG_BY_EXT = {".py": "python", ".c": "c", ".h": "c"}


@lru_cache(maxsize=4)
def _parser(lang: str):
    from tree_sitter import Language, Parser
    if lang == "python":
        import tree_sitter_python as m
    elif lang == "c":
        import tree_sitter_c as m
    else:
        raise KeyError(lang)
    return Parser(Language(m.language()))


def language_for(path: str, meta: dict) -> str:
    if meta.get("language"):
        return meta["language"]
    for ext, lang in _LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return "python"


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _py_symbols(src: bytes, root) -> list[dict]:
    """Top-level functions/classes (+ methods) with name, span, signature, docstring, calls, todos."""
    out: list[dict] = []

    def docstring(body) -> str:
        if body is None or body.child_count == 0:
            return ""
        first = body.children[0]
        if first.type == "expression_statement" and first.children and first.children[0].type == "string":
            s = _text(src, first.children[0]).strip()
            return s.strip("\"'").strip()
        return ""

    def calls(node) -> list[str]:
        names: list[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "call":
                fn = n.child_by_field_name("function")
                if fn is not None:
                    names.append(_text(src, fn))
            stack.extend(n.children)
        seen: list[str] = []
        for c in names:
            if c not in seen:
                seen.append(c)
        return seen

    def walk(node, qual: list[str], depth: int):
        for ch in node.children:
            t = ch.type
            if t in ("function_definition", "class_definition", "decorated_definition"):
                inner = ch
                if t == "decorated_definition":
                    inner = next((c for c in ch.children if c.type in ("function_definition", "class_definition")), None)
                    if inner is None:
                        continue
                name_node = inner.child_by_field_name("name")
                name = _text(src, name_node) if name_node else "?"
                q = qual + [name]
                body = inner.child_by_field_name("body")
                params = inner.child_by_field_name("parameters")
                sig = f"{'class' if inner.type == 'class_definition' else 'def'} {name}{_text(src, params) if params else ''}"
                rec = {"name": ".".join(q), "kind": "class" if inner.type == "class_definition" else ("method" if depth else "function"),
                       "span": (ch.start_byte, ch.end_byte), "signature": sig, "docstring": docstring(body),
                       "calls": calls(inner) if inner.type != "class_definition" else [], "depth": depth,
                       "body_span": (body.start_byte, body.end_byte) if body else (ch.end_byte, ch.end_byte),
                       "blocks": [(c.start_byte, c.end_byte) for c in (body.children if body else []) if c.is_named]}
                out.append(rec)
                if inner.type == "class_definition" and body is not None:
                    walk(body, q, depth + 1)
            elif t in ("import_statement", "import_from_statement"):
                out.append({"name": "", "kind": "import", "span": (ch.start_byte, ch.end_byte), "text": _text(src, ch)})
    walk(root, [], 0)
    return out


def _c_symbols(src: bytes, root) -> list[dict]:
    out: list[dict] = []

    def declarator_name(node) -> str:
        d = node
        while d is not None:
            if d.type == "identifier":
                return _text(src, d)
            nxt = d.child_by_field_name("declarator")
            if nxt is None:
                nxt = next((c for c in d.children if c.type in ("function_declarator", "pointer_declarator", "identifier", "parenthesized_declarator")), None)
            d = nxt
        return "?"

    def calls(node) -> list[str]:
        names: list[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "call_expression":
                fn = n.child_by_field_name("function")
                if fn is not None:
                    names.append(_text(src, fn))
            stack.extend(n.children)
        seen: list[str] = []
        for c in names:
            if c not in seen:
                seen.append(c)
        return seen

    for ch in root.children:
        if ch.type == "function_definition":
            decl = ch.child_by_field_name("declarator")
            name = declarator_name(decl) if decl is not None else "?"
            body = ch.child_by_field_name("body")
            sig = _text(src, ch)[: (body.start_byte - ch.start_byte) if body else None].strip()
            # A leading comment block is the docstring.
            out.append({"name": name, "kind": "function", "span": (ch.start_byte, ch.end_byte), "signature": sig,
                        "docstring": "", "calls": calls(ch), "depth": 0,
                        "body_span": (body.start_byte, body.end_byte) if body else (ch.end_byte, ch.end_byte),
                        "blocks": [(c.start_byte, c.end_byte) for c in (body.children if body else []) if c.is_named]})
        elif ch.type == "preproc_include":
            out.append({"name": "", "kind": "import", "span": (ch.start_byte, ch.end_byte), "text": _text(src, ch).strip()})
        elif ch.type in ("struct_specifier", "type_definition", "declaration") and "{" in _text(src, ch):
            name = "?"
            for c in ch.children:
                if c.type == "type_identifier":
                    name = _text(src, c)
                if c.type == "struct_specifier":
                    nm = c.child_by_field_name("name")
                    if nm is not None:
                        name = _text(src, nm)
            out.append({"name": name, "kind": "type", "span": (ch.start_byte, ch.end_byte),
                        "signature": _text(src, ch).split("{")[0].strip(), "docstring": "", "calls": [], "depth": 0,
                        "body_span": (ch.start_byte, ch.end_byte), "blocks": []})
    return out


def _b2c(src: bytes, b: int) -> int:
    return len(src[:b].decode("utf-8", "replace"))


def mask_secrets(text: str) -> str:
    """Same-length masking of key/token-shaped literals in the RENDERED text, so spans hold
    and a quoted passage can never carry what the witness is told to drop (§1.3 redact)."""
    def _mask(m: "re.Match[str]") -> str:
        s = m.group(0)
        q = s[-1] if s[-1] in "\"'" else ""
        head = s[: s.index("=") + 1] if "=" in s and ":" not in s.split("=")[0] else (s[: s.index(":") + 1] if ":" in s else "")
        body = s[len(head):]
        return head + "".join(ch if ch in "\"' " else "*" for ch in body)
    return _SECRET_RE.sub(_mask, text)


def _split(text: str, meta: dict):
    path = str(meta.get("key") or meta.get("path") or "")
    lang = language_for(path, meta)
    text = mask_secrets(text)
    src = text.encode("utf-8")
    tree = _parser(lang).parse(src)
    syms = _py_symbols(src, tree.root_node) if lang == "python" else _c_symbols(src, tree.root_node)
    project = str(meta.get("project") or "")
    units: list[Unit] = []
    by_name: dict[str, Unit] = {}
    for s in syms:
        if s["kind"] == "import":
            continue
        a, b = _b2c(src, s["span"][0]), _b2c(src, s["span"][1])
        role = "piece" if s["depth"] > 0 else "primary"
        parent = by_name.get(s["name"].rsplit(".", 1)[0]) if s["depth"] > 0 else None
        u = make_unit(path, role=role, path=[s["name"]], span=(a, b), text=text[a:b], unit_type="symbol", parent=parent,
                      keys={"symbol": s["name"], "lang": lang, "kind": s["kind"], "signature": s["signature"],
                            "docstring": s["docstring"], "calls": s["calls"], "project": project,
                            "identity": s["signature"] + ((" — " + s["docstring"].splitlines()[0]) if s["docstring"] else "")})
        units.append(u)
        by_name[s["name"]] = u
        # A function's top-level statements are pieces (used only when the function is oversize).
        if role == "primary" and s["kind"] in ("function", "method") and len(s["blocks"]) > 1:
            for bi, (ba, bb) in enumerate(s["blocks"]):
                ca, cb = _b2c(src, ba), _b2c(src, bb)
                units.append(make_unit(path, role="piece", path=[s["name"], f"block{bi}"], span=(ca, cb), text=text[ca:cb],
                                       unit_type="block", parent=u, keys={"identity": u.keys["identity"], "symbol": s["name"]}))
    # Imports: one primary "imports" unit for the file header.
    imports = [s for s in syms if s["kind"] == "import"]
    if imports:
        a, b = _b2c(src, imports[0]["span"][0]), _b2c(src, imports[-1]["span"][1])
        units.insert(0, make_unit(path, role="primary", path=["(imports)"], span=(a, b), text=text[a:b], unit_type="symbol",
                                  keys={"symbol": "(imports)", "lang": lang, "kind": "imports", "project": project,
                                        "imports": [s["text"] for s in imports], "identity": f"imports of {path}"}))
    return text, units


def parser_witness(doc_meta: dict, text: str, units: list[Unit]) -> list[dict]:
    """The code witness: facts from the parse, grounded by construction (§1.3)."""
    facts: list[dict] = []
    project = str(doc_meta.get("project") or "")
    file_path = str(doc_meta.get("key") or "")
    fam = project or file_path

    def fact(u: Unit, cls: str, txt: str, *, subject: str, entities=(), extra=None):
        rec = {"subject": subject, "subject_raw": subject, "text": txt, "fact_class": cls,
               "entities": list(entities), "when": "", "chunk_id": u.chunk_id, "span": list(u.span),
               "anchor": "exact", "grounded": True, "language": "lat", "version": str(doc_meta.get("version") or "")}
        if extra:
            rec.update(extra)
        facts.append(rec)

    for u in units:
        if u.unit_type != "symbol" or u.role == "piece" and u.keys.get("kind") not in ("method",):
            if u.unit_type == "block":
                continue
        k = u.keys
        if k.get("kind") == "imports":
            for imp in k.get("imports", []):
                fact(u, "imports", f"{file_path} imports {imp}", subject=f"ident:{fam}:{file_path}", entities=[imp],
                     extra={"rel": ["imports", f"ident:{fam}:{file_path}", imp]})
            continue
        sym = f"ident:{fam}:{k['symbol']}"
        fact(u, "signature", k.get("signature", ""), subject=sym)
        fact(u, "defines", f"{file_path} defines {k['symbol']} ({k.get('kind')})", subject=sym, entities=[file_path],
             extra={"rel": ["defines", f"ident:{fam}:{file_path}", sym]})
        if k.get("docstring"):
            fact(u, "docstring", k["docstring"].split("\n\n")[0].strip(), subject=sym)
        for c in k.get("calls", []):
            fact(u, "calls", f"{k['symbol']} calls {c}", subject=sym, entities=[c], extra={"rel": ["calls", sym, f"ident:{fam}:{c}"]})
        for m in _TODO_RE.finditer(u.text):
            note = (m.group(2).strip().splitlines()[0] if m.group(2) else "").strip().rstrip("\"'*/ ").strip()
            fact(u, "todo", f"{m.group(1).upper()}: {note}", subject=sym, extra={"need": True})
    return facts


def redact_code(fact: dict) -> Optional[dict]:
    if _SECRET_RE.search(fact.get("text", "")):
        return None
    return fact


CODE = register(KindSpec(
    name="code", split=_split, witnesses=("parser",),
    classes=("signature", "defines", "imports", "calls", "docstring", "todo"),
    facet_map={"signature": FACET_PROPERTY, "defines": FACET_PROPERTY, "imports": FACET_PROPERTY,
               "calls": FACET_PROPERTY, "docstring": FACET_PROPERTY, "todo": FACET_NEED,
               "unspecified": FACET_UNCLASSIFIED},
    namespace="ident", clock="commit", redact=redact_code,
))
