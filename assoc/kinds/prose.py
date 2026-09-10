"""Prose kinds: tech_doc, news, article (§1.3). All three split with the Markdown
splitter; they differ in witness framing, class vocabulary, subject namespace and clock.
"""

from __future__ import annotations

from . import KindSpec, register, FACET_PROPERTY, FACET_REPORT, FACET_EVENT, FACET_DEPICTION, \
    FACET_NORM, FACET_PROCEDURE, FACET_POSITION, FACET_UNCLASSIFIED
from .markdown import split_markdown


def _split(text: str, meta: dict):
    units = split_markdown(meta.get("key", ""), text, title=meta.get("title"))
    return text, units


def _split_paragraphs(text: str, meta: dict):
    units = split_markdown(meta.get("key", ""), text, title=meta.get("title"), paragraph_primary=True)
    return text, units


TECH_DOC = register(KindSpec(
    name="tech_doc", split=_split, witnesses=("structure", "llm"),
    classes=("spec", "procedure", "signature", "standing", "deprecated", "event"),
    facet_map={"spec": FACET_NORM, "procedure": FACET_PROCEDURE, "signature": FACET_PROPERTY,
               "standing": FACET_PROPERTY, "deprecated": FACET_NORM, "event": FACET_EVENT,
               "unspecified": FACET_UNCLASSIFIED},
    namespace="ident", clock="doc_version", prompt_file="witness_tech_doc.txt", thinking=False,
))

NEWS = register(KindSpec(
    name="news", split=_split_paragraphs, witnesses=("llm",),
    classes=("stated", "event", "standing"),
    facet_map={"stated": FACET_REPORT, "event": FACET_EVENT, "standing": FACET_PROPERTY,
               "unspecified": FACET_UNCLASSIFIED},
    namespace="entity", clock="article_date", prompt_file="witness_news.txt", thinking=False,
))

ARTICLE = register(KindSpec(
    name="article", split=_split_paragraphs, witnesses=("llm",),
    classes=("standing", "event", "depicted", "stated"),
    facet_map={"standing": FACET_PROPERTY, "event": FACET_EVENT, "depicted": FACET_DEPICTION,
               "stated": FACET_REPORT, "unspecified": FACET_UNCLASSIFIED},
    namespace="entity", clock="fetch_date", prompt_file="witness_article.txt", thinking=False,
))

# `structured`: the Markdown splitter with the structure witness only — tables, FAQ entries,
# release-note items are facts by construction (§1.3).
STRUCTURED = register(KindSpec(
    name="structured", split=_split, witnesses=("structure",),
    classes=("standing", "procedure", "event"),
    facet_map={"standing": FACET_PROPERTY, "procedure": FACET_PROCEDURE, "event": FACET_EVENT,
               "unspecified": FACET_UNCLASSIFIED},
    namespace="ident", clock="page_date",
))
