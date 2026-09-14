"""The facts tree — the offline fold over the two fact-protocol lanes.

``core/chat_facts.py`` and ``core/til_facts.py`` each write an immutable, exhaustive
extraction record per source (``<stem>.facts.json``), and both were written for a consumer
that did not exist: *offline, a knowledge-graph build*. This package is that build.

The one architectural claim, which everything here follows from:

    **The tree is a derived, disposable fold — never a store.**

The ``.facts.json`` files stay the immutable sources; the tree is rebuilt from them and can
be deleted at any time with no loss. So a better resolver is a *rebuild*, not a migration;
a re-reflected protocol supersedes with no supersession logic; and the tree can never
disagree with its sources about what was said, only about how mentions resolve — which is
exactly what a rebuild fixes. This is the source/fold discipline the project already uses
for the ledger, live memory and the portraits.

Stage 1 (this) is pure and GPU-free: read → normalize → fold → dump. See ``FACTS_TREE.md``
for the full design and the staging plan, and ``DESIGN.md`` here for the detail.

GPU-free self-tests: ``python -m graph.nodes``, ``python -m graph.fold``,
``python -m graph.read``, ``python -m graph.selftest`` (all of them).
"""
