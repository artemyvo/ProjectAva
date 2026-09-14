"""Facts tab — curate the live ``[fact]`` set on the connected server.

The fact counterpart of the Persona tab, over the same folded ``rag_memory.jsonl``
view and with the same discipline: deleting only edits the local list, Upload appends
server-side tombstones to live RAG and consolidation state, and nothing else is touched
(no runnable snapshot, reflection archive, adapter weights, or digest artifact).

Facts differ from persona in what an operator needs to *see* before evicting one, so
the row carries its attribution — who the fact is about, and for hearsay whose account
it is (mirroring the ``— about X, per Y`` label RAG renders at recall time) — plus the
trigger it is embedded on, which is what actually decides when it comes back. Evicting
a fact drops it from chat recall, from the "I know that …" line injected into its host
exchange's CoT at the next build, and from the next standing portrait of the person it
is about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ui.memory_editor import MemoryEditorWidget

if TYPE_CHECKING:
    from core.backend_client import BackendClient


_TRIGGER_CLIP = 80


class FactsWidget(MemoryEditorWidget):
    """Edit a fetched live-fact set, then explicitly upload removals."""

    KIND = "fact"
    NOUN = "facts"
    ROW_NOUN = ("live fact", "live facts")
    REPLY_TYPE = "facts_updated"
    SEARCH_HINT = "Search fact text, subject or trigger…"
    UPLOAD_NOTE = ("They are out of live recall now; the next build drops them from their "
                   "host exchanges' reasoning.")

    def upload_rpc(self, client: "BackendClient", baseline: list[str],
                   retained: list[str]) -> dict:
        return client.update_facts(baseline, retained)

    def format_row(self, artifact: dict) -> str:
        row = f"[fact] {artifact['content'].strip()}"
        row += self._attribution(artifact)
        trigger = (artifact.get("trigger") or "").strip()
        if trigger:
            if len(trigger) > _TRIGGER_CLIP:
                trigger = trigger[:_TRIGGER_CLIP - 1].rstrip() + "…"
            # The trigger is what the fact embeds on, so it is the honest answer to
            # "when would this come back?" — worth seeing next to the content itself.
            row += f"  ·  recalled on: {trigger}"
        return row

    @staticmethod
    def _attribution(artifact: dict) -> str:
        """Mirror ``rag_engine._attribution_label`` so the row reads as recall does.

        Unattributed records — world facts, Ava's own reading, everything written
        before attribution existed — render bare, exactly as they are recalled.
        """
        about = (artifact.get("about") or "").strip()
        if not about:
            return ""
        source = (artifact.get("source") or "").strip()
        if artifact.get("source_class") == "hearsay" and source:
            return f" — about {about}, per {source}"
        return f" — about {about}"

    def eviction_warning(self, removed: int) -> str:
        return (
            f"Evict {removed} fact{'' if removed == 1 else 's'} from the connected server's "
            "live recall and next-training evidence?\n\n"
            "Each stops being recalled in chat, stops being injected into its host "
            "exchange's reasoning at the next build, and leaves the next standing portrait "
            "of the person it is about.\n\n"
            "No runnable snapshot or archived reflection bundle will be modified."
        )
