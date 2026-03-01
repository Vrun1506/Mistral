"""
In-memory per-user store for fetched conversations and pipeline results.

Structure per user:
    conversations : dict[convo_uuid, ConversationData]   — O(1) per convo
    topic_groups  : pipeline output (label → {keywords, segments})
    hierarchy     : pipeline output (root → sub → [labels])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationData:
    uuid: str
    name: str
    messages: list[dict]  # [{sender, text}, ...]


@dataclass
class UserData:
    conversations: dict[str, ConversationData] = field(default_factory=dict)
    topic_groups: dict[str, Any] | None = None
    hierarchy: dict[str, Any] | None = None

    # --- conversation helpers ---

    def upsert_conversation(self, uuid: str, name: str, messages: list[dict]) -> None:
        self.conversations[uuid] = ConversationData(uuid=uuid, name=name, messages=messages)

    def upsert_conversations_bulk(self, convos: list[dict]) -> None:
        """Accept the raw list format from fetch_all / ClaudeFetcher."""
        for c in convos:
            self.upsert_conversation(c["uuid"], c.get("name", "(untitled)"), c.get("messages", []))

    def get_conversation(self, uuid: str) -> ConversationData | None:
        return self.conversations.get(uuid)

    def as_pipeline_input(self) -> list[dict]:
        """Return conversations in the format pipeline.run_pipeline expects."""
        return [
            {"uuid": c.uuid, "name": c.name, "messages": c.messages}
            for c in self.conversations.values()
        ]

    # --- pipeline result helpers ---

    def set_pipeline_results(self, topic_groups: dict, hierarchy: dict) -> None:
        self.topic_groups = topic_groups
        self.hierarchy = hierarchy

    @property
    def is_classified(self) -> bool:
        return self.topic_groups is not None and self.hierarchy is not None


# ---------------------------------------------------------------------------
# Global store — one entry per user
# ---------------------------------------------------------------------------

_store: dict[str, UserData] = {}


def get_user(user_id: str) -> UserData:
    """Get or create the UserData for a given user."""
    if user_id not in _store:
        _store[user_id] = UserData()
    return _store[user_id]


def create_user_object_with_convos(user_id: str) -> UserData:
    """Create a fresh UserData for the given user, ready to receive conversations."""
    _store[user_id] = UserData()
    return _store[user_id]


def has_user(user_id: str) -> bool:
    return user_id in _store


def delete_user(user_id: str) -> None:
    _store.pop(user_id, None)


def all_user_ids() -> list[str]:
    return list(_store.keys())
