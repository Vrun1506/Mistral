"""Supabase-backed flashcard storage — replaces the old SQLite db.py.

Tables: public.decks, public.cards (see supabase migration SQL).
Uses the service-role client so RLS is bypassed server-side.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from supabase import create_client

_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------


def create_deck(user_id: str, topic_label: str) -> dict[str, Any]:
    """Insert a new deck row and return it."""
    row = {
        "user_id": user_id,
        "topic_label": topic_label,
        "created_at": datetime.now(UTC).isoformat(),
    }
    result = _client.table("decks").insert(row).execute()
    return result.data[0]


def list_decks(user_id: str) -> list[dict[str, Any]]:
    """Return all decks for a user, each enriched with card_count."""
    decks_res = (
        _client.table("decks")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    deck_list = decks_res.data or []
    if not deck_list:
        return []

    deck_ids = [d["id"] for d in deck_list]
    cards_res = (
        _client.table("cards")
        .select("deck_id")
        .in_("deck_id", deck_ids)
        .execute()
    )
    count_map: dict[int, int] = {}
    for c in cards_res.data or []:
        count_map[c["deck_id"]] = count_map.get(c["deck_id"], 0) + 1

    for d in deck_list:
        d["card_count"] = count_map.get(d["id"], 0)

    return deck_list


def get_deck_with_cards(deck_id: int) -> dict[str, Any] | None:
    """Fetch a single deck + its cards. Returns None if not found."""
    deck_res = _client.table("decks").select("*").eq("id", deck_id).execute()
    if not deck_res.data:
        return None

    cards_res = (
        _client.table("cards")
        .select("*")
        .eq("deck_id", deck_id)
        .order("id")
        .execute()
    )
    return {"deck": deck_res.data[0], "cards": cards_res.data or []}


def delete_deck(deck_id: int) -> None:
    """Delete a deck (cards are cascade-deleted by FK)."""
    _client.table("decks").delete().eq("id", deck_id).execute()


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def insert_cards(deck_id: int, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bulk-insert cards into a deck."""
    now = datetime.now(UTC).isoformat()
    rows = []
    for card in cards:
        card_dict = card if isinstance(card, dict) else card.model_dump()
        card_type = card_dict.get("card_type") or card_dict.get("type", "Basic")
        extra_data = {
            k: v for k, v in card_dict.items()
            if k not in ("type", "card_type", "question", "answer")
        }
        rows.append({
            "deck_id": deck_id,
            "card_type": card_type,
            "question": card_dict["question"],
            "answer": card_dict["answer"],
            "extra": json.dumps(extra_data) if extra_data else None,
            "created_at": now,
        })
    if not rows:
        return []
    result = _client.table("cards").insert(rows).execute()
    return result.data or []


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def save_deck(user_id: str, topic_label: str, cards: list[dict[str, Any]]) -> int:
    """Create a deck and insert its cards. Returns the deck id."""
    deck = create_deck(user_id, topic_label)
    deck_id: int = deck["id"]
    insert_cards(deck_id, cards)
    return deck_id