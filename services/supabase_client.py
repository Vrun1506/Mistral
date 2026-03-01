"""Thin wrapper around Supabase REST API for skill_progress table."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from supabase import create_client

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_skill_progress(user_id: str) -> list[dict[str, Any]]:
    """Fetch all skill_progress rows for a user."""
    result = _client.table("skill_progress").select("*").eq("user_id", user_id).execute()
    return result.data or []


def upsert_skill_progress(user_id: str, label: str, status: str) -> dict[str, Any]:
    """Insert or update a skill_progress row (upsert on user_id + topic_label)."""
    row: dict[str, Any] = {
        "user_id": user_id,
        "topic_label": label,
        "status": status,
    }
    if status == "unlocked":
        row["unlocked_at"] = datetime.now(UTC).isoformat()

    result = _client.table("skill_progress").upsert(row, on_conflict="user_id,topic_label").execute()
    return result.data[0] if result.data else row
