"""Skill tree API endpoints — constellation unlock progress."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user_id
from services.supabase_client import get_skill_progress, upsert_skill_progress
from store import get_user

router = APIRouter(prefix="/api", tags=["skills"])


@router.get("/skills")
async def get_skills(user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """Return the full skill tree: hierarchy merged with unlock status per topic."""
    from main import _is_sensitive

    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        return {"tree": {}}

    # Fetch existing unlock progress from Supabase
    progress_rows = get_skill_progress(user_id)
    unlocked: set[str] = {row["topic_label"] for row in progress_rows if row.get("status") == "unlocked"}

    tree: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for root_name, subcats in user.hierarchy.items():
        tree[root_name] = {}
        for sub_name, labels in subcats.items():
            topics: list[dict[str, Any]] = []
            for label in labels:
                info: dict[str, Any] = user.topic_groups.get(label, {})  # type: ignore[assignment]
                keywords = info.get("keywords", [])[:5]
                if _is_sensitive(label, keywords):
                    continue
                topics.append(
                    {
                        "label": label,
                        "keywords": keywords,
                        "segment_count": len(info.get("segments", [])),
                        "status": "unlocked" if label in unlocked else "locked",
                    }
                )
            if topics:
                tree[root_name][sub_name] = topics
        # Remove empty root categories
        if not tree[root_name]:
            del tree[root_name]

    return {"tree": tree}


@router.post("/skills/{label}/unlock")
async def unlock_skill(label: str, user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """Placeholder unlock — immediately marks topic as unlocked in Supabase."""
    user = get_user(user_id)
    if not user.topic_groups:
        raise HTTPException(status_code=404, detail="No pipeline results yet")

    if label not in user.topic_groups:
        raise HTTPException(status_code=404, detail="Topic not found")

    entry = upsert_skill_progress(user_id, label, "unlocked")
    return {"label": label, "status": "unlocked", "entry": entry}
