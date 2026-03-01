"""Topic browsing JSON API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user_id
from models.schemas import TopicDetail, TreeRoot, TreeSubcategory, TreeTopic
from store import get_user

router = APIRouter(prefix="/api", tags=["topics"])


@router.get("/tree", response_model=list[TreeRoot])
async def get_tree(user_id: str = Depends(get_current_user_id)) -> list[TreeRoot]:
    """Return the full topic tree as JSON (sensitive topics excluded)."""
    from main import _is_sensitive

    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        return []

    tree: list[TreeRoot] = []
    for root_name, subcats in user.hierarchy.items():
        root_seg = 0
        subs: list[TreeSubcategory] = []
        for sub_name, labels in subcats.items():
            topics: list[TreeTopic] = []
            sub_seg = 0
            for label in labels:
                info: dict[str, Any] = user.topic_groups.get(label, {})  # type: ignore[assignment]
                if not info:
                    continue
                keywords = info.get("keywords", [])[:5]
                if _is_sensitive(label, keywords):
                    continue
                count = len(info.get("segments", []))
                topics.append(TreeTopic(label=label, keywords=keywords, segment_count=count))
                sub_seg += count
            if topics:
                subs.append(TreeSubcategory(name=sub_name, topics=topics, segment_count=sub_seg))
                root_seg += sub_seg
        if subs:
            tree.append(TreeRoot(name=root_name, subcategories=subs, segment_count=root_seg))
    return tree


@router.get("/topic/{label}", response_model=TopicDetail)
async def get_topic(label: str, user_id: str = Depends(get_current_user_id)) -> TopicDetail:
    """Return topic detail as JSON (404 if not found or sensitive)."""
    from main import _is_sensitive

    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        raise HTTPException(status_code=404, detail="No pipeline results yet")

    info = user.topic_groups.get(label)
    if not info:
        raise HTTPException(status_code=404, detail="Topic not found")
    keywords = info.get("keywords", [])[:5]
    if _is_sensitive(label, keywords):
        raise HTTPException(status_code=404, detail="Topic not found")

    root_cat = None
    sub_cat = None
    for root_name, subcats in user.hierarchy.items():
        for sub_name, labels in subcats.items():
            if label in labels:
                root_cat = root_name
                sub_cat = sub_name
                break
        if root_cat:
            break

    return TopicDetail(
        label=label,
        keywords=info["keywords"],
        segments=info["segments"],  # type: ignore[arg-type]
        root_cat=root_cat,
        sub_cat=sub_cat,
    )
