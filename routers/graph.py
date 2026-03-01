"""Graph data API — serves node/link JSON for 3d-force-graph."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from auth import get_current_user_id
from store import get_user

router = APIRouter(tags=["graph"])


@router.get("/api/graph-data")
async def graph_data(user_id: str = Depends(get_current_user_id)) -> JSONResponse:
    """Return topic hierarchy as flat nodes/links for 3d-force-graph."""
    from main import _is_sensitive

    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        return JSONResponse({"nodes": [], "links": []})

    topic_groups = user.topic_groups
    hierarchy = user.hierarchy

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []

    for root_name, subcats in hierarchy.items():
        root_id = f"root::{root_name}"
        root_seg_count = 0
        has_children = False
        for sub_name, labels in subcats.items():
            sub_id = f"sub::{sub_name}"
            sub_seg_count = 0
            sub_has_topics = False
            for label in labels:
                info = topic_groups.get(label, {})
                keywords = info.get("keywords", [])[:5]
                if _is_sensitive(label, keywords):
                    continue
                count = len(info.get("segments", []))
                sub_seg_count += count
                sub_has_topics = True
                nodes.append(
                    {
                        "id": f"topic::{label}",
                        "name": label,
                        "level": 2,
                        "segment_count": count,
                        "type": "topic",
                        "keywords": keywords,
                    }
                )
                links.append({"source": sub_id, "target": f"topic::{label}"})
            if sub_has_topics:
                nodes.append(
                    {
                        "id": sub_id,
                        "name": sub_name,
                        "level": 1,
                        "segment_count": sub_seg_count,
                        "type": "subcategory",
                    }
                )
                links.append({"source": root_id, "target": sub_id})
                root_seg_count += sub_seg_count
                has_children = True
        if has_children:
            nodes.append(
                {
                    "id": root_id,
                    "name": root_name,
                    "level": 0,
                    "segment_count": root_seg_count,
                    "type": "root",
                }
            )

    return JSONResponse({"nodes": nodes, "links": links})
