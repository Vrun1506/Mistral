"""Graph data API — serves node/link JSON for 3d-force-graph."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.pipeline.events import get_run

router = APIRouter(tags=["graph"])


@router.get("/api/graph-data")
async def graph_data(run_id: str | None = None) -> JSONResponse:
    """Return topic hierarchy as flat nodes/links for 3d-force-graph.

    If run_id is provided, returns per-run data.
    Otherwise falls back to global data (legacy dev UI).
    Sensitive topics are filtered in both cases.
    """
    from main import HIERARCHY, TOPIC_GROUPS, _is_sensitive

    topic_groups: dict[str, Any]
    hierarchy: dict[str, Any]

    if run_id:
        run = get_run(run_id)
        if run and run.topic_groups and run.hierarchy:
            topic_groups = run.topic_groups
            hierarchy = run.hierarchy
        else:
            return JSONResponse({"nodes": [], "links": []})
    else:
        topic_groups = TOPIC_GROUPS
        hierarchy = HIERARCHY

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
