"""Pipeline API: start runs and stream progress via SSE."""

from __future__ import annotations

import asyncio
import contextlib
import json
import traceback
from collections.abc import AsyncGenerator, Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from models.schemas import (
    GraphData,
    GraphLink,
    GraphNode,
    PipelinePhase,
    PipelineStartRequest,
    PipelineStartResponse,
)
from services.pipeline.events import create_run, get_run, make_callback

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Keep references to background tasks so they aren't GC'd
_background_tasks: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Sensitive-topic filter (shared with main.py - imported at task time)
# ---------------------------------------------------------------------------


def _get_sensitive_filter() -> Callable[[str, list[str]], bool]:
    """Import the sensitive filter from main to avoid circular imports."""
    from main import _is_sensitive

    return _is_sensitive


# ---------------------------------------------------------------------------
# Pipeline async task
# ---------------------------------------------------------------------------


async def _run_pipeline_task(run_id: str, session_key: str, last_active_org: str) -> None:
    """Background task: fetch -> pipeline -> emit results via queue."""
    from services.claude_fetcher.fetch_all import async_fetch_conversations
    from services.pipeline.pipeline import run_pipeline_async

    run = get_run(run_id)
    if not run:
        return

    callback = make_callback(run)
    sensitive_filter = _get_sensitive_filter()

    try:
        # Phase 1: Fetch conversations
        callback(PipelinePhase.fetching, "Starting conversation fetch...", 0.0)
        conversations = await async_fetch_conversations(
            session_key=session_key,
            last_active_org=last_active_org,
            on_progress=callback,
        )

        if not conversations:
            callback(PipelinePhase.error, "No conversations fetched. Check session key.", 0.0)
            return

        callback(PipelinePhase.fetching, f"Fetched {len(conversations)} conversations", 1.0)

        # Phases 2-6: Pipeline
        topic_groups, hierarchy = await run_pipeline_async(
            conversations=conversations,
            on_progress=callback,
            sensitive_filter=sensitive_filter,
        )

        # Store per-run results
        run.topic_groups = topic_groups
        run.hierarchy = hierarchy

        # Build final graph snapshot for the frontend
        graph = _build_graph_data(topic_groups, hierarchy, sensitive_filter)
        callback(
            PipelinePhase.done,
            "Pipeline complete!",
            1.0,
            graph_snapshot=graph,
        )

        # Optionally persist to disk
        await _persist_results(topic_groups, hierarchy)

    except Exception as e:
        traceback.print_exc()
        callback(PipelinePhase.error, f"Pipeline error: {e}", 0.0)
    finally:
        run.is_complete = True
        # Sentinel to close SSE stream
        with contextlib.suppress(asyncio.QueueFull):
            run.queue.put_nowait(None)


_persist_lock = asyncio.Lock()


async def _persist_results(topic_groups: dict[str, Any], hierarchy: dict[str, Any]) -> None:
    """Save results to disk (optional, for legacy dev UI)."""
    import os

    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    async with _persist_lock:

        def _write() -> None:
            groups_path = os.path.join(data_dir, "topic_groups.json")
            hierarchy_path = os.path.join(data_dir, "topic_hierarchy.json")
            with open(groups_path, "w") as f:
                json.dump(topic_groups, f, indent=2, default=str)
            with open(hierarchy_path, "w") as f:
                json.dump(hierarchy, f, indent=2)

        await asyncio.to_thread(_write)


def _build_graph_data(
    topic_groups: dict[str, Any],
    hierarchy: dict[str, Any],
    sensitive_filter: Callable[[str, list[str]], bool],
) -> GraphData:
    """Build graph nodes/links from topic_groups + hierarchy."""
    nodes: list[GraphNode] = []
    links: list[GraphLink] = []

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
                if sensitive_filter(label, keywords):
                    continue
                count = len(info.get("segments", []))
                sub_seg_count += count
                sub_has_topics = True
                nodes.append(
                    GraphNode(
                        id=f"topic::{label}",
                        name=label,
                        level=2,
                        segment_count=count,
                        type="topic",
                        keywords=keywords,
                    )
                )
                links.append(GraphLink(source=sub_id, target=f"topic::{label}"))

            if sub_has_topics:
                nodes.append(
                    GraphNode(id=sub_id, name=sub_name, level=1, segment_count=sub_seg_count, type="subcategory")
                )
                links.append(GraphLink(source=root_id, target=sub_id))
                root_seg_count += sub_seg_count
                has_children = True

        if has_children:
            nodes.append(GraphNode(id=root_id, name=root_name, level=0, segment_count=root_seg_count, type="root"))

    return GraphData(nodes=nodes, links=links)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=PipelineStartResponse)
async def start_pipeline(req: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline run. Returns run_id for SSE streaming."""
    try:
        run = create_run()
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail="Too many concurrent pipeline runs. Try again later.") from exc

    task = asyncio.create_task(
        _run_pipeline_task(
            run_id=run.run_id,
            session_key=req.session_key,
            last_active_org=req.last_active_org,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return PipelineStartResponse(run_id=run.run_id)


@router.get("/stream/{run_id}")
async def stream_pipeline(run_id: str) -> EventSourceResponse:
    """SSE stream of pipeline progress events."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        while True:
            event = await run.queue.get()
            if event is None:
                break
            yield {
                "event": event.phase.value,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())
