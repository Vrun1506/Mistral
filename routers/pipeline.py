"""Pipeline API: start runs and stream progress via SSE."""

from __future__ import annotations

import asyncio
import contextlib
import json
import traceback
from collections.abc import AsyncGenerator, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from auth import get_current_user_id
from models.schemas import (
    GraphData,
    GraphLink,
    GraphNode,
    PipelineContinueRequest,
    PipelinePhase,
    PipelineStartRequest,
    PipelineStartResponse,
    ScanResult,
)
from services.pipeline.events import create_run, get_run, make_callback
from store import get_user

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


def _apply_category_filter(
    conversations: list[dict[str, Any]],
    scan_result: ScanResult,
    excluded_categories: list[str],
) -> list[dict[str, Any]]:
    """Remove conversations whose flags intersect the excluded categories."""
    if not excluded_categories:
        return conversations

    excluded_set = set(excluded_categories)
    excluded_uuids: set[str] = set()
    for uuid, flags in scan_result.conversation_flags.items():
        if excluded_set.intersection(flags):
            excluded_uuids.add(uuid)

    return [c for c in conversations if c["uuid"] not in excluded_uuids]


async def _run_pipeline_task(
    run_id: str, session_key: str, last_active_org: str, max_conversations: int = 2000, user_id: str | None = None
) -> None:
    """Background task: fetch -> scan -> (review) -> pipeline -> emit results via queue."""
    from services.claude_fetcher.fetch_all import async_fetch_conversations
    from services.pipeline.pipeline import run_pipeline_async
    from services.privacy.scanner import scan_conversations

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
            max_conversations=max_conversations,
        )

        if not conversations:
            callback(PipelinePhase.error, "No conversations fetched. Check session key.", 0.0)
            return

        callback(PipelinePhase.fetching, f"Fetched {len(conversations)} conversations", 1.0)

        # Phase 2: Privacy scan
        callback(PipelinePhase.scanning, "Starting privacy scan...", 0.0)
        scan_result = await scan_conversations(conversations, on_progress=callback)
        run.scan_result = scan_result

        # If flagged conversations found, pause for user review
        if scan_result.flagged_conversations > 0:
            run.conversations = conversations
            callback(
                PipelinePhase.awaiting_review,
                f"{scan_result.flagged_conversations} conversations flagged — review required",
                1.0,
                scan_result=scan_result,
            )

            # Wait for user to submit review, with keepalive to prevent SSE timeout
            while not run.review_event.is_set():
                try:
                    await asyncio.wait_for(
                        run.review_event.wait(),
                        timeout=15.0,
                    )
                except TimeoutError:
                    # Send keepalive to keep SSE connection alive
                    callback(
                        PipelinePhase.awaiting_review,
                        "Waiting for review...",
                        1.0,
                    )

            # Apply exclusions
            conversations = _apply_category_filter(conversations, scan_result, run.excluded_categories)
            run.conversations = None  # free memory

            if not conversations:
                callback(PipelinePhase.error, "All conversations were excluded. Nothing to process.", 0.0)
                return

            callback(
                PipelinePhase.scanning,
                f"Proceeding with {len(conversations)} conversations after filtering",
                1.0,
            )
        else:
            callback(PipelinePhase.scanning, "No sensitive content detected — skipping review", 1.0)

        # Phases 3-7: Pipeline (embedding, segmenting, clustering, labeling, hierarchy)
        topic_groups, hierarchy = await run_pipeline_async(
            conversations=conversations,
            on_progress=callback,
            sensitive_filter=sensitive_filter,
        )

        # Store per-run results
        run.topic_groups = topic_groups
        run.hierarchy = hierarchy

        # Store in user store so /api/topic/{label} and /api/tree work
        if user_id:
            user = get_user(user_id)
            user.set_pipeline_results(topic_groups, hierarchy)

        # Build final graph snapshot for the frontend
        graph = _build_graph_data(topic_groups, hierarchy, sensitive_filter)
        callback(
            PipelinePhase.done,
            "Pipeline complete!",
            1.0,
            graph_snapshot=graph,
        )

        # Optionally persist to disk
        await _persist_results(topic_groups, hierarchy, user_id)

        # Fire-and-forget: generate notes in background
        if user_id:
            from routers.notes import generate_all_notes

            notes_task = asyncio.create_task(generate_all_notes(user))
            _background_tasks.add(notes_task)
            notes_task.add_done_callback(_background_tasks.discard)

    except Exception as e:
        traceback.print_exc()
        callback(PipelinePhase.error, f"Pipeline error: {e}", 0.0)
    finally:
        run.is_complete = True
        # Sentinel to close SSE stream
        with contextlib.suppress(asyncio.QueueFull):
            run.queue.put_nowait(None)


_persist_lock = asyncio.Lock()


async def _persist_results(topic_groups: dict[str, Any], hierarchy: dict[str, Any], user_id: str | None = None) -> None:
    """Save results to disk, namespaced by user_id."""
    import os

    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    suffix = f"_{user_id}" if user_id else ""

    async with _persist_lock:

        def _write() -> None:
            groups_path = os.path.join(data_dir, f"topic_groups{suffix}.json")
            hierarchy_path = os.path.join(data_dir, f"topic_hierarchy{suffix}.json")
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
async def start_pipeline(
    req: PipelineStartRequest, user_id: str = Depends(get_current_user_id)
) -> PipelineStartResponse:
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
            max_conversations=req.max_conversations,
            user_id=user_id,
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


@router.post("/continue")
async def continue_pipeline(req: PipelineContinueRequest) -> dict[str, str]:
    """Resume a pipeline run after user reviews privacy scan results."""
    run = get_run(req.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    if run.review_continued:
        raise HTTPException(status_code=409, detail="Pipeline has already been continued")

    run.review_continued = True
    run.excluded_categories = req.excluded_categories
    run.review_event.set()
    return {"status": "continued"}
