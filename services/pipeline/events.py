"""Per-run pipeline state and progress event management."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from models.schemas import GraphData, GraphNode, PipelinePhase, PipelineProgressEvent

MAX_CONCURRENT_RUNS = 5
RUN_TTL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class PipelineRun:
    run_id: str
    queue: asyncio.Queue[PipelineProgressEvent | None] = field(default_factory=asyncio.Queue)
    is_complete: bool = False
    created_at: float = field(default_factory=time.time)
    # Per-run results — NOT stored in globals
    topic_groups: dict[str, Any] | None = None
    hierarchy: dict[str, Any] | None = None


# Active runs registry
_active_runs: dict[str, PipelineRun] = {}


def create_run() -> PipelineRun:
    """Create a new pipeline run. Raises RuntimeError if at capacity."""
    _cleanup_expired()
    active_count = sum(1 for r in _active_runs.values() if not r.is_complete)
    if active_count >= MAX_CONCURRENT_RUNS:
        raise RuntimeError(f"Max concurrent runs ({MAX_CONCURRENT_RUNS}) exceeded")
    run_id = uuid.uuid4().hex[:12]
    run = PipelineRun(run_id=run_id)
    _active_runs[run_id] = run
    return run


def get_run(run_id: str) -> PipelineRun | None:
    """Get a pipeline run by ID."""
    return _active_runs.get(run_id)


def make_callback(
    run: PipelineRun,
) -> Callable[..., None]:
    """Return a sync-safe callback that pushes events to the run's queue.

    The callback signature matches what pipeline steps expect:
        callback(phase, message, progress, node=None, graph_snapshot=None)
    """

    def callback(
        phase: PipelinePhase,
        message: str,
        progress: float,
        *,
        node: GraphNode | None = None,
        graph_snapshot: GraphData | None = None,
    ) -> None:
        event = PipelineProgressEvent(
            phase=phase,
            message=message,
            progress=progress,
            node=node,
            graph_snapshot=graph_snapshot,
        )
        with contextlib.suppress(asyncio.QueueFull):
            run.queue.put_nowait(event)

    return callback


def _cleanup_expired() -> None:
    """Remove completed runs older than TTL."""
    now = time.time()
    expired = [rid for rid, run in _active_runs.items() if run.is_complete and (now - run.created_at) > RUN_TTL_SECONDS]
    for rid in expired:
        del _active_runs[rid]
