"""Pydantic models for API request/response types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------


class PipelinePhase(StrEnum):
    fetching = "fetching"
    embedding = "embedding"
    segmenting = "segmenting"
    clustering = "clustering"
    labeling = "labeling"
    hierarchy = "hierarchy"
    done = "done"
    error = "error"


# ---------------------------------------------------------------------------
# Graph data types
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    name: str
    level: int
    segment_count: int = 0
    type: str  # "root" | "subcategory" | "topic"
    keywords: list[str] = Field(default_factory=list)


class GraphLink(BaseModel):
    source: str
    target: str


class GraphData(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE progress events
# ---------------------------------------------------------------------------


class PipelineProgressEvent(BaseModel):
    phase: PipelinePhase
    message: str
    progress: float = 0.0  # 0.0 - 1.0
    node: GraphNode | None = None
    graph_snapshot: GraphData | None = None


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class PipelineStartRequest(BaseModel):
    session_key: str
    last_active_org: str


class PipelineStartResponse(BaseModel):
    run_id: str
    status: str = "started"


class TreeTopic(BaseModel):
    label: str
    keywords: list[str]
    segment_count: int


class TreeSubcategory(BaseModel):
    name: str
    topics: list[TreeTopic]
    segment_count: int


class TreeRoot(BaseModel):
    name: str
    subcategories: list[TreeSubcategory]
    segment_count: int


class TopicDetail(BaseModel):
    label: str
    keywords: list[str]
    segments: list[dict[str, Any]]
    root_cat: str | None = None
    sub_cat: str | None = None
