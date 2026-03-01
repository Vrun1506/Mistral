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
    scanning = "scanning"
    awaiting_review = "awaiting_review"
    embedding = "embedding"
    segmenting = "segmenting"
    clustering = "clustering"
    labeling = "labeling"
    hierarchy = "hierarchy"
    done = "done"
    error = "error"


# ---------------------------------------------------------------------------
# Privacy scanning types
# ---------------------------------------------------------------------------


class PrivacyCategory(BaseModel):
    id: str
    name: str
    source: str  # "gliner" | "nemoguard"
    conversation_count: int = 0
    conversation_uuids: list[str] = Field(default_factory=list)


class ScanResult(BaseModel):
    total_conversations: int = 0
    flagged_conversations: int = 0
    categories: list[PrivacyCategory] = Field(default_factory=list)
    conversation_flags: dict[str, list[str]] = Field(default_factory=dict)  # uuid -> [category_ids]


class PipelineContinueRequest(BaseModel):
    run_id: str
    excluded_categories: list[str] = Field(default_factory=list)


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
    scan_result: ScanResult | None = None


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class PipelineStartRequest(BaseModel):
    session_key: str
    last_active_org: str
    max_conversations: int = 2000


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
