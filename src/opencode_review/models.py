from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class SessionMeta:
    id: str
    parent_id: str | None
    created_at: datetime
    message_count: int
    has_children: bool
    title: str | None
    project_path: str | None


@dataclass
class NormalizedTurn:
    role: str
    content: str
    turn_index: int
    message_id: str
    has_tool_calls: bool
    tool_names: list[str] = field(default_factory=list)


@dataclass
class SemanticChunk:
    chunk_id: str
    session_id: str
    turns: list[NormalizedTurn]
    start_message_index: int
    end_message_index: int
    token_count: int = 0


@dataclass
class Recommendation:
    type: str
    title: str
    detail: str
    priority: str
    source_chunk_id: str
    status: str = "pending"


@dataclass
class ChunkAnalysis:
    chunk_id: str
    topic_summary: str
    outcome: Literal["success", "partial", "failure", "unclear"]
    first_try_success: bool
    corrections_required: int
    skills_referenced: list[str] = field(default_factory=list)
    kb_referenced: list[str] = field(default_factory=list)
    failure_root_cause: str | None = None
    failure_detail: str | None = None
    missing_context: str | None = None
    recommendations: list[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
