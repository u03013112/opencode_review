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
class CorrectionDetail:
    id: str
    round_range: list[int]
    error_type: str
    severity: str
    what_went_wrong: str
    user_correction: str
    how_agent_fixed: str
    fix_was_durable: bool
    knowledge_gap: str | None = None


@dataclass
class Recommendation:
    type: str
    title: str
    detail: str
    priority: str
    source_chunk_id: str
    triggered_by: list[str] = field(default_factory=list)
    adoption_risk: str = "medium"
    adoption_risk_reason: str = ""
    skip_if: str = ""
    status: str = "pending"


@dataclass
class ChunkSummary:
    blocking_count: int = 0
    degrading_count: int = 0
    cosmetic_count: int = 0
    primary_root_cause: str = ""
    wasted_rounds_estimate: int = 0


@dataclass
class ChunkAnalysis:
    chunk_id: str
    topic_summary: str
    outcome: Literal["success", "partial", "failure", "unclear"]
    first_try_success: bool
    corrections_required: int
    corrections: list[CorrectionDetail] = field(default_factory=list)
    chunk_summary: ChunkSummary | None = None
    skills_referenced: list[str] = field(default_factory=list)
    kb_referenced: list[str] = field(default_factory=list)
    failure_root_cause: str | None = None
    failure_detail: str | None = None
    missing_context: str | None = None
    recommendations: list[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
