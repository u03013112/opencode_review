from __future__ import annotations

import re
import uuid
from datetime import datetime

from .models import NormalizedTurn, SemanticChunk

COMPLETION_SIGNALS = re.compile(
    r"(done|完成|let me know|已完成|如果.*需要|还需要.*吗|all set|that'?s it)",
    re.IGNORECASE,
)

TOPIC_SHIFT_SIGNALS = re.compile(
    r"(now let'?s|现在来|next|另外|顺便|不同的问题|can you also|还有一个|换个)",
    re.IGNORECASE,
)


def chunk_session(
    turns: list[NormalizedTurn],
    time_gap_minutes: int = 30,
    target_max_tokens: int = 50000,
    target_min_turns: int = 5,
) -> list[SemanticChunk]:
    if not turns:
        return []

    if len(turns) < 10:
        return [_make_chunk(turns, turns[0].message_id)]

    boundaries = _find_boundaries(turns, time_gap_minutes)
    chunks = _split_at_boundaries(turns, boundaries)

    # 合并 chunks 直到达到 target_max_tokens
    merged = _merge_by_token_budget(chunks, target_max_tokens, target_min_turns)

    return merged


def _find_boundaries(turns: list[NormalizedTurn], time_gap_minutes: int) -> list[int]:
    boundaries: list[int] = []

    for i in range(1, len(turns)):
        current = turns[i]
        prev = turns[i - 1]

        if current.role != "user":
            continue

        # 前一条 assistant 包含完成信号，当前是新 user message
        if prev.role == "assistant" and COMPLETION_SIGNALS.search(prev.content):
            boundaries.append(i)
            continue

        # 当前 user message 包含话题切换信号
        if TOPIC_SHIFT_SIGNALS.search(current.content):
            boundaries.append(i)
            continue

        # tool 模式突变：前一段全是某类 tool，现在完全不同
        if _tool_pattern_shift(turns, i):
            boundaries.append(i)
            continue

    return boundaries


def _tool_pattern_shift(turns: list[NormalizedTurn], idx: int) -> bool:
    lookback = 5
    start = max(0, idx - lookback)
    prev_tools = set()
    for t in turns[start:idx]:
        prev_tools.update(t.tool_names)

    lookahead = min(len(turns), idx + lookback)
    next_tools = set()
    for t in turns[idx:lookahead]:
        next_tools.update(t.tool_names)

    if not prev_tools or not next_tools:
        return False

    overlap = prev_tools & next_tools
    return len(overlap) == 0 and len(prev_tools) >= 2 and len(next_tools) >= 2


def _split_at_boundaries(
    turns: list[NormalizedTurn], boundaries: list[int]
) -> list[SemanticChunk]:
    if not boundaries:
        return [_make_chunk(turns, turns[0].message_id)]

    chunks: list[SemanticChunk] = []
    prev_idx = 0

    for boundary in boundaries:
        if boundary > prev_idx:
            chunk_turns = turns[prev_idx:boundary]
            chunks.append(_make_chunk(chunk_turns, chunk_turns[0].message_id))
        prev_idx = boundary

    if prev_idx < len(turns):
        chunk_turns = turns[prev_idx:]
        chunks.append(_make_chunk(chunk_turns, chunk_turns[0].message_id))

    return chunks


def _merge_by_token_budget(
    chunks: list[SemanticChunk], max_tokens: int, min_turns: int
) -> list[SemanticChunk]:
    if len(chunks) <= 1:
        return chunks

    merged: list[SemanticChunk] = []
    buffer: list[NormalizedTurn] = []
    buffer_tokens = 0

    for chunk in chunks:
        chunk_tokens = chunk.token_count
        if buffer and (buffer_tokens + chunk_tokens > max_tokens and len(buffer) >= min_turns):
            merged.append(_make_chunk(buffer, buffer[0].message_id))
            buffer = list(chunk.turns)
            buffer_tokens = chunk_tokens
        else:
            buffer.extend(chunk.turns)
            buffer_tokens += chunk_tokens

    if buffer:
        merged.append(_make_chunk(buffer, buffer[0].message_id))

    return merged


def _make_chunk(turns: list[NormalizedTurn], session_id: str) -> SemanticChunk:
    total_chars = sum(len(t.content) for t in turns)
    approx_tokens = total_chars // 3  # 中英混合粗略估算

    return SemanticChunk(
        chunk_id=str(uuid.uuid4())[:8],
        session_id=session_id,
        turns=turns,
        start_message_index=turns[0].turn_index,
        end_message_index=turns[-1].turn_index,
        token_count=approx_tokens,
    )
