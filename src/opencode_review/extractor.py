from __future__ import annotations

import json
import re

from .db import DB
from .models import NormalizedTurn


def extract_session(
    session_id: str,
    db: DB,
    offset: int = 0,
    max_reasoning_chars: int = 500,
    max_tool_summary_chars: int = 200,
) -> list[NormalizedTurn]:
    messages = db.get_messages(session_id, offset=offset)
    turns: list[NormalizedTurn] = []

    for idx, msg in enumerate(messages):
        parts = db.get_parts(msg["id"])
        content_pieces: list[str] = []
        tool_names: list[str] = []
        has_tool_calls = False

        for part in parts:
            ptype = part["type"]

            if ptype == "text":
                text = part.get("text") or ""
                if text:
                    content_pieces.append(text)

            elif ptype == "reasoning":
                text = part.get("text") or ""
                if text:
                    truncated = text[:max_reasoning_chars]
                    if len(text) > max_reasoning_chars:
                        truncated += "..."
                    content_pieces.append(f"[思考] {truncated}")

            elif ptype == "step-start":
                pass

            elif ptype == "step-finish":
                pass

            elif ptype == "tool":
                has_tool_calls = True
                tool_name = part.get("tool") or "unknown"
                tool_names.append(tool_name)
                summary = _summarize_tool_part(part, max_tool_summary_chars)
                if summary:
                    content_pieces.append(summary)

            elif ptype == "patch":
                raw = part.get("raw", {})
                summary = _summarize_patch(raw)
                if summary:
                    content_pieces.append(summary)

            elif ptype == "file":
                raw = part.get("raw", {})
                name = raw.get("name") or raw.get("filename") or "?"
                content_pieces.append(f"[文件] {name}")

            elif ptype == "compaction":
                text = part.get("text") or ""
                if text:
                    content_pieces.append(f"[摘要] {text[:300]}")

        content = "\n".join(content_pieces).strip()
        if not content:
            continue

        turns.append(NormalizedTurn(
            role=msg["role"],
            content=content,
            turn_index=offset + idx,
            message_id=msg["id"],
            has_tool_calls=has_tool_calls,
            tool_names=list(set(tool_names)),
        ))

    return turns


def _summarize_tool_part(part: dict, max_chars: int) -> str | None:
    tool_name = part.get("tool") or "unknown"
    state = part.get("state") or {}
    input_data = state.get("input", {})
    status = state.get("status", "")

    if tool_name == "bash":
        cmd = input_data.get("command", "")[:100]
        return f"bash({cmd}) → {status}"

    if tool_name in ("read", "Read"):
        fpath = input_data.get("filePath") or input_data.get("file") or ""
        return f"read({fpath}) → {status}"

    if tool_name in ("write", "Write"):
        fpath = input_data.get("filePath") or input_data.get("file") or ""
        return f"write({fpath}) → {status}"

    if tool_name == "edit":
        fpath = input_data.get("filePath") or ""
        return f"edit({fpath}) → {status}"

    if tool_name == "grep":
        pattern = input_data.get("pattern", "")[:50]
        return f"grep({pattern}) → {status}"

    if tool_name == "task":
        desc = input_data.get("description", "")[:60]
        subagent = input_data.get("subagent_type") or input_data.get("category") or ""
        return f"task({subagent}: {desc}) → {status}"

    if tool_name == "skill":
        name = input_data.get("name", "")
        return f"skill({name}) → {status}"

    # 通用 fallback
    input_str = json.dumps(input_data, ensure_ascii=False)[:max_chars] if input_data else ""
    return f"{tool_name}({input_str}) → {status}"


def _summarize_patch(raw: dict) -> str | None:
    fpath = raw.get("filePath") or raw.get("path") or "?"
    added = raw.get("added") or raw.get("additions") or 0
    removed = raw.get("removed") or raw.get("deletions") or 0
    return f"[patch] {fpath} (+{added}/-{removed})"
