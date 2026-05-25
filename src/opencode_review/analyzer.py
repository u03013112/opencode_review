from __future__ import annotations

import json

from openai import OpenAI

from .models import ChunkAnalysis, NormalizedTurn, Recommendation, SemanticChunk

ANALYSIS_PROMPT = """你正在分析一段 AI 编码助手会话的任务块。

对话内容：
{conversation}

上下文：
- 总轮数: {turn_count}
- 使用的工具: {tool_summary}
- 会话日期: {date}

任务：分析此块并返回 JSON：
{{
  "topic": "一句话描述尝试做什么",
  "outcome": "success|partial|failure|unclear",
  "first_try_success": true|false,
  "corrections_required": 0,
  "skills_referenced": [],
  "kb_referenced": [],
  "failure_root_cause": null,
  "failure_detail": null,
  "missing_context": null,
  "recommendations": [],
  "confidence": 0.0
}}

recommendations 格式（最多 3 条）：
{{
  "type": "new_skill|kb_entry|skill_update|workflow",
  "title": "简短标题",
  "detail": "具体建议",
  "priority": "high|medium|low"
}}

规则：
- first_try_success = true 仅当 agent 完成任务且用户没有发送纠正
- 用户纠正包括："不对"、"我是说"、"改一下"、"actually"、重述同样的请求
- outcome = "success" 要求任务完全完成且用户满意
- outcome = "partial" = 最终完成但经过纠正
- outcome = "failure" = 从未完成或用户放弃
- missing_context 要具体
- 忽略风格偏好 — 只标记实质性失败
- 只返回 JSON，不要其他文字"""


class Analyzer:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0, max_tokens: int = 4000):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_chunk(self, chunk: SemanticChunk, session_date: str = "") -> ChunkAnalysis:
        conversation = self._format_conversation(chunk.turns)
        tools_used = set()
        for t in chunk.turns:
            tools_used.update(t.tool_names)

        prompt = ANALYSIS_PROMPT.format(
            conversation=conversation,
            turn_count=len(chunk.turns),
            tool_summary=", ".join(sorted(tools_used)) or "无",
            date=session_date,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        recommendations = []
        for rec in data.get("recommendations", []):
            if isinstance(rec, dict):
                recommendations.append(Recommendation(
                    type=rec.get("type", "workflow"),
                    title=rec.get("title", ""),
                    detail=rec.get("detail", ""),
                    priority=rec.get("priority", "medium"),
                    source_chunk_id=chunk.chunk_id,
                ))

        return ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            topic_summary=data.get("topic", ""),
            outcome=data.get("outcome", "unclear"),
            first_try_success=data.get("first_try_success", False),
            corrections_required=data.get("corrections_required", 0),
            skills_referenced=data.get("skills_referenced", []),
            kb_referenced=data.get("kb_referenced", []),
            failure_root_cause=data.get("failure_root_cause"),
            failure_detail=data.get("failure_detail"),
            missing_context=data.get("missing_context"),
            recommendations=recommendations,
            confidence=data.get("confidence", 0.0),
        )

    def _format_conversation(self, turns: list[NormalizedTurn]) -> str:
        lines = []
        for t in turns:
            role_label = {"user": "用户", "assistant": "AI"}.get(t.role, t.role)
            lines.append(f"[{role_label}] {t.content}")
        return "\n\n".join(lines)
