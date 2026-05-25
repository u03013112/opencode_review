from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .models import (
    ChunkAnalysis,
    ChunkSummary,
    CorrectionDetail,
    NormalizedTurn,
    Recommendation,
    SemanticChunk,
)

ANALYSIS_PROMPT = """你正在分析一段 AI 编码助手会话的任务块。你的目标是提取每一次用户纠正的详细信息。

对话内容：
{conversation}

上下文：
- 总轮数: {turn_count}
- 使用的工具: {tool_summary}
- 会话日期: {date}

任务：逐条识别用户纠正（用户对 agent 输出不满、指出错误、要求修改的地方），并为每条纠正输出详情。

返回 JSON：
{{
  "topic": "一句话描述本块尝试完成的任务",
  "outcome": "success|partial|failure|unclear",
  "first_try_success": true|false,

  "corrections": [
    {{
      "id": "C1",
      "round_range": [起始轮, 结束轮],
      "error_type": "hallucination|missing_knowledge|misunderstood_requirement|wrong_approach|tool_failure|preference_adjustment",
      "severity": "blocking|degrading|cosmetic",
      "what_went_wrong": "Agent 做错了什么（具体描述，不要泛泛而谈）",
      "user_correction": "用户说了什么来纠正（引用或摘要用户原话）",
      "how_agent_fixed": "Agent 如何修复的",
      "fix_was_durable": true,
      "knowledge_gap": "如果适用，缺失的具体知识是什么"
    }}
  ],

  "chunk_summary": {{
    "blocking_count": 0,
    "degrading_count": 0,
    "cosmetic_count": 0,
    "primary_root_cause": "总结主要问题根因",
    "wasted_rounds_estimate": 0
  }},

  "recommendations": [
    {{
      "id": "R1",
      "triggered_by": ["C1", "C3"],
      "type": "add_kb_entry|add_skill|update_skill|process_change",
      "title": "简短标题",
      "what_to_add": "具体要新增/修改什么内容",
      "adoption_risk": "low|medium|high",
      "adoption_risk_reason": "为什么这个风险等级",
      "skip_if": "在什么情况下可以不采纳"
    }}
  ],

  "confidence": 0.0
}}

规则：
- 纠正 = 用户对 agent 行为的实质性否定或修正。包括：指出错误、要求重做、提供正确信息
- 偏好调整（"换个格式"、"表格太宽了"）归为 preference_adjustment + cosmetic
- error_type 分类：
  - hallucination: agent 编造了不存在的内容（表名、API、数据）
  - missing_knowledge: agent 缺乏项目/业务知识导致错误
  - misunderstood_requirement: agent 理解错了用户需求
  - wrong_approach: agent 选了错误的技术路径
  - tool_failure: 工具本身不稳定/报错导致需要重试
  - preference_adjustment: 非错误，仅是偏好微调
- severity 分类：
  - blocking: 阻断任务进展，必须修复才能继续
  - degrading: 不阻断但降低质量/效率
  - cosmetic: 纯表面问题
- fix_was_durable: 修复后同类错误是否在本块内再次出现（false=复现了）
- round_range: 粗略估计从哪一轮到哪一轮涉及这次纠正（基于对话位置）
- recommendations 的 triggered_by 必须引用具体的 correction id
- wasted_rounds_estimate: 因为纠正而额外消耗的对话轮数（粗略估计）
- 一次成功（无任何纠正）时 corrections 为空数组
- 最多输出 10 条 corrections，最多 5 条 recommendations
- 只返回 JSON，不要其他文字"""


class Analyzer:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0, max_tokens: int = 12000):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_chunk(self, chunk: SemanticChunk, session_date: str = "", retries: int = 3) -> ChunkAnalysis:
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

        last_error = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                break
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(10 * (attempt + 1))
        else:
            raise last_error  # type: ignore[misc]

        corrections = []
        for c in data.get("corrections", []):
            if isinstance(c, dict):
                corrections.append(CorrectionDetail(
                    id=c.get("id", "C?"),
                    round_range=c.get("round_range", [0, 0]),
                    error_type=c.get("error_type", "unknown"),
                    severity=c.get("severity", "degrading"),
                    what_went_wrong=c.get("what_went_wrong", ""),
                    user_correction=c.get("user_correction", ""),
                    how_agent_fixed=c.get("how_agent_fixed", ""),
                    fix_was_durable=c.get("fix_was_durable", True),
                    knowledge_gap=c.get("knowledge_gap"),
                ))

        chunk_summary_data = data.get("chunk_summary", {})
        chunk_summary = ChunkSummary(
            blocking_count=chunk_summary_data.get("blocking_count", 0),
            degrading_count=chunk_summary_data.get("degrading_count", 0),
            cosmetic_count=chunk_summary_data.get("cosmetic_count", 0),
            primary_root_cause=chunk_summary_data.get("primary_root_cause", ""),
            wasted_rounds_estimate=chunk_summary_data.get("wasted_rounds_estimate", 0),
        )

        recommendations = []
        for rec in data.get("recommendations", []):
            if isinstance(rec, dict):
                recommendations.append(Recommendation(
                    type=rec.get("type", "process_change"),
                    title=rec.get("title", ""),
                    detail=rec.get("what_to_add", ""),
                    priority="high" if rec.get("adoption_risk") == "low" else "medium",
                    source_chunk_id=chunk.chunk_id,
                    triggered_by=rec.get("triggered_by", []),
                    adoption_risk=rec.get("adoption_risk", "medium"),
                    adoption_risk_reason=rec.get("adoption_risk_reason", ""),
                    skip_if=rec.get("skip_if", ""),
                ))

        return ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            topic_summary=data.get("topic", ""),
            outcome=data.get("outcome", "unclear"),
            first_try_success=data.get("first_try_success", False),
            corrections_required=len(corrections),
            corrections=corrections,
            chunk_summary=chunk_summary,
            skills_referenced=data.get("skills_referenced", []),
            kb_referenced=data.get("kb_referenced", []),
            failure_root_cause=chunk_summary.primary_root_cause or None,
            failure_detail=None,
            missing_context=None,
            recommendations=recommendations,
            confidence=data.get("confidence", 0.0),
        )

    def analyze_chunks_concurrent(
        self, chunks: list[SemanticChunk], session_date: str = "", max_workers: int = 10
    ) -> list[ChunkAnalysis]:
        results: list[ChunkAnalysis | None] = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyze_chunk, chunk, session_date): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ChunkAnalysis(
                        chunk_id=chunks[idx].chunk_id,
                        topic_summary=f"[分析失败: {e}]",
                        outcome="unclear",
                        first_try_success=False,
                        corrections_required=0,
                        corrections=[],
                        chunk_summary=ChunkSummary(),
                        recommendations=[],
                        confidence=0.0,
                    )

        return results  # type: ignore[return-value]

    def _format_conversation(self, turns: list[NormalizedTurn]) -> str:
        lines = []
        for t in turns:
            role_label = {"user": "用户", "assistant": "AI"}.get(t.role, t.role)
            lines.append(f"[{role_label}] {t.content}")
        return "\n\n".join(lines)
