from __future__ import annotations

from .models import ChunkAnalysis, SemanticChunk


def generate_session_report(
    session_id: str,
    title: str | None,
    date: str,
    message_count: int,
    project_path: str | None,
    chunks: list[SemanticChunk],
    analyses: list[ChunkAnalysis],
) -> str:
    lines = [
        f"# 会话分析: {title or session_id}",
        f"**日期:** {date} | **消息数:** {message_count} | **项目:** {project_path or '未知'}",
        "",
        "## 任务块",
        "",
    ]

    all_recommendations = []
    success_count = 0
    failure_count = 0

    for i, (chunk, analysis) in enumerate(zip(chunks, analyses), 1):
        outcome_icon = {
            "success": "✅ 成功",
            "partial": "⚠️ 部分成功",
            "failure": "❌ 失败",
            "unclear": "❓ 不明确",
        }.get(analysis.outcome, "❓")

        first_try = "（一次成功）" if analysis.first_try_success else ""
        if analysis.corrections_required > 0:
            first_try = f"（{analysis.corrections_required} 次纠正）"

        tools = ", ".join(sorted(set(t for turn in chunk.turns for t in turn.tool_names))) or "无"

        lines.extend([
            f"### 块 {i}: {analysis.topic_summary}",
            f"- **结果:** {outcome_icon}{first_try}",
            f"- **工具:** {tools}",
            f"- **轮数:** {len(chunk.turns)}",
            "",
        ])

        if analysis.failure_root_cause:
            lines.append(f"**根因:** {analysis.failure_root_cause} — {analysis.failure_detail or ''}")
        if analysis.missing_context:
            lines.append(f"**缺失内容:** {analysis.missing_context}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if analysis.outcome == "success":
            success_count += 1
        elif analysis.outcome == "failure":
            failure_count += 1

        all_recommendations.extend(analysis.recommendations)

    total = len(analyses)
    success_rate = f"{success_count}/{total}" if total > 0 else "N/A"

    lines.extend([
        "## 质量评分卡",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 分析块数 | {total} |",
        f"| 成功数 | {success_count} ({success_rate}) |",
        f"| 失败数 | {failure_count} |",
        "",
    ])

    if all_recommendations:
        lines.extend([
            "## 待确认建议",
            "",
        ])
        for j, rec in enumerate(all_recommendations, 1):
            type_label = {
                "new_skill": "新 Skill",
                "kb_entry": "知识库",
                "skill_update": "Skill 更新",
                "workflow": "流程优化",
            }.get(rec.type, rec.type)
            lines.append(f"{j}. **[{type_label}]** {rec.title}")
            lines.append(f"   {rec.detail}")
            lines.append("")

    return "\n".join(lines)
