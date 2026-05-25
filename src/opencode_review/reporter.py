from __future__ import annotations

from .models import ChunkAnalysis, CorrectionDetail, Recommendation, SemanticChunk


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
    ]

    total_wasted = sum(
        (a.chunk_summary.wasted_rounds_estimate if a.chunk_summary else 0) for a in analyses
    )
    total_corrections = sum(a.corrections_required for a in analyses)
    if total_corrections > 0:
        lines.append(f"**总纠正次数:** {total_corrections} | **估计浪费轮数:** ~{total_wasted}")
        lines.append("")

    lines.extend(["## 任务块", ""])

    all_recommendations: list[tuple[int, Recommendation]] = []
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

        lines.extend([
            f"### 块 {i}: {analysis.topic_summary}",
            f"**状态:** {outcome_icon}{first_try} | **轮数:** {len(chunk.turns)}",
        ])

        if analysis.chunk_summary and analysis.chunk_summary.wasted_rounds_estimate > 0:
            lines.append(f"**浪费轮数估计:** ~{analysis.chunk_summary.wasted_rounds_estimate}")

        if analysis.chunk_summary and analysis.chunk_summary.primary_root_cause:
            lines.append(f"**主要根因:** {analysis.chunk_summary.primary_root_cause}")

        lines.append("")

        if analysis.corrections:
            lines.append("#### 纠正记录")
            lines.append("")
            for c in analysis.corrections:
                severity_icon = {"blocking": "🔴", "degrading": "🟡", "cosmetic": "⚪"}.get(c.severity, "⚪")
                error_label = {
                    "hallucination": "幻觉",
                    "missing_knowledge": "知识缺失",
                    "misunderstood_requirement": "需求理解错误",
                    "wrong_approach": "方法选错",
                    "tool_failure": "工具故障",
                    "preference_adjustment": "偏好微调",
                }.get(c.error_type, c.error_type)

                durable = "✅ 未复现" if c.fix_was_durable else "❌ 后续复现"

                lines.extend([
                    f"##### {c.id} · {error_label} {severity_icon} · 第 {c.round_range[0]}-{c.round_range[1]} 轮",
                    "",
                    f"| | |",
                    f"|---|---|",
                    f"| **出错** | {c.what_went_wrong} |",
                    f"| **用户纠正** | {c.user_correction} |",
                    f"| **Agent 修复** | {c.how_agent_fixed} |",
                    f"| **是否持久** | {durable} |",
                ])
                if c.knowledge_gap:
                    lines.append(f"| **知识缺口** | {c.knowledge_gap} |")
                lines.append("")

        if analysis.recommendations:
            lines.append("#### 建议")
            lines.append("")
            for rec in analysis.recommendations:
                risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(rec.adoption_risk, "🟡")
                type_label = {
                    "add_kb_entry": "新增 KB",
                    "add_skill": "新增 Skill",
                    "update_skill": "更新 Skill",
                    "process_change": "流程优化",
                }.get(rec.type, rec.type)

                triggered = ", ".join(rec.triggered_by) if rec.triggered_by else "整体"

                lines.extend([
                    f"**{rec.title}** [{type_label}]",
                    f"- 触发: {triggered}",
                    f"- 内容: {rec.detail}",
                    f"- 采纳风险: {risk_icon} {rec.adoption_risk} — {rec.adoption_risk_reason}",
                ])
                if rec.skip_if:
                    lines.append(f"- 跳过条件: {rec.skip_if}")
                lines.append("")

            all_recommendations.extend((i, rec) for rec in analysis.recommendations)

        lines.extend(["---", ""])

        if analysis.outcome == "success":
            success_count += 1
        elif analysis.outcome == "failure":
            failure_count += 1

    total = len(analyses)
    lines.extend([
        "## 质量评分卡",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 分析块数 | {total} |",
        f"| 一次成功 | {sum(1 for a in analyses if a.first_try_success)} |",
        f"| 部分成功 | {sum(1 for a in analyses if a.outcome == 'partial')} |",
        f"| 失败 | {failure_count} |",
        f"| 总纠正数 | {total_corrections} |",
        f"| 总浪费轮数 | ~{total_wasted} |",
        "",
    ])

    if all_recommendations:
        lines.extend(["## 全部建议汇总", ""])
        lines.append("| # | 块 | 类型 | 标题 | 触发 | 风险 |")
        lines.append("|---|---|------|------|------|------|")
        for j, (chunk_idx, rec) in enumerate(all_recommendations, 1):
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(rec.adoption_risk, "🟡")
            type_label = {
                "add_kb_entry": "KB",
                "add_skill": "Skill",
                "update_skill": "Skill↑",
                "process_change": "流程",
            }.get(rec.type, rec.type)
            triggered = ",".join(rec.triggered_by) if rec.triggered_by else "-"
            lines.append(f"| {j} | 块{chunk_idx} | {type_label} | {rec.title} | {triggered} | {risk_icon} |")
        lines.append("")

    return "\n".join(lines)
