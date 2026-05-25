from __future__ import annotations

from pathlib import Path

import click
import yaml

from .analyzer import Analyzer
from .chunker import chunk_session
from .db import DB
from .extractor import extract_session
from .reporter import generate_session_report
from .state import StateManager


def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        click.echo(f"配置文件不存在: {config_path}", err=True)
        raise SystemExit(1)
    return yaml.safe_load(p.read_text())


_cfg_cache: dict | None = None
_cfg_path: str = "config.yaml"


def get_config(ctx=None) -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config(_cfg_path)
    return _cfg_cache


@click.group()
@click.option("--config", "cfg_path", default="config.yaml", help="配置文件路径")
def cli(cfg_path):
    global _cfg_path, _cfg_cache
    _cfg_path = cfg_path
    _cfg_cache = None


@cli.command("list")
@click.option("--limit", default=20, help="显示数量")
@click.option("--since", default=None, type=int, help="最近 N 天")
@click.option("--unanalyzed-only", is_flag=True, help="只显示未分析的")
def list_sessions(limit, since, unanalyzed_only):
    cfg = get_config()
    db = DB(cfg["db_path"])
    state = StateManager(cfg["state_dir"])

    sessions = db.list_sessions(limit=limit, since_days=since)

    for s in sessions:
        analyzed = "✅" if state.get_session_state(s.id) else "⬜"
        if unanalyzed_only and state.get_session_state(s.id):
            continue
        title = (s.title or "")[:50]
        click.echo(f"{analyzed} {s.id[:16]} | {s.created_at:%Y-%m-%d %H:%M} | {s.message_count:>4} msgs | {title}")


@cli.command()
@click.argument("session_ids", nargs=-1)
@click.option("--all", "analyze_all", is_flag=True, help="分析所有未处理的")
@click.option("--since", default=None, type=int, help="最近 N 天")
def analyze(session_ids, analyze_all, since):
    cfg = get_config()
    db = DB(cfg["db_path"])
    state = StateManager(cfg["state_dir"])
    llm_cfg = cfg["llm"]
    analyzer = Analyzer(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0),
        max_tokens=llm_cfg.get("max_tokens", 4000),
    )

    if analyze_all or since:
        sessions = db.list_sessions(limit=9999, since_days=since)
        target_ids = [s.id for s in sessions]
    elif session_ids:
        target_ids = list(session_ids)
    else:
        click.echo("请指定 session_id 或使用 --all / --since", err=True)
        return

    extraction_cfg = cfg.get("extraction", {})
    chunking_cfg = cfg.get("chunking", {})
    output_dir = Path(cfg["output_dir"]) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    for sid in target_ids:
        msg_count = db.get_message_count(sid)

        if msg_count < extraction_cfg.get("skip_sessions_under_messages", 5):
            click.echo(f"跳过 {sid[:16]} (消息数 {msg_count} < 5)")
            continue

        needs_analysis, offset = state.should_analyze(sid, msg_count)
        if not needs_analysis:
            click.echo(f"跳过 {sid[:16]} (无变化)")
            continue

        click.echo(f"分析 {sid[:16]} (msgs {offset}→{msg_count})...")

        turns = extract_session(
            sid, db, offset=offset,
            max_reasoning_chars=extraction_cfg.get("max_reasoning_chars", 500),
            max_tool_summary_chars=extraction_cfg.get("max_tool_summary_chars", 200),
        )

        if not turns:
            click.echo(f"  无有效内容，跳过")
            continue

        chunks = chunk_session(
            turns,
            time_gap_minutes=chunking_cfg.get("time_gap_boundary_minutes", 30),
            target_max_tokens=chunking_cfg.get("target_max_tokens", 6000),
            target_min_turns=chunking_cfg.get("target_min_turns", 3),
        )

        click.echo(f"  切割为 {len(chunks)} 个块")

        sessions_meta = db.list_sessions(limit=1)
        session_date = ""
        for sm in db.list_sessions(limit=9999):
            if sm.id == sid:
                session_date = sm.created_at.strftime("%Y-%m-%d")
                session_title = sm.title
                session_project = sm.project_path
                break
        else:
            session_title = None
            session_project = None

        max_workers = chunking_cfg.get("max_concurrent_analysis", 10)
        click.echo(f"  并发分析 {len(chunks)} 个块 (workers={max_workers})...")
        analyses = analyzer.analyze_chunks_concurrent(chunks, session_date=session_date, max_workers=max_workers)

        report = generate_session_report(
            session_id=sid,
            title=session_title,
            date=session_date,
            message_count=msg_count,
            project_path=session_project,
            chunks=chunks,
            analyses=analyses,
        )

        report_path = output_dir / f"{sid}.md"
        report_path.write_text(report)
        click.echo(f"  报告已保存: {report_path}")

        outcome_summary = {}
        for a in analyses:
            outcome_summary[a.outcome] = outcome_summary.get(a.outcome, 0) + 1

        state.mark_analyzed(
            session_id=sid,
            message_index=msg_count,
            chunk_count=len(chunks),
            report_path=str(report_path),
            outcome_summary=outcome_summary,
        )


@cli.command()
@click.argument("session_id", required=False)
def report(session_id):
    cfg = get_config()
    output_dir = Path(cfg["output_dir"]) / "reports"

    if session_id:
        report_path = output_dir / f"{session_id}.md"
        if not report_path.exists():
            click.echo(f"报告不存在: {report_path}", err=True)
            return
        click.echo(report_path.read_text())
    else:
        reports = sorted(output_dir.glob("*.md"))
        click.echo(f"共 {len(reports)} 份报告:")
        for r in reports[-10:]:
            click.echo(f"  {r.stem}")


@cli.command()
def status():
    cfg = get_config()
    db = DB(cfg["db_path"])
    state = StateManager(cfg["state_dir"])

    total_sessions = len(db.list_sessions(limit=9999))
    processed = state.processed_count
    click.echo(f"总 sessions: {total_sessions}")
    click.echo(f"已分析: {processed}")
    click.echo(f"待分析: {total_sessions - processed}")


@cli.command()
@click.argument("session_id", required=False)
@click.option("--all", "reset_all", is_flag=True)
def reset(session_id, reset_all):
    cfg = get_config()
    state = StateManager(cfg["state_dir"])

    if reset_all:
        state.reset_all()
        click.echo("已重置所有分析状态")
    elif session_id:
        state.reset_session(session_id)
        click.echo(f"已重置 {session_id}")
    else:
        click.echo("请指定 session_id 或 --all", err=True)
