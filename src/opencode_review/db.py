from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import SessionMeta


class DB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def list_sessions(
        self,
        limit: int = 50,
        since_days: int | None = None,
        parent_only: bool = True,
        project_path: str | None = None,
    ) -> list[SessionMeta]:
        conn = self._connect()
        try:
            query = "SELECT id, parent_id, title, directory, time_created FROM session WHERE 1=1"
            params: list = []

            if parent_only:
                query += " AND parent_id IS NULL"

            if since_days:
                cutoff_ms = int((datetime.now().timestamp() - since_days * 86400) * 1000)
                query += " AND time_created > ?"
                params.append(cutoff_ms)

            if project_path:
                query += " AND directory = ?"
                params.append(project_path)

            query += " ORDER BY time_created DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            sessions = []
            for row in rows:
                msg_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM message WHERE session_id = ?",
                    (row["id"],),
                ).fetchone()["cnt"]

                has_children = conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM session WHERE parent_id = ?)",
                    (row["id"],),
                ).fetchone()[0]

                created_at = datetime.fromtimestamp(row["time_created"] / 1000)

                sessions.append(SessionMeta(
                    id=row["id"],
                    parent_id=row["parent_id"],
                    created_at=created_at,
                    message_count=msg_count,
                    has_children=bool(has_children),
                    title=row["title"] or None,
                    project_path=row["directory"] or None,
                ))
            return sessions
        finally:
            conn.close()

    def get_message_count(self, session_id: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM message WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def get_messages(
        self, session_id: str, offset: int = 0, limit: int | None = None
    ) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, session_id, time_created, data FROM message WHERE session_id = ? ORDER BY time_created ASC",
                (session_id,),
            ).fetchall()

            messages = []
            for r in rows:
                data = json.loads(r["data"])
                messages.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "time_created": r["time_created"],
                    "role": data.get("role", "unknown"),
                    "data": data,
                })

            if offset:
                messages = messages[offset:]
            if limit:
                messages = messages[:limit]

            return messages
        finally:
            conn.close()

    def get_parts(self, message_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, data FROM part WHERE message_id = ? ORDER BY time_created ASC",
                (message_id,),
            ).fetchall()

            parts = []
            for r in rows:
                data = json.loads(r["data"])
                parts.append({
                    "id": r["id"],
                    "type": data.get("type", "unknown"),
                    "text": data.get("text"),
                    "tool": data.get("tool"),
                    "state": data.get("state"),
                    "raw": data,
                })
            return parts
        finally:
            conn.close()

    def get_child_sessions(self, parent_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id FROM session WHERE parent_id = ?",
                (parent_id,),
            ).fetchall()
            return [r["id"] for r in rows]
        finally:
            conn.close()

    def get_last_assistant_text(self, session_id: str) -> str | None:
        conn = self._connect()
        try:
            msgs = conn.execute(
                "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created DESC",
                (session_id,),
            ).fetchall()

            for msg in msgs:
                msg_data = json.loads(msg["data"])
                if msg_data.get("role") != "assistant":
                    continue
                parts = conn.execute(
                    "SELECT data FROM part WHERE message_id = ? ORDER BY time_created ASC",
                    (msg["id"],),
                ).fetchall()
                for p in parts:
                    pdata = json.loads(p["data"])
                    if pdata.get("type") == "text" and pdata.get("text"):
                        return pdata["text"][:200]
            return None
        finally:
            conn.close()
