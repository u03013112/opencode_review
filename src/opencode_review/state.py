from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class StateManager:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "processed.json"
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"schema_version": 2, "last_run": None, "sessions": {}}

    def save(self):
        self._state["last_run"] = datetime.now().isoformat()
        self.state_file.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))

    def get_session_state(self, session_id: str) -> dict | None:
        return self._state["sessions"].get(session_id)

    def should_analyze(self, session_id: str, current_msg_count: int) -> tuple[bool, int]:
        prev = self._state["sessions"].get(session_id)
        if prev is None:
            return True, 0
        if current_msg_count > prev["analyzed_up_to_message_index"]:
            return True, prev["analyzed_up_to_message_index"]
        return False, prev["analyzed_up_to_message_index"]

    def mark_analyzed(
        self,
        session_id: str,
        message_index: int,
        chunk_count: int,
        report_path: str,
        outcome_summary: dict,
    ):
        self._state["sessions"][session_id] = {
            "analyzed_at": datetime.now().isoformat(),
            "analyzed_up_to_message_index": message_index,
            "chunk_count": chunk_count,
            "report_path": report_path,
            "outcome_summary": outcome_summary,
        }
        self.save()

    def reset_session(self, session_id: str):
        self._state["sessions"].pop(session_id, None)
        self.save()

    def reset_all(self):
        self._state["sessions"] = {}
        self.save()

    @property
    def processed_count(self) -> int:
        return len(self._state["sessions"])

    @property
    def sessions(self) -> dict:
        return self._state["sessions"]
