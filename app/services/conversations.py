"""Index of conversations (title, turns, dates).

The messages themselves live in the LangGraph checkpointer; this table only makes
conversations listable and deletable from the UI.
"""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

TITLE_LENGTH = 80


class ConversationStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._con.execute(
                """CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    turns INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            self._con.commit()

    def touch(self, thread_id: str, question: str) -> None:
        """Create the conversation on its first question, then bump turns and updated_at."""
        now = datetime.now(UTC).isoformat()
        title = " ".join(question.split())[:TITLE_LENGTH]
        with self._lock:
            self._con.execute(
                """INSERT INTO conversations(thread_id, title, turns, created_at, updated_at)
                   VALUES (?, ?, 1, ?, ?)
                   ON CONFLICT(thread_id) DO UPDATE
                   SET turns = turns + 1, updated_at = excluded.updated_at""",
                (thread_id, title, now, now),
            )
            self._con.commit()

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._con.execute(
                "SELECT thread_id, title, turns, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._con.execute(
                "SELECT thread_id, title, turns, created_at, updated_at FROM conversations "
                "WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._con.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
            self._con.commit()

    def ping(self) -> None:
        with self._lock:
            self._con.execute("SELECT 1").fetchone()

    def close(self) -> None:
        self._con.close()
