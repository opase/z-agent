"""会话持久化存储"""
import sqlite3
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from config import settings as config

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """会话存储抽象基类——后续可替换为 Redis / Postgres"""

    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str,
                     image_count: int = 0, summary: str = ""): ...

    @abstractmethod
    def get_messages(self, session_id: str) -> list[dict]: ...

    @abstractmethod
    def list_sessions(self) -> list[dict]: ...

    @abstractmethod
    def save_summary(self, session_id: str, summary: str): ...

    @abstractmethod
    def delete_session(self, session_id: str): ...

    @abstractmethod
    def session_exists(self, session_id: str) -> bool: ...

    @abstractmethod
    def touch_session(self, session_id: str, user_id: str): ...


class MemoryStore(SessionStore):
    """纯内存存储——重启丢失，单进程可用

    警告: 此实现不持久化任何数据，所有方法均为空操作。
    生产环境请使用 SQLiteStore。
    """

    def __init__(self):
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "MemoryStore 不会持久化任何会话数据，重启后全部丢失。"
            "生产环境请使用 SQLiteStore。"
        )

    def save_message(self, session_id, role, content, image_count=0, summary=""):
        pass  # SessionManager 的 dict 已在内存中维护

    def get_messages(self, session_id):
        return []

    def list_sessions(self):
        return []

    def save_summary(self, session_id, summary):
        pass

    def delete_session(self, session_id):
        pass

    def touch_session(self, session_id, user_id):
        pass

    def session_exists(self, session_id):
        return False


class SQLiteStore(SessionStore):
    """SQLite 持久化——零依赖，跨重启保留"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(config.DATA_DIR, "sessions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_count INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summary (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                    summary TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args TEXT NOT NULL,
                    server_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    resolved_at TEXT,
                    reject_reason TEXT,
                    operator_id TEXT
                )
            """)
            conn.commit()

    def touch_session(self, session_id: str, user_id: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, user_id, created_at, last_active)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
            """, (session_id, user_id, now, now))
            conn.commit()

    def save_message(self, session_id: str, role: str, content: str,
                     image_count: int = 0):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, image_count, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, image_count, now),
            )
            conn.commit()

    def save_summary(self, session_id: str, summary: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO session_summary (session_id, summary) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary",
                (session_id, summary),
            )
            conn.commit()

    def get_messages(self, session_id: str) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, image_count, timestamp "
                "FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self, session_id: str) -> str:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT summary FROM session_summary WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["summary"] if row else ""

    def list_sessions(self) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT s.session_id, s.user_id, s.created_at, "
                "COUNT(m.id) as turn_count "
                "FROM sessions s LEFT JOIN messages m ON s.session_id = m.session_id "
                "AND m.role = 'user' "
                "GROUP BY s.session_id ORDER BY s.last_active DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def session_exists(self, session_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row is not None

    def delete_session(self, session_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_summary WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
