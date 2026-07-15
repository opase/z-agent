"""审批记录持久化存储 — 基于 SQLite，与 SessionStore 共享同一数据库"""

import json
import logging
import sqlite3
import os

from config import settings as config

logger = logging.getLogger(__name__)

CREATE_APPROVAL_TABLE = """
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
);
"""


class ApprovalStore:
    """审批请求的持久化 CRUD 操作

    复用 data/sessions.db，与 SQLiteStore 共享连接模式。
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or os.path.join(config.DATA_DIR, "sessions.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（与 SQLiteStore 一致的配置）"""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """创建表（若不存在）"""
        with self._get_conn() as conn:
            conn.execute(CREATE_APPROVAL_TABLE)
            conn.commit()
        logger.info("ApprovalStore 初始化完成: %s", self._db_path)

    def create(self, approval_id: str, user_id: str, thread_id: str,
               tool_name: str, tool_args: dict, server_name: str,
               expires_at: str) -> None:
        """创建审批请求记录"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO approval_requests
                   (id, user_id, thread_id, tool_name, tool_args, server_name,
                    status, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', datetime('now'), ?)""",
                (approval_id, user_id, thread_id, tool_name,
                 json.dumps(tool_args, ensure_ascii=False),
                 server_name, expires_at),
            )
            conn.commit()

    def get(self, approval_id: str) -> dict | None:
        """获取审批请求详情"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE id = ?",
                (approval_id,),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_pending_for_thread(self, thread_id: str) -> list[dict]:
        """获取指定线程的所有待处理审批"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM approval_requests
                   WHERE thread_id = ? AND status = 'pending'
                   ORDER BY created_at ASC""",
                (thread_id,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_pending_first(self, thread_id: str) -> dict | None:
        """获取指定线程的第一个待处理审批"""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT * FROM approval_requests
                   WHERE thread_id = ? AND status = 'pending'
                   ORDER BY created_at ASC LIMIT 1""",
                (thread_id,),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def resolve(self, approval_id: str, status: str,
                operator_id: str | None = None,
                reject_reason: str | None = None) -> bool:
        """更新审批请求状态（仅当当前状态为 pending 时生效 — 幂等性保证）

        Returns:
            True 表示更新成功，False 表示审批已非 pending 状态
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """UPDATE approval_requests
                   SET status = ?, resolved_at = datetime('now'),
                       operator_id = ?, reject_reason = ?
                   WHERE id = ? AND status = 'pending'""",
                (status, operator_id, reject_reason, approval_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def expire_pending(self) -> int:
        """将过期的 pending 审批自动标记为 expired

        Returns:
            过期处理的数量
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """UPDATE approval_requests
                   SET status = 'expired', resolved_at = datetime('now'),
                       reject_reason = '审批超时'
                   WHERE status = 'pending' AND expires_at < datetime('now')""",
            )
            conn.commit()
            return cursor.rowcount

    def mark_interrupted_for_thread(self, thread_id: str) -> int:
        """将指定线程的所有 pending 审批标记为 interrupted（重启时调用）"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """UPDATE approval_requests
                   SET status = 'interrupted', resolved_at = datetime('now'),
                       reject_reason = '服务重启中断'
                   WHERE thread_id = ? AND status = 'pending'""",
                (thread_id,),
            )
            conn.commit()
            return cursor.rowcount

    def get_history(self, user_id: str | None = None,
                    limit: int = 50, offset: int = 0) -> list[dict]:
        """查询审批历史"""
        with self._get_conn() as conn:
            if user_id:
                rows = conn.execute(
                    """SELECT * FROM approval_requests
                       WHERE user_id = ?
                       ORDER BY created_at DESC
                       LIMIT ? OFFSET ?""",
                    (user_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM approval_requests
                       ORDER BY created_at DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """将 sqlite3.Row 转为普通字典"""
        d = dict(row)
        # 将 tool_args JSON 字符串解析回 dict
        if d.get("tool_args") and isinstance(d["tool_args"], str):
            try:
                d["tool_args"] = json.loads(d["tool_args"])
            except json.JSONDecodeError:
                pass
        return d
