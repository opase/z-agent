"""
父块持久化存储

存储 Level 1+ 父块，支持按 ID 批量查询。
子块在 ChromaDB 中检索匹配后，通过 parent_id 在此查询父块上下文。
"""
import logging
import os
import sqlite3
import threading
from config import settings as config

logger = logging.getLogger(__name__)


class ParentStore:
    """父块 SQLite 存储

    表结构:
        parent_chunks(
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            page INTEGER,
            page_end INTEGER,
            section TEXT DEFAULT '',
            heading_level INTEGER DEFAULT 0,
            source TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or os.path.join(
            config.persist_directory, "parent_chunks.db"
        )
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    page INTEGER,
                    page_end INTEGER,
                    section TEXT DEFAULT '',
                    heading_level INTEGER DEFAULT 0,
                    source TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_source
                ON parent_chunks(source)
            """)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── 写入 ──

    def insert(self, chunk_id: str, text: str, page: int = None,
               page_end: int = None, section: str = "", heading_level: int = 0,
               source: str = "") -> bool:
        """插入或替换父块"""
        with self._lock, self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO parent_chunks
                (id, text, page, page_end, section, heading_level, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk_id, text, page, page_end, section, heading_level, source))
            conn.commit()
        return True

    def insert_batch(self, chunks: list[dict]) -> int:
        """批量插入父块"""
        if not chunks:
            return 0
        with self._lock, self._get_conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO parent_chunks
                (id, text, page, page_end, section, heading_level, source)
                VALUES (:id, :text, :page, :page_end, :section, :heading_level, :source)
            """, chunks)
            conn.commit()
        return len(chunks)

    # ── 查询 ──

    def get(self, chunk_id: str) -> dict | None:
        """按 ID 查询单个父块"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM parent_chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_batch(self, chunk_ids: list[str]) -> list[dict]:
        """批量查询父块（去重 + 保持顺序）"""
        if not chunk_ids:
            return []
        seen = set()
        unique_ids = [x for x in chunk_ids if not (x in seen or seen.add(x))]

        placeholders = ",".join("?" * len(unique_ids))
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM parent_chunks WHERE id IN ({placeholders})",
                unique_ids,
            ).fetchall()

        # 按传入顺序排列
        row_map = {r["id"]: dict(r) for r in rows}
        return [row_map[cid] for cid in unique_ids if cid in row_map]

    def exists(self, chunk_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM parent_chunks WHERE id = ? LIMIT 1", (chunk_id,)
            ).fetchone()
        return row is not None

    # ── 删除 ──

    def delete_by_source(self, source: str) -> int:
        """删除某文档的所有父块"""
        with self._lock, self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM parent_chunks WHERE source = ?", (source,)
            )
            conn.commit()
            return cur.rowcount

    def clear(self):
        """清空所有父块"""
        with self._lock, self._get_conn() as conn:
            conn.execute("DELETE FROM parent_chunks")
            conn.commit()
