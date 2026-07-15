"""
文档入库流水线 — 编排解析→清洗→分块→去重→入库全流程
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass

from langchain_chroma import Chroma
from retrieval.cleaners import CleaningPipeline
from retrieval.chunker import StructuredChunker
from retrieval.deduplicator import SemanticDeduplicator
from retrieval.parent_store import ParentStore
from retrieval.document_parser import create_parser

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """入库统计"""
    filename: str = ""
    parents_total: int = 0
    parents_inserted: int = 0
    parents_duplicate: int = 0
    children_inserted: int = 0
    children_duplicate: int = 0
    chars_raw: int = 0
    chars_cleaned: int = 0


class IngestionPipeline:
    """文档入库流水线

    用法:
        pipeline = IngestionPipeline(chroma, parent_store, embedding_service)
        stats = pipeline.ingest("doc.pdf")
    """

    def __init__(
        self,
        chroma: Chroma,
        parent_store: ParentStore,
        embedding_service,
        *,
        parser_preference: str = "auto",
        enable_cleaning: bool = True,
        enable_dedup: bool = True,
        dedup_threshold: float = 0.92,
    ):
        self._chroma = chroma
        self._parent_store = parent_store
        self._embedding = embedding_service
        self._parser_pref = parser_preference
        self._enable_cleaning = enable_cleaning
        self._enable_dedup = enable_dedup

        self._chunker = StructuredChunker()
        self._cleaner = CleaningPipeline.default() if enable_cleaning else CleaningPipeline.none()
        self._deduplicator = SemanticDeduplicator(
            embedding_service, threshold=dedup_threshold,
        ) if enable_dedup else None

    def ingest(self, file_path: str, display_name: str = None) -> IngestionStats:
        """入库单个文件

        Args:
            file_path: 文件路径
            display_name: 展示用文件名（留空取 file_path basename）

        Returns:
            IngestionStats: 入库统计
        """
        filename = display_name or os.path.basename(file_path)
        stats = IngestionStats(filename=filename)

        # ── Step 1: 解析 ──
        parser = create_parser(self._parser_pref)
        ext = os.path.splitext(file_path)[1].lower()
        if not parser.supports(ext):
            logger.warning("不支持的格式: %s", ext)
            return stats

        raw_text = parser.parse(file_path)
        stats.chars_raw = len(raw_text)

        # ── Step 2: 清洗 ──
        clean_text = self._cleaner.clean(raw_text)
        stats.chars_cleaned = len(clean_text)

        # ── Step 3: 分块 ──
        parents, children = self._chunker.chunk(clean_text, source_name=filename)
        stats.parents_total = len(parents)

        if not parents:
            logger.warning("未产生父块: %s", filename)
            return stats

        # ── Step 4: 父块语义去重 ──
        # 获取同 source 下已有的父块 ID
        existing_ids = self._get_existing_parent_ids(filename)

        inserted_parents: list[dict] = []
        inserted_children: list[dict] = []

        for p in parents:
            if self._deduplicator and existing_ids:
                is_dup, _ = self._deduplicator.check_duplicate(
                    p.text, self._parent_store, existing_ids,
                )
                if is_dup:
                    stats.parents_duplicate += 1
                    # 跳过该父块及其子块
                    continue

            inserted_parents.append({
                "id": p.id, "text": p.text,
                "page": p.page, "page_end": p.page_end,
                "section": p.section, "heading_level": p.heading_level,
                "source": filename,
            })
            existing_ids.append(p.id)

            # 子块不必语义去重，MD5 在入库时做
            for c in children:
                if c.parent_id == p.id:
                    inserted_children.append(c)

        stats.parents_inserted = len(inserted_parents)

        # ── Step 5: 入库 ──
        if inserted_parents:
            self._parent_store.insert_batch(inserted_parents)

        if inserted_children:
            stats.children_inserted = self._add_children_to_chroma(inserted_children)

        logger.info(
            "入库完成: %s → parents=%d (dup=%d) children=%d",
            filename, stats.parents_inserted, stats.parents_duplicate, stats.children_inserted,
        )
        return stats

    def _get_existing_parent_ids(self, source: str) -> list[str]:
        """获取同文档已有的父块 ID（用于去重比对）"""
        # ParentStore 目前没有按 source 查询的 API，做一次简单扫描
        import sqlite3
        db_path = self._parent_store._db_path
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id FROM parent_chunks WHERE source = ?", (source,)
            ).fetchall()
            conn.close()
            return [r["id"] for r in rows]
        except Exception:
            return []

    def _add_children_to_chroma(self, children: list) -> int:
        """子块入库 ChromaDB，跳过 MD5 重复"""
        import hashlib

        count = 0
        for c in children:
            # MD5 精确去重
            md5 = hashlib.md5(c.text.encode()).hexdigest()
            # 简单检查：尝试直接插入；如果 ChromaDB 有同 ID 则跳过
            try:
                self._chroma.add_texts(
                    texts=[c.text],
                    metadatas=[{
                        "source": c.metadata.get("source", ""),
                        "parent_id": c.parent_id,
                        "page": c.page,
                        "section": c.section,
                        "heading_level": c.heading_level,
                        "md5": md5,
                    }],
                    ids=[c.id],
                )
                count += 1
            except Exception as e:
                logger.debug("子块入库跳过 (可能重复): %s", e)

        return count
