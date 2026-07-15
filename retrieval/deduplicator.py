"""
语义去重器 — 父块级基于 embedding 余弦相似度去重
"""
from __future__ import annotations
import hashlib
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """父块语义去重

    子块用 MD5 精确去重（在 KnowledgeService 已有逻辑）。
    父块用 embedding 余弦相似度去重（纯 math 实现，零额外依赖），阈值默认 0.92。
    """

    def __init__(
        self,
        embedding_service,
        threshold: float = 0.92,
    ):
        """
        Args:
            embedding_service: EmbeddingService 实例（用于生成 embedding）
            threshold: 余弦相似度阈值，≥此值视为重复
        """
        self._embedding = embedding_service
        self._threshold = threshold

    def check_duplicate(
        self, text: str, parent_store, existing_parent_ids: list[str],
    ) -> tuple[bool, Optional[str]]:
        """检查父块文本是否与已有父块语义重复

        Args:
            text: 待检查的父块文本
            parent_store: ParentStore 实例
            existing_parent_ids: 同一 source 下已有的父块 ID 列表

        Returns:
            (is_duplicate, duplicate_of_id): 是否重复及重复对象的 ID
        """
        if not existing_parent_ids:
            return False, None

        try:
            new_emb = self._embedding.embed_query(text)
        except Exception:
            return False, None

        existing = parent_store.get_batch(existing_parent_ids)
        best_sim = 0.0
        best_id = None

        for parent in existing:
            try:
                parent_emb = self._embedding.embed_query(parent["text"])
                sim = self._cosine_similarity(new_emb, parent_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = parent["id"]
            except Exception:
                continue

        if best_sim >= self._threshold:
            logger.info("语义去重: 父块与 %s 相似度 %.3f ≥ %.2f，跳过", best_id, best_sim, self._threshold)
            return True, best_id

        return False, None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    @staticmethod
    def exact_hash(text: str) -> str:
        """MD5 精确去重哈希（用于子块）"""
        return hashlib.md5(text.encode()).hexdigest()
