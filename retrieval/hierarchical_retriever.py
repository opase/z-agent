"""
层级检索器 — 子块匹配 + 父块上下文

检索流程:
    1. 查询 embedding 匹配子块 (ChromaDB)
    2. 通过 parent_id 查找父块 (ParentStore SQLite)
    3. 去重合并父块
    4. 返回父块正文 + 来源元数据
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from retrieval.parent_store import ParentStore

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """来源引用"""
    document: str       # 文档名
    page: int = None    # 页码
    section: str = ""   # 章节路径
    snippet: str = ""   # 匹配片段预览


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str                          # 父块完整正文（送 LLM）
    score: float = 0.0                 # 子块最高匹配分数
    page: int = None                   # 页码
    section: str = ""                  # 章节路径
    source: str = ""                   # 文档名
    child_snippet: str = ""            # 命中的子块片段

    def to_citation(self) -> SourceCitation:
        return SourceCitation(
            document=self.source,
            page=self.page,
            section=self.section,
            snippet=self.child_snippet[:200] if self.child_snippet else self.text[:200],
        )


class HierarchicalRetriever:
    """层级检索器

    子块在 ChromaDB 做 embedding 匹配，
    然后通过 parent_id 取回父块作为 LLM 上下文。
    """

    def __init__(self, chroma: Chroma, parent_store: ParentStore):
        self._chroma = chroma
        self._parent_store = parent_store

    def retrieve(
        self, query: str, top_k: int = 6, *, include_sources: bool = True,
    ) -> tuple[list[str], list[SourceCitation]]:
        """检索并返回父块上下文 + 来源引用

        Args:
            query: 查询文本
            top_k: 匹配子块数（>= 最终父块数）
            include_sources: 是否返回来源引用

        Returns:
            (contexts, sources): 父块正文列表 + 来源引用列表
        """
        # ── Step 1: 在 ChromaDB 中匹配子块 ──
        child_results = self._chroma.similarity_search_with_relevance_scores(
            query, k=top_k,
        )
        if not child_results:
            return [], []

        # ── Step 2: 收集 parent_id，去重 ──
        seen_parents: dict[str, float] = {}   # parent_id → max score
        child_snippets: dict[str, str] = {}    # parent_id → best child snippet

        for doc, score in child_results:
            pid = doc.metadata.get("parent_id", "")
            if not pid:
                # 回退：无 parent_id 的旧数据，直接当父块
                pid = doc.metadata.get("chunk_id", doc.page_content[:50])
                # 确保父块存在（可能来自旧数据）
                self._parent_store.insert(
                    chunk_id=pid, text=doc.page_content,
                    source=doc.metadata.get("source", ""),
                )

            if pid not in seen_parents or score > seen_parents[pid]:
                seen_parents[pid] = score
                child_snippets[pid] = doc.page_content[:300]

        # ── Step 3: 从 ParentStore 查询父块 ──
        parent_ids = list(seen_parents.keys())
        parents = self._parent_store.get_batch(parent_ids)

        # ── Step 4: 组装结果 ──
        contexts: list[str] = []
        sources: list[SourceCitation] = []

        for p in parents:
            pid = p["id"]
            score = seen_parents.get(pid, 0.0)
            contexts.append(p["text"])
            if include_sources:
                sources.append(SourceCitation(
                    document=p.get("source", ""),
                    page=p.get("page"),
                    section=p.get("section", ""),
                    snippet=child_snippets.get(pid, p["text"][:200]),
                ))

        logger.debug("层级检索: query='%s' → %d 子块匹配 → %d 父块",
                     query[:50], len(child_results), len(parents))
        return contexts, sources

    def retrieve_with_metadata(
        self, query: str, top_k: int = 6,
    ) -> list[RetrievalResult]:
        """检索并返回完整的 RetrievalResult 列表"""
        child_results = self._chroma.similarity_search_with_relevance_scores(
            query, k=top_k,
        )
        if not child_results:
            return []

        seen: dict[str, RetrievalResult] = {}
        for doc, score in child_results:
            pid = doc.metadata.get("parent_id", "")
            if not pid:
                pid = doc.page_content[:50]

            if pid in seen:
                if score > seen[pid].score:
                    seen[pid].score = score
                    seen[pid].child_snippet = doc.page_content[:300]
            else:
                seen[pid] = RetrievalResult(
                    text="",  # 待填充
                    score=score,
                    page=doc.metadata.get("page"),
                    section=doc.metadata.get("section", ""),
                    source=doc.metadata.get("source", ""),
                    child_snippet=doc.page_content[:300],
                )

        # 填充父块文本
        parent_ids = list(seen.keys())
        parents = self._parent_store.get_batch(parent_ids)
        for p in parents:
            if p["id"] in seen:
                seen[p["id"]].text = p["text"]

        # 过滤掉找不到父块的
        return [r for r in seen.values() if r.text]
