"""知识库管理服务"""
import os, hashlib, logging
from datetime import datetime
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval.embedding import EmbeddingService
from retrieval.parent_store import ParentStore
from config import settings as config

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self):
        self.embedding = EmbeddingService()
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )
        self._parent_store: ParentStore = None
        self._hierarchical_retriever = None

    @property
    def parent_store(self) -> ParentStore:
        if self._parent_store is None:
            self._parent_store = ParentStore()
        return self._parent_store

    @property
    def hierarchical_retriever(self):
        """层级检索器: 子块匹配 + 父块上下文 + 来源引用"""
        if self._hierarchical_retriever is None:
            from retrieval.hierarchical_retriever import HierarchicalRetriever
            self._hierarchical_retriever = HierarchicalRetriever(
                self.chroma, self.parent_store,
            )
        return self._hierarchical_retriever

    def upload(self, text: str, filename: str) -> str:
        if not text.strip():
            return "[失败] 内容为空"
        md5 = hashlib.md5(text.encode()).hexdigest()
        if self._check_md5(md5):
            return "[跳过] 内容已存在"
        chunks = self.splitter.split_text(text) if len(text) > config.max_split_char_number else [text]
        metadatas = [{"source": filename, "chunk_id": i, "time": datetime.now().isoformat()}
                     for i in range(len(chunks))]
        self.chroma.add_texts(texts=chunks, metadatas=metadatas)
        self._save_md5(md5)
        return f"[成功] 已载入 {len(chunks)} 条片段"

    def ingest_file(self, file_path: str, display_name: str = None) -> str:
        """入库文件 — 使用完整流水线（清洗 + 父子分块 + 语义去重）

        Args:
            file_path: 文件路径
            display_name: 前端展示用文件名（留空则取 file_path 的 basename）

        Returns:
            入库结果描述字符串
        """
        from retrieval.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            self.chroma, self.parent_store, self.embedding,
            parser_preference=config.document_parser,
            enable_cleaning=config.ingestion_cleaning_enabled,
            enable_dedup=config.ingestion_dedup_enabled,
        )
        stats = pipeline.ingest(file_path, display_name=display_name)

        if stats.parents_inserted == 0 and stats.parents_duplicate > 0:
            return f"[跳过] 内容重复（{stats.parents_duplicate} 个父块已存在）"
        if stats.parents_inserted == 0:
            return f"[失败] 未提取到有效内容"
        return (
            f"[成功] 父块 {stats.parents_inserted}/{stats.parents_total}"
            f"（去重跳过 {stats.parents_duplicate}），子块 {stats.children_inserted}"
            f"，清洗: {stats.chars_raw} → {stats.chars_cleaned} 字符"
        )

    def _check_md5(self, md5: str) -> bool:
        os.makedirs(os.path.dirname(config.md5_path), exist_ok=True)
        if not os.path.exists(config.md5_path):
            return False
        with open(config.md5_path, "r") as f:
            return md5 in f.read()

    def _save_md5(self, md5: str):
        with open(config.md5_path, "a") as f:
            f.write(md5 + "\n")
