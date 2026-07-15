"""向量检索 + 混合检索（RRF 融合）"""
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from retrieval.embedding import EmbeddingService
from retrieval.bm25 import BM25Retriever
from config import settings as config

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, embedding: EmbeddingService = None):
        self.embedding = embedding or EmbeddingService()
        self.store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def search(self, query: str, top_k: int = None) -> list[dict]:
        k = top_k or config.vector_top_k
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        return [{"text": doc.page_content, "metadata": doc.metadata,
                 "vector_score": float(score), "vector_rank": i + 1}
                for i, (doc, score) in enumerate(results)]


class HybridRetriever:
    """混合检索：BM25 + 向量 + RRF 融合"""

    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = config.rrf_k

    @staticmethod
    def _normalize(results, rank_key):
        out = []
        for d in results:
            d = d.copy()
            if rank_key == "bm25_rank" and "rank" in d and "bm25_rank" not in d:
                d["bm25_rank"] = d.pop("rank")
            if rank_key not in d:
                d[rank_key] = d.get("rank", 999)
            out.append(d)
        return out

    @staticmethod
    def _rrf_fusion(bm25_results, vector_results, rrf_k):
        doc_scores = {}
        def process(results, score_key, rank_key):
            for doc in results:
                key = doc["text"][:100] + "|" + str(doc.get("metadata", {}).get("source", ""))
                rrf_score = 1.0 / (rrf_k + doc[rank_key])
                if key in doc_scores:
                    doc_scores[key]["total_score"] += rrf_score
                else:
                    entry = doc.copy()
                    entry["total_score"] = rrf_score
                    doc_scores[key] = entry
        bm25_n = HybridRetriever._normalize(bm25_results, "bm25_rank")
        vector_n = HybridRetriever._normalize(vector_results, "vector_rank")
        process(bm25_n, "bm25_score", "bm25_rank")
        process(vector_n, "vector_score", "vector_rank")
        return sorted(doc_scores.values(), key=lambda x: x["total_score"], reverse=True)

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or config.hybrid_top_k
        bm25_results = self.bm25.search(query, config.bm25_top_k)
        vector_results = self.vector.search(query, config.vector_top_k)
        fused = self._rrf_fusion(bm25_results, vector_results, self.rrf_k)
        return fused[:top_k]

    def search_as_documents(self, query: str, top_k: int = None) -> list[Document]:
        results = self.search(query, top_k)
        return [Document(page_content=r["text"], metadata={
            **r.get("metadata", {}),
            "rrf_score": r.get("total_score", 0),
            "bm25_score": r.get("bm25_score", 0),
            "vector_score": r.get("vector_score", 0),
        }) for r in results]
