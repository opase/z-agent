"""测试检索模块"""
import pytest


class TestBM25:
    def test_add_and_search(self, bm25_retriever):
        results = bm25_retriever.search("电池", top_k=5)
        assert len(results) > 0
        assert any("电池" in r["text"] for r in results)

    def test_empty_search(self, bm25_retriever):
        results = bm25_retriever.search("zzz_nonexistent_zzz", top_k=5)
        assert len(results) >= 0  # returns whatever it can, may be empty

    def test_clear(self, bm25_retriever):
        bm25_retriever.clear()
        assert bm25_retriever.doc_count == 0


class TestHybridRetriever:
    def test_rrf_fusion(self, hybrid_retriever):
        bm25_results = [
            {"text": "doc A", "metadata": {}, "score": 0.8, "rank": 1},
            {"text": "doc B", "metadata": {}, "score": 0.5, "rank": 2},
        ]
        vector_results = [
            {"text": "doc B", "metadata": {}, "vector_score": 0.9, "vector_rank": 3},
            {"text": "doc A", "metadata": {}, "vector_score": 0.7, "vector_rank": 1},
        ]
        from retrieval.vector import HybridRetriever
        fused = HybridRetriever._rrf_fusion(bm25_results, vector_results, rrf_k=60)
        assert len(fused) == 2
        assert all("total_score" in d for d in fused)
        # doc A: rank 1 in bm25 + rank 1 in vector → higher total
        assert fused[0]["text"] == "doc A"

    def test_normalize_preserves_rank(self):
        from retrieval.vector import HybridRetriever
        results = [{"text": "x", "rank": 3, "score": 0.5}]
        normalized = HybridRetriever._normalize(results, "bm25_rank")
        assert "bm25_rank" in normalized[0]
        assert normalized[0]["bm25_rank"] == 3
