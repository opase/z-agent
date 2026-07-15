"""pytest 共享 fixtures"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def config():
    from config import settings as config
    return config


@pytest.fixture(scope="session")
def embedding_service():
    from retrieval.embedding import EmbeddingService
    return EmbeddingService()


@pytest.fixture
def bm25_retriever():
    from retrieval.bm25 import BM25Retriever
    retriever = BM25Retriever()
    retriever.add_documents(
        ["小米15 Pro 电池容量 5000mAh", "华为Mate 70 Pro 摄像头 5000万像素"],
        [{"source": "test1.txt"}, {"source": "test2.txt"}],
    )
    return retriever


@pytest.fixture
def hybrid_retriever(embedding_service, bm25_retriever):
    from retrieval.vector import VectorRetriever, HybridRetriever
    vector = VectorRetriever(embedding_service)
    return HybridRetriever(vector, bm25_retriever)
