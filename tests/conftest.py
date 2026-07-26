"""pytest 共享 fixtures"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试隔离：在任何测试模块（及其 lazy import 的 main）加载前，把 tracing 初始化降级为 noop，
# 避免单测向真实 Phoenix 导出 span/metric，并避开全局 TracerProvider/MeterProvider 单例。
# OTel 未安装时 import 失败也无妨——setup_tracing 内部本就会降级。
try:
    import core.tracing as _tracing
    _tracing.setup_tracing = lambda: None
    _tracing.shutdown_tracing = lambda: None
except Exception:
    pass


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
