"""向量嵌入服务"""
import logging
import traceback
import dashscope
from config import settings as config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str = None):
        self.model = model or config.embedding_model
        self._api_key = config.dashscope_api_key

    def embed_query(self, text):
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model, input=text, text_type="query",
                api_key=self._api_key,
            )
            return resp.output["embeddings"][0]["embedding"]
        except Exception:
            logger.error("嵌入失败 (query): %s", text[:50])
            traceback.print_exc()
            return [0.0] * 1024

    def embed_documents(self, texts):
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model, input=texts, text_type="document",
                api_key=self._api_key,
            )
            return [emb["embedding"] for emb in resp.output["embeddings"]]
        except Exception:
            logger.error("嵌入失败 (docs): %d 条", len(texts))
            traceback.print_exc()
            return [[0.0] * 1024 for _ in texts]
