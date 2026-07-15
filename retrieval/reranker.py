"""DashScope Rerank 重排序"""
import logging
import dashscope
from config import settings as config

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model: str = None):
        self.model = model or config.rerank_model
        self._api_key = config.dashscope_api_key

    def rerank(self, query: str, documents: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or config.rerank_top_k
        if not documents:
            return []
        docs = documents[:20]
        try:
            resp = dashscope.TextReRank.call(
                model=self.model, query=query,
                documents=[d["text"] for d in docs],
                top_n=min(top_k, len(docs)), return_documents=False,
                api_key=self._api_key,
            )
            if resp.status_code != 200:
                logger.warning("Rerank 失败: %s", resp.message)
                return documents[:top_k]
            reranked = []
            for i, r in enumerate(resp.output.results):
                doc = docs[r.index].copy()
                doc["rerank_score"] = r.relevance_score
                doc["rerank_rank"] = i + 1
                reranked.append(doc)
            return reranked
        except Exception as e:
            logger.error("Rerank 异常: %s", e)
            return documents[:top_k]
