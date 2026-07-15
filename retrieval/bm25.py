"""BM25 关键词检索"""
import logging
from rank_bm25 import BM25Okapi
import jieba

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self):
        self.corpus, self.metadata, self.tokenized = [], [], []
        self.bm25, self._dirty = None, True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        metadatas = metadatas or [{}] * len(texts)
        for text, meta in zip(texts, metadatas):
            self.corpus.append(text)
            self.metadata.append(meta)
            self.tokenized.append(self._tokenize(text))
        self._dirty = True

    def _rebuild(self):
        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)
            self._dirty = False

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._dirty:
            self._rebuild()
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"text": self.corpus[i], "metadata": self.metadata[i],
                 "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(indices)]

    def clear(self):
        self.corpus.clear(); self.metadata.clear(); self.tokenized.clear()
        self.bm25, self._dirty = None, True

    @property
    def doc_count(self) -> int:
        return len(self.corpus)
