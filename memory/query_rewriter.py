"""上下文感知查询改写"""
import logging
from langchain_community.chat_models import ChatTongyi
from config import settings as config

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatTongyi(model=config.chat_model, dashscope_api_key=config.dashscope_api_key)
        return self._llm

    def rewrite(self, query: str, chat_history: str = "") -> dict:
        if not self._needs_rewrite(query, chat_history):
            return {"original": query, "rewritten": query, "needs_rewrite": False}
        prompt = self._build_prompt(query, chat_history)
        try:
            rewritten = self._get_llm().invoke(prompt).content.strip()
            return {"original": query, "rewritten": rewritten, "needs_rewrite": rewritten != query}
        except Exception as e:
            logger.error("查询改写失败: %s", e)
            return {"original": query, "rewritten": query, "needs_rewrite": False}

    async def arewrite(self, query: str, chat_history: str = "", llm=None) -> dict:
        if not self._needs_rewrite(query, chat_history):
            return {"original": query, "rewritten": query, "needs_rewrite": False}
        prompt = self._build_prompt(query, chat_history)
        try:
            model = llm or self._get_llm()
            rewritten = (await model.ainvoke(prompt)).content.strip()
            return {"original": query, "rewritten": rewritten, "needs_rewrite": rewritten != query}
        except Exception as e:
            logger.error("查询改写失败: %s", e)
            return {"original": query, "rewritten": query, "needs_rewrite": False}

    @staticmethod
    def _needs_rewrite(query: str, chat_history: str) -> bool:
        if not chat_history or not chat_history.strip():
            return False
        has_ref = any(w in query for w in ["它", "这个", "那个", "这款", "那款", "呢"])
        return has_ref or len(query) <= 15

    @staticmethod
    def _build_prompt(query: str, chat_history: str) -> str:
        return f"根据对话历史将用户问题改写为独立完整的查询（指代消解、省略补全），只输出改写结果。\n\n对话历史：\n{chat_history}\n\n用户问题：{query}\n\n改写："
