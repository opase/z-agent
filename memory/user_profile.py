"""用户画像提取"""
import json, re, logging
from langchain_community.chat_models import ChatTongyi
from config import settings as config

logger = logging.getLogger(__name__)


class UserProfileExtractor:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatTongyi(model=config.classifier_model, dashscope_api_key=config.dashscope_api_key, temperature=0)
        return self._llm

    def extract(self, chat_history: str) -> dict:
        if not self._valid(chat_history):
            return self._empty()
        prompt = self._build_prompt(chat_history)
        try:
            response = self._get_llm().invoke(prompt).content.strip()
            return self._parse_response(response)
        except Exception as e:
            logger.error("画像提取失败: %s", e)
            return self._empty()

    async def aextract(self, chat_history: str, llm=None) -> dict:
        if not self._valid(chat_history):
            return self._empty()
        prompt = self._build_prompt(chat_history)
        try:
            model = llm or self._get_llm()
            response = (await model.ainvoke(prompt)).content.strip()
            return self._parse_response(response)
        except Exception as e:
            logger.error("画像提取失败: %s", e)
            return self._empty()

    @staticmethod
    def _valid(chat_history: str) -> bool:
        return bool(chat_history and len(chat_history.strip()) >= 50)

    @staticmethod
    def _build_prompt(chat_history: str) -> str:
        return (
            "分析以下对话，提取用户信息，JSON输出："
            '{"profile":{},"preferences":[],"mentioned_products":[],"summary":"一句话总结"}'
            f"\n\n{chat_history}"
        )

    def _empty(self) -> dict:
        return {"profile": {}, "preferences": [], "mentioned_products": [], "summary": ""}

    @staticmethod
    def _parse_response(response: str) -> dict:
        empty = {"profile": {}, "preferences": [], "mentioned_products": [], "summary": ""}
        try:
            result = json.loads(response)
        except Exception:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            result = json.loads(m.group()) if m else {}
        return {k: result.get(k, v) for k, v in empty.items()}
