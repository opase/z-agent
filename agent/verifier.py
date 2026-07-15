"""回答质量验证"""
import json, re, logging
from langchain_community.chat_models import ChatTongyi
from config import settings as config

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """回答质量验证器 — 使用轻量 LLM 评估回答的相关性、忠实性、完整性。

    只有 async 接口；调用方负责注入 llm（复用已有的 light_llm 实例）。
    """

    async def averify(self, question: str, answer: str, context: str, llm=None) -> dict:
        prompt = self._build_prompt(question, answer, context)
        try:
            model = llm
            if model is None:
                model = ChatTongyi(
                    model=config.classifier_model,
                    dashscope_api_key=config.dashscope_api_key,
                    temperature=0,
                )
            response = (await model.ainvoke(prompt)).content.strip()
            return self._parse_response(response)
        except Exception as e:
            logger.error("验证失败: %s", e)
            return {"pass": True, "score": 3, "reason": str(e)}

    @staticmethod
    def _build_prompt(question: str, answer: str, context: str) -> str:
        return (
            "验证回答质量。标准：相关性、忠实性、完整性。"
            'JSON输出：{"pass":true,"score":4,"reason":"原因","suggestion":"建议"}'
            f"\n\n问题：{question}\n参考资料：{context[:800]}\n回答：{answer}"
        )

    @staticmethod
    def _parse_response(response: str) -> dict:
        try:
            return json.loads(response)
        except Exception:
            m = re.search(r"\{[^}]+\}", response)
            return json.loads(m.group()) if m else {"pass": True, "score": 3}
