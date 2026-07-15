"""多模态视觉分析器（Qwen3-VL）"""
import json
import re
import logging
import asyncio
import dashscope
from config import settings as config

logger = logging.getLogger(__name__)

VISION_PROMPT = """请分析用户发送的图片。

请识别并输出以下信息（严格JSON格式）：
{
  "description": "图片内容描述（50字以内）",
  "detected_products": ["识别到内容"],
  "detected_issues": ["发现信息"],
  "scene_type": "product_photo|error_screenshot|comparison|other"
}

规则：
- product_photo: 参数截图
- error_screenshot: 报错/故障截图
- comparison: 多张对比图
- other: 其他

只输出JSON，不要其他文字。"""


class VisionAnalyzer:
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.vision_model
        self._api_key = api_key or config.dashscope_api_key

    def _build_messages(self, images: list[str], question: str) -> list[dict]:
        content = []
        for img in images:
            content.append({"image": img})
        content.append({"text": f"{VISION_PROMPT}\n\n用户问题：{question}"})
        return [
            {"role": "system", "content": [{"text": "你是一名分析专家。"}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _parse_response(text: str) -> dict:
        default = {"description": "", "detected_products": [],
                    "detected_issues": [], "scene_type": "other"}
        try:
            result = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return default
            try:
                result = json.loads(m.group())
            except Exception:
                return default
        return {k: result.get(k, v) for k, v in default.items()}

    def analyze(self, images: list[str], question: str = "") -> dict:
        """同步分析图片（在 async 上下文中通过 to_thread 调用）"""
        messages = self._build_messages(images, question)
        try:
            resp = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self._api_key,
            )
            if resp.status_code != 200:
                logger.warning("视觉分析失败: status=%s, msg=%s", resp.status_code, resp.message)
                return self._empty()
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and "text" in item
                )
            else:
                text = str(content)
            return self._parse_response(text)
        except Exception as e:
            logger.error("视觉分析异常: %s", e)
            return self._empty()

    async def aanalyze(self, images: list[str], question: str = "") -> dict:
        """异步分析图片"""
        try:
            return await asyncio.to_thread(self.analyze, images, question)
        except Exception as e:
            logger.error("异步视觉分析异常: %s", e)
            return self._empty()

    @staticmethod
    def _empty() -> dict:
        return {"description": "", "detected_products": [],
                "detected_issues": [], "scene_type": "other"}
