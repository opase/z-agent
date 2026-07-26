"""语义成功率评估：LLM-as-judge 对 (问题, 回答, 上下面) 打分。

可选启用（EVAL_SEMANTIC_ENABLED，默认关——每轮多一次 LLM 调用）。
结果写入 z_agent.chat span 的属性 eval.semantic_pass / _score / _reason，
Phoenix 按 eval.semantic_pass 聚合即得语义成功率。
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "你是一个严格的评估员。判断【回答】是否正确且完整地回答了【问题】。"
    "若提供了【参考资料】，事实性以资料为准；资料为空则按常识判断相关性。"
    "只输出严格的 JSON，不要任何额外文字或代码块标记："
    '{"pass": true或false, "score": 0.0到1.0, "reason": "一句话中文理由"}'
)


async def judge_answer(llm, question: str, answer: str, context: str = "") -> dict:
    """对单轮问答打分。

    Returns:
        {"pass": bool, "score": float, "reason": str}；评估异常时降级为未通过。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user = (
        f"【问题】\n{question}\n\n"
        f"【回答】\n{answer}\n\n"
        f"【参考资料】\n{context or '（无）'}"
    )
    try:
        resp = await llm.ainvoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = _parse_json(text)
        return {
            "pass": bool(data.get("pass", False)),
            "score": float(data.get("score", 0.0) or 0.0),
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:
        logger.warning("语义评估失败（降级为未通过）: %s", e)
        return {"pass": False, "score": 0.0, "reason": f"评估异常: {e}"}


def _parse_json(text: str) -> dict:
    """从 LLM 输出提取首个 JSON 对象（容错：去 markdown 代码块 + 正则兜底）。"""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return {}
