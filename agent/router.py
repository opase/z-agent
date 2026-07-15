"""意图路由（选择 Prompt 模板和检索策略）"""
from langchain_core.prompts import ChatPromptTemplate


def _make_prompt(system_msg: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system_msg), ("user", "{input}")])


PROMPTS = {
    "product_query": _make_prompt("你是 Zagent 智能助手，根据参考资料准确、简洁地回答用户问题；若资料不足，请如实说明。\n\n参考资料：\n{context}"),
    "product_compare": _make_prompt("你是 Zagent 智能助手，当用户需要对比时，用表格清晰列出关键项并给出总结建议。\n\n参考资料：\n{context}"),
    "troubleshoot": _make_prompt("你是 Zagent 技术支持助手，按步骤引导用户排查问题，语气耐心友好。\n\n参考资料：\n{context}"),
    "purchase_advice": _make_prompt("你是 Zagent 助手，根据用户需求给出 2-3 个可行方案并说明理由。\n\n参考资料：\n{context}"),
    "chitchat": _make_prompt("你是 Zagent 助手，简洁友好地回复日常闲聊。"),
}

STRATEGIES = {
    "product_query": {"top_k": 6, "need_rerank": True},
    "product_compare": {"top_k": 10, "need_rerank": True},
    "troubleshoot": {"top_k": 8, "need_rerank": True},
    "purchase_advice": {"top_k": 10, "need_rerank": True},
    "chitchat": {"top_k": 0, "need_rerank": False},
}


def get_prompt(intent: str) -> ChatPromptTemplate:
    return PROMPTS.get(intent, PROMPTS["product_query"])


# 加载原始 prompt 文件
import os as _os

def _load_prompt_file(filename: str) -> str:
    path = _os.path.join(_os.path.dirname(__file__), "prompts", filename)
    if _os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""

_BASE_PROMPT = _load_prompt_file("base.md")
_AGENT_PROMPT = _load_prompt_file("agent.md")

# 通用工具型 prompt — base.md + agent.md + skill 索引 + 参考资料
_GENERAL_TEMPLATE = _BASE_PROMPT + "\n\n" + _AGENT_PROMPT + "\n\n{skills}\n参考资料：\n{context}"
GENERAL_TOOL_PROMPT = _make_prompt(_GENERAL_TEMPLATE)


def get_prompt_with_tools(intent: str, has_mcp_tools: bool = False) -> ChatPromptTemplate:
    """获取 prompt 模板——有外部工具时使用通用 prompt 避免角色限定"""
    if has_mcp_tools:
        return GENERAL_TOOL_PROMPT
    return PROMPTS.get(intent, PROMPTS["product_query"])


# ==================== Plan-and-Execute Prompts ====================

PLANNER_PROMPT = """你是一个任务规划专家。你的职责是将用户目标拆解为清晰的执行步骤。

请按以下 JSON 格式输出执行计划：

```json
{
  "summary": "一句话摘要",
  "tasks": [
    {
      "id": "task_1",
      "description": "列出当前目录下的所有文件",
      "type": "FILE_READ",
      "dependencies": []
    }
  ]
}
```

规则：
1. description 用自然语言描述要做什么，不需要指定具体工具名（Worker 会自动选择合适的工具）
2. 每个任务必须有唯一 id（task_1, task_2, ...）
3. dependencies 列出依赖的前置任务 id
4. 简单任务 1-3 步，复杂任务 5-10 步
5. 多个任务可独立完成时，不要添加依赖，以便并行执行
6. ANALYSIS 和 VERIFICATION 类型任务直接分析上下文，不需要工具
7. 不要为了凑步数引入无关操作

只输出 JSON，不要有其他内容。"""


PLAN_TASK_PROMPT = """你是 Plan-and-Execute 中的任务执行专家。请根据当前任务和上下文，选择合适的工具或生成回复。
当前任务类型：{task_type}
任务描述：{task_description}
如果任务是 FILE_READ / FILE_WRITE / COMMAND 类型，请调用可用工具完成操作。
如果是 ANALYSIS 或 VERIFICATION 类型，且上下文已足够，请直接输出分析结果，不需要调用工具。"""


# ==================== Multi-Agent Prompts ====================

TEAM_PLANNER_PROMPT = """你是 Multi-Agent 协作中的任务规划专家。你的职责是分析用户需求，将其拆解为清晰的执行步骤。

请按以下 JSON 格式输出执行计划：

```json
{
  "summary": "任务摘要",
  "steps": [
    {
      "id": "step_1",
      "description": "列出当前目录下的所有文件",
      "type": "FILE_READ",
      "dependencies": []
    }
  ]
}
```

规则：
1. description 中明确写出要用哪个工具（如"使用 write_file 创建 world.txt"）
2. 每个步骤必须有唯一 id，如 step_1、step_2
3. dependencies 列出依赖的步骤 id
4. 简单任务可以只拆成 1-3 步，复杂任务拆成 5-10 步
5. 多个步骤可以独立完成时，不要添加依赖，让编排器并行分配给多个 Worker
6. 不要为了凑步数引入无关操作

只输出 JSON，不要有其他内容。"""


TEAM_WORKER_PROMPT = """你是 Multi-Agent 协作中的任务执行专家。请根据任务步骤描述，调用工具完成具体操作。
如果任务是 FILE_READ / FILE_WRITE / COMMAND 类型，请调用可用工具完成操作。
如果是 ANALYSIS 或 VERIFICATION 类型，且上下文已足够，请直接输出分析结果。"""


TEAM_REVIEWER_PROMPT = """你是 Multi-Agent 协作中的质量检查专家。你的职责是检查执行结果是否正确、完整和高质量。

检查要点：
1. 任务是否按要求完成
2. 结果是否正确，有无明显错误
3. 是否遗漏重要步骤或细节
4. 输出格式是否规范

请以 JSON 格式输出检查结果：

```json
{
  "approved": true,
  "summary": "检查摘要",
  "issues": [],
  "suggestions": []
}
```

如果 approved 为 true，issues 为空即可。如果 approved 为 false，请详细说明问题并给出改进建议。

只输出 JSON，不要有其他内容。"""


def needs_retrieval(intent: str) -> bool:
    return intent != "chitchat"


def get_strategy(intent: str) -> dict:
    return STRATEGIES.get(intent, STRATEGIES["product_query"])
