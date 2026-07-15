"""
Multi-Agent 角色定义 + SubAgent + AgentMessage

核心设计:
- AgentRole: PLANNER / WORKER / REVIEWER 三种角色
- AgentMessage: Agent 间通信消息（TASK / RESULT / ERROR）
- SubAgent: 可配置角色的轻量 Agent，独立对话历史
  - 只有 WORKER 角色绑定工具
  - clear_history() 保留 system prompt 清空其余
"""
import logging
from enum import Enum
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent 角色

    - PLANNER: 分析任务 → 拆解为执行步骤 JSON（无工具）
    - WORKER: 执行具体步骤 → 调用工具完成操作（有工具）
    - REVIEWER: 检查执行结果质量 → 输出审批 JSON（无工具）
    """
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"


@dataclass
class AgentMessage:
    """Agent 间通信消息

    消息类型:
    - task:      编排器分配给子代理的任务
    - result:    子代理返回的执行结果
    - feedback:  检查者的反馈
    - approval:  检查者认可
    - rejection: 检查者拒绝
    - error:     子代理系统级错误
    """
    from_agent: str
    role: AgentRole | None
    content: str
    type: str

    @classmethod
    def task(cls, from_agent: str, content: str) -> "AgentMessage":
        """创建任务消息"""
        return cls(from_agent, None, content, "task")

    @classmethod
    def result(cls, from_agent: str, role: AgentRole, content: str) -> "AgentMessage":
        """创建结果消息"""
        return cls(from_agent, role, content, "result")

    @classmethod
    def error(cls, from_agent: str, role: AgentRole, content: str) -> "AgentMessage":
        """创建错误消息"""
        return cls(from_agent, role, content, "error")


class SubAgent:
    """可配置角色的轻量 Agent

    关键设计:
    - 独立对话历史: history (已发送消息列表)
    - 只有 WORKER 有工具: _should_use_tools() → role == WORKER
    - clear_history(): 保留 system prompt (history[0])，清空其余
    - execute(): 纯文本推理（PLANNER / REVIEWER）
    - execute_with_tools(): 带 ReAct 工具循环（WORKER）
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm: BaseChatModel,
        tools: list = None,
        system_prompt: str = "",
    ):
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = tools or []
        self._system_prompt = system_prompt
        # 对话历史: [0] = SystemMessage (保留), 后续 = 用户/助手消息
        self.history: list = []
        if system_prompt:
            self.history.append(SystemMessage(content=system_prompt))

    def _should_use_tools(self) -> bool:
        """只有 WORKER 角色使用工具"""
        return self.role == AgentRole.WORKER and len(self.tools) > 0

    async def execute(self, task: AgentMessage) -> AgentMessage:
        """执行任务 → 返回结果消息

        PLANNER / REVIEWER 走此路径: 纯文本推理，无工具。
        """
        logger.info("[%s] 执行任务: type=%s content_chars=%d",
                     self.name, task.type, len(task.content))

        self.history.append(HumanMessage(content=task.content))

        try:
            response = await self.llm.ainvoke(self.history)
            content = response.content if hasattr(response, "content") else str(response)
            self.history.append(AIMessage(content=content))

            logger.info("[%s] 任务完成: result_chars=%d", self.name, len(content))
            return AgentMessage.result(self.name, self.role, content)

        except Exception as e:
            logger.error("[%s] LLM 调用失败: %s", self.name, e)
            return AgentMessage.error(self.name, self.role, f"LLM 调用失败: {e}")

    async def execute_with_tools(
        self, task: AgentMessage, rag_service,
        thread_id: str = "default",
        user_id: str = "default",
    ) -> AgentMessage:
        """Worker 专用: 带工具的 ReAct 循环执行

        ReAct 循环: LLM 调用工具 → 获取结果 → 继续推理 → 输出最终结果
        """
        from .graph import _execute_tool

        self.history.append(HumanMessage(content=task.content))

        if not self._should_use_tools():
            # 无工具时降级为纯文本推理
            return await self.execute(task)

        llm_with_tools = self.llm.bind_tools(self.tools)

        from memory.token_budget import compact_react_messages
        for iteration in range(self.MAX_ITERATIONS):
            # 调用 LLM 前按需压缩上下文
            self.history = await compact_react_messages(self.history, self.llm)
            response = await llm_with_tools.ainvoke(self.history)

            if response.tool_calls:
                self.history.append(response)
                for tc in response.tool_calls:
                    result = await _execute_tool(
                        tc, rag_service,
                        thread_id=thread_id,
                        user_id=user_id,
                    )
                    self.history.append(ToolMessage(
                        content=result, tool_call_id=tc["id"],
                    ))
                logger.info("[%s] 第 %d 轮: %d 个工具调用",
                             self.name, iteration + 1, len(response.tool_calls))
            else:
                content = response.content or ""
                self.history.append(AIMessage(content=content))
                logger.info("[%s] 任务完成: result_chars=%d",
                             self.name, len(content))
                return AgentMessage.result(self.name, self.role, content)

        logger.warning("[%s] 超过最大迭代次数 %d", self.name, self.MAX_ITERATIONS)
        return AgentMessage.error(
            self.name, self.role,
            f"超过最大迭代次数 ({self.MAX_ITERATIONS})",
        )

    def clear_history(self):
        """保留 system prompt，清空其余历史"""
        if self.history and isinstance(self.history[0], SystemMessage):
            self.history = [self.history[0]]
        else:
            self.history = []
            if self._system_prompt:
                self.history.append(SystemMessage(content=self._system_prompt))

    def get_name(self) -> str:
        return self.name

    def get_role(self) -> AgentRole:
        return self.role
