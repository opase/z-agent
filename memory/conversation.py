"""短期对话记忆 — Phase 3: Map-Reduce 异步压缩

三个独立 prompt + 事实过滤逻辑：
- MAP:   保留关键信息（需求意图、操作结果、决策结论、技术细节），≤200字
- REDUCE: 合并多段摘要为整体（含旧摘要），≤300字
- EXTRACT: 过滤临时任务性事实，只保留持久化事实
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from config import settings as config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 事实过滤器：区分临时任务和持久化事实
# ═══════════════════════════════════════════════════════════════════

# 临时性"下一步行动"前缀——以这些词开头的句子通常是任务描述，不持久化
EPHEMERAL_FACT_PREFIXES = [
    "用户想", "用户要", "用户需要", "用户请求", "帮我", "让我",
    "新建", "创建", "删除", "修改", "生成",
    "补充要求", "当前这一轮", "本次任务",
]

# 推测性词语——包含这些词的内容不持久化（可能不准确）
SPECULATION_CUES = ["可能", "应该", "猜测", "推测", "笔误", "提醒"]

# 持久化事实暗示词——包含这些词的内容更可能是长期有效的
DURABLE_FACT_HINTS = [
    "用户偏好", "用户习惯", "喜欢", "倾向",
    "项目", "仓库", "路径", "技术栈", "版本", "模型",
    "接口", "配置", "环境变量", "命令", "约定", "规则", "默认",
]


@dataclass
class Message:
    role: str
    content: str
    image_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class ConversationMemory:
    def __init__(self, window_size: int = None, llm=None,
                 light_llm=None,           # Phase 3: 压缩用轻量 LLM
                 on_message: callable = None, on_summary: callable = None,
                 session_id: str = "",      # Phase 3: 用于事实提取
                 user_id: str = "",
                 long_term_memory=None,     # Phase 3: 长期记忆引用
                 token_trigger: int = None):  # 短期记忆压缩的 token 触发阈值
        self.messages: list[Message] = []
        self.summary: str = ""
        self.window_size = window_size or config.memory_window_size
        # 兼容旧参数名：llm → _legacy_llm，优先用 light_llm
        self._light_llm = light_llm if light_llm is not None else llm
        self._on_message = on_message
        self._on_summary = on_summary
        self.session_id = session_id
        self.user_id = user_id
        self._long_term_memory = long_term_memory
        self._token_trigger = (token_trigger if token_trigger is not None
                               else config.short_term_memory_trigger_tokens)
        self._compression_task: asyncio.Task | None = None  # 待完成的异步压缩任务

    def add_message(self, role: str, content: str, image_count: int = 0):
        """添加消息，短期记忆 token 占用超阈值时异步调度 Map-Reduce 压缩"""
        self.messages.append(Message(role=role, content=content, image_count=image_count))
        if self._on_message:
            self._on_message(role, content, image_count)

        # 按估算 token 触发：短期记忆占用达到阈值即压缩
        from memory.token_budget import estimate_tokens
        token_count = sum(estimate_tokens(m.content) for m in self.messages)
        if token_count >= self._token_trigger:
            # 异步调度压缩（不阻塞当前请求）
            # 如果已有压缩任务在运行则跳过，避免多任务并发竞争
            if self._compression_task and not self._compression_task.done():
                return
            try:
                loop = asyncio.get_running_loop()
                self._compression_task = loop.create_task(self._compress())
            except RuntimeError:
                logger.warning("无事件循环，跳过异步压缩（测试模式）")

    async def await_compression(self):
        """等待压缩任务完成（会话结束/删除前调用，防止摘要丢失）"""
        if self._compression_task and not self._compression_task.done():
            try:
                await self._compression_task
            except Exception as e:
                logger.error("等待压缩任务失败: %s", e)

    def get_recent(self, n: int = None) -> list[Message]:
        return self.messages[-(n or self.window_size):]

    def get_context_string(self, n: int = None) -> str:
        recent = self.get_recent(n)
        parts = []
        if self.summary:
            parts.append(f"【历史摘要】\n{self.summary}")
        if recent:
            parts.append("【最近对话】")
            for m in recent:
                label = "用户" if m.role == "user" else "助手"
                extra = f"[附图{m.image_count}张] " if m.image_count else ""
                parts.append(f"{label}: {extra}{m.content}")
        return "\n".join(parts)

    def get_chat_history(self, n: int = None) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.get_recent(n)]

    def clear(self):
        self.messages.clear()
        self.summary = ""

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Map-Reduce 压缩
    # ═══════════════════════════════════════════════════════════════════

    def _get_light_llm(self):
        """懒初始化压缩 LLM（优先用注入的 light_llm，否则用 config.classifier_model）"""
        if self._light_llm is not None:
            return self._light_llm
        try:
            from langchain_community.chat_models import ChatTongyi
            self._light_llm = ChatTongyi(
                model=config.classifier_model,
                dashscope_api_key=config.dashscope_api_key,
                temperature=0,
            )
            return self._light_llm
        except Exception as e:
            logger.error("懒初始化 light_llm 失败: %s", e)
            return None

    async def _compress(self):
        """Map-Reduce 异步压缩

        1. 保留最近 N 轮（retained_rounds × 2 条消息）
        2. 溢出消息分块 → Map 阶段并行摘要（每块 compression_chunk_size 条）
        3. 多块时 Reduce 阶段合并（旧摘要 + 新 map 摘要 → 单一摘要）
        4. 硬截断摘要
        5. 事实提取写入长期记忆（含持久化过滤）
        """
        retained_count = config.compression_retained_rounds * 2
        if len(self.messages) <= retained_count:
            return

        overflow = self.messages[:-retained_count]
        self.messages = self.messages[-retained_count:]
        if not overflow:
            return

        llm = self._get_light_llm()
        if llm is None:
            logger.warning("无可用 LLM，跳过 Map-Reduce 压缩")
            return

        # 分块
        chunks = [
            overflow[i:i + config.compression_chunk_size]
            for i in range(0, len(overflow), config.compression_chunk_size)
        ]

        try:
            # Map 阶段：并行摘要各块
            map_summaries = await asyncio.gather(
                *[self._map_summarize(llm, chunk) for chunk in chunks],
            )

            # Reduce 阶段：合并摘要（含旧摘要）
            if len(map_summaries) == 1:
                new_summary = map_summaries[0]
            else:
                new_summary = await self._reduce_summaries(llm, map_summaries)

            # 硬截断（安全网）
            self.summary = new_summary[:config.compression_max_summary_chars]

            # 持久化回调
            if self._on_summary:
                self._on_summary(self.summary)

            # 事实提取写入长期记忆（过滤临时事实）
            if self._long_term_memory and self.user_id:
                await self._extract_facts(llm, new_summary)

        except Exception as e:
            logger.error("Map-Reduce 压缩失败: %s", e)

    async def _map_summarize(self, llm, chunk: list[Message]) -> str:
        """Map 阶段：单块对话摘要

        保留用户需求和意图、已执行操作和结果、决策和结论、重要技术细节。
        """
        history = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in chunk
        )
        prompt = (
            "请将以下对话片段压缩成一段简洁摘要。保留以下关键信息：\n"
            "- 用户的需求和意图\n"
            "- 已执行的操作和结果\n"
            "- 做出的决策和结论\n"
            "- 重要的技术细节（文件名、路径、配置等）\n"
            "不要包含临时性的中间过程或无关闲聊。\n"
            f"控制在{config.compression_map_max_chars}字以内。\n\n"
            f"对话片段：\n{history}"
        )
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("Map 摘要失败: %s，截断兜底", e)
            return history[:config.compression_map_max_chars] + "..."

    async def _reduce_summaries(self, llm, summaries: list[str]) -> str:
        """Reduce 阶段：合并多段摘要 + 旧摘要为单一摘要

        合并为一段连贯的整体摘要，去除重复，保留完整性。
        """
        old_context = f"\n【旧历史摘要】\n{self.summary}" if self.summary else ""
        joined = "\n".join(f"- {s}" for s in summaries)
        prompt = (
            "请将以下多个对话片段摘要合并为一段连贯的整体摘要。\n"
            "要求：\n"
            "- 按时间顺序组织，保持逻辑连贯\n"
            "- 去除重复内容\n"
            "- 保留所有关键事实、决策和技术细节\n"
            "- 如果旧摘要与新摘要涉及同一话题，以新摘要为准\n"
            f"控制在{config.compression_reduce_max_chars}字以内。\n"
            f"{old_context}\n\n"
            f"各片段摘要：\n{joined}"
        )
        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("Reduce 摘要失败: %s，拼接兜底", e)
            return "；".join(summaries)

    async def _extract_facts(self, llm, summary: str):
        """从压缩摘要中提取持久化事实 → LongTermMemory

        区分临时任务性事实和持久化事实，只保留跨会话仍有价值的内容。
        额外在 Python 层做 _is_persistent_fact() 过滤。
        """
        prompt = (
            "从以下对话摘要中提取事实。请注意区分：\n"
            "❌ 临时任务（不提取）：用户本次想做什么操作、让你创建/删除/修改什么文件、"
            "当前这一轮的具体请求……这些只在当次会话有效\n"
            "✅ 持久化事实（提取）：用户偏好和习惯、项目路径和技术栈、"
            "常用的命令和配置、长期有效的约定和规则\n\n"
            "以 JSON 格式输出：\n"
            '{"preferences": ["用户偏好或习惯"], '
            '"profile": {"key": "value"}, '
            '"mentioned_products": ["提及的产品或工具"]}\n'
            "没有对应字段的用空数组/对象。只提取确定且长期有效的信息。\n\n"
            f"摘要：\n{summary}"
        )
        try:
            response = (await llm.ainvoke(prompt)).content.strip()
            if response:
                # 二次过滤：Python 层再次过滤临时事实
                filtered = self._filter_facts(response)
                if filtered:
                    self._long_term_memory.extract_facts(filtered)
        except Exception as e:
            logger.error("事实提取失败: %s", e)

    # ═══════════════════════════════════════════════════════════════════
    # 事实过滤器
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_persistent_fact(text: str) -> bool:
        """判断一条事实是否值得持久化

        规则：
        - 以临时前缀开头 → 不持久化
        - 包含推测性词语 → 不持久化
        - 默认：持久化
        """
        if not text or not isinstance(text, str):
            return False
        text = text.strip()

        # 以临时任务前缀开头 → 不是持久化事实
        for prefix in EPHEMERAL_FACT_PREFIXES:
            if text.startswith(prefix):
                return False

        # 包含推测性词语 → 不是确定的持久化事实
        for cue in SPECULATION_CUES:
            if cue in text:
                return False

        return True

    @classmethod
    def _filter_facts(cls, llm_response: str) -> str:
        """过滤 LLM 提取的事实：移除临时任务性内容

        1. 先尝试 JSON 解析，对 preferences 字段做逐条过滤
        2. 对 profile 的 key/value 做过滤
        3. JSON 解析失败则返回原文本（容错）
        4. 若过滤后所有字段为空，返回空字符串
        """
        import json as _json
        import re as _re

        try:
            facts = _json.loads(llm_response)
        except _json.JSONDecodeError:
            match = _re.search(r"\{.*\}", llm_response, _re.DOTALL)
            if match:
                try:
                    facts = _json.loads(match.group())
                except _json.JSONDecodeError:
                    logger.warning("事实过滤 JSON 解析失败，保留原文本")
                    return llm_response
            else:
                return llm_response

        if not isinstance(facts, dict):
            return llm_response

        # 过滤 preferences
        raw_prefs = facts.get("preferences", [])
        if isinstance(raw_prefs, list):
            filtered_prefs = [
                p for p in raw_prefs
                if cls._is_persistent_fact(p)
            ]
            facts["preferences"] = filtered_prefs

        # 过滤 profile（值和键）
        raw_profile = facts.get("profile", {})
        if isinstance(raw_profile, dict):
            filtered_profile = {}
            for k, v in raw_profile.items():
                # profile key 也检查是否临时
                if not cls._is_persistent_fact(str(k)):
                    continue
                # profile value 检查是否临时
                if isinstance(v, str) and not cls._is_persistent_fact(v):
                    continue
                filtered_profile[k] = v
            facts["profile"] = filtered_profile

        # 检查是否所有字段都为空
        has_content = (
            (facts.get("preferences") and len(facts["preferences"]) > 0) or
            (facts.get("profile") and len(facts["profile"]) > 0) or
            (facts.get("mentioned_products") and len(facts["mentioned_products"]) > 0)
        )
        if not has_content:
            return ""

        return _json.dumps(facts, ensure_ascii=False)
