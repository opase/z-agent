"""
RAG 核心服务
组装各层组件，提供统一的业务接口
"""
import json
import logging
from typing import AsyncGenerator
from memory.conversation import ConversationMemory
from memory.long_term import LongTermMemory
from config import settings as config
from core.container import (
    RetrievalServices, LLMProvider, MCPServices, VisionServices, MemoryServices,
)
from core import metrics

logger = logging.getLogger(__name__)


class RagService:
    """统一业务入口 — 装配各子系统并暴露公共属性。

    组件按子系统分组构建（core.container），__init__ 只做高层装配，
    不再内联 15+ 个组件的构造细节。
    """

    def __init__(self):
        # ── 检索层 ──
        _retrieval = RetrievalServices()
        self.embedding = _retrieval.embedding
        self.bm25 = _retrieval.bm25
        self.vector = _retrieval.vector
        self.hybrid = _retrieval.hybrid
        self.reranker = _retrieval.reranker

        # ── LLM 层 ──
        _llm = LLMProvider()
        self.llm = _llm.llm
        self.light_llm = _llm.light_llm
        self.react_llm = _llm.react_llm

        # ── MCP 层 ──
        _mcp = MCPServices()
        self.mcp_registry = _mcp.registry
        self.mcp_manager = _mcp.manager
        self.approval_store = _mcp.approval_store
        self.approval_mgr = _mcp.approval_mgr

        # ── 视觉层 ──
        _vision = VisionServices()
        self.vision_analyzer = _vision.vision_analyzer

        # ── 记忆层 ──
        _memory = MemoryServices()
        self.profile_extractor = _memory.profile_extractor
        self._long_term_cache: dict[str, LongTermMemory] = {}

        # ── Skill 层 ──
        from skill import SkillRegistry, set_skill_registry
        self.skill_registry = SkillRegistry()
        self.skill_registry.reload()
        set_skill_registry(self.skill_registry)

        # ── Agent 图 ──
        from agent.graph import build_graph
        from agent.tools import set_rag_service
        self.agent_graph = build_graph(self)
        set_rag_service(self)

        logger.info("RagService 初始化完成")

    def sync_bm25(self):
        try:
            collection = self.vector.store._collection
            offset = 0
            batch_size = 500
            total = 0
            while True:
                data = collection.get(
                    include=["documents", "metadatas"],
                    limit=batch_size, offset=offset,
                )
                if not data["documents"]:
                    break
                if offset == 0:
                    self.bm25.clear()
                self.bm25.add_documents(data["documents"], data["metadatas"])
                total += len(data["documents"])
                offset += batch_size
            logger.info("BM25 同步完成: %d 条", total)
        except Exception as e:
            logger.error("BM25 同步失败: %s", e)

    def _get_long_term(self, user_id: str) -> LongTermMemory:
        if user_id not in self._long_term_cache:
            self._long_term_cache[user_id] = LongTermMemory(user_id)
        return self._long_term_cache[user_id]

    async def chat(
        self, question: str, memory: ConversationMemory = None,
        user_id: str = "default", images: list[str] = None,
        mode: str = "auto",
    ) -> dict:
        """主对话接口（异步），支持图片"""
        metrics.chat_requests.labels(mode=mode, stream="false", has_image="true" if images else "false").inc()
        long_mem = self._get_long_term(user_id)
        # Phase 3: 注入长期记忆供压缩时事实提取（仅首次）
        if memory and not memory._long_term_memory:
            memory._long_term_memory = long_mem
            memory.user_id = user_id
        has_mcp = self.mcp_registry.available_count > 0
        # 使用 session_id 作为 thread_id（如果有 memory）
        thread_id = memory.session_id if memory and hasattr(memory, 'session_id') else "default"
        initial_state = {
            "question": question,
            "chat_history": memory.get_context_string() if memory else "",
            "user_profile": long_mem.get_context_string(),
            "rewritten_query": "",
            "context": "", "answer": "", "verification": {},
            "retry_count": 0, "final_output": {},
            "images": images or [], "image_desc": "", "detected_products": [],
            "thread_id": thread_id, "user_id": user_id,
            "has_mcp_tools": has_mcp,
            "mode": mode,  # Phase 2: 从 API 传入，route_mode 自动判断（auto 时）
            "plan_result": "",
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.agent_graph.ainvoke(initial_state, config)

        # LangGraph 1.x: interrupt() 后 ainvoke 返回带 __interrupt__ 键的状态
        if isinstance(result, dict) and "__interrupt__" in result:
            interrupt_info = result["__interrupt__"]
            if isinstance(interrupt_info, list) and len(interrupt_info) > 0:
                item = interrupt_info[0]
                # LangGraph Interrupt 对象: 取 .value 属性
                interrupt_payload = getattr(item, "value", item)
                if isinstance(interrupt_payload, dict):
                    return {
                        "status": "interrupted",
                        "interrupt": interrupt_payload,
                        "session_id": thread_id,
                    }
                logger.warning("无法解析 interrupt payload: type=%s", type(interrupt_payload).__name__)
                return {
                    "status": "interrupted",
                    "interrupt": str(interrupt_payload),
                    "session_id": thread_id,
                }

        output = result.get("final_output", result)
        if memory:
            img_count = len(images) if images else 0
            memory.add_message("user", question, image_count=img_count)
            memory.add_message("assistant", output.get("answer", ""))
        return output

    async def chat_stream(
        self, question: str, memory: ConversationMemory = None,
        user_id: str = "default", images: list[str] = None,
        session_id: str = "", mode: str = "auto",
    ) -> AsyncGenerator[str, None]:
        """流式对话接口 — 统一使用 astream_events() 支持三种执行模式

        Phase 2c: 从手写 ReAct 循环迁移为 LangGraph astream_events() 驱动。
        - react / plan / multi_agent 模式均通过同一图执行
        - 自定义事件（plan_created, task_started 等）转为 SSE 进度事件
        - interrupt() 暂停图执行 → 产出审批 SSE 事件 → 等待用户决议
        - 图完成后流式输出最终回答
        """
        metrics.chat_requests.labels(mode=mode, stream="true", has_image="true" if images else "false").inc()
        long_mem = self._get_long_term(user_id)
        # Phase 3: 注入长期记忆供压缩时事实提取
        if memory and not memory._long_term_memory:
            memory._long_term_memory = long_mem
            memory.user_id = user_id
        image_list = images or []
        has_image = bool(image_list)
        has_mcp = self.mcp_registry.available_count > 0
        thread_id = session_id or "default"

        # 构建初始状态（与 chat() 一致）
        initial_state = {
            "question": question,
            "chat_history": memory.get_context_string() if memory else "",
            "user_profile": long_mem.get_context_string(),
            "rewritten_query": "",
            "context": "", "answer": "", "verification": {},
            "retry_count": 0, "final_output": {},
            "images": image_list, "image_desc": "", "detected_products": [],
            "thread_id": thread_id, "user_id": user_id,
            "has_mcp_tools": has_mcp,
            "mode": mode,       # 从 API 传入，route_mode 自动判断（auto 时）
            "plan_result": "",
        }
        config = {"configurable": {"thread_id": thread_id}}

        # ── Phase 1: 通过 astream_events 运行图，收集进度事件和中断 ──
        graph_result = None
        mode_info = ""
        verification_info = {}

        try:
            async for event in self.agent_graph.astream_events(
                initial_state, config, version="v2",
            ):
                kind = event.get("event", "")

                # 自定义事件 → 进度 SSE
                if kind == "on_custom_event":
                    name = event.get("name", "")
                    data = event.get("data", {})
                    if name in ("plan_created", "task_started", "task_completed",
                                "task_failed", "plan_completed", "review_escalation",
                                "approval_required",
                                "thinking", "tool_call", "tool_result"):
                        metrics.sse_events.labels(event_type=name).inc()
                        sse_evt = {"type": name, **data}
                        yield f"data: {json.dumps(sse_evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("astream_events 异常: %s\n%s", e, tb)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return

        # ── Phase 2: 检查 plan_review / review_escalation 中断 ──
        # 工具审批已改用 asyncio.Event 原地等待，不再走 LangGraph interrupt()
        state_snapshot = await self.agent_graph.aget_state(config)

        if state_snapshot and state_snapshot.next and state_snapshot.interrupts:
            interrupt_list = state_snapshot.interrupts
            item = interrupt_list[0] if interrupt_list else None
            if item is not None:
                payload = getattr(item, "value", item)
                if isinstance(payload, dict):
                    evt_type = payload.get("type", "")
                    # 只处理 plan_review / review_escalation，工具审批已走 Event
                    if evt_type in ("plan_review", "review_escalation"):
                        sse_evt = {"type": evt_type, **payload}
                        yield f"data: {json.dumps(sse_evt, ensure_ascii=False)}\n\n"
                        logger.info("Stream 路径中断: %s", evt_type)
                        meta = json.dumps({
                            "status": "interrupted",
                            "interrupt_type": evt_type,
                            "session_id": thread_id,
                            "turn_count": memory.turn_count if memory else 1,
                        }, ensure_ascii=False)
                        yield f"\n__CA_META__{meta}__CA_META_END__"
                        return

        # ── Phase 3: 图正常完成 → 提取结果 ──
        if state_snapshot.values:
            final_output = state_snapshot.values.get("final_output", {})
            if isinstance(final_output, dict):
                answer = final_output.get("answer", "")
                context = final_output.get("context", "")
                mode_info = final_output.get("mode", mode)
                verification_info = final_output.get("verification", {})
                image_desc = final_output.get("image_desc", "")
                detected_products = final_output.get("detected_products", [])
            else:
                answer = str(final_output) if final_output else ""
                context = ""
                image_desc = ""
                detected_products = []
        else:
            answer = ""
            context = ""
            image_desc = ""
            detected_products = []

        # ── Phase 4: 流式输出最终回答 ──
        # 使用 self.llm.astream() 将图生成的回答逐 token 推送给前端
        full_answer = ""
        if answer.strip():
            # 直接流式输出已有的回答文本（避免重复 LLM 调用）
            # 按字符分块模拟流式效果
            chunk_size = 4
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                full_answer += chunk
                yield chunk
        else:
            # 兜底: 图为产出回答时，用 LLM 直接生成
            logger.warning("图未产出回答，使用 LLM 兜底生成")
            prompt_text = f"请回答以下问题：\n{question}"
            async for chunk in self.llm.astream(prompt_text):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield token

        # ── Phase 5: 更新记忆 + 元数据 ──
        if memory:
            img_count = len(image_list)
            memory.add_message("user", question, image_count=img_count)
            memory.add_message("assistant", full_answer)

        meta = json.dumps({
            "mode": mode_info,
            "verification": verification_info,
            "session_id": thread_id,
            "turn_count": memory.turn_count if memory else 1,
            "image_desc": image_desc if image_desc else "",
            "detected_products": detected_products if detected_products else [],
            "sources": final_output.get("sources", []),
        }, ensure_ascii=False)
        yield f"\n__CA_META__{meta}__CA_META_END__"


    async def end_session(self, user_id: str, memory: ConversationMemory):
        if memory.is_empty:
            return
        # Phase 3: 等待异步压缩任务完成再做最终提取
        await memory.await_compression()
        long_mem = self._get_long_term(user_id)
        extracted = await self.profile_extractor.aextract(
            memory.get_context_string(), llm=self.light_llm,
        )
        if extracted.get("profile"):
            long_mem.update_profile(extracted["profile"])
        for p in extracted.get("preferences", []):
            long_mem.add_preference(p)
        for p in extracted.get("mentioned_products", []):
            long_mem.add_mentioned_product(p)
        if extracted.get("summary"):
            long_mem.add_session_summary(extracted["summary"])
