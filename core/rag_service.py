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
from core import tracing

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
        thread_id = memory.session_id if memory and hasattr(memory, "session_id") else "default"
        # root span：记录整轮对话，task.status 由 span() 在成功/失败时自动设置
        with tracing.span(
            "z_agent.chat",
            **{
                "user.query": tracing.sanitize_query(question),
                "chat.mode": mode,
                "chat.has_image": bool(images),
                "chat.stream": False,
                "chat.thread_id": thread_id,
            },
        ) as cspan:
            tracing.set_io(cspan, kind=tracing.KIND_CHAIN, input_value=question)
            tracing.set_ablation(cspan)
            result = await self._run_chat(question, memory, user_id, images, mode)
            # output.value：正常回答取 answer，中断/异常取 status 摘要
            if isinstance(result, dict):
                answer = result.get("answer") or ""
                tracing.set_io(cspan, output_value=answer or result.get("status", ""))
                if config.eval_semantic_enabled and answer.strip():
                    await self._record_semantic_eval(
                        cspan, question, answer, result.get("context", ""),
                    )
            return result

    async def _run_chat(
        self, question: str, memory: ConversationMemory = None,
        user_id: str = "default", images: list[str] = None,
        mode: str = "auto",
    ) -> dict:
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
        graph_config = {"configurable": {"thread_id": thread_id}}
        result = await self.agent_graph.ainvoke(initial_state, graph_config)

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
        """流式对话接口 — 统一使用 astream_events() 支持三种执行模式"""
        from opentelemetry import trace as _otel_trace
        thread_id = session_id or "default"
        # root span（流式）：用裸 span 手动管控 task.status，避免生成器内吞异常被误标成功
        with tracing.get_tracer().start_as_current_span("z_agent.chat") as chat_span:
            chat_span.set_attribute("user.query", tracing.sanitize_query(question))
            chat_span.set_attribute("chat.mode", mode)
            chat_span.set_attribute("chat.has_image", bool(images))
            chat_span.set_attribute("chat.stream", True)
            chat_span.set_attribute("chat.thread_id", thread_id)
            tracing.set_io(chat_span, kind=tracing.KIND_CHAIN, input_value=question)
            tracing.set_ablation(chat_span)
            # TTFT 子 span：duration = 首字延迟，Phoenix 直接按 chat.ttft 聚合 p50/p95。
            # 仅作计时标记，不设为 current span，故内部 LLM span 仍挂在 z_agent.chat 下。
            ttft_span = tracing.get_tracer().start_span(
                "chat.ttft", context=_otel_trace.set_span_in_context(chat_span),
            )
            ttft_span.set_attribute(tracing.SPAN_KIND, tracing.KIND_CHAIN)
            ttft_span.set_attribute("chat.thread_id", thread_id)
            _ttft_done = False
            try:
                async for chunk in self._run_chat_stream(
                    question, memory, user_id, images, session_id, mode,
                ):
                    # 第一个答案 token 到达即结束 TTFT（进度事件 data:{...} / __CA_META__ 不是答案）
                    if not _ttft_done and isinstance(chunk, str) and chunk \
                            and not chunk.startswith("data:") and "__CA_META__" not in chunk:
                        tracing.end_span(ttft_span, ok=True)
                        _ttft_done = True
                    yield chunk
            finally:
                # 审批中断 / 异常 / 无 token：未吐字即结束，留 note 不计成功也不计失败
                if not _ttft_done:
                    tracing.end_span(ttft_span, ok=False, note="no_token")

    async def _run_chat_stream(
        self, question: str, memory: ConversationMemory = None,
        user_id: str = "default", images: list[str] = None,
        session_id: str = "", mode: str = "auto",
    ) -> AsyncGenerator[str, None]:
        """Phase 2c 起：从手写 ReAct 循环迁移为 LangGraph astream_events() 驱动。

        - react / plan / multi_agent 模式均通过同一图执行
        - 自定义事件（plan_created, task_started 等）转为 SSE 进度事件
        - interrupt() 暂停图执行 → 产出审批 SSE 事件 → 等待用户决议
        - 图完成后流式输出最终回答
        """
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
        graph_config = {"configurable": {"thread_id": thread_id}}

        # ── Phase 1: 通过 astream_events 运行图，收集进度事件和中断 ──
        graph_result = None
        mode_info = ""
        verification_info = {}

        try:
            async for event in self.agent_graph.astream_events(
                initial_state, graph_config, version="v2",
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
                        sse_evt = {"type": name, **data}
                        yield f"data: {json.dumps(sse_evt, ensure_ascii=False)}\n\n"
        except Exception as e:
            tracing.record_error(tracing.current_span(), "stream_error", e)
            import traceback
            tb = traceback.format_exc()
            logger.error("astream_events 异常: %s\n%s", e, tb)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return

        # ── Phase 2: 检查 LangGraph 中断（工具审批 / 计划审批 / 审查升级） ──
        state_snapshot = await self.agent_graph.aget_state(graph_config)

        if state_snapshot and state_snapshot.interrupts:
            interrupt_list = state_snapshot.interrupts
            item = interrupt_list[0] if interrupt_list else None
            if item is not None:
                payload = getattr(item, "value", item)
                if isinstance(payload, dict):
                    evt_type = payload.get("type", "")
                    if evt_type in ("approval_required", "plan_review", "review_escalation"):
                        sse_evt = {"type": evt_type, **payload}
                        yield f"data: {json.dumps(sse_evt, ensure_ascii=False)}\n\n"
                        logger.info("Stream 路径中断: %s", evt_type)
                        meta = json.dumps({
                            "status": "interrupted",
                            "interrupt_type": evt_type,
                            "session_id": thread_id,
                            "mode": mode,
                            "turn_count": memory.turn_count if memory else 1,
                        }, ensure_ascii=False)
                        yield f"\n__CA_META__{meta}__CA_META_END__"
                        # interrupt 是正常暂停（等待审批），不是错误：收尾 span，
                        # 否则 z_agent.chat 会 status=UNSET、无 output（Phoenix 显示空）。
                        tracing.mark_success(
                            tracing.current_span(),
                            output_value=f"[已暂停等待审批: {evt_type}]",
                        )
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
        tracing.mark_success(tracing.current_span(), output_value=full_answer)
        if config.eval_semantic_enabled and full_answer.strip():
            await self._record_semantic_eval(
                tracing.current_span(), question, full_answer, context,
            )

    async def _record_semantic_eval(self, span_, question: str, answer: str, context: str = "") -> None:
        """LLM-as-judge 语义评估，结果写入 span 的 eval.semantic_* 属性（供 Phoenix 聚合成功率）。"""
        try:
            from agent.semantic_eval import judge_answer
            verdict = await judge_answer(self.light_llm, question, answer, context)
            span_.set_attribute("eval.semantic_pass", verdict["pass"])
            span_.set_attribute("eval.semantic_score", verdict["score"])
            span_.set_attribute("eval.semantic_reason", verdict["reason"])
        except Exception as e:
            logger.warning("记录语义评估失败: %s", e)


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
