"""LangGraph Agent 状态机编排 — ReAct 模式（含 HITL 审批 + MCP 集成）"""
import logging
import re
import time
import uuid
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from config import settings as config

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = config.max_react_iterations

# 来源标记正则: [文件名 | 第N页 | 章节路径]
_SOURCE_RE = re.compile(
    r'\[([^\]|]+?)\s*(?:\|\s*第(\d+)页)?\s*(?:\|\s*(.+?))?\]'
)


def _extract_sources(context: str) -> list[dict]:
    """从上下文字符串中提取结构化来源信息"""
    if not context:
        return []
    sources = []
    seen = set()
    for m in _SOURCE_RE.finditer(context):
        doc = m.group(1).strip()
        page = m.group(2)
        section = (m.group(3) or "").strip()
        key = f"{doc}|{page}|{section}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "document": doc,
            "page": int(page) if page else None,
            "section": section,
        })
    return sources

# ── 工具执行器注册表 ──
# 新增内置工具只需在此注册
# 支持两种匹配：精确匹配（"search_knowledge"）和前缀匹配（"mcp__"）。

from typing import Callable, Awaitable

ToolExecutor = Callable[[str, dict, object], Awaitable[str]]
_TOOL_EXECUTORS: dict[str, ToolExecutor] = {}


def register_tool_executor(name: str, executor: ToolExecutor) -> None:
    """注册工具执行器。

    Args:
        name: 工具名或前缀（如 "mcp__" 匹配所有 mcp__* 工具）
        executor: async (tool_name, args, rag_service) -> str
    """
    _TOOL_EXECUTORS[name] = executor


def _register_builtin_executors() -> None:
    """注册内置工具（惰性导入避免循环依赖）"""

    async def _exec_search_knowledge(name: str, args: dict, _rag) -> str:
        from agent.tools import search_knowledge
        return search_knowledge.invoke(args)

    async def _exec_load_skill(name: str, args: dict, _rag) -> str:
        from skill.tool import load_skill
        return load_skill.invoke(args)

    async def _exec_list_skills(name: str, args: dict, _rag) -> str:
        from skill.tool import list_skills
        return list_skills.invoke(args)

    async def _exec_mcp(name: str, args: dict, rag) -> str:
        parts = name.split("__", 2)
        if len(parts) == 3:
            return str(await rag.mcp_manager.call_tool(parts[1], parts[2], args))
        return f"[错误] 无效的 MCP 工具名: {name}"

    _TOOL_EXECUTORS.update({
        "search_knowledge": _exec_search_knowledge,
        "load_skill": _exec_load_skill,
        "list_skills": _exec_list_skills,
        "mcp__": _exec_mcp,
    })


_register_builtin_executors()


class AgentState(TypedDict):
    question: str
    rewritten_query: str
    context: str
    answer: str
    verification: dict
    retry_count: int
    chat_history: str
    user_profile: str
    final_output: dict
    # 多模态字段
    images: list[str]
    image_desc: str
    detected_products: list[str]
    # 会话标识（用于 HITL 审批恢复）
    thread_id: str
    user_id: str
    has_mcp_tools: bool
    # Phase 2: 执行模式
    mode: str          # "react" | "plan" | "multi_agent"
    plan_result: str   # Plan-and-Execute / Multi-Agent 的汇总结果


def route_mode(state: AgentState) -> str:
    """条件边: 根据 mode 路由到不同的执行节点

    优先级:
    1. 用户显式指定 mode=plan / multi_agent / react → 直接使用
    2. mode=auto → Planner.is_simple_goal() 判断，复杂任务自动升级 plan
    """
    mode = state.get("mode", "auto")
    question = state.get("question", "")
    q_lower = question.lower()

    # 显式指定优先
    if mode == "plan" or "mode=plan" in q_lower:
        logger.info("用户指定 Plan-and-Execute 模式")
        return "plan"
    if mode == "multi_agent" or "mode=multi" in q_lower:
        logger.info("用户指定 Multi-Agent 模式")
        return "multi_agent"
    if mode == "react" or "mode=react" in q_lower:
        return "react"

    # auto 模式: 简单目标走 react，复杂目标自动升级 plan
    if mode == "auto":
        from .planner import Planner
        if Planner().is_simple_goal(question):
            return "react"
        logger.info("复杂任务自动升级 Plan-and-Execute 模式")
        return "plan"

    return mode


def check_verify(state: AgentState) -> str:
    v = state.get("verification", {})
    if v.get("pass", True) or state.get("retry_count", 0) >= 1:
        return "output"
    return "retry"


def _build_input_text(state: AgentState) -> str:
    """构建 ReAct Agent 的输入上下文"""
    parts = []
    if state.get("user_profile"):
        parts.append(f"【用户画像】\n{state['user_profile']}")
    if state.get("chat_history"):
        parts.append(f"【对话历史】\n{state['chat_history']}")
    if state.get("image_desc"):
        products = "、".join(state.get("detected_products", [])) or "未知"
        parts.append(f"【图片识别】用户发送了图片，识别到产品: {products}\n描述: {state['image_desc']}")
    if state.get("retry_count", 0) > 0:
        # 重试时把上一轮已获取的工具结果带回来，避免从零重新探查、重复执行副作用工具
        prev_context = state.get("context", "")
        if prev_context and prev_context != "无相关资料":
            parts.append(
                "【上一次已获取的信息】以下是上一轮工具调用的结果，请直接复用，"
                "不要重复调用相同工具、不要重复执行写入等副作用操作：\n" + prev_context
            )
        if state.get("verification", {}).get("suggestion"):
            parts.append(f"【改进要求】{state['verification']['suggestion']}")
    parts.append(f"【用户问题】{state['question']}")
    return "\n\n".join(parts)


async def _execute_tool(tool_call: dict, rag_service,
                        thread_id: str = "default", user_id: str = "default",
                        skip_hitl: bool = False) -> str:
    """执行单个工具调用，MCP 工具带 HITL 审批检查

    asyncio.Event 原地等待审批决议，不重启节点函数。
    消息历史自然保留，无需缓存/恢复逻辑。
    """
    name = tool_call["name"]
    args = tool_call.get("args", {})
    from core import metrics
    metrics.tool_calls.labels(tool_name=name).inc()
    logger.info("ReAct 调用工具: %s(%s)", name, args)

    # HITL 审批检查（MCP 工具 + require_approval）
    if not skip_hitl and name.startswith("mcp__") and not rag_service.approval_mgr.is_approve_all(thread_id):
        mode = rag_service.mcp_registry.get_approval_mode(name)
        if mode != "auto_approve":
            parts = name.split("__", 2)
            server_name = parts[1] if len(parts) == 3 else "unknown"
            approval_id = str(uuid.uuid4())
            rag_service.approval_mgr.create_request(
                user_id, thread_id, name, args, server_name, approval_id,
            )

            # 检测 stream 上下文：adispatch_custom_event 成功 → stream 路径
            # 注意：非 stream 上下文中 ensure_config() 或 adispatch_custom_event 会抛异常，
            # 此时走 interrupt() 兜底。其他异常也应走 interrupt() 而非静默忽略。
            is_stream = False
            try:
                from langchain_core.runnables import ensure_config
                _cfg = ensure_config()
                from langchain_core.callbacks.manager import adispatch_custom_event
                await adispatch_custom_event("approval_required", {
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "tool": name, "args": args,
                    "server": server_name, "thread_id": thread_id,
                }, config=_cfg)
                is_stream = True
            except (RuntimeError, ValueError, LookupError):
                logger.debug("非 stream 上下文，审批走 interrupt 路径")
            except Exception as e:
                logger.warning("事件派发异常，降级为 interrupt 路径: %s", e)

            if is_stream:
                # asyncio.Event 原地等待（stream 路径，不重启节点）
                decision = await rag_service.approval_mgr.await_decision(approval_id)
            else:
                # 非 stream 路径兜底: LangGraph interrupt()（ainvoke 可检测中断）
                decision = interrupt({
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "tool": name, "args": args,
                    "server": server_name, "thread_id": thread_id,
                })

            if isinstance(decision, dict) and decision.get("decision") == "approved":
                pass  # 继续往下执行工具
            else:
                reason = (decision or {}).get("reject_reason", "") if isinstance(decision, dict) else ""
                msg = f"[审批拒绝] {reason}" if reason else "[审批拒绝] 操作被拒绝"
                logger.info("审批拒绝: %s", name)
                return msg

    try:
        # 注册表分发：精确匹配 → 前缀匹配 → 未知工具
        executor = _TOOL_EXECUTORS.get(name)
        if executor is None:
            # 前缀匹配（如 "mcp__filesystem__read_file" 匹配 "mcp__"）
            for prefix, exec_fn in sorted(_TOOL_EXECUTORS.items(), key=lambda kv: len(kv[0]), reverse=True):
                if name.startswith(prefix):
                    executor = exec_fn
                    break

        if executor is not None:
            result = await executor(name, args, rag_service)
        else:
            return f"[错误] 未知工具: {name}"
        return str(result)
    except Exception as e:
        logger.error("工具执行失败 %s: %s", name, e)
        return f"[错误] 工具 {name} 执行失败: {e}"


async def _emit_event(name: str, payload: dict) -> None:
    """派发思考过程自定义事件（stream 上下文有效，非 stream 静默跳过）。

    供前端展示 ReAct 的推理文本 / 工具调用 / 观察结果。
    """
    try:
        from langchain_core.runnables import ensure_config
        from langchain_core.callbacks.manager import adispatch_custom_event
        await adispatch_custom_event(name, {"type": name, **payload}, config=ensure_config())
    except RuntimeError:
        logger.debug("非 stream 上下文，跳过事件派发: %s", name)
    except Exception as e:
        logger.debug("事件派发失败: name=%s error=%s", name, type(e).__name__)



def build_graph(rag_service):
    """构建 LangGraph 状态图，注入 rag_service 供节点使用"""

    async def _classify_intent(state: AgentState) -> dict:
        # 预留 hook：意图分析已移除，此节点保留为占位 passthrough。
        # 后续如需重新引入意图识别 / 闲聊分流，可在此填充逻辑。
        return {}

    async def _vision_analyze(state: AgentState) -> dict:
        images = state.get("images", [])
        if not images:
            return {"image_desc": "", "detected_products": [], "images": []}
        result = await rag_service.vision_analyzer.aanalyze(
            images, state["question"],
        )
        desc = result["description"]
        products = result.get("detected_products", [])
        logger.info("视觉分析完成: desc=%s, products=%s", desc[:50], products)
        return {
            "image_desc": desc,
            "detected_products": products,
            "images": [],  # 清除 base64，不往后续节点传递
        }

    async def _rewrite_query(state: AgentState) -> dict:
        from memory.query_rewriter import QueryRewriter
        result = await QueryRewriter().arewrite(
            state["question"], state.get("chat_history", ""),
            llm=rag_service.llm,
        )
        return {"rewritten_query": result["rewritten"]}

    async def _react_generate(state: AgentState) -> dict:
        """ReAct 循环：LLM 自主决定何时检索、用什么工具、何时回答"""
        from agent.router import get_prompt_with_tools
        from agent.tools import search_knowledge, register_mcp_tools

        tools = [search_knowledge]
        has_mcp = rag_service.mcp_registry.available_count > 0
        # 注册 MCP 工具（如有已连接服务器）
        if has_mcp:
            mcp_tools = register_mcp_tools(rag_service.mcp_registry)
            tools.extend(mcp_tools)
        # 注册 Skill 工具
        from skill.tool import load_skill as load_skill_tool, list_skills as list_skills_tool
        tools.extend([load_skill_tool, list_skills_tool])
        llm_with_tools = rag_service.react_llm.bind_tools(tools)

        # Skill buffer: drain 已加载 skill → 拼到输入前面
        thread_id = state.get("thread_id", "unknown")
        from skill.tool import get_buffer
        skill_buf = get_buffer(thread_id)
        skill_prefix = skill_buf.drain()

        # 构建系统提示 —  prompt + skill 索引
        intent = "product_query"  # 意图分析已移除，统一使用通用产品 prompt
        has_mcp = rag_service.mcp_registry.available_count > 0
        prompt_template = get_prompt_with_tools(intent, has_mcp)
        system_text = prompt_template.messages[0].prompt.template

        # skill 索引注入 system prompt（替换 {skills} 占位符）
        from skill.index_formatter import format_skill_index
        skill_index = format_skill_index(rag_service.skill_registry.enabled_skills())
        system_text = system_text.replace("{skills}", skill_index)
        if skill_index:
            logger.info("Skill 索引已注入 system prompt: %d chars, skills=%s",
                        len(skill_index), [s.name for s in rag_service.skill_registry.enabled_skills()])

        system = SystemMessage(content=system_text)

        input_text = _build_input_text(state)
        if skill_prefix:
            input_text = f"{skill_prefix}\n{input_text}"
        messages = [system, HumanMessage(content=input_text)]

        context_chunks: list[str] = []

        # ReAct 循环
        from core import metrics
        from memory.token_budget import compact_react_messages
        for iteration in range(MAX_REACT_ITERATIONS):
            # 调用 LLM 前按需压缩上下文，避免工具返回累积撑爆窗口
            messages = await compact_react_messages(messages, rag_service.light_llm)
            t_llm = time.time()
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            metrics.llm_duration.labels(model=config.chat_model, node="react_generate").observe(time.time() - t_llm)

            if response.tool_calls:
                # LLM 决定调用工具 → HITL 检查在 _execute_tool 内完成
                messages.append(response)
                # 派发推理文本（思考过程）
                if response.content:
                    await _emit_event("thinking", {"text": response.content})
                for tc in response.tool_calls:
                    # 派发工具调用（观察前，便于前端在审批弹窗前先展示）
                    await _emit_event("tool_call", {"tool": tc["name"], "args": tc.get("args", {})})
                    result_text = await _execute_tool(
                        tc, rag_service,
                        thread_id=state.get("thread_id", "unknown"),
                        user_id=state.get("user_id", "default"),
                    )
                    # 派发工具观察结果
                    await _emit_event("tool_result", {"tool": tc["name"], "result_preview": result_text[:300]})
                    messages.append(ToolMessage(
                        content=result_text, tool_call_id=tc["id"],
                    ))
                    # load_skill 成功后 push 到 buffer
                    if tc["name"] == "load_skill" and "[已加载]" in result_text:
                        skill_name = tc.get("args", {}).get("name", "")
                        if skill_name:
                            skill_obj = rag_service.skill_registry.find(skill_name)
                            if skill_obj:
                                skill_buf.push(skill_name, skill_obj.body)
                    context_chunks.append(result_text)
                logger.info("ReAct 第 %d 轮: 调用了 %d 个工具", iteration + 1, len(response.tool_calls))
            else:
                # LLM 输出最终回答
                metrics.react_iterations.observe(iteration + 1)
                logger.info("ReAct 完成: 共 %d 轮, %d 次工具调用",
                            iteration + 1, len(context_chunks))
                return {
                    "answer": response.content or "",
                    "context": "\n\n".join(context_chunks) if context_chunks else "无相关资料",
                    "mode": "react",
                }

        # 超过最大迭代次数 → 强制生成回答
        logger.warning("ReAct 达到最大迭代次数 %d，强制生成回答", MAX_REACT_ITERATIONS)
        messages.append(HumanMessage(content="请基于已获取的所有信息，直接回答用户问题。"))
        response = await rag_service.react_llm.with_config(
            {"tags": ["stream_answer"]}
        ).ainvoke(messages)
        return {
            "answer": response.content or "",
            "context": "\n\n".join(context_chunks) if context_chunks else "无相关资料",
            "mode": "react",
        }

    # 缓存编译后的子图，避免每次请求重复编译
    _plan_graph = None
    _ma_graph = None

    async def _plan_execute(state: AgentState) -> dict:
        """Plan-and-Execute 模式 — 调用子图（惰性编译 + 缓存）"""
        nonlocal _plan_graph
        if _plan_graph is None:
            from .plan_executor import build_plan_execute_graph
            _plan_graph = build_plan_execute_graph(rag_service, checkpointer=checkpointer)

        from langchain_core.runnables import ensure_config

        sub_state = {
            "goal": state["question"],
            "plan_dict": {},
            "current_task_id": "",
            "final_summary": "",
            "plan_approved": False,
            "plan_review_enabled": False,
            "tasks": {},
            "thread_id": state.get("thread_id", "default"),
            "user_id": state.get("user_id", "default"),
        }
        logger.info("启动 Plan-and-Execute 子图: goal=%s...", state["question"][:60])
        cfg = ensure_config()
        result = await _plan_graph.ainvoke(sub_state, cfg)

        final = result.get("final_summary", "")
        plan_dict = result.get("plan_dict", {})
        return {
            "answer": final,
            "context": plan_dict.get("summary", ""),
            "plan_result": final,
            "mode": "plan",
        }

    async def _multi_agent(state: AgentState) -> dict:
        """Multi-Agent 模式 — 调用子图（惰性编译 + 缓存）"""
        nonlocal _ma_graph
        if _ma_graph is None:
            from .orchestrator import build_multi_agent_graph
            _ma_graph = build_multi_agent_graph(rag_service, checkpointer=checkpointer)

        from langchain_core.runnables import ensure_config

        sub_state = {
            "goal": state["question"],
            "steps": {},
            "current_step_id": "",
            "final_summary": "",
            "thread_id": state.get("thread_id", "default"),
            "user_id": state.get("user_id", "default"),
        }
        logger.info("启动 Multi-Agent 子图: goal=%s...", state["question"][:60])
        cfg = ensure_config()
        result = await _ma_graph.ainvoke(sub_state, cfg)

        final = result.get("final_summary", "")
        return {
            "answer": final,
            "context": "Multi-Agent 协作",
            "plan_result": final,
            "mode": "multi_agent",
        }

    async def _verify(state: AgentState) -> dict:
        from agent.verifier import AnswerVerifier
        from core import metrics
        t0 = time.time()
        result = await AnswerVerifier().averify(
            state["question"], state["answer"], state.get("context", ""),
            llm=rag_service.light_llm,
        )
        metrics.llm_duration.labels(model=config.classifier_model, node="verify").observe(time.time() - t0)
        metrics.verification_results.labels(result="pass" if result.get("pass", True) else "fail").inc()
        return {"verification": result}

    def _output(state: AgentState) -> dict:
        context = state.get("context", "")
        sources = _extract_sources(context)
        return {"final_output": {
            "answer": state["answer"],
            "context": context,
            "mode": state.get("mode", "react"),
            "rewritten_query": {"rewritten": state.get("rewritten_query", state["question"])},
            "verification": state.get("verification", {}),
            "retry_count": state.get("retry_count", 0),
            "image_desc": state.get("image_desc", ""),
            "detected_products": state.get("detected_products", []),
            "sources": sources,
        }}

    def _retry(state: AgentState) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", _classify_intent)
    graph.add_node("vision_analyze", _vision_analyze)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("react_generate", _react_generate)
    graph.add_node("plan_execute", _plan_execute)
    graph.add_node("multi_agent", _multi_agent)
    graph.add_node("verify", _verify)
    graph.add_node("output", _output)
    graph.add_node("retry", _retry)

    # classify_intent 现为占位 passthrough，所有请求统一进入执行流程
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "vision_analyze")
    graph.add_edge("vision_analyze", "rewrite_query")

    # Phase 2: rewrite_query 后根据 mode 路由到不同执行节点
    graph.add_conditional_edges("rewrite_query", route_mode, {
        "react": "react_generate",
        "plan": "plan_execute",
        "multi_agent": "multi_agent",
    })

    # 三种执行模式都连接到 verify
    graph.add_edge("react_generate", "verify")
    graph.add_edge("plan_execute", "verify")
    graph.add_edge("multi_agent", "verify")

    graph.add_conditional_edges("verify", check_verify, {"output": "output", "retry": "retry"})
    graph.add_edge("retry", "react_generate")  # 验证失败 → 重新执行 ReAct
    graph.add_edge("output", END)

    # 使用 MemorySaver 支持 interrupt() / Command(resume=...)
    # 主图和子图共享同一个 checkpointer，确保中断状态一致
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
